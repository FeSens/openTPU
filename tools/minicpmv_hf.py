"""Hugging Face's MiniCPM-V 4.7 (transformers >= 5.18, `minicpmv4_7`) on a host without the RAM for
its experts: the reference of docs/minicpmv.md 5. The whole model is HF's own code (processor,
vision tower, projector, canvas M-RoPE, Qwen3.5-MoE language model, generate) but every MoE
layer's experts are LazyExperts: HF's eager expert loop with each expert's weights read from the
safetensors files when a token routes to it, so the 35B-A3B runs in about 12 GB (fp32) beside
its 64 GB of bf16 experts.

    python3 tools/minicpmv_hf.py greedy MODEL OUT.json [--prompt P] [--image IMG] [-n 32]
        [--max-slices 9] [--dtype float32|bfloat16] [--think]
        HF's greedy tokens of a chat prompt (and an image), with each step's 8 best (id, logit):
        tools/offload/moe_card.py --check reads OUT.json (ids, tokens, top)
    python3 tools/minicpmv_hf.py logits MODEL OUT.npz [--prompt P] [--image IMG] [--ids J]
        [--hidden] [--max-slices 9] [--dtype ...]
        one teacher-forced forward of the prompt (or J's "ids" + "tokens": a greedy run's prompt
        and reply): the logits of every position (float32), with --hidden every layer's output
        rows; and for an image the processor's outputs and the canvas positions
    python3 tools/minicpmv_hf.py check
        a tiny random MiniCPM-V checkpoint: LazyExperts give the logits of HF's own loading
    python3 tools/minicpmv_hf.py procheck MODEL --images IMG... [--slices 9 1]
        the host's image port (opentpu.llm.minicpmv) against HF's processor and canvas
        positions, bit for bit (the model directory's tokenizer and configs; no weights)
    python3 tools/minicpmv_hf.py vision MODEL [OUT.npz] --images IMG... [--card int8 F...]
        [--prompt P] [--max-slices 9] [-n 32] [--rows PREV.npz]
        the vision side against HF's: vision_reference (erf, tanh) and the card's tower on the
        ISA simulator per format, row cosines, and the dKL of HF's greedy reply (through its
        first end token) with each variant's rows (OUT.json too); --rows takes the card's rows
        of a previous run's OUT.npz (its card-<format> arrays) instead of running the tower

MODEL: a checkpoint directory (models/MiniCPM-V-4.7-35B-A3B). transformers 5.18 may sit on its
own path (PYTHONPATH=~/mcpv/tf518 on the simulation host, with pillow and torchvision 0.29.1+cpu for
the image processor).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


class LazyExperts(nn.Module):
    """A Qwen3_5MoeExperts whose weights stay in the checkpoint: the forward is HF's eager loop
    (modeling_qwen3_5_moe.Qwen3_5MoeExperts.forward, v5.18.0) with expert e's gate_up_proj[e]
    and down_proj[e] read from the safetensors slices (bf16) and cast to `dtype`."""

    def __init__(self, handles: dict, prefix: str, config, dtype):
        super().__init__()
        self.num_experts = config.num_experts
        self.prefix, self.dtype = prefix, dtype
        self.h = {n: handles[prefix + n] for n in ("gate_up_proj", "down_proj")}
        from transformers.activations import ACT2FN
        self.act_fn = ACT2FN[config.hidden_act]

    def weight(self, name: str, e: int) -> torch.Tensor:
        f = self.h[name]
        return f.get_slice(self.prefix + name)[e].to(self.dtype)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts + 1)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            e = int(expert_idx)
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state,
                                            self.weight("gate_up_proj", e)).chunk(2, dim=-1)
            h = nn.functional.linear(self.act_fn(gate) * up, self.weight("down_proj", e))
            h = h * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, h.to(final_hidden_states.dtype))
        return final_hidden_states


def load(model_dir, dtype=torch.float32, attn: str = "sdpa"):
    """(model, processor): HF's MiniCPMV4_7ForConditionalGeneration built on the meta device,
    its experts LazyExperts, every other tensor read from the checkpoint and cast to dtype; the
    non-persistent RoPE buffers recomputed (to_empty leaves them uninitialized)."""
    from safetensors import safe_open
    from transformers import AutoConfig, AutoProcessor, MiniCPMV4_7ForConditionalGeneration
    model_dir = Path(model_dir)
    config = AutoConfig.from_pretrained(model_dir)
    with torch.device("meta"):
        model = MiniCPMV4_7ForConditionalGeneration._from_config(
            config, attn_implementation=attn, dtype=dtype)
    handles = {}
    for f in sorted(model_dir.glob("*.safetensors")):
        h = safe_open(str(f), "pt")
        for k in h.keys():
            handles[k] = h
    tc = config.text_config
    lm = model.model.language_model
    for i, layer in enumerate(lm.layers):
        layer.mlp.experts = LazyExperts(handles, f"model.language_model.layers.{i}.mlp.experts.",
                                        tc, dtype)
    model.to_empty(device="cpu")
    want = dict(model.named_parameters())
    seen = set()
    with torch.no_grad():
        for k, h in handles.items():
            if ".mlp.experts." in k:
                continue
            if k not in want:
                raise KeyError(f"{k}: in the checkpoint, not in the model")
            want[k].copy_(h.get_tensor(k).to(dtype))
            seen.add(k)
    missing = sorted(set(want) - seen)
    if missing:
        raise KeyError(f"not in the checkpoint: {missing[:8]} ({len(missing)})")
    for name, mod in model.named_modules():         # the buffers to_empty left uninitialized
        if hasattr(mod, "inv_freq"):
            inv, sc = mod.compute_default_rope_parameters(mod.config)
            mod.inv_freq = inv.to(torch.float32)
            mod.original_inv_freq = inv.clone().to(torch.float32)
            mod.attention_scaling = sc
    bufs = [n for n, _ in model.named_buffers() if not n.endswith(("inv_freq",))]
    if bufs:
        raise RuntimeError(f"buffers this loader does not recompute: {bufs[:8]}")
    model.eval()
    return model, AutoProcessor.from_pretrained(model_dir)


def inputs(proc, prompt: str, image=None, max_slices: int = 9, think: bool = False):
    """The processor's model inputs for one user turn (the chat template, generation prompt)."""
    content = [{"type": "text", "text": prompt}]
    if image is not None:
        content = [{"type": "image", "path": str(image)}] + content
    msgs = [{"role": "user", "content": content}]
    return proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                    return_dict=True, return_tensors="pt",
                                    max_slice_nums=max_slices, enable_thinking=think)


def greedy(a) -> dict:
    model, proc = load(a.model, getattr(torch, a.dtype))
    x = inputs(proc, a.prompt, a.image, a.max_slices, a.think)
    ids = [int(t) for t in x["input_ids"][0]]
    t = time.time()
    with torch.no_grad():
        out = model.generate(**x, max_new_tokens=a.n, min_new_tokens=a.n, do_sample=False,
                             output_scores=True, return_dict_in_generate=True)
    new = [int(v) for v in out.sequences[0, len(ids):]]
    top = []
    for sc in out.scores:
        v, i = torch.topk(sc[0].float(), 8)
        top.append([[int(p), round(float(q), 4)] for p, q in zip(i, v)])
    return dict(model=str(a.model), prompt=a.prompt, image=a.image, max_slices=a.max_slices,
                dtype=a.dtype, ids=ids, tokens=new, text=proc.tokenizer.decode(new), top=top,
                seconds=round(time.time() - t, 1))


def logits(a) -> None:
    model, proc = load(a.model, getattr(torch, a.dtype))
    if a.ids:
        j = json.loads(Path(a.ids).read_text())
        if j.get("image") or a.image:
            raise SystemExit("--ids with an image: give the image run's prompt instead")
        x = {"input_ids": torch.tensor([j["ids"] + j.get("tokens", [])])}
        x["attention_mask"] = torch.ones_like(x["input_ids"])
    else:
        x = inputs(proc, a.prompt, a.image, a.max_slices, a.think)
    z = {"ids": x["input_ids"][0].numpy()}
    with torch.no_grad():
        if "target_sizes" in x:
            pos, delta = model.model.get_rope_index(
                x["input_ids"], mm_token_type_ids=x["mm_token_type_ids"],
                target_sizes=x["target_sizes"], attention_mask=x.get("attention_mask"))
            z.update(positions=pos[:, 0].numpy(), rope_delta=delta.numpy(),
                     pixel_values=x["pixel_values"].float().numpy(),
                     target_sizes=x["target_sizes"].numpy(),
                     mm_token_type_ids=x["mm_token_type_ids"][0].numpy())
        out = model(**x, output_hidden_states=a.hidden, use_cache=False)
    z["logits"] = out.logits[0].float().numpy()
    if a.hidden:
        z["hidden"] = torch.stack([h[0] for h in out.hidden_states]).float().numpy()
    np.savez(a.out, **z)
    print(f"{a.out}: {len(z['ids'])} positions", file=sys.stderr)


# =============================================================================== vision
def card_tower(vs, W, fmt: str = "int8"):
    """(VisionImage, IsaBackend): the card's vision tower and projector in `fmt`
    (VisionImage.formats) on the ISA simulator, the board's configuration with 1 GiB of DRAM."""
    from opentpu.isasim import board_config
    from opentpu.llm import minicpmv as MV
    from opentpu.llm.qwen3 import IsaBackend, _Bump
    cfg = board_config(DRAM_BYTES=1 << 30)
    vi = MV.VisionImage(vs, cfg, _Bump(), nmax=2048, out_rows=128, fmt=fmt)
    img = np.zeros(cfg.DRAM_BYTES, np.uint8)

    def put(addr, a):
        v = np.ascontiguousarray(a).view(np.uint8).reshape(-1)
        img[addr:addr + v.size] = v
    vi.build(W, put)
    return vi, IsaBackend(cfg, [img])


def _reply_logits(model, x, reply, feats=None) -> np.ndarray:
    """The logits [len(reply) + 1, V] of the reply's positions, teacher-forced on `reply` (HF's
    greedy tokens): the first token's and each next one's; feats: every crop's rows to use
    instead of HF's get_image_features (its pooler_output)."""
    from types import SimpleNamespace
    x = dict(x)
    if reply:
        add = torch.tensor([reply], dtype=x["input_ids"].dtype)
        x["input_ids"] = torch.cat([x["input_ids"], add], 1)
        x["attention_mask"] = torch.ones_like(x["input_ids"])
        if "mm_token_type_ids" in x:
            x["mm_token_type_ids"] = torch.cat([x["mm_token_type_ids"], torch.zeros_like(add)], 1)
    if feats is not None:
        t = [torch.from_numpy(np.asarray(f, np.float32)).to(model.dtype) for f in feats]
        model.model.get_image_features = lambda *a, **k: SimpleNamespace(pooler_output=t)
    try:
        out = model(**x, use_cache=False, logits_to_keep=len(reply) + 1)
    finally:
        if feats is not None:
            del model.model.get_image_features
    return out.logits[0].double().numpy()


def _dkl(p_logits, q_logits) -> float:
    lp = p_logits - p_logits.max()
    lp = lp - np.log(np.exp(lp).sum())
    lq = q_logits - q_logits.max()
    lq = lq - np.log(np.exp(lq).sum())
    return float((np.exp(lp) * (lp - lq)).sum())


def _cos(a, b) -> np.ndarray:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)


def vision(a) -> None:
    """docs/minicpmv.md 5: per image, HF's crop features (fp32) against vision_reference with
    the projector's exact GELU (erf) and with its tanh form, and against the card's tower on
    the ISA simulator in each of --card's formats: every crop's row cosines (min / mean / 1st
    percentile); then HF's greedy reply (-n tokens, through its first end token: the positions
    after it are not a reply) teacher-forced with each variant's rows put in HF's place: the
    dKL against HF's own at the reply's first token and over the reply, and the share of
    positions whose top token is HF's. a.rows: a previous run's npz, whose card-<format> rows
    (each image's, its crops' in order) stand for the tower's on the ISA simulator."""
    from opentpu.llm import load_spec
    from opentpu.llm import minicpmv as MV
    model, proc = load(a.model, torch.float32)
    vs = MV.VisionSpec.from_hf(a.model)
    W = MV.load_vision(a.model)
    eos = set(load_spec(a.model).eos)
    prev = None if a.rows is None else np.load(a.rows)
    cards = {f: None if prev is not None else card_tower(vs, W, f) for f in a.card}
    z, report = {}, []
    for ii, img in enumerate(a.images):
        x = inputs(proc, a.prompt, img, a.max_slices)
        im = MV.prepare(img, vs, a.max_slices)
        assert [list(t) for t in x["target_sizes"].tolist()] == [list(s) for s in im.sizes]
        hf = [t.float().numpy() for t in model.model.get_image_features(
            x["pixel_values"], x["target_sizes"]).pooler_output]
        var = {"erf": [], "tanh": []}
        var.update({"card-" + f: [] for f in a.card})
        secs = {f: 0.0 for f in a.card}
        at = np.cumsum([0] + [h * w // 16 for h, w in im.sizes])
        for ci, (P, size) in enumerate(zip(im.patches, im.sizes)):
            var["erf"].append(MV.vision_reference(vs, W, P, size))
            var["tanh"].append(MV.vision_reference(vs, W, P, size, gelu=MV._gelu_tanh))
            for f, card in cards.items():
                if card is None:            # (a previous run's rows)
                    var["card-" + f].append(prev[f"{ii}/card-{f}"][at[ci]:at[ci + 1]])
                    continue
                vi, be = card
                t = time.time()
                var["card-" + f].append(vi.encode(be, P, size))
                secs[f] += time.time() - t
                print(f"{Path(img).name}: crop {list(size)} {f} {time.time() - t:.0f} s",
                      file=sys.stderr, flush=True)
        g = model.generate(**x, max_new_tokens=a.n, do_sample=False)
        gen = [int(t) for t in g[0, x["input_ids"].shape[1]:]]
        gen = gen[:next((i + 1 for i, t in enumerate(gen) if t in eos), len(gen))]
        reply = gen[:-1]                    # (the logits of its positions predict all of gen)
        base = _reply_logits(model, x, reply)
        r = dict(image=str(img), crops=[list(s) for s in im.sizes], rows=sum(len(h) for h in hf),
                 reply=proc.tokenizer.decode(gen), reply_tokens=len(gen),
                 top=int(base[0].argmax()),
                 isa_s={f: round(v, 1) for f, v in secs.items()})
        for k, v in var.items():
            c = np.concatenate([_cos(p, q) for p, q in zip(v, hf)])
            lg = _reply_logits(model, x, reply, v)
            d = [_dkl(p, q) for p, q in zip(base, lg)]
            r[k] = dict(cos_min=round(float(c.min()), 6), cos_mean=round(float(c.mean()), 6),
                        cos_p1=round(float(np.percentile(c, 1)), 6),
                        dkl_first=d[0], dkl_reply=float(np.mean(d)),
                        top_reply=float(np.mean(lg.argmax(-1) == base.argmax(-1))))
            z[f"{ii}/{k}"] = np.concatenate(v)
        z[f"{ii}/hf"] = np.concatenate(hf)
        report.append(r)
        print(json.dumps(r), flush=True)
    if a.out:
        np.savez(a.out, **z)
        Path(a.out).with_suffix(".json").write_text(json.dumps(report, indent=1))


# =============================================================================== check
def tiny_checkpoint(d: Path, vocab: int = 1080) -> Path:
    """A tiny random MiniCPM-V 4.7 checkpoint in d (config, safetensors, the real tokenizer's
    files are not needed: the check feeds ids)."""
    from transformers import MiniCPMV4_7Config, MiniCPMV4_7ForConditionalGeneration
    torch.manual_seed(0)
    kinds = ["linear_attention", "linear_attention", "full_attention"]
    cfg = MiniCPMV4_7Config(
        text_config=dict(model_type="qwen3_5_moe_text", hidden_size=256, num_hidden_layers=3,
                         num_attention_heads=8, num_key_value_heads=2, head_dim=256,
                         vocab_size=vocab, layer_types=kinds, linear_num_key_heads=4,
                         linear_num_value_heads=8, linear_key_head_dim=128,
                         linear_value_head_dim=128, num_experts=8, num_experts_per_tok=2,
                         moe_intermediate_size=128, shared_expert_intermediate_size=128,
                         rope_parameters={"rope_type": "default", "rope_theta": 1e7,
                                          "partial_rotary_factor": 0.25,
                                          "mrope_interleaved": True,
                                          "mrope_section": [11, 11, 10]}),
        vision_config=dict(hidden_size=128, intermediate_size=192, num_hidden_layers=3,
                           num_attention_heads=2, image_size=70, patch_size=14),
        insert_layer_id=1, image_token_id=1000, video_token_id=1001, image_start_id=1002,
        image_end_id=1003, slice_start_id=1004, slice_end_id=1005, newline_id=198)
    m = MiniCPMV4_7ForConditionalGeneration(cfg).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            p.copy_(0.05 * torch.randn_like(p) + (1.0 if n.endswith("norm.weight") and
                                                    "linear_attn" in n else 0.0))
    # the real checkpoint's layout: fused experts (save_pretrained would split them per expert)
    from safetensors.torch import save_file
    cfg.save_pretrained(d)
    save_file({k: v.contiguous() for k, v in m.state_dict().items()},
              str(d / "model.safetensors"))
    return d


def check() -> None:
    import tempfile
    from transformers import MiniCPMV4_7ForConditionalGeneration
    with tempfile.TemporaryDirectory() as td:
        d = tiny_checkpoint(Path(td))
        ref = MiniCPMV4_7ForConditionalGeneration.from_pretrained(d, dtype=torch.float32).eval()
        lazy = load_model_only(d, torch.float32)
        ids =torch.tensor([[int(t) for t in np.random.default_rng(0).integers(0, 1000, 20)]])
        with torch.no_grad():
            a = ref(input_ids=ids).logits
            b = lazy(input_ids=ids).logits
        err = float((a - b).abs().max())
        print(f"tiny MiniCPM-V: LazyExperts against from_pretrained, max |logit diff| {err:.3g}")
        assert err == 0.0, err


def procheck(a) -> None:
    """The host's port (opentpu.llm.minicpmv: decode, crops, patches, placeholder text, canvas
    positions) against HF's processor and get_rope_index, bit for bit, on each image at each
    max-slices."""
    from transformers import AutoConfig, AutoProcessor, MiniCPMV4_7ForConditionalGeneration
    from opentpu.llm import minicpmv as MV
    proc = AutoProcessor.from_pretrained(a.model)
    vs = MV.VisionSpec.from_hf(a.model)
    with torch.device("meta"):
        m = MiniCPMV4_7ForConditionalGeneration._from_config(AutoConfig.from_pretrained(a.model))
    tok = proc.tokenizer
    bad = 0
    for img in a.images:
        for ms in a.slices:
            x = inputs(proc, a.prompt, img, ms)
            im = MV.prepare(img, vs, ms)
            msgs = [{"role": "user", "content": [{"type": "image"},
                                                 {"type": "text", "text": a.prompt}]}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
            ids = tok(MV.expand(text, [im]))["input_ids"]
            pv = x["pixel_values"][0].float().numpy()            # [3, p, N p]
            p = vs.patch
            hf_rows = pv.reshape(3, p, -1, p).transpose(2, 0, 1, 3).reshape(-1, 3 * p * p)
            ours = np.concatenate(im.patches)
            pos, delta = MV.rope_index(ids, im.sizes, vs)
            hp, hd = m.model.get_rope_index(x["input_ids"],
                                            mm_token_type_ids=x["mm_token_type_ids"],
                                            target_sizes=x["target_sizes"])
            ok = dict(ids=list(x["input_ids"][0]) == ids,
                      sizes=[tuple(s) for s in x["target_sizes"].tolist()] == list(im.sizes),
                      pixels=hf_rows.shape == ours.shape and np.array_equal(hf_rows, ours),
                      positions=np.array_equal(hp[:, 0].numpy(), pos) and int(hd) == delta)
            bad += not all(ok.values())
            print(f"{Path(img).name} max_slices {ms}: grid {im.grid}, crops {im.sizes}, "
                  f"{im.tokens} image tokens, {len(ids)} ids, delta {delta}: "
                  + " ".join(f"{k} {'ok' if v else 'DIFF'}" for k, v in ok.items()))
            if not ok["pixels"] and hf_rows.shape == ours.shape:
                print(f"   pixels: max |diff| {np.abs(hf_rows - ours).max()}, "
                      f"{int((hf_rows != ours).sum())} of {ours.size} differ")
    if bad:
        raise SystemExit(f"{bad} cases differ")


def load_model_only(model_dir, dtype):
    """load() without the processor (a checkpoint without tokenizer files)."""
    import transformers
    real = transformers.AutoProcessor.from_pretrained
    transformers.AutoProcessor.from_pretrained = staticmethod(lambda *a, **k: None)
    try:
        return load(model_dir, dtype)[0]
    finally:
        transformers.AutoProcessor.from_pretrained = real


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=("greedy", "logits", "check", "procheck", "vision"))
    ap.add_argument("model", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--prompt", default="What is the capital of France? Answer in one sentence.")
    ap.add_argument("--image")
    ap.add_argument("--ids", help="logits: a greedy OUT.json whose prompt ids and tokens to feed")
    ap.add_argument("-n", type=int, default=32)
    ap.add_argument("--max-slices", type=int, default=9)
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--hidden", action="store_true")
    ap.add_argument("--images", nargs="*", default=[], help="procheck, vision: the images")
    ap.add_argument("--card", nargs="*", default=["int8"],
                    help="vision: the card tower's formats (VisionImage.formats) to check")
    ap.add_argument("--slices", nargs="*", type=int, default=[9, 1],
                    help="procheck: the max-slices to check")
    ap.add_argument("--rows", help="vision: a previous run's OUT.npz, whose card rows to take "
                                   "instead of the ISA simulator's tower")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    if a.mode == "check":
        check()
    elif a.mode == "procheck":
        procheck(a)
    elif a.mode == "vision":
        vision(a)
    elif a.mode == "greedy":
        r = greedy(a)
        Path(a.out).write_text(json.dumps(r))
        print(json.dumps({k: r[k] for k in ("tokens", "text", "seconds")}))
    else:
        logits(a)


if __name__ == "__main__":
    main()
