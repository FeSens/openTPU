#!/usr/bin/env python3
"""MTP transplant (docs/offload.md 15.6): Qwen3.5-35B-A3B's MTP head (its mtp.* layer: two input
norms, fc over [embedding, hidden], one full-attention MoE decoder layer, mtp.norm) as a k = 1
drafter, on the base model's own hidden states and on MiniCPM-V-4.7-35B-A3B's (the same backbone
fine-tuned, no mtp.* of its own). Host reference (a test measurement, not serving): HF
transformers 5.18 with LazyExperts (tools/minicpmv_hf.py, branch minicpmv: every expert read
from the safetensors when a token routes to it), float32. minicpmv_hf.py comes from this tree's
tools/ or from MCPV_TREE's; on the simulation host PYTHONPATH=~/mcpv/tf518:. (transformers 5.18).

    gen MODEL OUT.npz [-n 256] [--only 0,3]
        greedy replies (thinking off, the chat template) to tools/mtp_accept.py's nine prompts,
        then one teacher-forced forward per sequence: ids, prompt lengths and the final-normed
        hidden rows [T, H] (model.norm's output). MiniCPM-V loads through minicpmv_hf.load
        (the whole model, text-only inputs), the base as Qwen3_5MoeForCausalLM.
    accept BASE GEN.npz OUT.json [--embed MODEL] [--head MODEL]
        the base's MTP layer over (embedding of token t + 1, hidden t), row t drafting token
        t + 2 (mtp_accept.MTPHead's order and positions), with the LM head of --head (default
        the GEN model's) in float and as fp4 over its 32K lowest ids (QACT int8 activations, as
        mtp_accept.heads); greedy acceptance: the draft equals the model's token. Writes
        mtp_accept's JSON (n-gram drafts too), so mtp_accept.py --summary OUT.json --c2 C reads
        it. --embed: whose embedding table feeds the layer (default the GEN model's).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

LES = Path(__file__).resolve().parents[2]
MCPV = LES if (LES / "tools/minicpmv_hf.py").exists() else Path(
    os.environ.get("MCPV_TREE", Path.home() / "mcpv/tree"))


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


MA = _mod("mtp_accept", LES / "tools/mtp_accept.py")
HEAD_N = 32768


def is_minicpmv(d: Path) -> bool:
    return "minicpm" in json.loads((d / "config.json").read_text()).get("model_type", "")


def handles(d: Path) -> dict:
    from safetensors import safe_open
    h = {}
    for f in sorted(d.glob("*.safetensors")):
        s = safe_open(str(f), "pt")
        for k in s.keys():
            h[k] = s
    return h


def load_base(d: Path, dtype=torch.float32):
    """Qwen3.5-35B-A3B's language model (Qwen3_5MoeForCausalLM on its text_config), the
    checkpoint's model.language_model.* as model.*, every expert a LazyExperts (minicpmv_hf's),
    mtp.* and model.visual.* left out."""
    from transformers import AutoConfig
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M
    hf = _mod("minicpmv_hf", MCPV / "tools/minicpmv_hf.py")
    tc = AutoConfig.from_pretrained(d).text_config
    tc._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = M.Qwen3_5MoeForCausalLM._from_config(tc, attn_implementation="sdpa", dtype=dtype)
    h = handles(d)
    for i, layer in enumerate(model.model.layers):
        layer.mlp.experts = hf.LazyExperts(h, f"model.language_model.layers.{i}.mlp.experts.", tc,
                                           dtype)
    model.to_empty(device="cpu")
    want = dict(model.named_parameters())
    seen = set()
    with torch.no_grad():
        for k, s in h.items():
            if ".mlp.experts." in k or k.startswith(("mtp.", "model.visual.")):
                continue
            n = k.replace("model.language_model.", "model.", 1)
            want[n].copy_(s.get_tensor(k).to(dtype))
            seen.add(n)
    missing = sorted(set(want) - seen)
    if missing:
        raise KeyError(f"not in the checkpoint: {missing[:8]} ({len(missing)})")
    for _, mod in model.named_modules():
        if hasattr(mod, "inv_freq"):
            inv, sc = mod.compute_default_rope_parameters(mod.config)
            mod.inv_freq = inv.to(torch.float32)
            mod.original_inv_freq = inv.clone().to(torch.float32)
            mod.attention_scaling = sc
    model.eval()
    return model, model.model.norm


def gen(a) -> None:
    from transformers import AutoTokenizer
    d = Path(a.model)
    t0 = time.time()
    if is_minicpmv(d):
        hf = _mod("minicpmv_hf", MCPV / "tools/minicpmv_hf.py")
        model, _ = hf.load(d)
        norm = model.model.language_model.norm
    else:
        model, norm = load_base(d)
    tok = AutoTokenizer.from_pretrained(d)
    torch.set_num_threads(a.threads)
    print(f"{d.name}: loaded in {time.time() - t0:.0f} s", flush=True)
    cap = {}
    norm.register_forward_hook(lambda m, i, o: cap.__setitem__("h", o))
    out = {}
    for i, (kind, text) in enumerate(MA.PROMPTS):
        if a.only is not None and i not in a.only:
            continue
        ids = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
                                      add_generation_prompt=True, enable_thinking=False)
        ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
        t = time.time()
        with torch.no_grad():
            x = torch.tensor([ids])
            seq = model.generate(input_ids=x, attention_mask=torch.ones_like(x),
                                 max_new_tokens=a.n, do_sample=False)[0].tolist()
            y = torch.tensor([seq])
            lg = model(input_ids=y, attention_mask=torch.ones_like(y), use_cache=False).logits[0]
        am = lg.argmax(-1).numpy()
        P = len(ids)
        agree = float(np.mean(am[P - 1:-1] == np.array(seq[P:]))) if len(seq) > P else 1.0
        out[f"ids{i}"] = np.array(seq, np.int64)
        out[f"P{i}"] = np.array(P)
        out[f"kind{i}"] = np.array(kind)
        out[f"hidden{i}"] = cap["h"][0].float().numpy()
        print(f"{d.name} #{i} {kind}: {P}+{len(seq) - P} tokens in {time.time() - t:.0f} s, "
              f"teacher-forced argmax = generated {agree:.3f}; "
              f"{tok.decode(seq[P:P + 24])!r}", flush=True)
        np.savez(a.out, model=np.array(d.name), **out)
    print(f"{a.out} written", flush=True)


class MTP:
    """The base's mtp.* (HF's Qwen3_5Moe modules, float32): its MoE layer's experts stacked
    from the per-expert tensors into Qwen3_5MoeExperts' gate_up_proj / down_proj."""

    def __init__(self, base: Path):
        from transformers import AutoConfig
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M
        tc = AutoConfig.from_pretrained(base).text_config
        tc._attn_implementation = "eager"
        H, eps = tc.hidden_size, tc.rms_norm_eps
        li = list(tc.layer_types).index("full_attention")
        self.pre_e, self.pre_h = M.Qwen3_5MoeRMSNorm(H, eps), M.Qwen3_5MoeRMSNorm(H, eps)
        self.fc = torch.nn.Linear(2 * H, H, bias=False)
        self.layer = M.Qwen3_5MoeDecoderLayer(tc, li)
        self.norm = M.Qwen3_5MoeRMSNorm(H, eps)
        self.rope = M.Qwen3_5MoeTextRotaryEmbedding(tc)
        mods = {"pre_fc_norm_embedding": self.pre_e, "pre_fc_norm_hidden": self.pre_h,
                "fc": self.fc, "layers.0": self.layer, "norm": self.norm}
        h = handles(base)
        E, I = tc.num_experts, tc.moe_intermediate_size
        gu = torch.empty(E, 2 * I, H)
        dn = torch.empty(E, H, I)
        n = 0
        with torch.no_grad():
            for k, s in h.items():
                if not k.startswith("mtp."):
                    continue
                name = k[4:]
                if ".mlp.experts." in name:     # layers.0.mlp.experts.E.{gate,up,down}_proj.weight
                    e, w = name.split(".mlp.experts.")[1].split(".", 1)
                    t = s.get_tensor(k).float()
                    if w.startswith("gate_proj"):
                        gu[int(e), :I] = t
                    elif w.startswith("up_proj"):
                        gu[int(e), I:] = t
                    else:
                        dn[int(e)] = t
                    n += 1
                    continue
                pre = next(p for p in mods if name.startswith(p + "."))
                mods[pre].get_parameter(name[len(pre) + 1:]).data.copy_(s.get_tensor(k).float())
                n += 1
            self.layer.mlp.experts.gate_up_proj.data.copy_(gu)
            self.layer.mlp.experts.down_proj.data.copy_(dn)
        want = sum(len(list(m.parameters())) for m in mods.values()) - 2 + 3 * E
        if n != want:
            raise ValueError(f"{n} mtp tensors, the modules take {want}")
        for m in mods.values():
            m.float().eval()

    def __call__(self, emb_next: np.ndarray, hidden: np.ndarray) -> np.ndarray:
        """emb_next [L-1, H] (the embedding of tokens 1..L-1), hidden [L-1, H] (the model's
        rows 0..L-2 after model.norm) -> the normed outputs [L-1, H]: row t drafts t + 2."""
        with torch.no_grad():
            e = torch.as_tensor(emb_next)[None].float()
            hh = torch.as_tensor(hidden)[None].float()
            x = self.fc(torch.cat([self.pre_e(e), self.pre_h(hh)], -1))
            L = x.shape[1]
            pos = torch.arange(L)[None]
            pe = self.rope(x, pos[None].expand(3, 1, -1))
            mask = torch.full((L, L), float("-inf")).triu(1)[None, None]
            x = self.layer(x, position_embeddings=pe, attention_mask=mask, position_ids=pos)
            return self.norm(x)[0].numpy()


def tensor(d: Path, names) -> np.ndarray:
    h = handles(d)
    k = next(n for n in names if n in h)
    return h[k].get_tensor(k).float().numpy()


def accept(a) -> None:
    from opentpu import quant as Q
    z = np.load(a.gen)
    gdir = Path(a.model_dir) if a.model_dir else None
    name = str(z["model"])
    gdir = gdir or next(p for p in (Path.home() / "otpu-models").iterdir() if p.name == name)
    edir = Path(a.embed) if a.embed else gdir
    hdir = Path(a.head) if a.head else gdir
    t0 = time.time()
    mtp = MTP(Path(a.base))
    emb = tensor(edir, ("model.language_model.embed_tokens.weight", "model.embed_tokens.weight"))
    Wh = tensor(hdir, ("lm_head.weight",))
    q = np.asarray(Q.fake_w(Wh[:HEAD_N], "fp4", 128)).view(np.ndarray).astype(np.float32)
    print(f"MTP of {Path(a.base).name}, embedding {edir.name}, head {hdir.name}: "
          f"{time.time() - t0:.0f} s", flush=True)
    res = {"model": f"{name}+mtp({Path(a.base).name}) emb {edir.name} head {hdir.name}",
           "mode": "greedy", "sampling": None, "dtype": "float32", "tokens": None,
           "chat_template": True, "prompts": []}
    idx = sorted(int(k[3:]) for k in z.files if k.startswith("ids"))
    for i in idx:
        seq, P, kind = z[f"ids{i}"].tolist(), int(z[f"P{i}"]), str(z[f"kind{i}"])
        hid = z[f"hidden{i}"]
        L = len(seq)
        targets = range(P + 1, L)
        rec = {"kind": kind, "prompt_tokens": P, "generated": L - P, "ids": seq}
        for g in (2, 3):
            rec[f"ngram{g}"] = [None if (d := MA.ngram_draft(seq[:i2], g)) is None
                                else int(d == seq[i2]) for i2 in targets]
        t = time.time()
        x = mtp(emb[np.array(seq[1:])], hid[:-1])           # row t drafts t + 2
        d_full = np.argmax(x @ Wh.T, axis=1)
        d_32 = np.argmax(MA.act_int8(x) @ q.T, axis=1)
        rec["mtp-full"] = [int(d_full[i2 - 2] == seq[i2]) for i2 in targets]
        rec["mtp-fp4-32K"] = [int(d_32[i2 - 2] == seq[i2]) for i2 in targets]
        rec["in_ids"] = {"32K": float(np.mean([seq[i2] < HEAD_N for i2 in targets]))}
        res["prompts"].append(rec)
        line = " ".join(f"{k} {np.mean([v for v in rec[k] if v is not None] or [0]):.2f}"
                        for k in rec if k.startswith(("ngram", "mtp")))
        print(f"#{i} {kind}: {P}+{L - P} tokens, {time.time() - t:.0f} s; {line}", flush=True)
    Path(a.out).write_text(json.dumps(res))
    for k in ("mtp-full", "mtp-fp4-32K"):
        v = [x for p in res["prompts"] for x in p[k]]
        print(f"{k}: acceptance {np.mean(v):.3f} over {len(v)} drafts", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("model")
    g.add_argument("out")
    g.add_argument("-n", type=int, default=256)
    g.add_argument("--threads", type=int, default=8)
    g.add_argument("--only", type=lambda s: [int(x) for x in s.split(",")])
    c = sub.add_parser("accept")
    c.add_argument("base")
    c.add_argument("gen")
    c.add_argument("out")
    c.add_argument("--model-dir", help="the GEN model's checkpoint (default by its name)")
    c.add_argument("--embed")
    c.add_argument("--head")
    a = ap.parse_args()
    {"gen": gen, "accept": accept}[a.cmd](a)


if __name__ == "__main__":
    main()
