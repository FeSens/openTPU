"""GPT-OSS on its real weights (opentpu/llm/gptoss.py): Hugging Face's forward, the fp32 reference
against it, the experts' MXFP4 import, and a cut of the first layers on the ISA simulator.

    python3 tools/exp/gptoss_check.py hf MODEL OUT.npz [--dtype bf16|fp32] [--tokens IN.npz] [--new 16]
    python3 tools/exp/gptoss_check.py ref MODEL HF.npz [--emulation [--wformat fp4] [--embed f32]]
    python3 tools/exp/gptoss_check.py mx MODEL
    python3 tools/exp/gptoss_check.py cut MODEL [--layers 2] [--new 16] [--wformat fp4] [--embed f32]

hf: Hugging Face's GptOssForCausalLM (eager attention) one decoder layer at a time: every layer
is on the meta device but while it runs, its weights read from the checkpoint as it runs (the
experts' MXFP4 dequantized by transformers' own convert_moe_packed_tensors), so the 20B runs in
~5 GB; greedy generation of --new tokens after each prompt, then one teacher-forced forward over
the whole sequence (its logits saved, fp32). --tokens: the sequences of an earlier run instead,
teacher-forced only (no generation).
ref: the fp32 numpy reference (gptoss.reference_logits) over each sequence of an hf run: its
greedy choice at every generated position against the tokens, and the logits' cosine per
position. --emulation: the device's quantization points instead (gptoss.emulated_logits, the
whole model: what the card computes but for fp32's rounding near ties), its attention in
--wformat and the formats string (OTPU_FORMATS, else Spec.formats: int8), the embedding table
--embed (the Spec's by default); each position it picks otherwise listed with Hugging Face's
margin there.
mx: every expert's three matrices imported into fp4 slots (gptoss.expert_mats: exact, else the
values quantized), counted.
cut: the first --layers layers (the 20B's head and embedding) on the ISA simulator at --target's
Config (bcu1525-wg: WG 8, the channel hash; every expert resident, imported from MXFP4): greedy
generation on the device (generate_card) after each prompt, then the device's logits stepped
over the whole sequence against the emulation (gptoss.emulated_logits) and the fp32 reference
of the same cut.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from opentpu.llm import gptoss as GO                                    # noqa: E402

PROMPTS = ("The capital of France is", "def fibonacci(n):\n    ")


def _cos(a, b):
    return (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)


def _tokens(model):
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(Path(model) / "tokenizer.json"))
    return [tk.encode(p, add_special_tokens=False).ids for p in PROMPTS]


def hf_model(model, dtype):
    """GptOssForCausalLM with its decoder layers' weights read from the checkpoint as each
    layer runs (forward pre-hooks) and dropped after it (forward hooks)."""
    import torch
    import transformers
    from transformers.integrations.mxfp4 import convert_moe_packed_tensors
    from safetensors import safe_open
    cfg = transformers.AutoConfig.from_pretrained(model)
    cfg._attn_implementation = "eager"
    with torch.device("meta"):
        m = transformers.GptOssForCausalLM(cfg)
    m = m.to(dtype).eval()
    at = {}
    for f in sorted(Path(model).glob("*.safetensors")):
        h = safe_open(str(f), "pt")
        for k in h.keys():
            at[k] = h
    get = lambda k: at[k].get_tensor(k)                                 # noqa: E731
    top = {"model.embed_tokens.weight": m.model.embed_tokens.weight,
           "model.norm.weight": m.model.norm.weight, "lm_head.weight": m.lm_head.weight}
    m.load_state_dict({k: get(k).to(dtype) for k in top}, strict=False, assign=True)
    m.model.rotary_emb = type(m.model.rotary_emb)(cfg)                   # its buffers, on the CPU

    def load(layer, args, kwargs=None):
        i = layer.self_attn.layer_idx
        p = f"model.layers.{i}."
        sd = {}
        for k in at:
            if not k.startswith(p) or "_blocks" in k or "_scales" in k:
                continue
            sd[k[len(p):]] = get(k).to(dtype)
        for n in ("gate_up_proj", "down_proj"):
            e = p + "mlp.experts." + n
            sd["mlp.experts." + n] = convert_moe_packed_tensors(
                get(e + "_blocks"), get(e + "_scales"), dtype=dtype)
        layer.load_state_dict(sd, strict=True, assign=True)

    def drop(layer, args, out):
        layer.to("meta")
    for layer in m.model.layers:
        layer.register_forward_pre_hook(load)
        layer.register_forward_hook(drop)
    return m


def cmd_hf(a):
    import torch
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
    m = hf_model(a.model, dtype)
    if a.tokens:
        z = np.load(a.tokens)
        seqs = [(list(z[f"toks{j}"]), int(z[f"plen{j}"])) for j in range(int(z["n"]))]
    else:
        seqs = [(t, len(t)) for t in _tokens(a.model)]
    out = {"n": len(seqs)}
    for j, (toks, plen) in enumerate(seqs):
        t0 = time.time()
        with torch.no_grad():
            if not a.tokens:
                g = m.generate(torch.tensor([toks]), max_new_tokens=a.new,
                               min_new_tokens=a.new, do_sample=False, pad_token_id=199999)
                toks = g[0].tolist()
            lg = m(torch.tensor([toks]), use_cache=False).logits[0].float().numpy()
        out[f"toks{j}"], out[f"plen{j}"], out[f"logits{j}"] = np.array(toks), plen, lg
        print(f"prompt {j}: {plen} tokens + {len(toks) - plen}: {toks[plen:]} "
              f"({time.time() - t0:.0f} s)", flush=True)
    np.savez(a.out, **out)
    return 0


def cmd_ref(a):
    spec = GO.Spec.from_hf(a.model)
    if a.embed:
        spec = dataclasses.replace(spec, embed=a.embed)
    W = GO.load_weights(a.model)
    z = np.load(a.hf)
    ok = True
    if a.emulation:
        print(f"the emulation: attention {a.wformat} "
              f"{os.environ.get('OTPU_FORMATS', spec.formats) or '-'}, the embedding table "
              f"{spec.embed}")
    for j in range(int(z["n"])):
        toks, plen, hf = list(z[f"toks{j}"]), int(z[f"plen{j}"]), z[f"logits{j}"]
        t0 = time.time()
        ref = GO.emulated_logits(spec, W, toks, wformat=a.wformat) if a.emulation else \
            GO.reference_logits(spec, W, toks)
        c = _cos(ref, hf)
        pick = ref[plen - 1:-1].argmax(1)
        same = int((pick == np.array(toks[plen:])).sum())
        top2 = np.sort(hf[plen - 1:-1], 1)[:, -2:]
        print(f"prompt {j}: greedy {same} / {len(toks) - plen} of the tokens; cosine per "
              f"position min {c.min():.7f} median {np.median(c):.7f}; argmax equal at "
              f"{int((ref.argmax(1) == hf.argmax(1)).sum())} / {len(toks)} positions; HF's "
              f"margin at the generated positions min {np.min(top2[:, 1] - top2[:, 0]):.3f} "
              f"({time.time() - t0:.0f} s)", flush=True)
        print("  1 - cosine:", " ".join(f"{1 - x:.1e}" for x in c))
        for g in np.nonzero(pick != np.array(toks[plen:]))[0]:
            print(f"  generated position {g}: picks {pick[g]} for {toks[plen + g]}, HF's margin "
                  f"{top2[g, 1] - top2[g, 0]:.4f}")
        ok &= same == len(toks) - plen and (a.emulation or c.min() >= 0.999)
    return 0 if ok else 1


def cmd_mx(a):
    from opentpu.llm import moe as MO
    spec = GO.Spec.from_hf(a.model)
    W = GO.load_weights(a.model)
    Hp = -(-spec.hidden // 128) * 128
    fmt = MO.ExpertFormat(Hp, spec.moe.ffn, 128, "fp4", 8, bias=True, rows=spec.hidden,
                          tight=True)                 # (the Image's on bcu1525-wg)
    n, t0 = {}, time.time()
    for i in range(spec.layers):
        li = {}
        for e in range(spec.moe.E):
            for name, h in zip(("gate", "up", "down"), GO.expert_mats(W, i, e, fmt).how):
                n[name, h] = n.get((name, h), 0) + 1
                li[h] = li.get(h, 0) + 1
        print(f"layer {i}: {li} ({time.time() - t0:.0f} s)", flush=True)
    print("matrices:", {f"{k[0]} {k[1]}": v for k, v in sorted(n.items())})
    return 0


def cmd_cut(a):
    from opentpu import target as T
    from opentpu.llm.qwen3 import Engine
    spec = GO.Spec.from_hf(a.model)
    s = dataclasses.replace(spec, kinds=spec.kinds[:a.layers], embed=a.embed or spec.embed)
    W = GO.load_weights(a.model)
    cap = 512
    t0 = time.time()
    n = s.image(T.config(a.target, DRAM_BYTES=1 << 40), cap, wformat=a.wformat,
                head_format="int8", lookup=True).nbytes
    cfg = T.config(a.target, DRAM_BYTES=1 << max(20, (n - 1).bit_length()))
    eng = Engine(s, GO.load_weights(a.model), cap=cap, cfg=cfg, rows=1, wformat=a.wformat,
                 resident=True)
    print(f"{a.layers} layers: image {eng.image.nbytes / 2**20:.0f} MiB, DRAM "
          f"{cfg.DRAM_BYTES / 2**20:.0f} MiB, the embedding table {s.embed}, experts "
          f"{eng.image.imports} ({time.time() - t0:.0f} s)", flush=True)
    ok = True
    for j, toks in enumerate(_tokens(a.model)):
        t0 = time.time()
        eng.reset()
        for t in toks[:-1]:
            eng.step(t)
        gen = eng.generate_card(toks[-1], a.new, stop_ids=[])
        seq = toks + gen
        eng.reset()
        eng.server.history = []
        dev = np.array([eng.step(t) for t in seq])
        routes = []
        emu = GO.emulated_logits(s, W, seq, wformat=a.wformat, routes=routes)
        ref = GO.reference_logits(s, W, seq)
        p = len(toks)
        # the server's requests: [0] the one before this pass's first (a request is served when
        # the next one is posted), then (t, layer) at t * layers + layer + 1, the last pending
        card, chose, flips = eng.server.history, {}, 0
        for t, li, ids, margin in routes:
            i = t * a.layers + li + 1
            if i >= len(card):
                continue
            chose[t, li] = got = [g - li * s.moe.E for g in card[i]]
            if set(got) != set(ids):
                flips += 1
                print(f"  position {t} layer {li}: the card's experts {got}, the emulation's "
                      f"{list(ids)} (its k-th and (k+1)-th logits {margin:.2e} apart)")
        print("  1 - cosine against the emulation:",
              " ".join(f"{1 - x:.0e}" for x in _cos(dev, emu)))
        if flips:
            e2 = GO.emulated_logits(s, W, seq, wformat=a.wformat, experts=chose)
            print(f"  the emulation given the card's experts: greedy "
                  f"{int((e2[len(toks) - 1:-1].argmax(1) == np.array(gen)).sum())} / {a.new}, "
                  f"1 - cosine:", " ".join(f"{1 - x:.0e}" for x in _cos(dev, e2)))
        steps = bool(np.array_equal(dev[p - 1:-1].argmax(1), np.array(gen)))
        ce, cr = _cos(dev, emu), _cos(dev, ref)
        same_e = int((emu[p - 1:-1].argmax(1) == np.array(gen)).sum())
        same_r = int((ref[p - 1:-1].argmax(1) == np.array(gen)).sum())
        print(f"prompt {j}: {p} tokens, the device's {a.new}: {gen}; its steps' argmax the same "
              f"{steps}; the emulation's greedy {same_e} / {a.new}, argmax equal at "
              f"{int((dev.argmax(1) == emu.argmax(1)).sum())} / {len(seq)}, cosine min "
              f"{ce.min():.9f}; the fp32 reference's greedy {same_r} / {a.new}, argmax equal at "
              f"{int((dev.argmax(1) == ref.argmax(1)).sum())} / {len(seq)}, cosine min "
              f"{cr.min():.6f} median {np.median(cr):.6f} ({time.time() - t0:.0f} s)",
              flush=True)
        ok &= steps and same_e == a.new
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hf")
    h.add_argument("model")
    h.add_argument("out")
    h.add_argument("--dtype", default="bf16")
    h.add_argument("--tokens")
    h.add_argument("--new", type=int, default=16)
    r = sub.add_parser("ref")
    r.add_argument("model")
    r.add_argument("hf")
    r.add_argument("--emulation", action="store_true")
    r.add_argument("--wformat", default="fp4")
    r.add_argument("--embed", choices=("f32", "int8"))
    x = sub.add_parser("mx")
    x.add_argument("model")
    c = sub.add_parser("cut")
    c.add_argument("model")
    c.add_argument("--layers", type=int, default=2)
    c.add_argument("--new", type=int, default=16)
    c.add_argument("--wformat", default="fp4")
    c.add_argument("--target", default="bcu1525-wg")
    c.add_argument("--embed", choices=("f32", "int8"))
    a = ap.parse_args(argv)
    return {"hf": cmd_hf, "ref": cmd_ref, "mx": cmd_mx, "cut": cmd_cut}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
