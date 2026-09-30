"""Gemma 4's quantization error, on long sequences: gemma4.emulated_logits (float64, the device's
quantization points and none of its rounding) over the whole sequence at once, with the points
switched one at a time and per-kind weight formats (docs/gemma4.md, "Long context").

    python tools/gemma4_quant_eval.py step MODEL REF.npz STEP WF HEAD [--formats attn=int8]
                                      [--quant act,kv,p]
        the top 5 at REF's prompt (ids0) + its first STEP greedy tokens (gen0), and the rank of
        REF's next token (hf_long-style npz: ids0, gen0)
    python tools/gemma4_quant_eval.py nll MODEL REF.npz WF HEAD OUT.npz [--formats ..] [--quant ..]
        the next-token NLL of the soft-capped logits over ids0: perplexity, and the argmax per
        position into OUT.npz (nll, top1)
    python tools/gemma4_quant_eval.py check TINY_DIR
        the batched emulation against emulated_logits (40 tokens, window 16; equal up to
        rounding ties of exactly .5, which the two summation orders break differently)

WF / HEAD: int8, fp4, int4 or none (float weights). The whole sequence runs at once: 900 tokens of
E2B take about 1 min in float, 10 min with fp4 layers (the 4-bit quantization's search).
"""
import argparse
import math
import sys
import time
from dataclasses import replace

import numpy as np

from opentpu.kernels import gather as GA
from opentpu.llm import gemma4 as G
from opentpu.llm.qwen3 import _fake_q, _fake_w


def emulate(spec, W, tokens, D=128, wformat="int8", hf=None, ple_format="int8", rows=None,
            quant=None, wmap=None, on_rows=None):
    """gemma4.emulated_logits over the whole sequence at once: the logits [len(rows), vocab]
    before the soft cap (rows: default all), or with on_rows each group of up to 64 rows handed
    to on_rows(first row, logits). wformat / hf: the layers' and the head's formats, "none" for
    float weights (the PLE table then float too); wmap: per-kind layer formats {"attn", "mlp",
    "ple": format}. quant: the activation quantization points applied, a subset of {"act" (the
    matmul inputs), "kv" (K / V), "p" (P)}; default all, none with wformat "none"."""
    hf = hf or wformat
    none = wformat == "none"
    quant = (set() if none else {"act", "kv", "p"}) if quant is None else set(quant)
    ident = lambda v, d=D: np.asarray(v, np.float64)
    fq = _fake_q if "act" in quant else ident
    fk = _fake_q if "kv" in quant else ident
    fp = _fake_q if "p" in quant else ident

    wmap = wmap or {}                   # per-kind layer formats: attn, mlp, ple (else wformat)

    def kind_of(n):
        return "attn" if ".self_attn." in n else "mlp" if ".mlp." in n else "ple"

    def wq(n, fmt=wformat, a=None):
        if n is not None and fmt == wformat and n.startswith("model.layers."):
            fmt = wmap.get(kind_of(n), wformat)
        a = W[n] if a is None else a
        return np.asarray(a, np.float64) if fmt == "none" else _fake_w(a, D, fmt)

    T, H, P, L = len(tokens), spec.hidden, spec.ple_dim, spec.layers
    n_q, n_kv, eps = spec.n_q, spec.n_kv, spec.eps
    Gq = n_q // n_kv
    uniq, inv = np.unique(np.asarray(tokens), return_inverse=True)
    head = "model.embed_tokens.weight" if spec.tied else "lm_head.weight"
    x = wq(head, hf, G._rows(W, head, uniq.tolist()))[inv] * math.sqrt(H)
    cols = np.concatenate([np.arange(spec.src(i) * P, (spec.src(i) + 1) * P) for i in range(L)])
    pe = np.asarray(G._rows(W, G.Weights.PLE, uniq.tolist()), np.float32)[:, cols] * np.float32(
        (P / 2) ** 0.5)
    if not none:
        S = GA.record_blocks(-(-L * P // D), ple_format)
        pe = GA.dequant_records(GA.pack_records(pe, ple_format, D, S), ple_format, D, S)[:, :L * P]
    pe = np.asarray(pe, np.float64)[inv]
    wp = W["model.per_layer_model_projection.weight"].reshape(-1, P, H)[
        [spec.src(i) for i in range(L)]].reshape(-1, H)
    wpq = wq(None, wmap.get("ple", wformat), a=wp) * H ** -0.5
    gpl = np.asarray(W["model.per_layer_projection_norm.weight"], np.float64) * 2 ** -0.5
    pli = G._norm((fq(x) @ wpq.T).reshape(T, L, P), gpl, eps) + pe.reshape(T, L, P)
    del wp, wpq, pe
    ii = np.arange(T)[:, None]
    K, V = {}, {}
    for i in range(L):
        p = f"model.layers.{spec.src(i)}."
        a = p + "self_attn."
        kind, d = spec.kinds[i], spec.hd(i)
        tab = [G.rope_tables(spec, pos, kind) for pos in range(T)]
        c = np.stack([t[0] for t in tab])[:, None, :]
        s_ = np.stack([t[1] for t in tab])[:, None, :]
        h = fq(G._norm(x, W[p + "input_layernorm.weight"], eps))
        q = (h @ wq(a + "q_proj.weight").T).reshape(T, n_q, d)
        q = fq(G._rot(G._norm(q, W[a + "q_norm.weight"], eps), c, s_, d // 2))
        if spec.kv_src[i] == i:
            k = (h @ wq(a + "k_proj.weight").T).reshape(T, n_kv, d)
            v = (h @ wq(a + "v_proj.weight").T).reshape(T, n_kv, d)
            K[i] = fk(G._rot(G._norm(k, W[a + "k_norm.weight"], eps), c, s_, d // 2))
            V[i] = fk(G._norm(v, None, eps), d)
        Kh, Vh = K[spec.kv_src[i]], V[spec.kv_src[i]]
        o = np.zeros((T, n_q, d))
        for hq in range(n_q):
            sc = q[:, hq] @ Kh[:, hq // Gq].T                       # [T, T]
            if kind == G.SLIDE:                                      # P's blocks from the window start
                Wn = -(-spec.window // D) * D
                idx = np.maximum(0, ii[:, 0] + 1 - spec.window)[:, None] + np.arange(Wn)[None, :]
                ok = idx <= ii
                scp = np.concatenate([sc, np.full((T, Wn), -np.inf)], 1)
                sb = np.where(ok, np.take_along_axis(scp, idx, 1), -np.inf)
                pp = np.exp(sb - sb.max(1, keepdims=True))
                pf = np.zeros((T, T + Wn))
                np.put_along_axis(pf, idx, fp(pp), 1)
                pf = pf[:, :T]
            else:
                Tp = -(-T // D) * D
                sb = np.full((T, Tp), -np.inf)
                sb[:, :T] = np.where(np.arange(T)[None, :] <= ii, sc, -np.inf)
                pp = np.exp(sb - sb.max(1, keepdims=True))
                pf = fp(pp)[:, :T]
            o[:, hq] = (pf @ Vh[:, hq // Gq]) / pp.sum(1, keepdims=True)
        att = fq(o.reshape(T, -1)) @ wq(a + "o_proj.weight").T
        x = x + G._norm(att, W[p + "post_attention_layernorm.weight"], eps)
        h = fq(G._norm(x, W[p + "pre_feedforward_layernorm.weight"], eps))
        m = fq(G._gelu(h @ wq(p + "mlp.gate_proj.weight").T) * (h @ wq(p + "mlp.up_proj.weight").T)) \
            @ wq(p + "mlp.down_proj.weight").T
        x = x + G._norm(m, W[p + "post_feedforward_layernorm.weight"], eps)
        g = G._gelu(fq(x) @ wq(p + "per_layer_input_gate.weight").T) * pli[:, i]
        y = fq(g) @ wq(p + "per_layer_projection.weight").T
        x = (x + G._norm(y, W[p + "post_per_layer_input_norm.weight"], eps)) * \
            np.asarray(W[p + "layer_scalar"], np.float64)
    rows = list(range(T)) if rows is None else rows
    xl = fq(G._norm(x[rows], W["model.norm.weight"], eps))
    E = W[head]
    if on_rows is None:
        return np.concatenate([xl @ wq(None, hf, E[r0:r0 + 16384]).T
                               for r0 in range(0, len(E), 16384)], 1)
    Eq = [wq(None, hf, E[r0:r0 + 16384]).T for r0 in range(0, len(E), 16384)]
    del E
    for i0 in range(0, len(rows), 64):      # on_rows(first row, logits [<= 64, vocab])
        on_rows(i0, np.concatenate([xl[i0:i0 + 64] @ e for e in Eq], 1))


def top(spec, tok, lg, n=5):
    cap = G.softcap(spec, lg)
    t = np.argsort(-lg)[:n]
    return [(int(j), tok.decode([int(j)]), round(float(lg[j]), 4), round(float(cap[j]), 4)) for j in t]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=["step", "nll", "check"])
    ap.add_argument("args", nargs="+")
    ap.add_argument("--formats", default="", help="per-kind layer formats, e.g. attn=int8")
    ap.add_argument("--quant", default=None, help="the activation points (default all): act,kv,p")
    a = ap.parse_args()
    if a.mode == "check":
        spec, W = G.Spec.from_hf(a.args[0]), G.load_weights(a.args[0])
        spec = replace(spec, window=16)
        toks = np.random.default_rng(1).integers(0, spec.vocab, 40).tolist()
        for wf, hf in (("int8", "int8"), ("fp4", "int8"), ("fp4", "fp4")):
            ref = G.emulated_logits(spec, W, toks, wformat=wf, head_format=hf)
            got = emulate(spec, W, toks, wformat=wf, hf=hf)
            print(f"{wf} / {hf} head: max |diff| {np.abs(ref - got).max():.3g} of "
                  f"{np.abs(ref).max():.3g}")
        ref = G.reference_logits(spec, W, toks)
        print(f"none vs reference_logits (fp32): max |diff| "
              f"{np.abs(ref - emulate(spec, W, toks, wformat='none')).max():.3g}")
        return
    from transformers import AutoTokenizer
    quant = None if a.quant is None else [q for q in a.quant.split(",") if q]
    wmap = dict(kv.split("=") for kv in a.formats.split(",") if kv)
    path, ref = a.args[0], np.load(a.args[1])
    spec, W = G.Spec.from_hf(path), G.load_weights(path)
    wf, hf = (a.args[3], a.args[4]) if a.mode == "step" else (a.args[2], a.args[3])
    what = f"{wf} layers{' ' + str(wmap) if wmap else ''}, {hf} head, points {a.quant or 'all'}"
    t0 = time.time()
    if a.mode == "step":
        step = int(a.args[2])
        ids = [int(t) for t in ref["ids0"]] + [int(t) for t in ref["gen0"][:step]]
        lg = emulate(spec, W, ids, wformat=wf, hf=hf, rows=[len(ids) - 1], quant=quant,
                     wmap=wmap)[0]
        tok = AutoTokenizer.from_pretrained(path)
        want = int(ref["gen0"][step])
        print(f"{what}: {len(ids)} tokens, {time.time() - t0:.0f} s")
        print("  top 5 (token, logit, capped):", top(spec, tok, lg))
        print(f"  REF's token {want} {tok.decode([want])!r}: rank {int((lg > lg[want]).sum()) + 1}, "
              f"{float(lg.max() - lg[want]):.4f} below the top (before the cap)")
        return
    out = a.args[4]
    ids = [int(t) for t in ref["ids0"]]
    nll, top1 = np.zeros(len(ids) - 1), np.zeros(len(ids) - 1, np.int64)

    def on_rows(i0, lg):
        n = min(len(lg), len(ids) - 1 - i0)
        lg = G.softcap(spec, lg[:n])
        m = lg.max(1, keepdims=True)
        nll[i0:i0 + n] = m[:, 0] + np.log(np.exp(lg - m).sum(1)) - \
            lg[np.arange(n), ids[i0 + 1:i0 + 1 + n]]
        top1[i0:i0 + n] = lg.argmax(1)

    emulate(spec, W, ids, wformat=wf, hf=hf, quant=quant, wmap=wmap,
            rows=list(range(len(ids) - 1)), on_rows=on_rows)
    np.savez(out, nll=nll, top1=top1)
    print(f"{what}: {len(ids) - 1} positions, mean NLL {nll.mean():.4f}, ppl "
          f"{np.exp(nll.mean()):.3f}; {time.time() - t0:.0f} s")


if __name__ == "__main__":
    sys.exit(main())
