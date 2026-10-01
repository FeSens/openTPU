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
26B-A4B (its MoE block, K = V global layers, no PLE): --formats experts=fp4 for 4-bit experts
(about 1.2 s of quantization an expert, 3840 of them: THREADS at a time).
"""
import argparse
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np

from opentpu.kernels import gather as GA
from opentpu.llm import gemma4 as G
from opentpu.llm.qwen3 import _fake_q, _fake_w

THREADS = 4     # experts quantized at once (numpy's 4-bit search releases the GIL: 2.8x on 4)


def emulate(spec, W, tokens, D=128, wformat="int8", hf=None, ple_format="int8", rows=None,
            quant=None, wmap=None, on_rows=None):
    """gemma4.emulated_logits over the whole sequence at once: the logits [len(rows), vocab]
    before the soft cap (rows: default all), or with on_rows each group of up to 64 rows handed
    to on_rows(first row, logits). wformat / hf: the layers' and the head's formats, "none" for
    float weights (the PLE table then float too); wmap: per-kind layer formats {"attn", "mlp"
    (or "down" / "gateup" of it), "ple": format}, a key "kind@a-b" for layers a..b only
    (checkpoint layers; it wins over "kind"). quant: the activation quantization points
    applied, a subset of {"act" (the matmul inputs), "kv" (K / V), "p" (P)}; default all, none
    with wformat "none".

    A MoE block (26B-A4B) as the device runs it (docs/offload.md section 11): one activation,
    the weightless RMSNorm of x quantized, into the router (int8, router.scale / sqrt(H) folded
    into its columns; the top k of its logits, their softmax) and the experts (wmap "experts",
    else wformat; pre_feedforward_layernorm_2's gain folded into gate / up, per_expert_scale
    into the down projection, the width padded with zeros to whole D-blocks)."""
    hf = hf or wformat
    none = wformat == "none"
    quant = (set() if none else {"act", "kv", "p"}) if quant is None else set(quant)
    ident = lambda v, d=D: np.asarray(v, np.float64)
    fq = _fake_q if "act" in quant else ident
    fk = _fake_q if "kv" in quant else ident
    fp = _fake_q if "p" in quant else ident

    wmap = wmap or {}                   # per-kind layer formats: attn, mlp, ple (else wformat)

    def get(k, li, default):    # wmap[k] ("kind" or "kind@a-b": layers a..b only)
        for key, f in wmap.items():
            kind, _, span = key.partition("@")
            if kind == k and span:
                lo, _, hi = span.partition("-")
                if int(lo) <= li <= int(hi or lo):
                    return f
        return wmap.get(k, default)

    def fmt_of(n):              # attn; mlp, or down / gateup within it; ple
        li = int(n.split(".")[2])
        if ".experts." in n:
            return get("experts", li, wformat)
        if ".self_attn." in n:
            return get("attn", li, wformat)
        if ".mlp." in n:
            return get("down" if ".down_proj" in n else "gateup", li, get("mlp", li, wformat))
        return get("ple", li, wformat)

    def wq(n, fmt=wformat, a=None):
        if n is not None and fmt == wformat and n.startswith("model.layers."):
            fmt = fmt_of(n)
        a = np.asarray(W[n] if a is None else a, np.float32)
        a = np.pad(a, ((0, 0), (0, -a.shape[1] % D)))       # whole D-blocks (MLP widths)
        return a.astype(np.float64) if fmt == "none" else _fake_w(a, D, fmt)

    def padq(v, n):             # an activation [T, f] padded with zeros to n columns, quantized
        return fq(np.pad(v, ((0, 0), (0, n - v.shape[1]))))

    T, H, P, L = len(tokens), spec.hidden, spec.ple_dim, spec.layers
    n_q, eps = spec.n_q, spec.eps
    uniq, inv = np.unique(np.asarray(tokens), return_inverse=True)
    head = "model.embed_tokens.weight" if spec.tied else "lm_head.weight"
    x = wq(head, hf, G._rows(W, head, uniq.tolist()))[inv] * math.sqrt(H)
    pli = None
    if P:
        cols = np.concatenate([np.arange(spec.src(i) * P, (spec.src(i) + 1) * P)
                               for i in range(L)])
        pe = np.asarray(G._rows(W, G.Weights.PLE, uniq.tolist()), np.float32)[:, cols] * \
            np.float32((P / 2) ** 0.5)
        if not none:
            S = GA.record_blocks(-(-L * P // D), ple_format)
            pe = GA.dequant_records(GA.pack_records(pe, ple_format, D, S), ple_format, D,
                                    S)[:, :L * P]
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
        kind, d, n_kv = spec.kinds[i], spec.hd(i), spec.kvh(i)
        Gq = n_q // n_kv
        tab = [G.rope_tables(spec, pos, kind) for pos in range(T)]
        c = np.stack([t[0] for t in tab])[:, None, :]
        s_ = np.stack([t[1] for t in tab])[:, None, :]
        h = fq(G._norm(x, W[p + "input_layernorm.weight"], eps))
        q = (h @ wq(a + "q_proj.weight").T).reshape(T, n_q, d)
        q = fq(G._rot(G._norm(q, W[a + "q_norm.weight"], eps), c, s_, d // 2))
        if spec.kv_src[i] == i:
            k = (h @ wq(a + "k_proj.weight").T).reshape(T, n_kv, d)
            v = k if spec.kv_same(i) else (h @ wq(a + "v_proj.weight").T).reshape(T, n_kv, d)
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
        wd = wq(p + "mlp.down_proj.weight")
        m = padq(G._gelu(h @ wq(p + "mlp.gate_proj.weight").T) *
                 (h @ wq(p + "mlp.up_proj.weight").T), wd.shape[1]) @ wd.T
        del wd
        if spec.experts:
            m = G._norm(m, W[p + "post_feedforward_layernorm_1.weight"], eps) + \
                G._norm(_moe(spec, W, p, x, wq, fq, padq, none), W[
                    p + "post_feedforward_layernorm_2.weight"], eps)
        x = x + G._norm(m, W[p + "post_feedforward_layernorm.weight"], eps)
        if pli is not None:
            g = G._gelu(fq(x) @ wq(p + "per_layer_input_gate.weight").T) * pli[:, i]
            y = fq(g) @ wq(p + "per_layer_projection.weight").T
            x = x + G._norm(y, W[p + "post_per_layer_input_norm.weight"], eps)
        x = x * np.asarray(W[p + "layer_scalar"], np.float64)
    rows = list(range(T)) if rows is None else rows
    xl = fq(G._norm(x[rows], W["model.norm.weight"], eps))
    E = W[head]
    if on_rows is None:
        return np.concatenate([xl @ wq(None, hf, E[r0:r0 + 16384]).T
                               for r0 in range(0, len(E), 16384)], 1)
    # the quantized head held in float32 (E4B's in float64 would be 5.4 GB)
    Eq = [wq(None, hf, E[r0:r0 + 16384]).T.astype(np.float32) for r0 in range(0, len(E), 16384)]
    del E
    for i0 in range(0, len(rows), 64):      # on_rows(first row, logits [<= 64, vocab])
        on_rows(i0, np.concatenate([xl[i0:i0 + 64] @ e for e in Eq], 1))


def _moe(spec, W, p, x, wq, fq, padq, none):
    """The MoE block of layer prefix p on rows x [T, H] (emulate's docstring), float64."""
    r, F = p + "router.", spec.expert_ffn
    xs = fq(G._norm(x, None, spec.eps))
    wr = np.asarray(W[r + "proj.weight"], np.float32) * (
        np.asarray(W[r + "scale"], np.float32) * np.float32(spec.hidden ** -0.5))[None, :]
    lg = xs @ wq(None, "none" if none else "int8", a=wr).T                  # [T, E]
    top = np.argsort(-lg, axis=1, kind="stable")[:, :spec.top_k]
    w = np.take_along_axis(lg, top, 1)
    w = np.exp(w - w[:, :1])
    w /= w.sum(1, keepdims=True)
    g2 = np.asarray(W[p + "pre_feedforward_layernorm_2.weight"], np.float32)[None, :]
    pes = np.asarray(W[r + "per_expert_scale"], np.float32)
    n = p + "experts.down_proj"                 # (its name: the experts' format, wmap)
    es = [int(e) for e in np.unique(top)]

    def fake(gu, dn, e):        # expert e's gate, up, down [H, F padded] as the device holds them
        return wq(n, a=gu[:F]), wq(n, a=gu[F:]), wq(n, a=dn * pes[e])

    out = np.zeros_like(xs)
    with ThreadPoolExecutor(THREADS) as ex:    # THREADS experts at a time (their fp64 copies)
        for i in range(0, len(es), THREADS):
            batch = es[i:i + THREADS]
            jobs = [ex.submit(fake, G._rows(W, p + "experts.gate_up_proj", [e])[0] * g2,
                              G._rows(W, n, [e])[0], e) for e in batch]
            for e, job in zip(batch, jobs):
                gq, uq, dq = job.result()
                t, j = np.nonzero(top == e)
                u = G._gelu(xs[t] @ gq.T) * (xs[t] @ uq.T)
                out[t] += w[t, j][:, None] * (padq(u, dq.shape[1]) @ dq.T)
    return out


def top(spec, tok, lg, n=5):
    cap = G.softcap(spec, lg)
    t = np.argsort(-lg)[:n]
    return [(int(j), tok.decode([int(j)]), round(float(lg[j]), 4), round(float(cap[j]), 4)) for j in t]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=["step", "nll", "check"])
    ap.add_argument("args", nargs="+")
    ap.add_argument("--formats", default="", help="per-kind layer formats: attn, mlp (or down, "
                    "gateup), ple, experts, e.g. attn=int8,down=int8; kind@a-b: layers a..b "
                    "only (gateup@0-6=fp4)")
    ap.add_argument("--quant", default=None, help="the activation points (default all): act,kv,p")
    ap.add_argument("--out", help="step: save the logits (before the cap) to this npz")
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
    pts = a.quant if a.quant is not None else "none" if wf == "none" else "all"
    what = f"{wf} layers{' ' + str(wmap) if wmap else ''}, {hf} head, points {pts or 'none'}"
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
        if a.out:
            np.savez(a.out, logits=lg)
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
