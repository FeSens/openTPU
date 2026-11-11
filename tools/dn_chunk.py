"""Chunked gated delta rule on the MXU (docs/dataflow.md 4.9.1: the accuracy study): formats_scan's
Qwen3.5 emulation with its DeltaNet recurrence in HF's chunked form
(transformers torch_chunk_gated_delta_rule), each of the chunk's eight matmuls on the MXU's int8
operands, against the sequential recurrence (DSTEP's, fp32: float64 here), by KL against the
float model (HF fp32's emulation) over a text.

Per chunk of C rows and head (q scaled, k l2-normed; g the log decays, beta the gates; S [dk, dv]
the state at the chunk's start, fp32 in DRAM):
    gc = cumsum(g); Gam[r, s] = exp(gc_r - gc_s)
    M1  L = (kb . k^T) * Gam, strictly lower           (d-contraction)
        T = (I + L)^-1                                  (VPU, fp32 forward substitution)
    M2  U0 = T . vb                                     (token contraction)
    M3  Wc = T . (kb * exp(gc))                         (token contraction)
    M5  U  = U0 - Wc . S                                (d-contraction, S's columns streamed)
    M4  A  = (q . k^T) * Gam, lower with the diagonal   (d-contraction)
    M6  O  = (q * exp(gc)) . S                          (d-contraction)
    M7     + A . U                                      (token contraction)
    M8  S' = exp(gc_C) S + (k * exp(gc_C - gc))^T . U   (token contraction; the add on the VPU)
MXU operands as the device takes them today: a d-contraction's left rows QACT (fp32 scale a row,
power-of-two offsets a D/4 group: quant.fake_act), its streamed rows int8 with one scale a row
(ROW: the KV cache's K, qwen3._fake_k at d = D); a token contraction's right operand int8 a token
(one scale a token, folded into the left rows before their QACT: P.V's V, qwen3._v_parts / _pv),
the left rows QACT over the chunk's tokens padded to D. MM's window drops (mx_matmul) included.

Modes (a variant's label ends in ";dn=MODE"; parse): cN chunks of N rows, nothing quantized
(the algorithm alone); cN-q<ids> those matmuls on the MXU's int8 operands, one pass; cN-h<ids>
those in three passes of operands split into their int8 part and the int8 of the remainder
(hi.hi + hi.lo + lo.hi: ~16-bit operands on today's MXU); cN-a<ids> / -w<ids> two passes, only the
activation's (QACT) / the streamed side's operand split; cN-s<ids> as h, the three terms one MM
SPL (Config.SPLMM, docs/isa.md "Split terms": the lo terms' offsets + 8, so an activation group
whose offset is past 8 drops from them; the sum rounded once, which float64 does not see); e.g.
c64-q568 the state's matmuls int8, c64-h12347q568 the intra-chunk ones split. A suffix -s<fmt> stores the state 16-bit (Config.ST16:
e6m9 or bf16, isasim.st16_pack) at each chunk's end, as its DRAM copy between chunks; "r[-s<fmt>]"
is the sequential recurrence with the state rounded so after every row, as DSTEP with ST16 stores
it (r-se6m9: production's bcu1525 :bhw16 prefill). A suffix -o takes the operands as the openTPU
kernel without new RTL does (tools/dn_cost.py): a token contraction's streamed side X^T (M2's V^T,
M3's and M8's K^T, M7's U^T) by an identity MM from X's rows quantized a token (int8, or split when
the matmul splits that side), then quantized a row of X^T (ROW), and M8 as St' = e^gl St + U^T Kd
[dv, dk] with U^T (from U's rows likewise, split when M8 splits its activation side) the activation.

    python tools/dn_chunk.py check
    python tools/dn_chunk.py ppl MODEL --formats F [--wformat fp4] [--tokens 2000]
                                 [--modes r-se6m9,c64-h12345678-se6m9,...] [--ref r-se6m9] [--out J]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))      # (formats_scan)
import formats_scan as FS  # noqa: E402
from opentpu import isa as I  # noqa: E402
from opentpu import isasim as IS  # noqa: E402
from opentpu import quant as Q  # noqa: E402
from opentpu.llm import qwen35 as Q35  # noqa: E402
from opentpu.llm.qwen3 import _v_parts  # noqa: E402

D = 128
W8 = I.MX_W[I.W8]


def _pad(x, K):
    """x padded with zeros to K columns, its fp32 subnormals flushed to zero (as the device's
    fp32 VPU leaves them: a long-decayed row's exp(gc) is under fp32's range)."""
    x = np.where(np.abs(x) < 2.0 ** -126, 0.0, x)
    n = x.shape[-1]
    if n == K:
        return x
    out = np.zeros(x.shape[:-1] + (K,))
    out[..., :n] = x
    return out


def _mm(a, b):
    """a @ b^T per head: a an MXAct [h, R, K], b [h, N, K] (offsets 0: ROW or UNIT), with MM's
    window drops (none unless a group's offset is past the window)."""
    y = np.einsum("hrk,hnk->hrn", np.asarray(a).view(np.ndarray), b)
    if a.da is not None and a.da.max() > W8:
        for h in range(len(b)):
            y[h] = Q.mx_matmul(a.rows(h), b[h], None, W8, D)
    return y


def _mm_d(a, b):
    """a [h, R, d] @ b [h, N, d]^T on the MXU: a's rows QACT, b's rows int8 ROW."""
    K = -(-a.shape[-1] // D) * D
    bq = np.asarray(Q.fake_act(_pad(b, K), D, row=True)).view(np.ndarray)
    return _mm(Q.fake_act(_pad(a, K), D), bq)


def _mm_t(a, x):
    """a [h, R, n] @ x [h, n, m] (a token contraction) on the MXU: x int8 a token, its scales
    folded into a's columns, a's rows QACT (padded to D)."""
    xq, xs = _v_parts(_pad(x, x.shape[-1]))
    K = -(-a.shape[-1] // D) * D
    return _mm(Q.fake_act(_pad(a * xs[:, None, :], K), D),
               _pad(np.swapaxes(xq, 1, 2), K))


def _hi_d(a, b):
    """The operands' int8 parts as _mm_d quantizes them (dequantized)."""
    K = -(-a.shape[-1] // D) * D
    return (np.asarray(Q.fake_act(_pad(a, K), D)).view(np.ndarray)[..., :a.shape[-1]],
            np.asarray(Q.fake_act(_pad(b, K), D, row=True)).view(np.ndarray)[..., :b.shape[-1]])


def _hi_t(a, x):
    """The operands' int8 parts as _mm_t quantizes them (x's a token; a's QACT rows with x's
    scales folded, divided out again)."""
    xq, xs = _v_parts(_pad(x, x.shape[-1]))
    K = -(-a.shape[-1] // D) * D
    sc = np.where(xs == 0, 1.0, xs)[:, None, :]
    ah = np.asarray(Q.fake_act(_pad(a * sc, K), D)).view(np.ndarray)[..., :a.shape[-1]] / sc
    return ah, xq * xs[..., None]


def _lo_keep(kind: str, a, b):
    """MM SPL's lo terms (how "s"): 1 for a's elements whose QACT group (the hi part's, as _mm_d /
    _mm_t quantize a) keeps them, its offset at most W - 8 (the lo terms' offsets are hi's + 8),
    else 0."""
    K = -(-a.shape[-1] // D) * D
    if kind == "d":
        qa = Q.fake_act(_pad(a, K), D)
    else:
        xs = _v_parts(_pad(b, b.shape[-1]))[1]
        qa = Q.fake_act(_pad(a * np.where(xs == 0, 1.0, xs)[:, None, :], K), D)
    return np.repeat(qa.da <= W8 - 8, D // 8, -1)[..., :a.shape[-1]]


def mm(kind: str, a, b, how):
    """kind "d": a @ b^T over d; "t": a @ b over the chunk's tokens. how None: exact; "q": the
    MXU's int8 operands (one pass); "h": each operand split into its int8 part and the int8 of
    its remainder, three passes (hi.hi + hi.lo + lo.hi: ~16-bit operands on today's MXU); "s":
    "h"'s terms as one MM SPL (the lo terms without a's groups past offset W - 8: _lo_keep)."""
    exact = (lambda x, y: np.einsum("hrk,hnk->hrn", x, y)) if kind == "d" else \
        (lambda x, y: np.einsum("hrn,hnm->hrm", x, y))
    if how is None:
        return exact(a, b)
    one = _mm_d if kind == "d" else _mm_t
    if how == "q":
        return one(a, b)
    ah, bh = (_hi_d if kind == "d" else _hi_t)(a, b)
    if how == "a":
        return one(ah, b) + one(a - ah, b)
    if how == "w":
        return one(a, bh) + one(a, b - bh)
    if how == "s":
        m = _lo_keep(kind, a, b)
        return one(ah, bh) + one(ah * m, b - bh) + one((a - ah) * m, bh)
    return one(ah, bh) + one(ah, b - bh) + one(a - ah, bh)


def _tq(x, split: bool):
    """Rows of x [h, n, d] as an identity MM transposes them: quantized a row (QST ROW: int8, one
    scale a row), split into the int8 part and the int8 of the remainder (two passes) or not."""
    hi = np.asarray(Q.fake_act(_pad(x, x.shape[-1]), D, row=True)).view(np.ndarray)
    if not split:
        return hi
    return hi + np.asarray(Q.fake_act(_pad(x - hi, x.shape[-1]), D, row=True)).view(np.ndarray)


def _sw(a):
    return np.ascontiguousarray(np.swapaxes(a, 1, 2))


def _solve(L):
    """(I + L)^-1 of strictly lower L [h, n, n] by forward substitution."""
    n = L.shape[-1]
    T = np.broadcast_to(np.eye(n), L.shape).copy()
    for i in range(1, n):
        T[:, i, :i] = -np.einsum("hs,hsj->hj", L[:, i, :i], T[:, :i, :i])
    return T


def parse(mode: str):
    """(C, {matmul id: how}, state format) of a mode "cN[-<how><ids>...][-s<fmt>]": N rows a
    chunk; each how ("q" one int8 pass, "h" three passes of split operands, "s" those as one MM
    SPL, "a" / "w" one side split) followed by the
    matmul ids (1-8, module doc) it applies to; the rest exact; fmt (e6m9, bf16) the state's
    16-bit store at a chunk's end, None fp32. E.g. c64-q568: the state's matmuls int8, the intra-
    chunk ones exact; c64-h12347q568-se6m9. "r[-s<fmt>]": C 0, the sequential recurrence."""
    import re
    m = re.fullmatch(r"(?:r|c(\d+)(?:-((?:[qhasw]\d+)+))?)(?:-s(e6m9|bf16))?(-o)?", mode)
    if m is None or (m.group(1) is None) != mode.startswith("r"):
        raise ValueError(f"mode {mode!r}")
    how = {}
    for h, ids in re.findall(r"([qhasw])(\d+)", m.group(2) or ""):
        how.update({int(i): h for i in ids})
    return int(m.group(1) or 0), how, m.group(3), bool(m.group(4))


def _st16(S, fmt):
    """S as its 16-bit store reads back (fmt None: S)."""
    if fmt is None:
        return S
    return IS.st16_f32(IS.st16_pack(S.astype(np.float32), fmt), fmt).astype(np.float64)


def recur_seq(q, k, v, g, beta, sfmt=None):
    """formats_scan's recurrence: o [T, h, dv] (sfmt: the state stored 16-bit after each row,
    the row's output from the fp32 state before the store, as DSTEP's)."""
    T, nh, dk = q.shape
    S, o = np.zeros((nh, dk, v.shape[2])), np.zeros((T, nh, v.shape[2]))
    for t in range(T):
        S = S * np.exp(g[t])[:, None, None]
        delta = (v[t] - np.einsum("hij,hi->hj", S, k[t])) * beta[t][:, None]
        S = S + k[t][:, :, None] * delta[:, None, :]
        o[t] = np.einsum("hij,hi->hj", S, q[t])
        S = _st16(S, sfmt)
    return o, S


def recur(q, k, v, g, beta, mode: str):
    """The recurrence of a mode (parse): o [T, h, dv] and the final state."""
    C, how, sfmt, _ = parse(mode)
    if C == 0:
        return recur_seq(q, k, v, g, beta, sfmt)
    return recur_chunk(q, k, v, g, beta, mode)


def recur_chunk(q, k, v, g, beta, mode: str, S0=None):
    """The chunked recurrence (module doc): o [T, h, dv] and the final state (S0: the initial
    state [h, dk, dv], else zeros)."""
    C, how, sfmt, ot = parse(mode)
    h = how.get
    sa = {i: how.get(i) in ("a", "h", "s") for i in range(1, 9)}    # (-o: which sides split)
    sw = {i: how.get(i) in ("w", "h", "s") for i in range(1, 9)}
    T, nh, dk = q.shape
    dv = v.shape[2]
    S = np.zeros((nh, dk, dv)) if S0 is None else np.array(S0, np.float64)
    o = np.zeros((T, nh, dv))
    for c0 in range(0, T, C):
        n = min(C, T - c0)
        tr = lambda a: np.ascontiguousarray(np.swapaxes(a[c0:c0 + n], 0, 1))   # noqa: E731
        Qc, Kc, Vc = tr(q), tr(k), tr(v)                    # [h, n, d]
        b, gc = beta[c0:c0 + n].T, np.cumsum(g[c0:c0 + n].T, -1)      # [h, n]
        low, lowd = np.tril(np.ones((n, n)), -1), np.tril(np.ones((n, n)))
        Gam = np.exp((gc[:, :, None] - gc[:, None, :]) * lowd)  # (HF's: the exponent's upper
        #                                       triangle zeroed first, as it would overflow)
        Kb = Kc * b[..., None]
        Tm = _solve(mm("d", Kb, Kc, h(1)) * Gam * low)                          # M1
        if ot:          # V^T, K^T [h, d, n] streamed: identity-MM transposes of their rows
            Vt = _sw(_tq(Vc, sw[2]) if h(2) else Vc)
            Kt = _sw(_tq(Kc, sw[3] or sw[8]) if h(3) or h(8) else Kc)
            U0 = mm("d", Tm * b[:, None, :], Vt, h(2))                          # M2
            Wc = mm("d", Tm * (b * np.exp(gc))[:, None, :], Kt, h(3))           # M3
        else:
            U0 = mm("t", Tm, Vc * b[..., None], h(2))                           # M2
            Wc = mm("t", Tm, Kb * np.exp(gc)[..., None], h(3))                  # M3
        St = np.ascontiguousarray(np.swapaxes(S, 1, 2))     # [h, dv, dk]: S's columns
        U = U0 - mm("d", Wc, St, h(5))                                          # M5
        A = mm("d", Qc, Kc, h(4)) * Gam * lowd                                  # M4
        O = mm("d", Qc * np.exp(gc)[..., None], St, h(6))                       # M6
        gl = gc[:, -1]
        if ot:          # U^T [h, dv, n]: M7's streamed side, M8's activation rows
            Ut = _sw(_tq(U, sw[7] or sa[8]) if h(7) or h(8) else U)
            O = O + mm("d", A, Ut, h(7))                                        # M7
            St = np.exp(gl)[:, None, None] * St + mm(                           # M8
                "d", Ut * np.exp(gl[:, None] - gc)[:, None, :], Kt, h(8))
            S = _st16(np.swapaxes(St, 1, 2), sfmt)
        else:
            O = O + mm("t", A, U, h(7))                                         # M7
            Kd = Kc * np.exp(gl[:, None] - gc)[..., None]
            S = _st16(np.exp(gl)[:, None, None] * S + mm(
                "t", np.ascontiguousarray(np.swapaxes(Kd, 1, 2)), U, h(8)), sfmt)  # M8
        o[c0:c0 + n] = np.swapaxes(O, 0, 1)
    return o, S


def _mode() -> str:
    lab = FS._AT.get("lab") or ""
    return lab.partition(";dn=")[2]


def _qwen35(spec, W, i, x, w, fq, c, s, D, wf):
    """formats_scan._qwen35 with the recurrence of the running variant's mode (_mode)."""
    T, d, eps, K = len(x), spec.head_dim, spec.eps, spec.conv_k
    if spec.kinds[i] != Q35.LIN or not _mode():
        return _ORIG(spec, W, i, x, w, fq, c, s, D, wf)
    nh, nk, dk, dv = spec.lin_heads, spec.lin_nk, spec.lin_dk, spec.lin_dv
    og = 4 if nh % 4 == 0 else 2
    outr = [og * dv] * (nh // og)
    if og == 4 and nh > 2:
        outr[-1:] = [2 * dv, 2 * dv]
    p = f"model.layers.{i}."

    def g1(n):
        return 1 + np.asarray(W[n], np.float64)

    h = fq(Q35._norm(x, g1(p + "input_layernorm.weight"), eps), D)
    a = p + "linear_attn."
    qkv = h @ w(a + "in_proj_qkv.weight").T
    win = np.concatenate([np.zeros((K - 1, qkv.shape[1])), qkv])
    wc = W[a + "conv1d.weight"][:, 0, :]
    y = Q35._silu(sum(win[j:j + T] * wc[:, j] for j in range(K)))
    q, k, v = np.split(y, [nk * dk, 2 * nk * dk], -1)
    q = np.repeat(Q35._l2norm(q.reshape(T, nk, dk)) / math.sqrt(dk), nh // nk, axis=1)
    k = np.repeat(Q35._l2norm(k.reshape(T, nk, dk)), nh // nk, axis=1)
    v = v.reshape(T, nh, dv)
    z = (h @ w(a + "in_proj_z.weight").T).reshape(T, nh, dv)
    beta = 1 / (1 + np.exp(-(h @ w(a + "in_proj_b.weight").T)))
    g = -np.exp(np.asarray(W[a + "A_log"], np.float64)) * Q35._softplus(
        h @ w(a + "in_proj_a.weight").T + W[a + "dt_bias"])
    o, _ = recur(q, k, v, g, beta, _mode())
    o = Q35._norm(o, np.asarray(W[a + "norm.weight"], np.float64), eps) * Q35._silu(z)
    x = x + fq(o.reshape(T, -1), D, outr) @ w(a + "out_proj.weight").T
    h = fq(Q35._norm(x, g1(p + "post_attention_layernorm.weight"), eps), D)
    mp = spec.mlp_prefix(p)
    gg, u = h @ w(mp + "gate_proj.weight").T, h @ w(mp + "up_proj.weight").T
    dn = mp + "down_proj.weight"
    y = fq(Q35._silu(gg) * u, D, FS._down(spec, wf, dn, D)) @ w(dn).T
    if spec.moe is not None:
        return FS.MoEIn(x, h, y)
    return x + y


_ORIG = FS._qwen35
FS.LAYER[Q35] = _qwen35


def check():
    """The chunked recurrence against the sequential one on random gates and rows (the 35B's
    head shape): unquantized to rounding; quantized, the outputs' relative error."""
    rng = np.random.default_rng(3)
    T, nh, d = 200, 4, 128
    q = rng.standard_normal((T, nh, d))
    q /= np.linalg.norm(q, axis=-1, keepdims=True) * math.sqrt(d)
    k = rng.standard_normal((T, nh, d))
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    v = rng.standard_normal((T, nh, d))
    g = -np.exp(rng.uniform(-4, 0.5, (T, nh)))
    g[:, 0] = -np.exp(rng.uniform(2, 6, T))     # (a head that forgets at once: exp(-gc) overflows)
    beta = 1 / (1 + np.exp(-rng.standard_normal((T, nh))))
    o0, S0 = recur_seq(q, k, v, g, beta)
    ok = True
    for m in ("r", "c16", "c32", "c64", "c128", "c7", "c16-q12345678", "c64-q12345678",
              "c64-q12347", "c64-q568", "c64-q1", "c64-q23", "c64-q47", "c64-h12345678",
              "c64-h12347q568", "c16-h12345678", "r-se6m9", "r-sbf16", "c64-se6m9",
              "c64-h12345678-se6m9", "c64-h12345678-sbf16", "c128-h12345678-se6m9",
              "c64-q1h2345678-se6m9", "c64-o", "c64-h12345678-o", "c64-h12345678-se6m9-o",
              "c64-a12345678-se6m9", "c64-w12345678-se6m9", "c64-a2347h1568-se6m9-o",
              "c64-q14a2347h568-se6m9-o", "c64-q14a237h568-se6m9-o", "c64-q1s2345678-se6m9-o",
              "c64-s12345678"):
        o, S = recur(q, k, v, g, beta, m)
        e = np.abs(o - o0).max() / np.abs(o0).max()
        es = np.abs(S - S0).max() / np.abs(S0).max()
        rel = np.linalg.norm(o - o0) / np.linalg.norm(o0)
        print(f"{m:20s} max|do|/max|o| {e:.3e}  ||do||/||o|| {rel:.3e}  max|dS|/max|S| {es:.3e}")
        if not np.isfinite(o).all() or m.rstrip("-o").count("-") == 0 and (e > 1e-9 or es > 1e-9):
            ok = False
    print("check", "ok" if ok else "FAILED")
    return 0 if ok else 1


def ppl(model, formats: str, wformat: str, n_tok: int, modes: list, out=None, ref=None):
    """KL to float of the formats' row (the fp64 sequential recurrence) and of each mode's,
    dKL paired against the formats' row ("vs prod") and against mode ref's ("vs ref", e.g.
    r-se6m9: production's 16-bit state)."""
    path = FS.model_dir(model)
    spec, run, nbytes, _, _ = FS._open(path, D)
    ids = FS.text_ids(path, n_tok)
    t0 = time.time()
    specs = {"float": None, "int8": ("", "int8", None), formats: (formats, wformat, None)}
    specs.update({f"{formats};dn={m}": (formats, wformat, None) for m in modes})
    rows = run(specs, ids, lambda m: print(f"[{time.time() - t0:6.0f} s] {m}", flush=True))
    FS._finish(rows, nbytes)
    base = np.asarray(rows[formats]["kl_tok"])
    rb = np.asarray(rows[f"{formats};dn={ref}"]["kl_tok"]) if ref else None
    for r in rows.values():
        dk = np.asarray(r["kl_tok"]) - base
        r["dkl_prod"], r["se_prod"] = float(dk.mean()), float(dk.std(ddof=1) / math.sqrt(len(dk)))
        if rb is not None:
            dk = np.asarray(r["kl_tok"]) - rb
            r["dkl_ref"], r["se_ref"] = float(dk.mean()), float(dk.std(ddof=1) / math.sqrt(len(dk)))
    if out:
        Path(out).write_text(json.dumps({"model": str(model), "tokens": len(ids),
                                         "formats": formats, "wformat": wformat, "ref": ref,
                                         "variants": rows}, indent=1))
    for lab, r in rows.items():
        vs = f"  vs {ref} {100 * r['dkl_ref']:+.4f}% (+-{100 * r['se_ref']:.4f})" \
            if "dkl_ref" in r else ""
        print(f"{lab:60s} ppl {r['ppl']:8.4f}  KL/float {100 * r['kl_float']:.3f}%  dKL "
              f"{100 * r['dkl']:+.3f}% (+-{100 * r['se_dkl']:.3f})  vs prod "
              f"{100 * r['dkl_prod']:+.4f}% (+-{100 * r['se_prod']:.4f}){vs}  top1/int8 "
              f"{r['agree_int8']:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    a = sub.add_parser("ppl")
    a.add_argument("model")
    a.add_argument("--formats", required=True)
    a.add_argument("--wformat", default="fp4")
    a.add_argument("--tokens", type=int, default=2000)
    a.add_argument("--modes", default="r-se6m9,c64-h12345678,c64-h12345678-se6m9")
    a.add_argument("--ref", default="r-se6m9", help="a mode of --modes to pair against too "
                   "(empty: none)")
    a.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "check":
        return check()
    modes = a.modes.split(",")
    ppl(a.model, a.formats, a.wformat, a.tokens, modes, a.out, a.ref if a.ref in modes else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
