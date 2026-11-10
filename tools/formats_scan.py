"""Weight formats per kind and layer range (opentpu/llm/formats.py) for the Qwen3 / Llama-like
(SmolLM3, Phi-4-mini), LFM2, Qwen3.5 and Gemma 4 models: perplexity over the first --tokens
tokens of docs/isa.md (quant_eval.py's text: prose the models have not memorized) in float64
emulations of the device's quantization. emulate() is the families' emulated_logits with the
loops swapped: layers outer, the whole sequence and every variant of a pass at once, so each
layer's weights are quantized once per format (4-bit: through the image cache,
opentpu/qcache.py) and only one layer's weights are held. Gemma 4 (E2B) runs
tools/gemma4_quant_eval.py's emulation a variant at a time (its KV-shared layers read earlier
layers' K / V) up to the head's input, which --cache keeps, then one pass over the soft-capped
head for every variant; its kinds are attn, gateup, down and ple (a layer's PLE gate and
projection; unranged, the PLE model projection too), its groups the own-K / V layers and the
shared ones, each halved at a multiple of the attention pattern (E2B: 0-9, 10-14, 15-24,
25-34), and the head is not in the greedy order (head=fp4 a row of its own). A variant's PLE
table takes the format its image does at otpu-chat's 2048 tokens (int8 where it fits the card,
else fp4: E2B's int8 layers). Gemma 4 26B-A4B (no PLE: kinds attn, gateup, down; groups 0-11
and 12-29): its dense layers' formats on the card's experts and head, --base experts=fp4,
head=fp4 (the experts are one format, no kind here; a row's bytes are the dense layers' and the
head's, the experts' slots aside); each variant resumes after its last layer done (emulate's
ckpt).

    python tools/formats_scan.py scan MODEL OUT.json [--tokens 900] [--fmt fp4] [--base F]
                                  [--ids IDS.json]
        pass 1: float weights (no quantization at all), int8, the 4-bit image (WF --fmt, head
        int8), and int8 with one group of weights in --fmt: a kind in all layers, a kind in a
        quarter of them, the head. pass 2: the uniform mixes (each kind one format in every
        layer: what a model with one layer layout runs) and the quarter groups and the head
        added in order of dKL per byte saved (cumulative; the scans of docs/formats.md before
        Gemma 4's: NLL). OUT.json: every variant's
        formats string, mean NLL, perplexity, weight bytes per token, top-1 agreement and its
        KL divergence from float's next-token distribution beyond int8's (dKL: the log
        perplexity ratio to int8 where the float model is calibrated, without the noise of the
        sampled tokens), each with its standard error paired over the tokens. --base F: the
        formats every row starts from, the int8 row's own (no head row where F sets the head)
    python tools/formats_scan.py ppl MODEL FORMATS... [--wformat int8] [--tokens 900] [--out J]
                                 [--cache DIR] [--base F] [--ids IDS.json]
        the perplexity and dKL of formats strings (as OTPU_FORMATS; "" for none), each with the
        standard error of its difference from int8's (paired over the tokens); --ids: another
        text's token ids (tools/chat_text.py: a chat of an instruct model's own replies)
    --S N (scan and ppl): the device's QACT ranges on N slices (the down projections' F chunks
    of F / N; Qwen3.5's out_proj per slice's head groups), as emulated_logits(S=N): Qwen3.5-27B
    on bcu1525-x2's 4 slices. --target T (scan and ppl): T's slices and its S-exact choice, as
    emulated_logits(S=T's S, cfg=T's Config): an S-exact image (Config.SEXACT, bcu1525-s2's;
    OTPU_SEXACT overrides it, as for the device) takes S = 1's ranges for Qwen3.5's out_proj
    and dense down projections (qwen35.s_exact, mlp_s_exact)
    python tools/formats_scan.py check
        tiny random models of each family (Qwen3 with K outliers, SmolLM3, Phi-3, Qwen3.5
        with 4 key heads, LFM2, LFM2-MoE, Qwen3.5-MoE):
        emulate() against emulated_logits (formats including ranged ones) and the NLL of the
        chunked head against the full logits (Gemma 4's: tests/test_gemma4.py)

MODEL: a short name of opentpu.llm.MODELS or a checkpoint directory. A dense model, an LFM2-MoE
or Qwen3.5-MoE (_moe_ffn: every expert the tokens route to, each variant through its own
router; the experts in the wformat, e.g. ppl qwen35-35b attn=nvfp4,delta=nvfp4,mlp=nvfp4,head=int8
--wformat fp4), or Gemma 4's MoE with --base. --cache DIR (Gemma 4; scan: OUT.json's name +
.xl): each variant's rows into the head, by its tokens, formats and the emulation's sources.
"""
import argparse
import functools
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

from opentpu import isa as I
from opentpu import qcache as QC
from opentpu import quant as Q
from opentpu.kernels.mlp import _chunk, down_chunk
from opentpu.llm import formats as FM
from opentpu.llm import gemma4 as G
from opentpu.llm import lfm2 as L2
from opentpu.llm import load_spec, model_dir
from opentpu.llm import qwen3 as Q3
from opentpu.llm import qwen35 as Q35
from opentpu.llm.qwen3 import ATTN_BLOCK, _fake_k, _fake_q, _qk, _v_parts, rope_tables

ROOT = Path(__file__).resolve().parent.parent
HEAD_ROWS = 16384       # LM head rows per pass (quantize_mxu's pieces: the same quantization)
SLICES = [1]            # the slices whose QACT ranges the emulation takes (--S)
CFG = [None]            # the Config whose S-exact choice it follows (--target; qwen35.s_exact_on)


def family(spec):
    fams = {Q3.Spec: Q3, L2.Spec: L2, Q35.Spec: Q35}    # (a subclass: its base's, MiniCPM-V's
    return next(fams[t] for t in type(spec).__mro__ if t in fams)   # Spec qwen35's)


def _ident(v, d=None, n=None):
    return v


DROPS: dict = {}        # {label: {MM kind: what MM's window dropped}} (ppl: _counted_mm)
_AT = {"lab": None, "rows": None, "T": 0, "kind": None}  # the variant running; its rows' tokens
_MX_MATMUL = Q.mx_matmul


def _counted_mm(h, w, dw=None, W=16, D=128):
    """quant.mx_matmul, counting for the variant running (_AT) what the window drops, per MM
    kind ("w": a layer's weights, a MoE's router and experts; "qk": q.K^T; "pv": P.V^T; "head"):
    the MMs (and those with a drop), the (row, D/8 group) inputs and the (row, output, group)
    terms (and those dropped), and the dropped terms per token (each row a token;
    _AT["rows"]: the rows' tokens, a MoE expert's)."""
    y = _MX_MATMUL(h, w, dw, W, D)
    lab, da = _AT["lab"], getattr(h, "da", None)
    if lab is None or da is None:
        return y
    fn = sys._getframe(1).f_code.co_name        # (MXW's matmul, qwen3._qk, or an attention's P.V)
    kind = _AT["kind"] or {"__array_ufunc__": "w", "_qk": "qk"}.get(fn, "pv")
    N, K, G = w.shape[0], w.shape[1], D // Q.NVG
    dwa = np.zeros((1, K // G), np.int64) if dw is None else np.asarray(dw, np.int64)
    dg = np.asarray(da).reshape(-1, dwa.shape[1])
    R, T = len(dg), _AT["T"]
    c = DROPS.setdefault(lab, {}).setdefault(kind, {
        "mm": 0, "mm_drop": 0, "groups": 0, "groups_drop": 0, "terms": 0, "terms_drop": 0,
        "tok": np.zeros(T, np.int64)})
    c["mm"] += 1
    c["groups"] += R * dwa.shape[1]
    c["terms"] += R * N * dwa.shape[1]
    per = np.zeros(R, np.int64)
    dwmax = dwa.max(0)
    for gi in np.nonzero(dg.max(0) + dwmax > W)[0]:
        rows = np.nonzero(dg[:, gi] + dwmax[gi] > W)[0]
        n = ((dg[rows, gi][:, None] + dwa[None, :, gi]) > W).sum(1) * (N // len(dwa))
        per[rows] += n
        c["groups_drop"] += int((n > 0).sum())
    if per.any():
        c["mm_drop"] += 1
        c["terms_drop"] += int(per.sum())
        if _AT["rows"] is not None:
            np.add.at(c["tok"], _AT["rows"], per)
        elif T and R % T == 0:
            c["tok"] += per.reshape(T, -1).sum(1)
    return y


def dequant(a, fmt, D):
    """A weight as the device holds it, in float64 (None: unquantized): an MXW (quant.fake_w:
    its values and groups' offsets, MM's sums with an MXAct), the 4-bit formats' quantization
    through the image cache (opentpu.qcache)."""
    if fmt is None:
        return np.asarray(a, np.float64)
    a = np.asarray(a, np.float32)
    return Q.fake_w(a, fmt, D, QC.quantize_mxu(a, fmt, D) if fmt in QC.FORMATS else None)


def _down_slices(spec, f, D) -> int:
    """The slices whose F chunks a down projection in format f takes (qwen35.emulated_logits'):
    S = 1's for a Qwen3.5-MoE's shared expert (moe._moe_ffn_tp) and an S-exact image's dense
    MLP (qwen35.mlp_s_exact, CFG), else SLICES[0]."""
    S = SLICES[0]
    if isinstance(spec, Q35.Spec) and (spec.moe is not None
                                       or Q35.mlp_s_exact(spec, S, D, f, CFG[0])):
        return 1
    return S


def _down(spec, wf, name, D):
    """The F chunk of a down projection's input (kernels.mlp.down_chunk, a slice's F), None
    unquantized."""
    f = wf(name)
    if f is None:
        return None
    F = spec.ffn // _down_slices(spec, f, D)
    if isinstance(spec, L2.Spec):               # (lfm2.emulated_logits': LFM2-2.6B at S 4)
        return _chunk(F, D, L2._down_q(f, F, _down_slices(spec, f, D), D))
    return down_chunk(F, D, f)


def _fq_down(fq, a, spec, wf, name, D):
    """A down projection's input `a` quantized as its QACTs on _down_slices' slices (chunk c of
    every slice's F one range: quant.fake_act_gathered), or fq's (_ident: unquantized)."""
    C = _down(spec, wf, name, D)
    return fq(a, D, C) if fq is _ident or C is None else \
        Q.fake_act_gathered(a, D, C, _down_slices(spec, wf(name), D))


# The KV cache's format under test (docs/dataflow.md 4.8, a proposal): "" the device's (int8 K
# with a per-token scale, int8 V with a per-token scale); a variant whose label ends in ";kv=MODE"
# runs with MODE (emulate sets it per variant), parts joined by "+":
#   fp4k / nvfp4k   K as the MX column streams that weight format (quant.fake_w over the K rows:
#                   one Sw a token, the format's block scales along d)
#   fp4v / int4v    V as E2M1 (max |v| / 6) or int4 (max |v| / 7) values with one scale a token,
#                   folded into P as today's V scale (UNIT; the 4-bit windows): token-major V
#   fp4vt / nvfp4vt V^T as weight rows of that format, an attention block of tokens a row (one
#                   Sw a channel and block, the format's block scales along the tokens): P.V's
#                   streamed operand, the format's own use
#   int6k / fp6k    6-bit K (a proposal: no MXU format yet): int6 (+-31, one Sw a token and a
#                   power-of-two offset a D-block, as int8x's with 31) or MXFP6 (E2M3 values, an
#                   E8M0 power of two a 32-element block along d)
#   int6v / fp6vt   6-bit V: token-major int6 with one scale a token (folded into P as today's),
#                   or V^T rows in MXFP6 (the E8M0 blocks along the tokens)
# fp4kv / nvfp4kv: fp4k / nvfp4k + fp4v.
_KV = [""]
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _kv_mode(label: str) -> str:
    return label.partition(";kv=")[2]


def _v_e2m1(v):
    """V rows [..., d] as E2M1 values (round to the nearest grid point, ties to the even code)
    and one scale a row (max |v| / 6; an all-zero row's 0)."""
    a = np.abs(v).max(-1, keepdims=True)
    x = np.abs(v) / np.where(a == 0, 1, a / 6)
    dist = np.abs(x[..., None] - E2M1)
    best = dist.min(-1, keepdims=True)
    cand = np.where(np.isclose(dist, best, rtol=0, atol=1e-12), np.arange(8), 99)
    even = np.where(cand % 2 == 0, cand, 99).min(-1)
    idx = np.where(even < 99, even, cand.min(-1))
    return np.sign(v) * E2M1[idx], a[..., 0] / 6


def _v_int4(v):
    """V rows [..., d] as int4 values (-7 .. 7) and one scale a row (max |v| / 7)."""
    a = np.abs(v).max(-1, keepdims=True)
    return np.clip(np.rint(v / np.where(a == 0, 1, a / 7)), -7, 7), a[..., 0] / 7


E2M3 = np.concatenate([np.arange(8) / 8, 1 + np.arange(8) / 8, 2 + np.arange(8) / 4,
                       4 + np.arange(8) / 2])                   # 0 .. 7.5


def _int6_rows(x, D: int):
    """Rows [..., n] in int6 as int8x's quantizer with 31 levels: Sw = amax / 31 a row, each
    D-block's offset d the largest (<= 31) with the block's amax 2^d <= the row's; dequantized."""
    x = np.asarray(x, np.float64)
    n = x.shape[-1]
    B = min(D, n)
    xb = x.reshape(*x.shape[:-1], n // B, B)
    a = np.abs(x).max(-1, keepdims=True)[..., None]
    bm = np.abs(xb).max(-1, keepdims=True)
    with np.errstate(divide="ignore"):
        d = np.where(bm > 0, np.clip(np.floor(np.log2(a / np.where(bm > 0, bm, 1))), 0, 31), 0)
    u = np.where(a == 0, 1, a / 31) * np.exp2(-d)
    return (np.clip(np.rint(xb / u), -31, 31) * u).reshape(x.shape)


def _mxfp6_rows(x, B: int = 32):
    """Rows [..., n] in MXFP6 (OCP MX: E2M3 elements, an E8M0 scale a block of B along the row,
    2^(floor(log2 amax) - 2)); values to the nearest E2M3 point (ties to the even code), 7.5 at
    most; dequantized."""
    x = np.asarray(x, np.float64)
    n = x.shape[-1]
    xb = x.reshape(*x.shape[:-1], n // B, B)
    bm = np.abs(xb).max(-1, keepdims=True)
    with np.errstate(divide="ignore"):
        sc = np.where(bm > 0, np.exp2(np.floor(np.log2(np.where(bm > 0, bm, 1))) - 2), 1)
    y = np.minimum(np.abs(xb) / sc, 7.5)
    dist = np.abs(y[..., None] - E2M3)
    best = dist.min(-1, keepdims=True)
    cand = np.where(np.isclose(dist, best, rtol=0, atol=1e-12), np.arange(len(E2M3)), 999)
    even = np.where(cand % 2 == 0, cand, 999).min(-1)
    idx = np.where(even < 999, even, cand.min(-1))
    return (np.sign(xb) * E2M3[idx] * sc).reshape(x.shape)


def _kv_parts(kv: str, k, v, D: int) -> tuple:
    """(Kw, Vp, Vw, window) of KV format kv (_KV) over K and V [T, n_kv, d]: Kw [n_kv][T, d], K
    as the MX column streams that weight format (None: the device's int8 K, _fake_k); Vp (V,
    its per-token scales) in a 4-bit token-major format (None: int8, _v_parts); Vw [n_kv][d,
    Tb] V^T as weight rows (None: V per token); P.V's weight window."""
    parts = {"fp4kv": {"fp4k", "fp4v"}, "nvfp4kv": {"nvfp4k", "fp4v"}}.get(
        kv, set(kv.split("+")) if kv else set())
    Kw, Vp, Vw, win = None, None, None, I.MX_W[I.W8]
    T, n_kv, d = v.shape
    Tb = -(-T // ATTN_BLOCK) * ATTN_BLOCK
    for f in ("fp4", "nvfp4"):
        if f + "k" in parts:
            Kw = [Q.fake_w(np.asarray(k[:, h], np.float32), f, D) for h in range(n_kv)]
        if f + "vt" in parts:
            vp = np.zeros((Tb, n_kv, d), np.float32)
            vp[:T] = v
            Vw = [np.concatenate([np.asarray(Q.fake_w(np.ascontiguousarray(
                vp[t0:t0 + ATTN_BLOCK, h].T), f, D)) for t0 in range(0, Tb, ATTN_BLOCK)], 1)
                for h in range(n_kv)]
            win = I.MX_W[I.WFORMATS[f]]
    if "fp4v" in parts:
        Vp, win = _v_e2m1(v), I.MX_W[I.W4F]
    if "int4v" in parts:
        Vp, win = _v_int4(v), I.MX_W[I.W4I]
    if "int6k" in parts:
        Kw = [_int6_rows(k[:, h], D) for h in range(n_kv)]
    if "fp6k" in parts:
        Kw = [_mxfp6_rows(k[:, h]) for h in range(n_kv)]
    if "int6v" in parts:
        a = np.abs(v).max(-1, keepdims=True)
        Vp = (np.clip(np.rint(v / np.where(a == 0, 1, a / 31)), -31, 31), a[..., 0] / 31)
    if "fp6vt" in parts:
        vp = np.zeros((Tb, n_kv, d))
        vp[:T] = v
        Vw = [_mxfp6_rows(vp[:, h].T) for h in range(n_kv)]
    known = {"fp4k", "nvfp4k", "fp4v", "int4v", "fp4vt", "nvfp4vt", "int6k", "fp6k", "int6v",
             "fp6vt"}
    if parts - known:
        raise ValueError(f"KV format {kv!r}: unknown parts {sorted(parts - known)}")
    return Kw, Vp, Vw, win


def _attend(q, k, v, fq, D, scale, lo=None):
    """Causal attention of every row [T, n_q, d] on its query group's K and V [T, n_kv, d] as
    the device reads them (emulated_logits' points; fq _ident: none): q quantized per head row,
    K as the cache holds it (qwen3._fake_k: ROW at d = D, else an offset per D-block), V per
    token, P per attention block of ATTN_BLOCK positions with V's scales folded in, against
    V^T (UNIT: MM's drops of P's groups). lo [T]: each row's first key (a sliding window).
    _KV[0]: the KV format under test (above)."""
    T, nq, _ = q.shape
    quant = fq is not _ident
    G, Tb = nq // k.shape[1], -(-T // ATTN_BLOCK) * ATTN_BLOCK
    Kw, Vw, vwin = None, None, I.MX_W[I.W8]
    if quant:
        K, dK = _fake_k(k, D)
        Kw, Vp, Vw, vwin = _kv_parts(_KV[0], k, v, D)
        V, vs = _v_parts(v) if Vp is None else Vp
        if Vw is not None:
            vs = np.ones(v.shape[:2])
    else:
        K, dK, V, vs = k, None, v, np.ones(v.shape[:2])
    t = np.arange(T)
    ok = t[None, :] <= t[:, None]
    if lo is not None:
        ok &= t[None, :] >= np.asarray(lo)[:, None]
    o = np.zeros((T, nq, V.shape[2]))
    for hq in range(nq):
        h = hq // G
        qq = fq(q[:, hq] / scale, D)
        if Kw is not None:
            sc = qq @ Kw[h].T
        else:
            sc = (_qk(qq, K[:, h], None if dK is None else dK[:, h], D) if quant
                  else qq @ K[:, h].T)
        sc = np.where(ok, sc, -np.inf)
        pp = np.exp(sc - sc.max(1, keepdims=True))
        if quant:
            pf = np.zeros((T, Tb))
            pf[:, :T] = pp * vs[:, h]
            if Vw is not None:
                pv = fq(pf, D, ATTN_BLOCK) @ Vw[h].T
            else:
                vt = np.zeros((Tb, V.shape[2]))
                vt[:T] = V[:, h]
                pv = Q.mx_matmul(fq(pf, D, ATTN_BLOCK), vt.T, None, vwin, D)
        else:
            pv = pp @ V[:, h]
        o[:, hq] = pv / pp.sum(1, keepdims=True)
    return o


def _qwen3(spec, W, i, x, w, fq, c, s, D, wf):
    """qwen3.emulated_logits' layer i on the rows x [T, H] (wf: a weight's format)."""
    T, d = len(x), spec.head_dim
    p = f"model.layers.{i}."

    def norm(v, g):
        return (v / np.sqrt(np.mean(v * v, -1, keepdims=True) + spec.eps)) * g

    def rot(v):
        h, rd = spec.rope_dim // 2, spec.rope_dim
        v1, v2 = v[..., :h], v[..., h:rd]
        return np.concatenate([v1 * c - v2 * s, v2 * c + v1 * s, v[..., rd:]], -1)

    h = fq(norm(x, W[p + "input_layernorm.weight"]), D)
    q = (h @ w(p + "self_attn.q_proj.weight").T).reshape(T, spec.n_q, d)
    k = (h @ w(p + "self_attn.k_proj.weight").T).reshape(T, spec.n_kv, d)
    v = (h @ w(p + "self_attn.v_proj.weight").T).reshape(T, spec.n_kv, d)
    if spec.qk_norm:
        gq, gk = Q3.qk_gains(spec, W, p)
        q, k = norm(q, gq), norm(k, gk)
    if i not in spec.nope:
        q, k = rot(q), rot(k)
    o = _attend(q, k, v, fq, D, math.sqrt(d))
    x = x + fq(o.reshape(T, -1), D) @ w(p + "self_attn.o_proj.weight").T
    h = fq(norm(x, W[p + "post_attention_layernorm.weight"]), D)
    g, u = h @ w(p + "mlp.gate_proj.weight").T, h @ w(p + "mlp.up_proj.weight").T
    dn = p + "mlp.down_proj.weight"
    return x + _fq_down(fq, (g / (1 + np.exp(-g))) * u, spec, wf, dn, D) @ w(dn).T


def _lfm2(spec, W, i, x, w, fq, c, s, D, wf):
    """lfm2.emulated_logits' layer i (a dense one) on the rows x [T, H]."""
    T, d, H, K = len(x), spec.head_dim, spec.hidden, spec.conv_k
    dk = -(-d // D) * D                         # q and K zero-padded to whole D blocks
    p = f"model.layers.{i}."

    def norm(v, g):
        return L2._norm(v, g, spec.eps)

    h = fq(norm(x, W[p + "operator_norm.weight"]), D)
    if spec.kinds[i] == L2.CONV:
        B, C, xx = np.split(h @ w(p + "conv.in_proj.weight").T, 3, -1)
        bx = np.concatenate([np.zeros((K - 1, H)), B * xx])      # the last K rows of B * x
        wc = W[p + "conv.conv.weight"][:, 0, :]
        y = C * sum(bx[k:k + T] * wc[:, k] for k in range(K))
        x = x + fq(y, D) @ w(p + "conv.out_proj.weight").T
    else:
        a = p + "self_attn."
        q = (h @ w(a + "q_proj.weight").T).reshape(T, spec.n_q, d)
        k = (h @ w(a + "k_proj.weight").T).reshape(T, spec.n_kv, d)
        v = (h @ w(a + "v_proj.weight").T).reshape(T, spec.n_kv, d)

        def rot(v):
            v1, v2 = v[..., :d // 2], v[..., d // 2:]
            return np.concatenate([v1 * c - v2 * s, v2 * c + v1 * s], -1)

        q = rot(norm(q, W[a + "q_layernorm.weight"]))
        k = rot(norm(k, W[a + "k_layernorm.weight"]))
        pad = ((0, 0), (0, 0), (0, dk - d))
        o = _attend(np.pad(q, pad), np.pad(k, pad), v, fq, D, math.sqrt(d))
        x = x + fq(o.reshape(T, -1), D) @ w(a + "out_proj.weight").T
    h = fq(norm(x, W[p + "ffn_norm.weight"]), D)
    if spec.is_moe(i):                          # the routed experts: _moe_ffn, every variant
        return MoEIn(x, h, None)
    g = h @ w(p + "feed_forward.w1.weight").T
    u = h @ w(p + "feed_forward.w3.weight").T
    dn = p + "feed_forward.w2.weight"
    return x + _fq_down(fq, (g / (1 + np.exp(-g))) * u, spec, wf, dn, D) @ w(dn).T


def _qwen35(spec, W, i, x, w, fq, c, s, D, wf):
    """qwen35.emulated_logits' layer i (dense) on the rows x [T, H]; the DeltaNet recurrence
    runs over the rows in order."""
    T, d, eps, K = len(x), spec.head_dim, spec.eps, spec.conv_k
    nh, nk, dk, dv = spec.lin_heads, spec.lin_nk, spec.lin_dk, spec.lin_dv
    S = SLICES[0]                               # a slice's heads (S-exact: S = 1's groups)
    nl = nh // (1 if Q35.s_exact(spec, S, CFG[0]) else S)
    og = 4 if nl % 4 == 0 else 2                # out_proj's QACT ranges (Image.og, a slice's)
    outr = [og * dv] * (nl // og)
    if og == 4 and nl > 2:
        outr[-1:] = [2 * dv, 2 * dv]
    outr = outr * (nh // nl)
    p = f"model.layers.{i}."

    def g1(n):
        return 1 + np.asarray(W[n], np.float64)

    h = fq(Q35._norm(x, g1(p + "input_layernorm.weight"), eps), D)
    if spec.kinds[i] == Q35.LIN:
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
        o = RECUR[0](q, k, v, g, beta)
        o = Q35._norm(o, np.asarray(W[a + "norm.weight"], np.float64), eps) * Q35._silu(z)
        x = x + fq(o.reshape(T, -1), D, outr) @ w(a + "out_proj.weight").T
    else:
        a = p + "self_attn."
        qg = (h @ w(a + "q_proj.weight").T).reshape(T, spec.n_q, 2 * d)
        q, gate = qg[..., :d], qg[..., d:]
        k = (h @ w(a + "k_proj.weight").T).reshape(T, spec.n_kv, d)
        v = (h @ w(a + "v_proj.weight").T).reshape(T, spec.n_kv, d)
        q = Q35._rot(Q35._norm(q, g1(a + "q_norm.weight"), eps), c, s, spec.rope_dim)
        k = Q35._rot(Q35._norm(k, g1(a + "k_norm.weight"), eps), c, s, spec.rope_dim)
        o = _attend(q, k, v, fq, D, math.sqrt(d)) / (1 + np.exp(-gate))
        x = x + fq(o.reshape(T, -1), D) @ w(a + "o_proj.weight").T
    h = fq(Q35._norm(x, g1(p + "post_attention_layernorm.weight"), eps), D)
    mp = spec.mlp_prefix(p)
    gg, u = h @ w(mp + "gate_proj.weight").T, h @ w(mp + "up_proj.weight").T
    dn = mp + "down_proj.weight"
    y = _fq_down(fq, Q35._silu(gg) * u, spec, wf, dn, D) @ w(dn).T
    if spec.moe is not None:                    # y: the shared expert's (_moe_ffn gates it)
        return MoEIn(x, h, y)
    return x + y


def _recur(q, k, v, g, beta):
    """The DeltaNet recurrence over the rows in order (DSTEP's; its fp32 state float64 here):
    o [T, heads, dv] of q, k [T, heads, dk], v [T, heads, dv], the log decays g and the gates
    beta [T, heads]."""
    T, nh, dk = q.shape
    S, o = np.zeros((nh, dk, v.shape[2])), np.zeros((T, nh, v.shape[2]))
    for t in range(T):
        S = S * np.exp(g[t])[:, None, None]
        delta = (v[t] - np.einsum("hij,hi->hj", S, k[t])) * beta[t][:, None]
        S = S + k[t][:, :, None] * delta[:, None, :]
        o[t] = np.einsum("hij,hi->hj", S, q[t])
    return o


RECUR = [_recur]        # (tools/dn_state.py: the state stored in fewer bits between the rows)


class MoEIn(tuple):
    """A MoE layer's state before its routed experts: the rows x [T, H] after the mixer, the
    FFN's input h (quantized), the shared expert's output y (Qwen3.5) or None."""

    def __new__(cls, x, h, y):
        return tuple.__new__(cls, (x, h, y))


def _raw(W, n, e=None) -> np.ndarray:
    """Tensor n (expert e of a fused one) as fp32, not kept by W (qwen3.Weights keeps every
    tensor under 16 MB: a MoE's experts would fill the host)."""
    if hasattr(W, "_where"):                                    # qwen3.Weights
        import torch
        h, k, rows = W._where[n]
        t = (h.get_slice(k)[int(e)] if e is not None else h.get_tensor(k) if rows is None
             else h.get_slice(k)[rows[0]:rows[1]])
        return t.to(torch.float32).numpy()
    if e is not None and hasattr(W, "part"):                    # LazyWeights
        return np.asarray(W.part(n, e), np.float32)
    a = W[n]
    return np.asarray(a if e is None else a[e], np.float32)


def _expert(M, W, p: str, e: int) -> list:
    """Routed expert e of layer prefix p: W_gate, W_up [F, H], W_down [H, F] (fp32)."""
    if M is L2:
        f = f"{p}feed_forward.experts.{e}."
        return [_raw(W, f + n) for n in ("w1.weight", "w3.weight", "w2.weight")]
    gu = _raw(W, p + "mlp.experts.gate_up_proj", e)
    F = gu.shape[0] // 2
    return [gu[:F], gu[F:], _raw(W, p + "mlp.experts.down_proj", e)]


def _moe_ffn(spec, W, i, st: dict, variants: dict, D: int, log=None) -> dict:
    """Layer i's MoE FFN for every variant (st {label: MoEIn}) as lfm2 / qwen35.emulated_logits
    run it: the router int8 in every quantized format (Qwen3.5: the shared expert's gate its
    last row, gating y), moe.route per row, and each routed expert read once and quantized
    once per format (a variant's "experts" format), its down projection's input in the slot's
    QACT ranges (moe.ExpertFormat.ranges). {label: the layer's output rows}."""
    from opentpu.llm import moe as MO
    M, mo, p = family(spec), spec.moe, f"model.layers.{i}."
    held: dict = {}

    def router(f):
        if f not in held:
            if M is L2:
                a = _raw(W, p + "feed_forward.gate.weight")
            else:
                a = np.concatenate([_raw(W, p + "mlp.gate.weight"),
                                    _raw(W, p + "mlp.shared_expert_gate.weight")])
            held[f] = dequant(a, f, D)
        return held[f]

    plan, out = {}, {}
    for lab, (x, h, y) in st.items():
        f = variants[lab]
        _AT["lab"] = lab
        lg = np.asarray(h @ router(None if f is None else "int8").T).view(np.ndarray)
        _AT["lab"] = None
        x = np.array(x, np.float64)
        if M is L2:
            bias = np.asarray(W[p + "feed_forward.expert_bias"], np.float64)
        else:
            x = x + np.asarray(y) / (1 + np.exp(-lg[:, mo.E:mo.E + 1]))
            bias, lg = None, lg[:, :mo.E]
        r = [MO.route(row, bias, mo) for row in lg]
        plan[lab] = (h, np.array([a for a, _ in r]), np.array([b for _, b in r]),
                     None if f is None else f("experts", i))
        out[lab] = x
    need: dict = {}
    for h, ids, wts, fmt in plan.values():
        for e in np.unique(ids):
            need.setdefault(int(e), set()).add(fmt)
    t0 = time.time()
    for n, e in enumerate(sorted(need)):
        mats = _expert(M, W, p, e)
        for fmt in need[e]:
            wg, wu, wd = (dequant(a, fmt, D) for a in mats)
            ce = None if fmt is None else MO.ExpertFormat(spec.hidden, mo.ffn, D, fmt).ranges()
            for lab, (h, ids, wts, f) in plan.items():
                t, j = np.nonzero(ids == e)
                if f != fmt or not len(t):
                    continue
                hs = h.rows(t) if isinstance(h, Q.MXAct) else h[t]
                _AT.update(lab=lab, rows=t)
                g, u = hs @ wg.T, hs @ wu.T
                a = (g / (1 + np.exp(-g))) * u
                if fmt is not None:
                    a = _fake_q(a, D, ce)
                out[lab][t] += wts[t, j][:, None] * np.asarray(a @ wd.T)
                _AT.update(lab=None, rows=None)
        if log and (n + 1) % 64 == 0:
            log(f"  experts {n + 1}/{len(need)} ({time.time() - t0:.0f} s)")
    return out


LAYER = {Q3: _qwen3, L2: _lfm2, Q35: _qwen35}


def _final(spec, W, x):
    M = family(spec)
    if M is Q35:
        return Q35._norm(x, 1 + np.asarray(W["model.norm.weight"], np.float64), spec.eps)
    if M is L2:
        return L2._norm(x, W["model.embedding_norm.weight"], spec.eps)
    return (x / np.sqrt(np.mean(x * x, -1, keepdims=True) + spec.eps)) * W["model.norm.weight"]


def emulate(spec, W, tokens, variants: dict, D: int = 128, logits: bool = False,
            head_rows: int = HEAD_ROWS, shapes: dict | None = None, log=None,
            kl_to: tuple = ()) -> dict:
    """{label: (nll [T-1], top1 [T])} (logits: the logits [T, vocab]) of the variants {label:
    fmt(kind, layer) -> weight format (formats.resolver over the family's KINDS), or None:
    float weights and no quantization point}. shapes gets {(kind, layer): [(rows, cols)]}.
    kl_to: labels of variants to measure every variant against: (nll, top1, {label: KL [T]}),
    each position's KL divergence of the variant's next-token distribution from the label's
    (a second pass over the head)."""
    M = family(spec)
    T = len(tokens)
    shapes = {} if shapes is None else shapes
    x0 = np.asarray(W["model.embed_tokens.weight"][list(tokens)], np.float64)
    x8 = Q3.gathered_rows(x0, D).astype(np.float64) if spec.embed == "int8" else x0
    xs = {lab: (x8 if f is not None else x0) for lab, f in variants.items()}
    cs = [rope_tables(spec, p) for p in range(T)]
    c = np.stack([a for a, _ in cs]).astype(np.float64)[:, None, :]
    s = np.stack([b for _, b in cs]).astype(np.float64)[:, None, :]
    t0 = time.time()
    for i in range(spec.layers):
        held: dict = {}

        def w(n, fmt):
            if (n, fmt) not in held:
                a = W[n]
                shapes.setdefault(M.weight_kind(n), {})[n] = a.shape
                held[n, fmt] = dequant(a, fmt, D)
            return held[n, fmt]

        for lab, f in variants.items():
            fq = _fake_q if f is not None else _ident
            _KV[0] = _kv_mode(lab)
            wf = (lambda n, f=f: None if f is None else f(*M.weight_kind(n)))
            wl = (lambda n, wf=wf: w(n, wf(n)))
            _AT.update(lab=lab, T=T)
            xs[lab] = LAYER[M](spec, W, i, xs[lab], wl, fq, c, s, D, wf)
        _AT["lab"] = None
        held.clear()
        st = {lab: v for lab, v in xs.items() if isinstance(v, MoEIn)}
        if st:
            xs.update(_moe_ffn(spec, W, i, st, variants, D, log))
        if log:
            log(f"layer {i + 1}/{spec.layers} ({time.time() - t0:.0f} s)")
    hs = {lab: (_fake_q if f is not None else _ident)(_final(spec, W, xs[lab]), D)
          for lab, f in variants.items()}
    del xs
    head = "model.embed_tokens.weight" if spec.tied else "lm_head.weight"
    Wh = W[head]
    shapes[("head", 0)] = {head: Wh.shape}
    fmts = {lab: None if f is None else f("head") for lab, f in variants.items()}
    return _head(hs, fmts, lambda: ((r0, Wh[r0:r0 + head_rows])
                                    for r0 in range(0, Wh.shape[0], head_rows)),
                 tokens, D, logits, kl_to)


def _head(hs: dict, fmts: dict, chunks, tokens, D: int = 128, logits: bool = False,
          kl_to: tuple = (), cap: float | None = None) -> dict:
    """emulate's results from the head's inputs hs {label: [T, hidden]} (the final norm's,
    quantized) in the head formats fmts {label: format, None: unquantized}, over chunks()
    (the head's (first row, rows) in turn; twice with kl_to). cap: the logits' soft cap,
    c tanh(z / c) (Gemma 4)."""
    T = len(tokens)
    tgt = np.asarray(tokens[1:])
    hq: dict = {}                   # the head chunk's rows per format

    def lg_of(lab, a):
        fmt = fmts[lab]
        if fmt not in hq:
            hq[fmt] = dequant(a, fmt, D)
        z = hs[lab] @ hq[fmt].T
        return z if cap is None else cap * np.tanh(z / cap)

    acc = {lab: [np.full(T, -np.inf), np.zeros(T), np.zeros(T - 1), np.full(T, -np.inf),
                 np.zeros(T, np.int64), []] for lab in hs}
    for r0, a in chunks():
        for lab in hs:
            _AT.update(lab=lab, T=T, kind="head")       # (the head's drops: this pass only)
            lg = lg_of(lab, a)
            _AT.update(lab=None, kind=None)
            m, se, tl, best, arg, full = acc[lab]
            if logits:
                full.append(lg)
                continue
            mx = np.maximum(m, lg.max(1))
            se[:] = se * np.exp(m - mx) + np.exp(lg - mx[:, None]).sum(1)
            m[:] = mx
            hit = (tgt >= r0) & (tgt < r0 + len(a))
            tl[hit] = lg[:-1][hit, tgt[hit] - r0]
            j = lg.argmax(1)
            up = lg[np.arange(T), j] > best
            best[up], arg[up] = lg[np.arange(T), j][up], j[up] + r0
        hq.clear()
    if logits:
        return {lab: np.concatenate(v[5], 1) for lab, v in acc.items()}
    lse = {lab: m + np.log(se) for lab, (m, se, *_) in acc.items()}
    out = {lab: (lse[lab][:-1] - tl, arg) for lab, (_, _, tl, _, arg, _) in acc.items()}
    if not kl_to:
        return out
    kl = {lab: {r: np.zeros(T) for r in kl_to} for lab in hs}
    for r0, a in chunks():
        ref = {r: lg_of(r, a) - lse[r][:, None] for r in kl_to}
        pr = {r: np.exp(v) for r, v in ref.items()}
        for lab in hs:
            lq = ref[lab] if lab in ref else lg_of(lab, a) - lse[lab][:, None]
            for r in kl_to:
                kl[lab][r] += (pr[r] * (ref[r] - lq)).sum(1)
        hq.clear()
    return {lab: (*out[lab], kl[lab]) for lab in hs}


def weight_bytes(shapes: dict, fmt, D: int = 128) -> int:
    """DRAM bytes a token reads of the weights (rows and scale rows) under fmt(kind, layer)."""
    n = 0
    for (kind, layer), mats in shapes.items():
        if kind == "experts":
            continue
        f = fmt(kind, layer) if kind != "head" else fmt("head")
        for rows, cols in mats.values():
            n += rows * (Q.row_bytes(cols, f, D) + Q.scale_bytes(cols, f, D))
    return n


def text_ids(path, n: int, ids_file=None) -> list:
    """The first n tokens of docs/isa.md, as tools/quant_eval.py reads it (Gemma 4: after <bos>,
    which an instruct checkpoint's tokenizer does not add), or of ids_file's token ids (a JSON
    list, or an object with "ids": tools/chat_text.py's chat)."""
    if ids_file is not None:
        ids = json.loads(Path(ids_file).read_text())
        ids = (ids["ids"] if isinstance(ids, dict) else ids)[:n]
        if len(ids) < n:
            raise ValueError(f"{ids_file} has {len(ids)} tokens, {n} asked")
        return ids
    import transformers
    tok = transformers.AutoTokenizer.from_pretrained(path)
    text = re.sub(r"`|\|", "", (ROOT / "docs/isa.md").read_text())
    spec = load_spec(path)
    ids = (G.text_ids(tok, text, spec.bos) if isinstance(spec, G.Spec) else
           tok(text).input_ids)[:n]
    if len(ids) < n:
        raise ValueError(f"docs/isa.md has {len(ids)} tokens, {n} asked")
    return ids


def _resolver(M, formats: str, wformat: str = "int8", head: str | None = None):
    return FM.resolver(formats, M.KINDS, "", wformat, head)


def _run(spec, W, ids, specs: dict, shapes, D, log) -> dict:
    """{label: row} for specs {label: (formats, wformat, head) or None (float)}."""
    M = family(spec)
    variants = {lab: None if v is None else _resolver(M, *v) for lab, v in specs.items()}
    return _rows(specs, emulate(spec, W, ids, variants, D, shapes=shapes, log=log,
                                kl_to=("float", "int8")))


def _rows(specs: dict, res: dict) -> dict:
    """_run's rows of emulate's results."""
    out = {}
    for lab, (nll, top1, kl) in res.items():
        v = specs[lab]
        out[lab] = {"formats": None if v is None else v[0], "wformat": None if v is None else v[1],
                    "head": None if v is None else v[2], "nll": float(nll.mean()),
                    "ppl": float(np.exp(nll.mean())), "top1": top1.tolist(),
                    "nll_tok": np.round(nll, 5).tolist(), "kl_float": float(kl["float"].mean()),
                    "kl_int8": float(kl["int8"].mean()),
                    "kl_tok": np.round(kl["float"], 7).tolist()}
        if lab in DROPS:                # (ppl: what the window dropped, _counted_mm)
            out[lab]["drops"] = {kind: {k: (v.tolist() if k == "tok" else v)
                                        for k, v in c.items()}
                                 for kind, c in DROPS[lab].items()}
    return out


# Gemma 4: tools/gemma4_quant_eval.py's emulation, a variant at a time
G4KINDS = ("attn", "gateup", "down", "ple")     # a layer's formats (gemma4.layer_formats')
G4CAP = 2048        # the image whose PLE table format a variant takes (otpu-chat's --cap)
G4SRC = ("tools/gemma4_quant_eval.py", "opentpu/llm/gemma4.py", "opentpu/quant.py",
         "opentpu/fp32.py", "opentpu/kernels/gather.py", "opentpu/kernels/mlp.py",
         "opentpu/llm/moe.py", "opentpu/llm/qwen3.py", "opentpu/qcache.py")  # (_g4_run's keys)


def _g4q():
    """tools/gemma4_quant_eval.py (emulate)."""
    import importlib.util
    s = importlib.util.spec_from_file_location("gemma4_quant_eval",
                                               ROOT / "tools/gemma4_quant_eval.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def _g4_args(spec, v, cap: int = G4CAP) -> tuple:
    """emulate's wformat, head format, wmap and PLE table format for a variant (formats,
    wformat, head), None: float ("none"): each checkpoint layer's formats as the image resolves
    them (gemma4.layer_formats; wmap "kind@c-c"), the PLE projection's ("ple"), the head's
    (head, else the formats' head rule, else wformat) and the PLE table's the image of cap
    tokens takes (int8 where it fits the card beside the rest, else fp4: E2B's int8 layers);
    a MoE's experts ("experts": gemma4.expert_format's)."""
    if v is None:
        return "none", "none", {}, "none"
    formats, wformat, head = v
    lf, ple, hf = G.layer_formats(spec, wformat, formats)
    wmap = {"ple": ple}
    if spec.experts:                    # (26B-A4B: one format for the routed experts)
        wmap["experts"] = G.expert_format(spec, wformat, formats)
    for i, f in enumerate(lf):
        c = spec.src(i)
        wmap.update({f"{k}@{c}-{c}": x for k, x in zip(G4KINDS, f)})
    hf = head or hf or wformat
    return wformat, hf, wmap, _g4_ple(spec, v, cap)[0]


@functools.lru_cache(maxsize=None)
def _g4_ple(spec, v, cap: int) -> tuple:
    """The PLE table's (format, on the host) in variant v's image (a layout, no weights;
    OTPU_PLE_HOST=1: int8 on the host); ("none", False) without per-layer embeddings. The
    image is laid out in unbounded DRAM (its PLE choice is by the card's 4 GiB), so a
    reference that does not fit the card has one too (E4B's int8: the table on the host)."""
    from dataclasses import replace

    from opentpu.isasim import board_config
    if not spec.ple_dim:
        return "none", False
    img = spec.image(replace(board_config(), DRAM_BYTES=1 << 40), cap, 1, Q3.PREFILL_ROWS, v[1],
                     v[2], lookup=True, formats=v[0])
    return img.ple_format, img.ple_host


def _g4_spans(spec) -> list:
    """The scan's layer groups: the layers with their own K / V and the KV-shared ones, each
    halved at a multiple of the attention pattern's period (E2B: 0-9, 10-14, 15-24, 25-34)."""
    L = spec.layers
    s = next((i for i in range(L) if spec.kv_src[i] != i), L)
    per = spec.kinds.index(G.FULL) + 1
    out = []
    for a, b in ((0, s), (s, L)):
        m = a + max(1, round((b - a) / 2 / per)) * per
        out += [(a, m - 1), (m, b - 1)] if m < b else [(a, b - 1)] if a < b else []
    return out


def _g4_shapes(spec, W) -> dict:
    """{name: (rows, cols)} of the matrices a token reads but the head (the embedding's and
    the PLE table's rows are gathered): each layer's attention (K / V where it has its own),
    MLP, PLE gate and projection, and the layers' rows of the PLE model projection."""
    def shape(n):
        if isinstance(W, G.Weights):
            f, k = W._files[n]
            return tuple(f.get_slice(k).get_shape())
        return W[n].shape

    names = []
    for i in range(spec.layers):
        p = f"model.layers.{spec.src(i)}."
        a = p + "self_attn."
        names += [a + "q_proj.weight", a + "o_proj.weight"]
        if spec.kv_src[i] == i:
            names += [a + "k_proj.weight"] + ([] if spec.kv_same(i) else [a + "v_proj.weight"])
        names += [p + f"mlp.{m}_proj.weight" for m in ("gate", "up", "down")]
        if spec.ple_dim:
            names += [p + "per_layer_input_gate.weight", p + "per_layer_projection.weight"]
    out = {n: shape(n) for n in names}
    if spec.ple_dim:
        out["model.per_layer_model_projection.weight"] = (spec.layers * spec.ple_dim,
                                                          spec.hidden)
    return out


def _g4_bytes(spec, shapes: dict, v, D: int = 128) -> int:
    """weight_bytes for Gemma 4: the matrices shapes (_g4_shapes') and the LM head in variant
    v's formats (_g4_args')."""
    _, hf, wmap, _ = _g4_args(spec, v)

    def fmt(n):
        if not n.startswith("model.layers."):
            return wmap["ple"]                              # the PLE model projection
        c = n.split(".")[2]
        k = "attn" if ".self_attn." in n else "down" if ".down_proj" in n else \
            "gateup" if ".mlp." in n else "ple"
        return wmap[f"{k}@{c}-{c}"]

    n = 0
    for f, (rows, cols) in [(fmt(k), s) for k, s in shapes.items()] + \
            [(hf, (spec.vocab, spec.hidden))]:
        cols = -(-cols // D) * D
        n += rows * (Q.row_bytes(cols, f, D) + Q.scale_bytes(cols, f, D))
    return n


def _g4_run(spec, W, ids, specs: dict, D, log, cache=None, head_rows: int = HEAD_ROWS) -> dict:
    """_run for Gemma 4: each variant's head inputs (gemma4_quant_eval.emulate's, hidden), a
    variant at a time (the whole sequence layer after layer: a KV-shared layer reads an
    earlier one's K / V, so variants do not share a pass), each saved in the directory cache
    and read back from it (a run killed at the memory floor goes on at its variant, a model
    without KV-shared layers after its last layer done; a later run reuses the references),
    then _head over them all with the soft cap (a row's "ple_table": the PLE table's format)."""
    E = _g4q()
    src = b"".join((ROOT / p).read_bytes() for p in G4SRC)
    hs, fmts, pfs = {}, {}, {}
    for lab, v in specs.items():
        wf, hf, wmap, pf = _g4_args(spec, v)
        fmts[lab], pfs[lab] = None if hf == "none" else hf, pf
        kv = _kv_mode(lab)              # (the KV format under test: the emulation's hook)
        if E is not None:
            E.KV[0] = (lambda k, v, D, kv=kv: _kv_parts(kv, k, v, D)) if kv else None
        key = repr((str(getattr(W, "model_dir", "")), spec, list(ids), D, wf, hf, pf,
                    sorted(wmap.items()))) + (f";kv={kv}" if kv else "")
        f = Path(cache) / f"{hashlib.sha256(key.encode() + src).hexdigest()[:20]}.npz" \
            if cache else None
        if f is not None and f.exists():
            with np.load(f) as z:           # the rows and, quantized, their offsets (MXAct)
                hs[lab] = Q.MXAct.of(z["v"], z["da"] if "da" in z else None)
            continue
        t0 = time.time()
        if f is not None:
            f.parent.mkdir(parents=True, exist_ok=True)
        ck = f"{f}.ckpt.npz" if f is not None else None   # (no KV-shared layers: resumes)
        _AT.update(lab=lab, T=len(ids))
        hs[lab] = E.emulate(spec, W, ids, D, wformat=wf, hf=hf, ple_format=pf, wmap=wmap,
                            hidden=True, ckpt=ck)
        _AT["lab"] = None
        if f is not None:
            da = getattr(hs[lab], "da", None)
            np.savez(f"{f}.tmp.npz", v=np.asarray(hs[lab]).view(np.ndarray),
                     **({} if da is None else {"da": da}))
            os.replace(f"{f}.tmp.npz", f)
            if os.path.exists(ck):
                os.remove(ck)
        if log:
            log(f"{lab}: {time.time() - t0:.0f} s (PLE table {pf})")
    head = "model.embed_tokens.weight" if spec.tied else "lm_head.weight"

    def chunks():
        t = G._Rows(*W._files[head]) if isinstance(W, G.Weights) else W[head]
        for r0 in range(0, spec.vocab, head_rows):
            yield r0, t[r0:r0 + head_rows]
    rows = _rows(specs, _head(hs, fmts, chunks, ids, D, kl_to=("float", "int8"),
                              cap=spec.softcap))
    for lab, r in rows.items():
        r["ple_table"] = pfs[lab] + (" (host)" if specs[lab] is not None and
                                     _g4_ple(spec, specs[lab], G4CAP)[1] else "")
    return rows


def _open(path, D: int = 128, cache=None) -> tuple:
    """(spec, run, nbytes, kinds, spans) of a checkpoint: run(specs, ids, log) -> {label: row}
    (_run, or _g4_run), nbytes((formats, wformat, head)) -> weight bytes a token reads (after a
    run), the scan's kinds and its layer groups."""
    spec = load_spec(path)
    L = spec.layers
    if isinstance(spec, G.Spec):
        W = G.load_weights(path)
        shapes = _g4_shapes(spec, W)
        return (spec, lambda specs, ids, log: _g4_run(spec, W, ids, specs, D, log, cache),
                lambda v: _g4_bytes(spec, shapes, v, D),
                [k for k in G4KINDS if k != "ple" or spec.ple_dim], _g4_spans(spec))
    W = Q3.load_weights(path)
    M = family(spec)
    present = {M.weight_kind(n)[0] for n in W if n.startswith("model.layers.")}
    kinds = [k for k in M.KINDS if k not in ("mlp", "head") and k in present]
    shapes: dict = {}
    return (spec, lambda specs, ids, log: _run(spec, W, ids, specs, shapes, D, log),
            lambda v: weight_bytes(shapes, _resolver(M, *v), D), kinds,
            [(round(j * L / 4), round((j + 1) * L / 4) - 1) for j in range(4)])


def _finish(rows: dict, nbytes) -> None:
    """Weight bytes (nbytes((formats, wformat, head))), the bytes saved against int8, top-1
    agreement with int8 and float (the rows' "top1" lists dropped), and the standard error of
    the mean NLL's difference from int8's (paired over the tokens: the perplexity ratio's
    relative error). dkl: the mean KL
    divergence from float's distribution beyond int8's (the NLL a variant loses against int8
    where the float model's predictions are calibrated: an estimate of the log perplexity
    ratio without the sampled token's noise), se_dkl its standard error (paired)."""
    ref8, refF = rows["int8"]["top1"], rows["float"]["top1"]
    n8, k8 = np.asarray(rows["int8"]["nll_tok"]), np.asarray(rows["int8"]["kl_tok"])
    for r in rows.values():
        t = r.pop("top1")
        r["agree_int8"] = float(np.mean(np.equal(t, ref8)))
        r["agree_float"] = float(np.mean(np.equal(t, refF)))
        dn = np.asarray(r["nll_tok"]) - n8
        r["se_int8"] = float(dn.std(ddof=1) / math.sqrt(len(dn)))
        dk = np.asarray(r["kl_tok"]) - k8
        r["dkl"], r["se_dkl"] = float(dk.mean()), float(dk.std(ddof=1) / math.sqrt(len(dk)))
        if r["wformat"] is not None:
            r["bytes"] = nbytes((r["formats"], r["wformat"], r["head"]))
    for r in rows.values():
        if "bytes" in r:
            r["saved"] = rows["int8"]["bytes"] - r["bytes"]


def scan(name, out, n_tok: int, f4: str, D: int = 128, cache=None, base: str = "",
         ids_file=None) -> None:
    """base: formats every variant starts from, the "int8" row's (the 26B: experts=fp4,
    head=fp4, the card's); its rows then put the scan's kinds in f4 on top of it. ids_file:
    the text's token ids (text_ids)."""
    path = model_dir(name)
    spec, run, nbytes, kinds, qs = _open(path, D, cache or f"{out}.xl")
    g4 = isinstance(spec, G.Spec)
    ids = text_ids(path, n_tok, ids_file)
    t0 = time.time()
    on = lambda fs: ",".join(f for f in (base, fs) if f)           # noqa: E731

    def log(m):
        print(f"[{time.time() - t0:6.0f} s] {m}", flush=True)

    groups = {f"{k}@{a}-{b}": f"{k}@{a}-{b}={f4}" for k in kinds for a, b in qs}
    if not g4:                  # (Gemma 4's head: a row of its own)
        groups["head"] = f"head={f4}"
    p1 = {"float": None, "int8": (base, "int8", None),
          f4: (on(",".join(f"{k}={f4}" for k in kinds)), "int8", None) if base else
          ("", f4, "int8")}
    p1.update({f"{k}={f4}": (on(f"{k}={f4}"), "int8", None) for k in kinds})
    if g4 and spec.ple_dim:     # the layers' PLE alone
        p1[f"ple@0-{spec.layers - 1}={f4}"] = (on(f"ple@0-{spec.layers - 1}={f4}"), "int8", None)
    if g4 and "head=" not in base:      # the head
        p1[f"head={f4}"] = (on(f"head={f4}"), "int8", None)
    p1.update({g: (on(fs), "int8", None) for g, fs in groups.items()})
    log(f"{name}: {len(ids)} tokens, pass 1: {len(p1)} variants" + (f" on {base}" if base else ""))
    rows = run(p1, ids, log)
    b8 = nbytes((base, "int8", None))
    saved = {g: b8 - nbytes((on(groups[g]), "int8", None)) for g in groups}
    dkl = {g: rows[g]["kl_float"] - rows["int8"]["kl_float"] for g in groups}
    gain = sorted(groups, key=lambda g: dkl[g] / max(saved[g], 1))
    log("order (dKL per byte saved): " + ", ".join(gain))
    p2 = {"float": None, "int8": (base, "int8", None)}    # (the KL references again)
    pool = kinds if g4 else kinds + ["head"]
    for m in range(2, 1 << len(pool)):                  # every subset of the kinds (and the head)
        sel = [k for j, k in enumerate(pool) if m >> j & 1]
        if len(sel) > 1 and sel != kinds:               # (all the kinds: the f4 row)
            p2["uniform " + "+".join(sel)] = (on(",".join(f"{k}={f4}" for k in sel)), "int8",
                                              None)
    for j in range(2, len(gain) + 1):
        p2[f"greedy {j}"] = (on(",".join(groups[g] for g in gain[:j])), "int8", None)
    log(f"pass 2: {len(p2)} variants")
    rows.update(run(p2, ids, log))
    _finish(rows, nbytes)
    Path(out).write_text(json.dumps({"model": str(name), "tokens": len(ids), "fmt": f4,
                                     "base": base, "order": gain, "variants": rows}, indent=1))
    log(f"wrote {out}")
    for lab, r in sorted(rows.items(), key=lambda kv: kv[1].get("bytes", 1 << 62)):
        print(f"{lab:40s} ppl {r['ppl']:8.4f} (+-{100 * r['se_int8']:.2f}%)  dKL "
              f"{100 * r['dkl']:+.3f}% (+-{100 * r['se_dkl']:.3f})  "
              f"{r.get('bytes', 0) / 2**20:8.1f} MiB  top1/int8 {r['agree_int8']:.3f}")


def ppl(name, formats: list, wformat: str, n_tok: int, D: int = 128, out=None,
        cache=None, base: str = "", ids_file=None) -> None:
    path = model_dir(name)
    spec, run, nbytes, _, _ = _open(path, D, cache)
    ids = text_ids(path, n_tok, ids_file)
    specs = {"float": None, "int8": (base, "int8", None)}
    specs.update({f or "(none)": (",".join(x for x in (base, f.partition(";kv=")[0]) if x),
                                  wformat, None)
                  for f in formats})                    # f: a formats string, ";kv=MODE"
    DROPS.clear()
    Q.mx_matmul = _counted_mm           # (each variant's drops in its row: "drops")
    try:
        rows = run(specs, ids, (lambda m: print(m, flush=True)) if isinstance(spec, G.Spec)
                   else None)
    finally:
        Q.mx_matmul = _MX_MATMUL
    _finish(rows, nbytes)
    if out:
        Path(out).write_text(json.dumps({"model": str(name), "tokens": len(ids), "variants": rows},
                                        indent=1))
    for lab, r in rows.items():
        print(f"{lab:40s} ppl {r['ppl']:8.4f} (+-{100 * r['se_int8']:.2f}%)  dKL "
              f"{100 * r['dkl']:+.3f}% (+-{100 * r['se_dkl']:.3f})  "
              f"{r.get('bytes', 0) / 2**20:8.1f} MiB  top1/int8 {r['agree_int8']:.3f}")


def check() -> None:
    """emulate() == emulated_logits on tiny random models of each family."""
    import tempfile
    import torch
    import transformers
    torch.manual_seed(0)
    nope = [1, 1, 1, 0, 1, 1, 1, 0]
    short = [1.0 + 0.05 * i for i in range(48)]
    lt = ["linear_attention", "linear_attention", "full_attention"] * 2
    models = {
        "qwen3": transformers.Qwen3ForCausalLM(transformers.Qwen3Config(
            hidden_size=256, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
            head_dim=128, intermediate_size=512, vocab_size=1000, rms_norm_eps=1e-6,
            rope_theta=1e6, tie_word_embeddings=True, max_position_embeddings=4096)),
        "smollm3": transformers.SmolLM3ForCausalLM(transformers.SmolLM3Config(
            hidden_size=256, num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=2,
            head_dim=128, intermediate_size=512, vocab_size=1000, rms_norm_eps=1e-6,
            rope_parameters={"rope_type": "default", "rope_theta": 5e6}, no_rope_layers=nope,
            tie_word_embeddings=True, max_position_embeddings=4096, bos_token_id=1,
            eos_token_id=2, pad_token_id=0)),
        "phi3": transformers.Phi3ForCausalLM(transformers.Phi3Config(
            hidden_size=768, num_hidden_layers=2, num_attention_heads=6, num_key_value_heads=2,
            intermediate_size=512, vocab_size=1000, rms_norm_eps=1e-5, rope_theta=1e4,
            partial_rotary_factor=0.75, max_position_embeddings=16384,
            original_max_position_embeddings=4096,
            rope_scaling={"type": "longrope", "short_factor": short,
                          "long_factor": [4 * f for f in short]},
            tie_word_embeddings=False, bos_token_id=1, eos_token_id=2, pad_token_id=0)),
        "lfm2": transformers.Lfm2ForCausalLM(transformers.Lfm2Config(
            hidden_size=256, num_hidden_layers=5, num_attention_heads=4, num_key_value_heads=2,
            intermediate_size=512, vocab_size=1000, norm_eps=1e-5,
            layer_types=["conv", "full_attention", "conv", "full_attention", "conv"],
            conv_L_cache=3, conv_bias=False, block_auto_adjust_ff_dim=False,
            tie_word_embeddings=True, max_position_embeddings=4096,
            rope_parameters={"rope_type": "default", "rope_theta": 1e6})),
        "qwen35": transformers.Qwen3_5ForCausalLM(transformers.Qwen3_5TextConfig(
            hidden_size=256, num_hidden_layers=6, num_attention_heads=8, num_key_value_heads=2,
            head_dim=256, intermediate_size=512, vocab_size=1000, layer_types=lt,
            linear_num_key_heads=4, linear_num_value_heads=8, linear_key_head_dim=128,
            linear_value_head_dim=128, linear_conv_kernel_dim=4, tie_word_embeddings=True,
            max_position_embeddings=4096, rms_norm_eps=1e-6,
            rope_parameters={"rope_type": "default", "rope_theta": 1e7,
                             "partial_rotary_factor": 0.25})),
    }
    mixes = ["", "attn=fp4", "mlp=fp4,down@1=int8,head=fp4", "gateup@0-2=int4,attn@3-7=fp4",
             "conv=fp4,delta=fp4,down=fp4"]
    rng = np.random.default_rng(0)
    worst = 0.0
    for name, m in models.items():
        m = m.float().eval()
        with torch.no_grad():
            for n, p in m.named_parameters():
                if "norm" in n:
                    p.copy_((0.0 if name == "qwen35" and not n.endswith("linear_attn.norm.weight")
                             else 1.0) + 0.1 * torch.randn_like(p))
                    if name == "qwen3" and n.endswith("k_norm.weight"):   # K smoothed (qk_gains)
                        p[[5, 69]] = 60.0
        with tempfile.TemporaryDirectory() as d:
            m.save_pretrained(d)
            spec = load_spec(d)
            W = Q3.load_weights(d)
            M = family(spec)
            toks = [int(t) for t in rng.integers(3, 1000, 140)]          # P: two D-blocks
            for f in mixes:
                fs = ",".join(it for it in f.split(",") if it and it.split("@")[0].split("=")[0]
                              in M.KINDS)
                ref = M.emulated_logits(spec, W, toks, formats=fs)
                e = emulate(spec, W, toks, {"v": _resolver(M, fs)}, logits=True)["v"]
                err = np.abs(e - ref).max() / np.abs(ref).max()
                nll, top1, kl = emulate(spec, W, toks, {"v": _resolver(M, fs), "f": None},
                                        head_rows=256, kl_to=("f",))["v"]
                lse = np.log(np.exp(ref - ref.max(1, keepdims=True)).sum(1)) + ref.max(1)
                errn = np.abs(nll - (lse[:-1] - ref[np.arange(len(toks) - 1), toks[1:]])).max()
                agree = np.mean(top1 == ref.argmax(1))
                lf = emulate(spec, W, toks, {"f": None}, logits=True)["f"]   # KL from float's
                lpf = lf - (np.log(np.exp(lf - lf.max(1, keepdims=True)).sum(1))
                            + lf.max(1))[:, None]
                kl_ref = (np.exp(lpf) * (lpf - (ref - lse[:, None]))).sum(1)
                errk = np.abs(kl["f"] - kl_ref).max() / max(kl_ref.max(), 1e-12)
                worst = max(worst, err)
                print(f"{name:8s} {fs or '(int8)':34s} logits rel {err:.2e}  nll {errn:.2e}  "
                      f"KL rel {errk:.2e}  argmax {agree:.3f}")
                assert err < 1e-6 and errn < 1e-6 and errk < 1e-5 and agree > 0.99, (name, fs)
            if name == "qwen35":            # (--S 2: the slices' out_proj and down ranges;
                SLICES[0] = 2               # --target bcu1525-s2: S-exact, S = 1's)
                from opentpu.target import config as target_config
                for tg in (None, "bcu1525-s2"):
                    CFG[0] = tg and target_config(tg)
                    assert Q35.s_exact(spec, 2, CFG[0]) == (tg is not None)
                    for fs in ("", "delta=fp4,down=fp4"):
                        ref = M.emulated_logits(spec, W, toks, formats=fs, S=2, cfg=CFG[0])
                        e = emulate(spec, W, toks, {"v": _resolver(M, fs)}, logits=True)["v"]
                        err = np.abs(e - ref).max() / np.abs(ref).max()
                        print(f"{name:8s} S 2 {tg or '':10s} {fs or '(int8)':19s} "
                              f"logits rel {err:.2e}")
                        assert err < 1e-6, (name, fs, 2, tg)
                SLICES[0], CFG[0] = 1, None
            fl = emulate(spec, W, toks[:40], {"f": None}, logits=True)["f"]
            r = M.reference_logits(spec, W, toks[:40])
            print(f"{name:8s} float vs reference_logits rel "
                  f"{np.abs(fl - r).max() / np.abs(r).max():.2e}")
    import importlib.util                   # the MoE models: tools/exp/mx_formats.py's tiny ones
    sp = importlib.util.spec_from_file_location("mx_formats", ROOT / "tools/exp/mx_formats.py")
    X = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(X)
    for name, (W, spec) in (("lfm2-moe", X._tiny_lfm2_moe()), ("qwen35-moe", X._tiny_qwen35_moe())):
        M = family(spec)
        toks = [int(t) for t in rng.integers(3, 1000, 140)]
        for fs, wf in (("", "int8"), ("attn=nvfp4,head=fp4", "fp4"), ("head=int8", "nvfp4")):
            ref = M.emulated_logits(spec, W, toks, wformat=wf, formats=fs)
            e = emulate(spec, W, toks, {"v": _resolver(M, fs, wf)}, logits=True)["v"]
            err = np.abs(e - ref).max() / np.abs(ref).max()
            nll, top1, kl = emulate(spec, W, toks, {"v": _resolver(M, fs, wf), "f": None},
                                    head_rows=256, kl_to=("f",))["v"]
            lse = np.log(np.exp(ref - ref.max(1, keepdims=True)).sum(1)) + ref.max(1)
            errn = np.abs(nll - (lse[:-1] - ref[np.arange(len(toks) - 1), toks[1:]])).max()
            worst = max(worst, err)
            print(f"{name:10s} {wf:5s} {fs or '-':22s} logits rel {err:.2e}  nll {errn:.2e}")
            assert err < 1e-6 and errn < 1e-6, (name, wf, fs)
        if name == "qwen35-moe":            # (--S 2: the shared expert's down in S = 1's chunks,
            SLICES[0] = 2                   # an fp4 one's F 256 not two slices' chunks)
            fs = "delta=fp4,down=fp4"
            ref = M.emulated_logits(spec, W, toks, formats=fs, S=2)
            e = emulate(spec, W, toks, {"v": _resolver(M, fs)}, logits=True)["v"]
            err = np.abs(e - ref).max() / np.abs(ref).max()
            SLICES[0] = 1
            print(f"{name:10s} S 2   {fs:22s} logits rel {err:.2e}")
            assert err < 1e-6, (name, 2)
    print(f"OK (worst logits rel {worst:.2e})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("scan")
    a.add_argument("model")
    a.add_argument("out")
    a.add_argument("--tokens", type=int, default=900)
    a.add_argument("--fmt", default="fp4", choices=["fp4", "int4", "nvfp4"])
    a.add_argument("--cache", help="Gemma 4: the head inputs' directory (default OUT.json.xl)")
    a.add_argument("--base", default="", help="formats every variant starts from (the int8 "
                   "row's), e.g. experts=fp4,head=fp4 (the 26B's card)")
    a.add_argument("--ids", help="the text: token ids from this JSON file (tools/chat_text.py's "
                   "chat) instead of docs/isa.md")
    a.add_argument("--S", type=int, help="the device's slices (QACT ranges; default 1)")
    a.add_argument("--target", help="a target: its slices and S-exact choice (bcu1525-s2)")
    a = sub.add_parser("ppl")
    a.add_argument("model")
    a.add_argument("formats", nargs="+")
    a.add_argument("--wformat", default="int8")
    a.add_argument("--tokens", type=int, default=900)
    a.add_argument("--out", help="the rows as JSON (scan's format)")
    a.add_argument("--cache", help="Gemma 4: the head inputs' directory")
    a.add_argument("--base", default="", help="formats every row starts from, as scan's")
    a.add_argument("--ids", help="the text: token ids from this JSON file (tools/chat_text.py's "
                   "chat) instead of docs/isa.md")
    a.add_argument("--S", type=int, help="the device's slices (QACT ranges; default 1)")
    a.add_argument("--target", help="a target: its slices and S-exact choice (bcu1525-s2)")
    sub.add_parser("check")
    a = ap.parse_args()
    if getattr(a, "target", None):
        from opentpu.target import config as target_config
        CFG[0] = target_config(a.target)
        if a.S not in (None, CFG[0].S):
            ap.error(f"--S {a.S}: {a.target} has {CFG[0].S} slices")
    SLICES[0] = getattr(a, "S", None) or (CFG[0].S if CFG[0] else 1)
    if SLICES[0] > 1:                       # (no silent layout: --target or OTPU_SEXACT)
        print(f"formats_scan: S {SLICES[0]}, S-exact {'on' if Q35.s_exact_on(CFG[0]) else 'off'} "
              f"({getattr(a, 'target', None) or 'no --target'}, OTPU_SEXACT "
              f"{os.environ.get('OTPU_SEXACT', '-')})", file=sys.stderr)
    if a.cmd == "scan":
        scan(a.model, a.out, a.tokens, a.fmt, cache=a.cache, base=a.base, ids_file=a.ids)
    elif a.cmd == "ppl":
        ppl(a.model, a.formats, a.wformat, a.tokens, out=a.out, cache=a.cache, base=a.base,
            ids_file=a.ids)
    else:
        check()


if __name__ == "__main__":
    sys.exit(main())
