"""Weight formats per kind and layer range (opentpu/llm/formats.py) for the Qwen3 / Llama-like
(SmolLM3, Phi-4-mini), LFM2 and Qwen3.5 models: perplexity over the first --tokens tokens of
docs/isa.md (quant_eval.py's text: prose the models have not memorized) in float64 emulations
of the device's quantization. emulate() is the families' emulated_logits with the loops swapped:
layers outer, the whole sequence and every variant of a pass at once, so each layer's weights
are quantized once per format (4-bit: through the image cache, opentpu/qcache.py) and only one
layer's weights are held.

    python tools/formats_scan.py scan MODEL OUT.json [--tokens 900] [--fmt fp4]
        pass 1: float weights (no quantization at all), int8, the 4-bit image (WF --fmt, head
        int8), and int8 with one group of weights in --fmt: a kind in all layers, a kind in a
        quarter of them, the head. pass 2: the uniform mixes (each kind one format in every
        layer: what a model with one layer layout runs) and the quarter groups and the head
        added in order of NLL lost per byte saved (cumulative). OUT.json: every variant's
        formats string, mean NLL, perplexity, weight bytes per token and top-1 agreement
    python tools/formats_scan.py ppl MODEL FORMATS... [--wformat int8] [--tokens 900]
        the perplexity of formats strings (as OTPU_FORMATS; "" for none)
    python tools/formats_scan.py check
        tiny random models of each family (SmolLM3, Phi-3, Qwen3.5 with 4 key heads, LFM2):
        emulate() against emulated_logits (formats including ranged ones) and the NLL of the
        chunked head against the full logits

MODEL: a short name of opentpu.llm.MODELS or a checkpoint directory. A dense model only (the
MoE models' experts are not a kind here).
"""
import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

from opentpu import qcache as QC
from opentpu import quant as Q
from opentpu.llm import formats as FM
from opentpu.llm import lfm2 as L2
from opentpu.llm import load_spec, model_dir
from opentpu.llm import qwen3 as Q3
from opentpu.llm import qwen35 as Q35
from opentpu.llm.qwen3 import _fake_q, rope_tables

ROOT = Path(__file__).resolve().parent.parent
HEAD_ROWS = 16384       # LM head rows per pass (quantize_mxu's pieces: the same quantization)


def family(spec):
    return {Q3.Spec: Q3, L2.Spec: L2, Q35.Spec: Q35}[type(spec)]


def _ident(v, d=None):
    return v


def dequant(a, fmt, D):
    """A weight as the device holds it, in float64 (None: unquantized): int8 as _fake_w; 4-bit
    the image cache's DRAM rows and scale words, dequantized (the MXU's arithmetic)."""
    if fmt is None:
        return np.asarray(a, np.float64)
    if fmt == "int8":
        return _fake_q(np.asarray(a, np.float64), D)
    rows, words = QC.quantize_mxu(a, fmt, D)
    return Q.dequantize_w4(rows[:, :a.shape[1] // 2], words, fmt, D)


def _attend(q, K, V, fq, D, qd, scale):
    """Causal attention of every row [T, n_q, d] on its query group's K, V [T, n_kv, d] (int8
    already): q quantized per qd-block, P per D keys (emulated_logits' points)."""
    T, nq, _ = q.shape
    G, Dp = nq // K.shape[1], -(-T // D) * D
    causal = np.tri(T, dtype=bool)
    o = np.zeros((T, nq, V.shape[2]))
    for hq in range(nq):
        sc = np.where(causal, fq(q[:, hq] / scale, qd) @ K[:, hq // G].T, -np.inf)
        pp = np.exp(sc - sc.max(1, keepdims=True))
        ppad = np.zeros((T, Dp))
        ppad[:, :T] = pp
        o[:, hq] = (fq(ppad, D)[:, :T] @ V[:, hq // G]) / pp.sum(1, keepdims=True)
    return o


def _qwen3(spec, W, i, x, w, fq, c, s, D):
    """qwen3.emulated_logits' layer i on the rows x [T, H]."""
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
        q = norm(q, W[p + "self_attn.q_norm.weight"])
        k = norm(k, W[p + "self_attn.k_norm.weight"])
    if i not in spec.nope:
        q, k = rot(q), rot(k)
    o = _attend(q, fq(k, D), fq(v, d), fq, D, D, math.sqrt(d))
    x = x + fq(o.reshape(T, -1), D) @ w(p + "self_attn.o_proj.weight").T
    h = fq(norm(x, W[p + "post_attention_layernorm.weight"]), D)
    g, u = h @ w(p + "mlp.gate_proj.weight").T, h @ w(p + "mlp.up_proj.weight").T
    return x + fq((g / (1 + np.exp(-g))) * u, D) @ w(p + "mlp.down_proj.weight").T


def _lfm2(spec, W, i, x, w, fq, c, s, D):
    """lfm2.emulated_logits' layer i (a dense one) on the rows x [T, H]."""
    T, d, H, K = len(x), spec.head_dim, spec.hidden, spec.conv_k
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
        o = _attend(q, fq(k, min(d, D)), fq(v, d), fq, D, min(d, D), math.sqrt(d))
        x = x + fq(o.reshape(T, -1), D) @ w(a + "out_proj.weight").T
    if spec.is_moe(i):
        raise ValueError("formats_scan: dense models only")
    h = fq(norm(x, W[p + "ffn_norm.weight"]), D)
    g = h @ w(p + "feed_forward.w1.weight").T
    u = h @ w(p + "feed_forward.w3.weight").T
    return x + fq((g / (1 + np.exp(-g))) * u, D) @ w(p + "feed_forward.w2.weight").T


def _qwen35(spec, W, i, x, w, fq, c, s, D):
    """qwen35.emulated_logits' layer i (dense) on the rows x [T, H]; the DeltaNet recurrence
    runs over the rows in order."""
    T, d, eps, K = len(x), spec.head_dim, spec.eps, spec.conv_k
    nh, nk, dk, dv = spec.lin_heads, spec.lin_nk, spec.lin_dk, spec.lin_dv
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
        S, o = np.zeros((nh, dk, dv)), np.zeros((T, nh, dv))
        for t in range(T):
            S = S * np.exp(g[t])[:, None, None]
            delta = (v[t] - np.einsum("hij,hi->hj", S, k[t])) * beta[t][:, None]
            S = S + k[t][:, :, None] * delta[:, None, :]
            o[t] = np.einsum("hij,hi->hj", S, q[t])
        o = Q35._norm(o, np.asarray(W[a + "norm.weight"], np.float64), eps) * Q35._silu(z)
        x = x + fq(o.reshape(T, -1), D) @ w(a + "out_proj.weight").T
    else:
        a = p + "self_attn."
        qg = (h @ w(a + "q_proj.weight").T).reshape(T, spec.n_q, 2 * d)
        q, gate = qg[..., :d], qg[..., d:]
        k = (h @ w(a + "k_proj.weight").T).reshape(T, spec.n_kv, d)
        v = (h @ w(a + "v_proj.weight").T).reshape(T, spec.n_kv, d)
        q = Q35._rot(Q35._norm(q, g1(a + "q_norm.weight"), eps), c, s, spec.rope_dim)
        k = Q35._rot(Q35._norm(k, g1(a + "k_norm.weight"), eps), c, s, spec.rope_dim)
        o = _attend(q, fq(k, D), fq(v, d), fq, D, D, math.sqrt(d)) / (1 + np.exp(-gate))
        x = x + fq(o.reshape(T, -1), D) @ w(a + "o_proj.weight").T
    if spec.moe is not None:
        raise ValueError("formats_scan: dense models only")
    h = fq(Q35._norm(x, g1(p + "post_attention_layernorm.weight"), eps), D)
    mp = spec.mlp_prefix(p)
    gg, u = h @ w(mp + "gate_proj.weight").T, h @ w(mp + "up_proj.weight").T
    return x + fq(Q35._silu(gg) * u, D) @ w(mp + "down_proj.weight").T


LAYER = {Q3: _qwen3, L2: _lfm2, Q35: _qwen35}


def _final(spec, W, x):
    M = family(spec)
    if M is Q35:
        return Q35._norm(x, 1 + np.asarray(W["model.norm.weight"], np.float64), spec.eps)
    if M is L2:
        return L2._norm(x, W["model.embedding_norm.weight"], spec.eps)
    return (x / np.sqrt(np.mean(x * x, -1, keepdims=True) + spec.eps)) * W["model.norm.weight"]


def emulate(spec, W, tokens, variants: dict, D: int = 128, logits: bool = False,
            head_rows: int = HEAD_ROWS, shapes: dict | None = None, log=None) -> dict:
    """{label: (nll [T-1], top1 [T])} (logits: the logits [T, vocab]) of the variants {label:
    fmt(kind, layer) -> weight format (formats.resolver over the family's KINDS), or None:
    float weights and no quantization point}. shapes gets {(kind, layer): [(rows, cols)]}."""
    M = family(spec)
    T = len(tokens)
    shapes = {} if shapes is None else shapes
    x0 = np.asarray(W["model.embed_tokens.weight"][list(tokens)], np.float64)
    xs = {lab: (_fake_q(x0, D) if f is not None and spec.embed == "int8" else x0)
          for lab, f in variants.items()}
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
            wl = (lambda n, f=f: w(n, None if f is None else f(*M.weight_kind(n))))
            xs[lab] = LAYER[M](spec, W, i, xs[lab], wl, fq, c, s, D)
        held.clear()
        if log:
            log(f"layer {i + 1}/{spec.layers} ({time.time() - t0:.0f} s)")
    hs = {lab: (_fake_q if f is not None else _ident)(_final(spec, W, xs[lab]), D)
          for lab, f in variants.items()}
    del xs
    head = "model.embed_tokens.weight" if spec.tied else "lm_head.weight"
    Wh = W[head]
    shapes[("head", 0)] = {head: Wh.shape}
    tgt = np.asarray(tokens[1:])
    acc = {lab: [np.full(T, -np.inf), np.zeros(T), np.zeros(T - 1), np.full(T, -np.inf),
                 np.zeros(T, np.int64), []] for lab in variants}
    for r0 in range(0, Wh.shape[0], head_rows):
        a = Wh[r0:r0 + head_rows]
        deq = {}
        for lab, f in variants.items():
            fmt = None if f is None else f("head")
            if fmt not in deq:
                deq[fmt] = dequant(a, fmt, D)
            lg = hs[lab] @ deq[fmt].T
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
        del deq
    if logits:
        return {lab: np.concatenate(v[5], 1) for lab, v in acc.items()}
    return {lab: ((m + np.log(se))[:-1] - tl, arg) for lab, (m, se, tl, _, arg, _) in acc.items()}


def weight_bytes(shapes: dict, fmt, D: int = 128) -> int:
    """DRAM bytes a token reads of the weights (rows and block scales) under fmt(kind, layer)."""
    n = 0
    for (kind, layer), mats in shapes.items():
        if kind == "experts":
            continue
        f = fmt(kind, layer) if kind != "head" else fmt("head")
        for rows, cols in mats.values():
            n += rows * (Q.row_bytes(cols, f, D) + 4 * (cols // D))
    return n


def text_ids(path, n: int) -> list:
    """The first n tokens of docs/isa.md, as tools/quant_eval.py reads it."""
    import transformers
    tok = transformers.AutoTokenizer.from_pretrained(path)
    ids = tok(re.sub(r"`|\|", "", (ROOT / "docs/isa.md").read_text())).input_ids[:n]
    if len(ids) < n:
        raise ValueError(f"docs/isa.md has {len(ids)} tokens, {n} asked")
    return ids


def _resolver(M, formats: str, wformat: str = "int8", head: str | None = None):
    return FM.resolver(formats, M.KINDS, "", wformat, head)


def _run(spec, W, ids, specs: dict, shapes, D, log) -> dict:
    """{label: row} for specs {label: (formats, wformat, head) or None (float)}."""
    M = family(spec)
    variants = {lab: None if v is None else _resolver(M, *v) for lab, v in specs.items()}
    res = emulate(spec, W, ids, variants, D, shapes=shapes, log=log)
    out = {}
    for lab, (nll, top1) in res.items():
        v = specs[lab]
        out[lab] = {"formats": None if v is None else v[0], "wformat": None if v is None else v[1],
                    "head": None if v is None else v[2], "nll": float(nll.mean()),
                    "ppl": float(np.exp(nll.mean())), "top1": top1.tolist()}
    return out


def _finish(rows: dict, shapes, M, D) -> None:
    """Weight bytes, the bytes saved against int8, top-1 agreement with int8 and float (the
    rows' "top1" lists dropped)."""
    ref8, refF = rows["int8"]["top1"], rows["float"]["top1"]
    for r in rows.values():
        t = r.pop("top1")
        r["agree_int8"] = float(np.mean(np.equal(t, ref8)))
        r["agree_float"] = float(np.mean(np.equal(t, refF)))
        if r["wformat"] is not None:
            r["bytes"] = weight_bytes(shapes, _resolver(M, r["formats"], r["wformat"], r["head"]),
                                      D)
    for r in rows.values():
        if "bytes" in r:
            r["saved"] = rows["int8"]["bytes"] - r["bytes"]


def scan(name, out, n_tok: int, f4: str, D: int = 128) -> None:
    path = model_dir(name)
    spec, W = load_spec(path), Q3.load_weights(path)
    M = family(spec)
    ids = text_ids(path, n_tok)
    t0 = time.time()

    def log(m):
        print(f"[{time.time() - t0:6.0f} s] {m}", flush=True)

    kinds = [k for k in M.KINDS if k not in ("mlp", "head")]
    L = spec.layers
    qs = [(round(j * L / 4), round((j + 1) * L / 4) - 1) for j in range(4)]
    present = {M.weight_kind(n)[0] for n in W if n.startswith("model.layers.")}
    kinds = [k for k in kinds if k in present]
    groups = {f"{k}@{a}-{b}": f"{k}@{a}-{b}={f4}" for k in kinds for a, b in qs}
    groups["head"] = f"head={f4}"
    p1 = {"float": None, "int8": ("", "int8", None), f4: ("", f4, "int8")}
    p1.update({f"{k}={f4}": (f"{k}={f4}", "int8", None) for k in kinds})
    p1.update({g: (fs, "int8", None) for g, fs in groups.items()})
    shapes: dict = {}
    log(f"{name}: {len(ids)} tokens, pass 1: {len(p1)} variants")
    rows = _run(spec, W, ids, p1, shapes, D, log)
    nll8 = rows["int8"]["nll"]
    b8 = weight_bytes(shapes, _resolver(M, ""), D)
    saved = {g: b8 - weight_bytes(shapes, _resolver(M, groups[g]), D) for g in groups}
    gain = sorted(groups, key=lambda g: (rows[g]["nll"] - nll8) / max(saved[g], 1))
    log("order (NLL per byte saved): " + ", ".join(gain))
    p2 = {}
    for m in range(2, 1 << (len(kinds) + 1)):           # every subset of the kinds and the head
        sel = [k for j, k in enumerate(kinds + ["head"]) if m >> j & 1]
        if len(sel) > 1:
            p2["uniform " + "+".join(sel)] = (",".join(f"{k}={f4}" for k in sel), "int8", None)
    for j in range(2, len(gain) + 1):
        p2[f"greedy {j}"] = (",".join(groups[g] for g in gain[:j]), "int8", None)
    log(f"pass 2: {len(p2)} variants")
    rows.update(_run(spec, W, ids, p2, shapes, D, log))
    _finish(rows, shapes, M, D)
    Path(out).write_text(json.dumps({"model": str(name), "tokens": len(ids), "fmt": f4,
                                     "order": gain, "variants": rows}, indent=1))
    log(f"wrote {out}")
    for lab, r in sorted(rows.items(), key=lambda kv: kv[1].get("bytes", 1 << 62)):
        print(f"{lab:40s} ppl {r['ppl']:8.4f}  {r.get('bytes', 0) / 2**20:8.1f} MiB  "
              f"top1/int8 {r['agree_int8']:.3f}")


def ppl(name, formats: list, wformat: str, n_tok: int, D: int = 128) -> None:
    path = model_dir(name)
    spec, W = load_spec(path), Q3.load_weights(path)
    M = family(spec)
    ids = text_ids(path, n_tok)
    specs = {"float": None, "int8": ("", "int8", None)}
    specs.update({f or "(none)": (f, wformat, None) for f in formats})
    shapes: dict = {}
    rows = _run(spec, W, ids, specs, shapes, D, None)
    _finish(rows, shapes, M, D)
    for lab, r in rows.items():
        print(f"{lab:40s} ppl {r['ppl']:8.4f}  {r.get('bytes', 0) / 2**20:8.1f} MiB  "
              f"top1/int8 {r['agree_int8']:.3f}")


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
                nll, top1 = emulate(spec, W, toks, {"v": _resolver(M, fs)}, head_rows=256)["v"]
                lse = np.log(np.exp(ref - ref.max(1, keepdims=True)).sum(1)) + ref.max(1)
                errn = np.abs(nll - (lse[:-1] - ref[np.arange(len(toks) - 1), toks[1:]])).max()
                agree = np.mean(top1 == ref.argmax(1))
                worst = max(worst, err)
                print(f"{name:8s} {fs or '(int8)':34s} logits rel {err:.2e}  nll {errn:.2e}  "
                      f"argmax {agree:.3f}")
                assert err < 1e-6 and errn < 1e-6 and agree > 0.99, (name, fs)
            fl = emulate(spec, W, toks[:40], {"f": None}, logits=True)["f"]
            r = M.reference_logits(spec, W, toks[:40])
            print(f"{name:8s} float vs reference_logits rel "
                  f"{np.abs(fl - r).max() / np.abs(r).max():.2e}")
    print(f"OK (worst logits rel {worst:.2e})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("scan")
    a.add_argument("model")
    a.add_argument("out")
    a.add_argument("--tokens", type=int, default=900)
    a.add_argument("--fmt", default="fp4", choices=["fp4", "int4"])
    a = sub.add_parser("ppl")
    a.add_argument("model")
    a.add_argument("formats", nargs="+")
    a.add_argument("--wformat", default="int8")
    a.add_argument("--tokens", type=int, default=900)
    sub.add_parser("check")
    a = ap.parse_args()
    if a.cmd == "scan":
        scan(a.model, a.out, a.tokens, a.fmt)
    elif a.cmd == "ppl":
        ppl(a.model, a.formats, a.wformat, a.tokens)
    else:
        check()


if __name__ == "__main__":
    sys.exit(main())
