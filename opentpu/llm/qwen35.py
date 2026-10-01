"""Qwen3.5 hybrid decoder (e.g. Qwen3.5-0.8B, text only) on openTPU.

Qwen3.5 stacks two kinds of token mixers, each followed by Qwen3's pre-norm SwiGLU MLP:
  linear  Gated DeltaNet: in_proj_qkv -> a causal depthwise convolution (4 taps, SiLU) -> q, k,
          v of 16 heads (128 wide); q and k are L2-normalized. Per head a 128 x 128 fp32 state
          S is decayed by exp(g) and updated by the delta rule, S += k (beta (v - S^T k))^T,
          and read with o = S^T q; then RMSNorm(o) * w * silu(z) and out_proj. g and beta
          come from two tiny projections a and b: g = -exp(A_log) softplus(a + dt_bias),
          beta = sigmoid(b).
  attn    gated GQA attention: 8 query and 2 KV heads of 256, RMSNorm on each q and k head,
          RoPE on the first 64 dimensions of each head (theta 1e7), and an output gate: q_proj
          also yields a gate per query dimension, and the attention output is multiplied by
          sigmoid(gate) before o_proj.
All RMSNorms but the DeltaNet output norm are zero-centered: x * (1 + w). Qwen3.5-0.8B has 24
layers, (linear, linear, linear, attn) x 6.

How it maps onto openTPU (docs/qwen35.md):
  * The DeltaNet state (1 MiB per layer) lives in DRAM in fp32 and streams through TMEM one
    head at a time, double-buffered. The state is stored transposed, St[j, i] = S[i, j], so
    both reads of S are row dot products (RDOT) and the update is one in-place OUTER. The
    heads run in pairs: the small vector work of a pair is done on [2, n] tiles, and it is
    software-pipelined around the state passes (_deltanet).
  * The convolution state is a window of the pre-convolution q, k, v rows of the K - 1
    positions before, oldest first, stored per pair of heads after the pair's taps (one load);
    each step stores it back shifted by one row (_past), so a program's addresses do not
    depend on the position.
  * Fewer DeltaNet key heads than value heads (Qwen3.5-4B, 9B, 35B-A3B: 16 and 32): value
    head h takes the q and k of key head h // (value / key heads), as HF's repeat_interleave.
    The image repeats a key head's q and k rows of in_proj_qkv (and their convolution taps)
    for each of its value heads; the repeated rows cost 2 dk projection rows per extra value
    head. With an even number of value heads per key head (Spec.qk_share) a pair of heads has
    one key head, and its q and k rows are projected, convolved and normed once.
  * Resident decode (qwen3.Engine(resident=True), Image(lookup=True)): one program per
    attention bucket takes the token and the position as run arguments (qwen3.RunPos). With
    more than PAIR_LOOP pairs of heads per slice (the 4B, 9B, 35B-A3B: 16) the DeltaNet blocks
    are group-major (Image, DeltaNetParts), so it loops the pairs too and fits IMEM.
  * 256-wide attention heads are two MXU blocks. With MCOLS < 4 a query group of 4 heads is
    split in pairs, each streaming the KV head (qwen3._attention).

Pieces:
  Spec              model dimensions and layer kinds (from a Hugging Face config.json)
  reference_logits  plain numpy forward pass (the math, fp32)
  emulated_logits   float64 decode with openTPU's quantization points (see qwen3)
  Image             per-slice DRAM layout: equal-size layer blocks of either kind
  qwen35_step       the ol kernel for one decode token
  qwen35_rows       R consecutive prompt tokens per device run (chunked prefill,
                    qwen3.Engine.prefill_chunks): the projections stream once for the R rows,
                    each DeltaNet state is loaded once and updated row after row, qwen3's row
                    attention

Weights, activations and the KV cache use Qwen3's W8A8 scheme; decoding runs on qwen3.Engine
(one sequence: no batched decode).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .. import fp32 as F
from .. import quant as Q
from .. import language as ol
from ..compiler import Affine, KVDesc, QTensor, Tensor
from ..host.offload import ExpertServer, Layout
from ..isasim import Config
from ..kernels.layouts import head_parallel_attention_weights
from ..kernels.deltanet import gates, l2norm_rows
from ..kernels.lib import rmsnorm, silu
from ..kernels.mlp import _chunk
from .lfm2 import plan, run_layers
from . import generate as G
from . import moe as MO
from .qwen3 import (ATTN_BLOCK, RunPos, _attention, _attention_rows, _Bump, _fake_q, _fake_w,
                    _inputs, _inputs_rows, _lm_head, _lm_head_rows, _lookup_alloc, _lookup_build,
                    _lookup_desc, _mlp, _qdesc, _tdesc, _tok_arg, _tokens_arg, compile_decode,
                    rope_tables)

LIN, ATTN = "linear", "attn"
PAIR_LOOP = 8       # pairs of DeltaNet heads per slice that decode unrolled at a run-time position
EMBED_F32_MAX = 2 << 30     # bytes: a larger fp32 embedding table (over half the card's DRAM,
#                             Qwen3.5-4B and up) is int8, gathered from the head (Spec.embed)


# =============================================================================== model spec
@dataclass(frozen=True)
class Spec:
    hidden: int
    kinds: tuple            # LIN or ATTN per layer
    n_q: int
    n_kv: int
    head_dim: int
    rope_dim: int           # RoPE on the first rope_dim dimensions of each attention head
    lin_heads: int          # DeltaNet heads (value heads)
    lin_dk: int
    lin_dv: int
    ffn: int
    vocab: int
    conv_k: int = 4         # DeltaNet convolution taps
    eps: float = 1e-6
    theta: float = 1e7
    tied: bool = True
    eos: tuple = (248046, 248044)
    lin_kheads: int = 0     # DeltaNet key heads (0: lin_heads); value head h uses q and k of
                            # key head h // (lin_heads / lin_kheads), as HF's repeat_interleave
    embed: str = "f32"      # the embedding rows: fp32, or "int8" per D block (as qwen3.Spec:
                            # gathered on the device from the tied int8 head or a table)
    pair_loop: bool | None = None   # DeltaNet layers group-major, their pairs a hardware loop
                                    # at a run-time position too (Image; None: when a slice has
                                    # more than PAIR_LOOP pairs of heads)
    qk_share: bool | None = None    # a pair of value heads of one key head projects and convolves
                                    # its q and k once (Image; None: when the value heads per key
                                    # head are even)
    moe: MO.MoESpec | None = None   # Qwen3.5-MoE: every layer's MLP is routed experts plus a
                                    # shared expert (ffn: its width, the layer block's MLP)

    @property
    def layers(self) -> int:
        return len(self.kinds)

    @property
    def lin_nk(self) -> int:
        return self.lin_kheads or self.lin_heads

    def mlp_prefix(self, p: str) -> str:
        """The checkpoint prefix of layer prefix p's dense MLP (Qwen3.5-MoE: the shared
        expert)."""
        return p + ("mlp.shared_expert." if self.moe is not None else "mlp.")

    @staticmethod
    def from_hf(model_dir) -> "Spec":
        top = json.loads((Path(model_dir) / "config.json").read_text())
        c = top.get("text_config", top)
        rope = c.get("rope_parameters") or {}
        d = c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"]
        eos = c.get("eos_token_id", 248044)
        eos = tuple(eos) if isinstance(eos, list) else (eos,)
        moe = None
        if "num_experts" in c:                  # Qwen3.5-MoE: softmax top-k, a shared expert
            moe = MO.MoESpec(E=c["num_experts"], k=c["num_experts_per_tok"],
                             ffn=c["moe_intermediate_size"], rule="softmax",
                             norm=c.get("norm_topk_prob", True),
                             shared=c["shared_expert_intermediate_size"])
        return Spec(hidden=c["hidden_size"],
                    kinds=tuple(ATTN if t == "full_attention" else LIN for t in c["layer_types"]),
                    n_q=c["num_attention_heads"], n_kv=c["num_key_value_heads"], head_dim=d,
                    rope_dim=int(d * rope.get("partial_rotary_factor", 1.0)),
                    lin_heads=c["linear_num_value_heads"], lin_dk=c["linear_key_head_dim"],
                    lin_dv=c["linear_value_head_dim"],
                    ffn=moe.shared if moe is not None else c["intermediate_size"],
                    vocab=c["vocab_size"], conv_k=c.get("linear_conv_kernel_dim", 4),
                    eps=c.get("rms_norm_eps", 1e-6), theta=rope.get("rope_theta", 1e7),
                    tied=top.get("tie_word_embeddings", c.get("tie_word_embeddings", True)),
                    eos=(248046,) + tuple(e for e in eos if e != 248046),   # <|im_end|> first
                    lin_kheads=c["linear_num_key_heads"], moe=moe,
                    # a MoE's DRAM beside its layers is expert slots: its table int8 at any size
                    embed="int8" if moe is not None
                    or 4 * c["vocab_size"] * c["hidden_size"] > EMBED_F32_MAX else "f32")

    def check(self, cfg: Config) -> None:
        S, D = cfg.S, cfg.D
        need = [(self.head_dim % D == 0, f"head_dim {self.head_dim} % D"),
                (self.rope_dim % 2 == 0 and self.rope_dim <= self.head_dim, "rope_dim"),
                (self.lin_dk == D and self.lin_dv == D, "DeltaNet heads must be D wide"),
                (self.lin_heads % (2 * S) == 0, f"DeltaNet heads % 2S (pairs per slice)"),
                (self.lin_heads % self.lin_nk == 0, "DeltaNet value heads % key heads"),
                (self.hidden % (S * D) == 0, f"hidden {self.hidden} % S*D"),
                (self.ffn % (S * D) == 0, f"ffn {self.ffn} % S*D"),
                (self.n_kv % S == 0, f"n_kv {self.n_kv} % S"),
                (self.n_q % self.n_kv == 0, "n_q % n_kv"),
                (self.vocab % S == 0, f"vocab {self.vocab} % S"),
                (max(self.ffn, self.n_q * self.head_dim, self.hidden) <= cfg.ACT_BLOCKS * D,
                 "an inner dimension exceeds ACT RAM")]
        if self.moe is not None:
            mo = self.moe
            need += [(S == 1, "MoE layers run on one slice"),
                     (mo.ffn % D == 0, f"expert width {mo.ffn} % D"),
                     (mo.k <= mo.E, "top-k above the expert count"),
                     (mo.first == 0 and mo.shared == self.ffn,
                      "Qwen3.5-MoE: every layer routed, the dense MLP its shared expert")]
        bad = [m for ok, m in need if not ok]
        if bad:
            raise ValueError("model does not map onto this openTPU config: " + "; ".join(bad))

    def image(self, cfg: Config, cap: int, batch: int = 1, rows: int = 1,
              wformat: str = "int8", head_format: str | None = None,
              lookup: bool | str = False, experts: int | None = None) -> "Image":
        return Image(self, cfg, cap, batch, rows, wformat, head_format, lookup, experts)


# =============================================================================== reference
def _norm(v, g, eps):
    return (v / np.sqrt(np.mean(v * v, axis=-1, keepdims=True) + eps)) * g


def _l2norm(v, eps=1e-6):
    return v / np.sqrt(np.sum(v * v, axis=-1, keepdims=True) + eps)


def _silu(v):
    return v / (1 + np.exp(-v))


def _softplus(v):
    return np.maximum(v, 0) + np.log1p(np.exp(-np.abs(v)))


def _rot(v, c, s, rd):
    """Rotate-half RoPE on the first rd dimensions of the last axis."""
    h = rd // 2
    v1, v2 = v[..., :h], v[..., h:rd]
    return np.concatenate([v1 * c - v2 * s, v2 * c + v1 * s, v[..., rd:]], axis=-1)


def reference_logits(spec: Spec, W: dict, tokens) -> np.ndarray:
    """fp32 numpy forward of the whole sequence (causal); returns logits [T, vocab]."""
    tokens = list(tokens)
    T, d, G, eps, K = len(tokens), spec.head_dim, spec.n_q // spec.n_kv, spec.eps, spec.conv_k
    nh, nk, dk, dv = spec.lin_heads, spec.lin_nk, spec.lin_dk, spec.lin_dv
    f32 = np.float32
    x = W["model.embed_tokens.weight"][tokens].astype(f32)
    cs = [rope_tables(spec, p) for p in range(T)]
    cos = np.stack([c for c, _ in cs])[:, None, :]
    sin = np.stack([s for _, s in cs])[:, None, :]
    mask = np.triu(np.full((T, T), -np.inf, f32), 1)

    def g1(n):                                          # zero-centered norm weight
        return (1 + W[n]).astype(f32)

    for i, kind in enumerate(spec.kinds):
        p = f"model.layers.{i}."
        h = _norm(x, g1(p + "input_layernorm.weight"), eps)
        if kind == LIN:
            a = p + "linear_attn."
            qkv = h @ W[a + "in_proj_qkv.weight"].T                     # [T, 2dk*nh + dv*nh]
            w = W[a + "conv1d.weight"][:, 0, :]                          # [C, K]
            pad = np.concatenate([np.zeros((K - 1, qkv.shape[1]), f32), qkv])
            y = _silu(sum(pad[j:j + T] * w[:, j] for j in range(K)))
            q, k, v = np.split(y, [nk * dk, 2 * nk * dk], axis=1)
            q = np.repeat(_l2norm(q.reshape(T, nk, dk)) / f32(math.sqrt(dk)), nh // nk, axis=1)
            k = np.repeat(_l2norm(k.reshape(T, nk, dk)), nh // nk, axis=1)
            v = v.reshape(T, nh, dv)
            z = (h @ W[a + "in_proj_z.weight"].T).reshape(T, nh, dv)
            beta = 1 / (1 + np.exp(-(h @ W[a + "in_proj_b.weight"].T)))
            g = -np.exp(W[a + "A_log"]) * _softplus(h @ W[a + "in_proj_a.weight"].T
                                                    + W[a + "dt_bias"])
            S = np.zeros((nh, dk, dv), f32)
            o = np.zeros((T, nh, dv), f32)
            for t in range(T):
                S = S * np.exp(g[t])[:, None, None]
                mem = np.einsum("hij,hi->hj", S, k[t])
                delta = (v[t] - mem) * beta[t][:, None]
                S = S + k[t][:, :, None] * delta[:, None, :]
                o[t] = np.einsum("hij,hi->hj", S, q[t])
            o = _norm(o, W[a + "norm.weight"], eps) * _silu(z)
            x = x + o.reshape(T, -1) @ W[a + "out_proj.weight"].T
        else:
            a = p + "self_attn."
            qg = (h @ W[a + "q_proj.weight"].T).reshape(T, spec.n_q, 2 * d)
            q, gate = qg[..., :d], qg[..., d:]
            k = (h @ W[a + "k_proj.weight"].T).reshape(T, spec.n_kv, d)
            v = (h @ W[a + "v_proj.weight"].T).reshape(T, spec.n_kv, d)
            q = _rot(_norm(q, g1(a + "q_norm.weight"), eps), cos, sin, spec.rope_dim)
            k = _rot(_norm(k, g1(a + "k_norm.weight"), eps), cos, sin, spec.rope_dim)
            o = np.zeros((T, spec.n_q, d), f32)
            for hq in range(spec.n_q):
                s = q[:, hq] @ k[:, hq // G].T / math.sqrt(d) + mask
                s = np.exp(s - s.max(axis=1, keepdims=True))
                o[:, hq] = (s / s.sum(axis=1, keepdims=True)) @ v[:, hq // G]
            o = o / (1 + np.exp(-gate))
            x = x + o.reshape(T, -1) @ W[a + "o_proj.weight"].T
        h = _norm(x, g1(p + "post_attention_layernorm.weight"), eps)
        mp = spec.mlp_prefix(p)
        gg = h @ W[mp + "gate_proj.weight"].T
        u = h @ W[mp + "up_proj.weight"].T
        y = (_silu(gg) * u) @ W[mp + "down_proj.weight"].T
        if spec.moe is not None:                    # the shared expert, gated; the experts
            y = y / (1 + np.exp(-(h @ W[p + "mlp.shared_expert_gate.weight"].T)))
            y = _moe_reference(h, W, p, spec.moe) + y
        x = x + y
    x = _norm(x, g1("model.norm.weight"), eps)
    head = W["model.embed_tokens.weight"] if spec.tied else W["lm_head.weight"]
    return x @ head.T


def _expert(W, p: str, e: int):
    """Routed expert e of layer prefix p: W_gate, W_up [F, H], W_down [H, F] (the checkpoint's
    fused tensors; a LazyWeights reads the one expert)."""
    m = p + "mlp.experts."
    get = getattr(W, "part", None)
    gu, dn = ((get(m + "gate_up_proj", e), get(m + "down_proj", e)) if get is not None else
              (W[m + "gate_up_proj"][e], W[m + "down_proj"][e]))
    F_ = gu.shape[0] // 2
    return gu[:F_], gu[F_:], dn


def _moe_reference(h, W, p: str, mo) -> np.ndarray:
    """The routed experts' sum for the rows of h [T, H] (fp32, Hugging Face's math)."""
    logits = h @ W[p + "mlp.gate.weight"].T
    y = np.zeros_like(h)
    for t in range(h.shape[0]):
        ids, w = MO.route(logits[t], None, mo)
        for e, we in zip(ids, w):
            wg, wu, wd = _expert(W, p, e)
            y[t] += np.float32(we) * ((_silu(h[t] @ wg.T) * (h[t] @ wu.T)) @ wd.T)
    return y


def emulated_logits(spec: Spec, W: dict, tokens, D: int = 128, wformat: str = "int8",
                    head_format: str | None = None, routes: list | None = None) -> np.ndarray:
    """float64 decode with openTPU's quantization points and none of its rounding (as
    qwen3.emulated_logits): int8 weights and matmul inputs per D-block, int8 K and V, int8 P.
    The DeltaNet state, convolution and gates are exact (they are fp32 on the device).
    Qwen3.5-MoE: the router (with the shared expert's gate as its last row) in int8; `routes`
    gets (token, layer, ids, the k-th and (k+1)-th logits' gap) per MoE layer."""
    d, G, eps, K = spec.head_dim, spec.n_q // spec.n_kv, spec.eps, spec.conv_k
    nh, nk, dk, dv = spec.lin_heads, spec.lin_nk, spec.lin_dk, spec.lin_dv
    Wq: dict = {}

    head = "model.embed_tokens.weight" if spec.tied else "lm_head.weight"

    def w(n):
        if n not in Wq:        # the weight formats as in Image (wformat, head_format)
            Wq[n] = _fake_w(W[n], D, (head_format or wformat) if n == head else wformat)
        return Wq[n]

    def g1(n):
        return 1 + np.asarray(W[n], np.float64)

    lin = [i for i, k in enumerate(spec.kinds) if k == LIN]
    Kc = {i: [] for i, k in enumerate(spec.kinds) if k == ATTN}
    Vc = {i: [] for i in Kc}
    ring = {i: [np.zeros(2 * nk * dk + nh * dv)] * (K - 1) for i in lin}
    state = {i: np.zeros((nh, dk, dv)) for i in lin}
    out = []
    for pos, tk in enumerate(tokens):
        x = np.asarray(W["model.embed_tokens.weight"][tk], np.float64)
        if spec.embed == "int8":                    # the int8 embedding rows (Spec.embed)
            x = _fake_q(x, D)
        c, s = rope_tables(spec, pos)
        for i, kind in enumerate(spec.kinds):
            p = f"model.layers.{i}."
            h = _fake_q(_norm(x, g1(p + "input_layernorm.weight"), eps), D)
            if kind == LIN:
                a = p + "linear_attn."
                qkv = w(a + "in_proj_qkv.weight") @ h
                win = ring[i] + [qkv]
                ring[i] = win[1:]
                wc = W[a + "conv1d.weight"][:, 0, :]
                y = _silu(sum(win[j] * wc[:, j] for j in range(K)))
                q, k, v = np.split(y, [nk * dk, 2 * nk * dk])
                q = np.repeat(_l2norm(q.reshape(nk, dk)) / math.sqrt(dk), nh // nk, axis=0)
                k = np.repeat(_l2norm(k.reshape(nk, dk)), nh // nk, axis=0)
                v = v.reshape(nh, dv)
                z = (w(a + "in_proj_z.weight") @ h).reshape(nh, dv)
                beta = 1 / (1 + np.exp(-(w(a + "in_proj_b.weight") @ h)))
                g = -np.exp(np.asarray(W[a + "A_log"], np.float64)) * _softplus(
                    w(a + "in_proj_a.weight") @ h + W[a + "dt_bias"])
                S = state[i] * np.exp(g)[:, None, None]
                delta = (v - np.einsum("hij,hi->hj", S, k)) * beta[:, None]
                S = S + k[:, :, None] * delta[:, None, :]
                state[i] = S
                o = np.einsum("hij,hi->hj", S, q)
                o = _norm(o, np.asarray(W[a + "norm.weight"], np.float64), eps) * _silu(z)
                x = x + w(a + "out_proj.weight") @ _fake_q(o.reshape(-1), D)
            else:
                a = p + "self_attn."
                qg = (w(a + "q_proj.weight") @ h).reshape(spec.n_q, 2 * d)
                q, gate = qg[:, :d], qg[:, d:]
                k = (w(a + "k_proj.weight") @ h).reshape(spec.n_kv, d)
                v = (w(a + "v_proj.weight") @ h).reshape(spec.n_kv, d)
                q = _rot(_norm(q, g1(a + "q_norm.weight"), eps), c, s, spec.rope_dim)
                k = _rot(_norm(k, g1(a + "k_norm.weight"), eps), c, s, spec.rope_dim)
                Kc[i].append(_fake_q(k, D))
                Vc[i].append(_fake_q(v, d))
                Kh, Vh = np.stack(Kc[i], 1), np.stack(Vc[i], 1)
                o = np.zeros((spec.n_q, d))
                for hq in range(spec.n_q):
                    sc = Kh[hq // G] @ _fake_q(q[hq] / math.sqrt(d), D)
                    pp = np.exp(sc - sc.max())
                    T = len(pp)
                    ppad = np.zeros(-(-T // D) * D)
                    ppad[:T] = pp
                    o[hq] = (_fake_q(ppad, D)[:T] @ Vh[hq // G]) / pp.sum()
                o = o / (1 + np.exp(-gate))
                x = x + w(a + "o_proj.weight") @ _fake_q(o.reshape(-1), D)
            h = _fake_q(_norm(x, g1(p + "post_attention_layernorm.weight"), eps), D)
            mp = spec.mlp_prefix(p)
            gg = w(mp + "gate_proj.weight") @ h
            u = w(mp + "up_proj.weight") @ h
            y = w(mp + "down_proj.weight") @ _fake_q(_silu(gg) * u, D)
            if spec.moe is not None:
                mo, rn = spec.moe, p + "mlp.router"
                if rn not in Wq:                    # int8 in every format, the gate row last
                    Wq[rn] = _fake_w(np.concatenate([W[p + "mlp.gate.weight"],
                                                     W[p + "mlp.shared_expert_gate.weight"]]),
                                     D, "int8")
                lg = Wq[rn] @ h
                y = y / (1 + np.exp(-lg[mo.E]))
                ids, wts = MO.route(lg[:mo.E], None, mo)
                if routes is not None:
                    srt = np.sort(lg[:mo.E])[::-1]
                    routes.append((pos, i, ids, srt[mo.k - 1] - srt[mo.k] if mo.k < mo.E
                                   else np.inf))
                for e, we in zip(ids, wts):
                    key = f"{p}mlp.experts.{e}"
                    if key not in Wq:
                        Wq[key] = [_fake_w(a, D, wformat) for a in _expert(W, p, e)]
                    wg, wu, wd = Wq[key]
                    x = x + we * (wd @ _fake_q(_silu(wg @ h) * (wu @ h), D))
            x = x + y
        out.append(w(head) @ _fake_q(_norm(x, g1("model.norm.weight"), eps), D))
    return np.array(out)


# =============================================================================== DRAM image
class DeltaNetParts:
    """Where the kernels find a DeltaNet layer's per-pair parts: pair p's projection rows
    (wh, [2R, H]: q k v of a, of b, then z of a, of b), taps and window (cv, [CVW]), the
    states of its heads a, b ([dv, dk] each) and its gates of this token (gates, [4]), and
    head group g's out_proj block (wout, [H, og dv]). p is an int, a loop expression or (g,
    i): pair i of head group g.

    Array-major (grouped False): each part one array over the slice's pairs (wh, cv, state,
    wout: the layer's full descriptors; the gates in the I/O area, hs). Group-major: head
    group g's block (stride gs bytes) holds its og / 2 pairs' blocks (stride ps: projection
    rows, their scales, taps and window, the two states), its out_proj block and its pairs'
    gates, so every part of pair p is at one offset of p's group's block and a loop over the
    pairs steps one address register (wh, cv, state, wout, eb: the parts of pair 0, group
    0).

    shared (Image.shared): the pair's two value heads have one key head; its rows are the key
    head's q and k once, then v of a, of b, then z of a, of b, and its taps rows (block, tap) of
    four blocks of dk (q, k, v a, v b) instead of two of 2 dk + dv (the heads)."""

    def __init__(self, nl: int, og: int, wh: QTensor, cv: Tensor, state: Tensor,
                 wout: QTensor, H: int, gs: int = 0, ps: int = 0, eb: Tensor | None = None,
                 shared: bool = False):
        self.nl, self.og, self.gp, self.H, self.shared = nl, og, og // 2, H, shared
        self.grouped, self.gs, self.ps = gs > 0, gs, ps
        self._wh, self._cv, self._state, self._wout, self._eb = wh, cv, state, wout, eb
        self.R2 = wh.shape[0] if self.grouped else wh.shape[0] // (nl // 2)

    def split(self, p) -> tuple:
        """(g, i) of pair p (a loop expression: its loop terms multiples of og / 2)."""
        if isinstance(p, tuple):
            return p
        if isinstance(p, int):
            return divmod(p, self.gp)
        i = Affine.of(p).const % self.gp
        return (Affine.of(p) - i).div_exact(self.gp), i

    def index(self, p):
        """The pair's number in the slice."""
        if not isinstance(p, tuple):
            return p
        g, i = p
        return g * self.gp + i if isinstance(g, int) and isinstance(i, int) else \
            Affine.of(g) * self.gp + i

    def _off(self, p) -> Affine:
        g, i = self.split(p)
        return Affine.of(g) * self.gs + Affine.of(i) * self.ps

    def wh(self, p) -> QTensor:
        if not self.grouped:
            q = self.index(p)
            return self._wh[q * self.R2:q * self.R2 + self.R2, :]
        w, o = self._wh, self._off(p)
        return QTensor(w.data + o, w.scale + o, w.shape, w.rs, w.srs, w.D, wf=w.wf)

    def cv(self, p) -> Tensor:
        if not self.grouped:
            return self._cv[self.index(p)]
        c = self._cv
        return Tensor(c.base + self._off(p), c.shape, c.strides)

    def state(self, p, a: int) -> Tensor:
        """The state of head a (0, 1) of pair p."""
        if not self.grouped:
            return self._state[2 * self.index(p) + a]
        s = self._state
        return Tensor(s.base + self._off(p), s.shape, s.strides)[a]

    def wout(self, g) -> QTensor:
        if not self.grouped:
            return self._wout[g * self.H:(g + 1) * self.H, :]
        w, o = self._wout, Affine.of(g) * self.gs
        return QTensor(w.data + o, w.scale + o, w.shape, w.rs, w.srs, w.D, wf=w.wf)

    def _gates(self, g) -> Tensor:
        e = self._eb
        return Tensor(e.base + Affine.of(g) * self.gs, e.shape, e.strides)

    def gates(self, p, hs: Tensor) -> Tensor:
        """Pair p's gates [decay a, decay b, beta a, beta b] (hs: the I/O area's, array-major).
        Group-major they sit in the group's block: a loop over the pairs addresses them with
        the register of the pairs' other parts, not one of their own."""
        if not self.grouped:
            return hs[self.index(p), :]
        g, i = self.split(p)
        return self._gates(g)[i, :]

    def store_gates(self, eb, hs: Tensor) -> None:
        """Store every pair's gates, eb [pairs, 4] in TMEM (one store per head group,
        group-major)."""
        if not self.grouped:
            ol.store(hs, eb)
            return
        for g in range(self.nl // self.og):
            ol.store(self._gates(g), eb[g * self.gp:(g + 1) * self.gp, :])


class Image:
    """Per-slice DRAM layout of a Qwen3.5 model. Every slice uses the same addresses.

    [ I/O: x_in, cos, sin | final norm | logits | per-pair gates ] [ layer 0 block ] ...
    [ layer L-1 block ] [ LM head rows of this slice ]. All layer blocks have one size: both
    kinds start with the norms and this slice's MLP rows. A DeltaNet block then holds, for
    this slice's heads (a contiguous range), the projections pair by pair (the q, k, v rows of
    head 0, of head 1, then the z rows of heads 0 and 1; then heads 2 and 3, ...; with shared q
    and k, the pair's key head's q, k, then v of each head, then z of each), the a and b
    rows, out_proj as one [H, og * dv] column block per og heads, per pair the convolution taps
    and then the convolution ring, the recurrent state (per head [dv, dk] fp32, transposed) and
    the per-head constants (the window: K - 1 rows, _past). An attention block holds the q/k
    norms, the projections (the gate rows of q_proj as their own matrix) and this slice's KV
    heads with room for `cap` tokens. The I/O area holds `rows` token rows (x, cos, sin,
    logits) for chunked prefill.

    Group-major DeltaNet blocks (Spec.pair_loop; by default when a slice has more than
    PAIR_LOOP pairs of heads): the projections, the out_proj blocks, the taps and windows, the
    states and the per-pair gates go per head group instead, each group's pairs (a pair's
    projection rows, their scales, its taps and window, its two states), then the group's
    out_proj block and its pairs' gates (DeltaNetParts). A decode's loop over the pairs then steps one address register for all of
    them, so it fits the registers beside a run-time position's (resident decode), and the
    unrolled pairs of a bigger model need not fit IMEM.
    """

    def __init__(self, spec: Spec, cfg: Config, cap: int, batch: int = 1, rows: int = 1,
                 wformat: str = "int8", head_format: str | None = None, lookup: bool | str = False,
                 experts: int | None = None):
        spec.check(cfg)
        if batch != 1:
            raise ValueError("Qwen3.5 runs one sequence: batch=1")
        if cap % cfg.D:
            raise ValueError("KV capacity must be a multiple of D")
        S, D = cfg.S, cfg.D
        H, d, F_, K = spec.hidden, spec.head_dim, spec.ffn, spec.conv_k
        self.wformat, self.head_format = wformat, head_format or wformat
        rb = lambda k: Q.row_bytes(k, wformat, D)                       # noqa: E731
        dk, dv = spec.lin_dk, spec.lin_dv
        self.spec, self.cfg, self.cap, self.batch, self.rows = spec, cfg, cap, 1, rows
        self.nq_loc, self.nkv_loc = spec.n_q // S, spec.n_kv // S
        self.h_loc, self.f_loc, self.v_loc = H // S, F_ // S, spec.vocab // S
        self.nl = spec.lin_heads // S                   # DeltaNet heads of one slice
        self.C = 2 * dk + dv                            # convolved channels per head (q, k, v)
        self.R = self.C + dv                            # projected rows per head (and z)
        rk = spec.lin_heads // spec.lin_nk
        self.shared = rk % 2 == 0 if spec.qk_share is None else spec.qk_share
        if self.shared and rk % 2:
            raise ValueError(f"{rk} value heads per key head: a pair shares its q and k only "
                             "when they are even")
        # a pair's convolved channels, as blocks of equal width: q k v of a, of b; or shared, the
        # key head's q, k, then v of a, of b (dk = dv)
        self.nb, self.bw = (4, dk) if self.shared else (2, self.C)
        self.CP = self.nb * self.bw
        self.RP = self.CP + 2 * dv                      # a pair's projected rows (then z of a, b)
        self.CVW = K * self.CP + (K - 1) * self.CP      # a pair's taps, then its window
        self.plan = plan(spec.kinds)
        b = _Bump()
        self.io = {"x": b.alloc(4 * H * rows), "cos": b.alloc(2 * spec.rope_dim * rows),
                   "sin": b.alloc(2 * spec.rope_dim * rows), "gf": b.alloc(4 * H),
                   "logits": b.alloc(4 * spec.vocab * rows),
                   "hs": b.alloc(4 * 2 * self.nl),     # per pair: decays of a, b; betas
                   "gr": b.alloc(4 * 2 * self.nl * rows),  # chunked prefill, per row: decays,
                   "on": b.alloc(4 * 4 * spec.lin_dv * rows)}  # betas; a head group's outputs
        self.layer0 = b.next
        lb = _Bump()                                    # offsets inside one layer block
        common = {"g_in": lb.alloc(4 * H), "g_post": lb.alloc(4 * H)}
        mlp = {"wg": (self.f_loc, H), "wu": (self.f_loc, H)}
        for name, (n, k) in mlp.items():
            common[name] = (lb.alloc(n * rb(k)), lb.alloc(4 * n * (k // D)))
        self.dchunk = _chunk(self.f_loc, D, D if wformat == "int8" else 2 * D)
        common["wd"] = [(lb.alloc(self.h_loc * rb(self.dchunk)),
                         lb.alloc(4 * self.h_loc * (self.dchunk // D)))
                        for _ in range(F_ // self.dchunk)]
        mo = spec.moe
        if mo is not None:          # the router (int8, the shared expert's gate its last row)
            common.update(router=(lb.alloc((mo.E + 1) * H), lb.alloc(4 * (mo.E + 1) * (H // D))),
                          gbase=lb.alloc(4))
        nl, C = self.nl, self.C
        self.og = 4 if nl % 4 == 0 else 2               # heads per out_proj MM
        self.mats = {LIN: {"wh": (nl // 2 * self.RP, H), "wab": (2 * nl, H),
                           "wout": (nl // self.og * H, self.og * dv),
                           **mlp},
                     ATTN: {"wq": (self.nq_loc * d, H), "wgate": (self.nq_loc * d, H),
                            "wk": (self.nkv_loc * d, H), "wv": (self.nkv_loc * d, H),
                            "wo": (self.h_loc, spec.n_q * d), **mlp}}
        lnb = _Bump(lb.next)
        lin = dict(common, alog=lnb.alloc(4 * nl), dtb=lnb.alloc(4 * nl), gn=lnb.alloc(4 * dv))
        self.grouped = nl // 2 > PAIR_LOOP if spec.pair_loop is None else spec.pair_loop
        if self.grouped:            # per head group: its pairs' blocks, then its out_proj block
            og = self.og
            del self.mats[LIN]["wh"], self.mats[LIN]["wout"]
            pb, gb = _Bump(), _Bump()
            self.pofs = {"wh": (pb.alloc(self.RP * rb(H)), pb.alloc(4 * self.RP * (H // D))),
                         "cv": pb.alloc(4 * self.CVW), "state": pb.alloc(4 * 2 * dv * dk)}
            self.PS = pb.next                           # a pair's block, bytes
            for _ in range(og // 2):
                gb.alloc(self.PS)
            self.gofs = {"wout": (gb.alloc(H * rb(og * dv)), gb.alloc(4 * H * (og * dv // D))),
                         "eb": gb.alloc(4 * 4 * (og // 2))}     # its pairs' gates, per token
            self.GS = gb.next                           # a head group's block
            lin["groups"] = lnb.alloc(nl // og * self.GS)
        else:
            lin.update(cv=lnb.alloc(4 * nl // 2 * self.CVW),  # per pair: taps, then the ring
                       state=lnb.alloc(4 * nl * dv * dk))
        ab = _Bump(lb.next)
        attn = dict(common, qn=ab.alloc(4 * d), kn=ab.alloc(4 * d))
        for kind, bump, L in ((LIN, lnb, lin), (ATTN, ab, attn)):
            for name, (n, k) in self.mats[kind].items():
                if name not in L:
                    L[name] = (bump.alloc(n * rb(k)), bump.alloc(4 * n * (k // D)))
        attn["kv"] = [{"k": ab.alloc(cap * d), "ks": ab.alloc(4 * cap * (d // D)),
                       "vt": ab.alloc(d * cap), "vs": ab.alloc(4 * cap)}
                      for _ in range(self.nkv_loc)]
        self.lofs = {LIN: lin, ATTN: attn}
        self.LS = (max(lnb.next, ab.next) + 4095) // 4096 * 4096
        n_attn = spec.kinds.count(ATTN)
        head = cap * d + 4 * cap * (d // D) + d * cap + 4 * cap
        self.kv_bytes = (n_attn * self.nkv_loc * head                   # KV cache, conv ring
                         + (spec.layers - n_attn) * 4 * (nl // 2 * (K - 1) * self.CP
                                                         + nl * dv * dk))  # and state
        b.next = self.layer0 + spec.layers * self.LS
        self.head = (b.alloc(self.v_loc * Q.row_bytes(H, self.head_format, D)),
                     b.alloc(4 * self.v_loc * (H // D)))
        # the int8 embedding rows of the resident decode are the tied int8 head's (S = 1)
        shared = spec.tied and self.head_format == "int8" and S == 1
        self.lookup = _lookup_alloc(b, spec, cap, D=D, head=self.head if shared else None,
                                    M=cfg.MCOLS) if lookup else {}
        self.offload = None
        if mo is not None:          # path (a)'s words and expert slots (docs/offload.md)
            self.fmt = MO.ExpertFormat(H, mo.ffn, D, wformat)
            n = mo.E if experts is None else experts
            if not mo.k <= n <= mo.E:
                raise ValueError(f"{n} expert slots per layer: from top-k {mo.k} to {mo.E}")
            self.offload = Layout.build((b.next + 4095) // 4096 * 4096, mo.E, mo.k,
                                        [n] * spec.layers, self.fmt.nbytes)
            b.next = self.offload.end
        self.nbytes = b.next
        if self.nbytes > cfg.DRAM_BYTES:
            raise MemoryError(f"model image needs {self.nbytes / 2**20:.0f} MiB per slice, "
                              f"DRAM_BYTES is {cfg.DRAM_BYTES / 2**20:.0f} MiB")

    def pair_offset(self, q: int) -> int:
        """Pair q's block in a group-major DeltaNet layer block (bytes from its start)."""
        g, i = divmod(q, self.og // 2)
        return self.lofs[LIN]["groups"] + g * self.GS + i * self.PS

    def cv_offset(self, q: int) -> int:
        """Pair q's taps, then its window, in a DeltaNet layer block (bytes from its start)."""
        if self.grouped:
            return self.pair_offset(q) + self.pofs["cv"]
        return self.lofs[LIN]["cv"] + 4 * q * self.CVW

    # ---- contents
    def build(self, W: dict) -> list[np.ndarray]:
        """DRAM images (one per slice) with every weight quantized in place; KV cache,
        convolution ring and DeltaNet state empty."""
        spec, cfg = self.spec, self.cfg
        S, D, d, H, n = cfg.S, cfg.D, spec.head_dim, spec.hidden, self.h_loc
        dk, dv, nl, K = spec.lin_dk, spec.lin_dv, self.nl, spec.conv_k
        imgs = [np.zeros(self.nbytes, np.uint8) for _ in range(S)]

        def put(s, addr, a):
            v = np.ascontiguousarray(a).view(np.uint8).reshape(-1)
            imgs[s][addr:addr + v.size] = v

        def put_q1(s, addr_pair, a, fmt=self.wformat):
            q, sc = Q.quantize_mxu(a, fmt, D)
            put(s, addr_pair[0], q)
            put(s, addr_pair[1], sc)

        def put_q(addr_pair, parts, fmt=self.wformat):
            for s, p in enumerate(parts):
                put_q1(s, addr_pair, p, fmt)

        def rows(a, k):
            return [a[s * k:(s + 1) * k] for s in range(S)]

        def f32(a):
            return F.ftz(np.asarray(a, np.float32))

        def g1(name):                               # zero-centered norm weight, 1 + w
            return f32(1 + np.asarray(W[name], np.float32))

        for s in range(S):
            put(s, self.io["gf"], g1("model.norm.weight"))
        NK, rk = spec.lin_nk * dk, spec.lin_heads // spec.lin_nk
        for i, kind in enumerate(spec.kinds):
            p, base = f"model.layers.{i}.", self.layer0 + i * self.LS
            Lo = {k: (tuple(base + x for x in v) if isinstance(v, tuple) else
                      (base + v if isinstance(v, int) else v)) for k, v in self.lofs[kind].items()}
            for s in range(S):
                put(s, Lo["g_in"], g1(p + "input_layernorm.weight"))
                put(s, Lo["g_post"], g1(p + "post_attention_layernorm.weight"))
            if kind == LIN:
                a = p + "linear_attn."
                qkv, wz = W[a + "in_proj_qkv.weight"], W[a + "in_proj_z.weight"]
                wout = W[a + "out_proj.weight"]
                # the convolved channels of head h: the q and k rows of its key head h // rk
                # (a key head's rows repeat for each of its value heads), its v rows
                chans = [np.r_[h // rk * dk:(h // rk + 1) * dk,
                               NK + h // rk * dk:NK + (h // rk + 1) * dk,
                               2 * NK + h * dv:2 * NK + (h + 1) * dv]
                         for h in range(spec.lin_heads)]
                taps = W[a + "conv1d.weight"][:, 0, :]                   # [channels, K]
                hs = [range(s * nl, (s + 1) * nl) for s in range(S)]
                # per pair of heads (a, b): the q, k, v rows of a, of b, then the z rows of a, of
                # b; its taps; per head group, its out_proj column block (og heads)
                og = self.og
                # (shared: the key head's q and k rows, then v of a, of b; taps by block)
                blocks = ((lambda h: [chans[h][:2 * dk], chans[h][2 * dk:], chans[h + 1][2 * dk:]])
                          if self.shared else (lambda h: [chans[h], chans[h + 1]]))
                prows = [[np.concatenate([qkv[c] for c in blocks(h)] + [wz[h * dv:(h + 2) * dv]])
                          for h in hh[::2]] for hh in hs]
                ptaps = [[f32(np.concatenate([taps[c].reshape(-1, self.bw, K).transpose(0, 2, 1)
                                              .reshape(-1, self.bw) for c in blocks(h)]))
                          for h in hh[::2]] for hh in hs]      # rows (block, tap)
                gouts = [[wout[:, h * dv:(h + og) * dv] for h in hh[::og]] for hh in hs]
                if self.grouped:
                    for s in range(S):
                        for q, (pr, pt) in enumerate(zip(prows[s], ptaps[s])):
                            o = base + self.pair_offset(q)
                            put_q1(s, (o + self.pofs["wh"][0], o + self.pofs["wh"][1]), pr)
                            put(s, o + self.pofs["cv"], pt)
                        for g, go in enumerate(gouts[s]):
                            o = Lo["groups"] + g * self.GS
                            put_q1(s, (o + self.gofs["wout"][0], o + self.gofs["wout"][1]), go)
                else:
                    put_q(Lo["wh"], [np.concatenate(pr) for pr in prows])
                    put_q(Lo["wout"], [np.concatenate(go) for go in gouts])
                    for s in range(S):
                        for q, pt in enumerate(ptaps[s]):
                            put(s, Lo["cv"] + q * 4 * self.CVW, pt)
                put_q(Lo["wab"], [np.concatenate([W[a + "in_proj_a.weight"][hh.start:hh.stop],
                                                  W[a + "in_proj_b.weight"][hh.start:hh.stop]])
                                  for hh in hs])
                for s, hh in enumerate(hs):
                    put(s, Lo["alog"], f32(W[a + "A_log"][hh.start:hh.stop]))
                    put(s, Lo["dtb"], f32(W[a + "dt_bias"][hh.start:hh.stop]))
                    put(s, Lo["gn"], f32(W[a + "norm.weight"]))
            else:
                a = p + "self_attn."
                for s in range(S):
                    put(s, Lo["qn"], g1(a + "q_norm.weight"))
                    put(s, Lo["kn"], g1(a + "k_norm.weight"))
                qg = W[a + "q_proj.weight"].reshape(spec.n_q, 2, d, H)
                args = (W[a + "k_proj.weight"], W[a + "v_proj.weight"], W[a + "o_proj.weight"],
                        spec.n_q, spec.n_kv, d, S)
                wq, wk, wv, wo = head_parallel_attention_weights(qg[:, 0].reshape(-1, H), *args)
                wgate = head_parallel_attention_weights(qg[:, 1].reshape(-1, H), *args)[0]
                put_q(Lo["wq"], rows(wq, self.nq_loc * d))
                put_q(Lo["wgate"], rows(wgate, self.nq_loc * d))
                put_q(Lo["wk"], rows(wk, self.nkv_loc * d))
                put_q(Lo["wv"], rows(wv, self.nkv_loc * d))
                put_q(Lo["wo"], rows(wo, n))
            mp = spec.mlp_prefix(p)
            put_q(Lo["wg"], rows(W[mp + "gate_proj.weight"], self.f_loc))
            put_q(Lo["wu"], rows(W[mp + "up_proj.weight"], self.f_loc))
            C_ = self.dchunk
            for j, pair in enumerate(self.lofs[kind]["wd"]):
                put_q((base + pair[0], base + pair[1]),
                      [r[:, j * C_:(j + 1) * C_] for r in rows(W[mp + "down_proj.weight"], n)])
            if spec.moe is not None:
                put_q(Lo["router"], [np.concatenate([W[p + "mlp.gate.weight"],
                                                     W[p + "mlp.shared_expert_gate.weight"]])],
                      "int8")
                put(0, Lo["gbase"], f32([i * spec.moe.E]))
        head = W["model.embed_tokens.weight"] if spec.tied else W["lm_head.weight"]
        put_q(self.head, rows(head, self.v_loc), self.head_format)
        if self.lookup:
            _lookup_build(put, S, W, spec, self.cap, self.lookup)
        return imgs

    # ---- path (a): the expert pool and its server (docs/offload.md)
    def expert(self, W: dict, g: int) -> np.ndarray:
        """Global expert g (layer g // E, expert g % E) in its slot's bytes."""
        E = self.spec.moe.E
        return self.fmt.pack(*_expert(W, f"model.layers.{g // E}.", g % E))

    def serve(self, W: dict, backend, pool_file=None) -> ExpertServer:
        """The host's expert server on the backend's DRAM (moe.serve)."""
        return MO.serve(self.offload, lambda g: self.expert(W, g), backend, pool_file)

    # ---- programs
    def compile_decode(self, blocks: int, lo: int, block: int = ATTN_BLOCK):
        """(programs, run_args): qwen35_step at a run-time position (qwen3.compile_decode); the
        convolutions need lo >= conv_k - 1 (every tap of the ring is a past token)."""
        return compile_decode(self, qwen35_step, blocks, lo, block)

    def compile_generate(self, blocks: int, lo: int, block: int = ATTN_BLOCK,
                         chain: bool = True, samp=None, debug: bool = False,
                         part: int | None = None) -> list:
        """The decode loop on the device for bucket `blocks` (qwen35_step in it, generate.py)."""
        return G.compile_generate(self, qwen35_step, blocks, lo, block, chain, samp, debug, part)

    def compile_step(self, pos: int, block: int = ATTN_BLOCK, tok: int | None = None) -> list:
        """One program per slice: the decode token at position `pos` (qwen35_step)."""
        return [qwen35_step.trace(self.cfg, s, {"m": self.descriptors(s), "pos": pos,
                                                "block": block, **_tok_arg(self, tok)}).finish()
                for s in range(self.cfg.S)]

    def compile_rows(self, rows, logit_rows, block: int = ATTN_BLOCK, tokens=None) -> list:
        """One program per slice: consecutive positions of the sequence at once
        (qwen35_rows)."""
        if self.spec.moe is not None:
            raise ValueError("a MoE model runs its prompt through the decode step (its MoE "
                             "block routes one token): rows=1")
        if len(rows) > self.rows:
            raise ValueError(f"{len(rows)} rows, the image's I/O area holds {self.rows}")
        if any(r != (0, rows[0][1] + i) for i, r in enumerate(rows)):
            raise ValueError("Qwen3.5 rows must be consecutive positions of sequence 0")
        return [qwen35_rows.trace(self.cfg, s, {"m": self.descriptors(s), "p0": rows[0][1],
                                                "R": len(rows), "logit_rows": list(logit_rows),
                                                "block": block,
                                                **_tokens_arg(self, tokens, rows)}).finish()
                for s in range(self.cfg.S)]

    # ---- kernel descriptors
    def descriptors(self, sid: int) -> SimpleNamespace:
        spec, cfg = self.spec, self.cfg
        D, d, H, K, n = cfg.D, spec.head_dim, spec.hidden, spec.conv_k, self.h_loc
        dk, dv, nl, C = spec.lin_dk, spec.lin_dv, self.nl, self.C

        def layer(li, kind):
            """Descriptors of layer `li` (an int or a hardware-loop expression) of `kind`."""
            off = Affine.of(self.layer0) + Affine.of(li) * self.LS
            lofs = self.lofs[kind]
            fm, wf = self.wformat, Q.mxu_wf(self.wformat)
            ns = SimpleNamespace(g_in=Tensor(off + lofs["g_in"], (H,), (1,)),
                                 g_post=Tensor(off + lofs["g_post"], (H,), (1,)),
                                 moe=spec.moe is not None)
            if ns.moe:
                E = spec.moe.E
                da, sa = lofs["router"]
                ns.router = QTensor(off + da, off + sa, (E + 1, H), H, 4 * (H // D), D)
                ns.gbase = Tensor(off + lofs["gbase"], (1,), (1,))
            for name, (r, k) in self.mats[kind].items():
                da, sa = lofs[name]
                setattr(ns, name, QTensor(off + da, off + sa, (r, k), Q.row_bytes(k, fm, D),
                                          4 * (k // D), D, wf=wf))
            Cd = self.dchunk
            rc = Q.row_bytes(Cd, fm, D)
            parts = tuple(QTensor(off + da, off + sa, (n, Cd), rc, 4 * (Cd // D), D, wf=wf)
                          for da, sa in lofs["wd"])
            ns.wd = QTensor(parts[0].data, parts[0].scale, (n, spec.ffn), rc, 4 * (Cd // D), D,
                            parts=parts, pw=Cd, wf=wf)
            if kind == LIN:
                ns.alog = Tensor(off + lofs["alog"], (nl,), (1,))
                ns.dtb = Tensor(off + lofs["dtb"], (nl,), (1,))
                ns.gn = Tensor(off + lofs["gn"], (dv,), (1,))
                og, R2 = self.og, self.RP
                if self.grouped:            # the parts of pair 0 (group 0)
                    g0 = off + lofs["groups"]
                    (whd, whs), (wod, wos) = self.pofs["wh"], self.gofs["wout"]
                    ns.dn = DeltaNetParts(
                        nl, og, QTensor(g0 + whd, g0 + whs, (R2, H), Q.row_bytes(H, fm, D),
                                        4 * (H // D), D, wf=wf),
                        Tensor(g0 + self.pofs["cv"], (self.CVW,), (1,)),
                        Tensor(g0 + self.pofs["state"], (2, dv, dk), (dv * dk, dk, 1)),
                        QTensor(g0 + wod, g0 + wos, (H, og * dv), Q.row_bytes(og * dv, fm, D),
                                4 * (og * dv // D), D, wf=wf), H, self.GS, self.PS,
                        Tensor(g0 + self.gofs["eb"], (og // 2, 4), (4, 1)), self.shared)
                else:
                    ns.cv = Tensor(off + lofs["cv"], (nl // 2, self.CVW), (self.CVW, 1))
                    ns.state = Tensor(off + lofs["state"], (nl, dv, dk), (dv * dk, dk, 1))
                    ns.dn = DeltaNetParts(nl, og, ns.wh, ns.cv, ns.state, ns.wout, H,
                                          shared=self.shared)
            else:
                ns.qn = Tensor(off + lofs["qn"], (d,), (1,))
                ns.kn = Tensor(off + lofs["kn"], (d,), (1,))
                ns.kv = KVDesc({sid + j * cfg.S: {k: off + v for k, v in r.items()}
                                for j, r in enumerate(lofs["kv"])}, self.cap, d, D, cfg.S, sid)
                ns.kvs = [ns.kv]
            return ns

        dev = None
        if self.offload is not None:
            L = self.offload
            dev = SimpleNamespace(mbox=L.mbox, served=L.served, dir=L.dir, fmt=self.fmt)
        return SimpleNamespace(
            spec=spec, layer=layer, plan=self.plan, moe_dev=dev,
            x=_tdesc(self.io["x"], (1, H)), cos=_tdesc(self.io["cos"], (spec.rope_dim // 2,)),
            sin=_tdesc(self.io["sin"], (spec.rope_dim // 2,)), g_final=_tdesc(self.io["gf"], (H,)),
            logits=_tdesc(self.io["logits"], (1, spec.vocab)),
            xr=_tdesc(self.io["x"], (self.rows, H)),
            cosr=_tdesc(self.io["cos"], (self.rows, spec.rope_dim // 2)),
            sinr=_tdesc(self.io["sin"], (self.rows, spec.rope_dim // 2)),
            logitsr=_tdesc(self.io["logits"], (self.rows, spec.vocab)),
            hs=_tdesc(self.io["hs"], (nl // 2, 4)),
            gr=_tdesc(self.io["gr"], (self.rows, 2 * nl)),
            on=_tdesc(self.io["on"], (self.rows, self.og * dv)),
            head=_qdesc(*self.head, self.v_loc, H, D, self.head_format), v_loc=self.v_loc,
            **_lookup_desc(self.lookup, spec, self.cap))


# =============================================================================== kernel
def _past(K: int, pos) -> int:
    """How many positions before pos the convolution reads (all K - 1 at a run-time one). The
    window (a pair's convolution rows of the K - 1 positions before, oldest first, after its
    taps in DRAM) holds position pos - j in row K - 1 - j; rows of positions below 0 hold
    nothing useful. Each step stores it shifted by one row, its own rows last, so it does not
    depend on the position (resident decode)."""
    if isinstance(pos, RunPos):
        if pos.lo < K - 1:
            raise ValueError(f"a run-time position needs p >= {K - 1} (conv taps)")
        return K - 1
    return min(K - 1, pos)
def _deltanet(x, lw, pos: int, spec: Spec, hs):
    """x + out_proj(Gated DeltaNet(x)) for one token, this slice's heads; returns the new
    residual (replicated on every slice).

    First the decay exp(g) and beta of every head (kernels.deltanet.gates) go to DRAM (`hs`,
    or the head groups' blocks: DeltaNetParts.gates), [decay a, decay b, beta a, beta b] per
    pair of heads. The heads then run in pairs (a, b). The small vector work of a pair runs on
    [2, n] tiles: the convolution, SiLU, L2 norms and silu(z) before the recurrence ("prep"),
    the gated RMSNorm after it ("post"). Per head the recurrence is RDOT, OUTER, RDOT over its
    fp32 state (stored transposed; kernels.deltanet), which streams through TMEM in two
    buffers, one per head of the pair.

    The schedule is built around one fact: a TMEM bank takes one write per cycle, the DMA and
    the MXU go first, and a VPU op that writes TMEM stalls while they write its banks. An RDOT
    writes only its row sums, at its end. So the DMA's state loads (the only big TMEM writes)
    should run beside RDOTs, and each state pass is split in halves of 64 rows so that a
    buffer's store and the next head's load into it can start after the first half. Pair p's
    segment, in program order (the sequencer overlaps the units):

        MXU  projections of pair p+2
        VPU  RDOT1(a)  d(a) OUTER(a)  post(p-1)  RDOT2(a)  RDOT1(b)  d(b) OUTER(b)  prep(p+1)
             RDOT2(b)
        DMA  load b (beside RDOT1(a)), store a, load pair p+1's a (beside RDOT2(a), RDOT1(b)),
             store b, pair p+2's taps and convolution rows (beside RDOT2(b))
        MXU  out_proj of pair p-1 and the one before it (after post, every other pair)

    Buffers: two states, two projection tiles (the MXU runs two pairs ahead) and two sets of
    the per-pair vectors (by pair parity). out_proj multiplies og = 4 heads at a time (K =
    512): an MXU output write also stalls a writing VPU op, and it writes one output per og
    heads. The first pair's projections are split so that head a starts early, and the last
    group's out_proj is split in pairs."""
    eps, K = spec.eps, spec.conv_k
    dk, dv = spec.lin_dk, spec.lin_dv
    dn = lw.dn
    nl, C = dn.nl, 2 * dk + dv
    NP, og = nl // 2, dn.og
    sh = dn.shared                      # one key head's q and k for the pair (DeltaNetParts)
    nb, bw = (4, dk) if sh else (2, C)                  # the pair's blocks of channels
    CP, RP = nb * bw, nb * bw + 2 * dv                  # its convolved channels, its rows
    TP = K * CP                                         # taps words of a pair (then its ring)
    xs = ol.quantize(rmsnorm(x, ol.load(lw.g_in), eps))
    ab = ol.dot(xs, lw.wab)                             # [1, 2nl]: a, then b, of each head
    decay, beta = gates(ab[0, 0:nl], ab[0, nl:2 * nl], ol.load(lw.alog), ol.load(lw.dtb))
    eb = ol.empty([4 * NP]).reshape(NP, 4)              # per pair: decays of a, b; betas
    eb[:, 0:2].set(decay.reshape(NP, 2))
    eb[:, 2:4].set(beta.reshape(NP, 2))
    dn.store_gates(eb, hs)
    del decay, beta, eb
    past = _past(K, pos)

    def pairs(n, w):
        return [ol.empty([2 * w]).reshape(2, w) for _ in range(n)]

    St = [ol.empty([dv * dk]).reshape(dv, dk) for _ in range(2)]        # heads a, b of a pair
    P = [ol.empty([1, RP]) for _ in range(2)]        # q k v of a, of b, then z of a, of b
    CV = ol.empty([TP + (K - 1) * CP])     # taps (rows (block, tap)), the window (_past)
    nq = 1 if sh else 2                                 # q and k rows: shared, or per head
    U = [ol.empty([CP]).reshape(nb, bw) for _ in range(2)]
    Qn = [ol.empty([nq * dk]).reshape(nq, dk) for _ in range(2)]
    Kn = [ol.empty([nq * dk]).reshape(nq, dk) for _ in range(2)]
    GZ = pairs(2, dv)
    EB = [ol.empty([4]) for _ in range(2)]
    O = ol.empty([2 * dv]).reshape(2, dv)
    ON = ol.empty([og * dv]).reshape(og, dv)            # normed, gated o of og heads
    w = ol.empty([dv])
    y = ol.zeros([1, spec.hidden])
    gn = ol.load(lw.gn)
    halves = ((0, dv // 2), (dv // 2, dv))     # row halves of a state pass

    def hb(j):
        """Head j's blocks of the pair's channels (None: all): (first, count); shared, head a
        takes q, k and its v, head b its v."""
        if j is None:
            return 0, nb
        return ((0, 3), (3, 1))[j] if sh else (j, 1)

    def project(p, t, split=False):
        """The MXU: pair p's projection rows into P[t] (q, k, v first: prep starts on them)."""
        ha = hb(0)[1] * bw                              # head a's channels first
        cuts = ((0, ha), (ha, CP), (CP, RP)) if split else ((0, CP), (CP, RP))
        for c0, c1 in cuts:
            ol.dot(xs, dn.wh(p)[c0:c1, :], out=P[t][:, c0:c1])

    def fetch_cv(p):
        """Pair p's taps and convolution window (one load)."""
        ol.load(dn.cv(p), out=CV)

    def fetch_eb(p, t):
        ol.load(dn.gates(p, hs), out=EB[t])

    def state_in(p, j, S):
        """Head j of pair p's state into S."""
        if pos:
            for r, e in halves:
                ol.load(dn.state(p, j)[r:e, :], out=S[r:e, :])
        else:
            S.set(0.0)

    def state_out(p, j, S):
        for r, e in halves:
            ol.store(dn.state(p, j)[r:e, :], S[r:e, :])

    def conv(p, t, j=None):
        """Pair p's convolution into U[t] (j: only head j of the pair)."""
        a, n = hb(j)
        pre = P[t][0, a * bw:(a + n) * bw]
        s0 = TP + (K - 2) * CP + a * bw                 # the window's last row: this position
        ol.store(dn.cv(p)[s0:s0 + n * bw], pre)
        if a + n == nb:                                 # its other rows, one up (_past)
            ol.store(dn.cv(p)[TP:TP + (K - 2) * CP], CV[TP + CP:TP + (K - 1) * CP])
        taps = CV[0:TP].reshape(nb * K, bw)

        def tap(i):
            return taps.row_stride_view(a * K + i, n, K) if n > 1 else \
                taps[a * K + i:a * K + i + 1, :]
        # position p - j sits in window row K - 1 - j and is weighed by tap K - 1 - j
        terms = [(pre.reshape(n, bw), K - 1)] + [
            (CV[TP + (K - 1 - j) * CP + a * bw:TP + (K - 1 - j) * CP + (a + n) * bw]
             .reshape(n, bw), K - 1 - j) for j in range(1, past + 1)]
        out = U[t][a:a + n, :]
        if len(terms) == 1:
            out.set(terms[0][0] * tap(K - 1))
            return
        u = terms[0][0] * tap(terms[0][1])
        for x, i in terms[1:-1]:
            u = u + x * tap(i)
        x, i = terms[-1]
        out.set(u + x * tap(i))

    def gatez(t):
        """silu(z) of the pair in P[t]."""
        GZ[t].set(silu(P[t][0, CP:RP].reshape(2, dv)))

    def qk(t, j=None):
        """SiLU of the convolved channels (in place), then the L2-normed q and k (shared: once,
        with head a's blocks)."""
        a, n = hb(j)
        u = U[t][a:a + n, :]
        u.set(silu(u))
        if not sh:
            Qn[t][a:a + n, :].set(l2norm_rows(u[:, 0:dk], dk ** -0.5))
            Kn[t][a:a + n, :].set(l2norm_rows(u[:, dk:2 * dk]))
        elif a == 0:
            Qn[t].set(l2norm_rows(U[t][0:1, :], dk ** -0.5))
            Kn[t].set(l2norm_rows(U[t][1:2, :]))

    def kqv(t, j):
        """Head j's normed k and q, and its v, of the pair in buffers t."""
        r = 0 if sh else j
        return Kn[t][r, :], Qn[t][r, :], U[t][2 + j, :] if sh else U[t][j, 2 * dk:C]

    def post(t):
        """The gated RMSNorm of the pair in O (gates GZ[t]) into its rows of ON."""
        r0 = 2 * t if og == 4 else 0
        ON[r0:r0 + 2, :].set(rmsnorm(O, gn, eps) * GZ[t])

    def flush(g, k=None):
        """out_proj of head group g (k: only its k-th pair)."""
        c0, c1 = (0, og) if k is None else (2 * k, 2 * k + 2)
        ol.dot(ON[c0:c1, :].reshape(1, (c1 - c0) * dv),
               dn.wout(g)[:, c0 * dv:c1 * dv], acc=y)

    def rdot1(S, t, j):
        """kv = S k of head j of the pair in buffers t, into w."""
        w.set(S @ kqv(t, j)[0])

    def update(S, t, j):
        """d = beta (v - e^g kv), then S = e^g S + d k^T (OUTER, in place)."""
        k, _, v = kqv(t, j)
        w.set((v - w * EB[t][j:j + 1]) * EB[t][2 + j:3 + j])
        for r, e in halves:
            ol.outer(w[r:e], k, acc=S[r:e, :], decay=EB[t][j:j + 1])

    def rdot2(S, t, j):
        """o = S q of head j, into O[j]."""
        for r, e in halves:
            O[j, r:e].set(S[r:e, :] @ kqv(t, j)[1])

    def _pair_segment(p, t, last1, last2, g=None):
        """Pair p (buffers t = p % 2); last1: no pair p+1, last2: no pair p+2; g: the head
        group pair p-1 completes (to multiply by out_proj)."""
        first = isinstance(p, int) and p == 0
        if not last2 and not first:
            project(p + 2, t)
        rdot1(St[0], t, 0)
        state_in(p, 1, St[1])
        update(St[0], t, 0)
        state_out(p, 0, St[0])
        if first:                               # head b's prep, after head a's start
            conv(0, 0, 1)
            qk(0, 1)
            gatez(0)
            if not last2:
                project(2, 0)
            if NP > 1:
                fetch_cv(1)
                fetch_eb(1, 1)
        else:
            post(1 - t)
            if g is not None:
                flush(g)
            elif last1:                         # the last group: its first pair now
                flush(group(p), 0)
        rdot2(St[0], t, 0)
        rdot1(St[1], t, 1)
        if not last1:
            state_in(p + 1, 0, St[0])
        update(St[1], t, 1)
        state_out(p, 1, St[1])
        if not last1:
            conv(p + 1, 1 - t)
            qk(1 - t)
            gatez(1 - t)
        rdot2(St[1], t, 1)
        if not last2:
            fetch_cv(p + 2)
            fetch_eb(p + 2, t)

    def group(p):
        """The head group pair p completes, or None."""
        return p // (og // 2) if (p + 1) % (og // 2) == 0 else None

    project(0, 0, split=True)               # head a's rows first: its recurrence starts
    fetch_cv(0)
    fetch_eb(0, 0)
    state_in(0, 0, St[0])
    conv(0, 0, 0)
    qk(0, 0)
    if NP > 1:
        project(1, 1)
    _pair_segment(0, 0, NP == 1, NP <= 2)
    # segments 1 .. NP-3 have every part: loop them, but at a run-time position only in a
    # group-major layer (array-major, the loop's address registers, one per part, would leave
    # too few for the attention's run-time ones)
    n_it = 0 if isinstance(pos, RunPos) and not dn.grouped else max(0, (NP - 3) // 2)
    if n_it:
        for i in ol.range(n_it):
            _pair_segment(2 * i + 1, 1, False, False, None if og == 4 else 2 * i)
            _pair_segment(2 * i + 2, 0, False, False, i if og == 4 else 2 * i + 1)
    for p in range(1 + 2 * n_it, NP):
        _pair_segment(p, p % 2, p + 1 >= NP, p + 2 >= NP, group(p - 1))
    post((NP - 1) % 2)
    flush(group(NP - 1), 1 if og == 4 and NP > 1 else None)
    return x + ol.all_reduce(y)


def _deltanet_dstep(x, lw, pos: int, spec: Spec, hs):
    """_deltanet on a DMA that runs DSTEP (Config.DSTEP): each head's recurrence is one DSTEP,
    which streams the head's fp32 state from DRAM through the DMA's datapath and back (RDOT,
    MUL, SUB, MUL, OUTER, RDOT bit for bit), so the state never enters TMEM and the VPU keeps
    only the small vector work. The same results as _deltanet, word for word.

    Per pair p (buffers t = p % 2), in program order:

        MXU  projections of pair p+2
        DMA  DSTEP(a), DSTEP(b) of pair p (their q | k, v, decay and beta from pair p's prep)
        VPU  prep(p+1) beside them, then post(p) (the gated RMSNorm of pair p's o)
        MXU  out_proj of the head group pair p completes
        DMA  pair p+2's taps and convolution rows, its gates

    The per-pair buffers come in two sets by parity, o included."""
    eps, K = spec.eps, spec.conv_k
    dk, dv = spec.lin_dk, spec.lin_dv
    dn = lw.dn
    nl, C = dn.nl, 2 * dk + dv
    NP, og = nl // 2, dn.og
    sh = dn.shared                      # one key head's q and k for the pair (DeltaNetParts)
    nb, bw = (4, dk) if sh else (2, C)                  # the pair's blocks of channels
    CP, RP = nb * bw, nb * bw + 2 * dv                  # its convolved channels, its rows
    TP = K * CP
    xs = ol.quantize(rmsnorm(x, ol.load(lw.g_in), eps))
    ab = ol.dot(xs, lw.wab)
    decay, beta = gates(ab[0, 0:nl], ab[0, nl:2 * nl], ol.load(lw.alog), ol.load(lw.dtb))
    eb = ol.empty([4 * NP]).reshape(NP, 4)
    eb[:, 0:2].set(decay.reshape(NP, 2))
    eb[:, 2:4].set(beta.reshape(NP, 2))
    dn.store_gates(eb, hs)
    del decay, beta, eb
    past = _past(K, pos)

    def pairs(n, w):
        return [ol.empty([2 * w]).reshape(2, w) for _ in range(n)]

    P = [ol.empty([1, RP]) for _ in range(2)]
    CV = ol.empty([TP + (K - 1) * CP])
    nq = 1 if sh else 2                                 # q | k rows: shared, or per head
    U = [ol.empty([CP]).reshape(nb, bw) for _ in range(2)]
    GZ = pairs(2, dv)
    QK = [ol.empty([nq * 2 * dk]).reshape(nq, 2 * dk) for _ in range(2)]
    O = pairs(2, dv)
    EB = [ol.empty([4]) for _ in range(2)]
    ON = ol.empty([og * dv]).reshape(og, dv)
    y = ol.zeros([1, spec.hidden])
    gn = ol.load(lw.gn)

    def hb(j):
        """Head j's blocks of the pair's channels (None: all): (first, count); shared, head a
        takes q, k and its v, head b its v."""
        if j is None:
            return 0, nb
        return ((0, 3), (3, 1))[j] if sh else (j, 1)

    def project(p, t, split=False):
        ha = hb(0)[1] * bw                              # head a's channels first
        cuts = ((0, ha), (ha, CP), (CP, RP)) if split else ((0, CP), (CP, RP))
        for c0, c1 in cuts:
            ol.dot(xs, dn.wh(p)[c0:c1, :], out=P[t][:, c0:c1])

    def fetch_cv(p):
        ol.load(dn.cv(p), out=CV)

    def fetch_eb(p, t):
        ol.load(dn.gates(p, hs), out=EB[t])

    def conv(p, t, j=None):
        a, n = hb(j)
        pre = P[t][0, a * bw:(a + n) * bw]
        s0 = TP + (K - 2) * CP + a * bw                 # the window's last row: this position
        ol.store(dn.cv(p)[s0:s0 + n * bw], pre)
        if a + n == nb:                                 # its other rows, one up (_past)
            ol.store(dn.cv(p)[TP:TP + (K - 2) * CP], CV[TP + CP:TP + (K - 1) * CP])
        taps = CV[0:TP].reshape(nb * K, bw)

        def tap(i):
            return taps.row_stride_view(a * K + i, n, K) if n > 1 else \
                taps[a * K + i:a * K + i + 1, :]
        # position p - j sits in window row K - 1 - j and is weighed by tap K - 1 - j
        terms = [(pre.reshape(n, bw), K - 1)] + [
            (CV[TP + (K - 1 - j) * CP + a * bw:TP + (K - 1 - j) * CP + (a + n) * bw]
             .reshape(n, bw), K - 1 - j) for j in range(1, past + 1)]
        out = U[t][a:a + n, :]
        if len(terms) == 1:
            out.set(terms[0][0] * tap(K - 1))
            return
        u = terms[0][0] * tap(terms[0][1])
        for xx, i in terms[1:-1]:
            u = u + xx * tap(i)
        xx, i = terms[-1]
        out.set(u + xx * tap(i))

    def gatez(t):
        GZ[t].set(silu(P[t][0, CP:RP].reshape(2, dv)))

    def qk(t, j=None):
        """SiLU of the convolved channels (in place), then the L2-normed q and k into QK[t]
        (shared: once, with head a's blocks)."""
        a, n = hb(j)
        u = U[t][a:a + n, :]
        u.set(silu(u))
        if not sh:
            QK[t][a:a + n, 0:dk].set(l2norm_rows(u[:, 0:dk], dk ** -0.5))
            QK[t][a:a + n, dk:2 * dk].set(l2norm_rows(u[:, dk:2 * dk]))
        elif a == 0:
            QK[t][:, 0:dk].set(l2norm_rows(U[t][0:1, :], dk ** -0.5))
            QK[t][:, dk:2 * dk].set(l2norm_rows(U[t][1:2, :]))

    def prep(p, t):
        conv(p, t)
        qk(t)
        gatez(t)

    def post(t):
        r0 = 2 * t if og == 4 else 0
        ON[r0:r0 + 2, :].set(rmsnorm(O[t], gn, eps) * GZ[t])

    def flush(g, k=None):
        c0, c1 = (0, og) if k is None else (2 * k, 2 * k + 2)
        ol.dot(ON[c0:c1, :].reshape(1, (c1 - c0) * dv),
               dn.wout(g)[:, c0 * dv:c1 * dv], acc=y)

    def dstep(p, t, j):
        ol.deltanet_step(dn.state(p, j), QK[t][0 if sh else j, :],
                         U[t][2 + j, :] if sh else U[t][j, 2 * dk:C], EB[t][j:j + 1],
                         EB[t][2 + j:3 + j], O[t][j, :],
                         zero=not isinstance(pos, RunPos) and pos == 0)

    def group(p):
        return p // (og // 2) if (p + 1) % (og // 2) == 0 else None

    def _pair_segment(p, t, last1, last2, g=None):
        """Pair p (buffers t); last1: no pair p+1, last2: no pair p+2; g: the head group pair
        p completes."""
        first = isinstance(p, int) and p == 0
        if not last2 and not first:
            project(p + 2, t)
        dstep(p, t, 0)
        if first:                               # head b's prep, after head a's start
            conv(0, 0, 1)
            qk(0, 1)
            gatez(0)
            if not last2:
                project(2, 0)
            if not last1:
                fetch_cv(1)
                fetch_eb(1, 1)
        dstep(p, t, 1)
        if not last1:
            prep(p + 1, 1 - t)
        post(t)
        if og == 4 and isinstance(p, int) and p >= NP - 2:
            flush((NP - 1) // 2, p - (NP - 2))  # the last group pair by pair, as _deltanet
        elif g is not None:
            flush(g)
        if not last2:
            fetch_cv(p + 2)
            fetch_eb(p + 2, t)

    project(0, 0, split=True)
    fetch_cv(0)
    fetch_eb(0, 0)
    conv(0, 0, 0)
    qk(0, 0)
    if NP > 1:
        project(1, 1)
    _pair_segment(0, 0, NP == 1, NP <= 2, group(0))
    # segments 1 .. NP-3 have every part: loop them, but at a run-time position only in a
    # group-major layer (array-major, the loop's address registers, one per part, would leave
    # too few for the attention's run-time ones)
    n_it = 0 if isinstance(pos, RunPos) and not dn.grouped else max(0, (NP - 3) // 2)
    if n_it:
        for i in ol.range(n_it):
            _pair_segment(2 * i + 1, 1, False, False, i if og == 4 else 2 * i + 1)
            _pair_segment(2 * i + 2, 0, False, False, None if og == 4 else 2 * i + 2)
    for p in range(1 + 2 * n_it, NP):
        _pair_segment(p, p % 2, p + 1 >= NP, p + 2 >= NP, group(p))
    return x + ol.all_reduce(y)


@ol.jit
def qwen35_step(m, pos: int, block: int = ATTN_BLOCK, tok: int | None = None):
    """One decode token at position `pos`: x (the token's embedding) -> logits.

    Each run of m.plan with repeats is a hardware loop over its unit of layers; the others are
    unrolled. DeltaNet layers update their state and convolution ring, attention layers append
    K/V at `pos` and attend over positions 0..pos. Logits for this slice's vocabulary rows go
    to m.logits.
    """
    spec = m.spec
    x, c, s_ = _inputs(m, pos, tok)

    def layer(li, kind):
        lw = m.layer(li, kind)
        if kind == LIN:
            dn = _deltanet_dstep if ol.has_dstep() else _deltanet
            x.set(dn(x, lw, pos, spec, m.hs))
        else:
            x.set(_attention(x, lw, c, s_, pos, spec, block, gated=True))
        if lw.moe:
            x.set(MO.moe_ffn(x, lw, spec.moe, m.moe_dev, spec.eps))
        else:
            x.set(_mlp(x, lw, spec))

    run_layers(m.plan, layer)
    _lm_head(x, m, spec)


def _deltanet_rows(x, lw, p0: int, spec: Spec, gr, on):
    """_deltanet for R consecutive positions p0 .. p0+R-1 at once. The projections stream once
    for the R rows, pair of heads by pair; the convolution runs over the pair's rows before
    it in the chunk and its ring; each head's state is loaded once, updated and read row after
    row (the recurrence is sequential in the tokens), and stored once. Per row every value is
    computed by _deltanet's operations in its order, and out_proj accumulates over the same
    head groups (the last one in pairs when og = 4), so the result is bit-identical.

    The recurrence is unrolled over the rows, so the pairs run as hardware loops (a loop over
    the head groups but the last, each a loop over its pairs; then the last group's pairs),
    or the program would not fit IMEM. TMEM addresses are static: the per-head decays and
    betas go through `gr` in DRAM ([R, 2nl]: the decays, then the betas of each row), and a
    group's normed outputs through `on` ([R, og * dv]) before its out_proj.

    With DSTEP (ol.has_dstep()) each row's step is one DSTEP on the state in DRAM (the DMA
    streams it through its datapath and back), row after row, instead of the VPU passes."""
    eps, K, R = spec.eps, spec.conv_k, x.rows
    dk, dv = spec.lin_dk, spec.lin_dv
    dn = lw.dn
    nl, C = dn.nl, 2 * dk + dv
    og, sh = dn.og, dn.shared                           # sh: the pair's q and k shared
    nb, bw = (4, dk) if sh else (2, C)                  # the pair's blocks of channels
    CP, RP = nb * bw, nb * bw + 2 * dv                  # its convolved channels, its rows
    NP, ng, gp = nl // 2, nl // og, og // 2             # pairs, head groups, pairs per group
    TP = K * CP                                         # taps words of a pair (then its ring)
    xs = ol.quantize(rmsnorm(x, ol.load(lw.g_in), eps))
    ab = ol.dot(xs, lw.wab)                             # [R, 2nl]: a, then b, of each head
    alog, dtb = ol.load(lw.alog), ol.load(lw.dtb)
    for r in range(R):
        dr, br = gates(ab[r, 0:nl], ab[r, nl:2 * nl], alog, dtb)
        ol.store(gr[r, 0:nl], dr)
        ol.store(gr[r, nl:2 * nl], br)
        del dr, br
    del ab, alog, dtb
    gn = ol.load(lw.gn)
    y = ol.zeros([R, spec.hidden])
    dstep = ol.has_dstep()
    if not dstep:
        St = ol.empty([dv * dk]).reshape(dv, dk)
        w = ol.empty([dv])
    ONp = ol.empty([R, 2 * dv])                         # normed, gated o of a pair per row
    GDB = ol.empty([R, 4])                              # decays, then betas, of the pair's heads
    GD, GB = GDB[:, 0:2], GDB[:, 2:4]
    full = max(0, K - 1 - p0)                           # rows before it lack positions < 0
    rgroups = [(r, r + 1) for r in range(min(full, R))] + ([(full, R)] if full < R else [])
    split = og == 4 and NP > 1                          # _deltanet's last group, in pairs

    def flush(g, ON, c0, c1):
        """y += ON . out_proj columns [c0, c1) of head group g (heads of dv columns)."""
        ol.dot(ON, dn.wout(g)[:, c0 * dv:c1 * dv], acc=y)

    def head_in(X, taps, a):
        """Block a of the channels of the pair whose rows are in X (head a's q k v; shared, q, k,
        v of a, v of b): the convolution and SiLU -> [R, bw]."""
        U = ol.empty([R, bw])
        for r0, r1 in rgroups:                          # the convolution, as _deltanet's conv
            t = min(K - 1, p0 + r0)
            cur = X[K - 1 + r0:K - 1 + r1, a * bw:(a + 1) * bw]
            if t == 0:
                U[r0:r1, :].set(cur * taps[a * K + K - 1, :][None, :])
                continue
            u = cur * taps[a * K + K - 1, :][None, :]
            for j in range(1, t):
                u = u + X[K - 1 + r0 - j:K - 1 + r1 - j, a * bw:(a + 1) * bw] * \
                    taps[a * K + K - 1 - j, :][None, :]
            U[r0:r1, :].set(u + X[K - 1 + r0 - t:K - 1 + r1 - t, a * bw:(a + 1) * bw] *
                            taps[a * K + K - 1 - t, :][None, :])
            del u
        U.set(silu(U))
        return U

    def head(p, X, Z, taps, a):
        """Head a of pair p (whose q k v rows are in X, z in Z) -> ONp[:, a]."""
        U = head_in(X, taps, a)
        GZ = silu(Z[:, a * dv:(a + 1) * dv])
        if dstep:                                       # q | k per row, a DSTEP per row
            QK = ol.empty([R, 2 * dk])
            QK[:, 0:dk].set(l2norm_rows(U[:, 0:dk], dk ** -0.5))
            QK[:, dk:2 * dk].set(l2norm_rows(U[:, dk:2 * dk]))
            O = ol.empty([R, dv])
            for r in range(R):
                ol.deltanet_step(dn.state(p, a), QK[r, :], U[r, 2 * dk:C], GDB[r, a:a + 1],
                                 GDB[r, 2 + a:3 + a], O[r, :], zero=(p0 == 0 and r == 0))
            del QK
            ONp[:, a * dv:(a + 1) * dv].set(rmsnorm(O, gn, eps) * GZ)
            return
        Qn = l2norm_rows(U[:, 0:dk], dk ** -0.5)
        Kn = l2norm_rows(U[:, dk:2 * dk])
        if p0:
            ol.load(dn.state(p, a), out=St)
        else:
            St.set(0.0)
        O = ol.empty([R, dv])
        for r in range(R):                              # the recurrence, token by token
            dh, bh = GD[r, a:a + 1], GB[r, a:a + 1]
            w.set(St @ Kn[r, :])
            w.set((U[r, 2 * dk:C] - w * dh) * bh)
            ol.outer(w, Kn[r, :], acc=St, decay=dh)
            O[r, :].set(St @ Qn[r, :])
        ol.store(dn.state(p, a), St)
        ONp[:, a * dv:(a + 1) * dv].set(rmsnorm(O, gn, eps) * GZ)

    def shared_heads(p, X, Z, taps):
        """Pair p's heads, the key head's q and k shared (blocks q, k, v of a, v of b in X) ->
        ONp: q and k convolved and normed once. With DSTEP both heads' vector work before
        their DSTEPs, as heads()."""
        if dstep:
            QK = ol.empty([R, 2 * dk])
            for i in range(2):
                U = head_in(X, taps, i)
                QK[:, i * dk:(i + 1) * dk].set(l2norm_rows(U, dk ** -0.5 if i == 0 else 1.0))
                del U
            V = [head_in(X, taps, 2 + a) for a in range(2)]
            GZ = [silu(Z[:, a * dv:(a + 1) * dv]) for a in range(2)]
            O = [ol.empty([R, dv]) for _ in range(2)]
            for a in range(2):
                for r in range(R):
                    ol.deltanet_step(dn.state(p, a), QK[r, :], V[a][r, :], GDB[r, a:a + 1],
                                     GDB[r, 2 + a:3 + a], O[a][r, :], zero=(p0 == 0 and r == 0))
            del QK, V
            for a in range(2):
                ONp[:, a * dv:(a + 1) * dv].set(rmsnorm(O[a], gn, eps) * GZ[a])
            return
        U = head_in(X, taps, 0)
        Qn = l2norm_rows(U, dk ** -0.5)
        del U
        U = head_in(X, taps, 1)
        Kn = l2norm_rows(U)
        del U
        for a in range(2):
            V = head_in(X, taps, 2 + a)
            GZ = silu(Z[:, a * dv:(a + 1) * dv])
            if p0:
                ol.load(dn.state(p, a), out=St)
            else:
                St.set(0.0)
            O = ol.empty([R, dv])
            for r in range(R):                          # the recurrence, token by token
                dh, bh = GD[r, a:a + 1], GB[r, a:a + 1]
                w.set(St @ Kn[r, :])
                w.set((V[r, :] - w * dh) * bh)
                ol.outer(w, Kn[r, :], acc=St, decay=dh)
                O[r, :].set(St @ Qn[r, :])
            ol.store(dn.state(p, a), St)
            ONp[:, a * dv:(a + 1) * dv].set(rmsnorm(O, gn, eps) * GZ)
            del V, GZ, O

    # a pair's projections, taps and ring rows; two sets with DSTEP (pair p + 1's projections
    # stream while pair p's DSTEPs run)
    NB = 2 if dstep and split and gp == 2 else 1
    PB = [(ol.empty([K * CP]), ol.empty([K - 1 + R, CP]), ol.empty([R, 2 * dv]))
          for _ in range(NB)]

    def project(p, b, window=True):
        """Pair p's taps, q k v rows (ring, then its projections) and z into buffers b; window:
        store the next window (else store_window after the pair's DSTEPs: a store waiting for
        the projections would hold the DMA's queue, and the DSTEPs behind it)."""
        taps, X, Z = PB[b]
        ol.load(dn.cv(p)[0:TP], out=taps)              # rows (block, tap)
        for j in range(1, min(K, p0 + 1)):              # positions p0-K+1 ..: the window
            sl = TP + (K - 1 - j) * CP                  # (_past)
            ol.load(dn.cv(p)[sl:sl + CP], out=X[K - 1 - j, :])
        ol.dot(xs, dn.wh(p)[0:CP, :], out=X[K - 1:K - 1 + R, :])
        ol.dot(xs, dn.wh(p)[CP:RP, :], out=Z)                            # z of a, of b
        if window:
            store_window(p, b)

    def store_window(p, b):
        X = PB[b][1]
        for i in range(K - 1):                          # the next window
            ol.store(dn.cv(p)[TP + i * CP:TP + (i + 1) * CP], X[R + i, :])

    def heads(p, b):
        """Pair p's heads from buffers b -> ONp. With DSTEP both heads' VPU work (convolution,
        SiLU, norms) comes before their DSTEPs and both outputs after, so the DMA runs the pair's
        DSTEPs back to back (the same operations, in another order)."""
        taps, X, Z = PB[b]
        taps = taps.reshape(nb * K, bw)
        q = dn.index(p)
        for r in range(R):
            ol.load(gr[r, 2 * q:2 * q + 2], out=GD[r, :])
            ol.load(gr[r, nl + 2 * q:nl + 2 * q + 2], out=GB[r, :])
        if sh:
            shared_heads(p, X, Z, taps)
            return
        if not dstep:
            for a in range(2):
                head(p, X, Z, taps, a)
            return
        QK = [ol.empty([R, 2 * dk]) for _ in range(2)]
        V = [ol.empty([R, dv]) for _ in range(2)]
        GZ = [ol.empty([R, dv]) for _ in range(2)]
        for a in range(2):
            U = head_in(X, taps, a)
            QK[a][:, 0:dk].set(l2norm_rows(U[:, 0:dk], dk ** -0.5))
            QK[a][:, dk:2 * dk].set(l2norm_rows(U[:, dk:2 * dk]))
            V[a].set(U[:, 2 * dk:C])
            del U
            GZ[a].set(silu(Z[:, a * dv:(a + 1) * dv]))
        O = [ol.empty([R, dv]) for _ in range(2)]
        for a in range(2):
            for r in range(R):
                ol.deltanet_step(dn.state(p, a), QK[a][r, :], V[a][r, :], GDB[r, a:a + 1],
                                 GDB[r, 2 + a:3 + a], O[a][r, :], zero=(p0 == 0 and r == 0))
        del QK, V
        for a in range(2):
            ONp[:, a * dv:(a + 1) * dv].set(rmsnorm(O[a], gn, eps) * GZ[a])

    def pair(p):
        """Pair p (an int, a loop expression or (group, index): DeltaNetParts) -> ONp."""
        project(p, 0)
        heads(p, 0)

    def loop(n):
        """ol.range(n), or the single index 0 unrolled."""
        return ol.range(n) if n > 1 else range(n)

    if NB == 2:
        # groups of 2 pairs, software pipelined: the projections of the pair after the next
        # one stream while a pair's DSTEPs run (the same operations, in another order)
        project(0, 0, window=False)
        for g in loop(ng - 1):
            project(2 * g + 1, 1, window=False)
            heads(2 * g, 0)
            store_window(2 * g, 0)
            ol.store(on[:, 0:2 * dv], ONp)
            project(2 * g + 2, 0, window=False)
            heads(2 * g + 1, 1)
            store_window(2 * g + 1, 1)
            ol.store(on[:, 2 * dv:4 * dv], ONp)
            flush(g, ol.load(on), 0, og)
        q0 = 2 * (ng - 1)                               # the last group, pair by pair
        project(q0 + 1, 1, window=False)
        heads(q0, 0)
        store_window(q0, 0)
        flush(ng - 1, ONp, 0, 2)
        heads(q0 + 1, 1)
        store_window(q0 + 1, 1)
        flush(ng - 1, ONp, 2, 4)
        return x + ol.all_reduce(y)
    for g in loop(ng - 1 if split else ng):             # whole groups
        if gp == 1:
            pair(g)
            flush(g, ONp, 0, 2)
            continue
        for q in loop(gp):
            pair((g, q))
            ol.store(on[:, q * 2 * dv:(q + 1) * 2 * dv], ONp)
        flush(g, ol.load(on), 0, og)
    if split:                                           # the last group, pair by pair
        for q in loop(gp):
            pair((ng - 1, q))
            flush(ng - 1, ONp, 2 * q, 2 * q + 2)
    return x + ol.all_reduce(y)


@ol.jit
def qwen35_rows(m, p0: int, R: int, logit_rows, block: int = ATTN_BLOCK, tokens=None):
    """R prompt tokens at positions p0 .. p0+R-1 at once: their embeddings m.xr and RoPE
    tables m.cosr / m.sinr, or those of `tokens` from the image's tables (qwen3._inputs_rows) ->
    logits of the rows in `logit_rows` (a contiguous range, or empty). Bit-identical to R
    qwen35_step runs."""
    spec = m.spec
    rows = [(0, p0 + r) for r in range(R)]
    x, c, s_ = _inputs_rows(m, rows, tokens)

    def layer(li, kind):
        lw = m.layer(li, kind)
        if kind == LIN:
            x.set(_deltanet_rows(x, lw, p0, spec, m.gr[0:R, :], m.on[0:R, :]))
        else:
            x.set(_attention_rows(x, lw, c, s_, rows, spec, block, gated=True))
        x.set(_mlp(x, lw, spec))

    run_layers(m.plan, layer)
    _lm_head_rows(x, m, spec, logit_rows)
