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
from .. import qcache as QC
from .. import quant as Q
from .. import language as ol
from ..compiler import Affine, CompileError, KVDesc, QTensor, Tensor, current
from ..host.offload import LINE, ExpertServer, Layout
from ..isasim import Config
from ..kernels.layouts import head_parallel_attention_weights
from ..kernels.deltanet import gates, l2norm_rows
from ..kernels.lib import rmsnorm, silu
from ..kernels.mlp import _chunk
from .lfm2 import _place, plan, run_layers
from . import formats as FM
from . import generate as G
from . import moe as MO
from . import rope_parameters
from .qwen3 import (ATTN_BLOCK, HEAD_CHUNK, OutTokens, RunPos, RunRows, RunWords,
                    _amask, _attention, _attention_rows, _Bump,
                    _embed, _fake_q, _fake_w, _formats, _gather, _inputs, _inputs_rows,
                    _lm_head, _lm_head_rows, _lookup_alloc, _lookup_build, _lookup_desc, _mlp,
                    _pv, _qdesc, _tdesc, _tok_arg, _tokens_arg, _v_parts, compile_decode,
                    rope_tables, EmbedHost, fill_logits, step_descriptors)

LIN, ATTN = "linear", "attn"
PAIR_LOOP = 8       # pairs of DeltaNet heads per slice that decode unrolled at a run-time position
MTP_VOCAB = 32768   # the MTP's draft head: the fp4 rows of this many lowest ids (byte-level BPE
                    # ids follow merge frequency: they cover 94% of the generated tokens;
                    # docs/mtp.md 7.1)
EMBED_F32_MAX = 2 << 30     # bytes: a larger fp32 embedding table (over half the card's DRAM,
#                             Qwen3.5-4B and up) is int8, gathered from the head (Spec.embed)
PREFILL_CHUNK = 512 # a MoE model's layer-major prefill: prompt rows a chunk (Image's xbuf)
RUN_ROWS = 4        # and at most rows a run (_deltanet_rows' gr / on; moe_ffn_rows: R k ids)


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
    formats: str = ""       # weight formats per kind over the image's wformat (KINDS)
    mix: str = ""           # the recommended mix (wformat "mix": formats.named, MIXES)
    mtp: bool = False       # the image holds the MTP drafter (the checkpoint's mtp.* layer, a
                            # draft head) and two slots of every DeltaNet state and convolution
                            # window for the speculative verify (opentpu/llm/mtp.py, docs/mtp.md 9)

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
        rope = rope_parameters(c, model_dir)        # (mRoPE: 1-D RoPE for text)
        d = c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"]
        eos = c.get("eos_token_id", 248044)
        eos = tuple(eos) if isinstance(eos, list) else (eos,)
        moe = None
        if "num_experts" in c:                  # Qwen3.5-MoE: softmax top-k, a shared expert
            moe = MO.MoESpec(E=c["num_experts"], k=c["num_experts_per_tok"],
                             ffn=c["moe_intermediate_size"], rule="softmax",
                             norm=c.get("norm_topk_prob", True),
                             shared=c["shared_expert_intermediate_size"])
            # (MoESpec.hint, the router's prefetch hint before each mixer, stays off: on the
            # card it lost 5% against none, docs/offload.md 12.5)
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
                    or 4 * c["vocab_size"] * c["hidden_size"] > EMBED_F32_MAX else "f32",
                    mix=FM.mix_for(c))

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
        if self.mtp:
            need += [(self.moe is None, "MTP decoding of a MoE model"),
                     (S == 1, "MTP decoding runs on one slice")]
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
              lookup: bool | str = False, experts: int | None = None,
              embed_host: bool | None = None, formats: str | None = None,
              prefill_rows: int | None = None) -> "Image":
        return Image(self, cfg, cap, batch, rows, wformat, head_format, lookup, experts,
                     embed_host, formats, prefill_rows)


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


def reference_logits(spec: Spec, W: dict, tokens, hidden: bool = False):
    """fp32 numpy forward of the whole sequence (causal); returns logits [T, vocab] (hidden:
    and the final norm's rows [T, H], the MTP layer's input)."""
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
    return (x @ head.T, x) if hidden else x @ head.T


def mtp_reference(spec: Spec, W: dict, hid, tokens) -> np.ndarray:
    """fp32 numpy MTP drafter (docs/mtp.md 6.1) over T rows at positions 0 .. T-1: row q takes
    the final-normed hidden at q (hid [T, H]) and the token at q + 1 (tokens[q]); returns the
    draft head's logits [T, min(MTP_VOCAB, vocab)] (causal over the rows, its own KV)."""
    T, d, G, eps = len(tokens), spec.head_dim, spec.n_q // spec.n_kv, spec.eps
    f32 = np.float32
    emb = W["model.embed_tokens.weight"]

    def g1(n):
        return (1 + W[n]).astype(f32)

    e = _norm(emb[list(tokens)].astype(f32), g1("mtp.pre_fc_norm_embedding.weight"), eps)
    h = _norm(np.asarray(hid, f32), g1("mtp.pre_fc_norm_hidden.weight"), eps)
    x = np.concatenate([e, h], axis=1) @ W["mtp.fc.weight"].T
    cs = [rope_tables(spec, p) for p in range(T)]
    cos = np.stack([c for c, _ in cs])[:, None, :]
    sin = np.stack([s for _, s in cs])[:, None, :]
    mask = np.triu(np.full((T, T), -np.inf, f32), 1)
    p, a = "mtp.layers.0.", "mtp.layers.0.self_attn."
    hn = _norm(x, g1(p + "input_layernorm.weight"), eps)
    qg = (hn @ W[a + "q_proj.weight"].T).reshape(T, spec.n_q, 2 * d)
    q, gate = qg[..., :d], qg[..., d:]
    k = (hn @ W[a + "k_proj.weight"].T).reshape(T, spec.n_kv, d)
    v = (hn @ W[a + "v_proj.weight"].T).reshape(T, spec.n_kv, d)
    q = _rot(_norm(q, g1(a + "q_norm.weight"), eps), cos, sin, spec.rope_dim)
    k = _rot(_norm(k, g1(a + "k_norm.weight"), eps), cos, sin, spec.rope_dim)
    o = np.zeros((T, spec.n_q, d), f32)
    for hq in range(spec.n_q):
        sc = q[:, hq] @ k[:, hq // G].T / math.sqrt(d) + mask
        sc = np.exp(sc - sc.max(axis=1, keepdims=True))
        o[:, hq] = (sc / sc.sum(axis=1, keepdims=True)) @ v[:, hq // G]
    o = o / (1 + np.exp(-gate))
    x = x + o.reshape(T, -1) @ W[a + "o_proj.weight"].T
    hn = _norm(x, g1(p + "post_attention_layernorm.weight"), eps)
    x = x + (_silu(hn @ W[p + "mlp.gate_proj.weight"].T) * (hn @ W[p + "mlp.up_proj.weight"].T)) \
        @ W[p + "mlp.down_proj.weight"].T
    x = _norm(x, g1("mtp.norm.weight"), eps)
    head = emb if spec.tied else W["lm_head.weight"]
    return x @ head[:min(MTP_VOCAB, spec.vocab)].T


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


KINDS = ("attn", "delta", "mlp", "gateup", "down", "head")   # a formats string's weight kinds


def weight_kind(n: str) -> tuple:
    """(kind, layer) of checkpoint weight `n` (KINDS: the head, or a layer's attention,
    DeltaNet (in_proj_*, out_proj), MLP gate / up or down projection; Qwen3.5-MoE's shared
    expert is the MLP)."""
    if not n.startswith("model.layers."):
        return "head", 0
    return ("delta" if ".linear_attn." in n else "attn" if ".self_attn." in n else
            "down" if ".down_proj." in n else "gateup", int(n.split(".")[2]))


def emulated_logits(spec: Spec, W: dict, tokens, D: int = 128, wformat: str = "int8",
                    head_format: str | None = None, routes: list | None = None,
                    formats: str | None = None) -> np.ndarray:
    """float64 decode with openTPU's quantization points and none of its rounding (as
    qwen3.emulated_logits): int8 weights and matmul inputs per D-block, int8 K and V, int8 P
    with V's scales folded in.
    The DeltaNet state, convolution and gates are exact (they are fp32 on the device).
    Qwen3.5-MoE: the router (with the shared expert's gate as its last row) in int8; `routes`
    gets (token, layer, ids, the k-th and (k+1)-th logits' gap) per MoE layer."""
    d, G, eps, K = spec.head_dim, spec.n_q // spec.n_kv, spec.eps, spec.conv_k
    nh, nk, dk, dv = spec.lin_heads, spec.lin_nk, spec.lin_dk, spec.lin_dv
    Wq: dict = {}

    head = "model.embed_tokens.weight" if spec.tied else "lm_head.weight"
    wformat, formats = FM.named(spec, wformat, formats)
    fmt = FM.resolver(formats, KINDS, spec.formats, wformat, head_format)

    def w(n):
        if n not in Wq:        # the weight formats as in Image (wformat, head_format, formats)
            Wq[n] = _fake_w(W[n], D, fmt(*weight_kind(n)))
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
                Vc[i].append(_v_parts(v))
                Kh = np.stack(Kc[i], 1)
                Vq, Vs = (np.stack(z, 1) for z in zip(*Vc[i]))
                o = np.zeros((spec.n_q, d))
                for hq in range(spec.n_q):
                    sc = Kh[hq // G] @ _fake_q(q[hq] / math.sqrt(d), D)
                    pp = np.exp(sc - sc.max())
                    o[hq] = _pv(pp, Vq[hq // G], Vs[hq // G], D) / pp.sum()
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
    four blocks of dk (q, k, v a, v b) instead of two of 2 dk + dv (the heads).

    Slots (Spec.mtp): the states and windows come in two slots, `slot` the one holding the
    committed ones (state, window), the other at sbytes bytes / wwords words after slot 0's
    (the speculative verify's rejectable row writes there: docs/mtp.md 9). tp: the taps' words
    before the window in cv."""

    def __init__(self, nl: int, og: int, wh: QTensor, cv: Tensor, state: Tensor,
                 wout: QTensor, H: int, gs: int = 0, ps: int = 0, eb: Tensor | None = None,
                 shared: bool = False, tp: int = 0, wwords: int = 0, slot: int = 0,
                 sbytes: int = 0):
        self.nl, self.og, self.gp, self.H, self.shared = nl, og, og // 2, H, shared
        self.grouped, self.gs, self.ps = gs > 0, gs, ps
        self._wh, self._cv, self._state, self._wout, self._eb = wh, cv, state, wout, eb
        self.R2 = wh.shape[0] if self.grouped else wh.shape[0] // (nl // 2)
        self.tp, self.wwords, self.slot, self.sbytes = tp, wwords, slot, sbytes
        if slot and not sbytes:
            raise CompileError("state slot 1 needs an image with slots (Spec.mtp)")

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

    def state(self, p, a: int, slot: int | None = None) -> Tensor:
        """The state of head a (0, 1) of pair p: in the committed slot, or in `slot`."""
        slot = self.slot if slot is None else slot
        if slot and not self.sbytes:
            raise CompileError("state slot 1 needs an image with slots (Spec.mtp)")
        if not self.grouped:
            t = self._state[2 * self.index(p) + a]
        else:
            s = self._state
            t = Tensor(s.base + self._off(p), s.shape, s.strides)[a]
        return Tensor(t.base + slot * self.sbytes, t.shape, t.strides) if slot else t

    def window(self, p, slot: int | None = None) -> Tensor:
        """Pair p's convolution window (K - 1 rows of its channels, oldest first, after its
        taps): in the committed slot, or in `slot`."""
        slot = self.slot if slot is None else slot
        if slot and not self.sbytes:
            raise CompileError("window slot 1 needs an image with slots (Spec.mtp)")
        c = self.cv(p)
        return Tensor(c.base + 4 * (self.tp + slot * self.wwords), (self.wwords,), (1,))

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


class Image(EmbedHost):
    """Per-slice DRAM layout of a Qwen3.5 model. Every slice uses the same addresses.

    [ I/O: x_in, cos, sin | final norm | logits | per-pair gates ] [ layer 0 block ] ...
    [ layer L-1 block ] [ LM head rows of this slice ]. A layer block's layout and size are its
    kind's in its formats group (a layer's DeltaNet, attention, gate / up and down formats:
    Image.lf); both kinds start with the norms and this slice's MLP rows, at the same offsets
    in a group. A DeltaNet block then holds, for this slice's heads (a contiguous range), the
    projections pair by pair (the q, k, v rows of head 0, of head 1, then the z rows of heads 0
    and 1; then heads 2 and 3, ...; with shared q and k, the pair's key head's q, k, then v of
    each head, then z of each), the a and b rows, out_proj as one [H, og * dv] column block per
    og heads, per pair the convolution taps and then the convolution ring, the recurrent state
    (per head [dv, dk] fp32, transposed) and the per-head constants (the window: K - 1 rows,
    _past). An attention block holds the q/k norms, the projections (the gate rows of q_proj as
    their own matrix) and this slice's KV heads with room for `cap` tokens. The I/O area holds
    `rows` token rows (x, cos, sin, logits) for chunked prefill.

    Group-major DeltaNet blocks (Spec.pair_loop; by default when a slice has more than
    PAIR_LOOP pairs of heads): the projections, the out_proj blocks, the taps and windows, the
    states and the per-pair gates go per head group instead, each group's pairs (a pair's
    projection rows, their scales, its taps and window, its two states), then the group's
    out_proj block and its pairs' gates (DeltaNetParts). A decode's loop over the pairs then steps one address register for all of
    them, so it fits the registers beside a run-time position's (resident decode), and the
    unrolled pairs of a bigger model need not fit IMEM.

    Weight formats as qwen3.Image (`formats` over the KINDS of this file, per layer range; a
    MoE's experts in `wformat`, its router int8): the layers run as the runs of lfm2.plan over
    their (kind, formats group) keys, each run's blocks one after the other; the MTP layer takes
    the formats without a range.

    Spec.mtp (MTP decoding, opentpu/llm/mtp.py): every DeltaNet state and convolution window
    has two slots (the state's second after the first, a pair's second window after its
    first), one more attention layer block after the model's holds the MTP layer
    (mtp.layers.0) with its own KV cache, then its norms, fc and the draft head (fp4 rows of
    the MTP_VOCAB lowest ids); the I/O area gains `hid` (the final norm's rows) and `draft`
    (each row's draft id).
    """

    def __init__(self, spec: Spec, cfg: Config, cap: int, batch: int = 1, rows: int = 1,
                 wformat: str = "int8", head_format: str | None = None, lookup: bool | str = False,
                 experts: int | None = None, embed_host: bool | None = None,
                 formats: str | None = None, prefill_rows: int | None = None):
        spec.check(cfg)
        if batch != 1:
            raise ValueError("Qwen3.5 runs one sequence: batch=1")
        if cap % cfg.D:
            raise ValueError("KV capacity must be a multiple of D")
        S, D = cfg.S, cfg.D
        H, d, F_, K = spec.hidden, spec.head_dim, spec.ffn, spec.conv_k
        wformat, formats = FM.named(spec, wformat, formats)
        fmt = FM.resolver(formats, KINDS, spec.formats, wformat, head_format)
        # each layer's formats group: DeltaNet, attention, gate / up, down (the layers of a
        # group share a block size and the common part's offsets, as one group did)
        self.lf = tuple((fmt("delta", i), fmt("attn", i), fmt("gateup", i), fmt("down", i))
                        for i in range(spec.layers))
        self.wformat, self.head_format = wformat, fmt("head")
        self.formats = _formats(spec, formats)
        rb = lambda k, f: Q.row_bytes(k, f, D)                          # noqa: E731
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
        self.CV0 = K * self.CP + (K - 1) * self.CP      # a pair's taps, then its window
        self.slots = 2 if spec.mtp else 1               # DeltaNet state and window slots
        self.CVW = self.CV0 + (self.slots - 1) * (K - 1) * self.CP     # (then window 1)
        b = _Bump()
        # the layer-major prefill's rows a chunk (a MoE model's: compile_layer_run), and the
        # rows of _deltanet_rows' DRAM scratch (gr, on)
        self.prefill_rows = (PREFILL_CHUNK if spec.moe is not None else 0) \
            if prefill_rows is None else prefill_rows
        self.grows = max(rows, RUN_ROWS) if self.prefill_rows else rows
        self.io = {"x": b.alloc(4 * H * rows), "cos": b.alloc(2 * spec.rope_dim * rows),
                   "sin": b.alloc(2 * spec.rope_dim * rows), "gf": b.alloc(4 * H),
                   "logits": b.alloc(4 * spec.vocab * rows),
                   "hs": b.alloc(4 * 2 * self.nl),     # per pair: decays of a, b; betas
                   "gr": b.alloc(4 * 2 * self.nl * self.grows),    # chunked prefill, per row:
                   "on": b.alloc(4 * 4 * spec.lin_dv * self.grows)}    # decays, betas; a head
        if spec.mtp:                # the final norm's rows (the MTP's input), the draft ids
            self.io.update(hid=b.alloc(4 * H * rows), draft=b.alloc(4 * rows))   # group's outputs
        if self.prefill_rows:       # its residual stream, and moe_ffn_rows' outputs (a
            self.io["xbuf"] = b.alloc(4 * H * self.prefill_rows)     # request's ids and a sink)
            if spec.moe is not None:
                self.io["moe_scratch"] = b.alloc(4 * H * (LINE // 4 + 1))
        self.layer0 = b.next
        mlp = {"wg": (self.f_loc, H), "wu": (self.f_loc, H)}
        mo = spec.moe
        nl = self.nl
        self.og = 4 if nl % 4 == 0 else 2               # heads per out_proj MM
        self.mats = {LIN: {"wh": (nl // 2 * self.RP, H), "wab": (2 * nl, H),
                           "wout": (nl // self.og * H, self.og * dv),
                           **mlp},
                     ATTN: {"wq": (self.nq_loc * d, H), "wgate": (self.nq_loc * d, H),
                            "wk": (self.nkv_loc * d, H), "wv": (self.nkv_loc * d, H),
                            "wo": (self.h_loc, spec.n_q * d), **mlp}}
        self.grouped = nl // 2 > PAIR_LOOP if spec.pair_loop is None else spec.pair_loop
        if self.grouped:            # per head group: its pairs' blocks, then its out_proj block
            del self.mats[LIN]["wh"], self.mats[LIN]["wout"]
            self.sbytes = 4 * 2 * dv * dk               # a pair's state slots: [slot, head]
        else:
            self.sbytes = 4 * nl * dv * dk              # the layer's states: [slot, head]
        # formats group -> its block layouts (_layout); layer 0's those of the image
        self.layouts = {g: self._layout(g, mlp, rb) for g in dict.fromkeys(self.lf)}
        g0 = self.layouts[self.lf[0]]
        self.lofs, self.mf, self.dchunk = g0.lofs, g0.mf, g0.dchunk
        if self.grouped:
            self.pofs, self.PS, self.gofs, self.GS = g0.pofs, g0.PS, g0.gofs, g0.GS
        # the runs (plan): a layer's key is its kind and formats group; then the MTP layer
        self.keys = tuple(zip(spec.kinds, self.lf))
        self.plan = plan(self.keys)
        self.loc = _place(b, self.plan, lambda k: self.layouts[k[1]].size[k[0]])
        if spec.mtp:                # an attention block after the model's (the formats without a
            gm = tuple(fmt(k, spec.layers) for k in ("delta", "attn", "gateup", "down"))  # range)
            if gm not in self.layouts:
                self.layouts[gm] = self._layout(gm, mlp, rb)
            self.keys += ((ATTN, gm),)
            self.loc.update(_place(b, [(spec.layers, self.keys[-1:], 1)],
                                   lambda k: self.layouts[k[1]].size[k[0]]))
        n_attn = spec.kinds.count(ATTN)
        head = cap * d + 4 * cap * (d // D) + d * cap + 4 * cap
        self.kv_bytes = ((n_attn + spec.mtp) * self.nkv_loc * head      # KV cache, conv ring
                         + (spec.layers - n_attn) * 4 * self.slots *
                         (nl // 2 * (K - 1) * self.CP + nl * dv * dk))   # and state
        self.mtpo, self.nd = {}, min(MTP_VOCAB, spec.vocab)
        if spec.mtp:                # the MTP's norms (embedding, hidden, output), fc, draft head
            self.mtpo = {"ge": b.alloc(4 * H), "gh": b.alloc(4 * H), "gm": b.alloc(4 * H),
                         "fc": (b.alloc(H * rb(2 * H, self.mtp_fc)),       # (attn's)
                                b.alloc(4 * H * (2 * H // D))),
                         "dh": (b.alloc(self.nd * Q.row_bytes(H, "fp4", D)),
                                b.alloc(4 * self.nd * (H // D)))}
        self.head = (b.alloc(self.v_loc * Q.row_bytes(H, self.head_format, D)),
                     b.alloc(4 * self.v_loc * (H // D)))
        # the int8 embedding rows of the resident decode are the tied int8 head's (S = 1)
        shared = spec.tied and self.head_format == "int8" and S == 1
        if embed_host is None:      # an int8 table of its own on the host by default where its
            embed_host = (bool(lookup) and not shared         # DRAM is expert slots
                          and getattr(spec, "embed", "f32") == "int8" and spec.moe is not None)
        if embed_host and not lookup:
            raise ValueError("embed_host needs the image's lookup tables (lookup=True)")
        self.lookup = _lookup_alloc(b, spec, cap, D=D, head=self.head if shared else None,
                                    M=cfg.MCOLS, embed_host=bool(embed_host),
                                    rows=self.grows) if lookup else {}  # (layer 0's R rows)
        if lookup and spec.mtp:     # the MTP loop's chain area (opentpu/llm/mtp.py)
            from .mtp import gen_alloc
            self.lookup["mtpgen"] = gen_alloc(b, cap, spec)
        self.choices = {"embed_host": self.embed_host,          # (the compile worker's
                        "formats": self.formats}                # image)
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

    @property
    def mtp_fc(self) -> str:
        """The MTP's fc format: its layer's attention format."""
        return self.layouts[self.keys[self.spec.layers][1]].mf["wq"]

    def _layout(self, g: tuple, mlp: dict, rb) -> SimpleNamespace:
        """The layer block layouts of formats group g (DeltaNet, attention, gate / up, down):
        each projection's format (mf), W_down's chunk, the DeltaNet and attention blocks'
        offsets (the norms and the MLP first, at the same offsets in both) and sizes (size);
        group-major, a pair's block (pofs, PS) and a head group's (gofs, GS)."""
        spec, cfg, cap = self.spec, self.cfg, self.cap
        D, H, d, F_ = cfg.D, spec.hidden, spec.head_dim, spec.ffn
        dk, dv, nl, og = spec.lin_dk, spec.lin_dv, self.nl, self.og
        fl, fa, fg, fd = g
        mf = {"wh": fl, "wab": fl, "wout": fl, "wq": fa, "wgate": fa, "wk": fa, "wv": fa,
              "wo": fa, "wg": fg, "wu": fg, "wd": fd}       # each projection's format (KINDS)
        lb = _Bump()                                    # offsets inside one layer block
        common = {"g_in": lb.alloc(4 * H), "g_post": lb.alloc(4 * H)}
        for name, (n, k) in mlp.items():
            common[name] = (lb.alloc(n * rb(k, mf[name])), lb.alloc(4 * n * (k // D)))
        dchunk = _chunk(self.f_loc, D, D if fd == "int8" else 2 * D)
        common["wd"] = [(lb.alloc(self.h_loc * rb(dchunk, fd)),
                         lb.alloc(4 * self.h_loc * (dchunk // D)))
                        for _ in range(F_ // dchunk)]
        mo = spec.moe
        if mo is not None:          # the router (int8, the shared expert's gate its last row)
            common.update(router=(lb.alloc((mo.E + 1) * H), lb.alloc(4 * (mo.E + 1) * (H // D))),
                          gbase=lb.alloc(4))
        ns = SimpleNamespace(mf=mf, dchunk=dchunk)
        lnb = _Bump(lb.next)
        lin = dict(common, alog=lnb.alloc(4 * nl), dtb=lnb.alloc(4 * nl), gn=lnb.alloc(4 * dv))
        if self.grouped:            # per head group: its pairs' blocks, then its out_proj block
            pb, gb = _Bump(), _Bump()
            ns.pofs = {"wh": (pb.alloc(self.RP * rb(H, fl)), pb.alloc(4 * self.RP * (H // D))),
                       "cv": pb.alloc(4 * self.CVW),
                       "state": pb.alloc(4 * 2 * dv * dk * self.slots)}
            ns.PS = pb.next                             # a pair's block, bytes
            for _ in range(og // 2):
                gb.alloc(ns.PS)
            ns.gofs = {"wout": (gb.alloc(H * rb(og * dv, fl)),
                                gb.alloc(4 * H * (og * dv // D))),
                       "eb": gb.alloc(4 * 4 * (og // 2))}       # its pairs' gates, per token
            ns.GS = gb.next                             # a head group's block
            lin["groups"] = lnb.alloc(nl // og * ns.GS)
        else:
            lin.update(cv=lnb.alloc(4 * nl // 2 * self.CVW),  # per pair: taps, then the ring
                       state=lnb.alloc(4 * nl * dv * dk * self.slots))
        ab = _Bump(lb.next)
        attn = dict(common, qn=ab.alloc(4 * d), kn=ab.alloc(4 * d))
        for kind, bump, L in ((LIN, lnb, lin), (ATTN, ab, attn)):
            for name, (n, k) in self.mats[kind].items():
                if name not in L:
                    L[name] = (bump.alloc(n * rb(k, mf[name])), bump.alloc(4 * n * (k // D)))
        attn["kv"] = [{"k": ab.alloc(cap * d), "ks": ab.alloc(4 * cap * (d // D)),
                       "vt": ab.alloc(d * cap), "vs": ab.alloc(4 * cap)}
                      for _ in range(self.nkv_loc)]
        ns.lofs = {LIN: lin, ATTN: attn}
        ns.size = {LIN: (lnb.next + 4095) // 4096 * 4096, ATTN: (ab.next + 4095) // 4096 * 4096}
        return ns

    def _off(self, li, it=None) -> Affine:
        """The block address of layer li (static), or of element li of its run's unit at
        iteration `it` (a loop variable)."""
        base, us, i, o, _ = self.loc[li]
        return Affine(base + o) + Affine.of(i if it is None else it) * us

    def pair_offset(self, q: int, lay=None) -> int:
        """Pair q's block in a group-major DeltaNet layer block of layouts `lay` (layer 0's by
        default) (bytes from its start)."""
        lay = self.layouts[self.lf[0]] if lay is None else lay
        g, i = divmod(q, self.og // 2)
        return lay.lofs[LIN]["groups"] + g * lay.GS + i * lay.PS

    def cv_offset(self, q: int, lay=None) -> int:
        """Pair q's taps, then its window, in a DeltaNet layer block of layouts `lay` (layer
        0's by default) (bytes from its start)."""
        lay = self.layouts[self.lf[0]] if lay is None else lay
        if self.grouped:
            return self.pair_offset(q, lay) + lay.pofs["cv"]
        return lay.lofs[LIN]["cv"] + 4 * q * self.CVW

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

        def put_q1(s, addr_pair, a, fmt):
            q, sc = QC.quantize_mxu(a, fmt, D)
            put(s, addr_pair[0], q)
            put(s, addr_pair[1], sc)

        def put_q(addr_pair, parts, fmt):
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
        layers = [(i, kind, f"model.layers.{i}.") for i, kind in enumerate(spec.kinds)]
        if spec.mtp:                                # the MTP layer: one more attention block
            layers.append((spec.layers, ATTN, "mtp.layers.0."))
        for i, kind, p in layers:
            base, lay = self._off(i).const, self.layouts[self.keys[i][1]]
            mf = lay.mf
            Lo = {k: (tuple(base + x for x in v) if isinstance(v, tuple) else
                      (base + v if isinstance(v, int) else v)) for k, v in lay.lofs[kind].items()}
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
                            o = base + self.pair_offset(q, lay)
                            put_q1(s, (o + lay.pofs["wh"][0], o + lay.pofs["wh"][1]), pr,
                                   mf["wh"])
                            put(s, o + lay.pofs["cv"], pt)
                        for g, go in enumerate(gouts[s]):
                            o = Lo["groups"] + g * lay.GS
                            put_q1(s, (o + lay.gofs["wout"][0], o + lay.gofs["wout"][1]), go,
                                   mf["wout"])
                else:
                    put_q(Lo["wh"], [np.concatenate(pr) for pr in prows], mf["wh"])
                    put_q(Lo["wout"], [np.concatenate(go) for go in gouts], mf["wout"])
                    for s in range(S):
                        for q, pt in enumerate(ptaps[s]):
                            put(s, Lo["cv"] + q * 4 * self.CVW, pt)
                put_q(Lo["wab"], [np.concatenate([W[a + "in_proj_a.weight"][hh.start:hh.stop],
                                                  W[a + "in_proj_b.weight"][hh.start:hh.stop]])
                                  for hh in hs], mf["wab"])
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
                put_q(Lo["wq"], rows(wq, self.nq_loc * d), mf["wq"])
                put_q(Lo["wgate"], rows(wgate, self.nq_loc * d), mf["wgate"])
                put_q(Lo["wk"], rows(wk, self.nkv_loc * d), mf["wk"])
                put_q(Lo["wv"], rows(wv, self.nkv_loc * d), mf["wv"])
                put_q(Lo["wo"], rows(wo, n), mf["wo"])
            mp = spec.mlp_prefix(p)
            put_q(Lo["wg"], rows(W[mp + "gate_proj.weight"], self.f_loc), mf["wg"])
            put_q(Lo["wu"], rows(W[mp + "up_proj.weight"], self.f_loc), mf["wu"])
            C_ = lay.dchunk
            for j, pair in enumerate(lay.lofs[kind]["wd"]):
                put_q((base + pair[0], base + pair[1]),
                      [r[:, j * C_:(j + 1) * C_] for r in rows(W[mp + "down_proj.weight"], n)],
                      mf["wd"])
            if spec.moe is not None:
                put_q(Lo["router"], [np.concatenate([W[p + "mlp.gate.weight"],
                                                     W[p + "mlp.shared_expert_gate.weight"]])],
                      "int8")
                put(0, Lo["gbase"], f32([i * spec.moe.E]))
        head = W["model.embed_tokens.weight"] if spec.tied else W["lm_head.weight"]
        if spec.mtp:                # (one slice: Spec.check)
            mo = self.mtpo
            for k, n in (("ge", "pre_fc_norm_embedding"), ("gh", "pre_fc_norm_hidden"),
                         ("gm", "norm")):
                put(0, mo[k], g1(f"mtp.{n}.weight"))
            put_q(mo["fc"], [W["mtp.fc.weight"]], self.mtp_fc)
            put_q(mo["dh"], [head[:self.nd]], "fp4")
        put_q(self.head, rows(head, self.v_loc), self.head_format)
        if self.lookup:
            _lookup_build(put, S, W, spec, self.cap, self.lookup)
        if "mtpgen" in self.lookup:
            from .mtp import gen_build
            for s in range(S):
                gen_build(put, s, S, spec, self.lookup["mtpgen"])
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
        return [qwen35_step.trace(self.cfg, s, {"m": step_descriptors(self, s), "pos": pos,
                                                "block": block, **_tok_arg(self, tok)}).finish()
                for s in range(self.cfg.S)]

    def compile_rows(self, rows, logit_rows, block: int = ATTN_BLOCK, tokens=None,
                     slot: int = 0, fork: bool = False, hidden: bool = False) -> list:
        """One program per slice: consecutive positions of the sequence at once
        (qwen35_rows); slot, fork, hidden: MTP decoding's (qwen35_rows, docs/mtp.md 9)."""
        if self.spec.moe is not None:
            raise ValueError("a MoE model runs its prompt through the decode step (its MoE "
                             "block routes one token): rows=1")
        if len(rows) > self.rows:
            raise ValueError(f"{len(rows)} rows, the image's I/O area holds {self.rows}")
        if any(r != (0, rows[0][1] + i) for i, r in enumerate(rows)):
            raise ValueError("Qwen3.5 rows must be consecutive positions of sequence 0")
        if (fork or hidden) and not self.spec.mtp:
            raise ValueError("fork / hidden rows need an MTP image (Spec.mtp)")
        mtp = {"fork": True} if fork else {}
        if hidden:
            mtp["hidden"] = True
        return [qwen35_rows.trace(self.cfg, s, {"m": self.descriptors(s, slot), "p0": rows[0][1],
                                                "R": len(rows), "logit_rows": list(logit_rows),
                                                "block": block, **mtp,
                                                **_tokens_arg(self, tokens, rows)}).finish()
                for s in range(self.cfg.S)]

    def compile_mtp(self, p0: int, R: int, tokens=None, block: int = ATTN_BLOCK,
                    keep: bool = False, h0: int = 0) -> list:
        """One program per slice: the MTP drafter over R rows at positions p0 .. (qwen35_mtp;
        Spec.mtp), the tokens' embedding rows from the image's tables (`tokens`) or the I/O
        area's x rows (the host's, as qwen35_rows'). keep: the draft head's logits to the I/O
        area's logits rows too. h0: the first hidden row (m.hid[h0:h0 + R])."""
        if not self.spec.mtp:
            raise ValueError("the MTP drafter needs an MTP image (Spec.mtp)")
        if not 0 < R <= self.rows or not 0 <= h0 <= self.rows - R:
            raise ValueError(f"MTP rows {h0}..{h0 + R}, the image's I/O area holds {self.rows}")
        rows = [(0, p0 + r) for r in range(R)]
        return [qwen35_mtp.trace(self.cfg, s, {"m": self.descriptors(s), "p0": p0, "R": R,
                                               "block": block, "keep": keep, "h0": h0,
                                               **_tokens_arg(self, tokens, rows)}).finish()
                for s in range(self.cfg.S)]

    def compile_layer_run(self, li: int, blocks: int, block: int = ATTN_BLOCK, R: int = 1,
                          embedded: bool = False, at: int | None = None, hint: bool = False,
                          em: bool = False):
        """(programs, run_args): qwen35_layer_run of layer li, R rows at a run-time position of
        bucket `blocks` (from conv_k - 1: the convolutions' taps) and a run-time row of the
        prefill chunk (run arguments: RunPos.values and "row"), or at the compile-time
        position `at` (the rows before conv_k - 1; embedded); li < 0: qwen35_embed_run. hint:
        the run ends with the next MoE layer's hint (qwen35_layer_run); em: expert-major
        (docs/offload.md 13.11: the rows in the scratch's records, the MoE's prologue and the
        last layer's combine; compile_expert_run). The image needs lookup tables and its
        prefill rows."""
        from ..compiler import RunVar
        spec, K = self.spec, self.spec.conv_k
        if not self.lookup or not self.prefill_rows:
            raise ValueError("a layer run needs lookup tables and the image's prefill rows")
        if self.cfg.S != 1:
            raise ValueError("a layer run is one slice's (S = 1)")
        k = spec.moe.k if spec.moe is not None else 1
        if not 1 <= R <= RUN_ROWS or R * k > LINE // 4:
            raise ValueError(f"{R} rows a layer run")
        if (hint or em) and self.offload is None:
            raise ValueError("a layer run's hint or expert-major MoE needs the expert server's "
                             "words (offload)")
        if hint and em:
            raise ValueError("expert-major layer runs post their own layer's needs, no hints")
        row = RunVar("row", self.prefill_rows)
        if R > 1 and li == 0 and not embedded and not self.embed_host:
            raise ValueError("layer 0's run of rows gathers them from the host's slot "
                             "(embed_host)")
        if at is not None:
            if li < 0 or not embedded:
                raise ValueError("a compile-time layer run takes embedded rows")
            pos = at
        else:
            lo = max((blocks - 1) * block, K - 1 if li >= 0 else 0)
            pos = RunPos(blocks, block, lo, self.lookup["zmask"], self.cap)
            pos.tpos.bound -= R - 1     # the run's last row in the block too: tpos <= block - R
        m = self.descriptors(0)
        if li < 0:
            b = qwen35_embed_run.trace(self.cfg, 0, {"m": m, "pos": pos, "row": row, "em": em})
        else:
            b = qwen35_layer_run.trace(self.cfg, 0, {"m": m, "li": li, "pos": pos, "row": row,
                                                     "block": block, "R": R,
                                                     "embedded": embedded, "hint": hint,
                                                     "em": em})
        return [b.finish()], list(b.run_args)

    def compile_prefill_head(self, em: bool = False):
        """(programs, run_args): qwen35_prefill_head at a run-time row ("row"); em: from the
        expert-major records, after the last layer's combine."""
        from ..compiler import RunVar
        b = qwen35_prefill_head.trace(self.cfg, 0, {"m": self.descriptors(0),
                                                    "row": RunVar("row", self.prefill_rows),
                                                    "em": em})
        return [b.finish()], list(b.run_args)

    def compile_expert_run(self, li: int):
        """(programs, run_args): moe.moe_expert_run of MoE layer li (expert-major,
        docs/offload.md 13.11), the chunk's rows and entries (rows x k) run arguments ("rows",
        "entries")."""
        if self.offload is None or not self.prefill_rows or self.cfg.S != 1:
            raise ValueError("an expert run needs the expert server's words, the image's "
                             "prefill rows and one slice")
        b = qwen35_expert_run.trace(self.cfg, 0, {"m": self.descriptors(0), "li": li})
        return [b.finish()], list(b.run_args)

    def _run_rows(self, kernel, blocks: int, lo: int, R: int, block: int, slot: int, **kw):
        if not self.lookup:
            raise ValueError("a run at a run-time position needs lookup tables (lookup=True)")
        if not (blocks - 1) * block <= lo < min(blocks * block, self.cap):
            raise ValueError(f"lo {lo} is not in bucket {blocks}")
        rp = RunRows(blocks, block, lo, self.lookup["zmask"], self.cap, R)
        bs = [kernel.trace(self.cfg, s, {"m": self.descriptors(s, slot), "p0": rp, "R": R,
                                         "block": block, **kw}) for s in range(self.cfg.S)]
        return [b.finish() for b in bs], list(bs[0].run_args)

    def compile_rows_run(self, blocks: int, lo: int, R: int, logit_rows, block: int = ATTN_BLOCK,
                         slot: int = 0, fork: bool = False, hidden: bool = False):
        """compile_rows at a run-time position (qwen3.RunRows): R rows from p, every p of
        bucket `blocks` from lo with its R rows in the bucket, their tokens run-time values:
        (programs, run_args); compiler.arg_words of RunRows.values gives the arguments."""
        if fork or hidden:
            if not self.spec.mtp:
                raise ValueError("fork / hidden rows need an MTP image (Spec.mtp)")
        return self._run_rows(qwen35_rows, blocks, lo, R, block, slot,
                              logit_rows=list(logit_rows), fork=fork, hidden=hidden)

    def compile_prompt_run(self, blocks: int, R: int, kind: str, block: int = ATTN_BLOCK,
                           hidden: bool = False, slot: int = 0, p0: int | None = None):
        """qwen35_prompt_run's (programs, run_args) (docs/prefill.md), a program per slice: R
        rows of a prompt at a run-time position of bucket `blocks` (from conv_k - 1; the
        position in the state's tpos word), or at the compile-time position p0 (the rows
        before conv_k - 1; no run_args)."""
        spec = self.spec
        if spec.moe is not None:
            raise ValueError("a MoE model's prompt runs layer by layer (compile_layer_run)")
        if not self.lookup:
            raise ValueError("a prompt run needs lookup tables (lookup=True)")
        if (hidden or kind == "M") and not spec.mtp:
            raise ValueError("MTP's prompt runs need an MTP image (Spec.mtp)")
        if kind not in ("P", "L", "M"):
            raise ValueError(f"prompt run kind {kind!r}")
        if p0 is None:
            lo = max((blocks - 1) * block, spec.conv_k - 1)
            pos = RunRows(blocks, block, lo, self.lookup["zmask"], self.cap, R,
                          1 if kind == "M" else 0, _amask(self.lookup))
        else:
            pos = p0
        bs = [qwen35_prompt_run.trace(self.cfg, s, {"m": self.descriptors(s, slot), "pos": pos,
                                                    "R": R, "kind": kind, "block": block,
                                                    "hidden": hidden})
              for s in range(self.cfg.S)]
        return [b.finish() for b in bs], list(bs[0].run_args)

    def compile_mtp_run(self, blocks: int, lo: int, R: int, block: int = ATTN_BLOCK,
                        h0: int = 0):
        """compile_mtp at a run-time position (RunRows; see compile_rows_run)."""
        if not self.spec.mtp:
            raise ValueError("the MTP drafter needs an MTP image (Spec.mtp)")
        return self._run_rows(qwen35_mtp, blocks, lo, R, block, 0, h0=h0)

    # ---- kernel descriptors
    def descriptors(self, sid: int, slot: int = 0) -> SimpleNamespace:
        """The kernels' descriptors of slice sid; `slot`: the DeltaNet states' and windows'
        committed slot (DeltaNetParts; Spec.mtp)."""
        spec, cfg = self.spec, self.cfg
        D, d, H, K, n = cfg.D, spec.head_dim, spec.hidden, spec.conv_k, self.h_loc
        dk, dv, nl, C = spec.lin_dk, spec.lin_dv, self.nl, self.C
        sl = dict(tp=K * self.CP, wwords=(K - 1) * self.CP, slot=slot,
                  sbytes=self.sbytes if self.slots > 1 else 0)

        def layer(li, it=None):
            """Descriptors of layer `li` (static), or of element li of its run's unit at
            iteration `it` (a hardware-loop variable). Its kind and formats group from
            self.keys (the MTP layer's: li = spec.layers)."""
            kind, g = self.keys[li]
            lay = self.layouts[g]
            off = self._off(li, it)
            lofs, mf = lay.lofs[kind], lay.mf
            ns = SimpleNamespace(g_in=Tensor(off + lofs["g_in"], (H,), (1,)),
                                 g_post=Tensor(off + lofs["g_post"], (H,), (1,)),
                                 moe=spec.moe is not None, kind=kind)
            if ns.moe:
                E = spec.moe.E
                da, sa = lofs["router"]
                ns.router = QTensor(off + da, off + sa, (E + 1, H), H, 4 * (H // D), D)
                ns.gbase = Tensor(off + lofs["gbase"], (1,), (1,))
            for name, (r, k) in self.mats[kind].items():
                da, sa = lofs[name]
                fm = mf[name]
                setattr(ns, name, QTensor(off + da, off + sa, (r, k), Q.row_bytes(k, fm, D),
                                          4 * (k // D), D, wf=Q.mxu_wf(fm)))
            Cd = lay.dchunk
            fm = mf["wd"]
            wf = Q.mxu_wf(fm)
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
                    fm = mf["wh"]
                    wf = Q.mxu_wf(fm)
                    g0 = off + lofs["groups"]
                    (whd, whs), (wod, wos) = lay.pofs["wh"], lay.gofs["wout"]
                    ns.dn = DeltaNetParts(
                        nl, og, QTensor(g0 + whd, g0 + whs, (R2, H), Q.row_bytes(H, fm, D),
                                        4 * (H // D), D, wf=wf),
                        Tensor(g0 + lay.pofs["cv"], (self.CV0,), (1,)),
                        Tensor(g0 + lay.pofs["state"], (2, dv, dk), (dv * dk, dk, 1)),
                        QTensor(g0 + wod, g0 + wos, (H, og * dv), Q.row_bytes(og * dv, fm, D),
                                4 * (og * dv // D), D, wf=wf), H, lay.GS, lay.PS,
                        Tensor(g0 + lay.gofs["eb"], (og // 2, 4), (4, 1)), self.shared, **sl)
                else:
                    # (a pair's taps and window 0: the decode loads them at once; window 1
                    # after them, DeltaNetParts.window)
                    ns.cv = Tensor(off + lofs["cv"], (nl // 2, self.CV0), (self.CVW, 1))
                    ns.state = Tensor(off + lofs["state"], (nl, dv, dk), (dv * dk, dk, 1))
                    ns.dn = DeltaNetParts(nl, og, ns.wh, ns.cv, ns.state, ns.wout, H,
                                          shared=self.shared, **sl)
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
            dev = SimpleNamespace(mbox=L.mbox, served=L.served, answer=L.answer, dir=L.dir,
                                  tag=L.tag, fmt=self.fmt, hint_off=L.layers * L.E,
                                  scratch=self.io.get("moe_scratch"),
                                  need_off=2 * L.layers * L.E, em_base=L.slots[0][0],
                                  em_rec=MO.em_record(H, spec.moe.k),
                                  em_rows=self.prefill_rows)
        mtp = {}
        if spec.mtp:
            mo, ff = self.mtpo, self.mtp_fc
            mtp = dict(hid=_tdesc(self.io["hid"], (self.rows, H)), nd=self.nd,
                       draft=_tdesc(self.io["draft"], (self.rows,)),
                       mtp=SimpleNamespace(
                           layer=layer(spec.layers),
                           ge=_tdesc(mo["ge"], (H,)), gh=_tdesc(mo["gh"], (H,)),
                           gm=_tdesc(mo["gm"], (H,)),
                           fc=QTensor(*mo["fc"], (H, 2 * H), Q.row_bytes(2 * H, ff, D),
                                      4 * (2 * H // D), D, wf=Q.mxu_wf(ff)),
                           dh=_qdesc(*mo["dh"], self.nd, H, D, "fp4")),
                       mtpgen=self.lookup.get("mtpgen"))
        return SimpleNamespace(
            spec=spec, layer=layer, plan=self.plan, moe_dev=dev, **mtp,
            x=_tdesc(self.io["x"], (1, H)), cos=_tdesc(self.io["cos"], (spec.rope_dim // 2,)),
            sin=_tdesc(self.io["sin"], (spec.rope_dim // 2,)), g_final=_tdesc(self.io["gf"], (H,)),
            logits=_tdesc(self.io["logits"], (1, spec.vocab)),
            xr=_tdesc(self.io["x"], (self.rows, H)),
            cosr=_tdesc(self.io["cos"], (self.rows, spec.rope_dim // 2)),
            sinr=_tdesc(self.io["sin"], (self.rows, spec.rope_dim // 2)),
            logitsr=_tdesc(self.io["logits"], (self.rows, spec.vocab)),
            hs=_tdesc(self.io["hs"], (nl // 2, 4)),
            gr=_tdesc(self.io["gr"], (self.grows, 2 * nl)),
            on=_tdesc(self.io["on"], (self.grows, self.og * dv)),
            xbuf=_tdesc(self.io["xbuf"], (self.prefill_rows, H)) if self.prefill_rows else None,
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
        """Pair p's taps and convolution window (one load; slot 1's window: two)."""
        if dn.slot:
            ol.load(dn.cv(p)[0:TP], out=CV[0:TP])
            ol.load(dn.window(p), out=CV[TP:TP + (K - 1) * CP])
        else:
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
        win = dn.window(p)
        s0 = (K - 2) * CP + a * bw                      # the window's last row: this position
        ol.store(win[s0:s0 + n * bw], pre)
        if a + n == nb:                                 # its other rows, one up (_past)
            ol.store(win[0:(K - 2) * CP], CV[TP + CP:TP + (K - 1) * CP])
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
        if dn.slot:                                     # (slot 1's window: two loads)
            ol.load(dn.cv(p)[0:TP], out=CV[0:TP])
            ol.load(dn.window(p), out=CV[TP:TP + (K - 1) * CP])
        else:
            ol.load(dn.cv(p), out=CV)

    def fetch_eb(p, t):
        ol.load(dn.gates(p, hs), out=EB[t])

    def conv(p, t, j=None):
        a, n = hb(j)
        pre = P[t][0, a * bw:(a + n) * bw]
        win = dn.window(p)
        s0 = (K - 2) * CP + a * bw                      # the window's last row: this position
        ol.store(win[s0:s0 + n * bw], pre)
        if a + n == nb:                                 # its other rows, one up (_past)
            ol.store(win[0:(K - 2) * CP], CV[TP + CP:TP + (K - 1) * CP])
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
    fill_logits(m)
    spec = m.spec
    x, c, s_ = _inputs(m, pos, tok)

    def layer(li, it):
        lw = m.layer(li, it)
        if lw.moe and spec.moe.hint:        # the router's guess, before the mixer
            MO.moe_hint(x, lw, spec.moe, m.moe_dev, spec.eps)
        if lw.kind == LIN:
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


def _deltanet_rows(x, lw, p0: int, spec: Spec, gr, on, fork: bool = False):
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
    streams it through its datapath and back), row after row, instead of the VPU passes.

    fork (the speculative verify, docs/mtp.md 9): the last row's state and window go to the
    other slot (DeltaNetParts), so the committed slot keeps those after the row before it:
    the last row's step is a STREAM into the other slot (or, on the VPU path, the state is
    stored after the row before it too)."""
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
    if fork and (R < 2 or p0 < K - 1):
        raise CompileError("a forked rows run needs 2 rows or more, from position conv_k - 1")

    def fdst(p, a, r):
        """Where row r's step writes head a of pair p: in place, or (fork, the last row) the
        other slot."""
        return dn.state(p, a, 1 - dn.slot) if fork and r == R - 1 else None

    full = max(0, K - 1 - p0)                           # rows before it lack positions < 0
    rgroups = [(r, r + 1) for r in range(min(full, R))] + ([(full, R)] if full < R else [])
    split = og == 4 and NP > 1                          # _deltanet's last group, in pairs

    def flush(g, ON, c0, c1):
        """y += ON . out_proj columns [c0, c1) of head group g (heads of dv columns)."""
        ol.dot(ON, dn.wout(g)[:, c0 * dv:c1 * dv], acc=y)

    def head_in(X, taps, a, out=None, act=True):
        """Block a of the channels of the pair whose rows are in X (head a's q k v; shared, q, k,
        v of a, v of b): the convolution and SiLU (act) -> [R, bw] (into `out`)."""
        U = ol.empty([R, bw]) if out is None else out
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
        if act:
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
                                 GDB[r, 2 + a:3 + a], O[r, :], zero=(p0 == 0 and r == 0),
                                 dst=fdst(p, a, r))
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
            if fork and r == R - 1:
                ol.store(dn.state(p, a), St)
            dh, bh = GD[r, a:a + 1], GB[r, a:a + 1]
            w.set(St @ Kn[r, :])
            w.set((U[r, 2 * dk:C] - w * dh) * bh)
            ol.outer(w, Kn[r, :], acc=St, decay=dh)
            O[r, :].set(St @ Qn[r, :])
        ol.store(dn.state(p, a, 1 - dn.slot) if fork else dn.state(p, a), St)
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
                                     GDB[r, 2 + a:3 + a], O[a][r, :], zero=(p0 == 0 and r == 0),
                                     dst=fdst(p, a, r))
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
                if fork and r == R - 1:
                    ol.store(dn.state(p, a), St)
                dh, bh = GD[r, a:a + 1], GB[r, a:a + 1]
                w.set(St @ Kn[r, :])
                w.set((V[r, :] - w * dh) * bh)
                ol.outer(w, Kn[r, :], acc=St, decay=dh)
                O[r, :].set(St @ Qn[r, :])
            ol.store(dn.state(p, a, 1 - dn.slot) if fork else dn.state(p, a), St)
            ONp[:, a * dv:(a + 1) * dv].set(rmsnorm(O, gn, eps) * GZ)
            del V, GZ, O

    # a pair's projections, taps and ring rows; two sets with DSTEP (pair p + 1's projections
    # stream while pair p's DSTEPs run)
    NB = 2 if dstep and split and gp == 2 else 1
    taps1 = ol.empty([K * CP]) if NB == 2 else None     # (_rows_pipelined: one taps buffer)
    PB = [(ol.empty([K * CP]) if taps1 is None else taps1, ol.empty([K - 1 + R, CP]),
           ol.empty([R, 2 * dv])) for _ in range(NB)]

    def project(p, b, window=True):
        """Pair p's taps, q k v rows (ring, then its projections) and z into buffers b; window:
        store the next window (else store_window after the pair's DSTEPs: a store waiting for
        the projections would hold the DMA's queue, and the DSTEPs behind it)."""
        taps, X, Z = PB[b]
        ol.load(dn.cv(p)[0:TP], out=taps)              # rows (block, tap)
        win = dn.window(p)
        for j in range(1, min(K, p0 + 1)):              # positions p0-K+1 ..: the window
            sl = (K - 1 - j) * CP                       # (_past)
            ol.load(win[sl:sl + CP], out=X[K - 1 - j, :])
        ol.dot(xs, dn.wh(p)[0:CP, :], out=X[K - 1:K - 1 + R, :])
        ol.dot(xs, dn.wh(p)[CP:RP, :], out=Z)                            # z of a, of b
        if window:
            store_window(p, b)

    def store_window(p, b):
        """The next window; fork: the one after the row before the last in the committed slot,
        the last row's in the other."""
        X = PB[b][1]
        for s, r0 in ((dn.slot, R - 1), (1 - dn.slot, R)) if fork else ((dn.slot, R),):
            win = dn.window(p, s)
            for i in range(K - 1):
                ol.store(win[i * CP:(i + 1) * CP], X[r0 + i, :])

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
                                 GDB[r, 2 + a:3 + a], O[a][r, :], zero=(p0 == 0 and r == 0),
                                 dst=fdst(p, a, r))
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

    def _rows_pipelined():
        """The pairs software pipelined as _deltanet_dstep, two pairs ahead (pair p in buffer
        set t = p % 2), so that the MXU and the VPU work while the DMA runs the DSTEPs:

            MXU  the q k v projections of pair p + 2
            DMA  pair p's 2 R DSTEPs
            VPU  pair p + 1's convolution, SiLU and norms beside them (its window stored),
                 then pair p's gated RMSNorm into its columns of ON (TMEM)
            MXU  the z projections of pair p + 2 (after pair p's gate read its z), then the
                 out_proj of the head group pair p completes (the last group pair by pair)
            DMA  pair p + 2's taps, window and gates

        Per pair and row the operations of heads() / shared_heads(), out_proj over the same
        groups in the same order: the same words. The VPU's steps take both heads at once (a
        SiLU over the pair's channels, a norm over rows (row, head)), as _deltanet_dstep's
        take its heads as rows. A head's q, k (normed in place) and v stay in the pair's
        convolved rows, which DSTEP reads as they are. The taps have one buffer: pair p + 2's
        are loaded after pair p + 1's convolution read pair p + 1's."""
        Ub = [ol.empty([R, CP], dense=True) for _ in range(2)]  # the pair's convolved channels
        O2 = ol.empty([R, 2 * dv], dense=True)          # o of both heads
        O = [O2[:, 0:dv], O2[:, dv:2 * dv]]
        GDBb = [ol.empty([R, 4]) for _ in range(2)]     # decays, then betas, of the pair's heads
        ON = ol.empty([R, og * dv])                     # the group's normed, gated o

        def mm_x(p, t):
            ol.dot(xs, dn.wh(p)[0:CP, :], out=PB[t][1][K - 1:K - 1 + R, :])

        def mm_z(p, t):
            ol.dot(xs, dn.wh(p)[CP:RP, :], out=PB[t][2])                 # z of a, of b

        def fetch(p, t, taps=True):
            X = PB[t][1]
            if taps:
                ol.load(dn.cv(p)[0:TP], out=taps1)     # rows (block, tap)
            win = dn.window(p)
            for j in range(1, min(K, p0 + 1)):          # positions p0-K+1 ..: the window
                sl = (K - 1 - j) * CP                   # (_past)
                ol.load(win[sl:sl + CP], out=X[K - 1 - j, :])
            q = dn.index(p)
            for r in range(R):
                ol.load(gr[r, 2 * q:2 * q + 2], out=GDBb[t][r, 0:2])
                ol.load(gr[r, nl + 2 * q:nl + 2 * q + 2], out=GDBb[t][r, 2:4])

        def prep(p, t):
            taps, X, _ = PB[t]
            taps = taps.reshape(nb * K, bw)
            U = Ub[t]
            for a in range(nb):                         # each block's convolution
                head_in(X, taps, a, out=U[:, a * bw:(a + 1) * bw], act=False)
            U.set(silu(U))
            # q and k L2-normed in place: shared, the key head's; else both heads', as rows
            # (row, head) of the pair's channels
            H = U if sh else U.reshape(2 * R, C)
            H[:, 0:dk].set(l2norm_rows(H[:, 0:dk], dk ** -0.5))
            H[:, dk:2 * dk].set(l2norm_rows(H[:, dk:2 * dk]))
            store_window(p, t)

        def dsteps(p, t):
            U = Ub[t]
            for a in range(2):
                qk = U[:, 0:2 * dk] if sh else U[:, a * C:a * C + 2 * dk]
                v = U[:, (2 + a) * dk:(3 + a) * dk] if sh else U[:, a * C + 2 * dk:(a + 1) * C]
                for r in range(R):
                    ol.deltanet_step(dn.state(p, a), qk[r, :], v[r, :], GDBb[t][r, a:a + 1],
                                     GDBb[t][r, 2 + a:3 + a], O[a][r, :],
                                     zero=(p0 == 0 and r == 0), dst=fdst(p, a, r))

        def post(t):
            """Pair t's gated RMSNorm into its columns of ON: the norm over rows (row, head)."""
            gz = silu(PB[t][2])
            on = rmsnorm(O2.reshape(2 * R, dv), gn, eps).reshape(R, 2 * dv)
            ON[:, 2 * t * dv:(2 * t + 2) * dv].set(on * gz)
            del gz, on

        def segment(p, t, last1, last2, g=None, half=None):
            """Pair p; last1: no pair p + 1, last2: no pair p + 2; g: the head group pair p
            completes; half: pair p's half of the last group (flushed pair by pair)."""
            if not last2:
                mm_x(p + 2, t)
            dsteps(p, t)
            if not last1:
                prep(p + 1, 1 - t)
            post(t)
            if not last2:
                mm_z(p + 2, t)
            if half is not None:
                flush(ng - 1, ON[:, 2 * half * dv:(2 * half + 2) * dv], 2 * half, 2 * half + 2)
            elif g is not None:
                flush(g, ON, 0, og)
            if not last2:
                fetch(p + 2, t)

        def unrolled(p):
            last = p >= NP - 2
            segment(p, p % 2, p + 1 >= NP, p + 2 >= NP,
                    None if last or p % 2 == 0 else (p - 1) // 2, p - (NP - 2) if last else None)

        fetch(0, 0)
        mm_x(0, 0)
        mm_z(0, 0)
        fetch(1, 1, taps=False)
        mm_x(1, 1)
        mm_z(1, 1)
        prep(0, 0)
        ol.load(dn.cv(1)[0:TP], out=taps1)             # after pair 0's convolution
        unrolled(0)
        n_it = max(0, (NP - 3) // 2)                    # segments 1 .. NP - 3: every part
        for i in loop(n_it):
            segment(2 * i + 1, 1, False, False, g=i)
            segment(2 * i + 2, 0, False, False)
        for p in range(1 + 2 * n_it, NP):
            unrolled(p)

    if NB == 2:
        _rows_pipelined()
        return x + ol.all_reduce(y)
    ONp = ol.empty([R, 2 * dv])                         # normed, gated o of a pair per row
    GDB = ol.empty([R, 4])                              # decays, then betas, of the pair's heads
    GD, GB = GDB[:, 0:2], GDB[:, 2:4]
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
def qwen35_rows(m, p0: int, R: int, logit_rows, block: int = ATTN_BLOCK, tokens=None,
                fork: bool = False, hidden: bool = False):
    """R prompt tokens at positions p0 .. p0+R-1 at once: their embeddings m.xr and RoPE
    tables m.cosr / m.sinr, or those of `tokens` from the image's tables (qwen3._inputs_rows) ->
    logits of the rows in `logit_rows` (a contiguous range, or empty). Bit-identical to R
    qwen35_step runs, but for the order of a 4-bit MM's sums with PAIR when 2R > MCOLS: a
    step's MMs pair (column reuse), a wider run's do not (Builder.stationary), and a paired
    MM adds each even K-block to the odd one before the partial sums. The difference is a
    rounding of a sum, which an activation's int8 rounding downstream can turn into one
    quantization step (tests/test_qwen35.py test_prefill_pair_sum_order). Runs of at most
    MCOLS / 2 rows pair as the steps do and are bit-identical.

    MTP decoding (Spec.mtp, docs/mtp.md 9): `hidden` stores every row's final-normed hidden
    (the LM head's input) to m.hid for the MTP layer; `fork` (the verify run) leaves the
    DeltaNet states and windows after row R-2 in the committed slot and those after row R-1
    in the other (_deltanet_rows). p0 a RunRows (R its rows): the run at a run-time position,
    its tokens run-time values (docs/mtp.md 10); the DeltaNet rows do not depend on the
    position from K - 1 on, so they take the bucket's first position."""
    spec = m.spec
    run = isinstance(p0, RunRows)
    rows = p0 if run else [(0, p0 + r) for r in range(R)]
    pd = p0.lo if run else p0
    if run and (R != p0.R or pd < spec.conv_k - 1):
        raise CompileError(f"a run-time rows run: {p0.R} rows from position {pd}")
    x, c, s_ = _inputs_rows(m, rows, tokens)

    def layer(li, it):
        lw = m.layer(li, it)
        if lw.kind == LIN:
            x.set(_deltanet_rows(x, lw, pd, spec, m.gr[0:R, :], m.on[0:R, :], fork))
        else:
            x.set(_attention_rows(x, lw, c, s_, rows, spec, block, gated=True))
        x.set(_mlp(x, lw, spec))

    run_layers(m.plan, layer)
    if hidden:                  # row by row: one row of TMEM (a prompt run's R rows fit)
        g = ol.load(m.g_final)
        for r in range(R):
            ol.store(m.hid[r:r + 1, :], rmsnorm(x[r:r + 1, :], g, spec.eps))
        del g
    _lm_head_rows(x, m, spec, logit_rows)


@ol.jit
def qwen35_mtp(m, p0: int, R: int, block: int = ATTN_BLOCK, keep: bool = False, tokens=None,
               h0: int = 0):
    """The MTP drafter (Spec.mtp; docs/mtp.md 6.1, 9) over R rows at positions p0 .. p0+R-1:
    row r takes the main model's final-normed hidden at p0 + r (m.hid[h0 + r]: a verify run's
    or a prompt run's, qwen35_rows hidden) and the token at p0 + r + 1 (tokens[r], its embedding
    from the image's tables; without tokens m.xr's row, qwen3._inputs_rows) and drafts the
    token at p0 + r + 2, the id of the draft head's largest logit (its rows: the MTP_VOCAB
    lowest ids), into m.draft[r]:

        e = pre_fc_norm_embedding(embed(token)), h = pre_fc_norm_hidden(hid)
        x = fc([e, h])                          one MM, K = 2H, the embedding first
        x = the MTP decoder layer               gated attention over its own KV cache (RoPE at
                                                p0 + r), the MLP: an attention layer block
        draft = ARGMAX(draft head(mtp.norm(x)))

    The layer appends every row's K/V before any row attends, so a row whose hidden belongs to
    a rejected draft is overwritten by the next run's row at that position before anything
    reads it. keep: the draft head's logits to m.logitsr too (tests). p0 a RunRows (R its
    rows): the run at a run-time position, its tokens run-time values (docs/mtp.md 10).
    Returns the drafts' [1] tiles."""
    spec, mt, b = m.spec, m.mtp, current()
    H, eps = spec.hidden, spec.eps
    rows = p0 if isinstance(p0, RunRows) else [(0, p0 + r) for r in range(R)]
    e, c, s_ = _inputs_rows(m, rows, tokens)
    cat = ol.empty([R, 2 * H], dense=True)
    cat[:, 0:H].set(rmsnorm(e, ol.load(mt.ge), eps))
    del e
    cat[:, H:2 * H].set(rmsnorm(ol.load(m.hid[h0:h0 + R, :]), ol.load(mt.gh), eps))
    x = ol.dot(ol.quantize(cat), mt.fc)                 # [R, H] (one slice: Spec.check)
    del cat
    lw = mt.layer
    x.set(_attention_rows(x, lw, c, s_, rows, spec, block, gated=True))
    x.set(_mlp(x, lw, spec))
    xs = ol.quantize(rmsnorm(x, ol.load(mt.gm), eps))
    del x
    chunk = min(HEAD_CHUNK, ol.tmem_words() // (8 * R))
    sinks = [G.Greedy(b, m.nd, chunk) for _ in range(R)]
    for c0 in range(0, m.nd, chunk):
        n = min(chunk, m.nd - c0)
        y = ol.dot(xs, mt.dh[c0:c0 + n, :])
        if keep:
            ol.store(m.logitsr[0:R, c0:c0 + n], y)
        for r, sk in enumerate(sinks):
            sk(y[r:r + 1, :], c0)
        del y
    drafts = [sk.token() for sk in sinks]
    for r, t in enumerate(drafts):
        ol.store(m.draft[r:r + 1], t)
    return drafts


@ol.jit
def qwen35_prompt_run(m, pos, R: int, kind: str, block: int = ATTN_BLOCK, hidden: bool = False):
    """A prompt run of R rows (docs/prefill.md), its tokens from out[]: kind "P" (qwen35_rows),
    "L" (the prompt's last run: its last row's logits too) or "M" (qwen35_mtp over the rows: the
    tokens of the positions after them). pos: a RunRows (toks_at 0, M's 1) at the run-time
    position in the generate state's tpos word (run_words: no host arguments), or the run's
    first position, a compile-time one (the rows before conv_k - 1). hidden: MTP's rows."""
    toks = None if isinstance(pos, RunRows) else OutTokens(1 if kind == "M" else 0)
    with RunWords(m, pos):
        if kind == "M":
            qwen35_mtp.fn(m, pos, R, block, tokens=toks)
        else:
            qwen35_rows.fn(m, pos, R, [R - 1] if kind == "L" else [], block, toks, False, hidden)


@ol.jit
def qwen35_layer_run(m, li: int, pos, row, block: int = ATTN_BLOCK, R: int = 1,
                     embedded: bool = False, hint: bool = False, em: bool = False):
    """Layer-major prefill (a MoE model's, docs/offload.md 13: the whole prompt chunk through a
    layer before the next, so that the expert cache serves one layer at a time): the R prompt
    rows from `row` (a run-time value) at the positions pos .. pos + R - 1 (a RunPos, from
    conv_k - 1, in one attention block; or compile-time positions, the rows before conv_k - 1)
    through layer li alone. Their input is the chunk's residual stream rows m.xbuf[row:row + R]
    (layer 0 unless `embedded`: one row's embedding row, _inputs at pos.tok, or R rows' from the
    host's slot rows 0 .. R - 1 (embed_host); else qwen35_embed_run's), their output goes back
    there. One row runs qwen35_step's layer; more rows _deltanet_rows / _attention_rows and
    moe.moe_ffn_rows, each row bit for bit as the step's, so the chunk layer by layer leaves the
    states, windows, KV cache and residual rows the per-position programs make. No router hint
    of its own layer (the run's request follows at once); with `hint`, when layer li + 1 is a
    MoE layer, the run ends with its hint: layer li + 1's router on the output rows
    (moe.moe_hint_rows; docs/offload.md 13.9), which changes no row.

    em: expert-major (docs/offload.md 13.11). The residual rows are the scratch records' X
    (moe.em_x), not m.xbuf. The run first ends layer li - 1's MoE for its rows
    (moe.moe_combine_rows: the outputs that layer's expert run stored), and a MoE layer's run
    ends with its prologue (moe.moe_prologue_rows: the route, the need line, the shared
    expert), its experts left to the layer's expert run (qwen35_expert_run). Each row's
    arithmetic is still moe_ffn_rows'."""
    spec, K = m.spec, m.spec.conv_k
    run = isinstance(pos, RunPos)
    p = pos.pos if run else pos
    X = MO.em_x(m.moe_dev, row, R, spec.hidden) if em else m.xbuf[row:row + R, :]
    if R == 1 and li == 0 and not embedded:
        x, c, s_ = _inputs(m, pos)
    elif li == 0 and not embedded:  # the host's slot rows (its embed_host table), as _inputs_rows
        x = ol.empty([R, spec.hidden], dense=True)
        c, s_ = ol.load(m.cos_t[p:p + R, :]), ol.load(m.sin_t[p:p + R, :])
        for r, g in enumerate(_gather(m, m.embed_q, range(R))):
            x[r:r + 1, :].set(g)
    else:
        x = ol.load(X)
        if R == 1:
            c, s_ = ol.load(m.cos_t[p, :]), ol.load(m.sin_t[p, :])
        else:
            c, s_ = ol.load(m.cos_t[p:p + R, :]), ol.load(m.sin_t[p:p + R, :])
    if em and li > 0 and m.layer(li - 1).moe:
        x.set(MO.moe_combine_rows(x, spec.moe, m.moe_dev, row))
    lw = m.layer(li)
    if lw.kind == LIN:
        if R == 1:
            dn = _deltanet_dstep if ol.has_dstep() else _deltanet
            x.set(dn(x, lw, pos, spec, m.hs))
        else:                       # (from conv_k - 1 every position reads the whole window:
            x.set(_deltanet_rows(x, lw, K - 1 if run else pos, spec,    # one program for all)
                                 m.gr[0:R, :], m.on[0:R, :]))
    elif R == 1:
        x.set(_attention(x, lw, c, s_, pos, spec, block, gated=True))
    else:
        rows = [(0, pos.offset(r) if run else pos + r) for r in range(R)]
        x.set(_attention_rows(x, lw, c, s_, rows, spec, block, gated=True))
    if not lw.moe:
        x.set(_mlp(x, lw, spec))
    elif em:
        ol.store(X, x)
        MO.moe_prologue_rows(x, lw, spec.moe, m.moe_dev, spec.eps, row)
        return
    elif R == 1:
        x.set(MO.moe_ffn(x, lw, spec.moe, m.moe_dev, spec.eps))
    else:
        x.set(MO.moe_ffn_rows(x, lw, spec.moe, m.moe_dev, spec.eps))
    ol.store(X, x)
    if hint and li + 1 < spec.layers and m.layer(li + 1).moe:
        MO.moe_hint_rows(x, m.layer(li + 1), spec.moe, m.moe_dev, spec.eps)


@ol.jit
def qwen35_embed_run(m, pos, row, em: bool = False):
    """A layer-major prefill's input row (runs of more than one row, the rows before conv_k -
    1): the token's embedding row (pos.tok, from the image's tables) -> m.xbuf[row] (em: its
    record's X), as qwen35_step's."""
    X = MO.em_x(m.moe_dev, row, 1, m.spec.hidden) if em else m.xbuf[row:row + 1, :]
    ol.store(X, _embed(m, pos.tok))


@ol.jit
def qwen35_prefill_head(m, row, em: bool = False):
    """After a layer-major prefill's last layer: the final norm and the LM head of the residual
    stream row m.xbuf[row] (a run-time value) -> m.logits, as qwen35_step's. em: the row's
    record's X, after the last layer's combine (expert-major: its MoE ends here)."""
    spec = m.spec
    if not em:
        _lm_head(ol.load(m.xbuf[row:row + 1, :]), m, spec)
        return
    x = ol.load(MO.em_x(m.moe_dev, row, 1, spec.hidden))
    if m.layer(spec.layers - 1).moe:
        x.set(MO.moe_combine_rows(x, spec.moe, m.moe_dev, row))
    _lm_head(x, m, spec)


@ol.jit
def qwen35_expert_run(m, li: int):
    """Expert-major's expert run of MoE layer li (moe.moe_expert_run, docs/offload.md 13.11):
    after the layer's runs over a prefill chunk, each expert its rows chose once, in passes of
    two rows; the chunk's rows and their entries (rows x k) are run arguments."""
    from ..compiler import RunVar
    C, k = m.moe_dev.em_rows, m.spec.moe.k
    MO.moe_expert_run(m.layer(li), m.spec.moe, m.moe_dev, RunVar("rows", C + 1),
                      RunVar("entries", C * k + 1))
