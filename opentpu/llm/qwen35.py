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
    are group-major (Image, DeltaNetParts), so it loops the pairs too and fits IMEM; on one
    slice with the streams on an engine of their own (Config.SE_OWN), from PAIR_LOOP_OWN, and
    from PAIR_LOOP_QK where the slice's key heads' q and k are projected once (the 27B).
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
(Image(batch=N): N sequences' slots, their steps batched by opentpu/llm/batch35.py).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .. import fp32 as F
from .. import isa as I
from .. import qcache as QC
from .. import quant as Q
from .. import language as ol
from ..compiler import (Affine, CompileError, KVDesc, QTensor, Tensor, current, k_scale_bytes,
                        gall_wg, trace_fit)
from ..host.offload import LINE, ExpertServer, Layout
from ..isasim import Config
from ..kernels.attention import SPAN_P
from ..kernels.layouts import head_parallel_attention_weights
from ..kernels import dn_chunk as DC
from ..kernels.deltanet import gate_logs, gates, l2norm_rows
from ..kernels.lib import rmsnorm, silu, silu_l2
from ..kernels.mlp import _chunk, down_chunk, f_chunk
from .lfm2 import _place, plan, run_layers, subs
from . import formats as FM
from . import generate as G
from . import moe as MO
from . import rope_parameters
from .qwen3 import (ATTN_BLOCK, HEAD_CHUNK, OutTokens, RunPos, RunRows, RunWords,
                    _align, _amask, _attention, _attention_rows, _Bump,
                    _embed, _fake_k, _fake_q, _fake_w, _formats, _gather, _inputs, _inputs_rows,
                    _qk, gathered_rows,
                    _lm_head, _lm_head_rows, _lookup_alloc, _lookup_build, _lookup_desc, _mlp,
                    _pv, _qdesc, _tdesc, _tok_arg, _tokens_arg, _v_parts, compile_decode,
                    loop_ahead, put_head, rope_tables, EmbedHost, fill_logits, step_descriptors,
                    fit_high, high_base, low_check, mixer_base)

LIN, ATTN = "linear", "attn"
DN_C = -1.0 / math.log(2.0)     # Image.dn_fold's constant, -log2(e): the exp2 inputs' scale
PAIR_LOOP = 8       # pairs of DeltaNet heads per slice that decode unrolled at a run-time position
PAIR_LOOP_OWN = 16  # the same on one slice with the streams on an engine of their own (Config.SE_OWN):
                    # unrolled, every pair's taps and window load ahead of the DSTEPs (cv_ahead), off
                    # their chain (bcu1525-wg's Qwen3.5-4B: -4.1% a DeltaNet layer at SE_R 1, -3.4% at
                    # SE_R 2; its resident decode 3,097 instructions at most, docs/bcu1525_wg.md)
PAIR_LOOP_QK = 24   # and there where the slice's key heads' q and k are projected once (qk_once: a
                    # pair's rows only v and z): the 27B's 24 (48 value heads, 16 key heads), whose
                    # decode, generate, tile runs of 4 rows and prompt runs of 2 fit IMEM unrolled
                    # (at most 4,045 instructions); group-major would turn qk_once off
MTP_VOCAB = 32768   # the MTP's draft head: the fp4 rows of this many lowest ids (byte-level BPE
                    # ids follow merge frequency: they cover 94% of the generated tokens;
                    # docs/mtp.md 7.1)
EMBED_F32_MAX = 2 << 30     # bytes: a larger fp32 embedding table (over half the card's DRAM,
#                             Qwen3.5-4B and up) is int8, gathered from the head (Spec.embed)
PREFILL_CHUNK = 512 # a MoE model's layer-major prefill: prompt rows a chunk (Image's xbuf)
RUN_ROWS = 4        # and at most rows a run (_deltanet_rows' gr / on; moe_ffn_rows: R k ids)
NEST = True         # prefill programs loop a unit's DeltaNet layers inside the unit's loop (subs)


def s_exact_on(cfg=None) -> bool:
    """S-exact images at S > 1 (s_exact, mlp_s_exact): OTPU_SEXACT=1 or 0 when set, else cfg's
    SEXACT (its target's default: bcu1525-s2's on; the others' off, S > 1's own layout, which
    the WG targets images' card references follow); cfg None: OTPU_SEXACT alone."""
    e = os.environ.get("OTPU_SEXACT", "")
    if e not in ("", "0", "1"):
        raise ValueError(f"OTPU_SEXACT={e!r}: 0 or 1 (unset: the target's)")
    return e == "1" if e else bool(getattr(cfg, "SEXACT", False))


def s_exact(spec, S: int, cfg=None) -> bool:
    """DeltaNet's out_proj at S > 1 in S = 1's QACT ranges and adds (s_exact_on): S = 1
    quantizes its input per group of 4 heads, the last group in pairs, and adds the groups'
    MMs in head order. Slice s takes S = 1's groups s, s + S, ... (Image.build's heads), so
    the gathers deliver them in S = 1's order, and the slices' last groups come in the last
    group's two pair gathers: slice S - 1's (S = 1's last) a pair at a time, the others'
    whole (_last_pairs). Needs whole groups a slice (4 S | heads) and each pair's own q and k
    rows (not qkonce: its key heads a slice's)."""
    nh, rk = spec.lin_heads, spec.lin_heads // spec.lin_nk
    return (S > 1 and s_exact_on(cfg) and nh % (4 * S) == 0 and not (rk > 1 and rk % 2)
            and spec.qk_once is not True)


def mlp_s_exact(spec, S: int, D: int, fmt: str, cfg=None) -> bool:
    """A dense MLP at S > 1 in S = 1's QACT ranges and adds (s_exact_on), its W_down in
    format fmt: slice s holds part s of each of S = 1's F chunks (C1 = down_chunk(F): rows
    j C1 + s C1 / S .. of W_gate and W_up, j = 0, 1, ...) and W_down's columns in S = 1's
    order, so a step's all-gathered activations are S = 1's chunk j in order: its down MM one
    QACT over S = 1's columns, the MMs in S = 1's order. The image's placement only:
    swiglu_down's gather-order path at a slice's chunk C1 / S (Qwen3.5-27B: 512, the chunk
    of F / S already, so the same program). Needs C1 in S parts of whole D blocks."""
    return (S > 1 and spec.moe is None and s_exact_on(cfg)
            and down_chunk(spec.ffn, D, fmt) % (S * D) == 0)


def pair_loop(cfg, qk: bool = False) -> int:
    """The most pairs of DeltaNet heads a slice of `cfg` decodes unrolled (array-major blocks):
    PAIR_LOOP, or on one slice with the streams on an engine of their own PAIR_LOOP_OWN, and
    PAIR_LOOP_QK for a slice whose key heads' q and k are projected once (qk)."""
    if cfg.SE_OWN and cfg.S == 1:
        return PAIR_LOOP_QK if qk else PAIR_LOOP_OWN
    return PAIR_LOOP


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
                                    # more than pair_loop(cfg) pairs of heads)
    qk_share: bool | None = None    # a pair of value heads of one key head projects and convolves
                                    # its q and k once (Image; None: when the value heads per key
                                    # head are even)
    qk_once: bool | None = None     # a slice's key heads' q and k projected, convolved and normed
                                    # once a layer, its pairs' rows only v and z (Image; None:
                                    # when the value heads per key head are odd and above one, a
                                    # slice holds whole key heads and its pairs are unrolled)
    moe: MO.MoESpec | None = None   # Qwen3.5-MoE: every layer's MLP is routed experts plus a
                                    # shared expert (ffn: its width, the layer block's MLP)
    formats: str = ""       # weight formats per kind over the image's wformat (KINDS)
    mix: str = ""           # the recommended mix (wformat "mix": formats.named, MIXES)
    mtp: bool = False       # the image holds the MTP drafter (the checkpoint's mtp.* layer, a
                            # draft head) and two slots of every DeltaNet state and convolution
                            # window for the speculative verify (opentpu/llm/mtp.py, docs/mtp.md 9)
    cv_ahead: bool | None = None    # a DSTEP decode loads every pair's taps and window and keeps
                                    # the gates in TMEM before its steps, stores the windows
                                    # after them, and (qk_once) projects and norms key head 0
                                    # first (Image; None: with port W, WG > 1)
    cv_pre: bool | None = None      # a DSTEP decode at a run-time position loads the next
                                    # DeltaNet layer's taps and windows beside this layer's MLP
                                    # and stores this layer's windows there (qwen35_step's
                                    # _CvPipe), off the mixer's DMA, whose chain is its DSTEPs
                                    # (Image; None: with cv_ahead, unless qk_once)
    dn_fold: bool = False   # the DeltaNet's scale constants folded into its weights (Image
                            # .dn_fold): SiLU's and beta's -log2(e) into the taps and z's and b's
                            # rows, -log2(e) e^A_log in A_log's slot, the output norm's gain over
                            # log2(e), q's dk^-1/2 into its eps (_silu_dn, _l2n, _dn_eps); the
                            # state then holds log2(e) S. Image.build folds the checkpoint's
                            # weights (z's and b's by their rows' Sw: the codes unfolded)
    state16: bool | None = None     # the DeltaNet states in 16 bits (Config.ST16's format, e6m9
                                    # or bf16; DSTEP's and STREAM's flag ST16, fp32 in the step):
                                    # half their bytes each way (Image.st16; None: on a
                                    # Config.ST16 machine; False: fp32 states there too)

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
        eos = c.get("eos_token_id")             # (a null one: the default's)
        eos = tuple(e for e in (eos if isinstance(eos, list) else [eos]) if e is not None) \
            or (248044,)
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
        # the widest operand the MLP quantizes: the down MMs take a chunk of every slice's F
        # part, all-gathered on S > 1 slices (kernels.mlp's swiglu_down: S chunks), not F
        # (Qwen3.5-27B: 1024 of 17408 on one slice, 4 x 256 at S 4); a tile's chunks that do
        # not all fit ACT RAM go in groups (tiled.act_groups)
        f_loc = self.ffn // S
        f_in = S * max((_chunk(f_loc, D, q) for q in (D, 2 * D) if f_loc % q == 0),
                       default=self.ffn)
        need = [(self.head_dim % D == 0, f"head_dim {self.head_dim} % D"),
                (self.rope_dim % 2 == 0 and self.rope_dim <= self.head_dim, "rope_dim"),
                (self.lin_dk % D == 0 and self.lin_dv % D == 0,
                 f"DeltaNet heads ({self.lin_dk} x {self.lin_dv}) a multiple of D {D} wide"),
                (cfg.DSTEP or 2 * self.lin_dk * self.lin_dv < cfg.TMEM_WORDS,
                 f"DeltaNet on the VPU: a head pair's states ({self.lin_dk} x {self.lin_dv} "
                 f"words each) in TMEM ({cfg.TMEM_WORDS} words; DSTEP keeps them in DRAM)"),
                (self.lin_heads % (2 * S) == 0, f"DeltaNet heads % 2S (pairs per slice)"),
                (self.lin_heads % self.lin_nk == 0, "DeltaNet value heads % key heads"),
                (self.hidden % (S * D) == 0, f"hidden {self.hidden} % S*D"),
                (self.ffn % (S * D) == 0, f"ffn {self.ffn} % S*D"),
                # (fewer KV heads than slices: each on S / n_kv of them, KVDesc.kv_heads)
                (self.n_kv % S == 0 or S % self.n_kv == 0, f"n_kv {self.n_kv} % S"),
                (self.n_q % self.n_kv == 0 and self.n_q % S == 0, "n_q % n_kv, n_q % S"),
                (self.vocab % S == 0, f"vocab {self.vocab} % S"),
                (max(f_in, self.n_q * self.head_dim, self.hidden) <= cfg.ACT_BLOCKS * D,
                 "an inner dimension exceeds ACT RAM")]
        if self.mtp:
            need += [(S == 1, "MTP decoding runs on one slice")]
        if self.moe is not None:
            mo = self.moe
            need += [(mo.ffn % (S * D) == 0, f"expert width {mo.ffn} % S*D"),
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
              prefill_rows: int | None = None, high: bool | None = None,
              em: bool = False) -> "Image":
        return fit_high(lambda h: Image(self, cfg, cap, batch, rows, wformat, head_format,
                                        lookup, experts, embed_host, formats, prefill_rows, h,
                                        em=em), cfg, high)


# =============================================================================== reference
def _norm(v, g, eps):
    return (v / np.sqrt(np.mean(v * v, axis=-1, keepdims=True) + eps)) * g


def _l2norm(v, eps=1e-6):
    return v / np.sqrt(np.sum(v * v, axis=-1, keepdims=True) + eps)


def _silu(v):
    return v / (1 + np.exp(-v))


def _softplus(v):
    return np.maximum(v, 0) + np.log1p(np.exp(-np.abs(v)))


def _fold_sw(sc, rows):
    """Quantized rows' scale rows with the rows `rows`' Sw (a scale row's first word, fp32)
    times DN_C, rounded once to fp32 (Image.dn_fold: z's and b's rows; their codes are the
    unfolded weight's). rows None: sc as it is."""
    if rows is None:
        return sc
    sc = np.array(sc, np.uint8)
    sw = np.ascontiguousarray(sc[:, :4]).view(np.float32)[:, 0].astype(np.float64)
    sw[rows] *= DN_C
    sc[:, :4] = sw.astype(np.float32).reshape(-1, 1).view(np.uint8)
    return sc


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
    fused tensors; a LazyWeights reads the one expert; the MTP layer's, mtp.layers.0., are
    tensors of their own per expert)."""
    m = p + "mlp.experts."
    one = f"{m}{e}."
    if one + "gate_proj.weight" in W:
        return W[one + "gate_proj.weight"], W[one + "up_proj.weight"], W[one + "down_proj.weight"]
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
                    formats: str | None = None, S: int = 1, cfg=None) -> np.ndarray:
    """float64 decode with openTPU's quantization points and none of its rounding (as
    qwen3.emulated_logits, on S slices): the weights in their MXU formats, every MM input
    quantized as QACT does (DeltaNet's out_proj input per head group of og heads of a slice,
    its last group in pairs at og = 4: _deltanet's MMs; the down projections' in their F
    chunks), K and V as the cache holds them, P per attention block with V's scales folded in.
    The DeltaNet state, convolution and gates are exact (they are fp32 on the device).
    Qwen3.5-MoE: the router (with the shared expert's gate as its last row) in int8; `routes`
    gets (token, layer, ids, the k-th and (k+1)-th logits' gap) per MoE layer. At S > 1 the
    MoE layers are S = 1's (moe._moe_ffn_tp: the shared expert's down in S = 1's chunks).
    cfg: the image's Config, whose SEXACT (s_exact_on) the emulation follows at S > 1."""
    d, G, eps, K = spec.head_dim, spec.n_q // spec.n_kv, spec.eps, spec.conv_k
    nh, nk, dk, dv = spec.lin_heads, spec.lin_nk, spec.lin_dk, spec.lin_dv
    slices = S                                          # (S: the DeltaNet state below)
    nl = nh // (1 if s_exact(spec, S, cfg) else slices)     # a slice's heads, Image.og (S-exact:
    #                                                     S = 1's groups)
    og = 4 if nl % 4 == 0 else 2
    outr = [og * dv] * (nl // og)                       # out_proj's QACT ranges, a slice's
    if og == 4 and nl > 2:
        outr[-1:] = [2 * dv, 2 * dv]
    outr = outr * (nh // nl)
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

    def chunk(i, sl=slices):
        return down_chunk(spec.ffn // sl, D, fmt("down", i))

    lin = [i for i, k in enumerate(spec.kinds) if k == LIN]
    Kc = {i: [] for i, k in enumerate(spec.kinds) if k == ATTN}
    Vc = {i: [] for i in Kc}
    ring = {i: [np.zeros(2 * nk * dk + nh * dv)] * (K - 1) for i in lin}
    state = {i: np.zeros((nh, dk, dv)) for i in lin}
    out = []
    for pos, tk in enumerate(tokens):
        x = np.asarray(W["model.embed_tokens.weight"][tk], np.float64)
        if spec.embed == "int8":                    # the int8 embedding rows (Spec.embed)
            x = gathered_rows(x, D)[0].astype(np.float64)
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
                x = x + w(a + "out_proj.weight") @ _fake_q(o.reshape(-1), D, outr)
            else:
                a = p + "self_attn."
                qg = (w(a + "q_proj.weight") @ h).reshape(spec.n_q, 2 * d)
                q, gate = qg[:, :d], qg[:, d:]
                k = (w(a + "k_proj.weight") @ h).reshape(spec.n_kv, d)
                v = (w(a + "v_proj.weight") @ h).reshape(spec.n_kv, d)
                q = _rot(_norm(q, g1(a + "q_norm.weight"), eps), c, s, spec.rope_dim)
                k = _rot(_norm(k, g1(a + "k_norm.weight"), eps), c, s, spec.rope_dim)
                Kc[i].append(_fake_k(k, D))
                Vc[i].append(_v_parts(v))
                Kh = np.stack([a_ for a_, _ in Kc[i]], 1)
                dK = None if Kc[i][0][1] is None else np.stack([b_ for _, b_ in Kc[i]], 1)
                Vq, Vs = (np.stack(z, 1) for z in zip(*Vc[i]))
                o = np.zeros((spec.n_q, d))
                for hq in range(spec.n_q):
                    sc = _qk(_fake_q(q[hq] / math.sqrt(d), D), Kh[hq // G],
                             None if dK is None else dK[hq // G], D)
                    pp = np.exp(sc - sc.max())
                    o[hq] = _pv(pp, Vq[hq // G], Vs[hq // G], D) / pp.sum()
                o = o / (1 + np.exp(-gate))
                x = x + w(a + "o_proj.weight") @ _fake_q(o.reshape(-1), D)
            h = _fake_q(_norm(x, g1(p + "post_attention_layernorm.weight"), eps), D)
            mp = spec.mlp_prefix(p)
            gg = w(mp + "gate_proj.weight") @ h
            u = w(mp + "up_proj.weight") @ h
            sl = 1 if spec.moe is not None or mlp_s_exact(spec, slices, D, fmt("down", i), cfg) \
                else slices                     # (MoE: S = 1's, _moe_ffn_tp; mlp_s_exact)
            y = w(mp + "down_proj.weight") @ Q.fake_act_gathered(_silu(gg) * u, D, chunk(i, sl),
                                                                 sl)
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
                    ce = MO.ExpertFormat(spec.hidden, mo.ffn, D, wformat).ranges()
                    x = x + we * (wd @ _fake_q(_silu(wg @ h) * (wu @ h), D, ce))
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
    0). With Image.high the pairs' taps, windows and states are in blocks of their own below
    4 GiB (strides lgs, lps): the loop steps two.

    shared (Image.shared): the pair's two value heads have one key head; its rows are the key
    head's q and k once, then v of a, of b, then z of a, of b, and its taps rows (block, tap) of
    four blocks of dk (q, k, v a, v b) instead of two of 2 dk + dv (the heads).

    Slots (Spec.mtp): the states and windows come in two slots, `slot` the one holding the
    committed ones (state, window), the other at sbytes bytes / wwords words after slot 0's
    (the speculative verify's rejectable row writes there: docs/mtp.md 9). tp: the taps' words
    before the window in cv.

    qkonce (Image.qkonce): the slice's key heads' q and k once a layer, wqk ([QC, H], q | k
    of each key head) and qkcv (their taps, rows (key head, tap), qtp words, then their
    window); a pair's
    rows are v of a, of b, then z of a, of b, its taps rows (block, tap) of v of a, of b. Head
    j of pair p takes key head kh(p, j)."""

    def __init__(self, nl: int, og: int, wh: QTensor, cv: Tensor, state: Tensor,
                 wout: QTensor, H: int, gs: int = 0, ps: int = 0, eb: Tensor | None = None,
                 shared: bool = False, tp: int = 0, wwords: int = 0, slot: int = 0,
                 sbytes: int = 0, wqk: QTensor | None = None, qkcv: Tensor | None = None,
                 rk: int = 1, qtp: int = 0, ahead: bool = False, fold: bool = False,
                 lgs: int = 0, lps: int = 0, eb_low: bool = False, sexact: bool = False):
        self.nl, self.og, self.gp, self.H, self.shared = nl, og, og // 2, H, shared
        self.sexact = sexact            # Image.sexact (s_exact: _out_proj's last pairs)
        self.ahead = ahead              # Image.cv_ahead
        self.fold = fold                # Image.dn_fold
        self.grouped, self.gs, self.ps = gs > 0, gs, ps
        self.lgs, self.lps = lgs or gs, lps or ps       # (cv's and state's: Image.high's)
        self.eb_low = eb_low            # the gates in the states' group blocks (Image.mix)
        self._wh, self._cv, self._state, self._wout, self._eb = wh, cv, state, wout, eb
        self.R2 = wh.shape[0] if self.grouped else wh.shape[0] // (nl // 2)
        self.tp, self.wwords, self.slot, self.sbytes = tp, wwords, slot, sbytes
        if slot and not sbytes:
            raise CompileError("state slot 1 needs an image with slots (Spec.mtp)")
        self.qkonce, self.wqk, self._qkcv, self.rk, self.qtp = wqk is not None, wqk, qkcv, rk, qtp
        self.QC = 0 if wqk is None else wqk.shape[0]
        self.nkl = nl // rk if wqk is not None else 0

    def kh(self, p, j: int) -> int:
        """The slice's key head of head j of pair p (qkonce; p an int: the pairs unrolled)."""
        q = self.index(p)
        if not isinstance(q, int):
            raise CompileError("q and k once a layer: a pair's key head needs the pairs unrolled")
        return (2 * q + j) // self.rk

    def qk_cv(self) -> Tensor:
        """The key heads' taps and window (qkonce; one load)."""
        return self._qkcv

    def qk_window(self) -> Tensor:
        """The key heads' convolution window (K - 1 rows of their QC channels, oldest first,
        after their taps)."""
        c = self._qkcv
        return Tensor(c.base + 4 * self.qtp, (c.shape[0] - self.qtp,), (1,))

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

    def _off(self, p, low: bool = False) -> Affine:
        g, i = self.split(p)
        if low:
            return Affine.of(g) * self.lgs + Affine.of(i) * self.lps
        return Affine.of(g) * self.gs + Affine.of(i) * self.ps

    def wh(self, p) -> QTensor:
        if not self.grouped:
            q = self.index(p)
            return self._wh[q * self.R2:q * self.R2 + self.R2, :]
        w, o = self._wh, self._off(p)
        return replace(w, data=w.data + o, scale=w.scale + o)

    def cv(self, p) -> Tensor:
        if not self.grouped:
            return self._cv[self.index(p)]
        c = self._cv
        return Tensor(c.base + self._off(p, True), c.shape, c.strides)

    def cvs(self, q0: int, q1: int) -> Tensor | None:
        """Pairs q0 .. q1 - 1's taps and windows as one block (array-major without slots: one
        pair's right after the other's), or None."""
        c = self._cv
        if self.grouped or c.strides[0] != c.shape[1]:
            return None
        return Tensor(c[q0].base, ((q1 - q0) * c.shape[1],), (1,))

    def state(self, p, a: int, slot: int | None = None) -> Tensor:
        """The state of head a (0, 1) of pair p: in the committed slot, or in `slot`."""
        slot = self.slot if slot is None else slot
        if slot and not self.sbytes:
            raise CompileError("state slot 1 needs an image with slots (Spec.mtp)")
        if not self.grouped:
            t = self._state[2 * self.index(p) + a]
        else:
            s = self._state
            t = Tensor(s.base + self._off(p, True), s.shape, s.strides, esize=s.esize)[a]
        return Tensor(t.base + slot * self.sbytes, t.shape, t.strides, esize=t.esize) \
            if slot else t

    def states(self, p, slot: int | None = None) -> Tensor:
        """Pair p's two states as one [2 dv, dk] tensor, head a's rows then head b's (adjacent
        in both layouts): a DSTEP pair's (Config.DSTEP2)."""
        t = self.state(p, 0, slot)
        return Tensor(t.base, (2 * t.shape[0], t.shape[1]), (t.shape[1], 1), esize=t.esize)

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
        return replace(w, data=w.data + o, scale=w.scale + o)

    def _gates(self, g) -> Tensor:
        e = self._eb
        return Tensor(e.base + Affine.of(g) * (self.lgs if self.eb_low else self.gs), e.shape,
                      e.strides)

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


# a layer block's parts in its states' block (Image.high): what a decode writes, below HIGH
LOW_PARTS = ("cv", "state", "qkcv", "lgroups", "kv")
# and those in its mixers' block (Image.mix): the DeltaNet mixers' weights
MIXER_PARTS = ("wh", "wab", "wout", "wqk", "groups")


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

    high (above AW 32: qwen3.fit_high, by fit unless OTPU_HIGH says): the weights from HIGH (4
    GiB) on and below it what R0 + w must reach, the I/O area, the KV caches, the states and
    windows (the taps beside them), the lookup tables and the run-time areas past low_end. A layer
    block is then two: its weights' (loc: a run's units from HIGH on) and its states' (lloc:
    from the I/O area on, every run's units one after the other as the weights'), with the
    states' offsets (cv, state, qkcv, kv; group-major the pairs' cv and state in lgroups, LGS
    and LPS a group's and a pair's) from the second; a MoE with every expert in DRAM keeps them
    as a resident pool (pool: global expert g at pool.base + g pool.ebytes, no directory and no
    host server: moe_ffn's resident path), the head and the MTP's weights after the layers'.
    OTPU_POOL=1 keeps such a pool below HIGH too (a cut's runs on the board model, whose DIMM
    model holds no image above 4 GiB, take the full model's resident path that way).
    nbytes is the image's extent (its arrays' length), low_end the end of what is below HIGH.

    mix (the memory's bank-half window, Config.BHW, with high): the low area in the window's lower
    half (qwen3.low_top: at BHW 33 below 4 GiB, as high's), a DeltaNet layer's mixers' weights
    (MIXER_PARTS: its projections, out_proj, group-major its pairs' and groups' blocks) in a
    block of their own in the window's upper half (mloc, msize: every run's units one after the
    other from qwen3.mixer_base, as the weights'), every other weight past the window
    (qwen3.high_base), the pool among them; group-major, the pairs' gates (written every token)
    in the states' group block after its pairs' (leb). A decode's writes and the mixers' stream
    beside them then take different banks. The mixer matrices' scale rows (not those inside
    group-major's pairs' and groups' blocks) go in the layer's states' block (the layouts' slow:
    their names), in the lower half's banks: a mixer's data stream and its scale stream, read
    side by side, never take turns on one bank (in the upper half they shared its two bank
    indexes; 31-45% of the 27B's and 35B's row conflicts). wlo / wskip: where the weights start
    and the window's bytes past the mixers' (qwen3.Images high / skip). A resident pool that does not
    fit past the window with the other weights (the 35B at S = 2: 8.07 + 0.35 GiB of a slice's
    [8, 16)) spills into it, still one block (pool.base + g ebytes): at the highest place that
    fits (_spill: the layout again, the pool there and every other weight from its end on, to the
    slice's top), so the fewest experts stream from the window's half of the banks; wskip is
    then the bytes between the mixers' end and the pool's start. The pool is read-only: what a
    decode writes stays below 4 GiB.
    """

    def __init__(self, spec: Spec, cfg: Config, cap: int, batch: int = 1, rows: int = 1,
                 wformat: str = "int8", head_format: str | None = None, lookup: bool | str = False,
                 experts: int | None = None, embed_host: bool | None = None,
                 formats: str | None = None, prefill_rows: int | None = None,
                 high: bool = False, em: bool = False, _spill: tuple[int, int] | None = None):
        args = {k: v for k, v in locals().items() if k not in ("self", "_spill", "__class__")}
        spec.check(cfg)
        self.high = bool(high)
        if batch < 1 or batch > 1 and spec.mtp:
            raise ValueError(f"batch={batch}: Qwen3.5 batches sequences without MTP (Spec.mtp: "
                             "batch=1)")
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
        # every matrix in groups of WG rows (docs/isa.md "Weight groups"; WG 1: rows)
        mb = lambda n, k, f: Q.grp_bytes(n, k, f, D, cfg.WG)            # noqa: E731
        dk, dv = spec.lin_dk, spec.lin_dv
        self.spec, self.cfg, self.cap, self.batch, self.rows = spec, cfg, cap, batch, rows
        self.nq_loc, self.nkv_loc = spec.n_q // S, max(spec.n_kv // S, 1)     # (KVDesc.kv_heads)
        self.h_loc, self.f_loc, self.v_loc = H // S, F_ // S, spec.vocab // S
        self.nl = spec.lin_heads // S                   # DeltaNet heads of one slice
        self.C = 2 * dk + dv                            # convolved channels per head (q, k, v)
        self.R = self.C + dv                            # projected rows per head (and z)
        rk = self.rk = spec.lin_heads // spec.lin_nk
        self.shared = rk % 2 == 0 if spec.qk_share is None else spec.qk_share
        if self.shared and rk % 2:
            raise ValueError(f"{rk} value heads per key head: a pair shares its q and k only "
                             "when they are even")
        nl = self.nl
        # q and k once a layer, for the slice's key heads (rk odd: a pair can straddle two)
        qk = rk > 1 and rk % 2 == 1 and nl % rk == 0 and not spec.mtp and spec.qk_once is not False
        self.grouped = nl // 2 > pair_loop(cfg, qk) if spec.pair_loop is None else spec.pair_loop
        self.qkonce = (qk and not self.grouped) if spec.qk_once is None else spec.qk_once
        if self.qkonce and (self.shared or self.grouped or nl % rk or spec.mtp):
            raise ValueError("q and k once a layer: a slice of whole key heads (its value heads "
                             f"{nl} % {rk} per key head), pairs unrolled (not group-major: a "
                             "loop's pair would name its key head by a non-affine index), "
                             "neither shared q and k nor MTP slots")
        # every pair's taps, window and gates in TMEM before a DSTEP decode's steps, the
        # windows stored after them (_deltanet_dstep): off the DMA's chain of steps, the
        # mixer's critical path when the weights take port W (WG > 1); not past PAIR_LOOP_OWN
        # pairs with the key heads' q and k too (the 27B's 24 on bcu1525-wg: TMEM exhausted)
        self.cv_ahead = (cfg.WG > 1 and not (self.qkonce and nl // 2 > PAIR_LOOP_OWN)) \
            if spec.cv_ahead is None else spec.cv_ahead
        # and a layer ahead, the windows stored beside the MLP (qwen35_step's _CvPipe)
        self.cv_pre = self.cv_ahead and not self.qkonce if spec.cv_pre is None else spec.cv_pre
        self.dn_fold = spec.dn_fold     # (Spec.dn_fold: the kernels' and the weights' constants)
        # the DeltaNet states in 16 bits (Spec.state16): only DSTEP and STREAM touch them
        # (flag ST16), so the machine's DSTEP and ST16; se a state element's bytes
        self.st16 = bool(cfg.ST16 and cfg.DSTEP) if spec.state16 is None else spec.state16
        if self.st16 and not (cfg.ST16 and cfg.DSTEP):
            raise ValueError("16-bit DeltaNet states (Spec.state16) need Config.ST16 and DSTEP")
        self.se = 2 if self.st16 else 4
        self.nkl = nl // rk if self.qkonce else 0      # the slice's key heads (qkonce)
        self.QC = 2 * dk * self.nkl                     # their q, k channels (q | k a key head)
        self.QCV0 = K * self.QC + (K - 1) * self.QC     # their taps, then their window
        # a pair's convolved channels, as blocks of equal width: q k v of a, of b; shared, the
        # key head's q, k, then v of a, of b (dk = dv); qkonce, v of a, of b
        self.nb, self.bw = (4, dk) if self.shared else (2, dv) if self.qkonce else (2, self.C)
        self.CP = self.nb * self.bw
        self.RP = self.CP + 2 * dv                      # a pair's projected rows (then z of a, b)
        self.CV0 = K * self.CP + (K - 1) * self.CP      # a pair's taps, then its window
        # DeltaNet state and window slots: MTP's two (the committed and the verify's), else
        # one a sequence (batch: opentpu/llm/batch35.py)
        self.slots = 2 if spec.mtp else batch
        self.slot_runs = batch > 1      # (a slot's descriptors address its caches and states:
        #                                 its prompt by layer runs at it, batch35.prefill_slot)
        self.CVW = self.CV0 + (self.slots - 1) * (K - 1) * self.CP     # (then window 1)
        b = _Bump(align=_align(cfg))
        hb = _Bump(high_base(cfg) if _spill is None else _spill[1], _align(cfg)) if self.high \
            else b                                      # the weights' (high: from HIGH)
        self.mix = bool(cfg.BHW) and self.high          # (the mixers' own block: mloc)
        # the layer-major prefill's rows a chunk (a MoE model's: compile_layer_run), the rows of
        # a run from the host's slot (layer 0's, the MTP layer's K / V), and of _deltanet_rows'
        # DRAM scratch (gr, on): a tiled MoE image's, its tile's (its DeltaNet runs of MCOLS rows)
        self.prefill_rows = (PREFILL_CHUNK if spec.moe is not None else 0) \
            if prefill_rows is None else prefill_rows
        self.grows = self.dnrows = max(rows, RUN_ROWS) if self.prefill_rows else rows
        from .prefill import layer_tile_flag
        if self.prefill_rows and spec.moe is not None and layer_tile_flag(None, True):
            from .tiled import LAYER_ROWS
            self.dnrows = max(self.grows, min(LAYER_ROWS["qwen35"], cfg.act_rows))
        self.io = {"x": b.alloc(4 * H * rows), "cos": b.alloc(2 * spec.rope_dim * rows),
                   "sin": b.alloc(2 * spec.rope_dim * rows), "gf": b.alloc(4 * H),
                   "logits": b.alloc(4 * spec.vocab * rows),
                   "hs": b.alloc(4 * 2 * self.nl),     # per pair: decays of a, b; betas
                   "gr": b.alloc(4 * 2 * self.nl * self.dnrows),   # chunked prefill, per row:
                   "on": b.alloc(4 * 4 * spec.lin_dv * self.dnrows)}   # decays, betas; a head
        from .prefill import tile_scratch   # a tiled run's attention outputs, a part at a time
        if tile_scratch(spec, layer_runs=True, default=True):   # (tiled.py; a MoE image's: on
            #                                   unless OTPU_TILE=0, tiled.LAYER_ROWS)
            self.io["o"] = b.alloc(4 * cfg.act_rows * self.nq_loc * spec.head_dim)
        if spec.mtp:                # the final norm's rows (the MTP's input), the draft ids
            self.io.update(hid=b.alloc(4 * H * rows), draft=b.alloc(4 * rows))   # group's outputs
        if self.prefill_rows:       # its residual stream, and moe_ffn_rows' outputs (a
            self.io["xbuf"] = b.alloc(4 * H * self.prefill_rows)     # request's ids and a sink)
            if spec.moe is not None:
                self.io["moe_scratch"] = b.alloc(4 * H * (LINE // 4 + 1))
        self.layer0 = hb.next
        mlp = {"wg": (self.f_loc, H), "wu": (self.f_loc, H)}
        mo = spec.moe
        if mo is not None:          # the router's rows a slice: E + 1, at S > 1 split in whole
            R = Q.grp_r("int8", cfg.WG)     # groups (moe._moe_ffn_tp: zero rows past E + 1)
            self.er = mo.E + 1 if S == 1 else -(-(mo.E + 1) // (S * R)) * R
        self.og = 4 if nl % 4 == 0 else 2               # heads per out_proj MM
        self.sexact = s_exact(spec, S, cfg) and not self.qkonce     # (S = 1's groups interleaved)
        self.mats = {LIN: {"wh": (nl // 2 * self.RP, H), "wab": (2 * nl, H),
                           "wout": (nl // self.og * H, self.og * dv),
                           **({"wqk": (self.QC, H)} if self.qkonce else {}), **mlp},
                     ATTN: {"wq": (self.nq_loc * d, H), "wgate": (self.nq_loc * d, H),
                            "wk": (self.nkv_loc * d, H), "wv": (self.nkv_loc * d, H),
                            "wo": (self.h_loc, spec.n_q * d), **mlp}}
        if self.grouped:            # per head group: its pairs' blocks, then its out_proj block
            del self.mats[LIN]["wh"], self.mats[LIN]["wout"]
            self.sbytes = self.se * 2 * dv * dk         # a pair's state slots: [slot, head]
        else:
            self.sbytes = self.se * nl * dv * dk        # the layer's states: [slot, head]
        # formats group -> its block layouts (_layout); layer 0's those of the image
        self.layouts = {g: self._layout(g, mlp, mb) for g in dict.fromkeys(self.lf)}
        g0 = self.layouts[self.lf[0]]
        self.lofs, self.mf, self.dchunk = g0.lofs, g0.mf, g0.dchunk
        if self.grouped:
            self.pofs, self.PS, self.gofs, self.GS = g0.pofs, g0.PS, g0.gofs, g0.GS
        # the runs (plan): a layer's key is its kind and formats group; then the MTP layer
        self.keys = tuple(zip(spec.kinds, self.lf))
        self.plan = plan(self.keys)
        size = lambda k: self.layouts[k[1]].size[k[0]]                  # noqa: E731
        lsize = lambda k: self.layouts[k[1]].lsize[k[0]]                # noqa: E731
        self.loc = _place(hb, self.plan, size)
        self.lloc = _place(b, self.plan, lsize) if self.high else self.loc     # (the states')
        msize = lambda k: self.layouts[k[1]].msize[k[0]]                # noqa: E731
        mb_ = _Bump(mixer_base(cfg), _align(cfg)) if self.mix else None
        self.mloc = _place(mb_, self.plan, msize) if self.mix else self.loc    # (the mixers')
        # a unit's repeated part (the DeltaNet layers before each attention layer) loops again
        # inside the unit's loop in the prefill programs (NEST: qwen35_rows, tiled.qwen35_tile);
        # high: the part's bytes (its weights', its states')
        self.subs = subs(self.plan, size)
        if self.mix:
            ls, ms = subs(self.plan, lsize), subs(self.plan, msize)
            self.subs = {f: [(e0, su, r, (n, ln, mn)) for (e0, su, r, n), (*_, ln), (*_, mn)
                             in zip(v, ls[f], ms[f])] for f, v in self.subs.items()}
        elif self.high:
            ls = subs(self.plan, lsize)
            self.subs = {f: [(e0, su, r, (n, ln)) for (e0, su, r, n), (*_, ln) in zip(v, ls[f])]
                         for f, v in self.subs.items()}
        if spec.mtp:                # an attention block after the model's (the formats without a
            gm = tuple(fmt(k, spec.layers) for k in ("delta", "attn", "gateup", "down"))  # range)
            if gm not in self.layouts:
                self.layouts[gm] = self._layout(gm, mlp, mb)
            self.keys += ((ATTN, gm),)
            run = [(spec.layers, self.keys[-1:], 1)]
            self.loc.update(_place(hb, run, size))
            if self.high:
                self.lloc.update(_place(b, run, lsize))
            if self.mix:
                self.mloc.update(_place(mb_, run, msize))
        n_attn = spec.kinds.count(ATTN)
        head = cap * d + cap * k_scale_bytes(d, D) + d * cap + 4 * cap
        self.kv_bytes = ((n_attn + spec.mtp) * self.nkv_loc * head * batch     # KV caches,
                         + (spec.layers - n_attn) * self.slots *                # conv rings
                         (4 * (nl // 2 * (K - 1) * self.CP + (K - 1) * self.QC)  # (qkonce's)
                          + self.se * nl * dv * dk))                       # and states
        self.mtpo, self.nd = {}, min(MTP_VOCAB, spec.vocab)
        if spec.mtp:                # the MTP's norms (embedding, hidden, output), fc, draft head
            self.mtpo = {"ge": hb.alloc(4 * H), "gh": hb.alloc(4 * H), "gm": hb.alloc(4 * H),
                         "fc": tuple(hb.alloc(x) for x in mb(H, 2 * H, self.mtp_fc)),  # (attn's)
                         "dh": tuple(hb.alloc(x) for x in mb(self.nd, H, "fp4"))}
        self.head = tuple(hb.alloc(x) for x in mb(self.v_loc, H, self.head_format))
        # the int8 embedding rows of the resident decode are the tied int8 head's (S = 1; not
        # a grouped head's, qwen3.Image; not a head above HIGH: the rows' address is a token's)
        shared = spec.tied and self.head_format == "int8" and S == 1 and cfg.WG == 1 and \
            not self.high
        # every expert resident (a pool in DRAM after the head: above HIGH, OTPU_POOL=1 or S > 1,
        # a slot for each expert), else path (a)'s expert slots, the experts streamed
        resident_pool = mo is not None and (mo.E if experts is None else experts) == mo.E and \
            (self.high or os.environ.get("OTPU_POOL") == "1" or S > 1)
        if embed_host is None:      # an int8 table of its own on the host by default where its
            embed_host = (bool(lookup) and not shared         # DRAM is expert slots; with every
                          and getattr(spec, "embed", "f32") == "int8"   # expert resident on the
                          and mo is not None and not resident_pool)     # card, as a dense
                                                                        # model's (the generate
                                                                        # loop gathers its rows)
        if embed_host and not lookup:
            raise ValueError("embed_host needs the image's lookup tables (lookup=True)")
        # (a MoE model's mask tiles: its tiled layer runs, on unless OTPU_TILE=0 as it is built)
        from .prefill import layer_tile_flag
        self.lookup = _lookup_alloc(b, spec, cap, D=D, head=self.head if shared else None,
                                    M=cfg.MCOLS, embed_host=bool(embed_host),
                                    rows=self.grows,                # (layer 0's R rows)
                                    imem=cfg.IMEM_WORDS,
                                    masks=bool(self.prefill_rows) and layer_tile_flag(None, True),
                                    waitc=cfg.SEQ_WC, S=S, wg=gall_wg(cfg),
                                    chash=cfg.CHASH,
                                    mask_blocks=SPAN_P if batch > 1 else 1) \
            if lookup else {}
        if lookup and spec.mtp:     # the MTP loop's chain area (opentpu/llm/mtp.py)
            from .mtp import gen_alloc
            self.lookup["mtpgen"] = gen_alloc(b, cap, spec, imem_words=cfg.IMEM_WORDS,
                                              waitc=cfg.SEQ_WC)
        self.choices = {"embed_host": self.embed_host,          # (the compile worker's
                        "formats": self.formats}                # image)
        if self.high:
            self.choices["high"] = True
        self.before_slots(b)
        self.offload = self.pool = None
        if mo is not None:          # path (a)'s words and expert slots (docs/offload.md)
            self.fmt = MO.ExpertFormat(H, mo.ffn, D, wformat, gall_wg(cfg), S=S)
            # (in groups, with the channel hash: the slots' lanes by their addresses)
            self.slot_lanes = (cfg.WG, D, self.fmt.data) if self.fmt.grp > 1 and cfg.CHASH \
                else None
            n = mo.E if experts is None else experts
            if not mo.k <= n <= mo.E:
                raise ValueError(f"{n} expert slots per layer: from top-k {mo.k} to {mo.E}")
            if S > 1 and n != mo.E:
                raise ValueError(f"MoE at S = {S}: every expert resident ({n} of {mo.E} slots)")
            # (S > 1: tensor-parallel experts, each slice's parts in its pool: moe._moe_ffn_tp;
            # OTPU_MOE_GROUPS, its activations' gathers a layer)
            self.moe_groups = int(os.environ.get("OTPU_MOE_GROUPS", 0)) if S > 1 else 0
            if resident_pool:           # every expert in DRAM, after the head
                nx = (spec.layers + spec.mtp) * mo.E
                self.pool = MO.resident_pool(hb if _spill is None else
                                             _Bump(_spill[0], _align(cfg)), nx, self.fmt)
                if em and self.prefill_rows:    # expert-major's records, last
                    b.next = -(-b.next // LINE) * LINE          # in the low area: no other
                    self.io["em"] = b.alloc(self.prefill_rows    # (moves; em_q: QONCE's)
                                            * MO.em_record(H, mo.k, MO.em_qd(cfg)))
                    if cfg.DNCHUNK and LIN in spec.kinds:       # the DeltaNet rows' (dn_rows)
                        b.next = -(-b.next // LINE) * LINE
                        self.io["dnq"] = b.alloc(4 * self.prefill_rows * self.dnq_row())
                        self.io["dno"] = b.alloc(4 * self.prefill_rows * self.nl * dv)
                        if self.dn_core_ok():           # the chunked core's scratch (dn_run)
                            b.next = -(-b.next // DC.ALIGN) * DC.ALIGN
                            self.io["dns"] = b.alloc(self.dns_bytes())
            else:                       # (the directory's slots below 4 GiB: 32-bit entries)
                self.offload = Layout.build((b.next + 4095) // 4096 * 4096, mo.E, mo.k,
                                            [n] * (spec.layers + spec.mtp),  # (Spec.mtp: the
                                            self.fmt.nbytes)    # MTP layer's MoE, a layer more)
                b.next = self.offload.end
        self.low_end, self.nbytes = b.next, max(b.next, hb.next)
        if self.mix:                # (the mixers' end inside the window; the bytes past it unread)
            if mb_.next > high_base(cfg):
                raise MemoryError(f"the DeltaNet mixers' weights need "
                                  f"{(mb_.next - mixer_base(cfg)) / 2**20:.0f} MiB per slice, "
                                  f"above the bank-half window's upper half "
                                  f"({(high_base(cfg) - mixer_base(cfg)) / 2**20:.0f} MiB)")
            if _spill is None and self.pool is not None and hb.next > cfg.DRAM_BYTES:
                # the pool spills: the layout again with it at the highest place that fits
                # (from the window's upper half on), every other weight after it to the top
                A, pb = _align(cfg), self.pool.experts * self.pool.ebytes
                ob = self.pool.base - high_base(cfg)    # (the other weights')
                ws = (cfg.DRAM_BYTES - ob) // A * A
                at = (ws - pb) // A * A
                if at < mb_.next:
                    raise MemoryError(f"the resident pool and the other weights need "
                                      f"{(pb + ob) / 2**20:.0f} MiB per slice, above the "
                                      f"{(cfg.DRAM_BYTES - mb_.next) / 2**20:.0f} MiB past the "
                                      f"DeltaNet mixers' weights")
                Image.__init__(self, **args, _spill=(at, ws))
                return
            self.wlo = mixer_base(cfg)
            self.wskip = (mb_.next, high_base(cfg) if _spill is None else _spill[0])
        low_check(self, cfg, self.high)
        if self.nbytes > cfg.DRAM_BYTES:
            raise MemoryError(f"model image needs {self.nbytes / 2**20:.0f} MiB per slice, "
                              f"DRAM_BYTES is {cfg.DRAM_BYTES / 2**20:.0f} MiB")

    def before_slots(self, b) -> None:
        """A subclass's own DRAM, allocated from `b` (qwen3._Bump) after the model's and before
        the expert slots (which take the rest): MiniCPM-V's vision side (minicpmv.Image)."""

    @property
    def mtp_fc(self) -> str:
        """The MTP's fc format: its layer's attention format."""
        return self.layouts[self.keys[self.spec.layers][1]].mf["wq"]

    def _layout(self, g: tuple, mlp: dict, mb) -> SimpleNamespace:
        """The layer block layouts of formats group g (DeltaNet, attention, gate / up, down):
        each projection's format (mf), W_down's chunk, the DeltaNet and attention blocks'
        offsets (the norms and the MLP first, at the same offsets in both) and sizes (size);
        group-major, a pair's block (pofs, PS) and a head group's (gofs, GS). Image.high: the
        states' offsets in a block of their own, its size lsize (else 0), group-major a pair's
        cv and state in a block of LPS bytes, a group's pairs' of LGS (else PS and GS). Image.mix:
        slow, the mixer matrices whose scale rows are in the states' block (lofs' scale offset
        from its start). mb(n, k, f): an [n, k] matrix's (data, scale) bytes."""
        spec, cfg, cap = self.spec, self.cfg, self.cap
        D, H, d, F_ = cfg.D, spec.hidden, spec.head_dim, spec.ffn
        dk, dv, nl, og = spec.lin_dk, spec.lin_dv, self.nl, self.og
        A = _align(cfg)

        def mat(bump, n, k, f):
            return tuple(bump.alloc(x) for x in mb(n, k, f))
        fl, fa, fg, fd = g
        mf = {"wh": fl, "wab": fl, "wout": fl, "wq": fa, "wgate": fa, "wk": fa, "wv": fa,
              "wo": fa, "wg": fg, "wu": fg, "wd": fd}       # each projection's format (KINDS)
        if self.qkonce:
            mf["wqk"] = fl
        lb = _Bump(align=A)                             # offsets inside one layer block
        common = {"g_in": lb.alloc(4 * H), "g_post": lb.alloc(4 * H)}
        for name, (n, k) in mlp.items():
            common[name] = mat(lb, n, k, mf[name])
        S = gs = cfg.S              # W_down's parts S chunks, the all-gather's order (qwen3's)
        mo, q = spec.moe, D if fd == "int8" else 2 * D
        msx = mlp_s_exact(spec, S, D, fd, cfg)
        if mo is not None and S > 1:    # (MoE: the shared expert's in S = 1's parts, gathered
            dchunk, gs = _chunk(F_, D, q), 1                    # first: moe._moe_ffn_tp)
        elif msx:                   # (S = 1's chunks in S parts: their order the gather's)
            dchunk = _chunk(F_, D, q) // S
        else:
            dchunk = _chunk(self.f_loc, D, q)
        common["wd"] = [mat(lb, self.h_loc, gs * dchunk, fd) for _ in range(F_ // (gs * dchunk))]
        if mo is not None:          # the router (int8, the shared expert's gate its last row)
            common.update(router=mat(lb, self.er, H, "int8"), gbase=lb.alloc(4))
        ns = SimpleNamespace(mf=mf, dchunk=dchunk, gs=gs, msx=msx)
        own = lambda bump: _Bump(align=A) if self.high else bump       # noqa: E731 (states')
        lnb = _Bump(lb.next, A)
        lns = own(lnb)
        lnm = _Bump(align=A) if self.mix else lnb      # (the mixers': Image.mix)
        lin = dict(common, alog=lnb.alloc(4 * nl), dtb=lnb.alloc(4 * nl), gn=lnb.alloc(4 * dv))
        if self.grouped:            # per head group: its pairs' blocks, then its out_proj block
            pb, gb = _Bump(align=A), _Bump(align=A)
            pl = own(pb)
            ns.pofs = {"wh": mat(pb, self.RP, H, fl),
                       "cv": pl.alloc(4 * self.CVW),
                       "state": pl.alloc(self.se * 2 * dv * dk * self.slots)}
            ns.PS = pb.next                             # a pair's block, bytes
            ns.LPS = pl.next                            # (its states', high)
            for _ in range(og // 2):
                gb.alloc(ns.PS)
            ns.gofs = {"wout": mat(gb, H, og * dv, fl)}
            if not self.mix:
                ns.gofs["eb"] = gb.alloc(4 * 4 * (og // 2))     # its pairs' gates, per token
            ns.GS = gb.next                             # a head group's block
            lin["groups"] = lnm.alloc(nl // og * ns.GS)
            ns.LGS = og // 2 * ns.LPS if self.high else ns.GS
            if self.mix:            # (the gates after the group's pairs' states)
                ns.leb = ns.LGS
                ns.LGS = ns.leb + -(-4 * 4 * (og // 2) // A) * A
            lin["lgroups"] = lns.alloc(nl // og * ns.LGS) if self.high else lin["groups"]
        else:
            lin.update(cv=lns.alloc(4 * nl // 2 * self.CVW),  # per pair: taps, then the ring
                       state=lns.alloc(self.se * nl * dv * dk * self.slots))
            if self.qkonce:         # the key heads' q, k taps, then their ring
                lin["qkcv"] = lns.alloc(4 * self.QCV0)
        ab = _Bump(lb.next, A)
        abl = own(ab)
        attn = dict(common, qn=ab.alloc(4 * d), kn=ab.alloc(4 * d))
        # (Image.mix: a mixer's scale rows in the states' block, the window's lower half, apart
        # from its data's banks: slow, their names)
        ns.slow = tuple(n for n in self.mats[LIN] if n in MIXER_PARTS) if self.mix else ()
        for kind, bump, L in ((LIN, lnb, lin), (ATTN, ab, attn)):
            for name, (n, k) in self.mats[kind].items():
                if name in L:
                    continue
                if kind == LIN and name in ns.slow:
                    db, sb = mb(n, k, mf[name])
                    L[name] = (lnm.alloc(db), lns.alloc(sb))
                else:
                    L[name] = mat(lnm if kind == LIN and name in MIXER_PARTS else bump, n, k,
                                  mf[name])
        kv = lambda: [{"k": abl.alloc(cap * d), "ks": abl.alloc(cap * k_scale_bytes(d, D)),  # noqa: E731
                       "vt": abl.alloc(d * cap), "vs": abl.alloc(4 * cap)}
                      for _ in range(self.nkv_loc)]
        attn["kv"] = kv()
        if self.batch > 1:          # every sequence's caches, slot 0's first (batch 1: as before)
            attn["kvs"] = [attn["kv"]] + [kv() for _ in range(1, self.batch)]
        ns.lofs = {LIN: lin, ATTN: attn}
        r4k = lambda n: (n + 4095) // 4096 * 4096                     # noqa: E731
        ns.size = {LIN: r4k(lnb.next), ATTN: r4k(ab.next)}
        ns.lsize = {LIN: r4k(lns.next), ATTN: r4k(abl.next)} if self.high else {LIN: 0, ATTN: 0}
        ns.msize = {LIN: r4k(lnm.next), ATTN: 0} if self.mix else {LIN: 0, ATTN: 0}
        return ns

    def _off(self, li, it=None, loc=None) -> Affine:
        """The block address of layer li (static), or of element li of its run's unit at
        iteration `it` (a loop variable); loc: self.loc (the weights') or self.lloc."""
        base, us, i, o, _ = (self.loc if loc is None else loc)[li]
        return Affine(base + o) + Affine.of(i if it is None else it) * us

    def _loff(self, li, it=None) -> Affine:
        """_off of layer li's states' block (Image.high; else its block)."""
        return self._off(li, it, self.lloc)

    def _moff(self, li, it=None) -> Affine:
        """_off of layer li's mixers' block (Image.mix; else its block)."""
        return self._off(li, it, self.mloc)

    def pair_offset(self, q: int, lay=None, low: bool = False) -> int:
        """Pair q's block in a group-major DeltaNet layer block of layouts `lay` (layer 0's by
        default) (bytes from its start); low: its states' block's in the states' layer block
        (Image.high)."""
        lay = self.layouts[self.lf[0]] if lay is None else lay
        g, i = divmod(q, self.og // 2)
        if low:
            return lay.lofs[LIN]["lgroups"] + g * lay.LGS + i * lay.LPS
        return lay.lofs[LIN]["groups"] + g * lay.GS + i * lay.PS

    def cv_offset(self, q: int, lay=None) -> int:
        """Pair q's taps, then its window, in a DeltaNet layer's states' block (its block, but
        with Image.high) of layouts `lay` (layer 0's by default) (bytes from its start)."""
        lay = self.layouts[self.lf[0]] if lay is None else lay
        if self.grouped:
            return self.pair_offset(q, lay, low=True) + lay.pofs["cv"]
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

        qb = QC.Batch()                 # the matrices, quantized in worker processes

        gw = cfg.WG                     # in groups of WG rows (docs/isa.md "Weight groups")
        gh = cfg.CHASH                  # (their rows in the channel hash's order, at their address)

        def put_q1(s, addr_pair, a, fmt, fold=None):
            """fold: the rows whose fp32 Sw (a scale row's first word) takes -log2(e) after the
            quantization (Image.dn_fold: z's and b's rows, their codes unchanged)."""
            qb.submit(a, fmt, D, lambda q, sc, k=a.shape[1]: [
                put(s, x, y) for x, y in zip(addr_pair, Q.grouped(
                    q, _fold_sw(sc, fold), fmt, k, gw, D, at=addr_pair[0] if gh else None))])

        def put_q(addr_pair, parts, fmt, fold=None):
            for s, p in enumerate(parts):
                put_q1(s, addr_pair, p, fmt, fold)

        def put_parts(addrs, parts, fmt, C, gs=S):
            """Each slice's rows quantized whole and stored in column parts of gs C in the
            all-gather's order (qwen3's; gs 1: S = 1's parts, a MoE layer's at S > 1)."""
            for s, p in enumerate(parts):
                k = p.shape[1]
                qb.submit(p, fmt, D, lambda q, sc, s=s, k=k: [
                    (put(s, a0, pq), put(s, a1, ps))
                    for (a0, a1), pr in zip(addrs, Q.gathered_parts(q, sc, fmt, k, C, gs, D))
                    for pq, ps in [Q.grouped(*pr, fmt, gs * C, gw, D, at=a0 if gh else None)]])

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
            lbase = self._loff(i).const                 # (the states' block: Image.high)
            mbase = self._moff(i).const                 # (the mixers' block: Image.mix)
            mf = lay.mf
            Lo = {k: (((mbase + v[0], lbase + v[1]) if k in lay.slow else
                       tuple((mbase if k in MIXER_PARTS else base) + x for x in v))
                      if isinstance(v, tuple) else
                      ((lbase if k in LOW_PARTS else mbase if k in MIXER_PARTS else base) + v
                       if isinstance(v, int) else v))
                  for k, v in lay.lofs[kind].items()}
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
                if self.dn_fold:            # (SiLU's -log2(e): every convolved channel, q k v)
                    taps = (taps.astype(np.float64) * DN_C).astype(np.float32)
                hs = [range(s * nl, (s + 1) * nl) for s in range(S)]
                if self.sexact:     # S = 1's groups of og heads s, s + S, ... (s_exact)
                    og = self.og
                    hs = [[h for g in range(s, spec.lin_heads // og, S)
                           for h in range(g * og, (g + 1) * og)] for s in range(S)]
                # per pair of heads (a, b): the q, k, v rows of a, of b, then the z rows of a, of
                # b; its taps; per head group, its out_proj column block (og heads)
                og = self.og
                # (shared: the key head's q and k rows, then v of a, of b; taps by block;
                # qkonce: v of a, of b, the q and k rows the slice's key heads' block below)
                blocks = ((lambda h: [chans[h][:2 * dk], chans[h][2 * dk:], chans[h + 1][2 * dk:]])
                          if self.shared else
                          (lambda h: [chans[h][2 * dk:], chans[h + 1][2 * dk:]]) if self.qkonce
                          else (lambda h: [chans[h], chans[h + 1]]))
                prows = [[np.concatenate([qkv[c] for c in blocks(h)] + [wz[h * dv:(h + 2) * dv]])
                          for h in hh[::2]] for hh in hs]
                # (dn_fold: a pair's z rows, its last 2 dv, take -log2(e) in their Sw)
                zf = (lambda n: np.arange(n - 2 * dv, n)) if self.dn_fold else (lambda n: None)
                ptaps = [[f32(np.concatenate([taps[c].reshape(-1, self.bw, K).transpose(0, 2, 1)
                                              .reshape(-1, self.bw) for c in blocks(h)]))
                          for h in hh[::2]] for hh in hs]      # rows (block, tap)
                # out_proj's rows quantized whole (as the emulation's), its head groups'
                # column blocks their column parts (quant.column_parts)
                fo = mf["wout"]
                gouts = Q.column_parts(*QC.quantize_mxu(wout, fo, D), fo, wout.shape[1],
                                       og * dv, D)
                gouts = [gouts[s::S] if self.sexact else gouts[s * nl // og:(s + 1) * nl // og]
                         for s in range(S)]
                if S > 1:       # owned columns (_out_proj): slice s's block g, its n rows of
                    #             every slice's group g, slice by slice
                    gouts = [[tuple(np.concatenate([gouts[t][g][i][s * n:(s + 1) * n]
                                                    for t in range(S)]) for i in range(2))
                              for g in range(nl // og)] for s in range(S)]
                if self.grouped:
                    for s in range(S):
                        for q, (pr, pt) in enumerate(zip(prows[s], ptaps[s])):
                            o = mbase + self.pair_offset(q, lay)
                            put_q1(s, (o + lay.pofs["wh"][0], o + lay.pofs["wh"][1]), pr,
                                   mf["wh"], zf(len(pr)))
                            put(s, lbase + self.cv_offset(q, lay), pt)
                        for g, pr in enumerate(gouts[s]):
                            o = Lo["groups"] + g * lay.GS
                            gq, gs = Q.grouped(*pr, fo, og * dv, gw, D,
                                               at=o + lay.gofs["wout"][0] if gh else None)
                            put(s, o + lay.gofs["wout"][0], gq)
                            put(s, o + lay.gofs["wout"][1], gs)
                else:
                    zr = None
                    if self.dn_fold:        # every pair's z rows in the slice's concatenation
                        n0 = np.cumsum([0] + [len(x) for x in prows[0]])
                        zr = np.concatenate([n0[i] + zf(len(x)) for i, x in enumerate(prows[0])])
                    put_q(Lo["wh"], [np.concatenate(pr) for pr in prows], mf["wh"], zr)
                    for s in range(S):
                        gq, gs = Q.grouped(np.concatenate([gq for gq, _ in gouts[s]]),
                                           np.concatenate([gs for _, gs in gouts[s]]), fo,
                                           og * dv, gw, D, at=Lo["wout"][0] if gh else None)
                        put(s, Lo["wout"][0], gq)
                        put(s, Lo["wout"][1], gs)
                    for s in range(S):
                        for q, pt in enumerate(ptaps[s]):
                            put(s, Lo["cv"] + q * 4 * self.CVW, pt)
                if self.qkonce:     # the slice's key heads' q | k rows, their taps (block, tap)
                    kq = [np.r_[j * dk:(j + 1) * dk, NK + j * dk:NK + (j + 1) * dk]
                          for j in range(spec.lin_nk)]
                    kk = [range(s * self.nkl, (s + 1) * self.nkl) for s in range(S)]
                    put_q(Lo["wqk"], [np.concatenate([qkv[kq[j]] for j in js]) for js in kk],
                          mf["wqk"])
                    for s, js in enumerate(kk):
                        put(s, Lo["qkcv"], f32(np.concatenate(
                            [taps[kq[j]].T for j in js])))
                put_q(Lo["wab"], [np.concatenate([W[a + "in_proj_a.weight"][list(hh)],
                                                  W[a + "in_proj_b.weight"][list(hh)]])
                                  for hh in hs], mf["wab"],
                      np.arange(nl, 2 * nl) if self.dn_fold else None)    # (b's rows)
                alog, gn = W[a + "A_log"], W[a + "norm.weight"]
                if self.dn_fold:            # -log2(e) e^A_log; the output norm's gain / log2(e)
                    alog = DN_C * np.exp(np.asarray(alog, np.float64))
                    gn = np.asarray(gn, np.float64) / -DN_C
                for s, hh in enumerate(hs):
                    put(s, Lo["alog"], f32(np.asarray(alog)[list(hh)]))
                    put(s, Lo["dtb"], f32(np.asarray(W[a + "dt_bias"])[list(hh)]))
                    put(s, Lo["gn"], f32(gn))
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
            if lay.msx:             # (mlp_s_exact: part s of each of S = 1's chunks, W_down's
                C = lay.dchunk      # columns in order)
                fr = lambda a: [np.concatenate([a[j + s * C:j + (s + 1) * C]       # noqa: E731
                                                for j in range(0, len(a), S * C)])
                                for s in range(S)]
            else:
                fr = lambda a: rows(a, self.f_loc)                                  # noqa: E731
            put_q(Lo["wg"], fr(W[mp + "gate_proj.weight"]), mf["wg"])
            put_q(Lo["wu"], fr(W[mp + "up_proj.weight"]), mf["wu"])
            put_parts([(base + a0, base + a1) for a0, a1 in lay.lofs[kind]["wd"]],
                      rows(W[mp + "down_proj.weight"], n), mf["wd"],
                      *((S * lay.dchunk, 1) if lay.msx else (lay.dchunk, lay.gs)))
            if spec.moe is not None:
                rw = np.concatenate([W[p + "mlp.gate.weight"],
                                     W[p + "mlp.shared_expert_gate.weight"]])
                if S > 1:                           # each slice's rows, zero rows past E + 1
                    rw = rows(np.pad(rw, ((0, S * self.er - len(rw)), (0, 0))), self.er)
                put_q(Lo["router"], rw if S > 1 else [rw], "int8")
                for s in range(S):
                    put(s, Lo["gbase"], f32([i * spec.moe.E]))
        head = W["model.embed_tokens.weight"] if spec.tied else W["lm_head.weight"]
        if spec.mtp:                # (one slice: Spec.check)
            mo = self.mtpo
            for k, n in (("ge", "pre_fc_norm_embedding"), ("gh", "pre_fc_norm_hidden"),
                         ("gm", "norm")):
                put(0, mo[k], g1(f"mtp.{n}.weight"))
            put_q(mo["fc"], [W["mtp.fc.weight"]], self.mtp_fc)
            put_q(mo["dh"], [head[:self.nd]], "fp4")
        put_head(self, W, qb, put, put_q, rows)     # (and the tied int8 head's gather scales)
        qb.wait()
        if self.pool is not None:               # every expert in its place (Image.high)
            MO.pool_build(self.pool, lambda qb, g, done, s=0: self.fmt.submit(
                qb, *self._expert_w(W, g), done, part=s), getattr(self, "slot_lanes", None),
                put, S)
        if self.lookup:
            _lookup_build(put, S, W, spec, self.cap, self.lookup)
        if "mtpgen" in self.lookup:
            from .mtp import gen_build
            for s in range(S):
                gen_build(put, s, S, spec, self.lookup["mtpgen"])
        return imgs

    # ---- path (a): the expert pool and its server (docs/offload.md)
    def _expert_w(self, W: dict, g: int) -> tuple:
        """Global expert g's W_gate, W_up and W_down (layer g // E, expert g % E; Spec.mtp:
        layer `layers` is the MTP layer's)."""
        E, j = self.spec.moe.E, g // self.spec.moe.E
        p = "mtp.layers.0." if j == self.spec.layers else f"model.layers.{j}."
        return _expert(W, p, g % E)

    def expert(self, W: dict, g: int) -> np.ndarray:
        """Global expert g in its slot's bytes."""
        return self.fmt.pack(*self._expert_w(W, g))

    def serve(self, W: dict, backend, pool_file=None) -> ExpertServer:
        """The host's expert server on the backend's DRAM (moe.serve)."""
        return MO.serve(self.offload, lambda g: self.expert(W, g), backend, pool_file,
                        lanes=getattr(self, "slot_lanes", None),
                        entry_lanes=getattr(self, "entry_lanes", None))

    def state_regions(self, slot: int = 0) -> list:
        """The context's state that its KV caches do not hold, as DRAM regions (address,
        bytes; every slice's): each DeltaNet layer's states and convolution windows in `slot`
        (a rollback restores them from a snapshot: opentpu/llm/snapshot.py)."""
        m, out = self.descriptors(0, slot), []
        for li, kind in enumerate(self.spec.kinds):
            if kind != LIN:
                continue
            dn, pairs = m.layer(li).dn, range(self.nl // 2)
            # in address order (snapshot.merged joins neighbours): group-major, a pair's
            # window, then its states; array-major, the windows, then the states (then qkonce's
            # key heads' window)
            ts = [t for q in pairs for t in (dn.window(q), dn.state(q, 0), dn.state(q, 1))] \
                if self.grouped else [dn.window(q) for q in pairs] + \
                [dn.state(q, a) for q in pairs for a in (0, 1)] + \
                ([dn.qk_window()] if self.qkonce else [])
            out += [(Affine.of(t.base).static(), t.esize * math.prod(t.shape)) for t in ts]
        return out

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

    def compile_step(self, pos: int, block: int = ATTN_BLOCK, tok: int | None = None,
                     slot: int = 0) -> list:
        """One program per slice: the decode token at position `pos` (qwen35_step); slot: the
        sequence's (Image(batch=N): its KV caches, states and windows)."""
        def desc(s):
            m = step_descriptors(self, s)
            if slot:
                m = SimpleNamespace(**{**vars(self.descriptors(s, slot)), "fill": m.fill})
            return m
        return [trace_fit(qwen35_step, self.cfg, s,
                          lambda s=s: {"m": desc(s), "pos": pos,
                                       "block": block, **_tok_arg(self, tok)}).finish()
                for s in range(self.cfg.S)]

    def compile_rows(self, rows, logit_rows, block: int = ATTN_BLOCK, tokens=None,
                     slot: int = 0, fork: bool = False, hidden: bool = False,
                     nest: bool = True) -> list:
        """One program per slice: consecutive positions of the sequence at once
        (qwen35_rows); slot, fork, hidden: MTP decoding's (qwen35_rows, docs/mtp.md 9); nest
        False: the unit's DeltaNet layers unrolled (MTP's verify rows)."""
        mo = self.spec.moe
        if mo is not None and not self.spec.mtp:
            raise ValueError("a MoE model runs its prompt through the decode step (its MoE "
                             "block routes one token): rows=1")
        if mo is not None and len(rows) * mo.k > LINE // 4:   # (MTP's verify: 2 rows, one
                                                               # request for their union)
            raise ValueError(f"{len(rows)} rows of a MoE model: one request names their "
                             f"{len(rows) * mo.k} ids, a request row holds {LINE // 4}")
        if len(rows) > self.rows:
            raise ValueError(f"{len(rows)} rows, the image's I/O area holds {self.rows}")
        if any(r != (0, rows[0][1] + i) for i, r in enumerate(rows)):
            raise ValueError("Qwen3.5 rows must be consecutive positions of sequence 0")
        if (fork or hidden) and not self.spec.mtp:
            raise ValueError("fork / hidden rows need an MTP image (Spec.mtp)")
        mtp = {"fork": True} if fork else {}
        if hidden:
            mtp["hidden"] = True
        if not nest:
            mtp["nest"] = False
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

    def em_ok(self) -> bool:
        """Expert-major runs fit the image: its prefill rows and the expert server's words, or a
        resident pool with its records (Image(em=True): every expert in DRAM, no server)."""
        return bool(self.prefill_rows) and (self.offload is not None or "em" in self.io)

    def compile_layer_run(self, li: int, blocks: int, block: int = ATTN_BLOCK, R: int = 1,
                          embedded: bool = False, at: int | None = None, hint: bool = False,
                          em: bool = False, slot: int = 0):
        """(programs, run_args): qwen35_layer_run of layer li, R rows at a run-time position of
        bucket `blocks` (from conv_k - 1: the convolutions' taps) and a run-time row of the
        prefill chunk (run arguments: RunPos.values and "row"), or at the compile-time
        position `at` (the rows before conv_k - 1; embedded); li < 0: qwen35_embed_run. hint:
        the run ends with the next MoE layer's hint (qwen35_layer_run); em: expert-major
        (docs/offload.md 13.11: the rows in the scratch's records, the MoE's prologue and the
        last layer's combine; compile_expert_run). The image needs lookup tables and its
        prefill rows. slot: the DeltaNet states' and windows' (an MTP image's, Spec.mtp)."""
        from ..compiler import RunVar
        spec, K = self.spec, self.spec.conv_k
        if not self.lookup or not self.prefill_rows:
            raise ValueError("a layer run needs lookup tables and the image's prefill rows")
        if self.cfg.S != 1 and (hint or em and self.pool is None):
            raise ValueError("a layer run's hint or expert-major MoE is one slice's (S = 1; "
                             "expert-major at S > 1: a resident pool's)")
        k = spec.moe.k if spec.moe is not None else 1
        if not 1 <= R <= RUN_ROWS or R * k > LINE // 4:
            raise ValueError(f"{R} rows a layer run")
        if hint and self.offload is None or em and not self.em_ok():
            raise ValueError("a layer run's hint or expert-major MoE needs the expert server's "
                             "words (offload; expert-major: or a resident pool's records, em)")
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
        # a program per slice (slot: the DeltaNet states' of an MTP image); at S > 1 each
        # slice's layer is its part of the tensor-parallel step's, its rows' residual whole
        if li < 0:
            bs = [qwen35_embed_run.trace(self.cfg, s, {"m": self.descriptors(s, slot), "pos": pos,
                                                       "row": row, "em": em})
                  for s in range(self.cfg.S)]
        else:
            bs = [qwen35_layer_run.trace(self.cfg, s, {"m": self.descriptors(s, slot), "li": li,
                                                       "pos": pos, "row": row, "block": block,
                                                       "R": R, "embedded": embedded,
                                                       "hint": hint, "em": em})
                  for s in range(self.cfg.S)]
        return [b.finish() for b in bs], list(bs[0].run_args)

    def compile_mtp_kv_run(self, blocks: int, block: int = ATTN_BLOCK, R: int = 1,
                           em: bool = False, keep: bool = False):
        """(programs, run_args): qwen35_mtp_kv_run, R rows at a run-time position of bucket
        `blocks` (from conv_k - 1) and a run-time row of the prefill chunk (RunPos.values and
        "row"); em: expert-major's records (keep: their residual to the chunk's rows too). An
        MTP image's (Spec.mtp) with lookup tables, its prefill rows and the embedding rows from
        the host's slot (embed_host)."""
        from ..compiler import RunVar
        if not (self.spec.mtp and self.lookup and self.prefill_rows and self.embed_host):
            raise ValueError("an MTP prefill run needs an MTP image's lookup tables, its prefill "
                             "rows and the embedding rows from the host's slot (embed_host)")
        if self.cfg.S != 1:
            raise ValueError("an MTP prefill run is one slice's (S = 1)")
        if not 1 <= R <= self.grows:
            raise ValueError(f"{R} rows an MTP prefill run: the host's slot holds {self.grows}")
        lo = max((blocks - 1) * block, self.spec.conv_k - 1)
        pos = RunPos(blocks, block, lo, self.lookup["zmask"], self.cap)
        pos.tpos.bound -= R - 1     # the run's last row in the block too: tpos <= block - R
        b = qwen35_mtp_kv_run.trace(self.cfg, 0, {"m": self.descriptors(0), "pos": pos,
                                                  "row": RunVar("row", self.prefill_rows),
                                                  "block": block, "R": R, "em": em,
                                                  **({"keep": True} if keep else {})})
        return [b.finish()], list(b.run_args)

    def compile_prefill_head(self, em: bool = False):
        """(programs, run_args): qwen35_prefill_head at a run-time row ("row"); em: from the
        expert-major records, after the last layer's combine."""
        from ..compiler import RunVar
        row = RunVar("row", self.prefill_rows)
        bs = [qwen35_prefill_head.trace(self.cfg, s, {"m": self.descriptors(s), "row": row,
                                                      "em": em})
              for s in range(self.cfg.S)]
        return [b.finish() for b in bs], list(bs[0].run_args)

    def dnq_row(self) -> int:
        """Words a row of the chunked DeltaNet's inputs (io["dnq"], a slice's: q and k of its
        key heads, v and z of its value heads, log2 decays and betas;
        docs/deltanet_chunk.md)."""
        spec = self.spec
        nk = self.nl * spec.lin_nk // spec.lin_heads
        return 2 * nk * spec.lin_dk + 2 * self.nl * spec.lin_dv + 2 * self.nl

    def dn_core_ok(self) -> bool:
        """The chunked DeltaNet core runs on this image's machine (kernels.dn_chunk.check: heads
        of D, SPLITQ, GALL, chunks of Config.DNCHUNK rows) over its prefill rows."""
        spec, cfg = self.spec, self.cfg
        try:
            DC.check(cfg.DNCHUNK, spec.lin_dk, cfg)
        except CompileError:
            return False
        return spec.lin_dk == spec.lin_dv and self.prefill_rows % cfg.DNCHUNK == 0

    def dns_bytes(self) -> int:
        """Bytes of the chunked core's scratch (io["dns"], a slice's: kernels.dn_chunk.Scratch)."""
        return DC.Scratch.nbytes(self.cfg.DNCHUNK, self.spec.lin_dk, self.cfg.D, self.cfg.WG)

    def compile_dn_run(self, li: int, slot: int = 0, split1: bool = False):
        """(programs, run_args): DeltaNet layer li's chunked core (qwen35_dn_run)
        over rows "row" .. "row" + "rows" - 1 of the image's dnq (run arguments: "rows" a
        multiple of Config.DNCHUNK, "row" + "rows" at most the prefill rows), o into the same
        rows of dno, the slice's states in their committed slot `slot` read at its start and
        written at its end."""
        from ..compiler import RunVar
        if "dns" not in self.io or self.keys[li][0] != LIN:
            raise ValueError(f"layer {li}: a DeltaNet layer of an image with the chunked core's "
                             "regions (Config.DNCHUNK, dn_core_ok)")
        row, rows = RunVar("row", self.prefill_rows), RunVar("rows", self.prefill_rows + 1)
        bs = [qwen35_dn_run.trace(self.cfg, s, {
            "m": self.descriptors(s, slot) if slot else self.descriptors(s), "li": li,
            "row": row, "rows": rows, "C": self.cfg.DNCHUNK, "split1": split1})
            for s in range(self.cfg.S)]
        return [b.finish() for b in bs], list(bs[0].run_args)

    def compile_expert_run(self, li: int, C: int | None = None):
        """(programs, run_args): moe.moe_expert_run of MoE layer li (expert-major,
        docs/offload.md 13.11), the chunk's rows and entries (rows x k) run arguments ("rows",
        "entries")."""
        if not self.em_ok() or self.cfg.S != 1 and self.pool is None:
            raise ValueError("an expert run needs the expert server's words (or a resident "
                             "pool's records, em), the image's prefill rows and one slice "
                             "(S > 1: a resident pool's, tensor-parallel)")
        err, wf = None, self.fmt.wformat
        if C is not None and not 1 <= C <= self.cfg.act_rows:
            raise ValueError(f"passes of {C} entries: 1 .. ACT_ROWS ({self.cfg.act_rows})")
        ex = MO.em_exact(self.cfg)                      # (exact pass sizes: GALL)
        imem = self.cfg.IMEM_WORDS // 8
        for C in [C] if C else MO.em_pass_order(self.cfg, wf, ex):  # (the best that fits TMEM)
            S = 0 if ex else MO.em_short_rows(self.cfg, wf, C)
            for pipe in (2, 1, 0):                      # (pipelined passes, where they fit IMEM)
                try:
                    bs = [qwen35_expert_run.trace(self.cfg, s, {
                        "m": self.descriptors(s), "li": li, "C": C, "S": S, "exact": ex,
                        "pipe": pipe}) for s in range(self.cfg.S)]
                except CompileError as e:
                    if "TMEM exhausted" not in str(e):
                        raise
                    err = err or e
                    continue
                progs = [b.finish() for b in bs]
                if pipe and max(map(len, progs)) > imem:
                    continue
                self.em_pass = (C, S) if not ex else (C, "exact")
                return progs, list(bs[0].run_args)
        raise err

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
        committed slot (DeltaNetParts; Spec.mtp), or with Image(batch=N) the sequence's: its
        states, windows and KV caches (a layer's kv_slots: every sequence's)."""
        if not 0 <= slot < self.slots:
            raise ValueError(f"slot {slot} of an image with {self.slots}")
        spec, cfg = self.spec, self.cfg
        D, d, H, K, n = cfg.D, spec.head_dim, spec.hidden, spec.conv_k, self.h_loc
        dk, dv, nl, C = spec.lin_dk, spec.lin_dv, self.nl, self.C
        gw = cfg.WG
        sl = dict(tp=K * self.CP, wwords=(K - 1) * self.CP, slot=slot,
                  sbytes=self.sbytes if self.slots > 1 else 0, sexact=self.sexact)

        def layer(li, it=None, jt=None):
            """Descriptors of layer `li` (static), or of element li of its run's unit at
            iteration `it` (a hardware-loop variable) and, jt = (loop variable, layers, bytes):
            at iteration jt of the unit's inner loop over its repeated part (lfm2.run_layers'
            nest). Its kind and formats group from self.keys (the MTP layer's: li =
            spec.layers)."""
            kind, g = self.keys[li]
            lay = self.layouts[g]
            off, loff = self._off(li, it), self._loff(li, it)   # (the states': Image.high)
            moff = self._moff(li, it)                           # (the mixers': Image.mix)
            if jt is not None:
                nb, lnb = jt[2][:2] if isinstance(jt[2], tuple) else (jt[2], jt[2])
                mnb = jt[2][2] if isinstance(jt[2], tuple) and len(jt[2]) > 2 else nb
                off, loff = off + Affine.of(jt[0]) * nb, loff + Affine.of(jt[0]) * lnb
                moff = moff + Affine.of(jt[0]) * mnb
            lofs, mf = lay.lofs[kind], lay.mf
            ns = SimpleNamespace(g_in=Tensor(off + lofs["g_in"], (H,), (1,)),
                                 g_post=Tensor(off + lofs["g_post"], (H,), (1,)),
                                 moe=spec.moe is not None, kind=kind)
            if ns.moe:
                da, sa = lofs["router"]
                ns.router = QTensor.weight(off + da, off + sa, self.er, H, "int8", D, grp=gw)
                ns.gbase = Tensor(off + lofs["gbase"], (1,), (1,))
            for name, (r, k) in self.mats[kind].items():
                da, sa = lofs[name]
                fm, o = mf[name], moff if name in MIXER_PARTS else off
                so = loff if name in lay.slow else o        # (Image.mix: a mixer's scale rows)
                setattr(ns, name, QTensor.weight(o + da, so + sa, r, k, fm, D, grp=gw))
            Cd = lay.gs * lay.dchunk
            fm = mf["wd"]
            parts = tuple(QTensor.weight(off + da, off + sa, n, Cd, fm, D, grp=gw)
                          for da, sa in lofs["wd"])
            ns.wd = QTensor(parts[0].data, parts[0].scale, (n, spec.ffn), parts[0].rs,
                            parts[0].srs, D, parts=parts, pw=Cd, wf=parts[0].wf, grp=parts[0].grp,
                            gs=lay.gs)
            if kind == LIN:
                ns.alog = Tensor(off + lofs["alog"], (nl,), (1,))
                ns.dtb = Tensor(off + lofs["dtb"], (nl,), (1,))
                ns.gn = Tensor(off + lofs["gn"], (dv,), (1,))
                og, R2 = self.og, self.RP
                if self.grouped:            # the parts of pair 0 (group 0)
                    fm = mf["wh"]
                    g0, gl = moff + lofs["groups"], loff + lofs["lgroups"]
                    (whd, whs), (wod, wos) = lay.pofs["wh"], lay.gofs["wout"]
                    ns.dn = DeltaNetParts(
                        nl, og, QTensor.weight(g0 + whd, g0 + whs, R2, H, fm, D, grp=gw),
                        Tensor(gl + lay.pofs["cv"], (self.CV0,), (1,)),
                        Tensor(gl + lay.pofs["state"], (2, dv, dk), (dv * dk, dk, 1),
                               esize=self.se),
                        QTensor.weight(g0 + wod, g0 + wos, H, og * dv, fm, D, grp=gw), H,
                        lay.GS, lay.PS,
                        Tensor(gl + lay.leb if self.mix else g0 + lay.gofs["eb"], (og // 2, 4),
                               (4, 1)), self.shared, **sl,
                        fold=self.dn_fold, lgs=lay.LGS, lps=lay.LPS, eb_low=self.mix)
                else:
                    # (a pair's taps and window 0: the decode loads them at once; window 1
                    # after them, DeltaNetParts.window)
                    ns.cv = Tensor(loff + lofs["cv"], (nl // 2, self.CV0), (self.CVW, 1))
                    ns.state = Tensor(loff + lofs["state"], (nl, dv, dk), (dv * dk, dk, 1),
                                      esize=self.se)
                    qk = dict(wqk=ns.wqk, qkcv=Tensor(loff + lofs["qkcv"], (self.QCV0,), (1,)),
                              rk=self.rk, qtp=K * self.QC) if self.qkonce else {}
                    ns.dn = DeltaNetParts(nl, og, ns.wh, ns.cv, ns.state, ns.wout, H,
                                          shared=self.shared, ahead=self.cv_ahead, **sl, **qk,
                                          fold=self.dn_fold)
            else:
                ns.qn = Tensor(off + lofs["qn"], (d,), (1,))
                ns.kn = Tensor(off + lofs["kn"], (d,), (1,))
                # each sequence's caches (kv_slots; Image(batch=N)): the program of one
                # sequence sees its slot's as its own (kv, kvs[0]); a batched step's rows name
                # theirs in kv_slots (batch35)
                ns.kv_slots = [KVDesc({sid + j * cfg.S: {k: loff + v for k, v in r.items()}
                                       for j, r in enumerate(heads)}, self.cap, d, D, cfg.S,
                                      sid, wg=gall_wg(cfg))
                               for heads in lofs.get("kvs", [lofs["kv"]])]
                ns.kv = ns.kv_slots[slot if self.batch > 1 else 0]
                ns.kvs = [ns.kv]
            return ns

        dev = None
        if self.offload is not None:
            L = self.offload
            dev = SimpleNamespace(mbox=L.mbox, served=L.served, answer=L.answer, dir=L.dir,
                                  tag=L.tag, fmt=self.fmt, hint_off=L.layers * L.E,
                                  slots=L.slot_bounds, ids=L.layers * L.E,
                                  scratch=self.io.get("moe_scratch"),
                                  need_off=2 * L.layers * L.E, em_base=L.slots[0][0],
                                  em_rec=MO.em_record(H, spec.moe.k, MO.em_qd(cfg)),
                                  em_rows=self.prefill_rows)
            # an expert run's experts: past the prefill rows' scratch (moe.em_scratch)
            dev.em_slots = MO.em_expert_slots(L, self.prefill_rows * dev.em_rec, dev.slots)
        elif self.pool is not None:     # every expert in DRAM (moe_ffn's resident path)
            em = {"em_base": self.io["em"],
                  "em_rec": MO.em_record(H, spec.moe.k, MO.em_qd(cfg)),
                  "em_rows": self.prefill_rows} if "em" in self.io else {"em_rows": 0}
            dev = MO.pool_dev(self.pool, self.fmt, scratch=self.io.get("moe_scratch"), **em,
                              **({"groups": self.moe_groups} if self.moe_groups else {}))
        mtp = {}
        if spec.mtp:
            mo, ff = self.mtpo, self.mtp_fc
            mtp = dict(hid=_tdesc(self.io["hid"], (self.rows, H)), nd=self.nd,
                       draft=_tdesc(self.io["draft"], (self.rows,)),
                       mtp=SimpleNamespace(
                           layer=layer(spec.layers),
                           ge=_tdesc(mo["ge"], (H,)), gh=_tdesc(mo["gh"], (H,)),
                           gm=_tdesc(mo["gm"], (H,)),
                           fc=QTensor.weight(*mo["fc"], H, 2 * H, ff, D, grp=gw),
                           dh=_qdesc(*mo["dh"], self.nd, H, D, "fp4", gw)),
                       mtpgen=self.lookup.get("mtpgen"))
        return SimpleNamespace(
            spec=spec, layer=layer, plan=self.plan, subs=self.subs, moe_dev=dev, **mtp,
            cv_pre=self.cv_pre,
            x=_tdesc(self.io["x"], (1, H)), cos=_tdesc(self.io["cos"], (spec.rope_dim // 2,)),
            sin=_tdesc(self.io["sin"], (spec.rope_dim // 2,)), g_final=_tdesc(self.io["gf"], (H,)),
            logits=_tdesc(self.io["logits"], (1, spec.vocab)),
            xr=_tdesc(self.io["x"], (self.rows, H)),
            cosr=_tdesc(self.io["cos"], (self.rows, spec.rope_dim // 2)),
            sinr=_tdesc(self.io["sin"], (self.rows, spec.rope_dim // 2)),
            logitsr=_tdesc(self.io["logits"], (self.rows, spec.vocab)),
            oscr=_tdesc(self.io["o"], (self.cfg.act_rows, self.nq_loc * spec.head_dim))
            if "o" in self.io else None,
            hs=_tdesc(self.io["hs"], (nl // 2, 4)),
            gr=_tdesc(self.io["gr"], (self.dnrows, 2 * nl)),
            on=_tdesc(self.io["on"], (self.dnrows, self.og * dv)),
            xbuf=_tdesc(self.io["xbuf"], (self.prefill_rows, H)) if self.prefill_rows else None,
            head=_qdesc(*self.head, self.v_loc, H, D, self.head_format, gw), v_loc=self.v_loc,
            **self._dn_rows_desc(), **_lookup_desc(self.lookup, spec, self.cap))

    def _dn_rows_desc(self) -> dict:
        """The chunked DeltaNet's regions as tensors (Config.DNCHUNK;
        docs/deltanet_chunk.md), each slice's own: dnq.q / .k [nk, Rmax, dk], .v / .z
        [nl, Rmax, dv], .lg / .b [nl, Rmax] (head-major: one load a head's chunk), dno [Rmax,
        nl dv] (row-major: one load a tile's rows)."""
        if "dnq" not in self.io:
            return {"dnq": None, "dno": None, "dns": None}
        spec, R, nl = self.spec, self.prefill_rows, self.nl
        dk, dv = spec.lin_dk, spec.lin_dv
        nk = nl * spec.lin_nk // spec.lin_heads
        a, parts = self.io["dnq"], {}
        for name, shape in (("q", (nk, R, dk)), ("k", (nk, R, dk)), ("v", (nl, R, dv)),
                            ("z", (nl, R, dv)), ("lg", (nl, R)), ("b", (nl, R))):
            parts[name] = _tdesc(a, shape)
            a += 4 * int(np.prod(shape))
        dns = DC.Scratch(self.io["dns"], self.cfg.DNCHUNK, dk, self.cfg.D, self.cfg.WG) \
            if "dns" in self.io else None
        return {"dnq": SimpleNamespace(nk=nk, nl=nl, rows=R, **parts),
                "dno": _tdesc(self.io["dno"], (R, nl * dv)), "dns": dns}


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


def _silu_dn(dn, x):
    """SiLU of the convolved channels (q, k, v) or of z; on a folded image (Image.dn_fold) their
    inputs come -log2(e) times (the taps', z's rows) and the result is -log2(e) silu (silu_l2:
    no MUL)."""
    return silu_l2(x) if dn.fold else silu(x)


def _l2n(dn, x, scale: float = 1.0):
    """l2norm_rows of q (scale dk^-1/2) or k rows. Folded, x comes -log2(e) times (_silu_dn):
    eps log2(e)^2 times keeps the norm, negated, and q's scale goes to the output norm's eps
    (_dn_eps), so the state holds log2(e) S."""
    return l2norm_rows(x, 1.0, 1e-6 * ol.LOG2E ** 2) if dn.fold else l2norm_rows(x, scale)


def _l2n_qk(dn, u, out, dk: int) -> None:
    """_l2n of key heads' q and k into out (u, out [n, 2 dk]: a head's q, then its k), as one
    chain over their 2n rows of dk: one sum of squares, eps and rsqrt for both, unfolded q's
    dk^-1/2 on its rows' factors (as l2norm_rows takes it). Per element _l2n's operations, so
    out holds the words of _l2n's q and k; a view whose rows are not contiguous takes the two."""
    n = u.rows
    if not (u.contiguous and out.contiguous):
        out[:, 0:dk].set(_l2n(dn, u[:, 0:dk], dk ** -0.5))
        out[:, dk:2 * dk].set(_l2n(dn, u[:, dk:2 * dk]))
        return
    x = u.reshape(2 * n, dk)                            # rows q0, k0, q1, k1, ...
    r = ol.rsqrt(ol.sum(x * x, axis=1) + (1e-6 * ol.LOG2E ** 2 if dn.fold else 1e-6))
    if not dn.fold:
        rq = r.reshape(n, 2)[:, 0:1]                    # the q rows' factors
        rq.set(rq * dk ** -0.5)
    out.reshape(2 * n, dk).set(x * r[:, None])


def _dn_eps(dn, spec: Spec) -> float:
    """The gated output norm's eps. Folded, o comes -log2(e) dk^1/2 times (v's -log2(e), q's
    sign and scale: _l2n), so eps log2(e)^2 dk times keeps the norm, negated; silu(z)'s
    -log2(e) and the gain over log2(e) (Image) give back the product."""
    return spec.eps * ol.LOG2E ** 2 * spec.lin_dk if dn.fold else spec.eps


def _dn_gain(gn, dv: int):
    """The gated output norm's gain times sqrt(dv): the norm's 1 / dv out of its statistic
    (_dn_norm), into the gate's factor (_dn_gz)."""
    return gn * dv ** 0.5


def _dn_gz(dn, z, gs):
    """The gated output norm's gate, silu(z) times its gain gs (_dn_gain), off the norm's
    path: z's rows come long before o's."""
    return _silu_dn(dn, z) * gs[None, :]


def _dn_norm(o, eo: float):
    """o / sqrt(|o|^2 + dv eo) per row of o [n, dv]: the gated output norm's statistic without
    its 1 / dv (sqrt(dv) in _dn_gain), so that times _dn_gz's gate it is rmsnorm(o, gn, eo)
    silu(z) in two VOPs after the rsqrt, not four (the products in another order: within
    fp32 rounding)."""
    return o * ol.rsqrt(ol.sum(o * o, axis=1) + o.cols * eo)[:, None]



def _dn_conv(dn, X, taps, a: int, bw: int, p0: int, R: int, K: int, out=None, act=True):
    """Block a (bw channels) of the rows in X ([K - 1 + R, channels]: the window's K - 1 rows,
    then the R rows' projections, positions p0 ..) convolved with its taps (rows (block, tap)),
    as _deltanet's conv (rows before position K - 1 without the taps of positions < 0), and
    SiLU (act) -> [R, bw] (into `out`): _deltanet_rows' and _deltanet_prep's (the same
    words)."""
    U = ol.empty([R, bw]) if out is None else out
    mac = ol.has_mac()                                  # VOP MAC: the same words, a pass a tap
    full = max(0, K - 1 - p0)                           # rows before it lack positions < 0
    for r0, r1 in [(r, r + 1) for r in range(min(full, R))] + ([(full, R)] if full < R else []):
        t = min(K - 1, p0 + r0)
        cur = X[K - 1 + r0:K - 1 + r1, a * bw:(a + 1) * bw]
        if t == 0 or mac:
            U[r0:r1, :].set(cur * taps[a * K + K - 1, :][None, :])
            for j in range(1, t + 1):                   # (MAC's columns: 256 at most)
                for c0 in range(0, bw, I.MAC_MAX_COLS):
                    c1 = min(bw, c0 + I.MAC_MAX_COLS)
                    U[r0:r1, c0:c1].mac(X[K - 1 + r0 - j:K - 1 + r1 - j,
                                          a * bw + c0:a * bw + c1],
                                        taps[a * K + K - 1 - j, c0:c1][None, :])
            continue
        u = cur * taps[a * K + K - 1, :][None, :]
        for j in range(1, t):
            u = u + X[K - 1 + r0 - j:K - 1 + r1 - j, a * bw:(a + 1) * bw] * \
                taps[a * K + K - 1 - j, :][None, :]
        U[r0:r1, :].set(u + X[K - 1 + r0 - t:K - 1 + r1 - t, a * bw:(a + 1) * bw] *
                        taps[a * K + K - 1 - t, :][None, :])
        del u
    if act:
        U.set(_silu_dn(dn, U))
    return U

def _key_heads(xs, dn, pos, spec: Spec):
    """q | k of the slice's key heads (Image.qkonce), [nkl, 2 dk]: their projection, the
    convolution over their window (stored back shifted by one row, as a pair's), SiLU, then q
    L2-normed and scaled by dk^-1/2 and k L2-normed: per element the operations a pair's conv
    and qk run on the q and k rows a value head repeats, so the values are theirs bit for
    bit."""
    K, dk = spec.conv_k, spec.lin_dk
    nk, QC, TP = dn.nkl, dn.QC, dn.qtp
    past = _past(K, pos)
    pre = ol.dot(xs, dn.wqk)                            # [1, QC]
    CV = ol.load(dn.qk_cv())                            # taps (rows (key head, tap)), window
    win = dn.qk_window()
    ol.store(win[(K - 2) * QC:(K - 1) * QC], pre[0, :])     # the window's last row: this position
    ol.store(win[0:(K - 2) * QC], CV[TP + QC:TP + (K - 1) * QC])    # its other rows, one up
    taps = CV[0:TP].reshape(nk * K, 2 * dk)

    def tap(i):
        return taps.row_stride_view(i, nk, K) if nk > 1 else taps[i:i + 1, :]
    terms = [(pre.reshape(nk, 2 * dk), K - 1)] + [
        (CV[TP + (K - 1 - j) * QC:TP + (K - j) * QC].reshape(nk, 2 * dk), K - 1 - j)
        for j in range(1, past + 1)]
    u = terms[0][0] * tap(terms[0][1])
    for xx, i in terms[1:]:
        u = u + xx * tap(i)
    del pre, CV
    u = _silu_dn(dn, u)
    QK = ol.empty([nk * 2 * dk]).reshape(nk, 2 * dk)
    _l2n_qk(dn, u, QK, dk)
    return QK


def _key_heads_parts(xs, dn, pos, spec: Spec, defer: bool = False):
    """_key_heads by parts, so the first pairs' steps need not wait for every key head's
    (_deltanet_dstep): returns (QK, mm, vec, store). mm(h0, h1) the projection of key heads h0
    .. h1 - 1, vec(h0, h1) their convolution, SiLU and L2 norms into QK's rows, store() the
    window's last row (their projections), for the caller to place after the layer's steps
    (a store waiting on an MM holds the DMA's later loads and steps). Per element the
    operations of _key_heads, so QK is its words."""
    K, dk = spec.conv_k, spec.lin_dk
    nk, QC, TP, W2 = dn.nkl, dn.QC, dn.qtp, 2 * spec.lin_dk
    past = _past(K, pos)
    QK = ol.empty([nk * W2]).reshape(nk, W2)
    CV = ol.load(dn.qk_cv())                            # taps (rows (key head, tap)), window
    win = dn.qk_window()
    if not defer:
        ol.store(win[0:(K - 2) * QC], CV[TP + QC:TP + (K - 1) * QC])    # its other rows, one up
    taps = CV[0:TP].reshape(nk * K, W2)
    pre = {}

    def mm(h0, h1):
        pre[h0, h1] = ol.dot(xs, dn.wqk[h0 * W2:h1 * W2, :])        # [1, (h1 - h0) 2 dk]

    def vec(h0, h1):
        n, x0 = h1 - h0, h0 * W2

        def tap(i):
            return taps.row_stride_view(h0 * K + i, n, K) if n > 1 else \
                taps[h0 * K + i:h0 * K + i + 1, :]
        terms = [(pre[h0, h1].reshape(n, W2), K - 1)] + [
            (CV[TP + (K - 1 - j) * QC + x0:TP + (K - 1 - j) * QC + x0 + n * W2]
             .reshape(n, W2), K - 1 - j) for j in range(1, past + 1)]
        u = terms[0][0] * tap(terms[0][1])
        for xx, i in terms[1:]:
            u = u + xx * tap(i)
        u = _silu_dn(dn, u)
        _l2n_qk(dn, u, QK[h0:h1, :], dk)

    def store():
        if defer:
            ol.store(win[0:(K - 2) * QC], CV[TP + QC:TP + (K - 1) * QC])
        for (h0, h1), t in pre.items():     # the window's last row: this position
            ol.store(win[(K - 2) * QC + h0 * W2:(K - 2) * QC + h1 * W2], t[0, :])
    return QK, mm, vec, store


def _out_proj(dn, ON, g, c0: int, c1: int, dv: int, y, first: bool = False) -> None:
    """y += ON . out_proj of head group g's heads c0 .. c1 - 1 (ON [R, (c1 - c0) dv]). At S > 1
    on the slice's own H / S output columns, as the MLP's down projection: every slice's ON
    gathered, then one MM per slice's part (each its own QACT range: the words one slice's MM
    quantizes), the image's block g holding the slice's rows of every slice's group g
    (Image.build); y [R, H / S], not an all-reduce of [R, H]. `first`: y not yet written, the
    first MM writes it (y = ...: no FILL of y before)."""
    S = ol.num_programs()
    if S == 1:
        ol.dot(ON, dn.wout(g)[:, c0 * dv:c1 * dv], acc=None if first else y,
               out=y if first else None)
        return
    if getattr(dn, "sexact", False) and dn.og == 4 and c1 - c0 == 2:
        _last_pairs(dn, ON, g, c0, y, first)            # (the last group's pairs: s_exact)
        return
    n, hl = ON.cols, dn.H // S
    a = ol.all_gather(ON)                               # [R, S n]: every slice's heads
    for t in ol.static_range(S):
        w = first and t == 0
        ol.dot(a[:, t * n:(t + 1) * n], dn.wout(g)[t * hl:(t + 1) * hl, c0 * dv:c1 * dv],
               acc=None if w else y, out=y if w else None)


def _last_pairs(dn, ON, g, c0: int, y, first: bool) -> None:
    """_out_proj of the last group's pairs, S-exact (s_exact) at S > 1: pair 0's gather puts
    each slice's pair in the first half of its 2n words of a buffer (n = ON's 2 dv), pair 1's
    beside it, so slice t's last group is one tile; after pair 1 the MMs in S = 1's order and
    QACT ranges: slices 0 .. S - 2's whole groups (S = 1's whole groups), then slice S - 1's
    two pairs (S = 1's last group, in pairs). The buffer (and `first`) wait on y between the
    two calls."""
    b = current()
    S = ol.num_programs()
    ON = b.materialize(ON)
    b.check_live(ON)
    n, hl = ON.cols, dn.H // S
    rs = ON.rs if len(ON.shape) == 2 else n
    if c0 == 0:
        buf = ol.empty([ON.rows, S * 2 * n])
        y.last_pairs = (buf, first)
    else:
        buf, first = y.last_pairs
        del y.last_pairs
    b.emit(I.gather(ON.base, buf.base + (n if c0 else 0), ON.rows, n, rs, buf.rs, 2 * n,
                    comment="all-gather (the last group's pair)"))
    b.bump_version(buf.buf)
    if not c0:
        return
    w = dn.wout(g)
    for t in range(S):
        for a0, a1 in [(0, 2 * n)] if t < S - 1 else [(0, n), (n, 2 * n)]:
            ol.dot(buf[:, 2 * n * t + a0:2 * n * t + a1], w[t * hl:(t + 1) * hl, a0:a1],
                   acc=None if first else y, out=y if first else None)
            first = False


def _residual(x, y):
    """x + out_proj's y (_out_proj): at S > 1 the slice's columns, then all-gathered. At S = 1
    the ADD writes x in place (the caller's x.set(...) then emits nothing): x + y's temp would
    take x's words again beside everything the layer holds (cv_ahead's taps and windows)."""
    if ol.num_programs() == 1:
        return x.add(y)
    sid, hl = ol.program_id(), y.cols
    return ol.all_gather(x[:, sid * hl:(sid + 1) * hl] + y)


def _last_group(dn, ON, g, y) -> None:
    """_last_pairs' two calls at once, ON [R, 4 dv] the group's pairs: one gather, the same MMs."""
    S, n, hl = ol.num_programs(), ON.cols // 2, dn.H // ol.num_programs()
    a, w = ol.all_gather(ON), dn.wout(g)
    for t in range(S):
        for a0, a1 in [(0, 2 * n)] if t < S - 1 else [(0, n), (n, 2 * n)]:
            ol.dot(a[:, 2 * n * t + a0:2 * n * t + a1], w[t * hl:(t + 1) * hl, a0:a1], acc=y)


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
    eo = _dn_eps(dn, spec)              # (the output norm's)
    nl, C = dn.nl, 2 * dk + dv
    NP, og = nl // 2, dn.og
    sh, qo = dn.shared, dn.qkonce       # one key head's q and k for the pair; or a layer's
    nb, bw = (4, dk) if sh else (2, dv) if qo else (2, C)   # the pair's blocks of channels
    CP, RP = nb * bw, nb * bw + 2 * dv                  # its convolved channels, its rows
    TP = K * CP                                         # taps words of a pair (then its ring)
    xs = ol.quantize(rmsnorm(x, ol.load_home(lw.g_in, "g_in"), eps))
    ab = ol.dot(xs, lw.wab)                             # [1, 2nl]: a, then b, of each head
    decay, beta = gates(ab[0, 0:nl], ab[0, nl:2 * nl], ol.load(lw.alog), ol.load(lw.dtb),
                        fold=dn.fold)
    eb = ol.empty([4 * NP]).reshape(NP, 4)              # per pair: decays of a, b; betas
    eb[:, 0:2].set(decay.reshape(NP, 2))
    eb[:, 2:4].set(beta.reshape(NP, 2))
    dn.store_gates(eb, hs)
    del decay, beta, eb
    past = _past(K, pos)

    def pairs(n, w):
        return [ol.empty([2 * w]).reshape(2, w) for _ in range(n)]

    QKH = _key_heads(xs, dn, pos, spec) if qo else None     # the key heads' q | k (qkonce)
    St = [ol.empty([dv * dk]).reshape(dv, dk) for _ in range(2)]        # heads a, b of a pair
    P = [ol.empty([1, RP]) for _ in range(2)]        # q k v of a, of b, then z of a, of b
    CV = ol.empty([TP + (K - 1) * CP])     # taps (rows (block, tap)), the window (_past)
    nq = 1 if sh else 2                                 # q and k rows: shared, or per head
    U = [ol.empty([CP]).reshape(nb, bw) for _ in range(2)]
    Qn = None if qo else [ol.empty([nq * dk]).reshape(nq, dk) for _ in range(2)]
    Kn = None if qo else [ol.empty([nq * dk]).reshape(nq, dk) for _ in range(2)]
    GZ = pairs(2, dv)
    EB = [ol.empty([4]) for _ in range(2)]
    O = ol.empty([2 * dv]).reshape(2, dv)
    ON = ol.empty([og * dv]).reshape(og, dv)            # normed, gated o of og heads
    w = ol.empty([dv])
    y = ol.zeros([1, spec.hidden // ol.num_programs()])    # (_out_proj)
    gs = _dn_gain(ol.load(lw.gn), dv)
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
        GZ[t].set(_dn_gz(dn, P[t][0, CP:RP].reshape(2, dv), gs))

    def qk(t, j=None):
        """SiLU of the convolved channels (in place), then the L2-normed q and k (shared: once,
        with head a's blocks)."""
        a, n = hb(j)
        u = U[t][a:a + n, :]
        u.set(_silu_dn(dn, u))
        if qo:                                          # (qkonce: the key heads' in QKH)
            return
        if not sh:
            Qn[t][a:a + n, :].set(_l2n(dn, u[:, 0:dk], dk ** -0.5))
            Kn[t][a:a + n, :].set(_l2n(dn, u[:, dk:2 * dk]))
        elif a == 0:
            Qn[t].set(_l2n(dn, U[t][0:1, :], dk ** -0.5))
            Kn[t].set(_l2n(dn, U[t][1:2, :]))

    def kqv(t, j, p=None):
        """Head j's normed k and q, and its v, of the pair in buffers t (qkonce: pair p's)."""
        if qo:
            h = dn.kh(p, j)
            return QKH[h, dk:2 * dk], QKH[h, 0:dk], U[t][j, :]
        r = 0 if sh else j
        return Kn[t][r, :], Qn[t][r, :], U[t][2 + j, :] if sh else U[t][j, 2 * dk:C]

    def post(t):
        """The gated RMSNorm of the pair in O (gates GZ[t]) into its rows of ON."""
        r0 = 2 * t if og == 4 else 0
        ON[r0:r0 + 2, :].set(_dn_norm(O, eo) * GZ[t])

    def flush(g, k=None):
        """out_proj of head group g (k: only its k-th pair)."""
        c0, c1 = (0, og) if k is None else (2 * k, 2 * k + 2)
        _out_proj(dn, ON[c0:c1, :].reshape(1, (c1 - c0) * dv), g, c0, c1, dv, y)

    def rdot1(S, t, j, p):
        """kv = S k of head j of pair p in buffers t, into w."""
        w.set(S @ kqv(t, j, p)[0])

    def update(S, t, j, p):
        """d = beta (v - e^g kv), then S = e^g S + d k^T (OUTER, in place)."""
        k, _, v = kqv(t, j, p)
        w.set((v - w * EB[t][j:j + 1]) * EB[t][2 + j:3 + j])
        for r, e in halves:
            ol.outer(w[r:e], k, acc=S[r:e, :], decay=EB[t][j:j + 1])

    def rdot2(S, t, j, p):
        """o = S q of head j, into O[j]."""
        for r, e in halves:
            O[j, r:e].set(S[r:e, :] @ kqv(t, j, p)[1])

    def _pair_segment(p, t, last1, last2, g=None):
        """Pair p (buffers t = p % 2); last1: no pair p+1, last2: no pair p+2; g: the head
        group pair p-1 completes (to multiply by out_proj)."""
        first = isinstance(p, int) and p == 0
        if not last2 and not first:
            project(p + 2, t)
        rdot1(St[0], t, 0, p)
        state_in(p, 1, St[1])
        update(St[0], t, 0, p)
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
        rdot2(St[0], t, 0, p)
        rdot1(St[1], t, 1, p)
        if not last1:
            state_in(p + 1, 0, St[0])
        update(St[1], t, 1, p)
        state_out(p, 1, St[1])
        if not last1:
            conv(p + 1, 1 - t)
            qk(1 - t)
            gatez(1 - t)
        rdot2(St[1], t, 1, p)
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
    # too few for the attention's run-time ones); with qkonce never (a pair's key head)
    n_it = 0 if isinstance(pos, RunPos) and not dn.grouped or qo else max(0, (NP - 3) // 2)
    if n_it:
        for i in ol.range(n_it):
            _pair_segment(2 * i + 1, 1, False, False, None if og == 4 else 2 * i)
            _pair_segment(2 * i + 2, 0, False, False, i if og == 4 else 2 * i + 1)
    for p in range(1 + 2 * n_it, NP):
        _pair_segment(p, p % 2, p + 1 >= NP, p + 2 >= NP, group(p - 1))
    post((NP - 1) % 2)
    flush(group(NP - 1), 1 if og == 4 and NP > 1 else None)
    return _residual(x, y)


def _bursts(NP: int, beside: bool = False) -> list:
    """The pairs of _deltanet_dstep's bursts: pair 0 alone (its steps start once its own prep
    is done), then two pairs a burst, the last one alone when they do not come out even (its
    post, out_proj and the residual follow the layer's last step): 1, 2, 2, 1 at NP 6 (the
    27B's 12 heads a slice). Co-simulated on the WG targets' slice (docs/se_engine.md): 2, 2, 2;
    1, 2, 3 and 1, 3, 2 are slower. beside (the streams on an engine of their own, Config.SE_OWN:
    the VPU need not take turns with them): a burst a pair, -1.95% of the 4-layer unit at SE_R 1
    against 1, 2, 2, 1 (docs/se_engine.md)."""
    if beside:
        return [[q] for q in range(NP)]
    sizes = [1] + [2] * ((NP - 1) // 2) + [1] * ((NP - 1) % 2)
    out, q = [], 0
    for n in sizes[:NP]:
        out.append(list(range(q, q + n)))
        q += n
    return out


def _deltanet_dstep(x, lw, pos: int, spec: Spec, hs, pipe=None):
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

    The per-pair buffers come in two sets by parity, o included. Ahead (DeltaNetParts.ahead,
    the pairs unrolled): every pair's taps and window loaded after pair 0's, the gates kept in
    TMEM, a projection buffer per pair and the windows stored after the last DSTEP, so the
    DMA's chain between the DSTEPs is the DSTEPs alone (and a projection waits for no pair's
    prep); with q and k once a layer, key head 0's first (_key_heads_parts).

    Ahead with q and k once a layer (the WG targets' decode), by bursts of pairs (_bursts): a DSTEP
    holds the VPU (SE's front, otpu_vpu) for the whole stream, so the VPU alternates between
    VOPs and streams, and each switch costs a chain's latency. A burst's rows share a tile (its
    pairs' convolved channels and z rows, their DSTEPs' outputs), so its SiLUs are one VOP chain
    and its posts' norms another, and its pairs' convolutions go a step at a time across them.
    The MMs go a burst ahead (burst 0's key heads' and pairs' projections first, burst b + 1's
    after burst b's prep: MMs in flight hold the sequencer's window slots), the gates', the
    norm's and the key heads' small loads first, the taps and windows in two loads (burst 0's,
    then the others' as one block). Per burst the VPU does its pairs' convolutions, its new key
    heads, the SiLU and the previous burst's posts in one go, then the burst's DSTEPs run back
    to back (burst 0's z rows' SiLU after its DSTEPs: its steps need its channels, not z's MM,
    the last of its projections); the head groups' out_proj follow their posts (two ON buffers
    by the group's parity), the last group pair by pair, as above.

    pipe (qwen35_step's _CvPipe; ahead without bursts): every pair's taps and window are already
    in the pipe's tile (loaded beside the MLP before), and the windows' stores go to the pipe,
    for this layer's MLP."""
    eps, K = spec.eps, spec.conv_k
    dk, dv = spec.lin_dk, spec.lin_dv
    dn = lw.dn
    eo = _dn_eps(dn, spec)              # (the output norm's)
    nl, C = dn.nl, 2 * dk + dv
    NP, og = nl // 2, dn.og
    sh, qo = dn.shared, dn.qkonce       # one key head's q and k for the pair; or a layer's
    nb, bw = (4, dk) if sh else (2, dv) if qo else (2, C)   # the pair's blocks of channels
    CP, RP = nb * bw, nb * bw + 2 * dv                  # its convolved channels, its rows
    TP = K * CP
    # segments 1 .. NP-3 have every part: loop them, but at a run-time position only in a
    # group-major layer (array-major, the loop's address registers, one per part, would leave
    # too few for the attention's run-time ones); with qkonce never (a pair's key head)
    n_it = 0 if isinstance(pos, RunPos) and not dn.grouped or qo else max(0, (NP - 3) // 2)
    ahead = dn.ahead and not n_it and not dn.grouped and not dn.slot
    bursts = _bursts(NP, ol.streams_beside()) if ahead and qo else None   # (the WG targets' decode)
    if pipe is not None and not (ahead and not qo):
        raise CompileError("the taps and windows a layer ahead (_CvPipe) need the pairs' ahead, "
                           "q and k per pair")
    xs = ol.quantize(rmsnorm(x, ol.load_home(lw.g_in, "g_in"), eps))
    ab = ol.dot(xs, lw.wab)
    if not bursts:
        decay, beta = gates(ab[0, 0:nl], ab[0, nl:2 * nl], ol.load(lw.alog), ol.load(lw.dtb),
                            fold=dn.fold)
        eb = ol.empty([4 * NP]).reshape(NP, 4)
        eb[:, 0:2].set(decay.reshape(NP, 2))
        eb[:, 2:4].set(beta.reshape(NP, 2))
        if not ahead:
            dn.store_gates(eb, hs)
            del eb
        del decay, beta
    else:                   # (bursts: the gates' and the norm's vectors first on the DMA)
        G = ol.empty([2 * nl])          # every pair's decay, then beta (gates' out: no copies)
        alog, dtb, gs = ol.load(lw.alog), ol.load(lw.dtb), _dn_gain(ol.load(lw.gn), dv)
    past = _past(K, pos)

    def pairs(n, w):
        return [ol.empty([2 * w]).reshape(2, w) for _ in range(n)]

    if qo and ahead:    # the key heads' q | k, key head 0's first (pair 0 needs only it)
        QKH, kmm, kvec, kstore = _key_heads_parts(xs, dn, pos, spec, defer=bool(bursts))
        if not bursts:
            kmm(0, 1)
    elif qo:
        QKH = _key_heads(xs, dn, pos, spec)
    P = [ol.empty([1, CP if bursts else RP])            # (ahead: a pair's; bursts: z in GZ)
         for _ in range(NP if ahead else 2)]
    CW = TP + (K - 1) * CP                              # a pair's taps and window
    CVA = ol.empty([NP * CW]) if bursts else None       # (bursts: every pair's in one tile)
    CV = [CVA[q * CW:(q + 1) * CW] for q in range(NP)] if bursts else \
        [pipe.cv(q, CW) for q in range(NP)] if pipe is not None else \
        [ol.empty([CW]) for _ in range(NP if ahead else 1)]
    nq = 1 if sh else 2                                 # q | k rows: shared, or per head
    if bursts:
        # a burst's rows in one tile each, its pairs' side by side, so that its SiLUs and its
        # norms are one VOP chain: UZ its convolved channels' rows (nb of bw = dv a pair), then
        # its z rows (2 of dv a pair, GZ: in place), Ob its DSTEPs' outputs (2 of dv a pair)
        at = {q: (b, i) for b, ps in enumerate(bursts) for i, q in enumerate(ps)}
        UZ = [ol.empty([len(ps) * (CP + 2 * dv)]).reshape(len(ps) * (nb + 2), dv)
              for ps in bursts]
        Ob = [ol.empty([len(ps) * 2 * dv]).reshape(2 * len(ps), dv) for ps in bursts]
        U = [UZ[b][i * nb:(i + 1) * nb, :] for b, i in (at[q] for q in range(NP))]
        GZ = [UZ[b][len(bursts[b]) * nb + 2 * i:len(bursts[b]) * nb + 2 * i + 2, :]
              for b, i in (at[q] for q in range(NP))]
        O = [Ob[b][2 * i:2 * i + 2, :] for b, i in (at[q] for q in range(NP))]
        QK = None
    else:
        U = [ol.empty([CP]).reshape(nb, bw) for _ in range(2)]
        GZ = pairs(2, dv)
        QK = None if qo else [ol.empty([nq * 2 * dk]).reshape(nq, 2 * dk) for _ in range(2)]
        O = pairs(2, dv)
    EB = None if bursts else [eb[p, :] for p in range(NP)] if ahead else \
        [ol.empty([4]) for _ in range(2)]
    ONs = [ol.empty([og * dv]).reshape(og, dv) for _ in range(2 if bursts else 1)]
    ON = ONs[0]
    if not bursts:
        y = ol.zeros([1, spec.hidden // ol.num_programs()])    # (_out_proj)
        gs = _dn_gain(ol.load(lw.gn), dv)

    def hb(j):
        """Head j's blocks of the pair's channels (None: all): (first, count); shared, head a
        takes q, k and its v, head b its v."""
        if j is None:
            return 0, nb
        return ((0, 3), (3, 1))[j] if sh else (j, 1)

    def bufs(p, t):
        """Pair p's projection rows and taps / window (buffers t; ahead: its own)."""
        return (P[p], CV[p]) if ahead else (P[t], CV[0])

    def project(p, t, split=False):
        ha = hb(0)[1] * bw                              # head a's channels first
        cuts = ((0, ha), (ha, CP), (CP, RP)) if split else ((0, CP), (CP, RP))
        for c0, c1 in cuts:                             # (bursts: z into GZ's rows)
            ol.dot(xs, dn.wh(p)[c0:c1, :], out=GZ[p].reshape(1, 2 * dv) if bursts and c0 == CP
                   else bufs(p, t)[0][:, c0:c1])

    def fetch_cv(p):
        if pipe is not None:                            # (in the pipe's tile: _CvPipe)
            return
        cv = bufs(p, 0)[1]
        if dn.slot:                                     # (slot 1's window: two loads)
            ol.load(dn.cv(p)[0:TP], out=cv[0:TP])
            ol.load(dn.window(p), out=cv[TP:TP + (K - 1) * CP])
        else:
            ol.load(dn.cv(p), out=cv)

    def fetch_eb(p, t):
        if not ahead:
            ol.load(dn.gates(p, hs), out=EB[t])

    def store_window(p, t, a, n):
        """Pair p's window: its last row (this position: blocks a .. a + n - 1), then (the
        last blocks) its other rows, one up (_past)."""
        Pt, CVt = bufs(p, t)
        win = dn.window(p)
        s0 = (K - 2) * CP + a * bw
        ol.store(win[s0:s0 + n * bw], Pt[0, a * bw:(a + n) * bw])
        if a + n == nb:
            ol.store(win[0:(K - 2) * CP], CVt[TP + CP:TP + (K - 1) * CP])

    def conv(p, t, j=None):
        a, n = hb(j)
        if not ahead:                                   # (ahead: after the last DSTEP)
            store_window(p, t, a, n)
        convs([(p, t, a, n)])

    def convs(jobs):
        """The convolutions of jobs (pair, buffers, first block, blocks) into U, a step at a
        time across them (a burst's pairs: their chains side by side)."""
        J = []
        for p, t, a, n in jobs:
            Pt, CV_ = bufs(p, t)
            taps = CV_[0:TP].reshape(nb * K, bw)

            def tap(i, a=a, n=n, taps=taps):
                return taps.row_stride_view(a * K + i, n, K) if n > 1 else \
                    taps[a * K + i:a * K + i + 1, :]
            # position p - j sits in window row K - 1 - j and is weighed by tap K - 1 - j
            J.append(([(Pt[0, a * bw:(a + n) * bw].reshape(n, bw), tap(K - 1))] + [
                (CV_[TP + (K - 1 - j) * CP + a * bw:TP + (K - 1 - j) * CP + (a + n) * bw]
                 .reshape(n, bw), tap(K - 1 - j)) for j in range(1, past + 1)],
                U[t][a:a + n, :]))
        if not past:
            for terms, out in J:
                out.set(terms[0][0] * terms[0][1])
            return
        us = [terms[0][0] * terms[0][1] for terms, _ in J]
        for k in range(1, past):            # (each term's MUL freed by its ADD, as one chain)
            ms = [terms[k][0] * terms[k][1] for terms, _ in J]
            for i in range(len(J)):
                us[i] = us[i] + ms.pop(0)
        ms = [terms[past][0] * terms[past][1] for terms, _ in J]
        for (_, out), u in zip(J, us):
            out.set(u + ms.pop(0))

    def gatez(p, t):
        GZ[t].set(_dn_gz(dn, bufs(p, t)[0][0, CP:RP].reshape(2, dv), gs))

    def qk(t, j=None):
        """SiLU of the convolved channels (in place), then the L2-normed q and k into QK[t]
        (shared: once, with head a's blocks; qkonce: none, the key heads' are QKH)."""
        a, n = hb(j)
        u = U[t][a:a + n, :]
        u.set(_silu_dn(dn, u))
        if qo:
            return
        if not sh:
            QK[t][a:a + n, 0:dk].set(_l2n(dn, u[:, 0:dk], dk ** -0.5))
            QK[t][a:a + n, dk:2 * dk].set(_l2n(dn, u[:, dk:2 * dk]))
        elif a == 0:
            QK[t][:, 0:dk].set(_l2n(dn, U[t][0:1, :], dk ** -0.5))
            QK[t][:, dk:2 * dk].set(_l2n(dn, U[t][1:2, :]))

    def prep(p, t):
        conv(p, t)
        qk(t)
        gatez(p, t)

    def posts(b):
        """Burst b's posts: its rows' norm at once, then each pair's into its group's ON (by
        the group's parity), and the out_proj flushes they complete."""
        r = _dn_norm(Ob[b], eo)
        for i, q in enumerate(bursts[b]):
            r0 = 2 * (q % 2) if og == 4 else 0
            ONs[(q // dn.gp) % 2][r0:r0 + 2, :].set(r[2 * i:2 * i + 2, :] * GZ[q])
            after_post(q)

    def post(t):
        r0 = 2 * t if og == 4 else 0
        ON[r0:r0 + 2, :].set(_dn_norm(O[t], eo) * GZ[t])

    yw = [bool(bursts)]                 # (bursts: y not filled, the first flush writes it)

    def flush(g, k=None):
        c0, c1 = (0, og) if k is None else (2 * k, 2 * k + 2)
        on = ONs[g % 2] if bursts else ON
        _out_proj(dn, on[c0:c1, :].reshape(1, (c1 - c0) * dv), g, c0, c1, dv, y, yw[0])
        yw[0] = False

    def dstep(p, t, j):
        ol.deltanet_step(dn.state(p, j), QKH[dn.kh(p, j), :] if qo else
                         QK[t][0 if sh else j, :],
                         U[t][2 + j, :] if sh else U[t][j, :] if qo else U[t][j, 2 * dk:C],
                         EB[p if ahead else t][j:j + 1], EB[p if ahead else t][2 + j:3 + j],
                         O[t][j, :],
                         zero=not isinstance(pos, RunPos) and pos == 0)

    def dstep_sh(p, t):
        """Pair p's two heads on their key head's q | k (shared), in one DSTEP with DSTEP2
        (deltanet_step2: else dstep(p, t, 0), dstep(p, t, 1), the same words): one stream's
        setup, fill and write drain for the two."""
        e = EB[p if ahead else t]
        ol.deltanet_step2(dn.states(p), QK[t][0, :], U[t][2:4, :].reshape(1, 2 * dv)[0, :],
                          e[0:2], e[2:4], O[t].reshape(1, 2 * dv)[0, :],
                          zero=not isinstance(pos, RunPos) and pos == 0)

    def dstep2(p):
        """Pair p's two heads (bursts: its own buffers; qkonce), in one DSTEP with DSTEP2
        (deltanet_step2: else a DSTEP a head, as dstep); their decay and beta from G (beta nl
        words after decay)."""
        k0, k1 = dn.kh(p, 0), dn.kh(p, 1)       # (the same key head, or the next)
        qk = QKH[k0, :] if k1 == k0 else QKH[k0:k1 + 1, :].reshape(1, 4 * dk)[0, :]
        ol.deltanet_step2(dn.states(p), qk, U[p].reshape(1, 2 * dv)[0, :], G[2 * p:2 * p + 2],
                          G[nl + 2 * p:nl + 2 * p + 2], O[p].reshape(1, 2 * dv)[0, :],
                          zero=not isinstance(pos, RunPos) and pos == 0)

    def group(p):
        return p // (og // 2) if (p + 1) % (og // 2) == 0 else None

    def _pair_segment(p, t, last1, last2, g=None):
        """Pair p (buffers t); last1: no pair p+1, last2: no pair p+2; g: the head group pair
        p completes."""
        first = isinstance(p, int) and p == 0
        if not last2 and not first:
            project(p + 2, t)
        if sh and not first:                    # (pair 0: head b's prep beside head a's step)
            dstep_sh(p, t)
        else:
            dstep(p, t, 0)
        if first:                               # head b's prep, after head a's start
            conv(0, 0, 1)
            qk(0, 1)
            gatez(0, 0)
            if not last2:
                project(2, 0)
            if not last1 and not ahead:
                fetch_cv(1)
                fetch_eb(1, 1)
        if not sh or first:
            dstep(p, t, 1)
        if not last1:
            prep(p + 1, 1 - t)
        post(t)
        if og == 4 and isinstance(p, int) and p >= NP - 2:
            flush((NP - 1) // 2, p - (NP - 2))  # the last group pair by pair, as _deltanet
        elif g is not None:
            flush(g)
        if not last2 and not ahead:
            fetch_cv(p + 2)
            fetch_eb(p + 2, t)

    def after_post(p):
        if og == 4 and p >= NP - 2:
            flush((NP - 1) // 2, p - (NP - 2))  # the last group pair by pair, as _deltanet
        elif (p + 1) % (og // 2) == 0:
            flush(p // (og // 2))

    if bursts:
        # the MXU's work a burst ahead (the dispatch window's slots: with every MM first
        # they held it, and the VOPs behind them waited for MMs to finish): burst 0's key
        # heads' and pairs' projections first, burst b + 1's after burst b's prep; then the
        # DMA's in two loads (burst 0's taps and windows, the others'); per burst the VPU's
        # (its pairs' convolutions, its new key heads, one SiLU over its rows, the previous
        # burst's posts) before its DSTEPs, so the VPU switches between VOPs and streams once
        # a burst
        ks, k0 = [], 0
        for ps in bursts:
            k1 = max(dn.kh(q, 1) for q in ps) + 1
            ks.append((k0, max(k0, k1)))
            k0 = max(k0, k1)

        def mms(b):
            h0, h1 = ks[b]
            if h1 > h0:
                kmm(h0, h1)
            for q in bursts[b]:
                project(q, 0)
        mms(0)
        n0 = len(bursts[0])
        rest = dn.cvs(n0, NP)
        for q in range(NP if rest is None else n0):
            fetch_cv(q)
        if rest is not None and n0 < NP:
            ol.load(rest, out=CVA[n0 * CW:NP * CW])
        gates(ab[0, 0:nl], ab[0, nl:2 * nl], alog, dtb, out=G, fold=dn.fold)
        for b, ps in enumerate(bursts):
            h0, h1 = ks[b]
            convs([(q, q, 0, nb) for q in ps])
            if h1 > h0:
                kvec(h0, h1)
            u = UZ[b] if b else UZ[0][0:len(ps) * nb, :]   # (burst 0: z's rows after its steps)
            u.set(_silu_dn(dn, u))                      # (qk's SiLU, and gatez's in place)
            if b:                                       # (_dn_gz's gain)
                z = UZ[b][len(ps) * nb:, :]
                z.set(z * gs[None, :])
            if b + 1 < len(bursts):
                mms(b + 1)
            if not b:
                y = ol.empty([1, spec.hidden // ol.num_programs()])    # (_out_proj: flush)
            else:
                posts(b - 1)
            for q in ps:
                dstep2(q)
            if not b:       # burst 0's z rows beside its steps: they wait for z's MM, its last
                z = UZ[0][len(ps) * nb:, :]
                z.set(_dn_gz(dn, z, gs))
        posts(len(bursts) - 1)
        for q in range(NP):                     # the windows, beside the tail
            store_window(q, 0, 0, nb)
        kstore()
        return _residual(x, y)
    project(0, 0, split=True)
    nk = dn.nkl if qo and ahead else 1
    if qo and ahead:
        kvec(0, 1)
        if nk > 1:
            kmm(1, nk)
    fetch_cv(0)
    fetch_eb(0, 0)
    for q in range(1, NP if ahead else 1):        # (ahead: every pair's, while the DMA waits)
        fetch_cv(q)
    conv(0, 0, 0)
    qk(0, 0)
    if nk > 1:
        kvec(1, nk)
    if NP > 1:
        project(1, 1)
    _pair_segment(0, 0, NP == 1, NP <= 2, group(0))
    if n_it:
        for i in ol.range(n_it):
            _pair_segment(2 * i + 1, 1, False, False, i if og == 4 else 2 * i + 1)
            _pair_segment(2 * i + 2, 0, False, False, None if og == 4 else 2 * i + 2)
    for p in range(1 + 2 * n_it, NP):
        _pair_segment(p, p % 2, p + 1 >= NP, p + 2 >= NP, group(p))
    if pipe is not None:                        # the windows, beside the MLP (_CvPipe)
        pipe.stores = [partial(store_window, q, 0, 0, nb) for q in range(NP)]
    for q in range(NP if ahead and pipe is None else 0):    # the windows, beside the tail
        store_window(q, 0, 0, nb)
    if qo and ahead:
        kstore()                    # the key heads' window rows, beside the tail
    return _residual(x, y)


# the routed experts a resident pool's MoE layer spreads the pipe's DMA over (moe_ffn spread: a
# part after each, their loop's first iterations peeled; at most the layer's k). Peeled with
# nothing queued too: the board model's q36-4 cut ran 3.5 K cycles longer with its attention
# layer's experts left in their loop (400,348 against 396,839 at r60)
BESIDE_EXPERTS = 8


class _CvPipe:
    """qwen35_step's DeltaNet taps and windows a layer ahead (Image.cv_pre): one TMEM tile holds
    the next DeltaNet layer's (every pair's taps, then its window), loaded beside the MLP before
    it, and a layer's window stores wait for its MLP too, both beside the MLP's chunks
    (swiglu_down's each: the DMA is idle there), so the mixer's DMA chain is its DSTEPs alone
    (_deltanet_dstep). In a MoE layer they are queued before its FFN, or with a resident pool
    beside its routed experts (moe_ffn's beside: their MMs stream from the pool, the DMA is
    idle; before the FFN the router's gain load waited for them), in parts, one after each of the
    first BESIDE_EXPERTS (moe_ffn's spread: queued whole before them they filled the sequencer's
    window, and the first routed expert's MMs waited for the DMA). A run's first DeltaNet
    layer's are loaded as the run begins (lfm2.run_layers' before) unless the layer before loaded them; in a loop the unit's last
    DeltaNet layer loads the next iteration's first (after the last iteration, bytes past the
    run, unused, so the next run loads its own)."""

    def __init__(self, m, dn):
        self.m, self.np_, self.cw = m, dn.nl // 2, dn.cv(0).shape[0]
        self.buf = ol.empty([self.np_ * self.cw])
        self.held = self.looped = False     # (the tile holds the next DeltaNet layer's)
        self.run, self.stores, self.loads = None, [], []

    def cv(self, q: int, cw: int):
        """Pair q's taps and window in the tile (cw: the mixer's words a pair)."""
        if cw != self.cw:
            raise CompileError(f"a pair's taps and window: {cw} words, the pipe's {self.cw}")
        return self.buf[q * cw:(q + 1) * cw]

    def _loads(self, dn) -> list:
        return [partial(ol.load, dn.cv(q), out=self.buf[q * self.cw:(q + 1) * self.cw])
                for q in range(self.np_)]

    @staticmethod
    def _lins(unit) -> list:
        return [e for e, k in enumerate(unit) if k[0] == LIN]

    def prime(self) -> None:
        """The first run's first DeltaNet layer's loads, queued as the embedding's gather
        begins (qwen35_step's ahead: the DMA is idle beside its MMs and copies, and the layer's
        first instructions wait for them otherwise); its before() then finds them held."""
        first, unit, _ = self.m.plan[0]
        lin = self._lins(unit)
        if lin and not self.held:
            for f in self._loads(self.m.layer(first + lin[0]).dn):
                f()
            self.held = True

    def tail(self) -> None:
        """prime()'s loads again, for the generate loop's next token (the LM head's lm_tail,
        _loop_ahead: the DMA beside the head's MMs, and the token's first DeltaNet layer finds
        them held). They read the layer's taps and window after this token's stores (one DMA, in
        order); after the loop's last token they load this tile for no one: no DRAM is written,
        and nothing after the loop reads the tile (a run primes its own)."""
        first, unit, _ = self.m.plan[0]
        lin = self._lins(unit)
        if lin:
            for f in self._loads(self.m.layer(first + lin[0]).dn):
                f()
        self.held, self.looped = bool(lin), False

    def before(self, first: int, unit, reps: int) -> None:
        if self.looped:                 # (the loop's last iteration loaded past its run)
            self.held = False
        self.looped, lin = reps > 1, self._lins(unit)
        self.run = (first, reps, lin)
        if lin and not self.held:
            for f in self._loads(self.m.layer(first + lin[0]).dn):
                f()
            self.held = True

    def after(self, li: int, it) -> None:
        """Layer li's mixer (iteration it) is emitted, its window stores queued: the next
        DeltaNet layer's loads after them (they overwrite the windows the stores read)."""
        first, reps, lin = self.run
        later = [e for e in lin if e > li - first]
        nxt = self.m.layer(first + later[0], it) if later else \
            self.m.layer(first + lin[0], Affine.of(it) + 1) if reps > 1 else self._next_run(first)
        self.loads, self.held = ([] if nxt is None else self._loads(nxt.dn)), nxt is not None

    def _next_run(self, first: int):
        runs = self.m.plan
        i = next(i for i, r in enumerate(runs) if r[0] == first)
        for f2, u2, _ in runs[i + 1:]:
            lin = self._lins(u2)
            if lin:
                return self.m.layer(f2 + lin[0])
        return None

    def each(self, n: int):
        """swiglu_down's each over its n chunks: the stores over the first half, then the
        loads over the rest."""
        h = max(1, n // 2)

        def go(i):
            if i < h:
                self._emit(self.stores, h - i)
            else:
                self._emit(self.stores, 1)
                self._emit(self.loads, n - i)
        return go

    @staticmethod
    def _emit(q: list, parts: int) -> None:
        k = -(-len(q) // parts)
        for f in q[:k]:
            f()
        del q[:k]

    def flush(self) -> None:
        for q in (self.stores, self.loads):
            self._emit(q, 1)

    def share(self, parts: int = 1) -> None:
        """1/parts of what is queued (moe_ffn's spread), the stores first: the loads overwrite
        the windows the stores read."""
        n = -(-(len(self.stores) + len(self.loads)) // parts)
        for q in (self.stores, self.loads):
            m = min(n, len(q))
            for f in q[:m]:
                f()
            del q[:m]
            n -= m


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
    pipe = getattr(m, "cv_pipe", None)  # (the generate loop's, primed before it: _loop_ahead)
    if pipe is None:
        pipe = _pipe(m, pos)
    x, c, s_ = _inputs(m, pos, tok, ahead=None if pipe is None else pipe.prime)

    def layer(li, it):
        lw = m.layer(li, it)
        if lw.moe and spec.moe.hint:        # the router's guess, before the mixer
            MO.moe_hint(x, lw, spec.moe, m.moe_dev, spec.eps)
        if lw.kind == LIN and pipe is not None:
            x.set(_deltanet_dstep(x, lw, pos, spec, m.hs, pipe))
            pipe.after(li, it)
        elif lw.kind == LIN:
            dn = _deltanet_dstep if ol.has_dstep() else _deltanet
            x.set(dn(x, lw, pos, spec, m.hs))
        else:
            x.set(_attention(x, lw, c, s_, pos, spec, block, gated=True))
        if lw.moe:                  # (a resident pool: the pipe's DMA beside the experts)
            res = pipe is not None and getattr(m.moe_dev, "pool", None) is not None
            if pipe is not None and not res:
                pipe.flush()
            x.set(MO.moe_ffn(x, lw, spec.moe, m.moe_dev, spec.eps,
                             beside=pipe.share if res else None,
                             spread=BESIDE_EXPERTS if res else 0))
        else:
            x.set(_mlp(x, lw, spec, each=None if pipe is None else
                       pipe.each(lw.wg.shape[0] // f_chunk(lw.wg, lw.wd))))
        if pipe is not None:
            pipe.flush()

    run_layers(m.plan, layer, before=None if pipe is None else pipe.before)
    _lm_head(x, m, spec)


def _pipe(m, pos):
    """The taps and windows a layer ahead (Image.cv_pre: _CvPipe) where the step runs them, else
    None."""
    if not (getattr(m, "cv_pre", False) and ol.has_dstep() and isinstance(pos, RunPos)):
        return None
    d0 = next((m.layer(i).dn for i, k in enumerate(m.spec.kinds) if k == LIN), None)
    if d0 is not None and d0.ahead and not (d0.grouped or d0.slot or d0.qkonce):
        return _CvPipe(m, d0)
    return None


def _loop_ahead(m, pos):
    """qwen35_step's work before the generate loop (generate._generate): the gather's operand
    (qwen3.loop_ahead), and the pipe of taps and windows primed for the loop's first token, each
    token's LM head loading the next one's (_CvPipe.tail), so a token's first DeltaNet layer
    waits for no load of its own (the tile held for the loop, as each step held it)."""
    loop_ahead(m, pos)
    pipe = _pipe(m, pos)
    if pipe is not None:
        pipe.prime()
        m.cv_pipe, m.lm_tail = pipe, pipe.tail


qwen35_step.ahead = _loop_ahead


def _deltanet_rows(x, lw, p0: int, spec: Spec, gr, on, fork: bool = False, xs=None):
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
    stored after the row before it too). xs: x's normed rows already quantized (a tiled run's,
    tiled._norm_quantize: the same operand, without the normed [R, H] tile in TMEM)."""
    eps, K, R = spec.eps, spec.conv_k, x.rows
    dk, dv = spec.lin_dk, spec.lin_dv
    dn = lw.dn
    eo = _dn_eps(dn, spec)              # (the output norm's)
    nl, C = dn.nl, 2 * dk + dv
    og, sh, qo = dn.og, dn.shared, dn.qkonce            # sh: the pair's q and k shared; qo:
    nb, bw = (4, dk) if sh else (2, dv) if qo else (2, C)   # a layer's; the pair's blocks
    CP, RP = nb * bw, nb * bw + 2 * dv                  # its convolved channels, its rows
    NP, ng, gp = nl // 2, nl // og, og // 2             # pairs, head groups, pairs per group
    TP = K * CP                                         # taps words of a pair (then its ring)
    if xs is None:                                      # (a tiled run's: tiled.py)
        xs = ol.quantize(rmsnorm(x, ol.load_home(lw.g_in, "g_in"), eps))
    dstep = ol.has_dstep()
    split = og == 4 and NP > 1                          # _deltanet's last group, in pairs
    # qkonce's pipelined pairs are unrolled, so their DSTEPs read the gates in TMEM (G: each
    # row's decays, then its betas); else they go through `gr` in DRAM
    G = ol.empty([R, 2 * nl]) if dstep and split and qo else None
    ab = ol.dot(xs, lw.wab)                             # [R, 2nl]: a, then b, of each head
    alog, dtb = ol.load(lw.alog), ol.load(lw.dtb)
    for r in range(R):
        if G is not None:
            gates(ab[r, 0:nl], ab[r, nl:2 * nl], alog, dtb, out=G[r, :], fold=dn.fold)
            continue
        dr, br = gates(ab[r, 0:nl], ab[r, nl:2 * nl], alog, dtb, fold=dn.fold)
        ol.store(gr[r, 0:nl], dr)
        ol.store(gr[r, nl:2 * nl], br)
        del dr, br
    del ab, alog, dtb
    gs = _dn_gain(ol.load(lw.gn), dv)
    y = ol.zeros([R, spec.hidden // ol.num_programs()])    # (_out_proj)
    if not dstep:
        St = ol.empty([dv * dk]).reshape(dv, dk)
        w = ol.empty([dv])
    if fork and (R < 2 or p0 < K - 1):
        raise CompileError("a forked rows run needs 2 rows or more, from position conv_k - 1")
    if fork and qo:
        raise CompileError("a forked rows run: q and k once a layer has no window slots")

    def fdst(p, a, r):
        """Where row r's step writes head a of pair p: in place, or (fork, the last row) the
        other slot."""
        return dn.state(p, a, 1 - dn.slot) if fork and r == R - 1 else None


    def flush(g, ON, c0, c1):
        """y += ON . out_proj columns [c0, c1) of head group g (heads of dv columns)."""
        _out_proj(dn, ON, g, c0, c1, dv, y)

    def head_in(X, taps, a, out=None, act=True, bw=bw):
        """Block a of the channels of the pair whose rows are in X (head a's q k v; shared, q, k,
        v of a, v of b; qkonce, v of a, of b; or the key heads' q | k, blocks bw wide): the
        convolution and SiLU (act) -> [R, bw] (into `out`)."""
        return _dn_conv(dn, X, taps, a, bw, p0, R, K, out, act)

    def head(p, X, Z, taps, a):
        """Head a of pair p (whose q k v rows are in X, z in Z; qkonce: v, its q | k in QKR's
        columns of its key head) -> ONp[:, a]."""
        U = head_in(X, taps, a)
        GZ = _dn_gz(dn, Z[:, a * dv:(a + 1) * dv], gs)
        if qo:
            h = 2 * dk * dn.kh(p, a)
            V = U
        else:
            V = U[:, 2 * dk:C]
        if dstep:                                       # q | k per row, a DSTEP per row
            if qo:
                QK = QKR[:, h:h + 2 * dk]
            else:
                QK = ol.empty([R, 2 * dk])
                QK[:, 0:dk].set(_l2n(dn, U[:, 0:dk], dk ** -0.5))
                QK[:, dk:2 * dk].set(_l2n(dn, U[:, dk:2 * dk]))
            O = ol.empty([R, dv])
            for r in range(R):
                ol.deltanet_step(dn.state(p, a), QK[r, :], V[r, :], GDB[r, a:a + 1],
                                 GDB[r, 2 + a:3 + a], O[r, :], zero=(p0 == 0 and r == 0),
                                 dst=fdst(p, a, r))
            del QK
            ONp[:, a * dv:(a + 1) * dv].set(_dn_norm(O, eo) * GZ)
            return
        if qo:
            Qn, Kn = QKR[:, h:h + dk], QKR[:, h + dk:h + 2 * dk]
        else:
            Qn = _l2n(dn, U[:, 0:dk], dk ** -0.5)
            Kn = _l2n(dn, U[:, dk:2 * dk])
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
            w.set((V[r, :] - w * dh) * bh)
            ol.outer(w, Kn[r, :], acc=St, decay=dh)
            O[r, :].set(St @ Qn[r, :])
        ol.store(dn.state(p, a, 1 - dn.slot) if fork else dn.state(p, a), St)
        ONp[:, a * dv:(a + 1) * dv].set(_dn_norm(O, eo) * GZ)

    def shared_heads(p, X, Z, taps):
        """Pair p's heads, the key head's q and k shared (blocks q, k, v of a, v of b in X) ->
        ONp: q and k convolved and normed once. With DSTEP both heads' vector work before
        their DSTEPs, as heads()."""
        if dstep:
            QK = ol.empty([R, 2 * dk])
            for i in range(2):
                U = head_in(X, taps, i)
                QK[:, i * dk:(i + 1) * dk].set(_l2n(dn, U, dk ** -0.5 if i == 0 else 1.0))
                del U
            V = [head_in(X, taps, 2 + a) for a in range(2)]
            GZ = [_dn_gz(dn, Z[:, a * dv:(a + 1) * dv], gs) for a in range(2)]
            O = [ol.empty([R, dv]) for _ in range(2)]
            for a in range(2):
                for r in range(R):
                    ol.deltanet_step(dn.state(p, a), QK[r, :], V[a][r, :], GDB[r, a:a + 1],
                                     GDB[r, 2 + a:3 + a], O[a][r, :], zero=(p0 == 0 and r == 0),
                                     dst=fdst(p, a, r))
            del QK, V
            for a in range(2):
                ONp[:, a * dv:(a + 1) * dv].set(_dn_norm(O[a], eo) * GZ[a])
            return
        U = head_in(X, taps, 0)
        Qn = _l2n(dn, U, dk ** -0.5)
        del U
        U = head_in(X, taps, 1)
        Kn = _l2n(dn, U)
        del U
        for a in range(2):
            V = head_in(X, taps, 2 + a)
            GZ = _dn_gz(dn, Z[:, a * dv:(a + 1) * dv], gs)
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
            ONp[:, a * dv:(a + 1) * dv].set(_dn_norm(O, eo) * GZ)
            del V, GZ, O

    def key_heads():
        """q | k of the slice's key heads in each row ([R, QC], _key_heads' row by row): their
        projection after their window's rows, the convolution and SiLU, the norms over rows
        (row, key head); the next window stored."""
        QC, nk = dn.QC, dn.nkl
        taps = ol.load(dn.qk_cv()[0:dn.qtp]).reshape(nk * K, 2 * dk)
        X = ol.empty([K - 1 + R, QC])
        win = dn.qk_window()
        for j in range(1, min(K, p0 + 1)):              # positions p0-K+1 ..: the window
            sl = (K - 1 - j) * QC                       # (_past)
            ol.load(win[sl:sl + QC], out=X[K - 1 - j, :])
        ol.dot(xs, dn.wqk, out=X[K - 1:K - 1 + R, :])
        for i in range(K - 1):
            ol.store(win[i * QC:(i + 1) * QC], X[R + i, :])
        U = ol.empty([R, QC], dense=True)
        for a in range(nk):
            head_in(X, taps, a, out=U[:, a * 2 * dk:(a + 1) * 2 * dk], act=False, bw=2 * dk)
        del X, taps
        U.set(_silu_dn(dn, U))
        Hq = U.reshape(R * nk, 2 * dk)
        Hq[:, 0:dk].set(_l2n(dn, Hq[:, 0:dk], dk ** -0.5))
        Hq[:, dk:2 * dk].set(_l2n(dn, Hq[:, dk:2 * dk]))
        return U

    # qkonce's pipelined pairs take the key heads a group at a time (kc key heads, a pair's two
    # in one group), each just ahead of its first pair, so that the first DSTEPs need not wait
    # for every key head's projection: a group's projection, its taps and window rows, then
    # (beside the DSTEPs before its pairs') its convolution, SiLU and norms; its window stored a
    # segment later, its q | k kept until its last pair's DSTEPs
    kc = 2 if qo and dn.rk % 2 else 1
    KJ = G is not None
    QKR = key_heads() if qo and not KJ else None        # (qkonce)
    KG: dict = {}       # key head group -> [taps, X ([K - 1 + R, kc 2dk]), q | k ([R, kc 2dk])]

    def key_group_mm(g):
        """Key head group g's projection into the rows K - 1 .. of its X."""
        c0, c1 = g * kc * 2 * dk, (g + 1) * kc * 2 * dk
        X = ol.empty([K - 1 + R, c1 - c0])
        ol.dot(xs, dn.wqk[c0:c1, :], out=X[K - 1:K - 1 + R, :])
        KG[g] = [None, X, None]

    def key_group_ld(g):
        """Key head group g's taps (rows (key head, tap)) and window rows (X's first K - 1)."""
        c0, c1 = g * kc * 2 * dk, (g + 1) * kc * 2 * dk
        tw = kc * K * 2 * dk
        KG[g][0] = ol.load(dn.qk_cv()[g * tw:(g + 1) * tw]).reshape(kc * K, 2 * dk)
        win = dn.qk_window()
        for j in range(1, min(K, p0 + 1)):              # positions p0-K+1 ..: the window
            sl = (K - 1 - j) * dn.QC                    # (_past)
            ol.load(win[sl + c0:sl + c1], out=KG[g][1][K - 1 - j, :])

    def key_group_vec(g):
        """Key head group g's q | k in each row ([R, kc 2dk]: key_heads' columns of its key
        heads): the convolution, SiLU and the norms over rows (row, key head), key_heads'
        operations on its elements and rows."""
        taps, X, _ = KG[g]
        U = ol.empty([R, kc * 2 * dk], dense=True)
        for a in range(kc):
            head_in(X, taps, a, out=U[:, a * 2 * dk:(a + 1) * 2 * dk], act=False, bw=2 * dk)
        U.set(_silu_dn(dn, U))
        Hq = U.reshape(R * kc, 2 * dk)
        Hq[:, 0:dk].set(_l2n(dn, Hq[:, 0:dk], dk ** -0.5))
        Hq[:, dk:2 * dk].set(_l2n(dn, Hq[:, dk:2 * dk]))
        KG[g][2] = U

    def key_group_st(g):
        """Key head group g's next window (its X's last K - 1 rows); its taps and X freed."""
        c0, c1 = g * kc * 2 * dk, (g + 1) * kc * 2 * dk
        win = dn.qk_window()
        for i in range(K - 1):
            ol.store(win[i * dn.QC + c0:i * dn.QC + c1], KG[g][1][R + i, :])
        KG[g][0] = KG[g][1] = None

    def qk_rows(p, a0: int, a1: int):
        """q | k of heads a0 .. a1 - 1 of pair p (qkonce), [R, 2dk] (one key head) or [R, 4dk]
        (a pair's two, the second key head the next)."""
        k0, k1 = dn.kh(p, a0), dn.kh(p, a1 - 1)
        n = 2 * dk * (k1 - k0 + 1)
        if QKR is not None:
            return QKR[:, 2 * dk * k0:2 * dk * k0 + n]
        g = k0 // kc
        b = 2 * dk * (k0 - g * kc)
        return KG[g][2][:, b:b + n]

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
        if not dstep or qo:
            for a in range(2):
                head(p, X, Z, taps, a)
            return
        QK = [ol.empty([R, 2 * dk]) for _ in range(2)]
        V = [ol.empty([R, dv]) for _ in range(2)]
        GZ = [ol.empty([R, dv]) for _ in range(2)]
        for a in range(2):
            U = head_in(X, taps, a)
            QK[a][:, 0:dk].set(_l2n(dn, U[:, 0:dk], dk ** -0.5))
            QK[a][:, dk:2 * dk].set(_l2n(dn, U[:, dk:2 * dk]))
            V[a].set(U[:, 2 * dk:C])
            del U
            GZ[a].set(_dn_gz(dn, Z[:, a * dv:(a + 1) * dv], gs))
        O = [ol.empty([R, dv]) for _ in range(2)]
        for a in range(2):
            for r in range(R):
                ol.deltanet_step(dn.state(p, a), QK[a][r, :], V[a][r, :], GDB[r, a:a + 1],
                                 GDB[r, 2 + a:3 + a], O[a][r, :], zero=(p0 == 0 and r == 0),
                                 dst=fdst(p, a, r))
        del QK, V
        for a in range(2):
            ONp[:, a * dv:(a + 1) * dv].set(_dn_norm(O[a], eo) * GZ[a])

    def pair(p):
        """Pair p (an int, a loop expression or (group, index): DeltaNetParts) -> ONp."""
        project(p, 0)
        heads(p, 0)

    def loop(n):
        """ol.range(n), or the single index 0 unrolled (qkonce: every index, a pair's key
        head)."""
        return ol.range(n) if n > 1 and not qo else range(n)

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
        GDBb = None if G is not None else \
            [ol.empty([R, 4]) for _ in range(2)]        # decays, then betas, of the pair's heads
        ON = ol.empty([R, og * dv])                     # the group's normed, gated o
        pair2 = ol.has_dstep2() and (qo or sh)          # a pair's two heads a row in one DSTEP
        kfirst, klast = {}, {}                          # (KJ) key head group -> its first,
        if KJ:                                          # last pair
            for p in range(NP):
                kfirst.setdefault(dn.kh(p, 1) // kc, p)
                klast[dn.kh(p, 1) // kc] = p
        kst = []                                        # (KJ) groups whose window is to store

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
            if G is not None:                           # (qkonce: the gates in TMEM)
                return
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
            U.set(_silu_dn(dn, U))
            # q and k L2-normed in place: shared, the key head's; else both heads', as rows
            # (row, head) of the pair's channels; qkonce: none (the key heads', QKR)
            if not qo:
                H = U if sh else U.reshape(2 * R, C)
                H[:, 0:dk].set(_l2n(dn, H[:, 0:dk], dk ** -0.5))
                H[:, dk:2 * dk].set(_l2n(dn, H[:, dk:2 * dk]))
            store_window(p, t)

        def gate(p, t, r, a, n=1):
            """Decay and beta of heads a .. a + n - 1 of pair p (in buffers t), row r: G's
            (qkonce, beta nl words after decay) or GDBb[t]'s."""
            if G is not None:
                q = dn.index(p)
                return G[r, 2 * q + a:2 * q + a + n], G[r, nl + 2 * q + a:nl + 2 * q + a + n]
            return GDBb[t][r, a:a + n], GDBb[t][r, 2 + a:2 + a + n]

        def dsteps(p, t):
            U = Ub[t]
            if pair2:                                   # row by row, both heads a DSTEP
                qk2 = qk_rows(p, 0, 2) if qo else U[:, 0:2 * dk]
                v2 = U[:, 0:2 * dv] if qo else U[:, 2 * dk:4 * dk]
                for r in range(R):
                    if fork and r == R - 1:             # (shared: a STREAM a head)
                        for a in range(2):
                            ol.deltanet_step(dn.state(p, a), qk2[r, :],
                                             v2[r, a * dv:(a + 1) * dv], *gate(p, t, r, a),
                                             O[a][r, :], dst=fdst(p, a, r))
                        continue
                    ol.deltanet_step2(dn.states(p), qk2[r, :], v2[r, :], *gate(p, t, r, 0, 2),
                                      O2[r, :], zero=(p0 == 0 and r == 0))
                return
            for a in range(2):
                if qo:
                    qk, v = qk_rows(p, a, a + 1), U[:, a * dv:(a + 1) * dv]
                else:
                    qk = U[:, 0:2 * dk] if sh else U[:, a * C:a * C + 2 * dk]
                    v = U[:, (2 + a) * dk:(3 + a) * dk] if sh else \
                        U[:, a * C + 2 * dk:(a + 1) * C]
                for r in range(R):
                    ol.deltanet_step(dn.state(p, a), qk[r, :], v[r, :], *gate(p, t, r, a),
                                     O[a][r, :], zero=(p0 == 0 and r == 0), dst=fdst(p, a, r))

        def post(t):
            """Pair t's gated RMSNorm into its columns of ON: the norm over rows (row, head)."""
            gz = _silu_dn(dn, PB[t][2])
            for a in range(2):                          # (_dn_gz's gain, a head at a time)
                gz[:, a * dv:(a + 1) * dv].set(gz[:, a * dv:(a + 1) * dv] * gs[None, :])
            on = _dn_norm(O2.reshape(2 * R, dv), eo).reshape(R, 2 * dv)
            ON[:, 2 * t * dv:(2 * t + 2) * dv].set(on * gz)
            del gz, on

        def segment(p, t, last1, last2, g=None, half=None):
            """Pair p; last1: no pair p + 1, last2: no pair p + 2; g: the head group pair p
            completes; half: pair p's half of the last group (flushed pair by pair)."""
            kn = None if last2 else next((k for k, q in kfirst.items() if q == p + 2), None)
            if kn is not None:                          # (KJ: pair p + 2 starts a key group)
                key_group_mm(kn)
            if not last2:
                mm_x(p + 2, t)
            dsteps(p, t)
            for k in [k for k, q in klast.items() if q == p]:
                KG[k][2] = None                         # (its last pair's: its q | k freed)
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
            for k in kst:                               # (behind this segment's DSTEPs)
                key_group_st(k)
            kst.clear()
            if kn is not None:
                key_group_ld(kn)
                key_group_vec(kn)
                kst.append(kn)

        def unrolled(p):
            last = p >= NP - 2
            segment(p, p % 2, p + 1 >= NP, p + 2 >= NP,
                    None if last or p % 2 == 0 else (p - 1) // 2, p - (NP - 2) if last else None)

        k01 = [k for k, q in kfirst.items() if q < 2]   # (KJ: the groups of pairs 0 and 1)
        for k in k01:
            key_group_mm(k)
        fetch(0, 0)
        mm_x(0, 0)
        mm_z(0, 0)
        fetch(1, 1, taps=False)
        mm_x(1, 1)
        mm_z(1, 1)
        for k in k01:
            key_group_ld(k)
            key_group_vec(k)
            kst.append(k)
        prep(0, 0)
        ol.load(dn.cv(1)[0:TP], out=taps1)             # after pair 0's convolution
        unrolled(0)
        n_it = max(0, (NP - 3) // 2)                    # segments 1 .. NP - 3: every part
        for i in loop(n_it):
            segment(2 * i + 1, 1, False, False, g=i)
            segment(2 * i + 2, 0, False, False)
        for p in range(1 + 2 * n_it, NP):
            unrolled(p)
        for k in kst:
            key_group_st(k)

    if NB == 2:
        _rows_pipelined()
        return _residual(x, y)
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
        for q in range(gp):     # (unrolled: its out_proj columns' first block, k0, is an MM's
            #                     immediate)
            pair((ng - 1, q))
            flush(ng - 1, ONp, 2 * q, 2 * q + 2)
    return _residual(x, y)



# ------------------------------------------------- the DeltaNet rows by chunks (Config.DNCHUNK)
# A prefill chunk's DeltaNet layer as prep tiles (_deltanet_prep: everything before the
# recurrence, to the image's dnq), the core over the chunk's rows (the recurrence: dnq to dno;
# _deltanet_core_rows by DSTEPs, or item 1's chunked kernel) and post tiles (_deltanet_post:
# the gated norm, out_proj, the residual). docs/deltanet_chunk.md. The
# shared layout alone (a pair's two heads on one key head: Qwen3.5-MoE's), positions from
# conv_k - 1 (a tile's).

def _dn_shared(dn, what: str) -> None:
    if not dn.shared or dn.qkonce:
        raise CompileError(f"the chunked DeltaNet's {what}: pairs sharing their key head")


def _dn_pairs(dn):
    """The slice's pairs as (group, index): a loop over the head groups, their pairs unrolled
    (a group-major image's pair is a group's block and an offset in it: DeltaNetParts.split)."""
    ng = dn.nl // dn.og
    for g in (ol.range(ng) if ng > 1 else range(ng)):
        for i in range(dn.gp):
            yield (g, i)


def _deltanet_prep(x, lw, spec: Spec, dnq, row, xs, T: int = 0) -> None:
    """The chunked DeltaNet's prep of chunk rows row .. row + T - 1 (x [T, H], its normed rows
    quantized xs; row a run argument): the slice's heads' gates (kernels.deltanet.gate_logs:
    lg, beta) into dnq.lg / dnq.b ([nl, Rmax]: a head a row, the tile's columns), then per
    pair its projections, the convolution over its window (stored back, as _deltanet_rows'),
    SiLU, q and k L2-normed (q times dk^-1/2) into dnq: q, k of its key head, v and z (the
    projection) of its two heads, head-major. The words _deltanet_rows' DSTEPs take."""
    dn, K, T = lw.dn, spec.conv_k, x.rows if x is not None else T
    dk, dv, nl = spec.lin_dk, spec.lin_dv, dn.nl
    _dn_shared(dn, "prep")
    CP = 4 * dk                                         # a pair's convolved channels
    ab = ol.dot(xs, lw.wab)                             # [T, 2nl]: a, then b, of each head
    lg, be = gate_logs(ab[:, 0:nl], ab[:, nl:2 * nl], ol.load(lw.alog), ol.load(lw.dtb),
                       fold=dn.fold)
    del ab
    GT = ol.empty([2 * nl, T])                          # lg, then beta, a head a row
    for j in range(nl):
        GT[j, :].column().set(lg[:, j:j + 1])
        GT[nl + j, :].column().set(be[:, j:j + 1])
    del lg, be
    ol.store(Tensor(dnq.lg.base, (2 * nl, dnq.rows), (dnq.rows, 1))[:, row:row + T], GT)
    del GT
    taps, X, Z = ol.empty([K * CP]), ol.empty([K - 1 + T, CP]), ol.empty([T, 2 * dv])
    for p in _dn_pairs(dn):
        ol.load(dn.cv(p)[0:K * CP], out=taps)          # rows (block, tap)
        win = dn.window(p)
        ol.current().stage_wait(("win", p))             # (merged: the tile before's window)
        for j in range(1, K):                           # positions row - K + 1 ..: the window
            ol.load(win[(K - 1 - j) * CP:(K - j) * CP], out=X[K - 1 - j, :])
        ol.dot(xs, dn.wh(p)[0:CP, :], out=X[K - 1:K - 1 + T, :])
        ol.dot(xs, dn.wh(p)[CP:CP + 2 * dv, :], out=Z)                  # z of a, of b
        for i in range(K - 1):                          # the next window
            ol.store(win[i * CP:(i + 1) * CP], X[T + i, :])
        ol.current().stage_post(("win", p))
        tp, q = taps.reshape(4 * K, dk), Affine.of(dn.index(p))
        for i, (t, h) in enumerate(((dnq.q, q), (dnq.k, q), (dnq.v, q * 2), (dnq.v, q * 2 + 1))):
            U = _dn_conv(dn, X, tp, i, dk, K - 1, T, K)  # q, k, v of a, v of b
            if i < 2:
                U.set(_l2n(dn, U, dk ** -0.5 if i == 0 else 1.0))
            ol.store(t[h, row:row + T, :], U)
            del U
        for a in range(2):
            ol.store(dnq.z[q * 2 + a, row:row + T, :], Z[:, a * dv:(a + 1) * dv])


def _deltanet_core_rows(lw, spec: Spec, dnq, dno, row, T: int) -> None:
    """The chunked DeltaNet's core over chunk rows row .. row + T - 1 by today's recurrence:
    per head of the slice a DSTEP a row on its state (the committed slot), q | k, v, the
    decay (2^lg) and beta from dnq -> o into dno ([Rmax, nl dv]). With _deltanet_prep and
    _deltanet_post, _deltanet_rows' words (the reference for item 1's chunked core)."""
    dn, dk, dv = lw.dn, spec.lin_dk, spec.lin_dv
    _dn_shared(dn, "core")
    QK, V = ol.empty([T, 2 * dk], dense=True), ol.empty([T, dv])
    G, O = ol.empty([2, T]), ol.empty([T, dv])          # (lg -> the decays, then betas)
    for p in _dn_pairs(dn):
        q = Affine.of(dn.index(p))
        ol.load(dnq.q[q, row:row + T, :], out=QK[:, 0:dk])
        ol.load(dnq.k[q, row:row + T, :], out=QK[:, dk:2 * dk])
        for a in range(2):
            j = q * 2 + a
            ol.load(dnq.v[j, row:row + T, :], out=V)
            ol.load(dnq.lg[j, row:row + T], out=G[0, :])
            ol.load(dnq.b[j, row:row + T], out=G[1, :])
            G[0, :].set(ol.exp2(G[0, :]))              # (gates' decay: its EXP2 of lg)
            for r in range(T):                          # (DSTEP: beta after the decay)
                ol.deltanet_step(dn.state(p, a), QK[r, :], V[r, :], G[0, r:r + 1],
                                 G[1, r:r + 1], O[r, :])
            ol.store(dno[row:row + T, j * dv:j * dv + dv], O)


def _deltanet_post(x, lw, spec: Spec, dnq, dno, row, T: int = 0, X=None):
    """The chunked DeltaNet's post of chunk rows row .. row + T - 1 (x [T, H]): each pair's
    o (dno) through the gated output norm with silu(z) (dnq.z), out_proj over _deltanet_rows'
    head groups (the last one pair by pair when og = 4) -> x + out_proj(o) (_residual). x None
    (a merged stage, stage.py): T rows of X, the slice's own columns loaded only at the sum, the
    same words with TMEM free of x's until then and the rest of x's bytes not read."""
    dn, T = lw.dn, x.rows if x is not None else T
    dv, nl, og = spec.lin_dv, dn.nl, dn.og
    _dn_shared(dn, "post")
    eo = _dn_eps(dn, spec)
    NP, ng, gp = nl // 2, nl // og, og // 2
    split = og == 4 and NP > 1                          # _deltanet's last group, in pairs
    gs = _dn_gain(ol.load(lw.gn), dv)
    y = ol.zeros([T, spec.hidden // ol.num_programs()])    # (_out_proj)
    O2 = ol.empty([T, 2 * dv], dense=True)              # o of the pair's heads
    Z = ol.empty([T, 2 * dv])
    ON = ol.empty([T, og * dv])                         # the group's normed, gated o

    def pair(p, out):
        q = Affine.of(dn.index(p))
        ol.load(dno[row:row + T, q * (2 * dv):q * (2 * dv) + 2 * dv], out=O2)
        for a in range(2):
            ol.load(dnq.z[q * 2 + a, row:row + T, :], out=Z[:, a * dv:(a + 1) * dv])
        gz = _silu_dn(dn, Z)
        for a in range(2):                              # (_dn_gz's gain, a head at a time)
            gz[:, a * dv:(a + 1) * dv].set(gz[:, a * dv:(a + 1) * dv] * gs[None, :])
        on = _dn_norm(O2.reshape(2 * T, dv), eo).reshape(T, 2 * dv)
        out.set(on * gz)
        del gz, on

    n = ng - 1 if split else ng                         # whole groups
    for g in (ol.range(n) if n > 1 else range(n)):
        for i in range(gp):
            pair((g, i), ON[:, 2 * i * dv:(2 * i + 2) * dv])
        _out_proj(dn, ON, g, 0, og, dv, y)
    if split and x is None and ol.num_programs() > 1 and getattr(dn, "sexact", False):
        for i in range(gp):                             # (merged: _last_pairs' MMs from one
            pair((ng - 1, i), ON[:, 2 * i * dv:(2 * i + 2) * dv])   # gather, its buffer's words
        _last_group(dn, ON, ng - 1, y)                  # only from there to their QACTs)
    elif split:                                         # the last group, pair by pair
        for i in range(gp):
            pair((ng - 1, i), ON[:, 0:2 * dv])
            _out_proj(dn, ON[:, 0:2 * dv], ng - 1, 2 * i, 2 * i + 2, dv, y)
    if x is None:
        del O2, Z, ON
        if ol.num_programs() == 1:
            return ol.load(X).add(y)
        sid, hl = ol.program_id(), y.cols
        return ol.all_gather(ol.load(X[:, sid * hl:(sid + 1) * hl]) + y)
    return _residual(x, y)

def _moe_or_mlp(x, lw, m, spec):
    """x + the layer's MLP on x's rows: the dense one, or a MoE layer's routed experts and
    shared expert (one row: moe.moe_ffn, the decode step's; more: moe.moe_ffn_rows, one request
    for the rows' union, each row moe_ffn's bit for bit)."""
    if not lw.moe:
        return _mlp(x, lw, spec)
    if x.rows == 1:
        return MO.moe_ffn(x, lw, spec.moe, m.moe_dev, spec.eps)
    return MO.moe_ffn_rows(x, lw, spec.moe, m.moe_dev, spec.eps)


def plan_span(runs, lo: int, hi: int) -> list:
    """The runs of a plan (lfm2.plan's) over layers lo .. hi - 1, which must be whole runs (a
    run's layers address its blocks from the run's first, _place)."""
    out = []
    for first, unit, reps in runs:
        end = first + len(unit) * reps
        if lo < end and first < hi:
            if first < lo or hi < end:
                raise CompileError(f"layers {lo} .. {hi - 1} cut the run of layers {first} .. "
                                   f"{end - 1}")
            out.append((first, unit, reps))
    return out


@ol.jit
def qwen35_rows(m, p0: int, R: int, logit_rows, block: int = ATTN_BLOCK, tokens=None,
                fork: bool = False, hidden: bool = False, span=None, xs=None,
                nest: bool = True):
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
    position from K - 1 on, so they take the bucket's first position.

    span (lo, hi), a run-time rows run's: the layers lo .. hi - 1 alone, whole runs of the
    plan (a program over IMEM in two parts, mtp.py's split verify): from lo > 0 the rows'
    residual from xs (R rows in DRAM) instead of their embedding rows; below the last layer
    the residual to xs, and no hidden and no head."""
    spec = m.spec
    run = isinstance(p0, RunRows)
    rows = p0 if run else [(0, p0 + r) for r in range(R)]
    pd = p0.lo if run else p0
    if run and (R != p0.R or pd < spec.conv_k - 1):
        raise CompileError(f"a run-time rows run: {p0.R} rows from position {pd}")
    lo, hi = (0, spec.layers) if span is None else span
    if (lo or hi < spec.layers) and (not run or xs is None):
        raise CompileError("a span of the layers: a run-time rows run, its residual in xs")
    if lo:
        _, c, s_ = _inputs_rows(m, rows, tokens, embed=False)
        x = ol.empty([R, m.xr.shape[1]], dense=True)
        ol.load(xs[0:R, :], out=x)
    else:
        x, c, s_ = _inputs_rows(m, rows, tokens)

    def layer(li, it, jt=None):
        lw = m.layer(li, it, jt)
        if lw.kind == LIN:
            x.set(_deltanet_rows(x, lw, pd, spec, m.gr[0:R, :], m.on[0:R, :], fork))
        else:
            x.set(_attention_rows(x, lw, c, s_, rows, spec, block, gated=True))
        x.set(_moe_or_mlp(x, lw, m, spec))

    # prefill loops a unit's DeltaNet layers inside the unit's loop (NEST); MTP's verify rows
    # (nest False: decode-time programs; a span's runs too) keep a copy of each layer of the unit
    run_layers(m.plan if span is None else plan_span(m.plan, lo, hi), layer,
               m.subs if NEST and nest and span is None else None)
    if hi < spec.layers:
        ol.store(xs[0:R, :], x)
        return
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
    x.set(_moe_or_mlp(x, lw, m, spec))      # (Qwen3.5-MoE's: the MTP layer's MoE, a layer of
                                            # the expert server's after the model's)
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
    elif ol.num_programs() > 1:     # (tensor-parallel experts: a row at a time, the step's)
        for r in range(R):
            x[r:r + 1, :].set(MO.moe_ffn(x[r:r + 1, :], lw, spec.moe, m.moe_dev, spec.eps))
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
    record's X, after the last layer's combine (expert-major: its MoE ends here). An MTP image
    (Spec.mtp) also stores the row's final-normed hidden to m.hid[0]: the prompt's last row
    drafts after the head (qwen35_mtp, its token the head's pick)."""
    spec = m.spec
    if not em and not spec.mtp:
        _lm_head(ol.load(m.xbuf[row:row + 1, :]), m, spec)
        return
    if em:
        x = ol.load(MO.em_x(m.moe_dev, row, 1, spec.hidden))
        if m.layer(spec.layers - 1).moe:
            x.set(MO.moe_combine_rows(x, spec.moe, m.moe_dev, row))
    else:
        x = ol.load(m.xbuf[row:row + 1, :])
    if spec.mtp:
        ol.store(m.hid[0:1, :], rmsnorm(x, ol.load(m.g_final), spec.eps))
    _lm_head(x, m, spec)


@ol.jit
def qwen35_mtp_kv_run(m, pos, row, block: int = ATTN_BLOCK, R: int = 1, em: bool = False,
                      keep: bool = False):
    """A layer-major prefill's MTP rows (Spec.mtp, docs/mtp.md 12): after a chunk's last layer,
    its R rows from `row` (a run-time value) at the positions pos .. pos + R - 1 (a RunPos, in
    one attention block) append the MTP layer's K / V as qwen35_mtp's rows do: row r's
    final-normed hidden (the residual row m.xbuf[row + r]; em: its record's X after the last
    layer's combine) and the embedding row of the token after it (the host's slot rows 0 .. R -
    1, embed_host) -> fc -> the MTP layer's K and V. Its queries, MoE and draft head are left
    out (nothing reads them: only the prompt's last row drafts, qwen35_mtp after the head, its
    token the head's pick), so the run posts no expert request. keep (em): the rows' residual,
    the combine's, to m.xbuf[row:row + R] too (where a layer-major run keeps it), for the host
    to keep with a snapshot after them (mtp.MTPDecoder.snapshot: a later turn's K / V row of the
    last one again, the token after it changed, from that row alone)."""
    spec, mt = m.spec, m.mtp
    H, eps, p = spec.hidden, spec.eps, pos.pos
    if em:
        x = ol.load(MO.em_x(m.moe_dev, row, R, H))
        if m.layer(spec.layers - 1).moe:
            x.set(MO.moe_combine_rows(x, spec.moe, m.moe_dev, row))
        if keep:
            ol.store(m.xbuf[row:row + R, :], x)
    else:
        x = ol.load(m.xbuf[row:row + R, :])
    c, s_ = ol.load(m.cos_t[p:p + R, :]), ol.load(m.sin_t[p:p + R, :])
    e = ol.empty([R, H], dense=True)
    for r, g in enumerate(_gather(m, m.embed_q, range(R))):     # the next tokens' rows
        e[r:r + 1, :].set(g)
    cat = ol.empty([R, 2 * H], dense=True)
    cat[:, 0:H].set(rmsnorm(e, ol.load(mt.ge), eps))
    del e
    g = ol.load(m.g_final)          # the final norm row by row (qwen35_rows' hidden rows)
    for r in range(R):
        x[r:r + 1, :].set(rmsnorm(x[r:r + 1, :], g, eps))
    del g
    cat[:, H:2 * H].set(rmsnorm(x, ol.load(mt.gh), eps))
    del x
    y = ol.dot(ol.quantize(cat), mt.fc)                 # [R, H] (one slice: Spec.check)
    del cat
    _attention_rows(y, mt.layer, c, s_, [(0, pos.offset(r)) for r in range(R)], spec, block,
                    gated=True, kv_only=True)


@ol.jit
def qwen35_dn_run(m, li: int, row, rows, C: int, split1: bool = False):
    """The chunked DeltaNet core of DeltaNet layer li (kernels.dn_chunk.chunked;
    docs/deltanet_chunk.md): the image's dnq rows row .. row + rows - 1 (run-
    time values, rows a multiple of C) through the recurrence by chunks of C rows, each pair's
    two heads on its key head, o into the same rows of dno ([C, dv] a head and chunk), the
    slice's states (DeltaNetParts.states: the committed slot, 16-bit with ST16) read at the
    pass's start and written at its end; m.dns its scratch. split1: M1 split too."""
    dn = m.layer(li).dn
    _dn_shared(dn, "core")
    q = m.dnq
    DC.chunked(q.q, q.k, q.v, q.lg, q.b, m.dno, dn.states, m.dns, rows, C, split1, row)


@ol.jit
def qwen35_expert_run(m, li: int, C: int = 2, S: int = 0, exact: bool = False,
                      pipe: int = 2):
    """Expert-major's expert run of MoE layer li (moe.moe_expert_run, docs/offload.md 13.11):
    after the layer's runs over a prefill chunk, each expert its rows chose once, in passes of
    up to C rows (S: a short pass's, docs/offload.md 13.18); the chunk's rows and their entries
    (rows x k) are run arguments; exact: each pass its entries' rows (moe.em_exact); pipe: the
    passes software-pipelined, level 2 or 1 (0: not), where moe_expert_run can (EM_PIPE)."""
    from ..compiler import RunVar
    R, k = m.moe_dev.em_rows, m.spec.moe.k
    MO.moe_expert_run(m.layer(li), m.spec.moe, m.moe_dev, RunVar("rows", R + 1),
                      RunVar("entries", R * k + 1), C, S, exact, pipe)
