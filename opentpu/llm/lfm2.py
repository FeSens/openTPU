"""LFM2 hybrid decoder (Liquid AI LFM2 / LFM2.5, e.g. LFM2.5-230M) on openTPU.

LFM2 stacks two kinds of layers, each followed by Qwen3's pre-norm SwiGLU MLP:
  conv   gated short convolution: in_proj -> B, C, x; y = C * conv(B * x), a causal depthwise
         convolution (3 taps per channel, no bias); out_proj. Its state is the last two rows of
         B * x, kept in fp32 in a 3-slot ring in DRAM (the row of position p in slot p % 3),
         mirrored: each row is stored twice, so the last K rows are contiguous and one
         address register reaches them at a run-time position (_ring_rows).
  attn   GQA attention with RMSNorm on each q and k head, then RoPE: Qwen3's attention
         (qwen3._attention). The heads are 64 wide, half the MXU depth, so the queries and
         the cached K and V rows are padded with zeros to 128 (qwen3._padded); P.V reads only
         V's 64 real rows, and the head outputs are not padded.
LFM2.5-230M has 14 layers, c c A c A c A c A c A c A c: the step kernel runs the repeated
(conv, attn) pair as a hardware loop and the other layers unrolled (`plan`).

Pieces:
  Spec              model dimensions and layer kinds (from a Hugging Face config.json)
  reference_logits  plain numpy forward pass (the math, fp32)
  emulated_logits   float64 decode with openTPU's quantization points (see qwen3)
  Image             per-slice DRAM layout: equal-size layer blocks of either kind, so a
                    hardware loop over (conv, attn) pairs steps one address register
  lfm2_step         the ol kernel for one decode token

  lfm2_rows         R consecutive prompt tokens per device run (chunked prefill,
                    qwen3.Engine.prefill_chunks): the convolution over the chunk's rows and
                    the ring, qwen3's row attention

Weights, activations and the KV cache use Qwen3's W8A8 scheme, and decoding runs on
qwen3.Engine (one sequence: no batched decode).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .. import fp32 as F
from .. import qcache as QC
from .. import quant as Q
from .. import language as ol
from .. import isa as I
from ..compiler import Affine, KVDesc, QTensor, Tensor, current
from ..isasim import Config
from ..kernels.layouts import head_parallel_attention_weights
from ..kernels.lib import rmsnorm
from ..kernels.mlp import _chunk
from ..host.offload import ExpertServer, Layout
from ..runtime import quantize_rows
from . import formats as FM
from . import generate as G
from . import moe as MO
from .qwen3 import (OutTokens, RunPos, RunRows, RunWords, _formats, _inputs, _inputs_rows,
                    _amask, _lookup_alloc, _lookup_build, _lookup_desc, _tok_arg, _tokens_arg,
                    compile_decode)
from .qwen3 import (ATTN_BLOCK, _attention, _attention_rows, _Bump, _fake_q, _fake_w, _lm_head,
                    _lm_head_rows, _mlp, _pv, _qdesc, _tdesc, _v_parts, rope_tables, EmbedHost,
                    fill_logits, step_descriptors)

CONV, ATTN = "conv", "attn"
# A plan with more layer bodies than this runs each MLP's F chunks as a hardware loop
# (kernels.mlp.swiglu_down loop): LFM2-2.6B's 10 bodies with 12 unrolled chunks each would
# fill the 4K-instruction IMEM; LFM2.5-230M's 4 keep them unrolled. OTPU_MLP_UNROLL_BODIES
# overrides it (0: always a loop), e.g. to measure both on the same layers.
MLP_UNROLL_BODIES = int(os.environ.get("OTPU_MLP_UNROLL_BODIES", 8))


# =============================================================================== model spec
@dataclass(frozen=True)
class Spec:
    hidden: int
    kinds: tuple            # CONV or ATTN per layer
    n_q: int
    n_kv: int
    head_dim: int
    ffn: int
    vocab: int
    conv_k: int = 3         # convolution taps (the current row and conv_k - 1 before it)
    eps: float = 1e-5
    theta: float = 1e6
    tied: bool = True
    bos: int = 1
    eos: tuple = (7,)
    moe: MO.MoESpec | None = None   # LFM2-MoE: layers moe.first.. have routed experts
    embed: str = "f32"      # the embedding rows: fp32, or "int8" per D block (as qwen3.Spec:
                            # gathered on the device from the tied int8 head or a table)
    formats: str = ""       # weight formats per kind over the image's wformat (KINDS)
    mix: str = ""           # the recommended mix (wformat "mix": formats.named, MIXES)

    @property
    def layers(self) -> int:
        return len(self.kinds)

    def is_moe(self, i: int) -> bool:
        return self.moe is not None and i >= self.moe.first

    @property
    def lkinds(self) -> tuple:
        """Each layer's kind, and for a MoE model whether it is a MoE layer (what the plan
        loops over: a hardware loop runs layers of one kind)."""
        if self.moe is None:
            return self.kinds
        return tuple((k, self.is_moe(i)) for i, k in enumerate(self.kinds))

    @property
    def rope_dim(self) -> int:
        return self.head_dim

    @staticmethod
    def from_hf(model_dir) -> "Spec":
        c = json.loads((Path(model_dir) / "config.json").read_text())
        ff = c["intermediate_size"]
        if c.get("block_auto_adjust_ff_dim"):        # as Lfm2MLP sizes it
            ff = int(2 * ff / 3)
            if c.get("block_ffn_dim_multiplier") is not None:
                ff = int(c["block_ffn_dim_multiplier"] * ff)
                m = c["block_multiple_of"]
                ff = m * ((ff + m - 1) // m)
        eos = c.get("eos_token_id", 7)
        g = Path(model_dir) / "generation_config.json"
        if g.exists():
            eos = json.loads(g.read_text()).get("eos_token_id", eos)
        rope = c.get("rope_parameters") or {}
        moe = None
        if c.get("model_type") == "lfm2_moe":
            if not c.get("use_expert_bias", True):
                raise ValueError("LFM2-MoE without the expert bias is not supported")
            moe = MO.MoESpec(E=c["num_experts"], k=c["num_experts_per_tok"],
                             ffn=c["moe_intermediate_size"], first=c["num_dense_layers"],
                             norm=c.get("norm_topk_prob", True),
                             scale=c.get("routed_scaling_factor", 1.0))
        return Spec(hidden=c["hidden_size"],
                    kinds=tuple(ATTN if t == "full_attention" else CONV for t in c["layer_types"]),
                    n_q=c["num_attention_heads"], n_kv=c["num_key_value_heads"],
                    head_dim=c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"],
                    ffn=ff, vocab=c["vocab_size"], conv_k=c.get("conv_L_cache", 3),
                    eps=c.get("norm_eps", 1e-5),
                    theta=rope.get("rope_theta", c.get("rope_theta", 1e6)),
                    tied=c.get("tie_word_embeddings", c.get("tie_embedding", True)),
                    bos=c.get("bos_token_id", 1),
                    eos=tuple(eos) if isinstance(eos, list) else (eos,), moe=moe,
                    # a MoE's DRAM beside its layers is expert slots: its table int8
                    embed="int8" if moe is not None else "f32", mix=FM.mix_for(c))

    def check(self, cfg: Config) -> None:
        S, D = cfg.S, cfg.D
        need = [(self.head_dim <= D and self.head_dim % 2 == 0, f"head_dim {self.head_dim} > D"),
                (self.n_q * self.head_dim % D == 0, "n_q * head_dim % D"),
                (self.hidden % (S * D) == 0, f"hidden {self.hidden} % S*D"),
                (self.ffn % (S * D) == 0, f"ffn {self.ffn} % S*D"),
                (self.n_kv % S == 0, f"n_kv {self.n_kv} % S"),
                (self.n_q % self.n_kv == 0, "n_q % n_kv"),
                (self.n_q // self.n_kv <= cfg.MCOLS, "query group larger than MXU columns"),
                (self.vocab % S == 0, f"vocab {self.vocab} % S"),
                (max(self.ffn, self.n_q * self.head_dim, self.hidden) <= cfg.ACT_BLOCKS * D,
                 "an inner dimension exceeds ACT RAM")]
        if self.moe is not None:
            need += [(S == 1, "MoE layers run on one slice"),
                     (self.moe.ffn % D == 0, f"expert width {self.moe.ffn} % D"),
                     (self.moe.k <= self.moe.E, "top-k above the expert count")]
        bad = [m for ok, m in need if not ok]
        if bad:
            raise ValueError("model does not map onto this openTPU config: " + "; ".join(bad))

    def image(self, cfg: Config, cap: int, batch: int = 1, rows: int = 1,
              wformat: str = "int8", head_format: str | None = None,
              lookup: bool | str = False, experts: int | None = None,
              embed_host: bool | None = None, formats: str | None = None) -> "Image":
        return Image(self, cfg, cap, batch, rows, wformat, head_format, lookup, experts,
                     embed_host, formats)


def _place(b, runs, size) -> dict:
    """The layer blocks of the runs (plan's) from b.next on, each run's units one after the
    other (size(key): a block's bytes): {layer: (run base, unit stride, iteration, offset in
    the unit, unit length)}."""
    loc = {}
    for first, unit, reps in runs:
        offs = np.cumsum([0] + [size(k) for k in unit]).tolist()
        us, base = offs[-1], b.next
        for it in range(reps):
            for e in range(len(unit)):
                loc[first + it * len(unit) + e] = (base, us, it, offs[e], len(unit))
        b.next = base + reps * us
    return loc


def _kind(key) -> str:
    """A layer's kind from its Spec.lkinds entry (for a MoE model: kind, is a MoE layer)."""
    return key[0] if isinstance(key, tuple) else key


def plan(kinds) -> list:
    """The layers as runs [(first layer, unit of kinds, repeats)]: the repeated unit covering
    the most layers becomes one hardware loop; the layers before and after it are planned the
    same way, and a layer in no repetition is a run of its own."""
    kinds, n = tuple(kinds), len(kinds)
    best = None                                     # (layers covered, first, unit length)
    for u in range(1, n // 2 + 1):
        for f in range(n - 2 * u + 1):
            r = 1
            while kinds[f + r * u:f + (r + 1) * u] == kinds[f:f + u]:
                r += 1
            if r > 1 and (best is None or r * u > best[0]):
                best = (r * u, f, u)
    if best is None:
        return [(i, (k,), 1) for i, k in enumerate(kinds)]
    cov, f, u = best
    tail = plan(kinds[f + cov:])
    return plan(kinds[:f]) + [(f, kinds[f:f + u], cov // u)] + \
        [(f + cov + a, unit, r) for a, unit, r in tail]


# =============================================================================== reference
def _norm(v, g, eps):
    return (v / np.sqrt(np.mean(v * v, axis=-1, keepdims=True) + eps)) * g


def reference_logits(spec: Spec, W: dict, tokens) -> np.ndarray:
    """fp32 numpy forward of the whole sequence (causal); returns logits [T, vocab]."""
    tokens = list(tokens)
    T, d, G, H, K = len(tokens), spec.head_dim, spec.n_q // spec.n_kv, spec.hidden, spec.conv_k
    x = W["model.embed_tokens.weight"][tokens].astype(np.float32)
    cs = [rope_tables(spec, p) for p in range(T)]
    cos = np.stack([c for c, _ in cs])[:, None, :]
    sin = np.stack([s for _, s in cs])[:, None, :]

    def rot(v):
        v1, v2 = v[..., :d // 2], v[..., d // 2:]
        return np.concatenate([v1 * cos - v2 * sin, v2 * cos + v1 * sin], axis=-1)

    mask = np.triu(np.full((T, T), -np.inf, np.float32), 1)
    for i, kind in enumerate(spec.kinds):
        p = f"model.layers.{i}."
        h = _norm(x, W[p + "operator_norm.weight"], spec.eps)
        if kind == CONV:
            B, C, xx = np.split(h @ W[p + "conv.in_proj.weight"].T, 3, axis=1)
            bx = np.concatenate([np.zeros((K - 1, H), np.float32), B * xx])
            w = W[p + "conv.conv.weight"][:, 0, :]              # [H, K], w[:, K-1]: current
            conv = sum(bx[k:k + T] * w[:, k] for k in range(K))
            x = x + (C * conv) @ W[p + "conv.out_proj.weight"].T
        else:
            a = p + "self_attn."
            q = (h @ W[a + "q_proj.weight"].T).reshape(T, spec.n_q, d)
            k = (h @ W[a + "k_proj.weight"].T).reshape(T, spec.n_kv, d)
            v = (h @ W[a + "v_proj.weight"].T).reshape(T, spec.n_kv, d)
            q = rot(_norm(q, W[a + "q_layernorm.weight"], spec.eps))
            k = rot(_norm(k, W[a + "k_layernorm.weight"], spec.eps))
            o = np.zeros((T, spec.n_q, d), np.float32)
            for hq in range(spec.n_q):
                s = q[:, hq] @ k[:, hq // G].T / math.sqrt(d) + mask
                s = np.exp(s - s.max(axis=1, keepdims=True))
                o[:, hq] = (s / s.sum(axis=1, keepdims=True)) @ v[:, hq // G]
            x = x + o.reshape(T, -1) @ W[a + "out_proj.weight"].T
        h = _norm(x, W[p + "ffn_norm.weight"], spec.eps)
        if spec.is_moe(i):
            x = x + _moe_reference(h, W, p, spec.moe)
            continue
        g = h @ W[p + "feed_forward.w1.weight"].T
        u = h @ W[p + "feed_forward.w3.weight"].T
        x = x + ((g / (1 + np.exp(-g))) * u) @ W[p + "feed_forward.w2.weight"].T
    x = _norm(x, W["model.embedding_norm.weight"], spec.eps)
    head = W["model.embed_tokens.weight"] if spec.tied else W["lm_head.weight"]
    return x @ head.T


def _moe_reference(h, W, p: str, mo) -> np.ndarray:
    """The routed experts' sum for the rows of h [T, H] (fp32, Hugging Face's math)."""
    f = p + "feed_forward."
    logits = h @ W[f + "gate.weight"].T
    y = np.zeros_like(h)
    for t in range(h.shape[0]):
        ids, w = MO.route(logits[t], W[f + "expert_bias"], mo)
        for e, we in zip(ids, w):
            ep = f"{f}experts.{e}."
            g, u = h[t] @ W[ep + "w1.weight"].T, h[t] @ W[ep + "w3.weight"].T
            y[t] += np.float32(we) * (((g / (1 + np.exp(-g))) * u) @ W[ep + "w2.weight"].T)
    return y


KINDS = ("attn", "conv", "mlp", "gateup", "down", "head")    # a formats string's weight kinds


def weight_kind(n: str) -> tuple:
    """(kind, layer) of checkpoint weight `n` (KINDS: the head, or a layer's attention,
    convolution in / out, MLP w1 / w3 or w2 projection; a MoE's experts are the image's
    wformat: "experts")."""
    if not n.startswith("model.layers."):
        return "head", 0
    i = int(n.split(".")[2])
    if ".experts." in n:
        return "experts", i
    return ("conv" if ".conv." in n else "attn" if ".self_attn." in n else
            "down" if n.endswith(".w2.weight") else "gateup"), i


def emulated_logits(spec: Spec, W: dict, tokens, D: int = 128, wformat: str = "int8",
                    head_format: str | None = None, routes: list | None = None,
                    formats: str | None = None) -> np.ndarray:
    """float64 decode with openTPU's quantization points and none of its rounding (as
    qwen3.emulated_logits); a zero-padded K or q block quantizes like its head_dim values."""
    d, G, H, K = spec.head_dim, spec.n_q // spec.n_kv, spec.hidden, spec.conv_k
    Wq: dict = {}

    head = "model.embed_tokens.weight" if spec.tied else "lm_head.weight"
    wformat, formats = FM.named(spec, wformat, formats)
    fmt = FM.resolver(formats, KINDS, spec.formats, wformat, head_format)

    def w(n):
        if n not in Wq:        # the weight formats as in Image (wformat, head_format, formats)
            Wq[n] = _fake_w(W[n], D, fmt(*weight_kind(n)))
        return Wq[n]

    def norm(v, g):
        return _norm(v, g, spec.eps)

    Kc = {i: [] for i, k in enumerate(spec.kinds) if k == ATTN}
    Vc = {i: [] for i in Kc}
    state = {i: [np.zeros(H)] * (K - 1) for i, k in enumerate(spec.kinds) if k == CONV}
    out = []
    for pos, tk in enumerate(tokens):
        x = np.asarray(W["model.embed_tokens.weight"][tk], np.float64)
        if spec.embed == "int8":                    # the int8 embedding rows (Spec.embed)
            x = _fake_q(x, D)
        c, s = rope_tables(spec, pos)

        def rot(v):
            v1, v2 = v[..., :d // 2], v[..., d // 2:]
            return np.concatenate([v1 * c - v2 * s, v2 * c + v1 * s], -1)

        for i, kind in enumerate(spec.kinds):
            p = f"model.layers.{i}."
            h = _fake_q(norm(x, W[p + "operator_norm.weight"]), D)
            if kind == CONV:
                B, C, xx = np.split(w(p + "conv.in_proj.weight") @ h, 3)
                win = state[i] + [B * xx]                    # the last K rows of B * x
                state[i] = win[1:]
                wc = W[p + "conv.conv.weight"][:, 0, :]
                y = C * sum(win[k] * wc[:, k] for k in range(K))
                x = x + w(p + "conv.out_proj.weight") @ _fake_q(y, D)
            else:
                a = p + "self_attn."
                q = (w(a + "q_proj.weight") @ h).reshape(spec.n_q, d)
                k = (w(a + "k_proj.weight") @ h).reshape(spec.n_kv, d)
                v = (w(a + "v_proj.weight") @ h).reshape(spec.n_kv, d)
                q = rot(norm(q, W[a + "q_layernorm.weight"]))
                k = rot(norm(k, W[a + "k_layernorm.weight"]))
                Kc[i].append(_fake_q(k, min(d, D)))
                Vc[i].append(_v_parts(v))
                Kh = np.stack(Kc[i], 1)
                Vq, Vs = (np.stack(z, 1) for z in zip(*Vc[i]))
                o = np.zeros((spec.n_q, d))
                for hq in range(spec.n_q):
                    sc = Kh[hq // G] @ _fake_q(q[hq] / math.sqrt(d), min(d, D))
                    pp = np.exp(sc - sc.max())
                    o[hq] = _pv(pp, Vq[hq // G], Vs[hq // G], D) / pp.sum()
                x = x + w(a + "out_proj.weight") @ _fake_q(o.reshape(-1), D)
            h = _fake_q(norm(x, W[p + "ffn_norm.weight"]), D)
            if spec.is_moe(i):
                f = p + "feed_forward."
                if f + "gate.weight" not in Wq:     # the router is int8 in every format
                    Wq[f + "gate.weight"] = _fake_w(W[f + "gate.weight"], D, "int8")
                lg = Wq[f + "gate.weight"] @ h
                ids, wts = MO.route(lg, W[f + "expert_bias"], spec.moe)
                if routes is not None:              # (token, layer, ids, the choice's margin)
                    sel = np.sort(1 / (1 + np.exp(-lg)) + W[f + "expert_bias"])[::-1]
                    k = spec.moe.k
                    routes.append((len(out), i, ids, sel[k - 1] - sel[k] if k < len(sel)
                                   else np.inf))
                for e, we in zip(ids, wts):
                    ep = f"{f}experts.{e}."
                    g, u = w(ep + "w1.weight") @ h, w(ep + "w3.weight") @ h
                    x = x + we * (w(ep + "w2.weight") @ _fake_q((g / (1 + np.exp(-g))) * u, D))
                continue
            g = w(p + "feed_forward.w1.weight") @ h
            u = w(p + "feed_forward.w3.weight") @ h
            x = x + w(p + "feed_forward.w2.weight") @ _fake_q((g / (1 + np.exp(-g))) * u, D)
        out.append(w(head) @ _fake_q(norm(x, W["model.embedding_norm.weight"]), D))
    return np.array(out)


# =============================================================================== DRAM image
class Image(EmbedHost):
    """Per-slice DRAM layout of an LFM2 model. Every slice uses the same addresses.

    [ I/O: x_in, cos, sin | final norm | logits ] [ layer 0 block ] ... [ layer L-1 block ]
    [ LM head rows of this slice ]. A layer block's layout and size are its kind's in its
    formats group (a layer's conv, attention, gate / up and down formats: Image.lf); both kinds
    start with the norms and this slice's MLP rows, at the same offsets in a group; a conv
    block then holds the taps, the state ring and this slice's rows of in_proj (its channels of
    B, C and x) and out_proj; an attention block holds the q/k norms, the projections and this
    slice's KV heads with room for `cap` tokens. The I/O area holds `rows` token rows (x, cos,
    sin, logits) for chunked prefill. Weight formats as qwen3.Image (`formats` over the KINDS
    of this file, per layer range; a MoE's experts in `wformat`, its router int8, its dense
    layers' MLPs one format): the layers run as the runs of plan over their (kind, formats
    group) keys, each run's blocks one after the other.
    """

    prompt_words = ("ringo",)    # a prompt run reads it itself (_ring_store, prefill.run)

    def __init__(self, spec: Spec, cfg: Config, cap: int, batch: int = 1, rows: int = 1,
                 wformat: str = "int8", head_format: str | None = None, lookup: bool | str = False,
                 experts: int | None = None, embed_host: bool | None = None,
                 formats: str | None = None):
        spec.check(cfg)
        if batch != 1:
            raise ValueError("LFM2 runs one sequence: batch=1")
        if cap % cfg.D:
            raise ValueError("KV capacity must be a multiple of D")
        S, D = cfg.S, cfg.D
        H, d, F_, K = spec.hidden, spec.head_dim, spec.ffn, spec.conv_k
        wformat, formats = FM.named(spec, wformat, formats)
        fmt = FM.resolver(formats, KINDS, spec.formats, wformat, head_format)
        # each layer's formats group: conv, attention, gate / up, down where it is (the
        # layers of a group share a block size and the common part's offsets, as one group did)
        self.lf = tuple((fmt(CONV, i), fmt(ATTN, i), fmt("gateup", i), fmt("down", i))
                        for i in range(spec.layers))
        if spec.moe is not None:    # the dense layers' MLP region: one layout
            FM.uniform(fmt, dict.fromkeys(("gateup", "down"), range(spec.moe.first)))
        self.wformat, self.head_format = wformat, fmt("head")
        self.formats = _formats(spec, formats)
        rb = lambda k, f: Q.row_bytes(k, f, D)                          # noqa: E731
        self.spec, self.cfg, self.cap, self.batch, self.rows = spec, cfg, cap, 1, rows
        self.dk = -(-d // D) * D                        # cached K row / query width
        self.nq_loc, self.nkv_loc = spec.n_q // S, spec.n_kv // S
        self.h_loc, self.f_loc, self.v_loc = H // S, F_ // S, spec.vocab // S
        mo = spec.moe
        b = _Bump()
        R = rows
        self.io = {"x": b.alloc(4 * H * R), "cos": b.alloc(2 * d * R), "sin": b.alloc(2 * d * R),
                   "gf": b.alloc(4 * H), "logits": b.alloc(4 * spec.vocab * R)}
        self.layer0 = b.next
        mlp = {"wg": (self.f_loc, H), "wu": (self.f_loc, H)}
        self.mats = {CONV: {"win": (3 * self.h_loc, H), "wout": (self.h_loc, H)},
                     ATTN: {"wq": (self.nq_loc * d, H), "wk": (self.nkv_loc * d, H),
                            "wv": (self.nkv_loc * d, H), "wo": (self.h_loc, spec.n_q * d)}}
        if mo is None:
            for kind in (CONV, ATTN):
                self.mats[kind].update(mlp)
        # formats group -> its block layouts (_layout); layer 0's those of the image (and of
        # a MoE's dense MLP region)
        self.layouts = {g: self._layout(g, mlp, rb) for g in dict.fromkeys(self.lf)}
        g0 = self.layouts[self.lf[0]]
        self.lofs, self.mf, self.dchunk = g0.lofs, g0.mf, g0.dchunk
        if mo is not None:
            self.mlp_ofs, self.DS = g0.mlp_ofs, g0.DS
        # the runs (plan): a layer's key is its kind and formats group
        self.keys = tuple(zip(spec.lkinds, self.lf))
        self.plan = plan(self.keys)
        self.mlp_loop = sum(len(u) for _, u, _ in self.plan) > MLP_UNROLL_BODIES
        self.loc = _place(b, self.plan, lambda k: self.layouts[k[1]].size[_kind(k[0])])
        dk = self.dk
        n_attn = spec.kinds.count(ATTN)
        head = cap * dk + 4 * cap * (dk // D) + dk * cap + 4 * cap
        self.kv_bytes = (n_attn * self.nkv_loc * head                   # KV cache and
                         + (spec.layers - n_attn) * 8 * K * self.h_loc)  # conv state, per sequence
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
                                    rows=rows) if lookup else {}
        self.choices = {"embed_host": self.embed_host,          # (the compile worker's
                        "formats": self.formats}                # image)
        self.offload = None
        if mo is not None:          # the dense MLPs, then path (a)'s words and expert slots
            self.dense0 = (b.next + 4095) // 4096 * 4096
            b.next = self.dense0 + mo.first * self.DS
            self.fmt = MO.ExpertFormat(H, mo.ffn, D, wformat)
            n = mo.E if experts is None else experts
            if not mo.k <= n <= mo.E:
                raise ValueError(f"{n} expert slots per layer: from top-k {mo.k} to {mo.E}")
            self.offload = Layout.build((b.next + 4095) // 4096 * 4096, mo.E, mo.k,
                                        [n] * (spec.layers - mo.first), self.fmt.nbytes)
            b.next = self.offload.end
        self.nbytes = b.next
        if self.nbytes > cfg.DRAM_BYTES:
            raise MemoryError(f"model image needs {self.nbytes / 2**20:.0f} MiB per slice, "
                              f"DRAM_BYTES is {cfg.DRAM_BYTES / 2**20:.0f} MiB")

    def _layout(self, g: tuple, mlp: dict, rb) -> SimpleNamespace:
        """The layer block layouts of formats group g (conv, attention, gate / up, down): each
        projection's format (mf), W_down's chunk, the conv and attention blocks' offsets (the
        norms and the dense MLP first, at the same offsets in both) and sizes (size)."""
        spec, cfg, cap = self.spec, self.cfg, self.cap
        D, H, d, F_, K = cfg.D, spec.hidden, spec.head_dim, spec.ffn, spec.conv_k
        fc, fa, fg, fd = g
        mf = {"win": fc, "wout": fc, "wq": fa, "wk": fa, "wv": fa, "wo": fa, "wg": fg,
              "wu": fg, "wd": fd}
        mo = spec.moe
        lb = _Bump()                                    # offsets inside one layer block
        common = {"g_in": lb.alloc(4 * H), "g_post": lb.alloc(4 * H)}
        dchunk = _chunk(self.f_loc, D, D if fd == "int8" else 2 * D)
        # the dense MLP: in every layer block, or for a MoE model (whose MoE layers have none)
        # in a region of its own, one MLP per dense layer (layers 0 .. moe.first - 1)
        mb = lb if mo is None else _Bump()
        dense = {name: (mb.alloc(n * rb(k, mf[name])), mb.alloc(4 * n * (k // D)))
                 for name, (n, k) in mlp.items()}
        dense["wd"] = [(mb.alloc(self.h_loc * rb(dchunk, fd)),
                        mb.alloc(4 * self.h_loc * (dchunk // D)))
                       for _ in range(F_ // dchunk)]
        ns = SimpleNamespace(mf=mf, dchunk=dchunk)
        if mo is None:
            common.update(dense)
        else:                       # the router (int8 in every format), its bias, j * E
            ns.mlp_ofs, ns.DS = dense, (mb.next + 4095) // 4096 * 4096
            common.update(router=(lb.alloc(mo.E * H), lb.alloc(4 * mo.E * (H // D))),
                          ebias=lb.alloc(4 * mo.E), gbase=lb.alloc(4))
        cb = _Bump(lb.next)
        conv = dict(common, taps=cb.alloc(4 * K * self.h_loc),
                    state=cb.alloc(4 * 2 * K * self.h_loc))         # mirrored (_ring_rows)
        ab = _Bump(lb.next)
        attn = dict(common, qn=ab.alloc(4 * d), kn=ab.alloc(4 * d))
        for kind, bump, L in ((CONV, cb, conv), (ATTN, ab, attn)):
            for name, (n, k) in self.mats[kind].items():
                if name not in L:
                    L[name] = (bump.alloc(n * rb(k, mf[name])), bump.alloc(4 * n * (k // D)))
        dk = self.dk
        attn["kv"] = [{"k": ab.alloc(cap * dk), "ks": ab.alloc(4 * cap * (dk // D)),
                       "vt": ab.alloc(dk * cap), "vs": ab.alloc(4 * cap)}
                      for _ in range(self.nkv_loc)]
        ns.lofs = {CONV: conv, ATTN: attn}
        ns.size = {CONV: (cb.next + 4095) // 4096 * 4096, ATTN: (ab.next + 4095) // 4096 * 4096}
        return ns

    def _off(self, li, it=None) -> Affine:
        """The block address of layer li (static), or of element li of its run's unit at
        iteration `it` (a loop variable)."""
        base, us, i, o, _ = self.loc[li]
        return Affine(base + o) + Affine.of(i if it is None else it) * us

    def _idx(self, li, it=None) -> Affine:
        """Layer li's index (static), or the index of element li of its run's unit at
        iteration `it`."""
        base, us, i, o, n = self.loc[li]
        return Affine(li) if it is None else Affine(li - i * n) + Affine.of(it) * n

    # ---- contents
    def build(self, W: dict) -> list[np.ndarray]:
        """DRAM images (one per slice) with every weight quantized in place, KV cache and
        convolution state empty."""
        spec, cfg = self.spec, self.cfg
        S, D, d, H, n = cfg.S, cfg.D, spec.head_dim, spec.hidden, self.h_loc
        imgs = [np.zeros(self.nbytes, np.uint8) for _ in range(S)]

        def put(s, addr, a):
            v = np.ascontiguousarray(a).view(np.uint8).reshape(-1)
            imgs[s][addr:addr + v.size] = v

        def put_q(addr_pair, parts, fmt):
            for s, p in enumerate(parts):
                q, sc = QC.quantize_mxu(p, fmt, D)
                put(s, addr_pair[0], q)
                put(s, addr_pair[1], sc)

        def rows(a, k):
            return [a[s * k:(s + 1) * k] for s in range(S)]

        def f32(a):
            return F.ftz(np.asarray(a, np.float32))

        for s in range(S):
            put(s, self.io["gf"], f32(W["model.embedding_norm.weight"]))
        for i, kind in enumerate(spec.kinds):
            p, base, g = f"model.layers.{i}.", self._off(i).const, self.layouts[self.lf[i]]
            mf = g.mf
            Lo = {k: (tuple(base + x for x in v) if isinstance(v, tuple) else
                      (base + v if isinstance(v, int) else v)) for k, v in g.lofs[kind].items()}
            for s in range(S):
                put(s, Lo["g_in"], f32(W[p + "operator_norm.weight"]))
                put(s, Lo["g_post"], f32(W[p + "ffn_norm.weight"]))
            if kind == CONV:
                B, C, X = np.split(W[p + "conv.in_proj.weight"], 3)
                put_q(Lo["win"], [np.concatenate([B[s * n:(s + 1) * n], C[s * n:(s + 1) * n],
                                                  X[s * n:(s + 1) * n]]) for s in range(S)],
                      mf["win"])
                put_q(Lo["wout"], rows(W[p + "conv.out_proj.weight"], n), mf["wout"])
                taps = W[p + "conv.conv.weight"][:, 0, :].T            # [K, H]
                for s in range(S):
                    put(s, Lo["taps"], f32(taps[:, s * n:(s + 1) * n]))
            else:
                a = p + "self_attn."
                for s in range(S):
                    put(s, Lo["qn"], f32(W[a + "q_layernorm.weight"]))
                    put(s, Lo["kn"], f32(W[a + "k_layernorm.weight"]))
                wq, wk, wv, wo = head_parallel_attention_weights(
                    W[a + "q_proj.weight"], W[a + "k_proj.weight"], W[a + "v_proj.weight"],
                    W[a + "out_proj.weight"], spec.n_q, spec.n_kv, d, S)
                put_q(Lo["wq"], rows(wq, self.nq_loc * d), mf["wq"])
                put_q(Lo["wk"], rows(wk, self.nkv_loc * d), mf["wk"])
                put_q(Lo["wv"], rows(wv, self.nkv_loc * d), mf["wv"])
                put_q(Lo["wo"], rows(wo, n), mf["wo"])
            if spec.is_moe(i):
                f, mo = p + "feed_forward.", spec.moe
                put_q(Lo["router"], [W[f + "gate.weight"]], "int8")
                put(0, Lo["ebias"], f32(W[f + "expert_bias"]))
                put(0, Lo["gbase"], f32([(i - mo.first) * mo.E]))
                continue
            mb, mofs = base, g.lofs[kind]
            if spec.moe is not None:
                mb, mofs = self.dense0 + i * self.DS, self.mlp_ofs
            for name, hf in (("wg", "w1"), ("wu", "w3")):
                put_q(tuple(mb + x for x in mofs[name]),
                      rows(W[p + f"feed_forward.{hf}.weight"], self.f_loc), mf[name])
            C_ = g.dchunk
            for j, pair in enumerate(mofs["wd"]):
                put_q((mb + pair[0], mb + pair[1]),
                      [r[:, j * C_:(j + 1) * C_] for r in rows(W[p + "feed_forward.w2.weight"], n)],
                      mf["wd"])
        head = W["model.embed_tokens.weight"] if spec.tied else W["lm_head.weight"]
        put_q(self.head, rows(head, self.v_loc), self.head_format)
        if self.lookup:
            _lookup_build(put, S, W, spec, self.cap, self.lookup)
        return imgs

    # ---- path (a): the expert pool and its server (docs/offload.md)
    def expert(self, W: dict, g: int) -> np.ndarray:
        """Global expert g (MoE layer g // E, expert g % E) in its slot's bytes."""
        mo = self.spec.moe
        p = f"model.layers.{mo.first + g // mo.E}.feed_forward.experts.{g % mo.E}."
        return self.fmt.pack(W[p + "w1.weight"], W[p + "w3.weight"], W[p + "w2.weight"])

    def serve(self, W: dict, backend, pool_file=None) -> ExpertServer:
        """The host's expert server on the backend's DRAM (moe.serve)."""
        return MO.serve(self.offload, lambda g: self.expert(W, g), backend, pool_file)

    # ---- programs
    def compile_decode(self, blocks: int, lo: int, block: int = ATTN_BLOCK):
        """(programs, run_args): lfm2_step at a run-time position (qwen3.compile_decode); the
        convolutions need lo >= conv_k - 1 (every tap of the state ring is a past token)."""
        return compile_decode(self, lfm2_step, blocks, lo, block)

    def compile_generate(self, blocks: int, lo: int, block: int = ATTN_BLOCK,
                         chain: bool = True, samp=None, debug: bool = False,
                         part: int | None = None) -> list:
        """The decode loop on the device for bucket `blocks` (lfm2_step in it, generate.py)."""
        return G.compile_generate(self, lfm2_step, blocks, lo, block, chain, samp, debug, part)

    def compile_step(self, pos: int, block: int = ATTN_BLOCK, tok: int | None = None) -> list:
        """One program per slice: the decode token at position `pos` (lfm2_step)."""
        return [lfm2_step.trace(self.cfg, s, {"m": step_descriptors(self, s), "pos": pos,
                                              "block": block, **_tok_arg(self, tok)}).finish()
                for s in range(self.cfg.S)]

    def compile_rows(self, rows, logit_rows, block: int = ATTN_BLOCK, tokens=None) -> list:
        """One program per slice: consecutive positions of the sequence at once (lfm2_rows)."""
        if self.spec.moe is not None and len(rows) > 1:
            raise ValueError("a MoE model runs one row per program (its MoE block routes one "
                             "token)")
        if len(rows) > self.rows:
            raise ValueError(f"{len(rows)} rows, the image's I/O area holds {self.rows}")
        if any(r != (0, rows[0][1] + i) for i, r in enumerate(rows)):
            raise ValueError("LFM2 rows must be consecutive positions of sequence 0")
        return [lfm2_rows.trace(self.cfg, s, {"m": self.descriptors(s), "p0": rows[0][1],
                                              "R": len(rows), "logit_rows": list(logit_rows),
                                              "block": block,
                                              **_tokens_arg(self, tokens, rows)}).finish()
                for s in range(self.cfg.S)]

    def compile_prompt_run(self, blocks: int, R: int, kind: str, block: int = ATTN_BLOCK,
                           p0: int | None = None):
        """lfm2_prompt_run's (programs, run_args) (docs/prefill.md), a program per slice: R
        rows of a prompt at a run-time position of bucket `blocks` (from conv_k - 1: the state's
        tpos and ring words), or at the compile-time position p0 (the rows before conv_k - 1;
        no run_args)."""
        if self.spec.moe is not None:
            raise ValueError("a MoE model runs one row per program (its MoE block routes one "
                             "token)")
        if not self.lookup:
            raise ValueError("a prompt run needs lookup tables (lookup=True)")
        if kind not in ("P", "L"):
            raise ValueError(f"prompt run kind {kind!r}")
        if R > self.rows:
            raise ValueError(f"{R} rows, the image's I/O area holds {self.rows}")
        pos = p0 if p0 is not None else \
            RunRows(blocks, block, max((blocks - 1) * block, self.spec.conv_k - 1),
                    self.lookup["zmask"], self.cap, R, 0, _amask(self.lookup))
        bs = [lfm2_prompt_run.trace(self.cfg, s, {"m": self.descriptors(s), "pos": pos, "R": R,
                                                  "kind": kind, "block": block})
              for s in range(self.cfg.S)]
        return [b.finish() for b in bs], list(bs[0].run_args)

    # ---- kernel descriptors
    def descriptors(self, sid: int) -> SimpleNamespace:
        spec, cfg = self.spec, self.cfg
        D, d, H, K, n = cfg.D, spec.head_dim, spec.hidden, spec.conv_k, self.h_loc

        def layer(li, it=None):
            """Descriptors of layer `li` (static), or of element li of its run's unit at
            iteration `it` (a hardware-loop variable). Its kind (for a MoE model: kind, is a MoE
            layer; Spec.lkinds) and formats group from self.keys."""
            kind, g = self.keys[li]
            moe = False
            if isinstance(kind, tuple):
                kind, moe = kind
            g = self.layouts[g]
            off = self._off(li, it)
            lofs, mf = g.lofs[kind], g.mf
            ns = SimpleNamespace(g_in=Tensor(off + lofs["g_in"], (H,), (1,)),
                                 g_post=Tensor(off + lofs["g_post"], (H,), (1,)), moe=moe,
                                 mlp_loop=self.mlp_loop, kind=kind)
            for name, (r, k) in self.mats[kind].items():
                da, sa = lofs[name]
                fm = mf[name]
                setattr(ns, name, QTensor(off + da, off + sa, (r, k), Q.row_bytes(k, fm, D),
                                          4 * (k // D), D, wf=Q.mxu_wf(fm)))
            if moe:
                E = spec.moe.E
                da, sa = lofs["router"]
                ns.router = QTensor(off + da, off + sa, (E, H), H, 4 * (H // D), D)
                ns.ebias = Tensor(off + lofs["ebias"], (E,), (1,))
                ns.gbase = Tensor(off + lofs["gbase"], (1,), (1,))
            else:
                mofs, moff = lofs, off
                if spec.moe is not None:            # the dense layers' MLP region
                    mofs = self.mlp_ofs
                    moff = Affine.of(self.dense0) + self._idx(li, it) * self.DS
                    for name in ("wg", "wu"):
                        da, sa = mofs[name]
                        fm = mf[name]
                        setattr(ns, name, QTensor(moff + da, moff + sa, (self.f_loc, H),
                                                  Q.row_bytes(H, fm, D), 4 * (H // D), D,
                                                  wf=Q.mxu_wf(fm)))
                C = g.dchunk
                fm = mf["wd"]
                wf = Q.mxu_wf(fm)
                rc = Q.row_bytes(C, fm, D)
                parts = tuple(QTensor(moff + da, moff + sa, (n, C), rc, 4 * (C // D), D, wf=wf)
                              for da, sa in mofs["wd"])
                ns.wd = QTensor(parts[0].data, parts[0].scale, (n, spec.ffn), rc, 4 * (C // D),
                                D, parts=parts, pw=C, wf=wf)
            if kind == CONV:
                ns.taps = Tensor(off + lofs["taps"], (K, n), (n, 1))
                ns.state = Tensor(off + lofs["state"], (2 * K, n), (n, 1))     # _ring_rows
            else:
                ns.qn = Tensor(off + lofs["qn"], (d,), (1,))
                ns.kn = Tensor(off + lofs["kn"], (d,), (1,))
                ns.kv = KVDesc({sid + j * cfg.S: {k: off + v for k, v in r.items()}
                                for j, r in enumerate(lofs["kv"])}, self.cap, self.dk, D, cfg.S,
                               sid, dv=d)
                ns.kvs = [ns.kv]
            return ns

        dev = None
        if self.offload is not None:
            L = self.offload
            dev = SimpleNamespace(mbox=L.mbox, served=L.served, answer=L.answer, dir=L.dir,
                                  tag=L.tag, fmt=self.fmt)
        return SimpleNamespace(
            spec=spec, layer=layer, plan=self.plan, moe_dev=dev,
            x=_tdesc(self.io["x"], (1, H)), cos=_tdesc(self.io["cos"], (d // 2,)),
            sin=_tdesc(self.io["sin"], (d // 2,)), g_final=_tdesc(self.io["gf"], (H,)),
            logits=_tdesc(self.io["logits"], (1, spec.vocab)),
            xr=_tdesc(self.io["x"], (self.rows, H)),
            cosr=_tdesc(self.io["cos"], (self.rows, d // 2)),
            sinr=_tdesc(self.io["sin"], (self.rows, d // 2)),
            logitsr=_tdesc(self.io["logits"], (self.rows, spec.vocab)),
            head=_qdesc(*self.head, self.v_loc, H, D, self.head_format), v_loc=self.v_loc,
            **_lookup_desc(self.lookup, spec, self.cap))


# =============================================================================== kernel
def _ring_rows(K: int, p):
    """The state ring's rows: (the row holding position p - j, j = 1 .. K-1; the rows B * x of
    position p is stored to). 2K rows: slot s at rows 1 + s and 1 + s + K (the mirror, not
    kept for slot K-1), row 0 a scratch row. Rows w + 1 .. w + K, w = (p + 1) mod K, hold
    positions p-K+1 .. p in order, so at a run-time position (RunPos: its `ring` is w) every
    row is one register plus a constant; a position given as an int reads the slot rows."""
    if isinstance(p, RunPos):
        w = p.ring
        return [w + K - j for j in range(1, K)], [w, w + K]
    return ([1 + (p - j) % K for j in range(1, K)],
            [1 + p % K] + ([1 + p % K + K] if p % K < K - 1 else []))


def _conv(x, lw, pos: int, spec: Spec):
    """x + out_proj(C * conv(B * x)) for one token, this slice's channels; returns the new
    residual (replicated on every slice). B * x is stored to the state ring (_ring_rows); the
    rows of the K - 1 positions before (those >= 0) are read back from it."""
    K = spec.conv_k
    n = lw.taps.shape[1]
    taps = ol.load(lw.taps)                     # [K, n]; row K-1 weighs the current token
    if isinstance(pos, RunPos) and pos.lo < K - 1:
        raise ValueError(f"a run-time position needs p >= {K - 1} (conv taps)")
    prev_rows, put = _ring_rows(K, pos)
    past = K - 1 if isinstance(pos, RunPos) else min(K - 1, pos)
    prev = [(ol.load(lw.state[r:r + 1, :]), taps[K - 1 - j:K - j, :])
            for j, r in zip(range(1, past + 1), prev_rows)]  # loaded while in_proj streams
    xs = ol.quantize(rmsnorm(x, ol.load(lw.g_in), spec.eps))
    bcx = ol.dot(xs, lw.win)                    # [1, 3n]: B, C, x of this slice's channels
    bx = bcx[:, 0:n] * bcx[:, 2 * n:3 * n]
    for r in put:
        ol.store(lw.state[r:r + 1, :], bx)
    y = bx * taps[K - 1:K, :]
    for s, t in prev:
        y = y + s * t
    y_all = ol.all_gather(y * bcx[:, n:2 * n])  # [1, H]
    return x + ol.all_gather(ol.dot(y_all, lw.wout))


@ol.jit
def lfm2_step(m, pos: int, block: int = ATTN_BLOCK, tok: int | None = None):
    """One decode token at position `pos`: x (the token's embedding) -> logits.

    Each run of m.plan with repeats is a hardware loop over its unit of layers; the others are
    unrolled. Conv layers update their state ring, attention layers append K/V at `pos` and
    attend over positions 0..pos. Logits for this slice's vocabulary rows go to m.logits.
    """
    fill_logits(m)
    spec = m.spec
    x, c, s_ = _inputs(m, pos, tok)

    def layer(li, it):
        lw = m.layer(li, it)
        if lw.kind == CONV:
            x.set(_conv(x, lw, pos, spec))
        else:
            x.set(_attention(x, lw, c, s_, pos, spec, block))
        if lw.moe:
            x.set(MO.moe_ffn(x, lw, spec.moe, m.moe_dev, spec.eps))
        else:
            x.set(_mlp(x, lw, spec))

    run_layers(m.plan, layer)
    _lm_head(x, m, spec)


def _ring_store(lw, last, ringo: int, K: int, n: int) -> None:
    """The state ring's rows 1 .. 2K-1 (_ring_rows) from `last`, the B * x rows of the last K
    positions q = p'-K .. p'-1 (p' the next position): row y holds slot (y - 1) mod K, its
    position's row last[(y - 1 - p') mod K]. A run-time p' makes that a rotation: the K rows
    and again their first K - 1 go to a TMEM window, and row y (and its mirror y + K) is the
    window's row o + y - 1 (o + y - K - 1), o = -p' mod K (the state word at TMEM `ringo`,
    prefill.words). One ST per row from a register offset, to fixed DRAM rows; every copy is
    rewritten as R steps would leave it (row 0, the scratch row, not)."""
    b = current()
    win = ol.empty([2 * K - 1, n])
    win[0:K, :].set(last)
    win[K:2 * K - 1, :].set(last[0:K - 1, :])
    r = b.scratch()
    b.emit(I.rld(r, ringo, mul=win.rs, comment="the ring's rotation"))
    for y in range(1, 2 * K):
        ra, imm = b.addr(Affine.of(lw.state.base) + 4 * n * y)
        j = y - 1 if y <= K else y - K - 1
        b.emit(I.st(imm, win.base + j * win.rs, n, ra=ra, rb=r, comment="ring row"))
    b.unscratch(r)
    del win


def _conv_rows(x, lw, p0: int, spec: Spec):
    """_conv for R consecutive positions p0 .. p0+R-1 at once: in_proj streams once for the R
    rows (B and x first, C after the convolution: TMEM would not hold all three), and row r
    convolves over the rows before it in the chunk and the ring (positions
    before p0). The same products and sums per row as _conv, in the same order, so the result
    is bit-identical; the ring ends up holding the chunk's last K rows, each in its slot. p0 a
    RunRows (a prompt run, docs/prefill.md): the ring's rows by its ring word, and the whole ring
    stored from the last K rows (_ring_store)."""
    K, R = spec.conv_k, x.rows
    n = lw.taps.shape[1]
    run = isinstance(p0, RunRows)
    taps = ol.load(lw.taps)                     # [K, n]; row K-1 weighs the current token
    E = ol.empty([K - 1 + R, n])                # B * x of positions p0-K+1 .. p0+R-1
    prev = zip(range(1, K), _ring_rows(K, p0)[0]) if run else \
        ((j, 1 + (p0 - j) % K) for j in range(1, min(K, p0 + 1)))
    for j, row in prev:                         # the ring's rows before the chunk
        ol.load(lw.state[row, :], out=E[K - 1 - j, :])
    xs = ol.quantize(rmsnorm(x, ol.load(lw.g_in), spec.eps))
    bx = E[K - 1:K - 1 + R, :]                  # B * x; B, C, x: this slice's channels
    bx.set(ol.dot(xs, lw.win[0:n, :]) * ol.dot(xs, lw.win[2 * n:3 * n, :]))
    if run:
        _ring_store(lw, E[R - 1:R - 1 + K, :], p0.words["ringo"], K, n)
    else:
        for r in range(max(0, R - K), R):
            for sl in _ring_rows(K, p0 + r)[1]:
                ol.store(lw.state[sl:sl + 1, :], E[K - 1 + r:K + r, :])
    full = 0 if run else max(0, K - 1 - p0)     # rows before it lack positions < 0
    groups = [(r, r + 1) for r in range(min(full, R))] + ([(full, R)] if full < R else [])
    y = ol.empty([R, n]) if len(groups) > 1 else None
    for r0, r1 in groups:
        acc = bx[r0:r1, :] * taps[K - 1, :][None, :]
        for j in range(1, (K - 1 if run else min(K - 1, p0 + r0)) + 1):
            acc = acc + E[K - 1 + r0 - j:K - 1 + r1 - j, :] * taps[K - 1 - j, :][None, :]
        if y is None:
            y = acc
        else:
            y[r0:r1, :].set(acc)
    del E, bx, acc, taps
    y_all = ol.all_gather(y * ol.dot(xs, lw.win[n:2 * n, :]))  # [R, H]
    return x + ol.all_gather(ol.dot(y_all, lw.wout))


@ol.jit
def lfm2_rows(m, p0: int, R: int, logit_rows, block: int = ATTN_BLOCK, tokens=None):
    """R prompt tokens at positions p0 .. p0+R-1 at once: their embeddings m.xr and RoPE
    tables m.cosr / m.sinr, or those of `tokens` from the image's tables (qwen3._inputs_rows) ->
    logits of the rows in `logit_rows` (a contiguous range, or empty). Bit-identical to R
    lfm2_step runs."""
    spec = m.spec
    rows = p0 if isinstance(p0, RunRows) else [(0, p0 + r) for r in range(R)]
    x, c, s_ = _inputs_rows(m, rows, tokens)

    def layer(li, it):
        lw = m.layer(li, it)
        if lw.kind == CONV:
            x.set(_conv_rows(x, lw, p0, spec))
        else:
            x.set(_attention_rows(x, lw, c, s_, rows, spec, block))
        x.set(_mlp(x, lw, spec))

    run_layers(m.plan, layer)
    _lm_head_rows(x, m, spec, logit_rows)


@ol.jit
def lfm2_prompt_run(m, pos, R: int, kind: str, block: int = ATTN_BLOCK):
    """A prompt run of R rows (docs/prefill.md), lfm2_rows with its tokens from out[]: kind
    "P", or "L" (the prompt's last run: its last row's logits). pos: a RunRows (toks_at 0) at
    the run-time position (the state's tpos and ring words, RunWords), or the run's first
    position, a compile-time one (the rows before conv_k - 1)."""
    run = isinstance(pos, RunRows)
    with RunWords(m, pos):
        lfm2_rows.fn(m, pos, R, [R - 1] if kind == "L" else [], block,
                     None if run else OutTokens(0))


def run_layers(runs, layer) -> None:
    """Emit the layers of a plan (runs of `plan`) as layer(index, it): a run with repeats is
    one hardware loop over its unit (the index is the element's layer in the first iteration,
    `it` the loop variable), the others are unrolled (`it` None)."""
    for first, unit, reps in runs:
        if reps == 1:
            for e in range(len(unit)):
                layer(first + e, None)
            continue
        for i in ol.range(reps):
            for e in range(len(unit)):
                layer(first + e, i)
