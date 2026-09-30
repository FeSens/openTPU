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
from .. import quant as Q
from .. import language as ol
from ..compiler import Affine, KVDesc, QTensor, Tensor
from ..isasim import Config
from ..kernels.layouts import head_parallel_attention_weights
from ..kernels.lib import rmsnorm
from ..kernels.mlp import _chunk
from ..runtime import quantize_rows
from .qwen3 import (RunPos, _inputs, _inputs_rows, _lookup_alloc, _lookup_build, _lookup_desc,
                    _tok_arg, _tokens_arg, compile_decode)
from .qwen3 import (ATTN_BLOCK, _attention, _attention_rows, _Bump, _fake_q, _fake_w, _lm_head,
                    _lm_head_rows, _mlp, _qdesc, _tdesc, rope_tables)

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
    embed: str = "f32"      # the embedding rows: fp32, or "int8" per D block (as qwen3.Spec:
                            # gathered on the device from the tied int8 head or a table)

    @property
    def layers(self) -> int:
        return len(self.kinds)

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
        return Spec(hidden=c["hidden_size"],
                    kinds=tuple(ATTN if t == "full_attention" else CONV for t in c["layer_types"]),
                    n_q=c["num_attention_heads"], n_kv=c["num_key_value_heads"],
                    head_dim=c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"],
                    ffn=ff, vocab=c["vocab_size"], conv_k=c.get("conv_L_cache", 3),
                    eps=c.get("norm_eps", 1e-5),
                    theta=rope.get("rope_theta", c.get("rope_theta", 1e6)),
                    tied=c.get("tie_word_embeddings", c.get("tie_embedding", True)),
                    bos=c.get("bos_token_id", 1),
                    eos=tuple(eos) if isinstance(eos, list) else (eos,))

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
        bad = [m for ok, m in need if not ok]
        if bad:
            raise ValueError("model does not map onto this openTPU config: " + "; ".join(bad))

    def image(self, cfg: Config, cap: int, batch: int = 1, rows: int = 1,
              wformat: str = "int8", head_format: str | None = None,
              lookup: bool = False) -> "Image":
        return Image(self, cfg, cap, batch, rows, wformat, head_format, lookup)


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
        g = h @ W[p + "feed_forward.w1.weight"].T
        u = h @ W[p + "feed_forward.w3.weight"].T
        x = x + ((g / (1 + np.exp(-g))) * u) @ W[p + "feed_forward.w2.weight"].T
    x = _norm(x, W["model.embedding_norm.weight"], spec.eps)
    head = W["model.embed_tokens.weight"] if spec.tied else W["lm_head.weight"]
    return x @ head.T


def emulated_logits(spec: Spec, W: dict, tokens, D: int = 128, wformat: str = "int8",
                    head_format: str | None = None) -> np.ndarray:
    """float64 decode with openTPU's quantization points and none of its rounding (as
    qwen3.emulated_logits); a zero-padded K or q block quantizes like its head_dim values."""
    d, G, H, K = spec.head_dim, spec.n_q // spec.n_kv, spec.hidden, spec.conv_k
    Wq: dict = {}

    head = "model.embed_tokens.weight" if spec.tied else "lm_head.weight"

    def w(n):
        if n not in Wq:        # the weight formats as in Image (wformat, head_format)
            Wq[n] = _fake_w(W[n], D, (head_format or wformat) if n == head else wformat)
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
                Vc[i].append(_fake_q(v, d))
                Kh, Vh = np.stack(Kc[i], 1), np.stack(Vc[i], 1)
                o = np.zeros((spec.n_q, d))
                for hq in range(spec.n_q):
                    sc = Kh[hq // G] @ _fake_q(q[hq] / math.sqrt(d), min(d, D))
                    pp = np.exp(sc - sc.max())
                    T = len(pp)
                    ppad = np.zeros(-(-T // D) * D)
                    ppad[:T] = pp
                    o[hq] = (_fake_q(ppad, D)[:T] @ Vh[hq // G]) / pp.sum()
                x = x + w(a + "out_proj.weight") @ _fake_q(o.reshape(-1), D)
            h = _fake_q(norm(x, W[p + "ffn_norm.weight"]), D)
            g = w(p + "feed_forward.w1.weight") @ h
            u = w(p + "feed_forward.w3.weight") @ h
            x = x + w(p + "feed_forward.w2.weight") @ _fake_q((g / (1 + np.exp(-g))) * u, D)
        out.append(w(head) @ _fake_q(norm(x, W["model.embedding_norm.weight"]), D))
    return np.array(out)


# =============================================================================== DRAM image
class Image:
    """Per-slice DRAM layout of an LFM2 model. Every slice uses the same addresses.

    [ I/O: x_in, cos, sin | final norm | logits ] [ layer 0 block ] ... [ layer L-1 block ]
    [ LM head rows of this slice ]. All layer blocks have one size: both kinds start with the
    norms and this slice's MLP rows; a conv block then holds the taps, the state ring and this
    slice's rows of in_proj (its channels of B, C and x) and out_proj; an attention block holds
    the q/k norms, the projections and this slice's KV heads with room for `cap` tokens. The
    I/O area holds `rows` token rows (x, cos, sin, logits) for chunked prefill.
    """

    def __init__(self, spec: Spec, cfg: Config, cap: int, batch: int = 1, rows: int = 1,
                 wformat: str = "int8", head_format: str | None = None, lookup: bool = False):
        spec.check(cfg)
        if batch != 1:
            raise ValueError("LFM2 runs one sequence: batch=1")
        if cap % cfg.D:
            raise ValueError("KV capacity must be a multiple of D")
        S, D = cfg.S, cfg.D
        H, d, F_, K = spec.hidden, spec.head_dim, spec.ffn, spec.conv_k
        self.wformat, self.head_format = wformat, head_format or wformat
        rb = lambda k: Q.row_bytes(k, wformat, D)                       # noqa: E731
        self.spec, self.cfg, self.cap, self.batch, self.rows = spec, cfg, cap, 1, rows
        self.dk = -(-d // D) * D                        # cached K row / query width
        self.nq_loc, self.nkv_loc = spec.n_q // S, spec.n_kv // S
        self.h_loc, self.f_loc, self.v_loc = H // S, F_ // S, spec.vocab // S
        self.plan = plan(spec.kinds)
        self.mlp_loop = sum(len(u) for _, u, _ in self.plan) > MLP_UNROLL_BODIES
        b = _Bump()
        R = rows
        self.io = {"x": b.alloc(4 * H * R), "cos": b.alloc(2 * d * R), "sin": b.alloc(2 * d * R),
                   "gf": b.alloc(4 * H), "logits": b.alloc(4 * spec.vocab * R)}
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
        self.mats = {CONV: {"win": (3 * self.h_loc, H), "wout": (self.h_loc, H), **mlp},
                     ATTN: {"wq": (self.nq_loc * d, H), "wk": (self.nkv_loc * d, H),
                            "wv": (self.nkv_loc * d, H), "wo": (self.h_loc, spec.n_q * d), **mlp}}
        cb = _Bump(lb.next)
        conv = dict(common, taps=cb.alloc(4 * K * self.h_loc),
                    state=cb.alloc(4 * 2 * K * self.h_loc))         # mirrored (_ring_rows)
        ab = _Bump(lb.next)
        attn = dict(common, qn=ab.alloc(4 * d), kn=ab.alloc(4 * d))
        for kind, bump, L in ((CONV, cb, conv), (ATTN, ab, attn)):
            for name, (n, k) in self.mats[kind].items():
                if name not in L:
                    L[name] = (bump.alloc(n * rb(k)), bump.alloc(4 * n * (k // D)))
        dk = self.dk
        attn["kv"] = [{"k": ab.alloc(cap * dk), "ks": ab.alloc(4 * cap * (dk // D)),
                       "vt": ab.alloc(dk * cap), "vs": ab.alloc(4 * cap)}
                      for _ in range(self.nkv_loc)]
        self.lofs = {CONV: conv, ATTN: attn}
        self.LS = (max(cb.next, ab.next) + 4095) // 4096 * 4096
        n_attn = spec.kinds.count(ATTN)
        head = cap * dk + 4 * cap * (dk // D) + dk * cap + 4 * cap
        self.kv_bytes = (n_attn * self.nkv_loc * head                   # KV cache and
                         + (spec.layers - n_attn) * 8 * K * self.h_loc)  # conv state, per sequence
        b.next = self.layer0 + spec.layers * self.LS
        self.head = (b.alloc(self.v_loc * Q.row_bytes(H, self.head_format, D)),
                     b.alloc(4 * self.v_loc * (H // D)))
        # the int8 embedding rows of the resident decode are the tied int8 head's (S = 1)
        shared = spec.tied and self.head_format == "int8" and S == 1
        self.lookup = _lookup_alloc(b, spec, cap, D=D, head=self.head if shared else None,
                                    M=cfg.MCOLS) if lookup else {}
        self.nbytes = b.next
        if self.nbytes > cfg.DRAM_BYTES:
            raise MemoryError(f"model image needs {self.nbytes / 2**20:.0f} MiB per slice, "
                              f"DRAM_BYTES is {cfg.DRAM_BYTES / 2**20:.0f} MiB")

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

        def put_q(addr_pair, parts, fmt=self.wformat):
            for s, p in enumerate(parts):
                q, sc = Q.quantize_mxu(p, fmt, D)
                put(s, addr_pair[0], q)
                put(s, addr_pair[1], sc)

        def rows(a, k):
            return [a[s * k:(s + 1) * k] for s in range(S)]

        def f32(a):
            return F.ftz(np.asarray(a, np.float32))

        for s in range(S):
            put(s, self.io["gf"], f32(W["model.embedding_norm.weight"]))
        for i, kind in enumerate(spec.kinds):
            p, base = f"model.layers.{i}.", self.layer0 + i * self.LS
            Lo = {k: (tuple(base + x for x in v) if isinstance(v, tuple) else
                      (base + v if isinstance(v, int) else v)) for k, v in self.lofs[kind].items()}
            for s in range(S):
                put(s, Lo["g_in"], f32(W[p + "operator_norm.weight"]))
                put(s, Lo["g_post"], f32(W[p + "ffn_norm.weight"]))
            if kind == CONV:
                B, C, X = np.split(W[p + "conv.in_proj.weight"], 3)
                put_q(Lo["win"], [np.concatenate([B[s * n:(s + 1) * n], C[s * n:(s + 1) * n],
                                                  X[s * n:(s + 1) * n]]) for s in range(S)])
                put_q(Lo["wout"], rows(W[p + "conv.out_proj.weight"], n))
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
                put_q(Lo["wq"], rows(wq, self.nq_loc * d))
                put_q(Lo["wk"], rows(wk, self.nkv_loc * d))
                put_q(Lo["wv"], rows(wv, self.nkv_loc * d))
                put_q(Lo["wo"], rows(wo, n))
            put_q(Lo["wg"], rows(W[p + "feed_forward.w1.weight"], self.f_loc))
            put_q(Lo["wu"], rows(W[p + "feed_forward.w3.weight"], self.f_loc))
            C_ = self.dchunk
            for j, pair in enumerate(self.lofs[kind]["wd"]):
                put_q((base + pair[0], base + pair[1]),
                      [r[:, j * C_:(j + 1) * C_] for r in rows(W[p + "feed_forward.w2.weight"], n)])
        head = W["model.embed_tokens.weight"] if spec.tied else W["lm_head.weight"]
        put_q(self.head, rows(head, self.v_loc), self.head_format)
        if self.lookup:
            _lookup_build(put, S, W, spec, self.cap, self.lookup)
        return imgs

    # ---- programs
    def compile_decode(self, blocks: int, lo: int, block: int = ATTN_BLOCK):
        """(programs, run_args): lfm2_step at a run-time position (qwen3.compile_decode); the
        convolutions need lo >= conv_k - 1 (every tap of the state ring is a past token)."""
        return compile_decode(self, lfm2_step, blocks, lo, block)

    def compile_step(self, pos: int, block: int = ATTN_BLOCK, tok: int | None = None) -> list:
        """One program per slice: the decode token at position `pos` (lfm2_step)."""
        return [lfm2_step.trace(self.cfg, s, {"m": self.descriptors(s), "pos": pos,
                                              "block": block, **_tok_arg(self, tok)}).finish()
                for s in range(self.cfg.S)]

    def compile_rows(self, rows, logit_rows, block: int = ATTN_BLOCK, tokens=None) -> list:
        """One program per slice: consecutive positions of the sequence at once (lfm2_rows)."""
        if len(rows) > self.rows:
            raise ValueError(f"{len(rows)} rows, the image's I/O area holds {self.rows}")
        if any(r != (0, rows[0][1] + i) for i, r in enumerate(rows)):
            raise ValueError("LFM2 rows must be consecutive positions of sequence 0")
        return [lfm2_rows.trace(self.cfg, s, {"m": self.descriptors(s), "p0": rows[0][1],
                                              "R": len(rows), "logit_rows": list(logit_rows),
                                              "block": block,
                                              **_tokens_arg(self, tokens, rows)}).finish()
                for s in range(self.cfg.S)]

    # ---- kernel descriptors
    def descriptors(self, sid: int) -> SimpleNamespace:
        spec, cfg = self.spec, self.cfg
        D, d, H, K, n = cfg.D, spec.head_dim, spec.hidden, spec.conv_k, self.h_loc

        def layer(li, kind):
            """Descriptors of layer `li` (an int or a hardware-loop expression) of `kind`."""
            off = Affine.of(self.layer0) + Affine.of(li) * self.LS
            lofs = self.lofs[kind]
            fm, wf = self.wformat, Q.mxu_wf(self.wformat)
            ns = SimpleNamespace(g_in=Tensor(off + lofs["g_in"], (H,), (1,)),
                                 g_post=Tensor(off + lofs["g_post"], (H,), (1,)),
                                 mlp_loop=self.mlp_loop)
            for name, (r, k) in self.mats[kind].items():
                da, sa = lofs[name]
                setattr(ns, name, QTensor(off + da, off + sa, (r, k), Q.row_bytes(k, fm, D),
                                          4 * (k // D), D, wf=wf))
            C = self.dchunk
            rc = Q.row_bytes(C, fm, D)
            parts = tuple(QTensor(off + da, off + sa, (n, C), rc, 4 * (C // D), D, wf=wf)
                          for da, sa in lofs["wd"])
            ns.wd = QTensor(parts[0].data, parts[0].scale, (n, spec.ffn), rc, 4 * (C // D), D,
                            parts=parts, pw=C, wf=wf)
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

        return SimpleNamespace(
            spec=spec, layer=layer, plan=self.plan,
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
    spec = m.spec
    x, c, s_ = _inputs(m, pos, tok)

    def layer(li, kind):
        lw = m.layer(li, kind)
        if kind == CONV:
            x.set(_conv(x, lw, pos, spec))
        else:
            x.set(_attention(x, lw, c, s_, pos, spec, block))
        x.set(_mlp(x, lw, spec))

    run_layers(m.plan, layer)
    _lm_head(x, m, spec)


def _conv_rows(x, lw, p0: int, spec: Spec):
    """_conv for R consecutive positions p0 .. p0+R-1 at once: in_proj streams once for the R
    rows (B and x first, C after the convolution: TMEM would not hold all three), and row r
    convolves over the rows before it in the chunk and the ring (positions
    before p0). The same products and sums per row as _conv, in the same order, so the result
    is bit-identical; the ring ends up holding the chunk's last K rows, each in its slot."""
    K, R = spec.conv_k, x.rows
    n = lw.taps.shape[1]
    taps = ol.load(lw.taps)                     # [K, n]; row K-1 weighs the current token
    E = ol.empty([K - 1 + R, n])                # B * x of positions p0-K+1 .. p0+R-1
    for j in range(1, min(K, p0 + 1)):          # the ring's rows before the chunk
        ol.load(lw.state[1 + (p0 - j) % K, :], out=E[K - 1 - j, :])
    xs = ol.quantize(rmsnorm(x, ol.load(lw.g_in), spec.eps))
    bx = E[K - 1:K - 1 + R, :]                  # B * x; B, C, x: this slice's channels
    bx.set(ol.dot(xs, lw.win[0:n, :]) * ol.dot(xs, lw.win[2 * n:3 * n, :]))
    for r in range(max(0, R - K), R):
        for sl in _ring_rows(K, p0 + r)[1]:
            ol.store(lw.state[sl:sl + 1, :], E[K - 1 + r:K + r, :])
    full = max(0, K - 1 - p0)                   # rows before it lack positions < 0
    groups = [(r, r + 1) for r in range(min(full, R))] + ([(full, R)] if full < R else [])
    y = ol.empty([R, n]) if len(groups) > 1 else None
    for r0, r1 in groups:
        acc = bx[r0:r1, :] * taps[K - 1, :][None, :]
        for j in range(1, min(K - 1, p0 + r0) + 1):
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
    rows = [(0, p0 + r) for r in range(R)]
    x, c, s_ = _inputs_rows(m, rows, tokens)

    def layer(li, kind):
        lw = m.layer(li, kind)
        if kind == CONV:
            x.set(_conv_rows(x, lw, p0, spec))
        else:
            x.set(_attention_rows(x, lw, c, s_, rows, spec, block))
        x.set(_mlp(x, lw, spec))

    run_layers(m.plan, layer)
    _lm_head_rows(x, m, spec, logit_rows)


def run_layers(runs, layer) -> None:
    """Emit the layers of a plan (runs of `plan`) as layer(index, kind): a run with repeats is
    one hardware loop over its unit (the index is then a loop expression), the others are
    unrolled."""
    for first, unit, reps in runs:
        if reps == 1:
            for e, kind in enumerate(unit):
                layer(first + e, kind)
            continue
        for i in ol.range(reps):
            for e, kind in enumerate(unit):
                layer(first + i * len(unit) + e, kind)
