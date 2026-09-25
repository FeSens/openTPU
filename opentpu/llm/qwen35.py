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
  * The convolution state is a 4-slot ring of the pre-convolution q, k, v rows in DRAM
    (position p in slot p % 4), as LFM2's (lfm2.py), stored per pair of heads after the
    pair's taps.
  * 256-wide attention heads are two MXU blocks. With MCOLS < 4 a query group of 4 heads is
    split in pairs, each streaming the KV head (qwen3._attention).

Pieces:
  Spec              model dimensions and layer kinds (from a Hugging Face config.json)
  reference_logits  plain numpy forward pass (the math, fp32)
  emulated_logits   float64 decode with openTPU's quantization points (see qwen3)
  Image             per-slice DRAM layout: equal-size layer blocks of either kind
  qwen35_step       the ol kernel for one decode token

Weights, activations and the KV cache use Qwen3's W8A8 scheme; decoding runs on qwen3.Engine,
one token per device run.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .. import fp32 as F
from .. import language as ol
from ..compiler import Affine, KVDesc, QTensor, Tensor
from ..isasim import Config
from ..kernels.layouts import head_parallel_attention_weights
from ..kernels.deltanet import gates, l2norm_rows
from ..kernels.lib import rmsnorm, silu
from ..kernels.mlp import _chunk
from ..runtime import quantize_rows
from .lfm2 import plan, run_layers
from .qwen3 import (ATTN_BLOCK, _attention, _Bump, _fake_q, _lm_head, _mlp, _qdesc, _tdesc,
                    rope_tables)

LIN, ATTN = "linear", "attn"


# =============================================================================== model spec
@dataclass(frozen=True)
class Spec:
    hidden: int
    kinds: tuple            # LIN or ATTN per layer
    n_q: int
    n_kv: int
    head_dim: int
    rope_dim: int           # RoPE on the first rope_dim dimensions of each attention head
    lin_heads: int          # DeltaNet heads
    lin_dk: int
    lin_dv: int
    ffn: int
    vocab: int
    conv_k: int = 4         # DeltaNet convolution taps
    eps: float = 1e-6
    theta: float = 1e7
    tied: bool = True
    eos: tuple = (248046, 248044)

    @property
    def layers(self) -> int:
        return len(self.kinds)

    @staticmethod
    def from_hf(model_dir) -> "Spec":
        top = json.loads((Path(model_dir) / "config.json").read_text())
        c = top.get("text_config", top)
        if c["linear_num_key_heads"] != c["linear_num_value_heads"]:
            raise ValueError("DeltaNet with fewer key than value heads is not supported")
        rope = c.get("rope_parameters") or {}
        d = c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"]
        eos = c.get("eos_token_id", 248044)
        eos = tuple(eos) if isinstance(eos, list) else (eos,)
        return Spec(hidden=c["hidden_size"],
                    kinds=tuple(ATTN if t == "full_attention" else LIN for t in c["layer_types"]),
                    n_q=c["num_attention_heads"], n_kv=c["num_key_value_heads"], head_dim=d,
                    rope_dim=int(d * rope.get("partial_rotary_factor", 1.0)),
                    lin_heads=c["linear_num_value_heads"], lin_dk=c["linear_key_head_dim"],
                    lin_dv=c["linear_value_head_dim"], ffn=c["intermediate_size"],
                    vocab=c["vocab_size"], conv_k=c.get("linear_conv_kernel_dim", 4),
                    eps=c.get("rms_norm_eps", 1e-6), theta=rope.get("rope_theta", 1e7),
                    tied=top.get("tie_word_embeddings", c.get("tie_word_embeddings", True)),
                    eos=(248046,) + tuple(e for e in eos if e != 248046))   # <|im_end|> first

    def check(self, cfg: Config) -> None:
        S, D = cfg.S, cfg.D
        need = [(self.head_dim % D == 0, f"head_dim {self.head_dim} % D"),
                (self.rope_dim % 2 == 0 and self.rope_dim <= self.head_dim, "rope_dim"),
                (self.lin_dk == D and self.lin_dv == D, "DeltaNet heads must be D wide"),
                (self.lin_heads % (2 * S) == 0, f"DeltaNet heads % 2S (pairs per slice)"),
                (self.hidden % (S * D) == 0, f"hidden {self.hidden} % S*D"),
                (self.ffn % (S * D) == 0, f"ffn {self.ffn} % S*D"),
                (self.n_kv % S == 0, f"n_kv {self.n_kv} % S"),
                (self.n_q % self.n_kv == 0, "n_q % n_kv"),
                (self.vocab % S == 0, f"vocab {self.vocab} % S"),
                (max(self.ffn, self.n_q * self.head_dim, self.hidden) <= cfg.ACT_BLOCKS * D,
                 "an inner dimension exceeds ACT RAM")]
        bad = [m for ok, m in need if not ok]
        if bad:
            raise ValueError("model does not map onto this openTPU config: " + "; ".join(bad))

    def image(self, cfg: Config, cap: int, batch: int = 1, rows: int = 1) -> "Image":
        return Image(self, cfg, cap, batch, rows)


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
    nh, dk, dv = spec.lin_heads, spec.lin_dk, spec.lin_dv
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
            q, k, v = np.split(y, [nh * dk, 2 * nh * dk], axis=1)
            q = _l2norm(q.reshape(T, nh, dk)) / f32(math.sqrt(dk))
            k = _l2norm(k.reshape(T, nh, dk))
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
        gg = h @ W[p + "mlp.gate_proj.weight"].T
        u = h @ W[p + "mlp.up_proj.weight"].T
        x = x + (_silu(gg) * u) @ W[p + "mlp.down_proj.weight"].T
    x = _norm(x, g1("model.norm.weight"), eps)
    head = W["model.embed_tokens.weight"] if spec.tied else W["lm_head.weight"]
    return x @ head.T


def emulated_logits(spec: Spec, W: dict, tokens, D: int = 128) -> np.ndarray:
    """float64 decode with openTPU's quantization points and none of its rounding (as
    qwen3.emulated_logits): int8 weights and matmul inputs per D-block, int8 K and V, int8 P.
    The DeltaNet state, convolution and gates are exact (they are fp32 on the device)."""
    d, G, eps, K = spec.head_dim, spec.n_q // spec.n_kv, spec.eps, spec.conv_k
    nh, dk, dv = spec.lin_heads, spec.lin_dk, spec.lin_dv
    Wq: dict = {}

    def w(n):
        if n not in Wq:
            Wq[n] = _fake_q(np.asarray(W[n], np.float64), D)
        return Wq[n]

    def g1(n):
        return 1 + np.asarray(W[n], np.float64)

    lin = [i for i, k in enumerate(spec.kinds) if k == LIN]
    Kc = {i: [] for i, k in enumerate(spec.kinds) if k == ATTN}
    Vc = {i: [] for i in Kc}
    ring = {i: [np.zeros(nh * (2 * dk + dv))] * (K - 1) for i in lin}
    state = {i: np.zeros((nh, dk, dv)) for i in lin}
    out = []
    head = "model.embed_tokens.weight" if spec.tied else "lm_head.weight"
    for pos, tk in enumerate(tokens):
        x = np.asarray(W["model.embed_tokens.weight"][tk], np.float64)
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
                q, k, v = np.split(y, [nh * dk, 2 * nh * dk])
                q = _l2norm(q.reshape(nh, dk)) / math.sqrt(dk)
                k = _l2norm(k.reshape(nh, dk))
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
            gg = w(p + "mlp.gate_proj.weight") @ h
            u = w(p + "mlp.up_proj.weight") @ h
            x = x + w(p + "mlp.down_proj.weight") @ _fake_q(_silu(gg) * u, D)
        out.append(w(head) @ _fake_q(_norm(x, g1("model.norm.weight"), eps), D))
    return np.array(out)


# =============================================================================== DRAM image
class Image:
    """Per-slice DRAM layout of a Qwen3.5 model. Every slice uses the same addresses.

    [ I/O: x_in, cos, sin | final norm | logits | per-pair gates ] [ layer 0 block ] ...
    [ layer L-1 block ] [ LM head rows of this slice ]. All layer blocks have one size: both
    kinds start with the norms and this slice's MLP rows. A DeltaNet block then holds, for
    this slice's heads (a contiguous range), the projections pair by pair (the q, k, v rows of
    head 0, of head 1, then the z rows of heads 0 and 1; then heads 2 and 3, ...), the a and b
    rows, out_proj as one [H, og * dv] column block per og heads, per pair the convolution taps
    and then the convolution ring, the recurrent state (per head [dv, dk] fp32, transposed) and
    the per-head constants. An attention block holds the q/k
    norms, the projections (the gate rows of q_proj as their own matrix) and this slice's KV
    heads with room for `cap` tokens.
    """

    def __init__(self, spec: Spec, cfg: Config, cap: int, batch: int = 1, rows: int = 1):
        spec.check(cfg)
        if batch != 1 or rows != 1:
            raise ValueError("Qwen3.5 runs one token per device run: batch=1, rows=1")
        if cap % cfg.D:
            raise ValueError("KV capacity must be a multiple of D")
        S, D = cfg.S, cfg.D
        H, d, F_, K = spec.hidden, spec.head_dim, spec.ffn, spec.conv_k
        dk, dv = spec.lin_dk, spec.lin_dv
        self.spec, self.cfg, self.cap, self.batch, self.rows = spec, cfg, cap, 1, 1
        self.nq_loc, self.nkv_loc = spec.n_q // S, spec.n_kv // S
        self.h_loc, self.f_loc, self.v_loc = H // S, F_ // S, spec.vocab // S
        self.nl = spec.lin_heads // S                   # DeltaNet heads of one slice
        self.C = 2 * dk + dv                            # convolved channels per head (q, k, v)
        self.R = self.C + dv                            # projected rows per head (and z)
        self.plan = plan(spec.kinds)
        b = _Bump()
        self.io = {"x": b.alloc(4 * H), "cos": b.alloc(2 * spec.rope_dim),
                   "sin": b.alloc(2 * spec.rope_dim), "gf": b.alloc(4 * H),
                   "logits": b.alloc(4 * spec.vocab),
                   "hs": b.alloc(4 * 2 * self.nl)}     # per pair: decays of a, b; betas
        self.layer0 = b.next
        lb = _Bump()                                    # offsets inside one layer block
        common = {"g_in": lb.alloc(4 * H), "g_post": lb.alloc(4 * H)}
        mlp = {"wg": (self.f_loc, H), "wu": (self.f_loc, H)}
        for name, (n, k) in mlp.items():
            common[name] = (lb.alloc(n * k), lb.alloc(4 * n * (k // D)))
        self.dchunk = _chunk(self.f_loc, D)
        common["wd"] = [(lb.alloc(self.h_loc * self.dchunk),
                         lb.alloc(4 * self.h_loc * (self.dchunk // D)))
                        for _ in range(F_ // self.dchunk)]
        nl, C = self.nl, self.C
        self.og = 4 if nl % 4 == 0 else 2               # heads per out_proj MM
        self.mats = {LIN: {"wh": (nl * self.R, H), "wab": (2 * nl, H),
                           "wout": (nl // self.og * H, self.og * dv),
                           **mlp},
                     ATTN: {"wq": (self.nq_loc * d, H), "wgate": (self.nq_loc * d, H),
                            "wk": (self.nkv_loc * d, H), "wv": (self.nkv_loc * d, H),
                            "wo": (self.h_loc, spec.n_q * d), **mlp}}
        lnb = _Bump(lb.next)
        lin = dict(common, alog=lnb.alloc(4 * nl), dtb=lnb.alloc(4 * nl), gn=lnb.alloc(4 * dv),
                   cv=lnb.alloc(4 * nl * 2 * K * C),    # per pair: taps, then ring slots
                   state=lnb.alloc(4 * nl * dv * dk))
        ab = _Bump(lb.next)
        attn = dict(common, qn=ab.alloc(4 * d), kn=ab.alloc(4 * d))
        for kind, bump, L in ((LIN, lnb, lin), (ATTN, ab, attn)):
            for name, (n, k) in self.mats[kind].items():
                if name not in L:
                    L[name] = (bump.alloc(n * k), bump.alloc(4 * n * (k // D)))
        attn["kv"] = [{"k": ab.alloc(cap * d), "ks": ab.alloc(4 * cap * (d // D)),
                       "vt": ab.alloc(d * cap), "vs": ab.alloc(4 * cap)}
                      for _ in range(self.nkv_loc)]
        self.lofs = {LIN: lin, ATTN: attn}
        self.LS = (max(lnb.next, ab.next) + 4095) // 4096 * 4096
        n_attn = spec.kinds.count(ATTN)
        head = cap * d + 4 * cap * (d // D) + d * cap + 4 * cap
        self.kv_bytes = (n_attn * self.nkv_loc * head                   # KV cache, conv ring
                         + (spec.layers - n_attn) * 4 * nl * (K * C + dv * dk))  # and state
        b.next = self.layer0 + spec.layers * self.LS
        self.head = (b.alloc(self.v_loc * H), b.alloc(4 * self.v_loc * (H // D)))
        self.nbytes = b.next
        if self.nbytes > cfg.DRAM_BYTES:
            raise MemoryError(f"model image needs {self.nbytes / 2**20:.0f} MiB per slice, "
                              f"DRAM_BYTES is {cfg.DRAM_BYTES / 2**20:.0f} MiB")

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

        def put_q(addr_pair, parts):
            for s, p in enumerate(parts):
                q, sc = quantize_rows(p, D)
                put(s, addr_pair[0], q)
                put(s, addr_pair[1], sc)

        def rows(a, k):
            return [a[s * k:(s + 1) * k] for s in range(S)]

        def f32(a):
            return F.ftz(np.asarray(a, np.float32))

        def g1(name):                               # zero-centered norm weight, 1 + w
            return f32(1 + np.asarray(W[name], np.float32))

        for s in range(S):
            put(s, self.io["gf"], g1("model.norm.weight"))
        NK = spec.lin_heads * dk
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
                # the convolved channels of head h: its q, k and v rows of in_proj_qkv
                chans = [np.r_[h * dk:(h + 1) * dk, NK + h * dk:NK + (h + 1) * dk,
                               2 * NK + h * dv:2 * NK + (h + 1) * dv]
                         for h in range(spec.lin_heads)]
                taps = W[a + "conv1d.weight"][:, 0, :]                   # [channels, K]
                hs = [range(s * nl, (s + 1) * nl) for s in range(S)]
                # per pair of heads (a, b): the q, k, v rows of a, of b, then the z rows of a, of b
                put_q(Lo["wh"], [np.concatenate([np.concatenate(
                    [qkv[chans[h]], qkv[chans[h + 1]], wz[h * dv:(h + 2) * dv]])
                    for h in hh[::2]]) for hh in hs])
                put_q(Lo["wab"], [np.concatenate([W[a + "in_proj_a.weight"][hh.start:hh.stop],
                                                  W[a + "in_proj_b.weight"][hh.start:hh.stop]])
                                  for hh in hs])
                og = self.og                # out_proj column blocks of og heads
                put_q(Lo["wout"], [np.concatenate([wout[:, h * dv:(h + og) * dv]
                                                   for h in hh[::og]]) for hh in hs])
                for s, hh in enumerate(hs):
                    put(s, Lo["alog"], f32(W[a + "A_log"][hh.start:hh.stop]))
                    put(s, Lo["dtb"], f32(W[a + "dt_bias"][hh.start:hh.stop]))
                    put(s, Lo["gn"], f32(W[a + "norm.weight"]))
                    for q, h in enumerate(hh[::2]):
                        put(s, Lo["cv"] + q * 4 * 4 * K * self.C,
                            f32(np.stack([taps[chans[h]].T, taps[chans[h + 1]].T])))
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
            put_q(Lo["wg"], rows(W[p + "mlp.gate_proj.weight"], self.f_loc))
            put_q(Lo["wu"], rows(W[p + "mlp.up_proj.weight"], self.f_loc))
            C_ = self.dchunk
            for j, pair in enumerate(self.lofs[kind]["wd"]):
                put_q((base + pair[0], base + pair[1]),
                      [r[:, j * C_:(j + 1) * C_] for r in rows(W[p + "mlp.down_proj.weight"], n)])
        head = W["model.embed_tokens.weight"] if spec.tied else W["lm_head.weight"]
        put_q(self.head, rows(head, self.v_loc))
        return imgs

    # ---- programs
    def compile_step(self, pos: int, block: int = ATTN_BLOCK) -> list:
        """One program per slice: the decode token at position `pos` (qwen35_step)."""
        return [qwen35_step.trace(self.cfg, s, {"m": self.descriptors(s), "pos": pos,
                                                "block": block}).finish()
                for s in range(self.cfg.S)]

    def compile_rows(self, rows, logit_rows, block: int = ATTN_BLOCK) -> list:
        raise NotImplementedError("Qwen3.5 runs one token per device run (Engine.step)")

    # ---- kernel descriptors
    def descriptors(self, sid: int) -> SimpleNamespace:
        spec, cfg = self.spec, self.cfg
        D, d, H, K, n = cfg.D, spec.head_dim, spec.hidden, spec.conv_k, self.h_loc
        dk, dv, nl, C = spec.lin_dk, spec.lin_dv, self.nl, self.C

        def layer(li, kind):
            """Descriptors of layer `li` (an int or a hardware-loop expression) of `kind`."""
            off = Affine.of(self.layer0) + Affine.of(li) * self.LS
            lofs = self.lofs[kind]
            ns = SimpleNamespace(g_in=Tensor(off + lofs["g_in"], (H,), (1,)),
                                 g_post=Tensor(off + lofs["g_post"], (H,), (1,)))
            for name, (r, k) in self.mats[kind].items():
                da, sa = lofs[name]
                setattr(ns, name, QTensor(off + da, off + sa, (r, k), k, 4 * (k // D), D))
            Cd = self.dchunk
            parts = tuple(QTensor(off + da, off + sa, (n, Cd), Cd, 4 * (Cd // D), D)
                          for da, sa in lofs["wd"])
            ns.wd = QTensor(parts[0].data, parts[0].scale, (n, spec.ffn), Cd, 4 * (Cd // D), D,
                            parts=parts, pw=Cd)
            if kind == LIN:
                ns.alog = Tensor(off + lofs["alog"], (nl,), (1,))
                ns.dtb = Tensor(off + lofs["dtb"], (nl,), (1,))
                ns.gn = Tensor(off + lofs["gn"], (dv,), (1,))
                ns.cv = Tensor(off + lofs["cv"], (nl // 2, 4 * K * C), (4 * K * C, 1))
                ns.state = Tensor(off + lofs["state"], (nl, dv, dk), (dv * dk, dk, 1))
            else:
                ns.qn = Tensor(off + lofs["qn"], (d,), (1,))
                ns.kn = Tensor(off + lofs["kn"], (d,), (1,))
                ns.kv = KVDesc({sid + j * cfg.S: {k: off + v for k, v in r.items()}
                                for j, r in enumerate(lofs["kv"])}, self.cap, d, D, cfg.S, sid)
            return ns

        return SimpleNamespace(
            spec=spec, layer=layer, plan=self.plan,
            x=_tdesc(self.io["x"], (1, H)), cos=_tdesc(self.io["cos"], (spec.rope_dim // 2,)),
            sin=_tdesc(self.io["sin"], (spec.rope_dim // 2,)), g_final=_tdesc(self.io["gf"], (H,)),
            logits=_tdesc(self.io["logits"], (1, spec.vocab)),
            hs=_tdesc(self.io["hs"], (nl // 2, 4)),
            head=_qdesc(*self.head, self.v_loc, H, D), v_loc=self.v_loc)


# =============================================================================== kernel
def _deltanet(x, lw, pos: int, spec: Spec, hs):
    """x + out_proj(Gated DeltaNet(x)) for one token, this slice's heads; returns the new
    residual (replicated on every slice).

    First the decay exp(g) and beta of every head (kernels.deltanet.gates) go to `hs` in DRAM,
    [decay a, decay b, beta a, beta b] per pair of heads. The heads then run in pairs (a, b).
    The small vector work of a pair runs on [2, n] tiles: the convolution, SiLU, L2 norms and
    silu(z) before the recurrence ("prep"), the gated RMSNorm after it ("post"). Per head the
    recurrence is RDOT, OUTER, RDOT over its fp32 state (stored transposed; kernels.deltanet),
    which streams through TMEM in two buffers, one per head of the pair.

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
    nl, C = lw.state.shape[0], 2 * dk + dv
    R, NP, og = C + dv, nl // 2, lw.wout.shape[1] // dv
    TP = 2 * K * C                                      # taps words of a pair (then its ring)
    xs = ol.quantize(rmsnorm(x, ol.load(lw.g_in), eps))
    ab = ol.dot(xs, lw.wab)                             # [1, 2nl]: a, then b, of each head
    decay, beta = gates(ab[0, 0:nl], ab[0, nl:2 * nl], ol.load(lw.alog), ol.load(lw.dtb))
    eb = ol.empty([4 * NP]).reshape(NP, 4)              # per pair: decays of a, b; betas
    eb[:, 0:2].set(decay.reshape(NP, 2))
    eb[:, 2:4].set(beta.reshape(NP, 2))
    ol.store(hs, eb)
    del decay, beta, eb
    prevs = [(pos - j) % K for j in range(1, min(K, pos + 1))]          # ring slots of p-1, ...

    def pairs(n, w):
        return [ol.empty([2 * w]).reshape(2, w) for _ in range(n)]

    St = [ol.empty([dv * dk]).reshape(dv, dk) for _ in range(2)]        # heads a, b of a pair
    P = [ol.empty([1, 2 * R]) for _ in range(2)]        # q k v of a, of b, then z of a, of b
    CV = ol.empty([2 * TP])                             # taps (rows (head, tap)), ring slots
    U, Qn, Kn, GZ = pairs(2, C), pairs(2, dk), pairs(2, dk), pairs(2, dv)
    EB = [ol.empty([4]) for _ in range(2)]
    O = ol.empty([2 * dv]).reshape(2, dv)
    ON = ol.empty([og * dv]).reshape(og, dv)            # normed, gated o of og heads
    w = ol.empty([dv])
    y = ol.zeros([1, spec.hidden])
    gn = ol.load(lw.gn)
    halves = ((0, dv // 2), (dv // 2, dv))     # row halves of a state pass

    def project(p, t, split=False):
        """The MXU: pair p's projection rows into P[t] (q, k, v first: prep starts on them)."""
        cuts = ((0, C), (C, 2 * C), (2 * C, 2 * R)) if split else ((0, 2 * C), (2 * C, 2 * R))
        for c0, c1 in cuts:
            ol.dot(xs, lw.wh[p * 2 * R + c0:p * 2 * R + c1, :], out=P[t][:, c0:c1])

    def fetch_cv(p):
        """Pair p's taps and convolution ring (one load)."""
        ol.load(lw.cv[p, :], out=CV)

    def fetch_eb(p, t):
        ol.load(hs[p, :], out=EB[t])

    def state_in(h, S):
        if pos:
            for r, e in halves:
                ol.load(lw.state[h][r:e, :], out=S[r:e, :])
        else:
            S.set(0.0)

    def state_out(h, S):
        for r, e in halves:
            ol.store(lw.state[h][r:e, :], S[r:e, :])

    def conv(p, t, j=None):
        """Pair p's convolution into U[t] (j: only head j of the pair)."""
        a, n = (0, 2) if j is None else (j, 1)
        pre = P[t][0, a * C:(a + n) * C]
        s0 = TP + (pos % K) * 2 * C + a * C
        ol.store(lw.cv[p, s0:s0 + n * C], pre)
        taps = CV[0:TP].reshape(2 * K, C)

        def tap(i):
            return taps.row_stride_view(i, 2, K) if n == 2 else taps[a * K + i:a * K + i + 1, :]
        terms = [(pre.reshape(n, C), K - 1)] + [
            (CV[TP + s * 2 * C + a * C:TP + s * 2 * C + (a + n) * C].reshape(n, C), K - 2 - i)
            for i, s in enumerate(prevs)]
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
        GZ[t].set(silu(P[t][0, 2 * C:2 * R].reshape(2, dv)))

    def qk(t, j=None):
        """SiLU of the convolved q, k, v (in place), then the L2-normed q and k."""
        a, n = (0, 2) if j is None else (j, 1)
        u = U[t][a:a + n, :]
        u.set(silu(u))
        Qn[t][a:a + n, :].set(l2norm_rows(u[:, 0:dk], dk ** -0.5))
        Kn[t][a:a + n, :].set(l2norm_rows(u[:, dk:2 * dk]))

    def post(t):
        """The gated RMSNorm of the pair in O (gates GZ[t]) into its rows of ON."""
        r0 = 2 * t if og == 4 else 0
        ON[r0:r0 + 2, :].set(rmsnorm(O, gn, eps) * GZ[t])

    def flush(g, k=None):
        """out_proj of head group g (k: only its k-th pair)."""
        c0, c1 = (0, og) if k is None else (2 * k, 2 * k + 2)
        ol.dot(ON[c0:c1, :].reshape(1, (c1 - c0) * dv),
               lw.wout[g * spec.hidden:(g + 1) * spec.hidden, c0 * dv:c1 * dv], acc=y)

    def rdot1(S, t, j):
        """kv = S k of head j of the pair in buffers t, into w."""
        w.set(S @ Kn[t][j, :])

    def update(S, t, j):
        """d = beta (v - e^g kv), then S = e^g S + d k^T (OUTER, in place)."""
        w.set((U[t][j, 2 * dk:C] - w * EB[t][j:j + 1]) * EB[t][2 + j:3 + j])
        for r, e in halves:
            ol.outer(w[r:e], Kn[t][j, :], acc=S[r:e, :], decay=EB[t][j:j + 1])

    def rdot2(S, t, j):
        """o = S q of head j, into O[j]."""
        for r, e in halves:
            O[j, r:e].set(S[r:e, :] @ Qn[t][j, :])

    def _pair_segment(p, t, last1, last2, g=None):
        """Pair p (buffers t = p % 2); last1: no pair p+1, last2: no pair p+2; g: the head
        group pair p-1 completes (to multiply by out_proj)."""
        a, b = 2 * p, 2 * p + 1
        first = isinstance(p, int) and p == 0
        if not last2 and not first:
            project(p + 2, t)
        rdot1(St[0], t, 0)
        state_in(b, St[1])
        update(St[0], t, 0)
        state_out(a, St[0])
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
            state_in(a + 2, St[0])
        update(St[1], t, 1)
        state_out(b, St[1])
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
    state_in(0, St[0])
    conv(0, 0, 0)
    qk(0, 0)
    if NP > 1:
        project(1, 1)
    _pair_segment(0, 0, NP == 1, NP <= 2)
    n_it = max(0, (NP - 3) // 2)            # segments 1 .. NP-3 have every part: loop them
    if n_it:
        for i in ol.range(n_it):
            _pair_segment(2 * i + 1, 1, False, False, None if og == 4 else 2 * i)
            _pair_segment(2 * i + 2, 0, False, False, i if og == 4 else 2 * i + 1)
    for p in range(1 + 2 * n_it, NP):
        _pair_segment(p, p % 2, p + 1 >= NP, p + 2 >= NP, group(p - 1))
    post((NP - 1) % 2)
    flush(group(NP - 1), 1 if og == 4 and NP > 1 else None)
    return x + ol.all_reduce(y)


@ol.jit
def qwen35_step(m, pos: int, block: int = ATTN_BLOCK):
    """One decode token at position `pos`: x (the token's embedding) -> logits.

    Each run of m.plan with repeats is a hardware loop over its unit of layers; the others are
    unrolled. DeltaNet layers update their state and convolution ring, attention layers append
    K/V at `pos` and attend over positions 0..pos. Logits for this slice's vocabulary rows go
    to m.logits.
    """
    spec = m.spec
    x = ol.load(m.x)
    c, s_ = ol.load(m.cos), ol.load(m.sin)

    def layer(li, kind):
        lw = m.layer(li, kind)
        if kind == LIN:
            x.set(_deltanet(x, lw, pos, spec, m.hs))
        else:
            x.set(_attention(x, lw, c, s_, pos, spec, block, gated=True))
        x.set(_mlp(x, lw, spec))

    run_layers(m.plan, layer)
    _lm_head(x, m, spec)
