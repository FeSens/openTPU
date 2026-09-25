"""Qwen3 dense decoder (e.g. Qwen3-0.6B / 1.7B) on openTPU.

Pieces:
  Spec              model dimensions (from a Hugging Face config.json)
  load_weights      Hugging Face safetensors -> fp32 numpy arrays (HF parameter names)
  reference_logits  plain numpy forward pass (the math, fp32), for debugging
  Image             the per-slice DRAM layout: every layer's weights, norms and KV cache in one
                    fixed-size block (so a hardware loop walks the layers with one address
                    register), the tied LM head, and a small I/O area
  qwen3_step        the ol kernel for one decode token: 28 layers, final norm, LM head
  qwen3_rows        R token rows at once, each row (sequence, position): batched decode (b
                    sequences, one KV cache each) and chunked prefill (consecutive positions
                    of one sequence; causal because each token attends over its own prefix)
  Engine            runs tokens on a backend (ISA simulator, RTL simulation or the board) and
                    keeps the KV caches in device DRAM between tokens

Weights are int8 with one fp32 scale per 128 inputs (per row); activations are quantized the
same way on the fly (W8A8). The residual stream, norms, RoPE and softmax are fp32.
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
from ..isasim import Config, Machine, design_config
from ..kernels.attention import _attend_heads
from ..kernels.layouts import head_parallel_attention_weights
from ..kernels.lib import rmsnorm, rope, rope_rows
from ..kernels.mlp import _chunk, swiglu_down
from ..runtime import ALIGN, quantize_rows


# =============================================================================== model spec
@dataclass(frozen=True)
class Spec:
    hidden: int
    layers: int
    n_q: int
    n_kv: int
    head_dim: int
    ffn: int
    vocab: int
    eps: float = 1e-6
    theta: float = 1e6
    tied: bool = True
    bos: int = 151643
    eos: tuple = (151645, 151643)

    @staticmethod
    def from_hf(model_dir) -> "Spec":
        c = json.loads((Path(model_dir) / "config.json").read_text())
        eos = c.get("eos_token_id", 151645)
        g = Path(model_dir) / "generation_config.json"
        if g.exists():
            eos = json.loads(g.read_text()).get("eos_token_id", eos)
        return Spec(hidden=c["hidden_size"], layers=c["num_hidden_layers"],
                    n_q=c["num_attention_heads"], n_kv=c["num_key_value_heads"],
                    head_dim=c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"],
                    ffn=c["intermediate_size"], vocab=c["vocab_size"], eps=c["rms_norm_eps"],
                    theta=c.get("rope_theta", 1e6), tied=c.get("tie_word_embeddings", True),
                    bos=c.get("bos_token_id", 151643),
                    eos=tuple(eos) if isinstance(eos, list) else (eos,))

    def check(self, cfg: Config) -> None:
        S, D = cfg.S, cfg.D
        need = [(self.head_dim % D == 0, f"head_dim {self.head_dim} % D {D}"),
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

    def image(self, cfg: Config, cap: int, batch: int = 1, rows: int = 1) -> "Image":
        return Image(self, cfg, cap, batch, rows)


def load_weights(model_dir) -> dict:
    """All tensors of a HF safetensors checkpoint as fp32 numpy arrays."""
    import torch
    from safetensors.torch import load_file
    out = {}
    for f in sorted(Path(model_dir).glob("*.safetensors")):
        for k, v in load_file(str(f)).items():
            out[k] = v.to(torch.float32).numpy()
    return out


def rope_tables(spec: Spec, pos: int) -> tuple[np.ndarray, np.ndarray]:
    """cos, sin [head_dim/2] for one position (HF rotate-half convention)."""
    half = spec.head_dim // 2
    inv = 1.0 / (spec.theta ** (np.arange(half, dtype=np.float64) * 2 / spec.head_dim))
    ang = pos * inv
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


# =============================================================================== reference
def reference_logits(spec: Spec, W: dict, tokens) -> np.ndarray:
    """fp32 numpy forward of the whole sequence (causal); returns logits [T, vocab]."""
    tokens = list(tokens)
    T, d, G = len(tokens), spec.head_dim, spec.n_q // spec.n_kv
    x = W["model.embed_tokens.weight"][tokens].astype(np.float32)

    def norm(v, g):
        return (v / np.sqrt(np.mean(v * v, axis=-1, keepdims=True) + spec.eps)) * g

    cs = [rope_tables(spec, p) for p in range(T)]
    cos = np.stack([c for c, _ in cs])[:, None, :]
    sin = np.stack([s for _, s in cs])[:, None, :]

    def rot(v):
        h = d // 2
        v1, v2 = v[..., :h], v[..., h:]
        return np.concatenate([v1 * cos - v2 * sin, v2 * cos + v1 * sin], axis=-1)

    mask = np.triu(np.full((T, T), -np.inf, np.float32), 1)
    for i in range(spec.layers):
        p = f"model.layers.{i}."
        h = norm(x, W[p + "input_layernorm.weight"])
        q = (h @ W[p + "self_attn.q_proj.weight"].T).reshape(T, spec.n_q, d)
        k = (h @ W[p + "self_attn.k_proj.weight"].T).reshape(T, spec.n_kv, d)
        v = (h @ W[p + "self_attn.v_proj.weight"].T).reshape(T, spec.n_kv, d)
        q = rot(norm(q, W[p + "self_attn.q_norm.weight"]))
        k = rot(norm(k, W[p + "self_attn.k_norm.weight"]))
        o = np.zeros((T, spec.n_q, d), np.float32)
        for hq in range(spec.n_q):
            s = q[:, hq] @ k[:, hq // G].T / math.sqrt(d) + mask
            s = np.exp(s - s.max(axis=1, keepdims=True))
            o[:, hq] = (s / s.sum(axis=1, keepdims=True)) @ v[:, hq // G]
        x = x + o.reshape(T, -1) @ W[p + "self_attn.o_proj.weight"].T
        h = norm(x, W[p + "post_attention_layernorm.weight"])
        g = h @ W[p + "mlp.gate_proj.weight"].T
        u = h @ W[p + "mlp.up_proj.weight"].T
        x = x + ((g / (1 + np.exp(-g))) * u) @ W[p + "mlp.down_proj.weight"].T
    x = norm(x, W["model.norm.weight"])
    head = W["model.embed_tokens.weight"] if spec.tied else W["lm_head.weight"]
    return x @ head.T


def _fake_q(x, D: int = 128):
    """int8 quantize-dequantize per (row, D-block) of the last axis (numerics of QACT/QST)."""
    sh = x.shape
    xb = x.reshape(*sh[:-1], sh[-1] // D, D)
    a = np.abs(xb).max(-1, keepdims=True)
    s = np.where(a == 0, 1, a / 127)
    return (np.clip(np.rint(xb / s), -127, 127) * s).reshape(sh)


def emulated_logits(spec: Spec, W: dict, tokens, D: int = 128) -> np.ndarray:
    """float64 decode that applies openTPU's quantization points but none of its rounding:
    int8 weights and matmul inputs per D-block, int8 K (per token, D-block) and V (per token),
    int8 P (per D tokens). Separates quantization error from kernel bugs."""
    d, G = spec.head_dim, spec.n_q // spec.n_kv
    Wq: dict = {}

    def w(n):
        if n not in Wq:
            Wq[n] = _fake_q(np.asarray(W[n], np.float64), D)
        return Wq[n]

    def norm(v, g):
        return (v / np.sqrt(np.mean(v * v, -1, keepdims=True) + spec.eps)) * g

    Kc = [[] for _ in range(spec.layers)]
    Vc = [[] for _ in range(spec.layers)]
    out = []
    head = "model.embed_tokens.weight" if spec.tied else "lm_head.weight"
    for pos, tk in enumerate(tokens):
        x = np.asarray(W["model.embed_tokens.weight"][tk], np.float64)
        c, s = rope_tables(spec, pos)

        def rot(v):
            h = d // 2
            v1, v2 = v[..., :h], v[..., h:]
            return np.concatenate([v1 * c - v2 * s, v2 * c + v1 * s], -1)

        for i in range(spec.layers):
            p = f"model.layers.{i}."
            h = _fake_q(norm(x, W[p + "input_layernorm.weight"]), D)
            q = (w(p + "self_attn.q_proj.weight") @ h).reshape(spec.n_q, d)
            k = (w(p + "self_attn.k_proj.weight") @ h).reshape(spec.n_kv, d)
            v = (w(p + "self_attn.v_proj.weight") @ h).reshape(spec.n_kv, d)
            q = rot(norm(q, W[p + "self_attn.q_norm.weight"]))
            k = rot(norm(k, W[p + "self_attn.k_norm.weight"]))
            Kc[i].append(_fake_q(k, D))
            Vc[i].append(_fake_q(v, v.shape[-1]))
            K, V = np.stack(Kc[i], 1), np.stack(Vc[i], 1)
            o = np.zeros((spec.n_q, d))
            for hq in range(spec.n_q):
                sc = K[hq // G] @ _fake_q(q[hq] / math.sqrt(d), D)
                pp = np.exp(sc - sc.max())
                T = len(pp)
                ppad = np.zeros(-(-T // D) * D)
                ppad[:T] = pp
                o[hq] = (_fake_q(ppad, D)[:T] @ V[hq // G]) / pp.sum()
            x = x + w(p + "self_attn.o_proj.weight") @ _fake_q(o.reshape(-1), D)
            h = _fake_q(norm(x, W[p + "post_attention_layernorm.weight"]), D)
            g = w(p + "mlp.gate_proj.weight") @ h
            u = w(p + "mlp.up_proj.weight") @ h
            x = x + w(p + "mlp.down_proj.weight") @ _fake_q((g / (1 + np.exp(-g))) * u, D)
        out.append(w(head) @ _fake_q(norm(x, W["model.norm.weight"]), D))
    return np.array(out)


# =============================================================================== DRAM image
# Attention: tokens per flash block (256 halves the per-block vector-unit latency overhead of
# 128 at long contexts) and score blocks in flight per head.
ATTN_BLOCK = 256
ATTN_DEPTH = 3


class _Bump:
    def __init__(self, start: int = 0):
        self.next = start

    def alloc(self, nbytes: int) -> int:
        a = self.next
        self.next = (a + nbytes + ALIGN - 1) // ALIGN * ALIGN
        return a


def _qdesc(daddr: int, saddr: int, n: int, k: int, D: int) -> QTensor:
    return QTensor(Affine(daddr), Affine(saddr), (n, k), k, 4 * (k // D), D)


def _tdesc(addr: int, shape) -> Tensor:
    shape = tuple(shape)
    strides, st = [], 1
    for n in reversed(shape):
        strides.append(st)
        st *= n
    return Tensor(Affine(addr), shape, tuple(reversed(strides)))


class Image:
    """Per-slice DRAM layout of a Qwen3 model. Every slice uses the same addresses.

    [ I/O: x_in, cos, sin | final norm | logits ] [ layer 0 block ] ... [ layer L-1 block ]
    [ LM head rows of this slice ]. A layer block holds the norms, this slice's rows of every
    projection (int8 + scales) and this slice's KV heads with room for `cap` tokens, for each
    of `batch` sequences. The I/O area holds `rows` token rows (x, cos, sin, logits).
    """

    def __init__(self, spec: Spec, cfg: Config, cap: int, batch: int = 1, rows: int = 1):
        spec.check(cfg)
        if cap % cfg.D:
            raise ValueError("KV capacity must be a multiple of D")
        S, D = cfg.S, cfg.D
        H, d, F_ = spec.hidden, spec.head_dim, spec.ffn
        self.spec, self.cfg, self.cap = spec, cfg, cap
        self.batch, self.rows = batch, rows
        self.nq_loc, self.nkv_loc = spec.n_q // S, spec.n_kv // S
        self.h_loc, self.f_loc, self.v_loc = H // S, F_ // S, spec.vocab // S
        b = _Bump()
        R = rows
        self.io = {"x": b.alloc(4 * H * R), "cos": b.alloc(2 * d * R), "sin": b.alloc(2 * d * R),
                   "gf": b.alloc(4 * H), "logits": b.alloc(4 * spec.vocab * R)}
        self.layer0 = b.next
        lb = _Bump()                                    # offsets inside one layer block
        L = {"g_in": lb.alloc(4 * H), "g_post": lb.alloc(4 * H),
             "qn": lb.alloc(4 * d), "kn": lb.alloc(4 * d)}
        self.mats = {"wq": (self.nq_loc * d, H), "wk": (self.nkv_loc * d, H),
                     "wv": (self.nkv_loc * d, H), "wo": (self.h_loc, spec.n_q * d),
                     "wg": (self.f_loc, H), "wu": (self.f_loc, H)}
        for name, (n, k) in self.mats.items():
            L[name] = (lb.alloc(n * k), lb.alloc(4 * n * (k // D)))
        # W_down in column parts of the MLP's F chunk: each down MM streams one part, whose
        # scales are then contiguous (with row-major scales every row would cost a DRAM beat)
        self.dchunk = _chunk(self.f_loc, D)
        L["wd"] = [(lb.alloc(self.h_loc * self.dchunk), lb.alloc(4 * self.h_loc * (self.dchunk // D)))
                   for _ in range(F_ // self.dchunk)]
        L["kvs"] = [[{"k": lb.alloc(cap * d), "ks": lb.alloc(4 * cap * (d // D)),
                      "vt": lb.alloc(d * cap), "vs": lb.alloc(4 * cap)}
                     for _ in range(self.nkv_loc)] for _ in range(batch)]
        L["kv"] = L["kvs"][0]
        self.lofs, self.LS = L, (lb.next + 4095) // 4096 * 4096
        head = cap * d + 4 * cap * (d // D) + d * cap + 4 * cap   # k, k scales, v^T, v scales
        self.kv_bytes = spec.layers * self.nkv_loc * head         # per sequence
        b.next = self.layer0 + spec.layers * self.LS
        self.head = (b.alloc(self.v_loc * H), b.alloc(4 * self.v_loc * (H // D)))
        self.nbytes = b.next
        if self.nbytes > cfg.DRAM_BYTES:
            raise MemoryError(f"model image needs {self.nbytes / 2**20:.0f} MiB per slice, "
                              f"DRAM_BYTES is {cfg.DRAM_BYTES / 2**20:.0f} MiB")

    # ---- contents
    def build(self, W: dict) -> list[np.ndarray]:
        """DRAM images (one per slice) with every weight quantized in place, KV cache empty."""
        spec, cfg = self.spec, self.cfg
        S, D, d = cfg.S, cfg.D, spec.head_dim
        imgs = [np.zeros(self.nbytes, np.uint8) for _ in range(S)]

        def put(s, addr, a):
            v = np.ascontiguousarray(a).view(np.uint8).reshape(-1)
            imgs[s][addr:addr + v.size] = v

        def put_q(addr_pair, parts):
            for s, p in enumerate(parts):
                q, sc = quantize_rows(p, D)
                put(s, addr_pair[0], q)
                put(s, addr_pair[1], sc)

        def rows(a, n):
            return [a[s * n:(s + 1) * n] for s in range(S)]

        def f32(a):
            return F.ftz(np.asarray(a, np.float32))

        for s in range(S):
            put(s, self.io["gf"], f32(W["model.norm.weight"]))
        for i in range(spec.layers):
            p, base = f"model.layers.{i}.", self.layer0 + i * self.LS
            Lo = {k: (tuple(base + x for x in v) if isinstance(v, tuple) else
                      (base + v if isinstance(v, int) else v)) for k, v in self.lofs.items()}
            for s in range(S):
                put(s, Lo["g_in"], f32(W[p + "input_layernorm.weight"]))
                put(s, Lo["g_post"], f32(W[p + "post_attention_layernorm.weight"]))
                put(s, Lo["qn"], f32(W[p + "self_attn.q_norm.weight"]))
                put(s, Lo["kn"], f32(W[p + "self_attn.k_norm.weight"]))
            wq, wk, wv, wo = head_parallel_attention_weights(
                W[p + "self_attn.q_proj.weight"], W[p + "self_attn.k_proj.weight"],
                W[p + "self_attn.v_proj.weight"], W[p + "self_attn.o_proj.weight"],
                spec.n_q, spec.n_kv, d, S)
            put_q(Lo["wq"], rows(wq, self.nq_loc * d))
            put_q(Lo["wk"], rows(wk, self.nkv_loc * d))
            put_q(Lo["wv"], rows(wv, self.nkv_loc * d))
            put_q(Lo["wo"], rows(wo, self.h_loc))
            put_q(Lo["wg"], rows(W[p + "mlp.gate_proj.weight"], self.f_loc))
            put_q(Lo["wu"], rows(W[p + "mlp.up_proj.weight"], self.f_loc))
            C = self.dchunk
            for j, pair in enumerate(self.lofs["wd"]):
                put_q((base + pair[0], base + pair[1]),
                      [r[:, j * C:(j + 1) * C] for r in rows(W[p + "mlp.down_proj.weight"],
                                                              self.h_loc)])
        head = W["model.embed_tokens.weight"] if spec.tied else W["lm_head.weight"]
        put_q(self.head, rows(head, self.v_loc))
        return imgs

    # ---- programs
    def compile_step(self, pos: int, block: int = ATTN_BLOCK) -> list:
        """One program per slice: the decode token at position `pos` (qwen3_step)."""
        return [qwen3_step.trace(self.cfg, s, {"m": self.descriptors(s), "pos": pos,
                                               "block": block}).finish()
                for s in range(self.cfg.S)]

    def compile_rows(self, rows, logit_rows, block: int = ATTN_BLOCK) -> list:
        """One program per slice: token rows (sequence, position) at once (qwen3_rows)."""
        if len(rows) > self.rows:
            raise ValueError(f"{len(rows)} rows, the image's I/O area holds {self.rows}")
        return [qwen3_rows.trace(self.cfg, s, {"m": self.descriptors(s), "rows": list(rows),
                                               "logit_rows": list(logit_rows),
                                               "block": block}).finish()
                for s in range(self.cfg.S)]

    # ---- kernel descriptors
    def descriptors(self, sid: int) -> SimpleNamespace:
        spec, cfg = self.spec, self.cfg
        D, d, H = cfg.D, spec.head_dim, spec.hidden
        L0 = self.layer0
        lofs = self.lofs

        def layer(li):
            """Descriptors of layer `li` (an int or a hardware-loop variable)."""
            off = Affine.of(L0) + Affine.of(li) * self.LS
            ns = SimpleNamespace(
                g_in=Tensor(off + lofs["g_in"], (H,), (1,)),
                g_post=Tensor(off + lofs["g_post"], (H,), (1,)),
                qn=Tensor(off + lofs["qn"], (d,), (1,)),
                kn=Tensor(off + lofs["kn"], (d,), (1,)))
            for name, (n, k) in self.mats.items():
                da, sa = lofs[name]
                setattr(ns, name, QTensor(off + da, off + sa, (n, k), k, 4 * (k // D), D))
            C, n = self.dchunk, self.h_loc
            parts = tuple(QTensor(off + da, off + sa, (n, C), C, 4 * (C // D), D)
                          for da, sa in lofs["wd"])
            ns.wd = QTensor(parts[0].data, parts[0].scale, (n, spec.ffn), C, 4 * (C // D), D,
                            parts=parts, pw=C)
            ns.kvs = [KVDesc({sid + j * cfg.S: {k: off + v for k, v in r.items()}
                              for j, r in enumerate(heads)}, self.cap, d, D, cfg.S, sid)
                      for heads in lofs["kvs"]]
            ns.kv = ns.kvs[0]
            return ns

        return SimpleNamespace(
            spec=spec, layer=layer, n_layers=spec.layers,
            x=_tdesc(self.io["x"], (1, H)), cos=_tdesc(self.io["cos"], (d // 2,)),
            sin=_tdesc(self.io["sin"], (d // 2,)), g_final=_tdesc(self.io["gf"], (H,)),
            logits=_tdesc(self.io["logits"], (1, spec.vocab)),
            xr=_tdesc(self.io["x"], (self.rows, H)),
            cosr=_tdesc(self.io["cos"], (self.rows, d // 2)),
            sinr=_tdesc(self.io["sin"], (self.rows, d // 2)),
            logitsr=_tdesc(self.io["logits"], (self.rows, spec.vocab)),
            head=_qdesc(*self.head, self.v_loc, H, D), v_loc=self.v_loc)


# =============================================================================== kernel
def _padded(x):
    """Rows of one head each, as the KV cache stores them: with head_dim < D (LFM2: 64) padded
    with zeros to a whole MXU block. q.K^T contracts over D, so the cached K rows and the
    queries carry zeros beyond head_dim (the scores are unchanged); a quantized store writes
    whole blocks, so V's zero rows are stored too, but P.V reads only its head_dim rows."""
    d, D = x.cols, ol.block_size()
    if d % D == 0:
        return x
    out = ol.zeros([x.rows, -(-d // D) * D])
    out[:, :d].set(x)
    return out


def _rope_padded(x, c, s_):
    """rope(x), padded like _padded (RoPE writes straight into the padded tile)."""
    d, D = x.cols, ol.block_size()
    if d % D == 0:
        return rope(x, c, s_)
    out = ol.zeros([x.rows, -(-d // D) * D])
    rope(x, c, s_, out=out[:, :d])
    return out


def _attention(x, lw, c, s_, pos: int, spec: Spec, block: int):
    """x + W_o . attention(x) for one token, this slice's heads; returns the new residual
    (replicated on every slice).

    Schedule (the MXU streams weights in program order, so what sits between two MMs in the
    stream overlaps them): K and V are projected first and their norms, RoPE and cache appends
    run while the Q projection streams; Q is projected one KV head's query group at a time,
    interleaved with the attention of the heads before it (_attend_heads), so each head's
    query preparation and softmax hide behind the next heads' Q weights."""
    d, G, eps = spec.head_dim, spec.n_q // spec.n_kv, spec.eps
    kv = lw.kv
    xs = ol.quantize(rmsnorm(x, ol.load(lw.g_in), eps))
    k = ol.dot(xs, lw.wk)                       # [1, nkv_loc*d]
    v = ol.dot(xs, lw.wv)
    qn, kn = ol.load(lw.qn), ol.load(lw.kn)
    scale = ol.LOG2E / math.sqrt(d)
    heads = list(kv.owned_heads(spec.n_kv))
    nh = len(heads)
    kh = _rope_padded(rmsnorm(k.reshape(nh, d), kn, eps), c, s_)  # [nkv_loc, d or D]
    vh = _padded(v.reshape(nh, d))
    for j, hh in enumerate(heads):
        ol.kv_append(kv, hh, pos, kh[j:j + 1, :], vh[j:j + 1, :])

    def queries(j):
        def emit():
            qj = ol.dot(xs, lw.wq[j * G * d:(j + 1) * G * d, :])  # [1, G*d]
            return _rope_padded(rmsnorm(qj.reshape(G, d), qn, eps), c, s_)
        return emit

    outs = _attend_heads([queries(j) for j in range(nh)], kv, heads, pos + 1, block, scale,
                         depth=ATTN_DEPTH, ahead=2)
    o_row = ol.empty([1, G * nh * d])
    o_loc = o_row.reshape(G * nh, d)            # the heads' outputs, written in place
    for j, (acc, l) in enumerate(outs):
        o_loc[j * G:(j + 1) * G, :].set(acc / l[:, None])
    o_all = ol.all_gather(o_row)                # [1, n_q*d], slice-major head order
    y = ol.all_gather(ol.dot(o_all, lw.wo))     # [1, H]
    return x + y


def _mlp(x, lw, spec: Spec):
    sid = ol.program_id()
    xs = ol.quantize(rmsnorm(x, ol.load(lw.g_post), spec.eps))
    y = swiglu_down(xs, lw.wg, lw.wu, lw.wd)
    h_loc = lw.wd.shape[0]
    mine = slice(sid * h_loc, (sid + 1) * h_loc)
    return ol.all_gather(x[:, mine] + y)


HEAD_CHUNK = 8192     # LM head rows per MM (the fp32 logits of one chunk must fit TMEM)


@ol.jit
def qwen3_step(m, pos: int, block: int = ATTN_BLOCK):
    """One decode token at position `pos`: x (the token's embedding) -> logits.

    The layers run as a hardware loop; each layer appends its K/V at `pos` and attends over
    positions 0..pos. Logits for this slice's vocabulary rows are stored to m.logits.
    """
    spec = m.spec
    x = ol.load(m.x)
    c, s_ = ol.load(m.cos), ol.load(m.sin)
    for li in ol.range(m.n_layers):
        lw = m.layer(li)
        x.set(_attention(x, lw, c, s_, pos, spec, block))
        x.set(_mlp(x, lw, spec))
    _lm_head(x, m, spec)


def _lm_head(x, m, spec):
    """Final norm and this slice's vocabulary rows of the LM head -> m.logits."""
    sid = ol.program_id()
    xs = ol.quantize(rmsnorm(x, ol.load(m.g_final), spec.eps))
    chunk = min(HEAD_CHUNK, ol.tmem_words() // 8)
    for c0 in range(0, m.v_loc, chunk):
        n = min(chunk, m.v_loc - c0)
        col = sid * m.v_loc + c0
        ol.store(m.logits[:, col:col + n], ol.dot(xs, m.head[c0:c0 + n, :]))


def _runs(rows):
    """Maximal runs of rows that continue one sequence at consecutive positions:
    [(seq, first position, first row, count)]."""
    out = []
    for r, (sq, p) in enumerate(rows):
        if out and out[-1][0] == sq and out[-1][1] + out[-1][3] == p:
            out[-1] = (sq, out[-1][1], out[-1][2], out[-1][3] + 1)
        else:
            out.append((sq, p, r, 1))
    return out


def _attention_rows(x, lw, c, s_, rows, spec: Spec, block: int):
    """x + W_o . attention(x) for R token rows; row r is token position rows[r][1] of sequence
    rows[r][0] (its own KV cache). Every row's K/V is appended first, then each row attends
    over positions 0..pos of its sequence -- for consecutive rows of one sequence (a prefill
    chunk) that is exactly the causal mask. Each projection streams its weights once for all
    R rows (ceil(R / MCOLS) MMs); the (row, KV head) pairs then run as one pipelined
    flash-attention stream (_attend_heads)."""
    d, G, eps = spec.head_dim, spec.n_q // spec.n_kv, spec.eps
    R = len(rows)
    xs = ol.quantize(rmsnorm(x, ol.load(lw.g_in), eps))
    k = ol.dot(xs, lw.wk)                       # [R, nkv_loc*d]
    v = ol.dot(xs, lw.wv)
    q = ol.dot(xs, lw.wq)                       # [R, nq_loc*d]
    qn, kn = ol.load(lw.qn), ol.load(lw.kn)
    scale = ol.LOG2E / math.sqrt(d)
    heads = list(lw.kv.owned_heads(spec.n_kv))
    nh = len(heads)
    nq = nh * G
    for j, hh in enumerate(heads):
        kj = rope_rows(rmsnorm(k[:, j * d:(j + 1) * d], kn, eps), c, s_)
        vj = v[:, j * d:(j + 1) * d]
        for sq, p0, r0, n in _runs(rows):
            ol.kv_append(lw.kvs[sq], hh, p0, kj[r0:r0 + n, :], vj[r0:r0 + n, :])
        del kj, vj
    del k, v
    # queries as [R * nq, d]: (row r, KV head j) is the G contiguous rows r*nq + j*G ...
    Q = ol.empty([R * nq, d])
    for h in range(nq):
        rope_rows(rmsnorm(q[:, h * d:(h + 1) * d], qn, eps), c, s_,
                  out=Q.row_stride_view(h, R, nq))
    del q
    ent = [(r, j) for r in range(R) for j in range(nh)]
    o = ol.empty([R, nq * d])

    def emit(i, acc, l):
        r, j = ent[i]
        o[r, j * G * d:(j + 1) * G * d].reshape(G, d).set(acc / l[:, None])

    _attend_heads([Q[r * nq + j * G:r * nq + (j + 1) * G, :] for r, j in ent],
                  [lw.kvs[rows[r][0]] for r, _ in ent], [heads[j] for _, j in ent],
                  [rows[r][1] + 1 for r, _ in ent], block, scale, depth=ATTN_DEPTH,
                  emit=emit)
    del Q
    o_all = ol.all_gather(o)                    # [R, n_q*d]
    y = ol.all_gather(ol.dot(o_all, lw.wo))     # [R, H]
    return x + y


@ol.jit
def qwen3_rows(m, rows, logit_rows, block: int = ATTN_BLOCK):
    """R token rows at once (rows[r] = (sequence, position)): the rows' embeddings m.xr and
    RoPE tables m.cosr / m.sinr -> logits of the rows in `logit_rows` (a contiguous range, or
    empty: a prefill chunk that is not the last one skips the LM head)."""
    spec, sid = m.spec, ol.program_id()
    R = len(rows)
    x = ol.load(m.xr[0:R, :])
    c, s_ = ol.load(m.cosr[0:R, :]), ol.load(m.sinr[0:R, :])
    for li in ol.range(m.n_layers):
        lw = m.layer(li)
        x.set(_attention_rows(x, lw, c, s_, rows, spec, block))
        x.set(_mlp(x, lw, spec))
    if not logit_rows:
        return
    a, e = logit_rows[0], logit_rows[-1] + 1
    xs = ol.quantize(rmsnorm(x[a:e, :], ol.load(m.g_final), spec.eps))
    chunk = min(HEAD_CHUNK, ol.tmem_words() // (8 * (e - a)))
    for c0 in range(0, m.v_loc, chunk):
        n = min(chunk, m.v_loc - c0)
        col = sid * m.v_loc + c0
        ol.store(m.logitsr[a:e, col:col + n], ol.dot(xs, m.head[c0:c0 + n, :]))


# =============================================================================== engine
def device_config(spec: Spec, cap: int, batch: int = 1, rows: int = 1, **kw) -> Config:
    """The design configuration with DRAM sized for this model (power of two MiB)."""
    probe = spec.image(design_config(DRAM_BYTES=1 << 40, **kw), cap, batch, rows)
    size = 1 << max(20, (probe.nbytes - 1).bit_length())
    return design_config(DRAM_BYTES=size, **kw)


class IsaBackend:
    """The bit-exact ISA simulator, one persistent machine (DRAM keeps the KV cache)."""

    def __init__(self, cfg: Config, images: list):
        self.machine = Machine(cfg, [[] for _ in range(cfg.S)], images)

    def write(self, s: int, addr: int, data: np.ndarray) -> None:
        v = np.ascontiguousarray(data).view(np.uint8).reshape(-1)
        self.machine.slices[s].dram[addr:addr + v.size] = v

    def read(self, s: int, addr: int, nbytes: int) -> np.ndarray:
        return self.machine.slices[s].dram[addr:addr + nbytes].copy()

    def run(self, programs: list) -> dict:
        self.machine.load(programs).run(max_steps=1 << 40)
        return {"instructions": [s.icount for s in self.machine.slices]}


class Engine:
    """Token-by-token decoding on an openTPU backend: Qwen3, or any model whose Spec builds an
    image with compile_step (LFM2: opentpu.llm.lfm2).

    backend: "isa" (default), or any object with write/read/run like IsaBackend (the RTL
    simulator and the PCIe board driver implement the same interface). Optional backend hooks:
    attach(engine), called once the engine exists; prepare(programs), called on the compile
    thread with every precompiled program (the board assembles it there).

    pipeline: step() compiles the next position's program (it depends on the position only,
    not on the token) on a worker thread while the backend runs the current one. Default: on
    for every backend but "isa" (whose run holds the GIL: nothing to overlap). A precompile is
    used only for the position it was made for; otherwise it is waited for and dropped (one
    trace at a time), so results do not change.
    """

    def __init__(self, spec: Spec, W: dict, cap: int = 4096, cfg: Config | None = None,
                 backend="isa", block: int = ATTN_BLOCK, batch: int = 1, rows: int = 1,
                 pipeline: bool | None = None):
        self.spec, self.cap, self.block = spec, cap, block
        self.batch, self.rows = batch, max(rows, batch)
        self.cfg = cfg or device_config(spec, cap, batch=batch, rows=self.rows)
        self.image = spec.image(self.cfg, cap, batch, self.rows)
        self.embed = np.asarray(W["model.embed_tokens.weight"], np.float32)
        images = self.image.build(W)
        self.backend = IsaBackend(self.cfg, images) if backend == "isa" else backend(
            self.cfg, images)
        self.poss = [0] * batch
        self.stats = []
        self.pipeline = backend != "isa" if pipeline is None else pipeline
        self._pool = None
        self._next = None                   # (pos, Future of its compiled programs)
        if hasattr(self.backend, "attach"):
            self.backend.attach(self)

    # ---- the compile pipeline
    def _compile(self, pos: int) -> list:
        progs = self.image.compile_step(pos, self.block)
        prep = getattr(self.backend, "prepare", None)
        if prep is not None:
            prep(progs)
        return progs

    def _program(self, pos: int) -> list:
        """The step program for `pos`: the precompiled one when it is for `pos`."""
        nxt, self._next = self._next, None
        if nxt is not None:
            progs = nxt[1].result()             # always wait: one compile at a time
            if nxt[0] == pos:
                return progs
        return self._compile(pos)

    def _prefetch(self, pos: int) -> None:
        if not self.pipeline or pos >= self.cap:
            return
        if self._pool is None:
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(1, thread_name_prefix="otpu-compile")
        self._next = (pos, self._pool.submit(self._compile, pos))

    def _drain(self) -> None:
        nxt, self._next = self._next, None
        if nxt is not None:
            nxt[1].result()

    @property
    def pos(self) -> int:
        """Next position of sequence 0 (the only one unless batch > 1)."""
        return self.poss[0]

    @pos.setter
    def pos(self, v: int) -> None:
        self.poss[0] = v

    def reset(self, seq: int | None = None) -> None:
        """Forget the context (the KV cache is overwritten from position 0 on)."""
        for s in range(self.batch) if seq is None else [seq]:
            self.poss[s] = 0

    def step(self, token: int) -> np.ndarray:
        """Feed one token at the next position; returns the logits [vocab] for the next one."""
        if self.pos >= self.cap:
            raise RuntimeError("KV cache full")
        io, S = self.image.io, self.cfg.S
        x = F.ftz(self.embed[token].astype(np.float32))
        cos, sin = rope_tables(self.spec, self.pos)
        for s in range(S):
            self.backend.write(s, io["x"], x)
            self.backend.write(s, io["cos"], cos)
            self.backend.write(s, io["sin"], sin)
        progs = self._program(self.pos)
        self._prefetch(self.pos + 1)
        st = self.backend.run(progs)
        self.stats.append(st)
        v_loc = self.image.v_loc
        parts = [self.backend.read(s, io["logits"] + 4 * s * v_loc, 4 * v_loc).view(np.float32)
                 for s in range(S)]
        self.pos += 1
        return np.concatenate(parts)

    def run_rows(self, rows, tokens, logit_rows) -> np.ndarray:
        """One device run over token rows (rows[r] = (sequence, position)); returns the logits
        of `logit_rows` ([n, vocab])."""
        io, S, spec = self.image.io, self.cfg.S, self.spec
        if any(p >= self.cap for _, p in rows):
            raise RuntimeError("KV cache full")
        x = F.ftz(self.embed[[int(t) for t in tokens]].astype(np.float32))
        cs = [rope_tables(spec, p) for _, p in rows]
        cos = np.stack([c for c, _ in cs]).astype(np.float32)
        sin = np.stack([s_ for _, s_ in cs]).astype(np.float32)
        for s in range(S):
            self.backend.write(s, io["x"], x)
            self.backend.write(s, io["cos"], cos)
            self.backend.write(s, io["sin"], sin)
        self._drain()
        st = self.backend.run(self.image.compile_rows(rows, logit_rows, self.block))
        st["rows"] = len(rows)
        self.stats.append(st)
        v, v_loc = spec.vocab, self.image.v_loc
        out = [np.concatenate([self.backend.read(s, io["logits"] + 4 * (r * v + s * v_loc),
                                                 4 * v_loc).view(np.float32) for s in range(S)])
               for r in logit_rows]
        return np.array(out, np.float32).reshape(len(logit_rows), v)

    def prefill(self, tokens, seq: int = 0, chunk: int | None = None) -> np.ndarray:
        """Feed a prompt to sequence `seq`; returns the logits after its last token.

        chunk=1 (the default for a one-row image) runs token by token with the decode kernel;
        chunk > 1 runs up to `chunk` prompt tokens per device run: their projections share each
        weight stream, and only the last chunk runs the LM head (for its last token)."""
        tokens = [int(t) for t in tokens]
        chunk = self.rows if chunk is None else max(1, min(chunk, self.rows))
        if chunk == 1 and seq == 0:
            logits = None
            for t in tokens:
                logits = self.step(t)
            return logits
        logits = None
        for i in range(0, len(tokens), chunk):
            part = tokens[i:i + chunk]
            last = i + chunk >= len(tokens)
            p0 = self.poss[seq]
            lg = self.run_rows([(seq, p0 + j) for j in range(len(part))], part,
                               [len(part) - 1] if last else [])
            self.poss[seq] += len(part)
            if last:
                logits = lg[0]
        return logits

    def step_batch(self, tokens) -> np.ndarray:
        """One token for each of the first len(tokens) sequences, each at its own next
        position (weights streamed once for all); returns logits [len(tokens), vocab]."""
        n = len(tokens)
        if n > self.batch:
            raise ValueError(f"{n} tokens for {self.batch} sequences")
        lg = self.run_rows([(s, self.poss[s]) for s in range(n)], tokens, list(range(n)))
        for s in range(n):
            self.poss[s] += 1
        return lg

    def generate_batch(self, prompts, max_new: int = 32, chunk: int | None = None) -> list:
        """Greedy generation for several prompts (one sequence each) decoded together; a
        finished sequence keeps its row (its extra tokens are dropped) until all finish."""
        n = len(prompts)
        nxt = [int(np.argmax(self.prefill(p, seq=s, chunk=chunk)))
               for s, p in enumerate(prompts)]
        out = [[] for _ in range(n)]
        done = [False] * n
        for _ in range(max_new):
            for s in range(n):
                if not done[s]:
                    out[s].append(nxt[s])
                    done[s] = (nxt[s] in self.spec.eos or len(out[s]) >= max_new
                               or self.poss[s] >= self.cap)
            if all(done):
                break
            nxt = [int(np.argmax(r)) for r in self.step_batch(nxt)]
        return out

    def generate(self, prompt, max_new: int = 32, sampler=None, on_token=None) -> list:
        """Greedy (or `sampler(logits) -> id`) generation; stops at an EOS token."""
        logits = self.prefill(prompt)
        out = []
        for _ in range(max_new):
            t = int(np.argmax(logits)) if sampler is None else int(sampler(logits))
            out.append(t)
            if on_token:
                on_token(t)
            if t in self.spec.eos or self.pos >= self.cap:
                break
            logits = self.step(t)
        return out
