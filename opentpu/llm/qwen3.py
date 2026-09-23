"""Qwen3 dense decoder (e.g. Qwen3-0.6B / 1.7B) on openTPU.

Pieces:
  Spec              model dimensions (from a Hugging Face config.json)
  load_weights      Hugging Face safetensors -> fp32 numpy arrays (HF parameter names)
  reference_logits  plain numpy forward pass (the math, fp32), for debugging
  Image             the per-slice DRAM layout: every layer's weights, norms and KV cache in one
                    fixed-size block (so a hardware loop walks the layers with one address
                    register), the tied LM head, and a small I/O area
  qwen3_step        the ol kernel for one decode token: 28 layers, final norm, LM head
  Engine            runs tokens on a backend (ISA simulator, RTL simulation or the board) and
                    keeps the KV cache in device DRAM between tokens

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
from ..kernels.lib import rmsnorm, rope
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
    projection (int8 + scales) and this slice's KV heads with room for `cap` tokens.
    """

    def __init__(self, spec: Spec, cfg: Config, cap: int):
        spec.check(cfg)
        if cap % cfg.D:
            raise ValueError("KV capacity must be a multiple of D")
        S, D = cfg.S, cfg.D
        H, d, F_ = spec.hidden, spec.head_dim, spec.ffn
        self.spec, self.cfg, self.cap = spec, cfg, cap
        self.nq_loc, self.nkv_loc = spec.n_q // S, spec.n_kv // S
        self.h_loc, self.f_loc, self.v_loc = H // S, F_ // S, spec.vocab // S
        b = _Bump()
        self.io = {"x": b.alloc(4 * H), "cos": b.alloc(2 * d), "sin": b.alloc(2 * d),
                   "gf": b.alloc(4 * H), "logits": b.alloc(4 * spec.vocab)}
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
        L["kv"] = [{"k": lb.alloc(cap * d), "ks": lb.alloc(4 * cap * (d // D)),
                    "vt": lb.alloc(d * cap), "vs": lb.alloc(4 * cap)}
                   for _ in range(self.nkv_loc)]
        self.lofs, self.LS = L, (lb.next + 4095) // 4096 * 4096
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
            heads = {sid + j * cfg.S: {k: off + v for k, v in r.items()}
                     for j, r in enumerate(lofs["kv"])}
            ns.kv = KVDesc(heads, self.cap, d, D, cfg.S, sid)
            return ns

        return SimpleNamespace(
            spec=spec, layer=layer, n_layers=spec.layers,
            x=_tdesc(self.io["x"], (1, H)), cos=_tdesc(self.io["cos"], (d // 2,)),
            sin=_tdesc(self.io["sin"], (d // 2,)), g_final=_tdesc(self.io["gf"], (H,)),
            logits=_tdesc(self.io["logits"], (1, spec.vocab)),
            head=_qdesc(*self.head, self.v_loc, H, D), v_loc=self.v_loc)


# =============================================================================== kernel
# Attention: tokens per flash block (256 halves the per-block vector-unit latency overhead of
# 128 at long contexts) and score blocks in flight per head.
ATTN_BLOCK = 256
ATTN_DEPTH = 3


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
    kh = rope(rmsnorm(k.reshape(nh, d), kn, eps), c, s_)          # [nkv_loc, d]
    vh = v.reshape(nh, d)
    for j, hh in enumerate(heads):
        ol.kv_append(kv, hh, pos, kh[j:j + 1, :], vh[j:j + 1, :])

    def queries(j):
        def emit():
            qj = ol.dot(xs, lw.wq[j * G * d:(j + 1) * G * d, :])  # [1, G*d]
            return rope(rmsnorm(qj.reshape(G, d), qn, eps), c, s_)
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
    spec, sid = m.spec, ol.program_id()
    x = ol.load(m.x)
    c, s_ = ol.load(m.cos), ol.load(m.sin)
    for li in ol.range(m.n_layers):
        lw = m.layer(li)
        x.set(_attention(x, lw, c, s_, pos, spec, block))
        x.set(_mlp(x, lw, spec))
    xs = ol.quantize(rmsnorm(x, ol.load(m.g_final), spec.eps))
    chunk = min(HEAD_CHUNK, ol.tmem_words() // 8)
    for c0 in range(0, m.v_loc, chunk):
        n = min(chunk, m.v_loc - c0)
        col = sid * m.v_loc + c0
        ol.store(m.logits[:, col:col + n], ol.dot(xs, m.head[c0:c0 + n, :]))


def compile_step(image: Image, pos: int, block: int = ATTN_BLOCK) -> list:
    progs = []
    for s in range(image.cfg.S):
        b = qwen3_step.trace(image.cfg, s, {"m": image.descriptors(s), "pos": pos,
                                            "block": block})
        progs.append(b.finish())
    return progs


# =============================================================================== engine
def device_config(spec: Spec, cap: int, **kw) -> Config:
    """The design configuration with DRAM sized for this model (power of two MiB)."""
    probe = Image(spec, design_config(DRAM_BYTES=1 << 40, **kw), cap)
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
    """Token-by-token Qwen3 on an openTPU backend.

    backend: "isa" (default), or any object with write/read/run like IsaBackend (the RTL
    simulator and the PCIe board driver implement the same interface).
    """

    def __init__(self, spec: Spec, W: dict, cap: int = 4096, cfg: Config | None = None,
                 backend="isa", block: int = ATTN_BLOCK):
        self.spec, self.cap, self.block = spec, cap, block
        self.cfg = cfg or device_config(spec, cap)
        self.image = Image(spec, self.cfg, cap)
        self.embed = np.asarray(W["model.embed_tokens.weight"], np.float32)
        images = self.image.build(W)
        self.backend = IsaBackend(self.cfg, images) if backend == "isa" else backend(
            self.cfg, images)
        self.pos = 0
        self.stats = []

    def reset(self) -> None:
        """Forget the context (the KV cache is overwritten from position 0 on)."""
        self.pos = 0

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
        st = self.backend.run(compile_step(self.image, self.pos, self.block))
        self.stats.append(st)
        v_loc = self.image.v_loc
        parts = [self.backend.read(s, io["logits"] + 4 * s * v_loc, 4 * v_loc).view(np.float32)
                 for s in range(S)]
        self.pos += 1
        return np.concatenate(parts)

    def prefill(self, tokens) -> np.ndarray:
        logits = None
        for t in tokens:
            logits = self.step(int(t))
        return logits

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
