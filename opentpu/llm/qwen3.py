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
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .. import fp32 as F
from .. import isa as I
from .. import quant as Q
from .. import language as ol
from ..compiler import Affine, CompileError, KVDesc, QTensor, RunVar, Tensor, arg_words
from . import generate as G
from ..isasim import Config, Machine, design_config
from ..kernels.attention import Bucket, _attend_heads
from ..kernels.layouts import head_parallel_attention_weights
from ..kernels.gather import dequant_row, gather_row, onehot, onehot_blocks
from ..kernels.lib import rmsnorm, rope, rope_rows, sigmoid, softcap
from ..kernels.mlp import _chunk, swiglu_down
from ..runtime import ALIGN


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
    # the Llama-like models of llama.py (SmolLM3, Phi-3 / Phi-4-mini) run on this code too:
    qk_norm: bool = True      # RMSNorm on each q and k head (Qwen3); Llama-likes have none
    nope: tuple = ()          # layers without RoPE (SmolLM3: every 4th)
    rotary: int = 0           # RoPE dimensions of a head, the first ones (0: all; Phi-4-mini 96)
    rope_div: tuple = ()      # per-frequency divisors of the angle (LongRoPE's short factors)
    rope_scale: float = 1.0   # factor on cos and sin (LongRoPE's attention factor)
    ctx: int = 0              # the most positions the RoPE tables hold (LongRoPE: its short
    #                           factors' range; 0: no limit)
    embed: str = "f32"        # the embedding rows: fp32, or "int8" (per D block, as the tied
    #                           int8 LM head holds them: the device gathers them from it,
    #                           kernels.gather.gather_row)

    @property
    def rope_dim(self) -> int:
        """RoPE rotates the first `rotary` dimensions of each head (default all of them)."""
        return self.rotary or self.head_dim

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

    def image(self, cfg: Config, cap: int, batch: int = 1, rows: int = 1,
              wformat: str = "int8", head_format: str | None = None,
              lookup: bool = False) -> "Image":
        return Image(self, cfg, cap, batch, rows, wformat, head_format, lookup)


class Weights(Mapping):
    """The tensors of a HF safetensors checkpoint, read and converted to fp32 numpy arrays
    when first used (load_weights): a multi-billion-parameter model is never all in memory in
    fp32 (LFM2-2.6B: 10 GB), the image build converts one tensor at a time. Tensors of at most
    CACHE bytes stay cached (norms, conv taps: read per token by the references), and the
    last larger one (the embedding table a reference indexes per token)."""

    CACHE = 16 << 20

    def __init__(self, model_dir):
        from safetensors import safe_open
        self._files, self._where = [], {}
        for f in sorted(Path(model_dir).glob("*.safetensors")):
            h = safe_open(str(f), framework="pt")
            self._files.append(h)
            for k in h.keys():
                if k.startswith(("model.visual.", "mtp.")):
                    continue
                self._where[k.replace("model.language_model.", "model.", 1)] = (h, k, None)
        self._split_fused(Path(model_dir))
        self._small: dict = {}
        self._big: tuple | None = None

    def _split_fused(self, model_dir: Path) -> None:
        """Phi-3's fused projections also under the names of separate ones: qkv_proj's rows
        are q, k and v (num_attention_heads, num_key_value_heads, num_key_value_heads heads),
        gate_up_proj's the gate's, then the up projection's (Phi3MLP: chunk(2))."""
        fused = [n for n in self._where if n.endswith(("self_attn.qkv_proj.weight",
                                                        "mlp.gate_up_proj.weight"))]
        if not fused:
            return
        c = json.loads((model_dir / "config.json").read_text())
        d = c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"]
        nq, nkv = c["num_attention_heads"] * d, c["num_key_value_heads"] * d
        ff = c["intermediate_size"]
        for n in fused:
            h, k, _ = self._where[n]
            if n.endswith("qkv_proj.weight"):
                p = n[:-len("qkv_proj.weight")]
                parts = (("q_proj", 0, nq), ("k_proj", nq, nq + nkv),
                         ("v_proj", nq + nkv, nq + 2 * nkv))
            else:
                p = n[:-len("gate_up_proj.weight")]
                parts = (("gate_proj", 0, ff), ("up_proj", ff, 2 * ff))
            for name, a, b in parts:
                self._where[p + name + ".weight"] = (h, k, (a, b))

    def __getitem__(self, name) -> np.ndarray:
        if name in self._small:
            return self._small[name]
        if self._big is not None and self._big[0] == name:
            return self._big[1]
        import torch
        h, k, rows = self._where[name]
        t = h.get_tensor(k) if rows is None else h.get_slice(k)[rows[0]:rows[1]]
        v = t.to(torch.float32).numpy()
        if v.nbytes <= self.CACHE:
            self._small[name] = v
        else:
            self._big = (name, v)
        return v

    def __iter__(self):
        return iter(self._where)

    def __len__(self) -> int:
        return len(self._where)

    def __contains__(self, name) -> bool:
        return name in self._where


def load_weights(model_dir) -> Weights:
    """All tensors of a HF safetensors checkpoint as fp32 numpy arrays, converted when first
    used (Weights). Of a multimodal checkpoint (Qwen3.5) only the language model is loaded,
    under the names of a text-only one (model.language_model.* -> model.*): not the vision
    tower or the multi-token prediction layers. Gemma 4: gemma4.load_weights (its PLE table
    read by rows)."""
    cfg = Path(model_dir) / "config.json"
    if cfg.exists() and json.loads(cfg.read_text()).get("model_type") in ("gemma4",
                                                                          "gemma4_text"):
        from .gemma4 import load_weights as gemma4_weights
        return gemma4_weights(model_dir)
    return Weights(model_dir)


def rope_tables(spec: Spec, pos: int) -> tuple[np.ndarray, np.ndarray]:
    """cos, sin [rope_dim/2] for one position (HF rotate-half convention); with LongRoPE's
    (spec.rope_div, spec.rope_scale) each frequency divided by its factor and both tables
    scaled by the attention factor."""
    half = spec.rope_dim // 2
    inv = 1.0 / (spec.theta ** (np.arange(half, dtype=np.float64) * 2 / spec.rope_dim))
    div = getattr(spec, "rope_div", ())
    if div:
        inv = inv / np.asarray(div, np.float64)
    ang = pos * inv
    sc = getattr(spec, "rope_scale", 1.0)
    return (np.cos(ang) * sc).astype(np.float32), (np.sin(ang) * sc).astype(np.float32)


# =============================================================================== lookup tables
def _lookup_alloc(b: "_Bump", spec, cap: int, block: int = None, D: int = 128,
                  head: tuple | None = None, M: int = 4) -> dict:
    """DRAM for the device-side inputs of a run-time position (RunPos): every token's embedding
    row, the RoPE cos / sin rows of every position, and the attention mask table
    (attention.Bucket: cap entries +inf, then a block of -inf). The embedding rows are fp32
    (flushed, as the host writes them), or with spec.embed "int8" int8 with a scale per D block
    (kernels.gather.gather_row, with its one-hot operand for M MXU columns): the LM head's
    rows when `head` (its data and scale addresses) is given -- a tied int8 head whole on one
    slice -- else a table of their own."""
    block = block or ATTN_BLOCK
    half = len(rope_tables(spec, 0)[0])
    V, H = spec.vocab, spec.hidden
    if getattr(spec, "embed", "f32") == "int8":
        emb = {"embed_q": head, "own": False} if head is not None else \
            {"embed_q": (b.alloc(V * H), b.alloc(4 * V * (H // D))), "own": True}
        emb.update(onehot=b.alloc(4 * M * onehot_blocks(D, M, "int8") * D), M=M)
    else:
        emb = {"embed": b.alloc(4 * V * H)}
    return {**emb, "cos_t": b.alloc(4 * cap * half), "sin_t": b.alloc(4 * cap * half),
            "zmask": b.alloc(4 * (cap + block)), "half": half, "block": block, "D": D,
            "gen": G.alloc(b, spec, cap, block)}


def _lookup_build(put, S: int, W: dict, spec, cap: int, lk: dict) -> None:
    cs = [rope_tables(spec, p) for p in range(cap)]
    z = np.concatenate([np.full(cap, np.inf, np.float32), np.full(lk["block"], -np.inf,
                                                                  np.float32)])
    if "embed" in lk:
        e = F.ftz(np.asarray(W["model.embed_tokens.weight"], np.float32))
    elif lk["own"]:
        eq, es = Q.quantize_mxu(W["model.embed_tokens.weight"], "int8", lk["D"])
    for s in range(S):
        if "embed" in lk:
            put(s, lk["embed"], e)
        elif lk["own"]:
            put(s, lk["embed_q"][0], eq)
            put(s, lk["embed_q"][1], es)
        put(s, lk["cos_t"], np.stack([c for c, _ in cs]))
        put(s, lk["sin_t"], np.stack([x for _, x in cs]))
        put(s, lk["zmask"], z)
        if "onehot" in lk:
            put(s, lk["onehot"], onehot(lk["D"], lk["M"], "int8"))
        G.build(put, s, S, spec, cap, lk["gen"])


def _lookup_desc(lk: dict, spec, cap: int) -> dict:
    if not lk:
        return {}
    d = {"cos_t": _tdesc(lk["cos_t"], (cap, lk["half"])),
         "sin_t": _tdesc(lk["sin_t"], (cap, lk["half"])), "zmask": lk["zmask"]}
    if "embed" in lk:
        d["embed"] = _tdesc(lk["embed"], (spec.vocab, spec.hidden))
    else:
        d["embed_q"] = _qdesc(*lk["embed_q"], spec.vocab, spec.hidden, lk["D"])
        M, D = lk["M"], lk["D"]
        d["onehot"] = _tdesc(lk["onehot"], (M, onehot_blocks(D, M, "int8") * D))
    d["gen"] = G.desc(lk["gen"], spec, cap)
    return d


class Embedding:
    """The embedding rows the device computes with, on the host: the checkpoint's fp32 rows
    (Engine.embed: the inputs the host writes for an image without lookup tables), or with
    spec.embed "int8" their int8 quantization as the device's gather dequantizes it, bit for bit
    (the reference; an Engine of such a model reads them from the image, batch 1)."""

    def __init__(self, spec, W, D: int):
        self.W, self.table = W, None        # read at the first row (never with device inputs)
        self.int8, self.D = getattr(spec, "embed", "f32") == "int8", D

    def __getitem__(self, idx) -> np.ndarray:
        if self.table is None:
            self.table = self.W["model.embed_tokens.weight"]
        rows = np.asarray(self.table[idx], np.float32)
        if not self.int8:
            return rows
        return gathered_rows(rows, self.D).reshape(rows.shape)


def gathered_rows(rows, D: int) -> np.ndarray:
    """The values kernels.gather.gather_row gives for these fp32 rows [n, K] held as int8 per D
    block (quant.quantize_mxu, as the LM head is stored): [n, K], bit for bit."""
    rows = np.atleast_2d(np.asarray(rows, np.float32))
    q, sc = Q.quantize_mxu(rows, "int8", D)
    return np.stack([dequant_row(q[i], sc[i], "int8", D) for i in range(len(rows))])


def _tok_arg(image, tok) -> dict:
    """The kernel argument of a compile-time token (none without one: the program is the
    host-input one)."""
    if tok is None:
        return {}
    if not image.lookup:
        raise ValueError("a program with its token's inputs from the image needs lookup tables")
    return {"tok": int(tok)}


def _tokens_arg(image, tokens, rows) -> dict:
    if tokens is None:
        return {}
    if not image.lookup:
        raise ValueError("rows with their inputs from the image need lookup tables")
    if len(tokens) != len(rows):
        raise ValueError(f"{len(tokens)} tokens for {len(rows)} rows")
    return {"tokens": [int(t) for t in tokens]}


def has_lookup(spec) -> bool:
    """The model's image can hold the resident decode's tables (Spec.image(lookup=True)) and
    compile_decode: Qwen3, LFM2, Qwen3.5."""
    import inspect
    return "lookup" in inspect.signature(spec.image).parameters


def compile_decode(image, kernel, blocks: int, lo: int, block: int | None = None):
    """The decode program of every position p in [lo, blocks * block), lo >= (blocks - 1) *
    block, at a run-time position (RunPos; the image needs lookup=True): (programs, run_args),
    run_args the programs' arguments (RunVar, coefficient) (compiler.arg_words gives the ARG
    register words of RunPos.values)."""
    if not image.lookup:
        raise ValueError("compile_decode needs an image with lookup tables (lookup=True)")
    block = block or ATTN_BLOCK
    if not (blocks - 1) * block <= lo < min(blocks * block, image.cap):
        raise ValueError(f"lo {lo} is not in bucket {blocks}")
    rp = RunPos(blocks, block, lo, image.lookup["zmask"], image.cap)
    bs = [kernel.trace(image.cfg, s, {"m": image.descriptors(s), "pos": rp, "block": block})
          for s in range(image.cfg.S)]
    progs = [b.finish() for b in bs]
    if any(b.run_args != bs[0].run_args for b in bs):
        raise CompileError("the slices' programs take different run-time arguments")
    return progs, list(bs[0].run_args)


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
        h, rd = spec.rope_dim // 2, spec.rope_dim
        v1, v2 = v[..., :h], v[..., h:rd]
        return np.concatenate([v1 * cos - v2 * sin, v2 * cos + v1 * sin, v[..., rd:]], axis=-1)

    mask = np.triu(np.full((T, T), -np.inf, np.float32), 1)
    for i in range(spec.layers):
        p = f"model.layers.{i}."
        h = norm(x, W[p + "input_layernorm.weight"])
        q = (h @ W[p + "self_attn.q_proj.weight"].T).reshape(T, spec.n_q, d)
        k = (h @ W[p + "self_attn.k_proj.weight"].T).reshape(T, spec.n_kv, d)
        v = (h @ W[p + "self_attn.v_proj.weight"].T).reshape(T, spec.n_kv, d)
        if spec.qk_norm:
            q = norm(q, W[p + "self_attn.q_norm.weight"])
            k = norm(k, W[p + "self_attn.k_norm.weight"])
        if i not in spec.nope:
            q, k = rot(q), rot(k)
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


def _fake_w(a, D: int, fmt: str = "int8"):
    """A weight matrix as the device holds it, in float64: int8 per (row, D-block) (_fake_q) or
    the 4-bit formats of opentpu/quant.py."""
    if fmt == "int8":
        return _fake_q(np.asarray(a, np.float64), D)
    return Q.quantize_w4(a, fmt, D)[2].astype(np.float64)


def emulated_logits(spec: Spec, W: dict, tokens, D: int = 128, wformat: str = "int8",
                    head_format: str | None = None) -> np.ndarray:
    """float64 decode that applies openTPU's quantization points but none of its rounding:
    int8 (or 4-bit: `wformat`, `head_format` as in Image) weights, int8 matmul inputs per
    D-block, int8 K (per token, D-block) and V (per token), int8 P (per D tokens). Separates
    quantization error from kernel bugs."""
    d, G = spec.head_dim, spec.n_q // spec.n_kv
    Wq: dict = {}
    head = "model.embed_tokens.weight" if spec.tied else "lm_head.weight"

    def w(n):
        if n not in Wq:
            Wq[n] = _fake_w(W[n], D, (head_format or wformat) if n == head else wformat)
        return Wq[n]

    def norm(v, g):
        return (v / np.sqrt(np.mean(v * v, -1, keepdims=True) + spec.eps)) * g

    Kc = [[] for _ in range(spec.layers)]
    Vc = [[] for _ in range(spec.layers)]
    out = []
    for pos, tk in enumerate(tokens):
        x = np.asarray(W["model.embed_tokens.weight"][tk], np.float64)
        if spec.embed == "int8":                    # the int8 embedding rows (Spec.embed)
            x = _fake_q(x, D)
        c, s = rope_tables(spec, pos)

        def rot(v):
            h, rd = spec.rope_dim // 2, spec.rope_dim
            v1, v2 = v[..., :h], v[..., h:rd]
            return np.concatenate([v1 * c - v2 * s, v2 * c + v1 * s, v[..., rd:]], -1)

        for i in range(spec.layers):
            p = f"model.layers.{i}."
            h = _fake_q(norm(x, W[p + "input_layernorm.weight"]), D)
            q = (w(p + "self_attn.q_proj.weight") @ h).reshape(spec.n_q, d)
            k = (w(p + "self_attn.k_proj.weight") @ h).reshape(spec.n_kv, d)
            v = (w(p + "self_attn.v_proj.weight") @ h).reshape(spec.n_kv, d)
            if spec.qk_norm:
                q = norm(q, W[p + "self_attn.q_norm.weight"])
                k = norm(k, W[p + "self_attn.k_norm.weight"])
            if i not in spec.nope:
                q, k = rot(q), rot(k)
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


def _qdesc(daddr: int, saddr: int, n: int, k: int, D: int, fmt: str = "int8") -> QTensor:
    return QTensor(Affine(daddr), Affine(saddr), (n, k), Q.row_bytes(k, fmt, D), 4 * (k // D), D,
                   wf=Q.mxu_wf(fmt))


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
    projection (quantized + scales) and this slice's KV heads with room for `cap` tokens, for
    each of `batch` sequences. The I/O area holds `rows` token rows (x, cos, sin, logits).
    A model with layers without RoPE (spec.nope) has a rope gate per layer block: (1, 0) or
    (0, 1), and each layer rotates with cos * g0 + g1 and sin * g0 (_rope_gate).

    Weight formats (opentpu/quant.py): `wformat` for the layers' projections, `head_format`
    (default: the same) for the LM head: "int8", or 4-bit "int4" / "fp4". The KV cache and the
    activations stay int8.
    """

    def __init__(self, spec: Spec, cfg: Config, cap: int, batch: int = 1, rows: int = 1,
                 wformat: str = "int8", head_format: str | None = None, lookup: bool = False):
        spec.check(cfg)
        if cap % cfg.D:
            raise ValueError("KV capacity must be a multiple of D")
        if spec.ctx and cap > spec.ctx:
            raise ValueError(f"KV capacity {cap} above the model's RoPE range ({spec.ctx})")
        S, D = cfg.S, cfg.D
        H, d, F_ = spec.hidden, spec.head_dim, spec.ffn
        self.spec, self.cfg, self.cap = spec, cfg, cap
        self.wformat, self.head_format = wformat, head_format or wformat
        rb = lambda k: Q.row_bytes(k, wformat, D)                       # noqa: E731
        self.batch, self.rows = batch, rows
        self.nq_loc, self.nkv_loc = spec.n_q // S, spec.n_kv // S
        self.h_loc, self.f_loc, self.v_loc = H // S, F_ // S, spec.vocab // S
        b = _Bump()
        R = rows
        rd = spec.rope_dim
        self.io = {"x": b.alloc(4 * H * R), "cos": b.alloc(2 * rd * R),
                   "sin": b.alloc(2 * rd * R), "gf": b.alloc(4 * H),
                   "logits": b.alloc(4 * spec.vocab * R)}
        self.layer0 = b.next
        lb = _Bump()                                    # offsets inside one layer block
        L = {"g_in": lb.alloc(4 * H), "g_post": lb.alloc(4 * H)}
        if spec.qk_norm:
            L.update(qn=lb.alloc(4 * d), kn=lb.alloc(4 * d))
        if spec.nope:
            L["rg"] = lb.alloc(8)
        self.mats = {"wq": (self.nq_loc * d, H), "wk": (self.nkv_loc * d, H),
                     "wv": (self.nkv_loc * d, H), "wo": (self.h_loc, spec.n_q * d),
                     "wg": (self.f_loc, H), "wu": (self.f_loc, H)}
        for name, (n, k) in self.mats.items():
            L[name] = (lb.alloc(n * rb(k)), lb.alloc(4 * n * (k // D)))
        # W_down in column parts of the MLP's F chunk: each down MM streams one part, whose
        # scales are then contiguous (with row-major scales every row would cost a DRAM beat)
        self.dchunk = _chunk(self.f_loc, D, D if wformat == "int8" else 2 * D)
        L["wd"] = [(lb.alloc(self.h_loc * rb(self.dchunk)),
                    lb.alloc(4 * self.h_loc * (self.dchunk // D)))
                   for _ in range(F_ // self.dchunk)]
        L["kvs"] = [[{"k": lb.alloc(cap * d), "ks": lb.alloc(4 * cap * (d // D)),
                      "vt": lb.alloc(d * cap), "vs": lb.alloc(4 * cap)}
                     for _ in range(self.nkv_loc)] for _ in range(batch)]
        L["kv"] = L["kvs"][0]
        self.lofs, self.LS = L, (lb.next + 4095) // 4096 * 4096
        head = cap * d + 4 * cap * (d // D) + d * cap + 4 * cap   # k, k scales, v^T, v scales
        self.kv_bytes = spec.layers * self.nkv_loc * head         # per sequence
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
        """DRAM images (one per slice) with every weight quantized in place, KV cache empty."""
        spec, cfg = self.spec, self.cfg
        S, D, d = cfg.S, cfg.D, spec.head_dim
        imgs = [np.zeros(self.nbytes, np.uint8) for _ in range(S)]

        def put(s, addr, a):
            v = np.ascontiguousarray(a).view(np.uint8).reshape(-1)
            imgs[s][addr:addr + v.size] = v

        def put_q(addr_pair, parts, fmt=self.wformat):
            for s, p in enumerate(parts):
                q, sc = Q.quantize_mxu(p, fmt, D)
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
                if spec.qk_norm:
                    put(s, Lo["qn"], f32(W[p + "self_attn.q_norm.weight"]))
                    put(s, Lo["kn"], f32(W[p + "self_attn.k_norm.weight"]))
                if spec.nope:
                    put(s, Lo["rg"], np.array([0, 1] if i in spec.nope else [1, 0], np.float32))
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
        put_q(self.head, rows(head, self.v_loc), self.head_format)
        if self.lookup:
            _lookup_build(put, S, W, spec, self.cap, self.lookup)
        return imgs

    # ---- programs
    def compile_decode(self, blocks: int, lo: int, block: int = ATTN_BLOCK):
        """(programs, run_args): qwen3_step at a run-time position (compile_decode)."""
        return compile_decode(self, qwen3_step, blocks, lo, block)

    def compile_generate(self, blocks: int, lo: int, block: int = ATTN_BLOCK,
                         chain: bool = True, samp=None, debug: bool = False,
                         part: int | None = None) -> list:
        """The decode loop on the device for bucket `blocks` (qwen3_step in it, generate.py)."""
        return G.compile_generate(self, qwen3_step, blocks, lo, block, chain, samp, debug, part)

    def compile_step(self, pos: int, block: int = ATTN_BLOCK, tok: int | None = None) -> list:
        """One program per slice: the decode token at position `pos` (qwen3_step); with `tok`
        (an image with lookup tables) its inputs come from the tables, not the host."""
        return [qwen3_step.trace(self.cfg, s, {"m": self.descriptors(s), "pos": pos,
                                               "block": block, **_tok_arg(self, tok)}).finish()
                for s in range(self.cfg.S)]

    def compile_rows(self, rows, logit_rows, block: int = ATTN_BLOCK, tokens=None) -> list:
        """One program per slice: token rows (sequence, position) at once (qwen3_rows); with
        `tokens` (an image with lookup tables) their inputs come from the tables."""
        if len(rows) > self.rows:
            raise ValueError(f"{len(rows)} rows, the image's I/O area holds {self.rows}")
        return [qwen3_rows.trace(self.cfg, s, {"m": self.descriptors(s), "rows": list(rows),
                                               "logit_rows": list(logit_rows), "block": block,
                                               **_tokens_arg(self, tokens, rows)}).finish()
                for s in range(self.cfg.S)]

    # ---- kernel descriptors
    def descriptors(self, sid: int) -> SimpleNamespace:
        spec, cfg = self.spec, self.cfg
        D, d, H = cfg.D, spec.head_dim, spec.hidden
        rh = spec.rope_dim // 2
        L0 = self.layer0
        lofs = self.lofs

        def layer(li):
            """Descriptors of layer `li` (an int or a hardware-loop variable)."""
            off = Affine.of(L0) + Affine.of(li) * self.LS
            ns = SimpleNamespace(
                g_in=Tensor(off + lofs["g_in"], (H,), (1,)),
                g_post=Tensor(off + lofs["g_post"], (H,), (1,)),
                qn=Tensor(off + lofs["qn"], (d,), (1,)) if spec.qk_norm else None,
                kn=Tensor(off + lofs["kn"], (d,), (1,)) if spec.qk_norm else None,
                rg=Tensor(off + lofs["rg"], (2,), (1,)) if spec.nope else None)
            fm, wf = self.wformat, Q.mxu_wf(self.wformat)
            for name, (n, k) in self.mats.items():
                da, sa = lofs[name]
                setattr(ns, name, QTensor(off + da, off + sa, (n, k), Q.row_bytes(k, fm, D),
                                          4 * (k // D), D, wf=wf))
            C, n = self.dchunk, self.h_loc
            rc = Q.row_bytes(C, fm, D)
            parts = tuple(QTensor(off + da, off + sa, (n, C), rc, 4 * (C // D), D, wf=wf)
                          for da, sa in lofs["wd"])
            ns.wd = QTensor(parts[0].data, parts[0].scale, (n, spec.ffn), rc, 4 * (C // D), D,
                            parts=parts, pw=C, wf=wf)
            ns.kvs = [KVDesc({sid + j * cfg.S: {k: off + v for k, v in r.items()}
                              for j, r in enumerate(heads)}, self.cap, d, D, cfg.S, sid)
                      for heads in lofs["kvs"]]
            ns.kv = ns.kvs[0]
            return ns

        return SimpleNamespace(
            spec=spec, layer=layer, n_layers=spec.layers,
            x=_tdesc(self.io["x"], (1, H)), cos=_tdesc(self.io["cos"], (rh,)),
            sin=_tdesc(self.io["sin"], (rh,)), g_final=_tdesc(self.io["gf"], (H,)),
            logits=_tdesc(self.io["logits"], (1, spec.vocab)),
            xr=_tdesc(self.io["x"], (self.rows, H)),
            cosr=_tdesc(self.io["cos"], (self.rows, rh)),
            sinr=_tdesc(self.io["sin"], (self.rows, rh)),
            logitsr=_tdesc(self.io["logits"], (self.rows, spec.vocab)),
            head=_qdesc(*self.head, self.v_loc, H, D, self.head_format), v_loc=self.v_loc,
            **_lookup_desc(self.lookup, spec, self.cap))


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


def _rope_padded(x, c, s_, out=None):
    """rope(x) on the first 2 * len(c) dimensions of each row (the others pass through:
    Qwen3.5's partial RoPE), padded like _padded (RoPE writes straight into the padded tile),
    or into `out` (a view whose columns beyond x.cols are already zero)."""
    d, D, rd = x.cols, ol.block_size(), 2 * c.cols
    if out is None and d % D == 0 and rd == d:
        return rope(x, c, s_)
    if out is None:
        out = ol.zeros([x.rows, -(-d // D) * D]) if d % D else ol.empty(x.shape)
    rope(x[:, :rd], c, s_, out=out[:, :rd])
    if rd < d:
        out[:, rd:d].set(x[:, rd:])
    return out


def _norm_heads(x, g, eps):
    """RMSNorm on each row (one head) with gamma `g` (a loaded q_norm / k_norm), or x itself
    for a model without them (g None: the Llama-likes)."""
    return x if g is None else rmsnorm(x, g, eps)


def _rope_gate(lw, c, s_):
    """The layer's RoPE tables: (c, s_) themselves, or with a rope gate (spec.nope: a model
    with layers without RoPE) c * g0 + g1 and s_ * g0 -- (c, s_) bit for bit where g = (1, 0),
    the identity rotation (1, +-0) where g = (0, 1). One loop body then serves both kinds."""
    if getattr(lw, "rg", None) is None:
        return c, s_
    g = ol.load(lw.rg)
    return c * g[0:1] + g[1:2], s_ * g[0:1]


def _load_opt(desc):
    return None if desc is None else ol.load(desc)


class RunPos:
    """The decode position as a run-time value (docs/isa.md "Arguments"): one program serves
    every position p of a bucket, lo <= p < blocks * block, whose attention spans `blocks`
    blocks (the last one masked: attention.Bucket). The program sees p as t0 + tpos, t0 =
    (blocks - 1) * block the bucket's first position and tpos = p - t0 < block the run-time
    value (so the V^T append stays in its 256-token tile: compiler.tile_split). The token's
    embedding and RoPE rows come from the image's tables (Image(lookup=True)) at the token id
    and the position; the KV cache appends at the position; ring is (p + 1) mod K, the first
    row of the last K in a mirrored K-row state ring (LFM2's convolutions: lfm2._ring_rows).
    values(token, p, K, block) and compiler.arg_words give the argument words of a program's
    run_args."""

    tok, ring = RunVar("tok"), RunVar("ring")

    def __init__(self, blocks: int, block: int, lo: int, zmask: int, cap: int):
        self.blocks, self.block, self.lo = blocks, block, lo
        self.t0 = (blocks - 1) * block
        self.tpos = RunVar("tpos", bound=min(block, cap - self.t0))
        self.pos = self.tpos + self.t0
        # the mask row of a block at t0' starts at entry cap - 1 - p + t0' of the table
        self.bucket = Bucket(blocks, Affine(zmask + 4 * (cap - 1 - self.t0)) + self.tpos * -4)

    @staticmethod
    def values(token: int, p: int, K: int = 1, block: int = ATTN_BLOCK) -> dict:
        return {"tpos": p % block, "tok": token, "ring": (p + 1) % K}


def _attention(x, lw, c, s_, pos: int, spec: Spec, block: int, gated: bool = False):
    """x + W_o . attention(x) for one token, this slice's heads; returns the new residual
    (replicated on every slice).

    Schedule (the MXU streams weights in program order, so what sits between two MMs in the
    stream overlaps them): K, V and then Q (one MM per KV head's query group) are projected
    first, back to back, and the K / V norms, RoPE and K appends run while Q streams. Each
    head's V^T append and query preparation follow in the attention pipeline (_attend_heads),
    so the quantizer's slow V^T appends (a byte into each of d cache rows) overlap the heads
    before it instead of holding all queries back. The Q MMs come before the appends and
    queries in program order so that the sequencer's window, which the appends fill, never
    holds the MXU's next MM back.

    A query group of more heads than the MXU has columns attends in parts of MCOLS heads, each
    streaming the KV head again. `gated` (Qwen3.5): lw.wgate projects a gate per query
    dimension, and the attention output is multiplied by sigmoid(gate) before W_o."""
    d, G, eps = spec.head_dim, spec.n_q // spec.n_kv, spec.eps
    kv = lw.kv
    kpos = pos.pos if isinstance(pos, RunPos) else pos
    xs = ol.quantize(rmsnorm(x, ol.load(lw.g_in), eps))
    k = ol.dot(xs, lw.wk)                       # [1, nkv_loc*d]
    v = ol.dot(xs, lw.wv)
    sg = sigmoid(ol.dot(xs, lw.wgate)) if gated else None       # [1, nq_loc*d]
    qn, kn = _load_opt(lw.qn), _load_opt(lw.kn)
    c, s_ = _rope_gate(lw, c, s_)
    scale = ol.LOG2E / math.sqrt(d)
    heads = list(kv.owned_heads(spec.n_kv))
    nh = len(heads)
    qps = [ol.dot(xs, lw.wq[j * G * d:(j + 1) * G * d, :]) for j in range(nh)]  # [1, G*d]
    kh = _rope_padded(_norm_heads(k.reshape(nh, d), kn, eps), c, s_)  # [nkv_loc, d or D]
    vh = _padded(v.reshape(nh, d))
    for j, hh in enumerate(heads):
        ol.kv_append(kv, hh, kpos, kh[j:j + 1, :], None)

    def queries(j):
        def emit():
            # the head's V^T append (byte-strided, the quantizer's slowest store) just ahead of
            # its queries: head j's scores start after K and V_0..V_j, not after all V appends
            ol.kv_append(kv, heads[j], kpos, None, vh[j:j + 1, :])
            return _rope_padded(_norm_heads(qps[j].reshape(G, d), qn, eps), c, s_)
        return emit

    mc = min(G, ol.mxu_columns())
    parts = [(j, g0, min(G, g0 + mc)) for j in range(nh) for g0 in range(0, G, mc)]
    if mc == G:
        qhs = [queries(j) for j in range(nh)]
    else:
        groups = {}

        def part(j, g0, g1):
            def emit():                         # the group's queries, made for its first part
                if j not in groups:
                    groups[j] = queries(j)()
                qg = groups.pop(j) if g1 == G else groups[j]
                return qg[g0:g1, :]
            return emit
        qhs = [part(*p) for p in parts]
    seq = pos.bucket if isinstance(pos, RunPos) else pos + 1
    outs = _attend_heads(qhs, kv, [heads[j] for j, _, _ in parts], seq, block, scale,
                         depth=ATTN_DEPTH, ahead=2)
    o_row = ol.empty([1, G * nh * d])
    o_loc = o_row.reshape(G * nh, d)            # the heads' outputs, written in place
    for (j, g0, g1), (acc, l) in zip(parts, outs):
        r0, r1 = j * G + g0, j * G + g1
        if sg is None:
            o_loc[r0:r1, :].set(acc / l[:, None])
        else:
            o_loc[r0:r1, :].set((acc / l[:, None]) * sg.reshape(G * nh, d)[r0:r1, :])
    o_all = ol.all_gather(o_row)                # [1, n_q*d], slice-major head order
    y = ol.all_gather(ol.dot(o_all, lw.wo))     # [1, H]
    return x + y


def _mlp(x, lw, spec: Spec):
    sid = ol.program_id()
    xs = ol.quantize(rmsnorm(x, ol.load(lw.g_post), spec.eps))
    y = swiglu_down(xs, lw.wg, lw.wu, lw.wd, loop=getattr(lw, "mlp_loop", False))
    h_loc = lw.wd.shape[0]
    mine = slice(sid * h_loc, (sid + 1) * h_loc)
    return ol.all_gather(x[:, mine] + y)


# Token rows per device run of a prefill: the I/O area holds this many, and TMEM (64K words)
# holds the activations of 8 rows of Qwen3-0.6B (Engine.prefill_chunks shrinks a run that does
# not fit).
PREFILL_ROWS = 8
COMPILE_AHEAD = 3    # decode programs compiled ahead by the worker processes (Engine)
DECODE_LEAD = 16     # resident decode: the next bucket's program is compiled from this many
                     # positions before the current bucket ends (Engine)
HEAD_CHUNK = 8192     # LM head rows per MM (the fp32 logits of one chunk must fit TMEM)


@ol.jit
def qwen3_step(m, pos: int, block: int = ATTN_BLOCK, tok: int | None = None):
    """One decode token at position `pos`: x (the token's embedding) -> logits.

    The layers run as a hardware loop; each layer appends its K/V at `pos` and attends over
    positions 0..pos. Logits for this slice's vocabulary rows are stored to m.logits. `tok`:
    the token's id, its inputs read from the image's tables (_inputs).
    """
    spec = m.spec
    x, c, s_ = _inputs(m, pos, tok)
    for li in ol.range(m.n_layers):
        lw = m.layer(li)
        x.set(_attention(x, lw, c, s_, pos, spec, block))
        x.set(_mlp(x, lw, spec))
    _lm_head(x, m, spec)


def _inputs(m, pos, tok=None):
    """The token's embedding row and its RoPE rows: from the I/O area (the host writes them),
    or from the image's tables (Image(lookup=True)) at the token id and position -- at a
    run-time position (RunPos), or at a compile-time token `tok` (a per-position program)."""
    if isinstance(pos, RunPos):
        x = _embed(m, pos.tok)
        ol.release(pos.tok)                 # its argument registers serve addresses from here on
        return x, ol.load(m.cos_t[pos.pos, :]), ol.load(m.sin_t[pos.pos, :])
    if tok is None:
        return ol.load(m.x), ol.load(m.cos), ol.load(m.sin)
    return _embed(m, tok), ol.load(m.cos_t[pos, :]), ol.load(m.sin_t[pos, :])


def _embed(m, tok):
    """Token `tok`'s embedding row [1, H] (an int or a run-time value) from the image's tables:
    the fp32 table's row, or the int8 row gathered on the device (Spec.embed "int8")."""
    eq = getattr(m, "embed_q", None)
    return ol.load(m.embed[tok:tok + 1, :]) if eq is None else next(_gather(m, eq, [tok]))


def _inputs_rows(m, rows, tokens=None):
    """_inputs for token rows (rows[r] = (sequence, position)): from the I/O area, or with
    `tokens` (their ids, compile-time values) from the image's tables, the embedding row of
    each token and the RoPE rows of each run of consecutive positions."""
    R = len(rows)
    if tokens is None:
        return ol.load(m.xr[0:R, :]), ol.load(m.cosr[0:R, :]), ol.load(m.sinr[0:R, :])
    eq = getattr(m, "embed_q", None)
    # x, c and s first, side by side as the host path loads them: the gathers' temporaries
    # after them are freed whole (TMEM does not fragment)
    x = ol.empty([R, m.xr.shape[1]], dense=True)
    c, s_ = ol.empty([R, m.cosr.shape[1]], dense=True), ol.empty([R, m.sinr.shape[1]], dense=True)
    if eq is None:
        for r, t in enumerate(tokens):
            ol.load(m.embed[t:t + 1, :], out=x[r:r + 1, :])
    for _, p0, r0, n in _runs(rows):
        ol.load(m.cos_t[p0:p0 + n, :], out=c[r0:r0 + n, :])
        ol.load(m.sin_t[p0:p0 + n, :], out=s_[r0:r0 + n, :])
    if eq is not None:
        for r, g in enumerate(_gather(m, eq, tokens)):
            x[r:r + 1, :].set(g)
    return x, c, s_


def _gather(m, eq, toks):
    """The int8 embedding rows of tokens `toks` (ints or run-time values), gathered on the
    device (kernels.gather.gather_row, one one-hot operand for all), one [1, H] tile at a
    time."""
    oh = ol.quantize(ol.load(m.onehot))
    for t in toks:
        yield gather_row(oh, eq, t, "int8").reshape(1, eq.shape[1])


def _lm_head(x, m, spec):
    """Final norm and this slice's vocabulary rows of the LM head -> m.logits, or, with
    m.lm_sink set (the generate loop: opentpu/llm/generate.py), each chunk's logits tile to
    m.lm_sink(tile, first vocabulary row) instead (and to m.logits too with m.lm_keep: the
    generate loop's debug mode). With m.lm_split (the first part of a split generate program)
    x itself to that DRAM tensor instead: the second part runs the head. A spec with a
    softcap (Gemma's final_logit_softcapping) caps the chunks a sink samples from (lib.softcap;
    not Greedy's, a raw sink: the cap keeps the order, and m.logits stays raw)."""
    split = getattr(m, "lm_split", None)
    if split is not None:
        ol.store(split, x)
        return
    sid = ol.program_id()
    xs = ol.quantize(rmsnorm(x, ol.load(m.g_final), spec.eps))
    chunk = min(HEAD_CHUNK, ol.tmem_words() // 8)
    sink = getattr(m, "lm_sink", None)
    for c0 in range(0, m.v_loc, chunk):
        n = min(chunk, m.v_loc - c0)
        col = sid * m.v_loc + c0
        if sink is None:
            ol.store(m.logits[:, col:col + n], ol.dot(xs, m.head[c0:c0 + n, :]))
        else:
            y = ol.dot(xs, m.head[c0:c0 + n, :])
            if getattr(m, "lm_keep", False):
                ol.store(m.logits[:, col:col + n], y)
            cap = getattr(spec, "softcap", None)
            if cap and not getattr(sink, "raw", False):
                y = softcap(y, cap)
            sink(y, col)


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


def _rope_rows_padded(x, c, s_, out=None):
    """rope_rows(x) on the first 2 * c.cols dimensions of each row (the others pass through),
    into `out` (a view whose columns beyond x.cols are already zero) or a new tile padded like
    _padded: the multi-row twin of _rope_padded, bit for bit."""
    d, D, rd = x.cols, ol.block_size(), 2 * c.cols
    if out is None:
        out = ol.zeros([x.rows, -(-d // D) * D]) if d % D else ol.empty(x.shape)
    rope_rows(x[:, :rd], c, s_, out=out[:, :rd])
    if rd < d:
        out[:, rd:d].set(x[:, rd:])
    return out


def _attention_rows(x, lw, c, s_, rows, spec, block: int, gated: bool = False):
    """x + W_o . attention(x) for R token rows; row r is token position rows[r][1] of sequence
    rows[r][0] (its own KV cache). Every row's K/V is appended first, then each row attends
    over positions 0..pos of its sequence -- for consecutive rows of one sequence (a prefill
    chunk) that is exactly the causal mask. Each projection streams its weights once for all
    R rows (ceil(R / MCOLS) MMs); the (row, KV head) pairs then run as one pipelined
    flash-attention stream (_attend_heads). Per row the arithmetic is _attention's, so the
    results are bit-identical to R decode steps: heads narrower than D (LFM2) are padded, RoPE
    may cover part of a head (Qwen3.5), `gated` multiplies the output by sigmoid(W_gate x),
    and a query group wider than the MXU attends in parts of MCOLS heads."""
    d, G, eps = spec.head_dim, spec.n_q // spec.n_kv, spec.eps
    R = len(rows)
    xs = ol.quantize(rmsnorm(x, ol.load(lw.g_in), eps))
    k = ol.dot(xs, lw.wk)                       # [R, nkv_loc*d]
    v = ol.dot(xs, lw.wv)
    q = ol.dot(xs, lw.wq)                       # [R, nq_loc*d]
    qn, kn = _load_opt(lw.qn), _load_opt(lw.kn)
    c, s_ = _rope_gate(lw, c, s_)
    scale = ol.LOG2E / math.sqrt(d)
    heads = list(lw.kv.owned_heads(spec.n_kv))
    nh = len(heads)
    nq = nh * G
    for j, hh in enumerate(heads):
        kj = _rope_rows_padded(_norm_heads(k[:, j * d:(j + 1) * d], kn, eps), c, s_)
        vj = _padded(v[:, j * d:(j + 1) * d])
        for sq, p0, r0, n in _runs(rows):
            ol.kv_append(lw.kvs[sq], hh, p0, kj[r0:r0 + n, :], vj[r0:r0 + n, :])
        del kj, vj
    del k, v
    # queries as [R * nq, dk]: (row r, KV head j) is the G contiguous rows r*nq + j*G ...
    dk = -(-d // ol.block_size()) * ol.block_size()
    Q = ol.zeros([R * nq, dk]) if dk > d else ol.empty([R * nq, d])
    for r in range(R):                          # a row's heads at once, as _attention's
        _rope_padded(_norm_heads(q[r:r + 1, :].reshape(nq, d), qn, eps), c[r], s_[r],
                     out=Q[r * nq:(r + 1) * nq, :])
    del q
    mc = min(G, ol.mxu_columns())
    ent = [(r, j, g0, min(G, g0 + mc)) for r in range(R) for j in range(nh)
           for g0 in range(0, G, mc)]
    o = ol.empty([R, nq * d])

    def emit(i, acc, l):
        r, j, g0, g1 = ent[i]
        o[r, (j * G + g0) * d:(j * G + g1) * d].reshape(g1 - g0, d).set(acc / l[:, None])

    _attend_heads([Q[r * nq + j * G + g0:r * nq + j * G + g1, :] for r, j, g0, g1 in ent],
                  [lw.kvs[rows[r][0]] for r, *_ in ent], [heads[j] for _, j, _, _ in ent],
                  [rows[r][1] + 1 for r, *_ in ent], block, scale, depth=ATTN_DEPTH,
                  emit=emit)
    del Q
    if gated:                                   # after the heads: o * sigmoid(gate), rounded
        for h0 in range(0, nq, mc):             # as _attention's (acc / l) * sg; mc heads at
            cols = slice(h0 * d, min(nq, h0 + mc) * d)      # a time (TMEM)
            o[:, cols].set(o[:, cols] * sigmoid(ol.dot(xs, lw.wgate[cols, :])))
    o_all = ol.all_gather(o)                    # [R, n_q*d]
    y = ol.all_gather(ol.dot(o_all, lw.wo))     # [R, H]
    return x + y


@ol.jit
def qwen3_rows(m, rows, logit_rows, block: int = ATTN_BLOCK, tokens=None):
    """R token rows at once (rows[r] = (sequence, position)): the rows' embeddings m.xr and
    RoPE tables m.cosr / m.sinr, or those of `tokens` from the image's tables (_inputs_rows)
    -> logits of the rows in `logit_rows` (a contiguous range, or empty: a prefill chunk that
    is not the last one skips the LM head)."""
    spec = m.spec
    x, c, s_ = _inputs_rows(m, rows, tokens)
    for li in ol.range(m.n_layers):
        lw = m.layer(li)
        x.set(_attention_rows(x, lw, c, s_, rows, spec, block))
        x.set(_mlp(x, lw, spec))
    _lm_head_rows(x, m, spec, logit_rows)


def _lm_head_rows(x, m, spec, logit_rows):
    """Final norm and this slice's vocabulary rows of the LM head for the rows `logit_rows`
    of x (a contiguous range, or empty: nothing) -> m.logitsr."""
    if not logit_rows:
        return
    sid = ol.program_id()
    a, e = logit_rows[0], logit_rows[-1] + 1
    xs = ol.quantize(rmsnorm(x[a:e, :], ol.load(m.g_final), spec.eps))
    chunk = min(HEAD_CHUNK, ol.tmem_words() // (8 * (e - a)))
    for c0 in range(0, m.v_loc, chunk):
        n = min(chunk, m.v_loc - c0)
        col = sid * m.v_loc + c0
        ol.store(m.logitsr[a:e, col:col + n], ol.dot(xs, m.head[c0:c0 + n, :]))


# =============================================================================== engine
def device_config(spec: Spec, cap: int, batch: int = 1, rows: int = 1, wformat: str = "int8",
                  head_format: str | None = None, lookup: bool = False, **kw) -> Config:
    """The design configuration with DRAM sized for this model (power of two MiB)."""
    probe = spec.image(design_config(DRAM_BYTES=1 << 40, **kw), cap, batch, rows, wformat,
                       head_format, **({"lookup": True} if lookup and has_lookup(spec) else {}))
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

    args = True                 # run(programs, args): the run's arguments (R8..R15)
    generates = True            # runs the generate loop (RLD, ARGMAX: Engine.generate_card)
    chains = True               # and HALT CHAIN: one run crosses the attention buckets

    def run(self, programs: list, args=None) -> dict:
        self.machine.load(programs, args).run(max_steps=1 << 40)
        return {"instructions": [s.icount for s in self.machine.slices]}


_WORKER: tuple | None = None                # (image, block) in the compile worker process


def _worker_init(spec, cfg, cap, batch, rows, block, wformat, head_format,
                 lookup: bool = False) -> None:
    """The worker's image: the engine's layout (weight formats, and the resident decode's
    lookup tables: its programs must address the same image)."""
    global _WORKER
    _WORKER = (spec.image(cfg, cap, batch, rows, wformat, head_format,
                          **({"lookup": True} if lookup else {})), block)
    _exit_with_parent()


def _exit_with_parent() -> None:
    """A worker whose parent died (killed, or os._exit) would wait for work forever: it holds
    its own end of the task pipe, so it never sees EOF. A watcher thread ends it."""
    import os
    import threading
    import time
    parent = os.getppid()

    def watch():
        while os.getppid() == parent:
            time.sleep(1.0)
        os._exit(0)
    threading.Thread(target=watch, daemon=True, name="otpu-parent-watch").start()


def _worker_ready() -> bool:
    return True


def _worker_compile(pos: int) -> np.ndarray:
    """The worker process: the decode program for `pos`, assembled (one slice)."""
    from ..isa import assemble
    image, block = _WORKER
    return np.asarray(assemble(image.compile_step(pos, block)[0]), np.uint32)


def _worker_decode(blocks: int, lo: int):
    """The worker process: the resident decode program of a bucket, assembled (one slice),
    and its run_args (compile_decode)."""
    from ..isa import assemble
    image, block = _WORKER
    progs, ra = image.compile_decode(blocks, lo, block)
    return np.asarray(assemble(progs[0]), np.uint32), ra


def _worker_chunk(seq: int, p0: int, n: int, left: int, fit: int, toks=None):
    """The worker process: fit_chunk's run, its program assembled (one slice)."""
    from ..isa import assemble
    image, block = _WORKER
    n, progs, fit = fit_chunk(image, block, seq, p0, n, left, fit, toks)
    return n, None if progs is None else np.asarray(assemble(progs[0]), np.uint32), fit


def fit_chunk(image, block: int, seq: int, p0: int, n: int, left: int, fit: int,
              tokens=None):
    """The next prefill run of sequence `seq` at position p0: up to n of the `left` remaining
    prompt tokens, as many as fit TMEM and ACT RAM (at most `fit` rows) and the instruction
    memory (attention is unrolled per row, head and block: the program grows with the context).
    Only the prompt's last run computes logits (its last row). Returns (R, compile_rows'
    programs or None for R = 1, the rows that fit TMEM as far as known). `tokens` (at least n):
    the run's inputs come from the image's tables (compile_rows tokens)."""
    imem = image.cfg.IMEM_WORDS
    n = min(n, fit, left)
    while n > 1:
        try:
            progs = image.compile_rows([(seq, p0 + j) for j in range(n)],
                                       [n - 1] if n == left else [], block,
                                       **({} if tokens is None else {"tokens": tokens[:n]}))
        except CompileError as e:
            if "TMEM" not in str(e) and "ACT RAM full" not in str(e):
                raise
            n = fit = n - 1
            continue
        size = max(map(len, progs))
        if size * 8 <= imem:
            return n, progs, fit
        n = min(n - 1, n * imem // (8 * size))  # about proportional to the rows
    return 1, None, fit


class Engine:
    """Decoding on an openTPU backend: Qwen3, or any model whose Spec builds an image with
    compile_step and compile_rows (LFM2: opentpu.llm.lfm2; Qwen3.5: opentpu.llm.qwen35).
    step() feeds one token per device run; prefill() / prefill_chunks() feed a prompt `rows`
    tokens per run (default PREFILL_ROWS), bit-identical to feeding it token by token.

    backend: "isa" (default), or any object with write/read/run like IsaBackend (the RTL
    simulator and the PCIe board driver implement the same interface). Optional backend hooks:
    attach(engine), called once the engine exists; prepare(programs), called on the compile
    thread with every precompiled program (the board assembles it there); start(programs) and
    wait() -> stats, the two halves of run() (step compiles the next program in between);
    streams (true: start(programs, stream=(addr, nbytes, piece)) and wait(feed) hand the logits
    over in pieces, most of them during the run; the engine's stream_logits turns it off);
    args (true: run / start take args=words, the run's arguments: resident decode).

    resident: decode with programs that take the position and the token as run arguments
    (compile_decode, docs/isa.md "Arguments"): one program per attention bucket of `block`
    positions, compiled once and kept in IMEM, the token's embedding and RoPE rows read from
    tables in the image (Image(lookup=True)), so a step writes no inputs and loads no program.
    Needs a backend with `args` (else it falls back), a model whose image has compile_decode
    (Qwen3, LFM2, Qwen3.5) and batch 1; positions below the model's first run-time one
    (LFM2, Qwen3.5: conv_k - 1) run per-position programs. Bit-identical to the per-position
    programs.

    device_inputs: with the image's tables (resident, or a model with an int8 embedding,
    spec.embed: its rows are dequantized on the device only) the prefill runs and per-position
    programs read their inputs from the image too, the token ids compiled into them
    (compile_rows tokens, compile_step tok): the host writes no inputs and computes nothing of
    the model. Without the tables the host writes the embedding rows and RoPE rows (of a table
    computed once, as the image's). A per-position program with its token is compiled in its
    step (pipeline: not ahead); resident decode compiles none.

    pipeline: step() compiles the next position's program (it depends on the position only,
    not on the token) while the backend runs the current one. Default: on for every backend
    but "isa" (whose run holds the GIL: nothing to overlap). A precompile is used only for the
    position it was made for; otherwise it is waited for and dropped (one trace at a time), so
    results do not change. The compile runs in a worker process when the backend runs
    assembled programs (runs_words: the board) and there is one slice -- a trace is 10-30 ms of
    Python, and in a thread it would hold the GIL the step's host work needs -- else on a worker
    thread; pipeline="thread" forces the thread. The process starts with the engine; until it
    is ready, steps compile in line.
    """

    def __init__(self, spec: Spec, W: dict, cap: int = 4096, cfg: Config | None = None,
                 backend="isa", block: int = ATTN_BLOCK, batch: int = 1,
                 rows: int = PREFILL_ROWS, pipeline: bool | str | None = None,
                 wformat: str = "int8", head_format: str | None = None,
                 resident: bool = False):
        self.spec, self.cap, self.block = spec, cap, block
        self.batch, self.rows = batch, max(rows, batch)
        wkw = dict(wformat=wformat, head_format=head_format)
        # an int8 embedding is dequantized on the device (the image's tables), never the host
        int8_embed = getattr(spec, "embed", "f32") == "int8"
        lookup = (bool(resident) or int8_embed) and batch == 1 and has_lookup(spec)
        if lookup:
            wkw["lookup"] = True
        self.cfg = cfg or device_config(spec, cap, batch=batch, rows=self.rows, **wkw)
        self.image = spec.image(self.cfg, cap, batch, self.rows, **wkw)
        # with the tables every run reads its inputs from the image (the token ids are compiled
        # into the per-position and prefill programs); else the host writes them: the
        # embedding rows and the RoPE rows of a table computed once, here
        self.device_inputs = bool(getattr(self.image, "lookup", None))
        self.embed = Embedding(spec, W, self.cfg.D)
        self._rope = None if self.device_inputs else \
            [np.stack(t) for t in zip(*(rope_tables(spec, p) for p in range(cap)))]
        images = self.image.build(W)
        self.backend = IsaBackend(self.cfg, images) if backend == "isa" else backend(
            self.cfg, images)
        self.resident = bool(resident) and lookup and bool(getattr(self.backend, "args", False))
        self._conv_lo = getattr(spec, "conv_k", 1) - 1    # the first run-time position
        self._decodes: dict = {}            # resident: blocks -> (programs, run_args)
        self._gens: dict = {}               # the generate loop: (blocks, mode) -> programs
                                            # (a list, or split: (first parts, second parts))
        self.gen_split = None               # split generate programs: None when a bucket's
                                            # does not fit, True always, False never
        self._chained: dict = {}            # mode -> (key, its buckets in the chain area)
        self.gen_debug = False              # generate_card: the logits too (gen_logits)
        self.gen_logits = None
        self.poss = [0] * batch
        self.stream_logits = True           # step(): stream the logits when the backend can
        # rows per run that fit TMEM (prefill_chunks), at most the image's fit_rows (Gemma 4:
        # the ACT rows, so that a run streams the weights once)
        self._fit_rows = min(self.rows, getattr(self.image, "fit_rows", self.rows))
        self._run_rows_n = self.rows        # rows of the last prefill run (IMEM may cut it)
        self.stats = []
        self.pipeline = backend != "isa" if pipeline is None else bool(pipeline)
        self._procs = (self.pipeline and pipeline != "thread" and self.cfg.S == 1
                       and getattr(self.backend, "runs_words", False))
        self._pool = None
        self._ready = None                  # the worker process's start (process pipeline)
        self._next = []                     # [(key, Future of a coming run's programs)]
        # per-position decode programs precompiled ahead: three worker processes, so a
        # compile can take three device runs (LFM2 fp4 on the card: a 13.5 ms run against a
        # 16-30 ms trace, more on a host busy with other work; resident decode compiles none)
        self._ahead = COMPILE_AHEAD if self._procs else 1
        if self._procs:
            self._start_pool()
        if hasattr(self.backend, "attach"):
            self.backend.attach(self)

    # ---- the compile pipeline
    def _compile(self, pos: int, tok: int | None = None) -> list:
        progs = self.image.compile_step(pos, self.block,
                                        **({} if tok is None else {"tok": int(tok)}))
        prep = getattr(self.backend, "prepare", None)
        if prep is not None:
            prep(progs)
        return progs

    def _start_pool(self) -> None:
        """The compile worker process (spawned: it inherits no device or lock descriptor)."""
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        self._pool = ProcessPoolExecutor(
            self._ahead, mp_context=mp.get_context("spawn"), initializer=_worker_init,
            initargs=(self.spec, self.cfg, self.cap, self.batch, self.rows, self.block,
                      self.image.wformat, self.image.head_format,
                      bool(getattr(self.image, "lookup", None))))
        self._ready = self._pool.submit(_worker_ready)

    def _take(self, key, fn, *args):
        """fn(*args), or the precompiled result (programs, or the worker process's assembled
        words) when a compile in flight was made for `key`; the ones queued before it, or all
        of them when none is for `key`, are waited for and dropped (the worker processes
        start with no compile queued)."""
        while self._next:
            k, fut = self._next.pop(0)
            res = fut.result()
            if k == key:
                return res
        return fn(*args)

    def _submit(self, key, fn, proc_fn, *args) -> None:
        """Start the compile for `key` (pipeline only): proc_fn(*args) in the worker process,
        else fn(*args) on the compile thread."""
        if not self.pipeline:
            return
        if self._procs:
            if self._ready.done():
                self._next.append((key, self._pool.submit(proc_fn, *args)))
            return
        if self._pool is None:
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(1, thread_name_prefix="otpu-compile")
        self._next.append((key, self._pool.submit(fn, *args)))

    def _program(self, pos: int):
        """The step program for `pos`: the precompiled one when it is for `pos`."""
        return self._take(("step", pos), self._compile, pos)

    def _compile_decode(self, blocks: int):
        progs, ra = self.image.compile_decode(blocks, max((blocks - 1) * self.block,
                                                          self._conv_lo), self.block)
        prep = getattr(self.backend, "prepare", None)
        if prep is not None:
            prep(progs)
        return progs, ra

    def _decode(self, pos: int):
        """Resident decode: (programs, run_args) of the bucket of `pos` (compiled once), or
        None for a per-position program."""
        if not self.resident or pos < self._conv_lo:
            return None
        b = pos // self.block + 1
        if b not in self._decodes:
            self._decodes[b] = self._take(("decode", b), self._compile_decode, b)
        return self._decodes[b]

    def _prefetch(self, pos: int) -> None:
        """Precompile the steps at pos .. pos + ahead - 1 (those not in flight yet); resident:
        the bucket of pos and, DECODE_LEAD positions before its end, the next one."""
        queued = {k for k, _ in self._next}
        if self.resident and pos >= self._conv_lo:
            for p in (pos, pos + DECODE_LEAD):
                b = p // self.block + 1
                if p < self.cap and b not in self._decodes and ("decode", b) not in queued:
                    self._submit(("decode", b), self._compile_decode, _worker_decode, b,
                                 max((b - 1) * self.block, self._conv_lo))
            return
        if self.device_inputs:              # its program has the token: compiled in step()
            return
        for p in range(pos, min(pos + self._ahead, self.cap)):
            if ("step", p) not in queued:
                self._submit(("step", p), self._compile, _worker_compile, p)

    def _drain(self) -> None:
        nxt, self._next = self._next, []
        for _, fut in nxt:
            fut.result()

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

    def step(self, token: int, on_start=None, sink=None) -> np.ndarray:
        """Feed one token at the next position; returns the logits [vocab] for the next one.
        on_start() is called once the device runs (host work that can overlap the run: the
        chat hands the previous token to its interface there). sink (a sampler's
        pick.stream(context), chat.sampler) gets the logits too: begin(vocab), then feed(lo,
        values) per piece -- on a backend that streams them (BoardBackend.streams), most
        pieces while the run goes on, so the sampler's work on them is off the token's
        critical path."""
        if self.pos >= self.cap:
            raise RuntimeError("KV cache full")
        io, S = self.image.io, self.cfg.S
        dec = self._decode(self.pos)
        if dec is None and self.device_inputs:  # the token's inputs from the image's tables
            progs, kw = self._compile(self.pos, token), {}
        elif dec is None:
            x = F.ftz(self.embed[token].astype(np.float32))
            cos, sin = self._rope[0][self.pos], self._rope[1][self.pos]
            if io["cos"] == io["x"] + x.nbytes and io["sin"] == io["cos"] + cos.nbytes:
                parts = [(io["x"], np.concatenate([x, cos, sin]))]      # one transfer
            else:
                parts = [(io["x"], x), (io["cos"], cos), (io["sin"], sin)]
            for s in range(S):
                for a, v in parts:
                    self.backend.write(s, a, v)
            progs, kw = self._program(self.pos), {}
        else:                               # the token and position as run arguments
            progs, ra = dec
            kw = {"args": arg_words(ra, RunPos.values(int(token), self.pos,
                                                      getattr(self.spec, "conv_k", 1),
                                                      self.block))}
        start = getattr(self.backend, "start", None)
        v_loc = self.image.v_loc
        vocab = S * v_loc
        stream = None
        if start is not None and S == 1 and self.stream_logits and \
                getattr(self.backend, "streams", False):
            piece = 4 * min(HEAD_CHUNK, self.cfg.TMEM_WORDS // 8)     # _lm_head's chunks
            stream = (io["logits"], 4 * vocab, piece)
        if start is None:
            if sink is not None:
                sink.begin(vocab)
            self._prefetch(self.pos + 1)
            if on_start is not None:
                on_start()
            st = self.backend.run(progs, **kw)
        elif stream is None:                # compile while the device runs, not while the
            start(progs, **kw)              # host copies the program
            if sink is not None:            # after the start: off the halt -> run path
                sink.begin(vocab)
            self._prefetch(self.pos + 1)
            if on_start is not None:
                on_start()
            st = self.backend.wait()
        else:
            logits = np.empty(vocab, np.float32)

            def feed(o, w):
                v = logits[o // 4:o // 4 + len(w)]
                v[:] = w.view(np.float32)
                if sink is not None:
                    sink.feed(o // 4, v)
            start(progs, stream=stream, **kw)
            if sink is not None:            # after the start: off the halt -> run path
                sink.begin(vocab)
            self._prefetch(self.pos + 1)
            if on_start is not None:
                on_start()
            st = self.backend.wait(feed)
        self.stats.append(st)
        if stream is None:
            logits = np.concatenate([self.backend.read(s, io["logits"] + 4 * s * v_loc,
                                                       4 * v_loc).view(np.float32)
                                     for s in range(S)])
            if sink is not None:
                sink.feed(0, logits)
        self.pos += 1
        return logits

    def run_rows(self, rows, tokens, logit_rows) -> np.ndarray:
        """One device run over token rows (rows[r] = (sequence, position)); returns the logits
        of `logit_rows` ([n, vocab])."""
        self._drain()
        return self._run_rows(rows, tokens, logit_rows,
                              self.image.compile_rows(rows, logit_rows, self.block,
                                                      **self._tokens_kw(tokens)))

    def _tokens_kw(self, tokens) -> dict:
        """compile_rows' tokens, with device inputs (the host writes none)."""
        return {"tokens": [int(t) for t in tokens]} if self.device_inputs else {}

    def _run_rows(self, rows, tokens, logit_rows, programs) -> np.ndarray:
        io, S, spec = self.image.io, self.cfg.S, self.spec
        if any(p >= self.cap for _, p in rows):
            raise RuntimeError("KV cache full")
        if not self.device_inputs:
            x = F.ftz(self.embed[[int(t) for t in tokens]].astype(np.float32))
            ps = [p for _, p in rows]
            for s in range(S):
                self.backend.write(s, io["x"], x)
                self.backend.write(s, io["cos"], self._rope[0][ps])
                self.backend.write(s, io["sin"], self._rope[1][ps])
        st = self.backend.run(programs)
        st["rows"] = len(rows)
        self.stats.append(st)
        v, v_loc = spec.vocab, self.image.v_loc
        out = [np.concatenate([self.backend.read(s, io["logits"] + 4 * (r * v + s * v_loc),
                                                 4 * v_loc).view(np.float32) for s in range(S)])
               for r in logit_rows]
        return np.array(out, np.float32).reshape(len(logit_rows), v)

    def _chunk(self, seq: int, p0: int, n: int, left: int, fit: int, toks=None):
        """fit_chunk, its programs prepared for the backend."""
        n, progs, fit = fit_chunk(self.image, self.block, seq, p0, n, left, fit, toks)
        prep = getattr(self.backend, "prepare", None)
        if progs is not None and prep is not None:
            prep(progs)
        return n, progs, fit

    def prefill_chunks(self, tokens, seq: int = 0, chunk: int | None = None):
        """Feed a prompt to sequence `seq` in device runs of up to `chunk` tokens (default:
        the image's rows); yields (the tokens of the run, logits) after each run, the logits
        after the prompt's last token with the last run and None before.

        A run of R > 1 tokens is one qwen3_rows program (the model's compile_rows): every
        weight streams once for the R rows (ceil(R / MCOLS) MMs), each row attends causally
        over the cache and the rows before it, and only the last run computes logits, for its
        last row. Per row the arithmetic is the decode kernel's, so the KV cache and logits
        are bit-identical to feeding the tokens one by one. A run shrinks when its program
        does not fit TMEM or IMEM (long contexts); a single token runs the decode kernel.
        With the pipeline, the next run's program (after the last run: the first decode
        step's) is compiled while the device runs the current one."""
        tokens = [int(t) for t in tokens]
        chunk = self.rows if chunk is None else max(1, min(chunk, self.rows))
        i = 0
        while i < len(tokens):
            p0, left = self.poss[seq], len(tokens) - i
            key = ("rows", seq, p0, chunk, left, self._fit_rows, self._chunk_toks(tokens[i:],
                                                                                  chunk))
            n, progs, self._fit_rows = self._take(key, self._chunk, *key[1:])
            part, last = tokens[i:i + n], n == left
            if not last:
                self._run_rows_n = n
            if progs is None and seq == 0:
                lg = self.step(part[0])
            else:
                rows, lr = [(seq, p0 + j) for j in range(n)], [n - 1] if last else []
                if progs is None:
                    progs = self.image.compile_rows(rows, lr, self.block,
                                                    **self._tokens_kw(part))
                if not last:
                    self._prefetch_chunks(seq, p0 + n, chunk, left - n, tokens[i + n:])
                elif seq == 0:
                    self._prefetch(p0 + n)
                lg = self._run_rows(rows, part, lr, progs)
                lg = lg[0] if last else None
                self.poss[seq] += n
            i += n
            yield part, (lg if last else None)

    def _chunk_toks(self, rest, chunk: int):
        """The tokens a prefill run's program is compiled with (device inputs: its first
        `chunk` of `rest`, the run takes as many as fit), or None."""
        return tuple(rest[:chunk]) if self.device_inputs else None

    def _prefetch_chunks(self, seq: int, p0: int, chunk: int, left: int, rest=()) -> None:
        """Precompile the runs of a prompt from position p0 on, as many as the pipeline has
        workers (a chunk's trace can take longer than its run: Qwen3.5 on the card), each
        predicted to take as many rows as the last one (TMEM and IMEM limit a run: the program
        grows with the context); a run whose prediction turns out wrong is waited for and
        dropped (_take), so the programs do not change."""
        queued = {k for k, _ in self._next}
        for _ in range(self._ahead):
            if left <= 0:
                break
            key = ("rows", seq, p0, chunk, left, self._fit_rows, self._chunk_toks(rest, chunk))
            if key not in queued:
                self._submit(key, self._chunk, _worker_chunk, *key[1:])
            n = min(chunk, self._fit_rows, self._run_rows_n, left)
            p0, left, rest = p0 + n, left - n, rest[n:]

    def prefill(self, tokens, seq: int = 0, chunk: int | None = None) -> np.ndarray:
        """Feed a prompt to sequence `seq`; returns the logits after its last token
        (prefill_chunks; chunk=1 runs token by token with the decode kernel)."""
        logits = None
        for _, logits in self.prefill_chunks(tokens, seq, chunk):
            pass
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

    # ---- the decode loop on the device (opentpu/llm/generate.py)
    @property
    def can_generate(self) -> bool:
        """generate_card works here: resident decode on a backend that runs the generate
        programs (the ISA simulator; the card with the ISA's RLD and ARGMAX)."""
        return (self.resident and self.image.lookup.get("gen") is not None
                and bool(getattr(self.backend, "generates", False)))

    def _generate_prog(self, blocks: int, samp=None):
        """The generate program of a bucket, greedy or sampled as `samp` (G.Sampling) is
        compiled for (once per bucket and samp.key), chaining to the next bucket's when the
        backend runs HALT CHAIN; with self.gen_debug the LM head also stores the logits (each
        token's over the last: generate_card reads the last token's)."""
        key = (blocks, None if samp is None else samp.key, self.gen_debug)
        if key not in self._gens:
            lo = max((blocks - 1) * self.block, self._conv_lo)
            progs = G.compile_bucket(self.image, blocks, lo, self.block,
                                     chain=bool(getattr(self.backend, "chains", False)),
                                     samp=samp, debug=self.gen_debug, split=self.gen_split)
            prep = getattr(self.backend, "prepare", None)
            if prep is not None:
                for p in (progs if isinstance(progs, tuple) else (progs,)):
                    prep(p)
            self._gens[key] = progs
        return self._gens[key]

    def _generate_chain(self, b0: int, b1: int, samp=None) -> None:
        """Buckets b0 + 1 .. b1 in the chain area of samp's mode and its table (each written
        once per samp.key)."""
        g = self.image.lookup["gen"]
        mode = int(samp is not None)
        key = (None if samp is None else samp.key, self.gen_debug)
        if self._chained.get(mode, (key,))[0] != key:
            self._chained.pop(mode)            # compiled for other sampling buffers
        have = self._chained.setdefault(mode, (key, set()))[1]
        # a split bucket chains back to its own first part: b0's programs too
        lo = b0 if isinstance(self._generate_prog(b0, samp), tuple) else b0 + 1
        new = [k for k in range(lo, b1 + 1) if k not in have]
        if not new:
            return
        for k in new:
            progs = self._generate_prog(k, samp)
            for j, pj in enumerate(progs if isinstance(progs, tuple) else (progs,)):
                for s, prog in enumerate(pj):
                    self.backend.write(s, G.prog_slot(g, k, mode, j), I.assemble(prog))
        have |= set(new)
        for s in range(self.cfg.S):
            words = {}
            for k in have:
                progs = self._generate_prog(k, samp)
                words[k] = (tuple(I.assemble(p[s]) for p in progs) if isinstance(progs, tuple)
                            else I.assemble(progs[s]))
            self.backend.write(s, G.ptab_addr(g, mode), G.ptab_words(g, {
                k: w[0] if isinstance(w, tuple) else w for k, w in words.items()}, mode))
            self.backend.write(s, G.ptab2_addr(g, mode), G.ptab_words(g, {
                k: w if isinstance(w, tuple) else (w,) for k, w in words.items()}, mode,
                split=True))

    def _generate_inputs(self, samp, p: int, nb: int, context, rng) -> None:
        """The sampled loop's per-run inputs: the uniforms of positions p + 1 .. p + nb (the
        host's generator), and, with the repetition penalty, its factors for the context's ids
        (the device adds the tokens it generates)."""
        g = self.image.lookup["gen"]
        u = np.minimum(rng.random(nb).astype(np.float32), np.float32(1 - 2.0 ** -24))
        pa = pb = None
        if samp.pen:
            V = G._vpad(self.spec)
            pa, pb = np.ones(V, np.float32), np.ones(V, np.float32)
            ix = np.unique(np.asarray(list(context), np.int64))
            pa[ix] = np.float32(1.0) / np.float32(samp.penalty)
            pb[ix] = samp.penalty
        for s in range(self.cfg.S):
            self.backend.write(s, g["uni"] + 4 * (p + 1), u)
            if pa is not None:
                self.backend.write(s, g["pa"], pa)
                self.backend.write(s, g["pb"], pb)

    def generate_card(self, tok: int, n: int, stop_ids=None, on_token=None,
                      stop=None, sampling=None, context=(), rng=None) -> list:
        """Feed `tok` at the next position and generate up to n tokens after it on the device:
        the generate loop picks each token there and feeds it back, and stops at a stop id
        (default: the model's EOS ids; the stop id is returned, not fed). Greedy, or with
        `sampling` (G.Sampling: temperature, top-k, top-p, repetition penalty over `context`,
        the ids so far with tok; the uniforms from `rng`, a numpy Generator). One device run
        per attention bucket reached (one in all with HALT CHAIN); the logits never leave the
        device. on_token(t) for each token as the host reads it; stop() (polled while the card
        runs, on backends that stream the tokens) halts the loop after the token in flight.
        Returns the tokens; self.pos is then the position of the last one (fed next, unless it
        is a stop id)."""
        if not self.can_generate:
            raise RuntimeError("generate_card needs resident decode on a backend that runs "
                               "the generate loop")
        samp = sampling
        if samp is not None and rng is None:
            raise ValueError("a sampled generate_card needs rng (the uniforms)")
        ids = list(self.spec.eos if stop_ids is None else stop_ids)
        n = min(n, self.cap - self.pos)
        out, g, ctx = [], self.image.lookup["gen"], list(context)
        while n > 0:
            p = self.pos
            if p < self._conv_lo:             # before the first run-time position
                lg = self.step(tok)
                got = [int(np.argmax(lg)) if samp is None else
                       G.reference_pick(lg, samp, ctx, rng.random(), self.cfg.S,
                                        getattr(self.spec, "softcap", None))]
                if on_token is not None:
                    on_token(got[0])
            else:
                b0 = p // self.block + 1
                b1 = b0
                if getattr(self.backend, "chains", False):     # the whole run on the device
                    b1 = (p + n - 1) // self.block + 1
                    self._generate_chain(b0, b1, samp)
                progs = self._generate_prog(b0, samp)
                if isinstance(progs, tuple):      # split: the run starts at the first part
                    progs = progs[0]
                nb = min(n, b1 * self.block - p)
                if samp is not None:
                    self._generate_inputs(samp, p, nb, ctx, rng)
                for s in range(self.cfg.S):
                    self.backend.write(s, g["state"], G.state_words(self.spec, tok, p, n, ids,
                                                                    self.block, samp))
                    self.backend.write(s, g["out"] + 4 * (p + 1),
                                       np.full(nb, G.OUT_MARK, np.uint32))
                runner = getattr(self.backend, "run_generate", None)
                if runner is not None:     # the card: tokens as they land, stop() halts
                    st, got = runner(progs, g["out"] + 4 * (p + 1), nb, on_token, stop,
                                     g["state"])
                else:
                    st = self.backend.run(progs)
                    w = self.backend.read(0, g["out"] + 4 * (p + 1), 4 * nb).view(np.uint32)
                    k = int(np.argmax(w == G.OUT_MARK)) if (w == G.OUT_MARK).any() else nb
                    got = [int(x) for x in w[:k].view(np.float32)]
                    if on_token is not None:
                        for t in got:
                            on_token(t)
                self.stats.append(st)
                self.pos += len(got)
                if self.gen_debug:          # the logits of the run's last token
                    io, v = self.image.io, self.image.v_loc
                    self.gen_logits = np.concatenate([
                        self.backend.read(s, io["logits"] + 4 * s * v, 4 * v).view(np.float32)
                        for s in range(self.cfg.S)])
            out += got
            ctx += got
            n -= len(got)
            if not got or got[-1] in ids or (stop is not None and stop()):
                break
            tok = got[-1]
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
