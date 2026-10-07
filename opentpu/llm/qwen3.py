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
from .. import progcache as PC
from .. import qcache as QC
from .. import quant as Q
from .. import language as ol
from ..compiler import (Affine, CompileError, DevVar, KVDesc, QTensor, RunVar, Tensor, arg_words,
                        current)
from . import formats as FM
from . import generate as G
from ..isasim import Config, Machine, design_config
from ..kernels.attention import Additive, Bucket, _attend_heads
from ..kernels.layouts import head_parallel_attention_weights
from ..host.offload import BackendDram, RowLayout, RowServer
from ..kernels import mailbox as MB
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
    formats: str = ""         # weight formats per kind over the image's wformat (KINDS)
    mix: str = ""             # the recommended mix (wformat "mix": formats.named, MIXES)

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
                    eos=tuple(eos) if isinstance(eos, list) else (eos,), mix=FM.mix_for(c))

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
              lookup: bool = False, embed_host: bool | None = None,
              formats: str | None = None) -> "Image":
        return Image(self, cfg, cap, batch, rows, wformat, head_format, lookup, embed_host,
                     formats)


class Weights(Mapping):
    """The tensors of a HF safetensors checkpoint, read and converted to fp32 numpy arrays
    when first used (load_weights): a multi-billion-parameter model is never all in memory in
    fp32 (LFM2-2.6B: 10 GB), the image build converts one tensor at a time. Tensors of at most
    CACHE bytes stay cached (norms, conv taps: read per token by the references), and the
    last larger one (the embedding table a reference indexes per token)."""

    CACHE = 16 << 20

    def __init__(self, model_dir, mtp: bool = False):
        from safetensors import safe_open
        self._files, self._where = [], {}
        skip = ("model.visual.",) if mtp else ("model.visual.", "mtp.")
        for f in sorted(Path(model_dir).glob("*.safetensors")):
            h = safe_open(str(f), framework="pt")
            self._files.append(h)
            for k in h.keys():
                if k.startswith(skip):
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


def load_weights(model_dir, mtp: bool = False) -> Weights:
    """All tensors of a HF safetensors checkpoint as fp32 numpy arrays, converted when first
    used (Weights). Of a multimodal checkpoint (Qwen3.5) only the language model is loaded,
    under the names of a text-only one (model.language_model.* -> model.*): not the vision
    tower, nor the multi-token prediction layers (mtp.*) unless `mtp` (Qwen3.5's drafter,
    opentpu/llm/mtp.py). Gemma 4: gemma4.load_weights (its PLE table read by rows)."""
    cfg = Path(model_dir) / "config.json"
    if cfg.exists() and json.loads(cfg.read_text()).get("model_type") in ("gemma4",
                                                                          "gemma4_text"):
        from .gemma4 import load_weights as gemma4_weights
        return gemma4_weights(model_dir)
    return Weights(model_dir, mtp)


class LazyWeights(dict):
    """load_weights' names over a checkpoint's safetensors files, each tensor read as fp32 when
    it is asked for and not kept: for a model whose fp32 weights would not fit host RAM (a
    MoE; its experts are packed one at a time, opentpu.llm.moe). release() gives the files'
    pages back once the image is built (a later read reopens its file)."""

    def __init__(self, model_dir):
        super().__init__()
        from safetensors import safe_open
        self._at, self._h = {}, {}          # name -> (file, its name there); file -> handle
        for f in sorted(Path(model_dir).glob("*.safetensors")):
            h = self._h[str(f)] = safe_open(str(f), "pt")
            for k in h.keys():
                if not k.startswith(("model.visual.", "mtp.")):
                    self._at[k.replace("model.language_model.", "model.", 1)] = (str(f), k)

    def _open(self, k):
        from safetensors import safe_open
        f, name = self._at[k]
        if f not in self._h:
            self._h[f] = safe_open(f, "pt")
        return self._h[f], name

    def __getitem__(self, k):
        import torch
        h, name = self._open(k)
        return h.get_tensor(name).to(torch.float32).numpy()

    def part(self, k, i):
        """Tensor k's i-th entry along its first axis (one expert of a fused expert tensor),
        read alone."""
        import torch
        h, name = self._open(k)
        return h.get_slice(name)[i].to(torch.float32).numpy()

    def release(self) -> None:
        """Close the files (safe_open maps each whole: the pages a read touched stay mapped,
        and the kernel keeps them over other page cache, a pool file's) and drop their pages
        (POSIX_FADV_DONTNEED, where the host has it). A tensor read after reopens its file."""
        import gc
        import os
        files = list(self._h)
        self._h.clear()
        gc.collect()                        # (a handle in a cycle would keep its map)
        for f in files if hasattr(os, "posix_fadvise") else ():
            fd = os.open(f, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)

    def __contains__(self, k):
        return k in self._at

    def __iter__(self):
        return iter(self._at)

    def __len__(self):
        return len(self._at)

    def keys(self):
        return self._at.keys()

    def items(self):
        return ((k, self[k]) for k in self._at)


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
                  head: tuple | None = None, M: int = 4, embed_host: bool = False,
                  rows: int = 1) -> dict:
    """DRAM for the device-side inputs of a run-time position (RunPos): every token's embedding
    row, the RoPE cos / sin rows of every position, and the attention mask table
    (attention.Bucket: cap entries +inf, then a block of -inf). The embedding rows are fp32
    (flushed, as the host writes them), or with spec.embed "int8" int8 with a scale per D block
    (kernels.gather.gather_row, with its one-hot operand for M MXU columns): the LM head's
    rows when `head` (its data and scale addresses) is given -- a tied int8 head whole on one
    slice -- else a table of their own, or with `embed_host` a slot of `rows` of them that the
    host fills from the table it keeps (embed_record: each row's int8 data, then its scale
    words) and a mailbox for the generate loop's requests (opentpu.host.offload.RowLayout).
    A dense model's image also holds the prompt runs' additive mask tiles (amask_table)."""
    block = block or ATTN_BLOCK
    half = len(rope_tables(spec, 0)[0])
    V, H = spec.vocab, spec.hidden
    if embed_host and (getattr(spec, "embed", "f32") != "int8" or head is not None):
        raise ValueError("embed_host: the model's int8 embedding table of its own (Spec.embed "
                         "int8, not the tied int8 head's rows)")
    if getattr(spec, "embed", "f32") == "int8":
        if head is not None:
            emb = {"embed_q": head, "own": False}
        elif embed_host:
            rec = embed_record(H, D)
            slot = b.alloc(rows * rec)
            emb = {"embed_q": (slot, slot + H), "own": False, "host": True, "rec": rec,
                   "rows": rows, "mbox": b.alloc(RowLayout.WORDS)}
        else:
            emb = {"embed_q": (b.alloc(V * H), b.alloc(4 * V * (H // D))), "own": True}
        emb.update(onehot=b.alloc(4 * M * onehot_blocks(D, M, "int8") * D), M=M)
    else:
        emb = {"embed": b.alloc(4 * V * H)}
    if getattr(spec, "moe", None) is None and not getattr(spec, "experts", 0):
        emb.update(amask=b.alloc(amask_table(block, M).nbytes), amask_M=M)
    return {**emb, "cos_t": b.alloc(4 * cap * half), "sin_t": b.alloc(4 * cap * half),
            "zmask": b.alloc(4 * (cap + block)), "half": half, "block": block, "D": D,
            "gen": G.alloc(b, spec, cap, block)}


def amask_table(block: int, M: int, start: bool = False) -> np.ndarray:
    """attention.Additive's mask tiles for a prompt run's rows (docs/prefill.md 9), one a
    position q in an attention block: M rows (MCOLS: a score buffer's) of the block's entries c
    and a pad word (the buffer's row stride), -0 for c <= q (the row's own token and those
    before it) and -inf after; [block, M, block + 1] fp32, 1 MiB at 256 and MCOLS 4. start:
    the opposite (a sliding window's first block: -inf for c <= q, -0 after)."""
    c = np.arange(block + 1)[None, :]
    keep = ((c > np.arange(block)[:, None]) if start else (c <= np.arange(block)[:, None])) | \
        (c == block)
    tile = np.where(keep, np.float32(-0.0), np.float32(-np.inf)).astype(np.float32)
    return np.ascontiguousarray(np.broadcast_to(tile[:, None, :], (block, M, block + 1)))


def _amask(lk: dict):
    """RunRows' amask: the table's address and its bytes a position, or None (no table)."""
    if "amask" not in lk:
        return None
    return lk["amask"], 4 * lk["amask_M"] * (lk["block"] + 1)


def _lookup_build(put, S: int, W: dict, spec, cap: int, lk: dict) -> None:
    cs = [rope_tables(spec, p) for p in range(cap)]
    z = np.concatenate([np.full(cap, np.inf, np.float32), np.full(lk["block"], -np.inf,
                                                                  np.float32)])
    if "embed" in lk:
        e = F.ftz(np.asarray(W["model.embed_tokens.weight"], np.float32))
    elif lk["own"]:
        eq, es = Q.quantize_mxu(W["model.embed_tokens.weight"], "int8", lk["D"])
    elif lk.get("host"):                    # the host's table, in the slot's record format
        lk["store"] = embed_store(W["model.embed_tokens.weight"], lk["D"])
    for s in range(S):
        if "embed" in lk:
            put(s, lk["embed"], e)
        elif lk["own"]:
            put(s, lk["embed_q"][0], eq)
            put(s, lk["embed_q"][1], es)
        put(s, lk["cos_t"], np.stack([c for c, _ in cs]))
        put(s, lk["sin_t"], np.stack([x for _, x in cs]))
        put(s, lk["zmask"], z)
        if "amask" in lk:
            put(s, lk["amask"], amask_table(lk["block"], lk["amask_M"]))
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
    elif lk.get("host"):        # the slot: rows of records (the int8 row, then its scales)
        a, sa = lk["embed_q"]
        d["embed_q"] = QTensor(Affine(a), Affine(sa), (lk["rows"], spec.hidden), lk["rec"],
                               lk["rec"], lk["D"], wf=Q.mxu_wf("int8"))
        d["embed_mbox"] = mbox = lk["mbox"]
        d["post_token"] = lambda tok: MB.post(mbox, tok)    # the generate loop's request
    else:
        d["embed_q"] = _qdesc(*lk["embed_q"], spec.vocab, spec.hidden, lk["D"])
    if "embed" not in lk:
        M, D = lk["M"], lk["D"]
        d["onehot"] = _tdesc(lk["onehot"], (M, onehot_blocks(D, M, "int8") * D))
    d["gen"] = G.desc(lk["gen"], spec, cap)
    return d


def embed_record(H: int, D: int) -> int:
    """Bytes of one int8 embedding row in a host-filled slot (embed_host): its H int8 values,
    then its H / D scale words (the LM head's format, quant.quantize_mxu), whole MXU chunks."""
    return -(-(H + 4 * (H // D)) // ALIGN) * ALIGN


def embed_store(table, D: int) -> np.ndarray:
    """The int8 embedding table [V, H] the host keeps (embed_host), in the slot's record format
    (embed_record): [V, record bytes] uint8, quantized as the on-card table would be."""
    V, H = table.shape
    out = np.zeros((V, embed_record(H, D)), np.uint8)
    for r0 in range(0, V, Q.QUANT_ROWS):
        q, sc = Q.quantize_mxu(np.asarray(table[r0:r0 + Q.QUANT_ROWS], np.float32), "int8", D)
        n = len(q)
        out[r0:r0 + n, :H] = np.asarray(q).view(np.uint8).reshape(n, H)
        out[r0:r0 + n, H:H + 4 * (H // D)] = np.asarray(sc).view(np.uint8).reshape(n, -1)
    return out


class EmbedHost:
    """An Image's side of embed_host (its int8 embedding table on the host, the card holding a
    slot of the run's rows: _lookup_alloc): the rows the host writes before a run, and the
    server of the generate loop's requests (each sampled token's row into slot row 0). Data
    movement only, as Gemma 4 E4B's PLE records (docs/offload.md 5.9)."""

    @property
    def embed_host(self) -> bool:
        return bool(getattr(self, "lookup", None) and self.lookup.get("host"))

    def host_rows(self, tokens) -> list:
        """[(slot address, the tokens' records)] before a run of these token rows ([] with the
        table on the card)."""
        if not self.embed_host:
            return []
        lk = self.lookup
        if len(tokens) > lk["rows"]:
            raise ValueError(f"{len(tokens)} tokens, the embedding slot holds {lk['rows']}")
        return [(lk["embed_q"][0], lk["store"][[int(t) for t in tokens]].reshape(-1))]

    def row_server(self, backend) -> RowServer | None:
        """The host's server of the generate loop's requests, on the backend's DRAM (the
        Engine gives it the expert server's memory when there is one); None with the table on
        the card."""
        if not self.embed_host:
            return None
        lk = self.lookup
        return RowServer(BackendDram(backend), RowLayout(lk["mbox"], lk["embed_q"][0],
                                                         lk["rec"]),
                         lambda t: lk["store"][t])


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


def has_embed_host(spec) -> bool:
    """The model's image can keep its int8 embedding table on the host (embed_host)."""
    import inspect
    return "embed_host" in inspect.signature(spec.image).parameters


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
    bs = [kernel.trace(image.cfg, s, {"m": step_descriptors(image, s), "pos": rp,
                                      "block": block})
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


KINDS = ("attn", "mlp", "gateup", "down", "head")     # the weight kinds of a formats string


def _formats(spec, formats: str | None) -> str:
    """An image's formats string: `formats`, else OTPU_FORMATS, else spec.formats."""
    import os
    return formats if formats is not None else os.environ.get("OTPU_FORMATS", spec.formats)


def weight_kind(n: str) -> tuple:
    """(kind, layer) of checkpoint weight `n` (KINDS: the head, or a layer's attention, MLP
    gate / up or down projection)."""
    if not n.startswith("model.layers."):
        return "head", 0
    return ("attn" if ".self_attn." in n else "down" if ".down_proj." in n else "gateup",
            int(n.split(".")[2]))


def emulated_logits(spec: Spec, W: dict, tokens, D: int = 128, wformat: str = "int8",
                    head_format: str | None = None, formats: str | None = None) -> np.ndarray:
    """float64 decode that applies openTPU's quantization points but none of its rounding:
    int8 (or 4-bit: `wformat`, `head_format`, `formats` as in Image) weights, int8 matmul
    inputs per D-block, int8 K (per token, D-block) and V (per token), int8 P (per D tokens).
    Separates quantization error from kernel bugs."""
    d, G = spec.head_dim, spec.n_q // spec.n_kv
    Wq: dict = {}
    head = "model.embed_tokens.weight" if spec.tied else "lm_head.weight"
    wformat, formats = FM.named(spec, wformat, formats)
    fmt = FM.resolver(formats, KINDS, spec.formats, wformat, head_format)

    def w(n):
        if n not in Wq:
            Wq[n] = _fake_w(W[n], D, fmt(*weight_kind(n)))
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


class Image(EmbedHost):
    """Per-slice DRAM layout of a Qwen3 model. Every slice uses the same addresses.

    [ I/O: x_in, cos, sin | final norm | logits ] [ layer 0 block ] ... [ layer L-1 block ]
    [ LM head rows of this slice ]. A layer block holds the norms, this slice's rows of every
    projection (quantized + scales) and this slice's KV heads with room for `cap` tokens, for
    each of `batch` sequences; its layout follows the layer's weight formats. The layers run
    as `runs` (first layer, (layout,), layers): each range of one layout a hardware loop over
    its blocks, contiguous with their stride. The I/O area holds `rows` token rows (x, cos,
    sin, logits).
    A model with layers without RoPE (spec.nope) has a rope gate per layer block: (1, 0) or
    (0, 1), and each layer rotates with cos * g0 + g1 and sin * g0 (_rope_gate).

    Weight formats (opentpu/quant.py): `wformat` for the layers' projections, `head_format`
    (default: the same) for the LM head: "int8", or 4-bit "int4" / "fp4". `formats` sets a
    kind's format over wformat per layer range (opentpu/llm/formats.py, KINDS; None:
    OTPU_FORMATS, else spec.formats): a layer block layout per combination, a run (a loop)
    per range of one layout. The KV cache and the activations stay int8.
    """

    def __init__(self, spec: Spec, cfg: Config, cap: int, batch: int = 1, rows: int = 1,
                 wformat: str = "int8", head_format: str | None = None, lookup: bool = False,
                 embed_host: bool | None = None, formats: str | None = None):
        spec.check(cfg)
        if cap % cfg.D:
            raise ValueError("KV capacity must be a multiple of D")
        if spec.ctx and cap > spec.ctx:
            raise ValueError(f"KV capacity {cap} above the model's RoPE range ({spec.ctx})")
        S, D = cfg.S, cfg.D
        H, d, F_ = spec.hidden, spec.head_dim, spec.ffn
        self.spec, self.cfg, self.cap = spec, cfg, cap
        wformat, formats = FM.named(spec, wformat, formats)
        fmt = FM.resolver(formats, KINDS, spec.formats, wformat, head_format)
        # each layer's formats (attention, gate / up, down: its block's layout), the head's,
        # and the formats string (for the compile worker)
        self.lf = tuple((fmt("attn", i), fmt("gateup", i), fmt("down", i))
                        for i in range(spec.layers))
        self.wformat, self.head_format = wformat, fmt("head")
        self.mf = self.mats_formats(self.lf[0])     # layer 0's projections' formats
        self.formats = _formats(spec, formats)
        rb = lambda k, f: Q.row_bytes(k, f, D)                          # noqa: E731
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
        self.mats = {"wq": (self.nq_loc * d, H), "wk": (self.nkv_loc * d, H),
                     "wv": (self.nkv_loc * d, H), "wo": (self.h_loc, spec.n_q * d),
                     "wg": (self.f_loc, H), "wu": (self.f_loc, H)}
        # W_down in column parts of the MLP's F chunk: each down MM streams one part, whose
        # scales are then contiguous (with row-major scales every row would cost a DRAM beat);
        # the chunk by W_down's format
        self.dchunks = {fd: _chunk(self.f_loc, D, D if fd == "int8" else 2 * D)
                        for fd in sorted({f[2] for f in self.lf})}
        self.dchunk = self.dchunks[self.lf[0][2]]
        self.bofs, self.bsize = {}, {}                  # per layout: offsets inside its block
        for key in dict.fromkeys(self.lf):
            mf, C = self.mats_formats(key), self.dchunks[key[2]]
            lb = _Bump()
            L = {"g_in": lb.alloc(4 * H), "g_post": lb.alloc(4 * H)}
            if spec.qk_norm:
                L.update(qn=lb.alloc(4 * d), kn=lb.alloc(4 * d))
            if spec.nope:
                L["rg"] = lb.alloc(8)
            for name, (n, k) in self.mats.items():
                L[name] = (lb.alloc(n * rb(k, mf[name])), lb.alloc(4 * n * (k // D)))
            L["wd"] = [(lb.alloc(self.h_loc * rb(C, mf["wd"])), lb.alloc(4 * self.h_loc * (C // D)))
                       for _ in range(F_ // C)]
            L["kvs"] = [[{"k": lb.alloc(cap * d), "ks": lb.alloc(4 * cap * (d // D)),
                          "vt": lb.alloc(d * cap), "vs": lb.alloc(4 * cap)}
                         for _ in range(self.nkv_loc)] for _ in range(batch)]
            L["kv"] = L["kvs"][0]
            self.bofs[key], self.bsize[key] = L, (lb.next + 4095) // 4096 * 4096
        self.lofs, self.LS = self.bofs[self.lf[0]], self.bsize[self.lf[0]]   # layer 0's
        # the runs: each range of layers of one layout a hardware loop (one layer body in the
        # program per run: a pattern repeated in units, lfm2.plan's, would unroll each unit)
        self.runs = []
        for i, key in enumerate(self.lf):
            if self.runs and self.runs[-1][1] == (key,):
                first, unit, reps = self.runs[-1]
                self.runs[-1] = (first, unit, reps + 1)
            else:
                self.runs.append((i, (key,), 1))
        self.loc = {}                   # layer -> (run base, unit stride, iteration, offset)
        b.next = self.layer0
        for first, unit, reps in self.runs:
            offs = np.cumsum([0] + [self.bsize[k] for k in unit]).tolist()
            us, base = offs[-1], b.next
            for it in range(reps):
                for e in range(len(unit)):
                    self.loc[first + it * len(unit) + e] = (base, us, it, offs[e])
            b.next = base + reps * us
        head = cap * d + 4 * cap * (d // D) + d * cap + 4 * cap   # k, k scales, v^T, v scales
        self.kv_bytes = spec.layers * self.nkv_loc * head         # per sequence
        self.head = (b.alloc(self.v_loc * Q.row_bytes(H, self.head_format, D)),
                     b.alloc(4 * self.v_loc * (H // D)))
        # the int8 embedding rows of the resident decode are the tied int8 head's (S = 1)
        shared = spec.tied and self.head_format == "int8" and S == 1
        if embed_host is None:      # dense: the int8 table stays on the card (the Engine moves
            embed_host = False      # it to the host when the image does not fit otherwise)
        if embed_host and not lookup:
            raise ValueError("embed_host needs the image's lookup tables (lookup=True)")
        self.lookup = _lookup_alloc(b, spec, cap, D=D, head=self.head if shared else None,
                                    M=cfg.MCOLS, embed_host=bool(embed_host),
                                    rows=rows) if lookup else {}
        self.choices = {"embed_host": self.embed_host,          # (the compile worker's
                        "formats": self.formats}                # image)
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

        def put_q(addr_pair, parts, fmt):
            for s, p in enumerate(parts):
                q, sc = QC.quantize_mxu(p, fmt, D)
                put(s, addr_pair[0], q)
                put(s, addr_pair[1], sc)

        def rows(a, n):
            return [a[s * n:(s + 1) * n] for s in range(S)]

        def f32(a):
            return F.ftz(np.asarray(a, np.float32))

        for s in range(S):
            put(s, self.io["gf"], f32(W["model.norm.weight"]))
        for i in range(spec.layers):
            p, base, lofs = f"model.layers.{i}.", self._off(i).const, self.bofs[self.lf[i]]
            mf = self.mats_formats(self.lf[i])
            Lo = {k: (tuple(base + x for x in v) if isinstance(v, tuple) else
                      (base + v if isinstance(v, int) else v)) for k, v in lofs.items()}
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
            put_q(Lo["wq"], rows(wq, self.nq_loc * d), mf["wq"])
            put_q(Lo["wk"], rows(wk, self.nkv_loc * d), mf["wk"])
            put_q(Lo["wv"], rows(wv, self.nkv_loc * d), mf["wv"])
            put_q(Lo["wo"], rows(wo, self.h_loc), mf["wo"])
            put_q(Lo["wg"], rows(W[p + "mlp.gate_proj.weight"], self.f_loc), mf["wg"])
            put_q(Lo["wu"], rows(W[p + "mlp.up_proj.weight"], self.f_loc), mf["wu"])
            C = self.dchunks[mf["wd"]]
            for j, pair in enumerate(lofs["wd"]):
                put_q((base + pair[0], base + pair[1]),
                      [r[:, j * C:(j + 1) * C] for r in rows(W[p + "mlp.down_proj.weight"],
                                                              self.h_loc)], mf["wd"])
        head = W["model.embed_tokens.weight"] if spec.tied else W["lm_head.weight"]
        put_q(self.head, rows(head, self.v_loc), self.head_format)
        if self.lookup:
            _lookup_build(put, S, W, spec, self.cap, self.lookup)
        return imgs

    @staticmethod
    def mats_formats(key) -> dict:
        """The projections' formats of a layer block of formats `key` (attention, gate / up,
        down)."""
        fa, fg, fd = key
        return {"wq": fa, "wk": fa, "wv": fa, "wo": fa, "wg": fg, "wu": fg, "wd": fd}

    def _off(self, li, it=None) -> Affine:
        """The block address of layer li (static), or of element li of its run's unit at
        iteration `it` (a loop variable)."""
        base, us, i, o = self.loc[li]
        return Affine(base + o) + Affine.of(i if it is None else it) * us

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
        return [qwen3_step.trace(self.cfg, s, {"m": step_descriptors(self, s), "pos": pos,
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

    def compile_prompt_run(self, blocks: int, R: int, kind: str, block: int = ATTN_BLOCK):
        """qwen3_prompt_run's (programs, run_args) (docs/prefill.md), a program per slice: R
        rows of a prompt at a run-time position of bucket `blocks` (the position in the state's
        tpos word; run_args name the words, RunWords)."""
        if not self.lookup:
            raise ValueError("a prompt run needs lookup tables (lookup=True)")
        if kind not in ("P", "L"):
            raise ValueError(f"prompt run kind {kind!r}")
        pos = RunRows(blocks, block, (blocks - 1) * block, self.lookup["zmask"], self.cap, R, 0,
                      _amask(self.lookup))
        bs = [qwen3_prompt_run.trace(self.cfg, s, {"m": self.descriptors(s), "pos": pos, "R": R,
                                                   "kind": kind, "block": block})
              for s in range(self.cfg.S)]
        return [b.finish() for b in bs], list(bs[0].run_args)

    # ---- kernel descriptors
    def descriptors(self, sid: int) -> SimpleNamespace:
        spec, cfg = self.spec, self.cfg
        D, d, H = cfg.D, spec.head_dim, spec.hidden
        rh = spec.rope_dim // 2

        def layer(li, it=None):
            """Descriptors of layer `li` (static), or of element li of its run's unit at
            iteration `it` (a hardware-loop variable)."""
            off = self._off(li, it)
            lofs, mf = self.bofs[self.lf[li]], self.mats_formats(self.lf[li])
            ns = SimpleNamespace(
                g_in=Tensor(off + lofs["g_in"], (H,), (1,)),
                g_post=Tensor(off + lofs["g_post"], (H,), (1,)),
                qn=Tensor(off + lofs["qn"], (d,), (1,)) if spec.qk_norm else None,
                kn=Tensor(off + lofs["kn"], (d,), (1,)) if spec.qk_norm else None,
                rg=Tensor(off + lofs["rg"], (2,), (1,)) if spec.nope else None)
            for name, (n, k) in self.mats.items():
                da, sa = lofs[name]
                fm = mf[name]
                setattr(ns, name, QTensor(off + da, off + sa, (n, k), Q.row_bytes(k, fm, D),
                                          4 * (k // D), D, wf=Q.mxu_wf(fm)))
            fm = mf["wd"]
            C, n = self.dchunks[fm], self.h_loc
            wf = Q.mxu_wf(fm)
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
            spec=spec, layer=layer, runs=self.runs,
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

    def offset(self, r: int) -> "RunPos":
        """Row r of a run of rows from this position, in its attention block (a layer-major
        prefill run, Engine.prefill_layers): position p + r, its mask row r entries on (the
        same run-time values)."""
        q = object.__new__(type(self))
        q.__dict__.update(self.__dict__)
        q.tpos, q.pos = self.tpos + r, self.pos + r
        q.bucket = Bucket(self.blocks, self.bucket.z - 4 * r)
        return q


class RunRows(RunPos):
    """R consecutive rows at a run-time position (MTP's verify and draft runs, docs/mtp.md 10):
    row r is position t0 + tpos + r, every row in the bucket (tpos <= block - R), its token the
    run-time value toks[r] (tok, tok1, ...); row r attends over the bucket with the mask row
    of its own position (bucket_row). values(tokens, p) gives the run's argument values.
    toks_at k (a prefill run, docs/prefill.md): the tokens from the generate area's out[]
    instead, row r's at out[p + k + r], loaded once by the run: no argument per token. amask
    (a prompt run's: the image's _amask): row r's masked block takes the additive tile of its
    position in the block (attention.Additive, docs/prefill.md 9) instead of a mask row."""

    def __init__(self, blocks: int, block: int, lo: int, zmask: int, cap: int, R: int,
                 toks_at: int | None = None, amask: tuple | None = None):
        super().__init__(blocks, block, lo, zmask, cap)
        if not 0 < R <= min(block, cap - self.t0):
            raise ValueError(f"{R} rows do not fit bucket {blocks}")
        self.R, self.toks_at, self.amask = R, toks_at, amask
        self.tpos.bound = min(block, cap - self.t0) - R + 1
        self.toks = [] if toks_at is not None else \
            [self.tok] + [RunVar(f"tok{r}") for r in range(1, R)]

    def bucket_row(self, r: int) -> Bucket:
        if self.amask is None:
            return Bucket(self.blocks, self.bucket.z - 4 * r)
        base, step = self.amask
        return Bucket(self.blocks, self.bucket.z - 4 * r,
                      Additive(Affine(base + step * r) + self.tpos * step))

    @staticmethod
    def values(tokens, p: int, K: int = 1, block: int = ATTN_BLOCK) -> dict:
        v = RunPos.values(tokens[0], p, K, block)
        v.update({f"tok{r}": t for r, t in enumerate(tokens) if r})
        return v


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
AHEAD_PART = 1 << 20  # layer_ahead "hint" and expert_major: an idle poll's part of a queued expert
                      # (docs/offload.md 13.9: 1 MiB parts against 13.8's 512 KiB, the per-part
                      # cost halved)


def fill_logits(m) -> None:
    """A step program's first instructions when compiled with fill (m.fill: the engine's
    streamed logits, BoardBackend.start(stream=...)): FILL_SENTINEL (-inf: VOP FILL keeps it;
    the LM head's logits are finite) over this slice's logits in HEAD_CHUNK stores, so that
    the host tells the pieces the LM head has written from the last token's. Then one word
    loaded back from the last store (it waits for that ST, which completes only once the DRAM
    has taken every write so far: the host's reads see the fill) into an RLD, which holds
    every later instruction until the word is back: once the card's ICOUNT has counted the
    instruction after the RLD, the fill has landed (fill_gate). Static addresses and no loop,
    so ICOUNT counts each instruction before it once; the rest of the program is the one
    compiled without the fill (TMEM's next-fit cursor put back). Nothing in the generate loop's
    programs (lm_sink / lm_split)."""
    if not getattr(m, "fill", False) or getattr(m, "lm_sink", None) is not None or \
            getattr(m, "lm_split", None) is not None:
        return
    b, sid = current(), ol.program_id()
    cursor = b.tmem_next
    n = min(HEAD_CHUNK, ol.tmem_words() // 8, m.v_loc)
    t = ol.full([n], -math.inf)                     # FILL_SENTINEL
    for c0 in range(0, m.v_loc, n):
        k = min(n, m.v_loc - c0)
        col = sid * m.v_loc + c0
        ol.store(m.logits[0, col:col + k], t[:k])
    last = sid * m.v_loc + m.v_loc - 1
    b.rld(0, ol.load(m.logits[0, last:last + 1], out=t[0:1]), comment="logits fill landed")
    del t
    b.tmem_next = cursor        # the rest of the program as compiled without the fill


def fill_gate(prog) -> int:
    """The ICOUNT past fill_logits' RLD in `prog` (a slice's instructions, or their words): the
    instructions up to the first RLD, plus the one after it. (Instructions read back from the
    program cache have no comments: there the first RLD must be into r0, as the fill's.)"""
    if isinstance(prog, np.ndarray):
        ops = np.asarray(prog, np.uint32).reshape(-1, 8)[:, 0] & 0xFF
        at = np.flatnonzero(ops[:512] == I.RLD)         # near the start: no whole scan
        if not len(at):
            at = np.flatnonzero(ops == I.RLD)
        if not len(at):
            raise ValueError("no RLD: not a program compiled with fill")
        return int(at[0]) + 2
    for i, ins in enumerate(prog):
        if ins.op == I.RLD:
            if ins.comment != "logits fill landed" and (ins.comment or ins.rd != 0):
                raise ValueError("the program's first RLD is not fill_logits'")
            return i + 2
    raise ValueError("no RLD: not a program compiled with fill")


def step_descriptors(image, sid: int) -> SimpleNamespace:
    """image.descriptors(sid) for a decode step's program (compile_step, compile_decode): with
    fill_logits when the image's engine streams the logits (image.stream_fill)."""
    m = image.descriptors(sid)
    m.fill = bool(getattr(image, "stream_fill", False))
    return m


@ol.jit
def qwen3_step(m, pos: int, block: int = ATTN_BLOCK, tok: int | None = None):
    """One decode token at position `pos`: x (the token's embedding) -> logits.

    The layers run as a hardware loop; each layer appends its K/V at `pos` and attends over
    positions 0..pos. Logits for this slice's vocabulary rows are stored to m.logits. `tok`:
    the token's id, its inputs read from the image's tables (_inputs).
    """
    fill_logits(m)
    spec = m.spec
    x, c, s_ = _inputs(m, pos, tok)
    for lw in _layers(m):
        x.set(_attention(x, lw, c, s_, pos, spec, block))
        x.set(_mlp(x, lw, spec))
    _lm_head(x, m, spec)


def _layers(m):
    """Each layer's descriptors in the order the layers run (m.runs, Image): a run of repeats
    is one hardware loop over its unit (inside the loop the descriptors are the iteration's)."""
    for first, unit, reps in m.runs:
        if reps == 1:
            for e in range(len(unit)):
                yield m.layer(first + e)
            continue
        for it in ol.range(reps):
            for e in range(len(unit)):
                yield m.layer(first + e, it)


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
    the fp32 table's row, or the int8 row gathered on the device (Spec.embed "int8"); with the
    table on the host (embed_host) the slot's row 0, once the host has served every request
    (the generate loop's post of the token; before other runs the host writes the row)."""
    eq = getattr(m, "embed_q", None)
    if eq is None:
        return ol.load(m.embed[tok:tok + 1, :])
    mbox = getattr(m, "embed_mbox", None)
    if mbox is not None:
        MB.wait_served(mbox)
        tok = 0
    return next(_gather(m, eq, [tok]))


class OutTokens:
    """Rows' tokens from the generate area's out[] (a prefill run, docs/prefill.md): row r's
    at out[p + k + r], p the run's first position; loaded by the run, so the program does not
    depend on them (_inputs_rows's `tokens`)."""

    def __init__(self, k: int = 0):
        self.k = k


S_RINGO = 32        # prompt runs: LFM2's ring rotation after the run (lfm2._ring_store)
RUN_WORDS = {"tpos": G.S_TPOS, "ring": G.S_RING, "ringo": S_RINGO}


class RunWords:
    """A prompt run's position (docs/prefill.md): with a RunRows, the run-time arguments come
    from the generate state's words (RUN_WORDS: tpos, LFM2's ring; the compiler's run_words,
    no host arguments), loaded at the program's start once the kernel has named them, and
    pos.words gives the kernel their TMEM addresses (LFM2's ringo); with a compile-time
    position, nothing. `with RunWords(m, pos): kernel(...)`; the host writes the words before
    each run (prefill.run: those the run_args name and the image's prompt_words)."""

    def __init__(self, m, pos):
        self.m, self.pos, self.run = m, pos, isinstance(pos, RunRows)

    def __enter__(self):
        if self.run:
            b = current()
            n = max(RUN_WORDS.values()) + 1
            self.st = ol.load(self.m.gen.state[0:n])
            self.at = b.stack[-1][-1]
            b.run_words = {k: self.st.base + w for k, w in RUN_WORDS.items()}
            self.pos.words = dict(b.run_words)          # (TMEM addresses for the kernel's RLDs)
        return self

    def __exit__(self, *exc):
        if self.run and exc[0] is None:
            b = current()
            body = b.stack[-1]
            i = next(j for j, x in enumerate(body) if x is self.at) + 1
            body[i:i] = [I.rld(15 - k, b.run_words[v.name], mul=int(c),
                               comment=f"argument {c}*{v.name}")
                         for k, (v, c) in enumerate(b.run_args)]
            b.run_words = None
        return False


def _embed_word(m, t):
    """_embed of the token id in the one-word tile t (a prefill run's, docs/prefill.md): its
    row's address from a scratch register (RLD MUL into a compiler.DevVar) instead of a
    run-time argument; the int8 table's row and its scales from two."""
    b = current()
    b.check_live(t)
    if getattr(m, "embed_mbox", None) is not None:
        raise CompileError("a token from DRAM needs the embedding table on the card")
    eq = getattr(m, "embed_q", None)
    if eq is None:
        e = m.embed
        r = b.scratch()
        b.emit(I.rld(r, t.base, mul=4 * e.strides[0], comment="the token's row"))
        row = ol.load(Tensor(Affine.of(e.base) + DevVar("token row", r), (1, e.shape[1]),
                             e.strides))
        b.unscratch(r)
        return row
    rd, rs = b.scratch(), b.scratch()
    b.emit(I.rld(rd, t.base, mul=eq.rs, comment="the token's row"))
    b.emit(I.rld(rs, t.base, mul=eq.srs, comment="its scales"))
    one = QTensor(Affine.of(eq.data) + DevVar("token row", rd),
                  Affine.of(eq.scale) + DevVar("token scales", rs), (1, eq.shape[1]), eq.rs,
                  eq.srs, eq.D, wf=eq.wf)
    row = next(_gather(m, one, [0]))
    b.unscratch(rd)
    b.unscratch(rs)
    return row


def _inputs_rows(m, rows, tokens=None):
    """_inputs for token rows (rows[r] = (sequence, position)): from the I/O area, or with
    `tokens` (their ids, compile-time values) from the image's tables, the embedding row of
    each token and the RoPE rows of each run of consecutive positions. rows a RunRows: its
    tokens' rows (run-time ids) and its positions' RoPE rows from the tables."""
    if isinstance(rows, RunRows):
        R = rows.R
        x = ol.empty([R, m.xr.shape[1]], dense=True)
        c = ol.empty([R, m.cosr.shape[1]], dense=True)
        s_ = ol.empty([R, m.sinr.shape[1]], dense=True)
        ol.load(m.cos_t[rows.pos:rows.pos + R, :], out=c)
        ol.load(m.sin_t[rows.pos:rows.pos + R, :], out=s_)
        if rows.toks_at is not None:        # the tokens from out[] (docs/prefill.md)
            k = rows.toks_at
            tk = ol.load(m.gen.out[rows.pos + k:rows.pos + k + R])
            for r in range(R):
                x[r:r + 1, :].set(_embed_word(m, tk[r:r + 1]))
            del tk
        for r, t in enumerate(rows.toks):
            x[r:r + 1, :].set(_embed(m, t))
            ol.release(t)
        return x, c, s_
    R = len(rows)
    if tokens is None:
        return ol.load(m.xr[0:R, :]), ol.load(m.cosr[0:R, :]), ol.load(m.sinr[0:R, :])
    if isinstance(tokens, OutTokens):
        x = ol.empty([R, m.xr.shape[1]], dense=True)
        c = ol.empty([R, m.cosr.shape[1]], dense=True)
        s_ = ol.empty([R, m.sinr.shape[1]], dense=True)
        p0 = rows[0][1] + tokens.k
        tk = ol.load(m.gen.out[p0:p0 + R])
        for r in range(R):
            x[r:r + 1, :].set(_embed_word(m, tk[r:r + 1]))
        del tk
        for _, q0, r0, n in _runs(rows):
            ol.load(m.cos_t[q0:q0 + n, :], out=c[r0:r0 + n, :])
            ol.load(m.sin_t[q0:q0 + n, :], out=s_[r0:r0 + n, :])
        return x, c, s_
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
    if eq is not None:              # embed_host: the slot's rows (the host writes them first)
        idx = range(len(tokens)) if getattr(m, "embed_mbox", None) is not None else tokens
        for r, g in enumerate(_gather(m, eq, idx)):
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
    rows[r][0] (its own KV cache), or rows is a RunRows (sequence 0, run-time positions; the
    V^T appends a row at a time, each row's mask its own). Every row's K/V is appended first,
    then each row attends
    over positions 0..pos of its sequence -- for consecutive rows of one sequence (a prefill
    chunk) that is exactly the causal mask. Each projection streams its weights once for all
    R rows (ceil(R / MCOLS) MMs); the (row, KV head) pairs then run as one pipelined
    flash-attention stream (_attend_heads). Per row the arithmetic is _attention's, so the
    results are bit-identical to R decode steps: heads narrower than D (LFM2) are padded, RoPE
    may cover part of a head (Qwen3.5), `gated` multiplies the output by sigmoid(W_gate x),
    and a query group wider than the MXU attends in parts of MCOLS heads. A row's position may
    be a run-time one (RunPos.offset: a layer-major prefill run), its K / V appended row by row."""
    d, G, eps = spec.head_dim, spec.n_q // spec.n_kv, spec.eps
    R = rows.R if isinstance(rows, RunRows) else len(rows)
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
    run = isinstance(rows, RunRows)
    for j, hh in enumerate(heads):
        kj = _rope_rows_padded(_norm_heads(k[:, j * d:(j + 1) * d], kn, eps), c, s_)
        vj = _padded(v[:, j * d:(j + 1) * d])
        if run:                                 # V^T at a run-time position: a row at a time
            for r in range(R):
                ol.kv_append(lw.kvs[0], hh, rows.pos + r, kj[r:r + 1, :], vj[r:r + 1, :])
        elif isinstance(rows[0][1], RunPos):    # (rows of their own RunPos, likewise)
            for r, (sq, p) in enumerate(rows):
                ol.kv_append(lw.kvs[sq], hh, p.pos, kj[r:r + 1, :], vj[r:r + 1, :])
        else:
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

    def seq(r):                                 # row r's extent: its bucket, or pos + 1
        if run:
            return rows.bucket_row(r)
        p = rows[r][1]
        return p.bucket if isinstance(p, RunPos) else p + 1

    _attend_heads([Q[r * nq + j * G + g0:r * nq + j * G + g1, :] for r, j, g0, g1 in ent],
                  [lw.kvs[0 if run else rows[r][0]] for r, *_ in ent],
                  [heads[j] for _, j, _, _ in ent], [seq(r) for r, *_ in ent], block,
                  scale, depth=ATTN_DEPTH, emit=emit)
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
    for lw in _layers(m):
        x.set(_attention_rows(x, lw, c, s_, rows, spec, block))
        x.set(_mlp(x, lw, spec))
    _lm_head_rows(x, m, spec, logit_rows)


@ol.jit
def qwen3_prompt_run(m, pos, R: int, kind: str, block: int = ATTN_BLOCK):
    """A prompt run of R rows (docs/prefill.md), qwen3_rows with its tokens from out[]: kind
    "P", or "L" (the prompt's last run: its last row's logits). pos: a RunRows (toks_at 0) at
    the run-time position in the generate state's tpos word (RunWords)."""
    with RunWords(m, pos):
        qwen3_rows.fn(m, pos, [R - 1] if kind == "L" else [], block)


def head_rows_chunk(rows: int) -> int:
    """The LM head's vocabulary rows per MM for `rows` token rows (their fp32 logits fit TMEM)."""
    return min(HEAD_CHUNK, ol.tmem_words() // (8 * rows))


def _lm_head_rows(x, m, spec, logit_rows):
    """Final norm and this slice's vocabulary rows of the LM head for the rows `logit_rows`
    of x (a contiguous range, or empty: nothing) -> m.logitsr, or, with m.lm_sinks (one sink
    per logit row, generate.Greedy over head_rows_chunk's chunks: MTP's verify run), each
    chunk's row r to sink r instead."""
    if not logit_rows:
        return
    sid = ol.program_id()
    a, e = logit_rows[0], logit_rows[-1] + 1
    xs = ol.quantize(rmsnorm(x[a:e, :], ol.load(m.g_final), spec.eps))
    chunk = head_rows_chunk(e - a)
    sinks = getattr(m, "lm_sinks", None)
    for c0 in range(0, m.v_loc, chunk):
        n = min(chunk, m.v_loc - c0)
        col = sid * m.v_loc + c0
        if sinks is None:
            ol.store(m.logitsr[a:e, col:col + n], ol.dot(xs, m.head[c0:c0 + n, :]))
            continue
        y = ol.dot(xs, m.head[c0:c0 + n, :])
        for r, sk in enumerate(sinks):
            sk(y[r:r + 1, :], col)
        del y


# =============================================================================== engine
def device_config(spec: Spec, cap: int, batch: int = 1, rows: int = 1, wformat: str = "int8",
                  head_format: str | None = None, lookup: bool = False,
                  experts: int | None = None, embed_host: bool | None = None, **kw) -> Config:
    """The design configuration with DRAM sized for this model (power of two MiB)."""
    probe = spec.image(design_config(DRAM_BYTES=1 << 40, **kw), cap, batch, rows, wformat,
                       head_format, **({"lookup": True} if lookup and has_lookup(spec) else {}),
                       **({"experts": experts} if experts is not None else {}),
                       **({"embed_host": embed_host} if embed_host is not None else {}))
    size = 1 << max(20, (probe.nbytes - 1).bit_length())
    return design_config(DRAM_BYTES=size, **kw)


class IsaBackend:
    """The bit-exact ISA simulator, one persistent machine (DRAM keeps the KV cache).
    adopt: the images (arrays of their own) become the machine's DRAM, grown to DRAM_BYTES in
    place instead of copied, so a 4 GiB image is held once; the caller gives them up."""

    def __init__(self, cfg: Config, images: list, adopt: bool = False):
        own = adopt and all(isinstance(m, np.ndarray) and m.dtype == np.uint8 and m.ndim == 1
                            and m.flags.owndata and len(m) <= cfg.DRAM_BYTES for m in images)
        self.machine = Machine(cfg, [[] for _ in range(cfg.S)],
                               [None] * cfg.S if own else images)
        if own:
            for s, m in zip(self.machine.slices, images):
                m.resize(cfg.DRAM_BYTES, refcheck=False)
                s.dram = m

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


def _worker_init(spec, cfg, cap, batch, rows, block, image_kw: dict,
                 stream_fill: bool = False) -> None:
    """The worker's image: the engine's layout (spec.image with the engine's keywords: weight
    formats, the resident decode's lookup tables, a MoE's expert slots; its programs must
    address the same image), and its step programs as the engine's (stream_fill)."""
    global _WORKER
    image = spec.image(cfg, cap, batch, rows, **image_kw)
    image.stream_fill = stream_fill
    _WORKER = (image, block)
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


def layer_programs(image, key, block: int, hint: bool = False, em: bool = False):
    """Engine.prefill_layers' run of `key`: (layer (-1: the embed run), blocks, rows,
    embedded), (layer, None, rows, True, compile-time position), ("x", layer) (an expert run)
    or "head" -> (programs, run_args); hint: a layer's run ends with the next MoE layer's hint
    (Engine's layer_ahead "hint"); em: expert-major (Engine's expert_major)."""
    if key == "head":
        return image.compile_prefill_head(**({"em": True} if em else {}))
    if key[0] == "x":
        return image.compile_expert_run(key[1])
    kw = {"hint": True} if hint and key[0] >= 0 else {}
    if em:
        kw["em"] = True
    if key[1] is None:                      # (a compile-time position)
        return image.compile_layer_run(key[0], key[4] // block + 1, block, R=key[2],
                                       embedded=True, at=key[4], **kw)
    return image.compile_layer_run(key[0], key[1], block, R=key[2], embedded=key[3], **kw)


def _worker_layer(key, hint: bool = False, em: bool = False):
    """The worker process: layer_programs' program, assembled (one slice), and its
    run_args."""
    from ..isa import assemble
    image, block = _WORKER
    progs, ra = layer_programs(image, key, block, hint, em)
    return np.asarray(assemble(progs[0]), np.uint32), ra


def _worker_chunk(seq: int, p0: int, n: int, left: int, fit: int, toks=None,
                  whole: bool = True):
    """The worker process: fit_chunk's run, its program assembled (one slice)."""
    from ..isa import assemble
    image, block = _WORKER
    n, progs, fit = fit_chunk(image, block, seq, p0, n, left, fit, toks, whole)
    return n, None if progs is None else np.asarray(assemble(progs[0]), np.uint32), fit


def fit_chunk(image, block: int, seq: int, p0: int, n: int, left: int, fit: int,
              tokens=None, whole: bool = True, **kw):
    """The next prefill run of sequence `seq` at position p0: up to n of the `left` remaining
    prompt tokens, as many as fit TMEM and ACT RAM (at most `fit` rows) and the instruction
    memory (attention is unrolled per row, head and block: the program grows with the context).
    Only the prompt's last run computes logits (its last row). Returns (R, compile_rows'
    programs or None for R = 1, the rows that fit TMEM as far as known). `tokens` (at least n):
    the run's inputs come from the image's tables (compile_rows tokens). whole: R is at most
    MCOLS or a multiple of it (_whole_passes): each weight streams once per MCOLS rows, so 5
    rows at MCOLS 4 cost what 8 do; and an odd R takes R - 1 rows where they cost less MXU
    time per row (prefer_rows: PAIR). kw: compile_rows' own (MTP decoding's hidden, slot)."""
    imem, mc = image.cfg.IMEM_WORDS, image.cfg.MCOLS if whole else 1 << 30

    def rows(r, logits):
        return image.compile_rows([(seq, p0 + j) for j in range(r)], [r - 1] if logits else [],
                                  block, **({} if tokens is None else {"tokens": tokens[:r]}),
                                  **kw)
    n = _whole_passes(min(n, fit, left), mc)
    while n > 1:
        try:
            progs = rows(n, n == left)
        except CompileError as e:
            if "TMEM" not in str(e) and "ACT RAM full" not in str(e):
                raise
            fit = n - 1
            n = _whole_passes(fit, mc)
            continue
        size = max(map(len, progs))
        if size * 8 <= imem:
            m = prefer_rows(image, n, lambda r: rows(r, False)) if whole else n
            if m == n:
                return n, progs, fit
            return m, rows(m, False), fit
        n = _whole_passes(min(n - 1, n * imem // (8 * size)), mc)  # ~ proportional to the rows
    return 1, None, fit


def mxu_time(progs) -> float:
    """A run's MXU time, statically, in weight blocks: each MM's streamed rows times its K
    blocks, half of that PAIRed (two 4-bit blocks a cycle), times its loops' counts (a loop
    with a run-time count once); the slowest slice's."""
    best = 0.0
    for prog in progs:
        t, loops = 0.0, []                      # loops: (last instruction, count)
        for pc, ins in enumerate(prog):
            while loops and loops[-1][0] < pc:
                loops.pop()
            if ins.op == I.LOOP:
                loops.append((pc + ins.w[0], ins.w[1] if ins.ra == 0 else 1))
            elif ins.op == I.MM:
                mult = math.prod(c for _, c in loops)
                t += mult * (ins.w[3] & 0xFFFF) * (ins.w[3] >> 16) * \
                    (0.5 if ins.flags & I.F_PAIR else 1.0)
        best = max(best, t)
    return best


# prefer_rows: R - 1 rows take over only at 10% less MXU time a row. mxu_time leaves out what a
# run and a layer pay once (the run's start, the first weight chunks' latency, the instructions
# besides the MMs), which R - 1 rows spread over fewer rows: in ld-memch's RTL co-sim (board
# config, DDR3-1066, 133.33 MHz) Phi-4-mini's fp4-MLP layer is 6.1% cheaper a row at 2 rows by
# mxu_time but 0.5% dearer (2 rows reach 89% of their MXU roofline, 3 rows 95%), and the 4B's
# -25% is -13.8% (the card: -13.3%). The real layouts' R - 1 picks sit at -19 to -25%.
PREFER_MARGIN = 0.10


def prefer_rows(image, n: int, compile) -> int:
    """The rows of a prefill run where n fit: n, or n - 1 for an odd n > 1 whose n - 1 rows
    cost PREFER_MARGIN less MXU time per row (mxu_time of compile(r), a run of r rows without
    logits). With PAIR a 4-bit MM of at most MCOLS / 2 rows streams two blocks a cycle, so 3
    rows at MCOLS 4 pay 4's time and 2 cost less per row (the 4B: -13.8% a row in the co-sim);
    an int8 MM streams once per MCOLS rows, so 3 rows cost less per row than 2 (Phi-4-mini),
    and a mix falls between (docs/prefill.md 7). Kept per image and n, so a prompt run's
    probe (prefill.r_max) and today's runs (fit_chunk) choose alike."""
    if n < 3 or n % 2 == 0:
        return n
    done = image.__dict__.setdefault("_prefer_rows", {})
    if n not in done:
        a, b = mxu_time(compile(n)) / n, mxu_time(compile(n - 1)) / (n - 1)
        done[n] = n - 1 if b <= (1 - PREFER_MARGIN) * a else n
    return done[n]


def _whole_passes(n: int, mc: int) -> int:
    """The rows of a prefill run of at most n: n up to MCOLS (one pass of each weight), else
    whole passes of MCOLS rows."""
    return n if n <= mc else n - n % mc



def _isa_host(poll):
    """The ISA simulator's WAITW hook (called when every slice waits on one): the host polls
    until a waiting slice's WAITW holds or it has nothing left to do, as it keeps polling while
    a card waits. One poll serves a request whole; an idle poll moves one part of an expert,
    so an expert run waiting on a need's entry (docs/offload.md 13.11) takes several."""
    def host(m):
        for _ in range(1 << 20):
            if not poll() or any(s.holds(s.polling) for s in m.slices if s.polling is not None):
                return
    return host

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
    fills (true, with streams: the step programs fill their logits region themselves,
    fill_logits, and the stream carries their ICOUNT gate, fill_gate); args (true: run / start
    take args=words, the run's arguments: resident decode).

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

    release_weights: with the experts streamed from a pool file, the checkpoint's files are
    released once the image is written and the slots warm (W.release(), a LazyWeights': its
    mapped pages would stay in the page cache over the pool's, docs/offload.md 10.6; a later
    read reopens its file). Default on; False keeps them. pool_map: the pool file's reads
    touched through a read-only map (PoolFile's mapped, docs/offload.md 10.7). Default on.

    layer_major: a MoE model's prompt layer by layer, runs of that many rows (prefill_layers,
    docs/offload.md 13); pooled: the expert slots pooled for it, given back by `restore`
    ("lazy" or "eager"); embed_runs: with the embedding rows from the host's table, its embed
    runs and compile-time-position runs (default off while the card's port A keeps its beats
    across runs, 13.6: token steps first, and layer 0's runs gather their rows); layer_ahead:
    with pooled slots and a server that streams ahead (ExpertServer.ahead_layer), the next MoE
    layer's experts sent while a layer runs (13.7): True for each layer's experts in index
    order, or per MoE layer its expert indices in the order to send them (a static profile, most
    used first; a shorter list sends fewer), or "hint": the card's own guess (13.9), each layer
    run ends with the next MoE layer's router on its output rows, posted as a hint the server
    adds to that layer's queue (each queue started empty; idle-poll parts of `ahead_part`
    bytes, AHEAD_PART by default). expert_major: with layer_major and pooled slots, each MoE
    layer of a chunk routes in its runs and computes its experts in one expert run, each chosen
    expert once (docs/offload.md 13.11, 13.13); its server takes the scratch and the need lines
    (ExpertServer.begin_prefill(expert_major=True, scratch=...)), the needs sent in idle-poll
    parts of `ahead_part` bytes.

    prog_cache: the bucket programs (the generate loop's, the resident decode's, MTP's loop's)
    come from opentpu/progcache.py: compiled once per process and image layout, and kept on
    disk for the next process. Default: on when OTPU_PROG_CACHE is set (to a directory, or 1
    for the default one) and not 0. Off, every engine compiles its own (tests patch the
    kernels' constants, which the cache's key does not see).

    prompt_runs (docs/prefill.md): prefill_chunks runs a prompt (sequence 0) from programs at
    run-time positions, its tokens from out[] (opentpu/llm/prefill.py): compiled once per
    bucket and kind, not per prompt. A dense model's resident image (prefill.supported). With
    the pipeline, bucket 1's are loaded or compiled on a thread as the engine starts
    (prefill.warm).
    """

    def __init__(self, spec: Spec, W: dict, cap: int = 4096, cfg: Config | None = None,
                 backend="isa", block: int = ATTN_BLOCK, batch: int = 1,
                 rows: int = PREFILL_ROWS, pipeline: bool | str | None = None,
                 wformat: str = "int8", head_format: str | None = None,
                 resident: bool = False, experts: int | None = None, pool_file=None,
                 embed_host: bool | None = None, layer_major: int = 0, pooled: bool = True,
                 restore: str = "lazy", embed_runs: bool = False, release_weights: bool = True,
                 pool_map: bool = True, prog_cache: bool | None = None,
                 prompt_runs: bool = False, layer_ahead=None, expert_major: bool = False):
        self.spec, self.cap, self.block = spec, cap, block
        self.prog_cache = PC.enabled() if prog_cache is None else bool(prog_cache)
        self.prompt_runs = prompt_runs
        self.batch, self.rows = batch, max(rows, batch)
        wkw = dict(wformat=wformat, head_format=head_format)
        if experts is not None:             # a MoE model's expert slots per layer
            wkw["experts"] = experts
        if embed_host is not None:          # the int8 embedding table on the host (default: the
            wkw["embed_host"] = embed_host  # image's choice, a MoE's untied table there)
        # an int8 embedding is dequantized on the device (the image's tables), never the host
        int8_embed = getattr(spec, "embed", "f32") == "int8"
        lookup = (bool(resident) or int8_embed) and batch == 1 and has_lookup(spec)
        if lookup:
            wkw["lookup"] = True
        self.cfg = cfg or device_config(spec, cap, batch=batch, rows=self.rows, **wkw)
        try:
            self.image = spec.image(self.cfg, cap, batch, self.rows, **wkw)
        except MemoryError as e:            # an int8 embedding table of its own that does not
            if embed_host is not None or not (lookup and int8_embed and has_embed_host(spec)):
                raise                       # fit beside the rest: on the host (docs/offload.md
            try:                            # 5.9)
                self.image = spec.image(self.cfg, cap, batch, self.rows,
                                        **{**wkw, "embed_host": True})
            except (MemoryError, ValueError):
                raise e from None
        # the compile worker's image is built the same way, with the choices this image made
        # (Gemma 4: the PLE table's place and format, the formats by fit)
        self._image_kw = {**wkw, **getattr(self.image, "choices", {})}
        # with the tables every run reads its inputs from the image (the token ids are compiled
        # into the per-position and prefill programs); else the host writes them: the
        # embedding rows and the RoPE rows of a table computed once, here
        self.device_inputs = bool(getattr(self.image, "lookup", None))
        self.embed = Embedding(spec, W, self.cfg.D)
        # rows of tables the host keeps (Gemma 4 E4B's PLE records): read from the image's
        # store and written before each run, data movement only
        self._host_rows = getattr(self.image, "host_rows", None)
        self._rope = None if self.device_inputs else \
            [np.stack(t) for t in zip(*(rope_tables(spec, p) for p in range(cap)))]
        if pool_file is not None and getattr(self.image, "offload", None) is not None:
            from .moe import open_pool      # its read into the page cache runs during the build
            pool_file = open_pool(self.image.offload, pool_file, mapped=pool_map)
        images = self.image.build(W)
        self.backend = IsaBackend(self.cfg, images, adopt=True) if backend == "isa" else backend(
            self.cfg, images)
        self.resident = bool(resident) and lookup and bool(getattr(self.backend, "args", False))
        # path (a), docs/offload.md: the experts stream into the image's slots; the host's
        # server moves them (the ISA simulator calls it when every slice waits on WAITW; the
        # card's backend polls it while a run is in flight)
        self.server = None
        if getattr(self.image, "offload", None) is not None:
            self.server = self.image.serve(W, self.backend, pool_file)
            if release_weights and pool_file is not None and hasattr(W, "release"):
                W.release()                 # (the pool holds the experts; the image is written)
        # rows of tables the host keeps, asked for by the generate loop (Gemma 4 E4B: each
        # token's PLE record into the slot, opentpu.host.offload.RowServer)
        rs = getattr(self.image, "row_server", None)
        self.row_server = rs(self.backend) if rs is not None else None
        if self.row_server is not None and self.server is not None:
            self.row_server.mem = self.server.mem   # one memory, in order (BoardDram's queue)
        servers = [x for x in (self.server, self.row_server) if x is not None]
        if servers:
            poll = servers[0].poll if len(servers) == 1 else \
                (lambda: sum(x.poll() for x in servers))
            if isinstance(self.backend, IsaBackend):
                self.backend.machine.host = _isa_host(poll)
            elif hasattr(self.backend, "host"):
                self.backend.host = poll
                self.backend.servers = servers      # (rebased between runs: BoardBackend.start)
        self._conv_lo = getattr(spec, "conv_k", 1) - 1    # the first run-time position
        self._decodes: dict = {}            # resident: blocks -> (programs, run_args)
        # a MoE model's prompt layer by layer, `layer_major` rows a run (prefill_layers; 0:
        # token by token): its programs, (layer, blocks, rows, embedded) and "head" ->
        # (programs, run_args)
        self.layer_major = int(layer_major)
        self.pooled, self.restore = pooled, restore     # (the expert slots during it)
        self.embed_runs = embed_runs        # (the embed and compile-time-position runs with
                                            # the embedding rows from the host: prefill_layers)
        self.layer_ahead = layer_ahead      # (the next layer's experts sent during a layer's)
        self.layer_hint = layer_ahead == "hint"     # (the runs' own guess of them: hints)
        self.ahead_part = AHEAD_PART        # (with them: an idle poll's part, begin_prefill's)
        self.expert_major = bool(expert_major)      # (a layer's experts in one run a chunk)
        self._layer_runs: dict = {}
        self._layer_next: dict = {}         # their compiles in the worker processes: Futures
        if self.layer_major and not (getattr(self.image, "prefill_rows", 0) and self.device_inputs
                                     and getattr(self.backend, "args", False) and batch == 1):
            raise ValueError("layer-major prefill needs an image with prefill rows and lookup "
                             "tables (compile_layer_run), batch 1 and a backend with run "
                             "arguments")
        if self.expert_major and not (self.layer_major and self.server is not None and pooled
                                      and not self.layer_hint):
            raise ValueError("expert-major MoE needs layer-major prefill, the expert server "
                             "with pooled slots, and no layer hints (its runs post needs)")
        self._gens: dict = {}               # the generate loop: (blocks, mode) -> programs
                                            # (a list, or split: (first parts, second parts))
        self.gen_split = None               # split generate programs: None when a bucket's
                                            # does not fit, True always, False never
        self._chained: dict = {}            # mode -> (key, its buckets in the chain area)
        self.gen_debug = False              # generate_card: the logits too (gen_logits)
        self.gen_logits = None
        self.poss = [0] * batch
        # step(): stream the logits when the backend can (not while its wait serves the host)
        self.stream_logits = getattr(self.backend, "host", None) is None
        # ... and the step programs fill the region themselves (fill_logits) when the backend
        # polls ICOUNT for the fill (BoardBackend.fills): the host writes nothing there
        self.image.stream_fill = bool(self.stream_logits and self.cfg.S == 1
                                      and getattr(self.backend, "fills", False))
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
        if self.prompt_runs:                # bucket 1's prompt programs, ahead (prefill.warm)
            from . import prefill as PF
            PF.warm(self)

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
                      self._image_kw, self.image.stream_fill))
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

    @property
    def layout(self) -> tuple:
        """What decides the image's layout and so its programs (opentpu/progcache.py)."""
        return (self.spec, self.cfg, self.cap, self.batch, self.rows, self.block,
                tuple(sorted(self._image_kw.items())))

    def _compile_decode(self, blocks: int, lo: int | None = None):
        """lo: the bucket's first position (_worker_decode's; the compile thread gets it too)."""
        lo = max((blocks - 1) * self.block, self._conv_lo) if lo is None else lo
        progs, ra = self.image.compile_decode(blocks, lo, self.block)
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
            self._decodes[b] = self.cached(self._decode_what(b),
                                           lambda: self._take(("decode", b),
                                                              self._compile_decode, b))
        return self._decodes[b]

    def cached(self, what, compile):
        """compile() -> (programs, run_args), through the program cache with prog_cache."""
        return PC.get(self.layout, what, compile) if self.prog_cache else compile()

    def _decode_what(self, b: int) -> tuple:
        what = ("decode", b, max((b - 1) * self.block, self._conv_lo))
        return what + ("fill",) if self.image.stream_fill else what    # fill_logits' or not

    def _prefetch(self, pos: int) -> None:
        """Precompile the steps at pos .. pos + ahead - 1 (those not in flight yet); resident:
        the bucket of pos and, DECODE_LEAD positions before its end, the next one."""
        queued = {k for k, _ in self._next}
        if self.resident and pos >= self._conv_lo:
            for p in (pos, pos + DECODE_LEAD):
                b = p // self.block + 1
                if p < self.cap and b not in self._decodes and ("decode", b) not in queued \
                        and not (self.prog_cache and PC.has(self.layout, self._decode_what(b))):
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
        self._write_host_rows([token])
        start = getattr(self.backend, "start", None)
        v_loc = self.image.v_loc
        vocab = S * v_loc
        stream = None
        if start is not None and S == 1 and self.stream_logits and \
                getattr(self.backend, "streams", False):
            piece = 4 * min(HEAD_CHUNK, self.cfg.TMEM_WORDS // 8)     # _lm_head's chunks
            stream = (io["logits"], 4 * vocab, piece)
            if self.image.stream_fill:      # the program fills the region (fill_logits)
                stream += (fill_gate(progs if isinstance(progs, np.ndarray) else progs[0]),)
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

    def _write_host_rows(self, tokens) -> None:
        if self.row_server is not None:     # a request the last run posted, served first: its
            self.row_server.poll()          # row must not land after these
            flush = getattr(self.row_server.mem, "flush", None)
            if flush is not None:           # (BoardDram: written by its DMA thread)
                flush()
        if self._host_rows is not None:
            for a, v in self._host_rows(tokens):
                for s in range(self.cfg.S):
                    self.backend.write(s, a, v)

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
        self._write_host_rows(tokens)
        st = self.backend.run(programs)
        st["rows"] = len(rows)
        self.stats.append(st)
        v, v_loc = spec.vocab, self.image.v_loc
        out = [np.concatenate([self.backend.read(s, io["logits"] + 4 * (r * v + s * v_loc),
                                                 4 * v_loc).view(np.float32) for s in range(S)])
               for r in logit_rows]
        return np.array(out, np.float32).reshape(len(logit_rows), v)

    def _chunk(self, seq: int, p0: int, n: int, left: int, fit: int, toks=None,
               whole: bool = True):
        """fit_chunk, its programs prepared for the backend."""
        n, progs, fit = fit_chunk(self.image, self.block, seq, p0, n, left, fit, toks, whole)
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
        are bit-identical to feeding the tokens one by one -- but with PAIR, a 4-bit MM of a
        run of more than MCOLS / 2 rows does not pair as the decode kernel's does, and its
        sums round in another order (qwen35_rows; tests/test_qwen35.py
        test_prefill_pair_sum_order). A run shrinks when its program
        does not fit TMEM or IMEM (long contexts); a single token runs the decode kernel.
        Without `chunk` a run of more than MCOLS rows takes whole passes of MCOLS rows (each
        pass streams every weight: fit_chunk).
        With the pipeline, the next run's program (after the last run: the first decode
        step's) is compiled while the device runs the current one."""
        from . import prefill as PF
        tokens = [int(t) for t in tokens]
        if self.prompt_runs and seq == 0 and chunk is None and PF.supported(self) \
                and PF.covers(self, self.poss[0], self.poss[0] + len(tokens)):
            yield from PF.chunks(self, tokens)          # docs/prefill.md
            return
        whole = chunk is None               # else runs of exactly `chunk` where they fit
        chunk = self.rows if chunk is None else max(1, min(chunk, self.rows))
        i = 0
        while i < len(tokens):
            p0, left = self.poss[seq], len(tokens) - i
            key = ("rows", seq, p0, chunk, left, self._fit_rows,
                   self._chunk_toks(tokens[i:], chunk), whole)
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
                    self._prefetch_chunks(seq, p0 + n, chunk, left - n, tokens[i + n:],
                                          whole)
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

    def _prefetch_chunks(self, seq: int, p0: int, chunk: int, left: int, rest=(),
                         whole: bool = True) -> None:
        """Precompile the runs of a prompt from position p0 on, as many as the pipeline has
        workers (a chunk's trace can take longer than its run: Qwen3.5 on the card), each
        predicted to take as many rows as the last one (TMEM and IMEM limit a run: the program
        grows with the context); a run whose prediction turns out wrong is waited for and
        dropped (_take), so the programs do not change."""
        queued = {k for k, _ in self._next}
        for _ in range(self._ahead):
            if left <= 0:
                break
            key = ("rows", seq, p0, chunk, left, self._fit_rows, self._chunk_toks(rest, chunk),
                   whole)
            if key not in queued:
                self._submit(key, self._chunk, _worker_chunk, *key[1:])
            n = min(chunk, self._fit_rows, self._run_rows_n, left)
            n = _whole_passes(n, self.cfg.MCOLS) if whole else n
            p0, left, rest = p0 + n, left - n, rest[n:]

    def prefill(self, tokens, seq: int = 0, chunk: int | None = None) -> np.ndarray:
        """Feed a prompt to sequence `seq`; returns the logits after its last token
        (prefill_chunks; chunk=1 runs token by token with the decode kernel; layer_major:
        prefill_layers)."""
        if self.layer_major and seq == 0 and chunk is None:
            return self.prefill_layers(tokens)
        logits = None
        for _, logits in self.prefill_chunks(tokens, seq, chunk):
            pass
        return logits

    def prefill_layers(self, tokens) -> np.ndarray:
        """A MoE model's prompt layer by layer (docs/offload.md, layer-major prefill): each
        chunk of the image's prefill rows passes a layer before the next, in runs of
        `layer_major` rows (the image's compile_layer_run: the first row's position and chunk
        row as run arguments; a run stays in one attention block), so that the expert slots
        serve one layer at a time; then the LM head of the last row. Runs of more than one row
        start from the chunk's embedding rows (one embed run a token). A model with
        convolutions (conv_k > 1) runs its rows before position conv_k - 1 at compile-time
        positions, from their embedding rows too. With the embedding rows from the host's table
        (embed_host) and not `embed_runs`, there are no embed runs: the rows before conv_k - 1
        run token by token (prefill_chunks) and layer 0's runs gather their rows from the
        host's slot. (The card's port A keeps the beat of its last read and its prefetch run
        across runs, and the host's writes do not drop them: embed runs back to back read the
        slot's scale beat first and got the first row's scales, docs/offload.md 13.6. A run
        whose first scale read follows a layer's is safe; embed_runs=True brings the embed and
        compile-time-position runs back on a bitstream that drops them at RUN.) The expert
        server's slots are pooled for the prompt (ExpertServer.begin_prefill: every slot serves
        the running layer) and given back to their layers before the head runs (end_prefill:
        `restore`, "lazy" by default). With the process pipeline the runs' programs compile in
        the worker processes ahead of their runs, in their order (_precompile_layers). With
        `layer_ahead` the server is given the next MoE layer's experts to send while a layer
        runs (ExpertServer.ahead_layer, between runs: the first MoE layer's after begin_prefill,
        the next one's before each layer's first run in a chunk, the first one's again before a
        chunk's last layer when another chunk follows); a request still names its own. With
        "hint" those calls name none: the layer's runs post the next layer's experts as hints
        (their router on the run's output rows), which the server queues. With `expert_major`
        (docs/offload.md 13.11) each MoE layer of a chunk is its runs (their rows routed, the
        experts they chose posted as needs, the shared expert) and then one expert run (each
        chosen expert once, on the rows that chose it); the next layer's runs (the last
        layer's: the head's) sum the outputs. The rows live in a scratch the server carves
        from the first expert slots for the prefill (begin_prefill's scratch: the first
        chunk's records, moe.em_record a row), so the head runs before end_prefill hands them
        back. Bit-identical to step() token by token (the states, the KV cache, the logits):
        only the slots the experts sit in move. Returns the logits after the last token."""
        img, K, B, R = self.image, getattr(self.spec, "conv_k", 1), self.block, self.layer_major
        self._drain()
        tokens = [int(t) for t in tokens]
        if self.pos + len(tokens) > self.cap:
            raise RuntimeError("KV cache full")
        part = []

        def run(key, vals):
            progs, ra, words = self._layer_run(key)
            progs = progs if words is None else words   # (the board: the same words object
            start = getattr(self.backend, "start", None)    # again loads nothing)
            if start is None:
                self.stats.append(self.backend.run(progs, args=arg_words(ra, vals)))
            else:
                start(progs, args=arg_words(ra, vals))
                self.stats.append(self.backend.wait())

        host = bool(getattr(img, "embed_host", False)) and not self.embed_runs
        if host and self.pos < K - 1:   # the rows before conv_k - 1 token by token (no embed
            n = min(len(tokens), K - 1 - self.pos)          # or compile-time-position runs)
            for _, lg in self.prefill_chunks(tokens[:n], 0, None):
                pass
            tokens = tokens[n:]
            if not tokens:
                return lg
        em = self.expert_major
        mo = getattr(self.spec, "moe", None)
        chunks, p0 = [], self.pos       # each chunk's runs: (key, run arguments, the host's rows
        for c0 in range(0, len(tokens), img.prefill_rows):  # written before it, or None)
            part, runs = tokens[c0:c0 + img.prefill_rows], []
            low = max(0, min(len(part), K - 1 - p0))    # rows at compile-time positions
            for i, t in enumerate([] if host else part[:len(part) if R > 1 else low]):
                runs.append(((-1, (p0 + i) // B + 1, 1, True),  # (embed_host: the token's row)
                             dict(RunPos.values(t, p0 + i, K, B), row=i), [t]))
            for li in range(self.spec.layers):
                i = 0
                while i < len(part):
                    p = p0 + i
                    if i < low:
                        n = min(R, low - i)
                        runs.append(((li, None, n, True, p), {"row": i}, None))
                    else:
                        n = min(R, len(part) - i, B - p % B)
                        gather = li == 0 and (R == 1 or host)   # (the run gathers its rows:
                        runs.append(((li, p // B + 1, n, R > 1 and not gather),  # the host's)
                                     dict(RunPos.values(part[i], p, K, B), row=i),
                                     part[i:i + n] if gather else None))
                    i += n
                if em and li >= mo.first:               # the layer's expert run
                    runs.append((("x", li), {"rows": len(part), "entries": len(part) * mo.k},
                                 None))
            chunks.append((part, runs))
            p0 += len(part)
        self._precompile_layers([k for _, runs in chunks for k, _, _ in runs] + ["head"])
        srv = self.server if self.pooled else None
        send = getattr(srv, "ahead_layer", None) if self.layer_ahead else None
        if srv is not None and hasattr(srv, "begin_prefill"):
            kw = {} if send is None else {"ahead": True} if not self.layer_hint else \
                {"ahead": True, "part": self.ahead_part}
            if em:                          # (the scratch: the first chunk's rows' records;
                from .moe import em_record  # the needs go in idle-poll parts, as hints')
                kw.update(expert_major=True, part=self.ahead_part, scratch=len(chunks[0][0]) *
                          em_record(self.spec.hidden, mo.k))
            srv.begin_prefill(**kw)
        if send is not None:
            self._send_ahead(send, 0)
            first, nm = self.spec.moe.first, img.offload.layers
        for c, (part, runs) in enumerate(chunks):
            cur = -1
            for key, vals, rows in runs:
                if send is not None and key[0] != "x" and key[0] > cur:  # a layer's first run:
                    cur = key[0]                        # the next MoE layer's experts, or
                    j = cur + 1 - first                 # the first's for the next chunk
                    if cur + 1 == self.spec.layers and c + 1 < len(chunks):
                        self._send_ahead(send, 0)
                    elif 0 < j < nm:
                        self._send_ahead(send, j)
                if rows is not None:
                    self._write_host_rows(rows)
                try:
                    run(key, vals)
                except Exception as e:      # an expert run's WAITW timeout: where its needs
                    if key[0] != "x" or not hasattr(srv, "need_report"):    # stand (13.14)
                        raise
                    raise RuntimeError(f"expert run of layer {key[1]}: {e} "
                                       f"({srv.need_report()})") from e
            self.pos += len(part)
        if em:                              # (the head reads the scratch: before it goes)
            run("head", {"row": len(part) - 1})
        if srv is not None and hasattr(srv, "end_prefill"):    # (the last run has halted; an
            srv.end_prefill(self.restore)                       # all-hit request it posted is
                                                                # served first: settle)
        if not em:
            run("head", {"row": len(part) - 1})
        io, S, v_loc = img.io, self.cfg.S, img.v_loc
        return np.concatenate([self.backend.read(s, io["logits"] + 4 * s * v_loc, 4 * v_loc)
                               .view(np.float32) for s in range(S)])

    def _send_ahead(self, send, j: int) -> None:
        """MoE layer j's experts to the server's ahead_layer, as global ids in layer_ahead's
        order."""
        E = self.image.offload.E
        order = range(E) if self.layer_ahead is True else () if self.layer_hint else \
            self.layer_ahead[j]
        send(j, [j * E + int(e) for e in order])

    def _layer_run(self, key):
        """prefill_layers' programs, compiled once (layer_programs' key) -> (programs, run_args,
        the program assembled for a backend that runs words, else None); programs None when a
        worker process compiled it (_precompile_layers)."""
        if key not in self._layer_runs:
            fut = self._layer_next.pop(key, None)
            if fut is not None and not fut.done() and not self._ready.done():
                fut.cancel()                # the workers still starting: compiled here
                fut = None
            if fut is not None:
                words, ra = fut.result()
                self._layer_runs[key] = (None, ra, words)
            else:
                progs, ra = layer_programs(self.image, key, self.block, self.layer_hint,
                                           self.expert_major)
                words = np.asarray(I.assemble(progs[0]), np.uint32) \
                    if getattr(self.backend, "runs_words", False) else None
                self._layer_runs[key] = (progs, ra, words)
        return self._layer_runs[key]

    def _precompile_layers(self, keys) -> None:
        """prefill_layers' runs not compiled yet, compiled by the worker processes in the order
        they run (the process pipeline): the card runs a layer while the next one's programs
        compile (22-46 ms each against a layer's 200-460 ms, docs/offload.md 13.7). Queued
        while the workers start too (a prompt right after the engine: the 26B's); until they
        are up, _layer_run compiles a run here and drops its compile."""
        if not self._procs:
            return
        for k in dict.fromkeys(keys):
            if k not in self._layer_runs and k not in self._layer_next:
                self._layer_next[k] = self._pool.submit(_worker_layer, k, self.layer_hint,
                                                        self.expert_major)

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
            chain = bool(getattr(self.backend, "chains", False))
            progs = self.cached(("gen", blocks, lo, chain, key[1], self.gen_debug,
                                 self.gen_split),
                                lambda: (G.compile_bucket(self.image, blocks, lo, self.block,
                                                          chain=chain, samp=samp,
                                                          debug=self.gen_debug,
                                                          split=self.gen_split), None))[0]
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
                self._write_host_rows([tok])        # the first token's rows (then the loop's
                for s in range(self.cfg.S):         # requests, served while it runs)
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
