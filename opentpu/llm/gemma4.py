"""Gemma 4 text decoder (google/gemma-4-E2B: the language model; the vision and audio encoders
are not loaded) on openTPU.

Gemma 4 E2B has 35 layers, hidden size 1536, a 262,144-token vocabulary and a tied LM head:
  attention  8 query heads and 1 KV head. Sliding layers attend over the last 512 tokens with
             256-wide heads (RoPE theta 1e4); every fifth layer is global: all tokens, 512-wide
             heads, "proportional" RoPE (theta 1e6) that rotates only the pairs (i, i + 256),
             i < 64. RMSNorm on each q, k and v head (v without a weight), attention scaling 1.
             The last 20 layers compute no K / V: they attend to the cache of the last earlier
             layer of their kind (layer 13, sliding; layer 14, global).
  MLP        gelu_tanh(W_gate x) * W_up x, 6144 wide, 12288 in the 20 KV-shared layers.
  per-layer  a second embedding table (PLE, 262144 x 35 x 256) gives each layer a 256-vector
  inputs     per token: pli = (RMSNorm(W_proj x0 / sqrt(H)) + PLE[t] * 16) / sqrt(2), and after
             the MLP each layer adds RMSNorm(W_p (gelu_tanh(W_g x) * pli_l)).
  norms      plain RMSNorm x * w (not Gemma 3's 1 + w) around every block: x += norm(attn(
             norm(x))), x += norm(mlp(norm(x))), then the per-layer input, then x *= a scalar
             of the layer (layer_scalar). The embedding is scaled by sqrt(H); the final logits
             are soft-capped, 30 tanh(z / 30) (monotonic: the host applies it).

Gemma 4 26B-A4B (30 layers, hidden 2816) differs: 16 query heads on 8 KV heads of 256 (sliding,
a 1024-token window) and on 2 of 512 (global) whose V is the K projection (attention_k_eq_v: no
v_proj; V the weightless RMSNorm of the raw K projection, K normed by k_norm and RoPE'd); no
per-layer inputs; a 2112-wide MLP (the image pads it to its down projection's column quantum:
2176 int8, 2304 4-bit) beside a mixture of 128 experts (top 8, 704 wide, gelu_tanh):
x += norm(norm_1(MLP(norm(x))) + norm_2(MoE(norm'(x)))), the router on x itself (RMSNorm,
x scale / sqrt(H), softmax, the top 8 renormalized, x per_expert_scale). On the device the
experts stream into DRAM slots (moe.moe_ffn, path (a) of docs/offload.md, section 11): one
token per program (rows = 1); the router reads the quantized unit norm (its scales folded
into its weights: moe.gemma_router), the experts the norm times pre_feedforward_layernorm_2's
gain, quantized (moe_ffn's g_exp; per_expert_scale folded into W_down: gemma_expert); the
dense MLP is emitted beside the expert request, while the host streams.

How it maps onto openTPU:
  * The embedding and PLE rows are gathered on the device (kernels/gather.py): the tied int8 LM
    head is the embedding table, and the PLE table (int8 or 4-bit, one record per token laid
    out for the gather) stays in DRAM; nothing per token comes from the host but the token id
    and the position, as run arguments (resident decode, qwen3.Engine(resident=True)). The
    per-layer inputs of all layers are computed at the token's start into a DRAM area that
    each layer loads its row of.
  * Sliding layers keep K / V in a ring of window + one attention block (768 slots): position
    p in slot p mod 768, and attention over the window's blocks, the first masked at its start
    and the last at its end (attention.Blocks). One block more than the window, so a prefill
    chunk's appends never overwrite a slot an earlier row of the chunk still reads. Global
    layers keep every position. The KV-shared layers read their source layer's cache.
  * K rows (and their block scales, right after the data) are POS bytes apart, as are the
    RoPE table's rows, so that a run-time position needs few argument words (6 of 8: the
    token x 3 for the gathers, the position x 3).
  * The layers run as hardware loops over their repeated unit (lfm2.plan: (4 sliding + 1
    global) x 3 with their own K / V, then x 4 sharing it); every layer kind has a block of
    its own size.

Pieces:
  Spec              model dimensions, layer kinds, KV sources and MLP widths (config.json)
  load_weights      the checkpoint's language model as a lazy mapping of fp32 numpy arrays
                    (HF names, model.language_model.* -> model.*); the PLE table as rows
  reference_logits  plain numpy forward pass (the math, fp32)
  Image             per-slice DRAM layout (one slice) and its contents
  gemma4_step       the ol kernel for one decode token (host inputs, or gathered on the device
                    at a run-time position)
  gemma4_rows       R consecutive prompt tokens per device run (chunked prefill)
  gemma4_prompt_run a prompt run (docs/prefill.md): R rows at a run-time position, their
                    tokens from the generate area's out[]

Weights are int8 or 4-bit (quant.py) with int8 activations (W8A8 / W4A8), the LM head int8 or
4-bit, the PLE table int8 or 4-bit (ple_format); decoding runs on qwen3.Engine (one sequence).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .. import fp32 as F
from .. import isa as I
from .. import qcache as QC
from .. import quant as Q
from .. import language as ol
from ..compiler import Affine, DevVar, KVDesc, QTensor, Tensor, current
from ..isasim import Config
from ..kernels import gather as GA
from ..kernels import mailbox as MB
from ..kernels.attention import Blocks, Bucket, _attend_heads
from ..kernels.lib import gelu_tanh, rmsnorm, rope
from ..kernels.mlp import _chunk, swiglu_down
from . import formats as FM
from . import generate as G
from . import moe as MO
from .lfm2 import plan
from ..host.offload import LINE, BackendDram, ExpertServer, Layout, RowLayout, RowServer
from .qwen3 import (ATTN_BLOCK, ATTN_DEPTH, RunPos, _Bump, _lm_head, _lm_head_rows, _qdesc,
                    _tdesc, fill_logits, step_descriptors)

SLIDE, FULL = "sliding", "full"


# =============================================================================== model spec
@dataclass(frozen=True)
class Spec:
    hidden: int
    kinds: tuple            # SLIDE or FULL per layer
    kv_src: tuple           # per layer: the layer whose K / V it attends to (itself: its own)
    ffn: tuple              # MLP width per layer
    n_q: int
    n_kv: int
    head_dim: int           # sliding layers
    global_head_dim: int    # full-attention layers
    vocab: int
    ple_dim: int            # per-layer input width (hidden_size_per_layer_input)
    window: int = 512       # sliding window, the current token included
    theta: float = 1e4
    global_theta: float = 1e6
    global_rot: int = 64    # rotated pairs of a global head (partial_rotary_factor * d / 2)
    eps: float = 1e-6
    softcap: float | None = 30.0
    tied: bool = True
    bos: int = 2
    eos: tuple = (1, 106)   # <eos>, <turn|>
    ckpt_layers: tuple | None = None   # the checkpoint layer of each layer (None: the same)
    embed: str = "int8"     # the embedding rows are dequantized on the device only (from the
                            # quantized LM head): an Engine always gets the lookup tables
    formats: str = ""       # per-layer weight formats over the image's wformat (layer_formats)
    fit_formats: str = ""   # the formats an int8 image takes when it does not fit the card
                            # (Image): from_hf's, the LM head and the own-KV layers' down
                            # projections in fp4 (docs/gemma4_e4b.md)
    mix: str = ""           # the recommended mix (wformat "mix": formats.named, MIXES)
    n_kv_global: int = 0    # the global layers' KV heads (0: n_kv)
    k_eq_v: bool = False    # global layers: V is the K projection (no v_proj), normed
                            # without a weight before K's norm and RoPE (26B-A4B)
    experts: int = 0        # MoE beside each layer's MLP (0: none): experts, top-k, width
    top_k: int = 0
    expert_ffn: int = 0
    k_rows: bool = False    # sliding K rows at their own stride, not the RoPE row's (Image.ks):
                            # 26B-A4B's 8 KV heads x 25 layers, 414 -> 146 MB at 1152 slots

    @property
    def layers(self) -> int:
        return len(self.kinds)

    def src(self, i: int) -> int:
        """The checkpoint layer of layer i."""
        return i if self.ckpt_layers is None else self.ckpt_layers[i]

    def hd(self, i: int) -> int:
        return self.global_head_dim if self.kinds[i] == FULL else self.head_dim

    def kvh(self, i: int) -> int:
        """Layer i's KV heads."""
        return self.n_kv_global if self.kinds[i] == FULL and self.n_kv_global else self.n_kv

    def kv_same(self, i: int) -> bool:
        """Layer i's V is its K projection (attention_k_eq_v, global layers)."""
        return self.k_eq_v and self.kinds[i] == FULL

    @property
    def moe(self) -> MO.MoESpec | None:
        """The MoE block beside every layer's MLP (moe.moe_ffn): HF's softmax over the experts,
        its top k renormalized (the softmax of the k largest logits), gelu_tanh experts."""
        if not self.experts:
            return None
        return MO.MoESpec(E=self.experts, k=self.top_k, ffn=self.expert_ffn, rule="softmax",
                          act="gelu_tanh")

    @staticmethod
    def from_hf(model_dir) -> "Spec":
        top = json.loads((Path(model_dir) / "config.json").read_text())
        c = top.get("text_config", top)
        L = c["num_hidden_layers"]
        kinds = tuple(FULL if t == "full_attention" else SLIDE for t in c["layer_types"])
        # the global layers' head size and KV heads: global_head_dim / num_global_key_value_
        # heads (the checkpoint's config), or per-layer overrides (transformers >= 5.17's)
        over = {int(i): v for i, v in (c.get("per_layer_config") or {}).items()}
        if any(kinds[i] != FULL or set(v) - {"head_dim", "num_key_value_heads"}
               for i, v in over.items()):
            raise ValueError(f"per-layer config {over} is not supported")
        # (HF: num_global_key_value_heads only with attention_k_eq_v)
        kev = bool(c.get("attention_k_eq_v"))
        gkv = {v.get("num_key_value_heads") for v in over.values()} | \
            {c.get("num_global_key_value_heads") if kev else None}
        gkv -= {None}
        if len(gkv) > 1 or (not kev and gkv - {c["num_key_value_heads"]}):
            raise ValueError(f"global layers' KV heads {sorted(gkv)} are not supported")
        gds = {v["head_dim"] for v in over.values() if "head_dim" in v}
        if len(gds) > 1:
            raise ValueError(f"global layers with different head sizes {sorted(gds)}")
        first = L - c.get("num_kv_shared_layers", 0)
        src = tuple(i if i < first else max(j for j in range(first) if kinds[j] == kinds[i])
                    for i in range(L))
        wide = c.get("use_double_wide_mlp", False)
        ff = c["intermediate_size"]
        rp = c.get("rope_parameters") or {}
        g, s = rp.get("full_attention", {}), rp.get("sliding_attention", {})
        gd = c.get("global_head_dim") or (gds.pop() if gds else c["head_dim"])
        if g.get("rope_type", "proportional") != "proportional" or \
                s.get("rope_type", "default") != "default":
            raise ValueError("RoPE types other than default (sliding) / proportional (global)")
        if c.get("hidden_activation", "gelu_pytorch_tanh") != "gelu_pytorch_tanh":
            raise ValueError(f"activation {c['hidden_activation']} is not gelu_tanh")
        moe = c.get("enable_moe_block", False)
        # int8 layers that do not fit: the head and the down projections in fp4 in the layers
        # with their own K / V, one loop (E4B's best mix that keeps two loops)
        fit = f"head=fp4,down@0-{first - 1}=fp4" if first > 0 else "head=fp4"
        if moe:                 # a MoE's DRAM beside its layers is expert slots (Image experts)
            fit = ""
        return Spec(hidden=c["hidden_size"], kinds=kinds, kv_src=src,
                    ffn=tuple(2 * ff if wide and i >= first > 0 else ff for i in range(L)),
                    n_q=c["num_attention_heads"], n_kv=c["num_key_value_heads"],
                    head_dim=c["head_dim"], global_head_dim=gd, vocab=c["vocab_size"],
                    ple_dim=c["hidden_size_per_layer_input"], window=c["sliding_window"],
                    theta=s.get("rope_theta", 1e4), global_theta=g.get("rope_theta", 1e6),
                    global_rot=int(g.get("partial_rotary_factor", 1.0) * gd // 2),
                    eps=c.get("rms_norm_eps", 1e-6),
                    softcap=c.get("final_logit_softcapping"),
                    tied=top.get("tie_word_embeddings", c.get("tie_word_embeddings", True)),
                    bos=c.get("bos_token_id", 2),
                    eos=(c.get("eos_token_id", 1), 106), fit_formats=fit, mix=FM.mix_for(c),
                    n_kv_global=gkv.pop() if gkv else 0, k_eq_v=kev,
                    experts=c["num_experts"] if moe else 0,
                    top_k=c["top_k_experts"] if moe else 0,
                    expert_ffn=c["moe_intermediate_size"] if moe else 0, k_rows=moe)

    def check(self, cfg: Config) -> None:
        D = cfg.D
        need = [(cfg.S == 1, "one slice (a single KV head)"),
                (self.head_dim % D == 0 and self.global_head_dim % D == 0, "head dims % D"),
                (self.hidden % D == 0 and self.ple_dim % D == 0, "hidden, ple_dim % D"),
                (all(self.n_q % self.kvh(i) == 0 for i in range(self.layers)),
                 "n_q % KV heads"),
                (self.global_rot <= self.global_head_dim // 2, "global_rot"),
                (self.window % ATTN_BLOCK == 0, f"window % {ATTN_BLOCK}"),
                (max(self.n_q * self.global_head_dim, self.hidden) <= cfg.ACT_BLOCKS * D,
                 "an inner dimension exceeds ACT RAM")]
        bad = [m for ok, m in need if not ok]
        if bad:
            raise ValueError("model does not map onto this openTPU config: " + "; ".join(bad))

    def image(self, cfg: Config, cap: int, batch: int = 1, rows: int = 1,
              wformat: str = "int8", head_format: str | None = None, lookup: bool = False,
              ple_format: str | None = None, ple_host: bool | None = None,
              formats: str | None = None, experts: int | None = None,
              prefill_rows: int | None = None) -> "Image":
        return Image(self, cfg, cap, batch, rows, wformat, head_format, lookup, ple_format,
                     ple_host=ple_host, formats=formats, experts=experts,
                     prefill_rows=prefill_rows)

    def truncated(self, layers) -> "Spec":
        """A model of some of the checkpoint's layers (in order): each shared layer attends to
        the last earlier own layer of its kind, which must be among them (tests, RTL runs)."""
        layers = list(layers)
        kinds = tuple(self.kinds[i] for i in layers)
        own = [self.kv_src[i] == i for i in layers]
        src = []
        for n, i in enumerate(layers):
            if own[n]:
                src.append(n)
                continue
            c = [m for m in range(n) if own[m] and kinds[m] == kinds[n]]
            if not c:
                raise ValueError(f"layer {i} shares K / V but no earlier own {kinds[n]} layer "
                                 f"is in {layers}")
            src.append(c[-1])
        return replace(self, kinds=kinds, kv_src=tuple(src),
                       ffn=tuple(self.ffn[i] for i in layers),
                       ckpt_layers=tuple(self.src(i) for i in layers))


# =============================================================================== weights
class _Rows:
    """A checkpoint tensor read by rows on demand (the PLE table: 262144 x 8960 bf16, 4.7 GB):
    t[i], t[a:b] and t[[i, j, ...]] give fp32 numpy rows."""

    def __init__(self, f, name: str):
        self.f, self.name = f, name
        self.shape = tuple(f.get_slice(name).get_shape())

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        sl = self.f.get_slice(self.name)
        if isinstance(key, slice):
            return sl[key].float().numpy()
        if isinstance(key, (int, np.integer)):
            return sl[int(key):int(key) + 1].float().numpy()[0]
        return np.stack([sl[int(i):int(i) + 1].float().numpy()[0] for i in key])


class Weights(dict):
    """The language model of a Gemma 4 checkpoint, loaded lazily: W[name] reads one tensor as
    fp32 (HF names of a text-only model, model.language_model.* -> model.*); the PLE table
    stays in the file (_Rows). Holds nothing but the open file."""

    PLE = "model.embed_tokens_per_layer.weight"

    def __init__(self, model_dir):
        from safetensors import safe_open
        super().__init__()
        self.model_dir = Path(model_dir)
        self._files = {}
        for p in sorted(Path(model_dir).glob("*.safetensors")):
            f = safe_open(str(p), "pt")
            for k in f.keys():
                if k.startswith("model.language_model."):
                    self._files[k.replace("model.language_model.", "model.", 1)] = (f, k)
        if not self._files:
            raise ValueError(f"{model_dir}: no Gemma 4 language model tensors")

    def keys(self):
        return self._files.keys()

    def __contains__(self, k):
        return k in self._files

    def __iter__(self):
        return iter(self._files)

    def __len__(self):
        return len(self._files)

    def __getitem__(self, k):
        f, name = self._files[k]
        if k == self.PLE:
            return _Rows(f, name)
        return f.get_tensor(name).float().numpy()

    def get(self, k, default=None):
        return self[k] if k in self._files else default

    def part(self, k, i) -> np.ndarray:
        """Tensor k's i-th entry along its first axis (one expert of a fused expert tensor),
        read alone, fp32."""
        f, name = self._files[k]
        return f.get_slice(name)[int(i)].float().numpy()

    def rows(self, k, idx) -> np.ndarray:
        """Rows idx (a list) of tensor k, fp32, without loading the rest."""
        f, name = self._files[k]
        return _Rows(f, name)[list(idx)]


def load_weights(model_dir) -> Weights:
    return Weights(model_dir)


def _rows(W, k, idx) -> np.ndarray:
    """Rows idx of W[k] (a lazy Weights reads only them)."""
    if isinstance(W, Weights):
        return W.rows(k, idx)
    return np.asarray(W[k], np.float32)[list(idx)]


# =============================================================================== reference
def rope_freqs(spec: Spec, kind: str) -> np.ndarray:
    """The inverse frequencies of a kind's RoPE (float64): sliding, head_dim / 2 of them;
    global ("proportional"), the global_rot rotated pairs over the global head dimension."""
    if kind == FULL:
        d = spec.global_head_dim
        return 1.0 / spec.global_theta ** (np.arange(spec.global_rot, dtype=np.float64) * 2 / d)
    d = spec.head_dim
    return 1.0 / spec.theta ** (np.arange(d // 2, dtype=np.float64) * 2 / d)


def rope_tables(spec: Spec, pos: int, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """cos, sin (fp32) of a position: [head_dim / 2] (sliding) or [global_rot] (global)."""
    ang = pos * rope_freqs(spec, kind)
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def _norm(v, g, eps):
    v = v / np.sqrt(np.mean(v * v, axis=-1, keepdims=True) + eps)
    return v if g is None else v * g


def _gelu(v):
    return 0.5 * v * (1 + np.tanh(math.sqrt(2 / math.pi) * (v + 0.044715 * v ** 3)))


def _rot(v, c, s, half: int):
    """Rotate-half RoPE on the pairs (i, i + half), i < len(c), of the last axis (the others
    pass through): the whole head when len(c) = half (sliding), the proportional RoPE's first
    global_rot pairs of each half (global)."""
    n = c.shape[-1]
    out = v.copy()
    a, b = v[..., :n], v[..., half:half + n]
    out[..., :n] = a * c - b * s
    out[..., half:half + n] = b * c + a * s
    return out


def ple_inputs(spec: Spec, W, x0, tokens) -> np.ndarray:
    """The per-layer inputs [T, L, ple_dim] of embeddings x0 [T, H] (already x sqrt(H))."""
    T, L, P = len(tokens), spec.layers, spec.ple_dim
    f32 = np.float32
    pp = (x0 @ W["model.per_layer_model_projection.weight"].T) * f32(spec.hidden ** -0.5)
    pp = _norm(pp.reshape(T, -1, P), W["model.per_layer_projection_norm.weight"], spec.eps)
    cols = np.concatenate([np.arange(spec.src(i) * P, (spec.src(i) + 1) * P) for i in range(L)])
    pe = W[Weights.PLE][list(tokens)].reshape(T, -1)[:, cols].reshape(T, L, P) * f32(P ** 0.5)
    if spec.ckpt_layers is not None:            # a truncated model: its layers' columns
        pp = pp[:, list(spec.ckpt_layers)]
    return (pp + pe) * f32(2 ** -0.5)


def moe_route(spec: Spec, W, p: str, x) -> tuple[np.ndarray, np.ndarray]:
    """The router of layer prefix p on rows x [T, H] (the residual stream before the MLP):
    (experts [T, top_k], weights [T, top_k]): softmax over RMSNorm(x) scale / sqrt(H) W_r, the
    top k renormalized, times their per_expert_scale."""
    r = p + "router."
    z = _norm(x, None, spec.eps) * W[r + "scale"] * np.float32(spec.hidden ** -0.5)
    s = z @ W[r + "proj.weight"].T
    pr = np.exp(s - s.max(axis=1, keepdims=True))
    pr /= pr.sum(axis=1, keepdims=True)
    top = np.argsort(-pr, axis=1, kind="stable")[:, :spec.top_k]
    w = np.take_along_axis(pr, top, 1)
    return top, w / w.sum(axis=1, keepdims=True) * W[r + "per_expert_scale"][top]


def _moe(spec: Spec, W, p: str, x) -> np.ndarray:
    """The MoE block of layer prefix p (fp32): sum over the routed experts e of w_e W_down,e
    (gelu_tanh(W_gate,e h) * W_up,e h), h = pre_feedforward_layernorm_2(x); each expert's
    weights read once."""
    top, wt = moe_route(spec, W, p, x)
    h = _norm(x, W[p + "pre_feedforward_layernorm_2.weight"], spec.eps)
    F = spec.expert_ffn
    out = np.zeros_like(h)
    for e in np.unique(top):
        t, j = np.nonzero(top == e)
        gu = _rows(W, p + "experts.gate_up_proj", [int(e)])[0]          # [2F, H]
        a = h[t] @ gu.T
        y = (_gelu(a[:, :F]) * a[:, F:]) @ _rows(W, p + "experts.down_proj", [int(e)])[0].T
        np.add.at(out, t, y * wt[t, j][:, None])
    return out


def reference_logits(spec: Spec, W, tokens) -> np.ndarray:
    """fp32 numpy forward of the whole sequence (causal); returns logits [T, vocab] before the
    soft cap (softcap() applies it)."""
    tokens = list(tokens)
    T, H = len(tokens), spec.hidden
    f32 = np.float32
    E = W["model.embed_tokens.weight"]
    x = E[tokens] * f32(math.sqrt(H))
    pli = ple_inputs(spec, W, x, tokens) if spec.ple_dim else None
    tab = {k: [rope_tables(spec, p, k) for p in range(T)] for k in (SLIDE, FULL)}
    ii, jj = np.arange(T)[:, None], np.arange(T)[None, :]
    masks = {FULL: np.where(jj <= ii, 0, -np.inf).astype(f32),
             SLIDE: np.where((jj <= ii) & (jj > ii - spec.window), 0, -np.inf).astype(f32)}
    kv = {}
    for i in range(spec.layers):
        p = f"model.layers.{spec.src(i)}."
        a = p + "self_attn."
        kind, d, nkv = spec.kinds[i], spec.hd(i), spec.kvh(i)
        G = spec.n_q // nkv
        cos = np.stack([c for c, _ in tab[kind]])[:, None, :]
        sin = np.stack([s for _, s in tab[kind]])[:, None, :]
        h = _norm(x, W[p + "input_layernorm.weight"], spec.eps)
        q = (h @ W[a + "q_proj.weight"].T).reshape(T, spec.n_q, d)
        q = _rot(_norm(q, W[a + "q_norm.weight"], spec.eps), cos, sin, d // 2)
        if spec.kv_src[i] == i:
            k = (h @ W[a + "k_proj.weight"].T).reshape(T, nkv, d)
            v = k if spec.kv_same(i) else (h @ W[a + "v_proj.weight"].T).reshape(T, nkv, d)
            kv[i] = (_rot(_norm(k, W[a + "k_norm.weight"], spec.eps), cos, sin, d // 2),
                     _norm(v, None, spec.eps))
        k, v = kv[spec.kv_src[i]]
        o = np.zeros((T, spec.n_q, d), f32)
        for hq in range(spec.n_q):
            s = q[:, hq] @ k[:, hq // G].T + masks[kind]
            s = np.exp(s - s.max(axis=1, keepdims=True))
            o[:, hq] = (s / s.sum(axis=1, keepdims=True)) @ v[:, hq // G]
        att = o.reshape(T, -1) @ W[a + "o_proj.weight"].T
        x = x + _norm(att, W[p + "post_attention_layernorm.weight"], spec.eps)
        h = _norm(x, W[p + "pre_feedforward_layernorm.weight"], spec.eps)
        m = (_gelu(h @ W[p + "mlp.gate_proj.weight"].T) * (h @ W[p + "mlp.up_proj.weight"].T)) \
            @ W[p + "mlp.down_proj.weight"].T
        if spec.experts:                        # the MoE beside the MLP, on the same residual
            m = _norm(m, W[p + "post_feedforward_layernorm_1.weight"], spec.eps) + \
                _norm(_moe(spec, W, p, x), W[p + "post_feedforward_layernorm_2.weight"],
                      spec.eps)
        x = x + _norm(m, W[p + "post_feedforward_layernorm.weight"], spec.eps)
        if pli is not None:
            g = _gelu(x @ W[p + "per_layer_input_gate.weight"].T) * pli[:, i]
            y = g @ W[p + "per_layer_projection.weight"].T
            x = x + _norm(y, W[p + "post_per_layer_input_norm.weight"], spec.eps)
        x = x * W[p + "layer_scalar"]
    x = _norm(x, W["model.norm.weight"], spec.eps)
    head = E if spec.tied else W["lm_head.weight"]
    return x @ head.T


def emulated_logits(spec: Spec, W, tokens, D: int = 128, wformat: str = "int8",
                    head_format: str | None = None, ple_format: str = "int8",
                    formats: str | None = None, routes: list | None = None,
                    routing: dict | None = None) -> np.ndarray:
    """float64 decode with openTPU's quantization points and none of its rounding (as
    qwen3.emulated_logits): weights in their formats (per layer: layer_formats, as Image),
    int8 matmul inputs per D-block, int8 K (per token and D-block) and V (per token), int8 P
    (per D tokens), the embedding and PLE rows as the device gathers them (int8 / 4-bit), the
    exact sliding window; the MoE block as moe_ffn runs it (the int8 router on the quantized
    unit norm, its scales folded: moe.gemma_router; the experts in expert_format on the norm
    times pre_feedforward_layernorm_2's gain, quantized: g_exp, gemma_expert), each token's
    (position, layer, experts, the k-th logit's margin over the next) appended to `routes`;
    `routing` {(position, layer): experts} replaces the top k where it has an entry (the
    card's choices: a near-tie routes either way). Before the soft cap."""
    from .qwen3 import _fake_q, _fake_w
    wformat, formats = FM.named(spec, wformat, formats)
    lf, pf, fh = layer_formats(spec, wformat, formats)
    ef = expert_format(spec, wformat, formats) if spec.experts else None
    hf = head_format or fh or wformat
    f64 = np.float64
    Wq: dict = {}

    def w(n, fmt):
        if (n, fmt) not in Wq:              # columns padded with zeros to whole D-blocks
            a = np.asarray(W[n], np.float32)    # (an MLP's down projection: _ffn_pad)
            Wq[n, fmt] = _fake_w(np.pad(a, ((0, 0), (0, -a.shape[1] % D))), D, fmt)
        return Wq[n, fmt]

    H, P, L = spec.hidden, spec.ple_dim, spec.layers
    head = "model.embed_tokens.weight" if spec.tied else "lm_head.weight"
    if P:
        cols = np.concatenate([np.arange(spec.src(i) * P, (spec.src(i) + 1) * P)
                               for i in range(L)])
        S = GA.record_blocks(-(-L * P // D), ple_format)
        wp = W["model.per_layer_model_projection.weight"].reshape(-1, P, H)[
            [spec.src(i) for i in range(L)]].reshape(-1, H)
        wpq = _fake_w(wp, D, pf) * H ** -0.5
        gpl = np.asarray(W["model.per_layer_projection_norm.weight"], f64) * 2 ** -0.5
    K, V = {}, {}
    out = []
    for pos, tk in enumerate(tokens):
        x = w(head, hf)[tk] * math.sqrt(H)
        if P:
            pe = GA.dequant_records(GA.pack_records(
                np.asarray(W[Weights.PLE][[tk]], np.float32)[:, cols] * np.float32(
                    (P / 2) ** 0.5), ple_format, D, S), ple_format, D, S)[0, :L * P]
            pli = _norm((wpq @ _fake_q(x, D)).reshape(L, P), gpl, spec.eps) + \
                pe.reshape(L, P)
        for i in range(L):
            p = f"model.layers.{spec.src(i)}."
            a = p + "self_attn."
            kind, d, nkv = spec.kinds[i], spec.hd(i), spec.kvh(i)
            G = spec.n_q // nkv
            fa, fg, fd, fp = lf[i]
            c, s_ = rope_tables(spec, pos, kind)
            h = _fake_q(_norm(x, W[p + "input_layernorm.weight"], spec.eps), D)
            q = (w(a + "q_proj.weight", fa) @ h).reshape(spec.n_q, d)
            q = _rot(_norm(q, W[a + "q_norm.weight"], spec.eps), c, s_, d // 2)
            if spec.kv_src[i] == i:
                k = (w(a + "k_proj.weight", fa) @ h).reshape(nkv, d)
                v = k if spec.kv_same(i) else (w(a + "v_proj.weight", fa) @ h).reshape(nkv, d)
                k = _rot(_norm(k, W[a + "k_norm.weight"], spec.eps), c, s_, d // 2)
                K.setdefault(i, []).append(_fake_q(k, D))
                V.setdefault(i, []).append(_fake_q(_norm(v, None, spec.eps), d))
            src = spec.kv_src[i]
            lo = max(0, pos + 1 - spec.window) if kind == SLIDE else 0
            Kh, Vh = np.stack(K[src][lo:], 1), np.stack(V[src][lo:], 1)
            o = np.zeros((spec.n_q, d))
            for hq in range(spec.n_q):
                sc = Kh[hq // G] @ _fake_q(q[hq], D)
                pp = np.exp(sc - sc.max())
                T = len(pp)
                ppad = np.zeros(-(-T // D) * D)
                ppad[:T] = pp
                o[hq] = (_fake_q(ppad, D)[:T] @ Vh[hq // G]) / pp.sum()
            att = w(a + "o_proj.weight", fa) @ _fake_q(o.reshape(-1), D)
            x = x + _norm(att, W[p + "post_attention_layernorm.weight"], spec.eps)
            h = _fake_q(_norm(x, W[p + "pre_feedforward_layernorm.weight"], spec.eps), D)
            wd = w(p + "mlp.down_proj.weight", fd)
            u = _gelu(w(p + "mlp.gate_proj.weight", fg) @ h) * \
                (w(p + "mlp.up_proj.weight", fg) @ h)
            m = wd @ _fake_q(np.pad(u, (0, wd.shape[1] - len(u))), D)
            if spec.experts:
                xn = _norm(x, None, spec.eps)
                g2 = np.asarray(W[p + "pre_feedforward_layernorm_2.weight"], f64)
                xs, xe = _fake_q(xn, D), _fake_q(xn * g2, D)
                if (p, "router") not in Wq:
                    Wq[p, "router"] = _fake_w(MO.gemma_router(W, p), D, "int8")
                lg = Wq[p, "router"] @ xs
                top = np.argsort(-lg, kind="stable")[:spec.top_k]
                if routing is not None and (pos, i) in routing:
                    top = np.asarray(routing[pos, i])
                    top = top[np.argsort(-lg[top], kind="stable")]
                if routes is not None:
                    nxt = np.sort(lg)[::-1][spec.top_k] if spec.top_k < len(lg) else -np.inf
                    routes.append((pos, i, [int(e) for e in top], float(lg[top[-1]] - nxt)))
                wt = np.exp(lg[top] - lg[top[0]])
                y = np.zeros(H)
                for e, we in zip(top, wt / wt.sum()):
                    if (p, int(e)) not in Wq:
                        Wq[p, int(e)] = [_fake_w(np.pad(a, ((0, 0), (0, -a.shape[1] % D))), D, ef)
                                         for a in MO.gemma_expert(W, p, int(e))]
                    eg, eu, ed = Wq[p, int(e)]
                    u = _gelu(eg @ xe) * (eu @ xe)
                    y += we * (ed @ _fake_q(np.pad(u, (0, ed.shape[1] - len(u))), D))
                m = _norm(m, W[p + "post_feedforward_layernorm_1.weight"], spec.eps) + \
                    _norm(y, W[p + "post_feedforward_layernorm_2.weight"], spec.eps)
            x = x + _norm(m, W[p + "post_feedforward_layernorm.weight"], spec.eps)
            if P:
                g = _gelu(w(p + "per_layer_input_gate.weight", fp) @ _fake_q(x, D)) * pli[i]
                y = w(p + "per_layer_projection.weight", fp) @ _fake_q(g, D)
                x = x + _norm(y, W[p + "post_per_layer_input_norm.weight"], spec.eps)
            x = x * np.asarray(W[p + "layer_scalar"], f64)
        out.append(w(head, hf) @ _fake_q(_norm(x, W["model.norm.weight"], spec.eps), D))
    return np.array(out)


def softcap(spec: Spec, logits: np.ndarray) -> np.ndarray:
    """The final logit soft cap, c tanh(z / c) (monotonic: greedy picks the same token)."""
    if not spec.softcap:
        return logits
    c = np.float32(spec.softcap)
    return (np.tanh(logits / c) * c).astype(np.float32)


# =============================================================================== DRAM image
RING_BLOCKS = 1         # sliding K / V ring: the window plus this many attention blocks
CARD_BYTES = 1 << 32    # the card's DRAM: the PLE table's place and the formats follow the fit
PLE_CHUNK = 8192        # PLE records quantized per pass (host memory)
BIG = 2.0 ** 100        # (tpos + 0.5 - c) * BIG * BIG: +-inf, the run-time mask rows
MLP_CHUNK = 768         # the MLP's F chunk at most: 4 prefill rows' gate / up in flight fit TMEM
PREFILL_CHUNK = 512     # a MoE model's layer-major prefill: prompt rows a chunk (Image's xbuf)
RUN_ROWS = 4            # and at most rows a run (its mask rows; moe_ffn_rows' request: R k ids)


def _mlp_chunk(f: int, D: int, q: int) -> int:
    """mlp's F chunk (about 8 per MLP), else the largest multiple of q up to MLP_CHUNK that
    divides the MLP (E4B's 10240: 512 at fp4, where _chunk takes 1280)."""
    c = _chunk(f, D, q)
    if c <= MLP_CHUNK:
        return c
    return next((k for k in range(MLP_CHUNK - MLP_CHUNK % q, 0, -q) if f % k == 0), c)


def _ffn_pad(f: int, fd: str, D: int) -> int:
    """The MLP width a layer block holds: f rounded up to its down projection's column quantum
    (D for int8, 2D for 4-bit: 26B-A4B's 2112 -> 2176 / 2304), the gate / up rows and down
    columns past f zero (gelu_tanh(0) * 0 = 0: the same sums)."""
    q = D if fd == "int8" else 2 * D
    return -(-f // q) * q


def _norms(P: int, moe: bool = False) -> dict:
    """A layer block's norm weights and their checkpoint names (g_ple: with per-layer inputs;
    g_f1, g_f2: the dense MLP's and the MoE's output norms beside each other; g_exp: the
    experts' input gain, moe_ffn's)."""
    n = {"g_in": "input_layernorm", "g_attn": "post_attention_layernorm",
         "g_pre": "pre_feedforward_layernorm", "g_ffn": "post_feedforward_layernorm"}
    if P:
        n["g_ple"] = "post_per_layer_input_norm"
    if moe:
        n.update(g_f1="post_feedforward_layernorm_1", g_f2="post_feedforward_layernorm_2",
                 g_exp="pre_feedforward_layernorm_2")
    return n


def _key(spec: Spec, i: int, lf: tuple = ()) -> tuple:
    """What decides a layer block's layout: attention kind, own K / V, MLP width, and the
    layer's weight formats (layer_formats)."""
    return spec.kinds[i], spec.kv_src[i] == i, spec.ffn[i], lf


def layer_formats(spec: Spec, wformat: str, formats: str | None = None) -> tuple:
    """Each layer's weight formats (attention, gate / up, down, its PLE gate and projection),
    the PLE projection's and the LM head's: `wformat` (the head: None), except where `formats`
    says otherwise (None: the OTPU_FORMATS environment variable, else spec.formats).
    `formats` is comma-separated "kind=fmt" or "kind@a-b=fmt", checkpoint layers a..b (a range
    wins over the whole model); the kinds are attn (q, k, v, o), mlp (gateup and down),
    gateup, down, ple (a layer's PLE gate and projection; without a range, the PLE projection
    too), head (no range) and experts (expert_format). docs/gemma4_e4b.md."""
    rules = _format_rules(spec, formats)
    lf = []
    for i in range(spec.layers):
        c = spec.src(i)
        mlp = FM.pick(rules, "mlp", c, wformat)
        lf.append((FM.pick(rules, "attn", c, wformat), FM.pick(rules, "gateup", c, mlp),
                   FM.pick(rules, "down", c, mlp), FM.pick(rules, "ple", c, wformat)))
    plain = FM.plain(rules)
    return tuple(lf), plain.get("ple", wformat), plain.get("head")


KINDS = ("attn", "mlp", "gateup", "down", "ple", "head", "experts")     # formats.rules' kinds


def _format_rules(spec: Spec, formats: str | None) -> list:
    """layer_formats' rules: (kind, first, last, ranged, format); opentpu/llm/formats.py."""
    return FM.rules(formats, KINDS, spec.formats)


def expert_format(spec: Spec, wformat: str, formats: str | None = None) -> str:
    """The routed experts' format (one for every layer: their slots are one size): the
    "experts=fmt" rule of `formats` (as layer_formats), else `wformat`."""
    rules = [r for r in _format_rules(spec, formats) if r[0] == "experts"]
    if any(r[3] for r in rules):
        raise ValueError("expert formats per layer range are not on the device (one slot size)")
    return rules[0][4] if rules else wformat


class _KV(KVDesc):
    """qwen3's KV cache layout, but K rows `ps` bytes apart with each row's block scales right
    after its data: a run-time position adds pos * ps to both (global layers: the RoPE row's
    stride too, one argument word; sliding layers: their slot tpos * ps). A multi-row K append must go row by row (QST writes one row's scales after
    the other's)."""

    def __init__(self, heads: dict, cap: int, d: int, D: int, ps: int):
        super().__init__(heads, cap, d, D, 1, 0)
        self.ps = ps

    def k(self, h: int) -> QTensor:
        a = Affine.of(self._h(h)["k"])
        return QTensor(a, a + self.d, (self.cap, self.d), self.ps, self.ps, self.D)


class Image:
    """Per-slice DRAM layout of a Gemma 4 model (one slice).

    [ I/O: x, pe, rope rows | final norm | logits | per-layer inputs | mask rows ] [ PLE
    projection ] [ layer blocks, run by run ] [ LM head ] [ PLE records ] [ lookup: RoPE rows,
    iota, one-hot operands ]. A run of the layer plan (lfm2.plan over the layers' _key) is
    `reps` copies of its unit, the unit its layers' blocks in order, so a hardware loop over
    the unit steps one address register; a block holds the layer's norms, layer scalar,
    projections (MLP down projection in column parts, qwen3's), per-layer input projections
    and, for a layer with its own K / V, its cache: a ring of window + RING_BLOCKS blocks
    (sliding) or `cap` positions (global), _KV's layout. The I/O area holds `rows` token rows:
    the host-written inputs of a per-position or prefill run (the gathered embedding and PLE
    rows, the RoPE rows), the logits, and the per-layer inputs [rows, layers, ple_dim].

    Formats: `wformat` for the layers' projections and the PLE projection, except where
    `formats` sets a kind's format per layer (layer_formats: a layer block's layout and its
    loop follow its formats), `head_format` for the LM head (the embedding table too),
    `ple_format` for the PLE table ("int8", "int4", "fp4"). `ple_host`: the PLE table stays on
    the host (docs/gemma4_e4b.md): the image holds a slot of `rows` records, which the host
    writes before each run (host_rows; the records of the run's tokens, read from ple_store,
    which build fills) and the gathers read at the row.
    None for either (and OTPU_PLE_FORMAT / OTPU_PLE_HOST unset): the table on the card in int8
    if the image then fits 4 GiB, else in fp4, else on the host in int8. `formats` None (and
    OTPU_FORMATS unset): spec.formats, or, when it is empty, the layers are int8 and no PLE
    choice fits them in 4 GiB, spec.fit_formats (E4B: the head and down 0-23 in fp4).
    `choices` holds what was chosen.
    """

    def __init__(self, spec: Spec, cfg: Config, cap: int, batch: int = 1, rows: int = 1,
                 wformat: str = "int8", head_format: str | None = None, lookup: bool = False,
                 ple_format: str | None = None, block: int = ATTN_BLOCK,
                 ple_host: bool | None = None, formats: str | None = None,
                 experts: int | None = None, prefill_rows: int | None = None):
        spec.check(cfg)
        if batch != 1:
            raise ValueError("Gemma 4 runs one sequence: batch=1")
        if cap % block:
            raise ValueError(f"KV capacity must be a multiple of the attention block {block}")
        import os
        wformat, formats = FM.named(spec, wformat, formats)
        if formats is None:
            formats = os.environ.get("OTPU_FORMATS")
        if formats is None:
            formats = spec.formats
            if not formats and wformat == "int8" and spec.fit_formats and \
                    Image(spec, replace(cfg, DRAM_BYTES=1 << 40), cap, 1, rows, wformat,
                          head_format, lookup, ple_format, block, ple_host,
                          "", experts).nbytes > CARD_BYTES:
                formats = spec.fit_formats  # int8 layers do not fit beside any PLE choice
        if ple_format is None:
            ple_format = os.environ.get("OTPU_PLE_FORMAT")
        if ple_host is None and os.environ.get("OTPU_PLE_HOST") is not None:
            ple_host = os.environ["OTPU_PLE_HOST"] == "1"
        if not spec.ple_dim:                    # no per-layer inputs: no table to place
            ple_format, ple_host = "int8", False
        if ple_format is None or ple_host is None:
            opts = [(f, h) for h in ([False, True] if ple_host is None else [ple_host])
                    for f in (["int8", "fp4"] if ple_format is None else [ple_format])
                    if not (h and f != "int8" and ple_format is None)]
            for ple_format, ple_host in opts:           # the first that fits (else the last)
                if Image(spec, replace(cfg, DRAM_BYTES=1 << 40), cap, 1, rows, wformat,
                         head_format, lookup, ple_format, block, ple_host,
                         formats, experts).nbytes <= CARD_BYTES:
                    break
        self.ple_host = bool(ple_host)
        D, H, P, L = cfg.D, spec.hidden, spec.ple_dim, spec.layers
        self.spec, self.cfg, self.cap, self.batch, self.rows = spec, cfg, cap, 1, rows
        # prefill rows per run: an MM streams its weights once per ACT RAM row chunk
        self.fit_rows = max(cfg.MCOLS, cfg.ACT_ROWS)
        self.block = block
        self.lf, self.pformat, fh = layer_formats(spec, wformat, formats)
        self.wformat, self.head_format, self.ple_format = wformat, \
            head_format or fh or wformat, ple_format
        self.formats = formats
        mo = spec.moe
        self.efmt = expert_format(spec, wformat, formats) if mo else None
        # the choices made here, so that another image of these keywords is this one (the
        # Engine's compile worker)
        self.choices = dict(formats=formats, ple_format=ple_format, ple_host=self.ple_host,
                            head_format=self.head_format)
        rb = lambda k, f: Q.row_bytes(k, f, D)                          # noqa: E731
        self.v_loc = spec.vocab
        self.ring = min(cap, spec.window + RING_BLOCKS * block)         # sliding cache slots
        hs, hg, rot = spec.head_dim, spec.global_head_dim, spec.global_rot
        self.rw = hs + 2 * rot                  # a RoPE row: cos, sin (sliding), cos, sin (global)
        dmax = max(hs, hg)
        self.ps = -(-max(4 * self.rw, dmax + 4 * dmax // D) // D) * D  # K / RoPE row stride
        # K rows of a kind: global layers' at a run-time position pos * ps (the RoPE row's
        # argument word); sliding layers' too, or with spec.k_rows their own (a row and its
        # scales: tpos * that, an argument word and address register more)
        self.ks = {FULL: self.ps, SLIDE: -(-(hs + 4 * hs // D) // D) * D if spec.k_rows
                   else self.ps}
        self.ple_S = GA.record_blocks(-(-L * P // D), ple_format) if P else 0  # record blocks
        self.ple_rec = GA.record_bytes(self.ple_S, ple_format, D) if P else 0
        b = _Bump()
        R = rows
        # the layer-major prefill's rows a chunk (a MoE model's: compile_layer_run)
        self.prefill_rows = (PREFILL_CHUNK if spec.experts else 0) if prefill_rows is None \
            else prefill_rows
        # mask rows: a pair for each row of a run at a run-time position (a layer run's; with
        # lookup tables a prompt run's, up to the I/O rows)
        mr = RUN_ROWS if self.prefill_rows else (R if lookup else 1)
        self.io = {"x": b.alloc(4 * H * R), "pe": b.alloc(4 * self.ple_S * D * R),
                   "rope": b.alloc(4 * self.rw * R), "gf": b.alloc(4 * H),
                   "logits": b.alloc(4 * spec.vocab * R), "pli": b.alloc(4 * R * L * P),
                   "mask": b.alloc(4 * 2 * block * mr), "z2": b.alloc(4 * 2 * block),
                   "g_pln": b.alloc(4 * P)}
        if self.ple_host:           # the generate loop's requests for the next token's record
            self.io["ple_mbox"] = b.alloc(RowLayout.WORDS)
        if self.prefill_rows:       # its residual stream, and moe_ffn_rows' outputs (a
            self.io["xbuf"] = b.alloc(4 * H * self.prefill_rows)     # request's ids and a sink)
            if spec.experts:
                self.io["moe_scratch"] = b.alloc(4 * H * (LINE // 4 + 1))
        self.wproj = (b.alloc(L * P * rb(H, self.pformat)), b.alloc(4 * L * P * (H // D))) \
            if P else None
        # layer blocks: one layout per key
        self.dchunk = {(f, fd): _mlp_chunk(_ffn_pad(f, fd, D), D, D if fd == "int8" else 2 * D)
                       for f, (_, _, fd, _) in zip(spec.ffn, self.lf)}
        self.bofs, self.bsize = {}, {}
        for i in range(L):
            k = self._key(i)
            if k in self.bofs:
                continue
            kind, own, ff, (fa, fg, fd, fp) = k
            d = hg if kind == FULL else hs
            nq = spec.n_q * d
            lb = _Bump()
            o = {n: lb.alloc(4 * H) for n in _norms(P, bool(mo))}
            o["ls"], o["qn"] = lb.alloc(4), lb.alloc(4 * d)
            if mo:                  # moe_ffn's input norm (ones: the router's gains are
                o["g_post"], o["gbase"] = lb.alloc(4 * H), lb.alloc(4)      # folded), j * E,
                o["router"] = (lb.alloc(mo.E * H), lb.alloc(4 * mo.E * (H // D)))  # int8
            fp_ = _ffn_pad(ff, fd, D)
            mats = {"wq": (nq, H, fa), "wo": (H, nq, fa), "wg": (fp_, H, fg), "wu": (fp_, H, fg)}
            if P:
                mats.update(wpg=(P, H, fp), wpp=(H, P, fp))
            nkv = spec.kvh(i)
            if own:
                o["kn"] = lb.alloc(4 * d)
                mats.update(wk=(nkv * d, H, fa))
                if not spec.kv_same(i):
                    mats.update(wv=(nkv * d, H, fa))
            for n, (r, c, f) in mats.items():
                o[n] = (lb.alloc(r * rb(c, f)), lb.alloc(4 * r * (c // D)))
            C = self.dchunk[ff, fd]
            o["wd"] = [(lb.alloc(H * rb(C, fd)), lb.alloc(4 * H * (C // D)))
                       for _ in range(fp_ // C)]
            if own:
                ck = self.ring if kind == SLIDE else cap
                o["kv"] = [{"k": lb.alloc(ck * self.ks[kind]), "vt": lb.alloc(d * ck),
                            "vs": lb.alloc(4 * ck)} for _ in range(nkv)]
            o["mats"] = mats
            self.bofs[k], self.bsize[k] = o, (lb.next + 4095) // 4096 * 4096
        self.runs = plan([self._key(i) for i in range(L)])
        # a unit's own repeated part (4 sliding layers of 4 sliding + 1 global) loops again,
        # inside: {run's first layer: [(first element, part, repeats, the part's bytes)]}
        self.subs = {first: [(e0, su, r, sum(self.bsize[k] for k in su))
                             for e0, su, r in plan(unit)] for first, unit, reps in self.runs}
        for first, unit, reps in self.runs:         # a loop's shared layers: one source each
            for e0, su, r, _ in self.subs[first]:
                for e in range(len(su)):
                    ls = [first + it * len(unit) + e0 + j * len(su) + e for it in range(reps)
                          for j in range(r)]
                    src = {spec.kv_src[i] for i in ls}
                    if not su[e][1] and len(src) > 1:
                        raise ValueError(f"layers {ls} share K / V of different layers "
                                         f"{sorted(src)} in one loop")
        self.layer0 = b.next = (b.next + 4095) // 4096 * 4096
        self.loc = {}                   # layer -> (run base, unit stride, iteration, offset)
        for first, unit, reps in self.runs:
            offs = np.cumsum([0] + [self.bsize[k] for k in unit]).tolist()
            us, base = offs[-1], b.next
            for it in range(reps):
                for e in range(len(unit)):
                    self.loc[first + it * len(unit) + e] = (base, us, it, offs[e])
            b.next = base + reps * us
        self.head = (b.alloc(spec.vocab * Q.row_bytes(H, self.head_format, D)),
                     b.alloc(4 * spec.vocab * (H // D)))
        self.ple = b.alloc((rows if self.ple_host else spec.vocab) * self.ple_rec)
        self.ple_store = None           # ple_host: [vocab, ple_rec] uint8, filled by build
        self.lookup = {}
        if lookup:
            fmts = sorted({"int8" if self.head_format == "int8" else "4bit"} |
                          ({"int8" if ple_format == "int8" else "4bit"} if P else set()))
            self.lookup = {"rope_t": b.alloc(cap * self.ps), "iota": b.alloc(4 * block),
                           "onehot": {f: b.alloc(4 * cfg.MCOLS * D * GA.onehot_blocks(
                               D, cfg.MCOLS, "int8" if f == "int8" else "fp4")) for f in fmts},
                           "gen": G.alloc(b, spec, cap, block)}       # the decode loop's area
        self.offload = None
        if mo:                      # path (a)'s words and expert slots (docs/offload.md)
            self.fmt = MO.ExpertFormat(H, mo.ffn, D, self.efmt)
            n = mo.E if experts is None else experts
            if not mo.k <= n <= mo.E:
                raise ValueError(f"{n} expert slots per layer: from top-k {mo.k} to {mo.E}")
            self.offload = Layout.build((b.next + 4095) // 4096 * 4096, mo.E, mo.k, [n] * L,
                                        self.fmt.nbytes)
            b.next = self.offload.end
        self.kv_bytes = sum(self._kv_bytes(i) for i in range(L) if spec.kv_src[i] == i)
        self.nbytes = b.next
        if self.nbytes > cfg.DRAM_BYTES:
            raise MemoryError(f"model image needs {self.nbytes / 2**20:.0f} MiB per slice, "
                              f"DRAM_BYTES is {cfg.DRAM_BYTES / 2**20:.0f} MiB")

    def _key(self, i: int) -> tuple:
        return _key(self.spec, i, self.lf[i])

    def _kv_bytes(self, i: int) -> int:
        d = self.spec.hd(i)
        ck = self.ring if self.spec.kinds[i] == SLIDE else self.cap
        return self.spec.kvh(i) * ck * (self.ks[self.spec.kinds[i]] + d + 4)

    def _off(self, li, it=None) -> Affine:
        """The block address of layer li (static), or of element li of its run's unit at
        iteration `it` (a loop variable)."""
        base, us, i, o = self.loc[li]
        return Affine(base + o) + Affine.of(i if it is None else it) * us

    # ---- contents
    def build(self, W, jobs: int | None = None) -> list[np.ndarray]:
        """The DRAM image with every weight quantized in place, KV cache empty. The matrices
        are quantized by `jobs` worker processes (default: OTPU_BUILD_JOBS, else 4) when W is
        a checkpoint's Weights, the 4-bit ones through opentpu.qcache (_job)."""
        spec, cfg = self.spec, self.cfg
        D, H, P, L = cfg.D, spec.hidden, spec.ple_dim, spec.layers
        self._W = W                                     # host_inputs gathers its rows
        img = np.zeros(self.nbytes, np.uint8)

        def put(addr, a):
            v = np.ascontiguousarray(a).view(np.uint8).reshape(-1)
            img[addr:addr + v.size] = v

        def f32(a):
            return F.ftz(np.asarray(a, np.float32))

        tasks = []          # (DRAM addresses, (kind, tensor, selection, scale, format, ...))
        put(self.io["gf"], f32(W["model.norm.weight"]))
        if P:
            put(self.io["g_pln"], f32(W["model.per_layer_projection_norm.weight"]
                                      * np.float32(2 ** -0.5)))
        put(self.io["z2"], np.concatenate([np.full(self.block, -np.inf, np.float32),
                                           np.full(self.block, np.inf, np.float32)]))
        if P:
            rows = [r for i in range(L)
                    for r in range(spec.src(i) * P, (spec.src(i) + 1) * P)]
            tasks.append((self.wproj, ("mat", "model.per_layer_model_projection.weight",
                                       ("rows", tuple(rows)), H ** -0.5, self.pformat, D)))
        norms = _norms(P, spec.experts > 0)
        for i in range(L):
            p = f"model.layers.{spec.src(i)}."
            a = p + "self_attn."
            k = self._key(i)
            o, base = self.bofs[k], self._off(i).static()
            for n, hf in norms.items():
                put(base + o[n], f32(W[p + hf + ".weight"]))
            put(base + o["ls"], f32(W[p + "layer_scalar"]).reshape(1))
            put(base + o["qn"], f32(W[a + "q_norm.weight"]))
            if spec.experts:        # the router: proj x scale / sqrt(H) (int8), j * E
                put(base + o["g_post"], np.ones(H, np.float32))
                put(base + o["gbase"], np.float32([i * spec.experts]))
                for addr, arr in zip(o["router"], Q.quantize_mxu(MO.gemma_router(W, p),
                                                                 "int8", D)):
                    put(base + addr, arr)
            src = {"wq": a + "q_proj", "wo": a + "o_proj", "wg": p + "mlp.gate_proj",
                   "wu": p + "mlp.up_proj", "wpg": p + "per_layer_input_gate",
                   "wpp": p + "per_layer_projection", "wk": a + "k_proj", "wv": a + "v_proj"}
            if k[1]:
                put(base + o["kn"], f32(W[a + "k_norm.weight"]))
            for n, (r, _, f) in o["mats"].items():     # gate / up: rows padded with zeros
                sel = ("pad", r) if n in ("wg", "wu") and r != spec.ffn[i] else None
                tasks.append((tuple(base + x for x in o[n]),
                              ("mat", src[n] + ".weight", sel, 1.0, f, D)))
            fd = self.lf[i][2]
            C = self.dchunk[k[2], fd]
            for j, pair in enumerate(o["wd"]):
                tasks.append((tuple(base + x for x in pair),
                              ("mat", p + "mlp.down_proj.weight", ("cols", j * C, (j + 1) * C),
                               1.0, fd, D)))
        head = "model.embed_tokens.weight" if spec.tied else "lm_head.weight"
        rb = Q.row_bytes(H, self.head_format, D)
        for r in range(0, spec.vocab, HEAD_CHUNK):
            e = min(spec.vocab, r + HEAD_CHUNK)
            tasks.append(((self.head[0] + r * rb, self.head[1] + r * 4 * (H // D)),
                          ("mat", head, ("rowrange", r, e), 1.0, self.head_format, D)))
        if self.ple_host:                               # the host's store: record t at t rec
            self.ple_store = np.zeros((spec.vocab, self.ple_rec), np.uint8)
        for r in range(0, spec.vocab if P else 0, PLE_CHUNK):
            e = min(spec.vocab, r + PLE_CHUNK)
            tasks.append(((-1 - r if self.ple_host else self.ple + r * self.ple_rec,),
                          ("ple", Weights.PLE, ("rowrange", r, e), tuple(self._cols()),
                           self.ple_format, D, self.ple_S, (P / 2) ** 0.5)))
        for (addrs, _), arrays in zip(tasks, _run_tasks(W, [t for _, t in tasks], jobs)):
            for addr, arr in zip(addrs, arrays):
                if addr < 0:                            # PLE records from row -1 - addr
                    v = np.ascontiguousarray(arr).view(np.uint8).reshape(-1, self.ple_rec)
                    self.ple_store[-1 - addr:-1 - addr + len(v)] = v
                else:
                    put(addr, arr)
        if self.lookup:
            lk = self.lookup
            put(lk["rope_t"], np.concatenate(
                [self.rope_rows(range(self.cap)),
                 np.zeros((self.cap, self.ps // 4 - self.rw), np.float32)], axis=1))
            put(lk["iota"], np.arange(self.block, dtype=np.float32))
            for f, addr in lk["onehot"].items():
                put(addr, GA.onehot(D, cfg.MCOLS, "int8" if f == "int8" else "fp4"))
            G.build(lambda s, addr, a: put(addr, a), 0, 1, spec, self.cap, lk["gen"])
        return [img]

    def _cols(self) -> np.ndarray:
        """The columns of the checkpoint's PLE table that this model's layers read."""
        P = self.spec.ple_dim
        return np.concatenate([np.arange(self.spec.src(i) * P, (self.spec.src(i) + 1) * P)
                               for i in range(self.spec.layers)])

    def _ple_records(self, rows: np.ndarray, cols) -> np.ndarray:
        """PLE rows [n, ple_vocab_width] of the checkpoint -> this image's records: the model's
        layers' columns, x sqrt(ple_dim) (the table's scale) / sqrt(2) (the per-layer inputs'),
        packed for gather_record."""
        return _ple_records(rows, cols, self.ple_format, self.cfg.D, self.ple_S,
                            (self.spec.ple_dim / 2) ** 0.5)

    def rope_rows(self, positions) -> np.ndarray:
        """[n, rw] fp32: per position, cos and sin of the sliding RoPE, then of the global."""
        out = []
        for p in positions:
            cs, ss = rope_tables(self.spec, p, SLIDE)
            cg, sg = rope_tables(self.spec, p, FULL)
            out.append(np.concatenate([cs, ss, cg, sg]))
        return np.array(out, np.float32).reshape(-1, self.rw)

    def host_inputs(self, tokens, positions) -> list:
        """The host-written inputs of a run of token rows (per-position decode, prefill):
        [(address, array)] -- the embedding rows and PLE rows as the device's gathers make
        them (kernels/gather.py, bit for bit), and the RoPE rows."""
        spec, D, W = self.spec, self.cfg.D, self._W
        tokens = [int(t) for t in tokens]
        q, s = Q.quantize_mxu(_rows(W, "model.embed_tokens.weight" if spec.tied else
                                    "lm_head.weight", tokens), self.head_format, D)
        es = [GA.dequant_row(q[i], s[i], self.head_format, D) for i in range(len(tokens))]
        out = [(self.io["x"], np.array(es, np.float32)),
               (self.io["rope"], self.rope_rows(positions))]
        if spec.ple_dim:
            out.append((self.io["pe"], GA.dequant_records(self._ple_records(
                _rows(W, Weights.PLE, tokens), self._cols()), self.ple_format, D, self.ple_S)))
        return out

    def host_rows(self, tokens) -> list:
        """The writes of a run's rows from the tables the host keeps, before the run: with
        ple_host, the tokens' PLE records into the slot (data movement: records ple_store holds
        in the card's format). [] with every table on the card."""
        if not self.ple_host:
            return []
        if len(tokens) > self.rows:
            raise ValueError(f"{len(tokens)} tokens, the PLE slot holds {self.rows}")
        return [(self.ple, self.ple_store[[int(t) for t in tokens]].reshape(-1))]

    # ---- path (a): the expert pool and its server (docs/offload.md)
    def expert(self, W, g: int) -> np.ndarray:
        """Global expert g (layer g // E, expert g % E) in its slot's bytes, the norm gain and
        the per-expert scale folded in (moe.gemma_expert)."""
        E = self.spec.experts
        return self.fmt.pack(*MO.gemma_expert(W, f"model.layers.{self.spec.src(g // E)}.",
                                              g % E))

    def serve(self, W, backend, pool_file=None) -> ExpertServer:
        """The host's expert server on the backend's DRAM (moe.serve)."""
        return MO.serve(self.offload, lambda g: self.expert(W, g), backend, pool_file)

    def row_server(self, backend) -> RowServer | None:
        """ple_host: the host's server of the generate loop's requests (each token's id, posted
        after sampling; the token's record into slot row 0, docs/gemma4_e4b.md), on the
        backend's DRAM; None with the table on the card."""
        if not self.ple_host:
            return None
        return RowServer(BackendDram(backend), RowLayout(self.io["ple_mbox"], self.ple,
                                                         self.ple_rec),
                         lambda t: self.ple_store[t])

    # ---- programs
    def compile_decode(self, blocks: int, lo: int, block: int = ATTN_BLOCK):
        """(programs, run_args): gemma4_step at a run-time position (qwen3.RunPos; the image
        needs lookup=True): the token's rows gathered on the device."""
        from .qwen3 import RunPos
        if not self.lookup:
            raise ValueError("compile_decode needs an image with lookup tables (lookup=True)")
        if block != self.block:
            raise ValueError(f"the image is laid out for attention blocks of {self.block}")
        if not (blocks - 1) * block <= lo < min(blocks * block, self.cap):
            raise ValueError(f"lo {lo} is not in bucket {blocks}")
        rp = RunPos(blocks, block, lo, 0, self.cap)
        b = gemma4_step.trace(self.cfg, 0, {"m": step_descriptors(self, 0), "pos": rp,
                                            "block": block})
        return [b.finish()], list(b.run_args)

    def compile_layer_run(self, li: int, blocks: int, block: int = ATTN_BLOCK, R: int = 1,
                          embedded: bool = False, hint: bool = False, em: bool = False):
        """(programs, run_args): gemma4_layer_run of layer li, R rows at a run-time position of
        bucket `blocks` and a run-time row of the prefill chunk (run arguments: RunPos.values
        and "row"); li < 0: gemma4_embed_run. hint: the run ends with the next layer's hint
        (gemma4_layer_run); em: expert-major (gemma4_layer_run; compile_expert_run). The image
        needs lookup tables and its prefill rows."""
        from ..compiler import RunVar
        from .qwen3 import RunPos
        if not self.lookup or not self.prefill_rows:
            raise ValueError("a layer run needs lookup tables and the image's prefill rows")
        if block != self.block:
            raise ValueError(f"the image is laid out for attention blocks of {self.block}")
        if not 1 <= R <= RUN_ROWS or R * (self.spec.top_k or 1) > LINE // 4:
            raise ValueError(f"{R} rows a layer run")
        if (hint or em) and self.offload is None:
            raise ValueError("a layer run's hint or expert-major MoE needs the expert server's "
                             "words (offload)")
        if hint and em:
            raise ValueError("expert-major layer runs post their own layer's needs, no hints")
        rp = RunPos(blocks, block, (blocks - 1) * block, 0, self.cap)
        rp.tpos.bound -= R - 1          # the run's last row in the block too: tpos <= block - R
        row = RunVar("row", self.prefill_rows)
        if li < 0:
            b = gemma4_embed_run.trace(self.cfg, 0, {"m": self.descriptors(0), "pos": rp,
                                                     "row": row, "em": em})
        else:
            b = gemma4_layer_run.trace(self.cfg, 0, {"m": self.descriptors(0), "li": li,
                                                     "pos": rp, "row": row, "block": block,
                                                     "R": R, "embedded": embedded,
                                                     "hint": hint, "em": em})
        return [b.finish()], list(b.run_args)

    def compile_prefill_head(self, em: bool = False):
        """(programs, run_args): gemma4_prefill_head at a run-time row ("row"); em: from the
        expert-major records, after the last layer's end."""
        from ..compiler import RunVar
        b = gemma4_prefill_head.trace(self.cfg, 0, {"m": self.descriptors(0),
                                                    "row": RunVar("row", self.prefill_rows),
                                                    "em": em})
        return [b.finish()], list(b.run_args)

    def compile_expert_run(self, li: int):
        """(programs, run_args): moe.moe_expert_run of layer li (expert-major, docs/offload.md
        13.11), the chunk's rows and entries (rows x k) run arguments ("rows", "entries")."""
        if self.offload is None or not self.prefill_rows or self.cfg.S != 1:
            raise ValueError("an expert run needs the expert server's words, the image's "
                             "prefill rows and one slice")
        b = gemma4_expert_run.trace(self.cfg, 0, {"m": self.descriptors(0), "li": li})
        return [b.finish()], list(b.run_args)

    def compile_generate(self, blocks: int, lo: int, block: int = ATTN_BLOCK,
                         chain: bool = True, samp=None, debug: bool = False,
                         part: int | None = None) -> list:
        """The decode loop on the device for bucket `blocks` (gemma4_step at its RunPos in it,
        generate.py; the sampler's chunks soft-capped by qwen3._lm_head)."""
        if block != self.block:
            raise ValueError(f"the image is laid out for attention blocks of {self.block}")
        return G.compile_generate(self, gemma4_step, blocks, lo, block, chain, samp, debug, part)

    def compile_step(self, pos: int, block: int = ATTN_BLOCK, tok: int | None = None) -> list:
        """One program: the decode token at position `pos`; its inputs from the host
        (host_inputs), or with `tok` (lookup tables) gathered on the device."""
        return self.compile_rows([(0, pos)], [0], block, None if tok is None else [tok],
                                 fill=getattr(self, "stream_fill", False))

    def compile_rows(self, rows, logit_rows, block: int = ATTN_BLOCK, tokens=None,
                     fill: bool = False) -> list:
        """One program: consecutive positions of the sequence at once; their inputs from the
        host (host_inputs), or with `tokens` (their ids, compiled in; lookup tables) gathered
        and loaded on the device. fill: a streamed decode step's (compile_step: fill_logits
        first)."""
        if self.spec.experts and len(rows) > 1:
            raise ValueError("a MoE model runs one row per program (its MoE block routes one "
                             "token)")
        if len(rows) > self.rows:
            raise ValueError(f"{len(rows)} rows, the image's I/O area holds {self.rows}")
        if any(r != (0, rows[0][1] + i) for i, r in enumerate(rows)):
            raise ValueError("Gemma 4 rows must be consecutive positions of sequence 0")
        if block != self.block:
            raise ValueError(f"the image is laid out for attention blocks of {self.block}")
        kw = {}
        if tokens is not None:
            if not self.lookup:
                raise ValueError("rows with their inputs from the image need lookup tables")
            if len(tokens) != len(rows):
                raise ValueError(f"{len(tokens)} tokens for {len(rows)} rows")
            kw["tokens"] = [int(t) for t in tokens]
        m = self.descriptors(0)
        m.fill = bool(fill)
        return [gemma4_step.trace(self.cfg, 0, {"m": m,
                                                "pos": [p for _, p in rows],
                                                "logit_rows": list(logit_rows),
                                                "block": block, **kw}).finish()]

    def compile_prompt_run(self, blocks: int, R: int, kind: str, block: int = ATTN_BLOCK):
        """gemma4_prompt_run's (programs, run_args) (docs/prefill.md): R rows of a prompt at a
        run-time position of bucket `blocks` (the state's tpos word: qwen3.RunWords), their
        tokens from out[]; with ple_host the PLE slot's rows, which the host writes before the
        run (prompt_host_rows)."""
        from .qwen3 import RunRows
        if not self.lookup:
            raise ValueError("a prompt run needs lookup tables (lookup=True)")
        if self.spec.experts:
            raise ValueError("a MoE model's prompt runs layer-major (prefill_layers)")
        if kind not in ("P", "L"):
            raise ValueError(f"prompt run kind {kind!r}")
        if block != self.block:
            raise ValueError(f"the image is laid out for attention blocks of {self.block}")
        if R > self.rows:
            raise ValueError(f"{R} rows, the image's I/O area holds {self.rows}")
        pos = RunRows(blocks, block, (blocks - 1) * block, 0, self.cap, R, 0)
        b = gemma4_prompt_run.trace(self.cfg, 0, {"m": self.descriptors(0), "pos": pos, "R": R,
                                                  "kind": kind, "block": block})
        return [b.finish()], list(b.run_args)

    @property
    def prompt_host_rows(self) -> bool:
        """A prompt run reads rows the host writes before it (host_rows: ple_host's records)."""
        return bool(self.ple_host)

    # ---- kernel descriptors
    def descriptors(self, sid: int = 0) -> SimpleNamespace:
        spec, cfg = self.spec, self.cfg
        D, H, P, L, R = cfg.D, spec.hidden, spec.ple_dim, spec.layers, self.rows

        def kv(li, off):
            d, o = spec.hd(li), self.bofs[self._key(li)]
            ck = self.ring if spec.kinds[li] == SLIDE else self.cap
            return _KV({j: {n: off + v for n, v in e.items()} for j, e in enumerate(o["kv"])},
                       ck, d, D, self.ks[spec.kinds[li]])

        def layer(li, it=None, jt=None):
            """Descriptors of layer li (static), or of the layer at li's place in its run's
            unit at iteration `it` (a loop variable) and, jt = (loop variable, layers, bytes),
            at iteration jt of the unit's inner loop over a repeated part of it."""
            k = self._key(li)
            kind, own, ff, (_, _, fd, _) = k
            o, off = self.bofs[k], self._off(li, it)
            if jt is not None:
                off = off + Affine.of(jt[0]) * jt[2]
            d = spec.hd(li)
            ns = SimpleNamespace(kind=kind, own=own, hd=d, ffn=ff, nkv=spec.kvh(li),
                                 same=spec.kv_same(li),
                                 **{n: Tensor(off + o[n], (H,), (1,))
                                    for n in _norms(P, spec.experts > 0)})
            ns.ls = Tensor(off + o["ls"], (1,), (1,))
            ns.qn = Tensor(off + o["qn"], (d,), (1,))
            for n, (r, c, f) in o["mats"].items():
                da, sa = o[n]
                setattr(ns, n, QTensor(off + da, off + sa, (r, c), Q.row_bytes(c, f, D),
                                       4 * (c // D), D, wf=Q.mxu_wf(f)))
            C, wf = self.dchunk[ff, fd], Q.mxu_wf(fd)
            rc = Q.row_bytes(C, fd, D)
            parts = tuple(QTensor(off + da, off + sa, (H, C), rc, 4 * (C // D), D, wf=wf)
                          for da, sa in o["wd"])
            ns.wd = QTensor(parts[0].data, parts[0].scale, (H, C * len(parts)), rc,
                            4 * (C // D), D, parts=parts, pw=C, wf=wf)
            if own:
                ns.kn = Tensor(off + o["kn"], (d,), (1,))
                ns.kv = kv(li, off)
            else:
                src = spec.kv_src[li]
                ns.kv = kv(src, self._off(src))
            # this layer's per-layer inputs: row r at pli + (r L + l) P
            lidx = Affine(li) if it is None else \
                Affine(li - self.loc[li][2] * self._unit(li)) + Affine.of(it) * self._unit(li)
            if jt is not None:
                lidx = lidx + Affine.of(jt[0]) * jt[1]
            if P:
                ns.pli = Tensor(Affine(self.io["pli"]) + lidx * (4 * P), (R, P), (L * P, 1))
            if spec.experts:                        # moe_ffn's (lfm2's names)
                E = spec.experts
                ns.g_post = Tensor(off + o["g_post"], (H,), (1,))
                ns.gbase = Tensor(off + o["gbase"], (1,), (1,))
                da, sa = o["router"]
                ns.router = QTensor(off + da, off + sa, (E, H), H, 4 * (H // D), D)
            return ns

        ns = SimpleNamespace(
            spec=spec, layer=layer, runs=self.runs, subs=self.subs, rows=R, block=self.block,
            ring=self.ring, rw=self.rw, S=self.ple_S, ple_format=self.ple_format,
            head_format=self.head_format, ple_host=self.ple_host,
            ple_mbox=self.io.get("ple_mbox"),
            x=_tdesc(self.io["x"], (R, H)), pe=_tdesc(self.io["pe"], (R, self.ple_S * D)),
            rope=_tdesc(self.io["rope"], (R, self.rw)), g_final=_tdesc(self.io["gf"], (H,)),
            g_pln=_tdesc(self.io["g_pln"], (P,)),
            logits=_tdesc(self.io["logits"], (1, spec.vocab)),
            logitsr=_tdesc(self.io["logits"], (R, spec.vocab)),
            pli=_tdesc(self.io["pli"], (R, L, P)),
            mask=_tdesc(self.io["mask"], (2, self.block)), z2=self.io["z2"],
            xbuf=_tdesc(self.io["xbuf"], (self.prefill_rows, H)) if self.prefill_rows else None,
            wproj=_qdesc(*self.wproj, L * P, H, D, self.pformat) if P else None,
            head=_qdesc(*self.head, spec.vocab, H, D, self.head_format), v_loc=spec.vocab,
            ple=QTensor(Affine(self.ple), Affine(self.ple + self.ple_S * (
                D if self.ple_format == "int8" else D // 2)),
                (R if self.ple_host else spec.vocab, self.ple_S * D),
                self.ple_rec, self.ple_rec, D, wf=Q.mxu_wf(self.ple_format)) if P else None)
        if self.lookup:
            lk = self.lookup
            ns.rope_t = Tensor(Affine(lk["rope_t"]), (self.cap, self.rw), (self.ps // 4, 1))
            ns.iota = _tdesc(lk["iota"], (self.block,))
            ns.onehot = {f: _tdesc(a, (cfg.MCOLS, D * GA.onehot_blocks(
                D, cfg.MCOLS, "int8" if f == "int8" else "fp4"))) for f, a in lk["onehot"].items()}
            ns.gen = G.desc(lk["gen"], spec, self.cap)
        # the generate loop's post of the sampled token (generate.py): ple_host, a request for
        # its record, which the next token's gather waits for (_gathered)
        mbox = self.io.get("ple_mbox")
        ns.post_token = None if mbox is None else (lambda tok: MB.post(mbox, tok))
        ns.moe_dev = None
        if self.offload is not None:
            Lo = self.offload
            ns.moe_dev = SimpleNamespace(mbox=Lo.mbox, served=Lo.served, answer=Lo.answer,
                                         dir=Lo.dir, tag=Lo.tag, fmt=self.fmt,
                                         hint_off=Lo.layers * Lo.E,
                                         scratch=self.io.get("moe_scratch"),
                                         need_off=2 * Lo.layers * Lo.E,
                                         em_base=Lo.slots[0][0],
                                         em_rec=MO.em_record(spec.hidden, spec.top_k),
                                         em_rows=self.prefill_rows)
        return ns

    def _unit(self, li) -> int:
        """The number of layers in the unit of li's run."""
        for first, unit, reps in self.runs:
            if first <= li < first + len(unit) * reps:
                return len(unit)
        raise KeyError(li)


def _ple_records(rows, cols, fmt: str, D: int, S: int, scale: float) -> np.ndarray:
    x = np.asarray(rows, np.float32)[:, np.asarray(cols)] * np.float32(scale)
    return GA.pack_records(x, fmt, D, S)


# ---- Image.build's quantization jobs: in worker processes, cached by content
HEAD_CHUNK = 32768      # LM head rows quantized per job
_JOB_W = None           # the worker process's Weights


def _job_init(model_dir) -> None:
    global _JOB_W
    _JOB_W = Weights(model_dir)


def _job_matrix(W, name, sel, scale) -> np.ndarray:
    if sel is None:
        a = W[name]
    elif sel[0] == "rowrange":
        if isinstance(W, Weights):
            f, n = W._files[name]
            a = _Rows(f, n)[sel[1]:sel[2]]
        else:
            a = np.asarray(W[name], np.float32)[sel[1]:sel[2]]
    elif sel[0] == "rows":
        a = np.asarray(W[name], np.float32)[list(sel[1])]
    elif sel[0] == "pad":                   # the first sel[1] rows, zeros past the tensor's
        a = np.asarray(W[name], np.float32)
        a = np.pad(a, ((0, sel[1] - len(a)), (0, 0)))
    else:                                   # columns, zeros past the tensor's
        a = np.asarray(W[name], np.float32)[:, sel[1]:sel[2]]
        a = np.pad(a, ((0, 0), (0, sel[2] - sel[1] - a.shape[1])))
    a = np.asarray(a, np.float32)
    return a if scale == 1.0 else a * np.float32(scale)


def _job(W, job) -> tuple:
    """One job: ("mat", tensor, selection, scale, format, D) -> the MXU rows and scale words
    (4-bit through opentpu.qcache, the image caches' disk cache of the quantizer's results);
    ("ple", tensor, row range, columns, format, D, S, scale) -> the PLE records."""
    if job[0] == "mat":
        _, name, sel, scale, fmt, D = job
        return QC.quantize_mxu(_job_matrix(W, name, sel, scale), fmt, D)
    _, name, sel, cols, fmt, D, S, scale = job
    return (_ple_records(_job_matrix(W, name, sel, 1.0), cols, fmt, D, S, scale),)


def _worker_job(job) -> tuple:
    """_job in a worker process, with the opentpu.qcache counts it made (the parent adds them
    to its own: prebuild prints them)."""
    s0 = dict(QC.stats)
    out = _job(_JOB_W, job)
    return out, {k: QC.stats[k] - s0[k] for k in s0}


def _run_tasks(W, jobs: list, n: int | None = None):
    """Results of the jobs, in order: in n worker processes when W is a checkpoint's Weights,
    else in line."""
    import os
    n = int(os.environ.get("OTPU_BUILD_JOBS", 4)) if n is None else n
    if not isinstance(W, Weights) or n <= 1:
        for j in jobs:
            yield _job(W, j)
        return
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(n, mp_context=mp.get_context("spawn"), initializer=_job_init,
                             initargs=(str(W.model_dir),)) as pool:
        for out, counts in pool.map(_worker_job, jobs):
            for k, v in counts.items():
                QC.stats[k] += v
            yield out


# =============================================================================== kernel
def _rope(x, c, s, out=None):
    """RoPE of the rows of x [n, d] at one position (c, s: [nr] tiles) on the pairs (i, i +
    d/2), i < nr, the others passing through (Gemma 4's proportional RoPE); into `out`, else a
    new tile."""
    d, nr = x.cols, c.cols
    h = d // 2
    if nr == h:
        return rope(x, c, s, out)
    out = ol.empty(x.shape) if out is None else out
    a, b = x[:, 0:nr], x[:, h:h + nr]
    out[:, 0:nr].set(a * c[None, :] - b * s[None, :])
    out[:, h:h + nr].set(b * c[None, :] + a * s[None, :])
    out[:, nr:h].set(x[:, nr:h])
    out[:, h + nr:d].set(x[:, h + nr:d])
    return out


def _slide_seq(m, p, block: int):
    """The sliding window of position p (an int, or a RunPos) over the KV ring (ring = window
    + RING_BLOCKS blocks, position q in slot q mod ring): an int length before the ring wraps
    and the window is full, else attention.Blocks -- the window's first block masked at its
    start, its last at its end (at a static p its end is a partial block; at a run-time one
    both masks are the rows of m.mask the step computes)."""
    from .qwen3 import RunPos
    nw, nr = m.spec.window // block, m.ring // block
    run = isinstance(p, RunPos)
    mb = m.mask.base + 4 * 2 * block * getattr(p, "row", 0)  # its mask pair (a run's row)
    B = p.blocks if run else p // block + 1         # the position's block + 1
    if B <= nw:                                     # the window reaches position 0
        if not run:
            return p + 1
        return Blocks([(i * block, block, None) for i in range(B - 1)] +
                      [((B - 1) * block, block, Tensor(mb, (block,), (1,)))])
    t = None if run else p % block
    start = Tensor(mb + 4 * block, (block,), (1,)) if run else \
        Tensor(Affine(m.z2 + 4 * (block - 1 - t)), (block,), (1,))
    items = [(((B - 1 - nw) % nr) * block, block, start)]
    items += [(((B - 1 - nw + j) % nr) * block, block, None) for j in range(1, nw)]
    e = ((B - 1) % nr) * block
    items.append((e, block, Tensor(mb, (block,), (1,))) if run else (e, t + 1, None))
    return Blocks(items)


def _full_seq(m, p, block: int):
    """A global layer's attention span at position p: p + 1 tokens, or at a run-time position
    qwen3's bucket, its last block masked by the step's end-mask row."""
    from .qwen3 import RunPos
    if isinstance(p, RunPos):
        return Bucket(p.blocks, m.mask.base + 4 * 2 * block * getattr(p, "row", 0)
                      - 4 * (p.blocks - 1) * block)
    return p + 1


def _slot(m, kind, p):
    """The cache slot of position p (an int or a RunPos): the ring slot p mod ring (sliding)."""
    from .qwen3 import RunPos
    if isinstance(p, RunPos):
        if kind == SLIDE:
            return Affine(p.t0 % m.ring) + p.tpos
        return p.pos
    return p % m.ring if kind == SLIDE else p


def _attention(x, lw, m, pos, ropes, block: int):
    """x + norm(W_o attention(norm(x))) for the rows of x (row r at position pos[r]; one row at
    a RunPos). Own K / V: projected, normed, RoPE'd (K) and appended row by row; the KV-shared
    layers read their source's cache. The query group of each KV head attends in parts of
    MCOLS heads, every (row, part) one entry of the pipelined flash attention. K = V (lw.same):
    V is the K projection, normed without a weight (K: with k_norm, then RoPE'd)."""
    spec = m.spec
    R, d, nkv, eps = x.rows, lw.hd, lw.nkv, spec.eps
    G = spec.n_q // nkv
    c0, nr = (0, spec.head_dim // 2) if lw.kind == SLIDE else (spec.head_dim, spec.global_rot)

    def cos_sin(r):
        """Row r's cos and sin (a RoPE row: sliding cos, sin, then global cos, sin)."""
        return ropes[r, c0:c0 + nr], ropes[r, c0 + nr:c0 + 2 * nr]

    kv = lw.kv
    xs = ol.quantize(rmsnorm(x, ol.load(lw.g_in), eps))
    q = ol.dot(xs, lw.wq)                                   # [R, nq*d]
    if lw.own:
        k = ol.dot(xs, lw.wk)                               # [R, nkv*d]
        v = k if lw.same else ol.dot(xs, lw.wv)
        kn = ol.load(lw.kn)
        for r in range(R):
            c, s_ = cos_sin(r)
            kr = _rope(rmsnorm(k[r, :].reshape(nkv, d), kn, eps), c, s_)
            vr = rmsnorm(v[r, :].reshape(nkv, d), None, eps)
            slot = _slot(m, lw.kind, pos[r] if isinstance(pos, list) else pos)
            for j in range(nkv):
                ol.kv_append(kv, j, slot, kr[j:j + 1, :], vr[j:j + 1, :])
        del k, v, kr, vr, kn
    del xs
    qn = ol.load(lw.qn)
    mc = min(G, ol.mxu_columns())
    # the queries normed and RoPE'd in place, then in place the attention output: an entry's
    # output rows are its own queries' (quantized into ACT RAM when its attention starts)
    o = q
    Qs, ent = [], []
    for r in range(R):
        c, s_ = cos_sin(r)
        qr = o[r, :].reshape(spec.n_q, d)
        Qs.append(_rope(rmsnorm(qr, qn, eps), c, s_, qr))
        pr = pos[r] if isinstance(pos, list) else pos
        seq = (_slide_seq if lw.kind == SLIDE else _full_seq)(m, pr, block)
        ent += [(r, j, g0, min(G, g0 + mc), seq) for j in range(nkv)
                for g0 in range(0, G, mc)]
    del q, qr

    def emit(i, acc, l, o=o):                               # (o is freed below)
        r, j, g0, g1, _ = ent[i]
        o[r, (j * G + g0) * d:(j * G + g1) * d].reshape(g1 - g0, d).set(acc / l[:, None])

    _attend_heads([Qs[r][j * G + g0:j * G + g1, :] for r, j, g0, g1, _ in ent], kv,
                  [j for _, j, _, _, _ in ent], [e[4] for e in ent], block, ol.LOG2E,
                  depth=ATTN_DEPTH, emit=emit)
    del Qs
    y = ol.dot(o, lw.wo)                                    # [R, H]
    del o
    _add_norm(x, y, ol.load(lw.g_attn), eps)


def _add_norm(x, y, g, eps, scale=None):
    """x += norm(y) in place, then x *= scale (a [1] tile) if given. Rows of more than an
    eighth of TMEM (4 prefill rows of E4B's hidden 2560) a row at a time, so the temporaries
    are one row's; smaller ones as one tile (fewer instructions)."""
    rows = [slice(0, x.rows)] if x.rows * x.cols <= ol.tmem_words() // 8 else \
        [slice(r, r + 1) for r in range(x.rows)]
    for sl in rows:
        if scale is None:
            x[sl, :].set(x[sl, :] + rmsnorm(y[sl, :], g, eps))
        else:
            x[sl, :].set((x[sl, :] + rmsnorm(y[sl, :], g, eps)) * scale)


def _dense(x, lw, spec):
    """The dense MLP of norm(x)."""
    xs = ol.quantize(rmsnorm(x, ol.load(lw.g_pre), spec.eps))
    return swiglu_down(xs, lw.wg, lw.wu, lw.wd, chunk=lw.wd.pw, act=gelu_tanh)


def _mlp(x, lw, spec, m=None):
    """x + norm(MLP(norm(x))); with the MoE block (one row), x + norm(norm_1(MLP(norm(x))) +
    norm_2(MoE(x))), the MLP emitted beside the MoE's request (while the host streams its
    experts); without per-layer inputs, then x layer_scalar."""
    def dense():
        return _dense(x, lw, spec)

    if spec.experts:
        out = {}

        def beside():
            out["y"] = dense()

        if x.rows == 1:
            acc = MO.moe_ffn(x, lw, spec.moe, m.moe_dev, spec.eps, beside=beside,
                             residual=False, y_first=True)   # (26B: [8, 2816] in a fragmented
        else:                                                # TMEM)
            acc = MO.moe_ffn_rows(x, lw, spec.moe, m.moe_dev, spec.eps, beside=beside,
                                  residual=False)
        y = rmsnorm(out.pop("y"), ol.load(lw.g_f1), spec.eps) + \
            rmsnorm(acc, ol.load(lw.g_f2), spec.eps)
        del acc
    else:
        y = dense()
    _add_norm(x, y, ol.load(lw.g_ffn), spec.eps, None if spec.ple_dim else ol.load(lw.ls))


def _em_mlp(x, lw, spec, m, row):
    """Expert-major's MoE layer (docs/offload.md 13.11), its run's part: the residual rows x
    to their records, then the MoE's prologue with the dense MLP beside, its output normed
    (norm_1) to the records' SH; the layer ends in the next run (_em_end)."""
    ol.store(MO.em_x(m.moe_dev, row, x.rows, spec.hidden), x)
    MO.moe_prologue_rows(x, lw, spec.moe, m.moe_dev, spec.eps, row,
                         beside=lambda: rmsnorm(_dense(x, lw, spec), ol.load(lw.g_f1), spec.eps))


def _em_end(x, lw, spec, m, row):
    """The end of _em_mlp's layer (lw) for the rows x, after its expert run: x +
    norm(SH + norm_2(the experts' sum)), then x layer_scalar, as _mlp's."""
    acc = MO.moe_combine_rows(x, spec.moe, m.moe_dev, row, residual=False)
    sh = ol.load(Tensor(MO.em_row(m.moe_dev, row) + 4 * spec.hidden * (spec.top_k + 1),
                        (x.rows, spec.hidden), (m.moe_dev.em_rec // 4, 1)))
    y = sh + rmsnorm(acc, ol.load(lw.g_f2), spec.eps)
    del acc, sh
    _add_norm(x, y, ol.load(lw.g_ffn), spec.eps, None if spec.ple_dim else ol.load(lw.ls))


def _ple(x, lw, spec):
    """The per-layer input: x + norm(W_p (gelu_tanh(W_g x) * pli)), then x layer_scalar."""
    g = gelu_tanh(ol.dot(x, lw.wpg)) * ol.load(lw.pli[0:x.rows, :])     # [R, P]
    y = ol.dot(g, lw.wpp)                                   # [R, H]
    del g
    _add_norm(x, y, ol.load(lw.g_ple), spec.eps, ol.load(lw.ls))


def _ple_inputs(m, x, pe=None):
    """Every layer's per-layer input of the rows of x ([R, H], x sqrt(H)): norm(W_proj x) + pe
    per layer (the 1 / sqrt(H) and 1 / sqrt(2) folded into W_proj, the norm weight and the
    table) -> m.pli [R, L, P]. pe: the gathered PLE row [1, S D] (a run-time position), or
    None: the host-written rows m.pe, loaded a group of layers at a time."""
    spec = m.spec
    R, P, L = x.rows, spec.ple_dim, spec.layers
    xs = ol.quantize(x)
    g = ol.load(m.g_pln)
    n = max(1, min(L, (ol.tmem_words() // 8) // (R * P)))  # layers per projection MM
    for l0 in range(0, L, n):
        l1 = min(L, l0 + n)
        pp = ol.dot(xs, m.wproj[l0 * P:l1 * P, :])          # [R, (l1-l0) P]
        for r in range(R):                                  # a row's layers at once
            e = pe[0, l0 * P:l1 * P] if pe is not None else ol.load(m.pe[r, l0 * P:l1 * P])
            v = rmsnorm((pp[r, :] if R > 1 else pp).reshape(l1 - l0, P), g, spec.eps) + \
                e.reshape(l1 - l0, P)
            ol.store(m.pli[r, l0:l1, :], v)


def _gathered(m, pos):
    """At a run-time position: the token's embedding row [1, H] and PLE row [1, S D], gathered
    from the LM head and the PLE records (kernels/gather.py), its RoPE row [1, rw], and the
    position's mask rows -> m.mask (end: +inf where the block's token <= tpos; start: the
    opposite)."""
    spec = m.spec
    H, D = spec.hidden, ol.block_size()
    ops = {}

    def op(fmt):
        f = "int8" if fmt == "int8" else "4bit"
        if f not in ops:
            ops.clear()                                     # one operand at a time (TMEM)
            ops[f] = ol.quantize(ol.load(m.onehot[f]))
        return ops[f]

    e = GA.gather_row(op(m.head_format), m.head, pos.tok, m.head_format)
    if m.ple_host:          # the slot's row 0 holds the token's record once the host has
        MB.wait_served(m.ple_mbox)      # served every request (the generate loop's post)
    pe = None if m.ple is None else \
        GA.gather_record(op(m.ple_format), m.ple, 0 if m.ple_host else pos.tok, m.ple_format,
                         m.S).reshape(1, m.S * D)       # ple_host: the slot's row
    ops.clear()
    return e.reshape(1, H), pe, _pos_rows(m, pos)


def _pos_rows(m, pos, R: int = 1):
    """At a run-time position: the RoPE rows [R, rw] of it and the R - 1 after it (in its
    attention block), and each one's mask rows -> its pair of m.mask's rows (end: +inf where
    the block's token <= tpos; start: the opposite)."""
    ropes = ol.load(m.rope_t[pos.pos:pos.pos + R, :])       # [R, rw]
    for r in range(R):
        tp = ol.load(m.iota[pos.tpos + r:pos.tpos + r + 1])     # [1]: tpos + r as a float
        end = ((tp - ol.load(m.iota)) + 0.5) * BIG * BIG    # +inf where c <= tpos + r
        mb = m.mask.base + 4 * 2 * m.block * r
        ol.store(Tensor(mb, (m.block,), (1,)), end)
        ol.store(Tensor(mb + 4 * m.block, (m.block,), (1,)), end * -1.0)
        del end
    return ropes


class _RowPos(RunPos):
    """Row r of a layer run of R rows at a run-time position (qwen3.RunPos): position pos + r,
    in the same attention block; its mask rows are m.mask's pair r (_pos_rows)."""

    def __init__(self, rp: RunPos, r: int):     # (not RunPos's: the same run-time values)
        self.blocks, self.block, self.lo, self.t0 = rp.blocks, rp.block, rp.lo, rp.t0
        self.tpos, self.pos, self.bucket, self.row = rp.tpos + r, rp.pos + r, None, r


def _gathered_rows(m, tokens):
    """The embedding rows [R, H] of compile-time tokens, gathered on the device from the LM
    head; their PLE rows gathered into m.pe (DRAM: _ple_inputs loads them a group of layers at
    a time, as the host-written ones)."""
    H = m.spec.hidden
    e = ol.empty([len(tokens), H])
    for fmt, tab in ((m.head_format, "head"), (m.ple_format, "ple"))[:2 if m.ple else 1]:
        f = "int8" if fmt == "int8" else "4bit"
        oh = ol.quantize(ol.load(m.onehot[f]))
        for r, t in enumerate(tokens):
            if tab == "head":
                e[r, :].set(GA.gather_row(oh, m.head, t, fmt))
            else:
                ol.store(m.pe[r, :], GA.gather_record(oh, m.ple, r if m.ple_host else t, fmt,
                                                      m.S))
        del oh
    return e


def _gathered_out(m, pos):
    """_gathered_rows of a prompt run (docs/prefill.md; pos a qwen3.RunRows): the tokens of
    out[pos .. pos + R) (the generate area's), each one's table rows at an address from
    scratch registers (RLD MUL of its word, as qwen3._embed_word), not run-time arguments;
    with ple_host the PLE slot's row r, which the host writes before the run."""
    b, H = current(), m.spec.hidden
    tk = ol.load(m.gen.out[pos.pos:pos.pos + pos.R])
    e = ol.empty([pos.R, H])
    for fmt, tab in ((m.head_format, m.head), (m.ple_format, m.ple))[:2 if m.ple else 1]:
        oh = ol.quantize(ol.load(m.onehot["int8" if fmt == "int8" else "4bit"]))
        for r in range(pos.R):
            if tab is m.ple and m.ple_host:
                ol.store(m.pe[r, :], GA.gather_record(oh, tab, r, fmt, m.S))
                continue
            t = tk[r:r + 1]
            b.check_live(t)
            rd, rs = b.scratch(), b.scratch()
            b.emit(I.rld(rd, t.base, mul=tab.rs, comment="the token's row"))
            b.emit(I.rld(rs, t.base, mul=tab.srs, comment="its scales"))
            one = QTensor(Affine.of(tab.data) + DevVar("token row", rd),
                          Affine.of(tab.scale) + DevVar("token scales", rs), (1, tab.shape[1]),
                          tab.rs, tab.srs, tab.D, wf=tab.wf)
            if tab is m.head:
                e[r, :].set(GA.gather_row(oh, one, 0, fmt))
            else:
                ol.store(m.pe[r, :], GA.gather_record(oh, one, 0, fmt, m.S))
            b.unscratch(rd)
            b.unscratch(rs)
        del oh
    del tk
    return e


@ol.jit
def gemma4_step(m, pos, logit_rows=(0,), block: int = ATTN_BLOCK, tokens=None):
    """Token rows at consecutive positions `pos` (a list), or one decode token at a RunPos
    (its rows gathered on the device): the per-layer inputs, the layers (a hardware loop per
    repeated unit of the plan), the final norm and the LM head of the rows in `logit_rows`
    (a contiguous range; empty: none, a prefill chunk before the last). The rows' inputs: with
    `tokens` (compile-time ids) gathered on the device (_gathered_rows), else the host's (rows
    of m.x, m.pe, m.rope). pos a RunRows: a prompt run's rows (gemma4_prompt_run)."""
    from .qwen3 import RunPos, RunRows
    fill_logits(m)
    spec = m.spec
    if isinstance(pos, RunRows):
        R = pos.R
        e, pe = _gathered_out(m, pos), None
        ropes = _pos_rows(m, pos, R)
        pos = [_RowPos(pos, r) for r in range(R)]
    elif isinstance(pos, RunPos):
        e, pe, ropes = _gathered(m, pos)
        R = 1
    elif tokens is not None:
        R = len(pos)
        e, pe = _gathered_rows(m, tokens), None
        ropes = ol.load(m.rope_t[pos[0]:pos[0] + R, :])
    else:
        R = len(pos)
        e, pe, ropes = ol.load(m.x[0:R, :]), None, ol.load(m.rope[0:R, :])
    x = e * math.sqrt(spec.hidden)
    del e
    if spec.ple_dim:
        _ple_inputs(m, x, pe)
    del pe

    def layer(li, it=None, jt=None):
        lw = m.layer(li, it, jt)
        _attention(x, lw, m, pos, ropes, block)             # each adds to x in place
        _mlp(x, lw, spec, m)
        if spec.ple_dim:
            _ple(x, lw, spec)

    def unit(first, it=None):
        for e0, su, r, nb in m.subs[first]:
            if r == 1:
                for e_ in range(len(su)):
                    layer(first + e0 + e_, it)
                continue
            for jt in ol.range(r):                          # the unit's repeated part
                for e_ in range(len(su)):
                    layer(first + e0 + e_, it, (jt, len(su), nb))

    for first, _, reps in m.runs:
        if reps == 1:
            unit(first)
            continue
        for it in ol.range(reps):
            unit(first, it)
    if isinstance(pos, RunPos):
        _lm_head(x, m, spec)            # a sampler's sink gets the capped logits (spec.softcap)
    elif logit_rows:
        _lm_head_rows(x, m, spec, list(logit_rows))


@ol.jit
def gemma4_prompt_run(m, pos, R: int, kind: str, block: int = ATTN_BLOCK):
    """A prompt run of R rows (docs/prefill.md), gemma4_step's rows with their tokens from
    out[]: kind "P", or "L" (the prompt's last run: its last row's logits). pos: a RunRows
    (toks_at 0) at the run-time position in the generate state's tpos word (RunWords)."""
    from .qwen3 import RunWords
    with RunWords(m, pos):
        gemma4_step.fn(m, pos, [R - 1] if kind == "L" else [], block)


@ol.jit
def gemma4_layer_run(m, li: int, pos, row, block: int = ATTN_BLOCK, R: int = 1,
                     embedded: bool = False, hint: bool = False, em: bool = False):
    """Layer-major prefill (a MoE model's: the whole prompt chunk through a layer before the
    next, so that the expert cache serves one layer at a time): the R prompt rows from `row`
    (a run-time value) at the run-time positions pos .. pos + R - 1 (qwen3.RunPos; in one
    attention block) through layer li alone. Their input is the chunk's residual stream rows
    m.xbuf[row:row + R] (layer 0 of a one-row run unless `embedded`: the token's embedding row,
    gathered at pos.tok; else gemma4_embed_run's), their output goes back there; the MoE block
    of more than one row is moe.moe_ffn_rows. A row's arithmetic is gemma4_step's at its
    RunPos, so the chunk layer by layer leaves the KV cache and residual rows the per-position
    programs make, bit for bit. With `hint` (a MoE model, li + 1 a layer) the run ends with
    layer li + 1's hint: its router on the output rows (moe.moe_hint_rows; docs/offload.md
    13.9), which changes no row. em: expert-major (docs/offload.md 13.11): the rows in the
    scratch's records (moe.em_x), the layer before ended first (_em_end: its experts' outputs
    from its expert run, with the dense MLP's), and the MoE layer's run ends with the prologue
    (_em_mlp), its experts left to the layer's expert run (gemma4_expert_run)."""
    spec = m.spec
    if spec.ple_dim:
        raise ValueError("layer-major prefill: a model without per-layer inputs")
    if R == 1 and li == 0 and not embedded:
        e, _, ropes = _gathered(m, pos)
        x = e * math.sqrt(spec.hidden)
        del e
    else:
        ropes = _pos_rows(m, pos, R)
        x = ol.load(MO.em_x(m.moe_dev, row, R, spec.hidden) if em else m.xbuf[row:row + R, :])
    if em and li > 0:               # (expert-major: the layer before ends here, _em_end)
        _em_end(x, m.layer(li - 1), spec, m, row)
    lw = m.layer(li)
    _attention(x, lw, m, pos if R == 1 else [_RowPos(pos, r) for r in range(R)], ropes, block)
    if em:
        _em_mlp(x, lw, spec, m, row)
        return
    _mlp(x, lw, spec, m)
    ol.store(m.xbuf[row:row + R, :], x)
    if hint and spec.experts and li + 1 < spec.layers:
        MO.moe_hint_rows(x, m.layer(li + 1), spec.moe, m.moe_dev, spec.eps)


@ol.jit
def gemma4_embed_run(m, pos, row, em: bool = False):
    """A layer-major prefill's input row (runs of more than one row): the token's embedding row
    (gathered at pos.tok) times sqrt(H) -> m.xbuf[row] (em: its record's X), as
    gemma4_step's."""
    e, _, _ = _gathered(m, pos)
    X = MO.em_x(m.moe_dev, row, 1, m.spec.hidden) if em else m.xbuf[row:row + 1, :]
    ol.store(X, e * math.sqrt(m.spec.hidden))


@ol.jit
def gemma4_prefill_head(m, row, em: bool = False):
    """After a layer-major prefill's last layer: the final norm and the LM head of the residual
    stream row m.xbuf[row] (a run-time value) -> m.logits, as gemma4_step's. em: the row's
    record's X, after the last layer's end (_em_end)."""
    spec = m.spec
    if not em:
        _lm_head(ol.load(m.xbuf[row:row + 1, :]), m, spec)
        return
    x = ol.load(MO.em_x(m.moe_dev, row, 1, spec.hidden))
    _em_end(x, m.layer(spec.layers - 1), spec, m, row)
    _lm_head(x, m, spec)


@ol.jit
def gemma4_expert_run(m, li: int):
    """Expert-major's expert run of layer li (moe.moe_expert_run, docs/offload.md 13.11): after
    the layer's runs over a prefill chunk, each expert its rows chose once, in passes of two
    rows; the chunk's rows and their entries (rows x k) are run arguments."""
    from ..compiler import RunVar
    C, k = m.moe_dev.em_rows, m.spec.top_k
    MO.moe_expert_run(m.layer(li), m.spec.moe, m.moe_dev, RunVar("rows", C + 1),
                      RunVar("entries", C * k + 1))
