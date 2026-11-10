"""GPT-OSS (OpenAI gpt-oss-20b, Hugging Face model_type gpt_oss) on openTPU: decode, on one slice
or S (tensor-parallel: below).

Every layer is attention then a mixture of experts, both pre-norm (RMSNorm, fp32) with residuals:
  attention  GQA (64 q / 8 kv heads of 64: LFM2's padded path, q and K zero-padded to D) with
             biases on q, k, v and o, YaRN RoPE (rope_tables' rope_div / rope_scale: llm.yarn),
             and a learned sink logit per q head: the softmax over a head's scores and its sink,
             the sink's share dropped (the flash state starts at m = sink, l = 1:
             kernels.attention._Head). Layers alternate sliding (the last `window` = 128
             positions, the first layer sliding) and full attention.
  MoE        router linear + bias over E = 32 experts, the top k = 4, softmax over their logits;
             each expert gate_up (interleaved: even rows gate, odd up) with biases, g = min(g,
             7), u = clip(u, -7, 7), (u + 1) g sigmoid(1.702 g), down with its bias (moe.py:
             MoESpec act "oai", bias; moe.oai_glu).
The hidden size 2880 is 22.5 MXU blocks: the image pads every hidden-wide vector and matrix side
with zeros to Hp = 2944 (23 x 128), the norms average over the 2880 (rmsnorm's n), so the padding
stays zero through the residual and changes nothing.

The sliding layers keep K / V in a ring of one attention block (RING = 256 slots, position q in
slot q mod 256), attended as one block with a mask row from a table in the image (smask): at
position p the slots of positions p - window + 1 .. p count, the others (later laps, empty slots)
score -inf; one table for the first block of positions (no lap before), one for the later ones.
The full layers attend over the KV cache's buckets (qwen3.RunPos).

At S > 1 slices each slice holds n_kv / S KV heads (s, s + S, ...: KVDesc's) with their query
groups, their rows of W_q, W_k, W_v and their biases and sinks (qwen3's head-parallel order:
kernels.layouts), W_o's rows of its hidden / S output columns at the whole K (the heads'
outputs all-gathered first, in that order: their QACT range S = 1's), its rows of the router
(zero rows to whole groups), its rows of every expert (moe.ExpertFormat S parts in a resident
pool: moe._moe_ffn_tp) and its vocab / S rows of the LM head; x's hidden columns are
all-gathered after the attention and after the experts (hidden / S a slice, at x's first
columns: its padding stays zero). Every value is S = 1's: the logits are S = 1's bit for bit.

The experts come as MXFP4 (blocks of 32 E2M1 codes with an E8M0 scale): quant.import_mxfp4 puts
each matrix into the slot's fp4 (or nvfp4) exactly when its K-blocks' exponents fit (docs/
quant.md); a matrix that does not fit is requantized from its values (expert_mats, counted in
Image.imports). The attention projections are int8 (Spec.formats "attn=int8": in fp4 the model
drifts from float with the context) unless a formats string says otherwise (KINDS: "attn" all
four, "wq" / "wk" / "wv" / "wo" each one; the caller's, else OTPU_FORMATS, else Spec.formats:
opentpu/llm/formats.py; a kind none names is in the image's wformat), the router int8, the LM
head head_format (int8 by default). The embedding table is fp32 (a row loaded at the token
id; Spec.embed "int8": the MXU's int8 rows gathered on the device, a quarter of the bytes, but
GPT-OSS's rows' outliers lose 2.1-2.5% of a row's norm to it: on the 2-layer cut the emulation's
largest error, 1 - cos 8e-3 against the reference where the weights' and head's formats give
2.3e-3).

Pieces:
  Spec              the model's dimensions (from a Hugging Face config.json)
  load_weights      a checkpoint's tensors (bf16 -> fp32, the experts' MXFP4 blocks raw)
  reference_logits  plain numpy forward pass (fp32), Hugging Face's math
  emulated_logits   float64 decode with openTPU's quantization points (as qwen3's)
  Image             one slice's DRAM: the layer blocks (a hardware loop over (sliding, full)
                    pairs), the LM head, the lookup tables and the expert slots (path (a),
                    docs/offload.md: every expert resident at experts = E)
  gptoss_step       the ol kernel of one decode token (qwen3.Engine runs it: resident decode,
                    compile_decode, and the generate loop, compile_generate)
"""
from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .. import fp32 as FP
from .. import isa as I
from .. import qcache as QC
from .. import quant as Q
from .. import language as ol
from ..compiler import (Affine, KVDesc, QTensor, Tensor, current, gall_wg, k_scale_bytes,
                        trace_fit)
from ..isasim import Config
from ..kernels.attention import Blocks, _attend_heads
from ..kernels.lib import rmsnorm
from ..host.offload import ExpertServer, Layout
from . import formats as FM
from . import generate as G
from . import moe as MO
from . import rope_parameters, yarn
from .lfm2 import _place, plan, run_layers
from .qwen3 import (ATTN_BLOCK, ATTN_DEPTH, HIGH, EmbedHost, RunPos, _align, _Bump, _fake_k,
                    _fake_q, _inputs, _lm_head, _lookup_alloc, _lookup_build, _lookup_desc,
                    _formats, _padded, _qk, _rope_padded, _tdesc, _tok_arg, _v_parts,
                    compile_decode, fill_logits, finite, fit_high, gathered_rows, high_base,
                    low_check, rope_tables, step_descriptors)

SLIDE, FULL = "slide", "full"
RING = ATTN_BLOCK           # a sliding layer's K / V ring: one attention block
Q_EAGER = 3                 # _attention: the Q MMs before the K appends (ahead + 1)
ATTN_LEAD = 2               # _attention's _attend_heads: two heads' first scores in flight,
ATTN_LAZY = True            # each head's queries quantized when its scores need them,
ATTN_LAG = 1                # its output written a head late (the tail's orders)
KINDS = ("attn", "wq", "wk", "wv", "wo")    # a formats string's kinds: the attention
                                            # projections ("attn" each one's default)


# =============================================================================== model spec
@dataclass(frozen=True)
class Spec:
    hidden: int             # the model's (2880); the image's rows are padded to whole D blocks
    kinds: tuple            # SLIDE or FULL per layer
    n_q: int
    n_kv: int
    head_dim: int
    vocab: int
    moe: MO.MoESpec         # every layer's: E, k, ffn, rule softmax, act "oai", bias
    window: int = 128       # a sliding layer's positions: p - window + 1 .. p
    eps: float = 1e-5
    theta: float = 150000.0
    rope_div: tuple = ()    # YaRN's frequency divisors and attention factor (llm.yarn)
    rope_scale: float = 1.0
    tied: bool = False
    bos: int = 199998
    eos: tuple = (200002,)
    embed: str = "f32"      # the embedding rows: fp32, or "int8" (int8 per D block, gathered on
                            # the device from a table of the image's own: qwen3.Spec.embed)
    formats: str = "attn=int8"  # the attention projections' formats over wformat (KINDS):
                            # int8 (fp4's are KL 0.33 from float over 2000 tokens, top-1 0.60
                            # past 1500; int8's 0.049, float weights' 0.041; +11.7% a token)

    @property
    def layers(self) -> int:
        return len(self.kinds)

    @property
    def rope_dim(self) -> int:
        return self.head_dim

    def is_moe(self, i: int) -> bool:
        return True

    @staticmethod
    def from_hf(model_dir) -> "Spec":
        c = json.loads((Path(model_dir) / "config.json").read_text())
        if not c.get("attention_bias", True):
            raise ValueError(f"{model_dir}: GPT-OSS without attention biases is not supported")
        d = c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"]
        rp = rope_parameters(c, model_dir, types=("default", "yarn"))
        theta = rp.get("rope_theta", c.get("rope_theta", 150000.0))
        div, sc = ((), 1.0)
        if rp.get("rope_type", rp.get("type")) == "yarn":
            div, sc = yarn(rp, d, theta)
        L = c["num_hidden_layers"]
        types = c.get("layer_types") or ["sliding_attention" if (i + 1) % 2 else "full_attention"
                                         for i in range(L)]
        eos = c.get("eos_token_id", 200002)
        g = Path(model_dir) / "generation_config.json"
        if g.exists():
            eos = json.loads(g.read_text()).get("eos_token_id", eos)
        moe = MO.MoESpec(E=c["num_local_experts"], k=c["num_experts_per_tok"],
                         ffn=c["intermediate_size"], rule="softmax", act="oai", bias=True,
                         limit=float(c.get("swiglu_limit", 7.0)))
        return Spec(hidden=c["hidden_size"],
                    kinds=tuple(SLIDE if t == "sliding_attention" else FULL for t in types),
                    n_q=c["num_attention_heads"], n_kv=c["num_key_value_heads"], head_dim=d,
                    vocab=c["vocab_size"], moe=moe, window=c.get("sliding_window") or 128,
                    eps=c.get("rms_norm_eps", 1e-5), theta=theta, rope_div=div, rope_scale=sc,
                    tied=c.get("tie_word_embeddings", False), bos=c.get("bos_token_id", 199998),
                    eos=tuple(eos) if isinstance(eos, list) else (eos,))

    def check(self, cfg: Config) -> None:
        S, D = cfg.S, cfg.D
        Hp = -(-self.hidden // D) * D
        need = [(self.n_kv % S == 0, f"{self.n_kv} KV heads over {S} slices"),
                (self.hidden % S == 0 and self.vocab % S == 0,
                 f"hidden {self.hidden} and vocab {self.vocab} over {S} slices"),
                (D == 128, "D 128 (the experts' MXFP4 import)"),
                (self.head_dim <= D and self.head_dim % 2 == 0,
                 f"head_dim {self.head_dim}: at most D {D}, even"),
                (self.n_q * self.head_dim % D == 0, "n_q * head_dim % D"),
                (self.n_kv * self.head_dim % D == 0, "n_kv * head_dim % D"),
                (self.n_q % self.n_kv == 0, "n_q % n_kv"),
                (self.n_q // self.n_kv <= cfg.MCOLS, "query group larger than MXU columns"),
                (self.window <= RING, f"window {self.window} > the ring's {RING}"),
                (self.moe.ffn % (D // 8) == 0, f"expert width {self.moe.ffn} % D/8"),
                (self.moe.k <= self.moe.E, "top-k above the expert count"),
                (max(self.moe.ffn, self.n_q * self.head_dim, Hp) <= cfg.ACT_BLOCKS * D,
                 "an inner dimension exceeds ACT RAM")]
        bad = [m for ok, m in need if not ok]
        if bad:
            raise ValueError("model does not map onto this openTPU config: " + "; ".join(bad))

    def image(self, cfg: Config, cap: int, batch: int = 1, rows: int = 1,
              wformat: str = "fp4", head_format: str | None = None,
              lookup: bool | str = False, experts: int | None = None,
              embed_host: bool | None = None, formats: str | None = None,
              high: bool | None = None, pool: bool | None = None) -> "Image":
        import os
        if pool is None and os.environ.get("OTPU_POOL") == "1":    # (a cut on the resident
            pool = experts in (None, self.moe.E)                    # pool, as the model runs)
        return fit_high(lambda h: Image(self, cfg, cap, batch, rows, wformat, head_format,
                                        lookup, experts, embed_host, formats, h, pool),
                        cfg, high)


# =============================================================================== weights
class Weights(dict):
    """A GPT-OSS checkpoint's tensors by name, read when asked for and not kept: the bf16 ones
    as fp32 (a NaN or an infinity refused: qwen3.finite), the experts' MXFP4 blocks and scales
    (uint8 tensors) as they are. part(k, i): entry i along the first axis (one expert), read
    alone."""

    def __init__(self, model_dir):
        super().__init__()
        from safetensors import safe_open
        self.model_dir = Path(model_dir)
        self._at, self._h = {}, {}
        for f in sorted(Path(model_dir).glob("*.safetensors")):
            h = self._h[str(f)] = safe_open(str(f), "pt")
            for k in h.keys():
                self._at[k] = (str(f), k)

    def _get(self, k, i=None):
        import torch
        f, name = self._at[k]
        h = self._h[f]
        t = h.get_tensor(name) if i is None else h.get_slice(name)[i]
        if t.dtype == torch.uint8:
            return t.numpy()
        return finite(t.to(torch.float32).numpy(), k if i is None else f"{k}[{i}]")

    def __getitem__(self, k):
        return self._get(k)

    def part(self, k, i):
        return self._get(k, i)

    def release(self) -> None:
        self._h.clear()

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


def load_weights(model_dir) -> Weights:
    return Weights(model_dir)


def _part(W, k, i):
    get = getattr(W, "part", None)
    return np.asarray(get(k, i) if get is not None else W[k][i])


E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], np.float32)


def mxfp4_codes(blocks: np.ndarray) -> np.ndarray:
    """A checkpoint's MXFP4 blocks [N, K/32, 16] (two E2M1 codes a byte, the first in the low
    nibble) as codes [N, K]."""
    b = np.asarray(blocks, np.uint8)
    out = np.empty(b.shape[:-1] + (32,), np.uint8)
    out[..., 0::2] = b & 15
    out[..., 1::2] = b >> 4
    return out.reshape(b.shape[0], -1)


def mxfp4_values(codes: np.ndarray, e8m0: np.ndarray) -> np.ndarray:
    """MXFP4 codes [N, K] with their E8M0 scales [N, K/32] as fp32 values (Hugging Face's
    dequantization: E2M1 times 2^(e - 127); exact)."""
    N, K = codes.shape
    v = E2M1[codes].reshape(N, K // 32, 32)
    return (v * np.exp2(np.asarray(e8m0, np.float32) - 127)[..., None]).reshape(N, K)


def mxfp4_parts(W, i: int, e: int):
    """Layer i's expert e as the checkpoint's MXFP4: ((gate_up codes [2F, H], scales), (down
    codes [H, F], scales)), or None for a dense checkpoint (gate_up_proj [E, H, 2F] and
    down_proj [E, F, H], Hugging Face's parameters)."""
    p = f"model.layers.{i}.mlp.experts."
    if p + "gate_up_proj_blocks" not in W:
        return None
    return tuple((mxfp4_codes(_part(W, p + n + "_blocks", e)), _part(W, p + n + "_scales", e))
                 for n in ("gate_up_proj", "down_proj"))


def expert_biases(W, i: int, e: int) -> tuple:
    """Layer i's expert e's biases (fp32): b_gate, b_up [F] (gate_up's even and odd entries),
    b_down [H]."""
    p = f"model.layers.{i}.mlp.experts."
    b = _part(W, p + "gate_up_proj_bias", e).astype(np.float32)
    return b[0::2], b[1::2], _part(W, p + "down_proj_bias", e).astype(np.float32)


def expert(W, i: int, e: int) -> tuple:
    """Layer i's expert e in fp32: W_gate, W_up [F, H], W_down [H, F], b_gate, b_up [F], b_down
    [H] (gate_up's even rows are the gate's, its odd ones the up projection's)."""
    p = f"model.layers.{i}.mlp.experts."
    mx = mxfp4_parts(W, i, e)
    if mx is None:
        gu = _part(W, p + "gate_up_proj", e).astype(np.float32).T       # [2F, H]
        dn = _part(W, p + "down_proj", e).astype(np.float32).T          # [H, F]
    else:
        gu, dn = (mxfp4_values(c, s) for c, s in mx)
    return (gu[0::2], gu[1::2], dn) + expert_biases(W, i, e)


# =============================================================================== reference
def _norm(v, g, eps):
    return (v / np.sqrt(np.mean(v * v, axis=-1, keepdims=True) + eps)) * g


def _sigmoid(x):
    with np.errstate(over="ignore"):                # exp(-x) -> inf: 0, as torch's
        return 1 / (1 + np.exp(-x))


def _glu(g, u, mo):
    g = np.minimum(g, mo.limit)
    u = np.clip(u, -mo.limit, mo.limit)
    return (u + 1) * (g * _sigmoid(mo.alpha * g))


def reference_logits(spec: Spec, W: dict, tokens) -> np.ndarray:
    """fp32 numpy forward of the whole sequence (causal; a sliding layer's keys the last
    `window` positions); returns logits [T, vocab]."""
    tokens = list(tokens)
    T, d, G = len(tokens), spec.head_dim, spec.n_q // spec.n_kv
    x = np.asarray(W["model.embed_tokens.weight"][tokens], np.float32)
    cs = [rope_tables(spec, p) for p in range(T)]
    cos = np.stack([c for c, _ in cs])[:, None, :]
    sin = np.stack([s for _, s in cs])[:, None, :]

    def rot(v):
        v1, v2 = v[..., :d // 2], v[..., d // 2:]
        return np.concatenate([v1 * cos - v2 * sin, v2 * cos + v1 * sin], axis=-1)

    qi, ki = np.arange(T)[:, None], np.arange(T)[None, :]
    full = np.where(ki > qi, -np.inf, 0).astype(np.float32)
    slide = np.where((ki > qi) | (ki <= qi - spec.window), -np.inf, 0).astype(np.float32)
    for i, kind in enumerate(spec.kinds):
        p = f"model.layers.{i}."
        a = p + "self_attn."
        h = _norm(x, W[p + "input_layernorm.weight"], spec.eps)
        q = (h @ W[a + "q_proj.weight"].T + W[a + "q_proj.bias"]).reshape(T, spec.n_q, d)
        k = (h @ W[a + "k_proj.weight"].T + W[a + "k_proj.bias"]).reshape(T, spec.n_kv, d)
        v = (h @ W[a + "v_proj.weight"].T + W[a + "v_proj.bias"]).reshape(T, spec.n_kv, d)
        q, k = rot(q), rot(k)
        mask = slide if kind == SLIDE else full
        sinks = np.asarray(W[a + "sinks"], np.float32)
        o = np.zeros((T, spec.n_q, d), np.float32)
        for hq in range(spec.n_q):
            s = q[:, hq] @ k[:, hq // G].T / np.float32(math.sqrt(d)) + mask
            s = np.concatenate([s, np.full((T, 1), sinks[hq], np.float32)], axis=1)
            s = np.exp(s - s.max(axis=1, keepdims=True))
            o[:, hq] = (s[:, :T] / s.sum(axis=1, keepdims=True)) @ v[:, hq // G]
        x = x + o.reshape(T, -1) @ W[a + "o_proj.weight"].T + W[a + "o_proj.bias"]
        h = _norm(x, W[p + "post_attention_layernorm.weight"], spec.eps)
        x = x + _moe_reference(h, W, i, spec.moe)
    x = _norm(x, W["model.norm.weight"], spec.eps)
    head = W["model.embed_tokens.weight"] if spec.tied else W["lm_head.weight"]
    return x @ head.T


def _moe_reference(h, W, i: int, mo) -> np.ndarray:
    """The routed experts' sum for the rows of h [T, H] (fp32, Hugging Face's math)."""
    p = f"model.layers.{i}.mlp.router."
    logits = h @ W[p + "weight"].T + W[p + "bias"]
    y = np.zeros_like(h)
    ex = {}
    for t in range(h.shape[0]):
        ids, w = MO.route(logits[t], None, mo)
        for e, we in zip(ids, w):
            if e not in ex:
                ex[e] = expert(W, i, e)
            wg, wu, wd, bg, bu, bd = ex[e]
            a = _glu(wg @ h[t] + bg, wu @ h[t] + bu, mo)
            y[t] += np.float32(we) * (wd @ a + bd)
    return y


# =============================================================================== the experts' bytes
def expert_mats(W, i: int, e: int, fmt: MO.ExpertFormat) -> SimpleNamespace:
    """Layer i's expert e in its slot's format fmt (an ExpertFormat of width fmt.H, the padded
    hidden size, with biases; tight: W_down's rows Hd, the model's): g, u (W_gate, W_up
    [fmt.Fg, H]) and d (W_down [Hd, fmt.F]), each (rows, scale rows) as ExpertFormat.pack_q
    takes them, zero-padded; the biases bg, bu [F0] and bd [Hd]; how: per matrix "import" (the
    checkpoint's MXFP4 exactly, quant.import_mxfp4)
    or "quant" (quantized from its values: a dense checkpoint, an int8 slot, or a matrix whose
    exponents the slot's 4-bit format does not hold)."""
    D, H, F, f = fmt.D, fmt.H, fmt.F, fmt.wformat
    Fg, Hd = fmt.Fg, fmt.Hd
    mx = mxfp4_parts(W, i, e) if f in ("fp4", "nvfp4") else None
    wg, wu, wd, bg, bu, bd = expert(W, i, e) if mx is None else \
        (None,) * 3 + expert_biases(W, i, e)
    out = SimpleNamespace(bg=bg, bu=bu, bd=np.pad(bd, (0, Hd - len(bd))), how=[])

    def pad(a, n, k):
        z = np.zeros((n, k), a.dtype)
        z[:a.shape[0], :a.shape[1]] = a
        return z

    def put(name, vals, codes=None, sc=None, n=0, k=0):
        if codes is not None:
            c, s = pad(codes, n, k), np.full((n, k // 32), 127, np.uint8)
            s[:sc.shape[0], :sc.shape[1]] = sc
            try:
                rows, srows, _ = Q.import_mxfp4(c, s, D, want=f)
                setattr(out, name, (rows, srows))
                out.how.append("import")
                return
            except Q.NotExact:
                vals = mxfp4_values(c, s)
        setattr(out, name, Q.quantize_mxu(pad(np.asarray(vals, np.float32), n, k), f, D))
        out.how.append("quant")

    if mx is None:
        put("g", wg, n=Fg, k=H)
        put("u", wu, n=Fg, k=H)
        put("d", wd, n=Hd, k=F)
    else:
        (cgu, sgu), (cd, sd) = mx
        put("g", None, cgu[0::2], sgu[0::2], Fg, H)
        put("u", None, cgu[1::2], sgu[1::2], Fg, H)
        put("d", None, cd, sd, Hd, F)
    return out


def _mxw(qs, fmt: str, K: int, D: int, cols: int | None = None):
    """A quantized matrix (rows, scale rows) as an MXW of its first `cols` columns (quant.fake_w
    of its values)."""
    from ..isa import MX_W, WFORMATS
    rows, srows = qs
    v = Q.dequantize_mxu(rows, srows, fmt, K, D)
    dw = Q.group_offsets(srows, fmt, K, D)
    if cols is not None:
        v, dw = v[:, :cols], dw[:, :cols // (D // Q.NVG)]
    return Q.MXW.of(v, dw, MX_W[WFORMATS[fmt]], D)


# =============================================================================== emulation
def emulated_logits(spec: Spec, W: dict, tokens, D: int = 128, wformat: str = "fp4",
                    head_format: str | None = None, routes: list | None = None,
                    grp: int = 1, experts: dict | None = None,
                    formats: str | None = None) -> np.ndarray:
    """float64 decode with openTPU's quantization points and none of its rounding (as
    qwen3.emulated_logits; one slice): every hidden-wide vector padded with zeros to Hp (the
    norms over the model's width), the embedding rows as spec.embed, the attention projections
    in `wformat` and `formats` (as Image), the router int8, the head in head_format (int8),
    the experts as their slots hold them (expert_mats: the checkpoint's MXFP4 imported), q and
    K zero-padded to D, a sliding layer's keys in their ring slots as one attention block (its
    P one QACT), the sinks in the softmax's sum; a layer at a time over the tokens (one layer's
    experts held at once).
    routes: (token, layer, ids, the k-th and (k+1)-th logits' margin) appended per MoE layer
    and token (layer by layer). experts: {(token, layer): ids} the choices there in place of
    the emulation's own (e.g. the card's, its expert server's requests: where a near-tie of the
    router's logits went the other way in fp32), their weights the softmax of the emulation's
    logits over them."""
    d, G, mo = spec.head_dim, spec.n_q // spec.n_kv, spec.moe
    H0, Hp = spec.hidden, -(-spec.hidden // D) * D
    dk = -(-d // D) * D
    head_format = head_format or "int8"
    fmt = MO.ExpertFormat(Hp, mo.ffn, D, wformat, grp, bias=True, rows=H0, tight=True)
    ranges = fmt.ranges()
    af = FM.resolver(formats, KINDS, spec.formats, wformat)     # the projections' formats
    Wq: dict = {}

    def padc(a):                    # zero columns to Hp
        a = np.asarray(a, np.float32)
        return np.pad(a, [(0, 0)] * (a.ndim - 1) + [(0, Hp - a.shape[-1])])

    def padr(a):                    # zero rows to Hp
        a = np.asarray(a, np.float32)
        return np.pad(a, [(0, Hp - a.shape[0])] + [(0, 0)] * (a.ndim - 1))

    def w(n, f, pad=padc):
        if n not in Wq:
            Wq[n] = Q.fake_w(pad(W[n]), f, D)
        return Wq[n]

    def norm(v, g):
        g = padc(np.asarray(g, np.float32)[None])[0]
        return v / np.sqrt(np.sum(v * v) / H0 + spec.eps) * g

    def pad_d(v):
        return np.pad(v, [(0, 0)] * (v.ndim - 1) + [(0, dk - d)])

    emb = W["model.embed_tokens.weight"]
    xs = []
    for tk in tokens:
        x = padc(np.asarray(emb[tk], np.float32)[None])
        if spec.embed == "int8":                    # the int8 embedding rows (Spec.embed)
            x = gathered_rows(x, D)
        xs.append(x[0].astype(np.float64))
    ropes = [rope_tables(spec, pos) for pos in range(len(xs))]

    def rot(v, pos):
        c, s = ropes[pos]
        v1, v2 = v[..., :d // 2], v[..., d // 2:]
        return np.concatenate([v1 * c - v2 * s, v2 * c + v1 * s], -1)

    # a layer at a time over every position (a position's layer reads that layer's K / V of the
    # positions before it alone: the token-by-token decode's values), its weights and experts
    # dropped after it
    for i, kind in enumerate(spec.kinds):
        p = f"model.layers.{i}."
        a = p + "self_attn."
        sinks = np.asarray(W[a + "sinks"], np.float64)
        Kc, Vc, ex = [], [], {}
        for pos, x in enumerate(xs):
            h = _fake_q(norm(x, W[p + "input_layernorm.weight"]), D)
            q, k, v = (w(a + n + "_proj.weight", af("w" + n, i)) @ h + W[a + n + "_proj.bias"]
                       for n in "qkv")
            q, k, v = q.reshape(spec.n_q, d), k.reshape(spec.n_kv, d), v.reshape(spec.n_kv, d)
            q, k = rot(q, pos), rot(k, pos)
            Kc.append(_fake_k(pad_d(k), D))
            Vc.append(_v_parts(v))
            # the keys this position sees, and where they sit in the attention blocks
            lo = max(0, pos - spec.window + 1) if kind == SLIDE else 0
            keys = list(range(lo, pos + 1))
            Kh = np.stack([Kc[t][0] for t in keys], 1)
            dK = None if Kc[0][1] is None else np.stack([Kc[t][1] for t in keys], 1)
            Vq = np.stack([Vc[t][0] for t in keys], 1)
            Vs = np.stack([Vc[t][1] for t in keys], 1)
            slots = [t % RING for t in keys] if kind == SLIDE else keys
            o = np.zeros((spec.n_q, d))
            for hq in range(spec.n_q):
                sc = _qk(_fake_q(pad_d(q[hq] / math.sqrt(d)), D), Kh[hq // G],
                         None if dK is None else dK[hq // G], D)
                m = max(sc.max(), sinks[hq])
                pp = np.exp(sc - m)
                o[hq] = _pv_slots(pp, Vq[hq // G], Vs[hq // G], slots, D) / \
                    (pp.sum() + np.exp(sinks[hq] - m))
            x = x + w(a + "o_proj.weight", af("wo", i), padr) @ _fake_q(o.reshape(-1), D) + \
                padc(np.asarray(W[a + "o_proj.bias"])[None])[0]
            h = _fake_q(norm(x, W[p + "post_attention_layernorm.weight"]), D)
            r = p + "mlp.router."
            lg = w(r + "weight", "int8") @ h + W[r + "bias"]
            ids, wts = MO.route(lg, None, mo)
            if experts is not None and (pos, i) in experts:
                ids = list(experts[pos, i])
                wts = np.exp(lg[ids] - lg[ids].max())
                wts = wts / wts.sum()
            if routes is not None:
                srt = np.sort(lg)[::-1]
                routes.append((pos, i, ids, srt[mo.k - 1] - srt[mo.k] if mo.k < len(srt)
                               else np.inf))
            for e, we in zip(ids, wts):
                if e not in ex:
                    mt = expert_mats(W, i, e, fmt)
                    ex[e] = (_mxw(mt.g, wformat, Hp, D), _mxw(mt.u, wformat, Hp, D),
                             _mxw(mt.d, wformat, fmt.F, D, mo.ffn), mt.bg, mt.bu, mt.bd)
                mg, mu, md, bg, bu, bd = ex[e]
                g = (mg @ h)[:mo.ffn] + bg
                u = (mu @ h)[:mo.ffn] + bu
                o = md @ _fake_q(_glu(g, u, mo), D, ranges) + bd     # [H0]
                x = x + we * np.pad(o, (0, Hp - H0))
            xs[pos] = x
        Wq.clear()
    hd = "model.embed_tokens.weight" if spec.tied else "lm_head.weight"
    return np.array([w(hd, head_format) @ _fake_q(norm(x, W["model.norm.weight"]), D)
                     for x in xs])


def _pv_slots(pp, Vq, vs, slots, D: int = 128, block: int = ATTN_BLOCK):
    """P.V as the device's flash attention takes it (qwen3._pv) for keys at cache slots `slots`
    (in order; a sliding layer's ring slots, which wrap): per attention block of the slots one
    QACT of the numerators times V's row scales (the block's other slots zero), against V's int8
    values there; the last block padded to whole D blocks."""
    slots = np.asarray(slots)
    out = 0.0
    for b0 in sorted({int(t) // block * block for t in slots}):
        sel = (slots >= b0) & (slots < b0 + block)
        n = int(slots[sel].max()) - b0 + 1
        npad = -(-n // D) * D
        seg = np.zeros(npad)
        seg[slots[sel] - b0] = pp[sel] * vs[sel]
        vt = np.zeros((npad, Vq.shape[-1]))
        vt[slots[sel] - b0] = Vq[sel]
        out = out + Q.mx_matmul(_fake_q(seg, D), vt.T, None, I.MX_W[I.W8], D)
    return out


# =============================================================================== DRAM image
class Image(EmbedHost):
    """One slice's DRAM layout of a GPT-OSS model (at S > 1 slice s's: the module docstring).

    [ I/O: x, cos, sin | final norm | logits ] [ layer blocks ] [ LM head ] [ lookup tables:
    the embedding table (Spec.embed), RoPE rows, masks, the generate area; the sliding mask
    table ]
    [ path (a)'s words and expert slots ]. A layer block (sliding or full: the same offsets
    but for the KV cache's size) holds the two norms, q / k / v / o with their biases, the
    sinks, the router (int8) with its bias and the layer's base id (j E), then its KV heads: a
    ring of RING slots (sliding) or `cap` (full). q / k / v / o are in wformat but where the
    formats string (`formats`, else OTPU_FORMATS, else spec.formats; KINDS) says otherwise, each
    one format in every layer (mf). Every hidden-wide vector and matrix side is padded with
    zeros to Hp. The layers run as the runs of lfm2.plan over their kinds: a hardware loop over
    (sliding, full) pairs. experts: slots per layer (default E, every expert resident; fewer
    stream from the host's server, docs/offload.md).

    high (above AW 32: qwen3.fit_high, by fit unless OTPU_HIGH says; GPT-OSS-20B's 13 GB on
    the 32 GiB card): the layer blocks' weights, the head and the experts from HIGH (4 GiB) on,
    each layer block's KV heads in a block of their own below it (lloc, lsize: the runs' units
    one after the other as the weights'), after the I/O area and before the lookup tables. With
    every expert in DRAM (experts = E) they are a resident pool (pool: global expert g at
    pool.base + g pool.ebytes, no directory and no host server: moe_ffn's resident path); with
    fewer, path (a)'s words and slots below HIGH (32-bit directory entries). pool: the resident
    pool below HIGH too (True; a cut that fits there, as the whole model runs), or path (a)'s
    slots at experts = E above it (False); None: by high. nbytes is the image's extent,
    low_end the end of what is below HIGH."""

    def __init__(self, spec: Spec, cfg: Config, cap: int, batch: int = 1, rows: int = 1,
                 wformat: str = "fp4", head_format: str | None = None,
                 lookup: bool | str = False, experts: int | None = None,
                 embed_host: bool | None = None, formats: str | None = None,
                 high: bool = False, pool: bool | None = None):
        spec.check(cfg)
        self.high = bool(high)
        if batch != 1:
            raise ValueError("GPT-OSS runs one sequence: batch=1")
        if embed_host:
            raise ValueError("GPT-OSS keeps its embedding table on the card (embed_host)")
        if cap % cfg.D:
            raise ValueError("KV capacity must be a multiple of D")
        D, d, mo = cfg.D, spec.head_dim, spec.moe
        Hp = -(-spec.hidden // D) * D
        self.spec, self.cfg, self.cap, self.batch, self.rows = spec, cfg, cap, 1, rows
        self.H, self.ring, self.dk = Hp, RING, -(-d // D) * D
        # the lookup tables' and the generate area's model: hidden-wide rows of Hp
        self.pspec = dataclasses.replace(spec, hidden=Hp)
        self.wformat, self.head_format = wformat, head_format or "int8"
        S = cfg.S
        self.v_loc = spec.vocab // S
        mb = lambda n, k, f, t=False: Q.grp_bytes(n, k, f, D, cfg.WG, t)     # noqa: E731
        A = _align(cfg)
        b = _Bump(align=A)
        hb = _Bump(high_base(cfg), A) if self.high else b       # the weights' (high: from HIGH;
        #                                                         the bank-half window: past it)
        self.io = {"x": b.alloc(4 * Hp), "cos": b.alloc(2 * d), "sin": b.alloc(2 * d),
                   "gf": b.alloc(4 * Hp), "logits": b.alloc(4 * spec.vocab)}
        nq, nkv = spec.n_q * d, spec.n_kv * d
        nql, nkvl, hl = nq // S, nkv // S, spec.hidden // S             # (a slice's)
        self.mats = {"wq": (nql, Hp), "wk": (nkvl, Hp), "wv": (nkvl, Hp),
                     "wo": (hl, nq)}                    # (W_o's rows: the model's own)
        R = Q.grp_r("int8", cfg.WG)     # the router's rows a slice: S > 1 in whole groups
        self.er = mo.E if S == 1 else -(-mo.E // (S * R)) * R           # (zero rows past E)
        self.formats = _formats(spec, formats)
        self.mf = FM.uniform(FM.resolver(self.formats, KINDS, spec.formats, wformat),
                             dict.fromkeys(self.mats, range(spec.layers)))
        # Config.KT: a matrix whose own K is the model's 2880 (22.5 D-blocks) in MM KT's pairs
        # (docs/isa.md "K tail"): q / k / v, the router and the head, and in the expert slots
        # W_gate, W_up and W_down's last part (ExpertFormat kt); not W_o (K 4096)
        kt = lambda f, k: cfg.KT and cfg.WG > 1 and f != "nvfp4" and Q.kt_fits(k, D)  # noqa
        self.kt = {name: kt(self.mf[name], nq if name == "wo" else spec.hidden)
                   for name in self.mats}
        self.kt.update(router=kt("int8", spec.hidden), head=kt(self.head_format, spec.hidden))
        lb = _Bump(align=A)                             # offsets inside a layer block
        L = {"g_in": lb.alloc(4 * Hp), "g_post": lb.alloc(4 * Hp), "bq": lb.alloc(4 * nql),
             "bk": lb.alloc(4 * nkvl), "bv": lb.alloc(4 * nkvl), "bo": lb.alloc(4 * hl),
             "sinks": lb.alloc(4 * (spec.n_q // S)), "rbias": lb.alloc(4 * mo.E),
             "gbase": lb.alloc(4)}
        for name, (n, k) in self.mats.items():
            L[name] = tuple(lb.alloc(x) for x in mb(n, k, self.mf[name], self.kt[name]))
        L["router"] = tuple(lb.alloc(x) for x in mb(self.er, Hp, "int8", self.kt["router"]))
        dk = self.dk
        r4k = lambda n: (n + 4095) // 4096 * 4096                   # noqa: E731
        self.lofs, self.size, self.lsize = {}, {}, {}
        for kind in (SLIDE, FULL):
            kb = _Bump(align=A) if self.high else _Bump(lb.next, A)    # (high: a block of its
            ck = RING if kind == SLIDE else cap                         # own, below HIGH)
            self.lofs[kind] = dict(L, kv=[{"k": kb.alloc(ck * dk),
                                           "ks": kb.alloc(ck * k_scale_bytes(dk, D)),
                                           "vt": kb.alloc(dk * ck), "vs": kb.alloc(4 * ck)}
                                          for _ in range(spec.n_kv // S)])
            self.size[kind] = r4k(lb.next if self.high else kb.next)
            self.lsize[kind] = r4k(kb.next) if self.high else 0
        self.keys = spec.kinds
        self.plan = plan(self.keys)
        self.layer0 = hb.next
        self.loc = _place(hb, self.plan, lambda k: self.size[k])
        self.lloc = _place(b, self.plan, lambda k: self.lsize[k]) if self.high else self.loc
        self.head = tuple(hb.alloc(x) for x in mb(self.v_loc, Hp, self.head_format,
                                                  self.kt["head"]))
        self.lookup = _lookup_alloc(b, self.pspec, cap, D=D, M=cfg.MCOLS, rows=1,
                                    imem=cfg.IMEM_WORDS, waitc=cfg.SEQ_WC, S=S,
                                    wg=gall_wg(cfg), chash=cfg.CHASH) if lookup else {}
        self.smask = b.alloc(4 * 2 * 2 * RING)          # the sliding mask tables (smask_table)
        self.choices = {"high": True} if self.high else {}      # (the compile worker's image:
        if self.formats != spec.formats:                        # formats other than the Spec's)
            self.choices["formats"] = self.formats
        # path (a): the words and the expert slots
        self.fmt = MO.ExpertFormat(Hp, mo.ffn, D, wformat, gall_wg(cfg), bias=True,
                                   rows=spec.hidden, tight=True, kt=cfg.KT, S=S)
        self.slot_lanes = (cfg.WG, D, self.fmt.data) if self.fmt.grp > 1 and cfg.CHASH \
            else None
        n = mo.E if experts is None else experts
        if not mo.k <= n <= mo.E:
            raise ValueError(f"{n} expert slots per layer: from top-k {mo.k} to {mo.E}")
        if pool and n != mo.E:
            raise ValueError(f"a resident pool holds every expert: experts {n}, E {mo.E}")
        if S > 1 and (n != mo.E or pool is False):
            raise ValueError(f"GPT-OSS at S = {S}: every expert resident in a pool (its rows a "
                             f"slice: moe._moe_ffn_tp), experts {n} of {mo.E}")
        self.offload = self.pool = None
        self._last = None                               # (expert: the last import, its parts)
        if S > 1 or ((self.high and n == mo.E) if pool is None else pool):  # (all in DRAM)
            if pool:
                self.choices["pool"] = True
            self.pool = MO.resident_pool(hb, spec.layers * mo.E, self.fmt)
        else:                       # (the directory's slots below 4 GiB: 32-bit entries)
            if self.high and pool is False:
                raise ValueError("path (a)'s slots of every expert above 4 GiB: 32-bit "
                                 "directory entries")
            self.offload = Layout.build((b.next + 4095) // 4096 * 4096, mo.E, mo.k,
                                        [n] * spec.layers, self.fmt.nbytes)
            b.next = self.offload.end
        self.imports: dict = {}                         # expert_mats' how, counted (build)
        self.low_end, self.nbytes = b.next, max(b.next, hb.next)
        low_check(self, cfg, self.high)
        if self.nbytes > cfg.DRAM_BYTES:
            raise MemoryError(f"model image needs {self.nbytes / 2**20:.0f} MiB per slice, "
                              f"DRAM_BYTES is {cfg.DRAM_BYTES / 2**20:.0f} MiB")

    @property
    def rollback_slack(self) -> int:
        """A rollback of at most this many positions keeps every sliding window in its ring
        (Engine.rollback; further back the rings come from a snapshot)."""
        return self.ring - self.spec.window + 1

    def _off(self, li, it=None, loc=None) -> Affine:
        """The block address of layer li (static), or of element li of its run's unit at
        iteration `it` (a loop variable); loc: self.loc (the weights') or self.lloc."""
        base, us, i, o, _ = (self.loc if loc is None else loc)[li]
        return Affine(base + o) + Affine.of(i if it is None else it) * us

    def _loff(self, li, it=None) -> Affine:
        """_off of layer li's KV heads' block (Image.high; else its block)."""
        return self._off(li, it, self.lloc)

    # ---- contents
    def build(self, W: dict) -> list[np.ndarray]:
        """The DRAM images (one a slice) with every weight in place (the experts' slots are
        the server's: serve), the KV caches empty."""
        spec, cfg = self.spec, self.cfg
        D, Hp, mo, S, d = cfg.D, self.H, spec.moe, cfg.S, spec.head_dim
        imgs = [np.zeros(self.nbytes, np.uint8) for _ in range(S)]

        def put(s, addr, a):
            v = np.ascontiguousarray(a).view(np.uint8).reshape(-1)
            imgs[s][addr:addr + v.size] = v

        def every(addr, a):
            for s in range(S):
                put(s, addr, a)

        qb = QC.Batch()
        gw, gh = cfg.WG, cfg.CHASH

        def put_q(pair, a, fmt, kt, s=0):
            qb.submit(a, fmt, D, lambda q, sc, k=a.shape[1]: [
                put(s, x, y) for x, y in zip(pair, Q.grouped(q, sc, fmt, k, gw, D,
                                                             at=pair[0] if gh else None,
                                                             kt=kt))])

        def parts(a, n):                            # slice s's n rows (S = 1: a)
            return [a[s * n:(s + 1) * n] for s in range(S)]
        # the slices' heads (kernels.layouts' head-parallel order: KV heads s, s + S, ... and
        # their query groups, slice-major; S = 1 in order): q rows, KV rows, q heads
        G = spec.n_q // spec.n_kv
        kvo = [h for s in range(S) for h in range(s, spec.n_kv, S)]
        qo = [h * G + g for h in kvo for g in range(G)]
        qr = np.concatenate([np.arange(h * d, (h + 1) * d) for h in qo])
        kr = np.concatenate([np.arange(h * d, (h + 1) * d) for h in kvo])

        def f32(a):
            return FP.ftz(np.asarray(a, np.float32))

        def padc(a):
            a = np.asarray(a, np.float32)
            return np.pad(a, [(0, 0)] * (a.ndim - 1) + [(0, Hp - a.shape[-1])])

        every(self.io["gf"], f32(padc(W["model.norm.weight"])))
        nql, nkvl, hl = self.mats["wq"][0], self.mats["wk"][0], self.mats["wo"][0]
        for i, kind in enumerate(spec.kinds):
            p, a, base = f"model.layers.{i}.", f"model.layers.{i}.self_attn.", self._off(i).const
            Lo = {k: (tuple(base + x for x in v) if isinstance(v, tuple) else
                      base + v if isinstance(v, int) else v) for k, v in self.lofs[kind].items()}
            every(Lo["g_in"], f32(padc(W[p + "input_layernorm.weight"])))
            every(Lo["g_post"], f32(padc(W[p + "post_attention_layernorm.weight"])))
            for n, at, rows, k in (("q_proj", "bq", qr, nql), ("k_proj", "bk", kr, nkvl),
                                   ("v_proj", "bv", kr, nkvl)):
                wn = "w" + n[0]
                for s, (bb, ww) in enumerate(zip(parts(np.asarray(W[a + n + ".bias"])[rows], k),
                                                 parts(padc(W[a + n + ".weight"])[rows], k))):
                    put(s, Lo[at], f32(bb))
                    put_q(Lo[wn], ww, self.mf[wn], self.kt[wn], s)
            for s, sk in enumerate(parts(np.asarray(W[a + "sinks"])[qo], spec.n_q // S)):
                put(s, Lo["sinks"], f32(sk))
            wo = np.asarray(W[a + "o_proj.weight"], np.float32)[:, qr]   # (columns: the
            for s, (bb, ww) in enumerate(zip(parts(np.asarray(W[a + "o_proj.bias"]), hl),   # heads'
                                             parts(wo, hl))):                          # gather)
                put(s, Lo["bo"], f32(bb))
                put_q(Lo["wo"], ww, self.mf["wo"], self.kt["wo"], s)
            rw = padc(W[p + "mlp.router.weight"])
            rw = np.pad(rw, [(0, S * self.er - rw.shape[0]), (0, 0)])  # (S > 1: zero rows)
            for s, ww in enumerate(parts(rw, self.er)):
                put_q(Lo["router"], ww, "int8", self.kt["router"], s)
            every(Lo["rbias"], f32(W[p + "mlp.router.bias"]))
            every(Lo["gbase"], f32([i * mo.E]))
        hw = padc(W["model.embed_tokens.weight" if spec.tied else "lm_head.weight"])
        for s, ww in enumerate(parts(hw, self.v_loc)):
            put_q(self.head, ww, self.head_format, self.kt["head"], s)
        qb.wait()
        if self.lookup:
            _lookup_build(put, S, {"model.embed_tokens.weight":
                                   padc(W["model.embed_tokens.weight"])},
                          self.pspec, self.cap, self.lookup)
        every(self.smask, smask_table(RING, spec.window))
        if self.pool is not None:               # every expert in its place (Image.pool)
            MO.pool_build(self.pool, lambda qb, g, done, s=0: done(self.expert(W, g, s)),
                          self.slot_lanes, put, S)
        self._last = None
        return imgs

    # ---- path (a): the expert pool and its server (docs/offload.md)
    def expert(self, W: dict, g: int, part: int = 0) -> np.ndarray:
        """Global expert g (layer g // E, expert g % E) in its slot's bytes (S > 1: slice
        `part`'s; the import kept for the expert's next part, pool_build's order)."""
        if self._last is None or self._last[0] != g:
            i, e = divmod(g, self.spec.moe.E)
            mt = expert_mats(W, i, e, self.fmt)
            for h in mt.how:
                self.imports[h] = self.imports.get(h, 0) + 1
            self._last = (g, mt) if self.cfg.S > 1 else None
        else:
            mt = self._last[1]
        return self.fmt.pack_q(mt.g, mt.u, mt.d, (mt.bg, mt.bu, mt.bd), part=part)

    def serve(self, W: dict, backend, pool_file=None) -> ExpertServer:
        """The host's expert server on the backend's DRAM (moe.serve)."""
        return MO.serve(self.offload, lambda g: self.expert(W, g), backend, pool_file,
                        lanes=self.slot_lanes)

    def state_regions(self, slot: int = 0) -> list:
        """The context's state its KV caches do not hold once a ring has wrapped: each sliding
        layer's ring (address, bytes; a rollback past rollback_slack restores it from a
        snapshot, opentpu/llm/snapshot.py)."""
        out = []
        for li, kind in enumerate(self.spec.kinds):
            if kind != SLIDE:
                continue
            off = self._loff(li).const
            for e in self.lofs[SLIDE]["kv"]:
                out += [(off + e["k"], RING * self.dk), (off + e["ks"], RING * k_scale_bytes(
                    self.dk, self.cfg.D)), (off + e["vt"], self.dk * RING),
                    (off + e["vs"], 4 * RING)]
        return out

    # ---- programs
    def compile_decode(self, blocks: int, lo: int, block: int = ATTN_BLOCK):
        """(programs, run_args): gptoss_step at a run-time position (qwen3.compile_decode)."""
        _ring_block(block)
        return compile_decode(self, gptoss_step, blocks, lo, block)

    def compile_generate(self, blocks: int, lo: int, block: int = ATTN_BLOCK,
                         chain: bool = True, samp=None, debug: bool = False,
                         part: int | None = None) -> list:
        """The decode loop on the device for bucket `blocks` (gptoss_step in it, generate.py)."""
        _ring_block(block)
        return G.compile_generate(self, gptoss_step, blocks, lo, block, chain, samp, debug, part)

    def compile_step(self, pos: int, block: int = ATTN_BLOCK, tok: int | None = None) -> list:
        """The decode token at position `pos` (gptoss_step), its token's inputs from the
        image's tables."""
        _ring_block(block)
        return [trace_fit(gptoss_step, self.cfg, 0,
                          lambda: {"m": step_descriptors(self, 0), "pos": pos, "block": block,
                                   **_tok_arg(self, tok)}).finish()]

    def compile_rows(self, rows, logit_rows, block: int = ATTN_BLOCK, tokens=None) -> list:
        """One row per run (a MoE block routes one token): compile_step's program."""
        if len(rows) != 1 or rows[0][0] != 0:
            raise ValueError("GPT-OSS runs one row of sequence 0 per program")
        if list(logit_rows) not in ([0], []):
            raise ValueError("logit_rows: [0]")
        return self.compile_step(rows[0][1], block, None if tokens is None else tokens[0])

    # ---- kernel descriptors
    def descriptors(self, sid: int) -> SimpleNamespace:
        spec, cfg = self.spec, self.cfg
        D, d, Hp, mo, S = cfg.D, spec.head_dim, self.H, spec.moe, cfg.S

        def layer(li, it=None):
            """Descriptors of layer li (static), or of element li of its run's unit at
            iteration `it` (a hardware-loop variable)."""
            kind = spec.kinds[li]
            off, loff, L = self._off(li, it), self._loff(li, it), self.lofs[kind]

            def vec(name, n):
                return Tensor(off + L[name], (n,), (1,))
            ns = SimpleNamespace(kind=kind, g_in=vec("g_in", Hp), g_post=vec("g_post", Hp),
                                 bq=vec("bq", self.mats["wq"][0]),
                                 bk=vec("bk", self.mats["wk"][0]),
                                 bv=vec("bv", self.mats["wv"][0]),
                                 bo=vec("bo", self.mats["wo"][0]),
                                 sinks=vec("sinks", spec.n_q // S), rbias=vec("rbias", mo.E),
                                 gbase=vec("gbase", 1))
            for name, (r, k) in self.mats.items():
                da, sa = L[name]
                setattr(ns, name, QTensor.weight(off + da, off + sa, r, k, self.mf[name], D,
                                                 grp=cfg.WG, kt=self.kt[name]))
            da, sa = L["router"]
            ns.router = QTensor.weight(off + da, off + sa, self.er, Hp, "int8", D, grp=cfg.WG,
                                       kt=self.kt["router"])
            ns.kv = KVDesc({sid + j * S: {k: loff + v for k, v in r.items()}
                            for j, r in enumerate(L["kv"])},
                           RING if kind == SLIDE else self.cap, self.dk, D, S, sid, dv=d,
                           wg=gall_wg(cfg))
            return ns

        Lo, P = self.offload, self.pool
        if P is not None:           # every expert in DRAM (moe_ffn's resident path)
            dev = MO.pool_dev(P, self.fmt)
        else:
            dev = SimpleNamespace(mbox=Lo.mbox, served=Lo.served, answer=Lo.answer, dir=Lo.dir,
                                  tag=Lo.tag, fmt=self.fmt, slots=Lo.slot_bounds,
                                  ids=Lo.layers * Lo.E)
        return SimpleNamespace(
            spec=spec, layer=layer, plan=self.plan, moe_dev=dev, ring=RING, smask=self.smask,
            norm_n=spec.hidden, x=_tdesc(self.io["x"], (1, Hp)),
            cos=_tdesc(self.io["cos"], (d // 2,)), sin=_tdesc(self.io["sin"], (d // 2,)),
            g_final=_tdesc(self.io["gf"], (Hp,)),
            logits=_tdesc(self.io["logits"], (1, spec.vocab)),
            head=QTensor.weight(*self.head, self.v_loc, Hp, self.head_format, D, grp=cfg.WG,
                                kt=self.kt["head"]),
            v_loc=self.v_loc, **_lookup_desc(self.lookup, self.pspec, self.cap))


def smask_table(R: int, w: int) -> np.ndarray:
    """The sliding layers' mask tables (fp32, two of 2R entries): a ring slot j at position p
    (tpos = p mod R) takes entry R + j - tpos, +inf where the slot holds one of positions
    p - w + 1 .. p and -inf where not; table 0 for positions below R (slots after tpos are
    empty), table 1 after (slots after tpos hold the last lap's positions)."""
    i = np.arange(2 * R)
    now = (i >= R - w + 1) & (i <= R)
    lap = (i >= 2 * R - w + 1) & (i <= 2 * R - 1)
    return np.where(np.stack([now, now | lap]), np.float32(np.inf),
                    np.float32(-np.inf)).astype(np.float32)


def _ring_block(block: int) -> None:
    if block != RING:
        raise ValueError(f"GPT-OSS's sliding rings are one attention block of {RING}: block "
                         f"{block}")


# =============================================================================== kernel
def _slide(m, pos):
    """A sliding layer's ring slot of position `pos` (an int or a RunPos) and its attention:
    the ring as one block, masked by its row of the mask table (smask_table); in the first lap
    at a run-time position read up to the sub-block of tpos (attention.Blocks' tpos)."""
    R, tpos = m.ring, None
    if isinstance(pos, RunPos):
        t = 0 if pos.blocks == 1 else 1
        slot = Affine(pos.t0 % R) + pos.tpos
        row = Affine(m.smask + 4 * (2 * R * t + R - pos.t0 % R)) + pos.tpos * -4
        if t == 0:                      # the first lap: slots 0 .. tpos (Bucket's tpos)
            tpos = pos.bucket.tpos
    else:
        t = 0 if pos < R else 1
        slot = pos % R
        row = Affine(m.smask + 4 * (2 * R * t + R - pos % R))
    return slot, Blocks([(0, R, Tensor(row, (R,), (1,)))], tpos)


def _attention(x, lw, c, s_, pos, m, block: int):
    """x + W_o attention(x) + b_o for one token, in place (qwen3._attention's schedule but for
    Q; W_o's rows the model's width, x's first columns): K, V and the queries projected with
    their biases, K and the queries RoPE'd and padded to D, K / V appended at the position's
    slot (a sliding layer's ring slot), each KV head's query group attending with its sinks
    (log2(e) times the checkpoint's: the scores' units). The first Q_EAGER groups' Q MMs come
    before the K appends (qwen3's reason: the sequencer's window), the later ones with their
    group's queries, `ahead` heads early (_attend_heads), so the MXU streams them while the
    VPU runs the heads before (all eight ahead of the heads left the MXU only the heads' 16
    small MMs meanwhile; the same MMs and values in another order). At S > 1 the slice's KV
    heads and query groups (KVDesc's), their outputs all-gathered (slice-major: W_o's
    columns' order) before W_o's rows of the slice, x's columns of the slice all-gathered."""
    spec = m.spec
    d, G = spec.head_dim, spec.n_q // spec.n_kv
    kv = lw.kv
    xs = ol.quantize(rmsnorm(x, ol.load(lw.g_in), spec.eps, m.norm_n))
    k = ol.dot(xs, lw.wk) + ol.load(lw.bk)[None, :]             # [1, nkv*d]
    v = ol.dot(xs, lw.wv) + ol.load(lw.bv)[None, :]
    heads = list(kv.owned_heads(spec.n_kv))
    nh = len(heads)
    src = [xs, ol.load(lw.bq)]                  # (the Q MMs' operands, to the last group's)

    def qproj(j):
        """KV group j's queries [1, G*d]: its Q MM and bias."""
        (xq, bq), r = src, slice(j * G * d, (j + 1) * G * d)
        q = ol.dot(xq, lw.wq[r, :]) + bq[r][None, :]
        if j == nh - 1:
            src.clear()                         # (ACT RAM: the input's blocks free for W_o's)
        return q
    qps = {j: qproj(j) for j in range(min(nh, Q_EAGER))}
    del xs
    sk = ol.load(lw.sinks) * ol.LOG2E
    if lw.kind == SLIDE:
        kpos, seq = _slide(m, pos)
    else:
        kpos = pos.pos if isinstance(pos, RunPos) else pos
        seq = pos.bucket if isinstance(pos, RunPos) else pos + 1
    kh = _rope_padded(k.reshape(nh, d), c, s_)                   # [nkv, D]
    vh = _padded(v.reshape(nh, d))
    for j, hh in enumerate(heads):
        ol.kv_append(kv, hh, kpos, kh[j:j + 1, :], None)

    def queries(j):
        def emit():
            ol.kv_append(kv, heads[j], kpos, None, vh[j:j + 1, :])
            q = qps.pop(j) if j in qps else qproj(j)
            return _rope_padded(q.reshape(G, d), c, s_)
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
    o_row = ol.empty([1, G * nh * d])
    o_loc = o_row.reshape(G * nh, d)

    def emit(i, acc, l):                        # (a head's output: one head late, ATTN_LAG)
        j, g0, g1 = parts[i]
        o_loc[j * G + g0:j * G + g1, :].set(acc / l[:, None])
    _attend_heads(qhs, kv, [heads[j] for j, _, _ in parts], seq, block, ol.LOG2E / math.sqrt(d),
                  depth=ATTN_DEPTH, ahead=2, emit=emit, lead=ATTN_LEAD, lazy=ATTN_LAZY, lag=ATTN_LAG,
                  share=True, sinks=[sk[j * G + g0:j * G + g1] for j, g0, g1 in parts])
    del sk
    h = lw.wo.shape[0]                          # (the model's width, S > 1 a slice's: x's padding
    #                                            stays zero)
    S = ol.num_programs()
    if S == 1:
        x[:, 0:h].set(x[:, 0:h] + ol.dot(o_row, lw.wo) + ol.load(lw.bo)[None, :])
        return x
    o_all = ol.all_gather(o_row)                # [1, n_q d]: every slice's heads
    del o_row, o_loc
    sid = ol.program_id()
    out = x[:, sid * h:(sid + 1) * h] + ol.dot(o_all, lw.wo) + ol.load(lw.bo)[None, :]
    del o_all
    b = current()
    b.emit(I.gather(out.base, x.base, 1, h, h, x.rs, h, comment="all-gather into x"))
    b.bump_version(x.buf)                       # (x's first S h columns; its padding stays zero)
    return x


@ol.jit
def gptoss_step(m, pos: int, block: int = ATTN_BLOCK, tok: int | None = None):
    """One decode token at position `pos`: its embedding row (from the image's table) ->
    logits. The layers run as m.plan's runs (a hardware loop over (sliding, full) pairs), each
    attention then the MoE (moe.moe_ffn: routed on the card, the experts from their slots)."""
    fill_logits(m)
    spec = m.spec
    x, c, s_ = _inputs(m, pos, tok)

    def layer(li, it):
        lw = m.layer(li, it)
        _attention(x, lw, c, s_, pos, m, block)             # (both add to x in place)
        MO.moe_ffn(x, lw, spec.moe, m.moe_dev, spec.eps, n=m.norm_n)

    run_layers(m.plan, layer)
    _lm_head(x, m, spec)
