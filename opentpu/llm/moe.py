"""Mixture-of-experts FFN with its experts streamed into a DRAM cache: path (a) of
docs/offload.md. The card routes and computes every expert; opentpu.host.offload's server only
moves experts into per-layer LRU slots in the card's DRAM and keeps the directory.

A MoE layer on the device (`moe_ffn`), one token:

1. the router: an MM of the normalized residual -> E logits (and, with a shared expert, its
   gate's logit as row E of the same matrix), and the model's rule (MoESpec.rule);
2. the k best by ARGMAX with knock-out (in order, ties to the first), their global ids
   (the layer's base j * E + index) and weights: LFM2's sigmoid scores, chosen with the
   expert bias, renormalized and scaled; Qwen's softmax of the k largest logits;
3. the fence, WAITW served >= seq (the host has finished every earlier request: none of this
   layer's slots is being replaced, and the one request row and answer are free), then the
   ids and seq + 1 to the mailbox; then what needs no routed expert, while the host streams
   the missing ones: a shared expert (a dense SwiGLU of the layer block, times sigmoid of its
   gate's logit; docs/offload.md 15) and beside()'s work;
4. the directory: each id's present flag;
5. the experts present (a LOOP over the k ids, a LOOP R[present] inside), then the others
   (LOOP R[1 - present]). A present one takes its slot address from its directory entry, a
   missing one from its word of the request's answer (docs/offload.md 10.11), each by WAITW
   (!= 0: at once for an expert present; for the others once the host has written the
   answer) to a TMEM word and RLD (raw) into a register; a missing one then waits for its
   slot's tag word (WAITW != 0: the host writes it in its expert's DMA's last beat). Each
   zeroes its slot's tag (the next nonzero tag there is a later expert's), runs
   kernels.mlp.swiglu_down there and writes its weighted output to its row of a [k, H] tile;
   then the answer's words are zeroed (the host writes the next request's answer only after
   its post);
6. the rows summed in the router's order (the result does not depend on what the cache held),
   then the shared expert's output added (last, as before it was computed early: bit for bit
   the same sum), and the whole to the residual.

It holds one register (the resident decode's run arguments and the layer loops hold the
rest): the per-expert values are the columns of a small tile that each loop rotates, so
expert i is always at column 0 in iteration i.

With MoESpec.hint the layer also posts a prefetch hint before its mixer (`moe_hint`,
docs/offload.md section 12): its router on the layer's input, the k best as a request whose
ids are offset by layers x E; the host replaces their slots' victims at once and moves the
missing experts in the link's idle time, and the route's own request finds them landed or on
their way.

`moe_ffn_rows` is the layer on R rows at once (layer-major prefill, docs/offload.md section 13):
each row routes as moe_ffn's; one request carries the rows' R x k ids (the count at mbox + 4);
the card finds their union (each id's first place among them) and runs each union expert once
on all R rows, storing each row's unweighted output to its (row, rank) place in a DRAM scratch
of [R k + 1, H] (R k <= 16, the request's one line; the last row a sink for the outputs no row
chose); each row then sums its own k in its router's order, so a row's result is moe_ffn's bit
for bit. `moe_hint_rows` is moe_hint on R rows: a layer-major run's hint for the next layer, its
router on the run's output rows (section 13.9).

Expert-major (sections 13.11, 13.13) splits a layer-major MoE layer over a chunk's runs: each
run's `moe_prologue_rows` routes its rows, keeps their ids, weights and norm in the rows' records
of a scratch (em_record; the server carves it from the first expert slots for the prefill), and
posts the experts as a need line; then one `moe_expert_run` a layer runs each expert its rows
chose once, in passes of up to C rows (em_pass_rows, docs/offload.md 13.18; ld-memch's
bucketing), each pass waiting on its directory entry; the next layer's runs end the layer (`moe_combine_rows`). A row's arithmetic is
moe_ffn_rows', bit for bit.

`ExpertFormat` is one expert's slot: gate and up [F, H] and W_down's column parts, laid out as a
layer block's MLP (lfm2.Image), so the FFN is kernels.mlp.swiglu_down at the slot's address
(a compiler.DevVar: the register RLD sets).

At S > 1 (several slices, each with its own DRAM: `_moe_ffn_tp`) the experts are tensor-parallel
in a resident pool: slice s holds every expert's part at the same pool offset (ExpertFormat S),
its rows of W_gate and W_up (F / S) and of W_down (H / S output columns, the whole K). The
router's rows are split the same way (E + 1 rows, zero-padded to S whole groups) and gathered
(one GATHER of S Er words); every slice then routes as moe_ffn does, on the same logits. Each
expert's activations for the slice's F rows go to its row of a [k, F / S] tile, gathered once
into [k, F] (in dev.groups groups), and each expert's down MMs read the whole F in S = 1's
column parts (a QACT a part), so every QACT range, every MM and every add is S = 1's: the
layer's output is S = 1's bit for bit. The shared expert likewise (its activations gathered
before its down, in S = 1's parts); the slice's output columns, x + the FFN, are gathered into
x. Data moved a layer: the router's S Er words, k F words (and the shared expert's F) of
activations and H words of output, each slice sending its 1 / S; no weight crosses slices.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .. import isa as I
from .. import language as ol
from .. import quant as Q
from ..compiler import Affine, CompileError, DevVar, QTensor, Tensor, current
from ..host.offload import LINE, SPLIT, ExpertServer, Layout, PoolFile, dram_of, to_split
from ..kernels import mailbox as MB
from ..kernels.lib import gelu_tanh, rmsnorm, sigmoid, silu
from ..kernels.mlp import _chunk, swiglu_down

NEG = -3.0e38                       # knocked out (below every score)
ACTS = {"silu": silu, "gelu_tanh": gelu_tanh}


@dataclass(frozen=True)
class MoESpec:
    E: int                 # experts per layer
    k: int                 # experts per token
    ffn: int               # an expert's width
    first: int = 0         # the first MoE layer (the ones before it have a dense MLP)
    rule: str = "sigmoid_bias"       # LFM2: sigmoid scores, the choice on score + expert bias;
                                     # "softmax": Qwen, the k largest logits' softmax
    norm: bool = True                # renormalize the chosen weights (softmax: always)
    scale: float = 1.0               # then multiply them by this
    shared: int = 0                  # a shared expert's width (Qwen: the layer's dense MLP,
                                     # times sigmoid(x . w_gate)), 0: none
    act: str = "silu"                # the experts' gate: SiLU, "gelu_tanh" (Gemma 4), or "oai"
                                     # (GPT-OSS: oai_glu's clamped SwiGLU, limit and alpha)
    hint: bool = False               # each layer posts its router's k best on its input as a
                                     # prefetch hint before its mixer (moe_hint; Qwen3.5)
    bias: bool = False               # GPT-OSS: the router's logits and each expert's gate, up and
                                     # down projections take a bias (lw.rbias, the slot's: the
                                     # ExpertFormat's bias)
    limit: float = 7.0               # act "oai": the clamp (GPT-OSS's swiglu_limit)
    alpha: float = 1.702             # act "oai": the gate's sigmoid slope

    def __post_init__(self):
        if self.rule not in ("sigmoid_bias", "softmax"):
            raise ValueError(f"MoE router rule {self.rule!r}")
        if self.act not in ACTS and self.act != "oai":
            raise ValueError(f"MoE expert activation {self.act!r}")
        if self.rule == "softmax" and not self.norm:
            raise ValueError("a softmax router over all experts is not supported (norm=False)")


class _Now:
    """ExpertFormat.pack's batch: each matrix quantized at once (quant.quantize_mxu)."""

    @staticmethod
    def submit(W, fmt: str, D: int, done) -> None:
        done(*Q.quantize_mxu(W, fmt, D))


class ExpertFormat:
    """One expert in its slot: W_gate, W_up [F, H] (rows, then their scale rows) and W_down
    [H, F] in column parts of `C` (swiglu_down's chunks; its rows quantized whole, each part's
    scale rows the row's Sw and its blocks': quant.column_parts), each D-byte aligned. A width
    that is not a whole number of the format's chunks (Gemma 4's 704) is padded with zero rows
    of W_gate and W_up and zero columns of W_down to `F` (768): exact, act(0) * 0 = 0.
    grp > 1 (GALL, docs/weight_groups.md): each matrix (each W_down part) in groups
    of the format's rows for a machine of WG grp (quant.grouped), every one P = grp x D aligned
    in the slot; a slot at A holds them as quant.hash_rows(pack(), A) on a machine with the
    channel hash (the expert server's writes). bias (GPT-OSS): W_gate's and W_up's biases [Fg]
    and W_down's [Hd] after the matrices, fp32 (bg, bu, bd; zeros in the padding). kt
    (Config.KT, in groups): a matrix whose own K ends in the first half of its last D-block
    (quant.kt_fits: GPT-OSS's W_gate / W_up at Hd 2880 of H 2944, W_down's last part at 64 of
    128) in MM KT's pairs (docs/isa.md "K tail"; kts: W_gate's, W_up's, then each W_down part's).

    tight (GPT-OSS: no padding a token streams): W_gate and W_up hold the expert's own rows in
    whole groups (Fg: 2880, not 3072), W_down's K is the width in whole D-blocks (F: 2944; 4-bit
    in groups streams a block a chunk) in parts of C with a narrower last one (cw: 11 x 256 +
    128), and `rows` W_down's rows (Hd: the model's hidden size, 2880 where H is the image's
    2944). Untight, Fg = F, Hd = H and every part is C wide.

    S > 1 (tensor-parallel experts, moe_ffn at S > 1): the slot is one slice's part of the
    expert, every slice's at the same offsets: W_gate's and W_up's rows Fs = Fg / S of slice s
    (s Fs ..), W_down's rows Hs = Hd / S of slice s (its output columns) at the whole K F in the
    same column parts, the biases' matching entries. Rows are quantized on their own and the
    parts start at whole groups, so a part's bytes are the whole slot's rows (pack's part)."""

    def __init__(self, H: int, F: int, D: int, wformat: str, grp: int = 1, bias: bool = False,
                 rows: int | None = None, tight: bool = False, kt: bool = False, S: int = 1):
        c = D if wformat == "int8" else 2 * D      # a W_down part starts on a whole chunk
        R = Q.grp_r(wformat, grp)
        self.F0 = F                                 # the expert's own width
        if tight:
            kq = D if wformat == "int8" or grp > 1 else c
            F, self.Fg = -(-F // kq) * kq, -(-F // R) * R
            self.C = _chunk(-(-self.F0 // (2 * D)) * (2 * D), D, c)
        else:
            q = c * R // math.gcd(c, R)             # (in groups: whole groups of rows too)
            F = self.Fg = -(-F // q) * q            # the slot's
            self.C = _chunk(F, D, c)
        self.H, self.F, self.D, self.wformat, self.grp = H, F, D, wformat, grp
        self.Hd, self.tight = H if rows is None else rows, tight
        self.cw = [min(self.C, F - c0) for c0 in range(0, F, self.C)]   # W_down's parts
        if H % R or self.Hd % R:
            raise ValueError(f"an expert of width {H} ({self.Hd} rows) in groups of {R} rows")
        if self.Fg % (S * R) or self.Hd % (S * R):
            raise ValueError(f"an expert of {self.Fg} rows ({self.Hd} down) on {S} slices: a "
                             f"slice's part in whole groups of {R} rows")
        self.S, self.Fs, self.Hs = S, self.Fg // S, self.Hd // S    # (a slice's rows)
        a, align = 0, max(LINE, D * grp)    # the MXU streams whole D-byte (group) chunks
        kt = kt and grp > 1 and wformat != "nvfp4"
        self.kts = [kt and Q.kt_fits(self.Hd, D)] * 2 + [   # (W_gate's own K: the model's
            kt and Q.kt_fits(max(0, min(w, self.F0 - c0)), D)       # hidden size, Hd)
            for c0, w in zip(range(0, F, self.C), self.cw)]
        mats = [(self.Fs, H)] * 2 + [(self.Hs, w) for w in self.cw]     # (a slice's rows)

        def take(n):
            nonlocal a
            at, a = a, a + -(-n // align) * align
            return at

        def mat(n, k, t):                   # an [n, k] matrix's data and scales
            d, sc = Q.grp_bytes(n, k, wformat, D, grp, t)
            return take(d), take(sc)
        self.wg, self.wu, *self.wd = (mat(n, k, t) for (n, k), t in zip(mats, self.kts))
        self.bias = bias
        self.bg = self.bu = self.bd = None
        if bias:
            self.bg, self.bu, self.bd = take(4 * self.Fs), take(4 * self.Fs), take(4 * self.Hs)
        self.nbytes = a
        # (grp > 1) the matrices' data in the slot: (offset, bytes), whole P-byte chunks
        self.data = [(m[0], Q.grp_bytes(n, k, wformat, D, grp, t)[0]) for m, (n, k), t in
                     zip([self.wg, self.wu] + self.wd, mats, self.kts)]

    def ranges(self) -> list[int]:
        """The down projection's QACT ranges over the expert's own width F0, for an emulation
        of the unpadded expert: a chunk of C each, the last cut at F0 (the slot's zero columns
        change no range's amax)."""
        return [min(self.C, self.F0 - c) for c in range(0, self.F0, self.C)]

    def pack(self, wg: np.ndarray, wu: np.ndarray, wd: np.ndarray, bias=None,
             part: int = 0) -> np.ndarray:
        """The slot's bytes for W_gate, W_up [F0, H] and W_down [Hd, F0] (float); bias: (b_gate,
        b_up [F0], b_down [Hd]) of a format with biases; part: the slice's (S > 1)."""
        got = []
        self.submit(_Now, wg, wu, wd, got.append, bias, part)
        return got[0]

    def submit(self, qb, wg: np.ndarray, wu: np.ndarray, wd: np.ndarray, done,
               bias=None, part: int = 0) -> None:
        """pack()'s bytes through a quantization batch (qcache.Batch: its worker processes and
        the 4-bit matrices' cache): done(the slot's bytes) once its three matrices are back,
        from this call, a later submit or qb.wait() (Batch's results are quantize_mxu's bit for
        bit, so the bytes are pack()'s). part: slice part's rows (S > 1)."""
        out = np.zeros(self.nbytes, np.uint8)
        if self.F != self.F0 or self.Fg != self.F0:     # the zero padding
            wg, wu = (np.pad(np.asarray(w, np.float32), ((0, self.Fg - self.F0), (0, 0)))
                      for w in (wg, wu))
            wd = np.pad(np.asarray(wd, np.float32), ((0, 0), (0, self.F - self.F0)))
        if self.S > 1:                                  # the slice's rows
            g, h = self._rows(part)
            wg, wu, wd = wg[g], wu[g], wd[h]
            bias = self._part_bias(bias, part)
        fm, D, left = self.wformat, self.D, [3]
        self._biases(out, bias)

        def back():
            left[0] -= 1
            if not left[0]:
                done(out)

        def down(q, s):                     # W_down's rows quantized whole (one Sw a row)
            self._down(out, (q, s))
            back()
        qb.submit(wg, fm, D, lambda q, s: (self._put(out, self.wg, (q, s), self.H,
                                                     self.kts[0]), back()))
        qb.submit(wu, fm, D, lambda q, s: (self._put(out, self.wu, (q, s), self.H,
                                                     self.kts[1]), back()))
        qb.submit(np.asarray(wd, np.float32), fm, D, down)

    def pack_q(self, g: tuple, u: tuple, d: tuple, bias=None, part: int = 0) -> np.ndarray:
        """The slot's bytes for W_gate, W_up [Fg, H] and W_down [Hd, F] already in the slot's
        format, each (rows, scale rows) as quantize_mxu gives them (or an exact import's:
        quant.import_mxfp4), at the slot's widths; W_down stored in column parts of C (the last
        cw[-1]). bias: pack's (zero-padded to Fg). part: the slice's (S > 1)."""
        out = np.zeros(self.nbytes, np.uint8)
        if self.S > 1:                                  # the slice's rows
            gr, hr = self._rows(part)
            g, u = ((a[gr], b[gr]) for a, b in (g, u))
            d = (d[0][hr], d[1][hr])
            bias = self._part_bias(bias, part)
        self._biases(out, bias)
        self._put(out, self.wg, g, self.H, self.kts[0])
        self._put(out, self.wu, u, self.H, self.kts[1])
        self._down(out, d)
        return out

    def _rows(self, part: int) -> tuple:
        """Slice part's rows of W_gate / W_up (Fs of Fg) and of W_down (Hs of Hd)."""
        if not 0 <= part < self.S:
            raise ValueError(f"part {part} of {self.S} slices")
        return (slice(part * self.Fs, (part + 1) * self.Fs),
                slice(part * self.Hs, (part + 1) * self.Hs))

    def _part_bias(self, bias, part: int):
        """pack's biases (b_gate, b_up [<= Fg], b_down [Hd]) at slice part's entries."""
        if bias is None:
            return None
        g, h = self._rows(part)
        bg, bu = (np.pad(np.asarray(b, np.float32), (0, self.Fg - len(b)))[g] for b in bias[:2])
        return bg, bu, np.asarray(bias[2], np.float32)[h]

    def _put(self, out, pair, qs, k, kt=False) -> None:
        """A matrix's (rows, scale rows) in groups (Q.grouped; kt: KT's pairs) at its pair of
        offsets."""
        for at, v in zip(pair, Q.grouped(*qs, self.wformat, k, self.grp, self.D, kt=kt)):
            v = np.ascontiguousarray(v).view(np.uint8).reshape(-1)
            out[at:at + v.size] = v

    def _down(self, out, d) -> None:
        """W_down's (rows, scale rows) in column parts of C (the last cw[-1])."""
        fm, D = self.wformat, self.D
        if self.tight:                          # (quant.column_parts', the last part narrower)
            parts = [(Q.part_rows(d[0], fm, c, c + w, D), Q.part_scales(d[1], fm, c // D, w // D))
                     for c, w in zip(range(0, self.F, self.C), self.cw)]
        else:
            parts = Q.column_parts(*d, fm, self.F, self.C, D)
        for pair, qs, w, kt in zip(self.wd, parts, self.cw, self.kts[2:]):
            self._put(out, pair, qs, w, kt)

    def _biases(self, out, bias) -> None:
        from .. import fp32 as FP
        if (bias is None) != (not self.bias):
            raise ValueError("an expert's biases: given to a format with them, and only to one")
        for at, v in zip((self.bg, self.bu, self.bd), bias or ()):
            v = np.ascontiguousarray(FP.ftz(np.asarray(v, np.float32))).view(np.uint8)
            out[at:at + v.size] = v

    def descs(self, base) -> SimpleNamespace:
        """wg, wu, wd QTensors of the expert whose slot starts at `base` (an address
        expression: a DevVar for a slot found at run time)."""
        H, F, D, C, fm = self.H, self.F, self.D, self.C, self.wformat
        Fg, Hd = self.Fs, self.Hs                       # (a slice's: S > 1)
        base = Affine.of(base)

        def q(pair, n, k, kt):
            return QTensor.weight(base + pair[0], base + pair[1], n, k, fm, D, grp=self.grp,
                                  kt=kt)
        parts = tuple(q(p, Hd, w, kt) for p, w, kt in zip(self.wd, self.cw, self.kts[2:]))
        wd = QTensor(parts[0].data, parts[0].scale, (Hd, F), parts[0].rs, parts[0].srs, D,
                     parts=parts, pw=C, wf=parts[0].wf, grp=parts[0].grp)
        ns = SimpleNamespace(wg=q(self.wg, Fg, H, self.kts[0]), wu=q(self.wu, Fg, H, self.kts[1]),
                             wd=wd, bg=None, bu=None, bd=None)
        if self.bias:
            ns.bg, ns.bu = (Tensor(base + a, (Fg,), (1,)) for a in (self.bg, self.bu))
            ns.bd = Tensor(base + self.bd, (Hd,), (1,))
        return ns


def gemma_expert(W, p: str, e: int):
    """Gemma 4's routed expert e of the layer at prefix p (`p + "experts.gate_up_proj"` [E, 2F,
    H], gate rows first; `p + "experts.down_proj"` [E, H, F]) as W_gate, W_up [F, H] and W_down
    [H, F] for ExpertFormat.pack, router.per_expert_scale[e] folded into W_down (it multiplies
    the expert's output). pre_feedforward_layernorm_2's gain stays out: moe_ffn applies it to
    the experts' input before quantizing it (lw.g_exp; docs/offload.md 11.2). A lazy W with
    `part` reads the one expert."""
    k = p + "experts."
    get = getattr(W, "part", None)
    gu, dn = ((get(k + "gate_up_proj", e), get(k + "down_proj", e)) if get is not None else
              (W[k + "gate_up_proj"][e], W[k + "down_proj"][e]))
    gu, dn = np.asarray(gu, np.float32), np.asarray(dn, np.float32)
    F = gu.shape[0] // 2
    s = np.asarray(W[p + "router.per_expert_scale"], np.float32)[e]
    return gu[:F], gu[F:], dn * s


def gemma_router(W, p: str) -> np.ndarray:
    """Gemma 4's router [E, H] on the residual's unit RMSNorm (moe_ffn's input): router.proj
    with router.scale and H^-0.5 folded into its columns."""
    w = np.asarray(W[p + "router.proj.weight"], np.float32)
    return w * (np.asarray(W[p + "router.scale"], np.float32) * np.float32(w.shape[1] ** -0.5))


# ---------------------------------------------------------------------------- the pool's key
def quant_source(wformat: str) -> str:
    """The quantizer's code an expert's bytes depend on: quant.py and fp32.py (int8x's
    quantize_mx, the 4-bit formats' scale arithmetic). Not numpy's version (qcache._source's):
    a pool is packed on one host and read on another."""
    from .. import fp32 as F
    h = hashlib.blake2b(digest_size=16)
    for f in (Q.__file__, F.__file__):
        h.update(Path(f).read_bytes())
    return h.hexdigest()


def checkpoint_id(model_dir) -> str | None:
    """A checkpoint's identity: its config.json's hash (strip_experts.py copies the file as it
    is, so the card host's checkpoint without the experts has the packing host's); None
    without one."""
    f = Path(model_dir) / "config.json" if model_dir is not None else None
    if f is None or not f.exists():
        return None
    return hashlib.blake2b(f.read_bytes(), digest_size=16).hexdigest()


def pool_key(image, model_dir=None, mtp_from=None) -> dict:
    """The key of `image`'s expert pool (`<pool>.key`, JSON): what decides its experts' bytes,
    the MXU's numerics (quant.MX_NUMERICS: key 2 and on; a key 1 pool holds the scale words of
    the MXU before MX, docs/mx_mxu.md), the quantizer's code (quant_source), the format, D, the
    layout (the pool's and the slot's: ExpertFormat's parts; its groups, "grp", when > 1:
    GALL's experts, the same offsets in other bytes) and the checkpoint (checkpoint_id);
    "numpy" and "model" are notes, not compared. mtp_from: the checkpoint an
    MTP image's MTP layer (its experts the pool's last layer) comes from when not the model's
    (LazyWeights.mtp_from: MiniCPM-V 4.7's, docs/offload.md 15.6), its checkpoint_id as
    "mtp_from". An image with entries past the experts (`pool_extra`: (their count, their key),
    MiniCPM-V's vision pieces staged through the slots, docs/minicpmv.md 4.2) adds "vision":
    their key, which open_pool compares on its own (key_diff does not: the experts' part of a
    pool is the same)."""
    f, L = image.fmt, image.offload
    key = {"key": 2, "numerics": Q.MX_NUMERICS, "quant": quant_source(f.wformat),
           "format": f.wformat, "D": f.D,
           "layout": {"layers": L.layers, "E": L.E, "slot_bytes": L.slot_bytes, "H": f.H,
                      "F": f.F0, "C": f.C, "wg": list(f.wg), "wu": list(f.wu),
                      "wd": [list(x) for x in f.wd]},
           "checkpoint": checkpoint_id(model_dir), "numpy": np.__version__,
           "model": None if model_dir is None else Path(model_dir).name}
    if getattr(f, "grp", 1) > 1:
        key["layout"]["grp"] = f.grp
    if getattr(f, "tight", False):
        key["layout"].update(tight=True, Hd=f.Hd)
    if getattr(f, "S", 1) > 1:
        key["layout"]["S"] = f.S
    if mtp_from is not None:
        key["mtp_from"] = checkpoint_id(mtp_from)
    if getattr(image, "pool_extra", None) is not None:
        key["vision"] = image.pool_extra[1]
    return key


def key_diff(have: dict, want: dict) -> list:
    """The fields in which a pool's key `have` differs from `want` (the checkpoint only when
    both know it; the MTP layer's checkpoint, mtp_from, whenever either names one)."""
    out = [k for k in ("key", "numerics", "quant", "format", "D", "layout")
           if have.get(k) != want.get(k)]
    if have.get("mtp_from") != want.get("mtp_from"):
        out.append("mtp_from")
    if None not in (have.get("checkpoint"), want.get("checkpoint")) and \
            have["checkpoint"] != want["checkpoint"]:
        out.append("checkpoint")
    return out


def check_pool_key(pool_file, want: dict) -> None:
    """Refuse an expert pool whose key (`<pool>.key`) is not `want`: its experts were packed by
    another quantizer, for the MXU before MX, in another format or layout, or from another
    checkpoint. A pool without a key is refused too: it was packed before keys, so before the
    MX numerics (its scale words are the former MXU's); pack_pool stamp gives one its key once
    `sums` shows it holds this tree's bytes."""
    kf = Path(str(pool_file) + ".key")
    if not kf.exists():
        raise ValueError(
            f"{pool_file} has no key file: a pool packed before keys, for the MXU before MX "
            f"(docs/mx_mxu.md), whose experts' bytes are not this image's. Repack it: remove "
            f"{pool_file} and its .packed and .format, then python3 tools/offload/pack_pool.py "
            f"MODEL {pool_file} init and its pack workers (or let the run pack the experts it "
            f"asks for); a pool known to hold this tree's bytes (pack_pool.py ... sums) gets "
            f"its key by pack_pool.py MODEL {pool_file} stamp")
    have = json.loads(kf.read_text())
    diff = key_diff(have, want)
    if diff:
        note = {k: (have.get(k), want.get(k)) for k in diff if k != "layout"}
        raise ValueError(
            f"{pool_file} was packed for another model image ({', '.join(diff)} differ"
            f"{': ' + str(note) if note else ''}; {have.get('format')} D {have.get('D')} of "
            f"{have.get('model')}, this run is {want['format']} D {want['D']} of "
            f"{want.get('model')}): its experts' bytes are not this image's. Repack it: remove "
            f"{pool_file} and its .packed, .format and .key, then python3 "
            f"tools/offload/pack_pool.py MODEL {pool_file} init and its pack workers (or let "
            f"the run pack the experts it asks for)")


def pool_entries(path: Path, n: int) -> int:
    """The entries a pool file holds by its key: n experts and its "vision" key's entries."""
    kf = Path(str(path) + ".key")
    have = json.loads(kf.read_text()) if kf.exists() else {}
    return n + int((have.get("vision") or {}).get("entries", 0))


def open_pool(layout: Layout, pool_file, mapped: bool = True,
              key: dict | None = None, extra: int = 0) -> PoolFile:
    """The expert pool file for `layout`, opened, and its packed experts' read into the page
    cache started (PoolFile.warm: the host's RAM tier; the Engine opens it before it builds the
    image, so the read runs during the build). The file is the whole pool's size (sparse until
    packed), and `<pool_file>.packed` marks the experts in it (a file of the right size without
    one is a pool packed whole). A new file is in the split format
    (opentpu.host.offload.split_order), as `<pool_file>.format` says; a file without it holds
    the slot format. With `key` (pool_key) a new file gets it as `<pool_file>.key`, and an
    existing one must have it (check_pool_key; one without a key file is refused).
    The page cache stays the kernel's to reclaim: nothing is pinned. mapped: PoolFile's (its
    reads touched through a read-only map: docs/offload.md 10.7).

    extra: entries past the experts (layers x E on; key["vision"] their key: pool_key), each a
    slot's bytes. A pool whose own extra entries are not these (none, or another key's) keeps
    its experts and gets these, not packed yet (the file grown or cut past its experts, its
    key's "vision" replaced); a run without extra opens a pool with them as its experts'."""
    L = layout
    n = L.layers * L.E
    N = n + extra
    path = Path(pool_file)
    done = Path(str(path) + ".packed")
    kf = Path(str(path) + ".key")
    fresh = not path.exists() or path.stat().st_size != pool_entries(path, n) * L.slot_bytes
    if not fresh and key is not None:
        check_pool_key(path, key)
    want_x = (key or {}).get("vision") if extra else None
    have_x = json.loads(kf.read_text()).get("vision") if not fresh and kf.exists() else None
    if fresh:
        with open(path, "wb") as f:
            f.truncate(N * L.slot_bytes)
    elif extra and have_x != want_x:    # its extra entries replaced, its experts kept
        with open(path, "r+b") as f:
            f.truncate(n * L.slot_bytes)
            f.truncate(N * L.slot_bytes)
        pk = np.fromfile(done, np.uint8)[:n] if done.exists() else np.ones(n, np.uint8)
        done.write_bytes(np.concatenate([pk, np.zeros(N - n, np.uint8)]).tobytes())
        have = json.loads(kf.read_text()) if kf.exists() else dict(key)
        have["vision"] = want_x
        kf.write_text(json.dumps(have, indent=1) + "\n")
    M = pool_entries(path, n) if not fresh else N      # (the file's entries)
    if fresh or not done.exists() or done.stat().st_size != M:
        done.write_bytes(bytes(M) if fresh else bytes([1]) * M)
    fmt = Path(str(path) + ".format")
    if fresh:
        fmt.write_text(SPLIT + "\n")
        if key is not None:
            kf.write_text(json.dumps(key, indent=1) + "\n")
        elif kf.exists():
            kf.unlink()
    pf = PoolFile(path, L.slot_bytes, fmt.exists() and fmt.read_text().strip() == SPLIT,
                  mapped=mapped)
    pf.arr = np.memmap(path, np.uint8, "r+", shape=(N, L.slot_bytes))
    pf.packed = np.memmap(done, np.uint8, "r+", shape=(N,))
    pf.ids = np.nonzero(np.asarray(pf.packed))[0]
    pf.resident_at_open = pf.resident(pf.ids)
    pf.warm(pf.ids)
    return pf


def rewarm(pf: PoolFile):
    """The pool's experts that left the page cache since the open's read began, read into it
    again; the Engine calls it once the image is built and the build's memory freed. A 35B
    MoE image's build peaks at 28-30 GB of the card host's 31 and takes the pool out of the
    page cache again (docs/minicpmv.md 6, pooldisk: its turns then read their experts from the
    disk, ~15-32 ms each against 0.5 ms). The open's read is stopped (it may still run: an
    image from the image cache), then those of the packed experts with bytes out of the page
    cache (mincore) are read in file order on a new thread. Returns the warm thread now: the
    open's when it had ended and nothing left the page cache, or where the host cannot tell
    (no mincore: the open's read is left to run)."""
    t = pf.warm_t
    if pf._pages() is None:                 # (no mincore here; and no map of its own made:
        return t                            # pool_map False keeps none)
    stopped = t is not None and t.is_alive()
    if stopped:
        t.stop.set()
        t.join()
    gone = pf.absent(pf.ids)
    if gone is None and stopped:            # (mincore failed: all of them again)
        gone = pf.ids
    # (a stopped read with nothing left out: an empty one, so the record says done)
    return t if gone is None or not (len(gone) or stopped) else pf.warm(gone)


def serve(layout: Layout, expert, backend, pool_file=None, warm=True,
          policy: str = "lfu", lanes: tuple | None = None, entry_lanes=None) -> ExpertServer:
    """The host's expert server on the backend's DRAM (slice 0); with `warm`, the slots filled
    with each layer's first experts. expert(g): global expert g's slot bytes. An expert is
    packed when it is first asked for and kept in host RAM, or, with `pool_file` (a path, or
    open_pool's PoolFile), in that file (the page cache, or the SSD tier), which keeps it for
    later runs (`server.pool`: the PoolFile). `policy`: the slots' replacement (ExpertServer's;
    least decayed use, docs/offload.md 5.5); lanes, entry_lanes: ExpertServer's (the slots'
    grouped data on a machine with the channel hash)."""
    L = layout
    pf = None
    if pool_file is None:
        cache: dict = {}

        def pool(g):
            if g not in cache:
                cache[g] = expert(g).tobytes()
            return cache[g]
    else:
        pf = pool_file if isinstance(pool_file, PoolFile) else open_pool(L, pool_file)

        def pool(g):                    # read from the file (the memmap's writes are in the
            if not pf.packed[g]:        # page cache preadv reads)
                pf.arr[g] = to_split(expert(g)) if pf.split else expert(g)
                pf.packed[g] = 1
            return pf.get(g)
    srv = ExpertServer(dram_of(backend, L), L, pool, policy=policy, lanes=lanes,
                       entry_lanes=entry_lanes)
    srv.pool_file = pf
    if pf is not None:                  # (its reads' touches after each request is served)
        pf.defer_touch = True
    srv.pool_warm = None if pf is None else pf.warm_t
    srv.load([j * L.E + e for j in range(L.layers) for e in range(L.E)] if warm else ())
    return srv


# ---------------------------------------------------------------------------- reference
def route(logits: np.ndarray, bias: np.ndarray | None, mo: MoESpec):
    """The model's choice for one token's router logits [E]: (ids in the device's order, their
    weights). The k largest (ties: the first) of: LFM2, sigmoid scores + bias, the chosen
    scores renormalized (+ 1e-6, as Hugging Face) and scaled; softmax (Qwen), the logits, the
    chosen ones' softmax (Hugging Face's softmax over all, renormalized over the k)."""
    lg = np.asarray(logits, np.float64)
    if mo.rule == "softmax":
        s, sel = lg, lg.copy()
    else:
        s = 1.0 / (1.0 + np.exp(-lg))
        sel = s + bias
    ids = []
    for _ in range(mo.k):
        i = int(np.argmax(sel))
        ids.append(i)
        sel[i] = -np.inf
    w = s[ids]
    if mo.rule == "softmax":
        w = np.exp(w - w[0])
        w = w / w.sum()
    elif mo.norm:
        w = w / (w.sum() + 1e-6)
    return ids, w * mo.scale


# ---------------------------------------------------------------------------- the device
def moe_hint(x, lw, mo: MoESpec, dev: SimpleNamespace, eps: float) -> None:
    """Path (a)'s prefetch hint (docs/offload.md section 12), before the layer's mixer: its
    router on the layer's input x through moe_ffn's norm, the k best by the model's rule (no
    weights), posted as a request whose global ids are offset by dev.hint_off (layers x E: the
    host's mark of a hint). The host writes served once it has replaced the hinted experts'
    victims, before their transfers, so the route's fence (moe_ffn) waits for no expert; an
    expert still on its way reads as missing there and is waited for by its answer word and
    its tag. lw and dev as moe_ffn's. One register. None with a resident pool."""
    if getattr(dev, "pool", None) is not None:
        return
    b = current()
    _plain(mo, "moe_hint")
    E, k = mo.E, mo.k
    xs = ol.quantize(rmsnorm(x, ol.load(lw.g_post), eps))
    lg = ol.dot(xs, lw.router)
    del xs
    sel, pr, ids, tmp = ol.empty((E,)), ol.empty((2,)), ol.empty((k,)), ol.empty((k,))
    if mo.rule == "softmax":
        sel.set(lg[0, 0:E])
    else:
        sel.set(sigmoid(lg[0, 0:E]) + ol.load(lw.ebias))
    del lg
    r = b.scratch()
    lp = b.begin_loop(k)                # the k best, as the route picks them
    b.emit(I.argmax(pr.base, sel.base, 1, E, comment="hint: best"))
    b.emit(I.rld(r, pr.base + 1, bound=(0, E - 1), comment="its index"))
    b.emit(I.vop(I.V_FILL, sel.base, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, NEG, ra=r,
                 comment="knock out"))
    if k > 1:
        tmp[0:k - 1].set(ids[1:k])
        ids[0:k - 1].set(tmp[0:k - 1])
    b.emit(I.vop(I.V_COPY, ids.base + k - 1, pr.base + 1, 0, 1, 1, 0, 0, 0, comment="id"))
    b.end_loop(lp)
    b.unscratch(r)
    ids.set(ids + ol.load(lw.gbase) + float(dev.hint_off))
    seq = MB.wait_served(dev.mbox)                  # MB.post, with the request's count
    ol.store(Tensor(Affine(dev.mbox + LINE), (k,), (1,)), ids)
    ol.store(Tensor(Affine(dev.mbox + 4), (1,), (1,)), ol.full((1,), float(k)))
    seq.set(seq + 1.0)
    ol.store(Tensor(Affine(dev.mbox), (1,), (1,)), seq)


def moe_hint_rows(x, lw, mo: MoESpec, dev: SimpleNamespace, eps: float) -> None:
    """moe_hint on R rows at once (x [R, H], R * k <= LINE / 4: one request row): each row's k
    best by the router of lw's layer, through its norm (moe_ffn's route), posted as one hint
    of the rows' R * k ids (repeats included) and their count. A layer-major prefill run's
    hint for the next layer (docs/offload.md 13.9): layer j's run on its output rows, with
    layer j + 1's lw, ahead of layer j + 1's runs. One register. None with a resident pool."""
    if getattr(dev, "pool", None) is not None:
        return
    b = current()
    _plain(mo, "moe_hint_rows")
    R, E, k = x.rows, mo.E, mo.k
    N = R * k
    if N > LINE // 4:
        raise CompileError(f"{R} rows of {k} experts: more ids than a request row holds")
    xs = ol.quantize(rmsnorm(x, ol.load(lw.g_post), eps))
    lg = ol.dot(xs, lw.router)                          # [R, E] (+ the shared expert's gate)
    del xs
    gid = ol.empty((R, k), dense=True)
    pr, tmp, ids = ol.empty((2,)), ol.empty((k,)), ol.empty((k,))
    off = ol.load(lw.gbase) + float(dev.hint_off)
    r = b.scratch()
    for q in range(R):                                  # each row's k best, as the route's
        sel = ol.empty((E,))
        if mo.rule == "softmax":
            sel.set(lg[q, 0:E])
        else:
            sel.set(sigmoid(lg[q, 0:E]) + ol.load(lw.ebias))
        lp = b.begin_loop(k)
        b.emit(I.argmax(pr.base, sel.base, 1, E, comment="hint: best"))
        b.emit(I.rld(r, pr.base + 1, bound=(0, E - 1), comment="its index"))
        b.emit(I.vop(I.V_FILL, sel.base, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, NEG, ra=r,
                     comment="knock out"))
        if k > 1:
            tmp[0:k - 1].set(ids[1:k])
            ids[0:k - 1].set(tmp[0:k - 1])
        b.emit(I.vop(I.V_COPY, ids.base + k - 1, pr.base + 1, 0, 1, 1, 0, 0, 0, comment="id"))
        b.end_loop(lp)
        del sel
        gid[q, :].set(ids + off)
    b.unscratch(r)
    del lg, ids, tmp, off
    seq = MB.wait_served(dev.mbox)                  # MB.post, with the request's count
    ol.store(Tensor(Affine(dev.mbox + LINE), (N,), (1,)), gid.reshape(1, N)[0, :])
    ol.store(Tensor(Affine(dev.mbox + 4), (1,), (1,)), ol.full((1,), float(N)))
    seq.set(seq + 1.0)
    ol.store(Tensor(Affine(dev.mbox), (1,), (1,)), seq)


def _ebound(dev):
    """A directory entry's byte offset (8 a global id): its RLD bound, if the device says how
    many ids there are."""
    n = getattr(dev, "ids", None)
    return (0, 8 * (n - 1)) if n else None


def slot_of(r: int, word, off: int, dev: SimpleNamespace, missing: bool,
            off_bound=None) -> None:
    """R[r] = the slot of the expert whose offset (TMEM word `off`: its directory entry's, or
    for a missing one its answer word's) the loop has at column 0, its data landed, and the
    slot's tag zeroed. Present: WAITW on its entry (at once). Missing: WAITW on its answer
    word (once the host has written the answer), then on the slot's tag (once the expert's
    DMA has landed: the host writes the tag in its last beat). The tag is zeroed for every
    expert the card uses, so a nonzero tag is always a later expert's (docs/offload.md 10.11).
    word: a one-word TMEM tile; off_bound: the offset's RLD bound (isa.rld); the slot's is the
    pool's (dev.slots)."""
    b = current()
    b.emit(I.rld(r, off, bound=off_bound, comment="answer word" if missing else "entry offset"))
    b.waitw(word, dev.answer if missing else dev.dir, 0, I.C_NE, ra=r,
            comment="wait: its slot (the answer)" if missing else "its slot")
    b.rld(r, word, raw=True, bound=getattr(dev, "slots", None), comment="its slot")
    if missing:
        b.waitw(word, dev.tag, 0, I.C_NE, ra=r, comment="wait: its data (the slot's tag)")
    ol.store(Tensor(Affine(dev.tag) + DevVar("expert slot", r), (1,), (1,)), ol.zeros((1,)))


def oai_glu(mo: MoESpec, ex: SimpleNamespace):
    """swiglu_down's glu of GPT-OSS's experts (MoESpec act "oai") for the expert whose slot
    descriptors are `ex` (ExpertFormat.descs): with the slot's gate and up biases (bias),
    g = min(g, limit), u = clip(u, -limit, limit), (u + 1) g sigmoid(alpha g)."""
    parts = _oai_factors(mo, ex)

    def glu(g, u, c0):
        a, b = parts(g, u, c0)
        return a * b
    return glu


def _oai_factors(mo: MoESpec, ex: SimpleNamespace, held=None):
    """oai_glu's two factors, u + 1 and g sigmoid(alpha g) (clamped, with the biases), in its
    order: their product is its activations' last VOP. held: the gate's and up's biases
    already in TMEM tiles (_moe_ffn_tp loads an expert's with its place), not loaded a chunk at
    a time."""
    def parts(g, u, c0):
        n = g.cols
        if held is not None:
            g = g + held[0][c0:c0 + n][None, :]
            u = u + held[1][c0:c0 + n][None, :]
        elif ex.bg is not None:
            g = g + ol.load(ex.bg[c0:c0 + n])[None, :]
            u = u + ol.load(ex.bu[c0:c0 + n])[None, :]
        gc = ol.minimum(g, mo.limit)
        uc = ol.maximum(ol.minimum(u, mo.limit), -mo.limit)
        return uc + 1.0, gc * sigmoid(gc * mo.alpha)
    return parts


def _plain(mo: MoESpec, what: str) -> None:
    """The prefill paths' experts are SwiGLU / GeGLU without biases."""
    if mo.bias or mo.act == "oai":
        raise CompileError(f"{what}: experts with biases or act 'oai' (GPT-OSS) run in moe_ffn "
                           f"only")


def resident_pool(bump, experts: int, fmt: ExpertFormat) -> SimpleNamespace:
    """A resident pool (an image's with every expert in DRAM: qwen35 / gptoss Image.high) of
    `experts` experts in fmt's bytes, from bump.next on: global expert g at base + g ebytes, no
    directory and no host server (moe_ffn's resident path)."""
    return SimpleNamespace(base=bump.alloc(experts * fmt.nbytes), ebytes=fmt.nbytes,
                           experts=experts)


def pool_build(pool: SimpleNamespace, submit, lanes, put, S: int = 1) -> None:
    """The resident pool's experts written by put(slice, address, bytes): expert g's bytes at
    base + g ebytes, as the expert server writes a slot (lanes: the image's slot_lanes; with the
    channel hash the matrices' lanes by their addresses, quant.hash_rows). submit(qb, g, done)
    makes them: done(its bytes), now or through the batch qb (a qcache.Batch:
    ExpertFormat.submit quantizes the matrices as the image's others, in worker processes and
    through the 4-bit matrices' cache; one process took 0.7 s an expert of Qwen3.5-35B-A3B's
    on a laptop, its 10,240 about two hours). S > 1 (ExpertFormat S): every slice's part at the
    same address of its own DRAM, submit(qb, g, done, s) making slice s's."""
    from .. import qcache as QC

    def place(at, s=0):
        def done(data):
            if lanes is not None:
                wg, D, own = lanes
                for o, n in own:
                    data[o:o + n] = Q.hash_rows(data[o:o + n], at + o, wg, D)
            put(s, at, data)
        return done
    qb = QC.Batch()
    for g in range(pool.experts):
        at = pool.base + g * pool.ebytes
        if S == 1:
            submit(qb, g, place(at))
        for s in range(S if S > 1 else 0):
            submit(qb, g, place(at, s), s)
    qb.wait()


def pool_dev(pool: SimpleNamespace, fmt: ExpertFormat, **kw) -> SimpleNamespace:
    """moe_ffn's device words of a resident pool (dev.pool its base, ebytes, ids: its experts),
    with the image's own (kw)."""
    return SimpleNamespace(pool=pool.base, ebytes=pool.ebytes, fmt=fmt, ids=pool.experts, **kw)


def resident(r: int, gid: int, dev: SimpleNamespace) -> None:
    """R[r] = the place of the expert whose global id (TMEM word `gid`, fp32) the loop has at
    column 0 in a resident pool (qwen35.Image.pool: every expert in DRAM, above 4 GiB): ADDI
    (RLD MUL(id, expert bytes), pool base), no directory and no host (docs/isa.md "Address
    bits"). The RLD's bound is the pool's (dev.ids experts: AW bits past 4 GiB), so the waits
    take R[r] as a place in the pool, not any address: unbounded, every expert MM waited for
    the DMA's and the quantizer's older work and the layer's next QSTs for the expert MMs."""
    b = current()
    for x in I.rld_mul(r, gid, dev.ebytes, comment="its id x expert bytes", aw=b.aw,
                       bound=(0, (dev.ids - 1) * dev.ebytes)):
        b.emit(x)
    b.emit(I.addi(r, r, dev.pool, comment="its place in the pool", aw=b.aw))


def em_slot(r: int, word: int, ra: int, dev: SimpleNamespace, wait=None, rd: int = 0,
            bound=None) -> None:
    """An expert-major pass's expert place, R[r], from its global id at T[R[ra] + word] (the
    pass table's expert column): in a resident pool (pool_dev with em_*) the
    pool's place (resident()'s ADDI of RLD MUL(id, expert bytes)), else the expert server's
    slot (bound), after a WAITW on its directory entry (R[rd], default r; wait: the entry's
    word), its slot's tag zeroed (moe_expert_run's)."""
    b = current()
    if getattr(dev, "pool", None) is not None:
        for x in I.rld_mul(r, word, dev.ebytes, ra=ra, comment="its id x expert bytes",
                           aw=b.aw, bound=(0, (dev.ids - 1) * dev.ebytes)):
            b.emit(x)
        b.emit(I.addi(r, r, dev.pool, comment="its place in the pool", aw=b.aw))
        return
    rd = rd or r
    b.emit(I.rld(rd, word, ra=ra, mul=8, comment="its directory entry"))
    b.waitw(wait, dev.dir, 0, I.C_NE, ra=rd, timeout=MB.TIMEOUT,
            comment="wait: its slot (the entry)")
    b.rld(r, wait, raw=True, bound=bound, comment="its slot")
    ol.store(Tensor(Affine(dev.tag) + DevVar("expert slot", r), (1,), (1,)), ol.zeros((1,)))


def em_slot_bound(dev: SimpleNamespace):
    """The bound of an expert-major pass's expert place (em_slot): the pool's, else the slots'
    past the run's scratch (em_slots) or every slot."""
    if getattr(dev, "pool", None) is not None:
        return (dev.pool, dev.pool + (dev.ids - 1) * dev.ebytes)
    return getattr(dev, "em_slots", None) or getattr(dev, "slots", None)


def moe_ffn(x, lw, mo: MoESpec, dev: SimpleNamespace, eps: float, beside=None,
            residual: bool = True, y_first: bool = False, n: int | None = None,
            spread: int = 0):
    """x + the MoE FFN of one token (module docstring). lw: the layer's g_post [H], router
    QTensor [E, H] ([E + 1, H] with a shared expert: its gate is row E), ebias Tensor [E]
    (sigmoid_bias), gbase Tensor [1] (j * E as fp32), and with a shared expert its SwiGLU
    wg, wu, wd; with g_exp [H] (Gemma 4: pre_feedforward_layernorm_2's gain) the routed
    experts read their own quantized input, the norm times g_exp, and the router the norm (the
    norm alone, quantized, has its blocks scaled by the residual's outlier channels, which the
    gain zeroes: docs/offload.md 11.2). dev: the offload words (mbox, served, answer, dir:
    static DRAM addresses), `tag` (a slot's tag word, from the slot) and `fmt`, the
    ExpertFormat.

    beside(): emits work that needs no expert (Gemma 4's dense MLP) right after the request is
    posted, so that it runs while the host streams the missing experts (docs/offload.md 5.3);
    moe_ffn's register is free during it. A shared expert runs there too, before beside()'s
    work (docs/offload.md 15: the card waits for its misses in most MoE layers, and a shared
    expert computed after them lengthened every wait by its time); with a resident pool (no
    request) its gate and up MMs are issued before the routing instead, and the routing runs
    while they stream (swiglu_down's each(0)). spread (a resident pool): beside's work in that
    many parts instead (at most k), beside(p) emitting 1/p of what it has left, a part after each
    of the first `spread` routed experts (their loop's first iterations peeled), so that its DMA
    runs while their MMs stream: before them it filled the sequencer's window, and the first
    routed expert's MMs waited for it to drain (qwen35's _CvPipe, ~30 stores and loads); with
    nothing queued beside(p) emits nothing and the peeled bodies stay (faster than the loop's).
    residual=False: the FFN's output without x (Gemma 4 norms the experts' sum before its residual). y_first: the experts' [k, H] outputs take
    their TMEM before the router's input does (Gemma 4 26B-A4B: 22,544 words, which only the
    free space before the step's later tiles holds). n: the norm's width where x is padded
    (GPT-OSS: rmsnorm's n). With MoESpec.bias the router's logits add lw.rbias [E] and each
    expert its slot's biases (act "oai": oai_glu's gate and up, then W_down's). Experts of
    fewer rows than x's width (ExpertFormat tight: Hd, GPT-OSS's 2880 of 2944) add to x's
    first Hd columns in place (x is returned: its padding stays zero)."""
    b = current()
    if ol.num_programs() != 1:                  # (tensor-parallel experts: _moe_ffn_tp)
        return _moe_ffn_tp(x, lw, mo, dev, eps, beside, residual, y_first, n, spread)
    spread = min(spread, mo.k) if beside is not None and getattr(dev, "pool", None) is not None \
        else 0
    E, k, H = mo.E, mo.k, dev.fmt.Hd if getattr(dev.fmt, "tight", False) else x.cols
    if H != x.cols and (mo.shared or not residual):
        raise CompileError(f"moe_ffn: experts of {H} rows on x of {x.cols} with a shared "
                           f"expert or residual=False")
    y = ol.empty((k, H)) if y_first else None
    if getattr(lw, "g_exp", None) is None:
        xs = xe = ol.quantize(rmsnorm(x, ol.load(lw.g_post), eps, n))
    else:
        xn = rmsnorm(x, ol.load(lw.g_post), eps, n)
        xs, xe = ol.quantize(xn), ol.quantize(xn * ol.load(lw.g_exp)[None, :])   # (QACT
        del xn                                                                   # CSCALE)
    # A resident pool sums the experts into ys from -0 (below). Its FILL is here, under the
    # router's MM, where the VPU waits for the logits anyway: after the routing it sat in the
    # VPU's order before the first expert's place, and behind a pipe's LDs beside the experts
    # (qwen35's) on TMEM's write port, where the DMA comes first (2,063 cycles for 2,048 words)
    ys = ol.full((1, H), -0.0) if getattr(dev, "pool", None) is not None else None
    lg = ol.dot(xs, lw.router)
    sc, sel = ol.empty((E,)), ol.empty((E,))
    if mo.rule == "softmax":
        sc.set(lg[0, 0:E] + ol.load(lw.rbias) if mo.bias else lg[0, 0:E])
        sel.set(sc)
    else:
        sc.set(sigmoid(lg[0, 0:E]))
        sel.set(sc + ol.load(lw.ebias))
    gate = sigmoid(lg[0, E:E + 1]) if mo.shared else None
    # One register in all (the resident decode's arguments and the layer loops hold the
    # rest): r holds one value at a time (an index, the seq bits, an offset, a count, a slot
    # address, a row). The k experts' values are the columns of pe, a [5, k] tile that every
    # loop over the experts rotates left by one at the end of its body: expert i is at column
    # 0 in iteration i, and after k iterations every row is back in order. (A slot address
    # never passes through the VPU, which would flush it as a denormal: WAITW copies it from
    # the directory to the word `word`, at once for an expert present, and RLD raw takes its
    # bits into r.) With dev.pool (a resident pool: every expert in DRAM) there is no request
    # and no directory: each expert's place is resident()'s, from its global id in OFF.
    EP, MISS, OFF, WT, ROW, ANS = range(6)          # pe's rows
    pe, tmp, ids, pr = ol.empty((6, k)), ol.empty((6, k)), ol.empty((k,)), ol.empty((2,))
    word = ol.empty((1,))
    wt = pe[WT, :]

    def rotate(rows=slice(0, 6)):
        a, e = rows.start, rows.stop
        if k > 1:
            tmp[a:e, 0:k - 1].set(pe[a:e, 1:k])
        tmp[a:e, k - 1:k].set(pe[a:e, 0:1])
        pe[a:e, :].set(tmp[a:e, :])

    pool = getattr(dev, "pool", None)       # (every expert in DRAM: resident())
    r = gid = None

    def route():
        """The k best (ARGMAX, knock-out; in the router's order, the new one to column k - 1),
        their weights, global ids and rows; r taken."""
        nonlocal r, y, gid
        r = b.scratch()
        lp = b.begin_loop(k)
        b.emit(I.argmax(pr.base, sel.base, 1, E, comment="moe: best"))
        b.emit(I.rld(r, pr.base + 1, bound=(0, E - 1), comment="its index"))
        b.emit(I.vop(I.V_FILL, sel.base, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, NEG, ra=r,
                     comment="knock out"))
        rotate(slice(WT, WT + 1))
        b.emit(I.vop(I.V_COPY, wt.base + k - 1, sc.base, 0, 1, 1, 0, 0, 0, rb=r,
                     comment="weight"))
        if k > 1:
            tmp[0, 0:k - 1].set(ids[1:k])
            ids[0:k - 1].set(tmp[0, 0:k - 1])
        b.emit(I.vop(I.V_COPY, ids.base + k - 1, pr.base + 1, 0, 1, 1, 0, 0, 0, comment="id"))
        b.end_loop(lp)
        if mo.rule == "softmax":                        # column 0 holds the largest
            e = ol.exp2((wt - wt[0:1]) * ol.LOG2E)
            wt.set(e * ol.recip(ol.sum(e)))
        elif mo.norm:
            wt.set(wt * ol.recip(ol.sum(wt) + 1e-6))
        if mo.scale != 1.0:
            wt.set(wt * float(mo.scale))
        gid = ol.empty((k,))
        gid.set(ids + ol.load(lw.gbase))                # global ids: j * E + index
        pe[OFF, :].set(gid if pool is not None else gid * 8.0)  # (their directory entries')
        if pool is not None:        # (a resident pool sums into ys as it goes: no y)
            return
        if y is None:
            y = ol.empty((k, H))
        for i in range(k):          # y's rows (ol.empty may pad them), the answer's words
            pe[ROW, i:i + 1].set(float(i * y.rs))
            pe[ANS, i:i + 1].set(float(4 * i))

    # A resident pool (no request: the experts stream from DRAM as their MMs ask) issues the
    # shared expert's gate and up MMs, which need only xs, before the routing, so the DRAM
    # streams them while the VPU routes (its result is added last all the same)
    sh = None
    early = pool is not None and mo.shared
    if early:
        sh = swiglu_down(xs, lw.wg, lw.wu, lw.wd, each=lambda i: i == 0 and route()) * \
            gate[:, None]
        del gate
    else:
        route()
    if pool is None:                # the fence, then the request: seq + 1 and the ids
        seq = ol.load(Tensor(Affine(dev.mbox), (1,), (1,)))
        b.rld(r, seq, raw=True, comment="seq (bits)")
        b.waitw(word, dev.served, 0, I.C_GE, rc=r, comment="fence: served >= seq")
        ol.store(Tensor(Affine(dev.mbox + LINE), (k,), (1,)), gid)
        ol.store(Tensor(Affine(dev.mbox + 4), (1,), (1,)), ol.full((1,), float(k)))  # (count)
        seq.set(seq + 1.0)
        ol.store(Tensor(Affine(dev.mbox), (1,), (1,)), seq)
    if (mo.shared and not early) or beside is not None:     # while the host streams: the
        b.unscratch(r)                                      # work that needs no routed expert
        if mo.shared and not early:                     # (the shared expert, added last)
            sh = swiglu_down(xs, lw.wg, lw.wu, lw.wd) * gate[:, None]
            del gate
        if beside is not None and not spread:
            beside()
        r = b.scratch()
    # the directory: each entry's present flag
    col = lambda row: pe.base + row * pe.rs                       # noqa: E731 (column 0)
    if pool is None:
        lp = b.begin_loop(k)
        b.emit(I.rld(r, col(OFF), bound=_ebound(dev), comment="entry offset"))
        b.emit(I.ld(dev.dir + 4, col(EP), 1, ra=r, comment="entry: present"))
        rotate()
        b.end_loop(lp)
        pe[MISS, :].set(pe[EP, :] * -1.0 + 1.0)
    ex = dev.fmt.descs(DevVar("expert slot", r, align=LINE))   # (slots: LINE-aligned)
    glu = oai_glu(mo, ex) if mo.act == "oai" else None

    def expert(into=None):
        """The expert whose slot is R[r], weighted, into its row of y (into: added to that
        [1, H] tile instead)."""
        o = swiglu_down(xe, ex.wg, ex.wu, ex.wd, act=ACTS.get(mo.act), glu=glu)
        if ex.bd is not None:
            o = o + ol.load(ex.bd)[None, :]
        b.check_live(o)
        if into is not None:
            w = ol.empty((1, H))
            b.emit(I.vop(I.V_MUL, w.base, o.base, col(WT), 1, H, 0, 0, 0, I.B_ROW,
                         comment="weighted"))
            into.set(into + w)
            return
        b.emit(I.rld(r, col(ROW), bound=(0, (k - 1) * y.rs), comment="its row"))
        b.emit(I.vop(I.V_MUL, y.base, o.base, col(WT), 1, H, 0, 0, 0, I.B_ROW, ra=r,
                     comment="weighted, to its row"))

    if pool is not None:
        # The next expert's place is read at the end of this one's body, its MMs issued, so the
        # next MMs queue behind this expert's while the VPU finishes it: the RLD MUL waits for
        # the body's rotate alone (at the body's head it waited for the last expert's bias add
        # too: a VPU mask names producers by rank, and on the way round the rank of the
        # entry's producer, the ids' store, was the bias add's). Each expert's weighted output
        # adds to ys there, in the router's order: y's rows' sum bit for bit from -0 (-0 + y0 is
        # y0's bits, a -0 included, where +0 + -0 is +0), off the layer's tail. With spread the
        # first `spread` experts are the loop's body peeled, each followed by a part of beside's
        # work (its DMA beside the MMs queued before it)
        resident(r, col(OFF), dev)

        def body():
            rotate(slice(OFF, OFF + 1))
            expert(ys)
            resident(r, col(OFF), dev)
            rotate(slice(WT, WT + 1))
        for i in range(spread):
            body()
            beside(spread - i)
        if k > spread:
            lp = b.begin_loop(k - spread)
            body()
            b.end_loop(lp)
    for flag, wait in ((EP, False), (MISS, True)) if pool is None else ():
        lp = b.begin_loop(k)
        b.emit(I.rld(r, col(flag), bound=(0, 1), comment="missing" if wait else "present"))
        inner = b.begin_loop(0, rcount=r)               # (the count is read here: r is free)
        slot_of(r, word, col(ANS) if wait else col(OFF), dev, wait,
                (0, 4 * (k - 1)) if wait else _ebound(dev))
        expert()
        b.end_loop(inner)
        rotate()
        b.end_loop(lp)
    b.unscratch(r)
    if pool is None:
        ol.store(Tensor(Affine(dev.answer), (k,), (1,)), ol.zeros((k,)))   # (the next request)
    acc = ys
    if acc is None:
        acc = y[0:1, :]
        for i in range(1, k):
            acc = acc + y[i:i + 1, :]
    if sh is not None:
        acc = acc + sh
    if H != x.cols:
        x[:, 0:H].set(x[:, 0:H] + acc)
        return x
    return x + acc if residual else acc


def moe_ffn_tp_rows(x, lw, mo: MoESpec, dev: SimpleNamespace, eps: float):
    """x + the MoE FFN of each of x's R token rows at S > 1 (x [R, H]: batched decode's rows,
    opentpu/llm/batch35.py), each row bit for bit moe_ffn's of that row alone: the router and
    the shared expert once over the rows (one stream of their weights for all of them), each
    row's routed experts in turn (_moe_ffn_tp)."""
    return _moe_ffn_tp(x, lw, mo, dev, eps, None, True, False, None)


def _moe_ffn_tp(x, lw, mo: MoESpec, dev: SimpleNamespace, eps: float, beside, residual: bool,
                y_first: bool, n: int | None, spread: int = 0):
    """moe_ffn at S > 1: tensor-parallel experts in a resident pool (module docstring, "At
    S > 1"). Each value moe_ffn computes at S = 1 one slice computes here for its rows (its
    router rows, its gate and up rows, its W_down rows) by the same operations, and the
    activations each W_down reads are gathered first, so every QACT range is S = 1's: the
    layer's output is S = 1's bit for bit. lw: moe_ffn's, with the slice's rows of the router
    (S Er rows in all: E, the shared gate, zero rows) and of the shared expert's gate and up,
    and its W_down's rows in S = 1's column parts (gs 1). dev: pool_dev's of a pool of
    ExpertFormat S parts; dev.groups (default: 1 with a shared expert, else 2): the routed
    experts' activations gathered in that many groups, each group's down MMs after the next
    group's gate and up MMs (and the shared expert's down after the last group's), which hide
    its gather; with groups > 1 the first expert's down MMs (and its place) are issued before
    the last chunk's activations, so they stream while the VPU and the last group's gather work
    (its place, an RLD, otherwise waited for that gather); the shared expert's activations are
    gathered while the first expert's gate and up MMs stream. spread: beside's work in that many parts (at most k, as moe_ffn's), a part after
    each of the first `spread` experts' activations, the next expert's first gate and up MMs
    issued before it (its DMA beside their streams). x [R, H] (moe_ffn_tp_rows; no beside,
    tight experts or oai): the router's and the shared expert's MMs over the R rows, each row
    routed, then each row's routed experts as one row's, the shared expert's activations and
    down beside the first row's (as above); the later rows' in two groups where the shared
    expert's down is not there to hide their last gather (k even)."""
    b = current()
    S, sid, fmt = ol.num_programs(), ol.program_id(), dev.fmt
    R = x.rows
    spread = min(spread, mo.k) if beside is not None else 0
    if getattr(dev, "pool", None) is None:
        raise CompileError(f"moe_ffn at S = {S}: experts in a resident pool only (no server)")
    if not residual or y_first:
        raise CompileError(f"moe_ffn at S = {S}: residual=False or y_first")
    if fmt.S != S:
        raise CompileError(f"moe_ffn at S = {S}: experts in parts of {fmt.S} slices")
    E, k, Hs, Fs = mo.E, mo.k, fmt.Hs, fmt.Fs
    tight = fmt.Hd != x.cols
    if tight and (mo.shared or not fmt.tight):
        raise CompileError(f"moe_ffn: experts of {fmt.Hd} rows on x of {x.cols} with a shared "
                           f"expert")
    if fmt.Hd != S * Hs or (not tight and x.cols != S * Hs):
        raise CompileError(f"moe_ffn at S = {S}: experts of {fmt.Hd} rows on x of {x.cols}")
    if S * lw.router.shape[0] < E + bool(mo.shared):
        raise CompileError(f"moe_ffn at S = {S}: router rows {lw.router.shape[0]} a slice")
    if R > 1 and (beside is not None or tight or mo.act == "oai"):
        raise CompileError(f"moe_ffn at S = {S}: {R} rows with beside, tight experts or oai")
    groups = getattr(dev, "groups", None) or (1 if mo.shared else 2)
    if k % groups:
        raise CompileError(f"moe_ffn: {k} experts in {groups} groups")
    later = 2 if groups == 1 and k % 2 == 0 else groups     # (the rows after the first)
    ys = [ol.full((1, Hs), -0.0) for _ in range(R)]     # each row's routed experts' sum, as
    gx = getattr(lw, "g_exp", None)                     # moe_ffn's (from -0)
    if R > 1:                           # (the normed rows kept: each row's experts read its own
        xn = rmsnorm(x, ol.load(lw.g_post), eps, n)     # operand, quantized as it alone is)
        xs = ol.quantize(xn)
    elif gx is None:
        xs = xe = ol.quantize(rmsnorm(x, ol.load(lw.g_post), eps, n))
    else:
        xn = rmsnorm(x, ol.load(lw.g_post), eps, n)
        xs, xe = ol.quantize(xn), ol.quantize(xn * ol.load(lw.g_exp)[None, :])
        del xn
    lr = ol.dot(xs, lw.router)          # the slice's router rows
    # the shared expert's gate and up MMs (its rows of the slice) stream while the VPU routes
    act = ACTS.get(mo.act)
    if mo.shared:
        fsh = lw.wg.shape[0]
        csh = min(fsh, lw.wd.pw if lw.wd.parts is not None else fsh)   # (S = 1's chunk)
        gu_sh = [(c0, ol.dot(xs, lw.wg[c0:c0 + min(csh, fsh - c0), :]),
                  ol.dot(xs, lw.wu[c0:c0 + min(csh, fsh - c0), :])) for c0 in range(0, fsh, csh)]
    lg = ol.all_gather(lr)              # S = 1's logits (and a zero row's)
    scs, sels, gates = [], [], []
    for q in range(R):
        sc, sel = ol.empty((E,)), ol.empty((E,))
        if mo.rule == "softmax":
            sc.set(lg[q, 0:E] + ol.load(lw.rbias) if mo.bias else lg[q, 0:E])
            sel.set(sc)
        else:
            sc.set(sigmoid(lg[q, 0:E]))
            sel.set(sc + ol.load(lw.ebias))
        scs.append(sc)
        sels.append(sel)
        gates.append(sigmoid(lg[q, E:E + 1]) if mo.shared else None)
    del lg, sc, sel
    # the route: moe_ffn's operations (ARGMAX with knock-out, the new one to column k - 1),
    # each expert's weight and global id at its column of wt and gid (read at static columns:
    # the experts are unrolled)
    wts, gids, held_e, r = [], [], [], None
    for q in range(R):
        sc, sel = scs[q], sels[q]
        ids, wt, tmp, pr = ol.empty((k,)), ol.empty((k,)), ol.empty((k,)), ol.empty((2,))
        if r is None:
            r = b.scratch()
        lp = b.begin_loop(k)
        b.emit(I.argmax(pr.base, sel.base, 1, E, comment="moe: best"))
        b.emit(I.rld(r, pr.base + 1, bound=(0, E - 1), comment="its index"))
        b.emit(I.vop(I.V_FILL, sel.base, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, NEG, ra=r,
                     comment="knock out"))
        for t, src, rb in ((wt, sc.base, r), (ids, pr.base + 1, 0)):
            if k > 1:
                tmp[0:k - 1].set(t[1:k])
                t[0:k - 1].set(tmp[0:k - 1])
            b.emit(I.vop(I.V_COPY, t.base + k - 1, src, 0, 1, 1, 0, 0, 0, rb=rb,
                         comment="weight" if rb else "id"))
        b.end_loop(lp)
        sels[q] = None
        del tmp, pr, sel
        if mo.rule == "softmax":                    # column 0 holds the largest
            e = ol.exp2((wt - wt[0:1]) * ol.LOG2E)
            wt.set(e * ol.recip(ol.sum(e)))
            held_e.append(e)
            del e
        elif mo.norm:
            wt.set(wt * ol.recip(ol.sum(wt) + 1e-6))
        if mo.scale != 1.0:
            wt.set(wt * float(mo.scale))
        gid = ol.empty((k,))
        gid.set(ids + ol.load(lw.gbase))            # global ids: j * E + index
        del ids
        wts.append(wt)
        gids.append(gid)
        del sc, wt, gid
    if beside is not None and not spread:   # (r is free during it)
        b.unscratch(r)
        beside()
        r = b.scratch()
    ex = fmt.descs(DevVar("expert slot", r, align=LINE))   # (slots: LINE-aligned)
    C = fmt.C
    for q in range(R):
        wt, gid = wts[q], gids[q]
        xq = xe if R == 1 else ol.quantize(xn[q:q + 1, :] if gx is None else
                                           xn[q:q + 1, :] * ol.load(gx)[None, :])
        if q == R - 1 and R > 1:
            del xn
        kg = k // (groups if q == 0 else later)

        def place(i):                   # R[r] = expert i's slot; GPT-OSS: its biases loaded
            resident(r, gid.base + i, dev)
            return None if ex.bg is None else (ol.load(ex.bg), ol.load(ex.bu))

        def down(i):                    # expert i's down MMs over its gathered activations
            o = _down_parts(hg[i // kg][i % kg:i % kg + 1, :], ex.wd)
            return o, None if ex.bd is None else ol.load(ex.bd)

        # Phase A: each expert's activations for the slice's F rows, into its row of hs (its
        # group's); the next chunk's gate and up MMs (the next expert's: its place first)
        # issued before the VPU computes a chunk's; before the last chunk's, phase B's first
        # down MMs where their activations are gathered already (groups > 1)
        steps = [(i, c0, min(C, Fs - c0)) for i in range(k) for c0 in range(0, Fs, C)]
        hs, hg, first = [], [], None

        def gate_up(c0, w):
            return ol.dot(xq, ex.wg[c0:c0 + w, :]), ol.dot(xq, ex.wu[c0:c0 + w, :])
        held = place(0)
        nxt = gate_up(*steps[0][1:])
        if mo.shared and q == 0:        # its activations, gathered (S = 1's [R, F]) while the
            hsh = ol.empty((R, fsh))    # first expert's MMs stream
            for c0, g, u in gu_sh:
                hsh[:, c0:c0 + g.cols].set(act(g) * u)
            del gu_sh, g, u
            hsh = ol.all_gather(hsh)
        for j, (i, c0, w) in enumerate(steps):
            g, u = nxt
            cur = held
            if j + 1 < len(steps):
                i2, c2, w2 = steps[j + 1]
                if i2 != i:
                    held = place(i2)
                nxt = gate_up(c2, w2)
            elif hg:                    # (expert 0's group gathered: its down MMs stream
                resident(r, gid.base, dev)  # while this chunk's activations and the last
                first = down(0)         # group's gather work)
            if i % kg == 0 and c0 == 0:
                hs.append(ol.empty((kg, Fs)))
            row = hs[-1][i % kg:i % kg + 1, c0:c0 + w]
            if mo.act == "oai":
                p, pq = _oai_factors(mo, ex, cur)(g, u, c0)
                row.set(p * pq)
                del p, pq
            else:
                row.set(act(g) * u)
            del row, g, u
            if i % kg == kg - 1 and c0 + w == Fs:       # the group's activations, gathered
                hg.append(_gather_cols(hs.pop(), fmt.F))
            if i < spread and c0 + w == Fs:             # a part of beside's (r held, as moe_ffn's)
                beside(spread - i)
        del held, cur
        if mo.shared and q == 0:        # its down in S = 1's ranges: it hides the last gather
            ysh = _down_parts(hsh, lw.wd)
            del hsh
        # Phase B: each expert's down MMs over its gathered activations, in S = 1's QACT
        # ranges (the next expert's issued before the VPU weighs this one's), weighted, summed
        # in the router's order
        if first is None:
            resident(r, gid.base, dev)
            first = down(0)
        nxt = first
        del first
        for i in range(k):
            o, bd = nxt
            if i + 1 < k:
                resident(r, gid.base + i + 1, dev)
                nxt = down(i + 1)
            if bd is not None:
                o = o + bd[None, :]
            b.check_live(o)
            y = ol.empty((1, Hs))
            b.emit(I.vop(I.V_MUL, y.base, o.base, wt.base + i, 1, Hs, 0, 0, 0, I.B_ROW,
                         comment="weighted"))
            ys[q].set(ys[q] + y)
            del o, bd, y
        if q + 1 < R:                   # (the last row's stay as one row's: its TMEM order)
            del hs, hg, nxt
    b.unscratch(r)
    mine = slice(sid * Hs, (sid + 1) * Hs)
    if R == 1:
        acc = ys[0]
        if mo.shared:
            sh = ysh * gates[0][:, None]
            acc = acc + sh
        out = x[:, mine] + acc
    else:                               # (each row's: one row's operations)
        out = ol.empty((R, Hs))
        for q in range(R):
            acc = ys[q] + ysh[q:q + 1, :] * gates[q][:, None] if mo.shared else ys[q]
            out[q:q + 1, :].set(x[q:q + 1, mine] + acc)
            del acc
    if not tight:
        return ol.all_gather(out)
    b.emit(I.gather(out.base, x.base, 1, Hs, Hs, x.rs, Hs, comment="all-gather into x"))
    b.bump_version(x.buf)               # (x's first S Hs columns; its padding stays zero)
    return x


def _gather_cols(h, F: int):
    """Every slice's [rows, Fs] h side by side: [rows, F] with zero columns past S Fs (a tight
    expert's: W_down's last part reads its whole D-blocks, as swiglu_down's zero-filled short
    chunk)."""
    b = current()
    S = ol.num_programs()
    if S * h.cols == F:
        return ol.all_gather(h)
    out = ol.empty((h.rows, F))
    out[:, S * h.cols:F].set(0.0)
    b.emit(I.gather(h.base, out.base, h.rows, h.cols, h.rs, out.rs, h.cols,
                    comment="all-gather"))
    b.bump_version(out.buf)
    return out


def _swiglu_rows_tp(xs, wg, wu, wd, F: int, C: int, act=silu, rscale=None, out=None,
                    each=None):
    """swiglu_down of xs's rows ([M, H] quantized) at S > 1 as S = 1's, bit for bit (_moe_ffn_tp's
    order): the slice's gate and up rows (its Fs of F) in chunks of C, their activations
    gathered whole ([M, F], _gather_cols), then W_down's rows of the slice (its Hs output
    columns) in S = 1's column parts (_down_parts): [M, Hs]. rscale, out: _down_parts'.
    each(i): called once chunk i's gate and up MMs are issued (swiglu_down's)."""
    Fs = wg.shape[0]
    h = ol.empty((xs.src.rows, Fs))
    for i, c0 in enumerate(range(0, Fs, C)):
        w = min(C, Fs - c0)
        g, u = ol.dot(xs, wg[c0:c0 + w, :]), ol.dot(xs, wu[c0:c0 + w, :])
        if each is not None:
            each(i)
        h[:, c0:c0 + w].set(act(g) * u)
        del g, u
    hg = _gather_cols(h, F)
    del h
    return _down_parts(hg, wd, rscale, out)


def _down_parts(h, wd, rscale=None, out=None):
    """h [1, K] . W_down^T in W_down's column parts (S = 1's swiglu_down chunks: a QACT and an
    MM a part, the first overwriting, the rest accumulating, in order). rscale [M]: each row of
    h times its factor in the QACTs (QACT RSCALE: EM_GSCALE's router weights). out: the tile
    the first MM overwrites (ol.dot's out=) and the result."""
    if wd.gs != 1:
        raise CompileError("W_down in the gather's order: S = 1's parts expected")
    y, K = None, wd.shape[1]
    w = wd.pw if wd.parts is not None else K
    for c0 in range(0, K, w):
        a, cols = h[:, c0:min(K, c0 + w)], wd[:, c0:min(K, c0 + w)]
        if rscale is not None:
            a = ol.quantize(a, rscale=rscale)
        if y is None:
            y = ol.dot(a, cols, out=out)
        else:
            ol.dot(a, cols, acc=y)
    return y


def moe_ffn_rows(x, lw, mo: MoESpec, dev: SimpleNamespace, eps: float, beside=None,
                 residual: bool = True):
    """x + the MoE FFN of R token rows at once (x [R, H], R * k <= LINE / 4: one request row):
    a layer-major prefill run's (docs/offload.md 13). Every row routes as moe_ffn's; one
    request names the rows' R * k ids (repeats included: the host serves each once) and their
    count (the word after seq); each expert of the rows' union runs once, on every row (one MM
    of R rows: with PAIR, R <= 2 cost what one does), and each row that chose it stores its
    output, unweighted, to its place (row, rank) of dev.scratch [R * k + 1, H] (the last row a
    sink for the rows that did not); then each row sums its experts' outputs, weighted, in its
    router's order. So a row's result is moe_ffn's on that row alone, bit for bit, whatever
    the cache held. lw, dev (and dev.scratch), beside and residual as moe_ffn's.

    The union: for each row q, eq[n, j] = 1 - min(1, |id_n - id_q,j|) over every entry n of
    the request (the ids are integers: 1 where equal); its row sums say whether row q chose
    entry n's expert (found) and at which rank (the sum of j eq[n, j]), which gives the byte
    offset of entry n's output for row q in the scratch; an entry whose expert an earlier row
    chose is not computed again. One register, as moe_ffn's."""
    b = current()
    _plain(mo, "moe_ffn_rows")
    if ol.num_programs() != 1:
        raise CompileError("moe_ffn_rows runs on one slice")
    R, H, E, k = x.rows, x.cols, mo.E, mo.k
    N = R * k
    if N > LINE // 4:
        raise CompileError(f"{R} rows of {k} experts: more ids than a request row holds")
    if getattr(lw, "g_exp", None) is None:
        xs = xe = ol.quantize(rmsnorm(x, ol.load(lw.g_post), eps))
    else:
        xn = rmsnorm(x, ol.load(lw.g_post), eps)
        xs, xe = ol.quantize(xn), ol.quantize(xn * ol.load(lw.g_exp)[None, :])
        del xn
    lg = ol.dot(xs, lw.router)                          # [R, E] (+ the shared expert's gate)
    gid, wts = ol.empty((R, k), dense=True), ol.empty((R, k), dense=True)
    pr, tmp, ids = ol.empty((2,)), ol.empty((k,)), ol.empty((k,))
    gb = ol.load(lw.gbase)
    pool = getattr(dev, "pool", None)       # (every expert in DRAM: resident())
    r = None

    def route():
        """Each row's k best, as moe_ffn's, their weights and global ids; r taken."""
        nonlocal r
        r = b.scratch()
        for q in range(R):
            sc, sel = ol.empty((E,)), ol.empty((E,))
            if mo.rule == "softmax":
                sc.set(lg[q, 0:E])
                sel.set(sc)
            else:
                sc.set(sigmoid(lg[q, 0:E]))
                sel.set(sc + ol.load(lw.ebias))
            wt = wts[q, :]
            lp = b.begin_loop(k)
            b.emit(I.argmax(pr.base, sel.base, 1, E, comment="moe: best"))
            b.emit(I.rld(r, pr.base + 1, bound=(0, E - 1), comment="its index"))
            b.emit(I.vop(I.V_FILL, sel.base, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, NEG, ra=r,
                         comment="knock out"))
            if k > 1:
                tmp[0:k - 1].set(wt[1:k])
                wt[0:k - 1].set(tmp[0:k - 1])
                tmp[0:k - 1].set(ids[1:k])
                ids[0:k - 1].set(tmp[0:k - 1])
            b.emit(I.vop(I.V_COPY, wt.base + k - 1, sc.base, 0, 1, 1, 0, 0, 0, rb=r,
                         comment="weight"))
            b.emit(I.vop(I.V_COPY, ids.base + k - 1, pr.base + 1, 0, 1, 1, 0, 0, 0,
                         comment="id"))
            b.end_loop(lp)
            del sc, sel
            if mo.rule == "softmax":                    # column 0 holds the largest
                e = ol.exp2((wt - wt[0:1]) * ol.LOG2E)
                wt.set(e * ol.recip(ol.sum(e)))
                del e
            elif mo.norm:
                wt.set(wt * ol.recip(ol.sum(wt) + 1e-6))
            if mo.scale != 1.0:
                wt.set(wt * float(mo.scale))
            gid[q, :].set(ids + gb)                     # global ids: j * E + index

    def shared():
        """The shared expert's output on every row, gated (added last)."""
        g = ol.empty((R,))
        g.column().set(lg[:, E:E + 1])
        gate = sigmoid(g)
        del g
        return swiglu_down(xs, lw.wg, lw.wu, lw.wd,
                           each=lambda i: i == 0 and early and route()) * gate[:, None]

    # a resident pool: the shared expert's gate and up MMs before the routing (moe_ffn's)
    sh = None
    early = pool is not None and mo.shared
    if early:
        sh = shared()
        del lg
    else:
        route()
    del ids, tmp
    gf = gid.reshape(1, N)[0, :]                        # the request's entries, row by row
    # pe's rows (columns: the entries; every loop over them rotates pe left by one, so entry
    # n is at column 0 in iteration n): present and computed here, missing, the directory
    # offset, and per row q the byte offset of the entry's output for q in the scratch
    EP, MISS, OFF, ANS, TG = 0, 1, 2, 3, 4
    RB, SINK = 4 * H, N * 4 * H
    pe, tpe = ol.empty((TG + R, N)), ol.empty((TG + R, N))
    rank = ol.empty((k,))
    for j in range(k):
        rank[j:j + 1].set(float(j))
    dup = ol.zeros((N,))
    for q in range(R):
        d = ol.empty((N, k))
        d.set(gf[:, None])
        eq = 1.0 - ol.minimum(ol.abs(d - gid[q, :][None, :]), 1.0)
        del d
        found = ol.sum(eq)                              # [N]: row q chose entry n's expert
        at = ol.sum(eq * rank[None, :])                 # its rank in row q
        del eq
        pe[TG + q, :].set(found * ((at + float(q * k)) * float(RB) - float(SINK)) + float(SINK))
        if q + 1 < R:                                   # later rows' entries chosen before
            dup[(q + 1) * k:N].set(dup[(q + 1) * k:N] + found[(q + 1) * k:N])
        del found, at
    pe[OFF, :].set(gf if pool is not None else gf * 8.0)
    for n in range(N) if pool is None else ():          # each entry's word of the answer
        pe[ANS, n:n + 1].set(float(4 * n))

    def rotate():
        tpe[:, 0:N - 1].set(pe[:, 1:N])
        tpe[:, N - 1:N].set(pe[:, 0:1])
        pe[:, :].set(tpe[:, :])

    word = ol.empty((1,))
    if pool is None:                # the fence, then the request: the ids, their count, seq + 1
        seq = ol.load(Tensor(Affine(dev.mbox), (1,), (1,)))
        b.rld(r, seq, raw=True, comment="seq (bits)")
        b.waitw(word, dev.served, 0, I.C_GE, rc=r, comment="fence: served >= seq")
        ol.store(Tensor(Affine(dev.mbox + LINE), (N,), (1,)), gf)
        ol.store(Tensor(Affine(dev.mbox + 4), (1,), (1,)), ol.full((1,), float(N)))
        seq.set(seq + 1.0)
        ol.store(Tensor(Affine(dev.mbox), (1,), (1,)), seq)
    if (mo.shared and not early) or beside is not None:     # while the host streams: the
        b.unscratch(r)                                      # work that needs no routed expert
        if mo.shared and not early:                     # (the shared expert, added last)
            sh = shared()
            del lg
        if beside is not None:
            beside()
        r = b.scratch()
    col = lambda row: pe.base + row * pe.rs                       # noqa: E731 (column 0)
    if pool is None:
        lp = b.begin_loop(N)                            # the directory: present flags
        b.emit(I.rld(r, col(OFF), bound=_ebound(dev), comment="entry offset"))
        b.emit(I.ld(dev.dir + 4, col(EP), 1, ra=r, comment="entry: present"))
        rotate()
        b.end_loop(lp)
    keep = 1.0 - ol.minimum(dup, 1.0)                   # entries computed here: the union
    if pool is None:
        pe[MISS, :].set((pe[EP, :] * -1.0 + 1.0) * keep)
        pe[EP, :].set(pe[EP, :] * keep)
    else:                                               # (resident: every one present)
        pe[EP, :].set(keep)
    del keep, dup
    ex = dev.fmt.descs(DevVar("expert slot", r, align=LINE))      # (slots: LINE-aligned)

    def expert():
        """The expert whose slot is R[r] on every row; each row's output to its place."""
        o = swiglu_down(xe, ex.wg, ex.wu, ex.wd, act=ACTS[mo.act])
        b.check_live(o)
        for q in range(R):
            b.emit(I.rld(r, col(TG + q), bound=(0, SINK), comment="its place for this row"))
            ol.store(Tensor(Affine(dev.scratch) + DevVar("place", r), (H,), (1,)), o[q, :])

    for flag, wait in ((EP, False), (MISS, True)) if pool is None else ((EP, False),):
        lp = b.begin_loop(N)
        b.emit(I.rld(r, col(flag), bound=(0, 1), comment="missing" if wait else "present"))
        inner = b.begin_loop(0, rcount=r)               # (the count is read here: r is free)
        if pool is not None:
            resident(r, col(OFF), dev)
        else:
            slot_of(r, word, col(ANS) if wait else col(OFF), dev, wait,
                    (0, 4 * (N - 1)) if wait else _ebound(dev))
        expert()
        b.end_loop(inner)
        rotate()
        b.end_loop(lp)
    b.unscratch(r)
    if pool is None:
        ol.store(Tensor(Affine(dev.answer), (N,), (1,)), ol.zeros((N,)))   # (the next request)
    out = ol.empty((R, H))
    for q in range(R):                                  # each row's sum in its router's order
        acc = None
        for i in range(k):
            y = ol.load(Tensor(Affine(dev.scratch + (q * k + i) * RB), (1, H), (H, 1))) * \
                wts[q, i:i + 1]
            acc = y if acc is None else acc + y
        out[q:q + 1, :].set(acc)
        del acc, y
    if sh is not None:
        out = out + sh
    return x + out if residual else out


# ---------------------------------------------------------------------------- expert-major
EM_RHO = 1.12               # an expert's stream from DRAM, MXU cycles a chunk (the card: 17.07 / 15.2
                            # GB/s); any value in (1, 2) orders the pass sizes alike (em_pass_order)
TOPK_ROWS = 4               # moe_prologue_rows: rows a top-k pass (one ARGMAX over them, a register
                            # each: their ARGMAX -> RLD -> knock-out chains side by side)
_TWO23 = float(1 << 23)     # FILL's immediate plus a register: the float 2^23 + R (0 <= R < 2^23)


def em_meta(k: int) -> int:
    """Words of a record's tail: the row's k ids and k weights and its norm's 1 / rms, in whole
    lines."""
    w = LINE // 4
    return -(-(2 * k + 1) // w) * w


def em_record(H: int, k: int, D: int = 0) -> int:
    """Bytes of a chunk row's record in the expert-major scratch (docs/offload.md 13.11): the
    row's residual X [H], its k experts' outputs OUT [k, H] (rank order), the shared expert's
    output or the normed dense MLP's SH [H], then its tail (em_meta): global ids [k], weights
    [k] and the 1 / rms of the norm before the router and the experts (fp32); D > 0 (em_qd:
    Config.QONCE), then its em_q area (em_qbytes)."""
    return em_qoff(H, k) + em_qbytes(H, D)


def em_qoff(H: int, k: int) -> int:
    """The byte offset of a record's em_q area (after its tail)."""
    return 4 * ((k + 2) * H + em_meta(k))


def em_qbytes(H: int, D: int) -> int:
    """Bytes of a record's em_q area (0 at D 0): the row's expert input quantized once, as the
    MoE prologue's QACT ST stores it (isa.qrow_bytes: H bytes, then a record of 8 bytes a block
    of D), in whole lines; the expert run loads it into ACT RAM a row at a time (LDA) instead
    of X and 1 / rms and a QACT an entry."""
    return -(-I.qrow_bytes(H // D, D) // LINE) * LINE if D else 0


def em_qd(cfg) -> int:
    """em_record's D for a configuration: its D on a QONCE machine (QACT ST, LDA; the rows
    quantized once), else 0 (no em_q area)."""
    return cfg.D if getattr(cfg, "QONCE", False) else 0


def em_q(dev: SimpleNamespace, H: int, k: int):
    """The em_q area's byte offset in dev's records when the rows are quantized once (the
    machine's QONCE, records with the area: em_record's D em_qd), else None: the prologue
    stores each row's expert input there and the expert run loads it (both or neither)."""
    D = em_qd(current().cfg)
    return em_qoff(H, k) if D and dev.em_rec >= em_qoff(H, k) + em_qbytes(H, D) else None


def em_slots(layout: Layout, nbytes: int) -> list:
    """The slots a scratch of nbytes covers: the slot area's first ones, from the first slot on
    (the slots follow one another, layer after layer). The server takes them for the prefill
    and hands them back after it, with their tag beats zeroed."""
    s = layout.all_slots()
    n = -(-nbytes // layout.pitch)
    if n > len(s):
        raise ValueError(f"a scratch of {nbytes} bytes: {n} slots, the image has {len(s)}")
    return s[:n]


def em_expert_slots(layout: Layout, nbytes: int, bounds: tuple):
    """The slots' bound (lo, bounds' hi) an expert run's experts are in: those past its scratch of
    nbytes (em_slots: the records' slots, which the run's stores write) when they leave the needs
    room (streamed.expert_major_fits: 3 E slots), else None (every slot)."""
    s = layout.all_slots()
    n = -(-nbytes // layout.pitch)
    return (s[n], bounds[1]) if 0 < n and len(s) - n >= 3 * layout.E else None


def em_scratch(layout: Layout, rows: int, prefill_rows: int, rec: int) -> int:
    """An expert-major prefill's scratch in bytes (ExpertServer.begin_prefill's) for a first
    chunk of `rows` rows: the image's prefill rows' records when its expert runs' experts are in
    the slots past those (em_expert_slots: no expert in a slot the run's stores may write), else
    the chunk's rows'."""
    big = prefill_rows * rec
    return big if em_expert_slots(layout, big, (0, 0)) is not None else rows * rec


def em_row(dev: SimpleNamespace, row, r: int = 0) -> Affine:
    """The byte address of the record of chunk row row + r (row: a run argument or an int)."""
    return Affine(dev.em_base + r * dev.em_rec) + row * dev.em_rec


def em_x(dev: SimpleNamespace, row, R: int, H: int) -> Tensor:
    """The residual rows X of chunk rows row .. row + R - 1, [R, H] (expert-major's residual
    stream: the records' first words)."""
    return Tensor(em_row(dev, row), (R, H), (dev.em_rec // 4, 1))


def need_rows(k: int) -> int:
    """The rows a need line holds: LINE / 4 ids, k a row (0: not one)."""
    return (LINE // 4) // max(1, k)


def moe_prologue_rows(x, lw, mo: MoESpec, dev: SimpleNamespace, eps: float, row,
                      beside=None) -> None:
    """Expert-major's MoE prologue (docs/offload.md 13.11) of a layer-major run's R rows x [R,
    H] (chunk rows from `row`, a run argument): each row's norm and route as moe_ffn_rows'; to
    each row's record tail its k global ids and weights and the norm's 1 / rms (the expert run
    norms the row again from X and it: bit for bit); then the rows' ids, offset by
    dev.need_off (2 x layers x E), posted as need lines of need_rows(k) rows each (a hint's
    post: the server streams the experts no slot holds and answers nothing; the layer's expert
    run waits on their directory entries); then to each row's SH the shared expert's output
    (SwiGLU of the layer block times sigmoid of its gate's logit), or beside()'s ([R, H]:
    Gemma 4's normed dense MLP, while the host streams). The router's MM takes the R rows at
    once (a tile's: tiled.qwen35_layer_tile), the shared expert's _shared_rows a pass; the k
    best in passes of TOPK_ROWS rows (an ARGMAX over them, a register a row). Of more need
    lines than one, the first goes out before the shared expert and the others as its chunks
    begin (each post waits for the host to serve the one before: the MXU works meanwhile), the
    rest after it; a resident pool's shared expert (one pass) goes before the k best instead,
    its stores after them. The caller stores x itself to the records' X (em_x). TOPK_ROWS + 1
    registers."""
    b = current()
    _plain(mo, "moe_prologue_rows")
    S, sid = ol.num_programs(), ol.program_id()
    if S != 1 and (getattr(dev, "pool", None) is None or beside is not None):
        raise CompileError("moe_prologue_rows at S > 1: a resident pool's (tensor-parallel: the "
                           "router's rows and the shared expert's parts a slice)")
    R, H, E, k = x.rows, x.cols, mo.E, mo.k
    Hs = H // S                                         # (a slice's SH / OUT columns)
    per = need_rows(k)                                  # rows a need line
    if per < 1:
        raise CompileError(f"{k} experts a row: more ids than a need line holds")
    ss = ol.sumsq(x) if b.unroll else ol.sum(x * x, axis=1)    # rmsnorm's, its 1 / rms kept
    rinv = ol.rsqrt(ss * (1.0 / H) + eps)
    del ss
    g = ol.load(lw.g_post)
    qo = em_q(dev, H, k)                                # (quantize-once: the records' em_q)

    def normed(r0, n, store=False):                     # rows r0 .. r0 + n, normed, quantized
        xr, rr = (x, rinv) if n == R else (x[r0:r0 + n, :], rinv[r0:r0 + n])
        sto = (em_row(dev, row, r0) + qo, dev.em_rec) if store else None
        if getattr(lw, "g_exp", None) is None:          # (as moe_ffn_rows' quantize(rmsnorm);
            if b.unroll and n == R:                     # store: the experts' input too, QACT ST;
                q = ol.quantize(x, rscale=rr, cscale=g, store=sto)    # merged, stage.py: no products' words,
                b.free_now(x)                           # x in the records and not read after this)
                return q
            return ol.quantize((xr * rr[:, None]) * g[None, :], store=sto)
        xn = (xr * rr[:, None]) * g[None, :]
        if store:                                       # (g_exp: the experts' is xn * g_exp,
            ge = ol.load(lw.g_exp)                      # the expert run's VOPs' product)
            ol.quantize(xn * ge[None, :], free=True, store=sto)
            del ge
        return ol.quantize(xn)

    G = _shared_rows(R, H, lw) if mo.shared else R       # rows a shared expert pass
    xs = normed(0, R, store=qo is not None)
    if G == R:
        del g
    lg = ol.dot(xs, lw.router)                          # [R, E] (+ the shared expert's gate)
    if S > 1:                                           # (the slices' router rows: S = 1's)
        lg = ol.all_gather(lg)
    resident = getattr(dev, "pool", None) is not None   # (no server, no need lines)
    gid, wts = ol.empty((R, k), dense=True), ol.empty((R, k), dense=True)
    sc, sel, pr = ol.empty((R, E)), ol.empty((R, E)), ol.empty((R, 2), dense=True)
    if mo.rule == "softmax":                            # (allocated before the shared expert:
        sc.set(lg[:, 0:E])                              # none of its words a WAR of the top-k)
        sel.set(sc)
    else:
        sc.set(sigmoid(lg[:, 0:E]))
        sel.set(sc + ol.load(lw.ebias)[None, :])
    gate = None
    if mo.shared:
        gs = ol.empty((R,))
        gs.column().set(lg[:, E:E + 1])
        gate = sigmoid(gs)
        del gs
    del lg
    lines, held = [], []

    def post(i=None):                                   # the next need line (MB.post, its count)
        if not lines:
            return
        r0 = lines.pop(0)
        N = min(per, R - r0) * k
        seq = MB.wait_served(dev.mbox)
        ol.store(Tensor(Affine(dev.mbox + LINE), (N,), (1,)),
                 held[0][r0:r0 + N // k, :].reshape(1, N)[0, :] + float(dev.need_off))
        ol.store(Tensor(Affine(dev.mbox + 4), (1,), (1,)), ol.full((1,), float(N)))
        seq.set(seq + 1.0)
        ol.store(Tensor(Affine(dev.mbox), (1,), (1,)), seq)
        del seq
        if not lines:
            held.clear()

    def store_sh(r0, sh):
        for q in range(sh.rows):
            ol.store(Tensor(em_row(dev, row, r0 + q) + 4 * (H * (k + 1) + sid * Hs),
                            (1, Hs), (Hs, 1)), sh[q:q + 1, :])

    def shared(store=True):                             # each row's SH (store: else returned,
        nonlocal xs, gate                               # G = R)
        out = None
        for r0 in range(0, R, G):
            n = min(G, R - r0)
            xg = xs if G == R else normed(r0, n)
            if S > 1:                                   # (its Hs columns: _moe_ffn_tp's)
                fsh = lw.wg.shape[0]
                csh = min(fsh, lw.wd.pw if lw.wd.parts is not None else fsh)
                y = _swiglu_rows_tp(xg, lw.wg, lw.wu, lw.wd, S * fsh, csh, ACTS[mo.act])
            else:
                y = swiglu_down(xg, lw.wg, lw.wu, lw.wd, each=post if lines else None)
            sh = y * (gate if G == R else gate[r0:r0 + n])[:, None]
            del xg, y
            if store:
                store_sh(r0, sh)
            else:
                out = sh
            del sh
        xs = gate = None
        return out

    # A resident pool's shared expert first, its MMs under the top-k's VPU chains, its stores
    # after them (the stores' DMA count, 4 bits, would hold the top-k's first ARGMAX).
    early = resident and mo.shared and G == R
    sh = shared(store=False) if early else None
    if not early and G < R:                             # (each pass quantizes its rows)
        xs = None
    tk = max(1, min(TOPK_ROWS, len(b.free_regs) - 1))   # (a register a row, one the loop's)
    for q0 in range(0, R, tk):                          # each row's k best, as moe_ffn_rows':
        n = min(tk, R - q0)                             # n rows a pass, the i-th best of each
        s0, e0, p0 = sc.base + q0 * sc.rs, sel.base + q0 * sel.rs, pr.base + 2 * q0
        rs = [b.scratch() for _ in range(n)]            # into column i (one ARGMAX, then a
        lp = b.begin_loop(k)                            # knock-out and a weight a row)
        b.emit(I.argmax(p0, e0, n, E, 2, sel.rs, comment="moe: best"))
        for j, r in enumerate(rs):
            b.emit(I.rld(r, p0 + 2 * j + 1, bound=(0, E - 1), comment="its index"))
            b.emit(I.vop(I.V_FILL, e0 + j * sel.rs, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, NEG, ra=r,
                         comment="knock out"))
        ri, ci = b.addr(Affine(gid.base + q0 * k) + lp)  # (wts' column i: the same register)
        b.emit(I.vop(I.V_COPY, ci, p0 + 1, 0, n, 1, k, 2, 0, ra=ri, comment="ids"))
        for j, r in enumerate(rs):
            b.emit(I.vop(I.V_COPY, ci - gid.base + wts.base + j * k, s0 + j * sc.rs, 0, 1, 1, 0,
                         0, 0, ra=ri, rb=r, comment="weight"))
        b.end_loop(lp)
        for r in rs:
            b.unscratch(r)
    del sc, sel, pr
    if sh is not None:
        store_sh(0, sh)
        del sh
    gb = ol.load(lw.gbase)
    gid.set(gid + gb)                                   # global ids: j * E + index
    del gb
    if mo.rule == "softmax":                            # column 0 holds the largest
        m = ol.empty((R,))
        m.column().set(wts[:, 0:1])
        e = ol.exp2((wts - m[:, None]) * ol.LOG2E)
        del m
        wts.set(e * ol.recip(ol.sum(e, axis=1))[:, None])
        del e
    elif mo.norm:
        wts.set(wts * ol.recip(ol.sum(wts, axis=1) + 1e-6)[:, None])
    if mo.scale != 1.0:
        wts.set(wts * float(mo.scale))
    tail = 4 * H * (k + 2)
    for q in range(R):                                  # the record tails
        t = em_row(dev, row, q) + tail
        ol.store(Tensor(t, (k,), (1,)), gid[q, :])
        ol.store(Tensor(t + 4 * k, (k,), (1,)), wts[q, :])
        ol.store(Tensor(t + 8 * k, (1,), (1,)), rinv[q:q + 1])
    del wts
    if G == R:
        del rinv
    if not resident:                                    # each need line's first row; the ids
        lines[:], held[:] = list(range(0, R, per)), [gid]   # (freed after the last line)
    del gid
    post()
    if mo.shared:
        if not early:
            shared()
        if G < R:
            del g, rinv
    elif beside is not None:
        sh = beside()
        for q in range(R):
            ol.store(Tensor(em_row(dev, row, q) + 4 * H * (k + 1), (1, H), (H, 1)),
                     sh[q:q + 1, :])
    while lines:
        post()


def _shared_rows(R: int, H: int, lw) -> int:
    """moe_prologue_rows' rows a shared expert pass: at S > 1 all R where one MM takes them
    (ACT_ROWS) and the pass's tiles a row (the slice's gate and up chunks, its activations and
    their gather, its Hs output columns and their product) take a quarter of TMEM at most: the
    router's QACT serves it and its weights stream once; else whole column groups (MCOLS rows:
    m4's TMEM at S = 1)."""
    cfg, S = current().cfg, ol.num_programs()
    if S > 1 and R <= cfg.ACT_ROWS:
        fsh = lw.wg.shape[0]
        csh = min(fsh, lw.wd.pw if lw.wd.parts is not None else fsh)
        if R * (fsh + S * fsh + 2 * csh + 2 * (H // S)) <= cfg.TMEM_WORDS // 4:
            return R
    return min(R, cfg.MCOLS)


def combine_stage(mo: MoESpec, H: int) -> list:
    """moe_combine_rows' staging: [k, H / S] tiles a row's expert outputs load into, the rows
    taking them in turn (two where they take at most an eighth of TMEM, else one)."""
    k, Hs = mo.k, H // ol.num_programs()
    return [ol.empty((k, Hs)) for _ in range(2 if 2 * k * Hs <= ol.tmem_words() // 8 else 1)]


def moe_combine_rows(x, mo: MoESpec, dev: SimpleNamespace, row, residual: bool = True,
                     stage: list | None = None):
    """Expert-major's end of a MoE layer (docs/offload.md 13.11), in the next layer's run (or
    the prefill's head): for each of the R rows x [R, H] from chunk row `row`, its experts'
    outputs weighted and summed in its router's order, from its record (OUT, the tail's
    weights), then the shared expert's output (SH) added, and x + that: moe_ffn_rows' sums,
    bit for bit. residual=False: the routed sum alone (Gemma 4 norms it beside the dense MLP's,
    which SH holds). A row's k outputs load into a staging tile's rows and are weighted there
    (stage: combine_stage's, the rows taking them in turn; a caller's across its calls), so no
    load waits on the reads of the one before: a load a fresh tile reused the words of the one
    two before, and the combine ran as one chain of k R load -> mul links."""
    R, H, k = x.rows, x.cols, mo.k
    S, sid = ol.num_programs(), ol.program_id()
    Hs, c0 = H // S, 4 * (H // S) * sid                 # (S > 1: the slice's columns, then
    out = ol.empty((R, Hs))                             # gathered: _moe_ffn_tp's)
    if S > 1 and not residual:
        raise CompileError("moe_combine_rows at S > 1: residual only")
    stage = combine_stage(mo, H) if stage is None else stage
    for q in range(R):                                  # each row's sum in its router's order
        t = em_row(dev, row, q)
        wt = None if EM_GSCALE else ol.load(Tensor(t + 4 * H * (k + 2) + 4 * k, (k,), (1,)))
        ys = stage[q % len(stage)]
        for i in range(k):
            ol.load(Tensor(t + 4 * H * (1 + i) + c0, (1, Hs), (Hs, 1)), ys[i:i + 1, :])
        o = out[q:q + 1, :]
        for i in range(k):                              # (EM_GSCALE: the outputs weighted)
            y = ys[i:i + 1, :]
            if wt is not None:
                y.set(y * wt[i:i + 1])
            if i == 1:
                o.set(ys[0:1, :] + y)
            elif i > 1:
                o.add(y)
        if k == 1:
            o.set(ys[0:1, :])
        del o, y, wt, ys
    if not residual:
        return out
    if mo.shared:
        out = out + ol.load(Tensor(em_row(dev, row) + 4 * H * (k + 1) + c0, (R, Hs),
                                   (dev.em_rec // 4, 1)))
    if S == 1:
        return x + out
    return ol.all_gather(x[:, sid * Hs:(sid + 1) * Hs] + out)


def em_mxu_cost(cfg, wformat: str, m: int) -> int:
    """An expert pass's MXU time on m rows, in cycles a chunk of its weights (docs/isa.md MM):
    4-bit weights a chunk a cycle with PAIR (2 m <= MCOLS), half a chunk without; int8 a chunk a
    cycle; the ACT RAM's rows past MCOLS replay each chunk, ceil(m / MCOLS) groups."""
    g = -(-m // cfg.MCOLS)
    if Q.mxu_wf(wformat) == 0:
        return g
    return 1 if cfg.PAIR and 2 * m <= cfg.MCOLS else 2 * g


def em_exact(cfg) -> bool:
    """A pass's MXU time follows its rows (GALL: an MM's groups advance a row at a time, so
    no rows ride free on a chunk): moe_expert_run's exact pass sizes, C as large as fits."""
    return bool(getattr(cfg, "GALL", False))


def em_pass_order(cfg, wformat: str, exact: bool = False) -> list:
    """Every pass size m <= ACT_ROWS, the best first: the least time an entry,
    max(EM_RHO, em_mxu_cost(m)) / m, the fewest rows on a tie (docs/offload.md 13.17). A pass
    streams its expert's weights once, so it costs the stream until the MXU takes longer; the
    stream is a little under a chunk a cycle (EM_RHO), and any value in (1, 2) orders them the
    same."""
    act = cfg.act_rows
    if exact:                                           # (em_exact: the fewest passes first)
        return list(range(act, 0, -1))
    return sorted(range(1, act + 1),
                  key=lambda m: (max(EM_RHO, em_mxu_cost(cfg, wformat, m)) / m, m))


def em_pass_rows(cfg, wformat: str) -> int:
    """C, the entries of one expert an expert-major pass takes: em_pass_order's first. 4 on the
    card (MCOLS 4, PAIR: 2 rows at the stream's rate, 4 at twice its MXU time), 16 at MCOLS 16
    with ACT_ROWS >= 16, 4 for int8 at MCOLS 4 (a chunk a cycle for every row)."""
    return em_pass_order(cfg, wformat)[0]


def em_short_rows(cfg, wformat: str, c: int) -> int:
    """The most rows a pass of C entries computes for less MXU time than C's (0: none): an
    expert's last pass, if it holds no more entries, computes only that many (on the card 2 of
    C = 4)."""
    cost = em_mxu_cost(cfg, wformat, c)
    return max([m for m in range(1, c) if em_mxu_cost(cfg, wformat, m) < cost], default=0)


# moe_expert_run's passes software-pipelined when it has no short ones: 1 pass p + 1's rows in and
# its QACT, and p - 1's stores, under pass p's MMs; 2 also pass p + 1's gate and up MMs before p's
# down ones (p's activations, GATHER and QACTs under them: _em_pipelined); 0 off
EM_PIPE = 2
# Pipelined exact passes (em_exact): a pass's rows rounded up to a multiple of EM_PIPE_STEP (its
# class: the MMs and QACTs of each class a LOOP of its flag; the rows past its entries its first
# entry's again, computed and stored again)
EM_PIPE_STEP = 1
# Each entry's router weight as its down MM's row scale (QACT RSCALE of the activations), so the
# expert run's outputs come weighted and moe_combine_rows only adds them (no MUL a rank: at R
# 512 the combine is ~1.2K VPU cycles a row a slice). Not moe_ffn_rows' bits (the weight before
# the activations' rounding): off until a dKL check (the formats rule) passes.
EM_GSCALE = False


def _em_pipelined(v: SimpleNamespace) -> None:
    """moe_expert_run's passes software-pipelined (S = 0; v: its locals, the tables built):
    pass p's gate and up MMs stream while pass p + 1's rows come in (their loads under the MXU;
    with g_exp their norm's VOPs too), pass p + 1's QACT follows p's down MMs (the norm's
    products in it: RSCALE / CSCALE, as moe_expert_run's), and pass p's outputs go to
    their places under pass p + 1's MMs. Every WAITW is a write of all DRAM to the wait entries
    (waits.footprint), so pass p + 1's entry is waited for, its slot read and its tag zeroed
    before pass p - 1's stores and p's MMs: it then waits only for p - 1's MMs, and nothing
    behind it for p's. The expert slots' bound is the slots past the run's scratch
    (dev.em_slots: the records' slots, which the stores write, are not an expert's), so the
    MMs do not wait for the stores. A half per pass, two a loop iteration: the passes' rows are
    quantized into two stationaries in turn (ACT RAM), the rows and outputs have one TMEM tile
    each (a pass's loads after the last one's QACT, its down MMs after the last one's stores).
    The pass table's flags F (a pass) and Fw (a pass before the row) gate each stage (LOOPs of
    count 0 / 1); floor(passes / 2) + 1 iterations take the last pass's stores. Each row's
    operations are moe_expert_run's, so the records are its bit for bit.

    Exact sizes (v.cls: more than one class): a pass takes its class's rows (its size rounded
    up, EM_PIPE_STEP; the places past its entries hold its first entry, computed and stored
    again), its QACT and MMs those of its class under a LOOP of the class's flag (the first rows
    of the stationaries and tiles: view), its rows in and out in bands, a band's under the
    "class q or above" flag. At S > 1 (tensor-parallel, a resident pool's) the MLP is
    _swiglu_rows_tp's and the outputs the slice's Hs columns (rows_of_pass').

    v.pipe 2 (deep): the MXU takes pass p + 1's gate and up MMs before pass p's down MMs, so
    p's activations, GATHER (S > 1) and down QACTs run under p + 1's gate / up, and the
    quantizer takes p's down QACTs, then p + 2's rows: the MXU and Q each busy from pass to
    pass. A half (pass p): p + 1's slot, p - 1's stores, p + 1's gate / up (S = 1: p + 2's
    rows in under them), p's GATHER (S > 1) and down (S > 1: then p + 2's rows in), p + 1's
    activations, p + 2's QACT. At S > 1 p's GATHER is in its class's LOOP before p's down MMs,
    after p + 1's gate / up MMs, not after p's activations: an RLD is a collective
    instruction, in order behind a GATHER, and the next flag's RLD (the next class's LOOP)
    held dispatch until the GATHER was done, there until p + 1's gate / up MMs, activations
    and GATHER were, with only p's down MMs queued. p + 2's rows come in once, not in each of
    p + 1's classes (the program half the size). x2:wmx's 35B expert run at R 512 (sequencer
    model, pk2): 3,028,757 -> 2,550,909 cycles, 5,956 -> 3,028 instructions. The gate and up
    outputs have one tile each, the activations two (one a pass in flight; S > 1: their
    GATHERs' destinations, the slice's own one tile), the stationaries two. Each pass's
    operations are the serial form's in its order (S = 1: swiglu_down's down MMs a chunk at a
    time; S > 1: _swiglu_rows_tp's), so the records are its bit for bit."""
    from ..compiler import Stationary, _Parts
    b, PT, PW, cE, C, H, dev = v.b, v.PT, v.PW, v.cE, v.C, v.H, v.dev
    one, cls, NS, Hs = v.one, v.cls, v.NS, v.Hs
    sb = em_slot_bound(dev)
    bq = (0, (v.PM + 2) * PW)
    xr, ri, o = v.pio
    st = [ol.stationary(C, H), ol.stationary(C, H)]
    rq, rn, rcount = v.rq, v.rn, v.rs
    rw, r1, rd, sc, s1, rf = (b.scratch() for _ in range(6))

    def flag(r: int, col: int, what: str):
        """A LOOP of count PT[R[r] + col] (a flag: 0 or 1)."""
        b.emit(I.rld(rf, PT.base + col, ra=r, bound=(0, 1), comment=what))
        return b.begin_loop(0, rcount=rf)

    def view(s_, m: int):
        """The stationary s_'s first m rows (its blocks) as one of their own: a class's."""
        if m == C:
            return s_
        (ab, _), = s_.chunks
        w = Stationary(_Parts(b, m, H), [(ab, m)], s_.KB, list(s_.owners))
        w.loop, w.parent = s_.loop, s_
        return w

    def classes(r: int, fn) -> None:
        """fn(m) for the pass at row R[r]: its class's m rows (each class under its flag)."""
        if len(cls) == 1:
            fn(C)
            return
        for q, m in enumerate(cls):
            lp = flag(r, v.cQ + q, f"rows {m}")
            fn(m)
            b.end_loop(lp)

    def bands(r: int, fn, gate: bool = False) -> None:
        """fn(j0, j1) for the pass at row R[r]: its rows j0 .. j1 - 1 by the classes' bands, a
        band past the first under the "class q or above" flag; gate: the first under F (a
        pass), beside them (LOOPs four deep at most)."""
        lp = flag(r, PW - 1, "a pass") if gate else None
        fn(0, cls[0])
        if lp is not None:
            b.end_loop(lp)
        for q in range(1, len(cls)):
            lp = flag(r, v.cG + q - 1, f"rows {cls[q - 1]} .. {cls[q] - 1}")
            fn(cls[q - 1], cls[q])
            b.end_loop(lp)

    def expert(r: int) -> None:
        """The pass at row R[r]: R[s1] its expert's place (em_slot: its entry waited for, its
        tag zeroed; a resident pool's place)."""
        em_slot(s1, PT.base + cE, r, dev, v.word, rd=rd, bound=sb)

    def rows_in(r: int, gate: bool = False, into=None) -> None:
        """The pass at row R[r]: its rows into xr, normed (moe_expert_run's rows_of_pass);
        gate: bands'. Quantize-once (v.qo): into the stationary `into` instead, each row's
        bytes from its record (LDA; quant then emits nothing)."""
        rec = DevVar("record", rn, align=LINE)

        def band(j0: int, j1: int) -> None:
            for j in range(j0, j1):
                b.emit(I.rld(rn, PT.base + cE + 2 + j, ra=r, bound=v.be, comment="entry"))
                b.emit(I.rld(rn, v.QR.base, ra=rn, mul=v.REC, bound=v.brec,
                             comment="its row's record"))
                if v.qo is not None:
                    ol.act_load(into, j, Affine(dev.em_base + v.qo) + rec)
                    continue
                ol.load(Tensor(Affine(dev.em_base) + rec, (1, H), (H, 1)), out=xr[j:j + 1, :])
                ol.load(Tensor(Affine(dev.em_base + v.tail + 8 * v.k) + rec, (1,), (1,)),
                        out=ri[j:j + 1])
                if v.ge is not None:
                    xr[j:j + 1, :].set((xr[j:j + 1, :] * ri[j:j + 1][:, None]) * v.g[None, :])
                    xr[j:j + 1, :].set(xr[j:j + 1, :] * v.ge[None, :])
        bands(r, band, gate)

    def quant(s_, m: int) -> None:
        """The pass's first m rows into the stationary s_: (x * 1 / rms) * g in the QACT
        (RSCALE / CSCALE; with g_exp rows_in's VOPs normed them)."""
        if v.qo is not None:                            # (rows_in's LDAs filled it)
            return
        x_, r_ = (xr, ri) if m == C else (xr[0:m, :], ri[0:m])
        if v.ge is None:
            ol.quantize_into(view(s_, m), x_, 0, cscale=v.g, rscale=r_)
        else:
            ol.quantize_into(view(s_, m), x_, 0)

    def rows_out(r: int) -> None:
        """The pass at row R[r]: its outputs o to their places (the slice's columns)."""
        def band(j0: int, j1: int) -> None:
            for j in range(j0, j1):
                b.emit(I.rld(rn, PT.base + cE + 2 + j, ra=r, bound=v.be, comment="entry"))
                b.emit(I.rld(rn, v.PL.base, ra=rn, mul=LINE, bound=v.bpl, comment="its place"))
                ol.store(Tensor(Affine(dev.em_base + 4 * Hs * v.sid)
                                + DevVar("place", rn, align=LINE), (1, Hs), (Hs, 1)),
                         o[j:j + 1, :])
        bands(r, band)

    def half(cur, nxt) -> None:
        """Pass p (row R[rq]; R[rw] row p - 1, R[sc] its slot) from the stationary cur, pass
        p + 1's rows into nxt."""
        b.emit(I.rld(r1, one.base, mul=PW, rb=rq, bound=bq, comment="row p + 1"))
        lp = flag(r1, PW - 1, "p + 1 a pass")
        expert(r1)
        b.end_loop(lp)
        lp = flag(rq, 0, "p - 1 a pass")
        rows_out(rw)
        b.end_loop(lp)
        lp = flag(rq, PW - 1, "p a pass")
        ex = dev.fmt.descs(DevVar("expert slot", sc, align=LINE))

        def each(i: int) -> None:                       # (gate and up MMs issued)
            if i == 0 and len(cls) > 1:                 # (in bands: p + 1's flags)
                rows_in(r1, gate=True, into=nxt)
            elif i == 0:
                lr = flag(r1, PW - 1, "p + 1 a pass")
                rows_in(r1, into=nxt)
                b.end_loop(lr)

        def mlp(m: int) -> None:
            om = o if m == C else o[0:m, :]
            if NS > 1:                                  # (its Hs columns: _moe_ffn_tp's)
                _swiglu_rows_tp(view(cur, m), ex.wg, ex.wu, ex.wd, dev.fmt.F, dev.fmt.C,
                                ACTS[v.mo.act], out=om, each=each)
            else:
                swiglu_down(view(cur, m), ex.wg, ex.wu, ex.wd, act=ACTS[v.mo.act], out=om,
                            each=each)
        classes(rq, mlp)
        lr = flag(r1, PW - 1, "p + 1 a pass")
        classes(r1, lambda m: quant(nxt, m))
        b.end_loop(lr)
        b.end_loop(lp)
        b.emit(I.rld(rw, one.base, mul=0, rb=rq, bound=bq, comment="row p (p - 1 next)"))
        b.emit(I.rld(rq, one.base, mul=PW, rb=rq, bound=bq, comment="row p + 1 (p next)"))
        b.emit(I.rld(sc, one.base, mul=0, rb=s1, bound=sb, comment="its slot (p's next)"))

    if v.level == 2 and _em_deep(SimpleNamespace(**locals())):
        return
    expert(rq)                                          # pass 0 (row 0, R[rw] 0)
    rows_in(rq, into=st[0])
    classes(rq, lambda m: quant(st[0], m))
    b.emit(I.rld(sc, one.base, mul=0, rb=s1, bound=sb, comment="its slot"))
    nh = v.NP * 0.5 + 1.0
    b.rld(rcount, nh, comment="passes / 2 + 1")
    lp = b.begin_loop(0, rcount=rcount)                 # (the count is read here)
    half(st[0], st[1])
    half(st[1], st[0])
    b.end_loop(lp)
    for r in (rq, rn, rcount, rw, r1, rd, sc, s1, rf):
        b.unscratch(r)


def _em_deep(u: SimpleNamespace) -> bool:
    """_em_pipelined's deep schedule (v.level 2; u: its locals, the helpers defined); False,
    nothing emitted, where it does not apply (a short chunk, W_down in the gather's order)."""
    from ..kernels.mlp import f_chunk
    v, b, PT, PW, C, dev, one = u.v, u.b, u.PT, u.PW, u.C, u.dev, u.one
    flag, view, classes, expert, rows_in, rows_out, quant = (
        u.flag, u.view, u.classes, u.expert, u.rows_in, u.rows_out, u.quant)
    rq, rw, r1, sc, s1, rcount, sb, st = u.rq, u.rw, u.r1, u.sc, u.s1, u.rcount, u.sb, u.st
    NS, act = v.NS, ACTS[v.mo.act]
    fmt = dev.fmt
    ex0 = fmt.descs(DevVar("expert slot", sc, align=LINE))     # (pass p's)
    ex1 = fmt.descs(DevVar("expert slot", s1, align=LINE))     # (pass p + 1's)
    Fs = ex0.wg.shape[0]
    Cc = min(fmt.C, Fs) if NS > 1 else f_chunk(ex0.wg, ex0.wd)
    if Fs % Cc or ex0.wd.gs != 1:
        return False
    gb, ub = ol.empty((C, Fs)), ol.empty((C, Fs))
    h = ol.empty((C, Fs)) if NS > 1 else None
    hg = [ol.empty((C, fmt.F if NS > 1 else Fs)) for _ in range(2)]
    if NS > 1 and NS * Fs < fmt.F:                      # (_gather_cols' zero columns)
        for t in hg:
            t[:, NS * Fs:fmt.F].set(0.0)
    r2 = b.scratch()
    bq2 = (0, (v.PM + 3) * PW)

    def rows(t, m: int):
        return t if m == C else t[0:m, :]

    def gate_up(m: int, s_, ex, c0s) -> None:
        """The pass's gate and up MMs (its m rows from the stationary s_) into gb / ub, of the
        chunks at columns c0s."""
        xs = view(s_, m)
        for c0 in c0s:
            ol.dot(xs, ex.wg[c0:c0 + Cc, :], out=gb[0:m, c0:c0 + Cc])
            ol.dot(xs, ex.wu[c0:c0 + Cc, :], out=ub[0:m, c0:c0 + Cc])

    def gate_up_in(r: int, s_, ex, nxt: int | None, into=None) -> None:
        """The gate and up MMs of the pass at row R[r] (its class's); nxt: the rows of the pass
        at R[nxt] in once the first chunk's are issued (once, between two LOOPs of the classes,
        not in each class's; quantize-once: into the stationary `into`)."""
        if nxt is None:
            classes(r, lambda m: gate_up(m, s_, ex, range(0, Fs, Cc)))
            return
        classes(r, lambda m: gate_up(m, s_, ex, (0,)))
        next_rows(nxt, into)
        if Fs > Cc:
            classes(r, lambda m: gate_up(m, s_, ex, range(Cc, Fs, Cc)))

    def acts(m: int, k: int) -> None:
        """The pass's activations from gb / ub into hg[k] (S > 1: into h, down's GATHER)."""
        dst = h if NS > 1 else hg[k]
        for c0 in range(0, Fs, Cc):
            dst[0:m, c0:c0 + Cc].set(act(gb[0:m, c0:c0 + Cc]) * ub[0:m, c0:c0 + Cc])

    def down(m: int, k: int, ex) -> None:
        """The pass's down MMs from hg[k] into o (the first overwriting); S > 1: h gathered into
        hg[k] first (half's: after pass p + 1's gate and up MMs)."""
        om = rows(u.o, m)
        if NS > 1:
            hm, gm = rows(h, m), rows(hg[k], m)
            b.emit(I.gather(hm.base, gm.base, m, Fs, hm.rs, gm.rs, Fs, comment="all-gather"))
            b.bump_version(hg[k].buf)
            _down_parts(rows(hg[k], m), ex.wd, None, om)
            return
        a = rows(hg[k], m)
        for i, c0 in enumerate(range(0, Fs, Cc)):
            cols = ex.wd[:, c0:c0 + Cc]
            if i == 0:
                ol.dot(a[:, c0:c0 + Cc], cols, out=om)
            else:
                ol.dot(a[:, c0:c0 + Cc], cols, acc=om)

    def next_rows(r: int, into) -> None:
        """The rows of the pass at row R[r] in (under its F; quantize-once: into the stationary
        `into`)."""
        if len(u.cls) > 1:
            rows_in(r, gate=True, into=into)
        else:
            lr = flag(r, PW - 1, "its a pass")
            rows_in(r, into=into)
            b.end_loop(lr)

    def half(k: int) -> None:
        """Pass p (row R[rq], k = p mod 2; R[rw] row p - 1, R[sc] p's slot)."""
        b.emit(I.rld(r1, one.base, mul=PW, rb=rq, bound=u.bq, comment="row p + 1"))
        b.emit(I.rld(r2, one.base, mul=2 * PW, rb=rq, bound=bq2, comment="row p + 2"))
        lp = flag(r1, PW - 1, "p + 1 a pass")
        expert(r1)
        b.end_loop(lp)
        lp = flag(rq, 0, "p - 1 a pass")
        rows_out(rw)
        b.end_loop(lp)
        lp = flag(r1, PW - 1, "p + 1 a pass")
        gate_up_in(r1, st[1 - k], ex1, r2 if NS == 1 else None, st[k])
        b.end_loop(lp)
        lp = flag(rq, PW - 1, "p a pass")
        classes(rq, lambda m: down(m, k, ex0))
        b.end_loop(lp)
        if NS > 1:                                      # (p + 2's rows: after p's GATHER)
            next_rows(r2, st[k])
        lp = flag(r1, PW - 1, "p + 1 a pass")
        classes(r1, lambda m: acts(m, 1 - k))
        b.end_loop(lp)
        lp = flag(r2, PW - 1, "p + 2 a pass")
        classes(r2, lambda m: quant(st[k], m))
        b.end_loop(lp)
        b.emit(I.rld(rw, one.base, mul=0, rb=rq, bound=u.bq, comment="row p (p - 1 next)"))
        b.emit(I.rld(rq, one.base, mul=0, rb=r1, bound=u.bq, comment="row p + 1 (p next)"))
        b.emit(I.rld(sc, one.base, mul=0, rb=s1, bound=sb, comment="its slot (p's next)"))

    expert(rq)                                          # pass 0 (row 0, R[rw] 0)
    rows_in(rq, into=st[0])
    classes(rq, lambda m: quant(st[0], m))
    b.emit(I.rld(sc, one.base, mul=0, rb=s1, bound=sb, comment="its slot"))
    b.emit(I.rld(r1, one.base, mul=PW, rb=rq, bound=u.bq, comment="row 1"))
    gate_up_in(rq, st[0], ex0, r1, st[1])
    classes(rq, lambda m: acts(m, 0))
    lp = flag(r1, PW - 1, "pass 1")
    classes(r1, lambda m: quant(st[1], m))
    b.end_loop(lp)
    nh = v.NP * 0.5 + 1.0
    b.rld(rcount, nh, comment="passes / 2 + 1")
    lp = b.begin_loop(0, rcount=rcount)                 # (the count is read here)
    half(0)
    half(1)
    b.end_loop(lp)
    for r in (rq, u.rn, rcount, rw, r1, r2, u.rd, sc, s1, u.rf):
        b.unscratch(r)
    return True


def moe_expert_run(lw, mo: MoESpec, dev: SimpleNamespace, rows, entries, C: int = 2,
                   S: int = 0, exact: bool = False, pipe: int = 2) -> None:
    """Expert-major's expert run of one MoE layer (docs/offload.md 13.11, 13.17), after the
    layer's mixer runs over a chunk's `rows` rows (run argument rows; entries = rows x k, its
    callers' second, the program no longer reads):
    every (row, rank) entry's expert once on its row, each expert in passes of up to C of its
    entries, each pass streaming the expert's weights once (C: em_pass_rows).

    1. The tables: per row (a loop of `rows`) its k ids from its record tail, at TMEM's top;
       each entry's row and its output's place (the record's OUT row of its rank, in lines) by
       VOPs over every row at once (both affine in the row and the rank).
    2. The bucketing (a loop of `rows`, each a loop of its k entries): each expert's next place
       (POS, a word of the pass table; at first the sink row's M), and past each pass's C places
       a mark M = 1. Three RLDs an entry: its expert, its next place, the word there; a LOOP of
       that word (1: the pass is full, or the expert has none) opens a pass (its expert, a short
       flag 0, its sizes; its first place the next), then the entry joins the place and POS
       moves on. The RLDs are bounded (the row's first entry too, an RLD a row), so that the
       waits pass keeps their reads and the VOPs' ranges in their tables. After the loop each
       expert's free places in its last pass (BASE - POS), and a pass's unused places its first
       entry's row, so that a pass of fewer entries computes its first row again and stores the
       same output again. Registers reach the tables as FILL's 2^23 + R, made exact after. The
       pass table holds entries / C + E passes at most, in the order they open.
       With S (em_short_rows: a smaller pass's MXU costs less) a loop over the experts then
       flags each one's last pass that holds at most S entries (free >= C - S); the others'
       flags go to a sink row.
    3. The passes (a loop of the passes counted): WAITW on the expert's directory entry (its
       slot word != 0: the server writes an entry after its expert's data and tag, and keeps
       a needed expert's slot until this run has read it), its slot's tag zeroed (as slot_of),
       then the pass's C rows (S for a flagged pass, a LOOP of count flag and one of 1 - flag)
       normed again from their X and 1 / rms and quantized (moe_ffn_rows' expression),
       swiglu_down at the slot, each output row to its place. Each row's quantization and MM
       outputs are its own (docs/isa.md: one exact integer an output), so the outputs are
       C = 2's bit for bit. Without S (EM_PIPE, pipe) the passes are software-pipelined, a
       pass's rows in and outputs out under its neighbours' MMs (_em_pipelined), the same
       records.

    exact (em_exact: the MXU's time follows a pass's rows): each pass computes its own entries
    only, no first row again. A pass's row of the table carries C size flags (1 at its
    entries' count: C when it opens, then each expert's last pass its C - free, as S's flags),
    and the passes' loop runs rows_of_pass(m) under a LOOP of count flag m for m = 1 .. C;
    pipelined, each pass's rows rounded up to a class (EM_PIPE_STEP), its flags in the table
    (one a class, and "class q or above" for the rows' bands).

    The entry wait stops at MB.TIMEOUT with an error the host sees: the prefill never runs on
    without an expert."""
    b = current()
    _plain(mo, "moe_expert_run")
    NS, sid = ol.num_programs(), ol.program_id()
    if NS != 1 and getattr(dev, "pool", None) is None:
        raise CompileError("moe_expert_run at S > 1: a resident pool's (tensor-parallel)")
    if not 1 <= C or not 0 <= S < C or exact and S:
        raise CompileError(f"expert run passes of {C} entries, short ones of {S}"
                           + (", exact sizes" if exact else ""))
    E, k, H = mo.E, mo.k, dev.fmt.H
    Hs = H // NS                                        # (S > 1: a slice's output columns)
    gsc = EM_GSCALE                                     # (the router weights in the passes)
    # (3: pipelined, level 1 or 2 (EM_PIPE's); exact sizes on views of the passes'
    # stationaries: not with PAIR's rows)
    level = min(int(pipe), int(EM_PIPE)) if not S and not gsc and not (
        exact and b.cfg.PAIR and 2 * C <= b.cfg.MCOLS) else 0
    pipe = level > 0
    # the pass table's rows: [expert, short, C entries, M], pipelined [Fw, expert, short, C
    # entries, M, F] (M: the bucketing's mark past the entries; F: a pass; Fw: a pass before this
    # row, the last row's F and the next row's Fw one FILL); exact: C size flags after M,
    # pipelined then a pass's class flags (one-hot; a class a size, EM_PIPE_STEP 1: the size
    # flags) and its "class q or above" flags (q = 1 ..) before F
    cE = 1 if pipe else 0
    cls = sorted({min(C, m) for m in range(EM_PIPE_STEP, C + EM_PIPE_STEP, EM_PIPE_STEP)}) \
        if pipe and exact else [C]
    sizes = cls == list(range(1, C + 1))                # (the classes the sizes)
    NQ = (len(cls) - 1 if sizes else 2 * len(cls) - 1) if len(cls) > 1 else 0
    N, REC, PW = dev.em_rows * k, dev.em_rec, C + 3 + 2 * cE + (C if exact else 0) + NQ
    cM = cE + 2 + C                                     # (M)
    cF = cM + 1                                         # (exact: the size flags' columns)
    cQ = cF if sizes else cF + C                        # (the class flags, "or above" flags)
    cG = cF + C + (0 if sizes else len(cls))
    PM = N // C + E
    tail = 4 * H * (k + 2)
    # pipelined: the passes' rows and outputs (one tile each for the run) before the tables, whose
    # and the bucketing's leftovers leave no gap of the rows' [C, H] (x2:wmx: 32,688 of 32,768)
    # (quantize-once: no rows tile, the rows' bytes go from their records into ACT RAM)
    qo = em_q(dev, H, k)
    pio = ((ol.empty((C, H), dense=True), ol.empty((C,))) if qo is None else (None, None)) \
        + (ol.empty((C, Hs)),) if pipe else None
    # 1. the tables: ids (global), each entry's row, each entry's place (lines from the base);
    # EM_GSCALE: each entry's router weight. The ids (and weights) at TMEM's top: their loads at
    # L + R[rt] and the bucketing's reads at L + R[rp] (registers the waits pass widens) then
    # overlap only them, not a table the bucketing writes nor the passes' rows. The rows and
    # places are affine in the row and the rank, every row's at once (VOPs, no loop): the
    # row-at-a-time FILLs' ranges at R[rt] overlapped each other, each waiting for the one before
    L = b.alloc((N,), top=True)
    WT = b.alloc((N,), top=True) if gsc else None
    one = ol.full((1,), 1.0)                            # (the bucketing's RLDs' 1: filled here,
                                                        # its FILL past their wait entries' VOPs)
    QR, PL = ol.empty((N,)), ol.empty((N,))
    nr = dev.em_rows
    QR2, PL2 = QR.reshape(nr, k), PL.reshape(nr, k)
    q0 = QR2[:, 0:1]                                    # (row r: r, by doubling)
    q0[0:1, :].set(0.0)
    s = 1
    while s < nr:
        m = min(s, nr - s)
        q0[s:s + m, :].set(q0[0:m, :] + float(s))
        s *= 2
    if k > 1:
        QR2[:, 1:k].set(q0[:, None])                    # its entries' row
    for i in range(k):                                  # row 0's places: (1 + i) H / 16 lines
        PL2[0, i:i + 1].set(float((1 + i) * H // 16))
    if nr > 1:                                          # row r's: r records more
        PL2[1:, :].set(QR2[1:, :] * float(REC // LINE))
        PL2[1:, :].add(PL2[0, :][None, :])
    rr = b.arg_reg(rows, 1)
    rd, rt = b.scratch(), b.scratch()
    lp = b.begin_loop(0, rcount=rr)
    b.emit(I.ld(dev.em_base + tail, L.base, k, ra=rd, rb=rt, comment="the row's ids"))
    if gsc:
        b.emit(I.ld(dev.em_base + tail + 4 * k, WT.base, k, ra=rd, rb=rt,
                    comment="its weights"))
    b.emit(I.addi(rd, rd, REC, comment="next record"))
    b.emit(I.addi(rt, rt, k, comment="its entries"))
    b.end_loop(lp)
    for r in (rd, rt):
        b.unscratch(r)
    gb = ol.load(lw.gbase)
    L.set(L - gb)                                       # the layer's own indices
    # 2. the bucketing
    # (S: each expert's last pass, BASE, below the ids at the top: the loop over the experts
    # reads it at BASE + R[re], a register the waits pass widens, which then overlaps no table
    # it writes)
    BASE = b.alloc((E,), top=True) if S else None
    if S:
        BASE.set(_TWO23 + PM * PW)                      # (the sink row, for the unused)
    PT = ol.empty((PM + 2 + level if pipe else PM + 1, PW), dense=True)
    POS = ol.full((E,), _TWO23 + PM * PW + cM)          # (no pass: the sink row's M; before PT's
    if pipe:                                            # FILLs, so that the place's RLD finds its
        PT.set(_TWO23)                                  # FILL back: rows past the passes F, Fw 0)
    PT[:, cE + 2:cM].set(0.0)                           # (no entry yet)
    PT[:, cM:cM + 1].set(1.0)
    rp, rnp, r1, r3, r4 = (b.scratch() for _ in range(5))
    bpos = (1 << 23) + cE + 2, (1 << 23) + PM * PW + cM
    # (each entry: its expert, its next place and the word there: 0 a place, 1 M (its pass full,
    # or none); a LOOP of that count opens a pass, its first place the next; the entry into the
    # place. Three RLDs, the next one's place read once the place's FILL is done; bounded, so
    # that the waits pass keeps their VOPs' ranges in their tables. A loop of the rows, each a
    # loop of its k: the row's first entry from a bounded RLD, so that its expert's read at
    # L + R[rp] is L's, not any word, and does not wait for the join's FILL before it)
    lp = b.begin_loop(0, rcount=rr)
    b.emit(I.rld(rp, one.base, mul=0, rb=rp, bound=(0, N - k), comment="the row's first entry"))
    lk = b.begin_loop(k)
    b.emit(I.rld(r1, L.base, ra=rp, bound=(0, E - 1), comment="its expert"))
    b.emit(I.rld(r4, POS.base, ra=r1, bound=bpos, comment="its next place (2^23 + word)"))
    b.emit(I.rld(r3, PT.base - (1 << 23), ra=r4, bound=(0, 1),
                 comment="the word there: 1 (M) its pass full or none"))
    opn = b.begin_loop(0, rcount=r3)
    b.emit(I.vop(I.V_FILL, PT.base + cE, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, _TWO23, ra=rnp,
                 rd=r1, comment="open: its expert"))
    b.emit(I.vop(I.V_FILL, PT.base + cE + 1, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, _TWO23, ra=rnp,
                 comment="open: not short"))
    if pipe:
        b.emit(I.vop(I.V_FILL, PT.base + PW - 1, 0, 0, 1, 2, 0, 0, 0, I.B_SCALAR, _TWO23 + 1,
                     ra=rnp, comment="open: a pass (F), one before the next row (its Fw)"))
    if S:
        b.emit(I.vop(I.V_FILL, BASE.base, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, _TWO23, ra=r1,
                     rd=rnp, comment="its open pass"))
    b.emit(I.rld(r4, one.base, mul=(1 << 23) + cE + 2, rb=rnp,
                 bound=((1 << 23) + cE + 2, (1 << 23) + (PM - 1) * PW + cE + 2),
                 comment="its first place (2^23 + word)"))
    b.emit(I.rld(rnp, one.base, mul=PW, rb=rnp, bound=(PW, PM * PW), comment="next pass"))
    b.end_loop(opn)
    b.emit(I.addi(r3, r4, 1 - (1 << 23), comment="the next place's word"))
    b.emit(I.vop(I.V_FILL, POS.base, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, _TWO23, ra=r1, rd=r3,
                 comment="next place"))                # (first: the next entry's RLD waits on it)
    b.emit(I.vop(I.V_FILL, PT.base - (1 << 23), 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, _TWO23,
                 ra=r4, rd=rp, comment="join: the entry"))
    b.emit(I.addi(rp, rp, 1, comment="next entry"))
    b.end_loop(lk)
    b.end_loop(lp)
    NP = ol.full((1,), _TWO23)
    b.emit(I.vop(I.V_FILL, NP.base, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, _TWO23, rd=rnp,
                 comment="passes x PW"))
    for r in (rp, rnp, r1, r3, r4):
        b.unscratch(r)
    b.release_arg(rows)
    # (its last pass's free places; none: 0)
    FREE = (BASE - POS) + float(C + cE + 2) if S else None
    del L, POS, one
    U = ol.minimum(PT[:, cE + 2:cM], 1.0)               # (1: a place an entry took)
    if exact:                                           # each pass's size flags: its places
        if C > 1:                                       # taken a prefix, flag m (column m - 1)
            PT[:, cF:cF + C - 1].set(U[:, 0:C - 1] - U[:, 1:C])     # [m taken, m + 1 not]
        PT[:, cF + C - 1:cF + C].set(U[:, C - 1:C])
        if NQ and sizes:                                # class q or above: place q taken
            PT[:, cG:cG + C - 1].set(U[:, 1:C])
        elif NQ:                                        # class q: sizes lo + 1 .. hi
            for q, hi in enumerate(cls):
                lo = cls[q - 1] if q else 0
                fq = PT[:, cQ + q:cQ + q + 1]
                fq.set(U[:, lo:lo + 1] - U[:, hi:hi + 1] if hi < C else U[:, lo:lo + 1])
                if q:
                    PT[:, cG + q - 1:cG + q].set(U[:, lo:lo + 1])
                del fq
        PT[:, cF:cF + C + NQ].add(_TWO23)
    P0 = PT[:, cE + 2:cE + 3]                           # (a pass of fewer entries: its first
    P0.set(ol.maximum(P0, _TWO23))                      # entry's row again, the same output
    if C > 1:                                           # again; rows of no pass 2^23)
        EP, T = PT[:, cE + 3:cM], U[:, 1:C]
        T.set(1.0 - T)                                  # (1: a place no entry took)
        b.emit(I.vop(I.V_MUL, T.base, T.base, P0.base, PT.shape[0], C - 1, T.rs, T.rs, PW,
                     I.B_ROW, comment="its first entry's row there"))
        b.bump_version(T.buf)
        EP.set(EP + T)
        del EP, T
    del P0, U
    if S:                                               # each expert's last pass: short?
        SH = ol.empty((E,))
        b.emit(I.vop(I.V_SUB, SH.base, FREE.base, 0, 1, E, 0, 0, 0, I.B_SCALAR,
                     float(C - S - 1), comment="free - (C - S - 1)"))
        b.emit(I.vop(I.V_MAX, SH.base, SH.base, 0, 1, E, 0, 0, 0, I.B_SCALAR, 0.0))
        b.emit(I.vop(I.V_MIN, SH.base, SH.base, 0, 1, E, 0, 0, 0, I.B_SCALAR, 1.0,
                     comment="short: free >= C - S"))
        re, r4 = b.scratch(), b.scratch()
        lp = b.begin_loop(E)
        b.emit(I.rld(r4, BASE.base, ra=re, bound=(1 << 23, (1 << 23) + PM * PW),
                     comment="its last pass (2^23 + word)"))
        b.emit(I.vop(I.V_ADD, PT.base + cE + 1 - (1 << 23), SH.base, 0, 1, 1, 0, 0, 0,
                     I.B_SCALAR, _TWO23, ra=r4, rb=re, comment="its short flag"))
        b.emit(I.addi(re, re, 1, comment="next expert"))
        b.end_loop(lp)
        b.unscratch(re)
        b.unscratch(r4)
        del SH
    del FREE, BASE
    PT.add(-_TWO23)                                     # (in place: no PT-sized temp)
    PT[:, cE:cE + 1].set(PT[:, cE:cE + 1] + gb)         # global ids
    NP.set((NP - _TWO23) * (1.0 / PW))
    del gb
    # 3. the passes (quantize-once, qo: each row's input from its record's em_q, LDA)
    g = ol.load(lw.g_post) if qo is None else None
    ge = ol.load(lw.g_exp) if getattr(lw, "g_exp", None) is not None and qo is None else None
    word, one = ol.empty((1,)), ol.full((1,), 1.0)
    rq, rs, rn = b.scratch(), b.scratch(), b.scratch()
    # what the run knows of its registers (isa.rld's bounds, for the wait entries): a pass's
    # row of the table, an entry, its row's record and its output's place, in bytes
    bq = (0, (PM - 1) * PW)
    be = (0, N - 1)
    brec = (0, (dev.em_rows - 1) * REC)
    bpl = (0, ((dev.em_rows - 1) * (REC // LINE) + k * H // 16) * LINE)
    if pipe:
        _em_pipelined(SimpleNamespace(**locals()))
        return

    def rows_of_pass(M: int) -> None:
        """The pass's first M places: their rows normed, quantized, through the expert, out.
        Each row loads into its own row of the [M, H] tile and its 1 / rms into its own word,
        and the RLDs read words no VOP of the pass writes (bounded); the norm's products in the
        QACT (moe_ffn_rows' expressions, a row at a time; with g_exp VOPs a row, row j + 1's
        loads under row j's)."""
        xr = ol.empty((M, H), dense=True) if qo is None else None
        ri = ol.empty((M,)) if qo is None else None
        xe = ol.stationary(M, H) if qo is not None else None
        wv = ol.empty((M,)) if gsc else None
        rec = DevVar("record", rn, align=LINE)
        for j in range(M):
            b.emit(I.rld(rn, PT.base + 2 + j, ra=rq, bound=be, comment="entry"))
            if gsc:
                b.emit(I.vop(I.V_COPY, wv.base + j, WT.base, 0, 1, 1, 0, 0, 0, rb=rn,
                             comment="its weight"))
            b.emit(I.rld(rn, QR.base, ra=rn, mul=REC, bound=brec, comment="its row's record"))
            if qo is not None:                          # (quantized once: the prologue's)
                ol.act_load(xe, j, Affine(dev.em_base + qo) + rec)
                continue
            ol.load(Tensor(Affine(dev.em_base) + rec, (1, H), (H, 1)), out=xr[j:j + 1, :])
            ol.load(Tensor(Affine(dev.em_base + tail + 8 * k) + rec, (1,), (1,)),
                    out=ri[j:j + 1])
            if ge is not None:
                xr[j:j + 1, :].set((xr[j:j + 1, :] * ri[j:j + 1][:, None]) * g[None, :])
                xr[j:j + 1, :].set(xr[j:j + 1, :] * ge[None, :])
        # (x * 1 / rms) * g in the quantizer (QACT RSCALE / CSCALE: the VOPs' products, no
        # VOP a row: the norm of every entry's row is the expert run's VPU time at large R)
        if qo is None:
            xe = ol.quantize(xr) if ge is not None else ol.quantize(xr, rscale=ri, cscale=g)
        del xr, ri
        ex = dev.fmt.descs(DevVar("expert slot", rs, align=LINE))
        if NS > 1:                                      # (its Hs columns: _moe_ffn_tp's)
            o = _swiglu_rows_tp(xe, ex.wg, ex.wu, ex.wd, dev.fmt.F, dev.fmt.C, ACTS[mo.act],
                                rscale=wv)
        else:
            o = swiglu_down(xe, ex.wg, ex.wu, ex.wd, act=ACTS[mo.act], rscale=wv)
        b.check_live(o)
        for j in range(M):
            b.emit(I.rld(rn, PT.base + 2 + j, ra=rq, bound=be, comment="entry"))
            b.emit(I.rld(rn, PL.base, ra=rn, mul=LINE, bound=bpl, comment="its place"))
            ol.store(Tensor(Affine(dev.em_base + 4 * Hs * sid) + DevVar("place", rn, align=LINE),
                            (1, Hs), (Hs, 1)), o[j:j + 1, :])
        del o, xe

    b.rld(rs, NP, comment="passes")
    lp = b.begin_loop(0, rcount=rs)                     # (the count is read here: rs is free)
    # rq again, as 1.0 x rq with the table's bound: the loop's count is the run's, so the wait
    # entries take rq as any value, and every RLD of the table would wait for every TMEM write
    # in flight (each row's for the row before's VOPs)
    b.emit(I.rld(rq, one.base, mul=0, rb=rq, bound=bq, comment="the pass's row (bounded)"))
    em_slot(rs, PT.base, rq, dev, word, bound=getattr(dev, "slots", None))   # its expert's place
    if S:
        rf, rg = b.scratch(), b.scratch()
        b.emit(I.rld(rf, PT.base + 1, ra=rq, bound=(0, 1), comment="short"))
        for x in I.rld_mul(rg, PT.base + 1, -1, ra=rq, comment="-short", aw=b.aw):
            b.emit(x)
        short = b.begin_loop(0, rcount=rf)
        rows_of_pass(S)
        b.end_loop(short)
        full = b.begin_loop(1, rcount=rg)               # 1 - short
        rows_of_pass(C)
        b.end_loop(full)
        b.unscratch(rf)
        b.unscratch(rg)
    elif exact:                                         # its entries' count of rows
        rf = b.scratch()
        for m in range(1, C + 1):
            b.emit(I.rld(rf, PT.base + cF + m - 1, ra=rq, bound=(0, 1), comment=f"size {m}"))
            size = b.begin_loop(0, rcount=rf)
            rows_of_pass(m)
            b.end_loop(size)
        b.unscratch(rf)
    else:
        rows_of_pass(C)
    b.emit(I.addi(rq, rq, PW, comment="next pass"))
    b.end_loop(lp)
    for r in (rq, rs, rn):
        b.unscratch(r)
