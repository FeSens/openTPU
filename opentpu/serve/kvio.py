"""A context's bytes on the host: the KV caches' first n positions of an Engine's sequence and
the state they do not hold, read from the device and written back (the server's KV store,
opentpu/serve/store.py).

Every positional KV cache (compiler.KVDesc: K token-major [cap, d] int8 with its scales [cap,
d / D], V^T dim-major in tiles of P = vt_tile(cap) tokens, each tile [d, P] contiguous, one V
scale per token; Gemma 4's _KV with each K row's scales after its data) keeps the positions
[0, n) in contiguous byte ranges: K's first n rows, its scales' first n, V^T's first ceil(n / P)
tiles, V's first n scales. A page is P positions of every cache of every layer: page i holds
what the tokens [0, (i + 1) P) put there and nothing else, so contexts that share a prefix
share its whole pages, and a context of n tokens is its whole pages, its last partial one
(m = n - i P positions; its V^T tile whole) and the state the caches do not hold
(Image.state_regions: Qwen3.5's DeltaNet states and convolution windows, LFM2's convolution
rings, Gemma 4's sliding rings when shorter than the cache), at the context's position. An MTP
image's (Spec.mtp, opentpu/llm/mtp.py) pages hold its MTP layer's KV cache too, and its state is
the one in the decoder's committed slot (ContextIO.slot: the DeltaNet states' two slots).

Restoring pages [0, ceil(k / P)) and the position k gives the context of the k tokens they were
saved for, bit for bit: what lies past k in the last page is overwritten by the next appends
and masked before (as Engine.rollback's). A model with state restores only at a position whose
state was saved.

Two ways to move the bytes: direct (the backend's read / write per region: the ISA simulator,
or a card without room) and staged (on the card: LD / ST programs gather the regions into a
contiguous staging area of the DRAM past the image and the program area, one DMA moves it, and
the other way scatter them: a page of Qwen3-0.6B is 896 regions, ~30 us of DMA set-up each).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..compiler import Affine, k_scale_bytes, vt_tile
from ..llm import snapshot as SN

STAGING_MAX = 256 << 20     # the staging area's bytes at most (a batch of pages per copy)


@dataclass(frozen=True)
class Array:
    """One KV cache array of one head: `w` bytes per position (tile: V^T, the bytes of one
    position in each of its `w` rows, a tile [w, P] per page)."""
    base: int
    w: int
    tile: bool = False


def _static(x) -> int:
    return Affine.of(x).static()


def kv_arrays(image, seq: int = 0) -> list[Array]:
    """The positional KV caches' arrays of sequence `seq` (every slice's: the same addresses),
    layer by layer, head by head, an MTP image's MTP layer (Spec.mtp: layer `layers`) last. A
    layer that shares another's cache (Gemma 4) and a ring shorter than the cache (its sliding
    layers: state_regions) add none."""
    m = image.descriptors(0)
    out, seen = [], set()
    for li in range(image.spec.layers + bool(getattr(image.spec, "mtp", False))):
        lw = m.layer(li)
        kvs = getattr(lw, "kvs", None)
        if kvs is None:
            kv = getattr(lw, "kv", None)
            kvs = [kv] if kv is not None else []
        if not kvs:
            continue
        kv = kvs[seq]
        if kv.cap != image.cap:
            continue
        for h in sorted(kv.heads):
            e = kv.heads[h]
            k = _static(e["k"])
            if k in seen:
                continue
            seen.add(k)
            if hasattr(kv, "ps"):           # Gemma 4: K rows with their scales after them
                out += [Array(k, kv.ps), Array(_static(e["vt"]), kv.d, True),
                        Array(_static(e["vs"]), 4)]
            else:
                out += [Array(k, kv.d), Array(_static(e["ks"]), k_scale_bytes(kv.d, kv.D)),
                        Array(_static(e["vt"]), kv.d, True), Array(_static(e["vs"]), 4)]
    return out


class Layout:
    """Where an Engine's context lives in its DRAM: page regions and state regions (one slice's;
    every slice has the same addresses)."""

    def __init__(self, image, S: int, seq: int = 0):
        self.cap, self.S = image.cap, S
        self.P = vt_tile(image.cap)
        self.arrays = kv_arrays(image, seq)
        self.image = image
        self.state = SN.merged(image.state_regions(0)) if SN.stateful(image) else []
        self.state_bytes = S * sum(n for _, n in self.state)
        self.stateful = bool(self.state)
        self.pages = -(-self.cap // self.P)
        self._slots = {0: self.state}

    def state_in(self, slot: int) -> list[tuple[int, int]]:
        """The state regions of the DeltaNet states' `slot` (an MTP image's: two; the same
        bytes as slot 0's)."""
        if slot not in self._slots:
            self._slots[slot] = SN.merged(self.image.state_regions(slot))
        return self._slots[slot]

    def regions(self, i: int, m: int | None = None) -> list[tuple[int, int]]:
        """Page i's regions holding its first m positions (default all P): rows m at a time,
        V^T tiles whole."""
        P = self.P
        m = P if m is None else m
        if not 0 < m <= P or not 0 <= i < self.pages:
            raise ValueError(f"page {i} with {m} positions")
        out = []
        for a in self.arrays:
            if a.tile:
                out.append((a.base + i * P * a.w, P * a.w))
            else:
                out.append((a.base + i * P * a.w, m * a.w))
        return out

    def page_bytes(self, m: int | None = None) -> int:
        """A page's bytes, every slice's (m positions)."""
        return self.S * sum(n for _, n in self.regions(0, m))

    def split(self, n: int) -> list[tuple[int, int]]:
        """The pages (index, positions) holding a context of n tokens."""
        return [(i, min(self.P, n - i * self.P)) for i in range(-(-n // self.P))]


class ContextIO:
    """Reads and writes an Engine's pages and state as host bytes (each slice's after the
    other's). area: the staging area (base, bytes) in the device's DRAM; default
    staging_area(eng) (a card's: the DRAM past the program area), None or too small for a page:
    direct reads and writes. slot: the state's slot read and written (an MTP decoder's committed
    one, mtp.MTPDecoder.slot)."""

    def __init__(self, eng, seq: int = 0, area=False):
        self.eng, self.seq, self.slot = eng, seq, 0
        self.layout = Layout(eng.image, eng.cfg.S, seq)
        self.area = staging_area(eng) if area is False else area
        if self.area is not None and self.area[1] < self.layout.page_bytes() // eng.cfg.S:
            self.area = None
        self.stats = {"read": 0, "written": 0, "copies": 0}

    # ---- pages
    def read_pages(self, pages) -> list[np.ndarray]:
        """[(i, m)] -> each page's bytes."""
        L, out = self.layout, []
        for batch in _batches([L.regions(i, m) for i, m in pages], self._room()):
            out += self._read(batch)
        self.stats["read"] += sum(len(b) for b in out)
        return out

    def write_pages(self, pages) -> None:
        """[(i, m, bytes)] back into the caches."""
        L = self.layout
        items = [(L.regions(i, m), d) for i, m, d in pages]
        for batch in _batches(items, self._room(), key=lambda x: x[0]):
            self._write([r for r, _ in batch], [d for _, d in batch])
        self.stats["written"] += sum(len(d) for _, _, d in pages)

    def copy_from(self, other: "ContextIO", pages) -> None:
        """[(i, m)]: pages of another sequence's caches (another KV slot of the same image)
        into this one's, on the device (LD / ST programs: no bytes cross to the host)."""
        pairs = []
        for i, m in pages:
            for (src, n), (dst, n2) in zip(other.layout.regions(i, m), self.layout.regions(i, m)):
                assert n == n2
                pairs.append((src, dst, n))
        self._copy(pairs)

    # ---- state
    def read_state(self) -> np.ndarray:
        if not self.layout.stateful:
            return np.zeros(0, np.uint8)
        be, S = self.eng.backend, self.eng.cfg.S
        b = np.concatenate([be.read(s, a, n) for s in range(S)
                            for a, n in self.layout.state_in(self.slot)])
        self.stats["read"] += len(b)
        return b

    def write_state(self, data: np.ndarray) -> None:
        if not self.layout.stateful:
            return
        if len(data) != self.layout.state_bytes:
            raise ValueError(f"state of {len(data)} bytes, the image's is "
                             f"{self.layout.state_bytes}")
        be, S, o = self.eng.backend, self.eng.cfg.S, 0
        for s in range(S):
            for a, n in self.layout.state_in(self.slot):
                be.write(s, a, data[o:o + n])
                o += n
        self.stats["written"] += o

    # ---- the moves
    def _room(self) -> int | None:
        return None if self.area is None else self.area[1]

    def _read(self, regs_pages) -> list[np.ndarray]:
        """Regions of a batch of pages (a list per page) -> per page its bytes, each slice's
        after the other's."""
        be, S = self.eng.backend, self.eng.cfg.S
        if self.area is None:
            return [np.concatenate([be.read(s, a, n) for s in range(S) for a, n in regs])
                    for regs in regs_pages]
        base = self.area[0]
        pairs, o = [], 0
        for regs in regs_pages:
            for a, n in regs:
                pairs.append((a, base + o, n))
                o += n
        self._copy(pairs)
        flat = [be.read(s, base, o) for s in range(S)]
        out, o = [], 0
        for regs in regs_pages:
            n = sum(x for _, x in regs)
            out.append(np.concatenate([f[o:o + n] for f in flat]))
            o += n
        return out

    def _write(self, regs_pages, datas) -> None:
        be, S = self.eng.backend, self.eng.cfg.S
        for regs, d in zip(regs_pages, datas):
            if len(d) != S * sum(n for _, n in regs):
                raise ValueError(f"a page of {len(d)} bytes for "
                                 f"{S * sum(n for _, n in regs)}")
        if self.area is None:
            for regs, d in zip(regs_pages, datas):
                o = 0
                for s in range(S):
                    for a, n in regs:
                        be.write(s, a, d[o:o + n])
                        o += n
            return
        base = self.area[0]
        per_slice = [[] for _ in range(S)]
        pairs, o = [], 0
        for regs, d in zip(regs_pages, datas):
            n1 = sum(n for _, n in regs)
            for s in range(S):
                per_slice[s].append(d[s * n1:(s + 1) * n1])
            for a, n in regs:
                pairs.append((base + o, a, n))
                o += n
        for s in range(S):
            be.write(s, base, np.concatenate(per_slice[s]))
        self._copy(pairs)

    def _copy(self, pairs) -> None:
        for p in SN.copy_programs(pairs, self.eng.cfg):
            self.eng.backend.run(p)
            self.stats["copies"] += 1


def _batches(items, room: int | None, key=lambda x: x):
    """Consecutive batches of pages whose regions fit `room` bytes (one slice's; None: one
    batch)."""
    if room is None:
        return [items] if items else []
    out, cur, used = [], [], 0
    for it in items:
        n = sum(x for _, x in key(it))
        if cur and used + n > room:
            out.append(cur)
            cur, used = [], 0
        cur.append(it)
        used += n
    if cur:
        out.append(cur)
    return out


def staging_area(eng) -> tuple[int, int] | None:
    """(base, bytes) of the device DRAM free for staging: past the program area on the card
    (BoardBackend's prog_at), past the image elsewhere; None when the backend has no device
    copy worth it (the ISA simulator moves memory directly)."""
    be, cfg = eng.backend, eng.cfg
    at = getattr(be, "prog_at", None)
    if at is None:
        return None
    from opentpu import waits as W
    base = -(-(at + W.area(cfg.IMEM_WORDS, getattr(cfg, "SEQ_WC", False))) // 4096) * 4096
    from opentpu.llm.regions import dram_low     # (above DAW 32: region 0, past the program
    n = min(STAGING_MAX, dram_low(cfg) - base)    # area of an image laid out over regions)
    return (base, n) if n > 0 else None
