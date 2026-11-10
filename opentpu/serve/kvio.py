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
(m = n - i P positions; its V^T tile whole; with the caches in groups, GALL's KVDesc wg, m up to
whole groups of wg tokens: a group's bytes hold its tokens together) and the state the caches
do not hold
(Image.state_regions: Qwen3.5's DeltaNet states and convolution windows, LFM2's convolution
rings, Gemma 4's sliding rings when shorter than the cache), at the context's position. An MTP
image's (Spec.mtp, opentpu/llm/mtp.py) pages hold its MTP layer's KV cache too, and its state is
the one in the decoder's committed slot (ContextIO.slot: the DeltaNet states' two slots).

Restoring pages [0, ceil(k / P)) and the position k gives the context of the k tokens they were
saved for, bit for bit: what lies past k in the last page is overwritten by the next appends
and masked before (as Engine.rollback's). A model with state restores only at a position whose
state was saved.

On a machine with the channel hash (Config.CHASH) K's and V^T's group chunks hold their lanes
permuted by their DRAM address (quant.hash_rows), so one context's K and V^T are other bytes in
each sequence's caches of an Image(batch=N). Pages are read and written here in the first
sequence's lanes (Layout.relane, nothing to do for that sequence): any sequence restores what
any other saved, and contexts share pages across sequences.

Two ways to move the bytes: direct (the backend's read / write per region: the ISA simulator,
or a card without room) and staged (on the card: LD / ST programs gather the regions into a
contiguous staging area of the DRAM past the image and the program area, one DMA moves it, and
the other way scatter them: a page of Qwen3-0.6B is 896 regions, ~30 us of DMA set-up each).
The state is staged too, in batches of the area's room: the 35B's is 480 regions (34.4 MB a
slice: the DeltaNet layers' windows and states), 3,840 XDMA calls of its 8 channels direct
against a copy and one pipelined transfer. A save at a snapshot's position reads the state from
the snapshot (snapshot_at: the same bytes, one range), with no copy.

Where a move's time goes (ContextIO.t, seconds since it was made; a phase's share is the
difference across it, the scheduler's restore.* and save.* phases): "io" the public calls' wall
time, of which "prog" building the copy programs, "link" the backend's reads and writes (the
card's XDMA: PCIe, a bounce buffer for an unaligned write), "load" starting a copy (assembling
it, its wait entries, IMEM, RUN: BoardBackend.start) and "run" the copy on the device until it
halts (BoardBackend.wait; a backend without start / wait: the whole run), the rest the host's
own (region lists, slicing, concatenation). ContextIO.moved counts the link's bytes and calls
and the copies' runs and bytes (every slice's). gc_seconds(): the process's time in Python's
cyclic GC (any thread's: it holds the interpreter), for a phase's share of it.
"""
from __future__ import annotations

import gc
import time
from dataclasses import dataclass

import numpy as np

from .. import quant as Q
from ..compiler import Affine, gall_wg, k_scale_bytes, vt_tile
from ..llm import snapshot as SN

STAGING_MAX = 256 << 20     # the staging area's bytes at most (a batch of pages per copy)
PARTS = ("prog", "link", "load", "run")     # ContextIO.t's parts of "io" (the rest: the host's)

_GC = {"s": 0.0, "t0": None}


def _gc_timer(phase: str, info: dict) -> None:
    if phase == "start":
        _GC["t0"] = time.perf_counter()
    elif _GC["t0"] is not None:
        _GC["s"] += time.perf_counter() - _GC["t0"]
        _GC["t0"] = None


gc.callbacks.append(_gc_timer)


def gc_seconds() -> float:
    """Seconds in Python's cyclic GC since kvio was imported (gc.callbacks)."""
    return _GC["s"]


def _timed(f):
    """A public move: its wall time into self.t["io"]."""
    def call(self, *a, **kw):
        t0 = time.perf_counter()
        try:
            return f(self, *a, **kw)
        finally:
            self.t["io"] += time.perf_counter() - t0
    call.__name__, call.__doc__ = f.__name__, f.__doc__
    return call


@dataclass(frozen=True)
class Array:
    """One KV cache array of one head: `w` bytes per position (tile: V^T, the bytes of one
    position in each of its `w` rows, a tile [w, P] per page); grouped: its bytes in the group
    chunks of KVDesc's wg (K, V^T: their lanes under the channel hash)."""
    base: int
    w: int
    tile: bool = False
    grouped: bool = False


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
        # (every sequence's: a Qwen3.5 image's slot view holds its own, kv_slots all of them)
        kvs = getattr(lw, "kv_slots", None) or getattr(lw, "kvs", None)
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
                out += [Array(k, kv.d, grouped=True),
                        Array(_static(e["ks"]), k_scale_bytes(kv.d, kv.D)),
                        Array(_static(e["vt"]), kv.d, True, True), Array(_static(e["vs"]), 4)]
    return out


class Layout:
    """Where an Engine's context lives in its DRAM: page regions and state regions (one slice's;
    every slice has the same addresses)."""

    def __init__(self, image, S: int, seq: int = 0):
        self.cap, self.S = image.cap, S
        self.P = vt_tile(image.cap)
        self.wg = gall_wg(image.cfg)            # (KVDesc's groups of tokens)
        self.arrays = kv_arrays(image, seq)
        # (a later sequence's pages under the channel hash: relane, against the first's arrays)
        self.first = kv_arrays(image, 0) if seq and self.wg > 1 and image.cfg.CHASH else None
        self.D = image.cfg.D
        self.image = image
        self.state = SN.merged(image.state_regions(0)) if SN.stateful(image) else []
        self.state_bytes = S * sum(n for _, n in self.state)
        self.stateful = bool(self.state)
        self.sf = SN.state_format(image) if self.stateful else ""     # (store.Entry.sf)
        self.pages = -(-self.cap // self.P)
        self._slots = {0: self.state}

    def state_in(self, slot: int) -> list[tuple[int, int]]:
        """The state regions of the DeltaNet states' `slot` (an MTP image's: two; the same
        bytes as slot 0's)."""
        if slot not in self._slots:
            self._slots[slot] = SN.merged(self.image.state_regions(slot))
        return self._slots[slot]

    def regions(self, i: int, m: int | None = None) -> list[tuple[int, int]]:
        """Page i's regions holding its first m positions (default all P): rows m at a time
        (in groups: whole groups), V^T tiles whole."""
        P = self.P
        m = P if m is None else m
        if not 0 < m <= P or not 0 <= i < self.pages:
            raise ValueError(f"page {i} with {m} positions")
        mg = min(P, -(-m // self.wg) * self.wg)
        out = []
        for a in self.arrays:
            if a.tile:
                out.append((a.base + i * P * a.w, P * a.w))
            else:
                out.append((a.base + i * P * a.w, mg * a.w))
        return out

    def relane(self, i: int, m: int | None, data: np.ndarray) -> np.ndarray:
        """Page i's bytes (m positions, each slice's regions after the other's) from this
        sequence's lanes to the first sequence's, or back: its own inverse. Under the channel
        hash the lane l of the chunk at address A is stored at l ^ chunk_fold(A / P), so a chunk
        moves by the two addresses' folds together (quant.hash_rows, both ways in one)."""
        if self.first is None:
            return data
        wg, D = self.wg, self.D
        P, n1, o, out = wg * D, len(data) // self.S, 0, data.copy()
        for a, a0, (at, n) in zip(self.arrays, self.first, self.regions(i, m)):
            if a.grouped:
                k = np.arange(n // P)
                h = Q.chunk_fold(at // P + k, wg) ^ Q.chunk_fold((at - a.base + a0.base) // P + k,
                                                                 wg)
                lanes = k[:, None], np.arange(wg)[None, :] ^ h[:, None]
                for b in range(o, o + self.S * n1, n1):
                    out[b:b + n] = data[b:b + n].reshape(-1, wg, D)[lanes].reshape(-1)
            o += n
        return out

    def page_bytes(self, m: int | None = None) -> int:
        """A page's bytes, every slice's (m positions)."""
        return self.S * sum(n for _, n in self.regions(0, m))

    def split(self, n: int) -> list[tuple[int, int]]:
        """The pages (index, positions) holding a context of n tokens."""
        return [(i, min(self.P, n - i * self.P)) for i in range(-(-n // self.P))]


class Aborted(Exception):
    """A read given up (ContextIO.abort said so between two of its device reads)."""


class ContextIO:
    """Reads and writes an Engine's pages and state as host bytes (each slice's after the
    other's). area: the staging area (base, bytes) in the device's DRAM; default
    staging_area(eng) (a card's: the DRAM past the program area), None or too small for a page:
    direct reads and writes. slot: the state's slot read and written (an MTP decoder's committed
    one, mtp.MTPDecoder.slot). abort: None, or a callable asked before each device read of
    read_pages and read_state; True raises Aborted (the device untouched: reads only)."""

    def __init__(self, eng, seq: int = 0, area=False):
        # (a Qwen3.5 image of several sequences: sequence seq's state in its slot)
        self.eng, self.seq = eng, seq
        self.slot = seq if getattr(eng.image, "slot_runs", False) else 0
        self.layout = Layout(eng.image, eng.cfg.S, seq)
        self.area = staging_area(eng) if area is False else area
        if self.area is not None and self.area[1] < self.layout.page_bytes() // eng.cfg.S:
            self.area = None
        self.stats = {"read": 0, "written": 0, "copies": 0}
        self.t = dict.fromkeys(("io",) + PARTS, 0.0)       # seconds (the module's docstring)
        self.moved = {"link_bytes": 0, "link_calls": 0, "copies": 0, "copied_bytes": 0}
        self.abort = None
        self._progs: dict = {}          # the state's copy programs (its batches: the same each time)

    def _check(self) -> None:
        if self.abort is not None and self.abort():
            raise Aborted("a read given up")

    # ---- pages
    @_timed
    def read_pages(self, pages) -> list[np.ndarray]:
        """[(i, m)] -> each page's bytes."""
        L, out = self.layout, []
        for batch in _batches([L.regions(i, m) for i, m in pages], self._room()):
            self._check()
            out += self._read(batch)
        self.stats["read"] += sum(len(b) for b in out)
        return [L.relane(i, m, d) for (i, m), d in zip(pages, out)]

    @_timed
    def write_pages(self, pages) -> None:
        """[(i, m, bytes)] back into the caches."""
        L = self.layout
        items = [(L.regions(i, m), L.relane(i, m, d)) for i, m, d in pages]
        for batch in _batches(items, self._room(), key=lambda x: x[0]):
            self._write([r for r, _ in batch], [d for _, d in batch])
        self.stats["written"] += sum(len(d) for _, _, d in pages)

    def copy_from(self, other: "ContextIO", pages) -> None:
        """[(i, m)]: pages of another sequence's caches (another KV slot of the same image)
        into this one's, on the device (LD / ST programs: no bytes cross to the host); under the
        channel hash through the host (a copy cannot move the chunks' lanes: Layout.relane)."""
        if self.layout.first is not None or other.layout.first is not None:
            got = other.read_pages(pages)
            return self.write_pages([(i, m, d) for (i, m), d in zip(pages, got)])
        self._copy_pages(other, pages)

    @_timed
    def _copy_pages(self, other: "ContextIO", pages) -> None:
        pairs = []
        for i, m in pages:
            for (src, n), (dst, n2) in zip(other.layout.regions(i, m), self.layout.regions(i, m)):
                assert n == n2
                pairs.append((src, dst, n))
        self._copy(pairs)

    # ---- state
    @_timed
    def read_state(self, at: int | None = None) -> np.ndarray:
        """The state regions' bytes, each slice's after the other's. at: a device address
        holding them in that order as one range (a snapshot of them: snapshot_at), read from
        there (a transfer a slice); else gathered into the staging area (a copy and a transfer a
        slice per batch of its room), or region by region without one."""
        if not self.layout.stateful:
            return np.zeros(0, np.uint8)
        S, regs = self.eng.cfg.S, self.layout.state_in(self.slot)
        if at is not None:
            self._check()
            parts = [self._rd(s, at, self.layout.state_bytes // S) for s in range(S)]
        elif self.area is None:
            parts = []
            for s in range(S):
                for a, n in regs:
                    self._check()
                    parts.append(self._rd(s, a, n))
        else:
            per = [[] for _ in range(S)]
            for i, batch in enumerate(_fill(regs, self.area[1])):
                self._check()
                for s, f in enumerate(self._gather(batch, ("r", self.slot, i))):
                    per[s].append(f)
            parts = [f for p in per for f in p]
        b = np.concatenate(parts)
        self.stats["read"] += len(b)
        return b

    @_timed
    def write_state(self, data: np.ndarray) -> None:
        """The state's bytes (read_state's) back into its regions: staged (a transfer a slice
        and a copy per batch of the area's room), or region by region without an area."""
        if not self.layout.stateful:
            return
        if len(data) != self.layout.state_bytes:
            raise ValueError(f"state of {len(data)} bytes, the image's is "
                             f"{self.layout.state_bytes}")
        S, regs = self.eng.cfg.S, self.layout.state_in(self.slot)
        if self.area is None:
            o = 0
            for s in range(S):
                for a, n in regs:
                    self._wr(s, a, data[o:o + n])
                    o += n
        else:
            n1, o = len(data) // S, 0
            for i, batch in enumerate(_fill(regs, self.area[1])):
                k = sum(n for _, n in batch)
                self._scatter(batch, [data[s * n1 + o:s * n1 + o + k] for s in range(S)],
                              ("w", self.slot, i))
                o += k
        self.stats["written"] += len(data)

    # ---- the moves
    def _room(self) -> int | None:
        return None if self.area is None else self.area[1]

    def _rd(self, s: int, a: int, n: int) -> np.ndarray:
        t0 = time.perf_counter()
        b = self.eng.backend.read(s, a, n)
        self.t["link"] += time.perf_counter() - t0
        self.moved["link_bytes"] += n
        self.moved["link_calls"] += 1
        return b

    def _wr(self, s: int, a: int, d: np.ndarray) -> None:
        t0 = time.perf_counter()
        self.eng.backend.write(s, a, d)
        self.t["link"] += time.perf_counter() - t0
        self.moved["link_bytes"] += len(d)
        self.moved["link_calls"] += 1

    def _gather(self, regs, key=None) -> list[np.ndarray]:
        """Regions copied into the staging area one after the other, then read: each slice's
        bytes (key: the copy's programs kept under it)."""
        base, pairs, o = self.area[0], [], 0
        for a, n in regs:
            pairs.append((a, base + o, n))
            o += n
        self._copy(pairs, key)
        return [self._rd(s, base, o) for s in range(self.eng.cfg.S)]

    def _scatter(self, regs, parts, key=None) -> None:
        """Each slice's bytes (parts[s]) into the staging area, then copied out to the
        regions one after the other."""
        base = self.area[0]
        for s, d in enumerate(parts):
            self._wr(s, base, d)
        pairs, o = [], 0
        for a, n in regs:
            pairs.append((base + o, a, n))
            o += n
        self._copy(pairs, key)

    def _read(self, regs_pages) -> list[np.ndarray]:
        """Regions of a batch of pages (a list per page) -> per page its bytes, each slice's
        after the other's."""
        S = self.eng.cfg.S
        if self.area is None:
            return [np.concatenate([self._rd(s, a, n) for s in range(S) for a, n in regs])
                    for regs in regs_pages]
        flat = self._gather([r for regs in regs_pages for r in regs])
        out, o = [], 0
        for regs in regs_pages:
            n = sum(x for _, x in regs)
            out.append(np.concatenate([f[o:o + n] for f in flat]))
            o += n
        return out

    def _write(self, regs_pages, datas) -> None:
        S = self.eng.cfg.S
        for regs, d in zip(regs_pages, datas):
            if len(d) != S * sum(n for _, n in regs):
                raise ValueError(f"a page of {len(d)} bytes for "
                                 f"{S * sum(n for _, n in regs)}")
        if self.area is None:
            for regs, d in zip(regs_pages, datas):
                o = 0
                for s in range(S):
                    for a, n in regs:
                        self._wr(s, a, d[o:o + n])
                        o += n
            return
        per_slice = [[] for _ in range(S)]
        for regs, d in zip(regs_pages, datas):
            n1 = sum(n for _, n in regs)
            for s in range(S):
                per_slice[s].append(d[s * n1:(s + 1) * n1])
        self._scatter([r for regs in regs_pages for r in regs],
                      [np.concatenate(p) for p in per_slice])

    def _copy(self, pairs, key=None) -> None:
        """The copies (src, dst, n) on the device: its programs (key: kept under it) run, each
        started then waited for (t's "load" and "run") where the backend has the two halves."""
        t0 = time.perf_counter()
        progs = self._progs.get(key) if key is not None else None
        if progs is None:
            progs = SN.copy_programs(pairs, self.eng.cfg)
            if key is not None:
                self._progs[key] = progs
        be = self.eng.backend
        start, wait = getattr(be, "start", None), getattr(be, "wait", None)
        t1 = time.perf_counter()
        self.t["prog"] += t1 - t0
        for p in progs:
            if start is not None and wait is not None:      # (BoardBackend.run's halves)
                start(p)
                t2 = time.perf_counter()
                wait()
            else:
                t2 = t1
                be.run(p)
            t3 = time.perf_counter()
            self.t["load"] += t2 - t1
            self.t["run"] += t3 - t2
            t1 = t3
            self.stats["copies"] += 1
            self.moved["copies"] += 1
        self.moved["copied_bytes"] += sum(n for _, _, n in pairs) * self.eng.cfg.S


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


def _fill(regs, room: int) -> list[list[tuple[int, int]]]:
    """The regions in consecutive batches of at most `room` bytes (one slice's), a region
    larger than the room in pieces of it (whole words)."""
    room -= room % 4
    out, cur, used = [], [], 0
    for a, n in regs:
        while n:
            k = min(n, room - used)
            cur.append((a, k))
            a, n, used = a + k, n - k, used + k
            if used == room:
                out.append(cur)
                cur, used = [], 0
    if cur:
        out.append(cur)
    return out


def snapshot_at(eng, pos: int) -> int | None:
    """The device address of the Engine's snapshot at position `pos` (Engine(snapshots=):
    its state regions one after the other, read_state's bytes a slice), or None: no snapshot
    there, or the snapshots on the host."""
    sn = getattr(eng, "_snaps", None)
    if sn is None or sn.base is None:
        return None
    j = next((j for j, p in sn.at.items() if p == pos), None)
    return None if j is None else sn.base + j * sn.nbytes


def staging_area(eng) -> tuple[int, int] | None:
    """(base, bytes) of the device DRAM free for staging: past the program area on the card
    (BoardBackend's prog_at), past the image elsewhere; None when the backend has no device
    copy worth it (the ISA simulator moves memory directly). An image with its weights out of
    the low area (Image.high: the 35B on the BCU's bank-half window) keeps the run-time areas
    below its weights (qwen3.area_limit, the backend's `high`): the area ends there, not at the
    DRAM's end, where its weights are."""
    be, cfg = eng.backend, eng.cfg
    at = getattr(be, "prog_at", None)
    if at is None:
        return None
    from opentpu import waits as W
    from ..llm.qwen3 import area_limit
    base = -(-(at + W.area(cfg.IMEM_WORDS, getattr(cfg, "SEQ_WC", False))) // 4096) * 4096
    top = min(area_limit(eng.image, cfg), getattr(be, "high", None) or cfg.DRAM_BYTES)
    n = min(STAGING_MAX, top - base)
    return (base, n) if n > 0 else None
