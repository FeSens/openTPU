"""Path (a)'s host side: the expert server (docs/offload.md section 5.2).

The card computes everything: it routes, posts the ids of the experts a MoE layer needs to a
mailbox in its DRAM, computes the ones its directory says it has and waits (WAITW) on the
directory entries of the others. This module only moves bytes: it serves each request from the
expert pool (host RAM, every expert already in the card's slot format) into per-layer LRU
slots in the card's DRAM, and keeps the directory. The same code serves the ISA simulator (its
WAITW host hook) and the card (BoardBackend.host: polled while a run is in flight).

DRAM words (`Layout`), all 4-byte words at 64-byte aligned bases:

    mbox                   seq: the card's last request, as a float (0.0 before the first)
    mbox + 64              the k global expert ids of request seq (floats)
    served                 the last request the host has finished, as a float
    dir + 8 * g            expert g's entry: {slot address (u32), present (f32: 1.0 or 0.0)}

A global expert id is `j * E + e` for the j-th MoE layer's expert e (the card's ARGMAX gives it
with base j * E). Per request the host, for each id missing from its copy of the directory:
picks the layer's least recently used expert that the request does not name, writes {0, 0.0}
to its entry, writes the new expert into its slot and then {slot, 1.0} to the new entry; after
the last id, `served = seq`. The card fences each layer before it posts, `WAITW served >= seq`
(its last request), so one request row is enough, and no eviction for a layer is in flight
while it uses that layer's slots (docs/offload.md 5.2).

A request whose ids are at G = layers x E and above is a hint (docs/offload.md 12: the layer's
router on its input, before its mixer). With the "lfu" policy the host gives each hinted expert
not in a slot one now (a free one, or the victim, its entry cleared), writes served, and moves
the experts on the link's idle time: one part per poll that finds no request, the entry when
the last part has landed. A request naming one still on its way sends the rest at once. "lru"
ignores hints.

On the card the server's memory is `BoardDram` (`dram_of`): the experts' DMA at the link's
rate, in a worker thread, the host's own words without a read of the card first.

The pool can be a file (`PoolFile`). Its split format (`split_order`) keeps each 4 KiB of an
expert as its two channel runs under CHASH, so that BoardDram reads an expert from the file
straight into the runs it DMAs (one os.preadv, the GIL released) instead of gathering them.
"""
from __future__ import annotations

import math
import os
import queue
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

LINE = 64              # the mailbox's words, its row and each flag on their own 64-byte lines
RUN = 4096             # the split pool format's block: 32 chunks, its two channel runs
SPLIT = "split4k"      # the split format's name in a pool file's <file>.format
IOV_MAX = 1024         # buffers per os.preadv (Linux's UIO_MAXIOV)


def _parity(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint64)
    for s in (32, 16, 8, 4, 2, 1):
        x ^= x >> np.uint64(s)
    return (x & np.uint64(1)).astype(np.intp)


_ORDERS: dict = {}


def split_order(n: int) -> np.ndarray:
    """The split pool format of an n-byte expert (n a multiple of 128): the file's 64-byte beat
    k is the expert's beat order[k]. Each RUN-byte block (32 chunks of two beats; the last
    block may be shorter) holds first the beats channel 0 takes when the block lands where the
    card's chunk index has even parity above its low 5 bits (CHASH: chunk i's beat parity(i)),
    then the others: the block's two channel runs. Where that parity is odd the runs trade
    channels (BoardDram.write_slot)."""
    if n not in _ORDERS:
        m = np.arange(n // 128)
        i, j = m % 32, m // 32
        c = np.minimum(32, n // 128 - 32 * j)            # the chunks in m's block
        p = _parity(i)
        order = np.empty(n // 64, np.intp)
        order[64 * j + i] = 2 * m + p
        order[64 * j + c + i] = 2 * m + 1 - p
        _ORDERS[n] = order
    return _ORDERS[n]


def to_split(data) -> np.ndarray:
    """An expert's bytes (the card's slot format) in the split pool format."""
    b = np.ascontiguousarray(data).view(np.uint8).reshape(-1)
    return b.view("V64")[split_order(len(b))].view(np.uint8)


def preadv(fd: int, bufs: list, off: int) -> None:
    """Fill the buffers (writable memoryviews of bytes), in order, from file offset `off`:
    os.preadv in groups of IOV_MAX, a short read resumed."""
    bufs, i = list(bufs), 0
    while i < len(bufs):
        grp = bufs[i:i + IOV_MAX]
        k = os.preadv(fd, grp, off)
        if k <= 0:
            raise OSError(f"preadv at {off}: end of file")
        off += k
        if k == sum(map(len, grp)):             # the whole group: the usual case
            i += len(grp)
            continue
        while k >= len(bufs[i]):
            k -= len(bufs[i])
            i += 1
        if k:
            bufs[i] = bufs[i][k:]


class SplitRecord:
    """An expert in the split format, read from its file when it is written: readv(bufs, at)
    reads its bytes from byte `at` on, in file order, into the buffers. BoardDram on a CHASH
    card reads it straight into its two channel runs; as bytes (bytes(), np.asarray) it is the
    slot's own bytes."""

    def __init__(self, n: int, readv):
        self.n, self.readv = n, readv

    def __len__(self) -> int:
        return self.n

    def part(self, a: int, b: int) -> "SplitRecord":
        """Its bytes a..b (whole RUN blocks, or to its end), in the split format: the blocks
        are its own."""
        if a % RUN or (b % RUN and b != self.n) or not 0 <= a < b <= self.n:
            raise ValueError(f"a split record's part {a}..{b} of {self.n} bytes")
        return SplitRecord(b - a, lambda bufs, at=0: self.readv(bufs, a + at))

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        f = np.empty(self.n, np.uint8)
        self.readv([memoryview(f)])
        out = np.empty(self.n, np.uint8)
        out.view("V64")[split_order(self.n)] = f.view("V64")
        return out if dtype is None else out.astype(dtype)

    def __bytes__(self) -> bytes:
        return self.__array__().tobytes()


class PoolFile:
    """The expert pool in a file (every expert's `slot` bytes at g * slot; the page cache is
    the host's RAM tier, the disk below it), read with os.preadv (the GIL released: a memmap
    page not in the page cache would block the whole process on the disk). `split`: the split
    format, get(g) a SplitRecord; else the slot format, get(g) the bytes in one of two
    page-aligned buffers in turn (used before the next get: the server writes an expert
    before it asks for the next). warm(ids) reads those experts once in a thread, into the
    page cache: the card's host disk serves about 115 MB/s to scattered reads, 13 ms an
    expert of 1.67 MB. It moves bytes only."""

    def __init__(self, path, slot: int, split: bool):
        import mmap
        self.fd, self.slot, self.split = os.open(path, os.O_RDONLY), slot, split
        self.bufs = [mmap.mmap(-1, slot) for _ in range(2)]
        self.k, self.warm_t = 0, None
        self.arr = self.packed = self.ids = self.resident_at_open = None    # (moe.open_pool)

    def resident(self, ids) -> int | None:
        """Bytes of these experts in the page cache (mincore over the file's pages), or None
        where the host cannot tell."""
        import ctypes
        import mmap
        size = os.fstat(self.fd).st_size
        if not size or not len(ids):
            return 0
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            mm = mmap.mmap(self.fd, size, prot=mmap.PROT_READ)
        except (OSError, AttributeError, ValueError):
            return None
        pg = mmap.PAGESIZE
        vec = (ctypes.c_ubyte * -(-size // pg))()
        try:
            v = np.frombuffer(mm, np.uint8)
            rc = libc.mincore(ctypes.c_void_p(v.ctypes.data), ctypes.c_size_t(size), vec)
            del v
        finally:
            mm.close()
        if rc != 0:
            return None
        cum = np.concatenate([[0], np.cumsum(np.frombuffer(vec, np.uint8) & 1)])
        g = np.asarray(ids, np.int64)
        lo, hi = g * self.slot // pg, -(-(g + 1) * self.slot // pg)
        return int(np.minimum((cum[hi] - cum[lo]) * pg, self.slot).sum())

    def get(self, g: int):
        if self.split:
            return SplitRecord(self.slot,
                               lambda bufs, at=0: preadv(self.fd, bufs, g * self.slot + at))
        self.k ^= 1
        preadv(self.fd, [memoryview(self.bufs[self.k])], g * self.slot)
        return np.frombuffer(self.bufs[self.k], np.uint8)

    def warm(self, ids) -> threading.Thread:
        import mmap
        scratch = memoryview(mmap.mmap(-1, self.slot))

        def run():
            for g in ids:
                preadv(self.fd, [scratch], int(g) * self.slot)
                t.bytes += self.slot
        t = threading.Thread(target=run, daemon=True, name="otpu-pool-warm")
        t.bytes = 0
        t.start()
        self.warm_t = t
        return t


@dataclass(frozen=True)
class Layout:
    """Where path (a)'s words and slots live in the card's DRAM (byte addresses)."""
    E: int                 # experts per MoE layer
    k: int                 # experts per request (at most LINE / 4)
    slots: tuple           # per MoE layer: (first slot's address, slots)
    slot_bytes: int        # one expert in the card's format (a multiple of 64)
    mbox: int
    served: int
    dir: int

    @staticmethod
    def build(base: int, E: int, k: int, slots_per_layer, slot_bytes: int) -> "Layout":
        """The words from `base` up (64-byte aligned), then the slots, layer after layer, from
        the next 4 KiB page (slot_bytes keeps them D-byte aligned: the MXU streams whole
        chunks)."""
        if base % LINE or slot_bytes % LINE or not 0 < k <= LINE // 4:
            raise ValueError("unaligned base or slot size, or k too large")
        mbox = base
        served = mbox + 2 * LINE
        d = served + LINE
        a = -(-(d + 8 * E * len(slots_per_layer)) // 4096) * 4096     # slots page-aligned
        slots = []
        for n in slots_per_layer:
            slots.append((a, int(n)))
            a += int(n) * slot_bytes
        return Layout(E, k, tuple(slots), slot_bytes, mbox, served, d)

    @property
    def layers(self) -> int:
        return len(self.slots)

    @property
    def end(self) -> int:
        a, n = self.slots[-1]
        return a + n * self.slot_bytes

    def entry(self, g: int) -> int:
        return self.dir + 8 * g

    @property
    def row(self) -> int:
        return self.mbox + LINE


def _f32(x: float) -> bytes:
    return np.float32(x).tobytes()


class ExpertServer:
    """Serves the card's expert requests and hints (module docstring).

    mem: write(addr, bytes-like) and read(addr, n) -> bytes-like on the card's DRAM (the ISA
    simulator's slice DRAM through SimDram, or the board through BoardBackend's read / write).
    pool(g): expert g in the card's slot format, `layout.slot_bytes` long (host RAM or a file;
    the host never computes with it). part: the bytes of a hinted expert one idle poll sends
    (a request waits for at most one part on the link)."""

    def __init__(self, mem, layout: Layout, pool, policy: str = "lru", half: float = 32.0,
                 part: int = 512 << 10):
        self.mem, self.L, self.pool = mem, layout, pool
        if policy not in ("lru", "lfu"):
            raise ValueError(f"replacement policy {policy!r}")
        if part <= 0 or part % RUN:
            raise ValueError(f"a hint's part of {part} bytes (RUN blocks)")
        # the victim: the layer's least recently used expert the request does not name, or with
        # "lfu" the one of least use, each use decaying by half every `half` requests of its
        # layer (docs/offload.md 10.3: 9-15% fewer misses than LRU on the traces)
        self.policy, self.half, self.part = policy, half, part
        self.lru = [OrderedDict() for _ in range(layout.layers)]    # g -> slot address
        self.t = [0] * layout.layers                                # requests per layer
        self.use: list = [{} for _ in range(layout.layers)]          # g -> log2 use + t / half
        self.free = [[a + i * layout.slot_bytes for i in range(n)] for a, n in layout.slots]
        self.pending: OrderedDict = OrderedDict()   # hinted, in a slot, not landed: g -> bytes sent
        self.seq = 0                        # the last request served
        self.hits = self.misses = self.bytes = 0
        # hints served; hinted experts landed on idle time, sent by the request that named
        # them, replaced before they landed
        self.hints = self.prefetched = self.promoted = self.dropped = 0
        self.history: list | None = None    # a list: each request's ids are appended

    def load(self, warm=()) -> None:
        """At image load: an empty directory and mailbox, then the experts `warm` names (global
        ids, most wanted first, e.g. a profile's order) in their layers' slots while slots
        last."""
        L = self.L
        self.mem.write(L.mbox, np.zeros(2 * LINE // 4, np.float32))
        self.mem.write(L.served, _f32(0.0))
        self.mem.write(L.dir, np.zeros(2 * L.E * L.layers, np.uint32))
        for lru, fr in zip(self.lru, self.free):
            fr.extend(lru.values())
            lru.clear()
        self.pending.clear()
        self.seq = 0
        self.t = [0] * L.layers
        self.use = [{} for _ in range(L.layers)]
        for g in warm:
            j = g // L.E
            if self.free[j] and g not in self.lru[j]:
                self._insert(j, g, self.free[j].pop(0))
                self.use[j][g] = 0.0            # one use, before the first request
        for lru in self.lru:                # the profile's first expert is the most recent
            for g in reversed(list(lru)):
                lru.move_to_end(g)
        self._flush()
        self.hits = self.misses = self.bytes = 0     # counted from here: the requests'
        self.hints = self.prefetched = self.promoted = self.dropped = 0

    def poll(self) -> int:
        """Serve the card's request if it posted one since the last served, else send a part of
        a hinted expert; returns 1 if it did either. The card's seq is read first, then the
        row."""
        seq = int(np.frombuffer(bytes(self.mem.read(self.L.mbox, 4)), np.float32)[0])
        if seq == self.seq:
            if not self.pending:
                return 0
            self.step()
            self._flush()
            return 1
        if seq != self.seq + 1:
            raise RuntimeError(f"the card posted request {seq} with {self.seq} served: its "
                               f"fence (WAITW served >= seq) is missing")
        ids = [int(g) for g in np.frombuffer(bytes(self.mem.read(self.L.row, 4 * self.L.k)),
                                             np.float32)]
        G = self.L.E * self.L.layers
        if ids[0] >= G:
            self.hint([g - G for g in ids])
        else:
            if self.history is not None:
                self.history.append(ids)
            self.serve(ids)
        self.seq = seq
        self.mem.write(self.L.served, _f32(seq))
        self._flush()                       # (no DMA of the server's in flight after poll)
        return 1

    def _flush(self) -> None:
        f = getattr(self.mem, "flush", None)
        if f is not None:
            f()

    def _layer(self, ids) -> int:
        E = self.L.E
        j = ids[0] // E
        if any(g // E != j or not 0 <= g < E * self.L.layers for g in ids):
            raise ValueError(f"request {ids} is not one layer's experts")
        return j

    def serve(self, ids) -> None:
        """One request: its k global ids, all of one MoE layer."""
        j = self._layer(ids)
        self.t[j] += 1
        lru, use, t = self.lru[j], self.use[j], self.t[j] / self.half
        for g in ids:                       # the decayed use count, kept as log2 + t / half
            use[g] = math.log2(2.0 ** (use[g] - t) + 1.0) + t if g in use else t
        for g in ids:
            if g in lru:
                lru.move_to_end(g)
                if g in self.pending:       # hinted, on its way: the rest now
                    self.misses += 1
                    self.promoted += 1
                    self._send(g, self.L.slot_bytes)
                else:
                    self.hits += 1
                continue
            self.misses += 1
            self._insert(j, g, self._slot(j, ids))

    def hint(self, ids) -> None:
        """A hint: the k global ids the layer's router picks on its input (docs/offload.md 12).
        With "lfu" each one not in a slot gets one now (a free slot, or the victim's: its entry
        cleared) and waits in `pending` for the link's idle time (step); no use counted, so
        a hinted expert no request names is the next victim. "lru" ignores hints (section
        5.4: an LRU victim of a wrong hint is a recent expert)."""
        j = self._layer(ids)
        self.hints += 1
        if self.policy != "lfu":
            return
        lru = self.lru[j]
        for g in ids:
            if g not in lru:
                lru[g] = self._slot(j, ids)
                self.pending[g] = 0

    def step(self) -> None:
        """The next part of the oldest hinted expert on its way (the link is idle: no request
        since the last served); its entry once its last part is sent."""
        g, a = next(iter(self.pending.items()))
        n = self.L.slot_bytes
        if self._send(g, min(a + self.part, n)) == n:
            self.prefetched += 1

    def _send(self, g: int, b: int) -> int:
        """Hinted expert g's bytes from where it stands to b; its entry when that is the end.
        Returns b."""
        slot, a = self.lru[g // self.L.E][g], self.pending[g]
        data = self.pool(g)
        n = len(data)
        if n != self.L.slot_bytes:
            raise ValueError(f"expert {g}: {n} bytes, slots hold {self.L.slot_bytes}")
        if a == 0 and b == n:
            part = data
        elif isinstance(data, SplitRecord):
            part = data.part(a, b)
        else:
            part = (np.ascontiguousarray(data).view(np.uint8).reshape(-1) if isinstance(
                data, np.ndarray) else np.frombuffer(bytes(data), np.uint8))[a:b]
        getattr(self.mem, "write_slot", self.mem.write)(slot + a, part)
        self.bytes += b - a
        if b < n:
            self.pending[g] = b
            return b
        del self.pending[g]
        self.mem.write(self.L.entry(g), np.array([slot], np.uint32).tobytes() + _f32(1.0))
        return b

    def _slot(self, j: int, ids) -> int:
        """A slot of layer j for one of the request's experts: a free one, or the victim's (its
        entry cleared; a hinted expert still on its way is dropped)."""
        if self.free[j]:
            return self.free[j].pop(0)
        lru, use = self.lru[j], self.use[j]
        if self.policy == "lfu":
            victim = min((v for v in lru if v not in ids),
                         key=lambda v: use.get(v, -math.inf), default=None)
        else:
            victim = next((v for v in lru if v not in ids), None)
        if victim is None:
            raise RuntimeError(f"layer {j}: {len(lru)} slots for a request of {len(ids)}")
        if self.pending.pop(victim, None) is not None:
            self.dropped += 1               # (its entry reads 0 already)
        else:
            self.mem.write(self.L.entry(victim), np.zeros(2, np.uint32))
        return lru.pop(victim)

    def _insert(self, j: int, g: int, slot: int) -> None:
        data = self.pool(g)
        if len(data) != self.L.slot_bytes:
            raise ValueError(f"expert {g}: {len(data)} bytes, slots hold {self.L.slot_bytes}")
        getattr(self.mem, "write_slot", self.mem.write)(slot, data)
        self.mem.write(self.L.entry(g), np.array([slot], np.uint32).tobytes() + _f32(1.0))
        self.lru[j][g] = slot
        self.bytes += len(data)


@dataclass(frozen=True)
class RowLayout:
    """A mailbox of its own for rows the host keeps (Gemma 4 E4B's PLE records,
    docs/gemma4_e4b.md), in Layout's format: seq at mbox, the request's id at mbox + 64,
    served at mbox + 128, each on its own line; the row goes to `slot`."""
    mbox: int
    slot: int
    row_bytes: int

    WORDS = 4 * LINE        # the mailbox's bytes: served's 128-byte block too (a board write
                            # of one word reads and writes back its whole block)

    @property
    def row(self) -> int:
        return self.mbox + LINE

    @property
    def served(self) -> int:
        return self.mbox + 2 * LINE


class RowServer:
    """Serves the card's row requests: request seq names one id (the card posts it after a
    fence, as a MoE layer its experts), the host writes rows(id) into the slot, then
    served = seq. Data movement only: rows(id) is the row in the card's format, read from
    host RAM or a file. mem as ExpertServer's."""

    def __init__(self, mem, layout: RowLayout, rows):
        self.mem, self.L, self.rows = mem, layout, rows
        self.seq = 0                        # the last request served
        self.bytes = 0
        self.history: list | None = None    # a list: each request's id is appended

    def load(self) -> None:
        """At image load: an empty mailbox."""
        self.mem.write(self.L.mbox, np.zeros(RowLayout.WORDS // 4, np.float32))
        self.seq = 0

    def poll(self) -> int:
        """Serve the card's request if it posted one since the last served; returns 1 if so."""
        seq = int(np.frombuffer(bytes(self.mem.read(self.L.mbox, 4)), np.float32)[0])
        if seq == self.seq:
            return 0
        if seq != self.seq + 1:
            raise RuntimeError(f"the card posted row request {seq} with {self.seq} served: "
                               f"its fence (WAITW served >= seq) is missing")
        g = int(np.frombuffer(bytes(self.mem.read(self.L.row, 4)), np.float32)[0])
        if self.history is not None:
            self.history.append(g)
        data = self.rows(g)
        if len(data) != self.L.row_bytes:
            raise ValueError(f"row {g}: {len(data)} bytes, the slot holds {self.L.row_bytes}")
        self.mem.write(self.L.slot, data)
        self.bytes += len(data)
        self.seq = seq
        self.mem.write(self.L.served, _f32(seq))
        return 1


class SimDram:
    """An ISA simulator slice's DRAM (a uint8 array) as ExpertServer's memory."""

    def __init__(self, dram: np.ndarray):
        self.dram = dram

    def write(self, addr: int, data) -> None:
        b = np.frombuffer(bytes(data) if not isinstance(data, np.ndarray) else data.tobytes(),
                          np.uint8)
        self.dram[addr:addr + len(b)] = b

    def read(self, addr: int, n: int) -> bytes:
        return self.dram[addr:addr + n].tobytes()


class BackendDram:
    """A backend's DRAM (write(s, addr, array) / read(s, addr, n): IsaBackend, RtlBackend,
    BoardBackend) as ExpertServer's memory; one slice (the board's)."""

    def __init__(self, backend, s: int = 0):
        self.backend, self.s = backend, s

    def write(self, addr: int, data) -> None:
        b = data if isinstance(data, np.ndarray) else np.frombuffer(bytes(data), np.uint8)
        self.backend.write(self.s, addr, np.ascontiguousarray(b).view(np.uint8).reshape(-1))

    def read(self, addr: int, n: int) -> bytes:
        return np.asarray(self.backend.read(self.s, addr, n)).view(np.uint8).tobytes()


class BoardDram:
    """The card's DRAM as ExpertServer's memory at the link's rate (a BoardBackend's Board over
    a transport that may DMA from a worker thread: XdmaTransport). Board.write, which
    BackendDram calls, copies an expert's 1.67-5.85 MB four times on its way (the slot's bytes
    out of the pool, CHASH's swaps, the two channel runs, a bounce for the DMA's alignment) and
    widens every smaller write with a read of the card; the 35B-A3B's card run wrote 544 MB/s
    against the link's 1.37. Here:
    - write_slot: an expert's bytes (or a part's: a hint's, ExpertServer.step) go to the two
      channels' runs in one pass (the beat interleave and CHASH's swaps, `np.take` of 64-byte
      beats) into page-aligned staging buffers, then one DMA call per channel; an expert of a
      split-format pool file (SplitRecord) is read straight into the runs, and when nothing
      is in flight (a request's first miss) in `pieces` parts, each DMAed as soon as it is
      read;
    - the host's own words (served and the directory: the card only reads them) are kept in a
      shadow and written with no read first: a write within one 64-byte beat (an entry,
      served) as that beat alone, one DMA call on its channel; a longer one as whole 128-byte
      blocks;
    - one worker thread makes every DMA call in order, while the server stages the next
      expert: an expert's data lands before its directory entry, every entry before served
      (one queue, one h2c stream). flush() waits for the queue and raises a worker's error;
      ExpertServer calls it before poll returns, so no DMA of the server's is in flight
      while anything else uses the card."""

    def __init__(self, backend, layout: Layout, depth: int = 3, pieces: int = 2):
        from .board import BEAT
        self.board, self.L = backend.board, layout
        blk = 2 * BEAT                        # a chunk: one beat on each channel
        end = layout.dir + 8 * layout.E * layout.layers
        if layout.served % blk or layout.served < layout.mbox + 2 * LINE:
            raise ValueError("served must start a 128-byte block of the host's own words")
        self.blk, self.lo = blk, layout.served
        self.shadow = np.zeros(-(-(end - self.lo) // blk) * blk, np.uint8)
        self.depth, self._bufs, self._base, self._i1 = depth, None, None, None
        self._idx: dict = {}                  # (address, chunks) -> channel 0's beat indices
        self._pieces: dict = {}               # bytes -> per staging pair: its split pieces
        self._free: queue.Queue = queue.Queue()
        self._q: queue.Queue = queue.Queue()
        self._err: BaseException | None = None
        self._thread: threading.Thread | None = None
        self.dma_s, self.dma_bytes = 0.0, 0     # the worker's slot DMA: seconds and bytes
        self.direct = 0                         # experts read from the file into their runs
        self.pieces, self._lead = pieces, True  # _lead: no DMA in flight since the last flush

    # ---- the worker
    def _work(self) -> None:
        while True:
            fn, slot = self._q.get()
            try:
                if self._err is None:
                    fn()
            except BaseException as e:          # noqa: BLE001 (raised by flush)
                self._err = e
            finally:
                if slot is not None:
                    self._free.put(slot)
                self._q.task_done()

    def _put(self, fn, slot=None) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._work, daemon=True,
                                            name="otpu-offload-dma")
            self._thread.start()
        self._q.put((fn, slot))

    def flush(self) -> None:
        self._q.join()
        self._lead = True
        if self._err is not None:
            e, self._err = self._err, None
            raise e

    # ---- the memory
    def write(self, addr: int, data) -> None:
        b = (np.ascontiguousarray(data).view(np.uint8).reshape(-1)
             if isinstance(data, np.ndarray) else np.frombuffer(bytes(data), np.uint8))
        a = addr - self.lo
        if 0 <= a and a + len(b) <= len(self.shadow):         # the host's own words
            self.shadow[a:a + len(b)] = b
            h = self.blk // 2
            if a % h + len(b) <= h:                             # one beat: one DMA call
                out, at = self.shadow[a - a % h:a - a % h + h].copy(), self.lo + a - a % h
                self._put(lambda: self._beat(at, out))
                return
            a0, a1 = a // self.blk * self.blk, -(-(a + len(b)) // self.blk) * self.blk
            out, at = self.shadow[a0:a1].copy(), self.lo + a0
            self._put(lambda: self._blocks(at, out))
        else:                                                   # the card's words: in order
            self.flush()
            self.board.write(addr, b)

    def _staging(self, nbytes: int) -> None:
        """depth pairs of page-aligned buffers for an expert's two channel runs (a shorter
        write, a hint's part, uses their first bytes)."""
        if self._bufs is not None and len(self._bufs[0][0]) >= nbytes // 2:
            return
        self.flush()
        n = nbytes // self.blk
        self._bufs = [[_page_buf(nbytes // 2) for _ in (0, 1)] for _ in range(self.depth)]
        self._base = 2 * np.arange(n, dtype=np.intp)
        self._i1, self._idx, self._pieces = np.empty(n, np.intp), {}, {}
        while not self._free.empty():
            self._free.get()
        for i in range(self.depth):
            self._free.put(i)

    def _split_pieces(self, nbytes: int) -> list:
        """Per staging pair, an nbytes record's RUN / 2-byte pieces of the two runs, in file
        order for either parity ([block, parity, run])."""
        if nbytes not in self._pieces:
            h, nb, ps = RUN // 2, -(-nbytes // RUN), []
            for b0, b1 in self._bufs:
                b0, b1 = memoryview(b0)[:nbytes // 2], memoryview(b1)[:nbytes // 2]
                v = np.empty((nb, 2, 2), object)
                for j in range(nb):
                    p0, p1 = b0[j * h:(j + 1) * h], b1[j * h:(j + 1) * h]
                    v[j, 0, 0], v[j, 0, 1], v[j, 1, 0], v[j, 1, 1] = p0, p1, p1, p0
                ps.append(v)
            self._pieces[nbytes] = ps
        return self._pieces[nbytes]

    def write_slot(self, addr: int, data) -> None:
        from .board import swapped
        if (isinstance(data, SplitRecord) and self.board.chash and addr % RUN == 0
                and len(data) % self.blk == 0):
            self._staging(len(data))            # the file's runs, read in place
            i = self._free.get()
            pieces = self._split_pieces(len(data))[i]
            nb = pieces.shape[0]
            iov = pieces[np.arange(nb), _parity(addr // RUN + np.arange(nb))]
            bufs, h = self._bufs[i], RUN // 2
            # nothing in flight: the link waits for this read, so it goes in parts (each one
            # DMA call per channel more)
            parts = self.pieces if self._lead or not self._q.unfinished_tasks else 1
            self._lead = False
            cut = sorted({nb * p // parts for p in range(parts + 1)})
            for j0, j1 in zip(cut[:-1], cut[1:]):
                data.readv(iov[j0:j1].reshape(-1).tolist(), j0 * RUN)
                self._put(lambda a=j0 * h, b=min(j1 * h, len(data) // 2):
                          self._dma(addr // 2, bufs, a, b), i if j1 == nb else None)
            self.direct += 1
            return
        src = (np.ascontiguousarray(data).view(np.uint8).reshape(-1)
               if isinstance(data, np.ndarray) else np.frombuffer(bytes(data), np.uint8))
        n = len(src) // self.blk
        if addr % self.blk or len(src) % self.blk:
            return self.write(addr, src)
        self._staging(len(src))
        i0 = self._idx.get((addr, n))
        if i0 is None:                          # once per slot: CHASH's swaps at its address
            i0 = self._idx[addr, n] = (self._base[:n] + swapped(addr, n) if self.board.chash
                                       else self._base[:n])
        i1 = self._i1[:n]
        np.bitwise_xor(i0, 1, out=i1)           # channel 1 takes each chunk's other beat
        i = self._free.get()                    # a staging pair the worker is done with
        bufs, beats, h = self._bufs[i], src.view("V64"), n * self.blk // 2
        np.take(beats, i0, out=bufs[0][:h].view("V64"))       # channel 0's run
        np.take(beats, i1, out=bufs[1][:h].view("V64"))       # channel 1's
        self._put(lambda: self._dma(addr // 2, bufs, 0, h), i)

    def _beat(self, at: int, data: np.ndarray) -> None:
        """One 64-byte beat of the host's words (at: 64-byte aligned) to the channel that holds
        it (CHASH: beat b of chunk m on channel b ^ parity(m))."""
        m, c = at // self.blk, at // (self.blk // 2) % 2
        if self.board.chash:
            c ^= m.bit_count() & 1
        self.board.t.mem_write(c, m * (self.blk // 2), data)

    def _blocks(self, at: int, data: np.ndarray) -> None:
        """Whole chunks of the host's words (at, len: multiples of 128) to the two channels, as
        Board.write does (CHASH's swaps), without its general path's copies."""
        from .board import BEAT, swapped
        v = data.reshape(-1, 2, BEAT)
        sw = swapped(at, len(v)) if self.board.chash else np.zeros(len(v), bool)
        for c in (0, 1):
            run = np.where(sw[:, None], v[:, 1 - c], v[:, c])
            self.board.t.mem_write(c, at // 2, run.reshape(-1))

    def _dma(self, off: int, bufs, a: int = 0, b: int | None = None) -> None:
        """bufs[c][a:b] to channel c's run from channel offset off + a."""
        t0 = time.perf_counter()
        for c in (0, 1):
            self.board.t.mem_write(c, off + a, bufs[c][a:b])
        self.dma_s += time.perf_counter() - t0
        self.dma_bytes += 2 * len(bufs[0][a:b])

    def read(self, addr: int, n: int) -> bytes:
        return np.asarray(self.board.read(addr, n)).view(np.uint8).tobytes()


def _page_buf(n: int) -> np.ndarray:
    """n bytes of page-aligned host memory: an h2c DMA at full speed from any 64-byte aligned
    card address (opentpu.host.board, DMA_PLACE: writes need host - card = 0 mod 64)."""
    import mmap
    return np.frombuffer(mmap.mmap(-1, n), np.uint8)


def dram_of(backend, layout: Layout):
    """ExpertServer's memory on `backend`: BoardDram on a card whose transport DMAs from a
    worker thread (XdmaTransport), else BackendDram (the ISA simulator, the board models)."""
    t = getattr(getattr(backend, "board", None), "t", None)
    return BoardDram(backend, layout) if getattr(t, "threaded", False) else BackendDram(backend)
