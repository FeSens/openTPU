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
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

LINE = 64              # the mailbox's words, its row and each flag on their own 64-byte lines


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
    """Serves the card's expert requests (module docstring).

    mem: write(addr, bytes-like) and read(addr, n) -> bytes-like on the card's DRAM (the ISA
    simulator's slice DRAM through SimDram, or the board through BoardBackend's read / write).
    pool(g): expert g in the card's slot format, `layout.slot_bytes` long (host RAM or a file;
    the host never computes with it)."""

    def __init__(self, mem, layout: Layout, pool):
        self.mem, self.L, self.pool = mem, layout, pool
        self.lru = [OrderedDict() for _ in range(layout.layers)]    # g -> slot address
        self.free = [[a + i * layout.slot_bytes for i in range(n)] for a, n in layout.slots]
        self.seq = 0                        # the last request served
        self.hits = self.misses = self.bytes = 0
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
        self.seq = 0
        for g in warm:
            j = g // L.E
            if self.free[j] and g not in self.lru[j]:
                self._insert(j, g, self.free[j].pop(0))
        for lru in self.lru:                # the profile's first expert is the most recent
            for g in reversed(list(lru)):
                lru.move_to_end(g)
        self.hits = self.misses = self.bytes = 0     # counted from here: the requests'

    def poll(self) -> int:
        """Serve the card's request if it posted one since the last served; returns 1 if so.
        The card's seq is read first, then the row."""
        seq = int(np.frombuffer(bytes(self.mem.read(self.L.mbox, 4)), np.float32)[0])
        if seq == self.seq:
            return 0
        if seq != self.seq + 1:
            raise RuntimeError(f"the card posted request {seq} with {self.seq} served: its "
                               f"fence (WAITW served >= seq) is missing")
        ids = [int(g) for g in np.frombuffer(bytes(self.mem.read(self.L.row, 4 * self.L.k)),
                                             np.float32)]
        if self.history is not None:
            self.history.append(ids)
        self.serve(ids)
        self.seq = seq
        self.mem.write(self.L.served, _f32(seq))
        return 1

    def serve(self, ids) -> None:
        """One request: its k global ids, all of one MoE layer."""
        E = self.L.E
        j = ids[0] // E
        if any(g // E != j or not 0 <= g < E * self.L.layers for g in ids):
            raise ValueError(f"request {ids} is not one layer's experts")
        lru = self.lru[j]
        for g in ids:
            if g in lru:
                lru.move_to_end(g)
                self.hits += 1
                continue
            self.misses += 1
            if self.free[j]:
                slot = self.free[j].pop(0)
            else:
                victim = next((v for v in lru if v not in ids), None)
                if victim is None:
                    raise RuntimeError(f"layer {j}: {len(lru)} slots for a request of "
                                       f"{len(ids)}")
                slot = lru.pop(victim)
                self.mem.write(self.L.entry(victim), np.zeros(2, np.uint32))
            self._insert(j, g, slot)

    def _insert(self, j: int, g: int, slot: int) -> None:
        data = self.pool(g)
        if len(data) != self.L.slot_bytes:
            raise ValueError(f"expert {g}: {len(data)} bytes, slots hold {self.L.slot_bytes}")
        self.mem.write(slot, data)
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

    WORDS = 3 * LINE        # the mailbox's bytes

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
