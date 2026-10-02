"""Path (a)'s host side: the expert server (docs/offload.md section 5.2).

The card computes everything: it routes, posts the ids of the experts a MoE layer needs to a
mailbox in its DRAM, computes the ones its directory says it has and waits (WAITW) on the
directory entries of the others. This module only moves bytes: it serves each request from the
expert pool (host RAM, every expert already in the card's slot format) into per-layer LRU
slots in the card's DRAM, and keeps the directory. The same code serves the ISA simulator (its
WAITW host hook) and the card (BoardBackend.host: polled while a run is in flight).

DRAM words (`Layout`), all 4-byte words at 64-byte aligned bases:

    mbox                   seq: the card's last request, as a float (0.0 before the first)
    mbox + 4               its count of ids, as a float (the card writes it with each request:
                           k; a prefill run of R rows R k, repeats included, moe.moe_ffn_rows;
                           0.0 is read as k)
    mbox + 64              the global expert ids of request seq (floats): the first 16
    served                 the last request the host has finished, as a float
    answer + 4 * n         the request's answer: the slot of its n-th id (the id's first place
                           in the request) when that expert is missing, else 0 (u32); the card
                           zeroes the words after its experts
    dir + 8 * g            expert g's entry: {slot address (u32), present (f32: 1.0 or 0.0)}
    row2                   ids 17-32 of a request of more than 16 (a layout of 2 lines only:
                           the 128-byte block after the directory)
    slot + tag             the slot's tag word (u32, its tag chunk's first): nonzero once the
                           expert the answer named has landed; the card zeroes it when it uses
                           the slot's expert

A global expert id is `j * E + e` for the j-th MoE layer's expert e (the card's ARGMAX gives it
with base j * E). Per request (docs/offload.md 10.11) the host, for each id missing from its
copy of the directory, picks the layer's least recently used expert that the request does not
name; writes each missing expert into its slot with its tag chunk, the tag word in the DMA's
last beat, and the answer (one 64-byte beat: the missing ids' slots) once the first expert's
first part is on its way; then the directory: {slot, 1.0} to the new entries and {0, 0.0} to
the victims'; then `served = seq`. The card fences each
layer before it posts, `WAITW served >= seq` (its last request), so one request row and one
answer are enough, and no eviction for a layer is in flight while it uses that layer's slots
(docs/offload.md 5.2). It reads the present flags of the ids it posts from the directory: a
present expert's slot comes from its entry; a missing one's from the answer, its data once its
tag is nonzero. An entry says present only after its expert has landed, so an expert can read
as present (the card read the flag after the directory's write) but never the other way. The
card zeroes the tag of every expert it uses and then the answer, before its next post: a tag
the host writes later is a new expert's. A slot whose tag the card has not zeroed (a hinted
expert no request used) has it cleared before the slot takes another expert (`armed`).

During a layer-major prefill (docs/offload.md 13: one MoE layer at a time over the whole
prompt) the slots are pooled (`begin_prefill`): a missing expert of the running layer takes a
free slot of any layer, else the slot of the least recently used expert the request does not
name, of any layer (a finished layer's first). `end_prefill` restores each layer's own number
of slots: a layer keeps its experts of most decayed use up to it, the rest leave (their entries
cleared), and with "lazy" (the default; docs/offload.md 13) decode's misses fill the slots, with
"eager" each layer's experts of most use in the prompt are loaded at once.

A request whose ids are at G = layers x E and above is a hint (docs/offload.md 12: the layer's
router on its input, before its mixer). With the "lfu" policy the host gives each hinted expert
not in a slot one now (a free one, or the victim, its entry cleared; hint_n / hint_top cap them,
docs/offload.md 12.7), writes served, and moves the experts on the link's idle time: one part
per poll that finds no request, the tag with the last part, then the entry. A request naming one
still on its way names its slot in the answer and sends the rest at once. "lru" ignores hints.

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
TAG = 128              # a slot's tag chunk (one beat on each channel; the tag word its first)
SPLIT = "split4k"      # the split format's name in a pool file's <file>.format
IOV_MAX = 1024         # buffers per os.preadv (Linux's UIO_MAXIOV)
# halt_aware idle parts (docs/offload.md 13.10): a part's time before any is measured (a call
# pair's 122 us and the pairs' 3.15 GB/s, session 16, and the poll), and how far past its
# expected end a run gets idle parts again
PART_S0, PART_GBS, HOLD_LATE = 0.2e-3, 3.15e9, 1e-3


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


_LIBC: list = []


def _libc_preadv():
    """libc's preadv through ctypes (the GIL released for the call), or None."""
    if not _LIBC:
        import ctypes
        try:
            f = ctypes.CDLL(None, use_errno=True).preadv
            f.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong]
            f.restype = ctypes.c_ssize_t
        except (OSError, AttributeError):
            f = None
        _LIBC.append(f)
    return _LIBC[0]


def preadv_iov(fd: int, iov: np.ndarray, off: int) -> None:
    """preadv's fill from an iovec array: iov[i] = (address, length) as uint64, C order, of
    buffers that outlive the call. libc's preadv in groups of IOV_MAX, a short read resumed: no
    buffer objects built and the GIL released for the whole read (os.preadv holds it while it
    takes each buffer: 816 of them an expert of the 35B, ~170 us on the card's host, docs/
    offload.md 10.8), so the DMA thread runs meanwhile."""
    import ctypes
    f = _libc_preadv()
    iov = np.ascontiguousarray(iov, np.uint64)
    i, n = 0, len(iov)
    while i < n:
        g = min(IOV_MAX, n - i)
        k = f(fd, iov.ctypes.data + 16 * i, g, off)
        if k < 0:
            e = ctypes.get_errno()
            raise OSError(e, f"preadv at {off}: {os.strerror(e)}")
        if k == 0:
            raise OSError(f"preadv at {off}: end of file")
        off += k
        lens = np.cumsum(iov[i:i + g, 1])
        if k == lens[-1]:                       # the whole group: the usual case
            i += g
            continue
        j = int(np.searchsorted(lens, k, side="right"))     # buffers read whole
        part = k - (int(lens[j - 1]) if j else 0)
        iov = np.array(iov[i + j:], np.uint64)              # the rest, the first one cut
        iov[0, 0] += np.uint64(part)
        iov[0, 1] -= np.uint64(part)
        i, n = 0, len(iov)


class SplitRecord:
    """An expert in the split format, read from its file when it is written: readv(bufs, at)
    reads its bytes from byte `at` on, in file order, into the buffers; readiov(iov, at), when
    given, the same into an iovec array (preadv_iov's). BoardDram on a CHASH card reads it
    straight into its two channel runs; as bytes (bytes(), np.asarray) it is the slot's own
    bytes."""

    def __init__(self, n: int, readv, readiov=None):
        self.n, self.readv, self.readiov = n, readv, readiov

    def __len__(self) -> int:
        return self.n

    def part(self, a: int, b: int) -> "SplitRecord":
        """Its bytes a..b (whole RUN blocks, or to its end), in the split format: the blocks
        are its own."""
        if a % RUN or (b % RUN and b != self.n) or not 0 <= a < b <= self.n:
            raise ValueError(f"a split record's part {a}..{b} of {self.n} bytes")
        return SplitRecord(b - a, lambda bufs, at=0: self.readv(bufs, a + at),
                           None if self.readiov is None else
                           lambda iov, at=0: self.readiov(iov, a + at))

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
    page cache: the card's host SSD reads an expert of 1.67 MB that is not in the page cache
    in 4.2 ms (400 MB/s; docs/offload.md 10.6). It moves bytes only.

    `mapped` (the default): every expert read (and warmed) is then touched through a read-only
    map of the file, a byte a page. Under MGLRU (Linux's multi-generational LRU) a file page
    read only through read() ages out before one a process has mapped and touched: a
    checkpoint any process mmapped once outlived the pool's pages (docs/offload.md 10.7).

    `io`, when set to {} (moe_card's decode), counts the reads by where they came from:
    "cached" the reads whose pages were all in the page cache just before (mincore), "disk"
    the others ("unknown" where mincore cannot tell), each [reads, seconds, bytes, bytes not
    in the page cache]."""

    def __init__(self, path, slot: int, split: bool, mapped: bool = True):
        import mmap
        self.fd, self.slot, self.split = os.open(path, os.O_RDONLY), slot, split
        self.bufs = [mmap.mmap(-1, slot) for _ in range(2)]
        self.k, self.warm_t = 0, None
        self.arr = self.packed = self.ids = self.resident_at_open = None    # (moe.open_pool)
        self.io: dict | None = None
        self.iov = True                 # get's records read through preadv_iov where libc has it
        self.mapped = mapped
        # mapped: each read's touch waits for touch_deferred (ExpertServer, once a request is
        # served) instead of delaying the read's DMA (docs/offload.md 10.11)
        self.defer_touch = False
        self._touches: list = []
        self._mc = None                 # the file's read-only map: (map, its view, libc,
                                        # the view's address)
        self._vec = None                # mincore's vector (_absent: reused)
        if mapped:                      # (made here: the warm thread and the server share it)
            self._map()

    def _map(self):
        """The file's read-only map, its uint8 view and libc (io's mincore, mapped's touches),
        or None where the host has none."""
        import ctypes
        import mmap
        if self._mc is None:
            try:
                mm = mmap.mmap(self.fd, os.fstat(self.fd).st_size, prot=mmap.PROT_READ)
                v = np.frombuffer(mm, np.uint8)
                libc = ctypes.CDLL(None, use_errno=True)
                libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
                self._mc = (mm, v, libc, v.ctypes.data)
            except (OSError, AttributeError, ValueError):
                self._mc = False
        return self._mc or None

    def _touch(self, off: int, n: int) -> None:
        """A byte of every page under [off, off + n) read through the map (mapped)."""
        import mmap
        m = self._map()
        if m is not None and n > 0:
            int(m[1][off // mmap.PAGESIZE * mmap.PAGESIZE:off + n:mmap.PAGESIZE].sum())

    def touch_deferred(self) -> None:
        """The touches defer_touch held back, now."""
        while self._touches:
            self._touch(*self._touches.pop())

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

    def _absent(self, off: int, n: int) -> int | None:
        """Bytes of the file's pages under [off, off + n) not in the page cache (mincore), or
        None where the host cannot tell."""
        import ctypes
        import mmap
        m = self._map()
        if m is None or n <= 0:
            return None
        pg = mmap.PAGESIZE
        lo, hi = off // pg, -(-(off + n) // pg)
        if self._vec is None or len(self._vec[1]) < hi - lo:
            k = max(hi - lo, -(-self.slot // pg) + 1)
            b = (ctypes.c_ubyte * k)()
            self._vec = (b, np.frombuffer(b, np.uint8))
        if m[2].mincore(m[3] + lo * pg, (hi - lo) * pg, self._vec[0]) != 0:
            return None
        return (hi - lo - int(np.count_nonzero(self._vec[1][:hi - lo] & 1))) * pg

    def _read(self, bufs, off: int) -> None:
        self._io(sum(map(len, bufs)), off, lambda: preadv(self.fd, bufs, off))

    def _read_iov(self, iov: np.ndarray, off: int) -> None:
        self._io(int(iov[:, 1].sum()), off, lambda: preadv_iov(self.fd, iov, off))

    def _io(self, n: int, off: int, read) -> None:
        """read() of n bytes at off, counted in io, then touched through the map."""
        if self.io is None:
            read()
        else:
            gone = self._absent(off, n)
            t0 = time.perf_counter()
            read()
            s = self.io.setdefault("unknown" if gone is None else "disk" if gone else "cached",
                                   [0, 0.0, 0, 0])
            s[0] += 1
            s[1] += time.perf_counter() - t0
            s[2] += n
            s[3] += gone or 0
        if self.mapped:
            if self.defer_touch:
                self._touches.append((off, n))
            else:
                self._touch(off, n)

    def get(self, g: int):
        if self.split:
            return SplitRecord(self.slot,
                               lambda bufs, at=0: self._read(bufs, g * self.slot + at),
                               None if not self.iov or _libc_preadv() is None else
                               lambda iov, at=0: self._read_iov(iov, g * self.slot + at))
        self.k ^= 1
        self._read([memoryview(self.bufs[self.k])], g * self.slot)
        return np.frombuffer(self.bufs[self.k], np.uint8)

    def willneed(self, ids) -> None:
        """Queue these experts' reads into the page cache at once (POSIX_FADV_WILLNEED, where
        the host has it): a request's misses that are not there read in parallel, not one
        after another (opentpu's SATA SSD: 4.2 -> 3.4-3.6 ms an expert of 1.67 MB)."""
        if hasattr(os, "posix_fadvise"):
            for g in ids:
                os.posix_fadvise(self.fd, int(g) * self.slot, self.slot, os.POSIX_FADV_WILLNEED)

    def warm(self, ids) -> threading.Thread:
        import mmap
        scratch = memoryview(mmap.mmap(-1, self.slot))

        def run():
            for g in ids:
                preadv(self.fd, [scratch], int(g) * self.slot)
                if self.mapped:
                    self._touch(int(g) * self.slot, self.slot)
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
    row2: int = 0          # a request's second line of ids (0: requests of at most 16)
    pitch: int = 0         # from a slot to the next (0: slot_bytes + its tag chunk)
    answer: int = 0        # the request's answer: per id, its slot when missing (docs/offload.md
                           # 10.11)
    tag: int = 0           # a slot's tag word, from the slot (0: slot_bytes up to whole chunks)

    def __post_init__(self):
        if not self.tag:
            object.__setattr__(self, "tag", -(-self.slot_bytes // TAG) * TAG)
        if not self.pitch:
            object.__setattr__(self, "pitch", self.tag + TAG)

    @staticmethod
    def build(base: int, E: int, k: int, slots_per_layer, slot_bytes: int,
              lines: int = 1) -> "Layout":
        """The words from `base` up (64-byte aligned), then the slots, layer after layer, from
        the next 4 KiB page. A slot is the expert's slot_bytes, then its tag chunk (TAG bytes
        from the first whole chunk after them: the tag word that says the expert's DMA has
        landed, docs/offload.md 10.11), rounded up to whole RUN blocks from the last (the split
        pool format's blocks land on the card's: BoardDram reads an expert straight into its
        DMA runs only at a RUN-aligned slot), or under RUN to the alignment slot_bytes has (a
        power of two up to RUN: D-byte aligned slots, the MXU streams whole chunks). lines=2:
        requests of up to 32 ids (a layer-major prefill's runs of 4 rows), their answer two
        lines, their second line after the directory on a 128-byte block of its own (outside
        BoardDram's shadow of the host's words)."""
        if base % LINE or slot_bytes % LINE or not 0 < k <= LINE // 4 or lines not in (1, 2):
            raise ValueError("unaligned base or slot size, k too large, or not 1 or 2 lines")
        mbox = base
        served = mbox + 2 * LINE
        answer = served + LINE
        d = answer + lines * LINE
        end = d + 8 * E * len(slots_per_layer)
        row2 = -(-end // (2 * LINE)) * 2 * LINE if lines == 2 else 0
        a = -(-(row2 + LINE if row2 else end) // 4096) * 4096        # slots page-aligned
        tag = -(-slot_bytes // TAG) * TAG
        g = min(RUN, slot_bytes & -slot_bytes)                         # its alignment
        pitch = -(-(tag + TAG) // RUN) * RUN if tag + TAG > RUN else -(-(tag + TAG) // g) * g
        slots = []
        for n in slots_per_layer:
            slots.append((a, int(n)))
            a += int(n) * pitch
        return Layout(E, k, tuple(slots), slot_bytes, mbox, served, d, row2, pitch, answer,
                      tag)

    @property
    def max_ids(self) -> int:
        """The most ids a request carries."""
        return (2 if self.row2 else 1) * LINE // 4

    @property
    def layers(self) -> int:
        return len(self.slots)

    @property
    def end(self) -> int:
        a, n = self.slots[-1]
        return a + (n - 1) * self.pitch + self.tag + TAG if n else a

    def all_slots(self):
        """Every slot's address, layer after layer."""
        return [a + i * self.pitch for a, n in self.slots for i in range(n)]

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
    (a request waits for at most one part on the link). drop: a request withdraws its layer's
    hinted experts it does not name that have not landed (their slots free again, the link
    kept for the next layer's hints)."""

    def __init__(self, mem, layout: Layout, pool, policy: str = "lru", half: float = 32.0,
                 part: int = 512 << 10, drop: bool = False):
        self.mem, self.L, self.pool = mem, layout, pool
        if policy not in ("lru", "lfu"):
            raise ValueError(f"replacement policy {policy!r}")
        if part <= 0 or part % RUN:
            raise ValueError(f"a hint's part of {part} bytes (RUN blocks)")
        # the victim: the layer's least recently used expert the request does not name, or with
        # "lfu" the one of least use, each use decaying by half every `half` requests of its
        # layer (docs/offload.md 10.3: 9-15% fewer misses than LRU on the traces)
        self.policy, self.half, self.part, self.drop = policy, half, part, drop
        self.lru = [OrderedDict() for _ in range(layout.layers)]    # g -> slot address
        self.t = [0] * layout.layers                                # requests per layer
        self.use: list = [{} for _ in range(layout.layers)]          # g -> log2 use + t / half
        self.free = [[a + i * layout.pitch for i in range(n)] for a, n in layout.slots]
        self.pending: OrderedDict = OrderedDict()   # hinted, in a slot, not landed: g -> bytes sent
        self.pooled = False                 # a layer-major prefill: every slot serves its layer
        self.order: OrderedDict = OrderedDict()      # pooled: the experts in slots, oldest first
        self.seq = 0                        # the last request served
        self.hits = self.misses = self.bytes = 0
        # hints served; hinted experts landed on idle time, sent by the request that named
        # them, replaced before they landed, withdrawn by their layer's request (drop)
        self.hints = self.prefetched = self.promoted = self.dropped = self.withdrawn = 0
        # a hint's caps (docs/offload.md 12.7): of its first hint_top ids (its router's best
        # first; 0: all), the first hint_n not in a slot get one (0: every one not in a slot)
        self.hint_n = self.hint_top = 0
        # idle parts (docs/offload.md 13.10), opt-in until the card's A/B: read_ahead, each read
        # by the poll before, beside its DMA, and sent as one DMA call per channel (a memory
        # with stage: BoardDram); halt_aware, none started when the running program's expected
        # end is nearer than an idle part takes (part_s: the measured parts' average). The end:
        # the memory's run_clock (the run's start and its time with no waits) plus the run's
        # own waits (_waits: each of its requests with misses, seen to served)
        self.read_ahead = self.halt_aware = False
        self._staged = None                 # ((g, slot, from, to), the memory's staged part)
        self.part_s: float | None = None
        self.holds = 0                      # polls that held an idle part back (halt_aware)
        self._run, self._waits = None, 0.0
        self.history: list | None = None    # a list: each request's ids are appended
        # a list: (perf_counter when seen, when done, "h" hint / "d" request / "p" a hint's
        # part, its layer or expert, misses or the expert's bytes sent): the host's timeline of
        # the hints (moe_card --hint-trace)
        self.events: list | None = None
        # called with a request's missing ids but its first once that one's first part is on
        # its way (PoolFile.willneed: the reads of those not in the page cache queued at once)
        self.ahead = None
        # the PoolFile behind pool, when there is one (moe.serve): the touches it defers run
        # once a request is served, while the card computes
        self.pool_file = None
        self._victims: list = []            # this request's victims: entries cleared at its end
        # the experts the last poll landed on idle time (step: their last part and entry): the
        # card may have posted a request naming one after that poll read seq and read its entry
        # before the entry landed, as missing; that request's answer names their slots (late)
        self._fresh: set = set()
        self._racy: set = set()             # (serve's: the poll before's _fresh)
        self.late = 0                       # hits answered so (docs/offload.md 12.8)
        self.last = None                    # what the last poll served: ("d" / "h", its layer),
                                            # None for a hinted expert's part (PollPacer's key)
        self.clear_late = True              # (False: each at once, before its slot is written)
        # the host's copy of the directory ({slot, present's bits} per expert) and the entries
        # this request changed (written together after its experts' data: _dir_flush)
        self.dirv = np.zeros((layout.E * layout.layers, 2), np.uint32)
        self._dirty: set = set()
        # slots whose tag word may be nonzero: written with a tag the card has not yet zeroed
        # (it zeroes the tags of the experts it uses before its next post: _used, the last
        # request's slots, are clear once a later request is seen)
        self.armed: set = set()
        self._used: set = set()
        # a layer-major prefill's layer ahead (ahead_layer): its experts not yet given a slot,
        # most wanted first (each takes one when its first part goes on an idle poll), the ones
        # of them on their way (in `pending`), and the layers the runs until the next call may
        # use (no victim from them); begin_prefill's part size while the prefill lasts; the
        # queue's layer (a hint for it adds to the queue: the card's own router, 13.4)
        self.queue: OrderedDict = OrderedDict()
        self._ahead: set = set()
        self._spare: frozenset = frozenset()
        self._part, self._ahead_off, self._qlayer = None, False, None
        self.aheads = self.landed = 0       # ahead_layer's calls; its experts landed whole
        self.hinted_ahead = 0               # ids hints added to its queue

    def load(self, warm=()) -> None:
        """At image load: an empty directory, mailbox and answer, every slot's tag zero, then
        the experts `warm` names (global ids, most wanted first, e.g. a profile's order) in
        their layers' slots while slots last."""
        L = self.L
        self.mem.write(L.mbox, np.zeros(2 * LINE // 4, np.float32))
        self.mem.write(L.served, _f32(0.0))
        self.mem.write(L.answer, np.zeros(L.max_ids, np.uint32))
        self.dirv[:] = 0
        self._dirty.clear()
        self.mem.write(L.dir, self.dirv)
        for a in L.all_slots():
            self.mem.write(a + L.tag, _tag_beat(0))
        self.armed.clear()
        self._used.clear()
        for lru, fr in zip(self.lru, self.free):
            fr.extend(lru.values())
            lru.clear()
        self.pending.clear()
        self._victims.clear()
        self.pooled = False
        self.order.clear()
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
        self.hints = self.prefetched = self.promoted = self.dropped = self.withdrawn = 0

    def poll(self) -> int:
        """Serve the card's request if it posted one since the last served, else send a part of
        a hinted expert; returns 1 if it did either. The card's seq is read first, then the
        row."""
        seq, n = (int(v) for v in np.frombuffer(bytes(self.mem.read(self.L.mbox, 8)),
                                                np.float32))
        t0 = time.perf_counter()
        fresh, self._fresh = self._fresh, set()
        if seq == self.seq:
            if not self.pending and not self._next_ahead():
                return 0
            if self._hold():
                self._flush()               # (a victim's entry _next_ahead cleared)
                return 0
            g = next(iter(self.pending))
            self.step()
            if g not in self.pending:       # landed: its entry after this poll read seq
                self._fresh.add(g)
            self._stage_next()              # (the next part's read beside this one's DMA)
            self._flush()
            self._touched()
            self.last = None
            dt = time.perf_counter() - t0
            self.part_s = dt if self.part_s is None else 0.8 * self.part_s + 0.2 * dt
            if self.events is not None:
                self.events.append((t0, time.perf_counter(), "p", g,
                                    self.pending.get(g, self.L.slot_bytes)))
            return 1
        if seq != self.seq + 1:
            raise RuntimeError(f"the card posted request {seq} with {self.seq} served: its "
                               f"fence (WAITW served >= seq) is missing")
        n = n or self.L.k                   # (a multi-row request's count: its ids, repeats
        if not 0 < n <= self.L.max_ids:     # included, each served once)
            raise RuntimeError(f"request {seq} of {n} ids: the layout's lines hold "
                               f"{self.L.max_ids}")
        h = LINE // 4
        raw = bytes(self.mem.read(self.L.row, 4 * min(n, h)))
        if n > h:
            raw += bytes(self.mem.read(self.L.row2, 4 * (n - h)))
        pos: dict = {}                      # each id's first place: its answer word
        for i, v in enumerate(np.frombuffer(raw, np.float32)):
            pos.setdefault(int(v), i)
        ids = list(pos)
        self.armed -= self._used            # the card zeroed their tags before this post
        self._used = set()
        G = self.L.E * self.L.layers
        m0 = self.misses
        if ids[0] >= G:
            self.hint([g - G for g in ids])
        else:
            if self.history is not None:
                self.history.append(ids)
            self._racy = fresh
            try:
                self.serve(ids, [pos[g] for g in ids])
            finally:
                self._racy = set()
            self._used = {self.lru[g // self.L.E][g] for g in ids}
        self.seq = seq
        self.mem.write(self.L.served, _f32(seq))
        self._stage_next()                  # (a hint's first part: read while the card computes)
        self._flush()                       # (no DMA of the server's in flight after poll)
        self._touched()
        self.last = ("h" if ids[0] >= G else "d", ids[0] % G // self.L.E)
        if self.misses > m0 and self.halt_aware:    # (the card waited for them)
            self._wait(time.perf_counter() - t0)
        if self.events is not None:
            self.events.append((t0, time.perf_counter(), "h" if ids[0] >= G else "d",
                                ids[0] % G // self.L.E, self.misses - m0))
        return 1

    def _flush(self) -> None:
        f = getattr(self.mem, "flush", None)
        if f is not None:
            f()

    def _touched(self) -> None:
        if self.pool_file is not None:
            self.pool_file.touch_deferred()

    def _layer(self, ids) -> int:
        E = self.L.E
        j = ids[0] // E
        if any(g // E != j or not 0 <= g < E * self.L.layers for g in ids):
            raise ValueError(f"request {ids} is not one layer's experts")
        return j

    def serve(self, ids, pos=None) -> None:
        """One request: its k global ids, all of one MoE layer; pos: each one's answer word
        (its first place in the request; default its index). The missing ones' slots are
        chosen first, then each one's bytes go with its tag, the answer once the first one's
        first part is on its way (the link starts on the expert), then the directory's new and
        cleared entries (docs/offload.md 10.11). A hit that landed in the poll before this one's
        (_racy) gets its slot in the answer as well (12.8): the card may have read its entry as
        missing."""
        j = self._layer(ids)
        self._unstage()
        pos = list(range(len(ids))) if pos is None else pos
        self.t[j] += 1
        lru, use, t = self.lru[j], self.use[j], self.t[j] / self.half
        for g in ids:                       # the decayed use count, kept as log2 + t / half
            use[g] = math.log2(2.0 ** (use[g] - t) + 1.0) + t if g in use else t
        plan, late = [], []                 # (id, its answer word, slot, its rest only)
        for g, p in zip(ids, pos):
            if g in lru:
                lru.move_to_end(g)
                if g in self.pending:       # hinted, on its way: the rest now
                    self.misses += 1
                    self.promoted += 1
                    self._ahead.discard(g)
                    plan.append((g, p, lru[g], True))
                else:
                    self.hits += 1
                    if g in self._racy:     # landed as the card may have read its entry
                        late.append((p, lru[g]))
                continue
            self.misses += 1
            slot = self._pool_slot(j, ids) if self.pooled else self._slot(j, ids)
            lru[g] = slot                   # (no victim of this request: they are not in ids)
            plan.append((g, p, slot, False))
        self.late += len(late)
        if plan or late:
            ans = np.zeros(self.L.max_ids, np.uint32)
            for g, p, slot, _ in plan:
                ans[p] = slot
            for p, slot in late:            # (its tag landed with it: the card goes on)
                ans[p] = slot
            if not plan:
                self.mem.write(self.L.answer, ans)

            later = [g for g, _, _, rest in plan[1:] if not rest]

            def answer():                   # the first expert's first part on its way: the
                self.mem.write(self.L.answer, ans)          # answer, and the other misses'
                if self.ahead is not None and later:        # reads queued (ahead)
                    self.ahead(later)
            for n, (g, p, slot, rest) in enumerate(plan):
                then = None if n else answer
                if rest:
                    self._send(g, self.L.slot_bytes, entry=False, then=then)
                else:
                    self._fetch(g, slot, then)
                self._dir(g, slot)
        if self.pooled:
            for g in ids:
                self.order[g] = None
                self.order.move_to_end(g)
        if self.drop:                       # its layer's hints it does not name, withdrawn
            for g in [g for g in self.pending if g // self.L.E == j and g not in ids
                      and g not in self._ahead]:
                del self.pending[g]
                self.free[j].append(lru.pop(g))
                self.withdrawn += 1
        self._clear()

    def _clear(self) -> None:
        """The request's victims' entries cleared, with its new entries, after its experts (the
        card waits on their tags, not these: it reads a victim's entry only in a later request,
        after served)."""
        # Safe late because of moe.moe_ffn's contract (its steps 3-5): the card reads the
        # present flag and the entry of the ids it posted only, and a victim is never one of
        # them (_slot / _pool_slot pick outside ids); its next request (the only one that could
        # name a victim) is posted after its fence, WAITW served >= seq, and served is written
        # after these on the same in-order queue. A new entry is written after its expert's
        # data, so it never says present before the expert has landed. A card program that
        # read any other entry during a request would need the clear before the victim's slot
        # is written (clear_late False).
        for g in self._victims:
            self._dir(g, 0)
        self._victims.clear()
        self._dir_flush()

    def _dir(self, g: int, slot: int) -> None:
        """Expert g's entry in the host's copy: {slot, 1.0}, or {0, 0.0} for slot 0."""
        self.dirv[g] = (slot, np.float32(1.0).view(np.uint32)) if slot else (0, 0)
        self._dirty.add(g)

    def _dir_flush(self) -> None:
        """The changed entries to the card: each changed 64-byte beat of the directory, or, from
        three beats of one layer, the layer's span of them as one write (BoardDram: one DMA
        call per channel)."""
        if not self._dirty:
            return
        E, per = self.L.E, LINE // 8
        by: dict = {}
        for g in self._dirty:
            by.setdefault(g // E, set()).add(g // per)
        self._dirty.clear()
        for beats in by.values():
            spans = [(b, b + 1) for b in sorted(beats)] if len(beats) <= 2 else \
                [(min(beats), max(beats) + 1)]
            for b0, b1 in spans:
                self.mem.write(self.L.entry(b0 * per), self.dirv[b0 * per:b1 * per])

    def hint(self, ids) -> None:
        """A hint: the k global ids the layer's router picks on its input (docs/offload.md 12).
        With "lfu" each one not in a slot gets one now (a free slot, or the victim's: its entry
        cleared) and waits in `pending` for the link's idle time (step); no use counted, so
        a hinted expert no request names is the next victim. hint_n / hint_top cap them: of the
        first hint_top ids the first hint_n not in a slot (docs/offload.md 12.7). The slot is
        taken here, not when the first part goes: the victim's entry must be cleared before
        `served` covers the hint, since the card may post the layer's request (and read the
        victim's entry) as soon as it sees served. "lru" ignores hints (section 5.4: an LRU
        victim of a wrong hint is a recent expert). A hint for ahead_layer's layer
        (the card's router on the layer before it, in a layer-major prefill) adds to the end
        of its queue instead (docs/offload.md 13.4)."""
        j = self._layer(ids)
        self._unstage()
        self.hints += 1
        if j == self._qlayer:               # ahead_layer's layer: to the end of its queue, those
            lru, q = self.lru[j], self.queue    # in no slot (on their way: in one) nor queued;
            for g in ids:                   # no slot taken now (each when its first part goes)
                if g not in lru and g not in q:
                    q[g] = None
                    self.hinted_ahead += 1
            return
        if self.policy != "lfu" or self.pooled:
            return
        lru = self.lru[j]
        want = [g for g in (ids[:self.hint_top] if self.hint_top else ids) if g not in lru]
        for g in want[:self.hint_n] if self.hint_n else want:
            lru[g] = self._slot(j, ids)
            self.pending[g] = 0
        self._clear()

    def step(self) -> None:
        """The next part of the oldest hinted expert on its way (the link is idle: no request
        since the last served); its tag with the last part, then its entry."""
        g, a = next(iter(self.pending.items()))
        n = self.L.slot_bytes
        if self._send(g, min(a + self.part, n), idle=True) == n:
            self.prefetched += 1
            if g in self._ahead:            # (ahead_layer's: landed, now an expert in a slot)
                self._ahead.discard(g)
                self.landed += 1
                if self.pooled:
                    self.order[g] = None

    def _send(self, g: int, b: int, entry: bool = True, then=None, idle: bool = False) -> int:
        """Hinted expert g's bytes from where it stands to b; its tag with them when that is
        the end, then (entry) its entry. idle (step's part): its bytes as _stage_next read
        them, if it did, in one DMA call per channel. Returns b."""
        slot, a = self.lru[g // self.L.E][g], self.pending[g]
        n, part = self.L.slot_bytes, self._part_of(g, a, b)
        st, kw = self._staged, {}
        self._staged = None
        if st is not None and st[0] != (g, slot, a, b):
            self.mem.unstage(st[1])
            st = None
        if idle and st is not None:         # read already: one DMA call per channel
            kw = dict(cut=False, staged=st[1])
        self._write(slot, a, part, g if b == n else None, then, **kw)
        self.bytes += b - a
        if b < n:
            self.pending[g] = b
            return b
        del self.pending[g]
        if entry:
            self._dir(g, slot)
            self._dir_flush()
        return b

    def _part_of(self, g: int, a: int, b: int):
        """Expert g's bytes a..b from the pool (a split record's part: read when written)."""
        data = self.pool(g)
        n = len(data)
        if n != self.L.slot_bytes:
            raise ValueError(f"expert {g}: {n} bytes, slots hold {self.L.slot_bytes}")
        if a == 0 and b == n:
            return data
        if isinstance(data, SplitRecord):
            return data.part(a, b)
        return (np.ascontiguousarray(data).view(np.uint8).reshape(-1) if isinstance(
            data, np.ndarray) else np.frombuffer(bytes(data), np.uint8))[a:b]

    def _stage_next(self) -> None:
        """Read-ahead (docs/offload.md 13.10): the part the next idle poll would send, read into
        a staging pair now (the memory's stage), while this poll's DMA runs (a part's, or a
        request's experts and served); the next queued expert of ahead_layer takes its slot for
        it now, as it would then. A request first drops it (_unstage): the read is lost, not
        the link's time."""
        stage = getattr(self.mem, "stage", None)
        if stage is None or not self.read_ahead or self._staged is not None:
            return
        if not self.pending and not self._next_ahead():
            return
        g, a = next(iter(self.pending.items()))
        n, slot = self.L.slot_bytes, self.lru[g // self.L.E][g]
        b = min(a + self.part, n)
        t = (slot + self.L.tag, _tag_beat(g + 1)) if b == n else None
        h = stage(slot + a, self._part_of(g, a, b), t)
        if h is not None:
            self._staged = ((g, slot, a, b), h)

    def _clock(self):
        """The memory's run_clock (the running program's start and time with no waits), its
        waits so far reset at a new run's start; None without one."""
        rc = getattr(self.mem, "run_clock", None)
        c = rc() if rc is not None else None
        if c is not None and c[0] != self._run:
            self._run, self._waits = c[0], 0.0
        return c

    def _wait(self, s: float) -> None:
        """A request of the running program's with misses took s from seen to served."""
        if self._clock() is not None:
            self._waits += s

    def _hold(self) -> bool:
        """halt_aware (docs/offload.md 13.10): no idle part now if the running program is
        expected to end before one would be done (its halt would be seen after the part): its
        start, its time with no waits and its own waits so far. A run past that end by
        HOLD_LATE gets parts again (the estimate was short), as does one with no estimate.
        Requests are served as always."""
        c = self._clock() if self.halt_aware else None
        if c is None:
            return False
        left = c[0] + c[1] + self._waits - time.perf_counter()
        need = self.part_s if self.part_s is not None else PART_S0 + self.part / PART_GBS
        if -HOLD_LATE < left < need:
            self.holds += 1
            return True
        return False

    def _unstage(self) -> None:
        """A staged part dropped (the server changes its slots or serves a request first)."""
        if self._staged is not None:
            self.mem.unstage(self._staged[1])
            self._staged = None

    def _write(self, slot: int, at: int, data, tag: int | None = None, then=None,
               **kw) -> None:
        """data to slot + at; with tag (expert g's), the slot's tag word {g + 1} after it, in
        the same DMA's last beat (memories without write_slot: a write after it). then():
        called once data's first part is on its way (the request's answer: write_slot); kw:
        write_slot's cut and staged (an idle part)."""
        t = None if tag is None else (slot + self.L.tag, _tag_beat(tag + 1))
        w = getattr(self.mem, "write_slot", None)
        if w is not None:
            w(slot + at, data, t, then, **kw)
        else:
            self.mem.write(slot + at, data)
            if t is not None:
                self.mem.write(*t)
            if then is not None:
                then()
        if t is not None:
            self.armed.add(slot)

    def _fetch(self, g: int, slot: int, then=None) -> None:
        """A missing expert into its slot, with its tag (the card waits on that)."""
        data = self.pool(g)
        if len(data) != self.L.slot_bytes:
            raise ValueError(f"expert {g}: {len(data)} bytes, slots hold {self.L.slot_bytes}")
        self._write(slot, 0, data, g, then)
        self.bytes += len(data)

    def _reuse(self, slot: int) -> int:
        """A slot about to take another expert: its tag cleared first if the card may not have
        zeroed it (a hinted expert no request used)."""
        if slot in self.armed:
            self.armed.discard(slot)
            self.mem.write(slot + self.L.tag, _tag_beat(0))
        return slot

    def _slot(self, j: int, ids) -> int:
        """A slot of layer j for one of the request's experts: a free one, or the victim's (its
        entry cleared at the request's end, _clear; a hinted expert still on its way is
        dropped)."""
        if self.free[j]:
            return self._reuse(self.free[j].pop(0))
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
            self._victims.append(victim)
            if not self.clear_late:
                self._clear()
        return self._reuse(lru.pop(victim))

    def _pool_slot(self, j: int, ids) -> int:
        """Pooled: a free slot of layer j, else of any layer, else the slot of the least
        recently used expert of any layer the request does not name (its entry cleared)."""
        for fr in [self.free[j]] + self.free:
            if fr:
                return self._reuse(fr.pop(0))
        victim = next((v for v in self.order if v not in ids), None)
        if victim is None:
            raise RuntimeError(f"{len(self.order)} slots for a request of {len(ids)}")
        del self.order[victim]
        self._victims.append(victim)
        if not self.clear_late:
            self._clear()
        return self._reuse(self.lru[victim // self.L.E].pop(victim))

    def settle(self) -> None:
        """Serve the card's last request if it is still unserved, before the host changes its
        slots outside a request (begin_prefill, end_prefill). A request whose experts were all
        present does not wait for the host, so its run can halt before the host has seen it.
        Served later against changed slots, it would name misses: an answer line and tags
        the card never reads or zeroes, which the next request would take for its own. Served
        now, it is all hits, as the card ran it."""
        seq = int(np.frombuffer(bytes(self.mem.read(self.L.mbox, 4)), np.float32)[0])
        if seq != self.seq:
            self.poll()

    def begin_prefill(self, ahead: bool = False, part: int | None = None) -> None:
        """A layer-major prefill starts (docs/offload.md 13): every slot serves the layer its
        requests name. Hints still on their way are dropped (their slots free; their entries
        read 0 already). ahead: ahead_layer's calls act during it (False: they only settle);
        part: the bytes an idle poll sends of an expert while it lasts (RUN blocks; default the
        server's part)."""
        if part is not None and (part <= 0 or part % RUN):
            raise ValueError(f"an idle poll's part of {part} bytes (RUN blocks)")
        self.settle()
        self._drop_ahead()
        self._ahead_off = not ahead
        if part is not None:
            self._part, self.part = self.part, part
        for g in list(self.pending):
            del self.pending[g]
            j = g // self.L.E
            self.free[j].append(self.lru[j].pop(g))
            self.dropped += 1
        self.order = OrderedDict((g, None) for lru in self.lru for g in lru)
        self.pooled = True

    def end_prefill(self, restore: str = "lazy") -> None:
        """The prefill ends: each layer gets its own number of slots back. A layer keeps its
        experts of most decayed use (the prompt's requests) up to it; the others leave, their
        entries cleared. "lazy": decode's misses fill the free slots (docs/offload.md 13: a
        restore of every layer's set costs more than the misses it saves); "eager": each
        layer's experts of most use not in a slot are loaded now."""
        if restore not in ("lazy", "eager"):
            raise ValueError(f"restore {restore!r}")
        self.settle()
        self._drop_ahead()
        self._ahead_off = False
        if self._part is not None:
            self.part, self._part = self._part, None
        self.armed -= self._used            # (the prefill's runs are done: the card zeroed
        self._used = set()                  # the last request's tags)
        spare = [a for fr in self.free for a in fr]
        for fr in self.free:
            fr.clear()
        for j, (_, n) in enumerate(self.L.slots):
            lru, use = self.lru[j], self.use[j]
            if len(lru) > n:
                keep = set(sorted(lru, key=lambda g: use.get(g, -math.inf), reverse=True)[:n])
                for g in [g for g in lru if g not in keep]:
                    self._dir(g, 0)
                    spare.append(lru.pop(g))
        self._dir_flush()
        for j, (_, n) in enumerate(self.L.slots):
            while len(self.lru[j]) + len(self.free[j]) < n:
                self.free[j].append(spare.pop())
        self.pooled = False
        self.order.clear()
        if restore == "eager":
            for j in range(self.L.layers):
                use = self.use[j]
                for g in sorted(use, key=lambda g: use[g], reverse=True):
                    if not self.free[j]:
                        break
                    if g not in self.lru[j]:
                        self._insert(j, g, self.free[j].pop(0))
        self._flush()

    def ahead_layer(self, j: int, ids) -> None:
        """A layer-major prefill's next layer (docs/offload.md 13): MoE layer j's experts `ids`
        (global, most wanted first) go to slots on the link's idle time, before the layer's runs
        ask for them. Called between runs, and the runs until the next call ask only for layers
        j - 1 and j (prefill_layers: before layer j - 1's first run of a chunk; the first call,
        before layer j's own). It replaces what the last call queued: those not in a slot yet
        leave the queue, and one on its way is dropped (its slot freed; its tag and entry were
        never written). The card's last request is served first (settle) and its tags, zeroed
        by the halted run, are no longer armed. Each queued expert takes a slot when its first
        part goes (an idle poll): a free one, else (pooled) the oldest expert of a layer outside
        j - 1 and j, so nothing the running layer may read changes under it. A request that
        names a queued expert takes it as a miss; one on its way, its rest (as a hint's)."""
        self.settle()
        self.armed -= self._used
        self._used = set()
        self._drop_ahead()
        if self._ahead_off:
            return
        E, L = self.L.E, self.L.layers
        if not 0 <= j < L or any(g // E != j for g in ids):
            raise ValueError(f"ahead_layer({j}): ids {list(ids)[:4]}... not all of layer {j}")
        self.aheads += 1
        self._spare = frozenset((j, (j - 1) % L))
        lru = self.lru[j]
        self.queue = OrderedDict((int(g), None) for g in ids if g not in lru)
        self._qlayer = j

    def _drop_ahead(self) -> None:
        """ahead_layer's queue emptied, and its experts on their way dropped: their slots free
        (no tag or entry was written for them)."""
        self._unstage()
        self.queue.clear()
        self._qlayer = None
        for g in [g for g in self.pending if g in self._ahead]:
            del self.pending[g]
            j = g // self.L.E
            self.free[j].append(self.lru[j].pop(g))
            self.dropped += 1
        self._ahead.clear()

    def _next_ahead(self) -> bool:
        """The next queued expert gets a slot and is on its way (pending); False with none to
        start, or no slot it may take."""
        E = self.L.E
        while self.queue:
            g = next(iter(self.queue))
            del self.queue[g]
            j = g // E
            if g in self.lru[j]:                # (a request took it meanwhile)
                continue
            spare = _Layers(self._spare, E)
            try:
                slot = (self._pool_slot(j, spare) if self.pooled else
                        self._reuse(self.free[j].pop(0)) if self.free[j] else None)
            except RuntimeError:                # no victim outside the running layers
                slot = None
            if slot is None:
                self.queue.clear()
                return False
            self._clear()                       # (a victim's entry before the expert's bytes)
            self.lru[j][g] = slot
            self.pending[g] = 0
            self._ahead.add(g)
            return True
        return False

    def _insert(self, j: int, g: int, slot: int) -> None:
        """Expert g into a slot outside a request (load's warm experts, an eager restore): its
        bytes, then its entry; no tag (no request waits on it: the card finds it present)."""
        data = self.pool(g)
        if len(data) != self.L.slot_bytes:
            raise ValueError(f"expert {g}: {len(data)} bytes, slots hold {self.L.slot_bytes}")
        self._write(slot, 0, data)
        self._dir(g, slot)
        self._dir_flush()
        self.lru[j][g] = slot
        self.bytes += len(data)


class _Layers:
    """`g in` it: expert g is of one of these layers (ahead_layer's victims stay outside)."""

    def __init__(self, layers, E: int):
        self.layers, self.E = layers, E

    def __contains__(self, g) -> bool:
        return g // self.E in self.layers

    def __len__(self) -> int:
        return len(self.layers) * self.E


def _tag_beat(v: int) -> np.ndarray:
    """A slot's tag beat: the tag word v (u32), then zeros."""
    b = np.zeros(LINE // 4, np.uint32)
    b[0] = v
    return b


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
        self.last = "row"                   # (PollPacer's key: every request one kind)

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


class PollPacer:
    """BoardBackend.host for its servers (an ExpertServer, a RowServer) that sleeps through the
    quiet part of each gap (docs/offload.md 10.10). After a request is served the card computes
    before it posts the next one (the layer's last experts, the next layer's attention and
    router): no sooner, so far, than the shortest of the last `keep` gaps that followed a request
    of the same kind (server, layer). poll sleeps `share` of that (at most `cap`, less the
    sleep's own lateness `late`), then polls at the loop's pace, so a request is seen as soon as
    with the loop spinning, with fewer of its card reads (a DMA call each, every ~33 us). A gap
    is measured from served to the poll that saw the next request: a sleep that ran past a post
    makes the next one shorter. It never sleeps while hinted or layer-ahead experts wait for idle
    polls (their parts go then)."""

    def __init__(self, servers, share: float = 0.75, keep: int = 8, cap: float = 5e-3,
                 late: float = 100e-6, sleep=time.sleep, clock=time.perf_counter):
        if not 0 < share < 1:
            raise ValueError(f"share {share}: of the shortest recent gap, in (0, 1)")
        self.servers, self.share, self.keep, self.cap, self.late = (list(servers), share, keep,
                                                                   cap, late)
        self.sleep, self.clock = sleep, clock
        self.gaps: dict = {}            # a request's kind -> the gaps after its last ones (s)
        self.key, self.t_served, self.until = None, 0.0, 0.0
        self.slept, self.sleeps = 0.0, 0

    def poll(self) -> int:
        t = self.clock()
        if t < self.until:              # (a poll that sleeps polls nothing: the next one does)
            d, self.until = self.until - t, 0.0
            self.sleep(d)
            self.slept += d
            self.sleeps += 1
            return 0
        for i, srv in enumerate(self.servers):
            if not srv.poll():
                continue
            kind = getattr(srv, "last", None)
            if kind is None:            # a hinted expert's part, on idle time
                return 1
            if self.key is not None:
                g = self.gaps.setdefault(self.key, [])
                g.append(t - self.t_served)
                del g[:-self.keep]
            self.key, self.t_served = (i, kind), self.clock()
            g = self.gaps.get(self.key)
            if g and not any(getattr(x, "pending", None) or getattr(x, "queue", None)
                             for x in self.servers):
                d = min(self.share * min(g), self.cap) - self.late
                if d > self.late:
                    self.until = self.t_served + d
            return 1
        return 0


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
      split-format pool file (SplitRecord) is read straight into the runs (one preadv of an
      iovec array, preadv_iov, where the record has readiov), and when nothing is in flight
      (a request's first miss) in `pieces` parts, the first `lead` of it (None: equal parts),
      each DMAed as soon as it is read: the link starts after a fifth of the expert is read,
      not half (docs/offload.md 10.8); an idle part read ahead (stage, beside the DMA before
      it) goes whole (docs/offload.md 13.10);
    - the host's own words (served and the directory: the card only reads them) are kept in a
      shadow and written with no read first: a write within one 64-byte beat (an entry,
      served) as that beat alone, one DMA call on its channel; a longer one as whole 128-byte
      blocks;
    - one worker thread makes every DMA call in order, while the server stages the next
      expert: an expert's data lands before its directory entry, every entry before served
      (one queue, one h2c stream). flush() waits for the queue and raises a worker's error;
      ExpertServer calls it before poll returns, so no DMA of the server's is in flight
      while anything else uses the card.
    `calls`, when set to a list (moe_card --hint-trace): each DMA call of the worker's appended,
    (perf_counter at its start, at its end, bytes)."""

    def __init__(self, backend, layout: Layout, depth: int = 3, pieces: int = 2,
                 lead: float | None = 0.2):
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
        self._iovs: dict = {}                 # bytes -> per staging pair: its iovec template
        self._par: dict = {}                  # (address, blocks) -> each block's parity
        self._free: queue.Queue = queue.Queue()
        self._q: queue.Queue = queue.Queue()
        self._err: BaseException | None = None
        self._thread: threading.Thread | None = None
        self.dma_s, self.dma_bytes = 0.0, 0     # the worker's slot DMA: seconds and bytes
        self.direct = 0                         # experts read from the file into their runs
        self.staged = 0                         # of them, idle parts read ahead (stage)
        self.wait_s = 0.0                       # the server's waits for a free staging pair
        self.pieces, self._lead = pieces, True  # _lead: no DMA in flight since the last flush
        self._held: set = set()                 # staging pairs holding a staged part (stage)
        # the running program's start and time with no waits (BoardBackend.run_clock), or None
        self.run_clock = getattr(backend, "run_clock", None)
        if lead is not None and not 0 < lead < 1:
            raise ValueError(f"lead {lead}: the first part's share of the expert, in (0, 1)")
        self.lead = lead
        self.calls: list | None = None

    # ---- the worker
    def _work(self) -> None:
        import contextlib
        lock = getattr(self.board.t, "_dma", None) or contextlib.nullcontext()
        while True:
            fn, slot = self._q.get()
            with lock:                          # (held while calls are queued: each call's
                while True:                     # own is then a reentry, no flock)
                    try:
                        if self._err is None:
                            fn()
                    except BaseException as e:      # noqa: BLE001 (raised by flush)
                        self._err = e
                    finally:
                        if slot is not None:
                            self._free.put(slot)
                        self._q.task_done()
                    try:
                        fn, slot = self._q.get_nowait()
                    except queue.Empty:
                        break

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
        elif len(b) == self.blk // 2 and addr % len(b) == 0:   # a whole beat (a slot's tag):
            out = b.copy()                                      # no read, one DMA call
            self._put(lambda: self._beat(addr, out))
        else:                                                   # the card's words: in order
            self.flush()
            self.board.write(addr, b)

    def _staging(self, nbytes: int) -> None:
        """depth pairs of page-aligned buffers for an expert's two channel runs (a shorter
        write, a hint's part, uses their first bytes)."""
        if self._bufs is not None and len(self._bufs[0][0]) >= nbytes // 2:
            return
        if self._held:
            raise RuntimeError(f"staging pairs for {nbytes} bytes with a staged part held")
        self.flush()
        n = nbytes // self.blk
        self._bufs = [[_page_buf(nbytes // 2) for _ in (0, 1)] for _ in range(self.depth)]
        self._base = 2 * np.arange(n, dtype=np.intp)
        self._i1, self._idx, self._pieces, self._iovs = np.empty(n, np.intp), {}, {}, {}
        while not self._free.empty():
            self._free.get()
        for i in range(self.depth):
            self._free.put(i)

    def _pair(self) -> int:
        """A staging pair the worker is done with (waiting for it: wait_s)."""
        t0 = time.perf_counter()
        i = self._free.get()
        self.wait_s += time.perf_counter() - t0
        return i

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

    def _iov(self, nbytes: int) -> list:
        """Per staging pair, an nbytes record's iovec template [block, run, (address, length)]
        (uint64): the two runs' RUN / 2-byte pieces of each block, in file order where the
        block lands on even parity (BoardDram.write_slot swaps the two where it is odd)."""
        if nbytes not in self._iovs:
            h, nb = RUN // 2, -(-nbytes // RUN)
            j = np.arange(nb, dtype=np.uint64) * np.uint64(h)
            ln = np.minimum(np.uint64(h), np.uint64(nbytes // 2) - j)
            ts = []
            for b0, b1 in self._bufs:
                t = np.empty((nb, 2, 2), np.uint64)
                t[:, 0, 0] = np.uint64(b0.ctypes.data) + j
                t[:, 1, 0] = np.uint64(b1.ctypes.data) + j
                t[:, :, 1] = ln[:, None]
                ts.append(t)
            self._iovs[nbytes] = ts
        return self._iovs[nbytes]

    def _cuts(self, nb: int, lead: bool) -> list:
        """The block ranges an expert of nb blocks is read and DMAed in: one, or with nothing
        in flight (`lead`) `pieces`, the first `self.lead` of it and the rest in equal parts
        (equal parts all where self.lead is None)."""
        if not lead or self.pieces <= 1 or nb <= 1:
            return [(0, nb)]
        if self.lead is None:
            c = [nb * p // self.pieces for p in range(self.pieces + 1)]
        else:
            c1, r = max(1, min(nb - 1, round(nb * self.lead))), self.pieces - 1
            c = [0, c1] + [c1 + (nb - c1) * p // r for p in range(1, r + 1)]
        c = sorted(set(c))
        return list(zip(c[:-1], c[1:]))

    def _tag_channel(self, at: int) -> int:
        """The channel that holds the beat at `at` (64-byte aligned)."""
        m, c = at // self.blk, at // (self.blk // 2) % 2
        return c ^ (m.bit_count() & 1) if self.board.chash else c

    def _in_place(self, addr: int, data, n: int) -> bool:
        """data is read from its file straight into the runs (a split record on a CHASH card,
        at a RUN-aligned address, of whole chunks)."""
        return (isinstance(data, SplitRecord) and self.board.chash and addr % RUN == 0
                and n % self.blk == 0)

    def _runs(self, addr: int, data, n: int, tag):
        """A free staging pair i for a split record's n bytes to addr (and the slot's tag beat
        in the chunk after them): (i, its blocks, the runs' end, the channels' order, read(j0,
        j1): blocks j0..j1 from the file into the pair's runs)."""
        self._staging(n + (self.blk if tag else 0))     # the file's runs, read in place
        i = self._pair()
        nb, bufs = -(-n // RUN), self._bufs[i]
        order, e = (0, 1), n // 2
        if tag is not None:                     # the tag chunk after the runs: its tag beat on
            ct = self._tag_channel(tag[0])      # its channel, a zero beat on the other
            tb = np.ascontiguousarray(tag[1]).view(np.uint8).reshape(-1)
            bufs[ct][e:e + len(tb)] = tb
            bufs[1 - ct][e:e + len(tb)] = 0
            order, e = (1 - ct, ct), e + len(tb)
        par = self._par.get((addr, nb))
        if par is None:                         # (CHASH: where each block lands)
            par = self._par[addr, nb] = _parity(addr // RUN + np.arange(nb)).astype(bool)
        if data.readiov is not None:            # an iovec array: no buffer objects (each
            t = self._iov(n)[i]                 # part's built as it is read)

            def read(j0, j1):
                p = par[j0:j1, None, None]
                data.readiov(np.where(p, t[j0:j1, ::-1], t[j0:j1]).reshape(-1, 2), j0 * RUN)
        else:
            pv = self._split_pieces(n)[i][np.arange(nb), par.astype(np.intp)]

            def read(j0, j1):
                data.readv(pv[j0:j1].reshape(-1).tolist(), j0 * RUN)
        return i, nb, e, order, read

    def stage(self, addr: int, data, tag=None):
        """Read-ahead (ExpertServer's next idle part, docs/offload.md 13.10): data's bytes read
        into a free staging pair now, while the DMA ahead of it runs, for write_slot(addr, data,
        tag, staged=it) to queue as one DMA call per channel later; None where write_slot
        would not read it in place. The pair stays the server's until then, or unstage."""
        n = len(data)
        if not self._in_place(addr, data, n) or (tag is not None and tag[0] != addr + n):
            return None
        self._staging(self.L.slot_bytes + self.blk)     # (no larger write after: the pairs
        i, nb, e, order, read = self._runs(addr, data, n, tag)  # are never reallocated
        read(0, nb)                                     # under a staged one)
        self._held.add(i)
        return (addr, n, None if tag is None else tag[0], i, e, order)

    def unstage(self, staged) -> None:
        """A staged part not sent (a request came first): its pair back."""
        self._held.discard(staged[3])
        self._free.put(staged[3])

    def write_slot(self, addr: int, data, tag=None, then=None, cut: bool = True,
                   staged=None) -> None:
        """An expert's bytes (or a part's) to addr; tag: (address, its 64-byte beat), the slot's
        tag beat, which goes in the same DMA, last: right after the bytes (the slot's tag chunk
        starts there), the other channel's call first and the tag's channel's after it, the tag
        its last beat (docs/offload.md 10.11); elsewhere, a call of its own after them. then():
        called once the first part's DMA is queued (ExpertServer: the request's answer, so the
        link starts on the expert's lead). cut False: one DMA call per channel even with nothing
        in flight (an idle part: no request waits on its first bytes); staged: stage's read of
        these bytes, queued as they are."""
        from .board import swapped
        n = data.nbytes if isinstance(data, np.ndarray) else len(data)
        if staged is not None:
            if staged[:3] != (addr, n, None if tag is None else tag[0]):
                raise ValueError(f"a staged part of {staged[1]} bytes to {staged[0]:#x} written "
                                 f"as {n} bytes to {addr:#x}")
            _, _, _, i, e, order = staged
            self._held.discard(i)
            bufs = self._bufs[i]
            self._lead = False
            self._put(lambda: self._dma(addr // 2, bufs, 0, e, order), i)
            if then is not None:
                then()
            self.direct += 1
            self.staged += 1
            return
        if tag is not None and (tag[0] != addr + n or n % self.blk):
            self.write_slot(addr, data, None, then, cut)
            self.write(*tag)
            return
        if self._in_place(addr, data, n):
            i, nb, e, order, read = self._runs(addr, data, n, tag)
            h, bufs = RUN // 2, self._bufs[i]
            # nothing in flight: the link waits for this read, so it goes in parts, a short
            # one first (each part one DMA call per channel more)
            lead = cut and (self._lead or not self._q.unfinished_tasks)
            self._lead = False
            for j0, j1 in self._cuts(nb, lead):
                read(j0, j1)
                last = j1 == nb
                self._put(lambda a=j0 * h, b=e if last else j1 * h, o=order if last else (0, 1):
                          self._dma(addr // 2, bufs, a, b, o), i if last else None)
                if then is not None:
                    then()
                    then = None
            self.direct += 1
            return
        src = (np.ascontiguousarray(data).view(np.uint8).reshape(-1)
               if isinstance(data, np.ndarray) else np.frombuffer(bytes(data), np.uint8))
        if addr % self.blk or len(src) % self.blk:
            self.write(addr, src)
            if then is not None:
                then()
            if tag is not None:
                self.write(*tag)
            return
        order = (0, 1)
        if tag is not None:                     # the tag chunk after the bytes, in their DMA
            tb = np.ascontiguousarray(tag[1]).view(np.uint8).reshape(-1)
            src = np.concatenate([src, tb, np.zeros(self.blk - len(tb), np.uint8)])
            ct = self._tag_channel(tag[0])
            order = (1 - ct, ct)
        n = len(src) // self.blk
        self._staging(len(src))
        i0 = self._idx.get((addr, n))
        if i0 is None:                          # once per slot: CHASH's swaps at its address
            i0 = self._idx[addr, n] = (self._base[:n] + swapped(addr, n) if self.board.chash
                                       else self._base[:n])
        i1 = self._i1[:n]
        np.bitwise_xor(i0, 1, out=i1)           # channel 1 takes each chunk's other beat
        i = self._pair()                        # a staging pair the worker is done with
        bufs, beats, h = self._bufs[i], src.view("V64"), n * self.blk // 2
        np.take(beats, i0, out=bufs[0][:h].view("V64"))       # channel 0's run
        np.take(beats, i1, out=bufs[1][:h].view("V64"))       # channel 1's
        self._put(lambda: self._dma(addr // 2, bufs, 0, h, order), i)
        if then is not None:
            then()

    def _beat(self, at: int, data: np.ndarray) -> None:
        """One 64-byte beat of the host's words (at: 64-byte aligned) to the channel that holds
        it (CHASH: beat b of chunk m on channel b ^ parity(m))."""
        t0 = time.perf_counter()
        m, c = at // self.blk, at // (self.blk // 2) % 2
        if self.board.chash:
            c ^= m.bit_count() & 1
        self.board.t.mem_write(c, m * (self.blk // 2), data)
        if self.calls is not None:
            self.calls.append((t0, time.perf_counter(), len(data)))

    def _blocks(self, at: int, data: np.ndarray) -> None:
        """Whole chunks of the host's words (at, len: multiples of 128) to the two channels, as
        Board.write does (CHASH's swaps), without its general path's copies."""
        from .board import BEAT, swapped
        t0 = time.perf_counter()
        v = data.reshape(-1, 2, BEAT)
        sw = swapped(at, len(v)) if self.board.chash else np.zeros(len(v), bool)
        for c in (0, 1):
            run = np.where(sw[:, None], v[:, 1 - c], v[:, c])
            self.board.t.mem_write(c, at // 2, run.reshape(-1))
        if self.calls is not None:
            self.calls.append((t0, time.perf_counter(), len(data)))

    def _dma(self, off: int, bufs, a: int = 0, b: int | None = None, order=(0, 1)) -> None:
        """bufs[c][a:b] to channel c's run from channel offset off + a, the channels in
        `order` (a tag's channel last: each call ends before the next starts)."""
        t0 = time.perf_counter()
        for c in order:
            self.board.t.mem_write(c, off + a, bufs[c][a:b])
        t1 = time.perf_counter()
        self.dma_s += t1 - t0
        self.dma_bytes += 2 * len(bufs[0][a:b])
        if self.calls is not None:
            self.calls.append((t0, t1, 2 * len(bufs[0][a:b])))

    def read(self, addr: int, n: int) -> bytes:
        """n bytes from the card. Within one 64-byte beat (a poll's seq, a request's row) that
        beat alone, one DMA call on its channel (Board.read reads the whole 128-byte chunk, one
        call per channel: a poll's read took 90 us on the card). Raises with the worker's DMA
        queued or in flight: the server's reads never meet its writes (ExpertServer flushes
        before poll returns), as the card needs (a card->host call beside a host->card one slips
        the card's writes, board._DmaLock)."""
        if self._q.unfinished_tasks:
            raise RuntimeError(f"BoardDram.read with {self._q.unfinished_tasks} of the worker's "
                               f"DMA calls unfinished: flush first")
        h = self.blk // 2
        if addr % h + n <= h:
            m, c = addr // self.blk, addr // h % 2
            if self.board.chash:
                c ^= m.bit_count() & 1
            beat = np.asarray(self.board.t.mem_read(c, m * h, h)).view(np.uint8)
            return beat[addr % h:addr % h + n].tobytes()
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
