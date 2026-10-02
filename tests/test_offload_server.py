"""opentpu/host/offload.py: the expert server against a fake card that posts requests the way
docs/offload.md section 5.2 describes (mailbox ring, seq, the directory)."""
import numpy as np
import pytest

from opentpu.host.offload import LINE, ExpertServer, Layout, SimDram

E, K, SLOT = 8, 2, 128


def _pool(g):
    return np.full(SLOT, g + 1, np.uint8).tobytes()


def _setup(slots=(3, 3), warm=()):
    lay = Layout.build(4096, E, K, slots, SLOT)
    mem = SimDram(np.zeros(lay.end + 4096, np.uint8))
    srv = ExpertServer(mem, lay, _pool)
    srv.load(warm)
    return lay, mem, srv


def _post(mem, lay, seq, ids):
    """The card's side (after its fence): the ids to the row, then seq."""
    mem.write(lay.row, np.array(ids + [0] * (LINE // 4 - len(ids)), np.float32))
    mem.write(lay.mbox, np.float32(seq).tobytes())


def _entry(mem, lay, g):
    w = np.frombuffer(mem.read(lay.entry(g), 8), np.uint32)
    return int(w[0]), float(w[1:].view(np.float32)[0])


def _served(mem, lay):
    return float(np.frombuffer(mem.read(lay.served, 4), np.float32)[0])


def test_layout_is_aligned_and_disjoint():
    lay = Layout.build(4096, E, K, (3, 5), SLOT)
    words = [lay.mbox, lay.served, lay.dir] + [a for a, _ in lay.slots]
    assert all(a % LINE == 0 for a in words)
    assert lay.served >= lay.row + LINE
    assert lay.slots[0][0] >= lay.dir + 8 * E * 2
    assert lay.slots[1][0] == lay.slots[0][0] + 3 * SLOT
    with pytest.raises(ValueError):
        Layout.build(4100, E, K, (3,), SLOT)


def test_warm_start_fills_slots_and_the_directory():
    lay, mem, srv = _setup(warm=[0, 1, 2, 3, E + 5])
    for g in (0, 1, 2, E + 5):
        a, p = _entry(mem, lay, g)
        assert p == 1.0 and mem.read(a, SLOT) == _pool(g)
    assert _entry(mem, lay, 3) == (0, 0.0)          # layer 0 has 3 slots
    assert _served(mem, lay) == 0.0
    assert srv.poll() == 0                          # nothing posted


def test_misses_evict_the_least_recent_not_requested():
    lay, mem, srv = _setup(warm=[0, 1, 2])          # layer 0 LRU order: 2, 1, 0 (0 newest)
    _post(mem, lay, 1, [2, 5])                      # 2 hits; 5 evicts the oldest other: 1
    assert srv.poll() == 1
    assert _served(mem, lay) == 1.0
    assert _entry(mem, lay, 1) == (0, 0.0)
    a, p = _entry(mem, lay, 5)
    assert p == 1.0 and mem.read(a, SLOT) == _pool(5)
    assert (srv.hits, srv.misses, srv.bytes) == (1, 1, SLOT)
    # a request never evicts its own ids: 0 and 6 with {0, 2, 5} resident evicts 2, not 0
    _post(mem, lay, 2, [0, 6])
    srv.poll()
    assert _entry(mem, lay, 2) == (0, 0.0)
    assert _entry(mem, lay, 0)[1] == 1.0 and _entry(mem, lay, 6)[1] == 1.0


def test_ahead_sees_a_requests_misses_before_the_first_is_staged():
    """ExpertServer.ahead (moe_card --willneed: PoolFile.willneed) is called with a request's
    missing ids, before the first of them is read from the pool; a request of hits only does
    not call it."""
    lay, mem, srv = _setup(warm=[0, 1, 2])
    seen, reads = [], []
    pool = srv.pool
    srv.pool = lambda g: (reads.append(g), pool(g))[1]
    srv.ahead = lambda ids: seen.append((list(ids), len(reads)))
    _post(mem, lay, 1, [5, 4])
    srv.poll()
    _post(mem, lay, 2, [0, 6])                      # 0 hits (2 was a victim)
    srv.poll()
    assert seen == [([5, 4], 0), ([6], 2)] and reads == [5, 4, 6]
    _post(mem, lay, 3, [6, 0])                      # hits only
    srv.poll()
    assert len(seen) == 2


def test_one_request_at_a_time():
    lay, mem, srv = _setup()
    for s in range(1, 6):
        _post(mem, lay, s, [(s % 2) * E + s % E, (s % 2) * E + (s + 1) % E])
        assert srv.poll() == 1 and srv.poll() == 0
        assert srv.seq == s and _served(mem, lay) == float(s)
    _post(mem, lay, 7, [0, 1])                      # 6 skipped: no fence on the card
    with pytest.raises(RuntimeError, match="fence"):
        srv.poll()


def test_bad_requests():
    lay, mem, srv = _setup(slots=(1, 3))
    with pytest.raises(ValueError):
        srv.serve([0, E + 1])                       # two layers
    with pytest.raises(RuntimeError):
        srv.serve([0, 1])                           # 2 ids, 1 slot


def test_split_format_is_the_slot_bytes_reordered(tmp_path):
    """split_order is a permutation of the slot's beats (a short last block included);
    to_split and a SplitRecord read back from a file (PoolFile, os.preadv) give the slot's
    bytes."""
    from opentpu.host.offload import PoolFile, SplitRecord, split_order, to_split
    for n in (4096 * 3, 128 * 37, 128):
        o = split_order(n)
        assert np.array_equal(np.sort(o), np.arange(n // 64))
        x = np.random.default_rng(n).integers(0, 256, (3, n), dtype=np.uint8)
        f = tmp_path / f"pool{n}.bin"
        f.write_bytes(b"".join(to_split(r).tobytes() for r in x))
        pf = PoolFile(f, n, split=True)
        for g in (2, 0, 1):
            r = pf.get(g)
            assert isinstance(r, SplitRecord) and len(r) == n
            assert bytes(r) == x[g].tobytes() and np.array_equal(np.asarray(r), x[g])


def test_preadv_resumes_short_reads(tmp_path, monkeypatch):
    """offload.preadv fills every buffer in order when os.preadv stops short (in the middle of
    a buffer) and over more buffers than one call takes (IOV_MAX)."""
    import os

    from opentpu.host import offload as O
    data = np.random.default_rng(0).integers(0, 256, 20000, dtype=np.uint8).tobytes()
    f = tmp_path / "f.bin"
    f.write_bytes(data)
    sizes = np.random.default_rng(1).integers(1, 9, 3000)
    bufs = [memoryview(bytearray(int(n))) for n in sizes]
    real, calls = os.preadv, []

    def stingy(fd, bs, off):                # 6 buffers and half the 7th at most
        calls.append(len(bs))
        assert len(bs) <= O.IOV_MAX
        return real(fd, list(bs[:6]) + ([bs[6][:len(bs[6]) // 2]] if len(bs) > 6 else []), off)
    monkeypatch.setattr(os, "preadv", stingy)
    fd = os.open(f, os.O_RDONLY)
    try:
        O.preadv(fd, bufs, 100)
    finally:
        os.close(fd)
    assert b"".join(bytes(b) for b in bufs) == data[100:100 + int(sizes.sum())]
    assert max(calls) == O.IOV_MAX and len(calls) > len(bufs) // 7


def test_preadv_iov_fills_as_preadv_and_resumes_short_reads(tmp_path, monkeypatch):
    """offload.preadv_iov (an iovec array of addresses and lengths through libc's preadv: no
    buffer objects, the GIL released for the read) fills the buffers as preadv does: in order,
    over more of them than one call takes (IOV_MAX), and when a call stops short in the middle
    of a buffer."""
    import ctypes
    import os

    from opentpu.host import offload as O
    if O._libc_preadv() is None:
        pytest.skip("no libc preadv")
    data = np.random.default_rng(0).integers(0, 256, 20000, dtype=np.uint8).tobytes()
    f = tmp_path / "f.bin"
    f.write_bytes(data)
    sizes = np.random.default_rng(1).integers(1, 9, 3000)
    buf = np.zeros(int(sizes.sum()), np.uint8)
    at = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    iov = np.stack([buf.ctypes.data + at, sizes], 1).astype(np.uint64)
    want = data[100:100 + int(sizes.sum())]
    real, calls = O._libc_preadv(), []

    def stingy(fd, a, g, off):              # 6 buffers and half the 7th at most
        calls.append(g)
        assert g <= O.IOV_MAX
        v = np.ctypeslib.as_array((ctypes.c_uint64 * (2 * g)).from_address(a)).reshape(g, 2)
        v = np.array(v[:7] if g > 6 else v)
        if g > 6:
            v[6, 1] //= 2
        return real(fd, v.ctypes.data, len(v), off)
    fd = os.open(f, os.O_RDONLY)
    try:
        O.preadv_iov(fd, iov, 100)
        assert buf.tobytes() == want
        buf[:] = 0
        monkeypatch.setattr(O, "_libc_preadv", lambda: stingy)
        O.preadv_iov(fd, iov, 100)
    finally:
        os.close(fd)
    assert buf.tobytes() == want
    assert max(calls) == O.IOV_MAX and len(calls) > len(sizes) // 7


def test_victims_entries_are_cleared_after_the_requests_experts():
    """A request's victims' entries are cleared after its last new entry and before served (the
    card waits on the new entries only and reads a victim's entry in a later request: after
    served), so the link starts with the first expert's bytes; clear_late False clears each one
    before its slot is written, as before docs/offload.md 10.8. Either way the victims' entries
    read empty once poll returns."""
    for late in (True, False):
        lay, mem, srv = _setup(warm=[0, 1, 2])      # layer 0 LRU order: 2, 1, 0 (0 newest)
        srv.clear_late = late
        log, w = [], mem.write

        def rec(addr, data, w=w):
            b = np.frombuffer(data if isinstance(data, bytes) else np.asarray(data).tobytes(),
                              np.uint8)
            log.append("served" if addr == lay.served else
                       ("entry" if b[4:8].any() else "clear")
                       if lay.dir <= addr < lay.dir + 8 * E * 2 else "slot")
            w(addr, data)
        _post(mem, lay, 1, [5, 6])                  # two misses: victims 2 and 1
        mem.write = rec
        assert srv.poll() == 1
        assert log == (["slot", "entry", "slot", "entry", "clear", "clear", "served"] if late
                       else ["clear", "slot", "entry", "clear", "slot", "entry", "served"])
        assert _entry(mem, lay, 2) == _entry(mem, lay, 1) == (0, 0.0)
        assert _entry(mem, lay, 5)[1] == _entry(mem, lay, 6)[1] == 1.0
        assert not srv._victims


def test_board_dram_sends_a_requests_first_miss_a_fifth_first():
    """BoardDram._cuts: with nothing in flight (a request's first miss) an expert goes in
    `pieces` parts, the first `lead` of its blocks (the link starts after a fifth is read, not
    half), the rest in equal parts; lead None: equal parts; one part otherwise."""
    from types import SimpleNamespace

    from opentpu.host.offload import BoardDram
    lay = Layout.build(4096, E, K, (3, 3), SLOT)
    m = BoardDram(SimpleNamespace(board=None), lay)
    assert m.lead == 0.2 and m.pieces == 2
    assert m._cuts(408, True) == [(0, 82), (82, 408)] and m._cuts(408, False) == [(0, 408)]
    assert m._cuts(1, True) == [(0, 1)] and m._cuts(3, True) == [(0, 1), (1, 3)]
    m.pieces = 3
    assert m._cuts(408, True) == [(0, 82), (82, 245), (245, 408)]
    m.lead = None
    assert m._cuts(408, True) == [(0, 136), (136, 272), (272, 408)]
    with pytest.raises(ValueError, match="lead"):
        BoardDram(SimpleNamespace(board=None), lay, lead=1.0)


@pytest.mark.parametrize("hints", [False, True])
@pytest.mark.parametrize("chash", [False, True])
@pytest.mark.parametrize("slot,fmt,pieces", [(128 * 37, "bytes", 2), (64 * 75, "bytes", 2),
                                             (4096 * 3, "split", 3), (4096 * 3, "split", 1),
                                             (128 * 37, "split", 2), (4096 * 3, "readv", 3),
                                             (128 * 37, "readv", 2)])
def test_board_dram_writes_what_board_write_writes(chash, slot, fmt, pieces, hints, tmp_path):
    """BoardDram (the card's fast path: one-pass channel runs, a worker thread's DMA, the
    host's words from a shadow without reading the card) leaves the card's two channel memories
    exactly as BackendDram's Board.write does, request after request: the slots (with CHASH's
    swaps or not), the directory and served (each entry and served one DMA call: its beat). A
    slot of a half chunk falls back to Board.write.
    split: the pool a file in the split format, read with preadv straight into the channel
    runs under CHASH at a page-aligned slot (pages of either parity; a slot of 37 chunks is
    two pages apart: Layout's pitch); a request's first miss in `pieces` parts (the first a
    fifth of it), each DMAed when it is read. readv: the same through os.preadv's buffer list
    (a record without readiov: PoolFile.iov False) instead of preadv_iov's iovec array.
    hints: both servers by decayed use, each request after a hint (one of its experts and one
    it does not name) and 0, 2 or 4 polls with no request: the hinted experts' 4 KiB parts (a
    staging pair's first bytes) land as Board.write writes them."""
    from types import SimpleNamespace

    from opentpu.host.board import Board
    from opentpu.host.fake import FakeTransport
    from opentpu.host.offload import RUN, BackendDram, BoardDram, PoolFile, to_split

    def board():
        b = Board(FakeTransport(ch_bytes=1 << 20, devname=None))
        b.info()["caps"]["chash"] = chash
        return b

    def pool(g):
        return np.random.default_rng(g).integers(0, 256, slot, dtype=np.uint8)

    lay = Layout.build(4096, 4, 2, (2, 3), slot)
    fast_pool = pool
    if fmt in ("split", "readv"):
        f = tmp_path / "pool.bin"
        f.write_bytes(b"".join(to_split(pool(g)).tobytes() for g in range(8)))
        pf = PoolFile(f, slot, split=True)
        pf.iov = fmt == "split"
        fast_pool = pf.get
        assert (pf.get(0).readiov is not None) == (fmt == "split")
    ba, bb = board(), board()
    kw = dict(policy="lfu", part=RUN) if hints else {}
    fast = ExpertServer(BoardDram(SimpleNamespace(board=ba), lay, pieces=pieces), lay,
                        fast_pool, **kw)
    dmas, dma = [], fast.mem._dma

    def counted(off, bufs, a=0, b=None):
        dmas.append(len(bufs[0][a:b]) < len(bufs[0]))
        dma(off, bufs, a, b)
    fast.mem._dma = counted
    plain = ExpertServer(BackendDram(SimpleNamespace(write=lambda s, a, d: bb.write(a, d),
                                                     read=lambda s, a, n: bb.read(a, n))),
                         lay, pool, **kw)
    for srv in (fast, plain):
        srv.load([0, 4])
    assert isinstance(fast.mem, BoardDram) and ba.chash == chash

    def same():
        for c in (0, 1):
            assert np.array_equal(ba.t.ch[c], bb.t.ch[c]), c
    same()
    m, lo, hi = fast.mem, fast.mem.lo, fast.mem.lo + len(fast.mem.shadow)
    words, calls, mw, w = [0], [0], ba.t.mem_write, m.write

    def call(ch, off, data):                # DMA calls into the host's words
        calls[0] += lo // 2 <= off < hi // 2
        mw(ch, off, data)

    def write(addr, data):                  # the server's writes of them
        words[0] += lo <= addr < hi
        w(addr, data)
    ba.t.mem_write, m.write = call, write
    reqs = [[0, 1], [4, 6], [2, 3], [5, 7], [1, 2], [6, 4], [3, 0], [7, 5]]
    seq, G = 0, 4 * 2

    def post(ids):
        for b in (ba, bb):                  # the card's post: the row, then seq
            b.write(lay.row, np.array(ids + [0] * (LINE // 4 - len(ids)), np.float32))
            b.write(lay.mbox, np.float32(seq).tobytes())
        assert fast.poll() == plain.poll() == 1
        same()
    for i, ids in enumerate(reqs):
        if hints:
            seq += 1
            post([G + ids[1], G + next(g for g in range(ids[0] // 4 * 4, 8) if g not in ids)])
            for _ in range(2 * (i % 3)):
                assert fast.poll() == plain.poll()
                same()
        seq += 1
        post(ids)
    assert calls[0] == words[0] > len(reqs)     # an entry or served: one beat, one DMA call
    assert fast.misses == plain.misses > 0 and fast.bytes == plain.bytes
    assert (fast.mem.direct > 0) == (fmt != "bytes" and chash)
    if not hints:
        assert any(dmas) == (fmt != "bytes" and chash and pieces > 1)     # a part's DMA
    else:
        assert fast.prefetched == plain.prefetched > 0 and fast.promoted == plain.promoted > 0
    assert float(np.frombuffer(ba.read(lay.served, 4), np.float32)[0]) == seq


def test_board_dram_raises_a_dma_error_at_flush():
    """A DMA call that fails in BoardDram's worker is raised by the next flush (ExpertServer.poll
    calls it), not lost."""
    from types import SimpleNamespace

    from opentpu.host.board import Board
    from opentpu.host.fake import FakeTransport
    from opentpu.host.offload import BoardDram

    class Broken(FakeTransport):
        def mem_write(self, ch, off, data):
            if len(data) > 1024:
                raise IOError("XDMA h2c write failed")
            super().mem_write(ch, off, data)

    lay = Layout.build(4096, 4, 2, (2, 2), 128 * 37)
    srv = ExpertServer(BoardDram(SimpleNamespace(board=Board(Broken(ch_bytes=1 << 20,
                                                                    devname=None))), lay),
                       lay, lambda g: np.zeros(128 * 37, np.uint8))
    with pytest.raises(IOError, match="h2c"):
        srv.load([0])


def test_board_dram_reads_only_with_its_queue_drained():
    """BoardDram.read raises while the worker's DMA is queued or in flight (a card->host call
    beside a host->card one slips the card's writes); after flush it reads. ExpertServer's polls
    on BoardDram (test_board_dram_writes_what_board_write_writes) never meet it."""
    import threading
    from types import SimpleNamespace

    from opentpu.host.board import Board
    from opentpu.host.fake import FakeTransport
    from opentpu.host.offload import BoardDram, _f32

    go = threading.Event()

    class Slow(FakeTransport):
        def mem_write(self, ch, off, data):
            go.wait(10)
            super().mem_write(ch, off, data)

    lay = Layout.build(4096, 4, 2, (2, 2), 128 * 37)
    m = BoardDram(SimpleNamespace(board=Board(Slow(ch_bytes=1 << 20, devname=None))), lay)
    m.write(lay.served, _f32(3.0))                  # one beat: one DMA call in the worker
    with pytest.raises(RuntimeError, match="flush first"):
        m.read(lay.mbox, 4)
    go.set()
    m.flush()
    assert np.frombuffer(m.read(lay.served, 4), np.float32)[0] == 3.0


def test_pool_split_tool_copies_the_packed_experts(tmp_path, monkeypatch):
    """tools/offload/pool_split.py: a slot-format pool file's packed experts, in the split
    format; MO.serve's reader gives back their bytes."""
    import runpy
    import sys
    from pathlib import Path

    from opentpu.host.offload import SPLIT, PoolFile
    n, slot = 5, 128 * 37
    x = np.random.default_rng(0).integers(0, 256, (n, slot), dtype=np.uint8)
    packed = np.array([1, 0, 1, 1, 0], np.uint8)
    src, dst = tmp_path / "pool.bin", tmp_path / "pool.split.bin"
    src.write_bytes((x * packed[:, None]).tobytes())
    Path(str(src) + ".packed").write_bytes(packed.tobytes())
    tool = Path(__file__).parent.parent / "tools/offload/pool_split.py"
    monkeypatch.setattr(sys, "argv", ["pool_split.py", str(src), str(dst)])
    runpy.run_path(str(tool), run_name="__main__")
    assert Path(str(dst) + ".format").read_text().strip() == SPLIT
    assert Path(str(dst) + ".packed").read_bytes() == packed.tobytes()
    pf = PoolFile(dst, slot, split=True)
    for g in np.nonzero(packed)[0]:
        assert bytes(pf.get(int(g))) == x[g].tobytes()


def test_pool_file_residency(tmp_path):
    """PoolFile.resident: the packed experts' bytes in the page cache (mincore). Experts just
    written are there, a hole of the sparse file (never written or read) is not. On Linux, a
    file the kernel drops from the cache (POSIX_FADV_DONTNEED after fsync; not tmpfs) comes
    back with the warm thread."""
    import os
    import sys

    from opentpu.host.offload import PoolFile
    slot, n = 4096 * 4, 6
    f = tmp_path / "pool.bin"
    f.write_bytes(np.random.default_rng(0).integers(0, 256, slot * 4, dtype=np.uint8).tobytes())
    os.truncate(f, slot * n)                            # experts 4 and 5: a hole
    pf = PoolFile(f, slot, split=False)
    if pf.resident(range(n)) is None:
        pytest.skip("no mincore here")
    assert pf.resident(range(n)) == 4 * slot and pf.resident([2, 3, 5]) == 2 * slot
    if sys.platform != "linux":
        return
    fd = os.open(f, os.O_RDONLY)
    os.fsync(fd)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)
    if pf.resident(range(4)) < 4 * slot:                # (tmpfs keeps its pages)
        pf.warm(range(4)).join(timeout=60)
        assert pf.resident(range(4)) == 4 * slot


def test_slots_are_whole_run_blocks_apart_so_every_expert_reads_in_place(tmp_path):
    """Layout's pitch: a slot of 3.5 RUN blocks (gemma-4-26B-A4B's 841.5) starts every slot on a
    RUN block (the last one's end its own bytes), so BoardDram reads every expert of a split
    pool straight into its runs (`direct`), none through the copy path; a slot under RUN keeps
    its size as its pitch."""
    from types import SimpleNamespace

    from opentpu.host.board import Board
    from opentpu.host.fake import FakeTransport
    from opentpu.host.offload import RUN, BoardDram, PoolFile, to_split
    slot = 3 * RUN + RUN // 2
    lay = Layout.build(4096, 4, 2, (2, 3), slot)
    assert lay.pitch == 4 * RUN and all(a % RUN == 0 for a, _ in lay.slots)
    assert lay.slots[1][0] == lay.slots[0][0] + 2 * lay.pitch
    assert lay.end == lay.slots[1][0] + 2 * lay.pitch + slot
    assert Layout.build(4096, 4, 2, (2, 3), 128 * 7).pitch == 128 * 7
    x = np.random.default_rng(0).integers(0, 256, (8, slot), dtype=np.uint8)
    f = tmp_path / "pool.bin"
    f.write_bytes(b"".join(to_split(r).tobytes() for r in x))
    b = Board(FakeTransport(ch_bytes=1 << 20, devname=None))
    b.info()["caps"]["chash"] = True
    m = BoardDram(SimpleNamespace(board=b), lay)
    srv = ExpertServer(m, lay, PoolFile(f, slot, split=True).get)
    srv.load(range(8))
    assert m.direct == 5
    for g in srv.lru[0].keys() | srv.lru[1].keys():
        assert srv.lru[g // 4][g] % RUN == 0
        assert np.array_equal(np.asarray(b.read(srv.lru[g // 4][g], slot)).view(np.uint8), x[g])


def test_pool_file_touches_what_it_reads_through_its_map(tmp_path):
    """PoolFile's `mapped` (the default): each expert read (get, warm) is touched through a
    read-only map of the pool, so its pages stand as mapped ones under MGLRU (docs/offload.md
    10.7); the bytes are preadv's either way. On Linux the map's resident pages
    (/proc/self/smaps) cover what was read, and mapped=False leaves no map."""
    import sys

    from opentpu.host.offload import PoolFile
    slot, n = 4096 * 3 + 2048, 6                      # (experts off a page too)
    x = np.random.default_rng(0).integers(0, 256, (n, slot), dtype=np.uint8)
    f = tmp_path / "pool.bin"
    f.write_bytes(x.tobytes())

    def rss():                                        # the pool's maps' resident bytes
        if sys.platform != "linux":
            return None
        out, cur = 0, False
        for ln in open("/proc/self/smaps"):
            if ln[0] in "0123456789abcdef" and "-" in ln.split()[0]:
                cur = ln.rstrip().endswith(str(f))
            elif cur and ln.startswith("Rss:"):
                out += int(ln.split()[1]) * 1024
        return out
    for mapped in (True, False):
        pf = PoolFile(f, slot, split=False, mapped=mapped)
        for g in (1, 4):
            assert bytes(pf.get(g)) == x[g].tobytes()
        pf.warm([5]).join(timeout=60)
        if mapped:                                    # (read-only: nothing writes through it)
            assert pf._mc and (rss() is None or rss() >= 3 * slot)
            assert not pf._mc[1].flags.writeable
        else:
            assert pf._mc is None
        del pf


def test_board_dram_keeps_a_staging_pair_until_its_last_part(tmp_path):
    """A staging pair goes back to the free list only after its expert's last part is on the
    card: with one pair and a slow link, the next expert's read waits for it (else it would
    overwrite the parts still queued)."""
    import time
    from types import SimpleNamespace

    from opentpu.host.board import Board
    from opentpu.host.fake import FakeTransport
    from opentpu.host.offload import BoardDram, PoolFile, to_split

    class Slow(FakeTransport):
        def mem_write(self, ch, off, data):
            if len(data) > 1024:
                time.sleep(2e-3)
            super().mem_write(ch, off, data)

    slot = 4096 * 3
    x = np.random.default_rng(0).integers(0, 256, (8, slot), dtype=np.uint8)
    f = tmp_path / "pool.bin"
    f.write_bytes(b"".join(to_split(r).tobytes() for r in x))
    b = Board(Slow(ch_bytes=1 << 20, devname=None))
    b.info()["caps"]["chash"] = True
    lay = Layout.build(4096, 4, 2, (2, 3), slot)
    m, pf = BoardDram(SimpleNamespace(board=b), lay, depth=1, pieces=3), PoolFile(f, slot, True)
    pf.io = {}
    srv = ExpertServer(m, lay, pf.get)
    srv.load(range(8))
    for g in srv.lru[0].keys() | srv.lru[1].keys():
        s = srv.lru[g // 4][g]
        assert np.array_equal(np.asarray(b.read(s, slot)).view(np.uint8), x[g]), g
    assert m.wait_s > 2e-3                              # (moe_card's stage_wait)
    assert sum(v[2] for v in pf.io.values()) == 5 * slot   # each read counted once, in parts


def test_pool_file_counts_its_reads_by_the_page_cache(tmp_path):
    """PoolFile.io (moe_card's decode): each read counted as "cached" (all its pages in the
    page cache just before it) or "disk", [reads, seconds, bytes, bytes not in the cache];
    nothing without io. A hole of the sparse file (never written or read) is not cached."""
    import os

    from opentpu.host.offload import PoolFile, to_split
    slot, n = 4096 * 4, 6
    x = np.random.default_rng(0).integers(0, 256, (4, slot), dtype=np.uint8)
    f = tmp_path / "pool.bin"
    f.write_bytes(x.tobytes())
    os.truncate(f, slot * n)                            # experts 4 and 5: holes
    for split, hole in ((False, 4), (True, 5)):
        pf = PoolFile(f, slot, split=split)
        pf.willneed([1, 2])                             # (a hint to the kernel, or nothing)
        if pf.resident(range(n)) is None:
            pytest.skip("no mincore here")
        np.asarray(pf.get(1))
        assert pf.io is None
        pf.io = {}
        got = np.asarray(pf.get(2))
        assert np.array_equal(to_split(got) if split else got, x[2])    # (x as written)
        np.asarray(pf.get(hole))
        assert pf.io["cached"][0] == 1 and pf.io["cached"][2:] == [slot, 0], pf.io
        assert pf.io["disk"][0] == 1 and pf.io["disk"][2:] == [slot, slot], pf.io


def test_lfu_policy_evicts_the_least_decayed_use():
    """policy="lfu": the victim is the cached expert, not in the request, whose uses (each
    halving every `half` requests of its layer; a warm expert one use at the start) sum least,
    ties to the least recently used; checked request by request against that sum computed
    directly, on a skewed random stream. LRU stays the default."""
    half, slots = 4.0, 5
    lay = Layout.build(4096, E, K, (slots,), SLOT)
    srv = ExpertServer(SimDram(np.zeros(lay.end + 4096, np.uint8)), lay, _pool, policy="lfu",
                       half=half)
    srv.load([0, 1, 2, 3, 4])
    assert ExpertServer(srv.mem, lay, _pool).policy == "lru"
    rng = np.random.default_rng(3)
    p = 1.0 / np.arange(1, E + 1) ** 1.2
    uses = {g: [0] for g in range(5)}                  # request times of each expert's uses
    cache, order = set(range(5)), [4, 3, 2, 1, 0]      # recency: least recent first
    for t in range(1, 300):
        ids = [int(g) for g in rng.choice(E, K, replace=False, p=p / p.sum())]
        for g in ids:
            uses.setdefault(g, []).append(t)
        miss = [g for g in ids if g not in cache]
        for g in ids:
            if g in cache:
                order.remove(g)
                order.append(g)
        want = []
        for g in miss:
            if len(cache) >= slots:
                cand = [v for v in order if v not in ids]
                sc = [sum(0.5 ** ((t - u) / half) for u in uses[v]) for v in cand]
                v = cand[int(np.argmin(sc))]
                cache.discard(v)
                order.remove(v)
                want.append(v)
            cache.add(g)
            order.append(g)
        before = set(srv.lru[0])
        srv.serve(ids)
        assert before - set(srv.lru[0]) == set(want), (t, ids, want)
        assert set(srv.lru[0]) == cache


def _hint_setup(policy="lfu", warm=(0, 1, 2), slot=2 * 4096 + 128, **kw):
    """Layer 0 of 3 slots with `warm` (LRU order 2, 1, 0: 0 newest), experts of three 4 KiB parts
    (the last 128 bytes)."""
    from opentpu.host.offload import RUN
    lay = Layout.build(4096, E, K, (3, 3), slot)
    mem = SimDram(np.zeros(lay.end + 4096, np.uint8))
    srv = ExpertServer(mem, lay, lambda g: np.full(slot, g + 1, np.uint8).tobytes(),
                       policy=policy, part=RUN, **kw)
    srv.load(list(warm))
    return lay, mem, srv, E * 2


def _landed(mem, lay, srv, g):
    a, p = _entry(mem, lay, g)
    return p == 1.0 and mem.read(a, lay.slot_bytes) == srv.pool(g)


def test_a_hint_takes_its_slots_at_once_and_lands_on_idle_polls():
    """A hint (ids at G = layers x E and above): each named expert not in a slot gets one at
    once (here the least recent of the equally used warm ones that the hint does not name),
    the victim's entry cleared, and served written; no use is counted. Each poll that finds no
    request then sends one part of it, its entry once the last part is in. A request naming it
    afterwards hits."""
    lay, mem, srv, G = _hint_setup()
    _post(mem, lay, 1, [G + 3, G + 1])              # 1 resident; 3 replaces 2
    assert srv.poll() == 1 and _served(mem, lay) == 1.0
    assert _entry(mem, lay, 2) == (0, 0.0) and _entry(mem, lay, 3) == (0, 0.0)
    assert dict(srv.pending) == {3: 0} and srv.t == [0, 0] and 3 not in srv.use[0]
    assert (srv.hints, srv.hits, srv.misses) == (1, 0, 0)
    for sent in (4096, 8192):
        assert srv.poll() == 1 and dict(srv.pending) == {3: sent}
        assert _entry(mem, lay, 3) == (0, 0.0)
    assert srv.poll() == 1 and not srv.pending and _landed(mem, lay, srv, 3)
    assert srv.poll() == 0 and srv.prefetched == 1 and srv.bytes == lay.slot_bytes
    _post(mem, lay, 2, [3, 1])
    assert srv.poll() == 1 and (srv.hits, srv.misses) == (2, 0)


def test_a_request_sends_the_rest_of_a_hinted_expert_or_replaces_it():
    """A request naming a hinted expert still on its way sends the rest of it at once (a miss,
    `promoted`); one that names neither hinted expert may replace one (no use: the first
    victim under decayed use), dropping what was sent of it."""
    lay, mem, srv, G = _hint_setup()
    _post(mem, lay, 1, [G + 3, G + 4])              # 3 replaces 2, 4 replaces 1
    srv.poll()
    assert _entry(mem, lay, 1) == (0, 0.0) and list(srv.pending) == [3, 4]
    srv.poll()                                      # one part of 3
    assert dict(srv.pending) == {3: 4096, 4: 0}
    _post(mem, lay, 2, [4, 0])
    assert srv.poll() == 1 and _landed(mem, lay, srv, 4) and _landed(mem, lay, srv, 0)
    assert (srv.hits, srv.misses, srv.promoted) == (1, 1, 1) and dict(srv.pending) == {3: 4096}
    _post(mem, lay, 3, [5, 4])                      # 3 (no use) is the victim, half sent
    assert srv.poll() == 1 and not srv.pending and srv.dropped == 1
    assert _entry(mem, lay, 3) == (0, 0.0) and _landed(mem, lay, srv, 5)
    assert srv.poll() == 0 and srv.prefetched == 0
    assert srv.bytes == 4096 + 2 * lay.slot_bytes   # 3's first part, 4, 5


def test_lru_ignores_hints_and_a_hint_is_one_layers():
    """The "lru" policy answers a hint with served alone (an LRU victim of a wrong hint is a
    recent expert, docs/offload.md 5.4); a hint naming two layers is refused."""
    lay, mem, srv, G = _hint_setup(policy="lru")
    before = mem.read(lay.dir, 8 * E * 2)
    _post(mem, lay, 1, [G + 3, G + 4])
    assert srv.poll() == 1 and _served(mem, lay) == 1.0 and srv.poll() == 0
    assert mem.read(lay.dir, 8 * E * 2) == before and not srv.pending and srv.hints == 1
    _post(mem, lay, 2, [G + 3, G + E + 4])
    with pytest.raises(ValueError, match="one layer"):
        srv.poll()


def test_a_request_withdraws_its_layers_unnamed_hints_with_drop():
    """drop=True: a request withdraws its layer's hinted experts it does not name that have not
    landed (their slots free again, nothing more sent), and keeps the others' layers' hints; the
    next miss in the layer takes a freed slot without replacing a cached expert."""
    lay, mem, srv, G = _hint_setup(drop=True, warm=(0, 1, 2, E))
    _post(mem, lay, 1, [G + E + 5, G + E + 6])     # layer 1: two free slots, 5 and 6
    srv.poll()
    _post(mem, lay, 2, [G + 3, G + 4])             # layer 0: 3 replaces 2, 4 replaces 1
    srv.poll()
    srv.poll()                                      # one part of 5 (the oldest)
    assert dict(srv.pending) == {E + 5: 4096, E + 6: 0, 3: 0, 4: 0}
    _post(mem, lay, 3, [4, 0])                      # 3 withdrawn; layer 1's stay
    assert srv.poll() == 1 and srv.withdrawn == 1 and srv.dropped == 0
    assert list(srv.pending) == [E + 5, E + 6] and 3 not in srv.lru[0]
    assert _entry(mem, lay, 3) == (0, 0.0) and len(srv.free[0]) == 1
    _post(mem, lay, 4, [7, 4])                      # 7 takes 3's slot: 0 stays
    srv.poll()
    assert set(srv.lru[0]) == {0, 4, 7} and _landed(mem, lay, srv, 7)


@pytest.mark.parametrize("chash", [False, True])
def test_board_dram_reads_a_beat_as_board_read_does(chash):
    """BoardDram.read of bytes within one 64-byte beat (a poll's seq, a request's row) reads that
    beat alone from its channel (CHASH's swap or not) and gives Board.read's bytes; a longer
    read is Board.read's."""
    from types import SimpleNamespace

    from opentpu.host.board import Board
    from opentpu.host.fake import FakeTransport
    from opentpu.host.offload import BoardDram

    b = Board(FakeTransport(ch_bytes=1 << 16, devname=None))
    b.info()["caps"]["chash"] = chash
    x = np.random.default_rng(1).integers(0, 256, 1 << 14, dtype=np.uint8)
    b.write(0, x)
    m = BoardDram(SimpleNamespace(board=b), Layout.build(4096, 4, 2, (2,), 128))
    calls, mr = [], b.t.mem_read
    b.t.mem_read = lambda *a, **k: calls.append(a) or mr(*a, **k)
    for addr, n in [(0, 4), (64, 4), (4096 + 64, 16), (100 * 128 + 60, 4), (5 * 128 + 64, 64),
                    (7 * 128 + 60, 8), (3 * 128, 128)]:
        calls.clear()
        assert m.read(addr, n) == x[addr:addr + n].tobytes(), (addr, n)
        assert len(calls) == (1 if addr % 64 + n <= 64 else 2), (addr, n)


def _post_n(mem, lay, seq, ids):
    """A post of len(ids) ids (repeats kept): the row, then row2, then the count and seq."""
    h = LINE // 4
    mem.write(lay.row, np.array(ids[:h] + [0] * (h - len(ids[:h])), np.float32))
    if len(ids) > h:
        mem.write(lay.row2, np.array(ids[h:] + [0] * (2 * h - len(ids)), np.float32))
    mem.write(lay.mbox, np.array([seq, len(ids)], np.float32))


def _consistent(mem, lay, srv):
    """The directory is the server's map: each expert in a slot reads {slot, 1.0}, every other
    {0, 0.0}; no slot twice; each layer's slots and free slots are its own number outside
    pooled mode."""
    held = {g: a for lru in srv.lru for g, a in lru.items()}
    assert len(set(held.values())) == len(held)
    for g in range(lay.E * lay.layers):
        assert _entry(mem, lay, g) == ((held[g], 1.0) if g in held else (0, 0.0)), g
    for g, a in held.items():
        assert mem.read(a, lay.slot_bytes) == srv.pool(g), g
    if not srv.pooled:
        for j, (_, n) in enumerate(lay.slots):
            assert len(srv.lru[j]) + len(srv.free[j]) == n, j


def test_two_line_layout_moves_nothing_else():
    """lines=2: a second line of ids on its own 128-byte block after the directory, the slots
    after it; lines=1 is the old layout, address for address."""
    one, two = (Layout.build(4096, E, K, (3, 5), SLOT, lines=n) for n in (1, 2))
    assert one == Layout.build(4096, E, K, (3, 5), SLOT) and one.row2 == 0
    assert (one.mbox, one.served, one.dir) == (two.mbox, two.served, two.dir)
    assert two.row2 % (2 * LINE) == 0 and two.row2 >= two.dir + 8 * E * 2
    assert two.slots[0][0] >= two.row2 + LINE and two.slots[0][0] % 4096 == 0
    assert (one.max_ids, two.max_ids) == (16, 32)
    with pytest.raises(ValueError):
        Layout.build(4096, E, K, (3,), SLOT, lines=3)


def test_a_request_of_more_than_16_ids_reads_its_second_line():
    """A 4-row run's request (moe.moe_ffn_rows: R k ids, repeats included) of 20 ids: 16 on the
    row, 4 on row2; each served once. A one-line layout refuses more than 16."""
    for lines in (2, 1):
        lay = Layout.build(4096, 40, K, (24,), SLOT, lines=lines)
        mem = SimDram(np.zeros(lay.end + 4096, np.uint8))
        srv = ExpertServer(mem, lay, _pool)
        srv.load()
        ids = list(range(18)) + [3, 17]
        _post_n(mem, lay, 1, ids)
        if lines == 1:
            with pytest.raises(RuntimeError, match="lines hold 16"):
                srv.poll()
            continue
        assert srv.poll() == 1 and _served(mem, lay) == 1.0
        assert srv.misses == 18 and srv.history is None
        _consistent(mem, lay, srv)


def _pooled_run(srv, mem, lay, restore):
    """A layer-major prefill of three layers (each its union over two runs of 2 rows), the
    restore, then decode requests."""
    seq = 0
    srv.begin_prefill()
    for run in ([0, 1, 1, 2], [1, 2, 2, 0], [8, 9, 9, 10], [9, 9, 8, 10],
                [16, 17, 17, 18], [17, 17, 16, 18]):
        seq += 1
        _post_n(mem, lay, seq, run)
        assert srv.poll() == 1
        _consistent(mem, lay, srv)
    srv.end_prefill(restore)
    _consistent(mem, lay, srv)
    for ids in ([4, 5], [12, 13], [20, 21], [1, 2]):
        seq += 1
        _post_n(mem, lay, seq, ids)
        assert srv.poll() == 1
        _consistent(mem, lay, srv)
    return seq


def test_pooled_prefill_takes_any_layers_slots_then_restores_each_layers():
    """begin_prefill: a layer's experts take any layer's slots (2 a layer, 6 in all: each
    layer's union of 3 fits), the least recent of another layer the victim. end_prefill: each
    layer back to its 2, keeping its experts of most use (a request's repeats count once);
    lazy loads nothing, eager each layer's most used; decode's requests then evict within
    their layer."""
    for restore in ("lazy", "eager"):
        lay, mem, srv = _setup(slots=(2, 2, 2))
        srv.begin_prefill()
        assert srv.pooled
        runs = ([0, 1, 1, 2], [1, 1, 2, 2], [8, 9, 9, 10], [10, 10, 9, 9])
        for seq, run in enumerate(runs, 1):
            _post_n(mem, lay, seq, run)
            assert srv.poll() == 1
        assert set(srv.lru[0]) == {0, 1, 2} and set(srv.lru[1]) == {8, 9, 10}  # 3 > 2 each
        _post_n(mem, lay, 5, [16, 17, 17, 18])          # layer 0's three, the oldest, leave
        srv.poll()
        _post_n(mem, lay, 6, [18, 17, 18, 17])
        srv.poll()
        assert not srv.lru[0] and set(srv.lru[2]) == {16, 17, 18}
        _consistent(mem, lay, srv)
        b = srv.bytes
        srv.end_prefill(restore)
        assert not srv.pooled
        assert set(srv.lru[1]) == {9, 10} and set(srv.lru[2]) == {17, 18}     # two uses each
        assert set(srv.lru[0]) == ({1, 2} if restore == "eager" else set())
        assert srv.bytes - b == (2 * SLOT if restore == "eager" else 0)
        _consistent(mem, lay, srv)
        _post_n(mem, lay, 7, [11, 12])                  # layer 1's two slots, its own victims
        srv.poll()
        assert set(srv.lru[1]) == {11, 12} and set(srv.lru[2]) == {17, 18}
        _consistent(mem, lay, srv)


def test_pooled_prefill_on_board_dram_writes_what_board_write_writes():
    """The pooled prefill and both restores through BoardDram (the card's: the worker, the
    shadow of the host's words, CHASH) leave the board's DRAM as Board.write's server does."""
    from types import SimpleNamespace

    from opentpu.host.board import Board
    from opentpu.host.fake import FakeTransport
    from opentpu.host.offload import BackendDram, BoardDram

    slot = 2 * 4096 + 128
    lay = Layout.build(4096, E, K, (2, 2, 2), slot, lines=2)

    def pool(g):
        return np.random.default_rng(g).integers(0, 256, slot, dtype=np.uint8).tobytes()
    for restore in ("lazy", "eager"):
        ba, bb = (Board(FakeTransport(ch_bytes=1 << 20, devname=None)) for _ in range(2))
        for b in (ba, bb):
            b.info()["caps"]["chash"] = True
        fast = ExpertServer(BoardDram(SimpleNamespace(board=ba), lay), lay, pool, policy="lfu")
        plain = ExpertServer(BackendDram(SimpleNamespace(write=lambda s, a, d: bb.write(a, d),
                                                         read=lambda s, a, n: bb.read(a, n))),
                             lay, pool, policy="lfu")
        seqs = []
        for srv in (fast, plain):
            srv.load()
            seqs.append(_pooled_run(srv, srv.mem, lay, restore))
        assert seqs[0] == seqs[1] and fast.misses == plain.misses and fast.bytes == plain.bytes
        for c in (0, 1):
            assert np.array_equal(ba.t.ch[c], bb.t.ch[c]), (restore, c)
