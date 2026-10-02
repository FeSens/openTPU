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
    words = [lay.mbox, lay.served, lay.answer, lay.dir] + [a for a, _ in lay.slots]
    assert all(a % LINE == 0 for a in words)
    assert lay.served >= lay.row + LINE and lay.answer >= lay.served + LINE
    assert lay.dir >= lay.answer + LINE
    assert lay.slots[0][0] >= lay.dir + 8 * E * 2
    assert lay.tag == SLOT and lay.pitch == SLOT + 128      # a slot, then its tag chunk
    assert lay.slots[1][0] == lay.slots[0][0] + 3 * lay.pitch
    assert lay.all_slots() == [a + i * lay.pitch for a, n in lay.slots for i in range(n)]
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


def test_ahead_sees_a_requests_later_misses_once_the_first_is_on_its_way():
    """ExpertServer.ahead (moe_card --willneed: PoolFile.willneed) is called with a request's
    missing ids but its first, once that one's first part is on its way (with the answer: the
    link starts on the first expert, the others' reads queued behind it); a request of one
    miss or of hits only does not call it."""
    lay, mem, srv = _setup(warm=[0, 1, 2])
    seen, reads = [], []
    pool = srv.pool
    srv.pool = lambda g: (reads.append(g), pool(g))[1]
    srv.ahead = lambda ids: seen.append((list(ids), len(reads)))
    _post(mem, lay, 1, [5, 4])
    srv.poll()
    _post(mem, lay, 2, [0, 6])                      # 0 hits (2 was a victim)
    srv.poll()
    assert seen == [([4], 1)] and reads == [5, 4, 6]
    _post(mem, lay, 3, [6, 0])                      # hits only
    srv.poll()
    assert len(seen) == 1


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


def test_a_requests_answer_follows_its_first_experts_first_part():
    """A request's answer (its missing experts' slots) is written once the first expert's
    first part is on its way (a memory without write_slot: after that expert), each expert's
    tag right after its bytes, and the directory's new entries and victims' clears after the
    last expert, before served (the card waits on the answer and the tags; it reads a victim's
    entry only in a later request: after served), so the link starts with the first expert's
    bytes; clear_late False clears each victim before its slot is written, as before
    docs/offload.md 10.8. Either way the victims' entries read empty once poll returns."""
    for late in (True, False):
        lay, mem, srv = _setup(warm=[0, 1, 2])      # layer 0 LRU order: 2, 1, 0 (0 newest)
        srv.clear_late = late
        log, w = [], mem.write
        tags = {a + lay.tag for a in lay.all_slots()}

        def rec(addr, data, w=w):
            log.append("served" if addr == lay.served else "answer" if addr == lay.answer else
                       "dir" if lay.dir <= addr < lay.dir + 8 * E * 2 else
                       "tag" if addr in tags else "slot")
            w(addr, data)
        _post(mem, lay, 1, [5, 6])                  # two misses: victims 2 and 1
        mem.write = rec
        assert srv.poll() == 1
        assert log == (["slot", "tag", "answer", "slot", "tag", "dir", "served"] if late else
                       ["dir", "dir", "slot", "tag", "answer", "slot", "tag", "dir", "served"])
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

    def counted(off, bufs, a=0, b=None, order=(0, 1)):
        dmas.append(len(bufs[0][a:b]) < len(bufs[0]) - 64)   # (the tag beat after the runs)
        dma(off, bufs, a, b, order)
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


@pytest.mark.parametrize("ahead", [True, False])
def test_idle_parts_go_as_one_call_each_and_are_read_ahead(ahead, tmp_path):
    """read_ahead (docs/offload.md 13.12): on BoardDram (CHASH, a split pool file) an idle
    poll's part was read by the poll before (stage, beside that poll's DMA: a part's, or the
    hint's served) and goes as one DMA call per channel; one not read ahead (the first after
    ahead_layer) keeps the lead cut. A request first drops the staged part (its pair free
    again). ahead_layer's next expert takes its slot when its part is staged. The card's
    memories stay as Board.write leaves them, with read_ahead and without (each idle part of 2
    blocks cut in two)."""
    from types import SimpleNamespace

    from opentpu.host.board import Board
    from opentpu.host.fake import FakeTransport
    from opentpu.host.offload import RUN, BackendDram, BoardDram, PoolFile, to_split

    def board():
        b = Board(FakeTransport(ch_bytes=1 << 21, devname=None))
        b.info()["caps"]["chash"] = True
        return b

    slot = 4 * RUN
    x = np.random.default_rng(3).integers(0, 256, (24, slot), dtype=np.uint8)
    f = tmp_path / "pool.bin"
    f.write_bytes(b"".join(to_split(r).tobytes() for r in x))
    pf = PoolFile(f, slot, split=True)
    lay = Layout.build(4096, 8, 2, (3, 3, 3), slot)
    ba, bb = board(), board()
    fast = ExpertServer(BoardDram(SimpleNamespace(board=ba), lay, pieces=2), lay, pf.get,
                        policy="lfu", part=2 * RUN)
    plain = ExpertServer(BackendDram(SimpleNamespace(write=lambda s, a, d: bb.write(a, d),
                                                     read=lambda s, a, n: bb.read(a, n))),
                         lay, lambda g: x[g], policy="lfu", part=2 * RUN)
    fast.read_ahead = ahead
    m = fast.mem
    recs, staged, dma, stage = [], [], m._dma, m.stage
    m._dma = lambda *a, **kw: (recs.append(a[3] if len(a) > 3 else 0), dma(*a, **kw))
    m.stage = lambda *a: (lambda h: (staged.append(h is not None), h)[1])(stage(*a))
    for srv in (fast, plain):
        srv.load([0, 1, 8, 9, 16, 17])
    seq, G = 0, 24

    def poll():
        n0 = len(recs)
        assert fast.poll() == plain.poll()
        for c in (0, 1):
            assert np.array_equal(ba.t.ch[c], bb.t.ch[c]), c
        return len(recs) - n0

    def post(ids):
        nonlocal seq
        seq += 1
        for b in (ba, bb):
            b.write(lay.row, np.array(ids + [0] * (LINE // 4 - len(ids)), np.float32))
            b.write(lay.mbox, np.float32(seq).tobytes())
        return poll()
    r = 1 if ahead else 2                           # an idle part's DMA calls (or a lead cut)
    post([G + 2, G + 3])                            # hint: 2 and 3 take slots, 4 parts to send
    assert staged == ([True] if ahead else [])      # 2's first part read beside served
    assert poll() == r                              # ... sent, and 2's second part read
    assert poll() == r and fast.pending == {3: 0}   # 2's second (with its tag): staged
    assert len(m._held) == ahead                    # 3's first part staged
    assert post([3, 1]) == 3                        # a request first: 3's staged part dropped;
    assert not m._held and fast.promoted == 1       # its misses as before: the first (3's rest)
    assert m._free.qsize() == m.depth               # with the lead cut, the next (1) one call
    assert post([4, 5]) == 3
    for srv in (fast, plain):                       # pooled: 20 a free slot, 21 a victim
        srv.begin_prefill(ahead=True)               # of layer 0 (outside 1 and 2)
        srv.ahead_layer(2, [20, 21])
    assert poll() == 2 and 20 in fast.lru[2] and 21 not in fast.lru[2]     # (not read ahead)
    assert poll() == r and (21 in fast.lru[2]) == ahead     # 21's slot taken as it is staged
    while fast.pending or fast.queue:
        assert poll() == r
    assert poll() == 0 and fast.landed == plain.landed == 2 and not m._held
    assert fast.bytes == plain.bytes and fast.misses == plain.misses
    assert sum(staged) == (6 if ahead else 0)       # each idle part but 20's first (3's
                                                    # first dropped)


def test_halt_aware_idle_parts_wait_near_a_runs_expected_end():
    """halt_aware (docs/offload.md 13.12): no idle part starts while the running program's
    expected end is nearer than a part takes (part_s, or PART_S0 + part / PART_GBS before any
    is measured), so its halt is not seen after a part. The end: the memory's run_clock (the
    run's start and its time with no waits) plus the run's own waits (each request with misses,
    seen to served; a new run starts them at 0). A request is served at once all the same; a
    run past its end by HOLD_LATE, or with no clock, gets parts again. Off: no hold."""
    import time

    from opentpu.host.offload import HOLD_LATE, PART_GBS, PART_S0
    lay, mem, srv, G = _hint_setup()
    clock = [None]
    mem.run_clock = lambda: clock[0]

    def run(left):                                  # a run started now, `left` its time
        clock[0] = (time.perf_counter(), left)
    srv.halt_aware = True
    _post(mem, lay, 1, [G + 3, G + 4])              # 3 and 4 hinted: 6 parts to send
    assert srv.poll() == 1 and len(srv.pending) == 2
    need = PART_S0 + srv.part / PART_GBS
    run(need / 2)                                   # the run ends before a part would
    assert srv.poll() == 0 and srv.poll() == 0 and srv.holds == 2
    assert dict(srv.pending) == {3: 0, 4: 0} and srv._waits == 0.0
    _post(mem, lay, 2, [E + 0, E + 1])              # a request (layer 1, 2 misses): served at
    assert srv.poll() == 1 and _served(mem, lay) == 2.0 and srv.holds == 2   # once, its time
    assert srv._waits > 0                           # the run's own wait
    srv._wait(2 * need)                             # (a wait of 2 parts: the end moves out)
    assert srv.poll() == 1 and srv.pending[3] == srv.part and srv.part_s is not None
    srv.part_s = 1.0                                # (measured parts of a second)
    run(0.5)                                        # a new run: its waits from 0
    assert srv.poll() == 0 and srv._waits == 0.0 and srv.holds == 3
    assert srv.poll() == 0 and srv.holds == 4
    run(-2 * HOLD_LATE)                             # past its end: parts again
    assert srv.poll() == 1
    run(0.5)
    srv.halt_aware = False                          # off: no hold
    assert srv.poll() == 1 and srv.holds == 4
    clock[0], srv.halt_aware = None, True           # no clock: no hold
    while srv.pending:
        assert srv.poll() == 1
    assert _landed(mem, lay, srv, 3) and _landed(mem, lay, srv, 4)


def test_a_held_idle_poll_leaves_no_dma_in_flight():
    """halt_aware holding the first part of an ahead expert that has just taken a victim's
    slot (_next_ahead clears the victim's entry, a DMA): the poll flushes before it returns 0,
    so the next poll's read of seq meets no DMA in flight (BoardDram.read raises with one)."""
    import time
    from types import SimpleNamespace

    from opentpu.host.board import Board
    from opentpu.host.fake import FakeTransport
    from opentpu.host.offload import RUN, BoardDram

    class Slow(FakeTransport):
        def mem_write(self, ch, off, data):
            time.sleep(5e-3)
            super().mem_write(ch, off, data)

    lay = Layout.build(4096, 8, 2, (2, 2, 2), RUN)
    srv = ExpertServer(BoardDram(SimpleNamespace(board=Board(Slow(ch_bytes=1 << 21,
                                                                  devname=None))), lay),
                       lay, lambda g: np.full(RUN, g + 1, np.uint8), policy="lfu", part=RUN)
    srv.load([0, 1, 8, 9, 16, 17])                  # every slot full: 20 needs a victim
    srv.begin_prefill(ahead=True)
    srv.ahead_layer(2, [20])
    srv.halt_aware, srv.mem.run_clock = True, lambda: (time.perf_counter(), 1e-6)
    assert srv.poll() == 0 and srv.holds == 1 and 20 in srv.pending
    assert srv.poll() == 0 and srv.holds == 2       # (its read of seq: no DMA in flight)
    srv.halt_aware = False
    assert srv.poll() == 1 and not srv.pending and srv.landed == 1


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
    RUN block (its tag chunk in the half block after it), so BoardDram reads every expert of a
    split pool straight into its runs (`direct`), none through the copy path; a slot under RUN
    keeps its size and tag chunk, rounded up to its own alignment."""
    from types import SimpleNamespace

    from opentpu.host.board import Board
    from opentpu.host.fake import FakeTransport
    from opentpu.host.offload import RUN, BoardDram, PoolFile, to_split
    slot = 3 * RUN + RUN // 2
    lay = Layout.build(4096, 4, 2, (2, 3), slot)
    assert lay.pitch == 4 * RUN and all(a % RUN == 0 for a, _ in lay.slots)
    assert lay.slots[1][0] == lay.slots[0][0] + 2 * lay.pitch
    assert lay.tag == slot and lay.end == lay.slots[1][0] + 2 * lay.pitch + slot + 128
    assert Layout.build(4096, 4, 2, (2, 3), 128 * 7).pitch == 128 * 8     # + its tag chunk
    assert Layout.build(4096, 4, 2, (2, 3), 256 * 3).pitch == 256 * 4     # at its alignment
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


def test_a_requests_reads_are_touched_once_it_is_served(tmp_path):
    """PoolFile.defer_touch (moe.serve sets it): a read's touch waits until the request is
    served (ExpertServer, after served and the DMA's flush: while the card computes), not
    between the read and its DMA; with io counting (moe_card's decode) each read still says
    whether its pages were in the page cache."""
    from opentpu.host.offload import PoolFile
    lay = Layout.build(4096, E, K, (3, 3), 4096)
    x = np.random.default_rng(0).integers(0, 256, (2 * E, 4096), dtype=np.uint8)
    f = tmp_path / "pool.bin"
    f.write_bytes(x.tobytes())
    pf = PoolFile(f, 4096, split=False)
    pf.defer_touch, pf.io = True, {}
    mem = SimDram(np.zeros(lay.end + 4096, np.uint8))
    srv = ExpertServer(mem, lay, lambda g: pf.get(g).tobytes())
    srv.pool_file = pf
    srv.load()
    log, touch, w = [], pf._touch, mem.write
    pf._touch = lambda off, n: (log.append(("touch", off // 4096)), touch(off, n))
    mem.write = lambda addr, d: (log.append("served") if addr == lay.served else None,
                                 w(addr, d))
    _post(mem, lay, 1, [5, 4])
    srv.poll()
    assert log == ["served", ("touch", 4), ("touch", 5)] and not pf._touches
    assert sum(v[0] for v in pf.io.values()) == 2
    for g in (4, 5):
        assert mem.read(srv.lru[0][g], 4096) == x[g].tobytes()


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


def test_an_expert_landed_in_the_poll_before_a_request_is_answered():
    """docs/offload.md 12.8: a request posted while an idle poll lands one of its experts
    (after the poll read seq, before the expert's entry landed): the card may read that entry
    as missing and wait on its answer word. The host serves that expert as a hit, so the
    request's answer names its slot all the same (its tag landed with it), even with no miss.
    An expert landed two polls before a request is not answered (the card reads its entry
    after the poll between read seq, by when the entry had landed)."""
    from opentpu.host.offload import RUN
    lay, mem, srv, G = _hint_setup(slot=RUN)        # experts of one part
    _post(mem, lay, 1, [G + 3, G + 4])              # hint: 3 and 4 take slots (2 and 1 leave)
    assert srv.poll() == 1 and dict(srv.pending) == {3: 0, 4: 0}
    read, posted = mem.read, []

    def late(a, n):                                 # the card posts right after the poll's
        r = read(a, n)                              # read of seq
        if a == lay.mbox and not posted:
            posted.append(1)
            _post(mem, lay, 2, [0, 3])
        return r
    mem.read = late
    assert _entry(mem, lay, 3) == (0, 0.0)          # what the card reads: 3 missing
    assert srv.poll() == 1 and dict(srv.pending) == {4: 0}  # this poll lands 3 (its entry)
    mem.read = read
    assert _landed(mem, lay, srv, 3)
    assert srv.poll() == 1 and srv.misses == 0 and srv.late == 1    # 0 and 3: hits
    ans = np.frombuffer(mem.read(lay.answer, 8), np.uint32)
    assert ans[1] == srv.lru[0][3] and ans[0] == 0  # 3's slot named, 0's not
    mem.write(lay.answer, np.zeros(2, np.uint32))   # (the card zeroes it)
    assert srv.poll() == 1 and not srv.pending      # 4 lands, then a poll with nothing
    assert srv.poll() == 0
    _post(mem, lay, 3, [4, 0])
    assert srv.poll() == 1 and srv.late == 1
    assert not np.frombuffer(mem.read(lay.answer, 8), np.uint32).any()


def test_hint_caps_send_the_first_n_of_its_top_ids_not_in_a_slot():
    """hint_n / hint_top (docs/offload.md 12.7): of a hint's first hint_top ids (the router's
    best first), the first hint_n not in a slot get one, at once (the victim's entry cleared
    before served covers the hint), the others nothing: no slot, no eviction."""
    lay, mem, srv, G = _hint_setup()                # layer 0: 0, 1, 2 in slots
    srv.hint_n, srv.hint_top = 1, 3
    _post(mem, lay, 1, [G + 1, G + 3, G + 4, G + 5])    # top 3: 1 (in a slot), 3, 4: 3 only
    assert srv.poll() == 1 and _served(mem, lay) == 1.0
    assert list(srv.pending) == [3] and 4 not in srv.lru[0] and 5 not in srv.lru[0]
    assert _entry(mem, lay, 2) == (0, 0.0) and set(srv.lru[0]) == {0, 1, 3}
    while srv.poll():
        pass
    assert _landed(mem, lay, srv, 3) and srv.prefetched == 1
    srv.hint_top = 1                                # top 1: 0, in a slot: nothing
    _post(mem, lay, 2, [G + 0, G + 6])
    assert srv.poll() == 1 and not srv.pending and 6 not in srv.lru[0]
    srv.hint_n = srv.hint_top = 0                   # uncapped: every one not in a slot
    _post(mem, lay, 3, [G + 6, G + 7])
    assert srv.poll() == 1 and list(srv.pending) == [6, 7]


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
    """lines=2: a second line of the answer before the directory and a second line of ids on
    its own 128-byte block after it, the slots after that."""
    one, two = (Layout.build(4096, E, K, (3, 5), SLOT, lines=n) for n in (1, 2))
    assert one == Layout.build(4096, E, K, (3, 5), SLOT) and one.row2 == 0
    assert (one.mbox, one.served, one.answer) == (two.mbox, two.served, two.answer)
    assert two.dir == one.dir + LINE
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


def test_poll_pacer_sleeps_through_the_quiet_part_of_each_gap():
    """PollPacer (moe_card --poll-idle predict): after a request it sleeps share x the shortest
    of the last gaps after that kind of request (less its lateness), then polls; each request is
    still seen within a poll of its post, with far fewer polls than spinning. A gap shorter than
    expected is seen at the sleep's end and shortens the next sleep; nothing sleeps while hinted
    experts wait (pending), and a hinted part (last None) is not a request."""
    from opentpu.host.offload import PollPacer
    now = [0.0]
    POLL = 30e-6                            # a poll's card read

    class Card:                             # posts kind k's request gap[k] after the last served
        def __init__(self, gaps):
            self.gaps, self.k, self.due, self.last, self.pending = gaps, 0, 0.0, None, {}
            self.seen = []

        def poll(self):
            now[0] += POLL
            if now[0] < self.due:
                return 0
            self.seen.append(now[0] - self.due)
            self.last = ("d", self.k)
            now[0] += 1e-3                  # serving
            self.k = (self.k + 1) % len(self.gaps)
            self.due = now[0] + self.gaps[self.k]
            return 1

    def run(card, pacer, n):
        polls = 0
        while len(card.seen) < n:
            pacer.poll()
            polls += 1
        return polls

    def sleep(d):
        now[0] += d
    gaps = [2.7e-3, 1.5e-3, 3.0e-3]
    card = Card(gaps)
    pacer = PollPacer([card], sleep=sleep, clock=lambda: now[0])
    first = run(card, pacer, len(gaps) + 1)         # learning: one gap of each kind
    polls = run(card, pacer, 61)
    assert max(card.seen) <= POLL + 1e-12            # within a poll of the post, every time
    spin = sum(int(g / POLL) + 1 for g in gaps) * 20
    assert polls < 0.4 * spin and pacer.sleeps >= 55 and first > 0
    assert pacer.slept > 0.7 * 20 * sum(g * 0.75 - 2e-4 for g in gaps)
    card.gaps[1] = 0.8e-3                           # a shorter gap: seen at the sleep's end
    run(card, pacer, len(card.seen) + 6)
    late = max(card.seen[-6:])
    assert POLL < late < 0.75 * 1.5e-3
    run(card, pacer, len(card.seen) + 6)            # then learned: within a poll again
    assert max(card.seen[-3:]) <= POLL + 1e-12
    card.pending = {7: 0}                           # hinted experts waiting: no sleep
    run(card, pacer, len(card.seen) + 1)            # (after the sleep already set)
    s0 = pacer.sleeps
    run(card, pacer, len(card.seen) + 6)
    assert pacer.sleeps == s0
    with pytest.raises(ValueError, match="share"):
        PollPacer([card], share=1.0)


def _tag(mem, lay, slot):
    return int(np.frombuffer(mem.read(slot + lay.tag, 4), np.uint32)[0])


def _answer(mem, lay):
    return [int(v) for v in np.frombuffer(mem.read(lay.answer, LINE), np.uint32)]


def test_the_answer_names_the_missing_experts_slots_and_their_tags_land():
    """docs/offload.md 10.11: load zeroes every slot's tag and the answer. A request's answer
    holds each missing expert's slot at the id's place (0 for the hits); each one's tag reads
    g + 1 once poll returns. The card zeroes the tags of the experts it used and then the
    answer before its next post: tags the host wrote are armed until a later request comes."""
    lay, mem, srv = _setup(warm=[0, 1, 2])
    assert all(_tag(mem, lay, a) == 0 for a in lay.all_slots())
    assert _answer(mem, lay) == [0] * 16
    _post(mem, lay, 1, [2, 5])                      # 2 hits; 5 misses
    srv.poll()
    s5 = srv.lru[0][5]
    assert _answer(mem, lay) == [0, s5] + [0] * 14
    assert _tag(mem, lay, s5) == 6 and srv.armed == {s5}
    assert _tag(mem, lay, srv.lru[0][2]) == 0       # (a hit's tag is left alone)
    for g in (2, 5):                                # the card: each used expert's tag, then
        mem.write(srv.lru[0][g] + lay.tag, np.zeros(1, np.uint32))    # the answer, zeroed
    mem.write(lay.answer, np.zeros(16, np.uint32))
    _post(mem, lay, 2, [6, 2])                      # 6 misses
    srv.poll()
    s6 = srv.lru[0][6]
    assert srv.armed == {s6} and _answer(mem, lay) == [s6] + [0] * 15
    assert _tag(mem, lay, s6) == 7 and _tag(mem, lay, s5) == 0


def test_an_armed_victims_tag_is_cleared_before_its_slot_takes_another_expert():
    """A hinted expert that lands on idle polls (its tag with its last part) and that no
    request uses keeps its tag nonzero: the card never zeroed it. When its slot is a victim,
    its tag is cleared before the new expert's bytes, so the card cannot take the old tag for
    the new expert's."""
    lay, mem, srv, G = _hint_setup()
    _post(mem, lay, 1, [G + 6, G + 7])              # 6, 7 hinted: victims 2 and 1
    srv.poll()
    while srv.poll():                               # both land on idle polls
        pass
    s6 = srv.lru[0][6]
    assert _tag(mem, lay, s6) == 7 and s6 in srv.armed
    _post(mem, lay, 2, [0, 7])                      # the card zeroes 0's and 7's tags
    srv.poll()
    for g in (0, 7):
        mem.write(srv.lru[0][g] + lay.tag, np.zeros(1, np.uint32))
    log, w = [], mem.write

    def rec(addr, data, w=w):
        b = np.frombuffer(np.asarray(data).tobytes() if not isinstance(data, bytes) else data,
                          np.uint8)
        log.append(("tag", int(b[:4].view(np.uint32)[0])) if addr == s6 + lay.tag else
                   "slot" if addr == s6 else None)
        w(addr, data)
    mem.write = rec
    _post(mem, lay, 3, [0, 3])                      # 3 misses: the victim is 6 (never used)
    srv.poll()
    assert srv.lru[0][3] == s6
    assert [x for x in log if x] == [("tag", 0), "slot", ("tag", 4)]
    assert s6 in srv.armed and srv.lru[0][7] not in srv.armed


def test_an_all_hit_request_left_at_a_runs_end_is_served_before_the_slots_change():
    """A request whose experts are all present does not wait for the host, so the run that
    posted it can halt first (a layer-major prefill's last run). end_prefill and begin_prefill
    serve it before they change the slots, as the card ran it: all hits, no answer, no tags.
    Served after the restore it could name misses, and the next request would read their
    answer for its own (the card never zeroes it)."""
    for restore in ("lazy", "eager"):
        lay, mem, srv = _setup(slots=(2, 2, 2))
        srv.begin_prefill()
        for seq, run in enumerate(([0, 1, 1, 2], [1, 1, 2, 2], [8, 9, 9, 10]), 1):
            _post_n(mem, lay, seq, run)
            assert srv.poll() == 1
            mem.write(lay.answer, np.zeros(16, np.uint32))      # (the card, after its experts)
        _post_n(mem, lay, 4, [0, 0, 0, 0])              # all present; the run halts unseen
        srv.end_prefill(restore)
        assert srv.seq == 4 and _served(mem, lay) == 4.0 and srv.misses == 6
        assert _answer(mem, lay) == [0] * 16
        _consistent(mem, lay, srv)
        missing = [g not in srv.lru[0] for g in (0, 3)]
        _post_n(mem, lay, 5, [0, 3])                    # the next request's answer: its own
        assert srv.poll() == 1
        assert _answer(mem, lay)[:2] == [srv.lru[0][g] if m else 0
                                         for g, m in zip((0, 3), missing)]
        _consistent(mem, lay, srv)
    lay, mem, srv = _setup(warm=[0, 1, 2])
    _post(mem, lay, 1, [0, 1])                          # a step's last request, unseen
    srv.begin_prefill()
    assert srv.seq == 1 and srv.misses == 0 and _answer(mem, lay) == [0] * 16


@pytest.mark.parametrize("fmt", ["bytes", "split"])
@pytest.mark.parametrize("chash", [False, True])
def test_board_dram_sends_each_tag_as_its_experts_last_beat(fmt, chash, tmp_path):
    """The link's side of docs/offload.md 10.11, call by call: per request each missing
    expert's DMA calls (a fifth first for the request's first: two parts), its last part on
    the other channel and then on its tag's channel, the tag that call's last 64 bytes
    (nothing of the expert after it); the answer's beat right after the first expert's first
    part (its two calls); then the directory, then served."""
    from types import SimpleNamespace

    from opentpu.host.board import Board
    from opentpu.host.fake import FakeTransport
    from opentpu.host.offload import RUN, BoardDram, PoolFile, to_split
    slot = 3 * RUN
    lay = Layout.build(4096, 4, 2, (2, 3), slot)
    x = np.random.default_rng(1).integers(0, 256, (8, slot), dtype=np.uint8)
    pool = lambda g: x[g].tobytes()                 # noqa: E731
    if fmt == "split":
        f = tmp_path / "pool.bin"
        f.write_bytes(b"".join(to_split(r).tobytes() for r in x))
        pool = PoolFile(f, slot, split=True).get
    b = Board(FakeTransport(ch_bytes=1 << 20, devname=None))
    b.info()["caps"]["chash"] = chash
    m = BoardDram(SimpleNamespace(board=b), lay)
    srv = ExpertServer(m, lay, pool)
    srv.load([0, 1])
    calls, mw = [], b.t.mem_write

    def call(ch, off, data):
        calls.append((ch, off, len(data), bytes(np.asarray(data, np.uint8)[-64:])))
        mw(ch, off, data)
    b.t.mem_write = call

    def where(a):                                   # a beat's (channel, channel offset)
        return m._tag_channel(a), a // 128 * 64

    def covers(c, a):                               # call c writes the beat at a
        ch, off = where(a)
        return c[0] == ch and c[1] <= off < c[1] + c[2]
    for seq, ids in enumerate(([2, 3], [0, 3]), 1):
        b.write(lay.row, np.array(ids + [0] * 14, np.float32))
        b.write(lay.mbox, np.float32(seq).tobytes())
        calls.clear()                               # (the card's post: not the server's)
        miss = [g for g in ids if g not in srv.lru[0]]
        assert srv.poll() == 1
        ans = [i for i, c in enumerate(calls) if covers(c, lay.answer)]
        assert covers(calls[-1], lay.served)
        last = -1
        for g in miss:
            s = srv.lru[0][g]
            mine = [i for i, c in enumerate(calls)
                    if any(covers(c, a) for a in range(s, s + lay.tag + 128, 64))]
            if g == miss[0]:
                assert mine[0] == 0 and ans == [2]
            assert mine[0] > last and [i for i in range(mine[0], mine[-1] + 1)
                                       if i not in ans] == mine
            *_, other, final = mine
            ct, ot = where(s + lay.tag)
            assert calls[final][0] == ct and calls[other][0] == 1 - ct
            assert calls[final][1] + calls[final][2] == ot + 64       # the tag: its last beat
            assert calls[final][3][:4] == np.uint32(g + 1).tobytes()
            parts = 2 if g == miss[0] and m.direct > 2 else 1   # (the first: a fifth, the rest)
            assert len(mine) == 2 * parts
            last = mine[-1]
        rest = [c for i, c in enumerate(calls[:-1]) if i > last and i not in ans]    # the
        # directory's beats, after them all
        assert rest and all(any(covers(c, a) for a in range(lay.dir, lay.dir + 64, 64))
                            for c in rest)
    assert (m.direct > 2) == (fmt == "split" and chash)
    for g in srv.lru[0]:                            # the bytes as Board.write would have them
        assert bytes(b.read(srv.lru[0][g], slot)) == x[g].tobytes()


def test_a_multi_row_requests_answer_is_at_each_ids_first_place():
    """A layer-major run's request (moe.moe_ffn_rows: R k ids, repeats included, the count at
    mbox + 4): each missing expert's slot is at its id's first place in the request, where the
    card's union loop reads it; the repeats' words stay 0."""
    lay, mem, srv = _setup(slots=(4, 3), warm=[0, 1, 2])
    mem.write(lay.mbox + 4, np.float32(6).tobytes())
    _post(mem, lay, 1, [5, 2, 5, 6, 2, 6])          # rows (5, 2), (5, 6), (2, 6)
    srv.poll()
    assert srv.history is None and (srv.hits, srv.misses) == (1, 2)
    assert _answer(mem, lay) == [srv.lru[0][5], 0, 0, srv.lru[0][6]] + [0] * 12


def _ahead_setup(slots=(2, 2, 2), ahead=True):
    """A pooled prefill on layers of `slots` (empty), experts of three 4 KiB parts (the last
    128 bytes), an idle poll's part 4 KiB."""
    from opentpu.host.offload import RUN
    slot = 2 * RUN + 128
    lay = Layout.build(4096, E, K, slots, slot)
    mem = SimDram(np.zeros(lay.end + 4096, np.uint8))
    srv = ExpertServer(mem, lay, lambda g: np.full(slot, g + 1, np.uint8).tobytes(),
                       policy="lfu")
    srv.load()
    srv.begin_prefill(ahead=ahead, part=RUN)
    return lay, mem, srv


def _empty(mem, lay, g):
    return _entry(mem, lay, g) == (0, 0.0)


def test_ahead_layer_lands_a_layers_experts_on_idle_polls_most_wanted_first():
    """ahead_layer(j, ids): layer j's experts, most wanted first, each given a slot when its
    first part goes on an idle poll (a free one here), its tag with its last part, then its
    entry; a landed one joins the pooled order. With no slot left outside the layers the runs
    may read (j - 1 and j) the queue stops."""
    lay, mem, srv = _ahead_setup()
    _post_n(mem, lay, 1, [0, 1, 1, 0])              # layer 0's run: two of the six slots
    assert srv.poll() == 1
    srv.ahead_layer(1, [9, 8, 10, 11, 12])
    assert list(srv.queue) == [9, 8, 10, 11, 12] and not srv.pending
    for i in range(3):                              # 9: three parts, the tag with the last
        assert srv.poll() == 1
        assert _empty(mem, lay, 9) == (i < 2) and (9 in srv.pending) == (i < 2)
    s9 = srv.lru[1][9]
    assert _landed(mem, lay, srv, 9) and _tag(mem, lay, s9) == 10 and s9 in srv.armed
    while srv.poll():
        pass
    assert all(_landed(mem, lay, srv, g) for g in (9, 8, 10, 11))
    assert 12 not in srv.lru[1] and _empty(mem, lay, 12) and not srv.queue  # (no slot left)
    assert (srv.landed, srv.aheads) == (4, 1) and set(srv.lru[0]) == {0, 1}
    assert list(srv.order)[-4:] == [9, 8, 10, 11]


def test_a_new_ahead_layer_call_replaces_the_queue():
    """Each call replaces what the last one queued: those without a slot leave, the one on its
    way is dropped (its slot free again; its tag and entry never written)."""
    lay, mem, srv = _ahead_setup()
    srv.ahead_layer(1, [9, 8])
    assert srv.poll() == 1                          # 9's first part
    s9 = srv.lru[1][9]
    srv.ahead_layer(2, [17, 16])
    assert 9 not in srv.lru[1] and not srv.pending and list(srv.queue) == [17, 16]
    assert _empty(mem, lay, 9) and _tag(mem, lay, s9) == 0 and srv.dropped == 1
    while srv.poll():
        pass
    assert _landed(mem, lay, srv, 17) and _landed(mem, lay, srv, 16)
    assert 8 not in srv.lru[1] and _empty(mem, lay, 8)


def test_a_request_takes_a_queued_expert_as_a_miss_and_one_on_its_way_its_rest():
    """A run's request that names a queued expert takes it as an ordinary miss; one on its way
    is promoted: the rest of its bytes with its tag, its slot in the answer. Either way the
    queue does not send it again."""
    lay, mem, srv = _ahead_setup()
    srv.ahead_layer(1, [9, 8])
    assert srv.poll() == 1                          # 9 on its way, 8 queued
    _post_n(mem, lay, 1, [9, 8, 8, 9])
    assert srv.poll() == 1
    assert (srv.promoted, srv.misses) == (1, 2) and not srv.pending
    assert _answer(mem, lay)[:2] == [srv.lru[1][9], srv.lru[1][8]]
    assert _landed(mem, lay, srv, 9) and _landed(mem, lay, srv, 8)
    assert srv.poll() == 0 and not srv.queue


def test_ahead_victims_spare_the_layers_the_runs_may_read():
    """With no free slot, an ahead expert takes the oldest expert of a layer outside j - 1 and
    j (the runs until the next call read only those): what the running layer may read never
    changes under it."""
    lay, mem, srv = _ahead_setup(slots=(1, 1, 1, 1))
    for seq, g in enumerate((0, 8, 16, 24), 1):     # one run a layer: every slot taken
        _post_n(mem, lay, seq, [g])
        assert srv.poll() == 1
    srv.ahead_layer(1, [9])                         # spared: layers 0 and 1; 16 the oldest
    while srv.poll():
        pass
    assert _landed(mem, lay, srv, 9) and _empty(mem, lay, 16) and 16 not in srv.lru[2]
    assert all(_landed(mem, lay, srv, g) for g in (0, 8, 24))
    srv.ahead_layer(0, [1])                         # spared: layers 3 and 0; 8 the oldest left
    while srv.poll():
        pass
    assert _landed(mem, lay, srv, 1) and _empty(mem, lay, 8)


def test_a_hint_for_the_ahead_layer_adds_to_the_end_of_its_queue():
    """A hint for ahead_layer's layer (the card's router on the layer before it) adds its ids
    to the end of the queue in arrival order: once each, none in a slot or on its way, none
    queued already; it takes no slot (each when its first part goes) and is served as a hint.
    A hint for another layer is ignored (pooled), as is any hint with no queue."""
    lay, mem, srv = _ahead_setup()
    G = lay.E * lay.layers
    _post_n(mem, lay, 1, [0, 1])                    # layer 0's run
    assert srv.poll() == 1
    srv.ahead_layer(1, [9, 8])
    assert srv.poll() == 1 and 9 in srv.pending     # 9 on its way, 8 queued
    _post_n(mem, lay, 2, [g + G for g in (9, 10, 8, 10, 11)])
    assert srv.poll() == 1 and _served(mem, lay) == 2
    assert list(srv.queue) == [8, 10, 11] and srv.hinted_ahead == 2
    assert 10 not in srv.lru[1] and 11 not in srv.lru[1]
    _post_n(mem, lay, 3, [11 + G, 12 + G])
    assert srv.poll() == 1 and list(srv.queue) == [8, 10, 11, 12] and srv.hinted_ahead == 3
    _post_n(mem, lay, 4, [17 + G, 16 + G])          # layer 2's: not the queue's layer
    assert srv.poll() == 1 and _served(mem, lay) == 4
    assert list(srv.queue) == [8, 10, 11, 12] and 17 not in srv.lru[2] and srv.hints == 3
    while srv.poll():
        pass
    assert all(_landed(mem, lay, srv, g) for g in (9, 8, 10, 11))
    assert 12 not in srv.lru[1] and not srv.queue   # (no slot left outside layers 0 and 1)
    srv.end_prefill()
    lay, mem, srv = _ahead_setup(ahead=False)       # no queue: a hint as before (pooled)
    srv.ahead_layer(1, [9])
    _post_n(mem, lay, 1, [9 + G, 10 + G])
    assert srv.poll() == 1 and not srv.queue and srv.hinted_ahead == 0 and not srv.lru[1]


def test_ahead_layer_serves_the_last_request_first_and_end_prefill_drops_the_queue():
    """ahead_layer settles first (a run's all-hit request the host has not seen). end_prefill
    drops the queue and the expert on its way and gives the server its part back; a prefill
    begun with ahead=False ignores the calls."""
    from opentpu.host.offload import RUN
    lay, mem, srv = _ahead_setup()
    part = srv._part
    _post_n(mem, lay, 1, [0, 1])
    assert srv.poll() == 1
    _post_n(mem, lay, 2, [1, 0])                    # all hits: the run halts unseen
    srv.ahead_layer(1, [9, 8])
    assert srv.seq == 2 and srv.hits == 2
    assert srv.poll() == 1 and 9 in srv.pending
    srv.end_prefill()
    assert not srv.pending and not srv.queue and 9 not in srv.lru[1] and _empty(mem, lay, 9)
    assert srv.part == part != RUN and not srv.pooled
    lay, mem, srv = _ahead_setup(ahead=False)
    srv.ahead_layer(1, [9, 8])
    assert not srv.queue and srv.poll() == 0


def _em_setup(slots=(2, 2, 2, 2), scratch=0, warm=()):
    """An expert-major prefill (docs/offload.md 13.11) on layers of `slots`, experts of three
    4 KiB parts (the last 128 bytes), an idle poll's part 4 KiB; begin_prefill's scratch."""
    from opentpu.host.offload import RUN
    slot = 2 * RUN + 128
    lay = Layout.build(4096, E, K, slots, slot, lines=2)
    mem = SimDram(np.zeros(lay.end + 4096, np.uint8))
    srv = ExpertServer(mem, lay, lambda g: np.full(slot, g + 1, np.uint8).tobytes(),
                       policy="lfu")
    srv.load(list(warm))
    base = srv.begin_prefill(ahead=True, part=RUN, expert_major=True, scratch=scratch)
    return lay, mem, srv, base


def test_a_need_line_queues_its_experts_and_idle_polls_land_them_ahead_of_the_queue():
    """A need line (ids from Layout.need_off = 2 x layers x E up): served at once, no answer,
    no use counted, no slot taken; each id in no slot joins `needs` once, in arrival order.
    Idle polls send them before ahead_layer's queue: each takes a slot when its first part
    goes, its tag with its last part, then its entry. Outside an expert-major prefill a need
    line is an error."""
    lay, mem, srv, _ = _em_setup()
    N = lay.need_off
    assert N == 2 * lay.E * lay.layers
    srv.ahead_layer(1, [9, 8])
    _post_n(mem, lay, 1, [g + N for g in (1, 0, 1, 3)])
    assert srv.poll() == 1 and _served(mem, lay) == 1 and srv.last == ("n", 0)
    assert list(srv.needs) == [1, 0, 3] and not srv.lru[0] and not srv.pending
    assert _answer(mem, lay) == [0] * 16 and srv.use[0] == {} and srv.t[0] == 0
    assert (srv.need_lines, srv.needs_queued, srv.misses, srv.hits) == (1, 3, 0, 0)
    for i in range(9):                              # 1, 0, 3: three parts each
        assert srv.poll() == 1
    assert all(_landed(mem, lay, srv, g) for g in (1, 0, 3)) and not srv.needs
    assert _tag(mem, lay, srv.lru[0][1]) == 2 and not srv.lru[1] and srv.needs_landed == 3
    while srv.poll():                               # then the queue's
        pass
    assert _landed(mem, lay, srv, 9) and _landed(mem, lay, srv, 8)
    assert list(srv.order)[-5:] == [1, 0, 3, 9, 8]
    _post_n(mem, lay, 2, [g + N for g in (1, 2)])   # 1 in a slot: a hit
    assert srv.poll() == 1 and list(srv.needs) == [2] and srv.need_hits == 1
    lay, mem, srv = _ahead_setup()                  # layer-major, not expert-major
    _post_n(mem, lay, 1, [lay.need_off])
    with pytest.raises(RuntimeError, match="need line outside"):
        srv.poll()


def test_a_need_takes_over_an_ahead_expert_and_no_ahead_layer_call_drops_it():
    """A need naming an expert ahead_layer queued moves it to `needs`; one ahead_layer has on
    its way becomes a need's, so the next call (which drops its own on their way) keeps it.
    A hint for the queue's layer does not queue a need again."""
    lay, mem, srv, _ = _em_setup()
    N, G = lay.need_off, lay.E * lay.layers
    srv.ahead_layer(1, [9, 8, 10])
    assert srv.poll() == 1 and 9 in srv.pending     # 9's first part
    _post_n(mem, lay, 1, [9 + N, 8 + N])
    assert srv.poll() == 1
    assert list(srv.needs) == [8] and list(srv.queue) == [10] and srv.need_hits == 1
    _post_n(mem, lay, 2, [8 + G, 11 + G])           # the queue's layer: 11 only
    assert srv.poll() == 1 and list(srv.queue) == [10, 11]
    srv.ahead_layer(2, [17])
    assert 9 in srv.pending and srv.dropped == 0 and list(srv.queue) == [17]
    while srv.poll():
        pass
    assert all(_landed(mem, lay, srv, g) for g in (9, 8, 17)) and srv.needs_landed == 2
    assert 10 not in srv.lru[1] and 11 not in srv.lru[1]


def test_expert_major_victims_spare_the_running_layer_and_its_neighbours():
    """Expert-major, every slot taken during layer j's runs (a need's, a request's miss, an
    ahead expert's) has its victim outside layers j - 1, j and j + 1, the running layer the
    last need line's or request's. A need with no such victim is an error, not dropped."""
    lay, mem, srv, _ = _em_setup(slots=(1, 1, 1, 1, 1))
    N = lay.need_off
    for seq, g in enumerate((0, 8, 16, 24, 32), 1):     # one need a layer: every slot taken
        _post_n(mem, lay, seq, [g + N])
        assert srv.poll() == 1
        while srv.poll():
            pass
    _post_n(mem, lay, 6, [1 + N])                   # layer 0: spared 4, 0, 1; 16 the oldest
    srv.poll()
    while srv.poll():
        pass
    assert _landed(mem, lay, srv, 1) and _empty(mem, lay, 16) and 16 not in srv.lru[2]
    _post_n(mem, lay, 7, [9 + N])                   # layer 1: spared 0, 1, 2; 24 the oldest
    srv.poll()
    while srv.poll():
        pass
    assert _landed(mem, lay, srv, 9) and _empty(mem, lay, 24)
    _post_n(mem, lay, 8, [17])                      # a request of layer 2: spared 1, 2, 3
    assert srv.poll() == 1 and srv.misses == 1
    assert _landed(mem, lay, srv, 17) and _empty(mem, lay, 0) and 32 in srv.lru[4]
    srv.ahead_layer(3, [25])                        # ahead: also outside 2 and 3; 32 is layer 4
    while srv.poll():
        pass
    assert _landed(mem, lay, srv, 25) and _empty(mem, lay, 32)
    lay, mem, srv, _ = _em_setup(slots=(1, 1, 1))
    N = lay.need_off
    for seq, g in enumerate((0, 8, 16), 1):
        _post_n(mem, lay, seq, [g + N])
        srv.poll()
        while srv.poll():
            pass
    _post_n(mem, lay, 4, [9 + N])                   # every layer spared
    with pytest.raises(RuntimeError, match="need 9"):
        while srv.poll():
            pass


def test_the_next_layers_need_line_disarms_the_tags_the_expert_run_zeroed():
    """The first need line of layer j + 1 means layer j's expert run is done, and it zeroed
    the tags of the experts its need lines named: their slots are no longer armed, so a slot
    reused later does not cost a write of a zero tag first."""
    lay, mem, srv, _ = _em_setup()
    N = lay.need_off
    _post_n(mem, lay, 1, [0 + N, 1 + N])
    srv.poll()
    while srv.poll():
        pass
    s0, s1 = srv.lru[0][0], srv.lru[0][1]
    assert {s0, s1} <= srv.armed
    for s in (s0, s1):                              # the expert run zeroes them
        mem.write(s + lay.tag, np.zeros(1, np.uint32))
    _post_n(mem, lay, 2, [8 + N])
    srv.poll()
    assert s0 not in srv.armed and s1 not in srv.armed


def _em_pitch():
    """_em_setup's slot pitch (two 4 KiB blocks and a tag chunk, in whole blocks)."""
    from opentpu.host.offload import RUN
    return Layout.build(4096, E, K, (1,), 2 * RUN + 128, lines=2).pitch


def test_end_prefill_drains_the_needs_and_hands_the_scratch_back_with_zeroed_tags():
    """begin_prefill(scratch=n): the slot region's first ceil(n / pitch) slots (Layout.scratch,
    as moe.em_slots: here layer 0's two and layer 1's first) leave the free lists, their
    experts evicted with their entries cleared on the card before the base is returned, and no
    need or miss is given one. end_prefill sends every need still queued or on its way whole
    (drained), zeroes the scratch's tag beats (the card's scratch data there would read as a
    landed expert's tag) and gives each layer its slots back."""
    lay, mem, srv, base = _em_setup(scratch=2 * _em_pitch() + 1, warm=(0, 1))
    p, a0, a1 = lay.pitch, lay.slots[0][0], lay.slots[1][0]
    scratch = [a0, a0 + p, a1]
    assert base == a0 == lay.scratch(2 * p + 1)[0] and srv.scratch_slots == scratch
    assert _empty(mem, lay, 0) and _empty(mem, lay, 1) and not srv.lru[0]
    assert not srv.free[0] and srv.free[1] == [a1 + p] and 0 not in srv.order
    mem.write(base, np.full(3 * p, 0xAB, np.uint8))     # the card's scratch
    N = lay.need_off
    for seq, ids in enumerate(([8, 9], [16, 17], [24, 25, 26]), 1):
        _post_n(mem, lay, seq, [g + N for g in ids])
        assert srv.poll() == 1
    assert len(srv.needs) == 7 and srv.poll() == 1  # 8's first part only
    while len(srv.needs) > 1:                       # (up to 25: a victim of layer 1)
        srv.poll()
    assert not {a for lru in srv.lru for a in lru.values()} & set(scratch)
    srv.end_prefill()
    assert not srv.needs and not srv.pending and 0 < srv.drained < 7 and srv.needs_landed == 7
    assert all(_tag(mem, lay, a) == 0 for a in scratch)
    assert not srv.scratch_slots and not srv.expert_major and not srv.pooled
    _consistent(mem, lay, srv)
    back = {a for lru in srv.lru for a in lru.values()} | {a for fr in srv.free for a in fr}
    assert set(scratch) <= back and back == set(lay.all_slots())


def test_need_report_names_the_running_layers_experts_without_an_entry():
    """For an expert run's WAITW timeout: the running layer's needed experts whose entries are
    not present, and where each stands."""
    lay, mem, srv, _ = _em_setup()
    N = lay.need_off
    assert "no need line" in srv.need_report()
    _post_n(mem, lay, 1, [1 + N, 0 + N, 3 + N])
    srv.poll()
    srv.poll()                                      # 1's first part
    r = srv.need_report()
    assert r.startswith("layer 0: 3 experts needed, 3 without an entry")
    assert "1 (expert 1): on its way, 4096 of" in r and "0 (expert 0): queued" in r
    while srv.poll():
        pass
    assert srv.need_report() == "layer 0: 3 experts needed, 0 without an entry"


def test_the_isa_hosts_hook_polls_until_the_expert_runs_wait_holds():
    """qwen3._isa_host, the ISA simulator's WAITW hook (called once when every slice waits; a
    wait still not holding after it is the timeout): it polls until a waiting slice's WAITW
    holds, as the host keeps polling while a card waits. An expert run waiting on a need's
    entry takes one idle poll a part. With nothing to do it stops after one poll."""
    from types import SimpleNamespace
    from opentpu.llm.qwen3 import _isa_host
    lay, mem, srv, _ = _em_setup()
    N = lay.need_off
    _post_n(mem, lay, 1, [3 + N, 1 + N])

    class Pass:                                     # the expert run, waiting on 1's entry
        polling = "WAITW"

        def holds(self, ins):
            return _entry(mem, lay, 1)[0] != 0
    m = SimpleNamespace(slices=[Pass()])
    _isa_host(srv.poll)(m)
    assert _landed(mem, lay, srv, 3) and _landed(mem, lay, srv, 1)    # (3 first: in order)
    assert not srv.needs and not srv.pending and srv.needs_landed == 2
    calls = []
    _isa_host(lambda: calls.append(1) or 0)(m)
    assert calls == [1]
