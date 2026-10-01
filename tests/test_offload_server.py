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


@pytest.mark.parametrize("hints", [False, True])
@pytest.mark.parametrize("chash", [False, True])
@pytest.mark.parametrize("slot,fmt,pieces", [(128 * 37, "bytes", 2), (64 * 75, "bytes", 2),
                                             (4096 * 3, "split", 3), (4096 * 3, "split", 1),
                                             (128 * 37, "split", 2)])
def test_board_dram_writes_what_board_write_writes(chash, slot, fmt, pieces, hints, tmp_path):
    """BoardDram (the card's fast path: one-pass channel runs, a worker thread's DMA, the
    host's words from a shadow without reading the card) leaves the card's two channel memories
    exactly as BackendDram's Board.write does, request after request: the slots (with CHASH's
    swaps or not), the directory and served (each entry and served one DMA call: its beat). A
    slot of a half chunk falls back to Board.write.
    split: the pool a file in the split format, read with preadv straight into the channel
    runs under CHASH at a page-aligned slot (pages of either parity; a slot of 37 chunks is
    every other slot off a page: the slot's bytes there, as without CHASH); a request's first
    miss in `pieces` parts, each DMAed when it is read.
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
    if fmt == "split":
        f = tmp_path / "pool.bin"
        f.write_bytes(b"".join(to_split(pool(g)).tobytes() for g in range(8)))
        fast_pool = PoolFile(f, slot, split=True).get
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
    assert (fast.mem.direct > 0) == (fmt == "split" and chash)
    if not hints:
        assert any(dmas) == (fmt == "split" and chash and pieces > 1)     # a part's DMA
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
    srv = ExpertServer(BoardDram(SimpleNamespace(board=b), lay, depth=1, pieces=3), lay,
                       PoolFile(f, slot, split=True).get)
    srv.load(range(8))
    for g in srv.lru[0].keys() | srv.lru[1].keys():
        s = srv.lru[g // 4][g]
        assert np.array_equal(np.asarray(b.read(s, slot)).view(np.uint8), x[g]), g


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
