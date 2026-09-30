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
