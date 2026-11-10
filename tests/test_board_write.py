"""Board.write's pipelined split (each piece gathered straight into its channels' buffers by the
split threads) and BoardBackend's image ranges (images.runs: the scrub leaves them, only they are
written after it), on fake cards: no card, no RTL."""
import numpy as np
import pytest

from opentpu.host import board as B
from opentpu.host import fake as F
from opentpu.host import regs as R
from opentpu.host.board import Board, BoardBackend, device_config, keep_ranges


def _cfg(S: int, nch: int, beat: int, ch_bytes: int) -> dict:
    """The CFG_* registers of a build of a platform this tree has not (by_platform None: no
    target checked, no DDR4 cores' CSRs): S slices of nch channels of ch_bytes, beat-byte
    beats, CFG_HASH 0 (unchecked)."""
    from opentpu.isasim import board_config
    c = board_config()
    return dict(F._target_cfg(c.D, c.MCOLS, c.LANES, False, False, False, c.act_rows), hash=0,
                S=S, channels=nch, channel_bytes=ch_bytes, DRAM_BYTES=nch * ch_bytes,
                beat_bytes=beat, platform=0xEE)


class Card(F.FakeTransport):
    """The fake card of _cfg's build, its DRAM random at first, every DMA write logged
    (channel, offset, bytes). With ecc a card Board.scrub writes: a read of a 64-byte beat no
    write has reached raises (the card's uncorrectable error), so does a write of part of a beat
    (XdmaTransport reads its edge beats first), and each beat's writes are counted."""

    def __init__(self, S=1, nch=4, beat=64, ch_bytes=1 << 20, chash=True, threaded=True,
                 ecc=False, seed=0):
        super().__init__(devname=None, regmap=4, cfg=_cfg(S, nch, beat, ch_bytes),
                         ch_bytes=ch_bytes, chash=chash)
        rng = np.random.default_rng(seed)
        for c in self.ch:
            c[:] = rng.integers(0, 256, len(c), dtype=np.uint8)
        self.threaded, self.ecc, self.bases = threaded, ecc, B.BASE
        self.calls: list = []
        self.wrote = [np.zeros(len(c) // 64, np.int32) for c in self.ch]

    def mem_write(self, ch, off, data):
        b0, b1 = off // 64, -(-(off + len(data)) // 64)
        if self.ecc and (off % 64 or len(data) % 64) and not self.wrote[ch][[b0, b1 - 1]].all():
            raise AssertionError(f"a write of part of channel {ch}'s beat never written at "
                                 f"{off:#x}")
        self.calls.append((ch, off, len(data)))
        self.wrote[ch][b0:b1] += 1
        super().mem_write(ch, off, data)

    def mem_read(self, ch, off, n, out=None):
        if self.ecc and not self.wrote[ch][off // 64:-(-(off + n) // 64)].all():
            raise AssertionError(f"a read of channel {ch}'s beats never written at {off:#x}")
        return super().mem_read(ch, off, n, out)


def _same(a: Card, b: Card) -> bool:
    return all(np.array_equal(x, y) for x, y in zip(a.ch, b.ch))


# ------------------------------------------------------------------------------ the split
@pytest.mark.parametrize("chash", [False, True])
@pytest.mark.parametrize("nch,beat", [(2, 64), (4, 64), (2, 128), (4, 128), (8, 128)])
def test_pipelined_write_is_the_one_call_writes_bytes(nch, beat, chash, monkeypatch):
    """Board.write on a threaded card (pipelined: pieces of PIPE / 2 bytes a channel, gathered
    straight into their buffers by the split threads, PIPE_SETS sets) leaves every channel of
    both slices as the one-call path of an unthreaded card leaves it, byte for byte: unaligned
    starts and ends (the edges' beats), a write of PIPE and a chunk, pieces of whole and part
    blocks of gather's tables (4,096 chunks) at blocks whose index's fold is not 0, a last
    piece short, writes over earlier ones. The pieces' DMA calls are whole beats, at most PIPE /
    2 bytes, and hash_swap copies nothing; a run in flight still writes RUN_H2C a call."""
    bl = 4096
    monkeypatch.setattr(B, "PIPE", 2 * (bl + bl // 2) * beat)  # 1.5 blocks a channel a piece
    monkeypatch.setattr(B, "SPLIT_THREADS", 3)
    P, h = nch * beat, B.PIPE // 2
    piece = nch * h
    a, b = (Card(S=2, nch=nch, beat=beat, ch_bytes=6 * h, chash=chash, threaded=t)
            for t in (True, False))
    ba, bb = Board(a, lock=False), Board(b, lock=False)
    assert (ba.nch, ba.beat, ba.chash, ba.slices) == (nch, beat, chash, 2)
    top = nch * 6 * h                                   # a slice's logical bytes
    swaps, hs = [], B.hash_swap
    monkeypatch.setattr(B, "hash_swap", lambda ad, v: (swaps.append(v.nbytes), hs(ad, v))[1])
    rng = np.random.default_rng(nch * beat + chash)
    writes = [(0, 3 * P + 77, 2 * piece + 5 * P + 300),     # unaligned, two pieces and a part
              (1, piece, piece),                            # one whole piece
              (0, 2 * piece + 4 * P, B.PIPE + P),           # PIPE and a chunk
              (1, 5, B.PIPE - 3),                           # (not pipelined: the same path)
              (0, top - piece - 3 * P - 321, piece + 3 * P + 321),     # to the slice's end
              (1, top // 2 + 64 * 3 + 1, piece // 2 + 2 * P + 9),   # part of a piece
              (0, P, 3 * piece)]                            # over the first ones
    for s, addr, n in writes:
        d = rng.integers(0, 256, n, dtype=np.uint8)
        k, c0 = len(swaps), len(a.calls)
        ba.write(addr, d, s)
        k = len(swaps) - k
        bb.write(addr, d, s)
        assert _same(a, b), (s, addr, n)
        if n > B.PIPE:                                      # (pipelined)
            assert k == 0
            whole = [c for c in a.calls[c0:] if c[2] > beat]
            assert whole and all(o % beat == 0 and m % beat == 0 and m <= h
                                 for _, o, m in whole)
            assert {ch for ch, _, _ in a.calls[c0:]} == set(range(nch * s, nch * s + nch))
    for c in (a, b):
        c.run_h2c = 2048
    ba.in_run = bb.in_run = True
    k = len(a.calls)
    d = rng.integers(0, 256, piece + 1000, dtype=np.uint8)
    ba.write(7 * P, d, 1)
    bb.write(7 * P, d, 1)
    ba.in_run = bb.in_run = False
    assert _same(a, b) and max(m for _, _, m in a.calls[k:]) <= 2048
    ba.close()
    bb.close()


def test_write_chunks_is_writes_of_each_piece(monkeypatch):
    """Board.write_chunks: whole wide chunks at several addresses (some shorter than a piece,
    some of several) through one pipeline on a threaded card, each written as write() writes it
    (an unthreaded card, piece by piece); the calls in the pieces' order; a piece off the
    chunk grid is refused."""
    monkeypatch.setattr(B, "PIPE", 2 * 1024 * 64)
    a, b = Card(nch=4, beat=64, ch_bytes=1 << 20), Card(nch=4, beat=64, ch_bytes=1 << 20,
                                                        threaded=False)
    ba, bb = Board(a, lock=False), Board(b, lock=False)
    P = 256
    rng = np.random.default_rng(3)
    pieces = [(x * P, rng.integers(0, 256, n * P, dtype=np.uint8))
              for x, n in ((0, 1), (5, 3), (100, 4096), (5000, 1), (9000, 700), (12000, 2))]
    ba.write_chunks(pieces)
    for x, d in pieces:
        bb.write(x, d)
    assert _same(a, b)
    offs = [o for ch, o, _ in a.calls if ch == 0]
    assert offs == sorted(offs) and len(offs) == 1 + 1 + 4 + 1 + 1 + 1
    with pytest.raises(ValueError, match="whole wide chunks"):
        ba.write_chunks([(P // 2, np.zeros(P, np.uint8))])
    ba.close()
    bb.close()


# ------------------------------------------------------------------------------ image runs
def test_keep_ranges():
    """A slice's runs -> the scrub's kept ranges: each run's part inside the spans (the bytes
    a backend writes), widened to the alignment, joined when less than the gap apart, within
    the slice's DRAM; a run outside every span keeps nothing."""
    K = 1 << 10
    spans = [(0, 10 * K + 5), (64 * K, 100 * K), (120 * K, 300 * K + 7)]
    runs = [(100, 200), (5 * K, 12 * K),               # (its part to the span's end)
            (20 * K, 30 * K),                          # (between the spans: none)
            (66 * K + 1, 67 * K), (69 * K, 70 * K),    # joined (3 K apart < 4 K)
            (90 * K, 125 * K),                         # across two spans: a range in each
            (299 * K, 400 * K)]                        # past the last span and the DRAM top
    got = keep_ranges(runs, spans, 256, 302 * K, align=2 * K, gap=4 * K)
    assert got == [(0, 12 * K), (66 * K, 70 * K), (90 * K, 100 * K), (120 * K, 126 * K),
                   (298 * K, 302 * K)]
    assert keep_ranges([(20 * K, 30 * K)], spans, 256, 302 * K) == []
    assert keep_ranges(runs, spans, 256, 302 * K, align=2 * K, gap=21 * K) == \
        [(0, 12 * K), (66 * K, 126 * K), (298 * K, 302 * K)]


def _image(seed: int, n: int, end: int, high: int, skip: tuple) -> np.ndarray:
    """A slice's image array: zeros, random bytes in a few places, some across a span's edge
    (to the low area's end, the skip's edges, the array's end) and some outside every span
    (past the low area's end, inside the skip: bytes a backend never writes)."""
    rng = np.random.default_rng(seed)
    m = np.zeros(n, np.uint8)
    M = 1 << 20
    for lo, hi in ((0, 2000), (300 << 10, (300 << 10) + 5), (end - 100, end + 50),
                   (2 * M, 2 * M + 4096), (high, high + (10 << 10)), (5 * M, skip[0] + 1000),
                   (7 * M, 7 * M + (64 << 10)), (skip[1] - 77, 9 * M), (11 * M, n)):
        m[lo:hi] = rng.integers(1, 256, hi - lo, dtype=np.uint8)
    return m


def _page_runs(m: np.ndarray) -> list:
    """The 4 KiB pages of m holding a non-zero byte, as runs (an image cache entry's: the
    whole array's, outside its spans too)."""
    pages = np.flatnonzero(np.add.reduceat(m != 0, np.arange(0, len(m), 4096)))
    runs: list = []
    for p in pages:
        if runs and runs[-1][1] == p * 4096:
            runs[-1][1] = min(len(m), p * 4096 + 4096)
        else:
            runs.append([p * 4096, min(len(m), p * 4096 + 4096)])
    return [tuple(r) for r in runs]


def _backend(card, images):
    b = Board(card, lock=False)
    info = b.info()
    b.close()
    cfg = device_config(info, DRAM_BYTES=info["cfg"]["DRAM_BYTES"])
    return BoardBackend(cfg, images, transport=card, status=False)


@pytest.mark.parametrize("S,nch,beat,threaded,align", [(1, 4, 64, True, None),
                                                       (2, 4, 64, True, 64 << 10),
                                                       (1, 8, 128, True, 64 << 10),
                                                       (2, 4, 64, False, 64 << 10)])
def test_image_runs_are_written_once_after_a_scrub_that_leaves_them(S, nch, beat, threaded,
                                                                    align, monkeypatch):
    """BoardBackend with images.runs (each slice's non-zero pages, those outside the spans
    too) on a card with ECC: the scrub zeroes all but the runs' ranges (keep_ranges: inside the
    spans, widened, joined), the backend writes only those (Board.write_chunks: m's bytes inside
    the spans, the chunks of the spans' unaligned edges built, zeros elsewhere), and the DRAM is
    what the scrub of it all and the spans' writes leave, byte for byte, every 64-byte beat of
    it written exactly once and none read before it was. Default alignment and gap (the slice's
    image one range) and 64 KiB ones with a 128 KiB gap (many); an unthreaded card writes
    the ranges one write() each."""
    from opentpu.llm.qwen3 import Images
    if align is not None:
        monkeypatch.setattr(B, "KEEP_ALIGN", align)
        monkeypatch.setattr(B, "KEEP_GAP", 2 * align)
    M = 1 << 20
    end, high, n = M + 1000, 3 * M, 12 * M + 300
    skip = (6 * M + 197, 8 * M + 777)
    arrays = [_image(s, n, end, high, skip) for s in range(S)]
    kw = dict(S=S, nch=nch, beat=beat, ch_bytes=16 * M // nch, threaded=threaded, ecc=True)
    a, ref = Card(**kw), Card(**kw)
    im = Images(arrays, end, high, skip)
    im.runs = [_page_runs(m) for m in arrays]
    be = _backend(a, im)
    keep = be._keep(im, im.runs)
    assert len(keep) == S and (len(keep[0]) > 4) == (align is not None)
    be.close()
    _backend(ref, Images(arrays, end, high, skip)).close()
    assert _same(a, ref)
    assert all((w == 1).all() for w in a.wrote)
    assert sum(m for _, _, m in a.calls) == S * 16 * M
    for s in range(S):                                  # (and the image is where it belongs)
        b = Board(a, lock=False)
        for lo, hi in im.spans(n):
            assert np.array_equal(b.read(lo, hi - lo, s), arrays[s][lo:hi])
        assert not b.read(end, high - end, s).any() and not b.read(skip[0], skip[1] - skip[0],
                                                                   s).any()
        b.close()


@pytest.mark.parametrize("why", ["scrubbed", "no ecc"])
def test_image_runs_without_a_scrub_write_the_spans(why):
    """A scrub skipped (SCRATCH = DRAM_INIT: the DRAM holds an earlier image's bytes) or a card
    without ECC (the simulated boards: no scrub): BoardBackend with images.runs writes the
    whole spans, as without them: the same DRAM and the same DMA calls."""
    from opentpu.llm.qwen3 import Images
    M = 1 << 20
    end, high, n = M + 1000, 3 * M, 12 * M + 300
    skip = (6 * M + 197, 8 * M + 777)
    arrays = [_image(7, n, end, high, skip)]
    cards = [Card(ch_bytes=4 * M, ecc=why == "scrubbed") for _ in range(2)]
    for c in cards:
        c.regs[R.R_SCRATCH] = R.DRAM_INIT
        c.wrote = [np.ones_like(w) for w in c.wrote]     # (written by the earlier image)
    im = Images(arrays, end, high, skip)
    im.runs = [_page_runs(arrays[0])]
    _backend(cards[0], im).close()
    _backend(cards[1], Images(arrays, end, high, skip)).close()
    assert _same(*cards) and cards[0].calls == cards[1].calls
    assert sum(m for _, _, m in cards[0].calls) == sum(hi - lo for lo, hi in im.spans(n))


def test_image_runs_with_the_load_centring(monkeypatch):
    """The first scrub of a configuration that centres the read taps under load (loadcal): its
    first pass zeroes every beat (the load's reads may be of any), the centring writes where it
    likes, the second pass leaves the runs' ranges and the image's write covers them: the DRAM
    is the full scrub's and spans' write's."""
    from opentpu.host import loadcal
    from opentpu.llm.qwen3 import Images
    M = 1 << 20
    end, high, n = M + 1000, 3 * M, 12 * M + 300
    skip = (6 * M + 197, 8 * M + 777)
    arrays = [_image(9, n, end, high, skip)]
    seen = []

    def center(b, log=print):
        seen.append(all(w.all() for w in b.t.wrote))
        b.write(0, np.full(16 * M, 0xA5, np.uint8))     # (the BIST's and the load's bytes)
    monkeypatch.setattr(loadcal, "wanted_on", lambda b: True)
    monkeypatch.setattr(loadcal, "center", center)
    a, ref = Card(ch_bytes=4 * M, ecc=True), Card(ch_bytes=4 * M, ecc=True)
    im = Images(arrays, end, high, skip)
    im.runs = [_page_runs(arrays[0])]
    _backend(a, im).close()
    _backend(ref, Images(arrays, end, high, skip)).close()
    assert seen == [True, True] and _same(a, ref)
