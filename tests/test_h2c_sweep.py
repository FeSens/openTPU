"""tools/h2c_sweep.py's bookkeeping, on a file in place of the card's h2c device: placed buffers,
the points' calls and rates from several threads, the channels' addresses, the read-back check."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import h2c_sweep as S  # noqa: E402

from opentpu.host import board as B  # noqa: E402


def test_buffers_are_placed_for_their_card_address():
    for n in (64 << 10, 834_944):
        for i in range(3):
            for thp in (True, False):
                b = S.alloc(n, S.card_addr(0, i), thp)
                assert len(b) == n
                assert (B._addr(b) - S.card_addr(1, i)) % 4096 == B.DMA_PLACE


def test_points_write_every_call_where_its_channel_says_and_the_check_reads_it_back(tmp_path):
    """Three threads, alternating channels, on a sparse file: each thread's region of both channels
    holds its buffer, the rate counts every call, and a flipped byte fails the check."""
    f = tmp_path / "h2c"
    fd = os.open(f, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        n = 256 << 10
        bufs = [S.alloc(n, S.card_addr(0, i), True) for i in range(3)]
        p = S.run_point(fd, bufs, n, "alt", 0.05)
        assert p["calls"] >= 9 and p["gbs"] > 0 and p["engine_gbs"] is None
        assert all(c in (0, 1) for c in p["last_ch"])

        def mem_read(ch, off, k):
            return np.frombuffer(os.pread(fd, k, B.BASE[ch] + off), np.uint8)
        t = SimpleNamespace(mem_read=mem_read)
        for i, b in enumerate(bufs):
            for ch in (0, 1):
                assert np.array_equal(mem_read(ch, S.card_addr(ch, i) - B.BASE[ch], n), b)
        assert S.check(t, bufs, n, p["last_ch"]) is None
        os.pwrite(fd, bytes([bufs[1][n - 1] ^ 1]), S.card_addr(p["last_ch"][1], 1) + n - 1)
        assert "thread 1" in S.check(t, bufs, n, p["last_ch"])

        one = S.run_point(fd, bufs[:1], n, "1", 0.02)
        assert "engine_gbs" in one and one["last_ch"] == [1] and one["cpu_us"] <= one["wall_us"]
    finally:
        os.close(fd)


class _FileCard:
    """XdmaTransport's surface that h2c_sweep uses, on a sparse file."""
    def __init__(self, path):
        self.h2c = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        self._dma = B._DmaLock("", flock=False)

    def reg_read(self, off):
        return 0

    def mem_write(self, ch, off, data):
        assert B._write_ok(B._addr(data), B.BASE[ch] + off) is False    # the bounced case
        os.pwrite(self.h2c, np.ascontiguousarray(data).tobytes(), B.BASE[ch] + off)

    def mem_read(self, ch, off, n, out=None):
        b = np.frombuffer(os.pread(self.h2c, n, B.BASE[ch] + off), np.uint8)
        if out is None:
            return b.copy()
        out[:] = b
        return out

    def close(self):
        os.close(self.h2c)


def test_the_sweep_runs_its_grid_and_writes_its_json(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(B, "XdmaTransport", lambda dev: _FileCard(tmp_path / "card"))
    assert S.main(["--quick", "--secs", "0.01", "--qd", "1,2", "--json", str(tmp_path / "o.json")]) == 0
    out = json.loads((tmp_path / "o.json").read_text())
    assert out["result"] == "PASS"
    assert len(out["points"]) == len(S.QUICK) * 3 * 2 * 2
    assert {(p["bytes"], p["ch"], p["qd"], p["thp"]) for p in out["points"]} == \
        {(n, c, q, h) for n in S.QUICK for c in ("0", "1", "alt") for q in (1, 2) for h in (True, False)}
    assert [r["bytes"] for r in out["bounced"]] == S.QUICK and not any(r["placed"] for r in out["bounced"])
    assert [r["bytes"] for r in out["c2h"]] == S.QUICK
