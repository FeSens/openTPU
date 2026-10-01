"""opentpu/host/xmon.py: the debug build's self-describing data and register reads (the RTL's own
bench: tools/xmon_test.py)."""
import re
from pathlib import Path

import numpy as np
import pytest

from opentpu.host import xmon as X

ROOT = Path(__file__).resolve().parents[1]


def test_data_describes_its_card_address():
    d = X.data(1, 0x1000, 64, tag=7).view(np.uint32).reshape(-1, 4)
    assert d[:, 0].tolist() == [X.MAGIC] * 4
    assert d[:, 1].tolist() == [0x8000_1000 + 16 * i for i in range(4)]
    assert (d[:, 2] == ~d[:, 1]).all() and d[:, 3].tolist() == [7] * 4
    assert X.first_bad(X.data(1, 0x1000, 64, tag=7), 1, 0x1000) is None


def test_first_bad_finds_a_64_byte_slip():
    good = X.data(0, 0x2000, 256)
    got = np.concatenate([good[:64], X.data(0, 0x2000, 192)])     # 64 B behind from beat 4
    assert X.first_bad(got, 0, 0x2000) == (64, 0x2000)
    junk = good.copy()
    junk[128:144] = 0
    assert X.first_bad(junk, 0, 0x2000) == (128, -1)


class _Regs:
    """otpu_xmon's register side: SNAP acknowledged at once by both clocks."""
    def __init__(self, words):
        self.w, self.snaps = dict(words), 0

    def reg_write(self, off, val):
        assert off == X.REG
        if val & 1:
            self.snaps = (self.snaps + 1) & 0xFF
        if val & 2:
            self.w = {k: (0x584D0000 if k == 0 else 0) for k in self.w}

    def reg_read(self, off):
        k = (off - X.REG) // 4
        return self.snaps * 0x10101 if k == 1 else self.w.get(k, 0)

    def reg_read_many(self, offs):
        return [self.reg_read(o) for o in offs]


def test_snap_and_describe():
    w = {0: 0x584D_0101, 2: 5, 3: 80, 9: 1, 11: 0x0010_0040, 12: 0x0010_0000, 13: 9, 26: 1,
         28: 0x0010_0040, 29: 0x0010_0000, 16: 0x0108_9749, 17: 0x0F08_0004}
    t = _Regs(w)
    assert X.present(t) and X.flag_names(X.flags(t)) == ["X_WSHIFT", "N_WSHIFT"]
    s = X.snap(t)
    assert s["X_AW"] == 5 and s["SNAPS"] & 0xFF == 1
    text = X.describe(s)
    assert "first bad W beat: at 0x00100040 it held 0x00100000 (-64 B), tag 0x9" in text
    assert "first bad channel-0 W beat: port 0, command beat 0x00100040, data described 0x00100000 (-64 B)" in text
    assert "watchdogs W; high: awvalid wready bready arready rvalid rready" in text
    X.clear(t)
    assert X.flags(t) == 0


def test_snap_times_out_without_acknowledgement():
    class Dead(_Regs):
        def reg_read(self, off):
            return 0 if off == X.REG + 4 else super().reg_read(off)
    with pytest.raises(TimeoutError):
        X.snap(Dead({0: 0x584D0000}), timeout=0.01)


def test_names_match_the_rtl():
    """The word names here, in the RTL's header and in the bench are one list."""
    rtl = (ROOT / "rtl/boards/ypcb-00338/otpu_xmon.sv").read_text()
    tb = (ROOT / "sim/verilator/tb_xmon.sv").read_text()
    assert re.findall(r'"(\w+)"', tb[tb.index("NAMES [32]"):tb.index("};", tb.index("NAMES [32]"))]) == X.NAMES
    hdr = " ".join(re.findall(r"^//\s+\d+(?:\.\.\d+|, \d+)?\s+(.*)$", rtl, re.M))
    for n in X.NAMES[2:]:
        assert n in hdr, n
    flat = re.sub(r"\n//\s*", " ", rtl)
    for i, n in enumerate(X.FLAGS):
        assert f"bit{i} {n}" in flat, n
