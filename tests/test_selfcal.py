"""The calibration CPU's firmware (tools/litedram/selfcal_fw) against the host calibration it ports
(opentpu/host/ddrcal.py): both run on identical simulated PHYs (ddrcal.FakeBoard) and must write
the same CSRs with the same values in the same order (every decision ddrcal takes shows up in a
write), and the firmware's mailbox, decoded by opentpu/host/selfcal.py, must equal
ddrcal.calibrate_channel's result. The firmware is built for this machine (SELFCAL_HOST, every
CSR access a ctypes callback); tools/litedram/selfcal_sim.py runs the RV32I image on the CPU."""
import random
import shutil
from pathlib import Path

import pytest

from opentpu.host import ddrcal as C
from opentpu.host import selfcal as S

import selfcal_harness as H

DATA = Path(__file__).parent / "data"
# the production core (gen_core.py --phy wl, ld-top 14875bf: cal_ready per channel), the WL and
# the A7 test images (ld_test.py)
CORE, WL, A7 = DATA / "litedram_core_wl", DATA / "litedram_wl", DATA / "litedram"

pytestmark = pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")


@pytest.fixture(scope="module")
def fws(tmp_path_factory):
    cache = {}

    def get(build):
        if build not in cache:
            cache[build] = H.Firmware(build, tmp_path_factory.mktemp("fw"))
        return cache[build]
    return get


def phy(build):
    ns = {}
    exec((Path(build) / "sdram_init.py").read_text(), ns)
    return ns["phy"]


def compare(fw, build, make, stride, mask=3):
    """ddrcal and the firmware on two boards from make(); the traces, then each channel's result
    or failure. Returns (ddrcal's results, the decoded mailbox, the firmware's board)."""
    ref_log, ref = H.reference(build, make(), stride, mask)
    board = make()
    t = H.Trace(board)
    mb, own, _ = fw.run(t, mask, stride)
    assert len(t.log) == len(ref_log) and t.log == ref_log, H.first_diff(ref_log, t.log)
    p = phy(build)
    dec = S.decode(mb, p, p.get("groups"))
    assert sorted(dec) == sorted(ref)
    for ch, r in ref.items():
        d = dec[ch]
        if isinstance(r, Exception):
            assert d["state"] == "failed" and d["error"], (ch, r, d)
            continue
        assert d["state"] == "ok" and not d["error"], d
        assert {k: d[k] for k in r} == r
        f = board.ch[ch] if hasattr(board, "ch") else board
        assert [[f.rbb[m][i] - rl["bitslip"] for i in range(8)]
                for m, rl in enumerate(d["read"])] == d["bit_offsets"]
    st = S.state(None, own[-1][1])
    assert own[-1][0] == "selfcal_state" and st["done"]
    assert {ch: s for ch, (s, _) in st["channels"].items() if s != "idle"} == \
        {ch: "failed" if isinstance(r, Exception) else "ok" for ch, r in ref.items()}
    return ref, dec, board


def test_core_both_channels(fws):
    ref, dec, board = compare(fws(CORE), CORE, lambda: C.FakeBoard(CORE), stride=4)
    assert all(not isinstance(r, Exception) for r in ref.values())
    for ch in (0, 1):
        assert board.ch[ch].v["cal_ready"] == 1
        assert dec[ch]["group_runs_steps"] == {0: dec[ch]["window_steps"]}


def test_wl_test_image(fws):
    compare(fws(WL), WL, lambda: C.FakeBoard(WL), stride=8)


def ldtest3e(seed, build=CORE):
    """ldtest3e's channel 1: lane 3 bit 3 and lane 8 bits 1, 2, 5, 7 read at their lane's
    bitslip + 2 (docs/litedram.md section 8); ldtest3d's: dq27 +2, dq64 and dq67 -2."""
    def make():
        b = C.FakeBoard(build)
        f = b.ch[1]
        f.RB[3], f.RB[8] = 3, 4
        f.BOFF = {(3, 3): 2, (8, 1): 2, (8, 2): 2, (8, 5): 2, (8, 7): 2} if seed == 0 else \
            {(3, 3): 2, (8, 0): -2, (8, 3): -2}
        for m in range(f.nm):
            f.WLO[m], f.WHI[m] = 20 + m, 90 - m
        return b
    return make


@pytest.mark.parametrize("case", [0, 1])
def test_wl_per_bit_framing(fws, case):
    ref, dec, _ = compare(fws(CORE), CORE, ldtest3e(case), stride=8)
    assert not isinstance(ref[1], Exception)
    offs = {(m, i): o for m, row in enumerate(dec[1]["bit_offsets"]) for i, o in enumerate(row) if o}
    assert offs == ldtest3e(case)().ch[1].BOFF


@pytest.mark.parametrize("seed", range(3, 9))
def test_wl_random_phys(fws, seed):
    # random lanes (read / write windows, latencies) and random misframed bits on both channels
    def make():
        b = C.FakeBoard(CORE)
        rnd = random.Random(seed)
        for k, f in enumerate(b.ch):
            g = C.FakeCsr(CORE, seed=10 * seed + k)
            f.RB, f.WB, f.LO, f.HI, f.WLO, f.WHI = g.RB, g.WB, g.LO, g.HI, g.WLO, g.WHI
            f.BOFF = {(rnd.randrange(9), rnd.randrange(8)): rnd.choice((2, -2))
                      for _ in range(rnd.randrange(4))}
        return b
    compare(fws(CORE), CORE, make, stride=rnd_stride(seed))


def rnd_stride(seed):
    return (8, 16, 5, 7, 16, 8)[seed % 6]


def test_no_common_phase_fails_like_ddrcal(fws):
    def make():
        b = C.FakeBoard(CORE)
        f = b.ch[0]
        f.WLO[0], f.WHI[0] = 0, 10
        f.WLO[1], f.WHI[1] = 60, 70
        return b
    ref, dec, board = compare(fws(CORE), CORE, make, stride=8)
    assert isinstance(ref[0], C.CalError) and "no CK phase" in dec[0]["error"]
    assert board.ch[0].v["cal_ready"] == 0 and board.ch[1].v["cal_ready"] == 1


def test_write_clock_check_fails_like_ddrcal(fws):
    def make():
        b = C.FakeBoard(CORE)
        b.ch[1].drp[0x10] = 0x6041            # CLKOUT4 (group 0's DQS) at 3/8, not 4/8 after DQ
        return b
    ref, dec, board = compare(fws(CORE), CORE, make, stride=16)
    assert isinstance(ref[1], C.CalError) and "phases" in dec[1]["error"]
    assert board.ch[1].v["cal_ready"] == 0


def test_write_clock_groups(fws):
    # two write clock groups with different write ranges: each group's run is reported
    groups = [0, 0, 0, 0, 1, 1, 1, 1, 1]

    def make():
        b = C.FakeBoard(CORE)
        for f in b.ch:
            f.groups = groups
            for m in range(f.nm):
                f.WLO[m], f.WHI[m] = (20 + m, 60 - m) if groups[m] == 0 else (40 + m, 80 - m)
        return b
    ref, dec, _ = compare(fws(CORE), CORE, make, stride=2, mask=2)
    assert dec[1]["window_steps"] == 10


def test_channel_mask_and_one_channel(fws):
    ref, dec, board = compare(fws(CORE), CORE, lambda: C.FakeBoard(CORE), stride=16, mask=2)
    assert list(dec) == [1] and board.ch[0].v.get("cal_ready", 0) == 0


def test_a7_image(fws):
    # the A7DDRPHY test image (tests/data/litedram): the DQS phase scan without write clocks
    compare(fws(A7), A7, lambda: C.FakeBoard(A7), stride=4)


def test_without_per_bit_csr(fws, tmp_path):
    # a WL7DDRPHY core without ddrphy_dly_sel_bits: per-lane framing only, as ddrcal
    for f in ("csr.csv", "sdram_init.py"):
        text = (CORE / f).read_text()
        if f == "csr.csv":
            text = "\n".join(line for line in text.splitlines() if "dly_sel_bits" not in line)
        (tmp_path / f).write_text(text)
    ref, dec, _ = compare(H.Firmware(tmp_path, tmp_path / "fw"), tmp_path, ldtest3e(1, tmp_path),
                          stride=8)
    assert isinstance(ref[1], C.CalError)


@pytest.mark.slow
def test_core_stride_1(fws):
    # production's stride: the whole 113-step scan on both channels (80 s)
    compare(fws(CORE), CORE, ldtest3e(0), stride=1)


# ------------------------------------------------------------------------------ otpu-memcal
from opentpu.host import memcal, regs as R          # noqa: E402

SELF = DATA / "litedram_core_selfcal"               # gen_core.py --phy wl --selfcal's map


class SelfcalCsr:
    """The core's CSRs by name (ddrcal's w / r) with its calibration CPU: FakeBoard's PHYs,
    controllers and BISTs, the selfcal CSRs, and the CPU: the firmware (built for this machine)
    runs on the FakeBoard when the CPU is released, at power-up or when the host writes
    selfcal_hold 0; `lazy`: not before the host first looks at selfcal_state (a CPU still running
    when the host comes). fault(board, on): a fault present while the CPU runs. The scan stride
    is selfcal_config's (8 here: as if set before the release)."""
    def __init__(self, fw, lazy=False, fault=None, stride=8, build=SELF):
        self.fw, self.fault = fw, fault
        self.b = C.FakeBoard(build)
        self.regs = self.b.regs
        self.own = {"selfcal_hold": 0, "selfcal_config": 3 | stride << 8, "selfcal_state": 0,
                    "selfcal_mbox_adr": 0}
        self.mbox, self.cpu_runs, self.host_writes, self.due = [0] * 256, 0, [], True
        if not lazy:
            self.cpu()

    def cpu(self):
        self.due = False
        if self.fault:
            self.fault(self.b, True)
        cfg = self.own["selfcal_config"]
        self.mbox, own, _ = self.fw.run(self.b, cfg & 3, cfg >> 8 & 0xFF)
        self.own["selfcal_state"] = own[-1][1]
        self.cpu_runs += 1
        if self.fault:
            self.fault(self.b, False)

    def r(self, name):
        if name == "selfcal_status":
            return S.MAGIC << 16 | self.own["selfcal_hold"]
        if name == "selfcal_state" and self.due and not self.own["selfcal_hold"]:
            self.cpu()
        if name == "selfcal_mbox_dat":
            return self.mbox[self.own["selfcal_mbox_adr"]]
        return self.own[name] if name in self.own else self.b.r(name)

    def w(self, name, v):
        if name == "selfcal_hold":
            if self.own[name] and not v:
                self.due = True             # released: the CPU starts over
            self.own[name] = v
        elif name in self.own:
            self.own[name] = v
        else:
            self.host_writes.append(name)
            self.b.w(name, v)


class SelfcalCard(SelfcalCsr):
    """BAR0 of a LiteDRAM bitstream whose core calibrates itself: REGMAP 3, CAPS bit27, STATUS
    CALIB0/1 from the channels' cal_ready, and the core's CSRs (SelfcalCsr) at R_MEMCAL."""
    devname = None

    def __init__(self, fw, **kw):
        super().__init__(fw, **kw)
        self.words = {off + 4 * i: (name, i, n) for name, (off, n) in self.regs.items()
                      for i in range(n)}
        self.pending, self.csr_ops = {}, 0

    def reg_read(self, off):
        if off == R.R_ID:
            return R.ID_OTPU
        if off == R.R_REGMAP:
            return 3
        if off == R.R_CAPS:
            return R.CAP_HOSTCAL
        if off == R.R_STATUS:
            return (R.ST_CALIB0 if self.b.r("cal_ready") else 0) | \
                   (R.ST_CALIB1 if self.b.r("cal1_ready") else 0)
        name, i, n = self.words[off - R.R_MEMCAL]
        self.csr_ops += 1
        return (self.r(name) >> (32 * (n - 1 - i))) & 0xFFFFFFFF

    def reg_write(self, off, v):
        name, i, n = self.words[off - R.R_MEMCAL]
        self.csr_ops += 1
        acc = (self.pending.pop(name, 0) << 32) | v
        if i == n - 1:
            self.w(name, acc)
        else:
            self.pending[name] = acc


@pytest.fixture
def run_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("OTPU_RUN_DIR", str(tmp_path))
    monkeypatch.setattr(memcal, "DATA", SELF)


def test_memcal_calibrated_by_the_core(fws, run_dir, capsys):
    card = SelfcalCard(fws(SELF))
    assert card.reg_read(R.R_STATUS) == R.ST_CALIB0 | R.ST_CALIB1
    assert memcal.ensure(card, log=lambda *_: None) is None and card.csr_ops == 0
    assert memcal.main([], open_transport=lambda: card) == 0
    out = capsys.readouterr().out
    assert "calibrated by the core's CPU" in out and "the core's CPU: done" in out
    assert "channel 0: CK step" in out and "channel 1: CK step" in out


def test_memcal_waits_for_the_core(fws, run_dir):
    card = SelfcalCard(fws(SELF), lazy=True)
    assert card.reg_read(R.R_STATUS) == 0                 # the CPU still at work
    res = memcal.ensure(card, data=SELF, log=lambda *_: None)
    assert res["by"] == "selfcal" and sorted(res["channels"]) == [0, 1]
    assert card.cpu_runs == 1 and not card.host_writes     # the host calibrated nothing
    assert card.reg_read(R.R_STATUS) == R.ST_CALIB0 | R.ST_CALIB1
    for ch, f in enumerate(card.b.ch):
        assert res["channels"][ch]["write_latency"] == f.WB and all(f.good(m) for m in range(9))
    assert memcal.last(card)["by"] == "selfcal"


def test_memcal_calibrates_what_the_core_failed(fws, run_dir):
    def fault(b, on):                                       # ch1's write clock DRP misread
        b.ch[1].drp[0x10] = 0x6041 if on else 0x8041
    card = SelfcalCard(fws(SELF), lazy=True, fault=fault)
    res = memcal.ensure(card, data=SELF, stride=16, log=lambda *_: None)
    assert res["by"] == "selfcal+host" and sorted(res["channels"]) == [0, 1]
    assert "dqs_steps" in res["channels"][1] and "state" not in res["channels"][1]   # ddrcal's
    assert card.own["selfcal_hold"] == 1                     # held: the host calibrated
    assert card.reg_read(R.R_STATUS) == R.ST_CALIB0 | R.ST_CALIB1


def test_memcal_force_holds_the_core(fws, run_dir):
    card = SelfcalCard(fws(SELF))
    res = memcal.ensure(card, force=True, data=SELF, stride=16, log=lambda *_: None)
    assert res["by"] == "host" and sorted(res["channels"]) == [0, 1]
    assert card.own["selfcal_hold"] == 1 and card.cpu_runs == 1
    assert card.reg_read(R.R_STATUS) == R.ST_CALIB0 | R.ST_CALIB1


def test_memcal_selfcal_again(fws, run_dir):
    card = SelfcalCard(fws(SELF))
    memcal.ensure(card, force=True, data=SELF, stride=16, log=lambda *_: None)
    res = memcal.selfcal_again(card, data=SELF, log=lambda *_: None)
    assert card.cpu_runs == 2 and card.own["selfcal_hold"] == 0
    assert all(r["state"] == "ok" for r in res.values()) and sorted(res) == [0, 1]


def test_ld_host_selfcal(fws, capsys):
    # tools/litedram/ld_host.py selfcal --rerun --soak --compare on a simulated image
    import importlib.util
    import types
    spec = importlib.util.spec_from_file_location("ld_host", H.ROOT / "tools/litedram/ld_host.py")
    ld_host = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ld_host)
    c = SelfcalCsr(fws(SELF))
    a = types.SimpleNamespace(rerun=True, soak=True, seconds=0.2, compare=True, gib=1 / 64,
                              build=SELF)
    rc = ld_host.selfcal_run(a, c, [0, 1], 112, phy(SELF))
    out = capsys.readouterr().out
    assert rc == 0 and "selfcal: PASS" in out, out[-3000:]
    assert c.cpu_runs == 2 and c.own["selfcal_hold"] == 1
    assert out.count("within 2 steps") == 2 and "soak:" in out
