#!/usr/bin/env python3
"""Host side of the LiteDRAM test image (tools/litedram/ld_test.py): DDR3 channel 0 at 72 bits,
calibrated from the host over BAR0 (/dev/xdma0_user), then tested with the image's BIST.

    python3 ld_host.py BUILD_DIR all          # init, DQS scan, calibration, BIST (the report)
    python3 ld_host.py BUILD_DIR info         # identifier, DQS phase, scratch register check
    python3 ld_host.py BUILD_DIR cal          # init + write latency + read leveling at the
                                              # current DQS phase
    python3 ld_host.py BUILD_DIR bist [--gib 2]
    python3 ld_host.py BUILD_DIR selftest     # the calibration logic against a simulated PHY

BUILD_DIR holds csr.csv and sdram_init.py (written by ld_test.py). Only numpy-free stdlib.

Calibration follows LiteX's liblitedram/sdram.c for a PHY without write leveling (A7DDRPHY):
the JEDEC init sequence through DFII, write latency calibration (the write bitslip, in tCK steps,
with the best read window after a read scan), then read leveling (read bitslip, IDELAY tap at the
centre of the widest window). Every scan runs on all nine byte lanes at once (dly_sel selects them
all; the test pattern checks each lane's bytes), a nine-fold saving over the BIOS's lane-by-lane
loop; the chosen values are then set lane by lane. The test pattern: LFSR data on every phase,
written to row 0 / bank 0 / column 0 on the write phase, read back on the read phase.

Write margin: the banks have no ODELAY, so there is no write leveling; the write DQS clock's phase
(one MMCM output with fine phase shift, 1/56 VCO period per step) moves every lane's DQS against
CK and DQ together. `all` scans it over one DQS period and, per lane, records where a write
latency exists with a read window of at least MIN_WINDOW taps: those phase ranges are the write
margins; the chosen phase is the centre of the range common to all lanes.
"""
import argparse
import csv
import mmap
import os
import random
import re
import sys
import time
from pathlib import Path

MIN_WINDOW = 3            # read taps (78 ps each) for a lane to count as working
TAP_PS = 1e12 / (32 * 2 * 200e6)   # IDELAYE2 tap at a 200 MHz reference: 78.125 ps
SEEDS = (42, 84, 36)


# ------------------------------------------------------------------------------ CSR access
class Bar0:
    """CSRs over /dev/xdma0_user: one 32-bit load / store per word (see board.py's transport)."""
    def __init__(self, build: Path, dev: str = "/dev/xdma0_user"):
        self.regs = {}
        for row in csv.reader(open(build / "csr.csv")):
            if row and row[0] == "csr_register":
                self.regs[row[1]] = (int(row[2], 16), int(row[3]))
        fd = os.open(dev, os.O_RDWR | os.O_SYNC)
        self.mm = mmap.mmap(fd, 1 << 16, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        os.close(fd)
        self.words = memoryview(self.mm).cast("I")

    def w(self, name, v):
        a, n = self.regs[name]
        for i in range(n):            # most significant word first
            self.words[(a >> 2) + i] = (v >> (32 * (n - 1 - i))) & 0xFFFFFFFF

    def r(self, name):
        a, n = self.regs[name]
        v = 0
        for i in range(n):
            v = (v << 32) | self.words[(a >> 2) + i]
        return v


# ------------------------------------------------------------------------------ DRAM
class Dram:
    CTL_SEL, CTL_CKE, CTL_ODT, CTL_RESET_N = 1, 2, 4, 8
    CS, WE, CAS, RAS, WRDATA, RDDATA = 1, 2, 4, 8, 16, 32

    def __init__(self, csr, build: Path):
        self.c = csr
        ns = {}
        text = (build / "sdram_init.py").read_text()
        exec(text, ns)
        self.phy = ns["phy"]
        # (comment, address, bank, value, delay, is a DFII control write): the generated header
        # evaluates the command names; its text tells control writes from commands
        self.init_sequence = [
            (c, int(a), int(ba), eval(expr, ns), int(dl), "dfii_control" in expr)
            for c, a, ba, expr, dl in re.findall(
                r'\("([^"]*)", (\d+), (\d+), ([a-z_|]+), (\d+)\)', text)]
        assert len(self.init_sequence) == len(ns["init_sequence"])
        p = self.phy
        self.nm, self.nph, self.db = p["modules"], p["nphases"], p["databits"]
        self.all = (1 << self.nm) - 1
        self.wb = [0] * self.nm            # write bitslip per lane (as set)
        self.rb = [0] * self.nm            # read bitslip
        self.rd = [0] * self.nm            # read delay (taps)

    # DFII
    def ctl(self, v):
        self.c.w("sdram_dfii_control", v)

    def cmd(self, ph, c, a=0, ba=0):
        self.c.w(f"sdram_dfii_pi{ph}_address", a)
        self.c.w(f"sdram_dfii_pi{ph}_baddress", ba)
        self.c.w(f"sdram_dfii_pi{ph}_command", c)
        self.c.w(f"sdram_dfii_pi{ph}_command_issue", 1)

    def pattern(self, seed):
        rng = random.Random(seed)
        return [rng.getrandbits(2 * self.db) for _ in range(self.nph)]

    def test(self, seeds=SEEDS):
        """Write / read back LFSR data on row 0; the wrong bits per lane (rising and falling
        edge bytes of every phase)."""
        err = [0] * self.nm
        for s in seeds:
            pat = self.pattern(s)
            self.cmd(0, self.RAS | self.CS)                                  # activate row 0
            for ph in range(self.nph):
                self.c.w(f"sdram_dfii_pi{ph}_wrdata", pat[ph])
            self.cmd(self.phy["wrphase"], self.CAS | self.WE | self.CS | self.WRDATA)
            self.cmd(self.phy["rdphase"], self.CAS | self.CS | self.RDDATA)
            self.cmd(0, self.RAS | self.WE | self.CS)                        # precharge
            for ph in range(self.nph):
                x = self.c.r(f"sdram_dfii_pi{ph}_rddata") ^ pat[ph]
                for m in range(self.nm):
                    err[m] += bin((x >> (8 * m)) & 0xFF).count("1") + \
                              bin((x >> (self.db + 8 * m)) & 0xFF).count("1")
        return err

    # PHY: dly_sel selects the lanes the next strobes act on
    def sel(self, mask):
        self.c.w("ddrphy_dly_sel", mask)

    def strobe(self, name, mask, n=1):
        self.sel(mask)
        for _ in range(n):
            self.c.w(name, 1)
        self.sel(0)

    def set_wbitslip(self, m, v):
        self.strobe("ddrphy_wdly_dq_bitslip_rst", 1 << m)
        self.strobe("ddrphy_wdly_dq_bitslip", 1 << m, v)
        self.wb[m] = v

    def set_rbitslip(self, m, v):
        self.strobe("ddrphy_rdly_dq_bitslip_rst", 1 << m)
        self.strobe("ddrphy_rdly_dq_bitslip", 1 << m, v)
        self.rb[m] = v

    def set_rdelay(self, m, v):
        self.strobe("ddrphy_rdly_dq_rst", 1 << m)
        self.strobe("ddrphy_rdly_dq_inc", 1 << m, v)
        self.rd[m] = v

    def init(self):
        """PHY reset, the JEDEC sequence, every delay and bitslip reset (software control)."""
        self.c.w("ddrphy_rdphase", self.phy["rdphase"])
        self.c.w("ddrphy_wrphase", self.phy["wrphase"])
        self.ctl(self.CTL_CKE | self.CTL_ODT | self.CTL_RESET_N)
        self.c.w("ddrphy_rst", 1)
        time.sleep(0.001)
        self.c.w("ddrphy_rst", 0)
        time.sleep(0.001)
        for _comment, a, ba, cmd, delay, control in self.init_sequence:
            self.c.w("sdram_dfii_pi0_address", a)
            self.c.w("sdram_dfii_pi0_baddress", ba)
            if control:
                self.ctl(cmd)
            else:
                self.c.w("sdram_dfii_pi0_command", cmd)
                self.c.w("sdram_dfii_pi0_command_issue", 1)
            time.sleep(max(delay / self.phy["sys_hz"] * 4, 20e-6))
        for name in ("ddrphy_wdly_dq_bitslip_rst", "ddrphy_rdly_dq_rst", "ddrphy_rdly_dq_bitslip_rst"):
            self.strobe(name, self.all)
        self.wb, self.rb, self.rd = [0] * self.nm, [0] * self.nm, [0] * self.nm

    def hardware(self):
        self.ctl(self.CTL_SEL)

    # scans (all lanes at once)
    def read_scan(self, seeds=SEEDS):
        """err[bitslip][tap][lane] at the current write bitslips."""
        d, nb = self.phy["delays"], self.phy["bitslips"]
        out = []
        self.strobe("ddrphy_rdly_dq_bitslip_rst", self.all)
        for b in range(nb):
            self.strobe("ddrphy_rdly_dq_rst", self.all)
            row = []
            for t in range(d):
                row.append(self.test(seeds))
                self.strobe("ddrphy_rdly_dq_inc", self.all)
            out.append(row)
            self.strobe("ddrphy_rdly_dq_bitslip", self.all)
        # the scan leaves bitslip 0 (8 increments wrap) and taps wrapped to 0
        return out

    @staticmethod
    def windows(taps_err):
        """The longest run of error-free taps: (start, length)."""
        best, cur, start = (0, 0), 0, 0
        for t, e in enumerate(taps_err):
            if e == 0:
                if cur == 0:
                    start = t
                cur += 1
                if cur > best[1]:
                    best = (start, cur)
            else:
                cur = 0
        return best

    def lane_best(self, scan, m):
        """(window length, bitslip, window start) of lane m's best read bitslip in a scan."""
        best = (0, 0, 0)
        for b, rows in enumerate(scan):
            start, n = self.windows([r[m] for r in rows])
            if n > best[0]:
                best = (n, b, start)
        return best

    def write_latency(self, seeds=SEEDS, verbose=True):
        """Per lane, the write bitslip (0, 2, 4, 6: tCK steps) with the widest read window."""
        res = {}
        for wbs in range(0, self.phy["bitslips"], 2):
            self.strobe("ddrphy_wdly_dq_bitslip_rst", self.all)
            self.strobe("ddrphy_wdly_dq_bitslip", self.all, wbs)
            scan = self.read_scan(seeds)
            res[wbs] = [self.lane_best(scan, m) for m in range(self.nm)]
        choice = []
        for m in range(self.nm):
            wbs = max(res, key=lambda k: res[k][m][0])
            choice.append(wbs if res[wbs][m][0] > 0 else -1)
            self.set_wbitslip(m, max(wbs, 0))
        if verbose:
            print("write latency: " + " ".join(
                f"m{m}:{'-' if c < 0 else c}" for m, c in enumerate(choice)))
            for wbs in res:
                print(f"  wb{wbs}: " + " ".join(f"{n:2d}" for n, _, _ in res[wbs]) + "  (widest read window per lane, taps)")
        return choice, res

    def read_leveling(self, seeds=SEEDS, verbose=True):
        scan = self.read_scan(seeds)
        out = []
        for m in range(self.nm):
            n, b, start = self.lane_best(scan, m)
            if verbose:
                line = "".join("1" if r[m] == 0 else "0" for r in scan[b])
                print(f"  m{m}: b{b} |{line}| window {n} taps ({n * TAP_PS:.0f} ps)"
                      + (f", tap {start + n // 2}" if n else ""))
            self.set_rbitslip(m, b)
            self.set_rdelay(m, start + n // 2 if n else 0)
            out.append((n, b, start))
        err = self.test(seeds)
        if verbose:
            print("  check at the chosen taps: " + ("pass" if not any(err) else f"errors per lane {err}"))
        return out, err

    def calibrate(self, seeds=SEEDS, verbose=True):
        self.init()
        wl, _ = self.write_latency(seeds, verbose)
        if verbose:
            print("read leveling:")
        rl, err = self.read_leveling(seeds, verbose)
        return wl, rl, err


# ------------------------------------------------------------------------------ DQS phase
class DqsPhase:
    def __init__(self, csr, vco_hz):
        self.c = csr
        self.step_ps = 1e12 / vco_hz / 56

    def steps(self):
        v = self.c.r("phase_dqs_steps")
        return v - (1 << 32) if v & (1 << 31) else v

    def move(self, target):
        while True:
            s = self.steps()
            if s == target:
                return
            while self.c.r("phase_dqs_busy"):
                pass
            self.c.w("phase_dqs_shift", 1 if target > s else 0)


def dqs_scan(dram, dqs, period_steps, stride, seeds=(42,), csr=None, mib=64):
    """Per DQS phase step: calibrate (write latency, read leveling) and, with `csr`, write and
    read back `mib` MiB with the BIST. Per lane: the read window (taps) if its DFI check and its
    BIST beats are all right, else 0 (the BIST's wrong beats are printed)."""
    table = {}
    start = dqs.steps()
    for k in range(0, period_steps + 1, stride):
        dqs.move(start + k)
        wl, rl, err = dram.calibrate(seeds, verbose=False)
        win = [n if w >= 0 and not e else 0 for (n, _, _), w, e in zip(rl, wl, err)]
        bad = [0] * dram.nm
        if csr is not None:
            dram.hardware()
            beats = mib * (1 << 20) // 64
            bist(csr, beats, 0, 0x0123456789ABCDEF + k, dram.phy["sys_hz"])
            bad = bist(csr, beats, 1, 0x0123456789ABCDEF + k, dram.phy["sys_hz"])["lanes"]
            win = [0 if b else n for n, b in zip(win, bad)]
        table[k] = win
        print(f"  dqs +{k:3d} ({k * dqs.step_ps:6.0f} ps): windows " + " ".join(f"{n:2d}" for n in win)
              + "  write latency " + "".join("-" if w < 0 else str(w) for w in wl)
              + ("  BIST wrong beats " + " ".join(str(x) for x in bad) if any(bad) else ""))
    dqs.move(start)
    return table


def margins(table, nm, min_window=MIN_WINDOW, period=None):
    """Per lane, the passing phase steps; the common passing set; the chosen step (the centre
    of the longest common run, taken around the circle when the scan covers a whole tCK of
    `period` steps: the phase wraps) and that run."""
    ks = sorted(table)
    per = [[k for k in ks if table[k][m] >= min_window] for m in range(nm)]
    common = [k for k in ks if all(table[k][m] >= min_window for m in range(nm))]
    circular = period is not None and ks[-1] - ks[0] >= period
    seq = [k for k in ks if not (circular and k - ks[0] >= period)]     # one period, no repeat
    order = seq + seq if circular else seq
    best, cur = [], []
    for k in order:
        if k in common:
            cur.append(k)
            if len(cur) > len(best) and len(cur) <= len(seq):
                best = list(cur)
        else:
            cur = []
    pick = best[len(best) // 2] if best else None
    return per, common, pick, best


# ------------------------------------------------------------------------------ BIST
def bist(csr, beats, mode, seed, sys_hz, timeout=60.0):
    csr.w("bist_base", 0)
    csr.w("bist_length", beats)
    csr.w("bist_mode", mode)
    csr.w("bist_seed", seed)
    csr.w("bist_start", 1)
    t0 = time.time()
    while not csr.r("bist_done"):
        if time.time() - t0 > timeout:
            raise TimeoutError(f"BIST mode {mode} not done after {timeout} s "
                               f"({csr.r('bist_beats')} of {beats} beats)")
        time.sleep(0.01)
    ticks = csr.r("bist_ticks")
    r = {"beats": csr.r("bist_beats"), "ticks": ticks,
         "gbs": beats * 64 / (ticks / sys_hz) / 1e9 if ticks else 0.0}
    if mode & 1:
        r["errors"] = csr.r("bist_errors")
        r["lanes"] = [csr.r(f"bist_lane{m}_errors") for m in range(9)]
        r["bits"] = csr.r("bist_lane_bits")
    return r


def run_bist(csr, sys_hz, gib, passes=2):
    beats = int(gib * (1 << 30)) // 64
    peak = sys_hz * 64 / 1e9          # one 64-byte beat (8 DQ beats x 64 data bits) per sys cycle
    ok = True
    for p in range(passes):
        for dmode, name in ((0, "random data"), (2, "address data")):
            seed = (0x0123456789ABCDEF ^ (p * 0x9E3779B97F4A7C15)) & ((1 << 64) - 1)
            w = bist(csr, beats, dmode, seed, sys_hz)
            r = bist(csr, beats, dmode | 1, seed, sys_hz)
            print(f"  pass {p} {name}: write {w['gbs']:.2f} GB/s ({w['gbs'] / peak:.1%}), "
                  f"read {r['gbs']:.2f} GB/s ({r['gbs'] / peak:.1%}), "
                  f"errors {r['errors']} (per lane {r['lanes']}, bits {r['bits']:#x})")
            ok &= r["errors"] == 0 and r["beats"] == beats
    return ok


def bist_read_scan(csr, dram, sys_hz, mib=256, seed=0x0123456789ABCDEF, mode=0):
    """Read windows under traffic: write a region once (BIST), then for every read tap (all
    lanes at the same tap, each lane at its calibrated bitslip) read it back with the BIST and
    count each lane's wrong beats. Returns err[tap][lane]; leaves every lane at the centre of
    its passing run under traffic (or its calibrated tap if none)."""
    beats = mib * (1 << 20) // 64
    w = bist(csr, beats, mode, seed, sys_hz)
    table = []
    dram.strobe("ddrphy_rdly_dq_rst", dram.all)
    for tap in range(dram.phy["delays"]):
        r = bist(csr, beats, mode | 1, seed, sys_hz)
        table.append(r["lanes"])
        dram.strobe("ddrphy_rdly_dq_inc", dram.all)
    for m in range(dram.nm):
        line = "".join("1" if row[m] == 0 else ("0" if row[m] > beats // 100 else "x") for row in table)
        start, n = Dram.windows([row[m] for row in table])
        print(f"  m{m}: b{dram.rb[m]} |{line}| {n} taps under traffic"
              + (f" (DFI window centre {dram.rd[m]}, traffic centre {start + n // 2})" if n else ""))
        dram.set_rdelay(m, start + n // 2 if n else dram.rd[m])
    return table


# ------------------------------------------------------------------------------ simulated PHY
class FakeCsr:
    """A PHY model for `selftest`: lane m reads back right only at its read bitslip RB[m], taps
    in [LO[m], HI[m]], and its write bitslip WB[m] (+-0 tCK), when the DQS phase step is within
    its write range."""
    def __init__(self, build, nm=9):
        self.regs = {}
        for row in csv.reader(open(build / "csr.csv")):
            if row and row[0] == "csr_register":
                self.regs[row[1]] = (int(row[2], 16), int(row[3]))
        rnd = random.Random(1)
        self.nm = nm
        self.RB = [rnd.randrange(8) for _ in range(nm)]
        self.WB = [rnd.choice((0, 2, 4)) for _ in range(nm)]
        self.LO = [rnd.randrange(0, 18) for _ in range(nm)]
        self.HI = [lo + rnd.randrange(4, 12) for lo in self.LO]
        self.WLO = [rnd.randrange(10, 30) for _ in range(nm)]
        self.WHI = [lo + rnd.randrange(40, 70) for lo in self.WLO]
        self.v = {}
        self.sel, self.rb, self.wb, self.rd = 0, [0] * nm, [0] * nm, [0] * nm
        self.wr = {}
        self.steps = 0

    def w(self, name, v):
        self.v[name] = v
        lanes = [m for m in range(self.nm) if self.sel >> m & 1]
        if name == "ddrphy_dly_sel":
            self.sel = v
        elif name == "ddrphy_rdly_dq_rst":
            for m in lanes: self.rd[m] = 0
        elif name == "ddrphy_rdly_dq_inc":
            for m in lanes: self.rd[m] = (self.rd[m] + 1) % 32
        elif name == "ddrphy_rdly_dq_bitslip_rst":
            for m in lanes: self.rb[m] = 0
        elif name == "ddrphy_rdly_dq_bitslip":
            for m in lanes: self.rb[m] = (self.rb[m] + 1) % 8
        elif name == "ddrphy_wdly_dq_bitslip_rst":
            for m in lanes: self.wb[m] = 0
        elif name == "ddrphy_wdly_dq_bitslip":
            for m in lanes: self.wb[m] = (self.wb[m] + 1) % 8
        elif name == "phase_dqs_shift":
            self.steps += 1 if v else -1
        elif name.endswith("_command_issue") and self.v.get(name[:-6]) and name.startswith("sdram_dfii_pi"):
            ph = int(name[len("sdram_dfii_pi")])
            c = self.v[name[:-6]]
            if c & Dram.WRDATA:
                self.wr = {p: self.v.get(f"sdram_dfii_pi{p}_wrdata", 0) for p in range(4)}
            if c & Dram.RDDATA:
                for p in range(4):
                    x = self.wr.get(p, 0)
                    for m in range(self.nm):
                        good = (self.rb[m] == self.RB[m] and self.LO[m] <= self.rd[m] <= self.HI[m]
                                and self.wb[m] == self.WB[m]
                                and self.WLO[m] <= self.steps % 112 <= self.WHI[m])
                        if not good:
                            for sh in (8 * m, 72 + 8 * m):
                                x ^= (random.getrandbits(8) | 1) << sh
                    self.v[f"sdram_dfii_pi{p}_rddata"] = x

    def r(self, name):
        if name == "phase_dqs_steps":
            return self.steps & 0xFFFFFFFF
        if name == "phase_dqs_busy":
            return 0
        return self.v.get(name, 0)


def selftest(build):
    fake = FakeCsr(build)
    d = Dram(fake, build)
    dqs = DqsPhase(fake, d.phy["vco_hz"])
    table = dqs_scan(d, dqs, 112, 8)
    per, common, pick, run = margins(table, d.nm)
    print("pick", pick, "run", run)
    for m in range(d.nm):
        assert all(fake.WLO[m] <= k <= fake.WHI[m] for k in per[m]), (m, per[m])
    dqs.move(pick)
    wl, rl, err = d.calibrate(seeds=(42,), verbose=True)
    assert wl == fake.WB, (wl, fake.WB)
    for m in range(d.nm):
        n, b, start = rl[m]
        assert b == fake.RB[m] and start == fake.LO[m] and n == fake.HI[m] - fake.LO[m] + 1, (m, rl[m])
    assert not any(err)
    print("selftest: PASS")


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("build", type=Path)
    ap.add_argument("what", choices=["info", "cal", "bist", "all", "selftest", "dqs", "rscan", "soak"])
    ap.add_argument("--seconds", type=float, default=300, help="soak: how long to repeat the BIST")
    ap.add_argument("--span", type=int, help="all: scan only this many steps (default one tCK)")
    ap.add_argument("--gib", type=float, default=2.0, help="BIST size (the channel: 2 GiB)")
    ap.add_argument("--stride", type=int, default=4, help="DQS scan stride, fine steps")
    ap.add_argument("--dev", default="/dev/xdma0_user")
    ap.add_argument("--steps", type=int, help="dqs: move the DQS phase to this step")
    a = ap.parse_args()
    if a.what == "selftest":
        return selftest(a.build)
    csr = Bar0(a.build, a.dev)
    d = Dram(csr, a.build)
    dqs = DqsPhase(csr, d.phy["vco_hz"])
    period = round(56 * d.phy["vco_hz"] / (4 * d.phy["sys_hz"]))     # fine steps per tCK
    if a.what in ("info", "all"):
        csr.w("ctrl_scratch", 0xA5C3_5A3C)
        s = csr.r("ctrl_scratch") ^ 0xA5C3_5A3C ^ 0x12345678      # 0x12345678 when it holds
        print(f"scratch {s:#x} ({'ok' if s == 0x12345678 else 'BAD'}), bus errors "
              f"{csr.r('ctrl_bus_errors')}, DQS phase {dqs.steps()} steps "
              f"({dqs.step_ps:.1f} ps each, {period} per tCK), sys {d.phy['sys_hz'] / 1e6:.2f} MHz, "
              f"CL {d.phy['cl']} CWL {d.phy['cwl']}")
        if s != 0x12345678:
            return 1
    if a.what == "dqs":
        dqs.move(a.steps)
        print("DQS phase", dqs.steps())
    if a.what == "all":
        print(f"DQS phase scan (every {a.stride} steps = {a.stride * dqs.step_ps:.0f} ps over one "
              f"tCK; per lane: calibrated read window in taps, 0 if its calibration check or a "
              f"64 MiB BIST write / read failed):")
        table = dqs_scan(d, dqs, a.span or period, a.stride, csr=csr)
        per, common, pick, run = margins(table, d.nm, period=a.span or period)
        for m in range(d.nm):
            ks = per[m]
            print(f"  lane {m}: writes at {len(ks)} of {len(table)} phases"
                  + (f", {min(ks) * dqs.step_ps:.0f}..{max(ks) * dqs.step_ps:.0f} ps" if ks else ""))
        if pick is None:
            print("no DQS phase works for every lane: FAIL")
            return 1
        print(f"common: {len(run)} scanned phases, {(len(run) - 1) * a.stride * dqs.step_ps:.0f} ps "
              f"from first to last (steps +{run[0]}..+{run[-1]}, around the tCK); DQS at +{pick} "
              f"steps ({pick * dqs.step_ps:.0f} ps)")
        dqs.move(dqs.steps() + pick)
    if a.what in ("cal", "all", "rscan"):
        wl, rl, err = d.calibrate()
        d.hardware()
        if any(err) or min(w for w in wl) < 0:
            print("calibration: FAIL")
            return 1
        print("calibration: PASS")
    if a.what == "soak":
        # the current DQS phase: calibrate, then repeat the 2 GiB BIST (both data modes) until
        # `seconds` have passed, new seeds every pass
        wl, rl, err = d.calibrate(verbose=False)
        d.hardware()
        t0, n, bad = time.time(), 0, 0
        while time.time() - t0 < a.seconds:
            ok = run_bist(csr, d.phy["sys_hz"], a.gib, passes=1) if n % 10 == 0 else \
                all(bist(csr, int(a.gib * (1 << 30)) // 64, mode | 1, seed, d.phy["sys_hz"])["errors"] == 0
                    for mode, seed in [(0, n * 7919 + 1), (2, n * 104729 + 3)]
                    if bist(csr, int(a.gib * (1 << 30)) // 64, mode, seed, d.phy["sys_hz"]) is not None)
            n += 1
            bad += not ok
        print(f"soak: {n} passes of {a.gib:g} GiB (x2 data modes) in {time.time() - t0:.0f} s at DQS "
              f"{dqs.steps()} steps, {bad} with errors: " + ("PASS" if not bad else "FAIL"))
        return 1 if bad else 0
    if a.what == "rscan":
        print("read taps under traffic (1: no wrong beat, x: < 1% wrong, 0: more):")
        bist_read_scan(csr, d, d.phy["sys_hz"])
        a.what = "bist"
    if a.what in ("bist", "all"):
        print(f"BIST over {a.gib:g} GiB:")
        ok = run_bist(csr, d.phy["sys_hz"], a.gib)
        print("BIST:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
