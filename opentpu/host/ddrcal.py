"""DDR3 calibration for LiteDRAM's A7DDRPHY from the host (docs/litedram.md, section 7): the
LiteDRAM test image (tools/litedram/ld_host.py) and, once LiteDRAM replaces the MIG, the board's
bring-up use it. Stdlib only, no package imports, so a copy next to ld_host.py works on a card
host whose installed opentpu package predates it.

The CSRs are reached through `csr`, any object with w(name, value) / r(name) on LiteX CSR names
(csr.csv); Chan(csr, ch) gives channel ch's view (channel 0's names bare, channel 1's with a 1).
The PHY settings and the JEDEC init sequence come from the generated sdram_init.py (load_config).

Calibration follows LiteX's liblitedram/sdram.c for a PHY without write leveling (A7DDRPHY):
the JEDEC init sequence through DFII, write latency calibration (the write bitslip, in tCK steps,
with the best read window after a read scan), then read leveling (read bitslip, IDELAY tap at the
centre of the widest window). Every scan runs on all nine byte lanes at once (dly_sel selects them
all; the test pattern checks each lane's bytes), a nine-fold saving over the BIOS's lane-by-lane
loop; the chosen values are then set lane by lane. The test pattern: LFSR data on every phase,
written to row 0 / bank 0 / column 0 on the write phase, read back on the read phase.

Write margin: the banks have no ODELAY, so there is no write leveling; the write DQS clock's phase
(an MMCM output with fine phase shift, 1/56 VCO period per step) moves every lane's DQS against
CK and against its DQ. dqs_scan scans it over one tCK and, per lane, records where a write latency
exists with a read window of at least MIN_WINDOW taps and (with a BIST) where a traffic test
passes: the DFII check alone passed at phases where the BIST failed a lane (section 7). margins
picks the centre of the range common to all lanes; calibrate_channel does the whole flow.
"""
import csv
import random
import re
import time
from pathlib import Path

MIN_WINDOW = 3            # read taps (78 ps each) for a lane to count as working
TAP_PS = 1e12 / (32 * 2 * 200e6)   # IDELAYE2 tap at a 200 MHz reference: 78.125 ps
SEEDS = (42, 84, 36)


PREFIXES = ("ddrphy", "sdram", "bist", "phase")


class Chan:
    """Channel ch's view of the CSRs: channel 0 keeps the one-channel image's names (ddrphy_*,
    sdram_*, bist_*, phase_*), channel 1's are ddrphy1_*, sdram1_*, bist1_*, phase1_*."""
    def __init__(self, csr, ch):
        self.csr, self.ch, self.sfx = csr, ch, "" if ch == 0 else str(ch)

    def name(self, n):
        p, _, rest = n.partition("_")
        return f"{p}{self.sfx}_{rest}" if self.sfx and p in PREFIXES else n

    def w(self, n, v):
        self.csr.w(self.name(n), v)

    def r(self, n):
        return self.csr.r(self.name(n))


# ------------------------------------------------------------------------------ DRAM
class Dram:
    CTL_SEL, CTL_CKE, CTL_ODT, CTL_RESET_N = 1, 2, 4, 8
    CS, WE, CAS, RAS, WRDATA, RDDATA = 1, 2, 4, 8, 16, 32

    def __init__(self, csr, config):
        """config: the build's sdram_init.py, or the directory holding it (load_config)."""
        self.c = csr
        ns = {}
        text = Path(load_config(config)).read_text()
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


def dqs_scan(dram, dqs, period_steps, stride, seeds=(42,), csr=None, mib=64, show=True):
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
        if show:
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
    bist_start(csr, beats, mode, seed)
    return bist_wait(csr, beats, mode, sys_hz, timeout)


def bist_start(csr, beats, mode, seed):
    csr.w("bist_base", 0)
    csr.w("bist_length", beats)
    csr.w("bist_mode", mode)
    csr.w("bist_seed", seed)
    csr.w("bist_start", 1)


def bist_wait(csr, beats, mode, sys_hz, timeout=60.0):
    t0 = time.time()
    while not csr.r("bist_done"):
        if time.time() - t0 > timeout:
            raise TimeoutError(f"BIST mode {mode} not done after {timeout} s "
                               f"({csr.r('bist_beats')} of {beats} beats)")
        time.sleep(0.005)
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
    bist(csr, beats, mode, seed, sys_hz)
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


def pass_map(table, nm, period, min_window=MIN_WINDOW):
    """Per lane, the scan as one character per step from -period/2 to +period/2 around the
    scan's start (the phase the channel runs at): '#' = writes and reads right, '.' = not, ' ' =
    not scanned."""
    ks = range(-(period // 2), period // 2 + 1)
    return ["".join(" " if k % period not in table else "#" if table[k % period][m] >= min_window
                    else "." for k in ks) for m in range(nm)]


def offsets(run, period):
    """The common window `run` (scan steps from the start, possibly around the tCK) as signed
    offsets from the start: (lo, hi), or None when the start is not in it."""
    signed = sorted(k if k <= period // 2 else k - period for k in run)
    if 0 not in signed and period not in run:
        return None
    return signed[0], signed[-1]


# ------------------------------------------------------------------------------ simulated PHY
class FakeCsr:
    """A PHY model for `selftest`: lane m reads back right only at its read bitslip RB[m], taps
    in [LO[m], HI[m]], and its write bitslip WB[m] (+-0 tCK), when the DQS phase step is within
    its write range."""
    def __init__(self, build, nm=9, seed=1):
        self.regs = csr_map(Path(build) / "csr.csv")
        rnd = random.Random(seed)
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

    def good(self, m):
        return (self.rb[m] == self.RB[m] and self.LO[m] <= self.rd[m] <= self.HI[m]
                and self.wb[m] == self.WB[m] and self.WLO[m] <= self.steps % 112 <= self.WHI[m])

    def r(self, name):
        if name == "phase_dqs_steps":
            return self.steps & 0xFFFFFFFF
        if name in ("phase_dqs_busy", "ctrl_bus_errors"):
            return 0
        n = self.v.get("bist_length", 0)
        bad = [0 if self.good(m) else n for m in range(self.nm)] if self.v.get("bist_mode", 0) & 1 \
            else [0] * self.nm
        if name == "bist_done":
            return 1
        if name in ("bist_beats", "bist_ticks"):
            return n
        if name == "bist_errors":
            return max(bad)
        if name.startswith("bist_lane") and name.endswith("_errors"):
            return bad[int(name[len("bist_lane"):-len("_errors")])]
        if name == "xadc_temperature":
            return round((45 + 273.15) * 4096 / 503.975)
        if name == "i2c_in":
            return 0x1F                 # every line released (high): no device answers
        return self.v.get(name, 0)


class FakeBoard:
    """Two FakeCsr channels behind one CSR space: channel 1's names (ddrphy1_*, ...) go to the
    second, with its own lanes."""
    def __init__(self, build):
        self.ch = [FakeCsr(build), FakeCsr(build, seed=2)]
        self.regs = self.ch[0].regs

    def route(self, name):
        p, _, rest = name.partition("_")
        if p.endswith("1") and p[:-1] in PREFIXES:
            return self.ch[1], f"{p[:-1]}_{rest}"
        return self.ch[0], name

    def w(self, name, v):
        f, n = self.route(name)
        f.w(n, v)

    def r(self, name):
        f, n = self.route(name)
        return f.r(n)


# ------------------------------------------------------------------------------ configuration
def load_config(build):
    """The PHY settings and init sequence of a LiteDRAM build: its sdram_init.py (ld_test.py
    writes it into the build directory)."""
    p = Path(build)
    return p / "sdram_init.py" if p.is_dir() else p


def csr_map(path):
    """LiteX csr.csv -> {name: (byte offset, 32-bit words)}."""
    regs = {}
    for row in csv.reader(open(path)):
        if row and row[0] == "csr_register":
            regs[row[1]] = (int(row[2], 16), int(row[3]))
    return regs


class WordCsr:
    """w / r on CSR names over a 32-bit word transport (read32(offset), write32(offset, v)), a
    window at `base`; multi-word CSRs most significant word first (csr_data_width 32)."""
    def __init__(self, regs, read32, write32, base=0):
        self.regs, self.rd, self.wr, self.base = regs, read32, write32, base

    def w(self, name, v):
        a, n = self.regs[name]
        for i in range(n):
            self.wr(self.base + a + 4 * i, (v >> (32 * (n - 1 - i))) & 0xFFFFFFFF)

    def r(self, name):
        a, n = self.regs[name]
        v = 0
        for i in range(n):
            v = (v << 32) | self.rd(self.base + a + 4 * i)
        return v


# ------------------------------------------------------------------------------ the whole flow
class CalError(RuntimeError):
    pass


def calibrate_channel(csr, config, stride=1, mib=64, log=print):
    """One channel from reset to the controller: the DQS phase scan (a BIST traffic test per
    phase when the channel has one, else the DFII check only), the phase at the centre of the
    window common to all lanes, write latency and read leveling there, then the controller
    takes the PHY. Returns the result; raises CalError when no phase or calibration works."""
    d = Dram(csr, config)
    dqs = DqsPhase(csr, d.phy["vco_hz"])
    period = round(56 * d.phy["vco_hz"] / (4 * d.phy["sys_hz"]))     # fine steps per tCK
    has_bist = has_csr(csr, "bist_start")
    table = dqs_scan(d, dqs, period, stride, csr=csr if has_bist else None, mib=mib, show=False)
    per, common, pick, run = margins(table, d.nm, period=period)
    if pick is None:
        raise CalError("no DQS phase works for every lane")
    dqs.move(dqs.steps() + pick)
    wl, rl, err = d.calibrate(verbose=False)
    if any(err) or min(wl) < 0:
        raise CalError(f"calibration at DQS step {dqs.steps()} failed: write latency {wl}, "
                       f"errors per lane {err}")
    d.hardware()
    res = {"dqs_steps": dqs.steps(), "window_steps": len(run) * stride,
           "window_ps": round(len(run) * stride * dqs.step_ps), "traffic_checked": has_bist,
           "write_latency": wl, "read": [{"taps": n, "bitslip": b, "tap": s + n // 2}
                                         for n, b, s in rl],
           "lanes": pass_map(table, d.nm, period)}
    log(f"DQS step {res['dqs_steps']}, common write window {res['window_ps']} ps"
        f"{'' if has_bist else ' (DFII check only: no BIST)'}, write latency {wl}, read windows "
        f"{[n for n, _, _ in rl]} taps")
    return res


def has_csr(csr, name):
    regs = getattr(csr, "regs", None) or getattr(getattr(csr, "csr", None), "regs", {})
    return (csr.name(name) if isinstance(csr, Chan) else name) in regs
