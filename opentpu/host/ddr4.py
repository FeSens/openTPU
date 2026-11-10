"""otpu-ddr4: the memory bring-up of a board with one LiteDRAM DDR4 core per DIMM (the BCU1525:
docs/bcu1525.md section 11), through the cores' CSRs. DIMM d's are in BAR0 at R_DDR4 + d x
DDR4_STRIDE (vivado/bd.tcl's M_AXI_MEMCAL<d>). Each core's CPU (LiteX's BIOS) calibrates its DIMM
at reset and logs on the UART (DIMM C1's core drives the board's); the host reads and tests what
the CSRs reach:

    python -m opentpu.host.ddr4 [status]    per DIMM: the core's identifier and scratch register
                                            (the CSR path), calibrated (cal_ready, or the BIOS's
                                            hand-over: DFII hardware control), the ECC and its
                                            counters; STATUS CALIB0 / CALIB1 / ECC_DED
    python -m opentpu.host.ddr4 bist [--dimm D] [--gib G] [--passes N]
                                            each DIMM's BIST: write, then read and check, G GiB
                                            from the DIMM's start, random then address data, the
                                            wrong beats per byte lane. It writes past the ECC
                                            (the check bits are left wrong), so it clears Board's
                                            scrub marker: the next BoardBackend scrubs the DRAM
                                            (a UDIMM core: no ECC, eight lanes)
    python -m opentpu.host.ddr4 ecc [--clear]
                                            the ECC counters: 64-bit words corrected and
                                            uncorrectable since the last clear
    python -m opentpu.host.ddr4 calibrate [--dimm D] [--vref PCT] [--gib G]
                                            the host's calibration of a UDIMM core's DIMM after
                                            the BIOS's (calibrate_dimm: write leveling, write
                                            latency, read leveling), then its BIST; cal_ready 1
                                            when both pass, the DIMM out of the controller's
                                            hands when not. It writes over the DRAM, as bist

Board channel j = 2s + c (slice s's channel c) is on DIMM CHANNEL_DIMM[j] (otpu_fpga_top_bcu's
dimm()). The cores' CSR map and PHY settings ship in opentpu/host/litedram4/ (the x4 ECC RDIMMs'
core: copies of boards/bcu1525/litedram/'s, tools/litedram/gen_core_ddr4.py's output) and
litedram4u/ (the x8 UDIMMs' core without ECC: boards/bcu1525/litedram_udimm/'s, whose ECC CSRs
read enable 0 and no errors). The host takes a DIMM's by its core's identifier ("... UDIMM"),
which both maps place alike (data_of). Only a card: the board model has no LiteDRAM cores
(present()).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from . import ddrcal
from . import regs as R

DATA = Path(__file__).with_name("litedram4")
DATA_UDIMM = Path(__file__).with_name("litedram4u")
CHANNEL_DIMM = (0, 1, 3, 2)     # board channel j -> DIMM C<d> (otpu_fpga_top_bcu dimm())
SCRATCH_RESET = 0x12345678      # LiteX's ctrl_scratch after the core's reset
IDENT_MAX = 256                 # bytes of identifier_mem read at most
DFII_SEL = 1                    # sdram_dfii_control bit0: the controller has the PHY
TAPS = 512                      # USPDDRPHY's delay lines' taps (and its DQS counter's wrap)
WL_HYST = 8                     # a write leveling edge: WL_HYST taps reading 0, then WL_HYST 1
MR1_WL = 1 << 7                 # MR1 A7: write leveling
MR6_TRAIN = 1 << 7              # MR6 A7: VrefDQ training (the value latches on its exit)


def csr_map(path: Path = DATA / "csr.csv") -> tuple[dict, dict]:
    """csr.csv -> ({register: (offset, 32-bit words)}, {block: offset}), the offsets in the CSR
    region the host's port reaches (16 bits: the addresses less the region's base, 0xF000_0000
    with the core's CPU, whose ROM is at 0)."""
    regs, bases = {}, {}
    for row in csv.reader(open(path)):
        if row and row[0] == "csr_base":
            bases[row[1]] = int(row[2], 16)
        elif row and row[0] == "csr_register":
            regs[row[1]] = (int(row[2], 16), int(row[3]))
    region = min(bases.values()) & ~0xFFFF
    regs = {n: (a - region, w) for n, (a, w) in regs.items()}
    bases = {n: a - region for n, a in bases.items()}
    if any(a + 4 * w > R.DDR4_STRIDE for a, w in regs.values()):
        raise ValueError(f"{path}: a CSR outside the {R.DDR4_STRIDE:#x}-byte window")
    return regs, bases


def present(t) -> bool:
    """t is a card (the board model has no CSR windows: `ecc`, as memcal.ecc) whose bitstream
    (register map 4) names a target with LiteDRAM DDR4 cores."""
    if not getattr(t, "ecc", False) or R.regmap(t.reg_read(R.R_REGMAP)) < R.CFG_SINCE:
        return False
    from opentpu.target import by_platform
    tg = by_platform(t.reg_read(R.R_CFG_PLATFORM))
    return tg is not None and tg.platform.get("mem") == "ddr4" \
        and tg.platform.get("mem_ctrl") == "litedram"


def dimms(t) -> list[int]:
    """The DIMMs of the bitstream's channels, in channel order."""
    c = R.cfg([t.reg_read(o) for o in R.CFG_REGS])
    return [CHANNEL_DIMM[j] for j in range(c["S"] * c["channels"])]


def data_of(t, d: int) -> Path:
    """DIMM d's core's CSR map and PHY settings: litedram4u/ for a UDIMM core (its identifier
    ends "UDIMM"), litedram4/ otherwise. Both maps have identifier_mem at the same offset."""
    return DATA_UDIMM if ident(t, d, DATA).endswith("UDIMM") else DATA


def csr(t, d: int, data: Path | None = None) -> ddrcal.WordCsr:
    """DIMM d's core's CSRs (w / r by csr.csv's names)."""
    regs, _ = csr_map((data or data_of(t, d)) / "csr.csv")
    return ddrcal.WordCsr(regs, t.reg_read, t.reg_write, base=R.R_DDR4 + d * R.DDR4_STRIDE)


def ident(t, d: int, data: Path = DATA) -> str:
    """DIMM d's core's identifier string (identifier_mem: a byte per word; at the same offset in
    every core's map)."""
    _, bases = csr_map(data / "csr.csv")
    a = R.R_DDR4 + d * R.DDR4_STRIDE + bases["identifier_mem"]
    out = bytearray()
    for i in range(IDENT_MAX):
        b = t.reg_read(a + 4 * i)
        if b == 0 or b > 0xFF:          # the end, or no core answering (all ones)
            break
        out.append(b)
    return out.decode("ascii", "replace")


def status(t, d: int, data: Path | None = None) -> dict:
    """DIMM d: its core's identifier, the scratch register's write / read-back (restored),
    calibrated (cal_ready: a host calibration; DFII hardware control: the BIOS's hand-over after
    a passing init), the ECC enable and counters (a UDIMM core's: off, 0)."""
    data = data or data_of(t, d)
    c = csr(t, d, data)
    was = c.r("ctrl_scratch")
    pat = 0xA5C3_0F00 | d
    c.w("ctrl_scratch", pat)
    ok = c.r("ctrl_scratch") == pat
    c.w("ctrl_scratch", was)
    dfii = c.r("sdram_dfii_control")
    return {"dimm": d, "ident": ident(t, d, data), "scratch_ok": ok, "scratch": was,
            "cal_ready": bool(c.r("cal_ready") & 1), "dfii_hw": bool(dfii & DFII_SEL),
            "calibrated": bool(c.r("cal_ready") & 1 or dfii & DFII_SEL),
            "ecc_enable": bool(c.r("ecc_enable") & 1),
            "ecc_sec": c.r("ecc_sec_errors"), "ecc_ded": c.r("ecc_ded_errors")}


def ecc(t, clear: bool = False, data: Path | None = None) -> dict | None:
    """Every DIMM's ECC counters {d: (corrected, uncorrectable)}, cleared after with `clear`;
    None when t is not a DDR4 card (Board.scrub calls it on every board)."""
    if not present(t):
        return None
    out = {}
    for d in dimms(t):
        c = csr(t, d, data)
        out[d] = (c.r("ecc_sec_errors"), c.r("ecc_ded_errors"))
        if clear:
            c.w("ecc_clear", 1)
    return out


def sys_hz(data: Path = DATA) -> float:
    """The controllers' clock (sdram_init.py's phy settings): one 64-byte beat per cycle."""
    ns = {}
    exec((data / "sdram_init.py").read_text(), ns)
    return float(ns["phy"]["sys_hz"])


def bist(t, d: int, gib: float = 1.0, passes: int = 1, data: Path | None = None) -> list[dict]:
    """DIMM d's BIST over its first `gib` GiB: per pass, random then address data, written
    then read back and checked (ddrcal.bist; the beats are the 576-bit raw port's, past the
    ECC). Clears SCRATCH's DRAM_INIT, so the next Board.scrub writes the DRAM again."""
    data = data or data_of(t, d)
    c = csr(t, d, data)
    hz = sys_hz(data)
    beats = int(gib * (1 << 30)) // R.DRAM_BEAT
    out = []
    t.reg_write(R.R_SCRATCH, 0)
    for p in range(passes):
        for dmode, name in ((0, "random"), (2, "address")):
            seed = (0x0123456789ABCDEF ^ (p * 0x9E3779B97F4A7C15)) & ((1 << 64) - 1)
            w = ddrcal.bist(c, beats, dmode, seed, hz)
            r = ddrcal.bist(c, beats, dmode | 1, seed, hz)
            out.append({"dimm": d, "pass": p, "data": name, "beats": beats,
                        "write_gbs": round(w["gbs"], 2), "read_gbs": round(r["gbs"], 2),
                        "errors": r["errors"], "lanes": r["lanes"], "bits": r["bits"],
                        "ok": r["errors"] == 0 and r["beats"] == beats})
    return out


class UspDram(ddrcal.Dram):
    """ddrcal's DFII commands and read scans on a DIMM of LiteDRAM's USPDDRPHY (a DDR4 core's):
    its write leveling (each lane's DQS ODELAY against the DRAM's feedback of CK at its DQS), the
    DQ ODELAYs with their DQS's, VrefDQ through MR6."""
    def mode(self, ba: int) -> int:
        """MR<ba>'s value in the init sequence (the BIOS's)."""
        return next(a for c, a, b, *_ in self.init_sequence if b == ba and c.startswith("Load Mode"))

    def dqs_count(self, m: int) -> int:
        self.sel(1 << m)
        n = self.c.r("ddrphy_wdly_dqs_inc_count")
        self.sel(0)
        return n

    def dqs_to(self, m: int, target: int):
        """The DQS delay's counter has no reset on this PHY (wdly_dqs_rst leaves it): increments
        round to the target."""
        self.strobe("ddrphy_wdly_dqs_inc", 1 << m, (target - self.dqs_count(m)) % TAPS)

    def dq_to(self, m: int, target: int):
        self.strobe("ddrphy_wdly_dq_rst", 1 << m)
        self.strobe("ddrphy_wdly_dq_inc", 1 << m, target)

    def set_wdelay(self, m: int, n: int):
        """Lane m's DQS at count n, its DQ with it (the PHY adds tCK / 4 to the DQS's: the
        strobe in the data eye's centre)."""
        self.dqs_to(m, n)
        self.dq_to(m, n)
        self.wd[m] = n

    def tck_taps(self) -> int:
        """Taps per tCK: 4 x the DQS's static delay (DELAY_VALUE tCK / 4; half_sys8x_taps is
        module 0's DQS CNTVALUEOUT, the count plus that)."""
        return 4 * ((self.c.r("ddrphy_half_sys8x_taps") - self.dqs_count(0)) % TAPS)

    def wl_scan(self, span: int, loops: int = 8) -> list[list[int]]:
        """Per lane, the DRAM's write leveling feedback (CK's level at the DQS, the majority of
        `loops` strobes) at DQS counts 0..span-1; the DQS ends at span."""
        mr1 = self.mode(1)
        self.cmd(0, self.RAS | self.WE | self.CS, 0x400)                   # precharge all
        self.cmd(0, self.RAS | self.CAS | self.WE | self.CS, mr1 | MR1_WL, 1)
        self.c.w("ddrphy_wlevel_en", 1)
        time.sleep(0.001)
        out = []
        try:
            for m in range(self.nm):
                self.dqs_to(m, 0)
                self.sel(1 << m)
                bits = []
                for _ in range(span):
                    ones = 0
                    for _ in range(loops):
                        self.c.w("ddrphy_wlevel_strobe", 1)
                        ones += ((self.c.r("sdram_dfii_pi0_rddata") >> (self.db + 8 * m)) & 0xFF) != 0
                    bits.append(int(2 * ones > loops))
                    self.c.w("ddrphy_wdly_dqs_inc", 1)
                self.sel(0)
                out.append(bits)
        finally:
            self.c.w("ddrphy_wlevel_en", 0)
            self.cmd(0, self.RAS | self.CAS | self.WE | self.CS, mr1, 1)
        return out

    @staticmethod
    def edges(bits: list[int]) -> list[int]:
        """The scan's 0 -> 1 edges (CK's rising edges at the DQS)."""
        return [j for j in range(WL_HYST, len(bits) - WL_HYST + 1)
                if not any(bits[j - WL_HYST:j]) and all(bits[j:j + WL_HYST])]

    def set_vrefdq(self, value: int):
        """VrefDQ (MR6 A5:A0, range 1 with A6 0: 60 + 0.65 x value % of VDDQ) through MR6's
        training mode: enter with the value, exit with it (JEDEC tVREFDQE / tVREFDQX, 150 ns:
        well under the CSR writes' spacing). The init's MR6 keeps its other fields."""
        self.cmd(0, self.RAS | self.WE | self.CS, 0x400)                   # precharge all
        mr6 = (self.mode(6) & ~0xFF) | value
        for v in (mr6 | MR6_TRAIN, mr6 | MR6_TRAIN, mr6):
            self.cmd(0, self.RAS | self.CAS | self.WE | self.CS, v, 6)
            time.sleep(0.001)


def vref_value(pct: float) -> int:
    """MR6 A5:A0 for VrefDQ `pct` % of VDDQ in range 1 (60 + 0.65 x value, 0..50)."""
    return max(0, min(50, round((pct - 60) / 0.65)))


def _bitstr(bits: list[int], cols: int = 64) -> str:
    k = max(1, len(bits) // cols)
    return "".join(str(bits[j]) for j in range(0, len(bits), k))


def calibrate_dimm(c, data: Path = DATA_UDIMM, vref: float | None = None,
                   seeds=ddrcal.SEEDS, log=print) -> dict:
    """A DIMM's calibration from the host on its core's CSRs `c` (csr()), after the BIOS's:
    write leveling per lane at the last 0 -> 1 edge of its DQS scan (the earlier edge when the
    write latency finds no bitslip there), the DQ delays with the DQS's, ddrcal's write latency
    and read leveling (every lane per scan), then the DIMM back to the controller. The BIOS takes
    the longest 1-window of its scan, and one that starts before the scan (the DQS a tCK early:
    a write bitslip of 8, past the PHY's) left DIMM C1's lane 6 without a write latency (2026-10-
    21); a lane whose scan has no edge but opens with a 1-window over tCK / 4 long takes the
    BIOS's choice (0). Keeps the BIOS's command / clock delay and mode registers (VrefDQ at
    `vref` % of VDDQ with it set: set_vrefdq). EN_VTC is off while the delays move (UltraScale's
    delay lines take no steps with it on) and on after. cal_ready is 0 throughout; a calibration
    that fails leaves the DIMM in software control (the core's ready 0: the DRAM not refreshed).
    Returns {"ok", "tck_taps", "edges", "wdelay", "write_latency", "read", "errors"}."""
    d = UspDram(c, data)
    if d.phy.get("rdimm"):
        raise ddrcal.CalError("an RDIMM core: the host calibrates the UDIMM core's DIMMs only "
                              "(an RDIMM's write leveling goes through its RCD)")
    d.wd = [0] * d.nm
    c.w("cal_ready", 0)
    c.w("ddrphy_en_vtc", 0)
    d.ctl(d.CTL_CKE | d.CTL_ODT | d.CTL_RESET_N)
    ok = False
    try:
        if vref is not None:
            d.set_vrefdq(vref_value(vref))
            log(f"  VrefDQ {60 + 0.65 * vref_value(vref):.2f} % (MR6 value {vref_value(vref)})")
        tck = d.tck_taps()
        span = TAPS - tck // 4                  # the DQS's own delay is count + tCK / 4 <= 511
        bios = [d.dqs_count(m) for m in range(d.nm)]
        cands = []
        log(f"  write leveling: tCK {tck} taps, DQS counts 0..{span - 1} (cdly "
            f"{c.r('ddrphy_cdly_value')}):")
        for m, bits in enumerate(d.wl_scan(span)):
            e = d.edges(bits)
            if not e and bits[0] and (bits + [0]).index(0) > tck // 4:
                e = [0]
            cands.append(e)
            log(f"    m{m}: |{_bitstr(bits)}| edges {e} (BIOS {bios[m]})")
        pick = [len(e) - 1 for e in cands]                 # the last edge first
        for m in range(d.nm):
            d.set_wdelay(m, cands[m][pick[m]] if cands[m] else 0)
        while True:
            wl, _ = d.write_latency(seeds, verbose=False)
            retry = [m for m in range(d.nm) if wl[m] < 0 and pick[m] > 0]
            if not retry:
                break
            for m in retry:
                pick[m] -= 1
                d.set_wdelay(m, cands[m][pick[m]])
                log(f"    m{m}: no write latency at the last edge: edge {cands[m][pick[m]]}")
        log(f"  write delays {d.wd}, write latency (bitslips) "
            + " ".join("-" if w < 0 else str(w) for w in wl))
        rl, err = d.read_leveling(seeds, verbose=False)
        log("  read leveling: " + ", ".join(f"m{m} b{b} {n} taps" + (f" @{s + n // 2}" if n else "")
                                            for m, (n, b, s) in enumerate(rl)))
        log("  check at the chosen taps: " + ("pass" if not any(err) else f"errors per lane {err}"))
        ok = not any(err) and min(wl) >= 0 and all(n > 0 for n, _, _ in rl)
    finally:
        if ok:
            d.hardware()
        c.w("ddrphy_en_vtc", 1)
    return {"ok": ok, "tck_taps": tck, "edges": cands, "wdelay": list(d.wd), "write_latency": wl,
            "read": [{"taps": n, "bitslip": b, "tap": s + n // 2} for n, b, s in rl],
            "errors": err}


# repair_lanes' changes to a lane's write side from the BIOS's, in the order tried: (its DQS and
# DQ write delay in eighths of tCK, its write bitslip's offset), the smallest moves first
REPAIR_TRIES = ((-1, 0), (1, 0), (-2, 0), (2, 0), (0, 1), (0, -1), (-1, 1), (1, -1), (-2, 1),
                (2, -1))


def write_eye(c, d: "UspDram", m: int, wd: int, tck: int, hz: float, step: int, mib: int,
              seed: int = 0x0123456789ABCDEF) -> dict:
    """Lane m's write eye at its read settings now: its DQS and DQ write delay over wd - tCK / 2
    .. wd + tCK / 2 (within the delay line) by `step` taps, at each a BIST written and read over
    `mib` MiB (random data) and the lane's wrong beats; the lane left at the centre of the clean
    run around wd (the longest when wd's point is not clean). The read side does not move with
    the writes' delay, so the read settings hold throughout. Returns {"lo", "step", "errors",
    "wdelay", "taps"}; taps 0 (and the delay back at wd) when no point is clean."""
    beats = mib * (1 << 20) // 64
    lo, hi = max(0, wd - tck // 2), min(TAPS - tck // 4 - 1, wd + tck // 2)
    d.set_wdelay(m, lo)
    errs = []
    for _ in range(lo, hi + 1, step):
        ddrcal.bist(c, beats, 0, seed, hz)
        errs.append(ddrcal.bist(c, beats, 1, seed, hz)["lanes"][m])
        d.strobe("ddrphy_wdly_dqs_inc", 1 << m, step)
        d.strobe("ddrphy_wdly_dq_inc", 1 << m, step)
    k = (wd - lo) // step
    runs, start = [], None
    for i, e in enumerate(errs + [1]):
        if e == 0 and start is None:
            start = i
        elif e and start is not None:
            runs.append((start, i - start))
            start = None
    run = next((r for r in runs if r[0] <= k < r[0] + r[1]),
               max(runs, key=lambda r: r[1], default=None))
    best = wd if run is None else lo + (run[0] + run[1] // 2) * step
    d.set_wdelay(m, best)
    return {"lo": lo, "step": step, "errors": errs, "wdelay": best,
            "taps": 0 if run is None else run[1] * step}


def repair_lanes(c, lanes, wbitslip, data: Path = DATA_UDIMM, step: int = 32, fine: int = 4,
                 mib: int = 16, min_points: int = 3, wstep: int = 16, wmib: int = 256,
                 log=print) -> dict:
    """Lanes the BIOS left without a read window (its read leveling found none at any bitslip),
    repaired from the host with the DIMM in hardware control, every other lane as the BIOS set
    it. Each lane's write side moves from the BIOS's settings by REPAIR_TRIES in turn (its DQS
    and DQ write delay by eighths of tCK, its write bitslip by one), each tried with a BIST grid
    over its read bitslips and every `step`-th tap (ddrcal.bist_lane_grid), until one reads a
    clean run of `min_points` points at a bitslip (the longest of all when none does); then the
    read tap at the centre of that bitslip's clean run in a scan of every `fine`-th tap, and the
    write delay at the centre of the lane's write eye at those reads (write_eye: every `wstep`-th
    tap over a tCK, a `wmib` MiB BIST at each). On
    2026-10-26 (DDR4-1200) DIMM C1's lane 6 read 765 wrong beats in 16 MiB at the BIOS's best
    point (its write leveling's DQS count 206 of a 484-tap tCK), none with its write delay an
    eighth or a quarter of tCK down or its write bitslip one up (the BIOS's write latency steps
    it by a tCK, two at a time), and 2291 an eighth up. `wbitslip`: each lane's write bitslip
    as the BIOS set it (its log: the PHY reads none back). EN_VTC is off while the delays move,
    on after. A lane with no clean run anywhere goes back to the BIOS's write settings. Returns
    {lane: {"ok", "wdelay", "wbitslip", "bitslip", "tap", "taps", "tries", "write_taps", ...}},
    `taps` the fine scan's clean run, `write_taps` the write eye's."""
    d = UspDram(c, data)
    hz, nb = sys_hz(data), d.phy["bitslips"]
    d.wd = [d.dqs_count(m) for m in range(d.nm)]
    tck = d.tck_taps()
    out = {}
    c.w("ddrphy_en_vtc", 0)
    try:
        for m in lanes:
            wd0, wb0 = d.wd[m], wbitslip[m]
            tries, best = [], None
            for k, db in REPAIR_TRIES:
                wd, wb = wd0 + k * (tck // 8), (wb0 + db) % nb
                if not 0 <= wd < TAPS - tck // 4:
                    continue
                d.set_wdelay(m, wd)
                d.set_wbitslip(m, wb)
                n, b, _ = ddrcal.grid_best(ddrcal.bist_lane_grid(c, d, m, hz, step, mib)["rows"])
                tries.append({"wdelay": wd, "wbitslip": wb, "points": n, "bitslip": b})
                log(f"  m{m}: write delay {wd}, write bitslip {wb}: "
                    + (f"{n} clean points at bitslip {b}" if n else "no clean point"))
                if n and (best is None or n > best[0]):
                    best = (n, wd, wb, b)
                if n >= min_points:
                    break
            r = {"ok": False, "wdelay": wd0, "wbitslip": wb0, "bitslip": None, "tap": None,
                 "taps": 0, "tries": tries}
            if best:
                _, wd, wb, b = best
                d.set_wdelay(m, wd)
                d.set_wbitslip(m, wb)
                g = ddrcal.bist_lane_grid(c, d, m, hz, fine, mib, bitslips=[b])
                n, _, s = ddrcal.grid_best(g["rows"])
                if n:
                    tap = (s + n // 2) * fine
                    d.set_rbitslip(m, b)
                    d.set_rdelay(m, tap)
                    r.update(ok=True, wdelay=wd, wbitslip=wb, bitslip=b, tap=tap, taps=n * fine)
                    # the try that read clean can sit near its write eye's edge (DIMM C3's lane 1,
                    # 2026-10-26: 2 wrong beats in 8 GiB, the same 2 every pass): centre it
                    e = write_eye(c, d, m, wd, tck, hz, wstep, wmib)
                    r.update(wdelay=e["wdelay"], write_taps=e["taps"], write_eye=e["errors"],
                             write_lo=e["lo"])
                    log(f"  m{m}: write eye {e['taps']} taps clean at {wmib} MiB, write delay "
                        f"{wd} -> {e['wdelay']}")
            if not r["ok"]:
                d.set_wdelay(m, wd0)
                d.set_wbitslip(m, wb0)
            log(f"  m{m}: " + (f"write delay {r['wdelay']}, write bitslip {r['wbitslip']}, read "
                               f"bitslip {r['bitslip']} tap {r['tap']} ({r['taps']} taps clean)"
                               if r["ok"] else "not repaired (the BIOS's write settings)"))
            out[m] = r
    finally:
        c.w("ddrphy_en_vtc", 1)
    return out


def centre_writes(c, lanes=None, data: Path = DATA_UDIMM, wstep: int = 16, wmib: int = 256,
                  log=print) -> dict:
    """Each lane's write eye at its read settings now (write_eye) and its write delay at the
    eye's centre, the DIMM in hardware control: after the BIOS (and repair_lanes), whose write
    leveling puts the DQS where the DRAM's feedback turns, not in the middle of the writes'
    window. EN_VTC off while the delays move, on after. Returns {lane: write_eye's result, with
    "wdelay0" the delay before}."""
    d = UspDram(c, data)
    d.wd = [d.dqs_count(m) for m in range(d.nm)]
    tck, hz = d.tck_taps(), sys_hz(data)
    out = {}
    c.w("ddrphy_en_vtc", 0)
    try:
        for m in range(d.nm) if lanes is None else lanes:
            wd0 = d.wd[m]
            e = write_eye(c, d, m, wd0, tck, hz, wstep, wmib)
            e["wdelay0"] = wd0
            out[m] = e
            log(f"  m{m}: write eye {e['taps']} taps clean at {wmib} MiB (from {e['lo']}), "
                f"write delay {wd0} -> {e['wdelay']}")
    finally:
        c.w("ddrphy_en_vtc", 1)
    return out


def calibrate(t, d: int, vref: float | None = None, gib: float = 1.0, log=print) -> dict:
    """DIMM d's host calibration (calibrate_dimm) and its BIST over `gib` GiB; cal_ready 1
    when both pass, else the DIMM back in software control (ready 0). Clears SCRATCH's
    DRAM_INIT first: the scans write row 0 and the BIST the rest."""
    data = data_of(t, d)
    c = csr(t, d, data)
    t.reg_write(R.R_SCRATCH, 0)
    r = calibrate_dimm(c, data, vref, log=log)
    r["bist"] = bist(t, d, gib, data=data) if r["ok"] else []
    r["ok"] = r["ok"] and all(x["ok"] for x in r["bist"])
    if r["ok"]:
        c.w("cal_ready", 1)
    else:
        c.w("sdram_dfii_control", UspDram.CTL_CKE | UspDram.CTL_ODT | UspDram.CTL_RESET_N)
    return r


def describe(s: dict, j: int) -> str:
    # the BIOS hands the PHY to the controller (DFII hardware control) whether or not its
    # leveling and memtest passed: its hand-over alone says nothing of the DIMM (bist does)
    cal = ("calibrated (host)" if s["cal_ready"] else "calibrated (BIOS, unchecked)"
           if s["dfii_hw"] else "NOT calibrated")
    return (f"DIMM C{s['dimm']} (channel {j}, slice {j // 2}): '{s['ident']}', scratch "
            f"{'ok' if s['scratch_ok'] else 'FAILED'} ({s['scratch']:#010x}), {cal}, ECC "
            f"{'on' if s['ecc_enable'] else 'off'}: {s['ecc_sec']} corrected, {s['ecc_ded']} "
            f"uncorrectable")


def main(argv: list[str] | None = None, open_transport=None) -> int:
    ap = argparse.ArgumentParser(prog="otpu-ddr4", description=__doc__.split("\n")[0])
    ap.add_argument("cmd", nargs="?", default="status",
                    choices=("status", "bist", "ecc", "calibrate"))
    ap.add_argument("--dev", default="/dev/xdma0")
    ap.add_argument("--dimm", type=int, action="append", help="DIMM (default: every one)")
    ap.add_argument("--gib", type=float, default=1.0,
                    help="bist, calibrate: GiB per DIMM (from its start)")
    ap.add_argument("--vref", type=float, help="calibrate: VrefDQ, %% of VDDQ (MR6 range 1, "
                    "60-92.5; default: the init's)")
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--clear", action="store_true",
                    help="ecc: clear the counters after (every DIMM's)")
    a = ap.parse_args(argv)
    if open_transport is None:
        from .board import XdmaTransport
        # the BIST and the calibration write over the DRAM: they hold the device lock (a DMA
        # transport's); status and ecc only read registers, as otpu-smi
        t = XdmaTransport(a.dev, dma=a.cmd in ("bist", "calibrate"))
    else:
        t = open_transport()
    try:
        return _run(a, t)
    finally:
        if open_transport is None:
            t.close()


def _run(a, t) -> int:
    if not present(t):
        print("otpu-ddr4: not a DDR4 LiteDRAM card (register map 4 with a DDR4 target, "
              "CFG_PLATFORM)", file=sys.stderr)
        return 2
    chans = {d: j for j, d in enumerate(dimms(t))}
    sel = a.dimm or list(chans)
    bad = [d for d in sel if d not in chans]
    if bad:
        print(f"otpu-ddr4: no DIMM {bad} (the bitstream's: {list(chans)})", file=sys.stderr)
        return 2
    ok = True
    if a.cmd == "status":
        st = t.reg_read(R.R_STATUS)
        print(f"STATUS {st:#010x}: CALIB0 {int(bool(st & R.ST_CALIB0))} CALIB1 "
              f"{int(bool(st & R.ST_CALIB1))} ECC_DED {int(bool(st & R.ST_ECC_DED))}")
        for d in sel:
            s = status(t, d)
            print(describe(s, chans[d]))
            ok &= s["scratch_ok"] and s["calibrated"]
    elif a.cmd == "ecc":
        for d, (sec, ded) in ecc(t, clear=a.clear).items():
            if d in sel:
                print(f"DIMM C{d} (channel {chans[d]}): {sec} corrected, {ded} uncorrectable")
                ok &= ded == 0
    elif a.cmd == "calibrate":
        for d in sel:
            print(f"DIMM C{d} (channel {chans[d]}):")
            t0 = time.time()
            try:
                r = calibrate(t, d, a.vref, a.gib)
            except ddrcal.CalError as e:
                print(f"otpu-ddr4: DIMM C{d}: {e}", file=sys.stderr)
                return 2
            for x in r["bist"]:
                print(f"  BIST {x['data']}: errors {x['errors']} (per lane {x['lanes']}, bits "
                      f"{x['bits']:#x})")
            print(f"DIMM C{d}: calibration {'passed' if r['ok'] else 'FAILED'} in "
                  f"{time.time() - t0:.0f} s" + ("" if r["ok"] else
                  ": left in software control (NOT calibrated)"))
            ok &= r["ok"]
        print("otpu-ddr4: the calibration wrote over the DRAM: the next BoardBackend scrubs it")
    else:
        for d in sel:
            for r in bist(t, d, a.gib, a.passes):
                print(f"DIMM C{d} pass {r['pass']} {r['data']}: write {r['write_gbs']:.2f} GB/s, "
                      f"read {r['read_gbs']:.2f} GB/s, errors {r['errors']} (per lane "
                      f"{r['lanes']}, bits {r['bits']:#x})")
                ok &= r["ok"]
        print("otpu-ddr4: the BIST wrote past the ECC: the next BoardBackend scrubs the DRAM")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
