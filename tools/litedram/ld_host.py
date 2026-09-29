#!/usr/bin/env python3
"""Host side of the LiteDRAM test image (tools/litedram/ld_test.py): its DDR3 channels at 72 bits,
calibrated from the host over BAR0 (/dev/xdma0_user), then tested with the image's BIST.

    python3 ld_host.py BUILD_DIR all          # init, DQS scan, calibration, BIST (the report)
    python3 ld_host.py BUILD_DIR info         # identifier, DQS phase, scratch register check
    python3 ld_host.py BUILD_DIR cal          # init + write latency + read leveling at the
                                              # current DQS phase
    python3 ld_host.py BUILD_DIR bist [--gib 2]
    python3 ld_host.py BUILD_DIR temp [--minutes 35 --every 300]   # the temperature run
    python3 ld_host.py BUILD_DIR wscan [--wl 0,6 --stride 1]  # DQS scan, write latency forced
    python3 ld_host.py BUILD_DIR g1 --eighths N   # WL7DDRPHY image: group 1's offset (DRP)

WL7DDRPHY images (ld_test.py --phy wl, docs/litedram.md section 8): the DQS phase moves each
channel's whole write side (DQ with its DQS) in two clock groups; `all` and `temp` pick group 1's
static offset (ddrcal.calibrate_groups) before the common phase.
    python3 ld_host.py BUILD_DIR selftest     # the calibration logic against a simulated PHY

--ch 0, 1 or both (default: every channel of the image, one after the other; `temp` runs them
together). Channel 0's CSRs keep the one-channel image's names, channel 1's carry a 1 (Chan).

The temperature run: per channel a DQS scan with a BIST per phase picks the phase (the centre of
the window common to all lanes) and calibrates; then both channels run the BIST back to back for
--minutes (2 GiB write + read-check per pass, random and address data in turn, new seeds), which
also heats the card, and every --every seconds each channel's window is scanned again from the
phase it runs at (so the scan's offsets are that phase's margins) before it goes back to that
phase and recalibrates (write latency, read leveling), as production would on a temperature
change. The FPGA die temperature (XADC) and the board's LM73 (over the image's I2C pins with
opentpu.host.i2c, when that package is importable) are logged with every line.

BUILD_DIR holds csr.csv and sdram_init.py (written by ld_test.py). Only numpy-free stdlib.

The calibration itself (DFII init, write latency, read leveling, the DQS phase scan, the BIST
helpers, a simulated PHY) is opentpu/host/ddrcal.py; on a card host whose installed opentpu
predates it, a copy next to this file is used.
"""
import argparse
import mmap
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))          # the repo's opentpu before an installed one
try:
    from opentpu.host import ddrcal
except ImportError:         # a card host whose installed opentpu predates ddrcal: a copy here
    sys.path.insert(0, str(HERE))
    import ddrcal
sys.modules.setdefault("ddrcal", ddrcal)
from ddrcal import (MIN_WINDOW, CalError, Chan, Dram, DqsPhase, FakeBoard, FakeCsr,  # noqa: E402
                    WriteClocks, bist_read_scan, bist, bist_start, bist_wait, calibrate_groups,
                    csr_map, dqs_scan, group_windows, margins, offsets, pass_map, run_bist)
# ------------------------------------------------------------------------------ CSR access
class Bar0:
    """CSRs over /dev/xdma0_user: one 32-bit load / store per word (see board.py's transport)."""
    def __init__(self, build: Path, dev: str = "/dev/xdma0_user"):
        self.regs = csr_map(build / "csr.csv")
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


class Temps:
    """The FPGA die temperature (XADC) and the board's LM73, read over the image's i2c_ctrl /
    i2c_in (production's I2C_CTRL / I2C_IN bits) with opentpu.host.i2c when it is importable."""
    def __init__(self, csr):
        self.c, self.bus, self.addr, self.note = csr, None, None, ""
        try:
            from opentpu.host import i2c
        except ImportError as e:
            self.note = f"no board sensor (opentpu.host.i2c: {e})"
            return

        class T:                        # a transport as i2c.Bus wants it
            devname = None              # no I2C lock file: the test image is ours alone
            def reg_read(_, a):
                return csr.r({0x220: "i2c_ctrl", 0x224: "i2c_in"}[a])
            def reg_write(_, a, v):
                csr.w({0x220: "i2c_ctrl"}[a], v)
        self.i2c, self.bus = i2c, i2c.Bus(T(), 0)
        try:
            with self.bus:
                for a in i2c.LM73_ADDRS:
                    if i2c.lm73_identify(self.bus, a):
                        self.addr = a
                        break
            self.note = f"LM73 at {self.addr:#x}" if self.addr is not None else "no LM73 found"
        except Exception as e:          # a stuck bus must not stop the DRAM run
            self.note = f"LM73 bus error: {e}"

    def read(self):
        d = {"fpga_c": self.c.r("xadc_temperature") * 503.975 / 4096 - 273.15}
        if self.addr is not None:
            try:
                with self.bus:
                    d["board_c"] = self.i2c.lm73_temp(
                        int.from_bytes(self.bus.read(self.addr, 0x00, 2), "big"))
            except Exception:
                pass
        return d

    @staticmethod
    def fmt(d):
        return f"FPGA {d['fpga_c']:.1f} C" + (f", board {d['board_c']:.1f} C" if "board_c" in d else "")


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
    # two channels (Chan's names), the temperature run's loop and its verdict
    assert Chan(None, 1).name("ddrphy_dly_sel") == "ddrphy1_dly_sel"
    assert Chan(None, 1).name("xadc_temperature") == "xadc_temperature"
    assert Chan(None, 0).name("bist_start") == "bist_start"
    board = FakeBoard(build)
    a = argparse.Namespace(build=build, minutes=0.05, every=1.0, stride=8, gib=1 / 1024)
    assert temp_run(a, board, [0, 1], 112) == 0
    for f in board.ch:
        assert all(f.good(m) for m in range(f.nm))
    # the forced write latency scan: each lane's run at its latency is its write range
    import json
    import tempfile
    board = FakeBoard(build)
    tmp = Path(tempfile.mkdtemp())
    assert wl_scan(argparse.Namespace(build=build, stride=4, mib=1, wl="0,2,4", json_dir=tmp),
                   Chan(board, 1), 112) == 0
    s = json.loads((tmp / "wscan_ch1.json").read_text())["summary"]
    f = board.ch[1]
    for m in range(f.nm):
        r = s[str(m)][str(f.WB[m])]
        assert len(r) == 1 and r[0]["first"] - 4 < f.WLO[m] <= r[0]["first"] \
            and r[0]["last"] <= f.WHI[m] < r[0]["last"] + 4, (m, r, f.WLO[m], f.WHI[m])
        assert all(not s[str(m)][str(w)] for w in (0, 2, 4) if w != f.WB[m]), m
    print("selftest: PASS")


# ------------------------------------------------------------------------------ temperature run
def temp_run(a, csr, chans, period):
    temps = Temps(csr)
    t0 = time.time()
    stamp = lambda: f"{int((time.time() - t0) // 60):3d}:{int((time.time() - t0) % 60):02d}"
    print(f"temperature run: channels {chans}, {a.minutes:g} min of BIST, window rescans every "
          f"{a.every:g} s (scan stride {a.stride}); {temps.note}")
    print(f"{stamp()} {Temps.fmt(temps.read())}")
    st = {}
    for ch in chans:
        c = Chan(csr, ch)
        d = Dram(c, a.build)
        dqs = DqsPhase(c, d.phy["vco_hz"])
        groups = d.phy["groups"][str(ch)] if d.phy.get("phy") == "wl" else None
        if groups:                  # group 1's offset first (it resets the DQS phase to 0)
            try:
                res = calibrate_groups(d, dqs, WriteClocks(c), groups, period, stride=a.stride,
                                       csr=c, log=lambda s: print(f"{stamp()} ch{ch} {s.strip()}"))
            except CalError as e:
                print(f"channel {ch}: {e}: FAIL")
                return 1
            table, pick, run = res["scan1"], res["pick"], res["run"]
            dqs.move(0)             # the scan's start: the pick is from there
        else:
            table = dqs_scan(d, dqs, period, a.stride, csr=c, show=False)
            per, common, pick, run = margins(table, d.nm, period=period)
        print(f"{stamp()} channel {ch} scan from step {dqs.steps()}, {Temps.fmt(temps.read())}:")
        for m, line in enumerate(pass_map(table, d.nm, period)):
            print(f"    m{m}{f' g{groups[m]}' if groups else ''} {line}")
        if pick is None:
            print(f"channel {ch}: no DQS phase works for every lane: FAIL")
            return 1
        dqs.move(dqs.steps() + pick)
        wl, rl, err = d.calibrate(verbose=False)
        d.hardware()
        if any(err) or min(wl) < 0:
            print(f"channel {ch}: calibration at step {dqs.steps()}: FAIL")
            return 1
        print(f"    common window {len(run) * a.stride} steps ({len(run) * a.stride * dqs.step_ps:.0f} ps);"
              f" runs at its centre, step {dqs.steps()}; write latency {wl}, read windows "
              f"{[n for n, _, _ in rl]}")
        st[ch] = {"c": c, "d": d, "dqs": dqs, "home": dqs.steps(), "passes": 0, "errors": 0,
                  "groups": groups, "group_scans": [],
                  "lanes": [0] * d.nm, "scans": [(0.0, len(run) * a.stride, offsets(
                      [(k - pick) % period for k in run], period))]}
    sys_hz = st[chans[0]]["d"].phy["sys_hz"]
    beats = int(a.gib * (1 << 30)) // 64
    n, last_line, last_scan, since = 0, time.time(), time.time(), {ch: 0 for ch in chans}
    while time.time() - t0 < a.minutes * 60:
        mode = 0 if n % 2 == 0 else 2
        seed = (0x0123456789ABCDEF ^ (n * 0x9E3779B97F4A7C15)) & ((1 << 64) - 1)
        for ch in chans:
            bist_start(st[ch]["c"], beats, mode, seed)
        for ch in chans:
            bist_wait(st[ch]["c"], beats, mode, sys_hz)
        for ch in chans:
            bist_start(st[ch]["c"], beats, mode | 1, seed)
        for ch in chans:
            r = bist_wait(st[ch]["c"], beats, mode | 1, sys_hz)
            s_ = st[ch]
            s_["passes"] += 1
            s_["errors"] += r["errors"] + (beats - r["beats"])
            since[ch] += r["errors"]
            s_["lanes"] = [x + y for x, y in zip(s_["lanes"], r["lanes"])]
        n += 1
        if time.time() - last_line >= 60:
            print(f"{stamp()} {Temps.fmt(temps.read())}; BIST passes {n}, wrong beats in the last "
                  f"minute: " + ", ".join(f"ch{ch} {since[ch]}" for ch in chans))
            last_line, since = time.time(), {ch: 0 for ch in chans}
        if time.time() - last_scan >= a.every:
            for ch in chans:
                s_ = st[ch]
                table = dqs_scan(s_["d"], s_["dqs"], period, a.stride, csr=s_["c"], show=False)
                per, common, pick, run = margins(table, s_["d"].nm, period=period)
                off = offsets(run, period) if run else None
                t = (time.time() - t0) / 60
                s_["scans"].append((t, len(run) * a.stride, off))
                gtxt = ""
                if s_["groups"]:        # each group's own run (the scan moves both groups)
                    gw = group_windows(table, s_["groups"], period)
                    s_["group_scans"].append((t, {g: len(r) * a.stride for g, (_, _, _, r) in gw.items()}))
                    gtxt = "; groups " + ", ".join(
                        f"{g} {len(r) * a.stride} steps" + (
                            f" ({offsets(r, period)[0]:+d}..{offsets(r, period)[1]:+d})"
                            if r and offsets(r, period) else " (running phase outside)")
                        for g, (_, _, _, r) in gw.items())
                print(f"{stamp()} channel {ch} rescan from its step {s_['home']}, "
                      f"{Temps.fmt(temps.read())}: common window {len(run) * a.stride} steps "
                      f"({len(run) * a.stride * s_['dqs'].step_ps:.0f} ps), "
                      + (f"from {off[0]:+d} to {off[1]:+d} around the running phase" if off else
                         "the running phase is OUTSIDE it") + gtxt)
                for m, line in enumerate(pass_map(table, s_["d"].nm, period)):
                    print(f"    m{m}{' g%d' % s_['groups'][m] if s_['groups'] else ''} {line}")
                s_["dqs"].move(s_["home"])
                wl, rl, err = s_["d"].calibrate(verbose=False)
                s_["d"].hardware()
                if any(err) or min(wl) < 0:
                    print(f"    channel {ch}: recalibration at step {s_['home']}: FAIL")
            last_scan = time.time()
    ok = True
    print(f"{stamp()} {Temps.fmt(temps.read())}; done: {n} BIST passes of {a.gib:g} GiB per channel")
    for ch in chans:
        s_ = st[ch]
        step_ps = s_["dqs"].step_ps
        widths = [w for _, w, _ in s_["scans"]]
        inside = all(o is not None for _, _, o in s_["scans"])
        print(f"channel {ch}: {s_['passes']} passes, {s_['errors']} wrong beats (per lane "
              f"{s_['lanes']}); common window {min(widths) * step_ps:.0f} to "
              f"{max(widths) * step_ps:.0f} ps over {len(widths)} scans; the running phase "
              + ("stayed inside every window" if inside else "fell OUTSIDE a window"))
        for t, w, o in s_["scans"]:
            print(f"    {t:5.1f} min: {w:3d} steps ({w * step_ps:4.0f} ps)"
                  + (f", margins {o[0]:+d} / {o[1]:+d} steps" if o else ", running phase outside"))
        if s_["groups"] and s_["group_scans"]:
            for g in sorted(s_["group_scans"][0][1]):
                ws = [gs[g] for _, gs in s_["group_scans"]]
                print(f"    group {g}: {min(ws) * step_ps:.0f} to {max(ws) * step_ps:.0f} ps over "
                      f"{len(ws)} rescans")
        ok &= s_["errors"] == 0 and inside
    print("temperature run:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ------------------------------------------------------------------------------ forced write latency
def runs(steps, period):
    """The runs of consecutive steps in `steps` (a set of steps within one period; a run may
    wrap): [(first, last)], last past period - 1 for a run around the wrap."""
    if len(steps) == period:
        return [(0, period - 1)]
    out = []
    for k in sorted(steps):
        if (k - 1) % period in steps:
            continue                      # not a run's first step
        e = k
        while (e + 1) % period in steps:
            e += 1
        out.append((k, e))
    return out


def wl_scan(a, c, period):
    """The write DQS phase over one tCK with every lane's write latency forced to each of --wl
    (bitslips 0 and 6, the two the calibration picks between), so that each lane's window at a
    latency shows whole instead of cut where the calibration flips it. Per step: init, the
    forced bitslip, read leveling, a --mib MiB BIST write and read-back. A lane passes a step
    when its read window is >= MIN_WINDOW taps, its DFI check passes and the BIST has no wrong
    beat. Per lane and latency: the passing runs, and each run's edges (steps from its last
    passing step to the first with half the beats wrong, or no read window: how steep the error
    rate rises). The raw table goes to BUILD_DIR/wscan_ch<ch>.json."""
    import json
    d = Dram(c, a.build)
    dqs = DqsPhase(c, d.phy["vco_hz"])
    sys_hz = d.phy["sys_hz"]
    beats = a.mib * (1 << 20) // 64
    start = dqs.steps()
    wls = [int(x) for x in a.wl.split(",")]
    temps = Temps(c.csr)
    print(f"forced write latency scan, channel {c.ch}, from step {start}, every {a.stride} steps "
          f"({a.stride * dqs.step_ps:.1f} ps), {a.mib} MiB BIST per step, {Temps.fmt(temps.read())}")
    raw = {}
    for wl in wls:
        rows = {}
        for k in range(0, period, a.stride):
            dqs.move(start + k)
            d.init()
            d.strobe("ddrphy_wdly_dq_bitslip", d.all, wl)
            d.wb = [wl] * d.nm
            rl, err = d.read_leveling(verbose=False)
            d.hardware()
            seed = 0x0123456789ABCDEF + 7919 * k + wl
            bist(c, beats, 0, seed, sys_hz)
            bad = bist(c, beats, 1, seed, sys_hz)["lanes"]
            rows[k] = [[n, e, b] for (n, _, _), e, b in zip(rl, err, bad)]
            print(f"  wl {wl} +{k:3d} "
                  + "".join("#" if n >= MIN_WINDOW and not e and not b else "." for n, e, b in rows[k])
                  + "  windows " + " ".join(f"{n:2d}" for n, _, _ in rows[k])
                  + "  BIST wrong beats " + " ".join(str(b) for _, _, b in rows[k]), flush=True)
        raw[wl] = rows
    dqs.move(start)
    d.calibrate(verbose=False)
    d.hardware()

    def wrong(rows, k, m):                  # the fraction of lane m's beats wrong at step k
        r = rows.get(k % period)
        return None if r is None else 1.0 if r[m][0] == 0 else r[m][2] / beats

    def edge(rows, k, m, sign):
        for i in range(1, period):
            f = wrong(rows, k + sign * i, m)
            if f is not None and f >= 0.5:
                return i
        return None

    print(f"per lane and write latency: the passing runs in steps from {start} (width in steps / "
          f"ps; edges: steps from the run's end to half the beats wrong), {Temps.fmt(temps.read())}")
    summary = {}
    for wl in wls:
        rows = raw[wl]
        for m in range(d.nm):
            good = {k for k, r in rows.items() if r[m][0] >= MIN_WINDOW and not r[m][1] and not r[m][2]}
            rr = []
            for s, e in runs({k // a.stride for k in good}, period // a.stride):
                s, e = s * a.stride, e * a.stride
                rr.append({"first": s, "last": e, "width": e - s + a.stride,
                           "edge_lo": edge(rows, s, m, -1), "edge_hi": edge(rows, e, m, 1)})
            rr.sort(key=lambda r: -r["width"])
            summary.setdefault(m, {})[wl] = rr
            print(f"  m{m} wl {wl}: " + (", ".join(
                f"+{r['first']}..+{r['last']} ({r['width']} / {r['width'] * dqs.step_ps:.0f} ps; "
                f"edges {r['edge_lo']} / {r['edge_hi']})" for r in rr) or "none"))
    out = Path(getattr(a, "json_dir", None) or a.build) / f"wscan_ch{c.ch}.json"
    out.write_text(json.dumps({"channel": c.ch, "start": start, "stride": a.stride, "mib": a.mib,
                               "beats": beats, "step_ps": dqs.step_ps, "period": period,
                               "raw": raw, "summary": summary}))
    print(f"raw data: {out}")
    return 0


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("build", type=Path)
    ap.add_argument("what", choices=["info", "cal", "bist", "all", "selftest", "dqs", "rscan", "soak",
                                     "temp", "wscan", "g1"])
    ap.add_argument("--ch", help="0, 1 or both (default: every channel of the image)")
    ap.add_argument("--seconds", type=float, default=300, help="soak: how long to repeat the BIST")
    ap.add_argument("--minutes", type=float, default=35, help="temp: how long to run the BIST")
    ap.add_argument("--every", type=float, default=300, help="temp: seconds between window scans")
    ap.add_argument("--span", type=int, help="all: scan only this many steps (default one tCK)")
    ap.add_argument("--gib", type=float, default=2.0, help="BIST size (the channel: 2 GiB)")
    ap.add_argument("--stride", type=int, default=4, help="DQS scan stride, fine steps")
    ap.add_argument("--dev", default="/dev/xdma0_user")
    ap.add_argument("--steps", type=int, help="dqs: move the DQS phase to this step")
    ap.add_argument("--wl", default="0,6", help="wscan: the forced write latencies (bitslips)")
    ap.add_argument("--mib", type=int, default=64, help="wscan: BIST size per step, MiB")
    ap.add_argument("--eighths", type=int, help="g1: group 1's offset, 1/8 VCO periods (7 steps)")
    a = ap.parse_args()
    if a.what == "selftest":
        return selftest(a.build)
    csr = Bar0(a.build, a.dev)
    phy = Dram(csr, a.build).phy
    chans = phy.get("channels", [0])
    if a.ch:
        chans = [0, 1] if a.ch == "both" else [int(a.ch)]
    period = round(56 * phy["vco_hz"] / (4 * phy["sys_hz"]))     # fine steps per tCK
    if a.what in ("info", "all", "temp"):
        csr.w("ctrl_scratch", 0xA5C3_5A3C)
        s = csr.r("ctrl_scratch") ^ 0xA5C3_5A3C ^ 0x12345678      # 0x12345678 when it holds
        print(f"scratch {s:#x} ({'ok' if s == 0x12345678 else 'BAD'}), bus errors "
              f"{csr.r('ctrl_bus_errors')}, channels {phy.get('channels', [0])}, DQS phase steps "
              + ", ".join(f"ch{ch} {DqsPhase(Chan(csr, ch), phy['vco_hz']).steps()}" for ch in chans)
              + f" ({1e12 / phy['vco_hz'] / 56:.1f} ps each, {period} per tCK), sys "
              f"{phy['sys_hz'] / 1e6:.2f} MHz, CL {phy['cl']} CWL {phy['cwl']}"
              + (f", {Temps.fmt(Temps(csr).read())}" if "xadc_temperature" in csr.regs else ""))
        if s != 0x12345678:
            return 1
        if phy.get("phy") == "wl":
            for ch in chans:
                w = WriteClocks(Chan(csr, ch))
                p = w.check()
                print(f"ch{ch} write clocks: groups {phy['groups'][str(ch)]}, group 1 offset "
                      f"{w.group1()} x 1/8 VCO (CLKOUT3..6 static phases {p}), MMCM locked "
                      f"{Chan(csr, ch).r('wclk_mmcm_locked')}")
    if a.what == "temp":
        return temp_run(a, csr, chans, period)
    rc = 0
    for ch in chans:
        if len(chans) > 1:
            print(f"==== channel {ch}")
        rc |= channel(a, Chan(csr, ch), period)
    return rc


def channel(a, csr, period):
    if a.what == "wscan":
        return wl_scan(a, csr, period)
    d = Dram(csr, a.build)
    dqs = DqsPhase(csr, d.phy["vco_hz"])
    wl_phy = d.phy.get("phy") == "wl"
    if a.what == "g1":
        w = WriteClocks(csr)
        w.set_group1(a.eighths)
        d.ctl(0)                                  # the DRAM's reset: its clocks stopped
        time.sleep(0.001)
        print(f"group 1 offset {w.group1()} x 1/8 VCO, DQS phase {dqs.steps()} (after the MMCM reset)")
        a.what = "cal"
    if a.what == "dqs":
        dqs.move(a.steps)
        print("DQS phase", dqs.steps())
    if a.what == "all" and wl_phy:
        groups = d.phy["groups"][str(csr.ch)]
        print(f"write clock groups {groups}: scan of the common phase with group 1 on group 0, "
              f"every {a.stride} steps, a 64 MiB BIST per step:")
        try:
            res = calibrate_groups(d, dqs, WriteClocks(csr), groups, period, stride=a.stride,
                                   csr=csr)
        except CalError as e:
            print(f"  {e}: FAIL")
            return 1
        for name, t in (("group 1 at 0", res["scan0"]),
                        (f"group 1 at {res['offset_eighths']}/8", res["scan1"])):
            print(f"  lanes, {name} (phase steps from -{period // 2} to +{period // 2}):")
            for m, line in enumerate(pass_map(t, d.nm, period)):
                print(f"    m{m} g{groups[m]} {line}")
        run = res["run"]
        print(f"common: {len(run) * a.stride} steps ({len(run) * a.stride * dqs.step_ps:.0f} ps; "
              f"groups alone at offset 0: " + ", ".join(
                  f"{g} {w} steps ({w * dqs.step_ps:.0f} ps)" for g, w in res["group_runs"].items())
              + f"); DQS at +{res['pick']} steps")
    elif a.what == "all":
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
