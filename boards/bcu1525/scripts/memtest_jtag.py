#!/usr/bin/env python3
"""The BCU1525's DIMMs tested over JTAG on the memory-test image (IMAGE=mem; docs/bcu1525.md
section 14), under the thermal governor and the host's guard (opentpu/host/gov.py): the card has
no fan.

    memtest_jtag.py --serial <SERIAL> --out DIR [--bit otpu.bit] [--tty /dev/ttyUSB2]
                    [--dimms 0,1,2,3] [--gib 8] [--scan] [--hostcal | --repair]
                    [--write-eyes] [--warm C]

Cool: per DIMM, alone, from the die at or below RESUME: GO, its core's BIOS calibrates it (its
UART log through the governor's UART select), the read windows under traffic per lane (--scan:
ddrcal.bist_read_scan), the host's calibration when the BIOS's failed (--hostcal) or a repair of
the lanes it left without a read window (--repair: ddr4.repair_lanes), every lane's write delay
centred in its write eye (--write-eyes: ddr4.centre_writes), then the BIST
over --gib GiB (random, then address data; the wrong beats per byte lane); SAFE. Warm (--warm C):
every DIMM on (calibrated cool), BIST traffic on all until the die reaches C or stops rising, then
each DIMM's BIST and scan warm; then C1 (or --warm-dimms) restarted warm, so its BIOS calibrates it
warm, and tested again. Writes DIR/results.json, DIR/thermal.csv (the guard's), DIR/uart_*.log.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import select
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from opentpu.host import ddr4, ddrcal
from opentpu.host import gov as G
from opentpu.host.jtagaxi import JtagAxi, JtagAxiError

DATA = ddr4.DATA_UDIMM
HZ = ddr4.sys_hz(DATA)


class Tripped(RuntimeError):
    pass


class Uart:
    """A tty captured in a thread; mark() starts a new file and text()."""

    def __init__(self, dev: str | None):
        self.dev, self.f, self.buf, self.lock = dev, None, bytearray(), threading.Lock()
        if not dev:
            return
        subprocess.run(["stty", "-F", dev, "115200", "raw", "-echo", "-ixon"], check=True)
        self.fd = os.open(dev, os.O_RDONLY | os.O_NONBLOCK)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            r, _, _ = select.select([self.fd], [], [], 1.0)
            if not r:
                continue
            try:
                b = os.read(self.fd, 4096)
            except BlockingIOError:
                continue
            with self.lock:
                self.buf += b
                if self.f:
                    self.f.write(b)
                    self.f.flush()

    def mark(self, path: Path):
        with self.lock:
            if self.f:
                self.f.close()
            self.f = open(path, "wb") if self.dev else None
            self.buf = bytearray()

    def text(self) -> str:
        with self.lock:
            return self.buf.decode("ascii", "replace")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def check(g: G.Gov, guard: G.Guard):
    if guard.tripped or guard.booted:
        raise Tripped(f"guard: {guard.tripped}")
    s = g.status()
    if not s["run"]:
        raise Tripped(f"governor SAFE: {s['causes']} at {s['temp_c']:.1f} C")
    return s


def wait_cool(g: G.Gov, guard: G.Guard, limit_s: float = 7200):
    t0 = time.time()
    while True:
        s = g.status()
        reg = (f", regulator {s['reg_c']:.2f} C ({s['reg_age_ms']} ms old)" if s["has_reg"]
               else "")
        if s["cool"] and not s["alarm"] and not s["ot"] and s["temp_valid"] and s["reg_cool"]:
            log(f"cool: {s['temp_c']:.1f} C{reg}")
            return s["temp_c"]
        if time.time() - t0 > limit_s:
            raise Tripped(f"not cool after {limit_s} s ({s['temp_c']:.1f} C{reg})")
        if int(time.time() - t0) % 60 < 5:
            res = (f", REG_RESUME {G.quarter_c(g.rd(G.REG_RESUME)):.2f}" if s["has_reg"] else "")
            log(f"cooling: {s['temp_c']:.1f} C{reg} (RESUME {G.celsius(g.rd(G.RESUME)):.1f}{res})")
        time.sleep(5)


def go_cool(g: G.Gov, guard: G.Guard, dimms, limit_s: float = 7200):
    """GO once the die is at or below RESUME: the idle die sits about there (54.5 - 55.5 C on
    2026-10-26 with every DIMM off), so a GO a moment after a cool reading can find it a code
    above; the governor refuses that, and this retries."""
    t0 = time.time()
    while True:
        wait_cool(g, guard, limit_s=max(60.0, limit_s - (time.time() - t0)))
        if g.go(dimms=dimms):
            return
        if time.time() - t0 > limit_s:
            raise Tripped(f"GO refused for {limit_s} s: {g.status()}")
        time.sleep(1)


def csr_up(t, d: int, timeout: float = 10.0):
    """DIMM d's CSRs once its core answers (otpu_axil_off answers SLVERR before)."""
    c = ddr4.csr(t, d, DATA)
    t0 = time.time()
    while True:
        try:
            c.r("ctrl_scratch")
            return c
        except JtagAxiError:
            if time.time() - t0 > timeout:
                raise
            time.sleep(0.1)


def wait_bios(c, uart: Uart, timeout: float = 90.0) -> dict:
    """The core's BIOS done. LiteX's BIOS hands DFII to hardware control after its leveling and
    before its memtest, also when the leveling failed, so with a UART its memtest's verdict decides
    (bios_ok); without one, DFII in hardware control for 20 s."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        hw = bool(c.r("sdram_dfii_control") & ddr4.DFII_SEL)
        txt = uart.text()
        if "Memtest OK" in txt or "Memtest KO" in txt or "Memory initialization failed" in txt:
            time.sleep(1)
            break
        if hw and not uart.dev and time.time() - t0 > 20:
            break
        time.sleep(0.5)
    hw = bool(c.r("sdram_dfii_control") & ddr4.DFII_SEL)
    txt = uart.text()
    memtest = ("OK" if "Memtest OK" in txt else
               "KO" if "Memtest KO" in txt or "Memory initialization failed" in txt else None)
    return {"bios_hw": hw, "bios_ok": hw and (memtest == "OK" if uart.dev else True),
            "bios_s": round(time.time() - t0, 1), "bios_memtest": memtest}


def bist(c, gib: float, passes: int = 1) -> list[dict]:
    """ddr4.bist's passes on csr c (the full image's SCRATCH marker has no place here)."""
    beats = int(gib * (1 << 30)) // 64
    out = []
    for p in range(passes):
        for dmode, name in ((0, "random"), (2, "address")):
            seed = (0x0123456789ABCDEF ^ (p * 0x9E3779B97F4A7C15)) & ((1 << 64) - 1)
            w = ddrcal.bist(c, beats, dmode, seed, HZ, timeout=120)
            r = ddrcal.bist(c, beats, dmode | 1, seed, HZ, timeout=120)
            out.append({"pass": p, "data": name, "beats": beats, "write_gbs": round(w["gbs"], 2),
                        "read_gbs": round(r["gbs"], 2), "errors": r["errors"],
                        "lanes": r["lanes"], "bits": r["bits"],
                        "ok": r["errors"] == 0 and r["beats"] == beats})
    return out


# the BIOS's read leveling choice per module: "best: m0, b07 delays: 353+-76" (bitslip, centre tap)
BIOS_READ = re.compile(r"best: m(\d+), b(\d+) delays: (\d+)\+-(\d+)")


def bios_reads(txt: str) -> dict:
    """{module: (bitslip, tap)} from the BIOS's log (a module with no window has none)."""
    return {int(m): (int(b), int(t)) for m, b, t, _ in BIOS_READ.findall(txt)}


# its write latency: "Write latency calibration:" then "m0:6 m1:6 ..." (each lane's write bitslip)
BIOS_WLAT = re.compile(r"Write latency calibration:\s*((?:m\d+:-?\d+\s+)+)")


def bios_wlat(txt: str) -> list[int] | None:
    """Each module's write bitslip from the BIOS's log (0 for one it found none for)."""
    m = BIOS_WLAT.search(txt)
    if not m:
        return None
    w = {int(k): int(v) for k, v in re.findall(r"m(\d+):(-?\d+)", m.group(1))}
    return [max(w.get(k, 0), 0) for k in range(max(w) + 1)]


def scan(c, reads: dict) -> list[int]:
    """Each lane's read window under traffic (taps), at its calibrated bitslip; every lane left
    at its window's centre, or at its calibrated tap (`reads`: {module: (bitslip, tap)}) with
    none. EN_VTC off while the delays move (UltraScale's delay lines take no steps with it on:
    the BIOS's and calibrate_dimm's leveling do the same), on after."""
    d = ddr4.UspDram(c, DATA)
    for m, (b, tap) in reads.items():
        d.rb[m], d.rd[m] = b, tap
    c.w("ddrphy_en_vtc", 0)
    try:
        table = ddrcal.bist_read_scan(c, d, HZ, mib=64)
    finally:
        c.w("ddrphy_en_vtc", 1)
    return [ddrcal.Dram.windows([row[m] for row in table])[1] for m in range(d.nm)]


def test_dimm(t, g, guard, uart, d: int, label: str, a, out: Path) -> dict:
    """DIMM d, already turned on: its BIOS, then scan / host calibration / BIST."""
    r = {"dimm": d, "phase": label}
    c = csr_up(t, d)
    r.update(wait_bios(c, uart, timeout=a.bios_s))
    r["die_c_cal"] = round(check(g, guard)["temp_c"], 1)
    log(f"C{d} {label}: BIOS {'passed' if r['bios_ok'] else 'FAILED'} in {r['bios_s']} s, "
        f"memtest {r['bios_memtest']}, DFII {'hardware' if r['bios_hw'] else 'software'}, "
        f"die {r['die_c_cal']} C")
    if not r["bios_ok"] and a.hostcal:
        hc = ddr4.calibrate_dimm(c, DATA, log=lambda s: log(f"C{d} hostcal{s}"))
        r["hostcal"] = {k: hc[k] for k in ("ok", "wdelay", "write_latency", "read", "errors")}
        check(g, guard)
    elif r["bios_hw"] and a.repair:
        txt = uart.text()
        bad = [m for m in range(8) if m not in bios_reads(txt)]
        wl = bios_wlat(txt)
        if bad and wl:
            log(f"C{d} {label}: repairing lanes {bad} (BIOS write bitslips {wl})")
            r["repair"] = ddr4.repair_lanes(c, bad, wl, DATA, log=lambda s: log(f"C{d} repair{s}"))
            check(g, guard)
        elif bad:
            log(f"C{d} {label}: lanes {bad} have no window, and the BIOS's log no write latency")
    if r["bios_hw"] and a.write_eyes and all(x["ok"] for x in r.get("repair", {}).values()):
        r["write_eyes"] = ddr4.centre_writes(c, data=DATA, log=lambda s: log(f"C{d} writes{s}"))
        check(g, guard)
    # the BIST also after a failed BIOS memtest (each lane's errors), not on a failed host
    # calibration (it leaves DFII in software control)
    if r["hostcal"]["ok"] if "hostcal" in r else r["bios_hw"]:
        if a.scan:
            r["reads"] = ({m: (x["bitslip"], x["tap"]) for m, x in enumerate(r["hostcal"]["read"])}
                          if "hostcal" in r else bios_reads(uart.text()))
            r["reads"].update({m: (x["bitslip"], x["tap"]) for m, x in r.get("repair", {}).items()
                               if x["ok"]})
            r["windows"] = scan(c, r["reads"])
            log(f"C{d} {label}: read windows under traffic {r['windows']} taps")
            check(g, guard)
        r["bist"] = bist(c, a.gib, a.passes)
        s = check(g, guard)
        r["die_c_bist"] = round(s["temp_c"], 1)
        for b in r["bist"]:
            log(f"C{d} {label}: BIST {b['data']} {a.gib} GiB: write {b['write_gbs']} GB/s, read "
                f"{b['read_gbs']} GB/s, errors {b['errors']} per lane {b['lanes']} "
                f"({'ok' if b['ok'] else 'FAIL'})")
        r["ok"] = all(b["ok"] for b in r["bist"])
    else:
        r["ok"] = False
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--serial", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--bit")
    ap.add_argument("--tty")
    ap.add_argument("--dimms", default="0,1,2,3")
    ap.add_argument("--gib", type=float, default=8.0)
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--hostcal", action="store_true")
    ap.add_argument("--repair", action="store_true",
                    help="after the BIOS, repair the lanes it left without a read window "
                         "(ddr4.repair_lanes: the lane's write delay / write bitslip)")
    ap.add_argument("--write-eyes", action="store_true",
                    help="after the BIOS (and --repair), every lane's write eye and its write "
                         "delay at the centre (ddr4.centre_writes)")
    ap.add_argument("--warm", type=float, help="the die's target for the warm phase (C, <= 76)")
    ap.add_argument("--warm-dimms", default="1")
    ap.add_argument("--warm-max-s", type=float, default=2400)
    ap.add_argument("--no-bmc", action="store_true")
    ap.add_argument("--resume", type=float,
                    help="the governor's RESUME (C, <= 70: the fabric clamps it); the idle die "
                         "sits near 55-58 C on a card without a fan, cold, and 66 C after hours of tests")
    ap.add_argument("--bios-s", type=float, default=240.0,
                    help="how long a BIOS may calibrate (a failing lane's read leveling is slow)")
    a = ap.parse_args(argv)

    def interrupted(sig, frame):
        raise KeyboardInterrupt(f"signal {sig}")
    # a background job starts with SIGINT ignored: both signals end the run through its finally
    # (the guard's SAFE, the transport closed)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    if a.warm is not None and a.warm > 76:
        ap.error("--warm above 76 C: the governor trips at 80, the guard at 81")
    a.out.mkdir(parents=True, exist_ok=True)
    dimms = [int(x) for x in a.dimms.split(",")]
    res = {"serial": a.serial, "bit": a.bit, "start": time.strftime("%F %T"), "tests": []}

    def save():
        (a.out / "results.json").write_text(json.dumps(res, indent=1))

    t = JtagAxi(a.serial, log=a.out / "vivado.log")
    log(f"JTAG: {t.target}")
    if a.bit:
        n = t.program(a.bit)
        log(f"programmed {a.bit}: {n} JTAG-to-AXI core(s), DONE {t.done()}")
        if n < 1:
            raise SystemExit("no JTAG-to-AXI core after programming")
    g = G.Gov(t)
    if not g.present():
        raise SystemExit(f"no governor at {G.R_GOV:#x} (ID {g.rd(G.ID):#x})")
    if a.resume is not None:
        g.set_limits(resume_c=a.resume)
    res["limits"] = g.limits()
    log(f"governor: {res['limits']}, {g.status()}")
    guard = G.Guard(t, csv=a.out / "thermal.csv", use_bmc=not a.no_bmc)
    if not a.no_bmc:
        guard.read_first()
        log(f"BMC: VCCINT regulator {guard.reg}")
    guard.start()
    uart = Uart(a.tty)
    rc = 1
    try:
        for d in dimms:
            g.uart(d)
            uart.mark(a.out / f"uart_C{d}_cool.log")
            go_cool(g, guard, (d,))
            res.setdefault("die_c_start", {})[d] = round(g.status()["temp_c"], 1)
            r = test_dimm(t, g, guard, uart, d, "cool", a, a.out)
            r["uart"] = uart.text()[-4000:]
            res["tests"].append(r)
            g.safe()
            save()
        if a.warm is not None:
            wd = [int(x) for x in a.warm_dimms.split(",")]
            g.uart(wd[0])
            uart.mark(a.out / f"uart_C{wd[0]}_warm_start.log")
            go_cool(g, guard, dimms)
            cs = {d: csr_up(t, d) for d in dimms}
            for d in dimms:
                wait_bios(cs[d], uart, timeout=a.bios_s)
            t0, hist = time.time(), []
            while True:                         # heat: BIST traffic on every DIMM
                for d in dimms:
                    if cs[d].r("sdram_dfii_control") & ddr4.DFII_SEL:
                        ddrcal.bist(cs[d], int(a.gib * (1 << 30)) // 64, 0, 7 + d, HZ, timeout=120)
                s = check(g, guard)
                hist.append((time.time(), s["temp_c"]))
                old = [x for tt, x in hist if tt < time.time() - 300]
                log(f"heating: {s['temp_c']:.1f} C")
                if s["temp_c"] >= a.warm or time.time() - t0 > a.warm_max_s or \
                        (old and s["temp_c"] - old[-1] < 0.5):
                    break
            res["warm_reached_c"] = round(s["temp_c"], 1)
            for d in dimms:                     # calibrated cool, tested warm
                uart.mark(a.out / f"uart_C{d}_coolcal_warm.log")
                r = {"dimm": d, "phase": "cool-cal warm-test", "die_c": round(check(g, guard)["temp_c"], 1)}
                if cs[d].r("sdram_dfii_control") & ddr4.DFII_SEL:
                    if a.scan:
                        # (the warm start's BIOS logs are not all on the UART: the cool test's
                        # choices are the taps a lane with no window goes back to)
                        cool = [x for x in res["tests"] if x["dimm"] == d and x["phase"] == "cool"]
                        r["windows"] = scan(cs[d], cool[-1].get("reads", {}) if cool else {})
                        log(f"C{d} cool-cal warm: windows {r['windows']}")
                    r["bist"] = bist(cs[d], a.gib, a.passes)
                    r["ok"] = all(b["ok"] for b in r["bist"])
                    log(f"C{d} cool-cal warm: BIST {[(b['errors'], b['lanes']) for b in r['bist']]}")
                res["tests"].append(r)
                save()
            for d in wd:                        # calibrated warm
                g.uart(d)
                g.set_dimms([x for x in dimms if x != d])
                time.sleep(1)
                uart.mark(a.out / f"uart_C{d}_warm.log")
                g.set_dimms(dimms)
                r = test_dimm(t, g, guard, uart, d, "warm", a, a.out)
                r["uart"] = uart.text()[-4000:]
                res["tests"].append(r)
                save()
            g.safe()
        rc = 0 if all(r.get("ok") for r in res["tests"]) else 1
    except (Tripped, JtagAxiError, TimeoutError, KeyboardInterrupt) as e:
        log(f"STOPPED: {e}")
        res["stopped"] = str(e)
    finally:
        guard.stop()
        res["guard"] = {"tripped": guard.tripped, "booted": guard.booted, "errors": guard.errors[-20:]}
        res["end"] = time.strftime("%F %T")
        save()
        t.close()
    log(f"done: {a.out / 'results.json'} rc {rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
