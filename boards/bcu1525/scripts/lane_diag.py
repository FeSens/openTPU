#!/usr/bin/env python3
"""One byte lane of a BCU1525 DIMM left without a read window by its BIOS, on the memory-test
image over JTAG (IMAGE=mem; docs/bcu1525.md section 14), under the governor and the host's guard:
the lane's BIST errors over every read bitslip and a grid of IDELAY taps, at the BIOS's settings
and then with one change at a time on the lane's write side: its write bitslip, its DQS / DQ write
delay, the data mask off (MR5 A10, every lane), VrefDQ (MR6, every DRAM). A change is undone
before the next; the other lanes keep the BIOS's settings throughout. A lane whose reads come
right at some bitslip and tap only after a write-side change has a write fault; one that reads
wrong everywhere whatever the writes do, a read or DRAM fault.

    lane_diag.py --serial <SERIAL> --out DIR --dimm 1 --lane 6 [--bit otpu.bit]
                 [--tty /dev/ttyUSB2] [--resume 64.5] [--step 32] [--mib 16]

Writes DIR/lane_diag.json, DIR/thermal.csv (the guard's), DIR/uart_C<d>.log.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from opentpu.host import ddr4, ddrcal                            # noqa: E402
from opentpu.host import gov as G                                # noqa: E402
from opentpu.host.jtagaxi import JtagAxi, JtagAxiError           # noqa: E402
import memtest_jtag as M                                         # noqa: E402

log = M.log
SEED = 0x0123456789ABCDEF
MR5_DM = 1 << 10                # MR5 A10: the data mask on


def grid(c, d: ddr4.UspDram, lane: int, step: int, mib: int) -> dict:
    """The region written once (BIST random data, every lane at its write settings now), then
    read back at each read bitslip and every step-th tap of the lane: its wrong beats, and the
    wrong bits at the fewest. The lane's taps move by increments from a reset per bitslip."""
    beats = mib * (1 << 20) // 64
    ddrcal.bist(c, beats, 0, SEED, M.HZ)
    rows, best = {}, None
    for b in range(d.phy["bitslips"]):
        d.set_rbitslip(lane, b)
        d.strobe("ddrphy_rdly_dq_rst", 1 << lane)
        row = []
        for tap in range(0, d.phy["delays"], step):
            r = ddrcal.bist(c, beats, 1, SEED, M.HZ)
            e, bits = r["lanes"][lane], r["bits"] >> (8 * lane) & 0xFF
            row.append(e)
            if best is None or e < best["errors"]:
                best = {"bitslip": b, "tap": tap, "errors": e, "bits": f"{bits:08b}"}
            d.strobe("ddrphy_rdly_dq_inc", 1 << lane, step)
        rows[b] = row
    clean = sum(e == 0 for row in rows.values() for e in row)
    return {"beats": beats, "rows": rows, "best": best, "clean": clean,
            "map": {b: "".join("1" if e == 0 else "x" if e <= beats // 100 else "0" for e in row)
                    for b, row in rows.items()}}


def mrs(d: ddr4.UspDram, ba: int, value: int):
    """A mode register write in software control (all banks precharged), then the PHY back to
    the controller."""
    d.ctl(d.CTL_CKE | d.CTL_ODT | d.CTL_RESET_N)
    d.cmd(0, d.RAS | d.WE | d.CS, 0x400)
    d.cmd(0, d.RAS | d.CAS | d.WE | d.CS, value, ba)
    time.sleep(0.001)
    d.hardware()


def vrefdq(d: ddr4.UspDram, value: int):
    d.ctl(d.CTL_CKE | d.CTL_ODT | d.CTL_RESET_N)
    d.set_vrefdq(value)
    d.hardware()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--serial", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--bit")
    ap.add_argument("--tty")
    ap.add_argument("--dimm", type=int, required=True)
    ap.add_argument("--lane", type=int, required=True)
    ap.add_argument("--resume", type=float, help="the governor's RESUME (C, <= 70)")
    ap.add_argument("--step", type=int, default=32, help="taps between grid points (the changes)")
    ap.add_argument("--mib", type=int, default=16, help="the region each grid point reads")
    ap.add_argument("--bios-s", type=float, default=240.0)
    a = ap.parse_args(argv)

    def interrupted(sig, frame):
        raise KeyboardInterrupt(f"signal {sig}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    a.out.mkdir(parents=True, exist_ok=True)
    dm, ln = a.dimm, a.lane
    res = {"serial": a.serial, "dimm": dm, "lane": ln, "start": time.strftime("%F %T"), "runs": []}

    def save():
        (a.out / "lane_diag.json").write_text(json.dumps(res, indent=1))

    t = JtagAxi(a.serial, log=a.out / "vivado.log")
    log(f"JTAG: {t.target}")
    if a.bit:
        n = t.program(a.bit)
        log(f"programmed {a.bit}: {n} JTAG-to-AXI core(s), DONE {t.done()}")
    g = G.Gov(t)
    if not g.present():
        raise SystemExit(f"no governor at {G.R_GOV:#x} (ID {g.rd(G.ID):#x})")
    if a.resume is not None:
        g.set_limits(resume_c=a.resume)
    log(f"governor: {g.limits()}, {g.status()}")
    guard = G.Guard(t, csv=a.out / "thermal.csv")
    guard.read_first()
    guard.start()
    uart = M.Uart(a.tty)
    rc = 1
    c = None
    try:
        g.uart(dm)
        uart.mark(a.out / f"uart_C{dm}.log")
        M.go_cool(g, guard, (dm,))
        c = M.csr_up(t, dm)
        r = M.wait_bios(c, uart, timeout=a.bios_s)
        res["bios"] = r
        log(f"C{dm}: BIOS {'passed' if r['bios_ok'] else 'FAILED'} in {r['bios_s']} s, "
            f"memtest {r['bios_memtest']}")
        if not r["bios_hw"]:
            raise SystemExit("the BIOS left the DIMM in software control")
        d = ddr4.UspDram(c, M.DATA)
        reads = M.bios_reads(uart.text())
        res["bios_reads"] = reads
        d.wd = [d.dqs_count(m) for m in range(d.nm)]
        wd0 = d.wd[ln]
        tck = d.tck_taps()
        res["wdelay0"], res["tck_taps"] = wd0, tck
        log(f"C{dm} m{ln}: the BIOS's DQS count {wd0}, tCK {tck} taps; reads {reads}")
        c.w("ddrphy_en_vtc", 0)

        def run(name, step):
            M.check(g, guard)
            x = grid(c, d, ln, step, a.mib)
            x["name"] = name
            res["runs"].append(x)
            save()
            log(f"{name}: {x['clean']} clean points, best {x['best']}")
            for b, line in x["map"].items():
                log(f"    b{b} |{line}|")
            return x

        run("bios", 8)
        # the write bitslip: the BIOS's write latency leaves no record here, so each value is set
        # outright, and the lane's BIOS value is back from the BIOS's log afterwards
        for wb in range(d.phy["bitslips"]):
            d.set_wbitslip(ln, wb)
            run(f"write bitslip {wb}", a.step)
        wl = [int(x.split(":")[1]) for x in uart.text().split("Write latency calibration:")[1]
              .split("\n")[1].split()] if "Write latency calibration:" in uart.text() else None
        res["bios_write_latency"] = wl
        if wl:
            d.set_wbitslip(ln, wl[ln])
        for k in (-2, -1, 1, 2):                    # the DQS (and DQ) write delay, eighths of tCK
            n = wd0 + k * tck // 8
            if 0 <= n < ddr4.TAPS - tck // 4:
                d.set_wdelay(ln, n)
                run(f"write delay {n} ({k:+d}/8 tCK)", a.step)
        d.set_wdelay(ln, wd0)
        mr5 = d.mode(5)
        mrs(d, 5, mr5 & ~MR5_DM)
        run("data mask off", a.step)
        mrs(d, 5, mr5)
        for pct in (64.0, 68.0, 72.0, 76.0):
            v = ddr4.vref_value(pct)
            vrefdq(d, v)
            run(f"VrefDQ {60 + 0.65 * v:.2f} %", a.step)
        vrefdq(d, d.mode(6) & 0x3F)
        run("bios again", a.step)
        rc = 0
    except (M.Tripped, JtagAxiError, TimeoutError, KeyboardInterrupt) as e:
        log(f"STOPPED: {e}")
        res["stopped"] = str(e)
    finally:
        try:
            if c is not None:
                c.w("ddrphy_en_vtc", 1)
            g.safe()
        except OSError:
            pass
        guard.stop()
        res["guard"] = {"tripped": guard.tripped, "booted": guard.booted, "errors": guard.errors[-20:]}
        res["end"] = time.strftime("%F %T")
        save()
        t.close()
    log(f"done: {a.out / 'lane_diag.json'} rc {rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
