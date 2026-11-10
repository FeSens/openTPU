#!/usr/bin/env python3
"""The full BCU1525 image (the openTPU top under the thermal governor: bd.tcl, otpu_fpga_top_bcu)
over JTAG (docs/bcu1525.md section 14): program it (or, with no --bit, the image loaded before a
warm reboot, its PCIe link enumerated: --xdma); the governor's
SAFE state as configured (no DIMM, the accelerator off, its control registers fenced); a forced
thermal trip and a heartbeat trip; each DIMM turned on in turn and calibrated by its core's BIOS
(its UART log; the lanes it leaves without a read window repaired, ddr4.repair_lanes), a short
BIST on it (--parallel: every DIMM at once, its BIOSes together, the UART on the last: the four
in 80 s instead of 5-6 min, the card with no fan); the accelerator on (the ID register); the
AXI4 JTAG master's burst order; then otpu-selftest's stages through jtagaxi.JtagTransport (the
target OTPU_TARGET=bcu1525), or through the PCIe link with --xdma; SAFE. The host's guard
(gov.Guard) kicks the governor and watches the die and the BMC throughout.

    PYTHONPATH=. python3 boards/bcu1525/scripts/fullimage_jtag.py --serial <SERIAL> \\
        --bit otpu.bit --out DIR [--tty /dev/ttyUSB2] [--dimms 0,1,2,3] [--gib 0.25]
        [--passes 1] [--write-eyes] [--parallel] [--kernel | --no-selftest] [--resume 60]
        [--xdma /dev/xdma0 [--selftest-args "--model qwen3"]] [--hold MIN]

Writes DIR/fullimage.json, DIR/thermal*.csv (the guard's), DIR/uart_C<d>.log.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from opentpu.host import ddr4                                    # noqa: E402
from opentpu.host import gov as G                                # noqa: E402
from opentpu.host import regs as R                               # noqa: E402
from opentpu.host.jtagaxi import JtagAxi, JtagAxiError, JtagTransport   # noqa: E402
import memtest_jtag as M                                         # noqa: E402

FENCED = 0xDEAD0FF0             # otpu_axil_off's read data (OKAY) while its slave is off
log = M.log


def wait_run(g: G.Gov, want: bool, limit: float) -> dict:
    t0 = time.time()
    while time.time() - t0 < limit:
        s = g.status()
        if s["run"] == want:
            s["after_s"] = round(time.time() - t0, 2)
            return s
        time.sleep(0.1)
    raise SystemExit(f"governor RUN not {want} after {limit} s: {g.status()}")


def csr_up(t, d: int, timeout: float = 10.0):
    """DIMM d's CSRs once its core answers: the full image's gate answers OKAY with FENCED
    before (XDMA would make an error response a completer abort), not SLVERR as image M's."""
    c = ddr4.csr(t, d, M.DATA)
    t0 = time.time()
    while c.r("ctrl_scratch") == FENCED:
        if time.time() - t0 > timeout:
            raise SystemExit(f"DIMM C{d}'s CSRs still fenced after {timeout} s")
        time.sleep(0.1)
    return c


def burst_order(t: JtagAxi, addr: int) -> dict:
    """The AXI4 master's DATA order: four words written as one burst, read back one word a burst;
    the other order when they come back reversed."""
    want = [0x11110000 + k for k in range(4)]
    data = b"".join(w.to_bytes(4, "little") for w in want)
    for lsb_right in (True, False):
        t.set_order(lsb_right)
        t.mem_write(addr, data)
        got = [int.from_bytes(t.mem_read(addr + 4 * k, 4), "little") for k in range(4)]
        if got == want and t.mem_read(addr, 16) == data:
            return {"lsb_right": lsb_right, "ok": True}
    return {"ok": False, "got": [hex(x) for x in got]}


def run_dimm(a, g: G.Gov, guard: G.Guard, d: int, c, r: dict, txt: str) -> dict:
    """DIMM d after its BIOS (r, wait_bios's; txt its UART log, "" without one): the lanes the BIOS
    left without a read window repaired, the write eyes (--write-eyes), the BIST, which decides
    (a BIOS memtest KO with its lanes repaired passes here)."""
    r["die_c"] = round(M.check(g, guard)["temp_c"], 1)
    log(f"C{d}: BIOS {'passed' if r['bios_ok'] else 'FAILED'} in {r['bios_s']} s, "
        f"memtest {r['bios_memtest']}, die {r['die_c']} C")
    bad, wl = [m for m in range(8) if m not in M.bios_reads(txt)], M.bios_wlat(txt)
    if r["bios_hw"] and not r["bios_ok"] and bad and wl:
        log(f"C{d}: repairing lanes {bad} (BIOS write bitslips {wl})")
        r["repair"] = ddr4.repair_lanes(c, bad, wl, M.DATA, log=lambda s: log(f"C{d} repair{s}"))
        M.check(g, guard)
    if r["bios_hw"] and a.write_eyes and all(x["ok"] for x in r.get("repair", {}).values()):
        r["write_eyes"] = ddr4.centre_writes(c, data=M.DATA, log=lambda s: log(f"C{d} writes{s}"))
        M.check(g, guard)
    r["ok"] = False
    if r["bios_hw"]:
        r["bist"] = M.bist(c, a.gib, a.passes)
        r["ok"] = all(b["ok"] for b in r["bist"])
        for b in r["bist"]:
            log(f"C{d}: BIST {b['data']} {a.gib} GiB: {b['write_gbs']} / {b['read_gbs']} "
                f"GB/s, errors {b['errors']} {b['lanes']}")
    r["uart"] = txt[-3000:]
    return r


def kernel_check(t: JtagAxi, guard: G.Guard, g: G.Gov) -> dict:
    """otpu-selftest's kernel stage alone (--kernel): the demo program on every unit, then the
    masked one (partial writes), each against the ISA simulator (checks.run_demo: 7.7 MB over
    JTAG each, the DRAM image written and read back), the guard checked after each."""
    from opentpu.host.board import CH_BYTES, Board, device_config
    from opentpu.host.checks import masked_program, run_demo
    board = Board(JtagTransport(t), check=False)
    cfg = device_config(board.info(), DRAM_BYTES=2 * CH_BYTES)
    out = {}
    for name, prog in (("all units", None), ("partial writes", masked_program())):
        t0 = time.time()
        ok, msg, st = run_demo(board, cfg, prog)
        M.check(g, guard)
        out[name] = {"ok": ok, "msg": msg, "s": round(time.time() - t0, 1),
                     "cycles": st.get("cycles"), "instructions": st.get("instructions")}
        log(f"kernel, {name}: {'PASS' if ok else 'FAIL'} {msg} ({out[name]['s']} s)")
        if not ok:
            break
    out["ok"] = all(v["ok"] for v in out.values())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--serial", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--bit")
    ap.add_argument("--tty")
    ap.add_argument("--dimms", default="0,1,2,3")
    ap.add_argument("--gib", type=float, default=0.25, help="each DIMM's BIST (GiB)")
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--write-eyes", action="store_true",
                    help="after the BIOS (and the repair), every lane's write delay at the centre of "
                         "its write eye (ddr4.centre_writes)")
    ap.add_argument("--no-selftest", action="store_true")
    ap.add_argument("--kernel", action="store_true",
                    help="the selftest's kernel stage alone (each check 7.7 MB over JTAG)")
    ap.add_argument("--parallel", action="store_true",
                    help="every DIMM's BIOS at once (the UART on the last; no lane repair on the "
                         "others)")
    ap.add_argument("--resume", type=float, help="the governor's RESUME (C, <= 70)")
    ap.add_argument("--bios-s", type=float, default=240.0)
    ap.add_argument("--xdma", metavar="DEV",
                    help="otpu-selftest through the PCIe link (the image enumerated: loaded over JTAG "
                         "before a warm reboot) instead of JTAG")
    ap.add_argument("--selftest-args", default="", help="more otpu-selftest arguments (--xdma)")
    ap.add_argument("--hold", type=float, default=0.0, metavar="MIN",
                    help="after the selftest (or --no-selftest), RUN held under the guard for MIN "
                         "minutes for other processes on the PCIe link; SIGTERM ends it")
    a = ap.parse_args(argv)

    def interrupted(sig, frame):
        raise KeyboardInterrupt(f"signal {sig}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    a.out.mkdir(parents=True, exist_ok=True)
    dimms = [int(x) for x in a.dimms.split(",")]
    res = {"serial": a.serial, "bit": a.bit, "start": time.strftime("%F %T")}

    def save():
        (a.out / "fullimage.json").write_text(json.dumps(res, indent=1, default=str))

    t = JtagAxi(a.serial, log=a.out / "vivado.log")
    guard = None
    rc = 1
    try:
        log(f"JTAG {t.target}; before: {t.mon()}")
        if a.bit:
            n = t.program(a.bit)
            log(f"programmed {a.bit}: {n} JTAG-to-AXI core(s), DONE {t.done()}")
        res["axis"] = t.axis()
        res["mon_after_prog"] = t.mon()
        log(f"JTAG-to-AXI: {res['axis']}")
        g = G.Gov(t)
        if not g.present():
            raise SystemExit(f"no governor at {G.R_GOV:#x} (ID {g.rd(G.ID):#x})")
        if a.resume is not None:
            g.set_limits(resume_c=a.resume)
        res["limits"], res["status0"] = g.limits(), g.status()
        log(f"governor {res['limits']}\n   {res['status0']}")
        # 1. SAFE as configured: nothing on, the control registers and every DIMM's CSRs fenced
        s0 = res["status0"]
        fenced = {"id": t.reg_read(R.R_ID),
                  "csr": [ddr4.csr(t, d, M.DATA).r("ctrl_scratch") for d in range(4)]}
        res["safe0"] = {"ok": not s0["run"] and not s0["dimms_on"] and not s0["acc_on"]
                        and fenced["id"] == FENCED and all(v == FENCED for v in fenced["csr"]),
                        "id": hex(fenced["id"]), "csr": [hex(v) for v in fenced["csr"]]}
        log(f"SAFE after configuration: {res['safe0']}")
        # the die from the governor and the SYSMON's JTAG port: the JTAG-to-AXI master's reads
        # of the wizard's DRP registers would be port B's, which can stop its temperature bus
        # (docs/bcu1525_sysmon.md)
        res["die"] = G.die_readings(g, t.mon)
        log(f"die {res['die']}")
        save()
        if not res["safe0"]["ok"]:
            raise SystemExit("not SAFE after configuration")

        guard = G.Guard(t, csv=a.out / "thermal.csv")
        guard.read_first()
        res["bmc"] = guard.reg
        log(f"BMC {guard.reg}")
        guard.start()
        # 2. a thermal trip (TRIP under the die) and a heartbeat trip (no kicks), no DIMM on
        M.go_cool(g, guard, (), limit_s=1800)
        g.set_limits(trip_c=g.status()["temp_c"] - 3)
        s = wait_run(g, False, 5)
        res["trip_thermal"] = s
        log(f"thermal trip: RUN off after {s['after_s']} s, causes {s['causes']}")
        g.set_limits(trip_c=80.0)
        g.clear()
        g.set_limits(hb_ms=1000)
        M.go_cool(g, guard, (), limit_s=1800)
        guard.stop(safe=False)
        s = wait_run(g, False, 4)
        res["trip_heartbeat"] = s
        log(f"heartbeat trip: RUN off after {s['after_s']} s, causes {s['causes']}")
        g.set_limits(hb_ms=2000)
        g.clear()
        guard = G.Guard(t, csv=a.out / "thermal2.csv")
        guard.read_first()
        guard.start()
        save()

        # 3. the DIMMs, one at a time: its BIOS (UART), a BIST. --parallel: every BIOS at once,
        # the UART on the last DIMM (the others' verdict their BIST; no lane repair without their
        # logs): a quarter of the time hot on a card with no fan
        uart = M.Uart(a.tty)
        on, res["dimms"] = [], {}
        M.go_cool(g, guard, (), limit_s=1800)
        for grp in ([dimms] if a.parallel else [[d] for d in dimms]):
            g.uart(grp[-1])
            uart.mark(a.out / f"uart_C{grp[-1]}.log")
            g.set_dimms(on + grp, acc=False)
            cs = {d: csr_up(t, d) for d in grp}
            rs = {grp[-1]: M.wait_bios(cs[grp[-1]], uart, timeout=a.bios_s)}
            if len(grp) > 1:
                time.sleep(10)          # the others' BIOS memtests (on the BIST's addresses) done
            for d in grp[:-1]:
                hw = bool(cs[d].r("sdram_dfii_control") & ddr4.DFII_SEL)
                rs[d] = {"bios_hw": hw, "bios_ok": hw, "bios_s": rs[grp[-1]]["bios_s"],
                         "bios_memtest": None}
            for d in grp:
                r = res["dimms"][d] = run_dimm(a, g, guard, d, cs[d], rs[d],
                                               uart.text() if d == grp[-1] else "")
                if r["ok"]:
                    on.append(d)
                save()
            g.set_dimms(on, acc=False)
        if sorted(on) != [0, 1, 2, 3]:
            raise SystemExit(f"DIMMs calibrated: {on}; the accelerator needs all four")

        # 4. the accelerator: on, its ID; the calibration bits in STATUS
        g.set_dimms(on, acc=True)
        t0 = time.time()
        while not g.status()["acc_on"] and time.time() - t0 < 5:
            time.sleep(0.05)
        ident, st = t.reg_read(R.R_ID), t.reg_read(R.R_STATUS)
        res["acc"] = {"acc_on": g.status()["acc_on"], "id": hex(ident), "status": hex(st),
                      "ok": ident == R.ID_OTPU}
        log(f"accelerator: {res['acc']}")
        if not res["acc"]["ok"]:
            raise SystemExit("the accelerator's ID register does not answer")
        # 5. the AXI4 master (XDMA's clock and reset: the link up)
        try:
            res["order"] = burst_order(t, 0x1000)
        except JtagAxiError as e:
            res["order"] = {"ok": False, "error": str(e)}
        log(f"AXI4 burst order: {res['order']}")
        save()
        if not res["order"]["ok"]:
            raise SystemExit("no DRAM access through the AXI4 JTAG master")
        # 6. otpu-selftest's stages over JTAG (--kernel: its kernel stage alone)
        if a.kernel:
            res["kernel"] = kernel_check(t, guard, g)
            res["selftest_rc"] = 0 if res["kernel"]["ok"] else 1
        elif not a.no_selftest:
            from opentpu.host import selftest
            if a.xdma:
                res["selftest_rc"] = selftest.main(["--dev", a.xdma] + a.selftest_args.split())
            else:
                res["selftest_rc"] = selftest.main(["--bw-mib", "1"], transport=JtagTransport(t))
            log(f"selftest: rc {res['selftest_rc']}")
        rc = 0 if res.get("selftest_rc", 0) == 0 else 1
        if a.hold:
            # 7. RUN held for other processes on the PCIe link (otpu-chat, otpu-serve, perf):
            # the guard kicks and watches until the time is up, a signal or a trip
            log(f"holding RUN for {a.hold:g} min (SIGTERM ends it); DIR/HOLDING while it does")
            hold = a.out / "HOLDING"
            hold.write_text(time.strftime("%F %T\n"))
            end = time.time() + 60 * a.hold
            try:
                while time.time() < end:
                    M.check(g, guard)
                    time.sleep(5)
                res["hold_end"] = "time up"
            except KeyboardInterrupt as e:
                res["hold_end"] = f"stopped ({e})"
            except M.Tripped as e:
                res["hold_end"] = f"tripped: {e}"
                rc = 1
            finally:
                hold.unlink(missing_ok=True)
            log(f"hold: {res['hold_end']}")
        res["status_end"] = g.status()
    finally:
        try:
            G.Gov(t).safe()
        except OSError:
            pass
        if guard:
            guard.stop()
            res["guard"] = {"tripped": guard.tripped, "booted": guard.booted,
                            "errors": guard.errors[-20:]}
        try:
            res["status_safe"] = G.Gov(t).status()
            res["mon_end"] = t.mon()
        except OSError as e:
            res["end_error"] = str(e)
        save()
        t.close()
    log(f"done {a.out / 'fullimage.json'}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
