#!/usr/bin/env python3
"""The BCU1525's WG image (otpu_fpga_top_bcu_wg: one WG slice on four DDR4-2400 DIMMs behind AMD's
MIG, docs/bcu1525_wg.md) on the card, in two steps:

    wg_bringup.py load --bit otpu.bit --out DIR      # over JTAG; then a warm reboot of the host
    wg_bringup.py up --out DIR [--ltx otpu.ltx] [--mem quick|full|none] [--no-trips]
        [--resume 65] [--selftest-args "--model qwen3"]
        [--run "tools.q35_split --model qwen35-4b" ...] [--hold MIN]

load programs the image over JTAG (volatile: until the card's power is cycled) and checks DONE.
Its PCIe link enumerates at the host's next boot (a warm reboot keeps the image), and nothing
more can happen before it: the debug hub, the JTAG-to-AXI masters' way to the registers, runs
on a MIG's debug clock, and SAFE holds every MIG in reset.

up works through PCIe: the governor and the registers through BAR0 (registers only, without the
device lock, so otpu-selftest and otpu-diag run beside it), the SYSMON and the configuration's
way out (JPROGRAM: the flash's image) over JTAG. In order: the governor's RESUME at --resume
(65 C: its reset value, 55 C, is under the idle die fan_pid holds at 60 C, and GO waits for the
die at or below RESUME; at most the build's RESUME_MAX, 69.9 C, and TRIP less 10 C; written again
with TRIP after the thermal trip; wg.json's "resume"); SAFE as configured (nothing on, the control
registers fenced); the guard (gov.Guard: heartbeat, die, regulator); a forced thermal trip and a
heartbeat trip (--no-trips skips them); the four DIMMs on together (each MIG calibrates itself,
no BIOS); every MIG's calibration report over JTAG (DIR/migs.txt, --ltx);
the accelerator on, its ID answering the proof that all four calibrated (it runs only with
every DIMM up, and nothing else tells the host a DIMM is up: the DMA gate answers SLVERR for a
DIMM that is not, and XDMA turns that into an engine error, so no DMA before it); otpu-diag's
memory section on all 8 channels, the 4 GiB this image reaches (--mem quick: walking data,
address bits, random blocks, partial writes, bandwidth; full: also a march C- over every byte);
otpu-selftest (OTPU_TARGET=bcu1525-wg; --selftest-args); each --run "MODULE ARGS" (python -m,
the arguments split as a shell would, the repository on its path: DIR/run<i>.log); --hold MIN
with RUN held under the guard for other processes; SAFE at the end. otpu-diag, otpu-selftest
and the runs are processes of their own (each takes the device lock), the guard checked while
they run.

Writes DIR/wg.json, DIR/thermal*.csv (the guard's), DIR/migs.txt, DIR/diag.json, DIR/*.log.
"""
from __future__ import annotations

import os

os.environ.setdefault("OTPU_TARGET", "bcu1525-wg")      # before opentpu reads it

import argparse                                         # noqa: E402
import json                                             # noqa: E402
import shlex                                            # noqa: E402
import signal                                           # noqa: E402
import subprocess                                       # noqa: E402
import sys                                              # noqa: E402
import threading                                        # noqa: E402
import time                                             # noqa: E402
from pathlib import Path                                # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from opentpu.host import gov as G                       # noqa: E402
from opentpu.host import regs as R                      # noqa: E402

FENCED = 0xDEAD0FF0             # otpu_axil_off's read data (OKAY) while its slave is off
SERIAL = os.environ.get("BCU_CARD0_SERIAL", "")     # the card's FT4232H serial (--serial)
DIMMS = (0, 1, 2, 3)
RESUME_C = 65.0                 # up's RESUME (the governor's reset value, 55 C, is under the idle
                                # die that fan_pid holds at 60 C: GO never came, 2026-10-29 11:47)
RESUME_MAX_C = 70.0             # --resume's ceiling (the build's RESUME_MAX: 69.9 C)
RESUME_MARGIN = 20              # RESUME at most TRIP less this many codes (10 C: the build's
                                # RESUME_MAX is TRIP_MAX less 2 x WARN_D, 20 codes)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Tripped(RuntimeError):
    pass


class Card:
    """The guard's transport: the registers through BAR0 (`pcie`), the SYSMON and JPROGRAM over
    JTAG (`jtag`, a JtagAxi; without it neither). While `busy` is set (a long JTAG command of
    ours, the MIG report) mon fails at once instead of queueing behind it: a reading thread
    held up longer than the guard's stale_s would stop the heartbeat, and the governor would go
    SAFE on it; the die still comes from the governor meanwhile."""

    def __init__(self, pcie, jtag=None):
        self.pcie, self.jtag = pcie, jtag
        self.busy = threading.Event()
        if jtag is not None:
            self.mon, self.boot = self._mon, jtag.boot

    def _mon(self) -> dict:
        if self.busy.is_set():
            raise OSError("JTAG busy (the MIG report)")
        return self.jtag.mon()

    def reg_read(self, a: int) -> int:
        return self.pcie.reg_read(a)

    def reg_write(self, a: int, v: int) -> None:
        self.pcie.reg_write(a, v)

    def reg_read_many(self, offs) -> list[int]:
        return self.pcie.reg_read_many(offs)


def check(g: G.Gov, guard) -> dict:
    """The governor still RUN and the guard not tripped."""
    if guard.tripped or guard.booted:
        raise Tripped(f"guard: {guard.tripped}")
    s = g.status()
    if not s["run"]:
        raise Tripped(f"governor SAFE: {s['causes']} at {s['temp_c']:.1f} C")
    return s


GO_SAY_S = 30.0                 # go_cool's line while the governor refuses GO, every so often


def waiting_line(s: dict, resume_c: float, waited_s: float, limit_s: float) -> str:
    """What go_cool logs while GO is refused: the die against RESUME, and what else holds it."""
    why = [w for k, w in (("reg_cool", "the regulator above its RESUME"),
                          ("temp_valid", "no valid temperature")) if not s[k]]
    why += [w for k, w in (("alarm", "the temperature alarm"), ("ot", "over-temperature"))
            if s[k]]
    return (f"waiting for GO: die {s['temp_c']:.1f} C, RESUME {resume_c:.1f} C"
            + ("" if s["cool"] else " (the die must cool to it; a fan target above RESUME never "
               f"does: --resume, at most {RESUME_MAX_C - 0.1:g})")
            + "".join(f"; {w}" for w in why)
            + f"; {waited_s:.0f} of {limit_s:.0f} s")


def go_cool(g: G.Gov, guard, dimms, limit_s: float = 1800.0, sleep=time.sleep) -> None:
    """GO once the die (and the regulator) are at or below RESUME; the governor refuses a GO
    above it, so this retries, and says what it waits for every GO_SAY_S."""
    t0 = said = time.time()
    while True:
        s = g.status()
        if s["cool"] and s["reg_cool"] and s["temp_valid"] and not s["alarm"] and not s["ot"] \
                and g.go(dimms=dimms):
            return
        if time.time() - said >= GO_SAY_S:      # (the wrappers show the latest of these lines)
            said = time.time()
            log(waiting_line(s, g.limits()["resume_c"], said - t0, limit_s))
        if guard is not None and (guard.tripped or guard.booted):
            raise Tripped(f"guard: {guard.tripped}")
        if time.time() - t0 > limit_s:
            raise Tripped(f"GO refused for {limit_s:.0f} s: {s}")
        sleep(2)


def set_resume(g: G.Gov, want_c: float, lim: dict) -> dict:
    """RESUME at `want_c`, inside the governor's clamp: at most the build's RESUME_MAX and TRIP
    (`lim`: limits() before) less RESUME_MARGIN codes. What was asked, the register before and
    after, the ceiling (C)."""
    top = min(G.code_of(lim["resume_max_c"]), G.code_of(lim["trip_c"]) - RESUME_MARGIN)
    g.set_limits(resume_c=G.celsius(min(G.code_of(want_c), top)))
    return {"asked_c": want_c, "was_c": round(lim["resume_c"], 2),
            "set_c": round(g.limits()["resume_c"], 2), "max_c": round(G.celsius(top), 2)}


def wait_run(g: G.Gov, want: bool, limit: float) -> dict:
    t0 = time.time()
    while time.time() - t0 < limit:
        s = g.status()
        if s["run"] == want:
            s["after_s"] = round(time.time() - t0, 2)
            return s
        time.sleep(0.1)
    raise SystemExit(f"governor RUN not {want} after {limit} s: {g.status()}")


def safe0(t, g: G.Gov) -> dict:
    """SAFE as configured: not RUN, no DIMM, the accelerator off, its control registers fenced."""
    s, ident = g.status(), t.reg_read(R.R_ID)
    return {"ok": not s["run"] and not s["dimms_on"] and not s["acc_on"] and ident == FENCED,
            "id": hex(ident), "status": s}


def trips(g: G.Gov, guard_of, out: Path, resume_c: float | None = None) -> tuple[dict, object]:
    """A thermal trip (TRIP under the die, below RESUME too) and a heartbeat trip (no kicks), no
    DIMM on; TRIP back at 80 C and RESUME at `resume_c` again with it (None: as it was); returns
    the results and the guard started again after the second."""
    res = {}
    guard = guard_of(out / "thermal.csv")
    go_cool(g, guard, ())
    g.set_limits(trip_c=g.status()["temp_c"] - 3)
    s = wait_run(g, False, 5)
    res["thermal"] = {"after_s": s["after_s"], "causes": s["causes"]}
    log(f"thermal trip: RUN off after {s['after_s']} s, causes {s['causes']}")
    g.set_limits(trip_c=80.0, resume_c=resume_c)
    lim = g.limits()
    res["thermal"].update(trip_c=round(lim["trip_c"], 2), resume_c=round(lim["resume_c"], 2))
    g.clear()
    g.set_limits(hb_ms=1000)
    go_cool(g, guard, ())
    guard.stop(safe=False)
    s = wait_run(g, False, 4)
    res["heartbeat"] = {"after_s": s["after_s"], "causes": s["causes"]}
    log(f"heartbeat trip: RUN off after {s['after_s']} s, causes {s['causes']}")
    g.set_limits(hb_ms=2000)
    g.clear()
    return res, guard_of(out / "thermal2.csv")


def dimms_up(t, g: G.Gov, guard, card: Card, a, res: dict) -> None:
    """The four DIMMs on, each MIG's report (--ltx), then the accelerator: up or SystemExit."""
    go_cool(g, guard, ())
    log("the four DIMMs on")
    g.set_dimms(list(DIMMS), acc=False)
    time.sleep(a.cal_s)
    s = check(g, guard)
    res["dimms_on"] = {"dimms": s["dimms_on"], "die_c": round(s["temp_c"], 1)}
    if card.jtag is not None and a.ltx:
        card.busy.set()
        try:
            res["migs"] = card.jtag.migs(a.ltx, a.out / "migs.txt")
        except OSError as e:
            res["migs"] = {"error": str(e)}
        finally:
            card.busy.clear()
        log(f"MIG calibration: {res['migs']}")
        check(g, guard)
    g.set_dimms(list(DIMMS), acc=True)
    t0, ident = time.time(), FENCED
    while time.time() - t0 < a.acc_s:
        ident = t.reg_read(R.R_ID)
        if ident == R.ID_OTPU:
            break
        time.sleep(0.1)
    s = check(g, guard)
    res["acc"] = {"ok": ident == R.ID_OTPU, "id": hex(ident), "acc_on": s["acc_on"],
                  "after_s": round(time.time() - t0, 2), "status": hex(t.reg_read(R.R_STATUS))}
    log(f"accelerator: {res['acc']}")
    if not res["acc"]["ok"]:
        raise SystemExit("the accelerator did not start: a DIMM did not calibrate "
                         "(DIR/migs.txt with --ltx names it)")


def child(name: str, argv: list[str], out: Path, g: G.Gov, guard, timeout_s: float) -> int:
    """One of opentpu's tools in a process of its own (its own XdmaTransport and device lock),
    its output into DIR/name.log; the guard checked every 5 s while it runs, the child stopped
    (SIGTERM, then SIGKILL) on a trip."""
    env = {**os.environ, "PYTHONPATH": str(ROOT), "OTPU_TARGET": os.environ["OTPU_TARGET"]}
    with open(out / f"{name}.log", "w") as f:
        p = subprocess.Popen([sys.executable, "-u", "-m", *argv], stdout=f, stderr=subprocess.STDOUT,
                             env=env, cwd=ROOT, start_new_session=True)
        t0 = time.time()
        try:
            while p.poll() is None:
                check(g, guard)
                if time.time() - t0 > timeout_s:
                    raise Tripped(f"{name}: no end in {timeout_s:.0f} s")
                time.sleep(5)
        except BaseException:
            p.terminate()
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
            raise
    log(f"{name}: rc {p.returncode} ({time.time() - t0:.0f} s; {out / (name + '.log')})")
    return p.returncode


def hold(g: G.Gov, guard, minutes: float, out: Path) -> str:
    log(f"holding RUN for {minutes:g} min (SIGTERM ends it); DIR/HOLDING while it does")
    f = out / "HOLDING"
    f.write_text(time.strftime("%F %T\n"))
    end = time.time() + 60 * minutes
    try:
        while time.time() < end:
            check(g, guard)
            time.sleep(5)
        return "time up"
    except KeyboardInterrupt as e:
        return f"stopped ({e})"
    finally:
        f.unlink(missing_ok=True)


def cmd_load(a) -> int:
    from opentpu.host.jtagaxi import JtagAxi
    a.out.mkdir(parents=True, exist_ok=True)
    res = {"bit": str(a.bit), "start": time.strftime("%F %T")}
    t = JtagAxi(a.serial, log=a.out / "vivado.log")
    try:
        res["mon_before"] = t.mon()
        log(f"JTAG {t.target}; before: {res['mon_before']}")
        res["axis"] = t.program(a.bit)
        res["done"] = t.done()
        res["mon_after"] = t.mon()
        log(f"programmed {a.bit}: DONE {res['done']}, {res['mon_after']}")
    finally:
        (a.out / "load.json").write_text(json.dumps(res, indent=1, default=str))
        t.close()
    if res["done"] != [1, 1, 1]:
        log("DONE is not 1 in every SLR")
        return 1
    log("loaded: warm-reboot the host for its PCIe link (the image stays), then `up`")
    return 0


def cmd_up(a) -> int:
    from opentpu.host.board import XdmaTransport
    a.out.mkdir(parents=True, exist_ok=True)
    res = {"start": time.strftime("%F %T"), "dev": a.dev, "target": os.environ["OTPU_TARGET"]}

    def save():
        (a.out / "wg.json").write_text(json.dumps(res, indent=1, default=str))

    pcie = XdmaTransport(a.dev, dma=False)
    jt = None
    if not a.no_jtag:
        from opentpu.host.jtagaxi import JtagAxi
        try:
            jt = JtagAxi(a.serial, log=a.out / "vivado.log")
        except BaseException:
            pcie.close()
            raise
    card = Card(pcie, jt)
    g = G.Gov(card)

    def guard_of(csv):
        gd = G.Guard(card, csv=csv)
        gd.read_first()
        return gd.start()
    guard, rc = None, 1
    try:
        if not g.present():
            raise SystemExit(f"no governor at {G.R_GOV:#x} (ID {g.rd(G.ID):#x})")
        res["limits"] = g.limits()
        res["resume"] = set_resume(g, a.resume, res["limits"])
        res["safe0"] = safe0(card, g)
        # the die from the governor and the SYSMON's JTAG port, not the wizard's DRP registers
        # over BAR0: f150m15's five reads here stopped its temperature bus, the governor's die
        # (docs/bcu1525_sysmon.md)
        res["die"] = G.die_readings(g, getattr(card, "mon", None))
        log(f"governor {res['limits']}\n   RESUME {res['resume']}\n"
            f"   SAFE after configuration: {res['safe0']}\n   die {res['die']}")
        save()
        if not res["safe0"]["ok"]:
            raise SystemExit("not SAFE after configuration")
        if a.no_trips:
            guard = guard_of(a.out / "thermal.csv")
        else:
            res["trips"], guard = trips(g, guard_of, a.out, res["resume"]["set_c"])
        res["bmc"] = guard.reg
        save()
        dimms_up(card, g, guard, card, a, res)
        # the image as its accelerator reports it, kept for the times the card is in SAFE and
        # its registers fenced (opentpu.host.cardimage: otpu-smi's Image row, otpu-chat)
        from opentpu.host import cardimage as CI
        try:
            res["image"] = CI.note(pcie, a.dev)
        except Exception as e:              # (a record is a convenience: the bring-up goes on)
            res["image"] = {"error": str(e)}
        if res["image"] and "error" not in res["image"]:
            log(f"image: {CI.text({**res['image'], 'source': 'registers'})}")
        save()
        if a.mem != "none":
            argv = ["opentpu.host.diag", "--dev", a.dev, "--only", "mem", "--mem", a.mem,
                    "--json", str(a.out / "diag.json")]
            res["diag_rc"] = child("diag", argv, a.out, g, guard, 7200)
            save()
        if not a.no_selftest:
            argv = ["opentpu.host.selftest", "--dev", a.dev] + a.selftest_args.split()
            res["selftest_rc"] = child("selftest", argv, a.out, g, guard, 7200)
            save()
        for i, r in enumerate(a.run):
            res.setdefault("run_rc", []).append(child(f"run{i}", shlex.split(r), a.out, g, guard,
                                                       7200))
            save()
        rc = 0 if all(res.get(k, 0) == 0 for k in ("diag_rc", "selftest_rc")) and \
            not any(res.get("run_rc", [])) else 1
        if a.hold:
            res["hold_end"] = hold(g, guard, a.hold, a.out)
            log(f"hold: {res['hold_end']}")
        res["status_end"] = g.status()
    except Tripped as e:
        res["tripped"] = str(e)
        log(f"TRIPPED: {e}")
        rc = 1
    except KeyboardInterrupt as e:
        res["stopped"] = str(e)
        log(f"stopped: {e}")
        rc = 1
    finally:
        try:
            g.safe()
        except OSError:
            pass
        if guard:
            guard.stop()
            res["guard"] = {"tripped": guard.tripped, "booted": guard.booted,
                            "errors": guard.errors[-20:]}
        try:
            res["status_safe"] = g.status()
            if jt:
                res["mon_end"] = jt.mon()
        except OSError as e:
            res["end_error"] = str(e)
        save()
        if jt:
            jt.close()
        pcie.close()
    log(f"done {a.out / 'wg.json'}: rc {rc}")
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ld = sub.add_parser("load", help="program the image over JTAG")
    ld.add_argument("--bit", required=True, type=Path)
    up = sub.add_parser("up", help="after the warm reboot: the image through PCIe")
    up.add_argument("--dev", default="/dev/xdma0")
    up.add_argument("--ltx", type=Path, help="the build's probes file: each MIG's report")
    up.add_argument("--no-jtag", action="store_true",
                    help="no JTAG (no SYSMON over JTAG, no JPROGRAM, no MIG report)")
    up.add_argument("--no-trips", action="store_true")
    up.add_argument("--resume", type=float, default=RESUME_C, metavar="C",
                    help=f"the governor's RESUME for the GOs (%(default)s C; at most "
                         f"{RESUME_MAX_C:g}, and TRIP less 10)")
    up.add_argument("--cal-s", type=float, default=5.0,
                    help="seconds from the DIMMs on to the accelerator (their calibration)")
    up.add_argument("--acc-s", type=float, default=10.0,
                    help="seconds for the accelerator's ID to answer")
    up.add_argument("--mem", choices=("quick", "full", "none"), default="quick")
    up.add_argument("--no-selftest", action="store_true")
    up.add_argument("--selftest-args", default="")
    up.add_argument("--run", action="append", default=[], metavar="MODULE ARGS",
                    help="a module to run after the self-test, under the guard (repeatable)")
    up.add_argument("--hold", type=float, default=0.0, metavar="MIN")
    for p in (ld, up):
        p.add_argument("--out", required=True, type=Path)
        p.add_argument("--serial", default=SERIAL)
    a = ap.parse_args(argv)
    if not a.serial and (a.cmd == "load" or not a.no_jtag):
        ap.error("--serial: the card's FT4232H serial (or BCU_CARD0_SERIAL)")
    if a.cmd == "up" and not a.resume <= RESUME_MAX_C:
        ap.error(f"--resume: at most {RESUME_MAX_C:g} C (the governor's RESUME_MAX)")

    def interrupted(sig, frame):
        raise KeyboardInterrupt(f"signal {sig}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    return cmd_load(a) if a.cmd == "load" else cmd_up(a)


if __name__ == "__main__":
    sys.exit(main())
