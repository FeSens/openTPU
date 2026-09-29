"""otpu-memcal: the DDR3 calibration of a bitstream whose memory controllers are LiteDRAM (CAPS
bit27, docs/litedram.md sections 7 and 9). The MIG bitstreams calibrate themselves in hardware.
LiteDRAM's PHY is calibrated through each controller's CSRs in the BAR0 window at R_MEMCAL
(0x10000) by opentpu.host.ddrcal's algorithm: per channel the CK phase is scanned over one tCK
with the channel's BIST as the traffic check, the phase goes to the centre of the window common
to all nine byte lanes, write latency and read leveling are set there, and the channel's ready
bit (STATUS CALIB0 / CALIB1) rises.

Two ways to run it:
- by the core itself (gen_core.py --selfcal, opentpu.host.selfcal): a small CPU in the LiteDRAM
  core runs that algorithm at reset, so the channels come up calibrated. ensure() finds the
  STATUS bits set, or waits for the CPU when it is still running; a channel the CPU failed is
  calibrated from the host (the CPU held first);
- from the host (ddrcal: cores without the CPU, or `cal --force`). The host holds the core's
  CPU first, if there is one, and leaves it held, so it does not calibrate again behind the
  host's back; `selfcal` releases it, and it calibrates both channels again.

    otpu-memcal                 status: CAPS, the STATUS calibration bits, the core's CPU and
                                its result, the last calibration run by ensure()
    otpu-memcal cal [--force]   calibrate the channels not calibrated yet (--force: both, from
                                the host)
    otpu-memcal selfcal         the core's CPU calibrates both channels again

Board() calls ensure(), so every tool that opens the card calibrates it once per configuration;
the scan writes over each channel's first 64 MiB, which holds nothing before calibration. The
controller's CSR map and init sequence (csr.csv, sdram_init.py) ship in opentpu/host/litedram/,
written by tools/litedram/gen_core.py together with the core the bitstream was built from. The
last result per device is kept in the run directory (<device>.memcal.json).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import ddrcal
from . import regs as R
from . import selfcal
from .runstate import run_dir

DATA = Path(__file__).with_name("litedram")
CALIB = (R.ST_CALIB0, R.ST_CALIB1)
SELFCAL_WAIT = 60.0             # s; the CPU needs a few seconds per channel


def hostcal(t) -> bool:
    """The bitstream's controllers are calibrated through their CSRs (register map 2 and CAPS
    bit27: LiteDRAM, by the host or by the core's own CPU)."""
    return R.regmap(t.reg_read(R.R_REGMAP)) >= 2 and bool(t.reg_read(R.R_CAPS) & R.CAP_HOSTCAL)


def csr(t, data: Path = DATA):
    """The controllers' CSRs over the transport's BAR0 window."""
    return ddrcal.WordCsr(ddrcal.csr_map(data / "csr.csv"), t.reg_read, t.reg_write,
                          base=R.R_MEMCAL)


def uncalibrated(t, force: bool = False) -> list:
    st = t.reg_read(R.R_STATUS)
    return [ch for ch, bit in enumerate(CALIB) if force or not st & bit]


def ensure(t, force: bool = False, stride: int = 1, data: Path = DATA, log=print) -> dict | None:
    """Calibrate the channels whose STATUS bit is low (all with `force`, from the host) on a
    LiteDRAM bitstream; None when there is nothing to do. With the core's CPU: its run is waited
    for and its result returned, and the channels it failed are calibrated from the host. Raises
    ddrcal.CalError when a channel fails."""
    if not hostcal(t):
        return None
    todo = uncalibrated(t, force)
    if not todo:
        return None
    c = csr(t, data)
    out = {"time": time.time(), "by": "host", "channels": {}}
    if selfcal.present(c):
        if not force:
            log("DDR3: the LiteDRAM core's CPU is calibrating; waiting ...")
            try:
                selfcal.wait(c, SELFCAL_WAIT)
            except TimeoutError as e:
                log(f"DDR3: {e}")
            res = selfcal.result(c, data / "sdram_init.py")
            out["by"] = "selfcal"
            out["channels"] = {ch: r for ch, r in res.items() if r["state"] == "ok"}
            todo = uncalibrated(t)
            if not todo:
                _save(t, out)
                return out
            log("DDR3: the core's CPU left channel(s) uncalibrated: " + ", ".join(
                f"{ch} ({res[ch]['error'] if ch in res else 'not run'})" for ch in todo)
                + "; the host calibrates them")
            out["by"] = "selfcal+host"
        selfcal.hold(c)
    for ch in todo:
        t0 = time.time()
        log(f"DDR3 channel {ch}: calibrating (LiteDRAM, host-driven) ...")
        res = ddrcal.calibrate_channel(ddrcal.Chan(c, ch), data / "sdram_init.py", stride=stride,
                                       log=lambda m, ch=ch: log(f"DDR3 channel {ch}: {m}"))
        res["seconds"] = round(time.time() - t0, 1)
        out["channels"][ch] = res
    low = uncalibrated(t)
    if low:
        raise ddrcal.CalError(f"channel(s) {low} calibrated but their STATUS bit stays low")
    _save(t, out)
    return out


def selfcal_again(t, data: Path = DATA, log=print) -> dict:
    """The core's CPU calibrates every channel again (hold, release, wait): its result."""
    c = csr(t, data)
    if not selfcal.present(c):
        raise ddrcal.CalError("the bitstream's LiteDRAM core has no calibration CPU")
    selfcal.hold(c)
    selfcal.release(c)
    t0 = time.time()
    selfcal.wait(c, SELFCAL_WAIT)
    log(f"the core's CPU finished in {time.time() - t0:.1f} s")
    return selfcal.result(c, data / "sdram_init.py")


def _path(t) -> Path:
    return run_dir() / f"{getattr(t, 'devname', None) or 'card'}.memcal.json"


def _save(t, out: dict) -> None:
    try:
        p = _path(t)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1))
    except OSError:
        pass                        # the result is also in the log


def last(t) -> dict | None:
    try:
        return json.loads(_path(t).read_text())
    except (OSError, ValueError):
        return None


def describe(ch, r: dict) -> str:
    """A channel's result (ddrcal's or the core's CPU's) on one line."""
    if r.get("state", "ok") != "ok":
        return f"channel {ch}: {r['state']} ({r['error']})"
    s = (f"channel {ch}: CK step {r['dqs_steps']}, common write window {r['window_ps']} ps "
         f"({r['window_steps']} steps), write latency {r['write_latency']}, read taps "
         f"{[x['tap'] for x in r['read']]}")
    offs = [f"lane {m} bit {i} {o:+d}" for m, row in enumerate(r.get("bit_offsets", []))
            for i, o in enumerate(row) if o]
    if offs:
        s += f", read bitslip offsets: {', '.join(offs)}"
    return s + (f", {r['seconds']} s" if "seconds" in r else "")


def main(argv: list[str] | None = None, open_transport=None) -> int:
    ap = argparse.ArgumentParser(prog="otpu-memcal", description=__doc__.split("\n")[0])
    ap.add_argument("what", nargs="?", default="status", choices=["status", "cal", "selfcal"])
    ap.add_argument("--force", action="store_true", help="cal: both channels, from the host")
    ap.add_argument("--stride", type=int, default=1, help="DQS scan stride, fine steps")
    ap.add_argument("--dev", default="/dev/xdma0")
    a = ap.parse_args(argv)
    from .board import Board, XdmaTransport
    t = open_transport() if open_transport else XdmaTransport(a.dev)
    with Board(t, calibrate=False) as b:
        st = b.t.reg_read(R.R_STATUS)
        hc = hostcal(b.t)
        own = hc and selfcal.present(csr(b.t, DATA))
        kind = ("LiteDRAM, calibrated by the core's CPU" if own else
                "LiteDRAM (host calibration)" if hc else "self-calibrating (MIG)")
        print(f"controllers: {kind}; STATUS calibration: " + ", ".join(
            f"channel {ch} {'yes' if st & bit else 'no'}" for ch, bit in enumerate(CALIB)))
        if a.what == "selfcal":
            if not own:
                print("nothing to do: the bitstream's LiteDRAM core has no calibration CPU")
                return 1
            res = selfcal_again(b.t, DATA)
            for ch, r in sorted(res.items()):
                print(describe(ch, r))
            return 0 if res and all(r["state"] == "ok" for r in res.values()) else 1
        if a.what == "cal":
            if not hc:
                print("nothing to do: the bitstream's controllers calibrate themselves")
                return 0
            try:
                res = ensure(b.t, force=a.force, stride=a.stride, data=DATA)
            except ddrcal.CalError as e:
                print(f"calibration FAILED: {e}")
                return 1
            if res is None:
                print("both channels are calibrated (--force to redo)")
            return 0
        if own:
            c = csr(b.t, DATA)
            s = selfcal.state(c)
            print("the core's CPU: " + ("held (the host calibrates)" if selfcal.held(c) else
                                        "done" if s["done"] else "running") + "; " + ", ".join(
                f"channel {ch} {x}{' (' + e + ')' if e else ''}"
                for ch, (x, e) in s["channels"].items()))
            for ch, r in sorted(selfcal.result(c, DATA / "sdram_init.py").items()):
                print("  " + describe(ch, r))
        prev = last(b.t)
        if prev:
            for ch, r in sorted(prev["channels"].items()):
                print(f"last calibration ({prev.get('by', 'host')}), " + describe(ch, r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
