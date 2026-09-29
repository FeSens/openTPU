"""otpu-memcal: the host's DDR3 calibration for a bitstream whose memory controllers are
LiteDRAM (CAPS bit26, docs/litedram.md section 7). The MIG bitstreams calibrate themselves in
hardware; LiteDRAM's A7DDRPHY is calibrated from the host, through each controller's CSRs in the
BAR0 window at R_MEMCAL (0x10000), by opentpu.host.ddrcal: per channel the write DQS phase is
scanned over one tCK with the channel's BIST as the traffic check, the phase goes to the centre
of the window common to all nine byte lanes, write latency and read leveling are set there, and
the channel's ready bit (STATUS CALIB0 / CALIB1) rises.

    otpu-memcal                 status: CAPS, the STATUS calibration bits, the last calibration
    otpu-memcal cal [--force]   calibrate the channels not calibrated yet (--force: both)

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
from .runstate import run_dir

DATA = Path(__file__).with_name("litedram")
CALIB = (R.ST_CALIB0, R.ST_CALIB1)


def hostcal(t) -> bool:
    """The bitstream's controllers want the host's calibration (register map 2 and CAPS bit26)."""
    return R.regmap(t.reg_read(R.R_REGMAP)) >= 2 and bool(t.reg_read(R.R_CAPS) & R.CAP_HOSTCAL)


def csr(t, data: Path = DATA):
    """The controllers' CSRs over the transport's BAR0 window."""
    return ddrcal.WordCsr(ddrcal.csr_map(data / "csr.csv"), t.reg_read, t.reg_write,
                          base=R.R_MEMCAL)


def ensure(t, force: bool = False, stride: int = 1, data: Path = DATA, log=print) -> dict | None:
    """Calibrate the channels whose STATUS bit is low (all with `force`) on a host-calibrated
    bitstream; None when there is nothing to do. Raises ddrcal.CalError when a channel fails."""
    if not hostcal(t):
        return None
    st = t.reg_read(R.R_STATUS)
    todo = [ch for ch, bit in enumerate(CALIB) if force or not st & bit]
    if not todo:
        return None
    c = csr(t, data)
    out = {"time": time.time(), "channels": {}}
    for ch in todo:
        t0 = time.time()
        log(f"DDR3 channel {ch}: calibrating (LiteDRAM, host-driven) ...")
        res = ddrcal.calibrate_channel(ddrcal.Chan(c, ch), data / "sdram_init.py", stride=stride,
                                       log=lambda m, ch=ch: log(f"DDR3 channel {ch}: {m}"))
        res["seconds"] = round(time.time() - t0, 1)
        out["channels"][ch] = res
    st = t.reg_read(R.R_STATUS)
    for ch in todo:
        if not st & CALIB[ch]:
            raise ddrcal.CalError(f"channel {ch} calibrated but its STATUS bit stays low")
    _save(t, out)
    return out


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


def main(argv: list[str] | None = None, open_transport=None) -> int:
    ap = argparse.ArgumentParser(prog="otpu-memcal", description=__doc__.split("\n")[0])
    ap.add_argument("what", nargs="?", default="status", choices=["status", "cal"])
    ap.add_argument("--force", action="store_true", help="cal: both channels, calibrated or not")
    ap.add_argument("--stride", type=int, default=1, help="DQS scan stride, fine steps")
    ap.add_argument("--dev", default="/dev/xdma0")
    a = ap.parse_args(argv)
    from .board import Board, XdmaTransport
    t = open_transport() if open_transport else XdmaTransport(a.dev)
    with Board(t, calibrate=False) as b:
        st = b.t.reg_read(R.R_STATUS)
        hc = hostcal(b.t)
        print(f"controllers: {'LiteDRAM (host calibration)' if hc else 'self-calibrating (MIG)'}; "
              "STATUS calibration: " + ", ".join(
                  f"channel {ch} {'yes' if st & bit else 'no'}" for ch, bit in enumerate(CALIB)))
        if a.what == "cal":
            if not hc:
                print("nothing to do: the bitstream's controllers calibrate themselves")
                return 0
            try:
                res = ensure(b.t, force=a.force, stride=a.stride)
            except ddrcal.CalError as e:
                print(f"calibration FAILED: {e}")
                return 1
            if res is None:
                print("both channels are calibrated (--force to redo)")
            return 0
        prev = last(b.t)
        if prev:
            for ch, r in sorted(prev["channels"].items()):
                print(f"last calibration, channel {ch}: DQS step {r['dqs_steps']}, common write "
                      f"window {r['window_ps']} ps, write latency {r['write_latency']}, "
                      f"{r['seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
