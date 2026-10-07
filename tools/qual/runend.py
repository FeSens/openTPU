"""Card check of how runs end (docs/host.md, "Ending a run"): a run cut short by CTRL = 0 with
DRAM reads and writes in flight leaves the card quiet (Board.stop: STATUS QUIET within its wait)
and the next program runs bit for bit as the ISA simulator (checks.run_demo); a holder killed
mid-run (SIGKILL) leaves its run to the next holder's open, which waits and refuses, or cuts it
short with OTPU_STOP_RUN=1; a holder sent SIGTERM mid-run stops its run on the way out (exit
status 143, the next open finds RUN clear).

    otpu-lock -- python3 tools/qual/runend.py [--dev /dev/xdma0]

Takes the card three times (the killed holders are child processes), so run it under otpu-lock
(OTPU_LOCK_HELD) or on a card nobody else waits for. Prints a [PASS] / [FAIL] line per check;
exit status 1 when one fails.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from opentpu import isa as I  # noqa: E402
from opentpu.host import board as B  # noqa: E402
from opentpu.host import regs as R  # noqa: E402
from opentpu.host.board import Board, XdmaTransport, device_config  # noqa: E402
from opentpu.host.checks import run_demo  # noqa: E402

BUSY_AT, BUSY_OUT = 0x8000000, 0x9000000    # 128 / 144 MiB up: no other check's region
BUSY_PROG = 0xA000000


def busy_program(n: int = 50_000_000) -> np.ndarray:
    """n iterations of a 16 KiB LD and ST (minutes): DRAM reads and writes in flight always."""
    return np.asarray(I.assemble([I.loop(2, n), I.ld(BUSY_AT, 0, 4096),
                                  I.st(BUSY_OUT, 0, 4096), I.halt()]), np.uint32)


CHILD = """
import os, sys, time
sys.path.insert(0, {root!r})
from opentpu.host.board import Board, XdmaTransport
sys.path.insert(0, {tools!r})
from runend import BUSY_PROG, busy_program
b = Board(XdmaTransport({dev!r}))
b.load_program(BUSY_PROG, busy_program())
b.start()
print("running", flush=True)
time.sleep(600)
"""


def _child(dev: str):
    code = CHILD.format(root=str(ROOT), tools=str(Path(__file__).parent), dev=dev)
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    if p.stdout.readline().strip() != "running":
        p.kill()
        raise RuntimeError("the child holder did not start its run")
    return p


def cut_short(dev: str) -> tuple[bool, str]:
    with Board(XdmaTransport(dev)) as b:
        cfg = device_config(b.info())
        b.load_program(BUSY_PROG, busy_program())
        b.start()
        time.sleep(0.05)
        t0 = time.perf_counter()
        whole = b.stop()
        dt = time.perf_counter() - t0
        st = b.t.reg_read(R.R_STATUS)
        ok, msg, _ = run_demo(b, cfg)
    good = not whole and not st & R.ST_RUN and (st & B.QUIET) == B.QUIET and ok
    return good, (f"stop {1e3 * dt:.1f} ms (cut short: {not whole}), STATUS {st:#x}; "
                  f"then the demo: {msg}")


def killed_holder(dev: str) -> tuple[bool, str]:
    p = _child(dev)
    p.send_signal(signal.SIGKILL)
    p.wait()
    out = []
    B.QUIESCE_WAIT = 2.0
    try:
        Board(XdmaTransport(dev)).close()
        out.append("open did not refuse a run left going")
    except B.CardRunning:
        pass
    os.environ["OTPU_STOP_RUN"] = "1"
    try:
        with Board(XdmaTransport(dev)) as b:
            st = b.t.reg_read(R.R_STATUS)
            ok, msg, _ = run_demo(b, device_config(b.info()))
    finally:
        del os.environ["OTPU_STOP_RUN"]
    if st & R.ST_RUN or not ok:
        out.append(f"after OTPU_STOP_RUN=1: STATUS {st:#x}, demo {msg}")
    return not out, "; ".join(out) or f"refused, then cut short; the demo: {msg}"


def terminated_holder(dev: str) -> tuple[bool, str]:
    p = _child(dev)
    p.send_signal(signal.SIGTERM)
    rc = p.wait(timeout=30)
    t = XdmaTransport(dev, dma=False)
    try:
        st = t.reg_read(R.R_STATUS)
    finally:
        t.close()
    good = rc == 128 + signal.SIGTERM and not st & R.ST_RUN
    return good, f"exit status {rc}, then STATUS {st:#x} (RUN clear: the run was stopped)"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dev", default="/dev/xdma0")
    a = ap.parse_args(argv)
    rows = []
    for name, fn in (("a run cut short", cut_short), ("a holder killed mid-run", killed_holder),
                     ("a holder sent SIGTERM mid-run", terminated_holder)):
        try:
            rows.append((name, fn(a.dev)))
        except Exception as e:          # noqa: BLE001
            rows.append((name, (False, f"{type(e).__name__}: {e}")))
        ok, msg = rows[-1][1]
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {msg}", flush=True)
    return 0 if all(ok for _, (ok, _) in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
