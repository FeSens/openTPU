#!/usr/bin/env python3
"""Simulation host only (a heavy Verilator run): replay a prepared cut through step f and the two-board RTL model.

No compile or reference is reconstructed from a model config here: this loads exactly the
pack's program/wait bytes and initial ranges, then uses x2_card.run_cuts for every check.
The cable stand-in has no PING/counter engine, so physical step b is intentionally omitted;
normal CardsTransport sequence/status checks and step e link snapshots still run. One RTL
run at a time, with four compiler workers; use nice 15 and on-disk OTPU_SIM_TMP externally.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pack", type=Path)
    ap.add_argument("--json", type=Path, required=True)
    a = ap.parse_args(argv)
    from tools.x2_card import clean_links, quiet_environment, run_cuts, snapshot
    quiet_environment()
    from tools.x2_cutpack import load_pack
    meta, _ = load_pack(a.pack)
    os.environ["OTPU_TARGET"] = meta["target"]
    from opentpu import rtlsim, target as T
    from opentpu.host.board import Board, device_config
    from opentpu.host.cards import CardsTransport, X2SimTransport
    flags = list(rtlsim.VFLAGS)
    flags[flags.index("-j") + 1] = "4"
    rtlsim.VFLAGS = flags + ["--output-groups", "0"]
    baseline = meta["cosim"]
    sim = X2SimTransport(target=meta["target"], ch_bytes=1 << 21, params=baseline["params"],
                         plusargs=baseline["plusargs"], link_lat=baseline["link_lat"])
    ct = CardsTransport(sim.cards, target=T.load(meta["target"]))
    board = Board(ct, lock=False)
    report = {"ok": False, "hardware_measurement": False, "steps": []}

    def record(step, data):
        report["steps"].append({"step": step, **data})
        a.json.write_text(json.dumps(report, indent=2) + "\n")
        if not data.get("prepare"):
            print(json.dumps({"step": step, **data}), flush=True)

    def checked(name, fn, count):
        # Each batched flush is a fresh RTL machine: an out-of-band pre-run register read
        # would flush LOAD before RUN. Board.wait queues its baseline and completion reads
        # in the same simulation, then preserves the successful deltas across later flushes.
        st = fn()
        delta = ct.collective_delta
        snap = snapshot(ct)
        record("e", {"run": name, "delta": delta, "links": snap})
        clean_links(snap)
        if delta != [count] * 2:
            raise RuntimeError(f"{name}: sequence delta {delta} != {[count] * 2}")
        return st

    try:
        run_cuts(board, ct, device_config(board.info()), [a.pack], 1, checked, record, lambda: None)
        got = report["steps"][-1]["cycles"][0]
        if got != baseline["cycles"]:
            raise RuntimeError(f"pack cycles {got} != co-sim cycles {baseline['cycles']}")
        report["ok"] = True
        print(f"PASS pack RTL replay: {got} cycles", flush=True)
    except Exception as e:
        report["error"] = " ".join(str(e).splitlines())
        raise
    finally:
        board.close()
        a.json.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
