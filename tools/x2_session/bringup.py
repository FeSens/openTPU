#!/usr/bin/env python3
"""Two-card (bcu1525-x2) session adapter around a tree's unmodified wg_bringup.py
(boards/bcu1525/scripts): `python3 tools/x2_session/bringup.py [--tree T] load|up ...` takes
wg_bringup's own arguments (--serial, --dev, --out, ...), one card at a time.

The reference bring-up guard normally has an emergency image fallback. This session instead
stops at 80 C and leaves the loaded image in place. The x2 image also supports local collectives
at X2_CARD=0 | (1 << 8): its two-slice children need the same platform-7 target with S=2,
not platform 6 (bcu1525-s2) or the four-slice machine target. No target file is edited.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent
X2_ROOT = HERE.parents[1]


def local_target():
    from opentpu import target as T
    t = T.load("bcu1525-x2:st16")
    return replace(t, arch=dict(t.arch, S=2), platform=dict(t.platform, cards=1))


def child(argv):
    from opentpu.host.board import XdmaTransport
    from opentpu.host import regs as R
    init = XdmaTransport.__init__

    def solo(self, *args, **kw):
        init(self, *args, **kw)
        self.want_target = local_target()
        self.reg_write(R.R_X2_CARD, 1 << 8)
    XdmaTransport.__init__ = solo
    module, *args = argv
    sys.argv = [module, *args]
    runpy.run_module(module, run_name="__main__")


def known_link(out: Path, link: str) -> bool:
    """`up`'s rc 1 is the slot's known PCIe link only: the memory diag's one FAIL is its platform
    "PCIe link" row at exactly `link` (a card in a slot that trains below the card's own link on every image), the
    bring-up reached its end with the accelerator up, the self-test and runs passed, and the
    guard neither tripped nor booted."""
    try:
        r = json.loads((out / "wg.json").read_text())
        d = json.loads((out / "diag.json").read_text())
    except (OSError, ValueError):
        return False
    if "status_end" not in r or "tripped" in r or "stopped" in r or r.get("diag_rc") != 1 \
            or r.get("selftest_rc", 0) or any(r.get("run_rc", [])) \
            or not r.get("acc", {}).get("ok") \
            or r.get("guard", {}).get("tripped") or r.get("guard", {}).get("booted"):
        return False
    bad = [x for x in d.get("rows", []) if x.get("status") == "FAIL"]
    return d.get("failed") == 1 and len(bad) == 1 and \
        (bad[0]["section"], bad[0]["name"]) == ("platform", "PCIe link") and \
        bad[0]["msg"].startswith(f"{link} (")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree", type=Path, default=X2_ROOT)
    ap.add_argument("--known-link", help='up: accept a diag whose only FAIL is the PCIe link at '
                    'exactly this speed and width, e.g. "5.0 GT/s PCIe x4" (a slot narrower than the card)')
    ap.add_argument("args", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(a.tree))
    if a.args[0] == "child":
        return child(a.args[1:])
    spec = importlib.util.spec_from_file_location("session_wg", a.tree /
                                                "boards/bcu1525/scripts/wg_bringup.py")
    wg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wg)
    # Load our guard by filename even when the production host tree predates tools.x2_card.
    spec = importlib.util.spec_from_file_location("session_qual", HERE / "safety.py")
    qual = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qual)
    wg.G.Guard = qual.SessionGuard
    readings = wg.G.die_readings

    def die(*args, **kw):
        r = readings(*args, **kw)
        if any((r.get(k) or -273) >= 80 for k in ("gov_c", "jtag_c")):
            raise wg.Tripped(f"FPGA temperature >= 80 C: {r}")
        return r
    wg.G.die_readings = die
    # Check JTAG temperature before a load as well as after. A hot card is never loaded.
    if a.args[0] == "load":
        from opentpu.host.jtagaxi import JtagAxi
        mon = JtagAxi.mon

        def cool(self):
            r = mon(self)
            if r["t"] >= 80:
                raise wg.Tripped(f"FPGA temperature {r['t']} C >= 80 C")
            return r
        JtagAxi.mon = cool
    if a.args[0] == "up" and a.tree.resolve() == X2_ROOT:
        run_child = wg.child

        def with_local_view(name, args, *rest, **kw):
            return run_child(name, ["tools.x2_session.bringup", "child", *args],
                             *rest, **kw)
        wg.child = with_local_view
    rc = wg.main(a.args)
    if a.args[0] == "up":
        out = Path(a.args[a.args.index("--out") + 1])
        r = json.loads((out / "wg.json").read_text())
        if r.get("guard", {}).get("tripped") or r.get("guard", {}).get("booted"):
            return 1
        if rc == 1 and a.known_link and known_link(out, a.known_link):
            print(f"known slot link accepted: the diag's only FAIL is PCIe {a.known_link}",
                  flush=True)
            return 0
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
