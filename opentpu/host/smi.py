"""otpu-smi: the state of openTPU cards, like nvidia-smi.

    otpu-smi                      one table per device (/dev/xdma*): bitstream, link, DDR3
                                  calibration, temperature, estimated power, DRAM, utilization
                                  over a short interval, the owning process
    otpu-smi -l 1                 again every second (utilization over each second)
    otpu-smi --json               the same as JSON (a list, one object per device)
    otpu-smi -q                   every detail, raw counters included
    otpu-smi --dev /dev/xdma1     one device (repeatable)
    otpu-smi --sim                the Verilator board model: counters over one run of the
                                  bring-up demo program (both samples in one simulation)
    otpu-smi --fake               an in-memory card with synthetic counters (demo, tests)

A monitor: it only reads registers (and writes SNAP, which latches the free-running counters
into their shadows without disturbing anything), never takes the device lock, and reads the
runner's status file for the process, the model, DRAM use and tokens/s. Utilization is the
counter delta between two SNAPs over the UPTIME delta. On a register map 1 bitstream the
counters, the temperature and the power estimate are n/a.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import sys
import time
from pathlib import Path

from . import power as P
from . import regs as R
from .board import Board, rates
from .runstate import devname, read_status

VERSION = "0.2.0"
ROOT = Path(__file__).resolve().parents[2]
POWER_JSON = ROOT / "build" / "vivado" / "reports" / "power.json"


def find_devices() -> list[str]:
    return sorted(p[:-len("_user")] for p in glob.glob("/dev/xdma*_user"))


def pcie_link(dev: str) -> str | None:
    """'2.5 GT/s PCIe x8' from sysfs (the XDMA driver's class device), None if unknown."""
    base = Path(f"/sys/class/xdma/{devname(dev)}_user/device")
    try:
        sp = (base / "current_link_speed").read_text().strip()
        wd = (base / "current_link_width").read_text().strip()
    except OSError:
        return None
    return f"{sp} x{wd}"


# ------------------------------------------------------------------------------ sampling
def query(t, dev: str, interval: float = 0.2, prev: dict | None = None,
          power_json=POWER_JSON, sleep=time.sleep) -> dict:
    """One device's state. Utilization over `interval` seconds (two snapshots), or since
    `prev` (the "counters" of an earlier query) when given."""
    b = Board(t, check=False, lock=False)
    ident = t.reg_read(R.R_ID)
    d = {"device": dev, "time": time.time(), "id": ident, "ok": ident == R.ID_OTPU,
         "link": "ok" if ident == R.ID_OTPU else
         "down (registers read 0xffffffff)" if ident == 0xFFFFFFFF else
         f"no openTPU (ID {ident:#010x})", "pcie": pcie_link(dev)}
    if not d["ok"]:
        return d
    i = b.info()
    d.update(regmap=i["regmap"],
             bitstream={"D": i["D"], "MCOLS": i["MCOLS"], "LANES": i["LANES"],
                        "core_mhz": i["core_khz"] / 1e3 if i["core_khz"] else None,
                        "build_id": i["build_id"]},
             calib=i["calib"], status=i["status"], running=i["running"], caps=i["caps"],
             temp_c=i["temp_c"])
    if b.v2:
        s0 = prev if prev else b.snapshot()
        if not prev:
            sleep(interval)
        s1 = b.snapshot()
        _derive(d, s0, s1, i["core_khz"])
    else:
        d.update(counters=None, util=None, sample=None, dram_gbs=None)
    st = read_status(devname(dev))
    d["process"] = st
    d["dram"] = st["dram"] if st and not st.get("stale") else None
    d["power"] = _power(d, power_json)
    return d


def _derive(d: dict, s0: dict, s1: dict, core_khz: int | None) -> None:
    r = rates(s0, s1, core_khz)
    d["counters"] = s1
    d["sample"] = {"cycles": r["cycles"], "seconds": r["seconds"], "ipc": r["ipc"],
                   "dram_beats": r["dram_beats"]}
    d["util"] = r["util"]
    d["util"]["DRAM"] = r["dram_beats"] / r["cycles"] if r["cycles"] else 0.0
    d["dram_gbs"] = r["dram_gbs"]
    d["dram_rd_gbs"], d["dram_wr_gbs"] = r["dram_rd_gbs"], r["dram_wr_gbs"]


def _power(d: dict, power_json) -> dict | None:
    pj = P.load(power_json) if power_json else None
    if pj is None or not d.get("util"):
        return None
    e = P.estimate(pj, d["util"])
    e["source"] = str(power_json)
    return e


def query_sim(interval_cycles: int = 0) -> dict:
    """The board model: the bring-up demo program (every unit) runs between the two samples,
    all in one simulation (SimTransport resets the machine per flush). interval_cycles > 0
    samples an idle window of that many cycles instead (the testbench's C command)."""
    import numpy as np
    from opentpu import isa as I
    from .board import SimTransport
    from .checks import PROG_AT, demo_image, demo_program
    t = SimTransport(ch_bytes=1 << 22)
    b = Board(t, check=False, lock=False)
    dev = "sim (tb_board)"
    ident = t.reg_read(R.R_ID)
    d = {"device": dev, "time": time.time(), "id": ident, "ok": ident == R.ID_OTPU,
         "link": "ok (model)" if ident == R.ID_OTPU else f"no openTPU (ID {ident:#010x})",
         "pcie": None}
    i = b.info()
    d.update(regmap=i["regmap"],
             bitstream={"D": i["D"], "MCOLS": i["MCOLS"], "LANES": i["LANES"],
                        "core_mhz": i["core_khz"] / 1e3 if i["core_khz"] else None,
                        "build_id": i["build_id"]},
             calib=i["calib"], status=i["status"], running=False, caps=i["caps"],
             temp_c=i["temp_c"], process=None, dram=None, power=None)
    prog = demo_program()
    if not interval_cycles:
        b.write(0, demo_image())
        b.load_program(PROG_AT, np.asarray(I.assemble(prog), np.uint32))   # queued, same flush
    snap = []
    if b.v2:
        t.reg_write(R.R_SNAP, 1)
        snap.append([t.queue_read(o) for o in Board.SNAP_OFFS])
    if interval_cycles:
        t.wait_cycles(interval_cycles)
    else:
        t.reg_write(R.R_CTRL, R.CTRL_CLEAR)
        t.reg_write(R.R_CTRL, R.CTRL_RUN)
        t.poll(R.R_STATUS, R.ST_HALTED, R.ST_HALTED)
    if b.v2:
        t.reg_write(R.R_SNAP, 1)
        snap.append([t.queue_read(o) for o in Board.SNAP_OFFS])
    run = [t.queue_read(o) for o in (R.R_CYCLES, R.R_CYCLES_HI, R.R_ICOUNT)]
    t.flush()
    if not interval_cycles:
        v = [t.results[k] for k in run]
        d["run"] = {"program": "checks.demo_program", "cycles": v[0] | v[1] << 32,
                    "instructions": v[2], "of": len(prog)}
    if b.v2:
        s0, s1 = (Board.snap_dict([t.results[k] for k in ix]) for ix in snap)
        _derive(d, s0, s1, i["core_khz"])
    else:
        d.update(counters=None, util=None, sample=None, dram_gbs=None)
    return d


# ------------------------------------------------------------------------------ output
COLS = (36, 24, 29)                         # the device box's three columns
W = sum(COLS) + len(COLS) + 1               # table width (91)
UNITS = [("RUNNING", "RUN"), ("MXU_BUSY", "MXU"), ("MXU_MAC", "MAC"), ("VPU_BUSY", "VPU"),
         ("QNT_BUSY", "QNT"), ("DMA_BUSY", "DMA")]
STALLS = [("TMEM_DENY", "TMEM-deny"), ("DRAM_WAIT", "DRAM-wait")]


def _mib(n) -> str:
    return "n/a" if n is None else f"{n / 2**20:,.0f}MiB"


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def _lr(left: str, right: str, w: int) -> str:
    """left-aligned and right-aligned text in one cell of inner width w."""
    return left + right.rjust(max(w - len(left), len(right) + 1))


def _cells(cells, widths=COLS) -> str:
    return "|" + "|".join(" " + str(c)[:w - 2].ljust(w - 2) + " "
                          for c, w in zip(cells, widths)) + "|"


def _rule(ch: str = "-", widths=COLS, edge: str = "+", join: str = "+") -> str:
    return edge + join.join(ch * w for w in widths) + edge


def _full(text: str = "") -> str:
    return _cells([text], (W - 2,))


def _gen(pcie: str | None) -> str:
    """'2.5 GT/s PCIe x8' -> 'Gen1 x8'."""
    if not pcie:
        return "n/a"
    gen = {"2.5": "Gen1", "5.0": "Gen2", "5": "Gen2", "8.0": "Gen3", "8": "Gen3"}
    sp, _, wd = pcie.partition(" x")
    return f"{gen.get(sp.split()[0], sp)} x{wd}" if wd else pcie


def bus_id(dev: str) -> str:
    p = Path(f"/sys/class/xdma/{devname(dev)}_user/device")
    return p.resolve().name if p.exists() else "n/a"


def table(devs: list[dict]) -> str:
    now = _dt.datetime.now().strftime("%a %b %d %H:%M:%S %Y")
    w1, w2, w3 = (w - 2 for w in COLS)
    out = [now, _rule("-", (W - 2,)), _full(_lr(f"OTPU-SMI {VERSION}", "openTPU on PCIe (XDMA)",
                                                  W - 4))]
    out += [_rule(), _cells([_lr("Dev  Name", "", w1), _lr("Bus-Id", "Link", w2),
                             _lr("DDR3 calib", "Temp  Power", w3)]),
            _cells([_lr("Build     Clock    Regmap", "State", w1), _lr("DRAM-Usage", "", w2),
                    _lr("DRAM-BW", "MXU-Util", w3)]),
            _rule("=", edge="|")]
    for n, d in enumerate(devs):
        dev = f"{n:>3}  "
        if not d["ok"]:
            out += [_cells([dev + d["device"], bus_id(d["device"]), "link " + d["link"]]), _rule()]
            continue
        bs, u = d["bitstream"], d.get("util")
        c0, c1 = d["calib"]
        temp = "n/a" if d["temp_c"] is None else f"{d['temp_c']:.0f}C"
        pw = d.get("power")
        pws = f"{pw['w']:.1f}W" if pw else "n/a"
        out.append(_cells([dev + f"openTPU D={bs['D']} MCOLS={bs['MCOLS']} LANES={bs['LANES']}",
                           _lr(bus_id(d["device"]), _gen(d.get("pcie")), w2),
                           _lr(f"ch0 {'ok' if c0 else 'NO'}  ch1 {'ok' if c1 else 'NO'}",
                               f"{temp}  {pws:>5}", w3)]))
        bid = f"{bs['build_id']:08x}" if bs["build_id"] is not None else "n/a"
        mhz = f"{bs['core_mhz']:.0f} MHz" if bs["core_mhz"] else "n/a"
        dr = d.get("dram")
        drs = f"{_mib(dr['total'] - dr['free'])} / {_mib(dr['total'])}" if dr else "n/a"
        bw = "n/a" if d.get("dram_gbs") is None else f"{d['dram_gbs']:.2f} GB/s"
        out.append(_cells([_lr(f"{bid}  {mhz:<8} v{d['regmap']}",
                               "Running" if d.get("running") else "Idle", w1),
                           _lr(drs, "", w2),
                           _lr(bw, _pct(u["MXU_BUSY"]) if u else "n/a", w3)]))
        if dr and dr.get("kv_capacity"):
            out.append(_cells(["", f"KV {_mib(dr['kv_used'])} / {_mib(dr['kv_capacity'])}", ""]))
        out.append(_rule())

    # ---- utilization
    out += ["", _rule("-", (W - 2,))]
    hdr = f"{'Dev':>3}  " + "".join(f"{lbl:>6}" for _, lbl in UNITS) + "   " + \
        "".join(f"{lbl:>10}" for _, lbl in STALLS) + f"{'IPC':>7}{'RD/WR GB/s':>14}"
    out += [_full("Utilization (counters over the sample window)"), _full(hdr),
            _rule("=", (W - 2,), edge="|")]
    for n, d in enumerate(devs):
        u = d.get("util") if d["ok"] else None
        if not u:
            out.append(_full(f"{n:>3}  n/a" + ("" if not d["ok"] else
                             " (register map 1 bitstream: no free-running counters)")))
            continue
        smp = d["sample"]
        win = f"{smp['seconds'] * 1e3:.0f} ms" if smp["seconds"] else f"{smp['cycles']} cycles"
        rw = "n/a" if d.get("dram_rd_gbs") is None else \
            f"{d['dram_rd_gbs']:.2f}/{d['dram_wr_gbs']:.2f}"
        out.append(_full(f"{n:>3}  " + "".join(f"{_pct(u[k]):>6}" for k, _ in UNITS) + "   " +
                         "".join(f"{_pct(u[k]):>10}" for k, _ in STALLS) +
                         f"{smp['ipc']:>7.3f}{rw:>14}"))
        out.append(_full(f"     window {win}"))
        if d.get("run"):
            r = d["run"]
            out.append(_full(f"     run {r['program']}: {r['cycles']} cycles, "
                             f"{r['instructions']}/{r['of']} instructions"))
    out.append(_rule("-", (W - 2,)))

    # ---- processes
    out += ["", _rule("-", (W - 2,))]
    ph = f"{'Dev':>3}  {'PID':>7}  {'Process':<30}{'Model':<14}{'Tokens':>7}{'tok/s dev':>11}" \
         f"{'tok/s wall':>11}"
    out += [_full("Processes:"), _full(ph), _rule("=", (W - 2,), edge="|")]
    any_p = False
    for n, d in enumerate(devs):
        p = d.get("process")
        if not p:
            continue
        any_p = True
        if p.get("stale"):
            out.append(_full(f"{n:>3}  {p['pid']:>7}  (exited: stale status file)"))
            continue
        argv = p.get("argv") or ["?"]
        cmd = " ".join([Path(argv[0]).name] + argv[1:])
        cmd = cmd if len(cmd) <= 29 else cmd[:28] + "…"
        f2 = lambda x: f"{x:.2f}" if x else "n/a"          # noqa: E731
        out.append(_full(f"{n:>3}  {p['pid']:>7}  {cmd:<30}{(p.get('model') or '?')[:13]:<14}"
                         f"{p.get('tokens', 0):>7}{f2(p.get('tok_s_device')):>11}"
                         f"{f2(p.get('tok_s_wall')):>11}"))
    if not any_p:
        out.append(_full("  No running processes found"))
    out.append(_rule("-", (W - 2,)))
    return "\n".join(out)


def details(d: dict) -> str:
    """-q: every field, nested dicts flattened."""
    lines = [f"==== {d['device']} ===="]

    def walk(k, v, ind):
        if isinstance(v, dict):
            lines.append("  " * ind + f"{k}:")
            for kk, vv in v.items():
                walk(kk, vv, ind + 1)
        else:
            if isinstance(v, float):
                v = f"{v:.6g}"
            elif k in ("build_id", "id", "status") and isinstance(v, int):
                v = f"{v:#010x}"
            lines.append("  " * ind + f"{k:<16} {v}")
    for k, v in d.items():
        if k != "device":
            walk(k, v, 1)
    return "\n".join(lines)


def _jsonable(d):
    return json.loads(json.dumps(d, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


# ------------------------------------------------------------------------------ CLI
def main(argv=None, open_transport=None) -> int:
    ap = argparse.ArgumentParser(prog="otpu-smi", description="openTPU card status: bitstream, "
                                 "link, temperature, power (estimated), DRAM, utilization, "
                                 "process.")
    ap.add_argument("--dev", action="append", help="XDMA device prefix, e.g. /dev/xdma0 "
                    "(repeatable; default: every /dev/xdma*_user)")
    ap.add_argument("-l", "--loop", type=float, metavar="SEC",
                    help="repeat every SEC seconds (utilization over each period)")
    ap.add_argument("--json", action="store_true", help="JSON output")
    ap.add_argument("-q", "--query", action="store_true", help="every detail")
    ap.add_argument("-i", "--interval", type=float, default=0.2,
                    help="seconds between the two counter samples (default 0.2)")
    ap.add_argument("--power-json", default=str(POWER_JSON),
                    help="Vivado power summary for the estimate (default: "
                         "build/vivado/reports/power.json in the repository)")
    ap.add_argument("--sim", action="store_true", help="the Verilator board model")
    ap.add_argument("--sim-idle", type=int, metavar="CYCLES", default=0,
                    help="--sim: sample an idle window of CYCLES instead of a program run")
    ap.add_argument("--fake", action="store_true",
                    help="an in-memory card with synthetic counters (demo)")
    a = ap.parse_args(argv)

    def emit(devs):
        if a.json:
            print(json.dumps(_jsonable(devs), indent=1))
        elif a.query:
            print("\n\n".join(details(d) for d in devs))
        else:
            print(table(devs))
        sys.stdout.flush()

    if a.sim:
        emit([query_sim(a.sim_idle)])
        return 0
    if a.fake:
        from .fake import FakeTransport
        fk = FakeTransport()
        open_transport = open_transport or (lambda dev: fk)
        devs = a.dev or ["/dev/fake0"]
    else:
        devs = a.dev or find_devices()
    if not devs:
        print("otpu-smi: no openTPU device (/dev/xdma*_user): is the XDMA driver loaded? "
              "(docs/host.md; --sim for the board model)", file=sys.stderr)
        return 1
    if open_transport is None:
        from .board import XdmaTransport
        open_transport = lambda dev: XdmaTransport(dev, dma=False)   # noqa: E731
    try:
        return _loop(a, devs, open_transport, emit)
    except KeyboardInterrupt:
        return 0


def _loop(a, devs, open_transport, emit) -> int:
    ts, prev = {}, {}
    rc = 0
    while True:
        out = []
        for dev in devs:
            try:
                t = ts.get(dev) or ts.setdefault(dev, open_transport(dev))
                d = query(t, dev, a.interval, prev.get(dev), a.power_json)
                prev[dev] = d.get("counters")
            except OSError as e:
                d = {"device": dev, "ok": False, "link": f"cannot open ({e.strerror or e})"}
                rc = 1
            out.append(d)
        emit(out)
        if not a.loop:
            return rc
        time.sleep(a.loop)


if __name__ == "__main__":
    sys.exit(main())
