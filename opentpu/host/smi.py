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

VERSION = "0.1.0"
ROOT = Path(__file__).resolve().parents[2]
POWER_JSON = ROOT / "build" / "vivado" / "reports" / "power.json"
UTIL_SHOW = [("RUNNING", "RUN"), ("MXU_BUSY", "MXU"), ("MXU_MAC", "MAC"), ("VPU_BUSY", "VPU"),
             ("QNT_BUSY", "QNT"), ("DMA_BUSY", "DMA"), ("TMEM_DENY", "TMEM-deny"),
             ("DRAM_WAIT", "DRAM-wait")]
W = 88                                      # table width


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
def _mib(n) -> str:
    return "n/a" if n is None else f"{n / 2**20:,.0f}"


def _row(s: str = "") -> str:
    return "| " + s[:W - 4].ljust(W - 4) + " |"


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def table(devs: list[dict]) -> str:
    rule = "+" + "-" * (W - 2) + "+"
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out = [f"otpu-smi {VERSION}".ljust(W - len(now)) + now, rule]
    for d in devs:
        if not d["ok"]:
            out += [_row(f"{d['device']}   link {d['link']}"), rule]
            continue
        bs = d["bitstream"]
        mhz = f"{bs['core_mhz']:.1f} MHz" if bs["core_mhz"] else "clock n/a"
        bid = f"build {bs['build_id']:08x}" if bs["build_id"] is not None else "build n/a"
        out.append(_row(f"{d['device']:<16} openTPU D={bs['D']} MCOLS={bs['MCOLS']} "
                        f"LANES={bs['LANES']}  {mhz}  {bid}  regmap v{d['regmap']}"))
        c0, c1 = d["calib"]
        link = d["link"] + (f" ({d['pcie']})" if d.get("pcie") else "")
        temp = "n/a" if d["temp_c"] is None else f"{d['temp_c']:.1f} C"
        out.append(_row(f"link {link}   DDR3 calib ch0 {'ok' if c0 else 'NO'} "
                        f"ch1 {'ok' if c1 else 'NO'}   temp {temp}   "
                        f"{'running' if d.get('running') else 'idle'}"))
        pw = d.get("power")
        pws = f"{pw['w']:.2f} W est." if pw else "n/a"
        dr = d.get("dram")
        drs = "n/a"
        if dr:
            drs = f"{_mib(dr['total'] - dr['free'])} / {_mib(dr['total'])} MiB"
            if dr.get("kv_capacity"):
                drs += f" (KV {_mib(dr['kv_used'])} / {_mib(dr['kv_capacity'])})"
        bw = "n/a" if d.get("dram_gbs") is None else f"{d['dram_gbs']:.2f} GB/s"
        out.append(_row(f"power {pws}   DRAM {drs}   DRAM bw {bw}"))
        out.append(rule)
        u = d.get("util")
        if u:
            smp = d["sample"]
            win = f"{smp['seconds'] * 1e3:.0f} ms" if smp["seconds"] else f"{smp['cycles']} cycles"
            cell = [f"{lbl} {_pct(u[k])}" for k, lbl in UTIL_SHOW]
            out.append(_row("util  " + "  ".join(cell[:6])))
            out.append(_row("stall " + "  ".join(cell[6:]) + f"   IPC {smp['ipc']:.3f}   "
                            f"over {win}"))
        else:
            out.append(_row("util  n/a (register map 1 bitstream: no free-running counters)"))
        if d.get("run"):
            r = d["run"]
            out.append(_row(f"run   {r['program']}: {r['cycles']} cycles, "
                            f"{r['instructions']}/{r['of']} instructions"))
        p = d.get("process")
        if p and not p.get("stale"):
            argv = p.get("argv") or ["?"]
            cmd = " ".join([Path(argv[0]).name] + argv[1:])
            tps = []
            if p.get("tok_s_wall"):
                tps.append(f"{p['tok_s_wall']:.2f} tok/s wall")
            if p.get("tok_s_device"):
                tps.append(f"device {p['tok_s_device']:.2f} tok/s")
            out.append(_row(f"pid {p['pid']}  {cmd}"))
            out.append(_row(f"      model {p.get('model') or '?'}   tokens {p.get('tokens', 0)}   "
                            + ("   ".join(tps) or "tok/s n/a")))
        elif p:
            out.append(_row(f"no process (stale status file from pid {p['pid']})"))
        else:
            out.append(_row("no process"))
        out.append(rule)
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
