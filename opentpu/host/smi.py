"""otpu-smi: the state of openTPU cards, like nvidia-smi.

    otpu-smi                      one table per device (/dev/xdma*): bitstream, link, DDR3
                                  calibration, temperature, power (measured or estimated),
                                  DRAM, utilization over a short interval, the owning process
    otpu-smi -l 1                 again every second (utilization over each second)
    otpu-smi --json               the same as JSON (a list, one object per device)
    otpu-smi -q                   every detail, raw counters included
    otpu-smi --dev /dev/xdma1     one device (repeatable)
    otpu-smi --sim                the Verilator board model: counters over one run of the
                                  bring-up demo program (both samples in one simulation)
    otpu-smi --fake               an in-memory card with synthetic counters (demo, tests)
    otpu-smi -l 0.5 --samples 20 --csv util.csv
                                  loop, stop after 20 samples, append each to a CSV

The default table names the bitstream's capability bits (CAPS) and, in -l mode, a sparkline
of MAC / DRAM / RUN over the last samples.

A monitor: it only reads registers (and writes SNAP, which latches the free-running counters
into their shadows without disturbing anything), never takes the device lock, and reads the
runner's status file for the process, the model, DRAM use and tokens/s. Utilization is the
counter delta between two SNAPs over the UPTIME delta. On a register map 1 bitstream the
counters, the temperature and the power estimate are n/a.

Power is measured when the bitstream has the I2C pins (CAPS.i2c) and a PMBus device on the
card's I2C buses reports it (opentpu/host/i2c.py: read-only transactions under the I2C lock;
the first query scans the buses, later ones reuse <run dir>/<dev>.i2c.json). Otherwise it is
the estimate from Vivado's power report, marked "~ ... estimate". --no-i2c skips the buses.
"""
from __future__ import annotations

import argparse
import csv
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

VERSION = "0.3.0"
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
          power_json=POWER_JSON, sleep=time.sleep, i2c: bool = True) -> dict:
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
                        "build_id": i["build_id"], "ddr_mts": i["ddr_mts"]},
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
    d["measured"] = _measured(t, i["caps"]) if i2c else None
    return d


def _measured(t, caps: dict | None) -> dict | None:
    """Board power (PMBus) and temperature (LM73) over I2C, or None without the I2C pins."""
    if not caps or not caps.get("i2c"):
        return None
    from . import i2c as I2C
    try:
        return I2C.measured(t)
    except OSError:
        return None


def _derive(d: dict, s0: dict, s1: dict, core_khz: int | None) -> None:
    r = rates(s0, s1, core_khz)
    d["counters"] = s1
    d["sample"] = {"cycles": r["cycles"], "seconds": r["seconds"], "ipc": r["ipc"],
                   "dram_beats": r["dram_beats"]}
    d["util"] = r["util"]
    d["util"]["DRAM"] = r["dram_beats"] / r["cycles"] if r["cycles"] else 0.0
    d["dram_gbs"] = r["dram_gbs"]
    d["dram_rd_gbs"], d["dram_wr_gbs"] = r["dram_rd_gbs"], r["dram_wr_gbs"]


def power_report(d: dict, power_json=POWER_JSON) -> Path | None:
    """The Vivado power report for the loaded bitstream: power_json when it is not the default,
    else the saved build whose directory names the bitstream's BUILD_ID
    (build/deploy_*_<sha7>/reports/power.json), else the last build's."""
    if power_json and Path(power_json) != POWER_JSON:
        return Path(power_json)
    bid = (d.get("bitstream") or {}).get("build_id")
    if bid is not None:
        for f in sorted(ROOT.glob(f"build/*{bid:08x}"[:-1] + "*/reports/power.json")):
            return f
    return POWER_JSON if POWER_JSON.exists() else None


def _power(d: dict, power_json) -> dict | None:
    power_json = power_report(d, power_json)
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
                        "build_id": i["build_id"], "ddr_mts": i["ddr_mts"]},
             calib=i["calib"], status=i["status"], running=False, caps=i["caps"],
             temp_c=i["temp_c"], process=None, dram=None, power=None, measured=None)
    prog = demo_program()
    if not interval_cycles:
        b.write(0, demo_image())
        b.load_program(PROG_AT, np.asarray(I.assemble(prog), np.uint32))   # queued, same flush
    snap = []
    if b.v2:
        t.reg_write(R.R_SNAP, 1)
        snap.append([t.queue_read(o) for o in Board.snap_offs(i["regmap"])])
    if interval_cycles:
        t.wait_cycles(interval_cycles)
    else:
        t.reg_write(R.R_CTRL, R.CTRL_CLEAR)
        t.reg_write(R.R_CTRL, R.CTRL_RUN)
        t.poll(R.R_STATUS, R.ST_HALTED, R.ST_HALTED)
    if b.v2:
        t.reg_write(R.R_SNAP, 1)
        snap.append([t.queue_read(o) for o in Board.snap_offs(i["regmap"])])
    run = [t.queue_read(o) for o in (R.R_CYCLES, R.R_CYCLES_HI, R.R_ICOUNT)]
    t.flush()
    if not interval_cycles:
        v = [t.results[k] for k in run]
        d["run"] = {"program": "checks.demo_program", "cycles": v[0] | v[1] << 32,
                    "instructions": v[2], "of": len(prog)}
    if b.v2:
        s0, s1 = (Board.snap_dict([t.results[k] for k in ix], i["regmap"]) for ix in snap)
        _derive(d, s0, s1, i["core_khz"])
    else:
        d.update(counters=None, util=None, sample=None, dram_gbs=None)
    return d


# ------------------------------------------------------------------------------ output
W = 88                                      # table width
UNITS = [("RUNNING", "RUN"), ("MXU_BUSY", "MXU"), ("MXU_MAC", "MAC"), ("VPU_BUSY", "VPU"),
         ("QNT_BUSY", "QNT"), ("DMA_BUSY", "DMA")]


def _mib(n) -> str:
    return "n/a" if n is None else f"{n / 2**20:,.0f}"


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def _bar(x: float, n: int = 10) -> str:
    k = max(0, min(n, round(x * n)))
    return "█" * k + "░" * (n - k)


def _line(text: str = "") -> str:
    return "│ " + text[:W - 4].ljust(W - 4) + " │"


def _head(title: str, first: bool = False) -> str:
    l, r = ("╭", "╮") if first else ("├", "┤")
    t = f"─ {title} " if title else ""
    return l + t + "─" * (W - 2 - len(t)) + r


def _kv(*pairs, widths=(32, 29)) -> str:
    """'Label  value' cells in columns: the first label is the row's name (11 wide)."""
    name, *cells = pairs
    out = f"{name:<11}"
    for i, c in enumerate(cells):
        out += c.ljust(widths[i]) if i < len(widths) else c
    return _line(out.rstrip())


def _gen(pcie: str | None) -> str:
    """'2.5 GT/s PCIe x8' -> 'PCIe Gen1 x8'."""
    if not pcie:
        return "PCIe n/a"
    gen = {"2.5": "Gen1", "5.0": "Gen2", "5": "Gen2", "8.0": "Gen3", "8": "Gen3"}
    sp, _, wd = pcie.partition(" x")
    return f"PCIe {gen.get(sp.split()[0], sp)} x{wd}" if wd else pcie


def bus_id(dev: str) -> str:
    p = Path(f"/sys/class/xdma/{devname(dev)}_user/device")
    return p.resolve().name if p.exists() else "n/a"


def ddr_name(mts: int | None) -> str:
    """"DDR3-1066" from the bitstream's DDR_MTS register, "DDR3" when it does not have one."""
    return f"DDR3-{mts}" if mts else "DDR3"


# capability bits shown in the default table (docs/observability.md CAPS); order is the
# features a reader looks for first, not the bit numbers
_CAP_FLAGS = (("w4", "w4"), ("stream", "stream"), ("gen", "gen"), ("args", "args"),
              ("pair", "pair"), ("dstep", "dstep"), ("chash", "chash"), ("waitw", "waitw"),
              ("hostcal", "hostcal"), ("i2c", "i2c"), ("ddr", "ddr"), ("act_rows", "act-rows"),
              ("temp", "temp"))
_SPARK = "▁▂▃▄▅▆▇█"
CSV_FIELDS = ["time", "iso", "device", "ok", "running", "dram_gbs", "dram_rd_gbs",
              "dram_wr_gbs", "temp_c", "ipc"] + [k for k, _ in UNITS] + [
              "DRAM", "MXU_STARVE", "TMEM_DENY", "DRAM_WAIT",
              "tok_s_device", "tok_s_wall", "tokens", "build_id"]


def caps_text(caps: dict | None) -> str:
    """One line of capability names from Board.info()['caps'] / regs.caps()."""
    if not caps:
        return "n/a"
    bits = []
    if caps.get("trace"):
        d = int(caps.get("trace_depth") or 0)
        bits.append(f"trace·{d // 1024}K" if d >= 1024 else (f"trace·{d}" if d else "trace"))
    for key, lab in _CAP_FLAGS:
        if caps.get(key):
            bits.append(lab)
    return "  ".join(bits) if bits else "none"


def caps_rows(caps: dict | None, width: int = W - 15) -> list[str]:
    """caps_text() in lines of at most `width` characters (the table's value column): a
    bitstream with every CAPS bit set does not fit one row."""
    rows = [""]
    for c in caps_text(caps).split("  "):
        if rows[-1] and len(rows[-1]) + 2 + len(c) > width:
            rows.append(c)
        else:
            rows[-1] = f"{rows[-1]}  {c}" if rows[-1] else c
    return rows


def sparkline(values, width: int = 24) -> str:
    """Unicode bars of the last `width` samples. A constant series is a mid-height line."""
    xs = [float(v) for v in values if v is not None][-width:]
    if not xs:
        return ""
    lo, hi = min(xs), max(xs)
    if hi <= lo:
        return "▄" * len(xs)
    n = len(_SPARK) - 1
    return "".join(_SPARK[min(n, int(round((v - lo) / (hi - lo) * n)))] for v in xs)


def _csv_num(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def csv_row(d: dict) -> dict:
    """One CSV record of a query() result (missing util / process: empty cells)."""
    u, smp, p = d.get("util") or {}, d.get("sample") or {}, d.get("process") or {}
    bs = d.get("bitstream") or {}
    ts = d.get("time") or 0.0
    iso = _dt.datetime.fromtimestamp(ts, _dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    bid = bs.get("build_id")
    row = {"time": f"{ts:.3f}", "iso": iso, "device": d.get("device") or "",
           "ok": _csv_num(bool(d.get("ok"))), "running": _csv_num(bool(d.get("running"))),
           "dram_gbs": _csv_num(d.get("dram_gbs")), "dram_rd_gbs": _csv_num(d.get("dram_rd_gbs")),
           "dram_wr_gbs": _csv_num(d.get("dram_wr_gbs")), "temp_c": _csv_num(d.get("temp_c")),
           "ipc": _csv_num(smp.get("ipc")),
           "tok_s_device": _csv_num(p.get("tok_s_device") if p and not p.get("stale") else None),
           "tok_s_wall": _csv_num(p.get("tok_s_wall") if p and not p.get("stale") else None),
           "tokens": _csv_num(p.get("tokens") if p and not p.get("stale") else None),
           "build_id": f"{bid:08x}" if isinstance(bid, int) else ""}
    for k, _ in UNITS:
        row[k] = _csv_num(u.get(k))
    for k in ("DRAM", "MXU_STARVE", "TMEM_DENY", "DRAM_WAIT"):
        row[k] = _csv_num(u.get(k))
    return row


def table(devs: list[dict], history: dict | None = None) -> str:
    now = _dt.datetime.now().strftime("%a %b %d %H:%M:%S %Y")
    out = [_lr(f"otpu-smi {VERSION}", now, W)]
    for n, d in enumerate(devs):
        bid = bus_id(d["device"])
        out.append(_head(f"Device {n} · {d['device']}" + (f" · {bid}" if bid != "n/a" else ""),
                         first=True))
        if not d["ok"]:
            out += [_kv("Link", d["link"]), "╰" + "─" * (W - 2) + "╯"]
            continue
        bs, u = d["bitstream"], d.get("util")
        c0, c1 = d["calib"]
        bid = f"build {bs['build_id']:08x}" if bs["build_id"] is not None else "build n/a"
        mhz = f"{bs['core_mhz']:.0f} MHz" if bs["core_mhz"] else "clock n/a"
        out.append(_kv("Bitstream", f"D={bs['D']} MCOLS={bs['MCOLS']} LANES={bs['LANES']}",
                       f"{bid}   {mhz}", f"regmap v{d['regmap']}"))
        for i, row in enumerate(caps_rows(d.get("caps"))):
            out.append(_kv("Caps" if i == 0 else "", row))
        temp = "n/a" if d["temp_c"] is None else f"{d['temp_c']:.0f} °C"
        out.append(_kv("Link", _gen(d.get("pcie")),
                       f"{ddr_name(bs.get('ddr_mts'))} ch0 {'ok' if c0 else 'FAIL'}  "
                       f"ch1 {'ok' if c1 else 'FAIL'}",
                       f"Temp {temp}"))
        pw, ms = d.get("power"), d.get("measured") or {}
        if ms.get("w") is not None:
            ptxt = f"Power {ms['w']:.1f}W measured"
        else:
            ptxt = f"Power ~{pw['w']:.1f}W estimate" if pw else "Power n/a"
        bt = ms.get("board_temp_c")
        out.append(_kv("State", "Running" if d.get("running") else "Idle", ptxt,
                       f"Board {bt:.0f} °C" if bt is not None else ""))
        dr = d.get("dram")
        if dr:
            used = dr["total"] - dr["free"]
            kv = ""
            if dr.get("kv_capacity"):
                kv = (f"KV {_bar(dr['kv_used'] / dr['kv_capacity'], 8)} "
                      f"{_mib(dr['kv_used'])} / {_mib(dr['kv_capacity'])} MiB")
            out.append(_kv("DRAM", f"{_bar(used / dr['total'], 8)} {_mib(used)} / "
                           f"{_mib(dr['total'])} MiB", kv))
        else:
            out.append(_kv("DRAM", "n/a (no process status)"))
        if d.get("dram_gbs") is not None:
            out.append(_kv("DRAM BW", f"{d['dram_gbs']:.2f} GB/s",
                           f"read {d['dram_rd_gbs']:.2f}  write {d['dram_wr_gbs']:.2f} GB/s"))
        # ---- utilization
        if u:
            smp = d["sample"]
            win = f"{smp['seconds'] * 1e3:.0f} ms" if smp["seconds"] else f"{smp['cycles']} cycles"
            out.append(_head(f"Utilization over {win}"))
            cells = [f"{lbl:<4}{_bar(u[k])} {_pct(u[k]):>4}" for k, lbl in UNITS]
            for i in range(0, len(cells), 3):
                out.append(_line("     ".join(cells[i:i + 3])))
            idle = max(0.0, u["MXU_BUSY"] - u["MXU_MAC"])
            # MXU_STARVE (register map 3): the part of no-MAC spent with no weight chunk
            why = (f"MXU-starve {_pct(u['MXU_STARVE'])}" if "MXU_STARVE" in u
                   else "mostly awaiting weights")
            out.append(_line(f"Stalls  MXU no-MAC {_pct(idle)} ({why})   "
                             f"TMEM-deny {_pct(u['TMEM_DENY'])}   "
                             f"DRAM-req-wait {_pct(u['DRAM_WAIT'])}"))
            out.append(_line(f"IPC     {smp['ipc']:.2e}"))
            hist = (history or {}).get(d["device"]) or []
            if len(hist) >= 2:
                mac = [h.get("MXU_MAC") for h in hist]
                dram = [h.get("DRAM") for h in hist]
                run = [h.get("RUNNING") for h in hist]
                out.append(_line(f"Watch  MAC {sparkline(mac, 16)}  DRAM {sparkline(dram, 16)}  "
                                 f"RUN {sparkline(run, 16)}"))
        else:
            out.append(_head("Utilization"))
            out.append(_line("n/a (register map 1 bitstream: no free-running counters)"))
        if d.get("run"):
            r = d["run"]
            out.append(_line(f"Run  {r['program']}: {r['cycles']} cycles, "
                             f"{r['instructions']}/{r['of']} instructions"))
        # ---- process
        out.append(_head("Process"))
        p = d.get("process")
        if not p:
            out.append(_line("No running process"))
        elif p.get("stale"):
            out.append(_line(f"pid {p['pid']} exited (stale status file)"))
        else:
            argv = p.get("argv") or ["?"]
            cmd = " ".join([Path(argv[0]).name] + argv[1:])
            f2 = lambda x: f"{x:.2f}" if x else "n/a"          # noqa: E731
            out.append(_line(f"PID {p['pid']}   {cmd}"))
            out.append(_line(f"Model {p.get('model') or '?'}   tokens {p.get('tokens', 0)}   "
                             f"{f2(p.get('tok_s_device'))} tok/s device   "
                             f"{f2(p.get('tok_s_wall'))} tok/s wall"))
            if p.get("weights"):
                out.append(_line(f"Weights {p['weights']}"))
        out.append("╰" + "─" * (W - 2) + "╯")
    return "\n".join(out)


def _lr(left: str, right: str, w: int) -> str:
    return left + right.rjust(max(w - len(left), len(right) + 1))


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
                                 "link, temperature, power (measured or estimated), DRAM, "
                                 "utilization, "
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
    ap.add_argument("--no-i2c", action="store_true",
                    help="do not read the board's I2C sensors (measured power, board temperature)")
    ap.add_argument("--fake", action="store_true",
                    help="an in-memory card with synthetic counters (demo)")
    ap.add_argument("--csv", metavar="FILE",
                    help="append one CSV row per sample (header on the first write)")
    ap.add_argument("--samples", type=int, metavar="N",
                    help="stop after N samples (with -l; a CSV log without an infinite loop)")
    a = ap.parse_args(argv)

    def emit(devs, hist=None):
        if a.json:
            print(json.dumps(_jsonable(devs), indent=1))
        elif a.query:
            print("\n\n".join(details(d) for d in devs))
        else:
            print(table(devs, hist))
        sys.stdout.flush()

    if a.sim:
        d = query_sim(a.sim_idle)
        emit([d])
        if a.csv:
            with open(a.csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                w.writeheader()
                w.writerow(csv_row(d))
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
    ts, prev, hist = {}, {}, {}
    rc = 0
    csvf = writer = None
    if a.csv:
        csvf = open(a.csv, "w", newline="")
        writer = csv.DictWriter(csvf, fieldnames=CSV_FIELDS)
        writer.writeheader()
        csvf.flush()
    n = 0
    try:
        while True:
            out = []
            for dev in devs:
                try:
                    t = ts.get(dev) or ts.setdefault(dev, open_transport(dev))
                    d = query(t, dev, a.interval, prev.get(dev), a.power_json, i2c=not a.no_i2c)
                    prev[dev] = d.get("counters")
                except OSError as e:
                    d = {"device": dev, "ok": False, "link": f"cannot open ({e.strerror or e})"}
                    rc = 1
                out.append(d)
            for d in out:
                if d.get("ok") and d.get("util"):
                    h = hist.setdefault(d["device"], [])
                    h.append(d["util"])
                    del h[:-48]
            emit(out, hist)
            if writer:
                for d in out:
                    writer.writerow(csv_row(d))
                csvf.flush()
            n += 1
            if a.samples is not None and n >= a.samples:
                return rc
            if not a.loop:
                return rc
            time.sleep(a.loop)
    finally:
        if csvf:
            csvf.close()


if __name__ == "__main__":
    sys.exit(main())
