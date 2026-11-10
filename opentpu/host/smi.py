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
the first query scans the buses, later ones reuse <run dir>/<dev>.i2c.json). On the BCU1525
(the BMC's UART bridge, CAPS2.bmc) it is the VCCINT regulator's input power, read through the
board management controller (opentpu/host/bmc.py: read-only, under the BMC lock), and a VCCINT
row gives the regulator's output, input and temperatures. Otherwise it is the estimate from
Vivado's power report, marked "~ ... estimate". --no-i2c skips the buses, --no-bmc the BMC.

The BCU1525 (a governor at BAR0 0x10000: docs/bcu1525.md sections 13 and 14) gets its board's
rows, from registers alone, nothing written there: the governor (RUN or SAFE, the DIMMs and
the accelerator it runs, the heartbeat, the trips), the governor's die, the VCCINT regulator's
input power and VIN with the reading's age and its temperatures, from the BMC bridge's watch (a
bridge with the watch, "BMC2", is never sent a request: a session would hold the line the
governor's regulator reading needs). The SYSMON's own registers are read only when the governor's
die is gone (not valid, or the same for gov.FROZEN_S: "gov frozen", by the governor's TEMP_AGE
on a build with it, else over -l's samples; then the die alone, one read a sample) or with
--sysmon (all of them, the die's supplies too): an AXI read of them can stop the wizard's
temperature bus, the governor's die (gov.py). With the accelerator off (SAFE: its control
registers read 0xDEAD0FF0) those rows are the table, the core clock from the BMC UART's divider.
A card of a two-card machine (CAPS2 bit15, bcu1525-x2) adds its link port's rows: its place
(card 0 or 1 of 2, once a host has set it), and each QSFP28 link: up or down, the peer its
hello named, the sticky errors standing and the last ping. Each card is a device of its own
here; the machine's state is both tables.
Each reading is what it names: READ_PIN is VCCINT's regulator input, not the card's total (no
DIMMs, transceivers, aux rails or fan). otpu-bmc reads each page's output through a session.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import glob
import json
import sys
import textwrap
import time
from pathlib import Path

from . import bmc as B
from . import gov as G
from . import power as P
from . import regs as R
from .board import Board, counter_units, rates
from .runstate import devname, read_status

VERSION = "0.4.2"
FENCED = 0xDEAD0FF0             # otpu_axil_off's read data while the accelerator is off (BCU1525)
_clock = time.monotonic         # bcu_state's (how long the governor's die has read the same)
ROOT = Path(__file__).resolve().parents[2]
POWER_JSON = ROOT / "build" / "vivado" / "reports" / "power.json"


def find_devices() -> list[str]:
    return sorted(p[:-len("_user")] for p in glob.glob("/dev/xdma*_user"))


def pcie_link(dev: str, which: str = "current") -> str | None:
    """'2.5 GT/s PCIe x8' from sysfs (the XDMA driver's class device), None if unknown; which
    "max": the device's own (the link it would train to in a slot that has it)."""
    base = Path(f"/sys/class/xdma/{devname(dev)}_user/device")
    try:
        sp = (base / f"{which}_link_speed").read_text().strip()
        wd = (base / f"{which}_link_width").read_text().strip()
    except OSError:
        return None
    return f"{sp} x{wd}"


# ------------------------------------------------------------------------------ sampling
def query(t, dev: str, interval: float = 0.2, prev: dict | None = None,
          power_json=POWER_JSON, sleep=time.sleep, i2c: bool = True, bmc: bool = True,
          sysmon: bool = False, track: dict | None = None) -> dict:
    """One device's state. Utilization over `interval` seconds (two snapshots), or since
    `prev` (the "counters" of an earlier query) when given. `sysmon` and `track`: bcu_state's."""
    b = Board(t, check=False, lock=False)
    ident = t.reg_read(R.R_ID)
    d = {"device": dev, "time": time.time(), "id": ident, "ok": ident == R.ID_OTPU,
         "link": "ok" if ident == R.ID_OTPU else
         "down (registers read 0xffffffff)" if ident == 0xFFFFFFFF else
         f"no openTPU (ID {ident:#010x})", "pcie": pcie_link(dev)}
    if not d["ok"]:
        bcu = bcu_state(t, bmc, sysmon, track) if ident == FENCED else None
        if bcu is not None:                     # a BCU1525 image in SAFE: the board's side
            st = read_status(devname(dev))
            d.update(link="accelerator off (its registers fenced)", bcu=bcu,
                     pcie_max=pcie_link(dev, "max"), process=st,
                     measured=watch_power(bcu.get("bmc")))
        return d
    i = b.info()
    d.update(regmap=i["regmap"],
             bitstream={"D": i["D"], "MCOLS": i["MCOLS"], "LANES": i["LANES"],
                        "core_mhz": i["core_khz"] / 1e3 if i["core_khz"] else None,
                        "build_id": i["build_id"], "ddr_mts": i["ddr_mts"]},
             calib=i["calib"], status=i["status"], running=i["running"], caps=i["caps"],
             caps2=i["caps2"], temp_c=i["temp_c"])
    tg = i.get("target")
    if b.v2:
        s0 = prev if prev else b.snapshot()
        if not prev:
            sleep(interval)
        s1 = b.snapshot()
        _derive(d, s0, s1, i["core_khz"], _units(tg))
    else:
        d.update(counters=None, util=None, sample=None, dram_gbs=None)
    st = read_status(devname(dev))
    d["process"] = st
    d["dram"] = st["dram"] if st and not st.get("stale") else None
    d["power"] = _power(d, power_json)
    bcu = None
    if (tg is not None and str(tg.platform.get("board", "")).startswith("bcu1525")) or \
            i["caps2"].get("bmc"):
        bcu = bcu_state(t, bmc, sysmon, track)
    if bcu is not None:
        bcu.update(target=tg.name if tg is not None else None, cfg=i["cfg"],
                   mem=str(tg.platform.get("mem", "ddr4")).upper() if tg is not None else "DDR4")
        d.update(bcu=bcu, pcie_max=pcie_link(dev, "max"))
    if i["caps2"].get("xcard"):                 # a card of a two-card machine: its link port
        d["x2"] = x2_state(t)
    bridge = (bcu or {}).get("bmc") if bcu is not None else \
        bmc_bridge(t) if bmc and i["caps2"].get("bmc") else None
    if bridge and bridge["id"] == B.ID_BMC2:
        d["measured"] = watch_power(bridge)     # the watch's registers: no session on its line
    else:
        d["measured"] = _measured(t, i["caps"] if i2c else None, i["caps2"] if bmc else None)
    return d


def x2_state(t) -> dict:
    """A card's link port (CAPS2 bit15: tensor parallel across cards, docs/x2_link.md section
    5), from registers alone, nothing written (the sticky errors go only with X2_CTRL's clear,
    the host's at its link reset): "card" and "ncards" (X2_CARD: this card's place, 0 of 1
    until a host sets it), "seq" (the collectives issued since the link reset), "coll" (the
    collective's causes standing, by name) and per QSFP28 port "ports": up, the sticky errors
    by name, CRC errors counted, the peer its hello named ("peer": card, port; None without a
    hello) and the last ping's round trip in core cycles (None before one)."""
    offs = [R.R_X2_CARD, R.R_X2_SEQ, R.R_COLL_STATUS, *R.R_X2_STATUS, *R.R_X2_HELLO,
            *R.R_X2_CRC, *R.R_X2_PING]
    topo, seq, coll, *v = [t.reg_read(o) for o in offs]
    names = ((R.X2_CRC, "CRC"), (R.X2_WATCHDOG, "watchdog"), (R.X2_ABORT, "abort"),
             (R.X2_DROPPED, "dropped"))
    ports = []
    for p in range(2):
        st, h, crc, ping = v[p], v[2 + p], v[4 + p], v[6 + p]
        ports.append({"up": bool(st & R.X2_UP), "status": st,
                      "errors": [nm for bit, nm in names if st & bit], "crc": crc,
                      "peer": {"card": h & 0xFF, "port": h >> 8 & 0xF}
                      if h >> 24 == R.X2_HELLO_MAGIC else None,
                      "ping_cycles": ping & 0x7FFFFFFF if ping >> 31 else None})
    return {"card": topo & 0xFF, "ncards": topo >> 8 & 0xFF, "seq": seq & 0xFFFF,
            "coll": [nm for k, nm in enumerate(R.COLL_CAUSES) if coll >> k & 1],
            "ports": ports}


def x2_rows(x: dict) -> list[str]:
    """The link port's rows (x2_state): the card's place in the machine, then each port."""
    def port(p, q):
        peer = f"card {q['peer']['card']} port {q['peer']['port']}" if q["peer"] else "no hello"
        bad = ", ".join(q["errors"] + ([f"{q['crc']} CRC errors"] if q["crc"] else []))
        ping = f", ping {q['ping_cycles']} cycles" if q["ping_cycles"] is not None else ""
        return f"QSFP{p} {'up' if q['up'] else 'DOWN'} ({peer}{ping})" + (f" ! {bad}" if bad else "")
    # (0 of 1 is the image's reset value: a two-card image no host has brought up since it loaded)
    place = "not set yet (a bring-up sets them)" if x["ncards"] <= 1 \
        else f"card {x['card']} of {x['ncards']}"
    coll = f"  collective: {', '.join(x['coll'])}" if x["coll"] else ""
    out = [_line(f"{'Cards':<11}{place}  collectives {x['seq']}{coll}")]
    out += [_line(" " * 11 + port(p, q)) for p, q in enumerate(x["ports"])]
    return out


def _units(tg) -> tuple[int, int]:
    """The bytes a DRAM_RD and a DRAM_WR count stand for on the bitstream's target
    (board.counter_units: half beats with NAT_RHB / NAT_WHB); the DDR3 cards' 64 unknown."""
    if tg is None:
        return R.DRAM_BEAT, R.DRAM_BEAT
    try:
        return counter_units(tg)
    except KeyError:
        return R.DRAM_BEAT, R.DRAM_BEAT


def bcu_state(t, bmc: bool = True, sysmon: bool = False, track: dict | None = None,
              clock=None) -> dict | None:
    """A BCU1525 image's board side, from registers alone (none written; not the BMC bridge's RX
    or STATUS, whose reads take a byte or clear flags): the governor ("gov": gov.Gov's status and
    limits, the heartbeat's age); how long its die has read the same ("gov_same_s": the
    governor's own TEMP_AGE on a build with it, else over the queries that shared `track`; None
    without either) and whether it is gone ("gov_frozen": 'frozen 75 s' once the governor says
    stale or gov.FROZEN_S passes, 'not valid', else None); the SYSMON's own registers ("sysmon":
    gov.sysmon, all 13 with `sysmon`, else the die alone while the governor's is gone, else None:
    a read of them can stop the governor's die, gov.py); and, with `bmc`, the BMC bridge ("bmc":
    its ID, the core clock from its divider, the watch's readings: bmc.watch, None without the
    watch; None itself without `bmc`, absent without a bridge). None without a governor at
    gov.R_GOV."""
    g = G.Gov(t)
    if not g.present():
        return None
    gs = {**g.status(), **g.limits(), "heartbeat_ms": g.heartbeat_ms()}
    same = None
    if gs["temp_age_ms"] is not None:           # the governor counts it (TEMP_AGE)
        same = gs["temp_age_ms"] / 1e3
    elif track is not None:
        now, c = (clock or _clock)(), gs["temp_c"] if gs["temp_valid"] else None
        if "since" not in track or track.get("c") != c:
            track.update(c=c, since=now)
        same = now - track["since"]
    frozen = (f"frozen {same:.0f} s" if gs["temp_stale"] else
              "not valid" if not gs["temp_valid"] else
              f"frozen {same:.0f} s" if same is not None and same >= G.FROZEN_S else None)
    out = {"gov": gs, "gov_same_s": same, "gov_frozen": frozen,
           "sysmon": G.sysmon(t) if sysmon else G.sysmon(t, full=False) if frozen else None}
    if not bmc:
        out["bmc"] = None                       # (--no-bmc: the bridge left alone)
    elif (br := bmc_bridge(t)) is not None:
        out["bmc"] = br
    return out


def die_now(bc: dict) -> tuple[float | None, str]:
    """The die the board's rows go by: the SYSMON's register when it was read (the sensor's own
    reading), else the governor's while it is not gone; (None, '') with neither."""
    sm, g = bc.get("sysmon") or {}, bc["gov"]
    if sm.get("die_c") is not None:
        return sm["die_c"], "SYSMON"
    if g["temp_valid"] and not bc.get("gov_frozen"):
        return g["temp_c"], "governor"
    return None, ""


def bmc_bridge(t) -> dict | None:
    """The BMC's UART bridge at R_BMC, registers only (ID, DIV and the watch's: never RX or
    STATUS): {"id", "name" ("BMC1", "BMC2": with the watch), "core_mhz" (its clock: DIV cycles a
    bit at bmc.BAUD), "watch" (bmc.watch; None on BMC1)}; None without one."""
    u = B.Uart(t)
    ident = u.rd(B.ID)
    if ident not in (B.ID_BMC, B.ID_BMC2):
        return None
    div = u.rd(B.DIV) & 0xFFFFFF
    return {"id": ident, "name": "BMC2" if ident == B.ID_BMC2 else "BMC1",
            "core_mhz": div * B.BAUD / 1e6 if div else None,
            "watch": B.watch(t) if ident == B.ID_BMC2 else None}


def watch_power(bridge: dict) -> dict | None:
    """otpu-smi's measurement (i2c.measured's shape) from the BMC watch's READ_PIN (`bridge`:
    bmc_bridge's): VCCINT's regulator input power, with its VIN and the readings' ages (ms); None
    without the watch or before its first reading (or once it is 65.5 s old: the register's age
    saturates there)."""
    w = (bridge or {}).get("watch")
    if not w or w.get("pin") is None:
        return None
    ages = w["ages_ms"]
    return {"w": w["pin"], "how": "PIN", "label": "VCCINT regulator input", "source": "watch",
            "age_ms": ages["pin"], "vin": w["vin"],
            "vin_age_ms": ages["vin"] if w["vin"] is not None else None,
            "rails": [{"bus": "bmc", "addr": B.ADDR, "w": w["pin"], "how": "PIN"}],
            "board_temp_c": None}


def _measured(t, caps: dict | None, caps2: dict | None = None) -> dict | None:
    """VCCINT's power and temperatures through the BMC (CAPS2.bmc), else board power (PMBus)
    and temperature (LM73) over I2C (CAPS.i2c); None without either, or when it fails."""
    try:
        if caps2 and caps2.get("bmc"):
            from . import bmc as BMC
            return BMC.measured(t)
        if caps and caps.get("i2c"):
            from . import i2c as I2C
            return I2C.measured(t)
    except OSError:
        pass
    return None


def _derive(d: dict, s0: dict, s1: dict, core_khz: int | None,
            units: tuple[int, int] = (R.DRAM_BEAT, R.DRAM_BEAT)) -> None:
    """Utilization and DRAM GB/s between two snapshots; `units` the bytes a DRAM_RD and a
    DRAM_WR count stand for (_units)."""
    r = rates(s0, s1, core_khz)
    d["counters"] = s1
    d["sample"] = {"cycles": r["cycles"], "seconds": r["seconds"], "ipc": r["ipc"],
                   "dram_beats": r["dram_beats"]}
    d["util"] = r["util"]
    d["util"]["DRAM"] = r["dram_beats"] / r["cycles"] if r["cycles"] else 0.0
    d["dram_gbs"] = r["dram_gbs"]
    d["dram_rd_gbs"], d["dram_wr_gbs"] = r["dram_rd_gbs"], r["dram_wr_gbs"]
    if r["seconds"] and tuple(units) != (R.DRAM_BEAT, R.DRAM_BEAT):
        rd = (s1["DRAM_RD"] - s0["DRAM_RD"]) * units[0] / r["seconds"] / 1e9
        wr = (s1["DRAM_WR"] - s0["DRAM_WR"]) * units[1] / r["seconds"] / 1e9
        d["dram_rd_gbs"], d["dram_wr_gbs"], d["dram_gbs"] = rd, wr, rd + wr


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
        b.load(PROG_AT, prog)                               # queued, same flush
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
              ("temp", "temp"), ("bmc", "bmc"))
_SPARK = "▁▂▃▄▅▆▇█"
CSV_FIELDS = ["time", "iso", "device", "ok", "running", "dram_gbs", "dram_rd_gbs",
              "dram_wr_gbs", "temp_c", "ipc"] + [k for k, _ in UNITS] + [
              "DRAM", "MXU_STARVE", "TMEM_DENY", "DRAM_WAIT",
              "tok_s_device", "tok_s_wall", "tokens", "build_id", "power_w", "power_how",
              # the BCU1525's (empty elsewhere): the power reading's age, the regulator's input
              # voltage, the SYSMON register's die (when read), the regulator's hottest, the
              # governor's state; its die (the wizard's temperature bus) and that die gone (1)
              "power_age_s", "vin", "die_sysmon_c", "vreg_c", "governor",
              "die_gov_c", "gov_frozen"]


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
    ms = d.get("measured") or {}
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
           "build_id": f"{bid:08x}" if isinstance(bid, int) else "",
           "power_w": _csv_num(ms.get("w")), "power_how": ms.get("label") or ms.get("how") or ""}
    bc = d.get("bcu") or {}
    w = (bc.get("bmc") or {}).get("watch") or {}
    age = ms.get("age_ms")
    row.update(power_age_s=_csv_num(age / 1e3 if age is not None else None),
               vin=_csv_num(ms.get("vin")),
               die_sysmon_c=_csv_num((bc.get("sysmon") or {}).get("die_c")),
               vreg_c=_csv_num(w.get("hot_c")),
               governor=("RUN" if bc["gov"]["run"] else "SAFE") if bc else "",
               die_gov_c=_csv_num(bc["gov"]["temp_c"] if bc and bc["gov"]["temp_valid"]
                                  else None),
               gov_frozen=("1" if bc.get("gov_frozen") else "0") if bc else "")
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
        hist = (history or {}).get(d["device"]) or []
        if not d["ok"]:
            if d.get("bcu"):                    # a BCU1525 image in SAFE: the board's rows
                bm = d["bcu"].get("bmc") or {}
                mhz = (f"core ~{bm['core_mhz']:.1f} MHz (BMC UART DIV)" if bm.get("core_mhz")
                       else "core clock n/a")
                fw = (bm.get("watch") or {}).get("firmware")
                out.append(_kv("Bitstream", "n/a (accelerator off)", mhz,
                               f"BMC fw {fw:#x}" if fw else "", widths=(24, 32)))
                out.append(_kv("Link", _gen_max(d.get("pcie"), d.get("pcie_max")),
                               "accelerator off: its registers fenced"))
                out += bcu_rows(d, hist) + _process_rows(d)
            else:
                out.append(_kv("Link", d["link"]))
            out.append("╰" + "─" * (W - 2) + "╯")
            continue
        bs, u = d["bitstream"], d.get("util")
        c0, c1 = d["calib"]
        bc = d.get("bcu")
        bid = f"build {bs['build_id']:08x}" if bs["build_id"] is not None else "build n/a"
        mhz = f"{bs['core_mhz']:.0f} MHz" if bs["core_mhz"] else "clock n/a"
        out.append(_kv("Bitstream", f"D={bs['D']} MCOLS={bs['MCOLS']} LANES={bs['LANES']}",
                       f"{bid}   {mhz}", f"regmap v{d['regmap']}"))
        if bc and bc.get("cfg"):
            out.append(_kv("Config", cfg_text(bc)))
        caps = d.get("caps") and {**d["caps"], "bmc": (d.get("caps2") or {}).get("bmc")}
        for i, row in enumerate(caps_rows(caps)):
            out.append(_kv("Caps" if i == 0 else "", row))
        if bc:                                  # the DIMMs the governor runs, the SYSMON's die
            dimms = " ".join(f"C{x}" for x in bc["gov"]["dimms_on"]) or "none"
            mem = f"{bc.get('mem') or 'DDR4'}-{bs['ddr_mts']}" if bs.get("ddr_mts") else \
                bc.get("mem") or "DDR4"
            out.append(_kv("Link", _gen_max(d.get("pcie"), d.get("pcie_max")),
                           f"{mem} {dimms} on",
                           "Die n/a" if die_now(bc)[0] is None else
                           f"Die {die_now(bc)[0]:.1f} °C"))
        else:
            temp = "n/a" if d["temp_c"] is None else f"{d['temp_c']:.0f} °C"
            out.append(_kv("Link", _gen(d.get("pcie")),
                           f"{ddr_name(bs.get('ddr_mts'))} ch0 {'ok' if c0 else 'FAIL'}  "
                           f"ch1 {'ok' if c1 else 'FAIL'}",
                           f"Temp {temp}"))
        pw, ms = d.get("power"), d.get("measured") or {}
        if bc:                                  # (the power is bcu_rows' Power row)
            out += bcu_rows(d, hist)
            out.append(_kv("State", "Running" if d.get("running") else "Idle"))
        else:
            if ms.get("w") is not None:
                ptxt = f"Power {ms['w']:.1f}W {ms.get('label') or 'measured'}"
            else:
                ptxt = f"Power ~{pw['w']:.1f}W estimate" if pw else "Power n/a"
            bt = ms.get("board_temp_c")
            out.append(_kv("State", "Running" if d.get("running") else "Idle", ptxt,
                           f"Board {bt:.0f} °C" if bt is not None else ""))
        if d.get("x2"):                         # a card of a two-card machine: its link port
            out += x2_rows(d["x2"])
        if ms.get("vccint"):
            out.append(vccint_row(ms["vccint"]))
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
        out += _process_rows(d)
        out.append("╰" + "─" * (W - 2) + "╯")
    return "\n".join(out)


def _process_rows(d: dict) -> list[str]:
    """The Process section: the runner's status file (runstate), or why there is none."""
    out = [_head("Process")]
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
    return out


def _gen_max(pcie: str | None, pmax: str | None) -> str:
    """_gen's link, and the device's own where the slot trained it lower: 'PCIe Gen2 x4 (of
    Gen3 x16)'."""
    cur = _gen(pcie)
    if pcie and pmax and pmax != pcie:
        return f"{cur} (of {_gen(pmax)[5:]})"
    return cur


def cfg_text(bc: dict) -> str:
    """The configuration registers a BCU1525 bitstream reports (CFG_*): its target, slices,
    channels, beat, weight groups and CFG_HASH."""
    c = bc["cfg"]
    return (f"{bc.get('target') or 'platform ' + str(c['platform'])}  S={c['S']}  "
            f"{c['channels']} ch x {c['channel_bytes'] >> 20} MiB  beat {c['beat_bytes']} B  "
            f"WG {c['WG']}  CFG_HASH {c['hash']:08x}")


def _span(xs) -> str:
    """'2.8-2.9' (the lowest and highest of the readings there are), 'n/a' with none."""
    v = [x for x in xs if x is not None]
    return f"{min(v):.1f}-{max(v):.1f}" if v else "n/a"


def _dur(s: float) -> str:
    """'12.3 s', '4m05s', '1h02m'."""
    if s < 60:
        return f"{s:.1f} s"
    m, s = divmod(int(s), 60)
    return f"{m}m{s:02d}s" if m < 60 else f"{m // 60}h{m % 60:02d}m"


def bcu_rows(d: dict, hist: list | None = None) -> list[str]:
    """The BCU1525's board rows (d["bcu"]: bcu_state): the governor's state; the die, as the
    governor reads it (the wizard's temperature bus) and, when read, the SYSMON's register, with
    the governor's limits; VCCINT's regulator input power, VIN and the reading's age (the BMC
    watch, or a BMC1 bridge's session); the regulator's temperatures and limits; with --sysmon
    the die's supplies; with `hist` (-l) the power's and the die's trend; then a note for each
    alarm standing (the governor's STATUS: the SYSMON's ALM0 and OT), what would refuse GO in
    SAFE, the governor's die gone (gov frozen), and with --sysmon a governor die GOV_GAP_C off
    the register's."""
    bc = d["bcu"]
    g, sm = bc["gov"], bc.get("sysmon")
    full = bool(sm and sm.get("full"))
    w = (bc.get("bmc") or {}).get("watch")
    out, notes = [], []
    row = lambda name, *cells: _line(f"{name:<11}" + "  ".join(c for c in cells if c))  # noqa
    f1 = lambda x: "n/a" if x is None else f"{x:.1f}"                                   # noqa
    if g["run"]:
        hb = g["heartbeat_ms"]
        out.append(row("Governor", f"RUN {_dur(g['run_ms'] / 1e3)}",
                       "DIMMs " + (" ".join(f"C{x}" for x in g["dimms_on"]) or "none"),
                       f"accelerator {'on' if g['acc_on'] else 'off'}",
                       "heartbeat " + (f"{hb / 1e3:.1f} s ago" if hb < 0xFFFF else "none")))
    else:
        why = ", ".join(g["causes"])
        out.append(row("Governor", "SAFE" + (f" ({why})" if why else ""),
                       f"trips {g['trips_thermal']} thermal, {g['trips_heartbeat']} heartbeat",
                       f"last RUN {_dur(g['run_ms'] / 1e3)}" if g["run_ms"] else ""))
    fz = bc.get("gov_frozen")
    gtxt = f"{g['temp_c']:.1f} °C" if g["temp_valid"] or g.get("temp_stale") else "n/a"
    if sm is None:                          # (the register unread: the governor's die alone)
        cells = [f"die {gtxt} (governor)"]
    else:
        cells = ["die n/a (no SYSMON)" if sm["die_c"] is None else
                 f"die {sm['die_c']:.1f} °C SYSMON" + (f" (max {f1(sm['max_c'])})" if full
                                                         else ""),
                 f"gov {gtxt}" + (" NOT VALID" if fz == "not valid" else
                                  f" FROZEN {bc['gov_same_s']:.0f} s" if fz else "")]
    out.append(row("Thermal", *cells, f"trip {g['trip_c']:.1f}  resume {g['resume_c']:.1f}"))
    ms = d.get("measured") or {}
    if ms.get("w") is not None and ms.get("source") == "watch":
        age = ms["age_ms"] / 1e3                # (older than the governor's REG_STALE: STALE)
        out.append(row("Power", f"{ms['w']:.1f} W VCCINT regulator input (not board total)",
                       f"VIN {ms['vin']:.2f} V" if ms.get("vin") is not None else "VIN n/a",
                       f"{age:.0f} s STALE" if ms["age_ms"] >= g.get("reg_stale_ms", 10000)
                       else f"{age:.1f} s old"))
    elif ms.get("w") is not None:
        out.append(row("Power", f"{ms['w']:.1f} W VCCINT regulator input (not board total)",
                       "BMC session"))
    else:
        pw = d.get("power")
        why = ("the watch stopped" if w and w.get("stopped") else
               "no reading yet" if w else "--no-bmc" if bc.get("bmc", 0) is None else
               "no watch" if bc.get("bmc") else "no BMC bridge")
        out.append(row("Power", f"~{pw['w']:.1f} W estimate (Vivado)" if pw else f"n/a ({why})"))
    if w:
        out.append(row("VCCINT VR", f"pages {f1(w['p0_c'])} / {f1(w['p1_c'])} °C",
                       f"controller {f1(w['ctl_c'])} °C",
                       f"trip {f1(g.get('reg_trip_c'))}  resume {f1(g.get('reg_resume_c'))}",
                       f"{w['bad']} failed" if w.get("bad") else ""))
    if full and sm["valid"]:
        f3 = lambda x: "n/a" if x is None else f"{x:.3f}"                               # noqa
        out.append(row("Rails", f"VCCINT {f3(sm['vccint'])} V", f"VCCAUX {f3(sm['vccaux'])} V",
                       f"VBRAM {f3(sm['vbram'])} V", "(SYSMON, at the die)"))
    pts = [h for h in (hist or []) if "_power_w" in h]
    if len(pts) >= 2:
        pws, die = [h["_power_w"] for h in pts], [h["_die_c"] for h in pts]
        out.append(row("Trend", f"VCCINT in {sparkline(pws, 12)} {_span(pws)} W",
                       f"die {sparkline(die, 12)} {_span(die)} °C"))
    lim = full and sm["valid"]
    if g["alarm"] or lim and sm["alarm"]:
        notes.append("SYSMON temperature alarm" + (f" (set at {sm['alarm_up_c']:.1f}, clears at "
                                                   f"{sm['alarm_lo_c']:.1f} °C)" if lim else "")
                     + (": GO refused until it clears" if not g["run"] else ""))
    if g["ot"] or lim and sm["ot"]:
        notes.append("SYSMON over-temperature alarm" + (f" (at {sm['ot_up_c']:.1f} °C the FPGA "
                                                        f"shuts down)" if lim else ""))
    if not g["run"]:
        waits = ([] if g["cool"] else [f"the governor's die above RESUME {g['resume_c']:.1f}"]) + \
            ([] if g["reg_cool"] else ["the regulator hot or unread"])
        if waits:
            notes.append("GO refused: " + ", ".join(waits))
    if fz:
        notes.append("gov frozen: the governor's die (the wizard's temperature bus) "
                     + ("is not valid" if fz == "not valid" else
                        f"has read {gtxt} for {bc['gov_same_s']:.0f} s")
                     + ("; the die is the SYSMON's register" if (sm or {}).get("die_c") is not None
                        else "; the SYSMON reads nothing either: the die is unknown")
                     + ". The governor's TRIP and WARN still use its own; its alarm and OT "
                       "follow the die")
    elif full and sm["die_c"] is not None and g["temp_valid"] and \
            abs(g["temp_c"] - sm["die_c"]) >= G.GOV_GAP_C:
        gap = g["temp_c"] - sm["die_c"]
        notes.append(f"the governor's die reads {abs(gap):.1f} °C "
                     f"{'below' if gap < 0 else 'above'} the SYSMON's: its bus may have stopped "
                     f"(with -l, gov frozen after {G.FROZEN_S:.0f} s the same)")
    if w and w.get("stopped"):
        notes.append(f"the BMC watch stopped: firmware {w['firmware']:#x} is not one it knows")
    for n in notes:                         # (wrapped at the box: a note is never cut)
        for k, part in enumerate(textwrap.wrap(n, W - 4 - 13)):
            out.append(_line(" " * 11 + ("! " if k == 0 else "  ") + part))
    return out


def vccint_row(v: dict) -> str:
    """VCCINT through the BMC: the pages' output (their mean voltage, summed current and power),
    the hotter page's temperature, the controller's die and the regulator's input voltage."""
    f = lambda x, fmt: "n/a" if x is None else format(x, fmt)          # noqa: E731
    pg = v.get("pages") or []

    def agg(k, fn):
        xs = [p[k] for p in pg if p.get(k) is not None]
        return fn(xs) if xs else None
    vout = agg("vout", lambda xs: sum(xs) / len(xs))
    return _kv("VCCINT", f"{f(vout, '.3f')} V  {f(agg('iout', sum), '.1f')} A  "
               f"{f(agg('pout', sum), '.1f')} W out",
               f"VR {f(agg('temp_c', max), '.0f')} °C  ctrl {f(v.get('die_c'), '.0f')} °C",
               f"VIN {f(v.get('vin'), '.1f')} V")


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
    ap.add_argument("--no-bmc", action="store_true",
                    help="do not read VCCINT through the board management controller (BCU1525)")
    ap.add_argument("--sysmon", action="store_true",
                    help="the BCU1525's SYSMON registers too (die, supplies, limits: 13 reads a "
                         "sample). Each read can freeze the governor's temperature bus for good "
                         "(about 1 in 1000 a read: gov.py); without it the SYSMON's die is read "
                         "only once the governor's is gone (not valid, or frozen with -l)")
    ap.add_argument("--fake", action="store_true",
                    help="an in-memory card with synthetic counters (demo)")
    ap.add_argument("--fake-bcu", nargs="?", const="safe", choices=("safe", "run"),
                    help="an in-memory BCU1525 at WG 8 (fake.FakeBcu; demo): the governor in "
                         "SAFE (default) or RUN with the accelerator on")
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
    elif a.fake_bcu:
        from .fake import FakeBcu
        fk = FakeBcu(run=a.fake_bcu == "run", kick_ms=300 if a.fake_bcu == "run" else 0xFFFF,
                     run_ms=754_000 if a.fake_bcu == "run" else 15_103)
        open_transport = open_transport or (lambda dev: fk)
        devs = a.dev or ["/dev/fakebcu0"]
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
    ts, prev, hist, track = {}, {}, {}, {}
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
                    d = query(t, dev, a.interval, prev.get(dev), a.power_json, i2c=not a.no_i2c,
                              bmc=not a.no_bmc, sysmon=a.sysmon, track=track.setdefault(dev, {}))
                    prev[dev] = d.get("counters")
                except OSError as e:
                    d = {"device": dev, "ok": False, "link": f"cannot open ({e.strerror or e})"}
                    rc = 1
                out.append(d)
            for d in out:
                e = dict(d["util"]) if d.get("ok") and d.get("util") else {}
                if d.get("bcu"):                # the BCU1525's power and die (bcu_rows' Trend)
                    e.update(_power_w=(d.get("measured") or {}).get("w"),
                             _die_c=die_now(d["bcu"])[0])
                if e:
                    h = hist.setdefault(d["device"], [])
                    h.append(e)
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
