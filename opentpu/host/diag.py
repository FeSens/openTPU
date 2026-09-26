"""otpu-diag: the full hardware diagnostic of the openTPU card (or its board model, --sim).

    otpu-diag                             # every check on /dev/xdma0, about a minute
    otpu-diag --mem full                  # plus a march C- over all 4 GiB (both channels)
    otpu-diag --soak 20                   # rerun the kernel set 20 times, count intermittents
    otpu-diag --model qwen3               # plus the model check of otpu-selftest
    otpu-diag --json diag.json            # the report as JSON too
    otpu-diag --only mem,isa              # platform plus some sections
    otpu-diag --sim                       # the board model (memory tests scaled to it)

Unlike otpu-selftest it does not stop at the first failure: it runs every check whose
prerequisites passed, marks the others SKIP (naming the failed prerequisite), and ends with a
works / does-not-work matrix per section and diagnosis hints from the pattern of failures.
The exit code is 1 when any check failed.

Sections: platform (PCIe, driver, ID, configuration, calibration, STATUS errors, temperature,
power estimate), regs (read/write patterns, read-only sanity), mem (per channel: walking 1 / 0
data bits, address bits, random blocks, partial writes, DMA bandwidth; the interleave; --mem
full: march C-), isa (one program per instruction variant, opentpu/host/opchecks.py, bit for
bit against the ISA simulator), system (the demo, masked-write and vops programs, the cycle
counters, --soak), model (--model).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import memtest as M
from . import power as P
from . import regs as R
from .board import CH_BYTES, Board, ConfigMismatch, SimTransport, XdmaTransport, device_config
from .checks import (PROG_AT, masked_program, model_check, partial_writes, pattern_test,
                     run_demo, vops_program)
from .opchecks import diag_image, op_checks

PASS, FAIL, SKIP, INFO = "PASS", "FAIL", "SKIP", "INFO"
SECTIONS = ("platform", "regs", "mem", "isa", "system", "model")
POWER_JSON = Path(__file__).resolve().parents[2] / "build" / "vivado" / "reports" / "power.json"


@dataclass
class Row:
    section: str
    name: str
    status: str
    msg: str
    seconds: float = 0.0
    group: str = ""                 # the ISA group (opchecks) or the memory channel
    data: dict = field(default_factory=dict)


class Diag:
    """Runs checks, records a Row for each and prints a line per check."""

    def __init__(self, out=sys.stdout):
        self.rows: list[Row] = []
        self.out = out

    def status(self, name: str) -> str | None:
        for r in reversed(self.rows):
            if r.name == name:
                return r.status
        return None

    def check(self, section: str, name: str, fn, needs=(), group: str = "") -> Row:
        """fn() -> (ok, msg) or (ok, msg, data); ok True / False, or INFO / SKIP. A check whose
        prerequisite (a row name in `needs`) failed or was skipped is skipped."""
        blocked = [n for n in needs if self.status(n) in (FAIL, SKIP)]
        t0 = time.time()
        if blocked:
            row = Row(section, name, SKIP, f"needs {blocked[0]} ({self.status(blocked[0])})",
                      group=group)
        else:
            try:
                res = fn()
                ok, msg, data = (res + ({},))[:3]
                st = ok if isinstance(ok, str) else PASS if ok else FAIL
                row = Row(section, name, st, msg, time.time() - t0, group, data)
            except Exception as e:                                  # noqa: BLE001
                tb = traceback.format_exc(limit=2).strip().splitlines()[-1]
                row = Row(section, name, FAIL, f"{type(e).__name__}: {e}", time.time() - t0,
                          group, {"traceback": tb})
        self.rows.append(row)
        print(f"  [{row.status}] {section:<8} {name:<44} {row.msg}"
              + (f"  ({row.seconds:.1f}s)" if row.seconds >= 0.1 else ""),
              file=self.out, flush=True)
        return row


# ------------------------------------------------------------------------------ platform
def pcie_link(dev: str) -> tuple:
    """(speed, width) of the card's link from sysfs, or None."""
    base = Path(f"/sys/class/xdma/{Path(dev).name}_user/device")
    try:
        return ((base / "current_link_speed").read_text().strip(),
                (base / "current_link_width").read_text().strip())
    except OSError:
        return None


def driver_state(dev: str) -> tuple[bool, str]:
    nodes = [f"{dev}_{n}" for n in ("user", "h2c_0", "c2h_0")]
    missing = [n for n in nodes if not os.path.exists(n)]
    try:
        loaded = any(line.split()[0] == "xdma" for line in open("/proc/modules"))
    except OSError:
        loaded = None
    mod = {True: "xdma module loaded", False: "xdma module NOT loaded",
           None: "no /proc/modules"}[loaded]
    return not missing, mod + ("; " + ", ".join(nodes) if not missing
                               else f"; missing {', '.join(missing)}")


def _rw(t, off: int, vals: list[int]) -> list[int]:
    """Write each value to a register and read it back (one simulation on the board model)."""
    if getattr(t, "batched", False):
        idx = []
        for v in vals:
            t.reg_write(off, v)
            idx.append(t.queue_read(off))
        t.flush()
        return [t.results[i] for i in idx]
    out = []
    for v in vals:
        t.reg_write(off, v)
        out.append(t.reg_read(off))
    return out


RW_PATTERNS = ([0, 0xFFFFFFFF, 0xA5A5_5A5A, 0x5A5A_A5A5] + [1 << k for k in range(32)]
               + [0xFFFFFFFF ^ (1 << k) for k in range(32)])


def reg_patterns(t, off: int, mask: int = 0xFFFFFFFF) -> tuple:
    vals = [v & mask for v in RW_PATTERNS]
    got = _rw(t, off, vals)
    t.reg_write(off, 0)
    bad = [(v, g) for v, g in zip(vals, got) if g & mask != v]
    stuck1 = stuck0 = 0
    for v, g in bad:
        stuck1 |= g & ~v & mask
        stuck0 |= v & ~g & mask
    if not bad:
        return True, f"{len(vals)} patterns"
    return False, (f"{len(bad)} of {len(vals)} patterns wrong (bits read 1 when written 0: "
                   f"{stuck1:#010x}, read 0 when written 1: {stuck0:#010x})")


# ------------------------------------------------------------------------------ diagnosis
def _groups(rows: list[Row], section: str) -> dict:
    """{group: (passed, failed)} of a section's rows."""
    g = {}
    for r in rows:
        if r.section == section and r.status in (PASS, FAIL):
            p, f = g.get(r.group, (0, 0))
            g[r.group] = (p + (r.status == PASS), f + (r.status == FAIL))
    return g


def diagnose(rows: list[Row]) -> list[str]:
    hints = []
    st = {r.name: r.status for r in rows}
    if st.get("ID register") == FAIL:
        hints.append("no openTPU answers on BAR0: link down, bitstream not loaded, or the "
                     "card was reprogrammed after enumeration (rescan: setup_pcie.sh --rescan)")
    if st.get("PCIe link") == FAIL:
        hints.append("PCIe below Gen1 x8: lane order / GT placement (board.md section 6.3), "
                     "or the slot's width")
    for c in (0, 1):
        if st.get(f"DDR3 calibration channel {c}") == FAIL:
            hints.append(f"channel {c} did not calibrate: MIG pinout / clocking of that channel, "
                         "memory voltage (board.md section 6.2)")
    # memory: byte lanes and address bits, per channel
    for c in (0, 1):
        lanes, bits, addr = np.zeros(8, int), np.zeros(64, int), {}
        for r in rows:
            if r.section == "mem" and r.group == f"ch{c}" and r.data:
                if "lanes" in r.data:
                    lanes += np.array(r.data["lanes"])
                    bits += np.array(r.data["bits"])
                addr.update(r.data.get("addr_bits") or {})
        bad = [i for i in range(8) if lanes[i]]
        if len(bad) == 8:
            hints.append(f"channel {c}: errors on every byte lane -> the whole channel "
                         "(calibration, CK / address / command, or its AXI path), not one lane")
        for i in (bad if len(bad) < 8 else []):
            dq = [b for b in range(8 * i, 8 * i + 8) if bits[b]]
            one = f"; only DQ{dq[0]} (stuck or shorted bit)" if len(dq) == 1 else ""
            hints.append(f"channel {c} byte lane {i} errors -> DQ[{8 * i + 7}:{8 * i}] pinout / "
                         f"calibration of that lane{one}")
        for b, how in sorted(addr.items(), key=lambda x: int(x[0])):
            hints.append(f"channel {c} address bit {b} {how} -> that address line (A / BA "
                         "pinout) or the MIG address width")
    if st.get("interleave patterns") == FAIL and not any(
            "channel" in h and "lane" in h for h in hints):
        hints.append("the channels pass alone but the interleave fails: the host map "
                     "(board.py split / join) against rtl/mem/otpu_axi_dram.sv")
    # instruction set
    g = _groups(rows, "isa")
    fails = {k for k, (p, f) in g.items() if f}
    if g and all(p == 0 for p, f in g.values()):
        hints.append("every program fails: program load / sequencer, core clock or reset, or "
                     "DRAM (see the memory section)")
    elif fails:
        if fails == {"vpu-new"}:
            hints.append("only RDOT / OUTER / LOG2 fail: a bitstream built before ddec900 "
                         "(Qwen3 and LFM2 run; Qwen3.5 does not)")
        if "control" in fails:
            hints.append("control flow fails: the sequencer (LI / ADDI / LOOP registers)")
        if "dma" in fails:
            hints.append("LD / ST fail: the DMA unit or otpu_axi_dram (unaligned and short "
                         "transfers go through its byte shifter and write masks)")
        if "mxu" in fails and g["mxu"][0] == 0 and not fails & {"vpu", "dma"}:
            hints.append("all MXU rows fail but the VPU and DMA pass -> MXU / DSP path")
        elif "mxu" in fails:
            hints.append("some MXU variants fail: the flag's path (ACC / RMAX / ASCALE "
                         "epilogue, ACT RAM addressing)")
        if "quant" in fails and "mxu" not in fails:
            hints.append("QACT / QST fail while MM passes -> the quantizer")
        if "vpu-composite" in fails and "vpu" not in fails:
            hints.append("exp2 / recip / rsqrt fail but the simple VPU functions pass -> the "
                         "composite lanes (VPU_CL)")
        if "vpu" in fails:
            hints.append("simple VPU functions fail -> the VPU lanes or the TMEM ports")
        if "vpu-edge" in fails and not fails & {"vpu", "vpu-composite"}:
            hints.append("only edge values differ: flush-to-zero / inf / NaN handling in the "
                         "RTL against opentpu/fp32.py")
    soak = [r for r in rows if r.name.startswith("soak") and r.status == FAIL]
    if soak:
        hints.append("intermittent failures under repetition: timing margin (WNS), "
                     "temperature, or DRAM calibration drift")
    bw = [r for r in rows if r.name.endswith("DMA bandwidth") and r.status == FAIL]
    if bw:
        hints.append("slow DMA: LnkSta width / speed, the IOMMU (iommu=pt), or the driver "
                     "(poll_mode)")
    return hints


def summary(rows: list[Row], hints: list[str]) -> str:
    lines = ["", "summary", f"  {'section':<10}{'PASS':>6}{'FAIL':>6}{'SKIP':>6}{'INFO':>6}"]
    for s in SECTIONS:
        rs = [r for r in rows if r.section == s]
        if rs:
            n = {k: sum(r.status == k for r in rs) for k in (PASS, FAIL, SKIP, INFO)}
            lines.append(f"  {s:<10}" + "".join(f"{n[k]:>6}" for k in (PASS, FAIL, SKIP, INFO)))
    for label, want in (("does not work", FAIL), ("not run", SKIP)):
        bad = [r for r in rows if r.status == want]
        if bad:
            lines += ["", f"{label}:"] + [f"  {r.section}: {r.name}: {r.msg}" for r in bad]
    if hints:
        lines += ["", "diagnosis:"] + [f"  - {h}" for h in hints]
    fails = sum(r.status == FAIL for r in rows)
    lines += ["", f"{fails} check{'s' * (fails != 1)} failed" if fails else "ALL PASS"]
    return "\n".join(lines)


# ------------------------------------------------------------------------------ the run
def run(a, t, dev: str, sim: bool) -> tuple[list[Row], list[str]]:
    d = Diag()
    want = set(a.only.split(",")) if a.only else set(SECTIONS)
    ch_bytes = len(t.ch[0]) if hasattr(t, "ch") else CH_BYTES
    ctx = {"cfg": None, "board": None, "info": None}
    link, calib = "ID register", ["DDR3 calibration channel 0", "DDR3 calibration channel 1"]
    core = [link, "configuration (VERSION)"] + calib

    # ---- platform
    print("platform", flush=True)
    if not isinstance(t, XdmaTransport):
        d.check("platform", "PCIe link", lambda: (SKIP, "not a PCIe device"))
        d.check("platform", "XDMA driver", lambda: (SKIP, "not a PCIe device"))
    else:
        def pcie():
            lk = pcie_link(dev)
            if lk is None:
                return SKIP, "no sysfs link information (lspci -vv: LnkSta)"
            ok = lk[0].startswith("2.5") and lk[1] == "8"
            return ok, f"{lk[0]} x{lk[1]} (expected 2.5 GT/s x8)"
        d.check("platform", "PCIe link", pcie)
        d.check("platform", "XDMA driver", lambda: driver_state(dev))

    def ident():
        v = t.reg_read(R.R_ID)
        ctx["board"] = Board(t, check=False)
        return v == R.ID_OTPU, f"{v:#010x}" + ("" if v == R.ID_OTPU else
                                               f", want {R.ID_OTPU:#010x}")
    d.check("platform", link, ident)

    def config():
        i = ctx["info"] = ctx["board"].info()
        ctx["cfg"] = device_config(i, DRAM_BYTES=2 * ch_bytes)
        msg = f"D={i['D']} MCOLS={i['MCOLS']} LANES={i['LANES']}, register map {i['regmap']}"
        if i["core_khz"]:
            msg += f", core {i['core_khz'] / 1e3:g} MHz"
        if i["build_id"] is not None:
            msg += f", build {i['build_id']:08x}"
        return True, msg, {"info": {k: v for k, v in i.items() if k != "caps"}}
    d.check("platform", "configuration (VERSION)", config, [link])

    for c, bit in ((0, R.ST_CALIB0), (1, R.ST_CALIB1)):
        def cal(bit=bit):
            deadline = time.time() + (0 if sim else 5)
            while not t.reg_read(R.R_STATUS) & bit and time.time() < deadline:
                time.sleep(0.1)
            ok = bool(t.reg_read(R.R_STATUS) & bit)
            return ok, "calibrated" if ok else "NOT calibrated (STATUS bit " \
                f"{bit.bit_length() - 1})"
        d.check("platform", f"DDR3 calibration channel {c}", cal, [link])

    def status_errors():
        st = t.reg_read(R.R_STATUS)
        err = st & (R.ST_ERROR | R.ST_AXI_ERR)
        if not err:
            return True, f"STATUS {st:#x}: no ERROR / AXI_ERR"
        t.reg_write(R.R_CTRL, R.CTRL_CLEAR)
        t.reg_write(R.R_CTRL, 0)
        st2 = t.reg_read(R.R_STATUS)
        if st2 & (R.ST_ERROR | R.ST_AXI_ERR):
            return False, f"STATUS {st2:#x}: ERROR / AXI_ERR stay set after CLEAR"
        return True, f"STATUS {st:#x} had ERROR / AXI_ERR from an earlier run; CLEAR cleared it"
    d.check("platform", "STATUS error bits", status_errors, [link])

    def temp():
        i = ctx["info"]
        if i["regmap"] < 2 or not i["caps"]["temp"]:
            return SKIP, "no temperature sensor in this bitstream"
        if i["temp_c"] is None:
            return False, "TEMP not valid (the XADC reports through channel 0's MIG)"
        return 0 < i["temp_c"] < 85, f"{i['temp_c']:.1f} C"
    d.check("platform", "die temperature", temp, ["configuration (VERSION)"])

    def power():
        pj = P.load(a.power_json)
        if pj is None:
            return SKIP, f"no {a.power_json} (Vivado's report, copied next to the host)"
        idle = P.estimate(pj, {k: 0.0 for k in R.COUNTERS} | {"DRAM": 0.0})
        return INFO, (f"estimated (not measured): idle {idle['w']:.2f} W, all units busy "
                      f"{pj['total_w']:.2f} W ({pj.get('confidence', '?')} confidence)")
    d.check("platform", "power estimate", power)

    b = ctx["board"]

    # ---- registers
    if "regs" in want:
        print("regs", flush=True)
        for name, off, mask in (("SCRATCH", R.R_SCRATCH, 0xFFFFFFFF),
                                ("PROG_ADDR", R.R_PROG_ADDR, 0xFFFFFFFF),
                                ("PROG_N", R.R_PROG_N, 0xFFFFFFFF)):
            d.check("regs", f"{name} read / write", lambda off=off, mask=mask:
                    reg_patterns(t, off, mask), [link])

        def trace_regs():
            i = ctx["info"]
            if i["regmap"] < 2 or not i["caps"]["trace"]:
                return SKIP, "no trace buffer"
            ok1, m1 = reg_patterns(t, R.R_TRACE_ADDR)
            got = _rw(t, R.R_TRACE_CTRL, [R.TR_ENABLE | R.TR_STOP_WHEN_FULL, 0])
            ok2 = got == [R.TR_ENABLE | R.TR_STOP_WHEN_FULL, 0]
            return ok1 and ok2, f"TRACE_ADDR {m1}; TRACE_CTRL " + (
                "ENABLE / STOP_WHEN_FULL ok" if ok2 else f"read {got}")
        d.check("regs", "TRACE_ADDR / TRACE_CTRL read / write", trace_regs,
                ["configuration (VERSION)"])

        def ro_sane():
            i = ctx["info"]
            v = t.reg_read_many([R.R_ID, R.R_VERSION, R.R_REGMAP, R.R_CAPS, R.R_CORE_KHZ,
                                 R.R_BUILD_ID, R.R_STATUS, 0xFFC])
            bad = []
            if i["MCOLS"] not in (2, 4) or i["LANES"] not in (8, 16):
                bad.append(f"VERSION {v[1]:#x}")
            if i["regmap"] >= 2:
                if v[2] != 2:
                    bad.append(f"REGMAP {v[2]}")
                if not 10_000 <= v[4] <= 300_000:
                    bad.append(f"CORE_KHZ {v[4]}")
                if i["caps"]["trace"] and not 8 <= (v[3] >> 8 & 0xFF) <= 16:
                    bad.append(f"CAPS {v[3]:#x}")
                if v[7] != R.UNMAPPED:
                    bad.append(f"undefined offset 0xFFC reads {v[7]:#x}, want 0xdeadbeef")
            if v[6] & ~0xFF:
                bad.append(f"STATUS reserved bits {v[6]:#x}")
            return not bad, "; ".join(bad) or (
                f"VERSION {v[1]:#x}, REGMAP {v[2]}, CAPS {v[3]:#x}, CORE_KHZ {v[4]}, "
                f"BUILD_ID {v[5]:#010x}")
        d.check("regs", "read-only registers: values", ro_sane, ["configuration (VERSION)"])

        def ro_writes():
            offs = [R.R_ID, R.R_VERSION] + ([R.R_REGMAP, R.R_CAPS, R.R_CORE_KHZ, R.R_BUILD_ID]
                                            if ctx["info"]["regmap"] >= 2 else [])
            before = t.reg_read_many(offs)
            for o, v in zip(offs, before):
                t.reg_write(o, ~v)
            after = t.reg_read_many(offs)
            bad = [f"{o:#x}" for o, x, y in zip(offs, before, after) if x != y]
            return not bad, "writes ignored" if not bad else f"changed by a write: {bad}"
        d.check("regs", "read-only registers: ignore writes", ro_writes,
                ["configuration (VERSION)"])

        def counters():
            if ctx["info"]["regmap"] < 2:
                return SKIP, "register map 1: no free-running counters"
            s0, s1 = snapshots(b, t)
            du = s1["UPTIME"] - s0["UPTIME"]
            over = [k for k in R.COUNTERS if k not in R.EVENTS and k != "UPTIME"
                    and s1[k] - s0[k] > du]
            ok = du > 0 and s1["snaps"] == s0["snaps"] + 1 and not over and all(
                s1[k] >= s0[k] for k in R.COUNTERS)
            return ok, (f"UPTIME +{du}, SNAP {s0['snaps']} -> {s1['snaps']}"
                        + (f"; above UPTIME: {over}" if over else ""))
        d.check("regs", "SNAP and the free-running counters", counters,
                ["configuration (VERSION)"])

    # ---- memory
    if "mem" in want:
        print("mem", flush=True)
        for c in (0, 1):
            need = [link, f"DDR3 calibration channel {c}"]
            g = f"ch{c}"

            def res(r):
                return r["ok"], r["msg"], {k: v for k, v in r.items() if k not in ("ok", "msg")}
            d.check("mem", f"channel {c} data bits (walking 1 / 0)",
                    lambda c=c: res(M.walking_data(t, c, ch_bytes // 4)), need, g)
            d.check("mem", f"channel {c} address bits",
                    lambda c=c: res(M.address_bits(t, c, ch_bytes)), need, g)
            d.check("mem", f"channel {c} random blocks",
                    lambda c=c: res(M.random_blocks(t, c, ch_bytes)), need, g)
            d.check("mem", f"channel {c} partial (byte-strobe) writes",
                    lambda c=c: partial_writes(t, c, base=ch_bytes // 2), need, g)
            if sim:
                d.check("mem", f"channel {c} DMA bandwidth",
                        lambda: (SKIP, "the board model has no DMA timing"), need, g)
            else:
                d.check("mem", f"channel {c} DMA bandwidth",
                        lambda c=c: res(M.bandwidth(t, c, a.bw_mib << 20)), need, g)
        top = 2 * ch_bytes

        def interleave():
            m = 1 << 20
            regions = [(0, m), (m + 12345, 1000), (2 * m + 60, 70), (top // 2 - 4096, 8192),
                       (top - 2 * m, m), (top - 100, 100)]
            return pattern_test(b, regions)
        d.check("mem", "interleave patterns", interleave, [link] + calib)
        if a.mem == "full":
            for c in (0, 1):
                d.check("mem", f"channel {c} march C- (full)",
                        lambda c=c: res(M.march(t, c, ch_bytes, progress=M.progress_line)),
                        [link, f"DDR3 calibration channel {c}"], f"ch{c}")

    # ---- instruction set
    img = diag_image()
    if "isa" in want:
        print("isa", flush=True)
        for group, name, prog in (op_checks(ctx["cfg"]) if ctx["cfg"] else []):
            d.check("isa", name, lambda prog=prog: run_demo(b, ctx["cfg"], prog, img)[:2],
                    core, group)
        if not ctx["cfg"]:
            d.check("isa", "instruction set", lambda: (SKIP, "no configuration"), core)

    # ---- system
    if "system" in want:
        print("system", flush=True)
        d.check("system", "all-units demo", lambda: run_demo(b, ctx["cfg"])[:2], core)
        d.check("system", "masked (partial) DRAM writes",
                lambda: run_demo(b, ctx["cfg"], masked_program())[:2], core)
        d.check("system", "RDOT / OUTER / LOG2 program",
                lambda: run_demo(b, ctx["cfg"], vops_program())[:2], core)
        d.check("system", "cycle counters", lambda: cycle_check(b, t, ctx["info"], sim), core)
        if a.soak:
            d.check("system", f"soak x{a.soak}", lambda: soak(b, ctx["cfg"], img, a.soak),
                    core)

    # ---- model
    if a.model and "model" in want:
        print("model", flush=True)
        d.check("model", f"model {a.model}", lambda: model_check(t, ctx["cfg"], a.model,
                                                                  a.tokens, sim), core)
    return d.rows, diagnose(d.rows)


def snapshots(b, t) -> tuple[dict, dict]:
    """Two counter snapshots some time apart; on the board model inside one simulation (every
    flush starts a fresh machine)."""
    if not getattr(t, "batched", False):
        s0 = b.snapshot()
        time.sleep(0.01)
        return s0, b.snapshot()
    idx = []
    for k in range(2):
        if k:
            t.wait_cycles(1000)
        t.reg_write(R.R_SNAP, 1)
        idx.append([t.queue_read(o) for o in b.SNAP_OFFS])
    t.flush()
    return tuple(b.snap_dict([t.results[i] for i in ix]) for ix in idx)


def cycle_check(b, t, info, sim: bool) -> tuple:
    """A NOP loop of n and 2n iterations: CYCLES > 0, growing with n, at least n; on the card
    also consistent with the wall time at CORE_KHZ, and UPTIME (register map 2) covering it."""
    from opentpu import isa as I
    n = 20_000 if sim else 20_000_000
    cyc, walls, ups = [], [], []
    for k in (n, 2 * n):
        b.load_program(PROG_AT, np.asarray(I.assemble([I.loop(1, k), I.nop(), I.halt()]),
                                           np.uint32))
        s0 = b.snapshot() if not sim else None
        t0 = time.perf_counter()
        st = b.run(timeout=30)
        walls.append(time.perf_counter() - t0)
        s1 = b.snapshot() if not sim else None
        cyc.append(st["cycles"])
        if s0 and s1:
            ups.append(s1["UPTIME"] - s0["UPTIME"])
    bad = []
    if not (cyc[0] >= n and cyc[1] > cyc[0]):
        bad.append(f"cycles {cyc[0]}, {cyc[1]} for {n}, {2 * n} iterations")
    msg = f"{cyc[0]} / {cyc[1]} cycles for {n} / {2 * n} iterations"
    khz = info.get("core_khz") if info else None
    if not sim and khz:
        dev_s = cyc[1] / (khz * 1e3)
        msg += f"; {dev_s * 1e3:.0f} ms at {khz / 1e3:g} MHz vs {walls[1] * 1e3:.0f} ms wall"
        if not 0.5 * walls[1] < dev_s <= walls[1] * 1.02:
            bad.append("device time and wall time disagree (CORE_KHZ, or the clock)")
        if ups and ups[1] < cyc[1]:
            bad.append(f"UPTIME advanced {ups[1]} over a {cyc[1]}-cycle run")
    return not bad, "; ".join(bad) or msg


def soak(b, cfg, img, n: int) -> tuple:
    progs = [(name, p) for _, name, p in op_checks(cfg)] + [("demo", None),
                                                          ("masked", masked_program())]
    fails = {}
    for k in range(n):
        for name, p in progs:
            try:
                ok = run_demo(b, cfg, p, img if p is not None else None)[0]
            except Exception:                                       # noqa: BLE001
                ok = False
            if not ok:
                fails[name] = fails.get(name, 0) + 1
        M.progress_line(f"soak {k + 1}/{n}, {sum(fails.values())} failures")
    if sys.stdout.isatty():
        print()
    msg = f"{n} x {len(progs)} programs"
    if fails:
        msg += ": intermittent " + ", ".join(f"{k} {v}/{n}" for k, v in fails.items())
    return not fails, msg, {"fails": fails}


def main(argv=None, open_transport=None) -> int:
    ap = argparse.ArgumentParser(prog="otpu-diag", description=__doc__.split("\n")[0])
    ap.add_argument("--sim", action="store_true", help="the Verilator board model")
    ap.add_argument("--dev", default="/dev/xdma0")
    ap.add_argument("--mem", choices=["quick", "full"], default="quick",
                    help="full: also a march C- over every byte of both channels")
    ap.add_argument("--soak", type=int, default=0, metavar="N",
                    help="rerun the kernel set N times, counting intermittent failures")
    ap.add_argument("--model", help="qwen3, lfm2, qwen35 or a checkpoint directory")
    ap.add_argument("--tokens", type=int, default=8, help="tokens to generate (--model)")
    ap.add_argument("--only", help="comma-separated sections besides platform: "
                    + ",".join(SECTIONS[1:]))
    ap.add_argument("--bw-mib", type=int, default=256, help="bandwidth test size per channel")
    ap.add_argument("--power-json", default=str(POWER_JSON))
    ap.add_argument("--json", metavar="PATH", help="also write the report as JSON")
    a = ap.parse_args(argv)
    if a.only and not set(a.only.split(",")) <= set(SECTIONS):
        ap.error(f"--only takes {','.join(SECTIONS)}")

    dev = "sim (tb_board)" if a.sim else a.dev
    print(f"otpu-diag on {'the board model' if a.sim else a.dev}", flush=True)
    try:
        if open_transport:
            t = open_transport(a.dev)
        elif a.sim:
            t = SimTransport(ch_bytes=1 << 22)
        else:
            t = XdmaTransport(a.dev)
    except OSError as e:
        rows = [Row("platform", "XDMA driver", FAIL, driver_state(a.dev)[1]),
                Row("platform", "ID register", FAIL, f"cannot open {a.dev}: {e}")]
        for r in rows:
            print(f"  [{r.status}] {r.section:<8} {r.name:<44} {r.msg}")
        hints = diagnose(rows) + ["no device nodes: load the XDMA driver "
                                  "(opentpu/host/setup_pcie.sh; docs/host.md)"]
    else:
        try:
            rows, hints = run(a, t, dev, a.sim)
        except ConfigMismatch as e:                  # raised outside a check: not expected
            print(f"otpu-diag: {e}", file=sys.stderr)
            return 1
    print(summary(rows, hints))
    if a.json:
        rep = {"tool": "otpu-diag", "device": dev,
               "time": _dt.datetime.now().isoformat(timespec="seconds"),
               "argv": sys.argv[1:] if argv is None else list(argv),
               "rows": [asdict(r) for r in rows], "hints": hints,
               "failed": sum(r.status == FAIL for r in rows)}
        Path(a.json).write_text(json.dumps(rep, indent=1, default=lambda o: o.item()
                                           if hasattr(o, "item") else str(o)))
        print(f"wrote {a.json}")
    return 1 if any(r.status == FAIL for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
