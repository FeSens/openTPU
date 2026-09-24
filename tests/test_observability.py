"""Observability on the board model (sim/verilator/tb_board.sv; docs/observability.md): the
version 2 register map, the free-running activity counters and their snapshots, and the
hardware trace -- whose records, decoded by opentpu/hwtrace.py, must give exactly the trace
lines the simulator prints (+trace) for the same run.
"""
import re

import numpy as np
import pytest

from opentpu.host.board import (CTRL_CLEAR, CTRL_RUN, R_B_RD, R_B_WR, R_CTRL, R_CYCLES, R_CYCLES_HI,
                        R_ICOUNT, R_SCRATCH, R_STATUS, R_VERSION, ST_HALTED, Board, SimTransport)
from opentpu.host.checks import PROG_AT, demo_image, demo_program
from opentpu import isa as I
from opentpu.hwtrace import (R_TRACE_ADDR, R_TRACE_COUNT, R_TRACE_CTRL, R_TRACE_DROP,
                             R_TRACE_HI, R_TRACE_LO, TRACE_BUSY, TRACE_CLEAR, TRACE_ENABLE,
                             TRACE_STOP_WHEN_FULL, records_to_trace, ring_order)
from opentpu.isasim import board_config
from opentpu.kernels import attention_decode, mlp
from opentpu.runtime import compile_kernel

from test_kernels import attn_args, mlp_args

R_REGMAP, R_CAPS, R_CORE_KHZ, R_BUILD_ID, R_TEMP, R_SNAP = 0x3C, 0x40, 0x44, 0x48, 0x4C, 0x50
COUNTERS = ["UPTIME", "RUNNING", "MXU_BUSY", "MXU_MAC", "VPU_BUSY", "QNT_BUSY", "DMA_BUSY",
            "TMEM_DENY", "DRAM_RD", "DRAM_WR", "DRAM_WAIT", "INSTR"]
TB_BUILD_ID, TB_TEMP = 0x0B0A4D00, 0xA1A        # tb_board.sv
LINE = re.compile(r"^T\d+ [DSGEUHPQ] ")
CFG = board_config(DRAM_BYTES=1 << 22)


def sim_lines(out: str) -> list[str]:
    return [line for line in out.splitlines() if LINE.match(line)]


def run_traced(t: SimTransport, img, prog: list, trace_ctrl: int = TRACE_ENABLE,
               nrec: int = 0, snap: bool = False) -> dict:
    """One simulation: load and run `prog`, wait for the trace to drain, read the counters and
    the first `nrec` trace records (raw ring order)."""
    b = Board(t, check=False)
    img = np.asarray(img, np.uint8)
    at = max(PROG_AT, -(-len(img) // 4096) * 4096)
    b.write(0, img)
    b.load_program(at, np.asarray(I.assemble(prog), np.uint32))
    t.reg_write(R_TRACE_CTRL, TRACE_CLEAR)
    t.reg_write(R_TRACE_CTRL, trace_ctrl)
    if snap:
        t.reg_write(R_SNAP, 1)
    t.reg_write(R_CTRL, CTRL_CLEAR)
    t.reg_write(R_CTRL, CTRL_RUN)
    t.poll(R_STATUS, ST_HALTED, ST_HALTED)
    t.poll(R_TRACE_CTRL, TRACE_BUSY, 0)
    if snap:
        t.reg_write(R_SNAP, 1)
    t.reg_write(R_TRACE_ADDR, 0)
    head = [R_STATUS, R_CYCLES, R_CYCLES_HI, R_ICOUNT, R_B_RD, R_B_WR, R_TRACE_COUNT,
            R_TRACE_DROP, R_SNAP]
    shadows = [0x100 + 4 * k for k in range(2 * len(COUNTERS))] if snap else []
    v = t.reg_read_many(head + shadows + [R_TRACE_LO, R_TRACE_HI] * nrec)
    st, lo, hi, ic, brd, bwr, cnt, drop, nsnap = v[:len(head)]
    r = {"status": st, "cycles": lo | hi << 32, "icount": ic, "b_rd": brd, "b_wr": bwr,
         "count": cnt, "drop": drop, "snaps": nsnap, "sim": sim_lines(t.out)}
    v = v[len(head):]
    if snap:
        r["counters"] = {n: v[2 * k] | v[2 * k + 1] << 32 for k, n in enumerate(COUNTERS)}
        v = v[len(shadows):]
    r["raw"] = [v[2 * k] | v[2 * k + 1] << 32 for k in range(nrec)]
    t.reg_write(R_CTRL, 0)
    return r


# ------------------------------------------------------------------------------ registers
def test_register_map(have_verilator):
    t = SimTransport(ch_bytes=1 << 20, params={"TRACE_DEPTH": 1024, "PQ_WIN": 256,
                                               "CORE_KHZ": 75294})
    t.reg_write(R_SCRATCH, 0x1234_5678)
    t.reg_write(R_TRACE_CTRL, TRACE_ENABLE | TRACE_STOP_WHEN_FULL)
    t.reg_write(R_TRACE_ADDR, 77)
    ident, ver, regmap, caps, khz, bid, temp, snap, scr, tctl, taddr, bad = t.reg_read_many(
        [0x0, R_VERSION, R_REGMAP, R_CAPS, R_CORE_KHZ, R_BUILD_ID, R_TEMP, R_SNAP, R_SCRATCH,
         R_TRACE_CTRL, R_TRACE_ADDR, 0x0FC])
    assert ident == 0x4F545055
    assert ver == (CFG.D << 16) | (CFG.MCOLS << 8) | CFG.LANES
    assert regmap == 2
    assert caps == (8 << 16) | (10 << 8) | 0b11          # log2 256, log2 1024, trace, temp
    assert khz == 75294 and bid == TB_BUILD_ID
    assert temp == (1 << 31) | TB_TEMP
    assert 44.5 < TB_TEMP * 503.975 / 4096 - 273.15 < 45.5
    assert snap == 0 and scr == 0x1234_5678 and taddr == 77
    assert tctl == TRACE_ENABLE | TRACE_STOP_WHEN_FULL     # CLEAR reads 0, not busy
    assert bad == 0xDEADBEEF


def test_register_map_board_defaults(have_verilator):
    t = SimTransport(ch_bytes=1 << 20)
    caps, khz = t.reg_read_many([R_CAPS, R_CORE_KHZ])
    assert caps == (10 << 16) | (14 << 8) | 0b11          # 1024-cycle windows, 16384 records
    assert khz == 100000


# ------------------------------------------------------------------------------ counters
def test_free_running_counters(have_verilator):
    t = SimTransport(ch_bytes=CFG.DRAM_BYTES // 2, stall=30, seed=3, params={"TRACE_DEPTH": 1024})
    prog = demo_program()
    r = run_traced(t, demo_image(), prog, snap=True)
    c = r["counters"]
    assert r["status"] & ST_HALTED and r["snaps"] == 2
    # the shadows hold the second snapshot: totals since reset (one run, and the program load)
    assert c["UPTIME"] >= c["RUNNING"] >= c["MXU_BUSY"] >= c["MXU_MAC"] > 0
    assert c["RUNNING"] >= max(c["VPU_BUSY"], c["QNT_BUSY"], c["DMA_BUSY"]) > 0
    assert c["INSTR"] == r["icount"] == len(prog)
    assert c["DRAM_RD"] >= 2 * r["b_rd"] > 0


def _deltas(t, prog, img):
    """Counter deltas over one run: snapshot, run, snapshot (one simulation)."""
    b = Board(t, check=False)
    b.write(0, img)
    b.load_program(PROG_AT, np.asarray(I.assemble(prog), np.uint32))
    t.reg_write(R_SNAP, 1)
    t.reg_write(R_CTRL, CTRL_CLEAR)
    t.reg_write(R_CTRL, CTRL_RUN)
    t.poll(R_STATUS, ST_HALTED, ST_HALTED)
    # the first snapshot's shadows are read before the second snapshot, in the same simulation
    shadows = [0x100 + 4 * k for k in range(2 * len(COUNTERS))]
    idx = [t.queue_read(o) for o in shadows]
    t.reg_write(R_SNAP, 1)
    v = t.reg_read_many(shadows + [R_CYCLES, R_ICOUNT, R_B_RD, R_B_WR, R_SNAP])
    first = [t.results[i] for i in idx]
    c0 = {n: first[2 * k] | first[2 * k + 1] << 32 for k, n in enumerate(COUNTERS)}
    c1 = {n: v[2 * k] | v[2 * k + 1] << 32 for k, n in enumerate(COUNTERS)}
    cyc, ic, brd, bwr, nsnap = v[len(shadows):]
    return {n: c1[n] - c0[n] for n in COUNTERS}, cyc, ic, brd, bwr, nsnap


def test_counter_snapshots_bracket_a_run(have_verilator):
    """Deltas between two snapshots around a run: RUNNING is the run's CYCLES, INSTR its ICOUNT,
    and the DRAM beats are the port B requests' (otpu_axi_dram.sv): a read fetches the whole
    chunk (a beat on each channel), an aligned LANES-word write touches one channel's beat."""
    t = SimTransport(ch_bytes=CFG.DRAM_BYTES // 2, stall=30, seed=4, params={"TRACE_DEPTH": 1024})
    prog = [I.ld(0, 0, 4096), I.st(0x20000, 0, 4096), I.ld(0x8000, 4096, 1024),
            I.st(0x30000, 4096, 1024), I.halt()]
    img = np.random.default_rng(5).integers(0, 256, 0x40000).astype(np.uint8)
    d, cyc, ic, brd, bwr, nsnap = _deltas(t, prog, img)
    assert nsnap == 2
    assert d["RUNNING"] == cyc and d["UPTIME"] > cyc
    assert d["INSTR"] == ic == len(prog)
    assert brd == (4096 + 1024) // CFG.LANES and bwr == brd     # a request per LANES words
    assert d["DRAM_RD"] == 2 * brd and d["DRAM_WR"] == bwr
    assert d["DMA_BUSY"] > 0 and d["MXU_BUSY"] == d["MXU_MAC"] == d["VPU_BUSY"] == 0
    assert 0 < d["DRAM_WAIT"] < cyc                      # the AXI model stalls (30%)
    # the demo: every unit, the MXU consumes its 32 chunks
    t = SimTransport(ch_bytes=CFG.DRAM_BYTES // 2, stall=30, seed=3, params={"TRACE_DEPTH": 1024})
    d, cyc, ic, brd, bwr, nsnap = _deltas(t, demo_program(), demo_image())
    assert d["UPTIME"] >= d["RUNNING"] == cyc >= d["MXU_BUSY"] >= d["MXU_MAC"] == 32
    assert min(d["VPU_BUSY"], d["QNT_BUSY"], d["DMA_BUSY"]) > 0
    assert d["INSTR"] == ic == len(demo_program())
    assert d["DRAM_RD"] >= 2 * brd and d["DRAM_WR"] >= bwr
    assert d["TMEM_DENY"] >= 0


# ------------------------------------------------------------------------------ trace
def _kernel(name):
    cfg = board_config(DRAM_BYTES=1 << 23)
    if name == "mlp":
        a, _ = mlp_args(np.random.default_rng(0), M=1, H=512, Fd=1024)
        comp, imgs = compile_kernel(mlp, cfg, **a)
    else:
        a, _ = attn_args(np.random.default_rng(1), Hq=8, Hkv=4, d=128, T=384, cap=384, block=128)
        comp, imgs = compile_kernel(attention_decode, cfg, **a)
    return cfg, imgs[0], comp.programs[0]


@pytest.mark.parametrize("name,bucket", [("mlp", None), ("attention", None), ("mlp", 64),
                                         ("attention", 32)])
def test_trace_matches_simulator(have_verilator, name, bucket):
    """The records, read back through TRACE_ADDR / TRACE_LO / TRACE_HI and decoded, are exactly
    the simulator's trace lines (D S G E U from the sequencer and units, P Q H counters)."""
    cfg, img, prog = _kernel(name)
    depth = 8192
    plus = ["+trace"] + ([f"+bucket={bucket}"] if bucket else [])
    t = SimTransport(ch_bytes=cfg.DRAM_BYTES, stall=20, seed=7, plusargs=plus,   # + program
                     params={"TRACE_DEPTH": depth})
    r = run_traced(t, img, prog, nrec=depth)
    assert r["status"] & ST_HALTED
    assert r["drop"] == 0 and 0 < r["count"] <= depth
    hw = records_to_trace(ring_order(r["raw"], r["count"], depth)).splitlines()
    sim = r["sim"]
    kinds = {line.split()[1] for line in sim}
    assert kinds == set("DSGEUHPQ") if name == "mlp" else kinds >= set("DSEUHPQ")
    assert hw == sim
    assert sum(line.split()[1] == "P" for line in sim) >= r["cycles"] // (bucket or 1024)


def test_trace_long_wait_is_exact(have_verilator):
    """A start more than 65534 cycles after its instruction became ready (S's 16-bit delay field
    saturates): the R record that follows carries the ready cycle."""
    n = 60000
    prog = [I.ld(0, 0, n), I.ld(0, 61000, 64), I.st(0x40000, 61000, 64), I.halt()]
    img = np.random.default_rng(3).integers(0, 256, 4 * n).astype(np.uint8)
    t = SimTransport(ch_bytes=CFG.DRAM_BYTES // 2, stall=92, seed=2,
                     plusargs=["+trace", "+bucket=1000000"], params={"TRACE_DEPTH": 1024})
    r = run_traced(t, img, prog, nrec=64)
    sim = r["sim"]
    assert r["count"] <= 64
    waits = [c - rr for c, rr in ((int(x.split("c=")[1].split()[0]), int(x.split("r=")[1]))
                                  for x in sim if " S " in x)]
    assert max(waits) > 0xFFFF
    kinds = [(rec >> 60) & 0xF for rec in r["raw"][:r["count"]]]
    assert 10 in kinds                                      # an R record
    assert records_to_trace(ring_order(r["raw"], r["count"], 64)).splitlines() == sim


def test_decoder_skips_cut_groups():
    """A group cut by the ring's start (its tail only) or the buffer's end (its head only) is
    skipped; the complete groups decode."""
    e = (4 << 60) | (3 << 52) | 100                             # T0 E c=100 s=3
    p = (6 << 60) | (5 << 32) | 200                             # P, n=5, then 9 V records
    vs = [(11 << 60) | (k << 56) | k for k in range(9)]
    q = (7 << 60) | (5 << 32) | 200
    o = (9 << 60) | (1 << 52) | 7                               # an O record's second half
    d = (1 << 60) | (2 << 52) | (9 << 32) | 300                 # a D without its O records
    recs = [o] + vs[5:] + [e, p] + vs + [q] + vs[:6] + [e, d]
    assert records_to_trace(recs) == (
        "T0 E c=100 s=3\n"
        "T0 P c=200 n=5 bm=0 bd=1 am=2 aq=3 mx=4 fm=5 fq=6 fv=7 fc=8\n"
        "T0 Q c=200 n=5 bs=0 as=1 ms=2 mb=3 ff=4 ld=5\n"
        "T0 E c=100 s=3\n")
    assert records_to_trace([], sid=1) == ""


def test_trace_ring_and_stop_when_full(have_verilator):
    """A ring smaller than the trace keeps the last records (a suffix of the lines); with
    STOP_WHEN_FULL it keeps the first DEPTH records (a prefix)."""
    depth = 256
    for ctrl in (TRACE_ENABLE, TRACE_ENABLE | TRACE_STOP_WHEN_FULL):
        t = SimTransport(ch_bytes=CFG.DRAM_BYTES // 2, stall=30, seed=3,
                         plusargs=["+trace", "+bucket=64"], params={"TRACE_DEPTH": depth})
        r = run_traced(t, demo_image(), demo_program(), trace_ctrl=ctrl, nrec=depth)
        sim = r["sim"]
        hw = records_to_trace(ring_order(r["raw"], r["count"], depth)).splitlines()
        assert r["drop"] == 0 and len(hw) > depth // 10        # up to 10 records a line (P)
        if ctrl & TRACE_STOP_WHEN_FULL:
            assert r["count"] == depth
            assert hw == sim[:len(hw)]
        else:
            assert r["count"] > 2 * depth                  # wrapped
            assert hw == sim[len(sim) - len(hw):]


def test_trace_drops_are_counted(have_verilator):
    """A two-bundle capture queue and a P/Q window every 4 cycles overflow: the lost events are
    counted exactly, and the rest are recorded in order."""
    depth = 4096
    t = SimTransport(ch_bytes=CFG.DRAM_BYTES // 2, stall=30, seed=3,
                     plusargs=["+trace", "+bucket=4"], params={"TRACE_DEPTH": depth, "TRACE_QD": 2})
    r = run_traced(t, demo_image(), demo_program(), nrec=depth)
    sim = r["sim"]
    assert 0 < r["count"] < depth
    hw = records_to_trace(ring_order(r["raw"], r["count"], depth)).splitlines()
    assert r["drop"] > 0 and len(sim) - len(hw) == r["drop"]
    it = iter(sim)
    assert all(line in it for line in hw)                 # a subsequence, in order
    # not recording: nothing is captured, nothing is dropped
    t = SimTransport(ch_bytes=CFG.DRAM_BYTES // 2, stall=30, seed=3,
                     plusargs=["+bucket=4"], params={"TRACE_DEPTH": depth, "TRACE_QD": 2})
    r = run_traced(t, demo_image(), demo_program(), trace_ctrl=0)
    assert r["count"] == 0 and r["drop"] == 0
