"""Bandwidth and latency of an accelerator board's memory paths, and its decode, generate and
prefill programs' cycles: the same programs on the card and on its board model (the
co-simulation's calibration at the card's clocks).

    PYTHONPATH=. python3 tools/board_perf.py [bench ...] [--target T] --dev /dev/xdma0
                                    [--points P,...] [--json F]
                                    (an XDMA card: its core clock from CORE_KHZ)
    python3 tools/board_perf.py [bench ...] --target bcu1525-wg --sim --plus +cp=3333,+dimm_lat=N,...
                                    [--params ARD=64,RLY=0x22555599]   (tb_bcuwg: the bridges,
                                    the pairs and otpu_dimm_model's four DIMMs; --params its build's)
    python3 tools/board_perf.py [bench ...] --target bcu1525-s2 --sim --plus ... --params FP=c,...
                                    (two slices, two DIMMs each: FP a floorplan of boards/bcu1525/
                                    scripts/bcu_fp.py, its DMAP and RLY)
    python3 tools/board_perf.py [bench ...] --cosim [--ch-gbps G] [--lat-ns N]   (rtlsim)
    python3 tools/board_perf.py [bench ...] --isa                    (the ISA simulator: no time)

--sim is the board model: the channel bridges, the clock crossings and the memory's model in
its own clock. --cosim is the co-simulation: tb_top's otpu_top at the target's
micro-architecture in the core clock, a channel the native model's sustained rate (--ch-gbps)
and read latency (--lat-ns), opentpu.target.mem_model's arguments: by default the target's
sim_ch_gbps / sim_lat_ns. At S > 1 the collective is the card's: otpu_board runs it through
otpu_coll_hub, at COLL_PIPE 0 one read in flight across the slices' register stages, and
otpu_top builds the same at COLL_HUB (an FPGA target's sim_uarch; OTPU_COLL_HUB=0 the unit on
the slices' own ports, OTPU_XRS the stages each way, and OTPU_COLL_PIPE the hub's reads in
flight: the round trip's reads in flight, a read a cycle).

benches (default all), on every slice at once unless a point says alone (slice 0 streams, the
others halt):
  wstream  port W, the decode's weight stream: M = 1 grouped MMs (GALL: fp4 in groups of 2 WG
           rows, int8 of WG; K 16,384, an MM 4,096 rows) over a matrix of each point's size a
           slice, random weight bytes and valid scales, the slices' images alike; every point's
           rows (stored to DRAM) against the ISA simulator, bit for bit, a mismatch named by its
           slice, point and first row. GB/s = the weights' bytes (scales: port A, apart) / the
           run's time at the core clock; a point's slope is its bytes over the cycles it adds
           to the format's previous point. passes 2: the matrix twice, the second ACC
  latency  a pointer chase: LD a 128-byte beat, RLD its first word as the next beat's address
           (random beats of a 64 MiB region, every channel); cycles a hop = the slope between two
           chain lengths; the last beat checked
  portb    port B: LDs of 64 KiB into eight TMEM slots in turn; the last eight checked; GB/s and
           the slope between two sizes
  stb      port B's writes: STs of 64 KiB from eight TMEM slots in turn (random words LD first);
           the first and the last eight checked; GB/s and the slope between two sizes
  dstep    the Gated DeltaNet state's read-modify-write (the decode's DSTEP): DSTEPs over N fp32
           states in DRAM, 128 x 128 a head as Qwen3.5's (pair: two heads a DSTEP, 256 x 128,
           DSTEP2), random states and inputs; FMT:MIB: with wstream's matrix streamed beside
           them (its MMs first in the program, the units running together: the decode's mixed
           traffic); the states, outputs and rows against the ISA simulator, bit for bit; GB/s =
           the states' bytes read and written / the run's time, cycles a head, and the slope
           between two sizes of a kind
  mixed    port B's loads (ld) or stores (st) of MIB beside wstream's matrix (FMT:WMIB: its MMs
           first in the program, the units running together, as dstep's): 64 KiB an LD or ST
           through two TMEM slots; the loaded slots, the stores' first and last 64 KiB and the
           MM rows against the ISA simulator. Without FMT:WMIB, alone. Against wstream's same
           point alone, what the side traffic costs the stream: a read's (row conflicts beside
           the stream) apart from a write's (the turnarounds); the slope between two sizes of a
           kind
  gather   the collective's all-gather as the decode runs it (Qwen3.5's: one row of 256, 512 or
           1,024 words a slice): N GATHERs of a COLS-word segment a slice into every slice's
           TMEM, each slice's words its own; the gathered rows against the ISA simulator (N 0:
           the program's own cycles, nothing checked); cycles a GATHER (the slope between two
           counts) and words a cycle into each slice
  decode   a model's decode step (Image.compile_step at a position) on a zero image: cycles (three runs on the card, the first after the image's
           writes), instruction counts alike, tok/s at the core clock. An image whose experts
           are in path (a)'s slots (GPT-OSS's) runs without its host: served past every
           request, each expert present in a slot of its layer (present_words)
  generate a model's generate loop (Image.compile_generate of a run-time position's bucket, no
           chain) on a zero image from the position for n tokens, a run for each n: the cycles a
           token (the slope over the n), the instructions before its token loop (the step's
           ahead); every embedding row served (embed_host) as decode's experts
  prefill  a dense model's tiled prompt run (tiled.compile_tile_run, kind P: no logits) of T rows
           at the first position of bucket BLOCKS (from conv_k - 1) on a zero image, the run's
           position in the generate state's tpos word: cycles, rows a second; with --trace the
           run's phases (tools/prefill_phases.py: cycles, DRAM bytes by kind, unit busy) to
           TRACE-pf<i>.json and the trace's text to TRACE-pf<i>.txt.gz
  emrun    a Qwen3.5-MoE model's expert-major expert run of a chunk (em_case: its first layer, the
           records and every used expert landed as tools/moe_prefill_cosim.py em sets them):
           cycles, cycles a pass, the experts' GB/s; every used expert's tag zeroed, checked
A point: MODEL:WFORMAT:HEAD:BLOCKS:T[:lLAYER][:K=V,...] for prefill (WFORMAT prod: the family's
production formats; a MoE model's LAYER: its layer run; the Spec's fields as decode's);
MODEL:ROWS[:USED[:WFORMAT]] for emrun;
MODEL:WFORMAT:HEAD:POS for decode (no default; e.g. qwen35-2b:fp4:int8:512; POS rN:
position N at run time, the chat's program: Image.compile_decode, the token and position in
the ARG registers; a fifth field K=V,... sets the Spec's fields, e.g. pair_loop=0),
decode's at rPOS for generate, then @N,N,... the token counts (default 1,2,3),
FMT:MIB[:passes][:alone] for wstream (default fp4:8 fp4:32 fp4:128 fp4:512 fp4:512:2
int8:64 int8:1024 fp4:512:1:alone int8:1024:1:alone; --sim fp4:2 fp4:8 fp4:32 int8:8 int8:32
fp4:8:1:alone), HOPS for latency (16 1040; --sim 16 272), MIB for portb and stb (16 64; --sim
4 16), HEADS[:pair][:FMT:MIB] for dstep (16 64 256 16:pair 64:pair 32:fp4:128; --sim 8 32
8:pair 32:pair 8:fp4:32), ld|st:MIB[:FMT:WMIB] for mixed (ld:2 st:2 ld:2:fp4:128 ld:4:fp4:128
st:2:fp4:128 st:4:fp4:128; --sim ld:1 st:1 ld:1:fp4:32 st:1:fp4:32), N:COLS for gather (64:256 256:256 64:1024; --sim 16:256 64:256 16:1024).
On the card the counters' deltas a run (DRAM_RD, MXU_BUSY, MXU_STARVE, DRAM_WAIT). The
core's peak: one chunk (WG x D bytes) a core cycle a slice.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from opentpu import isa as I                                    # noqa: E402
from opentpu import quant as Q                                  # noqa: E402
from opentpu import target as T                                 # noqa: E402
from opentpu.host.board import Board, device_config             # noqa: E402
from opentpu.isasim import Machine                              # noqa: E402

BENCHES = ("wstream", "latency", "portb", "stb", "dstep", "mixed", "gather", "decode",
           "generate", "prefill", "emrun")
X_AT, PROG_AT, OUT_AT, SA_AT = 0x0, 0x100000, 0x200000, 0x400000   # DRAM, a slice's
K = 16384                       # wstream: the MMs' inner dimension (ACT_BLOCKS 128 x D)
MM_ROWS = 4096                  # rows an MM
CHASE_AT, CHASE_SPAN = 0x800000, 1 << 26
PB_AT, PB_LD = 0x800000, 16384  # portb: words a LD (64 KiB), eight TMEM slots (stb: an ST's)
DS_IN, DS_AT = 0x800000, 0x1000000  # dstep: its inputs (TMEM's first 4,096 words), the states
DS_QK, DS_V, DS_G, DS_O = 0, 1024, 2048, 4096   # dstep's TMEM: q | k, v, decay | beta, o (8 slots
#                                 of a DSTEP's rows: the card's TMEM keeps other runs' words)
GA_SRC, GA_DST = 0, 8192        # gather's TMEM: a slice's segment, the gathered rows
POINTS = {"wstream": "fp4:8 fp4:32 fp4:128 fp4:512 fp4:512:2 int8:64 int8:1024 fp4:512:1:alone "
                     "int8:1024:1:alone",
          "latency": "16 1040", "portb": "16 64", "stb": "16 64",
          "dstep": "16 64 256 16:pair 64:pair 32:fp4:128",
          "mixed": "ld:2 st:2 ld:2:fp4:128 ld:4:fp4:128 st:2:fp4:128 st:4:fp4:128",
          "gather": "64:256 256:256 64:1024", "decode": "", "generate": "", "prefill": "",
          "emrun": ""}
SIM_POINTS = {"wstream": "fp4:2 fp4:8 fp4:32 int8:8 int8:32 int8:64 fp4:8:1:alone",
              "latency": "16 272", "portb": "4 16", "stb": "4 16",
              "dstep": "8 32 8:pair 32:pair 8:fp4:32",
              "mixed": "ld:1 st:1 ld:1:fp4:32 st:1:fp4:32",
              "gather": "16:256 64:256 16:1024", "decode": "", "generate": "", "prefill": "",
              "emrun": ""}
# the board model's default clock: the core at 100 MHz (its half period in ps)
SIM_PLUS = ["+cp=5000"]


def _mib(n: int) -> str:
    return f"{n / 2**20:g} MiB"


class IsaBoard:
    """Board's write / read / load / run on the ISA simulator, S slices (--isa: the programs
    and the checks without a card; cycles 0)."""

    def __init__(self, cfg):
        self.cfg, self.beat = cfg, 128
        self.dram = [np.zeros(cfg.DRAM_BYTES, np.uint8) for _ in range(cfg.S)]
        self.top = 0                    # the bytes written up to (every slice's)

    def write(self, addr, data, s=0):
        d = np.ascontiguousarray(data).view(np.uint8).ravel()
        self.dram[s][addr:addr + d.size] = d
        self.top = max(self.top, addr + d.size)

    def read(self, addr, n, s=0):
        return self.dram[s][addr:addr + n].copy()

    def load(self, addr, progs):
        self.progs = progs if isinstance(progs[0], (list, tuple)) else [progs]
        self.args = None

    def release_program(self):
        pass

    def set_args(self, words):
        self.args = list(words)

    def run(self, timeout=None):
        m = Machine(self.cfg, self.progs, self.dram, *([self.args] if self.args else [])).run()
        for s, sl in enumerate(m.slices):
            self.dram[s][:] = sl.dram
        return {"instructions": [sl.icount for sl in m.slices], "cycles": 0}

    def snapshot(self):
        return None


class CosimBoard(IsaBoard):
    """Board's calls on the co-simulation (rtlsim: tb_top's otpu_board, the native channels at
    `plus`, opentpu.target.mem_model's arguments); cycles the RTL's."""

    def __init__(self, cfg, plus: list):
        super().__init__(cfg)
        self.plus = plus

    def run(self, timeout=None):
        from opentpu import rtlsim
        # the images up to the bytes written (rtlsim boots the programs right after them; a
        # whole DRAM's image would put them above it, the simulated DRAM doubled)
        imgs = [d[:self.top] for d in self.dram]
        drams, _, st = rtlsim.run(self.cfg, self.progs, imgs, uarch=dict(rtlsim.BOARD_UARCH),
                                  axi=True, boot=True, stall=0, max_cycles=1 << 40,
                                  plusargs=self.plus)
        for s in range(self.cfg.S):
            n = min(len(drams[s]), len(self.dram[s]))
            self.dram[s][:n] = drams[s][:n]
        return {"instructions": list(st["instructions"]), "cycles": int(st["cycles"])}


def cosim_plus(t, mhz: float, gbps: float | None, lat_ns: float | None) -> list:
    """mem_model's arguments at `mhz` with the channel's rate and latency given (default the
    target's sim_ch_gbps / sim_lat_ns)."""
    p = t.platform
    gbps = p["sim_ch_gbps"] if gbps is None else gbps
    lat_ns = p["sim_lat_ns"] if lat_ns is None else lat_ns
    return [f"+nat_bpc={round(gbps * 1e6 / mhz)}", f"+axi_lat={round(lat_ns * mhz / 1000)}",
            "+nat_refi=0", "+nat_rfc=0"]


# ---------------------------------------------------------------- wstream
def wlayout(cfg, fmt: str, nbytes: int) -> dict:
    """A matrix of `nbytes` weight bytes in `fmt` grouped for cfg (GALL): its rows, groups, group
    stride and scale stride, and where its scales and weights go (the weights after the scales,
    at a whole MiB)."""
    R = Q.grp_r(fmt, cfg.WG)
    rs, srs = Q.grp_row_bytes(K, fmt, cfg.D, cfg.WG), Q.grp_scale_bytes(K, fmt, cfg.D, cfg.WG)
    G = nbytes // rs
    rows = G * R
    if rows % MM_ROWS and rows > MM_ROWS or G * rs != nbytes:
        raise ValueError(f"{fmt} {_mib(nbytes)}: not whole groups of {rs} bytes / MMs of "
                         f"{MM_ROWS} rows")
    if K + rows > cfg.TMEM_WORDS:
        raise ValueError(f"{fmt} {_mib(nbytes)}: {rows} rows' outputs past TMEM")
    wa = -(-(SA_AT + G * srs) // (1 << 20)) << 20
    return dict(fmt=fmt, wg=cfg.WG, R=R, rs=rs, srs=srs, G=G, rows=rows, wa=wa,
                sbytes=G * srs, end=wa + nbytes)


def wimage(L: dict, seed: int = 11) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The activation (K floats), the grouped scale areas and the weight bytes. The weights are
    random bytes in place (the grouped, hashed layout of random rows is random; int8 without
    -128); the scales are random rows' (Sw ~ 0.01, block data in the quantizer's range) grouped
    by quant.group_scales."""
    rng = np.random.default_rng(seed)
    fmt, N = L["fmt"], L["rows"]
    x = (rng.standard_normal(K) * np.exp(rng.uniform(-2, 2, K // 32).repeat(32))).astype(np.float32)
    w = rng.integers(0, 256, L["G"] * L["rs"], dtype=np.uint8)
    if fmt == "int8":
        w[w == 0x80] = 0x7F
    KB = K // 128
    rows = np.zeros((N, Q.scale_bytes(K, fmt)), np.uint8)
    rows[:, :4] = (rng.standard_normal(N) * 0.01).astype(np.float32)[:, None].view(np.uint8)
    h = Q.swp(fmt, 1)
    if fmt == "int8":
        rows[:, h:h + KB] = rng.integers(0, 20, (N, KB))
    else:
        word = rng.integers(0, 20, (N, KB)) | sum(
            rng.integers(0, 64, (N, KB)) << (8 + 6 * b) for b in range(4))
        rows[:, h:h + 4 * KB] = word.astype(np.uint32).view(np.uint8).reshape(N, -1)
    return x, Q.group_scales(rows, fmt, K, L["wg"]), w


def wprogram(L: dict, passes: int = 1) -> list:
    """LD the activation, QACT it (one row, K / D blocks), the MMs over the matrix `passes` times
    (the later passes ACC), the rows' results to OUT_AT."""
    wf = I.W8 if L["fmt"] == "int8" else I.W4F
    KB = K // 128
    prog = [I.ld(X_AT, 0, K), I.qact(0, 1, 0, KB, K)]
    for p in range(passes):
        for r0 in range(0, L["rows"], MM_ROWS):
            n, g0 = min(MM_ROWS, L["rows"] - r0), r0 // L["R"]
            prog.append(I.mm(L["wa"] + g0 * L["rs"], SA_AT + g0 * L["srs"], K + r0, n, KB,
                             L["rs"], n, 1, 0, L["srs"], acc=p > 0, wf=wf, grp=L["R"]))
    return prog + [I.st(OUT_AT, K, L["rows"]), I.halt()]


def wref(cfg, L: dict, prog: list, img: tuple) -> tuple[np.ndarray, int]:
    """The ISA simulator's rows (one slice: the slices' images are alike) and instructions."""
    c = dataclasses.replace(cfg, S=1, DRAM_BYTES=1 << (L["end"] - 1).bit_length())
    m = Machine(c, [prog], [_dram(c.DRAM_BYTES, L, img)]).run()
    sl = m.slices[0]
    return sl.dram[OUT_AT:OUT_AT + 4 * L["rows"]].copy(), sl.icount


def _dram(n: int, L: dict, img: tuple) -> np.ndarray:
    d = np.zeros(n, np.uint8)
    x, s, w = img
    d[X_AT:X_AT + x.nbytes] = x.view(np.uint8)
    d[SA_AT:SA_AT + s.size] = s.ravel()
    d[L["wa"]:L["wa"] + w.size] = w
    return d


def dlayout(cfg, pt: str) -> dict:
    """A dstep point HEADS[:pair][:FMT:MIB]: its heads' state shape and bytes, and where its
    inputs, states, TMEM slots and outputs go; with FMT:MIB, wstream's matrix (wlayout) streamed
    beside the DSTEPs (one program: its MMs first, the units running together), the DSTEPs' DRAM
    after the matrix and their TMEM after the MMs' rows."""
    f = pt.split(":")
    n, pair = int(f[0]), "pair" in f[1:]
    mm = [x for x in f[1:] if x != "pair"]
    rows, cols = (2 if pair else 1) * 128, 128
    sb = 4 * rows * cols
    L = wlayout(cfg, mm[0], int(mm[1]) << 20) if mm else None
    if L:
        a = -(-L["end"] // (1 << 20)) << 20
        D = {"in": a, "at": a + (1 << 20), "tb": -(-(K + L["rows"]) // 1024) * 1024,
             "o": OUT_AT + (1 << 20)}
    else:
        D = {"in": DS_IN, "at": DS_AT, "tb": 0, "o": OUT_AT}
    return D | dict(n=n, pair=pair, rows=rows, cols=cols, sb=sb, L=L, end=D["at"] + n * sb)


def mlayout(cfg, pt: str) -> dict:
    """A mixed point ld|st:MIB[:FMT:WMIB]: its kind, bytes and LDs / STs (64 KiB each), where its
    DRAM region, its source (st: the two slots' words, loaded first), its outputs (ld: the two
    slots, stored last) and its TMEM slots go; with FMT:WMIB wstream's matrix (wlayout) beside,
    the region after the matrix and the slots after the MMs' rows."""
    f = pt.split(":")
    kind, nb = f[0], int(f[1]) << 20
    if kind not in ("ld", "st") or nb % (4 * PB_LD):
        raise ValueError(f"mixed point {pt}: ld|st:MIB[:FMT:WMIB]")
    L = wlayout(cfg, f[2], int(f[3]) << 20) if len(f) > 3 else None
    a = (-(-L["end"] // (1 << 20)) << 20) if L else PB_AT
    tb = -(-(K + L["rows"]) // 1024) * 1024 if L else 0
    if tb + 2 * PB_LD > cfg.TMEM_WORDS:
        raise ValueError(f"mixed point {pt}: its slots past TMEM")
    return dict(kind=kind, nb=nb, n=nb // (4 * PB_LD), L=L, at=a, src=a + nb,
                o=OUT_AT + (1 << 20), tb=tb, end=a + nb + 8 * PB_LD)


# ---------------------------------------------------------------- the runs
class Bench:
    def __init__(self, board, cfg, sim: bool, isa: bool, mhz: float, out: dict):
        self.b, self.cfg, self.sim, self.isa, self.mhz, self.out = board, cfg, sim, isa, mhz, out
        self.S = cfg.S
        self.failed = False
        self.trace = None               # decode: the last run's hardware trace, to this prefix
        self.solo = False               # decode: one slice's program on every slice (--solo)
        self.em_pass = None             # emrun: the passes' entries (--em-pass; None: the rule's)
        self.em_hi: set = set()         # (emrun: the 16 MiB chunks past `at` written)
        self.em_isa = False             # emrun: random words, the records checked against the ISA

    def run(self, progs, counters: bool = True, at: int = PROG_AT, args=None,
            trace: bool = False) -> dict:
        """Load (at `at`) and run (a program a slice) with run arguments `args` (ARG0..7, after
        the load as the engine sets them: the simulation's start from reset every flush); the
        counters' deltas on the card; trace: the hardware trace's first records (Board.run)."""
        b = self.b
        b.load(at, progs if self.S > 1 else progs[0])
        if args is not None:
            b.set_args(list(args) + [0] * (8 - len(args)))
        s0 = b.snapshot() if counters and not self.sim else None
        st = b.run(timeout=60.0, **({"trace": {"keep": "first"}} if trace else {}))
        s1 = b.snapshot() if s0 is not None else None
        if s0 and s1:
            st["counters"] = {k: s1[k] - s0[k] for k in ("DRAM_RD", "DRAM_WR", "MXU_BUSY",
                                                         "MXU_STARVE", "DRAM_WAIT", "RUNNING")
                              if k in s0}
        return st

    def gbs(self, nbytes: int, cycles: int) -> float:
        return nbytes / (cycles / (self.mhz * 1e6)) / 1e9 if cycles else 0.0

    def note(self, bench: str, rec: dict) -> None:
        self.out.setdefault(bench, []).append(rec)
        print(f"{bench:8} " + "  ".join(f"{k} {v}" for k, v in rec.items()
                                          if k not in ("counters",)), flush=True)
        if not rec.get("ok", True):
            self.failed = True

    def wstream(self, points: list[str]) -> None:
        prev: dict = {}
        for pt in points:
            f = pt.split(":")
            fmt, mib = f[0], int(f[1])
            passes = int(f[2]) if len(f) > 2 else 1
            alone = len(f) > 3 and f[3] == "alone"
            L = wlayout(self.cfg, fmt, mib << 20)
            img = wimage(L)
            prog = wprogram(L, passes)
            t0 = time.time()
            want, icount = wref(self.cfg, L, prog, img)
            t_isa = time.time() - t0
            x, s, w = img
            on = [0] if alone else list(range(self.S))
            for k in on:
                self.b.write(X_AT, x, k)
                self.b.write(SA_AT, s.ravel(), k)
                self.b.write(L["wa"], w, k)
            progs = [prog if k in on else [I.halt()] for k in range(self.S)]
            t0 = time.time()
            st = self.run(progs)
            t_run = time.time() - t0
            bad = []
            for k in on:
                if st["instructions"][k] != icount:
                    bad.append(f"slice {k}: {st['instructions'][k]} of {icount} instructions")
                    continue
                got = self.b.read(OUT_AT, 4 * L["rows"], k)
                g, h = got.view("<u4"), want.view("<u4")
                d = np.flatnonzero(g != h)
                if d.size:
                    r = int(d[0])
                    bad.append(f"slice {k}: {d.size} of {L['rows']} rows differ, first row {r} "
                               f"(MM {r // MM_ROWS}, group {r // L['R']}): got "
                               f"{int(g[r]):#010x} want {int(h[r]):#010x}")
            nb = (mib << 20) * passes
            cyc = st["cycles"]
            rec = {"point": pt, "ok": not bad, "slices": len(on), "weights": _mib(nb),
                   "scales": _mib(L["sbytes"] * passes), "cycles": cyc,
                   "GB/s a slice": round(self.gbs(nb, cyc), 2),
                   "GB/s all": round(self.gbs(nb * len(on), cyc), 2)}
            key = (fmt, alone)
            if key in prev and cyc > prev[key][1]:
                rec["slope GB/s a slice"] = round(self.gbs(nb - prev[key][0],
                                                           cyc - prev[key][1]), 2)
            prev[key] = (nb, cyc)
            rec["isa s"], rec["run s"] = round(t_isa, 1), round(t_run, 1)
            if bad:
                rec["mismatch"] = "; ".join(bad)
            if "counters" in st:
                rec["counters"] = st["counters"]
            self.note("wstream", rec)

    def latency(self, points: list[str]) -> None:
        rng = np.random.default_rng(13)
        prev = None
        for pt in points:
            n = int(pt)
            beats = rng.choice(CHASE_SPAN // 128, n + 1, replace=False)
            addrs = CHASE_AT + 128 * beats.astype(np.int64)
            img = np.zeros(CHASE_SPAN, np.uint8)
            words = img.view("<u4")
            for i in range(n):
                words[(addrs[i] - CHASE_AT) // 4] = addrs[i + 1]
                words[(addrs[i] - CHASE_AT) // 4 + 1:(addrs[i] - CHASE_AT) // 4 + 32] = i
            prog = [I.li(1, int(addrs[0])), I.loop(2, n),
                    I.ld(0, 0, 32, ra=1),
                    I.rld(1, 0, raw=True, bound=(CHASE_AT, CHASE_AT + CHASE_SPAN - 128)),
                    I.st(OUT_AT, 0, 32), I.halt()]
            for k in range(self.S):
                self.b.write(CHASE_AT, img, k)
            st = self.run([prog] * self.S)
            last = addrs[n - 1] - CHASE_AT
            want = img[last:last + 128]
            ok = all(np.array_equal(self.b.read(OUT_AT, 128, k), want) for k in range(self.S))
            rec = {"point": f"{n} hops", "ok": ok, "cycles": st["cycles"]}
            if prev and st["cycles"] > prev[1]:
                c = (st["cycles"] - prev[1]) / (n - prev[0])
                rec["cycles a hop"] = round(c, 2)
                rec["ns a hop"] = round(c * 1e3 / self.mhz, 1)
            prev = (n, st["cycles"])
            self.note("latency", rec)

    def portb(self, points: list[str]) -> None:
        rng = np.random.default_rng(17)
        prev = None
        for pt in points:
            nb = int(pt) << 20
            nld = nb // (4 * PB_LD)
            data = rng.integers(0, 256, nb, dtype=np.uint8)
            prog = [I.ld(PB_AT + 4 * PB_LD * i, (i % 8) * PB_LD, PB_LD) for i in range(nld)]
            prog += [I.st(OUT_AT, 0, 8 * PB_LD), I.halt()]
            for k in range(self.S):
                self.b.write(PB_AT, data, k)
            st = self.run([prog] * self.S)
            want = np.zeros(32 * PB_LD, np.uint8)
            for i in range(max(0, nld - 8), nld):
                want[(i % 8) * 4 * PB_LD:(i % 8 + 1) * 4 * PB_LD] = \
                    data[4 * PB_LD * i:4 * PB_LD * (i + 1)]
            ok = all(np.array_equal(self.b.read(OUT_AT, 32 * PB_LD, k), want)
                     for k in range(self.S))
            rec = {"point": _mib(nb), "ok": ok, "cycles": st["cycles"],
                   "GB/s a slice": round(self.gbs(nb, st["cycles"]), 2)}
            if prev and st["cycles"] > prev[1]:
                rec["slope GB/s a slice"] = round(self.gbs(nb - prev[0], st["cycles"] - prev[1]), 2)
            prev = (nb, st["cycles"])
            if "counters" in st:
                rec["counters"] = st["counters"]
            self.note("portb", rec)

    def stb(self, points: list[str]) -> None:
        rng = np.random.default_rng(29)
        prev = None
        for pt in points:
            nb = int(pt) << 20
            nst = nb // (4 * PB_LD)
            data = rng.integers(0, 256, 32 * PB_LD, dtype=np.uint8)
            prog = [I.ld(OUT_AT, 0, 8 * PB_LD)]
            prog += [I.st(PB_AT + 4 * PB_LD * i, (i % 8) * PB_LD, PB_LD) for i in range(nst)]
            prog += [I.halt()]
            for k in range(self.S):
                self.b.write(OUT_AT, data, k)
            st = self.run([prog] * self.S)
            slot = lambda i: data[(i % 8) * 4 * PB_LD:(i % 8 + 1) * 4 * PB_LD]
            want = np.concatenate([slot(0)] + [slot(i) for i in range(max(1, nst - 8), nst)])
            ok = all(np.array_equal(np.concatenate(
                [self.b.read(PB_AT, 4 * PB_LD, k),
                 self.b.read(PB_AT + 4 * PB_LD * max(1, nst - 8),
                             4 * PB_LD * (nst - max(1, nst - 8)), k)]), want)
                for k in range(self.S))
            rec = {"point": _mib(nb), "ok": ok, "cycles": st["cycles"],
                   "GB/s a slice": round(self.gbs(nb, st["cycles"]), 2)}
            if prev and st["cycles"] > prev[1]:
                rec["slope GB/s a slice"] = round(self.gbs(nb - prev[0], st["cycles"] - prev[1]), 2)
            prev = (nb, st["cycles"])
            if "counters" in st:
                rec["counters"] = st["counters"]
            self.note("stb", rec)

    def dstep(self, points: list[str]) -> None:
        rng = np.random.default_rng(23)
        prev: dict = {}
        for pt in points:
            D = dlayout(self.cfg, pt)
            n, rows, cols, sb, L, tb = D["n"], D["rows"], D["cols"], D["sb"], D["L"], D["tb"]
            states = (rng.standard_normal(n * rows * cols) * 0.1).astype(np.float32)
            inp = np.zeros(DS_O, np.float32)
            inp[DS_QK:DS_QK + 4 * cols] = rng.standard_normal(4 * cols) * 0.09
            inp[DS_V:DS_V + rows] = rng.standard_normal(rows)
            inp[DS_G:DS_G + 4] = (0.9, 0.95, 0.5, 0.25) if D["pair"] else (0.9, 0.5, 0.0, 0.0)
            prog = wprogram(L)[:-2] if L else []
            prog += [I.ld(D["in"], tb, DS_O)]
            prog += [I.dstep(D["at"] + sb * i, tb + DS_QK, tb + DS_V, rows, cols, tb + DS_G,
                             2 if D["pair"] else 1, tb + DS_O + rows * (i % 8), pair=D["pair"],
                             qkb=D["pair"]) for i in range(n)]
            no = min(n, 8) * rows                                # (the words the DSTEPs wrote)
            prog += [I.st(D["o"], tb + DS_O, no)]
            prog += ([I.st(OUT_AT, K, L["rows"])] if L else []) + [I.halt()]
            img = wimage(L) if L else None
            c = dataclasses.replace(self.cfg, S=1, DRAM_BYTES=1 << (D["end"] - 1).bit_length())
            d = _dram(c.DRAM_BYTES, L, img) if L else np.zeros(c.DRAM_BYTES, np.uint8)
            d[D["in"]:D["in"] + inp.nbytes] = inp.view(np.uint8)
            d[D["at"]:D["at"] + states.nbytes] = states.view(np.uint8)
            ref = Machine(c, [prog], [d]).run().slices[0]
            for k in range(self.S):
                if L:
                    x, sc, w = img
                    self.b.write(X_AT, x, k)
                    self.b.write(SA_AT, sc.ravel(), k)
                    self.b.write(L["wa"], w, k)
                self.b.write(D["in"], inp, k)
                self.b.write(D["at"], states, k)
            st = self.run([prog] * self.S)
            bad = []
            checks = [("states", D["at"], n * sb), ("outputs", D["o"], 4 * no)]
            checks += [("MM rows", OUT_AT, 4 * L["rows"])] if L else []
            for k in range(self.S):
                if st["instructions"][k] != ref.icount:
                    bad.append(f"slice {k}: {st['instructions'][k]} of {ref.icount} instructions")
                    continue
                for what, at, nb in checks:
                    g = self.b.read(at, nb, k).view("<u4")
                    h = ref.dram[at:at + nb].view("<u4")
                    dif = np.flatnonzero(g != h)
                    if dif.size:
                        w = int(dif[0])
                        bad.append(f"slice {k}: {what}: {dif.size} of {nb // 4} words differ, "
                                   f"first word {w}" + (f" (head {w * 4 // sb})"
                                                        if what == "states" else ""))
            cyc = st["cycles"]
            rec = {"point": pt, "ok": not bad, "states": _mib(n * sb)}
            if L:
                rec["weights"] = _mib(L["G"] * L["rs"])
            rec |= {"cycles": cyc, "cycles a head": round(cyc / (n * rows // 128), 1),
                    "GB/s a slice": round(self.gbs(2 * n * sb, cyc), 2)}
            kind = pt.split(":", 1)[1] if ":" in pt else ""
            if kind in prev and cyc > prev[kind][1]:
                rec["slope GB/s a slice"] = round(self.gbs(2 * (n - prev[kind][0]) * sb,
                                                           cyc - prev[kind][1]), 2)
                rec["slope cycles a head"] = round((cyc - prev[kind][1]) /
                                                   ((n - prev[kind][0]) * rows // 128), 1)
            prev[kind] = (n, cyc)
            if bad:
                rec["mismatch"] = "; ".join(bad)
            if "counters" in st:
                rec["counters"] = st["counters"]
            self.note("dstep", rec)

    def mixed(self, points: list[str]) -> None:
        rng = np.random.default_rng(37)
        prev: dict = {}
        for pt in points:
            D = mlayout(self.cfg, pt)
            L, n, tb, at = D["L"], D["n"], D["tb"], D["at"]
            data = rng.integers(0, 256, D["nb"] if D["kind"] == "ld" else 8 * PB_LD, dtype=np.uint8)
            prog = wprogram(L)[:-2] if L else []
            if D["kind"] == "ld":
                prog += [I.ld(at + 4 * PB_LD * i, tb + (i % 2) * PB_LD, PB_LD) for i in range(n)]
                prog += [I.st(D["o"], tb, 2 * PB_LD)]
                checks = [("loaded slots", D["o"], 8 * PB_LD)]
            else:
                prog += [I.ld(D["src"], tb, 2 * PB_LD)]
                prog += [I.st(at + 4 * PB_LD * i, tb + (i % 2) * PB_LD, PB_LD) for i in range(n)]
                checks = [("first stores", at, 8 * PB_LD),
                          ("last stores", at + D["nb"] - 8 * PB_LD, 8 * PB_LD)]
            prog += ([I.st(OUT_AT, K, L["rows"])] if L else []) + [I.halt()]
            checks += [("MM rows", OUT_AT, 4 * L["rows"])] if L else []
            img = wimage(L) if L else None
            c = dataclasses.replace(self.cfg, S=1, DRAM_BYTES=1 << (D["end"] - 1).bit_length())
            d = _dram(c.DRAM_BYTES, L, img) if L else np.zeros(c.DRAM_BYTES, np.uint8)
            src = at if D["kind"] == "ld" else D["src"]
            d[src:src + data.size] = data
            ref = Machine(c, [prog], [d]).run().slices[0]
            for k in range(self.S):
                if L:
                    x, sc, w = img
                    self.b.write(X_AT, x, k)
                    self.b.write(SA_AT, sc.ravel(), k)
                    self.b.write(L["wa"], w, k)
                self.b.write(src, data, k)
            st = self.run([prog] * self.S)
            bad = []
            for k in range(self.S):
                if st["instructions"][k] != ref.icount:
                    bad.append(f"slice {k}: {st['instructions'][k]} of {ref.icount} instructions")
                    continue
                for what, a0, nb in checks:
                    g = self.b.read(a0, nb, k).view("<u4")
                    dif = np.flatnonzero(g != ref.dram[a0:a0 + nb].view("<u4"))
                    if dif.size:
                        bad.append(f"slice {k}: {what}: {dif.size} of {nb // 4} words differ, "
                                   f"first word {int(dif[0])}")
            cyc = st["cycles"]
            rec = {"point": pt, "ok": not bad, D["kind"]: _mib(D["nb"])}
            if L:
                rec["weights"] = _mib(L["G"] * L["rs"])
            rec |= {"cycles": cyc, "GB/s a slice": round(self.gbs(D["nb"], cyc), 2)}
            kind = pt.split(":", 2)[0] + (":" + pt.split(":", 2)[2] if L else "")
            if kind in prev and cyc > prev[kind][1]:
                rec["slope cycles a MiB"] = round((cyc - prev[kind][1]) /
                                                  ((D["nb"] - prev[kind][0]) / (1 << 20)), 1)
            prev[kind] = (D["nb"], cyc)
            if bad:
                rec["mismatch"] = "; ".join(bad)
            if "counters" in st:
                rec["counters"] = st["counters"]
            self.note("mixed", rec)

    def gather(self, points: list[str]) -> None:
        rng = np.random.default_rng(31)
        prev: dict = {}
        S = self.S
        for pt in points:
            n, cols = (int(x) for x in pt.split(":"))
            data = [rng.integers(0, 1 << 32, cols, dtype=np.uint32) for _ in range(S)]
            prog = [I.ld(DS_IN, GA_SRC, cols)]
            prog += [I.gather(GA_SRC, GA_DST, 1, cols, cols, S * cols, cols) for _ in range(n)]
            prog += [I.st(OUT_AT, GA_DST, S * cols), I.halt()]
            c = dataclasses.replace(self.cfg, DRAM_BYTES=1 << 24)
            d = [np.zeros(c.DRAM_BYTES, np.uint8) for _ in range(S)]
            for k in range(S):
                d[k][DS_IN:DS_IN + 4 * cols] = data[k].view(np.uint8)
                self.b.write(DS_IN, data[k], k)
            m = Machine(c, [prog] * S, d).run()
            st = self.run([prog] * S)
            bad = []
            for k in range(S):
                if st["instructions"][k] != m.slices[k].icount:
                    bad.append(f"slice {k}: {st['instructions'][k]} of {m.slices[k].icount} "
                               f"instructions")
                    continue
                if not n:                       # (no GATHER: the rows stored were never written)
                    continue
                g = self.b.read(OUT_AT, 4 * S * cols, k).view("<u4")
                h = m.slices[k].dram[OUT_AT:OUT_AT + 4 * S * cols].view("<u4")
                dif = np.flatnonzero(g != h)
                if dif.size:
                    bad.append(f"slice {k}: {dif.size} of {S * cols} words differ, first word "
                               f"{int(dif[0])} (slice {int(dif[0]) // cols}'s segment)")
            cyc = st["cycles"]
            rec = {"point": pt, "ok": not bad, "cycles": cyc}
            if cols in prev and n > prev[cols][0] and cyc > prev[cols][1]:
                g1 = (cyc - prev[cols][1]) / (n - prev[cols][0])
                rec |= {"slope cycles a GATHER": round(g1, 1),
                        "words a cycle": round(S * cols / g1, 2)}
            prev[cols] = (n, cyc)
            if bad:
                rec["mismatch"] = "; ".join(bad)
            self.note("gather", rec)

    def decode(self, points: list[str], reps: int = 3) -> None:
        """A model's decode step at a position on a zero image (the timing does not depend on
        the data, as the co-simulation's), `reps` runs on the card (the first one a warm-up
        after the image's writes), one in a simulation; tok/s = the core clock / cycles."""
        for pt in points:
            if self.solo and self.S > 1:    # (one slice's compile, its program on every slice)
                c, img, progs, at, args = decode_case(dataclasses.replace(self.cfg, S=1), pt)
                c, progs = dataclasses.replace(c, S=self.S), progs * self.S
            else:
                c, img, progs, at, args = decode_case(self.cfg, pt)
            if args is not None and (self.isa or isinstance(self.b, CosimBoard)):
                raise ValueError(f"{pt}: a run-time position runs on the card or --sim")
            if self.isa or self.sim or isinstance(self.b, CosimBoard):
                if c.DRAM_BYTES > self.cfg.DRAM_BYTES:
                    raise ValueError(f"{pt}: {c.DRAM_BYTES >> 20} MiB a slice, the model's "
                                     f"{self.cfg.DRAM_BYTES >> 20}")
            self.b.release_program()        # (ICACHE: the last point's program is in the zeros)
            z = np.zeros(1 << 24, np.uint8)
            for k in range(self.S):
                for a in range(0, at, z.size):
                    self.b.write(a, z[:min(z.size, at - a)], k)
                for a, v in present_words(img):
                    self.b.write(a, v, k)
            n = 1 if self.sim or self.isa or isinstance(self.b, CosimBoard) else reps
            runs = [self.run(progs, at=at, args=args, trace=bool(self.trace) and i == n - 1)
                    for i in range(n)]
            if self.trace:
                from opentpu.host.hwlens import records_to_trace
                from opentpu.profile import parse
                from q35_trace import _dump
                tr = runs[-1]["trace"]
                f = f"{self.trace}-{len(self.out.get('decode', []))}.json"
                _dump(parse(records_to_trace(tr["records"], progs), self.cfg, progs, "card"),
                      progs, runs[-1]["cycles"], f,
                      {"point": pt, "trace": {k: tr[k] for k in ("count", "drop", "depth",
                                                                  "lost")}})
            cyc = [r["cycles"] for r in runs]
            rec = {"point": pt, "ok": len({tuple(r["instructions"]) for r in runs}) == 1,
                   "image": _mib(img.nbytes), "instructions": runs[-1]["instructions"],
                   "cycles": cyc, "tok/s": round(self.mhz * 1e6 / cyc[-1], 2) if cyc[-1] else 0}
            if "counters" in runs[-1]:
                rec["counters"] = runs[-1]["counters"]
            self.note("decode", rec)


    def generate(self, points: list[str]) -> None:
        """A model's generate loop (generate_case: POINT[@N,N,...]) on a zero image, from POS for
        n tokens, one run for each n (default 1, 2, 3), greedy, no stop id; the cycles a token:
        the least-squares slope over the n (decode's tok/s from it). ahead: the program's
        instructions before its token loop (generate._generate: the step kernel's ahead, if it
        compiled; the state's load and the count's otherwise)."""
        from opentpu.host.offload import LINE
        from opentpu.llm import generate as G
        from opentpu.llm.qwen3 import ATTN_BLOCK
        for item in points:
            pt, _, ns = item.partition("@")
            ns = [int(x) for x in (ns or "1,2,3").split(",")]
            if self.solo and self.S > 1:    # (one slice's compile, its program on every slice)
                c, img, progs, at, pos = generate_case(dataclasses.replace(self.cfg, S=1), pt)
                c, progs = dataclasses.replace(c, S=self.S), progs * self.S
            else:
                c, img, progs, at, pos = generate_case(self.cfg, pt)
            if self.isa or self.sim or isinstance(self.b, CosimBoard):
                if c.DRAM_BYTES > self.cfg.DRAM_BYTES:
                    raise ValueError(f"{pt}: {c.DRAM_BYTES >> 20} MiB a slice, the model's "
                                     f"{self.cfg.DRAM_BYTES >> 20}")
            g, mbox = img.lookup["gen"], img.lookup.get("mbox")
            self.b.release_program()        # (ICACHE: the last point's program is in the zeros)
            z = np.zeros(1 << 24, np.uint8)
            for k in range(self.S):
                for a in range(0, at, z.size):
                    self.b.write(a, z[:min(z.size, at - a)], k)
                for a, v in present_words(img):
                    self.b.write(a, v, k)
                if mbox is not None:        # (embed_host: every row served)
                    self.b.write(mbox + 2 * LINE, np.array([3e38], np.float32), k)
            runs = []
            for n in ns:
                for k in range(self.S):
                    self.b.write(g["state"], G.state_words(img.spec, 0, pos, n, [], ATTN_BLOCK),
                                 k)
                runs.append(self.run(progs, at=at))
            cyc = [r["cycles"] for r in runs]
            x = np.array(ns, float)
            slope = float(np.polyfit(x, cyc, 1)[0]) if len(ns) > 1 else float(cyc[0])
            lo = next(i for i, ins in enumerate(progs[0]) if ins.op == I.LOOP)
            rec = {"point": item, "ok": all(r["cycles"] > 0 for r in runs),
                   "image": _mib(img.nbytes), "program": len(progs[0]), "tokens": ns,
                   "instructions": [r["instructions"] for r in runs], "cycles": cyc,
                   "cycles a token": round(slope, 1),
                   "tok/s": round(self.mhz * 1e6 / slope, 2) if slope > 0 else 0,
                   "ahead": dict(collections.Counter(I.OPNAMES.get(ins.op, ins.op)
                                                     for ins in progs[0][:lo]))}
            self.note("generate", rec)

    def prefill(self, points: list[str]) -> None:
        """A dense model's tiled prompt run (prefill_case) on a zero image, one run; with
        --trace its phases (tools/prefill_phases.py) beside the trace's text. --solo at S > 1:
        one slice's compile on every slice (tiled prefill compiles one slice)."""
        from opentpu.llm.qwen3 import RUN_WORDS
        for pt in points:
            solo = self.solo and self.S > 1         # (one slice's compile on every slice)
            c, img, progs, at, tpos, T_, args = prefill_case(_solo(self.cfg, solo), pt)
            if solo:
                progs = progs * self.S
            if c.DRAM_BYTES > self.cfg.DRAM_BYTES:
                raise ValueError(f"{pt}: {c.DRAM_BYTES >> 20} MiB a slice, the model's "
                                 f"{self.cfg.DRAM_BYTES >> 20}")
            self.b.release_program()        # (ICACHE: the last point's program is in the zeros)
            z = np.zeros(1 << 24, np.uint8)
            for k in range(self.S):
                for a in range(0, at, z.size):
                    self.b.write(a, z[:min(z.size, at - a)], k)
                if args is None:
                    self.b.write(img.lookup["gen"]["state"] + 4 * RUN_WORDS["tpos"],
                                 np.array([tpos], np.float32), k)
                for a, v in present_words(img):       # (a MoE layer run's fences: served)
                    self.b.write(a, v, k)
            t0 = time.time()
            st = self.run(progs, at=at, args=args, trace=bool(self.trace))
            cyc = st["cycles"]
            rec = {"point": pt, "image": _mib(img.nbytes), "T": T_,
                   "program": len(progs[0]), "imem": self.cfg.IMEM_WORDS // 8,
                   "instructions": st["instructions"], "cycles": cyc,
                   "cycles a row": round(cyc / T_), "rows/s": round(self.mhz * 1e6 * T_ / cyc, 2)
                   if cyc else 0, "run s": round(time.time() - t0)}
            if self.trace:
                import gzip
                from opentpu.host.hwlens import drop_orphans, records_to_trace
                from opentpu.profile import parse
                import prefill_phases as PP
                tr = st["trace"]
                text = records_to_trace(tr["records"], progs)
                f = f"{self.trace}-pf{len(self.out.get('prefill', []))}"
                with gzip.open(f + ".txt.gz", "wt") as fh:
                    fh.write(text)
                p = parse(drop_orphans(text), self.cfg, progs, "card")
                sm = PP.summarize(p, progs, self.cfg)
                rec["trace"] = {k: tr[k] for k in ("count", "drop", "depth", "lost")}
                rec["traced"] = sm["traced"]
                Path(f + ".json").write_text(json.dumps({"point": pt, "cycles": cyc, "T": T_,
                                                         **sm, "ports": p.ports[0]}))
                rec["phases"] = {k: v["cycles"] for k, v in sm["phases"].items()}
            self.note("prefill", rec)

    def emrun(self, points: list[str]) -> None:
        """Expert-major's expert run of a chunk (em_case) on its image's words, one run: cycles,
        ms, cycles a pass, the experts' bytes a row; every used expert's tag zeroed after it
        (each pass zeroes its slot's), checked. --solo at S > 1: one slice's compile on every
        slice (expert-major compiles one slice)."""
        for pt in points:
            solo = self.solo and self.S > 1         # (one slice's compile on every slice)
            pt, _, hack = pt.partition(":hack=")    # (scratch: hack=MODES, decode_case's)
            c, img, dram, progs, at, args, rec, es = em_case(
                _solo(self.cfg, solo), pt, self.em_pass, 1 if self.em_isa else None)
            if hack:
                progs = [hack_program(p, hack) for p in progs]
            if solo:
                progs = progs * self.S
            if c.DRAM_BYTES > self.cfg.DRAM_BYTES:
                raise ValueError(f"{pt}: {c.DRAM_BYTES >> 20} MiB a slice, the model's "
                                 f"{self.cfg.DRAM_BYTES >> 20}")
            # the words below the programs, and past them (an Image.high's: its slots, BHW's
            # 16 GiB) the chunks with any, and those an earlier point wrote (zeroed)
            hi = {a for a in range(at, len(dram), 1 << 24) if dram[a:a + (1 << 24)].any()}
            for k in range(self.S):
                for a in range(0, at, 1 << 24):
                    self.b.write(a, dram[a:min(at, a + (1 << 24), len(dram))], k)
                for a in sorted(hi | self.em_hi):
                    self.b.write(a, dram[a:a + (1 << 24)] if a in hi
                                 else np.zeros(1 << 24, np.uint8), k)
            self.em_hi = hi
            t0 = time.time()
            st = self.run(progs, at=at, args=args, trace=bool(self.trace))
            cyc = st["cycles"]
            tags = [int(self.b.read(a + img.offload.tag, 4, 0).view("<u4")[0]) for a in es]
            ok = not any(tags)
            rec = {"point": pt, "ok": ok, "image": _mib(img.nbytes), "program": len(progs[0]),
                   "instructions": st["instructions"], "cycles": cyc,
                   "ms": round(cyc / (self.mhz * 1e3), 3),
                   "cycles a pass": round(cyc / max(rec["passes"], 1)), **rec,
                   "GB/s experts": round(self.gbs(rec["bytes"]["weights"], cyc), 2),
                   "run s": round(time.time() - t0)}
            if not ok:
                rec["mismatch"] = f"{sum(map(bool, tags))} of {len(es)} tags not zeroed"
            if self.em_isa:                     # every record (its rows' outputs) as the ISA's
                ref = Machine(c, progs[:1], [dram], args).run().slices[0]
                dev = img.descriptors(0).moe_dev
                lo, nb = dev.em_base, rec["rows"] * dev.em_rec
                bad = []
                for k in range(self.S):
                    if st["instructions"][k] != ref.icount:
                        bad.append(f"slice {k}: {st['instructions'][k]} of {ref.icount} "
                                   f"instructions")
                        continue
                    dif = np.flatnonzero(self.b.read(lo, nb, k).view("<u4")
                                         != ref.dram[lo:lo + nb].view("<u4"))
                    if dif.size:
                        bad.append(f"slice {k}: {dif.size} of {nb // 4} record words differ, "
                                   f"first word {int(dif[0])}")
                rec["isa"] = "equal" if not bad else "; ".join(bad)
                if bad:
                    rec["ok"] = False
                    rec["mismatch"] = "; ".join(filter(None, [rec.get("mismatch")] + bad))
            if self.trace:
                import gzip
                from opentpu.host.hwlens import drop_orphans, records_to_trace
                from opentpu.profile import parse
                import prefill_phases as PP
                tr = st["trace"]
                text = records_to_trace(tr["records"], progs)
                f = f"{self.trace}-em{len(self.out.get('emrun', []))}"
                with gzip.open(f + ".txt.gz", "wt") as fh:
                    fh.write(text)
                p = parse(drop_orphans(text), self.cfg, progs, "card")
                sm = PP.summarize(p, progs, self.cfg)
                rec["trace"] = {k: tr[k] for k in ("count", "drop", "depth", "lost")}
                Path(f + ".json").write_text(json.dumps({"point": pt, "cycles": cyc, **sm,
                                                         "ports": p.ports[0]}))
                rec["busy"] = {k: sm["total"][k] for k in ("DMA", "MXU", "VPU")}
            self.note("emrun", rec)


def present_words(img) -> list:
    """(address, bytes) of path (a)'s words that let a decode on a zero image run without the
    host (moe_prefill_cosim.timing_hack's, each expert in a slot of its own): served past every
    request and every directory entry present, expert e of layer j at slot e mod n of the
    layer's n; none for an image without offload slots."""
    Lo = getattr(img, "offload", None)
    if Lo is None:
        return []
    ent = np.zeros((len(Lo.slots), Lo.E, 2), np.uint32)
    for j, (a, n) in enumerate(Lo.slots):
        ent[j, :, 0] = a + np.arange(Lo.E) % n * Lo.pitch
    ent[..., 1] = np.float32(1.0).view(np.uint32)
    return [(Lo.served, np.frombuffer(np.float32(1e6).tobytes(), np.uint8)),
            (Lo.dir, ent.view(np.uint8).reshape(-1))]


def decode_case(cfg, pt: str):
    """A decode point MODEL:WFORMAT:HEAD:POS (MODEL a short name of opentpu.llm.MODELS or a
    checkpoint directory; only its config.json is read): the image (its layout) and the decode
    step compiled at POS (Image.compile_step, KV capacity the 256-position bucket past POS), as
    vpu_hack.compile_case compiles it, for cfg at the smallest power-of-two DRAM
    that holds the image and the programs (the programs do not depend on it; at: theirs, past
    the image, or past an Image.high's low area, as BoardBackend loads them); args None. POS
    rN: the step at run-time position N (Image.compile_decode over N's attention block, an
    image with lookup tables, as the chat's engine runs it), args its ARG words (token 0)."""
    from opentpu.compiler import arg_words
    from opentpu.llm.qwen3 import ATTN_BLOCK, RunPos
    model, wf, hf, pos, *sets = pt.split(":")       # (scratch: hack=MODES, vpu_hack's, as prefill_case's)
    kv = [x for x in ",".join(sets).split(",") if x]
    hack = "".join(x.split("=", 1)[1] for x in kv if x.startswith("hack="))
    kv = [x for x in kv if not x.startswith("hack=")]
    c, img, at, spec, pos, run = decode_image(cfg, ":".join([model, wf, hf, pos] + ([",".join(kv)] if kv else [])))
    if not run:
        progs, args = img.compile_step(pos), None
    else:
        K, blocks = getattr(spec, "conv_k", 1), pos // ATTN_BLOCK + 1
        lo = max((blocks - 1) * ATTN_BLOCK, K - 1)
        progs, ra = img.compile_decode(blocks, lo)
        args = arg_words(ra, RunPos.values(0, pos, K, ATTN_BLOCK))
    if hack:
        progs = [hack_program(p, hack) for p in progs]
    return c, img, progs, at, args


def generate_case(cfg, pt: str):
    """A generate point, decode_case's at a run-time position rPOS: the image and its generate
    loop of POS's bucket (Image.compile_generate, no chain: the tokens, then the state's store),
    as the chat's engine runs it from POS (cfg, the image, the programs, where they go, POS)."""
    from opentpu.llm.qwen3 import ATTN_BLOCK
    c, img, at, spec, pos, run = decode_image(cfg, pt)
    if not run:
        raise ValueError(f"{pt}: a generate point is at a run-time position (rPOS)")
    K, blocks = getattr(spec, "conv_k", 1), pos // ATTN_BLOCK + 1
    lo = max((blocks - 1) * ATTN_BLOCK, K - 1)
    return c, img, img.compile_generate(blocks, lo, ATTN_BLOCK, chain=False), at, pos


def decode_image(cfg, pt: str):
    """decode_case's image: (cfg at the smallest power-of-two DRAM that holds it and the
    programs, the image, where the programs go, the spec, POS, at a run-time position)."""
    from opentpu.llm import load_spec, model_dir
    model, wf, hf, pos, *sets = pt.split(":")
    run = pos.startswith("r")
    pos = int(pos[1:] if run else pos)
    spec = load_spec(model_dir(model))
    for kv in filter(None, ",".join(sets).split(",")):
        k, v = kv.split("=", 1)
        spec = dataclasses.replace(spec, **{k: None if v == "-" else int(v, 0)})
    cap = 256 * (pos // 256 + 1)
    kw = dict(wformat=wf, head_format=None if hf == "-" else hf, rows=1,
              **({"lookup": True} if run else {}))
    img = spec.image(dataclasses.replace(cfg, DRAM_BYTES=1 << 40), cap, **kw)
    from opentpu.llm.qwen3 import image_end
    at = -(-(image_end(img) if getattr(img, "high", False) else img.nbytes) // 4096) * 4096
    c = dataclasses.replace(cfg, DRAM_BYTES=1 << (max(prog_end(cfg, at), img.nbytes) - 1)
                            .bit_length())
    return c, spec.image(c, cap, **kw), at, spec, pos, run


def prog_end(cfg, at: int) -> int:
    """The end of the programs and the run-time areas loaded at `at`."""
    from opentpu import waits as W
    return at + W.area(cfg.IMEM_WORDS, cfg.SEQ_WC)


def prefill_case(cfg, pt: str):
    """A prefill point MODEL:WFORMAT:HEAD:BLOCKS:T[:lLAYER][:K=V,...] (MODEL as decode_case's;
    WFORMAT prod: the family's production formats, streamed.FORMATS): the image (rows max(T,
    ACT_ROWS), the lookup tables, KV capacity the bucket's end; a MoE model's with its top-k
    expert slots a layer) and the run of T rows in bucket BLOCKS at its first position (lo:
    from conv_k - 1): a dense model's tiled prompt run (tiled.compile_tile_run, kind P), a MoE
    model's tiled expert-major layer run of layer LAYER (tiled.compile_layer_tile: its mixer and
    shared expert; the experts are the expert run's) at chunk row 0; (cfg at the smallest
    power-of-two DRAM that holds the image and the programs, the image, the programs, where they
    go (past the image, or past an Image.high's low area: decode_case's), the tpos word, T, the
    run's ARG words or None)."""
    from opentpu import waits as W
    from opentpu.compiler import arg_words
    from opentpu.llm import load_spec, model_dir
    from opentpu.llm import tiled as TL
    from opentpu.llm.qwen3 import ATTN_BLOCK, image_end
    model, wf, hf, blocks, T_, *sets = pt.split(":")
    blocks, T_ = int(blocks), int(T_)
    li = int(sets.pop(0)[1:]) if sets and sets[0][:1] == "l" and sets[0][1:].isdigit() else None
    spec = load_spec(model_dir(model))
    hack = ""
    for kv in filter(None, ",".join(sets).split(",")):
        k, v = kv.split("=", 1)
        if k == "hack":
            hack = v
            continue
        spec = dataclasses.replace(spec, **{k: None if v == "-" else int(v, 0)})
    moe = getattr(spec, "moe", None) is not None
    if moe and li is None:
        raise ValueError(f"{pt}: a MoE model's point names its layer (lLAYER)")
    cap = ATTN_BLOCK * blocks
    kw = dict(wformat=wf, head_format=None if hf == "-" else hf, rows=max(T_, cfg.act_rows),
              lookup=True)
    if wf == "prod":
        from opentpu.host.streamed import FORMATS
        fam = type(spec).__module__.rsplit(".", 1)[-1]
        kw["wformat"], kw["head_format"], kw["formats"] = FORMATS[fam]
    if moe:
        kw["experts"] = spec.moe.k
    img = spec.image(dataclasses.replace(cfg, DRAM_BYTES=1 << 40), cap, **kw)
    # the programs past the image, or past an Image.high's low area (decode_image's, as
    # BoardBackend loads them: PROG_ADDR is a 32-bit address, and past the weights' 4 GiB a
    # :bhw image's programs loaded from their address mod 2^32, zeros, NOPs to no HALT)
    at = -(-(image_end(img) if getattr(img, "high", False) else img.nbytes) // 4096) * 4096
    end = at + W.area(cfg.IMEM_WORDS, cfg.SEQ_WC)
    c = dataclasses.replace(cfg, DRAM_BYTES=1 << (max(end, img.nbytes) - 1).bit_length())
    img = spec.image(c, cap, **kw)
    t0 = (blocks - 1) * ATTN_BLOCK
    lo = max(t0, getattr(spec, "conv_k", 1) - 1)
    if moe:
        progs, ra = TL.compile_layer_tile(img, li, blocks, T_, ATTN_BLOCK)
        args = arg_words(ra, {"tpos": lo - t0, "row": 0})
    else:
        progs, _ = TL.compile_tile_run(img, blocks, T_, "P", ATTN_BLOCK)
        args = None
    if hack:
        progs = [hack_program(p, hack) for p in progs]
    return c, img, progs, at, lo - t0, T_, args


def hack_program(prog, hack: str):
    """prog with timing-only transforms (the results are wrong; the cycles bound a schedule):
    hack is MODE[/PHASE]+...: tools/vpu_hack.py's modes, or nomm (every MM a NOP: the MXU
    free) or nold (every LD / ST a NOP), each over the instructions of one phase
    (tools/prefill_phases.py) or all."""
    import prefill_phases as PP
    import vpu_hack as VH
    from opentpu import isa as I
    for item in filter(None, hack.split("+")):
        mode, _, ph = item.partition("/")
        inph = (lambda ins: True) if not ph else (lambda ins, ph=ph: PP.phase(ins) == ph)
        if mode in ("nomm", "nold"):
            ops = (I.MM,) if mode == "nomm" else (I.LD, I.ST)
            prog = [I.nop(f"dropped {ins.comment}") if ins.op in ops and inph(ins) else ins
                    for ins in prog]
        else:
            prog, _ = VH.apply(prog, mode, keep=lambda ins, inph=inph: not inph(ins))
    return prog


def em_case(cfg, pt: str, C: int | None = None, seed: int | None = None):
    """An expert-run point MODEL:ROWS[:USED[:WFORMAT]] (a Qwen3.5-MoE MODEL as decode_case's):
    expert-major's expert run of the model's first layer, in an image of MP.em_layers layers of
    its kind with MP.em_layer_slots' slots (the card's slot bound; Image.compile_expert_run, its
    passes of C entries: given, else the configuration's, moe.em_pass_rows) over a chunk of ROWS
    rows, each row k distinct experts of USED (default the prompts' measured 144 a layer),
    uniform, seed 1, the image's words as tools/moe_prefill_cosim.py em sets them (em_dram:
    every used expert landed in a slot of its own; seed: random rows, gains and weights, else
    zeros); WFORMAT (default prod: streamed.FORMATS) the experts'
    format. The programs past the image, or past an Image.high's low area (decode_case's). (cfg,
    the image, its DRAM words, the programs, where they go, the ARG words, a record of the run's
    passes and bytes, the used experts' slots)."""
    import moe_prefill_cosim as MP
    from opentpu.compiler import arg_words
    from opentpu.llm import load_spec, model_dir
    from opentpu.llm import moe as MO
    f = pt.partition(":hack=")[0].split(":")         # (emrun takes the hack: board_perf.emrun)
    model, rows = f[0], int(f[1])
    used = int(f[2]) if len(f) > 2 else MP.EM_USED["qwen35"]
    wf = f[3] if len(f) > 3 else "prod"
    spec = load_spec(model_dir(model))
    spec = dataclasses.replace(spec, kinds=spec.kinds[:1])
    mo = spec.moe
    kw = dict(rows=1, wformat=wf, head_format="int8")
    if wf == "prod":
        from opentpu.host.streamed import FORMATS
        kw["wformat"], kw["head_format"], kw["formats"] = FORMATS["qwen35"]
    ids = MP.em_ids(rows, mo.k, mo.E, used, 1)
    distinct = MP.em_passes(ids, mo.E)[1]
    big = dataclasses.replace(cfg, DRAM_BYTES=1 << 40)
    img = spec.image(big, 256, experts=mo.k, **kw)
    sc = max(rows, img.prefill_rows) * MO.em_record(spec.hidden, mo.k, MO.em_qd(img.cfg))
    nl = MP.em_layers(img.offload, sc)                  # (the first layer's kind each)
    if nl > 1:
        spec = dataclasses.replace(spec, kinds=spec.kinds[:1] * nl)
        img = spec.image(big, 256, experts=mo.k, **kw)
    n = MP.em_layer_slots(img.offload, sc, distinct, mo.k)
    img = spec.image(big, 256, experts=n, **kw)
    from opentpu.llm.qwen3 import image_end
    at = -(-(image_end(img) if getattr(img, "high", False) else img.nbytes) // 4096) * 4096
    c = dataclasses.replace(cfg, DRAM_BYTES=1 << (max(prog_end(cfg, at), img.nbytes) - 1)
                            .bit_length())
    img = spec.image(c, 256, experts=n, **kw)
    progs, ra = img.compile_expert_run(0, C)
    C, S = img.em_pass
    passes, _, loads = MP.em_passes(ids, mo.E, C, S)
    args = arg_words(ra, {"rows": rows, "entries": rows * mo.k})
    dram, es = MP.em_dram(img, 0, ids, seed)
    fb = img.fmt.nbytes
    mv = MP.em_bytes(ids, passes, loads, fb, spec.hidden)
    rec = {"rows": rows, "k": mo.k, "E": mo.E, "used": distinct, "C": C, "S": S,
           "passes": passes, "slots": n, "expert MiB": round(fb / 2**20, 3),
           "bytes": mv, "bytes a row": round(sum(mv.values()) / rows)}
    return c, img, dram, progs, at, args, rec, es


def bcuwg_transport(t, dram: int, plusargs: list, params: str):
    """bcu1525-wg's (and bcu1525-s2's) board model (tb_bcuwg through SimTransport bcuwg, as
    test_bcuwg_board's, no random stalls: the timing's), `params` (K=V,...: ARD, PRD, MWAIT, RLY,
    DMAP, ints in any base; FP=a|b|c a floorplan's DMAP and RLY, boards/bcu1525/scripts/bcu_fp.py,
    before any RLY / DMAP given) its build's parameters beside the target's."""
    from opentpu.host.board import SimTransport
    rp = T.rtl_params(t)
    p = {k: rp[k] for k in ("MCOLS", "LANES", "TMEM_WORDS", "SEQ_WC", "OUTER", "DSTEP2")}
    p |= {"WG": t.arch["WG"], "ACT_ROWS": t.arch["ACT_ROWS"], "NAT_XPOST": 1}
    p |= {k: rp[k] for k in ("ST16",) if k in rp}
    for kv in filter(None, params.split(",")):
        k, v = kv.split("=", 1)
        if k == "FP":
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "boards/bcu1525/scripts"))
            import bcu_fp
            fp = bcu_fp.plan(v, t.platform["channels_per_slice"])
            p.setdefault("DMAP", fp["dmap"])
            p.setdefault("RLY", fp["rly"])
            continue
        p[k] = int(v, 0)
    return SimTransport(ch_bytes=dram // 8, stall=0, seed=5, target=t, bcuwg=True, params=p,
                        plusargs=plusargs)


def _solo(cfg, on: bool):
    """--solo's compile configuration: one slice."""
    return dataclasses.replace(cfg, S=1) if on else cfg


def _cfg_params(t, params: str):
    """The target's configuration with --params' architecture sizes (a scratch board model's
    IMEM_WORDS / TMEM_WORDS / ACT_ROWS: what its programs compile for)."""
    p = {}
    for kv in filter(None, params.split(",")):
        k, v = kv.split("=", 1)
        if k in ("IMEM_WORDS", "TMEM_WORDS", "ACT_ROWS"):
            p[k] = int(v, 0)
    return T.config(t, **p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("benches", nargs="*", help=f"of {', '.join(BENCHES)} (default all)")
    ap.add_argument("--target", default="bcu1525-wg")
    ap.add_argument("--points", default="", help="bench=P P ...;bench=... (default each bench's)")
    ap.add_argument("--dev", help="the card (/dev/xdmaN; default OTPU_DEV, else /dev/xdma0)")
    ap.add_argument("--sim", action="store_true", help="the board model (tb_bcuwg)")
    ap.add_argument("--plus", default=",".join(SIM_PLUS), help="--sim: the simulator's plusargs")
    ap.add_argument("--params", default="",
                    help="--sim of bcu1525-wg: tb_bcuwg's parameters, K=V,... (ARD, PRD, MWAIT, RLY)")
    ap.add_argument("--cosim", action="store_true", help="the co-simulation (rtlsim, native model)")
    ap.add_argument("--ch-gbps", type=float, help="--cosim: a channel's GB/s")
    ap.add_argument("--lat-ns", type=float, help="--cosim: a channel's read latency, ns")
    ap.add_argument("--isa", action="store_true", help="the ISA simulator as the board (no time)")
    ap.add_argument("--json", help="the records, here")
    ap.add_argument("--trace", help="decode: the last run's hardware trace (keep first) to "
                    "TRACE-<i>.json, as tools/q35_trace.py dumps it; prefill: its phases")
    ap.add_argument("--solo", action="store_true",
                    help="decode, prefill and emrun at S > 1: compile each point at one slice and "
                    "run that program on every slice, the same image in each one's DRAM (no "
                    "collective: with a model of each matrix's slice share, a tensor-parallel "
                    "step's time without its GATHERs)")
    ap.add_argument("--em-pass", type=int, default=None,
                    help="emrun: passes of this many entries (default: moe.em_pass_rows')")
    ap.add_argument("--em-isa", action="store_true",
                    help="emrun: random rows, gains and experts (em_dram's seed 1), every record "
                    "checked against the ISA simulator's, bit for bit")
    a = ap.parse_args(argv)
    t = T.load(a.target)
    benches = a.benches or list(BENCHES)
    pts = dict(SIM_POINTS if a.sim or a.isa or a.cosim else POINTS)
    for item in filter(None, a.points.split(";")):
        k, v = item.split("=", 1)
        pts[k] = v
    where = "sim" if a.sim else "cosim" if a.cosim else "isa" if a.isa else "card"
    if a.trace and (a.isa or a.cosim):
        ap.error("--trace reads the hardware trace: the card or --sim, not --isa / --cosim")
    out: dict = {"target": T.full_name(t), "where": where}
    tr = None
    ends = ([wlayout(T.config(t), p.split(":")[0], int(p.split(":")[1]) << 20)["end"]
             for p in pts["wstream"].split()] + [PB_AT + (1 << 26)]
            + [dlayout(T.config(t), p)["end"] for p in pts["dstep"].split()]
            + [mlayout(T.config(t), p)["end"] for p in pts["mixed"].split()]
            + [prefill_case(_solo(_cfg_params(t, a.params), a.solo), p)[0].DRAM_BYTES
               for p in pts["prefill"].split() if "prefill" in benches])
    dec = [decode_case(dataclasses.replace(T.config(t), S=1) if a.solo else T.config(t), p)
           for p in pts["decode"].split()] if "decode" in benches else []
    em = [(e[0], e[4]) for e in (em_case(_solo(_cfg_params(t, a.params), a.solo), p, a.em_pass)
                                 for p in pts["emrun"].split())] if "emrun" in benches else []
    dec += [(c, img, None, at, None) for c, img, at, *_ in (
        decode_image(dataclasses.replace(T.config(t), S=1) if a.solo else T.config(t),
                     p.partition("@")[0]) for p in pts["generate"].split())] \
        if "generate" in benches else []
    dram = 1 << (max(ends + [c.DRAM_BYTES for c, *_ in dec + em]) - 1).bit_length()
    # bcu1525-wg's board model: dense channel images to an Image.high's programs below its
    # weights (decode_case's and em_case's at), the weights' beats past them sparse (decode's
    # never written: zeros, as the decode's zero image; an expert run's slots written there)
    dense = 1 << (max(ends + [prog_end(c, at) for c, _, _, at, _ in dec]
                      + [prog_end(c, at) for c, at in em]) - 1).bit_length()
    try:
        if a.isa or a.cosim:
            cfg = T.config(t, DRAM_BYTES=dram)
            mhz = t.platform["core_mhz"]
            if a.cosim:
                os.environ["OTPU_TARGET"] = T.full_name(t)      # (rtlsim's micro-architecture)
                for k, v in (("OTPU_PAIR", "0"), ("OTPU_DSTEP", "1"), ("OTPU_STREAM", "1"),
                             ("OTPU_SEQ_WC", "1")):
                    os.environ.setdefault(k, v)                 # (the target's)
                plus = cosim_plus(t, mhz, a.ch_gbps, a.lat_ns)
                out["plus"] = plus
                board = CosimBoard(cfg, plus)
            else:
                board = IsaBoard(cfg)
        else:
            if a.sim:
                plus = [x for x in a.plus.split(",") if x]
                if t.name not in ("bcu1525-wg", "bcu1525-s2"):
                    raise SystemExit(f"--sim: {t.name} has no board model here (bcu1525-wg, "
                                     f"bcu1525-s2: tb_bcuwg)")
                tr = bcuwg_transport(t, dense, plus, a.params)
                out["params"] = a.params
                out["plus"] = a.plus
                mhz = 1e6 / (2 * int(next((x for x in plus if x.startswith("+cp=")),
                                         "+cp=5000")[4:]))
            else:
                from opentpu.host.board import XdmaTransport, default_dev
                tr, mhz = XdmaTransport(a.dev or default_dev()), None
            board = Board(tr)
            info = board.info()
            cfg = device_config(info, DRAM_BYTES=info["cfg"]["DRAM_BYTES"])
            if mhz is None:                                     # (--dev: the card's own)
                mhz = info["core_khz"] / 1e3
        out["core MHz"] = mhz
        B = Bench(board, cfg, a.sim, a.isa, mhz, out)
        B.trace = a.trace
        B.solo = a.solo
        B.em_pass, B.em_isa = a.em_pass, a.em_isa
        for name in BENCHES:
            if name in benches:
                getattr(B, name)(pts[name].split())
        ok = not B.failed
    finally:
        if hasattr(tr, "close"):
            tr.close()
        if a.json:
            with open(a.json, "w") as f:
                json.dump(out, f, indent=1, default=str)
    print("ALL OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
