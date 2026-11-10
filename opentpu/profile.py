"""Profiling: run a kernel on the RTL with tracing and turn the trace into per-instruction
records, per-unit utilisation, DRAM-port utilisation and a roofline.

Trace events (printed by the RTL with +trace, one line each):

    T<s> D c=<cyc> s=<slot> pc=<pc> op=<op> w1=.. w2=.. w3=..   dispatched into the window
    T<s> S c=<cyc> s=<slot> u=<unit> r=<ready cycle>             started on a unit
    T<s> G c=<cyc> s=<slot>                                      MM released (may consume)
    T<s> E c=<cyc> s=<slot>                                      completed
    T<s> U c=<cyc> u=<unit> k=v ...                              unit counters for that completion
    T<s> H c=<cyc> bmxu=.. bdma=.. amxu=.. aq=..                 slice halted: port counters
    T<s> P c=<cyc> n=<cycles> bm bd am aq mx fm fq fv fc         activity of the last n cycles:
         DRAM port B (MXU/DMA), port A (MXU/QST), MXU compute, and cycles the MXU drain, QUANT,
         VPU and collective lost to TMEM bank arbitration
    T<s> Q c=<cyc> n=<cycles> bs as ms mb ff ld                   memory-side stalls of the same
         window: port B / A requests waiting for the memory, MXU starved (chunk FIFO empty) or
         blocked (chunks but no consumption), summed MXU FIFO level, loader traffic

The slice prints the lines a cycle after the fact, in a fixed order within a cycle (G, E, S, D,
U, H, P, Q), from the same signals the board's hardware trace records (opentpu/hwtrace.py
rebuilds these lines from the trace buffer; docs/observability.md).

Roofline. Per slice the DRAM burst port (B) moves one D-byte chunk per cycle and the MXU
consumes one chunk per cycle, so both the memory and the compute roof are "chunks per cycle".
The bound for a program is the number of port-B transfers it needs (MM chunks + the chunks LD/ST touch),
the port-A cycles (MM scales, one word per cycle or a PAIR MM's scale pair per cycle, + QST
bytes) and the MXU's own cycles (4-bit: one block per cycle, PAIR one chunk); the roofline is
the largest, taken over slices. Efficiency = roofline cycles / measured cycles. This is a bound
at the core's ports; dram_efficiency() compares against the DDR3 peak instead.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import isa as I
from . import rtlsim

UNITS = ["DMA", "MXU", "QUANT", "VPU", "COLL"]
OPNAMES = {I.LD: "LD", I.ST: "ST", I.DSTEP: "DSTEP", I.STREAM: "STREAM", I.LDA: "LDA", I.MM: "MM",
           I.QACT: "QACT", I.QST: "QST", I.VOP: "VOP", I.GATHER: "GATHER", I.BAR: "BAR",
           I.RLD: "RLD", I.WAITW: "WAITW"}
VFUNCS = {I.V_ADD: "add", I.V_SUB: "sub", I.V_RSUB: "rsub", I.V_MUL: "mul", I.V_MAX: "max",
          I.V_MIN: "min", I.V_COPY: "copy", I.V_EXP2: "exp2", I.V_RECIP: "recip",
          I.V_RSQRT: "rsqrt", I.V_ABS: "abs", I.V_FILL: "fill", I.V_EXP2SUB: "exp2sub",
          I.V_RSUM: "rsum", I.V_RMAX: "rmax", I.V_RSSQ: "rssq", I.V_ARGMAX: "argmax",
          I.V_SIGMOID: "sigmoid", I.V_SILU: "silu", I.V_MAC: "mac"}


# ------------------------------------------------------------------------------ DDR3 peak
DDR_CHANNELS, DDR_BYTES = 2, 8        # two 64-bit DDR3 channels (ECC bits not counted)


def ddr3_peak(mts: float) -> float:
    """Peak DRAM bandwidth of the board in bytes/s at `mts` MT/s (DDR3-1066: 17.07 GB/s)."""
    return DDR_CHANNELS * DDR_BYTES * mts * 1e6


def dram_efficiency(nbytes: float, cycles: float, mts: float, mhz: float) -> float:
    """DRAM efficiency: bytes moved per token (weights, scales, KV cache, I/O; reads and
    writes) over what the DDR3 peak moves in the token's time (cycles at `mhz`)."""
    return nbytes / (cycles / (mhz * 1e6) * ddr3_peak(mts))


def ddr3_plusargs(mts: float, mhz: float) -> list[str]:
    """Simulator arguments of the DDR3 bank model (sim/verilator/otpu_native_mem.sv) for
    DDR3-`mts` with the core at `mhz`: the controller clock is mts / 8 (4:1; LiteDRAM's sys, as
    the MIG's ui_clk was), the data-sheet tRAS / tRC / tRFC / tREFI in controller cycles, and two
    of the parameters fitted on the card at DDR3-800 (docs/board.md: tRP = tRCD = 3 controller
    cycles, a read-modify-write holds the channel 230 ns). A read/write turnaround costs 4
    controller cycles (tturn, fitted at DDR3-1066 / 120.755 MHz on the MIG bitstreams, docs/board.md
    section 4, with their per-transaction costs, which a native command does not have). The
    caller sets the read latency (tools/perf_qwen.py: 300 ns)."""
    import math
    fu = mts / 8                       # MHz
    ps = lambda f: round(1e6 / f)      # clock period in ps, the model's tick
    cyc = lambda ns: math.ceil(ns * fu / 1e3 - 1e-3)
    t = dict(tpc=ps(mhz), tpu=ps(fu), trp=3, trcd=3, tras=cyc(37.5), trc=cyc(50.625),
             trfc=cyc(160), trefi=round(7.8 * fu), trmw=round(0.23 * mhz), tturn=4)
    return ["+axi_dram=1", "+axi_map=1"] + [f"+axi_{k}={v}" for k, v in t.items()]


@dataclass
class Rec:
    """One dynamic instruction."""
    slice: int
    idx: int                 # dynamic index within the slice (dispatch order)
    pc: int
    op: int
    unit: int
    dispatch: int
    ready: int = -1
    start: int = -1
    release: int = -1        # MM: when all dependencies cleared (streaming may start earlier)
    end: int = -1
    stats: dict = field(default_factory=dict)
    name: str = ""
    detail: str = ""
    comment: str = ""
    work: int = 0            # nominal busy cycles (chunks for MM, lane-cycles for VOP ...)
    portb: int = 0           # DRAM port-B transfers it needs
    porta: int = 0           # DRAM port-A transfers it needs


def _describe(ins: I.Instr, cfg) -> tuple[str, str, int, int, int]:
    """(name, detail, nominal cycles, port-B transfers, port-A transfers) of an instruction."""
    w, D, L = ins.w, cfg.D, cfg.LANES
    burst = min(D // 4, L)
    op = ins.op
    if op in (I.LD, I.ST):
        # one TMEM segment per cycle; each D-byte chunk once on port B (chunk-aligned count)
        n = w[2]
        return OPNAMES[op], f"{n} words", -(-n // burst), -(-n // (D // 4)), 0
    if op == I.DSTEP:
        # DSTEP_LANES state words per cycle; every state chunk read (unless zero) and written
        rows, cols = w[3] & 0xFFFF, w[3] >> 16
        n = -(-rows * cols * 4 // D)
        return ("DSTEP", f"{rows}x{cols}{' zero' if ins.flags & I.F_DZERO else ''}",
                -(-rows * cols // I.DSTEP_LANES), n if ins.flags & I.F_DZERO else 2 * n, 0)
    if op == I.STREAM:
        # the shape is in the TMEM descriptor; the compiler's comment names it ("stream RxC")
        m = re.search(r"stream (\d+)x(\d+)", ins.comment)
        rows, cols = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
        n = -(-rows * cols * 4 // D)
        zero = ins.flags & I.F_SZERO
        return ("STREAM", f"{rows}x{cols}{' zero' if zero else ''}",
                -(-rows * cols // I.DSTEP_LANES), n if zero else 2 * n, 0)
    if op == I.MM:
        N, KB, M = w[3] & 0xFFFF, (w[3] >> 16) & 0xFF, (w[5] >> 16) & 0xFF
        acc = " +acc" if ins.flags & I.F_ACC else ""
        scales = 0 if ins.flags & I.F_UNIT else N * KB
        wf = (ins.flags >> I.WF_SHIFT) & 3
        if wf == I.W8:
            return "MM", f"{M}x{KB * D} . {N}x{KB * D}^T{acc}", N * KB, N * KB, scales
        # 4-bit: one block per cycle, two blocks per streamed chunk; PAIR: one chunk per cycle
        pair = bool(ins.flags & I.F_PAIR)
        return ("MM", f"{M}x{KB * D} . {N}x{KB * D}^T {'int4' if wf == I.W4I else 'fp4'}"
                      f"{' pair' if pair else ''}{acc}",
                N * (-(-KB // 2) if pair else KB), N * -(-KB // 2), scales)
    if op == I.QACT:
        rows, KB = w[1] & 0xFF, w[1] >> 16
        n = rows * KB * D
        if ins.flags & I.F_QST:                 # (ST: its rows' bytes and records to DRAM too)
            nb = rows * I.qrow_bytes(KB, D)
            return "QACT", f"{rows}x{KB * D} -> ACT + DRAM", 2 * -(-n // L), 0, -(-nb // D)
        return "QACT", f"{rows}x{KB * D} -> ACT", 2 * -(-n // L), 0, 0
    if op == I.LDA:
        # ACT RAM takes LANES bytes a cycle (a row's KB x D bytes, then a record a block)
        rows, KB = w[1] & 0xFF, w[1] >> 16
        nb = rows * I.qrow_bytes(KB, D)
        return "LDA", f"{rows}x{KB * D} DRAM -> ACT", rows * KB * (D // (2 * L) + 1), \
            -(-nb // D), 0
    if op == I.QST:
        rows, KB = w[3] & 0xFFFF, w[3] >> 16
        n = rows * KB * D
        nw = n // 2 if ins.flags & I.F_HALF else n     # HALF: the rows' first halves written
        return "QST", f"{rows}x{KB * D} -> DRAM int8", -(-n // L) + nw, 0, nw + rows * KB
    if op == I.VOP:
        rows, cols = w[3] & 0xFFFF, w[3] >> 16
        func = (w[5] >> 16) & 0xFF
        return (f"VOP.{VFUNCS.get(func, func)}", f"{rows}x{cols}", rows * -(-cols // L), 0, 0)
    if op == I.GATHER:
        rows, cols = w[2] & 0xFFFF, w[2] >> 16
        return "GATHER", f"{rows}x{cols} x S", cfg.S * rows * -(-cols // L), 0, 0
    if op == I.BAR:
        return "BAR", "", 0, 0, 0
    if op == I.RLD:
        return "RLD", " ".join(f for b, f in ((I.F_RAW, "raw"), (I.F_MUL, "mul"))
                               if ins.flags & b), 1, 0, 0
    if op == I.WAITW:                   # one DRAM word per poll, one TMEM word; the wait is the host's
        return "WAITW", ("EQ", "NE", "GE", "?")[ins.flags & 3], 1, 1, 0
    return f"op{op:#x}", "", 0, 0, 0


@dataclass
class Profile:
    name: str
    cfg: object
    cycles: int
    recs: list[Rec]
    ports: list[dict]          # per slice: bmxu, bdma, amxu, aq, c
    programs: list

    # ------------------------------------------------------------------ derived metrics
    def slice_recs(self, s: int) -> list[Rec]:
        return [r for r in self.recs if r.slice == s]

    def roofline(self) -> dict:
        """Lower bound on cycles from the DRAM ports (and the MXU, which shares their rate)."""
        per = []
        for s in range(self.cfg.S):
            rs = self.slice_recs(s)
            b = sum(r.portb for r in rs)
            a = sum(r.porta for r in rs)
            # port A delivers one scale word per MXU cycle, a PAIR MM's 8-byte scale pair at once
            ac = sum(r.porta // 2 if r.detail.endswith((" pair", " pair +acc")) else r.porta
                     for r in rs)
            m = sum(r.work for r in rs if r.op == I.MM)
            per.append({"portb": b, "porta": a, "porta_cycles": ac, "mxu": m,
                        "bound": max(ac, b, m)})
        bound = max(p["bound"] for p in per) if per else 0
        return {"bound": bound, "per_slice": per,
                "efficiency": bound / self.cycles if self.cycles else 0.0}

    def unit_busy(self, s: int = 0) -> dict:
        """Cycles each unit had an instruction in flight (union of intervals)."""
        out = {}
        for u, name in enumerate(UNITS):
            iv = sorted((r.start, r.end) for r in self.slice_recs(s) if r.unit == u and r.start >= 0)
            tot, cur_s, cur_e = 0, None, None
            for a, b in iv:
                if cur_e is None or a > cur_e:
                    if cur_e is not None:
                        tot += cur_e - cur_s
                    cur_s, cur_e = a, b
                else:
                    cur_e = max(cur_e, b)
            if cur_e is not None:
                tot += cur_e - cur_s
            out[name] = tot
        return out

    def by_class(self, s: int = 0) -> dict:
        agg = {}
        for r in self.slice_recs(s):
            k = r.name
            a = agg.setdefault(k, {"count": 0, "busy": 0, "wait_dep": 0, "wait_unit": 0,
                                   "work": 0})
            a["count"] += 1
            a["busy"] += r.end - r.start
            a["wait_dep"] += max(0, r.ready - r.dispatch)
            a["wait_unit"] += max(0, r.start - r.ready)
            a["work"] += r.work
        return agg

    def macs(self) -> int:
        """Useful multiply-accumulates: every MM does M x N x KB*D."""
        tot = 0
        for r in self.recs:
            if r.op == I.MM:
                ins = self.programs[r.slice][r.pc]
                N, KB, M = ins.w[3] & 0xFFFF, (ins.w[3] >> 16) & 0xFF, (ins.w[5] >> 16) & 0xFF
                tot += M * N * KB * self.cfg.D
        return tot

    def mxu_gaps(self, s: int = 0) -> dict:
        """Where the MXU was not consuming: before its first MM could consume, after its last
        MM, and gaps in between -- each gap blamed on the instruction that finished last before
        the MM was released (its dependency)."""
        rs = self.slice_recs(s)
        mms = sorted((r for r in rs if r.op == I.MM and r.end >= 0), key=lambda r: r.start)
        if not mms:
            return {"prologue": self.cycles, "epilogue": 0, "gaps": [], "blame": {}}
        def rel(r):
            return max(r.release if r.release >= 0 else r.start, r.start)
        busy_to = rel(mms[0])
        gaps, blame = [], {}
        prologue = rel(mms[0])
        for prev, m in zip(mms, mms[1:]):
            busy_to = max(busy_to, prev.end)
            g = rel(m) - busy_to
            if g > 4:
                cands = [r for r in rs if r.op != I.MM and 0 <= r.end <= rel(m) + 1]
                who = max(cands, key=lambda r: r.end) if cands else None
                gaps.append((busy_to, rel(m), who.idx if who else -1))
                key = (who.name, who.pc) if who else ("?", -1)
                blame[key] = blame.get(key, 0) + g
        epilogue = self.cycles - max(m.end for m in mms)
        return {"prologue": prologue, "epilogue": epilogue, "gaps": gaps, "blame": blame}

    def phases(self, phase_of, s: int = 0) -> dict:
        """Split the run into phases: phase_of(rec) names the phase of each instruction; the
        instructions in dispatch order form runs of one phase, and a run is charged the cycles
        from the latest end before it to the latest end within it (so the phases' cycles add up
        to the run's). Per phase also its DRAM transfers and its VPU and MXU busy time (union of
        the phase's own instructions' intervals)."""
        out: dict = {}
        done = 0
        for r in sorted(self.slice_recs(s), key=lambda r: r.idx):
            a = out.setdefault(phase_of(r), {"cycles": 0, "portb": 0, "porta": 0,
                                             "iv": {1: [], 3: []}})
            a["portb"] += r.portb
            a["porta"] += r.porta
            if r.unit in a["iv"] and r.start >= 0:
                a["iv"][r.unit].append((r.start, r.end))
            if r.end > done:
                a["cycles"] += r.end - done
                done = r.end
        for a in out.values():
            for u, iv in a.pop("iv").items():
                tot, cur_s, cur_e = 0, None, None
                for x, y in sorted(iv):
                    if cur_e is None or x > cur_e:
                        tot += 0 if cur_e is None else cur_e - cur_s
                        cur_s, cur_e = x, y
                    else:
                        cur_e = max(cur_e, y)
                a[UNITS[u]] = tot + (0 if cur_e is None else cur_e - cur_s)
        return out

    def loop_waits(self, s: int = 0) -> dict:
        """The cycles a hardware loop's TMEM writers wait on its previous iteration (slice s).
        Next-fit placement wraps within a loop's body, so a body tile can take the words an
        instruction before it in the body writes; that instruction, an iteration later, waits
        for the tile's last reader or writer (the scoreboard's WAR / WAW, which the data does
        not need). For each record of iteration k >= 1 that writes TMEM: the cycles from its
        dispatch to the last release (else end) of the records of iteration k - 1 whose TMEM
        reads or writes overlap its writes, at most to its ready. Returns {"loops": per LOOP
        {"loop", "body", "iterations", "wait_ld", "wait_all", "by_pc": {pc: {"name", "detail",
        "wait", "n", "blocker": {pc: cycles}}}}, "wait_ld", "wait_all", "unresolved": records
        with a TMEM field a register moves, not resolved}."""
        from . import waits as W
        prog, R0 = self.programs[s], [(0, 0)] * 16
        regs = {I.LD: ("rb",), I.ST: ("rb",), I.VOP: ("ra", "rb", "rc", "rd"), I.MM: ("rc",),
                I.QACT: ("ra",), I.GATHER: ("ra", "rb"), I.DSTEP: ("rb", "rc", "rd")}
        fps: dict = {}

        def tm(pc):            # (reads, writes) as TMEM [lo, hi) lists; None: a register moves one
            if pc not in fps:
                ins = prog[pc]
                if any(getattr(ins, r, 0) for r in regs.get(ins.op, ())):
                    fps[pc] = None
                else:
                    f = W.footprint(ins, R0, self.cfg.D, self.cfg.S)
                    fps[pc] = tuple([(lo, hi) for sp, lo, hi in x if sp == W.SP_TMEM]
                                    for x in (f.rd, f.wr))
            return fps[pc]

        recs = sorted(self.slice_recs(s), key=lambda r: r.idx)
        out: dict = {"loops": [], "unresolved": 0}
        for lp in (pc for pc, ins in enumerate(prog) if ins.op == I.LOOP):
            b0, b1 = lp + 1, lp + 1 + (prog[lp].w[0] & 0xFFFF)
            body = [r for r in recs if b0 <= r.pc < b1]
            if not body:
                continue
            first = min(r.pc for r in body)
            at = [i for i, r in enumerate(body) if r.pc == first]
            its = [body[a:b] for a, b in zip(at, at[1:] + [len(body)])]
            per: dict = {}
            for prev, cur in zip(its, its[1:]):
                for r in cur:
                    f = tm(r.pc)
                    if f is None:
                        out["unresolved"] += 1
                        continue
                    if not f[1]:
                        continue
                    t, who = -1, None
                    for e in prev:
                        g = tm(e.pc)
                        if g and any(a < d and c < b for a, b in f[1] for c, d in g[0] + g[1]):
                            u = e.release if e.release >= 0 else e.end
                            if u > t:
                                t, who = u, e.pc
                    w = max(0, min(r.ready, t) - r.dispatch) if who is not None else 0
                    if w:
                        x = per.setdefault(r.pc, {"name": r.name, "detail": r.detail, "wait": 0,
                                                  "n": 0, "blocker": {}})
                        x["wait"] += w
                        x["n"] += 1
                        x["blocker"][who] = x["blocker"].get(who, 0) + w
            out["loops"].append({
                "loop": lp, "body": [b0, b1], "iterations": len(its),
                "wait_ld": sum(x["wait"] for x in per.values() if x["name"] == "LD"),
                "wait_all": sum(x["wait"] for x in per.values()),
                "by_pc": dict(sorted(per.items(), key=lambda kv: (kv[1]["name"] != "LD",
                                                                  -kv[1]["wait"])))})
        out["wait_ld"] = sum(x["wait_ld"] for x in out["loops"])
        out["wait_all"] = sum(x["wait_all"] for x in out["loops"])
        return out

    def summary(self) -> str:
        rl = self.roofline()
        lines = [f"{self.name}: {self.cycles} cycles, roofline {rl['bound']} "
                 f"({100 * rl['efficiency']:.1f}% of roofline)"]
        busy = self.unit_busy(0)
        lines.append("  slice 0 unit busy: " + "  ".join(
            f"{k} {100 * v / self.cycles:.0f}%" for k, v in busy.items() if v))
        p = self.ports[0]
        lines.append(f"  slice 0 DRAM port B busy {100 * (p['bmxu'] + p['bdma']) / self.cycles:.0f}%"
                     f" (MXU {p['bmxu']}, DMA {p['bdma']})")
        return "\n".join(lines)


_EV = re.compile(r"^T(\d+) ([DSEUHGPQ]) c=(\d+)(.*)$")


def parse(trace: str, cfg, programs, name: str = "") -> Profile:
    recs: list[Rec] = []
    open_slot: dict[tuple[int, int], Rec] = {}
    unit_done: dict[tuple[int, int], list[Rec]] = {}
    unit_stats: dict[tuple[int, int], list[dict]] = {}
    ports = [dict() for _ in range(cfg.S)]
    buckets = [dict() for _ in range(cfg.S)]
    counts = [0] * cfg.S
    last = 0
    for line in trace.splitlines():
        m = _EV.match(line.strip())
        if not m:
            continue
        s, kind, c = int(m.group(1)), m.group(2), int(m.group(3))
        kv = dict(re.findall(r"(\w+)=([0-9a-fA-F]+)", m.group(4)))
        last = max(last, c)
        if kind == "D":
            slot, pc, op = int(kv["s"]), int(kv["pc"]), int(kv["op"], 16)
            ins = programs[s][pc]
            nm, det, work, pb, pa = _describe(ins, cfg)
            unit = {I.LD: 0, I.ST: 0, I.DSTEP: 0, I.STREAM: 0, I.WAITW: 0, I.LDA: 0, I.MM: 1,
                    I.QACT: 2, I.QST: 2, I.VOP: 3}.get(op, 4)
            r = Rec(s, counts[s], pc, op, unit, c, name=nm, detail=det, comment=ins.comment,
                    work=work, portb=pb, porta=pa)
            counts[s] += 1
            recs.append(r)
            open_slot[(s, slot)] = r
        elif kind == "S":
            r = open_slot[(s, int(kv["s"]))]
            r.start, r.ready = c, int(kv["r"])
        elif kind == "G":
            open_slot[(s, int(kv["s"]))].release = c
        elif kind == "E":
            r = open_slot.pop((s, int(kv["s"])))
            r.end = c
            unit_done.setdefault((s, r.unit), []).append(r)
        elif kind == "U":
            unit_stats.setdefault((s, int(kv["u"])), []).append(
                {k: int(v) for k, v in kv.items() if k != "u"})
        elif kind == "H":
            ports[s] = {k: int(v) for k, v in kv.items()}
            ports[s]["c"] = c
        elif kind == "P":
            b = buckets[s]
            b.setdefault("c", []).append(c)
            for k, v in kv.items():
                b.setdefault(k, []).append(int(v))
        elif kind == "Q":                       # same windows as P: add its counters
            b = buckets[s]
            for k, v in kv.items():
                if k != "n":
                    b.setdefault(k, []).append(int(v))
    # attach unit counters in completion order
    for key, rs in unit_done.items():
        for r, st in zip(sorted(rs, key=lambda x: x.end), unit_stats.get(key, [])):
            r.stats = st
    p = Profile(name, cfg, last, recs, ports, programs)
    p.buckets = buckets
    return p


def profile(kernel, cfg, name: str = "", dram_lat: int = 8, uarch: dict | None = None,
            run_kw: dict | None = None, **args) -> Profile:
    from .runtime import compile_kernel
    comp, imgs = compile_kernel(kernel, cfg, **args)
    _, _, st = rtlsim.run(cfg, comp.programs, imgs, dram_lat=dram_lat, trace=True, uarch=uarch,
                          **(run_kw or {}))
    p = parse(st["trace"], cfg, comp.programs, name or kernel.__name__)
    p.cycles = st["cycles"]
    return p
