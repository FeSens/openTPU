"""openTPU Lens: record profiles of openTPU runs into files and explore them in a browser app.

Profile files (`.otpuprof`) are gzip-compressed JSON (plain JSON is accepted too):

    {"format": "openTPU-profile", "version": 1, "created": ..., "tool": ...,
     "profiles": [<profile>, ...]}

A <profile> is one run of one workload (see to_data for every field):
    kind       "rtl" (cycle-accurate RTL trace), "isa" (ISA simulator: executed instructions
               with an analytic serial timing), or "board" (counters read from the card)
    config     the machine configuration and micro-architecture knobs
    programs   per slice, the static program: [pc, name, detail, comment, source id]
    instrs     per dynamic instruction: [slice, idx, pc, unit, name, detail, dispatch, ready,
               release, start, end, source id, unit counters, nominal work, port-B, port-A]
    slices     per slice: DRAM port totals, unit busy cycles, TMEM-arbitration losses and the
               bucketed counters (P and Q trace lines: see profile.py)
    roofline   DRAM-bound minimum cycles and the achieved fraction
    sources    kernel source lines the instructions came from
    board      counter snapshot from the card (kind "board")

Command line (also installed as `lens`):

    python -m opentpu.lens record mlp -o mlp.otpuprof          # RTL trace of a workload
    python -m opentpu.lens record qwen-tiny --board -o q.otpuprof
    python -m opentpu.lens record mlp --isa -o mlp_isa.otpuprof
    python -m opentpu.lens open q.otpuprof                     # the app in the browser
    python -m opentpu.lens html q.otpuprof -o q.html           # standalone page
    python -m opentpu.lens info q.otpuprof
    python -m opentpu.lens list                                # workloads

The app (lens_app.html) has an overview (roofline, where the cycles went, utilisation), a
zoomable timeline, a floorplan of the machine that replays the run with data movement and unit
states, and per-instruction and per-source-line tables. It also opens any profile file by
drag and drop, so the HTML works on its own.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import http.server
import json
import linecache
import os
import socketserver
import sys
import threading
import webbrowser
from pathlib import Path

from . import isa as I
from . import rtlsim
from .profile import UNITS, OPNAMES, Profile, _describe

FORMAT, VERSION = "openTPU-profile", 1
CLOCK_MHZ = 100          # board core clock
APP = Path(__file__).with_name("lens_app.html")
ROOT = Path(__file__).resolve().parent.parent


def _rel(path: str) -> str:
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:
        return path


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


# ============================================================================ analysis notes
def notes(p: Profile) -> list[dict]:
    """Bottleneck analysis in plain sentences, most important first."""
    out = []
    rl = p.roofline()
    eff = rl["efficiency"]
    cyc = p.cycles
    b = p.ports[0] if p.ports and p.ports[0] else {}
    portb = (b.get("bmxu", 0) + b.get("bdma", 0)) / cyc if cyc else 0
    gaps = p.mxu_gaps(0)
    if eff >= 0.95:
        out.append({"level": "good", "text":
                    f"At the roofline: {_pct(eff)} of the DRAM-bound minimum. DRAM port B was "
                    f"busy {_pct(portb)} of the run on slice 0."})
    elif eff >= 0.85:
        out.append({"level": "warn", "text":
                    f"Near the roofline: {_pct(eff)} of the DRAM-bound minimum "
                    f"({cyc - rl['bound']} cycles above it)."})
    else:
        out.append({"level": "bad", "text":
                    f"Below the roofline: {_pct(eff)} of the DRAM-bound minimum. DRAM port B "
                    f"idled {_pct(1 - portb)} of the run."})
    lost = cyc - rl["bound"]
    if lost > 0:
        parts = []
        if gaps["prologue"]:
            parts.append(f"{gaps['prologue']} before the MXU could consume its first stream")
        gsum = sum(e - s_ for s_, e, _ in gaps["gaps"])
        if gsum:
            parts.append(f"{gsum} in MXU gaps mid-run")
        if gaps["epilogue"]:
            parts.append(f"{gaps['epilogue']} after the last matmul")
        if parts:
            out.append({"level": "info", "text": "Cycles off the stream (slice 0): "
                        + ", ".join(parts) + "."})
    by_loc = {}
    for (name, pc), g in gaps["blame"].items():
        if pc < 0:
            continue
        key = (name, _loc(p.programs[0][pc]))
        by_loc[key] = by_loc.get(key, 0) + g
    for (name, loc), g in sorted(by_loc.items(), key=lambda kv: -kv[1])[:3]:
        if g >= 0.01 * cyc:
            out.append({"level": "warn", "text":
                        f"The MXU waited {g} cycles ({_pct(g / cyc)}) for {name} at {loc}."})
    lose = _losses(p, 0)
    for u, n in sorted(lose.items(), key=lambda kv: -kv[1]):
        if n > 0.02 * cyc:
            out.append({"level": "warn", "text":
                        f"The {u} lost {n} cycles ({_pct(n / cyc)}) to TMEM bank arbitration: "
                        f"another unit held the banks it needed."})
    bk = p.buckets[0] if getattr(p, "buckets", None) else {}
    for key, what in (("ms", "the MXU had work but its chunk FIFO was empty (DRAM latency or "
                              "bandwidth)"),
                      ("bs", "a DRAM port B request waited for the memory")):
        n = sum(bk.get(key, []))
        if n > 0.02 * cyc:
            out.append({"level": "warn", "text": f"For {n} cycles ({_pct(n / cyc)}) {what}."})
    busy = p.unit_busy(0)
    if busy.get("VPU", 0) > 0.85 * cyc and eff < 0.95:
        out.append({"level": "bad", "text":
                    "The VPU is busy almost all the time: the run is vector-bound. Fuse "
                    "elementwise work into the MXU/quantizer epilogues or widen the VPU."})
    if b and b.get("bdma", 0) > 0.1 * cyc:
        out.append({"level": "info", "text":
                    f"The DMA used {_pct(b['bdma'] / cyc)} of DRAM port B for loads and "
                    f"stores; that traffic counts toward the roofline."})
    return out


def _loc(ins) -> str:
    if not ins.src:
        return "(compiler)"
    f, line, fn = ins.src[0]
    return f"{os.path.basename(f)}:{line} in {fn}"


def _losses(p: Profile, s: int) -> dict:
    b = p.buckets[s] if getattr(p, "buckets", None) else {}
    return {"MXU drain": sum(b.get("fm", [])), "QUANT": sum(b.get("fq", [])),
            "VPU": sum(b.get("fv", [])), "COLL": sum(b.get("fc", []))}


# ============================================================================ profile data
class _Sources:
    def __init__(self):
        self.list, self.index = [], {}

    def id(self, ins) -> int:
        if not ins.src:
            return -1
        key = ins.src[0]              # group by the innermost line; keep its first call chain
        if key not in self.index:
            f, line, fn = ins.src[0]
            self.index[key] = len(self.list)
            self.list.append({"file": _rel(f), "line": line, "func": fn,
                              "text": linecache.getline(f, line).strip(),
                              "chain": [f"{os.path.basename(a)}:{b} {c}" for a, b, c in ins.src]})
        return self.index[key]


def _config(cfg, uarch: dict | None) -> dict:
    ua = dict(rtlsim.UARCH)
    ua.update(uarch or {})
    lanes = cfg.LANES
    return {"S": cfg.S, "D": cfg.D, "MCOLS": cfg.MCOLS, "LANES": lanes,
            "CL": lanes // 4 if lanes >= 8 else 1, "ACT_BLOCKS": cfg.ACT_BLOCKS,
            "TMEM_WORDS": cfg.TMEM_WORDS, "DRAM_BYTES": cfg.DRAM_BYTES,
            "IMEM_WORDS": cfg.IMEM_WORDS, "FIFO_DEPTH": 128, "NRP": 8, "NWP": 4, **ua}


def _programs(programs, cfg, src: _Sources) -> list:
    out = []
    for prog in programs:
        rows = []
        for pc, ins in enumerate(prog):
            if ins.op in OPNAMES or ins.op == I.VOP:
                nm, det, *_ = _describe(ins, cfg)
            else:
                nm, det = {I.HALT: "HALT", I.NOP: "NOP", I.LI: "LI", I.ADDI: "ADDI",
                           I.LOOP: "LOOP"}.get(ins.op, f"op{ins.op:#x}"), ""
                if ins.op == I.LOOP:
                    det = f"body {ins.w[0]} x {ins.w[1]}{' + R%d' % ins.ra if ins.ra else ''}"
                elif ins.op in (I.LI, I.ADDI):
                    det = f"R{ins.rd} = {'R%d + ' % ins.ra if ins.op == I.ADDI else ''}{ins.w[0]}"
            rows.append([pc, nm, det, ins.comment or "", src.id(ins), list(ins.w),
                         [ins.ra, ins.rb, ins.rc, ins.rd], ins.flags, ins.op])
        out.append(rows)
    return out


def to_data(p: Profile, uarch: dict | None = None, kind: str = "rtl") -> dict:
    """Everything the app needs about one RTL-traced run, as plain JSON."""
    cfg = p.cfg
    rl = p.roofline()
    src = _Sources()
    instrs = []
    for r in p.recs:
        ins = p.programs[r.slice][r.pc]
        instrs.append([r.slice, r.idx, r.pc, r.unit, r.name, r.detail, r.dispatch, r.ready,
                       r.release, r.start, r.end, src.id(ins), r.stats, r.work, r.portb, r.porta])
    slices = []
    for s in range(cfg.S):
        slices.append({"ports": p.ports[s], "busy": p.unit_busy(s), "lose": _losses(p, s),
                       "buckets": p.buckets[s] if getattr(p, "buckets", None) else {}})
    macs = p.macs()
    bytes_b = sum(x["portb"] for x in rl["per_slice"]) * cfg.D
    return {
        "kind": kind, "name": p.name, "cycles": p.cycles, "clock_mhz": CLOCK_MHZ,
        "config": _config(cfg, uarch),
        "roofline": {"bound": rl["bound"], "efficiency": rl["efficiency"],
                     "per_slice": rl["per_slice"]},
        "macs": macs, "bytes": bytes_b,
        "peak_macs": cfg.D * cfg.MCOLS * cfg.S, "peak_bytes": cfg.D * cfg.S,
        "units": UNITS, "slices": slices, "instrs": instrs,
        "programs": _programs(p.programs, cfg, src), "sources": src.list,
        "notes": notes(p), "mxu_gaps": {k: v for k, v in p.mxu_gaps(0).items() if k != "blame"},
    }


def isa_data(name: str, cfg, programs: list, images: list, uarch: dict | None = None) -> dict:
    """An ISA-simulator run: the executed instruction stream (with loops unrolled) and an
    analytic timing -- every instruction issued back to back for its nominal work, no overlap
    (an upper bound; the RTL overlaps units). Useful when the RTL is too slow to run."""
    from .isasim import Machine
    m = Machine(cfg, programs, [None if i is None else i.copy() for i in images])
    seqs = [[] for _ in range(cfg.S)]
    for s, sl in enumerate(m.slices):
        orig = sl.execute

        def execute(ins, _orig=orig, _s=s, _sl=sl):
            seqs[_s].append(_sl.pc)
            return _orig(ins)
        sl.execute = execute
    m.run()
    src = _Sources()
    unit_of = {I.LD: 0, I.ST: 0, I.MM: 1, I.QACT: 2, I.QST: 2, I.VOP: 3, I.GATHER: 4}
    instrs, cycles, per = [], 0, []
    for s in range(cfg.S):
        t, b, a = 0, 0, 0
        for idx, pc in enumerate(seqs[s]):
            ins = programs[s][pc]
            if ins.op not in unit_of:
                continue
            nm, det, work, pb, pa = _describe(ins, cfg)
            work = max(1, work)
            instrs.append([s, idx, pc, unit_of[ins.op], nm, det, t, t, -1, t, t + work,
                           src.id(ins), {}, work, pb, pa])
            t += work
            b += pb
            a += pa
        cycles = max(cycles, t)
        per.append({"portb": b, "porta": a, "bound": max(a, b)})
    bound = max((x["bound"] for x in per), default=0)
    busy = []
    for s in range(cfg.S):
        u = {n: 0 for n in UNITS}
        for r in instrs:
            if r[0] == s:
                u[UNITS[r[3]]] += r[10] - r[9]
        busy.append(u)
    return {
        "kind": "isa", "name": name, "cycles": cycles, "clock_mhz": CLOCK_MHZ,
        "config": _config(cfg, uarch),
        "roofline": {"bound": bound, "efficiency": bound / cycles if cycles else 0,
                     "per_slice": per},
        "macs": 0, "bytes": sum(x["portb"] for x in per) * cfg.D,
        "peak_macs": cfg.D * cfg.MCOLS * cfg.S, "peak_bytes": cfg.D * cfg.S,
        "units": UNITS,
        "slices": [{"ports": {}, "busy": busy[s], "lose": {}, "buckets": {}}
                   for s in range(cfg.S)],
        "instrs": instrs, "programs": _programs(programs, cfg, src), "sources": src.list,
        "notes": [{"level": "info", "text":
                   "ISA simulator run: analytic timing (each instruction back to back for its "
                   "nominal work, no overlap between units). The roofline fraction is a lower "
                   "bound on what the RTL achieves; record with the RTL for real timing."}],
        "mxu_gaps": {},
    }


def board_data(name: str, cfg, programs: list, stats: dict, uarch: dict | None = None) -> dict:
    """Counters read from the card after a run (host/board.py Board.run)."""
    src = _Sources()
    per = []
    for prog in programs:
        b = a = 0
        for ins in prog:
            if ins.op in OPNAMES or ins.op == I.VOP:
                _, _, _, pb, pa = _describe(ins, cfg)
                b, a = b + pb, a + pa
        per.append({"portb": b, "porta": a, "bound": max(a, b)})
    cyc = int(stats.get("cycles", 0))
    bound = int(stats.get("b_reads", 0)) + int(stats.get("b_writes", 0))
    return {
        "kind": "board", "name": name, "cycles": cyc, "clock_mhz": CLOCK_MHZ,
        "config": _config(cfg, uarch),
        "roofline": {"bound": bound, "efficiency": bound / cyc if cyc else 0,
                     "per_slice": per, "note": "bound = port-B transfers counted by the card"},
        "macs": 0, "bytes": bound * cfg.D, "peak_macs": cfg.D * cfg.MCOLS * cfg.S,
        "peak_bytes": cfg.D * cfg.S, "units": UNITS,
        "slices": [{"ports": {"bmxu": stats.get("b_reads", 0), "bdma": stats.get("b_writes", 0),
                              "amxu": stats.get("a_reads", 0), "aq": stats.get("a_writes", 0),
                              "c": cyc}, "busy": {}, "lose": {}, "buckets": {}}],
        "instrs": [], "programs": _programs(programs, cfg, src), "sources": src.list,
        "notes": [{"level": "info", "text": "Board run: totals from the card's counters "
                   "(no per-instruction timeline)."}],
        "mxu_gaps": {}, "board": dict(stats),
    }


# ============================================================================ files
def save(profiles: list[dict], path) -> Path:
    """Write profile dicts (to_data / isa_data / board_data) to a .otpuprof file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"format": FORMAT, "version": VERSION,
           "created": _dt.datetime.now().isoformat(timespec="seconds"),
           "tool": "openTPU Lens", "profiles": profiles}
    raw = json.dumps(doc, separators=(",", ":")).encode()
    with gzip.open(path, "wb", compresslevel=6) as f:
        f.write(raw)
    return path


def load(path) -> dict:
    """Read a profile file (gzip or plain JSON); checks format and version."""
    raw = Path(path).read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    doc = json.loads(raw)
    if doc.get("format") != FORMAT:
        raise ValueError(f"{path}: not an openTPU profile")
    if int(doc.get("version", 0)) > VERSION:
        raise ValueError(f"{path}: profile version {doc['version']} is newer than this Lens "
                         f"({VERSION})")
    return doc


# ============================================================================ the app
def render(doc_or_profiles, uarch: dict | None = None, title: str = "openTPU Lens") -> str:
    """The app as one standalone HTML page with the profiles embedded."""
    if isinstance(doc_or_profiles, dict):
        doc = doc_or_profiles
    else:
        profs = [x if isinstance(x, dict) else to_data(x, uarch) for x in doc_or_profiles]
        doc = {"format": FORMAT, "version": VERSION, "profiles": profs}
    payload = json.dumps(doc, separators=(",", ":")).replace("</", "<\\/")
    html = APP.read_text()
    return html.replace("/*__DATA__*/null", payload).replace("__TITLE__", title)


def write_report(profiles, path, uarch: dict | None = None, standalone: bool = True,
                 title: str = "openTPU Lens") -> Path:
    """Standalone HTML report of Profile objects or profile dicts (kept for tools/lens.py)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(profiles, uarch, title))
    return path


def serve(path, port: int = 0, open_browser: bool = True, block: bool = True):
    """Serve the app and the profile file at http://127.0.0.1:<port>/ (the app fetches
    /profile). Returns the server (and runs forever when `block`)."""
    path = Path(path) if path else None
    app = APP.read_text().replace("__TITLE__", "openTPU Lens")
    if path:
        app = app.replace("/*__URL__*/null", json.dumps("/profile"))

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body, ctype = app.encode(), "text/html; charset=utf-8"
            elif self.path == "/profile" and path:
                body, ctype = path.read_bytes(), "application/octet-stream"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    socketserver.TCPServer.allow_reuse_address = True
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", port), H)
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    srv.url = url
    if not block:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv
    print(f"openTPU Lens at {url}  (Ctrl-C to stop)", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return srv


# ============================================================================ workloads
def _args_mod():
    sys.path.insert(0, str(ROOT / "tests"))
    import test_kernels
    return test_kernels


def _tiny_qwen():
    """A small random Qwen3 (2 layers, hidden 256, 4 q / 2 kv heads of 128, vocab 1000)."""
    import numpy as np
    from .llm.qwen3 import Spec
    spec = Spec(256, 2, 4, 2, 128, 512, 1000)
    rng = np.random.default_rng(0)
    H, d, F, V = 256, 128, 512, 1000
    W = {"model.embed_tokens.weight": rng.normal(0, 0.02, (V, H)).astype(np.float32),
         "model.norm.weight": (1 + 0.1 * rng.normal(size=H)).astype(np.float32)}
    for i in range(2):
        p = f"model.layers.{i}."
        W[p + "input_layernorm.weight"] = (1 + 0.1 * rng.normal(size=H)).astype(np.float32)
        W[p + "post_attention_layernorm.weight"] = (1 + 0.1 * rng.normal(size=H)).astype(np.float32)
        W[p + "self_attn.q_norm.weight"] = (1 + 0.1 * rng.normal(size=d)).astype(np.float32)
        W[p + "self_attn.k_norm.weight"] = (1 + 0.1 * rng.normal(size=d)).astype(np.float32)
        for n, shp in (("self_attn.q_proj", (4 * d, H)), ("self_attn.k_proj", (2 * d, H)),
                       ("self_attn.v_proj", (2 * d, H)), ("self_attn.o_proj", (H, 4 * d)),
                       ("mlp.gate_proj", (F, H)), ("mlp.up_proj", (F, H)),
                       ("mlp.down_proj", (H, F))):
            W[p + n + ".weight"] = rng.normal(0, 0.05, shp).astype(np.float32)
    return spec, W


class _Capture:
    """Engine backend: ISA simulator for every token but the last, which runs on the RTL with
    tracing; keeps (programs, trace, stats)."""

    def __init__(self, cfg, images, uarch, run_kw):
        from .llm.qwen3 import IsaBackend
        self.cfg, self.uarch, self.run_kw = cfg, uarch, run_kw
        self.isa = IsaBackend(cfg, images)
        self.traced = False
        self.last = None

    def write(self, s, addr, data):
        self.isa.write(s, addr, data)

    def read(self, s, addr, n):
        return self.isa.read(s, addr, n)

    def run(self, programs):
        if not self.traced:
            return self.isa.run(programs)
        n = [len(sl.dram) for sl in self.isa.machine.slices]
        drams, _, st = rtlsim.run(self.cfg, programs, [sl.dram for sl in self.isa.machine.slices],
                                  max_cycles=1 << 40, trace=True, uarch=self.uarch,
                                  **self.run_kw)
        for sl, d, k in zip(self.isa.machine.slices, drams, n):
            sl.dram[:] = d[:k]
        self.last = (programs, st)
        return st


def _qwen(real: bool, pos: int, board: bool, run_kw: dict, uarch):
    import numpy as np
    from .isasim import board_config
    from .llm.qwen3 import Engine, Spec, load_weights, device_config
    from .profile import parse
    if real:
        model = ROOT / "models" / "Qwen3-0.6B"
        spec, W = Spec.from_hf(model), load_weights(model)
        cap = max(256, pos + 1)
    else:
        spec, W = _tiny_qwen()
        cap = max(256, pos + 1)
    if board:
        from .llm.qwen3 import Image
        probe = Image(spec, board_config(DRAM_BYTES=1 << 40), cap)
        size = 1 << max(20, (probe.nbytes - 1).bit_length())
        cfg = board_config(DRAM_BYTES=size)
    else:
        cfg = device_config(spec, cap)
    holder = {}

    def mk(c, images):
        holder["b"] = _Capture(c, images, uarch, run_kw)
        return holder["b"]
    eng = Engine(spec, W, cap=cap, cfg=cfg, backend=mk)
    toks = [int(t) for t in np.random.default_rng(1).integers(0, spec.vocab, pos + 1)]
    for t in toks[:-1]:
        eng.step(t)
    holder["b"].traced = True
    eng.step(toks[-1])
    programs, st = holder["b"].last
    name = f"{'Qwen3-0.6B' if real else 'tiny Qwen3'} token at pos {pos}" + (
        " (board config)" if board else "")
    p = parse(st["trace"], cfg, programs, name)
    p.cycles = st["cycles"]
    return p, cfg, programs


def _kernel_workloads():
    import numpy as np
    from .isasim import design_config, board_config
    from .kernels import attention_decode, attention_layer, mlp
    tk = _args_mod()

    def mk(kernel, argf, cfgf, name):
        def build(board: bool):
            cfg = board_config(DRAM_BYTES=1 << 24) if board else cfgf()
            a = argf(cfg)
            return kernel, cfg, a, name + (" (board config)" if board else "")
        return build

    def mlp_a(M):
        return lambda cfg: tk.mlp_args(np.random.default_rng(0), M=M, H=1024, Fd=4096)[0]

    def attn_a(Hq, Hkv, T):
        return lambda cfg: tk.attn_args(np.random.default_rng(1), Hq=Hq, Hkv=Hkv, d=128, T=T,
                                        cap=T, block=128)[0]

    def layer_a(pos):
        def f(cfg):
            a, _ = tk.layer_args(np.random.default_rng(2), cfg.S, H=1024, Hq=16, Hkv=4, d=128,
                                 pos=pos, cap=pos + 129)
            a["block"] = 128
            return a
        return f

    small = lambda: __import__("opentpu").Config()       # noqa: E731
    return {
        "mlp-small": mk(mlp, lambda cfg: tk.mlp_args(np.random.default_rng(0), M=3)[0], small,
                        "MLP (small)"),
        "mlp": mk(mlp, mlp_a(1), design_config, "MLP decode M=1"),
        "mlp8": mk(mlp, mlp_a(8), design_config, "MLP M=8"),
        "attn": mk(attention_decode, attn_a(16, 4, 2048), design_config,
                   "Flash attention 16q/4kv T=2048"),
        "attn-qwen": mk(attention_decode, attn_a(16, 8, 1024), design_config,
                        "Flash attention 16q/8kv T=1024"),
        "layer": mk(attention_layer, layer_a(1023), design_config, "Attention layer pos=1023"),
    }


WORKLOAD_HELP = {
    "mlp-small": "tiny MLP at the default config (seconds)",
    "mlp": "MLP decode, H=1024 F=4096, design config",
    "mlp8": "MLP with 8 rows",
    "attn": "flash attention 16 q / 4 kv heads, 2048 positions",
    "attn-qwen": "flash attention with Qwen3-0.6B's head layout, 1024 positions",
    "layer": "full attention layer at position 1023",
    "qwen-tiny": "one decode token of a small random Qwen3 (use --pos)",
    "qwen": "one decode token of the real Qwen3-0.6B (models/Qwen3-0.6B; minutes)",
}


def record(workload: str, board: bool = False, isa: bool = False, pos: int = 8,
           bucket: int = 64, axi: bool | None = None, stall: int | None = None) -> dict:
    """Run one workload and return its profile dict."""
    from .profile import profile
    uarch = dict(rtlsim.BOARD_UARCH) if board else None
    run_kw = {"plusargs": [f"+bucket={bucket}"]}
    if axi is not None:
        run_kw["axi"] = axi
    if stall is not None:
        run_kw["stall"] = stall
    if workload in ("qwen", "qwen-tiny"):
        if isa:
            raise SystemExit("--isa is not supported for qwen workloads")
        p, cfg, _ = _qwen(workload == "qwen", pos, board, run_kw, uarch)
        return _mem(to_data(p, uarch), cfg, axi)
    wl = _kernel_workloads()
    if workload not in wl:
        raise SystemExit(f"unknown workload {workload!r}; try `lens list`")
    kernel, cfg, a, name = wl[workload](board)
    if isa:
        from .runtime import compile_kernel
        comp, imgs = compile_kernel(kernel, cfg, **a)
        return isa_data(name + " [ISA]", cfg, comp.programs, imgs, uarch)
    p = profile(kernel, cfg, name, uarch=uarch, run_kw=run_kw, **a)
    return _mem(to_data(p, uarch), cfg, axi)


def _mem(d: dict, cfg, axi) -> dict:
    """Record which memory model the RTL ran against."""
    used = (rtlsim.MEMORY["AXI"] if axi is None else axi) and cfg.D == 128
    d["config"]["MEM"] = "axi" if used else "fixed"
    return d


# ============================================================================ CLI
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="lens", description="openTPU Lens: record and explore "
                                 "openTPU profiles")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="run workloads and write a profile file")
    r.add_argument("workloads", nargs="+")
    r.add_argument("-o", "--out", required=True)
    r.add_argument("--board", action="store_true",
                   help="the board configuration and micro-architecture")
    r.add_argument("--isa", action="store_true", help="ISA simulator with analytic timing")
    r.add_argument("--axi", action="store_true", help="the board's AXI memory path (D=128)")
    r.add_argument("--stall", type=int, help="AXI model stall percent")
    r.add_argument("--pos", type=int, default=8, help="qwen workloads: token position")
    r.add_argument("--bucket", type=int, default=64, help="cycles per counter bucket")
    r.add_argument("--open", action="store_true", help="open the app afterwards")
    o = sub.add_parser("open", help="open a profile file in the app")
    o.add_argument("file", nargs="?")
    o.add_argument("--port", type=int, default=0)
    o.add_argument("--no-browser", action="store_true")
    h = sub.add_parser("html", help="write a standalone HTML page with the profiles embedded")
    h.add_argument("file")
    h.add_argument("-o", "--out", required=True)
    i = sub.add_parser("info", help="summarize a profile file")
    i.add_argument("file")
    sub.add_parser("list", help="list the workloads")
    a = ap.parse_args(argv)

    if a.cmd == "list":
        for k, v in WORKLOAD_HELP.items():
            print(f"  {k:10s} {v}")
        return
    if a.cmd == "record":
        profs = []
        for w in a.workloads:
            d = record(w, board=a.board, isa=a.isa, pos=a.pos, bucket=a.bucket,
                       axi=True if a.axi else None, stall=a.stall)
            print(f"{d['name']}: {d['cycles']} cycles, "
                  f"{100 * d['roofline']['efficiency']:.1f}% of roofline", flush=True)
            profs.append(d)
        out = save(profs, a.out)
        print(f"wrote {out}")
        if a.open:
            serve(out)
        return
    if a.cmd == "open":
        if a.file:
            load(a.file)                      # validate before serving
        serve(a.file, a.port, not a.no_browser)
        return
    if a.cmd == "html":
        out = Path(a.out)
        out.write_text(render(load(a.file)))
        print(f"wrote {out}")
        return
    if a.cmd == "info":
        doc = load(a.file)
        print(f"{a.file}: {doc['format']} v{doc['version']}, created {doc.get('created', '?')}")
        for d in doc["profiles"]:
            c = d["config"]
            print(f"  [{d['kind']}] {d['name']}: {d['cycles']} cycles, roofline "
                  f"{d['roofline']['bound']} ({100 * d['roofline']['efficiency']:.1f}%), "
                  f"{len(d['instrs'])} instructions, S={c['S']} D={c['D']} MCOLS={c['MCOLS']} "
                  f"LANES={c['LANES']}")
        return


if __name__ == "__main__":
    main()
