"""Decode speed over core clocks on the RTL: one decode token (the resident decode program at
--pos, as the card runs it) per core clock (and DDR3 rate: 1066 only, the board's DDR3 never runs
faster), through the board's memory path with either the card's controller (--mem ldc: LiteDRAM's
own, co-simulated, rtlsim.MEMORY's LDC) or the calibrated bank model (--mem model:
otpu_native_mem, profile.ddr3_plusargs, 300 ns read latency).

    python3 tools/perf_ddr.py CACHE.json IMGDIR --model qwen3 --format fp4 [--mem ldc]
                              [--mhz 100,133.33,150,200] [--ddr 1066] [--prefixes]
    python3 tools/perf_ddr.py CACHE.json IMGDIR --table [--models qwen3,qwen35,lfm2]
                              [--formats fp4,int8] [--mem ldc] [--mhz ...] [--ddr ...]
                              [--prefixes]

A point is the whole model's token, simulated (2-3 minutes per point for these models on the
co-simulated controller). --prefixes builds it from layer prefixes instead, as
tools/perf_prefill.py builds prefill: the first layer, then per layer kind the difference of two
prefixes (a run is one or two layers plus the LM head). That is faster but wrong on the card's
two-port controller: there the second layer costs less than the layers after it (Qwen3 fp4 at
133.33 MHz: 83.9k cycles, against 91.8k for every layer from the third on), and Qwen3's token
comes out 5.4% short (3.691 against 3.903 Mcycles; the card: 3.939). The whole model is within
1.1% of the card (docs/litedram.md section 11). Each image (whole model or prefix) is built
once, streamed to IMGDIR as tb_top loads it (the image, then the program at the boot address),
and every point then runs the simulator on that file (the DRAM dump goes nowhere). Points are
cached in CACHE.json (one entry per layer count and point, with the DRAM commands), so a stopped
run resumes. --table prints Mcycles/token, tokens/s, the
DRAM bytes read and written per token and GB/s. The configuration comes from the environment
(the board's: OTPU_MCOLS=4 OTPU_MXU=systolic OTPU_PAIR=1 OTPU_DSTEP=1 OTPU_STREAM=1). The KV cache
holds --cap positions (default 2048), the LM head is int8 (--head). OTPU_LDC_BREAK=1 (--mem ldc)
adds each channel's cycle breakdown (BRK lines: sim/verilator/otpu_ldc_mem.sv +ldc_break; cached
under the point's key + "|brk"; docs/litedram.md section 11, "What is left after memeff").
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from opentpu import isa as I  # noqa: E402
from opentpu import rtlsim, simtmp  # noqa: E402
from opentpu.compiler import arg_words  # noqa: E402
from opentpu.isasim import board_config  # noqa: E402
from opentpu.llm import load_spec, model_dir  # noqa: E402
from opentpu.llm.qwen3 import ATTN_BLOCK, RunPos  # noqa: E402
from opentpu.profile import ddr3_plusargs  # noqa: E402
from perf_prefill import _kinds, _prefix, _proxies  # noqa: E402

MTS = {1066: 3200 / 3}                 # the board's DDR3 never runs faster (its HR banks' rate)
MIB = 1 << 20
MAX_CYCLES = 1 << 26                   # a prefix's decode token is a few million cycles


def peak_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / MIB if sys.platform == "darwin" else r / 1024


class LazyW:
    """The checkpoint's tensors (load_weights' names, fp32), each read when indexed."""

    def __init__(self, path):
        from safetensors import safe_open
        self.at = {}
        for f in sorted(Path(path).glob("*.safetensors")):
            with safe_open(str(f), "pt") as h:
                for k in h.keys():
                    if not k.startswith(("model.visual.", "mtp.")):
                        self.at[k.replace("model.language_model.", "model.", 1)] = (str(f), k)

    def __contains__(self, k):
        return k in self.at

    def __getitem__(self, k):
        import torch
        from safetensors import safe_open
        f, name = self.at[k]
        with safe_open(f, "pt") as h:
            return h.get_tensor(name).to(torch.float32).numpy()


class Grid:
    def __init__(self, cache: Path, imgdir: Path, mem: str, pos: int, cap: int, head: str,
                 plus: list | None = None, prefixes: bool = False):
        self.cache_p, self.imgdir, self.mem = cache, imgdir, mem
        self.pos, self.cap, self.head = pos, cap, head
        self.use_prefixes = prefixes
        self.plus = list(plus or [])
        self.cache = json.loads(cache.read_text()) if cache.exists() else {}

    def key(self, name, n, wf, mhz, ddr) -> str:
        env = "|".join(os.environ.get(k, "-") for k in
                       ("OTPU_MCOLS", "OTPU_MXU", "OTPU_PAIR", "OTPU_DSTEP", "OTPU_STREAM"))
        return (f"{name}|{env}|{wf}|{self.head}|{mhz}|{ddr}|{n}|step|{self.pos}|{self.cap}|"
                f"resident" + ("|ldc" if self.mem == "ldc" else "") +
                "".join(f"|{x}" for x in self.plus))

    @staticmethod
    def prefixes(spec):
        kinds = _kinds(spec)
        _, diff = _proxies(kinds)
        return kinds, diff, sorted({1} | {x for p in diff.values() for x in p})

    def image(self, name, spec, W, n, wf) -> dict:
        """The prefix's DRAM file, program and run metadata in imgdir (built once)."""
        stem = self.imgdir.resolve() / f"{name}_{wf}_{self.head}_{self.pos}_{self.cap}_{n}"
        meta_p = stem.with_suffix(".json")
        if meta_p.exists():
            return json.loads(meta_p.read_text())
        t = time.time()
        sp = _prefix(spec, n)
        kw = dict(wformat=wf, head_format=self.head, lookup=True)
        need = sp.image(board_config(DRAM_BYTES=1 << 40), self.cap, **kw).nbytes
        # the simulated DRAM: a power of two up to 1 GiB, else in 256 MiB steps
        size = 1 << max(20, (need - 1).bit_length())
        if size > 1 << 30:
            size = -(-(need + 16 * MIB) // (256 * MIB)) * 256 * MIB
        img = sp.image(board_config(DRAM_BYTES=size), self.cap, **kw)
        K = getattr(img.spec, "conv_k", 1)
        b = self.pos // ATTN_BLOCK + 1
        progs, ra = img.compile_decode(b, max((b - 1) * ATTN_BLOCK, K - 1), ATTN_BLOCK)
        args = arg_words(ra, RunPos.values(791, self.pos, K, ATTN_BLOCK))
        cfg = img.cfg
        if 8 * len(progs[0]) > cfg.IMEM_WORDS:
            cfg = dataclasses.replace(cfg, IMEM_WORDS=1 << (8 * len(progs[0]) - 1).bit_length())
        prog = np.asarray(I.assemble(progs[0]), "<u4")
        dram = img.build(W)[0]
        n4 = -(-len(dram) // 4) * 4
        at = -(-n4 // cfg.D) * cfg.D
        grow = at + 4 * len(prog) > size          # rtlsim._run: the program above the DRAM
        if grow:
            at = size
        tmp = stem.with_suffix(".bin.tmp")
        with open(tmp, "wb") as f:                # rtlsim._run's image file, in pieces
            step = 64 * MIB
            for i in range(0, len(dram), step):
                c = dram[i:i + step]
                if len(c) % 4:
                    c = np.concatenate([c, np.zeros(-len(c) % 4, np.uint8)])
                c.view("<u4").astype(">u4").tofile(f)
            z = at - n4
            while z > 0:
                np.zeros(min(z, step) // 4, ">u4").tofile(f)
                z -= min(z, step)
            prog.astype(">u4").tofile(f)
        tmp.rename(stem.with_suffix(".bin"))
        rtlsim._write_hex(stem.with_suffix(".hex"), prog)
        meta = {"DRAM_BYTES": size, "IMEM_WORDS": cfg.IMEM_WORDS, "at": at, "grow": grow,
                "boot_n": len(prog) // 8, "args": [int(a) for a in args],
                "nbytes": int(len(dram)), "bin": str(stem.with_suffix(".bin")),
                "hex": str(stem.with_suffix(".hex"))}
        meta_p.write_text(json.dumps(meta))
        print(f"  {name} {wf} {n} layers: image {meta['nbytes'] / MIB:.0f} MiB, DRAM "
              f"{size / MIB:.0f} MiB, program {len(prog)} words ({time.time() - t:.0f} s, "
              f"driver peak {peak_mb():.0f} MB)", flush=True)
        return meta

    def simulate(self, meta, mhz, ddr) -> tuple[int, list]:
        """One token: cycles and the DRAM commands (reads, writes, row changes, partial)."""
        size = meta["DRAM_BYTES"]
        cfg = dataclasses.replace(board_config(DRAM_BYTES=size), IMEM_WORDS=meta["IMEM_WORDS"])
        ldc = ddr if self.mem == "ldc" else 0
        exe = rtlsim.build_top(cfg, 20, rtlsim.BOARD_UARCH, True,
                               2 * size if meta["grow"] else None, ldc)
        with simtmp.tempdir("otpu_ddr_") as d:
            (d / "dram_0.bin").symlink_to(meta["bin"])
            (d / "prog_0.hex").symlink_to(meta["hex"])
            (d / "dram_out_0.bin").symlink_to("/dev/null")
            args = [str(exe), f"+dir={d}", f"+max_cycles={MAX_CYCLES}"] + self.plus
            args += rtlsim.ldc_plusargs(ddr, mhz) if ldc else ddr3_plusargs(MTS[ddr], mhz)
            args += [f"+arg{k}={int(v) & 0xFFFFFFFF}" for k, v in enumerate(meta["args"])]
            args += ["+axi_stall=0", f"+axi_seed={rtlsim.MEMORY['SEED']}", "+axi_bw=100",
                     f"+axi_lat={round(0.3 * mhz)}"]
            args += ["+boot", f"+boot_addr={meta['at']}", f"+boot_n={meta['boot_n']}"]
            if ldc and os.environ.get("OTPU_LDC_BREAK"):
                args += ["+ldc_break"]     # the controller's cycle breakdown (otpu_ldc_mem.sv)
            r = rtlsim.run_sim(args, timeout=3600)
        out = r.stdout + r.stderr
        self.brk = re.findall(r"^BRK .*$", out, re.M)
        m = re.search(r"RESULT cycles=(\d+) halted=(\d+) error=(\d+)", out)
        if not m or m.group(2) != "1" or m.group(3) != "0":
            raise RuntimeError(f"RTL simulation failed:\n{out[-3000:]}")
        mem = [tuple(map(int, x)) for x in
               re.findall(r"MEM ch\d rd=(\d+) wr=(\d+) row_miss=(\d+) rmw=(\d+)", out)]
        return int(m.group(1)), [sum(x[i] for x in mem) for i in range(4)]

    def layer_counts(self, spec) -> list:
        """The layer counts a point simulates: the whole model's, or its prefixes'."""
        return self.prefixes(spec)[2] if self.use_prefixes else [len(_kinds(spec))]

    def full(self, name, spec, wf, mhz, ddr):
        """The whole model's token cycles and DRAM commands (None if not simulated yet): the
        whole model's run, or (--prefixes) built from its prefixes."""
        if not self.use_prefixes:
            k = self.key(name, len(_kinds(spec)), wf, mhz, ddr)
            return self.cache.get(k), self.cache.get(k + "|mem")
        kinds, diff, ns = self.prefixes(spec)
        c = {n: self.cache.get(self.key(name, n, wf, mhz, ddr)) for n in ns}
        m = {n: self.cache.get(self.key(name, n, wf, mhz, ddr) + "|mem") for n in ns}
        if None in c.values():
            return None, None
        cyc = c[1] + sum(c[diff[k][1]] - c[diff[k][0]] for k in kinds[1:])
        mm = None if None in m.values() else [
            m[1][i] + sum(m[diff[k][1]][i] - m[diff[k][0]][i] for k in kinds[1:])
            for i in range(4)]
        return cyc, mm

    def run(self, name, wf, mhzs, ddrs):
        spec = load_spec(model_dir(name))
        ns = self.layer_counts(spec)
        self.imgdir.mkdir(parents=True, exist_ok=True)
        W = LazyW(model_dir(name))
        metas = {n: self.image(name, spec, W, n, wf) for n in ns}
        del W
        for ddr in ddrs:
            for mhz in mhzs:
                for n in ns:
                    k = self.key(name, n, wf, mhz, ddr)
                    if k in self.cache:
                        continue
                    t = time.time()
                    cyc, mem = self.simulate(metas[n], mhz, ddr)
                    self.cache[k], self.cache[k + "|mem"] = cyc, mem
                    if getattr(self, "brk", None):
                        self.cache[k + "|brk"] = self.brk
                        print("\n".join(self.brk), flush=True)
                    self.cache_p.write_text(json.dumps(self.cache, indent=1))
                    print(f"  {name} {wf} {mhz} MHz DDR3-{ddr}: {n} layers: {cyc} cycles "
                          f"({time.time() - t:.0f} s) rd/wr/row_miss/rmw {mem}", flush=True)
                cyc, mm = self.full(name, spec, wf, mhz, ddr)
                print(f"{name} {wf} DDR3-{ddr} {mhz} MHz: {cyc} cycles, rd {mm[0]} wr {mm[1]} "
                      f"(native commands; driver peak {peak_mb():.0f} MB)", flush=True)

    def table(self, names, wfs, mhzs, ddrs):
        print(f"{'model':7} {'fmt':4} {'DDR3':>5} {'MHz':>7} {'Mcyc/token':>10} {'tok/s':>6} "
              f"{'MB read':>8} {'MB written':>10} {'GB/s':>6}")
        for name in names:
            spec = load_spec(model_dir(name))
            for wf in wfs:
                for ddr in ddrs:
                    for mhz in mhzs:
                        cyc, mm = self.full(name, spec, wf, mhz, ddr)
                        if cyc is None:
                            continue
                        rd, wr = (mm[0] * 64 / 1e6, mm[1] * 64 / 1e6) if mm else (0, 0)
                        print(f"{name:7} {wf:4} {ddr:5d} {mhz:7g} {cyc / 1e6:10.3f} "
                              f"{mhz * 1e6 / cyc:6.1f} {rd:8.1f} {wr:10.1f} "
                              f"{(rd + wr) * mhz / cyc * 1e3:6.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache", type=Path)
    ap.add_argument("imgdir", type=Path)
    ap.add_argument("--model", help="run this model's points")
    ap.add_argument("--format", default="fp4", choices=["int8", "int4", "fp4"])
    ap.add_argument("--table", action="store_true", help="print the cached points")
    ap.add_argument("--models", default="qwen3,qwen35,lfm2")
    ap.add_argument("--formats", default="fp4,int8")
    ap.add_argument("--mem", default="ldc", choices=["ldc", "model"])
    ap.add_argument("--mhz", default="100,133.33,150,200")
    ap.add_argument("--ddr", default="1066")
    ap.add_argument("--pos", type=int, default=544)
    ap.add_argument("--cap", type=int, default=2048)
    ap.add_argument("--head", default="int8", choices=["int8", "int4", "fp4"])
    ap.add_argument("--plus", action="append", default=[],
                    help="extra simulator argument (repeatable; part of the cache key)")
    ap.add_argument("--prefixes", action="store_true",
                    help="build the token from layer prefixes (fast; short on two ports)")
    a = ap.parse_args()
    g = Grid(a.cache, a.imgdir, a.mem, a.pos, a.cap, a.head, a.plus, a.prefixes)
    mhzs = [float(x) for x in a.mhz.split(",")]
    ddrs = [int(x) for x in a.ddr.split(",")]
    if a.table:
        g.table(a.models.split(","), a.formats.split(","), mhzs, ddrs)
    elif a.model:
        g.run(a.model, a.format, mhzs, ddrs)
    else:
        ap.error("--model or --table")


if __name__ == "__main__":
    main()
