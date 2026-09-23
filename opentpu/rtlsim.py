"""Build and run the SystemVerilog RTL with Verilator."""
from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RTL = ROOT / "rtl"
TB = ROOT / "sim" / "verilator"
BUILD = ROOT / "build" / "verilator"

RTL_SOURCES = [
    "vpu/otpu_fp.sv", "vpu/otpu_fpipe.sv", "top/otpu_pkg.sv", "mem/otpu_dram.sv", "mem/otpu_tmem.sv",
    "mem/otpu_axi_dram.sv", "mem/otpu_actram.sv", "seq/otpu_seq.sv", "dma/otpu_dma.sv", "mxu/otpu_mxu.sv",
    "vpu/otpu_quant.sv", "vpu/otpu_vpu.sv", "top/otpu_coll.sv", "top/otpu_slice.sv",
    "top/otpu_top.sv",
]


def build(top: str, sources: list[Path], params: dict | None = None) -> Path:
    """Compile `top` with Verilator (--binary); cached on source contents and parameters."""
    params = params or {}
    h = hashlib.sha1()
    for s in sources:
        h.update(Path(s).read_bytes())
    h.update(repr(sorted(params.items())).encode())
    out = BUILD / f"{top}_{h.hexdigest()[:12]}"
    exe = out / f"V{top}"
    if exe.exists():
        return exe
    out.mkdir(parents=True, exist_ok=True)
    cmd = ["verilator", "--binary", "-j", "0", "--top-module", top, "-Wno-fatal",
           "-Wno-WIDTHEXPAND", "-Wno-WIDTHTRUNC", "-Wno-UNUSEDSIGNAL", "-Wno-UNUSEDPARAM",
           "-O3", "--x-assign", "0", "--x-initial", "0", "-Mdir", str(out)]
    cmd += [f"-G{k}={v}" for k, v in params.items()]
    cmd += [str(s) for s in sources]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 and "linker command failed" in r.stdout + r.stderr:
        # the parallel make occasionally archives a stale object: rebuild the archive once
        for f in out.glob("*__ALL.a"):
            f.unlink()
        r = subprocess.run(["make", "-C", str(out), "-f", f"V{top}.mk", "-j", "8"],
                           capture_output=True, text=True)
    if r.returncode != 0 or not exe.exists():
        raise RuntimeError(f"verilator failed:\n{r.stdout[-4000:]}\n{r.stderr[-4000:]}")
    return exe


def run_fp_vectors(vec_path: Path) -> str:
    exe = build("tb_fp", [RTL / "vpu/otpu_fp.sv", TB / "tb_fp.sv"])
    r = subprocess.run([str(exe), f"+vec={vec_path}"], capture_output=True, text=True, timeout=600)
    return r.stdout + r.stderr


def _write_hex(path: Path, words: np.ndarray) -> None:
    nz = np.nonzero(words)[0]
    n = int(nz[-1]) + 1 if len(nz) else 1
    path.write_text("\n".join(f"{int(w):08x}" for w in words[:n]) + "\n")


def _read_hex(path: Path, n: int) -> np.ndarray:
    toks = [t for t in path.read_text().split() if not t.startswith(("@", "//"))]
    out = np.zeros(n, dtype=np.uint32)
    vals = np.fromiter((int(t, 16) for t in toks), dtype=np.uint64, count=len(toks))
    out[: len(vals)] = vals.astype(np.uint32)
    return out


# Micro-architecture knobs that change timing only (never results). Tests and tools may
# override them through `rtlsim.UARCH` or the `uarch` argument of run().
UARCH = {"WIN": 32, "RPB": 4, "WPB": 2}
# The board build: TMEM is replicated per read port (reads never conflict), one write per bank
# per cycle (simple dual-port block RAM), a 16-entry dispatch window, and a 1024-chunk MXU
# prefetch FIFO (128 KiB of block RAM: the weight stream runs ahead through the serial
# norm -> quantize -> MM dependency chains).
BOARD_UARCH = {"WIN": 16, "RPB": 64, "WPB": 1, "FIFO_DEPTH": 1024}
if os.environ.get("OTPU_UARCH") == "board":
    UARCH = dict(BOARD_UARCH)
# MXU dot-product implementation (timing only): OTPU_MXU=cascade selects the DSP cascade
# chains (OTPU_MXU_CL products per chain) instead of the adder tree.
if os.environ.get("OTPU_MXU") == "cascade":
    UARCH["MXU_IMPL"] = 1
    UARCH["MXU_CL"] = int(os.environ.get("OTPU_MXU_CL", "16"))

# The memory path. AXI: the board's AXI adapter in front of a two-channel AXI memory model with
# random stalls (percent) and latency (D = 128 only; other configurations keep the behavioural
# DRAM). BOOT: the program is placed in DRAM and copied into IMEM by the slice's loader, as on
# the board. The environment (OTPU_AXI=1, OTPU_BOOT=1, OTPU_STALL=n) sets the defaults, so the
# whole suite can be run on the board's memory path.
MEMORY = {"AXI": os.environ.get("OTPU_AXI", "0") == "1",
          "BOOT": os.environ.get("OTPU_BOOT", "0") == "1",
          "STALL": int(os.environ.get("OTPU_STALL", "20")),
          "SEED": int(os.environ.get("OTPU_SEED", "1")),
          "BW": int(os.environ.get("OTPU_BW", "100")),        # percent of a beat/cycle/channel
          "LAT": int(os.environ.get("OTPU_LAT", "20"))}


def top_params(cfg, dram_lat: int = 8, uarch: dict | None = None, axi: bool = False,
               dram_bytes: int | None = None) -> dict:
    p = {"S": cfg.S, "D": cfg.D, "MCOLS": cfg.MCOLS, "ACT_BLOCKS": cfg.ACT_BLOCKS,
         "TMEM_WORDS": cfg.TMEM_WORDS, "IMEM_WORDS": cfg.IMEM_WORDS,
         "DRAM_WORDS": (dram_bytes or cfg.DRAM_BYTES) // 4, "DRAM_LAT": dram_lat,
         "LANES": cfg.LANES,
         "AXI": int(axi)}
    p.update(UARCH)
    p.update(uarch or {})
    return p


def build_top(cfg, dram_lat: int = 8, uarch: dict | None = None, axi: bool = False,
              dram_bytes: int | None = None) -> Path:
    return build("tb_top", [RTL / s for s in RTL_SOURCES] +
                 [TB / "otpu_axi_mem.sv", TB / "tb_top.sv"],
                 top_params(cfg, dram_lat, uarch, axi, dram_bytes))


def run(cfg, programs: list, images: list, dram_lat: int = 8, max_cycles: int = 50_000_000,
        keep: Path | None = None, trace: bool = False, uarch: dict | None = None,
        axi: bool | None = None, boot: bool | None = None, stall: int | None = None,
        seed: int | None = None, bw: int | None = None, lat: int | None = None,
        plusargs: list | None = None):
    """Run the RTL; returns (drams as uint8 arrays, tmems as uint32 arrays, stats)."""
    from . import isa as I
    axi = MEMORY["AXI"] if axi is None else axi
    axi = axi and cfg.D == 128
    boot = MEMORY["BOOT"] if boot is None else boot
    stall = MEMORY["STALL"] if stall is None else stall
    seed = MEMORY["SEED"] if seed is None else seed
    tmp = Path(keep) if keep else Path(tempfile.mkdtemp(prefix="otpu_"))
    tmp.mkdir(parents=True, exist_ok=True)
    imgs, progs = [], []
    for s in range(cfg.S):
        words = I.assemble(programs[s])
        if len(words) > cfg.IMEM_WORDS:
            raise ValueError("program does not fit IMEM")
        _write_hex(tmp / f"prog_{s}.hex", words)
        img = np.asarray(images[s], np.uint8)
        imgs.append(np.concatenate([img, np.zeros(-len(img) % 4, np.uint8)]))
        progs.append(np.asarray(words, "<u4"))
    # boot: every slice's program right after the largest image (one loader address), or, if
    # there is no room, above the machine's DRAM (the simulated DRAM is then doubled)
    at = -(-max(len(i) for i in imgs) // cfg.D) * cfg.D
    grow = boot and at + 4 * max(len(p) for p in progs) > cfg.DRAM_BYTES
    if grow:
        at = cfg.DRAM_BYTES
    exe = build_top(cfg, 20 if axi else dram_lat, uarch, axi,
                    2 * cfg.DRAM_BYTES if grow else None)
    for s in range(cfg.S):
        img = imgs[s]
        if boot:
            if 4 * len(progs[s]) > cfg.DRAM_BYTES:
                raise ValueError("no room in DRAM for the program")
            img = np.concatenate([img, np.zeros(at - len(img), np.uint8), progs[s].view(np.uint8)])
        img.view("<u4").astype(">u4").tofile(tmp / f"dram_{s}.bin")
    args = [str(exe), f"+dir={tmp}", f"+max_cycles={max_cycles}"] + (["+trace"] if trace else [])
    args += list(plusargs or [])
    if axi:
        args += [f"+axi_stall={stall}", f"+axi_seed={seed}",
                 f"+axi_bw={MEMORY['BW'] if bw is None else bw}",
                 f"+axi_lat={MEMORY['LAT'] if lat is None else lat}"]
    if boot:
        args += ["+boot", f"+boot_addr={at}", f"+boot_n={max(len(p) for p in progs) // 8}"]
    r = subprocess.run(args,
                       capture_output=True, text=True, timeout=3600)
    out = r.stdout + r.stderr
    import re
    m = re.search(r"RESULT cycles=(\d+) halted=(\d+) error=(\d+)", out)
    if not m:
        raise RuntimeError(f"RTL simulation failed:\n{out[-3000:]}")
    cycles, halted, err = (int(x) for x in m.groups())
    if not halted or err:
        raise RuntimeError(f"RTL did not halt cleanly (halted={halted} error={err}):\n{out[-2000:]}")
    icounts = [int(x) for x in re.findall(r"SLICE \d+ icount=(\d+)", out)]
    drams = [np.fromfile(tmp / f"dram_out_{s}.bin", dtype=np.uint8) for s in range(cfg.S)]
    if boot:                           # the program is not part of the result
        drams = [d[:cfg.DRAM_BYTES] for d in drams]
        for s in range(cfg.S):
            drams[s][at:at + 4 * len(progs[s])] = 0
    tmems = [_read_hex(tmp / f"tmem_{s}.hex", cfg.TMEM_WORDS) for s in range(cfg.S)]
    stats = {"cycles": cycles, "instructions": icounts}
    if trace:
        stats["trace"] = out
    return drams, tmems, stats
