"""Build and run the SystemVerilog RTL with Verilator."""
from __future__ import annotations

import fcntl
import hashlib
import os
import re as _re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from . import simtmp
from . import target as _target

ROOT = Path(__file__).resolve().parent.parent
RTL = ROOT / "rtl"
TB = ROOT / "sim" / "verilator"
BUILD = ROOT / "build" / "verilator"

RTL_SOURCES = [
    "vpu/otpu_fp.sv", "vpu/otpu_fpipe.sv", "top/otpu_pkg.sv", "mem/otpu_dram.sv", "mem/otpu_tmem.sv",
    "mem/otpu_native_dram.sv", "mem/otpu_xstg.sv", "mem/otpu_actram.sv", "seq/otpu_seq.sv",
    "vpu/otpu_vtree.sv", "dma/otpu_dma.sv", "mxu/otpu_mxu.sv", "tech/generic/otpu_tech_dsp.sv",
    "tech/generic/otpu_tech_ram.sv",
    "vpu/otpu_quant.sv", "vpu/otpu_se_comp.sv", "vpu/otpu_se_tail.sv", "vpu/otpu_vpu.sv",
    "top/otpu_coll.sv", "exp/tmem_nocopy/otpu_tmem_model.sv", "top/otpu_slice.sv",
    "top/otpu_top.sv",
]


# Verilator's flags (in the build cache's key, with its version and the sources' names)
VFLAGS = ["--binary", "-j", "0", "-Wno-fatal", "-Wno-WIDTHEXPAND", "-Wno-WIDTHTRUNC",
          "-Wno-UNUSEDSIGNAL", "-Wno-UNUSEDPARAM", "-O3", "--x-assign", "0", "--x-initial", "0"]
_VERSION: str | None = None


def vflags() -> list[str]:
    """VFLAGS, with OTPU_X_INITIAL=unique Verilator's --x-initial unique: every variable an
    `initial` block does not set powers up at a random value (seeded by OTPU_X_SEED, 1 if unset:
    run_sim gives a Verilator binary +verilator+rand+reset+2 +verilator+seed+N), as on an ASIC,
    instead of 0, as on an FPGA (docs/asic.md "Power-up state"). With it --no-assert-case: a
    one-hot select (the VPU's decoded function) is one-hot from its reset on, not at power-up,
    and Verilator checks a `unique case` from time 0 ($assertoff does not reach it)."""
    x = os.environ.get("OTPU_X_INITIAL", "0")
    if x not in ("0", "unique"):
        raise ValueError(f"OTPU_X_INITIAL={x!r}: 0 or unique")
    f = list(VFLAGS)
    if "--x-initial" in f:
        f[f.index("--x-initial") + 1] = x
    if x == "unique":
        f.append("--no-assert-case")
    return f


def x_plusargs() -> list[str]:
    """A simulation's power-up plusargs: OTPU_X_INITIAL=unique's random values (vflags)."""
    if os.environ.get("OTPU_X_INITIAL", "0") != "unique":
        return []
    return ["+verilator+rand+reset+2", f"+verilator+seed+{int(os.environ.get('OTPU_X_SEED', '1'))}"]


def verilator_version() -> str:
    """`verilator --version` (once per process); "" without Verilator."""
    global _VERSION
    if _VERSION is None:
        try:
            _VERSION = subprocess.run(["verilator", "--version"], capture_output=True,
                                      text=True).stdout.strip()
        except FileNotFoundError:
            _VERSION = ""
    return _VERSION


def build_dir(top: str, sources: list[Path], params: dict | None = None) -> Path:
    """The build's directory: keyed on the sources' names (in the tree: OTPU_REMOTE_BUILD shares
    build/ between trees) and contents, the parameters, vflags() and Verilator's version."""
    h = hashlib.sha1()
    for s in sources:
        s = Path(s).resolve()
        name = s.relative_to(ROOT).as_posix() if s.is_relative_to(ROOT) else str(s)
        h.update(name.encode() + b"\0" + s.read_bytes() + b"\0")
    h.update(repr(sorted((params or {}).items())).encode())
    h.update(repr(vflags()).encode() + b"\0" + verilator_version().encode())
    return BUILD / f"{top}_{h.hexdigest()[:12]}"


def build(top: str, sources: list[Path], params: dict | None = None) -> Path:
    """Compile `top` with Verilator (--binary); cached (build_dir). The build runs in a
    temporary directory, renamed into place once linked, under a lock per build: a concurrent
    build of the same waits for it instead of running a half-linked executable, and a build
    killed or failed leaves no executable for the next to take."""
    params = params or {}
    out = build_dir(top, sources, params)
    exe = out / f"V{top}"
    if exe.exists():
        return exe
    BUILD.mkdir(parents=True, exist_ok=True)
    with open(BUILD / f".{out.name}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if exe.exists():                        # built while this one waited
            return exe
        for d in BUILD.glob(f".{out.name}.*"):  # a killed build's (none runs: the lock)
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
        tmp = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=BUILD))
        try:
            cmd = ["verilator", *vflags(), "--top-module", top, "-Mdir", str(tmp)]
            cmd += [f"-G{k}={v}" for k, v in params.items()]
            cmd += [str(s) for s in sources]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0 and "linker command failed" in r.stdout + r.stderr:
                # the parallel make occasionally archives a stale object: rebuild the archive once
                for f in tmp.glob("*__ALL.a"):
                    f.unlink()
                r = subprocess.run(["make", "-C", str(tmp), "-f", f"V{top}.mk", "-j", "8"],
                                   capture_output=True, text=True)
            if r.returncode != 0 or not (tmp / f"V{top}").exists():
                raise RuntimeError(f"verilator failed:\n{r.stdout[-4000:]}\n{r.stderr[-4000:]}")
            if out.exists():                    # an unfinished build's (no executable)
                shutil.rmtree(out)
            tmp.rename(out)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return exe


# A simulator whose Python parent dies (killed, or os._exit) would run on under launchd for
# hours: the child is started under a shell that kills it once the parent is gone (macOS has
# no PR_SET_PDEATHSIG), and in its own process group, killed on any exception here (a timeout,
# KeyboardInterrupt).
_WATCH = ('"$@" & c=$!; (while kill -0 "$OTPU_PARENT" 2>/dev/null; do sleep 1; done; '
          'kill -9 $c 2>/dev/null) & w=$!; wait $c; r=$?; kill $w 2>/dev/null; exit $r')


def run_sim(cmd: list, timeout: float | None = None) -> subprocess.CompletedProcess:
    """subprocess.run(cmd, capture_output=True, text=True, timeout=timeout) for a simulator
    binary, which never outlives this process (see _WATCH). A Verilator binary (V<top>) takes
    the power-up plusargs (x_plusargs)."""
    import signal
    if cmd and Path(str(cmd[0])).name.startswith("V"):
        cmd = [*cmd, *x_plusargs()]
    env = {**os.environ, "OTPU_PARENT": str(os.getpid())}
    p = subprocess.Popen(["/bin/sh", "-c", _WATCH, "sh", *map(str, cmd)], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         start_new_session=True)
    simtmp.track(p.pid)                 # (killed on SIGTERM / SIGHUP: simtmp)
    try:
        out, err = p.communicate(timeout=timeout)
    except BaseException:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.wait()
        raise
    finally:
        simtmp.untrack(p.pid)
    return subprocess.CompletedProcess(cmd, p.returncode, out, err)


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
    if len(toks) != n:                  # a short dump (e.g. a full disk) is an error
        raise RuntimeError(f"{path.name}: {len(toks)} words, expected {n}")
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
# RPB covers every read lane (8 ports x up to 16 lanes), which selects the slice's shallow
# write-mask arbiter, as on the board. The values are the selected target's (OTPU_TARGET,
# default the card: targets/ypcb-00338.toml; opentpu.target.sim_uarch).
BOARD_UARCH = _target.sim_uarch(_target.current())
# A one-slice board builds its collective unit without GATHER (COLL_GATHER 0: BAR alone,
# otpu_coll_bar; boards/ypcb-00338/run_vivado.sh's default since release r2, docs/infra_hunt.md
# 5): no compiled program has a GATHER at S = 1, and the host refuses one there. Its simulations
# build the same unit (tb_top / tb_board COLL_GATHER; at S > 1 the unit runs GATHER).
if _target.current().kind == "fpga" and _target.current().arch["S"] == 1:
    BOARD_UARCH["COLL_GATHER"] = 0
if os.environ.get("OTPU_UARCH") == "board":
    UARCH = dict(BOARD_UARCH)
# The MXU: the MX column's 2D systolic array, as the board builds it (weights hop column to
# column; docs/mxu_systolic.md, docs/mx_mxu.md), the only dot product (OTPU_MXU=systolic, the
# only value). OTPU_MXU_E: its epilogue lanes (otpu_mxu E; timing only, 0: MCOLS / 2).
if os.environ.get("OTPU_MXU", "systolic") != "systolic":
    raise ValueError("OTPU_MXU: the systolic MXU is the only one")
UARCH["MXU_IMPL"] = 2
if os.environ.get("OTPU_MXU_E"):
    UARCH["MXU_E"] = int(os.environ["OTPU_MXU_E"])
# OTPU_SE_R: the streams on an engine of their own, SE_R rows a cycle (otpu_slice SE_R; timing
# only, 0: the VPU's engine; docs/se_engine.md)
if os.environ.get("OTPU_SE_R"):
    UARCH["SE_R"] = BOARD_UARCH["SE_R"] = int(os.environ["OTPU_SE_R"])
# OTPU_MXU_QD: the MXU's commands queued (otpu_mxu QD, 2..7; timing only)
if os.environ.get("OTPU_MXU_QD"):
    UARCH["MXU_QD"] = BOARD_UARCH["MXU_QD"] = int(os.environ["OTPU_MXU_QD"])
# OTPU_DMA_LDQ: the DMA's commands in flight (otpu_dma LDQ: 2, an LD's DRAM reads under the
# command before it; timing only)
if os.environ.get("OTPU_DMA_LDQ"):
    UARCH["DMA_LDQ"] = BOARD_UARCH["DMA_LDQ"] = int(os.environ["OTPU_DMA_LDQ"])
# OTPU_COLL_HUB: 0 builds the collective on the slices' own ports at S > 1 too (otpu_top COLL_HUB;
# BOARD_UARCH's is an FPGA target's otpu_board, its hub; timing only)
if os.environ.get("OTPU_COLL_HUB"):
    BOARD_UARCH["COLL_HUB"] = int(os.environ["OTPU_COLL_HUB"])
# OTPU_XRS: the stages each way between the hub and each slice's port (otpu_top XRS, otpu_board's
# crossings, >= 2; timing only)
if os.environ.get("OTPU_XRS"):
    BOARD_UARCH["XRS"] = int(os.environ["OTPU_XRS"])
# OTPU_COLL_PIPE: the hub's reads in flight (otpu_top COLL_PIPE, otpu_board's; 0 one at a time;
# timing only)
if os.environ.get("OTPU_COLL_PIPE"):
    BOARD_UARCH["COLL_PIPE"] = int(os.environ["OTPU_COLL_PIPE"])
# OTPU_NAT_XSTG: register stages each way between a slice and its adapter (otpu_xstg; timing
# only, the native memory paths); OTPU_NAT_XSTG_P the posts' (default NAT_XSTG's)
for _k in ("NAT_XSTG", "NAT_XSTG_P"):
    if os.environ.get(f"OTPU_{_k}"):
        UARCH[_k] = BOARD_UARCH[_k] = int(os.environ[f"OTPU_{_k}"])
# OTPU_QUANT_XD: the quantizer's row buffer in chunks (otpu_quant XD; timing only, 0: QACT reads
# its rows from TMEM twice)
if os.environ.get("OTPU_QUANT_XD"):
    UARCH["QUANT_XD"] = int(os.environ["OTPU_QUANT_XD"])
# OTPU_QUANT_R1=1: RSCALE's row factor read once a row (otpu_quant R1; timing only)
if os.environ.get("OTPU_QUANT_R1"):
    UARCH["QUANT_R1"] = int(os.environ["OTPU_QUANT_R1"])
# OTPU_QUANT_PW: a buffered QACT pass 1's chunks a cycle (otpu_quant PW; timing only)
if os.environ.get("OTPU_QUANT_PW"):
    UARCH["QUANT_PW"] = int(os.environ["OTPU_QUANT_PW"])
# OTPU_QUANT_SWW: the SW port's words, QST pass 1's 4 x SWW bytes a cycle (otpu_quant SWW; timing
# only)
if os.environ.get("OTPU_QUANT_SWW"):
    UARCH["QUANT_SWW"] = int(os.environ["OTPU_QUANT_SWW"])
# OTPU_QUANT_QSTB=1: QST rows (ROW) through the quantizer's row buffer (otpu_quant QSTB; timing
# only)
if os.environ.get("OTPU_QUANT_QSTB"):
    UARCH["QUANT_QSTB"] = int(os.environ["OTPU_QUANT_QSTB"])
# VPU lanes with the composite functions (exp2, recip, rsqrt; timing only): OTPU_VPU_CL=4
if os.environ.get("OTPU_VPU_CL"):
    UARCH["VPU_CL"] = int(os.environ["OTPU_VPU_CL"])
# EXPERIMENTAL (docs/seq_vpu.md): otpu_seq's wait entries instead of its scoreboard (WAITC):
# OTPU_SEQ_WC=1. The entries are the compiler's (opentpu.waits; the run's arguments bounded by
# their variables, OTPU_WAITS=args: known), or a +hz_sync file's (OTPU_SIM_PLUSARGS; or
# OTPU_WAITS=0). WAITC needs a window of 16 (the card's), and the simulator's default is 32
if os.environ.get("OTPU_SEQ_WC"):
    UARCH["SEQ_WC"] = BOARD_UARCH["SEQ_WC"] = int(os.environ["OTPU_SEQ_WC"])
    if UARCH["SEQ_WC"]:
        UARCH["WIN"] = min(UARCH.get("WIN", 32), 16)
# The banked TMEM's request-priority addresses (otpu_slice TMEM_RQ; timing only), with a
# TMEM_IMPL 1 target (OTPU_TARGET=nangate45): OTPU_TMEM_RQ=1 (docs/tmem_pump.md)
if os.environ.get("OTPU_TMEM_RQ"):
    UARCH["TMEM_RQ"] = BOARD_UARCH["TMEM_RQ"] = int(os.environ["OTPU_TMEM_RQ"])
# ... and OTPU_TMEM_NCP=2, 4, 5, 7: the read slots of pumped copies (otpu_slice TMEM_NCP)
if os.environ.get("OTPU_TMEM_NCP"):
    UARCH["TMEM_NCP"] = int(os.environ["OTPU_TMEM_NCP"])
# ... and OTPU_TMEM_IMPL=2: a candidate TMEM's read arbitration in simulation (otpu_tmem_model,
# set by +tm_* plusargs: OTPU_SIM_PLUSARGS; docs/tmem_nocopy.md)
if os.environ.get("OTPU_TMEM_IMPL"):
    UARCH["TMEM_IMPL"] = BOARD_UARCH["TMEM_IMPL"] = int(os.environ["OTPU_TMEM_IMPL"])
# ... OTPU_TMEM_IMPL=3: one copy of sub-banked RAMs (otpu_tmem IMPL 3), OTPU_TMEM_NSB RAMs a bank
# with the sub-bank at word-address bit OTPU_TMEM_SBP, hashed by OTPU_TMEM_SBH (1, 2); with
# OTPU_TMEM_RPL=1 lost reads are replayed under the board's arbiter (otpu_slice TMEM_RPL); with
# OTPU_TMEM_TDP=1 the RAMs are true dual-port, a unit's ports on different RAM ports; with
# OTPU_TMEM_GST=1 QUANT 3 and COLL are the DMA's guests (one data select for the three); with
# OTPU_TMEM_PMP=2 (TDP) the RAMs are pumped at 2x (their model: a write takes no read port)
# OTPU_TMEM_PRK / OTPU_TMEM_SDM (int, 0x... allowed): the read ports' ranks and RAM sides
# (otpu_slice TMEM_PRK / TMEM_SDM)
for _k in ("NSB", "SBP", "SBH", "RPL", "TDP", "GST", "PMP", "PRK", "SDM"):
    if os.environ.get("OTPU_TMEM_" + _k):
        UARCH["TMEM_" + _k] = int(os.environ["OTPU_TMEM_" + _k], 0)
# TMEM lanes of the MXU and the quantizer when fewer than LANES (timing only): OTPU_ULANES=8
if os.environ.get("OTPU_ULANES"):
    UARCH["ULANES"] = int(os.environ["OTPU_ULANES"])
# IMEM an instruction cache over the program in DRAM (otpu_seq ICACHE; docs/icache.md): a
# Config's ICACHE (its target's), or OTPU_ICACHE=1 under any Config (timing only: the program
# runs from DRAM, so a run boots, MEMORY's BOOT), OTPU_IC_LINES / OTPU_IC_PF its lines and the
# lines it prefetches ahead (otpu_seq IC_LINES, IC_PF)
for _k in ("ICACHE", "IC_LINES", "IC_PF"):
    if os.environ.get(f"OTPU_{_k}"):
        UARCH[_k] = BOARD_UARCH[_k] = int(os.environ[f"OTPU_{_k}"])

# The memory path. AXI: the board's memory path -- its adapter (otpu_native_dram) in front of a
# two-channel native memory model (sim/verilator/otpu_native_mem.sv) with random stalls (percent)
# and latency (D = 128 only; other configurations keep the behavioural DRAM). BOOT: the program is
# placed in DRAM and copied into IMEM by the slice's loader, as on the board. The environment
# (OTPU_AXI=1, OTPU_BOOT=1, OTPU_STALL=n) sets the defaults, so the whole suite can be run on the
# board's memory path. NATIVE: the board model's channels -- by default (OTPU_NATIVE=ld) the
# LiteDRAM build's (otpu_mem_ch in front of a LiteDRAM native-port model: sim/verilator/
# otpu_chmem.sv, tb_board MEM_NATIVE=2), with OTPU_NATIVE=1 otpu_native_mem alone (MEM_NATIVE=1).
MEMORY = {"AXI": os.environ.get("OTPU_AXI", "0") == "1",
          "NATIVE": True if os.environ.get("OTPU_NATIVE", "ld") == "1" else "ld",
          "BOOT": os.environ.get("OTPU_BOOT", "0") == "1",
          "STALL": int(os.environ.get("OTPU_STALL", "20")),
          "SEED": int(os.environ.get("OTPU_SEED", "1")),
          "BW": int(os.environ.get("OTPU_BW", "100")),        # percent of a beat/cycle/channel
          "LAT": int(os.environ.get("OTPU_LAT", "20")),
          "LDC": int(os.environ.get("OTPU_LDC", "0")),
          "LDC_MHZ": float(os.environ.get("OTPU_LDC_MHZ", "0")),
          "MODEL": float(os.environ.get("OTPU_MEM_MODEL", "0"))}
# MODEL: the native memory model as the selected target's channels at OTPU_MEM_MODEL MHz of
# core clock (opentpu.target.mem_model: the WG targets' memory channel rate, latency and duty loss) in
# place of BW and LAT (AXI without LDC; a target without a channel model keeps them).
# LDC: the channels behind the adapter are the card's instead of otpu_native_mem's model -- per
# channel the board's bridge (otpu_mem_ch) and LiteDRAM's own controller, generated with the
# production core's settings (tools/litedram/gen_ldc.py; sim/verilator/otpu_ldc_mem.sv holds the
# data) -- at DDR3-LDC (OTPU_LDC=1066, the card's and the board's only rate), in its controller
# clock (the rate / 8) with the core at LDC_MHZ (OTPU_LDC_MHZ; 0: the controller's clock). The
# stall, bandwidth and latency settings do not apply.
LDC_MODELS = {1066: "otpu_ldc_ch.v"}
_SIM_RUNS = 0                   # simulations run by this process (OTPU_SIM_PLUSARGS's %n)


def ldc_plusargs(mts: int, mhz: float = 0) -> list[str]:
    """Simulator arguments of the co-simulated controller (AXI with LDC): its clock, DDR3-`mts` /
    8, against the core's `mhz` (0: the same clock)."""
    ctl = {1066: 3200 / 3}[mts] / 8
    return [f"+ldc_ratio={round(1e6 * (mhz or ctl) / ctl)}"]


def top_params(cfg, dram_lat: int = 8, uarch: dict | None = None, axi: bool | int = False,
               dram_bytes: int | None = None) -> dict:
    p = {"S": cfg.S, "D": cfg.D, "MCOLS": cfg.MCOLS, "ACT_BLOCKS": cfg.ACT_BLOCKS,
         "ACT_ROWS": cfg.act_rows,
         "TMEM_WORDS": cfg.TMEM_WORDS, "IMEM_WORDS": cfg.IMEM_WORDS,
         "DRAM_WORDS": (dram_bytes or cfg.DRAM_BYTES) // 4, "DRAM_LAT": dram_lat,
         "LANES": cfg.LANES,
         "AXI": int(axi)}                                # otpu_top's memory path (2: LDC)
    # weight groups (otpu_mxu WG): WG 1 builds as before, without the parameter; at WG > 1 the
    # memory's channel hash is the Config's (otpu_top CHASH: the group chunks' row order), and
    # every MM is in groups (GALL: the MXU at WG > 1 has no rows of one)
    if cfg.WG > 1:
        if not cfg.GALL:
            raise ValueError(f"WG {cfg.WG} without GALL: the RTL at WG > 1 runs every MM in "
                             f"groups (Config GALL)")
        p["WG"], p["CHASH"] = cfg.WG, int(cfg.CHASH)
    if not cfg.OUTER:                                    # a VPU without VOP OUTER (otpu_vpu)
        p["OUTER"] = 0
    if getattr(cfg, "ICACHE", False):                    # IMEM an instruction cache (otpu_seq)
        p["ICACHE"] = 1
    if cfg.DSTEP2:                                       # DSTEP's pair (otpu_dma, otpu_se_tail)
        p["DSTEP2"] = 1
    if getattr(cfg, "FSUM", False):                      # EXP2SUB's row sums (otpu_vpu FSUM)
        p["VPU_FSUM"] = 1
    if getattr(cfg, "SPLITQ", False):                    # QACT / QST flag LO (otpu_quant SPLIT)
        p["QUANT_SPLIT"] = 1
    if getattr(cfg, "ST16S", False) and getattr(cfg, "ST16", ""):   # ST16R / ST16W (otpu_dma)
        p["ST16S"] = 1
    if getattr(cfg, "QHL", False) and getattr(cfg, "SPLITQ", False):  # QACT HL (otpu_quant)
        p["QUANT_HL"] = 1
    if getattr(cfg, "QSTHL", False) and getattr(cfg, "SPLITQ", False):  # QST HL (otpu_quant)
        p["QUANT_QSTHL"] = 1
    if getattr(cfg, "VSIG", False):                      # SIGMOID / SILU (otpu_vpu VSIG)
        p["VPU_SIG"] = 1
    if getattr(cfg, "VMAC", False):                      # VOP MAC (otpu_vpu MAC)
        p["VPU_MAC"] = 1
    if getattr(cfg, "OUTER1", False) and not cfg.OUTER:  # OUTER decay one (otpu_vpu OUTER1)
        p["VPU_OUTER1"] = 1
    if getattr(cfg, "QONCE", False):                     # QACT ST, LDA (otpu_quant / otpu_dma)
        p["QONCE"] = 1
    if getattr(cfg, "SPLMM", False):                     # MM SPL (otpu_mxu SPL)
        p["MXU_SPL"] = 1
    if getattr(cfg, "ST16", ""):                         # DSTEP's state in 16 bits (otpu_dma)
        p["ST16"] = {"e6m9": 1, "bf16": 2}[cfg.ST16]
    if cfg.KT:                                           # MM's K tail (otpu_mxu)
        if cfg.WG == 1:
            raise ValueError("KT at WG 1: MM's K tail is a grouped MM's (otpu_mxu at WG > 1)")
        p["KT"] = 1
    p.update(UARCH)
    p.update(uarch or {})
    # the posted W beats (NAT_XPOST: otpu_native_dram / otpu_mxu XPOST) where they apply: the
    # native model (AXI 1) at WG > 1; elsewhere the memory path as without them
    if p.get("NAT_XPOST") and (p["AXI"] != 1 or cfg.WG == 1):
        del p["NAT_XPOST"]
    if not p.get("NAT_XPOST"):                         # (a beat's writes: XPOST's alone)
        p.pop("NAT_XQB", None)
        p.pop("NAT_RHB", None)                         # (half reads: with the posted W beats)
    if not p.get("NAT_WPOST") or p["AXI"] != 1:        # (half writes: posted ones, the model's)
        p.pop("NAT_WHB", None)
    if not p.get("NAT_XSTG") or p["AXI"] == 0:         # (the stages: an adapter's alone)
        p.pop("NAT_XSTG", None)
    if p["AXI"] == 0:
        p.pop("NAT_XSTG_P", None)
    if p.get("TMEM_IMPL") == 3 and "TMEM_SBP" not in p:
        # the one copy's sub-bank bit sits above the lane banks' (otpu_tmem: SBP >= log2 LANES):
        # the pick's SBP 3 is LANES 8's, a LANES 16 configuration's 4
        p["TMEM_SBP"] = max(3, (cfg.LANES - 1).bit_length())
    return p


def gathers(S: int, uarch: dict | None = None) -> bool:
    """The simulated collective unit runs GATHER: at S > 1 always, at S = 1 unless the uarch
    (UARCH, with `uarch` over it: BOARD_UARCH's COLL_GATHER 0) builds it BAR-only."""
    return S > 1 or bool(int({**UARCH, **(uarch or {})}.get("COLL_GATHER", 1)))


def build_top(cfg, dram_lat: int = 8, uarch: dict | None = None, axi: bool = False,
              dram_bytes: int | None = None, ldc: int | None = None) -> Path:
    ldc = (MEMORY["LDC"] if ldc is None else ldc) if axi else 0
    shell = RTL / "shell"
    mem = ([shell / "otpu_afifo.sv", shell / "otpu_mem_ch.sv", TB / LDC_MODELS[ldc],
            TB / "otpu_ldc_mem.sv"] if ldc else [TB / "otpu_native_mem.sv"] if axi else [])
    return build("tb_top", [RTL / s for s in RTL_SOURCES] + mem + [TB / "tb_top.sv"],
                 top_params(cfg, dram_lat, uarch, 2 if ldc else axi, dram_bytes))


def run(cfg, programs: list, images: list, *args, keep: Path | None = None, **kw):
    """Run the RTL; returns (drams as uint8 arrays, tmems as uint32 arrays, stats). The run's
    files (DRAM images, up to the machine's DRAM size) live in a temporary directory that is
    removed afterwards, on any exit (simtmp.tempdir: on disk where /tmp is in RAM), unless
    `keep` names a directory to leave them in."""
    if keep:
        return _run(cfg, programs, images, *args, keep=keep, **kw)
    with simtmp.tempdir("otpu_") as d:
        return _run(cfg, programs, images, *args, keep=d, **kw)


def _run(cfg, programs: list, images: list, dram_lat: int = 8, max_cycles: int = 50_000_000,
        keep: Path | None = None, trace: bool = False, uarch: dict | None = None,
        axi: bool | None = None, boot: bool | None = None, stall: int | None = None,
        seed: int | None = None, bw: int | None = None, lat: int | None = None,
        plusargs: list | None = None, args=None, ldc: int | None = None,
        pokes: dict | None = None, again: list | None = None, reload: bool = False,
        error_ok: bool = False):
    """Run the RTL; returns (drams as uint8 arrays, tmems as uint32 arrays, stats). args: the
    run's arguments (R8..R15 at the start, as isasim.Machine). ldc: MEMORY's LDC for this run
    (the core at MEMORY's LDC_MHZ unless `plusargs` set +ldc_ratio: ldc_plusargs). pokes:
    {slice: [(cycle, DRAM byte address, uint32 word)]}, the host's writes during the run (the
    behavioural DRAM, or the native memory model: not the LDC). again: more runs of the same
    programs in the same simulation, after the first halts, as the host runs a program again
    (the memory path is not reset between them; tb_top +runs), each {"args": its arguments
    (default: the run before's), "pokes": {slice: [(DRAM byte address, uint32 word)]}, the
    host's writes before it (the native memory model only)}; reload: the programs loaded again
    before each (boot only). error_ok: a run that stops with an error (an illegal instruction,
    a WAITW timeout) returns its state too, stats["error"] set. The result is the state after
    the last run."""
    from . import isa as I
    run_args = args
    axi = MEMORY["AXI"] if axi is None else axi
    axi = axi and cfg.D == 128
    boot = MEMORY["BOOT"] if boot is None else boot
    # ICACHE: the program runs from DRAM, where the loader finds it (boot)
    if int({**UARCH, **(uarch or {})}.get("ICACHE", int(getattr(cfg, "ICACHE", False)))):
        boot = True
    stall = MEMORY["STALL"] if stall is None else stall
    seed = MEMORY["SEED"] if seed is None else seed
    ldc = (MEMORY["LDC"] if ldc is None else ldc) if axi else 0
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
    # EXPERIMENTAL (docs/seq_vpu.md): the compiler's wait entries (opentpu.waits) for otpu_seq
    # WAITC: in DRAM after the program's last chunk, 64 bits each (boot: the loader reads them
    # into the sidecar, docs/isa.md "Wait counts"), else from +hz_sync_s<k>. A WAITC run takes
    # them unless OTPU_WAITS=0 or the simulation names its own (+hz_sync)
    nins = [len(p) // 8 for p in progs]
    sync_arg: list = []
    ua = {**UARCH, **(uarch or {})}
    if ua.get("SEQ_WC") and ua.get("WIN", 16) > cfg.WC_WIN:
        raise ValueError(f"SEQ_WC (otpu_seq WAITC) at WIN {ua['WIN']}: the entries assume a window "
                         f"of {cfg.WC_WIN} (Config.WC_WIN)")
    wmode = os.environ.get("OTPU_WAITS")
    if wmode is None and "hz_sync" not in os.environ.get("OTPU_SIM_PLUSARGS", ""):
        wmode = "1"
    if wmode and wmode != "0" and ua.get("SEQ_WC"):
        from . import waits as W
        for s in range(cfg.S):
            wst: dict = {}
            ww = W.wait_counts(programs[s], cfg.D, cfg.S, stats=wst,
                               args=run_args if wmode == "args" else None, aw=cfg.AW,
                               win=cfg.WC_WIN)
            print(f"waits: slice {s}: {len(ww)} instructions, {wst['visits']} visits, "
                  f"{len(wst['widened'])} loops widened, {wst['unreached']} unreached",
                  flush=True)
            if boot:
                # as Board.load_program: the slices load one PROG_N, so a shorter program is
                # padded with NOPs to the longest (their entries 0) before its entries
                n = 8 * max(nins)
                w = np.concatenate([progs[s], np.zeros(n - len(progs[s]), "<u4")])
                progs[s] = W.loadable(w, np.concatenate([np.asarray(ww, np.uint64),
                                                         np.zeros(n // 8 - len(ww), np.uint64)]),
                                      cfg.D)
            else:
                W.sync_file(ww, tmp / f"sync_{s}.txt")
                sync_arg.append(f"+hz_sync_s{s}={tmp / f'sync_{s}.txt'}")
    # boot: every slice's program right after the largest image (one loader address), or, if
    # there is no room, above the machine's DRAM (the simulated DRAM is then doubled, which moves
    # the aliases: an address past the machine's DRAM, read through its low bits, then lands in
    # the doubled DRAM's top half, on the program; a test that reaches DRAM by its aliases, as
    # test_rtl's 4 GiB range, sizes its image to leave the program room below DRAM_BYTES)
    at = -(-max(len(i) for i in imgs) // cfg.D) * cfg.D
    grow = boot and at + 4 * max(len(p) for p in progs) > cfg.DRAM_BYTES
    if grow:
        at = cfg.DRAM_BYTES
    exe = build_top(cfg, 20 if axi else dram_lat, uarch, axi,
                    2 * cfg.DRAM_BYTES if grow else None, ldc)
    for s in range(cfg.S):
        img = imgs[s]
        if boot:
            if 4 * len(progs[s]) > cfg.DRAM_BYTES:
                raise ValueError("no room in DRAM for the program")
            img = np.concatenate([img, np.zeros(at - len(img), np.uint8), progs[s].view(np.uint8)])
        img.view("<u4").astype(">u4").tofile(tmp / f"dram_{s}.bin")
    for s, pk in (pokes or {}).items():
        if axi and ldc:
            raise ValueError("pokes: the behavioural DRAM or the native memory model (not LDC)")
        (tmp / f"poke_{s}.txt").write_text("".join(
            f"{int(c):x} {int(a) // 4:x} {int(v) & 0xFFFFFFFF:x}\n" for c, a, v in sorted(pk)))
    again = list(again or [])
    between = {}
    for r, a in enumerate(again, 1):
        for s, pk in (a.get("pokes") or {}).items():
            if not axi or ldc:
                raise ValueError("again's pokes: the native memory model only (axi, not LDC)")
            between.setdefault(s, []).extend((r, int(ad), int(v)) for ad, v in pk)
    for s, pk in between.items():
        (tmp / f"pokeb_{s}.txt").write_text("".join(
            f"{r:x} {ad // 4:x} {v & 0xFFFFFFFF:x}\n" for r, ad, v in pk))
    args = [str(exe), f"+dir={tmp}", f"+max_cycles={max_cycles}"] + (["+trace"] if trace else [])
    args += list(plusargs or [])
    # e.g. OTPU_SIM_PLUSARGS=+hz_rh=1,+hz_log=/path (docs/seq_vpu.md) or +tm_nsb=8,+tm_sbp=3
    # (docs/tmem_nocopy.md; commas or spaces); %n is this process's simulation count (0, 1,
    # ...), so each run of a tool gets its own files
    global _SIM_RUNS
    args += [x.replace("%n", str(_SIM_RUNS))
             for x in _re.split(r"[\s,]+", os.environ.get("OTPU_SIM_PLUSARGS", "")) if x]
    _SIM_RUNS += 1
    args += sync_arg
    args += [f"+arg{k}={int(v) & 0xFFFFFFFF}" for k, v in enumerate(run_args or [])]
    if again:
        args += [f"+runs={1 + len(again)}"] + (["+reload"] if reload else [])
        for r, a in enumerate(again, 1):
            args += [f"+r{r}_arg{k}={int(v) & 0xFFFFFFFF}" for k, v in enumerate(a.get("args") or [])]
    if axi and not ldc and MEMORY["MODEL"]:
        args += _target.mem_model(_target.current(), MEMORY["MODEL"])
    if axi:
        args += [f"+axi_stall={stall}", f"+axi_seed={seed}",
                 f"+axi_bw={MEMORY['BW'] if bw is None else bw}",
                 f"+axi_lat={MEMORY['LAT'] if lat is None else lat}"]
    if ldc:                            # the first of a plusarg wins: the caller's
        args += ldc_plusargs(ldc, MEMORY["LDC_MHZ"])
    if boot:
        args += ["+boot", f"+boot_addr={at}", f"+boot_n={max(nins)}"]
    r = run_sim(args, timeout=3600)
    out = r.stdout + r.stderr
    import re
    if os.environ.get("OTPU_TMEMSTAT_LOG"):   # the TMEM's read demand (docs/tmem_pump.md)
        with open(os.environ["OTPU_TMEMSTAT_LOG"], "a") as f:
            f.writelines(m + "\n" for m in re.findall(r"(?:TMEMSTAT|TMMODEL) .*", out))
    m = re.search(r"RESULT cycles=(\d+) halted=(\d+) error=(\d+)", out)
    if not m:
        raise RuntimeError(f"RTL simulation failed:\n{out[-3000:]}")
    cycles, halted, err = (int(x) for x in m.groups())
    if not halted or (err and not error_ok):
        raise RuntimeError(f"RTL did not halt cleanly (halted={halted} error={err}):\n{out[-2000:]}")
    icounts = [int(x) for x in re.findall(r"SLICE \d+ icount=(\d+)", out)]
    drams = [np.fromfile(tmp / f"dram_out_{s}.bin", dtype=np.uint8) for s in range(cfg.S)]
    for s in range(cfg.S):             # a short dump (e.g. a full disk) is an error, not a result
        if len(drams[s]) < len(imgs[s]):
            raise RuntimeError(f"dram_out_{s}.bin has {len(drams[s])} bytes, fewer than the "
                               f"{len(imgs[s])}-byte image: the dump was cut short")
    if boot:                           # the program is not part of the result
        drams = [d[:cfg.DRAM_BYTES] for d in drams]
        for s in range(cfg.S):
            drams[s][at:at + 4 * len(progs[s])] = 0
    tmems = [_read_hex(tmp / f"tmem_{s}.hex", cfg.TMEM_WORDS) for s in range(cfg.S)]
    stats = {"cycles": cycles, "instructions": icounts}
    # per slice, the MXU's scale reads: port A line requests, and those that fetched a row's Sw
    # alone (otpu_mxu xs_*)
    stats["mxu_scale"] = [{"areq": int(a), "sw": int(w)} for a, w in
                          re.findall(r"MXSCALE s\d+ areq=(\d+) sw=(\d+)", out)]
    # per slice, the ACT RAM's reads (otpu_mxu xa_*): cycles, column reads (each D bytes and an
    # 8-byte scale record), pops (advances), the columns of output rows at the pops, MCOLS, and
    # the mask of the columns that ever read
    stats["act_reads"] = [dict(zip(("cyc", "rd", "pop", "use", "cols", "any"), map(int, m)))
                          for m in re.findall(r"MXACT s\d+ cyc=(\d+) rd=(\d+) pop=(\d+) "
                                              r"use=(\d+) cols=(\d+) any=(\d+)", out)]
    # per slice, the instruction cache's lines and prefetch, its fills (demand, prefetch) and
    # the cycles the sequencer waited for a line (otpu_seq ICACHE)
    stats["icache"] = [dict(zip(("lines", "pf", "fills", "demand", "prefetch", "stall"),
                                map(int, m)))
                       for m in re.findall(r"ICACHE s\d+ lines=(\d+) pf=(\d+) fills=(\d+) "
                                           r"demand=(\d+) prefetch=(\d+) stall=(\d+)", out)]
    if error_ok:
        stats["error"] = bool(err)
    if axi:
        # per channel: the model's reads, writes, row opens and partial writes, and the adapter's
        # counters; reads and writes as (commands, beats) (a native command is one beat), ar_a:
        # the A runs and SW fill reads, rmw_a: the A and SW writes with a partial byte mask
        mem = [dict(zip(("rd", "wr", "row_miss", "rmw"), map(int, m))) for m in
               re.findall(r"MEM ch\d+ rd=(\d+) wr=(\d+) row_miss=(\d+) rmw=(\d+)", out)]
        nat = [dict(kv.split("=") for kv in m.split()) for m in
               re.findall(r"NATIVE ch\d+ (.*)", out)]
        nat = [{k: int(v) for k, v in d.items()} for d in nat]
        stats["native"] = nat
        stats["axi_reads"] = [(m["rd"], m["rd"]) for m in mem]
        stats["axi_writes"] = [(m["wr"], m["wr"]) for m in mem]
        stats["axi_detail"] = [{"ar_a": n["a_runs"] + n["sw_rd"], "row_miss": m["row_miss"],
                                "rmw": m["rmw"], "rmw_a": n["part_a"] + n["part_sw"]}
                               for m, n in zip(mem, nat)]
        # the co-simulated controller's cycle breakdown (+ldc_break) and the H2C load's bursts
        # and bytes per channel (+h2c_bpk; sim/verilator/otpu_ldc_mem.sv)
        stats["brk"] = re.findall(r"^BRK .*$", out, re.M)
        stats["h2c"] = [dict(zip(("bursts", "bytes", "bresp"), map(int, m))) for m in
                        re.findall(r"H2C ch\d bursts=(\d+) bytes=(\d+) bresp=(\d+)", out)]
    if trace:
        stats["trace"] = out
    return drams, tmems, stats
