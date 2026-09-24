"""Evaluation gates for one candidate (a slot worktree). Each gate raises GateFailure with a
short tail of its output; the orchestrator records `broken: <gate>: <tail>` and stops.

Order: sandbox -> lint -> fast (bit-exact subset) -> board (the same on the board's memory path
and micro-architecture) -> perf (Qwen3 proxy cycles) -> synth (area, fmax).
"""
from __future__ import annotations

import fnmatch
import os
import re
import subprocess
import sys
from pathlib import Path

from . import accept as A
from . import synth as S

# notes the agents are allowed to leave at the worktree root (never merged)
NOTES = ("HYPOTHESIS.md", "IMPLEMENTATION.md")
# symlinks the orchestrator puts into each worktree (shared build cache, model weights); a
# symlink does not match the `build/` / `models/` directory ignores, so skip them explicitly
SHARED = ("build", "models")

LINT_FLAGS = ["--lint-only", "-Wno-fatal", "-Wno-WIDTHEXPAND", "-Wno-WIDTHTRUNC",
              "-Wno-UNUSEDSIGNAL", "-Wno-UNUSEDPARAM", "-Wno-DECLFILENAME", "--timing"]
BOARD_ENV = {"OTPU_UARCH": "board", "OTPU_AXI": "1", "OTPU_BOOT": "1"}


class GateFailure(Exception):
    def __init__(self, gate: str, tail: str):
        super().__init__(f"{gate}: {tail}")
        self.gate, self.tail = gate, tail


def _tail(s: str, n: int = 1200) -> str:
    s = s.strip()
    return s if len(s) <= n else "..." + s[-n:]


# ------------------------------------------------------------------------------ sandbox
def changed_paths(wt: Path) -> list[str]:
    out = subprocess.run(["git", "-C", str(wt), "status", "--porcelain", "--untracked-files=all"],
                         capture_output=True, text=True, check=True).stdout
    paths = []
    for line in out.splitlines():
        if not line.strip():
            continue
        for p in (s.strip().strip('"') for s in line[3:].split(" -> ")):
            if p and p not in SHARED:
                paths.append(p)
    return paths


def offlimits(paths: list[str], allowed: list[str]) -> list[str]:
    """Paths not matching the component's allowed globs (or the agent notes)."""
    ok = list(allowed) + list(NOTES)
    return [p for p in paths if not any(fnmatch.fnmatch(p, g) for g in ok)]


# The fp operators' internals (stage functions, intermediate structs) belong to the otpu_fp
# tournament, which may change them at will: other components must use the operator modules
# (otpu_fmul / otpu_fadd / otpu_fmadd) or otpu_fp's whole-operation functions. A private
# operator built from the stages passes on its own branch and breaks when both are merged.
FP_FILES = ("rtl/vpu/otpu_fp.sv", "rtl/vpu/otpu_fpipe.sv")
FP_INTERNALS = re.compile(r"\b(fp_add_s\d|fp_mul_s\d|fadd_p\d_t|fadd_nm_t|fmul_mid_t)\b")


def fp_internal_uses(path: str, text: str) -> list[str]:
    """The fp stage internals a non-fp RTL file uses (empty for the fp files themselves)."""
    if path in FP_FILES or not path.endswith(".sv"):
        return []
    return sorted(set(m.group(1) for m in FP_INTERNALS.finditer(text)))


def sandbox(wt: Path, allowed: list[str]) -> list[str]:
    """Returns the RTL files the candidate changed; raises if anything else changed."""
    paths = changed_paths(wt)
    bad = offlimits(paths, allowed)
    if bad:
        raise GateFailure("sandbox", f"touched off-limits paths: {bad[:10]}")
    rtl = [p for p in paths if p not in NOTES]
    if not rtl:
        raise GateFailure("sandbox", "no RTL change")
    for p in rtl:
        f = wt / p
        uses = fp_internal_uses(p, f.read_text()) if f.exists() else []
        if uses:
            raise GateFailure("sandbox", f"{p} uses otpu_fp stage internals {uses}: use the "
                                         "operator modules (otpu_fmul/otpu_fadd/otpu_fmadd)")
    return rtl


# ------------------------------------------------------------------------------ lint
def _rtl_sources(wt: Path) -> list[str]:
    sys.path.insert(0, str(wt))
    try:
        txt = (wt / "opentpu" / "rtlsim.py").read_text()
        m = re.search(r"RTL_SOURCES\s*=\s*\[(.*?)\]", txt, re.S)
        return re.findall(r'"([^"]+)"', m.group(1))
    finally:
        sys.path.pop(0)


def lint(wt: Path, timeout: int = 900) -> None:
    """Verilator lint of the simulation top and of the board top (errors fail, warnings pass)."""
    rtl = [f"rtl/{s}" for s in _rtl_sources(wt)]
    sim = rtl + ["sim/verilator/otpu_axi_mem.sv"]
    board = [s for s in rtl if not s.endswith("otpu_top.sv")] + [
        "rtl/boards/ypcb-00338/otpu_ctrl.sv", "rtl/boards/ypcb-00338/otpu_trace.sv",
        "rtl/boards/ypcb-00338/otpu_board.sv"]
    for top, srcs, params in (("otpu_top", sim, ["-GD=128", "-GMCOLS=2", "-GAXI=1"]),
                              ("otpu_board", board, [])):
        cmd = ["verilator", *LINT_FLAGS, "--top-module", top, *params, *srcs]
        r = subprocess.run(cmd, cwd=wt, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            raise GateFailure("lint", f"{top}: " + _tail(r.stdout + r.stderr))


# ------------------------------------------------------------------------------ tests
def pytest(wt: Path, nodes: list[str], env_extra: dict | None, gate: str,
           timeout: int = 3600) -> str:
    if not nodes:
        return "no tests"
    env = dict(os.environ)
    for k in ("OTPU_UARCH", "OTPU_AXI", "OTPU_BOOT"):
        env.pop(k, None)
    env.update(env_extra or {})
    env["PYTHONPATH"] = str(wt)
    cmd = [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *nodes]
    try:
        r = subprocess.run(cmd, cwd=wt, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        raise GateFailure(gate, f"timeout after {timeout}s")
    out = r.stdout + r.stderr
    if r.returncode != 0:
        raise GateFailure(gate, _tail(out))
    last = [l for l in out.splitlines() if " passed" in l or " failed" in l]
    return last[-1] if last else "ok"


# ------------------------------------------------------------------------------ perf proxy
def models_dir(wt: Path) -> Path | None:
    """The Qwen3-0.6B weights: $OTPU_MODELS, the worktree's models/, or the main checkout's."""
    cands = []
    if os.environ.get("OTPU_MODELS"):
        cands.append(Path(os.environ["OTPU_MODELS"]))
    cands.append(wt / "models" / "Qwen3-0.6B")
    common = subprocess.run(["git", "-C", str(wt), "rev-parse", "--git-common-dir"],
                            capture_output=True, text=True).stdout.strip()
    if common:
        cands.append((wt / common).resolve().parent / "models" / "Qwen3-0.6B")
    for c in cands:
        if (c / "config.json").exists():
            return c
    return None


def perf(wt: Path, layers: int = 2, bw: int = 80, timeout: int = 3600) -> int | None:
    """Cycles of a Qwen3-0.6B decode token (first `layers` layers + the LM head) on the RTL at the
    board configuration, AXI memory path at `bw`% bandwidth (deterministic). None if the model
    weights are not available (the gate is then skipped)."""
    m = models_dir(wt)
    if m is None:
        return None
    env = dict(os.environ, PYTHONPATH=str(wt))
    cmd = [sys.executable, "tools/perf_qwen.py", "--model", str(m), "--layers", str(layers),
           "--bw", str(bw)]
    try:
        r = subprocess.run(cmd, cwd=wt, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        raise GateFailure("perf", f"timeout after {timeout}s")
    mm = re.search(r"^layers=\d+ .*?: (\d+) cycles", r.stdout, re.M)
    if r.returncode != 0 or not mm:
        raise GateFailure("perf", _tail(r.stdout + r.stderr))
    return int(mm.group(1))


# ------------------------------------------------------------------------------ synthesis
def synthesize(wt: Path, comp: dict, backend: str, out: Path) -> dict:
    parts = []
    for p in comp["synth"]["parts"]:
        try:
            m = S.run(backend, wt, p["top"], p["sources"], p.get("params", {}),
                      out / p["top"])
        except Exception as e:  # noqa: BLE001 -- any tool failure is a broken candidate
            raise GateFailure("synth", _tail(str(e)))
        parts.append((m, float(p.get("weight", 1))))
    res = A.fitness(A.combine(parts)) if len(parts) > 1 or parts[0][1] != 1 else A.fitness(parts[0][0])
    res["parts"] = {p["top"]: {k: v for k, v in m.items() if k != "log"}
                    for p, (m, _) in zip(comp["synth"]["parts"], parts)}
    res["backend"] = backend
    return res
