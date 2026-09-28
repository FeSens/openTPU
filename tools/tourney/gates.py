"""Evaluation gates for one candidate (a slot worktree). Each gate raises GateFailure with a
short tail of its output; the orchestrator records `broken: <gate>: <tail>` and stops.

Order: sandbox -> lint -> fast (bit-exact subset) -> board (the same on the board's memory path
and micro-architecture) -> perf (Qwen3 proxy cycles) -> synth (area, fmax). The fmax objective
(--objective fmax) adds a last gate, full: the whole board built in Vivado on the build host.

Where lint and the test gates run: EXEC=remote (default) ships the slot's tree to omarchy (the
test host, remote.TEST_HOST; Vivado runs on remote.HOST) with tools/omarchy_test.sh and runs them
there (Verilator 5.046 and the venv on omarchy);
EXEC=local runs them on this machine. Either way they go through test_slot(): at most
TEST_SLOTS["remote"] = 2 at a time from the tournament on omarchy (its CPU is shared with
up to 2 Vivado jobs of other streams), TEST_SLOTS["local"] = 1 here.
"""
from __future__ import annotations

import fcntl
import fnmatch
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from contextlib import contextmanager

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


# ------------------------------------------------------------------------------ where tests run
TEST_LOCK = Path(os.environ.get("OTPU_TOURNEY_LOCKDIR", "/tmp")) / "otpu-tourney-tests.lock"
TEST_SLOTS = {"remote": 2, "local": 1}
REMOTE_SCRIPT = "tools/omarchy_test.sh"
# the Verilator cache on the build host, shared by the tournament's trees (keyed by source hash)
REMOTE_BUILD = "otpu-test/.tourney-build"


def exec_mode() -> str:
    m = os.environ.get("EXEC", "remote")
    if m not in TEST_SLOTS:
        raise ValueError(f"EXEC must be remote or local, not {m!r}")
    return m


@contextmanager
def test_slot(path: Path | None = None, slots: int | None = None, poll: float = 1.0):
    """One of `slots` test-gate slots (fcntl locks on <path>.<i>, released when the process
    dies): across slots and tournament processes, at most `slots` gates run at once."""
    path = path or TEST_LOCK
    slots = slots or TEST_SLOTS[exec_mode()]
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        for i in range(slots):
            f = open(f"{path}.{i}", "w")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                f.close()
                continue
            try:
                yield i
                return
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
                f.close()
        time.sleep(poll)


def remote_name(wt: Path) -> str:
    """The slot tree's name on the build host (~/otpu-test/<name>)."""
    return f"tourney-{wt.parent.name}-{wt.name}"


def command(wt: Path, cmd: list[str], env_extra: dict | None = None,
            mode: str | None = None) -> tuple[list[str], dict]:
    """The argv and environment that run `cmd` (argv[0] "python" = the test interpreter) in the
    slot tree `wt`, here or on the build host."""
    mode = mode or exec_mode()
    env = dict(os.environ)
    for k in ("OTPU_UARCH", "OTPU_AXI", "OTPU_BOOT"):
        env.pop(k, None)
    if mode == "local":
        env.update(env_extra or {})
        env["PYTHONPATH"] = str(wt)
        return [sys.executable if c == "python" else c for c in cmd], env
    env["OTPU_REMOTE_NAME"], env["OTPU_REMOTE_BUILD"] = remote_name(wt), REMOTE_BUILD
    kv = [f"{k}={v}" for k, v in (env_extra or {}).items()]
    return ["bash", REMOTE_SCRIPT, "--exec", "env", *kv, *cmd], env


def execute(wt: Path, cmd: list[str], gate: str, timeout: int, env_extra: dict | None = None):
    """Runs one gate command in a test slot; returns the CompletedProcess (stdout + stderr text).
    On the build host a timeout ends the ssh session; the remote run then dies at its next
    output (SIGPIPE), which frees its omarchy test slot."""
    argv, env = command(wt, cmd, env_extra)
    with test_slot():
        try:
            return subprocess.run(argv, cwd=wt, capture_output=True, text=True, timeout=timeout,
                                  env=env)
        except subprocess.TimeoutExpired:
            raise GateFailure(gate, f"timeout after {timeout}s")


PRUNE_CMD = (f"find ~/{REMOTE_BUILD}/verilator -mindepth 1 -maxdepth 1 -type d -mmin +1440 "
             f"-exec rm -rf {{}} + 2>/dev/null; true")


def remote_prune() -> None:
    """Drops Verilator builds older than a day from the shared cache on the build host (every
    candidate adds its own source hashes, ~0.4 GB each; a champion's are rebuilt in seconds)."""
    if exec_mode() != "remote":
        return
    from . import remote as R
    R.ssh(PRUNE_CMD, timeout=300, check=False, host=R.TEST_HOST)


def remote_clean(wt: Path) -> None:
    """Removes the slot's tree from the build host (EXEC=remote)."""
    if exec_mode() != "remote":
        return
    from . import remote as R
    R.ssh(f"rm -rf ~/otpu-test/{remote_name(wt)}", timeout=120, check=False,
          host=R.TEST_HOST)


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


# Constraint files may move logic (pblocks, placement, max fanout) but may not buy fmax by
# relaxing what is timed: no new timing exceptions and no clock changes.
XDC_FORBIDDEN = re.compile(r"\b(set_false_path|set_multicycle_path|set_max_delay|set_min_delay|"
                           r"set_clock_groups|set_disable_timing|create_clock|"
                           r"create_generated_clock|set_input_delay|set_output_delay|"
                           r"set_clock_uncertainty)\b")


def xdc_violations(diff: str) -> list[str]:
    """The added or removed lines of a constraint diff (unified format) that touch timing
    exceptions or clocks."""
    bad = []
    for line in diff.splitlines():
        if line.startswith(("+++", "---")) or not line[:1] in ("+", "-"):
            continue
        body = line[1:].split("#", 1)[0]
        if XDC_FORBIDDEN.search(body):
            bad.append(line.strip()[:160])
    return bad


def xdc_diff(wt: Path, paths: list[str]) -> str:
    """The unified diff of the changed .xdc / .tcl files, untracked ones as all-added."""
    out = []
    for p in paths:
        if not p.endswith((".xdc", ".tcl")):
            continue
        d = subprocess.run(["git", "-C", str(wt), "diff", "-U0", "HEAD", "--", p],
                           capture_output=True, text=True).stdout
        if not d and (wt / p).exists():
            d = "".join("+" + l + "\n" for l in (wt / p).read_text().splitlines())
        out.append(d)
    return "\n".join(out)


def sandbox(wt: Path, allowed: list[str]) -> list[str]:
    """Returns the RTL files the candidate changed; raises if anything else changed."""
    paths = changed_paths(wt)
    bad = offlimits(paths, allowed)
    if bad:
        raise GateFailure("sandbox", f"touched off-limits paths: {bad[:10]}")
    rtl = [p for p in paths if p not in NOTES]
    if not rtl:
        raise GateFailure("sandbox", "no RTL change")
    tcl = [p for p in rtl if p.endswith(".tcl")]
    if tcl:
        raise GateFailure("sandbox", f"build scripts are off limits: {tcl}")
    xdc = xdc_violations(xdc_diff(wt, rtl))
    if xdc:
        raise GateFailure("sandbox", f"constraint change touches timing exceptions or clocks: "
                                     f"{xdc[:5]}")
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
        r = execute(wt, cmd, "lint", timeout)
        if r.returncode != 0:
            raise GateFailure("lint", f"{top}: " + _tail(r.stdout + r.stderr))


# ------------------------------------------------------------------------------ tests
def pytest(wt: Path, nodes: list[str], env_extra: dict | None, gate: str,
           timeout: int = 3600) -> str:
    if not nodes:
        return "no tests"
    cmd = ["python", "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *nodes]
    r = execute(wt, cmd, gate, timeout, env_extra)
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
    weights are not available here (EXEC=local; the gate is then skipped). On the build host the
    tree's models/ links to its checkpoints."""
    if exec_mode() == "remote":
        m = "models/Qwen3-0.6B"
    else:
        m = models_dir(wt)
        if m is None:
            return None
    cmd = ["python", "tools/perf_qwen.py", "--model", str(m), "--layers", str(layers),
           "--bw", str(bw)]
    r = execute(wt, cmd, "perf", timeout)
    mm = re.search(r"^layers=\d+ .*?: (\d+) cycles", r.stdout, re.M)
    if r.returncode != 0 or not mm:
        raise GateFailure("perf", _tail(r.stdout + r.stderr))
    return int(mm.group(1))


# ------------------------------------------------------------------------------ synthesis
def synthesize(wt: Path, comp: dict, backend: str, out: Path,
               period_ns: float | None = None) -> dict:
    """Area and fmax of the component's synthesis parts. backend yosys runs here; vivado-remote
    runs each part out of context on the build host (synth, place, route at period_ns).
    `out` names the run (its last component keys the remote tree)."""
    if backend == "vivado":
        raise GateFailure("synth", "Vivado does not run on this machine: use --eval vivado-remote")
    parts = []
    for p in comp["synth"]["parts"]:
        try:
            if backend == "vivado-remote":
                from . import remote as R
                m = R.ooc(wt, f"{comp['name']}-{out.name}-{p['top']}", p, period_ns or 7.5)
            else:
                m = S.run(backend, wt, p["top"], p["sources"], p.get("params", {}),
                          out / p["top"])
        except Exception as e:  # noqa: BLE001 -- any tool failure is a broken candidate
            raise GateFailure("synth", _tail(str(e)))
        parts.append((m, float(p.get("weight", 1))))
    res = A.fitness(A.combine(parts)) if len(parts) > 1 or parts[0][1] != 1 else A.fitness(parts[0][0])
    res["parts"] = {p["top"]: {k: v for k, v in m.items() if k not in ("log", "log_tail")}
                    for p, (m, _) in zip(comp["synth"]["parts"], parts)}
    res["backend"] = backend
    if backend == "vivado-remote":
        res["collisions"] = sum(m.get("collisions") or 0 for m, _ in parts)
        res["whs"] = min((m["whs"] for m, _ in parts if m.get("whs") is not None), default=None)
        res["timing"] = "\n".join(f"[{p['top']}]\n{m.get('timing', '')}"
                                  for p, (m, _) in zip(comp["synth"]["parts"], parts))
    return res


# ------------------------------------------------------------------------------ full design
def full_design(wt: Path, name: str, core_mhz: float, build_id: str) -> dict:
    """The whole board built on the build host at core_mhz (the fmax objective's last gate).
    Raises when the build gives no core_clk timing; the accept rule judges the rest."""
    from . import remote as R
    try:
        return R.full(wt, name, core_mhz, build_id)
    except Exception as e:  # noqa: BLE001 -- a failed build is a broken candidate
        raise GateFailure("full", _tail(str(e)))
