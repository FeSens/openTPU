"""Vivado on the build host for the tournament: EVAL=vivado-remote.

Vivado never runs on the Mac. A candidate's working tree (committed state plus its uncommitted
edits, without the build/ and models/ links) is sent with `git archive | ssh tar -x` to
`~/otpu-build/tv-<name>` on the host and built there, the same way `make bit` runs it.

Build hosts (OTPU_BUILD_HOST):
- opentpu (default): native Vivado 2026.1 (`vivado` on the ssh PATH), no card. Only
  ~/otpu-build/ is ours there.
- omarchy (OTPU_BUILD_HOST=omarchy.tail5bd214.ts.net): the vivado:2026.1 Docker image
  (VIVADO_AS_USER, the license node-locked to VIVADO_MAC, the native install mounted read-only).
VIVADO_NATIVE=1/0 overrides the choice (default: Docker on omarchy, native elsewhere). The test
gates' host (TEST_HOST, OTPU_REMOTE: omarchy) is separate: see gates.py.

Two kinds of job:
- ooc(): one component out of context: synth_design -mode out_of_context, place, phys_opt, route
  at the tournament's clock; post-route WNS -> fmax = 1000 / (period - WNS).
- full(): the whole board (`make bit DDR=1066 CORE_MHZ=<target>`); core_clk WNS / WHS from
  reports/SUMMARY.txt.

Both run detached on the host (nohup, a DONE file with the exit code) and are polled, so a dropped
ssh connection does not kill a 50-minute build. The host runs at most MAX_JOBS Vivado jobs,
counting everything there (run_vivado.sh builds of other people and agents, and our containers):
acquire() blocks until there is room.
"""
from __future__ import annotations

import fcntl
import os
import re
import shlex
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

HOST = os.environ.get("OTPU_BUILD_HOST", "opentpu")
# where tools/omarchy_test.sh runs the test gates (the same variable as the script's)
TEST_HOST = os.environ.get("OTPU_REMOTE", "omarchy.tail5bd214.ts.net")
NATIVE = os.environ.get("VIVADO_NATIVE", "0" if "omarchy" in HOST else "1") == "1"
REMOTE_DIR = os.environ.get("OTPU_BUILD_DIR", "otpu-build")        # relative to the remote $HOME
MAX_JOBS = int(os.environ.get("OTPU_MAX_VIVADO", "2"))
IMAGE = os.environ.get("VIVADO_DOCKER", "vivado:2026.1")
MOUNT = os.environ.get("VIVADO_MOUNT", "/mnt/ml/Xilinx:/opt/Xilinx:ro")
SETTINGS = os.environ.get("VIVADO_SETTINGS", "/opt/Xilinx/2026.1/Vivado/settings64.sh")
MAC = os.environ.get("VIVADO_MAC", "02:42:ac:11:26:01")
LABEL = "otpu-tourney=1"
PART = "xc7k480tffg1156-2"
# serializes our own job starts (the check-then-start below is not atomic across processes)
START_LOCK = Path(os.environ.get("OTPU_TOURNEY_LOCKDIR", "/tmp")) / "otpu-tourney-vivado.lock"


class RemoteError(RuntimeError):
    pass


def ssh(cmd: str, timeout: int = 300, input: bytes | None = None, check: bool = True,
        host: str | None = None) -> str:
    r = subprocess.run(["ssh", "-o", "ServerAliveInterval=30", "-o", "ConnectTimeout=20",
                        host or HOST, cmd], input=input, capture_output=True, timeout=timeout)
    out = r.stdout.decode(errors="replace")
    if check and r.returncode != 0:
        raise RemoteError(f"ssh {cmd[:120]!r} exited {r.returncode}: "
                          f"{r.stderr.decode(errors='replace')[-600:]}")
    return out


# ------------------------------------------------------------------------------ job counting
# the tournament's native OOC jobs run as `bash otpu_ooc.sh` (docker: labelled containers)
OOC_SCRIPT = "otpu_ooc.sh"
DOCKER_COUNT_CMD = ("printf 'runviv %s\\n' \"$(pgrep -fc 'bash \\./[r]un_vivado')\"; "
             "printf 'ours %s\\n' \"$(docker ps -q --filter label=" + LABEL + " | wc -l)\"; "
             "printf 'total %s\\n' \"$(docker ps -q --filter ancestor=" + IMAGE + " | wc -l)\"")
# native: `total` counts every Vivado process (a make bit's runs spawn several)
NATIVE_COUNT_CMD = ("printf 'runviv %s\\n' \"$(pgrep -fc 'bash \\./[r]un_vivado')\"; "
                    "printf 'ours %s\\n' \"$(pgrep -fc 'bash [o]tpu_ooc\\.sh')\"; "
                    "printf 'total %s\\n' \"$(pgrep -fc '[u]nwrapped/lnx64\\.o/vivado')\"")
COUNT_CMD = NATIVE_COUNT_CMD if NATIVE else DOCKER_COUNT_CMD


def parse_counts(text: str) -> dict:
    """{'runviv': n, 'ours': n, 'total': n} from COUNT_CMD's output (missing -> 0)."""
    out = {"runviv": 0, "ours": 0, "total": 0}
    for m in re.finditer(r"^(runviv|ours|total) (\d+)\s*$", text, re.M):
        out[m.group(1)] = int(m.group(2))
    return out


def busy(c: dict, native: bool | None = None) -> int:
    """Vivado jobs running on the host. A `make bit` (anyone's, the tournament's full builds
    included) is a run_vivado.sh process and, while Vivado runs, one unlabelled container of the
    image: count it once. The tournament's OOC jobs are labelled containers with no
    run_vivado.sh.
    Native host: a make bit is its run_vivado.sh, an OOC job its otpu_ooc.sh; Vivado processes
    under neither (someone's own session) count as one job."""
    if NATIVE if native is None else native:
        n = c["runviv"] + c["ours"]
        return n if n or not c["total"] else 1
    other = max(c["total"] - c["ours"], 0)
    return max(c["runviv"], other) + c["ours"]


def jobs() -> int:
    return busy(parse_counts(ssh(COUNT_CMD, timeout=60)))


@contextmanager
def acquire(poll: int = 60, count=jobs, sleep=time.sleep, log=print, max_jobs: int = MAX_JOBS):
    """Blocks until the host runs fewer than max_jobs Vivado jobs, then holds the local start
    lock while the caller starts its job (release happens once the job is visible remotely,
    i.e. when the with-block ends)."""
    START_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with START_LOCK.open("w") as f:
        waited = False
        while True:
            fcntl.flock(f, fcntl.LOCK_EX)
            n = count()
            if n < max_jobs:
                break
            fcntl.flock(f, fcntl.LOCK_UN)
            if not waited:
                log(f"[tourney] build host busy ({n}/{max_jobs} Vivado jobs): waiting")
                waited = True
            sleep(poll)
        try:
            yield n
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# ------------------------------------------------------------------------------ trees
_HOME: str | None = None


def remote_home() -> str:
    global _HOME
    if _HOME is None:
        _HOME = ssh("echo $HOME", timeout=60).strip()
    return _HOME


def remote_tree(name: str) -> str:
    return f"{remote_home()}/{REMOTE_DIR}/tv-{name}"


def tree_id(wt: Path) -> str:
    """A git tree of the worktree's current contents (tracked + untracked, uncommitted edits
    included), written through a temporary index so the worktree's own index is untouched."""
    idx = wt / ".git-tourney-index"
    env = dict(os.environ, GIT_INDEX_FILE=str(idx))
    try:
        subprocess.run(["git", "read-tree", "HEAD"], cwd=wt, env=env, check=True,
                       capture_output=True)
        subprocess.run(["git", "add", "-A", "--", ".", ":!build", ":!models",
                        ":!.git-tourney-index"], cwd=wt, env=env, check=True, capture_output=True)
        return subprocess.run(["git", "write-tree"], cwd=wt, env=env, check=True,
                              capture_output=True, text=True).stdout.strip()
    finally:
        idx.unlink(missing_ok=True)


def upload(wt: Path, name: str) -> str:
    """Sends the worktree to the host as a fresh tree; returns its remote path."""
    dst = remote_tree(name)
    tree = tree_id(wt)
    tar = subprocess.run(["git", "archive", "--format=tar", tree], cwd=wt, check=True,
                         capture_output=True).stdout
    ssh(f"rm -rf {shlex.quote(dst)} && mkdir -p {shlex.quote(dst)} && "
        f"tar -x -C {shlex.quote(dst)} && echo {tree} > {shlex.quote(dst)}/TREE", timeout=600,
        input=tar)
    return dst


def remove(name: str) -> None:
    ssh(f"rm -rf {shlex.quote(remote_tree(name))}", timeout=300, check=False)


def fetch(path: str, timeout: int = 300) -> str:
    return ssh(f"cat {shlex.quote(path)} 2>/dev/null || true", timeout=timeout)


# ------------------------------------------------------------------------------ detached jobs
def vivado_cmd(tree: str, script: str, log: str, native: bool | None = None) -> str:
    """The command line for one Vivado batch script: a docker run (as `make bit` runs Vivado)
    or, native, the OOC wrapper script next to it (written by ooc())."""
    if NATIVE if native is None else native:
        return f"cd {shlex.quote(os.path.dirname(script))} && bash {OOC_SCRIPT}"
    return docker_cmd(tree, script, log)


def native_script(script: str, log: str) -> str:
    return (f"#!/usr/bin/env bash\nvivado -mode batch -nojournal -log {shlex.quote(log)} "
            f"-source {shlex.quote(script)}\n")


def docker_cmd(tree: str, script: str, log: str) -> str:
    """The docker run line for one Vivado batch script (as `make bit` runs Vivado)."""
    lic = "$HOME/Xilinx.lic"
    return (f"docker run --rm --label {LABEL} -v {tree}:{tree} -w {tree} --mac-address {MAC} "
            f"-v {lic}:{lic}:ro -e XILINXD_LICENSE_FILE={lic} -v {MOUNT} "
            f"--user $(id -u):$(id -g) -e HOME={tree}/.home --entrypoint bash {IMAGE} "
            f"-lc {shlex.quote(f'source {SETTINGS} && vivado -mode batch -nojournal -log {log} -source {script}')}")


def full_cmd(tree: str, core_mhz: float, build_id: str, jobs: int = 2,
             native: bool | None = None) -> str:
    """`make bit` for the whole board, as the production builds run it."""
    if NATIVE if native is None else native:
        env = f"BUILD_ID={build_id} CORE_MHZ={core_mhz:g} JOBS={jobs}"
        return f"cd {tree}/boards/ypcb-00338 && {env} make bit DDR=1066"
    env = (f"BUILD_ID={build_id} CORE_MHZ={core_mhz:g} JOBS={jobs} VIVADO_DOCKER={IMAGE} "
           f"VIVADO_AS_USER=1 VIVADO_MOUNT={MOUNT} VIVADO_SETTINGS={SETTINGS} "
           f"VIVADO_MAC={MAC} XILINXD_LICENSE_FILE=$HOME/Xilinx.lic")
    return f"cd {tree}/boards/ypcb-00338 && {env} make bit DDR=1066"


def start_detached(tree: str, cmd: str, tag: str) -> None:
    """Runs `cmd` on the host under nohup; its exit code lands in <tree>/<tag>.DONE."""
    inner = f"{cmd}; echo $? > {tree}/{tag}.DONE"
    ssh(f"mkdir -p {tree}/.home && rm -f {tree}/{tag}.DONE && "
        f"nohup bash -c {shlex.quote(inner)} > {tree}/{tag}.out 2>&1 < /dev/null &", timeout=120)


def wait_done(tree: str, tag: str, timeout: int, poll: int = 60, sleep=time.sleep) -> int:
    t0 = time.time()
    while True:
        s = ssh(f"cat {tree}/{tag}.DONE 2>/dev/null || true", timeout=60, check=False).strip()
        if s:
            return int(s) if s.lstrip("-").isdigit() else 1
        if time.time() - t0 > timeout:
            raise RemoteError(f"{tag} on {HOST}:{tree} did not finish in {timeout}s")
        sleep(poll)


# ------------------------------------------------------------------------------ OOC
def ooc_tcl(tree: str, top: str, sources: list[str], params: dict, out: str,
            period_ns: float) -> str:
    files = " ".join(f"{{{tree}/{s}}}" for s in sources)
    gens = " ".join(f"-generic {k}={v}" for k, v in params.items())
    return f"""
create_project -in_memory -part {PART}
set_property source_mgmt_mode None [current_project]
read_verilog -sv {files}
synth_design -top {top} -part {PART} -mode out_of_context -flatten_hierarchy rebuilt \\
  -verilog_define SYNTHESIS {gens}
create_clock -period {period_ns} -name clk [get_ports clk]
opt_design
place_design
phys_opt_design
route_design
report_utilization -file {{{out}/util.txt}}
report_timing -max_paths 30 -nworst 1 -sort_by slack -file {{{out}/timing.txt}}
set p [lindex [get_timing_paths -setup -max_paths 1 -nworst 1] 0]
puts "OTPU_WNS [get_property SLACK $p]"
set h [lindex [get_timing_paths -hold -max_paths 1 -nworst 1] 0]
puts "OTPU_WHS [get_property SLACK $h]"
puts "OTPU_PERIOD {period_ns}"
"""


def ooc(wt: Path, name: str, part: dict, period_ns: float, timeout: int = 3 * 3600,
        keep: bool = False) -> dict:
    """One component out of context on the host. Returns the synth.parse_ooc() dict plus
    'timing' (the worst paths) and 'log_tail'."""
    from . import synth as S
    tree = upload(wt, name)
    out = f"{tree}/build/tourney-ooc/{part['top']}"
    tcl = ooc_tcl(tree, part["top"], part["sources"], part.get("params", {}), out, period_ns)
    ssh(f"mkdir -p {out} && cat > {out}/ooc.tcl", input=tcl.encode(), timeout=120)
    if NATIVE:
        ssh(f"cat > {out}/{OOC_SCRIPT}", input=native_script(f"{out}/ooc.tcl",
                                                              f"{out}/vivado.log").encode(),
            timeout=120)
    try:
        with acquire():
            start_detached(tree, vivado_cmd(tree, f"{out}/ooc.tcl", f"{out}/vivado.log"), "ooc")
            time.sleep(20)                       # let the container appear before the lock goes
        rc = wait_done(tree, "ooc", timeout)
        log = fetch(f"{out}/vivado.log", timeout=600)
        res = S.parse_ooc(log, fetch(f"{out}/util.txt"), period_ns)
        res["timing"] = S.worst_paths(fetch(f"{out}/timing.txt"), 30)
        res["log_tail"] = log[-1500:]
        if rc != 0 and res.get("wns") is None:
            raise RemoteError(f"OOC {part['top']} failed (exit {rc}): {log[-1500:]}")
        return res
    finally:
        if not keep:
            remove(name)


# ------------------------------------------------------------------------------ full design
def full(wt: Path, name: str, core_mhz: float, build_id: str, timeout: int = 4 * 3600,
         keep: bool = False) -> dict:
    """The whole board built on the host. Returns synth.parse_full() plus the reports the
    hypothesis agents read (worst paths, congestion, utilization)."""
    from . import synth as S
    tree = upload(wt, name)
    try:
        with acquire():
            start_detached(tree, full_cmd(tree, core_mhz, build_id), "full")
            time.sleep(30)
        rc = wait_done(tree, "full", timeout, poll=120)
        rep = f"{tree}/build/vivado/reports"
        logs = ssh(f"cat {tree}/build/vivado/*.log {tree}/build/vivado/otpu.runs/*/runme.log "
                   f"2>/dev/null | grep -E 'Synth 8-6430|Route 35-447|ERROR:' || true",
                   timeout=300, check=False)
        res = S.parse_full(fetch(f"{rep}/SUMMARY.txt"), logs, fetch(f"{rep}/util.rpt"))
        res["exit"] = rc
        res["timing"] = S.worst_paths(ssh(f"head -c 3000000 {rep}/timing_worst.rpt 2>/dev/null"
                                          f" || true", timeout=300, check=False), 30)
        res["congestion_report"] = fetch(f"{rep}/congestion.rpt")[:8000]
        res["log_tail"] = fetch(f"{tree}/full.out")[-1500:]
        if res.get("wns") is None:
            raise RemoteError(f"full build failed (exit {rc}): {res['log_tail']}")
        return res
    finally:
        if not keep:
            remove(name)

