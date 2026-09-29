"""Vivado on the build host for the tournament: EVAL=vivado-remote.

Vivado never runs on the Mac. A candidate's working tree (committed state plus its uncommitted
edits, without the build/ and models/ links) is sent with `git archive | ssh tar -x` to
`~/otpu-build/tv-<name>` on the host and built there, the same way `make bit` runs it.

Build hosts (OTPU_BUILD_HOSTS, comma-separated, in priority order; default
"opentpu,omarchy.tail5bd214.ts.net"): a job goes to the first host with room.
- opentpu: native Vivado 2026.1 (~/.local/bin/vivado), no card. Only ~/otpu-build/ is ours there.
- omarchy: native Vivado 2026.1 (~/.local/bin/vivado) too; second, as it also runs the test gates
  and the card.
Native commands put ~/.local/bin first on PATH (a non-interactive ssh may not have it). Hosts in
VIVADO_DOCKER_HOSTS use the vivado:2026.1 Docker image instead (VIVADO_AS_USER, the license
node-locked to VIVADO_MAC, the install mounted read-only). The test gates' host (TEST_HOST,
OTPU_REMOTE: omarchy) is separate: see gates.py.

Two kinds of job:
- ooc(): one component out of context: synth_design -mode out_of_context, place, phys_opt, route
  at the tournament's clock; post-route WNS -> fmax = 1000 / (period - WNS).
- full(): the whole board (`make bit DDR=1066 CORE_MHZ=<target>`); core_clk WNS / WHS from
  reports/SUMMARY.txt.

Both run detached on the host (nohup, a DONE file with the exit code) and are polled, so a dropped
ssh connection does not kill a 50-minute build. Each host runs at most MAX_JOBS Vivado jobs,
counting everything there (run_vivado.sh builds of other people and agents, Vivado containers,
our OOC jobs): acquire() blocks until some host has room.
"""
from __future__ import annotations

import fcntl
import os
import re
import shlex
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

HOSTS = [h.strip() for h in os.environ.get(
    "OTPU_BUILD_HOSTS", os.environ.get("OTPU_BUILD_HOST", "opentpu,omarchy.tail5bd214.ts.net")
).split(",") if h.strip()]
HOST = HOSTS[0]                                   # the default for ssh() without a host
DOCKER_HOSTS = {h.strip() for h in os.environ.get("VIVADO_DOCKER_HOSTS", "").split(",")
                if h.strip()}
# where tools/omarchy_test.sh runs the test gates (the same variable as the script's)
TEST_HOST = os.environ.get("OTPU_REMOTE", "omarchy.tail5bd214.ts.net")
NATIVE_PATH = "export PATH=$HOME/.local/bin:$PATH; "


def native(host: str | None = None) -> bool:
    return (host or HOST) not in DOCKER_HOSTS
REMOTE_DIR = os.environ.get("OTPU_BUILD_DIR", "otpu-build")        # relative to the remote $HOME
MAX_JOBS = int(os.environ.get("OTPU_MAX_VIVADO", "2"))
# per-host caps (host=n,...): opentpu takes one full build at a time. Two builds' IP synthesis
# phases (JOBS=2 each) filled its 31 GB and pushed 11 GB to swap (2026-09-28 00:07)
HOST_JOBS = {k.strip(): int(v) for k, v in (kv.split("=") for kv in os.environ.get(
    "OTPU_HOST_JOBS", "opentpu=1").split(",") if "=" in kv)}


# a running tournament can be moved between hosts without a restart: this file, when present,
# overrides OTPU_BUILD_HOSTS / OTPU_HOST_JOBS at every acquire(): lines `hosts=a,b` and `jobs=a=1,b=1`
HOSTS_FILE = Path(os.environ.get("OTPU_HOSTS_FILE", "/tmp/otpu-tourney-hosts"))


def _override() -> tuple[list[str] | None, dict | None]:
    try:
        txt = HOSTS_FILE.read_text()
    except OSError:
        return None, None
    hosts = jobs = None
    for line in txt.splitlines():
        k, _, v = line.strip().partition("=")
        if k == "hosts" and v:
            hosts = [h.strip() for h in v.split(",") if h.strip()]
        elif k == "jobs" and v:
            jobs = {a.strip(): int(b) for a, b in (kv.split("=") for kv in v.split(",") if "=" in kv)}
    return hosts, jobs


def max_jobs_on(host: str, default: int = MAX_JOBS, caps: dict | None = None) -> int:
    return min((HOST_JOBS if caps is None else caps).get(host, default), default)
IMAGE = os.environ.get("VIVADO_DOCKER", "vivado:2026.1")
MOUNT = os.environ.get("VIVADO_MOUNT", "/mnt/ml/Xilinx:/opt/Xilinx:ro")
SETTINGS = os.environ.get("VIVADO_SETTINGS", "/opt/Xilinx/2026.1/Vivado/settings64.sh")
MAC = os.environ.get("VIVADO_MAC", "02:42:ac:11:26:01")
LABEL = "otpu-tourney=1"
# extra `make bit` variables for every full build (champion and candidates alike): the image the
# tournament optimizes for. AXI_BL=32: 32-beat port B bursts, the production image's
# (deploy_pnbl32_e2521032; run_vivado.sh's default too, kept here for the cached results' names)
BUILD_ARGS = os.environ.get("OTPU_BUILD_ARGS", "AXI_BL=32").split()
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
# native: `total` counts every native Vivado process (a make bit's runs spawn several),
# `docker` the Vivado containers (another stream's Docker jobs on the same host)
NATIVE_COUNT_CMD = ("printf 'runviv %s\\n' \"$(pgrep -fc 'bash \\./[r]un_vivado')\"; "
                    "printf 'ours %s\\n' \"$(pgrep -fc '^bash [o]tpu_ooc\\.sh')\"; "
                    "printf 'total %s\\n' \"$(pgrep -fc '[u]nwrapped/lnx64\\.o/vivado')\"; "
                    "printf 'docker %s\\n' \"$(docker ps -q --filter ancestor=" + IMAGE +
                    " 2>/dev/null | wc -l)\"")


def count_cmd(host: str | None = None) -> str:
    return NATIVE_COUNT_CMD if native(host) else DOCKER_COUNT_CMD


def parse_counts(text: str) -> dict:
    """{'runviv': n, 'ours': n, 'total': n, 'docker': n} from count_cmd()'s output (missing -> 0)."""
    out = {"runviv": 0, "ours": 0, "total": 0, "docker": 0}
    for m in re.finditer(r"^(runviv|ours|total|docker) (\d+)\s*$", text, re.M):
        out[m.group(1)] = int(m.group(2))
    return out


def busy(c: dict, is_native: bool = True) -> int:
    """Vivado jobs running on the host. A `make bit` (anyone's, the tournament's full builds
    included) is a run_vivado.sh process and, while Vivado runs, one unlabelled container of the
    image: count it once. The tournament's OOC jobs are labelled containers with no
    run_vivado.sh.
    Native host: a make bit is its run_vivado.sh, an OOC job its otpu_ooc.sh, a Vivado container
    that is no make bit's (Docker make bits are run_vivado.sh too) one job; native Vivado
    processes under none of these (someone's own session) count as one job."""
    if is_native:
        n = c["runviv"] + c["ours"] + max(c["docker"] - c["runviv"], 0)
        return n if n or not c["total"] else 1
    other = max(c["total"] - c["ours"], 0)
    return max(c["runviv"], other) + c["ours"]


def jobs(host: str | None = None) -> int:
    host = host or HOST
    return busy(parse_counts(ssh(count_cmd(host), timeout=60, host=host)), native(host))


def _count(count, host: str) -> int:
    try:
        return count(host)
    except Exception as e:  # noqa: BLE001 -- an unreachable host has no room
        print(f"[tourney] could not count Vivado jobs on {host}: {e}")
        return 1 << 20


def free_slots(count=jobs) -> int:
    """Vivado jobs the build hosts could start now (their caps less what runs there; an
    unreachable host counts as full). A round full-builds that many candidates at once."""
    o_hosts, caps = _override()
    return sum(max(max_jobs_on(h, MAX_JOBS, caps) - _count(count, h), 0)
               for h in (o_hosts or HOSTS))


@contextmanager
def acquire(poll: int = 60, count=jobs, sleep=time.sleep, log=print, max_jobs: int = MAX_JOBS,
            hosts: list[str] | None = None):
    """Blocks until a host (in priority order) runs fewer than max_jobs Vivado jobs, then holds
    the local start lock while the caller starts its job there (release happens once the job is
    visible remotely, i.e. when the with-block ends). Yields (host, jobs running there).
    Without `hosts`, the hosts file is re-read at every poll, so a wait that started at cap 0
    picks up a raised cap (or a host taken out of the list)."""
    caps = None
    from_file = hosts is None
    START_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with START_LOCK.open("w") as f:
        waited = False
        while True:
            if from_file:
                o_hosts, caps = _override()
                hosts = o_hosts or HOSTS
            fcntl.flock(f, fcntl.LOCK_EX)
            ns = []
            for h in hosts:
                n = _count(count, h)
                ns.append(n)
                if n < max_jobs_on(h, max_jobs, caps):
                    break
            else:
                fcntl.flock(f, fcntl.LOCK_UN)
                if not waited:
                    log("[tourney] build hosts busy (" + ", ".join(
                        f"{h} {n}/{max_jobs_on(h, max_jobs, caps)}" for h, n in zip(hosts, ns)) +
                        " Vivado jobs): waiting")
                    waited = True
                sleep(poll)
                continue
            break
        try:
            yield h, n
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# ------------------------------------------------------------------------------ trees
_HOME: dict[str, str] = {}


def remote_home(host: str | None = None) -> str:
    host = host or HOST
    if host not in _HOME:
        _HOME[host] = ssh("echo $HOME", timeout=60, host=host).strip()
    return _HOME[host]


def remote_tree(name: str, host: str | None = None) -> str:
    return f"{remote_home(host)}/{REMOTE_DIR}/tv-{name}"


def tree_id(wt: Path) -> str:
    """A git tree of the worktree's current contents (tracked + untracked, uncommitted edits
    included), written through a temporary index (outside the tree, so neither it nor its lock
    file is picked up) so the worktree's own index is untouched. The build/ and models/ links are
    left out: build/ by pathspec, models/ by .gitignore (an explicit exclude of an ignored path
    makes `git add` fail)."""
    fd, name = tempfile.mkstemp(prefix="otpu-tourney-index-")
    os.close(fd)
    idx = Path(name)
    env = dict(os.environ, GIT_INDEX_FILE=str(idx))
    try:
        subprocess.run(["git", "read-tree", "HEAD"], cwd=wt, env=env, check=True,
                       capture_output=True)
        # an explicit exclude of an ignored path fails `git add`: the build/ *directory* is
        # ignored (.gitignore build/), a build *symlink* (the slot worktrees') is not
        ex = [f":!{d}" for d in ("build", "models") if (wt / d).is_symlink()
              and not subprocess.run(["git", "check-ignore", "-q", d], cwd=wt).returncode == 0]
        r = subprocess.run(["git", "add", "-A", "--", ".", *ex], cwd=wt, env=env,
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RemoteError(f"git add for the upload failed: {r.stderr[-600:]}")
        return subprocess.run(["git", "write-tree"], cwd=wt, env=env, check=True,
                              capture_output=True, text=True).stdout.strip()
    finally:
        idx.unlink(missing_ok=True)


def upload(wt: Path, name: str, host: str | None = None) -> str:
    """Sends the worktree to the host as a fresh tree; returns its remote path."""
    dst = remote_tree(name, host)
    tree = tree_id(wt)
    tar = subprocess.run(["git", "archive", "--format=tar", tree], cwd=wt, check=True,
                         capture_output=True).stdout
    ssh(f"rm -rf {shlex.quote(dst)} && mkdir -p {shlex.quote(dst)} && "
        f"tar -x -C {shlex.quote(dst)} && echo {tree} > {shlex.quote(dst)}/TREE", timeout=600,
        input=tar, host=host)
    return dst


def remove(name: str, host: str | None = None) -> None:
    ssh(f"rm -rf {shlex.quote(remote_tree(name, host))}", timeout=300, check=False, host=host)


def fetch(path: str, timeout: int = 300, host: str | None = None) -> str:
    return ssh(f"cat {shlex.quote(path)} 2>/dev/null || true", timeout=timeout, host=host)


# ------------------------------------------------------------------------------ detached jobs
def vivado_cmd(tree: str, script: str, log: str, is_native: bool = True) -> str:
    """The command line for one Vivado batch script: a docker run (as `make bit` runs Vivado)
    or, native, the OOC wrapper script next to it (written by ooc())."""
    if is_native:
        return f"{NATIVE_PATH}cd {shlex.quote(os.path.dirname(script))} && bash {OOC_SCRIPT}"
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
             is_native: bool = True) -> str:
    """`make bit` for the whole board, as the production builds run it."""
    if is_native:
        env = f"BUILD_ID={build_id} CORE_MHZ={core_mhz:g} JOBS={jobs}"
        return (f"{NATIVE_PATH}cd {tree}/boards/ypcb-00338 && {env} make bit DDR=1066"
                + "".join(f" {shlex.quote(a)}" for a in BUILD_ARGS))
    env = (f"BUILD_ID={build_id} CORE_MHZ={core_mhz:g} JOBS={jobs} VIVADO_DOCKER={IMAGE} "
           f"VIVADO_AS_USER=1 VIVADO_MOUNT={MOUNT} VIVADO_SETTINGS={SETTINGS} "
           f"VIVADO_MAC={MAC} XILINXD_LICENSE_FILE=$HOME/Xilinx.lic")
    return (f"cd {tree}/boards/ypcb-00338 && {env} make bit DDR=1066"
            + "".join(f" {shlex.quote(a)}" for a in BUILD_ARGS))


def start_detached(tree: str, cmd: str, tag: str, host: str | None = None) -> None:
    """Runs `cmd` on the host under nohup; its exit code lands in <tree>/<tag>.DONE."""
    inner = f"{cmd}; echo $? > {tree}/{tag}.DONE"
    # the braces: `&` backgrounds only the job, whose output is redirected, so ssh returns at
    # once (a trailing `&` on the whole && list kept a subshell on ssh's stdout until the build
    # ended; the start then timed out and the tree was removed under the running build)
    ssh(f"mkdir -p {tree}/.home && rm -f {tree}/{tag}.DONE && "
        f"{{ setsid nohup bash -c {shlex.quote(inner)} > {tree}/{tag}.out 2>&1 < /dev/null & }}",
        timeout=120, host=host)


def idle(tree: str, tag: str, host: str | None = None) -> bool:
    """True when no job `tag` can be running in `tree`: never started, or finished. A tree is
    only removed when idle (never under a running build)."""
    s = ssh(f"test -e {tree}/{tag}.out || echo none; test -e {tree}/{tag}.DONE && echo done",
            timeout=60, check=False, host=host)
    return "none" in s or "done" in s


def wait_done(tree: str, tag: str, timeout: int, poll: int = 60, sleep=time.sleep,
              host: str | None = None) -> int:
    t0 = time.time()
    while True:
        try:
            s = ssh(f"cat {tree}/{tag}.DONE 2>/dev/null || true", timeout=60, check=False,
                    host=host).strip()
        except subprocess.TimeoutExpired:    # a slow ssh is not a failed build: poll again
            s = ""
        if s:
            return int(s) if s.lstrip("-").isdigit() else 1
        if time.time() - t0 > timeout:
            raise RemoteError(f"{tag} on {host or HOST}:{tree} did not finish in {timeout}s")
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
    """One component out of context on a build host. Returns the synth.parse_ooc() dict plus
    'timing' (the worst paths), 'log_tail' and 'host'."""
    from . import synth as S
    with acquire() as (host, _):
        nat = native(host)
        tree = upload(wt, name, host)
        out = f"{tree}/build/tourney-ooc/{part['top']}"
        tcl = ooc_tcl(tree, part["top"], part["sources"], part.get("params", {}), out, period_ns)
        ssh(f"mkdir -p {out} && cat > {out}/ooc.tcl", input=tcl.encode(), timeout=120, host=host)
        if nat:
            ssh(f"cat > {out}/{OOC_SCRIPT}", input=native_script(f"{out}/ooc.tcl",
                                                                  f"{out}/vivado.log").encode(),
                timeout=120, host=host)
        start_detached(tree, vivado_cmd(tree, f"{out}/ooc.tcl", f"{out}/vivado.log", nat), "ooc",
                       host)
        time.sleep(20)                       # let the job appear before the lock goes
    try:
        rc = wait_done(tree, "ooc", timeout, host=host)
        log = fetch(f"{out}/vivado.log", timeout=600, host=host)
        res = S.parse_ooc(log, fetch(f"{out}/util.txt", host=host), period_ns)
        res["timing"] = S.worst_paths(fetch(f"{out}/timing.txt", host=host), 30)
        res["log_tail"] = log[-1500:]
        res["host"] = host
        if rc != 0 and res.get("wns") is None:
            raise RemoteError(f"OOC {part['top']} failed on {host} (exit {rc}): {log[-1500:]}")
        return res
    finally:
        if not keep and idle(tree, "ooc", host):
            remove(name, host)


# ------------------------------------------------------------------------------ full design
def full(wt: Path, name: str, core_mhz: float, build_id: str, timeout: int = 4 * 3600,
         keep: bool = False) -> dict:
    """The whole board built on a build host. Returns synth.parse_full() plus the reports the
    hypothesis agents read (worst paths, congestion, utilization) and 'host'."""
    from . import synth as S
    with acquire() as (host, _):
        tree = upload(wt, name, host)
        start_detached(tree, full_cmd(tree, core_mhz, build_id, is_native=native(host)), "full",
                       host)
        time.sleep(30)
    try:
        rc = wait_done(tree, "full", timeout, poll=120, host=host)
        rep = f"{tree}/build/vivado/reports"
        logs = ssh(f"cat {tree}/build/vivado/*.log {tree}/build/vivado/otpu.runs/*/runme.log "
                   f"2>/dev/null | grep -E 'Synth 8-6430|Route 35-447|ERROR:' || true",
                   timeout=300, check=False, host=host)
        res = S.parse_full(fetch(f"{rep}/SUMMARY.txt", host=host), logs,
                           fetch(f"{rep}/util.rpt", host=host))
        res["exit"] = rc
        res["host"] = host
        res["timing"] = S.worst_paths(ssh(f"head -c 3000000 {rep}/timing_worst.rpt 2>/dev/null"
                                          f" || true", timeout=300, check=False, host=host), 30)
        res["congestion_report"] = fetch(f"{rep}/congestion.rpt", host=host)[:8000]
        res["log_tail"] = fetch(f"{tree}/full.out", host=host)[-1500:]
        if res.get("wns") is None:
            raise RemoteError(f"full build failed on {host} (exit {rc}): {res['log_tail']}")
        return res
    finally:
        if not keep and idle(tree, "full", host):
            # the reports stay (small), the tree goes
            ssh(f"mkdir -p {remote_home(host)}/{REMOTE_DIR}/reports && "
                f"cp -r {tree}/build/vivado/reports {remote_home(host)}/{REMOTE_DIR}/reports/"
                f"tv-{name} 2>/dev/null; true", timeout=300, check=False, host=host)
            remove(name, host)
