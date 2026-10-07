"""tools/qual/qual.sh without a card: a crashed step counts as a FAIL line, the model phases need
their checkpoints, the card must run the candidate's build, and the exit status is 0 only for a
qual that ran to its end without a FAIL line. The script runs from a scratch tree whose python
is a stub: each tool prints what the real one would on a pass, or crashes (a traceback, exit
status QUAL_CRASH_EXIT) when its command line contains QUAL_CRASH, or hangs when it contains
QUAL_HANG. The card is a state file: the build it runs (QUAL_BUILD at first, then each JTAG
load's: the bitstream's directory's last 8 hex digits; a load whose path contains QUAL_JTAG_FAIL
fails, one that contains QUAL_JTAG_NOOP leaves the card as it was), the probe of BUILD_ID and
CAPS reads it (QUAL_PROBE_FAIL: it crashes; QUAL_CAPS: bits 30 and 31), and dmesg prints
QUAL_DMESG, then QUAL_DMESG_NEW too from its second call on. The model check (python -c) runs
for real, against the scratch tree's copy of opentpu/llm and its models/."""
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from opentpu.llm import MODELS  # noqa: E402

STUB = r'''#!{python}
import os, sys, time
a = sys.argv[1:]
state = os.environ["QUAL_STATE"]
if a[0] == "-c" and "Board(" in a[1]:              # qual.sh's probe of BUILD_ID and CAPS
    if os.environ.get("QUAL_PROBE_FAIL"):
        print("Traceback (most recent call last):")
        print("OSError: [Errno 2] No such file or directory: '/dev/xdma0_user'")
        sys.exit(1)
    build = open(state).read().strip() if os.path.exists(state) else os.environ["QUAL_BUILD"]
    print("PROBE", build, os.environ.get("QUAL_CAPS", "0 0"))
    sys.exit(0)
if a[0] == "-c":                                    # qual.sh's model check: the real thing
    os.execv(sys.executable, [sys.executable] + a)
crash = os.environ.get("QUAL_CRASH")
if crash and crash in " ".join(a):
    print("Traceback (most recent call last):")
    print('  File "tool.py", line 1, in <module>')
    print("FileNotFoundError: [Errno 2] No such file or directory: 'models/Qwen3-0.6B/config.json'")
    sys.exit(int(os.environ.get("QUAL_CRASH_EXIT", "1")))
hang = os.environ.get("QUAL_HANG")
if hang and hang in " ".join(a):
    open(os.environ["QUAL_HUNG"], "w").write(str(os.getpid()))
    print("hanging", flush=True)
    time.sleep(600)
if a[:2] == ["-m", "opentpu.host.selftest"]:
    print("  [PASS] link       ID 0x4f545055  (0.0s)")
    print("ALL PASS")
elif a[:2] == ["-m", "opentpu.host.diag"]:
    print("summary")
    print("ALL PASS")
elif a[:2] == ["-m", "opentpu.host.smi"]:
    print("Temp 60 C")
elif a[:2] == ["-m", "opentpu.host.runstate"]:      # otpu-lock: prebuild, then CMD under the lock
    k = a.index("--")
    pre = [a[i + 1] for i, x in enumerate(a[:k]) if x == "--prebuild"]
    if pre:
        print("prebuilt", *pre, flush=True)
    os.execvpe(a[k + 1], a[k + 1:], dict(os.environ, OTPU_LOCK_HELD="xdma0"))
else:
    tool = os.path.basename(a[0])
    if tool == "refs.py" and a[1] == "cfg":
        open(a[2], "w").write("cfg")
        print("config")
    elif tool == "refs.py" and a[1] == "card":
        res = {"--resident", "--card-loop", "--prompt-runs"} & set(a)     # as refs.py's label
        what = "".join(f" {w}" for w, on in (("long", "--long" in a), ("resident", res),
                                              ("card loop", "--card-loop" in a),
                                              ("prompt runs", "--prompt-runs" in a)) if on)
        print(f"  [PASS] model {a[3]} {a[4]}/{a[5]}{what}: 32 tokens and the logits bit-exact")
    elif tool == "turnaround.py":
        print("  [PASS] DRAM turnarounds, data: 1500 runs in 30.0 s")
        print("  [PASS] DRAM turnarounds, ECC: ch0 sec 0 ded 0, ch1 sec 0 ded 0")
    elif tool == "waitw.py":
        print("  [PASS] WAITW on the host's writes: 200 rounds")
        print("  [PASS] WAITW timeout: ERROR at the timeout")
    elif tool == "refs.py" and a[1] == "compute":
        print("compute", *a[3:])
    elif tool == "decode_profile.py":
        print("wall 10.00 tok/s, device 10.00 tok/s")
    else:
        print(f"{tool} ok")
'''

LOADER = r'''#!{python}
import os, re, sys
bit = sys.argv[-1]
for k in ("QUAL_JTAG_FAIL", "QUAL_JTAG_NOOP"):
    v = os.environ.get(k)
    if v and v in bit:
        if k == "QUAL_JTAG_FAIL":
            print("unable to open ftdi device: -3 (device not found)")
            sys.exit(1)
        break
else:
    m = re.search(r"_([0-9a-f]{8})$", os.path.basename(os.path.dirname(os.path.realpath(bit))))
    open(os.environ["QUAL_STATE"], "w").write(m.group(1) if m else "00000000")
print("Load SRAM: [==================================================] 100.00%")
print("ir: 1 isc_done 1 isc_ena 0 init 1 done 1")
'''

SUDO = r'''#!{python}
import os, sys
a = " ".join(sys.argv[1:])
if "dmesg" in a:
    n = os.environ["QUAL_STATE"] + ".dmesg"
    calls = int(open(n).read()) + 1 if os.path.exists(n) else 1
    open(n, "w").write(str(calls))
    for k in ("QUAL_DMESG",) + (("QUAL_DMESG_NEW",) if calls > 1 else ()):
        if os.environ.get(k):
            print(os.environ[k])
elif "rescan" in a:
    print("otpu device 0000:01:00.0 ID 0x4f545055")
'''


@pytest.fixture
def qual(tmp_path):
    """run(models=True, **env) -> (stdout + stderr, checks.txt, exit status) of a fast qual.sh
    run of deploy_t_e776703a."""
    tree = tmp_path / "tree"
    (tree / "tools/qual").mkdir(parents=True)
    shutil.copy(ROOT / "tools/qual/qual.sh", tree / "tools/qual/qual.sh")
    (tree / "opentpu/llm").mkdir(parents=True)
    (tree / "opentpu/__init__.py").write_text("")
    shutil.copy(ROOT / "opentpu/llm/__init__.py", tree / "opentpu/llm/__init__.py")
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    for name, text in [("python", STUB), ("openFPGALoader", LOADER), ("sudo", SUDO),
                       ("timeout", "#!/bin/sh\nshift\nexec \"$@\"\n")]:   # timeout SECONDS CMD...
        (bin_ / name).write_text(text.replace("{python}", sys.executable))
        (bin_ / name).chmod(0o755)
    for d in ("deploy_t_e776703a", "deploy_prod_542fc43a"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "otpu.bit").write_text("bit")
    (tmp_path / "production").symlink_to(tmp_path / "deploy_prod_542fc43a")

    def env(models=True, **kw):
        if models:
            for d in MODELS.values():
                (tree / "models" / d).mkdir(parents=True, exist_ok=True)
                (tree / "models" / d / "config.json").write_text("{}")
        out = tmp_path / "out"
        for f in (tmp_path / "state", tmp_path / "state.dmesg"):
            f.unlink(missing_ok=True)
        e = dict(os.environ, PATH=f"{bin_}:{os.environ['PATH']}", PYTHON=str(bin_ / "python"),
                 LOAD="0", OUT=str(out), SOAK="1", QUAL_STATE=str(tmp_path / "state"),
                 QUAL_BUILD="e776703a", QUAL_HUNG=str(tmp_path / "hung"))
        e.update(kw)
        return e, out

    def run(models=True, **kw):
        e, out = env(models, **kw)
        r = subprocess.run(["bash", str(tree / "tools/qual/qual.sh"),
                            str(tmp_path / "deploy_t_e776703a"), "fast"], env=e,
                           capture_output=True, text=True, timeout=300)
        return r.stdout + r.stderr, (out / "checks.txt").read_text(), r.returncode
    run.env, run.tree = env, tree
    return run


def summary(text):
    return re.search(r"; (\d+) FAIL lines \((\d+) PASS\)", text).groups()


def verdict(text):
    return re.search(r"QUAL DONE deploy_t_e776703a \S+: (PASS|FAIL)", text).group(1)


def test_a_clean_run(qual):
    text, checks, rc = qual()
    # the 2 selftests', the build checks' (start, end), the 12 token-exact runs', the long
    # prompt's 3 and the 2 DRAM turnaround lines
    assert summary(text) == ("0", "21"), text[-2000:]
    assert checks.count("[PASS] model") == 15 and "[FAIL]" not in checks
    assert "prefill + decode counters" in text and "warm soak: " in text
    assert "[PASS] load + selftest: the candidate: the card runs build e776703a" in checks
    assert "[PASS] final selftest: the candidate, at the end: the card runs build e776703a" \
        in checks
    assert "[PASS] model lfm2 int8/- long resident prompt runs" in checks
    assert rc == 0 and verdict(text) == "PASS"


@pytest.mark.parametrize("crash, exit_code, where", [
    ("perf.py qwen3 int8", "1", "prefill + decode counters (6): perf qwen3 int8 -: exit 1"),
    ("decode_profile.py --model lfm2", "0", "decode_profile (3): decode_profile lfm2 fp4 int8: exit 0"),
    ("opentpu.host.diag", "1", "diag warm (quick memory test): diag warm: exit 1"),
])
def test_a_crashed_step_is_a_fail_line(qual, crash, exit_code, where):
    # a non-zero exit, or a traceback under exit 0, counts; before, the filters dropped both
    text, checks, rc = qual(QUAL_CRASH=crash, QUAL_CRASH_EXIT=exit_code)
    assert f"[FAIL] {where}, FileNotFoundError: [Errno 2]" in checks, checks
    assert summary(text)[0] == "1", text[-2000:]
    assert rc == 1 and verdict(text) == "FAIL"            # before: always 0


def test_a_failed_token_exact_run(qual):
    text, checks, rc = qual(QUAL_CRASH="qwen35 fp4 int8 32 --resident")
    assert "token-exact qwen35 fp4 int8--resident: exit 1" in checks
    assert "14 of 15 token-exact runs passed" in checks
    assert summary(text)[0] == "2" and rc == 1, text[-2000:]
    text, checks, rc = qual(QUAL_CRASH="lfm2 int8 - 32 --long --resident --prompt-runs")
    assert "token-exact long lfm2 int8 - --resident --prompt-runs: exit 1" in checks
    assert "14 of 15 token-exact runs passed" in checks and rc == 1


def test_the_decode_loop_on_the_card(qual):
    # a bitstream with CAPS bit30: the six token-exact runs and the long prompt's two (plain
    # and in prompt runs) again on the card's decode loop
    text, checks, rc = qual(QUAL_CAPS="1 0")
    assert summary(text) == ("0", "29"), text[-2000:]
    assert "decode loop on the card (6 + 2 long token-exact + 3 x 2 decode_profile)" in text
    assert "[PASS] model lfm2 int8/- long resident card loop prompt runs" in checks
    text, checks, rc = qual(GEN="1", QUAL_CRASH="--model qwen35 --tokens 96")    # the sampled run
    assert "decode_profile card loop qwen35 fp4 int8 sampled: exit 1" in checks
    text, checks, rc = qual(GEN="1", QUAL_CRASH="lfm2 int8 - 32 --card-loop")
    assert "card loop lfm2 int8 -: exit 1" in checks
    assert "7 of 8 card-loop token-exact runs passed" in checks
    text, _, _ = qual(QUAL_CAPS="1 0", GEN="0")                   # GEN overrides the bit
    assert "decode loop on the card (CAPS bit30): no" in text and "=== decode loop" not in text


def test_waitw_on_the_host_writes(qual):
    # a bitstream with CAPS bit31: tools/qual/waitw.py after the warm diag, its lines counted
    text, checks, rc = qual(QUAL_CAPS="0 1")
    assert summary(text) == ("0", "23"), text[-2000:]
    assert "WAITW (CAPS bit31): yes" in text and "=== WAITW on the host's writes" in text
    assert "[PASS] WAITW timeout" in checks
    text, checks, rc = qual(WAITW="1", QUAL_CRASH="waitw.py")
    assert "[FAIL] WAITW on the host's writes: waitw: exit 1" in checks
    text, _, _ = qual()
    assert "WAITW (CAPS bit31): no" in text and "=== WAITW" not in text


def test_the_dram_turnarounds(qual):
    # tools/qual/turnaround.py before the final selftest, in every qual: its lines counted, a
    # crash a FAIL line
    text, checks, rc = qual()
    assert "=== DRAM turnarounds + ECC (30 s)" in text
    assert "[PASS] DRAM turnarounds, ECC: ch0 sec 0 ded 0" in checks
    assert text.index("=== DRAM turnarounds") < text.index("=== final selftest")
    text, checks, rc = qual(QUAL_CRASH="turnaround.py", TURN="5")
    assert "=== DRAM turnarounds + ECC (5 s)" in text
    assert "[FAIL] DRAM turnarounds + ECC (5 s): turnaround: exit 1" in checks


def test_a_failed_soak_run_ends_the_soak(qual):
    text, checks, rc = qual(QUAL_CRASH="history of France", SOAK="30")
    assert checks.count("[FAIL]") == 1 and "soak run 1: exit 1" in checks
    assert "warm soak: 1 runs" in text


def test_runs_picks_the_models(qual, tmp_path):
    """RUNS: the references, prefill, decode_profile (4-bit only) and token-exact runs of the
    models it names, and no others; LONG names the long prompt's run (empty: none)."""
    text, checks, rc = qual(RUNS="lfm2-2.6b:int8:- smollm3:fp4:int8")
    # 2 selftests, 2 build checks, 4 token-exact, 3 for the long prompt, 2 turnaround
    assert summary(text) == ("0", "13"), text[-2000:]
    assert "[PASS] model lfm2-2.6b int8/-" in checks and "[PASS] model smollm3 fp4/int8 " \
        "resident" in checks and "qwen3" not in checks
    assert "compute --ntok 32 --runs lfm2-2.6b:int8:- smollm3:fp4:int8 lfm2:int8:-:long " \
        "lfm2:int8:-:long,pr" in (tmp_path / "out/refs.log").read_text()
    logs = sorted(f.name[4:-4] for f in (tmp_path / "out/logs").iterdir())
    assert [n for n in logs if n.startswith(("perf", "decode_profile"))] == [
        "decode_profile_smollm3_fp4_int8", "perf_lfm2-2.6b_int8_-", "perf_smollm3_fp4_int8"]
    text, checks, rc = qual(RUNS="lfm2:int8:-", LONG="", NTOK="8")
    assert summary(text) == ("0", "8") and "long" not in checks, text[-2000:]
    assert "compute --ntok 8 --runs lfm2:int8:-\n" in (tmp_path / "out/refs.log").read_text()


def test_without_models_the_model_phases_are_skipped(qual):
    # the staged tree of 2026-09-29 had no models link: every model phase crashed, 0 FAIL lines
    text, checks, rc = qual(models=False)
    assert "[FAIL] load + selftest: no checkpoint for lfm2" in checks and "models link" in checks
    assert "prefill + decode counters" not in text and "token-exact" not in text
    assert "diag warm" in text and summary(text) == ("1", "6"), text[-2000:]
    assert rc == 1


def test_prebuild_outside_the_lock_then_the_lock(qual):
    # started without the lock: the 4-bit runs into the image cache, then qual.sh again under
    # otpu-lock; PREBUILD=0 skips the prebuild; started under the lock, neither
    text, checks, rc = qual()
    assert "prebuilt qwen3:fp4:int8 lfm2:fp4:int8 qwen35:fp4:int8" in text
    assert text.index("=== prebuild outside the card lock") < text.index("################")
    assert summary(text) == ("0", "21"), text[-2000:]
    assert rc == 0                                      # through otpu-lock's exit status
    text, _, rc = qual(PREBUILD="0")
    assert "=== prebuild outside the card lock;" in text and "prebuilt" not in text
    assert summary(text) == ("0", "21"), text[-2000:]
    text, _, rc = qual(OTPU_LOCK_HELD="xdma0")
    assert "=== prebuild" not in text and summary(text) == ("0", "21"), text[-2000:]


def test_the_jtag_load_and_the_build_on_the_card(qual, tmp_path):
    # LOAD=1: the candidate's JTAG load, then its build on the card (the rescan only shows that
    # some openTPU bitstream answers); REST: its load and its build (production -> the deploy
    # directory) before the final selftest
    text, checks, rc = qual(LOAD="1", REST=str(tmp_path / "production/otpu.bit"))
    assert "[PASS] load + selftest: the candidate: the card runs build e776703a" in checks
    assert f"[PASS] final selftest: REST ({tmp_path}/production/otpu.bit): the card runs " \
        "build 542fc43a" in checks
    assert rc == 0 and summary(text) == ("0", "21"), text[-2000:]
    assert "deploy_t_e776703a/otpu.bit" in (tmp_path / "out/jtag.log").read_text()


def test_a_failed_jtag_load_stops_the_qual(qual):
    # openFPGALoader failed (the cable busy, a wrong path): before, its exit status was lost
    # in a pipe, the rescan found the old bitstream's ID, and the qual passed the candidate
    text, checks, rc = qual(LOAD="1", QUAL_BUILD="542fc43a", QUAL_JTAG_FAIL="deploy_t_")
    assert "[FAIL] load + selftest: the JTAG load of" in checks
    assert "openFPGALoader exit 1" in text and "the qual stops here" in text
    assert "token-exact" not in text and "[PASS] link" not in checks       # no selftest
    assert rc == 1 and summary(text)[0] == "1", text[-2000:]


def test_another_build_on_the_card_stops_the_qual(qual, tmp_path):
    # the loader said yes but the old bitstream runs on; LOAD=0 on a card that runs another
    # build; EXPECT names the build when the deploy's name does not
    text, checks, rc = qual(LOAD="1", QUAL_BUILD="542fc43a", QUAL_JTAG_NOOP="deploy_t_")
    assert "[FAIL] load + selftest: the candidate: the card runs build 542fc43a, not " \
        "e776703a" in checks
    assert rc == 1 and "token-exact" not in text and "the qual stops here" in text
    text, checks, rc = qual(QUAL_BUILD="542fc43a")
    assert "the card runs build 542fc43a, not e776703a" in checks and rc == 1
    text, checks, rc = qual(QUAL_BUILD="0badc0de", EXPECT="0badc0de")
    assert rc == 0 and "the card runs build 0badc0de" in checks
    # a REST that does not load: a FAIL line, no final selftest
    text, checks, rc = qual(LOAD="1", REST=str(tmp_path / "production/otpu.bit"),
                            QUAL_JTAG_FAIL="production")
    assert "[FAIL] final selftest: the JTAG load of REST" in checks and rc == 1


def test_a_failed_probe_stops_the_qual(qual):
    # before, a probe that failed gave GEN=0 WAITW=0: the card-loop and WAITW phases were
    # skipped without a FAIL line
    text, checks, rc = qual(QUAL_PROBE_FAIL="1")
    assert "[FAIL] load + selftest: probe (the candidate): exit 1" in checks
    assert "[FAIL] load + selftest: the candidate: the probe of BUILD_ID and CAPS failed" \
        in checks
    assert rc == 1 and "token-exact" not in text


def test_new_xdma_errors_are_fail_lines(qual):
    # the kernel's xdma errors since the load, each a FAIL line; older ones are not the qual's
    text, checks, rc = qual(QUAL_DMESG="[ 1234.500000] xdma:xdma_xfer_submit: xfer timed out",
                            QUAL_DMESG_NEW="[ 5678.250000] xdma:engine_status: engine error")
    assert checks.count("[FAIL]") == 1 and rc == 1
    assert "[FAIL] DRAM turnarounds + ECC (30 s): dmesg: [ 5678.250000] xdma:engine_status: " \
        "engine error" in checks


def test_the_references_job_is_bounded(qual, tmp_path):
    # a references' job that never ends: killed REFS_WAIT s after the last token-exact run
    # (before, qual.sh's wait for it had no bound), and a FAIL line
    text, checks, rc = qual(QUAL_HANG="refs.py compute", REFS_WAIT="2")
    assert "refs compute: still running 2 s after the last token-exact run: killed" in checks
    pid = int((tmp_path / "hung").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert rc == 1 and "=== DRAM turnarounds" in text


def test_a_killed_qual_kills_the_references_job_and_fails(qual, tmp_path):
    # SIGTERM while the references compute: the job goes too, the summary says FAIL, status 1
    e, out = qual.env(QUAL_HANG="refs.py compute")
    p = subprocess.Popen(["bash", str(qual.tree / "tools/qual/qual.sh"),
                          str(tmp_path / "deploy_t_e776703a"), "fast"], env=e,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    t0 = time.time()
    while not (tmp_path / "hung").exists() and time.time() - t0 < 60:
        time.sleep(0.2)
    time.sleep(0.5)
    p.send_signal(signal.SIGTERM)
    text = p.communicate(timeout=120)[0]
    assert p.returncode == 1, text[-2000:]
    assert "qual.sh exited early (status 143)" in (out / "checks.txt").read_text()
    assert "killed the references' job" in text and "QUAL DONE" in text
    with pytest.raises(ProcessLookupError):
        os.kill(int((tmp_path / "hung").read_text()), 0)
