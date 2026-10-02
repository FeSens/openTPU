"""tools/qual/qual.sh without a card: a crashed step counts as a FAIL line, and the model phases
need their checkpoints. The script runs from a scratch tree whose python is a stub: each tool
prints what the real one would on a pass, or crashes (a traceback, exit status QUAL_CRASH_EXIT)
when its command line contains QUAL_CRASH. The model check (python -c) runs for real, against the
scratch tree's copy of opentpu/llm and its models/."""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from opentpu.llm import MODELS  # noqa: E402

STUB = r'''#!{python}
import os, sys
a = sys.argv[1:]
if a[0] == "-c":                                    # qual.sh's model check: the real thing
    os.execv(sys.executable, [sys.executable] + a)
crash = os.environ.get("QUAL_CRASH")
if crash and crash in " ".join(a):
    print("Traceback (most recent call last):")
    print('  File "tool.py", line 1, in <module>')
    print("FileNotFoundError: [Errno 2] No such file or directory: 'models/Qwen3-0.6B/config.json'")
    sys.exit(int(os.environ.get("QUAL_CRASH_EXIT", "1")))
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
        print(f"  [PASS] model {a[3]} {a[4]}/{a[5]}{' resident' if '--resident' in a else ''}: "
              "'The capital of France is Paris.'")
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


@pytest.fixture
def qual(tmp_path):
    """run(models=True, **env) -> (stdout + stderr, checks.txt) of a fast qual.sh run."""
    tree = tmp_path / "tree"
    (tree / "tools/qual").mkdir(parents=True)
    shutil.copy(ROOT / "tools/qual/qual.sh", tree / "tools/qual/qual.sh")
    (tree / "opentpu/llm").mkdir(parents=True)
    (tree / "opentpu/__init__.py").write_text("")
    shutil.copy(ROOT / "opentpu/llm/__init__.py", tree / "opentpu/llm/__init__.py")
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    for name, text in [("python", STUB.replace("{python}", sys.executable)),
                       ("timeout", "#!/bin/sh\nshift\nexec \"$@\"\n"),   # timeout SECONDS CMD...
                       ("sudo", "#!/bin/sh\nexit 0\n")]:
        (bin_ / name).write_text(text)
        (bin_ / name).chmod(0o755)

    def run(models=True, **env):
        if models:
            for d in MODELS.values():
                (tree / "models" / d).mkdir(parents=True, exist_ok=True)
                (tree / "models" / d / "config.json").write_text("{}")
        out = tmp_path / "out"
        e = dict(os.environ, PATH=f"{bin_}:{os.environ['PATH']}", PYTHON=str(bin_ / "python"),
                 LOAD="0", OUT=str(out), SOAK="1")
        e.update(env)
        r = subprocess.run(["bash", str(tree / "tools/qual/qual.sh"), str(tmp_path / "deploy"),
                            "fast"], env=e, capture_output=True, text=True, timeout=300)
        return r.stdout + r.stderr, (out / "checks.txt").read_text()
    return run


def summary(text):
    return re.search(r"; (\d+) FAIL lines \((\d+) PASS\)", text).groups()


def test_a_clean_run(qual):
    text, checks = qual()
    # the 2 selftests', the 12 token-exact runs' and the 2 DRAM turnaround lines
    assert summary(text) == ("0", "16"), text[-2000:]
    assert checks.count("[PASS] model") == 12 and "[FAIL]" not in checks
    assert "prefill + decode counters" in text and "warm soak: " in text


@pytest.mark.parametrize("crash, exit_code, where", [
    ("perf.py qwen3 int8", "1", "prefill + decode counters (6): perf qwen3 int8 -: exit 1"),
    ("decode_profile.py --model lfm2", "0", "decode_profile (3): decode_profile lfm2 fp4 int8: exit 0"),
    ("opentpu.host.diag", "1", "diag warm (quick memory test): diag warm: exit 1"),
])
def test_a_crashed_step_is_a_fail_line(qual, crash, exit_code, where):
    # a non-zero exit, or a traceback under exit 0, counts; before, the filters dropped both
    text, checks = qual(QUAL_CRASH=crash, QUAL_CRASH_EXIT=exit_code)
    assert f"[FAIL] {where}, FileNotFoundError: [Errno 2]" in checks, checks
    assert summary(text)[0] == "1", text[-2000:]


def test_a_failed_token_exact_run(qual):
    text, checks = qual(QUAL_CRASH="qwen35 fp4 int8 32 --resident")
    assert "token-exact qwen35 fp4 int8--resident: exit 1" in checks
    assert "11 of 12 token-exact runs passed" in checks
    assert summary(text)[0] == "2", text[-2000:]


def test_the_decode_loop_on_the_card(qual):
    # a bitstream with CAPS bit30: the six token-exact runs again on the card's decode loop
    text, checks = qual(GEN="1")
    assert summary(text) == ("0", "22"), text[-2000:]
    assert "decode loop on the card (6 token-exact + 3 x 2 decode_profile)" in text
    text, checks = qual(GEN="1", QUAL_CRASH="--model qwen35 --tokens 96")    # the sampled run
    assert "decode_profile card loop qwen35 fp4 int8 sampled: exit 1" in checks
    text, checks = qual(GEN="1", QUAL_CRASH="lfm2 int8 - 32 --card-loop")
    assert "card loop lfm2 int8 -: exit 1" in checks
    assert "5 of 6 card-loop token-exact runs passed" in checks


def test_waitw_on_the_host_writes(qual):
    # a bitstream with CAPS bit31: tools/qual/waitw.py after the warm diag, its lines counted
    text, checks = qual(WAITW="1")
    assert summary(text) == ("0", "18"), text[-2000:]
    assert "WAITW (CAPS bit31): yes" in text and "=== WAITW on the host's writes" in text
    assert "[PASS] WAITW timeout" in checks
    text, checks = qual(WAITW="1", QUAL_CRASH="waitw.py")
    assert "[FAIL] WAITW on the host's writes: waitw: exit 1" in checks
    text, _ = qual()
    assert "WAITW (CAPS bit31): no" in text and "=== WAITW" not in text


def test_the_dram_turnarounds(qual):
    # tools/qual/turnaround.py before the final selftest, in every qual: its lines counted, a
    # crash a FAIL line
    text, checks = qual()
    assert "=== DRAM turnarounds + ECC (30 s)" in text
    assert "[PASS] DRAM turnarounds, ECC: ch0 sec 0 ded 0" in checks
    assert text.index("=== DRAM turnarounds") < text.index("=== final selftest")
    text, checks = qual(QUAL_CRASH="turnaround.py", TURN="5")
    assert "=== DRAM turnarounds + ECC (5 s)" in text
    assert "[FAIL] DRAM turnarounds + ECC (5 s): turnaround: exit 1" in checks


def test_a_failed_soak_run_ends_the_soak(qual):
    text, checks = qual(QUAL_CRASH="history of France", SOAK="30")
    assert checks.count("[FAIL]") == 1 and "soak run 1: exit 1" in checks
    assert "warm soak: 1 runs" in text


def test_runs_picks_the_models(qual, tmp_path):
    """RUNS: the references, prefill, decode_profile (4-bit only) and token-exact runs of the
    models it names, and no others."""
    text, checks = qual(RUNS="lfm2-2.6b:int8:- smollm3:fp4:int8")
    assert summary(text) == ("0", "8"), text[-2000:]    # 2 selftests, 4 token-exact, 2 turnaround
    assert "[PASS] model lfm2-2.6b int8/-" in checks and "[PASS] model smollm3 fp4/int8 " \
        "resident" in checks and "qwen3" not in checks
    assert "compute --runs lfm2-2.6b:int8:- smollm3:fp4:int8" in (tmp_path / "out/refs.log") \
        .read_text()
    logs = sorted(f.name[4:-4] for f in (tmp_path / "out/logs").iterdir())
    assert [n for n in logs if n.startswith(("perf", "decode_profile"))] == [
        "decode_profile_smollm3_fp4_int8", "perf_lfm2-2.6b_int8_-", "perf_smollm3_fp4_int8"]


def test_without_models_the_model_phases_are_skipped(qual):
    # the staged tree of 2026-09-29 had no models link: every model phase crashed, 0 FAIL lines
    text, checks = qual(models=False)
    assert "[FAIL] load + selftest: no checkpoint for lfm2" in checks and "models link" in checks
    assert "prefill + decode counters" not in text and "token-exact" not in text
    assert "diag warm" in text and summary(text) == ("1", "4"), text[-2000:]


def test_prebuild_outside_the_lock_then_the_lock(qual):
    # started without the lock: the 4-bit runs into the image cache, then qual.sh again under
    # otpu-lock; PREBUILD=0 skips the prebuild; started under the lock, neither
    text, checks = qual()
    assert "prebuilt qwen3:fp4:int8 lfm2:fp4:int8 qwen35:fp4:int8" in text
    assert text.index("=== prebuild outside the card lock") < text.index("################")
    assert summary(text) == ("0", "16"), text[-2000:]
    text, _ = qual(PREBUILD="0")
    assert "=== prebuild outside the card lock;" in text and "prebuilt" not in text
    assert summary(text) == ("0", "16"), text[-2000:]
    text, _ = qual(OTPU_LOCK_HELD="xdma0")
    assert "=== prebuild" not in text and summary(text) == ("0", "16"), text[-2000:]
