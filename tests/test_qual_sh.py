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
else:
    tool = os.path.basename(a[0])
    if tool == "refs.py" and a[1] == "cfg":
        open(a[2], "w").write("cfg")
        print("config")
    elif tool == "refs.py" and a[1] == "card":
        print(f"  [PASS] model {a[3]} {a[4]}/{a[5]}{' resident' if '--resident' in a else ''}: "
              "'The capital of France is Paris.'")
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
    assert summary(text) == ("0", "14"), text[-2000:]      # the 2 selftests' + 12 token-exact
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


def test_a_failed_soak_run_ends_the_soak(qual):
    text, checks = qual(QUAL_CRASH="history of France", SOAK="30")
    assert checks.count("[FAIL]") == 1 and "soak run 1: exit 1" in checks
    assert "warm soak: 1 runs" in text


def test_without_models_the_model_phases_are_skipped(qual):
    # the staged tree of 2026-09-29 had no models link: every model phase crashed, 0 FAIL lines
    text, checks = qual(models=False)
    assert "[FAIL] load + selftest: no checkpoint for lfm2" in checks and "models link" in checks
    assert "prefill + decode counters" not in text and "token-exact" not in text
    assert "diag warm" in text and summary(text) == ("1", "2"), text[-2000:]
