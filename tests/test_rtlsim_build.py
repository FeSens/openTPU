"""opentpu.rtlsim.build's cache, with a fake Verilator: the key holds the flags, Verilator's
version and the sources' names; two builds of the same run Verilator once, and a build that
fails half-way leaves no executable for the next to take."""
import os
import sys
import threading

import pytest

from opentpu import rtlsim

FAKE = r'''#!{python}
import os, sys, time
a = sys.argv[1:]
if a == ["--version"]:
    print(os.environ.get("FAKE_VERSION", "Verilator 5.046 fake"))
    sys.exit(0)
open(os.environ["FAKE_LOG"], "a").write("build\n")
d, top = a[a.index("-Mdir") + 1], a[a.index("--top-module") + 1]
time.sleep(float(os.environ.get("FAKE_SLEEP", "0")))
with open(os.path.join(d, "V" + top), "w") as f:
    f.write("half")                        # a link that has begun
    if os.environ.get("FAKE_FAIL"):
        print("ld: error", file=sys.stderr)
        sys.exit(1)
    f.write(" and the rest")
'''


@pytest.fixture
def fake(tmp_path, monkeypatch):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "verilator").write_text(FAKE.replace("{python}", sys.executable))
    (bin_ / "verilator").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_}:{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "log"))
    monkeypatch.setattr(rtlsim, "BUILD", tmp_path / "build")
    monkeypatch.setattr(rtlsim, "_VERSION", None)
    src = tmp_path / "a.sv"
    src.write_text("module a; endmodule\n")
    return src, lambda: len((tmp_path / "log").read_text().splitlines()) \
        if (tmp_path / "log").exists() else 0


def test_the_key_holds_flags_version_and_names(fake, monkeypatch, tmp_path):
    src, _ = fake
    d0 = rtlsim.build_dir("a", [src], {"N": 1})
    assert d0 == rtlsim.build_dir("a", [src], {"N": 1})
    assert d0 != rtlsim.build_dir("a", [src], {"N": 2})
    flags = rtlsim.VFLAGS
    monkeypatch.setattr(rtlsim, "VFLAGS", flags + ["--assert"])
    assert rtlsim.build_dir("a", [src], {"N": 1}) != d0          # a flag (before: the same)
    monkeypatch.setattr(rtlsim, "VFLAGS", flags)
    monkeypatch.setattr(rtlsim, "_VERSION", "Verilator 5.047")
    assert rtlsim.build_dir("a", [src], {"N": 1}) != d0          # another Verilator
    monkeypatch.setattr(rtlsim, "_VERSION", None)
    other = tmp_path / "b.sv"
    other.write_text(src.read_text())
    assert rtlsim.build_dir("a", [other], {"N": 1}) != d0        # another file, same text


def test_concurrent_builds_run_verilator_once(fake, monkeypatch):
    src, builds = fake
    monkeypatch.setenv("FAKE_SLEEP", "1")
    got = []
    ts = [threading.Thread(target=lambda: got.append(rtlsim.build("a", [src]))) for _ in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert builds() == 1 and got[0] == got[1]
    assert got[0].read_text() == "half and the rest"


def test_a_failed_build_leaves_no_executable(fake, monkeypatch):
    src, builds = fake
    monkeypatch.setenv("FAKE_FAIL", "1")
    with pytest.raises(RuntimeError, match="verilator failed"):
        rtlsim.build("a", [src])
    assert not rtlsim.build_dir("a", [src]).exists()
    left = rtlsim.BUILD / f".{rtlsim.build_dir('a', [src]).name}.killed"    # a SIGKILLed one's
    left.mkdir()
    (left / "Va").write_text("half")
    monkeypatch.delenv("FAKE_FAIL")
    exe = rtlsim.build("a", [src])                  # before: the half-linked one, again
    assert builds() == 2 and exe.read_text() == "half and the rest"
    assert [p.name for p in rtlsim.BUILD.iterdir() if not p.name.endswith(".lock")] == \
        [exe.parent.name]
