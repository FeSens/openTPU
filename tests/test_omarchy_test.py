"""tools/omarchy_test.sh without omarchy: ssh is a stub that runs the remote command here, with
HOME a scratch directory (~/otpu-test/<name> under it), and flock one that always takes the
slot. The script ships the worktree it is in, whatever the current directory, says which tree
and commit, and deletes there what the worktree no longer tracks."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
                           "-c", "commit.gpgsign=false", *args], check=True, capture_output=True,
                          text=True).stdout.strip()


@pytest.mark.skipif(sys.platform != "darwin", reason="the script ships from the Mac (bsdtar)")
def test_ships_its_own_tree_and_drops_what_it_no_longer_tracks(tmp_path):
    wt = tmp_path / "wt"
    (wt / "tools").mkdir(parents=True)
    shutil.copy(ROOT / "tools/omarchy_test.sh", wt / "tools/omarchy_test.sh")
    (wt / "pkg").mkdir()
    for f, text in [("a.py", "A = 1\n"), ("pkg/__init__.py", ""), ("pkg/mod.py", "M = 1\n"),
                    ("tests/test_gone.py", "def test_x(): pass\n")]:
        (wt / f).parent.mkdir(parents=True, exist_ok=True)
        (wt / f).write_text(text)
    _git(wt, "init", "-q")
    _git(wt, "add", "-A")
    _git(wt, "commit", "-q", "-m", "one")
    bin_, home, elsewhere = tmp_path / "bin", tmp_path / "home", tmp_path / "elsewhere"
    for d in (bin_, home, elsewhere):
        d.mkdir()
    (bin_ / "ssh").write_text('#!/bin/bash\nshift\nexec bash -c "$*"\n')     # ssh HOST CMD
    (bin_ / "flock").write_text("#!/bin/sh\nexit 0\n")
    for f in ("ssh", "flock"):
        (bin_ / f).chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_}:{os.environ['PATH']}", HOME=str(home),
               OTPU_REMOTE_NAME="t")

    def run():                          # from another directory: the script's tree counts
        r = subprocess.run(["bash", str(wt / "tools/omarchy_test.sh"), "--exec", "true"],
                           cwd=elsewhere, env=env, capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr
        return r.stderr
    err = run()
    there = home / "otpu-test/t"
    assert f"omarchy_test: {wt.resolve()} at {_git(wt, 'rev-parse', '--short', 'HEAD')} ->" in err
    assert (there / "pkg/mod.py").exists() and (there / "tests/test_gone.py").exists()
    (there / "pkg/__pycache__").mkdir()
    (there / "pkg/__pycache__/mod.cpython-312.pyc").write_text("")
    (there / "build").mkdir()                           # the Verilator cache stays
    (there / "build/Vtb_top").write_text("")
    # a package and a test removed, a file edited and not committed
    _git(wt, "rm", "-q", "-r", "pkg", "tests/test_gone.py")
    _git(wt, "commit", "-q", "-m", "two")
    (wt / "a.py").write_text("A = 2\n")
    err = run()
    assert "+ 1 uncommitted" in err
    assert not (there / "pkg").exists() and not (there / "tests").exists()
    assert (there / "a.py").read_text() == "A = 2\n" and (there / "build/Vtb_top").exists()
    assert (there / "tools/omarchy_test.sh").exists() and (there / "models").is_symlink()
