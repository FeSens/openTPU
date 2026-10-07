"""Who may drive the card (opentpu/host/runstate.py): otpu-lock never lets go while its command
runs, OTPU_LOCK_HELD is checked rather than trusted, and the shared run directory's files are
opened safely and shared by users. Real processes, no card."""
import os
import signal
import subprocess
import sys
import textwrap

import pytest

from opentpu.host import runstate as rs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = dict(os.environ, PYTHONPATH=ROOT)


@pytest.fixture(autouse=True)
def run_dir(tmp_path, monkeypatch):
    d = tmp_path / "otpu"
    monkeypatch.setenv("OTPU_RUN_DIR", str(d))
    for k in ("OTPU_LOCK_HELD", "OTPU_LOCK_PID"):
        monkeypatch.delenv(k, raising=False)
        ENV.pop(k, None)
    ENV["OTPU_RUN_DIR"] = str(d)
    return d


def _otpu_lock(dev: str, code: str, **kw):
    return subprocess.Popen([sys.executable, "-m", "opentpu.host.runstate", "--dev", dev, "--",
                             sys.executable, "-c", textwrap.dedent(code)],
                            stdout=subprocess.PIPE, text=True, env=ENV, **kw)


def test_killing_otpu_lock_ends_its_command_first():
    """SIGTERM to otpu-lock (an agent killing "the wrapper", `timeout`): it passes the signal
    on to CMD and exits after it, so the lock is never free while CMD runs (it exited at once,
    and a second runner took the card beside the first's CMD)."""
    p = _otpu_lock("/dev/fakeK", """
        import os, time
        from opentpu.host.runstate import DeviceLock
        lk = DeviceLock("fakeK")
        print(os.getpid(), lk.holder, flush=True)
        time.sleep(30)
    """)
    pid, holder = (int(x) for x in p.stdout.readline().split())
    assert holder == p.pid                              # inherited, from this otpu-lock
    p.send_signal(signal.SIGTERM)
    p.wait(timeout=20)
    assert not rs._running(pid)                         # CMD ended before otpu-lock did
    rs.DeviceLock("fakeK", wait=0).release()


def test_otpu_lock_held_is_checked_not_trusted(run_dir):
    """OTPU_LOCK_HELD names the device (as older trees' otpu-lock did, and their tools expect),
    OTPU_LOCK_PID otpu-lock's pid; a tool trusts them only when the lock file's pid is its
    ancestor, OTPU_LOCK_PID's when set, and holds the lock. A holder that is not an ancestor
    (another runner's lock), or another pid than OTPU_LOCK_PID's, is not the lock: the tool
    takes it itself (DeviceBusy here)."""
    tool = [sys.executable, "-c", "from opentpu.host.runstate import DeviceLock, DeviceBusy\n"
            "try:\n print(DeviceLock('fakeV', wait=0).holder)\nexcept DeviceBusy: print('busy')"]
    p = subprocess.Popen([sys.executable, "-c", "import time; from opentpu.host.runstate import "
                          "DeviceLock; lk = DeviceLock('fakeV'); print('locked', flush=True); "
                          "time.sleep(30)"], stdout=subprocess.PIPE, text=True, env=ENV)
    try:
        assert p.stdout.readline().strip() == "locked"
        for pid in (None, str(p.pid)):
            env = dict(ENV, OTPU_LOCK_HELD="fakeV", **({"OTPU_LOCK_PID": pid} if pid else {}))
            r = subprocess.run(tool, capture_output=True, text=True, env=env)
            assert r.stdout.strip() == "busy", (pid, r.stderr)
    finally:
        p.kill()
        p.wait()
    lk = rs.DeviceLock("fakeV", wait=0)                 # this process, the tool's parent
    try:
        for pid, want in ((None, str(os.getpid())), (str(os.getpid()), str(os.getpid())),
                          ("1", "busy")):
            env = dict(ENV, OTPU_LOCK_HELD="fakeV", **({"OTPU_LOCK_PID": pid} if pid else {}))
            r = subprocess.run(tool, capture_output=True, text=True, env=env)
            assert r.stdout.strip() == want, (pid, r.stderr)
    finally:
        lk.release()


# ------------------------------------------------------------------------------ run directory
def test_a_planted_symlink_is_not_followed(run_dir, tmp_path):
    """The shared run directory belongs to whoever made it first: a symbolic link there named
    as a lock is refused, and the file it points to is not truncated."""
    run_dir.mkdir(parents=True)
    victim = tmp_path / "victim"
    victim.write_text("keep me")
    (run_dir / "fakeY.lock").symlink_to(victim)
    with pytest.raises(rs.UnsafePath, match="symbolic link"):
        rs.DeviceLock("fakeY", wait=0)
    (run_dir / "fakeY.dma").symlink_to(victim)
    with pytest.raises(rs.UnsafePath):
        rs.open_shared(run_dir / "fakeY.dma")
    assert victim.read_text() == "keep me"


def test_lock_files_are_shared_by_users(run_dir):
    """Lock files are mode 0666 whatever the umask (0644 left a second user PermissionError
    instead of DeviceBusy), and a lock file this user cannot write is locked read-only."""
    old = os.umask(0o022)
    try:
        lk = rs.DeviceLock("fakeU")
        assert (run_dir / "fakeU.lock").stat().st_mode & 0o777 == 0o666
        lk.release()
        os.chmod(run_dir / "fakeU.lock", 0o444)         # as another user's 0644 file is
        lk = rs.DeviceLock("fakeU")
        if os.geteuid() != 0:
            with pytest.raises(rs.DeviceBusy):
                rs.DeviceLock("fakeU", wait=0)
        lk.release()
    finally:
        os.umask(old)


def test_a_status_file_it_cannot_replace_is_not_fatal(run_dir, monkeypatch, capsys):
    """Another user's status file left in the sticky directory cannot be replaced: the runner
    says so once and goes on (it raised PermissionError from BoardBackend's open)."""
    def no(*a):
        raise PermissionError(1, "Operation not permitted")
    monkeypatch.setattr(os, "replace", no)
    st = rs.RunnerStatus("fakeR")
    st.token(1000, 100_000)
    st.remove()
    assert capsys.readouterr().err.count("no status file") == 1
