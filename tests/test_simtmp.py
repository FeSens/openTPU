"""opentpu.simtmp: where the simulators' temporary directories go, and that they go away on
any exit (return, exception, SIGTERM: the simulator killed too; a SIGKILLed run's at the next
run's sweep)."""
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from opentpu import simtmp

ROOT = Path(__file__).resolve().parent.parent
MOUNTS = "/dev/nvme0n1p2 / ext4 rw 0 0\ntmpfs /otpu-fake/tmp tmpfs rw,nosuid 0 0\n"


def test_root_env_then_ram_filesystems(monkeypatch, tmp_path):
    monkeypatch.setenv(simtmp.ENV, str(tmp_path))
    assert simtmp.root(MOUNTS) == tmp_path
    monkeypatch.delenv(simtmp.ENV)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: "/otpu-fake/tmp")
    assert simtmp.fstype("/otpu-fake/tmp/x", MOUNTS) == "tmpfs"
    assert simtmp.root(MOUNTS) == simtmp.DISK
    monkeypatch.setattr(tempfile, "gettempdir", lambda: "/otpu-fake/tmpx")
    assert simtmp.fstype("/otpu-fake/tmpx", MOUNTS) == "ext4"
    assert simtmp.root(MOUNTS) is None
    assert simtmp.fstype("/x", "") == ""


def test_tempdir_goes_on_return_and_exception(monkeypatch, tmp_path):
    monkeypatch.setenv(simtmp.ENV, str(tmp_path / "sim"))
    with simtmp.tempdir("otpu_t_") as d:
        (d / "dram_0.bin").write_bytes(b"x" * 64)
        assert d.parent == tmp_path / "sim" and d.name.startswith("otpu_t_")
        assert (d / simtmp.MARKER).read_text().split() == [socket.gethostname(),
                                                            str(os.getpid())]
    assert not d.exists()
    with pytest.raises(RuntimeError):
        with simtmp.tempdir() as d:
            raise RuntimeError("the run failed")
    assert not d.exists()


def _holder(tmp_path, body):
    """A python process that makes a simulator directory under tmp_path, prints it, then runs
    body inside the block."""
    code = ("import sys, time\nfrom opentpu import simtmp, rtlsim\n"
            "with simtmp.tempdir() as d:\n    print(d, flush=True)\n    " + body + "\n")
    env = {**os.environ, simtmp.ENV: str(tmp_path), "PYTHONPATH": str(ROOT)}
    p = subprocess.Popen([sys.executable, "-c", code], env=env, stdout=subprocess.PIPE,
                         text=True)
    return p, Path(p.stdout.readline().strip())


def test_sigterm_removes_the_directory_and_the_simulator(tmp_path):
    p, d = _holder(tmp_path, 'rtlsim.run_sim(["sleep", "60"])')
    assert d.is_dir()
    time.sleep(0.5)
    t = time.time()
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=20) == -signal.SIGTERM     # the default action, after the cleanup
    assert time.time() - t < 15                     # the simulator was killed, not waited for
    assert not d.exists()


def test_sigterm_during_a_worker_threads_run(tmp_path):
    """The run in a worker thread (bench_llm's pool): its simulator killed and its directory
    removed by the main thread's handler, then the process ends at once."""
    code = ("import threading\nfrom opentpu import simtmp, rtlsim\n"
            "def run():\n    with simtmp.tempdir() as d:\n        print(d, flush=True)\n"
            "        rtlsim.run_sim(['sleep', '61.5'])\n"
            "t = threading.Thread(target=run)\nt.start()\nt.join()\n")
    env = {**os.environ, simtmp.ENV: str(tmp_path), "PYTHONPATH": str(ROOT)}
    p = subprocess.Popen([sys.executable, "-c", code], env=env, stdout=subprocess.PIPE,
                         text=True)
    d = Path(p.stdout.readline().strip())
    assert d.is_dir()
    time.sleep(0.5)
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=20) == -signal.SIGTERM
    assert not d.exists()
    time.sleep(0.5)
    left = subprocess.run(["pgrep", "-f", "sleep 61.5"], capture_output=True, text=True)
    assert left.stdout.strip() == ""



def test_a_forked_childs_sigterm_leaves_the_parents_run(tmp_path):
    """A forked child (a pool's worker) inherits the handler but not the parent's runs."""
    p, d = _holder(tmp_path, "import os, signal\n    c = os.fork()\n"
                   "    if c == 0:\n        time.sleep(60)\n    time.sleep(0.5)\n"
                   "    os.kill(c, signal.SIGTERM)\n"
                   "    print(os.waitpid(c, 0)[1] & 0x7f, d.exists(), flush=True)")
    assert p.stdout.readline().split() == [str(int(signal.SIGTERM)), "True"]
    assert p.wait(timeout=20) == 0
    assert not d.exists()


def test_sigkilled_runs_directory_goes_at_the_next_run(tmp_path, monkeypatch):
    p, d = _holder(tmp_path, "time.sleep(60)")
    p.kill()
    p.wait(timeout=20)
    assert d.is_dir()                               # SIGKILL: nothing ran
    keep = [tmp_path / "otpu_nomarker", tmp_path / "otpu_otherhost", tmp_path / "otpu_live"]
    for k in keep:
        k.mkdir()
    (keep[1] / simtmp.MARKER).write_text(f"not-{socket.gethostname()} {p.pid}\n")
    (keep[2] / simtmp.MARKER).write_text(f"{socket.gethostname()} {os.getppid()}\n")
    monkeypatch.setenv(simtmp.ENV, str(tmp_path))
    with simtmp.tempdir() as mine:
        assert not d.exists()
        assert all(k.is_dir() for k in keep)
    assert not mine.exists()
