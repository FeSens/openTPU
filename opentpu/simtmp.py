"""Temporary directories of the simulators: rtlsim's runs (otpu_*), the board model's
(otpu_board_*) and the tools' (otpu_bench_, otpu_prefill_, otpu_ddr_). A run's DRAM images are
up to gigabytes, so:

- **Where.** OTPU_SIM_TMP names the parent directory; else, where the system's temporary
  directory is a tmpfs or ramfs (Linux: /proc/mounts), DISK (its files would hold RAM:
  omarchy's /tmp is a 16 GB tmpfs); else the system's.
- **Removed on any exit.** The directory goes when its `tempdir()` block ends, by return or
  exception (SIGINT is Python's KeyboardInterrupt). SIGTERM and SIGHUP, where the program has
  no handler of its own: the handler (installed at import, from the main thread) kills the
  simulators run_sim started (track) and removes the live directories, then the signal's
  default action ends the process as before (any thread's runs; nothing else runs first, as
  without the handler). A process killed outright (SIGKILL, the OOM killer) leaves its
  directory: each new one sweeps its parent of those whose marker (MARKER: host and pid) names
  a process of this host that is gone.
"""
from __future__ import annotations

import contextlib
import os
import shutil
import signal
import socket
import tempfile
import threading
from pathlib import Path

ENV = "OTPU_SIM_TMP"
DISK = Path.home() / "otpu-build" / "tmp"
MARKER = ".otpu_owner"              # "host pid" of the process that made the directory
RAM_FS = ("tmpfs", "ramfs")
SIGNALS = ("SIGTERM", "SIGHUP")


def fstype(path, mounts: str | None = None) -> str:
    """The type of the filesystem holding `path`, from /proc/mounts (or the text `mounts`);
    "" where there is none (not Linux)."""
    if mounts is None:
        try:
            mounts = Path("/proc/mounts").read_text()
        except OSError:
            return ""
    real, best, kind = os.path.realpath(path), None, ""
    for line in mounts.splitlines():
        f = line.split()
        if len(f) < 3:
            continue
        mp = f[1].replace("\\040", " ")
        inside = real == mp or real.startswith(mp.rstrip("/") + "/")
        if inside and (best is None or len(mp) > len(best)):
            best, kind = mp, f[2]
    return kind


def root(mounts: str | None = None) -> Path | None:
    """The parent of new simulator directories: OTPU_SIM_TMP, else DISK where the system's
    temporary directory is in RAM, else None (tempfile's default)."""
    env = os.environ.get(ENV)
    if env:
        return Path(env).expanduser()
    if fstype(tempfile.gettempdir(), mounts) in RAM_FS:
        return DISK
    return None


_live: set = set()                  # the directories of tempdir() blocks not ended
_groups: set = set()                # the process groups of run_sim's simulators running


def track(pgid: int) -> None:
    """A simulator's process group, killed on SIGTERM / SIGHUP (rtlsim.run_sim)."""
    _groups.add(pgid)


def untrack(pgid: int) -> None:
    _groups.discard(pgid)


def _on_signal(signum, frame):
    """Kill the simulators, remove the live directories, then the signal's default action."""
    for g in tuple(_groups):
        try:
            os.killpg(g, signal.SIGKILL)
        except OSError:
            pass
    for d in tuple(_live):
        shutil.rmtree(d, ignore_errors=True)
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


_installed = False


def install() -> None:
    """The SIGTERM / SIGHUP handler, where theirs is the default; from the main thread only
    (signal.signal's rule: elsewhere a kill leaves the directories to the sweep)."""
    global _installed
    if _installed or threading.current_thread() is not threading.main_thread():
        return
    _installed = True
    for name in SIGNALS:
        sig = getattr(signal, name, None)
        if sig is not None and signal.getsignal(sig) == signal.SIG_DFL:
            signal.signal(sig, _on_signal)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def sweep(parent: Path) -> list:
    """Remove parent's simulator directories (otpu_*) whose marker names a process of this host
    that is gone; returns them. Directories without a marker, of another host or another user
    are left alone."""
    host, uid, gone = socket.gethostname(), os.getuid(), []
    for d in Path(parent).glob("otpu_*"):
        try:
            if not d.is_dir() or d.stat().st_uid != uid:
                continue
            h, pid = (d / MARKER).read_text().split()
            pid = int(pid)
        except (OSError, ValueError):
            continue
        if h != host or pid == os.getpid() or _alive(pid):
            continue
        shutil.rmtree(d, ignore_errors=True)
        gone.append(d)
    return gone


@contextlib.contextmanager
def tempdir(prefix: str = "otpu_"):
    """A new simulator directory (a Path) under root(), removed when the block ends; see the
    module for kills."""
    base = root()
    if base is not None:
        base.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix=prefix, dir=base))
    _live.add(str(d))
    try:
        (d / MARKER).write_text(f"{socket.gethostname()} {os.getpid()}\n")
        install()
        sweep(d.parent)
        yield d
    finally:
        _live.discard(str(d))
        shutil.rmtree(d, ignore_errors=True)


install()
