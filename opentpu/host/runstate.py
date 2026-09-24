"""Host-side device state: the runner's exclusive lock and its status file.

    /tmp/otpu/<device>.lock   flock(LOCK_EX) by the process that runs programs or writes DRAM
                              (Board, BoardBackend, otpu-selftest, otpu-chat, otpu-lens record);
                              it holds the owner's pid (read to name it in the error)
    /tmp/otpu/<device>.json   the runner's status (RunnerStatus), rewritten atomically after
                              every token and removed at exit; otpu-smi reads it

<device> is the device node's basename (xdma0 for /dev/xdma0). OTPU_RUN_DIR moves the
directory (tests). Monitors never lock: they only read registers.

The lock is flock(2): it belongs to the open file description, so it is released when the
process dies however it dies, and a second open() of the file -- in another process or in the
same one -- does not get it. A status file left by a killed runner (SIGKILL skips the atexit
hook) is recognized as stale: its pid is gone.
"""
from __future__ import annotations

import atexit
import errno
import fcntl
import json
import os
import sys
import time
from pathlib import Path


def run_dir() -> Path:
    return Path(os.environ.get("OTPU_RUN_DIR", "/tmp/otpu"))


def devname(dev: str) -> str:
    return Path(dev).name


class DeviceBusy(RuntimeError):
    def __init__(self, name: str, pid: int | None, cmd: str = ""):
        self.pid = pid
        who = f"process {pid}" + (f" ({cmd})" if cmd else "") if pid else "another process"
        super().__init__(f"{name} is in use by {who}: stop it, or wait "
                         f"(lock {run_dir() / (name + '.lock')})")


def _cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode().strip()
    except OSError:
        return ""


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError as e:
        return e.errno == errno.EPERM
    return True


class DeviceLock:
    """Exclusive lock on a device, held until release() (or process exit). Monitors must not
    probe it with flock (even LOCK_SH for an instant would make a starting runner fail): they
    read the status file instead."""

    fd: int | None = None

    def __init__(self, name: str):
        self.name = name
        d = run_dir()
        d.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(d, 0o1777)          # shared by users (sticky, like /tmp); best effort
        except OSError:
            pass
        self.path = d / f"{name}.lock"
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o666)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            if e.errno not in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
                os.close(fd)
                raise
            try:
                pid = int(os.pread(fd, 32, 0).split()[0])
            except (ValueError, IndexError, OSError):
                pid = None
            os.close(fd)
            raise DeviceBusy(name, pid, _cmdline(pid) if pid else "") from None
        os.ftruncate(fd, 0)
        os.pwrite(fd, f"{os.getpid()}\n".encode(), 0)
        self.fd = fd
        atexit.register(self.release)

    def release(self) -> None:
        if self.fd is not None:
            try:
                os.ftruncate(self.fd, 0)
            except OSError:
                pass
            os.close(self.fd)            # drops the flock
            self.fd = None
        atexit.unregister(self.release)


class RunnerStatus:
    """The status file of the process that holds a device. Fields (all optional but pid):

    pid, argv, start (unix time), dev, model, core_khz,
    dram: {total, image, weights, kv_capacity, kv_used, program, free} (bytes),
    tokens (device runs so far), last_cycles, tok_s_device (CORE_KHZ / last_cycles),
    tok_s_wall (over the last WALL_WINDOW runs, host work included), updated (unix time).
    """

    WALL_WINDOW = 8

    def __init__(self, name: str, **fields):
        self.path = run_dir() / f"{name}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = {"pid": os.getpid(), "argv": list(sys.argv), "start": time.time(),
                     "dev": name, "model": None, "tokens": 0, "last_cycles": None,
                     "tok_s_device": None, "tok_s_wall": None, "dram": None}
        self.data.update(fields)
        self._ends: list[float] = []
        self.write()
        atexit.register(self.remove)

    def update(self, **fields) -> None:
        self.data.update(fields)
        self.write()

    def token(self, cycles: int | None, core_khz: int | None, **fields) -> None:
        """One device run finished."""
        now = time.time()
        self._ends = (self._ends + [now])[-(self.WALL_WINDOW + 1):]
        d = self.data
        d["tokens"] += 1
        d["last_cycles"] = cycles
        d["tok_s_device"] = core_khz * 1e3 / cycles if cycles and core_khz else None
        if len(self._ends) > 1:
            d["tok_s_wall"] = (len(self._ends) - 1) / max(self._ends[-1] - self._ends[0], 1e-9)
        d.update(fields)
        self.write()

    def write(self) -> None:
        self.data["updated"] = time.time()
        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(self.data))
        os.replace(tmp, self.path)          # readers see the old file or the new, never half

    def remove(self) -> None:
        try:
            cur = json.loads(self.path.read_text())
            if cur.get("pid") == os.getpid():
                self.path.unlink()
        except (OSError, ValueError):
            pass
        atexit.unregister(self.remove)


def read_status(name: str) -> dict | None:
    """The runner's status for a device; None when there is none. A file whose pid is gone is
    returned with "stale": True (a runner killed with SIGKILL leaves it behind)."""
    p = run_dir() / f"{name}.json"
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    d["stale"] = not pid_alive(int(d.get("pid", 0) or 0))
    return d
