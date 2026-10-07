"""Host-side device state: the runner's exclusive lock and its status file.

    /tmp/otpu/<device>.lock   flock(LOCK_EX) by the process that runs programs or writes DRAM
                              (Board, BoardBackend, otpu-selftest, otpu-chat, otpu-lens record);
                              it holds the owner's pid (read to name it in the error)
    /tmp/otpu/<device>.json   the runner's status (RunnerStatus), rewritten atomically after
                              every token (or at most every min_interval seconds) and removed
                              at exit; otpu-smi reads it

<device> is the device node's basename (xdma0 for /dev/xdma0). OTPU_RUN_DIR moves the
directory (tests). Monitors never lock: they only read registers.

The lock is flock(2): it belongs to the open file description, so it is released when the
process dies however it dies, and a second open() of the file -- in another process or in the
same one -- does not get it. A status file left by a killed runner (SIGKILL skips the atexit
hook) is recognized as stale: its pid is gone.

The directory is shared by the host's users (mode 1777, like /tmp), and whoever creates it
first owns it. So its files are opened without following a symbolic link (O_NOFOLLOW) and must
be plain files of one link (open_shared), and replaced only by renaming a new file of this
process's over them (write_shared): neither the directory's owner nor an entry someone planted
makes another user's tool truncate or write a file elsewhere. Lock files are made mode 0666
(umask aside), and one another user left unwritable is locked read-only, so a second user
waits for the card (DeviceBusy, OTPU_LOCK_WAIT) instead of failing with PermissionError.

While a process holds a card, SIGTERM and SIGHUP raise SystemExit in it (where their handler
was the default), so that its cleanup runs: Board stops the card's run before the lock goes
(board.py). `otpu-lock -- CMD` (hold_main) passes the signals it gets on to CMD and exits after
it, never before; CMD's openTPU tools check that it still runs (OTPU_LOCK_HELD names it).
"""
from __future__ import annotations

import atexit
import errno
import functools
import fcntl
import json
import os
import signal
import stat
import sys
import tempfile
import threading
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
        super().__init__(f"{name} is in use by {who}: stop it, or wait for it with "
                         f"OTPU_LOCK_WAIT=<seconds> (lock {run_dir() / (name + '.lock')})")


class LockLost(DeviceBusy):
    """The otpu-lock this process runs under (OTPU_LOCK_HELD) has exited, and the card lock
    with it: the card may be another runner's now."""

    def __init__(self, name: str, pid: int):
        self.pid = pid
        RuntimeError.__init__(self, f"{name}: the otpu-lock this process runs under (pid {pid}) "
                                    "has exited, and with it the card lock: stopped before "
                                    "touching the card again")


class UnsafePath(RuntimeError):
    """An entry of the shared run directory is not what the tools make there (a symbolic link,
    a hard link, a directory)."""


def busy_exits(main):
    """A command's main(): a busy card ends it with one line on stderr and exit status 3
    instead of a traceback."""
    @functools.wraps(main)
    def run(*args, **kw):
        try:
            return main(*args, **kw)
        except DeviceBusy as e:
            print(f"{Path(sys.argv[0]).name}: {e}", file=sys.stderr)
            return 3
    return run


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


def _stat_fields(pid: int) -> list[bytes] | None:
    """/proc/<pid>/stat after the command's parentheses: [state, ppid, ...] (None off Linux,
    or for a process that is gone)."""
    try:
        s = Path(f"/proc/{pid}/stat").read_bytes()
    except OSError:
        return None
    return s[s.rindex(b")") + 2:].split()


def _running(pid: int) -> bool:
    """pid is a live process (not a zombie its parent has not reaped yet: a killed otpu-lock is
    one for a moment, and its lock is free already)."""
    if not pid_alive(pid):
        return False
    f = _stat_fields(pid)
    return f is None or f[0] != b"Z"


def _ppid(pid: int) -> int:
    """pid's parent (0 when unknown)."""
    f = _stat_fields(pid)
    if f is not None:
        return int(f[1])
    import subprocess                       # (no /proc: macOS)
    try:
        out = subprocess.run(["ps", "-o", "ppid=", "-p", str(pid)], capture_output=True,
                             text=True, timeout=10).stdout
        return int(out.strip() or 0)
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0


def _is_ancestor(pid: int) -> bool:
    p, seen = os.getppid(), 0
    while p > 1 and seen < 64:
        if p == pid:
            return True
        p, seen = _ppid(p), seen + 1
    return False


# ------------------------------------------------------------------------------ shared files
def ensure_run_dir() -> Path:
    """run_dir(), made if missing (mode 1777: shared by users, sticky, like /tmp). A symbolic
    link there is followed only when it is this user's or root's (anyone may make /tmp/otpu)."""
    d = run_dir()
    try:
        d.parent.mkdir(parents=True, exist_ok=True)
        os.mkdir(d)
    except FileExistsError:
        pass
    st = os.lstat(d)
    if stat.S_ISLNK(st.st_mode):
        if st.st_uid not in (os.getuid(), 0):
            raise UnsafePath(f"{d} is a symbolic link of user {st.st_uid}: remove it")
        st = os.stat(d)
    if not stat.S_ISDIR(st.st_mode):
        raise UnsafePath(f"{d} is not a directory: remove it")
    if st.st_uid == os.getuid() and stat.S_IMODE(st.st_mode) != 0o1777:
        try:
            os.chmod(d, 0o1777)
        except OSError:
            pass
    return d


def open_shared(path: Path, mode: int = 0o666) -> int:
    """A lock file of the run directory, made with `mode` (umask aside) if missing: read-write,
    or read-only where it is another user's and not writable for this one (flock works on
    either). Never through a symbolic link; it must be a plain file of one link."""
    path = Path(path)
    nf = os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        try:
            fd = os.open(path, os.O_RDWR | os.O_CREAT | nf, mode)
        except PermissionError:
            fd = os.open(path, os.O_RDONLY | nf)
    except OSError as e:
        if e.errno == errno.ELOOP:
            raise UnsafePath(f"{path} is a symbolic link: remove it") from None
        raise
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
            raise UnsafePath(f"{path} is not a plain file of one link: remove it")
        if st.st_uid == os.getuid() and stat.S_IMODE(st.st_mode) != mode:
            try:
                os.fchmod(fd, mode)             # (the umask took the others' write bits)
            except OSError:
                pass
    except BaseException:
        os.close(fd)
        raise
    return fd


def read_shared(path: Path, limit: int = 1 << 20) -> str:
    """A file of the run directory (a status or cache file): never through a symbolic link, a
    plain file, at most `limit` bytes read. OSError where it is not one."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(errno.EINVAL, "not a plain file", str(path))
        return os.read(fd, limit).decode(errors="replace")
    finally:
        os.close(fd)


def write_shared(path: Path, text: str) -> None:
    """Replace the run directory's file `path` with `text` atomically (readers see the old file
    or the new, never half): a new file of this process's (a fresh name, O_EXCL: never an
    existing entry or a link), mode 0644, renamed over it. OSError where it cannot (another
    user's file in the sticky directory)."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            os.fchmod(f.fileno(), 0o644)
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ------------------------------------------------------------------------------ signals
SIGNALS = ("SIGTERM", "SIGHUP")
REPEAT_GRACE = 2.0      # seconds: the same signal again within this of the first is the same
                        # request (a group kill reaches a process under otpu-lock twice: from
                        # the group and from otpu-lock), ignored; later, the default action
_sig_users = 0          # DeviceLocks of this process holding the handlers
_sig_saved: dict = {}
_sig_first: float | None = None


def _exit_on_signal(signum, frame):
    global _sig_first
    now = time.monotonic()
    if _sig_first is None:
        _sig_first = now
        raise SystemExit(128 + signum)
    if now - _sig_first < REPEAT_GRACE:
        return
    signal.signal(signum, signal.SIG_DFL)   # asked again: end at once
    os.kill(os.getpid(), signum)


def _hold_signals() -> bool:
    """While this process holds a card: SIGTERM and SIGHUP raise SystemExit, so that the
    cleanup runs (the finally blocks; Board stopping its run). Only where the handler is the
    default, or simtmp's (the unwinding kills the simulators and removes their directories
    too), and from the main thread (signal.signal's rule); True when this lock counts."""
    global _sig_users
    if threading.current_thread() is not threading.main_thread():
        return False
    if _sig_users == 0:
        for name in SIGNALS:
            sig = getattr(signal, name, None)
            h = signal.getsignal(sig) if sig is not None else None
            if h is signal.SIG_DFL or getattr(h, "__module__", None) == "opentpu.simtmp":
                _sig_saved[sig] = h
                signal.signal(sig, _exit_on_signal)
    _sig_users += 1
    return True


def _release_signals() -> None:
    global _sig_users, _sig_first
    _sig_users -= 1
    if _sig_users or threading.current_thread() is not threading.main_thread():
        return
    for sig, h in _sig_saved.items():
        if signal.getsignal(sig) is _exit_on_signal:
            signal.signal(sig, h)
    _sig_saved.clear()
    _sig_first = None


# ------------------------------------------------------------------------------ the lock
def _held_by(name: str, value: str) -> int | None:
    """OTPU_LOCK_HELD's holder ("<device>:<pid>", hold_main's) when it really holds the lock:
    it is an ancestor of this process (so it runs), the lock file names it, and the flock is
    held (a shared probe fails; it only runs where the holder is alive and should hold it)."""
    try:
        pid = int(value.partition(":")[2])
    except ValueError:
        return None
    if not _is_ancestor(pid):
        return None
    try:
        fd = os.open(run_dir() / f"{name}.lock", os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError:
        return None
    try:
        try:
            if int(os.pread(fd, 32, 0).split()[0]) != pid:
                return None
        except (ValueError, IndexError, OSError):
            return None
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError as e:
            return pid if e.errno in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES) else None
        return None                         # free (closing the probe's fd drops it)
    finally:
        os.close(fd)


class DeviceLock:
    """Exclusive lock on a device, held until release() (or process exit). Monitors must not
    probe it with flock (even LOCK_SH for an instant would make a starting runner fail): they
    read the status file instead.

    Inside `otpu-lock -- CMD` (OTPU_LOCK_HELD = "<device>:<otpu-lock's pid>") the lock is
    otpu-lock's: `holder` is its pid, checked to hold the lock here and to still run by
    lost() (Board, before it writes the card or starts a run)."""

    fd: int | None = None
    holder: int | None = None
    _sig = False

    def __init__(self, name: str, wait: float | None = None):
        """wait: seconds to wait for a busy device before DeviceBusy (default: the environment's
        OTPU_LOCK_WAIT, else 0). Waiting polls the lock, so it is fair only in the sense that
        whoever tries when it is free gets it."""
        self.name = name
        # called by release() before the lock goes (at exit too), weakly: Board's stop of a run
        # in flight (weakref.WeakMethod; a callable that returns None)
        self.cleanup: list = []
        held = os.environ.get("OTPU_LOCK_HELD", "")
        if held.partition(":")[0] == name:      # inside `otpu-lock -- CMD`: it holds it
            pid = _held_by(name, held)
            if pid is not None:
                self.holder = pid
                self._sig = _hold_signals()
                atexit.register(self.release)
                return
            print(f"otpu: OTPU_LOCK_HELD={held} does not hold {name} (its otpu-lock is gone, "
                  "or not this process's ancestor): taking the lock", file=sys.stderr,
                  flush=True)
        if wait is None:
            wait = float(os.environ.get("OTPU_LOCK_WAIT", "0") or 0)
        deadline = time.monotonic() + wait
        while True:
            try:
                self._take(name)
                return
            except DeviceBusy as e:
                if time.monotonic() >= deadline:
                    raise
                if not getattr(self, "_told", False):
                    print(f"otpu: {e}; waiting up to {wait:.0f}s (OTPU_LOCK_WAIT)",
                          file=sys.stderr, flush=True)
                    self._told = True
                time.sleep(1.0)

    def _take(self, name: str) -> None:
        self.path = ensure_run_dir() / f"{name}.lock"
        fd = open_shared(self.path)
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
        try:                                # (a lock file locked read-only keeps the old pid)
            os.ftruncate(fd, 0)
            os.pwrite(fd, f"{os.getpid()}\n".encode(), 0)
        except OSError:
            pass
        self.fd = fd
        self._sig = _hold_signals()
        atexit.register(self.release)

    def lost(self) -> bool:
        """The otpu-lock this lock is inherited from has exited (the lock is free, or another
        runner's): this process must not touch the card any more."""
        return self.holder is not None and not _running(self.holder)

    def release(self) -> None:
        cleanup, self.cleanup = getattr(self, "cleanup", []), []
        for ref in cleanup:
            f = ref()
            if f is not None:
                f()
        if self.fd is not None:
            try:
                os.ftruncate(self.fd, 0)
            except OSError:
                pass
            os.close(self.fd)            # drops the flock
            self.fd = None
        if self._sig:
            self._sig = False
            _release_signals()
        atexit.unregister(self.release)


class RunnerStatus:
    """The status file of the process that holds a device. Fields (all optional but pid):

    pid, argv, start (unix time), dev, model, weights (otpu-chat: its formats in words), core_khz,
    dram: {total, image, weights, kv_capacity, kv_used, program, free} (bytes),
    tokens (device runs so far), last_cycles, tok_s_device (CORE_KHZ / last_cycles),
    tok_s_wall (over the last WALL_WINDOW runs, host work included), updated (unix time).

    min_interval: the file is rewritten after a token at most this often (seconds), by a
    writer thread (TOKEN_DEFER after the token when the interval is already up), so the file is never behind for
    longer and the rewrite (0.2-0.5 ms on the card's host) is never on the token's critical
    path; 0: token() rewrites it itself. A file that cannot be written (another user's, left
    in the shared directory) is reported once on stderr; the runner goes on without it.
    """

    WALL_WINDOW = 8
    TOKEN_DEFER = 2e-3                      # seconds between a token and its rewrite (at least)

    def __init__(self, name: str, min_interval: float = 0.0, **fields):
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._due = threading.Event()       # a token's rewrite is due (the writer thread's)
        self._writer: threading.Thread | None = None
        self._closed = False
        self._failed = False                # a write failed (told once)
        self._tok_written = -1e9            # perf_counter of the last write after a token
        self.path = ensure_run_dir() / f"{name}.json"
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
        if self.min_interval <= 0:
            self.write()
            return
        # the rewrite is the writer thread's, never the caller's: token() only sets an event
        # (starting a threading.Timer per rewrite cost the caller 0.1-0.2 ms on the card's host)
        if self._writer is None:
            self._writer = threading.Thread(target=self._write_loop, daemon=True)
            self._writer.start()
        self._due.set()

    def _write_loop(self) -> None:
        while True:
            self._due.wait()
            if self._closed:
                return
            # at least TOKEN_DEFER after the token: the caller is then back on the device's
            # run, not between its HALTED and the next RUN (a rewrite there held the GIL for
            # 0.2-0.6 ms of some LFM2 tokens' critical path on the card)
            wait = self.min_interval - (time.perf_counter() - self._tok_written)
            time.sleep(max(wait, self.TOKEN_DEFER))
            if self._closed:
                return
            self._due.clear()
            self._tok_written = time.perf_counter()
            self.write()

    def write(self) -> None:
        with self._lock:
            self.data["updated"] = time.time()
            try:
                write_shared(self.path, json.dumps(self.data))
            except OSError as e:
                if not self._failed:
                    print(f"otpu: no status file ({e}); otpu-smi will not show this run",
                          file=sys.stderr, flush=True)
                    self._failed = True

    def remove(self) -> None:
        self._closed = True
        self._due.set()                     # the writer thread ends
        try:
            cur = json.loads(read_shared(self.path))
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
        d = json.loads(read_shared(p))
    except (OSError, ValueError):
        return None
    d["stale"] = not pid_alive(int(d.get("pid", 0) or 0))
    return d


# ------------------------------------------------------------------------------ otpu-lock
FORWARD = ("SIGTERM", "SIGHUP", "SIGINT")


def _foreground() -> bool:
    """otpu-lock is its terminal's foreground job (an interactive CMD: the terminal sends it its
    own signals)."""
    try:
        fd = os.open("/dev/tty", os.O_RDONLY | os.O_NOCTTY)
    except OSError:
        return False
    try:
        return os.tcgetpgrp(fd) == os.getpgrp()
    except OSError:
        return False
    finally:
        os.close(fd)


def _pdeathsig():
    """Linux: a function that makes the calling process get SIGTERM when its parent dies
    (prctl PR_SET_PDEATHSIG), else None."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        import ctypes
        prctl = ctypes.CDLL(None, use_errno=True).prctl
    except (OSError, AttributeError):
        return None
    return lambda: prctl(1, int(signal.SIGTERM))


def _run(cmd: list, env: dict) -> int:
    """CMD under the lock; otpu-lock exits after it, never before, so the lock goes only once
    CMD has ended. The signals otpu-lock gets go on to CMD: to its process group, a group of
    its own unless otpu-lock is its terminal's foreground job (so a script's tools get them
    too); in the foreground CMD stays in the terminal's group, which sends it the terminal's
    signals itself, and a SIGTERM goes to CMD. A signal ignored where otpu-lock started stays
    ignored (nohup). On Linux CMD gets SIGTERM if otpu-lock dies (SIGKILL; PR_SET_PDEATHSIG),
    and the openTPU tools under a dead otpu-lock stop before touching the card (Board's
    LockLost)."""
    import subprocess
    fg = _foreground()
    child: list = []
    pending: list = []

    def forward(signum, frame):
        if fg and signum != signal.SIGTERM:
            return                          # the terminal sent it to CMD as well
        if not child:
            pending.append(signum)
            return
        try:
            (os.kill if fg else os.killpg)(child[0], signum)
        except OSError:
            pass

    old = {}
    for name in FORWARD:
        sig = getattr(signal, name, None)
        if sig is not None and signal.getsignal(sig) is not signal.SIG_IGN:
            old[sig] = signal.signal(sig, forward)
    try:
        death, parent = _pdeathsig(), os.getpid()
        kw: dict = {}
        if death is not None:
            def pre():
                if not fg:
                    os.setpgid(0, 0)
                death()
                if os.getppid() != parent:  # (otpu-lock died before that)
                    os.kill(os.getpid(), signal.SIGTERM)
            kw["preexec_fn"] = pre
        elif not fg:
            kw["process_group"] = 0
        p = subprocess.Popen(cmd, env=env, **kw)
        child.append(p.pid)
        for s in pending:
            forward(s, None)
        return p.wait()
    finally:
        for sig, h in old.items():
            signal.signal(sig, h)


def hold_main(argv=None) -> int:
    """otpu-lock [--dev /dev/xdma0] [--wait SEC] [--prebuild MODEL:WF:HF ...] -- CMD...: run CMD
    while holding the device lock (for steps that are not openTPU tools but must not overlap a
    run: a JTAG reload, a driver reload, a rescan), or a sequence of runs that must not be
    interleaved with others (a reload, then tests on the new image). openTPU tools inside CMD
    run under this lock (OTPU_LOCK_HELD = <device>:<pid>). Signals to otpu-lock go on to CMD,
    and otpu-lock ends after it (_run). --prebuild first quantizes those runs' 4-bit weights
    into the image cache at nice 19, before waiting for the lock (opentpu.host.prebuild, from
    the tree on PYTHONPATH), so that CMD's tools do not quantize under it."""
    import argparse
    import subprocess
    ap = argparse.ArgumentParser(prog="otpu-lock", description=hold_main.__doc__.split("\n")[0])
    ap.add_argument("--dev", default="/dev/xdma0")
    ap.add_argument("--wait", type=float, default=3600.0, help="seconds to wait for the lock")
    ap.add_argument("--prebuild", action="append", default=[], metavar="MODEL:WF:HF",
                    help="quantize into the image cache first, outside the lock")
    ap.add_argument("--prebuild-cfg", help="the card configuration for --prebuild")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("no command")
    if a.prebuild:
        rc = subprocess.call(["nice", "-n", "19", sys.executable, "-m", "opentpu.host.prebuild"]
                             + (["--cfg", a.prebuild_cfg] if a.prebuild_cfg else [])
                             + a.prebuild)
        if rc:
            print(f"otpu-lock: prebuild exit {rc}; the tools quantize under the lock",
                  file=sys.stderr, flush=True)
    name = devname(a.dev)
    lock = DeviceLock(name, wait=a.wait)
    env = dict(os.environ)
    if lock.fd is not None:         # (inside another otpu-lock: its holder stays named)
        env["OTPU_LOCK_HELD"] = f"{name}:{os.getpid()}"
    try:           # openTPU tools inside CMD run under this lock instead of waiting for it
        return _run(cmd, env)
    finally:
        lock.release()


if __name__ == "__main__":         # python -m opentpu.host.runstate: otpu-lock
    raise SystemExit(hold_main())
