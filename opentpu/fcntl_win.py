"""Windows compatibility shims for the Unix ``fcntl`` and ``resource`` modules.

openTPU targets Linux as its first-class platform. Two stdlib modules it
imports do not exist on Windows:

``fcntl``
    Used for ``flock`` on shared resources (the RTL build directory, the PCIe
    card, the /tmp/otpu lock files). Backed here by ``msvcrt.locking``.

``resource``
    Used for ``getrusage(RUSAGE_SELF).ru_maxrss`` (peak resident memory).
    Backed here by ``GetProcessMemoryInfo`` from ``psapi``.

Importing either one unguarded takes the whole test suite down at collection
time, so the affected modules import this shim as a fallback.

This module is Windows-only; on other platforms the real modules are used.
"""

import os
import sys

if sys.platform != "win32":  # pragma: no cover - the real modules are present
    raise ImportError("fcntl_win is a Windows-only shim")

import msvcrt  # noqa: E402

# --------------------------------------------------------------------------
# fcntl
# --------------------------------------------------------------------------

LOCK_SH = 1  # shared / read lock
LOCK_EX = 2  # exclusive / write lock
LOCK_NB = 4  # do not block
LOCK_UN = 8  # unlock

__all__ = [
    "LOCK_SH",
    "LOCK_EX",
    "LOCK_NB",
    "LOCK_UN",
    "flock",
    "lockf",
    "RUSAGE_SELF",
    "RUSAGE_CHILDREN",
    "getrusage",
    "struct_rusage",
]


def _as_fd(fd):
    """Accept either an int file descriptor or a file object.

    Python 3's ``fcntl.flock`` accepts both; the project calls it with a
    file object (``fcntl.flock(lock, fcntl.LOCK_EX)`` on an open file), so
    the shim has to as well.
    """
    if isinstance(fd, int):
        return fd
    fileno = getattr(fd, "fileno", None)
    if fileno is None:
        raise TypeError(
            f"flock() argument must be an int fd or a file object, not "
            f"{type(fd).__name__!r}"
        )
    return fileno()


def flock(fd, operation):
    """Take or release an advisory lock on the file behind ``fd``.

    Mirrors ``fcntl.flock``: ``LOCK_EX``/``LOCK_SH`` take a lock (with
    ``LOCK_NB`` failing immediately instead of blocking), ``LOCK_UN`` releases
    it. Raises ``OSError`` when a non-blocking lock cannot be granted.
    """
    fd = _as_fd(fd)

    if operation & LOCK_UN:
        _unlock(fd)
        return

    blocking = not (operation & LOCK_NB)

    # msvcrt.locking needs a byte range; lock the first byte of the file.
    pos = os.lseek(fd, 0, os.SEEK_CUR)
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
        msvcrt.locking(fd, mode, 1)
    finally:
        os.lseek(fd, pos, os.SEEK_SET)

    # A shared lock on Windows is emulated as an exclusive byte lock; the
    # project only takes LOCK_SH on read-only inspection paths, so the
    # stricter behaviour is acceptable and safe.


def _unlock(fd):
    pos = os.lseek(fd, 0, os.SEEK_CUR)
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    except OSError:
        # Not locked (or the file was closed): nothing to release.
        pass
    finally:
        os.lseek(fd, pos, os.SEEK_SET)


_F_LOCK = 1
_F_RLOCK = 2
_F_UNLCK = 8
_F_NB = 4


def lockf(fd, cmd, len_=0, start=0, whence=0):  # pragma: no cover - unused
    """POSIX record locking, emulated on top of ``flock``.

    The project does not use ``lockf``; this exists only so code that
    references it does not fail at import time.
    """
    if cmd & _F_UNLCK:
        return flock(fd, LOCK_UN)
    if cmd & _F_LOCK:
        return flock(fd, LOCK_EX | (LOCK_NB if cmd & _F_NB else 0))
    if cmd & _F_RLOCK:
        return flock(fd, LOCK_SH | (LOCK_NB if cmd & _F_NB else 0))
    return None


# --------------------------------------------------------------------------
# resource
# --------------------------------------------------------------------------

RUSAGE_SELF = 0
RUSAGE_CHILDREN = -1


class struct_rusage:
    """A stand-in for the POSIX ``struct rusage``.

    Only ``ru_maxrss`` is populated (peak resident set size, in kilobytes on
    both platforms). Other fields stay at zero.
    """

    __slots__ = (
        "ru_utime",
        "ru_stime",
        "ru_maxrss",
        "ru_ixrss",
        "ru_idrss",
        "ru_isrss",
        "ru_minflt",
        "ru_majflt",
        "ru_nswap",
        "ru_inblock",
        "ru_oublock",
        "ru_msgsnd",
        "ru_msgrcv",
        "ru_nsignals",
        "ru_nvcsw",
        "ru_nivcsw",
    )

    def __init__(self):
        self.ru_utime = 0.0
        self.ru_stime = 0.0
        self.ru_maxrss = 0
        self.ru_ixrss = 0
        self.ru_idrss = 0
        self.ru_isrss = 0
        self.ru_minflt = 0
        self.ru_majflt = 0
        self.ru_nswap = 0
        self.ru_inblock = 0
        self.ru_oublock = 0
        self.ru_msgsnd = 0
        self.ru_msgrcv = 0
        self.ru_nsignals = 0
        self.ru_nvcsw = 0
        self.ru_nivcsw = 0

    def __repr__(self):
        return f"struct_rusage(ru_maxrss={self.ru_maxrss})"


def getrusage(who=RUSAGE_SELF):
    """Return a ``struct_rusage`` for this process.

    ``ru_maxrss`` comes from ``GetProcessMemoryInfo``'s PeakWorkingSetSize,
    reported in bytes on Windows and converted to kilobytes to match POSIX.
    """
    ru = struct_rusage()
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)

        psapi = ctypes.WinDLL("psapi")
        kernel32 = ctypes.WinDLL("kernel32")

        # The argtypes matter: without them ctypes defaults every argument
        # to a 32-bit int, which truncates the pseudo-handle -1 and makes the
        # call fail silently.
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.GetCurrentProcess.argtypes = []
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]

        handle = kernel32.GetCurrentProcess()
        if psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        ):
            ru.ru_maxrss = counters.PeakWorkingSetSize // 1024
    except Exception:
        # Best effort: an unknown peak RSS is better than an import failure.
        pass
    return ru
