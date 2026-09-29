"""The calibration CPU's firmware (tools/litedram/selfcal_fw) built for this machine and run on
ddrcal's simulated PHY (FakeBoard), next to ddrcal itself on an identical one: every CSR write
either makes goes into a trace. Used by tests/test_selfcal.py and tools/litedram/selfcal_sim.py.
"""
import ctypes
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "litedram" / "selfcal_fw"))
import fw                                           # noqa: E402

from opentpu.host import ddrcal as C                # noqa: E402

RD = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_uint32)
WR = ctypes.CFUNCTYPE(None, ctypes.c_uint32, ctypes.c_uint32)


class Trace:
    """A CSR space (FakeBoard) whose writes are logged as (name, value)."""
    def __init__(self, board):
        self.b, self.regs, self.log = board, board.regs, []

    def w(self, name, v):
        self.log.append((name, v))
        self.b.w(name, v)

    def r(self, name):
        return self.b.r(name)


def reference(build, board, stride, mask=3, sleep=False):
    """ddrcal.calibrate_channel on each channel in `mask`, as memcal.ensure runs them: (trace,
    {ch: result or the exception})."""
    t = Trace(board)
    res = {}
    real = time.sleep
    if not sleep:
        time.sleep = lambda s: None
    try:
        for ch in (0, 1):
            if mask >> ch & 1 and C.Chan(t, ch).name("ddrphy_dly_sel") in t.regs:
                try:
                    res[ch] = C.calibrate_channel(C.Chan(t, ch), Path(build) / "sdram_init.py",
                                                  stride=stride, log=lambda *_: None)
                except (C.CalError, TimeoutError) as e:
                    res[ch] = e
    finally:
        time.sleep = real
    return t.log, res


def first_diff(a, b):
    """Where two write traces part."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"write {i}: ddrcal {x}, firmware {y}; before: {a[max(0, i - 6):i]}"
    return f"lengths {len(a)} (ddrcal) / {len(b)} (firmware)"


class Firmware:
    """The firmware (SELFCAL_HOST) for one build's CSR map, as a ctypes library."""
    def __init__(self, build, out):
        self.build = Path(build)
        self.lib = ctypes.CDLL(str(fw.host(build, out)))
        self.lib.selfcal_run.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
        self.lib.selfcal_host_now.restype = ctypes.c_uint64
        regs, _ = fw.config(build)
        self.words = {a + 4 * i: (name, i, n) for name, (a, n) in regs.items() for i in range(n)}

    def run(self, csr, mask=3, stride=1):
        """selfcal_run on `csr` (ddrcal's w / r on names), multi-word CSRs assembled as the CSR
        bus does (most significant word first, the last word's write takes effect). Returns the
        mailbox, the selfcal_* writes and the simulated sys cycles."""
        mbox, own, pending, errors = [0] * 256, [], {}, []

        def rd(a):
            try:
                name, i, n = self.words[a]
                if name.startswith("selfcal_"):
                    return {"selfcal_config": mask | stride << 8}.get(name, 0)
                return (csr.r(name) >> 32 * (n - 1 - i)) & 0xFFFFFFFF
            except Exception as e:           # noqa: BLE001 (raised again after the run)
                errors.append(e)
                return 0

        def wr(a, v):
            try:
                name, i, n = self.words[a]
                acc = (pending.pop(name, 0) << 32) | v
                if i < n - 1:
                    pending[name] = acc
                elif name.startswith("selfcal_"):
                    own.append((name, acc))
                else:
                    csr.w(name, acc)
            except Exception as e:           # noqa: BLE001
                errors.append(e)

        def mb(i, v):
            mbox[i] = v

        cbs = (RD(rd), WR(wr), WR(mb))      # kept alive for the call
        self.lib.selfcal_host_setup(*cbs)
        self.lib.selfcal_run(mask, stride)
        if errors:
            raise errors[0]
        return mbox, own, self.lib.selfcal_host_now()
