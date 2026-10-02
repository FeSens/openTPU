import os
import time

import numpy as np
import pytest

from opentpu import rtlsim
from opentpu.host import regs as R
from opentpu.host.fake import FakeTransport

# the tests quantize afresh, not through a host's image cache (opentpu/qcache.py), unless one
# is set (test_qcache.py sets its own)
os.environ.setdefault("OTPU_IMAGE_CACHE", "0")


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true",
                     help="also run the tests marked slow (full-size models on the RTL)")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long; runs only with --runslow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="slow: needs --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


def rel(a, b):
    return float(np.linalg.norm(np.asarray(a, np.float64) - b) / np.linalg.norm(b))


def assert_same_state(ri, rr):
    """RTL and ISA simulator must agree bit for bit on every slice's DRAM and TMEM."""
    for s, (a, b) in enumerate(zip(ri.drams, rr.drams)):
        bad = np.nonzero(a != b)[0]
        assert len(bad) == 0, f"slice {s}: {len(bad)} DRAM bytes differ, first at {bad[:8]}"
    for s, (a, b) in enumerate(zip(ri.tmems, rr.tmems)):
        bad = np.nonzero(a != b)[0]
        assert len(bad) == 0, f"slice {s}: {len(bad)} TMEM words differ, first at {bad[:8]}"



def assert_fill_is_transparent(make, toks):
    """Step programs compiled with fill (qwen3.fill_logits, the card's streamed decode) against
    the plain ones on the ISA simulator: the logits and every slice's DRAM after each step,
    bit for bit. make() builds an Engine. The filling engine's TMEM is random before each step
    (the simulator keeps TMEM between runs, as the card does): no step program reads what an
    earlier run left there, so the fill's tile can take no state the steps keep across tokens."""
    a, b = make(), make()
    a.image.stream_fill = True
    rng = np.random.default_rng(11)
    for t in toks:
        for s in a.backend.machine.slices:
            s.tmem[:] = rng.integers(0, 1 << 32, s.tmem.shape, np.uint64).astype(s.tmem.dtype)
        assert np.array_equal(a.step(t).view(np.uint32), b.step(t).view(np.uint32)), a.pos
    for sa, sb in zip(a.backend.machine.slices, b.backend.machine.slices):
        assert np.array_equal(sa.dram, sb.dram)

@pytest.fixture(scope="session")
def have_verilator():
    import shutil
    if shutil.which("verilator") is None:
        pytest.skip("verilator not installed")
    return True


class IsaCard(FakeTransport):
    """A fake card that computes: RUN runs the loaded program on the ISA simulator over the
    channel memories (with args, CAPS bit25: and ARG0..7). What the run writes shows at once,
    except [late_addr, +late_n) (the logits): piece i of `piece` bytes shows at run_s * (0.4 +
    0.5 * i / pieces), its first half of beats a little before the rest (the beats of one
    store land out of order). With fill_at (a fraction of the run, or a function of the run
    number giving it) a program that fills its logits itself (qwen3.fill_logits) shows the
    region as it was before the run, and ICOUNT 0, until then; FILL_SENTINEL and the whole
    PROG_N after. With anchor (a function of the run number: True for the runs it holds) such a
    run's timeline (the fill, the pieces, HALTED) starts at the host's first look instead of at
    RUN: its first ICOUNT read or memory read (a card as late as the host, however late that
    is; a host that never looks gets the run after a second)."""
    streams = True

    def __init__(self, cfg, late, piece, run_s=0.06, fill_at=None, anchor=None, **kw):
        super().__init__(ch_bytes=cfg.DRAM_BYTES // 2, devname=None, run_s=run_s, **kw)
        self.cfg, self.late, self.piece, self.fill_at = cfg, late, piece, fill_at
        self.anchor = anchor
        self.pending = []                                 # (time, channel, offset, bytes)
        self.t_fill = None
        self.ahead = None                                 # a held run's (t0, its pending)

    def _flat(self):
        from opentpu.host.board import join
        return join([c for c in self.ch])

    def _apply(self):
        now = time.perf_counter()
        keep = []
        for t, c, off, b in self.pending:
            if t <= now:
                self.ch[c][off:off + len(b)] = b
            else:
                keep.append((t, c, off, b))
        self.pending = keep

    def _look(self):
        """A held run starts its timeline now."""
        t0, ev = self.ahead
        d = time.perf_counter() - t0
        self.pending += [(t + d, c, off, b) for t, c, off, b in ev]
        self.t_fill = None if self.t_fill is None else self.t_fill + d
        self.t_run, self.ahead = t0 + d, None

    def reg_write(self, off, val):
        from opentpu import isa as I
        from opentpu.host.board import FILL_SENTINEL, split
        from opentpu.isasim import Machine
        rising = off == R.R_CTRL and val & R.CTRL_RUN and \
            not self.regs[R.R_CTRL] & R.CTRL_RUN
        super().reg_write(off, val)
        if not rising:
            return
        self._apply()
        dram = self._flat()
        a, n = self.regs[R.R_PROG_ADDR], self.regs[R.R_PROG_N]
        w = dram[a:a + 32 * n].view(np.uint32).reshape(n, 8)
        prog = [I.Instr.decode(x) for x in w]
        args = [self.regs.get(R.R_ARG0 + 4 * k, 0) for k in range(8)] if self.args else None
        m = Machine(self.cfg, [prog], [dram.copy()], args)
        m.run()
        new = m.slices[0].dram
        la, ln = self.late
        t0 = self.t_run = time.perf_counter()           # the run starts now
        new_late = new[la:la + ln].copy()
        new[la:la + ln] = dram[la:la + ln]                # the logits come later
        for c, off, part in split(0, new):
            self.ch[c][:] = part
        self.t_fill, self.ahead, ev = None, None, []
        fills = any(x.op == I.RLD and x.rd == 0 for x in prog[:64])
        if fills and self.fill_at is not None:
            f = self.fill_at(self.runs) if callable(self.fill_at) else self.fill_at
            self.t_fill = t0 + self.run_s * f
            lo, hi = la // 128 * 128, -(-(la + ln) // 128) * 128
            span = self._flat()[lo:hi].copy()             # the run's writes around it
            span[la - lo:la - lo + ln] = np.full(ln // 4, FILL_SENTINEL, np.uint32).view(np.uint8)
            for c, off, part in split(lo, span):
                ev.append((self.t_fill, c, off, part))
        npieces = -(-ln // self.piece)
        for i, o in enumerate(range(0, ln, self.piece)):
            k = min(self.piece, ln - o)
            t = t0 + self.run_s * (0.4 + 0.5 * i / npieces)
            buf = np.zeros(-(-k // 128) * 128, np.uint8)
            buf[:k] = new_late[o:o + k]
            for j, (c, off, part) in enumerate(split(la + o, buf)):
                # channel 0's beats a little before channel 1's
                ev.append((t + 0.002 * j, c, off, part[:len(part)]))
        if fills and self.anchor is not None and self.anchor(self.runs):
            self.t_run, self.ahead = float("inf"), (t0, ev)      # HALTED waits too
        else:
            self.pending += ev

    def reg_read(self, off):
        if self.ahead is not None and (off == R.R_ICOUNT or
                                       time.perf_counter() - self.ahead[0] > 1.0):
            self._look()
            if off == R.R_ICOUNT and self.t_fill is not None:
                return 0                                  # the fill lands after the look
        self._apply()
        if off == R.R_ICOUNT and self.t_fill is not None and \
                time.perf_counter() < self.t_fill:
            return 0
        return super().reg_read(off)

    def mem_read(self, ch, off, n, out=None):
        if self.ahead is not None:
            self._look()
        self._apply()
        return super().mem_read(ch, off, n, out)
