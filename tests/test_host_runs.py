"""Who drives the card, and how a run ends (opentpu/host/board.py, runstate.py): a locked open
ends what an earlier holder left, a run stopped by an error or a signal is ended before the lock
goes, otpu-lock never lets go while its command runs, the shared run directory's files, the
host's stop word, the bounded waits. Against FakeTransport and real processes (no card)."""
import os
import signal
import subprocess
import sys
import textwrap
import time

import numpy as np
import pytest

from opentpu.host import board as B
from opentpu.host import regs as R
from opentpu.host import runstate as rs
from opentpu.host.board import Board, BoardBackend
from opentpu.host.fake import RATES, FakeTransport
from opentpu.isasim import board_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = dict(os.environ, PYTHONPATH=ROOT)


@pytest.fixture(autouse=True)
def run_dir(tmp_path, monkeypatch):
    d = tmp_path / "otpu"
    monkeypatch.setenv("OTPU_RUN_DIR", str(d))
    for k in ("OTPU_LOCK_HELD", "OTPU_LOCK_PID", "OTPU_STOP_RUN"):
        monkeypatch.delenv(k, raising=False)
        ENV.pop(k, None)
    ENV["OTPU_RUN_DIR"] = str(d)
    return d


def _running(t) -> bool:
    st = t.reg_read(R.R_STATUS)
    return bool(st & R.ST_RUN) and not st & R.ST_HALTED


def _record_writes(t) -> list:
    """Each DMA write from now on: (bytes, whether the card was running a program then)."""
    calls, mw = [], t.mem_write

    def rec(ch, off, data):
        calls.append((len(data), _running(t)))
        mw(ch, off, data)
    t.mem_write = rec
    return calls


# ------------------------------------------------------------------------------ a locked open
def test_a_locked_open_waits_for_a_run_left_going():
    """A process killed mid-run (SIGKILL, the OOM killer) leaves RUN set and its program storing
    into DRAM: the next holder's Board waits for it to halt and stops it (CTRL = 0, WR_IDLE)
    before anything writes the card (BoardBackend writes its image next)."""
    t = FakeTransport(ch_bytes=1 << 20, run_s=0.3)
    t.reg_write(R.R_CTRL, R.CTRL_RUN)                   # the killed holder's run
    calls = _record_writes(t)
    t0 = time.perf_counter()
    b = Board(t)
    assert time.perf_counter() - t0 >= 0.25
    assert not t.regs[R.R_CTRL] & R.CTRL_RUN and not b.info()["running"]
    b.write(0, np.zeros(1 << 16, np.uint8))
    assert calls and not any(r for _, r in calls)
    b.close()


def test_a_run_that_does_not_halt_is_refused_unless_cut_short(monkeypatch):
    """A run still going after QUIESCE_WAIT: the open is refused and the run left alone (it may
    be a long generate run); OTPU_STOP_RUN=1 cuts it short at once. A monitor (lock=False)
    never touches it."""
    monkeypatch.setattr(B, "QUIESCE_WAIT", 0.2)
    t = FakeTransport(run_s=1e9)
    t.reg_write(R.R_CTRL, R.CTRL_RUN)
    with pytest.raises(B.CardRunning, match="OTPU_STOP_RUN=1"):
        Board(t)
    Board(t, lock=False).info()
    assert _running(t)
    monkeypatch.setenv("OTPU_STOP_RUN", "1")
    t0 = time.perf_counter()
    b = Board(t)                                        # (the failed open let the lock go)
    assert time.perf_counter() - t0 < 0.15 and not t.regs[R.R_CTRL] & R.CTRL_RUN
    b.close()


# ------------------------------------------------------------------------------ ending runs
def test_a_wait_that_times_out_cuts_the_run_short():
    """Board.wait's TimeoutError leaves no run going (a load right after would take its reads
    in flight for program rows; the next holder would find it running); a KeyboardInterrupt in
    the wait gives the run STOP_WAIT to halt first, then stops it the same way."""
    t = FakeTransport(run_s=1e9, devname=None)
    b = Board(t)
    b.load_program(4096, np.zeros(16, np.uint32))
    b.start()
    with pytest.raises(TimeoutError):
        b.wait(timeout=0.05)
    assert not t.regs[R.R_CTRL] & R.CTRL_RUN and not b.in_run
    t.run_s = 0.05
    b.start()

    def interrupted(*a, **k):
        del t.poll                                      # (the stop's polls are the fake's)
        raise KeyboardInterrupt
    t.poll = interrupted
    with pytest.raises(KeyboardInterrupt):
        b.wait()
    assert not t.regs[R.R_CTRL] & R.CTRL_RUN and not b.in_run
    t.run_s = 1e9
    b.start()                                           # a start without its wait: the load
    ctrl = []                                           # stops it before it writes
    rw = t.reg_write
    t.reg_write = lambda off, v: (ctrl.append(v) if off == R.R_CTRL else None, rw(off, v))
    calls = _record_writes(t)
    b.load_program(4096, np.zeros(16, np.uint32))
    assert ctrl[:1] == [0] and calls and not any(r for _, r in calls) and not b.in_run


def test_close_and_exit_stop_a_run_in_flight():
    """A Board closed, or a process ending (SystemExit, an uncaught KeyboardInterrupt) with a
    run of its own in flight, stops that run before the lock goes."""
    t = FakeTransport(run_s=1e9)
    b = Board(t)
    b.load_program(4096, np.zeros(16, np.uint32))
    b.start()
    b.close()
    assert not t.regs[R.R_CTRL] & R.CTRL_RUN


def test_sigterm_ends_a_card_holder_through_its_cleanup():
    """SIGTERM to a process that holds the card: SystemExit (exit status 143) instead of the
    default action, so its run is stopped on the way out (Board's exit hook): the card's CTRL
    reads 0 after it. Without the lock held, SIGTERM is the default again."""
    code = textwrap.dedent("""
        import atexit, time
        import numpy as np
        from opentpu.host import regs as R
        from opentpu.host.board import Board
        from opentpu.host.fake import RATES, FakeTransport
        t = FakeTransport(run_s=1e9, devname="fakeS")
        atexit.register(lambda: print("ctrl", t.regs[R.R_CTRL], flush=True))
        b = Board(t)
        b.load_program(4096, np.zeros(16, np.uint32))
        b.start()
        print("running", flush=True)
        time.sleep(30)
    """)
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True,
                         env=ENV)
    assert p.stdout.readline().strip() == "running"
    p.send_signal(signal.SIGTERM)
    out = p.stdout.read()
    assert p.wait(timeout=20) == 128 + signal.SIGTERM and "ctrl 0" in out


def test_an_interrupted_generate_run_halts_after_its_token(monkeypatch):
    """run_generate left by an error (KeyboardInterrupt here, from stop()): the stop word goes
    out (the card halts after the token in flight) and the run gets STOP_WAIT to halt, then is
    cut short."""
    from opentpu.llm import generate as G
    monkeypatch.setattr(B, "STOP_WAIT", 0.1)
    t = FakeTransport(run_s=1e9, devname=None, gen=True)
    be = BoardBackend(board_config(DRAM_BYTES=1 << 21), [np.zeros(1 << 16, np.uint8)],
                      transport=t, status=False)
    state, out = 1 << 16, (1 << 16) + 4096
    be.board.write(out, np.full(64, G.OUT_MARK, np.uint32))

    def stop():
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        be.run_generate(np.zeros(16, np.uint32), out, 16, stop=stop, state=state)
    assert not t.regs[R.R_CTRL] & R.CTRL_RUN
    assert be.board.read(state + 4 * G.S_HALT, 4).view(np.float32)[0] == 1.0
    be.close()


# ------------------------------------------------------------------------------ the stop word
def test_the_stop_write_leaves_the_cards_state_stores_alone():
    """The host's stop word mid-run: Board.write writes its beat only. Widened to its 128-byte
    chunk, it read words 0..15 and wrote them back after the card had stored the next token's
    state there (tok, tpos, left: a token lost or fed twice). A store landing between the
    host's reads and its write must stay."""
    from opentpu.llm import generate as G
    t = FakeTransport(ch_bytes=1 << 20, run_s=30.0, devname=None)
    b = Board(t)
    state = 0x4000
    w = np.zeros(G.STATE_WORDS, np.float32)
    w[G.S_TOK], w[G.S_TPOS], w[G.S_LEFT] = 100, 7, 50
    b.write(state, w)
    b.start()
    mw, card, calls = t.mem_write, np.array([200, 8, 0, 49] + [0] * 12, np.float32), []

    def stores(ch, off, data):          # the card's ST of words 0..15 lands before the write
        t._put(state, card.view(np.uint8))
        calls.append(len(data))
        mw(ch, off, data)
    t.mem_write = stores
    b.write(state + 4 * G.S_HALT, np.ones(1, np.float32))
    del t.mem_write
    got = b.read(state, 4 * G.STATE_WORDS).view(np.float32)
    assert got[G.S_TOK] == 200 and got[G.S_TPOS] == 8 and got[G.S_LEFT] == 49
    assert got[G.S_HALT] == 1.0 and calls == [4]
    b.stop()


@pytest.mark.parametrize("chash", [False, True])
def test_unaligned_writes_touch_only_their_beats(chash):
    """Board.write off the chunk grid: the bytes, and nothing else of their beats' chunks."""
    t = FakeTransport(ch_bytes=1 << 16, devname=None, chash=chash)
    b = Board(t)
    rng = np.random.default_rng(0)
    for addr, n in ((64, 4), (60, 8), (100, 300), (192, 128), (64, 128), (5, 1000), (128, 128)):
        bg = rng.integers(0, 256, 4096, dtype=np.uint8)
        b.write(0, bg)
        calls = _record_writes(t)
        d = rng.integers(0, 256, n, dtype=np.uint8)
        b.write(addr, d)
        del t.mem_write
        want = bg.copy()
        want[addr:addr + n] = d
        assert np.array_equal(b.read(0, 4096), want), (addr, n)
        lo, hi = addr // 64 * 64, -(-(addr + n) // 64) * 64
        assert sum(k for k, _ in calls) <= hi - lo, (addr, n)


def test_mtp_state_words_are_outside_the_stop_words_beat():
    """The state block's beats have one writer each during a run: the card's run-time words
    (0..15), the host's stop word and the sampler's (16..31), the MTP loop's (32..47)."""
    from opentpu.llm import generate as G
    from opentpu.llm import mtp as M
    host = {G.S_HALT, G.S_PEN, G.S_PENINV}
    card = {G.S_TOK, G.S_TPOS, G.S_RING, G.S_LEFT}
    mtp = {M.S_DRAFT, M.S_A0, M.S_A1, M.S_N, M.S_PAR, M.S_TPOS0, M.S_ITER, M.S_ACC, M.S_END}
    assert {w // 16 for w in host} == {G.S_HALT // 16} and G.S_HALT % 16 == 0
    assert {w // 16 for w in card} == {0} and G.S_STOP + G.N_STOP <= G.S_HALT
    assert {w // 16 for w in mtp} == {2} and max(mtp) < G.STATE_WORDS


# ------------------------------------------------------------------------------ bounded waits
def test_wr_idle_that_never_rises_is_an_error(monkeypatch):
    """HALTED with STATUS WR_IDLE clear: Board.wait reads it back to back for WR_SETTLE, then
    polls up to QUIET_WAIT; not set by then, the results are not all in DRAM: an error (it
    returned them as a success)."""
    monkeypatch.setattr(B, "QUIET_WAIT", 0.1)
    t = FakeTransport(run_s=0.01, devname=None)
    t.logits, t.logits_lag = (1 << 16, 4096, 1024), 0.05      # stores landing 50 ms late
    b = Board(t)
    b.load_program(4096, np.zeros(16, np.uint32))
    t0 = time.perf_counter()
    b.run()                                     # (late, but it lands: waited for)
    assert time.perf_counter() - t0 >= 0.05
    t.logits_lag = 10.0
    with pytest.raises(RuntimeError, match="WR_IDLE"):
        b.run()


def test_the_host_hooks_wait_and_a_silent_generate_run_are_bounded(monkeypatch):
    """BoardBackend._serve (the expert server's polls while a MoE run waits on them) and
    run_generate (no token for RUN_TIMEOUT) end with a TimeoutError instead of spinning with
    the lock held, and stop the run."""
    monkeypatch.setattr(B, "RUN_TIMEOUT", 0.2)
    monkeypatch.setattr(B, "STOP_WAIT", 0.05)
    t = FakeTransport(run_s=1e9, devname=None, gen=True)
    be = BoardBackend(board_config(DRAM_BYTES=1 << 21), [np.zeros(1 << 16, np.uint8)],
                      transport=t, status=False)
    be.host = lambda: 0
    with pytest.raises(TimeoutError, match="host served"):
        be.run(np.zeros(16, np.uint32))
    assert not t.regs[R.R_CTRL] & R.CTRL_RUN
    be.host = None
    from opentpu.llm import generate as G
    out = (1 << 16) + 4096
    be.board.write(out, np.full(64, G.OUT_MARK, np.uint32))
    with pytest.raises(TimeoutError, match="no token"):
        be.run_generate(np.zeros(16, np.uint32), out, 16)
    assert not t.regs[R.R_CTRL] & R.CTRL_RUN
    be.close()


# ------------------------------------------------------------------------------ CHASH, SNAP
def test_the_streamed_logits_probe_reads_the_pieces_last_beat_under_chash(run_dir, monkeypatch):
    """The probe of a streamed logits piece reads that piece's last beat on its channel under
    CHASH too (it read the beat's partner where the chunk's parity is odd, so a whole-piece
    read was wasted on a piece not yet complete, or a complete one waited a probe)."""
    from opentpu import lens as L
    from opentpu.host.board import BEAT, sim_config
    from opentpu.llm import qwen3 as Q
    monkeypatch.setattr(Q, "HEAD_CHUNK", 128)      # pieces of 128 logits: 8 in the tiny vocab
    spec, W = L._tiny_qwen()
    cfg = sim_config(spec, 256)
    t = FakeTransport(ch_bytes=cfg.DRAM_BYTES // 2, devname=None, run_s=0.02, chash=True)
    t.streams = True
    eng = Q.Engine(spec, W, cap=256, cfg=cfg, pipeline=False,
                   backend=lambda c, imgs: BoardBackend(c, imgs, transport=t, status=False))
    v, a, piece = eng.image.v_loc, eng.image.io["logits"], 4 * 128
    t.logits = (a, 4 * v, piece)
    lasts = {a + o + min(piece, 4 * v - o) - BEAT for o in range(0, 4 * v, piece)}
    probes, mr = [], t.mem_read

    def rec(ch, off, n, out=None):      # a beat of the region read while the run goes
        if n == BEAT and _running(t):
            m = off // BEAT
            x = m * 2 * BEAT + (ch ^ (m.bit_count() & 1)) * BEAT   # its logical address
            if a <= x < a + 4 * v:
                probes.append(x)
        return mr(ch, off, n, out)
    t.mem_read = rec
    for tok in (1, 2, 3):
        eng.step(tok)
    eng.backend.close()
    assert probes and set(probes) <= lasts
    assert any(B.beat_at(x, True)[0] != x // BEAT % 2 for x in lasts)   # (odd parity ones)


def test_snapshot_is_one_latch_with_another_process_snapping():
    """SNAP is the card's: another process's (otpu-smi -l) between a snapshot's reads latches
    the shadows again. snapshot() reads SNAP's count before and after the shadows and reads
    again until they agree: every counter of one latch."""
    t = FakeTransport(devname=None)
    b = Board(t)
    b.info()
    rr, k = t.reg_read, [0]

    def other(off):                             # the other process snaps mid-read, twice
        v = rr(off)
        if off == R.COUNTERS["DRAM_RD"] and k[0] < 2:
            k[0] += 1
            t.reg_write(R.R_SNAP, 1)
        return v
    t.reg_read = other
    s = b.snapshot()
    n = s["UPTIME"] // t.step                   # the latch's: every counter from that one
    assert k[0] == 2 and s["snaps"] == n
    for name in R.counters(3):
        if name != "UPTIME":
            assert s[name] == n * int(RATES[name] * t.step), name


# ------------------------------------------------------------------------------ otpu-lock
def _otpu_lock(dev: str, code: str, **kw):
    return subprocess.Popen([sys.executable, "-m", "opentpu.host.runstate", "--dev", dev, "--",
                             sys.executable, "-c", textwrap.dedent(code)],
                            stdout=subprocess.PIPE, text=True, env=ENV, **kw)


def test_a_tool_under_a_dead_otpu_lock_stops_before_touching_the_card(tmp_path):
    """otpu-lock killed with SIGKILL (no signal to pass on): its lock is free at once. A tool
    still running under it finds its holder gone at its next write or run and raises LockLost
    (on Linux it is sent SIGTERM as otpu-lock dies, PR_SET_PDEATHSIG)."""
    p = _otpu_lock("/dev/fakeD", """
        import sys
        import numpy as np
        from opentpu.host.board import Board
        from opentpu.host.fake import RATES, FakeTransport
        from opentpu.host.runstate import LockLost
        b = Board(FakeTransport(devname="fakeD"))
        print("open", flush=True)
        sys.stdin.readline()
        try:
            b.write(0, np.zeros(128, np.uint8))
            print("wrote", flush=True)
        except LockLost:
            print("lost", flush=True)
    """, stdin=subprocess.PIPE)
    assert p.stdout.readline().strip() == "open"
    p.kill()
    p.wait()
    rs.DeviceLock("fakeD", wait=0).release()            # free: the tool does not hold it
    time.sleep(0.2)
    try:
        p.stdin.write("go\n")
        p.stdin.flush()
    except BrokenPipeError:
        pass
    out = p.stdout.read().strip()
    assert out in ("lost", ""), out                     # "": it got SIGTERM (Linux)
