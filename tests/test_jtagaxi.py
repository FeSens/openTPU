"""opentpu.host.jtagaxi's DRAM access over the full BCU1525 image's AXI4 JTAG-to-AXI master, on a
fake of jtagaxi.tcl's mr / mw commands (no Vivado): the bursts (at most 256 words, none across
4 KiB), the commands' chunks, and JtagTransport's whole-beat writes."""
import threading

import numpy as np
import pytest

from opentpu.host import jtagaxi as J


class FakeJtag(J.JtagAxi):
    """jtagaxi.tcl's mr / mw on a dict of 32-bit words, checking each burst as AXI4 takes it."""

    def __init__(self):
        self.devname, self.lock, self.words, self.cmds = "bcu1525-fake", threading.Lock(), {}, []
        self.prio, self.prio_cv = 0, threading.Condition()

    def cmd(self, line, timeout=None):
        op, *args = line.split()
        self.cmds.append(op)
        out = []
        while args:
            a, n = args[0], int(args[1])
            assert len(a) == 16 and 1 <= n <= J.BURST
            a = int(a, 16)
            assert a % 4 == 0 and a // 4096 == (a + 4 * n - 1) // 4096, (hex(a), n)
            if op == "mw":
                for k, w in enumerate(args[2:2 + n]):
                    self.words[a + 4 * k] = int(w, 16)
                args = args[2 + n:]
            else:
                out += [f"{self.words.get(a + 4 * k, 0):08x}" for k in range(n)]
                args = args[2:]
        return out


def test_bursts():
    assert J.bursts(0, 600) == [(0, 256), (1024, 256), (2048, 88)]
    assert J.bursts(0xFF0, 10) == [(0xFF0, 4), (0x1000, 6)]
    assert J.bursts(0x8000_0000 - 8, 4) == [(0x7FFF_FFF8, 2), (0x8000_0000, 2)]
    with pytest.raises(ValueError):
        J.bursts(2, 1)


def test_mem_round_trip():
    f = FakeJtag()
    rng = np.random.default_rng(1)
    data = rng.integers(0, 256, 4 * (J.MEM_CHUNK + 777), np.uint8).tobytes()
    addr = 0x1_8000_0F00                                  # channel 3, across 4 KiB boundaries
    f.mem_write(addr, data)
    assert f.cmds.count("mw") == 2                        # MEM_CHUNK words per command
    assert f.mem_read(addr, len(data)) == data
    assert f.mem_read(addr + 4, 8) == data[4:12]
    with pytest.raises(ValueError):
        f.mem_read(addr, 6)


def test_transport_whole_beats():
    f = FakeJtag()
    t = J.JtagTransport(f)
    base = np.arange(256, dtype=np.uint8)
    t.mem_write(1, 0x40, base)
    t.mem_write(1, 0x40 + 70, np.full(10, 0xEE, np.uint8))   # off the beats: widened, merged
    want = base.copy()
    want[70:80] = 0xEE
    assert np.array_equal(t.mem_read(1, 0x40, 256), want)
    assert np.array_equal(t.mem_read(1, 0x40 + 3, 9), want[3:12])  # off the words
    out = np.zeros(16, np.uint8)
    assert t.mem_read(1, 0x40 + 16, 16, out) is out and np.array_equal(out, want[16:32])
    # every write the fake saw was whole 64-byte beats at the channel's DMA address
    assert min(a for a in f.words) == t.bases[1] + 0x40
    assert len(f.words) * 4 == 256


def test_priority_holds_transfers():
    """A priority section (the guard's regulator reading) goes before a DRAM transfer: the
    transfer's next mr / mw waits while it is open, so the section's commands run back to back."""
    f = FakeJtag()
    data = bytes(4 * 3 * J.MEM_CHUNK)                     # three mw commands
    inside, done = threading.Event(), threading.Event()

    def reading():
        with f.priority():
            inside.set()
            done.wait(5)
            for _ in range(3):
                f.cmds.append("r")

    th = threading.Thread(target=reading)
    th.start()
    inside.wait(5)
    w = threading.Thread(target=f.mem_write, args=(0x1000, data))
    w.start()
    w.join(0.3)
    assert w.is_alive() and "mw" not in f.cmds            # held before its first command
    done.set()
    th.join(5)
    w.join(5)
    assert f.cmds == ["r", "r", "r", "mw", "mw", "mw"]
    assert f.prio == 0


def test_a_late_answer_is_not_the_next_commands():
    """A command that times out leaves its answer to come (Vivado answers every line, in order):
    the next command's answer is its own, not that one."""
    import queue

    class Proc:
        class stdin:
            lines = []
            write = lines.append
            flush = staticmethod(lambda: None)
        poll = staticmethod(lambda: None)

    jt = object.__new__(J.JtagAxi)
    jt.lock, jt.p, jt.q, jt.timeout, jt.late = threading.Lock(), Proc, queue.Queue(), 30.0, 0
    with pytest.raises(J.JtagAxiError, match="no answer"):
        jt.cmd("r 0", 0.01)
    with pytest.raises(J.JtagAxiError, match="no answer"):
        jt.cmd("r 4", 0.01)
    for a in ("OK 11", "OK 22", "OK 33", "OK 44"):         # the two late ones, then two more
        jt.q.put(a)
    assert jt.cmd("r 8", 1.0) == ["33"]
    assert jt.cmd("r c", 1.0) == ["44"] and jt.late == 0
    jt.q.put("ERR no target")                               # (an error is an answer too)
    with pytest.raises(J.JtagAxiError, match="no target"):
        jt.cmd("r 10", 1.0)
    jt.q.put("OK 55")
    assert jt.cmd("r 14", 1.0) == ["55"]
