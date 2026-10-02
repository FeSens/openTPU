"""An adversarial link for path (a) on the ISA simulator (docs/offload.md 10.11): the expert
server's writes land one 64-byte beat at a time, in the order the server makes them, and the
machine runs again as soon as a WAITW of a waiting slice holds. The card reads whatever has
landed by then, so the beats it reads before its tags are proved to be there.

tag_first: each slot's tag beat lands before its expert's bytes (the negative control: the
card must then compute experts that have not landed, and the logits move)."""
from collections import deque

import numpy as np

from opentpu import isa as I


def _holds(s, ins) -> bool:
    a = (s.reg(ins.ra) + ins.w[0]) & 0xFFFFFFFF
    return I.waitw_holds(int(s.m32[a // 4]), s.reg(ins.rc) + ins.w[2], ins.flags & 3, ins.w[3])


class BeatLink:
    """The engine's expert server's memory, and its machine's WAITW host (the engine has one
    server: no row server)."""

    def __init__(self, eng, tag_first: bool = False):
        assert eng.row_server is None
        self.srv, self.mem, self.tag_first = eng.server, eng.server.mem, tag_first
        self.q: deque = deque()
        self.beats = self.held = 0          # beats landed; WAITWs that held with beats queued
        self.srv.mem = self
        eng.backend.machine.host = self.host

    def _queue(self, addr: int, data) -> None:
        b = np.frombuffer(np.ascontiguousarray(data).tobytes() if isinstance(data, np.ndarray)
                          else bytes(data), np.uint8)
        while len(b):
            n = min(len(b), 64 - addr % 64)
            self.q.append((addr, b[:n].copy()))
            addr, b = addr + n, b[n:]

    def write(self, addr: int, data) -> None:
        self._queue(addr, data)

    def write_slot(self, addr: int, data, tag=None, then=None) -> None:
        """As BoardDram's: then (the request's answer) after the expert's first fifth."""
        b = np.frombuffer(np.ascontiguousarray(data).tobytes() if isinstance(data, np.ndarray)
                          else bytes(data), np.uint8)
        c = len(b) // 5 // 64 * 64
        if tag is not None and self.tag_first:
            self._queue(*tag)
        self._queue(addr, b[:c])
        if then is not None:
            then()
        self._queue(addr + c, b[c:])
        if tag is not None and not self.tag_first:
            self._queue(*tag)

    def read(self, addr: int, n: int) -> bytes:
        return self.mem.read(addr, n)       # (the card's words: seq, the row)

    def step(self) -> bool:
        if not self.q:
            return False
        self.mem.write(*self.q.popleft())
        self.beats += 1
        return True

    def host(self, m) -> None:
        """Every slice waits: land one beat at a time until a waiting WAITW holds; with none
        queued, the server polls (a new request queues its writes)."""
        while not any(s.polling is not None and _holds(s, s.polling) for s in m.slices):
            if not self.step() and not self.srv.poll():
                return                      # (nothing to land: the simulator's timeout)
        self.held += bool(self.q)
