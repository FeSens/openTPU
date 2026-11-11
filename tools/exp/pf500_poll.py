"""BoardBackend's run expectations on a FakeTransport with run arguments, in real time (pf500
item 6's card A/B, docs/pf500.md). The opentpu on PYTHONPATH
is the one measured (an older tree: git archive REV opentpu targets | tar -x -C DIR).

    paste loop1|loop0: the A/B's paste (2,625 new rows from position 64, chunks of 512,
        ATTN_BLOCK 256, 40 layers: 3 DeltaNet + 1 attention), a layer's runs in a bucket as one
        run (CHAT35_LOOP 1) or runs of 2 rows (0). Device times from window 2's lp1a waits: a
        DeltaNet row 0.633 ms (its 256-row loop 0.162 s), an attention row 0.42 + 0.02 * blocks
        ms; DeltaNet programs one length for every bucket (the loop's arguments tpos0, row0,
        runs), attention's one length per bucket count.
    decode [JITTER]: one resident program, the position an argument each token, 23 ms + 0.7 us a
        position, times (1 + JITTER * u), u uniform in [-1, 1] (200 tokens).

Prints the waits past the device's end (t_seen - t_run - run_s) and the wall time."""
import sys
import time

import numpy as np

from opentpu import isa as I
from opentpu.host import regs as R
from opentpu.host.board import BoardBackend
from opentpu.host.fake import FakeTransport
from opentpu.isasim import board_config

B, CHUNK, P0, NEW, LAYERS, NR = 256, 512, 64, 2625, 40, 2


class Card(FakeTransport):
    """A run's time: next_s, or f(the ARG registers) at RUN (100 MHz cycles)."""
    def __init__(self, f=None):
        super().__init__(devname=None, args=True)
        self.f, self.next_s = f, 0.0

    def reg_write(self, off, val):
        if off == R.R_CTRL and val & R.CTRL_RUN:
            s = self.next_s if self.f is None else \
                self.f([self.regs.get(R.R_ARG0 + 4 * k, 0) for k in range(8)])
            self.run_s, self.cycles_per_run = s, int(s * 1e8)
        super().reg_write(off, val)


def backend(card):
    return BoardBackend(board_config(DRAM_BYTES=1 << 21), [np.zeros(1 << 16, np.uint8)],
                        transport=card, status=False)


def row_s(attn, blocks):
    return (0.42 + 0.02 * blocks) * 1e-3 if attn else 0.633e-3


def paste(mode):
    card = Card()
    be = backend(card)
    progs = {}

    def prog(attn, blocks):         # DeltaNet: one length; attention: by its bucket's blocks
        k = (attn, blocks if attn else 0)
        if k not in progs:
            progs[k] = [[I.ld(0, 0, 64)] * (8 + (64 + k[1] if attn else 0)) + [I.halt()]]
        return progs[k]

    late = dev = 0.0
    n_runs = 0
    t0 = time.perf_counter()
    p0 = P0
    while p0 < P0 + NEW:
        rows = min(CHUNK, P0 + NEW - p0)
        for li in range(LAYERS):
            attn = li % 4 == 3
            i = 0
            while i < rows:             # a bucket's rows
                p = p0 + i
                m = min(rows - i, B - p % B)
                blocks = p // B + 1
                if mode == "loop1":
                    work = [(m, [p, i, -(-m // NR)])]
                else:
                    work = [(min(NR, m - j), [p + j, i + j]) for j in range(0, m, NR)]
                for r, args in work:
                    card.next_s = r * row_s(attn, blocks)
                    be.start(prog(attn, blocks), args=args)
                    be.wait()
                    late += be.board.t_seen - be.board._t_run - card.run_s
                    dev += card.run_s
                    n_runs += 1
                i += m
        p0 += rows
    wall = time.perf_counter() - t0
    be.close()
    print(f"paste {mode}: {n_runs} runs, device {dev:.2f} s, waits past the end {late:.2f} s, "
          f"wall {wall:.2f} s")


def decode(jit, n=200, seed=1):
    u = np.random.default_rng(seed).uniform(-1, 1, n + 1)
    card = Card(lambda a: (23e-3 + 0.7e-6 * a[0]) * (1 + jit * u[a[0] % (n + 1)]))
    be = backend(card)
    prog = [[I.ld(0, 0, 64)] * 8 + [I.halt()]]
    late = []
    for t in range(n):
        be.start(prog, args=[t])
        be.wait()
        if t >= 4:
            late.append(be.board.t_seen - be.board._t_run - card.run_s)
    be.close()
    us = np.array(late) * 1e6
    print(f"decode, jitter +-{100 * jit:.2f} %: {len(us)} tokens, late us median "
          f"{np.median(us):.1f}, p95 {np.percentile(us, 95):.1f}, max {us.max():.1f}")


if __name__ == "__main__":
    if sys.argv[1:2] in (["-h"], ["--help"]):
        print(__doc__)
    elif sys.argv[1:2] == ["paste"] and sys.argv[2:3] in (["loop1"], ["loop0"]):
        paste(sys.argv[2])
    elif sys.argv[1:2] == ["decode"]:
        decode(float(sys.argv[2]) if len(sys.argv) > 2 else 0.0)
    else:
        sys.exit("usage: pf500_poll.py paste loop1|loop0 | decode [JITTER]")
