"""A merged stage: several tiles' streams of instructions interleaved in one program, so the units a
tile leaves idle serve the next tile's work (a compile choice, Config.DNMERGE; docs/pf500.md
"The tile pipeline"). The sequencer is in order and the wait entries hold what conflicts, so a merged
program is the streams' programs bit for bit when each stream's own order stands, no two streams
share a TMEM word or ACT block while either holds it, and what one stream's DRAM words feed the next
(the DeltaNet's windows) is ordered by stage_wait / stage_post.

Each stream is the tile's ordinary kernel, traced on a thread of its own on a _Stream (a Builder
with its own instruction list, loops and window register, and the slice's registers, TMEM and ACT
RAM shared), one thread running at a time. A stream stops before each instruction that goes to a
unit, the instruction in hand, and the schedule picks which head goes next:

  - Clock: of the heads of the N oldest unfinished streams, the one that can start first on a small
    model of the sequencer (dispatch one a cycle, the window, each unit's next free cycle, what the
    head conflicts with among the recent instructions). Slice 0 picks; the other slices replay its
    picks, as their GATHERs must come in the same order on every slice.
  - TMEM: when a stream's peak (from one stream traced alone) fits an Nth of TMEM, each stream has
    an Nth of its own, so no stream waits on words another just freed. If not, a tile that does not
    fit stops its stream until some other frees words, and a younger stream's tile is taken only if
    every older one can still finish, each at its own peak (the banker's rule). ACT RAM: as that
    stop.

Loops stay whole: language.range unrolls in a stream (Builder.unroll), and of the loops left (a
device-computed count's, the MoE's knock-outs) a stream stops only before the first instruction to a
unit, for the whole loop."""
from __future__ import annotations

import bisect
import gc
import threading
import weakref
from collections import deque

from . import isa as I
from . import waits as W
from .compiler import Builder, CompileError, _stack
from .isasim import Config

# a stream's own: its instructions and loops, its window register's base, its live address keys.
# All else on a Builder is the slice's.
_OWN = ("cfg", "sid", "root", "stack", "loops", "homes", "regs", "spare_regs", "pending", "emitted",
        "win_reg", "win_base", "end_zero", "loop_uses", "aw", "S")
_SKIP = frozenset((I.RLD,))        # to a unit, but its register's next user follows at once
BIG = 8                            # a tile of 1 / BIG of TMEM or more: Stage.place's big ones
SLACK = 8                          # Stage.admit's margin, 1 / SLACK of TMEM: its words are not gaps


class _Abort(BaseException):
    """A stream given up: another's error, or the stage's."""


def _shared(name: str) -> property:
    return property(lambda self: getattr(self.stage.base, name),
                    lambda self, v: setattr(self.stage.base, name, v))


class _Stream(Builder):
    unroll = True
    state = "new"

    def __init__(self, stage: "Stage", j: int):
        fresh = Builder(stage.base.cfg, stage.base.sid)
        fresh.S = stage.base.S
        self.__dict__.update({k: v for k, v in fresh.__dict__.items() if k in _OWN})
        self.stage, self.j = stage, j
        self.sem = threading.Semaphore(0)
        self.head = self.pred = None
        self.tiles: list = []           # (weakref to the tile's _Buf, words)
        self.allocs = 0
        self.loop_item = None           # an outermost loop not yet given its place
        self.bodies: list = []          # the instructions of each loop open, for Clock

    # ---- the stops
    def emit(self, ins: I.Instr) -> None:
        self.stage.before(self, ins)
        Builder.emit(self, ins)
        ins.seq = self.stage.tick()

    def begin_loop(self, n: int, rcount: int = 0):
        loop = Builder.begin_loop(self, n, rcount)
        if len(self.loops) == 1:
            self.loop_item = self.stack[-2][-1]
        self.bodies.append([])
        return loop

    def end_loop(self, loop) -> None:
        Builder.end_loop(self, loop)
        if not self.loops and self.loop_item is not None:
            self.loop_item.seq = self.stage.tick()
            self.loop_item = None
        body = self.bodies.pop() * max(loop.count, 1)
        if self.stage.clock is not None:
            for ins in body[len(body) // max(loop.count, 1):]:
                self.stage.clock.issue(self.stage.clock.head(ins))
        if self.bodies:
            self.bodies[-1] += body

    def stage_wait(self, key) -> None:
        s, k = self.stage, (self.j - 1, key)
        while self.j and k not in s.posted:
            s.block(self, lambda: k in s.posted)

    def stage_post(self, key) -> None:
        self.stage.posted.add((self.j, key))

    # ---- the slice's TMEM and ACT RAM
    def _tmem_fit(self, n: int) -> int:
        s = self.stage
        if self.loops:                  # (no stop inside a loop)
            return s.place(self, n)
        failed = [None]

        def ok():
            return s.admit(self, n) and (failed[0] is None or s.live() < failed[0])
        while True:
            if ok():
                try:
                    return s.place(self, n)
                except CompileError:
                    failed[0] = s.live()
            s.block(self, ok)

    def alloc(self, shape, rs=None, spare=0, top=False):
        t = Builder.alloc(self, shape, rs, spare, top)
        b, e, _ = next(r for r in self.tmem_regions if r[2]() is t.buf)
        self.tiles.append((weakref.ref(t.buf), e - b))
        self.allocs += 1
        self.stage.alloced(self)
        return t

    def _tmem_top(self, n: int) -> int:
        return self.stage.place(self, n, top=True)

    def skip(self, n: int) -> None:
        pass

    def act_alloc(self, KB: int):
        s = self.stage
        while True:
            free = s.act_free()
            try:
                return Builder.act_alloc(self, KB)
            except CompileError as e:
                if self.loops or "ACT RAM full" not in str(e):
                    raise
            s.block(self, lambda: s.act_free() > free)

    def live_words(self) -> int:
        self.tiles = [x for x in self.tiles if x[0]() is not None]
        return sum(n for _, n in self.tiles)

    # ---- the run's arguments stay: the other streams use them
    def release_arg(self, var) -> None:
        pass

    def _take_reg(self, what: str) -> int:
        pool = self.stage.win_pool
        return pool.pop() if pool and what == "the window register" else \
            Builder._take_reg(self, what)


for _k in vars(Builder(Config(), 0)):
    if _k not in _OWN:
        setattr(_Stream, _k, _shared(_k))
del _k


class Clock:
    """The schedule's small model of the sequencer: not a simulator (tools/pf500's board model and
    the co-simulation are); its services are the ledger's rates and fits at B2's rates, close enough
    to rank which head can start first."""
    LD_A, LD_R, ST_A, ST_R = 62, 30, 73, 32       # (latency, bytes a cycle)
    PASSES = {I.V_EXP2: 3, I.V_EXP2SUB: 3, I.V_RECIP: 2, I.V_RSQRT: 4, I.V_LOG2: 4,
              I.V_SIGMOID: 6, I.V_SILU: 6}
    SUMS = (I.V_RSUM, I.V_RSSQ, I.V_RDOT)
    KEEP = 96                                     # recent instructions a new head looks at

    def __init__(self, cfg):
        self.cfg = cfg
        self.free = [0] * W.NUNITS
        self.disp = 0
        self.win = max(cfg.WC_WIN, 16)
        self.fin: list = []                       # the finishes in flight, sorted
        self.recent = deque(maxlen=self.KEEP)
        self.R = [(0, 0)] * 16
        self.rld = [0] * 16                       # when each register's RLD has its value
        self.heads: list = []                     # the heads waiting, [ins, unit, svc, lat, hold, fp, dep, regs]
        self.lanes = cfg.LANES
        self.vpu: list = []                       # the VPU's busy spans ahead, sorted: it starts around a waiting VOP

    def service(self, ins) -> tuple:
        """(unit, its busy cycles, latency after, the cycles it holds the unit)."""
        op, w, cfg = ins.op, ins.w, self.cfg
        u = W.UNIT[op]
        if op == I.LD:
            s = 4 * w[2] / self.LD_R
            return u, s, self.LD_A, max(s, self.LD_A / 2)
        if op == I.ST:
            s = 4 * w[2] / self.ST_R
            return u, s, self.ST_A, s + self.ST_A
        if op == I.LDA:
            s = (w[1] & 0xFF) * I.qrow_bytes(w[1] >> 16, cfg.D) / 16
            return u, s, self.LD_A, max(s, self.LD_A / 2)
        if op == I.VOP:
            rows, cols = w[3] & 0xFFFF, w[3] >> 16
            f = (w[5] >> 16) & 0xFF
            ch = -(-cols // self.lanes)
            s = rows * ch * self.PASSES.get(f, 1) if f not in self.SUMS else \
                rows * -(-ch // (64 // self.lanes)) * (64 // self.lanes)
            return u, s + 2, 8 + 12 * (f in self.PASSES), s + 2
        if op == I.MM:
            N, KB, M = w[3] & 0xFFFF, (w[3] >> 16) & 0xFF, max(1, (w[5] >> 16) & 0xFF)
            G = -(-N // (1 << ins.rd))
            rows = [2] * (M // 2) + [1] * (M % 2)         # (MXU_PUMP's passes)
            s = G * sum(max(KB, r) for r in rows)
            return u, s, 600, s + 30
        if op == I.QACT:
            rows = w[1] & 0xFF
            s = rows * ((w[1] >> 16) * cfg.D / self.lanes + 8)
            return u, s + 16, 16, s + 16
        if op == I.QST:
            s = 2 * (w[3] & 0xFFFF) * (w[3] >> 16)
            return u, s + 16, 16, s + 16
        if op == I.GATHER:
            s = (w[2] & 0xFFFF) * (w[2] >> 16) * cfg.S / self.lanes
            return u, s + 16, 30, s + 16
        return u, 8, 0, 8

    def fp(self, ins):
        f = W.footprint(ins, self.R, self.cfg.D, self.cfg.S, self.cfg.AW)
        return W.Fp(f.all, tuple(x for x in f.rd if x[0] != W.SP_DRAM),
                    tuple(x for x in f.wr if x[0] != W.SP_DRAM))

    def head(self, ins) -> list:
        u, s, lat, hold = self.service(ins)
        fp = self.fp(ins)
        dep = 0
        for e, done in self.recent:
            if done > dep and W.conflict(fp, e):
                dep = done
        regs = max(self.rld[r] for r in (ins.ra, ins.rb, ins.rc))
        h = [ins, u, s, lat, hold, fp, dep, regs]
        self.heads.append(h)
        return h

    def dispatch(self, h: list) -> float:
        """In order, one a cycle, a register behind its RLD, a free place in the window."""
        d = max(self.disp + 1, h[7])
        k = bisect.bisect_right(self.fin, d)
        if len(self.fin) - k >= self.win:
            d = self.fin[len(self.fin) - self.win]
        return d

    def start(self, h: list) -> float:
        t = max(self.dispatch(h), h[6])
        if h[1] != W.U_VPU:
            return max(t, self.free[h[1]])
        for b, e in self.vpu:                     # (the first gap the VOP fits)
            if t + h[4] <= b:
                break
            t = max(t, e)
        return t

    def issue(self, h: list) -> None:
        d, s = self.dispatch(h), self.start(h)
        done = s + h[2] + h[3]
        self.disp = d
        if h[1] == W.U_VPU:
            self.vpu = [x for x in self.vpu if x[1] > d]
            bisect.insort(self.vpu, (s, s + h[4]))
        else:
            self.free[h[1]] = s + h[4]
        if h[0].op == I.RLD:
            self.rld[h[0].rd] = done
        del self.fin[:bisect.bisect_right(self.fin, d)]
        bisect.insort(self.fin, done)
        self.recent.append((h[5], done))
        self.heads = [o for o in self.heads if o is not h]
        for o in self.heads:
            if o[6] < done and W.conflict(o[5], h[5]):
                o[6] = done


class Stage:
    """One slice's merged trace of `fns` (each a stream: the tile's kernel, called with no arguments,
    on the thread's _Stream), N of them at a time. run() leaves `base`, a Builder whose root is the
    merged instructions (base.finish(): the program), and `picks`, the streams in the order they ran:
    order (slice 0's picks) replays them. curve: the live words after each of a stream's allocations
    (one stream traced alone), the banker's rule (none: no reservation)."""

    def __init__(self, cfg, sid: int, fns: list, N: int, fit: str = "next", curve: list | None = None,
                 order: list | None = None):
        self.base = Builder(cfg, sid)
        self.base.S = cfg.S
        self.base.fit = fit
        self.fns, self.N, self.order = fns, N, order
        self.ctl = threading.Semaphore(0)
        self.streams = [_Stream(self, j) for j in range(len(fns))]
        self.err = None
        self.abort = False
        self.seq = 0
        self.posted: set = set()
        self.win_pool: list = []
        self.picks: list = []
        self.clock = Clock(cfg) if order is None and N > 1 else None
        self.curve = curve
        self.trail: list = []           # the live words after each of stream 0's allocations
        W_ = cfg.TMEM_WORDS
        self.parts = N if curve and max(curve) * N <= W_ else 1
        self.cursor = [k * W_ // self.parts for k in range(self.parts)]
        if curve is not None:
            self.peak = list(curve)     # the most live words from each allocation on
            for k in range(len(curve) - 2, -1, -1):
                self.peak[k] = max(curve[k], self.peak[k + 1])

    def tick(self) -> int:
        self.seq += 1
        return self.seq

    # ---- what a stream calls (on its thread)
    def pause(self, st: _Stream) -> None:
        self.ctl.release()
        st.sem.acquire()
        if self.abort:
            raise _Abort()

    def before(self, st: _Stream, ins: I.Instr) -> None:
        if ins.op not in W.UNIT:
            return
        for body in st.bodies[-1:]:             # (Clock: repeated at the loop's end)
            body.append(ins)
        if st.loop_item is not None:            # (the stop for a whole loop)
            item, st.loop_item = st.loop_item, None
            st.state, st.head = "ready", ins
            self.pause(st)
            item.seq = self.tick()
            return
        if st.loops or ins.op in _SKIP:
            if self.clock is not None:
                self.clock.issue(self.clock.head(ins))
            return
        st.state, st.head = "ready", ins
        self.pause(st)

    def block(self, st: _Stream, pred) -> None:
        """Stop the stream until pred(). The instruction before it is sealed: a peephole
        (Builder.retarget) that moved its output into a tile taken after the stop would put it
        before what the other streams did with those words meanwhile."""
        Builder.emit(st, I.nop("stage: stream stopped"))
        st.stack[-1][-1].seq = self.tick()
        st.state, st.pred = "blocked", pred
        self.pause(st)

    def alloced(self, st: _Stream) -> None:
        if st.j == 0:
            self.trail.append(self.live())

    def live(self) -> int:
        return sum(e - b for b, e, r in self.base.tmem_regions if r() is not None)

    def act_free(self) -> int:
        return sum(1 for r in self.base.act_live if r is None or r() is None)

    def place(self, st: _Stream, n: int, top: bool = False) -> int:
        """A tile's words in the stream's part of TMEM: the big ones from its top down, the others
        next-fit in its lower half while they fit there, so its small tiles do not split the room
        its big ones need."""
        b_, W_ = self.base, self.base.cfg.TMEM_WORDS
        k = st.j % self.parts
        lo, hi = k * W_ // self.parts, (k + 1) * W_ // self.parts
        for attempt in range(2):
            regions = b_.tmem_regions = [r for r in b_.tmem_regions if r[2]() is not None]
            if top or n * BIG >= W_:
                e = hi
                for rb, re, _ in reversed(regions):
                    if rb < hi and re > lo:
                        if e - re >= n:
                            return e - n
                        e = min(e, rb)
                if e - lo >= n:
                    return e - n
            else:
                gaps, a = [], lo
                for rb, re, _ in regions:
                    if re > lo and rb < hi:
                        gaps.append((a, rb))
                        a = max(a, re)
                gaps.append((a, hi))
                if b_.fit == "best":
                    fits = [x for x in gaps if x[1] - x[0] >= n]
                    if fits:
                        return min(fits, key=lambda x: x[1] - x[0])[0]
                else:
                    for limit in ((lo + hi) // 2, hi):
                        for start in (self.cursor[k], lo):
                            for g0, g1 in gaps:
                                a = max(g0, start)
                                if min(g1, limit) - a >= n:
                                    self.cursor[k] = a + n
                                    return a
            gc.collect()
        raise CompileError(f"TMEM exhausted: a tile of {n} words in [{lo}, {hi}), "
                           f"{self.live()} live")

    def admit(self, st: _Stream, n: int) -> bool:
        """The banker's rule: with n more words on stream st, the streams can finish oldest first,
        each up to its peak from where it is. (Words, not gaps: a TMEM too split to place a tile
        stops the stage; the next fit or fewer streams then.)"""
        if self.parts > 1 or self.curve is None or \
                st is next(o for o in self.streams if o.state != "done"):
            return True
        cap = self.base.cfg.TMEM_WORDS - (self.base.desc_area if self.base.cfg.TMEM_WORDS <= 1 << 16
                                          else 0)
        avail = cap - self.live() - n - cap // SLACK
        for o in self.streams:
            if o.state == "done" or o.state == "new" and o is not st:
                continue
            own = o.live_words() + (n if o is st else 0)
            if o.allocs < len(self.peak) and avail < self.peak[o.allocs] - own:
                return False
            avail += own
            if o is st:
                break
        return True

    # ---- the controller
    def _body(self, st: _Stream, fn) -> None:
        st.sem.acquire()
        if self.abort:
            return
        stack = _stack()
        stack.append(st)
        try:
            fn()
            if st.loops:
                raise CompileError("unterminated loop")
            st.emitted += 1
            st._drain()
        except _Abort:
            pass
        except BaseException as e:      # noqa: B902 (raised again on the caller's thread)
            self.err = e
        finally:
            stack.pop()
            if st.win_reg is not None:
                self.win_pool.append(st.win_reg)
            st.state = "done"
            st.tiles = []
            self.ctl.release()

    def _resume(self, st: _Stream) -> None:
        if st.state == "done":
            raise CompileError("a merged stage's replay on another slice: a stream ended early")
        self.picks.append(st.j)
        if st.state == "ready" and self.clock is not None:
            self.clock.issue(st.clk)
        st.state, st.head, st.pred = "run", None, None
        st.sem.release()
        self.ctl.acquire()
        if self.err is not None:
            raise self.err
        if st.state == "ready" and self.clock is not None:
            st.clk = self.clock.head(st.head)

    def _next(self) -> _Stream:
        live = [s for s in self.streams if s.state != "done"][:self.N]
        for s in live:
            if s.state == "new" or s.state == "blocked" and s.pred():
                return s
        ready = [s for s in live if s.state == "ready"]
        if not ready:
            gc.collect()
            for s in live:
                if s.state == "blocked" and s.pred():
                    return s
            raise CompileError(f"TMEM exhausted: a merged stage's {self.N} streams all stopped, "
                               f"{self.live()} words live")
        if self.clock is None or len(ready) == 1:
            return ready[0]
        return min(ready, key=lambda s: (self.clock.start(s.clk), s.j))

    def run(self) -> "Stage":
        ts = [threading.Thread(target=self._body, args=(s, f), daemon=True)
              for s, f in zip(self.streams, self.fns)]
        for t in ts:
            t.start()
        try:
            if self.order is not None:
                for j in self.order:
                    self._resume(self.streams[j])
            while any(s.state != "done" for s in self.streams):
                if self.order is not None:
                    raise CompileError("a merged stage's replay on another slice stopped short")
                self._resume(self._next())
        except BaseException:
            self.abort = True
            for s in self.streams:
                s.sem.release()
            for t in ts:
                t.join()
            raise
        for t in ts:
            t.join()
        self._merge()
        return self

    def _merge(self) -> None:
        """The streams' items by the order they were emitted in; what a stream put in before an
        item later (an address register's start, Builder._init_before_loops) just before it."""
        items = []
        for st in self.streams:
            seq = 1 << 62
            for i in range(len(st.root) - 1, -1, -1):
                seq = getattr(st.root[i], "seq", seq)
                items.append((seq, st.j, i, st.root[i]))
        items.sort(key=lambda x: x[:3])
        self.base.root[:] = [x[3] for x in items]
        self.base.stack = [self.base.root]


def _gathers(b: Builder) -> list:
    return [x.w[2] for x in b.root if isinstance(x, I.Instr) and x.op == I.GATHER]


def trace_merged(cfg, fns_of_slice, N: int, fit: str = "next") -> list:
    """The stage on every slice: [Builder] (its finish(): the slice's program). fns_of_slice(s): the
    streams on slice s. CompileError where it does not fit (TMEM, ACT RAM, registers)."""
    alone = Stage(cfg, 0, fns_of_slice(0)[:1], 1, fit).run()
    first = Stage(cfg, 0, fns_of_slice(0), N, fit, curve=alone.trail).run()
    out = [first.base]
    for s in range(1, cfg.S):
        out.append(Stage(cfg, s, fns_of_slice(s), N, fit, curve=alone.trail,
                         order=first.picks).run().base)
        if _gathers(out[-1]) != _gathers(first.base):
            raise CompileError("a merged stage's GATHERs differ between slices")
    return out
