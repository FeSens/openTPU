"""Cards in tensor parallel: one machine of S slices over NCARDS cards of S / NCARDS each, their
collectives joined by the link port (targets/bcu1525-x2.toml: two BCU1525 of two slices; the
multi-FPGA tensor-parallel study's section 7; the link port's
CSRs, docs/x2_link.md).

CardsTransport puts the cards' transports together as one transport of S slices, so Board,
BoardBackend and the Engine run on them as on one card:
- global slice s is card s // spc's local slice s % spc (spc = S / NCARDS): its channels
  nch s + c are that card's nch (s % spc) + c, so each card's DRAM holds its own slices' shards
  (the images of one compile at S) and its slices' programs, at the same addresses on every card;
- every register write goes to every card, the last card first and card 0 last (CTRL RUN too:
  the start skew is harmless, the first collective absorbs it);
- reads: STATUS the cards' together (HALTED, WR_IDLE, RUN and the calibration bits when every
  card has them, ERROR and the other bits when one has), slice s's ICOUNT its card's, CFG_SLICES
  their sum, CFG_HASH the whole machine's (arch_hash of the configuration a card reports at all
  the slices; each card's own hash checked first, at its own slices), SCRATCH card 0's when every
  card has the same (else 0: Board.scrub scrubs); every other register card 0's (its counters,
  the trace buffer, the run's CYCLES);
- a poll waits for every card, card 0 first.

set_cards() writes each card's X2_CARD (CARD_ID / NCARDS) and brings the link
up; links() reads each port's hello: the cable map, checked against the ring (two cards: each port
of a card on the other card, either way round). CardsTransport does both as it opens.

The host computes nothing per token (the user's rule, docs: no host compute): the embedding and
the sampling run on the cards (the generate loop); every card holds the same logits after the
head's gather and picks the same token, and the host reads the tokens from card 0 (slice 0's
DRAM).

The cards' transports:
- open_cards(devs): XDMA (opentpu.host.board.XdmaTransport), e.g. two BCU1525s in one host,
  /dev/xdma0 as card 0 and /dev/xdma1 as card 1 (the cards' order is the caller's);
  open_transport("/dev/xdma0+/dev/xdma1") is the same;
- X2SimTransport(...).cards: the two-board co-simulation (both cards and the link model in one
  Verilator simulation: each card's register lines "@<card> ", its channels 16 card + 8 local
  slice + c);
- IsaCards(cfg).cards: the ISA simulator behind every card's registers (the host's protocol end to
  end without RTL: tests/test_cards.py).
"""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np

from . import regs as R
from .board import ConfigMismatch, SimTransport, channel_map, chunk_hash
from .fake import FakeTransport

LINK_UP_WAIT = 2.0              # set_cards: the longest wait for every card's link port up (s)


class LinkError(RuntimeError):
    """The cards' link is not up, failed during a run, or is not cabled as the ring needs."""


def card_info(t) -> dict:
    """A card's identity and configuration from its registers (Board.info's, without its target
    check: a card reports its own slices, the machine's target all of them), "xcard" its link
    port (CAPS2 bit15). One reg_read_many."""
    offs = [R.R_ID, R.R_VERSION, R.R_REGMAP, R.R_CAPS, R.R_CAPS2, R.R_BUILD_ID, R.R_ACT_ROWS,
            *R.CFG_REGS]
    ident, v, rm, cp, c2, bid, rows, *cfg = t.reg_read_many(offs)
    if ident != R.ID_OTPU:
        raise RuntimeError(f"no openTPU on the card (ID register {ident:#x})")
    rm = R.regmap(rm)
    if rm < R.CAPS2_SINCE:
        raise ConfigMismatch(f"register map {rm}: a card of a multi-card machine reports its "
                             f"configuration and CAPS2 (map {R.CAPS2_SINCE} or later)")
    caps = R.caps(cp)
    return {"D": v >> 16, "MCOLS": (v >> 8) & 0xFF, "LANES": v & 0xFF, "regmap": rm,
            "caps": caps, "caps_raw": cp, "caps2": R.caps2(c2), "caps2_raw": c2,
            "xcard": bool(c2 & R.CAP2_XCARD), "build_id": bid,
            "act_rows": rows if cp & R.CAP_ACT else (v >> 8) & 0xFF, "cfg": R.cfg(cfg)}


class _CardsLock:
    """The cards' device locks as one (runstate.DeviceLock's interface, Board's use of it)."""

    def __init__(self, locks: list):
        self.locks = locks
        self.name = "+".join(lk.name for lk in locks)
        self.holder = locks[0].holder
        self.cleanup: list = []

    def lost(self) -> bool:
        return any(lk.lost() for lk in self.locks)

    def release(self) -> None:
        cleanup, self.cleanup = self.cleanup, []
        for ref in cleanup:
            f = ref()
            if f is not None:
                f()
        for lk in self.locks:
            lk.release()


class _Results:
    """CardsTransport.results: a queued read's value, its cards' reads combined."""

    def __init__(self, ct: CardsTransport):
        self.ct = ct

    def __getitem__(self, k: int) -> int:
        reads, comb = self.ct._done[k]
        return comb([self.ct.cards[c].results[i] for c, i in reads])

    def __len__(self) -> int:
        return len(self.ct._done)


# STATUS bits every card must have for the machine to (AND), the others any card's (OR)
ST_ALL = R.ST_HALTED | R.ST_WR_IDLE | R.ST_CALIB0 | R.ST_CALIB1 | R.ST_RUN


def status(vals: list[int]) -> int:
    """The cards' STATUS values as the machine's."""
    a, o = ST_ALL, 0
    for v in vals:
        a &= v
        o |= v
    return a | (o & ~ST_ALL)


class CardsTransport:
    """The cards' transports (card c's at index c; each card's slices spc = S / NCARDS) as one
    transport of S slices (the module's docstring). target: the machine's (a Target or a name:
    Board.info checks the cards are its build, check_target); link False: CARD_ID / NCARDS and
    the link are left as they are (set_cards and links not run)."""

    def __init__(self, cards: list, target=None, link: bool = True,
                 watchdog: int = R.COLL_WD_DEFAULT):
        if not cards:
            raise ValueError("no cards")
        if not 0 <= watchdog <= 0xFFFFFFFF:
            raise ValueError("collective watchdog must be a uint32 core-cycle count")
        self.watchdog = watchdog
        self._checks: list = []                 # batched completion reads, checked at flush
        self.collectives: int | None = None     # known dynamic count of the loaded program
        self._before = None                    # this run's sequences (batched: read indices)
        self.sequences = self.collective_delta = None  # last successful completion check
        self.cards = list(cards)
        n = self.ncards = len(cards)
        modes = {bool(getattr(t, "batched", False)) for t in cards}
        if len(modes) > 1:
            raise ValueError("cards of a batched transport (a simulation) and of a direct one")
        self.batched = modes.pop()
        self.keeps_state = all(getattr(t, "keeps_state", True) for t in cards)
        self.ecc = any(getattr(t, "ecc", False) for t in cards)
        self.threaded = all(getattr(t, "threaded", False) for t in cards)
        self.streams = all(getattr(t, "streams", False) for t in cards)
        self.run_h2c = min(getattr(t, "run_h2c", 0) for t in cards)
        if all(hasattr(t, "host_idle") for t in cards):
            self.host_idle = max(t.host_idle for t in cards)
        names = [getattr(t, "devname", None) for t in cards]
        self.devname = "+".join(names) if all(names) else None
        self.dev = "+".join(getattr(t, "dev", None) or nm for t, nm in zip(cards, names)) \
            if self.devname else None
        locks = [getattr(t, "_otpu_lock", None) for t in cards]
        self._otpu_lock = _CardsLock(locks) if all(lk is not None for lk in locks) else None
        if target is not None:
            self.want_target = target
        self.poll_exact = False
        self._q: list = []                      # (batched) queued reads: ([(card, index)], comb)
        self._done: list = []                   # the last flush's
        self.infos = infos = [card_info(t) for t in cards]
        self._check(infos)
        i0, c0 = infos[0], infos[0]["cfg"]
        self.spc, self.nch, self.beat = c0["S"], c0["channels"], c0["beat_bytes"]
        self.slices = n * self.spc
        self.gather = True
        self.icache = bool(i0["caps2"]["icache"])
        for t, i in zip(cards, infos):          # each card's DMA at its own channels' addresses
            if hasattr(t, "bases"):
                t.bases = channel_map(i)
        self.link = link and i0["xcard"]        # (one card of a bitstream without the port: none)
        self.map: dict = {}
        if self.link:
            self.set_cards()

    def _check(self, infos: list) -> None:
        """Every card the same bitstream (identity, capabilities, configuration, BUILD_ID), with a
        link port when there are several, its CFG_HASH its own configuration's; then the
        machine's CFG_HASH (_hash) from it at every card's slices."""
        from opentpu.target import arch_hash
        from .board import _reg_config, check_hash, device_aw, device_wc_win
        i0 = infos[0]
        key = ("D", "MCOLS", "LANES", "regmap", "caps_raw", "caps2_raw", "build_id", "act_rows",
               "cfg")
        for c, i in enumerate(infos[1:], 1):
            diff = [k for k in key if i[k] != i0[k]]
            if diff:
                raise ConfigMismatch(f"card {c}'s bitstream is not card 0's ({', '.join(diff)} "
                                     f"differ): every card runs the same build")
        if len(infos) > 1 and not i0["xcard"]:
            raise ConfigMismatch("the cards' bitstream has no link port (CAPS2 bit15 clear): "
                                 "its collectives do not span cards")
        check_hash(i0)
        aw = device_aw(i0)
        win = device_wc_win(i0, aw)
        whole = dict(i0, cfg=dict(i0["cfg"], S=i0["cfg"]["S"] * len(infos)))
        self._card_hash = i0["cfg"]["hash"]
        self._hash = arch_hash(_reg_config(whole, aw, win)) if self._card_hash else 0

    # ------------------------------------------------------------------ registers
    def _plan(self, off: int) -> tuple[list, object]:
        """The cards' reads a read of `off` takes ([(card, offset)]) and their combination."""
        n, spc = self.ncards, self.spc
        if off == R.R_STATUS:
            return [(c, off) for c in range(n)], status
        if R.R_ICOUNT_S + 4 <= off < R.R_ICOUNT_S + 4 * n * spc:
            c, k = divmod((off - R.R_ICOUNT_S) // 4, spc)
            return [(c, R.R_ICOUNT if k == 0 else R.R_ICOUNT_S + 4 * k)], lambda v: v[0]
        if off == R.R_CFG_SLICES:
            return [(c, off) for c in range(n)], sum
        if off == R.R_CFG_HASH:
            return [(0, off)], lambda v: self._hash if v[0] == self._card_hash else v[0]
        if off == R.R_SCRATCH:
            return [(c, off) for c in range(n)], lambda v: v[0] if len(set(v)) == 1 else 0
        return [(0, off)], lambda v: v[0]

    def reg_write(self, off: int, val: int) -> None:
        if self.link and off == R.R_CTRL and val & R.CTRL_RUN:
            # Read before either RUN write. A replay transport's indices refer to this same
            # script, so its baseline is LRST's zero, or a previous run in this script.
            self._before = [t.queue_read(R.R_X2_SEQ) if self.batched else
                            t.reg_read(R.R_X2_SEQ) for t in self.cards]
        for t in reversed(self.cards):
            t.reg_write(off, val)

    def collective_program(self, programs) -> None:
        """Board.load_program's assembled programs: count immediate LOOPs without unrolling.
        A register-counted collective loop or HALT CHAIN is unknown; its caller can supply
        collectives explicitly when it knows the dynamic count. Zero is a known count too."""
        from opentpu import isa as I

        def count(ins, nested=False):
            n, pc = 0, 0
            while pc < len(ins):
                i = ins[pc]
                if i.op == I.HALT:
                    return None if nested or i.flags & I.F_CHAIN else n
                if i.op == I.LOOP:
                    end = pc + 1 + i.w[0]
                    if end > len(ins) or i.w[0] < 1:
                        return None
                    if not i.ra and not i.w[1]:    # a skipped body need not have a known count
                        pc = end
                        continue
                    body = count(ins[pc + 1:end], nested=True)
                    if body is None or (i.ra and body):
                        return None
                    n += body * i.w[1]
                    pc = end
                else:
                    n += i.op in (I.BAR, I.GATHER)
                    pc += 1
            return n

        counts = [count([I.Instr.decode(w[j:j + 8]) for j in range(0, len(w), 8)])
                  for w in programs]
        self.collectives = counts[0] if len(set(counts)) == 1 else None

    def reg_read(self, off: int) -> int:
        return self.reg_read_many([off])[0]

    def reg_read_many(self, offs: list[int]) -> list[int]:
        plans = [self._plan(o) for o in offs]
        if self.batched:
            idx = [self.queue_read(o) for o in offs]
            self.flush()
            return [self.results[k] for k in idx]
        want: list[list[int]] = [[] for _ in self.cards]
        for reads, _ in plans:
            for c, o in reads:
                want[c].append(o)
        got = [iter(t.reg_read_many(w)) if w else iter(()) for t, w in zip(self.cards, want)]
        return [comb([next(got[c]) for c, _ in reads]) for reads, comb in plans]

    def poll(self, off: int, mask: int, val: int, timeout: float = 600.0,
             expect: float = 0.0) -> int:
        """Until (reg & mask) == val on every card: card 0 first (with `expect`), then each
        other in the time left. A STATUS poll returns the cards' STATUS together; a run that
        halted with any hub/port cause, STATUS ERROR or a sequence mismatch raises LinkError."""
        if self.batched:
            for t in self.cards:
                t.poll(off, mask, val, timeout, expect)
            if self.link and off == R.R_STATUS and mask & val & R.ST_HALTED:
                self._checks.append(([[t.queue_read(o) for o in self.RUN_CHECK_OFFS]
                                      for t in self.cards], self._before, self.collectives))
            return val
        t0, vals = time.perf_counter(), []
        for c, t in enumerate(self.cards):
            left = max(0.0, timeout - (time.perf_counter() - t0))
            try:
                vals.append(t.poll(off, mask, val, left, expect if c == 0 else 0.0))
            except TimeoutError as e:
                raise TimeoutError(f"card {c}: {e}") from None
        self.poll_exact = bool(getattr(self.cards[0], "poll_exact", False))
        if off != R.R_STATUS:
            return vals[0]
        st = status(vals)
        if self.link and mask & val & R.ST_HALTED:
            self.check_run()
        return st

    # (batched: the cards' views share one simulation)
    def queue_read(self, off: int) -> int:
        reads, comb = self._plan(off)
        self._q.append(([(c, self.cards[c].queue_read(o)) for c, o in reads], comb))
        return len(self._q) - 1

    @property
    def results(self) -> _Results:
        return _Results(self)

    def flush(self) -> None:
        for t in self.cards:
            t.flush()
        self._done, self._q = self._q, []
        checks, self._checks = self._checks, []
        for idx, before, expected in checks:
            self._check_run([[t.results[k] for k in ix] for t, ix in zip(self.cards, idx)],
                            None if before is None else
                            [t.results[k] for t, k in zip(self.cards, before)], expected)
        self._before = None                    # indices belong only to the completed flush

    def wait_cycles(self, n: int) -> None:
        self.cards[0].wait_cycles(n)

    # ------------------------------------------------------------------ memory
    def _card(self, ch: int) -> tuple:
        c, k = divmod(ch, self.nch * self.spc)
        return self.cards[c], k

    def mem_write(self, ch: int, off: int, data: np.ndarray) -> None:
        t, k = self._card(ch)
        t.mem_write(k, off, data)

    def mem_read(self, ch: int, off: int, n: int, out: np.ndarray | None = None) -> np.ndarray:
        t, k = self._card(ch)
        return t.mem_read(k, off, n, out)

    @property
    def bases(self) -> tuple:
        """The channels' card addresses (each card's own: Board's DMA placement)."""
        return tuple(b for t in self.cards for b in getattr(t, "bases", ()))

    @bases.setter
    def bases(self, v) -> None:
        pass                    # (each card's are its own channels': set as it opened)

    def close(self) -> None:
        for t in self.cards:
            if hasattr(t, "close"):
                t.close()

    # ------------------------------------------------------------------ the link
    def _writer(self, t):
        """A card's register write that lasts: a simulation's for every flush (`persist`)."""
        return t.persist if not getattr(t, "keeps_state", True) else t.reg_write

    def _stopped(self) -> None:
        """LRST is legal only with every core stopped and its stores drained."""
        mask = R.ST_RUN | R.ST_LOADING | R.ST_WR_IDLE
        for c, t in enumerate(self.cards):
            if self.batched:
                t.persist_poll(R.R_STATUS, mask, R.ST_WR_IDLE)
            elif t.reg_read(R.R_STATUS) & mask != R.ST_WR_IDLE:
                raise LinkError(f"card {c}: stop both cards and drain their stores before LRST")

    def set_cards(self, wait: float = LINK_UP_WAIT) -> None:
        """Write X2_CARD everywhere, LRST both stopped cards, wait for both ports on each,
        read and validate the hellos, then set COLL_WD (core cycles, 0 off). Starts need no
        barrier: the first collective absorbs the start skew; RUN never resets the port."""
        self._stopped()
        self._before = None
        for c, t in enumerate(self.cards):
            self._writer(t)(R.R_X2_CARD, c | self.ncards << 8)
        for t in self.cards:
            self._writer(t)(R.R_X2_CTRL, R.X2_LRST)
        t0 = time.perf_counter()
        for c, t in enumerate(self.cards):
            for p, off in enumerate(R.R_X2_STATUS):
                if self.batched:
                    t.persist_poll(off, R.X2_UP, R.X2_UP)
                    continue
                try:
                    t.poll(off, R.X2_UP, R.X2_UP,
                           max(0.0, wait - (time.perf_counter() - t0)))
                except TimeoutError:
                    raise LinkError(f"card {c} port {p} not up {wait:g} s after LRST: "
                                    f"{self.describe()}") from None
        self.map = self.links() if self.ncards > 1 else {}
        for t in self.cards:
            self._writer(t)(R.R_COLL_WD, self.watchdog)

    def recover(self, wait: float = LINK_UP_WAIT) -> None:
        """Stop both cores, drain stores and LRST both links. Does not replay a token: the
        caller restores its token state and re-runs it (at most one retry, never a loop)."""
        self.reg_write(R.R_CTRL, 0)
        for c, t in enumerate(self.cards):
            try:
                t.poll(R.R_STATUS, R.ST_RUN | R.ST_LOADING | R.ST_WR_IDLE, R.ST_WR_IDLE, wait)
            except TimeoutError:
                raise LinkError(f"card {c}: stores did not drain before recovery LRST") from None
        self.set_cards(wait)

    def _link_regs(self) -> list[dict]:
        offs = [R.R_X2_CARD, R.R_X2_SEQ, R.R_COLL_STATUS,
                *R.R_X2_STATUS, *R.R_X2_HELLO, *R.R_X2_CRC]
        if self.batched:
            idx = [[t.queue_read(o) for o in offs] for t in self.cards]
            self.cards[0].flush()
            vals = [[t.results[i] for i in ix] for t, ix in zip(self.cards, idx)]
        else:
            vals = [t.reg_read_many(offs) for t in self.cards]
        out = []
        for topo, seq, coll, s0, s1, h0, h1, crc0, crc1 in vals:
            out.append({"card": topo & 0xFF, "ncards": topo >> 8 & 0xFF,
                        "seq": seq & 0xFFFF, "coll": coll, "stat": [s0, s1],
                        "crc": [crc0, crc1],
                        "peers": [{"card": h & 0xFF, "port": h >> 8 & 0xF,
                                   "ncards": h >> 12 & 0xFF,
                                   "seen": h >> 24 == R.X2_HELLO_MAGIC} for h in (h0, h1)]})
        return out

    @staticmethod
    def causes(v: int) -> list[str]:
        return [nm for i, nm in enumerate(R.COLL_CAUSES) if v >> i & 1]

    def describe(self, regs: list | None = None) -> str:
        """Each card's X2 status and collective sequence, causes and the peers' hellos."""
        regs = self._link_regs() if regs is None else regs
        out = []
        for c, r in enumerate(regs):
            ps = ", ".join(f"port {p}: status {r['stat'][p]:#x}, CRC {r['crc'][p]}, " +
                           (f"card {q['card']} port {q['port']} of {q['ncards']}"
                            if q["seen"] else "no hello") for p, q in enumerate(r["peers"]))
            out.append(f"card {c} (CARD_ID {r['card']}/{r['ncards']}, X2_SEQ {r['seq']}, "
                       f"COLL_STATUS {r['coll']:#x} {'/'.join(self.causes(r['coll']))}): {ps}")
        return "; ".join(out)

    RUN_CHECK_OFFS = [R.R_STATUS, R.R_X2_SEQ, R.R_COLL_STATUS, *R.R_X2_STATUS, *R.R_X2_CRC]

    def _check_run(self, vals: list, before=None, expected: int | None = None) -> None:
        bad, seqs = [], []
        for c, (st, seq, coll, p0, p1, crc0, crc1) in enumerate(vals):
            seqs.append(seq & 0xFFFF)
            causes = self.causes(coll)
            if st & R.ST_ERROR:
                causes.append(f"STATUS ERROR ({st:#x})")
            for p, ps, crc in ((0, p0, crc0), (1, p1, crc1)):
                names = [nm for bit, nm in ((R.X2_CRC, "CRC error"),
                         (R.X2_WATCHDOG, "credit watchdog"), (R.X2_ABORT, "ABORT received"),
                         (R.X2_DROPPED, "dropped while up")) if ps & bit]
                if not ps & R.X2_UP:
                    names.append("not up")
                if names:
                    causes.append(f"port {p}: {'/'.join(names)} (status {ps:#x}, CRC {crc})")
            if causes:
                bad.append(f"card {c}: {'; '.join(causes)}")
        if len(set(seqs)) != 1:
            bad.append(f"X2_SEQ mismatch: {seqs}")
        deltas = None if before is None else [(s - b) & 0xFFFF for s, b in zip(seqs, before)]
        if deltas is not None and expected is not None:
            if any(d != expected & 0xFFFF for d in deltas):
                bad.append(f"X2_SEQ per-run delta {deltas}, expected {expected} modulo 65536 "
                           f"(before {before}, after {seqs})")
        if bad:
            raise LinkError("the cards' run: " + "; ".join(bad))
        self.sequences, self.collective_delta = seqs, deltas

    def check_run(self) -> None:
        """Check a completed run before returning results, even without STATUS ERROR. Batched
        transports queue these reads after their HALTED polls and check them at the flush."""
        if self.batched:
            # An explicit check must be queued with its baseline, just like poll's check.
            idx = [[t.queue_read(o) for o in self.RUN_CHECK_OFFS] for t in self.cards]
            self._checks.append((idx, self._before, self.collectives))
            self.flush()
        else:
            vals = [t.reg_read_many(self.RUN_CHECK_OFFS) for t in self.cards]
            self._check_run(vals, self._before, self.collectives)

    def link_errors(self) -> str:
        regs = self._link_regs()
        return self.describe(regs) if any(r["coll"] & R.COLL_ERRORS or
            any(s & R.X2_ERRORS or not s & R.X2_UP for s in r["stat"]) for r in regs) else ""

    def links(self) -> dict:
        """Read the hello cable map. At two cards either straight or crossed cables work;
        every hello must name the other card and agree on NCARDS, and the map is symmetric.
        At N >= 4 port 0 meets port 1 of the next card (the ring)."""
        n = self.ncards
        regs = self._link_regs()
        bad, m = [], {}
        for c, r in enumerate(regs):
            if (r["card"], r["ncards"]) != (c, n):
                bad.append(f"card {c}: CARD_ID/NCARDS {r['card']}/{r['ncards']}, not {c}/{n}")
            for p, q in enumerate(r["peers"]):
                st = r["stat"][p]
                if not st & R.X2_UP or st & R.X2_ERRORS:
                    bad.append(f"card {c} port {p}: not up or sticky error (status {st:#x})")
                if not q["seen"]:
                    bad.append(f"card {c} port {p}: no hello")
                    continue
                m[(c, p)] = (q["card"], q["port"])
                if q["ncards"] != n:
                    bad.append(f"card {c} port {p}: its peer is of {q['ncards']} cards, not {n}")
                if n <= 3:
                    ok = q["card"] != c and q["card"] < n and q["port"] in (0, 1)
                else:
                    ok = (q["card"], q["port"]) == (((c + 1) % n, 1) if p == 0 else
                                                    ((c - 1) % n, 0))
                if not ok:
                    bad.append(f"card {c} port {p} is cabled to card {q['card']} port "
                               f"{q['port']}: not the ring's")
        for a, b in m.items():
            if m.get(b) != a:
                bad.append(f"card {a[0]} port {a[1]} sees card {b[0]} port {b[1]}, "
                           f"which sees {m.get(b)}")
        if bad:
            raise LinkError("the cards' link: " + "; ".join(bad) + f" ({self.describe(regs)})")
        return m


def open_cards(devs: list[str], dma: bool = True, target=None, link: bool = True):
    """The cards at `devs` (XDMA device paths, card c at index c) as one CardsTransport; each
    card's device lock taken as it opens (a card busy: runstate.DeviceBusy, the others closed)."""
    from .board import XdmaTransport
    ts = []
    try:
        for d in devs:
            ts.append(XdmaTransport(d, dma=dma))
        return CardsTransport(ts, target=target, link=link)
    except BaseException:
        for t in ts:
            t.close()
        raise




# ------------------------------------------------------------------------------ simulations
class _SimCard:
    """Card c of an X2SimTransport: its register lines "@c ...", its channels the simulation's
    from nch spc c."""

    batched = True
    devname = None
    keeps_state = False

    def __init__(self, sim, c: int):
        self.sim, self.c = sim, c
        self.ch0 = c * sim.nch * sim.spc

    def mem_write(self, ch, off, data):
        self.sim.mem_write(self.ch0 + ch, off, data)

    def mem_read(self, ch, off, n, out=None):
        return self.sim.mem_read(self.ch0 + ch, off, n, out)

    def reg_write(self, off, val):
        self.sim.script.append(f"@{self.c} W {off:x} {val & 0xFFFFFFFF:x}")

    def persist(self, off, val):
        """A write at the start of every simulation (each one a fresh machine)."""
        self.sim.prologue.append(f"@{self.c} W {off:x} {val & 0xFFFFFFFF:x}")

    def persist_poll(self, off, mask, val):
        self.sim.prologue.append(f"@{self.c} P {off:x} {mask:x} {val:x}")

    def queue_read(self, off):
        self.sim.script.append(f"@{self.c} R {off:x}")
        self.sim.nreads += 1
        return self.sim.nreads - 1

    def poll(self, off, mask, val, timeout=0, expect=0):
        self.sim.script.append(f"@{self.c} P {off:x} {mask:x} {val:x}")
        return val

    def reg_read_many(self, offs):
        idx = [self.queue_read(o) for o in offs]
        self.flush()
        return [self.sim.results[i] for i in idx]

    def reg_read(self, off):
        return self.reg_read_many([off])[0]

    def wait_cycles(self, n):
        self.sim.wait_cycles(n)

    def flush(self):
        self.sim.flush()

    @property
    def results(self):
        return self.sim.results


class X2SimTransport(SimTransport):
    """The two-board co-simulation (bf16st's testbench `top`: each card the BCU1525's spc slices,
    tb_bcuwg's otpu_bcu_wg_sys with its link port, the cards' ports joined by the link model
    otpu_link_sim): one Verilator simulation of every card, as SimTransport's (a fresh machine
    every flush, the DRAM in the channel files). The host's script: each card's lines
    "@<card> W|R|P ...", "C n" the simulation's; the channel files ch<g>.bin (and _hi.txt) at
    g = nch spc card + nch local slice + c, the machine's global channels. `cards` are the cards'
    transports (CardsTransport's), their CARD_ID / NCARDS writes and the link's wait replayed at
    the start of every flush (`prologue`). target: the machine's (bcu1525-x2 or a variant), each
    card's build its configuration at spc slices (SimTransport's parameters, ARCH_HASH that
    configuration's), NCARDS the cards; link_lat the link's one-way core cycles (+link_lat:
    otpu_link_sim's L, the QSFP28's 40, docs/x2_link.md).
    tb_x2 models status / hello / INFO, but ping, CRC/frame and word/wait/error counters read
    zero. Engine completion uses the hubs' X2_SEQ / COLL_STATUS and port status, never ping
    or traffic counts as evidence of progress. Protocol counters are tested with otpu_link,
    not this cable stand-in. The source is sim/verilator/otpu_link_sim.sv."""

    def __init__(self, target="bcu1525-x2", ch_bytes: int = 1 << 24, top: str = "tb_x2",
                 sources: list | None = None, link_lat: int = 40, **kw):
        from opentpu import target as T
        t = T.load(target) if isinstance(target, str) else target
        self.machine_target = t
        self.ncards = T.cards(t)
        self.spc = t.arch["S"] // self.ncards
        card = replace(t, arch=dict(t.arch, S=self.spc))       # a card's build
        rp = T.rtl_params(card)
        # SimTransport's legacy defaults follow OTPU_TARGET; this explicit two-card target
        # must also build correctly when a caller has not selected it in the environment.
        p = {k: rp[k] for k in ("D", "MCOLS", "LANES", "ACT_ROWS", "ACT_BLOCKS", "TMEM_WORDS",
                               "IMEM_WORDS", "DSTEP", "SEQ_WC")}
        p |= {k: int(card.arch[k]) for k in ("WG", "OUTER", "DSTEP2", "KT", "ICACHE")}
        p["ST16"] = rp.get("ST16", 0)
        p.update(kw.pop("params", None) or {})
        plus = list(kw.pop("plusargs", None) or []) + [f"+link_lat={link_lat}"]
        super().__init__(ch_bytes=ch_bytes, target=card, bcuwg=True, plusargs=plus, params=p, **kw)
        nch = self.nch * self.spc * self.ncards
        self.ch = [np.zeros(len(self.ch[0]), np.uint8) for _ in range(nch)]
        self.hi = [{} for _ in range(nch)]
        self.top, self.sources = top, sources
        self.prologue: list[str] = []
        self.cards = [_SimCard(self, c) for c in range(self.ncards)]

    def sim_build(self) -> tuple[str, list, dict]:
        from opentpu import rtlsim
        top, srcs, p = super().sim_build()
        srcs = [s for s in srcs if not str(s).endswith(f"{top}.sv")]
        srcs += self.sources if self.sources is not None else [
            rtlsim.RTL / "link/otpu_link_pkg.sv", rtlsim.TB / "otpu_link_sim.sv",
            rtlsim.TB / f"{self.top}.sv"]
        return self.top, srcs, dict(p, NCARDS=self.ncards)

    def flush(self) -> None:
        if not self.script:
            return
        self.script = self.prologue + self.script
        super().flush()


class IsaCards:
    """The ISA simulator (opentpu.isasim.Machine at cfg.S: every card's slices, the collectives
    across them as on one machine) behind `ncards` fake cards' registers: the host's multi-card
    protocol end to end without RTL. `cards` are the cards' transports (IsaCard: FakeTransport's
    registers, each card's CFG_* at its own slices with the CFG_HASH of its configuration,
    CAPS2 XCARD, the link port's CSRs in the X2 window). The machine runs once every card has
    RUN set (at the last card's RUN), its slices in the order the cards' CARD_IDs give (card c's
    local slice k is slice spc CARD_ID + k: a host that writes the wrong CARD_ID runs its programs
    on the wrong slices), and only with every card's link up (else STATUS ERROR and COLL_STATUS
    link check).
    cross: the two cards' cables crossed (card c's port p on the other card's port 1 - p)."""

    def __init__(self, cfg, ncards: int = 2, platform: int | None = None, cross: bool = False):
        from opentpu import target as T
        if cfg.S % ncards:
            raise ValueError(f"S {cfg.S} over {ncards} cards")
        self.cfg, self.n, self.spc = cfg, ncards, cfg.S // ncards
        self.cross = cross
        self.platform = T.load("bcu1525-x2").platform_id if platform is None else platform
        self.dram = [np.zeros(cfg.DRAM_BYTES, np.uint8) for _ in range(cfg.S)]
        self.machine = None
        self.order = None                   # card slice -> machine slice (its first run's)
        self.icount = [0] * cfg.S           # by card slice
        self.runs = 0
        self.cards: list = []
        self.cards = [IsaCard(self, c) for c in range(ncards)]

    def started(self) -> None:
        """A card's RUN rose: the machine runs once every card's has."""
        if not all(k.running and not k.ran for k in self.cards):
            return
        from opentpu import isa as I
        from opentpu.isasim import Machine
        n, spc = self.n, self.spc
        ids = [k.topo & 0xFF for k in self.cards]
        if not all(k.up() for k in self.cards) or sorted(ids) != list(range(n)):
            for k in self.cards:
                k.ran, k.error = True, True
                k.coll_status |= R.COLL_ABORTED | R.COLL_LINK
            return
        order = [ids[c] * spc + j for c in range(n) for j in range(spc)]
        if self.order is not None and order != self.order:
            raise RuntimeError("IsaCards: the cards' CARD_IDs changed between runs")
        progs = [None] * self.cfg.S
        for c, k in enumerate(self.cards):
            a, m = k.regs[R.R_PROG_ADDR], k.regs[R.R_PROG_N]
            for j in range(spc):
                w = self.dram[c * spc + j][a:a + 32 * m].view("<u4")
                progs[order[c * spc + j]] = [I.Instr.decode(w[8 * i:8 * i + 8])
                                             for i in range(m)]
        k0 = self.cards[0]
        args = [k0.regs.get(R.R_ARG0 + 4 * i, 0) for i in range(8)]
        if self.machine is None:
            self.order = order
            drams = [None] * self.cfg.S
            for p, g in enumerate(order):
                drams[g] = self.dram[p]
            self.machine = Machine(self.cfg, progs, drams, args)
            for p, g in enumerate(order):           # (the machine's copies from here on)
                self.dram[p] = self.machine.slices[g].dram
        else:
            self.machine.load(progs, args)
        before = self.machine.collectives
        self.machine.run()
        issued = (self.machine.collectives - before) & 0xFFFF
        for p, g in enumerate(order):
            self.icount[p] = self.machine.slices[g].icount
        self.runs += 1
        for k in self.cards:
            k.ran, k.error = True, False
            k.seq = (k.seq + issued) & 0xFFFF


class IsaCard(FakeTransport):
    """IsaCards' card c: a fake card's registers with the shared machine's DRAM behind its
    channels (its slices', in the bitstream's map: CHASH), its slices' ICOUNT, its link port's
    X2 CSRs."""

    def __init__(self, sh: IsaCards, c: int):
        from opentpu.target import arch_hash
        from .board import _reg_config
        cfg, spc = sh.cfg, sh.spc
        nch, beat = (8, 128) if cfg.WG > 1 else (2, 64)
        words = dict(S=spc, TMEM_WORDS=cfg.TMEM_WORDS, IMEM_WORDS=cfg.IMEM_WORDS,
                     ACT_BLOCKS=cfg.ACT_BLOCKS, DRAM_BYTES=cfg.DRAM_BYTES, channels=nch,
                     channel_bytes=cfg.DRAM_BYTES // nch, beat_bytes=beat, MXU_IMPL=2,
                     TMEM_IMPL=3, VPU_CL=2, ULANES=8, SEQ_WC=int(cfg.SEQ_WC), WG=cfg.WG,
                     ICACHE=int(cfg.ICACHE), platform=sh.platform, hash=0)
        self.sh, self.c, self.spc, self.mcfg = sh, c, spc, cfg
        self.running, self.ran, self.error = False, False, False
        self.topo, self.enabled = 1 << 8, False     # (CARD_ID 0 of 1 at reset)
        self.seq, self.coll_status, self.watchdog = 0, 0, R.COLL_WD_DEFAULT
        self.port_errors = [0, 0]
        self.crc_errors = [0, 0]
        super().__init__(ch_bytes=64, regmap=5, devname=None, D=cfg.D, MCOLS=cfg.MCOLS,
                         LANES=cfg.LANES, pair=cfg.PAIR, dstep=cfg.DSTEP, args=True,
                         stream=cfg.STREAM, gen=True, chash=cfg.CHASH, cfg=words,
                         act_rows=cfg.act_rows, seq_wc=cfg.SEQ_WC, outer=cfg.OUTER,
                         dstep2=cfg.DSTEP2, se_own=cfg.SE_OWN, kt=cfg.KT)
        self.ch = []                            # (the machine's DRAM behind the channels)
        # the CFG_HASH its registers report (check_hash's), at the machine's AW and window
        self.cfg_words[-1] = arch_hash(_reg_config(card_info(self), cfg.AW, cfg.WC_WIN))

    # ---- the shared machine's DRAM behind the channels
    def _idx(self, ch: int, off: int, n: int) -> tuple:
        """(card slice, the logical byte of each of channel ch's bytes off .. off + n)."""
        s, cc = divmod(ch, self.nch)
        j = np.arange(off, off + n, dtype=np.int64)
        m = j // self.beat                              # its wide chunk
        h = 0
        if self.chash and n:
            h = chunk_hash(int(m[0]) * self.nch * self.beat, int(m[-1] - m[0]) + 1, self.nch,
                           self.beat)[m - m[0]]
        return self.c * self.spc + s, m * self.nch * self.beat + (cc ^ h) * self.beat \
            + j % self.beat

    def mem_write(self, ch, off, data):
        s, ix = self._idx(ch, off, len(data))
        self.sh.dram[s][ix] = np.asarray(data, np.uint8).reshape(-1)

    def mem_read(self, ch, off, n, out=None):
        s, ix = self._idx(ch, off, n)
        r = self.sh.dram[s][ix]
        if out is None:
            return r
        out[:] = r
        return out

    # ---- the link port
    def up(self) -> bool:
        ks, n = self.sh.cards, self.topo >> 8 & 0xFF
        return self.enabled and (n == 1 or (
            n == self.sh.n and len({k.topo & 0xFF for k in ks}) == n
            and all(k.enabled and k.topo >> 8 & 0xFF == n for k in ks)))

    def peer(self, p: int) -> int:
        sh = self.sh
        if sh.n == 1 or not self.enabled:
            return 0
        if sh.n == 2:
            d, q = 1 - self.c, (1 - p if sh.cross else p)
        else:
            d, q = ((self.c + 1) % sh.n, 1) if p == 0 else ((self.c - 1) % sh.n, 0)
        k = sh.cards[d]
        if not k.enabled:
            return 0
        return R.X2_HELLO_MAGIC << 24 | (k.topo & 0xFF) | q << 8 | (k.topo >> 8 & 0xFF) << 12

    def _link_read(self, off: int) -> int:
        if off == R.R_X2_CARD:
            return self.topo
        if off == R.R_X2_INFO:
            return 1 | 32 << 8 | 2048 << 16
        if off == R.R_X2_SEQ:
            return self.seq
        if off == R.R_COLL_STATUS:
            return self.coll_status | (3 << 5 if self.up() else 0)
        if off == R.R_COLL_WD:
            return self.watchdog
        if off in R.R_X2_STATUS:
            p = R.R_X2_STATUS.index(off)
            return (R.X2_CHANNEL | R.X2_SEEN | R.X2_LANES | R.X2_UP | 2048 << 16
                    if self.up() else 0) | self.port_errors[p]
        if off in R.R_X2_HELLO:
            return self.peer(R.R_X2_HELLO.index(off))
        if off in R.R_X2_CRC:
            return self.crc_errors[R.R_X2_CRC.index(off)]
        return 0

    def reg_write(self, off, val):
        if 0x300 <= off < 0x400:
            if off == R.R_X2_CARD:
                self.topo = val & 0xFFFF
                self.enabled = False
            elif off == R.R_COLL_WD:
                self.watchdog = val & 0xFFFFFFFF
            elif off == R.R_X2_CTRL:
                if val & R.X2_LRST:
                    if self.running:
                        raise RuntimeError("IsaCards: LRST while RUN set")
                    self.enabled, self.error, self.seq, self.coll_status = True, False, 0, 0
                    if self.sh.machine is not None:
                        self.sh.machine.reset_link()
                    self.port_errors, self.crc_errors = [0, 0], [0, 0]
                elif val & R.X2_CLEAR:
                    self.port_errors = [0, 0]
            return
        run0 = self.regs[R.R_CTRL] & R.CTRL_RUN
        super().reg_write(off, val)
        if off == R.R_CTRL:
            run = bool(self.regs[R.R_CTRL] & R.CTRL_RUN)
            if run and not run0:
                self.running, self.ran = True, False
                self.sh.started()
            elif not run:
                self.running = self.ran = False

    def reg_read(self, off):
        if 0x300 <= off < 0x400:
            return self._link_read(off)
        if off == R.R_STATUS:
            run = bool(self.regs[R.R_CTRL] & R.CTRL_RUN)
            return (R.ST_HALTED if run and self.ran else 0) | R.ST_WR_IDLE | R.ST_CALIB0 \
                | R.ST_CALIB1 | (R.ST_RUN if run else 0) \
                | (R.ST_ERROR if run and self.error else 0)
        if off == R.R_ICOUNT or R.R_ICOUNT_S + 4 <= off < R.R_ICOUNT_S + 4 * self.spc:
            k = 0 if off == R.R_ICOUNT else (off - R.R_ICOUNT_S) // 4
            return self.sh.icount[self.c * self.spc + k] if self.sh.runs else 0
        if off in (R.R_CYCLES, R.R_CYCLES_HI):
            return 0
        if off == R.R_CAPS2:
            from opentpu.target import ST16_CODES
            cfg = self.mcfg
            return (R.CAP2_GATHER | (0 if cfg.OUTER else R.CAP2_NO_OUTER)
                    | (R.CAP2_DSTEP2 if cfg.DSTEP2 else 0)
                    | (R.CAP2_SE_OWN if cfg.SE_OWN else 0) | (R.CAP2_KT if cfg.KT else 0)
                    | cfg.BHW << R.CAP2_BHW_SHIFT | (R.CAP2_ICACHE if cfg.ICACHE else 0)
                    | (ST16_CODES[cfg.ST16] << R.CAP2_ST16_SHIFT if cfg.ST16 else 0)
                    | (R.CAP2_FSUM if cfg.FSUM else 0)
                    | (R.CAP2_SPLITQ if cfg.SPLITQ else 0)
                    | (R.CAP2_ST16S if cfg.ST16S and cfg.ST16 else 0)
                    | (R.CAP2_QHL if cfg.QHL and cfg.SPLITQ else 0)
                    | (R.CAP2_QSTHL if cfg.QSTHL and cfg.SPLITQ else 0)
                    | (R.CAP2_VSIG if cfg.VSIG else 0)
                    | (R.CAP2_VMAC if cfg.VMAC else 0)
                    | (R.CAP2_OUTER1 if cfg.OUTER1 and not cfg.OUTER else 0)
                    | (R.CAP2_QONCE if cfg.QONCE else 0)
                    | (R.CAP2_SPLMM if cfg.SPLMM else 0)
                    | (R.CAP2_XCARD if self.sh.n > 1 else 0))
        return super().reg_read(off)
