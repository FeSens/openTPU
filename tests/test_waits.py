"""opentpu.waits: the compiler's wait entries for otpu_seq WAITC (docs/isa.md "Wait counts")."""
from pathlib import Path

import numpy as np
import pytest

from opentpu import isa as I
from opentpu import waits as W

D = 128


def k(words, pc):
    return W.counts(words[pc])


def vm(words, pc):
    return W.vmask(words[pc])


def vop(func, dst, a, b, rows=1, cols=8, **kw):
    return I.vop(func, dst, a, b, rows, cols, cols, cols, cols, **kw)


def test_raw_across_units_and_none_without_overlap():
    prog = [I.ld(0, 100, 8), vop(I.V_ADD, 200, 100, 100), vop(I.V_ADD, 300, 400, 400),
            I.halt()]
    w = W.wait_counts(prog, args=[0] * 8)
    assert k(w, 1)[W.U_DMA] == 1                    # reads what the LD wrote
    assert k(w, 2)[W.U_DMA] == 0 and vm(w, 2) == 0
    assert k(w, 0) == [0] * 7 and vm(w, 0) == 0
    assert all(k(w, pc)[W.U_VPU] == 0 for pc in range(3))   # the VPU's is the mask


def test_war_across_units():
    """A younger writer on another unit waits for the reader (one-copy TMEM replay, RPL)."""
    prog = [vop(I.V_ADD, 200, 100, 100), I.ld(0, 100, 8), I.halt()]
    w = W.wait_counts(prog, args=[0] * 8)
    assert vm(w, 1) == 0b1


def test_waw_and_rank():
    prog = [vop(I.V_ADD, 200, 100, 100), vop(I.V_ADD, 300, 100, 100),
            vop(I.V_ADD, 400, 100, 100), I.ld(0, 200, 8), I.halt()]
    w = W.wait_counts(prog, args=[0] * 8)
    assert vm(w, 3) == 0b100                        # the 3rd last VPU instruction wrote 200


def test_the_vpu_mask_names_each_producer():
    """The VPU completes out of order: a consumer waits for its producers, not for the VPU
    instructions between them."""
    prog = [vop(I.V_ADD, 100, 500, 500), vop(I.V_ADD, 300, 500, 500),
            vop(I.V_ADD, 200, 500, 500), vop(I.V_ADD, 400, 500, 500),
            I.st(64, 100, 8), vop(I.V_ADD, 600, 100, 200), I.halt()]
    w = W.wait_counts(prog, args=[0] * 8)
    assert vm(w, 4) == 0b1000                       # the ST reads 100: 4 back
    assert vm(w, 5) == 0b1010                       # 100 (4 back) and 200 (2 back)


def test_own_unit_waits_only_for_the_vpu():
    prog = [I.ld(0, 100, 8), I.st(64, 100, 8), vop(I.V_ADD, 200, 100, 100),
            vop(I.V_ADD, 300, 200, 200), I.halt()]
    w = W.wait_counts(prog, args=[0] * 8)
    assert k(w, 1)[W.U_DMA] == 0                    # the DMA orders its own
    assert vm(w, 3) == 0b1                          # the VPU does not


def test_loop_carried_and_registers():
    # each iteration reads the tile the last one's LD wrote: R1 advances by 8
    prog = [I.li(1, 0), I.loop(3, 4),
            I.ld(0, 1000, 8, rb=1),
            vop(I.V_ADD, 2000, 1000, 1000, ra=0, rb=1, rc=1),
            I.addi(1, 1, 8),
            I.halt()]
    w = W.wait_counts(prog, args=[0] * 8)
    assert k(w, 3)[W.U_DMA] == 1
    # the LD of iteration t+1 writes 1008 + 8t, which the VOP of iteration t does not read:
    # no WAR; but the VOPs all write 2000 (WAW with each other: the 4 iterations' VOPs, each
    # with the ones before it in the mask)
    assert vm(w, 2) == 0
    assert vm(w, 3) == 0b111


def test_unknown_count_and_argument_are_conservative():
    prog = [I.loop(2, 0, rcount=8),
            I.ld(0, 1000, 8, rb=9),
            vop(I.V_ADD, 3000, 2000, 2000),
            I.ld(0, 2000, 8),
            I.halt()]
    w = W.wait_counts(prog)                          # R8, R9 unknown
    assert vm(w, 3) == W.VMASK                       # up to 15 iterations ran: WAR on 2000
    assert k(w, 2)[W.U_DMA] == 1                     # R9 unknown: the LD may hit 2000
    st = {}
    W.wait_counts(prog, stats=st)
    assert st["widened"]


def test_a_run_of_zero_or_one_loops_keeps_the_footprints_apart():
    """16 loops of 0 or 1 iterations (attention's sub-block variants: RLD, LOOP) each writing
    two tiles: the joins gather a range per path in each history slot (up to 17), which stay
    apart (MAXR), so an RLD of a word between the tiles waits for nothing."""
    prog, n = [], 0
    for j in range(16):
        prog += [I.rld(1, 9000 + j, bound=(0, 1)), I.loop(2, 0, rcount=1)]
        for _ in range(2):
            prog.append(vop(I.V_ADD, 7000 + 16 * n, 5000, 5000))
            n += 1
    prog += [I.rld(2, 7008 + 16 * 5), I.halt()]
    w = W.wait_counts(prog, args=[0] * 8)
    assert k(w, len(prog) - 2) == [0] * 7 and vm(w, len(prog) - 2) == 0
    prog[-2] = I.rld(2, 7000 + 16 * 5)              # a word the 6th tile writes
    w = W.wait_counts(prog, args=[0] * 8)
    assert vm(w, len(prog) - 2) != 0


def test_variant_paths_take_one_variant_a_table():
    """isa.rld's variant tags: a path per combination of the tables' variants (2 x 3), each with
    exactly one count 1 a table, every read of a word the same; a word read with another K,
    past PATHS combinations or without tags, one pass on the bounds."""
    prog = [I.rld(1, 100 + j, bound=(0, 1), variant=(j, 2)) for j in range(2)]
    prog += [I.rld(1, 200 + j, bound=(0, 1), variant=(j, 3)) for j in range(3)]
    prog += [I.rld(1, 101, bound=(0, 1), variant=(1, 2)), I.halt()]
    ps = W.variant_paths(prog)
    assert len({tuple(sorted(m.items())) for m in ps}) == len(ps) == 6
    for m in ps:
        assert all(m[pc] in ((0, 0), (1, 1)) for pc in range(6)) and m[5] == m[1]
        assert m[0][0] + m[1][0] == 1 and m[2][0] + m[3][0] + m[4][0] == 1
    assert W.variant_paths(prog[:5] + [I.rld(1, 102, bound=(0, 1), variant=(2, 3)),
                                       I.halt()]) == [None]
    many = [I.rld(1, 300 + 2 * t + j, bound=(0, 1), variant=(j, 2)) for t in range(5)
            for j in range(2)]
    assert W.variant_paths(many + [I.halt()]) == [None]          # 2^5 > PATHS
    assert W.variant_paths([I.rld(1, 100, bound=(0, 1)), I.halt()]) == [None]
    with pytest.raises(Exception):
        I.rld(1, 100, variant=(0, 2))                            # a variant's bound is (0, 1)


def test_variant_loops_wait_only_on_the_paths_a_run_takes():
    """Both variants of a table (loops of 0 or 1 iterations, a VOP each) between a tile's write
    and its read: a run takes one, so the write is always the read's VOP 2 back; with the counts
    apart (rld {}: their bounds) the joins also put it 1 and 3 back, and the mask named those."""
    prog = [vop(I.V_ADD, 1000, 5000, 5000)]
    for j in range(2):
        prog += [I.rld(1, 9000 + j, bound=(0, 1), variant=(j, 2)), I.loop(1, 0, rcount=1),
                 vop(I.V_ADD, 2000 + 16 * j, 5000, 5000)]
    prog += [vop(I.V_ADD, 3000, 1000, 1000), I.halt()]
    pc = len(prog) - 2
    assert vm(W.wait_counts(prog, args=[0] * 8), pc) == 0b10
    assert bin(vm(W.wait_counts(prog, args=[0] * 8, rld={}), pc)).count("1") == 3


def test_mm_stream_start_counts_dram_only():
    st_ = I.st(0x1000, 0, 32)
    mm = I.mm(0x1000, 0x8000, 500, 1, 1, 0, 1, 1, 0, 0)
    prog = [st_, I.qact(600, 1, 0, 1, 0), mm, I.halt()]
    w = W.wait_counts(prog, args=[0] * 8)
    c = k(w, 2)
    assert c[5] == 1 and c[W.U_DMA] == 1             # the weights the ST wrote
    assert c[W.U_Q] == 1 and c[6] == 0               # the ACT RAM: release only


def test_bar_waits_for_everything():
    prog = [I.ld(0, 0, 8), vop(I.V_ADD, 200, 100, 100), I.bar(), I.halt()]
    w = W.wait_counts(prog, args=[0] * 8)
    assert k(w, 2)[W.U_DMA] == 1 and vm(w, 2) == 0b1


def test_rank_beyond_the_window_needs_no_count():
    prog = [I.ld(0, 100, 8)] + [vop(I.V_ADD, 300 + 8 * i, 400, 400) for i in range(16)] + \
        [vop(I.V_ADD, 200, 100, 100), I.halt()]
    w = W.wait_counts(prog, args=[0] * 8)
    assert k(w, 17)[W.U_DMA] == 1                    # the DMA's last: rank 1
    prog = [I.ld(0, 100, 8)] + [I.ld(64, 500 + 8 * i, 8) for i in range(15)] + \
        [vop(I.V_ADD, 200, 100, 100), I.halt()]
    w = W.wait_counts(prog, args=[0] * 8)
    assert k(w, 16)[W.U_DMA] == 0                    # 16th last DMA instruction: done
    # the VPU starts out of order, and the sequencer dispatches a VPU instruction once the one
    # 31 (VMAX) before it has completed: the 31st last may run, the 32nd last has completed
    for n, want in ((W.VMAX - 1, 1 << (W.VMAX - 1)), (W.VMAX, 0)):
        prog = [vop(I.V_ADD, 100, 400, 400)] + [vop(I.V_ADD, 500 + 8 * i, 400, 400)
                                               for i in range(n)] + [I.ld(0, 100, 8), I.halt()]
        w = W.wait_counts(prog, args=[0] * 8)
        assert vm(w, n + 1) == want


def _dma_run(n):
    """An LD, n LDs elsewhere, then a VOP reading what the first wrote (pc n + 1): its producer is
    the DMA's (n + 1)-th last."""
    return [I.ld(0, 100, 8)] + [I.ld(64, 500 + 8 * i, 8) for i in range(n)] + \
        [vop(I.V_ADD, 200, 100, 100), I.halt()]


def test_a_window_of_32_saturates_the_counts():
    """At a window of 32 (Config.WC_WIN) an in-order unit's producer 16 to 31 back may still run:
    its count saturates at 15 (the unit's 15th last done, every older one is), 32 back it has
    completed. At 16 (the default) 16 back has completed."""
    for n, want in ((14, 15), (15, 15), (20, 15), (30, 15), (31, 0)):
        prog = _dma_run(n)
        assert k(W.wait_counts(prog, args=[0] * 8, win=32), n + 1)[W.U_DMA] == want, n
        assert k(W.wait_counts(prog, args=[0] * 8), n + 1)[W.U_DMA] == (want if n < 15 else 0)
    for n, want in ((W.VMAX - 1, 1 << (W.VMAX - 1)), (W.VMAX, 0)):   # the VPU's mask: the same
        prog = [vop(I.V_ADD, 100, 400, 400)] + [vop(I.V_ADD, 500 + 8 * i, 400, 400)
                                               for i in range(n)] + [I.ld(0, 100, 8), I.halt()]
        assert vm(W.wait_counts(prog, args=[0] * 8, win=32), n + 1) == want
    with pytest.raises(ValueError, match="window of 24"):
        W.wait_counts(_dma_run(1), win=24)


def test_the_checker_at_32_needs_the_saturated_count():
    """The checker at a window of 32 takes the saturated count for a producer 21 back, and finds
    the window-16 entries' none (that producer may still run)."""
    from opentpu.isasim import Machine, board_config
    prog = _dma_run(20)
    cfg = board_config(S=1, DRAM_BYTES=1 << 16, TMEM_WORDS=1 << 12, WC_WIN=32)
    assert W.check_isasim(Machine(cfg, [prog], [None]), [W.wait_counts(prog, win=32)]) > 0
    with pytest.raises(AssertionError, match="k=0 misses the 21-th last DMA"):
        W.check_isasim(Machine(cfg, [prog], [None]), [W.wait_counts(prog)])


def test_a_window_of_16_keeps_the_entries_and_their_key():
    """win=16 is the entries before the field: the same words and the same cache key."""
    rng = np.random.default_rng(3)
    prog = _rich_prog(rng, 20)
    assert np.array_equal(W.wait_counts(prog, win=16), W.wait_counts(prog))
    w = np.arange(64, dtype=np.uint32)
    assert W.entries_key(w, 128, 1, win=16) == W.entries_key(w, 128, 1)
    assert W.entries_key(w, 128, 1, win=32) != W.entries_key(w, 128, 1)


def _digest(arrs) -> str:
    import hashlib
    h = hashlib.blake2b(digest_size=8)
    for a in arrs:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def test_a_window_of_16_is_the_entries_before_the_field():
    """At a window of 16 (bcu1525-wg's: the production card) the entries of random programs, and
    the words and entries of the tiny Qwen3 / Qwen3.5 decode programs on bcu1525-wg, are phase-a
    562074ad's (before Config.WC_WIN) bit for bit: digests taken there. Re-take them when the
    compiler or this pass changes a program on purpose."""
    pytest.importorskip("transformers")
    from opentpu import target as T
    from test_bcuwg_board import _tiny
    ent = [W.wait_counts(_random_prog(np.random.default_rng(seed), 24)) for seed in range(40)]
    for seed in range(60):
        p = _rich_prog(np.random.default_rng(seed), 20)
        ent += [W.wait_counts(p), W.wait_counts(p, args=[3] + [0] * 7)]
    assert _digest(ent) == "9721bc3ab3c1ab5e"
    cfg = T.config("bcu1525-wg", DRAM_BYTES=1 << 34)
    assert cfg.WC_WIN == 16
    for model, wf, want in (("qwen3", "int8", ("669c343d3688dde7", "798a2637efd5a54d")),
                            ("qwen35", "fp4", ("82486baeeff4b129", "1efe8c6a127ebd7e"))):
        _, spec = _tiny(model)
        im = spec.image(cfg, 256, wformat=wf)
        words, ents = [], []
        for pos in (5, 100):
            prog = im.compile_step(pos)[0]
            ra = getattr(prog, "run_args", None)
            words.append(np.asarray(I.assemble(prog), "<u4"))
            ents.append(W.wait_counts(prog, cfg.D, cfg.S, run_args=ra if ra is not None else (),
                                      aw=cfg.AW, win=cfg.WC_WIN))
        assert (_digest(words), _digest(ents)) == want, model


def _random_prog(rng, n):
    prog = [I.li(1, int(rng.integers(0, 4)) * 8)]
    body = []
    for _ in range(n):
        t = lambda: int(rng.integers(0, 12)) * 8      # noqa: E731
        c = rng.integers(0, 5)
        if c == 0:
            body.append(I.ld(int(rng.integers(0, 4)) * 64, t(), 8, rb=int(rng.integers(0, 2))))
        elif c == 1:
            body.append(I.st(int(rng.integers(0, 4)) * 64, t(), 8))
        elif c == 2:
            body.append(vop(I.V_ADD, t(), t(), t(), rb=int(rng.integers(0, 2))))
        elif c == 3:
            body.append(vop(I.V_RSUM, t(), t(), 0))
        else:
            body.append(I.addi(1, 1, 8))
    cut = int(rng.integers(1, len(body)))
    return prog + [I.loop(cut, int(rng.integers(0, 4)), rcount=8)] + body + [I.halt()]


@pytest.mark.parametrize("win", W.WINS)
@pytest.mark.parametrize("seed", range(40))
def test_counts_cover_every_conflict_of_a_run(seed, win):
    """The counts computed without the run's argument (a loop count) cover each concrete run, at
    either window."""
    from opentpu.isasim import Machine, board_config
    rng = np.random.default_rng(seed)
    prog = _random_prog(rng, 24)
    words = W.wait_counts(prog, win=win)
    cfg = board_config(S=1, DRAM_BYTES=1 << 16, TMEM_WORDS=1 << 12, WC_WIN=win)
    for n in range(0, 6):
        m = Machine(cfg, [prog], [None], args=[n])
        assert W.check_isasim(m, [words]) > 0


def _rich_prog(rng, n):
    """A random program over every unit: LD / ST, VOPs, MMs (weights and scales in DRAM, the
    ACT RAM's blocks, a TMEM tile), QACTs, RLDs (bounded or not), BARs, in two nested loops
    whose counts are known or the run's argument, registers advancing."""
    def t():
        return int(rng.integers(0, 12)) * 8

    def r():
        return int(rng.integers(0, 3))

    def one():
        c = rng.integers(0, 10)
        if c == 0:
            return I.ld(int(rng.integers(0, 6)) * 64, t(), 8, ra=r(), rb=r())
        if c == 1:
            return I.st(int(rng.integers(0, 6)) * 64, t(), 8, ra=r())
        if c in (2, 3):
            return vop(I.V_ADD, t(), t(), t(), ra=r(), rb=r())
        if c == 4:
            return vop(I.V_RSUM, t(), t(), 0)
        if c == 5:
            return I.mm(int(rng.integers(0, 6)) * 64, 0x4000 + int(rng.integers(0, 4)) * 64,
                        t(), 1, 1, 0, 1, 1, int(rng.integers(0, 3)), 0, ra=r(), rc=r())
        if c == 6:
            return I.qact(t(), 1, int(rng.integers(0, 3)), 1, 0, ra=r())
        if c == 7:
            return I.rld(2, t(), bound=(0, 64) if rng.integers(0, 2) else None)
        if c == 8:
            return I.addi(1, 1, 8)
        return I.bar() if rng.integers(0, 4) == 0 else I.addi(2, 2, 64)

    inner = [one() for _ in range(n // 2)]
    outer = [one() for _ in range(n // 2)]
    body = outer[:2] + [I.loop(len(inner), int(rng.integers(1, 5)))] + inner + outer[2:]
    head = [I.li(1, int(rng.integers(0, 4)) * 8), I.li(2, 0)]
    cnt = (I.loop(len(body), 0, rcount=8) if rng.integers(0, 2) else
           I.loop(len(body), int(rng.integers(1, 6))))
    return head + [cnt] + body + [I.halt()]


@pytest.mark.parametrize("win", W.WINS)
@pytest.mark.parametrize("seed", range(60))
def test_the_memoized_pass_is_the_exact_one(seed, win):
    """_MemoPass (wait_counts' default: the local footprints' checks kept, the DRAM ranges
    checked apart) gives _Pass's entries on random programs over every unit, the run's argument
    known or not, at either window."""
    rng = np.random.default_rng(seed)
    prog = _rich_prog(rng, 20)
    for args in (None, [3] + [0] * 7):
        a = W.wait_counts(prog, args=args, memo=False, win=win)
        b = W.wait_counts(prog, args=args, win=win)
        assert np.array_equal(a, b), seed


def test_checker_finds_a_missing_wait():
    from opentpu.isasim import Machine, board_config
    prog = [I.ld(0, 100, 8), vop(I.V_ADD, 200, 100, 100), vop(I.V_ADD, 300, 500, 500),
            vop(I.V_ADD, 400, 200, 200), I.halt()]
    cfg = board_config(S=1, DRAM_BYTES=1 << 16, TMEM_WORDS=1 << 12)
    words = W.wait_counts(prog, args=[0] * 8)
    assert W.check_isasim(Machine(cfg, [prog], [None], args=[0]), [words]) == 4
    w1 = words.copy()
    w1[1] = 0
    with pytest.raises(AssertionError, match="misses the 1-th last DMA"):
        W.check_isasim(Machine(cfg, [prog], [None], args=[0]), [w1])
    w3 = words.copy()
    assert vm(w3, 3) == 0b10
    w3[3] = W.counts(0)[0]                          # no mask: the VOP 2 back is missed
    with pytest.raises(AssertionError, match="misses the 2-th last VPU"):
        W.check_isasim(Machine(cfg, [prog], [None], args=[0]), [w3])


def _follow_prog(S, C=256):
    """A residual's ADD, the all-gather of its row (one run: seg = cols) and reductions of what it
    writes; then a GATHER of two rows of their results (after them: a GATHER's writes take every
    bank, and a follower's own write would wait for them)."""
    return [I.ld(0, 0, C), vop(I.V_ADD, 1000, 0, 0, cols=C),
            I.gather(1000, 4096, 1, C, C, C, C),
            vop(I.V_RSSQ, 9000, 4096, 0, cols=S * C),        # the norm's sum of squares
            vop(I.V_RSUM, 9001, 4096 + 64, 0, cols=2 * C),    # from within it
            vop(I.V_ADD, 9100, 4096, 4096, cols=64),          # not a reduction of A alone
            vop(I.V_RSSQ, 4100, 4096, 0, cols=64),           # into the GATHER's dst
            I.gather(9000, 12000, 2, C // 2, C // 2, C // 2, C),
            vop(I.V_RSSQ, 9002, 12000, 0, cols=S * C),
            I.st(1 << 16, 9000, 8), I.halt()]


def test_a_reduction_follows_the_gather_it_reads():
    """FOLLOW (docs/isa.md "Wait counts"): a one-row reduction of A alone whose only conflict
    with the last collective instruction, a GATHER of one run, is reading what it writes."""
    S = 4
    w = W.wait_counts(_follow_prog(S), S=S, args=[0] * 8)
    assert [pc for pc in range(len(w)) if W.follow(w[pc])] == [3, 4]
    assert k(w, 3)[W.U_COLL] == 1 and k(w, 4)[W.U_COLL] == 1 and k(w, 8)[W.U_COLL] == 1
    assert W.counts(w[3]) == W.counts(int(w[3]) & ~W.FOLLOW)      # (beside the counts)
    # a BAR after the GATHER: the last collective instruction conflicts with everything
    p = _follow_prog(S)
    w = W.wait_counts(p[:3] + [I.bar(), p[3], I.halt()], S=S, args=[0] * 8)
    assert not W.follow(w[4]) and k(w, 4)[W.U_COLL] == 1
    # its dst through a register: followed; a dst the pass does not know may be under the
    # follower's own output (a write after a write): not
    w = W.wait_counts(p[:2] + [I.li(6, 96), I.gather(1000, 4000, 1, 256, 256, 256, 256, rb=6),
                               p[3], I.halt()], S=S, args=[0] * 8)
    assert W.follow(w[4])
    w = W.wait_counts(p[:2] + [I.gather(1000, 4096, 1, 256, 256, 256, 256, rb=9), p[3],
                               I.halt()], S=S)
    assert not W.follow(w[3]) and k(w, 3)[W.U_COLL] == 1


def test_the_checker_refuses_a_wrong_follow():
    from opentpu.isasim import Machine, board_config
    S = 2
    prog = _follow_prog(S)
    cfg = board_config(S=S, DRAM_BYTES=1 << 18, TMEM_WORDS=1 << 15)
    words = W.wait_counts(prog, S=S, args=[0] * 8)
    assert W.check_isasim(Machine(cfg, [prog] * S, [None] * S, args=[0]), [words] * S) > 0
    for pc in (5, 6, 8):                            # an ADD, a write into dst, after two rows
        bad = words.copy()
        bad[pc] |= np.uint64(W.FOLLOW)
        with pytest.raises(AssertionError, match=f"pc {pc} .*may not follow"):
            W.check_isasim(Machine(cfg, [prog] * S, [None] * S, args=[0]), [bad] * S)


@pytest.mark.parametrize("S,mode", [(2, "sim"), (4, "board"), (4, "wg"), (4, "wg_se1")])
def test_waitc_rtl_follow(have_verilator, monkeypatch, S, mode):
    """FOLLOW on the RTL (otpu_seq, otpu_slice and otpu_vpu with wait entries at S > 1): the
    reductions read behind their GATHER's writes, at the simulator's TMEM (the VPU's enable from
    the grant), the board's (its write buffer) and the WG targets' one copy with the stream engine in
    the VPU (SE's enable a cycle ahead) or (wg_se1) the streams on an engine of their own (SE_R
    1: the VPU's engine without its stream half, SE_SS 0) and no VOP OUTER. The state is the ISA
    simulator's, the VPU's own check (every read taken below the writes granted two cycles
    before) holds, and the run is shorter than with FOLLOW cleared, by about the sum of squares'
    reads (S x 512 / 8 cycles)."""
    from opentpu import Config, rtlsim
    from opentpu import target as T
    from opentpu.isasim import Machine
    monkeypatch.setenv("OTPU_WAITS", "1")
    wg = {k: v for k, v in T.sim_uarch("bcu1525-x2").items()
          if k.startswith("TMEM_") or k in ("RPB", "WPB")}
    ua = {"sim": {}, "board": dict(rtlsim.BOARD_UARCH), "wg": wg, "wg_se1": wg | {"SE_R": 1}}[mode]
    ua = {**ua, "SEQ_WC": 1, "WIN": 16}
    cfg = Config(S=S, D=128, MCOLS=8, DRAM_BYTES=1 << 20, DSTEP=mode.startswith("wg"),
                 OUTER=mode != "wg_se1")
    C = 512
    prog = _follow_prog(S, C)
    rng = np.random.default_rng(S)
    imgs = []
    for _ in range(S):
        img = np.zeros(cfg.DRAM_BYTES, np.uint8)
        img[:4 * C] = rng.uniform(-1, 1, C).astype(np.float32).view(np.uint8)
        imgs.append(img)
    m = Machine(cfg, [prog] * S, [i.copy() for i in imgs]).run()
    orig = W.wait_counts
    cyc = {}
    for fol in (True, False):
        if not fol:
            monkeypatch.setattr(W, "wait_counts",
                                lambda *a, **kw: orig(*a, **kw) & ~np.uint64(W.FOLLOW))
        assert sum(W.follow(x) for x in W.wait_counts(prog, cfg.D, S)) == 2 * fol
        drams, tmems, st = rtlsim.run(cfg, [prog] * S, [i.copy() for i in imgs], uarch=ua)
        for s in range(S):
            assert np.array_equal(tmems[s], m.slices[s].tmem), (fol, s)
            assert np.array_equal(drams[s], m.slices[s].dram), (fol, s)
        cyc[fol] = st["cycles"]
    assert cyc[False] - cyc[True] > S * C // 8 // 2, cyc


@pytest.mark.parametrize("S,boot", [(1, False), (2, True)])
def test_waitc_rtl_matches_isa(have_verilator, monkeypatch, S, boot):
    """otpu_seq WAITC on the compiler's counts: the RTL's state equals the ISA simulator's (boot:
    the loader reads the wait words from DRAM after the program; else +hz_sync_s<k>), and the
    RTL's own check (every conflict on the window waited for) holds."""
    from conftest import assert_same_state, rel
    from opentpu import Config, rtlsim
    from opentpu.kernels import attention_decode
    from opentpu.runtime import compile_kernel, run
    from test_kernels import attn_args
    monkeypatch.setitem(rtlsim.UARCH, "SEQ_WC", 1)
    monkeypatch.setitem(rtlsim.UARCH, "WIN", 16)          # the card's window (WAITC: 16)
    monkeypatch.setitem(rtlsim.MEMORY, "BOOT", boot)
    monkeypatch.setenv("OTPU_WAITS", "1")
    args, want = attn_args(np.random.default_rng(7), Hq=8, Hkv=2, T=100)
    comp, imgs = compile_kernel(attention_decode, Config(S=S), **args)
    ri = run(comp, [i.copy() for i in imgs], "isa")
    rr = run(comp, [i.copy() for i in imgs], "rtl")
    assert_same_state(ri, rr)
    assert rel(rr.outputs["out"], want) < 0.03


def _uneven_gather_programs(S):
    """S slices' GATHER programs, the odd slices' one instruction longer (an LI: the dst through a
    register), so they differ in length (test_rtl's test_gather_dst_through_registers)."""
    from test_rtl import DATA
    return [[I.ld(DATA, 0, 4096), I.gather(0, 5000, 2, 16, 32, 16, 64), I.halt()] if s % 2 == 0
            else [I.li(6, 40), I.ld(DATA, 0, 4096), I.gather(0, 4960, 2, 16, 32, 16, 64, rb=6),
                  I.halt()] for s in range(S)]


@pytest.mark.parametrize("S", [2, 4])
def test_boot_entries_follow_the_longest_program(monkeypatch, tmp_path, S):
    """rtlsim's boot images under WAITC, as Board.load_program writes them: the slices load one
    PROG_N (the longest program's), so each slice's program is padded with NOPs to it before its
    zeros to the chunk and its entries (the NOPs' 0), and the loader finds every slice's entry of
    instruction i at the same place. (A shorter program's entries right after its own words put
    the loader's rows off by the difference: its GATHER read a 0 count for the LD it waits on.)"""
    from opentpu import Config, rtlsim
    from test_rtl import _images
    seen = {}

    def fake_run(args, timeout=None):
        d = Path(next(a for a in args if a.startswith("+dir="))[5:])
        seen["args"] = args
        seen["imgs"] = {s: np.fromfile(d / f"dram_{s}.bin", ">u4") for s in range(S)}
        raise _Stop
    monkeypatch.setattr(rtlsim, "build_top", lambda *a, **k: tmp_path / "none")
    monkeypatch.setattr(rtlsim, "run_sim", fake_run)
    monkeypatch.setitem(rtlsim.UARCH, "SEQ_WC", 1)
    monkeypatch.setitem(rtlsim.UARCH, "WIN", 16)
    monkeypatch.setenv("OTPU_WAITS", "1")
    cfg = Config(S=S)
    progs = _uneven_gather_programs(S)
    with pytest.raises(_Stop):
        rtlsim.run(cfg, progs, _images(np.random.default_rng(0), S), boot=True)
    at = int(next(a for a in seen["args"] if a.startswith("+boot_addr="))[11:])
    n = int(next(a for a in seen["args"] if a.startswith("+boot_n="))[8:])
    assert n == 4
    e0 = at // 4 + -(-8 * n // (cfg.D // 4)) * (cfg.D // 4)       # the loader's entry rows
    for s, p in enumerate(progs):
        img = seen["imgs"][s]
        want = W.wait_counts(p, cfg.D, S)
        got = img[e0:e0 + 2 * n].astype(np.uint64)
        got = got[0::2] | got[1::2] << np.uint64(32)
        assert list(got[:len(p)]) == list(want) and not got[len(p):].any(), s
        assert want[p.index(next(i for i in p if i.op == I.GATHER))] != 0     # it waits


class _Stop(Exception):
    pass


@pytest.mark.parametrize("S", [2, 4])
def test_waitc_rtl_boot_programs_of_different_lengths(have_verilator, monkeypatch, S):
    """WAITC with boot on slices whose programs differ in length: the RTL's TMEM equals the ISA
    simulator's, and its own check (every conflict on the window waited for) holds."""
    from opentpu import Config, rtlsim
    from opentpu.isasim import Machine
    from test_rtl import _images
    monkeypatch.setitem(rtlsim.UARCH, "SEQ_WC", 1)
    monkeypatch.setitem(rtlsim.UARCH, "WIN", 16)
    monkeypatch.setenv("OTPU_WAITS", "1")
    cfg = Config(S=S)
    progs = _uneven_gather_programs(S)
    imgs = _images(np.random.default_rng(3300), S)
    m = Machine(cfg, progs, [i.copy() for i in imgs]).run()
    _, tmems, _ = rtlsim.run(cfg, progs, [i.copy() for i in imgs], boot=True)
    for s in range(S):
        assert np.array_equal(tmems[s], m.slices[s].tmem), s


def _ooo_program(n_after):
    """A VOP held by a long LD, then n_after independent VOPs (the VPU starts them around it,
    and from the 31st on each waits for the held one: the sequencer's hold), then a VOP that
    reads the held one's and the 3rd last's results, and an ST of it."""
    prog = [I.ld(0, 0, 4096), vop(I.V_ADD, 5000, 0, 0, rows=8, cols=64)]
    for i in range(n_after):
        prog.append(vop(I.V_MUL, 30000 + 64 * i, 20000 + 64 * i, 20000 + 64 * i, cols=64))
    prog += [vop(I.V_ADD, 40000, 5000, 30000 + 64 * (n_after - 3), cols=64),
             I.st(1 << 16, 40000, 64), I.halt()]
    return prog


@pytest.mark.parametrize("n_after", [6, 20, 40])
def test_waitc_rtl_vpu_out_of_order(have_verilator, monkeypatch, n_after):
    """WAITC: the VPU starts younger VOPs around a held one (a mask per consumer), and holds
    the 32nd; the RTL's state equals the ISA simulator's."""
    from opentpu import Config, rtlsim
    from opentpu.isasim import Machine
    monkeypatch.setenv("OTPU_WAITS", "1")
    cfg = Config(S=1)
    prog = _ooo_program(n_after)
    rng = np.random.default_rng(n_after)
    img = np.zeros(1 << 20, np.uint8)
    img[:4 * 4096] = rng.uniform(-1, 1, 4096).astype(np.float32).view(np.uint8)
    img[20000 * 4:23000 * 4] = rng.uniform(-1, 1, 3000).astype(np.float32).view(np.uint8)
    prog.insert(0, I.ld(20000 * 4, 20000, 3000))     # the VOPs' inputs first
    w = W.wait_counts(prog)
    # the held VOP n_after + 1 back (none past VMAX: it has completed), the 3rd last
    assert vm(w, len(prog) - 3) == ((1 << n_after) if n_after < W.VMAX else 0) | 0b100
    m = Machine(cfg, [prog], [img.copy()]).run()
    drams, tmems, _ = rtlsim.run(cfg, [prog], [img.copy()], uarch={"SEQ_WC": 1, "WIN": 16})
    assert np.array_equal(tmems[0], m.slices[0].tmem)
    assert np.array_equal(drams[0], m.slices[0].dram)


@pytest.mark.parametrize("win", W.WINS)
@pytest.mark.parametrize("seed", range(4))
def test_waitc_rtl_hazard_stress(have_verilator, monkeypatch, seed, win):
    """WAITC on the scoreboard stress programs (RAW / WAR / WAW everywhere): the compiler's
    entries, the RTL's state equals the ISA simulator's and its own check holds, at a window of
    16 and of 32 (the counts saturated)."""
    from opentpu import Config, rtlsim
    from opentpu.isasim import Machine
    from test_rtl import _hazard_program, _images
    monkeypatch.setenv("OTPU_WAITS", "1")
    rng = np.random.default_rng(3000 + seed)
    cfg = Config(S=1, WC_WIN=win)
    prog = _hazard_program(rng, cfg)
    imgs = _images(rng, 1)
    m = Machine(cfg, [prog], [i.copy() for i in imgs]).run()
    drams, tmems, _ = rtlsim.run(cfg, [prog], [i.copy() for i in imgs],
                                 uarch={"SEQ_WC": 1, "WIN": win})
    assert np.array_equal(tmems[0], m.slices[0].tmem)
    assert np.array_equal(drams[0], m.slices[0].dram)


def test_waitc_rtl_window_32(have_verilator, monkeypatch):
    """otpu_seq WAITC at WIN 32: a long LD, 20 short ones behind it, then a VOP reading the long
    one's rows. The 21 LDs are on the window at once, the long one still running when the VOP
    issues: its saturated count (15, the DMA's 15th last: the long LD done before it) holds it,
    and the RTL's state is the ISA simulator's. The window-16 entries (none: 21 back has
    completed on a window of 16) trip the RTL's own check; a window of 16 runs them; and an RTL
    window over the entries' is refused."""
    from opentpu import Config, rtlsim
    from opentpu.isasim import Machine
    monkeypatch.setenv("OTPU_WAITS", "1")
    prog = [I.ld(0, 100, 4096)] + [I.ld(20000 + 64 * i, 6000 + 8 * i, 8) for i in range(20)] + \
        [vop(I.V_ADD, 9000, 100, 100), I.halt()]
    assert k(W.wait_counts(prog, win=32), 21)[W.U_DMA] == 15
    assert k(W.wait_counts(prog), 21)[W.U_DMA] == 0
    rng = np.random.default_rng(32)
    img = np.zeros(1 << 20, np.uint8)
    img[:4 * 8192] = rng.uniform(-1, 1, 8192).astype(np.float32).view(np.uint8)
    for win in W.WINS:
        cfg = Config(S=1, WC_WIN=win)
        m = Machine(cfg, [prog], [img.copy()]).run()
        drams, tmems, _ = rtlsim.run(cfg, [prog], [img.copy()], uarch={"SEQ_WC": 1, "WIN": win})
        assert np.array_equal(tmems[0], m.slices[0].tmem), win
        assert np.array_equal(drams[0], m.slices[0].dram), win
    with pytest.raises(ValueError, match="window of 16"):
        rtlsim.run(Config(S=1), [prog], [img.copy()], uarch={"SEQ_WC": 1, "WIN": 32})
    orig = W.wait_counts
    monkeypatch.setattr(W, "wait_counts", lambda *a, **kw: orig(*a, **{**kw, "win": 16}))
    with pytest.raises(Exception, match="WAITC: pc 21's wait 0 .k 0. misses"):
        rtlsim.run(Config(S=1, WC_WIN=32), [prog], [img.copy()], uarch={"SEQ_WC": 1, "WIN": 32})


@pytest.mark.parametrize("wf,rsc,size", [(I.W8, False, 4 + 4 + 2), (I.W4I, False, 12 + 4 * 6),
                                         (I.W4F, False, 12 + 4 * 6), (I.WNV, False, 24 + 8 * 6),
                                         (I.W4F, True, 4)])
def test_mx_mm_scale_rows(wf, rsc, size):
    """The MX column's otpu_pkg: an MM reads each scale row up to block k0 + KB's (RSC: Sw;
    int8x: Sw and a byte a block; int4x / fp4x6: Sw, a word and a word a block; NVFP4: Sw,
    three words and 8 bytes a block); KB is w4[23:16], k0 w4[31:24]."""
    R = [(0, 0)] * 16
    ins = I.mm(0, 0x10000, 100, 3, 2, 256, 3, 1, 0, 64, wf=wf, k0=4, rsc=rsc)
    fp = W.footprint(ins, R)
    assert (W.SP_DRAM, 0x10000, 0x10000 + 2 * 64 + size) in fp.rd
    assert (W.SP_ACT, 0, 2) in fp.rd                 # ab 0, KB 2 (not w4[31:16])


def test_mx_qst_scale_rows():
    """QST writes a scale row of 1 + ceil(KB / 4) words a row (Sw, then a byte a block)."""
    R = [(0, 0)] * 16
    ins = I.qst(0, 0x1000, 0x8000, 3, 5, 5 * D, 5 * D, 1)
    fp = W.footprint(ins, R)
    assert (W.SP_DRAM, 0x8000, (0x8000 + 4 * 3 * (1 + 2)) | 63) in fp.wr


# ------------------------------------------------------------------------------ the host's loads
# A bitstream whose sequencer takes wait entries (CFG_UARCH bit24, Config.SEQ_WC) loads every
# program with them; the host refuses a program without them there, and changes nothing elsewhere.
def _prog():
    return [I.ld(0, 100, 8), vop(I.V_ADD, 200, 100, 100), vop(I.V_MUL, 300, 200, 100),
            I.st(4096, 300, 8), I.halt()]


def _board(seq_wc: bool):
    from opentpu.host.board import Board
    from opentpu.host.fake import FakeTransport
    t = FakeTransport(regmap=4, devname=None, seq_wc=seq_wc)
    return Board(t), t


def test_a_waitc_bitstream_refuses_a_program_without_entries():
    from opentpu.host import regs as RG
    from opentpu.host.board import ConfigMismatch
    b, t = _board(True)
    assert b.waitc
    words = np.asarray(I.assemble(_prog()), np.uint32)
    with pytest.raises(ConfigMismatch, match="wait entries"):
        b.load_program(0x10000, words)
    with pytest.raises(ValueError, match="do not match"):
        b.load_program(0x10000, words, np.zeros(2, np.uint64))
    assert t.regs[RG.R_PROG_N] == 0                         # nothing was loaded


def test_a_waitc_bitstream_loads_the_entries_after_the_program():
    """Board.load: the words, zeros to the next chunk, the 64-bit entries (waits.loadable), at
    the program's address; PROG_N counts the instructions alone."""
    from opentpu.host import regs as RG
    b, t = _board(True)
    prog = _prog()
    b.load(0x10000, prog)
    words = np.asarray(I.assemble(prog), np.uint32)
    ent = W.wait_counts(prog, D, 1)
    got = b.read(0x10000, 2 * D + 8 * len(prog)).view("<u4")
    assert np.array_equal(got[:len(words)], words)
    assert not got[len(words):2 * D // 4].any()             # 160 bytes of program: to 256
    assert np.array_equal(got[2 * D // 4:].view("<u8"), ent)
    assert vm(ent, 2) == 0b1 and t.regs[RG.R_PROG_N] == len(prog)


def test_a_bitstream_without_seq_wc_loads_the_words_alone():
    """No CFG_UARCH bit24: load_program as before (entries, if given, unused), Board.load the
    same words."""
    b, t = _board(False)
    assert not b.waitc
    prog = _prog()
    words = np.asarray(I.assemble(prog), np.uint32)
    b.write(0x10000, np.full(512, 0xAB, np.uint8))
    b.load_program(0x10000, words)
    got = b.read(0x10000, 512)
    assert np.array_equal(got[:4 * len(words)].view("<u4"), words)
    assert (got[4 * len(words):] == 0xAB).all()             # nothing after the program
    b.load(0x10000, prog)
    assert np.array_equal(b.read(0x10000, 512), got)


def test_the_window_guard_on_the_host(monkeypatch):
    """A WAITC bitstream of a target whose window is over 16 and not 32 is refused: the entries
    name a unit's last 15 instructions (at 32 they saturate, Config.WC_WIN)."""
    import dataclasses

    from opentpu import target as T
    from opentpu.host.board import ConfigMismatch
    b, _ = _board(True)
    t = T.current()
    wide = dataclasses.replace(t, uarch={**t.uarch, "WIN": 24})
    monkeypatch.setattr(T, "by_platform", lambda pid: wide)
    with pytest.raises(ConfigMismatch, match="window of 24"):
        b.load(0x10000, _prog())


def test_a_bitstream_at_a_window_of_32_takes_saturated_entries():
    """A WAITC bitstream whose CFG_HASH holds WC_WIN 32 (a target at WIN 32: bcu1525-wg:bhw,
    bcu1525-s2): device_wc_win finds the window, check_hash passes, device_config takes it and
    Board.load writes the entries at 32; the same registers with the hash at 16 load the
    window-16 entries."""
    from opentpu.host import board as B
    from opentpu.host import regs as RG
    from opentpu.host.fake import FakeTransport
    from opentpu.target import arch_hash
    b16, t16 = _board(True)
    info = b16.info()
    aw = B.device_aw(info)
    assert B.device_wc_win(info, aw) == 16 and B.device_config(info).WC_WIN == 16
    c = dict(RG.cfg(t16.cfg_words), hash=arch_hash(B._reg_config(info, aw, 32)))
    b = B.Board(FakeTransport(regmap=4, devname=None, seq_wc=True, cfg=c))
    i32 = b.info()
    assert B.device_aw(i32) == aw and B.device_wc_win(i32, aw) == 32
    B.check_hash(i32)
    assert B.device_config(i32).WC_WIN == 32
    prog = _dma_run(20)
    words = np.asarray(I.assemble(prog), np.uint32)
    for brd, win in ((b16, 16), (b, 32)):
        ent = W.wait_counts(prog, D, 1, win=win)
        assert k(ent, 21)[W.U_DMA] == (15 if win == 32 else 0)
        want = W.loadable(words, ent, D)
        brd.load(0x10000, prog)
        assert np.array_equal(brd.read(0x10000, 4 * len(want)).view("<u4"), want), win


def test_device_config_takes_seq_wc_and_check_fits_needs_its_room():
    """device_config's SEQ_WC is CFG_UARCH bit24 (the CFG_HASH has it: check_hash passes);
    check_fits refuses a configuration without the entries' room on such a bitstream, and a
    configuration with the room runs on a bitstream without them."""
    import dataclasses

    from opentpu.host.board import ConfigMismatch, check_fits, device_config
    b, _ = _board(True)
    cfg = device_config(b.info())
    assert cfg.SEQ_WC
    check_fits(cfg, b.info())
    with pytest.raises(ConfigMismatch, match="no room for them"):
        check_fits(dataclasses.replace(cfg, SEQ_WC=False), b.info())
    b2, _ = _board(False)
    assert not device_config(b2.info()).SEQ_WC
    check_fits(cfg, b2.info())


def test_a_host_without_seq_wc_refuses_a_waitc_bitstream():
    """The CFG_HASH of a build with SEQ_WC has it (target.arch_hash), so a host that writes no
    entries, whose hash leaves it out, refuses the bitstream (check_hash); without it the hash
    is every earlier build's."""
    import dataclasses

    from opentpu import target as T
    card = dataclasses.replace(T.config("ypcb-00338"), SEQ_WC=False)
    # MX production 8b33206e's (MCOLS 4 and 64K TMEM words, no entries): every earlier build's hash
    assert f"{T.arch_hash(dataclasses.replace(card, MCOLS=4, TMEM_WORDS=1 << 16)):08x}" == "89417856"
    on = dataclasses.replace(card, SEQ_WC=True)
    assert T.arch_hash(on) != T.arch_hash(card)
    assert on == T.config("ypcb-00338")                 # release r2's card takes the entries
    assert T.main(["hash", "ypcb-00338", "SEQ_WC=1"]) == 0          # (run_vivado.sh's CFG_HASH)
    assert T.arch_hash(T.build_config("ypcb-00338", SEQ_WC="1")) == T.arch_hash(on)


def test_board_backend_runs_with_the_entries_and_its_chain_slots_hold_them():
    """BoardBackend on a WAITC bitstream: start() loads each program with its entries (those of
    its run_args' bounds, for a worker's words too: compiler.Words), slot_words
    (generate.slot_words) puts them after a chained program; on a bitstream without the bit
    the words alone, as before. A configuration without their room is refused there."""
    from opentpu.compiler import Program, RunVar, Words
    from opentpu.host.board import BoardBackend, ConfigMismatch
    from opentpu.host.fake import FakeTransport
    from opentpu.isasim import board_config
    from opentpu.llm import generate as G
    pos = RunVar("pos", 64)
    # the LD reads below 4 KiB with the bound, anywhere without: the QST's DRAM write waits
    prog = Program([I.addi(1, 15, 0), I.ld(0, 100, 8, ra=1),
                    I.qst(2000, 0x10000, 0x18000, 1, 1, D, D, 1), I.halt()], [(pos, 64)])
    ent = W.wait_counts(prog, D, 1)
    assert k(ent, 2)[W.U_DMA] == 0 and k(W.wait_counts(prog, D, 1, run_args=()), 2)[W.U_DMA] == 1
    words = np.asarray(I.assemble(prog), np.uint32)
    n = 4 * len(words)
    for seq_wc in (True, False):
        t = FakeTransport(regmap=4, devname=None, seq_wc=seq_wc)
        cfg = board_config(DRAM_BYTES=1 << 21, SEQ_WC=True)
        be = BoardBackend(cfg, [np.zeros(1 << 16, np.uint8)], transport=t, status=False)
        assert be.waitc == seq_wc
        for p in ([prog], Words.of(words, prog.run_args)):
            be.board.write(be.prog_at, np.zeros(1024, np.uint8))
            be.start(p)
            be.wait()
            got = be.board.read(be.prog_at, 1024)
            assert np.array_equal(got[:n].view("<u4"), words)
            if seq_wc:                                  # (from the next chunk)
                off = -(-n // D) * D
                assert np.array_equal(got[off:off + 8 * len(prog)].view("<u8"), ent)
            else:
                assert not got[n:].any()
        assert np.array_equal(G.slot_words(be, prog),
                              W.loadable(words, ent, D) if seq_wc else words)
        be.close()
    t = FakeTransport(regmap=4, devname=None, seq_wc=True)
    with pytest.raises(ConfigMismatch, match="no room"):
        BoardBackend(board_config(DRAM_BYTES=1 << 21), [np.zeros(4096, np.uint8)], transport=t,
                     status=False)


def test_the_entries_room():
    """A program of IMEM's instructions with its entries is waits.area's 5 / 4 of its words (IMEM
    fills whole chunks), a chain slot two of them (generate.prog_bytes)."""
    from opentpu.llm import generate as G
    n = 32768 // 8
    words = np.zeros(32768, np.uint32)
    assert 4 * len(W.loadable(words, np.zeros(n, np.uint64), D)) == W.area(32768, True) == 163840
    assert W.area(32768) == 131072
    assert G.prog_bytes(32768) == 1 << 18 and G.prog_bytes(32768, True) == 327680


def test_the_caches_key_on_seq_wc_and_the_waits_pass(monkeypatch):
    """The program and image caches' layouts hold the Config, SEQ_WC in it (its chain slots and
    program area differ); waits.py is in the keyed source (progcache.keyed), so an entry and a
    program compiled before a change to the pass miss; the entries' key holds the words, their
    arguments' bounds, D, S and the source."""
    import dataclasses

    from opentpu import imgcache as IC
    from opentpu import progcache as PC
    from opentpu.compiler import RunVar
    from opentpu.isasim import board_config
    cfg = board_config()
    on = dataclasses.replace(cfg, SEQ_WC=True)
    assert PC.key((cfg,), ("decode", 1)) != PC.key((on,), ("decode", 1))
    img = object()
    assert IC.key(img, (cfg,), "ck") != IC.key(img, (on,), "ck")
    assert "waits.py" in {p.name for p in PC.keyed()}
    words = np.asarray(I.assemble(_prog()), np.uint32)
    k0 = W.entries_key(words, D, 1)
    assert W.entries_key(words, D, 1, [(RunVar("pos", 64), 1)]) != k0
    assert W.entries_key(words, D, 2) != k0
    monkeypatch.setattr(PC, "_SRC", "another tree")
    assert W.entries_key(words, D, 1) != k0


def test_entries_are_kept(monkeypatch, tmp_path):
    """waits.entries computes once per program: then from this process, or the program cache's
    disk (an entry beside its programs)."""
    monkeypatch.setenv("OTPU_PROG_CACHE", str(tmp_path))
    monkeypatch.setattr(W, "_mem", W.OrderedDict())
    prog = _prog()
    words = np.asarray(I.assemble(prog), np.uint32)
    n0 = dict(W.stats)
    e = W.entries(words, D, 1)
    assert np.array_equal(e, W.wait_counts(prog, D, 1, run_args=()))
    assert W.stats["computed"] == n0["computed"] + 1
    assert np.array_equal(W.entries(words, D, 1, prog=prog), e)
    assert W.stats["memory"] == n0["memory"] + 1
    W._mem.clear()
    assert np.array_equal(W.entries(words, D, 1), e)
    assert W.stats["disk"] == n0["disk"] + 1 and len(list(tmp_path.glob("*/*.npz"))) == 1


# ------------------------------------------------------------------------------ off the critical path
def test_words_pickle_with_their_run_args_and_entries():
    """A compile worker's words reach the engine with their run_args and entries (Words)."""
    import pickle

    from opentpu.compiler import RunVar, Words
    w = Words.of(I.assemble(_prog()), [(RunVar("pos", 64), 1)], np.arange(5, dtype=np.uint64))
    r = pickle.loads(pickle.dumps(w))
    assert isinstance(r, Words) and np.array_equal(r, w) and r.run_args[0][1] == 1
    assert r.run_args[0][0].bound == 64 and np.array_equal(r.waits, w.waits)
    assert Words.of(r, r.run_args).waits is r.waits        # (kept through Words.of)


def test_worker_words_carry_the_entries_on_a_waitc_configuration():
    """qwen3.worker_words: the entries computed with the words (in the worker process) when the
    image's configuration takes them, none otherwise; BoardBackend loads a worker's entries as
    they come."""
    from types import SimpleNamespace

    from opentpu.compiler import Program, RunVar
    from opentpu.host.board import BoardBackend
    from opentpu.host.fake import FakeTransport
    from opentpu.isasim import board_config
    from opentpu.llm.qwen3 import worker_words
    prog = Program([I.addi(1, 15, 0), I.ld(0, 100, 8, ra=1),
                    I.qst(2000, 0x10000, 0x18000, 1, 1, D, D, 1), I.halt()],
                   [(RunVar("pos", 64), 64)])
    on = SimpleNamespace(cfg=board_config(SEQ_WC=True))
    w = worker_words(on, prog)
    assert np.array_equal(w.waits, W.wait_counts(prog, D, 1)) and w.run_args == prog.run_args
    assert worker_words(SimpleNamespace(cfg=board_config()), prog).waits is None
    t = FakeTransport(regmap=4, devname=None, seq_wc=True)
    be = BoardBackend(board_config(DRAM_BYTES=1 << 21, SEQ_WC=True), [np.zeros(4096, np.uint8)],
                      transport=t, status=False)
    w.waits = np.full(len(prog), W.DEFAULT, np.uint64)     # (what the worker said: loaded)
    be.start(w)
    be.wait()
    got = be.board.read(be.prog_at + D, 8 * len(prog)).view("<u8")
    assert (got == W.DEFAULT).all()
    be.close()


def test_entries_many_in_a_pool():
    """waits.entries_many: the programs not kept yet computed in the pool at once, then kept."""
    from concurrent.futures import ThreadPoolExecutor
    W._mem.clear()
    progs = [_prog(), [I.ld(0, 300, 8), vop(I.V_ADD, 300, 300, 300), I.halt()]]
    items = [(np.asarray(I.assemble(p), np.uint32), None, p) for p in progs]
    with ThreadPoolExecutor(2) as pool:
        got = W.entries_many(items, D, 1, pool)
    assert all(np.array_equal(g, W.wait_counts(p, D, 1)) for g, p in zip(got, progs))
    n = W.stats["memory"]
    assert all(np.array_equal(a, b) for a, b in zip(W.entries_many(items, D, 1), got))
    assert W.stats["memory"] == n + 2


def test_prebuild_keeps_the_sessions_entries(tmp_path, monkeypatch):
    """prebuild's entries step (waits_ahead) on a WAITC configuration: each bucket's resident
    decode and generate programs and the warm prompt runs, kept on the disk under the keys a
    session's engine looks them up by; nothing without SEQ_WC."""
    import dataclasses

    from opentpu.compiler import Words
    from opentpu.host import prebuild as PB
    from opentpu.llm.qwen3 import PREFILL_ROWS, Engine, device_config
    from test_autodecode import _tiny
    monkeypatch.setenv("OTPU_PROG_CACHE", str(tmp_path))
    monkeypatch.setattr(W, "_mem", W.OrderedDict())
    Wt, spec = _tiny("qwen3")
    cfg = dataclasses.replace(device_config(spec, 512, rows=PREFILL_ROWS, lookup=True),
                              SEQ_WC=True)
    eng = Engine(spec, Wt, cap=512, cfg=cfg, backend=PB._Drop, resident=True, pipeline=False)
    n = PB.waits_ahead(eng)
    assert n["programs"] >= 4 and n["computed"] == n["programs"]
    assert PB.waits_ahead(Engine(spec, Wt, cap=512, cfg=dataclasses.replace(cfg, SEQ_WC=False),
                                 backend=PB._Drop, resident=True, pipeline=False)) is None
    W._mem.clear()                          # (a session: another process, the card's flags)
    ses = Engine(spec, Wt, cap=512, cfg=cfg, backend=PB._Drop, resident=True)
    assert ses.resident
    for b in (1, 2):
        progs, ra = ses._decode((b - 1) * 256)
        w = progs if isinstance(progs, Words) else I.assemble(progs[0])
        assert (tmp_path / (k := W.entries_key(w, cfg.D, cfg.S, ra))[:2] / f"{k}.npz").is_file()
        g = ses._generate_prog(b)[0]
        k = W.entries_key(I.assemble(g), cfg.D, cfg.S, g.run_args)
        assert (tmp_path / k[:2] / f"{k}.npz").is_file()


def test_prebuilds_backend_has_the_cards_flags():
    """prebuild's _Drop compiles what a card session does: BoardBackend's flags on a bitstream
    with run arguments and the decode loop, on a transport that streams."""
    from opentpu.host import prebuild as PB
    from opentpu.host.board import BoardBackend
    from opentpu.host.fake import FakeTransport
    from opentpu.isasim import board_config
    t = FakeTransport(regmap=4, devname=None, args=True, gen=True)
    t.streams = True
    be = BoardBackend(board_config(DRAM_BYTES=1 << 21), [np.zeros(4096, np.uint8)], transport=t,
                      status=False)
    for k in ("args", "chains", "generates", "fills"):
        assert bool(getattr(be, k)) and getattr(PB._Drop, k), k
    be.close()


def test_an_rld_bound_narrows_the_footprints_and_the_isa_simulator_checks_it():
    """isa.rld's bound (flag bit 2, w3..w4: what the kernel knows of the word, e.g. an index
    below E): the waits pass gives the register those values instead of any, so a VOP addressed
    by it conflicts only with what its range can reach; the ISA simulator stops on a value
    outside it; the hardware ignores the field (unbounded RLDs are as before)."""
    from opentpu import Config
    from opentpu.isasim import Machine, SimError
    ins = I.rld(1, 100, bound=(0, 7))
    assert ins.flags & I.F_BOUND and ins.w[2:4] == [0, 7]
    back = I.Instr.decode(ins.encode())
    assert back.flags == ins.flags and back.w == ins.w
    assert I.rld(1, 100).flags & I.F_BOUND == 0
    with pytest.raises(ValueError):
        I.rld(1, 100, bound=(5, 4))

    def prog(bound):
        return [I.rld(1, 100, bound=bound),
                vop(I.V_FILL, 1000, 0, 0, cols=1, ra=1, bmode=I.B_SCALAR, imm=1.0),
                vop(I.V_ADD, 3000, 2000, 2000),            # nowhere near 1000 .. 1007
                I.ld(0, 2000, 8),
                I.halt()]
    loose, tight = (W.wait_counts(prog(b), args=[0] * 8) for b in (None, (0, 7)))
    assert vm(loose, 2) == 0b1 and vm(tight, 2) == 0    # R1 any: the FILL may hit 2000
    assert vm(loose, 3) == 0b11 and vm(tight, 3) == 0b1  # (the ADD reads 2000: WAR either way)
    cfg = Config(S=1, D=128, DRAM_BYTES=1 << 16)
    for v, ok in ((7.0, True), (8.0, False)):
        p = [vop(I.V_FILL, 100, 0, 0, cols=1, bmode=I.B_SCALAR, imm=v)] + prog((0, 7))
        if ok:
            Machine(cfg, [p], [None]).run()
        else:
            with pytest.raises(SimError, match="outside its bound"):
                Machine(cfg, [p], [None]).run()


# ------------------------------------------------------------------------- AW > 32 (the BCU's 36)
def _hi_prog(st_at, mm_at, aw):
    """An ST of 32 bytes at st_at, then an MM streaming one 128-byte row at mm_at (both through
    registers: LI's {w2, w1}), the MM's output read by a VOP at TMEM R3 + 0 (R3 an AW-bit value
    whose low bits are 600)."""
    return [I.li(1, st_at, aw=aw), I.li(2, mm_at, aw=aw), I.li(3, (1 << 32) + 600, aw=aw),
            I.st(0, 100, 8, ra=1),
            I.mm(0, 0x4000, 600, 1, 1, 128, 1, 1, 0, 0, unit=True, ra=2),
            vop(I.V_ADD, 700, 0, 0, rb=3, rc=3),
            I.halt()]


def test_a_range_across_4gib_waits_at_aw():
    """At AW 36 an MM whose row crosses 4 GiB (0xFFFF_FFC0 + 128 bytes) reads what an ST wrote
    at 0x1_0000_0000: its stream waits for the DMA. Mod 2^32 its range would be clamped at
    4 GiB (and the ST's at 0) and the dependency missed: with SEQ_WC nothing else catches it."""
    prog = _hi_prog(1 << 32, 0xFFFF_FFC0, 36)
    w = W.wait_counts(prog, args=[0] * 8, aw=36)
    assert k(w, 4)[5] == 1 or k(w, 4)[W.U_DMA] == 1
    assert vm(w, 5) == 0 and k(w, 5)[W.U_MXU] == 1        # T[600]: the low 32 bits of R3
    w32 = W.wait_counts(prog, args=[0] * 8)               # the same words, a 32-bit machine's
    assert k(w32, 4)[5] == 0 and k(w32, 4)[W.U_DMA] == 0
    # the checker on the AW 36 registers finds the 32-bit entries' miss
    regs = [0, 1 << 32, 0xFFFF_FFC0, (1 << 32) + 600] + [0] * 12
    c = W.Checker(prog, w32, aw=36)
    c.issue(3, regs)
    with pytest.raises(AssertionError, match="misses the 1-th last DMA"):
        c.issue(4, regs)
    c = W.Checker(prog, w, aw=36)
    for pc in (3, 4, 5):
        c.issue(pc, regs)


def test_ranges_4gib_apart_do_not_alias_at_aw():
    """The KV caches below 4 GiB and the weights above: an ST at 0x40 and an MM at
    0x1_0000_0040 do not conflict at AW 36 (mod 2^32 they would, a wait for nothing)."""
    prog = _hi_prog(0x40, (1 << 32) + 0x40, 36)
    w = W.wait_counts(prog, args=[0] * 8, aw=36)
    assert k(w, 4)[5] == 0 and k(w, 4)[W.U_DMA] == 0
    w32 = W.wait_counts(prog, args=[0] * 8)
    assert k(w32, 4)[5] == 1 or k(w32, 4)[W.U_DMA] == 1


def test_counts_cover_a_run_above_4gib():
    """A loop walking down from 8 GiB by a negative ADDI (w2 0xFFFFFFFF): LD, VOP, ST each
    iteration, and an MM streaming the row the ST before last wrote (just under R1 + 4 GiB, so
    mod 2^32 below R1); the entries at AW 36 cover the ISA simulator's runs (registers of 36
    bits), the loop's count known (exact: the MM waits for the 3rd last DMA) or not (widened)."""
    from opentpu.isasim import Config, Machine
    cfg = Config(S=1, D=D, AW=36, DRAM_BYTES=1 << 34, TMEM_WORDS=1 << 12)
    for count, rc in ((4, 0), (0, 8)):
        prog = [I.li(1, 2 << 32, aw=36),
                I.loop(5, count, rcount=rc),
                I.ld(0, 100, 8, ra=1),
                vop(I.V_ADD, 200, 100, 100),
                I.st(0xFFFF_FF00, 200, 8, ra=1),
                I.mm(0xFFFF_FF80, 0x4000, 600, 1, 1, 128, 1, 1, 0, 0, unit=True, ra=1),
                I.addi(1, 1, -0x80, aw=36),
                I.halt()]
        words = W.wait_counts(prog, D, args=None if rc else [0] * 8, aw=36)
        assert k(words, 3)[W.U_DMA] == 1 and vm(words, 4) & 1    # (and the VOPs before)
        assert max(k(words, 5)[0], k(words, 5)[5]) == (1 if rc else 3)
        for n in (1, 3):
            m = Machine(cfg, [prog], [None], args=[n])
            assert W.check_isasim(m, [words]) == 4 * (n if rc else count)
            assert m.slices[0].R[1] == (2 << 32) - (n if rc else count) * 0x80


def test_entries_key_keeps_its_value_at_aw_32():
    w = np.asarray(I.assemble([I.halt()]), np.uint32)
    assert W.entries_key(w, D, 1) == W.entries_key(w, D, 1, aw=32) != W.entries_key(w, D, 1, aw=36)


def test_an_rld_bound_past_4gib_bounds_an_aw_product():
    """isa.rld's bound above 2^32 (a resident expert's place, id x expert bytes: llm/moe.py
    resident): its high words in w5 and w6 ({w5, w3} .. {w6, w4}; a 32-bit bound's words as
    before). At AW 36 the waits pass gives the register those values, so an MM from the pool
    (4 GiB + R1) waits for an ST below 4 GiB only without the bound, and for one into the pool
    either way; the ISA simulator stops on a product outside it."""
    from opentpu.isasim import Config, Machine, SimError
    assert I.rld(1, 100, bound=(0, 7)).w == [100, 0, 0, 7, 0, 0, 0]
    ins = I.rld(1, 100, mul=1 << 28, bound=(0, 5 << 32))
    assert ins.w[:6] == [100, 1 << 28, 0, 0, 0, 5]
    back = I.Instr.decode(ins.encode())
    assert back.flags == ins.flags and back.w == ins.w
    with pytest.raises(ValueError):
        I.rld(1, 100, bound=(0, 1 << 64))

    def prog(bound, st_at, v=3.0):
        return [vop(I.V_FILL, 100, 0, 0, cols=1, bmode=I.B_SCALAR, imm=v),
                I.rld(1, 100, mul=1 << 28, bound=bound),       # an id of at most 80
                I.addi(1, 1, 1 << 32, aw=36),
                I.li(2, st_at, aw=36),
                I.st(0, 200, 8, ra=2),
                I.mm(0, 0x4000, 600, 1, 1, 128, 1, 1, 0, 0, unit=True, ra=1),
                I.halt()]

    def dma(bound, st_at):
        w = W.wait_counts(prog(bound, st_at), args=[0] * 8, aw=36)
        return max(k(w, 5)[W.U_DMA], k(w, 5)[5])
    pool = (0, 5 << 32)
    assert (dma(pool, 0x40), dma(None, 0x40)) == (0, 1)           # (the KV below 4 GiB)
    assert dma(pool, (4 << 32) + 0x40) == dma(None, (4 << 32) + 0x40) == 1
    cfg = Config(S=1, D=D, AW=36, DRAM_BYTES=1 << 16, TMEM_WORDS=1 << 12)
    for v, ok in ((80.0, True), (81.0, False)):
        p = prog(pool, 0x40, v)[:2] + [I.halt()]
        if ok:
            assert Machine(cfg, [p], [None]).run().slices[0].R[1] == 5 << 32
        else:
            with pytest.raises(SimError, match="outside its bound"):
                Machine(cfg, [p], [None]).run()
