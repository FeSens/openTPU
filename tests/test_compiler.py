"""Compiler lowering: loops, address registers, layouts, and compile-time errors."""
import numpy as np
import pytest

from opentpu import Config, isa as I
from opentpu import language as ol
from opentpu.compiler import Affine, CompileError, RunVar, Tensor, arg_words, current
from opentpu.isasim import Machine
from opentpu.runtime import Input, KVCache, Output, Weight, compile_kernel, launch


def test_hardware_loop_lowering_uses_address_registers():
    @ol.jit
    def k(x, out):
        acc = ol.zeros([8])
        for i in ol.range(4):
            acc.set(acc + ol.load(x[i * 8:(i + 1) * 8]))
        ol.store(out, acc)

    xs = np.arange(32, dtype=np.float32)
    comp, _ = compile_kernel(k, Config(), x=Input(xs), out=Output((8,)))
    prog = comp.programs[0]
    loops = [p for p in prog if p.op == I.LOOP]
    assert len(loops) == 1 and loops[0].w[1] == 4
    lds = [p for p in prog if p.op == I.LD]
    assert len(lds) == 1 and lds[0].ra != 0            # loop-dependent address via a register
    r = launch(k, Config(), x=Input(xs), out=Output((8,)))
    assert np.array_equal(r.outputs["out"], xs.reshape(4, 8).sum(0))


def test_an_accumulator_at_its_dram_stride_drains_slower_than_its_array():
    """compiler.slow_drains: an MM of 4 rows (a replay group at MCOLS 4) and one block adding
    into a tile loaded from DRAM (row stride 256, a multiple of the TMEM banks: the 4 results
    of a streamed row in one bank, one a cycle) drains slower than its array, 1 cycle a column;
    into an ol.zeros tile (an odd row stride) it does not."""
    from opentpu.compiler import slow_drains

    @ol.jit
    def k(x, a, w, out, loaded: bool):
        y = ol.load(x) if loaded else ol.zeros([4, 256])
        ol.dot(ol.load(a), w, acc=y)
        ol.store(out, y)

    cfg = Config(MCOLS=4)
    rng = np.random.default_rng(0)
    xs, a_ = rng.standard_normal((4, 256), np.float32), rng.standard_normal((4, 32), np.float32)
    w = rng.standard_normal((256, 32), np.float32)
    for loaded in (True, False):
        comp, _ = compile_kernel(k, cfg, x=Input(xs), a=Input(a_), w=Weight(w),
                                 out=Output((4, 256)), loaded=loaded)
        slow = slow_drains(comp.programs[0], cfg)
        assert [c for _, c in slow] == (["mm 4x32 . 256x32^T"] if loaded else [])


def test_rows_loaded_into_an_accumulators_tile_go_one_ld_a_row():
    """Back-to-back rows loaded into an ol.empty tile (an MXU accumulator's odd row stride) go
    one LD a row, each to its tile row."""
    @ol.jit
    def k(x, out):
        t = ol.load(x, out=ol.empty([4, 16]))
        ol.store(out, t + 1.0)

    xs = np.arange(64, dtype=np.float32).reshape(4, 16)
    comp, _ = compile_kernel(k, Config(), x=Input(xs), out=Output((4, 16)))
    lds = [p for p in comp.programs[0] if p.op == I.LD]
    assert [ld.w[2] for ld in lds] == [16] * 4
    assert [ld.w[1] - lds[0].w[1] for ld in lds] == [0, 17, 34, 51]
    r = launch(k, Config(), x=Input(xs), out=Output((4, 16)))
    assert np.array_equal(r.outputs["out"], xs + 1)


def test_nested_loops_affine_addresses():
    @ol.jit
    def k(x, out):
        acc = ol.zeros([4])
        for i in ol.range(3):
            for j in ol.range(2):
                acc.set(acc + ol.load(x[i * 8 + j * 4: i * 8 + j * 4 + 4]))
        ol.store(out, acc)

    xs = np.arange(24, dtype=np.float32)
    r = launch(k, Config(), x=Input(xs), out=Output((4,)))
    assert np.array_equal(r.outputs["out"], xs.reshape(6, 4).sum(0))


def test_register_freed_inside_a_loop_is_not_reused_with_that_loops_terms():
    """A register freed after an inner loop (value 0 again) must not take an address that
    also depends on an enclosing loop: that loop's step would leave it non-zero for the
    register's earlier use in the next iteration."""
    @ol.jit
    def k(x, out):
        acc = ol.zeros([4])
        for i in ol.range(3):
            for j in ol.range(2):                   # an address of j only
                acc.set(acc * 2.0 + ol.load(x[j * 4:j * 4 + 4]))
            for j in ol.range(2):                   # addresses of i and j
                acc.set(acc * 3.0 + ol.load(x[8 + i * 8 + j * 4:8 + i * 8 + j * 4 + 4]))
        ol.store(out, acc)

    xs = (np.arange(32, dtype=np.float32) % 5) - 2
    want = np.zeros(4, np.float32)
    for i in range(3):
        for j in range(2):
            want = want * 2 + xs[j * 4:j * 4 + 4]
        for j in range(2):
            want = want * 3 + xs[8 + i * 8 + j * 4:8 + i * 8 + j * 4 + 4]
    r = launch(k, Config(), x=Input(xs), out=Output((4,)))
    assert np.array_equal(r.outputs["out"], want)


@pytest.mark.parametrize("flag", [0, 1])
def test_a_raw_guard_at_a_loop_body_end_gets_a_nop_after_it(flag):
    """A guard emitted as instructions (generate.py's: LOOP R[r] + its body) last in a loop's
    body: the compiler ends the body with a NOP, so the two bodies do not end together (at
    count 0 the guard would skip the loop's back edge)."""
    @ol.jit
    def k(out):
        b = current()
        acc = ol.zeros([4])
        f = ol.full([1], float(flag))
        r = b.scratch()
        b.rld(r, f)
        for i in ol.range(3):
            acc.set(acc + 1.0)
            b.emit(I.loop(1, 0, rcount=r))
            b.emit(I.vop(I.V_ADD, acc.base, acc.base, 0, 1, 4, 0, 0, 0, I.B_SCALAR, 10.0))
        b.unscratch(r)
        ol.store(out, acc)

    prog = compile_kernel(k, Config(), out=Output((4,)))[0].programs[0]
    outer = next(i for i, p in enumerate(prog) if p.op == I.LOOP and p.w[1] == 3)
    assert prog[outer + prog[outer].w[0]].op == I.NOP
    r = launch(k, Config(), out=Output((4,)))
    assert np.array_equal(r.outputs["out"], np.full(4, 3.0 * (1 + 10 * flag), np.float32))


def test_tile_views_index_like_python_or_refuse():
    """Tile views: a negative row counts from the end; a row outside the tile, a slice step and
    an empty view are compile errors (they would address words outside the view)."""
    @ol.jit
    def k():
        t = ol.zeros([3, 8])
        assert t[-1].base == t[2].base == t.base + 2 * t.rs and t[-1].shape == (8,)
        assert t[1:, -4:].base == t.base + t.rs + 4 and t[1:, -4:].shape == (2, 4)
        v = ol.zeros([8])
        assert v[-3:].base == v.base + 5
        for bad in (lambda: t[3], lambda: t[-4], lambda: t[::2, :], lambda: t[:, 1::2],
                    lambda: v[::2], lambda: v[5:2], lambda: t[2:2, :]):
            with pytest.raises(CompileError):
                bad()

    k.trace(Config(), 0, {})


def test_every_op_refuses_a_dead_tile():
    """A tile moved into another by .set() is dead: reductions, all-gathers and quantized
    stores refuse it as the elementwise ops do."""
    @ol.jit
    def k():
        b = current()
        t = ol.zeros([2, 32])
        t.dead = True
        for bad in (lambda: ol.sum(t, axis=1), lambda: ol.max(t, axis=1),
                    lambda: b.all_gather(t, 2),
                    lambda: b.store_quantized(t, Affine(0), Affine(1024), 32, 1, False)):
            with pytest.raises(CompileError, match="dead"):
                bad()

    k.trace(Config(S=1, D=32), 0, {})


def test_views_broadcast_and_division():
    @ol.jit
    def k(x, v, out):
        t = ol.load(x)                       # [3, 4]
        c = ol.load(v)                       # [4]
        m = ol.max(t, axis=1)                # [3]
        y = (t - m[:, None]) * c[None, :] / (m + 10.0)[:, None]
        z = ol.empty([3, 4])
        z[:, :2].set(y[:, 2:])
        z[:, 2:].set(y[:, :2])
        ol.store(out, z)

    x = np.random.default_rng(0).standard_normal((3, 4)).astype(np.float32)
    v = np.array([1, 2, 3, 4], np.float32)
    r = launch(k, Config(), x=Input(x), v=Input(v), out=Output((3, 4)))
    m = x.max(1, keepdims=True)
    y = (x - m) * v / (m + 10)
    assert np.allclose(r.outputs["out"], np.concatenate([y[:, 2:], y[:, :2]], 1), rtol=1e-5)


def test_dot_chunks_rows_beyond_mxu_columns():
    @ol.jit
    def k(x, w, out):
        ol.store(out, ol.dot(ol.load(x), w))

    rng = np.random.default_rng(1)
    x = rng.standard_normal((11, 64)).astype(np.float32)          # 11 > MCOLS = 8
    w = rng.standard_normal((16, 64)).astype(np.float32)
    comp, _ = compile_kernel(k, Config(), x=Input(x), w=Weight(w), out=Output((11, 16)))
    assert sum(p.op == I.MM for p in comp.programs[0]) == 2
    r = launch(k, Config(), x=Input(x), w=Weight(w), out=Output((11, 16)))
    assert np.linalg.norm(r.outputs["out"] - x @ w.T) / np.linalg.norm(x @ w.T) < 0.02


def test_stationary_overwritten_inside_loop_is_an_error():
    @ol.jit
    def k(x, w, out):
        cfg_blocks = 64
        xs = ol.quantize(ol.load(x))                      # 2 blocks at ACT RAM [0, 2)
        acc = ol.zeros([1, 8])
        for i in ol.range(2):
            big = ol.zeros([1, 32 * (cfg_blocks - 1)])
            ol.quantize(big)                              # wraps the ACT RAM ring over xs
            acc.set(acc + ol.dot(xs, w))
        ol.store(out, acc)

    with pytest.raises(CompileError):
        compile_kernel(k, Config(), x=Input(np.zeros((1, 64), np.float32)),
                       w=Weight(np.zeros((8, 64), np.float32)), out=Output((1, 8)))


def test_a_build_given_up_inside_loops_leaves_them_open():
    """A build given up inside nested ol.range loops (as the generate loop's fallback to its
    split form gives up one too deeply nested) leaves their generators suspended; closing them
    later, in any order (garbage collection: the outer one first here), does not end the loops
    on the abandoned builder (end_loop would assert on its loop stack, an exception Python can
    only print)."""
    gens = []

    @ol.jit
    def k(x, out):
        for g in (ol.range(2), ol.range(2)):
            next(g)
            gens.append(g)
        raise CompileError("given up")

    with pytest.raises(CompileError, match="given up"):
        compile_kernel(k, Config(), x=Input(np.zeros((1, 8), np.float32)), out=Output((1, 8)))
    for g in gens:
        g.close()


def test_shape_errors():
    @ol.jit
    def k(x, out):
        t = ol.load(x)
        ol.store(out, t + ol.zeros([2, 2]))

    with pytest.raises(CompileError):
        compile_kernel(k, Config(), x=Input(np.zeros((3, 3), np.float32)), out=Output((3, 3)))


def test_program_is_spmd_over_slices():
    from opentpu.kernels import attention_decode
    rng = np.random.default_rng(0)
    comp, _ = compile_kernel(attention_decode, Config(S=2),
                             q=Input(rng.standard_normal((8, 64))),
                             kv=KVCache(rng.standard_normal((2, 200, 64)),
                                        rng.standard_normal((2, 200, 64)), 256),
                             out=Output((8, 64)), n_q_heads=8, n_kv_heads=2, seq_len=200, block=32)
    assert len(comp.programs) == 2
    assert all(any(p.op == I.LOOP for p in prog) for prog in comp.programs)


def _body_ops(prog):
    s = next(i for i, p in enumerate(prog) if p.op == I.LOOP)
    return [p.comment for p in prog[s + 1:s + 1 + prog[s].w[0]]]


def test_softmax_fuses_exp2_sub_and_elides_set_copies():
    @ol.jit
    def k(x, out):
        acc = ol.zeros([2, 8])
        for i in ol.range(2):
            t = ol.load(x[i])                          # [2, 8]
            m = ol.max(t, axis=1)
            acc.set(acc + ol.exp2(t - m[:, None]))     # inline temporaries: fuse + no copy
        ol.store(out, acc)

    x = np.random.default_rng(0).standard_normal((2, 2, 8)).astype(np.float32)
    comp, _ = compile_kernel(k, Config(), x=Input(x), out=Output((2, 8)))
    ops = _body_ops(comp.programs[0])
    assert "exp2sub" in ops and "copy" not in ops
    r = launch(k, Config(), x=Input(x), out=Output((2, 8)))
    want = sum(np.exp2(x[i] - x[i].max(1, keepdims=True)) for i in range(2))
    assert np.allclose(r.outputs["out"], want, rtol=1e-5)


@pytest.mark.parametrize("home", [False, True])
def test_a_home_tile_keeps_a_loops_head_load_off_its_body_tiles(home):
    """ol.load_home: a hardware loop's head load (a layer's norm gain) into a tile the program
    keeps. Next-fit placement wraps within the body, so with ol.load a later body tile lands on
    the gain's words (the next iteration's load then waits for that tile's last read: a false
    WAR); with load_home no instruction but the load writes there. The results are the same."""
    N, L = 256, 3

    @ol.jit
    def k(x, gain, out):
        acc = ol.load(x)
        for i in ol.range(L):
            g = ol.load_home(gain[i * N:(i + 1) * N], "g") if home else \
                ol.load(gain[i * N:(i + 1) * N])
            y = acc * g
            del g
            for j in range(12):                 # fresh tiles: next-fit walks on and wraps
                y = (y * 0.5) + float(j)
            acc.set(y)
        ol.store(out, acc)

    cfg = Config(TMEM_WORDS=4096)
    x = np.random.default_rng(0).standard_normal(N).astype(np.float32)
    gn = np.random.default_rng(1).standard_normal(L * N).astype(np.float32)
    prog = _prog(k, cfg, x=Input(x), gain=Input(gn), out=Output((N,)))
    (lo, n), = [(p.w[1], p.w[2]) for p in prog if p.op == I.LD and p.ra]     # the gain's
    on = [p for p in prog if p.op == I.VOP and p.w[0] < lo + n and
          lo < p.w[0] + (p.w[3] & 0xFFFF) * (p.w[3] >> 16)]
    assert bool(on) != home
    r = launch(k, cfg, x=Input(x), gain=Input(gn), out=Output((N,)))
    want = x.astype(np.float32)
    for i in range(L):
        want = want * gn[i * N:(i + 1) * N]
        for j in range(12):
            want = want * np.float32(0.5) + np.float32(j)
    assert np.allclose(r.outputs["out"], want, rtol=1e-5)


def test_named_temporaries_are_not_clobbered():
    @ol.jit
    def k(x, out):
        t = ol.load(x)
        d = t - 1.0                     # named: must keep its value after exp2(d)
        e = ol.exp2(d)
        acc = ol.zeros([2, 8])
        s = acc + e                     # named: acc.set(s) must copy, s stays usable
        acc.set(s)
        ol.store(out, acc + s + d)

    x = np.random.default_rng(1).standard_normal((2, 8)).astype(np.float32)
    comp, _ = compile_kernel(k, Config(), x=Input(x), out=Output((2, 8)))
    ops = [p.comment for p in comp.programs[0]]
    assert "exp2sub" not in ops and "copy" in ops
    r = launch(k, Config(), x=Input(x), out=Output((2, 8)))
    assert np.allclose(r.outputs["out"], 2 * np.exp2(x - 1) + (x - 1), rtol=1e-5)


def test_an_add_in_place_is_the_set_of_a_sum_without_its_temp():
    """x.add(y) (Tile.add: the DeltaNet layers' residual, qwen35._residual) is the ADD that
    x.set(x + y) ends with once set moves it into x, and the next tiles go where they went
    after x + y's temp (Builder.skip): the same program, x.set(x.add(y)) adding nothing. Where
    TMEM has no room for that temp only the add compiles."""
    N = 256

    @ol.jit
    def k(x, y, out, form: str, hold: int):
        a, b = ol.load(x), ol.load(y)
        h = ol.zeros([hold]) if hold else None
        a.set(a + b if form == "set" else a.add(b))
        del b, h
        ol.store(out, a * 2.0 + 1.0)            # later tiles: placed after the add

    rng = np.random.default_rng(0)
    xs, ys = (rng.standard_normal(N).astype(np.float32) for _ in range(2))
    args = dict(x=Input(xs), y=Input(ys), out=Output((N,)))
    progs = {f: _prog(k, Config(TMEM_WORDS=4096), **args, form=f, hold=0)
             for f in ("set", "add")}
    assert np.array_equal(I.assemble(progs["set"]), I.assemble(progs["add"]))
    add = next(p for p in progs["add"] if p.comment == "add")
    assert add.w[0] == add.w[1] and "copy" not in [p.comment for p in progs["add"]]  # into a
    cfg = Config(TMEM_WORDS=1024)                # a, b and 300 held words: no room for a + b
    with pytest.raises(CompileError, match="TMEM exhausted"):
        _prog(k, cfg, **args, form="set", hold=300)
    r = launch(k, cfg, **args, form="add", hold=300)
    assert np.array_equal(r.outputs["out"], (xs + ys) * np.float32(2) + np.float32(1))


# ------------------------------------------------------------------ fusions into the hardware
def _prog(kernel, cfg, **args):
    comp, _ = compile_kernel(kernel, cfg, **args)
    return comp.programs[0]


def test_rmsnorm_quantize_is_one_rssq_pass_and_one_qact():
    from opentpu.kernels.lib import rmsnorm

    @ol.jit
    def k(x, g, w, out):
        y = ol.dot(rmsnorm(ol.load(x), ol.load(g), 1e-6), w)
        ol.store(out, y)

    rng = np.random.default_rng(0)
    args = dict(x=Input(rng.standard_normal((4, 64))), g=Input(rng.standard_normal(64)),
                w=Weight(rng.standard_normal((32, 64))), out=Output((4, 32)))
    prog = _prog(k, Config(), **args)
    vops = [(p.w[5] >> 16) & 0xFF for p in prog if p.op == I.VOP]
    full = [p for p in prog if p.op == I.VOP and (p.w[3] >> 16) == 64]
    assert I.V_RSSQ in vops and len(full) == 1            # only the sum of squares touches x
    q = next(p for p in prog if p.op == I.QACT)
    assert q.flags & I.F_CSCALE and q.flags & I.F_RSCALE  # gamma and 1/rms applied by QACT
    ref = launch(k, Config(), **args).outputs["out"]
    xs = args["x"].array
    xn = xs / np.sqrt((xs * xs).mean(1, keepdims=True) + 1e-6) * args["g"].array
    assert np.abs(ref - xn @ args["w"].array.T).max() < 0.05 * np.abs(ref).max()


def test_flash_attention_uses_mxu_epilogue_fusions():
    from opentpu.kernels import attention_decode
    rng = np.random.default_rng(0)
    prog = _prog(attention_decode, Config(),
                 q=Input(rng.standard_normal((4, 64))),
                 kv=KVCache(rng.standard_normal((1, 256, 64)), rng.standard_normal((1, 256, 64)),
                            256),
                 out=Output((4, 64)), n_q_heads=4, n_kv_heads=1, seq_len=256, block=32)
    mms = [p for p in prog if p.op == I.MM]
    assert any(p.flags & I.F_RMAX for p in mms)            # softmax row max from the MXU
    assert any(p.flags & I.F_ASCALE for p in mms)          # acc*alpha + P.V in the MXU
    qacts = [p for p in prog if p.op == I.QACT]
    assert all(p.flags & I.F_CSCALE for p in qacts)       # q scale and V scales in the quantizer
    funcs = [(p.w[5] >> 16) & 0xFF for p in prog if p.op == I.VOP]
    assert I.V_RMAX not in funcs
    body = prog[next(i for i, p in enumerate(prog) if p.op == I.LOOP) + 1:]
    assert [p.op for p in body[:40]].count(I.MM) >= 4     # two blocks per loop body (pipelined)


def test_dot_rowmax_peephole_and_explicit_form_agree():
    @ol.jit
    def k(x, w, out1, out2):
        xs = ol.quantize(ol.load(x))
        s = ol.dot(xs, w)
        ol.store(out1, ol.max(s, axis=1))                  # peephole: RMAX on that MM
        t = ol.empty([4, 32])
        ol.dot(xs, w, out=t, rowmax=True)
        ol.store(out2, t.rowmax)

    rng = np.random.default_rng(1)
    args = dict(x=Input(rng.standard_normal((4, 64))), w=Weight(rng.standard_normal((32, 64))),
                out1=Output((4,)), out2=Output((4,)))
    prog = _prog(k, Config(), **args)
    assert all(p.flags & I.F_RMAX for p in prog if p.op == I.MM)
    assert not any(p.op == I.VOP and (p.w[5] >> 16) & 0xFF == I.V_RMAX for p in prog)
    r = launch(k, Config(), **args).outputs
    assert np.array_equal(r["out1"], r["out2"])


def test_act_and_tmem_are_reused_once_dead():
    @ol.jit
    def k(x, w, out):
        acc = ol.zeros([1, 32])
        for _ in ol.static_range(40):                     # 40 x (TMEM temp + ACT operand)
            ol.dot(ol.load(x) * 2.0, w, acc=acc)
        ol.store(out, acc)

    rng = np.random.default_rng(2)
    cfg = Config(TMEM_WORDS=2048, ACT_BLOCKS=8)
    args = dict(x=Input(rng.standard_normal((1, 128))), w=Weight(rng.standard_normal((32, 128))),
                out=Output((1, 32)))
    r = launch(k, cfg, **args)                             # would not fit without reuse
    want = 80 * (args["x"].array @ args["w"].array.T)
    assert np.abs(r.outputs["out"] - want).max() < 0.05 * np.abs(want).max()


def test_a_top_tile_is_kept_apart_from_next_fit():
    """alloc(top=True) takes the highest free words and leaves next-fit's cursor where it was,
    so a small tile kept for the whole program (attention's variant table) does not split the
    gaps next-fit turns through (888fbb98: the 35B's sampled decode loop at pos 2000 and 4000,
    top-k 64, found no gap for its head's 8193-word logits chunks)."""
    seen = {}

    @ol.jit
    def k(x, out):
        b = current()
        a = ol.load(x)
        at = b.tmem_next
        t = b.alloc((2,), top=True)
        u = b.alloc((3,), top=True)
        seen.update(a=a.base, at=at, next=b.tmem_next, t=t.base, u=u.base, b=ol.load(x).base)
        t.set(1.0)
        u.set(2.0)
        ol.store(out, a + ol.load(x))

    cfg = Config(TMEM_WORDS=2048)
    rng = np.random.default_rng(3)
    _prog(k, cfg, x=Input(rng.standard_normal((16,))), out=Output((16,)))
    top = 2048 - (I.STREAM_DESC_AREA if cfg.STREAM else 0)
    assert seen["t"] == top - 2 and seen["u"] == top - 5
    assert seen["next"] == seen["at"] and seen["b"] == seen["a"] + 16

def _released_kernel(n_keys: int, late_arg: bool):
    """Two arguments of tok, then tok released; a loop holding n_keys address registers at
    once; then (late_arg) an argument of tpos."""
    X = Tensor(Affine(0), (4096,), (1,))
    tok, tpos = RunVar("tok"), RunVar("tpos")

    @ol.jit
    def k(out):
        acc = ol.load(X[tok * 4:tok * 4 + 4]) + ol.load(X[tok * 8:tok * 8 + 4])
        ol.release(tok)
        for i in ol.range(2):
            for j in range(n_keys):             # a register per j: address (j + 1) 32 i + 4 j
                acc.set(acc + ol.load(X[(j + 1) * 32 * i + 4 * j:(j + 1) * 32 * i + 4 * j + 4]))
        if late_arg:
            acc = acc + ol.load(X[tpos * 4:tpos * 4 + 4])
        ol.store(out, acc)

    def want(xs, t, p):
        acc = xs[4 * t:4 * t + 4] + xs[8 * t:8 * t + 4]
        for i in range(2):
            for j in range(n_keys):
                acc = acc + xs[(j + 1) * 32 * i + 4 * j:(j + 1) * 32 * i + 4 * j + 4]
        return acc + (xs[4 * p:4 * p + 4] if late_arg else 0)
    return k, want


@pytest.mark.parametrize("n_keys,late_arg", [(13, True), (15, False)])
def test_released_argument_registers(n_keys, late_arg):
    """ol.release(tok): tok's argument registers serve the rest of the program. 13 loop
    addresses take R1-R13, so tpos's argument (R13 at the start) moves into a released register
    at the release; 15 take the two released registers too (zeroed before the loop)."""
    cfg = Config(DRAM_BYTES=1 << 16)
    k, want = _released_kernel(n_keys, late_arg)
    out = Tensor(Affine(1 << 15), (4,), (1,))
    b = k.trace(cfg, 0, {"out": out})
    prog = b.finish()
    xs = np.random.default_rng(1).standard_normal(4096).astype(np.float32)
    for t, p in ((3, 5), (7, 1)):
        dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
        dram[:4 * 4096] = xs.view(np.uint8)
        vals = {"tok": t, "tpos": p}
        m = Machine(cfg, [prog], [dram], args=arg_words(b.run_args, vals)).run()
        got = m.slices[0].dram[1 << 15:(1 << 15) + 16].view(np.float32)
        assert np.array_equal(got, want(xs, t, p).astype(np.float32)), (t, p)
    k2, _ = _released_kernel(15, True)
    with pytest.raises(CompileError, match="R13 is taken"):
        k2.trace(cfg, 0, {"out": out})


def _run_out(cfg, b, prog, xs, vals, out):
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    dram[:4 * len(xs)] = xs.view(np.uint8)
    args = arg_words(b.run_args, vals) if b.run_args else None
    m = Machine(cfg, [prog], [dram], args=args).run()
    return m.slices[0].dram[out:out + 16].view(np.float32)


def test_a_run_time_address_does_not_take_a_register_freed_in_a_live_loop():
    """A register freed inside a loop that is still running (its earlier use, in the loop's
    body, starts from 0) must not take an address that starts at a run-time value: its init
    (ADDI r, R_arg, 0) before the inner loop would leave c * tok in it at the end of the body,
    for the earlier use in the next iteration."""
    tok = RunVar("tok", bound=16)
    cfg = Config(DRAM_BYTES=1 << 16)
    X = Tensor(Affine(0), (4096,), (1,))

    @ol.jit
    def k():
        acc = ol.zeros([4])
        for i in ol.range(2):
            for j in ol.range(1):                   # j only: takes R1, frees it in i's body
                acc.set(acc + ol.load(X[j * 8:j * 8 + 4]))
            for j in ol.range(1):                   # tok + j
                acc.set(acc + ol.load(X[tok * 8 + j * 8:tok * 8 + j * 8 + 4]))
        ol.store(Tensor(Affine(1 << 15), (4,), (1,)), acc)

    b = k.trace(cfg, 0, {})
    xs = np.arange(4096, dtype=np.float32)
    got = _run_out(cfg, b, b.finish(), xs, {"tok": 5}, 1 << 15)
    assert np.array_equal(got, 2 * (xs[0:4] + xs[40:44]))


def test_a_borrowed_spare_comes_back_from_its_twin():
    """Builder.borrow with no register free (a step over slots' attention, RunSpan): a spare (an
    inner loop's register, its outer terms left) whose terms a live register holds too; the
    caller steps it in a loop of a run-time count (no reset after it), and give_back copies the
    twin into it, so the inner loop's next pass reads where it would have. Transient run words
    (Builder.transient): c * var alone in a scratch register for its one instruction, no
    argument register."""
    from opentpu.compiler import DevVar
    tok = RunVar("tok", bound=8)
    cfg = Config(DRAM_BYTES=1 << 16)
    X = Tensor(Affine(0), (4096,), (1,))
    W = Tensor(Affine(1 << 14), (8,), (1,))             # the run words: tok, the count

    @ol.jit
    def k():
        b = current()
        w = ol.load(W)
        b.run_words, b.transient = {"tok": w.base}, True
        acc = ol.zeros([4])
        acc.set(acc + ol.load(X[tok * 4:tok * 4 + 4]))   # tok alone: transient
        for i in ol.range(3):
            acc.set(acc + ol.load(X[i * 16:i * 16 + 4]))     # {i}: live
            for j in ol.range(2):                           # {i, j}: a spare {i} after j
                acc.set(acc * 2.0 + ol.load(X[64 + i * 16 + j * 4:64 + i * 16 + j * 4 + 4]))
            c = b.scratch()
            b.emit(I.rld(c, w.base + 1, bound=(1, 4)))
            held = []
            while b.free_regs:
                held.append(b.scratch())
            r, how = b.borrow()
            assert how is not None
            b.emit(I.addi(r, 0, 4 * 256))
            lp = b.begin_loop(0, rcount=c)
            b.unscratch(c)
            acc.set(acc * 3.0 + ol.load(Tensor(DevVar("v", r) + 0, (4,), (1,))))
            b.emit(I.addi(r, r, 16))
            b.end_loop(lp)
            b.give_back(r, how)
            for h in held:
                b.unscratch(h)
        ol.store(Tensor(Affine(1 << 15), (4,), (1,)), acc)
        b.run_words, b.transient = None, False

    b = k.trace(cfg, 0, {})
    prog = b.finish()
    assert not b.run_args and any("spare register back" in x.comment for x in prog)
    xs = np.arange(4096, dtype=np.float32) % 7 - 3
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    dram[:4 * len(xs)] = xs.view(np.uint8)
    dram[1 << 14:(1 << 14) + 8] = np.array([5.0, 2.0], np.float32).view(np.uint8)
    m = Machine(cfg, [prog], [dram]).run()
    got = m.slices[0].dram[1 << 15:(1 << 15) + 16].view(np.float32)
    want = xs[20:24].copy()
    for i in range(3):
        want = want + xs[i * 16:i * 16 + 4]
        for j in range(2):
            want = want * 2 + xs[64 + i * 16 + j * 4:64 + i * 16 + j * 4 + 4]
        for g in range(2):
            want = want * 3 + xs[256 + g * 4:256 + g * 4 + 4]
    assert np.array_equal(got, want)


@pytest.mark.parametrize("head", [True, False])
def test_a_run_time_word_load_in_the_token_loop(head):
    """The generate loop's form (run_words): the eight argument registers taken, c * tpos
    alone is loaded from tpos's TMEM word into an address register where it is used (RLD MUL).
    It must not take a register a layer loop before it freed in the token loop's body (that
    loop's addresses would start from c * tpos in the next token), unless the body zeroes that
    register first: the token count's register given back at the body's head (head, as
    generate.py does; LFM2.5-8B-A1B's generate programs take it)."""
    cfg = Config(DRAM_BYTES=1 << 16)
    X = Tensor(Affine(0), (4096,), (1,))
    args = [RunVar(f"a{i}", bound=16) for i in range(8)]
    tpos = RunVar("tpos", bound=16)
    vals = {**{f"a{i}": i + 1 for i in range(8)}, "tpos": 5}
    count = []

    @ol.jit
    def k():
        b = current()
        acc = ol.zeros([4])
        w = ol.full([1], float(vals["tpos"]))
        r = b.scratch()
        count.append(r)
        b.rld(r, ol.full([1], 2.0))
        g = b.begin_loop(0, rcount=r)                # the token loop, 2 tokens
        if head:
            b.unscratch(r)
        b.run_words = {"tpos": w.base}
        for v in args:                               # the argument registers, all taken
            acc.set(acc + ol.load(X[v * 4:v * 4 + 4]))
        for i in ol.range(2):                        # a layer loop: takes a register, frees it
            acc.set(acc + ol.load(X[i * 8:i * 8 + 4]) * 10.0)
        acc.set(acc + ol.load(X[tpos * 4:tpos * 4 + 4]) * 100.0)    # an RLD MUL of tpos's word
        b.end_loop(g)
        b.run_words = None
        if not head:
            b.unscratch(r)
        ol.store(Tensor(Affine(1 << 15), (4,), (1,)), acc)

    b = k.trace(cfg, 0, {})
    prog = b.finish()
    xs = np.arange(4096, dtype=np.float32)
    one = sum(xs[4 * (i + 1):4 * (i + 1) + 4] for i in range(8)) + \
        10 * (xs[0:4] + xs[8:12]) + 100 * xs[20:24]
    assert np.array_equal(_run_out(cfg, b, prog, xs, vals, 1 << 15), 2 * one)
    rld = next(p for p in prog if p.op == I.RLD and p.flags & I.F_MUL)
    assert (rld.rd == count[0]) == head          # the zeroed count register, or another


@pytest.mark.parametrize("flag", [0, 1, 2])
def test_a_register_set_inside_a_guard_is_not_counted_on_after_it(flag):
    """A run-time address first used inside a device-count loop (a guard: 0, 1 or 2 times)
    is initialized there; after the guard its register holds c * tok only if the guard ran,
    so a later address of the same run-time term starts from an init of its own."""
    tok = RunVar("tok", bound=16)
    cfg = Config(DRAM_BYTES=1 << 16)
    X = Tensor(Affine(0), (4096,), (1,))

    @ol.jit
    def k():
        b = current()
        acc = ol.zeros([4])
        f = ol.full([1], float(flag))
        r = b.scratch()
        b.rld(r, f)
        g = b.begin_loop(0, rcount=r)
        for j in ol.range(1):
            acc.set(acc + ol.load(X[tok * 8 + j * 8:tok * 8 + j * 8 + 4]) * 1000.0)
        b.end_loop(g)
        b.unscratch(r)
        for j in ol.range(1):                       # the same run-time term after the guard
            acc.set(acc + ol.load(X[tok * 8 + j * 8:tok * 8 + j * 8 + 4]))
        ol.store(Tensor(Affine(1 << 15), (4,), (1,)), acc)

    b = k.trace(cfg, 0, {})
    xs = np.arange(4096, dtype=np.float32)
    got = _run_out(cfg, b, b.finish(), xs, {"tok": 5}, 1 << 15)
    assert np.array_equal(got, xs[40:44] * (1000.0 * flag + 1))


def test_a_released_argument_taken_by_scratch_keeps_it_until_then():
    """scratch() of a released argument's register zeroes it there: the program's start must
    not zero it (it holds the argument for the addresses before the release)."""
    tok = RunVar("tok", bound=16)
    cfg = Config(DRAM_BYTES=1 << 16)
    X = Tensor(Affine(0), (4096,), (1,))

    @ol.jit
    def k():
        b = current()
        acc = ol.zeros([4])
        acc.set(acc + ol.load(X[tok * 8:tok * 8 + 4]))
        ol.release(tok)
        rs = [b.scratch() for _ in range(15)]       # the last one is tok's register
        for r in rs:
            b.unscratch(r)
        ol.store(Tensor(Affine(1 << 15), (4,), (1,)), acc)

    b = k.trace(cfg, 0, {})
    xs = np.arange(4096, dtype=np.float32)
    assert np.array_equal(_run_out(cfg, b, b.finish(), xs, {"tok": 5}, 1 << 15), xs[40:44])


NX = 1 << 14


def _fuzz_tree(rng, nvars):
    """A random kernel: nested hardware loops (1-3 iterations), guards (device-computed counts
    0-2, their own count register freed inside or after them), device values (DevVar, a
    scratch register), scratch registers held across a part, top-level releases of run-time
    values, and loads at affine addresses (loop terms, at most one run-time or device term);
    in random places the shapes of the two run-time register hazards (a run-time address in a
    loop after a sibling loop freed a register in the same body; one inside a guard and again
    after it)."""
    avail = set(range(nvars))
    budget = [rng.randint(6, 30)]

    def gen(depth, loops, dvs):
        items = []
        for _ in range(rng.randint(1, 5)):
            if budget[0] <= 0:
                break
            budget[0] -= 1
            x = rng.random()
            if depth == 0 and avail and x < 0.08:
                v = rng.choice(sorted(avail))
                avail.discard(v)
                items.append(("rel", v))
            elif depth < 3 and avail and x < 0.2:      # the hazards' shapes, in random parts
                v, l1, l2 = rng.choice(sorted(avail)), rng.randrange(1 << 30), \
                    rng.randrange(1 << 30)
                ld = lambda l, run: ("ld", rng.randint(0, 200) * 4,       # noqa: E731
                                     {l: 4 * rng.randint(1, 2)}, run, float(rng.randint(1, 5)))
                inner = [("loop", l1, rng.randint(1, 2), [ld(l1, None)]),
                         ("loop", l2, rng.randint(1, 2), [ld(l2, ("var", v, 4))])]
                if rng.random() < 0.5:
                    lid = rng.randrange(1 << 30)
                    items.append(("loop", lid, rng.randint(2, 3), inner))
                else:
                    items.append(("guard", rng.randint(0, 2), rng.random() < 0.5, inner[1:]))
                    items.append(inner[1])
            elif depth < 4 and x < 0.42:
                lid = rng.randrange(1 << 30)
                items.append(("loop", lid, rng.randint(1, 3), gen(depth + 1, loops + [lid], [])))
            elif depth < 4 and x < 0.5:
                items.append(("guard", rng.randint(0, 2), rng.random() < 0.5,
                              gen(depth + 1, loops, dvs)))
            elif x < 0.54:
                did = rng.randrange(1 << 30)
                items.append(("dv", did, 4 * rng.randint(0, 50), gen(depth, [], dvs + [did])))
            elif x < 0.57:
                items.append(("scr", gen(depth, loops, dvs)))
            else:
                terms = {l: 4 * rng.randint(1, 2) for l in loops if rng.random() < 0.6}
                run = None
                if avail and rng.random() < 0.5:
                    run = ("var", rng.choice(sorted(avail)), 4)
                elif dvs and rng.random() < 0.4:
                    run = ("dv", rng.choice(dvs), 1)
                items.append(("ld", rng.randint(0, 200) * 4, terms, run, float(rng.randint(1, 5))))
        return items
    return gen(0, [], [])


def _fuzz_one(seed):
    import random
    rng = random.Random(seed)
    nvars = rng.randint(0, 3)
    tree = _fuzz_tree(rng, nvars)
    words = rng.random() < 0.3              # the generate loop's form: run-time values in TMEM
    vars_ = [RunVar(f"v{i}", bound=64) for i in range(nvars)]
    vals = {f"v{i}": rng.randrange(64) for i in range(nvars)}
    cfg = Config(DRAM_BYTES=1 << 17)
    out = 4 * NX
    from opentpu.compiler import DevVar

    @ol.jit
    def k():
        b = current()
        X = Tensor(Affine(0), (NX,), (1,))
        acc = ol.zeros([4])
        if words:
            w = ol.empty([max(nvars, 1)])
            for i in range(nvars):
                w[i:i + 1].set(float(vals[f"v{i}"]))
            b.run_words = {f"v{i}": w.base + i for i in range(nvars)}
        ivs, dvs = {}, {}

        def emit(items):
            for it in items:
                if it[0] == "rel":
                    ol.release(vars_[it[1]])
                elif it[0] == "loop":
                    for i in ol.range(it[2]):
                        ivs[it[1]] = i
                        emit(it[3])
                elif it[0] == "guard":
                    r = b.scratch()
                    b.rld(r, ol.full([1], float(it[1])))
                    g = b.begin_loop(0, rcount=r)
                    if it[2]:
                        b.unscratch(r)                  # the count is read at the start
                    emit(it[3])
                    b.end_loop(g)
                    if not it[2]:
                        b.unscratch(r)
                elif it[0] == "dv":
                    r = b.scratch()
                    b.rld(r, ol.full([1], float(it[2])))
                    dvs[it[1]] = DevVar(f"d{it[1]}", r)
                    emit(it[3])
                    b.unscratch(r)
                elif it[0] == "scr":
                    r = b.scratch()
                    emit(it[1])
                    b.unscratch(r)
                else:
                    _, c, terms, run, m = it
                    a = Affine(c)
                    for l, co in terms.items():
                        a = a + ivs[l] * co
                    T = X
                    if run is not None and run[0] == "var":
                        a = a + vars_[run[1]] * run[2]
                    elif run is not None:                   # a byte address term
                        T = Tensor(Affine(0) + dvs[run[1]], (NX,), (1,))
                    acc.set(acc + ol.load(T[a:a + 4]) * m)
        emit(tree)
        ol.store(Tensor(Affine(out), (4,), (1,)), acc)

    try:
        b = k.trace(cfg, 0, {})
        prog = b.finish()
    except CompileError as e:
        if "registers" in str(e) or "argument" in str(e):
            return None                     # out of registers: a program too big to test
        raise
    xs = np.random.default_rng(seed).integers(0, 16, NX).astype(np.float32)
    acc = np.zeros(4, np.float64)

    def ref(items, ivs, dvs):
        nonlocal acc
        for it in items:
            if it[0] == "loop":
                for i in range(it[2]):
                    ref(it[3], {**ivs, it[1]: i}, dvs)
            elif it[0] == "guard":
                for _ in range(it[1]):
                    ref(it[3], ivs, dvs)
            elif it[0] == "dv":
                ref(it[3], ivs, {**dvs, it[1]: it[2]})
            elif it[0] == "scr":
                ref(it[1], ivs, dvs)
            elif it[0] == "ld":
                _, c, terms, run, m = it
                a = c + sum(ivs[l] * co for l, co in terms.items())
                if run is not None:
                    a += vals[f"v{run[1]}"] * run[2] if run[0] == "var" else dvs[run[1]] // 4
                acc = acc + xs[a:a + 4] * m
    ref(tree, {}, {})
    got = _run_out(cfg, b, prog, xs, vals, out)
    return np.array_equal(got, acc.astype(np.float32))


def test_register_allocation_fuzz():
    """Random kernels (_fuzz_tree) against a Python model of their loads: address registers
    reused across loops, guards, device values, scratch registers and released run-time
    values give the model's sums exactly."""
    bad = [s for s in range(600) if _fuzz_one(s) is False]
    assert not bad, f"seeds {bad[:10]}"


@pytest.mark.parametrize("words", [False, True], ids=["argument", "run_words"])
def test_a_counter_steps_where_the_program_says(words):
    """compiler.Counter (Builder.counter, step, end_counter; tiled._lfm2_stack's layer block):
    five layers in a loop, each one's kind from a TMEM table, the counter stepped by its kind's
    block size (256 or 128 bytes) in loops of 0 or 1 iterations after the layer's loads. Each
    layer loads at the counter alone and, in an inner loop, at the counter plus a run-time value
    (Builder._derived: a register of their sum stepped with it, from tok's argument register or,
    under run_words, an RLD MUL of tok's TMEM word): the words of every layer's block, for two
    values of tok. Stepping it while a loop's address adds it is refused."""
    tok = RunVar("tok", bound=16)
    cfg = Config(DRAM_BYTES=1 << 16)
    kinds = [1, 0, 0, 1, 0]
    size = {1: 256, 0: 128}

    def kernel(t, early=False):
        @ol.jit
        def k():
            b = current()
            if words:
                b.run_words = {"tok": ol.full([1], float(t)).base}
            tab = ol.zeros([10])                    # [kind 1? | kind 0?]
            for i, kd in enumerate(kinds):
                tab[i + 5 * (1 - kd):i + 5 * (1 - kd) + 1].set(1.0)
            acc = ol.zeros([4])
            blk = b.counter("block")
            r = [b.scratch(), b.scratch()]
            for it in ol.range(5):
                for o in (0, 1):
                    ra, imm = b.addr(Affine(tab.base + 5 * o) + Affine.of(it))
                    b.emit(I.rld(r[o], imm, ra=ra, bound=(0, 1)))
                acc.set(acc + ol.load(Tensor(Affine(0) + blk, (4,), (1,))))
                for j in ol.range(2):
                    acc.set(acc + ol.load(Tensor(Affine(2048) + blk + tok * 16 + j * 64, (4,),
                                                 (1,))))
                    if early:
                        b.step(blk, 16)
                for o, kd in ((0, 1), (1, 0)):
                    lp = b.begin_loop(0, rcount=r[o])
                    b.step(blk, size[kd])
                    b.end_loop(lp)
            for x in r:
                b.unscratch(x)
            b.end_counter(blk)
            ol.store(Tensor(Affine(1 << 15), (4,), (1,)), acc)
        return k

    xs = np.random.default_rng(3).integers(0, 16, 4096).astype(np.float32)
    for t in (3, 9):
        b = kernel(t).trace(cfg, 0, {})
        prog = b.finish()
        steps = [i for i in prog if i.op == I.ADDI and i.comment in ("block + 256", "block + 128")]
        assert len(steps) == 4                  # the counter's and its sum with 16 * tok, twice
        got = _run_out(cfg, b, prog, xs, {"tok": t}, 1 << 15)
        want, blk = np.zeros(4), 0
        for kd in kinds:
            want = want + xs[blk // 4:blk // 4 + 4]
            for j in range(2):
                a = (2048 + blk + 16 * t + 64 * j) // 4
                want = want + xs[a:a + 4]
            blk += size[kd]
        assert np.array_equal(got, want.astype(np.float32)), t
    with pytest.raises(CompileError, match="stepped while a loop's address adds it"):
        kernel(3, early=True).trace(cfg, 0, {})


# ------------------------------------------------------------------------- AW > 32 (the BCU's 36)
def _run_segments(cfg, kernel, args, weights_at=None, **kw):
    """compile_kernel's steps with the weights at `weights_at`, run on the ISA simulator from
    the images' written areas alone (a DRAM above 4 GiB: isasim.dram_zeros' sparse file)."""
    from opentpu.runtime import collect, place
    dram, layout, bound = place(cfg, args, weights_at=weights_at)
    b = kernel.trace(cfg, 0, {**bound[0], **kw})
    prog = b.finish()
    m = Machine(cfg, [prog], [None])
    img = dram.images[0]
    m.slices[0].dram[:dram.next] = img[:dram.next]
    if weights_at is not None:
        m.slices[0].dram[weights_at:dram.high_next] = img[weights_at:dram.high_next]
    m.run()
    from opentpu.compiler import Compiled
    return prog, collect(Compiled(cfg, [prog], [b.stores], layout), [m.slices[0].dram], [], {})


AW_CFG = Config(AW=36, DRAM_BYTES=1 << 34, D=32, MCOLS=4)


def test_weights_above_4gib_through_the_window_and_biased_registers():
    """docs/compiler.md "Addresses above 4 GiB": the same kernel with its weights at 0 and at
    9 GiB + 1 MiB gives the same outputs bit for bit. A static MM there takes the window
    register (LI {w2, w1} where it moves: once for two MMs 4 GiB within, again after a loop),
    a loop's MMs a register biased by the window's base (LI before the loop, stepped and reset
    as any); every offset a 32-bit word."""
    @ol.jit
    def k(x, w, v, out, out2):
        xs = ol.load(x)
        ol.store(out, ol.dot(xs, w[0:16, :]) + ol.dot(xs, w[16:32, :]))
        acc = ol.zeros([4, 8])
        for i in ol.range(4):
            acc.set(acc + ol.dot(xs, v[i * 8:(i + 1) * 8, :]))
        ol.store(out2, acc + ol.dot(xs, w[32:40, :]))

    rng = np.random.default_rng(1)
    args = dict(x=Input(rng.standard_normal((4, 64), np.float32)),
                w=Weight(rng.standard_normal((40, 64), np.float32)),
                v=Weight(rng.standard_normal((32, 64), np.float32)),
                out=Output((4, 16)), out2=Output((4, 8)))
    lo_prog, lo = _run_segments(AW_CFG, k, args)
    at = (9 << 30) + (1 << 20)
    hi_prog, hi = _run_segments(AW_CFG, k, args, weights_at=at)
    for name in ("out", "out2"):
        assert np.array_equal(lo.outputs[name].view(np.uint32), hi.outputs[name].view(np.uint32))
    lis = [p for p in hi_prog if p.op == I.LI and p.w[1]]
    assert [x.comment for x in lis].count("window") == 2       # the first MMs; after the loop
    assert any(x.comment == "address base" for x in lis)        # the loop's register
    mms = [p for p in hi_prog if p.op == I.MM]
    assert all(p.ra for p in mms) and all(p.w[0] < 1 << 32 for p in mms)
    # the low layout at AW 36 is AW 32's program but for the negative ADDIs' w2
    p32, _ = _run_segments(Config(D=32, MCOLS=4, DRAM_BYTES=1 << 24), k, args)
    assert len(p32) == len(lo_prog) and all(p.w[1] == 0 for p in p32 if p.op in (I.LI, I.ADDI))
    assert all(x.op == y.op and (x.w[0] == y.w[0] and x.w[2:] == y.w[2:] if x.op == I.ADDI
                                 else x.w == y.w) for x, y in zip(p32, lo_prog))
    resets = [p for p in hi_prog if p.op == I.ADDI and "reset" in p.comment]
    assert resets and all(p.w[1] == 0xFFFFFFFF for p in resets)         # negative: w2 set


@pytest.mark.parametrize("regs", [15, 1])
def test_the_window_register_is_not_one_an_earlier_loop_of_the_body_steps(monkeypatch, regs):
    """A static MM above 4 GiB after an inner loop over rows below, both in an outer loop: the
    window register is not the inner loop's (stepped, reset and given back in the outer loop's
    body: a free register holds 0 where that body starts again, so the window's base in it
    would start the inner loop's next pass 4 GiB off; generate.py's token loop at Image.high
    read its KV rows from the weights). With that register the only one (regs 1) it is zeroed
    at the outer body's end instead. Low and high give the same outputs bit for bit."""
    from opentpu import compiler as C
    init = C.Builder.__init__

    def few(self, *a, **kw):
        init(self, *a, **kw)
        self.free_regs = self.free_regs[-regs:]
    monkeypatch.setattr(C.Builder, "__init__", few)

    @ol.jit
    def k(x, u, w, out):
        xs = ol.load(x)
        acc = ol.zeros([4, 8])
        for _ in ol.range(2):
            for i in ol.range(4):
                acc.set(acc + ol.load(u[i * 4:(i + 1) * 4, :]))
            acc.set(acc + ol.dot(xs, w[0:8, :]))
        ol.store(out, acc)

    rng = np.random.default_rng(3)
    args = dict(x=Input(rng.standard_normal((4, 64), np.float32)),
                u=Input(rng.standard_normal((16, 8), np.float32)),
                w=Weight(rng.standard_normal((8, 64), np.float32)), out=Output((4, 8)))
    _, lo = _run_segments(AW_CFG, k, args)
    hi_prog, hi = _run_segments(AW_CFG, k, args, weights_at=(9 << 30) + (1 << 20))
    assert any(p.comment == "window" for p in hi_prog)
    assert any(p.comment.startswith("register reset (the next") for p in hi_prog) == (regs == 1)
    assert np.array_equal(lo.outputs["out"].view(np.uint32), hi.outputs["out"].view(np.uint32))


def test_an_address_above_4gib_a_device_value_cannot_reach_is_an_error():
    """A device value (DevVar: its own register) plus a constant at or above 4 GiB: no register
    of the compiler's holds the base, so it is a CompileError, not a wrapped offset."""
    from opentpu.compiler import DevVar

    @ol.jit
    def k(out):
        r = current().scratch()
        ol.store(Tensor(Affine(5 << 30) + DevVar("slot", r), (8,), (1,)), ol.zeros([8]))
        ol.store(out, ol.zeros([8]))

    from opentpu.runtime import place
    _, _, bound = place(AW_CFG, dict(out=Output((8,))))
    with pytest.raises(CompileError, match="cannot be reached at AW 36"):
        k.trace(AW_CFG, 0, bound[0])


def test_runs_above_4gib_share_a_base_and_a_run_time_row_adds_it():
    """Two loops over weights above 4 GiB: the second takes the first's registers, which hold
    the base after it (spares: no second LI); a static MM inside a loop LIs the window in the
    body (each iteration); a run-time row (c * pos, an argument register) with a loop and the
    base: its register starts at ADDI R_arg + base before the loop."""
    @ol.jit
    def k(x, w, v, out, pos):
        xs = ol.load(x)
        acc = ol.zeros([4, 8])
        for i in ol.range(2):
            acc.set(acc + ol.dot(xs, v[i * 8:(i + 1) * 8, :]) + ol.dot(xs, w[0:8, :]))
        for i in ol.range(2):
            acc.set(acc + ol.dot(xs, v[16 + i * 8:24 + i * 8, :]))
        for i in ol.range(2):
            acc.set(acc + ol.dot(xs, w[pos * 8 + i * 8:pos * 8 + i * 8 + 8, :]))
        ol.store(out, acc)

    rng = np.random.default_rng(2)
    args = dict(x=Input(rng.standard_normal((4, 64), np.float32)),
                w=Weight(rng.standard_normal((40, 64), np.float32)),
                v=Weight(rng.standard_normal((32, 64), np.float32)), out=Output((4, 8)))
    pos = RunVar("pos", bound=3)
    res = {}
    for at in (None, (5 << 30) + (3 << 20)):
        prog, _ = _run_segments(AW_CFG, k, args, weights_at=at, pos=pos)
        words = arg_words(prog.run_args, {"pos": 2})
        from opentpu.runtime import collect, place
        from opentpu.compiler import Compiled
        dram, layout, bound = place(AW_CFG, args, weights_at=at)
        m = Machine(AW_CFG, [prog], [None], args=words)
        m.slices[0].dram[:dram.next] = dram.images[0][:dram.next]
        if at is not None:
            m.slices[0].dram[at:dram.high_next] = dram.images[0][at:dram.high_next]
        m.run()
        b = k.trace(AW_CFG, 0, {**bound[0], "pos": pos})
        res[at] = collect(Compiled(AW_CFG, [prog], [b.stores], layout), [m.slices[0].dram],
                          [], {}).outputs["out"]
        if at is not None:
            c = [p.comment for p in prog if p.op in (I.LI, I.ADDI) and p.w[1]]
            # (the rows' and the scale rows' registers: two keys, the second loop's the same)
            assert c.count("address base") == 2 and c.count("window") == 1
            assert any(p.op == I.ADDI and p.ra >= 8 and p.w[1] == 1 and "pos" in p.comment
                       for p in prog)
            body = next(j for j, p in enumerate(prog) if p.op == I.LOOP)
            assert any(p.comment == "window" for p in prog[body:body + prog[body].w[0] + 1])
    lo, hi = res.values()
    assert np.array_equal(lo.view(np.uint32), hi.view(np.uint32))


def test_a_negative_argument_above_aw_32_is_its_complement():
    """Above AW 32 an argument register holds the host's word zero-extended, so c * pos with
    c < 0 (a row counted down from the end) is the argument -c * (bound - 1 - pos), "pos~",
    the constant less -c (bound - 1): at AW 36, low and with the weights at 5 GiB (a bare
    argument's register plus a base: ADDI R_arg, B), every pos reads the rows AW 32 reads.
    progcache's run_var gives it back by name."""
    from opentpu.compiler import Comp, run_var
    from opentpu.runtime import collect, place
    from opentpu.compiler import Compiled

    @ol.jit
    def k(x, w, out, pos):
        xs = ol.load(x)
        ol.store(out, ol.dot(xs, w[16 + pos * -8:24 + pos * -8, :]))

    rng = np.random.default_rng(3)
    x, w = rng.standard_normal((4, 64), np.float32), rng.standard_normal((24, 64), np.float32)
    args = dict(x=Input(x), w=Weight(w), out=Output((4, 8)))
    pos = RunVar("pos", bound=3)
    res = {}
    c32 = Config(D=32, MCOLS=4, DRAM_BYTES=1 << 24)
    for cfg, at in ((c32, None), (AW_CFG, None), (AW_CFG, (5 << 30) + (3 << 20))):
        dram, layout, bound = place(cfg, args, weights_at=at)
        b = k.trace(cfg, 0, {**bound[0], "pos": pos})
        prog = b.finish()
        ra = prog.run_args            # (the rows' and their scales')
        if at is not None:            # R_arg + 5 GiB's base, the offset below 4 GiB
            assert sum(p.op == I.ADDI and p.ra >= 8 and p.w[1] == 1 for p in prog) == 2
        assert [(type(v), c) for v, c in ra] == ([(RunVar, -512), (RunVar, -64)] if cfg.AW == 32
                                                 else [(Comp, 512), (Comp, 64)])
        for p in range(3):
            words = arg_words(ra, {"pos": p})
            assert words[-2:] == [(c * (p if cfg.AW == 32 else 2 - p)) & 0xFFFFFFFF
                                  for v, c in ra[::-1]]
            m = Machine(cfg, [prog], [None], args=words)
            m.slices[0].dram[:dram.next] = dram.images[0][:dram.next]
            if at is not None:
                m.slices[0].dram[at:dram.high_next] = dram.images[0][at:dram.high_next]
            m.run()
            got = collect(Compiled(cfg, [prog], [b.stores], layout), [m.slices[0].dram],
                          [], {}).outputs["out"]
            res.setdefault(p, []).append(got.view(np.uint32))
            assert np.allclose(got, x @ w[16 - 8 * p:24 - 8 * p].T, rtol=0.1, atol=0.5)
    assert all(np.array_equal(r[0], y) for r in res.values() for y in r[1:])
    back = run_var(ra[0][0].name, ra[0][0].bound)
    assert (back.name, back.bound, back.value({"pos": 1})) == ("pos~", 3, 1)
    with pytest.raises(CompileError, match="needs pos's bound"):
        k.trace(AW_CFG, 0, {**place(AW_CFG, args)[2][0], "pos": RunVar("pos")})
