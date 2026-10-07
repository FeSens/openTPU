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
