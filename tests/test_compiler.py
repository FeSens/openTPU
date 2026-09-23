"""Compiler lowering: loops, address registers, layouts, and compile-time errors."""
import numpy as np
import pytest

from opentpu import Config, isa as I
from opentpu import language as ol
from opentpu.compiler import CompileError
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
