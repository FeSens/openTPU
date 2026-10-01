"""The on-card decode loop's ISA (docs/isa.md): VOP ARGMAX, RLD and HALT CHAIN on the ISA
simulator."""
import time
from pathlib import Path

import numpy as np
import pytest

from opentpu import Config, fp32 as F, isa as I
from opentpu.isasim import Machine, SimError
from opentpu.llm import generate as G


def f(x):
    return np.asarray(x, np.float32)


def run1(prog, tmem_init=None, dram=None, cfg=Config(S=1), args=None):
    m = Machine(cfg, [prog], [dram], args=args)
    for addr, v in (tmem_init or {}).items():
        m.slices[0].tput(addr + np.arange(np.size(v)), f(v).reshape(-1))
    return m.run().slices[0]


# ---------------------------------------------------------------------------------- ARGMAX
def test_argmax_first_column_of_the_total_order_max():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((5, 300)).astype(np.float32)
    a[1, [7, 40, 299]] = 9.0                       # ties: the first column
    a[2] = -np.inf
    a[2, 250] = -1e30
    a[3, :] = 0.0
    a[3, 100] = -0.0                               # -0 < +0: +0 at column 0 wins
    a[4, :] = -1.0
    a[4, 5], a[4, 9] = 1e-41, 0.0                  # a denormal flushes to +0: column 5 first
    s = run1([I.argmax(2000, 0, 5, 300, 2, 300), I.halt()], {0: a})
    out = s.tget(2000 + np.arange(10)).reshape(5, 2)
    assert list(out[:, 1].astype(int)) == [int(np.argmax(F._key(F.ftz(r)))) for r in a]
    assert list(out[:, 1].astype(int)) == [int(np.argmax(a[0])), 7, 250, 0, 5]
    assert np.array_equal(F.bits(out[:, 0]), F.bits(F.chain_max(a)))


def test_argmax_index_base_and_row_stride():
    a = f(np.arange(64)[::-1].reshape(2, 32))       # row maxima at column 0
    prog = [I.li(3, 1000), I.argmax(100, 0, 2, 32, drs=3, ars=32, base=-7, rd=3), I.halt()]
    s = run1(prog, {0: a})
    assert list(s.tget([100, 101, 103, 104])) == [63.0, 993.0, 31.0, 993.0]  # 0 + R3 + base
    assert s.tget([102])[0] == 0.0                       # the stride's gap is not written


def test_argmax_pairs_need_a_row_stride_and_no_self_overlap():
    a = f(np.arange(64).reshape(2, 32))
    with pytest.raises(SimError, match="drs"):
        run1([I.Instr(I.VOP, w=[100, 0, 0, 2 | (32 << 16), 1 | (32 << 16), I.V_ARGMAX << 16,
                                0]), I.halt()], {0: a})
    with pytest.raises(SimError, match="hazard"):
        run1([I.argmax(33, 0, 2, 32, ars=32), I.halt()], {0: a})   # row 0's pair at 33, 34


# ---------------------------------------------------------------------------------- RLD
@pytest.mark.parametrize("x,want", [(0.0, 0), (-0.0, 0), (0.99, 0), (1.0, 1), (151935.0, 151935),
                                    (-1020.0, -1020), (-1.5, -1), (2.0 ** 30 + 128, 2 ** 30 + 128),
                                    (2.0 ** 31, -2 ** 31), (-3e9, -2 ** 31), (np.inf, -2 ** 31),
                                    (np.nan, -2 ** 31), (1e-40, 0)])
def test_rld_truncates_to_int(x, want):
    s = run1([I.li(2, 10), I.rld(5, 7, ra=2), I.halt()], {17: [x]})
    assert s.R[5] == want & 0xFFFFFFFF


def test_rld_raw_and_r0():
    s = run1([I.rld(4, 0, raw=True), I.rld(0, 0), I.halt()], {0: [1.5]})
    assert s.R[4] == I.f32bits(1.5) and s.R[0] == 0


@pytest.mark.parametrize("x,raw,rb,mul,want", [
    (262143.0, False, 0, 9344, 262143 * 9344),                # past 2^31: a PLE row's bytes
    (-5.0, False, 0, -7, 35), (65536.0, False, 0, 65537, 65536),   # mod 2^32
    (3.0, True, 0, 3, 3 * 0x40400000), (100.0, False, 3, 5, 100 * 1005), (7.9, False, 0, 0, 0)])
def test_rld_mul_times_a_register_plus_w2_mod_2_32(x, raw, rb, mul, want):
    s = run1([I.li(3, 1000), I.rld(5, 0, raw=raw, mul=mul, rb=rb), I.halt()], {0: [x]})
    assert s.R[5] == want & 0xFFFFFFFF


def test_rld_drives_addresses_and_loop_counts():
    """A device-computed index picks a TMEM word (a gather), and a device-computed count runs a
    loop: the value of ARGMAX's index through RLD, as the decode loop uses them."""
    t = f([3.0, 1.0, 7.0, 2.0])
    prog = [I.argmax(10, 0, 1, 4),                                   # T[11] = 2.0
            I.vop(I.V_MUL, 12, 11, 0, 1, 1, 0, 0, 0, I.B_SCALAR, 8.0),  # 16 words per row
            I.rld(6, 12),
            I.vop(I.V_COPY, 20, 100, 0, 1, 1, 0, 0, 0, rb=6),            # T[20] = T[100 + 16]
            I.rld(7, 11),
            I.vop(I.V_FILL, 21, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, 0.0),
            I.loop(1, 1, rcount=7),                                      # R7 + 1 = 3 times
            I.vop(I.V_ADD, 21, 21, 0, 1, 1, 0, 0, 0, I.B_SCALAR, 1.0),
            I.halt()]
    s = run1(prog, {0: t, 116: [42.0]})
    assert s.tget([20, 21]).tolist() == [42.0, 3.0]


def test_conditional_halt_with_loop():
    """LOOP R {HALT}: a device-computed flag stops the program."""
    for flag, want in ((0.0, 5.0), (1.0, 0.0)):
        prog = [I.rld(3, 0), I.loop(1, 0, rcount=3), I.halt(),
                I.vop(I.V_FILL, 1, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, 5.0), I.halt()]
        assert run1(prog, {0: [flag]}).tget([1])[0] == want


# ---------------------------------------------------------------------------------- CHAIN
def test_halt_chain_starts_the_next_program_with_the_run_arguments():
    """HALT CHAIN loads R[rb] instructions from DRAM R[ra] and starts them: R0..7 = 0,
    R8..15 = the run's arguments; TMEM and DRAM are kept."""
    cfg = Config(S=1)
    second = [I.vop(I.V_ADD, 0, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, 1.0),
              I.st(0x40, 0, 1), I.st(0x80, 0, 1, rb=0, ra=15), I.halt()]
    dram = np.zeros(1 << 16, np.uint8)
    dram[0x1000:0x1000 + 32 * len(second)] = I.assemble(second).view(np.uint8)
    first = [I.vop(I.V_FILL, 0, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, 41.0),
             I.li(1, 0x1000), I.li(2, len(second)), I.li(3, 99),
             I.halt(chain=True, ra=1, rb=2)]
    m = Machine(cfg, [first], [dram], args=[0, 0, 0, 0, 0, 0, 0, 0x100]).run()
    s = m.slices[0]
    assert m.chains == 1 and s.halted and s.R[3] == 0 and s.R[15] == 0x100
    assert s.m32[0x40 // 4].view(np.float32) == 42.0
    assert s.m32[(0x100 + 0x80) // 4].view(np.float32) == 42.0


def test_halt_chain_checks_the_target():
    first = [I.li(1, 0x1004), I.li(2, 1), I.halt(chain=True, ra=1, rb=2)]
    with pytest.raises(SimError, match="CHAIN"):
        run1(first, dram=np.zeros(1 << 16, np.uint8), cfg=Config(S=1, D=32))


def test_the_gen_op_checks_run_on_the_isa_simulator():
    """otpu-diag's programs for the decode loop's instructions (opchecks group gen, only for a
    bitstream with CAPS bit30): each stores what it computed, HALT CHAIN's second program too."""
    import dataclasses

    from opentpu.host.checks import PROG_AT, ZERO_AT
    from opentpu.host.opchecks import diag_image, op_checks
    from opentpu.isasim import board_config
    cfg = board_config(DRAM_BYTES=1 << 23)
    img = diag_image()
    checks = [(n, p) for g, n, p in op_checks(cfg, gen=True) if g == "gen"]
    assert len(checks) == 6 and not [g for g, _, _ in op_checks(cfg) if g == "gen"]
    for name, prog in checks:
        ref = np.zeros(1 << 23, np.uint8)
        ref[:len(img)] = img
        m = Machine(dataclasses.replace(cfg, DRAM_BYTES=len(ref)),
                    [[I.ld(ZERO_AT, 0, cfg.TMEM_WORDS)] + prog], [ref]).run()
        assert (m.slices[0].dram[:PROG_AT] != ref[:PROG_AT]).sum() >= 27, name
        assert m.chains == (name == "HALT CHAIN")


# ---------------------------------------------------------------------------------- the RTL
def _on_rtl(prog, data=None, dram_bytes=1 << 20, cfg=None, args=None, extra=()):
    """prog after an LD of `data` (fp32 words, from DRAM 0 to TMEM 0) on the ISA simulator and
    the RTL: the same DRAM and TMEM. extra: (DRAM byte address, uint8 array) placed in DRAM."""
    from opentpu import rtlsim
    cfg = cfg or Config(S=1, DRAM_BYTES=dram_bytes)
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    pre = []
    if data is not None:
        d = f(data).reshape(-1)
        dram[:4 * len(d)] = d.view(np.uint8)
        pre = [I.ld(0, 0, len(d))]
    for a, b in extra:
        dram[a:a + len(b)] = b
    prog = pre + list(prog)
    m = Machine(cfg, [prog], [dram.copy()], args=args).run()
    drams, tmems, st = rtlsim.run(cfg, [prog], [dram.copy()], args=args)
    assert np.array_equal(drams[0], m.slices[0].dram)
    assert np.array_equal(tmems[0], m.slices[0].tmem)
    return m, st


def test_argmax_on_rtl(have_verilator):
    """VOP ARGMAX on the RTL: ties to the first column across lanes and chunks, -0 / +0,
    denormals, -inf rows, one-chunk and long rows, row strides, a negative base plus R[rd]."""
    rng = np.random.default_rng(5)
    a = rng.standard_normal((6, 300)).astype(np.float32)
    a[1, [7, 40, 299]] = 9.0
    a[2] = -np.inf
    a[2, 250] = -1e30
    a[3, :] = 0.0
    a[3, 100] = -0.0
    a[4, :] = -1.0
    a[4, 5], a[4, 9] = 1e-41, 0.0
    a[5, :] = 3.0                                   # all equal: column 0
    long_ = rng.standard_normal(5000).astype(np.float32)
    long_[[123, 4000]] = 50.0
    data = np.concatenate([a.reshape(-1), long_])
    L = 1800
    _on_rtl([I.argmax(8000, 0, 6, 300, 2, 300),
             I.li(3, 1000), I.argmax(8100, 0, 6, 5, drs=3, ars=300, base=-7, rd=3),
             I.argmax(8200, L, 1, 5000, base=1 << 20),
             I.argmax(8300, L + 3, 3, 1, drs=2, ars=1),
             I.argmax(8400, 0, 1, 300, base=-(1 << 26) - 3),      # rounds (above 2^24)
             I.st(0x40000, 8000, 512), I.halt()], data)


def test_rld_on_rtl(have_verilator):
    """RLD on the RTL: truncation, the bounds, inf / NaN, RAW, R0, register-relative reads, a
    read right after the VOP that writes the word, the value driving a LOOP count, an address
    and a conditional HALT, and MUL."""
    vals = [0.0, -0.0, 0.99, 1.0, 151935.0, -1020.0, -1.5, 2.0 ** 30 + 128, 2.0 ** 31, -3e9,
            np.inf, np.nan, 1e-40, 2.0 ** 23 + 1, -(2.0 ** 31)]
    prog = []
    for k, v in enumerate(vals):                  # R5 = want if RLD is right: the ST lands
        want = I.f2i(I.f32bits(v))
        prog += [I.rld(5, k), I.addi(5, 5, -want & 0xFFFFFFFF),
                 I.st(0x40000 + 64 * k, 64, 8, ra=5)]
    prog += [I.li(2, 3), I.rld(6, 1, ra=2),        # T[4] = 151935 via R2 = 3
             I.addi(6, 6, -151935 & 0xFFFFFFFF), I.st(0x41000, 64, 4, ra=6),
             I.rld(7, 3, raw=True), I.addi(7, 7, -0x3F800000 & 0xFFFFFFFF),
             I.st(0x41100, 64, 4, ra=7), I.rld(0, 4), I.st(0x41200, 64, 4, ra=0),
             I.vop(I.V_FILL, 40, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, 3.0),   # read after write
             I.rld(8, 40), I.loop(2, 0, rcount=8),
             I.addi(9, 9, 256), I.st(0x42000, 64, 16, ra=9),
             I.vop(I.V_FILL, 41, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, 0.0),
             I.rld(10, 41), I.loop(1, 0, rcount=10), I.halt(),
             I.st(0x43000, 64, 16)]
    # MUL: times R[rb] + w2 mod 2^32 (past 2^31, negative, RAW, a register multiplier)
    for k, (v, raw, c, rb) in enumerate([(151935.0, False, 9344, 0), (-1020.0, False, -7, 0),
                                         (1.0, True, 3, 0), (2.0 ** 23 + 1, False, 5, 2)]):
        x = I.f32bits(v) if raw else I.f2i(I.f32bits(v))
        m = c + (3 if rb else 0)
        prog += [I.rld(11, [4, 5, 3, 13][k], raw=raw, mul=c, rb=rb),
                 I.addi(11, 11, (64 * k - x * m) & 0xFFFFFFFF),
                 I.st(0x44000, 64, 16, ra=11)]
    prog += [I.halt()]
    data = np.concatenate([f(vals), np.arange(100, dtype=np.float32)])
    _on_rtl(prog, data)


def test_halt_chain_on_rtl(have_verilator):
    """HALT CHAIN on the RTL: the second program is loaded from DRAM and starts with the run's
    arguments and the first one's TMEM; a third follows it; ICOUNT counts all three."""
    cfg = Config(S=1, DRAM_BYTES=1 << 20)
    third = [I.vop(I.V_ADD, 0, 0, 0, 1, 16, 16, 16, 0, I.B_SCALAR, 100.0),
             I.st(0x800, 0, 16, ra=14), I.halt()]
    second = [I.vop(I.V_ADD, 0, 0, 0, 1, 16, 16, 16, 0, I.B_SCALAR, 1.0),
              I.st(0x400, 0, 16, ra=15), I.li(1, 0x2000), I.li(2, len(third)),
              I.halt(chain=True, ra=1, rb=2)]
    first = [I.vop(I.V_FILL, 0, 0, 0, 1, 16, 16, 16, 0, I.B_SCALAR, 41.0),
             I.li(1, 0x1000), I.li(2, len(second)), I.li(3, 99),
             I.halt(chain=True, ra=1, rb=2)]
    m, st = _on_rtl(first, cfg=cfg, args=[0, 0, 0, 0, 0, 0, 0x40, 0x100],
                    extra=[(0x1000, I.assemble(second).view(np.uint8)),
                           (0x2000, I.assemble(third).view(np.uint8))])
    assert m.chains == 2 and st["instructions"] == [m.slices[0].icount]


def test_gen_op_checks_on_the_board_config_rtl(have_verilator):
    """otpu-diag's gen programs (opchecks) on the board's configuration (D = 128, 16 lanes)."""
    import dataclasses

    from opentpu.host.checks import ZERO_AT
    from opentpu.host.opchecks import diag_image, op_checks
    from opentpu.isasim import board_config
    cfg = board_config(DRAM_BYTES=1 << 23)
    img = diag_image()
    for name, prog in [(n, p) for g, n, p in op_checks(cfg, gen=True) if g == "gen"]:
        _on_rtl([I.ld(ZERO_AT, 0, cfg.TMEM_WORDS)] + prog, cfg=cfg,
                extra=[(0, img)])


# ---------------------------------------------------------------------------------- the sampler
@pytest.mark.parametrize("S", [1, 2])
@pytest.mark.parametrize("temperature,top_k,top_p,penalty", [
    (0.7, 20, 0.8, 1.0), (0.1, 50, 1.0, 1.05), (1.0, 64, 0.95, 1.2), (2.0, 3, 1.0, 1.0)])
def test_the_device_sampler_draws_from_the_host_samplers_distribution(S, temperature, top_k,
                                                                      top_p, penalty):
    """generate.reference_pick (the device's sampler, bit for bit) as a function of its uniform
    u: each id's share of u in [0, 1) is its probability under otpu-chat's sampler
    (chat.sampler: the penalty, top-k, the temperature's softmax, top-p) within 1e-4, and it
    picks only ids that sampler keeps."""
    from opentpu.host.chat import _top_k_f64
    from opentpu.llm import generate as G
    rng = np.random.default_rng(int(100 * temperature) + top_k)
    V = 3000
    lg = (rng.standard_normal(V) * 3).astype(np.float32)
    ctx = [int(x) for x in rng.integers(0, V, 40)] + [int(np.argmax(lg))]
    w = lg.copy()
    ix = np.unique(ctx)
    w[ix] = np.where(w[ix] > 0, w[ix] / np.float32(penalty), w[ix] * np.float32(penalty))
    idx, z = _top_k_f64(w, top_k, temperature)
    p = np.exp(z - z[0])
    p /= p.sum()
    keep = min(len(p), np.searchsorted(np.cumsum(p), top_p) + 1)
    p = p[:keep] / p[:keep].sum()
    samp = G.Sampling(temperature, top_k, top_p, penalty)
    n = 100000
    picks = G.reference_pick(lg, samp, ctx, (np.arange(n) + 0.5) / n, S)
    ids, cnt = np.unique(picks, return_counts=True)
    assert set(ids) <= set(idx[:keep].tolist()) and len(ids) >= min(keep, 3)
    share = dict(zip(ids.tolist(), cnt / n))
    assert max(abs(share.get(i, 0.0) - q) for i, q in zip(idx[:keep].tolist(), p)) < 1e-4


# ---------------------------------------------------------------------------------- the loop
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _tiny(name):
    """Tiny random models whose greedy tokens vary (initializer_range 0.2: at the default
    0.02 they repeat one token)."""
    torch.manual_seed(0)
    if name == "qwen3":
        from opentpu.llm.qwen3 import Spec
        hc = transformers.Qwen3Config(
            hidden_size=256, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=128, intermediate_size=512, vocab_size=1000, rms_norm_eps=1e-6,
            rope_theta=1e6, tie_word_embeddings=True, max_position_embeddings=4096,
            initializer_range=0.2)
        m, spec = transformers.Qwen3ForCausalLM(hc), Spec(256, 2, 4, 2, 128, 512, 1000)
    elif name == "lfm2":
        from opentpu.llm.lfm2 import Spec
        kinds = ("conv", "attn", "conv", "attn", "conv")
        hc = transformers.Lfm2Config(
            hidden_size=256, num_hidden_layers=len(kinds), num_attention_heads=4,
            num_key_value_heads=2, intermediate_size=512, vocab_size=1000, norm_eps=1e-5,
            layer_types=["full_attention" if k == "attn" else "conv" for k in kinds],
            conv_L_cache=3, conv_bias=False, block_auto_adjust_ff_dim=False,
            tie_word_embeddings=True, max_position_embeddings=4096, initializer_range=0.2,
            rope_parameters={"rope_type": "default", "rope_theta": 1e6})
        m, spec = transformers.Lfm2ForCausalLM(hc), Spec(256, kinds, 4, 2, 64, 512, 1000)
    else:
        from opentpu.llm.qwen35 import Spec
        kinds = ("linear", "linear", "attn", "linear", "linear", "attn")
        hc = transformers.Qwen3_5TextConfig(
            hidden_size=256, num_hidden_layers=len(kinds), num_attention_heads=8,
            num_key_value_heads=2, head_dim=256, intermediate_size=512, vocab_size=1000,
            layer_types=["full_attention" if k == "attn" else "linear_attention" for k in kinds],
            linear_num_key_heads=8, linear_num_value_heads=8, linear_key_head_dim=128,
            linear_value_head_dim=128, linear_conv_kernel_dim=4, tie_word_embeddings=True,
            max_position_embeddings=4096, rms_norm_eps=1e-6, initializer_range=0.2,
            rope_parameters={"rope_type": "default", "rope_theta": 1e7,
                             "partial_rotary_factor": 0.25})
        m = transformers.Qwen3_5ForCausalLM(hc)
        spec = Spec(256, kinds, 8, 2, 256, 64, 8, 128, 128, 512, 1000)
    m = m.float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:
                base = 0.0 if name == "qwen35" and not n.endswith("linear_attn.norm.weight") else 1.0
                p.copy_(base + 0.1 * torch.randn_like(p))
    return {k: v.float().numpy() for k, v in m.state_dict().items()}, spec


@pytest.fixture(scope="module", params=["qwen3", "lfm2", "qwen35"])
def tiny(request):
    return (request.param,) + _tiny(request.param)


@pytest.mark.parametrize("S", [1, 2])
@pytest.mark.parametrize("split", [None, True])
def test_generate_matches_the_host_loop(tiny, S, split):
    """The decode loop on the device gives the host's resident greedy loop token for token,
    across the attention bucket boundary at 256 (a second run of the next bucket's program),
    and stops at a stop id (returned, not fed). split: every bucket in the split form, two
    programs per token chaining to each other (generate.compile_bucket)."""
    from opentpu.llm.qwen3 import PREFILL_ROWS, Engine, device_config
    name, W, spec = tiny
    cfg = device_config(spec, 512, rows=PREFILL_ROWS, lookup=True, S=S)
    a, b = (Engine(spec, W, cap=512, cfg=cfg, resident=True) for _ in range(2))
    a.gen_split = split
    assert a.can_generate
    toks = [int(t) for t in np.random.default_rng(1).integers(0, 1000, 248)]
    t0 = int(np.argmax(a.prefill(toks)))
    assert int(np.argmax(b.prefill(toks))) == t0
    ref, t = [], t0
    for _ in range(20):
        t = int(np.argmax(b.step(t)))
        ref.append(t)
    assert len(set(ref)) > 4                        # the tokens vary
    got = a.generate_card(t0, 12, stop_ids=[])
    assert got == ref[:12] and a.pos == 248 + 12
    assert sorted(a._gens) == [(1, None, False), (2, None, False)]   # buckets 1, 2, greedy
    assert all(isinstance(p, tuple) == bool(split) for p in a._gens.values())
    j = next(j for j in range(13, 20) if ref[j] not in ref[12:j])   # a token not seen since
    assert a.generate_card(ref[11], 30, stop_ids=[ref[j], 1001]) == ref[12:j + 1]
    assert a.pos == 248 + j + 1                      # the stop id is not fed


@pytest.mark.parametrize("split", [None, True])
def test_generate_with_the_int8_embedding(tiny, split):
    """Spec.embed "int8": the token's row gathered on the device at the run-time token (qwen3._embed)
    inside the loop, one program or split: the host resident loop's tokens."""
    import dataclasses
    from opentpu.llm.qwen3 import PREFILL_ROWS, Engine, device_config
    name, W, spec = tiny
    spec = dataclasses.replace(spec, embed="int8")
    cfg = device_config(spec, 512, rows=PREFILL_ROWS, lookup=True, S=1)
    a, b = (Engine(spec, W, cap=512, cfg=cfg, resident=True) for _ in range(2))
    a.gen_split = split
    toks = [int(t) for t in np.random.default_rng(1).integers(0, 1000, 248)]
    t0 = int(np.argmax(a.prefill(toks)))
    assert int(np.argmax(b.prefill(toks))) == t0
    ref, t = [], t0
    for _ in range(12):
        t = int(np.argmax(b.step(t)))
        ref.append(t)
    assert len(set(ref)) > 3
    assert a.generate_card(t0, 12, stop_ids=[]) == ref


@pytest.mark.parametrize("S,sampled,split", [(1, False, None), (1, True, None), (2, False, None),
                                             (2, True, None), (1, False, True), (2, True, True)])
def test_generate_on_rtl(have_verilator, tiny, S, sampled, split):
    """The generate loop on the Verilator RTL: from the same DRAM state (a 248-token prefill
    on the ISA simulator) one run of 12 tokens across the bucket boundary (HALT CHAIN), greedy
    or sampled (top-k, top-p, the penalty), in one program per bucket or split (two per
    token): the tokens and the whole DRAM of the ISA simulator's run."""
    from opentpu.llm.qwen3 import PREFILL_ROWS, Engine, device_config
    from opentpu.llm.rtl_backend import RtlBackend
    name, W, spec = tiny
    cfg = device_config(spec, 512, rows=PREFILL_ROWS, lookup=True, S=S)
    eng = Engine(spec, W, cap=512, cfg=cfg, resident=True)
    eng.gen_split = split
    toks = [int(t) for t in np.random.default_rng(1).integers(0, 1000, 248)]
    t0 = int(np.argmax(eng.prefill(toks)))
    samp = G.Sampling(0.8, 5, 0.9, 1.1) if sampled else None
    kw = dict(sampling=samp, context=toks + [t0], rng=np.random.default_rng(3)) if sampled \
        else {}
    isa = eng.backend
    n = eng.image.nbytes
    rtl = RtlBackend(eng.cfg, [s.dram[:n] for s in isa.machine.slices])
    pos = eng.pos
    want = eng.generate_card(t0, 12, stop_ids=[], **kw)
    if sampled:
        kw["rng"] = np.random.default_rng(3)
    eng.backend, eng.pos, eng._chained = rtl, pos, {}
    got = eng.generate_card(t0, 12, stop_ids=[], **kw)
    assert got == want and len(set(got)) > 3
    for s in range(eng.cfg.S):
        assert np.array_equal(isa.machine.slices[s].dram[:n], rtl.drams[s][:n])


def test_generate_debug_keeps_the_logits(tiny):
    """Engine.gen_debug: the generate loop's LM head also stores the logits, and after a run
    they are the last token's, bit for bit the resident step's at that position."""
    from opentpu.llm.qwen3 import PREFILL_ROWS, Engine, device_config
    name, W, spec = tiny
    cfg = device_config(spec, 512, rows=PREFILL_ROWS, lookup=True, S=2)
    a, b = (Engine(spec, W, cap=512, cfg=cfg, resident=True) for _ in range(2))
    a.gen_debug = True
    toks = [int(t) for t in np.random.default_rng(2).integers(0, 1000, 20)]
    t0 = int(np.argmax(a.prefill(toks)))
    b.prefill(toks)
    got = a.generate_card(t0, 6, stop_ids=[])
    lg, t = None, t0
    for want in got:
        lg = b.step(t)
        t = int(np.argmax(lg))
        assert t == want
    assert np.array_equal(a.gen_logits.view(np.uint32), lg.view(np.uint32))


@pytest.mark.parametrize("S", [1, 2])
@pytest.mark.parametrize("temperature,top_k,top_p,penalty", [(0.8, 5, 0.9, 1.1), (0.0, 0, 1.0, 1.3),
                                                           (1.5, 20, 1.0, 1.0)])
@pytest.mark.parametrize("split", [None, True])
def test_sampled_generate_matches_the_reference_pick(tiny, S, temperature, top_k, top_p,
                                                     penalty, split):
    """The sampled loop on the device (generate.Sampler) picks the ids of its numpy model
    (generate.reference_pick) from the host loop's logits, with the same uniforms, across the
    bucket boundary: top-k, top-p, the temperature and the repetition penalty (greedy with the
    penalty too), the device adding each id to the penalty's context."""
    from opentpu.llm import generate as G
    from opentpu.llm.qwen3 import PREFILL_ROWS, Engine, device_config
    name, W, spec = tiny
    if name != "qwen3" and (S, top_k) != (1, 5):
        pytest.skip("the other models: one case")
    if split and top_k != 5:
        pytest.skip("split programs: one case per model")
    cfg = device_config(spec, 512, rows=PREFILL_ROWS, lookup=True, S=S)
    a, b = (Engine(spec, W, cap=512, cfg=cfg, resident=True) for _ in range(2))
    a.gen_split = split
    samp = G.Sampling(temperature, top_k, top_p, penalty)
    toks = [int(t) for t in np.random.default_rng(1).integers(0, 1000, 248)]
    lg = a.prefill(toks)
    assert np.array_equal(lg, b.prefill(toks))
    ctx = toks + [int(np.argmax(lg))]
    got = a.generate_card(ctx[-1], 12, stop_ids=[], sampling=samp, context=ctx,
                          rng=np.random.default_rng(7))
    u, want, c = np.random.default_rng(7).random(12), [], list(ctx)
    for i in range(12):
        t = G.reference_pick(b.step(c[-1]), samp, c, u[i], S)
        want.append(t)
        c.append(t)
    assert got == want and a.pos == 260
    assert len(set(got)) > 3 or temperature == 0


@pytest.mark.parametrize("split", [None, True])
def test_a_softcap_caps_the_sampled_logits(tiny, split):
    """A spec with a softcap (Gemma's final_logit_softcapping): the LM head caps the chunks the
    sampler takes (kernels.lib.softcap), so the device picks reference_pick's ids from the
    capped logits (generate.softcap_ref, bit for bit), in one program or split; greedy takes
    the raw logits' argmax (the cap keeps the order)."""
    import copy
    from opentpu.llm import generate as G
    from opentpu.llm.qwen3 import PREFILL_ROWS, Engine, device_config
    name, W, spec = tiny
    if name != "qwen3":
        pytest.skip("one model")
    spec = copy.copy(spec)
    object.__setattr__(spec, "softcap", 2.0)    # (a frozen Spec); the logits reach past +-2
    cfg = device_config(spec, 512, rows=PREFILL_ROWS, lookup=True, S=1)
    a, b = (Engine(spec, W, cap=512, cfg=cfg, resident=True) for _ in range(2))
    a.gen_split = split
    samp = G.Sampling(0.8, 5, 0.9, 1.1)
    toks = [int(t) for t in np.random.default_rng(1).integers(0, 1000, 248)]
    lg = a.prefill(toks)
    b.prefill(toks)
    assert np.abs(lg).max() > 4.0
    ctx = toks + [int(np.argmax(lg))]
    got = a.generate_card(ctx[-1], 12, stop_ids=[], sampling=samp, context=ctx,
                          rng=np.random.default_rng(7))
    u, want, raw, c = np.random.default_rng(7).random(12), [], [], list(ctx)
    for i in range(12):
        lgi = b.step(c[-1])
        t = G.reference_pick(lgi, samp, c, u[i], 1, softcap=2.0)
        raw.append(G.reference_pick(lgi, samp, c, u[i], 1))     # the uncapped pick
        want.append(t)
        c.append(t)
    assert got == want and raw != want
    assert np.abs(G.softcap_ref(lg, 2.0)).max() <= 2.0
    # greedy: the raw argmax, as the host loop's
    t0 = want[-1]
    ref, t = [], t0
    for _ in range(6):
        t = int(np.argmax(b.step(t)))
        ref.append(t)
    assert a.generate_card(t0, 6, stop_ids=[]) == ref


@pytest.fixture(scope="module")
def tiny_gemma4(tmp_path_factory):
    """tests/test_gemma4.py's tiny Gemma 4: sliding (head_dim 128) and global (256) layers,
    shared KV layers, per-layer inputs gathered on the device, the softcap 30 (its greedy
    tokens vary at the default initializer_range; at 0.1 and up they repeat one token)."""
    import json
    pytest.importorskip("transformers.models.gemma4")
    from opentpu.llm import gemma4
    S_, F_ = "sliding_attention", "full_attention"
    kinds = (S_, S_, F_, S_, S_, F_, S_, F_, F_)
    torch.manual_seed(0)
    hc = transformers.Gemma4TextConfig(
        hidden_size=256, num_hidden_layers=len(kinds), num_attention_heads=8,
        num_key_value_heads=1, head_dim=128, global_head_dim=256, intermediate_size=512,
        vocab_size=1000, vocab_size_per_layer_input=1000, hidden_size_per_layer_input=128,
        layer_types=list(kinds), num_kv_shared_layers=3, use_double_wide_mlp=True,
        sliding_window=512, final_logit_softcapping=30.0, max_position_embeddings=4096)
    m = transformers.models.gemma4.modeling_gemma4.Gemma4ForCausalLM(hc).float().eval()
    with torch.no_grad():
        for n, q in m.named_parameters():
            if "norm" in n:
                k = "k_norm" in n
                q.copy_((0.13 if k else 1.0) + (0.01 if k else 0.1) * torch.randn_like(q))
        for n, b in m.named_buffers():
            if n.endswith("layer_scalar"):
                b.copy_(0.5 + torch.rand_like(b))
    d = tmp_path_factory.mktemp("gemma4")
    (d / "config.json").write_text(json.dumps(hc.to_dict()))
    return {k: v.float().numpy() for k, v in m.state_dict().items()}, gemma4.Spec.from_hf(d)


def _gemma4_engines(tiny_gemma4, n=2, **kw):
    from opentpu.isasim import board_config
    from opentpu.llm.qwen3 import Engine
    W, spec = tiny_gemma4
    return [Engine(spec, W, cap=1024, cfg=board_config(DRAM_BYTES=1 << 26), wformat="fp4",
                   head_format="int8", resident=True, **kw) for _ in range(n)]


@pytest.mark.parametrize("split", [None, True])
def test_gemma4_generate_matches_the_host_loop(tiny_gemma4, split):
    """Gemma 4's decode loop on the device (gemma4_step at a RunPos: the token's embedding and
    PLE rows gathered, the run-time masks) gives the host's resident greedy loop token for
    token, across the bucket boundary at 256, in one program or split; then sampled with the
    model's softcap (30) in _lm_head, reference_pick's ids from the capped logits."""
    W, spec = tiny_gemma4
    a, b = _gemma4_engines(tiny_gemma4)
    a.gen_split = split
    assert a.can_generate and spec.softcap == 30.0
    toks = [int(t) for t in np.random.default_rng(1).integers(0, 1000, 248)]
    lg = a.prefill(toks)
    t0 = int(np.argmax(lg))
    assert int(np.argmax(b.prefill(toks))) == t0
    ref, t = [], t0
    for _ in range(12):
        t = int(np.argmax(b.step(t)))
        ref.append(t)
    assert len(set(ref)) > 3
    assert a.generate_card(t0, 12, stop_ids=[]) == ref
    assert all(isinstance(p, tuple) == bool(split) for p in a._gens.values())
    samp = G.Sampling(0.8, 5, 0.9, 1.1)
    ctx = toks + [t0] + ref
    got = a.generate_card(ref[-1], 8, stop_ids=[], sampling=samp, context=ctx,
                          rng=np.random.default_rng(7))
    u, want, c = np.random.default_rng(7).random(8), [], list(ctx)
    for i in range(8):
        t = G.reference_pick(b.step(c[-1]), samp, c, u[i], 1, softcap=spec.softcap)
        want.append(t)
        c.append(t)
    assert got == want


@pytest.mark.parametrize("split", [None, True])
def test_gemma4_generate_on_rtl(have_verilator, tiny_gemma4, split):
    """Gemma 4's generate loop on the Verilator RTL: from the ISA simulator's DRAM after a
    248-token prefill, 12 greedy tokens across the bucket boundary in one run: the tokens and
    the whole DRAM of the ISA simulator's run."""
    from opentpu.llm.rtl_backend import RtlBackend
    eng, = _gemma4_engines(tiny_gemma4, 1)
    eng.gen_split = split
    toks = [int(t) for t in np.random.default_rng(1).integers(0, 1000, 248)]
    t0 = int(np.argmax(eng.prefill(toks)))
    isa, n = eng.backend, eng.image.nbytes
    rtl = RtlBackend(eng.cfg, [s.dram[:n] for s in isa.machine.slices])
    pos = eng.pos
    want = eng.generate_card(t0, 12, stop_ids=[])
    eng.backend, eng.pos, eng._chained = rtl, pos, {}
    got = eng.generate_card(t0, 12, stop_ids=[])
    assert got == want and len(set(got)) > 3
    assert np.array_equal(isa.machine.slices[0].dram[:n], rtl.drams[0][:n])


@pytest.mark.skipif(not (Path(__file__).resolve().parent.parent / "models" / "gemma-4-E2B" /
                     "config.json").exists(), reason="no models/gemma-4-E2B")
def test_gemma4_e2b_generate_fits_imem():
    """Gemma 4 E2B on the board (fp4 layers, int8 LM head and PLE, 4096 tokens, the generate
    area in the 4 GiB image): every bucket's generate program, greedy and sampled, is one
    program under IMEM (layout only, no weights)."""
    from opentpu.isasim import board_config
    from opentpu.llm import gemma4
    spec = gemma4.Spec.from_hf(Path(__file__).resolve().parent.parent / "models" / "gemma-4-E2B")
    img = spec.image(board_config(), 4096, 1, 8, "fp4", "int8", lookup=True)
    assert img.nbytes < 1 << 32
    for samp in (None, G.Sampling(1.0, 64, 0.95, 1.0)):
        for blocks in (1, 3, 8, 16):
            progs = G.compile_bucket(img, blocks, (blocks - 1) * 256, 256, samp=samp)
            assert not isinstance(progs, tuple) and G.fits(img, progs), (blocks, samp)


class _Tok:
    """One token per character (its code point): the ids fit the tiny vocabulary."""

    def apply_chat_template(self, history, add_generation_prompt, enable_thinking, tokenize):
        return [ord(c) for m in history for c in m["content"]]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(f"<{i}>" for i in ids)


def test_chat_decodes_on_the_device():
    """otpu-chat's greedy turns run the decode loop on the device (Chat.on_card): the reply of
    the host's loop, shown token by token; a reply cut at max_new resumes from its last token
    (picked on the device, not fed yet); an EOS id ends it (not shown, not fed)."""
    from dataclasses import replace

    from opentpu.host.chat import Chat, sampler
    from opentpu.llm.qwen3 import PREFILL_ROWS, Engine, device_config
    W, spec = _tiny("qwen3")
    cfg = device_config(spec, 512, rows=PREFILL_ROWS, lookup=True, S=1)

    def chat(pick, max_new):
        return Chat(Engine(spec, W, cap=512, cfg=cfg, resident=True), _Tok(), False, pick,
                    max_new)
    host = chat(lambda logits, ctx: int(np.argmax(logits)), 40)
    card = chat(sampler(0, 0, 1.0, None), 16)
    assert card.on_card and not host.on_card
    host.ask("hello there")
    ref = host._reply
    assert len(ref) == 40 and len(set(ref)) > 4
    shown = []
    reply, t = card.ask("hello there", lambda d, turn: shown.append(d))
    assert card._reply == ref[:16] and reply == "".join(shown) and t.end == "max_new"
    assert t.gen_tokens == 16 and t.decode_steps == 15 and card.eng.pos == len(card.fed) == 26
    card.resume()
    assert card._reply == ref[:32] and card.fed == host.fed[:len(card.fed)]
    j = next(j for j in range(33, 40) if ref[j] not in ref[32:j])
    card.eng.spec = replace(spec, eos=(ref[j],))
    reply, t = card.resume()
    assert card._reply == ref[:j] and t.end == "eos" and not card.can_resume
    assert card.eng.pos == len(card.fed) == 11 + j


def test_chat_samples_on_the_device():
    """A sampled otpu-chat turn on the device: the host's sampler picks the first token from
    the prefill's logits, the device the rest (generate.reference_pick with the uniforms of the
    sampler's generator, which the run draws next); top_k 0 stays on the host."""
    from opentpu.host.chat import Chat, sampler
    from opentpu.llm import generate as G
    from opentpu.llm.qwen3 import PREFILL_ROWS, Engine, device_config
    W, spec = _tiny("qwen3")
    cfg = device_config(spec, 512, rows=PREFILL_ROWS, lookup=True, S=1)
    args = (0.8, 5, 0.9, 3, 1.1)
    card = Chat(Engine(spec, W, cap=512, cfg=cfg, resident=True), _Tok(), False, sampler(*args),
                12)
    assert card.on_card
    assert not Chat(card.eng, _Tok(), False, sampler(0.8, 0, 0.9, 3), 12).on_card
    card.ask("hello there")
    ref, pick = Engine(spec, W, cap=512, cfg=cfg, resident=True), sampler(*args)
    c = [ord(x) for x in "hello there"]
    c.append(pick(ref.prefill(c), c))
    u = pick.rng.random(11)
    samp = G.Sampling(0.8, 5, 0.9, 1.1)
    for i in range(11):
        c.append(G.reference_pick(ref.step(c[-1]), samp, c, u[i]))
    assert card._reply == c[11:] and len(set(c[11:])) > 3


# ---------------------------------------------------------------------------------- the card
class _GenCard:
    """A fake card that computes: RUN runs the program on the ISA simulator over the channel
    memories (with the ARG registers), and the generate loop's out[] words land one by one,
    `gap` seconds apart (HALTED after the last); the host's stop word written during the run
    halts it after the token in flight (a card that computed ahead: only attention models, whose
    later positions' K/V the next run writes again)."""

    def __new__(cls, *a, **kw):
        from opentpu.host.fake import FakeTransport

        class Card(FakeTransport):
            def __init__(self, cfg, out, state, gap=0.003):
                super().__init__(ch_bytes=cfg.DRAM_BYTES // 2, devname=None, args=True,
                                 gen=True)
                self.cfg, self.area, self.state, self.gap = cfg, out, state, gap
                self.pending, self.runs_gen = [], 0

            def _logical(self, ch, off):
                return off // 64 * 128 + ch * 64 + off % 64

            def _apply(self):
                now = time.perf_counter()
                while self.pending and self.pending[0][0] <= now:
                    _, a, w = self.pending.pop(0)
                    self._put(a, w)

            def mem_write(self, ch, off, data):
                a = self._logical(ch, off)
                if (self.t_run is not None and self.pending
                        and a <= self.state + 4 * G.S_HALT < a + len(data)):
                    self._apply()
                    t_next = self.pending[0][0]          # the token in flight lands, no later
                    self.pending = self.pending[:1]
                    self.run_s = t_next - self.t_run + 1e-4
                super().mem_write(ch, off, data)

            def mem_read(self, ch, off, n, out=None):
                self._apply()
                return super().mem_read(ch, off, n, out)

            def reg_read(self, off):
                self._apply()
                return super().reg_read(off)

            def reg_write(self, off, val):
                from opentpu.host import regs as R
                from opentpu.host.board import join, split
                from opentpu.isasim import Machine
                rise = off == R.R_CTRL and val & R.CTRL_RUN and not self.regs[R.R_CTRL] & R.CTRL_RUN
                super().reg_write(off, val)
                if not rise:
                    return
                self._apply()
                dram = join(self.ch)
                a, n = self.regs[R.R_PROG_ADDR], self.regs[R.R_PROG_N]
                w = dram[a:a + 32 * n].view(np.uint32).reshape(n, 8)
                args = [self.regs.get(R.R_ARG0 + 4 * k, 0) for k in range(8)]
                m = Machine(self.cfg, [[I.Instr.decode(x) for x in w]], [dram.copy()], args=args)
                m.run(max_steps=1 << 40)
                new = m.slices[0].dram
                o, k = self.area
                before = dram[o:o + 4 * k].view(np.uint32)
                after = new[o:o + 4 * k].view(np.uint32).copy()
                land = np.flatnonzero(before != after)
                new[o:o + 4 * k] = dram[o:o + 4 * k]      # the tokens land later
                for c, off, part in split(0, new):
                    self.ch[c][:] = part
                t0 = self.t_run = time.perf_counter()
                self.pending = [(t0 + self.gap * (i + 1), o + 4 * int(j),
                                 after[j:j + 1].view(np.uint8).copy())
                                for i, j in enumerate(land)]
                self.run_s = self.gap * (len(land) + 0.5)
                self.runs_gen += bool(len(land))

        return Card(*a, **kw)


def test_generate_on_a_card_streams_the_tokens(tmp_path, monkeypatch):
    """BoardBackend.run_generate: the tokens reach on_token while the card runs (one run for
    the reply), the same as the host's greedy loop; stop() halts the loop on the card after the
    token in flight, and the next run continues from there."""
    from opentpu import lens as L
    from opentpu.host.board import BoardBackend, sim_config
    from opentpu.llm.qwen3 import Engine
    for k in ("OTPU_MCOLS", "OTPU_LANES", "OTPU_PAIR", "OTPU_DSTEP"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OTPU_RUN_DIR", str(tmp_path / "otpu"))
    spec, W = L._tiny_qwen()
    cfg = sim_config(spec, 512, lookup=True)
    ref = Engine(spec, W, cap=512, cfg=cfg, resident=True)   # names the gen area's addresses
    g = ref.image.lookup["gen"]
    card = [_GenCard(cfg, (g["out"], 513), g["state"])]
    eng = Engine(spec, W, cap=512, cfg=cfg, resident=True, pipeline=False,
                 backend=lambda c, imgs: BoardBackend(c, imgs, transport=card[0], status=False))
    assert eng.can_generate and eng.image.lookup["gen"]["out"] == g["out"]
    prompt = [3, 1, 4, 1, 5]
    t0 = int(np.argmax(eng.prefill(prompt)))
    assert int(np.argmax(ref.prefill(prompt))) == t0
    want, t = [], t0
    for _ in range(30):
        t = int(np.argmax(ref.step(t)))
        want.append(t)
    seen, t_start = [], time.perf_counter()
    got = eng.generate_card(t0, 16, stop_ids=[], on_token=lambda x: seen.append(
        (x, time.perf_counter() - t_start)))
    assert got == want[:16] and [x for x, _ in seen] == got and card[0].runs_gen == 1
    assert seen[-1][1] - seen[0][1] > 10 * card[0].gap   # as they land, not at the end
    seen.clear()
    got = eng.generate_card(want[15], 14, stop_ids=[], on_token=lambda x: seen.append((x, 0)),
                            stop=lambda: len(seen) >= 4)
    assert 4 <= len(got) <= 6 and got == want[16:16 + len(got)]
    assert eng.pos == len(prompt) + 16 + len(got)
    got2 = eng.generate_card(got[-1], 4, stop_ids=[])
    assert got2 == want[16 + len(got):20 + len(got)]
    eng.backend.close()
