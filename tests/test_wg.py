"""Weight groups (docs/isa.md "Weight groups", docs/weight_groups.md): an MM with GRP over a
matrix grouped by WG rows (int8) or 2 WG rows (4-bit: two rows a lane, a chunk a K-block;
quant.group_rows / group_scales; CHASH: each chunk's lanes in the channel hash's order,
quant.hash_rows) gives the bits the same MM gives at WG 1 over the natural rows (with PAIR
there), for every weight format, UNIT, RSC, ACC, k0 (odd too), N not a multiple of a group, odd
KB and M past MCOLS (replay); and the layouts' round trips and the field checks."""
import numpy as np
import pytest

from opentpu import Config, isa as I, quant as Q
from opentpu.isasim import Machine, SimError

X, WA, SA, OUT = 0, 1 << 16, 3 << 18, 1 << 12
FMT = {I.W8: "int8", I.W4I: "int4", I.W4F: "fp4", I.WNV: "nvfp4"}


def _cfg(wg, D=32, **kw):
    return Config(S=1, D=D, MCOLS=8, ACT_BLOCKS=16, ACT_ROWS=16, PAIR=True, WG=wg,
                  DRAM_BYTES=1 << 21, **kw)


def _matrix(rng, wf, N, KB, KBr, rsc, D):
    """Random natural element rows [N, nc D] and scale rows [N, scale_bytes(KBr)]."""
    fmt = FMT[wf]
    if wf == I.W8:
        data = rng.integers(-127, 128, (N, KB * D)).astype(np.int8).view(np.uint8)
    else:
        data = np.zeros((N, Q.row_bytes(KB * D, fmt, D)), np.uint8)
        data[:, :KB * D // 2] = Q.pack4(rng.integers(0, 16, (N, KB * D)))
    sw = (rng.standard_normal(N) * 0.01).astype(np.float32)
    rows = np.zeros((N, Q.scale_bytes(KBr * D, fmt, D, rsc)), np.uint8)
    rows[:, :4] = sw[:, None].view(np.uint8)
    if not rsc:
        h, sb = Q.swp(fmt, 1), Q.SCALE_DATA[fmt]
        if wf == I.WNV:
            dat = rng.integers(0, 0x7F, (N, KBr * sb))
        elif wf == I.W8:
            dat = rng.integers(0, 20, (N, KBr))
        else:
            word = rng.integers(0, 20, (N, KBr)) | sum(
                rng.integers(0, 64, (N, KBr)) << (8 + 6 * b) for b in range(4))
            dat = word.astype(np.uint32).view(np.uint8).reshape(N, -1)
        rows[:, h:h + KBr * sb] = dat
    return data, rows


def _act(rng, M, KB, D):
    return (rng.standard_normal((M, KB * D)) * np.exp(rng.uniform(-6, 2, (M, KB * 4, 1)))
            .repeat(D // 4, axis=2).reshape(M, -1)).astype(np.float32)


def _place(dram, at, a):
    b = np.ascontiguousarray(a).view(np.uint8).reshape(-1)
    dram[at:at + b.size] = b


def _mm_both(wf, N, KB, M, k0, unit=False, rsc=False, pair=False, acc=False, D=32, seed=0,
             chash=False):
    """The MM's TMEM outputs at WG 1 (natural rows, PAIR as asked) and WG 8 (GRP, grouped rows,
    no PAIR; chash: in the channel hash's order at their address)."""
    rng = np.random.default_rng(seed)
    fmt = FMT[wf]
    x = _act(rng, M, KB, D)
    data, rows = _matrix(rng, wf, N, KB, k0 + KB, rsc, D)
    ors = N + 1
    outs = []
    for wg in (1, 8):
        cfg = _cfg(wg, D, CHASH=chash and wg > 1)
        dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
        _place(dram, X, x)
        if wg == 1:
            rs, srs = data.shape[1], rows.shape[1]
            d, s = data, rows
        else:
            d = Q.group_rows(data, wg, D, fmt, KB * D)
            s = Q.group_scales(rows, fmt, (k0 + KB) * D, wg, D, rsc)
            rs, srs = d.shape[1], s.shape[1]
            if chash:
                d = Q.hash_rows(d, WA, wg, D)
        _place(dram, WA, d)
        _place(dram, SA, s)
        p = pair and wg == 1
        mm = dict(sa=WA, ssa=SA, out=OUT, n=N, kb=KB, rs=rs, ors=ors, m=M, ab=1, srs=srs,
                  unit=unit, wf=wf, pair=p, k0=k0, rsc=rsc, grp=Q.grp_r(fmt, wg))
        prog = [I.ld(X, 0, x.size), I.qact(0, M, 1, KB, KB * D, dup=p), I.mm(**mm)]
        if acc:
            prog.append(I.mm(**mm, acc=True))
        s = Machine(cfg, [prog + [I.halt()]], [dram]).run().slices[0]
        outs.append(s.tget(OUT + np.arange(M)[:, None] * ors + np.arange(N)[None, :]))
    return outs


CASES = [  # wf, unit, rsc, pair
    (I.W8, False, False, False), (I.W8, True, False, False), (I.W8, False, True, False),
    (I.W4I, False, False, False), (I.W4I, False, False, True), (I.W4I, False, True, False),
    (I.W4F, False, False, False), (I.W4F, False, False, True), (I.W4F, True, False, True),
    (I.W4F, False, True, False), (I.WNV, False, False, False), (I.WNV, False, False, True),
    (I.WNV, True, False, False),
]


@pytest.mark.parametrize("wf,unit,rsc,pair", CASES)
@pytest.mark.parametrize("acc", [False, True])
@pytest.mark.parametrize("chash", [False, True])
def test_grouped_mm_equals_natural(wf, unit, rsc, pair, acc, chash):
    i = 0
    for N in (1, 8, 20, 33):
        for KB in (1, 2, 3, 5):
            for M in ((1, 3) if pair else (1, 3, 9)):
                for k0 in ((0, 2) if pair else (0, 1)):
                    a, b = _mm_both(wf, N, KB, M, k0, unit, rsc, pair, acc, chash=chash,
                                    seed=1000 * wf + 100 * unit + 10 * rsc + pair + 7 * i)
                    i += 1
                    assert a.view(np.uint32).tolist() == b.view(np.uint32).tolist(), \
                        (FMT[wf], N, KB, M, k0)


@pytest.mark.parametrize("wf", [I.W8, I.W4F, I.WNV])
@pytest.mark.parametrize("chash", [False, True])
def test_grouped_mm_at_d128(wf, chash):
    a, b = _mm_both(wf, 20, 3, 2, 1, D=128, seed=wf, chash=chash)
    assert a.view(np.uint32).tolist() == b.view(np.uint32).tolist()


def test_hashed_rows_are_the_channels():
    """quant.hash_rows (CHASH, docs/isa.md "Weight groups"): lane j of the group chunk at A at
    lane j ^ h(A / (R D)), its own inverse at an address; through the memory's map (the host's,
    opentpu/host/board.py split: R channels of D-byte beats, hashed) channel c holds lane c of
    every chunk, at any chunk-aligned address; without the hash the lanes go on in order."""
    from opentpu.host.board import split
    rng = np.random.default_rng(9)
    D, R, N, nc = 128, 8, 24, 5
    rows = rng.integers(0, 256, (N, nc * D)).astype(np.uint8)
    g = Q.group_rows(rows, R, D)                               # [3 groups, nc chunks of R rows]
    nat = g.reshape(-1, R, D)                                  # [chunk, row, D]
    assert Q.chunk_fold(0b100_011_110, R) == 0b100 ^ 0b011 ^ 0b110 == 1
    assert Q.chunk_fold(np.arange(5), 1).tolist() == [0] * 5
    for at in (0, R * D, 37 * R * D, (1 << 31) - 4 * R * D):
        h = Q.hash_rows(g, at, R, D)
        assert h.shape == g.shape and np.array_equal(Q.hash_rows(h, at, R, D), g)
        m = at // (R * D) + np.arange(len(nat))
        hs = h.reshape(-1, R, D)
        for t in range(len(nat)):
            f = int(Q.chunk_fold(m[t], R))
            for j in range(R):
                assert np.array_equal(hs[t, j ^ f], nat[t, j])
        for c, off, v in split(at, h.reshape(-1), chash=True, nch=R, beat=D):
            assert off == at // R and np.array_equal(v.reshape(-1, D), nat[:, c]), (at, c)
        moved = Q.hash_rows(Q.hash_rows(h, at, R, D), at + 3 * R * D, R, D)
        assert np.array_equal(moved, Q.hash_rows(g, at + 3 * R * D, R, D))
    for c, off, v in split(0, g.reshape(-1), chash=False, nch=R, beat=D):
        assert np.array_equal(v.reshape(-1, D), nat[:, c])
    with pytest.raises(ValueError, match="whole 1024-byte chunks"):
        Q.hash_rows(g, D, R, D)


@pytest.mark.parametrize("fmt", ["int8", "int4", "fp4", "nvfp4"])
@pytest.mark.parametrize("R", [1, 2, 8])
def test_layout_round_trips(fmt, R):
    rng = np.random.default_rng(3)
    D, N, K = 32, 13, 5 * 32
    W = rng.standard_normal((N, K)).astype(np.float32)
    q, s = Q.quantize_mxu(W, fmt, D)
    g, a = Q.quantize_grouped(W, fmt, R, D)
    G = -(-N // Q.grp_r(fmt, R))                    # (R the machine's WG: 2 R rows of 4-bit)
    assert g.shape == (G, Q.grp_row_bytes(K, fmt, D, R))
    assert a.shape == (G, Q.grp_scale_bytes(K, fmt, D, R))
    assert np.array_equal(Q.ungroup_rows(g, N, R, D, fmt), q)
    assert np.array_equal(Q.ungroup_scales(a, fmt, K, N, R, D), s)
    if R == 1:                                      # the natural layout, byte for byte
        assert np.array_equal(g, q) and np.array_equal(a, s)
        assert Q.grp_scale_bytes(K, fmt, D, 1) == Q.scale_bytes(K, fmt, D)
    sr = s[:, :4].copy()
    assert np.array_equal(Q.ungroup_scales(Q.group_scales(sr, fmt, K, R, D, rsc=True), fmt, K,
                                           N, R, D, rsc=True), sr)


@pytest.mark.parametrize("fmt,R,p", [("int8", 8, 32), ("fp4", 16, 64), ("nvfp4", 16, 128)])
def test_grouped_scale_area_offsets(fmt, R, p):
    """docs/isa.md "Weight groups": a group of R rows (WG 8: int8 8, 4-bit 16), row j's Sw at
    4j, block kk's data at SWP + (kk R + j) SB."""
    D, KB, sb = 32, 3, Q.SCALE_DATA[fmt]
    h = Q.swp(fmt, 1)
    rows = np.zeros((R, Q.scale_bytes(KB * D, fmt, D)), np.uint8)
    for j in range(R):
        rows[j, :4] = np.array([j + 1], np.uint32).view(np.uint8)
        for kk in range(KB):
            rows[j, h + sb * kk:h + sb * (kk + 1)] = (np.arange(sb) + 31 * kk + j) % 127
    assert Q.grp_r(fmt, 8) == R and Q.swp(fmt, R) == p
    a = Q.group_scales(rows, fmt, KB * D, 8, D)[0]
    assert a.size == Q.grp_scale_bytes(KB * D, fmt, D, 8) == -(-(p + KB * R * sb) // p) * p
    assert a[:4 * R].view(np.uint32).tolist() == list(range(1, R + 1))
    for kk in range(KB):
        for j in range(R):
            assert a[p + sb * (kk * R + j):][:sb].tolist() == rows[j, h + sb * kk:][:sb].tolist()


def test_grouped_4bit_chunk_layout():
    """docs/isa.md "Weight groups": a 4-bit group is 2 WG rows; its chunk c is K-block c, lane l's
    byte i element i of block c of row 2l (the low nibble) and of row 2l + 1 (the high), so a
    chunk is WG x D bytes as int8's; KB odd is no half-used chunk; the rows past N are 0."""
    rng = np.random.default_rng(4)
    D, wg, N, KB = 32, 8, 21, 3
    codes = rng.integers(0, 16, (N, KB * D))
    rows = np.zeros((N, Q.row_bytes(KB * D, "fp4", D)), np.uint8)
    rows[:, :KB * D // 2] = Q.pack4(codes)
    g = Q.group_rows(rows, wg, D, "fp4", KB * D)
    assert g.shape == (2, KB * wg * D) == (2, Q.grp_row_bytes(KB * D, "fp4", D, wg))
    c = g.reshape(2, KB, wg, D)                                # [group, chunk, lane, byte]
    for n in range(32):
        q, l, hi = n // 16, n % 16 // 2, n % 2
        got = (c[q, :, l] >> 4) if hi else (c[q, :, l] & 15)
        want = codes[n].reshape(KB, D) if n < N else np.zeros((KB, D))
        assert np.array_equal(got, want), n
    assert np.array_equal(Q.ungroup_rows(g, N, wg, D, "fp4"), rows)
    assert Q.grp_bytes(N, KB * D, "fp4", D, wg) == (2 * KB * wg * D, 2 * (64 + KB * 16 * 4))


def test_grp_field_checks():
    D = 32
    rng = np.random.default_rng(5)
    x = _act(rng, 1, 1, D)
    data, rows = _matrix(rng, I.W8, 8, 1, 1, False, D)

    def run(cfg, **kw):
        dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
        _place(dram, X, x)
        _place(dram, WA, Q.group_rows(data, 8, D))
        _place(dram, SA, Q.group_scales(rows, "int8", D, 8, D))
        mm = dict(sa=WA, ssa=SA, out=OUT, n=8, kb=1, rs=8 * D, ors=8, m=1, ab=0, srs=32,
                  grp=8) | kw
        prog = [I.ld(X, 0, x.size), I.qact(0, 1, 0, 1, D), I.mm(**mm), I.halt()]
        return Machine(cfg, [prog], [dram]).run()

    run(_cfg(8))
    with pytest.raises(SimError, match="the machine's are 1"):
        run(_cfg(1))
    with pytest.raises(SimError, match="the machine's are 8"):
        run(_cfg(8), grp=4)
    with pytest.raises(SimError, match="fp4 groups of 8 rows, the machine's are 16"):
        run(_cfg(8), wf=I.W4F, unit=True)
    with pytest.raises(SimError, match="multiples of WG x D"):
        run(_cfg(8), sa=WA + D)
    with pytest.raises(SimError, match="multiples of WG x D"):
        run(_cfg(8), rs=9 * D)
    with pytest.raises(SimError, match="ssa and srs must be multiples of 32"):
        run(_cfg(8), srs=36)
    with pytest.raises(SimError, match="neither RMAX, ASCALE"):
        run(_cfg(8), rmax=True)
    run(_cfg(8, GALL=True), rmax=True)                 # GALL: RMAX and ASCALE on groups
    with pytest.raises(SimError, match="rows of one at WG > 1 under GALL"):
        run(_cfg(8, GALL=True), grp=1, rs=D, srs=4 + 4)
    with pytest.raises(ValueError, match="groups take no PAIR"):
        I.mm(WA, SA, OUT, 8, 1, 8 * D, 8, 1, 0, 32, grp=16, wf=I.W4F, pair=True)
    with pytest.raises(ValueError, match="6 rows a group"):
        I.mm(WA, SA, OUT, 8, 1, 8 * D, 8, 1, 0, 32, grp=6)
    ins = I.mm(WA, SA, OUT, 8, 1, 8 * D, 8, 1, 0, 32, grp=8)
    assert ins.rd == 3 and I.Instr.decode(ins.encode()).rd == 3
    ins = I.mm(WA, SA, OUT, 8, 1, 8 * D, 8, 1, 0, 64, grp=16, wf=I.W4F)
    assert ins.rd == 4 and I.Instr.decode(ins.encode()).rd == 4
    assert I.mm(WA, SA, OUT, 8, 1, 8 * D, 8, 1, 0, 32).rd == 0


# ------------------------------------------------------------------------- compiled kernels
def _weights_fmt(args, fmt):
    from opentpu.runtime import Weight
    return {k: Weight(v.array, v.shard, fmt) if isinstance(v, Weight) else v
            for k, v in args.items()}


@pytest.mark.parametrize("fmt", ["int8", "int4", "fp4", "nvfp4"])
@pytest.mark.parametrize("S,M,pair", [(1, 1, False), (1, 1, True), (2, 3, True), (4, 11, False)])
def test_mlp_at_wg8_is_wg1s(fmt, S, M, pair):
    """runtime.place groups every Weight at Config.WG; the programs differ only in addresses,
    strides and MM's groups, so the outputs are the same bits."""
    import test_kernels as TK
    from opentpu.kernels import mlp
    from opentpu.runtime import launch
    args, _ = TK.mlp_args(np.random.default_rng(M + 10 * S), M=M)
    args = _weights_fmt(args, fmt)
    outs = [launch(mlp, Config(S=S, MCOLS=8, PAIR=pair, WG=wg, CHASH=wg > 1),
                   **args).outputs["out"] for wg in (1, 8)]
    assert outs[0].view(np.uint32).tolist() == outs[1].view(np.uint32).tolist()


@pytest.mark.parametrize("S,pos,gall", [(1, 0, False), (2, 40, False), (1, 0, True),
                                        (2, 40, True), (1, 63, True)])
def test_attention_layer_at_wg8_is_wg1s(S, pos, gall):
    """Grouped projections around the natural KV cache (q.K^T, P.V: rows of one); GALL: the
    KV cache in groups too (QST GRP appends, q.K^T with RMAX and P.V with ASCALE grouped)."""
    import test_kernels as TK
    from opentpu.kernels import attention_layer
    from opentpu.runtime import launch
    args, _ = TK.layer_args(np.random.default_rng(pos), S, pos=pos)
    rs = [launch(attention_layer, Config(S=S, MCOLS=8, WG=wg, CHASH=wg > 1, GALL=gall and wg > 1),
                 **args) for wg in (1, 8)]
    a, b = (r.outputs["out"] for r in rs)
    assert a.view(np.uint32).tolist() == b.view(np.uint32).tolist()
    for x, y in zip(rs[0].kv("kv"), rs[1].kv("kv")):
        assert np.array_equal(x.view(np.uint32), y.view(np.uint32))
    progs = rs[1].compiled.programs[0]
    # int8 weights grouped, the KV cache too under GALL
    assert {p.rd for p in progs if p.op == I.MM} == ({3} if gall else {0, 3})
    assert any(p.op == I.QST and p.flags & I.F_QGRP for p in progs) == gall


def test_a_row_slice_off_a_group_is_refused():
    from opentpu.compiler import CompileError, QTensor
    w = QTensor.weight(0, 1 << 16, 24, 64, "int8", 32, grp=8)
    assert w.grp == 8 and w.rs == 2 * 8 * 32 and w.srs == 64   # 32 + 2 x 8, padded to 32
    v = w[8:16, 32:64]
    assert v.data.const == w.rs + 8 * 32 and v.scale.const == (1 << 16) + w.srs and v.k0 == 1
    with pytest.raises(CompileError, match="starts at a group"):
        w[4:12, :]
    # 4-bit: groups of 16 rows, a chunk a K-block (any first block)
    w = QTensor.weight(0, 1 << 16, 40, 96, "fp4", 32, grp=8)
    assert w.grp == 16 and w.rs == 3 * 8 * 32 and w.srs == 64 + 3 * 16 * 4
    v = w[16:32, 32:96]
    assert v.data.const == w.rs + 8 * 32 and v.scale.const == (1 << 16) + w.srs and v.k0 == 1
    with pytest.raises(CompileError, match="groups of 16 rows starts at a group"):
        w[8:16, :]


@pytest.mark.parametrize("wf,rsc", [(I.W8, False), (I.W8, True), (I.W4F, False), (I.WNV, False)])
def test_grouped_footprint_covers_the_reads(wf, rsc):
    """waits.footprint (otpu_pkg footprint()) of a grouped MM: the groups' chunks and their
    scale areas up to block k0 + KB's entries; at R = 1 it is the rows' footprint as before."""
    from opentpu import waits as WT
    D, N, KB, k0, wg = 32, 37, 3, 1, 8
    fmt = FMT[wf]
    R = Q.grp_r(fmt, wg)
    rs = Q.grp_row_bytes(KB * D, fmt, D, wg)
    srs = Q.grp_scale_bytes((k0 + KB) * D, fmt, D, wg, rsc)
    ins = I.mm(WA, SA, OUT, N, KB, rs, N, 1, 0, srs, wf=wf, k0=k0, rsc=rsc, grp=R)
    regs = [(0, 0)] * 16
    fp = WT.footprint(ins, regs, D)
    (_, d0, d1), (_, s0, s1) = fp.rd[0], fp.rd[1]
    G = -(-N // R)
    assert d0 == WA and d1 >= WA + (G - 1) * rs + KB * wg * D
    p, sb = Q.swp(fmt, R, rsc), (0 if rsc else Q.SCALE_DATA[fmt])
    used = 4 * R if rsc else p + (k0 + KB) * R * sb
    assert s0 == SA and s1 >= SA + (G - 1) * srs + used
    nat = I.mm(WA, SA, OUT, N, KB, rs, N, 1, 0, srs, wf=wf, k0=k0, rsc=rsc)
    nfp = WT.footprint(nat, regs, D)
    assert nfp.rd[0][2] == WA + (N - 1) * rs + KB * D


def _engines_agree(monkeypatch, spec, W, cfg, toks, prefill=0, gall=False, **kw):
    """Engines at WG 1 and WG 8 (cfg's otherwise) decode toks (the first `prefill` in one
    chunk) to the same logits bit for bit; WG 8's MMs over the image's matrices grouped (rd
    LG 3 int8, 4 a 4-bit format), the attention's over the KV cache natural, or (gall) in groups
    too (GALL: no MM of rows of one)."""
    import dataclasses
    from opentpu import isasim
    from opentpu.llm.qwen3 import Engine
    rds, mm = [], isasim.Slice._mm
    monkeypatch.setattr(isasim.Slice, "_mm", lambda s, ins: (rds.append(ins.rd), mm(s, ins))[1])
    out = []
    for wg in (1, 8):
        rds.clear()
        eng = Engine(spec, W, cap=256, cfg=dataclasses.replace(cfg, WG=wg, CHASH=wg > 1,
                                                               GALL=gall and wg > 1), **kw)
        got = [eng.prefill(toks[:prefill], chunk=prefill)] if prefill else []
        out.append(np.array(got + [eng.step(t) for t in toks[prefill:]]).view(np.uint32))
        lg = {0} if wg == 1 else {3 if kw.get("wformat") == "int8" else 4} | (
            set() if gall else {0})
        assert lg <= set(rds) <= (lg | {3}), set(rds)          # (an int8 head's LG 3)
    assert np.array_equal(*out)


@pytest.fixture(scope="module")
def tiny3():
    """test_qwen3's tiny Qwen3 (tied head)."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from opentpu.llm.qwen3 import Spec
    torch.manual_seed(0)
    hc = transformers.Qwen3Config(hidden_size=256, num_hidden_layers=2, num_attention_heads=4,
                                  num_key_value_heads=2, head_dim=128, intermediate_size=512,
                                  vocab_size=1000, rms_norm_eps=1e-6, rope_theta=1e6,
                                  tie_word_embeddings=True, max_position_embeddings=4096)
    m = transformers.Qwen3ForCausalLM(hc).float().eval()
    return {k: v.float().numpy() for k, v in m.state_dict().items()}, Spec(256, 2, 4, 2, 128,
                                                                           512, 1000)


@pytest.mark.parametrize("S,wf,head,prefill,resident,gall", [
    (2, "int8", None, 0, False, False), (1, "fp4", "int8", 5, False, False),
    (2, "nvfp4", None, 3, True, False), (1, "int4", None, 0, True, False),
    (2, "int8", None, 0, False, True), (1, "fp4", "int8", 5, False, True),
    (2, "nvfp4", None, 3, True, True)])
def test_tiny_qwen3_at_wg8_is_wg1s(tiny3, monkeypatch, S, wf, head, prefill, resident, gall):
    """Tiny Qwen3's image grouped at WG 8 (qwen3.Image; S = 1's int8 head an embedding
    table of its own): WG 1's logits, per-position, prefilled and resident decode; gall: the
    KV cache in groups too (GALL)."""
    from opentpu.isasim import design_config
    W, spec = tiny3
    toks = [int(t) for t in np.random.default_rng(0).integers(0, 1000, 8)]
    _engines_agree(monkeypatch, spec, W, design_config(S=S, DRAM_BYTES=1 << 25, PAIR=not gall),
                   toks, prefill, gall, wformat=wf, head_format=head, resident=resident)


@pytest.mark.parametrize("nk,nv,S,pair_loop,wf,prefill,resident,gall", [
    (8, 8, 2, False, "int8", 0, False, False),
    (4, 8, 2, False, "nvfp4", 5, False, False),
    (8, 16, 1, True, "fp4", 0, True, False),
    (9, 18, 1, True, "int8", 3, False, False),
    (9, 18, 1, False, "int4", 0, True, False),
    (8, 8, 2, False, "int8", 0, False, True),
    (4, 8, 2, False, "nvfp4", 5, False, True),
    (8, 16, 1, True, "fp4", 3, True, True)])
def test_tiny_qwen35_at_wg8_is_wg1s(monkeypatch, nk, nv, S, pair_loop, wf, prefill, resident,
                                    gall):
    """Tiny Qwen3.5's image grouped at WG 8 (qwen35.Image): WG 1's logits, both DeltaNet
    layouts, q and k shared (kh4), 9 pairs (in_proj_a / b's 36 rows padded to 40); gall: the
    KV cache in groups too (GALL)."""
    import dataclasses
    pytest.importorskip("transformers")
    from opentpu.isasim import design_config
    from test_qwen35 import _tiny_model
    _, W, spec = _tiny_model(nk, nv)
    spec = dataclasses.replace(spec, pair_loop=pair_loop)
    toks = [int(t) for t in np.random.default_rng(0).integers(0, 1000, 8)]
    _engines_agree(monkeypatch, spec, W, design_config(S=S, DRAM_BYTES=1 << 25, PAIR=not gall),
                   toks, prefill, gall, wformat=wf, resident=resident)


@pytest.mark.parametrize("S,wf,prefill,resident,gall", [
    (2, "int8", 0, False, False), (1, "fp4", 4, True, False), (2, "int8", 0, False, True),
    (1, "fp4", 4, True, True)])
def test_tiny_lfm2_at_wg8_is_wg1s(monkeypatch, S, wf, prefill, resident, gall):
    """Tiny LFM2's image grouped at WG 8 (lfm2.Image: the conv's in_proj in three row parts
    of a slice's channels); gall: the KV cache in groups too (GALL; heads of 64, V^T's rows
    written HALF)."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from opentpu.isasim import design_config
    from opentpu.llm.lfm2 import Spec
    kinds = ("conv", "attn", "conv", "attn", "conv")
    torch.manual_seed(0)
    hc = transformers.Lfm2Config(
        hidden_size=256, num_hidden_layers=len(kinds), num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=512, vocab_size=1000, norm_eps=1e-5,
        layer_types=["full_attention" if k == "attn" else "conv" for k in kinds],
        conv_L_cache=3, conv_bias=False, block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True, max_position_embeddings=4096,
        rope_parameters={"rope_type": "default", "rope_theta": 1e6})
    m = transformers.Lfm2ForCausalLM(hc).float().eval()
    W = {k: v.float().numpy() for k, v in m.state_dict().items()}
    toks = [int(t) for t in np.random.default_rng(0).integers(0, 1000, 8)]
    _engines_agree(monkeypatch, Spec(256, kinds, 4, 2, 64, 512, 1000), W,
                   design_config(S=S, DRAM_BYTES=1 << 25, PAIR=not gall), toks, prefill, gall,
                   wformat=wf, resident=resident)


@pytest.fixture(scope="module", params=["smollm3", "phi3"])
def tinyl(request, tmp_path_factory):
    """test_llama's tiny SmolLM3 (two layers of 8 without RoPE) and Phi-3 (Phi-4-mini's
    attention in small: the fused projections, RoPE on 96 of 128 dimensions, LongRoPE)."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from test_llama import _tiny
    return _tiny(request.param, tmp_path_factory.mktemp(request.param + "wg"))


@pytest.mark.parametrize("S,wf,prefill,resident,gall", [
    (2, "int8", 0, False, False), (1, "fp4", 4, True, False), (2, "int8", 0, False, True),
    (1, "fp4", 4, True, True), (2, "nvfp4", 3, False, True)])
def test_tiny_llama_at_wg8_is_wg1s(tinyl, monkeypatch, S, wf, prefill, resident, gall):
    """Tiny SmolLM3's and Phi-3's images grouped at WG 8 (qwen3.Image on llama.py's specs):
    WG 1's logits, per-position, prefilled and resident decode; gall: the KV cache in groups
    too (GALL: Phi-3's heads rotated on 96 of 128, SmolLM3's layers without RoPE)."""
    from opentpu.isasim import design_config
    _, _, spec, W = tinyl
    toks = [int(t) for t in np.random.default_rng(0).integers(0, 1000, 8)]
    _engines_agree(monkeypatch, spec, W, design_config(S=S, DRAM_BYTES=1 << 25, PAIR=not gall),
                   toks, prefill, gall, wformat=wf, resident=resident)


@pytest.fixture(scope="module")
def tinyg4(tmp_path_factory):
    """test_gemma4's tiny Gemma 4 (a tied head, per-layer inputs, KV-shared layers)."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from test_gemma4 import _model
    return _model(tmp_path_factory.mktemp("gemma4wg"))


@pytest.mark.parametrize("wf,prefill,resident", [("int8", 0, False), ("fp4", 4, True)])
def test_tiny_gemma4_at_wg8_is_wg1s(tinyg4, monkeypatch, wf, prefill, resident):
    """Tiny Gemma 4's image grouped at WG 8 (gemma4.Image: the LM head's MM on a grouped
    copy of the gather table's rows): WG 1's logits, per-position and resident decode."""
    from opentpu.isasim import board_config
    pytest.importorskip("transformers.models.gemma4")
    _, W, spec = tinyg4
    toks = [int(t) for t in np.random.default_rng(1).integers(0, 1000, 7)]
    _engines_agree(monkeypatch, spec, W, board_config(DRAM_BYTES=1 << 27, MCOLS=8, PAIR=True),
                   toks, prefill, wformat=wf, resident=resident)


# ------------------------------------------------------------------ GALL (docs/weight_groups.md)
@pytest.mark.parametrize("chash", [False, True])
@pytest.mark.parametrize("KB,row,n0,rows", [(1, True, 0, 1), (2, False, 13, 1), (3, False, 5, 11),
                                            (1, True, 30, 9)])
def test_qst_grp_rows_are_the_grouped_matrix(chash, KB, row, n0, rows):
    """QST GRP (rows n0 .. of a grouped int8 matrix at R[rd] + drs, its scale areas es apart)
    writes the bytes quant.grouped (and hash_rows) gives the natural QST's rows; the other
    rows' bytes stay."""
    D, wg, N = 32, 8, 40
    cfg = _cfg(wg, D, CHASH=chash, GALL=True)
    rng = np.random.default_rng(KB * 100 + n0)
    x = rng.standard_normal((rows, KB * D)).astype(np.float32)
    srs = Q.grp_scale_bytes(KB * D, "int8", D, wg, rsc=row)
    nat = {}
    for grp in (False, True):
        dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
        dram[WA:WA + 1 << 14] = 7                          # the rows QST does not write
        dram[SA:SA + 1 << 12] = 7
        _place(dram, X, x)
        prog = [I.li(1, n0 - 3), I.ld(X, 0, x.size)]
        if grp:
            prog.append(I.qst(0, WA, SA, rows, KB, KB * D, 3, srs, row=row, grp=wg, rd=1))
        else:
            sb = 4 if row else Q.scale_bytes(KB * D, "int8", D)
            prog.append(I.qst(0, WA + n0 * KB * D, SA + n0 * sb, rows, KB, KB * D, KB * D, 1,
                              row=row))
        sl = Machine(cfg, [prog + [I.halt()]], [dram]).run().slices[0]
        nat[grp] = sl.dram
    # both over a field of 7s: the natural store's matrix grouped is the grouped store's
    q = nat[False][WA:WA + N * KB * D].reshape(N, KB * D)
    want = Q.group_rows(q, wg, D).reshape(-1)
    if chash:
        want = Q.hash_rows(want, WA, wg, D)
    assert np.array_equal(nat[True][WA:WA + want.size], want)
    sb = 4 if row else Q.scale_bytes(KB * D, "int8", D)
    s = nat[False][SA:SA + N * sb].reshape(N, sb)
    back = Q.ungroup_scales(nat[True][SA:SA + N // wg * srs].reshape(N // wg, srs), "int8",
                            KB * D, N, wg, D, rsc=row)
    nb = 4 if row else 4 + KB                              # S and the offset bytes
    assert np.array_equal(back[:, :nb], s[:, :nb])


@pytest.mark.parametrize("chash", [False, True])
@pytest.mark.parametrize("d,n0,rows,half", [(64, 0, 1, False), (64, 37, 1, True),
                                            (128, 21, 30, False)])
def test_qst_grp_columns_are_the_grouped_transpose(chash, d, n0, rows, half):
    """QST GRP COL (a token's d elements as column n of a grouped [d, cap] matrix: V^T): the
    bytes the natural column store's [d, cap] gives grouped (quant.group_rows, hash_rows)."""
    D, wg, cap = 32, 8, 64
    cfg = _cfg(wg, D, CHASH=chash, GALL=True)
    rng = np.random.default_rng(d + n0)
    x = rng.standard_normal((rows, d)).astype(np.float32)
    gs = (cap // D) * wg * D                               # the group stride: cap / D chunks
    out = {}
    for grp in (False, True):
        dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
        _place(dram, X, x)
        prog = [I.li(1, n0), I.ld(X, 0, x.size)]
        if grp:
            prog.append(I.qst(0, WA, SA + 4 * n0, rows, d // D, d, 0, gs, row=True, half=half,
                              grp=wg, col=True, rd=1))
        else:
            prog.append(I.qst(0, WA + n0, SA + 4 * n0, rows, d // D, d, 1, cap, row=True,
                              half=half))
        out[grp] = Machine(cfg, [prog + [I.halt()]], [dram]).run().slices[0].dram
    vt = out[False][WA:WA + d * cap].reshape(d, cap)
    want = Q.group_rows(vt, wg, D).reshape(-1)
    if chash:
        want = Q.hash_rows(want, WA, wg, D)
    assert np.array_equal(out[True][WA:WA + want.size], want)
    assert np.array_equal(out[True][SA:SA + 4 * cap], out[False][SA:SA + 4 * cap])


@pytest.mark.parametrize("col,KB,row,n0,rows", [(False, 2, False, 13, 1), (False, 3, True, 5, 11),
                                                (True, 2, True, 37, 1), (True, 4, True, 21, 30)])
def test_qst_grp_footprint_covers_the_writes(col, KB, row, n0, rows):
    """waits.footprint of QST GRP (rows R[rd] + w6 ..: the groups or the column chunks they
    reach, and their scale areas) holds every byte the ISA simulator's QST GRP writes, with the
    channel hash."""
    from opentpu import waits as WT
    D, wg, cap = 32, 8, 128
    cfg = _cfg(wg, D, CHASH=True, GALL=True)
    x = np.random.default_rng(n0).standard_normal((rows, KB * D)).astype(np.float32)
    es = (cap // D) * wg * D if col else Q.grp_scale_bytes(KB * D, "int8", D, wg, rsc=row)
    ins = I.qst(0, WA, SA + (4 * n0 if col else 0), rows, KB, KB * D, 3, es, row=row,
                grp=wg, col=col, rd=1)
    dram = np.full(cfg.DRAM_BYTES, 0xA5, np.uint8)
    _place(dram, X, x)
    before = dram.copy()
    prog = [I.li(1, n0 - 3), I.ld(X, 0, x.size), ins, I.halt()]
    after = Machine(cfg, [prog], [dram]).run().slices[0].dram
    regs = [(0, 0)] * 16
    regs[1] = (n0 - 3, n0 - 3)
    fp = WT.footprint(ins, regs, D)
    inside = np.zeros(cfg.DRAM_BYTES, bool)
    for sp, lo, hi in fp.wr:
        if sp == WT.SP_DRAM:
            inside[lo:hi] = True
    changed = np.flatnonzero(after != before)
    assert changed.size and inside[changed].all()
    assert int(inside.sum()) < cfg.DRAM_BYTES // 16             # (not the whole DRAM)


def test_qst_grp_needs_gall():
    cfg = _cfg(8, 32)
    prog = [I.qst(0, WA, SA, 1, 1, 32, 0, 32, row=True, grp=8), I.halt()]
    with pytest.raises(SimError, match="GRP needs WG > 1 and GALL"):
        Machine(cfg, [prog], [np.zeros(cfg.DRAM_BYTES, np.uint8)]).run()


@pytest.mark.parametrize("wf,experts", [("fp4", None), ("int8", 2), ("nvfp4", 2), ("fp4", 2)])
def test_tiny_qwen35_moe_gall_is_wg1s(monkeypatch, wf, experts):
    """Tiny Qwen3.5 MoE at WG 8 under GALL: the experts in groups in their slots (moe.
    ExpertFormat grp: the expert server writes their data's lanes in the channel hash's order),
    the KV cache in groups, no MM of rows of one; WG 1's logits bit for bit, the experts
    resident or streamed into 2 slots a layer."""
    pytest.importorskip("transformers")
    from opentpu import isasim
    from opentpu.llm.qwen3 import Engine, PREFILL_ROWS, device_config
    from test_qwen35_moe import _tiny
    _, W, spec = _tiny()
    toks = [int(t) for t in np.random.default_rng(2).integers(0, 1000, 6)]
    rds, mm = [], isasim.Slice._mm
    monkeypatch.setattr(isasim.Slice, "_mm", lambda s, ins: (rds.append(ins.rd), mm(s, ins))[1])
    out = []
    for wg in (1, 8):
        rds.clear()
        cfg = device_config(spec, 256, S=1, rows=PREFILL_ROWS, experts=experts, wformat=wf,
                            MCOLS=8, WG=wg, CHASH=wg > 1, GALL=wg > 1)
        eng = Engine(spec, W, cap=256, cfg=cfg, experts=experts, wformat=wf)
        out.append(np.array([eng.step(t) for t in toks]).view(np.uint32))
        assert (0 in rds) == (wg == 1), set(rds)
    assert np.array_equal(*out)


@pytest.mark.parametrize("wf,experts", [("int8", 2), ("nvfp4", 2), ("fp4", None)])
def test_tiny_lfm2_moe_gall_is_wg1s(monkeypatch, wf, experts):
    """Tiny LFM2-MoE (its first layer dense) at WG 8 under GALL: the experts in groups in their
    slots (lfm2.Image's ExpertFormat grp, the server's lanes by the slots' addresses), resident
    or streamed into 2 slots a layer; WG 1's logits bit for bit, no MM of rows of one."""
    pytest.importorskip("transformers")
    from opentpu import isasim
    from opentpu.llm.qwen3 import Engine, PREFILL_ROWS, device_config
    from test_lfm2_moe import _tiny
    _, W, spec = _tiny()
    toks = [int(t) for t in np.random.default_rng(2).integers(0, 1000, 6)]
    rds, mm = [], isasim.Slice._mm
    monkeypatch.setattr(isasim.Slice, "_mm", lambda s, ins: (rds.append(ins.rd), mm(s, ins))[1])
    out = []
    for wg in (1, 8):
        rds.clear()
        cfg = device_config(spec, 256, S=1, rows=PREFILL_ROWS, experts=experts, wformat=wf,
                            MCOLS=8, WG=wg, CHASH=wg > 1, GALL=wg > 1)
        eng = Engine(spec, W, cap=256, cfg=cfg, experts=experts, wformat=wf)
        assert eng.image.fmt.grp == (1 if wg == 1 else 8)
        out.append(np.array([eng.step(t) for t in toks]).view(np.uint32))
        assert (0 in rds) == (wg == 1), set(rds)
        if experts:
            assert eng.server.misses > len(toks)
    assert np.array_equal(*out)


def test_tiny_qwen35_moe_gall_embedding_rows_from_the_host(monkeypatch):
    """Tiny Qwen3.5 MoE with an int8 embedding table of its own on the host (embed_host) at WG
    8 under GALL and the channel hash: the slot's records the host writes before a run and the
    row server's in the card's generate loop, gathered in groups; WG 1's logits and tokens bit
    for bit, no MM of rows of one."""
    pytest.importorskip("transformers")
    from opentpu import isasim
    from opentpu.llm.qwen3 import Engine, device_config
    from test_qwen35_moe import K, _tiny, _untied
    spec, W = _untied(_tiny())
    toks = [int(t) for t in np.random.default_rng(3).integers(0, 1000, 6)]
    rds, mm = [], isasim.Slice._mm
    monkeypatch.setattr(isasim.Slice, "_mm", lambda s, ins: (rds.append(ins.rd), mm(s, ins))[1])
    out = []
    for wg in (1, 8):
        rds.clear()
        cfg = device_config(spec, 512, rows=1, lookup=True, S=1, experts=K, MCOLS=8, WG=wg,
                            CHASH=wg > 1, GALL=wg > 1)
        eng = Engine(spec, W, cap=512, cfg=cfg, rows=1, resident=True, experts=K)
        assert eng.image.embed_host and eng.row_server is not None
        lg = eng.prefill(toks)
        got = eng.generate_card(int(np.argmax(lg)), 6, stop_ids=[])
        out.append((lg.view(np.uint32), got, eng.row_server.seq))
        assert (0 in rds) == (wg == 1), set(rds)
    assert np.array_equal(out[0][0], out[1][0]) and out[0][1:] == out[1][1:]
    assert out[1][2] >= 5


def test_gall_context_restore_is_the_context():
    """serve/kvio's pages under GALL (the KV caches in groups of 8 tokens, CHASH): tiny Qwen3's
    300-token context (its second page 44 tokens: half a group past the last whole one) saved,
    a 310-token context run over it, the first written back at its position: the next tokens'
    logits and greedy steps equal a fresh engine fed it, bit for bit."""
    pytest.importorskip("transformers")
    from opentpu.llm.qwen3 import PREFILL_ROWS, Engine, device_config
    from opentpu.serve import kvio as KV
    from test_autodecode import _tiny
    W, spec = _tiny("qwen3")
    cfg = device_config(spec, 512, rows=PREFILL_ROWS, lookup=True, S=1, MCOLS=8, WG=8,
                        CHASH=True, GALL=True, PAIR=False)

    def make():
        return Engine(spec, W, cap=512, cfg=cfg, resident=True, prompt_runs=True)
    r = np.random.default_rng(1)
    A, B = ([int(x) for x in r.integers(0, 1000, n)] for n in (300, 310))
    nxt = [int(x) for x in r.integers(0, 1000, 9)]
    eng = make()
    io = KV.ContextIO(eng, area=None)
    assert io.layout.wg == 8
    eng.prefill(A)
    pages = io.layout.split(len(A))
    saved = io.read_pages(pages)
    eng.reset()
    eng.prefill(B)
    io.write_pages([(i, m, b) for (i, m), b in zip(pages, saved)])
    eng.adopt(len(A))
    got = eng.prefill(nxt)
    ref = make()
    assert np.array_equal(got.view(np.uint32), ref.prefill(A + nxt).view(np.uint32))
    t = int(np.argmax(got))
    for _ in range(3):
        x, y = eng.step(t), ref.step(t)
        assert np.array_equal(x.view(np.uint32), y.view(np.uint32))
        t = int(np.argmax(x))


@pytest.mark.parametrize("case", ["qwen3", "qwen35"])
def test_gall_rollback_then_feed_is_a_fresh_prefill(case, monkeypatch):
    """test_chatkv's turns under GALL (WG 8, CHASH: the KV caches in groups of 8 tokens): a
    context rolled back (to 21 and 30, inside a group; Qwen3.5 to its snapshot) and fed the rest
    gives a fresh engine's logits and greedy tokens bit for bit, no MM of rows of one."""
    pytest.importorskip("transformers")
    import dataclasses
    from opentpu import isasim
    from opentpu.llm.qwen3 import Engine
    from test_chatkv import TURNS, _model, _turns
    W, spec, cfg = _model(case)
    cfg = dataclasses.replace(cfg, MCOLS=8, WG=8, CHASH=True, GALL=True, PAIR=False)
    rds, mm = [], isasim.Slice._mm
    monkeypatch.setattr(isasim.Slice, "_mm", lambda s, ins: (rds.append(ins.rd), mm(s, ins))[1])

    def make(snapshots=0):
        return Engine(spec, W, cap=512, cfg=cfg, resident=True, prompt_runs=True,
                      snapshots=snapshots)
    got = _turns(make, TURNS)
    assert got == [0, 21, 26 if case == "qwen35" else 30] and rds and 0 not in rds


@pytest.mark.parametrize("fmt", ["int8", "fp4"])
def test_gall_mtp_loop_is_plain_greedy(fmt, monkeypatch):
    """MTP under GALL (WG 8, CHASH): the device's MTP loop (its own drafts, and all-right ones)
    and the host's give plain greedy decode's tokens at WG 1, no MM of rows of one."""
    pytest.importorskip("transformers")
    from opentpu import isasim
    from opentpu.isasim import board_config
    from opentpu.llm.mtp import MTPDecoder, mtp_engine
    from opentpu.llm.qwen3 import Engine
    from test_mtp import FP4, _mtp_weights
    from test_qwen35 import _tiny_model
    _, W, spec = _tiny_model(8, init=0.02)
    W = _mtp_weights(W, spec)
    kw = FP4 if fmt == "fp4" else {}
    base = dict(DRAM_BYTES=1 << 26, DSTEP=True, STREAM=True)
    c8 = board_config(**base, PAIR=False, MCOLS=8, WG=8, CHASH=True, GALL=True)
    P, N = 30, 8
    prompt = [int(t) for t in np.random.default_rng(P).integers(0, 1000, P)]
    want = Engine(spec, W, cap=512, cfg=board_config(**base, PAIR=True), resident=True,
                  **kw).generate(prompt, max_new=N)
    rds, mm = [], isasim.Slice._mm
    monkeypatch.setattr(isasim.Slice, "_mm", lambda s, ins: (rds.append(ins.rd), mm(s, ins))[1])
    right = np.zeros(514, np.float32)
    right[P:P + N] = want
    for drafts in (None, right):
        st = MTPDecoder(mtp_engine(spec, W, cap=512, cfg=c8, **kw)).generate_card(
            prompt, max_new=N, drafts=drafts)
        assert st.tokens == want and (drafts is None or sum(st.accepted) >= 2)
    assert MTPDecoder(mtp_engine(spec, W, cap=512, cfg=c8, **kw)).generate(
        prompt, max_new=N).tokens == want
    assert rds and 0 not in rds


@pytest.mark.parametrize("kw", [{}, {"resident": True}, {"embed_host": True}])
def test_tiny_27b_shape_gall_is_natural(monkeypatch, kw):
    """test_qwen35's tiny 27B shape on bcu1525-x2's configuration (4 slices, WG 8, the channel
    hash): under GALL (its KV cache and its int8 embedding gathers in groups, no MM of rows of
    one) the logits of rows of one, decode and a 5-row prefill, bit for bit; the table on the
    card split by columns, the resident decode, the table on the host (embed_host: the slot's
    records the host writes, hashed by their row)."""
    pytest.importorskip("transformers")
    import dataclasses
    from opentpu import isasim
    from opentpu.llm.qwen3 import Engine
    from test_qwen35 import _tiny27
    _, W, spec, cfg = _tiny27()
    toks = [int(t) for t in np.random.default_rng(0).integers(0, 1024, 8)]
    rds, mm = [], isasim.Slice._mm
    monkeypatch.setattr(isasim.Slice, "_mm", lambda s, ins: (rds.append(ins.rd), mm(s, ins))[1])
    out = []
    for gall in (False, True):
        rds.clear()
        eng = Engine(spec, W, cap=256, cfg=dataclasses.replace(cfg, GALL=gall), wformat="fp4",
                     **kw)
        got = [eng.prefill(toks[:5], chunk=5)]
        out.append(np.array(got + [eng.step(t) for t in toks[5:]]).view(np.uint32))
        assert (0 in rds) == (not gall), set(rds)
    assert np.array_equal(*out)


def _gall_pair(cfg):
    """(WG 1, WG 8 under GALL with the channel hash) of cfg at MCOLS 8, PAIR off."""
    import dataclasses
    c = dataclasses.replace(cfg, MCOLS=8, PAIR=False)
    return c, dataclasses.replace(c, WG=8, CHASH=True, GALL=True)


@pytest.mark.parametrize("head", ["int8", "fp4"])
@pytest.mark.parametrize("resident", [False, True])
def test_tiny_qwen3_int8_embedding_gall_is_wg1s(tiny3, monkeypatch, head, resident):
    """Spec.embed "int8" (SmolLM3, Phi, LFM2.5-8B-A1B) under GALL: the token's int8 row
    gathered in groups from an int8 table of the image's own (at WG 1 the tied int8 head's
    rows; a grouped head's rows are not a table's), in a prefill and decode steps: WG 1's
    logits bit for bit, no MM of rows of one."""
    import dataclasses
    from opentpu import isasim
    from opentpu.isasim import board_config
    from opentpu.llm.qwen3 import Engine
    W, spec = tiny3
    spec = dataclasses.replace(spec, embed="int8")
    wf = "int8" if head == "int8" else "fp4"
    toks = [int(t) for t in np.random.default_rng(5).integers(0, 1000, 12)]
    rds, mm = [], isasim.Slice._mm
    monkeypatch.setattr(isasim.Slice, "_mm", lambda s, ins: (rds.append(ins.rd), mm(s, ins))[1])
    out = []
    for cfg in _gall_pair(board_config(DRAM_BYTES=1 << 25)):
        rds.clear()
        e = Engine(spec, W, cap=256, cfg=cfg, resident=resident, wformat=wf, head_format=head)
        assert e.image.lookup["own"] == (head != "int8" or cfg.WG > 1)
        lg = [e.prefill(toks[:6])] + [e.step(t) for t in toks[6:]]
        out.append(np.array(lg).view(np.uint32))
        assert (0 in rds) == (not cfg.GALL), set(rds)
    assert np.array_equal(*out)


@pytest.mark.parametrize("wf,hf,ple,host", [("int8", "int8", "int8", "0"),
                                            ("fp4", "fp4", "fp4", "0"),
                                            ("fp4", "fp4", "int4", "1"),
                                            ("int8", "int8", "int8", "1")])
def test_tiny_gemma4_gall_is_wg1s(tinyg4, monkeypatch, wf, hf, ple, host):
    """Tiny Gemma 4 under GALL (WG 8, the channel hash; docs/weight_groups.md): its gathers
    in groups (the LM head's rows and the PLE records as grouped matrices of their blocks,
    4-bit ones too, int8's one-hot operand), its K / V caches in groups (_KV: K's groups and
    their scale areas in the K region): a chunked prefill, resident steps and the card's
    generate loop give WG 1's logits and tokens bit for bit, no MM of rows of one; also with
    the PLE table on the host (the row server's records hashed by their slot row)."""
    from opentpu import isasim
    from opentpu.isasim import board_config
    from opentpu.llm.qwen3 import Engine
    pytest.importorskip("transformers.models.gemma4")
    _, W, spec = tinyg4
    monkeypatch.setenv("OTPU_PLE_FORMAT", ple)
    monkeypatch.setenv("OTPU_PLE_HOST", host)
    toks = [int(t) for t in np.random.default_rng(1).integers(0, 1000, 34)]
    rds, mm = [], isasim.Slice._mm
    monkeypatch.setattr(isasim.Slice, "_mm", lambda s, ins: (rds.append(ins.rd), mm(s, ins))[1])
    out = []
    for cfg in _gall_pair(board_config(DRAM_BYTES=1 << 27)):
        rds.clear()
        e = Engine(spec, W, cap=1024, cfg=cfg, wformat=wf, head_format=hf, resident=True)
        assert (e.image.ple_format, e.image.ple_host) == (ple, host == "1")
        lg = [e.prefill(toks[:30], chunk=4)] + [e.step(t) for t in toks[30:]]
        gen = e.generate_card(int(np.argmax(lg[-1])), 6, stop_ids=[])
        out.append((np.array(lg).view(np.uint32), gen))
        assert (0 in rds) == (not cfg.GALL), set(rds)
    assert np.array_equal(out[0][0], out[1][0]) and out[0][1] == out[1][1]


@pytest.mark.parametrize("wf,R,em", [("fp4", 2, False), ("int8", 2, True)])
def test_tiny_gemma4_moe_gall_is_wg1s(tmp_path_factory, monkeypatch, wf, R, em):
    """test_gemma4_moe's tiny MoE under GALL: the experts in groups in their slots (lane-hashed
    by the expert server), 2 R slots a layer (streaming), the layer-major prefill in runs of R
    rows (expert-major: em), resident steps and the card's loop: WG 1's logits and tokens bit
    for bit, no MM of rows of one."""
    pytest.importorskip("transformers.models.gemma4")
    from opentpu import isasim
    from opentpu.isasim import board_config
    from opentpu.llm import gemma4 as G4
    from opentpu.llm.qwen3 import Engine
    from test_gemma4_moe import _model
    _, W, spec = _model(tmp_path_factory, "moe", enable_moe_block=True, num_experts=8,
                        top_k_experts=2, moe_intermediate_size=96)
    monkeypatch.setattr(G4, "PREFILL_CHUNK", 100)
    toks = [int(t) for t in np.random.default_rng(7).integers(0, 1000, 30)]
    rds, mm = [], isasim.Slice._mm
    monkeypatch.setattr(isasim.Slice, "_mm", lambda s, ins: (rds.append(ins.rd), mm(s, ins))[1])
    out = []
    for cfg in _gall_pair(board_config(DRAM_BYTES=1 << 27)):
        rds.clear()
        e = Engine(spec, W, cap=1024, cfg=cfg, rows=1, wformat=wf, experts=2 * R,
                   layer_major=R, expert_major=em, resident=True)
        lg = [e.prefill(toks)]
        lg.append(e.step(int(np.argmax(lg[0]))))
        gen = e.generate_card(int(np.argmax(lg[-1])), 5, stop_ids=[])
        out.append((np.array(lg).view(np.uint32), gen, e.server.misses))
        assert (0 in rds) == (not cfg.GALL), set(rds)
    assert np.array_equal(out[0][0], out[1][0]) and out[0][1:] == out[1][1:]
    assert out[1][2] > 0                            # (experts streamed into the slots)


def test_gall_batch_steps_are_wg1s(monkeypatch):
    """Batched decode steps (opentpu/llm/batch.py: qwen3's rows kernel, each row appending to
    its own slot's KV cache at its own position) under GALL: slots prefilled to different
    lengths, then steps of greedy and sampled rows with rows leaving and joining: WG 1's
    prefill logits and picks bit for bit, the slots' caches in groups (QST GRP at each row's
    run-time position), no MM of rows of one."""
    import dataclasses
    pytest.importorskip("torch")
    from opentpu import isasim
    from opentpu.llm import batch as BA
    from opentpu.llm import generate as G
    from opentpu.llm.qwen3 import PREFILL_ROWS, Engine, device_config
    from test_serve_kvio import _model
    W, spec, _ = _model("qwen3")
    base = device_config(spec, 512, batch=4, rows=PREFILL_ROWS, lookup=True, S=1)
    base = dataclasses.replace(base, DRAM_BYTES=2 * base.DRAM_BYTES)
    rng = np.random.default_rng(0)
    prompts = {s: [int(x) for x in rng.integers(0, 1000, n)] for s, n in ((0, 21), (1, 9),
                                                                          (3, 30))}
    rds, mm = [], isasim.Slice._mm
    monkeypatch.setattr(isasim.Slice, "_mm", lambda s, ins: (rds.append(ins.rd), mm(s, ins))[1])
    out = []
    for cfg in _gall_pair(base):
        rds.clear()
        eng = Engine(spec, W, cap=512, cfg=cfg, batch=4, resident=True, prompt_runs=True,
                     batch_area=True)
        st = BA.Steps(eng)
        lgs, nxt = [], {}
        for s, p in prompts.items():
            lgs.append(eng.prefill(p, seq=s))
            nxt[s] = int(np.argmax(lgs[-1]))
            st.set_sampling(s, G.Sampling(0.8, 20, 0.9, 1.3) if s == 1 else None,
                            p + [nxt[s]])
        picks, u = [], np.random.default_rng(5)
        for i in range(5):
            live = [0, 3] if i == 2 else [0, 1, 3]
            rows = [(s, eng.poss[s]) for s in live]
            got = st.run(rows, [nxt[s] for s in live],
                         uniforms=u.random(len(live)).astype(np.float32))
            nxt.update(zip(live, got))
            picks.append(list(got))
        out.append((np.array(lgs).view(np.uint32), picks))
        assert (0 in rds) == (not cfg.GALL), set(rds)
    assert np.array_equal(out[0][0], out[1][0]) and out[0][1] == out[1][1]


@pytest.mark.parametrize("fmt", ["int8", "attn=fp4,mlp=fp4,wm=fp4,proj=fp4"])
@pytest.mark.parametrize("table", [False, True], ids=["static", "table"])
def test_gall_vision_tower_is_wg1s(fmt, table, monkeypatch):
    """MiniCPM-V's tower, window merger and projector (kernels/vit.py) under GALL (WG 8, the
    channel hash): the pieces grouped (kernels/pieces.py wg; resident, hashed at their
    addresses), a head's q / k / v rows padded to whole groups (heads of 40: 48 rows at fp4),
    the KV scratch in groups (K by QST GRP, V^T's columns by QST GRP COL): a crop's rows give
    WG 1's bit for bit, no MM of rows of one; at static addresses and from the table."""
    from opentpu import isasim
    from opentpu.isasim import board_config
    from opentpu.kernels.lib import gelu_tanh
    from opentpu.llm import minicpmv as MV
    from opentpu.llm.qwen3 import IsaBackend, _Bump
    from test_minicpmv import _tiny_vision_weights
    vs = MV.VisionSpec(hidden=640, heads=16, ffn=300, layers=3, side=5, insert=1, out=256)
    W = _tiny_vision_weights(vs, np.random.default_rng(0))
    P = np.random.default_rng(1).uniform(-1, 1, (96, 3 * vs.patch ** 2)).astype(np.float32)
    rds, mm = [], isasim.Slice._mm
    monkeypatch.setattr(isasim.Slice, "_mm", lambda s, ins: (rds.append(ins.rd), mm(s, ins))[1])
    out = []
    for cfg in _gall_pair(board_config(DRAM_BYTES=1 << 26)):
        rds.clear()
        vi = MV.VisionImage(vs, cfg, _Bump(), nmax=256, out_rows=64, fmt=fmt, table=table)
        img = np.zeros(cfg.DRAM_BYTES, np.uint8)

        def put(addr, a):
            v = np.ascontiguousarray(a).view(np.uint8).reshape(-1)
            img[addr:addr + v.size] = v
        vi.build(W, put)
        out.append(vi.encode(IsaBackend(cfg, [img]), P, (8, 12), proj_gelu=gelu_tanh))
        assert (0 in rds) == (not cfg.GALL), set(rds)
        assert vi.dq == (48 if cfg.GALL and fmt != "int8" else 40)
    assert np.array_equal(out[0].view(np.uint32), out[1].view(np.uint32))


@pytest.mark.parametrize("pool", [False, True])
def test_gall_staged_vision_is_wg1s(pool, tmp_path, tmp_path_factory, monkeypatch):
    """test_minicpmv's staged vision under GALL (WG 8, the channel hash): the grouped pieces
    are the pool's entries past the experts, placed in expert slots by the server, each
    piece's data lane-hashed by its slot (ExpertServer entry_lanes: kernels.pieces.Layout.
    data_regions); two images' crops give the resident tower's rows at WG 1 bit for bit, then
    a prompt's layer-major prefill (the evicted experts missing) and two decode steps its
    logits; pool: the entries in a pool file."""
    from types import SimpleNamespace
    from opentpu import isasim
    from opentpu.llm import load_spec
    from opentpu.llm import minicpmv as MV
    from opentpu.llm.qwen3 import Engine, device_config, load_weights
    from dataclasses import replace
    import test_minicpmv as TM
    m, d = TM.tiny.__wrapped__(tmp_path_factory)
    spec = replace(load_spec(d), vision_patches=128, vision_rows=64)
    W = dict(load_weights(d).items())
    W.update(TM._tiny_vision_weights(spec.vision, np.random.default_rng(7)))
    base = device_config(spec, 256, rows=1, lookup=True, S=1, experts=TM.E)
    rds, mm = [], isasim.Slice._mm
    monkeypatch.setattr(isasim.Slice, "_mm", lambda s, ins: (rds.append(ins.rd), mm(s, ins))[1])
    rng = np.random.default_rng(9)
    ims = [SimpleNamespace(patches=[rng.uniform(-1, 1, (h * w, 3 * 14 * 14)).astype(
        np.float32)], sizes=[(h, w)], tokens=h * w // 16) for h, w in ((8, 8), (4, 12))]
    toks = [int(t) for t in rng.integers(0, TM.V - 40, 21)]
    out = []
    for cfg, staged in zip(_gall_pair(base), (False, True)):
        rds.clear()
        kw = dict(cap=256, cfg=cfg, rows=1, resident=True, experts=TM.E, layer_major=2)
        e = Engine(replace(spec, vision_staged=staged), W,
                   **(dict(kw, pool_file=tmp_path / "pool.bin") if pool and staged else kw))
        assert e.image.vision.resident != staged
        n = MV.Vision(e).encode(ims)
        rows = e.backend.read(0, e.image.vision.work["vis"], 4 * n * spec.vision.out)
        la = e.prefill_layers(toks)
        lg = [la]
        t = int(np.argmax(la))
        for _ in range(2):
            lg.append(e.step(t))
            t = int(np.argmax(lg[-1]))
        out.append((np.asarray(rows).view(np.uint32), np.array(lg).view(np.uint32)))
        assert (0 in rds) == (not cfg.GALL), set(rds)
    assert n == 7 and np.array_equal(out[0][0], out[1][0])
    assert np.array_equal(out[0][1], out[1][1])
