"""Expert-major MoE prefill on a resident pool (docs/pf500.md):
every expert in DRAM, no expert server, the chunk's records in the image's low area (Image(em=True)),
the layer's mixer in tiles of T rows (tiled.compile_layer_tile) and one expert run a layer
(moe.moe_expert_run, its passes at the pool's places: moe.em_slot)."""
import dataclasses
import hashlib

import numpy as np
import pytest

from opentpu.llm.qwen3 import Engine, device_config


@pytest.mark.parametrize("wf,pipe", [("int8", True), ("fp4", True), ("int8", False)])
def test_resident_expert_major_is_layer_majors_bits(monkeypatch, wf, pipe):
    """Qwen3.5-MoE in the 35B's shape, every expert in a resident pool (OTPU_POOL=1): a
    prompt of 43 rows expert-major (tiles of 8 rows, an expert run a layer, the records in
    the image) against layer-major runs of R = 2 (moe_ffn_rows): the logits and the decode
    steps after, bit for bit; no server, no need line. pipe: the passes software-pipelined
    (EM_PIPE), else a pass at a time (rows_of_pass: the rows' norm in their QACT)."""
    import test_qwen35_moe as TQM
    from opentpu.llm import moe as MO
    from opentpu.llm import qwen35 as Q35
    monkeypatch.setattr(Q35, "PREFILL_CHUNK", 24)
    monkeypatch.setattr(MO, "EM_PIPE", pipe)
    monkeypatch.setenv("OTPU_POOL", "1")
    spec, W = TQM._untied(TQM._tiny(4))
    cfg = device_config(spec, 256, rows=1, lookup=True, S=1, wformat=wf, MCOLS=4, PAIR=True,
                        DSTEP=True, STREAM=True)
    kw = dict(cap=256, cfg=dataclasses.replace(cfg, ACT_ROWS=16), rows=1, resident=True,
              wformat=wf, layer_major=2)
    a = Engine(spec, W, expert_major=True, tile_rows=8, **kw)
    b = Engine(spec, W, **kw)
    for e in (a, b):
        assert e.image.pool is not None and e.server is None
    assert "em" in a.image.io and "em" not in b.image.io
    assert a.expert_major and a.layer_tile and a.tile_rows == 8
    toks = [int(t) for t in np.random.default_rng(12).integers(0, 1000, 43)]
    la, lb = a.prefill(toks), b.prefill(toks)
    keys = list(a._layer_runs)
    assert any(isinstance(k, tuple) and k[-1] == "tile" for k in keys)
    assert any(isinstance(k, tuple) and k[0] == "x" for k in keys)
    assert np.array_equal(la.view(np.uint32), lb.view(np.uint32))
    t = int(np.argmax(la))
    for _ in range(3):
        ga, gb = a.step(t), b.step(t)
        assert np.array_equal(ga.view(np.uint32), gb.view(np.uint32))
        t = int(np.argmax(ga))


@pytest.mark.parametrize("S,kinds,shape,rows", [
    (2, ("linear", "attn"), {}, 8),
    (2, ("linear", "attn"), {}, 16),                       # (the shared expert: 16 rows a pass)
    (4, ("linear", "attn"), dict(nv=8, H=512, F=512), 8),  # (2 KV heads: each on two slices)
])
def test_tensor_parallel_expert_major_is_layer_majors_bits(monkeypatch, S, kinds, shape, rows):
    """At S slices (bcu1525-wg's shape, the experts' parts in each slice's pool; GALL: passes
    of their entries' exact rows, up to 16): the layer tiles' heads gathered before W_o (_o_residual's operand) and the expert run's passes in
    _moe_ffn_tp's order (_swiglu_rows_tp: gate / up on the slice's columns, h gathered, W_down's
    rows in S = 1's parts), the combine on the slice's columns: layer-major R = 2's logits and
    decode steps bit for bit. Tiles of 16 rows: the prologue's shared expert in one pass of
    the 16 (moe._shared_rows: past MCOLS 8), before the top-k's four passes of TOPK_ROWS."""
    from test_moe_s2 import _engine, _tiny
    from opentpu.llm import qwen35 as Q35
    monkeypatch.setattr(Q35, "PREFILL_CHUNK", 24)
    W, spec = _tiny(kinds, **shape)
    kw = dict(rows=1, resident=True, layer_major=2)
    a = _engine(spec, W, S, monkeypatch, expert_major=True, tile_rows=rows, **kw)
    b = _engine(spec, W, S, monkeypatch, **kw)
    assert a.image.fmt.S == S and a.server is None and "em" in a.image.io
    assert a.expert_major and a.layer_tile
    toks = [int(t) for t in np.random.default_rng(13).integers(0, 1000, 43)]
    la, lb = a.prefill(toks), b.prefill(toks)
    assert any(isinstance(k, tuple) and k[-1] == "tile" for k in a._layer_runs)
    assert a.image.em_pass == (16, "exact")             # (GALL: moe.em_exact, C = ACT_ROWS)
    assert np.array_equal(la.view(np.uint32), lb.view(np.uint32))
    t = int(np.argmax(la))
    for _ in range(2):
        ga, gb = a.step(t), b.step(t)
        assert np.array_equal(ga.view(np.uint32), gb.view(np.uint32))
        t = int(np.argmax(ga))


@pytest.mark.parametrize("S,shape,rows", [
    (1, {}, 8),
    (2, {}, 8),
    (2, {}, 16),
    (4, dict(nv=8, H=512, F=512), 8),
])
def test_deep_expert_run_is_layer_majors_bits(monkeypatch, S, shape, rows):
    """The deep expert run (moe.EM_PIPE 2, _em_deep, asserted used) at S slices: at S > 1 each
    pass's GATHER in its class's LOOP before its down MMs, after the next pass's gate and up
    MMs, and the pass after next's rows in once, after its down (at S 1 between two LOOPs of
    p + 1's classes, its first chunk's gate and up MMs and the others'). IMEM 16K, so that it
    stays this test's if the program grows past bcu1525-wg's 4K (1,577 / 1,609 / 1,673
    instructions at S 1 / 2 / 4 here; 6,075 / 6,139 at S 2 / 4 with the rows in each class's
    LOOP, which fell back to level 1 / 0).
    Logits against layer-major R = 2's and the serial expert run's (EM_PIPE 0), and the decode
    steps after, bit for bit."""
    from test_moe_s2 import _tiny
    from opentpu import target as T
    from opentpu.llm import moe as MO
    from opentpu.llm import qwen35 as Q35
    monkeypatch.setattr(Q35, "PREFILL_CHUNK", 24)
    monkeypatch.setenv("OTPU_HIGH", "0")
    monkeypatch.setenv("OTPU_POOL", "1")
    used, deep = [], MO._em_deep

    def spy(u):
        used.append(deep(u))
        return used[-1]
    monkeypatch.setattr(MO, "_em_deep", spy)
    W, spec = _tiny(("linear", "attn"), **shape)
    cfg = T.config("bcu1525-wg", S=S, DRAM_BYTES=1 << 30, IMEM_WORDS=1 << 17)
    kw = dict(cap=256, cfg=cfg, rows=1, resident=True, layer_major=2)
    toks = [int(t) for t in np.random.default_rng(13).integers(0, 1000, 43)]
    a = Engine(spec, W, expert_major=True, tile_rows=rows, **kw)
    la = a.prefill(toks)
    assert used and all(used), used                     # (the deep form, every expert run)
    assert a.image.em_pass[1] == "exact"
    b = Engine(spec, W, **kw)
    lb = b.prefill(toks)
    assert np.array_equal(la.view(np.uint32), lb.view(np.uint32))
    monkeypatch.setattr(MO, "EM_PIPE", 0)
    c = Engine(spec, W, expert_major=True, tile_rows=rows, **kw)
    assert np.array_equal(la.view(np.uint32), c.prefill(toks).view(np.uint32))
    t = int(np.argmax(la))
    for _ in range(2):
        ga, gb = a.step(t), b.step(t)
        assert np.array_equal(ga.view(np.uint32), gb.view(np.uint32))
        t = int(np.argmax(ga))


def test_router_weights_as_the_down_scale_are_close(monkeypatch):
    """EM_GSCALE (each entry's router weight in its down MM's QACT, the combine adds only) at
    S = 2: not layer-major's bits, but within fp32's rounding of them. The row scale moves the
    QACT's row maximum by the same factor, so the codes stay (but at a tie) and only the
    products' rounding order differs: max relative logit difference 1.1e-7, KL 1.9e-8 here."""
    from test_moe_s2 import _engine, _tiny
    from opentpu.llm import moe as MO
    from opentpu.llm import qwen35 as Q35
    monkeypatch.setattr(Q35, "PREFILL_CHUNK", 24)
    W, spec = _tiny(("linear", "attn"))
    kw = dict(rows=1, resident=True, layer_major=2)
    b = _engine(spec, W, 2, monkeypatch, **kw)
    monkeypatch.setattr(MO, "EM_GSCALE", True)
    a = _engine(spec, W, 2, monkeypatch, expert_major=True, tile_rows=8, **kw)
    toks = [int(t) for t in np.random.default_rng(14).integers(0, 1000, 43)]
    la, lb = a.prefill(toks), b.prefill(toks)
    assert not np.array_equal(la.view(np.uint32), lb.view(np.uint32))
    rel = np.abs(la - lb).max() / np.abs(lb).max()
    assert rel < 1e-5, rel
    assert int(np.argmax(la)) == int(np.argmax(lb))


@pytest.mark.parametrize("S,shape", [
    (1, dict(nk=4, nv=8)),                              # (og 4: a whole group, then the last
    (2, dict(nk=2, nv=4)),                              #  one pair by pair)
    (4, dict(nk=4, nv=8, H=512, F=512)),
])
def test_deltanet_by_chunks_is_the_recurrent_tiles_bits(monkeypatch, S, shape):
    """Config.DNCHUNK (docs/deltanet_chunk.md): each DeltaNet layer of a
    chunk as every tile's prep (the image's dnq), then the recurrence (DSTEPs from dnq into
    dno), then every tile's post (tiled.dn_parts) against the recurrent tiles: logits and decode
    steps bit for bit. Pairs share their key head (nv = 2 nk), as Qwen3.5-MoE's."""
    import test_moe_s2 as TM
    from opentpu import target as T
    from opentpu.llm import tiled as TL
    from opentpu.llm import qwen35 as Q35
    monkeypatch.setattr(Q35, "PREFILL_CHUNK", 24)
    monkeypatch.setenv("OTPU_HIGH", "0")
    monkeypatch.setenv("OTPU_POOL", "1")
    W, spec = TM._tiny(("linear", "attn", "linear"), **shape)
    cfg = T.config("bcu1525-wg", S=S, DRAM_BYTES=1 << 30)
    kw = dict(cap=256, rows=1, resident=True, layer_major=2, expert_major=True, tile_rows=8)
    a = Engine(spec, W, cfg=dataclasses.replace(cfg, DNCHUNK=8), **kw)
    b = Engine(spec, W, cfg=cfg, **kw)
    assert "dnq" in a.image.io and "dnq" not in b.image.io
    toks = [int(t) for t in np.random.default_rng(15).integers(0, 1000, 43)]
    la, lb = a.prefill(toks), b.prefill(toks)
    parts = {k[-1] for k in a._layer_runs if isinstance(k, tuple)}
    assert set(TL.DN_PARTS) <= parts
    assert np.array_equal(la.view(np.uint32), lb.view(np.uint32))
    t = int(np.argmax(la))
    for _ in range(2):
        ga, gb = a.step(t), b.step(t)
        assert np.array_equal(ga.view(np.uint32), gb.view(np.uint32))
        t = int(np.argmax(ga))


def _router_routes(monkeypatch, E: int, k: int):
    """-> routes(engine), forced(engine, r, fn). routes: the experts the engine's machine picked
    since the last call, [routes, k] in the order they ran, from the ISA's own router ARGMAXes
    (VOP ARGMAX over E columns, the indices it writes; k in a row per pass, in rank order), the
    same on every slice. forced: fn()'s result with r's routes written over each ARGMAX's output
    (index and value, row by row and rank by rank, all of r taken); routes then still reads what
    the machine picked before that."""
    from opentpu import isa as I
    from opentpu.isasim import Slice
    log, replay, vop = {}, {}, Slice._vop

    def spy(self, ins):
        vop(self, ins)
        w = ins.w
        if (w[5] >> 16) & 0xFF != I.V_ARGMAX or w[3] >> 16 != E:
            return
        n, drs, ars = w[3] & 0xFFFF, w[4] & 0xFFFF, w[4] >> 16
        dst, rr = self.ta(ins.ra, w[0]), np.arange(n)
        log.setdefault(id(self), []).append(self.tget(dst + rr * drs + 1).astype(int))
        st = replay.get(id(self))
        if st is not None:
            ids = st[0][st[1]:st[1] + n, st[2]]
            self.tput(dst + rr * drs + 1, ids.astype(np.float32))
            self.tput(dst + rr * drs, self.tget(self.ta(ins.rb, w[1]) + rr * ars + ids))
            st[1:] = (st[1] + n, 0) if st[2] == k - 1 else (st[1], st[2] + 1)
    monkeypatch.setattr(Slice, "_vop", spy)

    def routes(engine):
        out = []
        for s in engine.backend.machine.slices:
            c = log.pop(id(s))
            out.append(np.vstack([np.stack(c[i:i + k], 1) for i in range(0, len(c), k)]))
        assert all(np.array_equal(r, out[0]) for r in out)
        return out[0]

    def forced(engine, r, fn):
        replay.update((id(s), [r, 0, 0]) for s in engine.backend.machine.slices)
        y = fn()
        for s in engine.backend.machine.slices:
            assert replay.pop(id(s))[1:] == [len(r), 0]
        return y
    return routes, forced


def _float_with_routes(m):
    """-> logits(tokens, r): the float model m's logits at the last token with each layer's router
    taking r[layer]'s experts ([tokens, k]), its softmax's weights over them as its own gate's."""
    import torch
    cur = {}

    def gate(li, g):
        def forward(hs):
            hs = hs.reshape(-1, g.hidden_dim)
            lg = torch.nn.functional.linear(hs, g.weight)
            idx = torch.as_tensor(cur[li], dtype=torch.long)
            v = torch.softmax(lg, dim=-1).gather(1, idx)
            return lg, v / v.sum(-1, keepdim=True), idx
        return forward
    for li, layer in enumerate(m.model.layers):
        layer.mlp.gate.forward = gate(li, layer.mlp.gate)

    def logits(tokens, r):
        cur.update(enumerate(r))
        with torch.no_grad():
            return m(input_ids=torch.tensor([tokens]), use_cache=False).logits[0, -1].numpy()
    return logits


@pytest.mark.parametrize("S,shape", [
    (1, dict(nk=4, nv=8)),
    (4, dict(nk=4, nv=8, H=512, F=512)),
])
def test_deltanet_chunked_core_in_the_engine(monkeypatch, S, shape):
    """The chunked DeltaNet core (Image.compile_dn_run) in the engine's chunks on
    bcu1525-x2:wmx with Config.DNCHUNK 64 (tiled.dn_parts: a chunk's core tiles from its first
    tiled row in whole chunks of 64 rows as one run, the rest as the recurrent dn-core tiles)
    against the recurrent tiles and the float model, 3 weight seeds x 4 prompts of 192 tokens x
    (prefill's last row, the first step): 24 cases, each with every route on both device paths
    and the float model forced to the tiles' own (_router_routes), so that no case is exempt.

    This tests the wiring, and L0's accuracy on tiny weights. test_dn_chunk stays the bound on the
    core's accuracy, and the real weights' evidence is dn35, in numpy
    (2026-11-03-pf500-dn-chunk.md). The core is not bit-exact. It is judged by its distance to the float model and the tiles, not by bit
    equality, and that rests on the user's acceptance of the chunked DeltaNet's bit differences
    (docs/deltanet_chunk.md).

    Drawn heads have next to no memory (A_log, the log of uniform(0, 16)), so a state error would
    not reach the rows: every DN head here has a set memory, rates a row geomspace(0.005, 8), decays
    0.995 .. 3e-4. At 192 tokens prefill's last row is a core row and the first step reads the state
    the core left.

    The check: each prefill's L0 X rows, where no route or earlier layer enters, no further from
    float than L0_MAX x the tiles', core rows and tile rows apart; any one over fails. A step's X
    is not in a record, so steps have no such check. The logits' checks are for gross errors,
    since with memory in the heads a case's KL is noisy: KL(core, tiles) under 2 x the larger
    distance to float, KL(core, float) under 3 x the tiles', the same top token where the top two
    logits are 0.2 or more apart; over the 24, the mean of KL(core, float) at most 1.15 x the
    tiles', a drift that a limit per case can't catch. A failure prints L0's ratios and the
    weights' hash.

    What it misses:
    - A fault in L2 alone. L0's rows can't see it and L2's own rows carry the earlier layers'
      rounding: at S 1, on L2 alone, 1 LSB and 3 LSB on o, 3 % noise on o and state x 1.03 all
      pass, and a state never written back fires the mean only. test_dn_chunk covers the
      arithmetic, which is the same for every layer. Nothing covers L2's own wiring below that size.
    - An error only in the route taken. Every route is forced, so what the core would have picked
      is printed, not asserted (a router's near tie, one path takes the other expert, and which
      path does moves with the weights' last bits; two hosts' draws differ). Nothing
      covers it: no test holds the core's own routes to the tiles' or float's.

    Chunks of 128 and 64: the first's core from row 3 (conv_k - 1, where the core takes "row": its
    tiles cut at row 67, tiled.core_end), the second's from 0."""
    import test_moe_s2 as TM
    import torch
    from opentpu import target as T
    from opentpu.llm import qwen35 as Q35
    from opentpu.llm import tiled as TL
    if not hasattr(Q35.Image, "compile_dn_run"):
        pytest.skip("the chunked core (bf16st's dn-core) is not in this tree")
    monkeypatch.setattr(Q35, "PREFILL_CHUNK", 128)
    cores, parts = [], TL.dn_parts

    def spy(runs, core=None):
        out = parts(runs, core)
        cores.extend((v["row"], v["rows"]) for k, v, _ in out if k[0] == "dnc")
        return out
    monkeypatch.setattr(TL, "dn_parts", spy)
    monkeypatch.setenv("OTPU_HIGH", "0")
    monkeypatch.setenv("OTPU_POOL", "1")
    L, E, k = 3, TM.E, TM.K
    routes, forced = _router_routes(monkeypatch, E, k)
    cfg = T.config("bcu1525-x2:wmx", S=S, DRAM_BYTES=1 << 30, BHW=0)
    kw = dict(cap=256, rows=1, resident=True, layer_major=2, expert_major=True, tile_rows=8)
    # Clean, at both S: this set's seeds 1.007, another host's draw 1.007, seeds 3-5 and 6-8
    # 1.010: docs/deltanet_chunk.md section 5.
    L0_MAX = 1.015
    kls, bad, flips, differ, l0, drawn = [], [], [], [], [], {}
    x0, tap, layer_run = {}, [], Engine._layer_run

    def spy_x(self, key):
        if key[0] == "x" and key[1] == 0:               # (L0's X records, a chunk's)
            d, dev = x0.setdefault(id(self), []), self.image.descriptors(0).moe_dev
            d.append(np.stack([self.backend.read(0, dev.em_base + r * dev.em_rec,
                                                 4 * self.spec.hidden).view(np.float32)
                               for r in range(64 if d else 128)]).astype(np.float64))
        return layer_run(self, key)
    monkeypatch.setattr(Engine, "_layer_run", spy_x)

    def rows(name, xa, xb, t):
        """L0's X rows, the core's and the tiles', against float's t: rms relative distance."""
        core = np.zeros(len(t), bool)
        core[3:67] = core[128:192] = True
        for g, sel in (("core", core), ("tile", ~core)):
            ea, eb = (np.sqrt(np.mean((np.linalg.norm(x[sel] - t[sel], axis=1)
                                       / np.linalg.norm(t[sel], axis=1)) ** 2)) for x in (xa, xb))
            l0.append((g, ea / eb, float(np.linalg.norm(xa[sel] - xb[sel], axis=1).max()
                                         / np.linalg.norm(xb[sel], axis=1).mean())))
            if ea >= L0_MAX * eb:
                bad.append(f"{name}: L0's {g} rows to float, core {ea:.3e} >= {L0_MAX} x tiles "
                           f"{eb:.3e}")

    def kl(x, y):
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        px, py = np.exp(x - x.max()), np.exp(y - y.max())
        px, py = px / px.sum(), py / py.sum()
        return float(np.sum(py * (np.log(py) - np.log(px))))

    def case(name, x, y, f, rx, ry):
        """x the core's logits, y the tiles', f the float model's; rx what the core would have
        picked on the row compared, ry the tiles' routes there."""
        ct, cf, tf = kl(x, y), kl(x, f), kl(y, f)
        kls.append((ct, cf, tf))
        if ct >= 2 * max(cf, tf):
            bad.append(f"{name}: core to tiles {ct:.3e}, to float core {cf:.3e} tiles {tf:.3e}")
        if cf >= 3 * tf:
            bad.append(f"{name}: to float core {cf:.3e} >= 3 x tiles {tf:.3e}")
        top = np.sort(y)
        if top[-1] - top[-2] >= 0.2 and int(np.argmax(x)) != int(np.argmax(y)):
            bad.append(f"{name}: top tokens {int(np.argmax(x))} {int(np.argmax(y))}")
        if not np.array_equal(rx, ry):
            lay = np.nonzero((rx != ry).any(axis=1))[0].tolist()
            flips.append(f"{name}: L{lay}")

    for seed in range(3):
        m, W, spec = TM._tiny_model(("linear", "attn", "linear"), seed=seed, **shape)
        W = dict(W)
        with torch.no_grad():                   # (every DN head a set memory)
            for li, layer in enumerate(m.model.layers):
                if hasattr(layer, "linear_attn"):
                    d = layer.linear_attn
                    rate = torch.tensor(np.geomspace(0.005, 8, d.A_log.numel()), dtype=torch.float32)
                    d.A_log.copy_(torch.log(rate) - torch.log(torch.nn.functional.softplus(d.dt_bias)))
                    W[f"model.layers.{li}.linear_attn.A_log"] = d.A_log.numpy().copy()
        h = hashlib.sha256()
        for q in sorted(W):
            h.update(q.encode() + W[q].tobytes())
        drawn[seed] = h.hexdigest()[:12]
        hook = m.model.layers[0].post_attention_layernorm.register_forward_pre_hook(
            lambda mod, args: tap.append(args[0].reshape(-1, spec.hidden).detach().numpy()))
        flt = _float_with_routes(m)
        a = Engine(spec, W, cfg=dataclasses.replace(cfg, DNCHUNK=64), **kw)
        b = Engine(spec, W, cfg=cfg, **kw)
        assert a.image.dn_core_ok()
        dev = a.image.descriptors(0).moe_dev
        for p in range(4):
            toks = [int(t) for t in np.random.default_rng(16 + p).integers(0, 1000, 192)]
            a.reset(), b.reset()
            x0.clear(), tap.clear()
            lb = b.prefill(toks)                        # (chunks of 128 and 64 rows)
            rb = routes(b)
            la = forced(a, rb, lambda: a.prefill(toks))
            ra = routes(a)
            assert ra.shape == rb.shape == (L * 192, k)
            tails = [a.backend.read(0, dev.em_base + r * dev.em_rec + 4 * spec.hidden * (k + 2),
                                    4 * k).view(np.float32) for r in range(64)]
            assert np.array_equal(np.stack(tails).astype(int) % E, rb[-64:])    # (its records)
            differ.append(int((ra != rb).any(axis=1).sum()))
            n1 = L * 128                                # (a chunk's layers in turn)
            by = [np.vstack([rb[li * 128:(li + 1) * 128], rb[n1 + li * 64:n1 + (li + 1) * 64]])
                  for li in range(L)]
            last = [-64 * (L - 1 - li) - 1 for li in range(L)]
            case(f"seed {seed} prompt {16 + p} prefill", la, lb, flt(toks, by), ra[last], rb[last])
            rows(f"seed {seed} prompt {16 + p}", np.vstack(x0[id(a)]), np.vstack(x0[id(b)]),
                 tap[-1].astype(np.float64))
            t = int(np.argmax(lb))
            gb = b.step(t)
            sb = routes(b)
            ga = forced(a, sb, lambda: a.step(t))
            sa = routes(a)
            assert sa.shape == sb.shape == (L, k)
            gf = flt(toks + [t], [np.vstack([by[li], sb[li:li + 1]]) for li in range(L)])
            case(f"seed {seed} prompt {16 + p} step", ga, gb, gf, sa, sb)
        assert any(c[0] == "dnc" for c in a._layer_runs if isinstance(c, tuple))
        if TL.dn_core(a.image)[1]:              # (the core takes a start row: the first chunk's)
            assert (3, 64) in cores and (0, 64) in cores, cores
        else:
            assert set(cores) == {(0, 64)}, cores
        hook.remove()
        del a, b
    ct, cf, tf = np.array(kls).T
    print(f"S {S}: {len(kls)} cases, weights {drawn}, core to tiles {ct.min():.3e} .. "
          f"{ct.max():.3e}, to float mean core {cf.mean():.3e} tiles {tf.mean():.3e} "
          f"({cf.mean() / tf.mean():.3f}), worst case core / tiles {(cf / tf).max():.2f}, worst core "
          f"{cf.max():.3e} tiles {tf.max():.3e}; a prefill's routes the core would have picked "
          f"otherwise (of {L * 192}): {min(differ)} .. {max(differ)}, on the rows compared: {flips}")
    ratios = []
    for g in ("core", "tile"):
        v = np.array([q[1:] for q in l0 if q[0] == g]).T
        ratios.append(f"L0's {g} rows to float, core / tiles: mean {v[0].mean():.3f} least "
                      f"{v[0].min():.3f} most {v[0].max():.3f}; largest row core - tiles {v[1].max():.3e}")
        print(ratios[-1])
    if cf.mean() > 1.15 * tf.mean():
        bad.append(f"to float, mean: core {cf.mean():.3e} > 1.15 x tiles {tf.mean():.3e}")
    assert not bad, "\n".join(bad + ratios + [f"S {S}, weights' hash by seed {drawn}"])
