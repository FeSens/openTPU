"""GPT-OSS (opentpu/llm/gptoss.py): a tiny random GptOssConfig model against Hugging Face
transformers (the numpy reference), the quantized emulation against the ISA simulator's decode
(resident, every expert in its slot), and the experts' MXFP4 import. The tiny model's hidden
size 320 is not a whole number of D blocks (the image pads it to 384, as 2880 to 2944), its
sliding window 16 positions."""
import dataclasses

import numpy as np
import pytest

from opentpu.llm import gptoss as GO
from opentpu.llm.gptoss import Spec, emulated_logits, reference_logits

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

L, E, K, F, H, V = 4, 8, 4, 320, 320, 1000


def _cos(a, b):
    return (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)


def _tiny(window=16, layers=L, experts=E, k=K):
    torch.manual_seed(0)
    hc = transformers.GptOssConfig(
        hidden_size=H, intermediate_size=F, num_hidden_layers=layers, num_attention_heads=8,
        num_key_value_heads=2, head_dim=64, num_local_experts=experts, num_experts_per_tok=k,
        vocab_size=V, sliding_window=window, rms_norm_eps=1e-5, tie_word_embeddings=False,
        max_position_embeddings=131072, attn_implementation="eager")
    m = transformers.GptOssForCausalLM(hc).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            elif "sinks" in n:
                p.copy_(torch.randn_like(p))
            elif n.endswith("bias"):
                p.copy_(0.1 * torch.randn_like(p))
            elif "experts" in n:
                p.copy_(0.08 * torch.randn_like(p))
            else:
                p.copy_(0.05 * torch.randn_like(p))
    W = {k: v.float().numpy() for k, v in m.state_dict().items()}
    return m, W, hc


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    m, W, hc = _tiny()
    d = tmp_path_factory.mktemp("gptoss")
    hc.save_pretrained(d)
    return m, W, Spec.from_hf(d)


def test_spec_from_hf(tiny):
    """The config's layer kinds (sliding first), MoE, YaRN and window."""
    _, _, s = tiny
    assert s.kinds == ("slide", "full") * (L // 2) and s.window == 16
    assert (s.moe.E, s.moe.k, s.moe.ffn, s.moe.rule, s.moe.act, s.moe.bias) == \
        (E, K, F, "softmax", "oai", True)
    assert s.theta == 150000.0 and len(s.rope_div) == 32
    assert s.rope_scale == pytest.approx(0.1 * np.log(32) + 1)


def test_yarn_matches_hf(tiny):
    """rope_tables' YaRN (llm.yarn) is Hugging Face's rotary embedding to fp32 rounding."""
    m, _, s = tiny
    from opentpu.llm.qwen3 import rope_tables
    pos = torch.arange(0, 4096, 37)[None]
    x = torch.zeros(1, 1, 64)
    with torch.no_grad():
        cos, sin = m.model.rotary_emb(x, pos)
    for j, p in enumerate(pos[0].tolist()):
        c, sn = rope_tables(s, p)
        assert np.abs(c - cos[0, j, :32].numpy()).max() < 2e-3
        assert np.abs(sn - sin[0, j, :32].numpy()).max() < 2e-3


def test_reference_matches_hf(tiny):
    """The fp32 reference against HF's fp32 forward, past the sliding window."""
    m, W, s = tiny
    toks = [int(t) for t in np.random.default_rng(0).integers(0, V, 40)]
    with torch.no_grad():
        hf = m(torch.tensor([toks])).logits[0].numpy()
    ref = reference_logits(s, W, toks)
    assert np.abs(ref - hf).max() < 2e-4 * np.abs(hf).max()


def _to_mxfp4(w):
    """w [N, K] as a checkpoint's MXFP4 (blocks [N, K/32, 16]: two E2M1 codes a byte, the first
    in the low nibble; E8M0 scales [N, K/32]): each 32-block's scale the power of two that puts
    its largest magnitude at most 6, its values rounded to the nearest E2M1 value."""
    N, K = w.shape
    b = w.reshape(N, K // 32, 32).astype(np.float64)
    amax = np.abs(b).max(-1)
    e = np.where(amax > 0, np.ceil(np.log2(np.maximum(amax, 1e-30) / 6)), 0).astype(np.int64)
    v = b / np.exp2(e)[..., None]
    mags = np.abs(GO.E2M1[:8]).astype(np.float64)
    code = np.abs(np.abs(v)[..., None] - mags).argmin(-1) | np.where(v < 0, 8, 0)
    code = code.reshape(N, K).astype(np.uint8)
    blocks = (code[:, 0::2] | (code[:, 1::2] << 4)).reshape(N, K // 32, 16)
    return blocks, (e + 127).astype(np.uint8)


def _mxfp4_model(m, W):
    """The tiny model's experts as a checkpoint's MXFP4 tensors (W's dense ones dropped), and HF's
    model given their values."""
    W = dict(W)
    for i in range(len(m.model.layers)):
        p = f"model.layers.{i}.mlp.experts."
        ex = m.model.layers[i].mlp.experts
        for n, rows in (("gate_up_proj", lambda e: W[p + "gate_up_proj"][e].T),
                        ("down_proj", lambda e: W[p + "down_proj"][e].T)):
            bl, sc = zip(*(_to_mxfp4(rows(e)) for e in range(E)))
            W[p + n + "_blocks"], W[p + n + "_scales"] = np.stack(bl), np.stack(sc)
            vals = np.stack([GO.mxfp4_values(GO.mxfp4_codes(b), s) for b, s in zip(bl, sc)])
            with torch.no_grad():
                getattr(ex, n).copy_(torch.from_numpy(vals.transpose(0, 2, 1).copy()))
            del W[p + n]
    return W


@pytest.fixture(scope="module")
def tiny_mx(tiny):
    m, W, s = _tiny()[0], dict(tiny[1]), tiny[2]
    return m, _mxfp4_model(m, W), s


def test_mxfp4_checkpoint_matches_hf(tiny_mx):
    """Experts as a checkpoint's MXFP4 blocks (the nibbles' order, gate_up's interleaved rows):
    the reference against HF given their values; every expert matrix imports exactly into an
    fp4 slot and an nvfp4 one, and an int8 slot quantizes their values."""
    from opentpu.llm import moe as MO
    m, W, s = tiny_mx
    toks = [int(t) for t in np.random.default_rng(4).integers(0, V, 24)]
    with torch.no_grad():
        hf = m(torch.tensor([toks])).logits[0].numpy()
    assert np.abs(reference_logits(s, W, toks) - hf).max() < 2e-4 * np.abs(hf).max()
    for wf, how in (("fp4", "import"), ("nvfp4", "import"), ("int8", "quant")):
        fmt = MO.ExpertFormat(384, F, 128, wf, bias=True)
        assert GO.expert_mats(W, 1, 3, fmt).how == [how] * 3


def _engine(spec, W, wformat="int8", cap=256, experts=None, resident=True, **kw):
    from opentpu.llm.qwen3 import Engine, device_config
    cfg = device_config(spec, cap, S=1, rows=1, wformat=wformat, lookup=True, experts=experts,
                        MCOLS=8, **kw)
    return Engine(spec, W, cap=cap, cfg=cfg, rows=1, wformat=wformat, resident=resident,
                  experts=experts)


def _follows(eng, dev, emu, routes, n_layers):
    """The device's logits [T, V] against the emulation's in a model of several layers, and its
    routing against the emulation's. A QACT value within fp32's reach of a rounding tie (the
    emulation's float64 cannot place it; ~1% of a layer's positions) moves one int8 step: that
    position's later layers and every later position's keys follow (1 - cos ~1e-4 from there
    on, ~1e-1 where it flips an expert), so: the median within 1e-3, nine positions in ten
    within 1e-2, the greedy token at 85% of them; the card's choices in the first MoE layer
    (whose input no routing has touched) the emulation's wherever its k-th and (k+1)-th logits
    are 1e-2 apart (test_ring_wraps holds one layer, whose positions do not mix, to 1e-6)."""
    dc = 1 - _cos(dev, emu)
    assert np.median(dc) < 1e-3 and (dc < 1e-2).mean() >= 0.9, dc
    assert (dev.argmax(1) == emu.argmax(1)).mean() >= 0.85
    card, checked = eng.server.history, 0
    for t, layer, ids, margin in routes:
        if layer == 0 and margin > 1e-2:
            assert card[t * n_layers] == ids, (t, card[t * n_layers], ids, margin)
            checked += 1
    assert checked >= len(dev) // 2


@pytest.mark.parametrize("wformat,formats", [("int8", ""), ("fp4", ""),
                                             ("fp4", "wq=int8,wk=int8")])
def test_device_follows_emulation(tiny, monkeypatch, wformat, formats):
    """Resident decode on the ISA simulator against the float64 emulation of its quantization
    points, past the sliding window (_follows); int8 follows the fp32 reference but where a
    token's routing turns on the quantized router (median cosine). formats: Spec.formats (q and
    k int8 over fp4), the image's and the emulation's."""
    _, W, s = tiny
    monkeypatch.delenv("OTPU_FORMATS", raising=False)
    s = dataclasses.replace(s, formats=formats)
    toks = [int(t) for t in np.random.default_rng(1).integers(0, V, 24)]
    eng = _engine(s, W, wformat)
    eng.server.history = []
    dev = np.array([eng.step(t) for t in toks])
    routes = []
    emu = emulated_logits(s, W, toks, wformat=wformat, routes=routes)
    _follows(eng, dev, emu, routes, L)
    if wformat == "int8":
        assert np.median(_cos(dev, reference_logits(s, W, toks))) > 0.99


def test_attention_formats(tiny, monkeypatch):
    """The attention projections' formats (KINDS: "attn" each one's default, "wq" / "wk" / "wv" /
    "wo" its own; the caller's formats, else OTPU_FORMATS, else Spec.formats): each projection's
    bytes in the layer block and its descriptor in its format, the experts in wformat, the
    formats in the compile worker's choices; another model's kind, or a range that splits the
    layers (one block layout), refused."""
    from opentpu import isa as I
    from opentpu import quant as Q
    from opentpu.llm.qwen3 import _align, device_config
    _, _, s = tiny
    assert s.formats == "attn=int8"                     # (the default: every projection int8)
    s = dataclasses.replace(s, formats="")
    monkeypatch.delenv("OTPU_FORMATS", raising=False)
    cfg = device_config(s, 256, S=1, rows=1, wformat="fp4", lookup=True, MCOLS=8)
    names = ("wq", "wk", "wv", "wo")

    def image(formats, spec=s):
        return spec.image(cfg, 256, wformat="fp4", formats=formats, lookup=True)
    assert image(None).mf == image("").mf == dict.fromkeys(names, "fp4")
    assert image("attn=int8,wo=fp4").mf == {"wq": "int8", "wk": "int8", "wv": "int8",
                                            "wo": "fp4"}
    qk = dict(zip(names, ("int8", "int8", "fp4", "fp4")))
    assert image(None, dataclasses.replace(s, formats="wq=int8,wk=int8")).mf == qk
    monkeypatch.setenv("OTPU_FORMATS", "wq=int8,wk=int8")
    assert image(None).mf == qk and image("").mf == dict.fromkeys(names, "fp4")
    monkeypatch.delenv("OTPU_FORMATS")
    assert image(f"wv@0-{L - 1}=int8").mf["wv"] == "int8"
    for bad in ("mlp=int8", "wv@1=int8", "wq=int9"):
        with pytest.raises(ValueError):
            image(bad)
    a, b = image(""), image("wq=int8,wo=int8")
    assert b.choices["formats"] == "wq=int8,wo=int8"
    assert b.fmt.wformat == a.fmt.wformat == "fp4"
    la, lb = a.descriptors(0).layer(0), b.descriptors(0).layer(0)
    for name in names:
        assert getattr(la, name).wf == I.WFORMATS["fp4"]
        assert getattr(lb, name).wf == I.WFORMATS[b.mf[name]]
    # the layer block's matrices in their formats' bytes, each part aligned (the router after
    # them): q and o grow
    A = _align(cfg)
    mb = lambda n, k, f: sum(-(-x // A) * A for x in Q.grp_bytes(n, k, f, cfg.D, cfg.WG))  # noqa
    grow = [mb(n, k, b.mf[nm]) - mb(n, k, "fp4") for nm, (n, k) in a.mats.items()]
    assert grow[0] > 0 and grow[3] > 0 and grow[1] == grow[2] == 0
    assert (b.lofs[GO.SLIDE]["router"][0] - a.lofs[GO.SLIDE]["router"][0]) == sum(grow)


def test_mxfp4_experts_on_device(tiny_mx):
    """The checkpoint's MXFP4 experts imported into fp4 slots (no requantization): the device
    follows the emulation of those slots (_follows)."""
    _, W, s = tiny_mx
    toks = [int(t) for t in np.random.default_rng(5).integers(0, V, 12)]
    eng = _engine(s, W, "fp4")
    assert eng.image.imports == {"import": 3 * E * L}
    eng.server.history = []
    dev = np.array([eng.step(t) for t in toks])
    routes = []
    emu = emulated_logits(s, W, toks, wformat="fp4", routes=routes)
    _follows(eng, dev, emu, routes, L)


@pytest.mark.parametrize("wformat,experts", [("fp4", None), ("fp4", K), ("int8", K)])
def test_wg8_gall_is_wg1(tiny, monkeypatch, wformat, experts):
    """At WG 8 under GALL with the channel hash (the BCU1525 target's): the experts in groups in
    their slots, resident or streamed into k slots a layer; WG 1's logits bit for bit, and no MM
    of rows of one."""
    from opentpu import isasim
    _, W, s = tiny
    toks = [int(t) for t in np.random.default_rng(2).integers(0, V, 6)]
    rds, mm = [], isasim.Slice._mm
    monkeypatch.setattr(isasim.Slice, "_mm", lambda sl, ins: (rds.append(ins.rd), mm(sl, ins))[1])
    out = []
    for wg in (1, 8):
        rds.clear()
        eng = _engine(s, W, wformat, experts=experts, WG=wg, CHASH=wg > 1, GALL=wg > 1)
        out.append(np.array([eng.step(t) for t in toks]).view(np.uint32))
        assert (0 in rds) == (wg == 1), set(rds)
        if experts:
            assert eng.server.misses > len(toks)
    assert np.array_equal(*out)


@pytest.mark.parametrize("wformat", ["fp4", "int8"])
def test_pool_is_path_a(tiny, monkeypatch, wformat):
    """Every expert in a resident pool below 4 GiB (Image.pool, OTPU_POOL=1: moe_ffn's resident
    path, no directory and no host server) at WG 8 under GALL with the channel hash: path (a)'s
    logits with every expert in its slot bit for bit, and the generate loop's tokens."""
    _, W, s = tiny
    toks = [int(t) for t in np.random.default_rng(8).integers(0, V, 6)]
    kw = dict(WG=8, CHASH=True, GALL=True)
    a = _engine(s, W, wformat, **kw)
    monkeypatch.setenv("OTPU_POOL", "1")
    b = _engine(s, W, wformat, **kw)
    assert a.image.pool is None and a.server is not None
    assert b.image.offload is None and b.server is None and b.image.pool.experts == L * E
    assert b.image.choices == {"pool": True}
    for t in toks:
        x, y = a.step(t), b.step(t)
        assert np.array_equal(x.view(np.uint32), y.view(np.uint32))
    t0 = int(np.argmax(x))
    assert b.generate_card(t0, 4, stop_ids=[]) == a.generate_card(t0, 4, stop_ids=[])


@pytest.mark.parametrize("wformat,pool", [("fp4", False), ("fp4", True), ("int8", True)])
def test_kt_is_padded(tiny, monkeypatch, wformat, pool):
    """MM KT's pairs (Config.KT, docs/isa.md "K tail") at WG 8 under GALL with the channel hash:
    every matrix whose own K is the model's hidden size (320 of 384: q / k / v, the router, the
    head, the experts' W_gate and W_up) and W_down's last part (64 of 128) streams its pairs,
    a chunk a pair less; W_o (K 512) does not. The logits of the padded layout's bit for bit,
    on path (a)'s slots and on the resident pool, and the generate loop's tokens."""
    from opentpu import isa as I
    _, W, s = tiny
    toks = [int(t) for t in np.random.default_rng(8).integers(0, V, 6)]
    if pool:
        monkeypatch.setenv("OTPU_POOL", "1")
    kw = dict(WG=8, CHASH=True, GALL=True)
    a, b = _engine(s, W, wformat, **kw), _engine(s, W, wformat, KT=True, **kw)
    assert not any(a.image.kt.values()) and not any(a.image.fmt.kts)
    assert b.image.kt == {"wq": True, "wk": True, "wv": True, "wo": False, "router": True,
                          "head": True} and b.image.fmt.kts == [True, True, False, True]
    assert b.image.nbytes < a.image.nbytes and b.image.fmt.nbytes < a.image.fmt.nbytes
    # the same program but for the KT MMs' pair strides (two groups' chunks but one) and the
    # addresses: every MM over K 384 (q / k / v, the router, W_gate / W_up, the head) is one
    (pa,), (pb,) = a.image.compile_step(5), b.image.compile_step(5)
    assert [i.op for i in pa] == [i.op for i in pb]
    mm = [(i, j) for i, j in zip(pa, pb) if i.op == I.MM]
    kt = [(i, j) for i, j in mm if j.rd and j.w[4] & 1]
    assert all(j.w[4] - 1 == 2 * i.w[4] - 8 * 128 for i, j in kt)
    assert all(j.w[4] == i.w[4] for i, j in mm if not (j.rd and j.w[4] & 1))
    assert all(j.w[4] & 1 for i, j in mm if i.rd and i.w[4] == 3 * 8 * 128) and len(kt) > 2 * L
    for t in toks:
        x, y = a.step(t), b.step(t)
        assert np.array_equal(x.view(np.uint32), y.view(np.uint32))
    t0 = int(np.argmax(x))
    assert b.generate_card(t0, 4, stop_ids=[]) == a.generate_card(t0, 4, stop_ids=[])


def test_high_is_low(tiny, monkeypatch):
    """At AW 36 (bcu1525-wg at 8 GiB): the layers' weights, the head and the expert pool from 4
    GiB on (Image.high, OTPU_HIGH=1), the KV rings and caches, the tables and the programs
    below; the image below 4 GiB with its pool there gives its logits bit for bit (the
    resident decode) and the card's generate loop its tokens."""
    from opentpu import target as T
    from opentpu.llm.qwen3 import HIGH, Engine
    _, W, s = tiny
    toks = [int(t) for t in np.random.default_rng(9).integers(0, V, 4)]
    cfg = T.config("bcu1525-wg", DRAM_BYTES=1 << 33)
    assert cfg.AW == 36
    monkeypatch.setenv("OTPU_HIGH", "0")
    monkeypatch.setenv("OTPU_POOL", "1")
    lo = Engine(s, W, cap=256, cfg=cfg, rows=1, wformat="fp4", resident=True)
    monkeypatch.setenv("OTPU_HIGH", "1")
    monkeypatch.delenv("OTPU_POOL")
    hi = Engine(s, W, cap=256, cfg=cfg, rows=1, wformat="fp4", resident=True)
    a, b = lo.image, hi.image
    assert not a.high and a.pool.base + a.pool.experts * a.pool.ebytes <= a.nbytes < HIGH
    assert b.high and b.offload is None and b.low_end < HIGH <= b.layer0 < b.head[0] < b.pool.base
    assert b.choices == {"high": True} and b.lloc[0][0] < HIGH
    for t in toks:
        x, y = lo.step(t), hi.step(t)
        assert np.array_equal(x.view(np.uint32), y.view(np.uint32))
    t0 = int(np.argmax(x))
    assert hi.generate_card(t0, 4, stop_ids=[]) == lo.generate_card(t0, 4, stop_ids=[])


def test_the_bank_half_window_is_low(tiny, monkeypatch):
    """bcu1525-wg:bhw (Config.BHW 33: the slice's first 8 GiB the bank-half window): the image is
    high whatever OTPU_HIGH says, its low area below 4 GiB (the window's lower half), every
    weight and the expert pool from 8 GiB on, nothing in the window's upper half (no DeltaNet
    mixers); the ISA simulator's guard (every write below 4 GiB) holds and the logits are the
    low image's bit for bit (the resident decode), the generate loop's tokens its."""
    from opentpu import target as T
    from opentpu.llm.qwen3 import HIGH, Engine
    _, W, s = tiny
    toks = [int(t) for t in np.random.default_rng(9).integers(0, V, 4)]
    monkeypatch.setenv("OTPU_HIGH", "0")
    monkeypatch.setenv("OTPU_POOL", "1")
    lo = Engine(s, W, cap=256, cfg=T.config("bcu1525-wg", DRAM_BYTES=1 << 33), rows=1,
                wformat="fp4", resident=True)
    cfg = T.config("bcu1525-wg:bhw", DRAM_BYTES=1 << 34)
    assert cfg.BHW == 33
    hi = Engine(s, W, cap=256, cfg=cfg, rows=1, wformat="fp4", resident=True)
    b = hi.image
    assert b.high and b.low_end < HIGH and b.layer0 == 2 * HIGH < b.head[0] < b.pool.base
    assert all(sl.wlim == HIGH for sl in hi.backend.machine.slices)
    for t in toks:
        x, y = lo.step(t), hi.step(t)
        assert np.array_equal(x.view(np.uint32), y.view(np.uint32))
    t0 = int(np.argmax(x))
    assert hi.generate_card(t0, 4, stop_ids=[]) == lo.generate_card(t0, 4, stop_ids=[])


def test_per_position_is_resident(tiny):
    """The per-position programs (compile_step, the token's inputs from the image's tables) give
    the resident decode's logits bit for bit."""
    _, W, s = tiny
    toks = [int(t) for t in np.random.default_rng(6).integers(0, V, 6)]
    a, b = _engine(s, W, "fp4"), _engine(s, W, "fp4", resident=False)
    assert a.resident and not b.resident
    for t in toks:
        assert np.array_equal(a.step(t).view(np.uint32), b.step(t).view(np.uint32))


def test_masked_block_reads_its_sub_blocks():
    """The resident decode's masked block (a full layer's, and a sliding ring's in its first
    lap) read in variants of 128 and 256 tokens, chosen at run time by tpos (attention.Bucket's
    tpos): its logits are the per-position programs' (each a partial block of p + 1 tokens) bit
    for bit on both sides of tpos 128, at GPT-OSS's 128-token window; bucket 1's program holds
    both layers' variant loops, bucket 2's the full layer's alone (its ring is past the first
    lap). Heads whose variant work comes back to back share its loops (one table): bucket 1's
    layers take 4 RLDs each, bucket 2's full layer 8."""
    m, W, hc = _tiny(window=128, layers=2)
    import tempfile
    d = tempfile.mkdtemp()
    hc.save_pretrained(d)
    s = Spec.from_hf(d)
    toks = [int(t) for t in np.random.default_rng(9).integers(0, V, 136)]
    a, b = _engine(s, W, "fp4", cap=512), _engine(s, W, "fp4", cap=512, resident=False)
    for e in (a, b):
        e.prefill(toks[:120])
    for t in toks[120:]:
        assert np.array_equal(a.step(t).view(np.uint32), b.step(t).view(np.uint32)), a.pos

    def variants(prog):
        return sum(1 for x in prog if str(x.comment).startswith("tokens "))
    one = variants(a._decodes[1][0][0])
    two = variants(a.image.compile_decode(2, 256)[0][0])
    assert one == two == 8
    # the variants' table (kept for the program) at the top of TMEM, apart from next-fit's turn
    from opentpu import isa as I
    cfg = a.image.cfg
    top = cfg.TMEM_WORDS - (I.STREAM_DESC_AREA if cfg.STREAM and cfg.TMEM_WORDS <= 1 << 16 else 0)
    assert {x.w[0] for x in a._decodes[1][0][0] if str(x.comment).startswith("tokens ")} == \
        {top - 2, top - 1}


def test_sub_block_variants_waits_hold(monkeypatch):
    """The masked block's variant loops (test_masked_block_reads_its_sub_blocks's) and their wait
    entries a pass a variant (waits.variant_paths: one table, 2 paths): the counts hold on the ISA
    simulator's run on both sides of tpos 128 (OTPU_WAITS_CHECK), with fewer VPU mask bits than
    with the variants' counts apart (rld {}: one pass on their bounds)."""
    from opentpu import waits as WT
    m, W, hc = _tiny(window=128, layers=2)
    import tempfile
    d = tempfile.mkdtemp()
    hc.save_pretrained(d)
    s = Spec.from_hf(d)
    toks = [int(t) for t in np.random.default_rng(9).integers(0, V, 132)]
    a = _engine(s, W, "fp4", cap=512)
    a.prefill(toks[:124])
    monkeypatch.setenv("OTPU_WAITS_CHECK", "1")
    for t in toks[124:]:
        a.step(t)
    assert a.pos == 132
    prog = a._decodes[1][0][0]
    assert len(WT.variant_paths(prog)) == 2

    def bits(w):
        return sum(bin(WT.vmask(int(x))).count("1") for x in w)
    assert bits(WT.wait_counts(prog)) < bits(WT.wait_counts(prog, rld={}))



def test_attention_tail_orders_are_bit_exact(monkeypatch):
    """_attention's tail orders (ATTN_LEAD 2: the next two heads' first scores in flight; ATTN_LAZY:
    a head's queries quantized when its scores need them; ATTN_LAG 1: a head's output written
    after the next head's finish, in the variants' loops) move instructions, not math: the
    resident decode's logits across tpos 128 (the masked block's sub-blocks) and its generate
    loop's tokens are those of the plain orders (lead 1, eager, emit at once) bit for bit."""
    m, W, hc = _tiny(window=128, layers=2)
    import tempfile
    d = tempfile.mkdtemp()
    hc.save_pretrained(d)
    s = Spec.from_hf(d)
    toks = [int(t) for t in np.random.default_rng(10).integers(0, V, 134)]

    def run():                          # (the programs compile as the run needs them)
        e = _engine(s, W, "fp4", cap=512)
        e.prefill(toks[:120])
        lg = np.stack([e.step(t) for t in toks[120:]])
        return lg, e.generate_card(int(np.argmax(lg[-1])), 4, stop_ids=[]), e._decodes[1][0][0]
    a = run()
    monkeypatch.setattr(GO, "ATTN_LEAD", 1)
    monkeypatch.setattr(GO, "ATTN_LAZY", False)
    monkeypatch.setattr(GO, "ATTN_LAG", 0)
    b = run()
    assert np.array_equal(a[0].view(np.uint32), b[0].view(np.uint32)) and a[1] == b[1]
    assert [i.op for i in a[2]] != [i.op for i in b[2]]       # (the orders moved)

@pytest.mark.parametrize("pool", [False, True], ids=["slots", "pool"])
def test_moe_on_board_model(tiny, have_verilator, monkeypatch, pool):
    """The BCU1525 at WG 8 on its board model (test_bcuwg_board's tb_bcuwg: MCOLS 8, WG 8, the
    DIMMs stalling), fp4, every expert resident: in path (a)'s slots with `served` past any
    request, so each layer's fence and directory waits hold at their first read (no host serves
    during a run), or in the resident pool (OTPU_POOL=1, moe_ffn's resident path). The resident
    decode, then the card's generate loop: the ISA simulator's logits, instruction counts and
    tokens bit for bit (the schedule's orders on the RTL: _attention's lazy Q MMs, the pool's
    next expert's place ahead of this one's work)."""
    from test_bcuwg_board import TARGET, _transport
    from opentpu import target as T
    from opentpu.host.board import BoardBackend
    from opentpu.llm.qwen3 import Engine
    _, W, s = tiny
    if pool:
        monkeypatch.setenv("OTPU_POOL", "1")
    t = T.load(TARGET)
    need = s.image(T.config(t, DRAM_BYTES=1 << 40), 256, wformat="fp4", lookup=True).nbytes
    cfg = T.config(t, DRAM_BYTES=1 << max(22, (need - 1).bit_length() + 1))
    tr = _transport(cfg.DRAM_BYTES)
    isa = Engine(s, W, cap=256, cfg=cfg, rows=1, wformat="fp4", resident=True)
    brd = Engine(s, W, cap=256, cfg=cfg, rows=1, wformat="fp4", resident=True,
                 backend=lambda c, imgs: BoardBackend(c, imgs, transport=tr))
    assert brd.resident and brd.can_generate and (brd.image.pool is not None) == pool
    if not pool:
        assert brd.server.misses == 0
        brd.backend.write(0, brd.image.offload.served, np.array([3e38], np.float32))
    for tok in (11, 222, 333):
        a, b = isa.step(tok), brd.step(tok)
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32)), brd.pos
        assert brd.stats[-1]["instructions"] == isa.stats[-1]["instructions"]
    t0 = int(np.argmax(a))
    assert brd.generate_card(t0, 3, stop_ids=[]) == isa.generate_card(t0, 3, stop_ids=[])
    assert pool or isa.server.misses == 0
    assert brd.stats[-1]["cycles"] > 0


@pytest.mark.parametrize("kind,wformat,embed", [(GO.SLIDE, "int8", "f32"), (GO.FULL, "int8", "f32"),
                                                (GO.SLIDE, "fp4", "int8")])
def test_ring_wraps(kind, wformat, embed):
    """Past 256 positions: a sliding layer's ring wraps (its slots of the last lap masked by the
    second table), a full layer attends over two blocks; the device follows the emulation (int8,
    and fp4 on a sliding layer with the int8 embedding table, Spec.embed). One layer, so that
    each position's K and V are its embedding's alone: a QACT value within fp32 rounding of a
    tie (0.5 - 2e-5, which the emulation's float64 cannot place) changes that one position's
    logits (1 - cos ~1e-4 through the MoE), not every later one's; a ring or mask error would
    change every position past the wrap."""
    m, W, hc = _tiny(window=24, layers=1, experts=K)
    import tempfile
    d = tempfile.mkdtemp()
    hc.save_pretrained(d)
    s = dataclasses.replace(Spec.from_hf(d), kinds=(kind,), embed=embed)
    toks = [int(t) for t in np.random.default_rng(7).integers(0, V, 300)]
    eng = _engine(s, W, wformat, cap=512)
    dev = np.array([eng.step(t) for t in toks])
    emu = emulated_logits(s, W, toks, wformat=wformat)
    dc = 1 - _cos(dev, emu)
    off = np.nonzero(dc > 1e-6)[0]
    assert np.median(dc[256:]) < 1e-7 and len(off) <= 15, (np.median(dc[256:]), off, dc[off])
    assert dc.max() < 1e-3, (dc.argmax(), dc.max())
    keep = np.setdiff1d(np.arange(len(toks)), off)
    assert np.array_equal(dev[keep].argmax(1), emu[keep].argmax(1))


def test_generate_loop(tiny):
    """The decode loop on the device (generate.py: gptoss_step in its token loop) picks the
    greedy tokens of the resident step's logits."""
    _, W, s = tiny
    toks = [int(t) for t in np.random.default_rng(8).integers(0, V, 8)]
    a, b = _engine(s, W, "fp4"), _engine(s, W, "fp4")
    assert a.can_generate
    for t in toks[:-1]:
        a.step(t)
        b.step(t)
    ref, t = [], toks[-1]
    for _ in range(10):
        t = int(np.argmax(b.step(t)))
        ref.append(t)
    assert a.generate_card(toks[-1], 10, stop_ids=[]) == ref


@pytest.mark.parametrize("aw", [32, 36])
def test_generate_loop_bounds_its_registers(tiny, monkeypatch, aw):
    """The generate loop's registers loaded from the state block's tpos carry its bound
    (compiler.mul_bound): the argument RLD MULs, the sliding layers' mask row at -4 * tpos too
    (its bound wraps mod 2^AW), and the layer loop's register that starts at 4 * tpos (the V^T
    appends' rows), so the waits pass takes those footprints as their rows and not all of DRAM:
    fewer MMs and LDs wait for the quantizer's stores than with the bounds stripped (GPT-OSS-20B's
    loop ran 108k cycles a token behind its decode step on the card); the counts hold on the
    ISA simulator's run (OTPU_WAITS_CHECK) and the tokens are the resident step's."""
    from dataclasses import replace

    from opentpu import isa as I
    from opentpu import waits as WT
    _, W, s = tiny
    a, b = _engine(s, W, "fp4", AW=aw), _engine(s, W, "fp4", AW=aw)
    prog = a._generate_prog(1)[0]
    muls = [(i, x) for i, x in enumerate(prog)
            if x.op == I.RLD and x.flags & I.F_MUL and "tpos" in x.comment]
    assert muls and all(x.flags & I.F_BOUND for _, x in muls)
    neg = [x for _, x in muls if "-4*tpos" in x.comment]
    assert neg and all(x.w[3] | x.w[5] << 32 == 1 << aw for x in neg)
    assert any("argument" not in x.comment for _, x in muls)      # (the loop's register)
    picks = [x for x in prog if x.op == I.RLD and x.comment == "the best chunk's id"]
    assert picks and all(x.flags & I.F_BOUND for x in picks)        # (Greedy's: generate._pick)
    loose = list(prog)
    for i, x in muls:
        loose[i] = replace(x, flags=x.flags & ~I.F_BOUND, w=[*x.w[:2], 0, 0, 0, 0, x.w[6]])

    def quant_waits(p):
        wc = WT.wait_counts(p, S=a.cfg.S, aw=aw)
        return sum(1 for x, c in zip(p, wc) if x.op == I.MM and WT.counts(c)[6] or
                   x.op == I.LD and WT.counts(c)[2])
    assert quant_waits(list(prog)) < quant_waits(loose)
    toks = [int(t) for t in np.random.default_rng(8).integers(0, V, 8)]
    for t in toks[:-1]:
        a.step(t)
        b.step(t)
    ref, t = [], toks[-1]
    for _ in range(6):
        t = int(np.argmax(b.step(t)))
        ref.append(t)
    monkeypatch.setenv("OTPU_WAITS_CHECK", "1")
    assert a.generate_card(toks[-1], 6, stop_ids=[]) == ref


def _spec_20b(tmp_path) -> Spec:
    """GPT-OSS-20B's Spec from its config alone (no weights)."""
    import json
    c = {"model_type": "gpt_oss", "hidden_size": 2880, "intermediate_size": 2880,
         "num_hidden_layers": 24, "num_attention_heads": 64, "num_key_value_heads": 8,
         "head_dim": 64, "num_local_experts": 32, "num_experts_per_tok": 4,
         "vocab_size": 201088, "sliding_window": 128, "rms_norm_eps": 1e-5,
         "attention_bias": True, "tie_word_embeddings": False, "swiglu_limit": 7.0,
         "rope_theta": 150000, "layer_types": ["sliding_attention", "full_attention"] * 12,
         "rope_scaling": {"rope_type": "yarn", "factor": 32.0, "beta_fast": 32.0,
                          "beta_slow": 1.0, "original_max_position_embeddings": 4096,
                          "truncate": False}}
    (tmp_path / "config.json").write_text(json.dumps(c))
    return Spec.from_hf(tmp_path)


def test_20b_image_fit(tmp_path):
    """GPT-OSS-20B's layout on the BCU1525 at WG 8 (from its config alone, no weights):
    tools/image_fit.py's parts sum to what the image holds at 32 GiB (Image.high: the weights,
    the head and every expert in a resident pool from 4 GiB on), the attention projections
    int8 (Spec.formats) and the experts fp4, a token reads 4 experts a layer, the embedding
    table fp32; at 4 GiB, 4 slots a layer (the experts streamed, path (a))
    and the int8 table. Both decode programs fit IMEM."""
    import sys
    from pathlib import Path
    from opentpu import target as T
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    import image_fit
    s = _spec_20b(tmp_path)
    cfg = T.config("bcu1525-wg", DRAM_BYTES=32 << 30)
    img = s.image(cfg, 4096, wformat="fp4", head_format="int8", lookup=True)
    r = image_fit.gptoss_report(img, 4095)
    p, fmt, P = r["parts"], img.fmt, img.pool
    assert img.mf == dict.fromkeys(("wq", "wk", "wv", "wo"), "int8") and fmt.wformat == "fp4"
    assert img.high and img.offload is None and img.low_end < GO.HIGH == img.layer0 < P.base
    assert sum(p.values()) == img.low_end + img.nbytes - GO.HIGH and (fmt.H, fmt.Fg, fmt.F,
                                                                      fmt.Hd) == \
        (2944, 2880, 2944, 2880) and fmt.cw == [256] * 11 + [128]
    assert r["pool"][2] == P.experts == 24 * 32 and p["experts"] == 768 * P.ebytes
    assert P.ebytes == fmt.nbytes >= r["expert"][0] > r["expert"][1]
    rd = r["read"]
    assert sum(rd[k][0] for k in ("experts gate, up", "experts down", "experts biases")) == \
        24 * 4 * r["expert"][0]
    assert sum(rd[k][1] for k in ("experts gate, up", "experts down", "experts biases")) == \
        24 * 4 * r["expert"][1]
    assert all(a >= b > 0 for a, b in rd.values())
    assert p["embed"] == 201088 * 4 * 2944 and rd["embedding row"] == (4 * 2944, 4 * 2880)
    small = dataclasses.replace(s, embed="int8").image(
        T.config("bcu1525-wg"), 4096, wformat="fp4", head_format="int8", lookup=True, experts=4)
    assert not small.high and small.offload is not None
    for im in (img, small):
        progs, _ = im.compile_decode(16, 15 * 256 + 3)
        assert len(progs) == 1 and len(progs[0]) <= T.config("bcu1525-wg").IMEM_WORDS // 8


def _dram_ranges(prog, cfg, monkeypatch, run_args=None, rld=None) -> list:
    """(instruction, lo, hi): every DRAM range [lo, hi) the program may touch but WAITW's (all
    of DRAM, by design), as opentpu.waits' abstract execution has them for its entries
    (registers as intervals over every path, footprint() of every value they may hold);
    run_args and rld bound the arguments and RLDs (wait_counts)."""
    from opentpu import isa as I
    from opentpu import waits as W
    out, fp = [], W.footprint

    def rec(ins, R, D=128, S=1, aw=32):
        f = fp(ins, R, D, S, aw)
        if ins.op != I.WAITW:
            out.extend((ins, lo, hi) for sp, lo, hi in f.rd + f.wr if sp == W.SP_DRAM)
        return f
    with monkeypatch.context() as m:
        m.setattr(W, "footprint", rec)
        W.wait_counts(prog, cfg.D, cfg.S, aw=cfg.AW, run_args=run_args, rld=rld)
    return out


def test_20b_addresses(tmp_path, monkeypatch):
    """GPT-OSS-20B at 32 GiB on bcu1525-wg (AW 36), laid out and compiled: Image.high's low
    area (I/O, the KV caches, the tables) below HIGH - LOW_ROOM, every layer's weights and the
    head in [HIGH, the pool), the resident pool's 768 experts up to the image's end, within
    the DRAM and AW's 2^36. Every DRAM range the decode and generate programs of buckets 1 and
    16 may touch (opentpu.waits' footprints, the token below the vocabulary and the position
    below its bucket's bound, as the decode programs' arguments and the generate loop's RLDs
    of them) is below HIGH or in [HIGH, nbytes), none across HIGH, and the experts' within the
    pool, from its first expert to its last. The 4-experts image (path (a)): its words, its
    directory and its slots below 4 GiB (32-bit directory entries), its decode programs' ranges
    within the image and the expert MMs' within the slots."""
    import re
    from types import SimpleNamespace
    from opentpu import isa as I
    from opentpu import target as T
    from opentpu import waits as W
    from opentpu.llm.qwen3 import LOW_ROOM
    s = _spec_20b(tmp_path)
    cfg = T.config("bcu1525-wg", DRAM_BYTES=32 << 30)
    img = s.image(cfg, 4096, wformat="fp4", head_format="int8", lookup=True)
    P, imem = img.pool, cfg.IMEM_WORDS // 8
    end = P.base + P.experts * P.ebytes
    assert img.high and cfg.AW == 36 and end == img.nbytes <= cfg.DRAM_BYTES < 1 << cfg.AW
    assert img.low_end <= GO.HIGH - LOW_ROOM and img.layer0 == GO.HIGH
    at = {li: base + it * us + off for li, (base, us, it, off, _) in img.loc.items()}
    kv = {li: base + it * us + off for li, (base, us, it, off, _) in img.lloc.items()}
    assert len(at) == len(kv) == 24 and GO.HIGH <= min(at.values()) <= max(at.values()) < \
        min(img.head) <= max(img.head) < P.base and max(kv.values()) < img.low_end
    rs = []
    for b in (1, 16):
        progs, ra = img.compile_decode(b, (b - 1) * 256 + 3)
        gp = img.compile_generate(b, (b - 1) * 256 + 3)
        assert len(progs) == 1 and max(len(progs[0]), *map(len, gp)) <= imem
        bound = {v.name: v.bound or s.vocab for v, _ in ra}
        rs += _dram_ranges(progs[0], cfg, monkeypatch, [
            (SimpleNamespace(name=v.name, bound=bound[v.name]), c) for v, c in ra])
        rld = {}
        for pc, ins in enumerate(gp[0]):
            m = re.fullmatch(r"(?:argument )?(-?\d+)\*(\w+~?)", ins.comment)
            if ins.op == I.RLD and m:
                rld[pc] = W._norm(*sorted((0, int(m[1]) * (bound[m[2]] - 1))), cfg.AW)
        assert {re.sub(r"-?\d+\*", "", gp[0][pc].comment) for pc in rld} >= {"argument tok",
                                                                          "argument tpos"}
        rs += _dram_ranges(gp[0], cfg, monkeypatch, rld=rld)
    bad = [(i.comment, lo, hi) for i, lo, hi in rs
           if not (hi <= GO.HIGH or GO.HIGH <= lo and hi <= img.nbytes)
           or lo < end and hi > P.base and not P.base <= lo]
    assert not bad, bad[:8]
    pool = [(lo, hi) for _, lo, hi in rs if lo >= P.base]
    assert min(lo for lo, _ in pool) == P.base and max(hi for _, hi in pool) > end - P.ebytes
    small = dataclasses.replace(s, embed="int8").image(
        cfg, 4096, wformat="fp4", head_format="int8", lookup=True, experts=4)
    Lo = small.offload
    assert not small.high and small.pool is None and Lo.end <= small.nbytes < GO.HIGH - LOW_ROOM
    assert max(Lo.mbox, Lo.served, Lo.answer, Lo.dir) < Lo.slots[0][0] <= Lo.slot_bounds[1] < \
        Lo.end < 1 << 32
    for b in (1, 16):
        progs, ra = small.compile_decode(b, (b - 1) * 256 + 3)
        bound = {v.name: v.bound or s.vocab for v, _ in ra}
        rs = _dram_ranges(progs[0], cfg, monkeypatch, [
            (SimpleNamespace(name=v.name, bound=bound[v.name]), c) for v, c in ra])
        assert max(hi for _, _, hi in rs) <= small.nbytes
        sl = [(lo, hi) for _, lo, hi in rs if lo >= Lo.slots[0][0]]
        assert min(lo for lo, _ in sl) == Lo.slots[0][0] and max(hi for _, hi in sl) <= Lo.end


def test_board_perf_decode_runs_without_its_host(tmp_path, monkeypatch):
    """tools/board_perf.py decode on a zero image (here the ISA simulator's; --sim the board
    model's): a GPT-OSS image's experts are in path (a)'s slots, and with present_words' served
    and directory the decode runs without the host (without them the first layer's experts
    read as missing and their answer words never come; past them the second layer's fence,
    served >= seq, would never hold)."""
    import json
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    import board_perf
    from opentpu.isasim import SimError
    c = {"model_type": "gpt_oss", "hidden_size": H, "intermediate_size": F,
         "num_hidden_layers": 2, "num_attention_heads": 8, "num_key_value_heads": 2,
         "head_dim": 64, "num_local_experts": E, "num_experts_per_tok": K, "vocab_size": V,
         "sliding_window": 16, "rms_norm_eps": 1e-5, "attention_bias": True,
         "tie_word_embeddings": False, "swiglu_limit": 7.0, "rope_theta": 150000,
         "layer_types": ["sliding_attention", "full_attention"],
         "rope_scaling": {"rope_type": "yarn", "factor": 32.0, "beta_fast": 32.0,
                          "beta_slow": 1.0, "original_max_position_embeddings": 4096,
                          "truncate": False}}
    (tmp_path / "config.json").write_text(json.dumps(c))
    argv = ["decode", "--isa", "--target", "bcu1525-wg", "--points",
            f"decode={tmp_path}:fp4:int8:20", "--json", str(tmp_path / "out.json")]
    assert board_perf.main(argv) == 0
    rec = json.loads((tmp_path / "out.json").read_text())["decode"][0]
    assert rec["ok"] and rec["instructions"][0] > 500
    monkeypatch.setattr(board_perf, "present_words", lambda img: [])
    with pytest.raises(SimError):
        board_perf.main(argv)
