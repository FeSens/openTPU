"""Gemma 4 on openTPU: sliding and global attention with a KV ring and shared KV layers, GeGLU,
per-layer embeddings gathered on the device, against Hugging Face transformers. A tiny random
model always runs; the real model's programs are compiled when its config is in
models/gemma-4-E2B."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from opentpu.compiler import CompileError
from opentpu.isasim import board_config
from opentpu.kernels import gather as GA
from opentpu.llm import gemma4 as G
from opentpu.llm.qwen3 import Engine

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("transformers.models.gemma4")

REAL = Path(__file__).resolve().parent.parent / "models" / "gemma-4-E2B"
REAL_E4B = REAL.parent / "gemma-4-E4B"
# (two own sliding layers and a global one) x 2, a shared sliding one and two shared globals
# (HF forces the last layer global): layers 6..8 read layer 4's (sliding) and 5's (global) K / V.
# The layer loops: (s s f) x 2 with the two sliding layers a loop inside, s, then f x 2
S_, F_ = "sliding_attention", "full_attention"
KINDS = (S_, S_, F_, S_, S_, F_, S_, F_, F_)


def _cos(a, b):
    return (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)


def _model(d, kinds=KINDS, shared=3):
    """A tiny random Gemma 4 of these layer kinds, the last `shared` reading earlier layers'
    K / V: (HF model, weights, Spec), its config.json in d."""
    torch.manual_seed(0)
    hc = transformers.Gemma4TextConfig(
        hidden_size=256, num_hidden_layers=len(kinds), num_attention_heads=8,
        num_key_value_heads=1, head_dim=128, global_head_dim=256, intermediate_size=512,
        vocab_size=1000, vocab_size_per_layer_input=1000, hidden_size_per_layer_input=128,
        layer_types=list(kinds), num_kv_shared_layers=shared, use_double_wide_mlp=True,
        sliding_window=512, final_logit_softcapping=30.0, max_position_embeddings=4096)
    hc._attn_implementation = "eager"
    m = transformers.models.gemma4.modeling_gemma4.Gemma4ForCausalLM(hc).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:
                k = "k_norm" in n
                p.copy_((0.13 if k else 1.0) + (0.01 if k else 0.1) * torch.randn_like(p))
        for n, b in m.named_buffers():
            if n.endswith("layer_scalar"):
                b.copy_(0.5 + torch.rand_like(b))
    W = {k: v.float().numpy() for k, v in m.state_dict().items()}
    (d / "config.json").write_text(json.dumps(hc.to_dict()))
    return m, W, G.Spec.from_hf(d)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """head_dim 128 (sliding) and 256 (global, 32 rotated pairs), 8 query heads on 1 KV head
    (two parts of 4 on the board's MXU), a double-wide MLP in the shared layers, a 128-wide
    per-layer input. k_norm near the real model's 0.13 (attention is not scaled by 1/sqrt(d):
    with k_norm ~1 the scores are ~10x larger and int8 K dominates the error)."""
    return _model(tmp_path_factory.mktemp("gemma4"))


def _cfg():
    return board_config(DRAM_BYTES=1 << 26)


def test_spec_kv_sources(tiny):
    _, _, spec = tiny
    assert spec.kinds == ("sliding", "sliding", "full") * 2 + ("sliding", "full", "full")
    assert spec.kv_src == (0, 1, 2, 3, 4, 5, 4, 5, 5)
    assert spec.ffn == (512,) * 6 + (1024,) * 3
    assert spec.global_rot == 32
    img = spec.image(_cfg(), 1024)
    assert [(f, len(u), r) for f, u, r in img.runs] == [(0, 3, 2), (6, 1, 1), (7, 1, 2)]
    assert [(e0, len(su), r) for e0, su, r, _ in img.subs[0]] == [(0, 1, 2), (2, 1, 1)]


def test_reference_matches_hf(tiny):
    m, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(0).integers(0, 1000, 32)]
    with torch.no_grad():
        hf = m(torch.tensor([toks])).logits[0].numpy()
    assert np.abs(G.softcap(spec, G.reference_logits(spec, W, toks)) - hf).max() < 1e-4


def test_tiny_matches_hf_and_emulation(tiny):
    m, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(0).integers(0, 1000, 24)]
    with torch.no_grad():
        hf = m(torch.tensor([toks])).logits[0].numpy()
    eng = Engine(spec, W, cap=1024, cfg=_cfg())
    dev = np.array([G.softcap(spec, eng.step(t)) for t in toks])
    assert _cos(dev, hf).min() > 0.985          # int8 noise of a random model (emulation 0.991)
    emu = G.softcap(spec, G.emulated_logits(spec, W, toks[:12]))
    assert _cos(dev[:12], emu).min() > 0.99


def test_host_sampler_caps_as_hf(tiny):
    """The host's sampler (otpu-chat's picks off the card, its first token on the card) with the
    model's softcap samples Hugging Face's distribution: Gemma4ForCausalLM caps the logits
    before the penalty and the warpers. The final norm x30 puts the raw logits up to ~31 (a real
    model's range), where the cap matters: with the same uniforms the picks are HF's, from
    whole logits and streamed pieces; uncapped they are not. Engine.generate hands its sampler
    the capped logits; Chat takes no sampler of another softcap."""
    import copy
    import types

    from transformers.generation import logits_process as LP

    from opentpu.host.chat import Chat, sampler
    m, W, spec = tiny
    m = copy.deepcopy(m)
    toks = [int(t) for t in np.random.default_rng(0).integers(0, 1000, 16)]
    with torch.no_grad():
        m.model.norm.weight.mul_(30.0)
        capped = m(torch.tensor([toks])).logits[0, -1]
        m.config.final_logit_softcapping = None
        raw = m(torch.tensor([toks])).logits[0, -1].numpy()
    assert raw.max() > 25.0 and capped.max() < 0.8 * raw.max()
    for T, k, tp, rp in [(1.0, 64, 0.95, 1.0), (0.8, 40, 0.95, 1.1), (1.0, 0, 1.0, 1.0)]:
        procs = LP.LogitsProcessorList([LP.RepetitionPenaltyLogitsProcessor(rp),
                                        LP.TemperatureLogitsWarper(T),
                                        LP.TopKLogitsWarper(k or 1000), LP.TopPLogitsWarper(tp)])
        p = torch.softmax(procs(torch.tensor([toks]), capped[None].clone()), -1)[0]
        p = p.double().numpy()
        kept = np.argsort(-p, kind="stable")[:int((p > 0).sum())]
        rng = np.random.default_rng(3)
        want = [int(kept[rng.choice(len(kept), p=p[kept] / p[kept].sum())])
                for _ in range(300)]
        assert len(set(want)) > 5
        pick, stream = (sampler(T, k, tp, 3, rp, spec.softcap) for _ in range(2))
        assert [pick(raw, toks) for _ in range(300)] == want
        got = []
        for _ in range(300):
            s = stream.stream(toks)
            s.begin(len(raw))
            for lo in (512, 0):                         # pieces as the card streams them
                s.feed(lo, raw[lo:lo + 512])
            got.append(s.result())
        assert got == want
        uncapped = sampler(T, k, tp, 3, rp)
        assert sum(uncapped(raw, toks) != w for w in want) > 30
    # Engine.generate: the sampler gets the capped logits; a sampler that caps is refused
    eng, ref = (Engine(spec, W, cap=1024, cfg=_cfg()) for _ in range(2))
    seen = []
    eng.generate(toks[:4], max_new=1, sampler=lambda lg: seen.append(lg) or 0)
    assert np.array_equal(seen[0], G.softcap(spec, ref.prefill(toks[:4])))
    with pytest.raises(ValueError, match="softcap"):
        eng.generate(toks[:4], sampler=sampler(1.0, 64, 0.95, 3, softcap=spec.softcap))
    gemma = types.SimpleNamespace(spec=spec)
    with pytest.raises(ValueError, match="softcap"):
        Chat(gemma, None, False, sampler(1.0, 64, 0.95, 3), 8)
    Chat(gemma, None, False, sampler(1.0, 64, 0.95, 3, softcap=spec.softcap), 8)


def test_emulation_rounds_ties_as_the_device(tmp_path):
    """A 4-bit LM head's rows (the embedding rows) quantize to int8 with exact ties (fp4 codes
    3 and 6 make 63.5), at the input of the PLE projection: _fake_q rounds a tie as the
    device's quantizer does, so every row's int8 values are fp32.quantize's. On one global
    layer with per-layer inputs, in fp4, emulated_logits is then the ISA simulator's logits to
    fp32 rounding at every position (with the ties rounded half to even, 6 of the 12 positions
    were 0.5 to 1% off; Hugging Face's forward with the device's weights, PLE records and
    quantizer agrees with the device)."""
    from opentpu import fp32 as F
    from opentpu.llm.qwen3 import _fake_q, _fake_w
    _, W, spec = _model(tmp_path, (F_,), 0)
    xb = (_fake_w(W["model.embed_tokens.weight"], 128, "fp4") * 16).reshape(1000, -1, 128)
    s = np.maximum(np.abs(xb).max(-1, keepdims=True), 1e-30) / 127     # (token 0's row: 0)
    r = xb / s
    assert (np.abs(r - np.rint(r)) > 0.5 - 1e-9).sum() > 10000
    assert np.array_equal(np.rint(_fake_q(xb) / s), F.quantize(xb.astype(np.float32))[0])
    toks = [int(t) for t in np.random.default_rng(0).integers(0, 1000, 12)]
    eng = Engine(spec, W, cap=1024, cfg=_cfg(), wformat="fp4")
    dev = np.array([eng.step(t) for t in toks])
    emu = G.emulated_logits(spec, W, toks, wformat="fp4")
    assert np.abs(dev - emu).max() < 1e-5 * np.abs(emu).max()


MIX = "attn@0-2=fp4,down@3-5=fp4,gateup@6-8=fp4,ple@6-8=fp4"


def test_per_layer_formats(tiny):
    """Weight formats per layer and kind (Spec.formats, docs/gemma4_e4b.md): a layer block's
    layout follows its formats, so the loops split where they change. A 4-row prefill equals
    token-by-token decoding bit for bit, and the device follows the emulation of the same
    formats, closer than the all-int8 emulation."""
    _, W, spec = tiny
    mix = replace(spec, formats=MIX)
    img = mix.image(_cfg(), 1024)
    assert [(f, len(u), r) for f, u, r in img.runs] == [(0, 1, 2), (2, 1, 1), (3, 1, 2),
                                                         (5, 1, 1), (6, 1, 1), (7, 1, 2)]
    assert img.lf[:3] == (("fp4", "int8", "int8", "int8"),) * 3
    assert img.lf[3:6] == (("int8", "int8", "fp4", "int8"),) * 3
    assert img.lf[6:] == (("int8", "fp4", "int8", "fp4"),) * 3 and img.pformat == "int8"
    assert img.nbytes < spec.image(_cfg(), 1024).nbytes
    with pytest.raises(ValueError, match="weight format"):
        G.layer_formats(spec, "int8", "mlp@0-2=fp8")
    toks = [int(t) for t in np.random.default_rng(8).integers(0, 1000, 12)]
    a = Engine(mix, W, cap=1024, cfg=_cfg())
    b = Engine(mix, W, cap=1024, cfg=_cfg())
    dev = np.array([G.softcap(spec, b.step(t)) for t in toks])
    assert np.array_equal(G.softcap(spec, a.prefill(toks, chunk=4)), dev[-1])
    emu = G.softcap(spec, G.emulated_logits(mix, W, toks))
    e8 = G.softcap(spec, G.emulated_logits(spec, W, toks))
    assert _cos(dev, emu).min() > 0.99
    assert np.abs(dev - emu).mean() < 0.7 * np.abs(dev - e8).mean()


def test_formats_by_fit(tiny, monkeypatch):
    """The formats follow the fit (Spec.fit_formats, docs/gemma4_e4b.md): an int8 image that
    fits the card beside no PLE choice takes the head and the own-KV layers' down projections
    in fp4 (one loop's format, so no extra loop); an image that fits, or not int8, is as asked.
    The Engine's compile worker gets the choices (Engine._image_kw): its image is the same
    without choosing again."""
    _, W, spec = tiny
    assert spec.fit_formats == "head=fp4,down@0-5=fp4"
    monkeypatch.delenv("OTPU_FORMATS", raising=False)
    big = spec.image(_cfg(), 1024, ple_host=True, formats="")
    mix = spec.image(_cfg(), 1024, ple_host=True, formats=spec.fit_formats)
    assert spec.image(_cfg(), 1024).formats == ""
    monkeypatch.setattr(G, "CARD_BYTES", (big.nbytes + mix.nbytes) // 2)
    img = spec.image(_cfg(), 1024)
    assert img.formats == spec.fit_formats and img.head_format == "fp4" and img.ple_host
    assert img.lf[5] == ("int8", "int8", "fp4", "int8") and img.lf[6] == ("int8",) * 4
    assert img.nbytes == mix.nbytes and img.runs == mix.runs and len(img.runs) == len(big.runs)
    assert spec.image(_cfg(), 1024, wformat="fp4").formats == ""
    eng = Engine(spec, W, cap=1024, cfg=_cfg())
    kw = eng._image_kw
    assert kw["formats"] == spec.fit_formats and kw["ple_host"] and kw["head_format"] == "fp4"
    monkeypatch.setattr(G, "CARD_BYTES", 1 << 32)
    again = spec.image(eng.cfg, 1024, 1, eng.rows, **kw)
    assert (again.nbytes, again.lf, again.head_format) == (eng.image.nbytes, eng.image.lf, "fp4")


def test_build_goes_through_the_image_cache(tiny, tmp_path, monkeypatch):
    """The image build's 4-bit matrices go through opentpu.qcache (its key: the matrix's
    content): a second build of the same model takes them from the cache, byte for byte the
    first's, without quantizing; int8 is not cached."""
    from opentpu import qcache as QC
    from opentpu import quant as Q
    _, W, spec = tiny
    monkeypatch.setenv("OTPU_IMAGE_CACHE", str(tmp_path))
    monkeypatch.setattr(QC, "MIN_ELEMS", 0)                     # (the tiny model's matrices)
    monkeypatch.setattr(QC, "FREE_FLOOR", 0)                    # (omarchy's tmp: 16 GB tmpfs)
    first = spec.image(_cfg(), 1024, wformat="fp4", head_format="int8").build(W)[0]
    n = len(list(tmp_path.rglob("*.npz")))
    assert n > 0
    real = Q.quantize_mxu

    def int8_only(a, fmt, D=128):
        assert fmt == "int8", "a 4-bit matrix quantized again"
        return real(a, fmt, D)
    monkeypatch.setattr(Q, "quantize_mxu", int8_only)
    again = spec.image(_cfg(), 1024, wformat="fp4", head_format="int8").build(W)[0]
    assert np.array_equal(first, again) and len(list(tmp_path.rglob("*.npz"))) == n


def test_build_workers(tiny, tmp_path, monkeypatch):
    """Image.build from a checkpoint (Weights) quantizes in worker processes: the image is the
    in-line build's, byte for byte; a worker's job returns the opentpu.qcache counts it made,
    which the parent adds to its own (prebuild prints them)."""
    from safetensors.numpy import save_file
    from opentpu import qcache as QC
    _, W, spec = tiny
    save_file({"model.language_model." + k[6:]: np.ascontiguousarray(v) for k, v in W.items()
               if k.startswith("model.")}, str(tmp_path / "model.safetensors"))
    monkeypatch.setenv("OTPU_IMAGE_CACHE", "0")
    monkeypatch.setenv("OTPU_BUILD_JOBS", "2")
    img = spec.image(_cfg(), 1024, wformat="fp4", head_format="int8")
    want = img.build(W)
    got = img.build(G.Weights(tmp_path))
    assert len(got) == len(want) and all(np.array_equal(a, b) for a, b in zip(got, want))
    monkeypatch.setenv("OTPU_IMAGE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(QC, "MIN_ELEMS", 0)
    monkeypatch.setattr(QC, "FREE_FLOOR", 0)
    monkeypatch.setattr(G, "_JOB_W", W)
    job = ("mat", "model.layers.0.mlp.gate_proj.weight", None, 1.0, "fp4", 128)
    out, n = G._worker_job(job)
    assert n == {"hit": 0, "miss": 1, "write": 1, "skip": 0}
    again, n = G._worker_job(job)
    assert n == {"hit": 1, "miss": 0, "write": 0, "skip": 0}
    assert all(np.array_equal(a, b) for a, b in zip(out, again))


@pytest.mark.parametrize("ple", ["int8", "fp4"])
def test_records_roundtrip(ple):
    """pack_records / dequant_records: the device's gather values of a packed table."""
    rows = np.random.default_rng(1).standard_normal((5, 896)).astype(np.float32)
    S = GA.record_blocks(7, ple)
    assert S % 2 == 1 or S % 4 == 2
    back = GA.dequant_records(GA.pack_records(rows, ple, 128, S), ple, 128, S)[:, :896]
    assert np.abs(back - rows).max() / np.abs(rows).max() < (0.005 if ple == "int8" else 0.15)


@pytest.mark.parametrize("wf,ple,fm", [("int8", "int8", ""), ("fp4", "fp4", ""),
                                       ("int8", "int8", MIX)], ids=["int8", "fp4", "mix"])
def test_resident_gathers_are_bit_exact(tiny, wf, ple, fm, monkeypatch):
    """Resident decode gathers the embedding and PLE rows on the device and computes its
    attention masks at the run-time position: the logits equal the per-position programs'
    (host-written rows), bit for bit, across the first bucket boundary; also with formats per
    layer (OTPU_FORMATS)."""
    _, W, spec = tiny
    monkeypatch.setenv("OTPU_PLE_FORMAT", ple)
    monkeypatch.setenv("OTPU_FORMATS", fm)
    toks = [int(t) for t in np.random.default_rng(2).integers(0, 1000, 300)]
    a = Engine(spec, W, cap=1024, cfg=_cfg(), wformat=wf, resident=True)
    b = Engine(spec, W, cap=1024, cfg=_cfg(), wformat=wf)
    assert a.resident and not b.resident and a.image.ple_format == ple
    a.prefill(toks[:250])
    b.prefill(toks[:250])
    for t in toks[250:]:
        assert np.array_equal(a.step(t), b.step(t))


@pytest.mark.parametrize("resident", [False, True])
def test_filling_step_programs_are_transparent(tiny, resident):
    """The step programs that fill their logits first (the card's streamed decode) give the
    plain programs' logits and DRAM bit for bit, per position and resident."""
    from conftest import assert_fill_is_transparent
    _, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(8).integers(0, 1000, 6)]
    assert_fill_is_transparent(lambda: Engine(spec, W, cap=1024, cfg=_cfg(), resident=resident),
                               toks)


def test_ring_wrap_resident_is_bit_exact(tiny):
    """Positions 760..775: the sliding window's start and end blocks, across the ring's wrap
    (768 slots), resident against per-position."""
    _, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(3).integers(0, 1000, 776)]
    a = Engine(spec, W, cap=1024, cfg=_cfg(), resident=True)
    b = Engine(spec, W, cap=1024, cfg=_cfg())
    la, lb = a.prefill(toks[:760]), b.prefill(toks[:760])
    assert np.array_equal(la, lb)
    for t in toks[760:]:
        assert np.array_equal(a.step(t), b.step(t))


def test_chunked_prefill_is_bit_exact(tiny):
    """Prefill runs of several rows give the logits and caches of token-by-token decoding."""
    _, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(4).integers(0, 1000, 11)]
    a = Engine(spec, W, cap=1024, cfg=_cfg())
    b = Engine(spec, W, cap=1024, cfg=_cfg())
    la = a.prefill(toks, chunk=4)
    lb = b.prefill(toks, chunk=1)
    assert np.array_equal(la, lb)
    img = a.image
    da = a.backend.machine.slices[0].dram[img.layer0:img.head[0]]
    db = b.backend.machine.slices[0].dram[img.layer0:img.head[0]]
    assert np.array_equal(da, db)


def test_device_inputs_are_bit_exact(tiny):
    """A prefill run and a per-position program with their tokens compiled in gather the
    embedding and PLE rows and load the RoPE rows on the device (the host writes nothing):
    logits and caches equal those of the programs with host-written inputs (host_inputs)."""
    _, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(6).integers(0, 1000, 14)]
    engs = [Engine(spec, W, cap=1024, cfg=_cfg(), wformat="fp4", head_format="int8",
                   resident=True) for _ in range(2)]
    for e in engs:
        e.prefill(toks[:9], chunk=4)
    img, V = engs[0].image, spec.vocab

    def run(e, rows, tokens, device):
        if not device:
            for addr, v in e.image.host_inputs(tokens, [p for _, p in rows]):
                e.backend.write(0, addr, v)
        e.backend.run(e.image.compile_rows(rows, [len(rows) - 1],
                                           tokens=tokens if device else None))
        return e.backend.read(0, img.io["logits"] + 4 * V * (len(rows) - 1), 4 * V)

    for rows, tk in (([(0, 9 + r) for r in range(4)], toks[9:13]), ([(0, 13)], toks[13:])):
        assert np.array_equal(run(engs[0], rows, tk, False), run(engs[1], rows, tk, True))
    lo, hi = img.layer0, img.head[0]
    assert np.array_equal(engs[0].backend.machine.slices[0].dram[lo:hi],
                          engs[1].backend.machine.slices[0].dram[lo:hi])


@pytest.mark.slow
@pytest.mark.parametrize("resident,host", [(False, "0"), (True, "0"), (True, "1")],
                         ids=["per-position", "resident", "resident-ple-host"])
def test_token_on_board_rtl_is_bit_exact(have_verilator, tiny, resident, host, monkeypatch):
    """A token at position 600 (the window's first block masked at its start) on the Verilator
    RTL through the board's memory path (AXI adapter, boot loader) equals the ISA simulator bit
    for bit, logits and the whole DRAM image (caches); resident, with the device's gathers and
    run-time masks; and with the PLE table on the host (the slot the host writes, the fence
    on its mailbox: WAITW holding at its first read)."""
    from opentpu import rtlsim
    from opentpu.llm.rtl_backend import RtlBackend
    _, W, spec = tiny
    monkeypatch.setenv("OTPU_PLE_HOST", host)
    toks = [int(t) for t in np.random.default_rng(5).integers(0, 1000, 601)]
    eng = Engine(spec, W, cap=1024, cfg=_cfg(), wformat="fp4", head_format="int8",
                 resident=resident)
    assert eng.resident == resident
    eng.prefill(toks[:-1])
    n, isa = eng.image.nbytes, eng.backend
    rtl = RtlBackend(eng.cfg, [s.dram[:n] for s in isa.machine.slices],
                     uarch=rtlsim.BOARD_UARCH, axi=True, boot=True)
    want = eng.step(toks[-1])
    eng.backend, eng.pos = rtl, eng.pos - 1
    got = eng.step(toks[-1])
    assert np.array_equal(want.view(np.uint32), got.view(np.uint32))
    assert np.array_equal(isa.machine.slices[0].dram[:n], rtl.drams[0][:n])


@pytest.mark.parametrize("resident", [False, True], ids=["per-position", "resident"])
def test_ple_on_host_is_bit_exact(tiny, resident, monkeypatch):
    """The PLE table on the host (docs/gemma4_e4b.md): the image holds a slot that the Engine
    writes the run's records into (Image.host_rows, from the host's store) before each prefill
    run and step, and the gathers read it at the row. Logits and caches equal those of the
    table on the card, bit for bit."""
    _, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(7).integers(0, 1000, 36)]
    engs = []
    for host in ("0", "1"):
        monkeypatch.setenv("OTPU_PLE_HOST", host)
        engs.append(Engine(spec, W, cap=1024, cfg=_cfg(), wformat="fp4", head_format="int8",
                           resident=resident))
    a, b = engs
    assert b.image.ple_host and not a.image.ple_host and b.resident == resident
    assert b.image.ple_store.shape == (spec.vocab, b.image.ple_rec)
    assert b.image.nbytes < a.image.nbytes - (spec.vocab - b.image.rows) * b.image.ple_rec + 4096
    assert np.array_equal(a.prefill(toks[:30], chunk=4), b.prefill(toks[:30], chunk=4))
    for t in toks[30:]:
        assert np.array_equal(a.step(t), b.step(t))
    lo, hi = a.image.layer0, a.image.head[0]
    assert (lo, hi) == (b.image.layer0, b.image.head[0])
    assert np.array_equal(a.backend.machine.slices[0].dram[lo:hi],
                          b.backend.machine.slices[0].dram[lo:hi])


@pytest.mark.parametrize("split", [None, True], ids=["one", "split"])
def test_generate_with_ple_on_host(tiny, split, monkeypatch):
    """The card's generate loop with the PLE table on the host (docs/gemma4_e4b.md): after
    sampling, each token's id goes to the image's PLE mailbox (offload's format), the host's
    RowServer (here the ISA simulator's WAITW hook; on the card the backend's poll) writes
    its record into slot row 0, and the next token's step waits for that (WAITW served >= seq)
    before its gather. The tokens equal those of the table on the card, greedy across the
    bucket boundary at 256 (one program, or split) and sampled, and a host-written step after
    the loop (its last request served first) gives the same logits."""
    from opentpu.llm import generate as GEN
    _, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(1).integers(0, 1000, 248)]
    engs = []
    for host in ("0", "1"):
        monkeypatch.setenv("OTPU_PLE_HOST", host)
        e = Engine(spec, W, cap=1024, cfg=_cfg(), wformat="fp4", head_format="int8",
                   resident=True)
        e.gen_split = split
        engs.append(e)
    a, b = engs
    assert a.row_server is None and b.row_server is not None and b.can_generate
    b.row_server.history = []
    t0 = int(np.argmax(a.prefill(toks)))
    assert int(np.argmax(b.prefill(toks))) == t0
    ref = a.generate_card(t0, 12, stop_ids=[])
    assert b.generate_card(t0, 12, stop_ids=[]) == ref and len(set(ref)) > 3
    h = b.row_server.history
    assert len(h) >= 11 and h == ref[:len(h)]           # each sampled token, in order
    samp = GEN.Sampling(0.8, 5, 0.9, 1.1)
    ctx = toks + [t0] + ref
    got = [e.generate_card(ref[-1], 8, stop_ids=[], sampling=samp, context=ctx,
                           rng=np.random.default_rng(7)) for e in (a, b)]
    assert got[0] == got[1]
    assert np.array_equal(a.step(got[0][-1]), b.step(got[1][-1]))
    assert b.row_server.seq == len(ref) + len(got[1])


def test_ple_rows_served_during_a_live_cards_runs(tiny, monkeypatch):
    """The host's side as it runs beside a card: BoardBackend polls the RowServer while a run is
    in flight, over a fake card that computes in a thread on the host's DRAM
    (tests/test_lfm2_moe.py's _LiveCard), so the generate loop's PLE fences wait for the host's
    writes during the run (here the host is the faster: they hold at their first read).
    Prefill, resident steps and the generate loop give the ISA simulator's logits and tokens
    bit for bit."""
    from test_lfm2_moe import _LiveCard
    from opentpu.host.board import BoardBackend
    _, W, spec = tiny
    monkeypatch.setenv("OTPU_PLE_HOST", "1")
    cfg = _cfg()
    card = _LiveCard.make(cfg)
    kw = dict(cap=1024, cfg=cfg, wformat="fp4", head_format="int8", resident=True)
    isa = Engine(spec, W, **kw)
    brd = Engine(spec, W, **kw, backend=lambda c, imgs: BoardBackend(c, imgs, transport=card))
    assert brd.can_generate and brd.backend.host is not None and brd.image.ple_host
    toks = [int(t) for t in np.random.default_rng(9).integers(0, 1000, 6)]
    a, b = isa.prefill(toks), brd.prefill(toks)
    assert card.error is None, card.error
    assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
    t0 = int(np.argmax(a))
    for e in (isa, brd):
        e.step(t0)
    got = brd.generate_card(t0, 8, stop_ids=[])
    assert card.error is None, card.error
    assert got == isa.generate_card(t0, 8, stop_ids=[])
    assert brd.row_server.seq >= 7 and isa.row_server.seq >= 7    # served during the run:
                                                                  # else a fence never holds


def test_one_sequence_one_slice(tiny):
    _, W, spec = tiny
    with pytest.raises(ValueError, match="one sequence"):
        Engine(spec, W, cap=256, cfg=_cfg(), batch=2)
    from opentpu.isasim import design_config
    with pytest.raises(ValueError, match="one slice"):
        spec.image(design_config(), 256)
    # whole attention blocks only (the sliding ring and the global layers' masked block):
    # a capacity of 384 (a multiple of D) is refused, not run past the cache
    with pytest.raises(ValueError, match="multiple of the attention block"):
        spec.image(_cfg(), 384)


def test_named_mix(tiny, monkeypatch):
    """wformat "mix": int8 with Spec.mix (formats.named), in the image, emulated_logits and an
    Engine (its logits those of int8 with the mix as OTPU_FORMATS)."""
    monkeypatch.delenv("OTPU_FORMATS", raising=False)
    _, W, spec = tiny
    spec = replace(spec, mix="attn@6-8=fp4,mlp@6-8=fp4")
    img = spec.image(_cfg(), 1024, wformat="mix")
    assert img.wformat == "int8" and img.formats == spec.mix and img.head_format == "int8"
    assert img.lf[5] == ("int8",) * 4 and img.lf[6] == ("fp4", "fp4", "fp4", "int8")
    toks = [5, 6, 7, 8, 9]
    assert np.array_equal(G.emulated_logits(spec, W, toks, wformat="mix"),
                          G.emulated_logits(spec, W, toks, formats=spec.mix))
    a = Engine(spec, W, cap=1024, cfg=_cfg(), wformat="mix", resident=True)
    monkeypatch.setenv("OTPU_FORMATS", spec.mix)
    b = Engine(spec, W, cap=1024, cfg=_cfg(), resident=True)
    assert a.image.formats == b.image.formats == spec.mix
    assert np.array_equal(a.prefill(toks), b.prefill(toks))
    assert np.array_equal(a.step(10), b.step(10))


def _tool(name):
    import importlib.util
    s = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent.parent /
                                               "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(s)
    s.loader.exec_module(mod)
    return mod


def test_formats_scan(tiny, tmp_path, monkeypatch):
    """tools/formats_scan.py's Gemma 4 rows (gemma4_quant_eval.emulate's head inputs, then one
    pass over the soft-capped head in chunks): each variant's NLL, argmax and KL divergence
    from float's distribution are emulate's logits' under the formats as layer_formats
    resolves them, and within rounding ties of emulated_logits' (ranged rules, the PLE
    projection, the head); a second run reads the head inputs from the cache. A token reads
    K / V only in the layers with their own; the groups halve the own and the shared layers."""
    FS, E = _tool("formats_scan"), _tool("gemma4_quant_eval")
    _, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(4).integers(0, 1000, 40)]
    specs = {"float": None, "int8": ("", "int8", None),
             "mix": ("attn@0-3=fp4,down@6-8=fp4,ple=fp4", "int8", None),
             "head": ("gateup@4-6=fp4,head=fp4", "int8", None)}
    rows = FS._g4_run(spec, W, toks, specs, 128, None, tmp_path, head_rows=256)

    def logp(lg):
        lg = 30.0 * np.tanh(lg / 30.0)
        m = lg.max(1, keepdims=True)
        return lg - m - np.log(np.exp(lg - m).sum(1, keepdims=True))

    lf = logp(E.emulate(spec, W, toks, wformat="none"))
    tgt = np.arange(len(toks) - 1), toks[1:]
    for lab, v in specs.items():
        r = rows[lab]
        wf, hf, wmap, pf = FS._g4_args(spec, v)
        assert pf == r["ple_table"] == ("none" if v is None else "int8")    # (the tiny fits)
        lq = logp(E.emulate(spec, W, toks, wformat=wf, hf=hf, ple_format=pf, wmap=wmap))
        assert np.abs(np.asarray(r["nll_tok"]) + lq[:-1][tgt]).max() < 1e-4
        assert r["top1"] == lq.argmax(1).tolist()
        kl = (np.exp(lf) * (lf - lq)).sum(1)
        assert np.abs(np.asarray(r["kl_tok"]) - kl).max() < 1e-6
        if v is not None:
            lg = logp(G.emulated_logits(spec, W, toks, wformat=v[1], head_format=v[2],
                                        formats=v[0]))
            assert abs(r["nll"] + lg[:-1][tgt].mean()) < 1e-2 * r["nll"], lab
            assert np.mean(lg.argmax(1) == lq.argmax(1)) > 0.9, lab
    assert FS._g4_args(spec, specs["head"])[1] == "fp4"
    assert FS._g4_args(spec, specs["mix"])[2]["ple"] == "fp4"
    assert len(list(tmp_path.glob("*.npy"))) == 4
    monkeypatch.setattr(FS, "_g4q", lambda: None)               # (no emulate: the cache)
    assert FS._g4_run(spec, W, toks, specs, 128, None, tmp_path, head_rows=256) == rows

    shapes = FS._g4_shapes(spec, W)
    kv = {i for i in range(spec.layers) if f"model.layers.{i}.self_attn.k_proj.weight" in shapes}
    assert kv == set(range(6))
    by = {lab: FS._g4_bytes(spec, shapes, v) for lab, v in specs.items() if v is not None}
    n8 = sum(r * c * 33 // 32 for r, c in shapes.values()) + 1000 * 256 * 33 // 32
    assert by["int8"] == n8 > by["mix"] and by["int8"] > by["head"]
    assert FS._g4_spans(spec) == [(0, 2), (3, 5), (6, 8)]
    e2b = replace(spec, kinds=((G.SLIDE,) * 4 + (G.FULL,)) * 7,
                  kv_src=tuple(range(15)) + (13, 14, 13, 13, 14) * 4, ffn=(512,) * 35)
    assert FS._g4_spans(e2b) == [(0, 9), (10, 14), (15, 24), (25, 34)]


@pytest.mark.skipif(not (REAL / "config.json").exists(), reason="no models/gemma-4-E2B")
def test_real_model_programs_fit():
    """Gemma 4 E2B on the board: fp4 layers, int8 LM head and PLE fit 4 GiB at 4096 tokens;
    the resident decode programs fit IMEM and take 6 argument words. Its recommended mix too."""
    spec = G.Spec.from_hf(REAL)
    img = spec.image(board_config(), 4096, 1, 8, "fp4", "int8", lookup=True)
    assert img.ple_format == "int8" and img.nbytes < 1 << 32
    assert spec.image(board_config(), 4096, 1, 8, lookup=True).formats == ""   # int8 fits
    for blocks in (1, 3, 16):
        progs, ra = img.compile_decode(blocks, (blocks - 1) * 256)
        assert 8 * len(progs[0]) <= board_config().IMEM_WORDS
        assert len(ra) == 6
    with pytest.raises(CompileError, match="TMEM"):
        img.compile_rows([(0, p) for p in range(8)], [7])
    # the recommended mix (docs/formats.md): the shared layers' MLP and layers 15-24's attention
    # in fp4, the int8 PLE table on the card beside them; three runs (the splits at 15 and 25)
    mix = spec.image(board_config(), 4096, 1, 8, "mix", lookup=True)
    assert spec.mix == "attn@15-24=fp4,mlp@15-34=fp4" and mix.formats == spec.mix
    assert (mix.ple_format, mix.ple_host, mix.head_format) == ("int8", False, "int8")
    assert len(img.runs) == 2 and len(mix.runs) == 3 and mix.nbytes < 1 << 32
    progs, _ = mix.compile_decode(16, 15 * 256)
    assert 8 * len(progs[0]) <= board_config().IMEM_WORDS


@pytest.mark.skipif(not (REAL_E4B / "config.json").exists(), reason="no models/gemma-4-E4B")
def test_e4b_programs_fit():
    """Gemma 4 E4B on the board: with fp4 layers and the int8 head its PLE table does not fit
    beside them in either format, so the image keeps it on the host (int8 records) and fits
    4 GiB at 2048 tokens; the resident decode programs fit IMEM with 5 argument words (the PLE
    gather reads the slot, not the token's row); 4-row prefill runs fit TMEM."""
    spec = G.Spec.from_hf(REAL_E4B)
    img = spec.image(board_config(), 2048, 1, 8, "fp4", "int8", lookup=True)
    assert img.ple_host and img.ple_format == "int8" and img.nbytes < 3 << 30
    for blocks in (1, 8):
        progs, ra = img.compile_decode(blocks, (blocks - 1) * img.block)
        assert 8 * len(progs[0]) <= board_config().IMEM_WORDS
        assert len(ra) == 5
    img.compile_rows([(0, r) for r in range(4)], [3], img.block, tokens=[5, 6, 7, 8])
    # int8 layers do not fit: the formats by fit, two loops, decode and 4-row runs fit
    mix = spec.image(board_config(), 2048, 1, 8, lookup=True)
    assert mix.formats == "head=fp4,down@0-23=fp4" and mix.head_format == "fp4"
    assert mix.ple_host and 3.9 * 2**30 < mix.nbytes < 1 << 32 and len(mix.runs) == 2
    progs, ra = mix.compile_decode(8, 7 * mix.block)
    assert 8 * len(progs[0]) <= board_config().IMEM_WORDS
    mix.compile_rows([(0, r) for r in range(4)], [3], mix.block, tokens=[5, 6, 7, 8])
