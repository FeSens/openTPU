"""LFM2-MoE on openTPU with its experts streamed into DRAM slots (docs/offload.md, path (a)):
the card routes and computes every expert, the host's server moves the missing ones. A tiny
random model against Hugging Face transformers and the quantized emulation; a small cache
gives the same logits bit for bit as one holding every expert."""
import dataclasses

import numpy as np
import pytest

from opentpu.llm.lfm2 import Spec, emulated_logits, reference_logits
from opentpu.llm.qwen3 import Engine, device_config

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

KINDS = ("conv", "attn", "conv", "attn", "conv")
E, K, F = 8, 2, 256


def _cos(a, b):
    return (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)


def _tiny(F=F):
    torch.manual_seed(0)
    hc = transformers.Lfm2MoeConfig(
        hidden_size=256, num_hidden_layers=len(KINDS), num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=512, moe_intermediate_size=F, num_experts=E,
        num_experts_per_tok=K, num_dense_layers=1, vocab_size=1000, norm_eps=1e-5,
        layer_types=["full_attention" if k == "attn" else "conv" for k in KINDS],
        conv_L_cache=3, conv_bias=False, tie_word_embeddings=True,
        max_position_embeddings=4096, rope_parameters={"rope_type": "default", "rope_theta": 1e6})
    m = transformers.Lfm2MoeForCausalLM(hc).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            elif "experts" in n or "gate" in n:
                p.copy_(0.05 * torch.randn_like(p))
        for mod in m.modules():
            if hasattr(mod, "expert_bias"):
                mod.expert_bias.copy_(0.02 * torch.randn_like(mod.expert_bias))
    W = {}
    for k, v in m.state_dict().items():
        v = v.float().numpy()
        if k.endswith("experts.gate_up_proj"):          # the checkpoint's per-expert names
            p = k[:-len("gate_up_proj")]
            for e in range(E):
                W[f"{p}{e}.w1.weight"], W[f"{p}{e}.w3.weight"] = v[e, :F], v[e, F:]
        elif k.endswith("experts.down_proj"):
            p = k[:-len("down_proj")]
            for e in range(E):
                W[f"{p}{e}.w2.weight"] = v[e]
        else:
            W[k] = v
    from opentpu.llm.moe import MoESpec
    spec = Spec(256, KINDS, 4, 2, 64, 512, 1000, moe=MoESpec(E=E, k=K, ffn=F, first=1))
    return m, W, spec


@pytest.fixture(scope="module")
def tiny():
    return _tiny()


def test_from_hf_gives_a_moe_the_int8_embedding(tiny, tmp_path):
    """A LFM2-MoE checkpoint's Spec gathers its embedding rows on the device in int8 at any
    vocabulary size (its DRAM beside the layers is expert slots)."""
    tiny[0].config.save_pretrained(tmp_path)
    s = Spec.from_hf(tmp_path)
    assert s.moe is not None and s.embed == "int8"


def _engine(spec, W, experts=None, **kw):
    cfg = device_config(spec, 256, S=1, experts=experts, **kw)
    return Engine(spec, W, cap=256, cfg=cfg, experts=experts, **kw)


def _untied(spec, W):
    """The tiny model with an LM head of its own: its int8 embedding table is the image's own
    (a tied int8 head's rows are gathered from the head)."""
    W = dict(W, **{"lm_head.weight": W["model.embed_tokens.weight"] * np.float32(1.25)})
    return dataclasses.replace(spec, tied=False, embed="int8"), W


def test_reference_matches_hf(tiny):
    m, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(0).integers(0, 1000, 24)]
    with torch.no_grad():
        hf = m(torch.tensor([toks])).logits[0].numpy()
    assert np.abs(reference_logits(spec, W, toks) - hf).max() < 1e-4


def test_device_follows_hf_and_routes_on_the_card(tiny):
    """The logits follow HF but for a token whose routing turns on a near-tie (the card's
    quantized router against bf16's); the card's choices in the first MoE layer, whose input
    no routing has touched, are the emulation's wherever its k-th and (k+1)-th scores are apart
    by more than the quantization's reach."""
    m, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(1).integers(0, 1000, 12)]
    with torch.no_grad():
        hf = m(torch.tensor([toks])).logits[0].numpy()
    eng = _engine(spec, W)
    eng.server.history = []
    dev = np.array([eng.step(t) for t in toks])
    c = _cos(dev, hf)
    assert np.median(c) > 0.997 and (c > 0.99).mean() >= 0.75
    routes = []
    emulated_logits(spec, W, toks, routes=routes)
    moe_layers = len(KINDS) - 1
    assert eng.server.seq == len(toks) * moe_layers - 1       # the last waits for a next fence
    card = eng.server.history
    checked = 0
    for t, layer, ids, margin in routes:
        if layer == 1 and margin > 1e-3 and t * moe_layers < len(card):
            assert card[t * moe_layers] == ids, (t, card[t * moe_layers], ids, margin)
            checked += 1
    assert checked >= len(toks) - 2
    assert eng.server.misses == 0


def test_padded_expert_width(tiny):
    """An expert width that is not a whole number of its format's chunks (2D for 4-bit: 384
    here, Gemma 4's 704) is padded in the slot with zero rows of W_gate / W_up and zero columns
    of W_down (moe.ExpertFormat; exact, act(0) * 0 = 0): the logits follow the 4-bit
    emulation of the unpadded model (as the 256-wide model's do: median cosine 0.989; a token
    whose routing turns on a near-tie differs), and a small cache gives the full cache's bit for
    bit."""
    _, W, spec = _tiny(F=384)
    toks = [int(t) for t in np.random.default_rng(1).integers(0, 1000, 10)]
    full = _engine(spec, W, wformat="fp4")
    assert (full.image.fmt.F0, full.image.fmt.F) == (384, 512)
    ref = np.array([full.step(t) for t in toks])
    assert np.median(_cos(ref, emulated_logits(spec, W, toks, wformat="fp4"))) > 0.99
    small = _engine(spec, W, experts=K, wformat="fp4")
    got = np.array([small.step(t) for t in toks])
    assert small.server.misses > len(toks)
    assert np.array_equal(got.view(np.uint32), ref.view(np.uint32))


def test_formats_by_layer_range_keep_one_dense_layout(tiny, monkeypatch):
    """Weight formats per layer range on a MoE model: the mixers' ranges give layer block
    layouts of their own, the MoE layers in a loop of theirs; the dense layers' MLPs (a region
    of their own, one MLP per dense layer) keep one layout, so an MLP range inside them is
    refused. Layout-only images, no weights."""
    from opentpu.isasim import board_config
    monkeypatch.delenv("OTPU_FORMATS", raising=False)
    _, _, spec = tiny
    cfg = board_config(DRAM_BYTES=1 << 24)
    img = spec.image(cfg, 256, formats="conv@1-4=fp4,attn@1-4=fp4")
    g0, g = img.lf[0], ("fp4", "fp4", "int8", "int8")
    assert img.lf[1:] == (g,) * 4 and img.layouts[g0].size != img.layouts[g].size
    assert img.plan == [(0, ((("conv", False), g0),), 1),
                        (1, ((("attn", True), g), (("conv", True), g)), 2)]
    two = dataclasses.replace(spec, moe=dataclasses.replace(spec.moe, first=2))
    assert two.image(cfg, 256, formats="mlp@0-1=fp4").mf["wd"] == "fp4"
    with pytest.raises(ValueError, match="weight formats"):
        two.image(cfg, 256, formats="mlp@0=fp4")


def test_moe_ffn_beside_and_no_residual(tiny, monkeypatch):
    """moe_ffn's `beside` (work emitted right after the request is posted: Gemma 4's dense MLP,
    while the host streams), residual=False (the caller adds x) and y_first (the experts'
    outputs placed first in TMEM): with work beside every MoE layer, decode on the ISA simulator and on the live fake card (misses served during the
    runs) gives the plain programs' logits bit for bit."""
    from opentpu import language as ol
    from opentpu.host.board import BoardBackend
    from opentpu.isasim import board_config
    from opentpu.llm import moe as MO
    _, W, spec = tiny
    cfg = board_config(DRAM_BYTES=1 << 24)
    toks = [int(t) for t in np.random.default_rng(6).integers(0, 1000, 6)]
    plain = Engine(spec, W, cap=256, cfg=cfg, rows=1, resident=True, experts=K)
    ref = np.array([plain.step(t) for t in toks])
    real, n = MO.moe_ffn, [0]

    def moe_ffn(x, lw, mo, dev, eps):
        def beside():                       # VPU work the result does not use
            n[0] += 1
            t = ol.empty((1, x.cols))
            t.set(x * 2.0 + 1.0)
        return x + real(x, lw, mo, dev, eps, beside=beside, residual=False, y_first=True)
    monkeypatch.setattr(MO, "moe_ffn", moe_ffn)
    card = _LiveCard.make(cfg)
    for backend in ("isa", lambda c, imgs: BoardBackend(c, imgs, transport=card)):
        eng = Engine(spec, W, cap=256, cfg=cfg, rows=1, resident=True, experts=K,
                     backend=backend, pipeline=False)
        got = np.array([eng.step(t) for t in toks])
        assert card.error is None, card.error
        assert np.array_equal(got.view(np.uint32), ref.view(np.uint32))
        assert eng.server.misses > len(toks)
    assert n[0] > 0 and card.waits > 0


@pytest.mark.parametrize("wformat", ["int8", "fp4"])
def test_small_cache_is_bit_exact(tiny, wformat):
    """k slots per layer: nearly every request misses and streams; the logits do not move."""
    _, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(2).integers(0, 1000, 10)]
    full = _engine(spec, W, wformat=wformat)
    ref = np.array([full.step(t) for t in toks])
    small = _engine(spec, W, experts=K, wformat=wformat)
    got = np.array([small.step(t) for t in toks])
    assert small.server.misses > len(toks)
    assert np.array_equal(got.view(np.uint32), ref.view(np.uint32))


def test_experts_run_paired(tiny):
    """With column reuse (PAIR), a 4-bit expert's MMs run paired as the layers' own do: its slot
    address is a register (DevVar), but every slot is LINE-aligned, so its scale words pair
    8-byte aligned. On the board configuration that halves an expert's cycles (26B-A4B: 0.38
    -> 0.21 ms, co-simulated). A small cache still gives the logits of one holding every
    expert, bit for bit."""
    from opentpu import isa as I
    _, W, spec = tiny

    def paired(experts=None):
        cfg = device_config(spec, 256, S=1, experts=experts, wformat="fp4", PAIR=True)
        return Engine(spec, W, cap=256, cfg=cfg, experts=experts, wformat="fp4")
    full = paired()
    prog = full.image.compile_step(3)[0]
    ex = [i for i in prog if i.op == I.MM and any(fn == "expert" for _, _, fn in i.src)]
    assert ex and all(i.flags & I.F_PAIR for i in ex)
    toks = [int(t) for t in np.random.default_rng(5).integers(0, 1000, 8)]
    ref = np.array([full.step(t) for t in toks])
    small = paired(K)
    got = np.array([small.step(t) for t in toks])
    assert small.server.misses > len(toks)
    assert np.array_equal(got.view(np.uint32), ref.view(np.uint32))


def test_lfu_policy_is_bit_exact(tiny):
    """The slots replaced by least decayed use (ExpertServer policy "lfu", moe.serve's default)
    instead of LRU: other misses, the same logits bit for bit."""
    _, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(7).integers(0, 1000, 12)]
    full = _engine(spec, W)
    ref = np.array([full.step(t) for t in toks])
    got = {}
    for policy in ("lru", "lfu"):
        eng = _engine(spec, W, experts=4)
        assert eng.server.policy == "lfu"
        eng.server.policy = policy
        got[policy] = (np.array([eng.step(t) for t in toks]), eng.server.misses)
        assert np.array_equal(got[policy][0].view(np.uint32), ref.view(np.uint32))
    assert got["lfu"][1] > len(toks) and got["lfu"][1] != got["lru"][1]


def test_pool_file_is_the_same_pool(tiny, tmp_path):
    """The pool packed once into a file (the page cache or the SSD tier) serves the same bytes
    (read back with pread as they are packed); a second engine reuses the file, its warm
    thread reading the packed experts into the page cache."""
    _, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(4).integers(0, 1000, 3)]
    full = _engine(spec, W, experts=K)
    ref = np.array([full.step(t) for t in toks])
    f = tmp_path / "pool.bin"
    for i in range(2):
        eng = Engine(spec, W, cap=256, cfg=device_config(spec, 256, S=1, experts=K),
                     experts=K, pool_file=f)
        got = np.array([eng.step(t) for t in toks])
        assert np.array_equal(got.view(np.uint32), ref.view(np.uint32))
        warm = eng.server.pool_warm             # (the same tokens: no expert packed second)
        warm.join(timeout=60)
        packed = int(np.fromfile(str(f) + ".packed", np.uint8).sum())
        assert packed > 0 and warm.bytes == (packed if i else 0) * eng.image.offload.slot_bytes
    assert f.stat().st_size == (len(KINDS) - 1) * E * eng.image.offload.slot_bytes


def test_engine_releases_the_checkpoint_when_streaming_from_a_pool(tiny, tmp_path):
    """Engine with a pool file releases its LazyWeights once the image is written and the slots
    warm (release_weights, the default: the checkpoint's mapped pages would outlive the
    pool's in the page cache, docs/offload.md 10.6): its files closed, the same logits bit
    for bit, an expert packed later reading its file again; release_weights=False keeps
    them, and without a pool nothing is released."""
    from safetensors.numpy import save_file
    from opentpu.llm.qwen3 import LazyWeights
    _, W, spec = tiny
    save_file({k: np.ascontiguousarray(v) for k, v in W.items()},
              str(tmp_path / "model.safetensors"))
    toks = [int(t) for t in np.random.default_rng(4).integers(0, 1000, 3)]
    full = _engine(spec, W, experts=K)
    ref = np.array([full.step(t) for t in toks])
    cfg = device_config(spec, 256, S=1, experts=K)
    for i, keep in enumerate((False, True)):
        lw = LazyWeights(tmp_path)
        eng = Engine(spec, lw, cap=256, cfg=cfg, experts=K, pool_file=tmp_path / f"p{i}.bin",
                     release_weights=not keep)
        assert bool(lw._h) == keep
        got = np.array([eng.step(t) for t in toks])
        assert np.array_equal(got.view(np.uint32), ref.view(np.uint32))
        assert len(lw._h) == 1                          # (the experts packed since: reopened)
    lw = LazyWeights(tmp_path)
    Engine(spec, lw, cap=256, cfg=cfg, experts=K)
    assert lw._h


def test_compile_worker_builds_the_moe_image(tiny):
    """The compile worker process (the card's backend compiles ahead in it) builds the engine's
    image, K expert slots per layer included: with E it would be over the DRAM that
    device_config sized for K (the 8B's first card run: 4568 MiB for 4096)."""
    from opentpu import isa as I
    from opentpu.llm import qwen3 as Q
    _, W, spec = tiny
    cfg = device_config(spec, 256, rows=1, lookup=True, S=1, experts=K)
    eng = Engine(spec, W, cap=256, cfg=cfg, rows=1, resident=True, experts=K)
    Q._worker_init(spec, eng.cfg, eng.cap, eng.batch, eng.rows, eng.block, eng._image_kw)
    try:
        img = Q._WORKER[0]
        assert img.nbytes == eng.image.nbytes and img.offload == eng.image.offload
        words, _ = Q._worker_decode(1, 2)               # (from the first run-time position)
        assert np.array_equal(words, I.assemble(eng.image.compile_decode(1, 2)[0][0]))
    finally:
        Q._WORKER = None


@pytest.mark.parametrize("embed", ["f32", "int8", "host"])
def test_the_card_generates_with_streamed_experts(tiny, embed):
    """The decode loop on the card (autodecode's generate program, resident decode) with k
    slots per layer: the experts stream between the tokens it picks, and it gives the host's
    resident loop token for token (int8: the embedding row gathered from the tied head; host:
    an int8 table of its own kept on the host, embed_host, a MoE's default: each sampled
    token's row asked of the host's row server, against the table on the card)."""
    _, W, spec = tiny
    if embed == "host":
        spec, W = _untied(spec, W)
    else:
        spec = dataclasses.replace(spec, embed=embed)
    cfg = device_config(spec, 512, rows=1, lookup=True, S=1, experts=K, embed_host=False)
    a, b = (Engine(spec, W, cap=512, cfg=cfg, rows=1, resident=True, experts=K, **kw)
            for kw in ({}, {"embed_host": False}))
    assert a.can_generate and a.image.embed_host == (embed == "host") and not b.image.embed_host
    assert (a.row_server is None) == (embed != "host")
    toks = [int(t) for t in np.random.default_rng(3).integers(0, 1000, 20)]
    t0 = int(np.argmax(a.prefill(toks)))
    assert int(np.argmax(b.prefill(toks))) == t0
    ref, t = [], t0
    for _ in range(12):
        t = int(np.argmax(b.step(t)))
        ref.append(t)
    misses = a.server.misses
    assert a.generate_card(t0, 12, stop_ids=[]) == ref
    assert a.server.misses > misses
    if embed == "host":                     # each sampled token's row, from the host
        assert a.row_server.seq >= 11 and b.image.nbytes - a.image.nbytes >= 1000 * 256


def test_moe_on_board_model(tiny, have_verilator):
    """The MoE block on the RTL (WAITW: CAPS bit31) through the host driver: every expert
    resident and `served` past any request, so each layer's fence and directory waits hold at
    their first read and no host serves (the board model runs a script, with no host during a
    run). Per-position programs (positions 0, 1), then the resident decode, then the card's
    generate loop: the ISA simulator's logits and tokens bit for bit."""
    from opentpu.host.board import Board, BoardBackend, SimTransport
    from opentpu.isasim import board_config
    _, W, spec = tiny
    cfg = board_config(DRAM_BYTES=1 << 24)
    tr = SimTransport(ch_bytes=cfg.DRAM_BYTES // 2, stall=20, seed=5)
    if not Board(tr).info()["caps"].get("waitw"):
        pytest.skip("the RTL has no WAITW (CAPS bit31)")
    isa = Engine(spec, W, cap=256, cfg=cfg, rows=1, resident=True)
    brd = Engine(spec, W, cap=256, cfg=cfg, rows=1, resident=True,
                 backend=lambda c, imgs: BoardBackend(c, imgs, transport=tr))
    assert brd.resident and brd.can_generate and brd.server.misses == 0
    L = brd.image.offload
    brd.backend.write(0, L.served, np.array([3e38], np.float32))
    for tok in (11, 222, 333, 444):
        a, b = isa.step(tok), brd.step(tok)
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32)), brd.pos
    t0 = int(np.argmax(a))
    seq = int(brd.backend.read(0, L.mbox, 4).view(np.float32)[0])
    assert seq == 4 * len(L.slots)             # every MoE layer posted, every token
    assert brd.generate_card(t0, 4, stop_ids=[]) == isa.generate_card(t0, 4, stop_ids=[])
    assert isa.server.misses == 0 and brd.stats[-1]["cycles"] > 0
    progs = brd.backend.last[0]     # the generate program: per MoE block its fence and the
    assert sum(i.op == 0x07 for i in progs[0]) >= 3     # hits' and misses' directory waits


class _LiveCard:
    """A fake card that computes while the host works (built on opentpu.host.fake): LOAD takes
    the program from DRAM, RUN runs it with ARG0..7 on the ISA simulator in a thread, over one
    DRAM the host's reads and writes reach at once (the two channels interleaved in 64-byte
    beats, as the card's; with `chash` CHASH's map, chunk m's beat b on channel b ^ parity(m)).
    A MoE layer's WAITW then waits for the host's writes during the run, as on the card;
    HALTED rises when the program halts."""

    @staticmethod
    def make(cfg, chash=False):
        import threading
        import time
        from opentpu import isa as I
        from opentpu.host import regs as R
        from opentpu.host.board import swapped
        from opentpu.host.fake import FakeTransport
        from opentpu.isasim import Machine

        class Live(FakeTransport):
            def __init__(self):
                super().__init__(ch_bytes=128, devname=None, args=True, gen=True)
                self.mem = np.zeros((cfg.DRAM_BYTES // 128, 2, 64), np.uint8)
                self.flat = self.mem.reshape(-1)
                self.prog, self.thread, self.error, self.waits = None, None, None, 0

            def _beats(self, ch, b0, b1):
                """Channel ch's beats b0..b1: (chunks, halves) of mem."""
                m = np.arange(b0, b1)
                return m, ch ^ swapped(128 * b0, b1 - b0).astype(np.intp) if chash else ch

            def mem_write(self, ch, off, data):
                b0, b1 = off // 64, -(-(off + len(data)) // 64)
                m, h = self._beats(ch, b0, b1)
                rows = self.mem[m, h].reshape(-1)
                rows[off - 64 * b0:off - 64 * b0 + len(data)] = data
                self.mem[m, h] = rows.reshape(-1, 64)

            def mem_read(self, ch, off, n, out=None):
                b0, b1 = off // 64, -(-(off + n) // 64)
                m, h = self._beats(ch, b0, b1)
                r = self.mem[m, h].reshape(-1)[off - 64 * b0:off - 64 * b0 + n]
                if out is None:
                    return r.copy()
                out[:] = r
                return out

            def _host(self, m):
                """WAITW: every slice polls; wait for the host's write (a 60 s timeout)."""
                self.waits += 1
                t0 = time.perf_counter()
                while time.perf_counter() - t0 < 60:
                    for s in m.slices:
                        ins = s.polling
                        a = (s.reg(ins.ra) + ins.w[0]) & 0xFFFFFFFF
                        if I.waitw_holds(int(s.m32[a // 4]), s.reg(ins.rc) + ins.w[2],
                                         ins.flags & 3, ins.w[3]):
                            return
                    time.sleep(1e-4)

            def _run(self, m):
                try:
                    m.run(max_steps=1 << 40)
                except Exception as e:          # noqa: BLE001 (reported by the test)
                    self.error = e

            def reg_write(self, off, val):
                ctrl = self.regs[R.R_CTRL]
                super().reg_write(off, val)
                if off != R.R_CTRL:
                    return
                if val & R.CTRL_LOAD:
                    a, n = self.regs[R.R_PROG_ADDR], self.regs[R.R_PROG_N]
                    w = self.flat[a:a + 32 * n].view(np.uint32).reshape(n, 8)
                    self.prog = [I.Instr.decode(x) for x in w]
                if val & R.CTRL_RUN and not ctrl & R.CTRL_RUN:
                    args = [self.regs.get(R.R_ARG0 + 4 * k, 0) for k in range(8)]
                    m = Machine(cfg, [self.prog], [None], args)
                    m.slices[0].dram = self.flat        # the host's DRAM, not a copy
                    m.host = self._host
                    self.thread = threading.Thread(target=self._run, args=(m,), daemon=True)
                    self.thread.start()

            def reg_read(self, off):
                if off == R.R_CAPS:
                    return super().reg_read(off) | (R.CAP_CHASH if chash else 0)
                if off == R.R_STATUS:
                    run = bool(self.regs[R.R_CTRL] & R.CTRL_RUN)
                    done = run and self.thread is not None and not self.thread.is_alive()
                    return (R.ST_HALTED if done else 0) | R.ST_CALIB0 | R.ST_CALIB1 | \
                        R.ST_WR_IDLE | (R.ST_RUN if run else 0)
                return super().reg_read(off)

        return Live()


@pytest.mark.parametrize("mode", ["sync", "threaded", "split"])
def test_the_host_serves_the_card_during_its_runs(tiny, mode, tmp_path):
    """The card's side of path (a) with the host's server as it runs beside a card: the backend
    polls it while a run is in flight (BoardBackend.host), and it moves the missing experts
    into the slots while the card's MoE layers wait for them (a fake card that computes in a
    thread over the host's DRAM). k slots per layer: prefill, per-position and resident
    decode, and the card's generate loop give the ISA simulator's logits and tokens bit for
    bit, with misses served during the runs. threaded: a transport that DMAs from a worker
    thread (XdmaTransport's), so the server's memory is BoardDram (its DMA thread writing while
    the card computes). split: also CHASH's map, and the experts from a pool file in the split
    format, read straight into the channel runs at the page-aligned slots (every other slot
    here: the others take the slot's bytes)."""
    from opentpu.host.board import BoardBackend
    from opentpu.host.offload import BackendDram, BoardDram
    from opentpu.isasim import board_config
    _, W, spec = tiny
    cfg = board_config(DRAM_BYTES=1 << 24)
    card = _LiveCard.make(cfg, chash=mode == "split")
    card.threaded = threaded = mode != "sync"
    isa = Engine(spec, W, cap=256, cfg=cfg, rows=1, resident=True, experts=K)
    brd = Engine(spec, W, cap=256, cfg=cfg, rows=1, resident=True, experts=K,
                 backend=lambda c, imgs: BoardBackend(c, imgs, transport=card),
                 pool_file=tmp_path / "pool.bin" if mode == "split" else None)
    assert brd.backend.board.chash == (mode == "split")
    assert brd.resident and brd.can_generate and brd.backend.host is not None
    assert not brd.stream_logits
    assert isinstance(brd.server.mem, BoardDram if threaded else BackendDram)
    toks = [int(t) for t in np.random.default_rng(5).integers(0, 1000, 6)]
    for tok in toks:
        a, b = isa.step(tok), brd.step(tok)
        assert card.error is None, card.error
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32)), brd.pos
    assert brd.server.misses > len(toks) and card.waits > 0
    t0, misses, waits = int(np.argmax(a)), brd.server.misses, card.waits
    got = brd.generate_card(t0, 8, stop_ids=[])
    assert card.error is None, card.error
    assert got == isa.generate_card(t0, 8, stop_ids=[])
    assert brd.server.misses > misses and card.waits > waits      # served during the loop
    assert brd.server.seq == isa.server.seq == (len(toks) + 8) * len(brd.image.offload.slots)
    if threaded:                            # the experts went the one-pass way, not Board.write
        assert brd.server.mem._bufs is not None
    if mode == "split":                     # and read into the runs (the split format):
        # every one (each slot on a RUN block, Layout's pitch; direct counts the load's too)
        assert brd.server.mem.direct >= brd.server.misses > 0 and brd.server.pool_warm


@pytest.mark.parametrize("embed", ["f32", "int8"])
def test_lazy_weights_build_the_same_engine(tiny, tmp_path, embed):
    """Weights read tensor by tensor from safetensors files (qwen3.LazyWeights: a model whose
    fp32 weights do not fit host RAM; the image the simulator's DRAM without a copy) give the
    in-RAM weights' logits bit for bit, the host reading no embedding row (device inputs)."""
    from safetensors.numpy import save_file
    from opentpu.llm.qwen3 import LazyWeights
    _, W, spec = tiny
    save_file({k: np.ascontiguousarray(v) for k, v in W.items()},
              str(tmp_path / "model.safetensors"))
    spec = dataclasses.replace(spec, embed=embed)
    cfg = device_config(spec, 256, rows=1, lookup=True, S=1, experts=K)
    a, b = (Engine(spec, w, cap=256, cfg=cfg, rows=1, resident=True, experts=K)
            for w in (W, LazyWeights(tmp_path)))
    for t in (5, 77, 900, 13, 4):
        assert np.array_equal(a.step(t).view(np.uint32), b.step(t).view(np.uint32)), a.pos


def test_lazy_weights_release_closes_and_reopens(tiny, tmp_path):
    """LazyWeights.release (moe_card --release-weights): the files closed and their pages
    dropped; a tensor read after reopens its file and reads the same values."""
    from safetensors.numpy import save_file
    from opentpu.llm.qwen3 import LazyWeights
    _, W, _ = tiny
    save_file({k: np.ascontiguousarray(v) for k, v in W.items()},
              str(tmp_path / "model.safetensors"))
    lw = LazyWeights(tmp_path)
    k = "model.embed_tokens.weight"
    a = lw[k]
    lw.release()
    assert not lw._h
    assert np.array_equal(lw[k], a) and np.array_equal(lw.part(k, 3), a[3])
    assert len(lw._h) == 1
