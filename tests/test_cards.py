"""Cards in tensor parallel (opentpu/host/cards.py): CardsTransport over two cards of two slices
(bcu1525-x2), each card's registers and DRAM its own, on the ISA simulator behind them (IsaCards):
the routing (a global slice's channels on its card), the registers together (STATUS, ICOUNT,
CFG_SLICES and the machine's CFG_HASH), the writes card 0 last, CARD_ID / NCARDS and the cable map
from the hellos, and a tiny Qwen3 at S = 4 through BoardBackend and the Engine bit for bit
against the ISA simulator's one machine (decode logits, the generate loop's tokens read from
card 0). X2SimTransport's script and build without the two-board testbench."""
import numpy as np
import pytest

from opentpu import target as T
from opentpu.host import regs as R
from opentpu.host.board import Board, BoardBackend, ConfigMismatch
from opentpu.host.cards import CardsTransport, IsaCards, LinkError, X2SimTransport

TARGET = "bcu1525-x2"


def _cfg(dram: int = 1 << 22):
    return T.config(T.load(TARGET), DRAM_BYTES=dram)


def test_target_cards():
    t = T.load(TARGET)
    assert T.cards(t) == 2 and t.arch["S"] == 4 and T.cards("bcu1525-s2") == 1
    assert T.cards(TARGET + ":st16") == 2


def test_cards_registers_and_routing():
    cfg = _cfg()
    sh = IsaCards(cfg)
    ct = CardsTransport(sh.cards)
    assert (ct.slices, ct.spc, ct.nch, ct.ncards) == (4, 2, 8, 2)
    # every card got its CARD_ID / NCARDS and the link came up; the cables straight
    assert [k.topo for k in sh.cards] == [0 | 2 << 8, 1 | 2 << 8]
    assert ct.map == {(0, 0): (1, 0), (0, 1): (1, 1), (1, 0): (0, 0), (1, 1): (0, 1)}
    # a card reports its own two slices, the transport the machine's four and its CFG_HASH
    assert sh.cards[1].reg_read(R.R_CFG_SLICES) == 2 and ct.reg_read(R.R_CFG_SLICES) == 4
    info = Board(ct, lock=False).info()
    assert info["cfg"]["S"] == 4 and info["target"].name == TARGET
    assert ct.reg_read(R.R_CFG_HASH) != sh.cards[0].reg_read(R.R_CFG_HASH)
    # global slice 3's channel 1 is card 1's local slice 1's
    ct.mem_write(8 * 3 + 1, 256, np.arange(128, dtype=np.uint8))
    assert np.array_equal(sh.cards[1].mem_read(8 + 1, 256, 128), np.arange(128, dtype=np.uint8))
    assert not sh.cards[0].mem_read(8 + 1, 256, 128).any()
    # Board's logical bytes on slice 2 land in card 1's local slice 0, in the machine's map
    b = Board(ct, lock=False)
    data = np.random.default_rng(1).integers(0, 256, 5000, dtype=np.uint8)
    b.write(4096 + 7, data, 2)
    assert np.array_equal(sh.dram[2][4096 + 7:4096 + 7 + 5000], data)
    assert np.array_equal(b.read(4096 + 7, 5000, 2), data)
    # writes reach card 1 before card 0
    seen = []
    for k in sh.cards:
        k.reg_write = (lambda f, c: lambda o, v: (seen.append(c), f(o, v)))(k.reg_write, k.c)
    ct.reg_write(R.R_SCRATCH, 5)
    assert seen == [1, 0] and ct.reg_read(R.R_SCRATCH) == 5
    sh.cards[1].regs[R.R_SCRATCH] = 6                   # (cards that disagree: 0)
    assert ct.reg_read(R.R_SCRATCH) == 0


def test_cards_status_together():
    from opentpu.host.cards import status
    a = R.ST_HALTED | R.ST_RUN | R.ST_WR_IDLE | R.ST_CALIB0 | R.ST_CALIB1
    assert status([a, a]) == a
    assert status([a, a & ~R.ST_HALTED]) == a & ~R.ST_HALTED
    assert status([a, a | R.ST_ERROR]) == a | R.ST_ERROR
    assert status([a | R.ST_LOADING, a]) & R.ST_LOADING


def test_cards_links():
    cfg = _cfg()
    sh = IsaCards(cfg, cross=True)                      # (QSFP0 to QSFP1: either way at two)
    ct = CardsTransport(sh.cards)
    assert ct.map[(0, 0)] == (1, 1) and ct.map[(1, 1)] == (0, 0)
    # another machine's NCARDS keeps both ports down, with the v1 hello layout
    sh.cards[1].topo = 1 | 3 << 8
    with pytest.raises(LinkError, match="NCARDS 1/3|of 3 cards"):
        ct.links()
    sh.cards[1].topo = 1 | 2 << 8
    # a hello naming this card (or an invalid port) cannot describe a cable
    peer = sh.cards[0].peer
    for hello in (R.X2_HELLO_MAGIC << 24 | 2 << 12, peer(0) | 2 << 8, peer(0) ^ 1 << 24):
        sh.cards[0].peer = lambda p, h=hello: h
        with pytest.raises(LinkError, match="not the ring|no hello"):
            ct.links()
    sh.cards[0].peer = peer


def test_cards_refuse_mixed_builds():
    cfg = _cfg()
    sh = IsaCards(cfg)
    sh.cards[1].build_id ^= 1
    with pytest.raises(ConfigMismatch, match="card 1's bitstream is not card 0's.*build_id"):
        CardsTransport(sh.cards)


def _tiny():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from opentpu.llm.qwen3 import Spec
    torch.manual_seed(0)
    hc = transformers.Qwen3Config(hidden_size=512, num_hidden_layers=2, num_attention_heads=4,
                                  num_key_value_heads=4, head_dim=128, intermediate_size=1024,
                                  vocab_size=1000, rms_norm_eps=1e-6, rope_theta=1e6,
                                  tie_word_embeddings=True, max_position_embeddings=4096)
    mdl = transformers.Qwen3ForCausalLM(hc).float().eval()
    return {k: v.float().numpy() for k, v in mdl.state_dict().items()}, \
        Spec(512, 2, 4, 4, 128, 1024, 1000)


@pytest.mark.parametrize("cross", [False, True])
def test_cards_engine_tiny_qwen3(cross):
    """A tiny Qwen3 at S = 4 over two cards: decode logits and the generate loop's tokens bit
    for bit against the ISA simulator's one machine; each card's DRAM its own slices' images
    (without the bank-half window, BHW 0: the images' bytes from 0, a small DRAM)."""
    from opentpu.llm.qwen3 import Engine
    W, spec = _tiny()
    t = T.load(TARGET)
    cfg = T.config(t, DRAM_BYTES=1 << 24, BHW=0)
    assert cfg.S == 4
    sh = IsaCards(cfg, cross=cross)
    isa = Engine(spec, W, cap=256, cfg=cfg, wformat="int8", resident=True)
    brd = Engine(spec, W, cap=256, cfg=cfg, wformat="int8", resident=True,
                 backend=lambda c, imgs: BoardBackend(c, imgs, transport=CardsTransport(sh.cards),
                                                      status=False))
    assert brd.backend.info["cfg"]["S"] == 4
    for tok in (11, 222, 333):
        a, b = isa.step(tok), brd.step(tok)
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32)), isa.pos
        assert brd.stats[-1]["instructions"] == isa.stats[-1]["instructions"]
    n = isa.image.nbytes
    for s in range(4):                  # slice s's image and state in card s // 2's DRAM
        assert np.array_equal(sh.dram[s][:n], isa.backend.machine.slices[s].dram[:n]), s
    assert isa.can_generate and brd.can_generate
    tok = int(np.argmax(a))
    want = isa.generate_card(tok, 6, stop_ids=[])
    got = brd.generate_card(tok, 6, stop_ids=[])
    assert got == want and len(got) == 6, (got, want)


def test_cards_wrong_card_id_runs_wrong_slices():
    """The machine's slices follow the CARD_IDs the host wrote: swapped, card 1's programs run as
    slices 0 and 1 (the run halts, its results are another machine's)."""
    cfg = _cfg()
    sh = IsaCards(cfg)
    ct = CardsTransport(sh.cards, link=False)
    for c, k in enumerate(sh.cards):
        k.reg_write(R.R_X2_CARD, (1 - c) | 2 << 8)
        k.reg_write(R.R_X2_CTRL, R.X2_LRST)
    from opentpu import isa as I
    b = Board(ct, lock=False)
    b.load(0x10000, [[I.halt()]] * 4)
    b.run()
    assert sh.order == [2, 3, 0, 1]


def test_x2sim_script_and_build():
    """X2SimTransport (the two-board testbench's host side): each card's lines "@c", the CARD_ID
    writes and the link's wait at the start of every simulation, the channels 16 card + 8 local
    slice + c, each card's build at two slices with NCARDS."""
    sim = X2SimTransport(ch_bytes=1 << 16)
    assert len(sim.ch) == 32 and sim.slices == 2 and sim.ncards == 2
    c0, c1 = sim.cards
    c1.mem_write(8 + 3, 64, np.full(16, 7, np.uint8))     # (before the script: a write flushes)
    assert sim.ch[16 + 8 + 3][64:80].tolist() == [7] * 16
    c1.persist(R.R_X2_CARD, 1 | 2 << 8)
    c1.persist_poll(R.R_X2_STATUS[0], R.X2_UP, R.X2_UP)
    c0.reg_write(R.R_CTRL, R.CTRL_RUN)
    assert c1.queue_read(R.R_STATUS) == 0
    assert sim.script == ["@0 W 8 1", "@1 R c"]
    assert sim.prologue == ["@1 W 300 201", "@1 P 310 80 80"]
    top, srcs, p = sim.sim_build()
    names = [str(s).rsplit("/", 1)[-1] for s in srcs]
    assert top == "tb_x2" and "tb_x2.sv" in names and "tb_bcuwg.sv" not in names
    from opentpu import rtlsim
    assert rtlsim.TB / "otpu_link_sim.sv" in srcs and p["NCARDS"] == 2
    assert all(s.is_file() for s in srcs)
    assert "+link_lat=40" in sim.plusargs


def test_cards_sequence_runs_wrap_and_lrst():
    """Issued operations survive RUN, wrap at 16 bits and restart only at LRST, in the
    machine and both fake CSRs. Equal but missing operations must fail the delta check."""
    from opentpu import isa as I
    sh = IsaCards(_cfg())
    ct = CardsTransport(sh.cards)
    b = Board(ct, lock=False)
    p = [I.loop(1, 3), I.bar(), I.halt()]
    b.load(0x10000, [p] * 4)
    b.run()
    assert sh.machine.collectives == 3 and [k.seq for k in sh.cards] == [3, 3]
    assert ct.sequences == ct.collective_delta == [3, 3]
    sh.machine.collectives = 65535
    for k in sh.cards:
        k.seq = 65535
    b.run()
    assert sh.machine.collectives == 2 and [k.seq for k in sh.cards] == [2, 2]
    assert ct.sequences == [2, 2] and ct.collective_delta == [3, 3]
    b.start()
    for k in sh.cards:
        k.seq -= 1
    with pytest.raises(LinkError, match="per-run delta.*expected 3"):
        b.wait()
    ct.recover()
    assert sh.machine.collectives == 0 and [k.seq for k in sh.cards] == [0, 0]
    b.run()
    assert sh.machine.collectives == 3 and [k.seq for k in sh.cards] == [3, 3]


def test_cards_program_collective_count():
    from opentpu import isa as I
    ct = CardsTransport(IsaCards(_cfg()).cards)
    for p, expected in [([I.halt()], 0),
                        ([I.loop(3, 7), I.loop(1, 5), I.bar(), I.bar(), I.halt()], 42),
                        ([I.loop(1, 2, rcount=8), I.bar(), I.halt()], None),
                        ([I.loop(2, 0), I.loop(1, 2, rcount=8), I.bar(), I.halt()], 0),
                        ([I.loop(1, 2, rcount=8), I.li(1, 1), I.halt()], 0),
                        ([I.halt(chain=True, ra=1, rb=2)], None)]:
        ct.collective_program([np.asarray(I.assemble(p), np.uint32)] * 4)
        assert ct.collectives == expected


def test_cards_machine_counts_issued_collective():
    from opentpu import isa as I
    from opentpu.isasim import Machine, SimError
    m = Machine(_cfg(), [[I.gather(0, 0, 1, 1, 1, 1, 1), I.halt()]] * 4, [None] * 4)
    with pytest.raises(SimError, match="overlap"):
        m.run()
    assert m.collectives == 1                   # issued even though its data was rejected


def test_cards_wait_entries_wait_for_remote_gather():
    """A whole-row norm cannot follow the local-only top of an NC=2 gather. Preserve its
    ordinary collective count while removing FOLLOW in the sidecar actually uploaded."""
    from opentpu import isa as I, waits as W
    cfg = _cfg()
    sh = IsaCards(cfg)
    ct = CardsTransport(sh.cards)
    p = [I.gather(0, 4096, 1, 256, 256, 256, 256),
         I.vop(I.V_RSSQ, 9000, 4096, 0, 1, 1024, 1, 1024, 1024), I.halt()]
    words = np.asarray(I.assemble(p), np.uint32)
    ents = W.entries(words, cfg.D, cfg.S, prog=p, aw=cfg.AW, win=cfg.WC_WIN)
    assert W.follow(ents[1]) and W.counts(ents[1])[W.U_COLL] == 1
    b = Board(ct, lock=False)
    b.load_program(0x10000, [words] * 4, [ents] * 4)
    want = W.loadable(words, ents & ~np.uint64(W.FOLLOW), cfg.D).view(np.uint8)
    for s in range(4):
        assert np.array_equal(sh.dram[s][0x10000:0x10000 + len(want)], want)


@pytest.fixture
def x2_verilator(have_verilator, monkeypatch):
    """One compile at a time, four workers on split units: the simulation host's memory budget."""
    from opentpu import rtlsim
    flags = list(rtlsim.VFLAGS)
    flags[flags.index("-j") + 1] = "4"
    flags += ["--output-groups", "0"]
    monkeypatch.setattr(rtlsim, "VFLAGS", flags)


def _x2_rtl_engine(W, spec, prompt, ngen, dram):
    """Real board/DIMM paths on two S=2 cards, both hubs NC=2, cable L=40; an S=4 ISA
    machine supplies every logit bit and device-generated token. Replay starts with LRST on
    each flush, so each launch's checked sequence delta is independent of earlier flushes."""
    import json
    from opentpu.llm.qwen3 import Engine
    t = T.load(TARGET + ":st16")
    cfg = T.config(t, DRAM_BYTES=dram, BHW=0)
    sim = X2SimTransport(target=t, ch_bytes=dram // 8,
                         plusargs=["+cp=3333", "+dimm_mc=1", "+mc_ba=15", "+mc_rw=62",
                                   "+mc_strict=1", "+mc_xbsw=16", "+mc_xrtw=8",
                                   "+mc_xwtr=8", "+mc_xgq=12", "+max_cycles=200000000"])
    ct = CardsTransport(sim.cards)
    isa = Engine(spec, W, cap=256, cfg=cfg, wformat="int8", resident=True)
    brd = Engine(spec, W, cap=256, cfg=cfg, wformat="int8", resident=True,
                 backend=lambda c, imgs: BoardBackend(c, imgs, transport=ct, status=False))
    decode, next_tokens = [], []
    def sequence_delta(issued):
        # Board.wait's later CTRL=0 replay must not replace the completed run's evidence.
        assert ct.sequences == ct.collective_delta == [issued, issued]

    for tok in prompt:
        before = isa.backend.machine.collectives
        a, b = isa.step(tok), brd.step(tok)
        sequence_delta((isa.backend.machine.collectives - before) & 0xFFFF)
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32)), isa.pos
        assert int(np.argmax(a)) == int(np.argmax(b))
        assert brd.stats[-1]["instructions"] == isa.stats[-1]["instructions"]
        decode.append(brd.stats[-1]["cycles"])
        next_tokens.append(int(np.argmax(b)))
        print("X2_RTL_DECODE " + json.dumps({"pos": isa.pos - 1,
              "next_token": next_tokens[-1], "cycles": decode[-1],
              "instructions": brd.stats[-1]["instructions"]}), flush=True)
    tok = next_tokens[-1]
    before = isa.backend.machine.collectives
    want = isa.generate_card(tok, ngen, stop_ids=[])
    got = brd.generate_card(tok, ngen, stop_ids=[])
    sequence_delta((isa.backend.machine.collectives - before) & 0xFFFF)
    assert got == want and len(got) == ngen, (got, want)
    assert brd.stats[-1]["instructions"] == isa.stats[-1]["instructions"]
    print("X2_RTL_QWEN3 " + json.dumps({"layers": spec.layers, "prompt": prompt,
          "next_tokens": next_tokens, "generated": got, "decode_cycles": decode,
          "generate_cycles": brd.stats[-1]["cycles"],
          "generate_cycles_per_token": brd.stats[-1]["cycles"] / ngen}), flush=True)


@pytest.mark.rtl
def test_cards_rtl_tiny_qwen3(x2_verilator):
    W, spec = _tiny()
    _x2_rtl_engine(W, spec, [11, 222, 333], 6, 1 << 24)


def _tiny35b():
    """The long-context qualification's tiny model: a 35B-shaped Qwen3.5-MoE, 16
    query heads and 2 KV heads of 256, 8 key and 16 value DeltaNet heads of 128 (the 35B's slice
    at S 4 four pairs), 8 experts of which 2 a token, a DeltaNet and an attention layer."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from opentpu.llm.moe import MoESpec
    from opentpu.llm.qwen35 import Spec
    E, K, F, H = 8, 2, 512, 512
    kinds = ("linear", "attn")
    torch.manual_seed(0)
    hc = transformers.Qwen3_5MoeTextConfig(
        hidden_size=H, num_hidden_layers=len(kinds), num_attention_heads=16,
        num_key_value_heads=2, head_dim=256, vocab_size=1000,
        layer_types=["full_attention" if k == "attn" else "linear_attention" for k in kinds],
        linear_num_key_heads=8, linear_num_value_heads=16, linear_key_head_dim=128,
        linear_value_head_dim=128, linear_conv_kernel_dim=4, tie_word_embeddings=True,
        max_position_embeddings=4096, rms_norm_eps=1e-6, num_experts=E,
        num_experts_per_tok=K, moe_intermediate_size=F, shared_expert_intermediate_size=F,
        rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25})
    m = transformers.Qwen3_5MoeForCausalLM(hc).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:     # zero-centered norms, but the DeltaNet output norm is plain
                p.copy_((1.0 if n.endswith("linear_attn.norm.weight") else 0.0)
                        + 0.1 * torch.randn_like(p))
            elif "experts" in n or "gate" in n:
                p.copy_(0.05 * torch.randn_like(p))
    W = {k: v.float().numpy() for k, v in m.state_dict().items()}
    return W, Spec(H, kinds, 16, 2, 256, 64, 16, 128, 128, F, 1000, lin_kheads=8,
                   moe=MoESpec(E=E, k=K, ffn=F, rule="softmax", shared=F))


@pytest.mark.rtl
def test_cards_rtl_b2_prefill_tiny_moe(x2_verilator, monkeypatch):
    """bcu1525-x2:b2's cards (B2: QACT ST and LDA, QONCE; the DMA's second command, DMA_LDQ 2; MM
    SPL, MXU_SPL; SIGMOID / SILU at VPU_CNS 3; the pumped MXU; QST HL) on the two-board
    co-simulation, two S=2 cards: _tiny35b's prompt of 128 tokens B2's way (expert-major, the
    mixers in tiles of 16 rows, the DeltaNet's rows by the chunked core at DNCHUNK 64: its split
    products one MM SPL each, the expert runs' rows quantized once), then decode steps and
    generate_card: the prompt's and each step's logits, the steps' instruction counts and the
    tokens the S=4 ISA machine's bit for bit."""
    import dataclasses
    from opentpu import isa as I
    from opentpu.llm.qwen3 import Engine
    monkeypatch.setenv("OTPU_HIGH", "0")
    monkeypatch.setenv("OTPU_POOL", "1")
    W, spec = _tiny35b()
    t, dram = T.load(TARGET + ":b2"), 1 << 25
    cfg = dataclasses.replace(T.config(t, DRAM_BYTES=dram, BHW=0), DNCHUNK=64)
    assert cfg.S == 4 and cfg.SPLMM and cfg.QONCE and cfg.VSIG and not cfg.FSUM
    sim = X2SimTransport(target=t, ch_bytes=dram // 8,
                         plusargs=["+cp=3333", "+dimm_mc=1", "+mc_ba=15", "+mc_rw=62",
                                   "+mc_strict=1", "+mc_xbsw=16", "+mc_xrtw=8",
                                   "+mc_xwtr=8", "+mc_xgq=12", "+max_cycles=400000000"])
    srcs = sim.sim_build()[1]
    defs = next(s for s in srcs if str(s).rsplit("/", 1)[-1].startswith("otpu_board_")).read_text()
    assert all(f".{k}({v})" in defs for k, v in (("QONCE", 1), ("MXU_SPL", 1), ("DMA_LDQ", 2),
                                                 ("VPU_SIG", 1), ("MXU_PUMP", 1),
                                                 ("QUANT_QSTHL", 1), ("OUTER", 0),
                                                 ("VPU_OUTER1", 1))), defs
    ct = CardsTransport(sim.cards)
    kw = dict(wformat="fp4", rows=1, resident=True, layer_major=2, expert_major=True, tile_rows=16)
    isa = Engine(spec, W, cap=256, cfg=cfg, **kw)
    brd = Engine(spec, W, cap=256, cfg=cfg, **kw,
                 backend=lambda c, imgs: BoardBackend(c, imgs, transport=ct, status=False))
    assert isa.image.dn_core_ok() and isa.image.nbytes <= dram
    toks = [int(x) for x in np.random.default_rng(11).integers(0, 1000, 128)]
    a = list(isa.prefill_chunks(toks, chunk=None))[-1][1]
    b = list(brd.prefill_chunks(toks, chunk=None))[-1][1]
    assert np.array_equal(np.asarray(a, np.float32).view(np.uint32),
                          np.asarray(b, np.float32).view(np.uint32))

    def flat(x):
        if isinstance(x, (list, tuple)):
            for y in x:
                yield from flat(y)
        else:
            yield x
    runs = isa._layer_runs
    ins = [x for v in runs.values() for x in flat(v[0] or [])]
    assert any(isinstance(k, tuple) and k and k[0] == "dnc" for k in runs), list(runs)
    assert any(getattr(x, "op", None) == I.MM and x.w[6] & I.MM_SPL for x in ins)
    assert any(getattr(x, "op", None) == I.LDA for x in ins)
    tok = int(np.argmax(a))
    for _ in range(2):
        a, b = isa.step(tok), brd.step(tok)
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32)), isa.pos
        assert brd.stats[-1]["instructions"] == isa.stats[-1]["instructions"]
        tok = int(np.argmax(a))
    want = isa.generate_card(tok, 4, stop_ids=[])
    assert brd.generate_card(tok, 4, stop_ids=[]) == want and len(want) == 4


@pytest.mark.rtl
def test_cards_rtl_sequence_runs(x2_verilator):
    """Two RUN launches in a single replay: both hubs reach 3 then 6, LRST alone clears
    them. Completion's before/after reads must refer to each launch, in the same flush."""
    from opentpu import isa as I
    sim = X2SimTransport(ch_bytes=(1 << 24) // 8, target=TARGET + ":st16")
    ct = CardsTransport(sim.cards)
    b = Board(ct, lock=False)
    b.load(0x10000, [[I.loop(1, 3), I.bar(), I.halt()]] * 4)
    reads = []
    for _ in range(2):
        ct.reg_write(R.R_CTRL, R.CTRL_CLEAR)
        ct.reg_write(R.R_CTRL, R.CTRL_RUN)
        ct.poll(R.R_STATUS, R.ST_HALTED, R.ST_HALTED)
        reads.append([t.queue_read(R.R_X2_SEQ) for t in sim.cards])
        ct.reg_write(R.R_CTRL, 0)
    for t in sim.cards:
        t.reg_write(R.R_X2_CTRL, R.X2_LRST)
    ct.wait_cycles(80)
    reset = [t.queue_read(R.R_X2_SEQ) for t in sim.cards]
    ct.flush()
    assert [[sim.results[i] for i in r] for r in reads] == [[3, 3], [6, 6]]
    assert [sim.results[i] for i in reset] == [0, 0]


@pytest.mark.rtl
def test_cards_rtl_qwen3_0_6b(x2_verilator):
    """Opt in with OTPU_X2_QWEN3, same real checkpoint as the ISA-only gate.
    Three prompt tokens and three generated tokens bound simulation time; no layers cut."""
    import os
    from pathlib import Path
    from opentpu.llm.qwen3 import Spec, load_weights
    path = os.environ.get("OTPU_X2_QWEN3")
    if not path:
        pytest.skip("set OTPU_X2_QWEN3 for the real 0.6B RTL checkpoint gate")
    path = Path(path).expanduser()
    assert path.name == "Qwen3-0.6B"
    _x2_rtl_engine(load_weights(path), Spec.from_hf(path), [151644, 872, 198], 3, 1 << 30)


def test_caps2_xcard():
    assert R.caps2(R.CAP2_XCARD)["xcard"]
    assert not R.caps2(0)["xcard"] and not R.caps2(None)["xcard"]


def test_cards_open_order_watchdog_and_recovery():
    """Both topology writes precede either reset; both resets precede the up polls. Recovery
    drains stopped cores, clears the link's stickies and never silently replays a run."""
    sh = IsaCards(_cfg())
    seen = []
    for k in sh.cards:
        k.reg_write = (lambda f, c: lambda o, v: (seen.append((c, o, v)), f(o, v)))(
            k.reg_write, k.c)
        k.poll = (lambda f, c: lambda o, m, v, *a: (seen.append((c, o, m, v)),
                                                   f(o, m, v, *a))[1])(k.poll, k.c)
    ct = CardsTransport(sh.cards, watchdog=12345)
    assert seen == [(0, 0x300, 0x200), (1, 0x300, 0x201),
                    (0, 0x304, 4), (1, 0x304, 4),
                    (0, 0x310, 0x80, 0x80), (0, 0x320, 0x80, 0x80),
                    (1, 0x310, 0x80, 0x80), (1, 0x320, 0x80, 0x80),
                    (0, 0x364, 12345), (1, 0x364, 12345)]
    assert [k.reg_read(R.R_COLL_WD) for k in sh.cards] == [12345] * 2
    assert sh.cards[0].reg_read(R.R_X2_INFO) == 1 | 32 << 8 | 2048 << 16
    assert sh.cards[0].reg_read(R.R_X2_HELLO[1]) == 0xA5002101
    assert sh.cards[0].reg_read(R.R_X2_CTRL) == 0
    sh.cards[1].seq = 7
    sh.cards[1].coll_status = R.COLL_ABORTED | R.COLL_WATCHDOG
    sh.cards[1].port_errors[0] = R.X2_CRC
    sh.cards[1].crc_errors[0] = 1
    ct.recover()
    assert sh.runs == 0 and [k.seq for k in sh.cards] == [0, 0]
    assert not ct.link_errors() and ct.map[(0, 1)] == (1, 1)
    ct.check_run()
    ct.watchdog = 0
    ct.set_cards()
    assert [k.watchdog for k in sh.cards] == [0, 0]
    sh.cards[0].regs[R.R_CTRL] = R.CTRL_RUN
    with pytest.raises(LinkError, match="stop both cards"):
        ct.set_cards()


@pytest.mark.parametrize("fault,match", [
    ("seq", "X2_SEQ mismatch"), ("abort", "aborted"), ("watchdog", "watchdog"),
    ("header", "header out of place"), ("link", "link check"), ("abort_rx", "ABORT received"),
    ("status", "STATUS ERROR"), ("crc", "port 0: CRC error"), ("down", "not up"),
])
def test_cards_completion_faults(fault, match):
    """Every halted run is checked, including a hub abort without STATUS ERROR and a core
    ERROR without any link sticky. No bad results escape Board.wait's completion poll."""
    from opentpu import isa as I
    sh = IsaCards(_cfg())
    ct = CardsTransport(sh.cards)
    b = Board(ct, lock=False)
    b.load(0x10000, [[I.bar(), I.halt()]] * 4)
    b.start()
    assert [k.seq for k in sh.cards] == [1, 1]
    k = sh.cards[1]
    if fault == "seq":
        k.seq += 1
    elif fault == "status":
        k.error = True
    elif fault == "crc":
        k.port_errors[0], k.crc_errors[0] = R.X2_CRC, 1
    elif fault == "down":
        k.enabled = False
    else:
        k.coll_status = {"abort": R.COLL_ABORTED, "watchdog": R.COLL_WATCHDOG,
                         "header": R.COLL_HEADER, "link": R.COLL_LINK,
                         "abort_rx": R.COLL_ABORT_RX}[fault]
    with pytest.raises(LinkError, match=match):
        b.wait()
    ct.recover()
    assert sh.runs == 1                         # no automatic replay
    assert b.run()["instructions"] == [2] * 4
    assert sh.runs == 2


def test_cards_batched_completion_faults():
    """A batched HALTED poll's checks belong to that same flush, before RUN is cleared. A
    replay simulation's later read starts a fresh machine and cannot check the completed run."""
    sh = IsaCards(_cfg())
    ct = CardsTransport(sh.cards)
    ct.batched = True
    for k in sh.cards:
        k.results, k.queued = [], []
        def queue_read(o, k=k):
            k.queued.append(o)
            return len(k.queued) - 1
        def flush(k=k):
            k.results = [k.reg_read(o) for o in k.queued]
            k.queued = []
        k.queue_read, k.flush = queue_read, flush
        k.poll = lambda *a: R.ST_HALTED
    sh.cards[1].seq = 1
    ct.poll(R.R_STATUS, R.ST_HALTED, R.ST_HALTED)
    with pytest.raises(LinkError, match="X2_SEQ mismatch"):
        ct.flush()
    sh.cards[1].seq = 0
    ct.poll(R.R_STATUS, R.ST_HALTED, R.ST_HALTED)
    ct.flush()


def test_cards_st16_build_hashes():
    """The P3 build's per-card configuration, before board_params emits the otpu_board macro:
    change S in the target before hashing, retain platform 7 and all st16 arch/uarch knobs."""
    from dataclasses import replace
    t = T.load(TARGET + ":st16")
    card = replace(t, arch=dict(t.arch, S=t.arch["S"] // T.cards(t)))
    p = T.rtl_params(card)
    assert (p["S"], p["PLATFORM"], p["ST16"], p["NCH"], p["CB"]) == (2, 7, 1, 8, 128)
    assert p["ARCH_HASH"] == T.arch_hash(T.build_config(card)) == 0xE331F79C
    assert T.arch_hash(T.build_config(t)) == 0xD5239726


def test_one_card_is_its_targets_build():
    """Board.info on one card of bcu1525-x2 (a transport of its own, no target named: the
    two-slice board of test_coll_x2's aborted collectives): the card reports its own slices,
    S / cards = 2, and check_target takes it as bcu1525-x2's build, not a variant's (bcu1525-x2
    has variants: before this, ConfigMismatch "a 2-slice build of platform 7"). A slice count
    neither the machine's nor a card's is still a variant's, refused."""
    from opentpu.host.fake import FakeTransport, target_cfg
    card = dict(target_cfg(TARGET), S=T.load(TARGET).arch["S"] // T.cards(TARGET))
    i = Board(FakeTransport(devname=None, regmap=4, cfg=card, ch_bytes=1 << 12), lock=False).info()
    assert i["target"].name == TARGET and i["cfg"]["S"] == 2
    whole = Board(FakeTransport(devname=None, regmap=4, cfg=target_cfg(TARGET), ch_bytes=1 << 12),
                  lock=False).info()
    assert whole["target"].name == TARGET and whole["cfg"]["S"] == 4
    with pytest.raises(ConfigMismatch, match="a 3-slice build of platform 7"):
        Board(FakeTransport(devname=None, regmap=4, cfg=dict(card, S=3), ch_bytes=1 << 12),
              lock=False).info()


def test_cards_qwen3_0_6b():
    """Opt-in gate (OTPU_X2_QWEN3=~/otpu-models/Qwen3-0.6B): the real model's
    decode logits and device-side generation bit-identical to one S=4 ISA machine. Bound the
    simulated DRAM at 1 GiB a slice and disable BHW for a compact low-address image; every
    other knob is bcu1525-x2:st16's. The ISA has instruction counts, no cycle timing model."""
    import json
    import os
    from pathlib import Path
    from opentpu.llm.qwen3 import Engine, Spec, load_weights
    path = os.environ.get("OTPU_X2_QWEN3")
    if not path:
        pytest.skip("set OTPU_X2_QWEN3 for the real 0.6B checkpoint")
    path = Path(path).expanduser()
    assert path.name == "Qwen3-0.6B", "this gate is bounded to the 0.6B checkpoint"
    W, spec = load_weights(path), Spec.from_hf(path)
    cfg = T.config(T.load(TARGET + ":st16"), DRAM_BYTES=1 << 30, BHW=0)
    sh = IsaCards(cfg)
    isa = Engine(spec, W, cap=256, cfg=cfg, wformat="int8", resident=True)
    brd = Engine(spec, W, cap=256, cfg=cfg, wformat="int8", resident=True,
                 backend=lambda c, imgs: BoardBackend(c, imgs, transport=CardsTransport(sh.cards),
                                                      status=False))
    assert cfg.S == 4 and brd.backend.info["cfg"]["S"] == 4
    prompt = [151644, 872, 198, 3838, 374, 279, 6722, 315, 9625, 30]
    counts, next_tokens = [], []
    for tok in prompt:
        a, b = isa.step(tok), brd.step(tok)
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32)), isa.pos
        assert brd.stats[-1]["instructions"] == isa.stats[-1]["instructions"]
        counts.append(brd.stats[-1]["instructions"])
        next_tokens.append(int(np.argmax(a)))
        print(json.dumps({"decode_pos": isa.pos - 1, "next_token": next_tokens[-1],
                          "instructions": counts[-1], "seq": [k.seq for k in sh.cards]}),
              flush=True)
    tok = next_tokens[-1]
    assert isa.can_generate and brd.can_generate
    want = isa.generate_card(tok, 6, stop_ids=[])
    got = brd.generate_card(tok, 6, stop_ids=[])
    assert got == want and len(got) == 6, (got, want)
    assert brd.stats[-1]["instructions"] == isa.stats[-1]["instructions"]
    print("X2_QWEN3 " + json.dumps({"prompt": prompt, "decode_next_tokens": next_tokens,
          "generated": got, "decode_instructions": counts,
          "generate_instructions": brd.stats[-1]["instructions"],
          "cycles": None, "cycle_note": "ISA is functional; the fake cycle CSR reads zero",
          "seq": [k.seq for k in sh.cards], "runs": sh.runs}), flush=True)


def test_cards_missing_port_stops_open():
    sh = IsaCards(_cfg())
    read = sh.cards[1].reg_read
    sh.cards[1].reg_read = lambda o: 0 if o == R.R_X2_STATUS[1] else read(o)
    ct = CardsTransport(sh.cards, link=False)
    with pytest.raises(LinkError, match="card 1 port 1 not up"):
        ct.set_cards(wait=0.01)


def test_cards_recovery_waits_for_stores():
    sh = IsaCards(_cfg())
    ct = CardsTransport(sh.cards)
    seen, write = [], sh.cards[0].reg_write
    sh.cards[0].reg_write = lambda o, v: (seen.append((o, v)), write(o, v))
    def timeout(*a):
        raise TimeoutError("stores outstanding")
    sh.cards[1].poll = timeout
    with pytest.raises(LinkError, match="card 1: stores did not drain"):
        ct.recover()
    assert seen == [(R.R_CTRL, 0)]               # neither link was reset
