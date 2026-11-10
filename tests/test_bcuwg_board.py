"""The BCU1525 at WG 8 (targets/bcu1525-wg.toml) on its board model: otpu_bcu_wg_sys (otpu_board at
one slice of 8 channels, the WG targets' otpu_hbm_ch bridges, the DIMMs' pairs otpu_mig_pair) in front of
models of the four DIMMs' AXI4 ports (sim/verilator/tb_bcuwg.sv through SimTransport bcuwg),
built at the target's parameters (its Config and rtl_params: MCOLS 8, WG 8, port W's beats posted
by the bridges on aclk, NAT_XPOST): tiny Qwen3 and Qwen3.5 images bit for bit against the ISA
simulator (logits, instruction counts, the slice's DRAM read back through the host's channel
map), with the DIMMs answering out of order across the pair's two channels and stalling. bcu1525-s2:
two slices, four channels a DIMM, each floorplan's DIMM map (bcu_fp.py)."""
import re
from pathlib import Path

import numpy as np
import pytest

from opentpu import target as T
from opentpu.host.board import BoardBackend, SimTransport, device_config

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

TARGET = "bcu1525-wg"


def _transport(dram_bytes: int, plusargs: list | None = None,
               caw: int | None = None, target: str = TARGET) -> SimTransport:
    """The board model with `dram_bytes` of dense channel images (caw: tb_bcuwg's channels of
    2^(caw + 7) bytes, the beats past the dense images sparse) of `target`'s build (bcu1525-wg or
    a variant of it)."""
    t = T.load(target)
    rp = T.rtl_params(t)
    p = {k: rp[k] for k in ("MCOLS", "LANES", "TMEM_WORDS", "SEQ_WC", "OUTER", "DSTEP2")}
    p |= {"WG": t.arch["WG"], "ACT_ROWS": t.arch["ACT_ROWS"], "NAT_XPOST": 1}
    p |= {k: rp[k] for k in ("ST16",) if k in rp}
    return SimTransport(ch_bytes=dram_bytes // 8, stall=20, seed=5, target=t, bcuwg=True,
                        params=p, plusargs=plusargs, caw=caw)


def _tiny(model: str):
    if model == "qwen3":
        from opentpu.llm.qwen3 import Spec
        torch.manual_seed(0)
        hc = transformers.Qwen3Config(hidden_size=256, num_hidden_layers=2, num_attention_heads=4,
                                      num_key_value_heads=2, head_dim=128, intermediate_size=512,
                                      vocab_size=1000, rms_norm_eps=1e-6, rope_theta=1e6,
                                      tie_word_embeddings=True, max_position_embeddings=4096)
        mdl = transformers.Qwen3ForCausalLM(hc).float().eval()
        return {k: v.float().numpy() for k, v in mdl.state_dict().items()}, \
            Spec(256, 2, 4, 2, 128, 512, 1000)
    from test_qwen35 import _tiny_model
    _, W, spec = _tiny_model(8, init=0.2)
    return W, spec


OOO = ["+dimm_ooo=50", "+dimm_jit=60", "+dimm_st=30", "+dimm_bw=60"]


def _dram_diffs(backend, dram, images, n: int, piece: int = 64 << 20, gap: int = 4096) -> list:
    """The runs [lo, hi) of [0, n) where the board's DRAM (backend.read) and the ISA simulator's
    (dram) differ, runs closer than `gap` joined, each with its region: the image's spans
    (qwen3.Images.spans: its low part with the Engine's areas, its weights from `high`), the
    board's program area (prog_at: the programs and their wait entries), the bytes neither
    writes (unwritten). A piece at a time (an 8 GiB read of a high image and its compare in one
    took ~20 GB)."""
    spans = images.spans(n)
    prog = (backend.prog_at, backend.prog_at + backend.W.area(backend.cfg.IMEM_WORDS,
                                                              backend.cfg.SEQ_WC))
    regions = [(lo, hi, "image low" if not k else "image high") for k, (lo, hi) in
               enumerate(spans)] + [(*prog, "program area")]
    runs = []
    for a in range(0, n, piece):
        k = min(piece, n - a)
        d = np.flatnonzero(backend.read(0, a, k) != dram[a:a + k]) + a
        cut = np.flatnonzero(np.diff(d) > gap)
        for lo, hi in zip(np.r_[d[:1], d[cut + 1]], np.r_[d[cut], d[-1:]] + 1):
            if runs and lo - runs[-1][1] <= gap:
                runs[-1][1] = int(hi)
            else:
                runs.append([int(lo), int(hi)])

    def region(lo, hi):
        inside = [w for a, b, w in regions if a <= lo and hi <= b]
        return inside[0] if inside else "unwritten" if not any(
            lo < b and a < hi for a, b, _ in regions) else "spans regions"
    return [(hex(lo), hex(hi), region(lo, hi)) for lo, hi in runs]


@pytest.mark.parametrize("model,wf,plus,target", [
    ("qwen3", "int8", [], TARGET),
    ("qwen35", "fp4", [], TARGET),
    ("qwen35", "fp4", OOO, TARGET),
    ("qwen3", "int8", OOO, TARGET + ":ic"),
    ("qwen35", "fp4", OOO, TARGET + ":ic"),
    ("qwen3", "int8", OOO, TARGET + ":wmx"),
    ("qwen35", "fp4", OOO, TARGET + ":wmx"),
], ids=["qwen3-int8", "qwen35-fp4", "qwen35-fp4-ooo", "qwen3-int8-icache", "qwen35-fp4-icache",
        "qwen3-int8-wmx", "qwen35-fp4-wmx"])
def test_bcuwg_tiny_models_on_board_model(have_verilator, model, wf, plus, target):
    """(variant ic: IMEM an instruction cache, the programs run from DRAM: docs/icache.md; wmx:
    the instruction cache, the pumped MXU and the VPU at CNS 5, docs/wide_mxu.md)"""
    from opentpu.llm.qwen3 import Engine
    W, spec = _tiny(model)
    t = T.load(target)
    need = spec.image(T.config(t), 256, wformat=wf).nbytes
    cfg = T.config(t, DRAM_BYTES=1 << max(22, (need - 1).bit_length() + 1))
    assert cfg.S == 1 and cfg.WG == 8 and cfg.GALL and not cfg.PAIR
    tr = _transport(cfg.DRAM_BYTES, plus, target=target)
    defs = tr.sim_build()[1][0].read_text()      # (OTPU_BOARD: otpu_board's parameters)
    assert all(f".{k}({v})" in defs for k, v in (("NAT_XPOST", 1), ("WG", 8), ("MCOLS", 8)))
    assert (".ICACHE(1)" in defs) == tr.icache and tr.icache >= cfg.ICACHE   # (OTPU_ICACHE)
    # the VPU as the model compiles for it (the target's FSUM, SIGMOID / SILU and MAC)
    assert all((f".{k}(1)" in defs) == c for k, c in (("VPU_FSUM", cfg.FSUM),
                                                     ("VPU_SIG", cfg.VSIG), ("VPU_MAC", cfg.VMAC)))
    isa = Engine(spec, W, cap=256, cfg=cfg, wformat=wf)
    imgs = []                       # (the Images the board's backend writes: their spans)
    brd = Engine(spec, W, cap=256, cfg=cfg, wformat=wf,
                 backend=lambda c, m: imgs.append(m) or BoardBackend(c, m, transport=tr))
    info = brd.backend.info
    assert (info["cfg"]["S"], info["cfg"]["channels"], info["cfg"]["WG"]) == (1, 8, 8)
    assert info["cfg"]["platform"] == t.platform_id
    assert device_config(info).WG == 8
    assert brd.backend.board.icache == tr.icache == device_config(info).ICACHE
    rd = [0] * 4                    # each DIMM's R beats over every simulation (flush)
    flush = tr.flush

    def counted(*a, **k):
        ran = bool(tr.script)       # (an empty flush runs nothing and keeps the last output)
        r = flush(*a, **k)
        for d in range(4 if ran else 0):
            rd[d] += int(tr.out.split(f"dimm {d}: ")[1].split()[0])
        return r
    tr.flush = counted
    for tok in (11, 222, 333):
        a, b = isa.step(tok), brd.step(tok)
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32)), isa.pos
        assert brd.stats[-1]["instructions"] == isa.stats[-1]["instructions"]
    # the DRAM is the ISA simulator's but for the board's program area: an image with its weights
    # from 4 GiB on (BHW 33: wmx) has it inside [0, nbytes), after its low part (prog_at)
    diffs = _dram_diffs(brd.backend, isa.backend.machine.slices[0].dram, imgs[0],
                        isa.image.nbytes)
    print("dram diffs", target, model, diffs)
    assert all(w == "program area" for *_, w in diffs), diffs
    # every DIMM carried the weights' traffic
    assert all(r > 0 for r in rd), rd


@pytest.mark.parametrize("target,cns,pump", [(TARGET + ":wmx", 5, 1), (TARGET + ":bhw16", 0, 0),
                                             ("bcu1525-x2:wmx", 5, 1), ("bcu1525-x2:st16", 0, 0)])
def test_bcuwg_board_model_builds_the_targets_vpu_and_mxu(target, cns, pump):
    """The board model's VPU_CNS and MXU_PUMP are the target's build's (sim_uarch), as the card's
    (make_wg.py: rtl_params): bcu1525-wg:wmx's and bcu1525-x2:wmx's (the x2 co-simulation's cards)
    VPU at CNS 5 and its MXU pumped, bhw16 and st16 with neither (otpu_board's defaults). The
    board model's build only: no simulation."""
    from opentpu.host.cards import X2SimTransport
    tr = X2SimTransport(target) if T.cards(T.load(target)) > 1 else _transport(1 << 22,
                                                                               target=target)
    defs = tr.sim_build()[1][0].read_text()      # (OTPU_BOARD: otpu_board's parameters)
    assert (".VPU_CNS(5)" in defs) == (cns == 5) and (".MXU_PUMP(1)" in defs) == bool(pump)
    assert ".VPU_CNS(3)" not in defs and ".MXU_PUMP(0)" not in defs


@pytest.mark.parametrize("model", ["qwen3", "qwen35"])
def test_bcuwg_generate_with_weights_above_4gib(monkeypatch, model):
    """The on-device generate loop at 32 GiB's layout (OTPU_HIGH 1: the weights from 4 GiB, the
    LM head's MM through the window register) on the ISA simulator: its tokens are the low
    layout's (a window register that the layers' loop had stepped in the token loop's body
    started that loop's next pass 4 GiB off, so every token after the second read 0)."""
    from opentpu.llm.qwen3 import Engine
    W, spec = _tiny(model)
    cfg = T.config(T.load(TARGET), DRAM_BYTES=1 << 33)
    got = {}
    for high in ("0", "1"):
        monkeypatch.setenv("OTPU_HIGH", high)
        e = Engine(spec, W, cap=256, cfg=cfg, wformat="fp4", resident=True)
        assert bool(e.image.high) == (high == "1") and e.can_generate
        lg = [e.step(t) for t in (5, 17, 300)][-1]
        got[high] = e.generate_card(int(np.argmax(lg)), 6, stop_ids=[])
    assert got["1"] == got["0"] and len(got["0"]) == 6, got


@pytest.mark.parametrize("model", ["qwen3", "qwen35"])
def test_bcuwg_generate_bounds_its_arguments(monkeypatch, model):
    """The generate loop's argument RLD MULs carry their variables' bounds (c * [0, bound):
    generate.check_args), so the waits pass takes a KV append's footprint as its rows and not
    all of DRAM: fewer weight MMs wait for the DMA than with the bounds stripped, and the
    counts hold on the ISA simulator's run (OTPU_WAITS_CHECK) with the tokens of the loop."""
    from dataclasses import replace

    from opentpu import isa as I
    from opentpu import waits as WT
    from opentpu.llm.qwen3 import Engine
    monkeypatch.setenv("OTPU_HIGH", "1")
    W, spec = _tiny(model)
    cfg = T.config(T.load(TARGET), DRAM_BYTES=1 << 33)
    e = Engine(spec, W, cap=256, cfg=cfg, wformat="fp4", resident=True)
    prog = e._generate_prog(1)[0]
    # (a negative c's bound wraps mod 2^AW: compiler.mul_bound, test_gptoss's mask row)
    args = [(i, x) for i, x in enumerate(prog) if x.op == I.RLD and "argument" in x.comment
            and "argument -" not in x.comment]
    assert args and all(x.flags & I.F_BOUND and x.w[2] == 0 and x.w[3] > 0 for _, x in args)
    loose = list(prog)
    for i, x in args:
        loose[i] = replace(x, flags=x.flags & ~I.F_BOUND, w=[*x.w[:2], 0, 0, *x.w[4:]])

    def dma_waits(p):
        wc = WT.wait_counts(p, S=cfg.S, aw=cfg.AW)
        return sum(1 for x, c in zip(p, wc) if x.op == I.MM and any(WT.counts(c)[j]
                                                                    for j in (0, 5)))
    assert dma_waits(list(prog)) < dma_waits(loose)
    lg = [e.step(t) for t in (5, 17, 300)][-1]
    want = e.generate_card(int(np.argmax(lg)), 6, stop_ids=[])
    monkeypatch.setenv("OTPU_WAITS_CHECK", "1")
    c = Engine(spec, W, cap=256, cfg=cfg, wformat="fp4", resident=True)
    lg = [c.step(t) for t in (5, 17, 300)][-1]
    assert c.generate_card(int(np.argmax(lg)), 6, stop_ids=[]) == want


def test_bcuwg_weights_above_4gib_on_board_model(have_verilator, monkeypatch):
    """At 32 GiB (CAW 25: 4 GiB a channel; the DIMM models' beats past the dense images sparse)
    with otpu_board at the target's AW 36: the tiny Qwen3.5 (fp4) with its weights from 4 GiB on
    (OTPU_HIGH 1: qwen3.Images high, the program area below 4 GiB), its logits and instruction
    counts the ISA simulator's bit for bit, its low part and its weights read back through the
    host's channel map (4 GiB a channel) as the simulator's DRAM."""
    from opentpu.llm.qwen3 import HIGH, Engine
    monkeypatch.setenv("OTPU_HIGH", "1")
    W, spec = _tiny("qwen35")
    t = T.load(TARGET)
    cfg = T.config(t, DRAM_BYTES=1 << 33)
    assert cfg.AW == 36
    isa = Engine(spec, W, cap=256, cfg=cfg, wformat="fp4")
    im = isa.image
    assert im.high and im.layer0 == HIGH and im.nbytes > HIGH
    dense = 1 << 25                                 # (4 MiB a channel: the low part, programs)
    tr = _transport(dense, caw=25)
    _, srcs, tb = tr.sim_build()
    assert ".AW(36)" in srcs[0].read_text() and tb["CAW"] == 25
    brd = Engine(spec, W, cap=256, cfg=cfg, wformat="fp4",
                 backend=lambda c, imgs: BoardBackend(c, imgs, transport=tr))
    c = brd.backend.info["cfg"]
    assert (c["channels"], c["channel_bytes"]) == (8, 1 << 32)
    assert brd.backend.high == HIGH and brd.backend.prog_at < dense
    for tok in (11, 222, 333):
        a, b = isa.step(tok), brd.step(tok)
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32)), isa.pos
        assert brd.stats[-1]["instructions"] == isa.stats[-1]["instructions"]
    dram = isa.backend.machine.slices[0].dram
    assert np.array_equal(brd.backend.read(0, 0, im.low_end), dram[:im.low_end])
    assert np.array_equal(brd.backend.read(0, HIGH, im.nbytes - HIGH), dram[HIGH:im.nbytes])
    assert all(tr.hi), [len(h) for h in tr.hi]      # (every channel holds weights up there)


@pytest.mark.parametrize("model,wf,fp,plus,target", [
    ("qwen3", "int8", "c", [], "bcu1525-s2"),
    ("qwen35", "fp4", "a", ["+dimm_ooo=50", "+dimm_jit=60", "+dimm_st=30", "+dimm_bw=60"], "bcu1525-s2"),
    ("qwen35", "fp4", "c", ["+dimm_mc=1", "+mc_ba=15"], "bcu1525-s2"),
    ("qwen35", "fp4", "c", ["+dimm_mc=1", "+mc_ba=15"], "bcu1525-s2:st16"),
], ids=["qwen3-int8-c", "qwen35-fp4-a-ooo", "qwen35-fp4-c-mc", "qwen35-fp4-c-mc-st16"])
def test_bcus2_tiny_models_on_board_model(have_verilator, model, wf, fp, plus, target):
    """Two slices (targets/bcu1525-s2.toml: otpu_bcu_wg_sys S 2, a slice on two DIMMs, four channels
    a DIMM through otpu_mig_pair NC 4, the DIMM slots on the DIMMs by floorplan `fp`'s DMAP, its
    bridges' pipelines RLY; the collective through the pipelined hub; the bank-half window, BHW 33,
    the target's: the images high, the weights past the slice's first 8 GiB, a Qwen3.5's DeltaNet
    mixers' from 4 GiB, the channels past their dense images sparse): the tiny models
    tensor-parallel, their logits and instruction counts the ISA simulator's bit for bit, each
    slice's written spans read back through the host's channel map, every DIMM carrying traffic.
    bcu1525-s2:st16: the same with the DeltaNet states in e6m9 (ST16, CAPS2 bits 14:13 1)."""
    import sys
    from pathlib import Path
    from opentpu.llm.qwen3 import Engine, Images, high_base
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "boards/bcu1525/scripts"))
    import bcu_fp
    W, spec = _tiny(model)
    t = T.load(target)
    st16 = target.endswith(":st16")
    cfg = T.config(t)
    assert cfg.S == 2 and cfg.WG == 8 and cfg.GALL and not cfg.PAIR
    assert cfg.BHW == 33 and cfg.DRAM_BYTES == 1 << 34
    rp = T.rtl_params(t)
    p = {k: rp[k] for k in ("MCOLS", "LANES", "TMEM_WORDS", "SEQ_WC", "OUTER", "DSTEP2")}
    pl = bcu_fp.plan(fp)
    p |= {"WG": t.arch["WG"], "ACT_ROWS": t.arch["ACT_ROWS"], "NAT_XPOST": 1,
          "DMAP": pl["dmap"], "RLY": pl["rly"]}
    p |= {k: rp[k] for k in ("ST16",) if k in rp}
    tr = SimTransport(ch_bytes=1 << 22, stall=20, seed=5, target=t, bcuwg=True, params=p,
                      plusargs=plus)
    _, srcs, tb = tr.sim_build()
    defs = srcs[0].read_text()
    assert all(f".{k}({v})" in defs for k, v in (("COLL_PIPE", 1), ("XRS", 2), ("WG", 8)))
    assert (".ST16(1)" in defs) == st16 == bool(cfg.ST16)
    assert (tb["S"], tb["CAW"], tb["DMAP"], tb["BHW"]) == (2, 24, pl["dmap"], 30)
    isa = Engine(spec, W, cap=256, cfg=cfg, wformat=wf)
    im = isa.image
    assert im.high and im.nbytes > high_base(cfg)
    assert getattr(im, "mix", False) == (model == "qwen35")
    assert getattr(im, "st16", False) == st16
    brd = Engine(spec, W, cap=256, cfg=cfg, wformat=wf,
                 backend=lambda c, imgs: BoardBackend(c, imgs, transport=tr))
    info = brd.backend.info
    assert (info["cfg"]["S"], info["cfg"]["channels"], info["cfg"]["WG"]) == (2, 8, 8)
    assert info["cfg"]["platform"] == t.platform_id
    assert info["caps2"]["st16"] == cfg.ST16
    rd = [0] * 4
    flush = tr.flush

    def counted(*a, **k):
        ran = bool(tr.script)
        r = flush(*a, **k)
        for d in range(4 if ran else 0):     # (tb_bcuwg's line: +dimm_mc's model prints its own)
            rd[d] += int(re.search(rf"^dimm {d}: (\d+) R beats", tr.out, re.M).group(1))
        return r
    tr.flush = counted
    for tok in (11, 222, 333):
        a, b = isa.step(tok), brd.step(tok)
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32)), isa.pos
        assert brd.stats[-1]["instructions"] == isa.stats[-1]["instructions"]
    spans = Images([], im.low_end, getattr(im, "wlo", high_base(cfg)),
                   getattr(im, "wskip", None)).spans(im.nbytes)
    for s in range(2):
        dram = isa.backend.machine.slices[s].dram
        for lo, hi in spans:
            assert np.array_equal(brd.backend.read(s, lo, hi - lo), dram[lo:hi]), (s, hex(lo))
    assert all(r > 0 for r in rd), rd


@pytest.mark.parametrize("variant", ["bhw", "bhw16"])
def test_bcuwg_bank_half_window_on_board_model(have_verilator, variant):
    """bcu1525-wg:bhw (the pairs' bank-half window, otpu_mig_pair BHW 30: the slice's first 8 GiB
    with the DIMMs' bank-high bit from the window's top; CAPS2 BHW 33): the tiny Qwen3.5 (fp4) in
    the window's layout (qwen35 Image.mix: its low area below 4 GiB, the DeltaNet mixers' weights
    from 4 GiB, the other weights from 8 GiB), its logits and instruction counts the ISA
    simulator's bit for bit and every part read back through the host's channel map as the
    simulator's DRAM (the DIMM models undo the swap the pairs make). The variant's sequencer
    window is 32: the host finds it in the CFG_HASH and loads the entries' saturated counts
    (Config.WC_WIN 32), and otpu_seq's WAITC check holds them to the window. bhw16: the same with
    the DeltaNet states in e6m9 (ST16, docs/isa.md "ST16"; CAPS2 bits 14:13 1), half their bytes."""
    from opentpu.llm.qwen3 import HIGH, Engine
    W, spec = _tiny("qwen35")
    t = T.load(f"{TARGET}:{variant}")
    cfg = T.config(t, DRAM_BYTES=1 << 34)
    isa = Engine(spec, W, cap=256, cfg=cfg, wformat="fp4")
    im = isa.image
    assert cfg.BHW == 33 and im.mix and im.wlo == HIGH and im.layer0 == 2 * HIGH
    assert cfg.WC_WIN == 32 and im.st16 == (variant == "bhw16")
    tr = _transport(1 << 25, caw=25, target=f"{TARGET}:{variant}")
    _, srcs, tb = tr.sim_build()
    assert tb["BHW"] == 30 and tb["CAW"] == 25
    assert ".WIN(32)" in Path(srcs[0]).read_text() and ".SEQ_WC(1)" in Path(srcs[0]).read_text()
    assert (".ST16(1)" in Path(srcs[0]).read_text()) == im.st16
    brd = Engine(spec, W, cap=256, cfg=cfg, wformat="fp4",
                 backend=lambda c, imgs: BoardBackend(c, imgs, transport=tr))
    info = brd.backend.info
    assert info["caps2"]["bhw"] == 33 and device_config(info).BHW == 33
    assert device_config(info).WC_WIN == brd.backend.cfg.WC_WIN == 32
    assert info["caps2"]["st16"] == device_config(info).ST16 == cfg.ST16
    assert brd.image.st16 == im.st16 and brd.image.sbytes == im.sbytes
    for tok in (11, 222, 333):
        a, b = isa.step(tok), brd.step(tok)
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32)), isa.pos
        assert brd.stats[-1]["instructions"] == isa.stats[-1]["instructions"]
    dram = isa.backend.machine.slices[0].dram
    for lo, hi in ((0, im.low_end), (HIGH, im.wskip[0]), (im.wskip[1], im.nbytes)):
        assert np.array_equal(brd.backend.read(0, lo, hi - lo), dram[lo:hi]), hex(lo)
