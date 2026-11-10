"""Weight groups on the WG targets' board model (docs/weight_groups.md, acceptance): tiny Qwen3,
Qwen3.5, SmolLM3, Phi-3 and 27B-shaped images grouped at WG 8 (int8 groups of 8 rows, 4-bit of
16) on 4 slices of 8 channels of 128-byte beats (NCH 8, CB 128, MCOLS 8, WG 8), every MM in
groups on port W (GALL, section 12: the KV caches in groups, their appends QST GRP, attention's
MMs with RMAX and ASCALE): logits bit-identical to the ISA simulator, every slice's instruction
count and DRAM the simulator's; the bitstream reports WG (CFG_UARCH) and no column reuse (CAPS
bit5)."""
import dataclasses

import numpy as np
import pytest

from opentpu.host.board import BoardBackend, SimTransport, device_config

from conftest import sim_board_config
from test_board import _tiny_qwen3_s4, _tiny_qwen35_s4

pytest.importorskip("transformers")
WG8 = {"MCOLS": 8, "WG": 8}


def _transport(dram_bytes: int, seed: int = 5, stall: int = 20, **params) -> SimTransport:
    return SimTransport(ch_bytes=dram_bytes // 8, stall=stall, seed=seed, target="bcu1525-x2",
                        params=dict(WG8, **params))


@pytest.mark.parametrize("model,chash,fill,wf", [
    ("qwen3", True, True, "int8"), ("qwen35", True, True, "int8"), ("qwen3", False, True, "int8"),
    ("qwen35", True, False, "int8"), ("qwen3", True, True, "fp4"), ("qwen35", True, True, "nvfp4"),
    ("q27", True, True, "fp4"), ("q27f", True, True, "fp4")])
def test_tiny_models_four_slices_wg8_on_board_model(have_verilator, model, chash, fill, wf):
    """chash False: a build without the channels' hash (otpu_board CHASH 0, CAPS bit7 clear):
    the host's map, the adapter's and the images' row order all in order (h = 0). fill False:
    the adapters without the SW queue's fill reads (NAT_FILL 0: QST's partial beats, V^T's
    columns, go out with their byte masks). wf: the layers' format (fp4, NVFP4: the packed
    4-bit advance on groups of 16 rows). q27: test_qwen35's tiny 27B shape (12 DeltaNet value
    heads and 4 key heads a slice, an untied int8 head, the split int8 embedding table); q27f:
    the same with Spec.dn_fold (the DeltaNet's folded constants and kernels)."""
    _check(model, wf, chash, fill)


@pytest.mark.parametrize("model,wf,swg", [("qwen3", "int8", 0), ("qwen35", "fp4", 0),
                                          ("q27", "fp4", 0), ("q27f", "fp4", 0),
                                          ("qwen35", "fp4", 1), ("q27", "fp4", 1)])
def test_posted_w_beats_wg8_on_board_model(have_verilator, model, wf, swg):
    """(a) v2 (NAT_XPOST, docs/native_channels.md): port W's beats posted into the MXU's chunk FIFO
    by the channels (the board model's at the core clock): the logits, instruction counts and
    DRAM are the ISA simulator's. swg: the adapters with one SW gather beat and one posted beat a
    cycle for every channel (NAT_SWG 1, the WG targets' area cut)."""
    _check(model, wf, xpost=True, swg=bool(swg))


# test_llama's tiny SmolLM3 and Phi-3 on 4 slices (hidden and a 4-bit FFN's 2 x S x D at least,
# 4 KV heads of 128)
LLAMA_S4 = {"smollm3": {"hidden_size": 512, "num_key_value_heads": 4, "intermediate_size": 1024},
            "phi3": {"hidden_size": 1024, "num_attention_heads": 8, "num_key_value_heads": 4,
                     "intermediate_size": 1024}}


@pytest.mark.parametrize("model,wf", [("smollm3", "int8"), ("phi3", "fp4")])
def test_tiny_llama_four_slices_wg8_on_board_model(have_verilator, tmp_path, model, wf):
    """Tiny SmolLM3 (layers without RoPE) and Phi-3 (the fused projections, RoPE on 96 of 128
    dimensions, LongRoPE) at 4 slices (LLAMA_S4)."""
    _check(model, wf, path=tmp_path)


def _check(model, wf, chash=True, fill=True, xpost=False, path=None, swg=False):
    from opentpu.llm.qwen3 import Engine
    if model in ("q27", "q27f"):
        from dataclasses import replace
        from test_qwen35 import _tiny27
        _, W, spec, _ = _tiny27()
        spec = replace(spec, dn_fold=model == "q27f")
    elif model in LLAMA_S4:
        from test_llama import _tiny
        _, _, spec, W = _tiny(model, path, **LLAMA_S4[model])
    else:
        W, spec = _tiny_qwen3_s4() if model == "qwen3" else _tiny_qwen35_s4()
    lk = {"lookup": True} if model in LLAMA_S4 else {}         # (an int8 embedding's tables)
    need = spec.image(sim_board_config(DRAM_BYTES=1 << 40, S=4, **WG8), 256, wformat=wf,
                      **lk).nbytes
    cfg = sim_board_config(DRAM_BYTES=1 << max(22, (need - 1).bit_length() + 1), S=4, CHASH=chash,
                           **WG8)
    assert cfg.GALL and not cfg.PAIR
    isa = Engine(spec, W, cap=256, cfg=cfg, wformat=wf)
    tr = _transport(cfg.DRAM_BYTES, **({} if chash else {"CHASH": 0}),
                    **({} if fill else {"NAT_FILL": 0}), NAT_XPOST=int(xpost),   # (the target's 1)
                    **({"NAT_SWG": 1} if swg else {}))
    brd = Engine(spec, W, cap=256, cfg=cfg, wformat=wf,
                 backend=lambda c, imgs: BoardBackend(c, imgs, transport=tr))
    info = brd.backend.info
    assert (info["cfg"]["S"], info["cfg"]["channels"], info["cfg"]["WG"]) == (4, 8, 8)
    assert device_config(info).WG == 8 and device_config(info).CHASH == chash
    assert brd.backend.board.chash == chash
    assert not (info.get("caps") or {}).get("pair") and device_config(info).GALL
    toks = {"qwen3": (11, 222), "q27": (11, 222, 333, 444),
            "q27f": (11, 222, 333, 444)}.get(model, (11, 222, 333, 444, 555))
    toks = toks[:3] if model in LLAMA_S4 else toks
    for tok in toks:
        a, b = isa.step(tok), brd.step(tok)
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
        assert brd.stats[-1]["instructions"] == isa.stats[-1]["instructions"]
    n = isa.image.nbytes
    for s in range(4):
        assert np.array_equal(brd.backend.read(s, 0, n), isa.backend.machine.slices[s].dram[:n])


@pytest.mark.parametrize("fold", [False, True], ids=["plain", "dn_fold"])
def test_tiny_27b_shape_wg_decode_on_board_model(have_verilator, fold):
    """The WG targets' decode as the card runs it: test_qwen35's tiny 27B shape compiled for
    targets/bcu1525-x2.toml's Config (DSTEP's pair, no VOP OUTER, the sequencer's wait entries,
    the streams beside the VPU: SE_OWN, a DeltaNet burst a pair, a slice's gates in one decay |
    beta tile) on the board model built at the WG targets' parameters (its rtl_params: SE_R 1, DSTEP2,
    OUTER 0, SEQ_WC, the 128K-word TMEM, the chunk FIFO 2,048 deep in UltraRAM; port W's beats
    posted, NAT_XPOST 1), through a full turn of the convolution window: the logits, every
    slice's instruction count and DRAM the ISA simulator's. dn_fold: the same with Spec.dn_fold
    (the DeltaNet's rounding folds: Image.build's folded weights and the kernels' constants, the
    state log2(e) S that the DSTEP pairs carry), whose logits are the unfolded image's to
    rounding (cosine 0.99935 at the least; the folded kernels on unfolded weights: 0.005)."""
    from opentpu import isa as I
    from opentpu import target as T
    from opentpu.llm.qwen3 import Engine
    from test_qwen35 import _cos, _tiny27
    _, W, spec, _ = _tiny27()
    spec = dataclasses.replace(spec, dn_fold=fold)
    t = T.load("bcu1525-x2")
    # (BHW 0: the window is the DIMM pairs', tb_bcuwg's; this model's channels have none)
    need = spec.image(T.config(t, BHW=0), 256, wformat="fp4").nbytes
    cfg = T.config(t, BHW=0, DRAM_BYTES=1 << max(22, (need - 1).bit_length() + 1))
    assert cfg.DSTEP and cfg.DSTEP2 and cfg.SE_OWN and cfg.SEQ_WC and not cfg.OUTER
    rp = T.rtl_params(t)
    tr = _transport(cfg.DRAM_BYTES, NAT_XPOST=1,
                    **{k: rp[k] for k in ("TMEM_WORDS", "SEQ_WC", "OUTER", "DSTEP2")})
    p = tr.sim_build()[2]
    for k in ("MCOLS", "WG", "TMEM_WORDS", "SEQ_WC", "OUTER", "DSTEP2", "SE_R", "FIFO_DEPTH",
              "MXU_FIFO_STYLE", "TMEM_IMPL", "NAT_WPOST"):
        assert str(p.get(k)).strip('"') == str(rp[k]), (k, p.get(k), rp[k])
    assert p["NAT_XPOST"] == 1
    isa = Engine(spec, W, cap=256, cfg=cfg, wformat="fp4")
    brd = Engine(spec, W, cap=256, cfg=cfg, wformat="fp4",
                 backend=lambda c, imgs: BoardBackend(c, imgs, transport=tr))
    assert isa.image.dn_fold == fold
    ref = Engine(dataclasses.replace(spec, dn_fold=False), W, cap=256, cfg=cfg, wformat="fp4")
    dc = device_config(brd.backend.info)
    assert (dc.S, dc.WG, dc.DSTEP, dc.DSTEP2, dc.SE_OWN, dc.OUTER) == (4, 8, True, True, True,
                                                                        False)
    fl = [i.flags for q in isa.image.compile_step(3) for i in q if i.op == I.DSTEP]
    assert fl and all(f & I.F_DPAIR for f in fl)
    for tok in (11, 222, 333, 444, 555):
        a, b = isa.step(tok), brd.step(tok)
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32)), isa.pos
        assert brd.stats[-1]["instructions"] == isa.stats[-1]["instructions"]
        assert _cos(a, ref.step(tok)) > 0.998, isa.pos
    n = isa.image.nbytes
    for s in range(4):
        assert np.array_equal(brd.backend.read(s, 0, n), isa.backend.machine.slices[s].dram[:n])
