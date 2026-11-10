"""The compiler, the kernels and the models across architectures (docs/targets.md): every
target's in targets/ (opentpu.target.config, the hardware's truth: ypcb-00338, nangate45 (the
card's at 8 ACT RAM rows), bcu1525's two slices, bcu1525-x2's four (where the tiny models are
refused: their 2 KV heads), nangate45-small's D = 32 / 2 columns / 4 lanes / 16K-word TMEM /
1K-instruction IMEM), the simulators' design point, and two grown versions of
the small one (D = 64 and 128, 16K-word IMEM, DSTEP / STREAM / PAIR), which reach the paths the
small target refuses earlier: the one-hot gathers in parts, a decode step placed best-fit, DeltaNet
heads several D wide. On the ISA simulator; nothing here is tied to the card's sizes.

Where a model does not map onto an architecture its image refuses it naming the limit."""
import inspect
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from opentpu import Config
from opentpu import isa as I
from opentpu import target as T
from opentpu.compiler import CompileError
from opentpu.isasim import Machine, SimError, design_config
from opentpu.kernels import attention_decode, attention_layer, mlp
from opentpu.kernels import gather as GA
from opentpu.kernels.deltanet import gated_deltanet_step
from opentpu.llm import gemma4 as G4
from opentpu.llm import generate as GN
from opentpu.runtime import launch

from conftest import rel

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

FEAT = dict(PAIR=True, DSTEP=True, STREAM=True)


def _archs() -> dict:
    """name -> Config: the targets' (those of one arch under one name, joined by '='), the
    design point, the grown small ones."""
    out: dict = {}
    for n in T.names():
        cfg = T.config(n)
        same = next((k for k, c in out.items() if c == cfg), None)
        if same is None:
            out[n] = cfg
        else:
            out[f"{same}={n}"] = out.pop(same)
    out["design"] = design_config()
    small = T.config("nangate45-small")
    out["small-d64+"] = replace(small, D=64, ACT_BLOCKS=64, IMEM_WORDS=16384, **FEAT)
    out["small-d128+"] = replace(small, D=128, ACT_BLOCKS=32, IMEM_WORDS=16384, **FEAT)
    return out


ARCHS = _archs()
SMALL = "nangate45-small"


def _key(name: str) -> str:
    return next(k for k in ARCHS if name in k.split("="))


# ------------------------------------------------------------------------------ kernels
@pytest.mark.parametrize("name", list(ARCHS))
def test_kernels(name):
    """MLP, attention decode and an attention layer (heads over the slices) on each
    architecture, against float64 math; a Gated DeltaNet head step (VPU form: RDOT and OUTER, or
    its ISA form on a VPU without OUTER) where its two states fit TMEM."""
    import test_kernels as K
    cfg = replace(ARCHS[name], DRAM_BYTES=1 << 24)
    S, D = cfg.S, cfg.D
    H = max(128, S * D)
    rng = np.random.default_rng(1)
    args, want = K.mlp_args(rng, M=min(cfg.MCOLS, 4), H=H, Fd=2 * H)
    assert rel(launch(mlp, cfg, **args).outputs["out"], want) < 0.02
    d, block = max(64, D), max(32, D)
    args, want = K.attn_args(rng, Hq=4 * S, Hkv=2 * S, d=d, T=3 * block - 5, cap=4 * block,
                             block=block)
    assert rel(launch(attention_decode, cfg, **args).outputs["out"], want) < 0.03
    args, (want, knew, _) = K.layer_args(rng, S, H=H, Hq=2 * S, Hkv=S, d=d, pos=40, cap=block * 2)
    args["block"] = block
    r = launch(attention_layer, cfg, **args)
    assert rel(r.outputs["out"], want) < 0.01
    import test_vops as V
    dk = 2 * D if D < 128 else D
    if 2 * dk * dk < cfg.TMEM_WORDS:
        args, (Sw, ow) = V.deltanet_args(np.random.default_rng(7), dk=dk, dv=dk)
        o = launch(gated_deltanet_step, cfg, **args, fused=cfg.OUTER).outputs
        assert np.max(np.abs(o["state_out"].transpose(0, 2, 1) - Sw)) < 1e-5 * np.abs(Sw).max()


def test_a_program_over_imem_does_not_run():
    """The ISA simulator holds a program to IMEM's instructions, as the board's loader does."""
    cfg = Config(S=1, IMEM_WORDS=8 * 4)
    prog = [I.Instr(I.NOP)] * 4 + [I.halt()]
    with pytest.raises(SimError, match="5 instructions, IMEM holds 4"):
        Machine(cfg, [prog], [None])
    m = Machine(cfg, [prog[-4:]], [None])
    with pytest.raises(SimError, match="IMEM holds 4"):
        m.load([prog])


def test_sizes_follow_the_config():
    """The generate loop's program slots hold two programs of IMEM (the card's 256 KiB; a
    small IMEM's less); Gemma 4's MLP chunk scales with TMEM (the card's 768 at most) and its
    formats follow the fit to the configuration's DRAM (the card's 4 GiB at most)."""
    assert GN.prog_bytes(T.config("ypcb-00338").IMEM_WORDS) == 1 << 18
    assert GN.prog_bytes(T.config(SMALL).IMEM_WORDS) == 1 << 16
    assert GN.prog_bytes(design_config().IMEM_WORDS) == 1 << 19    # a split part is 8K
    assert G4._mlp_chunk(10240, 128, 256) == 512                     # E4B's fp4 (the card)
    assert G4._mlp_chunk(10240, 128, 256, 16384) == 256
    assert G4._mlp_chunk(6144, 128, 128, 1 << 16) == G4._mlp_chunk(6144, 128, 128)
    assert G4._fit_bytes(T.config("ypcb-00338")) == 1 << 32
    assert G4._fit_bytes(design_config(DRAM_BYTES=1 << 40)) == 1 << 32
    assert G4._fit_bytes(T.config(SMALL)) == 1 << 28


# ------------------------------------------------------------------------------ models
def _tiny(name):
    if name == "gemma4":
        import tempfile

        import test_gemma4
        _, W, spec = test_gemma4._model(Path(tempfile.mkdtemp()))
        return W, spec
    from test_autodecode import _tiny as tiny
    return tiny(name)


@pytest.fixture(scope="module")
def models():
    return {}


def _model(models, name):
    if name not in models:
        models[name] = _tiny(name)
    return models[name]


# (model, arch) -> what its image says where the model does not map onto it
REFUSED = {("qwen35", SMALL): "DeltaNet on the VPU.*inner dimension exceeds ACT RAM",
           ("gemma4", SMALL): "inner dimension exceeds ACT RAM",
           ("gemma4", "bcu1525"): "one slice", ("gemma4", "design"): "one slice",
           # Gemma 4's single KV head is not split over slices (S = 1 only: on the BCU its
           # bitstream is bcu1525-wg's)
           ("gemma4", "bcu1525-s2"): "one slice",
           # two BCU1525s' 4 slices (bcu1525-x2): a model's hidden size a multiple of S x D,
           # its KV heads of S (Qwen3.5: or a divisor of S, KVDesc.kv_heads)
           ("qwen3", "bcu1525-x2"): r"hidden 256 % S\*D; n_kv 2 % S",
           ("lfm2", "bcu1525-x2"): r"n_kv 2 % S", ("qwen35", "bcu1525-x2"): r"hidden 256 % S\*D",
           ("gemma4", "bcu1525-x2"): "one slice"}
CAP = 384           # (Gemma 4: 512, a multiple of its attention block)


def _cap(spec) -> int:
    return 512 if isinstance(spec, G4.Spec) else CAP


def _sized(spec, cfg):
    """cfg with its DRAM cut to the model's image (power of two; at most the architecture's):
    the simulator holds DRAM_BYTES per slice."""
    big = replace(cfg, DRAM_BYTES=1 << 40)
    need = spec.image(big, _cap(spec), 1, 8, lookup=True).nbytes + 4 * cfg.IMEM_WORDS
    return replace(cfg, DRAM_BYTES=min(cfg.DRAM_BYTES, 1 << max(22, (need - 1).bit_length())))


def _cos(a, b):
    return (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)


@pytest.mark.parametrize("model", ["qwen3", "lfm2", "qwen35", "gemma4"])
@pytest.mark.parametrize("name", list(ARCHS))
def test_model(models, model, name):
    """A tiny model on each architecture: a resident engine's prompt (runs of up to 8 rows,
    across the attention bucket at 256 where TMEM is small) gives the per-position engine's
    logits bit for bit, the decode loop on the device its greedy tokens; the device follows
    the quantized math at the architecture's D (emulated_logits, at the image's S-exact
    choice: Config.SEXACT), and an S-exact image (bcu1525-s2's) gives S = 1's logits bit for
    bit."""
    from opentpu.llm.qwen3 import Engine
    W, spec = _model(models, model)
    cfg = ARCHS[name]
    refused = next((m for (mo, a), m in REFUSED.items() if mo == model and a in name.split("=")),
                   None)
    if refused:
        with pytest.raises(ValueError, match=refused):
            spec.check(cfg)
        return
    cfg = _sized(spec, cfg)
    mod = sys.modules[type(spec).__module__]
    toks = [int(t) for t in np.random.default_rng(1).integers(0, 1000, 250)]
    P = len(toks) if cfg.TMEM_WORDS < 1 << 16 else 24
    a = Engine(spec, W, cap=_cap(spec), cfg=cfg, resident=True)
    b = Engine(spec, W, cap=_cap(spec), cfg=cfg)
    la, lb = a.prefill(toks[:P]), b.prefill(toks[:P])
    assert np.array_equal(la.view(np.uint32), lb.view(np.uint32))
    t0, ref = int(np.argmax(la)), []
    t = t0
    for _ in range(8):
        t = int(np.argmax(b.step(t)))
        ref.append(t)
    assert a.can_generate and a.generate_card(t0, 8, stop_ids=[]) == ref
    c = Engine(spec, W, cap=_cap(spec), cfg=cfg)
    raw = np.array([c.step(t) for t in toks[:6]])
    if getattr(c.image, "sexact", False):
        one = Engine(spec, W, cap=_cap(spec), cfg=replace(cfg, S=1))
        assert np.array_equal(raw.view(np.uint32),
                              np.array([one.step(t) for t in toks[:6]]).view(np.uint32))
    dev = raw.astype(np.float64)
    sig = inspect.signature(mod.emulated_logits).parameters
    sk = {k: v for k, v in (("S", cfg.S), ("cfg", cfg)) if k in sig}
    emu = mod.emulated_logits(spec, W, toks[:6], D=cfg.D, **sk)   # (S: the slices' QACTs;
    #                                                         cfg: S-exact's, S = 1's)
    if model == "gemma4":
        dev, emu = G4.softcap(spec, dev), G4.softcap(spec, emu)
    assert _cos(dev, emu).min() > (0.99 if model == "gemma4" else 0.999)


def test_one_hot_gathers_in_parts(models, monkeypatch):
    """The device's embedding gathers (kernels/gather.py) with the one-hot operand quantized a
    MM at a time, where it does not fit ACT RAM whole (D = 128 at 2 columns: 64 blocks,
    against 32), give the rows the whole operand gives, bit for bit: a tied int8 head's rows
    as the resident decode's embedding."""
    from opentpu.llm.qwen3 import Engine
    W, spec = _model(models, "qwen3")
    spec = replace(spec, embed="int8")
    base = replace(ARCHS["small-d128+"], DRAM_BYTES=1 << 26)
    assert GA.onehot_blocks(128, 2, "int8") > base.ACT_BLOCKS
    parts, lg = [], []
    init = GA._Parts.__init__

    def counted(self, desc):
        parts[-1] += 1
        init(self, desc)
    monkeypatch.setattr(GA._Parts, "__init__", counted)
    for cfg in (base, replace(base, ACT_BLOCKS=128, TMEM_WORDS=1 << 16)):   # (whole)
        parts.append(0)
        e = Engine(spec, W, cap=CAP, cfg=cfg, resident=True)
        assert e.resident and e.device_inputs
        lg.append(np.array([e.step(t) for t in (5, 17, 300)]))
    assert parts[0] > 0 and parts[1] == 0
    assert np.array_equal(lg[0].view(np.uint32), lg[1].view(np.uint32))


def test_a_decode_step_is_placed_best_fit_where_next_fit_fails(models, monkeypatch):
    """Qwen3.5's resident decode at a 15,872-word TMEM: its tiles placed next-fit leave the free
    words in gaps too small for a later one, best-fit fits them (compiler.trace_fit); the
    compiles that probe (prefill rows, prompt runs) keep next-fit's answer. (At 16K words
    next-fit fits now; best-fit alone fits from 15,616 to 16,128 words.)"""
    from opentpu.llm import qwen3 as Q3
    W, spec = _model(models, "qwen35")
    cfg = replace(_sized(spec, ARCHS["small-d64+"]), TMEM_WORDS=15872)
    img = spec.image(cfg, CAP, 1, 8, lookup=True)
    lo = spec.conv_k - 1
    progs, _ = img.compile_decode(1, lo)
    assert len(progs[0]) <= cfg.IMEM_WORDS // 8
    monkeypatch.setattr(Q3, "trace_fit", lambda k, c, s, bound: k.trace(c, s, bound()))
    with pytest.raises(CompileError, match="TMEM exhausted: a tile of"):
        img.compile_decode(1, lo)
