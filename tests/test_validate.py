"""tools/validate.py: the CPU golden (Hugging Face with openTPU's quantization) against the ISA
simulator on tiny random models of each port, and the comparisons that judge a device: the
golden's weights are the image's, a 1-ulp or one-token difference between two devices fails,
and so does a device whose logits stray from the golden."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from opentpu.isasim import board_config
from opentpu.llm.qwen3 import Engine

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("validate", ROOT / "tools/validate.py")
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)

PROMPTS = [[17, 401, 33, 870, 5, 912, 260, 48, 731],
           [3, 640, 211, 97, 455, 802, 19, 333, 506, 71, 118, 990, 264, 7]]
EOS = (999,)


def _norms(m, zero_centered=False, k_norm=None):
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:     # Qwen3.5: zero-centered, but the DeltaNet output norm is plain
                base = 0.0 if zero_centered and not n.endswith("linear_attn.norm.weight") else 1.0
                if k_norm is not None and "k_norm" in n:   # Gemma 4: near E2B's (test_gemma4)
                    p.copy_(k_norm + 0.01 * torch.randn_like(p))
                else:
                    p.copy_(base + 0.1 * torch.randn_like(p))
    return m


def _qwen3():
    from opentpu.llm.qwen3 import Spec
    hc = transformers.Qwen3Config(hidden_size=256, num_hidden_layers=2, num_attention_heads=4,
                                  num_key_value_heads=2, head_dim=128, intermediate_size=512,
                                  vocab_size=1000, rms_norm_eps=1e-6, rope_theta=1e6,
                                  tie_word_embeddings=True, max_position_embeddings=4096)
    return transformers.Qwen3ForCausalLM(hc), Spec(256, 2, 4, 2, 128, 512, 1000, eos=EOS), {}


def _lfm2():
    from opentpu.llm.lfm2 import Spec
    kinds = ("conv", "attn", "conv", "attn", "conv")
    hc = transformers.Lfm2Config(
        hidden_size=256, num_hidden_layers=len(kinds), num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=512, vocab_size=1000, norm_eps=1e-5,
        layer_types=["full_attention" if k == "attn" else "conv" for k in kinds],
        conv_L_cache=3, conv_bias=False, block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True, max_position_embeddings=4096,
        rope_parameters={"rope_type": "default", "rope_theta": 1e6})
    return transformers.Lfm2ForCausalLM(hc), Spec(256, kinds, 4, 2, 64, 512, 1000, eos=EOS), {}


def _qwen35():
    from opentpu.llm.qwen35 import Spec
    kinds = ("linear", "linear", "attn")
    hc = transformers.Qwen3_5TextConfig(
        hidden_size=256, num_hidden_layers=len(kinds), num_attention_heads=8,
        num_key_value_heads=2, head_dim=256, intermediate_size=512, vocab_size=1000,
        layer_types=["full_attention" if k == "attn" else "linear_attention" for k in kinds],
        linear_num_key_heads=4, linear_num_value_heads=4, linear_key_head_dim=128,
        linear_value_head_dim=128, linear_conv_kernel_dim=4, tie_word_embeddings=True,
        max_position_embeddings=4096, rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25})
    return (transformers.Qwen3_5ForCausalLM(hc),
            Spec(256, kinds, 8, 2, 256, 64, 4, 128, 128, 512, 1000, lin_kheads=4, eos=EOS),
            {"zero_centered": True})


def _gemma4():
    import json
    import tempfile
    from dataclasses import replace

    from opentpu.llm.gemma4 import Spec
    S, F = "sliding_attention", "full_attention"
    kinds = (S, S, F, S, F)          # (the last two read layers 1's and 2's K / V)
    hc = transformers.Gemma4TextConfig(
        hidden_size=256, num_hidden_layers=len(kinds), num_attention_heads=8,
        num_key_value_heads=1, head_dim=128, global_head_dim=256, intermediate_size=512,
        vocab_size=1000, vocab_size_per_layer_input=1000, hidden_size_per_layer_input=128,
        layer_types=list(kinds), num_kv_shared_layers=2, use_double_wide_mlp=True,
        sliding_window=512, final_logit_softcapping=30.0, max_position_embeddings=4096)
    m = transformers.models.gemma4.modeling_gemma4.Gemma4ForCausalLM(hc)
    with torch.no_grad():
        for n, b in m.named_buffers():
            if n.endswith("layer_scalar"):
                b.copy_(0.5 + torch.rand_like(b))
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "config.json").write_text(json.dumps(hc.to_dict()))
        spec = Spec.from_hf(d)
    return m, replace(spec, eos=EOS), {"k_norm": 0.13}


MODELS = {"qwen3": _qwen3, "lfm2": _lfm2, "qwen35": _qwen35, "gemma4": _gemma4}


def _tiny(name):
    torch.manual_seed(0)
    m, spec, kw = MODELS[name]()
    m = _norms(m.float().eval(), **kw)
    return m, {k: v.float().numpy().copy() for k, v in m.state_dict().items()}, spec


def _runs(spec, W, wformat="int8", n=8):
    eng = Engine(spec, W, cap=256, cfg=board_config(DRAM_BYTES=1 << 26), wformat=wformat)
    return eng, V.device_runs(eng, PROMPTS, n, spec.eos)


@pytest.fixture(scope="module")
def qwen3_int8():
    m, W, spec = _tiny("qwen3")
    eng, runs = _runs(spec, W)
    return m, W, spec, eng, runs


def _top1_where_clear(golden, runs, noise):
    """Per prompt, teacher forced on the device's tokens against the quantized golden: the steps
    whose top-1 leads its runner-up by more than `noise`, and whether the device's top-1 is the
    golden's at each step."""
    golden.set("quant")
    res = []
    for ids, r in zip(PROMPTS, runs):
        gl = golden.logits(list(ids) + r["tokens"][:-1])[len(ids) - 1:]
        v = min(gl.shape[1], r["logits"].shape[1])
        gl, dl = gl[:, :v], golden.device_logits(r["logits"][:, :v])
        top2 = np.sort(gl, -1)[:, -2:]
        res.append((top2[:, 1] - top2[:, 0] > noise, dl.argmax(-1) == gl.argmax(-1)))
    return res


@pytest.mark.parametrize("name,wformat", [("qwen3", "int8"), ("qwen3", "fp4"),
                                          ("lfm2", "fp4"), ("qwen35", "fp4"),
                                          ("gemma4", "int8"), ("gemma4", "fp4")])
def test_quantized_golden_follows_the_isa_simulator(name, wformat):
    """The ISA simulator's greedy run against the goldens, teacher forced: the quantized golden
    (W+A) agrees on every top-1 but near ties and is at least 5x closer in KL than fp32 (about
    150x with int8 weights here, over 1000x with fp4; a rounding difference that flips an int8
    value is what is left, e.g. the tiny Qwen3.5's int8 run: 9x). A near tie is a step whose
    top two logits are closer than the most the golden's own 1-ulp noise moves a logit (the
    floor's max |difference|): it may go either way (the tiny Gemma 4 has steps 4e-4 apart),
    and so may the greedy continuation from it (the golden's tokens part from the device's
    only at a top-1 that differs on the same context). Gemma 4 (sliding and global
    layers, KV-shared ones, per-layer inputs): its embedding rows the head's, its PLE rows the
    device's records, its formats the device image's."""
    m, W, spec = _tiny(name)
    eng, runs = _runs(spec, W, wformat)
    golden = V.Golden(m, spec, wformat, image=V.image_formats(eng.image))
    gold = V.against_golden(golden, PROMPTS, runs, ["quant", "fp32"], 8, spec.eos)
    q = V.summary([g["goldens"]["quant"] for g in gold])
    f = V.summary([g["goldens"]["fp32"] for g in gold])
    noise = V.summary([g["pairs"]["floor"] for g in gold])["max_abs"]
    for clear, same in _top1_where_clear(golden, runs, noise):
        assert same[clear].all()
    assert q["top1"] >= V.MIN_TOP1 and q["min_cos"] > 0.999
    assert q["kl_mean"] * 5 < f["kl_mean"]
    assert V.summary([g["pairs"]["fp32"] for g in gold])["kl_mean"] > q["kl_mean"]


class _DeviceCodes:
    """The int8 groups of a device run (every fp32.quantize call: the ISA simulator's QACT / QST
    and the image's host-side ones), each as r = x * 127 / amax and its codes, found again by
    content: the place of the largest |r|, then the smallest max |r - r_device| (under 0.25; a
    group the device did not quantize is 25 or more away)."""

    def __init__(self):
        self.parts, self.unmatched = [], 0

    def record(self, x, axis, q):
        x = np.moveaxis(np.asarray(x, np.float32), axis, -1)
        n = x.shape[-1]
        x = x.reshape(-1, n).astype(np.float64)
        a = np.abs(x).max(-1, keepdims=True)
        self.parts.append((x * 127 / np.where(a == 0, 1, a),
                           np.moveaxis(q, axis, -1).reshape(-1, n)))

    def index(self):
        self.R, self.Q, self.at = {}, {}, {}
        for n in {r.shape[1] for r, _ in self.parts}:
            self.R[n] = np.concatenate([r for r, _ in self.parts if r.shape[1] == n])
            self.Q[n] = np.concatenate([q for r, q in self.parts if r.shape[1] == n])
            k = np.abs(self.R[n]).argmax(-1)
            self.at[n] = {int(i): np.nonzero(k == i)[0] for i in np.unique(k)}

    def find(self, r):
        """(max |r - r_device|, the device's codes) of the group r [n] (a longer device group:
        its first n), or (inf, None)."""
        k = int(np.abs(r).argmax())
        for every in (False, True):           # the same largest place, else every group
            best = (0.25, None)
            for m, R in self.R.items():
                if m < len(r):
                    continue
                rows = np.arange(len(R)) if every else self.at[m].get(k, np.arange(0))
                if len(rows):
                    d = np.abs(R[rows, :len(r)] - r).max(-1)
                    i = int(np.argmin(d))
                    if d[i] < best[0]:
                        best = (d[i], self.Q[m][rows[i], :len(r)])
            if best[1] is not None:
                return best
        return np.inf, None

    def force(self, r, q):
        """q, the emulation's codes of the groups r [g, n], with the device's where r is within
        its group's distance from the device's (+ 1e-3: r_device is recomputed in float64) of a
        rounding tie, the only places the two can round apart."""
        q = q.copy()
        for g in np.nonzero(np.abs(r).max(-1) > 0)[0]:
            d, qd = self.find(r[g])
            if qd is None:
                self.unmatched += 1
                continue
            near = np.abs(r[g] - np.rint(r[g])) > 0.5 - d - 1e-3
            q[g, near] = qd[near]
        return q


def _teacher_forced(mp, mod, dev):
    """The emulation's int8 points (opentpu.llm.qwen3's _fake_q and _v_parts, which its _pv and
    _fake_w call, and the port module's imports of them) with the device's codes near ties:
    their own results but a forced code's value (code * scale)."""
    from opentpu.llm import qwen3 as Q3
    fake_q, v_parts = Q3._fake_q, Q3._v_parts

    def fq(x, D=128):
        out = np.asarray(fake_q(x, D), np.float64)
        xb = np.asarray(x, np.float64).reshape(-1, D)
        a = np.abs(xb).max(-1, keepdims=True)
        s = np.where(a == 0, 1, a / 127)
        q = np.rint(out.reshape(-1, D) / s)
        return out + ((dev.force(xb / s, q) - q) * s).reshape(out.shape)

    def vp(v):
        q, sc = v_parts(v)
        a = np.abs(v).max(-1, keepdims=True)
        r = (v / np.where(a == 0, 1, a / 127)).reshape(-1, v.shape[-1])
        return dev.force(r, q.reshape(r.shape)).reshape(q.shape), sc
    for m in {Q3, mod}:
        mp.setattr(m, "_fake_q", fq)
        mp.setattr(m, "_v_parts", vp)


def _unfolded_pv(pp, Vq, vs, D: int = 128):
    """P.V with V's per-token scales applied after P is quantized (a wrong emulation)."""
    from opentpu.llm import qwen3 as Q3
    ppad = np.zeros(-(-len(pp) // D) * D)
    ppad[:len(pp)] = pp
    return (Q3._fake_q(ppad, D)[:len(pp)] * vs) @ Vq


@pytest.mark.parametrize("name", ["qwen3", "lfm2", "qwen35"])
def test_emulated_logits_follow_the_isa_simulator(name, monkeypatch):
    """The ports' emulated_logits (float64, the device's quantization points) are the ISA
    simulator's logits to fp32 rounding with fp4 weights on these tiny models: P.V as the device
    takes it, V's per-token scales folded into P before P is quantized (QACT CSCALE). Without
    the fold the emulation is about 1% off, which the test asserts too. Teacher forced at the
    rounding ties: an int8 value whose float64 and fp32 sides lie across a tie takes the
    device's code; unforced, it rounds the other way now and then and the steps after it stray
    by 0.1 - 2.5% (on main 7 / 13 / 16 of 32 prompts over 16 model seeds of qwen3 / lfm2 /
    qwen35, which ones depending on the host's float64 summation order). Forced, every prompt
    of 16 seeds is within 4e-7 on the Mac and omarchy, and a 1% error in K or in V's scales,
    or K or V left unrounded, is 0.7 - 2.3% off. Every int8 group finds its device group but the
    head's inputs at the prompts' earlier positions (the device takes no logits there)."""
    import importlib
    from opentpu import fp32 as F
    from opentpu.llm import qwen3 as Q3
    _, W, spec = _tiny(name)
    dev, quantize = _DeviceCodes(), F.quantize

    def recorded(x, axis=-1):
        q, s = quantize(x, axis)
        dev.record(x, axis, q)
        return q, s
    with monkeypatch.context() as mp:
        mp.setattr(F, "quantize", recorded)
        _, runs = _runs(spec, W, "fp4")
    dev.index()
    mod = importlib.import_module(type(spec).__module__)
    _teacher_forced(monkeypatch, mod, dev)

    def errors():
        err = []
        for p, r in zip(PROMPTS, runs):
            seq = list(p) + list(r["tokens"][:-1])
            emu = mod.emulated_logits(spec, W, seq, wformat="fp4")[len(p) - 1:]
            err.append(np.abs(r["logits"] - emu).max() / np.abs(emu).max())
        return np.array(err)
    err = errors()
    assert (err < 1e-5).all(), err
    assert dev.unmatched <= sum(len(p) - 1 for p in PROMPTS) * spec.hidden // 128, dev.unmatched
    for m in {Q3, mod}:
        monkeypatch.setattr(m, "_pv", _unfolded_pv)
    err = errors()
    assert (err > 1e-3).all(), err


@pytest.mark.parametrize("fmt", ["int8", "fp4"])
def test_golden_weights_are_the_images(fmt):
    """device_weight of the tied LM head equals the head rows the Engine's image holds,
    dequantized as the MXU reads them (int8 q * s; 4-bit S * m * code)."""
    from opentpu import quant as Q
    m, W, spec = _tiny("qwen3")
    eng = Engine(spec, W, cap=256, cfg=board_config(DRAM_BYTES=1 << 24), head_format=fmt)
    im, dram = eng.image, eng.backend.machine.slices[0].dram
    V_, H, D = spec.vocab, spec.hidden, eng.cfg.D
    rb = Q.row_bytes(H, fmt, D)
    rows = dram[im.head[0]:im.head[0] + V_ * rb].reshape(V_, rb)
    sc = dram[im.head[1]:im.head[1] + 4 * V_ * (H // D)].view(np.uint32).reshape(V_, H // D)
    if fmt == "int8":
        want = (rows.view(np.int8).reshape(V_, H // D, D).astype(np.float32)
                * sc.view(np.float32)[..., None]).reshape(V_, H)
    else:
        want = Q.dequantize_w4(rows[:, :H // 2], sc, fmt, D).astype(np.float32)
    got = V.device_weight(W["model.embed_tokens.weight"], fmt, D)
    assert np.array_equal(got, want)
    assert V.Golden(m, spec, "int8", fmt).formats["lm_head.weight"] == fmt


def test_golden_takes_the_devices_k_smoothing():
    """Qwen3's K outliers planted (k_norm gains 60 on two RoPE pairs, their q gains 0.01, as
    test_qwen3's): the device smooths K's channels into the q_norm / k_norm gains
    (qwen3.qk_gains) and stays near the fp32 golden (KL 3e-5; 1.1e-3 unsmoothed), and the
    quantized golden smooths them too: it follows the ISA simulator as on the plain model (KL
    1e-14 here; a golden that did not smooth K would be 1e-3 away). The weights mode takes the
    device's gains, bit-identical to the checkpoint's there (sigma, a power of two, changes
    only how K rounds); fp32 is the checkpoint (the fp32 reference)."""
    from dataclasses import replace

    from opentpu.llm.qwen3 import reference_logits

    def planted():
        m, _, spec = _tiny("qwen3")
        with torch.no_grad():
            for n, p in m.named_parameters():
                if n.endswith(("q_norm.weight", "k_norm.weight")):
                    p[[5, 69, 30, 94]] = 0.01 if "q_norm" in n else 60.0
        return m, spec
    m, spec = planted()
    W = {k: v.float().numpy().copy() for k, v in m.state_dict().items()}
    eng, runs = _runs(spec, W)
    g = V.Golden(m, spec, "int8", image=V.image_formats(eng.image))
    gold = V.against_golden(g, PROMPTS, runs, ["quant", "fp32"], 8, spec.eos)
    q = V.summary([x["goldens"]["quant"] for x in gold])
    assert q["top1"] >= V.MIN_TOP1 and q["min_cos"] > 0.999 and q["kl_mean"] < 1e-6
    assert V.summary([x["goldens"]["fp32"] for x in gold])["kl_mean"] < 1e-4
    g.set("weights")
    smooth = g.logits(PROMPTS[1])
    g.set("fp32")
    assert np.abs(g.logits(PROMPTS[1]) - reference_logits(spec, W, PROMPTS[1])).max() < 1e-4
    plain = V.Golden(planted()[0], replace(spec, qk_smooth=False), "int8")
    plain.set("weights")
    assert np.array_equal(smooth, plain.logits(PROMPTS[1]))



def test_two_devices_must_agree_bit_for_bit(qwen3_int8):
    """agreement(): equal runs pass; one logit 1 ulp off fails at that step, a different token
    fails at that token (the steps after it are not compared)."""
    *_, runs = qwen3_int8
    same = V.agreement(runs, [dict(r, logits=r["logits"].copy()) for r in runs])
    assert same["pass"] and same["bit_exact"] == 2 and same["max_ulp"] == 0
    other = [dict(r, logits=r["logits"].copy(), tokens=list(r["tokens"])) for r in runs]
    lg = other[1]["logits"]
    lg[3, 17] = np.nextafter(lg[3, 17], np.float32(np.inf))
    a = V.agreement(runs, other)
    assert not a["pass"] and a["same_tokens"] == 2 and a["bit_exact"] == 1
    assert a["prompts"][1]["first_step"] == 3 and a["max_ulp"] == 1
    other[0]["tokens"][5] += 1
    a = V.agreement(runs, other)
    assert a["prompts"][0]["token_diff"] == 5 and a["prompts"][0]["steps"] == 6


def test_a_device_off_the_golden_fails(qwen3_int8, capsys):
    """The report passes the ISA run and fails the same run with an error injected into a
    step's logits (the golden's top-1 there is no longer the device's), and a run saved with
    --save reads back for --against."""
    m, W, spec, eng, runs = qwen3_int8
    golden = V.Golden(m, spec)
    gold = V.against_golden(golden, PROMPTS, runs, ["quant"], 8, spec.eos)
    res = V.report(None, ["a", "b"], PROMPTS, runs, gold, ["quant"], "isa", None, None,
                   (V.MIN_TOP1, V.KL_RATIO, V.MAX_KL))
    assert res["pass"] and "PASS" in capsys.readouterr().out
    bad = [dict(r, logits=r["logits"].copy()) for r in runs]
    for st in range(4):
        lg = bad[1]["logits"][st]
        lg[(np.argmax(lg) + 1) % len(lg)] += 4 * np.ptp(lg)
    gold = V.against_golden(golden, PROMPTS, bad, ["quant"], 8, spec.eos)
    res = V.report(None, ["a", "b"], PROMPTS, bad, gold, ["quant"], "isa", None, None,
                   (V.MIN_TOP1, V.KL_RATIO, V.MAX_KL))
    assert not res["pass"] and res["goldens"]["quant"]["top1"] < V.MIN_TOP1


def test_saved_runs_read_back(qwen3_int8, tmp_path):
    *_, runs = qwen3_int8
    meta = {"model": "tiny", "wformat": "int8", "head_format": None, "formats": None,
            "tokens": 8, "prompts": 2, "resident": False}
    f = str(tmp_path / "run.npz")
    V.save_run(f, runs, {**meta, "backend": "isa", "cfg": "c", "arch": "a"}, PROMPTS)
    back, m = V.load_run(f, meta, PROMPTS)
    assert m["backend"] == "isa" and V.agreement(runs, back)["pass"]
    with pytest.raises(SystemExit):
        V.load_run(f, meta, [PROMPTS[0], PROMPTS[1][:-1]])
    with pytest.raises(SystemExit):
        V.load_run(f, {**meta, "wformat": "fp4"}, PROMPTS)


def test_the_card_path_on_a_fake_card():
    """run_device's board path (an Engine on BoardBackend; prefill runs and streamed step
    logits) on a fake card that computes with the ISA simulator, against run_device's isa path
    in the card's configuration (--backend board --against isa): bit for bit."""
    from argparse import Namespace

    from conftest import IsaCard
    from opentpu.host.board import BoardBackend, sim_config
    from opentpu.llm.qwen3 import HEAD_CHUNK, PREFILL_ROWS
    _, W, spec = _tiny("qwen3")
    cfg = sim_config(spec, 256)
    card = IsaCard(cfg, None, 4 * min(HEAD_CHUNK, cfg.TMEM_WORDS // 8))
    # the logits show late, in pieces (IsaCard)
    card.late = (spec.image(cfg, 256, 1, PREFILL_ROWS).io["logits"], 4 * spec.vocab)

    def factory(c, imgs):
        return BoardBackend(c, imgs, transport=card, model="tiny")
    a = Namespace(wformat="int8", head_format=None, resident=False, tokens=6, cfg="board")
    runs, c, _ = V.run_device("board", spec, None, 256, PROMPTS, a, card=(factory, cfg), W=W)
    isa, c2, _ = V.run_device("isa", spec, None, 256, PROMPTS, a, card=(factory, cfg), W=W)
    assert V.arch(c) == V.arch(c2)
    assert V.agreement(runs, isa)["pass"] and sum(len(r["tokens"]) for r in runs) == 12


def _checkpoint(name, path) -> list:
    """A tiny model saved by Hugging Face with a word-level tokenizer ("t17" is token 17) in
    `path`: PROMPTS as text."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    m, _, _ = _tiny(name)
    m.config.eos_token_id = 999
    m.save_pretrained(path)
    tk = Tokenizer(models.WordLevel({f"t{i}": i for i in range(1000)}, unk_token="t0"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    transformers.PreTrainedTokenizerFast(tokenizer_object=tk).save_pretrained(path)
    return [" ".join(f"t{t}" for t in p) for p in PROMPTS]


def test_the_command_line_on_a_tiny_checkpoint(tmp_path, capsys):
    """main() end to end on a tiny Qwen3 checkpoint (and a word-level tokenizer) on disk: the
    ISA run saved without the golden, then a run against it with the golden: PASS, exit 0, the
    two runs bit for bit, --json written; a saved run of other prompts is refused."""
    import json
    prompts = _checkpoint("qwen3", tmp_path)
    run = str(tmp_path / "isa.npz")
    base = ["--model", str(tmp_path), "--tokens", "6"]
    assert V.main(base + ["--no-golden", "--save", run] + prompts) == 0
    assert V.main(base + ["--against", run, "--json", str(tmp_path / "r.json")] + prompts) == 0
    out = capsys.readouterr().out
    assert "logits bit-exact in 2/2" in out and out.rstrip().endswith("PASS")
    res = json.loads((tmp_path / "r.json").read_text())
    assert res["pass"] and res["against"]["pass"] and res["goldens"]["quant"]["top1"] == 1.0
    assert res["prompts"][1]["ids"] == PROMPTS[1]
    with pytest.raises(SystemExit):
        V.main(base + ["--against", run] + prompts[:1])


def test_the_command_line_on_a_tiny_gemma4(tmp_path):
    """main() on a tiny Gemma 4 checkpoint (text-only, as Hugging Face saves it) with fp4
    weights: the device image's formats reach the golden (--json's image), Hugging Face loads
    without its PLE table (the golden reads the rows it needs from the checkpoint): PASS (top-1
    agreement at least MIN_TOP1, KL near the floor; a near tie may flip, see above)."""
    import json
    prompts = _checkpoint("gemma4", tmp_path)
    assert V.main(["--model", str(tmp_path), "--tokens", "6", "--wformat", "fp4", "--json",
                   str(tmp_path / "r.json")] + prompts) == 0
    res = json.loads((tmp_path / "r.json").read_text())
    assert res["image"]["head"] == "fp4" and res["image"]["ple_table"] == "int8"
    assert res["formats"]["model.per_layer_model_projection.weight"] == "fp4"
    assert res["pass"]
