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


def _norms(m, zero_centered=False):
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:     # Qwen3.5: zero-centered, but the DeltaNet output norm is plain
                base = 0.0 if zero_centered and not n.endswith("linear_attn.norm.weight") else 1.0
                p.copy_(base + 0.1 * torch.randn_like(p))
    return m


def _qwen3():
    from opentpu.llm.qwen3 import Spec
    hc = transformers.Qwen3Config(hidden_size=256, num_hidden_layers=2, num_attention_heads=4,
                                  num_key_value_heads=2, head_dim=128, intermediate_size=512,
                                  vocab_size=1000, rms_norm_eps=1e-6, rope_theta=1e6,
                                  tie_word_embeddings=True, max_position_embeddings=4096)
    return transformers.Qwen3ForCausalLM(hc), Spec(256, 2, 4, 2, 128, 512, 1000, eos=EOS), False


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
    return transformers.Lfm2ForCausalLM(hc), Spec(256, kinds, 4, 2, 64, 512, 1000, eos=EOS), False


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
            Spec(256, kinds, 8, 2, 256, 64, 4, 128, 128, 512, 1000, lin_kheads=4, eos=EOS), True)


MODELS = {"qwen3": _qwen3, "lfm2": _lfm2, "qwen35": _qwen35}


def _tiny(name):
    torch.manual_seed(0)
    m, spec, zc = MODELS[name]()
    m = _norms(m.float().eval(), zc)
    return m, {k: v.float().numpy().copy() for k, v in m.state_dict().items()}, spec


def _runs(spec, W, wformat="int8", n=8):
    eng = Engine(spec, W, cap=256, cfg=board_config(DRAM_BYTES=1 << 24), wformat=wformat)
    return eng, V.device_runs(eng, PROMPTS, n, spec.eos)


@pytest.fixture(scope="module")
def qwen3_int8():
    m, W, spec = _tiny("qwen3")
    eng, runs = _runs(spec, W)
    return m, W, spec, eng, runs


@pytest.mark.parametrize("name,wformat", [("qwen3", "int8"), ("qwen3", "fp4"),
                                          ("lfm2", "fp4"), ("qwen35", "fp4")])
def test_quantized_golden_follows_the_isa_simulator(name, wformat):
    """The ISA simulator's greedy run against the goldens, teacher forced: the quantized golden
    (W+A) agrees on every top-1 and is at least 5x closer in KL than fp32 (about 150x with int8
    weights here, over 1000x with fp4; a rounding difference that flips an int8 value is what
    is left, e.g. the tiny Qwen3.5's int8 run: 9x); its greedy tokens are the device's."""
    m, W, spec = _tiny(name)
    _, runs = _runs(spec, W, wformat)
    gold = V.against_golden(V.Golden(m, spec, wformat), PROMPTS, runs, ["quant", "fp32"], 8,
                            spec.eos)
    q = V.summary([g["goldens"]["quant"] for g in gold])
    f = V.summary([g["goldens"]["fp32"] for g in gold])
    assert q["top1"] == 1.0 and q["min_cos"] > 0.999
    assert q["kl_mean"] * 5 < f["kl_mean"]
    assert all(g["goldens"]["quant"]["first_diff"] is None for g in gold)
    assert V.summary([g["pairs"]["fp32"] for g in gold])["kl_mean"] > q["kl_mean"]


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
    # the logits show late, in pieces (IsaCard; whole 128-byte beats: a piece is padded to them)
    card.late = (spec.image(cfg, 256, 1, PREFILL_ROWS).io["logits"], 4 * spec.vocab // 128 * 128)

    def factory(c, imgs):
        return BoardBackend(c, imgs, transport=card, model="tiny")
    a = Namespace(wformat="int8", head_format=None, resident=False, tokens=6, cfg="board")
    runs, c = V.run_device("board", spec, None, 256, PROMPTS, a, card=(factory, cfg), W=W)
    isa, c2 = V.run_device("isa", spec, None, 256, PROMPTS, a, card=(factory, cfg), W=W)
    assert V.arch(c) == V.arch(c2)
    assert V.agreement(runs, isa)["pass"] and sum(len(r["tokens"]) for r in runs) == 12


def test_the_command_line_on_a_tiny_checkpoint(tmp_path, capsys):
    """main() end to end on a tiny Qwen3 checkpoint (and a word-level tokenizer) on disk: the
    ISA run saved without the golden, then a run against it with the golden: PASS, exit 0, the
    two runs bit for bit, --json written; a saved run of other prompts is refused."""
    import json

    from tokenizers import Tokenizer, models, pre_tokenizers
    m, _, _ = _tiny("qwen3")
    m.config.eos_token_id = 999
    m.save_pretrained(tmp_path)
    tk = Tokenizer(models.WordLevel({f"t{i}": i for i in range(1000)}, unk_token="t0"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    transformers.PreTrainedTokenizerFast(tokenizer_object=tk).save_pretrained(tmp_path)
    prompts = [" ".join(f"t{t}" for t in p) for p in PROMPTS]
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
