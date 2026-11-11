"""The workload scoreboard (tools/scoreboard.py): the model reproduces the card's and the co-sim's
points it is calibrated on, moves the right way under every what-if knob, reads model shapes
and weight formats as the image builder does, and its CLI runs. Pure Python: no RTL, no
checkpoint, no card."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import scoreboard as SB  # noqa: E402

YPCB = SB.board("ypcb")


@pytest.fixture(scope="module")
def rows():
    return SB.validate(loo=False)


@pytest.mark.parametrize("group,mean,worst", [
    ("decode", 1.0, 3.5), ("prefill", 1.5, 5.0), ("batched", 1.0, 2.0), ("co-sim", 2.0, 5.0),
    ("co-sim layer", 2.0, 3.5), ("co-sim tile kernel", 3.5, 13.0),
    ("MoE layer run ms", 4.0, 6.0), ("MoE", 4.0, 6.0), ("co-sim campaign", 6.0, 11.5)])
def test_reproduces_card_and_cosim(rows, group, mean, worst):
    errs = [abs(e) for g, _, _, _, e, _ in rows if g == group]
    assert errs, group
    assert sum(errs) / len(errs) <= mean and max(errs) <= worst, (group, errs)


def test_decode_bytes_per_token_match_the_card():
    """The decode bytes a token, from the shapes alone (Gemma 4 left out: its card counters
    read ~100 MB a token more than its weights and KV, unexplained)."""
    for name, fmt, _, mb, _ in SB.CARD_DECODE:
        if name == "gemma4":
            continue
        got = SB.bytes_per_token(SB.model(name), fmt, 544) / SB.MB
        assert abs(got / mb - 1) < 0.03, (name, fmt, got, mb)


def test_leave_one_out_prefill():
    loo = [abs(el) for g, _, _, _, _, el in SB.validate(loo=True)
           if g == "prefill" and el is not None]
    assert loo and max(loo) < 8.0


def test_fit_reproduces_kappa():
    k = SB.fit()
    for name, (a, b) in SB.KAPPA.items():
        assert k[name] == pytest.approx((a, b), abs=2e-3), name


def _summary(hw, **kw):
    return SB.summary(SB.scoreboard(hw, names=["qwen3", "phi4-mini", "qwen35"], **kw))


def test_more_bandwidth_decodes_faster():
    lo = SB.decode("phi4-mini", YPCB)["tok_s"]
    wide = YPCB.replace(port_bytes=256, mts=2133.33)
    # the MXU takes D weights a cycle (2 D 4-bit, PAIR), the port's rate: a wider port alone
    # buys only the 4-bit stream's last ~10%
    assert 1.0 < SB.decode("phi4-mini", wide)["tok_s"] / lo < 1.2
    assert SB.decode("phi4-mini", wide.replace(mxu_pump=2.0))["tok_s"] > 1.7 * lo
    # the YPCB's DDR3 sits 3% under its port: a faster DRAM alone buys those 3%
    assert SB.decode("phi4-mini", YPCB.replace(mts=1600))["tok_s"] == pytest.approx(lo * 1.03,
                                                                                   rel=0.01)
    assert YPCB.bound() == "memory" and YPCB.replace(mts=1600).bound() == "port"


def test_the_clock_scales_a_port_bound_board():
    a = SB.decode("qwen3", YPCB)["tok_s"]
    b = SB.decode("qwen3", YPCB.replace(mhz=200.0, mts=1600))["tok_s"]
    assert b == pytest.approx(a * 200 / 133.33, rel=0.02)


@pytest.mark.parametrize("kw", [dict(mcols=8), dict(mxu_pump=2.0), dict(vpu=2.0),
                                dict(tmem_words=262144), dict(S=2)])
def test_a_bigger_unit_is_never_slower(kw):
    hw = YPCB.replace(**kw)
    for name in ("qwen3", "qwen35", "phi4-mini"):
        assert SB.decode(name, hw)["tok_s"] >= SB.decode(name, YPCB)["tok_s"] * 0.999, name
        assert SB.prefill(name, hw, P=256)["tok_s"] >= \
            SB.prefill(name, YPCB, P=256)["tok_s"] * 0.995, name
        assert SB.batch(name, hw, 4)["total_tok_s"] >= \
            SB.batch(name, YPCB, 4)["total_tok_s"] * 0.995, name


def test_mcols_8_speeds_prefill_and_batches_but_not_decode():
    r = SB.compare(YPCB, YPCB.replace(mcols=8))
    assert r["decode"] == pytest.approx(1.0, abs=0.02)
    assert r["prefill"] > 1.2 and r["batch"] > 1.4


def test_deltanet_layers_at_mcols_4_and_8():
    """A DeltaNet layer's further rows expose part of their VOPs with their DSTEPs, less what
    the gap between the stream and the MXU hides: the campaign's Qwen3.5-2B layer at R 4 takes
    547.2 K cycles at MCOLS 4 and 401.4 K at MCOLS 8 (0.734x; without the VOP term the model
    gave 0.665x). Decode (one row) is unchanged."""
    m = SB.model("qwen35-2b")
    fmt = SB.formats_of("qwen35-2b", "fp4")
    i, lay = next((i, lay) for i, lay in SB.layers(m) if lay["kind"] == "delta" and i == 1)
    ctxs = list(range(512, 516))
    c4, c8 = (SB.layer_cycles(m, i, lay, SB.board("ypcb", mcols=mc), fmt, ctxs, prompt=True)
              for mc in (4, 8))
    assert c4 == pytest.approx(547150, rel=0.03) and c8 == pytest.approx(401395, rel=0.03)
    assert c8 / c4 == pytest.approx(401395 / 547150, rel=0.03)
    one = SB.layer_cycles(m, i, lay, YPCB, fmt, [544])
    old = dict(SB.CAL)
    try:
        SB.CAL["dn_vop"] = SB.CAL["dn_slack"] = 0.0
        assert SB.layer_cycles(m, i, lay, YPCB, fmt, [544]) == one
    finally:
        SB.CAL.update(old)


def test_batched_total_grows_and_each_user_slows():
    one, four = SB.batch("qwen3", YPCB, 1), SB.batch("qwen3", YPCB, 4)
    assert four["total_tok_s"] > 2 * one["total_tok_s"]
    assert four["user_tok_s"] < one["user_tok_s"]
    assert SB.batch("qwen3", YPCB, 8)["runs"] == 2
    assert SB.batch("qwen3", YPCB.replace(batch_rows=8, act_rows=8), 8)["runs"] == 1


def test_prefill_rows():
    assert SB.prefill_rows("qwen3", YPCB) == 8                 # two passes of 4
    assert SB.prefill_rows("qwen35-4b", YPCB, fmt="fp4") == 2   # PAIR: odd R's R - 1
    assert SB.prefill_rows("qwen35-4b", YPCB, fmt="int8") == 3
    assert SB.prefill_rows("qwen3", YPCB.replace(rows=64)) == 64
    big = YPCB.replace(tmem_words=262144, act_rows=16)
    assert SB.prefill_rows("qwen3", big) > SB.prefill_rows("qwen3", YPCB)
    assert SB.prefill_rows("qwen3", big.replace(imem=0)) >= SB.prefill_rows("qwen3", big)


def test_longer_prompts_take_longer():
    a, b = SB.prefill("qwen3", YPCB, P=256), SB.prefill("qwen3", YPCB, P=1024)
    assert b["ttft_s"] > 3.5 * a["ttft_s"]


def test_moe_offload_and_residency():
    for name in SB.MOE:
        f = SB.moe_fit(name, YPCB)
        assert not f["resident"] and 0.05 < f["frac"] < 0.3, f
        d = SB.moe_decode(name, YPCB)
        assert d["misses"] > 50 and d["tok_s"] < d["card_tok_s"]
        # a faster host link exposes less
        assert SB.moe_decode(name, YPCB.replace(pcie_gbps=8.0))["tok_s"] > d["tok_s"]
    big = SB.board("bcu1525", S=4, fast_gb=8.0)         # (slices that hold the whole pool)
    assert SB.moe_fit("gemma4-26b", big)["resident"]
    assert SB.moe_decode("gemma4-26b", big)["misses"] == 0
    assert SB.moe_prefill("qwen35-35b", YPCB, P=256)["rows"] == 2      # R k <= 16
    assert SB.moe_prefill("qwen35-35b", YPCB.replace(rows=64), P=256)["rows"] == 2


def test_compare_against_itself_is_one():
    r = SB.compare(YPCB, YPCB)
    assert all(v == pytest.approx(1.0) for v in r.values())


def test_union_of_experts():
    assert SB.union(256, 8, 1) == 8
    assert 8 < SB.union(256, 8, 2) < 16
    assert SB.union(256, 8, 4096, 145.0) == pytest.approx(145.0)
    assert SB.union(64, 8, 4096) <= 64


def test_group_format_rules():
    f = "attn@15-34=fp4,mlp@15-34=fp4"
    assert SB.group_format("attn", 3, "int8", f) == "int8"
    assert SB.group_format("attn", 15, "int8", f) == "fp4"
    assert SB.group_format("gateup", 20, "int8", f) == "fp4"
    assert SB.group_format("down", 34, "int8", f) == "fp4"
    assert SB.group_format("down", 35, "int8", f) == "int8"
    # a ranged rule beats an unranged one, whatever their order
    assert SB.group_format("gateup", 5, "int8", "mlp=fp4,gateup@4-6=int8") == "int8"
    assert SB.group_format("gateup", 9, "int8", "mlp=fp4,gateup@4-6=int8") == "fp4"
    assert SB.group_format("experts", 0, "int8", "experts=fp4") == "fp4"
    assert SB.formats_of("phi4-mini", "auto") == ("int8", "int8", SB.MIXES["phi4-mini"])
    assert SB.formats_of("qwen3", "fp4") == ("fp4", "int8", "")


def test_shapes_of_a_tiny_checkpoint():
    cfg = {"model_type": "qwen3", "hidden_size": 256, "num_hidden_layers": 2,
           "vocab_size": 1000, "num_attention_heads": 4, "num_key_value_heads": 2,
           "head_dim": 64, "intermediate_size": 600, "tie_word_embeddings": True}
    t = {"model.embed_tokens.weight": ["BF16", [1000, 256]]}
    for i in range(2):
        p = f"model.layers.{i}."
        t.update({p + "self_attn.q_proj.weight": ["BF16", [256, 256]],
                  p + "self_attn.k_proj.weight": ["BF16", [128, 256]],
                  p + "self_attn.v_proj.weight": ["BF16", [128, 256]],
                  p + "self_attn.o_proj.weight": ["BF16", [256, 256]],
                  p + "mlp.gate_proj.weight": ["BF16", [600, 256]],
                  p + "mlp.up_proj.weight": ["BF16", [600, 256]],
                  p + "mlp.down_proj.weight": ["BF16", [256, 600]],
                  p + "input_layernorm.weight": ["BF16", [256]]})
    m = SB.shapes("tiny", cfg, t)
    assert m["family"] == "qwen3" and m["n_layers"] == 2 and not m["untied"]
    assert len(m["runs"]) == 1 and m["runs"][0][0] == 2
    lay = m["runs"][0][1]
    assert lay["kind"] == "attn" and lay["kv_heads"] == 2 and lay["head_dim"] == 64
    assert lay["groups"]["attn"] == 2 * 256 * 256 + 2 * 128 * 256
    assert lay["groups"]["gateup"] == 2 * 640 * 256              # padded to D = 128
    assert lay["groups"]["down"] == 256 * 640
    assert m["head"] == 1024 * 256                                # the tied embedding


def test_every_model_has_shapes():
    for name in SB.DENSE + SB.MOE:
        m = SB.model(name)
        assert sum(n for n, _ in m["runs"]) == m["n_layers"], name
        assert SB.decode(name, YPCB)["tok_s"] > 0 if name in SB.DENSE else \
            SB.moe_decode(name, YPCB)["tok_s"] > 0


def test_cli(tmp_path):
    out = tmp_path / "sc.json"
    r = subprocess.run([sys.executable, str(ROOT / "tools/scoreboard.py"), "report",
                        "--board", "bcu1525", "--mts", "2400", "--mcols", "8",
                        "--models", "qwen3,phi4-mini", "--json", str(out), "--markdown"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "| qwen3 |" in r.stdout and "MCOLS 8" in r.stdout
    sc = json.loads(out.read_text())
    assert set(sc["decode"]) == {"qwen3", "phi4-mini"}
    r = subprocess.run([sys.executable, str(ROOT / "tools/scoreboard.py"), "compare",
                        "--vpu", "2", "--models", "qwen3"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "score" in r.stdout, r.stderr


def test_cosim_layers_pair_the_prefixes(tmp_path):
    base = dict(model="Qwen3-0.6B", rows=4, pos=544, batch=False, wformat="fp4",
                head_format="int8", formats="", mcols=4, act_rows=0, lanes=8, tmem=65536,
                mem="ldc", S=1, mhz=133.33, bw=100, lat=40, mts=1066)
    hw = SB.board("ypcb")
    m = SB.model("qwen3")
    lay = list(SB.layers(m))[1][1]
    want = SB.layer_cycles(m, 1, lay, hw, ("fp4", "int8", ""), [544, 545, 546, 547])
    recs = [dict(base, layers="1", cycles=1_000_000),
            dict(base, layers="2", cycles=1_000_000 + round(want)),
            dict(base, layers="2", lanes=16, error="CompileError: x")]
    f = tmp_path / "c.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    rows = SB.cosim_layers(f)
    assert len(rows) == 1 and rows[0]["config"] == "ypcb"
    assert abs(rows[0]["error"]) < 0.01


def test_the_tile_kernel_is_taken_only_where_faster():
    tiled = YPCB.replace(kernel="tiled", rows=16)
    for name in ("qwen3", "smollm3", "qwen35"):
        a, b = SB.prefill(name, YPCB, P=256), SB.prefill(name, tiled, P=256)
        assert b["ttft_s"] <= a["ttft_s"] * 1.0001, name
    # batched users and MoE layer runs keep their own kernels
    assert SB.batch("qwen3", tiled, 4)["total_tok_s"] == \
        pytest.approx(SB.batch("qwen3", YPCB, 4)["total_tok_s"])
    assert SB.moe_prefill("gemma4-26b", tiled, 256)["ttft_s"] == \
        pytest.approx(SB.moe_prefill("gemma4-26b", YPCB, 256)["ttft_s"])


def test_what_binds():
    d = SB.decode("phi4-mini", YPCB)
    assert d["bound"].startswith("memory") and abs(sum(d["shares"].values()) - 1) < 1e-9
    assert SB.decode("phi4-mini", YPCB.replace(mts=1600))["bound"].startswith("port")
    assert SB.prefill("qwen3", YPCB, P=256)["bound"].startswith("MXU")
    assert SB.decode("phi4-mini", YPCB.replace(port_bytes=256))["bound"].startswith("memory")
    assert SB.moe_decode("qwen35-35b", YPCB)["shares"]["host link"] > 0.1
    b = SB.bounds(SB.scoreboard(YPCB, names=["qwen3"]))
    assert set(b) == {"decode", "prefill", "batch", "moe"}
