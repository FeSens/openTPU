"""tools/pf500.py's ISA mode (the pf500 benchmark, docs/pf500.md): the 35B
cut's layer runs at R 2 on the S = 1 target, the set routing (each row its own k experts), the units' work
and the model row; a point that does not compile is a row with its error."""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import pf500 as P  # noqa: E402


def test_pf500_isa_s1_r2(capsys):
    rows = P.cmd_isa(argparse.Namespace(
        cmd="isa", target="bcu1525-wg:bhw16", R="2,64", kinds="dn,attn", pos=1024, path="lm",
        route="uniform", seed=1, model=P.MODEL))
    by = {(r["R"], r["kind"]): r for r in rows}
    dn, at, m = by[2, "dn"], by[2, "attn"], by[2, "model"]
    # each row its own 8 experts (16 distinct): the MoE streams 16 experts' rows
    assert dn["experts"] == at["experts"] == 16
    w = dn["phases"]["mlp"]["rd"]["weights"]
    assert w > 16 * 3 * 512 * 2048 // 2          # 16 fp4 experts (+ the shared expert)
    # the DeltaNet layer: the state a row (e6m9, read and written) and its DSTEPs on the SE
    st = dn["phases"]["deltanet"]
    assert st["rd"]["state"] == st["wr"]["state"] == 2 * 32 * 128 * 128 * 2
    assert dn["binding"] == "SE" and dn["bound"] == 2 * 32 * 128 * 128 // 8
    assert at["phases"]["attn-core"]["rd"]["kv"] > 0 and "deltanet" not in at["phases"]
    assert m["Mcyc/tok"] == round((30 * dn["bound"] + 10 * at["bound"]) / 2 / 1e6, 3)
    assert 40 < m["tok/s"] < 100
    # R 64: the layer run's rows past qwen35.RUN_ROWS (item 2): a row with its error
    assert "rows a layer run" in by[64, "dn"]["error"]
    assert (64, "model") not in by


def test_pf500_route_words_one_hot():
    tg = P.target("bcu1525-wg:bhw16")
    img = P.image(tg, P.MODEL, 1024)
    ids = P.route_ids(3, img.spec.moe.E, img.spec.moe.k)
    assert all(len(set(r)) == img.spec.moe.k for r in ids.tolist())
    words, live = P.route_words(img, P.LAYER["dn"], ids)
    assert len(words) == len(live) == 1
    x = next(v for a, v in words[0] if a == img.io["xbuf"]).view(np.float32).reshape(3, -1)
    assert np.array_equal(np.argmax(x, 1), [0, 1, 2]) and x.sum() == 3


def test_pf500_fsum_take():
    """+FSUM (wide-mxu's EXP2SUB row sums): the attention's softmax RSUMs become 2-cycle takes, the
    same instructions; only the attention core's VPU changes."""
    assert P.target("bcu1525-wg:bhw16+FSUM").cfg.FSUM and not P.target("bcu1525-wg:bhw16").cfg.FSUM
    rows = P.cmd_isa(argparse.Namespace(
        cmd="isa", target="bcu1525-wg:bhw16,bcu1525-wg:bhw16+FSUM", R="2", kinds="attn", pos=1024,
        path="lm", route="uniform", seed=1, model=P.MODEL))
    a, f = (r for r in rows if r["kind"] == "attn")
    assert a["instructions"] == f["instructions"]
    pa, pf = a["phases"], f["phases"]
    assert pf["attn-core"]["VPU"] < 0.85 * pa["attn-core"]["VPU"]
    assert all(pa[k] == pf[k] for k in pa if k != "attn-core")


def test_pf500_layout_from_config(tmp_path):
    """--model's config.json sets the layers: the 35B's cut 4 (DeltaNet 1, attention 3; 30 + 10),
    another layer_types its own cut and counts (the model row by them), and a family the runs do
    not serve (GPT-OSS, LFM2-MoE) a row with its gap and its layout, nothing compiled."""
    import json
    L = P.layout(P.MODEL)
    assert (L.family, L.cut, L.layer, L.counts, L.moe, L.gap) == \
        ("qwen35", 4, {"dn": 1, "attn": 3}, {"dn": 30, "attn": 10}, 40, None)
    assert (P.CUT, P.LAYER, P.MODEL_LAYERS) == (4, {"dn": 1, "attn": 3}, {"dn": 30, "attn": 10})
    c = json.loads((Path(P.MODEL) / "config.json").read_text())
    c["text_config"]["layer_types"] = ["full_attention", "linear_attention", "full_attention"] + \
        ["linear_attention"] * 5
    (tmp_path / "q").mkdir()
    (tmp_path / "q" / "config.json").write_text(json.dumps(c))
    L = P.layout(tmp_path / "q")
    assert (L.cut, L.layer, L.counts, L.gap) == \
        (3, {"attn": 2, "dn": 1}, {"attn": 2, "dn": 6}, None)
    tot = {u: 0 for u in P.UNITS}
    per = {"dn": {"bound": 100, "total": tot | {"MXU": 100, "dram88": 0}},
           "attn": {"bound": 300, "total": tot | {"VPU": 300, "dram88": 0}}}
    m = P.model_row(P.target("bcu1525-x2:wmx"), 2, per, argparse.Namespace(
        cmd="isa", path="em", pos=1024, route="uniform"), L.counts)
    assert m["Mcyc/tok"] == round((6 * 100 + 2 * 300) / 2 / 1e6, 3)
    fams = {"gpt_oss": ({"num_local_experts": 32,
                         "layer_types": ["sliding_attention", "full_attention"] * 12},
                        {"swa": 12, "attn": 12}, 24, "LAYER_KERNELS"),
            "lfm2_moe": ({"num_experts": 64, "num_dense_layers": 2,
                          "layer_types": (["conv"] * 2 + ["full_attention", "conv"]) * 10},
                         {"conv": 30, "attn": 10}, 38, "one slice")}
    for mt, (cfg, counts, moe, gap) in fams.items():
        d = tmp_path / mt
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"model_type": mt, **cfg}))
        rows = P.cmd_isa(argparse.Namespace(
            cmd="isa", target="bcu1525-x2:wmx", R="512", kinds=None, pos=1024, path="em",
            route="uniform", seed=1, model=str(d)))
        assert len(rows) == 1 and rows[0]["kind"] == "model", rows
        assert rows[0]["layers"] == counts and rows[0]["moe"] == moe and gap in rows[0]["error"]
        assert "not compiled" in P.table(rows)


def test_mm_cycles_pass_rule():
    """An MM in groups by otpu_mxu's pass rule: a pass (two activation rows under MXU_PUMP) takes
    max(KB, its result FIFO rows) cycles (pu_hold; unpumped 4-bit rows, lk1)."""
    from opentpu import isa as I

    def mm(n, kb, m, wf=I.W8, grp=8):
        return I.mm(0, 0, 0, n, kb, 0, n + 1, m, 0, 0, unit=True, wf=wf, grp=grp)

    # the DeltaNet core's MM (16x128 . 128x128^T, int8): 8 passes of 2 cycles a group, not KB's 1
    assert P.mm_cycles(mm(128, 1, 16), True) == 16 * 8 * 2
    # a long K: KB cycles a pass, two rows a pass (the halved M)
    assert P.mm_cycles(mm(128, 16, 16), True) == 16 * 8 * 16
    # 4-bit (two result rows a row) in groups of 16 at KB 2: 4 cycles a pass (the experts' down MMs)
    assert P.mm_cycles(mm(512, 2, 16, I.W4F, 16), True) == 32 * 8 * 4
    # an odd M: the last pass one row; M 1 one pass of one row
    assert P.mm_cycles(mm(8, 1, 3), True) == 2 + 1
    assert P.mm_cycles(mm(512, 2, 1, I.W4F, 16), True) == 32 * 2
    # unpumped: a row a pass; 4-bit at KB 1 a bubble a row
    assert P.mm_cycles(mm(128, 1, 16), False) == 16 * 16
    assert P.mm_cycles(mm(32, 1, 4, I.W4F, 16), False) == 2 * 4 * 2


def test_pipelined_expert_run_fits_bhw16_imem(monkeypatch):
    """compile_expert_run on bcu1525-wg:bhw16 returns a pipelined expert run (moe.EM_PIPE's level
    1 or 2) inside the IMEM: exact passes of 15 at level 1, 3,865 of its 4,096 instructions
    (11-04, em-bucket; the deep form at 15 does not fit TMEM). wsim at R 512: level 1 6.74 M
    cycles (at 13 too), serial (C 14) 7.90 M, the deep form at 13 8.18 M (its flag RLDs wait on
    the passes' VOPs; docs/pf500.md). A compiler change that pushes the pipelined forms
    over the IMEM or TMEM would make compile_expert_run take the serial one without a word; this
    fails instead."""
    from opentpu.compiler import CompileError
    from opentpu.llm import moe as MO
    from opentpu.llm import qwen35 as Q35
    tg = P.target("bcu1525-wg:bhw16")
    img = P.image(tg, P.MODEL, 1024, True, 512)
    li = P.LAYER["attn"]
    assert MO.EM_PIPE == 2
    progs, _ = img.compile_expert_run(li)
    C, kind = img.em_pass
    assert kind == "exact"
    n, imem = len(progs[0]), tg.cfg.IMEM_WORDS // 8
    lens = {}
    for pipe in (2, 1, 0):
        try:
            b = Q35.qwen35_expert_run.trace(img.cfg, 0, {"m": img.descriptors(0), "li": li,
                                                          "C": C, "S": 0, "exact": True,
                                                          "pipe": pipe})
            lens[pipe] = len(b.finish())
        except CompileError:
            pass
    level = next(p for p in (2, 1, 0) if lens.get(p) == n)
    print(f"bhw16 expert run: C {C}, level {level}, {n} of {imem} instructions "
          f"(margin {imem - n})")
    assert n <= imem, f"the expert run is {n} instructions, over bhw16's IMEM {imem}"
    assert level >= 1, "compile_expert_run took the serial form"


def test_dump_recs_stats_and_windows(tmp_path):
    """dump_recs: each record line carries its unit's completion counters (an MM's starve / bp /
    frz / deny) as a ninth field, the header slice 0's P / Q activity windows."""
    import json
    from types import SimpleNamespace
    from opentpu import isa as I
    from opentpu.profile import Rec
    prog = [I.Instr(I.MM, comment="mm"), I.Instr(I.VOP, comment="v")]
    recs = [Rec(0, 0, 0, I.MM, 1, 5, 6, 7, 9, 40, stats={"starve": 3, "bp": 0, "frz": 2,
                                                          "deny": 1}),
            Rec(0, 1, 1, I.VOP, 3, 6, 6, 8, -1, 12)]
    p = SimpleNamespace(cycles=40, buckets=[{"c": [32], "ms": [3]}],
                        slice_recs=lambda s: recs)
    out = tmp_path / "d.jsonl"
    P.dump_recs(str(out), p, [prog])
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert lines[0]["windows"] == {"c": [32], "ms": [3]} and lines[0]["cycles"] == 40
    assert lines[1][:8] == [0, I.MM, 1, 5, 6, 7, 9, 40] and lines[1][8]["starve"] == 3
    assert lines[2][8] == {}
    # (two slices traced: slice 1 to d-s1.jsonl, every slice's last end in slice 0's header)
    r1 = Rec(1, 0, 1, I.VOP, 3, 4, 4, 5, -1, 55)
    p2 = SimpleNamespace(cycles=40, buckets=[{"c": [32]}, {"c": [48]}], recs=recs + [r1],
                         slice_recs=lambda s: [r for r in recs + [r1] if r.slice == s])
    P.dump_recs(str(out), p2, [prog, prog])
    h0 = json.loads(out.read_text().splitlines()[0])
    s1 = [json.loads(x) for x in (tmp_path / "d-s1.jsonl").read_text().splitlines()]
    assert h0["ends"] == {"0": 40, "1": 55}
    assert s1[0]["cycles"] == 55 and s1[0]["windows"] == {"c": [48]}
    assert s1[1][:8] == [1, I.VOP, 3, 4, 4, 5, -1, 55]


def test_b2_programs_fit_imem():
    """bcu1525-x2:b2's (B2's) 35B programs inside its IMEM (the instruction cache's 16,384
    instructions, IMEM_WORDS 131072): the decode step at run-time positions 60 and 262,000, the
    generate loop, the prefill's layer runs (R 2) and expert-major runs at R 512, the DeltaNet
    layer's chunked core too (DNCHUNK 64, QSTHL + OUTER1). 11-04: the longest is the expert run,
    6,148 (wmx 5,956: VSIG and FSUM off); the chunked core 2,113 with SPLMM (2,721 without;
    wmx 3,273); a run's labels (its tiles) at their longest."""
    import board_perf as FP
    from opentpu import target as T
    cfg = T.config("bcu1525-x2:b2")
    imem, top = cfg.IMEM_WORDS // 8, {}

    def put(k, progs):
        top[k] = max(top.get(k, 0), max(len(p) for p in progs))
    for pos in ("r60", "r262000"):
        put("decode " + pos, FP.decode_case(cfg, f"{P.MODEL}:fp4:int8:{pos}")[2])
    put("generate", FP.generate_case(cfg, f"{P.MODEL}:fp4:int8:r60")[2])
    for name, path, R in (("bcu1525-x2:b2", "lm", 2), ("bcu1525-x2:b2+DNCHUNK=64", "em", 512)):
        tg = P.target(name)
        for kind in ("dn", "attn"):
            img = P.image(tg, P.MODEL, 1024, path == "em", 512)
            for lab, progs, _ in P.compile_runs(img, kind, R, 1024, path=path):
                put(f"{path} {kind} {lab}", progs)
    assert {"em dn dnc512", "em dn expert", "lm dn lm", "decode r262000"} <= set(top)
    print("b2 programs:", top)
    assert max(top.values()) <= imem, top


