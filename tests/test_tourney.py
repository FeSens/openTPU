"""Tests for the architecture tournament harness (tools/tourney): the accept rule, fitness,
parsers, sandbox, model selection, usage accounting and the report; for the fmax objective the
Vivado report parsers (fixtures: excerpts of the fp4fx120 production build's reports in
tests/data/tourney), the fmax accept rule, the build-host job counting and the local test lock.
No agents, no synthesis, no ssh."""
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import pytest
import yaml

from tools.tourney import accept as A
from tools.tourney import agents as AG
from tools.tourney import gates as G
from tools.tourney import remote as RM
from tools.tourney import report as R
from tools.tourney import synth as S

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data" / "tourney"


def m(area, fmax):
    return {"area_eq": area, "fmax": fmax}


# ------------------------------------------------------------------------------ fitness
def test_area_eq_weights():
    assert A.area_eq({"lut": 100, "lutram": 8, "ff": 20, "dsp": 2, "bram36": 1, "bram18": 1}) == \
        100 + 8 + 10 + 80 + 80 + 40
    assert A.area_eq({}) == 0


def test_est_fmax():
    assert A.est_fmax(5.0) == pytest.approx(1000 / 8.5)
    assert A.est_fmax(None) == float("inf")


def test_combine_weights_and_min_fmax():
    c = A.combine([({"lut": 10, "ff": 4, "logic_ns": 2.0, "fmax": 200}, 3),
                   ({"lut": 1, "dsp": 1, "logic_ns": 3.0, "fmax": 150}, 2)])
    assert c["lut"] == 32 and c["ff"] == 12 and c["dsp"] == 2
    assert c["fmax"] == 150 and c["logic_ns"] == 3.0


# ------------------------------------------------------------------------------ accept rule
def test_accept_area_rule():
    ok, why = A.accept(m(1000, 120), m(989, 115), 110)
    assert ok and "area" in why
    assert not A.accept(m(1000, 120), m(995, 123), 110)[0]      # <1% smaller, <3% faster


def test_accept_area_needs_target_fmax():
    ok, why = A.accept(m(1000, 80), m(900, 79), 110)       # smaller but slower, below target
    assert not ok and "< target" in why


def test_accept_pareto_below_target():
    ok, why = A.accept(m(1000, 80), m(900, 81), 110)       # smaller and not slower
    assert ok and "below target" in why
    assert A.accept(m(1000, 80), m(990, 80), 110)[0]
    assert not A.accept(m(1000, 80), m(991, 80), 110)[0]   # < 1% smaller


def test_accept_speed_rule():
    assert A.accept(m(1000, 80), m(1010, 82.4), 110)[0]        # +3%, +1% area
    assert not A.accept(m(1000, 80), m(1011, 90), 110)[0]      # area +1.1%
    assert not A.accept(m(1000, 80), m(1000, 82.3), 110)[0]    # +2.9%


def test_perf_ok():
    assert A.perf_ok(100000, 100200)
    assert not A.perf_ok(100000, 100201)
    assert A.perf_ok(None, 5) and A.perf_ok(5, None)


# ------------------------------------------------------------------------------ parsers
STAT = """
5. Printing statistics.

=== otpu_coll ===

        +----------Local Count, excluding submodules.
        |
     2731 wires
     1350 cells
        1   BUFG
        9   DSP48E1
      300   LUT2
     1025   LUT6
      578 submodules
       40   CARRY4
      250   FDRE
       12   FDSE
        2   RAM32M
        3   SRLC32E
        1   RAMB36E1

=== design hierarchy ===

        +----------Count including submodules.
        |
     1350 otpu_coll
        9   LUT6
"""


def test_parse_yosys_stat():
    s = S.parse_yosys_stat(STAT)
    assert s == {"lut": 1325, "lutram": 11, "ff": 262, "dsp": 9, "bram36": 1, "bram18": 0,
                 "carry4": 40}


def test_parse_yosys_sta():
    assert S.parse_yosys_sta("Latest arrival time in 'x.y' is 8055:\n") == 8.055
    assert S.parse_yosys_sta("nothing") is None


def test_parse_vivado_util():
    txt = ("| LUT as Logic   | 1200 |     0 | 298600 | 0.40 |\n"
           "| LUT as Memory  |   16 |     0 | 130800 | 0.01 |\n"
           "| Slice Registers |  300 |     0 | 597200 | 0.05 |\n"
           "| DSPs           |    9 |     0 |   1920 | 0.47 |\n"
           "| RAMB36/FIFO*   |    2 |     0 |    955 | 0.21 |\n")
    u = S.parse_vivado_util(txt)
    assert (u["lut"], u["lutram"], u["ff"], u["dsp"], u["bram36"], u["bram18"]) == \
        (1200, 16, 300, 9, 2, 0)


# ------------------------------------------------------------------------------ sandbox
def test_offlimits():
    allowed = ["rtl/top/otpu_coll.sv"]
    assert G.offlimits(["rtl/top/otpu_coll.sv", "HYPOTHESIS.md", "IMPLEMENTATION.md"], allowed) == []
    assert G.offlimits(["tests/test_rtl.py", "rtl/top/otpu_top.sv"], allowed) == \
        ["tests/test_rtl.py", "rtl/top/otpu_top.sv"]
    assert G.offlimits(["rtl/vpu/a.sv"], ["rtl/vpu/*.sv"]) == []


def _repo(tmp_path):
    def git(*a):
        subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)
    git("init", "-q")
    (tmp_path / "rtl").mkdir()
    (tmp_path / "rtl" / "a.sv").write_text("module a; endmodule\n")
    (tmp_path / "t.py").write_text("x = 1\n")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
        "commit", "-qm", "init")
    return tmp_path


def test_sandbox_accepts_allowed_change(tmp_path):
    wt = _repo(tmp_path)
    (wt / "rtl" / "a.sv").write_text("module a; wire w; endmodule\n")
    (wt / "HYPOTHESIS.md").write_text("# h\n")
    (wt / "build").symlink_to(tmp_path)                 # the shared-cache symlink is ignored
    assert G.sandbox(wt, ["rtl/a.sv"]) == ["rtl/a.sv"]


def test_sandbox_rejects(tmp_path):
    wt = _repo(tmp_path)
    with pytest.raises(G.GateFailure, match="no RTL change"):
        G.sandbox(wt, ["rtl/a.sv"])
    (wt / "t.py").write_text("x = 2\n")
    (wt / "rtl" / "a.sv").write_text("module a; wire w; endmodule\n")
    with pytest.raises(G.GateFailure, match="off-limits"):
        G.sandbox(wt, ["rtl/a.sv"])
    (wt / "t.py").write_text("x = 1\n")
    (wt / "new.sv").write_text("")                      # untracked files count too
    with pytest.raises(G.GateFailure, match="off-limits"):
        G.sandbox(wt, ["rtl/a.sv"])


# ------------------------------------------------------------------------------ components
def test_component_configs_are_consistent():
    comps = sorted((ROOT / "tools" / "tourney" / "components").glob("*.yaml"))
    assert len(comps) == 13
    for p in comps:
        c = yaml.safe_load(p.read_text())
        assert c["name"] == p.stem
        for f in c["allowed"]:
            assert AG.expand(ROOT, [f]) and all((ROOT / g).exists() for g in AG.expand(ROOT, [f])), f
        for t in c["tests"]["fast"] + c["tests"]["board"]:
            assert (ROOT / t.split("::")[0]).exists(), t
        for part in c["synth"]["parts"]:
            for s in part["sources"]:
                assert (ROOT / s).exists(), s
            assert set(part["sources"]) & set(c["allowed"])
        assert c["tests"]["fast"] and c["tests"]["board"]


def test_every_module_has_a_component():
    """Every RTL file of the board build is some unit component's (otpu_full aside, which may
    change them all); otpu_dram.sv is the simulation's DRAM, not in the build."""
    import fnmatch
    comps = [yaml.safe_load(p.read_text())
             for p in (ROOT / "tools" / "tourney" / "components").glob("*.yaml")]
    rtl = [str(p.relative_to(ROOT)) for p in (ROOT / "rtl").rglob("*.sv")]
    unit = [g for c in comps if c["name"] != "otpu_full" for g in c["allowed"]]
    full = next(c for c in comps if c["name"] == "otpu_full")
    for f in rtl:
        assert any(fnmatch.fnmatch(f, g) for g in full["allowed"]), f
        if f != "rtl/mem/otpu_dram.sv":
            assert any(fnmatch.fnmatch(f, g) for g in unit), f
    # the components that can change the DSTEP datapath run its RTL test
    for c in comps:
        if any(fnmatch.fnmatch(f, g) for g in c["allowed"]
               for f in ("rtl/vpu/otpu_se_tail.sv", "rtl/vpu/otpu_vtree.sv", "rtl/vpu/otpu_fp.sv")):
            assert "tests/test_vops.py::test_dstep_rtl_bit_exact" in c["tests"]["fast"], c["name"]


def test_path_summary_groups_units():
    t = ("-0.586 ns VIOLATED u_board/u_slice/u_dma/cleft_reg[28]/C -> u_board/u_mem/qb_n_reg[1][5]/D"
         " levels 20, data 8.329 ns (logic 1.493, route 6.836)\n"
         "-0.582 ns VIOLATED u_board/u_slice/u_mxu/q_h_reg/C -> u_board/u_slice/u_tmem/pw_a_reg[5]/D"
         " levels 14, data 8.000 ns (logic 1.000, route 7.000)\n"
         "-0.500 ns VIOLATED u_board/u_slice/u_mxu/q_h_reg/C -> u_board/u_slice/u_tmem/pw_b_reg[1]/D"
         " levels 12, data 8.000 ns (logic 3.000, route 5.000)\n"
         "-0.400 ns VIOLATED u_board/u_slice/u_vpu/g_se.u_tail/x_reg/C -> u_board/u_slice/cnt_reg/D"
         " levels 3, data 7.000 ns (logic 3.500, route 3.500)\n")
    assert AG.unit_of("u_board/u_slice/u_vpu/g_se.u_tail/x_reg/C") == "u_vpu/u_tail"
    assert AG.unit_of("u_board/u_slice/cnt_reg/D") == "u_slice"
    assert AG.unit_of("u_board/u_mem/qc_reg[0][19]/C") == "u_mem"
    # the LiteDRAM build: otpu_native_sys around the board, the channels' bridges, the core
    assert AG.unit_of("u_sys/u_board/u_slice/u_mxu/q_h_reg/C") == "u_mxu"
    assert AG.unit_of("u_sys/u_board/u_mem/whb_reg[0][62][1]/R") == "u_mem"
    assert AG.unit_of("u_sys/u_board/u_slice/g_x.u_dma/cleft_reg[28]/C") == "u_dma"
    assert AG.unit_of("u_sys/u_ch1/u_xr/mem_reg_0_63_0_2/RAMA/CLK") == "u_ch"
    assert AG.unit_of("u_sys/u_split/o_q_reg[3]/C") == "u_split"
    assert AG.unit_of("u_sys/cal_s1_reg[0]/C") == "u_sys"
    assert AG.unit_of("u_ld/litedramnativeportecc1_ded_errors_status_reg[16]/CE") == "u_ld"
    s = AG.path_summary(t).splitlines()
    assert s[0].startswith("- u_dma -> u_mem: 1 of the paths, worst -0.586 ns, up to 20 levels")
    assert s[1] == ("- u_mxu -> u_tmem: 2 of the paths, worst -0.582 ns, up to 14 levels, "
                    "route 75% of the data delay")
    assert "u_mem = rtl/mem/otpu_native_dram.sv" in s[-1]
    assert AG.path_summary("") == "(no paths parsed)"


def test_full_component_prompt_does_not_quote_the_rtl():
    c = yaml.safe_load((ROOT / "tools" / "tourney" / "components" / "otpu_full.yaml").read_text())
    champ = {"sha": "x", "full": {"fmax": 116.9, "period": 7.969, "wns": -0.586,
                                  "timing": "PATH-LIST", "congestion_report": "CONG"}}
    p = AG.hypothesis_prompt_fmax(c, ROOT, champ, "(none)", "(none)", AG.FMAX_CATEGORIES[0],
                                  125.49)
    assert "=== rtl/" not in p and "read the ones on the paths you target" in p
    assert "rtl/vpu/otpu_se_tail.sv" in p and "rtl/boards/ypcb-00338/otpu_board.sv" in p
    assert len(p) < 40000


# ------------------------------------------------------------------------------ agents
def test_model_selection_defaults_and_round_robin():
    for role in AG.ROLES:
        assert AG.model_for(role, 0, "claude", {}) == "claude-opus-5-5"
    env = {"MODEL_IMPL": "opus,claude-opus-4-1"}
    assert [AG.model_for("impl", k, "claude", env) for k in range(3)] == \
        ["claude-opus-5-5", "claude-opus-4-1", "claude-opus-5-5"]
    assert AG.model_for("impl", 0, "codex", {}) is None
    assert AG.model_for("impl", 1, "codex", {"MODEL_IMPL": "a,b"}) == "b"


def test_effort_selection():
    assert [AG.effort_for(r, 0, {}) for r in AG.ROLES] == ["high", "high", "low"]
    env = {"EFFORT_IMPL": "high,xhigh", "EFFORT_SCRIBE": ""}
    assert [AG.effort_for("impl", k, env) for k in range(2)] == ["high", "xhigh"]
    assert AG.effort_for("scribe", 0, env) == "low"
    with pytest.raises(ValueError):
        AG.effort_for("hyp", 0, {"EFFORT_HYP": "huge"})


def test_build_cmd_passes_model():
    c = AG.build_cmd("claude", "hi", Path("."), Path("x.last"), "claude-opus-5-5", "low")
    assert c[:3] == ["claude", "-p", "hi"]
    assert c[-4:] == ["--model", "claude-opus-5-5", "--effort", "low"]
    c = AG.build_cmd("codex", "hi", Path("."), Path("x.last"), "gpt-x", "max")
    assert c[-5:] == ["-c", "model_reasoning_effort=high", "--model", "gpt-x", "hi"]
    with pytest.raises(ValueError):
        AG.build_cmd("other", "hi", Path("."), Path("x.last"))


def test_usage_accounting():
    acc = {}
    AG.usage_of("claude", {"type": "result", "total_cost_usd": 1.25, "num_turns": 7,
                           "usage": {"input_tokens": 10, "cache_read_input_tokens": 1000,
                                     "cache_creation_input_tokens": 100, "output_tokens": 50}}, acc)
    assert acc == {"input_tokens": 1110, "cache_read_tokens": 1000, "output_tokens": 50,
                   "cost_usd": 1.25, "turns": 7}
    acc = {}
    for _ in range(2):
        AG.usage_of("codex", {"type": "turn.completed", "usage": {
            "input_tokens": 5, "cached_input_tokens": 2, "output_tokens": 3}}, acc)
    assert acc == {"input_tokens": 10, "cache_read_tokens": 4, "output_tokens": 6}


# ------------------------------------------------------------------------------ report
def test_report(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "RUNS", tmp_path)
    d = tmp_path / "otpu_x"
    d.mkdir()
    rows = [
        {"id": "r1-s0", "outcome": "accepted", "title": "t0", "reason": "area -2%",
         "metrics": {"area_eq": 980, "fmax": 120}, "models": {"hyp": "o", "impl": "opus"},
         "cost_usd": 3.0, "seconds": 600,
         "roles": {"hyp": {"seconds": 60, "usage": {}}, "impl": {"seconds": 300, "usage": {}}}},
        {"id": "r1-s1", "outcome": "broken", "title": "t1", "reason": "fast: x",
         "models": {"hyp": "o", "impl": "fable"}, "cost_usd": 1.0, "seconds": 300, "roles": {}},
    ]
    (d / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "champion.json").write_text(json.dumps({"sha": "abcdef123", "area_eq": 1000,
                                                 "fmax": 115, "backend": "yosys"}))
    s = R.model_summary(rows)
    assert s["opus"]["accepted"] == 1 and s["fable"]["broken"] == 1
    txt = R.report("otpu_x")
    assert "| opus | 1 | 1 |" in txt and "100%" in txt and "| fable | 1 | 0 |" in txt
    assert (d / "REPORT.md").exists()


def test_fp_internals_rule():
    from tools.tourney import gates as G
    assert G.fp_internal_uses("rtl/vpu/otpu_quant.sv", "m = fp_mul_s1(a, b); fadd_p1_t r;") == \
        ["fadd_p1_t", "fp_mul_s1"]
    assert G.fp_internal_uses("rtl/vpu/otpu_fp.sv", "fp_mul_s1(a, b)") == []
    assert G.fp_internal_uses("rtl/vpu/otpu_vpu.sv", "otpu_fmadd u (.a, .b); fp_gt(x, y)") == []


# ------------------------------------------------------------------------------ fmax: Vivado reports
def test_parse_vivado_util_real_report():
    u = S.parse_vivado_util((DATA / "util.rpt").read_text())
    assert (u["lut"], u["lutram"], u["ff"], u["dsp"], u["bram36"], u["bram18"]) == \
        (148048, 19518, 133410, 283, 550, 22)


def test_parse_summary_and_full():
    sm = S.parse_summary((DATA / "SUMMARY.txt").read_text())
    assert (sm["wns"], sm["whs"]) == (0.149, 0.016)
    assert sm["clocks"]["core_clk_otpu_bd_clk_wiz_0_0_1"] == (8.281, 0.149)
    logs = ("WARNING: [Synth 8-6430] The Block RAM \"x\" may get collision\n"
            "WARNING: [Synth 8-6430] The Block RAM \"y\" may get collision\n")
    f = S.parse_full((DATA / "SUMMARY.txt").read_text(), logs, (DATA / "util.rpt").read_text())
    assert f["core_clock"] == "core_clk_otpu_bd_clk_wiz_0_0_1"
    assert (f["period"], f["wns"], f["whs"]) == (8.281, 0.149, 0.016)
    assert f["fmax"] == pytest.approx(1000 / (8.281 - 0.149))
    assert f["lut"] == 148048 and f["collisions"] == 2 and not f["congested"]
    assert S.congested("WARNING: [Route 35-447] Congestion is preventing the router")
    assert S.parse_full("", "", "")["fmax"] is None


def test_parse_ooc():
    log = "junk\nOTPU_WNS -0.250\nOTPU_WHS 0.031\nOTPU_PERIOD 7.5\n"
    r = S.parse_ooc(log, (DATA / "util.rpt").read_text(), 7.5)
    assert r["wns"] == -0.25 and r["whs"] == 0.031
    assert r["fmax"] == pytest.approx(1000 / 7.75) and r["logic_ns"] == pytest.approx(7.75)
    assert S.parse_ooc("failed", "", 7.5)["fmax"] is None


def test_worst_paths():
    w = S.worst_paths((DATA / "timing_worst.rpt").read_text(), 30).splitlines()
    assert len(w) == 3
    assert w[0].split()[:3] == ["0.149", "ns", "MET"]
    assert "u_board/u_coll/s_reg[23]/C -> " in w[0] and "levels 7" in w[0]
    assert "data 7.434 ns" in w[0] and "route 6.804" in w[0]
    assert S.worst_paths((DATA / "timing_worst.rpt").read_text(), 2).count("\n") == 1


# ------------------------------------------------------------------------------ fmax: accept rule
def full(fmax=None, wns=0.1, period=7.5, whs=0.02, **kw):
    fmax = 1000 / (period - wns) if fmax is None else fmax
    return dict(fmax=fmax, wns=wns, period=period, whs=whs, collisions=0, congested=False, **kw)


def test_accept_fmax():
    old = full(wns=-0.30)                                           # 128.2 MHz at 7.5 ns
    ok, why = A.accept_fmax(old, full(wns=-0.26))                   # +0.04 ns = +0.52% fmax
    assert ok and "WNS" in why
    assert not A.accept_fmax(old, full(wns=-0.29))[0]               # +0.01 ns = +0.13%
    slow = full(wns=-3.0, period=20.0)                              # 43.5 MHz at 20 ns
    assert A.accept_fmax(slow, full(wns=-2.95, period=20.0))[0]     # +0.05 ns (+0.29%): WNS rule
    # WNS gain only counts at the same period; fmax gain always
    assert not A.accept_fmax(old, full(wns=0.0, period=7.8))[0]     # 128.2 MHz at 7.8 ns
    assert A.accept_fmax(old, full(wns=0.0, period=7.7))[0]         # 129.9 MHz
    for bad, word in ((dict(whs=-0.01), "hold"), (dict(congested=True), "congestion"),
                      (dict(collisions=1), "8-6430"), (dict(fmax=None), "no core_clk")):
        n = full(wns=0.5)
        n.update(bad)
        ok, why = A.accept_fmax(old, n)
        assert not ok and word in why


def test_ooc_promising():
    old = {"fmax": 120.0}
    assert A.ooc_promising(old, {"fmax": 120.7}, 133.33)[0]          # +0.58%
    assert not A.ooc_promising(old, {"fmax": 120.3}, 133.33)[0]
    above = {"fmax": 140.0}                                          # already above the target
    assert A.ooc_promising(above, {"fmax": 139.0}, 133.33)[0]        # -0.7%: inter-unit change
    assert not A.ooc_promising(above, {"fmax": 138.0}, 133.33)[0]    # -1.4%
    assert not A.ooc_promising(old, {"fmax": 130.0, "collisions": 1}, 133.33)[0]
    assert A.ooc_promising(None, {"fmax": None}, 133.33)[0]          # no OOC baseline
    assert not A.ooc_promising(old, {"fmax": None}, 133.33)[0]


def test_accept_fmax_area():
    u = dict(lut=190000, lutram=22800, ff=166000, dsp=353, bram36=554)
    old = full(wns=-0.431, period=7.969, **u)
    small = dict(u, lut=186000)                                   # area_eq -1.1%
    ok, why = A.accept_fmax(old, full(wns=-0.46, period=7.969, **small))
    assert ok and why.startswith("area:") and "LUT -4000" in why  # WNS -0.03: "equal"
    assert not A.accept_fmax(old, full(wns=-0.50, period=7.969, **small))[0]   # WNS -0.07
    big = dict(u, lut=196000)                                     # area_eq +1.7%
    ok, why = A.accept_fmax(old, full(wns=-0.30, period=7.969, **big))
    assert not ok and "at most" in why                            # timing win, too big
    ok, why = A.accept_fmax(old, full(wns=-0.30, period=7.969, **u))
    assert ok and why.startswith("timing:")
    # no utilization (older results): area counts as unchanged
    assert A.accept_fmax(full(wns=-0.3), full(wns=-0.2))[0]


def test_ooc_area_candidate_and_score():
    old = {"fmax": 139.1, "area_eq": 29107.0}
    ok, why = A.ooc_promising(old, {"fmax": 138.5, "area_eq": 28500.0}, 125.49)
    assert ok and "area candidate" in why                        # -2.1% area, -0.4% fmax
    assert not A.ooc_promising(old, {"fmax": 136.0, "area_eq": 28500.0}, 100.0)[0]
    assert A.ooc_score(old, {"fmax": 139.1, "area_eq": 28816.0}) == pytest.approx(0.01, abs=1e-4)
    assert A.ooc_score(None, {"fmax": 1.0}) == 0.0


def test_rank_fmax_by_score():
    old = full(wns=-0.4, period=7.969, lut=1000)
    a = {"id": "a", "full": full(wns=-0.3, period=7.969, lut=1000)}   # +1.3% fmax
    b = {"id": "b", "full": full(wns=-0.4, period=7.969, lut=900)}    # -10% area
    assert [r["id"] for r in A.rank_fmax([a, b], old)] == ["b", "a"]


def test_rank_fmax():
    a = {"id": "a", "full": dict(fmax=130.0, lut=10)}
    b = {"id": "b", "full": dict(fmax=131.0, lut=20)}
    c = {"id": "c", "full": dict(fmax=131.0, lut=5)}
    assert [r["id"] for r in A.rank_fmax([a, b, c])] == ["c", "b", "a"]


# ------------------------------------------------------------------------------ fmax: build host
def test_parse_counts_and_busy():
    c = RM.parse_counts("runviv 1\nours 1\ntotal 2\n")
    assert c == {"runviv": 1, "ours": 1, "total": 2, "docker": 0}
    busy = lambda t: RM.busy(RM.parse_counts(t), False)     # Docker host (omarchy)
    assert RM.busy(c, False) == 2           # one make bit + one tournament OOC job
    assert busy("runviv 1\nours 0\ntotal 0\n") == 1   # make bit, pre-Vivado
    assert busy("runviv 0\nours 0\ntotal 1\n") == 1   # someone's bare docker
    assert busy("runviv 2\nours 0\ntotal 2\n") == 2
    assert busy("garbage") == 0
    nat = lambda t: RM.busy(RM.parse_counts(t), True)       # native Vivado (opentpu)
    assert nat("runviv 1\nours 1\ntotal 5\n") == 2    # a make bit's runs are several processes
    assert nat("runviv 0\nours 0\ntotal 3\n") == 1    # someone's own Vivado session
    assert nat("runviv 0\nours 2\ntotal 0\n") == 2    # OOC wrappers before Vivado starts
    assert nat("garbage") == 0
    assert nat("runviv 1\nours 0\ntotal 0\ndocker 1\n") == 1   # a Docker make bit, once
    assert nat("runviv 0\nours 0\ntotal 0\ndocker 1\n") == 1   # another stream's container


def test_acquire_waits_for_room(tmp_path, monkeypatch):
    monkeypatch.setattr(RM, "START_LOCK", tmp_path / "v.lock")
    counts = iter([2, 2, 1])
    slept, logs = [], []
    with RM.acquire(poll=7, count=lambda h: next(counts), sleep=slept.append, log=logs.append,
                    max_jobs=2, hosts=["a"]) as (h, n):
        assert (h, n) == ("a", 1)
    assert slept == [7, 7] and len(logs) == 1 and "a 2/2" in logs[0]


def test_acquire_takes_hosts_in_order(tmp_path, monkeypatch):
    """The first host with room wins; a full first host sends the job to the second; an
    unreachable host counts as full."""
    monkeypatch.setattr(RM, "START_LOCK", tmp_path / "v.lock")
    load = {"a": 1, "b": 0}
    with RM.acquire(count=lambda h: load[h], max_jobs=2, hosts=["a", "b"]) as (h, n):
        assert (h, n) == ("a", 1)
    load["a"] = 2
    with RM.acquire(count=lambda h: load[h], max_jobs=2, hosts=["a", "b"]) as (h, n):
        assert (h, n) == ("b", 0)

    monkeypatch.setattr(RM, "HOST_JOBS", {"a": 1})     # a per-host cap below max_jobs
    load["a"] = 1
    with RM.acquire(count=lambda h: load[h], max_jobs=2, hosts=["a", "b"]) as (h, n):
        assert (h, n) == ("b", 0)

    def down(h):
        if h == "a":
            raise RuntimeError("ssh: connect timed out")
        return 0
    with RM.acquire(count=down, max_jobs=2, hosts=["a", "b"]) as (h, n):
        assert h == "b"


def test_acquire_hosts_file_overrides(tmp_path, monkeypatch):
    monkeypatch.setattr(RM, "START_LOCK", tmp_path / "v.lock")
    monkeypatch.setattr(RM, "HOSTS_FILE", tmp_path / "hosts")
    monkeypatch.setattr(RM, "HOSTS", ["a"])
    (tmp_path / "hosts").write_text("hosts=a,b\njobs=a=1,b=1\n")
    with RM.acquire(count=lambda h: 1 if h == "a" else 0, max_jobs=2) as (h, n):
        assert (h, n) == ("b", 0)                  # a is at its file cap of 1


def test_acquire_rereads_hosts_file_while_waiting(tmp_path, monkeypatch):
    """A wait that started with every host capped at 0 takes the raised cap (and the shorter
    host list) written while it waits."""
    monkeypatch.setattr(RM, "START_LOCK", tmp_path / "v.lock")
    monkeypatch.setattr(RM, "HOSTS_FILE", tmp_path / "hosts")
    monkeypatch.setattr(RM, "HOSTS", ["a"])
    hf = tmp_path / "hosts"
    hf.write_text("hosts=a,b\njobs=a=0,b=0\n")
    counted = []

    def count(h):
        counted.append(h)
        return 1 if h == "b" else 0

    def sleep(s):
        hf.write_text("hosts=b\njobs=a=0,b=2\n")
    with RM.acquire(poll=1, count=count, sleep=sleep, log=lambda m: None, max_jobs=2) as (h, n):
        assert (h, n) == ("b", 1)
    assert counted == ["a", "b", "b"]              # a is no longer counted after the edit


def test_acquire_serializes_starts(tmp_path, monkeypatch):
    """Two threads see room for one more job; the start lock makes the second re-count after
    the first has started (count goes 1 -> 2 once a job is started)."""
    monkeypatch.setattr(RM, "START_LOCK", tmp_path / "v.lock")
    running = [1]
    lk = threading.Lock()
    started = []

    def worker(tag):
        with RM.acquire(poll=0, count=lambda h: running[0], sleep=lambda s: time.sleep(0.01),
                        log=lambda m: None, max_jobs=2, hosts=["a"]):
            with lk:
                running[0] += 1
                started.append((tag, running[0]))
        time.sleep(0.05)
        with lk:
            running[0] -= 1

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(5)
    assert [n for _, n in started] == [2, 2]       # never 3


def test_remote_commands():
    c = RM.full_cmd("/h/otpu-build/tv-x", 133.33, "0f3a0000", is_native=False)
    assert "make bit DDR=1066" in c and "CORE_MHZ=133.33" in c and "JOBS=2" in c
    assert "VIVADO_AS_USER=1" in c and "BUILD_ID=0f3a0000" in c
    n = RM.full_cmd("/h/otpu-build/tv-x", 125.49, "0f3a0000", is_native=True)
    assert n.startswith("export PATH=$HOME/.local/bin:$PATH; ")
    assert "make bit DDR=1066" in n and "CORE_MHZ=125.49" in n and "BUILD_ID=0f3a0000" in n
    assert "VIVADO_DOCKER" not in n and "docker" not in n
    assert n.endswith(" make bit DDR=1066" + "".join(" " + x for x in RM.BUILD_ARGS))
    d = RM.docker_cmd("/h/t", "/h/t/ooc.tcl", "/h/t/v.log")
    assert "--label otpu-tourney=1" in d and "--mac-address" in d and ":ro" in d
    assert RM.vivado_cmd("/h/t", "/h/t/o/ooc.tcl", "/h/t/o/v.log", False) == d.replace(
        "/h/t/ooc.tcl", "/h/t/o/ooc.tcl").replace("/h/t/v.log", "/h/t/o/v.log")
    v = RM.vivado_cmd("/h/t", "/h/t/o/ooc.tcl", "/h/t/o/v.log", True)
    assert v == RM.NATIVE_PATH + "cd /h/t/o && bash otpu_ooc.sh"   # what NATIVE_COUNT_CMD counts
    sh = RM.native_script("/h/t/o/ooc.tcl", "/h/t/o/v.log")
    assert "exec" not in sh and "vivado -mode batch" in sh and "-source /h/t/o/ooc.tcl" in sh
    assert "[o]tpu_ooc" in RM.NATIVE_COUNT_CMD and "[r]un_vivado" in RM.NATIVE_COUNT_CMD
    sent = []
    orig = RM.ssh
    RM.ssh = lambda cmd, **k: sent.append(cmd)
    try:
        RM.start_detached("/h/t", "make bit", "full")
    finally:
        RM.ssh = orig
    # only the job is backgrounded (in braces), so the ssh returns at once
    assert sent[0].endswith("< /dev/null & }") and "{ setsid nohup bash -c " in sent[0]
    t = RM.ooc_tcl("/h/t", "otpu_vpu", ["rtl/a.sv"], {"LANES": 8}, "/h/t/o", 7.5)
    assert "-mode out_of_context" in t and "-generic LANES=8" in t
    assert "create_clock -period 7.5 -name clk [get_ports clk]" in t and "route_design" in t


def test_tree_id_includes_uncommitted(tmp_path):
    wt = _repo(tmp_path)
    t0 = RM.tree_id(wt)
    (wt / "rtl" / "a.sv").write_text("module a; wire w; endmodule\n")
    (wt / "new.sv").write_text("x\n")
    (wt / "build").symlink_to(tmp_path)
    t1 = RM.tree_id(wt)
    assert t0 != t1
    ls = subprocess.run(["git", "ls-tree", "-r", "--name-only", t1], cwd=wt, capture_output=True,
                        text=True).stdout.split()
    assert "new.sv" in ls and "build" not in ls
    assert subprocess.run(["git", "status", "--porcelain"], cwd=wt, capture_output=True,
                          text=True).stdout.count("??") == 2    # own index untouched


# ------------------------------------------------------------------------------ fmax: gates
def test_xdc_guard(tmp_path):
    assert G.xdc_violations("+set_property LOC SLICE_X0Y0 [get_cells a]\n"
                            "+create_pblock pb_vpu\n") == []
    bad = G.xdc_violations("--- a/x.xdc\n+++ b/x.xdc\n"
                           "+set_false_path -from [get_cells a]\n"
                           "+# set_multicycle_path in a comment is fine\n"
                           "-create_clock -period 8 [get_ports c]\n"
                           " set_max_delay 3 (context line)\n")
    assert len(bad) == 2 and "set_false_path" in bad[0] and "create_clock" in bad[1]
    wt = _repo(tmp_path)
    (wt / "c").mkdir()
    (wt / "c" / "t.xdc").write_text("set_multicycle_path 2 -setup -to [get_pins x]\n")
    with pytest.raises(G.GateFailure, match="timing exceptions"):
        G.sandbox(wt, ["c/*.xdc"])
    (wt / "c" / "t.xdc").write_text("create_pblock pb\n")
    assert G.sandbox(wt, ["c/*.xdc"]) == ["c/t.xdc"]
    (wt / "c" / "b.tcl").write_text("puts x\n")
    with pytest.raises(G.GateFailure, match="build scripts"):
        G.sandbox(wt, ["c/*"])


def test_no_local_vivado():
    with pytest.raises(G.GateFailure, match="does not run on this machine"):
        G.synthesize(Path("."), {"synth": {"parts": []}}, "vivado", Path("x"))


def _peak(lock, slots, n):
    inside, peak = [0], [0]
    lk = threading.Lock()

    def worker():
        with G.test_slot(lock, slots, poll=0.005):
            with lk:
                inside[0] += 1
                peak[0] = max(peak[0], inside[0])
            time.sleep(0.03)
            with lk:
                inside[0] -= 1

    ts = [threading.Thread(target=worker) for _ in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(5)
    return peak[0]


def test_test_slots_cap_concurrency(tmp_path):
    assert _peak(tmp_path / "a.lock", 1, 3) == 1
    assert _peak(tmp_path / "b.lock", 2, 5) == 2
    assert G.TEST_SLOTS == {"remote": 2, "local": 1}


def test_exec_dispatch(tmp_path, monkeypatch):
    wt = tmp_path / "fmax-otpu_vpu" / "r1-s0"
    monkeypatch.setenv("OTPU_AXI", "1")                 # never leaks into a gate
    monkeypatch.delenv("EXEC", raising=False)
    assert G.exec_mode() == "remote"
    argv, env = G.command(wt, ["python", "-m", "pytest", "-q", "t.py"], G.BOARD_ENV)
    assert argv[:4] == ["bash", "tools/omarchy_test.sh", "--exec", "env"]
    assert argv[4:8] == ["OTPU_UARCH=board", "OTPU_AXI=1", "OTPU_BOOT=1", "OTPU_NATIVE=ld"]
    assert argv[8:] == ["python", "-m", "pytest", "-q", "t.py"]
    assert env["OTPU_REMOTE_NAME"] == "tourney-fmax-otpu_vpu-r1-s0"
    assert env["OTPU_REMOTE_BUILD"] == G.REMOTE_BUILD and "OTPU_AXI" not in env
    monkeypatch.setenv("EXEC", "local")
    argv, env = G.command(wt, ["python", "-m", "pytest"], {"OTPU_BOOT": "1"})
    import sys
    assert argv == [sys.executable, "-m", "pytest"]
    assert env["PYTHONPATH"] == str(wt) and env["OTPU_BOOT"] == "1" and "OTPU_AXI" not in env
    monkeypatch.setenv("EXEC", "cloud")
    with pytest.raises(ValueError):
        G.exec_mode()


def test_remote_housekeeping(monkeypatch):
    sent = []
    hosts = []
    monkeypatch.setattr(RM, "ssh", lambda cmd, **k: (sent.append(cmd), hosts.append(k.get("host"))))
    monkeypatch.setenv("EXEC", "remote")
    G.remote_clean(Path("/x/fmax-otpu_vpu/r1-s0"))
    G.remote_prune()
    assert sent[0] == "rm -rf ~/otpu-test/tourney-fmax-otpu_vpu-r1-s0"
    assert "otpu-test/.tourney-build/verilator" in sent[1] and "-mmin +1440" in sent[1]
    assert hosts == [RM.TEST_HOST] * 2              # test trees live on omarchy, not the Vivado host
    monkeypatch.setenv("EXEC", "local")
    G.remote_clean(Path("/x/y"))
    G.remote_prune()
    assert len(sent) == 2


def test_gates_go_through_execute(tmp_path, monkeypatch):
    """lint / pytest / perf all run via execute(): remote argv, test slot held, output parsed."""
    monkeypatch.setenv("EXEC", "remote")
    monkeypatch.setattr(G, "TEST_LOCK", tmp_path / "t.lock")
    seen = []

    class R:
        def __init__(self, out, rc=0):
            self.stdout, self.stderr, self.returncode = out, "", rc

    def fake_run(argv, cwd, env, timeout):
        seen.append(argv)
        assert env["OTPU_REMOTE_SLOT_MARK"] == "1"
        if "tools/perf_qwen.py" in argv:
            return R("layers=2 x: 123456 cycles\n")
        return R("5 passed in 3s\n")

    monkeypatch.setattr(G, "run_remote", fake_run)
    wt = tmp_path / "c" / "s0"
    assert G.pytest(wt, ["tests/a.py"], G.BOARD_ENV, "board") == "5 passed in 3s"
    assert G.perf(wt) == 123456
    assert all(a[:3] == ["bash", "tools/omarchy_test.sh", "--exec"] for a in seen)
    assert "models/Qwen3-0.6B" in seen[1]
    monkeypatch.setattr(G, "run_remote", lambda *a, **k: R("FAILED x", 1))
    with pytest.raises(G.GateFailure, match="board"):
        G.pytest(wt, ["tests/a.py"], None, "board")
    # the failure ends the tail, not the shipping's stderr (tar warnings, one per file)
    noisy = R("E assert 1 == 2\nFAILED tests/a.py::t - assert\n", 1)
    noisy.stderr = "tar: Ignoring unknown extended header keyword\n" * 200
    monkeypatch.setattr(G, "run_remote", lambda *a, **k: noisy)
    with pytest.raises(G.GateFailure) as e:
        G.pytest(wt, ["tests/a.py"], None, "fast")
    assert e.value.tail.rstrip().endswith("FAILED tests/a.py::t - assert")

    def slow(*a, **k):
        raise G.subprocess.TimeoutExpired("x", 1)

    monkeypatch.setattr(G, "run_remote", slow)
    with pytest.raises(G.GateFailure, match="timeout"):
        G.pytest(wt, ["tests/a.py"], None, "fast")


def test_gate_timeout_counts_from_the_slot(tmp_path):
    """The queue for an omarchy test slot is no part of a gate's timeout: the clock starts at the
    script's slot mark; a run past it is killed."""
    mark = f"echo '{G.SLOT_MARK}1' >&2"
    r = G.run_remote(["bash", "-c", f"sleep 2; {mark}; echo 5 passed"], tmp_path, dict(os.environ),
                     1, poll=0.1)
    assert r.returncode == 0 and r.stdout == "5 passed\n" and G.SLOT_MARK in r.stderr
    t = time.time()
    with pytest.raises(subprocess.TimeoutExpired):
        G.run_remote(["bash", "-c", f"{mark}; sleep 30"], tmp_path, dict(os.environ), 1, poll=0.1)
    assert time.time() - t < 10                                    # killed, sleep and all


def test_gate_queues_again_without_a_slot(tmp_path, monkeypatch):
    """omarchy had no free test slot for 2 h (the script's exit 2): the gate queues again, and
    only fails after QUEUE_RETRIES more waits."""
    monkeypatch.setenv("EXEC", "remote")
    monkeypatch.setattr(G, "TEST_LOCK", tmp_path / "t.lock")
    full = subprocess.CompletedProcess([], 2, "", "no free test slot on omarchy after 2 h\n")
    runs = iter([full, subprocess.CompletedProcess([], 0, "3 passed in 1s\n", "")])
    monkeypatch.setattr(G, "run_remote", lambda *a, **k: next(runs))
    assert G.pytest(tmp_path / "c" / "s0", ["tests/a.py"], None, "fast") == "3 passed in 1s"
    monkeypatch.setattr(G, "run_remote", lambda *a, **k: full)
    with pytest.raises(G.GateFailure, match="no free test slot on omarchy after 3 waits"):
        G.pytest(tmp_path / "c" / "s0", ["tests/a.py"], None, "fast")


def test_opus_only():
    for bad in ("claude-sonnet-5", "claude-haiku-4-5-20251001", "fable"):
        with pytest.raises(ValueError, match="Opus"):
            AG.model_for("impl", 0, "claude", {"MODEL_IMPL": bad})
    assert AG.model_for("scribe", 0, "claude", {"MODEL_SCRIBE": "opus"}) == "claude-opus-5-5"


def test_fmax_prompt_and_xunit(tmp_path):
    c = yaml.safe_load((ROOT / "tools" / "tourney" / "components" / "otpu_xunit.yaml").read_text())
    assert c["synth"]["parts"] == [] and c["objectives"] == ["fmax"]
    assert not any("pins" in f or "vendor" in f for f in c["allowed"])
    champ = {"sha": "x", "full": {"fmax": 122.97, "period": 7.5, "wns": -0.63, "whs": 0.01,
                                  "core_clock": "core_clk", "timing": "PATH-LIST",
                                  "congestion_report": "CONG-TABLE"}}
    p = AG.hypothesis_prompt_fmax(c, ROOT, champ, "(none)", "(none)", AG.FMAX_CATEGORIES[0],
                                  133.33)
    assert "PATH-LIST" in p and "CONG-TABLE" in p and "122.97 MHz" in p
    assert "straight to the full build" in p and "=== rtl/top/otpu_slice.sv ===" in p
    assert AG.expand(ROOT, ["rtl/boards/ypcb-00338/*.sv"]) == [
        f"rtl/boards/ypcb-00338/{n}.sv" for n in (
            "otpu_afifo", "otpu_axi_split2", "otpu_board", "otpu_ctrl", "otpu_fpga_top_ld",
            "otpu_mem_ch", "otpu_native_sys", "otpu_trace")]


# ------------------------------------------------------------------------------ fmax: the round's full build
def _run(tmp_path, monkeypatch=None, slots=1):
    from tools.tourney import orchestrator as O
    r = object.__new__(O.Run)
    r.a = type("A", (), {"comp": "otpu_vpu", "target_mhz": 133.33})()
    r.fulldir = tmp_path
    if monkeypatch is not None:              # free build slots, without asking the hosts
        monkeypatch.setattr(O.G, "free_build_slots", lambda: slots)
    return O, r


def test_full_step_builds_one_candidate(tmp_path, monkeypatch):
    O, run = _run(tmp_path, monkeypatch)
    built = []

    def fake_full(wt, name, mhz, build_id):
        built.append((name, mhz, build_id))
        return full(wns=-0.20)                                     # vs -0.30: +1.3%

    monkeypatch.setattr(O.G, "full_design", fake_full)
    champ = {"full": full(wns=-0.30)}
    recs = [{"id": "s0", "outcome": "candidate", "reason": "a", "ooc_gain": 0.006, "wt": "w0"},
            {"id": "s1", "outcome": "candidate", "reason": "b", "ooc_gain": 0.02, "wt": "w1"},
            {"id": "s2", "outcome": "broken", "reason": "fast: x", "wt": "w2"}]
    run.full_step(recs, champ)
    assert built == [("otpu_vpu-s1", 133.33, O.FULL_BUILD_ID),        # the build, then its
                     ("otpu_vpu-s1-c", 133.33, O.CONFIRM_BUILD_ID)]    # confirmation
    assert [x["outcome"] for x in recs] == ["not_built", "improvement", "broken"]
    assert "confirmed: WNS -0.200 / -0.200 ns" in recs[1]["reason"]
    assert recs[1]["gain"] == pytest.approx((1000 / 7.7 - 1000 / 7.8) / (1000 / 7.8))
    assert "gate_seconds" in recs[1] and recs[1]["full"]["wns"] == -0.20


def test_full_step_broken_build(tmp_path, monkeypatch):
    O, run = _run(tmp_path, monkeypatch)

    def fail(*a):
        raise G.GateFailure("full", "no core_clk timing")

    monkeypatch.setattr(O.G, "full_design", fail)
    recs = [{"id": "s0", "outcome": "candidate", "reason": "a", "ooc_gain": 0.0, "wt": "w0"}]
    run.full_step(recs, {"full": full()})
    assert recs[0]["outcome"] == "broken" and "full:" in recs[0]["reason"]


def test_full_step_unconfirmed(tmp_path, monkeypatch):
    """A winner whose second placement does not pass the rule is not accepted."""
    O, run = _run(tmp_path, monkeypatch)
    draws = iter([full(wns=-0.20), full(wns=-0.29)])                 # +1.3%, then +0.13%
    monkeypatch.setattr(O.G, "full_design", lambda *a: next(draws))
    recs = [{"id": "s0", "outcome": "candidate", "reason": "a", "ooc_gain": 0.01, "wt": "w0",
             "perf_cycles": 1000}]
    run.full_step(recs, {"full": full(wns=-0.30), "perf_cycles": 1000})
    assert recs[0]["outcome"] == "unconfirmed" and "-0.200 / -0.290" in recs[0]["reason"]


def test_full_step_two_free_slots(tmp_path, monkeypatch):
    """Both hosts free: the two best candidates are built at once; the better one that
    confirms wins, the other one that passed is the runner-up."""
    O, run = _run(tmp_path, monkeypatch, slots=2)
    wns = {"otpu_vpu-s0": -0.25, "otpu_vpu-s1": -0.20, "otpu_vpu-s0-c": -0.25,
           "otpu_vpu-s1-c": -0.20}
    built = []

    def fake_full(wt, name, mhz, build_id):
        built.append(name)
        return full(wns=wns[name])

    monkeypatch.setattr(O.G, "full_design", fake_full)
    recs = [{"id": "s0", "outcome": "candidate", "reason": "a", "ooc_gain": 0.03, "wt": "w0"},
            {"id": "s1", "outcome": "candidate", "reason": "b", "ooc_gain": 0.02, "wt": "w1"},
            {"id": "s2", "outcome": "candidate", "reason": "c", "ooc_gain": 0.01, "wt": "w2"}]
    run.full_step(recs, {"full": full(wns=-0.30)})
    assert sorted(built[:2]) == ["otpu_vpu-s0", "otpu_vpu-s1"]    # both builds, s2 left out
    assert built[2:] == ["otpu_vpu-s1-c"]                          # s1 scores higher: confirmed
    assert [x["outcome"] for x in recs] == ["runner_up", "improvement", "not_built"]
    assert "went to s0, s1" in recs[2]["reason"] and "winner is s1" in recs[0]["reason"]


def test_full_step_confirms_the_next(tmp_path, monkeypatch):
    """The best build's confirmation fails: the next one that passed is confirmed instead."""
    O, run = _run(tmp_path, monkeypatch, slots=2)
    wns = {"otpu_vpu-s0": -0.25, "otpu_vpu-s1": -0.20, "otpu_vpu-s1-c": -0.29,
           "otpu_vpu-s0-c": -0.24}
    monkeypatch.setattr(O.G, "full_design", lambda wt, name, *a: full(wns=wns[name]))
    recs = [{"id": "s0", "outcome": "candidate", "reason": "a", "ooc_gain": 0.03, "wt": "w0"},
            {"id": "s1", "outcome": "candidate", "reason": "b", "ooc_gain": 0.02, "wt": "w1"}]
    run.full_step(recs, {"full": full(wns=-0.30)})
    assert [x["outcome"] for x in recs] == ["improvement", "unconfirmed"]
    assert "confirmed: WNS -0.250 / -0.240 ns" in recs[0]["reason"]


def test_free_slots(monkeypatch):
    monkeypatch.setattr(RM, "_override", lambda: (["a", "b", "c"], {"a": 1, "b": 1, "c": 0}))
    counts = {"a": 0, "b": 1, "c": 0}
    assert RM.free_slots(lambda h: counts[h]) == 1              # a free, b full, c capped at 0
    counts["b"] = 0
    assert RM.free_slots(lambda h: counts[h]) == 2

    def down(h):
        raise OSError("ssh")
    assert RM.free_slots(down) == 0                             # unreachable: no room


def test_accept_fmax_counts_cycles():
    old = full(wns=-0.30)
    assert A.accept_fmax(old, full(wns=-0.20), 1000, 1000)[0]        # +1.3% fmax
    assert not A.accept_fmax(old, full(wns=-0.20), 1000, 1010)[0]    # ... spent on +1% cycles
    assert A.accept_fmax(old, full(wns=-0.20), 1000, 1004)[0]        # +0.9% net


def test_saved_patch_applies(tmp_path):
    """A diff whose last hunk ends in an empty context line survives being saved (git()
    strips its output, which cut that line and made the patch corrupt)."""
    from tools.tourney import orchestrator as O
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    f = tmp_path / "a.sv"
    f.write_text("a\nb\nc\n\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "a.sv"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t",
                    "-c", "commit.gpgsign=false", "commit", "-qm", "a"], check=True)
    f.write_text("A\nb\nc\n\n")
    raw = subprocess.run(["git", "diff"], cwd=tmp_path, capture_output=True, text=True).stdout
    assert raw != O.git("diff", cwd=tmp_path) + "\n"          # the stripped form differs
    src = (ROOT / "tools" / "tourney" / "orchestrator.py").read_text()
    assert 'write_text(git("diff", cwd=wt)' not in src         # patches are saved raw


def test_full_result_cached(tmp_path, monkeypatch):
    """A full build is cached by what the board build reads: the RTL files create_project.tcl
    lists and boards/. A commit that changes neither (docs, the simulation top) reuses it; older
    keys (the commit; the rtl/ and boards/ trees) move over."""
    O, run = _run(tmp_path / "full")
    run.fulldir.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    run.repo = repo

    def commit(files: dict) -> str:
        for f, text in files.items():
            (repo / f).parent.mkdir(parents=True, exist_ok=True)
            (repo / f).write_text(text)
        O.git("add", "-A", cwd=repo)
        O.git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c", cwd=repo)
        return O.git("rev-parse", "HEAD", cwd=repo)

    O.git("init", "-q", cwd=repo)
    tcl = "set rtl [list \\\n  rtl/a.sv \\\n  rtl/b/c.sv]\nadd_files x\n"
    c1 = commit({RM.PROJECT_TCL: tcl, "rtl/a.sv": "a", "rtl/b/c.sv": "c", "rtl/top.sv": "sim",
                 "docs/x.md": "x"})
    assert RM.board_rtl(tcl) == ["rtl/a.sv", "rtl/b/c.sv"] and RM.board_rtl("") == []
    monkeypatch.setattr(RM, "BUILD_ARGS", ["FAST=1"])
    # an older result cached under the commit moves to the new key
    (run.fulldir / f"{c1[:12]}-133.33-FAST1.json").write_text(json.dumps(full(wns=0.1)))
    monkeypatch.setattr(O.G, "full_design", lambda *a: pytest.fail("rebuilt a cached commit"))
    assert run.full_result(c1, None)["wns"] == 0.1
    key = run.full_key(c1)
    assert key.name.startswith("b") and key.name.endswith("-133.33-FAST1.json") and key.exists()
    # the simulation top and the docs are not the board's: the same build
    c2 = commit({"rtl/top.sv": "sim2", "docs/x.md": "y"})
    assert run.full_key(c2) == key and run.full_result(c2, None)["wns"] == 0.1
    # a listed RTL file, a board file: another build
    assert run.full_key(commit({"rtl/b/c.sv": "c2"})) != key
    c4 = commit({"boards/ypcb-00338/constraints/x.xdc": "x"})
    assert run.full_key(c4) not in (key, run.full_key(c2))
    # a result under the old tree key moves over
    trees = "".join(O.git("rev-parse", f"{c4}:{d}", cwd=repo)[:6] for d in ("rtl", "boards"))
    (run.fulldir / f"t{trees}-133.33-FAST1.json").write_text(json.dumps(full(wns=0.2)))
    assert run.full_result(c4, None)["wns"] == 0.2


def test_directives_sandbox(tmp_path):
    ok = """# placement for timing
set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE ExtraTimingOpt [get_runs impl_1]
set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING true [get_runs synth_1]
set_property -dict {STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore} [get_runs impl_1]
"""
    assert G.directive_violations(ok) == []
    for bad in ("set_property STEPS.ROUTE_DESIGN.IS_ENABLED false [get_runs impl_1]",
                "set_property STEPS.PLACE_DESIGN.TCL.PRE /tmp/x.tcl [get_runs impl_1]",
                "set_property -dict {STEPS.ROUTE_DESIGN.TCL.POST /x.tcl} [get_runs impl_1]",
                "set_false_path -from [get_cells a]",
                "source /tmp/evil.tcl",
                "set_property STEPS.OPT_DESIGN.IS_ENABLED 0 [get_runs impl_1]",
                "set_property INCREMENTAL_CHECKPOINT /x.dcp [get_runs impl_1]"):
        assert G.directive_violations(bad), bad
    wt = _repo(tmp_path)
    d = wt / G.DIRECTIVES
    d.parent.mkdir(parents=True)
    d.write_text(ok)
    assert G.sandbox(wt, [G.DIRECTIVES]) == [G.DIRECTIVES]
    d.write_text("source /tmp/evil.tcl\n")
    with pytest.raises(G.GateFailure, match="run properties"):
        G.sandbox(wt, [G.DIRECTIVES])
    (wt / "b.tcl").write_text("x\n")
    with pytest.raises(G.GateFailure, match="off limits"):
        G.sandbox(wt, [G.DIRECTIVES, "b.tcl"])



# ------------------------------------------------------------------------------ unit objective
def ooc(area, fmax, **kw):
    return dict(dict(area_eq=area, fmax=fmax, wns=7.5 - 1000 / fmax, whs=0.03, collisions=0), **kw)


def test_accept_unit():
    old = ooc(10000, 145.0)
    ok, why = A.accept_unit(old, ooc(9900, 145.0))                   # -1%, same clock
    assert ok and why.startswith("area:")
    assert A.accept_unit(old, ooc(9900, 144.3))[0]                   # -0.5% fmax: noise
    ok, why = A.accept_unit(old, ooc(9000, 144.2))                   # -0.55%: a regression
    assert not ok and "fmax >= 144.3" in why
    assert not A.accept_unit(old, ooc(9910, 147.9))[0]               # -0.9% area, +2% fmax
    ok, why = A.accept_unit(old, ooc(10100, 149.35))                 # +3% at +1% area
    assert ok and why.startswith("speed:")
    assert not A.accept_unit(old, ooc(10110, 160.0))[0]              # +1.1% area
    # an area win walks the clock down 0.5% at most from the anchor, however many wins follow
    walked = dict(ooc(9900, 144.3), anchor_fmax=145.0)
    assert A.unit_floor(walked) == pytest.approx(145.0 * 0.995)
    assert not A.accept_unit(walked, ooc(9700, 143.9))[0]
    assert A.accept_unit(walked, ooc(9700, 144.3))[0]
    assert not A.accept_unit(old, ooc(9000, 150.0, collisions=2))[0]
    assert not A.accept_unit(old, ooc(9000, 150.0, whs=-0.01))[0]


def _unit_run(tmp_path, monkeypatch, comp="otpu_quant", **kw):
    from tools.tourney import orchestrator as O
    (tmp_path / "repo").mkdir()
    repo = _repo(tmp_path / "repo")
    monkeypatch.chdir(repo)
    args = ["--objective", "unit", "--comp", comp, "--base", "HEAD", "--no-scribe"] + [
        a for k, v in kw.items() for a in (f"--{k.replace('_', '-')}", str(v))]
    ns = []
    monkeypatch.setattr(O.Run, "main", lambda self: ns.append(self))
    O.main(args)
    return O, ns[0], repo


def test_unit_objective_names(tmp_path, monkeypatch):
    O, run, repo = _unit_run(tmp_path, monkeypatch)
    assert run.a.eval == "vivado-remote" and run.period == 7.5 and run.unit and not run.fmax
    assert run.branch == "tourney/unit/otpu_quant"
    assert run.dir == repo / "tools" / "tourney" / "runs" / "unit" / "otpu_quant"
    # the test gates' tree, the slot branch and the OOC run names differ from the fmax loop's
    assert G.remote_name(run.wtroot / "r1-s0") == "tourney-unit-otpu_quant-r1-s0"
    assert run.slot_branch("r1-s0") == "tourney-slot/unit-otpu_quant/r1-s0"
    assert run.tag == "unit-"
    with pytest.raises(SystemExit, match="no unit objective"):
        O.main(["--objective", "unit", "--comp", "otpu_xunit"])


def test_unit_prompt():
    c = yaml.safe_load((ROOT / "tools" / "tourney" / "components" / "otpu_fp.yaml").read_text())
    champ = dict(ooc(2000, 150.0), lut=1000, ff=900, timing="PATH-LIST",
                 parts={"otpu_fadd": {"lut": 400, "fmax": 150.0},
                        "otpu_fmul": {"lut": 300, "dsp": 2, "fmax": 160.0}})
    p = AG.hypothesis_prompt_unit(c, ROOT, champ, "(none)", "(none)", AG.UNIT_CATEGORIES[0],
                                  133.33, A.unit_floor(champ))
    assert "PATH-LIST" in p and "7.500 ns" in p and "fmax >= 149.2 MHz" in p
    assert "otpu_fadd x64: LUT 400" in p and "otpu_fmul x78: LUT 300, DSP 2" in p
    assert "=== rtl/vpu/otpu_fp.sv ===" in p
    assert AG.UNIT_CATEGORIES[0].startswith("area")


def test_unit_winner_becomes_champion(tmp_path, monkeypatch):
    """A unit round's winner is committed onto tourney/unit/<comp>, its OOC result becomes the
    champion's (no second OOC run of the same tree) with the anchor carried over, and it is
    logged in runs/unit/WINNERS.jsonl."""
    O, run, repo = _unit_run(tmp_path, monkeypatch)
    for k, v in (("GIT_AUTHOR_NAME", "t"), ("GIT_AUTHOR_EMAIL", "t@t"),
                 ("GIT_COMMITTER_NAME", "t"), ("GIT_COMMITTER_EMAIL", "t@t")):
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(O.G, "remote_clean", lambda wt: None)
    monkeypatch.setattr(O.G, "remote_prune", lambda: None)
    base = run.ensure_branch()
    champ = dict(ooc(1000, 145.0), sha=base, anchor_fmax=146.0, perf_cycles=100)
    monkeypatch.setattr(run, "sync_base", lambda: None)
    monkeypatch.setattr(run, "champion", lambda: champ)
    new = ooc(900, 145.5, lut=800)

    def slot(rid, k, ch):
        sid = f"{rid}-s{k}"
        wt = run.worktree(sid, run.branch)
        (wt / "rtl" / "a.sv").write_text("module a; wire x; endmodule\n")
        ok, why = A.accept_unit(ch, new)
        return {"id": sid, "slot": k, "roles": {}, "wt": str(wt), "files": ["rtl/a.sv"],
                "title": "smaller", "outcome": "improvement" if ok else "no_gain",
                "reason": why, "gain": 0.1, "_m": new, "perf_cycles": 100,
                "fast": "12 passed", "board": "4 passed"}

    monkeypatch.setattr(run, "slot", slot)
    run.round(0)
    sha = O.git("rev-parse", run.branch, cwd=repo)
    assert sha != base and O.git("log", "-1", "--format=%s", sha, cwd=repo) == \
        "tourney unit otpu_quant: smaller"
    c = json.loads((run.dir / "champion.json").read_text())
    assert c["sha"] == sha and c["area_eq"] == 900 and c["anchor_fmax"] == 146.0
    assert c["period"] == 7.5 and c["backend"] == "vivado-remote" and c["perf_cycles"] == 100
    w = json.loads((run.dir.parent / "WINNERS.jsonl").read_text())
    assert w["comp"] == "otpu_quant" and w["sha"] == sha and w["old"] == base
    assert w["ooc_old"]["fmax"] == 145.0 and w["ooc"]["lut"] == 800
    assert w["tests"] == {"fast": "12 passed", "board": "4 passed"}
    log = json.loads(run.log.read_text())
    assert log["outcome"] == "accepted" and "_m" not in log and "wt" not in log
    assert not O.git("branch", "--list", "tourney-slot/*", cwd=repo)   # slot branch dropped
    assert not run.wtroot.exists()                                     # and the slots' root


def test_local_prune(tmp_path):
    v = tmp_path / "build" / "verilator"
    for n in ("tb_top_old", "tb_top_new", "tb_fp_old"):
        (v / n).mkdir(parents=True)
        (v / n / "Vtb").write_text("x")
    now = time.time()
    for n in ("tb_top_old", "tb_fp_old"):
        os.utime(v / n, (now - 7 * 3600, now - 7 * 3600))
    assert G.local_prune(tmp_path / "build", 360, now) == ["tb_fp_old", "tb_top_old"]
    assert [p.name for p in v.iterdir()] == ["tb_top_new"]
    assert G.local_prune(tmp_path / "nothing", 360) == []


def test_leftover_worktrees(tmp_path, monkeypatch):
    """A run removes the worktrees (and slot branches) an earlier run of the same component and
    objective left behind, unless that run is still live (it holds the lock)."""
    O, run, repo = _unit_run(tmp_path, monkeypatch)
    monkeypatch.setattr(O.G, "remote_clean", lambda wt: None)
    run.ensure_branch()
    assert not run.dedicated()                               # the main checkout: not pruned
    wt = run.worktree("r1-s0", run.branch)
    assert run.dedicated.__func__(type("R", (), {"repo": wt})())    # a linked worktree is
    live = (run.wtroot.parent / f"{run.wtroot.name}.lock").open("w")
    O.fcntl.flock(live, O.fcntl.LOCK_SH)                     # another run of it, still live
    run.clean_leftovers()
    assert wt.exists()
    run._live.close()
    live.close()
    run.clean_leftovers()                                    # that run is gone: leftovers
    assert not wt.exists()
    assert not O.git("branch", "--list", "tourney-slot/*", cwd=repo)


def test_forever_objective_and_control_files(tmp_path):
    """forever.sh with another objective and its own control files (a second loop beside the
    fmax one); WHOLE_EVERY=0 runs no whole-design rounds."""
    (tmp_path / "repo").mkdir()
    repo = _repo(tmp_path / "repo")
    comps = repo / "tools" / "tourney" / "components"
    comps.mkdir(parents=True)
    for c in ("otpu_quant", "otpu_coll", "otpu_full"):
        (comps / f"{c}.yaml").write_text("name: x\n")
    (repo / "tools" / "tourney" / "forever.sh").write_text(
        (ROOT / "tools" / "tourney" / "forever.sh").read_text())
    stub = tmp_path / "bin"
    stub.mkdir()
    calls, stop = tmp_path / "calls", tmp_path / "units-stop"
    # the orchestrator's stand-in: records its arguments and control files, stops after 3 rounds
    (stub / "python3").write_text(
        f'#!/bin/bash\necho "$* STOP=$OTPU_TOURNEY_STOP PAUSE=$OTPU_TOURNEY_PAUSE" >> {calls}\n'
        f'[ $(wc -l < {calls}) -ge 3 ] && touch {stop}\nexit 0\n')
    (stub / "python3").chmod(0o755)
    env = dict(os.environ, PATH=f"{stub}:{os.environ['PATH']}", OBJECTIVE="unit",
               WHOLE_EVERY="0", FOREVER_COMPS="otpu_quant otpu_coll", K="1",
               FOREVER_COMPS_FILE=str(tmp_path / "none"), FOREVER_STOP=str(stop),
               FOREVER_PAUSE=str(tmp_path / "units-pause"))
    r = subprocess.run(["bash", "tools/tourney/forever.sh"], cwd=repo, env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    rows = calls.read_text().splitlines()
    assert [x.split("--comp ")[1].split()[0] for x in rows] == ["otpu_quant", "otpu_coll",
                                                                "otpu_quant"]
    assert all("--objective unit" in x and f"STOP={stop}" in x and
               f"PAUSE={tmp_path / 'units-pause'}" in x for x in rows)
    assert f"{stop}: stopping" in r.stdout
    assert "division" not in r.stderr, r.stderr                     # WHOLE_EVERY=0 is no error


def test_replay_applies_a_logged_patch(tmp_path, monkeypatch):
    """--replay: a logged slot's saved patch in place of the agents, its hypothesis carried
    over (a slot whose gate failed for reasons not its own)."""
    O, run, repo = _unit_run(tmp_path, monkeypatch, replay="r1-s0")
    assert run.a.replay == "r1-s0" and run.a.slots == 1
    run.ensure_branch()
    (run.dir / "patches").mkdir(parents=True)
    (repo / "rtl" / "a.sv").write_text("module a; wire y; endmodule\n")
    (run.dir / "patches" / "r1-s0.patch").write_text(subprocess.run(
        ["git", "diff"], cwd=repo, capture_output=True, text=True).stdout)
    subprocess.run(["git", "checkout", "-q", "--", "rtl/a.sv"], cwd=repo, check=True)
    run.log.write_text(json.dumps({"id": "r1-s0", "title": "narrower", "hypothesis": "# narrower",
                                   "implementation": "done", "category": "area"}) + "\n")
    wt = run.worktree("r2-s0", run.branch)
    rec = {}
    run.replay_into(wt, rec)
    assert (wt / "rtl" / "a.sv").read_text() == "module a; wire y; endmodule\n"
    assert rec["replay_of"] == "r1-s0" and rec["title"] == "narrower" and rec["category"] == "area"
    run.a.replay = "r9-s0"
    with pytest.raises(G.GateFailure, match="no logged slot r9-s0"):
        run.replay_into(wt, {})
