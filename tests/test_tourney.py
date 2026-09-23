"""Tests for the architecture tournament harness (tools/tourney): the accept rule, fitness,
parsers, sandbox, model selection, usage accounting and the report. No agents, no synthesis."""
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from tools.tourney import accept as A
from tools.tourney import agents as AG
from tools.tourney import gates as G
from tools.tourney import report as R
from tools.tourney import synth as S

ROOT = Path(__file__).resolve().parents[1]


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
    ok, why = A.accept(m(1000, 80), m(900, 81), 110)
    assert not ok and "< target" in why


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
    assert len(comps) == 10
    for p in comps:
        c = yaml.safe_load(p.read_text())
        assert c["name"] == p.stem
        for f in c["allowed"]:
            assert (ROOT / f).exists(), f
        for part in c["synth"]["parts"]:
            for s in part["sources"]:
                assert (ROOT / s).exists(), s
            assert set(part["sources"]) & set(c["allowed"])
        assert c["tests"]["fast"] and c["tests"]["board"]


# ------------------------------------------------------------------------------ agents
def test_model_selection_defaults_and_round_robin():
    for role in AG.ROLES:
        assert AG.model_for(role, 0, "claude", {}) == "claude-opus-5-5"
    env = {"MODEL_IMPL": "opus,model-b"}
    assert [AG.model_for("impl", k, "claude", env) for k in range(3)] == \
        ["claude-opus-5-5", "model-b", "claude-opus-5-5"]
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
