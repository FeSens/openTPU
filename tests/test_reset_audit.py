"""The power-up state audit (tools/reset_audit.py, docs/asic.md "Power-up state"), with yosys and
its slang front end (YOSYS, else yosys on PATH; skipped without): read as synthesis reads it, no
flip-flop that can reach control is unknown after the reset, and TMEM under random traffic from
an unknown power-up keeps its replays and grants known (and does not without its reset: the
check sees the fault the ZU7EV port's XSim run hit)."""
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import reset_audit as ra  # noqa: E402


def _slang() -> bool:
    try:
        y = ra.yosys()
    except RuntimeError:
        return False
    for pre in (["-m", "slang"], []):
        r = subprocess.run([y, *pre, "-q", "-p", "help read_slang"], capture_output=True,
                           text=True)
        if r.returncode == 0:
            return True
    return False


pytestmark = pytest.mark.skipif(not _slang(), reason="no yosys with slang (YOSYS)")


def test_no_control_state_unknown_after_reset(tmp_path):
    """Every configuration (ra.CONFIGS: the ASIC's small point, the card's one TMEM copy at the
    core clock and pumped, a copy per read port with GATHER, wmx's VPU): each flip-flop X after
    the reset is in ra.ALLOW (data, its reason given), no register a simulation-only initial sets
    is X, and no ALLOW entry is stale."""
    fails = ra.audit(list(ra.CONFIGS), 3, tmp_path)
    assert not fails, "\n".join(fails)


def test_tmem_traffic_from_unknown_power_up(tmp_path):
    t = ra.tmem_traffic(tmp_path)
    assert t["x"] == 0 and t["xd"] == 0, t
    assert t["run"] > 500 and t["lost"] >= 100 and t["frz"] >= 100, t     # replays happened
    t0 = ra.tmem_traffic(tmp_path, no_rst=True)
    assert t0["x"] > 0.9 * t0["run"], t0      # without the reset: unknown from the start, for good


@pytest.mark.parametrize("name", ["pump", "wmx"])
def test_fpga_init_define_both_ways(tmp_path, name):
    """OTPU_FPGA_INIT, the FPGA flows' define (configuration's INIT 0 is the memories' reset
    state), read both ways on the pumped TMEM (pump) and on bcu1525-x2:wmx's VPU and quantizer
    (wmx): without it (the ASIC's and the simulations' path) the slice's rst_mem is a register
    and TMEM's and the ACT RAM's state resets from it; with it rst_mem is gone and so are those
    resets, and no other unit's reset changes. The FPGA flows set it (their builds and lints); the
    ASIC flow and the simulators never do."""
    asic, ar = ra.fpga_init_resets(name, tmp_path, False)
    fpga, fr = ra.fpga_init_resets(name, tmp_path, True)
    mem = ("mem/otpu_tmem.sv", "mem/otpu_actram.sv")
    assert ar == {"rst_mem"} and not fr, (ar, fr)
    # (what stays are yosys's folds of other conditions into a synchronous reset: we_i's dw_n,
    # the write ports' wi_en, an age's own clear; none from rst_mem, which is then a constant)
    assert all(fpga.get(f, 0) < asic.get(f, 0) for f in mem), (asic, fpga)
    assert {f: n for f, n in fpga.items() if f not in mem} == \
        {f: n for f, n in asic.items() if f not in mem}
    root = Path(ra.ROOT)
    for b in ("ypcb-00338", "bcu1525"):
        assert "-DSYNTHESIS -DOTPU_FPGA_INIT" in (root / f"boards/{b}/Makefile").read_text(), b
    assert "verilog_define {SYNTHESIS OTPU_FPGA_INIT}" in \
        (root / "boards/ypcb-00338/vivado/create_project.tcl").read_text()
    bcu = root / "boards/bcu1525/vivado"
    assert "verilog_define {SYNTHESIS OTPU_FPGA_INIT}" in (bcu / "create_project_wg.tcl").read_text()
    assert (bcu / "create_project.tcl").read_text().count("set defines {SYNTHESIS OTPU_FPGA_INIT") \
        == 2                                   # udimm and rdimm
    others = [p for d in ("tools", "opentpu", "sim", "rtl/asic") for p in (root / d).rglob("*")
              if p.is_file() and p.suffix in (".py", ".tcl", ".sh", ".sv", ".v", ".mk", ".cfg")
              and p.name != "reset_audit.py"]
    assert others and not [p for p in others if "OTPU_FPGA_INIT" in p.read_text(errors="ignore")]
