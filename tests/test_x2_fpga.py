"""P3's per-card build contract and offline Tcl/netlist checks. The full-top lint runs
with the real target parameters and explicit IP/primitive stubs; it is elaboration
evidence only. Nothing here invokes Vivado, programs a card or touches another worktree."""
from dataclasses import replace
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

from opentpu import target as T

ROOT = Path(__file__).resolve().parents[1]
BCU = ROOT / "boards/bcu1525"
X2 = BCU / "x2"


def _params(out, *args):
    return subprocess.run([sys.executable, str(BCU / "scripts/make_wg.py"), str(out),
                           "--target", "bcu1525-x2:st16", *args], capture_output=True, text=True)


@pytest.mark.parametrize("mhz", [100, 150])
def test_x2_card_hash_and_topology(tmp_path, mhz):
    r = _params(tmp_path, "--core-mhz", str(mhz))
    assert r.returncode == 0, r.stderr
    p = (tmp_path / "wg_params.tcl").read_text()
    vh = (tmp_path / "otpu_bcu_wg_defines.vh").read_text()
    for line in ("set WG_S 2", "set WG_NCARDS 2", "set ARCH_HASH e331f79c",
                 "set WG_SLRS [list 0 2]", "set RLY 2222555555552222", "set WG_CAW 24",
                 f"set CORE_KHZ {mhz * 1000}"):
        assert line in p
    assert ".PLATFORM(7)" in vh and "`define OTPU_WG_S 2" in vh
    assert "`define OTPU_WG_BHW 30" in vh and "`define OTPU_WG_X2\n" in vh
    assert "`define OTPU_WG_HASH 32'hE331F79C" in vh
    # (the governor's identity words: the card's platform and the machine's hash, in SAFE too)
    assert "`define OTPU_WG_PLATFORM 7\n" in vh and "`define OTPU_WG_CFG_HASH 32'hE331F79C\n" in vh
    assert ".ARCH_HASH(32'he331f79c)" in vh
    machine = T.load("bcu1525-x2:st16")
    card = replace(machine, arch=dict(machine.arch, S=2))
    assert machine.arch["S"] == 4
    assert T.arch_hash(T.build_config(machine)) == 0xD5239726
    assert T.arch_hash(T.build_config(card)) == 0xE331F79C


@pytest.mark.parametrize("args", [("--set", "S=2"), ("--set", "PLATFORM=6"),
                                  ("--plan", "a"), ("--plan", "b")])
def test_x2_rejects_topology_overrides(tmp_path, args):
    r = _params(tmp_path, *args)
    assert r.returncode != 0 and "make_wg.py: x2" in r.stderr
    assert not (tmp_path / "wg_params.tcl").exists()


def test_x2_prepare_launcher_has_no_fpga_tool_dependency(tmp_path):
    env = dict(os.environ, VIVADO="/no/vivado/here")
    r = subprocess.run(["bash", str(X2 / "build_x2.sh"), "--prepare", str(tmp_path)],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "prepared only: no Vivado invoked" in r.stdout
    assert "FAST=1 CORE_MHZ=100" in r.stdout and "WHS >= 0" in r.stdout
    assert "set ARCH_HASH e331f79c" in (tmp_path / "wg_params.tcl").read_text()


def test_x2_resume_does_not_relabel_another_synthesis(tmp_path):
    assert _params(tmp_path).returncode == 0
    params = tmp_path / "wg_params.tcl"
    before = params.read_text().replace("set ARCH_HASH e331f79c", "set ARCH_HASH deadbeef")
    params.write_text(before)
    (tmp_path / "otpu.xpr").write_text("mock project")
    r = subprocess.run(["bash", str(X2 / "build_x2.sh"), "--prepare", str(tmp_path)],
                       env=dict(os.environ, RESUME="1"), capture_output=True, text=True)
    assert r.returncode == 2 and "resume mismatch" in r.stderr
    assert params.read_text() == before


@pytest.mark.parametrize("key,value", [("CORE_MHZ", "150"), ("FAST", "0"), ("WG_SET", "ST16=0")])
def test_x2_dev_launcher_rejects_changed_contract(tmp_path, key, value):
    r = subprocess.run(["bash", str(X2 / "build_x2.sh"), "--prepare", str(tmp_path)],
                       env=dict(os.environ, **{key: value}), capture_output=True, text=True)
    assert r.returncode == 2 and "x2 dev requires" in r.stderr


def test_x2_qsfp_pin_coverage_and_exclusions():
    # The production DDR/PCIe pads plus all eight QSFP lanes, one reference, six cage outputs:
    # no duplicate pad and no SI5335 pin (s5's clock deadlock). Keep both REFCLK0 pairs absent.
    base = (BCU / "constraints/bcu1525_wg_pins.xdc").read_text()
    x2 = (X2 / "pins_x2.xdc").read_text()
    pins = re.findall(r"PACKAGE_PIN (\w+)", base + x2)
    assert len(pins) == len(set(pins))
    assert not set(pins) & {"AT22", "AT20", "AU22", "AR21", "AR22", "AU20"}
    gt = re.findall(r"get_ports \{(qsfp[01]_[rt]x_[pn]\[\d\])\}", x2)
    want = {f"qsfp{p}_{d}x_{pn}[{i}]" for p in range(2) for d in "rt"
            for pn in "pn" for i in range(4)}
    assert len(gt) == 32 and set(gt) == want
    assert "qsfp0_mgt_refclk_1_p" in x2 and "qsfp0_mgt_refclk_1_n" in x2
    assert "qsfp1_mgt_refclk" not in x2 and "mgt_refclk_0" not in x2


def _tcl(tmp_path, body):
    script = tmp_path / "check.tcl"
    script.write_text(body)
    return subprocess.run(["tclsh", str(script)], cwd=ROOT, capture_output=True, text=True)


def test_x2_tcl_complete_and_shell_syntax(tmp_path):
    for f in [*X2.glob("*.tcl"), *X2.glob("*.xdc"), BCU / "vivado/bd_wg.tcl",
              BCU / "vivado/create_project_wg.tcl", BCU / "vivado/post_impl.tcl"]:
        r = _tcl(tmp_path, f'set f [open {{{f}}}]; set s [read $f]; close $f\n'
                 'if {![info complete $s]} { error "incomplete Tcl" }\n')
        assert r.returncode == 0, (f, r.stderr)
    r = subprocess.run(["bash", "-n", str(X2 / "build_x2.sh"), str(BCU / "run_vivado.sh")])
    assert r.returncode == 0


def test_x2_late_constraints_find_their_files_without_info_script(tmp_path):
    # An implementation run reads the late Tcl with read_xdc -unmanaged: no [info script], its
    # directory the run's. The x2 files come from OTPU_WG_ROOT (run_vivado.sh) and both finish.
    x2 = tmp_path / "tree/boards/bcu1525/x2"
    x2.mkdir(parents=True)
    (x2 / "floorplan_x2.tcl").write_text("incr ::otpu::ok(x2_floorplan)\n")
    (x2 / "cdc_x2.tcl").write_text("incr ::otpu::ok(x2_cdc)\n")
    run = tmp_path / "impl_1"
    run.mkdir()
    body = f'''namespace eval ::otpu {{}}
set ::otpu::fail {{}}
set fh [open {{{X2 / 'constraints_x2.tcl'}}}]; set s [read $fh]; close $fh
eval $s
puts "fail=$::otpu::fail fp=$::otpu::ok(x2_floorplan) cdc=$::otpu::ok(x2_cdc)"
'''
    (tmp_path / "check.tcl").write_text(body)
    env = {**os.environ, "OTPU_WG_ROOT": str(tmp_path / "tree")}
    r = subprocess.run(["tclsh", str(tmp_path / "check.tcl")], cwd=run, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "fail= fp=1 cdc=1" in r.stdout
    assert 'export OTPU_WG_ROOT="$root"' in (BCU / "run_vivado.sh").read_text()


def test_x2_aurora_ip_configuration(tmp_path):
    # Execute the real IP stage against a CONFIG property recorder. Simulate the rate setter
    # clearing the reference frequency: the s6 order (rate THEN refclk) must survive integration.
    r = _tcl(tmp_path, f'''
set out {{{tmp_path}}}
proc ip_vlnv {{name}} {{ return xilinx.com:ip:$name:mock }}
array set cfg {{}}
proc create_ip {{args}} {{ set n [lindex $args [expr {{[lsearch $args -module_name]+1}}]]; lappend ::ips $n }}
proc get_ips {{args}} {{ return [lindex $args end] }}
proc set_property {{flag values ip}} {{
  foreach {{k v}} $values {{
    set ::cfg($ip,$k) $v
    if {{$k eq "CONFIG.C_LINE_RATE"}} {{ set ::cfg($ip,CONFIG.C_REFCLK_FREQUENCY) 156.25 }}
  }}
}}
proc get_property {{k ip}} {{ return $::cfg($ip,$k) }}
proc generate_target {{what ips}} {{ if {{$ips ne "aur_q0 aur_q1"}} {{ error "wrong IPs" }} }}
proc create_ip_run {{ips}} {{ if {{[llength $ips] != 1}} {{ error "12-3445: one sub-design" }}; puts "RUN $ips" }}
source {{{X2 / 'aurora_ip.tcl'}}}
foreach ip $ips {{ foreach k [lsort [array names cfg $ip,*]] {{ puts "$k=$cfg($k)" }} }}
''')
    assert r.returncode == 0, r.stderr
    assert "RUN aur_q0\n" in r.stdout and "RUN aur_q1\n" in r.stdout
    for ip, quad, lane in (("aur_q0", 12, 48), ("aur_q1", 11, 44)):
        for k, v in (("C_START_QUAD", f"Quad_X1Y{quad}"), ("C_START_LANE", f"X1Y{lane}"),
                     ("C_LINE_RATE", "25.78125"), ("C_REFCLK_FREQUENCY", "161.1328125"),
                     ("C_AURORA_LANES", "4"), ("C_INIT_CLK", "100"), ("SupportLevel", "0"),
                     ("interface_mode", "Streaming"), ("flow_mode", "None"), ("crc_mode", "false")):
            assert f"{ip},CONFIG.{k}={v}\n" in r.stdout


@pytest.mark.parametrize("missing", [False, True])
def test_x2_floorplan_moves_only_adapter_side_to_slr2(tmp_path, missing):
    assert _params(tmp_path).returncode == 0
    out = tmp_path / "reports"
    out.mkdir()
    r = subprocess.run(["tclsh", str(ROOT / "tests/data/vivado_mock_x2.tcl"), str(tmp_path),
                        str(BCU / "vivado"), str(out), "missing" if missing else "ok"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    if missing:
        assert "FAIL x2: no port 1 hub stage u_up_rx" in r.stdout
    else:
        assert "PASS x2 floorplan" in r.stdout


@pytest.mark.parametrize("mode", ["ok", "missing"])
def test_x2_cdc_selectors_and_clock_exception_precedence(mode):
    r = subprocess.run(["tclsh", str(ROOT / "tests/data/vivado_mock_x2_cdc.tcl"), str(X2), mode],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "PASS " in r.stdout, r.stdout + r.stderr


@pytest.mark.parametrize("wns,whs,ok,wmin,wpws", [(0.1, 0.01, True, "", 0.1), (-0.1, 0.01, False, "", 0.1),
                                                 (0.1, -0.01, False, "", 0.1), (0.1, "", False, "", 0.1),
                                                 (-0.042, 0.0, True, "-0.05", 0.1),
                                                 (-0.06, 0.01, False, "-0.05", 0.1),
                                                 (-0.01, -0.001, False, "-0.05", 0.1),
                                                 (0.1, 0.01, False, "0.2", 0.1), (0.1, 0.01, False, "", -0.005),
                                                 (0.1, 0.01, False, "-0.05", -0.005), (0.1, 0.01, False, "", "")])
def test_x2_post_route_requires_setup_and_hold(tmp_path, wns, whs, ok, wmin, wpws):
    # Execute the real pre-bitstream hook with a known GT/SLR netlist. A positive setup result
    # must never hide negative/missing hold, which the common WG post hook historically allowed.
    # OTPU_X2_WNS_MIN (a dev image's setup allowance, -0.1 .. 0) never relaxes hold, nor pulse
    # width (WPWS: the MIGs' TX bitslice Min Skew checks, x2c150's MIG 3), missing or negative.
    row = "" if wpws == "" else f"{wns} 0.000 0 1000 {whs} 0.000 0 1000 {wpws} 0.000 0 500"
    r = _tcl(tmp_path, f'''
set out {{{tmp_path}}}; file mkdir $out/reports
set wns {{{wns}}}; set whs {{{whs}}}
proc report_timing_summary {{args}} {{
  return "Design Timing Summary\n    WNS(ns) TNS(ns) TNS Failing Endpoints TNS Total Endpoints WHS(ns) THS(ns) THS Failing Endpoints THS Total Endpoints WPWS(ns) TPWS(ns) TPWS Failing Endpoints TPWS Total Endpoints\n    -------\n    {row}\n"
}}
if {{"{wmin}" ne ""}} {{ set ::env(OTPU_X2_WNS_MIN) {{{wmin}}} }} else {{ unset -nocomplain ::env(OTPU_X2_WNS_MIN) }}
proc get_cells {{args}} {{
  set f [lindex $args end]
  if {{[regexp {{g_q\\?([01])\\?}} $f -> p]}} {{
    if {{[string match *GTYE4_COMMON* $f]}} {{ return common$p }}
    if {{$p == 0}} {{ return {{GTYE4_CHANNEL_X1Y48 GTYE4_CHANNEL_X1Y49 GTYE4_CHANNEL_X1Y50 GTYE4_CHANNEL_X1Y51}} }}
    return {{GTYE4_CHANNEL_X1Y44 GTYE4_CHANNEL_X1Y45 GTYE4_CHANNEL_X1Y46 GTYE4_CHANNEL_X1Y47}}
  }}
  if {{[string match *u_chub* $f] || [string match *u_dn_tx* $f] || [string match *u_up_rx* $f]}} {{ return slr1 }}
  return slr2
}}
proc get_property {{k c}} {{
  if {{$c eq "common0"}} {{ return GTYE4_COMMON_X1Y12 }}
  if {{$c eq "common1"}} {{ return GTYE4_COMMON_X1Y11 }}
  return $c
}}
proc get_slrs {{args}} {{ return [string toupper [lindex $args end]] }}
proc get_ports {{args}} {{ return {{}} }}
proc report_bus_skew {{args}} {{}}
proc report_exceptions {{args}} {{}}
proc report_clock_utilization {{args}} {{}}
proc report_timing {{args}} {{}}
proc refclk_check {{args}} {{ return {{{{refclk-mock}} {{}}}} }}
rename source real_source
proc source {{f}} {{ if {{[file tail $f] ne "refclk_check.tcl"}} {{ uplevel 1 [list real_source $f] }} }}
set rc [catch {{real_source {{{X2 / 'post_x2.tcl'}}}}} e]
puts "RESULT $rc $e"
''')
    assert r.returncode == 0, r.stdout + r.stderr
    if ok:
        assert "RESULT 0" in r.stdout and "x2 timing / GT / refclk / SLR checks PASS" in r.stdout
    elif wmin == "0.2":
        assert "RESULT 1 OTPU_X2_WNS_MIN '0.2': -0.1 .. 0" in r.stdout
    elif wpws == "" or wpws < 0:
        assert "RESULT 1 x2 checks failed: pulse width failed: WPWS" in r.stdout
    else:
        assert "RESULT 1 x2 checks failed: setup/hold failed:" in r.stdout


@pytest.mark.parametrize("mhz", [100, 150])
def test_x2_full_top_rtl_lint(have_verilator, tmp_path, mhz):
    assert _params(tmp_path, "--core-mhz", str(mhz)).returncode == 0
    sources = [ROOT / "rtl" / f for f in T.sources("bcu1525-x2:st16")]
    extra = [ROOT / f for f in (X2 / "sources_x2.f").read_text().splitlines()]
    # Keep legacy diagnostics visible, as test_bcuwg_build does; new/integrated files must have
    # none beyond the WG top's existing intentional FPGA power-up initializers.
    r = subprocess.run(["verilator", "--lint-only", "-Wall", "-Wno-fatal",
                        "-Wno-UNUSEDSIGNAL", "-Wno-UNUSEDPARAM", "-Wno-DECLFILENAME",
                        "-Wno-PINCONNECTEMPTY", "-Wno-GENUNNAMED", "-Wno-VARHIDDEN",
                        "-DSYNTHESIS", "-DOTPU_FPGA_INIT", "--top-module", "otpu_fpga_top_bcu_wg",
                        f"-GCORE_KHZ={mhz * 1000}", "-GRLY=64'h2222555555552222",
                        str(tmp_path / "otpu_bcu_wg_defines.vh"),
                        str(BCU / "sim/unisim_dsp48e2_stub.sv"), str(BCU / "sim/unisim_stubs.sv"),
                        str(BCU / "link/sim/otpu_link_phy_stubs.sv"),
                        *map(str, extra), *map(str, sources), str(BCU / "sim/otpu_bdw_wrapper_stub.sv")],
                       capture_output=True, text=True, timeout=240)
    (tmp_path / "lint.log").write_text(r.stdout + r.stderr)
    msgs = [m for m in (r.stdout + r.stderr).splitlines() if m.startswith("%")]
    assert r.returncode == 0 and not [m for m in msgs if m.startswith("%Error")], msgs[:30]
    new = ("otpu_fpga_top_bcu_wg.sv", "otpu_link", "otpu_bcu_wg_sys.sv")
    assert not [m for m in msgs if any(f in m for f in new) and "PROCASSINIT" not in m], msgs[:30]


def test_x2_yosys_rtl_elaboration(tmp_path):
    yosys = os.environ.get("YOSYS") or shutil.which("yosys")
    if not yosys:
        pytest.skip("set YOSYS to a Yosys with the slang plugin")
    assert _params(tmp_path).returncode == 0
    sources = [tmp_path / "otpu_bcu_wg_defines.vh", BCU / "sim/unisim_dsp48e2_stub.sv",
               BCU / "sim/unisim_stubs.sv", BCU / "link/sim/otpu_link_phy_stubs.sv"]
    sources += [ROOT / f for f in (X2 / "sources_x2.f").read_text().splitlines()]
    sources += [ROOT / "rtl" / f for f in T.sources("bcu1525-x2:st16")]
    sources += [BCU / "sim/otpu_bdw_wrapper_stub.sv"]
    # Independent frontend, full hierarchy at the real card memory sizes. AST compilation checks
    # SV elaboration without expanding the enormous TMEM/MXU into a synthesis/technology mapping.
    cmd = "read_slang --single-unit --allow-use-before-declare --ast-compilation-only --top otpu_fpga_top_bcu_wg "
    files = tmp_path / "elaborate.f"
    files.write_text("\n".join(f'"{f}"' for f in sources) + "\n")
    cmd += f"-f {files}"
    script = tmp_path / "elaborate.ys"
    script.write_text(cmd + "\n")
    r = subprocess.run([yosys, "-Q", "-T", "-m", "slang", "-s", str(script)],
                       capture_output=True, text=True, timeout=240)
    (tmp_path / "yosys.log").write_text(r.stdout + r.stderr)
    assert r.returncode == 0, (r.stdout + r.stderr)[-6000:]
