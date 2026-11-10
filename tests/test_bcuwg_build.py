"""The BCU1525 WG build's inputs (targets/bcu1525-wg.toml; boards/bcu1525/run_vivado.sh IMAGE=wg):
make_wg.py's defines and parameters, the pin constraints (gen_wg_pins.py) against the top's ports,
and the Vivado project script's file list. No Vivado here."""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from opentpu import target as T

ROOT = Path(__file__).resolve().parents[1]
BCU = ROOT / "boards" / "bcu1525"
TOP = ROOT / "rtl" / "boards" / "bcu1525" / "otpu_fpga_top_bcu_wg.sv"


def _top_ports() -> dict[str, int]:
    """The WG top's ports and their widths (the DIMM macro expanded for C0 .. C3; PCIE_LANES 16)."""
    src = TOP.read_text()
    src = re.sub(r"`ifdef OTPU_WG_X2.*?`endif", "", src, flags=re.S)
    head = src[src.index("module otpu_fpga_top_bcu_wg"):src.index(");")]
    mac = re.search(r"`define OTPU_WG_DIMM\(N\)(.*?)\n  `OTPU_WG_DIMM\(0\)", head, re.S).group(1)
    body = re.sub(r"`define OTPU_WG_DIMM\(N\).*?`undef OTPU_WG_DIMM", "", head, flags=re.S)
    body += "".join(mac.replace("``N``", str(d)).replace("\\", "") for d in range(4))
    ports = {}
    for m in re.finditer(r"\b(?:input|output|inout)\s+(?:logic|wire)\s*(\[[^\]]+\])?\s*(\w+)", body):
        rng, name = m.groups()
        if rng:
            hi, lo = rng[1:-1].split(":")
            w = eval(hi.replace("PCIE_LANES", "16")) - int(lo) + 1
        else:
            w = 1
        ports[name] = w
    return ports


def _xdc_pins(text: str) -> dict[str, str]:
    pins = {}
    for m in re.finditer(r"set_property PACKAGE_PIN (\S+) \[get_ports (?:\{([^}]+)\}|(\S+?))\]$", text,
                         re.M):
        pin, port = m.group(1), m.group(2) or m.group(3)
        assert port not in pins, port
        pins[port] = pin
    return pins


def test_wg_pins_are_generated_and_cover_the_top(tmp_path):
    """bcu1525_wg_pins.xdc is gen_wg_pins.py's current output, and every bit of the top's ports has
    one pad (the GTs' own: XDMA places them) and every pad a port, none twice."""
    xdc = BCU / "constraints" / "bcu1525_wg_pins.xdc"
    clk = BCU / "constraints" / "bcu1525_wg_clocks.xdc"
    before, clocks = xdc.read_text(), clk.read_text()
    subprocess.run([sys.executable, str(BCU / "scripts" / "gen_wg_pins.py")], check=True,
                   capture_output=True)
    assert xdc.read_text() == before, "bcu1525_wg_pins.xdc is stale: run gen_wg_pins.py"
    assert clk.read_text() == clocks, "bcu1525_wg_clocks.xdc is stale: run gen_wg_pins.py"
    # the clocks no IP creates, in their own file (the normal order), none in the pins' (late)
    assert "create_clock" not in before
    assert re.findall(r"create_clock .*\[get_ports (\S+)\]", clocks) == ["clk300_1_p", "pcie_refclk_p"]
    pins = _xdc_pins(before)
    assert len(set(pins.values())) == len(pins), "a pad used twice"
    want = set()
    for name, w in _top_ports().items():
        if name.startswith("pcie_mgt_"):
            continue
        want |= {name} if w == 1 else {f"{name}[{i}]" for i in range(w)}
    assert set(pins) == want, (sorted(want - set(pins))[:8], sorted(set(pins) - want)[:8])
    # a MIG's pads only for its own signals: one DIMM's 116 (+ the 6 idle lines)
    for d in range(4):
        assert sum(p.startswith(f"ddr4_c{d}_") for p in pins) == 116 + 6
    # no bank VREF: the MIGs train each data byte's own (HPIO_VREF); LiteDRAM's 0.84 V left the
    # address banks at 0.600 with a conflict (BIIVRC-1)
    assert "INTERNAL_VREF" not in before


def test_make_wg_writes_the_targets_parameters(tmp_path):
    out = tmp_path / "wg"
    r = subprocess.run([sys.executable, str(BCU / "scripts" / "make_wg.py"), str(out),
                        "--core-mhz", "150"], check=True, capture_output=True, text=True)
    assert "core 150000 kHz" in r.stdout
    vh = (out / "otpu_bcu_wg_defines.vh").read_text()
    for k, v in (("MCOLS", 8), ("WG", 8), ("NAT_XPOST", 1), ("TMEM_PMP", 1), ("PLATFORM", 5)):
        assert f".{k}({v})" in vh, k
    for k in ("S", "NCH", "CB", "CORE_KHZ", "DDR_MTS"):         # the top's own
        assert f".{k}(" not in vh, k
    assert "`define OTPU_XPOST 1" in vh and "`define OTPU_XSW 11" in vh
    tcl = (out / "wg_params.tcl").read_text()
    assert "set CORE_KHZ 150000" in tcl
    # the governor's identity words (otpu_bcu_gov IMAGE ..): the board's own PLATFORM, ARCH_HASH
    # and BUILD_ID, the values OTPU_BOARD gives otpu_board
    h = re.search(r"set ARCH_HASH ([0-9a-f]{8})", tcl).group(1)
    b = re.search(r"set BUILD_ID ([0-9a-f]{8})", tcl).group(1)
    assert "`define OTPU_WG_PLATFORM 5\n" in vh
    assert f"`define OTPU_WG_CFG_HASH 32'h{h.upper()}\n" in vh and f".ARCH_HASH(32'h{h})" in vh
    assert f"`define OTPU_WG_BUILD_ID 32'h{b.upper()}\n" in vh
    top = (ROOT / "rtl/boards/bcu1525/otpu_fpga_top_bcu_wg.sv").read_text()
    assert ".IMG_PLATFORM(`OTPU_WG_PLATFORM), .IMG_SLICES(S), .IMG_CARDS(GOV_CARDS)" in top
    assert ".IMG_HASH(`OTPU_WG_CFG_HASH), .IMG_BUILD(`OTPU_WG_BUILD_ID)) u_gov (" in top
    # the slice in SLR2 (the default): C0 two crossings from it (9), C1 and C2 one (5), C3 beside it
    # (2); channel j is DIMM j >> 1's, the floorplan's SLR the same
    assert "set RLY 22555599" in tcl and "set WG_SLR 2" in tcl
    rtl = re.search(r"set RTL \[list (.*)\]", tcl).group(1).split()
    assert rtl == [f"rtl/{f}" for f in T.sources("bcu1525-wg")]
    assert rtl[-1] == "rtl/boards/bcu1525/otpu_fpga_top_bcu_wg.sv"
    assert all((ROOT / f).exists() for f in rtl)


def test_make_wg_caw_follows_channel_bytes(tmp_path):
    """The channels' beat address bits from the target's channel_bytes, the same in the defines
    (the top's OTPU_WG_CAW) and wg_params.tcl (the top's CAW generic, bd_wg.tcl's DMA windows):
    22 at 512 MiB a channel, 25 at 4 GiB (the 32 GB card); 16 GB DIMMs' 26 and odd sizes refused."""
    import importlib.util
    subprocess.run([sys.executable, str(BCU / "scripts" / "make_wg.py"), str(tmp_path)],
                   check=True, capture_output=True)
    n = (T.load("bcu1525-wg").platform["channel_bytes"] // 128 - 1).bit_length()
    assert f"`define OTPU_WG_CAW {n}\n" in (tmp_path / "otpu_bcu_wg_defines.vh").read_text()
    assert f"set WG_CAW {n}\n" in (tmp_path / "wg_params.tcl").read_text()
    spec = importlib.util.spec_from_file_location("make_wg", BCU / "scripts" / "make_wg.py")
    mw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mw)
    assert mw.caw(512 << 20, 128) == 22 and mw.caw(4 << 30, 128) == 25
    for bad in (8 << 30, 256 << 20, 3 << 28):
        with pytest.raises(SystemExit):
            mw.caw(bad, 128)
    tcl = (BCU / "vivado" / "create_project_wg.tcl").read_text()
    assert "CAW=$WG_CAW" in tcl and "WG_CAW BUILD_ID" in tcl
    bd = (BCU / "vivado" / "bd_wg.tcl").read_text()
    assert "-range ${DIMM_GIB}G" in bd and "CONFIG.ADDR_WIDTH 64" in bd
    assert "C0.DDR4_AxiAddressWidth 33" in bd and "MTA8ATF1G64AZ-2G3" in bd


def test_make_wg_bhw_follows_the_targets_variant(tmp_path):
    """The pairs' bank-half window (OTPU_WG_BHW: a channel's bytes, log2) from the target's BHW
    (the slice's): 0 for bcu1525-wg, 30 for bcu1525-wg:bhw (33: the slice's first 8 GiB over 8
    channels), its CFG_HASH the variant's; another target refused. bcu1525-wg:bhw16 is bhw with
    otpu_board's ST16 1 (e6m9: docs/isa.md "ST16"), which the others leave out."""
    mw = [sys.executable, str(BCU / "scripts" / "make_wg.py")]
    for target, bhw in (("bcu1525-wg", 0), ("bcu1525-wg:bhw", 30), ("bcu1525-wg:bhw16", 30)):
        out = tmp_path / target.replace(":", "_")
        r = subprocess.run([*mw, str(out), "--target", target], check=True, capture_output=True,
                           text=True)
        assert f"BHW {bhw}," in r.stdout
        vh = (out / "otpu_bcu_wg_defines.vh").read_text()
        assert f"`define OTPU_WG_BHW {bhw}\n" in vh and ".BHW(" not in vh   # (the top's own)
        assert (".ST16(1)" in vh) == ("ST16" in vh) == target.endswith("16")
        h = T.arch_hash(T.config(target))
        assert f"set ARCH_HASH {h:08x}\n" in (out / "wg_params.tcl").read_text()
    r = subprocess.run([*mw, str(tmp_path / "x"), "--target", "ypcb-00338"], capture_output=True,
                       text=True)
    assert r.returncode and "a variant of it" in r.stderr


@pytest.mark.parametrize("slr,rly", [("1", "55222255"), ("0", "99555522")])
def test_make_wg_rly_follows_the_slices_slr(tmp_path, slr, rly):
    r = subprocess.run([sys.executable, str(BCU / "scripts" / "make_wg.py"), str(tmp_path),
                        "--slr", slr], check=True, capture_output=True, text=True)
    assert f"slice in SLR{slr}" in r.stdout
    tcl = (tmp_path / "wg_params.tcl").read_text()
    assert f"set RLY {rly}" in tcl and f"set WG_SLR {slr}" in tcl


def test_wg_floorplan_takes_make_wgs_slr():
    fp = (BCU / "constraints" / "bcu1525_wg_floorplan.tcl").read_text()
    assert "wg_params.tcl" in fp and "resize_pblock $pb -add SLR$fp_slr" in fp
    assert "set WG_SLR" in (BCU / "scripts" / "make_wg.py").read_text()
    assert 'export OTPU_WG_OUT="$out"' in (BCU / "run_vivado.sh").read_text()


@pytest.mark.parametrize("name,ok", [("bcu1525-wg", True), ("bcu1525-s2", True), ("bcu1525-wg:bhw", True),
                                     ("bcu1525-nope", False), ("bhw", False)])
def test_run_vivado_takes_a_target_variant(name, ok):
    """run_vivado.sh's WG_TARGET check passes a variant (bcu1525-wg:bhw, which opentpu.target and
    make_wg.py take): it refused it, and pbhw's build stopped 10 s in (2026-10-31)."""
    line = next(l for l in (BCU / "run_vivado.sh").read_text().splitlines() if l.startswith('case "${tname'))
    r = subprocess.run(["bash", "-c", f"tname='{name}'; {line}; echo PASSED"], capture_output=True, text=True)
    assert ("PASSED" in r.stdout) == ok and (r.returncode == 0) == ok


@pytest.mark.parametrize("f", ["bcu1525_wg_floorplan.tcl", "bcu1525_wg_top.tcl"])
def test_wg_constraint_tcl_has_no_top_level_return(f):
    """In a file read as a constraint (read_xdc) a return does not end it: the floorplan went on to
    resize_pblock with no SLR after its "no floorplan" branch (dev100b, 2026-10-28)."""
    depth, in_proc = 0, False
    for n, line in enumerate((BCU / "constraints" / f).read_text().splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        if depth == 0 and re.match(r"\s*proc\b", line):
            in_proc = True
        assert in_proc or not re.match(r"\s*return\b", line), f"{f}:{n}: return outside a proc"
        depth += line.count("{") - line.count("}")
        if depth == 0:
            in_proc = False


def test_make_wg_refuses_an_odd_divide(tmp_path):
    r = subprocess.run([sys.executable, str(BCU / "scripts" / "make_wg.py"), str(tmp_path),
                        "--core-mhz", "133.33"], capture_output=True, text=True)
    assert r.returncode != 0 and "even divide" in r.stderr


def test_wg_project_reads_its_constraint_files():
    tcl = (BCU / "vivado" / "create_project_wg.tcl").read_text()
    for f in ("bcu1525_wg_pins.xdc", "bcu1525_wg_clocks.xdc", "otpu_hbm_ch.tcl", "otpu_mxu_stage.tcl", "otpu_clk2x.tcl",
              "bcu1525_wg_floorplan.tcl", "bcu1525_wg_top.tcl", "bcu1525_wg_check.tcl", "bd_wg.tcl",
              "bcu1525_wg_place_pre.tcl"):
        assert f in tcl, f
    for f in ("bcu1525_wg_floorplan.tcl", "bcu1525_wg_top.tcl"):
        assert (BCU / "constraints" / f).exists(), f
    for f in ("bcu1525_wg_check.tcl", "bcu1525_wg_place_pre.tcl", "impl_resume.tcl", "post_impl.tcl"):
        assert (BCU / "vivado" / f).exists(), f
    assert (ROOT / "boards/bcu1525/constraints/otpu_hbm_ch.tcl").exists()
    assert (ROOT / "boards/bcu1525/constraints/otpu_mxu_stage.tcl").exists()
    assert (ROOT / "boards/ypcb-00338/constraints/otpu_clk2x.tcl").exists()


@pytest.mark.parametrize("caw,bhw", [(None, None), (25, None), (25, 30)])
def test_wg_top_lints_against_the_block_designs_ports(have_verilator, tmp_path, caw, bhw):
    """Verilator lint of otpu_fpga_top_bcu_wg (its RTL, make_wg.py's defines) against the block
    design's wrapper as bd_wg.tcl declares it (boards/bcu1525/sim/otpu_bdw_wrapper_stub.sv): no
    error, and in the WG files no warning but the flip-flops' initial values (PROCASSINIT) -- the
    ports' widths included. At the target's CAW, and at 25 (4 GiB a channel, the 32 GB card:
    otpu_board's addresses at 36 bits), there with the bank-half window too (BHW 30: the pairs'
    swap, the host's remap, CAPS2's field)."""
    subprocess.run([sys.executable, str(BCU / "scripts" / "make_wg.py"), str(tmp_path)],
                   check=True, capture_output=True)
    vh = tmp_path / "otpu_bcu_wg_defines.vh"
    if caw is not None and ".AW(" not in vh.read_text():   # (until the target sets AW)
        vh.write_text(vh.read_text().replace("`define OTPU_BOARD ", "`define OTPU_BOARD , .AW(36)", 1))
    rtl = [str(ROOT / "rtl" / f) for f in T.sources("bcu1525-wg")]
    r = subprocess.run(
        ["verilator", "--lint-only", "-Wall", "-Wno-fatal", "-Wno-UNUSEDSIGNAL", "-Wno-UNUSEDPARAM",
         "-Wno-DECLFILENAME", "-Wno-PINCONNECTEMPTY", "-Wno-GENUNNAMED", "-Wno-VARHIDDEN",
         "-DSYNTHESIS", "-DOTPU_FPGA_INIT", "--top-module", "otpu_fpga_top_bcu_wg",
         *([] if caw is None else [f"-GCAW={caw}"]), *([] if bhw is None else [f"-GBHW={bhw}"]),
         str(BCU / "sim" / "unisim_dsp48e2_stub.sv"), str(BCU / "sim" / "unisim_stubs.sv"),
         str(tmp_path / "otpu_bcu_wg_defines.vh"), *rtl,
         str(BCU / "sim" / "otpu_bdw_wrapper_stub.sv")], capture_output=True, text=True)
    msgs = [m for m in (r.stdout + r.stderr).splitlines() if m.startswith("%")]
    assert not [m for m in msgs if m.startswith("%Error")], msgs[:20]
    wg = ("otpu_fpga_top_bcu_wg.sv", "otpu_bcu_wg_sys.sv", "otpu_mig_pair.sv", "otpu_mig_hmux.sv")
    bad = [m for m in msgs if any(f in m for f in wg) and "PROCASSINIT" not in m]
    assert not bad, bad[:20]


def _bcu_fp():
    import importlib.util
    spec = importlib.util.spec_from_file_location("bcu_fp", BCU / "scripts" / "bcu_fp.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize("fp,slrs,dmap,rly", [
    ("c", "0 2", "E4", "2222555555552222"),   # slice 0 SLR0 on C0 (local) + C1, slice 1 SLR2 on C2 + C3
    ("b", "1 2", "E4", "2222555522225555"),   # slice 0 SLR1 on C0 + C1 (local), slice 1 as c's
    ("a", "1 2", "39", "9999222222222222"),   # slice 0 SLR1 on C1 + C2, slice 1 SLR2 on C3 + C0
])
def test_make_wg_two_slices_follow_the_floorplan(tmp_path, fp, slrs, dmap, rly):
    """bcu1525-s2 (make_wg.py --target, --plan; scripts/bcu_fp.py): the slices' SLRs, the DIMM
    slots' DIMMs (DMAP: slot i on DIMM DMAP[2 i +: 2]) and the 16 channels' bridge pipelines (RLY,
    board channel g's N at bits 4 g: a DIMM k crossings from its slice 4 k + 1, a local one 2);
    CAW 24 (2 GiB a channel, four a DIMM); --slr / --rly and --plan at one slice refused."""
    r = subprocess.run([sys.executable, str(BCU / "scripts" / "make_wg.py"), str(tmp_path),
                        "--target", "bcu1525-s2", "--plan", fp, "--core-mhz", "150"],
                       check=True, capture_output=True, text=True)
    assert f"floorplan {fp}" in r.stdout and "RLY 64'h" in r.stdout
    vh = (tmp_path / "otpu_bcu_wg_defines.vh").read_text()
    assert "`define OTPU_WG_S 2\n" in vh and f"`define OTPU_WG_DMAP 8'h{dmap}\n" in vh
    assert "`define OTPU_WG_CAW 24\n" in vh and ".WG(8)" in vh and ".S(" not in vh
    assert ".COLL_PIPE(1)" in vh and ".XRS(2)" in vh
    tcl = (tmp_path / "wg_params.tcl").read_text()
    for s in (f"set RLY {rly}\n", "set WG_S 2\n", f"set WG_SLRS [list {slrs}]\n", f"set WG_PLAN {fp}\n",
              f"set WG_DMAP {dmap}\n", "set WG_CAW 24\n", "set CORE_KHZ 150000\n"):
        assert s in tcl, s
    p = _bcu_fp().plan(fp)
    assert f"{p['rly']:016x}" == rly and f"{p['dmap']:02X}" == dmap
    rtl = re.search(r"set RTL \[list (.*)\]", tcl).group(1).split()
    assert rtl == [f"rtl/{f}" for f in T.sources("bcu1525-s2")]
    for bad in (["--slr", "1"], ["--rly", "2,2,2,2"]):
        r = subprocess.run([sys.executable, str(BCU / "scripts" / "make_wg.py"), str(tmp_path / "x"),
                            "--target", "bcu1525-s2", *bad], capture_output=True, text=True)
        assert r.returncode != 0 and "--plan" in r.stderr
    r = subprocess.run([sys.executable, str(BCU / "scripts" / "make_wg.py"), str(tmp_path / "y"),
                        "--plan", "c"], capture_output=True, text=True)
    assert r.returncode != 0 and "--plan is bcu1525-s2's" in r.stderr


def test_bcu_fp_floorplans():
    """Each floorplan's slices own two DIMMs each, all four once; b and c the identity map (the
    host's DMA windows unpermuted), a's slice 1 two crossings from C0 (9 stages)."""
    m = _bcu_fp()
    for fp in m.FLOORPLANS:
        p = m.plan(fp)
        assert sorted(d for pair in p["dimms"] for d in pair) == [0, 1, 2, 3]
        assert all(0 <= n <= 2 for c in p["crossings"] for n in c)
        assert m.params(fp) == f"DMAP=0x{p['dmap']:02x},RLY=0x{p['rly']:016x}"
    assert m.plan("b")["dmap"] == m.plan("c")["dmap"] == 0xE4
    assert m.plan("a")["crossings"] == [[0, 0], [0, 2]] and m.stages(2, 0) == 9


def test_wg_vivado_scripts_take_two_slices():
    """create_project_wg.tcl's RLY generic 32 x WG_S bits; bd_wg.tcl's DMA window a DIMM's
    2 x WG_S channels (8 GiB at WG_S 2, WG_CAW 24); the floorplan a pblock a slice and the control's
    in SLR1; the top's constraints the 16 channels' posted-beat counts; run_vivado.sh's
    WG_TARGET / WG_PLAN into make_wg.py."""
    cp = (BCU / "vivado" / "create_project_wg.tcl").read_text()
    assert "set RLYW [expr {32 * $WG_S}]" in cp and "RLY=$RLYW'h$RLY" in cp
    bd = (BCU / "vivado" / "bd_wg.tcl").read_text()
    assert "set DIMM_GIB [expr {1 << ($WG_CAW - 23 + $WG_S)}]" in bd
    assert "$WG_CAW + $WG_S > 26" in bd
    fp = (BCU / "constraints" / "bcu1525_wg_floorplan.tcl").read_text()
    assert "set WG_SLRS" in fp and "create_pblock pblock_slice$s" in fp and "pblock_ctl" in fp
    assert "g_slice?$s?." in fp and "u_sys/g_ch?$j?.u_br/u_cs" in fp
    top = (BCU / "constraints" / "bcu1525_wg_top.tcl").read_text()
    assert "u_board/g_slice?$s?." in top and "$n_pc == $n_ch" in top
    rv = (BCU / "run_vivado.sh").read_text()
    assert 'tname="${WG_TARGET:-bcu1525-wg}"' in rv and 'wg_args+=(--plan "$WG_PLAN")' in rv


@pytest.mark.parametrize("fp", ["c", "a"])
def test_wg_top_lints_at_two_slices(have_verilator, tmp_path, fp):
    """The top at S 2 (make_wg.py --target bcu1525-s2 --plan fp: its defines, RLY 64 bits) against
    the block design's wrapper: no error, no warning in the WG files but PROCASSINIT (a's DIMM map
    permutes the DMA windows: g_dma's slot_of)."""
    subprocess.run([sys.executable, str(BCU / "scripts" / "make_wg.py"), str(tmp_path),
                    "--target", "bcu1525-s2", "--plan", fp], check=True, capture_output=True)
    rly = re.search(r"set RLY (\w+)", (tmp_path / "wg_params.tcl").read_text()).group(1)
    assert len(rly) == 16
    rtl = [str(ROOT / "rtl" / f) for f in T.sources("bcu1525-s2")]
    r = subprocess.run(
        ["verilator", "--lint-only", "-Wall", "-Wno-fatal", "-Wno-UNUSEDSIGNAL", "-Wno-UNUSEDPARAM",
         "-Wno-DECLFILENAME", "-Wno-PINCONNECTEMPTY", "-Wno-GENUNNAMED", "-Wno-VARHIDDEN",
         "-DSYNTHESIS", "-DOTPU_FPGA_INIT", "--top-module", "otpu_fpga_top_bcu_wg", f"-GRLY=64'h{rly}",
         str(BCU / "sim" / "unisim_dsp48e2_stub.sv"), str(BCU / "sim" / "unisim_stubs.sv"),
         str(tmp_path / "otpu_bcu_wg_defines.vh"), *rtl,
         str(BCU / "sim" / "otpu_bdw_wrapper_stub.sv")], capture_output=True, text=True)
    msgs = [m for m in (r.stdout + r.stderr).splitlines() if m.startswith("%")]
    assert not [m for m in msgs if m.startswith("%Error")], msgs[:20]
    wg = ("otpu_fpga_top_bcu_wg.sv", "otpu_bcu_wg_sys.sv", "otpu_mig_pair.sv", "otpu_mig_hmux.sv")
    bad = [m for m in msgs if any(f in m for f in wg) and "PROCASSINIT" not in m]
    assert not bad, bad[:20]


def _mock_vivado(tmp_path, s, mode, env=None, plan="c"):
    """tests/data/vivado_mock_wg.tcl on make_wg.py's parameters (S 1: bcu1525-wg, its slice in SLR2;
    S 2: bcu1525-s2 floorplan plan, default c)."""
    import os
    import shutil
    if not shutil.which("tclsh"):
        pytest.skip("no tclsh")
    out = tmp_path / f"wg{s}{plan}"
    args = ["--target", "bcu1525-s2", "--plan", plan] if s == 2 else []
    subprocess.run([sys.executable, str(BCU / "scripts" / "make_wg.py"), str(out), *args],
                   check=True, capture_output=True, text=True)
    e = {k: v for k, v in os.environ.items()
         if not k.startswith("OTPU_WG_") and k != "OTPU_CLK_HOOK"} | (env or {})
    r = subprocess.run(["tclsh", str(ROOT / "tests" / "data" / "vivado_mock_wg.tcl"), str(s), str(out),
                        mode, str(BCU / "vivado"), str(tmp_path)], capture_output=True, text=True,
                       env=e, check=True)
    return r.stdout


def test_wg_synth_check_finds_the_floorplans_cells_and_counts_each_pblock(tmp_path):
    """bcu1525_wg_synth_check.tcl (build.tcl after synthesis, at two slices): the floorplan's pblocks
    on the netlist, its cells all found, then each pblock's leaf primitives by type (a slice: its 12
    MXU DSPs and 3 more, its adapter's and 8 bridge core halves' LUTs, its side of the crossings'
    stages and its bridges' pipeline stages 0 .. N - 4 k (32 FFs); block RAM RAMB36 + RAMB18 / 2; a
    macro once, not its cells; the control's: its 4 crossing stages, sc_dma's FF, the crossing
    channels' stages 2-4, 48 FFs, and the top's reset hop, acc_rst_p's 5 FFs and acc_rq[0])."""
    o = _mock_vivado(tmp_path, 2, "chk")
    assert "rc 0" in o and "the floorplan's cells all found" in o
    for pb in ("pblock_ctl", "pblock_slice0", "pblock_slice1"):    # (-pblocks takes one: Common 17-69)
        assert f"RU -pblocks {pb} -append -file {tmp_path}/reports/synth_util_pblocks.rpt" in o
    for s, slr in ((0, 0), (1, 2)):
        assert (f"pblock_slice{s} (SLR{slr}): LUT 65 (0.0%), LUTRAM/SRL 2, FF 85 (0.0%), BRAM 1.5 (0.2%), "
                "URAM 1 (0.3%), DSP 15 (0.7%)") in o
    assert "pblock_ctl (SLR1): LUT 0 (0.0%), LUTRAM/SRL 0, FF 59 (0.0%)" in o
    assert "pblock_mxu" not in o


def test_wg_synth_check_runs_on_the_floorplan_open_run_made(tmp_path):
    """open_run synth_1 applies the late Tcl constraints, the floorplan among them, before build.tcl's
    synthesis check (s2syn 2026-10-31: the check's own run of the floorplan stopped at Vivado 12-795,
    pblock 'pblock_slice0' already exists): the check deletes those pblocks and counts its own run,
    the same pblocks and counts as on a netlist without them."""
    o = _mock_vivado(tmp_path, 2, "open", {"OTPU_WG_MXU_COLS0": "3 4"})
    assert "rc 0" in o and "the floorplan's cells all found" in o and "12-795" not in o
    assert [ln.split()[2] for ln in o.splitlines() if ln.startswith("RU -pblocks")] == [
        "pblock_ctl", "pblock_mxu0", "pblock_slice0", "pblock_slice1"]
    assert "pblock_slice1 (SLR2): LUT 65 (0.0%), LUTRAM/SRL 2, FF 85 (0.0%)" in o
    assert ("PB pblock_mxu0 ranges 2 DSP48E2_X3Y0:DSP48E2_X3Y119 DSP48E2_X4Y0:DSP48E2_X4Y119 "
            "parent pblock_slice0 cells 12") in o
    ck = (BCU / "vivado" / "bcu1525_wg_synth_check.tcl").read_text()
    assert ck.index("delete_pblock") < ck.index("bcu1525_wg_floorplan.tcl]")


def test_wg_floorplan_puts_each_slices_mxu_dsps_on_its_columns(tmp_path):
    """OTPU_WG_MXU_COLS<s> (run_vivado.sh WG_MXU_COLS<s>): slice s's MXU DSPs in pblock_mxu<s>, the
    columns over its SLR, nested in pblock_slice<s>; unset, no pblock_mxu (as before); a column
    without DSP sites in the SLR, or fewer sites than DSPs, fails the floorplan (the synthesis check
    stops the build)."""
    o = _mock_vivado(tmp_path, 2, "chk", {"OTPU_WG_MXU_COLS0": "3 4", "OTPU_WG_MXU_COLS1": "5,6"})
    assert "rc 0" in o
    assert ("PB pblock_mxu0 ranges 2 DSP48E2_X3Y0:DSP48E2_X3Y119 DSP48E2_X4Y0:DSP48E2_X4Y119 "
            "parent pblock_slice0 cells 12") in o
    assert ("PB pblock_mxu1 ranges 2 DSP48E2_X5Y240:DSP48E2_X5Y359 DSP48E2_X6Y240:DSP48E2_X6Y359 "
            "parent pblock_slice1 cells 12") in o
    assert "pblock_slice0 (SLR0): LUT 65 (0.0%), LUTRAM/SRL 2, FF 85 (0.0%)" in o   # (the slice's whole)
    for env, msg in (({"OTPU_WG_MXU_COLS0": "40"}, "columns without a DSP48E2 site in SLR0 '40'"),
                     ({"OTPU_WG_MXU_COLS1": "18"}, "12 MXU DSPs, columns without a DSP48E2 site in SLR2 '', 5 sites")):
        o = _mock_vivado(tmp_path, 2, "chk", env)
        assert "rc 1 bcu1525_wg_synth_check.tcl: the floorplan finished 0 of 1" in o and msg in o
    o = _mock_vivado(tmp_path, 1, "fp")
    assert "ok bcu1525_wg_floorplan 1 fail \n" in o and "PB pblock_slice ranges 1 SLR2 parent - cells 9\n" in o
    assert "pblock_mxu" not in o
    o = _mock_vivado(tmp_path, 1, "fp", {"OTPU_WG_MXU_COLS0": "2"})
    assert "PB pblock_mxu0 ranges 1 DSP48E2_X2Y240:DSP48E2_X2Y359 parent pblock_slice cells 12" in o


def test_wg_floorplan_places_the_bridge_pipelines_sc_dma_and_the_mmcm(tmp_path):
    """Two slices (s2syn's aclk, xdma_aclk and core -> acc_clk2x paths, 2026-11-01): each bridge
    pipeline's stages 0 .. N - 4 k in its slice's pblock, the stages after them in the next SLR's
    toward its DIMM (stage N with the placer; an SLR without a pblock gets pblock_slr<k>), sc_dma and
    the top's reset hop into acc_clk (acc_rst_p's core_clk stages, acc_rq[0]; s2c150's hold) in
    SLR1's pblock, the clock wizard's MMCM at OTPU_WG_MMCM (default MMCM_X1Y5; none: no LOC; a bad
    site fails the floorplan). One slice: none of it."""
    mm = "LOC u_bd/otpu_bdw_i/clk_wiz_0/inst/mmcme4_adv_inst"
    o = _mock_vivado(tmp_path, 2, "fp")
    assert "ok bcu1525_wg_floorplan 1 fail \n" in o
    # c: slices in SLR0 / SLR2, channels 4-11 one SLR from their DIMMs (N 5: stages 0, 1 with the
    # slice, 2-4 in SLR1), the others beside theirs (N 2: stages 0, 1); 2 FFs a stage
    assert "64 stage registers in their slices' SLRs, 48 on toward their DIMMs'" in o
    assert "sc_dma in SLR1 (pblock_ctl)" in o and f"{mm} MMCM_X1Y5\n" in o
    assert "the reset's hop in SLR1 (pblock_ctl): 5 acc_rst_p registers and g_arq.acc_rq_reg[0]" in o
    for pb, slr, n in (("pblock_ctl", 1, 10 + 1 + 48 + 6), ("pblock_slice0", 0, 15 + 32), ("pblock_slice1", 2, 14 + 32)):
        assert f"PB {pb} ranges 1 SLR{slr} parent - cells {n}\n" in o, pb
    # b: slice 0 in SLR1 (the control and sc_dma with it), C0's channels 0-3 one SLR down
    o = _mock_vivado(tmp_path, 2, "fp", plan="b")
    assert "ok bcu1525_wg_floorplan 1 fail \n" in o and "sc_dma in SLR1 (pblock_slice0)" in o
    assert "the reset's hop in SLR1 (pblock_slice0)" in o
    assert "PB pblock_slr0 ranges 1 SLR0 parent - cells 24\n" in o and "pblock_ctl" not in o
    o = _mock_vivado(tmp_path, 2, "fp", {"OTPU_WG_MMCM": "MMCM_X0Y5"})
    assert f"{mm} MMCM_X0Y5\n" in o
    o = _mock_vivado(tmp_path, 2, "fp", {"OTPU_WG_MMCM": "none"})
    assert "ok bcu1525_wg_floorplan 1 fail \n" in o and mm not in o
    o = _mock_vivado(tmp_path, 2, "chk", {"OTPU_WG_MMCM": "MMCM_X9Y5"})
    assert "rc 1 bcu1525_wg_synth_check.tcl:" in o and "LOC MMCM_X9Y5 on" in o
    o = _mock_vivado(tmp_path, 1, "fp")
    assert mm not in o and "sc_dma" not in o and "pipelines" not in o and "reset's hop" not in o


def test_wg_build_checks_the_synthesis_and_stops_on_request():
    """build.tcl sources the synthesis check at two slices (or with OTPU_WG_STOP=synth) in a scope of
    its own, before it closes the synthesized design, and OTPU_WG_STOP=synth (run_vivado.sh
    WG_STOP=synth) then stops it: the S=2 pre-check, synthesis and its checks only; WG_STOP=link
    stops in opt_design's pre-hook, the netlist linked and its constraints checked."""
    b = (BCU / "vivado" / "build.tcl").read_text()
    assert "regexp {set WG_S 2} $wg_txt] || $wg_stop" in b
    assert "apply {{here out} { source $here/bcu1525_wg_synth_check.tcl }} $here $out" in b
    assert b.index("bcu1525_wg_synth_check.tcl") < b.index("close_design") < b.index("OTPU_WG_STOP=synth)")
    ck = (BCU / "vivado" / "bcu1525_wg_check.tcl").read_text()        # (WG_STOP=link: opt_design's pre-hook)
    assert ck.index("every constraint file finished") < ck.index('$::env(OTPU_WG_STOP) eq "link"')
    rv = (BCU / "run_vivado.sh").read_text()
    for k in ("OTPU_WG_STOP", "OTPU_WG_MXU_COLS0", "OTPU_WG_MXU_COLS1", "OTPU_WG_MMCM"):
        assert f'export {k}="${{{k[5:]}:-${{{k}:-}}}}"' in rv, k


def _acc_clocks(tmp_path, no_clr=False) -> Path:
    """The top's accelerator clocks (its enable's register and the BUFGCE_DIV pair) copied out as a
    module of their own, otpu_wg_acc_clocks: the MMCM's 2x output and acc_ce in, the two clocks out
    (no_clr: the pair's CLR tied low, the test's control)."""
    src = TOP.read_text()
    m = re.search(r"^  logic ce2x_clk, .*?\bu_acc_buf \(.*?\);\n", src, re.S | re.M)
    assert m, "the top's clock block (ce2x_clk .. u_acc_buf)"
    blk = m.group(0)
    if no_clr:
        assert blk.count(".CLR(clr2x)") == 2
        blk = blk.replace(".CLR(clr2x)", ".CLR(1'b0)")
    f = tmp_path / "otpu_wg_acc_clocks.sv"
    f.write_text("module otpu_wg_acc_clocks (input logic acc_clk2x_u, input logic acc_ce,\n"
                 "                           output logic acc_clk, output logic acc_clk2x);\n"
                 f"{blk}endmodule\n")
    return f


def test_wg_acc_clocks_are_one_bufgce_div_pair():
    """acc_clk2x and acc_clk come from one MMCM output through two BUFGCE_DIVs, /1 and /2, on the
    same enable and clear (UG949's pair; different enables could leave the /2 a half cycle off
    without a timing report saying so, TIMING-49), the enable registered on that output's own
    BUFG; the block design has no acc_clk and no clock-enable buffer."""
    src = TOP.read_text()
    pair = re.findall(r"BUFGCE_DIV #\(\.BUFGCE_DIVIDE\((\d)\), \.SIM_DEVICE\(\"ULTRASCALE_PLUS\"\)\)\s+"
                      r"(\w+) \(\.I\((\w+)\), \.CE\((\w+)\), \.CLR\((\w+)\), \.O\((\w+)\)\);", src)
    assert sorted(pair) == [("1", "u_acc2x_buf", "acc_clk2x_u", "ce2x", "clr2x", "acc_clk2x"),
                            ("2", "u_acc_buf", "acc_clk2x_u", "ce2x", "clr2x", "acc_clk")]
    assert "BUFG u_ce2x_buf (.I(acc_clk2x_u), .O(ce2x_clk));" in src
    assert "BUFGCE " not in src and ".acc_clk_ce" not in src and ".acc_clk," not in src
    bd = (BCU / "vivado" / "bd_wg.tcl").read_text()
    assert "foreach p {core_clk acc_clk2x aclk xdma_aclk} { create_bd_port" in bd
    assert "acc_clk_ce" not in bd and "Buffer_with_CE" not in bd
    # the 2x on CLKOUT0 (the fractional divide: clk133.tcl), unbuffered, core_clk on CLKOUT1
    assert "CONFIG.CLK_OUT1_PORT {acc_clk2x}" in bd and "CONFIG.CLKOUT1_DRIVES {No_buffer}" in bd
    assert "CONFIG.CLK_OUT2_PORT {core_clk}" in bd
    stub = (BCU / "sim" / "otpu_bdw_wrapper_stub.sv").read_text()
    assert "acc_clk_ce" not in stub and " acc_clk," not in stub
    # CLOCK_DELAY_GROUP on the pair, off unless OTPU_WG_CDG (run_vivado.sh WG_CDG) is 1
    top = (BCU / "constraints" / "bcu1525_wg_top.tcl").read_text()
    assert '$::env(OTPU_WG_CDG) eq "1"' in top and "{u_acc_buf/O u_acc2x_buf/O}" in top
    rv = (BCU / "run_vivado.sh").read_text()
    assert 'export OTPU_WG_CDG="${WG_CDG:-${OTPU_WG_CDG:-}}"' in rv


def test_wg_acc_clocks_testbench_lints(have_verilator, tmp_path):
    """sim/tb_acc_clocks.sv over the top's clock block and unisim_stubs.sv's BUFGCE_DIV: no error."""
    r = subprocess.run(["verilator", "--lint-only", "--timing", "--timescale", "1ns/1ps", "-Wno-fatal",
                        "-Wno-lint", "-Wno-style", "--top-module", "tb_acc_clocks",
                        str(BCU / "sim" / "unisim_stubs.sv"),
                        str(_acc_clocks(tmp_path)), str(BCU / "sim" / "tb_acc_clocks.sv")],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "%Error" not in r.stderr, r.stderr[-2000:]


def _acc_clocks_build(tmp: Path, no_clr=False) -> Path:
    import shutil
    if shutil.which("verilator") is None:
        pytest.skip("verilator not installed")
    obj = tmp / "obj"
    r = subprocess.run(["verilator", "--binary", "--timing", "--timescale", "1ns/1ps", "-Wno-fatal",
                        "-Wno-lint", "-Wno-style", "--top-module", "tb_acc_clocks", "-Mdir", str(obj),
                        str(BCU / "sim" / "unisim_stubs.sv"), str(_acc_clocks(tmp, no_clr)),
                        str(BCU / "sim" / "tb_acc_clocks.sv")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return obj / "Vtb_acc_clocks"


@pytest.fixture(scope="module")
def acc_clocks_sim(tmp_path_factory):
    """sim/tb_acc_clocks.sv over the top's clock block, built once (Verilator --binary)."""
    return _acc_clocks_build(tmp_path_factory.mktemp("acc_clocks"))


@pytest.mark.rtl
@pytest.mark.parametrize("seed,start2x,o1", [(1, 0, 0), (2, 0, 0), (3, 0, 0),
                                             (4, 1, 0), (5, 0, 1), (6, 1, 1), (7, 4, 1), (8, 5, 1)])
def test_wg_acc_clocks_stop_and_start(acc_clocks_sim, seed, start2x, o1):
    """Random starts and stops of acc_ce (sim/tb_acc_clocks.sv, 4,000 runs, single-cycle runs and
    gaps among them, each start out of a stop with the clear high): acc_clk rises only with
    core_clk, two acc_clk2x edges in each of its cycles, both stop low, each run one core cycle
    after acc_ce. Also with the 2x output starting late, at a core edge (start2x even) or between
    two (odd), and with the /2 starting at the other phase (o1: only the clear puts it right)."""
    args = [f"+seed={seed}", f"+start2x={start2x}"] + (["+bufgce_div_o1"] if o1 else [])
    run = subprocess.run([str(acc_clocks_sim), *args], capture_output=True, text=True, timeout=300)
    assert "PASS" in run.stdout and "FAIL" not in run.stdout, run.stdout[-2000:]


@pytest.mark.rtl
def test_wg_acc_clocks_need_the_clear(tmp_path):
    """The control: with the pair's CLR tied low, the /2 started at the other phase stays there
    (acc_clk falls with core_clk's rises), and the testbench says FAIL."""
    run = subprocess.run([str(_acc_clocks_build(tmp_path, no_clr=True)), "+seed=5", "+bufgce_div_o1"],
                         capture_output=True, text=True, timeout=300)
    assert "FAIL" in run.stdout and "PASS" not in run.stdout, run.stdout[-2000:]


# ---- the 133.33 MHz fallback of a 150 MHz synthesis (vivado/clk133.tcl, OTPU_CLK_HOOK)

def _bd_clocks(tmp_path, mhz: float, ncards: int = 1) -> dict:
    """bd_wg.tcl's clock lines in tclsh at CORE_MHZ mhz: the divide's check, the wizard's outputs and
    the MMCM's override, every CONFIG property they set."""
    import shutil
    if not shutil.which("tclsh"):
        pytest.skip("no tclsh")
    bd = (BCU / "vivado" / "bd_wg.tcl").read_text()
    m1 = re.search(r"if \{!\[info exists CORE_MHZ\]\}.*?\nset ACLK_HZ \d+\n", bd, re.S)
    m2 = re.search(r"set clk \[create_bd_cell [^\n]*clk_wiz_0\]\n(set_property -dict \[list \\\n.*?\] \$clk\n)",
                   bd, re.S)
    m3 = re.search(r"if \{\[info exists WG_NCARDS\] && \$WG_NCARDS == 2\} \{\n  set_property -dict .*?\n\}\n", bd, re.S)
    m4 = re.search(r"set_property -dict \[list CONFIG\.OVERRIDE_MMCM .*?\] \$clk\n"
                   r"if \{\[dict exists \$want init_clk\]\} [^\n]*\n", bd, re.S)
    assert m1 and m2 and m3 and m4, "bd_wg.tcl's clock lines"
    tcl = tmp_path / f"bd_clocks_{mhz}.tcl"
    tcl.write_text(
        "proc set_property {args} {\n"
        "  if {[lindex $args 0] eq \"-dict\"} { foreach {k v} [lindex $args 1] { set ::P($k) $v } }"
        " else { set ::P([lindex $args 0]) [lindex $args 1] }\n}\n"
        f"set CORE_MHZ {mhz}\nset WG_NCARDS {ncards}\nset clk clk\n"
        + m1.group(0) + m2.group(1) + m3.group(0)
        + "set want [dict create]\nif {$WG_NCARDS == 2} { dict set want init_clk 100000000 }\n"
        + m4.group(0) + "foreach k [lsort [array names P]] { puts \"$k=$P($k)\" }\n")
    r = subprocess.run(["tclsh", str(tcl)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return dict(line.split("=", 1) for line in r.stdout.splitlines())


@pytest.mark.parametrize("ncards", [1, 2])
def test_wg_mmcm_puts_the_2x_on_clkout0_and_150_stays_4_and_8(tmp_path, ncards):
    """bd_wg.tcl's MMCM: the 2x on CLKOUT0 (DIVIDE_F, its one fractional divide) unbuffered, core_clk
    on CLKOUT1, aclk on CLKOUT2 at 4, init_clk on CLKOUT3 at 12 (two cards), the VCO 1200 MHz; at
    150 MHz 4.0 and 8 as before the swap, at 100 6.0 and 12; the override always (clk133.tcl takes
    exactly these). clk133.tcl's as-built check is that 150 MHz MMCM, and its edit 4.5 and 9 is
    133.33 MHz and its 2x."""
    p = _bd_clocks(tmp_path, 150, ncards)
    assert (p["CONFIG.CLK_OUT1_PORT"], p["CONFIG.CLKOUT1_DRIVES"]) == ("acc_clk2x", "No_buffer")
    assert p["CONFIG.CLK_OUT2_PORT"] == "core_clk" and "CONFIG.CLKOUT2_DRIVES" not in p
    assert p["CONFIG.CLK_OUT3_PORT"] == "aclk"
    assert float(p["CONFIG.CLKOUT1_REQUESTED_OUT_FREQ"]) == 300.0
    assert float(p["CONFIG.CLKOUT2_REQUESTED_OUT_FREQ"]) == 150.0
    assert p["CONFIG.OVERRIDE_MMCM"] == "true" and p["CONFIG.MMCM_DIVCLK_DIVIDE"] == "1"
    assert p["CONFIG.MMCM_CLKFBOUT_MULT_F"] == "4.000"                  # 300 MHz in, 1200 VCO
    assert (p["CONFIG.MMCM_CLKOUT0_DIVIDE_F"], p["CONFIG.MMCM_CLKOUT1_DIVIDE"],
            p["CONFIG.MMCM_CLKOUT2_DIVIDE"]) == ("4.000", "8", "4")
    assert p.get("CONFIG.MMCM_CLKOUT3_DIVIDE") == ("12" if ncards == 2 else None)
    assert p.get("CONFIG.CLK_OUT4_PORT") == ("init_clk" if ncards == 2 else None)
    p100 = _bd_clocks(tmp_path, 100, ncards)
    assert (p100["CONFIG.MMCM_CLKOUT0_DIVIDE_F"], p100["CONFIG.MMCM_CLKOUT1_DIVIDE"]) == ("6.000", "12")
    hook = (BCU / "vivado" / "clk133.tcl").read_text()
    built = re.search(r"foreach \{p v\} \{(CLKFBOUT_MULT_F [^}]*)\} \{\n    set g", hook).group(1).split()
    built = dict(zip(built[::2], built[1::2]))
    assert {k: float(v) for k, v in built.items()} == {
        "CLKFBOUT_MULT_F": 4.0, "DIVCLK_DIVIDE": 1.0, "CLKOUT0_DIVIDE_F": float(p["CONFIG.MMCM_CLKOUT0_DIVIDE_F"]),
        "CLKOUT1_DIVIDE": float(p["CONFIG.MMCM_CLKOUT1_DIVIDE"]), "CLKOUT2_DIVIDE": 4.0}
    assert "set_property CLKOUT0_DIVIDE_F 4.500 $mm" in hook and "set_property CLKOUT1_DIVIDE 9 $mm" in hook
    assert abs(1200 / 9 - 133.333) < 1e-3 and 1200 / 4.5 == 2 * 1200 / 9


def test_wg_clk133_values_are_the_rtls():
    """clk133.tcl's otpu_clk_k values are the RTL's constants at CORE_KHZ 150000 and CORE_KHZ1 133333
    (create_project_wg.tcl's generic): the UART's bit at 115200 baud (DIV, DIV - 1, DIV - 2,
    DIV / 2 - 1), the millisecond's last count, CORE_KHZ."""
    hook = (BCU / "vivado" / "clk133.tcl").read_text()
    want = {m[0]: (int(m[1]), int(m[2]), int(m[3]))
            for m in re.findall(r"^  (u_\w+) +\{(\d+) (\d+) (\d+)\}$", hook, re.M)}
    div = [(hz + 115_200 // 2) // 115_200 for hz in (150_000_000, 133_333_000)]
    assert want == {"u_div": (*div, 1), "u_div_m1": (div[0] - 1, div[1] - 1, 1),
                    "u_div_m2": (div[0] - 2, div[1] - 2, 1),
                    "u_div_h1": (div[0] // 2 - 1, div[1] // 2 - 1, 1),
                    "u_ms_m1": (149_999, 133_332, 2), "u_khz": (150_000, 133_333, 0)}
    assert div == [1302, 1157]
    cp = (BCU / "vivado" / "create_project_wg.tcl").read_text()
    assert "set CORE_KHZ1 [expr {$CORE_KHZ == 150000 ? 133333 : $CORE_KHZ}]" in cp
    assert 'set_property generic "CORE_KHZ=$CORE_KHZ CORE_KHZ1=$CORE_KHZ1 ' in cp
    top = TOP.read_text()
    assert ".CLK_HZ1(CORE_KHZ1 * 1000)" in top and top.count(".CLK_HZ1(CORE_KHZ1 * 1000)") == 2
    assert ".CORE_KHZ1(CORE_KHZ1)" in top


def test_wg_clk133_hook_is_wired():
    """OTPU_CLK_HOOK (run_vivado.sh CLK_HOOK): bcu1525_wg_check.tcl (opt_design's pre-hook) sources
    it after its checks and before WG_STOP=link's stop, place_screen.tcl after its open_checkpoint;
    build.tcl reports the fallback's constants after synthesis (clk_k.rpt) without stopping."""
    chk = (BCU / "vivado" / "bcu1525_wg_check.tcl").read_text()
    i = chk.index("OTPU_CLK_HOOK) ne")
    assert chk.index("every constraint file finished") < i < chk.index('$::env(OTPU_WG_STOP) eq "link"')
    sc = (BCU / "vivado" / "place_screen.tcl").read_text()
    assert sc.index("open_checkpoint $dcp") < sc.index("source $hook") < sc.index("place_design -directive")
    b = (BCU / "vivado" / "build.tcl").read_text()
    i = b.index("source $here/clk133.tcl")
    assert b.index("open_run synth_1") < i < b.index("close_design")
    assert "set ::otpu::clk_report_only 1" in b and "catch {source $here/clk133.tcl}" in b
    rv = (BCU / "run_vivado.sh").read_text()
    assert 'export OTPU_CLK_HOOK="${CLK_HOOK:-${OTPU_CLK_HOOK:-}}"' in rv
    hook = (BCU / "vivado" / "clk133.tcl").read_text()
    assert "report_clocks -file $rdir/clk133_clocks.rpt" in hook and "set_property INIT 1'b1 $c" in hook


@pytest.mark.parametrize("case,ok,msg", [
    ("ok", True, "4 selects INIT 1"),
    ("report", True, "both sets of constants in the netlist (7 otpu_clk_k"),
    ("fold", False, "u_bmc/g_clk1.u_div_m1: no bit on the select"),
    ("wrong", False, "u_bmc/g_clk1.u_div: bit 0 '0', want 0 at 150 MHz and 1 at 133.33"),
    ("nouart", False, "0 otpu_clk_k u_div, want 1"),
    ("nofrac", False, "CLKOUT0_DIVIDE_F 8.000, not 4: not bd_wg.tcl's 150 MHz MMCM"),
    ("badload", False, "CLKOUT5 has loads"),
    ("noreadback", False, "CLKOUT1_DIVIDE reads 8, not 9"),
])
def test_wg_clk133_hook_on_a_mock_netlist(tmp_path, case, ok, msg):
    """clk133.tcl on tests/data/vivado_mock_clk133.tcl's netlist: the report decodes each otpu_clk_k
    (the UART's DIV 1302 / 1157 among them) and the edit sets the MMCM to 4.5 / 9 and every select's
    INIT to 1 (DONT_TOUCH again after); a folded mux, a value off, no UART, an MMCM not
    bd_wg.tcl's, a loaded CLKOUT5 or a property that does not read back stop it."""
    import shutil
    if not shutil.which("tclsh"):
        pytest.skip("no tclsh")
    rd = tmp_path / "reports"
    r = subprocess.run(["tclsh", str(ROOT / "tests" / "data" / "vivado_mock_clk133.tcl"), case,
                        str(BCU / "vivado" / "clk133.tcl"), str(rd)], capture_output=True, text=True)
    out = r.stdout + r.stderr
    assert (r.returncode == 0) == ok and msg in out, out[-3000:]
    inits = re.findall(r"^\S+/u_sel_ff INIT (\S+) DONT_TOUCH (\d)$", r.stdout, re.M)
    if case == "ok":
        assert inits == [("1'b1", "1")] * 4 and "MMCM CLKOUT0 4.500 CLKOUT1 9" in r.stdout
    elif case == "report":
        assert inits == [("1'b0", "1")] * 4 and "MMCM CLKOUT0 4.000 CLKOUT1 8" in r.stdout
    if ok:
        rpt = (rd / "clk_k.rpt").read_text()
        assert re.search(r"^u_bmc/g_clk1\.u_div +1302 at 150 MHz, +1157 at 133\.33 MHz", rpt, re.M), rpt


def test_wg_umem_is_wired():
    """umem is one move (wg_umem.tcl's ::otpu::wg_umem) in both places: place_screen.tcl's
    OTPU_SCREEN_MOVES umem, and bcu1525_wg_place_pre.tcl's OTPU_WG_UMEM 1 (run_vivado.sh WG_UMEM:
    B2's place, before place_design as the screen's); a screen refuses both at once."""
    sc = (BCU / "vivado" / "place_screen.tcl").read_text()
    i = sc.index('if {"umem" in $moves} {')
    assert i < sc.index("[senv OTPU_WG_UMEM 0]") < sc.index("wg_umem.tcl]") < sc.index("::otpu::wg_umem $ctl") \
        < sc.index("source $pre") < sc.index("place_design -directive")
    pre = (BCU / "vivado" / "bcu1525_wg_place_pre.tcl").read_text()
    i = pre.index('$::env(OTPU_WG_UMEM) eq "1"')
    assert i < pre.index("get_pblocks -quiet pblock_ctl") < pre.index("wg_umem.tcl]") \
        < pre.index("::otpu::wg_umem $wg_ctl")
    assert 'export OTPU_WG_UMEM="${WG_UMEM:-${OTPU_WG_UMEM:-}}"' in (BCU / "run_vivado.sh").read_text()
    assert "proc ::otpu::wg_umem {ctl who}" in (BCU / "vivado" / "wg_umem.tcl").read_text()
    # the move's variables are the proc's: 333f2bb1's inline block set pre (each slice's prefix)
    # over the place_pre.tcl path, and source $pre failed (b1fp2aum)
    assert sc.count("set pre ") == 1 and pre.count("set pre ") == 0


def test_wg_x2port_is_wired():
    """x2port is one move (wg_x2port.tcl's ::otpu::wg_x2port) in both places: place_screen.tcl's
    OTPU_SCREEN_MOVES x2port, and bcu1525_wg_place_pre.tcl's OTPU_WG_X2PORT 1 (run_vivado.sh
    WG_X2PORT: B2's place, before place_design as the screen's); a screen refuses both at once."""
    sc = (BCU / "vivado" / "place_screen.tcl").read_text()
    i = sc.index('if {"x2port" in $moves} {')
    assert i < sc.index("[senv OTPU_WG_X2PORT 0]") < sc.index("wg_x2port.tcl]") \
        < sc.index("::otpu::wg_x2port $ctl place_screen.tcl") < sc.index("source $pre") \
        < sc.index("place_design -directive")
    assert "pblock_x2_link" not in sc[i:sc.index('if {"cport" in $moves} {')]
    pre = (BCU / "vivado" / "bcu1525_wg_place_pre.tcl").read_text()
    i = pre.index('$::env(OTPU_WG_X2PORT) eq "1"')
    assert i < pre.index("get_pblocks -quiet pblock_ctl", i) < pre.index("wg_x2port.tcl]") \
        < pre.index("::otpu::wg_x2port $wg_ctl")
    assert 'export OTPU_WG_X2PORT="${WG_X2PORT:-${OTPU_WG_X2PORT:-}}"' in (BCU / "run_vivado.sh").read_text()
    assert "proc ::otpu::wg_x2port {ctl who}" in (BCU / "vivado" / "wg_x2port.tcl").read_text()


@pytest.mark.parametrize("case,ok,msg", [
    ("ok", True, "mock: x2port: u_link/g_l[0].u_port u_link/g_l[1].u_port, 6 adapter leaves into pblock_ctl; "
                 "8 u_link cells stay in pblock_x2_link"),
    ("nopb", False, "mock: x2port: no pblock_x2_link"),
    ("oneport", False, "mock: x2port: 1 link ports (expected 2)"),
    ("nolx", False, "mock: x2port: no adapter leaves in pblock_x2_link"),
])
def test_wg_x2port_on_a_mock_netlist(case, ok, msg):
    """wg_x2port.tcl on tests/data/vivado_mock_x2port.tcl's x2 floorplan: the two link ports and the
    hub adapters' SLR2 leaves into pblock_ctl; u_link's other children (the lanes, the FIFOs) back in
    pblock_x2_link one by one, u_link itself in none; the adapters' u_dn_tx / u_up_rx stay in
    pblock_ctl. No pblock_x2_link, one port or no adapter leaves stop it."""
    if not shutil.which("tclsh"):
        pytest.skip("no tclsh")
    r = subprocess.run(["tclsh", str(ROOT / "tests" / "data" / "vivado_mock_x2port.tcl"), case,
                        str(BCU / "vivado" / "wg_x2port.tcl")], capture_output=True, text=True)
    out = r.stdout + r.stderr
    assert (r.returncode == 0) == ok and msg in out, out[-3000:]
    if not ok:
        return
    assert "returned 2 6 8" in r.stdout
    pb = dict(re.findall(r"^in (\S+) (\S+)$", r.stdout, re.M))
    assert pb["u_link"] == "-"
    for p in (0, 1):
        assert pb[f"u_link/g_l[{p}].u_port"] == "pblock_ctl"
        for c in ("u_lane", "u_txf", "u_rxf", "u_popped"):
            assert pb[f"u_link/g_l[{p}].{c}"] == "pblock_x2_link"
        a = f"u_sys/u_board/g_xs.g_link.g_l[{p}].u_lx"
        for c in ("s_reg[0]", "s_reg[1]", "v_reg", "u_dn_tx", "u_up_rx"):
            assert pb[f"{a}/{c}"] == "pblock_ctl", c


def test_wg_cport_is_wired():
    """cport is one move (wg_cport.tcl's ::otpu::wg_cport) in both places: place_screen.tcl's
    OTPU_SCREEN_MOVES cport, and bcu1525_wg_place_pre.tcl's OTPU_WG_CPORT 1 (run_vivado.sh WG_CPORT:
    B2's place, before place_design as the screen's); a screen refuses both at once."""
    sc = (BCU / "vivado" / "place_screen.tcl").read_text()
    i = sc.index('if {"cport" in $moves} {')
    assert i < sc.index("[senv OTPU_WG_CPORT 0]") < sc.index("wg_cport.tcl]") \
        < sc.index("::otpu::wg_cport $ctl place_screen.tcl") < sc.index("source $pre") \
        < sc.index("place_design -directive")
    assert "remove_cells_from_pblock" not in sc[i:sc.index('if {"umem" in $moves} {')]
    pre = (BCU / "vivado" / "bcu1525_wg_place_pre.tcl").read_text()
    i = pre.index('$::env(OTPU_WG_CPORT) eq "1"')
    assert i < pre.index("get_pblocks -quiet pblock_ctl", i) < pre.index("wg_cport.tcl]") \
        < pre.index("::otpu::wg_cport $wg_ctl")
    assert 'export OTPU_WG_CPORT="${WG_CPORT:-${OTPU_WG_CPORT:-}}"' in (BCU / "run_vivado.sh").read_text()
    assert "proc ::otpu::wg_cport {ctl who}" in (BCU / "vivado" / "wg_cport.tcl").read_text()


@pytest.mark.parametrize("case,ok,msg", [
    ("ok", True, "mock: cport: slice 1: u_sys/u_board/g_xs.g_x[1].u_cport u_sys/u_board/g_xs.g_x[1].u_dn_rx "
                 "u_sys/u_board/g_xs.g_x[1].u_up_tx into pblock_ctl (2 from pblock_slice1)"),
    ("nopb", False, "mock: cport: no pblock_slice1"),
    ("nocport", False, "mock: cport: 0 slice 1 collective port (u_sys/u_board/g_xs.g_x?1?.u_cport, expected 1)"),
])
def test_wg_cport_on_a_mock_netlist(case, ok, msg):
    """wg_cport.tcl on tests/data/vivado_mock_cport.tcl's two-slice floorplan: each slice's u_cport,
    u_dn_rx and u_up_tx into pblock_ctl, taken out of pblock_slice<s> only where the floorplan put
    them (u_up_tx is in none), the slice's other cells left; no pblock_slice1 or no u_cport stop it."""
    if not shutil.which("tclsh"):
        pytest.skip("no tclsh")
    r = subprocess.run(["tclsh", str(ROOT / "tests" / "data" / "vivado_mock_cport.tcl"), case,
                        str(BCU / "vivado" / "wg_cport.tcl")], capture_output=True, text=True)
    out = r.stdout + r.stderr
    assert (r.returncode == 0) == ok and msg in out, out[-3000:]
    if not ok:
        return
    assert "returned 6 4" in r.stdout
    pb = dict(re.findall(r"^in (\S+) (\S+)$", r.stdout, re.M))
    for s in (0, 1):
        x = f"u_sys/u_board/g_xs.g_x[{s}]"
        for c in ("u_cport", "u_dn_rx", "u_up_tx"):
            assert pb[f"{x}.{c}"] == "pblock_ctl", c
        assert pb[f"{x}.u_other"] == f"pblock_slice{s}"


def test_wg_preroute_hook(tmp_path):
    """impl_resume.tcl's OTPU_WG_PREPHYS and OTPU_WG_PREROUTE: each one's file (this directory's
    or a path) sourced at the top level, a missing one an error; PREPHYS on the placed design
    before phys_opt, PREROUTE after it and before route_design. route_holdfix.tcl turns the
    router's hold bailout off (Route 35-514's resolution)."""
    ir = (BCU / "vivado" / "impl_resume.tcl").read_text()
    h = ir.index("proc hook {k} {")
    assert h < ir.index("set f [file join $::here $::env($k)]") \
        < ir.index('error "impl_resume.tcl: $k: no $f"') < ir.index("uplevel #0 [list source $f]")
    assert h < ir.index("hook OTPU_WG_PREPHYS") < ir.index("phys_opt_design -directive [dir OTPU_WG_PHYS]") \
        < ir.index("hook OTPU_WG_PREROUTE") < ir.index("route_design -directive [dir OTPU_WG_ROUTE]")
    if not shutil.which("tclsh"):
        pytest.skip("no tclsh")
    m = tmp_path / "mock.tcl"
    m.write_text("array set P {}\nproc set_param {k v} { set ::P($k) $v }\nproc get_param {k} { return $::P($k) }\n"
                 f"source {BCU / 'vivado' / 'route_holdfix.tcl'}\n")
    r = subprocess.run(["tclsh", str(m)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "route_holdfix.tcl: route.enableHoldExpnBailout 0"


def test_wg_placed_stop_checks(tmp_path):
    """impl_resume.tcl's OTPU_WG_STOP placed: the staging checks' rows 6, 7, 10 and 11 on the placed
    design before it returns, each report in a catch; on a mock Vivado one failing report is logged
    and the rest are still written."""
    ir = (BCU / "vivado" / "impl_resume.tcl").read_text()
    s = ir.index('$::env(OTPU_WG_STOP) eq "placed"')
    e = ir.index('puts "impl_resume.tcl: OTPU_WG_STOP placed: stopped after place_design"')
    assert s < e < ir.index("return", e)
    blk = ir[ir.index("foreach {rpt cmd} {", s):e]
    rpts = ["placed_levels.rpt", "placed_clocks.rpt", "placed_clock_interaction.rpt", "placed_cdc.rpt",
            "placed_methodology.rpt", "placed_exceptions_ignored.rpt"]
    for r in rpts:
        assert r in blk, r
    if not shutil.which("tclsh"):
        pytest.skip("no tclsh")
    m = tmp_path / "mock.tcl"
    m.write_text("set out " + str(tmp_path) + "\nfile mkdir $out/reports\n"
                 "foreach c {report_design_analysis report_clocks report_clock_interaction report_cdc"
                 " report_exceptions} {\n"
                 "  proc $c {args} { close [open [lindex $args end] w] }\n}\n"
                 "proc report_methodology {args} { error \"mock: no methodology\" }\n" + blk + "\n")
    r = subprocess.run(["tclsh", str(m)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "impl_resume.tcl: placed_methodology.rpt: mock: no methodology"
    for rp in rpts:
        assert (tmp_path / "reports" / rp).exists() == (rp != "placed_methodology.rpt"), rp


def test_wg_pulse_skew(tmp_path):
    """post_impl.tcl's pulse-width skew report: reports/pulse_skew.rpt (report_pulse_width -min_skew
    -max_skew, -all if Vivado takes neither), its failing Min / Max Skew checks in SUMMARY.txt;
    pulse_skew.tcl counts each check once (B1's lines: the same check under two clocks' sections),
    passing ones and other kinds not, and names the worst."""
    pi = (BCU / "vivado" / "post_impl.tcl").read_text()
    assert pi.index("report_pulse_width -min_skew -max_skew") < pi.index("report_pulse_width -all") \
        < pi.index("source $here/pulse_skew.tcl") < pi.index('"pulse-width skew checks failing: $skew_n')
    if not shutil.which("tclsh"):
        pytest.skip("no tclsh")
    x = "u_bd/otpu_bdw_i/ddr4_{}/inst/u_ddr4_mem_intfc/u_mig_ddr4_phy/inst/generate_block1.u_ddr_xiphy/" \
        "byte_num[{}].xiphy_byte_wrapper.u_xiphy_byte_wrapper/I_BITSLICE_UPPER[4].GEN_RXTX_BITSLICE_EN." \
        "u_xiphy_bitslice_upper/xiphy_rxtx_bitslice/D[{}]"
    fail = ["Min Skew          Slow    RXTX_BITSLICE/D[2]  RXTX_BITSLICE/D[1]  0.090  -0.087  -0.177  "
            "BITSLICE_RX_TX_X0Y101  " + x.format(0, 3, 2),
            "Min Skew          Slow    RXTX_BITSLICE/D[1]  RXTX_BITSLICE/D[2]  0.090  0.026  -0.064  "
            "BITSLICE_RX_TX_X1Y582  " + x.format(3, 0, 1),
            "Min Skew          Fast    RXTX_BITSLICE/D[2]  RXTX_BITSLICE/D[1]  0.050  0.032  -0.018  "
            "BITSLICE_RX_TX_X0Y101  " + x.format(0, 3, 2)]
    ok = ["Max Skew          Fast    BITSLICE_CONTROL/RIU_CLK  BITSLICE_CONTROL/PLL_CLK  1.984  0.938  1.046  "
          "BITSLICE_CONTROL_X0Y73  u_bd/x/RIU_CLK",
          "Low Pulse Width   Slow    FDRE/C  n/a  0.275  1.666  -1.391  SLICE_X65Y194  u_bd/y/C"]
    rpt = tmp_path / "pulse_skew.rpt"
    rpt.write_text("\n".join(["Pulse Width Checks", *fail, *ok, "", "Pulse Width Checks", fail[0]]) + "\n")
    (tmp_path / "none.rpt").write_text("\n".join(["Pulse Width Checks", *ok]) + "\n")
    m = tmp_path / "m.tcl"
    m.write_text(f"source {BCU / 'vivado' / 'pulse_skew.tcl'}\n"
                 f"puts [::otpu::pulse_skew {rpt}]\nputs [::otpu::pulse_skew {tmp_path / 'none.rpt'}]\n")
    r = subprocess.run(["tclsh", str(m)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    got, none = r.stdout.strip().splitlines()
    assert got.startswith("3 -0.177 {Min Skew Slow RXTX_BITSLICE/D[2] / RXTX_BITSLICE/D[1] -0.177 ns at "
                          "BITSLICE_RX_TX_X0Y101 u_bd/otpu_bdw_i/ddr4_0/"), got
    assert none == "0 {} {}", none


@pytest.mark.parametrize("fo,picked", [("", ["m03", "wrs_p"]), ("600", ["m03"]),
                                       ("50", ["m03", "s00", "wrs_b", "wrs_p"]), ("700", [])])
def test_wg_screp(tmp_path, fo, picked):
    """phys_screp.tcl (OTPU_WG_PREPHYS) on a mock netlist: the SmartConnects' reset registers'
    and u_sys's DIMM reset copies' (otpu_bcu_wg_sys wrs_b / wrs_s / wrs_p) nets of fan-out
    OTPU_SCREP_FO (128) or more, force-replicated in one phys_opt_design; none, no
    phys_opt_design."""
    if not shutil.which("tclsh"):
        pytest.skip("no tclsh")
    sc = "u_bd/otpu_bdw_i/sc_dma/inst"
    regs = {f"{sc}/m03_nodes/m03_w_node/inst/inst_mi_handler/m_sc_areset_r_reg": 609,
            f"{sc}/s00_nodes/s00_ar_node/inst/inst_si_handler/s_sc_areset_r_reg": 65}
    wrs = {"u_sys/g_ch[3].wrs_b_reg": 97, "u_sys/g_ch[3].wrs_s_reg": 41,
           "u_sys/g_dimm[0].wrs_p_reg": 412}
    m = tmp_path / "mock.tcl"
    m.write_text(
        "array set FO {" + " ".join(f"{r}/Q_net {n}" for r, n in {**regs, **wrs}.items()) + "}\n"
        "proc get_cells {args} {\n"
        "  switch -- [lindex $args end] {\n"
        "    {IS_PRIMITIVE && REF_NAME =~ FD* && NAME =~ u_bd/*/sc_*/inst/*_sc_areset_r_reg} {\n"
        "      return {" + " ".join(regs) + "} }\n"
        "    {IS_PRIMITIVE && REF_NAME =~ FD* && NAME =~ u_sys/g_*.wrs_?_reg} {\n"
        "      return {" + " ".join("{%s}" % r for r in wrs) + "} }\n"
        "  }\n"
        "  error \"filter $args\"\n}\n"
        "proc get_pins {q p} { return $p }\n"
        "proc get_nets {q o p} { return ${p}_net }\n"
        "proc get_property {k n} { return $::FO($n) }\n"
        "proc phys_opt_design {f nets} { puts \"PHYS $f [llength $nets]\" }\n"
        f"source {BCU / 'vivado' / 'phys_screp.tcl'}\n")
    env = {**os.environ, "OTPU_SCREP_FO": fo}
    r = subprocess.run(["tclsh", str(m)], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (f"2 SmartConnect reset registers, 3 DIMM reset copies, {len(picked)} of fan-out >= "
            f"{fo or 128}") in r.stdout, r.stdout
    for k in ("m03", "s00", "wrs_b", "wrs_s", "wrs_p"):
        assert (f"{k}_" in r.stdout.split("copies,")[1]) == (k in picked), r.stdout
    assert ("PHYS -force_replication_on_nets %d" % len(picked) in r.stdout) == bool(picked), r.stdout
    assert ("fan-out 608 " in r.stdout) == ("m03" in picked), r.stdout
    assert ("fan-out 411 u_sys/g_dimm[0].wrs_p_reg/Q_net" in r.stdout) == ("wrs_p" in picked), r.stdout


@pytest.mark.parametrize("case,ok,msg", [
    ("ok", True, "u_xs: 4 pipes across SLR2 / SLR1, 10 registers on the Laguna sites"),
    ("onestage", False, "slice 0's u_xs pipe g_stg.u_ctl of 1 stage (NAT_XSTG >= 2)"),
    ("nodir", False, "slice 0's u_xs pipe g_stg.u_zz of no known direction"),
    ("slr1", False, "pblock_slice0's ranges 'SLR1' (SLR0 or SLR2)"),
    ("nomem", False, "no slice 1 adapter"),
])
def test_wg_umem_on_a_mock_netlist(case, ok, msg):
    """wg_umem.tcl on tests/data/vivado_mock_umem.tcl's two slices: u_mem and the 8 core halves into
    pblock_ctl; each crossing pipe's pair on the Laguna registers (u_ctl and u_f slice -> adapter:
    the last two stages; u_r and u_ra adapter -> slice: the first two), the stages before it with
    the source, after it with the destination; g_pc and the queue's own cells with the adapter;
    u_pp, g_ps and the idle holds with the slice. A 1-stage crossing, a pipe of no known direction,
    a slice pblock in SLR1 or a missing adapter stop it."""
    if not shutil.which("tclsh"):
        pytest.skip("no tclsh")
    r = subprocess.run(["tclsh", str(ROOT / "tests" / "data" / "vivado_mock_umem.tcl"), case,
                        str(BCU / "vivado" / "wg_umem.tcl")], capture_output=True, text=True)
    out = r.stdout + r.stderr
    assert (r.returncode == 0) == ok and msg in out, out[-3000:]
    if not ok:
        return
    leaf = {m[0]: (m[1], m[2]) for m in re.findall(r"^leaf (\S+) (\S+) (\S+)$", r.stdout, re.M)}
    for s, pre, pb in ((0, "u_sys/u_board/", "pblock_slice0"), (1, "u_sys/u_board/g_slice[1].", "pblock_slice1")):
        x = pre + "u_xs/"
        st = lambda p, i, k="g_next.", b="": leaf[f"{x}{p}/g_pipe.g_s[{i}].{k}r_reg{b}"]
        assert st("g_stg.u_ctl", 0, "g_first.", "[0]") == (pb, "SLL")
        assert st("g_stg.u_ctl", 1, b="[0]") == ("pblock_ctl", "SLL")
        assert st("g_stg.u_qa/u_f", 0, "g_first.", "[1]") == (pb, "-")
        assert st("g_stg.u_qa/u_f", 1, b="[1]") == (pb, "SLL")
        assert st("g_stg.u_qa/u_f", 2, b="[0]") == ("pblock_ctl", "SLL")
        assert st("g_stg.u_qa/u_r", 0, "g_first.", "[0]") == ("pblock_ctl", "SLL")
        assert st("g_stg.u_qa/u_r", 1, b="[0]") == (pb, "SLL")
        assert st("g_stg.u_ra", 0, "") == ("pblock_ctl", "SLL") and st("g_stg.u_ra", 1, "") == (pb, "SLL")
        assert st("g_pc[0].u_p", 1, b="[0]") == ("pblock_ctl", "-")
        assert st("u_pp", 1, b="[0]") == (pb, "-") and st("g_ps.u_ps", 0, "g_first.", "[0]") == (pb, "-")
        assert leaf[x + "g_stg.u_qa/u_f/g_pipe.g_s[0].d_lut"] == (pb, "-")
        assert leaf[x + "g_stg.u_qa/u_f/o_lut"] == ("pblock_ctl", "-")
        assert leaf[x + "g_stg.u_qa/ram_reg[0]"] == ("pblock_ctl", "-")
        assert leaf[x + "g_stg.w_h_reg"] == (pb, "-") and leaf[x + "g_stg.r_h_reg[0]"] == (pb, "-")
        assert leaf[x + "misc_lut"] == ("pblock_ctl", "-")
    moved = re.findall(r"^moved (\S+) (\S+)$", r.stdout, re.M)
    assert len(moved) == 18 and {p for _, p in moved} == {"pblock_ctl"}
    assert "sll 20" in r.stdout


def test_wg_top_lints_with_the_fallback_clock(have_verilator, tmp_path):
    """The WG top at CORE_KHZ 150000 with CORE_KHZ1 133333 (create_project_wg.tcl's generic): the
    fallback's otpu_clk_sel / otpu_clk_k under the governor, the UART, its watch and otpu_ctrl lint
    as the top does (no error; in the WG files and the muxes' no warning but PROCASSINIT)."""
    subprocess.run([sys.executable, str(BCU / "scripts" / "make_wg.py"), str(tmp_path)],
                   check=True, capture_output=True)
    rtl = [str(ROOT / "rtl" / f) for f in T.sources("bcu1525-wg")]
    r = subprocess.run(
        ["verilator", "--lint-only", "-Wall", "-Wno-fatal", "-Wno-UNUSEDSIGNAL", "-Wno-UNUSEDPARAM",
         "-Wno-DECLFILENAME", "-Wno-PINCONNECTEMPTY", "-Wno-GENUNNAMED", "-Wno-VARHIDDEN",
         "-DSYNTHESIS", "-DOTPU_FPGA_INIT", "--top-module", "otpu_fpga_top_bcu_wg", "-GCORE_KHZ=150000",
         "-GCORE_KHZ1=133333",
         str(BCU / "sim" / "unisim_dsp48e2_stub.sv"), str(BCU / "sim" / "unisim_stubs.sv"),
         str(tmp_path / "otpu_bcu_wg_defines.vh"), *rtl,
         str(BCU / "sim" / "otpu_bdw_wrapper_stub.sv")], capture_output=True, text=True)
    msgs = [m for m in (r.stdout + r.stderr).splitlines() if m.startswith("%")]
    assert not [m for m in msgs if m.startswith("%Error")], msgs[:20]
    wg = ("otpu_fpga_top_bcu_wg.sv", "otpu_bcu_wg_sys.sv", "otpu_bmc_uart.sv", "otpu_bmc_watch.sv",
          "otpu_bcu_gov.sv")
    bad = [m for m in msgs if (any(f in m for f in wg) or "clk_k" in m or "clk_sel" in m or "u_sel_ff" in m)
           and "PROCASSINIT" not in m]
    assert not bad, bad[:20]


def _yosys_slang() -> str | None:
    y = os.environ.get("YOSYS") or shutil.which("yosys")
    if y and subprocess.run([y, "-m", "slang", "-q", "-p", "help read_slang"],
                            capture_output=True).returncode == 0:
        return y
    return None


def _clk_k_value(mod: dict, sel: int) -> tuple[int, int]:
    """An otpu_clk_k's (K0, K1) from its netlist: each bit of k a constant, the select (K1's 1) or a
    NOT of it (K0's 1). A bit of anything else is no value: an assertion."""
    nots = {c["connections"]["Y"][0]: c["connections"]["A"][0] for c in mod["cells"].values()
            if c["type"] in ("$_NOT_", "$not")}
    k0 = k1 = 0
    for i, b in enumerate(mod["ports"]["k"]["bits"]):
        if b in ("0", "1"):
            k0 |= int(b) << i
            k1 |= int(b) << i
        elif b == sel:
            k1 |= 1 << i
        elif nots.get(b) == sel:
            k0 |= 1 << i
        else:
            raise AssertionError(f"k[{i}] is neither a constant nor the select: {b}")
    return k0, k1


def test_wg_clk133_constants_survive_synthesis(tmp_path):
    """Both constant sets in a netlist (yosys and slang, the hierarchy kept as keep_hierarchy keeps
    it; FDRE a black box, as Vivado's primitive): the UART (and its watch), the governor and otpu_ctrl at
    150 MHz with the 133.33 fallback synthesize each otpu_clk_k to clk133.tcl's clk_k_want values
    on its select, and each otpu_clk_sel to its FDRE alone (D its own Q, INIT 0, DONT_TOUCH): a
    constant select would fold the 133.33 set away, as yosys folds an inferred D = Q register with
    an initial value (through keep and dont_touch). Vivado's own: build.tcl's post-synth clk_k.rpt
    (clk133.tcl, report only)."""
    y = _yosys_slang()
    if not y:
        pytest.skip("set YOSYS to a Yosys with the slang plugin")
    rtl = ROOT / "rtl"
    ctrl, b = rtl / "shell" / "otpu_ctrl.sv", rtl / "boards" / "bcu1525"
    stubs = BCU / "sim" / "unisim_stubs.sv"
    hz = "-G CLK_HZ=150000000 -G CLK_HZ1=133333000"
    runs = {"otpu_bmc_uart": (hz, [ctrl, b / "otpu_bmc_watch.sv", b / "otpu_bmc_uart.sv"]),
            "otpu_bcu_gov": (hz, [ctrl, b / "otpu_bcu_gov.sv"]),
            "otpu_ctrl": ("-G CORE_KHZ=150000 -G CORE_KHZ1=133333", [ctrl])}
    script = []
    for top, (g, files) in runs.items():
        script += [f"read_slang --single-unit --allow-use-before-declare --keep-hierarchy "
                   f"--blackboxed-module FDRE --top {top} {g} " + " ".join(map(str, [stubs, *files])),
                   f"synth -top {top}", f"write_json {tmp_path / top}.json", "design -reset"]
    (tmp_path / "clk.ys").write_text("\n".join(script) + "\n")
    r = subprocess.run([y, "-q", "-m", "slang", "-s", str(tmp_path / "clk.ys")], capture_output=True,
                       text=True, timeout=600)
    (tmp_path / "yosys.log").write_text(r.stdout + r.stderr)
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
    hook = (BCU / "vivado" / "clk133.tcl").read_text()
    want = {m[0]: (int(m[1]), int(m[2]))
            for m in re.findall(r"^  (u_\w+) +\{(\d+) (\d+) \d+\}$", hook, re.M)}
    got = {}
    for top in runs:
        mods = json.loads((tmp_path / f"{top}.json").read_text())["modules"]

        def walk(name, path):
            m = mods[name]
            for cn, c in m["cells"].items():
                if c["type"] not in mods:
                    continue
                kind = mods[c["type"]]["attributes"].get("hdlname", c["type"])
                if "otpu_clk_sel" in kind or "otpu_clk_sel" in c["type"]:
                    sm = mods[c["type"]]
                    ff = sm["cells"].get("u_sel_ff", {})
                    q = sm["ports"]["sel"]["bits"][0]
                    assert list(sm["cells"]) == ["u_sel_ff"] and ff["type"] == "FDRE", \
                        f"{path}{cn}: {sm['cells']}"
                    assert ff["connections"]["Q"] == ff["connections"]["D"] == [q], \
                        f"{path}{cn}: D is not its Q, or Q not the select"
                    assert int(ff["parameters"]["INIT"], 2) == 0, f"{path}{cn}: INIT"
                    assert ff["attributes"].get("DONT_TOUCH") == "TRUE", f"{path}{cn}: no DONT_TOUCH"
                    got.setdefault("sel", []).append(f"{path}{cn}")
                elif "otpu_clk_k" in kind or "otpu_clk_k" in c["type"]:
                    sel = c["connections"]["sel"][0]
                    sm = mods[c["type"]]
                    got[f"{top}:{path}{cn}"] = _clk_k_value(sm, sm["ports"]["sel"]["bits"][0])
                    assert sel not in ("0", "1"), f"{path}{cn}: a constant select"
                else:
                    walk(c["type"], f"{path}{cn}/")

        top_mod = next(n for n, m in mods.items() if m["attributes"].get("top"))
        walk(top_mod, "")
    sels = got.pop("sel")
    assert len(sels) == 4, sels      # the UART's, its watch's, the governor's, otpu_ctrl's
    by = {}
    for path, v in got.items():
        by.setdefault(re.search(r"(u_\w+)$", path)[1], set()).add(v)
    assert by == {k: {v} for k, v in want.items()}, got
    assert got["otpu_bmc_uart:g_clk1.u_div"] == (1302, 1157)


def test_wg_migko_is_wired():
    """the MIG keep-out (wg_migko.tcl's ::otpu::wg_migko) is bcu1525_wg_place_pre.tcl's, after umem, behind
    OTPU_WG_MIGKO (run_vivado.sh WG_MIGKO, off unless set and not 0; OTPU_WG_MIGKO_W, default 12), so a
    place_screen.tcl screen takes it through source $pre; the screen's summary gets migko_report's lines."""
    pre = (BCU / "vivado" / "bcu1525_wg_place_pre.tcl").read_text()
    i = pre.index('$::env(OTPU_WG_MIGKO) ni {"" 0}')
    assert pre.index("::otpu::wg_umem $wg_ctl") < i < pre.index("wg_migko.tcl]") \
        < pre.index("::otpu::wg_migko $::env(OTPU_WG_MIGKO) $wg_mkw bcu1525_wg_place_pre.tcl")
    assert "$::env(OTPU_WG_MIGKO_W) : 12}" in pre and "unset wg_mkw wg_mk" in pre
    rv = (BCU / "run_vivado.sh").read_text()
    assert 'export OTPU_WG_MIGKO="${WG_MIGKO:-${OTPU_WG_MIGKO:-}}"' in rv
    assert 'export OTPU_WG_MIGKO_W="${WG_MIGKO_W:-${OTPU_WG_MIGKO_W:-}}"' in rv
    sc = (BCU / "vivado" / "place_screen.tcl").read_text()
    assert sc.index("source $pre") < sc.index("place_design -directive") \
        < sc.index("foreach l [::otpu::migko_report] { puts $f \"migko: $l\" }")
    mk = (BCU / "vivado" / "wg_migko.tcl").read_text()
    assert "proc ::otpu::wg_migko {spec w who}" in mk and "proc ::otpu::migko_report {}" in mk


# each DIMM's strip: its first SLICE column and SLICE rows (B1's: C0 SLICE_X55Y60:SLICE_X66Y239 at W 12)
_MIGKO = {0: (55, 60, 239), 1: (112, 360, 539), 2: (55, 420, 599), 3: (112, 660, 839)}


@pytest.mark.parametrize("case,spec,w,ok,msg", [
    ("ok", "1", "12", True, "pblock_migko0 SLICE_X55Y60:SLICE_X66Y239 (2160 sites, IOB column X0 rows 1-3, SLR0)"),
    ("pkg", "1", "12", True, "pblock_migko3 SLICE_X112Y660:SLICE_X123Y839 (2160 sites, IOB column X1 rows 11-13, SLR2)"),
    ("ok", "0 3", "8", True, "pblock_migko3 SLICE_X112Y660:SLICE_X119Y839 (1440 sites"),
    ("twocol", "1", "12", False, "DIMM 2's ports on IOB columns '0 1' (one expected)"),
    ("twoslr", "1", "12", False, "DIMM 0's ports on clock-region rows '1 2 3 5' (two SLRs)"),
    ("crx", "1", "12", False, "DIMM 1 port ddr4_c1_dq[2] at IOB_X1Y364: clock region X4Y8, not row 7"),
    ("nocells", "1", "12", False, "ddr4_3/inst/u_ddr4_mem_intfc: 0 CLB cells, 0 LUTs, 0 flip-flops"),
    ("inpb", "1", "12", False, "ddr4_1/inst/u_ddr4_mem_intfc's cells already in 'pblock_slice0'"),
    ("exists", "1", "12", False, "pblock_migko0 exists"),
    ("ok", "5", "12", False, "OTPU_WG_MIGKO '5' (1 or DIMM numbers 0-3)"),
    ("ok", "1", "40", False, "OTPU_WG_MIGKO_W '40' (SLICE columns, 1-28)"),
])
def test_wg_migko_on_a_mock_netlist(case, spec, w, ok, msg):
    """wg_migko.tcl on tests/data/vivado_mock_migko.tcl's four DIMMs (the board's banks: C0 rows 1-3 and C2
    7-9 on IOB column X0, C1 6-8 and C3 11-13 on X1): each keep-out W SLICE columns from X55 / X112 over its
    rows, IS_SOFT and EXCLUDE_PLACEMENT, with its MIG's u_ddr4_mem_intfc CLB cells (the block RAM left
    out), the IOB site from the port's LOC or its package pin's; then migko_report on a mock placement.
    Ports over two I/O columns or two SLRs, a clock region against the row, a MIG without cells or
    with cells in a pblock, a keep-out made before, a bad SPEC or W stop it."""
    if not shutil.which("tclsh"):
        pytest.skip("no tclsh")
    r = subprocess.run(["tclsh", str(ROOT / "tests" / "data" / "vivado_mock_migko.tcl"), case,
                        str(BCU / "vivado" / "wg_migko.tcl"), spec, w], capture_output=True, text=True)
    out = r.stdout + r.stderr
    assert (r.returncode == 0) == ok and msg in out, out[-3000:]
    if not ok:
        return
    pbs = dict(re.findall(r"^pb pblock_migko(\d) (.+)$", r.stdout, re.M))
    ds = [0, 1, 2, 3] if spec == "1" else [int(d) for d in spec.split()]
    want = {d: f"SLICE_X{x}Y{ya}:SLICE_X{x + int(w) - 1}Y{yb}" for d, (x, ya, yb) in _MIGKO.items() if d in ds}
    assert pbs == {str(d): f"{rg} soft TRUE exclude TRUE cells 7" for d, rg in want.items()}, r.stdout
    assert "made " + " ".join(f"{d} pblock_migko{d} {want[d]}" for d in ds) in r.stdout
    rep = dict(re.findall(r"^report pblock_migko(\d) \(.+\): (.+)$", r.stdout, re.M))
    assert rep["0"] == "4 of 7 placed cells inside, 1 other cells inside", r.stdout
    assert rep["3"] == "0 of 7 placed cells inside, 0 other cells inside", r.stdout


def _tcl_brace_end(s, i):
    """the index after the brace group opening at s[i] (Tcl text without escaped braces)"""
    assert s[i] == "{", s[i:i + 40]
    n = 0
    for j in range(i, len(s)):
        n += {"{": 1, "}": -1}.get(s[j], 0)
        if n == 0:
            return j + 1
    raise AssertionError("unbalanced braces")


@pytest.mark.parametrize("spec,ds,ok", [("", [], True), ("0", [], True), ("1", [0, 3], True),
                                        ("0 3", [0, 3], True), ("2", [2], True), ("7", [], False)])
def test_wg_hrs(tmp_path, spec, ds, ok):
    """bd_wg.tcl's OTPU_WG_HRS in tclsh (its hrs_dimms and the host port's block, Vivado's BD commands
    mocked): off by default; 1 = C0 and C3; the listed DIMMs' sc_h<d>/M00_AXI through rs_h<d>, an AXI
    register slice with every channel in Multi_SLR_Crossing (15) over one crossing, on aclk and
    sc_rstn<d>, into M_AXI_H<d>; the others straight; a DIMM number past 3 stops it."""
    if not shutil.which("tclsh"):
        pytest.skip("no tclsh")
    bd = (BCU / "vivado" / "bd_wg.tcl").read_text()
    i = bd.index("proc hrs_dimms {spec} {")
    proc = bd[i:_tcl_brace_end(bd, bd.index("{", bd.index("{spec}", i) + 6))]
    assert "set HRS [hrs_dimms [expr {[info exists ::env(OTPU_WG_HRS)] ? $::env(OTPU_WG_HRS) : \"\"}]]" in bd
    i = bd.index("  if {$d in $HRS} {")
    j = _tcl_brace_end(bd, bd.index("{", bd.index("}", i) + 1))
    assert bd[j:j + 7] == " else {"
    blk = bd[i:_tcl_brace_end(bd, j + 6)]
    assert bd.index("connect_bd_intf_net [get_bd_intf_ports S_AXI_H$d] [get_bd_intf_pins sc_h$d/S00_AXI]") < i
    m = tmp_path / "hrs.tcl"
    m.write_text(
        "array set CFG {}\n"
        "proc ip_vlnv {n} { return xilinx.com:ip:$n:2.1 }\n"
        "proc create_bd_cell {t ty v vl n} { puts \"cell $n $vl\"; return $n }\n"
        "proc set_property {d kv c} { foreach {k v} $kv { set ::CFG($c,$k) $v } }\n"
        "proc get_property {k c} { return $::CFG($c,$k) }\n"
        "proc get_bd_pins {p} { return pin:$p }\n"
        "proc get_bd_ports {p} { return port:$p }\n"
        "proc get_bd_intf_pins {p} { return ipin:$p }\n"
        "proc get_bd_intf_ports {p} { return iport:$p }\n"
        "proc connect_bd_net {a b} { puts \"net $a $b\" }\n"
        "proc connect_bd_intf_net {a b} { puts \"intf $a $b\" }\n"
        f"{proc}\n"
        "if {[catch {set HRS [hrs_dimms $::env(OTPU_WG_HRS)]} e]} { puts stderr $e; exit 1 }\n"
        "foreach d {0 1 2 3} {\n" + blk + "\n}\n"
        "foreach k [lsort [array names CFG]] { puts \"cfg $k $CFG($k)\" }\n")
    r = subprocess.run(["tclsh", str(m)], capture_output=True, text=True, env={**os.environ, "OTPU_WG_HRS": spec})
    assert (r.returncode == 0) == ok, r.stdout + r.stderr
    if not ok:
        assert "bd_wg: OTPU_WG_HRS '7' (1 or DIMM numbers 0-3)" in r.stderr
        return
    for d in range(4):
        if d in ds:
            assert f"cell rs_h{d} xilinx.com:ip:axi_register_slice:2.1" in r.stdout
            for k, v in (("REG_AW", 15), ("REG_AR", 15), ("REG_W", 15), ("REG_R", 15), ("REG_B", 15),
                         ("NUM_SLR_CROSSINGS", 1), ("DATA_WIDTH", 512), ("ADDR_WIDTH", 33), ("ID_WIDTH", 0)):
                assert f"cfg rs_h{d},CONFIG.{k} {v}" in r.stdout, r.stdout
            assert f"net pin:clk_wiz_0/aclk pin:rs_h{d}/aclk" in r.stdout
            assert f"net port:sc_rstn{d} pin:rs_h{d}/aresetn" in r.stdout
            assert f"intf ipin:sc_h{d}/M00_AXI ipin:rs_h{d}/S_AXI" in r.stdout
            assert f"intf ipin:rs_h{d}/M_AXI iport:M_AXI_H{d}" in r.stdout
        else:
            assert f"rs_h{d}" not in r.stdout
            assert f"intf ipin:sc_h{d}/M00_AXI iport:M_AXI_H{d}" in r.stdout
    rv = (BCU / "run_vivado.sh").read_text()
    assert 'export OTPU_WG_HRS="${WG_HRS:-${OTPU_WG_HRS:-}}"' in rv


@pytest.mark.parametrize("spec,ws,on,ok", [("", 1, False, True), ("", 2, True, True),
                                           ("0", 2, False, True), ("1", 1, True, True),
                                           ("2", 2, False, False)])
def test_wg_prs(tmp_path, spec, ws, on, ok):
    """bd_wg.tcl's OTPU_WG_PRS in tclsh (its prs_on and the DIMM port's block, Vivado's BD commands
    mocked): unset, on at WG_S 2 and off at 1; 0 / 1 off / on; each DIMM's S_AXI_P<d> through
    rs_p<d>, an AXI register slice (AW, AR, R, B fully registered, W straight; 4 ID bits), on aclk
    and sc_rstn<d>, into cc_mem<d>; off, straight into cc_mem<d>; another value stops it."""
    if not shutil.which("tclsh"):
        pytest.skip("no tclsh")
    bd = (BCU / "vivado" / "bd_wg.tcl").read_text()
    i = bd.index("proc prs_on {spec s} {")
    proc = bd[i:_tcl_brace_end(bd, bd.index("{", bd.index("{spec s}", i) + 8))]
    assert "set PRS [prs_on [expr {[info exists ::env(OTPU_WG_PRS)] ? $::env(OTPU_WG_PRS) : \"\"}] $WG_S]" in bd
    i = bd.index("  if {$PRS} {")
    j = _tcl_brace_end(bd, bd.index("{", bd.index("}", i) + 1))
    assert bd[j:j + 7] == " else {"
    blk = bd[i:_tcl_brace_end(bd, j + 6)]
    assert bd.index("set cc [create_bd_cell -type ip -vlnv [ip_vlnv axi_clock_converter] cc_mem$d]") < i
    m = tmp_path / "prs.tcl"
    m.write_text(
        "array set CFG {}\n"
        "proc ip_vlnv {n} { return xilinx.com:ip:$n:2.1 }\n"
        "proc create_bd_cell {t ty v vl n} { puts \"cell $n $vl\"; return $n }\n"
        "proc set_property {d kv c} { foreach {k v} $kv { set ::CFG($c,$k) $v } }\n"
        "proc get_property {k c} { return $::CFG($c,$k) }\n"
        "proc get_bd_pins {p} { return pin:$p }\n"
        "proc get_bd_ports {p} { return port:$p }\n"
        "proc get_bd_intf_pins {p} { return ipin:$p }\n"
        "proc get_bd_intf_ports {p} { return iport:$p }\n"
        "proc connect_bd_net {a b} { puts \"net $a $b\" }\n"
        "proc connect_bd_intf_net {a b} { puts \"intf $a $b\" }\n"
        f"set WG_S {ws}\n{proc}\n"
        "if {[catch {set PRS [prs_on $::env(OTPU_WG_PRS) $WG_S]} e]} { puts stderr $e; exit 1 }\n"
        "foreach d {0 1 2 3} {\n" + blk + "\n}\n"
        "foreach k [lsort [array names CFG]] { puts \"cfg $k $CFG($k)\" }\n")
    r = subprocess.run(["tclsh", str(m)], capture_output=True, text=True, env={**os.environ, "OTPU_WG_PRS": spec})
    assert (r.returncode == 0) == ok, r.stdout + r.stderr
    if not ok:
        assert "bd_wg: OTPU_WG_PRS '2' (0 or 1; unset: on at WG_S 2)" in r.stderr
        return
    for d in range(4):
        if on:
            assert f"cell rs_p{d} xilinx.com:ip:axi_register_slice:2.1" in r.stdout
            for k, v in (("REG_AW", 1), ("REG_AR", 1), ("REG_W", 0), ("REG_R", 1), ("REG_B", 1),
                         ("DATA_WIDTH", 512), ("ADDR_WIDTH", 33), ("ID_WIDTH", 4)):
                assert f"cfg rs_p{d},CONFIG.{k} {v}" in r.stdout, r.stdout
            assert f"net pin:clk_wiz_0/aclk pin:rs_p{d}/aclk" in r.stdout
            assert f"net port:sc_rstn{d} pin:rs_p{d}/aresetn" in r.stdout
            assert f"intf iport:S_AXI_P{d} ipin:rs_p{d}/S_AXI" in r.stdout
            assert f"intf ipin:rs_p{d}/M_AXI ipin:cc_mem{d}/S_AXI" in r.stdout
        else:
            assert f"rs_p{d}" not in r.stdout
            assert f"intf iport:S_AXI_P{d} ipin:cc_mem{d}/S_AXI" in r.stdout
    rv = (BCU / "run_vivado.sh").read_text()
    assert 'export OTPU_WG_PRS="${WG_PRS:-${OTPU_WG_PRS:-}}"' in rv


@pytest.mark.parametrize("bits,got,ok", [(256, "0", True), (256, "1", False), (256, "", False),
                                         (512, "1", True)])
def test_wg_hrs_narrow(tmp_path, bits, got, ok):
    """bd_wg.tcl's check after validate_bd_design: each rs_h<d>'s S_AXI takes M_AXI_H<d>'s
    SUPPORTS_NARROW_BURST (0) back to sc_h<d>, else at 256 bits (XDMA narrower than the hmux) the
    build stops; at 512 there is no upsizer and the value is only printed."""
    if not shutil.which("tclsh"):
        pytest.skip("no tclsh")
    bd = (BCU / "vivado" / "bd_wg.tcl").read_text()
    i = bd.index("foreach d $HRS {")
    assert bd.index("validate_bd_design") < i
    blk = bd[i:_tcl_brace_end(bd, i + len("foreach d $HRS "))]
    m = tmp_path / "hrsn.tcl"
    m.write_text(
        f"set AXI_BITS {bits}\nset HRS {{0 3}}\n"
        "proc get_bd_intf_ports {p} { return port:$p }\n"
        "proc get_bd_intf_pins {p} { return pin:$p }\n"
        "proc get_property {k o} { if {[string match port:* $o]} { return 0 }; return $::env(GOT) }\n"
        f"if {{[catch {{{blk}}} e]}} {{ puts stderr $e; exit 1 }}\n")
    r = subprocess.run(["tclsh", str(m)], capture_output=True, text=True, env={**os.environ, "GOT": got})
    assert (r.returncode == 0) == ok, r.stdout + r.stderr
    assert f"bd_wg: rs_h0/S_AXI SUPPORTS_NARROW_BURST '{got}' (M_AXI_H0 '0')" in r.stdout
    if not ok:
        assert (f"rs_h0/S_AXI: SUPPORTS_NARROW_BURST '{got}', M_AXI_H0's '0': sc_h0 could send narrow beats "
                f"at {bits} bits") in r.stderr


@pytest.mark.parametrize("log,rc", [("bd_wg: four DIMMs (MIG DDR4-2400, AXI4 512), M_AXI_DMA0..3 512 bits\n", 0),
                                    ("bd_wg: rs_h0 (OTPU_WG_HRS): ...\n", 1)])
def test_wg_step_bd(tmp_path, log, rc):
    """run_vivado.sh STEP=bd (IMAGE=wg) stops after create_project_wg.tcl, before build.tcl, when bd_wg.tcl
    printed its last line (the block design validated and checked); else exit 1. Run in bash on its own
    lines, with create_project_wg.tcl's log faked."""
    rv = (BCU / "run_vivado.sh").read_text()
    i = rv.index('  run "$here/vivado/create_project_wg.tcl" "$out"\n')
    j = rv.index('  run "$here/vivado/build.tcl"', i)
    blk = rv[i:j].split("\n", 1)[1]
    assert blk.startswith('  if [[ "${STEP:-}" == bd ]]; then')
    bd = (BCU / "vivado" / "bd_wg.tcl").read_text()
    assert 'puts "bd_wg: four DIMMs (MIG DDR4-2400, AXI4 512)' in bd
    (tmp_path / "create_project_wg.log").write_text(log)
    sh = tmp_path / "t.sh"
    sh.write_text(f"set -euo pipefail\nout={tmp_path}\nSTEP=bd\n{blk}echo NOT STOPPED\n")
    r = subprocess.run(["bash", str(sh)], capture_output=True, text=True)
    assert r.returncode == rc, r.stdout + r.stderr
    assert "NOT STOPPED" not in r.stdout
    assert ("done: block design" in r.stdout) == (rc == 0)
