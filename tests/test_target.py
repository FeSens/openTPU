"""Targets (opentpu/target.py, targets/*.toml) and the configuration registers (register map 4):
the loader, board_config from the selected target, the arch hash, the FPGA flow's generics
matching the RTL's defaults, and the host building its Config from the device."""
import dataclasses
import re
import subprocess
import sys
from pathlib import Path

import pytest

from opentpu import target as T
from opentpu.host import regs as R
from opentpu.host.board import Board, ConfigMismatch, check_fits, device_config
from opentpu.host.fake import FakeTransport
from opentpu.isasim import Config, board_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def no_env(monkeypatch):
    for k in ("OTPU_TARGET", "OTPU_MCOLS", "OTPU_LANES", "OTPU_PAIR", "OTPU_DSTEP", "OTPU_STREAM",
              "OTPU_ACT_ROWS"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_every_target_loads():
    assert {"ypcb-00338", "bcu1525", "nangate45", "nangate45-small"} <= set(T.names())
    for n in T.names():
        t = T.load(n)
        cfg = T.config(t)
        assert isinstance(cfg, Config) and cfg.S == t.arch["S"]
        assert set(T.rtl_params(t)) >= {"MCOLS", "PLATFORM", "ARCH_HASH", "TMEM_IMPL"}
        d = t.to_dict()
        assert d["arch_hash"] == f"{T.arch_hash(cfg):08x}"
    ids = [T.load(n).platform_id for n in T.names()]
    assert len(ids) == len(set(ids)), "platform ids are unique"


def test_board_config_is_the_card_by_default(no_env):
    """board_config() before targets: the same Config (its fields are in every cache key), but
    for the MX build's ACT_ROWS 16 (0 before; release r2's default variant m4: MCOLS 4, 64K TMEM
    words, as before)."""
    legacy = Config(S=1, D=128, MCOLS=4, ACT_BLOCKS=128, LANES=8, ACT_ROWS=16,
                    TMEM_WORDS=1 << 16, IMEM_WORDS=1 << 15, DRAM_BYTES=1 << 32, PAIR=False,
                    DSTEP=False, STREAM=False)
    assert board_config() == legacy
    assert T.current().name == "ypcb-00338"
    no_env.setenv("OTPU_MCOLS", "2")
    no_env.setenv("OTPU_PAIR", "1")
    assert board_config() == dataclasses.replace(legacy, MCOLS=2, PAIR=True)


def test_board_config_follows_otpu_target(no_env):
    no_env.setenv("OTPU_TARGET", "ypcb-00338:m8")              # a variant (test_card_variants)
    c = board_config()
    assert (c.MCOLS, c.ACT_ROWS, c.TMEM_WORDS) == (8, 16, 1 << 17)
    no_env.setenv("OTPU_TARGET", "bcu1525")
    c = board_config()
    assert (c.S, c.D, c.MCOLS, c.DRAM_BYTES) == (2, 128, 4, 1 << 32)
    no_env.setenv("OTPU_TARGET", "nangate45-small")
    c = board_config()
    assert (c.D, c.MCOLS, c.LANES, c.TMEM_WORDS, c.IMEM_WORDS) == (32, 2, 4, 16384, 8192)
    no_env.setenv("OTPU_TARGET", "no-such-board")
    with pytest.raises(T.TargetError, match="no target 'no-such-board'"):
        board_config()


def _hash_without_numerics(cfg) -> int:
    """arch_hash before the MX numerics (what today's production host computes)."""
    import hashlib
    import json
    d = dataclasses.asdict(cfg)
    d.pop("SEQ_WC")                     # (fields that host's Config does not have)
    d.pop("DUAL")
    d.pop("DAW")
    s = json.dumps({k: (int(v) if isinstance(v, bool) else v) for k, v in sorted(d.items())},
                   separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:4], "big")


def test_arch_hash_is_stable_and_sensitive():
    card = T.config("ypcb-00338")
    # in the bitstreams' CFG_HASH: keep them (release r2's m4: MCOLS 4 with wait entries, 64K TMEM
    # words; its m8 at 128K (test_card_variants) and r2's dev build 83740e79, m8 at 64K; MX
    # production 8b33206e, MCOLS 4 at ACT_ROWS 16 and 64K, without; ff270e6e at ACT_ROWS 4)
    assert f"{T.arch_hash(card):08x}" == "4ffd1e17"
    assert f"{T.arch_hash(dataclasses.replace(card, MCOLS=8)):08x}" == "19a2400e"
    mx = dataclasses.replace(card, MCOLS=4, SEQ_WC=False, TMEM_WORDS=1 << 16)
    assert f"{T.arch_hash(mx):08x}" == "89417856"
    four = dataclasses.replace(mx, ACT_ROWS=0)
    assert f"{T.arch_hash(four):08x}" == "ff270e6e"
    # production's CFG_HASH (ACT_ROWS 4, the previous numerics), what a host of the previous
    # numerics computes for it: it refuses MX builds (T.NUMERICS)
    assert f"{_hash_without_numerics(four):08x}" == "e170f967"
    # the ASIC's card point is the card's arch (release r2's m4) with 8 ACT RAM rows (docs/asic.md)
    assert T.arch_hash(dataclasses.replace(card, ACT_ROWS=8)) == T.arch_hash(T.config("nangate45"))
    for f in dataclasses.fields(Config):
        v = getattr(card, f.name)
        other = dataclasses.replace(card, **{f.name: (not v) if isinstance(v, bool) else v * 2
                                             if v else 1})
        assert T.arch_hash(other) != T.arch_hash(card), f.name


def test_card_variants(tmp_path):
    """targets/ypcb-00338.toml's variants (TARGET / OTPU_TARGET ypcb-00338:<name>): m4, the
    default (release r2: MX production 8b33206e's MXU, 4 columns at E 4, 64K TMEM words, with the
    rest of r2), and m8 (8 MX columns at E 8, 128K words: r2.1), each its own CFG_HASH (pinned:
    the bitstreams report them)."""
    m8, m4 = T.load("ypcb-00338:m8"), T.load("ypcb-00338:m4")
    assert T.load("ypcb-00338").variant == "m4" and T.config("ypcb-00338") == T.config(m4)
    c8 = T.config(m8)
    assert (c8.MCOLS, c8.TMEM_WORDS, m8.uarch["MXU_E"]) == (8, 1 << 17, 8)
    c4 = T.config(m4)
    assert (c4.MCOLS, c4.ACT_ROWS, c4.TMEM_WORDS, c4.SEQ_WC, m4.uarch["MXU_E"]) == \
        (4, 16, 65536, True, 4)
    assert f"{T.arch_hash(T.config(m8)):08x}" == "46b96e6d"
    assert f"{T.arch_hash(c4):08x}" == "4ffd1e17"
    p = T.rtl_params(m4)
    assert (p["MCOLS"], p["MXU_E"], p["TMEM_WORDS"], p["ARCH_HASH"]) == (4, 4, 65536, 0x4ffd1e17)
    # run_vivado.sh's CFG_HASH: the variant's knobs on the variant
    assert T.arch_hash(T.build_config(m4, MCOLS=4, LANES=8, ACT_ROWS=16, TMEM_WORDS=65536,
                                      DSTEP=1, SEQ_WC=1)) == 0x4ffd1e17
    with pytest.raises(T.TargetError, match="no variant 'm9'"):
        T.load("ypcb-00338:m9")
    # the file's [arch] / [uarch] state its default variant's values
    p = tmp_path / "ypcb-00338.toml"
    p.write_text((ROOT / "targets/ypcb-00338.toml").read_text().replace(
        'variant = "m4"', 'variant = "m8"'))
    with pytest.raises(T.TargetError, match="differ from the default variant 'm8'"):
        T.load(str(p))


def test_build_config_applies_the_flow_knobs():
    t = T.load("ypcb-00338")
    assert T.build_config(t, MCOLS=4, LANES=8, ACT_ROWS=16, DSTEP=1) == T.config(t)
    assert T.build_config(t, MCOLS=4, LANES=8, DSTEP=1) == T.config(t)       # the target's 16
    c = T.build_config(t, MCOLS=2, ACT_ROWS=2, DSTEP=0)
    assert (c.MCOLS, c.ACT_ROWS, c.DSTEP, c.STREAM, c.PAIR) == (2, 0, False, False, True)
    assert T.build_config(t, MCOLS=4, ACT_ROWS=8).ACT_ROWS == 8


def _sv_defaults(path: Path) -> dict:
    """`parameter int NAME = value` defaults of a SystemVerilog module (simple expressions)."""
    out = {}
    for m in re.finditer(r"parameter\s+(?:int|bit)\s+(\w+)\s*=\s*([^,;\n/]+)", path.read_text()):
        v = m.group(2).strip().replace("1'b", "")
        try:
            out[m.group(1)] = int(eval(v, {}, dict(out)))     # noqa: S307 (our own RTL)
        except Exception:
            pass
    return out


def test_card_target_matches_the_rtl_defaults():
    """The card's target states what otpu_board / otpu_fpga_top_ld build by default, so passing
    the target's values as generics leaves the bitstream's netlist as it was."""
    p = T.rtl_params("ypcb-00338")
    board = _sv_defaults(ROOT / "rtl/shell/otpu_board.sv")
    for k in ("D", "IMEM_WORDS", "ACT_BLOCKS", "FIFO_DEPTH", "WIN", "WPB",
              "TRACE_DEPTH", "PQ_WIN", "CHASH", "DSTEP"):
        assert p[k] == board[k], k
    assert p["RPB"] == 8 * p["LANES"]                       # otpu_board: RPB = 8 * LANES
    top = _sv_defaults(ROOT / "rtl/boards/ypcb-00338/otpu_fpga_top_ld.sv")
    # TMEM_WORDS from the top (release r2: 128K; otpu_board's default stays 64K, the others')
    for k in ("MCOLS", "ACT_ROWS", "VPU_CL", "MXU_IMPL", "MXU_E", "LANES", "ULANES", "DDR_MTS",
              "PLATFORM", "TRACE_DEPTH", "SEQ_WC", "TMEM_IMPL", "TMEM_WORDS") + T.TMEM3_KEYS:
        assert p.get(k, top[k]) == top[k], k
    assert _sv_defaults(ROOT / "rtl/boards/ypcb-00338/otpu_native_sys.sv")["TRACE_DEPTH"] \
        == p["TRACE_DEPTH"]
    # release r2: MCOLS 8 at E 8, WAITC, the one pumped TMEM copy (r2pgRs, docs/tmem_nocopy.md)
    assert T.sim_uarch("ypcb-00338") == {"WIN": 16, "RPB": 128, "WPB": 1, "FIFO_DEPTH": 1024,
                                         "MXU_E": 4, "SEQ_WC": 1, "TMEM_IMPL": 3, "TMEM_NSB": 2,
                                         "TMEM_RPL": 1, "TMEM_TDP": 1, "TMEM_GST": 1,
                                         "TMEM_PMP": 1}


def test_target_cli():
    run = lambda *a: subprocess.run([sys.executable, "-m", "opentpu.target", *a], cwd=ROOT,
                                    capture_output=True, text=True, check=True).stdout
    env = dict(line.split("=", 1) for line in run("env", "ypcb-00338").split())
    assert (env["T_MCOLS"], env["T_ACT_ROWS"], env["T_PLATFORM"], env["T_ARCH_HASH"],
            env["T_TMEM_WORDS"]) == ("4", "16", "1", "4ffd1e17", "65536")   # release r2 (m4)
    env = dict(line.split("=", 1) for line in run("env", "ypcb-00338:m8").split())
    assert (env["T_MCOLS"], env["T_MXU_E"], env["T_TMEM_WORDS"], env["T_ARCH_HASH"]) == \
        ("8", "8", "131072", "46b96e6d")                # its variant m8 (run_vivado.sh TARGET)
    # r2's dev build 83740e79 (64K words); MX production 8b33206e's (MCOLS 4, no wait entries),
    # and its 4-row build's
    assert run("hash", "ypcb-00338", "MCOLS=8", "LANES=8", "ACT_ROWS=16", "TMEM_WORDS=65536",
               "DSTEP=1", "SEQ_WC=1").strip() == "19a2400e"
    assert run("hash", "ypcb-00338", "MCOLS=4", "LANES=8", "ACT_ROWS=16", "TMEM_WORDS=65536",
               "DSTEP=1", "SEQ_WC=0").strip() == "89417856"
    assert run("hash", "ypcb-00338", "MCOLS=4", "LANES=8", "ACT_ROWS=4", "TMEM_WORDS=65536",
               "DSTEP=1", "SEQ_WC=0").strip() == "ff270e6e"
    assert "bcu1525" in run("list")


def test_bad_target_files(tmp_path):
    good = (ROOT / "targets/nangate45-small.toml").read_text()
    for edit, msg in ((("[uarch]", "[uarchx]"), "no \\[uarch\\]"),
                      (("TMEM_WORDS = 16384", "TMEM_WORDZ = 16384"), "not Config fields"),
                      (("DRAM_BYTES = 268435456", "DRAM_BYTES = 8589934592"), "over 4 GiB"),
                      (('kind = "asic"', 'kind = "rocket"'), "kind")):
        p = tmp_path / "nangate45-small.toml"
        p.write_text(good.replace(*edit))
        with pytest.raises(T.TargetError, match=msg):
            T.load(str(p))
    # otpu_seq's wait entries (SEQ_WC) count a unit's last 15 instructions: a window of 16 at most
    p = tmp_path / "nangate45-small.toml"
    wc = good.replace("SEQ_WC = false", "SEQ_WC = true")
    p.write_text(wc)
    assert T.load(str(p)).arch["SEQ_WC"] and T.sim_uarch(T.load(str(p)))["SEQ_WC"] == 1
    T.load.cache_clear()
    p.write_text(wc.replace("WIN = 16", "WIN = 32"))
    with pytest.raises(T.TargetError, match="SEQ_WC .* needs WIN <= 16"):
        T.load(str(p))


# ------------------------------------------------------------------------------ register map 4
def test_cfg_register_codec():
    c = dict(S=2, TMEM_WORDS=1 << 16, IMEM_WORDS=1 << 15, ACT_BLOCKS=128, DRAM_BYTES=1 << 32,
             channels=2, channel_bytes=1 << 31, MXU_IMPL=2, TMEM_IMPL=1, VPU_CL=2, ULANES=8,
             SEQ_WC=1, platform=2, hash=0x8B699C81)
    assert R.cfg(R.cfg_words(c)) == c
    assert R.cfg(R.cfg_words(dict(c, SEQ_WC=0)))["SEQ_WC"] == 0


def test_device_config_from_the_cfg_registers(no_env):
    """Map 4: the sizes come from the device; the hash must be the reported Config's."""
    t = FakeTransport(devname=None, regmap=4, pair=True, dstep=True, stream=True, seq_wc=True)
    info = Board(t).info()                      # (release r2's build: wait entries, CFG_UARCH 24)
    assert info["cfg"]["platform"] == 1 and info["cfg"]["hash"] == 0x4ffd1e17   # (r2's m4)
    assert info["act_rows"] == 16 and info["caps"]["act_rows"]
    cfg = device_config(info)
    assert cfg == T.config("ypcb-00338")
    small = dict(R.cfg(t.cfg_words), TMEM_WORDS=1 << 15, IMEM_WORDS=1 << 14, ACT_BLOCKS=64,
                 DRAM_BYTES=1 << 30)
    small["hash"] = T.arch_hash(dataclasses.replace(cfg, TMEM_WORDS=1 << 15, IMEM_WORDS=1 << 14,
                                                    ACT_BLOCKS=64, DRAM_BYTES=1 << 30))
    info = Board(FakeTransport(devname=None, regmap=4, pair=True, dstep=True, stream=True,
                               cfg=small)).info()
    c2 = device_config(info)
    assert (c2.TMEM_WORDS, c2.IMEM_WORDS, c2.ACT_BLOCKS, c2.DRAM_BYTES) == \
        (1 << 15, 1 << 14, 64, 1 << 30)
    with pytest.raises(ConfigMismatch, match="TMEM_WORDS=65536 and the device has 32768"):
        check_fits(cfg, info)
    check_fits(dataclasses.replace(c2, DRAM_BYTES=1 << 24), info)     # less DRAM: fine
    info["cfg"]["hash"] ^= 1
    with pytest.raises(ConfigMismatch, match="CFG_HASH"):
        device_config(info)
    with pytest.raises(ConfigMismatch, match="CFG_HASH"):           # programs built elsewhere
        check_fits(dataclasses.replace(c2, DRAM_BYTES=1 << 24), info)


def test_the_engine_backend_checks_the_hash(no_env):
    """BoardBackend (the Engine's, otpu-serve's: its Config from llm.device_config, not
    board.device_config) refuses a bitstream whose CFG_HASH is not this host's hash of the
    configuration its registers report: one built with other numerics (the MXU before MX
    against an MX host) would run the programs and compute something else."""
    import numpy as np

    from opentpu.host.board import BoardBackend
    t = FakeTransport(devname=None, regmap=4, pair=True, dstep=True, stream=True)
    cfg = device_config(Board(t).info())
    small = dataclasses.replace(cfg, DRAM_BYTES=1 << 22)
    BoardBackend(small, [np.zeros(4096, np.uint8)], transport=t, status=False).close()
    other = dict(R.cfg(t.cfg_words))
    other["hash"] ^= 0x10
    bad = FakeTransport(devname=None, regmap=4, pair=True, dstep=True, stream=True, cfg=other)
    with pytest.raises(ConfigMismatch, match="CFG_HASH .* numerics"):
        BoardBackend(small, [np.zeros(4096, np.uint8)], transport=bad, status=False)


@pytest.mark.parametrize("regmap", [3, 4])
def test_a_bitstream_before_mx_is_refused(no_env, regmap):
    """A bitstream without CAPS bit29 (the MX numerics: an MXU before MX, whatever its hash) is
    refused with ConfigMismatch by device_config (otpu-chat, selftest, diag, hwlens, the quals)
    and by BoardBackend (check_fits: the Engine's, otpu-serve's); one without CAPS at all
    (register map 1) too."""
    import numpy as np

    from opentpu.host.board import BoardBackend
    kw = dict(devname=None, regmap=regmap, pair=True, dstep=True, stream=True)
    good = Board(FakeTransport(**kw)).info()
    assert good["caps"]["mx"]
    cfg = dataclasses.replace(device_config(good), DRAM_BYTES=1 << 22)
    old = FakeTransport(mx=False, **kw)
    info = Board(old).info()
    assert not info["caps"]["mx"]
    for refuse in (lambda: device_config(info), lambda: check_fits(cfg, info),
                   lambda: BoardBackend(cfg, [np.zeros(4096, np.uint8)], transport=old,
                                        status=False)):
        with pytest.raises(ConfigMismatch, match="lacks the MX numerics \\(CAPS bit29\\)"):
            refuse()
    v1 = Board(FakeTransport(devname=None, regmap=1)).info()
    assert v1["caps"] is None
    with pytest.raises(ConfigMismatch, match="MX numerics"):
        device_config(v1)


def test_device_config_before_map_4_is_the_card(no_env):
    info = Board(FakeTransport(devname=None, regmap=3, pair=True, dstep=True,
                               stream=True)).info()
    assert info["cfg"] is None              # (no CFG_UARCH: a map 3 build takes no wait entries)
    assert device_config(info) == dataclasses.replace(T.config("ypcb-00338"), SEQ_WC=False)
    check_fits(board_config(TMEM_WORDS=1 << 20), info)              # nothing to check against


# ------------------------------------------------------------------------------ RTL file lists
def _make_rtl(path: Path) -> list[str]:
    """The board Makefile's RTL list (rtl/ paths)."""
    m = re.search(r"RTL := \$\(addprefix \$\(ROOT\)/rtl/, (.*?)\)\n", path.read_text(), re.S)
    return ["rtl/" + f for f in m.group(1).replace("\\", " ").split()]


def _tcl_rtl(path: Path) -> list[str]:
    m = re.search(r"set rtl \[list \\(.*?)\]\n", path.read_text(), re.S)
    return m.group(1).replace("\\", " ").split()


def test_rtl_file_lists_agree():
    """Every flow's RTL list is opentpu.target.sources(): the board Makefile's lint and
    create_project.tcl (the board's top last, the Makefile linting it separately), the
    simulators' (rtlsim.RTL_SOURCES: the generic cells, plus the simulation-only DRAM model,
    native adapter, top and single-copy TMEM grant model) and tools/synth's."""
    from opentpu import rtlsim
    card = [f"rtl/{f}" for f in T.sources("ypcb-00338")]
    tcl = _tcl_rtl(ROOT / "boards/ypcb-00338/vivado/create_project.tcl")
    assert tcl == card
    make = _make_rtl(ROOT / "boards/ypcb-00338/Makefile")
    assert make == [f for f in card if not f.endswith("otpu_fpga_top_ld.sv")]
    sim = [f for f in rtlsim.RTL_SOURCES
           if f not in ("mem/otpu_dram.sv", "mem/otpu_native_dram.sv",
                        "mem/otpu_native_dram2.sv", "top/otpu_top.sv",
                        "exp/tmem_nocopy/otpu_tmem_model.sv")]
    assert sim == T.sources("nangate45", "generic")
    for f in T.CORE + T.SHELL + sum(T.TECH.values(), []) + sum(T.BOARDS.values(), []):
        assert (ROOT / "rtl" / f).exists(), f
    bcu = [f"rtl/{f}" for f in T.sources("bcu1525")]
    assert "rtl/tech/xilinxus/otpu_tech_dsp.sv" in bcu
    assert _tcl_rtl(ROOT / "boards/bcu1525/vivado/create_project.tcl") == bcu
    assert _make_rtl(ROOT / "boards/bcu1525/Makefile") == \
        [f for f in bcu if not f.endswith("otpu_fpga_top_bcu.sv")]


def test_board_uarch_builds_the_cards_collective_unit():
    """The card's build has no GATHER (run_vivado.sh's COLL_GATHER default, and
    create_project.tcl's: 0 since release r2, docs/infra_hunt.md 5), and the board uarch
    simulates the same unit (rtlsim.BOARD_UARCH: tb_top / tb_board COLL_GATHER), so the gates'
    board runs are of the BAR-only unit; S > 1 and the default uarch keep GATHER."""
    from opentpu import rtlsim
    if T.current().name != "ypcb-00338":
        pytest.skip("the card's target only")
    sh = (ROOT / "boards/ypcb-00338/run_vivado.sh").read_text()
    tcl = (ROOT / "boards/ypcb-00338/vivado/create_project.tcl").read_text()
    build = re.findall(r'^export COLL_GATHER="\$\{COLL_GATHER:-(\d)\}"', sh, re.M)
    fallback = re.findall(r'^set COLL_GATHER \[expr .* : (\d)\}\]$', tcl, re.M)
    assert build == fallback == [str(rtlsim.BOARD_UARCH["COLL_GATHER"])] == ["0"]
    assert not rtlsim.gathers(1, rtlsim.BOARD_UARCH) and rtlsim.gathers(2, rtlsim.BOARD_UARCH)
    assert rtlsim.gathers(1, {"COLL_GATHER": 1})


def test_bcu1525_target_matches_its_rtl_defaults():
    """The BCU1525's top and sys wrapper default to its target's values (run_vivado.sh passes
    them as generics too), with the channel layout otpu_board and otpu_ctrl assume (two channels
    per slice, 2 GiB a slice addresses of each: CFG_DRAM)."""
    p = T.rtl_params("bcu1525")
    for f in ("otpu_fpga_top_bcu.sv", "otpu_bcu_sys.sv"):
        top = _sv_defaults(ROOT / "rtl/boards/bcu1525" / f)
        for k in ("S", "D", "MCOLS", "ACT_BLOCKS", "TMEM_WORDS", "IMEM_WORDS", "FIFO_DEPTH",
                  "LANES", "WIN", "WPB", "MXU_IMPL", "MXU_CL", "VPU_CL", "ULANES", "TRACE_DEPTH",
                  "PQ_WIN", "CHASH", "DSTEP", "DDR_MTS", "PLATFORM", "SEQ_WC"):
            assert p[k] == top[k], (f, k)
    assert p["RPB"] == 8 * p["LANES"]                       # the tops: RPB = 8 * LANES
    assert (p["NCH"], p["CH_LOG2"], p["DRAM_LOG2"]) == (2, 31, 32)


def test_tech_families_define_the_same_cells():
    """rtl/tech/<family>/ files define the same modules with the same ports and parameters."""
    def iface(path: Path) -> dict:
        txt = re.sub(r"//.*", "", path.read_text())
        mods = {}
        for m in re.finditer(r"module\s+(\w+)\s*#\((.*?)\)\s*\((.*?)\);", txt, re.S):
            norm = lambda s: re.sub(r"\s+", " ", s).strip()
            mods[m.group(1)] = (norm(m.group(2)), norm(m.group(3)))
        return mods
    ref = iface(ROOT / "rtl/tech/generic/otpu_tech_dsp.sv")
    assert set(ref) == {"otpu_pe", "otpu_mxce"}
    for fam in ("xilinx7", "xilinxus"):
        assert iface(ROOT / f"rtl/tech/{fam}/otpu_tech_dsp.sv") == ref, fam


# the DRAM's handshakes and read data that otpu_slice's units, loader and arbiters see only through
# the SW port's, port A's and port B's registers (sw_rdy_i, wr_idle_i, a_rdy_i, b_rdy_i, rd_idle_i,
# b_rvalid_i, ...)
_RAW = r"\b(sw_rdy|wr_idle|a_rdy|b_rdy|rd_idle|b_rvalid|b_rtag|b_rdata)\b"


def _slice_sw_uses(txt: str) -> list:
    """otpu_slice's reads of the DRAM's raw handshakes and read data (_RAW; comments out), as
    (line, text), outside its port list and the registers: the SW port's (from its input, wire
    sw_req_c, to the end of its block, end : sw_reg), port B's (from assign b_rdy_i to end :
    pb_reg) and port A's (from assign a_rdy_i to end : pa_reg); a unit's port of that name
    (.wr_idle(x)) is not one, its implicit connection (.wr_idle,) is."""
    code = re.sub(r"//.*", "", txt)
    head = re.search(r"module\s+otpu_slice\b.*?#\(.*?\)\s*\((.*?)\)\s*;", code, re.S)
    regs = [re.search(r"\bwire\s+sw_req_c\b.*?\bend\s*:\s*sw_reg\b", code, re.S),
            re.search(r"\bassign\s+b_rdy_i\b.*?\bend\s*:\s*pb_reg\b", code, re.S),
            re.search(r"\bassign\s+a_rdy_i\b.*?\bend\s*:\s*pa_reg\b", code, re.S)]
    assert head and all(regs), "otpu_slice: no port list or no SW / port B / port A register block"
    out = []
    for m in re.finditer(_RAW, code):
        if head.start(1) <= m.start() < head.end(1) or \
                any(r.start() <= m.start() < r.end() for r in regs):
            continue
        if code[m.start() - 1] == "." and re.match(r"\s*\(", code[m.end():]):
            continue                            # (a unit's port of that name: .wr_idle(wr_idle_i))
        line = code.count("\n", 0, m.start()) + 1
        out.append((line, code.splitlines()[line - 1].strip()))
    return out


def test_the_slice_reads_dram_writes_through_the_sw_register():
    """A QST write can sit a cycle in the SW port's register, a port B request in port B's output
    register and its read data in port B's input register, an MXU scale read in port A's output
    register (every target's), in neither the units nor the DRAM: the slice's units, loader and
    arbiters see the ports as sw_rdy_i / wr_idle_i / a_rdy_i / b_rdy_i / rd_idle_i / b_rvalid_i /
    b_rtag_i / b_rdata_i (the registers'), never the DRAM's raw handshakes or data, which only the
    registers read. A raw wr_idle would end a QST, an ST, a HALT CHAIN or a WAITW timeout before
    the DRAM has the write; a raw rd_idle would start the loader with a read still on its way; a
    raw a_rdy or b_rdy would let a unit's request pass a held one."""
    txt = (ROOT / "rtl/top/otpu_slice.sv").read_text()
    assert _slice_sw_uses(txt) == []
    # the check sees one: the quantizer's grant on the raw ready
    bad = txt.replace("if (g == G_Q && q_awant && !sw_rdy_i) aok[g] = 1'b0;",
                      "if (g == G_Q && q_awant && !sw_rdy) aok[g] = 1'b0;")
    assert bad != txt and [t for _, t in _slice_sw_uses(bad)] == \
        ["if (g == G_Q && q_awant && !sw_rdy) aok[g] = 1'b0;"]
    # and a unit connected to the raw one implicitly (.wr_idle, as before the register)
    bad = txt.replace(".a_want(q_awant), .wr_idle(wr_idle_i),", ".a_want(q_awant), .wr_idle,")
    assert bad != txt and [t for _, t in _slice_sw_uses(bad)] == [".a_want(q_awant), .wr_idle,"]
    # port B: the MXU's grant on the raw ready, the loader's start on the raw read idle, and the
    # MXU's read data connected implicitly; port A: the MXU's grant on the raw ready
    for a, b in (("assign mxu_bgnt = !dma_breq && b_rdy_i;", "assign mxu_bgnt = !dma_breq && b_rdy;"),
                 ("assign mxu_agnt = a_rdy_i;", "assign mxu_agnt = a_rdy;"),
                 ("else if (!ld_go) ld_go <= rd_idle_i;", "else if (!ld_go) ld_go <= rd_idle;"),
                 (".b_rdata(b_rdata_i),\n", ".b_rdata,\n")):
        assert txt.count(a) == 1, a
        bad = txt.replace(a, b)
        assert [t for _, t in _slice_sw_uses(bad)] == [b.strip()], b


def test_the_board_counts_the_port_registers_as_requests_pending():
    """otpu_board's STATUS WR_IDLE and go_ok see a slice's writes all taken (q_wr_idle) once its
    adapter has none pending and neither its SW register nor its port B output register holds one
    (sw_req, b_req && b_we: the registers' valid bits), and its reads all back (q_rd_idle) once
    the adapter has none, neither port A's output register (a_req: reads only) nor port B's holds
    one (b_req && !b_we) and port B's input register no data (m_rv: a b_rvalid the cycle before);
    the adapter's wr_idle and rd_idle (m_wr_idle, m_rd_idle) go to the slice alone, for slice 0 and
    the others. sim/verilator/tb_board.sv checks the same in every board simulation."""
    code = re.sub(r"//.*", "", (ROOT / "rtl/shell/otpu_board.sv").read_text())
    # (5: slice 0's adapter is otpu_native_dram or, DUAL, otpu_native_dram2)
    assert re.findall(r"\.wr_idle\(([^)]*)\)", code) == ["m_wr_idle"] * 5
    assert re.findall(r"\.rd_idle\(([^)]*)\)", code) == ["m_rd_idle"] * 5
    assert re.findall(r"assign\s+q_wr_idle\[(\w+)\]\s*=\s*([^;]*);", code) == \
        [(k, "m_wr_idle && !sw_req && !(b_req && b_we)") for k in ("0", "s")]
    assert re.findall(r"assign\s+q_rd_idle\[(\w+)\]\s*=\s*([^;]*);", code) == \
        [(k, "m_rd_idle && !a_req && !(b_req && !b_we) && !m_rv") for k in ("0", "s")]
    assert re.findall(r"m_rv\s*<=\s*([^;]*);", code) == \
        [f"!x_srst[{k}] && b_rvalid" for k in ("0", "s")]


def _lane_loops(code: str) -> dict:
    """The generate loops over lanes of a source (comments out): {label: body}, the body from
    "for (genvar l ...) begin : label" to its matching end."""
    out = {}
    for m in re.finditer(r"for\s*\(\s*genvar\s+l\b[^)]*\)\s*begin\s*:\s*(\w+)", code):
        depth = 1
        for t in re.finditer(r"\b(begin|end)\b", code[m.end():]):
            depth += 1 if t.group(1) == "begin" else -1
            if depth == 0:
                out[m.group(1)] = code[m.end():m.end() + t.start()]
                break
    return out


def test_the_vpu_lanes_take_ss_act_from_their_copies():
    """ss_act (the stream engine's grant: stream mode) selects every lane's RDOT term and the
    tail's VOP-mode operands; as one flip-flop it reached some 2,700 endpoints (the ASIC card
    point's worst register path, through 190 buffers; only Vivado's max_fanout replicated it).
    The lanes read copies instead, three a lane (otpu_vpu g_sar: sa_r, cm_u, cm_q, loaded with
    ss_act's next value, each its own kept instance: identical flip-flops would merge): no lane
    loop of otpu_vpu reads ss_act, otpu_se_tail has no one-bit mode, and every copy is kept."""
    vpu = re.sub(r"//.*", "", (ROOT / "rtl/vpu/otpu_vpu.sv").read_text())
    tail = re.sub(r"//.*", "", (ROOT / "rtl/vpu/otpu_se_tail.sv").read_text())

    def raw(code):
        return {n: re.findall(r"\bss_act\b", b) for n, b in _lane_loops(code).items()
                if re.search(r"\bss_act\b", b)}
    loops = _lane_loops(vpu)
    assert {"g_red", "g_slane", "g_sar"} <= set(loops) and raw(vpu) == {}
    assert len(re.findall(r'\(\*\s*keep_hierarchy\s*=\s*"yes"\s*\*\)\s*otpu_delay\b',
                          loops["g_sar"])) == 3
    assert re.search(r"\.cm_u\s*,\s*\.cm_q\b", vpu)
    tl = _lane_loops(tail)
    assert not re.search(r"\b(cm|uc)\b", tail)
    assert "cm_u[l]" in tl["g_u"] and "cm_q[l]" in tl["g_lq"]
    # the check sees one: a lane's term on ss_act again
    a = "assign rsa[l] = (SE && sa_r[l])"
    assert vpu.count(a) == 1
    assert raw(vpu.replace(a, "assign rsa[l] = (SE && ss_act)")) == {"g_red": ["ss_act"]}


def test_the_long_wire_drivers_are_copies_per_consumer():
    """A register whose loads spread over the die pays its route (r22's m8 at 133.33 MHz: 88% of
    the near-critical core delay, half the paths at most one LUT): the drivers that fanned out so
    have a copy per consumer group, each loaded with the same next value (cycle-exact), kept
    (identical flip-flops would merge). The VPU's enable per lane (g_enl: no lane loop, tail unit
    or composite lane reads the one en), the fill index per lane (g_sfi), the MXU's decoded scale
    per column (g_bd), the pumped TMEM's write stage per RAM (g_pmp g_sl) and, with wait counts,
    the sequencer's command store write per unit (g_cmd g_wq: a cycle late). The MXU's result
    FIFO needs none: a block RAM read ahead (test below). Wide data stays one register: the ACT
    RAM's write stage
    (wbe, wd) a copy per column cost the m8 card 9 k LUTs and 9.5 k flip-flops and a congested
    corner (docs/timing_ffs.md section 3); its pieces copy only their address and select."""
    def src(p):
        return re.sub(r"//.*", "", (ROOT / p).read_text())
    kept = r'\(\*\s*keep_hierarchy\s*=\s*"yes"\s*\*\)\s*otpu_delay\b'
    bare_en = r"(?<![.\w])en(?![\w(\[])"
    vpu, tail, comp = (src("rtl/vpu/otpu_vpu.sv"), src("rtl/vpu/otpu_se_tail.sv"),
                       src("rtl/vpu/otpu_se_comp.sv"))
    loops = _lane_loops(vpu)
    for n in ("g_cbuf", "g_lane", "g_slane", "g_red"):
        assert not re.search(bare_en, loops[n]) and "en_l[l]" in loops[n], n
    assert len(re.findall(kept, loops["g_enl"])) == 1 and len(re.findall(kept, loops["g_sfi"])) == 1
    assert re.search(r"\.cen\(en_l\)", vpu) and re.search(r"\.en_l\b", vpu)
    tl = _lane_loops(tail)
    assert "cen[l]" in tl["g_u"] and "cen[l]" in tl["g_lq"] and "fi[l]" in tl["g_buf"]
    assert not re.search(r"\bue\b", tail) and not re.search(r"\[fi\]", tail)
    cl = _lane_loops(comp)
    assert "en_l[l]" in cl["g_lane"] and not re.search(bare_en, cl["g_lane"])
    mxu = src("rtl/mxu/otpu_mxu.sv")
    assert re.search(r"begin\s*:\s*g_bd\s*" + kept, mxu)
    assert "dhi1[j].s[g]" in mxu and "vhi1[j]" in mxu and not re.search(r"\bdhi1\.s\b", mxu)
    assert re.search(r"begin\s*:\s*g_rfm.*?otpu_ram_1r1w_ra\b.*?\.ra_n\(RFW'\(rf_qn", mxu, re.S)
    act = src("rtl/mem/otpu_actram.sv")
    head, row = act.split("begin : g_row", 1)
    assert re.search(r"logic \[D\*8-1:0\]\s+wd;", head) and not re.search(r"\bwd\s*;", row.split("end")[0])
    assert re.search(r"wb <= w_idx_h\[DW \+: BW\];", row) and "wsel <= wsel_n[j];" in row
    tm = src("rtl/mem/otpu_tmem.sv")
    pmp = tm[tm.index("begin : g_pmp"):tm.index("begin : g_ram")]
    sl = pmp[pmp.index("begin : g_sl"):]
    assert "otpu_pump_bank" in sl and re.search(r"\.wv\(sv\)", sl) and re.search(r"\.wa\(sa\)", sl)
    seq = src("rtl/seq/otpu_seq.sv")
    cmd = seq[seq.index("begin : g_cmd"):]
    assert re.search(r"if \(WAITC\) begin\s*:\s*g_wq\s*" + kept, cmd)
    assert re.search(r"if \(we\) m\[wa\] <= WAITC \? c_cmd_w : c_cmd;", cmd)
    assert not re.search(r"m\[free_slot\]", seq)


def test_the_widest_lut_rams_are_block_rams_read_ahead():
    """A LUT RAM puts every address bit on each of its cells (6 data bits a RAM32M), so a wide one
    is a fanout of thousands per bit: on r22-era m8 the DMA's gather (39,424 address pins), the B
    response FIFO (38,528) and the MXU's result FIFO (37,632) were the most of any memories. They
    are otpu_ram_1r1w_ra (rtl/tech: a block RAM read at the address's next value, the same cycles
    as the asynchronous read; flip-flops on the ASIC), read ahead at their pointer's next value,
    and the per-RAM copies of their write addresses that the LUT RAMs needed are gone."""
    def src(p):
        return re.sub(r"//.*", "", (ROOT / p).read_text())
    ra = r"otpu_ram_1r1w_ra\s*#\([^;]*?\.ra_n\((\w+)"
    dma = src("rtl/dma/otpu_dma.sv")
    gb = dma[dma.index("begin : g_gb"):]
    assert re.search(ra, gb).group(1) == "gh_n" and "logic [31:0] gb [NGC]" not in dma
    assert re.search(r"assign gh_n = ", dma) and not re.search(r"gb_wa \[SPC\]\[W\]", dma)
    nd = src("rtl/mem/otpu_native_dram.sv")
    assert re.search(ra, nd[nd.index("begin : g_rb"):]).group(1) == "rb_hn"
    assert "rb_tq" not in nd and not re.search(r"\brbm\s*\[", nd)
    mxu = src("rtl/mxu/otpu_mxu.sv")
    rfm = mxu[mxu.index("begin : g_rfm"):]
    assert re.search(ra, rfm) and "begin : g_s" not in rfm.split("end\n  always_ff")[0]
    for fam in ("generic", "asic"):
        assert "module otpu_ram_1r1w_ra" in src(f"rtl/tech/{fam}/otpu_tech_ram.sv"), fam
    # the native adapter's SW queue data and fill data (11,980 address pins each), at qw_f's next
    g_mem = nd[nd.index("begin : g_mem"):nd.index("assign ra_head")]
    assert [m for m in re.findall(ra, g_mem) if m != "rb_hn"] == ["qw_fn"] * 2
    assert not re.search(r"\b(qwd|wrm)\s*\[", nd)


def test_late_pushes_reach_no_lut_ram_write_enable():
    """A queue whose LUT RAM is written on its push puts the push, which comes late (the end of a
    grant or of the native adapter's write issue: 9 to 12 levels on the card), on every cell's
    write enable (hundreds). The native adapter's A queue (qam) writes its tail every cycle (2 QD
    slots, as its B queue's qbm), and the in-order channel FIFOs (otpu_afifo OOO 0: u_aq, u_ad,
    u_xq, u_xd) write the tail slot every cycle there is room (wready: the pointers' registers);
    a push only moves the tail. And the MXU's ASCALE factors (alpha), registers that hold a value
    rather than a pipeline stage, load off the TMEM grant's fan-out."""
    def src(p):
        return re.sub(r"//.*", "", (ROOT / p).read_text())
    nd = src("rtl/mem/otpu_native_dram.sv")
    assert re.search(r"qam \[2 \* QD\];", nd)
    assert re.search(r"always_ff @\(posedge clk\) qam\[", nd)
    af = src("rtl/shell/otpu_afifo.sv")
    assert "if (OOO ? wput : wready) mem[waddr] <= wdata;" in af
    mxu = src("rtl/mxu/otpu_mxu.sv")
    load = mxu[mxu.index("if (!rst && al_st == 2'd3)"):]
    assert "alpha[j] <= t_rdata" in load.split("`ifndef SYNTHESIS")[0]
    assert "t_gnt" not in mxu[mxu.index("if (!rst && al_st == 2'd3)") - 40:
                              mxu.index("if (!rst && al_st == 2'd3)")]


def test_the_mxu_epilogue_advances_on_flip_flops():
    """The MXU's TMEM grant, the end of the slice's arbitration, was the epilogue's enable (the
    drain's lanes, the read-modify-write, RMAX, the ASCALE loads: on the m8 card ~11.3 k enables,
    the MXU's worst paths at a 150 MHz equivalent). With TQ (the slice's MXU_TQ: TMEM_IMPL 3 with
    replays) TMEM reads what the MXU asks (a lost read is replayed, as the VPU's), the grant only
    takes the writes from a queue (g_tq), and the epilogue advances on ea, a function of flip-flops
    the grant and t_lost reach only through their next values. A first version (72cbb745) kept the
    read in the granted beat: the grant and t_lost fanned out to ea's 118 replicas, the read
    address came out of the queue's LUT RAM, and a push enabled every slot's flip-flops (b_t8_150:
    -0.49 ns). So: TMEM's read address is the request register, the slots load from registers'
    conditions, and the grant reaches the queue's counters only. TQ 0 keeps the grant as the
    enable (g_ng)."""
    def src(p):
        return re.sub(r"//.*", "", (ROOT / p).read_text())
    mxu = src("rtl/mxu/otpu_mxu.sv")
    tq = mxu[mxu.index("if (TQ) begin : g_tq"):mxu.index("end else begin : g_ng")]
    ng = mxu[mxu.index("end else begin : g_ng"):]
    ng = ng[:ng.index("\n  end\n")]
    rest = mxu.replace(tq, "").replace(ng, "")
    # the grant reaches the port declaration and nothing else outside the TMEM interface
    assert re.findall(r"\bt_gnt\b", rest) == ["t_gnt"], re.findall(r".*\bt_gnt\b.*", rest)
    assert "assign ea = t_gnt;" in ng and "assign t_raddr = rq_ad;" in rest
    # in g_tq the grant only takes or pops writes, t_lost only reaches lost_q, and ea is flip-flops'
    def wires(sig):
        return {re.match(r"\s*wire\s+(?:\[[^]]*\]\s*)?(\w+)", l).group(1)
                for l in tq.splitlines() if re.search(rf"\b{sig}\b", l) and re.match(r"\s*wire\b", l)}
    assert wires("t_gnt") == {"take", "pop"}, wires("t_gnt")
    assert re.findall(r".*\bt_lost\b.*", tq.split("`ifndef SYNTHESIS")[0]) == ["        lost_q <= t_lost;"]
    ea = re.search(r"\(\*\s*max_fanout\s*=\s*\d+\s*\*\)\s*wire ea_t = ([^;]*);", tq).group(1)
    assert set(re.findall(r"\w+", ea)) == {"room_q", "lost_q", "tq_z", "rq_any"}, ea
    assert "assign ea = ea_t;" in tq and "assign t_ren = t_ren_q;" in tq
    # TMEM's read enable a register, its next value off t_lost (the slice masks a replay's cycle)
    assert re.search(r"t_ren_q <= \(ea_t \? t_ren_c : rq_en\) & ", tq)
    # the queue's slots: flip-flops written from the counters' registers, never on a push
    gs = tq[tq.index("begin : g_s"):]
    gs = gs[:gs.index("\n    end\n")]
    assert re.search(r"if \(tq_n != 2'd3 && tl == 2'\(i\)\) s\[i\] <=", gs)
    assert not re.search(r"\b(push|t_gnt|ea_t|ea)\b", gs) and "s[" not in tq.replace(gs, "").replace(
        "assign hd = s[tq_h];", "")
    sl = src("rtl/top/otpu_slice.sv")
    assert re.search(r"localparam bit MXU_TQ = TMEM_IMPL == 3 && TMEM_RPL;", sl)
    assert ".TQ(MXU_TQ)" in sl and re.search(r"\.t_lost\(tm_lost\[P_MXU\]\)", sl)
    assert "if (!MXU_TQ) r_en[P_MXU] = '0;" in sl
    assert "if (MXU_TQ && rpl_frz[G_MXU]) r_en[P_MXU] = '0;" in sl


def test_the_real_150_mhz_paths_start_at_flip_flops():
    """The m8 card at a real 150 MHz (r150a): the pumped TMEM's word into the units' capture
    registers is a full core cycle from a clk2x block RAM (multicycle), and it went through the
    bank's RAM select, the lane's bank select and the hold (4 levels, 3.9 ns of route, -0.112 ns):
    a host port of the pumped TMEM (no bypass) selects its lanes' words by {bank, RAM}, one select
    from the RAMs' outputs with its index from flip-flops. The MXU's drained test (c_left == 0, a
    32-bit compare of a 136-load register into the drain lanes' enables, -0.080) is a flip-flop
    (c_lz). The sequencer's command copies load on the unit's readiness, not its start decode.
    The MXU's read reaches TMEM by bank from its flip-flops (otpu_tmem PBR: rb_q / rb_a, decoded
    before the registers), not through the lanes' bank decode in front of TMEM's compares (r150c:
    rq_ad -> the decode -> the compares -> lost_q, 11 levels, -0.007 ns), and so do the DMA's,
    the quantizer's and the VPU's (otpu_tmem_breq, the rotators' merge, registered with their
    requests; r150d: their rotations in front of the ranks and the pumped RAMs' read pins). TMEM's
    replay flag rpv is kept once per bank (rpvb) for each bank's asker multiplexers, replay loads
    and data select."""
    def src(p):
        return re.sub(r"//.*", "", (ROOT / p).read_text())
    tm = src("rtl/mem/otpu_tmem.sv")
    # the pumped bank keeps its hierarchy (otpu_clk2x.tcl names its pins); otpu_tmem_breq beside it
    assert re.search(r'\(\* keep_hierarchy = "yes" \*\)\s*module otpu_pump_bank\b', tm)
    assert "lq[l] = pdir[p] ? pr[p][int'(bk[l]) * NSB + int'(pss[p][bk[l]])] : qb[bk[l]];" in tm
    assert "assign pdir[p] = !BYP;" in tm and "assign pr[p][b * NSB + k] = s_q[b][k][SD];" in tm
    mxu = src("rtl/mxu/otpu_mxu.sv")
    assert "wire more     = p_act && !c_lz;" in mxu and "(pn || c_lz)" in mxu
    assert not re.search(r"c_left\s*[!=]=\s*0\b", mxu.split("`ifndef SYNTHESIS")[0])
    seq = src("rtl/seq/otpu_seq.sv")
    assert "wire q_en = urdy[u] && !ustart[u];" in seq
    # the MXU's read by bank: TMEM takes a PBR port's banks as they come, the MXU gives registers
    assert "b_en = r_ben[p];" in tm and "b_a[b] = r_bad[p][b][BW +: IW];" in tm
    tq = mxu[mxu.index("if (TQ) begin : g_tq"):mxu.index("end else begin : g_ng")]
    assert "assign t_rben = rb_q;" in tq and "assign t_rbad = rb_a;" in tq
    assert re.search(r"always_ff @\(posedge clk\) if \(!rst && ea_t\) rb_a <= bk_ad\(t_raddr_c, t_ren_c\);", tq)
    assert re.search(r"rb_q <= \(ea_t \? bk_en\(t_raddr_c, t_ren_c\) : rb_v\) &", tq)
    sl = src("rtl/top/otpu_slice.sv")
    assert ".PBR(PBR)" in sl and ".TLANES(LANES)" in sl
    pbr = re.search(r"localparam logic \[NRP-1:0\] PBR = ([^;]*);", sl).group(1)
    assert set(re.findall(r"P_\w+", pbr)) == {"P_MXU", "P_DMA", "P_Q", "P_Q2", "P_VA", "P_VB"}, pbr
    assert re.search(r"r_ben\[p\] = \(.*off\[p\]\) \? '0 : rq_b\[p\];", sl)
    for f, regs in (("rtl/dma/otpu_dma.sv", ["t_rbv <= t_rben_c", "t_rb_q <= adv_nx ? (adv ? t_rben_c : t_rbv) : '0;"]),
                    ("rtl/vpu/otpu_vpu.sv", ["ta_ben <= ta_ben_c", "tb_bad <= tb_bad_c"]),
                    ("rtl/vpu/otpu_quant.sv", ["t_rben <= t_rben_c", "t_rbad2 <= t_rbad2_c"])):
        u = src(f)
        assert "otpu_tmem_breq #(" in u and all(r in u for r in regs), f
    assert '(* keep = "true" *) logic [NRP-1:0][LANES-1:0] rpvb;' in tm
    assert "wire rpb = rlive(i) && rpvb[i][b];" in tm and "assign aa[i][b] = rpb ? rpa[i][b] : qa[i][b];" in tm
    assert "if (!rpvb[p][b]) begin rpe[p][b] <= qe[p][b];" in tm and "if (RPL && rpvb[p][b] && rpe[p][b])" in tm


def test_config_summary_names_the_target(no_env):
    from opentpu.host.board import config_summary
    kw = dict(devname=None, ddr_mts=1066, pair=True, dstep=True, stream=True)
    old = config_summary(Board(FakeTransport(regmap=3, **kw)).info())
    assert old == "D=128 MCOLS=4 LANES=8, register map 3, core 100 MHz, DDR3-1066, build 1234abcd"
    new = config_summary(Board(FakeTransport(regmap=4, **kw)).info())
    assert "register map 4, ypcb-00338 (S=1, 2 channels x 2048 MiB)" in new and "DDR3-1066" in new
    assert T.by_platform(2).name == "bcu1525" and T.by_platform(999) is None
