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

# the targets at WG 8 (weight groups, the channel bridges' adapter knobs): the BCU1525 at one
# slice (targets/bcu1525-wg.toml), at two (bcu1525-s2.toml) and two cards of those (bcu1525-x2)
WG_TARGETS = {"bcu1525-wg", "bcu1525-s2", "bcu1525-x2"}

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
    no_env.setenv("OTPU_TARGET", "bcu1525-wg")                 # (no VOP OUTER, as built)
    assert not board_config().OUTER
    no_env.setenv("OTPU_TARGET", "no-such-board")
    with pytest.raises(T.TargetError, match="no target 'no-such-board'"):
        board_config()


def _hash_without_numerics(cfg) -> int:
    """arch_hash before the MX numerics (what today's production host computes)."""
    import hashlib
    import json
    d = dataclasses.asdict(cfg)
    d.pop("SEQ_WC")                     # (fields that host's Config does not have)
    d.pop("WG")
    d.pop("CHASH")
    d.pop("OUTER")
    d.pop("GALL")
    d.pop("DSTEP2")
    d.pop("SE_OWN")
    d.pop("KT")
    d.pop("AW")
    d.pop("ICACHE")
    d.pop("BHW")
    d.pop("SEXACT")
    d.pop("KVSPLIT")
    d.pop("DNCHUNK")
    d.pop("DNMERGE")
    d.pop("WC_WIN")
    d.pop("ST16")
    d.pop("FSUM")
    d.pop("SPLITQ")
    d.pop("ST16S")
    d.pop("QHL")
    d.pop("QSTHL")
    d.pop("VSIG")
    d.pop("VMAC")
    d.pop("OUTER1")
    d.pop("QONCE")
    d.pop("SPLMM")
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
        if f.name in ("CHASH", "GALL", "SEXACT", "KVSPLIT", "DNCHUNK", "DNMERGE"):    # (only at WG > 1, below; never)
            continue
        v = getattr(card, f.name)
        other = dataclasses.replace(card, **{f.name: (not v) if isinstance(v, bool) else
                                             "e6m9" if isinstance(v, str) else v * 2 if v else 1})
        assert T.arch_hash(other) != T.arch_hash(card), f.name
    # SEXACT (S-exact images) is a compile choice, never in it: the same machine
    assert T.arch_hash(dataclasses.replace(card, S=2, SEXACT=True)) == \
        T.arch_hash(dataclasses.replace(card, S=2))
    # KVSPLIT (the KV sequence split's buckets) neither: a compile choice
    assert T.arch_hash(dataclasses.replace(card, S=4, KVSPLIT=2)) == \
        T.arch_hash(dataclasses.replace(card, S=4))
    # DNCHUNK (the DeltaNet rows by chunks, pf500) neither
    assert T.arch_hash(dataclasses.replace(card, S=4, DNCHUNK=64)) == \
        T.arch_hash(dataclasses.replace(card, S=4))
    # DNMERGE (the DN stage's tile pipeline, stage.py) neither
    assert T.arch_hash(dataclasses.replace(card, S=4, DNMERGE=4)) == \
        T.arch_hash(dataclasses.replace(card, S=4))
    # CHASH and GALL are in it at WG > 1 alone (a machine without groups has no rows to order,
    # no MM to group); OUTER only when off (the pins above have it on: every VPU before the field)
    for k in ("CHASH", "GALL"):
        assert T.arch_hash(dataclasses.replace(card, **{k: True})) == T.arch_hash(card)
        w8 = dataclasses.replace(card, MCOLS=8, WG=8)
        assert T.arch_hash(dataclasses.replace(w8, **{k: True})) != T.arch_hash(w8)


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


# The existing targets' resolved RTL parameters and simulation uarch, as before the native
# adapter's channel parameters (NCH, CB: rtl_params' CB is new, the adapter's default 64 bytes
# on these): the wide channels leave every other target's build as it was
_TM3 = {"TMEM_NSB": 2, "TMEM_RPL": 1, "TMEM_TDP": 1, "TMEM_GST": 1}
_COMMON = {"D": 128, "ACT_BLOCKS": 128, "IMEM_WORDS": 32768, "LANES": 8, "DSTEP": 1,
           "MXU_IMPL": 2, "MXU_CL": 16, "VPU_CL": 2, "ULANES": 8, "RPB": 64, "WPB": 1, "WIN": 16,
           "FIFO_DEPTH": 1024, "PQ_WIN": 1024, "CHASH": 1, "NCH": 2, "CH_LOG2": 31,
           "DRAM_LOG2": 32}
_RESOLVED = {
    "ypcb-00338": _TM3 | _COMMON | {
        "TMEM_PMP": 1, "S": 1, "MCOLS": 4, "ACT_ROWS": 16, "TMEM_WORDS": 65536, "MXU_E": 4,
        "TMEM_IMPL": 3, "TRACE_DEPTH": 16384, "SEQ_WC": 1, "DDR_MTS": 1066, "PLATFORM": 1,
        "ARCH_HASH": 0x4ffd1e17},
    "ypcb-00338:m8": _TM3 | _COMMON | {
        "TMEM_PMP": 1, "S": 1, "MCOLS": 8, "ACT_ROWS": 16, "TMEM_WORDS": 131072, "MXU_E": 8,
        "TMEM_IMPL": 3, "TRACE_DEPTH": 16384, "SEQ_WC": 1, "DDR_MTS": 1066, "PLATFORM": 1,
        "ARCH_HASH": 0x46b96e6d},
    "bcu1525": _COMMON | {
        "S": 2, "MCOLS": 4, "ACT_ROWS": 4, "TMEM_WORDS": 65536, "MXU_E": 0, "TMEM_IMPL": 0,
        "TRACE_DEPTH": 16384, "SEQ_WC": 0, "DDR_MTS": 1200, "PLATFORM": 2,
        "ARCH_HASH": 3323358210},
    "nangate45": _TM3 | _COMMON | {
        "TMEM_PMP": 0, "S": 1, "MCOLS": 4, "ACT_ROWS": 8, "TMEM_WORDS": 65536, "MXU_E": 4,
        "TMEM_IMPL": 3, "TRACE_DEPTH": 0, "SEQ_WC": 1, "DDR_MTS": 0, "PLATFORM": 256,
        "ARCH_HASH": 715574571},
    "nangate45-small": {
        "S": 1, "D": 32, "MCOLS": 2, "ACT_ROWS": 2, "ACT_BLOCKS": 32, "TMEM_WORDS": 16384,
        "IMEM_WORDS": 8192, "LANES": 4, "DSTEP": 0, "MXU_IMPL": 2, "MXU_CL": 16, "MXU_E": 0,
        "VPU_CL": 1, "ULANES": 4, "TMEM_IMPL": 1, "RPB": 1, "WPB": 1, "WIN": 16,
        "FIFO_DEPTH": 64, "TRACE_DEPTH": 0, "PQ_WIN": 1024, "CHASH": 0, "SEQ_WC": 0,
        "DDR_MTS": 0, "NCH": 1, "CH_LOG2": 28, "DRAM_LOG2": 28, "PLATFORM": 257,
        "ARCH_HASH": 307112127},
}
_SIM = {"WIN": 16, "RPB": 128, "WPB": 1, "FIFO_DEPTH": 1024}
_SIM_RESOLVED = {
    "ypcb-00338": _SIM | {"TMEM_IMPL": 3} | _TM3 | {"TMEM_PMP": 1, "MXU_E": 4, "SEQ_WC": 1},
    "ypcb-00338:m8": _SIM | {"TMEM_IMPL": 3} | _TM3 | {"TMEM_PMP": 1, "MXU_E": 8, "SEQ_WC": 1},
    "bcu1525": _SIM | {"COLL_HUB": 1},     # two slices: otpu_board's collective (its hub)
    "nangate45": _SIM | {"TMEM_IMPL": 3} | _TM3 | {"TMEM_PMP": 0, "MXU_E": 4, "SEQ_WC": 1},
    "nangate45-small": {"WIN": 16, "RPB": 1, "WPB": 1, "FIFO_DEPTH": 64, "TMEM_IMPL": 1},
}


def test_existing_targets_resolve_unchanged():
    for n, want in _RESOLVED.items():
        p = T.rtl_params(n)
        assert p.pop("CB") == (16 if n == "nangate45-small" else 64), n
        assert p == want, n
        assert T.sim_uarch(n) == _SIM_RESOLVED[n], n
        assert T.mem_model(n) == [], n


def test_wg_target():
    """targets/bcu1525-x2.toml: 4 slices (two cards of two), each on 8 channels of 2 GiB in
    128-byte beats (otpu_native_dram NCH 8, CB 128: CH_LOG2 31, a slice's 16 GiB); the
    simulation tops take the channels (sim_uarch), the channel model its rate, latency and
    refresh in core cycles (mem_model: 4.56 GB/s a channel is 34.2 bytes a cycle at 133.33 MHz;
    250 ns is 33 cycles at 133.33); 8 MXU columns and the card m8's 128K TMEM; weight groups of
    8 rows (1 KiB on port W a step) with a 2048-deep chunk FIFO; RSCALE's row factor read once a
    row and a buffered QACT pass 1 at two chunks a cycle (QUANT_R1, QUANT_PW: the cards at WG 1
    leave them out); the MXU's command queue 4 deep (MXU_QD; the WG 1 cards' 2, left out)."""
    t = T.load("bcu1525-x2")
    p = T.rtl_params(t)
    assert (p["S"], p["D"], p["MCOLS"], p["NCH"], p["CB"], p["CH_LOG2"], p["DRAM_LOG2"],
            p["PLATFORM"]) == (4, 128, 8, 8, 128, 31, 34, 7)
    assert (p["MXU_E"], p["TMEM_WORDS"], p["FIFO_DEPTH"], p["WG"]) == (8, 1 << 17, 2048, 8)
    assert t.kind == "fpga" and t.family == "xilinxus" and t.platform["mem"] == "ddr4"
    c = T.config(t)
    assert (c.S, c.DRAM_BYTES, c.PAIR, c.DSTEP, c.STREAM, c.LANES, c.WG) == (4, 1 << 34, False,
                                                                             True, True, 8, 8)
    # every MM in groups (GALL: the MXU at WG > 1), in the hash; column reuse at WG 1 only
    # (otpu_board's CAPS bit5), and at WG 1 neither
    assert c.GALL and not any(T.config(n).GALL for n in T.names() if n not in WG_TARGETS)
    assert T.arch_hash(dataclasses.replace(c, GALL=False)) != p["ARCH_HASH"]
    w1 = T.build_config(t, WG=1)
    assert not w1.GALL and w1.PAIR and not T.config(t, WG=1).GALL
    assert T.arch_hash(w1) == T.arch_hash(dataclasses.replace(w1, GALL=True))
    # the group chunks' rows in the channel hash's order (docs/isa.md "Weight groups"); a build
    # without groups has none to order, and every target without them leaves it off
    assert c.CHASH and not T.config(t, WG=1).CHASH and not T.build_config(t, WG=1).CHASH
    assert p["ARCH_HASH"] == T.arch_hash(c) != T.arch_hash(dataclasses.replace(c, CHASH=False))
    # a build without the hash (otpu_board CHASH 0: make_wg.py --set CHASH=0) reports its hash
    assert T.build_config(t, CHASH=0) == dataclasses.replace(c, CHASH=False)
    assert T.build_config(t, CHASH=1) == c and not T.build_config(t, CHASH=1, WG=1).CHASH
    assert not any(T.config(n).CHASH for n in T.names() if n not in WG_TARGETS)
    # no VOP OUTER (its DeltaNet runs DSTEP): otpu_board OUTER 0 and the field in the hash; every
    # other target's VPU has it (no OUTER parameter, the hash without the field)
    assert not c.OUTER and p["OUTER"] == 0
    assert T.arch_hash(dataclasses.replace(c, OUTER=True)) != p["ARCH_HASH"]
    assert not T.build_config(t).OUTER and T.build_config(t, OUTER=1).OUTER
    assert all(T.config(n).OUTER and "OUTER" not in T.rtl_params(n) for n in T.names()
               if n not in WG_TARGETS)
    u = T.sim_uarch(t)
    assert (u["NCH"], u["CB"], u["TMEM_IMPL"], u["QUANT_R1"], p["QUANT_R1"]) == (8, 128, 3, 1, 1)
    assert (u["QUANT_PW"], p["QUANT_PW"]) == (2, 2)
    assert (u["MXU_QD"], p["MXU_QD"]) == (4, 4)
    for k in ("QUANT_R1", "QUANT_PW", "MXU_QD"):
        assert k not in T.rtl_params("ypcb-00338") and k not in T.sim_uarch("ypcb-00338")
    assert T.mem_model(t, 133.33) == ["+nat_bpc=34201", "+axi_lat=33", "+nat_refi=1040",
                                      "+nat_rfc=47"]
    assert T.mem_model(t)[:2] == ["+nat_bpc=45600", "+axi_lat=25"]        # its 100 MHz
    assert T.by_platform(7) is not None and T.by_platform(7).name == "bcu1525-x2"
    assert T.by_platform(4) is None


def test_native_channels_are_checked(tmp_path):
    """An FPGA target's channels are the native adapter's: a power of two of at least 2, beats
    of 64 or 128 bytes (a D chunk whole beats, a beat on every channel whole chunks), and the
    slice's DRAM exactly its channels' bytes; with weight groups a group step is a D-byte beat on
    every channel. (The WG targets' NAT_XPOST off here, and its half beats: they need beat_bytes = D and
    128, checked first.)"""
    good = (ROOT / "targets/bcu1525-wg.toml").read_text().replace(
        "NAT_XPOST = true", "NAT_XPOST = false", 1).replace("NAT_RHB = true", "NAT_RHB = false", 1) \

    p = tmp_path / "bcu1525-wg.toml"
    for edit, msg in ((("channel_bytes = 4294967296", "channel_bytes = 2147483648"),
                       "8 channels x 2147483648 bytes is not the slice's DRAM_BYTES"),
                      (("beat_bytes = 128", "beat_bytes = 32"), "beats of\n?.*64 or 128"),
                      (("channels_per_slice = 8", "channels_per_slice = 6"), "a power of two")):
        T.load.cache_clear()
        p.write_text(good.replace(*edit))
        with pytest.raises(T.TargetError, match=msg):
            T.load(str(p))
    # weight groups: a step one beat on every channel, a row a beat (port W's lane c column c's:
    # 16 channels of 64 bytes are WG x D but not a row a beat)
    T.load.cache_clear()
    p.write_text(good.replace("channels_per_slice = 8\nchannel_bytes = 4294967296",
                              "channels_per_slice = 16\nchannel_bytes = 2147483648")
                 .replace("beat_bytes = 128\n", "beat_bytes = 64\n"))
    with pytest.raises(T.TargetError, match="not one beat of D bytes"):
        T.load(str(p))
    T.load.cache_clear()


def test_native_depths_from_the_target(tmp_path):
    """otpu_native_dram's per-channel depths (RD, AD, XD, WQD), A's read runs (APF) and the SW
    queue's fill reads (SW_FILL: NAT_FILL) are the target's [uarch] NAT_* where it states them
    (rtl_params, sim_uarch: otpu_board / otpu_top NAT_*), else the adapter's defaults, which
    otpu_board's, otpu_top's and the testbenches' defaults repeat; the cards state none."""
    adp = (ROOT / "rtl/mem/otpu_native_dram.sv").read_text()
    want = {k: re.search(rf"parameter int {k} = ([^,]+),", adp).group(1).strip()
            for k in ("RD", "AD", "XD", "WQD", "APF")}
    for f in ("rtl/shell/otpu_board.sv", "rtl/top/otpu_top.sv", "sim/verilator/tb_board.sv",
              "sim/verilator/tb_top.sv", "sim/verilator/tb_ndram.sv"):
        txt = (ROOT / f).read_text()
        for k, v in want.items():
            n = k if f.endswith("tb_ndram.sv") else f"NAT_{k}"
            assert re.search(rf"parameter int {n}\s*=\s*{re.escape(v)}\s*[,;\n]", txt), (f, k)
        n = "SW_FILL" if f.endswith("tb_ndram.sv") else "NAT_FILL"
        assert re.search(rf"parameter bit {n}\s*=\s*1'b1\s*[,;]", txt), f
    assert re.search(r"parameter bit SW_FILL = 1'b1,", adp)
    for n in T.names():
        if n not in WG_TARGETS:
            assert not set(T.NAT_KEYS) & (set(T.rtl_params(n)) | set(T.sim_uarch(n))), n
    p = tmp_path / "bcu1525-wg.toml"
    p.write_text((ROOT / "targets/bcu1525-wg.toml").read_text().replace(
        "CHASH = true", "CHASH = true\nNAT_XD = 128", 1))
    T.load.cache_clear()
    t = T.load(str(p))
    assert (T.rtl_params(t)["NAT_RD"], T.sim_uarch(t)["NAT_XD"]) == (16, 128)
    assert (T.rtl_params(t)["NAT_FILL"], T.sim_uarch(t)["NAT_FILL"]) == (0, 0)
    assert "NAT_WQD" not in T.rtl_params(t)
    # the WG targets' own: A's read runs of 8 beats, its FIFO 16 deep (>= 2 APF - 1), B's reads 16 a
    # channel, partial SW beats as masked writes (the memory's data mask), B's
    # and SW's write data posted, one SW gather beat for every channel, port W's beats posted into
    # the MXUs' chunk FIFOs, 2 register stages each way between a slice and its adapter and 2 for
    # the posts (otpu_xstg), the channels' data in half beats (R1 / R2)
    assert {k: v for k, v in T.rtl_params("bcu1525-wg").items() if k in T.NAT_KEYS} == \
        {"NAT_APF": 8, "NAT_AD": 16, "NAT_RD": 16, "NAT_FILL": 0, "NAT_WPOST": 1, "NAT_SWG": 1,
         "NAT_XPOST": 1, "NAT_XSTG": 2, "NAT_XSTG_P": 2, "NAT_RHB": 1}
    T.load.cache_clear()


def test_posted_write_data_from_the_target(tmp_path):
    """The write data posted at the take (otpu_native_dram WPOST: the WG targets' bridges take a channel's
    data ahead of its commands) is the target's [uarch] NAT_WPOST: off by default in the adapter,
    otpu_board, otpu_top and the testbenches (tb_ndram's WPOST), stated by no card, on for the WG targets;
    a target that asks for it with NAT_FILL on (a fill read's data merged as its write goes out)
    is refused."""
    adp = (ROOT / "rtl/mem/otpu_native_dram.sv").read_text()
    assert re.search(r"parameter bit WPOST = 1'b0,", adp)
    for f in ("rtl/shell/otpu_board.sv", "rtl/top/otpu_top.sv", "sim/verilator/tb_board.sv",
              "sim/verilator/tb_top.sv", "sim/verilator/tb_ndram.sv"):
        n = "WPOST" if f.endswith("tb_ndram.sv") else "NAT_WPOST"
        assert re.search(rf"parameter bit {n}\s*=\s*1'b0\s*[,;]", (ROOT / f).read_text()), f
    for n in T.names():
        if n not in WG_TARGETS:
            assert "NAT_WPOST" not in T.rtl_params(n), n
    assert (T.rtl_params("bcu1525-wg")["NAT_WPOST"], T.sim_uarch("bcu1525-wg")["NAT_WPOST"]) == (1, 1)
    p = tmp_path / "bcu1525-wg.toml"
    p.write_text((ROOT / "targets/bcu1525-wg.toml").read_text()
                 .replace("NAT_FILL = false", "NAT_FILL = true", 1))
    T.load.cache_clear()
    with pytest.raises(T.TargetError, match="NAT_WPOST needs NAT_FILL false"):
        T.load(str(p))
    T.load.cache_clear()


def test_one_gather_beat_from_the_target(tmp_path):
    """One SW gather beat and one posted beat a cycle for every channel (otpu_native_dram SWG,
    the WG targets' area cut) is the target's [uarch] NAT_SWG: off by default in the adapter, otpu_board,
    otpu_top and the testbenches (tb_ndram's SWG), stated by the WG targets alone; a target that asks
    for it without NAT_WPOST (its one bus is the posted beat's) is refused."""
    adp = (ROOT / "rtl/mem/otpu_native_dram.sv").read_text()
    assert re.search(r"parameter bit SWG = 1'b0,", adp)
    for f in ("rtl/shell/otpu_board.sv", "rtl/top/otpu_top.sv", "sim/verilator/tb_board.sv",
              "sim/verilator/tb_top.sv", "sim/verilator/tb_ndram.sv"):
        n = "SWG" if f.endswith("tb_ndram.sv") else "NAT_SWG"
        assert re.search(rf"parameter bit {n}\s*=\s*1'b0\s*[,;]", (ROOT / f).read_text()), f
    for n in T.names():
        if n not in WG_TARGETS:
            assert "NAT_SWG" not in T.rtl_params(n), n
    assert (T.rtl_params("bcu1525-wg")["NAT_SWG"], T.sim_uarch("bcu1525-wg")["NAT_SWG"]) == (1, 1)
    src = (ROOT / "targets/bcu1525-wg.toml").read_text()
    p = tmp_path / "bcu1525-wg.toml"
    p.write_text(src.replace("NAT_WPOST = true", "NAT_WPOST = false", 1))
    T.load.cache_clear()
    with pytest.raises(T.TargetError, match="NAT_SWG needs NAT_WPOST"):
        T.load(str(p))
    T.load.cache_clear()


def test_half_beats_from_the_target(tmp_path):
    """The channels' read and write data in half beats (otpu_native_dram and otpu_hbm_ch RHB /
    WHB, the WG targets' R1 / R2: docs/native_channels.md) are the target's [uarch] NAT_RHB / NAT_WHB: off
    by default in the adapter, the bridge, otpu_board, otpu_top, the native memory model and the
    testbenches, so the cards build cycle for cycle as before; the WG targets alone state NAT_RHB
    (rtl_params, sim_uarch, the wrapper's OTPU_RHB), the BCU1525's and the YPCB-00338's targets do
    not, and no target NAT_WHB (a 27B decode measured 3.3% slower with it: the posted bus a beat every two
    cycles). They are not in the Config: the WG targets' arch_hash is the same with WHB on and RHB off. A
    target that asks for NAT_RHB without NAT_XPOST (W's beats would come back on n_rdata) or
    NAT_WHB without NAT_SWG (its halves go on the one posted bus) is refused."""
    for f, ks in (("rtl/mem/otpu_native_dram.sv", ("RHB", "WHB")), ("rtl/mem/otpu_hbm_ch.sv", ("RHB", "WHB")),
                  ("sim/verilator/otpu_native_mem.sv", ("RHB", "WHB")),
                  ("sim/verilator/tb_ndram.sv", ("RHB", "WHB")), ("sim/verilator/tb_hbmch.sv", ("RHB", "WHB")),
                  ("rtl/shell/otpu_board.sv", ("NAT_RHB", "NAT_WHB")), ("rtl/top/otpu_top.sv", ("NAT_RHB", "NAT_WHB")),
                  ("sim/verilator/tb_board.sv", ("NAT_RHB", "NAT_WHB")), ("sim/verilator/tb_top.sv", ("NAT_RHB", "NAT_WHB"))):
        txt = (ROOT / f).read_text()
        for k in ks:
            assert re.search(rf"parameter bit {k}\s*=\s*1'b0\s*([,;]|//[^\n]*\n\s*\))", txt), (f, k)
    wg = (ROOT / "rtl/boards/bcu1525/otpu_bcu_wg_sys.sv").read_text()
    assert "`define OTPU_RHB 0" in wg and "`define OTPU_WHB 0" in wg
    for n in T.names():
        assert "NAT_WHB" not in set(T.rtl_params(n)) | set(T.sim_uarch(n)), n
        if n not in WG_TARGETS:
            assert "NAT_RHB" not in set(T.rtl_params(n)) | set(T.sim_uarch(n)), n
    assert {"bcu1525", "ypcb-00338"} <= set(T.names())
    t = T.load("bcu1525-wg")
    assert (T.rtl_params(t)["NAT_RHB"], T.sim_uarch(t)["NAT_RHB"]) == (1, 1)
    src = (ROOT / "targets/bcu1525-wg.toml").read_text()
    assert "NAT_RHB = true" in src and "\nNAT_WHB" not in src
    p = tmp_path / "bcu1525-wg.toml"
    whb = src.replace("NAT_RHB = true", "NAT_WHB = true\nNAT_RHB = true", 1)
    p.write_text(whb.replace("NAT_RHB = true", "NAT_RHB = false", 1))
    T.load.cache_clear()
    other = T.load(str(p))
    assert T.rtl_params(other)["NAT_WHB"] == 1 and T.rtl_params(other)["NAT_RHB"] == 0
    assert T.arch_hash(T.config(other)) == T.arch_hash(T.config(t))
    assert T.rtl_params(other)["ARCH_HASH"] == T.rtl_params(t)["ARCH_HASH"]
    p.write_text(src.replace("NAT_XPOST = true", "NAT_XPOST = false", 1))
    T.load.cache_clear()
    with pytest.raises(T.TargetError, match="NAT_RHB .* needs NAT_XPOST"):
        T.load(str(p))
    p.write_text(whb.replace("NAT_SWG = true", "NAT_SWG = false", 1))
    T.load.cache_clear()
    with pytest.raises(T.TargetError, match="NAT_WHB .* needs NAT_WPOST and NAT_SWG"):
        T.load(str(p))
    T.load.cache_clear()


def test_result_fifo_rows_from_the_target(tmp_path):
    """The MXU's result FIFO rows (otpu_mxu RF_ROWS, an area cut) are the target's [uarch]
    MXU_RF_ROWS: 0 (its rule) by default in otpu_slice, otpu_board, otpu_top and the testbenches,
    stated by no target now (a target's value reaches rtl_params, sim_uarch and the board model's build); a value that is not a
    power of two of at least 16 is refused."""
    for f in ("rtl/top/otpu_slice.sv", "rtl/shell/otpu_board.sv", "rtl/top/otpu_top.sv",
              "sim/verilator/tb_board.sv", "sim/verilator/tb_top.sv"):
        assert re.search(r"parameter int MXU_RF_ROWS\s*=\s*0\s*[,;]", (ROOT / f).read_text()), f
    for n in T.names():
        assert "MXU_RF_ROWS" not in T.rtl_params(n) and "MXU_RF_ROWS" not in T.sim_uarch(n), n
    src = (ROOT / "targets/bcu1525-wg.toml").read_text()
    assert "MXU_QD = 4\n" in src
    p = tmp_path / "bcu1525-wg.toml"
    p.write_text(src.replace("MXU_QD = 4\n", "MXU_QD = 4\nMXU_RF_ROWS = 32\n", 1))
    T.load.cache_clear()
    t = T.load(str(p))
    assert (T.rtl_params(t)["MXU_RF_ROWS"], T.sim_uarch(t)["MXU_RF_ROWS"]) == (32, 32)
    for bad in (24, 8):
        p.write_text(src.replace("MXU_QD = 4\n", f"MXU_QD = 4\nMXU_RF_ROWS = {bad}\n", 1))
        T.load.cache_clear()
        with pytest.raises(T.TargetError, match=f"MXU_RF_ROWS {bad}: 0, or a power of two"):
            T.load(str(p))
    T.load.cache_clear()


def test_posted_w_beats_from_the_target(tmp_path):
    """Port W's beats posted into the MXU's chunk FIFO by the channels (otpu_native_dram /
    otpu_mxu XPOST, (a) v2: docs/native_channels.md) are the target's [uarch] NAT_XPOST: off by
    default in otpu_board, otpu_top and the testbenches, stated by the WG targets alone (their
    bridges post them). A target that states it away from an FPGA at WG > 1 is
    refused, and one at WG > 1 with PAIR on (the MXU runs every MM in groups there, GALL);
    otpu_board reports column reuse (CAPS bit5) at WG 1 only."""
    for f in ("rtl/shell/otpu_board.sv", "rtl/top/otpu_top.sv", "sim/verilator/tb_board.sv",
              "sim/verilator/tb_top.sv"):
        assert re.search(r"parameter bit NAT_XPOST\s*=\s*1'b0\s*[,;]", (ROOT / f).read_text()), f
    assert ".PAIR(WG == 1)" in (ROOT / "rtl/shell/otpu_board.sv").read_text()
    for n in T.names():
        assert ("NAT_XPOST" in T.rtl_params(n)) == (n in WG_TARGETS), n
    t = T.load("bcu1525-wg")
    assert (T.rtl_params(t)["NAT_XPOST"], T.sim_uarch(t)["NAT_XPOST"]) == (1, 1)
    assert not T.config(t).PAIR
    src = (ROOT / "targets/bcu1525-wg.toml").read_text()
    p = tmp_path / "bcu1525-wg.toml"
    p.write_text(src.replace("beat_bytes = 128", "beat_bytes = 256", 1))
    T.load.cache_clear()
    with pytest.raises(T.TargetError, match="NAT_XPOST"):
        T.load(str(p))
    p.write_text(src.replace("PAIR = false", "PAIR = true", 1))
    T.load.cache_clear()
    with pytest.raises(T.TargetError, match="WG > 1 needs PAIR false"):
        T.load(str(p))
    T.load.cache_clear()


def test_tmem_style_from_the_target(tmp_path):
    """The TMEM's RAM style is the target's [uarch] TMEM_STYLE (rtl_params: the string, a flow
    quotes it; sim_uarch: a Verilator -G string literal): the WG targets' pumped one copy in UltraRAM
    (its block RAM is the board's), absent elsewhere: otpu_tmem's "block", which otpu_slice,
    otpu_board, otpu_top and the testbenches repeat. Not "block" or "ultra": refused."""
    assert re.search(r'parameter string STYLE = "block"',
                     (ROOT / "rtl/mem/otpu_tmem.sv").read_text())
    for f in ("rtl/top/otpu_slice.sv", "rtl/shell/otpu_board.sv", "rtl/top/otpu_top.sv",
              "sim/verilator/tb_board.sv", "sim/verilator/tb_top.sv"):
        assert re.search(r'parameter string TMEM_STYLE = "block"\s*[,;]',
                         (ROOT / f).read_text()), f
    assert (T.rtl_params("bcu1525-wg")["TMEM_STYLE"], T.sim_uarch("bcu1525-wg")["TMEM_STYLE"]) == \
        ("ultra", '"ultra"')
    for n in T.names():
        if n not in WG_TARGETS:
            assert "TMEM_STYLE" not in T.rtl_params(n) and "TMEM_STYLE" not in T.sim_uarch(n), n
    p = tmp_path / "bcu1525-wg.toml"
    p.write_text((ROOT / "targets/bcu1525-wg.toml").read_text().replace(
        'TMEM_STYLE = "ultra"', 'TMEM_STYLE = "lutram"', 1))
    T.load.cache_clear()
    with pytest.raises(T.TargetError, match="TMEM_STYLE"):
        T.load(str(p))
    T.load.cache_clear()


def test_imem_style_and_act_ucols_from_the_target(tmp_path):
    """IMEM's RAMs' style (IMEM_STYLE, a string as TMEM_STYLE; absent: otpu_seq's "auto"), its
    wait entries' (IMEM_WC_STYLE; absent: IMEM_STYLE's, otpu_seq WSTYLE "") and the ACT RAM's
    columns in UltraRAM (ACT_UCOLS, 0 .. MCOLS; absent or 0: none, no parameter): the WG targets' IMEM
    and ACT RAM in block RAM, the wait entries in UltraRAM (the chunk FIFO's lanes take the rest,
    FIFO_DEPTH 2048), absent elsewhere. Every top repeats otpu_seq's and otpu_actram's defaults."""
    seq = (ROOT / "rtl/seq/otpu_seq.sv").read_text()
    assert re.search(r'parameter string STYLE\s*= "auto"', seq)
    assert re.search(r'parameter string WSTYLE\s*= ""', seq) and ".STYLE(WST)" in seq
    assert re.search(r"parameter int UCOLS\s*= 0\b", (ROOT / "rtl/mem/otpu_actram.sv").read_text())
    for f in ("rtl/top/otpu_slice.sv", "rtl/shell/otpu_board.sv", "rtl/top/otpu_top.sv",
              "sim/verilator/tb_board.sv", "sim/verilator/tb_top.sv"):
        src = (ROOT / f).read_text()
        assert re.search(r'parameter string IMEM_STYLE = "auto"\s*[,;]', src), f
        assert re.search(r'parameter string IMEM_WC_STYLE = ""\s*[,;]', src), f
        assert re.search(r"parameter int ACT_UCOLS\s*= 0\s*[,;]", src), f
        assert ".IMEM_WC_STYLE(IMEM_WC_STYLE)" in src or ".WSTYLE(IMEM_WC_STYLE)" in src, f
    p, u = T.rtl_params("bcu1525-wg"), T.sim_uarch("bcu1525-wg")
    assert (p["IMEM_STYLE"], u["IMEM_STYLE"], p["IMEM_WC_STYLE"], u["IMEM_WC_STYLE"]) == \
        ("block", '"block"', "ultra", '"ultra"')
    assert "ACT_UCOLS" not in p and "ACT_UCOLS" not in u      # (0: every column in block RAM)
    for n in T.names():
        if n not in WG_TARGETS:
            for k in ("IMEM_STYLE", "IMEM_WC_STYLE", "ACT_UCOLS"):
                assert k not in T.rtl_params(n) and k not in T.sim_uarch(n), (n, k)
    p = tmp_path / "bcu1525-wg.toml"
    for old, new, err in (('IMEM_STYLE = "block"', 'IMEM_STYLE = "lutram"', "IMEM_STYLE"),
                          ('IMEM_WC_STYLE = "ultra"', 'IMEM_WC_STYLE = "auto"', "IMEM_WC_STYLE"),
                          ("ACT_UCOLS = 0", "ACT_UCOLS = 9", "ACT_UCOLS")):
        p.write_text((ROOT / "targets/bcu1525-wg.toml").read_text().replace(old, new, 1))
        T.load.cache_clear()
        with pytest.raises(T.TargetError, match=err):
            T.load(str(p))
    T.load.cache_clear()


def test_mxu_result_fifo_style_from_the_target(tmp_path):
    """The MXU's result FIFO's copies (MXU_RF_STYLE, a string as TMEM_STYLE; absent: otpu_mxu's
    "block", read ahead): the WG targets' "distributed", LUT RAM read at its address (its LUTs bind; read
    ahead the FIFO took 28.5 block RAM tiles and 312 LUTs more a slice), absent elsewhere; a style
    otpu_mxu does not build is an error. Every top repeats otpu_mxu's default and passes it on,
    and otpu_mxu builds each style in a branch of its own."""
    mxu = (ROOT / "rtl/mxu/otpu_mxu.sv").read_text()
    assert re.search(r'parameter string RF_STYLE = "block"\s*[,;]', mxu)
    rfm = mxu[mxu.index("begin : g_rfm"):]
    rfm = rfm[:rfm.index("assign rq[c]")]
    lut, bram = rfm[rfm.index("begin : g_lut"):], rfm[rfm.index("begin : g_bram"):]
    assert 'if (RF_STYLE == "distributed") begin : g_lut' in rfm
    assert "assign rqm = m[rf_qc[c]];" in lut[:lut.index("end else")] and "otpu_ram_1r1w_ra" in bram
    assert ".RF_STYLE(MXU_RF_STYLE)" in (ROOT / "rtl/top/otpu_slice.sv").read_text()
    for f in ("rtl/top/otpu_slice.sv", "rtl/shell/otpu_board.sv", "rtl/top/otpu_top.sv",
              "sim/verilator/tb_board.sv", "sim/verilator/tb_top.sv"):
        src = (ROOT / f).read_text()
        assert re.search(r'parameter string MXU_RF_STYLE = "block"\s*[,;]', src), f
        assert f.endswith("otpu_slice.sv") or ".MXU_RF_STYLE(MXU_RF_STYLE)" in src, f
    p, u = T.rtl_params("bcu1525-wg"), T.sim_uarch("bcu1525-wg")
    assert (p["MXU_RF_STYLE"], u["MXU_RF_STYLE"]) == ("distributed", '"distributed"')
    for n in T.names():
        if n not in WG_TARGETS:
            assert "MXU_RF_STYLE" not in T.rtl_params(n) and "MXU_RF_STYLE" not in T.sim_uarch(n), n
    p = tmp_path / "bcu1525-wg.toml"
    p.write_text((ROOT / "targets/bcu1525-wg.toml").read_text().replace(
        'MXU_RF_STYLE = "distributed"', 'MXU_RF_STYLE = "ultra"', 1))
    T.load.cache_clear()
    with pytest.raises(T.TargetError, match="MXU_RF_STYLE"):
        T.load(str(p))
    T.load.cache_clear()


def test_mxu_chunk_fifo_style_from_the_target(tmp_path):
    """The MXU's chunk FIFO lanes (MXU_FIFO_STYLE, a string as TMEM_STYLE; absent: otpu_mxu's
    "block"): the WG targets' "ultra", UltraRAM in 72-bit column groups, at NAT_XPOST each lane's posted
    beats staged in LUT RAM across the clocks (g_xp's g_u, otpu_mxu_stage and its scoped
    constraints; option A's block RAMs in g_a), absent
    elsewhere; a style otpu_mxu does not build is an error. Every top repeats otpu_mxu's default
    and passes it on; the board model takes it (SimTransport)."""
    mxu = (ROOT / "rtl/mxu/otpu_mxu.sv").read_text()
    assert re.search(r'parameter string FIFO_STYLE = "block"\s*[,;]', mxu)
    assert 'localparam bit FU = FIFO_STYLE == "ultra";' in mxu
    assert "localparam int FB = FU ? 72 : 36;" in mxu
    xp = mxu[mxu.index("begin : g_xp"):mxu.index("begin : g_nfdw")]
    assert "if (FU) begin : g_u" in xp and "end else begin : g_a" in xp
    assert "otpu_mxu_stage #(.QW(QW), .N(NS)) u_sg" in xp and '.STYLE("ultra")) u_fd' in xp
    stage = mxu[mxu.index("module otpu_mxu_stage"):]
    assert '.STYLE("distributed")) u_q' in stage[:stage.index("endmodule")]
    # the staging's crossing: scoped to otpu_mxu_stage, read where a build has it
    assert (ROOT / "boards/bcu1525/constraints/otpu_mxu_stage.tcl").exists()
    assert "set_property SCOPED_TO_REF otpu_mxu_stage [get_files $cons/otpu_mxu_stage.tcl]" in \
        (ROOT / "boards/bcu1525/vivado/create_project_wg.tcl").read_text()
    assert ".FIFO_STYLE(MXU_FIFO_STYLE)" in (ROOT / "rtl/top/otpu_slice.sv").read_text()
    for f in ("rtl/top/otpu_slice.sv", "rtl/shell/otpu_board.sv", "rtl/top/otpu_top.sv",
              "sim/verilator/tb_board.sv", "sim/verilator/tb_top.sv"):
        src = (ROOT / f).read_text()
        assert re.search(r'parameter string MXU_FIFO_STYLE = "block"\s*[,;]', src), f
        assert f.endswith("otpu_slice.sv") or ".MXU_FIFO_STYLE(MXU_FIFO_STYLE)" in src, f
    assert '"MXU_FIFO_STYLE"' in (ROOT / "opentpu/host/board.py").read_text()
    p, u = T.rtl_params("bcu1525-wg"), T.sim_uarch("bcu1525-wg")
    assert (p["MXU_FIFO_STYLE"], u["MXU_FIFO_STYLE"]) == ("ultra", '"ultra"')
    for n in T.names():
        if n not in WG_TARGETS:
            assert "MXU_FIFO_STYLE" not in T.rtl_params(n), n
            assert "MXU_FIFO_STYLE" not in T.sim_uarch(n), n
    p = tmp_path / "bcu1525-wg.toml"
    p.write_text((ROOT / "targets/bcu1525-wg.toml").read_text().replace(
        'MXU_FIFO_STYLE = "ultra"', 'MXU_FIFO_STYLE = "distributed"', 1))
    T.load.cache_clear()
    with pytest.raises(T.TargetError, match="MXU_FIFO_STYLE"):
        T.load(str(p))
    T.load.cache_clear()


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
    # otpu_seq's wait entries (SEQ_WC) count a unit's last 15 instructions: a window of 16 at
    # most, or 32 (the counts saturate: Config.WC_WIN)
    p = tmp_path / "nangate45-small.toml"
    wc = good.replace("SEQ_WC = false", "SEQ_WC = true")
    p.write_text(wc)
    assert T.load(str(p)).arch["SEQ_WC"] and T.sim_uarch(T.load(str(p)))["SEQ_WC"] == 1
    assert T.config(T.load(str(p))).WC_WIN == 16
    T.load.cache_clear()
    p.write_text(wc.replace("WIN = 16", "WIN = 32"))
    assert T.config(T.load(str(p))).WC_WIN == 32
    T.load.cache_clear()
    p.write_text(wc.replace("WIN = 16", "WIN = 24"))
    with pytest.raises(T.TargetError, match="SEQ_WC .* needs WIN <= 16, or 32"):
        T.load(str(p))
    # the MXU's command queue: 2 .. 7 deep (otpu_mxu QD: q_n's 3 bits)
    for qd in (1, 8):
        T.load.cache_clear()
        p.write_text(good.replace("WIN = 16", f"WIN = 16\nMXU_QD = {qd}"))
        with pytest.raises(T.TargetError, match=f"MXU_QD {qd}: 2..7"):
            T.load(str(p))


def test_the_entries_window(monkeypatch):
    """Config.WC_WIN: 32 on a WAITC target at WIN 32 (bcu1525-wg's bhw variant, bcu1525-s2),
    else 16; arch_hash holds it above 16 only, so a window-16 target keeps its CFG_HASH (the
    production card's bcu1525-wg) and its programs' entries (opentpu.waits: the same words)."""
    wg, bhw, s2 = T.config("bcu1525-wg"), T.config("bcu1525-wg:bhw"), T.config("bcu1525-s2")
    assert (wg.WC_WIN, bhw.WC_WIN, s2.WC_WIN) == (16, 32, 32)
    assert T.arch_hash(wg) == 0x4B35EC47 and T.arch_hash(T.config("bcu1525-wg:kt")) == 0xD49B52F5
    assert (bhw.BHW, bhw.KT, s2.BHW, s2.KT) == (33, True, 33, True)   # (one S 1 bitstream, as S 2)
    assert T.load("bcu1525-wg:bhw").uarch["WIN"] == T.load("bcu1525-s2").uarch["WIN"] == 32
    assert T.arch_hash(dataclasses.replace(wg, WC_WIN=16)) == T.arch_hash(wg)
    assert T.arch_hash(dataclasses.replace(bhw, WC_WIN=16)) != T.arch_hash(bhw)
    assert T.build_config("bcu1525-wg:bhw").WC_WIN == 32
    assert T.build_config("bcu1525-wg:bhw", SEQ_WC=0).WC_WIN == 16    # (no entries)
    for n in T.names():
        t = T.load(n)
        assert T.config(t).WC_WIN == (32 if t.arch["SEQ_WC"] and t.uarch["WIN"] > 16 else 16), n


def test_st16_variants_keep_the_hashes():
    """[arch] ST16 (Config.ST16, docs/isa.md "ST16"): DSTEP's state in 16 bits, in arch_hash and
    rtl_params only when on, so every target and variant before the field keeps its CFG_HASH
    (pinned: bcu1525-wg:bhw 9df8e94f, the pbhw card's) and its RTL parameters; bcu1525-wg:bhw16
    is bhw with ST16 e6m9, bcu1525-s2:st16 the S 2 target's."""
    pins = {"bcu1525-s2": 0xf2f66919, "bcu1525-s2:kt": 0xf2f66919, "bcu1525-wg": 0x4b35ec47,
            "bcu1525-wg:bhw": 0x9df8e94f, "bcu1525-wg:ic": 0xc4849fa6,
            "bcu1525-wg:kt": 0xd49b52f5, "bcu1525": 0xc6166c02,
            "nangate45-small": 0x124e28bf, "nangate45": 0x2aa6cd2b, "ypcb-00338": 0x4ffd1e17,
            "ypcb-00338:m4": 0x4ffd1e17, "ypcb-00338:m8": 0x46b96e6d}
    for n, h in pins.items():
        c = T.config(n)
        assert (T.arch_hash(c), c.ST16) == (h, ""), n
        assert "ST16" not in T.rtl_params(n), n
    bhw, b16 = T.config("bcu1525-wg:bhw"), T.config("bcu1525-wg:bhw16")
    assert dataclasses.replace(b16, ST16="") == bhw and b16.ST16 == "e6m9"
    assert T.arch_hash(b16) == 0x4d5cbf8d != T.arch_hash(dataclasses.replace(b16, ST16="bf16"))
    assert T.arch_hash(T.config("bcu1525-s2:st16")) == 0xe331f79c
    pb, p16 = T.rtl_params("bcu1525-wg:bhw"), T.rtl_params("bcu1525-wg:bhw16")
    assert p16.pop("ST16") == 1 and p16.pop("ARCH_HASH") == 0x4d5cbf8d
    pb.pop("ARCH_HASH")
    assert p16 == pb
    assert T.build_config("bcu1525-wg:bhw16") == b16
    assert T.build_config("bcu1525-wg:bhw16", ST16=0) == bhw
    assert T.build_config("bcu1525-wg:bhw", ST16="bf16").ST16 == "bf16"
    assert T.build_config("bcu1525-wg:bhw16", DSTEP=0).ST16 == ""
    with pytest.raises(T.TargetError, match="ST16"):
        T.build_config("bcu1525-wg:bhw", ST16="fp16")


def test_fsum_variant_keeps_the_hashes(tmp_path):
    """[arch] FSUM (Config.FSUM: VOP EXP2SUB's fused row sums, otpu_vpu FSUM): in arch_hash and
    rtl_params only when on, so every target and variant before it keeps its CFG_HASH and RTL
    parameters; bcu1525-wg:bhw16fs is bhw16 with it. It needs the VPU's composites on the stream
    engine's (DSTEP, LANES 8, D >= 32)."""
    for n in ("bcu1525-s2", "bcu1525-wg", "bcu1525-wg:bhw", "bcu1525-wg:bhw16", "bcu1525-wg:ic",
              "bcu1525", "nangate45", "ypcb-00338", "ypcb-00338:m8"):
        c = T.config(n)
        assert not c.FSUM and "VPU_FSUM" not in T.rtl_params(n), n
    b16, fs = T.config("bcu1525-wg:bhw16"), T.config("bcu1525-wg:bhw16fs")
    assert T.arch_hash(b16) == 0x4d5cbf8d
    assert dataclasses.replace(fs, FSUM=False) == b16 and fs.FSUM
    assert T.arch_hash(fs) != T.arch_hash(b16)
    p16, pfs = T.rtl_params("bcu1525-wg:bhw16"), T.rtl_params("bcu1525-wg:bhw16fs")
    assert pfs.pop("VPU_FSUM") == 1 and pfs.pop("ARCH_HASH") == T.arch_hash(fs)
    p16.pop("ARCH_HASH")
    assert pfs == p16
    assert T.sim_uarch(T.load("bcu1525-wg:bhw16fs"))["VPU_FSUM"] == 1
    assert "VPU_FSUM" not in T.sim_uarch(T.load("bcu1525-wg:bhw16"))
    assert T.build_config("bcu1525-wg:bhw16fs") == fs
    assert T.build_config("bcu1525-wg:bhw16fs", FSUM=0) == b16
    assert T.build_config("bcu1525-wg:bhw16", FSUM=1) == fs
    assert not T.build_config("bcu1525-wg:bhw16fs", DSTEP=0).FSUM
    good = (ROOT / "targets/nangate45-small.toml").read_text()      # (no DSTEP)
    p = tmp_path / "nangate45-small.toml"
    p.write_text(good.replace("DSTEP = false", "DSTEP = false\nFSUM = true", 1))
    with pytest.raises(T.TargetError, match="FSUM"):
        T.load(str(p))


def test_vsig_keeps_the_hashes(tmp_path):
    """[arch] VSIG (Config.VSIG: VOP SIGMOID / SILU, otpu_se_comp SIG): in arch_hash and the RTL
    and simulator parameters only when on, so every target and variant without it keeps its
    CFG_HASH; the wmx variants have it (VSIG 0 there: the variant without it). It needs the
    composites on the stream engine's path (DSTEP, LANES 8, D >= 32), at VPU_CNS 3 or 5: the knob
    is off elsewhere, and a target stating it without them is refused."""
    for n in ("bcu1525-s2", "bcu1525-wg", "bcu1525-wg:bhw16", "bcu1525-wg:bhw16fs",
              "bcu1525-x2:st16", "bcu1525", "nangate45", "ypcb-00338"):
        c = T.config(n)
        assert not c.VSIG and "VPU_SIG" not in T.rtl_params(n), n
        assert "VPU_SIG" not in T.sim_uarch(T.load(n)), n
    assert T.arch_hash(T.config("bcu1525-wg:bhw16")) == 0x4d5cbf8d
    for n in ("bcu1525-x2:wmx", "bcu1525-wg:wmx"):
        v, w = T.build_config(n), T.build_config(n, VSIG=0)
        assert v == T.config(n) and v.VSIG and dataclasses.replace(v, VSIG=False) == w
        assert T.arch_hash(v) != T.arch_hash(w)
        assert T.rtl_params(n)["VPU_SIG"] == 1 and T.sim_uarch(T.load(n))["VPU_SIG"] == 1
        assert not T.build_config(n, DSTEP=0).VSIG
    v = T.build_config("bcu1525-x2:st16", VSIG=1)                      # VPU_CNS 3 (six passes)
    assert v.VSIG and T.arch_hash(v) != T.arch_hash(T.config("bcu1525-x2:st16"))
    src = (ROOT / "targets/bcu1525-wg.toml").read_text()
    p = tmp_path / "bcu1525-wg.toml"
    p.write_text(src.replace("DSTEP = true", "DSTEP = true\nVSIG = true", 1)
                 .replace("LANES = 8", "LANES = 4", 1))
    with pytest.raises(T.TargetError, match="VSIG"):
        T.load(str(p))                                                  # LANES 4
    for cns in ("", "VPU_CNS = 5\n"):                                  # VPU_CNS 3, 5
        p.write_text(src.replace("DSTEP = true", "DSTEP = true\nVSIG = true", 1)
                     .replace("[uarch]\n", "[uarch]\n" + cns, 1))
        t = T.load(str(p))
        assert T.config(t).VSIG and T.rtl_params(t)["VPU_SIG"] == 1 and T.sim_uarch(t)["VPU_SIG"] == 1


def test_vmac_variant_keeps_the_hashes():
    """[arch] VMAC (Config.VMAC: VOP MAC, otpu_vpu MAC): in arch_hash and rtl_params only when
    on, so every target and variant without it keeps its CFG_HASH and RTL parameters; the wmx
    variants (bcu1525-x2's and bcu1525-wg's) have it."""
    for n in ("bcu1525-s2", "bcu1525-wg", "bcu1525-wg:bhw16", "bcu1525-wg:bhw16fs", "bcu1525-x2",
              "bcu1525-x2:st16", "bcu1525", "nangate45", "ypcb-00338",
              "ypcb-00338:m8"):
        c = T.config(n)
        assert not c.VMAC and "VPU_MAC" not in T.rtl_params(n), n
    assert T.arch_hash(T.config("bcu1525-wg:bhw16")) == 0x4d5cbf8d
    for n in ("bcu1525-x2:wmx", "bcu1525-wg:wmx"):
        c = T.config(n)
        off = dataclasses.replace(c, VMAC=False)
        assert c.VMAC and T.arch_hash(c) != T.arch_hash(off), n
        p = T.rtl_params(n)
        assert p["VPU_MAC"] == 1 and T.sim_uarch(T.load(n))["VPU_MAC"] == 1, n
        assert T.build_config(n, VMAC=0) == off and T.build_config(n) == c, n


def test_outer1_knob_keeps_the_hashes(tmp_path):
    """[arch] OUTER1 (Config.OUTER1: VOP OUTER decay one on MAC's datapath, otpu_vpu OUTER1): no
    target has it yet, so every one keeps its CFG_HASH and RTL parameters; a build with it (the
    knob, wmx's: VMAC on, OUTER off) hashes apart and builds VPU_OUTER1, the knob is void without
    VMAC or with OUTER, and a target stating it there is refused."""
    for n in ("bcu1525-x2:wmx", "bcu1525-wg:wmx", "bcu1525-x2", "ypcb-00338"):
        c = T.config(n)
        assert not c.OUTER1 and "VPU_OUTER1" not in T.rtl_params(n), n
        assert "VPU_OUTER1" not in T.sim_uarch(T.load(n)), n
    c = T.config("bcu1525-x2:wmx")
    on = T.build_config("bcu1525-x2:wmx", OUTER1=1)
    assert on == dataclasses.replace(c, OUTER1=True) and T.arch_hash(on) != T.arch_hash(c)
    assert not T.build_config("bcu1525-x2", OUTER1=1).OUTER1            # (no VMAC)
    assert not T.build_config("ypcb-00338", OUTER1=1).OUTER1            # (OUTER)
    bad = tmp_path / "bcu1525-x2.toml"
    bad.write_text((ROOT / "targets/bcu1525-x2.toml").read_text().replace(
        "OUTER = false", "OUTER = false\nOUTER1 = true", 1))
    T.load.cache_clear()
    with pytest.raises(T.TargetError, match="OUTER1"):
        T.load(str(bad))
    T.load.cache_clear()


# every target and variant's CFG_HASH (the cards' bitstreams and the image and program caches'
# keys hold them): a variant added to a file leaves the others' as they were
HASHES = {"bcu1525": 0xc6166c02, "bcu1525-s2": 0xf2f66919, "bcu1525-s2:kt": 0xf2f66919,
          "bcu1525-s2:st16": 0xe331f79c, "bcu1525-wg": 0x4b35ec47, "bcu1525-wg:bhw": 0x9df8e94f,
          "bcu1525-wg:bhw16": 0x4d5cbf8d, "bcu1525-wg:bhw16fs": 0x2e196bbd,
          "bcu1525-wg:ic": 0xc4849fa6, "bcu1525-wg:kt": 0xd49b52f5, "bcu1525-wg:wmx": 0xa6d2f884,
          "bcu1525-x2": 0x2963590d, "bcu1525-x2:kt": 0x2963590d, "bcu1525-x2:st16": 0xd5239726,
          "bcu1525-x2:wmx": 0x92012fec, "bcu1525-x2:b2": 0xd4bd45a5, "nangate45": 0x2aa6cd2b,
          "nangate45-small": 0x124e28bf, "ypcb-00338": 0x4ffd1e17, "ypcb-00338:m4": 0x4ffd1e17,
          "ypcb-00338:m8": 0x46b96e6d}


def test_every_variant_keeps_its_hash():
    for n, h in HASHES.items():
        assert T.arch_hash(T.config(n)) == h, n
        assert T.rtl_params(n).get("ARCH_HASH", h) == h, n


def test_b2_variant(no_env):
    """bcu1525-x2:b2 (B2, docs/bcu1525_wg.md): x2:wmx's machine with the VPU's
    composites at three stages (option C: FSUM off with VPU_CNS 5, VSIG kept), QST HL and its row
    buffer (QSTHL, QUANT_QSTB), VOP OUTER of decay one (OUTER1), quantize-once (QONCE), MM SPL
    (SPLMM) and NAT_XSTG 3 (DMA_LDQ 2 as wmx's); wmx stays B1's. Its
    Config, build_config, RTL and simulator parameters differ from wmx's in exactly those, and a
    card of it reports its Config back (device_config from its registers: CFG_HASH, CAPS2)."""
    w, b = T.config("bcu1525-x2:wmx"), T.config("bcu1525-x2:b2")
    assert b == dataclasses.replace(w, FSUM=False, QSTHL=True, OUTER1=True, QONCE=True,
                                    SPLMM=True)
    assert T.build_config("bcu1525-x2:b2") == b and T.build_config("bcu1525-x2:wmx") == w
    assert (T.arch_hash(w), T.arch_hash(b)) == (0x92012fec, 0xd4bd45a5)
    pw, pb = T.rtl_params("bcu1525-x2:wmx"), T.rtl_params("bcu1525-x2:b2")
    diff = {k: (pw.get(k), pb.get(k)) for k in set(pw) | set(pb) if pw.get(k) != pb.get(k)}
    assert diff == {"ARCH_HASH": (0x92012fec, 0xd4bd45a5), "NAT_XSTG": (2, 3),
                    "QUANT_QSTB": (None, 1), "QUANT_QSTHL": (None, 1), "VPU_CNS": (5, None),
                    "VPU_FSUM": (1, None), "VPU_OUTER1": (None, 1), "QONCE": (None, 1),
                    "MXU_SPL": (None, 1)}
    assert pw["DMA_LDQ"] == pb["DMA_LDQ"] == 2 and pw["VPU_SIG"] == pb["VPU_SIG"] == 1
    sw, sb = T.sim_uarch(T.load("bcu1525-x2:wmx")), T.sim_uarch(T.load("bcu1525-x2:b2"))
    assert {k: (sw.get(k), sb.get(k)) for k in set(sw) | set(sb) if sw.get(k) != sb.get(k)} == \
        {k: v for k, v in diff.items() if k != "ARCH_HASH"}
    assert pb["NAT_XSTG_P"] == pw["NAT_XSTG_P"] == 2 and pb["QUANT_SWW"] == 4
    from opentpu.host.fake import target_cfg
    for n in ("bcu1525-x2:wmx", "bcu1525-x2:b2"):
        no_env.setenv("OTPU_TARGET", n)
        t = FakeTransport(devname=None, regmap=5, D=128, MCOLS=8, LANES=8, act_rows=16,
                          cfg=target_cfg(n), dstep=True, stream=True, chash=True, outer=False,
                          dstep2=True, kt=True, seq_wc=True, se_own=True)
        info = dict(Board(t).info(), target=T.load(n))
        assert device_config(info) == T.config(n), n
        assert info["caps2"]["outer1"] == info["caps2"]["qsthl"] == info["caps2"]["splmm"] == \
            info["caps2"]["qonce"] == (n.endswith("b2")), n
    nohash = info | {"cfg": None}                       # (B2's registers, B1's CAPS2 below)
    check_fits(b, nohash)
    with pytest.raises(ConfigMismatch, match="OUTER1"):
        check_fits(b, nohash | {"caps2": dict(info["caps2"], outer1=False)})
    with pytest.raises(ConfigMismatch, match="QSTHL"):  # (x2:b2's programs on B1)
        check_fits(b, nohash | {"caps2": dict(info["caps2"], qsthl=False)})
    with pytest.raises(ConfigMismatch, match="SPLMM"):
        check_fits(b, nohash | {"caps2": dict(info["caps2"], splmm=False)})
    with pytest.raises(ConfigMismatch, match="FSUM"):   # (x2:wmx's on B2)
        check_fits(w, nohash)
    with pytest.raises(ConfigMismatch, match="QONCE"):
        check_fits(b, nohash | {"caps2": dict(info["caps2"], qonce=False)})
    check_fits(dataclasses.replace(w, FSUM=False), nohash)     # (VSIG on both: option C)
    with pytest.raises(ConfigMismatch, match="VSIG"):
        check_fits(b, nohash | {"caps2": dict(info["caps2"], vsig=False)})


def test_splmm_knob_keeps_the_hashes(tmp_path):
    """[arch] SPLMM (Config.SPLMM: MM SPL, a split's terms as one MM, otpu_mxu SPL): only x2:b2
    states it (test_b2_variant), so the others keep their CFG_HASH and RTL parameters; a build with
    it (the knob, wmx's:
    WAITC at WG 8) hashes apart and builds MXU_SPL, the knob is void without the wait entries or
    groups, and a target stating it there is refused."""
    for n in ("bcu1525-x2:wmx", "bcu1525-wg:wmx", "bcu1525-x2", "ypcb-00338"):
        c = T.config(n)
        assert not c.SPLMM and "MXU_SPL" not in T.rtl_params(n), n
        assert "MXU_SPL" not in T.sim_uarch(T.load(n)), n
    c = T.config("bcu1525-x2:wmx")
    on = T.build_config("bcu1525-x2:wmx", SPLMM=1)
    assert on == dataclasses.replace(c, SPLMM=True) and T.arch_hash(on) != T.arch_hash(c)
    assert not T.build_config("ypcb-00338", SPLMM=1).SPLMM             # (WG 1)
    bad = tmp_path / "ypcb-00338.toml"
    bad.write_text((ROOT / "targets/ypcb-00338.toml").read_text().replace(
        "[arch]", "[arch]\nSPLMM = true", 1))
    T.load.cache_clear()
    with pytest.raises(T.TargetError, match="SPLMM"):
        T.load(str(bad))
    T.load.cache_clear()


def test_st16_needs_the_dma_state(tmp_path):
    """A target's ST16 is e6m9, bf16 or empty, and needs DSTEP and STREAM (otpu_dma's)."""
    good = (ROOT / "targets/bcu1525-wg.toml").read_text()
    p = tmp_path / "bcu1525-wg.toml"
    p.write_text(good.replace("DSTEP2 = true", 'DSTEP2 = true\nST16 = "fp16"', 1))
    with pytest.raises(T.TargetError, match="ST16 'fp16'"):
        T.load(str(p))
    p.write_text(good.replace("DSTEP2 = true", 'DSTEP2 = true\nST16 = "e6m9"', 1)
                 .replace("DSTEP = true", "DSTEP = false", 1))
    with pytest.raises(T.TargetError, match="ST16 'e6m9'"):
        T.load(str(p))


def test_dram_address_bits(tmp_path):
    """[arch] AW (Config.AW, docs/isa.md "Machine model"): DRAM addresses of 32 to AW_MAX bits,
    DRAM_BYTES a power of two of at most 2^AW; left out, 32 (every target but the BCU's), out of
    arch_hash and rtl_params at 32 (their hashes and parameters as before the field). The
    BCU1525 at WG 8: AW 36, its 32 GiB as 8 virtual channels of 4 GiB (two a DIMM), the RTL's
    AW, DRAM_LOG2 and CH_LOG2 from it."""
    for n in T.names():
        if n not in ("bcu1525-wg", "bcu1525-s2", "bcu1525-x2"):
            assert T.config(n).AW == 32 and "AW" not in T.rtl_params(n), n
    # (the BCU1525 at two slices: 16 GiB a slice, 8 virtual channels of 2 GiB, AW 36 as one's)
    t2 = T.load("bcu1525-s2")
    assert (T.config(t2).AW, T.config(t2).DRAM_BYTES) == (36, 1 << 34)
    assert (t2.platform["channels_per_slice"], t2.platform["channel_bytes"]) == (8, 1 << 31)
    p2 = T.rtl_params(t2)
    assert (p2["AW"], p2["DRAM_LOG2"], p2["CH_LOG2"], p2["NCH"]) == (36, 34, 31, 8)
    c = T.config("bcu1525-wg")
    assert (c.AW, c.DRAM_BYTES) == (36, 1 << 35)
    t = T.load("bcu1525-wg")
    assert (t.platform["channels_per_slice"], t.platform["channel_bytes"]) == (8, 1 << 32)
    p = T.rtl_params(t)
    assert (p["AW"], p["DRAM_LOG2"], p["CH_LOG2"], p["NCH"]) == (36, 35, 32, 8)
    assert p["ARCH_HASH"] == T.arch_hash(c) != T.arch_hash(dataclasses.replace(c, AW=37))
    assert T.arch_hash(dataclasses.replace(c, AW=32)) != T.arch_hash(c)
    # at 32 the field is not hashed: a Config without it hashes the same
    card = T.config("ypcb-00338")
    assert T.arch_hash(dataclasses.replace(card, AW=33)) != T.arch_hash(card) == 0x4ffd1e17
    good = (ROOT / "targets/nangate45-small.toml").read_text()
    q = tmp_path / "nangate45-small.toml"
    for edit, msg in ((("DRAM_BYTES = 268435456", "DRAM_BYTES = 268435456\nAW = 31"), "AW 31"),
                      (("DRAM_BYTES = 268435456", "DRAM_BYTES = 268435456\nAW = 41"), "AW 41"),
                      (("DRAM_BYTES = 268435456", "DRAM_BYTES = 201326592"), "a power of two"),
                      (("DRAM_BYTES = 268435456", "DRAM_BYTES = 137438953472\nAW = 36"),
                       "over 64 GiB")):
        T.load.cache_clear()
        q.write_text(good.replace(*edit))
        with pytest.raises(T.TargetError, match=msg):
            T.load(str(q))
    T.load.cache_clear()
    q.write_text(good.replace("DRAM_BYTES = 268435456", "DRAM_BYTES = 8589934592\nAW = 33"))
    big = T.load(str(q))
    assert (T.config(big).AW, T.config(big).DRAM_BYTES, T.rtl_params(big)["AW"],
            T.rtl_params(big)["DRAM_LOG2"]) == (33, 1 << 33, 33, 33)
    T.load.cache_clear()


# ------------------------------------------------------------------------------ register map 4
def test_cfg_register_codec():
    c = dict(S=2, TMEM_WORDS=1 << 16, IMEM_WORDS=1 << 15, ACT_BLOCKS=128, DRAM_BYTES=1 << 32,
             channels=2, channel_bytes=1 << 31, beat_bytes=64, MXU_IMPL=2, TMEM_IMPL=1,
             VPU_CL=2, ULANES=8, SEQ_WC=1, WG=1, ICACHE=0, platform=2, hash=0x8B699C81)
    assert R.cfg(R.cfg_words(c)) == c
    assert R.cfg(R.cfg_words(dict(c, SEQ_WC=0)))["SEQ_WC"] == 0
    assert R.cfg(R.cfg_words(dict(c, ICACHE=1))) == dict(c, ICACHE=1)
    # CFG_DRAM [31:24]: log2(beat bytes / 64), 0 on the cards (their register value as before)
    assert R.cfg_words(c)[4] == 32 | 2 << 8 | 31 << 16
    wide = dict(c, S=4, channels=8, channel_bytes=1 << 29, beat_bytes=128, platform=7)
    assert R.cfg(R.cfg_words(wide)) == wide and R.cfg_words(wide)[4] >> 24 == 1
    # CFG_UARCH [27:25]: log2 WG, 0 on every build before weight groups (WG 1)
    assert R.cfg_words(c)[5] >> 25 == 0
    w8 = dict(wide, WG=8)
    assert R.cfg(R.cfg_words(w8)) == w8 and R.cfg_words(w8)[5] >> 25 == 3


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
           if f not in ("mem/otpu_dram.sv", "mem/otpu_native_dram.sv", "mem/otpu_xstg.sv",
                        "top/otpu_top.sv",
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
    assert set(ref) == {"otpu_pe", "otpu_mxce", "otpu_ppe"}
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
    bad = txt.replace("wire  q_blk = (q_awant && !sw_rdy_i) || q_ahold;",
                      "wire  q_blk = (q_awant && !sw_rdy) || q_ahold;")
    assert bad != txt and [t for _, t in _slice_sw_uses(bad)] == \
        ["wire  q_blk = (q_awant && !sw_rdy) || q_ahold;"]
    # and a unit connected to the raw one implicitly (.wr_idle, as before the register)
    bad = txt.replace(".act_wv(q_actwv), .wr_idle(wr_idle_i),", ".act_wv(q_actwv), .wr_idle,")
    assert bad != txt and [t for _, t in _slice_sw_uses(bad)] == \
        [".a_want(q_awant), .a_hold(q_ahold), .a_rdy(sw_rdy_i), .act_wv(q_actwv), .wr_idle,"]
    # port B: the MXU's grant on the raw ready, the loader's start on the raw read idle, and the
    # MXU's read data connected implicitly; port A: the MXU's grant on the raw ready
    for a, b in (("assign mxu_bgnt = !dma_breq && b_rdy_i && !ic_req;",
                  "assign mxu_bgnt = !dma_breq && b_rdy && !ic_req;"),
                 ("assign mxu_agnt = a_rdy_i;", "assign mxu_agnt = a_rdy;"),
                 ("else if (!ld_go) ld_go <= rd_idle_i && !ic_busy;",
                  "else if (!ld_go) ld_go <= rd_idle && !ic_busy;"),
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
    the adapter's wr_idle and rd_idle (d_wr_idle, d_rd_idle) reach the slice alone, as u_xs gives
    them (m_wr_idle, m_rd_idle: at NAT_XSTG > 0 its stages', with its queues and the sends in
    flight), for slice 0 and the others. sim/verilator/tb_board.sv checks the same in every board
    simulation."""
    code = re.sub(r"//.*", "", (ROOT / "rtl/shell/otpu_board.sv").read_text())
    assert re.findall(r"\.wr_idle\(([^)]*)\)", code) == ["m_wr_idle", "d_wr_idle"] * 2
    assert re.findall(r"\.rd_idle\(([^)]*)\)", code) == ["m_rd_idle", "d_rd_idle"] * 2
    for k in ("wr", "rd"):
        assert re.findall(rf"\.s_{k}_idle\(([^)]*)\)", code) == [f"m_{k}_idle"] * 2
        assert re.findall(rf"\.m_{k}_idle\(([^)]*)\)", code) == [f"d_{k}_idle"] * 2
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
    assert re.search(r"wb <= w_idx_h\[DW \+: BW\];", row) and "wsel <= rst ? 1'b0 : wsel_n[j];" in row
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
    and the per-RAM copies of their write addresses that the LUT RAMs needed are gone. The B
    response FIFO only where it is deep (RBB: RD >= 128, the cards') or asked for (RB_BLOCK: a
    slice's, otpu_board NAT_RBB): the WG targets' 16 beats in 32 halves of 512 bits stay LUT RAM (as block
    RAM 8 RAMB36 a channel, a sixteenth of their depth used), its tail's copies (rb_tq) in that
    branch alone. The MXU's result FIFO at RF_STYLE "block" (the
    cards'; the WG targets' MXU_RF_STYLE "distributed", below)."""
    def src(p):
        return re.sub(r"//.*", "", (ROOT / p).read_text())
    ra = r"otpu_ram_1r1w_ra\s*#\([^;]*?\.ra_n\((\w+)"
    dma = src("rtl/dma/otpu_dma.sv")
    gb = dma[dma.index("begin : g_gb"):]
    assert re.search(ra, gb).group(1) == "gh_n" and "logic [31:0] gb [NGC]" not in dma
    assert re.search(r"assign gh_n = ", dma) and not re.search(r"gb_wa \[SPC\]\[W\]", dma)
    nd = src("rtl/mem/otpu_native_dram.sv")
    assert "localparam bit RBB = RD >= 128 || RB_BLOCK;" in nd
    g_rb = nd[nd.index("begin : g_rb"):nd.index("begin : g_lut")]
    assert "if (RBB) begin : g_ra" in g_rb and re.search(ra, g_rb).group(1) == "rb_hn"
    lut = nd[nd.index("begin : g_lut"):]
    lut = lut[:lut.index("\n      end\n")]
    assert len(re.findall(r"\brbm\s*\[", nd)) == len(re.findall(r"\brbm\s*\[", lut)) == 3
    tq_uses = [l for l in nd.splitlines() if re.search(r"rb_tq\[c\]\[k\] (<=|!=)", l)]
    assert len(tq_uses) == 3 and all("!RBB" in l for l in tq_uses), tq_uses
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
    assert "if (WCP == 1 && (OOO ? wput : wready)) mem[waddr] <= wdata;" in af
    assert "if (we_c[g]) cmem[wa_c[g]] <= wdata[g * GW +: GW];" in af     # (WCP > 1: wready's copies)
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
    tq = mxu[mxu.index("if (TQ) begin : g_tq"):mxu.index("end else begin : g_ng\n")]
    ng = mxu[mxu.index("end else begin : g_ng\n"):]
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
    tq = mxu[mxu.index("if (TQ) begin : g_tq"):mxu.index("end else begin : g_ng\n")]
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
