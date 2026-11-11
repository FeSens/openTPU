"""opentpu/host/boardmacro.py (otpu_board's parameters as a board wrapper's defines: OTPU_BOARD
and the bridges' switches, a build's board_params), the host's check of a bitstream against the
target a transport names (board.check_target), and tools/board_perf.py's programs on the ISA
simulator."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from opentpu.host import boardmacro as BM

ROOT = Path(__file__).resolve().parents[1]


def test_board_macro():
    """OTPU_BOARD: otpu_board's parameters other than the wrapper's own, in its order, strings
    quoted; a key otpu_board does not declare stops it."""
    names = BM.board_param_names()
    assert {"S", "MCOLS", "WG", "NAT_RD", "BUILD_ID", "ARCH_HASH"} <= set(names)
    m = BM.board_macro({"WG": 8, "S": 4, "MCOLS": 8, "NCH": 8, "BUILD_ID": 0x5EEC12B3,
                        "ARCH_HASH": "32'hf047e5ef"})
    assert m == ", .MCOLS(8), .BUILD_ID(32'h5eec12b3), .ARCH_HASH(32'hf047e5ef), .WG(8)"
    with pytest.raises(ValueError, match="BOGUS"):
        BM.board_macro({"MCOLS": 8, "BOGUS": 1})
    # the wrapper's defines: OTPU_BOARD, the board's NAT_WPOST / NAT_XPOST / NAT_RHB / NAT_WHB for
    # the bridges (OTPU_WPOST, OTPU_XPOST, OTPU_RHB, OTPU_WHB) and their chunk FIFO entry bits,
    # log2 FIFO_DEPTH (OTPU_XSW)
    assert BM.board_defines({"MCOLS": 8}) == ("`define OTPU_BOARD , .MCOLS(8)\n`define OTPU_WPOST 0\n"
                                              "`define OTPU_XPOST 0\n`define OTPU_RHB 0\n"
                                              "`define OTPU_WHB 0\n`define OTPU_XSW 10\n")
    assert "`define OTPU_WPOST 1\n" in BM.board_defines({"NAT_WPOST": 1})
    assert "`define OTPU_XPOST 1\n" in BM.board_defines({"NAT_XPOST": 1})
    assert "`define OTPU_RHB 1\n" in BM.board_defines({"NAT_RHB": 1})
    assert "`define OTPU_WHB 1\n" in BM.board_defines({"NAT_WHB": 1})
    assert BM.board_defines({"FIFO_DEPTH": 512}).endswith("`define OTPU_XSW 9\n")


def test_board_params_post_port_w(tmp_path):
    """A wrapper's bridges post port W's beats into the MXUs' chunk FIFOs (the target's NAT_XPOST
    1, OTPU_XPOST 1): board_params refuses a target at 0 unless NAT_XPOST=0 is asked for."""
    from opentpu import target as T
    t = T.load("bcu1525-wg")
    p = BM.board_params(t, 8, 128)
    assert p["NAT_XPOST"] == 1 and "`define OTPU_XPOST 1\n" in BM.board_defines(p)
    # and its read half beats reach the bridges; its write data whole (WHB off)
    assert p["NAT_RHB"] == 1 and "`define OTPU_RHB 1\n" in BM.board_defines(p)
    assert "NAT_WHB" not in p and "`define OTPU_WHB 0\n" in BM.board_defines(p)
    assert BM.board_params(t, 8, 128, {"NAT_XPOST": "0"})["NAT_XPOST"] == 0
    # a value as a number (a Python caller's) is the string's
    assert BM.board_params(t, 8, 128, {"NAT_XPOST": 0, "FIFO_DEPTH": 512}) == \
        BM.board_params(t, 8, 128, {"NAT_XPOST": "0", "FIFO_DEPTH": "512"})
    f = tmp_path / "bcu1525-wg.toml"
    src = (ROOT / "targets/bcu1525-wg.toml").read_text()
    assert "NAT_XPOST = true" in src and "NAT_RHB = true" in src
    f.write_text(src.replace("NAT_XPOST = true", "NAT_XPOST = false", 1).replace(
        "NAT_RHB = true", "NAT_RHB = false", 1))
    T.load.cache_clear()
    try:
        t0 = T.load(str(f))
        with pytest.raises(SystemExit, match="NAT_XPOST 0"):
            BM.board_params(t0, 8, 128)
        assert BM.board_params(t0, 8, 128, {"NAT_XPOST": "0"})["NAT_XPOST"] == 0
    finally:
        T.load.cache_clear()


@pytest.mark.parametrize("name,nc", [("bcu1525-wg", 8), ("bcu1525-s2", 16)])
def test_board_params_pass_every_parameter(name, nc):
    """board_params: every rtl_params key of a WG target is otpu_board's (in OTPU_BOARD) or
    follows from the channels; a set overrides one, a string quoted; a target with more channels
    than the wrapper stops it."""
    from opentpu import target as T
    t = T.load(name)
    p = BM.board_params(t, nc, 128, {"WG": "1"})
    rp = T.rtl_params(t)
    m = BM.board_macro(p)
    for k in set(rp) - set(BM.WRAP_OWN) - set(BM.DERIVED):
        assert f".{k}(" in m, k
    assert ".WG(1)" in m and p["S"] * p["NCH"] == nc
    for k, v in (("SE_R", 0 if int(rp.get("SE_R", 0)) else 1), ("DSTEP2", 0), ("CHASH", 0), ("OUTER", 1)):
        q = BM.board_params(t, nc, 128, {k: str(v)})          # an arch knob: build_config's hash
        assert q["ARCH_HASH"] == T.arch_hash(T.build_config(t, **{k: v})), k
    with pytest.raises(SystemExit):
        BM.board_params(t, nc, 128, {"CORE_KHZ": "1"})
    with pytest.raises(SystemExit):
        BM.board_params(t, nc // 2, 128)
    assert BM.board_macro({"TMEM_STYLE": "ultra"} if "TMEM_STYLE" in BM.board_param_names()
                          else {}) in ('', ', .TMEM_STYLE("ultra")')


def _cfg(name: str, **over) -> dict:
    """The CFG_* registers of target `name`'s build (a variant by name: bcu1525-wg:kt)."""
    from opentpu.target import arch_hash, build_config, load
    t = load(name)
    a, u, p = t.arch, t.uarch, t.platform
    return dict(S=a["S"], TMEM_WORDS=a["TMEM_WORDS"], IMEM_WORDS=a["IMEM_WORDS"],
                ACT_BLOCKS=a["ACT_BLOCKS"], DRAM_BYTES=a["DRAM_BYTES"],
                channels=p["channels_per_slice"], channel_bytes=p["channel_bytes"],
                beat_bytes=p["beat_bytes"], MXU_IMPL=u["MXU_IMPL"], TMEM_IMPL=u["TMEM_IMPL"],
                VPU_CL=u["VPU_CL"], ULANES=u["ULANES"], SEQ_WC=1, platform=t.platform_id,
                hash=arch_hash(build_config(t))) | over


@pytest.mark.parametrize("bit,want,err", [
    ("bcu1525-wg", None, None), ("bcu1525-wg", "bcu1525-wg", None),
    ("bcu1525-wg:kt", "bcu1525-wg:kt", None), ("bcu1525-s2", "bcu1525-s2", None),
    ("s2x", None, "Name it: OTPU_TARGET=bcu1525-wg:<variant>"),
    ("bcu1525-wg", "bcu1525-s2", "not a build of bcu1525-s2"),
    ("bcu1525-s2", "bcu1525-wg", "not a build of bcu1525-wg"),
    ("bcu1525-wg", "bcu1525-wg:kt", "is not bcu1525-wg:kt's build's"),
    ("hash", "bcu1525-wg", "CFG_HASH 0badc0de is not bcu1525-wg's build's")])
def test_the_host_takes_the_build_by_name(bit, want, err):
    """Board.info takes the build the transport names (want_target: OTPU_TARGET) and refuses
    another (board.check_target: its platform, slices, channels and CFG_HASH, the named
    build's ARCH_HASH; a variant shares its target's platform id, its hash its own); without a
    name a bitstream whose slices are not its platform's target's (s2x: bcu1525-wg's platform
    at 2 slices) is refused, the variants named. config_summary names the variant."""
    from opentpu.host.board import Board, ConfigMismatch, config_summary
    from opentpu.host.fake import FakeTransport
    from opentpu.target import load
    c = (_cfg("bcu1525-wg", hash=0x0BADC0DE) if bit == "hash" else
         _cfg("bcu1525-wg", S=2) if bit == "s2x" else _cfg(bit))
    f = FakeTransport(devname=None, regmap=4, cfg=c, ch_bytes=1 << 12)
    f.want_target = load(want) if want else None
    b = Board(f, lock=False)
    if err:
        with pytest.raises(ConfigMismatch, match=err):
            b.info()
        return
    i = b.info()
    name = want or bit
    assert (i["target"].name, i["target"].variant) == (name.partition(":")[0], name.partition(":")[2])
    assert f"{name} (S={c['S']}, 8 channels x " in config_summary(i)
    assert "in 128-byte beats)" in config_summary(i)


def test_board_perf_benches_on_the_isa_simulator(tmp_path):
    """tools/board_perf.py's programs on the ISA simulator as the board (--isa): wstream's grouped
    M = 1 MMs (fp4, int8, an ACC pass, slice 0 alone) against its reference rows, the pointer
    chase's last beat, port B's last eight loads, its stores, DSTEP's states and outputs (and an MM
    stream beside them), the gathered rows, the mixed bench's loads and stores (alone and beside
    an MM stream); a row a slice's run gets wrong is named (its slice, the point, the row), and a
    state's word (its head)."""
    sys.path.insert(0, str(ROOT / "tools"))
    import board_perf as P
    from opentpu import target as T
    pts = ("wstream=fp4:2 int8:8 fp4:2:2 fp4:2:1:alone;latency=16 48;portb=4 8;stb=4;"
           "dstep=4 2:pair 2:fp4:2;gather=4:256 8:256;mixed=ld:1 st:1 ld:1:fp4:2 st:1:fp4:2")
    out = tmp_path / "isa.json"
    assert P.main(["--isa", "--target", "bcu1525-s2", "--points", pts, "--json", str(out)]) == 0
    r = json.loads(out.read_text())
    assert [x["point"] for x in r["wstream"]] == pts.split(";")[0][8:].split()
    assert all(x["ok"] for b in P.BENCHES for x in r.get(b, []))
    cfg = T.config(T.load("bcu1525-s2"), DRAM_BYTES=1 << 30)

    class Bad(P.IsaBoard):
        def run(self, timeout=None):
            st = super().run(timeout)
            self.dram[1][P.OUT_AT + 4 * 100] ^= 1           # slice 1, row 100
            return st
    B = P.Bench(Bad(cfg), cfg, False, True, 100.0, {})
    B.wstream(["fp4:2"])
    assert B.failed and "slice 1: 1 of 256 rows differ, first row 100" in \
        B.out["wstream"][0]["mismatch"]

    class BadState(P.IsaBoard):
        def run(self, timeout=None):
            st = super().run(timeout)
            self.dram[1][P.DS_AT + 2 * 65536 + 8] ^= 1     # slice 1, head 2's word 2
            return st
    B = P.Bench(BadState(cfg), cfg, False, True, 100.0, {})
    B.dstep(["4"])
    assert B.failed and B.out["dstep"][0]["mismatch"] == \
        "slice 1: states: 1 of 65536 words differ, first word 32770 (head 2)"


def test_board_perf_prefill_programs_below_4_gib(tmp_path):
    """tools/board_perf.py prefill_case on an Image.high (bcu1525-wg:bhw, the bank-half window:
    the weights past 4 GiB): its programs past the image's low area, below 4 GiB, as BoardBackend
    loads them (PROG_ADDR is 32 bits: past the weights the load read zeros, NOPs that never
    halt)."""
    sys.path.insert(0, str(ROOT / "tools"))
    import board_perf as P
    from opentpu import target as T
    from opentpu import waits as W
    from opentpu.llm.qwen3 import image_end
    j = {"model_type": "qwen3", "hidden_size": 512, "num_hidden_layers": 2, "vocab_size": 1024,
         "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 128,
         "intermediate_size": 1024, "tie_word_embeddings": True, "rms_norm_eps": 1e-6}
    (tmp_path / "config.json").write_text(json.dumps(j))
    cfg = T.config(T.load("bcu1525-wg:bhw"))
    c, img, progs, at, *_ = P.prefill_case(cfg, f"{tmp_path}:fp4:int8:1:4")
    assert img.high and img.nbytes > 1 << 32
    assert at == -(-image_end(img) // 4096) * 4096
    assert at + W.area(cfg.IMEM_WORDS, cfg.SEQ_WC) <= 1 << 32 and c.DRAM_BYTES >= img.nbytes
