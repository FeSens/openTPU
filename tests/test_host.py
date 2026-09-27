"""The host package (opentpu/host): device lock, status file, register map v1 fallback, trace
readout, otpu-smi, the power report parser, Engine compile pipelining, otpu-lens.

Most tests run against FakeTransport (an in-memory card with register map 2). The ones that
need the RTL side of the observability work (the trace buffer in the board model,
opentpu.hwtrace) skip when it is absent.
"""
import json
import os
import subprocess
import sys
import textwrap
import time
import types
from dataclasses import replace

import numpy as np
import pytest

from opentpu import isa as I
from opentpu.host import power as P
from opentpu.host import regs as R
from opentpu.host import smi
from opentpu.host.board import Board, BoardBackend, ConfigMismatch, device_config, rates
from opentpu.host.fake import RATES, FakeTransport
from opentpu.host.runstate import DeviceBusy, RunnerStatus, read_status
from opentpu.isasim import board_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def run_dir(tmp_path, monkeypatch):
    d = tmp_path / "otpu"
    monkeypatch.setenv("OTPU_RUN_DIR", str(d))
    return d


# ------------------------------------------------------------------------------ lock
def test_second_board_on_a_device_fails_naming_the_pid():
    a = Board(FakeTransport(devname="fake7"))
    with pytest.raises(DeviceBusy, match=f"process {os.getpid()}") as e:
        Board(FakeTransport(devname="fake7"))
    assert e.value.pid == os.getpid()
    Board(FakeTransport(devname="fake8")).close()      # another device is free
    b2 = Board(a.t)                                     # same open device: shares the lock
    assert b2.lock is a.lock
    a.close()
    Board(FakeTransport(devname="fake7")).close()      # released


def test_lock_held_by_another_process(run_dir):
    code = textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {ROOT!r})
        from opentpu.host.runstate import DeviceLock
        lk = DeviceLock("fake9")
        print("locked", flush=True)
        time.sleep(30)
    """)
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True,
                         env=dict(os.environ, OTPU_RUN_DIR=str(run_dir)))
    try:
        assert p.stdout.readline().strip() == "locked"
        with pytest.raises(DeviceBusy, match=f"process {p.pid}"):
            Board(FakeTransport(devname="fake9"))
        # a monitor never locks: it still reads the card
        d = smi.query(FakeTransport(devname="fake9"), "/dev/fake9", sleep=lambda s: None)
        assert d["ok"] and d["util"]["RUNNING"] == pytest.approx(RATES["RUNNING"])
    finally:
        p.kill()
        p.wait()
    Board(FakeTransport(devname="fake9")).close()      # the dead process's lock is gone


# ------------------------------------------------------------------------------ register map
def test_v3_info_snapshot_and_rates():
    b = Board(FakeTransport(devname=None))
    i = b.info()
    assert i["regmap"] == 3 and i["core_khz"] == 100_000 and i["build_id"] == 0x1234ABCD
    assert i["caps"] == {"trace": True, "temp": True, "i2c": False, "ddr": False, "w4": True,
                         "pair": False, "trace_depth": 4096, "pq_window": 64}
    assert i["ddr_mts"] is None
    assert i["temp_c"] == pytest.approx(0x9C4 * 503.975 / 4096 - 273.15, abs=0.01)
    s0, s1 = b.snapshot(), b.snapshot()
    assert s1["snaps"] == s0["snaps"] + 1 and s1["UPTIME"] - s0["UPTIME"] == 1_000_000
    r = rates(s0, s1, i["core_khz"])
    for k in ("RUNNING", "MXU_MAC", "DRAM_WAIT", "MXU_STARVE"):
        assert r["util"][k] == pytest.approx(RATES[k])
    assert r["seconds"] == pytest.approx(0.01)
    assert r["dram_gbs"] == pytest.approx((RATES["DRAM_RD"] + RATES["DRAM_WR"]) * 1e6 * 64
                                          / 0.01 / 1e9)


def test_ddr_rate(monkeypatch, capsys):
    """DDR_MTS (CAPS bit3) names the DDR3 speed in smi's DDR3 row and the config lines; a
    bitstream without it (the bit clear, 0xDEADBEEF at 0x54) shows plain "DDR3"."""
    t = FakeTransport(devname=None, ddr_mts=1066)
    i = Board(t).info()
    assert i["caps"]["ddr"] and i["ddr_mts"] == 1066
    assert "DDR3-1066 ch0 ok  ch1 ok" in smi.table([smi.query(t, "/dev/fake",
                                                              sleep=lambda s: None)])
    old = FakeTransport(devname=None)
    assert old.reg_read(R.R_DDR_MTS) == R.UNMAPPED and Board(old).info()["ddr_mts"] is None
    row = smi.table([smi.query(old, "/dev/fake", sleep=lambda s: None)])
    assert "DDR3 ch0 ok  ch1 ok" in row and "DDR3-" not in row

    from opentpu.host import selftest

    def first_two(self, name, fn, stage=selftest.Runner.stage):
        return stage(self, name, fn) if name in ("link", "config") else None
    monkeypatch.setattr(selftest.Runner, "stage", first_two)
    for n, (mts, want) in enumerate(((1300, "core 100 MHz, DDR3-1300, build"),
                                     (None, "core 100 MHz, build"))):
        monkeypatch.setattr(selftest, "XdmaTransport",      # selftest keeps its lock: new device
                            lambda dev, n=n, mts=mts: FakeTransport(devname=f"fake{n}",
                                                                    ddr_mts=mts))
        selftest.main([])
        line = [s for s in capsys.readouterr().out.splitlines() if "[PASS] config" in s][0]
        assert want in line


def test_v2_bitstream_has_no_mxu_starve():
    """A register map 2 bitstream: no MXU_STARVE (it would read 0xDEADBEEF); the rest works."""
    t = FakeTransport(regmap=2, devname=None)
    assert t.reg_read(R.COUNTERS["MXU_STARVE"]) == R.UNMAPPED
    b = Board(t)
    assert b.info()["regmap"] == 2
    s0, s1 = b.snapshot(), b.snapshot()
    assert "MXU_STARVE" not in s1 and s1["INSTR"] > s0["INSTR"]
    r = rates(s0, s1, 100_000)
    assert "MXU_STARVE" not in r["util"] and r["util"]["MXU_MAC"] == pytest.approx(RATES["MXU_MAC"])
    d = smi.query(t, "/dev/fake", sleep=lambda s: None)
    assert d["regmap"] == 2 and "MXU-starve" not in smi.table([d])
    assert "MXU-starve" in smi.table([smi.query(FakeTransport(devname=None), "/dev/fake",
                                                sleep=lambda s: None)])


def test_v1_bitstream_fallback():
    t = FakeTransport(regmap=1, devname=None)
    assert t.reg_read(R.R_REGMAP) == R.UNMAPPED
    assert t.reg_read(R.COUNTERS["UPTIME"]) == R.ID_OTPU      # v1 aliases 0x100 onto 0x00
    b = Board(t)
    i = b.info()
    assert i["regmap"] == 1 and i["core_khz"] is None and i["temp_c"] is None
    assert i["caps"] is None and i["calibrated"]
    assert b.snapshot() is None
    b.write(100, np.arange(300, dtype=np.uint8))                 # DRAM, programs, runs work
    assert np.array_equal(b.read(100, 300), np.arange(300, dtype=np.uint8))
    b.load_program(4096, np.zeros(16, np.uint32))
    st = b.run(timeout=5)
    assert st["cycles"] == 1_000_000 and st["instructions"] == [2]
    with pytest.raises(RuntimeError, match="no trace buffer"):
        b.run(trace={"keep": "first"})
    d = smi.query(t, "/dev/fake", sleep=lambda s: None)
    assert d["ok"] and d["regmap"] == 1 and d["util"] is None and d["power"] is None
    assert d["temp_c"] is None and "n/a" in smi.table([d])


# ------------------------------------------------------------------------------ configuration
@pytest.fixture
def no_cfg_env(monkeypatch):
    for k in ("OTPU_MCOLS", "OTPU_LANES", "OTPU_PAIR"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_device_config_follows_the_bitstream(no_cfg_env):
    info = Board(FakeTransport(devname=None, MCOLS=4, LANES=16)).info()
    cfg = device_config(info, DRAM_BYTES=1 << 22)
    assert (cfg.D, cfg.MCOLS, cfg.LANES, cfg.DRAM_BYTES) == (128, 4, 16, 1 << 22)
    assert cfg == board_config(MCOLS=4, LANES=16, DRAM_BYTES=1 << 22)
    no_cfg_env.setenv("OTPU_MCOLS", "4")                        # agreeing: fine
    assert device_config(info).MCOLS == 4
    no_cfg_env.setenv("OTPU_MCOLS", "2")
    with pytest.raises(ConfigMismatch, match="MCOLS=4 but OTPU_MCOLS=2"):
        device_config(info)
    no_cfg_env.delenv("OTPU_MCOLS")
    no_cfg_env.setenv("OTPU_LANES", "8")
    with pytest.raises(ConfigMismatch, match="LANES=16 but OTPU_LANES=8"):
        device_config(info)
    with pytest.raises(ConfigMismatch, match="D=64"):
        device_config(Board(FakeTransport(devname=None, D=64)).info())


def test_device_config_takes_column_reuse_from_caps(no_cfg_env):
    """CAPS bit5 (MM PAIR / QACT DUP) sets Config.PAIR; OTPU_PAIR must agree with it."""
    for pair in (False, True):
        info = Board(FakeTransport(devname=None, pair=pair)).info()
        assert info["caps"]["pair"] == pair and device_config(info).PAIR == pair
        no_cfg_env.setenv("OTPU_PAIR", str(int(not pair)))
        with pytest.raises(ConfigMismatch, match="OTPU_PAIR"):
            device_config(info)
        no_cfg_env.delenv("OTPU_PAIR")


def test_board_backend_rejects_another_configuration(no_cfg_env):
    card = FakeTransport(devname="fake6", MCOLS=4)
    with pytest.raises(ConfigMismatch, match="MCOLS=4 LANES=8, the configuration D=128 MCOLS=2"):
        BoardBackend(board_config(DRAM_BYTES=1 << 21), [np.zeros(4096, np.uint8)],
                     transport=card)
    Board(FakeTransport(devname="fake6")).close()             # the lock was released


def test_chat_board_backend_takes_the_bitstream_configuration(no_cfg_env):
    from opentpu.host import board, chat
    card = FakeTransport(devname="fake4", MCOLS=4, LANES=16)
    no_cfg_env.setattr(board, "XdmaTransport", lambda dev: card)
    backend, cfg = chat.make_backend("board", None, 256, "/dev/fake4", "m0")
    assert (cfg.MCOLS, cfg.LANES) == (4, 16)
    be = backend(replace(cfg, DRAM_BYTES=1 << 21), [np.zeros(4096, np.uint8)])
    assert be.info["MCOLS"] == 4 and be.board.lock is not None       # the probe did not lock
    be.close()
    no_cfg_env.setenv("OTPU_MCOLS", "2")
    with pytest.raises(ConfigMismatch, match="OTPU_MCOLS=2"):
        chat.make_backend("board", None, 256, "/dev/fake4", "m0")


def test_selftest_stops_at_config_on_a_stale_environment(no_cfg_env, capsys):
    from opentpu.host import selftest
    no_cfg_env.setattr(selftest, "XdmaTransport", lambda dev: FakeTransport(MCOLS=4))
    no_cfg_env.setenv("OTPU_MCOLS", "2")
    assert selftest.main([]) == 1
    out = capsys.readouterr().out
    assert "[PASS] link" in out and "[FAIL] config" in out
    assert "MCOLS=4 but OTPU_MCOLS=2" in out and "stopped at stage 'config'" in out


def test_4bit_image_needs_a_4bit_bitstream(run_dir):
    """An Engine with 4-bit weights refuses a bitstream without 4-bit MM support (CAPS bit4)."""
    from opentpu import lens as L
    from opentpu.host.board import ConfigMismatch, sim_config
    from opentpu.llm.qwen3 import Engine
    spec, W = L._tiny_qwen()
    cfg = sim_config(spec, 256)
    for w4 in (False, True):
        t = FakeTransport(ch_bytes=cfg.DRAM_BYTES // 2, devname="fake5", w4=w4)

        def make():
            return Engine(spec, W, cap=256, cfg=cfg, wformat="fp4",
                          backend=lambda c, imgs: BoardBackend(c, imgs, transport=t))
        if not w4:
            with pytest.raises(ConfigMismatch, match="4-bit"):
                make()
        else:
            eng = make()
            assert eng.backend.info["caps"]["w4"]
            eng.backend.close()


def test_pair_programs_need_a_pair_bitstream(run_dir):
    """Programs compiled for column reuse refuse a bitstream without it (CAPS bit5)."""
    from opentpu import lens as L
    from opentpu.host.board import sim_config
    from opentpu.llm.qwen3 import Engine
    spec, W = L._tiny_qwen()
    cfg = replace(sim_config(spec, 256), PAIR=True)
    for pair in (False, True):
        t = FakeTransport(ch_bytes=cfg.DRAM_BYTES // 2, devname="fake6", pair=pair)

        def make():
            return Engine(spec, W, cap=256, cfg=cfg, wformat="fp4",
                          backend=lambda c, imgs: BoardBackend(c, imgs, transport=t))
        if not pair:
            with pytest.raises(ConfigMismatch, match="column reuse"):
                make()
        else:
            eng = make()
            eng.backend.close()


# ------------------------------------------------------------------------------ status file
def test_status_file_lifecycle(run_dir):
    from opentpu import lens as L
    from opentpu.host.board import sim_config
    from opentpu.llm.qwen3 import Engine
    spec, W = L._tiny_qwen()
    cfg = sim_config(spec, 256)
    t = FakeTransport(ch_bytes=cfg.DRAM_BYTES // 2, devname="fake3", cycles=2_000_000)
    eng = Engine(spec, W, cap=256, cfg=cfg,
                 backend=lambda c, imgs: BoardBackend(c, imgs, transport=t, model="tiny"))
    path = run_dir / "fake3.json"
    st = json.loads(path.read_text())
    assert st["pid"] == os.getpid() and st["model"] == "tiny" and st["tokens"] == 0
    lay = st["dram"]
    assert lay["total"] == cfg.DRAM_BYTES and lay["kv_used"] == 0
    assert lay["kv_capacity"] == 2 * 2 * (2 * 256 * 128 + 4 * 256 + 4 * 256)   # layers x heads
    assert lay["weights"] + lay["kv_capacity"] == lay["image"]
    assert lay["free"] == cfg.DRAM_BYTES - (-(-lay["image"] // 4096) * 4096) - lay["program"]
    for tok in (5, 6, 7):
        eng.step(tok)
    time.sleep(0.3)                             # the last tokens come with the timer's write
    st = read_status("fake3")
    assert st["tokens"] == 3 and st["last_cycles"] == 2_000_000 and not st["stale"]
    assert st["tok_s_device"] == pytest.approx(100e6 / 2e6)
    assert st["tok_s_wall"] > 0
    assert st["dram"]["kv_used"] == 3 * lay["kv_capacity"] // 256
    assert not list(run_dir.glob(".*.tmp"))                   # atomic replace left nothing
    eng.backend.close()
    assert not path.exists() and read_status("fake3") is None
    # a file left by a killed runner is stale
    path.write_text(json.dumps({"pid": 2 ** 22 + 12345, "dram": lay}))
    assert read_status("fake3")["stale"]
    d = smi.query(FakeTransport(devname="fake3"), "/dev/fake3", sleep=lambda s: None)
    assert d["process"]["stale"] and d["dram"] is None


def test_runner_status_is_atomic(run_dir):
    s = RunnerStatus("fake4", model="m")
    for k in range(50):
        s.token(1000 + k, 100_000)
        assert json.loads((run_dir / "fake4.json").read_text())["tokens"] == k + 1
    s.remove()
    assert not (run_dir / "fake4.json").exists()



def test_runner_status_min_interval_defers_to_a_timer(run_dir):
    s = RunnerStatus("fake6", min_interval=0.1)
    for k in range(5):
        s.token(1000 + k, 100_000)
    assert json.loads((run_dir / "fake6.json").read_text())["tokens"] == 1   # the first at once
    time.sleep(0.2)
    assert json.loads((run_dir / "fake6.json").read_text())["tokens"] == 5   # then the timer
    s.remove()

# ------------------------------------------------------------------------------ polling
def test_poll_backs_off():
    t = FakeTransport(devname=None, run_s=0.05)
    b = Board(t)
    t0 = time.perf_counter()
    b.run(timeout=5)
    dt = time.perf_counter() - t0
    assert 0.05 <= dt < 0.2                         # wakes up within POLL_MAX_SLEEP (+ slack)
    assert t.reads < 2000                           # not a hot spin (that is ~50k reads)


# ------------------------------------------------------------------------------ trace readout
PROG = [I.ld(0, 0, 64), I.st(4096, 0, 64), I.halt()]
LINES = ["T0 D c=1 s=0 pc=0", "T0 S c=2 s=0 u=0 r=2", "T0 D c=3 s=1 pc=1",
         "T0 E c=9 s=0", "T0 U c=9 u=0 n=2", "T0 S c=10 s=1 u=0 r=10", "T0 E c=16 s=1",
         "T0 U c=16 u=0 n=2", "T0 H c=18 bmxu=0 bdma=4 amxu=0 aq=0"]


@pytest.fixture
def stub_hwtrace(monkeypatch):
    """opentpu.hwtrace stand-in: a record is the index of its trace line (the real module
    decodes the 64-bit format; this checks the host plumbing around it)."""
    m = types.ModuleType("opentpu.hwtrace")
    m.records_to_trace = lambda records, sid=0: "\n".join(
        LINES[int(r)].replace("T0", f"T{sid}") for r in records)
    monkeypatch.setitem(sys.modules, "opentpu.hwtrace", m)
    import opentpu
    monkeypatch.setattr(opentpu, "hwtrace", m, raising=False)
    return m


def test_trace_keep_first_and_profile(stub_hwtrace):
    from opentpu.host.hwlens import hw_profile
    t = FakeTransport(devname=None, trace=list(range(len(LINES))), trace_drop=3, cycles=20)
    b = Board(t)
    st = b.run(trace={"keep": "first"})
    tr = st["trace"]
    assert list(tr["records"]) == list(range(len(LINES)))
    assert tr["count"] == len(LINES) and tr["drop"] == 3 and tr["lost"] == 0
    assert t.regs[R.R_TRACE_CTRL] == 0                          # disabled after the run
    d = hw_profile("stub", board_config(), [PROG], st, 100_000)
    assert d["kind"] == "hw" and d["cycles"] == 20 and len(d["instrs"]) == 2
    assert d["instrs"][0][9:11] == [2, 9] and d["instrs"][1][9:11] == [10, 16]
    assert d["slices"][0]["ports"]["bdma"] == 4
    assert d["hwtrace"]["drop"] == 3 and not d["hwtrace"]["complete"]
    assert "dropped 3 events" in d["notes"][0]["text"]


def test_trace_ring_wrap_order(stub_hwtrace):
    # depth 8, 11 records written: the ring holds records 3..10, the oldest at 11 % 8 = 3
    depth, count = 8, 11
    written = [100 + k for k in range(count)]
    ring = [0] * depth
    for k, r in enumerate(written):
        ring[k % depth] = r
    t = FakeTransport(devname=None, trace=ring, trace_extra=count - depth, trace_log2=3)
    st = Board(t).run(trace={"keep": "last"})
    tr = st["trace"]
    assert list(tr["records"]) == written[count - depth:]
    assert tr["wrapped"] and tr["lost"] == 3 and tr["keep"] == "last"


def test_record_traces_only_the_window(stub_hwtrace):
    """otpu-lens record's step loop: prompt tokens, then greedy; only positions pos0 ..
    pos0 + n - 1 run with the trace buffer on, one profile each."""
    from opentpu import lens as L
    from opentpu.host import hwlens
    from opentpu.host.board import sim_config
    from opentpu.llm.qwen3 import Engine
    spec, W = L._tiny_qwen()
    cfg = sim_config(spec, 256)
    t = FakeTransport(ch_bytes=cfg.DRAM_BYTES // 2, trace=list(range(len(LINES))), cycles=30)
    enables = []
    wr = t.reg_write
    t.reg_write = lambda off, v: (enables.append(v) if off == R.R_TRACE_CTRL and v & R.TR_ENABLE
                                  else None, wr(off, v))
    eng = Engine(spec, W, cap=256, cfg=cfg, pipeline="thread",
                 backend=lambda c, imgs: BoardBackend(c, imgs, transport=t))
    steps = hwlens.run_steps(eng, [3, 4, 5], pos0=2, n=2, keep="first", prompt_len=3)
    assert [p for p, _ in steps] == [2, 3] and len(enables) == 2 and eng.pos == 4
    assert eng.backend.trace is None
    profs = hwlens._profiles_of_steps(steps, "tiny", cfg, 100_000)
    assert [d["name"] for d in profs] == ["tiny token at pos 2", "tiny token at pos 3"]
    assert all(d["kind"] == "hw" and d["hwtrace"]["records"] == len(LINES) for d in profs)
    eng.backend.close()


def test_orphans_dropped_after_wrap():
    from opentpu.host.hwlens import drop_orphans
    text = "\n".join(LINES[3:])                 # slot 0's D was overwritten
    kept = drop_orphans(text).splitlines()
    assert "T0 E c=9 s=0" not in kept and "T0 S c=10 s=1 u=0 r=10" not in kept
    assert drop_orphans("\n".join(LINES)) == "\n".join(LINES)


def test_missing_hwtrace_is_a_clear_error(monkeypatch):
    import opentpu
    from opentpu.host import hwlens
    monkeypatch.setitem(sys.modules, "opentpu.hwtrace", None)
    monkeypatch.delattr(opentpu, "hwtrace", raising=False)     # imported by another test
    with pytest.raises(hwlens.NoHwTrace, match="records_to_trace"):
        hwlens.records_to_trace(np.zeros(1, np.uint64), [PROG])


# ------------------------------------------------------------------------------ otpu-smi
POWER_RPT = """\
Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
| Tool Version     : Vivado v.2022.2 (lin64) Build 3671981 Fri Oct 14 04:59:54 MDT 2022
| Design           : otpu_fpga_top
| Device           : xc7k480tffg1156-2

Power Report

Table of Contents
-----------------
1. Summary
1.1 On-Chip Components

1. Summary
----------

+--------------------------+--------------+
| Total On-Chip Power (W)  | 6.000        |
| Design Power Budget (W)  | Unspecified* |
| Dynamic (W)              | 5.200        |
| Device Static (W)        | 0.800        |
| Junction Temperature (C) | 33.4         |
| Confidence Level         | Low          |
+--------------------------+--------------+


1.1 On-Chip Components
----------------------

+----------------+-----------+----------+-----------+-----------------+
| On-Chip        | Power (W) | Used     | Available | Utilization (%) |
+----------------+-----------+----------+-----------+-----------------+
| Clocks         |     0.700 |       20 |       --- |             --- |
| Slice Logic    |     1.100 |   180000 |       --- |             --- |
| DSPs           |     0.600 |      256 |      1920 |           13.33 |
| Static Power   |     0.800 |          |           |                 |
| Total          |     6.000 |          |           |                 |
+----------------+-----------+----------+-----------+-----------------+


3.1 By Hierarchy
----------------

+-------------------+-----------+
| Name              | Power (W) |
+-------------------+-----------+
| otpu_fpga_top     |     5.200 |
|   u_bd            |     2.000 |
|   u_board         |     3.100 |
|     u_ctrl        |     0.050 |
|     u_slice       |     2.700 |
|       u_seq       |     0.100 |
|       u_tmem      |     0.300 |
|       u_mxu       |     1.500 |
|         u_dma     |     0.010 |
|       u_act       |     0.200 |
|       u_quant     |     0.150 |
|       u_vpu       |     0.400 |
|       u_dma       |     0.050 |
|     u_mem         |     0.250 |
|   u_other         |     0.100 |
|     u_mxu         |     0.090 |
+-------------------+-----------+
"""

POWER_XML = """<?xml version="1.0" encoding="UTF-8"?>
<RptDoc title="Power Report">
 <section title="Summary">
  <table>
   <tablerow><tablecell contents="Total On-Chip Power (W)"/><tablecell contents="6.000"/></tablerow>
   <tablerow><tablecell contents="Dynamic (W)"/><tablecell contents="5.200"/></tablerow>
   <tablerow><tablecell contents="Device Static (W)"/><tablecell contents="0.800"/></tablerow>
  </table>
 </section>
 <section title="By Hierarchy">
  <table>
   <tablerow><tableheader contents="Name"/><tableheader contents="Power (W)"/></tablerow>
   <tablerow><tablecell contents="otpu_fpga_top"/><tablecell contents="5.200"/>
    <tablerow><tablecell contents="u_board"/><tablecell contents="3.100"/>
     <tablerow><tablecell contents="u_slice"/><tablecell contents="2.700"/>
      <tablerow><tablecell contents="u_mxu"/><tablecell contents="1.500"/></tablerow>
      <tablerow><tablecell contents="u_vpu"/><tablecell contents="0.400"/></tablerow>
     </tablerow>
    </tablerow>
   </tablerow>
  </table>
 </section>
</RptDoc>
"""


def test_power_report_parser(tmp_path):
    d = P.parse_report(POWER_RPT, "power.rpt")
    assert d["total_w"] == 6.0 and d["dynamic_w"] == 5.2 and d["static_w"] == 0.8
    assert d["confidence"] == "Low" and d["junction_c"] == 33.4 and "2022.2" in d["tool"]
    assert d["components"] == {"Clocks": 0.7, "Slice Logic": 1.1, "DSPs": 0.6,
                               "Static Power": 0.8}
    assert [0, "otpu_fpga_top", 5.2] in d["hierarchy"] and [3, "u_mxu", 1.5] in d["hierarchy"]
    # u_dma inside u_mxu is part of the MXU; the slice's u_dma is the DMA; a u_mxu outside
    # u_board is not the accelerator's
    assert d["units"] == {"SEQ": 0.1, "TMEM": 0.3, "MXU": 1.7, "QNT": 0.15, "VPU": 0.4,
                          "DMA": 0.05, "DRAM": 0.25}
    assert d["fixed_w"] == pytest.approx(6.0 - 2.95)
    assert d["util"]["MXU"] == "MXU_BUSY" and d["util"]["DRAM"] == "DRAM"
    x = P.parse_report(POWER_XML, "power.xml")
    assert x["total_w"] == 6.0 and x["units"] == {"MXU": 1.5, "VPU": 0.4}
    assert [3, "u_mxu", 1.5] in x["hierarchy"]
    e = P.estimate(d, {"MXU_BUSY": 0.5, "VPU_BUSY": 0.0, "QNT_BUSY": 1.0, "DMA_BUSY": 0.0,
                       "RUNNING": 1.0, "DRAM": 0.5})
    assert e["w"] == pytest.approx(3.05 + 0.85 + 0.15 + 0.1 + 0.3 + 0.125, abs=1e-3)
    out, rpt = tmp_path / "power.json", tmp_path / "power.rpt"
    rpt.write_text(POWER_RPT)
    assert P.main([str(rpt), "-o", str(out)]) == 0
    assert P.load(out)["units"] == d["units"]
    assert P.load(tmp_path / "missing.json") is None
    with pytest.raises(ValueError, match="report_power"):
        P.parse_report("nothing here")


def test_smi_json(tmp_path, capsys):
    pj = tmp_path / "power.json"
    pj.write_text(json.dumps(P.parse_report(POWER_RPT, "power.rpt")))
    card = FakeTransport(devname="fake5", cycles=4_000_000)
    img = [np.zeros(1 << 16, np.uint8)]
    be = BoardBackend(board_config(DRAM_BYTES=1 << 21), img, transport=card, model="m0")
    be.run([PROG])
    rc = smi.main(["--json", "--dev", "/dev/fake5", "--power-json", str(pj), "-i", "0"],
                  open_transport=lambda dev: FakeTransport(devname="fake5"))
    assert rc == 0
    (d,) = json.loads(capsys.readouterr().out)
    assert d["device"] == "/dev/fake5" and d["ok"] and d["regmap"] == 3
    assert d["bitstream"] == {"D": 128, "MCOLS": 2, "LANES": 8, "core_mhz": 100.0,
                              "build_id": 0x1234ABCD, "ddr_mts": None}
    assert d["calib"] == [True, True] and d["temp_c"] == pytest.approx(34.45, abs=0.01)
    for k, v in RATES.items():
        if k not in R.EVENTS:
            assert d["util"][k] == pytest.approx(v)
    assert d["dram_gbs"] == pytest.approx(4.608)
    assert d["sample"]["seconds"] == pytest.approx(0.01)
    p = d["process"]
    assert p["pid"] == os.getpid() and p["model"] == "m0" and p["tokens"] == 1
    assert p["tok_s_device"] == pytest.approx(25.0) and not p["stale"]
    assert d["dram"]["total"] == 1 << 21 and d["dram"]["image"] == 1 << 16
    pw = d["power"]
    want = P.estimate(json.loads(pj.read_text()), d["util"])["w"]
    assert pw["w"] == pytest.approx(want) and pw["max_w"] == 6.0 and pw["w"] < 6.0
    # the table and -q render the same data
    smi.main(["--dev", "/dev/fake5", "--power-json", str(pj), "-i", "0"],
             open_transport=lambda dev: FakeTransport(devname="fake5"))
    tab = capsys.readouterr().out
    assert "4.61 GB/s" in tab and f"PID {os.getpid()}" in tab and "Model m0" in tab
    assert f"{want:.1f}W" in tab and "MAC █████░░░░░  50%" in tab
    assert all(len(line) == len(tab.splitlines()[1]) for line in tab.splitlines()
               if line[:1] in "│╭├╰")                          # a closed box
    smi.main(["-q", "--dev", "/dev/fake5", "--power-json", str(tmp_path / "none.json")],
             open_transport=lambda dev: FakeTransport(devname="fake5"))
    q = capsys.readouterr().out
    assert "MXU_MAC" in q and "power            None" in q
    be.close()


def test_sim_transport_batched_reads(have_verilator):
    """queue_read / wait_cycles: several reads (the same address too) and a wait in one
    simulation, results in order."""
    from opentpu.host.board import SimTransport
    t = SimTransport(ch_bytes=1 << 20)
    t.reg_write(R.R_SCRATCH, 0x1234)
    a = t.queue_read(R.R_SCRATCH)
    t.wait_cycles(100)
    t.reg_write(R.R_SCRATCH, 0x5678)
    b = t.queue_read(R.R_SCRATCH)
    c = t.queue_read(R.R_ID)
    t.flush()
    assert (a, b, c) == (0, 1, 2)
    assert t.results == [0x1234, 0x5678, R.ID_OTPU] and t.cycles > 100
    assert t.reg_read_many([R.R_ID, R.R_ID]) == [R.ID_OTPU, R.ID_OTPU]


def test_smi_sim(have_verilator, capsys):
    """otpu-smi --sim: the demo program between two samples in one simulation (counters with
    register map 2; the run's cycles either way)."""
    assert smi.main(["--sim", "--json"]) == 0
    (d,) = json.loads(capsys.readouterr().out)
    assert d["ok"] and d["run"]["cycles"] > 0 and d["run"]["instructions"] == d["run"]["of"]
    if d["regmap"] >= 2:
        assert 0 < d["util"]["RUNNING"] <= 1 and d["util"]["MXU_MAC"] > 0
        assert d["sample"]["cycles"] >= d["run"]["cycles"]
    else:
        assert d["util"] is None


def test_smi_no_device(capsys):
    assert smi.main(["--dev", "/dev/nonexistent_xdma"]) == 1
    assert "cannot open" in capsys.readouterr().out


# ------------------------------------------------------------------------------ pipelining
def test_pipelining_gives_identical_tokens():
    """The Engine compiles the next run (a prefill chunk, or position p + 1) while the backend
    runs the current one: same programs, same logits, same tokens as compiling in line."""
    from opentpu import lens as L
    from opentpu.llm.qwen3 import Engine, IsaBackend
    spec, W = L._tiny_qwen()

    class Slow(IsaBackend):                      # a device that takes time (the GIL is free)
        prepared = 0

        def prepare(self, programs):
            Slow.prepared += 1

        def run(self, programs):
            time.sleep(0.005)
            return super().run(programs)

    a = Engine(spec, W, cap=256)                              # ISA, no pipeline
    b = Engine(spec, W, cap=256, backend=Slow)                # pipelined
    assert not a.pipeline and b.pipeline
    prompt = [11, 222, 333, 44, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    assert a.generate(prompt, max_new=5) == b.generate(prompt, max_new=5)
    assert [st.get("rows") for st in b.stats] == [8, 8, 3] + [None] * 5
    assert Slow.prepared >= len(b.stats)                     # every run's program
    for tok in (7, 8):                                        # logits, bit for bit
        assert np.array_equal(a.step(tok).view(np.uint32), b.step(tok).view(np.uint32))
    a.reset()
    b.reset()                                                 # stale precompile is dropped
    assert np.array_equal(a.step(9).view(np.uint32), b.step(9).view(np.uint32))



def test_board_compiles_the_next_program_after_starting_the_card():
    """With a backend that has start / wait (the board), the next position's compile begins
    only once the program is on the card and running (not while the host copies it)."""
    from opentpu import lens as L
    from opentpu.host.board import sim_config
    from opentpu.llm.qwen3 import Engine
    spec, W = L._tiny_qwen()
    cfg = sim_config(spec, 256)
    t = FakeTransport(ch_bytes=cfg.DRAM_BYTES // 2, devname=None, run_s=0.01)
    starts, seen = [], []                   # seen: (position compiled, starts before it)

    class Rec(BoardBackend):
        def start(self, programs):
            starts.append(1)
            super().start(programs)

    eng = Engine(spec, W, cap=256, cfg=cfg, pipeline="thread",
                 backend=lambda c, imgs: Rec(c, imgs, transport=t))
    compile_ = eng._compile
    eng._compile = lambda pos: (seen.append((pos, len(starts))), compile_(pos))[1]
    for tok in (3, 4, 5):
        eng.step(tok)
    eng._drain()
    # position 0 compiles in line; position p + 1 only after the card started position p
    assert seen == [(0, 0), (1, 1), (2, 2), (3, 3)]
    assert eng.backend._expect == pytest.approx(t.cycles_per_run / 100e6)
    eng.backend.close()



def test_board_compiles_in_a_worker_process():
    """The board's default pipeline compiles in a spawned worker process: the words it sends
    are the in-process assembly of the same position's program (the board-model tests in
    test_board.py / test_lfm2.py check the logits against the ISA simulator through it)."""
    from opentpu import lens as L
    from opentpu.host.board import sim_config
    from opentpu.llm.qwen3 import Engine
    spec, W = L._tiny_qwen()
    cfg = sim_config(spec, 256)
    t = FakeTransport(ch_bytes=cfg.DRAM_BYTES // 2, devname=None, run_s=0.002)
    sent = []

    class Rec(BoardBackend):
        def start(self, programs):
            sent.append(programs)
            super().start(programs)

    eng = Engine(spec, W, cap=256, cfg=cfg, backend=lambda c, imgs: Rec(c, imgs, transport=t))
    assert eng._procs and eng._ready.result(timeout=60)
    for tok in (3, 4, 5, 6):
        eng.step(tok)
    eng._drain()
    words = [(p, w) for p, w in enumerate(sent) if isinstance(w, np.ndarray)]
    assert len(words) >= 3                  # position 0 compiles in line, then the worker
    for p, w in words:
        assert np.array_equal(w, np.asarray(I.assemble(eng.image.compile_step(p)[0]),
                                            np.uint32))
    eng.backend.close()

def test_board_compiles_prefill_chunks_in_the_worker_process():
    """Chunked prefill through the worker process: the first chunk compiles in line, the next
    ones in the worker while the card runs the one before, and after the last chunk the first
    decode step's program; the words are the in-process assembly of the same programs."""
    from opentpu import lens as L
    from opentpu.host.board import sim_config
    from opentpu.llm.qwen3 import Engine, fit_chunk
    spec, W = L._tiny_qwen()
    cfg = sim_config(spec, 256)
    t = FakeTransport(ch_bytes=cfg.DRAM_BYTES // 2, devname=None, run_s=0.002)
    sent = []

    class Rec(BoardBackend):
        def start(self, programs):
            sent.append(programs)
            super().start(programs)

    eng = Engine(spec, W, cap=256, cfg=cfg, backend=lambda c, imgs: Rec(c, imgs, transport=t))
    assert eng._procs and eng._ready.result(timeout=60)
    runs = [len(part) for part, _ in eng.prefill_chunks(list(range(3, 22)))]
    assert runs == [8, 8, 3] and eng.pos == 19
    eng.step(5)
    assert not isinstance(sent[0], np.ndarray)             # compiled in line
    want = [fit_chunk(eng.image, eng.block, 0, 8, 8, 11, 8)[1],
            fit_chunk(eng.image, eng.block, 0, 16, 8, 3, 8)[1], eng.image.compile_step(19)]
    for w, progs in zip(sent[1:], want):
        assert isinstance(w, np.ndarray)
        assert np.array_equal(w, np.asarray(I.assemble(progs[0]), np.uint32))
    eng.backend.close()


def test_compile_worker_exits_with_its_parent(tmp_path):
    """A parent that dies without shutting the pool down (killed, os._exit) takes its compile
    worker with it (the worker would otherwise wait for work forever)."""
    script = tmp_path / "child.py"
    script.write_text(textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {ROOT!r})
        from opentpu import lens as L
        from opentpu.host.board import BoardBackend, sim_config
        from opentpu.host.fake import FakeTransport
        from opentpu.llm.qwen3 import Engine

        if __name__ == "__main__":
            spec, W = L._tiny_qwen()
            cfg = sim_config(spec, 256)
            t = FakeTransport(ch_bytes=cfg.DRAM_BYTES // 2, devname=None)
            eng = Engine(spec, W, cap=256, cfg=cfg,
                         backend=lambda c, imgs: BoardBackend(c, imgs, transport=t))
            eng._ready.result()
            print(*eng._pool._processes, flush=True)
            os._exit(0)
    """))
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                         timeout=120)
    pids = [int(p) for p in out.stdout.split()]
    assert pids, out.stderr
    deadline = time.time() + 10
    alive = pids
    while alive and time.time() < deadline:
        time.sleep(0.2)
        alive = [p for p in alive if _alive(p)]
    assert not alive


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True

def test_poll_with_an_expected_wait_sleeps_then_reads_back_to_back():
    t = FakeTransport(devname=None, run_s=0.05)
    b = Board(t)
    t0 = time.perf_counter()
    b.run(timeout=5, expect=0.05)
    dt = time.perf_counter() - t0
    assert 0.05 <= dt < 0.1 and t.reads < 1_000_000   # one sleep, then a short spin


def test_detok_streams_the_text_of_a_full_decode():
    """Chat's incremental detokenization: the deltas add up to decode(all tokens), holding
    back an incomplete UTF-8 character until the token that completes it."""
    from opentpu.host.chat import Detok

    class ByteTok:                          # one token per byte (multi-byte characters split)
        def decode(self, ids, skip_special_tokens=True):
            return bytes(ids).decode("utf-8", errors="replace")
    text = "ab ü€ 🎉 end"
    ids = list(text.encode())
    d = Detok(ByteTok())
    deltas = [d.add(i) for i in ids]
    assert "".join(deltas) == text and not any("\ufffd" in x for x in deltas)
    assert deltas[ids.index(0xC3)] == ""    # the first byte of ü shows nothing yet

# ------------------------------------------------------------------------------ otpu-lens
def test_otpu_lens_passthrough(capsys):
    from opentpu.host import hwlens
    assert hwlens.main(["list"]) == 0
    assert "qwen-tiny" in capsys.readouterr().out


def _board_model_has_trace() -> bool:
    import shutil
    if shutil.which("verilator") is None:
        return False
    try:
        import opentpu.hwtrace  # noqa: F401
    except ImportError:
        return False
    from opentpu.host.board import SimTransport
    i = Board(SimTransport(ch_bytes=1 << 20), check=False).info()
    return i["regmap"] >= 2 and bool(i["caps"] and i["caps"]["trace"])


def test_otpu_lens_record_on_board_model(tmp_path, capsys):
    """otpu-lens record --sim: the hardware trace of a kernel on the board model gives a full
    profile (needs the RTL side: the trace buffer in tb_board and opentpu.hwtrace)."""
    if not _board_model_has_trace():
        pytest.skip("board model without the trace buffer (register map 1) or no "
                    "opentpu.hwtrace / verilator")
    from opentpu import lens as L
    from opentpu.host import hwlens
    out = tmp_path / "hw.otpuprof"
    assert hwlens.main(["record", "--sim", "--workload", "mlp-small", "-o", str(out)]) == 0
    (d,) = L.load(out)["profiles"]
    assert d["kind"] == "hw" and d["cycles"] > 0 and d["instrs"]
    h = d["hwtrace"]
    assert h["count"] >= len(d["instrs"]) and h["records"] == min(h["count"], h["depth"])
    assert "trace records" in capsys.readouterr().out


# ------------------------------------------------------------------------------ otpu-chat
def test_chat_sampling_defaults_per_model_and_repetition_penalty():
    from opentpu.host.chat import sampler, sampling
    from opentpu.llm import lfm2, qwen3
    q = qwen3.Spec(256, 2, 4, 2, 128, 512, 1000)
    f = lfm2.Spec(256, ("conv", "attn"), 4, 2, 64, 512, 1000)
    none = types.SimpleNamespace(temperature=None, top_k=None, top_p=None,
                                 repetition_penalty=None)
    assert sampling(q, none) == dict(temperature=0.7, top_k=20, top_p=0.8, repetition_penalty=1.0)
    assert sampling(f, none) == dict(temperature=0.1, top_k=50, top_p=1.0,
                                     repetition_penalty=1.05)
    flags = types.SimpleNamespace(temperature=0.5, top_k=None, top_p=None, repetition_penalty=1.0)
    assert sampling(f, flags) == dict(temperature=0.5, top_k=50, top_p=1.0,
                                      repetition_penalty=1.0)
    # Hugging Face's rule: a seen token's positive logit is divided, a negative one multiplied
    greedy = sampler(0, 50, 1.0, None, repetition_penalty=1.05)
    pos, neg = np.array([3.0, 2.9, -5.0]), np.array([-1.0, -1.02, -5.0])
    assert greedy(pos, [0]) == 1 and greedy(neg, [0, 0]) == 1 and greedy(pos, []) == 0
    assert pos[0] == 3.0                                      # the caller's logits are kept
    assert sampler(0, 20, 0.8, None)(pos, [0]) == 0           # no penalty: plain argmax
    # top_p = 1 keeps every top-k candidate (and does not overrun them)
    pick = sampler(1.0, 3, 1.0, 0)
    assert {pick(np.array([0.0, 0.0, 0.0, -50.0])) for _ in range(200)} == {0, 1, 2}



def test_sampler_fast_top_k_picks_as_the_float64_path():
    """The float32 top-k (block-max prefilter) gives the picks of the float64 argpartition
    path, with ties (which fall back to it), -0 / +0, a growing context and top_k 0."""
    from opentpu.host import chat as C

    def reference(temperature, top_k, top_p, seed, rp):
        rng = np.random.default_rng(seed)

        def pick(logits, context=()):
            if rp != 1.0 and len(context):
                logits = logits.copy()
                seen = np.unique(np.asarray(context, np.int64))
                v = logits[seen]
                logits[seen] = np.where(v > 0, v / rp, v * rp)
            if temperature <= 0:
                return int(np.argmax(logits))
            idx, z = C._top_k_f64(logits, top_k, temperature)
            p = np.exp(z - z[0])
            p /= p.sum()
            keep = min(len(p), np.searchsorted(np.cumsum(p), top_p) + 1)
            p = p[:keep] / p[:keep].sum()
            return int(idx[rng.choice(keep, p=p)])
        return pick

    rng = np.random.default_rng(7)
    fast = 0
    for V in (1000, 4099):                                  # 4099: a partial last block
        for T, k, tp, rp in [(0.1, 50, 1.0, 1.05), (0.7, 20, 0.8, 1.0), (1.0, 5, 1.0, 1.2),
                             (0, 50, 1.0, 1.05), (0.7, 0, 0.9, 1.0)]:
            a, b, ctx = reference(T, k, tp, 3, rp), C.sampler(T, k, tp, 3, rp), []
            for step in range(40):
                lg = (rng.standard_normal(V) * 3).astype(np.float32)
                if step % 4 == 1:
                    lg = np.round(lg * 4) / 4                   # ties everywhere
                elif step % 4 == 2:
                    lg[rng.integers(0, V, 5)] = lg.max()        # a tied maximum
                elif step % 4 == 3:
                    lg[rng.integers(0, V, 3)] = -0.0
                ctx.append(int(rng.integers(0, V)))
                if T > 0 and k:
                    fast += C._top_k_f32(lg, k, T) is not None
                assert a(lg, ctx) == b(lg, ctx)
    assert fast > 50                                        # the fast path did run

# ------------------------------------------------------------------------------ otpu-diag
def test_diag_sim_registers_and_memory_pass(have_verilator, tmp_path, no_cfg_env):
    from opentpu.host import diag
    out = tmp_path / "diag.json"
    assert diag.main(["--sim", "--only", "regs,mem", "--json", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert rep["failed"] == 0 and not rep["hints"]
    st = {r["name"]: r["status"] for r in rep["rows"]}
    assert st["channel 1 data bits (walking 1 / 0)"] == "PASS"
    assert st["SNAP and the free-running counters"] == "PASS"
    assert st["PCIe link"] == "SKIP"


class LaneFault(FakeTransport):
    """A card whose channel 1 DQ44 (byte lane 5, bit 4) reads inverted."""

    def mem_read(self, ch, off, n, out=None):
        d = super().mem_read(ch, off, n).copy()
        if ch == 1:
            d[(8 - off % 8 + 5) % 8::8] ^= 0x10
        if out is not None:
            out[:] = d
            return out
        return d


def test_diag_names_the_failing_byte_lane(tmp_path, capsys, no_cfg_env):
    from opentpu.host import diag
    card = LaneFault(devname="fake3")
    out = tmp_path / "diag.json"
    rc = diag.main(["--only", "mem", "--bw-mib", "1", "--json", str(out)],
                   open_transport=lambda dev: card)
    assert rc == 1
    rep = json.loads(out.read_text())
    st = {r["name"]: r["status"] for r in rep["rows"]}
    assert st["channel 1 data bits (walking 1 / 0)"] == "FAIL"
    assert st["channel 1 random blocks"] == "FAIL"
    assert st["channel 0 data bits (walking 1 / 0)"] == "PASS"
    assert st["channel 0 random blocks"] == "PASS"
    assert "channel 1 byte lane 5 errors -> DQ[47:40] pinout / calibration of that lane; " \
           "only DQ44 (stuck or shorted bit)" in rep["hints"]
    assert not any("channel 0" in h for h in rep["hints"])
    text = capsys.readouterr().out
    assert "does not work:" in text and "DQ[47:40]" in text


def test_diag_hints_from_the_pattern_of_failures():
    from opentpu.host.diag import FAIL, PASS, Row, diagnose

    def rows(**groups):
        return [Row("isa", f"{g} {k}", st, "", group=g) for g, st in groups.items()
                for k in range(3)]
    h = diagnose(rows(mxu=FAIL, vpu=PASS, dma=PASS, control=PASS))
    assert any("MXU / DSP path" in x for x in h)
    h = diagnose(rows(mxu=PASS, vpu=PASS, dma=PASS, **{"vpu-new": FAIL}))
    assert h == ["only RDOT / OUTER / LOG2 fail: a bitstream built before ddec900 (Qwen3 and "
                 "LFM2 run; Qwen3.5 does not)"]
    h = diagnose(rows(mxu=FAIL, vpu=FAIL, dma=FAIL, control=FAIL))
    assert h[0].startswith("every program fails")


# ------------------------------------------------------------------------------ otpu-chat TUI
class StubEngine:
    """Engine stand-in: one step per token, a prompt up to 4 tokens per run, a fixed cycle
    count per run."""

    def __init__(self, cap=64, cycles=2_000_000):
        self.spec = types.SimpleNamespace(eos={0})
        self.cap, self.pos, self.stats, self.cycles = cap, 0, [], cycles
        self.backend = types.SimpleNamespace()
        self.cfg = board_config()

    def step(self, t, on_start=None):
        assert self.pos < self.cap, "KV cache full"   # as Engine.step
        if on_start is not None:
            on_start()
        time.sleep(0.002)
        self.pos += 1
        self.stats.append({"cycles": self.cycles})
        return np.zeros(8, np.float32)

    def prefill_chunks(self, tokens):
        tokens = list(tokens)
        for i in range(0, len(tokens), 4):
            part = tokens[i:i + 4]
            assert self.pos + len(part) <= self.cap, "KV cache full"
            time.sleep(0.002)
            self.pos += len(part)
            self.stats.append({"cycles": self.cycles})
            yield part, (np.zeros(8, np.float32) if i + 4 >= len(tokens) else None)

    def reset(self):
        self.pos = 0


class StubTok:
    """One token per letter (a = 1 ... z = 26): the template of a longer history extends the
    shorter one's, as a real chat template does."""

    def apply_chat_template(self, history, add_generation_prompt, enable_thinking, tokenize):
        return [ord(c) - 96 for m in history for c in m["content"]]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(96 + i) for i in ids)


def _stub_chat(n_out=6, clock=100.0, max_new=32, cap=64):
    from opentpu.host.chat import Chat
    seq = iter([5] * n_out + [0] * 100)
    return Chat(StubEngine(cap=cap), StubTok(), False, lambda logits, ctx: next(seq), max_new,
                clock_mhz=clock)


def test_chat_turn_metrics_and_plain_line():
    chat = _stub_chat()
    reply, t = chat.ask("hi")
    assert reply == "eeeeee" and t.prefill_tokens == 2 and t.gen_tokens == 6 and t.end == "eos"
    assert t.decode_steps == 6 and t.context == 8 and t.ttft_s > 0
    assert t.prefill_dev_tok_s == pytest.approx(100.0)       # both prompt tokens in one run
    assert t.decode_dev_tok_s == pytest.approx(50.0)
    assert t.mcycles_per_token == pytest.approx(2.0) and t.decode_tok_s > 0
    line = t.line()
    assert line.startswith("[TTFT ") and "prefill 2 tokens" in line and "(device 50.0)" in line
    assert "decode 6 tokens" in line and "context 8/64" in line
    reply, t2 = chat.ask("again")                 # the KV cache keeps the first turn
    assert not t2.restarted and t2.prefill_tokens == 5 and t2.gen_tokens == 0
    assert chat.session.turns == 2 and chat.session.tokens_in == 7
    chat.reset()                                  # /reset forgets history and KV
    assert chat.eng.pos == 0 and chat.history == [] and chat.fed == []
    _, t = _stub_chat(clock=0.0).ask("hi")        # no device clock (ISA): wall numbers only
    assert t.mcycles_per_token is None and t.decode_dev_tok_s is None
    assert "device" not in t.line() and "Mcycles" not in t.line()


def test_chat_max_new_resume_and_cap():
    chat = _stub_chat(n_out=10, max_new=4)
    reply, t = chat.ask("hi")
    assert reply == "eeee" and t.end == "max_new" and chat.can_resume
    assert "stopped at max_new" in t.line()
    reply, t = chat.resume()                      # the same reply grows
    assert reply == "eeeeeeee" and t.end == "max_new" and t.prefill_tokens == 0 and t.ttft_s is None
    assert chat.history[-1] == {"role": "assistant", "content": "eeeeeeee"}
    reply, t = chat.resume()
    assert reply == "eeeeeeeeee" and t.end == "eos" and not chat.can_resume
    assert chat.eng.pos == len(chat.fed) == 12
    chat = _stub_chat(n_out=100, cap=10)          # the reply fills the KV cache
    reply, t = chat.ask("hi")
    assert t.end == "cap" and chat.eng.pos == 10 and "context full" in t.line()
    pos, n_hist = chat.eng.pos, len(chat.history)
    reply, t = chat.ask("more")                   # does not fit: refused, the cache is kept
    assert t.end == "cap" and reply == "" and t.prefill_tokens == 0
    assert chat.eng.pos == pos and len(chat.history) == n_hist


def test_chat_tui_reply_parts_split_at_whole_blocks():
    pytest.importorskip("textual")
    from opentpu.host.chat_tui import Reply
    r = Reply()
    r.PART_BLOCKS = 2
    r.text = "a\n\nb\n\n"
    assert r.split("c") == 0                      # after a blank line, the text at hand
    assert r.split("\n") is None                  # what follows is not known yet
    r.text = "a\n\nb"
    assert r.split("\n\nc") == 2 and r.split("\n\n  c") is None   # not an indented continuation
    r.text = "a\n\n```\nx\n\n"
    assert r.split("y\n```\n\nz") == len("y\n```\n\n")   # not inside the code fence
    r.text = "a\n\nb"
    assert r.split("c") is None
    r.PART_BLOCKS = 99
    assert r.split("\n\nc") is None              # a short reply stays one part


def test_chat_tui_long_reply_in_parts_reads_the_same():
    pytest.importorskip("textual")
    import asyncio

    from opentpu.host.chat import Turn
    from opentpu.host.chat_tui import ChatApp
    from textual.widgets import Markdown
    from textual.widgets._markdown import MarkdownFence
    blocks = []
    for i in range(12):
        blocks += [f"## Part {i}", f"Paragraph {i}.", "```\nx = 1\n\ny = 2\n```",
                   "- item\n\n  continued"]
    text = "\n\n".join(blocks) + "\n"
    pieces = [text[i:i + 3] for i in range(0, len(text), 3)]
    meta = {"model": "stub", "backend": "board", "device": "d", "short": "x",
            "bitstream": ["a", "b"], "sampling": {}, "dram": None}

    async def go():
        app = ChatApp(_stub_chat(), meta)
        async with app.run_test(size=(100, 500)) as pilot:
            await pilot.pause(0.1)
            turn = Turn(clock_mhz=100, cap=64, context=0)
            for p in pieces:
                app._post(p, turn)
                await asyncio.sleep(0)
            await app._done(None)
            await pilot.pause(0.2)
            parts = list(app.query(".reply Markdown").results(Markdown))
            fences = [f for m in parts for f in m.query(MarkdownFence)]
            return len(parts), [f.code for f in fences], _shot(app)
    n, codes, shot = asyncio.run(go())
    assert n >= 2                                 # 48 blocks, parts of Reply.PART_BLOCKS
    assert codes == ["x = 1\n\ny = 2"] * 12       # no fence cut at its blank line
    lines = [ln.strip() for ln in shot.splitlines()]
    i = lines.index("Part 3")                     # one blank line between blocks, as before
    assert lines[i - 2:i + 3] == ["continued", "", "Part 3", "", "Paragraph 3."]
    assert "⏺ Part 0" in lines and all(f"Part {i}" in lines for i in range(1, 12))


def _shot(app) -> str:
    import io

    from rich.console import Console
    c = Console(width=app.size.width, height=app.size.height, file=io.StringIO(), record=True)
    c.print(app.screen._compositor.render_update(full=True))
    return c.export_text(styles=False)


def test_chat_tui_shows_the_live_numbers(tmp_path):
    pytest.importorskip("textual")
    import asyncio

    from opentpu.host.chat_tui import ChatApp, _meter, status_line
    from textual.widgets import OptionList, Static
    assert _meter(10, 100)[1] == "bright_black" and _meter(80, 100)[1] == "#E0A030"
    assert _meter(95, 100) == ("▰" * 8, "#E05050")
    from textual.app import App                   # e.g. App._flush writes captured prints
    own = {n for n, v in vars(ChatApp).items() if n[:1] == "_" and n[:2] != "__"
           and getattr(v, "__qualname__", "").startswith("ChatApp.")}
    assert own and not own & set(dir(App))
    chat = _stub_chat(n_out=6, max_new=4)
    meta = {"model": "stub", "backend": "board", "device": "/dev/xdma0", "short": "board 100 MHz",
            "bitstream": ["D=128 MCOLS=2 LANES=8", "build 74d48591, 100 MHz"],
            "sampling": {"temperature": 0.7}, "dram": None}

    async def wait(app, pilot):
        for _ in range(300):
            await pilot.pause(0.02)
            if not app._busy:
                return

    async def go():
        app = ChatApp(chat, meta)
        async with app.run_test(size=(170, 40)) as pilot:
            await pilot.pause(0.1)
            r = {"welcome": _shot(app)}
            await pilot.press(*"hi", "enter")
            await wait(app, pilot)
            await pilot.pause(0.1)
            r["status"] = str(app.query_one("#status", Static).render())
            r["cut"] = _shot(app)
            await pilot.press(*"/co")                 # the popup, Enter takes /continue
            r["popup"] = app.query_one("#cmds", OptionList).display
            await pilot.press("enter")
            await wait(app, pilot)
            await pilot.pause(0.1)
            r["done"] = _shot(app)
            await pilot.press(*"/st", "enter")
            await pilot.pause(0.1)
            r["stats"] = [str(w.render()) for w in app.query(".block")]
            await pilot.press("ctrl+s")
            await pilot.pause(0.1)
            r["panel"] = str(app.query_one("#panel", Static).render())
            app.save_screenshot(str(tmp_path / "shot.svg"))
            return r
    r = asyncio.run(go())
    assert "openTPU chat" in r["welcome"] and "74d48591" in r["welcome"]
    assert "palette" not in r["welcome"] and "esc interrupt" in r["welcome"]
    s = r["status"]
    assert "stub · board 100 MHz" in s and "TTFT" in s and "decode" in s and "(dev 50.0)" in s
    assert "2.00 Mcyc/tok" in s and "ctx 6/64" in s
    narrow = str(status_line(meta, chat, chat.last, 70))   # drops the rest, keeps the context
    assert len(narrow) <= 70 and "decode" in narrow and "ctx 8/64" in narrow
    assert "⏺ eeee" in r["cut"] and "stopped at max_new=4 tokens · /continue" in r["cut"]
    assert r["popup"]
    assert "⏺ eeeeee" in r["done"] and "max_new=4" not in r["done"]   # the marker is gone
    assert any("session" in b and "2 tokens in, 6 out" in b for b in r["stats"])
    assert "DRAM" not in r["panel"] and "KV context" in r["panel"]
    assert (tmp_path / "shot.svg").stat().st_size > 1000


def test_setup_pcie_package():
    """otpu-setup's script and files ship together and agree: the module tag the patch adds is
    the version the script checks for, the dkms.conf is a template, the rules match the card."""
    import re
    from opentpu.host import pcie_setup
    host = os.path.join(ROOT, "opentpu", "host")
    script = open(pcie_setup.SCRIPT).read()
    ver = re.search(r"^VER=(\S+)", script, re.M).group(1)
    commit = re.search(r"^XDMA_COMMIT=([0-9a-f]{40})", script, re.M).group(1)
    patch = open(os.path.join(host, "pcie", "xdma-otpu.patch")).read()
    assert f'MODULE_INFO(otpu, "{ver}");' in patch and ver.startswith(commit[:7])
    assert '"@VERSION@"' in open(os.path.join(host, "pcie", "dkms.conf")).read()
    rules = open(os.path.join(host, "pcie", "59-otpu-xdma.rules")).read()
    assert 'ATTR{device}=="0x7028"' in rules and 'ATTR{subsystem_device}=="0x4f54"' in rules
    bd = open(os.path.join(ROOT, "boards", "ypcb-00338", "vivado", "bd.tcl")).read()
    assert "CONFIG.pf0_device_id {7028}" in bd and "CONFIG.pf0_subsystem_id {4F54}" in bd
    subprocess.run(["bash", "-n", str(pcie_setup.SCRIPT)], check=True)
    r = subprocess.run(["bash", str(pcie_setup.SCRIPT), "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "--rescan" in r.stdout and "set -euo" not in r.stdout


def test_busy_card_is_one_line_not_a_traceback(tmp_path, monkeypatch, capsys):
    """otpu-selftest / -diag / -chat / -lens on a card another process holds: one line on stderr
    naming the holder and OTPU_LOCK_WAIT, exit status 3."""
    from opentpu.host import runstate as rs
    from opentpu.host import selftest
    monkeypatch.setenv("OTPU_RUN_DIR", str(tmp_path))
    held = rs.DeviceLock("xdmaB")
    assert selftest.main(["--dev", str(tmp_path / "xdmaB")]) == 3
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and "in use by process" in err[0] and "OTPU_LOCK_WAIT" in err[0]
    held.release()


def test_xdma_transport_locks_before_opening(tmp_path, monkeypatch):
    """A transport waiting for a busy card holds no file on it (a rescan by the holder would be
    refused): the lock comes first, so a busy device raises DeviceBusy, not an open error."""
    from opentpu.host import runstate as rs
    from opentpu.host.board import XdmaTransport
    monkeypatch.setenv("OTPU_RUN_DIR", str(tmp_path))
    held = rs.DeviceLock("xdmaT")
    with pytest.raises(rs.DeviceBusy):
        XdmaTransport(str(tmp_path / "xdmaT"))
    held.release()
    with pytest.raises(FileNotFoundError):          # free: now it opens (no such device here)
        XdmaTransport(str(tmp_path / "xdmaT"))
    rs.DeviceLock("xdmaT", wait=0).release()        # and the failed open released the lock


def test_device_lock_waits_for_a_busy_card(tmp_path, monkeypatch):
    """OTPU_LOCK_WAIT: a second runner waits for the lock instead of failing at once; it gets the
    lock once the first releases it, and still fails with DeviceBusy when the wait runs out."""
    import threading
    from opentpu.host import runstate as rs
    monkeypatch.setenv("OTPU_RUN_DIR", str(tmp_path))
    first = rs.DeviceLock("w0")
    with pytest.raises(rs.DeviceBusy):
        rs.DeviceLock("w0", wait=0)
    threading.Timer(0.5, first.release).start()
    monkeypatch.setenv("OTPU_LOCK_WAIT", "5")
    second = rs.DeviceLock("w0")                    # waits ~0.5 s, then holds it
    with pytest.raises(rs.DeviceBusy):
        rs.DeviceLock("w0", wait=1)
    second.release()
    assert rs.hold_main(["--dev", "/dev/w0", "--wait", "1", "--", "true"]) == 0
    code = ("from opentpu.host import runstate as rs; rs.DeviceLock('w0', wait=0); "
            "print('inner ok')")                     # a tool inside otpu-lock: no second lock
    assert rs.hold_main(["--dev", "/dev/w0", "--", sys.executable, "-c", code]) == 0


def test_board_pipelined_dma(monkeypatch):
    """Large Board.write / read on a card run the DMA calls in a worker thread, piece by piece
    (board.PIPE); the bytes and the interleave are the same as the one-call path."""
    from opentpu.host import board
    monkeypatch.setattr(board, "PIPE", 1024)

    class Threaded(FakeTransport):
        threaded = True
    b = Board(Threaded(ch_bytes=1 << 16), check=False, lock=False)
    ref = Board(FakeTransport(ch_bytes=1 << 16), check=False, lock=False)
    data = np.random.default_rng(3).integers(0, 256, 10_000, dtype=np.uint8)
    for bb in (b, ref):
        bb.write(640, data)
    assert np.array_equal(b.t.ch[0], ref.t.ch[0]) and np.array_equal(b.t.ch[1], ref.t.ch[1])
    assert np.array_equal(b.read(640, len(data)), data)
    assert np.array_equal(b.read(600, 5000), ref.read(600, 5000))
    b.close()


def test_xdma_transport_writes_whole_beats(tmp_path, monkeypatch):
    """XdmaTransport sends only whole 64-byte beats (sub-beat DMA writes can wedge the card):
    an unaligned range is widened and its edge beats merged on the host; placement bounces
    keep the bytes. Driven against a sparse file in place of the XDMA device nodes."""
    from opentpu.host import board
    f = tmp_path / "card"
    f.write_bytes(b"")
    os.truncate(f, 1 << 16)
    fd = os.open(f, os.O_RDWR)
    t = board.XdmaTransport.__new__(board.XdmaTransport)
    t.h2c = t.c2h = fd
    writes = []
    pw = os.pwrite

    def logged(fd_, mv, off):
        writes.append((off, len(mv)))
        return pw(fd_, mv, off)
    monkeypatch.setattr(board.os, "pwrite", logged)
    ref = np.random.default_rng(5).integers(0, 256, 4096, dtype=np.uint8)
    t.mem_write(0, 0, ref)
    for off, n in ((70, 5), (130, 63), (1000, 200), (4000, 96), (64, 64)):
        d = np.arange(n, dtype=np.uint8) + 1
        t.mem_write(0, off, d)
        ref[off:off + n] = d
    assert all(o % 64 == 0 and n % 64 == 0 for o, n in writes)
    assert np.array_equal(t.mem_read(0, 0, 4096), ref)
    os.close(fd)
