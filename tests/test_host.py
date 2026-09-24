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

import numpy as np
import pytest

from opentpu import isa as I
from opentpu.host import power as P
from opentpu.host import regs as R
from opentpu.host import smi
from opentpu.host.board import Board, BoardBackend, rates
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
def test_v2_info_snapshot_and_rates():
    b = Board(FakeTransport(devname=None))
    i = b.info()
    assert i["regmap"] == 2 and i["core_khz"] == 100_000 and i["build_id"] == 0x1234ABCD
    assert i["caps"] == {"trace": True, "temp": True, "trace_depth": 4096, "pq_window": 64}
    assert i["temp_c"] == pytest.approx(0x9C4 * 503.975 / 4096 - 273.15, abs=0.01)
    s0, s1 = b.snapshot(), b.snapshot()
    assert s1["snaps"] == s0["snaps"] + 1 and s1["UPTIME"] - s0["UPTIME"] == 1_000_000
    r = rates(s0, s1, i["core_khz"])
    for k in ("RUNNING", "MXU_MAC", "DRAM_WAIT"):
        assert r["util"][k] == pytest.approx(RATES[k])
    assert r["seconds"] == pytest.approx(0.01)
    assert r["dram_gbs"] == pytest.approx((RATES["DRAM_RD"] + RATES["DRAM_WR"]) * 1e6 * 64
                                          / 0.01 / 1e9)


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
    eng = Engine(spec, W, cap=256, cfg=cfg,
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
    from opentpu.host import hwlens
    monkeypatch.setitem(sys.modules, "opentpu.hwtrace", None)
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
    assert d["device"] == "/dev/fake5" and d["ok"] and d["regmap"] == 2
    assert d["bitstream"] == {"D": 128, "MCOLS": 2, "LANES": 8, "core_mhz": 100.0,
                              "build_id": 0x1234ABCD}
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
    assert "MAC 50%" in tab and "4.61 GB/s" in tab and f"pid {os.getpid()}" in tab
    assert "model m0" in tab and "W est." in tab
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
    """The Engine compiles position p + 1 while the backend runs p: same programs, same
    logits, same tokens as compiling in line."""
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
    prompt = [11, 222, 333, 44]
    assert a.generate(prompt, max_new=5) == b.generate(prompt, max_new=5)
    assert Slow.prepared >= 8
    for tok in (7, 8):                                        # logits, bit for bit
        assert np.array_equal(a.step(tok).view(np.uint32), b.step(tok).view(np.uint32))
    a.reset()
    b.reset()                                                 # stale precompile is dropped
    assert np.array_equal(a.step(9).view(np.uint32), b.step(9).view(np.uint32))


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
