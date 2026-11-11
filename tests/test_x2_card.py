"""First-card session preparation without cards or Verilator: the actual qual CLI over
IsaCards, including the tiny Engine's device generation. Faults must leave a JSON failure,
stop further launches, and never become a PASS through a reset or a cleared sticky.
"""
from dataclasses import replace
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from opentpu import target as T
from opentpu.host import regs as R
from opentpu.host.board import BoardBackend
from opentpu.host.cards import IsaCards
from tools import x2_card as X

ROOT = Path(__file__).resolve().parents[1]



@pytest.fixture
def pair(monkeypatch):
    t = T.load("bcu1525-x2:st16")
    t = replace(t, arch=dict(t.arch, DRAM_BYTES=1 << 24, BHW=0),
                platform=dict(t.platform, channel_bytes=(1 << 24) // 8))
    load = T.load
    monkeypatch.setattr(T, "load", lambda name=None: t if name == "bcu1525-x2:st16" else load(name))
    cfg = T.config(t)
    return cfg, IsaCards(cfg, cross=True)


def invoke(tmp_path, pair, *, steps="a,b,c,d", mutate=None, wrong_token=False):
    from opentpu.llm.qwen3 import Engine, Spec
    cfg, sh = pair
    # A small deterministic model without torch / transformers or checkpoint I/O. This is
    # the same real Engine / BoardBackend path as test_cards, over four ISA slices.
    spec = Spec(512, 1, 4, 4, 128, 512, 128)
    rng = np.random.default_rng(3)
    W = {"model.embed_tokens.weight": rng.normal(0, .1, (128, 512)).astype(np.float32),
         "model.norm.weight": np.ones(512, np.float32)}
    for name in ("input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"):
        W[f"model.layers.0.{name}.weight"] = np.ones(128 if "_norm" in name else 512, np.float32)
    for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                 "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
        W[f"model.layers.0.{name}.weight"] = rng.normal(0, .03, (512, 512)).astype(np.float32)
    # Spec's default qk norm matches the checkpoint's norm vectors above.
    prompt = [11, 22, 33]
    next_tokens, generated = [1, 1, 1], [1, 1, 1]
    if "d" in steps and (mutate is None or wrong_token):
        ref = Engine(spec, W, cap=256, cfg=cfg, wformat="int8", resident=True)
        next_tokens = [int(np.argmax(ref.step(tok))) for tok in prompt]
        generated = ref.generate_card(next_tokens[-1], 3, stop_ids=[])
    expect = tmp_path / "expect.json"
    expect.write_text(json.dumps({"qwen3_0_6b": {"prompt": prompt, "next_tokens": next_tokens,
                                                 "generated": generated}}))
    if wrong_token:
        case = json.loads(expect.read_text())
        case["qwen3_0_6b"]["generated"][0] ^= 1
        expect.write_text(json.dumps(case))
    if mutate:
        mutate(sh)

    def engine(cfg, ct):
        return Engine(spec, W, cap=256, cfg=cfg, wformat="int8", resident=True,
                      backend=lambda c, imgs: BoardBackend(c, imgs, transport=ct, status=False))
    out = tmp_path / "qual.json"
    rc = X.main(["--dev", "/dev/xdma0+/dev/xdma1", "--target", "bcu1525-x2:st16",
                 "--expect", str(expect), "--json", str(out), "--steps", steps,
                 "--coll-wd", "12345"], cards=sh.cards, engine_factory=engine, cfg=cfg)
    return rc, json.loads(out.read_text()), sh


def test_pass(tmp_path, pair):
    # IsaCards models hello and counters, but has no PING engine: provide a completed reply
    # on this fake only. Real CLI must poll the actual valid bit on all four endpoints.
    for card in pair[1].cards:
        read = card._link_read
        card._link_read = lambda off, read=read: (1 << 31 | 40) if off in R.R_X2_PING else read(off)
    rc, out, sh = invoke(tmp_path, pair)
    assert rc == 0 and out["ok"]
    assert sh.runs == 6              # BAR, GATHER, three decode launches, one generate
    e = [s for s in out["steps"] if s["step"] == "e"]
    assert len(e) == 6 and [s["delta"] for s in e[:2]] == [[3, 3], [1, 1]]
    assert all(s["delta"][0] == s["delta"][1] > 0 for s in e)
    assert all(k.watchdog == 12345 and not k.running for k in sh.cards)
    assert out["steps"][1]["cable_map"][0]["peer_port"] == 1
    assert next(s for s in out["steps"] if s["step"] == "c" and s["run"] == "gather")[
        "checked_bytes_per_slice"] == 8192


@pytest.mark.parametrize("fault,reason", [("tokens", "token mismatch"), ("crc", "link error"),
    ("build", "build_id"), ("caps", "CAPS2 bit15"), ("soft", "link error"),
    ("ping", "ping timed out"), ("gather", "gathered data mismatch"), ("seq", "X2_SEQ")])
def test_faults(tmp_path, pair, fault, reason):
    cfg, sh = pair
    for card in sh.cards:
        read = card._link_read
        card._link_read = lambda off, read=read: (1 << 31 | 40) if off in R.R_X2_PING else read(off)
    def mutate(sh):
        if fault == "build":
            sh.cards[1].build_id ^= 1
        if fault in ("crc", "soft", "ping", "caps"):
            for card in sh.cards:
                read = card.reg_read
                card.reg_read = lambda off, read=read: (read(off) & ~R.CAP2_XCARD if
                    fault == "caps" and off == R.R_CAPS2 else 1 if fault == "crc" and
                    off == R.R_X2_CRC[0] else 1 if fault == "soft" and
                    off == R.R_X2_COUNTERS[1] + 12 else 0 if fault == "ping" and
                    off in R.R_X2_PING else read(off))
        if fault in ("gather", "seq"):
            started = sh.started
            def broken():
                started()
                if fault == "seq" and sh.runs:
                    sh.cards[1].seq ^= 1
                if fault == "gather" and sh.runs == 2:
                    sh.dram[3][0x4000] ^= 1
            sh.started = broken
    rc, out, sh = invoke(tmp_path, pair, mutate=mutate, wrong_token=fault == "tokens")
    assert rc == 1 and not out["ok"] and reason in out["error"]
    if fault in ("crc", "soft", "ping", "build", "caps"):
        assert sh.runs == 0
    assert not any(c.running for c in sh.cards)


def test_identity_only_does_not_reset_links(tmp_path, pair):
    rc, out, sh = invoke(tmp_path, pair, steps="a")
    assert rc == 0 and sh.runs == 0
    assert all(c.topo == 1 << 8 for c in sh.cards)


def test_solo_target_is_platform7_two_slices():
    from tools.x2_session.bringup import local_target
    t = local_target()
    assert (t.platform_id, t.arch["S"], T.cards(t)) == (7, 2, 1)
    assert T.arch_hash(T.build_config(t)) == 0xE331F79C


@pytest.mark.parametrize("case,ok", [("slot", True), ("other-link", False), ("second-fail", False),
    ("selftest", False), ("tripped", False), ("acc", False), ("unfinished", False), ("no-diag", False)])
def test_known_link(tmp_path, case, ok):
    # A card in a slot that trains Gen2 x4 on every image: its up stops on the diag's PCIe row alone;
    # only that exact row, with everything else passing, may become rc 0.
    from tools.x2_session.bringup import known_link
    link = "5.0 GT/s PCIe x4"
    rows = [{"section": "platform", "name": "PCIe link", "status": "FAIL",
             "msg": f"{'2.5 GT/s PCIe x4' if case == 'other-link' else link} (the card's 8.0 "
                    "GT/s PCIe x16: expected its speed and width)"},
            {"section": "platform", "name": "power estimate", "status": "SKIP", "msg": ""},
            {"section": "mem", "name": "channel 0 address bits", "status":
             "FAIL" if case == "second-fail" else "PASS", "msg": ""}]
    wg = {"acc": {"ok": case != "acc"}, "diag_rc": 1, "selftest_rc": int(case == "selftest"),
          "guard": {"tripped": "die 80 C" if case == "tripped" else None, "booted": False},
          "status_end": {}}
    if case == "unfinished":
        del wg["status_end"]
    (tmp_path / "wg.json").write_text(json.dumps(wg))
    if case != "no-diag":
        (tmp_path / "diag.json").write_text(json.dumps(
            {"rows": rows, "failed": sum(r["status"] == "FAIL" for r in rows)}))
    assert known_link(tmp_path, link) is ok


@pytest.mark.parametrize("die,reg,stops", [(79.99, 79.99, False), (80, 60, True),
                                         (60, 80, True), (81, 81, True)])
def test_thermal_threshold_in_safe(monkeypatch, die, reg, stops):
    from tools.x2_session.safety import SessionGuard
    guard = object.__new__(SessionGuard)
    events = []
    guard.last = {"die_c": die, "run": False}
    guard.reg_max = lambda: reg
    guard.trip = events.append
    monkeypatch.setattr(X.G.Guard, "step", lambda *a, **kw: (1, None))
    assert guard.step(0, 0) == (1, None)
    assert bool(events) == stops
    guard.emergency("test emergency")
    assert events[-1] == "test emergency"


@pytest.mark.parametrize("fails", [False, True])
def test_safe_before_backend_releases_locks(tmp_path, monkeypatch, fails):
    from types import SimpleNamespace
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    events = []
    class Card:
        def __init__(self, dev):
            self.dev = dev
        def close(self):
            events.append("close-" + self.dev)
        def reg_write(self, off, val):
            assert (off, val) == (R.R_CTRL, 0)
            events.append("stop-" + self.dev)
    monkeypatch.setattr(X, "open_transport", Card)
    def power(cards, out, stack):
        for c in range(2):
            stack.callback(events.append, f"safe-{c}")
        return []
    def qual(*args):
        args[-1].append(SimpleNamespace(close=lambda: events.append("backend-close"),
            board=SimpleNamespace(abort=lambda wait: events.append("drain-backend"))))
        if fails:
            raise X.QualError("injected run failure")
    monkeypatch.setattr(X, "power_up", power)
    monkeypatch.setattr(X, "qualify", qual)
    out = tmp_path / "otpu-build/bcu-wg/runs/qual.json"
    assert X.main(["--dev", "/dev/xdma0+/dev/xdma1", "--json", str(out), "--steps", "a"]) \
        == int(fails)
    assert events == ["drain-backend", "stop-/dev/xdma0", "stop-/dev/xdma1", "safe-1", "safe-0", "backend-close",
                      "close-/dev/xdma1", "close-/dev/xdma0"]


def test_lead_hold_borrowed_by_cards_child(tmp_path):
    import os
    import sys
    child = '''
import os
from opentpu import target as T
from opentpu.host.board import Board
from opentpu.host.cards import CardsTransport, IsaCards
from opentpu.host.runstate import DeviceLock, DeviceBusy
sh = IsaCards(T.config(T.load("bcu1525-x2:st16"), DRAM_BYTES=1 << 22))
for c, card in enumerate(sh.cards):
    card.devname = "xdma" + str(c)
    card._otpu_lock = DeviceLock(card.devname)
assert sh.cards[0]._otpu_lock.holder == int(os.environ["OTPU_LOCK_PID"])
ct = CardsTransport(sh.cards)
b = Board(ct)
b.close()
os.environ.pop("OTPU_LOCK_HELD")
try:
    DeviceLock("xdma0")
except DeviceBusy:
    pass
else:
    raise AssertionError("child released the caller's hold")
print("borrowed hold intact")
'''
    parent = '''
import os, subprocess, sys
from opentpu.host.runstate import DeviceLock
lock = DeviceLock("xdma0")
env = dict(os.environ, OTPU_LOCK_HELD="xdma0", OTPU_LOCK_PID=str(os.getpid()))
try:
    r = subprocess.run([sys.executable, "-c", sys.argv[1]], env=env)
    assert r.returncode == 0
finally:
    lock.release()
'''
    r = subprocess.run([sys.executable, "-c", parent, child], cwd=ROOT,
                       env=dict(os.environ, PYTHONPATH=str(ROOT), OTPU_RUN_DIR=str(tmp_path / "locks")),
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "borrowed hold intact" in r.stdout


@pytest.fixture
def cutpack(tmp_path, pair):
    from types import SimpleNamespace
    from opentpu import isa as I
    from tools.x2_cutpack import build_pack
    cfg, _ = pair
    # A hand-made configuration-only cut through the production pack builder, including a
    # run-time LOOP (three issued BARs, one static BAR) and nonzero initial/logit words.
    img = SimpleNamespace(low_end=0x10000, nbytes=0x401000, high=True, wlo=0x400000,
                          io={"logits": 0x4000}, v_loc=4)
    p = [I.loop(1, 3), I.bar(), I.ld(0x2000, 0, 512), I.st(0x4000, 0, 512),
         I.ld(0x400000, 512, 512), I.st(0x5000, 512, 512), I.halt()]
    present = [(0x2000, np.arange(128, dtype=np.float32).view(np.uint8))]
    path = tmp_path / "tiny-pack"
    build_pack(path, T.load("bcu1525-x2:st16"), cfg, img, [p] * 4, 0x20000, [7, 60],
               present, name="tiny", model="q36", baseline={"cycles": 100,
               "one_card_cycles": 200, "phase_cycles": {"prologue": 10, "HEAD": 20}})
    return path


def invoke_cut(tmp_path, pair, cutpack, *, fault=None, junk=False, steps=None):
    cfg, sh = pair
    for card in sh.cards:
        read = card._link_read
        card._link_read = lambda off, read=read: (1 << 31 | 40) if off in R.R_X2_PING else read(off)
    if junk:
        for d in sh.dram:
            d[:] = 0xA5
    started = sh.started
    def broken():
        before = sh.runs
        started()
        if sh.runs == before:
            return
        if fault == "logits":
            sh.dram[2][0x4000 + 2 * 16] ^= 1
        if fault == "sha":
            sh.dram[1][0x9000] ^= 1
        if fault == "instructions":
            sh.icount[3] += 1
        if fault == "collectives":
            for card in sh.cards:
                card.seq += 1
        # Make sure the next repetition also clears changed low memory, rather than relying
        # on fresh IsaCards or only clearing once for the whole multi-repetition session.
    sh.started = broken
    out = tmp_path / "perf.json"
    argv = ["--dev", "/dev/xdma0+/dev/xdma1", "--json", str(out), "--cuts", str(cutpack)]
    argv += ["--steps", steps or "a,b"]
    rc = X.main(argv, cards=sh.cards, cfg=cfg)
    return rc, json.loads(out.read_text()), sh


def test_cut_pass(tmp_path, pair, cutpack, monkeypatch):
    # Generation already happened. Prove the card path cannot invoke the compiler, wait
    # pass or ISA reference; the functional fake's own machine still executes its programs.
    from opentpu import waits
    from opentpu.host.board import Board
    from tools import board_perf
    def forbidden(*a, **kw):
        raise AssertionError("card host performed compilation")
    monkeypatch.setattr(Board, "load", forbidden)
    monkeypatch.setattr(board_perf, "decode_case", forbidden)
    monkeypatch.setattr(waits, "entries", forbidden)
    rc, out, sh = invoke_cut(tmp_path, pair, cutpack)
    assert rc == 0 and out["ok"] and sh.runs == 3
    assert out["requested_steps"] == ["a", "b", "f"]
    e = [s for s in out["steps"] if s["step"] == "e"]
    assert len(e) == 3 and all(s["delta"] == [3, 3] for s in e)
    f = [s for s in out["steps"] if s["step"] == "f"]
    assert f[-1]["min_cycles"] == f[-1]["median_cycles"] == 0  # functional fake, no timing
    assert len(f[1]["low_sha256"]) == 4 and f[1]["projection"] is None
    assert all(not c.running for c in sh.cards)


def test_cut_clears_leftovers(tmp_path, pair, cutpack):
    rc, out, sh = invoke_cut(tmp_path, pair, cutpack, junk=True, steps="f")
    assert rc == 0 and out["ok"] and sh.runs == 3
    assert all(not d[0x9000:0xA000].any() for d in sh.dram)
    assert all(not d[0x400000:0x401000].any() for d in sh.dram)
    assert all(d[0x30000:0x31000].min() == 0xA5 for d in sh.dram)  # unused gap preserved
    assert any(s["step"] == "b" for s in out["steps"])  # f requires link qualification


@pytest.mark.parametrize("fault,reason", [("logits", "logits mismatch"),
    ("sha", "low-area sha256 mismatch"), ("instructions", "instruction counts"),
    ("collectives", "X2_SEQ")])
def test_cut_mismatches_stop(tmp_path, pair, cutpack, fault, reason):
    rc, out, sh = invoke_cut(tmp_path, pair, cutpack, fault=fault)
    assert rc == 1 and not out["ok"] and reason in out["error"]
    assert sh.runs == 1 and all(not c.running for c in sh.cards)
    assert len([s for s in out["steps"] if s["step"] == "e"]) == 1


def test_cut_reinitialize_each_rep(tmp_path, pair, cutpack, monkeypatch):
    original = X.cut_projection
    def dirty(meta, stats, arrays, cfg):
        result = original(meta, stats, arrays, cfg)
        for d in pair[1].dram:
            d[0x9000:0xA000] = 0x99
        return result
    monkeypatch.setattr(X, "cut_projection", dirty)
    rc, out, sh = invoke_cut(tmp_path, pair, cutpack)
    assert rc == 0 and out["ok"] and sh.runs == 3


def test_cut_pack_corrupt(tmp_path, pair, cutpack):
    f = cutpack / "arrays.npz"
    f.write_bytes(f.read_bytes() + b"corrupt")
    rc, out, sh = invoke_cut(tmp_path, pair, cutpack)
    assert rc == 1 and "arrays sha256 mismatch" in out["error"] and sh.runs == 0


def test_cut_projection_labels_fraction_fallback(cutpack):
    from tools.x2_cutpack import load_pack
    meta, arrays = load_pack(cutpack)
    cfg = T.config(T.load("bcu1525-x2:st16"))
    result = X.cut_projection(meta, {"cycles": 120}, arrays, cfg)
    assert result["delta_cosim_percent"] == pytest.approx(20)
    assert result["one_card_speedup"] == pytest.approx(200 / 120)
    assert result["phase_cycles"] == {"prologue": 12, "HEAD": 24}
    assert result["phase_source"] == "co-sim span fractions (no trace)"
    assert result["projection"]["full_tok_s_150mhz"] > 0


def test_cut_projection_complete_trace(cutpack):
    from tools.x2_cutpack import load_pack
    from opentpu import hwtrace as H
    meta, arrays = load_pack(cutpack)
    meta["pc_phases"]["2"] = "prologue"
    meta["pc_phases"]["3"] = "HEAD"
    # Two complete instructions in the real hardware trace encoding, exercising the same
    # decoder/profile/latest-end accounting as the card, without a simulated clock claim.
    records = []
    for pc, start, end in [(2, 1, 20), (3, 21, 100)]:
        w = arrays["program_0"][8 * pc:8 * pc + 8]
        ow = int(w[1]) | int(w[2]) << 32 | int(w[3]) << 64 | int(w[0] & 255) << 96
        records += [H.T_D << 60 | pc << 32 | start,
                    H.T_O << 60 | (ow & ((1 << 52) - 1)),
                    H.T_O << 60 | 1 << 52 | ow >> 52,
                    H.T_S << 60 | (start + 1), H.T_E << 60 | end]
    cfg = T.config(T.load("bcu1525-x2:st16"))
    stats = {"cycles": 120, "trace": {"lost": 0, "drop": 0,
                                      "records": np.asarray(records, np.uint64)}}
    result = X.cut_projection(meta, stats, arrays, cfg)
    assert result["phase_source"] == "card-0 trace"
    assert result["phase_cycles"] == {"prologue": 20, "HEAD": 80}
    assert result["projection"]["body_cycles"] == 20
    stats["trace"]["lost"] = 1
    fallback = X.cut_projection(meta, stats, arrays, cfg)
    assert fallback["phase_source"] == "co-sim span fractions (trace incomplete)"
    assert fallback["phase_cycles"] == {"prologue": 12, "HEAD": 24}
    # keep-first on the card: TRACE_COUNT stops at the depth, lost reads 0
    stats["trace"].update(lost=0, keep="first", count=16384, depth=16384)
    full = X.cut_projection(meta, stats, arrays, cfg)
    assert full["phase_source"] == "co-sim span fractions (trace incomplete)"
    assert full["trace"]["count"] == 16384 and "records" not in full["trace"]
    stats["trace"]["count"] = 10
    meta["pc_phases"]["3"] = "ATT"
    headless = X.cut_projection(meta, stats, arrays, cfg)
    assert headless["phase_source"] == "co-sim span fractions (trace has no HEAD span)"
    assert headless["phase_cycles"] == {"prologue": 12, "HEAD": 24}
