"""tools/qual/refs.py: the reference cache key and the card check's fail-fast wait (no card)."""
import importlib.util
import json
import os
import socket
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("qual_refs", ROOT / "tools/qual/refs.py")
refs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refs)


def test_source_hash_ignores_host_code(tmp_path):
    pkg = tmp_path / "opentpu"
    (pkg / "host").mkdir(parents=True)
    (pkg / "compiler.py").write_text("a = 1\n")
    (pkg / "host" / "board.py").write_text("POLL = 1\n")
    h0 = refs.source_hash(pkg)
    (pkg / "host" / "board.py").write_text("POLL = 2\n")       # a poll fix: same references
    assert refs.source_hash(pkg) == h0
    (pkg / "compiler.py").write_text("a = 2\n")                 # a compiler change: new ones
    assert refs.source_hash(pkg) != h0


def _pending(kp, pid, age=0.0, host=None):
    refs.side(kp, ".pending").write_text(json.dumps(
        {"host": host or socket.gethostname(), "pid": pid, "t": time.time() - age}))


def test_wait_ref_fails_fast_without_a_job(tmp_path):
    kp = tmp_path / "r.pkl"
    t0 = time.time()
    assert "no job computing it" in refs.wait_ref(kp)
    assert time.time() - t0 < 1


def test_wait_ref_fails_fast_on_a_dead_job(tmp_path):
    kp = tmp_path / "r.pkl"
    _pending(kp, pid=2 ** 22 + 12345)                           # no such process
    assert "died" in refs.wait_ref(kp)
    _pending(kp, pid=os.getpid(), age=refs.STALE + 5)           # alive but no heartbeat
    assert "died" in refs.wait_ref(kp)


def test_wait_ref_reports_a_failed_job(tmp_path):
    kp = tmp_path / "r.pkl"
    refs.side(kp, ".failed").write_text("killed by signal 9 (the OOM killer?)")
    assert "OOM" in refs.wait_ref(kp)


def test_wait_ref_waits_for_a_live_job(tmp_path, monkeypatch):
    kp = tmp_path / "r.pkl"
    _pending(kp, pid=os.getpid())
    nap = time.sleep
    monkeypatch.setattr(refs.time, "sleep", lambda s: nap(0.05))
    threading.Timer(0.3, lambda: kp.write_bytes(b"x")).start()
    assert refs.wait_ref(kp) is None


@pytest.mark.skipif(not (ROOT / "models/LFM2.5-230M").exists(), reason="no LFM2 checkpoint")
def test_key_is_stable_and_format_specific():
    from opentpu.isasim import board_config
    cfg = board_config()
    a, parts = refs.key(cfg, "lfm2", "int8", "-", 32)
    b, _ = refs.key(cfg, "lfm2", "int8", "-", 32)
    c, _ = refs.key(cfg, "lfm2", "fp4", "int8", 32)
    assert a == b and a != c
    assert str(ROOT) not in json.dumps(parts)                   # no machine-specific paths
