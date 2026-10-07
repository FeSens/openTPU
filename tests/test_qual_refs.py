"""tools/qual/refs.py: the reference cache key, the references (tokens past EOS, every logits
vector's digest), the card check's comparison and its fail-fast wait (no card: a fake engine)."""
import importlib.util
import json
import os
import pickle
import re
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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
    (pkg / "host" / "offload.py").write_text("LINE = 64\n")
    h0 = refs.source_hash(pkg)
    (pkg / "host" / "board.py").write_text("POLL = 2\n")       # a poll fix: same references
    assert refs.source_hash(pkg) == h0
    (pkg / "compiler.py").write_text("a = 2\n")                 # a compiler change: new ones
    h1 = refs.source_hash(pkg)
    assert h1 != h0
    (pkg / "host" / "offload.py").write_text("LINE = 128\n")    # host code the models import
    assert refs.source_hash(pkg) != h1                          # (before: the same)


def test_the_host_code_the_models_import_is_hashed():
    """Every opentpu.host module that the hashed sources import (the reference's path) is in
    HOST_IN_REF: an edit to it computes new references."""
    pat = re.compile(r"from (?:\.\.|opentpu\.)host\.(\w+) import|import opentpu\.host\.(\w+)")
    used = set()
    for rel, p in refs.sources():
        if not rel.startswith("host/"):
            used |= {f"host/{a or b}.py" for a, b in pat.findall(p.read_text())}
    assert used and used <= set(refs.HOST_IN_REF), used


def test_env_key_has_the_kernels_variables_not_the_runtime_ones(tmp_path, monkeypatch):
    pkg = tmp_path / "opentpu"
    (pkg / "llm").mkdir(parents=True)
    (pkg / "host").mkdir()
    (pkg / "llm" / "gemma4.py").write_text('os.environ.get("OTPU_PLE_FORMAT")\n'
                                           'os.environ.get("OTPU_FORMATS")\n')
    (pkg / "progcache.py").write_text('os.environ.get("OTPU_PROG_CACHE")\n')
    (pkg / "simtmp.py").write_text('os.environ.get("OTPU_SIM_TMP")\n')
    (pkg / "host" / "runstate.py").write_text('os.environ.get("OTPU_LOCK_HELD")\n')
    for k in list(os.environ):
        if k.startswith("OTPU_"):
            monkeypatch.delenv(k)
    assert refs.env_key(pkg) == {}
    for k in ("OTPU_PROG_CACHE", "OTPU_SIM_TMP", "OTPU_LOCK_HELD"):     # runtime, host-only
        monkeypatch.setenv(k, "1")
    assert refs.env_key(pkg) == {}
    monkeypatch.setenv("OTPU_PLE_FORMAT", "fp4")
    monkeypatch.setenv("OTPU_FORMATS", "")                              # set, empty
    assert refs.env_key(pkg) == {"OTPU_FORMATS": "", "OTPU_PLE_FORMAT": "fp4"}


def test_parse_run():
    assert refs.parse_run("qwen3:int8:-") == ("qwen3", "int8", "-", False, False)
    assert refs.parse_run("lfm2:int8") == ("lfm2", "int8", "-", False, False)
    assert refs.parse_run("lfm2:int8:-:long,pr") == ("lfm2", "int8", "-", True, True)
    with pytest.raises(ValueError):
        refs.parse_run("lfm2:int8:-:loong")


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
def test_key_is_stable_and_format_specific(monkeypatch):
    from opentpu.isasim import board_config
    cfg = board_config()
    a, parts = refs.key(cfg, "lfm2", "int8", "-", 32)
    b, _ = refs.key(cfg, "lfm2", "int8", "-", 32)
    c, _ = refs.key(cfg, "lfm2", "fp4", "int8", 32)
    assert a == b and a != c and parts["v"] == refs.VERSION
    assert str(ROOT) not in json.dumps(parts)                   # no machine-specific paths
    long_, lp = refs.key(cfg, "lfm2", "int8", "-", 32, long=True)
    pr, _ = refs.key(cfg, "lfm2", "int8", "-", 32, long=True, pr=True)
    assert len({a, long_, pr}) == 3 and "_long_pr_" in pr.name and lp["ids"] != parts["ids"]
    monkeypatch.setenv("OTPU_LOCK_HELD", "xdma0")              # under otpu-lock: the same one
    assert refs.key(cfg, "lfm2", "int8", "-", 32)[0] == a
    monkeypatch.setenv("OTPU_MLP_UNROLL_BODIES", "4")           # lfm2's programs: another one
    assert refs.key(cfg, "lfm2", "int8", "-", 32)[0] != a       # (before: the same)


@pytest.mark.skipif(not (ROOT / "models/LFM2.5-230M").exists(), reason="no LFM2 checkpoint")
def test_the_long_prompt_crosses_the_bucket():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ROOT / "models/LFM2.5-230M")
    n = len(refs.prompt_ids(tok, long=True))
    assert refs.LONG_TOKENS - 8 <= n <= refs.LONG_TOKENS < refs.ATTN_BLOCK
    assert n + refs.NTOK - 2 >= refs.ATTN_BLOCK + 8             # decoded positions past 256
    assert len(refs.prompt_ids(tok)) < 32


# ---- the reference and the card check, with a fake engine
V = 64


def _logits(pos, tok):
    return np.random.default_rng(1000 * pos + tok).standard_normal(V).astype(np.float32)


class FakeEngine:
    """prefill / step / generate_card over seeded logits; `off` = {position: ulps} changes the
    logits there (a card that is not bit-exact), `engaged` whether resident decode engages."""
    off, engaged = {}, True

    def __init__(self, spec, W, cap, cfg, wformat, head_format, resident=False, backend=None,
                 prompt_runs=False):
        self.resident = resident and self.engaged
        self.can_generate = self.resident
        self.stats, self.pos = [], 0
        self.backend = SimpleNamespace(close=lambda: None)

    def _at(self, tok):
        lg = _logits(self.pos, tok)
        if self.pos in self.off:
            lg.view(np.uint32)[0] += self.off[self.pos]
        return lg

    def prefill(self, ids):
        self.pos = len(ids)
        self.stats.append({"rows": len(ids), "cycles": 1e6})
        return self._at(ids[-1])

    def step(self, tok):
        self.pos += 1
        self.stats.append({"cycles": 1e6})
        return self._at(tok)

    def generate_card(self, tok, n, stop_ids=None):
        assert stop_ids == []                   # past EOS
        out = []
        for _ in range(n):
            tok = int(np.argmax(self.step(tok)))
            out.append(tok)
        return out

    def _drain(self):
        pass


@pytest.fixture
def fake(tmp_path, monkeypatch):
    from opentpu.host import board
    from opentpu.llm import qwen3
    kp = tmp_path / "ref.pkl"
    tok = SimpleNamespace(decode=lambda ids, **kw: "".join(f"<{i}>" for i in ids))
    monkeypatch.setattr(refs, "key", lambda *a, **kw: (kp, {"model": "fake"}))
    monkeypatch.setattr(refs, "load", lambda m: (tmp_path, SimpleNamespace(eos=(0,)), {}, tok))
    monkeypatch.setattr(refs, "prompt_ids",
                        lambda t, long=False: list(range(1, 240 if long else 22)))
    monkeypatch.setattr(refs, "prompt_runs_taken", lambda eng, ids: True)
    monkeypatch.setattr(qwen3, "Engine", FakeEngine)
    monkeypatch.setattr(board, "sim_config", lambda *a, **kw: None)
    monkeypatch.setattr(board, "XdmaTransport", lambda *a, **kw: None)
    monkeypatch.setattr(FakeEngine, "off", {})
    monkeypatch.setattr(FakeEngine, "engaged", True)
    cfg = tmp_path / "cfg.pkl"
    cfg.write_bytes(pickle.dumps("cfg"))
    return kp, cfg


def test_a_reference_is_ntok_tokens_past_eos_and_every_logits_digest(fake, tmp_path):
    kp, cfg = fake
    refs.side(kp, ".pending").write_text("{}")                  # compute queued it
    assert refs.one("cfg", "m", "int8", "-", 32) == 0
    want = pickle.loads(kp.read_bytes())
    assert want["v"] == refs.VERSION and len(want["tokens"]) == len(want["logits"]) == 32
    toks, digs = refs.greedy(FakeEngine(None, None, 0, None, "int8", None), list(range(1, 22)),
                             32)
    assert want["tokens"] == toks and want["logits"] == [d for d, _ in digs]
    assert want["top"][0][0] == (toks[0], float(_logits(21, 21)[toks[0]]))
    assert not refs.side(kp, ".pending").exists() and not list(tmp_path.glob("*.tmp"))
    refs.side(kp, ".pending").write_text("{}")                  # cached: no .pending left
    assert refs.one("cfg", "m", "int8", "-", 32) == 0 and not refs.side(kp, ".pending").exists()


def test_the_card_check_compares_every_logits_vector(fake, capsys):
    kp, cfg = fake
    refs.one("cfg", "m", "int8", "-", 32)
    capsys.readouterr()
    assert refs.card(cfg, "m", "int8", "-", 32, resident=True) == 0
    assert "[PASS] model m int8/- resident: 32 tokens and the logits of all 32 bit-exact" \
        in capsys.readouterr().out
    FakeEngine.off = {30: 1}                    # one ulp of one logit: the tokens stay
    assert refs.card(cfg, "m", "int8", "-", 32, resident=False) == 1
    out = capsys.readouterr().out
    assert "[FAIL] model m int8/-: token 9's logits differ" in out, out
    FakeEngine.off = {21: 1}                    # the prefill's: the card loop sees it too
    assert refs.card(cfg, "m", "int8", "-", 32, resident=True, loop=True) == 1
    assert "[FAIL] model m int8/- resident card loop: token 0's logits differ" in \
        capsys.readouterr().out


def test_a_check_that_does_not_test_what_it_says_fails(fake, capsys, monkeypatch):
    kp, cfg = fake
    refs.one("cfg", "m", "int8", "-", 32)
    FakeEngine.engaged = False          # before: "[PASS] ... (resident decode not engaged)"
    assert refs.card(cfg, "m", "int8", "-", 32, resident=True) == 1
    assert "resident decode not engaged" in capsys.readouterr().out
    FakeEngine.engaged = True
    monkeypatch.setattr(refs, "prompt_runs_taken", lambda eng, ids: False)
    assert refs.card(cfg, "m", "int8", "-", 32, resident=True, prompt_runs=True) == 1
    assert "prompt runs not taken" in capsys.readouterr().out
    monkeypatch.setattr(refs, "prompt_ids", lambda t, long=False: list(range(1, 200)))
    assert refs.card(cfg, "m", "int8", "-", 32, resident=True, long=True) == 1
    assert "does not cross position 256" in capsys.readouterr().out


def test_an_old_reference_is_no_pass(fake, capsys):
    kp, cfg = fake
    kp.write_bytes(pickle.dumps([785, 6722, 315]))              # refs.py v1: a token list
    assert refs.card(cfg, "m", "int8", "-", 32, resident=False) == 1
    assert "not of format 2" in capsys.readouterr().out
