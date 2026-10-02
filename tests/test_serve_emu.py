"""tools/offload/serve_emu.py and card_fit.py (docs/offload.md 10.13): the host-side emulator
the offload predictions rest on, run on a tiny trace, and the card model's fit."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "offload"))
import card_fit  # noqa: E402


def _emu(tmp, *extra, check=True):
    env = dict(os.environ, PYTHONPATH=str(ROOT), OTPU_RUN_DIR=str(tmp / "run"),
               OTPU_QUIET=str(tmp / "QUIET"))
    out = subprocess.run([sys.executable, str(ROOT / "tools/offload/serve_emu.py"),
                          str(tmp / "tr.json"), str(tmp / "pool.bin"), "--model", "q35",
                          "--pool-n", "4", "--post-c", str(tmp / "C.json"), *extra],
                         env=env, capture_output=True, text=True, check=check, timeout=300)
    return json.loads(out.stdout.strip().splitlines()[-1]) if check else out


def test_serve_emu_replays_a_tiny_trace_and_the_card_model_holds_its_posts(tmp_path):
    """60 requests of 0-2 misses on the 35B's layout (a pool of 4 zero experts): every request
    served, every miss sent with its tag (design A: an answer per request with misses), and
    with --card-w the next post never sooner than a 1-miss request's post + W(1). It does not
    run while the card's lock has a live holder."""
    rng = np.random.default_rng(0)
    m = [int(x) for x in rng.integers(0, 3, 60)]
    json.dump([[2.5e-3 * i, 2.5e-3 * i + 1e-3, "d", i % 40, x] for i, x in enumerate(m)],
              open(tmp_path / "tr.json", "w"))
    with open(tmp_path / "pool.bin", "wb") as f:
        f.truncate(4 * 1671168)
    json.dump([0.0] + [1.5e-3] * 59, open(tmp_path / "C.json", "w"))
    a = _emu(tmp_path)
    w = _emu(tmp_path, "--card-w", "1.5e-3,1:6e-3")
    for r in (a, w):
        assert r["requests"] == 60 and r["misses"] == sum(m) and r["onecall"]
        assert r["calls"]["T"] == sum(m) and r["calls"]["a"] == sum(x > 0 for x in m)
        assert r["calls"]["s"] == 60 and r["crit_s"] > 0
    (tmp_path / "run").mkdir(exist_ok=True)     # a live holder of the card's lock: no run
    (tmp_path / "run" / "xdma0.lock").write_text(f"{os.getpid()}\n")
    busy = _emu(tmp_path, check=False)
    assert busy.returncode and "in use" in busy.stderr and "xdma0.lock" in busy.stderr
    n1 = sum(x == 1 for x in m[:-1])            # (each the next post at its post + 6 ms or
    assert w["wall_s"] >= n1 * 6e-3 + (59 - n1) * 1.5e-3    # later; the others' C 1.5 ms)
    assert w["wall_s"] > a["wall_s"] + n1 * 1e-3


def test_card_fit_finds_the_cards_own_time_and_the_wait_after_the_last_tag(tmp_path):
    """card_fit w on a made-up run: a request of 3 or more misses posts its next F after its
    last tag; one of 1 or 2 misses no sooner than its post + W(m): F and W back within 10 us."""
    F, W = 1.7e-3, {1: 4.0e-3, 2: 4.6e-3}
    rng = np.random.default_rng(1)
    ev, calls, t = [], [], 0.0
    for i in range(400):
        m = int(rng.integers(1, 5))
        seen = t + card_fit.DET
        ce = seen + 0.3e-3 + 0.7e-3 * m * rng.uniform(0.9, 1.1)
        calls.append([seen + 0.3e-3, ce, 1 << 20])
        ev.append([seen, ce + 0.1e-3, "d", i % 40, m])
        t = max(ce + F, t + W.get(m, 0.0))
    json.dump(ev, open(tmp_path / "r.trace.json", "w"))
    json.dump(calls, open(tmp_path / "r.trace.calls.json", "w"))
    r = card_fit._requests(str(tmp_path / "r.trace.json"))
    assert len(r) == 400 and np.isnan(r[-1, 2])
    lines = []
    card_fit.print = lambda *x, **k: lines.append(" ".join(map(str, x)))  # noqa: A001
    try:
        card_fit.main(["w", str(tmp_path)])
    finally:
        del card_fit.print
    v = lines[-1].split()[1].split(",")
    assert abs(float(v[0]) - F) < 10e-6
    got = {int(x): float(y) for x, y in (u.split(":") for u in v[1:])}
    assert set(got) == {1, 2} and all(abs(got[k] - W[k]) < 10e-6 for k in W)
