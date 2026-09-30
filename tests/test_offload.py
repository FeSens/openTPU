"""tools/offload/cachesim.py: the expert-cache policies on small hand-made traces."""
import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "cachesim", Path(__file__).resolve().parent.parent / "tools/offload/cachesim.py")
cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cs)


def _req(seq, k=1):
    """One MoE layer, one expert per token: requests [T, 1, k]."""
    return np.array(seq, np.int64).reshape(-1, 1, k)


def test_lru_and_opt_on_a_known_sequence():
    # capacity 2, empty start: LRU misses a, b, c, a, b (every one: the loop is longer than the
    # cache), OPT keeps a and evicts b at c: misses a, b, c, b
    req = _req([0, 1, 2, 0, 1])
    lru, _ = cs.simulate(req, 2, "lru", [])
    opt, _ = cs.simulate(req, 2, "opt", [])
    assert lru.sum() == 5
    assert opt.sum() == 4


def test_static_never_changes_and_warm_start_counts():
    req = _req([0, 1, 0, 2, 0])
    st, at = cs.simulate(req, 1, "static", [0])
    assert st.tolist() == [0, 1, 0, 1, 0]
    assert at == {(1, 0): [1], (3, 0): [2]}


def test_policies_order_on_random_traces():
    rng = np.random.default_rng(0)
    T, L, E, k = 300, 6, 16, 2
    p = rng.dirichlet(np.ones(E) * 0.3, size=L)
    req = np.stack([np.stack([rng.choice(E, k, replace=False, p=p[j]) + j * E
                              for j in range(L)]) for _ in range(T)])
    prof = cs.freq(req, E * L)
    for C in (L * k, 30, 60):
        warm = cs.top_set(prof, C)
        miss = {pol: cs.simulate(req, C, pol, cs.top_set(prof, E * L) if pol == "lru_layer"
                                 else warm, prof)[0].sum()
                for pol in ("static", "lru", "lru_layer", "lfu", "opt")}
        assert miss["opt"] == min(miss.values())
        assert all(m <= T * L * k for m in miss.values())


def test_tok_times_bounds():
    """No misses: streaming costs only the halts; the hybrid cannot beat the card alone plus
    the host alone (the two memory systems in parallel)."""
    req = _req([0, 1, 0, 1], k=1)
    hw = dict(dram=14e9, host=12e9, sync=60e-6, call=30e-6, pcie=[1.3e9])
    by = dict(x=3e6, d=30e6, head=0.0)
    r = cs.tok_times(req, {}, {}, hw, by)
    t_res, t_str = 1 / r["resident"], 1 / r["stream@1.3"]
    assert abs(t_str - t_res - 60e-6) < 1e-9
    _, at = cs.simulate(req, 1, "static", [2])            # every request misses
    r = cs.tok_times(req, at, {}, hw, by)
    assert r["hybrid"] <= 1 / ((by["d"] + by["x"]) / (hw["dram"] + hw["host"]))
