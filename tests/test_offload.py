"""tools/offload/cachesim.py: the expert-cache policies and path (a)'s timing on small
hand-made traces."""
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


def test_lfu_layer_is_the_servers_lfu():
    """lfu_layer replays ExpertServer(policy="lfu"): the same misses token by token on a skewed
    random trace of several layers from the same profile's warm start, and linksim's lfu streams
    as many experts (its demand transfers, no prefetch)."""
    from opentpu.host.offload import ExpertServer, Layout, SimDram
    rng = np.random.default_rng(5)
    T, L, E, k, cap, half = 300, 3, 16, 2, 5, 4.0
    p = 1.0 / np.arange(1, E + 1) ** 1.2
    perm = [rng.permutation(E) for _ in range(L)]
    req = np.array([[perm[j][rng.choice(E, k, replace=False, p=p / p.sum())] + j * E
                     for j in range(L)] for _ in range(T)])
    prof = cs.freq(req[:60], E * L)
    warm = cs.top_set(prof, E * L)
    lfu, _ = cs.simulate(req, cap * L, "lfu_layer", warm, prof, half)
    lru, _ = cs.simulate(req, cap * L, "lru_layer", warm, prof, half)
    lay = Layout.build(4096, E, k, (cap,) * L, 128)
    srv = ExpertServer(SimDram(np.zeros(lay.end + 4096, np.uint8)), lay,
                       lambda g: bytes(128), policy="lfu", half=half)
    srv.load(warm)
    got = []
    for t in range(T):
        m0 = srv.misses
        for j in range(L):
            srv.serve([int(e) for e in req[t, j]])
        got.append(srv.misses - m0)
    assert lfu.tolist() == got
    assert lfu.tolist() != lru.tolist()
    by = dict(x=1e6, head=0.0, d_pre=1e6, d_post=0.0)
    r = cs.linksim(req, {}, [cap] * L, [[e for e in warm if e // E == j] for j in range(L)],
                   _hw(), by, policy="lfu", half=half)
    assert abs(r["demand"] * T - sum(got)) < 1e-9


def test_prefetch_is_the_servers_capped_hints():
    """prefetch (docs/offload.md 12.7) replays ExpertServer's capped hints (hint_n / hint_top),
    each landed before its layer's request: the same misses and experts sent on a noisy
    prediction. A perfect prediction's experts are all used; a wrong one's cost misses."""
    from opentpu.host.offload import ExpertServer, Layout, SimDram
    rng = np.random.default_rng(7)
    T, L, E, k, cap, half = 300, 3, 16, 2, 5, 4.0
    p = 1.0 / np.arange(1, E + 1) ** 1.2
    perm = [rng.permutation(E) for _ in range(L)]
    req = np.array([[perm[j][rng.choice(E, k, replace=False, p=p / p.sum())] + j * E
                     for j in range(L)] for _ in range(T)])
    noisy = req.copy()                  # the router on the layer's input: each id right with
    for t in range(T):                  # p 0.6, else another of the layer's
        for j in range(L):
            for i in range(k):
                if rng.random() > 0.6:
                    noisy[t, j, i] = rng.choice([g for g in range(j * E, (j + 1) * E)
                                                 if g not in noisy[t, j]])
    pred = [noisy[:, j] for j in range(L)]
    warm = cs.top_set(cs.freq(req[:60], E * L), E * L)
    lay = Layout.build(4096, E, k, (cap,) * L, 128)
    for n, top in ((1, 2), (1, 1), (2, 2)):
        srv = ExpertServer(SimDram(np.zeros(lay.end + 4096, np.uint8)), lay,
                           lambda g: bytes(128), policy="lfu", half=half)
        srv.load(warm)
        srv.hint_n, srv.hint_top = n, top
        for t in range(T):
            for j in range(L):
                srv.hint([int(e) for e in noisy[t, j]])
                while srv.pending:
                    srv.step()
                srv.serve([int(e) for e in req[t, j]])
        m, s, _ = cs.prefetch(req, pred, E, cap * L, warm, n, top, half)
        assert (round(m * T), round(s * T)) == (srv.misses, srv.prefetched)
    none = cs.prefetch(req, None, E, cap * L, warm, 0, 0, half)[0]
    good = cs.prefetch(req, [req[:, j] for j in range(L)], E, cap * L, warm, 1, k, half)
    bad = cs.prefetch(req, [(req[:, j] + 7) % E + j * E for j in range(L)], E, cap * L, warm, 1,
                      k, half)
    assert good[0] < none < bad[0] and good[1] == good[2] > 0 and bad[2] < bad[1]


def _hw(pcie=1.3e9):
    return dict(dram=14e9, call=30e-6, req=30e-6, done=15e-6, ssd=0.5e9, pcie_one=pcie)


def _warm(req, L):
    """Each layer's experts in the order they first appear (a warm cache holds them all)."""
    return [list(dict.fromkeys(int(e) for e in req[:, j].ravel())) for j in range(L)]


def test_linksim_all_hits_is_the_resident_bound():
    rng = np.random.default_rng(1)
    T, L, E, k = 40, 3, 8, 2
    req = np.stack([np.stack([rng.choice(E, k, replace=False) + j * E for j in range(L)])
                    for _ in range(T)])
    by = dict(x=3e6, head=50e6, d_pre=20e6, d_post=5e6)
    for ov in ("each", "all"):
        r = cs.linksim(req, {}, [E] * L, _warm(req, L), _hw(), by, overlap=ov)
        assert r["demand"] == 0 and r["link"] == 0
        assert abs(1 / r["tok_s"] - cs.resident_time(L, k, by, _hw())) < 1e-12


def test_linksim_one_miss_costs_the_round_trip():
    """Token 1 misses (cache of 1 holding expert 0): the router, the request, the DMA call and
    its bytes, the flag, then the expert."""
    req = _req([0, 1], k=1)
    by = dict(x=3e6, head=0.0, d_pre=20e6, d_post=0.0)
    hw = _hw()
    for ov in ("each", "all"):
        r = cs.linksim(req, {}, [1], [[0]], hw, by, overlap=ov)
        t0 = (by["d_pre"] + by["x"]) / hw["dram"]
        t1 = (by["d_pre"] / hw["dram"] + hw["req"] + by["x"] / hw["pcie_one"] + hw["call"]
              + hw["done"] + by["x"] / hw["dram"])
        assert r["demand"] == 0.5
        assert abs(2 / r["tok_s"] - (t0 + t1)) < 1e-12


def test_linksim_hits_first_hides_the_resident_experts():
    """k = 4 with one miss per layer on a slow link: computing the three it has while the
    fourth streams saves up to their time; waiting for the last one does not."""
    T, L, E, k = 20, 2, 16, 4
    req = np.stack([np.stack([np.r_[0, 1, 2, 3 + t % 13] + j * E for j in range(L)])
                    for t in range(T)])
    by = dict(x=3e6, head=0.0, d_pre=20e6, d_post=0.0)
    warm = [[j * E + e for e in range(4)] for j in range(L)]
    each = cs.linksim(req, {}, [4] * L, warm, _hw(), by, overlap="each")
    alls = cs.linksim(req, {}, [4] * L, warm, _hw(), by, overlap="all")
    assert each["demand"] == alls["demand"] == L * (T - 1) / T
    assert each["tok_s"] > alls["tok_s"]
    # each missing layer saves k - 1 expert times (the transfer's DRAM writes hide in the wait)
    saved = L * (T - 1) * (k - 1) * by["x"] / _hw()["dram"]
    assert abs(T / alls["tok_s"] - T / each["tok_s"] - saved) < 1e-9


def test_linksim_perfect_prediction_prefetches_without_waste():
    """The next layer's experts named exactly, on a link fast enough to land them during the
    layer: no demand transfers after the first layer, nothing wasted."""
    T, L, E, k = 30, 3, 8, 1
    rng = np.random.default_rng(2)
    req = np.stack([np.stack([rng.choice(E, k) + j * E for j in range(L)]) for _ in range(T)])
    preds = {"prev_r": [None] + [req[:, j] for j in range(1, L)]}
    by = dict(x=1e6, head=0.0, d_pre=200e6, d_post=0.0)
    r = cs.linksim(req, preds, [1] * L, [[j * E] for j in range(L)], _hw(), by,
                   pred="prev_r", width=k)
    s = cs.linksim(req, preds, [1] * L, [[j * E] for j in range(L)], _hw(), by)
    assert r["wasted"] == 0
    assert r["demand"] < s["demand"]
    assert r["tok_s"] > s["tok_s"]
