"""The KV store across sequence slots on a machine with the channel hash (Config.CHASH: x2 here,
S = 4, CardsTransport over IsaCards; the tiny 16-head untied Qwen3.5-MoE at x2's width). K's and
V^T's group chunks hold their lanes by their DRAM address, so one context is other bytes in each
slot; the store holds them in the first slot's lanes (kvio.Layout.relane). A context saved from
any slot, restored into any slot and fed to the end is that slot's own feed from 0, bit for bit:
logits, K and V bytes as stored, DeltaNet states and convolution windows; with a partial last
page, with the destination holding the whole pages already, after a restart. A store written
before the lanes (another model key) is never read."""
import itertools
import os
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from opentpu import quant as Q
from opentpu.llm import batch35 as B35
from opentpu.llm.qwen3 import Engine
from opentpu.serve import api
from opentpu.serve import batching35 as BW35
from opentpu.serve import kvio as KV
from opentpu.serve import scheduler as SC
from opentpu.serve import store as ST

from test_batch35 import _cards, _tiny, _x2
from test_serve_scheduler import _Tok, _prompts

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

CAP = 1024
CUT = 280               # (a context of a whole page of 256 and part of the next)
END = 300
SLOTS = range(4)


@pytest.fixture(scope="module")
def tiny_x2():
    return _tiny(H=512, f=512, kinds=("linear", "attn"))


@pytest.fixture
def x2(tiny_x2, monkeypatch):
    W, spec = tiny_x2
    return W, spec, _x2(monkeypatch)


def _engine(x2, batch: int) -> Engine:
    W, spec, cfg = x2
    return Engine(spec, W, cap=CAP, cfg=cfg, batch=batch, resident=True, layer_major=2,
                  batch_area=True, backend=_cards(cfg))


def _state(eng, slot: int) -> dict:
    """Slot's DeltaNet states and convolution windows by name, each slice's bytes (not through
    kvio: its addresses are what is under test)."""
    import math

    from opentpu.compiler import Affine
    img, S = eng.image, eng.cfg.S
    m, out = img.descriptors(0, slot), {}

    def read(t):
        a, n = Affine.of(t.base).static(), t.esize * math.prod(t.shape)
        return np.concatenate([eng.backend.read(s, a, n) for s in range(S)])
    for li, kind in enumerate(eng.spec.kinds):
        if kind != "linear":
            continue
        dn = m.layer(li).dn
        for q in range(img.nl // 2):
            out[li, q, "window"] = read(dn.window(q))
            out[li, q, "state", 0] = read(dn.state(q, 0))
            out[li, q, "state", 1] = read(dn.state(q, 1))
    return out


def _stored(eng, io, n: int) -> list:
    """A slot's first n positions' K and V bytes as they stand in its DRAM (the backend's: not
    through kvio's lanes), region by region, slice by slice."""
    L = io.layout
    return [eng.backend.read(s, at, b).copy() for i, m in L.split(n) for at, b in L.regions(i, m)
            for s in range(eng.cfg.S)]


def _snap(eng, io, n: int, logits) -> NS:
    return NS(logits=logits, stored=_stored(eng, io, n), state=_state(eng, io.slot))


def _is(a: NS, b: NS, label) -> None:
    assert np.array_equal(a.logits, b.logits), label
    assert len(a.stored) == len(b.stored) and all(map(np.array_equal, a.stored, b.stored)), label
    assert a.state.keys() == b.state.keys() and any(k[2] == "window" for k in a.state), label
    for k in a.state:
        assert a.state[k].any() and np.array_equal(a.state[k], b.state[k]), (label, k)


def _scribble(eng, io, pages, rng) -> None:
    """Junk over a slot's pages and state: what it holds after a restore came from the store."""
    L = io.layout
    for at, b in [r for i, m in pages for r in L.regions(i, m)] + list(L.state_in(io.slot)):
        for s in range(eng.cfg.S):
            eng.backend.write(s, at, rng.integers(1, 255, b, np.uint8))


def test_relane_is_the_channel_hash_between_two_addresses():
    """Layout.relane: a page's K and V^T (grouped) from one sequence's lanes to the first's, as
    quant.hash_rows twice; its scales as they are; each slice's the same way; its own inverse. A
    partial page's K up to whole groups, its V^T tile whole. The first sequence: as it is."""
    wg, D, P, S = 8, 128, 256, 4
    L = KV.Layout.__new__(KV.Layout)
    L.wg, L.D, L.P, L.S, L.pages = wg, D, P, S, 4
    b0, b1 = 0x100000, 0x123400            # (chunks of 1024 bytes: b1's folds are others)
    L.first = [KV.Array(b0, 256, grouped=True), KV.Array(b0 + 0x80000, 8),
               KV.Array(b0 + 0x90000, 256, True, True), KV.Array(b0 + 0xC0000, 4)]
    L.arrays = [KV.Array(a.base + b1 - b0, a.w, a.tile, a.grouped) for a in L.first]
    for i, m in ((1, P), (2, 24), (3, 3)):
        regs = L.regions(i, m)
        n1 = sum(n for _, n in regs)
        d = np.random.default_rng(i).integers(0, 255, S * n1, np.uint8)
        got = L.relane(i, m, d)
        assert not np.array_equal(got, d)
        o = 0
        for s in range(S):
            for a, (at, n) in zip(L.arrays, regs):
                x = d[o:o + n]
                want = x if not a.grouped else Q.hash_rows(Q.hash_rows(x, at, wg, D),
                                                          at - a.base + (a.base - b1 + b0), wg, D)
                assert np.array_equal(got[o:o + n], want), (i, m, s, a)
                o += n
        assert np.array_equal(L.relane(i, m, got), d)
    L.first = None
    assert L.relane(1, P, d) is d


def test_every_slot_restores_every_slots_save(x2):
    """4 x 4 at S = 4. Each slot feeds a prompt from 0, saving at CUT into a store of its own: that
    feed is the slot's reference. Each slot's save is restored into each slot, over junk, and fed
    to the end: the destination's reference, bit for bit. Then with the destination's whole page
    left in place (have): only the tail (the partial page) and the state move."""
    eng = _engine(x2, 4)
    w = BW35.BatchWorker35(eng, _Tok())
    assert eng.cfg.CHASH and [io.slot for io in w.ios] == list(SLOTS)
    a = _prompts(30, [END])[0]
    L, rng = w.ios[0].layout, np.random.default_rng(5)
    assert L.split(CUT) == [(0, 256), (1, CUT - 256)]
    feed = lambda s, toks, logits=False: B35.prefill_slot(eng, toks, s, logits=logits)
    caches, fresh = {}, {}
    for s in SLOTS:
        caches[s] = ST.PrefixCache(ST.BlobStore(1 << 30), L, "m")
        eng.adopt(0, seq=s)
        feed(s, a[:CUT])
        caches[s].save(w.ios[s], a[:CUT], prefix=True)
        fresh[s] = _snap(eng, w.ios[s], END, feed(s, a[CUT:], True))
    # (one context, the same logits, states and pages in the store's lanes; other bytes stored)
    for s in SLOTS:
        assert np.array_equal(fresh[s].logits, fresh[0].logits), s
        assert all(np.array_equal(fresh[s].state[k], fresh[0].state[k]) for k in fresh[0].state), s
        for x, y in zip(w.ios[s].read_pages(L.split(END)), w.ios[0].read_pages(L.split(END))):
            assert x.any() and np.array_equal(x, y), s
    assert not np.array_equal(fresh[0].stored[0], fresh[1].stored[0])
    for s, d in itertools.product(SLOTS, SLOTS):
        io, c = w.ios[d], caches[s]
        m = c.match(a, limit=CUT)
        assert m.k == CUT and m.entry.m == CUT - 256
        _scribble(eng, io, L.split(CUT), rng)
        assert c.restore(io, m) == CUT
        eng.adopt(CUT, seq=d)
        _is(_snap(eng, io, END, feed(d, a[CUT:], True)), fresh[d], (s, d, "empty"))
        _scribble(eng, io, [(1, CUT - 256)], rng)
        n0 = io.stats["written"]
        assert c.restore(io, m, have=CUT) == CUT
        assert io.stats["written"] - n0 == L.page_bytes(CUT - 256) + L.state_bytes
        eng.adopt(CUT, seq=d)
        _is(_snap(eng, io, END, feed(d, a[CUT:], True)), fresh[d], (s, d, "have"))


def test_a_copy_between_slots_is_the_slots_own_bytes(x2):
    """ContextIO.copy_from (a dense model's slots: batching.BatchWorker) under the channel hash:
    a page copied from another slot stands in the destination's lanes (no copy on the device
    could put it there)."""
    eng = _engine(x2, 4)
    w = BW35.BatchWorker35(eng, _Tok())
    a, pages = _prompts(30, [END])[0], [(0, 256)]
    feed = lambda s: B35.prefill_slot(eng, a, s, logits=False)
    feed(0), feed(2)
    want = _stored(eng, w.ios[2], 256)
    _scribble(eng, w.ios[2], pages, np.random.default_rng(6))
    w.ios[2].copy_from(w.ios[0], pages)
    assert all(map(np.array_equal, _stored(eng, w.ios[2], 256), want))


def test_a_context_restored_after_a_restart_into_another_slot(x2, tmp_path):
    """A prefix and a whole context, saved from slot 1 onto the disk; a new server (its store on
    the directory, a device holding nothing): the prefix restored into slot 2 and the context into
    slot 3 and fed on are those slots' own feeds from 0."""
    a = _prompts(30, [END])[0]
    more = _prompts(31, [8])[0]
    eng = _engine(x2, 4)
    w = BW35.BatchWorker35(eng, _Tok())
    store = ST.BlobStore(1 << 30, 1 << 30, tmp_path)
    c = ST.PrefixCache(store, w.ios[0].layout, "m")
    eng.adopt(0, seq=1)
    B35.prefill_slot(eng, a[:CUT], 1, logits=False)
    c.save(w.ios[1], a[:CUT], prefix=True)
    B35.prefill_slot(eng, a[CUT:], 1, logits=False)
    c.save(w.ios[1], a)
    store.persist()

    eng = _engine(x2, 4)
    w = BW35.BatchWorker35(eng, _Tok())
    feed = lambda s, toks: B35.prefill_slot(eng, toks, s, logits=True)
    ref = {2: _snap(eng, w.ios[2], END, feed(2, a)), 3: _snap(eng, w.ios[3], END + 8, feed(3, a + more))}
    store = ST.BlobStore(1 << 30, 1 << 30, tmp_path)
    c = ST.PrefixCache(store, w.ios[0].layout, "m")
    assert {e.n: e.prefix for e in c.entries.values()} == {CUT: True, END: False}
    assert all("-e-" in k for k in store.ram)      # (the pages come from the disk)
    rng = np.random.default_rng(7)
    L = w.ios[0].layout
    _scribble(eng, w.ios[2], L.split(CUT), rng)
    assert c.restore(w.ios[2], c.match(a, limit=CUT)) == CUT
    eng.adopt(CUT, seq=2)
    _is(_snap(eng, w.ios[2], END, feed(2, a[CUT:])), ref[2], "the prefix")
    _scribble(eng, w.ios[3], L.split(END), rng)
    m = c.match(a + more, limit=END + 7)
    assert m.k == END and c.restore(w.ios[3], m) == END
    eng.adopt(END, seq=3)
    _is(_snap(eng, w.ios[3], END + 8, feed(3, more)), ref[3], "the context")
    assert store.stats["disk_hits"] > 0


def _old_key(eng, path) -> str:
    """The model key before the lanes (the store's name for a server of then)."""
    return ST.model_key(str(Path(path).resolve()), type(eng.image).__module__, eng.image.wformat,
                        eng.image.head_format, getattr(eng.image, "formats", None), eng.cap,
                        eng.block, eng.cfg.S, eng.cfg.D, eng.cfg.MCOLS, eng.cfg.PAIR)


def test_a_store_under_the_old_key_never_matches(x2, tmp_path):
    """A store written before the lanes held slots' pages in their own lanes. The key of a
    server with slots under the channel hash is another: its directory (--kv-dir/<key>) is empty,
    a request sharing the old store's prompt is fed from 0, and the old files stay. Even on the
    old directory its entries are not found. One sequence's key stays: its pages were right."""
    a, path = NS(mtp=False), tmp_path / "model"
    eng = _engine(x2, 2)
    new, old = api.store_key(a, eng, path), _old_key(eng, path)
    assert new != old
    one = NS(image=eng.image, cap=eng.cap, block=eng.block, cfg=eng.cfg, batch=1)
    assert api.store_key(a, one, path) == _old_key(one, path) == old
    w = BW35.BatchWorker35(eng, _Tok())
    L = w.ios[0].layout
    p = _prompts(30, [40])[0]
    kv = tmp_path / "kv"
    store = ST.BlobStore(1 << 30, 1 << 30, kv / old)
    B35.prefill_slot(eng, p[:32], 1, logits=False)
    ST.PrefixCache(store, L, old).save(w.ios[1], p[:32], prefix=True)
    store.persist()
    files = sorted(os.listdir(kv / old))
    assert files
    # (the old key's own server finds it: what the new key must not)
    assert ST.PrefixCache(ST.BlobStore(1 << 30, 1 << 30, kv / old), L, old).match(p).k == 32
    same = ST.BlobStore(1 << 30, 1 << 30, kv / old)
    c = ST.PrefixCache(same, L, new)
    assert not c.entries and c.match(p).k == 0
    store = ST.BlobStore(1 << 30, 1 << 30, kv / new)
    w = BW35.BatchWorker35(_engine(x2, 2), _Tok(), store=store, model_key=new).start()
    try:
        r = w.submit(SC.Request(p, max_new=2, stop_ids=frozenset())).wait(1800)
        assert r.error is None and r.cached == 0 and w.stats["restores"] == 0
    finally:
        w.stop(persist=False)
    assert sorted(os.listdir(kv / old)) == files
