"""The server's KV store (opentpu/serve/store.py): BlobStore's tiers (RAM, spilled to files by
a writer thread, read back; pins; files found by a later store) and PrefixCache's entries on a
fake device whose page bytes are a function of the tokens before (as a KV cache's): the longest
saved prefix, whole pages shared by contexts with a common prefix, partial restores where the
caches hold the whole context and only saved ends where there is state, entries dropped with
their blobs."""
import hashlib
import threading

import numpy as np
import pytest

from opentpu.serve import store as ST


def _arr(n, v):
    return np.full(n, v, np.uint8)


def test_blobstore_spills_reads_back_and_drops(tmp_path):
    dropped = []
    s = ST.BlobStore(ram_bytes=300, disk_bytes=500, dir=tmp_path, on_drop=dropped.append)
    for i in range(4):
        s.put(f"b{i}", _arr(100, i))
    s.flush()
    u = s.usage()
    assert u["ram_bytes"] <= 300 and u["disk_blobs"] == 1 and (tmp_path / "b0.bin").exists()
    assert np.array_equal(s.get("b0"), _arr(100, 0))        # read back into RAM
    assert s.usage()["disk_hits"] == 1
    for i in range(4, 12):
        s.put(f"b{i}", _arr(100, i))
    s.flush()
    u = s.usage()
    assert u["ram_bytes"] <= 300 and u["disk_bytes"] <= 500 and dropped
    assert all(k not in s for k in dropped)
    assert s.get(dropped[0]) is None
    # a later store on the same directory finds the files
    s2 = ST.BlobStore(ram_bytes=300, disk_bytes=500, dir=tmp_path)
    on_disk = [k for k in s.keys() if k in s.disk]
    assert on_disk and all(k in s2 for k in on_disk)
    assert np.array_equal(s2.get(on_disk[-1]), s.get(on_disk[-1]))


def test_blobstore_pins_and_no_disk():
    dropped = []
    s = ST.BlobStore(ram_bytes=250, on_drop=dropped.append)
    s.put("a", _arr(100, 1))
    s.pin(["a"])
    for i in range(5):
        s.put(f"x{i}", _arr(100, 2))
    assert "a" in s and "a" not in dropped and s.usage()["disk_blobs"] == 0
    s.unpin(["a"])
    s.put("y", _arr(100, 3))
    assert "a" not in s and "a" in dropped


def _flushes(s, timeout=10):
    """flush returns (a dead writer leaves the queue's join waiting for ever)."""
    t = threading.Thread(target=s.flush, daemon=True)
    t.start()
    t.join(timeout)
    return not t.is_alive()


def test_disk_eviction_passes_blobs_a_drop_took(tmp_path):
    """A drop takes the entry's other blobs (PrefixCache._remove), so the list of keys _evict_disk
    walks goes stale: on the card 11-06 KeyError, the writer dead, the disk over its limit."""
    a = [f"a{i}" for i in range(6)]
    s = ST.BlobStore(ram_bytes=1000, disk_bytes=2500, dir=tmp_path,
                     on_drop=lambda k: [s.discard(o) for o in a])
    for k in a:
        s.put(k, _arr(100, 1))
    for k in "bcde":
        s.put(k, _arr(900, 2))              # a's pages to disk, then the disk over its limit
    assert _flushes(s)
    u = s.usage()
    assert u["disk_bytes"] <= 2500 and u["spilling_bytes"] == 0 and u["dropped"] >= 1
    assert not any(k in s for k in a)
    n = u["spilled"]
    s.put("f", _arr(900, 3))                # the writer still works
    assert _flushes(s) and s.usage()["spilled"] > n and s.usage()["spilling_bytes"] == 0


def test_ram_eviction_passes_blobs_a_drop_took():
    a = [f"a{i}" for i in range(6)]
    s = ST.BlobStore(ram_bytes=1000, on_drop=lambda k: [s.discard(o) for o in a])
    for k in a:
        s.put(k, _arr(100, 1))
    s.put("b", _arr(300, 2))
    s.put("c", _arr(900, 3))                # over by more than all of a
    assert "c" in s and "b" not in s and s.usage()["ram_bytes"] == 900


@pytest.mark.parametrize("exc", [RuntimeError, SystemExit])
def test_writer_survives_a_failing_drop(tmp_path, exc):
    def drop(k):
        raise exc("a drop that fails")
    s = ST.BlobStore(ram_bytes=100, disk_bytes=250, dir=tmp_path, on_drop=drop)
    for i in range(6):
        s.put(f"b{i}", _arr(100, i))
    assert _flushes(s)
    u = s.usage()
    assert u["writer_errors"] >= 1 and u["spilling_bytes"] == 0 and u["disk_bytes"] <= 250


class _Layout:
    def __init__(self, P=4, stateful=False, sf=""):
        self.P, self.stateful, self.sf = P, stateful, sf
        self.state_bytes = 8 if stateful else 0

    def page_bytes(self, m=None):
        return 2 * (self.P if m is None else m)


class _Dev:
    """A device whose position p holds 2 bytes of the hash of tokens [0, p] (a KV cache's
    dependence); the state 8 bytes of the hash of the whole context."""

    def __init__(self, L):
        self.layout, self.toks, self.cells = L, [], {}
        self.reads = 0

    @staticmethod
    def cell(prefix):
        return np.frombuffer(hashlib.sha256(np.asarray(prefix, np.int64).tobytes()).digest()[:2],
                             np.uint8)

    def feed(self, toks):
        for t in toks:
            self.toks.append(t)
            self.cells[len(self.toks) - 1] = self.cell(self.toks)

    def read_pages(self, pages):
        P, out = self.layout.P, []
        for i, m in pages:
            self.reads += 1
            out.append(np.concatenate([self.cells.get(i * P + j, np.zeros(2, np.uint8))
                                       for j in range(m)]))
        return out

    def write_pages(self, pages):
        P = self.layout.P
        for i, m, d in pages:
            for j in range(m):
                self.cells[i * P + j] = d[2 * j:2 * j + 2]

    def read_state(self):
        return np.frombuffer(hashlib.sha256(np.asarray(self.toks, np.int64).tobytes()).digest()
                             [:8], np.uint8).copy()

    def write_state(self, d):
        self.state = d.copy()

    def holds(self, toks):
        return all(np.array_equal(self.cells.get(p), self.cell(toks[:p + 1]))
                   for p in range(len(toks)))


def _cache(stateful=False, ram=1 << 20):
    L = _Layout(stateful=stateful)
    return ST.PrefixCache(ST.BlobStore(ram), L, "m0"), L


def test_prefix_cache_longest_prefix_and_shared_pages():
    pc, L = _cache()
    a = list(range(1, 11))                  # 10 tokens: pages 0, 1 whole, 2 positions in page 2
    dev = _Dev(L)
    dev.feed(a)
    e = pc.save(dev, a)
    assert e.n == 10 and len(e.pages) == 2 and e.m == 2 and e.tail
    b = a[:6] + [99, 98, 97, 96, 95]        # shares page 0 and two positions of page 1
    dev2 = _Dev(L)
    dev2.feed(b)
    reads = dev2.reads
    e2 = pc.save(dev2, b)
    assert dev2.reads - reads == 2          # page 1 and the tail: page 0 was stored
    assert pc.stats["shared_pages"] == 1
    # the longest prefix: a's first 9 tokens of a query continuing a otherwise
    q = a[:9] + [7, 7]
    m = pc.match(q)
    assert m.k == 9 and m.entry is e
    fresh = _Dev(L)
    assert pc.restore(fresh, m) == 9 and fresh.holds(q[:9])
    m = pc.match(b[:6] + [99, 1])
    assert m.k == 7 and m.entry is e2
    assert pc.match([5, 5, 5]).k == 0
    assert pc.match(a, limit=len(a) - 1).k == 9     # a prompt keeps a token to feed


def test_prefix_cache_with_state_restores_saved_ends_only():
    pc, L = _cache(stateful=True)
    a = list(range(1, 11))
    dev = _Dev(L)
    dev.feed(a[:6])
    e6 = pc.save(dev, a[:6])                # a reply's start, say
    dev.feed(a[6:])
    e10 = pc.save(dev, a)
    assert e6.key in pc.entries and e10.key in pc.entries      # both kept (state)
    assert pc.match(a[:9] + [5]).k == 6     # not 9: no state there
    m = pc.match(a + [5])
    assert m.k == 10 and m.entry is e10
    fresh = _Dev(L)
    pc.restore(fresh, m)
    assert fresh.holds(a) and np.array_equal(fresh.state, dev.read_state())


def test_stateless_prefixes_are_covered_and_dropped_blobs_drop_entries():
    pc, L = _cache(ram=40)                  # small: blobs dropped (no disk)
    a = list(range(1, 11))
    dev = _Dev(L)
    dev.feed(a[:5])
    e5 = pc.save(dev, a[:5])
    dev.feed(a[5:])
    pc.save(dev, a)
    assert e5.key not in pc.entries         # covered by the longer context
    for s in range(5):                      # other contexts push a's blobs out
        d = _Dev(L)
        toks = [100 + 10 * s + j for j in range(10)]
        d.feed(toks)
        pc.save(d, toks)
    assert pc.match(a + [0]).k == 0 or pc.match(a + [0]).entry.key in pc.entries
    assert all(all(b in pc.store for b in e.blobs()) for e in pc.entries.values())


def test_prefix_cache_disk_limit_drops_whole_entries(tmp_path):
    """A big blob takes the disk over its limit by more than the oldest entry: that entry goes with
    its pages, the walk goes on past them, the writer lives and what is left restores."""
    L = _Layout()
    pc = ST.PrefixCache(ST.BlobStore(1000, 3500, tmp_path), L, "m0")
    ctxs = [[c * 1000 + t for t in range(1, 42)] for c in range(1, 5)]
    for a in ctxs:                          # about 800 bytes each, its record the most of it
        dev = _Dev(L)
        dev.feed(a)
        pc.save(dev, a)
    assert _flushes(pc.store)
    pc.store.put("big", _arr(1500, 1))      # (over RAM's limit: straight to disk)
    assert _flushes(pc.store)
    u = pc.store.usage()
    assert u["disk_bytes"] <= 3500 and u["spilling_bytes"] == 0 and u["dropped"] >= 2
    assert all(all(b in pc.store for b in e.blobs()) for e in pc.entries.values())
    assert 0 < len(pc.entries) < len(ctxs)
    last = ctxs[-1]
    fresh = _Dev(L)
    assert pc.restore(fresh, pc.match(last + [7])) == len(last) and fresh.holds(last)


def _saved(tmp_path, n=4, ram=1000, disk=3500, stateful=False):
    """A store on disk, whole: n contexts of 41 tokens (10 pages and a tail)."""
    L = _Layout(stateful=stateful)
    pc = ST.PrefixCache(ST.BlobStore(ram, disk, tmp_path), L, "m0")
    ctxs = [[c * 1000 + t for t in range(1, 42)] for c in range(1, n + 1)]
    for a in ctxs:
        dev = _Dev(L)
        dev.feed(a)
        pc.save(dev, a)
    assert _flushes(pc.store)
    return pc, L, ctxs


def _file(tmp_path, key):
    return tmp_path / f"{key}.bin"


def test_a_save_beside_the_writers_eviction_does_not_deadlock(tmp_path):
    """The writer drops entries holding the store's lock and save puts holding the cache's: two
    locks, each waiting for the other, would hang the server once the disk is at its limit."""
    pc, L, _ = _saved(tmp_path)
    dropping, done = threading.Event(), threading.Event()
    drop = pc.store.on_drop

    def slow(k):
        dropping.set()
        drop(k)

    pc.store.on_drop = slow

    def saver():
        with pc.lock:                       # (as save's second half)
            pc.store.put("big", _arr(1500, 1))
            dropping.wait(1)
            pc.store.put("next", _arr(10, 2))
        done.set()

    threading.Thread(target=saver, daemon=True).start()
    assert done.wait(10) and _flushes(pc.store)
    assert pc.store.usage()["dropped"] >= 1 and pc.store.usage()["writer_errors"] == 0


@pytest.mark.parametrize("stateful", [False, True])
@pytest.mark.parametrize("how", ["gone", "short"])
def test_a_restore_with_a_blob_gone_writes_nothing(tmp_path, how, stateful):
    """A page's file gone or cut short under a saved context: no part of it reaches the device,
    the entry goes and a request feeds from the start."""
    pc, L, ctxs = _saved(tmp_path, n=1, ram=10, disk=1 << 20, stateful=stateful)
    a = ctxs[0]
    m = pc.match(a)
    assert m.k == len(a)
    f = _file(tmp_path, m.entry.pages[6])
    if how == "gone":
        f.unlink()
    else:
        f.write_bytes(f.read_bytes()[:3])
    fresh = _Dev(L)
    assert pc.restore(fresh, m) == 0
    assert fresh.cells == {} and not hasattr(fresh, "state")
    assert pc.match(a).k == 0 and pc.stats["restores"] == 0


def test_a_failed_eviction_leaves_a_store_that_restores_clean(tmp_path):
    """A writer stopped within an eviction, by the real path (the page's file unlinked, its entry
    not dropped), so that the entry on disk and in RAM misses a blob. The running server restores
    nothing of it and the writer goes on; the next server on the directory, with what a dead
    writer leaves beside it (a .tmp, a blob no entry uses, a record cut short, an entry with a
    page cut short), loads only what is whole."""
    pc, L, ctxs = _saved(tmp_path)
    drop = pc.store.on_drop
    failed = []

    def fail_once(k):
        if not failed:
            failed.append(k)
            raise KeyError(k)
        drop(k)

    pc.store.on_drop = fail_once
    pc.store.put("m0-t-" + "ee" * 16, _arr(1500, 1))    # the disk over its limit: its oldest page
    assert _flushes(pc.store)
    u = pc.store.usage()
    assert failed and u["writer_errors"] == 1 and u["writer_alive"] is True
    victim = next(a for a in ctxs if failed[0] in pc.match(a).entry.blobs())
    assert not _file(tmp_path, failed[0]).exists()
    fresh = _Dev(L)
    assert pc.restore(fresh, pc.match(victim)) == 0 and fresh.cells == {}
    assert pc.match(victim).k == 0          # (fed from the start)
    pc.store.on_drop = drop

    alive = [a for a in ctxs if pc.match(a).k]
    assert len(alive) >= 2
    short = pc.match(alive[0]).entry
    keep = pc.match(alive[1]).entry
    # what else a dead writer and a kill leave beside them
    (tmp_path / f".{keep.pages[0]}.tmp").write_bytes(b"part of a file")
    _file(tmp_path, "m0-p9-" + "ab" * 16).write_bytes(bytes(L.page_bytes()))
    _file(tmp_path, "m0-e-" + "cd" * 16).write_bytes(b"\x40" + bytes(7) + b"{not json")
    f = _file(tmp_path, short.pages[3])
    f.write_bytes(f.read_bytes()[:3])
    missing = pc.match(alive[-1]).entry if len(alive) > 2 else None
    if missing is not None:
        _file(tmp_path, missing.tail).unlink()

    s2 = ST.BlobStore(1000, 3500, tmp_path)
    pc2 = ST.PrefixCache(s2, L, "m0")
    assert not list(tmp_path.glob(".*.tmp"))
    assert not _file(tmp_path, "m0-p9-" + "ab" * 16).exists()
    assert set(pc2.entries) == {keep.key}
    assert all(k in pc2.refs or k in pc2.entries for k in s2.keys())
    assert s2.disk_used == sum(f.stat().st_size for f in tmp_path.glob("*.bin"))
    assert s2.disk_used <= 3500
    for a in ctxs:
        fresh = _Dev(L)
        got = pc2.restore(fresh, pc2.match(a))
        assert (got == len(a)) == (pc2.entries.get(pc2.entry_key(np.asarray(a))) is not None)
        if got:
            assert fresh.holds(a) and s2.usage()["disk_hits"] > 0
        else:
            assert fresh.cells == {}


def test_a_directory_over_the_limit_loses_whole_entries_boosted_last(tmp_path):
    """A disk found over its limit: the blobs the entries restored most hold go last, and a
    dropped blob takes its entries (not a page gone from under a record the load then refuses)."""
    pc, L, ctxs = _saved(tmp_path)
    oldest = pc.match(ctxs[0])
    for _ in range(ST.BOOST_HITS):          # the oldest entry's restores: boosted, on its record
        assert pc.restore(_Dev(L), pc.match(ctxs[0])) == len(ctxs[0])
    assert _flushes(pc.store)

    s2 = ST.BlobStore(1000, 2000, tmp_path)
    pc2 = ST.PrefixCache(s2, L, "m0")
    assert s2.disk_used <= 2000
    assert oldest.entry.key in pc2.entries and 0 < len(pc2.entries) < len(ctxs)
    for e in pc2.entries.values():
        assert all(k in s2 for k in e.blobs())


def test_a_writer_gone_shows_and_flush_returns(tmp_path):
    s = ST.BlobStore(ram_bytes=100, disk_bytes=1000, dir=tmp_path)
    assert s.usage()["writer_alive"] is True and s.usage()["writer_errors"] == 0
    assert ST.BlobStore(ram_bytes=100).usage()["writer_alive"] is None

    def gone():
        raise SystemExit

    s._q.get = gone
    s.put("a", _arr(100, 1))
    s.put("b", _arr(100, 2))                # (a's item wakes the writer, its next get ends it)
    s._thread.join(10)
    assert s.usage()["writer_alive"] is False
    s.put("c", _arr(100, 3))
    assert _flushes(s)


def test_prefix_cache_entries_survive_a_restart(tmp_path):
    L = _Layout()
    pc = ST.PrefixCache(ST.BlobStore(1 << 20, 1 << 20, tmp_path), L, "m0")
    a = list(range(1, 11))
    dev = _Dev(L)
    dev.feed(a)
    pc.save(dev, a)
    pc.store.discard("nothing")
    # spill everything to disk: a store with no RAM to speak of
    for k in list(pc.store.ram):
        pc.store.ram_bytes = 0
    pc.store._evict_ram()
    pc.store.flush()
    pc2 = ST.PrefixCache(ST.BlobStore(1 << 20, 1 << 20, tmp_path), L, "m0")
    m = pc2.match(a + [3])
    assert m.k == 10
    fresh = _Dev(L)
    pc2.restore(fresh, m)
    assert fresh.holds(a)
    assert ST.PrefixCache(ST.BlobStore(1 << 20, 1 << 20, tmp_path), L, "m1").match(a).k == 0


def test_a_state_saved_in_another_format_is_not_restored(tmp_path):
    """A folded DeltaNet's state (Spec.dn_fold: log2(e) S; kvio.Layout.sf "dn_fold") and an
    unfolded one's in one store under one model key: each layout's keys and entries its own
    (across a restart too), and restore refuses an entry of the other format."""
    a = list(range(1, 11))
    lays = {sf: _Layout(stateful=True, sf=sf) for sf in ("", "dn_fold")}
    got = {}
    for sf, L in lays.items():
        pc = ST.PrefixCache(ST.BlobStore(1 << 20, 1 << 20, tmp_path), L, "m0")
        dev = _Dev(L)
        dev.feed(a)
        got[sf] = pc.save(dev, a)
        assert got[sf].sf == sf and pc.mk == ("m0.dn_fold" if sf else "m0")
        pc.store.ram_bytes = 0                      # everything onto disk
        pc.store._evict_ram()
        pc.store.flush()
    assert got[""].tail != got["dn_fold"].tail
    for sf, L in lays.items():                      # a restart on the same directory
        pc = ST.PrefixCache(ST.BlobStore(1 << 20, 1 << 20, tmp_path), L, "m0")
        assert [e.sf for e in pc.entries.values()] == [sf]
        m = pc.match(a + [5])
        assert m.k == 10 and m.entry.sf == sf
        assert pc.restore(_Dev(L), m) == m.k
        other = got["" if sf else "dn_fold"]
        with pytest.raises(ValueError, match="not restored"):
            pc.restore(_Dev(L), ST.Match(other.n, other))


def test_a_qwen35_layout_names_its_state_format():
    """kvio.Layout.sf from the image (llm.snapshot.state_format): "dn_fold" on a folded Qwen3.5
    image, "" on an unfolded one and on a model without state."""
    import dataclasses
    from opentpu.isasim import board_config
    from opentpu.llm import qwen3, qwen35
    from opentpu.serve.kvio import Layout
    spec = qwen35.Spec(256, (qwen35.LIN, qwen35.ATTN), 8, 2, 256, 64, 8, 128, 128, 512, 1000)
    cfg = board_config(DRAM_BYTES=1 << 25)
    sfs = [Layout(qwen35.Image(dataclasses.replace(spec, dn_fold=f), cfg, 128), cfg.S).sf
           for f in (False, True)]
    assert sfs == ["", "dn_fold"]
    q3 = qwen3.Spec(256, 2, 4, 2, 128, 512, 1000)
    assert Layout(qwen3.Image(q3, cfg, 128), cfg.S).sf == ""


def test_pinned_entries_stay():
    pc, L = _cache(ram=60)
    a = list(range(1, 11))
    dev = _Dev(L)
    dev.feed(a)
    e = pc.save(dev, a, pin=True)
    for s in range(6):
        d = _Dev(L)
        toks = [200 + 10 * s + j for j in range(10)]
        d.feed(toks)
        pc.save(d, toks)
    assert e.key in pc.entries and pc.match(a + [1]).k == 10
    pc.unpin(e)
    with pytest.raises(ValueError):
        pc.save(dev, [])


def test_a_prefix_restored_twice_outlasts_newer_contexts():
    """A prefix entry (Entry.prefix: kept as a prompt was fed) restored BOOST_HITS times is
    boosted: newer contexts filling the RAM push the others out first (another prefix restored
    once among them), and the entry count's limit takes the oldest unboosted entry; the hits on
    prefixes counted apart."""
    pc, L = _cache(stateful=True, ram=150)
    system, other = list(range(1, 9)), list(range(50, 58))
    for toks in (system, other):
        d = _Dev(L)
        d.feed(toks)
        pc.save(d, toks, prefix=True)
    es = pc.entries[pc.entry_key(system)]
    assert es.prefix and pc.stats["prefix_saves"] == 2
    for _ in range(ST.BOOST_HITS):
        assert pc.restore(_Dev(L), pc.match(system + [3])) == 8
    assert pc.restore(_Dev(L), pc.match(other + [3])) == 8
    assert pc.stats["prefix_hits"] == ST.BOOST_HITS + 1 and pc.stats["prefix_hit_tokens"] == 24
    assert pc.stats["boosted"] == 1 and set(es.blobs()) <= pc.store.boosted
    for s in range(8):                                  # a long chat's contexts
        d = _Dev(L)
        toks = [100 + 10 * s + j for j in range(10)]
        d.feed(toks)
        pc.save(d, toks)
    assert es.key in pc.entries and all(b in pc.store for b in es.blobs())
    assert pc.entry_key(other) not in pc.entries
    assert pc.match(system + [3]).k == 8
    pc.max_entries = len(pc.entries)                    # the entries' limit
    d = _Dev(L)
    d.feed([7] * 9)
    pc.save(d, [7] * 9)
    assert es.key in pc.entries


def test_a_prefix_survives_a_restart_boosted(tmp_path):
    """A prefix entry's blobs go to disk as it is saved (its shared pages too): a store started
    on the directory without the first's shutdown finds it, with its prefix flag and its hits
    (boosted again), and restores it."""
    L = _Layout(stateful=True)
    pc = ST.PrefixCache(ST.BlobStore(1 << 20, 1 << 20, tmp_path), L, "m0")
    a = list(range(1, 11))
    d = _Dev(L)
    d.feed(a[:8])
    pc.save(d, a[:8])                       # (not a prefix: its pages in RAM only)
    d.feed(a[8:])
    pc.save(d, a, prefix=True)              # shares page 0 and 1 with it
    for _ in range(ST.BOOST_HITS):
        pc.restore(_Dev(L), pc.match(a + [5]))
    pc.store.flush()
    pc2 = ST.PrefixCache(ST.BlobStore(1 << 20, 1 << 20, tmp_path), L, "m0")
    m = pc2.match(a + [5])
    assert m.k == 10 and m.entry.prefix and m.entry.hits == ST.BOOST_HITS
    assert set(m.entry.blobs()) <= pc2.store.boosted
    fresh = _Dev(L)
    assert pc2.restore(fresh, m) == 10 and fresh.holds(a)
    assert pc2.match(a[:8] + [5]).k == 0     # (the other's state never reached the disk)
