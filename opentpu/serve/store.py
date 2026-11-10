"""The server's KV store on the host: saved contexts (opentpu/serve/kvio.py's pages and state)
in RAM, spilled to files on disk, and the prefix index that finds the longest saved prefix of a
prompt (RadixAttention's role, at the granularity of a context's pages).

BlobStore holds byte blobs by key: the most recently used in RAM up to `ram_bytes`; past that
the least recently used go to files under `dir` (a writer thread writes them; a blob being
written is still read from memory) up to `disk_bytes`, past which the least recently used are
dropped. A blob read from disk comes back into RAM (its file stays). Pinned blobs (an active
request's context) are never evicted; boosted ones (a context restored BOOST_HITS times or more:
a system prompt that conversations share) go after every other in their tier. The files outlive
the process: a server started on the same directory finds the contexts saved before (their
entries are blobs too).

PrefixCache keeps contexts as entries: the tokens [0, n), the keys of their whole pages (page i
keyed by the model and the tokens [0, (i + 1) P): contexts with a common prefix share them) and
of the tail (the last partial page and, for a model with state, the state at n). A state's
format (kvio.Layout.sf, llm.snapshot.state_format: a folded DeltaNet's log2(e) S) is in the keys
and in the entry, and a state saved in another is never restored (PrefixCache.restore refuses
it). match(tokens)
finds where the device can continue a prompt from: for a model whose KV caches hold the whole
context, any common prefix k of a saved context (its pages up to k; what lies past k is
overwritten); for a model with state, the end of a saved context that the prompt continues (so
the scheduler keeps prefixes too, Entry.prefix: the contexts at a prompt's block positions and
at its last message boundary as it is fed, for other conversations that share them).
With an MTP drafter (mtp=True, opentpu/llm/mtp.py) the MTP layer's KV row at q took the token at
q + 1 (h_q, x_(q + 1)): page i is keyed by the tokens [0, (i + 1) P] (one more), the context's
last row's by the token after it (Entry.follow, the context's next token when it was saved), and
an entry may keep that row's residual (Entry.held: a part of a prompt before the reply's start),
from which a prompt that goes on with another token appends the row again
(mtp.MTPDecoder.refill).
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BOOST_HITS = 2          # an entry restored this often is evicted after the others (boosted)


class BlobStore:
    def __init__(self, ram_bytes: int, disk_bytes: int = 0, dir=None, on_drop=None):
        self.ram_bytes, self.disk_bytes = int(ram_bytes), int(disk_bytes)
        self.dir = Path(dir) if dir is not None and disk_bytes > 0 else None
        self.on_drop = on_drop
        self.lock = threading.RLock()
        self.ram: OrderedDict = OrderedDict()       # key -> bytes (least recently used first)
        self.spilling: dict = {}                    # key -> bytes being written to disk
        self.disk: OrderedDict = OrderedDict()      # key -> size of its file
        self.pins: dict = {}                        # key -> pin count
        self.boosted: set = set()                   # keys evicted after the others
        self.ram_used = self.disk_used = self.spill_used = 0
        self.stats = {"put": 0, "ram_hits": 0, "disk_hits": 0, "misses": 0, "spilled": 0,
                      "dropped": 0, "disk_read_s": 0.0, "writer_errors": 0}
        self._q: queue.Queue | None = None
        self._thread: threading.Thread | None = None
        if self.dir is not None:
            self.dir.mkdir(parents=True, exist_ok=True)
            for p in self.dir.glob(".*.tmp"):       # (a writer stopped within a file)
                try:
                    p.unlink()
                except OSError:
                    pass
            files = sorted(self.dir.glob("*.bin"), key=lambda p: p.stat().st_mtime)
            for p in files:                         # the files of an earlier server
                n = p.stat().st_size
                self.disk[p.stem] = n
                self.disk_used += n
            self._q = queue.Queue()
            self._thread = threading.Thread(target=self._writer, daemon=True,
                                            name="otpu-kv-writer")
            self._thread.start()

    # ---- the interface
    def __contains__(self, key: str) -> bool:
        with self.lock:
            return key in self.ram or key in self.spilling or key in self.disk

    def put(self, key: str, data: np.ndarray, durable: bool = False) -> None:
        """durable: on disk too as soon as the writer gets to it (kept in RAM)."""
        data = np.ascontiguousarray(data).view(np.uint8).reshape(-1)
        with self.lock:
            if key in self.ram:
                self.ram_used -= len(self.ram.pop(key))
            if durable and key in self.disk:        # (a record rewritten: its file again)
                self.disk_used -= self.disk.pop(key)
            self.ram[key] = data
            self.ram_used += len(data)
            self.stats["put"] += 1
            if durable and self._q is not None and key not in self.disk:
                self._q.put(("copy", key))
            self._evict_ram()

    def to_disk(self, keys) -> None:
        """These blobs onto disk too as the writer gets to them (kept in RAM)."""
        if self._q is None:
            return
        with self.lock:
            for k in keys:
                if k in self.ram and k not in self.disk:
                    self._q.put(("copy", k))

    def persist(self) -> None:
        """Every blob onto disk (a server's shutdown: a later one finds them), as the disk's
        budget allows, most recently used last."""
        if self._q is None:
            return
        with self.lock:
            for k in self.ram:
                if k not in self.disk:
                    self._q.put(("copy", k))
        self.flush()

    def get(self, key: str) -> np.ndarray | None:
        with self.lock:
            if key in self.ram:
                self.ram.move_to_end(key)
                self.stats["ram_hits"] += 1
                return self.ram[key]
            if key in self.spilling:
                self.stats["ram_hits"] += 1
                return self.spilling[key]
            if key not in self.disk:
                self.stats["misses"] += 1
                return None
            t0 = time.perf_counter()
            try:
                data = np.fromfile(self._path(key), np.uint8)
            except OSError:
                self._drop_disk(key)
                self.stats["misses"] += 1
                return None
            self.stats["disk_read_s"] += time.perf_counter() - t0
            self.stats["disk_hits"] += 1
            self.disk.move_to_end(key)
            self.ram[key] = data                    # back in RAM; the file stays
            self.ram_used += len(data)
            self._evict_ram()
            return data

    def pin(self, keys) -> None:
        with self.lock:
            for k in keys:
                self.pins[k] = self.pins.get(k, 0) + 1

    def unpin(self, keys) -> None:
        with self.lock:
            for k in keys:
                n = self.pins.get(k, 0) - 1
                if n > 0:
                    self.pins[k] = n
                else:
                    self.pins.pop(k, None)
            self._evict_ram()
            self._evict_disk()

    def boost(self, keys) -> None:
        """These blobs go after every other in their tier (until discarded)."""
        with self.lock:
            self.boosted.update(keys)

    def discard(self, key: str) -> None:
        """Remove a blob from every tier (no on_drop call)."""
        with self.lock:
            self.boosted.discard(key)
            if key in self.ram:
                self.ram_used -= len(self.ram.pop(key))
            if key in self.spilling:
                self.spill_used -= len(self.spilling.pop(key))
            if key in self.disk:
                self.disk_used -= self.disk.pop(key)
                self._unlink(key)

    def flush(self) -> None:
        """Wait until the blobs being spilled are on disk (or the writer is gone: never for
        ever)."""
        if self._q is None:
            return
        with self._q.all_tasks_done:
            while self._q.unfinished_tasks and self._thread.is_alive():
                self._q.all_tasks_done.wait(1.0)

    def size(self, key: str) -> int | None:
        """A blob's bytes in whichever tier holds it, or None."""
        with self.lock:
            d = self.ram.get(key)
            d = self.spilling.get(key) if d is None else d
            return self.disk.get(key) if d is None else len(d)

    def trim(self) -> None:
        """Back within the disk's limit (a directory found over it: after the entries are known,
        so that boosted blobs go last and a dropped blob takes its entries with it)."""
        with self.lock:
            self._evict_disk()

    def usage(self) -> dict:
        with self.lock:
            return {"ram_bytes": self.ram_used, "ram_blobs": len(self.ram),
                    "spilling_bytes": self.spill_used,
                    "disk_bytes": self.disk_used, "disk_blobs": len(self.disk),
                    "ram_limit": self.ram_bytes, "disk_limit": self.disk_bytes,
                    "writer_alive": None if self._thread is None else self._thread.is_alive(),
                    **self.stats}

    def keys(self) -> list[str]:
        with self.lock:
            return list(dict.fromkeys([*self.disk, *self.spilling, *self.ram]))

    # ---- the tiers
    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.bin"

    def _evict_ram(self) -> None:
        if self.ram_used <= self.ram_bytes:
            return
        for key in [k for k in self.ram if k not in self.boosted] + \
                [k for k in self.ram if k in self.boosted]:     # (boosted ones last)
            if self.ram_used <= self.ram_bytes:
                break
            if key in self.pins or key not in self.ram:     # (a drop took it with its entry)
                continue
            data = self.ram.pop(key)
            self.ram_used -= len(data)
            if key in self.disk:                    # already on disk: drop the RAM copy
                pass
            elif self.dir is not None and len(data) <= self.disk_bytes:
                self.spilling[key] = data           # written by the writer thread
                self.spill_used += len(data)
                self._q.put(("spill", key))
                self.stats["spilled"] += 1
            else:
                self._dropped(key)

    def _writer(self) -> None:
        while True:
            how, key = self._q.get()
            try:
                self._write(how, key)
            except BaseException:                   # (a dead writer would keep every later spill
                self._writer_failed()               # in RAM and leave flush waiting for ever)
            finally:
                self._q.task_done()

    def _writer_failed(self) -> None:
        self.stats["writer_errors"] += 1
        try:
            traceback.print_exc()
        except BaseException:                       # (no stderr left to tell)
            pass

    def _write(self, how: str, key: str) -> None:
        try:
            if how == "copy":                       # a RAM blob's file (it stays in RAM)
                with self.lock:
                    data = self.ram.get(key)
                if data is not None and key not in self.disk:
                    tmp = self.dir / f".{key}.tmp"
                    data.tofile(tmp)
                    os.replace(tmp, self._path(key))
                    with self.lock:
                        if key in self.ram and key not in self.disk:
                            self.disk[key] = len(data)
                            self.disk_used += len(data)
                            self._evict_disk()
                        elif key not in self.disk:
                            self._unlink(key)
                return
            with self.lock:
                data = self.spilling.get(key)
            if data is not None:
                tmp = self.dir / f".{key}.tmp"
                data.tofile(tmp)
                os.replace(tmp, self._path(key))
                with self.lock:
                    if self.spilling.pop(key, None) is not None:
                        self.spill_used -= len(data)
                        self.disk[key] = len(data)
                        self.disk_used += len(data)
                        self._evict_disk()
                    else:                           # discarded meanwhile
                        self._unlink(key)
        except OSError:
            self._writer_failed()
            try:
                (self.dir / f".{key}.tmp").unlink(missing_ok=True)
            except OSError:
                pass
            with self.lock:
                d = self.spilling.pop(key, None)
                if d is not None:
                    self.spill_used -= len(d)
                    self._dropped(key)

    def _evict_disk(self) -> None:
        if self.disk_used <= self.disk_bytes:
            return
        for key in [k for k in self.disk if k not in self.boosted] + \
                [k for k in self.disk if k in self.boosted]:
            if self.disk_used <= self.disk_bytes:
                break
            if key in self.pins or key in self.ram or key not in self.disk:   # (a drop took it)
                continue
            self._drop_disk(key)

    def _drop_disk(self, key: str) -> None:
        self.disk_used -= self.disk.pop(key)
        self._unlink(key)
        if key not in self.ram and key not in self.spilling:
            self._dropped(key)

    def _unlink(self, key: str) -> None:
        try:
            self._path(key).unlink()
        except OSError:
            pass

    def _dropped(self, key: str) -> None:
        self.stats["dropped"] += 1
        if self.on_drop is not None:
            self.on_drop(key)


def model_key(*parts) -> str:
    """A short key of what decides a context's bytes: the checkpoint, the image's formats and
    layout (16 hex digits)."""
    return hashlib.sha256(repr(parts).encode()).hexdigest()[:16]


def _tok_bytes(tokens) -> bytes:
    return np.asarray(tokens, np.int64).astype("<i4").tobytes()


@dataclass
class Entry:
    """A saved context: tokens [0, n), its whole pages' keys, its tail's key (the last partial
    page's m positions, then the state) or None; an MTP context's follow (the token its MTP
    layer's last row took, or None) and held (the key of that row's residual, or None); sf, its
    state's format (kvio.Layout.sf); prefix, a prompt's prefix kept as it was fed (no reply:
    for other conversations sharing it); hits, its restores (BOOST_HITS: boosted)."""
    key: str
    tokens: np.ndarray
    pages: list[str]
    tail: str | None
    m: int = 0
    used: float = field(default_factory=time.monotonic)
    follow: int | None = None
    held: str | None = None
    sf: str = ""
    prefix: bool = False
    hits: int = 0

    @property
    def n(self) -> int:
        return len(self.tokens)

    def blobs(self) -> list[str]:
        return self.pages + ([self.tail] if self.tail else []) + ([self.held] if self.held else [])


@dataclass
class Match:
    """Where a prompt continues from: its first k tokens, restored from `entry` (None: nothing
    saved matches)."""
    k: int
    entry: Entry | None = None


class PrefixCache:
    """Contexts saved in a BlobStore for one model's layout (kvio.Layout), by their tokens; the
    keys are mkey's and, with a state of a format of its own (Layout.sf), that format's."""

    def __init__(self, store: BlobStore, layout, mkey: str, max_entries: int = 4096,
                 mtp: bool = False):
        self.store, self.L = store, layout
        self.mk = f"{mkey}.{layout.sf}" if layout.sf else mkey
        self.max_entries, self.mtp = max_entries, mtp
        self.entries: OrderedDict = OrderedDict()   # entry key -> Entry (oldest use first)
        self.refs: dict = {}                        # blob key -> entries using it
        self.lock = store.lock                      # (its writer drops entries holding the
        #                                             store's lock, save puts holding this one)
        self.stats = {"saved_pages": 0, "shared_pages": 0, "saved_tails": 0, "restored_pages": 0,
                      "restores": 0, "hit_tokens": 0, "lookups": 0, "prefix_saves": 0,
                      "prefix_hits": 0, "prefix_hit_tokens": 0, "boosted": 0}
        prev = store.on_drop
        store.on_drop = lambda k: (self._blob_dropped(k), prev and prev(k))
        self._load()

    # ---- keys
    def page_key(self, tokens, i: int) -> str:
        """Page i's key: the tokens [0, (i + 1) P) (MTP: [0, (i + 1) P], `tokens` with the
        context's follow after them)."""
        P = self.L.P
        return f"{self.mk}-p{i}-" + hashlib.sha256(
            _tok_bytes(tokens[:(i + 1) * P + self.mtp])).hexdigest()[:32]

    def entry_key(self, tokens) -> str:
        return f"{self.mk}-e-" + hashlib.sha256(_tok_bytes(tokens)).hexdigest()[:32]

    # ---- lookup
    def match(self, tokens, limit: int | None = None) -> Match:
        """The longest prefix of `tokens` (at most `limit` of them) the device can continue
        from a saved context."""
        q = np.asarray(tokens, np.int64)
        limit = len(q) if limit is None else min(limit, len(q))
        best = Match(0)
        with self.lock:
            self.stats["lookups"] += 1
            for e in self.entries.values():
                n = min(e.n, limit)
                if n <= best.k:
                    continue
                eq = e.tokens[:n] == q[:n]
                c = n if eq.all() else int(np.argmin(eq))
                if self.L.stateful:
                    if c == e.n and c > best.k:
                        best = Match(c, e)
                elif c > best.k:
                    best = Match(c, e)
        return best

    # ---- save and restore
    def save(self, io, tokens, pin: bool = False, follow: int | None = None,
             held: np.ndarray | None = None, state_at: int | None = None,
             prefix: bool = False) -> Entry:
        """The device's context (its first len(tokens) positions hold `tokens`) into the store:
        the whole pages it does not have yet and the tail (last partial page, state); returns
        the entry (pinned: its blobs kept until unpin). MTP: follow, the token the MTP layer's
        last row took (None: not known, -1 in its pages' keys), and held, that row's residual
        (or None). state_at: where a snapshot of the context's state lies on the device
        (kvio.snapshot_at), read from there. prefix: a prompt's prefix as it is fed (Entry.
        prefix), its blobs on disk at once (durable: the next server finds it)."""
        P, L = self.L.P, self.L
        toks = np.asarray(tokens, np.int64)
        n = len(toks)
        if n == 0:
            raise ValueError("an empty context")
        key = self.entry_key(toks)
        full, m = n // P, n % P
        ext = np.append(toks, -1 if follow is None else follow) if self.mtp else toks
        pages = [self.page_key(ext, i) for i in range(full)]
        tail = None
        if m or L.stateful:
            tail = f"{self.mk}-t-" + key.rsplit("-", 1)[1]
        hkey = None if held is None else f"{self.mk}-h-" + key.rsplit("-", 1)[1]
        with self.lock:
            old = self.entries.get(key)
            if old is not None and all(b in self.store for b in old.blobs()):
                old.used = time.monotonic()
                self.entries.move_to_end(key)
                if hkey is not None and old.held is None:    # (the residual row: the
                    self.store.put(hkey, np.asarray(held, np.float32).view(np.uint8).copy())
                    old.held = hkey                         # tokens' alone)
                    self.refs.setdefault(hkey, set()).add(key)
                    self._persist(old)
                if prefix and not old.prefix:               # (kept before as a reply's start)
                    old.prefix = True
                    self.store.to_disk(old.blobs())
                    self._persist(old)
                if pin:
                    self.store.pin(old.blobs())
                return old
            missing = [i for i, k in enumerate(pages) if k not in self.store]
        # the device reads (no lock: the engine's thread)
        got = io.read_pages([(i, P) for i in missing]) if missing else []
        tail_data = None
        if tail is not None:
            parts = io.read_pages([(full, m)]) if m else []
            if L.stateful:
                parts.append(io.read_state() if state_at is None else io.read_state(at=state_at))
            tail_data = np.concatenate(parts) if len(parts) > 1 else parts[0]
        e = Entry(key, toks.copy(), pages, tail, m, follow=follow, held=hkey, sf=L.sf,
                  prefix=prefix)
        with self.lock:
            for i, d in zip(missing, got):
                self.store.put(pages[i], d, durable=prefix)
            if hkey is not None:
                self.store.put(hkey, np.asarray(held, np.float32).view(np.uint8).copy())
            self.stats["saved_pages"] += len(missing)
            self.stats["shared_pages"] += len(pages) - len(missing)
            if tail is not None:
                self.store.put(tail, tail_data, durable=prefix)
                self.stats["saved_tails"] += 1
            if prefix:
                self.stats["prefix_saves"] += 1
                self.store.to_disk(pages)           # (the pages it shares: on disk too)
            if pin:
                self.store.pin(e.blobs())
            self._add(e)
            self._persist(e)
            if not L.stateful:                  # contexts this one continues: covered by it
                for o in list(self.entries.values()):
                    if o is not e and o.n <= n and not self._is_pinned(o) and \
                            np.array_equal(o.tokens, toks[:o.n]):
                        self._remove(o)
            while len(self.entries) > self.max_entries:   # (the oldest, boosted ones last)
                self._remove(min(self.entries.values(),
                                 key=lambda o: (o.hits >= BOOST_HITS, o.used)))
        return e

    def restore(self, io, match: Match, have: int = 0) -> int:
        """Write the pages (and state) of match.entry for its first match.k tokens into the
        device's caches, but the whole pages within its first `have` positions (the device
        holds those tokens there already); returns k (0: nothing restored, a blob gone)."""
        e, k = match.entry, match.k
        if e is None or k == 0:
            return 0
        if self.L.stateful and e.sf != self.L.sf:
            raise ValueError(f"a state saved as {e.sf or 'unfolded'!r}, this image's "
                             f"{self.L.sf or 'unfolded'!r}: not restored")
        P = self.L.P
        need = -(-k // P)
        writes = []
        with self.lock:                             # (every blob in hand before the first write)
            for i in range(min(have, k) // P, need):
                if i < len(e.pages):
                    d = self._get(e.pages[i], self.L.page_bytes())
                    if d is None:
                        self._remove(e)
                        return 0
                    writes.append((i, P, d))
                else:                               # the tail's partial page
                    d = self._get(e.tail, self._tail_bytes(e))
                    if d is None:
                        self._remove(e)
                        return 0
                    writes.append((i, e.m, d[:self.L.page_bytes(e.m)]))
            state = None
            if self.L.stateful:
                d = self._get(e.tail, self._tail_bytes(e))
                if d is None:
                    self._remove(e)
                    return 0
                state = d[len(d) - self.L.state_bytes:]
            e.used = time.monotonic()
            self.entries.move_to_end(e.key)
        io.write_pages(writes)
        if state is not None:
            io.write_state(state)
        with self.lock:
            self.stats["restores"] += 1
            self.stats["restored_pages"] += len(writes)
            self.stats["hit_tokens"] += k
            if e.prefix:
                self.stats["prefix_hits"] += 1
                self.stats["prefix_hit_tokens"] += k
            e.hits += 1
            if e.hits == BOOST_HITS and e.key in self.entries:
                self._boost(e)
                self._persist(e)
        return k

    def held_row(self, e: Entry) -> np.ndarray | None:
        """The residual row an MTP entry keeps (Entry.held), or None."""
        d = None if e is None or e.held is None else self.store.get(e.held)
        return None if d is None else d.view(np.float32)

    def unpin(self, e: Entry) -> None:
        self.store.unpin(e.blobs())

    def clear(self) -> None:
        with self.lock:
            for e in list(self.entries.values()):
                self._remove(e)

    def usage(self) -> dict:
        with self.lock:
            return {"entries": len(self.entries),
                    "tokens": int(sum(e.n for e in self.entries.values())), **self.stats}

    # ---- bookkeeping
    def _tail_bytes(self, e: Entry) -> int:
        return (self.L.page_bytes(e.m) if e.m else 0) + (self.L.state_bytes if self.L.stateful else 0)

    def _get(self, key: str, size: int) -> np.ndarray | None:
        """A blob at its size, or None (a file cut short is as good as gone)."""
        d = self.store.get(key)
        return d if d is not None and len(d) == size else None

    def _whole(self, e: Entry) -> bool:
        """Every blob of e there, at its size."""
        return all(self.store.size(b) == self.L.page_bytes() for b in e.pages) and \
            (e.tail is None or self.store.size(e.tail) == self._tail_bytes(e)) and \
            (e.held is None or e.held in self.store)

    def _is_pinned(self, e: Entry) -> bool:
        return any(b in self.store.pins for b in e.blobs())

    def _add(self, e: Entry) -> None:
        self.entries[e.key] = e
        self.entries.move_to_end(e.key)
        for b in e.blobs():
            self.refs.setdefault(b, set()).add(e.key)

    def _boost(self, e: Entry) -> None:
        """e's blobs evicted after the others' (its shared pages with them)."""
        self.store.boost(e.blobs() + [e.key])
        self.stats["boosted"] += 1

    def _remove(self, e: Entry) -> None:
        if self.entries.pop(e.key, None) is None:
            return
        self.store.discard(e.key)
        for b in e.blobs():
            users = self.refs.get(b)
            if users is not None:
                users.discard(e.key)
                if not users:
                    del self.refs[b]
                    self.store.discard(b)

    def _blob_dropped(self, key: str) -> None:
        with self.lock:
            for ek in list(self.refs.get(key, ())):
                e = self.entries.get(ek)
                if e is not None:
                    self._remove(e)
            e = self.entries.get(key)              # an entry's own record
            if e is not None:
                self._remove(e)

    def _persist(self, e: Entry) -> None:
        if self.store.dir is None:
            return
        meta = json.dumps({"pages": e.pages, "tail": e.tail, "m": e.m, "follow": e.follow,
                           "held": e.held, "sf": e.sf, "prefix": e.prefix,
                           "hits": e.hits}).encode()
        rec = np.frombuffer(len(meta).to_bytes(8, "little") + meta + _tok_bytes(e.tokens),
                            np.uint8)
        self.store.put(e.key, rec, durable=True)

    def _load(self) -> None:
        """Entries an earlier server saved on disk whose blobs are all still there, whole; what
        it left beside them (an entry missing a blob, blobs no entry uses: a server killed or a
        writer stopped part way) goes, then the disk back within its limit."""
        if self.store.dir is None:
            return
        keep = set()
        for k in self.store.keys():
            if not k.startswith(f"{self.mk}-e-"):
                continue
            rec = self.store.get(k)
            try:
                n = int.from_bytes(rec[:8].tobytes(), "little")
                meta = json.loads(rec[8:8 + n].tobytes())
                toks = np.frombuffer(rec[8 + n:].tobytes(), "<i4").astype(np.int64)
                e = Entry(k, toks, list(meta["pages"]), meta["tail"], int(meta["m"]),
                          follow=meta.get("follow"), held=meta.get("held"), sf=meta.get("sf", ""),
                          prefix=bool(meta.get("prefix")), hits=int(meta.get("hits", 0)))
            except (ValueError, TypeError, KeyError, AttributeError):
                self.store.discard(k)
                continue
            if self.L.stateful and e.sf != self.L.sf:
                keep.add(k)                 # (another format's: restore refuses it)
                keep.update(e.blobs())
                continue
            if self._whole(e) and self.entry_key(toks) == k:
                self._add(e)
                if e.hits >= BOOST_HITS:
                    self._boost(e)
            else:
                self.store.discard(k)
        for k in self.store.keys():
            if k.startswith(f"{self.mk}-") and k not in self.entries and k not in self.refs \
                    and k not in keep:
                self.store.discard(k)
        self.store.trim()
