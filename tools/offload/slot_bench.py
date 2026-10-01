"""The expert server's write rate on the card, without a model (docs/offload.md section 10):
requests of k missing experts served into DRAM slots, through BackendDram (Board.write) and
BoardDram (one-pass channel runs, a DMA thread, the host's words from a shadow). The script plays
the card's side of the mailbox itself (it posts each request's ids and seq), so no program runs.
It writes whole 64-byte beats only, into a scratch layout at --base, after zeroing its mailbox
and directory (a scrubbed card: no read of a beat never written).

    python3 tools/offload/slot_bench.py                 # the card (/dev/xdma0), under its lock
    python3 tools/offload/slot_bench.py --null          # host only: a transport that drops
                                                        # the DMA (the staging's CPU cost)

The experts come from host RAM (the page cache's case, warm), 1.67 MB (Qwen3.5-35B-A3B) and
5.85 MB (LFM2.5-8B-A1B) by default.
"""
from __future__ import annotations

import argparse
import json
import time
from types import SimpleNamespace

import numpy as np

SIZES = {"q35": 1671168, "lfm8b": 5849088}      # bytes per expert slot (ExpertFormat, fp4)


def bench(board, slot: int, k: int, reqs: int, base: int, kind: str, pool_file=None,
          reader="pread", warm=False, pieces=2) -> dict:
    from pathlib import Path

    from opentpu.host.offload import (LINE, SPLIT, BackendDram, BoardDram, ExpertServer, Layout,
                                      PoolFile)
    E = k * (reqs + 1)                          # every id new: k misses per request
    lay = Layout.build(base, E, k, (E,), slot)
    if pool_file:                   # a real pool file's packed experts, each memory its own E
        done = np.fromfile(str(pool_file) + ".packed", np.uint8)
        if Path(pool_file).stat().st_size != len(done) * slot:
            raise ValueError(f"{pool_file}: not {len(done)} slots of {slot} bytes")
        gids = np.nonzero(done)[0][(kind == "board") * E:][:E]
        if len(gids) < E:
            raise ValueError(f"{pool_file}: fewer than {2 * E} packed experts")
        fmt = Path(str(pool_file) + ".format")
        split = fmt.exists() and fmt.read_text().strip() == SPLIT
        if reader == "mmap":        # the memmap's rows (its pages; the split format's bytes
            mm = np.memmap(pool_file, np.uint8, "r")         # as if the slot's: a rate only)
            mm = mm[:len(mm) // slot * slot].reshape(-1, slot)
            pool = [mm[g] for g in gids]
        else:                       # MO.serve's reader, its warm thread first with --warm
            pf = PoolFile(pool_file, slot, split)
            if warm:
                pf.warm(gids).join()
            pool = None
    else:
        pool = np.random.default_rng(0).integers(0, 256, (E, slot), dtype=np.uint8)
    if kind == "board":
        mem = BoardDram(SimpleNamespace(board=board), lay, pieces=pieces)
    else:
        mem = BackendDram(SimpleNamespace(write=lambda s, a, d: board.write(a, d),
                                          read=lambda s, a, n: board.read(a, n)))
    srv = ExpertServer(mem, lay, (lambda g: pool[g]) if pool is not None else
                       (lambda g: pf.get(int(gids[g]))))
    srv.load(())
    t_serve, tm = 0.0, {"stage": 0.0, "flush": 0.0}
    for name in tm:                             # BoardDram: the staging, the waits for the DMA
        f = getattr(mem, "write_slot" if name == "stage" else "flush", None)
        if f is not None:
            def timed(*a, _f=f, _n=name):
                t0 = time.perf_counter()
                try:
                    return _f(*a)
                finally:
                    tm[_n] += time.perf_counter() - t0
            setattr(mem, "write_slot" if name == "stage" else "flush", timed)
    for seq in range(1, reqs + 1):
        ids = [(seq - 1) * k + i for i in range(k)]
        board.write(lay.row, np.array(ids + [0] * (LINE // 4 - k), np.float32))
        board.write(lay.mbox, np.float32(seq).tobytes())
        t0 = time.perf_counter()
        assert srv.poll() == 1
        t_serve += time.perf_counter() - t0
    mb = srv.bytes / 1e6
    r = dict(memory=type(mem).__name__, slot_mb=round(slot / 1e6, 3), k=k, requests=reqs,
             ms_per_expert=round(1e3 * t_serve / (reqs * k), 3),
             ms_per_request=round(1e3 * t_serve / reqs, 3), mb_s=round(mb / t_serve))
    if kind == "board":
        r["dma_gb_s"] = round(mem.dma_bytes / mem.dma_s / 1e9, 3) if mem.dma_s else None
        r["direct"] = mem.direct                # read from the file into the runs
        r["pieces"] = mem.pieces
        r.update({f"{x}_ms_per_expert": round(1e3 * v / (reqs * k), 3) for x, v in tm.items()})
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--null", action="store_true", help="no card: a transport that drops DMA")
    ap.add_argument("--base", type=lambda x: int(x, 0), default=0x8000_0000,
                    help="the scratch layout's logical DRAM address")
    ap.add_argument("--reqs", type=int, default=40, help="requests per size and memory")
    ap.add_argument("-k", type=int, default=3, help="missing experts per request")
    ap.add_argument("--sizes", nargs="*", default=list(SIZES))
    ap.add_argument("--pool", help="read the experts from this pool file (one --sizes entry: "
                                   "its slot size), not host RAM")
    ap.add_argument("--reader", choices=("pread", "mmap"), default="pread",
                    help="--pool: MO.serve's reader (os.preadv; a split-format file straight "
                         "into the channel runs), or the memmap's rows")
    ap.add_argument("--pieces", type=int, default=2,
                    help="BoardDram: the parts of an expert read when no DMA is in flight")
    ap.add_argument("--warm", action="store_true", help="--pool pread: read the experts into "
                                                        "the page cache first (the warm thread)")
    a = ap.parse_args()
    from opentpu.host.board import Board
    if a.null:
        from opentpu.host.fake import FakeTransport

        class Null(FakeTransport):
            """Keeps the mailbox and the directory (its first MiB), drops the slots' DMA."""
            threaded = True

            def mem_write(self, ch, off, data):
                if off + len(data) <= len(self.ch[ch]):
                    super().mem_write(ch, off, data)
        board = Board(Null(ch_bytes=1 << 20, devname=None))
        board.info()["caps"]["chash"] = True            # the production map's swaps
        a.base = 0
    else:
        from opentpu.host.board import XdmaTransport
        board = Board(XdmaTransport("/dev/xdma0"))
    try:
        for s in a.sizes:
            for kind in ("backend", "board"):
                print(json.dumps(dict(size=s, chash=board.chash,
                                      **bench(board, SIZES[s], a.k, a.reqs, a.base, kind,
                                              a.pool, a.reader, a.warm, a.pieces))),
                      flush=True)
    finally:
        board.close()


if __name__ == "__main__":
    main()
