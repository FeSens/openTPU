"""What the expert pool's disk costs a warm process (docs/minicpmv.md 6): the same text turn with
the pool in the page cache and out of it.

    python3 tools/offload/pool_disk.py MODEL --pool POOL --backend board --cfg PKL
        [--arms cached,dropped,dropped,cached | ""] [--turn TEXT] [--max-new 32] [--out JSON]

The engine opens as otpu-chat's (tools/qual/chatmoe.py's: streamed.engine, the family's formats,
its image built or read from the image cache) and a first turn warms it (its programs, the
slots); the pool's disk (rotational or not, /sys/block) is reported. Then each arm, in order:
- cached: the whole pool read into the page cache (its seconds: the disk's sequential rate);
- dropped: the pool's pages dropped from the page cache (the engine's maps of it released, then
  posix_fadvise DONTNEED);
an arm whose page cache is not what it says (mincore: under 95% of the pool's experts in it
cached, over 2% dropped) is marked INVALID;
then the conversation reset and the same turn asked, greedy. Per arm: prefill and decode tok/s
(wall and device), the slots' misses, and the pool's reads by where they came from (PoolFile.io:
"cached" reads whose pages were all in the page cache, "disk" the others; each its reads,
seconds and bytes), so a miss's service time from the disk against the page cache. The open's
read of every packed expert (PoolFile.warm's thread) is waited for after the warm-up turn.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "qual"))

TURN = "Explain in three sentences how a bicycle's gears make climbing easier."


def disk_of(path) -> dict:
    """The block device under path: its name and whether it is rotational (Linux), or {}."""
    try:
        st = os.stat(path)
        dev = f"{os.major(st.st_dev)}:{os.minor(st.st_dev)}"
        p = Path(f"/sys/dev/block/{dev}").resolve()
        q = p / "queue" if (p / "queue").exists() else p.parent / "queue"
        return {"dev": p.name, "rotational": (q / "rotational").read_text().strip() == "1"}
    except (OSError, ValueError):
        return {}


def cache(path, on: bool, pf=None) -> float:
    """The file into the page cache (on: read through) or out of it; returns the seconds.
    Out of it: posix_fadvise DONTNEED drops only pages no process maps, and the engine's
    PoolFile maps the pool (its `mapped` touches, its packing memmap), so those maps' pages
    are unmapped from this process first (madvise DONTNEED: the maps stay, a later read
    faults the page in again), and the file is synced (a page a packing wrote stays until it is
    written back). Session mcpv3b's dropped arms (2026-10-13) dropped nothing."""
    import mmap
    t = time.time()
    if not on and pf is not None and hasattr(mmap, "MADV_DONTNEED"):
        m = pf._map() if pf.mapped else None
        maps = [m[0]] if m is not None else []
        maps += [x._mmap for x in (pf.arr, pf.packed) if getattr(x, "_mmap", None) is not None]
        for mm in maps:
            mm.madvise(mmap.MADV_DONTNEED)
    fd = os.open(path, os.O_RDONLY)
    try:
        if on:
            while os.read(fd, 16 << 20):
                pass
        elif hasattr(os, "posix_fadvise"):         # (Linux; elsewhere the cache stays)
            os.fsync(fd)                            # (a dirty page, a packing's, stays too)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return time.time() - t


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--backend", choices=("isa", "board"), default="board")
    ap.add_argument("--dev", help="the card's device prefix (default: /dev/otpu0 if present, "
                    "else /dev/xdma0)")
    ap.add_argument("--cfg")
    ap.add_argument("--cap", type=int, default=2048)
    ap.add_argument("--arms", default="cached,dropped,dropped,cached")
    ap.add_argument("--turn", default=TURN)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--create", action="store_true", help="a missing pool made, each expert "
                                                          "packed on its first use")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    import chatmoe as CM
    from opentpu.host import chat as C
    from opentpu.host import streamed as ST
    from opentpu.isasim import board_config
    from opentpu.llm import load_spec, model_dir
    path = model_dir(a.model)
    spec = load_spec(path)
    wformat, head, formats = ST.defaults(spec)
    be, cfg = C.make_backend(a.backend, spec, a.cap, a.dev, path.name)
    if a.cfg:
        want = pickle.loads(Path(a.cfg).read_bytes())
        if cfg is not None and cfg != want:
            raise SystemExit(f"the device's configuration is not {a.cfg}'s")
        cfg = want
    t = time.time()
    eng = ST.engine(path, cfg or board_config(), be, a.cap, a.pool, wformat=wformat,
                    head_format=head, formats=formats, snapshots=C.SNAPSHOTS, create=a.create)
    out = {"model": path.name, "pool": a.pool, "disk": disk_of(a.pool),
           "pool_gb": round(os.path.getsize(a.pool) / 1e9, 2), "load_s": round(time.time() - t, 1),
           "experts": ST.describe(eng, eng.pool_path), "arms": []}
    print(f"{path.name}: pool {a.pool} ({out['pool_gb']} GB) on {out['disk']}; load "
          f"{out['load_s']} s; {out['experts']}", flush=True)
    tok = CM._Pinned(AutoTokenizer.from_pretrained(path))
    sp = C.sampling(spec, argparse.Namespace(), path)
    pick = C.sampler(0, sp["top_k"], sp["top_p"], 0, sp["repetition_penalty"], None)
    khz = (getattr(eng.backend, "info", None) or {}).get("core_khz")
    chat = C.Chat(eng, tok, False, pick, a.max_new, clock_mhz=khz / 1e3 if khz else 0.0)
    chat.ask(a.turn)                            # the warm-up: programs, slots
    pf = eng.server.pool_file
    if pf.warm_t is not None:                   # (the open's read of every packed expert)
        pf.warm_t.join()
    for arm in filter(None, a.arms.split(",")):    # (--arms "": the open and the warm-up only)
        if arm not in ("cached", "dropped"):
            raise SystemExit(f"arm {arm!r}: cached or dropped")
        s = cache(a.pool, arm == "cached", pf)
        res = pf.resident(pf.ids)
        total = len(pf.ids) * pf.slot
        # an arm counts when the cache step did what it says: (nearly) all of the pool's
        # experts in the page cache, or (nearly) none
        valid = res is None or (res >= 0.95 * total if arm == "cached" else res <= 0.02 * total)
        if not valid:
            print(f"  {arm}: INVALID, {res / 1e9:.2f} GB of the pool's {total / 1e9:.2f} GB in "
                  f"the page cache", flush=True)
        chat.reset()
        pf.io = {}
        reply, tr = chat.ask(a.turn)
        r = lambda x, k=3: None if x is None else round(x, k)      # noqa: E731
        pf_io, pf.io = pf.io, None
        io = {k: {"reads": v[0], "s": r(v[1]), "ms_a_read": r(1e3 * v[1] / max(v[0], 1), 2),
                  "mb": r(v[2] / 1e6, 1)} for k, v in pf_io.items()}
        x = dict(arm=arm, valid=valid, cache_s=r(s, 1),
                 resident_gb=None if res is None else r(res / 1e9, 2),
                 ms_a_miss=r(1e3 * sum(v[1] for v in pf_io.values()) / max(tr.expert_misses, 1), 2),
                 prefill_tok_s=r(tr.prefill_tok_s, 2),
                 prefill_dev_tok_s=r(tr.prefill_dev_tok_s, 2), decode_tok_s=r(tr.decode_tok_s, 2),
                 decode_dev_tok_s=r(tr.decode_dev_tok_s, 2), misses=tr.expert_misses,
                 hit_rate=r(tr.expert_hit_rate, 4), gen_tokens=tr.gen_tokens, reads=io,
                 reply=reply)
        out["arms"].append(x)
        print(f"  {arm:8s} (page cache set in {x['cache_s']} s, the pool's {x['resident_gb']} GB "
              f"in it{'' if valid else ': INVALID'}): prefill {x['prefill_tok_s']} "
              f"tok/s (device {x['prefill_dev_tok_s']}), decode {x['decode_tok_s']} tok/s "
              f"(device {x['decode_dev_tok_s']}), {x['misses']} misses at {x['ms_a_miss']} ms "
              f"of reads each; reads "
              + ", ".join(f"{k} {v['reads']} at {v['ms_a_read']} ms" for k, v in io.items()),
              flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
