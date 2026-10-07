"""Pack a MoE model's expert pool file in the split format (docs/offload.md 10.1, 10.4, 11.4) from
the whole checkpoint: in parallel workers on its host, or streamed into a pool on another host.

    python3 tools/offload/pack_pool.py MODEL POOL init           # the sparse file, .packed, .format
    python3 tools/offload/pack_pool.py MODEL POOL pack I N       # worker I of N: its Nth of the
                                                                  # experts, those not packed yet
    python3 tools/offload/pack_pool.py MODEL - send IDS > stream # the experts IDS names (global ids
                                                                  # g = layer * E + e, one a line)
    python3 tools/offload/pack_pool.py MODEL POOL recv < stream  # on the pool's host: write them
    python3 tools/offload/pack_pool.py MODEL POOL stamp          # give a pool packed before keys
                                                                  # its key (once it is known current)
    python3 tools/offload/pack_pool.py MODEL POOL info           # its format, layout, packed experts

The card's host keeps a checkpoint without the experts (strip_experts.py), so its pool must be
whole before a run warms more slots than the experts it packed itself: session 5 filled the 35B's
(10240 experts) with four `send | ssh HOST ... recv` pipelines from omarchy in 52 min, and the
26B's (3840) was packed on omarchy by four workers, then copied. An expert is marked packed only
after its bytes. The bytes are the image's (Image.expert, the card's slot format, split as
offload.to_split): the pool MO.serve packs on demand. The expert format follows the spec's
formats (OTPU_FORMATS, e.g. experts=fp4) and --wformat; recv needs MODEL only for the slot size
and the key.

The pool's key, `POOL.key` (moe.pool_key: the quantizer's code, the format, D, the layout and
the checkpoint's config.json), is written by init and checked by pack, recv (against the key
send puts first in its stream) and every run that opens the pool (moe.open_pool): a pool packed
by another quantizer, in another format or from another checkpoint is refused. A pool packed
before keys has none: it is taken with a warning until `stamp` writes the key of MODEL and
--wformat (the slot size must be theirs; stamp only a pool known to be current). `info` prints
the key, or for a pool without one its format as its slot size says (int8, or 4-bit: fp4 and
int4 have the same size).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np


def image(model: str, wformat: str):
    from opentpu.isasim import board_config
    from opentpu.llm import load_spec
    spec = load_spec(model)
    cfg = dataclasses.replace(board_config(), DRAM_BYTES=1 << 40)   # the layout only
    return spec.image(cfg, 1024, 1, 1, wformat, "int8", lookup=True, experts=spec.moe.k)


def info(a, key: dict) -> None:
    """The pool's key, or a pool's format as its slot size says when it has no key."""
    kf, pk = a.pool + ".key", a.pool + ".packed"
    n = os.path.getsize(pk) if os.path.exists(pk) else 0
    packed = int(np.fromfile(pk, np.uint8).astype(bool).sum()) if n else 0
    size = os.path.getsize(a.pool)
    print(f"{a.pool}: {size} bytes, {packed} of {n} experts packed")
    if os.path.exists(kf):
        have = json.loads(open(kf).read())
        lay = have["layout"]
        print(f"key: {have['format']}, D {have['D']}, {lay['layers']} layers x {lay['E']} "
              f"experts of {lay['slot_bytes']} bytes, from {have.get('model')}")
        from opentpu.llm import moe as MO
        diff = MO.key_diff(have, key)
        print(f"against {a.model} --wformat {a.wformat}: " +
              (f"{', '.join(diff)} differ" if diff else "the same key"))
        return
    from opentpu.llm import load_spec
    from opentpu.llm.moe import ExpertFormat
    spec = load_spec(a.model)
    slot = size // n if n else 0
    sizes = {f: ExpertFormat(spec.hidden, spec.moe.ffn, key["D"], f).nbytes
             for f in ("int8", "fp4")}
    what = ("int8" if slot == sizes["int8"] else
            "4-bit (fp4 or int4: the same size)" if slot == sizes["fp4"] else
            f"neither ({sizes['int8']} bytes int8, {sizes['fp4']} 4-bit)")
    print(f"no key (packed before keys): slots of {slot} bytes: {what}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model")
    ap.add_argument("pool")
    ap.add_argument("mode", choices=("init", "pack", "send", "recv", "stamp", "info"))
    ap.add_argument("args", nargs="*")
    ap.add_argument("--wformat", default="fp4", help="the image's weight format (default fp4)")
    a = ap.parse_args()
    from opentpu.host.offload import SPLIT, to_split
    from opentpu.llm import moe as MO

    img = image(a.model, a.wformat)
    L = img.offload
    n, slot = L.layers * L.E, L.slot_bytes
    key = MO.pool_key(img, a.model)
    if a.mode == "info":
        info(a, key)
        return
    if a.mode == "init":
        if os.path.exists(a.pool):
            sys.exit(f"{a.pool} exists")
        with open(a.pool, "wb") as f:
            f.truncate(n * slot)
        open(a.pool + ".packed", "wb").write(bytes(n))
        open(a.pool + ".format", "w").write(SPLIT + "\n")
        open(a.pool + ".key", "w").write(json.dumps(key, indent=1) + "\n")
        print(f"init {a.pool}: {n} experts x {slot} bytes = {n * slot / 1e9:.2f} GB, "
              f"{key['format']}")
        return
    if a.mode == "stamp":
        if os.path.getsize(a.pool) != n * slot:
            sys.exit(f"{a.pool}: {os.path.getsize(a.pool)} bytes, not {n} experts of {slot} "
                     f"({key['format']}): not this model's pool in this format")
        if os.path.exists(a.pool + ".key"):
            have = json.loads(open(a.pool + ".key").read())
            diff = MO.key_diff(have, key)
            sys.exit(f"{a.pool} has a key already: " +
                     (f"{', '.join(diff)} differ from this one; repack it" if diff else "this one"))
        open(a.pool + ".key", "w").write(json.dumps(key, indent=1) + "\n")
        print(f"stamped {a.pool}: {key['format']}, D {key['D']}, {n} experts x {slot} bytes, "
              f"{key['model']}")
        return
    if a.mode in ("pack", "recv"):
        try:
            MO.check_pool_key(a.pool, key)
        except ValueError as e:
            sys.exit(str(e))
    if a.mode == "recv":
        fd, pk = os.open(a.pool, os.O_WRONLY), os.open(a.pool + ".packed", os.O_WRONLY)
        inp, k = os.fdopen(sys.stdin.fileno(), "rb", buffering=1 << 20, closefd=False), 0
        h = inp.read(8)
        if len(h) == 8 and int.from_bytes(h, "little", signed=True) == -1:    # send's key
            sent = json.loads(inp.read(int.from_bytes(inp.read(8), "little")))
            diff = MO.key_diff(sent, key)
            if diff:
                sys.exit(f"recv: the stream's experts were packed for another image "
                         f"({', '.join(diff)} differ: {sent['format']} from {sent.get('model')}, "
                         f"this pool {key['format']}); nothing written")
            h = inp.read(8)
        else:
            print("recv: a stream without a key (an older send): unchecked", file=sys.stderr)
        while h and len(h) == 8:
            g, b = int.from_bytes(h, "little", signed=True), inp.read(slot)
            if b is None or len(b) != slot:
                sys.exit(f"recv: a short record for expert {g} after {k}")
            os.pwrite(fd, b, g * slot)
            if k % 500 == 499:
                os.fdatasync(fd)
            os.pwrite(pk, b"\x01", g)
            k += 1
            h = inp.read(8)
        os.fsync(fd)
        os.fsync(pk)
        print(f"recv: {k} experts", file=sys.stderr)
        return

    from opentpu.llm.qwen3 import LazyWeights
    if a.mode == "send":
        ids = [int(x) for x in open(a.args[0]).read().split()]
        out, log = os.fdopen(sys.stdout.fileno(), "wb", buffering=0, closefd=False), sys.stderr
        js = json.dumps(key).encode()                   # the key first: g = -1, its length
        out.write(np.int64(-1).tobytes() + np.int64(len(js)).tobytes() + js)
    else:
        i, k = int(a.args[0]), int(a.args[1])
        packed = np.fromfile(a.pool + ".packed", np.uint8)
        assert len(packed) == n and os.path.getsize(a.pool) == n * slot, "not this model's pool"
        ids = [g for g in range(i * n // k, (i + 1) * n // k) if not packed[g]]   # a rerun resumes
        fd, pk = os.open(a.pool, os.O_WRONLY), os.open(a.pool + ".packed", os.O_WRONLY)
        log = sys.stdout
    W, t0 = LazyWeights(a.model), time.time()
    for c, g in enumerate(ids):
        b = to_split(img.expert(W, g))
        assert len(b) == slot
        if a.mode == "send":
            out.write(np.int64(g).tobytes())
            out.write(memoryview(b))
        else:
            os.pwrite(fd, memoryview(b), g * slot)
            os.pwrite(pk, b"\x01", g)                  # after its bytes
        if c % 100 == 0:
            print(f"{a.mode}: {c}/{len(ids)} g {g} {time.time() - t0:.0f} s", file=log, flush=True)
    if a.mode == "pack":
        os.fsync(fd)
        os.fsync(pk)
    print(f"{a.mode}: {len(ids)} experts in {time.time() - t0:.0f} s", file=log, flush=True)


if __name__ == "__main__":
    main()
