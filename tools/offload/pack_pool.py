"""Pack a MoE model's expert pool file in the split format (docs/offload.md 10.1, 10.4, 11.4) from
the whole checkpoint: in parallel workers on its host, or streamed into a pool on another host.

    python3 tools/offload/pack_pool.py MODEL POOL init           # the sparse file, .packed, .format
    python3 tools/offload/pack_pool.py MODEL POOL pack I N       # worker I of N: its Nth of the
                                                                  # experts, those not packed yet
    python3 tools/offload/pack_pool.py MODEL - send IDS > stream # the experts IDS names (global ids
                                                                  # g = layer * E + e, one a line)
    python3 tools/offload/pack_pool.py MODEL POOL recv < stream  # on the pool's host: write them

The card's host keeps a checkpoint without the experts (strip_experts.py), so its pool must be
whole before a run warms more slots than the experts it packed itself: session 5 filled the 35B's
(10240 experts) with four `send | ssh HOST ... recv` pipelines from omarchy in 52 min, and the
26B's (3840) was packed on omarchy by four workers, then copied. An expert is marked packed only
after its bytes. The bytes are the image's (Image.expert, the card's slot format, split as
offload.to_split): the pool MO.serve packs on demand. The expert format follows the spec's
formats (OTPU_FORMATS, e.g. experts=fp4) and --wformat; recv needs MODEL only for the slot size.
"""
from __future__ import annotations

import argparse
import dataclasses
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model")
    ap.add_argument("pool")
    ap.add_argument("mode", choices=("init", "pack", "send", "recv"))
    ap.add_argument("args", nargs="*")
    ap.add_argument("--wformat", default="fp4", help="the image's weight format (default fp4)")
    a = ap.parse_args()
    from opentpu.host.offload import SPLIT, to_split

    img = image(a.model, a.wformat)
    L = img.offload
    n, slot = L.layers * L.E, L.slot_bytes
    if a.mode == "init":
        if os.path.exists(a.pool):
            sys.exit(f"{a.pool} exists")
        with open(a.pool, "wb") as f:
            f.truncate(n * slot)
        open(a.pool + ".packed", "wb").write(bytes(n))
        open(a.pool + ".format", "w").write(SPLIT + "\n")
        print(f"init {a.pool}: {n} experts x {slot} bytes = {n * slot / 1e9:.2f} GB")
        return
    if a.mode == "recv":
        fd, pk = os.open(a.pool, os.O_WRONLY), os.open(a.pool + ".packed", os.O_WRONLY)
        inp, k = os.fdopen(sys.stdin.fileno(), "rb", buffering=1 << 20, closefd=False), 0
        while (h := inp.read(8)) and len(h) == 8:
            g, b = int.from_bytes(h, "little", signed=True), inp.read(slot)
            if b is None or len(b) != slot:
                sys.exit(f"recv: a short record for expert {g} after {k}")
            os.pwrite(fd, b, g * slot)
            if k % 500 == 499:
                os.fdatasync(fd)
            os.pwrite(pk, b"\x01", g)
            k += 1
        os.fsync(fd)
        os.fsync(pk)
        print(f"recv: {k} experts", file=sys.stderr)
        return

    from opentpu.llm.qwen3 import LazyWeights
    if a.mode == "send":
        ids = [int(x) for x in open(a.args[0]).read().split()]
        out, log = os.fdopen(sys.stdout.fileno(), "wb", buffering=0, closefd=False), sys.stderr
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
