"""Copy an expert pool file in the slot format into the split format (docs/offload.md 10.1):
each 4 KiB of an expert as its two channel runs under CHASH, which the card's expert server
reads straight into the runs it DMAs (opentpu.host.offload.split_order). Only the packed
experts are copied; the new file is sparse elsewhere, with its own .packed and .format.

    python3 tools/offload/pool_split.py pool-q35-fp4.bin pool-q35-fp4.split.bin

(MO.serve writes a new pool file in the split format itself; this is for files packed before.)
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from opentpu.host.offload import SPLIT, preadv, to_split


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    fmt = Path(str(src) + ".format")
    if fmt.exists() and fmt.read_text().strip() == SPLIT:
        sys.exit(f"{src} is in the split format already")
    if dst.exists():
        sys.exit(f"{dst} exists")
    packed = np.fromfile(str(src) + ".packed", np.uint8)
    size = src.stat().st_size
    if size % len(packed):
        sys.exit(f"{src}: {size} bytes is not {len(packed)} slots")
    slot = size // len(packed)
    ids = np.nonzero(packed)[0]
    buf = np.empty(slot, np.uint8)
    fi = os.open(src, os.O_RDONLY)
    fo = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    t0 = time.time()
    try:
        os.ftruncate(fo, size)
        for k, g in enumerate(ids):
            preadv(fi, [memoryview(buf)], int(g) * slot)
            out = to_split(buf)
            if os.pwrite(fo, out, int(g) * slot) != slot:
                raise OSError(f"short write at expert {g}")
            if k % 500 == 0:
                print(f"{k}/{len(ids)} experts, {time.time() - t0:.0f} s", flush=True)
        os.fsync(fo)
    finally:
        os.close(fi)
        os.close(fo)
    shutil.copyfile(str(src) + ".packed", str(dst) + ".packed")
    Path(str(dst) + ".format").write_text(SPLIT + "\n")
    print(f"{dst}: {len(ids)} experts of {slot} bytes in the split format, "
          f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
