"""Pack a MoE model's expert pool file in the split format (docs/offload.md 10.1, 10.4, 11.4) from
the whole checkpoint: in parallel workers on its host, or streamed into a pool on another host.

    python3 tools/offload/pack_pool.py MODEL POOL init           # the sparse file, .packed, .format
    python3 tools/offload/pack_pool.py MODEL POOL pack I N       # worker I of N: its Nth of the
                                                                  # experts, those not packed yet
    python3 tools/offload/pack_pool.py MODEL - send IDS > stream # the experts IDS names (global ids
                                                                  # g = layer * E + e, one a line)
    python3 tools/offload/pack_pool.py MODEL POOL recv < stream  # on the pool's host: write them
    python3 tools/offload/pack_pool.py MODEL POOL stamp          # a key for a keyless pool
                                                                  # (once sums shows it current)
    python3 tools/offload/pack_pool.py MODEL POOL info           # its format, layout, packed experts
    python3 tools/offload/pack_pool.py MODEL POOL sums IDS       # sha256 of the pool's records IDS
    python3 tools/offload/pack_pool.py MODEL - sums IDS          # ... of the experts packed now
    python3 tools/offload/pack_pool.py MODEL POOL extend SRC     # an MTP image's pool: SRC's
                                                                  # experts, the MTP layer's after

The card's host keeps a checkpoint without the experts (strip_experts.py), so its pool must be
whole before a run warms more slots than the experts it packed itself: session 5 filled the 35B's
(10240 experts) with four `send | ssh HOST ... recv` pipelines from the build host in 52 min, and the
26B's (3840) was packed on the build host by four workers, then copied. An expert is marked packed only
after its bytes. The bytes are the image's (Image.expert, the card's slot format, split as
offload.to_split): the pool MO.serve packs on demand. The expert format follows the spec's
formats (OTPU_FORMATS, e.g. experts=fp4) and --wformat; recv needs MODEL only for the slot size
and the key.

The pool's key, `POOL.key` (moe.pool_key: the MXU's numerics, the quantizer's code, the format,
D, the layout and the checkpoint's config.json), is written by init and checked by pack, recv
(against the key send puts first in its stream) and every run that opens the pool
(moe.open_pool): a pool packed for the MXU before MX (key 1), by another quantizer, in another
format or from another checkpoint is refused. A pool without a key is refused too: one packed
before keys holds the former MXU's scale words (docs/mx_mxu.md). `stamp` writes the key of
MODEL and --wformat on a pool without one (the slot size must be theirs; stamp only a pool
`sums` shows to be this tree's). `info` prints the key, or for a pool without one its format as
its slot size says (int8, or 4-bit: fp4 and int4 have the same size).

`sums` tells whether a pool is current before it is stamped: the sha256 of the records IDS names
(a file of global ids, one a line, or a number N: N ids spread over the pool) as the pool holds
them, or as this tree packs them from the whole checkpoint (`-`). The two lists are equal when
the pool's experts are this tree's (the split format's bytes, as pack writes them).

--mtp: the pool of an MTP image (Spec.mtp, Qwen3.5's drafter, docs/offload.md 15.6): its MTP
layer's MoE is one more layer of the pool's, its experts (mtp.layers.0.mlp.experts.E.*) after
the model's (g = layers * E + e); --mtp-from DIR takes the mtp.* tensors from another
checkpoint (MiniCPM-V 4.7 has none: its base Qwen3.5-35B-A3B's), named in the key. `extend SRC`
makes one from the model's own pool SRC (its key checked) without packing it again: SRC's
experts and marks (reflinked where the file system can: btrfs, XFS; --copy), the MTP layer's
experts not packed (the run packs each on its first use, or `pack ... --mtp` workers do, or
`recv --mtp` from a host with the whole checkpoint). SRC is only read.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time

import numpy as np


def image(model: str, wformat: str, mtp: bool = False):
    from opentpu.isasim import board_config
    from opentpu.llm import load_spec
    spec = load_spec(model)
    if mtp:                                 # (the MTP layer's MoE: one more layer; 2 rows)
        spec = dataclasses.replace(spec, mtp=True)
    cfg = dataclasses.replace(board_config(), DRAM_BYTES=1 << 40)   # the layout only
    return spec.image(cfg, 1024, 1, 2 if mtp else 1, wformat, "int8", lookup=True,
                      experts=spec.moe.k)


def weights(a):
    """The checkpoint's tensors as the pool's experts read them (an MTP pool's: with mtp.*)."""
    from opentpu.llm.qwen3 import LazyWeights
    return LazyWeights(a.model, mtp=True, mtp_from=a.mtp_from) if a.mtp else \
        LazyWeights(a.model)


def extend(a, n: int, slot: int, key: dict) -> None:
    """POOL: an MTP image's pool from the model's own pool SRC (a.args[0]), checked by its key:
    SRC's experts and marks (the file reflinked where the file system can, so no byte of it is
    written twice on the disk: a.copy), the MTP layer's experts after them not packed. Entries of
    SRC past its experts (MiniCPM-V's staged vision pieces, at layers * E + i) are not copied:
    the key written has no "vision", and a staged run adds them again after the MTP layer."""
    from opentpu.host.offload import SPLIT
    from opentpu.llm import moe as MO
    src = a.args[0]
    base = image(a.model, a.wformat)
    L0 = base.offload
    n0 = L0.layers * L0.E
    if os.path.exists(a.pool):
        sys.exit(f"{a.pool} exists")
    if L0.slot_bytes != slot or n0 >= n:
        sys.exit(f"the MTP image's pool ({n} experts of {slot} bytes) does not extend the "
                 f"model's ({n0} of {L0.slot_bytes})")
    try:
        MO.check_pool_key(src, MO.pool_key(base, a.model))
    except ValueError as e:
        sys.exit(str(e))
    size = os.path.getsize(src)
    if size < n0 * slot or size % slot:
        sys.exit(f"{src}: {size} bytes, not {n0} experts of {slot} (and whole entries after)")
    fmt = src + ".format"
    if not (os.path.exists(fmt) and open(fmt).read().strip() == SPLIT):
        sys.exit(f"{src} is not in the split format ({SPLIT}): repack it")
    pk = src + ".packed"
    done = np.fromfile(pk, np.uint8) if os.path.exists(pk) else np.ones(n0, np.uint8)
    if len(done) != size // slot:
        sys.exit(f"{pk}: {len(done)} marks, not {size // slot}")
    done = done[:n0]
    t0 = time.time()
    how = "reflinked"                       # (btrfs, XFS: SRC's extents shared, copy on write)
    if a.copy == "full" or subprocess.run(["cp", "--reflink=always", src, a.pool],
                                          capture_output=True).returncode:
        how = "copied"
        if a.copy == "reflink":
            sys.exit(f"--copy reflink: {a.pool}'s file system does not reflink {src}")
        with open(src, "rb") as f, open(a.pool, "wb") as g:     # SRC's experts only
            left = n0 * slot
            while left:
                b = f.read(min(left, 64 << 20))
                if not b:
                    sys.exit(f"{src}: short read")
                g.write(b)
                left -= len(b)
    with open(a.pool, "r+b") as g:          # (SRC's entries after its experts cut, then the
        g.truncate(n0 * slot)               # MTP layer's experts: sparse until packed)
        g.truncate(n * slot)
    np.concatenate([done, np.zeros(n - n0, np.uint8)]).tofile(a.pool + ".packed")
    open(a.pool + ".format", "w").write(SPLIT + "\n")
    open(a.pool + ".key", "w").write(json.dumps(key, indent=1) + "\n")
    print(f"extend {a.pool}: {src}'s {int(done.astype(bool).sum())} of {n0} experts packed, "
          f"{how} ({time.time() - t0:.0f} s), {n - n0} more not packed (the MTP layer's); "
          f"{n} x {slot} = {n * slot / 1e9:.2f} GB, {key['format']}")


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


def sums(a, img, n: int, slot: int) -> None:
    """Each record's "g sha256" line: the pool's (POOL), or packed from MODEL now (POOL -)."""
    import hashlib
    from opentpu.host.offload import SPLIT, to_split
    arg = a.args[0]
    ids = [i * n // int(arg) for i in range(int(arg))] if arg.isdigit() else \
        [int(x) for x in open(arg).read().split()]
    if a.pool == "-":
        W = weights(a)
        get = lambda g: to_split(img.expert(W, g))                     # noqa: E731
    else:
        if os.path.getsize(a.pool) != n * slot:
            sys.exit(f"{a.pool}: {os.path.getsize(a.pool)} bytes, not {n} experts of {slot}")
        split = os.path.exists(a.pool + ".format") and \
            open(a.pool + ".format").read().strip() == SPLIT
        arr = np.memmap(a.pool, np.uint8, "r", shape=(n, slot))
        get = lambda g: arr[g] if split else to_split(np.array(arr[g]))  # noqa: E731
    for g in ids:
        print(g, hashlib.sha256(np.ascontiguousarray(get(g)).tobytes()).hexdigest(), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model")
    ap.add_argument("pool")
    ap.add_argument("mode", choices=("init", "pack", "send", "recv", "stamp", "info", "sums",
                                     "extend"))
    ap.add_argument("args", nargs="*")
    ap.add_argument("--wformat", default="fp4", help="the image's weight format (default fp4)")
    ap.add_argument("--mtp", action="store_true",
                    help="an MTP image's pool: its MTP layer's experts after the model's "
                         "(extend: always)")
    ap.add_argument("--mtp-from", metavar="DIR",
                    help="the checkpoint of the MTP layer (mtp.*) when not MODEL's (--mtp)")
    ap.add_argument("--copy", choices=("auto", "reflink", "full"), default="auto",
                    help="extend: SRC's experts reflinked where the file system can (btrfs, "
                         "XFS: no byte duplicated on the disk), else copied (auto); only "
                         "reflinked; only copied")
    a = ap.parse_args()
    a.mtp = a.mtp or a.mode == "extend" or a.mtp_from is not None
    from opentpu.host.offload import SPLIT, to_split
    from opentpu.llm import moe as MO

    img = image(a.model, a.wformat, a.mtp)
    L = img.offload
    n, slot = L.layers * L.E, L.slot_bytes
    # the experts' key: a file of n experts (staged vision's entries past them, pool_key's
    # "vision", are appended by the first staged run's open_pool, which then adds "vision";
    # with it here, pool_entries would count them, and open_pool truncate the pool as new)
    key = {k: v for k, v in MO.pool_key(img, a.model, a.mtp_from).items() if k != "vision"}
    if a.mode == "extend":
        extend(a, n, slot, key)
        return
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
    if a.mode == "sums":
        sums(a, img, n, slot)
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
    W, t0 = weights(a), time.time()
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
