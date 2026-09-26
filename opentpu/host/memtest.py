"""DDR3 tests for otpu-diag, on one channel through raw channel addresses (transport.mem_write /
mem_read) or through the accelerator's interleave (Board).

Byte lanes: each channel is 64 data bits (plus an ECC byte the host never sees) behind a MIG
with a 512-bit AXI port; one 64-byte AXI beat is one BL8 burst, so byte b of a channel offset
travels on DQ byte lane b % 8 (DQ[8L+7:8L]). Error counts come per byte lane and per DQ bit
(64), from which otpu-diag names the failing lane.

Every test returns a dict: ok, msg, and the counts ("lanes": [8], "bits": [64], plus
"addr_bits" for the address tests).
"""
from __future__ import annotations

import sys
import time

import numpy as np

LANES = 8
BEAT = 64


def errors(got: np.ndarray, want: np.ndarray) -> dict:
    """Bytes at channel offsets that are multiples of 8: counts of wrong bytes per lane and of
    flipped bits per DQ bit."""
    x = np.bitwise_xor(got, want).reshape(-1, LANES)
    lanes = np.count_nonzero(x, axis=0)
    bits = np.unpackbits(x[:, :, None], axis=2, bitorder="little").sum(axis=0).reshape(-1)
    return {"lanes": lanes.astype(int).tolist(), "bits": bits.astype(int).tolist(),
            "bad_bytes": int(np.count_nonzero(x))}


def _merge(total: dict, e: dict) -> dict:
    if not total:
        return dict(e)
    return {"lanes": [a + b for a, b in zip(total["lanes"], e["lanes"])],
            "bits": [a + b for a, b in zip(total["bits"], e["bits"])],
            "bad_bytes": total["bad_bytes"] + e["bad_bytes"]}


def lane_text(e: dict) -> str:
    bad = [f"lane {i}: {n}" for i, n in enumerate(e["lanes"]) if n]
    return f"{e['bad_bytes']} bytes wrong ({', '.join(bad)})" if bad else "no errors"


def _result(e: dict, what: str) -> dict:
    ok = e["bad_bytes"] == 0
    return {**e, "ok": ok, "msg": what if ok else f"{what}: {lane_text(e)}"}


def walking_data(t, ch: int, off: int) -> dict:
    """Walking 1 and walking 0 over the 512 bits of a beat: beat k of the pattern has only bit
    k set (then only bit k clear); 2 x 512 beats written, read back."""
    one = np.zeros((512, BEAT), np.uint8)
    one[np.arange(512), np.arange(512) // 8] = 1 << (np.arange(512) % 8)
    want = np.concatenate([one, ~one]).reshape(-1)
    t.mem_write(ch, off, want)
    return _result(errors(t.mem_read(ch, off, len(want)), want),
                   "walking 1 and walking 0 over the 512-bit beat")


def address_bits(t, ch: int, ch_bytes: int) -> dict:
    """A unique 64-byte tag at offset 0 and at every power of two from 64 up; each tag names its
    own offset, so a tag found at the wrong place names the address bit it aliases with."""
    offs = [0] + [1 << k for k in range(6, (ch_bytes - 1).bit_length())]

    def tag(o):
        return np.frombuffer(np.uint64(0xA5A5_0000_0000_0000 | (ch << 40) | o).tobytes() * 8,
                             np.uint8)
    for o in offs:
        t.mem_write(ch, o, tag(o))
    bad = {}
    for o in offs:
        got = t.mem_read(ch, o, BEAT)
        if np.array_equal(got, tag(o)):
            continue
        v = int(np.frombuffer(got[:8].tobytes(), np.uint64)[0])
        src = v & 0xFF_FFFF_FFFF
        bit = o.bit_length() - 1
        alias = (f"aliases with {'offset 0' if src == 0 else f'bit {src.bit_length() - 1}'}"
                 if v >> 48 == 0xA5A5 and src in offs else f"reads {v:#018x}")
        bad[bit] = alias
    msg = f"{len(offs) - 1} address bits" if not bad else "; ".join(
        f"bit {b} {a}" for b, a in sorted(bad.items()))
    return {"ok": not bad, "msg": msg, "addr_bits": bad}


def random_blocks(t, ch: int, ch_bytes: int, n: int = 16, size: int = 1 << 20,
                  seed: int = 5) -> dict:
    """n blocks of random bytes spread evenly over the channel (the last one at its top)."""
    size = min(size, ch_bytes // (2 * n))
    rng = np.random.default_rng(seed + ch)
    offs = [k * (ch_bytes // n) // BEAT * BEAT for k in range(n - 1)] + [ch_bytes - size]
    data = [rng.integers(0, 256, size, dtype=np.uint8) for _ in offs]
    for o, d in zip(offs, data):
        t.mem_write(ch, o, d)
    total = {}
    for o, d in zip(offs, data):
        total = _merge(total, errors(t.mem_read(ch, o, size), d))
    return _result(total, f"{n} blocks of {size >> 10} KiB")


def bandwidth(t, ch: int, nbytes: int) -> dict:
    buf = np.random.default_rng(0).integers(0, 256, nbytes, dtype=np.uint8)
    t0 = time.perf_counter()
    t.mem_write(ch, 0, buf)
    w = nbytes / (time.perf_counter() - t0) / 1e9
    t0 = time.perf_counter()
    t.mem_read(ch, 0, nbytes)
    r = nbytes / (time.perf_counter() - t0) / 1e9
    ok = w > 0.25 and r > 0.25
    return {"ok": ok, "msg": f"host->card {w:.2f} GB/s, card->host {r:.2f} GB/s "
                             f"({nbytes >> 20} MiB)",
            "write_gbs": w, "read_gbs": r}


# ------------------------------------------------------------------------------ march
def march(t, ch: int, ch_bytes: int, chunk: int = 64 << 20, progress=None) -> dict:
    """March C- over the whole channel in DMA-sized chunks, with address-in-address data: the
    background A holds each 8-byte word's own offset (tagged with the channel), B its
    complement. Elements (each over every chunk, up or down):
    up w(A); up r(A) w(B); up r(B) w(A); down r(A) w(B); down r(B) w(A); up r(A).
    Every wrong 8-byte word counts per byte lane and DQ bit, and per address bit of its
    offset: an address bit that is 1 (or 0) in every wrong word points at a row / column / bank
    line, or at a region of the chip."""
    chunk = min(chunk, ch_bytes)
    starts = list(range(0, ch_bytes, chunk))
    tagv = np.uint64(0x5A00_0000_0000_0000 | (ch << 40))

    def bg(o, inv):
        w = tagv | (np.uint64(o) + np.arange(chunk // 8, dtype=np.uint64) * np.uint64(8))
        return (~w if inv else w).view(np.uint8)
    elems = [("up", None, False), ("up", False, True), ("up", True, False),
             ("down", False, True), ("down", True, False), ("up", False, None)]
    nbits = max(1, (ch_bytes - 1).bit_length())
    total, ones, nbad = {}, np.zeros(nbits, np.int64), 0
    steps, done, t0 = len(elems) * len(starts), 0, time.time()
    for ei, (direction, rd, wr) in enumerate(elems):
        for o in (starts if direction == "up" else starts[::-1]):
            if rd is not None:
                want = bg(o, rd)
                got = t.mem_read(ch, o, chunk)
                e = errors(got, want)
                if e["bad_bytes"]:
                    total = _merge(total, e)
                    word = o + 8 * np.nonzero(got.view(np.uint64) != want.view(np.uint64))[0]
                    nbad += len(word)
                    ones += [np.count_nonzero(word >> b & 1) for b in range(nbits)]
            if wr is not None:
                t.mem_write(ch, o, bg(o, wr))
            done += 1
            if progress and (100 * done // steps) // 5 != (100 * (done - 1) // steps) // 5:
                progress(f"march C- channel {ch}: element {ei + 1}/{len(elems)}, "
                         f"{100 * done // steps}%, {time.time() - t0:.0f}s")
    total = total or {"lanes": [0] * LANES, "bits": [0] * 64, "bad_bytes": 0}
    r = _result(total, f"march C-, {ch_bytes >> 20} MiB, {len(starts)} chunks")
    r["bad_words"] = nbad
    r["addr_bits"] = {} if nbad < 16 else {
        b: f"{'1' if ones[b] else '0'} in every wrong word"
        for b in range(3, nbits) if ones[b] in (0, nbad)}
    if r["addr_bits"]:
        r["msg"] += "; " + ", ".join(f"address bit {b} {v}" for b, v in r["addr_bits"].items())
    return r


def progress_line(msg: str) -> None:
    """One updating line on a terminal, a line per call otherwise."""
    if sys.stdout.isatty():
        print(f"\r    {msg}\033[K", end="", flush=True)
    else:
        print(f"    {msg}", flush=True)
