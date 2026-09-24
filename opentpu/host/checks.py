"""Bring-up checks shared by tools/board_selftest.py and tests/test_board.py."""
from __future__ import annotations

import dataclasses
import time

import numpy as np

from opentpu import isa as I
from opentpu.isasim import Machine

DATA, W8, SC, OUT = 0, 0x10000, 0x20000, 0x30000
PROG_AT = 0x3C0000
SPAN = 0x40000                  # bytes the demo program's results can touch (below PROG_AT)


def demo_image(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros(PROG_AT, np.uint8)
    img[DATA:DATA + 4 * 4096] = rng.uniform(-2, 2, 4096).astype(np.float32).view(np.uint8)
    img[W8:W8 + 16 * 256] = rng.integers(-127, 128, 16 * 256).astype(np.int8).view(np.uint8)
    img[SC:SC + 4 * 64] = rng.uniform(0.01, 0.02, 64).astype(np.float32).view(np.uint8)
    return img


def demo_program() -> list:
    """Every unit: DMA (aligned and unaligned), VPU simple and composite lanes, quantizer,
    MXU, QST byte writes."""
    return [
        I.ld(DATA, 0, 1024),
        I.ld(DATA + 4 * 1024 + 4, 1024, 777),                         # unaligned DRAM source
        I.vop(I.V_MUL, 2048, 0, 0, 4, 256, 256, 256, 0, I.B_SCALAR, 0.5),
        I.vop(I.V_EXP2, 3072, 0, 0, 2, 200, 200, 256, 0, I.B_SCALAR, 0.0),   # composite lanes
        I.vop(I.V_ADD, 3600, 1024, 0, 1, 300, 300, 300, 256, I.B_FULL),
        I.qact(0, 2, 0, 2, 256),
        I.mm(W8, SC, 4096, 16, 2, 256, 16, 2, 0, 8),
        I.qst(2048, OUT + 0x8000, OUT + 0xC000, 2, 2, 256, 256, 1),
        I.st(OUT, 2048, 1024),
        I.st(OUT + 4 * 1024 + 8, 3072, 400),                           # unaligned DRAM target
        I.st(OUT + 0x2000, 3600, 300),
        I.st(OUT + 0x3000, 4096, 32 + 2),
        I.halt(),
    ]


def masked_program() -> list:
    """Many partial DRAM writes from the accelerator: QST byte writes (dense and strided) and
    short word-masked stores at every alignment. The board's DDR3 has no data-mask pins, so
    each of these is a read-modify-write inside the memory controller."""
    prog = [I.ld(DATA, 0, 4096)]
    for k in range(24):
        n = 1 + (k * 7) % 23                                   # 1..23 words
        prog.append(I.st(OUT + 0x10000 + 4 * (37 * k + k % 5), 64 * k, n))
    prog.append(I.qst(0, OUT + 0x14000 + 3, OUT + 0x16000 + 4, 2, 2, 256, 256, 1))   # dense
    prog.append(I.qst(512, OUT + 0x18000 + 1, OUT + 0x1A000, 3, 1, 128, 1024, 3))    # strided
    prog.append(I.halt())
    return prog


def run_demo(board, cfg, prog: list | None = None) -> tuple[bool, str, dict]:
    """Run a program (default: the demo) on the board and on the ISA simulator; compare DRAM."""
    img = demo_image()
    prog = prog or demo_program()
    ref = np.zeros(min(cfg.DRAM_BYTES, 1 << 23), np.uint8)
    ref[:len(img)] = img
    m = Machine(dataclasses.replace(cfg, DRAM_BYTES=len(ref)), [prog], [ref]).run()
    board.write(0, img)
    board.load_program(PROG_AT, np.asarray(I.assemble(prog), np.uint32))
    st = board.run(timeout=10.0)
    got = board.read(0, PROG_AT)
    want = m.slices[0].dram[:PROG_AT]
    bad = np.nonzero(got != want)[0]
    if st["instructions"][0] != len(prog):
        return False, f"retired {st['instructions'][0]} of {len(prog)} instructions", st
    if len(bad):
        return False, f"{len(bad)} DRAM bytes differ from the ISA simulator, first at " \
                      f"{[hex(int(b)) for b in bad[:6]]}", st
    return True, f"{st['cycles']} cycles", st


def pattern_test(board, regions: list[tuple[int, int]], seed: int = 1) -> tuple[bool, str]:
    """Write random bytes to every region (logical addresses), read them back."""
    rng = np.random.default_rng(seed)
    data = [rng.integers(0, 256, n).astype(np.uint8) for _, n in regions]
    for (a, _), d in zip(regions, data):
        board.write(a, d)
    for (a, n), d in zip(regions, data):
        got = board.read(a, n)
        bad = np.nonzero(got != d)[0]
        if len(bad):
            return False, (f"region {a:#x}+{n:#x}: {len(bad)} bytes wrong, first at "
                           f"{a + int(bad[0]):#x} (logical beat {(a + int(bad[0])) // 64}, "
                           f"channel {((a + int(bad[0])) // 64) % 2})")
    return True, f"{len(regions)} regions, {sum(n for _, n in regions)} bytes"


def channel_patterns(transport, ch: int, ch_bytes: int, seed: int = 2) -> tuple[bool, str]:
    """Random data straight to one channel (raw channel addresses) at the bottom, middle and
    top of the channel."""
    rng = np.random.default_rng(seed + ch)
    n = min(65536, ch_bytes // 8)
    for off in (0, 4096, ch_bytes // 2, ch_bytes - n):
        d = rng.integers(0, 256, n).astype(np.uint8)
        transport.mem_write(ch, off, d)
        if not np.array_equal(transport.mem_read(ch, off, n), d):
            return False, f"channel {ch} offset {off:#x}: read-back mismatch"
    return True, "4 regions"


def partial_writes(transport, ch: int, base: int = 1 << 20, seed: int = 3) -> tuple[bool, str]:
    """Sub-beat host writes (1..63 bytes at odd offsets) into a filled region of one channel:
    the DMA engine sends them with partial byte strobes, which the memory controller turns into
    read-modify-writes (no DDR3 data-mask pins on this board)."""
    rng = np.random.default_rng(seed + ch)
    ref = rng.integers(0, 256, 4096).astype(np.uint8)
    transport.mem_write(ch, base, ref)
    for _ in range(200):
        n = int(rng.integers(1, 64))
        o = int(rng.integers(0, 4096 - n))
        d = rng.integers(0, 256, n).astype(np.uint8)
        transport.mem_write(ch, base + o, d)
        ref[o:o + n] = d
    got = transport.mem_read(ch, base, 4096)
    bad = np.nonzero(got != ref)[0]
    if len(bad):
        return False, (f"channel {ch}: {len(bad)} bytes wrong after partial writes, first at "
                       f"{base + int(bad[0]):#x}")
    return True, "200 partial writes"


def address_lines(transport, ch: int, ch_bytes: int) -> tuple[bool, str]:
    """Walking address bits on one channel (raw channel addresses): a unique 64-byte tag at
    offset 0 and at every power of two; an aliased or stuck address line shows up as a tag
    overwritten by another."""
    offs = [0] + [1 << k for k in range(6, (ch_bytes - 1).bit_length())]
    tags = {o: np.frombuffer(np.uint64(0xA5A5_0000_0000_0000 | (ch << 40) | o).tobytes() * 8,
                             np.uint8) for o in offs}
    for o in offs:
        transport.mem_write(ch, o, tags[o])
    for o in offs:
        got = transport.mem_read(ch, o, 64)
        if not np.array_equal(got, tags[o]):
            v = int(np.frombuffer(got[:8].tobytes(), np.uint64)[0])
            return False, (f"channel {ch} offset {o:#x}: read tag {v:#x} -- address bit "
                           f"{o.bit_length() - 1} aliases or is stuck")
    return True, f"{len(offs)} address bits"


def bandwidth(transport, nbytes: int) -> tuple[float, float]:
    """H2C and C2H GB/s over both channels."""
    buf = np.random.default_rng(0).integers(0, 256, nbytes // 2).astype(np.uint8)
    t = time.time()
    for c in (0, 1):
        transport.mem_write(c, 0, buf)
    w = nbytes / (time.time() - t) / 1e9
    t = time.time()
    for c in (0, 1):
        transport.mem_read(c, 0, nbytes // 2)
    r = nbytes / (time.time() - t) / 1e9
    return w, r
