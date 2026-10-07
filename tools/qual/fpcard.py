"""The fp32 corners on the card against the ISA simulator (tests/test_fp_rtl.py's programs at the
board's configuration): multiplies and adds at the flush boundary, the composite functions on
their scaled ranges and on NaN words (otpu_se_comp on a DSTEP build), the quantizer's recip on
huge and tiny amax (QST, rows and blocks), an MM whose partial sums flush to -0, and ARGMAX ties
inside a chunk of lanes. Each program runs on the card and on the ISA simulator from the same
DRAM image; the DRAM below checks.PROG_AT must be equal, bit for bit.

    python3 tools/qual/fpcard.py [--dev /dev/xdma0]   the card (run it under otpu-lock)
    python3 tools/qual/fpcard.py --rtl                 the same programs on the RTL (Verilator)

Prints one line per program and "FPCARD PASS" (exit 0) or "FPCARD FAIL" (exit 1)."""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from opentpu import fp32 as F, isa as I                          # noqa: E402
from opentpu.host.checks import PROG_AT, ZERO_AT                 # noqa: E402
from opentpu.isasim import Machine, board_config                 # noqa: E402

OUT = 0x100000                  # DRAM: the programs' stored TMEM results


def _image(*parts) -> np.ndarray:
    img = np.zeros(PROG_AT, np.uint8)
    for at, a in parts:
        b = np.ascontiguousarray(a).view(np.uint8).reshape(-1)
        img[at:at + len(b)] = b
    return img


def _corner_words(rng, n):
    """Words from recip's and rsqrt's scaled ranges (exponent fields 0..5 and 246..255, both
    signs), NaNs, and normals."""
    e = rng.choice(list(range(6)) + list(range(246, 255)), n).astype(np.uint32)
    m = rng.integers(0, 1 << 23, n, dtype=np.uint32)
    m[rng.random(n) < 0.1] = 0
    m[rng.random(n) < 0.1] = 0x7FFFFF
    s = rng.integers(0, 2, n, dtype=np.uint32)
    w = (s << 31) | (e << 23) | m
    extra = np.uint32([0x7FC00000, 0xFFC00000, 0x7F800001, 0xFFBFFFFF, 0x7F800000, 0xFF800000,
                       0x7E800000, 0x7E7311C3, 0x7E7311C4, 0x00800000, 0x00FFFFFF, 0x3F800000])
    return F.from_bits(np.concatenate([w, extra])).copy()


def mul_add(cfg):
    """MUL / ADD / SUB with results at the flush boundary: products of a power of two and an
    all-ones mantissa, on 2^-126 - 2^-150 or near it (IEEE rounds the tie up to 2^-126 on the
    subnormal grid), and sums of normals that land below 2^-126."""
    rng = np.random.default_rng(5)
    n = 2048
    e1 = rng.integers(1, 127, n).astype(np.int64)        # e1 + e2 = 127: 2^-126 (1 - 2^-24)
    e2 = np.clip(127 - e1 + rng.integers(-1, 2, n), 1, 254)
    s1, s2 = (rng.integers(0, 2, n).astype(np.int64) << 31 for _ in range(2))
    m2 = np.where(rng.random(n) < 0.7, 0x7FFFFF, rng.integers(0x7FFFF0, 0x800000, n))
    a = F.from_bits((s1 | (e1 << 23)).astype(np.uint32)).copy()
    b = F.from_bits((s2 | (e2 << 23) | m2).astype(np.uint32)).copy()
    c = F.from_bits(np.uint32(0x00800000) + rng.integers(0, 64, n).astype(np.uint32)).copy()
    d = F.from_bits((np.uint32(0x00800000) + rng.integers(0, 64, n).astype(np.uint32))
                    | np.uint32(0x80000000)).copy()
    x = np.concatenate([a, b, c, d])
    cols = 64
    rows = n // cols
    A, B, C, Dd, o = 0, n, 2 * n, 3 * n, 4 * n
    prog = [I.ld(0, 0, 4 * n),
            I.vop(I.V_MUL, o, A, B, rows, cols, cols, cols, cols),
            I.vop(I.V_MUL, o + n, B, A, rows, cols, cols, cols, cols),
            I.vop(I.V_ADD, o + 2 * n, C, Dd, rows, cols, cols, cols, cols),
            I.vop(I.V_SUB, o + 3 * n, C, Dd, rows, cols, cols, cols, cols),
            I.vop(I.V_RSUB, o + 4 * n, Dd, C, rows, cols, cols, cols, cols),
            I.st(OUT, o, 5 * n), I.halt()]
    return prog, _image((0, x))


def composites(cfg):
    """RECIP / RSQRT / EXP2 / LOG2 / ABS / COPY / EXP2SUB / MAX / MIN on the corner words."""
    rng = np.random.default_rng(19)
    cols = 64
    x = _corner_words(rng, 6 * cols - 12)
    n = len(x)
    rows = n // cols
    prog = [I.ld(0, 0, n)]
    out = n
    for func in (I.V_RECIP, I.V_RSQRT, I.V_EXP2, I.V_LOG2, I.V_ABS, I.V_COPY):
        prog.append(I.vop(func, out, 0, 0, rows, cols, cols, cols, 0))
        out += n
    for func in (I.V_EXP2SUB, I.V_MAX, I.V_MIN):
        prog.append(I.vop(func, out, 0, cols, rows - 1, cols, cols, cols, cols))
        out += n
    prog += [I.st(OUT, n, out - n), I.halt()]
    return prog, _image((0, x))


def quantizer(cfg):
    """QST (row groups and blocks): amax in [2^123, 2^126) (recip on amax / 16), at 2^126
    (inv = 0), tiny ones (inv = +inf: zeros give 0 * inf, q 0) and an inf element."""
    D, KB = cfg.D, 2
    rng = np.random.default_rng(3)
    rows = []
    for amax in (2.0 ** 123, 1.5 * 2.0 ** 125, 0x7E7311C4, 2.0 ** 126, 2.0 ** -126, 1e-37,
                 3.7e-37, 3.8e-37, np.inf, 1.0):
        a = F.from_bits(np.uint32(amax)) if isinstance(amax, int) else F.f32(amax)
        r = (rng.standard_normal(KB * D) * 0.3).astype(np.float32) * a
        r[rng.random(KB * D) < 0.4] = 0.0
        r[rng.random(KB * D) < 0.1] = -0.0
        r[0] = a
        rows.append(np.where(np.isfinite(r) | np.isinf(a), r, 0).astype(np.float32))
    x = np.stack(rows)
    R = len(rows)
    prog = [I.ld(0, 0, x.size),
            I.qst(0, OUT, OUT + 0x8000, R, KB, KB * D, KB * D, 1),
            I.qst(0, OUT + 0x10000, OUT + 0x18000, R, KB, KB * D, KB * D, 1, row=True),
            I.halt()]
    return prog, _image((0, x))


def mm_negzero(cfg):
    """KB = 10: each block's term (i2f(127) * ws_k) * (1/127); blocks 0..3 and 4..7 are
    -(2^-126 + 3 ulp) and 2^-126, so every partial of two real terms flushes to -0, and t8, t9
    are -0. The MXU adds no pad terms: the row sum is -0."""
    D, KB = cfg.D, 10
    s = F.mul(F.f32(1.0), F.INV127)
    t_of = lambda ws: F.mul(F.mul(F.f32(127.0), F.f32(ws)), s)   # noqa: E731
    ws_b = F.from_bits(np.uint32(0x00800000))
    while F.bits(t_of(ws_b)) < 0x00800000:
        ws_b = F.from_bits(F.bits(ws_b) + np.uint32(1))
    ws_a = F.from_bits((F.bits(ws_b) + np.uint32(3)) | np.uint32(0x80000000))
    ws = np.array([ws_a] * 4 + [ws_b] * 4 + [F.f32(-0.0)] * 2, np.float32)
    x = np.zeros(KB * D, np.float32)
    x[np.arange(KB) * D] = 1.0
    W = np.zeros(KB * D, np.int8)
    W[np.arange(KB) * D] = 1
    out = 2 * KB * D
    prog = [I.ld(0, 0, KB * D), I.qact(0, 1, 0, KB, KB * D),
            I.mm(0x10000, 0x20000, out, 1, KB, KB * D, 1, 1, 0, 4 * KB),
            I.st(OUT, out, 16), I.halt()]
    return prog, _image((0, x), (0x10000, W), (0x20000, ws))


def argmax_ties(cfg):
    """ARGMAX ties at the row's maximum inside one chunk of lanes go to the first column: every
    pair of tied lanes in the row's first chunk and in a later one, and rows over {0, 1, 2}."""
    lanes = cfg.LANES
    rng = np.random.default_rng(6)
    pairs = [(p, q) for p in range(lanes) for q in range(p + 1, lanes)]
    t = np.full((2 * len(pairs), 2 * lanes), -1.0, np.float32)
    for k, (p, q) in enumerate(pairs):
        t[k, [p, q]] = 5.0
        t[len(pairs) + k, [lanes + p, lanes + q]] = 5.0
    r = rng.integers(0, 3, (64, lanes)).astype(np.float32)
    data = np.concatenate([t.reshape(-1), r.reshape(-1)])
    o = len(data) + 64
    nout = 2 * t.shape[0] + 2 * 64
    prog = [I.ld(0, 0, len(data)),
            I.argmax(o, 0, t.shape[0], t.shape[1], 2, t.shape[1]),
            I.argmax(o + 2 * t.shape[0], t.size, 64, lanes, 2, lanes),
            I.st(OUT, o, nout), I.halt()]
    return prog, _image((0, data))


CHECKS = [mul_add, composites, quantizer, mm_negzero, argmax_ties]


def rtl_check(cfg, prog, img) -> tuple[bool, str]:
    """run_demo's comparison on the RTL instead of the card."""
    from opentpu import rtlsim
    prog = [I.ld(ZERO_AT, 0, cfg.TMEM_WORDS)] + prog
    ref = np.zeros(cfg.DRAM_BYTES, np.uint8)
    ref[:len(img)] = img
    sl = Machine(cfg, [prog], [ref.copy()]).run().slices[0]
    drams, _, _ = rtlsim.run(cfg, [prog], [ref.copy()], uarch=dict(rtlsim.BOARD_UARCH,
                                                                    MXU_IMPL=2))
    bad = np.nonzero(drams[0][:PROG_AT] != sl.dram[:PROG_AT])[0]
    if len(bad):
        return False, f"{len(bad)} DRAM bytes differ, first at {[hex(int(b)) for b in bad[:6]]}"
    return True, "equal"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dev", default="/dev/xdma0")
    ap.add_argument("--rtl", action="store_true", help="the RTL (Verilator), not the card")
    a = ap.parse_args(argv)
    if a.rtl:
        cfg = board_config(PAIR=True, DSTEP=True, STREAM=True, DRAM_BYTES=1 << 22)
        board = None
        print(f"fpcard on the RTL: {cfg}")
    else:
        from opentpu.host.board import CH_BYTES, Board, XdmaTransport, device_config
        board = Board(XdmaTransport(a.dev), check=False)
        # the DRAM's ECC check bits once per configuration first (a read of a beat not written
        # since counts as an uncorrectable error): run on a bitstream just loaded, it read some
        board.scrub()
        info = board.info()
        cfg = device_config(info, DRAM_BYTES=2 * CH_BYTES)
        bid = info.get("build_id")
        print(f"fpcard on {a.dev}: build {bid:08x}" if bid is not None else
              f"fpcard on {a.dev}: no build id", f"LANES={cfg.LANES} DSTEP={cfg.DSTEP}")
    fails = 0
    for chk in CHECKS:
        prog, img = chk(cfg)
        if board is None:
            ok, msg = rtl_check(cfg, prog, img)
        else:
            from opentpu.host.checks import run_demo
            ok, msg, _ = run_demo(board, cfg, prog, img)
        fails += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {chk.__name__}: {msg}", flush=True)
    if board is not None:
        board.close()
    print("FPCARD PASS" if fails == 0 else f"FPCARD FAIL ({fails} of {len(CHECKS)})")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
