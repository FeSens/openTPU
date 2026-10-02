"""Card qualification of the DRAM's read / write turnarounds (docs/litedram.md section 11, "The
chooser and the turnarounds"): with fastmux the multiplexer issues a write 3 cycles after the last
read (rtw 3), the least the PHY allows. A turnaround that comes too early corrupts the burst on
the bus. The DQ / DQS output enables would switch into the read's last data, or the write's
data would go out with the wrong preamble. Each beat carries its ECC byte, so either way the
channel's ECC decoder sees it.

One program, run back to back for --seconds. The MXU streams fp4 weights (port A) while the DMA
stores a 64 KiB tile and loads back the one stored before it (port B), so reads and writes take
turns in both channels all the time. The loads of what was just stored put any write the
turnaround corrupted through the ECC decoder in the same run. Then:
- the stored tiles and the MMs' results are read back and compared with the ISA simulator's;
- both channels' ECC counters (sec_errors / ded_errors, corrected and uncorrectable words since
  the configuration) must be 0.

    python3 tools/qual/turnaround.py [--seconds 30] [--mb 32]

Prints a [PASS] / [FAIL] line per check; exit status 1 when one fails. On the board model
(SimTransport: no ECC) only the data are checked.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from opentpu import isa as I  # noqa: E402
from opentpu import language as ol  # noqa: E402
from opentpu import quant as Q  # noqa: E402
from opentpu.llm.qwen3 import _qdesc, _tdesc  # noqa: E402

H, R, NS = 1024, 1024, 128          # an MM: R rows x H (fp4); a DMA tile: NS x NS words
MB = 1 << 20
TILE = 4 * NS * NS


def layout(n_mm: int, n_dma: int) -> dict:
    rb, sb = Q.row_bytes(H, "fp4", 128), 4 * (H // 128)
    a = {"w": 0, "ws": n_mm * R * rb}
    a["x"] = a["ws"] + n_mm * R * sb
    a["src"] = (a["x"] + 4 * H + MB - 1) // MB * MB
    a["dma"] = a["src"] + TILE
    a["out"] = a["dma"] + n_dma * TILE
    a["prog"] = (a["out"] + 4 * n_mm * R + 4095) // 4096 * 4096
    return a


@ol.jit
def mixed(m, n_mm: int, n_dma: int):
    xs = ol.quantize(ol.load(m.x))
    y = ol.empty([1, R])
    t = ol.load(m.src)
    u = ol.empty([NS * NS])
    per = -(-n_dma // n_mm)
    k = 0
    for i in range(n_mm):
        ol.dot(xs, m.w[i * R:(i + 1) * R, :], out=y)        # port A: the weights
        for _ in range(per):
            if k < n_dma:
                ol.store(m.dma[k, :], t)                    # port B: writes beside them,
                if k:
                    ol.load(m.dma[k - 1, :], out=u)         # and reads of the previous tile
                k += 1
        ol.store(m.out[i:i + 1, :], y)


def build(cfg, n_mm: int, n_dma: int, seed: int = 0):
    """The image, its layout and the assembled program."""
    a = layout(n_mm, n_dma)
    img = np.zeros(a["prog"], np.uint8)
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((R, H)).astype(np.float32)
    q, sc = Q.quantize_mxu(w, "fp4", 128)
    q = np.ascontiguousarray(q).view(np.uint8).ravel()
    sc = np.ascontiguousarray(sc).view(np.uint8).ravel()
    for i in range(n_mm):
        img[a["w"] + i * q.size:a["w"] + (i + 1) * q.size] = q
        img[a["ws"] + i * sc.size:a["ws"] + (i + 1) * sc.size] = sc
    img[a["x"]:a["x"] + 4 * H] = rng.standard_normal(H).astype(np.float32).view(np.uint8)
    img[a["src"]:a["src"] + TILE] = rng.integers(0, 1 << 32, NS * NS, np.uint32).view(np.uint8)
    m = SimpleNamespace(w=_qdesc(a["w"], a["ws"], n_mm * R, H, 128, "fp4"),
                        x=_tdesc(a["x"], (1, H)), src=_tdesc(a["src"], (NS * NS,)),
                        dma=_tdesc(a["dma"], (n_dma, NS * NS)), out=_tdesc(a["out"], (n_mm, R)))
    prog = mixed.trace(cfg, 0, {"m": m, "n_mm": n_mm, "n_dma": n_dma}).finish()
    return img, a, prog


def ecc_counts(t) -> list[tuple[int, int]] | None:
    """Both channels' (sec_errors, ded_errors), or None without ECC (the board model)."""
    if not getattr(t, "ecc", False):
        return None
    from opentpu.host import ddrcal, memcal
    c = memcal.csr(t)
    return [(ddrcal.Chan(c, ch).r("ecc_sec_errors"), ddrcal.Chan(c, ch).r("ecc_ded_errors"))
            for ch in (0, 1)]


def turnarounds(board, seconds: float = 30.0, mb: float = 32.0, runs: int | None = None):
    """[(name, (ok, message))]: the data after `seconds` of the program (or `runs` runs), then the
    ECC counters."""
    from opentpu.host.board import device_config
    from opentpu.isasim import Machine
    n_mm = max(1, round(mb * MB / (R * Q.row_bytes(H, "fp4", 128))))
    n_dma = 8 * n_mm
    size = 1 << (layout(n_mm, n_dma)["prog"] + MB - 1).bit_length()  # the image and the program
    cfg = device_config(board.info(), DRAM_BYTES=size)
    img, a, prog = build(cfg, n_mm, n_dma)
    ref = Machine(cfg, [prog], [np.concatenate([img, np.zeros(size - len(img), np.uint8)])]).run()
    want = ref.slices[0].dram[a["dma"]:a["out"] + 4 * n_mm * R]
    words = np.asarray(I.assemble(prog), np.uint32)
    e0 = ecc_counts(board.t)
    board.scrub()
    board.write(0, img)
    t0, n = time.time(), 0
    while (n < runs) if runs is not None else (n == 0 or time.time() - t0 < seconds):
        board.load_program(a["prog"], words)                # the board model starts from reset
        board.run()
        n += 1
    dt = time.time() - t0
    got = board.read(a["dma"], len(want))
    moved = n * (n_mm * R * (Q.row_bytes(H, "fp4", 128) + 4 * (H // 128)) + (2 * n_dma - 1) * TILE)
    what = (f"{n} runs in {dt:.1f} s ({moved / 1e9:.1f} GB: fp4 weights beside 64 KiB stores "
            f"and loads)")
    bad = np.nonzero(got != want)[0]
    if len(bad):
        o = int(bad[0]) + a["dma"]
        rows = [("data", (False, f"{what}: {len(bad)} bytes differ, the first at {o:#x}"))]
    else:
        rows = [("data", (True, f"{what}: the stored tiles and the MMs' results equal the ISA "
                                f"simulator's"))]
    e1 = ecc_counts(board.t)
    if e1 is None:
        rows.append(("ECC", (True, "no ECC on the board model")))
    else:
        txt = ", ".join(f"ch{ch} sec {s} ded {d}" for ch, (s, d) in enumerate(e1))
        before = ", ".join(f"ch{ch} {s}/{d}" for ch, (s, d) in enumerate(e0))
        rows.append(("ECC", (all(s == 0 and d == 0 for s, d in e1),
                             f"{txt} (before: {before}; the counts since the configuration)")))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--mb", type=float, default=32.0, help="MB of fp4 weights per run")
    ap.add_argument("--dev", default="/dev/xdma0")
    a = ap.parse_args(argv)
    from opentpu.host.board import Board, XdmaTransport
    with Board(XdmaTransport(a.dev)) as b:
        rows = turnarounds(b, a.seconds, a.mb)
    for name, (ok, msg) in rows:
        print(f"  [{'PASS' if ok else 'FAIL'}] DRAM turnarounds, {name}: {msg}", flush=True)
    return 0 if all(ok for _, (ok, _) in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
