"""The MXU's weight stream beside DMA traffic, on the card and on the RTL simulator: how much
of the DRAM a 4-bit weight stream gets while the DMA reads, writes or runs DSTEP (Qwen3.5's
DeltaNet heads) at the same time.

    python3 tools/rw_bench.py card|sim [--modes mm,mm+st,mm+ld,mm+dstep,st,ld,dstep]
                                       [--mb 8] [--mhz 120.755] [--json out.json]

Each mode is one program: `mm` streams --mb MB of fp4 weights as MMs of 1024 rows x 1024;
`st` / `ld` store / load 64 KiB tiles (distinct DRAM addresses), `dstep` runs DSTEPs on 128 x
128 states (64 KiB read, 64 KiB written each); `mm+X` issues them interleaved, so the units
overlap. Prints cycles and bytes per cycle (the card's counters, or the simulator's cycles).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from opentpu import isa as I  # noqa: E402
from opentpu import language as ol  # noqa: E402
from opentpu import quant as Q  # noqa: E402
from opentpu import rtlsim  # noqa: E402
from opentpu.llm.qwen3 import _qdesc, _tdesc  # noqa: E402
from opentpu.profile import ddr3_plusargs  # noqa: E402

H, R, NS = 1024, 1024, 128          # MM: rows x H; DMA tile: 16K words; DSTEP state rows
MB = 1 << 20


def layout(n_mm: int, n_dma: int) -> dict:
    rb, sb = Q.row_bytes(H, "fp4", 128), 4 * (H // 128)
    a = {"w": 0, "ws": n_mm * R * rb}
    a["x"] = a["ws"] + n_mm * R * sb
    a["st"] = (a["x"] + 4 * H + MB - 1) // MB * MB
    a["dma"] = a["st"] + n_dma * 4 * NS * NS
    a["end"] = a["dma"] + n_dma * 4 * NS * NS
    return a


@ol.jit
def bench(m, mode: str, n_mm: int, n_dma: int):
    xs = ol.quantize(ol.load(m.x))
    y = ol.empty([1, R])
    t = ol.zeros([NS * NS])
    qk, v, eb, o = ol.zeros([2 * NS]), ol.zeros([NS]), ol.zeros([4]), ol.empty([NS])
    eb[0:1].set(0.5)
    dma = mode.split("+")[-1] if mode != "mm" else None
    per = -(-n_dma // max(1, n_mm)) if "mm" in mode else n_dma
    k = 0
    for i in range(max(n_mm if "mm" in mode else 0, 1)):
        if "mm" in mode:
            ol.dot(xs, m.w[i * R:(i + 1) * R, :], out=y)
        for _ in range(per):
            if dma is None or k >= n_dma:
                break
            if dma == "st":
                ol.store(m.dma[k, :], t)
            elif dma == "ld":
                ol.load(m.dma[k, :], out=t)
            elif dma == "dstep":
                ol.deltanet_step(m.state[k], qk, v, eb[0:1], eb[2:3], o)
            k += 1
    ol.store(m.out, y)


def image(n_mm: int, n_dma: int) -> tuple[np.ndarray, dict]:
    a = layout(n_mm, n_dma)
    img = np.zeros(a["end"] + 4 * R, np.uint8)
    rng = np.random.default_rng(0)
    w = rng.standard_normal((R, H)).astype(np.float32)
    q, sc = Q.quantize_mxu(w, "fp4", 128)
    q, sc = np.ascontiguousarray(q).view(np.uint8).ravel(), np.ascontiguousarray(sc).view(np.uint8).ravel()
    for i in range(n_mm):
        img[a["w"] + i * q.size:a["w"] + (i + 1) * q.size] = q
        img[a["ws"] + i * sc.size:a["ws"] + (i + 1) * sc.size] = sc
    x = rng.standard_normal(H).astype(np.float32)
    img[a["x"]:a["x"] + 4 * H] = x.view(np.uint8)
    st = (0.01 * rng.standard_normal(n_dma * NS * NS)).astype(np.float32)
    img[a["st"]:a["st"] + st.nbytes] = st.view(np.uint8)
    img[a["dma"]:a["end"]] = 0
    return img, a


def descs(a: dict, n_mm: int, n_dma: int) -> SimpleNamespace:
    return SimpleNamespace(
        w=_qdesc(a["w"], a["ws"], n_mm * R, H, 128, "fp4"), x=_tdesc(a["x"], (1, H)),
        state=_tdesc(a["st"], (n_dma, NS, NS)), dma=_tdesc(a["dma"], (n_dma, NS * NS)),
        out=_tdesc(a["end"], (1, R)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("where", choices=["card", "sim"])
    ap.add_argument("--modes", default="mm,st,ld,dstep,mm+st,mm+ld,mm+dstep")
    ap.add_argument("--mb", type=float, default=8.0, help="MB of fp4 weights per MM mode")
    ap.add_argument("--mhz", type=float, default=120.755)
    ap.add_argument("--dev", default="/dev/xdma0")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    n_mm = max(1, round(a.mb * MB / (R * Q.row_bytes(H, "fp4", 128))))
    n_dma = {"st": 64, "ld": 64, "dstep": 32}
    if a.where == "card":
        from opentpu.host.board import Board, XdmaTransport, device_config
        board = Board(XdmaTransport(a.dev))
        board.scrub()
        cfg = device_config(board.info())
    else:
        from opentpu.isasim import board_config
        board, cfg = None, board_config(PAIR=True, DSTEP=True, DRAM_BYTES=1 << 28)
    out = []
    try:
        for mode in a.modes.split(","):
            nd = n_dma.get(mode.split("+")[-1], 0)
            img, lay = image(n_mm if "mm" in mode else 1, max(nd, 1))
            prog = bench.trace(cfg, 0, {"m": descs(lay, n_mm if "mm" in mode else 1, max(nd, 1)),
                                        "mode": mode, "n_mm": n_mm, "n_dma": nd}).finish()
            words = np.asarray(I.assemble(prog), np.uint32)
            if board is not None:
                at = -(-len(img) // 4096) * 4096
                board.write(0, img)
                board.load_program(at, words)
                st = board.run()
                cyc = st["cycles"]
            else:
                mts = 3200 / 3
                _, _, st = rtlsim.run(cfg, [prog], [img], uarch={**rtlsim.BOARD_UARCH, "AXI_BL": 8},
                                      axi=True, boot=True, stall=0, bw=100,
                                      lat=round(0.3 * a.mhz), arc=4, max_cycles=1 << 40,
                                      plusargs=ddr3_plusargs(mts, a.mhz))
                cyc = st["cycles"]
            mmb = (n_mm * R * (Q.row_bytes(H, "fp4", 128) + 4 * (H // 128))) if "mm" in mode else 0
            dmab = nd * 4 * NS * NS * (2 if mode.endswith("dstep") else 1)
            r = {"mode": mode, "cycles": int(cyc), "mm_bytes": mmb, "dma_bytes": dmab,
                 "bpc": (mmb + dmab) / cyc}
            out.append(r)
            print(f"{mode:9s} {cyc:9d} cycles  MM {mmb / 1e6:6.2f} MB  DMA {dmab / 1e6:6.2f} MB  "
                  f"{(mmb + dmab) / cyc:6.1f} B/cycle", flush=True)
    finally:
        if board is not None:
            board.close()
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
