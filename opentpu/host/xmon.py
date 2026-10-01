"""The debug build's DMA monitors (rtl/boards/ypcb-00338/otpu_xmon.sv; run_vivado.sh XMON=1):
self-describing data and the monitors' registers (otpu_ctrl 0xF00 + 4k).

Self-describing data: every 16-byte beat holds the 32-bit words {MAGIC, its byte address as on
XDMA's DMA master (board.BASE[ch] + offset: bit 31 is the channel), the address inverted, a tag}.
The monitors check each such beat on XDMA's master (against its AW / AR address) and on channel
0's controller ports (against its command's beat); see otpu_xmon.sv for the flags and words.
"""
from __future__ import annotations

import time

import numpy as np

from .board import BASE

MAGIC = 0x584D4F4E
REG = 0xF00                       # word k at REG + 4k; writing REG: bit0 SNAP, bit1 CLEAR
NAMES = ["FLAGS", "SNAPS", "X_AW", "X_W", "X_B", "X_AR", "X_R", "X_WCHK", "X_RCHK", "X_WBAD",
         "X_RBAD", "X_WEXP", "X_WGOT", "X_WTAG", "X_REXP", "X_RGOT", "X_STALL", "X_STALLN",
         "X_LIVE", "X_LIVEN", "N_WCMD", "N_WDAT", "N_RCMD", "N_RDAT", "N_WCHK", "N_RCHK", "N_WBAD",
         "N_RBAD", "N_WEXP", "N_WGOT", "N_REXP", "N_RGOT"]
FLAGS = ["X_WSHIFT", "X_RSHIFT", "X_WLAST", "X_RLAST", "X_WSTALL", "X_BSTALL", "X_RSTALL", "X_PROTO",
         "N_WSHIFT", "N_RSHIFT", "N_PROTO"]
HSK = ["awvalid", "awready", "wvalid", "wready", "wlast", "bvalid", "bready", "arvalid", "arready",
       "rvalid", "rready", "rlast"]


def present(t) -> bool:
    return t.reg_read(REG) >> 16 == 0x584D


def flags(t) -> int:
    """The sticky flags (live)."""
    return t.reg_read(REG) & 0xFFFF


def flag_names(f: int) -> list[str]:
    return [n for i, n in enumerate(FLAGS) if f >> i & 1]


def clear(t) -> None:
    t.reg_write(REG, 2)


def snap(t, timeout: float = 0.1) -> dict[str, int]:
    """SNAP, wait for both clocks' acknowledgements, read all words."""
    req = (t.reg_read(REG + 4) + 1) & 0xFF
    t.reg_write(REG, 1)
    t0 = time.time()
    while True:
        s = t.reg_read(REG + 4)
        if s & 0xFF == req and s >> 8 & 0xFF == req and s >> 16 & 0xFF == req:
            break
        if time.time() - t0 > timeout:
            raise TimeoutError(f"otpu_xmon SNAP not acknowledged (SNAPS {s:#010x}, want {req:#x})")
    return dict(zip(NAMES, t.reg_read_many([REG + 4 * k for k in range(32)])))


def _stall(w: int, n: int) -> str:
    if not w >> 15 & 1:
        return "none"
    fired = [k for k, b in (("W", 12), ("B", 13), ("R", 14)) if w >> b & 1]
    hs = [h for i, h in enumerate(HSK) if w >> i & 1]
    return (f"watchdogs {'/'.join(fired) or '-'}; high: {' '.join(hs) or '-'}; W beat {w >> 16 & 0xFF},"
            f" R beat {w >> 24 & 0xFF}; open: AW {n & 0xFF}, B {n >> 8 & 0xFF}, AR {n >> 16 & 0xFF},"
            f" W burst's AWLEN {n >> 24 & 0xFF}")


def describe(w: dict[str, int]) -> str:
    """The words, readable."""
    f = w["FLAGS"] & 0xFFFF
    out = [f"flags {f:#06x} {' '.join(flag_names(f)) or '(none)'}",
           f"XDMA master: AW {w['X_AW']} W {w['X_W']} B {w['X_B']} AR {w['X_AR']} R {w['X_R']};"
           f" checked W {w['X_WCHK']} R {w['X_RCHK']}; bad W {w['X_WBAD']} R {w['X_RBAD']}",
           f"channel 0 ports: write cmd {w['N_WCMD']} data {w['N_WDAT']}, read cmd {w['N_RCMD']}"
           f" data {w['N_RDAT']}; checked W {w['N_WCHK']} R {w['N_RCHK']}; bad W {w['N_WBAD']}"
           f" R {w['N_RBAD']}"]
    if w["X_WBAD"]:
        out.append(f"first bad W beat: at {w['X_WEXP']:#010x} it held {w['X_WGOT']:#010x}"
                   f" ({w['X_WGOT'] - w['X_WEXP']:+d} B), tag {w['X_WTAG']:#x}")
    if w["X_RBAD"]:
        out.append(f"first bad R beat: at {w['X_REXP']:#010x} it held {w['X_RGOT']:#010x}"
                   f" ({w['X_RGOT'] - w['X_REXP']:+d} B)")
    for d in ("W", "R"):
        if w[f"N_{d}BAD"]:
            e, g = w[f"N_{d}EXP"], w[f"N_{d}GOT"]
            out.append(f"first bad channel-0 {d} beat: port {e & 1}, command beat {e & ~0x3F:#010x},"
                       f" data described {g & ~0x3F:#010x} ({(g & ~0x3F) - (e & ~0x3F):+d} B)"
                       f"{', its parts disagreeing' if g & 1 else ''}")
    out.append(f"first stall: {_stall(w['X_STALL'], w['X_STALLN'])}")
    out.append(f"at the SNAP: {_stall(w['X_LIVE'], w['X_LIVEN'])}")
    return "\n".join(out)


def data(ch: int, off: int, n: int, tag: int = 0) -> np.ndarray:
    """Self-describing bytes for channel ch at offset off (16-byte aligned, n a multiple of 16)."""
    assert off % 16 == 0 and n % 16 == 0
    a = (BASE[ch] + off + 16 * np.arange(n // 16, dtype=np.uint64)).astype(np.uint32)
    w = np.empty((n // 16, 4), np.uint32)
    w[:, 0] = MAGIC
    w[:, 1] = a
    w[:, 2] = ~a
    w[:, 3] = tag & 0xFFFFFFFF
    return w.reshape(-1).view(np.uint8)


def first_bad(got: np.ndarray, ch: int, off: int) -> tuple[int, int] | None:
    """The first 16-byte beat of `got` (read from ch at off) that is not self-describing at its
    own address: (its offset in got, the address it described, or -1 if none)."""
    w = np.ascontiguousarray(got).view(np.uint32).reshape(-1, 4)
    a = (BASE[ch] + off + 16 * np.arange(len(w), dtype=np.uint64)).astype(np.uint32)
    ok = (w[:, 0] == MAGIC) & (w[:, 1] == a) & (w[:, 2] == ~a)
    if ok.all():
        return None
    i = int(np.flatnonzero(~ok)[0])
    sd = w[i, 0] == MAGIC and w[i, 2] == ~w[i, 1]
    return 16 * i, int(w[i, 1]) if sd else -1
