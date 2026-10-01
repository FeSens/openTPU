"""Print an otpu_xmon ILA capture (tools/xmon_ila.tcl's ila_x.csv / ila_n.csv) as one line per
sample around the trigger: the handshakes by name, the addresses and the events.

    python3 tools/xmon_ila_dump.py ila_x.csv [BEFORE] [AFTER]     (default 60 before, 20 after)
"""
from __future__ import annotations

import csv
import sys

HSK_X = ["awv", "awr", "wv", "wr", "wl", "bv", "br", "arv", "arr", "rv", "rr", "rl"]
EV_X = ["WSHIFT", "RSHIFT", "WLAST", "RLAST", "WSTALL", "BSTALL", "RSTALL", "PROTO", "hs", "aw/w",
        "N_FLAG"]
HSK_N = ["cv0", "cv1", "cr0", "cr1", "we0", "we1", "wv0", "wv1", "wr0", "wr1", "rv0", "rv1"]
EV_N = ["WBAD0", "WBAD1", "RBAD0", "RBAD1", "WOVF0", "WOVF1", "ROVF0", "ROVF1", "hs"]


def num(v: str) -> int:
    v = v.strip()
    try:
        return int(v, 16)
    except ValueError:
        return int(v, 2) if set(v) <= {"0", "1"} else 0


def col(names: list[str], pat: str) -> int | None:
    for i, n in enumerate(names):
        if pat in n:
            return i
    return None


def bits(v: int, names: list[str]) -> str:
    return " ".join(n for i, n in enumerate(names) if v >> i & 1)


def main(a: list[str]) -> int:
    path = a[0]
    before, after = (int(a[1]) if len(a) > 1 else 60), (int(a[2]) if len(a) > 2 else 20)
    rows = list(csv.reader(open(path)))
    names = rows[0]
    data = [r for r in rows[1:] if r and r[0].strip().isdigit()]
    trig = col(names, "TRIGGER")
    win = col(names, "Sample in Window")
    is_x = col(names, "ix_") is not None
    t = next((i for i, r in enumerate(data) if trig is not None and r[trig].strip() == "1"), len(data) - 1)
    print(f"{path}: {len(data)} samples, trigger at {t}; columns: {', '.join(names)}")
    for i in range(max(0, t - before), min(len(data), t + after + 1)):
        r = data[i]
        f = {n: num(r[j]) for j, n in enumerate(names)}
        g = lambda pat: next((v for n, v in f.items() if pat in n), 0)  # noqa: E731
        if is_x:
            hs, ad = g("ix_hsids"), g("ix_addr")
            line = (f"{bits(hs & 0xFFF, HSK_X):40s} aw {ad & 0xFFFFFFFF:08x}/{ad >> 32 & 0xFF:3d} "
                    f"w {g('ix_wtagaddr') & 0xFFFFFFFF:08x} t{g('ix_wtagaddr') >> 32:x} "
                    f"ar {ad >> 40 & 0xFFFFFFFF:08x}/{ad >> 72:3d} r {g('ix_rtagaddr') & 0xFFFFFFFF:08x} "
                    f"ids {hs >> 12:04x} {bits(g('ix_ev'), EV_X)}")
        else:
            a_ = g("in_addr")
            line = (f"{bits(g('in_hsk'), HSK_N):40s} c0 {a_ & 0x1FFFFFF:07x} c1 {a_ >> 25:07x} "
                    f"wd {g('in_waddr') & 0xFFFFFFFF:08x}/{g('in_waddr') >> 32:08x} "
                    f"rd {g('in_raddr') & 0xFFFFFFFF:08x}/{g('in_raddr') >> 32:08x} {bits(g('in_ev'), EV_N)}")
        mark = ">>" if i == t else "  "
        print(f"{mark}{(r[win].strip() if win is not None else str(i)):>5} {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
