"""The board model's ACTs by cause (sim/verilator/otpu_dimm_model.sv +mc_actlog=PREFIX: a line an
ACT in PREFIXd<DIMM>.txt), from each file's largest simulation (the decode; the small ones before
it append their own "#" sections):

    python3 tools/actbin.py PREFIXd0.txt PREFIXd1.txt ... [--bhw 33] [--nch 8] [--json OUT]

Per DIMM and in all: the ACTs by why the bank was closed (P a precharge for this row, R a
refresh, C never open); the precharges' by the two commands (the closed row's last CAS and the
one the row opens for: read or write; the same channel or another of the DIMM; how far apart
their channel bytes are; the regions they fall in: lo, the bank-half window's lower half, mix its
upper, hi above it, as a slice of --nch channels and its window at 2^--bhw bytes puts them); and
how many reopen the row the bank held before the last one (two rows taking turns on a bank)."""
import argparse
import collections
import json

DIST = [(1 << 14, "<16K"), (1 << 15, "16-32K"), (3 << 14, "32-48K"), (1 << 16, "48-64K"),
        (1 << 17, "64-128K"), (1 << 20, "128K-1M"), (1 << 24, "1-16M"), (1 << 28, "16-256M"),
        (None, ">=256M")]


def sections(path):
    """The file's "#" sections: (header, ACT lines)."""
    out, cur = [], None
    for ln in open(path):
        if ln.startswith("#"):
            cur = (ln[1:].strip(), [])
            out.append(cur)
        elif cur is not None and ln.strip():
            cur[1].append(ln.split())
    return out


def dist(d):
    for lim, name in DIST:
        if lim is None or d < lim:
            return name


def region(off, lo_end, mix_end):
    return "lo" if off < lo_end else "mix" if off < mix_end else "hi"


def binned(rows, lo_end, mix_end):
    c = collections.Counter()
    last = {}                                       # (group, bank): the rows it opened, last two
    for r in rows:
        cyc, g, b, row, old, why, we, ch, off, owe, och, ooff = r
        k = (g, b)
        c["ACT"] += 1
        c["why " + why] += 1
        prev = last.get(k, [])
        if len(prev) == 2 and prev[0] == row:
            c["turns"] += 1
        last[k] = (prev + [row])[-2:]
        if why != "P":
            continue
        c["P " + "RW"[int(owe)] + ">" + "RW"[int(we)]] += 1
        rel = "same" if ch == och else "other"
        c[f"P {rel} {dist(abs(int(off, 16) - int(ooff, 16)))}"] += 1
        ro, rn = region(int(ooff, 16), lo_end, mix_end), region(int(off, 16), lo_end, mix_end)
        c[f"P {ro}>{rn}"] += 1
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--bhw", type=int, default=33, help="the slice's bank-half window, log2 bytes")
    ap.add_argument("--nch", type=int, default=8, help="the slice's channels")
    ap.add_argument("--json")
    a = ap.parse_args()
    lo_end = (1 << (a.bhw - 1)) // a.nch            # (a slice byte x is about channel byte x / nch)
    mix_end = (1 << a.bhw) // a.nch
    per, tot = {}, collections.Counter()
    for f in a.files:
        secs = sections(f)
        if not secs:
            continue
        head, rows = max(secs, key=lambda s: len(s[1]))
        c = binned(rows, lo_end, mix_end)
        per[f] = {"header": head, **c}
        tot.update(c)
    keys = ["ACT", "why C", "why R", "why P", "turns", "P R>R", "P R>W", "P W>R", "P W>W"]
    keys += [f"P {rel} {n}" for rel in ("same", "other") for _, n in DIST]
    keys += sorted(k for k in tot if k[:2] == "P " and ">" in k and k[2:4] in ("lo", "mi", "hi"))
    w = max(len(k) for k in keys)
    names = " ".join(f"{f.rsplit('/', 1)[-1][-8:]:>9s}" for f in per)
    print(f"{'':{w}s} {names} {'all':>9s} {'%':>6s}")
    for k in keys:
        if not tot[k]:
            continue
        print(f"{k:{w}s} " + " ".join(f"{per[f].get(k, 0):9d}" for f in per) +
              f" {tot[k]:9d} {100 * tot[k] / max(1, tot['ACT']):6.1f}")
    if a.json:
        json.dump({"per": per, "all": dict(tot)}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
