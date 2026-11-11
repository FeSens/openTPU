"""x2.py: two cards in tensor parallel (targets/bcu1525-x2.toml; opentpu.host.cards).

  python3 tools/x2.py compile [--target bcu1525-x2:st16] POINT ...
      Each decode POINT (tools/board_perf.py decode_case's MODEL:WFORMAT:HEAD:POS[:K=V,...]; a
      checkpoint directory works as MODEL, only its config.json is read) compiled at the
      target's S, with its wait entries, then split by card as opentpu.host.cards loads it:
      card c runs global slices c * spc .. c * spc + spc - 1 (spc = S / cards), each from the
      same program area, every card the same addresses. "gen@POINT": the generate loop of
      POINT's run-time position (board_perf generate_case). Prints each card's instructions, the
      programs' IMEM rows against IMEM_WORDS and each card's image bytes (its slices'); exit 1
      when a program does not fit.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from opentpu import isa as I           # noqa: E402
from opentpu import target as T        # noqa: E402
from opentpu import waits as W         # noqa: E402


def compile_point(cfg, pt: str) -> dict:
    """POINT's programs and image at cfg (S slices), its wait entries, by card."""
    import board_perf
    gen = pt.startswith("gen@")
    if gen:
        c, img, progs, at, _ = board_perf.generate_case(cfg, pt[4:])
    else:
        c, img, progs, at, _ = board_perf.decode_case(cfg, pt)
    progs = list(progs)
    words = [np.asarray(I.assemble(p), np.uint32) for p in progs]
    ents = [W.entries(w, c.D, c.S, prog=p, aw=c.AW, win=c.WC_WIN) if c.SEQ_WC else None
            for w, p in zip(words, progs)]
    return {"cfg": c, "programs": progs, "words": words, "entries": ents, "image": img,
            "at": at, "gen": gen}


def by_card(r: dict, cards: int) -> list[dict]:
    """compile_point's result split as the cards load it: card c's slices, programs, image
    bytes (a slice's: every slice's image from address 0 in its own DRAM, the programs at
    `at` past it)."""
    S = r["cfg"].S
    spc = S // cards
    out = []
    for c in range(cards):
        ss = range(c * spc, (c + 1) * spc)
        out.append({"card": c, "slices": list(ss),
                    "instructions": [len(r["programs"][s]) for s in ss],
                    "rows": [len(r["words"][s]) // 8 for s in ss],
                    "image": r["image"].nbytes * spc})
    return out


def cmd_compile(a) -> int:
    t = T.load(a.target)
    cards = T.cards(t)
    cfg = T.config(t)
    bad = 0
    print(f"{T.full_name(t)}: S {cfg.S} over {cards} cards ({cfg.S // cards} a card), "
          f"IMEM {cfg.IMEM_WORDS} words, wait entries {'yes' if cfg.SEQ_WC else 'no'}")
    for pt in a.points:
        r = compile_point(cfg, pt)
        cap = r["cfg"].IMEM_WORDS // 8                  # (instruction rows of 8 words)
        print(f"{pt}: {'generate' if r['gen'] else 'decode'} at {r['at']:#x}, DRAM "
              f"{r['cfg'].DRAM_BYTES >> 20} MiB a slice")
        for d in by_card(r, cards):
            fit = all(n <= cap for n in d["rows"])
            bad += not fit
            print(f"  card {d['card']}: slices {d['slices']} instructions {d['instructions']}"
                  f" rows {d['rows']} of {cap}{'' if fit else ' (DOES NOT FIT)'}, images "
                  f"{d['image'] / 2**20:.1f} MiB")
        same = len({len(p) for p in r["programs"]}) == 1
        print(f"  slices' programs {'the same length' if same else 'of different lengths'}"
              f"{'; entries computed' if r['entries'][0] is not None else ''}")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compile", help="compile decode / generate points for the cards")
    c.add_argument("--target", default="bcu1525-x2:st16")
    c.add_argument("points", nargs="+")
    a = ap.parse_args(argv)
    return {"compile": cmd_compile}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
