"""Card qualification of WAITW (CAPS bit31) on the host's writes: the host writes data, then a flag
word, while the card waits on the flag; the LD after the card's WAITW must read the host's data.
That is the order of XDMA's writes through otpu_mem_ch into LiteDRAM against the accelerator's
reads (opentpu/host/checks.py waitw_host). Then a WAITW that never holds must stop the card with
ERROR at its timeout, and the next run must halt normally (waitw_timeout). With --tag-rounds,
docs/offload.md 10.11's order: a flag in the last beat of the DMA that carries 1-4 MiB of data
(checks.waitw_tag: an expert and its slot's tag, as BoardDram sends them).

    python3 tools/qual/waitw.py [--rounds 200] [--seed 5] [--tag-rounds 2000]

Prints a [PASS] / [FAIL] line per check; exit status 1 when one fails.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from opentpu.host.board import Board, XdmaTransport  # noqa: E402
from opentpu.host.checks import waitw_host, waitw_tag, waitw_timeout  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--tag-rounds", type=int, default=0,
                    help="rounds of the tag in the data's last beat (0: none)")
    ap.add_argument("--dev", default="/dev/xdma0")
    a = ap.parse_args(argv)
    with Board(XdmaTransport(a.dev)) as b:
        if not b.info()["caps"].get("waitw"):
            print("  [FAIL] WAITW: this bitstream has no WAITW (CAPS bit31 clear)")
            return 1
        rows = [("WAITW on the host's writes", waitw_host(b, a.rounds, a.seed)),
                ("WAITW timeout", waitw_timeout(b))]
        if a.tag_rounds:
            rows.append(("WAITW on a tag in the data's last beat",
                         waitw_tag(b, a.tag_rounds, a.seed)))
    for name, (ok, msg) in rows:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {msg}", flush=True)
    return 0 if all(ok for _, (ok, _) in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
