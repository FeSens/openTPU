"""Lens: profile the standard openTPU kernels on the RTL into one standalone HTML page.

(The Lens program itself is `python -m opentpu.lens` -- record / open / html; see docs/lens.md.)

    python3 tools/lens.py                       # the standard workloads -> build/lens.html
    python3 tools/lens.py mlp attn --out x.html
    python3 tools/lens.py --list

Every workload is compiled, run on the Verilator RTL with tracing, and turned into timeline,
roofline, unit-utilisation, bottleneck and source-line views (opentpu/lens.py).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from opentpu.isasim import design_config  # noqa: E402
from opentpu.kernels import attention_decode, attention_layer, mlp  # noqa: E402
from opentpu.lens import write_report  # noqa: E402
from opentpu.profile import profile  # noqa: E402
from test_kernels import attn_args, layer_args, mlp_args  # noqa: E402


def _mlp(M, H=1024, F=4096):
    def run():
        a, _ = mlp_args(np.random.default_rng(0), M=M, H=H, Fd=F)
        return profile(mlp, design_config(), f"MLP decode M={M}" if M == 1 else f"MLP M={M}", **a)
    return run


def _attn(Hq, Hkv, T, lanes=16, name=None):
    def run():
        a, _ = attn_args(np.random.default_rng(1), Hq=Hq, Hkv=Hkv, d=128, T=T, cap=T, block=128)
        cfg = design_config(LANES=lanes)
        return profile(attention_decode, cfg,
                       name or f"Flash attention {Hq}q/{Hkv}kv T={T}", **a)
    return run


def _layer(pos=1023):
    def run():
        cfg = design_config()
        a, _ = layer_args(np.random.default_rng(2), cfg.S, H=1024, Hq=16, Hkv=4, d=128, pos=pos,
                          cap=pos + 129)
        a["block"] = 128
        return profile(attention_layer, cfg, f"Attention layer pos={pos}", **a)
    return run


WORKLOADS = {
    "mlp": _mlp(1),
    "mlp8": _mlp(8),
    "attn": _attn(16, 4, 2048),
    "layer": _layer(),
    "attn-narrow": _attn(6, 1, 512, lanes=8, name="Attention G=6 T=512, 8 lanes"),
}
DEFAULT = ["mlp", "mlp8", "attn", "layer", "attn-narrow"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("workloads", nargs="*", default=DEFAULT)
    ap.add_argument("--out", default=str(ROOT / "build" / "lens.html"))
    ap.add_argument("--fragment", action="store_true",
                    help="omit the <html>/<head> skeleton (for embedding)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        print("\n".join(WORKLOADS))
        return
    profiles = []
    for w in args.workloads:
        p = WORKLOADS[w]()
        print(p.summary().splitlines()[0])
        profiles.append(p)
    out = write_report(profiles, args.out, standalone=not args.fragment)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
