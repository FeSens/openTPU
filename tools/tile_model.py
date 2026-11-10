"""The rows a weight pass needs (docs/tiled_prefill.md 2): MXU and weight-stream cycles per
weight chunk and row against the rows T of a pass, and what that gives a model's linear layers.

    python3 tools/tile_model.py knee                   # cycles per chunk and row, by T
    python3 tools/tile_model.py layers [--model qwen3|qwen35|DIR | --shape qwen3-4b]
    python3 tools/tile_model.py moe MODEL_DIR [--tokens 512] [--mcols 4] [--eta 0.85]

A pass of T rows over one 128-byte weight chunk costs max(stream, MXU) cycles:
- stream: 1 / eta (eta: the chunks a cycle the memory delivers to the core's port, at most 1;
  the card's DDR3-1066 through LiteDRAM 0.82-0.85 while decoding, docs/board.md; DDR4 on the
  BCU1525 delivers more than the port takes: 1);
- MXU (docs/isa.md MM): int8, ceil(T / MCOLS) (replayed groups); 4-bit, two blocks a chunk:
  1 with PAIR (T <= MCOLS / 2), else 2 ceil(T / MCOLS).
Per row that is max(stream, MXU) / T. The floor (the MXU's multiply rate) is 1 / MCOLS cycles
a chunk and row for int8 and 2 / MCOLS for 4-bit; the knee is the fewest rows a pass that
reach it.

moe: the expected rows each expert of a layer gets from a chunk of N tokens (k of E experts a
token, uniform routing: an upper bound on the spread; real routers concentrate), the passes of
today's expert-major run (pairs) against passes of up to ACT_ROWS rows (replay), and their
cycles from the one-expert costs c(M) of tools/moe_prefill_cosim.py expert.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def mxu_cycles(T: int, mcols: int, fmt: str) -> int:
    if fmt == "int8":
        return math.ceil(T / mcols)
    return 1 if 2 * T <= mcols else 2 * math.ceil(T / mcols)


def per_row(T: int, mcols: int, fmt: str, eta: float) -> float:
    return max(1.0 / eta, mxu_cycles(T, mcols, fmt)) / T


def floor(mcols: int, fmt: str) -> float:
    """The MXU's multiply rate as cycles per chunk and row: 1 / MCOLS (int8), 2 / MCOLS (4-bit:
    two blocks a chunk)."""
    return (1 if fmt == "int8" else 2) / mcols


def knee(mcols: int, fmt: str, eta: float) -> int:
    """The fewest rows a pass at the floor."""
    T = 1
    while per_row(T, mcols, fmt, eta) > floor(mcols, fmt) + 1e-9:
        T += 1
    return T


def cmd_knee(a):
    Ts = (1, 2, 4, 8, 16, 32, 64)
    for fmt in ("fp4", "int8"):
        for eta in (0.85, 1.0):
            print(f"{fmt}, eta {eta}: cycles per chunk and row (MXU-bound from the knee on)")
            print("  MCOLS " + "".join(f"{f'T={t}':>8s}" for t in Ts) + "    knee   floor")
            for mc in (4, 8, 16):
                fl = floor(mc, fmt)
                print(f"  {mc:5d} " + "".join(f"{per_row(t, mc, fmt, eta):8.3f}" for t in Ts)
                      + f"  {knee(mc, fmt, eta):6d}  {fl:6.3f}")


def spec_of(a):
    from opentpu.llm import load_spec, model_dir
    if a.shape:
        from opentpu.llm.qwen3 import Spec
        from tools.perf_tiled import SHAPES
        return Spec(**SHAPES[a.shape])
    return load_spec(model_dir(a.model))


def layer_chunks(spec, fmt: str) -> dict:
    """A dense attention layer's weight chunks (128 bytes) by kind (Qwen3's shapes)."""
    H, d, F = spec.hidden, spec.head_dim, spec.ffn
    per = 1 if fmt == "int8" else 0.5             # bytes a weight
    attn = (2 * spec.n_q * d * H + 2 * spec.n_kv * d * H) * per / 128
    mlp = 3 * H * F * per / 128
    return {"attn": attn, "mlp": mlp}


def cmd_layers(a):
    spec = spec_of(a)
    for fmt in ("fp4", "int8"):
        ch = layer_chunks(spec, fmt)
        tot = sum(ch.values())
        print(f"{fmt}: {tot / 1e3:.1f}K chunks a layer (attention {ch['attn'] / 1e3:.1f}K, "
              f"MLP {ch['mlp'] / 1e3:.1f}K); linear-layer cycles a row and layer:")
        for mc in (4, 8):
            for eta in (0.85, 1.0):
                cells = "  ".join(f"T={t}: {tot * per_row(t, mc, fmt, eta) / 1e3:6.1f}K"
                                  for t in (2, 4, 8, 16, 32))
                print(f"  MCOLS {mc} eta {eta}: {cells}")


def cmd_moe(a):
    """Rows per expert from a chunk of N tokens, uniform routing over E experts, k each."""
    c = json.loads((Path(a.model) / "config.json").read_text())
    t = c.get("text_config", c)
    E = t.get("num_experts")
    k = t.get("num_experts_per_tok") or t.get("top_k_experts")
    N = a.tokens
    p = k / E                                       # a token picks a given expert
    used = E * (1 - (1 - p) ** N)
    mean = N * k / used
    print(f"{Path(a.model).name}: E {E}, k {k}; a chunk of {N} tokens: {used:.0f} experts used "
          f"a layer, {mean:.1f} rows each on average (uniform routing)")
    # passes: today's pairs (ceil(n / 2)) against replayed passes of up to R rows
    from math import comb
    def pmf(n):
        return comb(N, n) * p ** n * (1 - p) ** (N - n)
    for R in (2, 4, 8, 16, 32):
        passes = sum(pmf(n) * math.ceil(n / R) for n in range(1, N + 1)) * E
        print(f"  passes of up to {R:2d} rows: {passes:6.0f} a layer "
              f"(each streams the expert once)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("knee")
    la = sub.add_parser("layers")
    la.add_argument("--model", default="qwen3")
    la.add_argument("--shape", default=None)
    mo = sub.add_parser("moe")
    mo.add_argument("model")
    mo.add_argument("--tokens", type=int, default=512)
    a = ap.parse_args()
    {"knee": cmd_knee, "layers": cmd_layers, "moe": cmd_moe}[a.cmd](a)


if __name__ == "__main__":
    main()
