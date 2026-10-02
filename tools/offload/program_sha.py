"""sha256 of the MoE models' programs (and two dense ones'), from their config.json files only (no
weights): two trees give the same programs when every line matches (docs/offload.md 12.4).

    python3 tools/offload/program_sha.py CFGDIR [--cfg card.pkl]

CFGDIR holds a directory per model with its config.json: gemma-4-E2B, gemma-4-E4B,
LFM2.5-8B-A1B, Qwen3.5-35B-A3B, gemma-4-26B-A4B (a missing one is skipped). --cfg: the card's
configuration (tools/qual/refs.py cfg: PAIR, DSTEP and STREAM on), else isasim.board_config(),
which has them off, so a change behind those flags (moe-pair's paired experts) shows only with
--cfg. Per model: decode and generate at buckets 1-3 (the 26B's 1 and 16; its generate in two
parts), two steps; then tests' tiny MoE models when run from a checkout's root.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import os
import pickle
import sys
from pathlib import Path

import numpy as np

from opentpu import isa as I
from opentpu.isasim import board_config
from opentpu.llm import load_spec


def walk(h, x) -> int:
    if isinstance(x, (list, tuple)) and x and isinstance(x[0], I.Instr):
        h.update(np.asarray(I.assemble(list(x))).tobytes())
        return 1
    return sum(walk(h, y) for y in x) if isinstance(x, (list, tuple)) else 0


def gen(img, b, lo):
    try:
        return img.compile_generate(b, lo)
    except Exception:                               # noqa: BLE001 (too long: in two parts)
        return [img.compile_generate(b, lo, part=p) for p in (0, 1)]


def progs(name, img, lo0=0, buckets=(1, 2, 3)) -> None:
    h = hashlib.sha256()
    blk = getattr(img, "block", None) or sys.modules[type(img).__module__].ATTN_BLOCK
    n = 0
    for b in buckets:
        lo = lo0 + (b - 1) * blk
        if lo >= img.cap:
            continue
        n += walk(h, img.compile_decode(b, lo)[0])
        n += walk(h, gen(img, b, lo))
    for p in (lo0 + 1, min(lo0 + 300, img.cap - 1)):
        n += walk(h, img.compile_step(p))
    print(f"{name} (cap {img.cap}): {n} programs {h.hexdigest()[:16]}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cfgdir")
    ap.add_argument("--cfg", help="the card's configuration (a pickle from tools/qual/refs.py cfg)")
    a = ap.parse_args()
    C = Path(a.cfgdir)
    cfg = pickle.load(open(a.cfg, "rb")) if a.cfg else board_config()
    have = lambda m: (C / m / "config.json").exists()    # noqa: E731
    if have("gemma-4-E2B"):
        s = load_spec(C / "gemma-4-E2B")
        progs("E2B fp4/int8 4096 rows 8", s.image(cfg, 4096, 1, 8, "fp4", "int8", lookup=True))
    if have("gemma-4-E4B"):
        s = load_spec(C / "gemma-4-E4B")
        progs("E4B fp4/int8 2048 rows 8", s.image(cfg, 2048, 1, 8, "fp4", "int8", lookup=True))
    if have("LFM2.5-8B-A1B"):
        s = load_spec(C / "LFM2.5-8B-A1B")
        progs("LFM2.5-8B-A1B fp4/int8 2048 rows 1 x28",
              s.image(cfg, 2048, 1, 1, "fp4", "int8", lookup=True, experts=28), s.conv_k - 1)
    if have("Qwen3.5-35B-A3B"):
        s = load_spec(C / "Qwen3.5-35B-A3B")             # from_hf's: hints off, table on the host
        for name, kw in (("", {}), (", table on card", dict(embed_host=False))):
            progs(f"Qwen3.5-35B-A3B fp4/int8 1024 rows 1 x32{name}",
                  s.image(cfg, 1024, 1, 1, "fp4", "int8", lookup=True, experts=32, **kw),
                  s.conv_k - 1)
    if have("gemma-4-26B-A4B"):
        s = load_spec(C / "gemma-4-26B-A4B")
        progs("gemma-4-26B-A4B fp4/int8 4096 rows 1 x22",
              s.image(cfg, 4096, 1, 1, "fp4", "int8", lookup=True, experts=22), buckets=(1, 16))
        os.environ["OTPU_FORMATS"] = "experts=fp4"      # the card's: int8 layers, fp4 experts, head
        progs("gemma-4-26B-A4B int8/fp4 experts/fp4 head 4096 rows 1 x18",
              s.image(cfg, 4096, 1, 1, "int8", "fp4", lookup=True, experts=18), buckets=(1, 16))
        del os.environ["OTPU_FORMATS"]
    if not Path("tests/test_qwen35_moe.py").exists():
        return
    sys.path.insert(0, "tests")
    import test_lfm2_moe
    import test_qwen35_moe

    from opentpu.llm.qwen3 import Engine, device_config
    for mod in (test_lfm2_moe, test_qwen35_moe):
        _, W, spec = mod._tiny()
        dc = device_config(spec, 512, rows=1, lookup=True, S=1, experts=mod.K)
        if a.cfg:
            dc = dataclasses.replace(dc, PAIR=cfg.PAIR, DSTEP=cfg.DSTEP, STREAM=cfg.STREAM)
        e = Engine(spec, W, cap=512, cfg=dc, rows=1, resident=True, experts=mod.K)
        progs(f"tiny {mod.__name__}", e.image, getattr(spec, "conv_k", 1) - 1)


if __name__ == "__main__":
    main()
