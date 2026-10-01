"""A MoE checkpoint without its routed experts, for the card's host (docs/offload.md 10.4, 11.4):
the card run reads the experts from the packed pool file only (pack_pool.py), so the host keeps
every other tensor, in its own dtype, in one safetensors file, with the checkpoint's other files.

    python3 tools/offload/strip_experts.py SRC DST                   # Qwen3.5-35B-A3B: 4.6 GiB
    python3 tools/offload/strip_experts.py SRC DST --only model.language_model.   # Gemma 4 26B

Dropped: the routed experts (`.experts.`; a shared expert is `shared_expert`), the vision tower
and the MTP head (`model.visual.`, `mtp.`: LazyWeights skips them), and with --only every tensor
outside that prefix (the 26B's vision and audio towers).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--only", help="keep only the tensors under this prefix")
    a = ap.parse_args()
    from safetensors import safe_open
    from safetensors.torch import save_file

    src, dst = Path(a.src), Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and f.suffix != ".safetensors" and f.name != "model.safetensors.index.json":
            shutil.copy2(f, dst / f.name)
    keep, skipped, nb = {}, 0, 0
    for f in sorted(src.glob("*.safetensors")):
        with safe_open(str(f), "pt") as h:
            for k in h.keys():
                if (".experts." in k or k.startswith(("model.visual.", "mtp."))
                        or (a.only and not k.startswith(a.only))):
                    skipped += 1
                    continue
                t = h.get_tensor(k)
                keep[k] = t.contiguous()
                nb += t.numel() * t.element_size()
    save_file(keep, str(dst / "model.safetensors"), metadata={"format": "pt"})
    print(f"kept {len(keep)} tensors ({nb / 2**30:.2f} GiB), skipped {skipped}")


if __name__ == "__main__":
    main()
