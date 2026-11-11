#!/usr/bin/env python3
"""Prepare configuration-only S=4 decode cuts away from the card host.

A pack contains assembled programs and wait sidecars, sparse initial words, ISA results and
bounded-memory SHA256s. It never contains dense zero weights or multi-GiB zero low areas.
The card only copies bytes, runs, and checks results; compilation and the single ISA run
happen here. Packs are directories containing pack.json and arrays.npz (no Python pickle).
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np

CHUNK = 4 << 20
TARGET = "bcu1525-x2:st16"
# the two-board co-simulation's results (tools/x2_cosim.py); each cut's "point" is its cut directory and
# formats, relative to the working directory (cuts/NAME:WFORMAT:HEAD:rN)
RESULTS = Path(__file__).resolve().parents[1] / "docs/data/x2_cosim_results.json"


def sha256(data):
    return hashlib.sha256(memoryview(np.ascontiguousarray(data)).cast("B")).hexdigest()


def file_sha(path):
    """Streaming file digest, also on the project's Python 3.10 minimum."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def low_sha(read, low, check=lambda: None):
    """Hash logical slice bytes in address order, retaining at most one DMA chunk."""
    h = hashlib.sha256()
    for off in range(0, low, CHUNK):
        check()
        h.update(memoryview(read(off, min(CHUNK, low - off))).cast("B"))
    return h.hexdigest()


def image_spans(cfg, img):
    """The backend's image spans, including zero weights above the bank-half window.
    Low-only clearing is insufficient after a checkpoint run: old high weights are readable
    by this zero-weight workload too. Unused address holes and the program area are skipped."""
    from opentpu.llm.qwen3 import high_base
    if not getattr(img, "high", False):
        return [(0, int(img.nbytes))]
    hi = getattr(img, "wlo", high_base(cfg))
    skip = getattr(img, "wskip", None)
    spans = [(0, int(img.low_end)), (hi, int(img.nbytes))] if skip is None else [
        (0, int(img.low_end)), (hi, skip[0]), (skip[1], int(img.nbytes))]
    return [(int(a), int(b)) for a, b in spans if b > a]


def assembled(cfg, progs):
    from opentpu import isa as I, waits as W
    words = [np.asarray(I.assemble(p), dtype="<u4") for p in progs]
    waits = [W.entries(w, cfg.D, cfg.S, prog=p, aw=cfg.AW, win=cfg.WC_WIN)
             .astype("<u8") for w, p in zip(words, progs)]
    return words, waits


def fingerprint(cfg, img, progs, at, args, present):
    words, waits = assembled(cfg, progs)
    return {"program_sha256": [sha256(w) for w in words],
            "wait_sha256": [sha256(w) for w in waits], "program_at": int(at),
            "low_end": int(img.low_end), "image_bytes": int(img.nbytes),
            "zero_ranges": image_spans(cfg, img), "args": list(args or []),
            "present": [{"addr": int(a), "bytes": int(d.nbytes), "sha256": sha256(d)}
                        for a, d in present], "logits_addr": int(img.io["logits"]),
            "logits_words_per_slice": int(img.v_loc)}


def build_pack(path, target, cfg, img, progs, at, args, present, *, name, model,
               baseline, comparison=None):
    """The same path for production decode_case and the tiny offline regression cut."""
    from opentpu import target as T
    from opentpu.isasim import Machine
    from tools.x2_cosim import phase_blocks
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    words, waits = assembled(cfg, progs)
    ref = Machine(cfg, progs, [None] * cfg.S, args)
    for s in ref.slices:
        for addr, data in present:
            s.dram[addr:addr + len(data)] = data
    print(f"{name}: ISA start", flush=True)
    ref.run(max_steps=1 << 30)
    for key, got in (("instructions", [s.icount for s in ref.slices]),
                     ("collectives_mod65536", ref.collectives & 0xFFFF),
                     ("checked_low_bytes_per_slice", int(img.low_end))):
        if key in baseline and got != baseline[key]:
            raise ValueError(f"{name}: ISA {key} {got} != co-sim {baseline[key]}")
    arrays = {}
    io, vloc = int(img.io["logits"]), int(img.v_loc)
    for k, (w, e, s) in enumerate(zip(words, waits, ref.slices)):
        arrays[f"program_{k}"] = w
        arrays[f"waits_{k}"] = e
        arrays[f"logits_{k}"] = s.dram[io + 4 * k * vloc:io + 4 * (k + 1) * vloc].view("<u4").copy()
    for k, (_, data) in enumerate(present):
        arrays[f"present_{k}"] = np.asarray(data, np.uint8)
    logits = np.concatenate([arrays[f"logits_{k}"] for k in range(cfg.S)])
    machine = T.config(target)
    meta = {"format": "otpu-x2-cutpack-v1", "name": name, "model": model,
            "target": TARGET, "config": asdict(cfg), "machine_hash": T.arch_hash(machine),
            "card_hash": T.arch_hash(replace(machine, S=machine.S // 2)),
            "slices": cfg.S, "program_at": int(at), "args": [int(v) for v in (args or [])],
            "low_end": int(img.low_end), "image_bytes": int(img.nbytes),
            "zero_ranges": image_spans(cfg, img),
            "present": [{"addr": int(a), "array": f"present_{k}"}
                        for k, (a, _) in enumerate(present)],
            "instructions": [s.icount for s in ref.slices],
            "collectives_mod65536": ref.collectives & 0xFFFF,
            "logits_addr": io, "logits_words_per_slice": vloc,
            "argmax": int(np.argmax(logits.view("<f4"))),
            "low_sha256": [low_sha(lambda a, n, s=s: s.dram[a:a + n], img.low_end)
                           for s in ref.slices],
            "pc_phases": phase_blocks(progs), "cosim": baseline,
            "comparison": comparison, "array_sha256": {k: sha256(v) for k, v in arrays.items()}}
    np.savez_compressed(path / "arrays.npz", **arrays)
    meta["arrays_file_sha256"] = file_sha(path / "arrays.npz")
    meta["arrays_bytes"] = (path / "arrays.npz").stat().st_size
    (path / "pack.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{name}: ISA instructions={meta['instructions']}, collectives={ref.collectives}; "
          f"pack bytes={sum(p.stat().st_size for p in path.iterdir())}", flush=True)
    return meta


def load_pack(path):
    """Read-only, bounded-size artifacts. Never reconstruct or run a compiler/ISA model."""
    path = Path(path)
    meta = json.loads((path / "pack.json").read_text())
    if meta["format"] != "otpu-x2-cutpack-v1" or meta["slices"] != 4:
        raise ValueError("unsupported cut pack")
    if file_sha(path / "arrays.npz") != meta["arrays_file_sha256"]:
        raise ValueError("cut pack arrays sha256 mismatch")
    with np.load(path / "arrays.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    if set(arrays) != set(meta["array_sha256"]) or any(
            sha256(v) != meta["array_sha256"][k] for k, v in arrays.items()):
        raise ValueError("cut pack array sha256 mismatch")
    return meta, arrays


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", default="q36,q38,goss")
    ap.add_argument("--cosim-results", type=Path, default=RESULTS)
    ap.add_argument("--st16-results", type=Path, help="directory of MODEL.json from --target st16 co-sim")
    a = ap.parse_args(argv)
    from tools.x2_card import quiet_environment
    quiet_environment()
    from opentpu import target as T
    from tools.board_perf import decode_case, present_words
    cuts = {c["model"]: c for c in json.loads(a.cosim_results.read_text())["cuts"]}
    target = T.load(TARGET)
    for model in a.models.split(","):
        old = cuts[model]
        point = old["point"]
        cfg, img, progs, at, args = decode_case(T.config(target), point)
        present = present_words(img)
        current = fingerprint(cfg, img, progs, at, args, present)
        oc, oi, op, oa, og = decode_case(T.config(T.load("bcu1525-x2")), point)
        prior = fingerprint(oc, oi, op, oa, og, present_words(oi))
        different = [k for k in current if current[k] != prior[k]]
        baseline = old
        if different:
            f = a.st16_results / f"{model}.json" if a.st16_results else None
            if f is None or not f.is_file():
                raise ValueError(f"{model}: st16 differs in {different}; run x2_cosim.py "
                                 f"--target {TARGET} and supply --st16-results DIR")
            baseline = json.loads(f.read_text())
            if not baseline.get("ok") or baseline.get("target") != TARGET or \
                    baseline.get("point") != point:
                raise ValueError(f"{model}: invalid st16 co-sim result")
        comparison = {"original_cycles": old["cycles"], "changed_fields": different,
                      "original_low_end": prior["low_end"], "st16_low_end": current["low_end"],
                      "cosim_target": TARGET if different else "bcu1525-x2",
                      "cycle_delta_percent": 100 * (baseline["cycles"] / old["cycles"] - 1),
                      "fingerprint": current}
        name = {"q36": "q36-4", "q38": "q38-4", "goss": "goss-2"}[model]
        build_pack(a.out / name, target, cfg, img, progs, at, args, present,
                   name=name, model=model, baseline=baseline, comparison=comparison)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
