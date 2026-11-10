"""The DeltaNet state stored in fewer bits (the bf16 state study, bf16st): formats_scan's Qwen3.5
emulation with each row's updated state rounded to a storage format before the next row reads it,
as a DSTEP whose DMA writes the state back rounded and reads it back expanded (fp32 inside the
engine) would run it, by KL against the float model over a text, paired with today's fp32 state.

Per row t and head (formats_scan._recur's step):
    S = S * e;  d = (v - S^T k) beta;  S = S + k d^T     (the engine: fp32, float64 here)
    o = S^T q                                            (from the engine's S, before the store)
    S = store(S)                                         (the DMA's write; the next row reads it)
Prefill runs the same step a row at a time (a DSTEP a row and head pair, the state through DRAM
every row), so the rows of the text see what decode's tokens would.

Modes (a variant's label ends in ";st=MODE"):
    fp32    fp32 round to nearest even (the device's own state: the emulation's float64 floor)
    bf16    bf16 (8 mantissa bits) round to nearest even
    bf16rz  bf16 truncated (round toward zero)
    fp24    fp32 with its low 8 mantissa bits rounded off (15 mantissa bits, nearest even)
    fp16    IEEE half (10 mantissa bits, nearest even): magnitudes past 65504 saturate to it,
            below 2^-14 (half's subnormals) flush to zero
    fp16s   IEEE half with its subnormals (numpy's: nearest even down to 2^-24), saturating
    eEmM    a 16-bit float of E exponent bits (bias 2^(E-1) - 1) and M = 15 - E mantissa bits,
            nearest even on the fp32 bits, saturating at its largest, flushing below 2^(2-2^(E-1))
            (e8m7 = bf16; e7m8, e6m9: what range buys back of bf16's precision)
fp32's subnormals flush to zero first (the VPU's). fp16 records the engine's state range per
layer (STATS: the largest magnitude, the nonzero fraction below 2^-14 and above 65504).

    python tools/dn_state.py check
    python tools/dn_state.py ppl MODEL --formats F [--wformat fp4] [--tokens 2000]
                                 [--modes bf16,fp24] [--fmodes fp32,bf16] [--out J]
        dKL of each mode against the formats' own fp32 state (dkl_prod, paired SE) and per
        position range; fmodes: the float model's state in each mode, its KL from float's
        (KL/float) and per position range: the format's own cost without the quantized
        model's chaos (a MoE's routing flips: any change of the state, fp24's too, moves the
        quantized model's tokens by about the noise of a second sample)
    python tools/dn_state.py greedy MODEL --formats F [--wformat fp4] [--modes bf16] [--n 8]
        greedy tokens of refs.py's short prompt (the chat template, thinking off) for float, the
        formats and each mode
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))      # (formats_scan)
import formats_scan as FS  # noqa: E402

D = 128
MODES = ("fp32", "bf16", "bf16rz", "fp24", "fp16", "fp16s", "e7m8", "e6m9", "e5m10")
STATS: dict = {}            # {label: [(max |S|, nonzero fraction < 2^-14, fraction > 65504)]}
RANGES = ((0, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000),
          (8000, 16000))
CUM = (2000, 4000, 8000, 16000)       # means over the first N tokens (a run of N's)


def store(S, mode: str):
    """S (float64) as the DRAM holds it in `mode` and the DMA expands it (module doc)."""
    x = np.asarray(S, np.float32)
    x = np.where(np.abs(x) < np.float32(2.0 ** -126), np.float32(0), x)
    if mode == "fp32":
        return x.astype(np.float64)
    if mode == "fp16":
        h = np.clip(x, -65504, 65504).astype(np.float16)
        return np.where(np.abs(x) < np.float32(2.0 ** -14), 0, h).astype(np.float64)
    if mode == "fp16s":
        return np.clip(x, -65504, 65504).astype(np.float16).astype(np.float64)
    if mode[:1] == "e":
        E, M = int(mode[1]), int(mode[3:])
        bias = (1 << (E - 1)) - 1
        top = (2 - 2.0 ** -M) * 2.0 ** ((1 << E) - 2 - bias)
        b = x.view(np.uint32).astype(np.uint64)
        drop = 23 - M
        b = (((b + ((1 << (drop - 1)) - 1) + ((b >> drop) & 1)) >> drop) << drop) & 0xFFFFFFFF
        r = b.astype(np.uint32).view(np.float32).astype(np.float64)
        r = np.clip(r, -top, top)
        return np.where(np.abs(x) < 2.0 ** (1 - bias), 0, r)
    b = x.view(np.uint32).astype(np.uint64)
    drop = {"bf16": 16, "bf16rz": 16, "fp24": 8}[mode]
    if mode != "bf16rz":
        b = b + ((1 << (drop - 1)) - 1) + ((b >> drop) & 1)
    b = (b >> drop) << drop
    return (b & 0xFFFFFFFF).astype(np.uint32).view(np.float32).astype(np.float64)


def recur_store(q, k, v, g, beta, mode: str):
    """formats_scan._recur with the state stored in `mode` after each row: o [T, h, dv]."""
    T, nh, dk = q.shape
    S, o = np.zeros((nh, dk, v.shape[2])), np.zeros((T, nh, v.shape[2]))
    mx, lo, hi, nz = 0.0, 0, 0, 0
    for t in range(T):
        S = S * np.exp(g[t])[:, None, None]
        delta = (v[t] - np.einsum("hij,hi->hj", S, k[t])) * beta[t][:, None]
        S = S + k[t][:, :, None] * delta[:, None, :]
        o[t] = np.einsum("hij,hi->hj", S, q[t])
        if mode in ("fp16", "e6m9"):
            a = np.abs(S)
            mx, nz = max(mx, float(a.max())), nz + int(np.count_nonzero(a))
            lo, hi = lo + int(np.count_nonzero((a > 0) & (a < 2.0 ** -14))), hi + int((a > 65504).sum())
        S = store(S, mode)
    if mode in ("fp16", "e6m9"):
        STATS.setdefault(FS._AT.get("lab"), []).append((mx, lo / max(nz, 1), hi / max(nz, 1)))
    return o


def _mode() -> str:
    return (FS._AT.get("lab") or "").partition(";st=")[2]


def _recur(q, k, v, g, beta):
    m = _mode()
    if not m:
        return FS._recur(q, k, v, g, beta)
    if m not in MODES:
        raise ValueError(f"dn_state: mode {m!r} (modes {', '.join(MODES)})")
    return recur_store(q, k, v, g, beta, m)


FS.RECUR[0] = _recur


def check() -> int:
    """store() against torch's bfloat16 (round to nearest even) and the recurrence's error on
    random gates (the 35B's head shape), with fp32's as the floor."""
    import torch
    rng = np.random.default_rng(5)
    x = np.concatenate([rng.standard_normal(100000) * 10.0 ** rng.uniform(-30, 30, 100000),
                        np.array([1 + 2 ** -8, 1 + 3 * 2 ** -8, -(1 + 2 ** -8), 2 ** -127, 0.0])])
    x32 = x.astype(np.float32)
    ref = torch.from_numpy(np.where(np.abs(x32) < 2.0 ** -126, 0, x32).astype(np.float32)).to(
        torch.bfloat16).to(torch.float32).numpy()
    ok = np.array_equal(store(x32, "bf16"), ref.astype(np.float64))
    print("store bf16 == torch bfloat16 (RNE):", ok)
    e87 = store(x32[np.abs(x32) < 1e38], "e8m7")
    ok &= np.array_equal(e87, store(x32[np.abs(x32) < 1e38], "bf16"))
    print("store e8m7 == bf16:", np.array_equal(e87, store(x32[np.abs(x32) < 1e38], "bf16")))
    e510 = store(x32, "e5m10")
    ok &= np.array_equal(e510, store(x32, "fp16"))
    print("store e5m10 == fp16:", np.array_equal(e510, store(x32, "fp16")))
    rz = store(x32, "bf16rz")
    ok &= bool(np.all(np.abs(rz) <= np.abs(x32)))
    f24 = store(x32, "fp24")
    nz = x32 != 0
    e24 = np.abs(f24[nz] - x32[nz]) / np.abs(x32[nz])
    print(f"fp24 max rel err {e24[np.abs(x32[nz]) > 2 ** -120].max():.3e} (2^-16 = {2 ** -16:.3e})")
    ok &= bool(e24[np.abs(x32[nz]) > 2 ** -120].max() <= 2 ** -16)
    T, nh, d = 400, 4, 128
    q = rng.standard_normal((T, nh, d))
    q /= np.linalg.norm(q, axis=-1, keepdims=True) * math.sqrt(d)
    k = rng.standard_normal((T, nh, d))
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    v = rng.standard_normal((T, nh, d))
    g = -np.exp(rng.uniform(-6, 0.5, (T, nh)))
    beta = 1 / (1 + np.exp(-rng.standard_normal((T, nh))))
    o0 = FS._recur(q, k, v, g, beta)
    for m in MODES:
        o = recur_store(q, k, v, g, beta, m)
        rel = np.linalg.norm(o - o0) / np.linalg.norm(o0)
        print(f"{m:7s} ||do||/||o|| {rel:.3e}")
    print("check", "ok" if ok else "FAILED")
    return 0 if ok else 1


def _ranges(dk) -> dict:
    out = {}
    for a, b in RANGES:
        if a >= len(dk):
            break
        s = dk[a:b]
        out[f"{a}-{a + len(s)}"] = (float(s.mean()),
                                    float(s.std(ddof=1) / math.sqrt(len(s))) if len(s) > 1 else 0.0)
    return out


def ppl(model, formats: str, wformat: str, n_tok: int, modes: list, out=None,
        fmodes: tuple = ()) -> None:
    path = FS.model_dir(model)
    spec, run, nbytes, _, _ = FS._open(path, D)
    ids = FS.text_ids(path, n_tok)
    t0 = time.time()
    specs = {"float": None, "int8": ("", "int8", None), "prod": (formats, wformat, None)}
    specs.update({f"prod;st={m}": (formats, wformat, None) for m in modes if m})
    specs.update({f"float;st={m}": None for m in fmodes if m})
    rows = run(specs, ids, lambda m: print(f"[{time.time() - t0:6.0f} s] {m}", flush=True))
    FS._finish(rows, nbytes)
    base = np.asarray(rows["prod"]["kl_tok"])
    for r in rows.values():
        dk = np.asarray(r["kl_tok"]) - base
        r["dkl_prod"], r["se_prod"] = float(dk.mean()), float(dk.std(ddof=1) / math.sqrt(len(dk)))
        r["dkl_prod_ranges"] = _ranges(dk)
        kf = np.asarray(r["kl_tok"])
        r["se_kl_float"] = float(kf.std(ddof=1) / math.sqrt(len(kf)))
        r["kl_float_ranges"] = _ranges(kf)
        r["kl_float_cum"] = {n: float(kf[:n].mean()) for n in CUM if n <= len(kf)}
    if out:
        Path(out).write_text(json.dumps({"model": str(model), "tokens": len(ids),
                                         "formats": formats, "wformat": wformat,
                                         "seconds": time.time() - t0, "variants": rows,
                                         "fp16_range": STATS}, indent=1))
    for lab, st in STATS.items():
        a = np.asarray(st)
        print(f"{lab}: state range by DN layer: max |S| {a[:, 0].max():.4g} (layers "
              f"{', '.join(f'{x:.3g}' for x in a[:, 0])}), nonzero < 2^-14 mean "
              f"{100 * a[:, 1].mean():.3f}% (max {100 * a[:, 1].max():.3f}%), > 65504 "
              f"{100 * a[:, 2].max():.4f}%", flush=True)
    for lab, r in rows.items():
        rg = "  ".join(f"{k} {100 * m:+.3f}" for k, (m, _) in r["dkl_prod_ranges"].items())
        if lab.startswith("float;"):
            rg = "  ".join(f"{k} {100 * m:.4f}" for k, (m, _) in r["kl_float_ranges"].items())
            print(f"{lab:16s} ppl {r['ppl']:8.4f}  KL/float {100 * r['kl_float']:.4f}% "
                  f"(+-{100 * r['se_kl_float']:.4f})  top1/float {r.get('agree_float', 0):.3f}"
                  f"  KL/float by position: {rg}  mean over the first N: " +
                  "  ".join(f"{n} {100 * m:.4f}" for n, m in r["kl_float_cum"].items()),
                  flush=True)
            continue
        print(f"{lab:16s} ppl {r['ppl']:8.4f}  KL/float {100 * r['kl_float']:.3f}%  dKL/int8 "
              f"{100 * r['dkl']:+.3f}% (+-{100 * r['se_dkl']:.3f})  vs prod "
              f"{100 * r['dkl_prod']:+.4f}% (+-{100 * r['se_prod']:.4f})  top1/int8 "
              f"{r['agree_int8']:.3f}  by position: {rg}", flush=True)


def greedy(model, formats: str, wformat: str, modes: list, n: int = 8) -> dict:
    """{label: n greedy tokens} of refs.py's short prompt (float, prod and each mode), the
    emulation over the whole sequence again for each token."""
    import transformers
    sys.path.insert(0, str(Path(__file__).resolve().parent / "qual"))
    import refs as R
    path = FS.model_dir(model)
    spec = FS.load_spec(path)
    W = FS.Q3.load_weights(path)
    M = FS.family(spec)
    ids = R.prompt_ids(transformers.AutoTokenizer.from_pretrained(path))
    labs = {"float": None, "prod": (formats, wformat, None)}
    labs.update({f"prod;st={m}": (formats, wformat, None) for m in modes})
    variants = {lab: None if v is None else FS._resolver(M, *v) for lab, v in labs.items()}
    seqs = {lab: list(ids) for lab in labs}
    t0 = time.time()
    for i in range(n):
        groups: dict = {}               # (one emulation a step for the variants of one sequence)
        for lab in labs:
            groups.setdefault(tuple(seqs[lab]), []).append(lab)
        for seq, ls in groups.items():
            lg = FS.emulate(spec, W, list(seq), {lab: variants[lab] for lab in ls}, D,
                            logits=True)
            for lab in ls:
                seqs[lab].append(int(np.argmax(lg[lab][-1])))
        print(f"[{time.time() - t0:6.0f} s] step {i + 1}: " +
              "  ".join(f"{lab} {s[len(ids):]}" for lab, s in seqs.items()), flush=True)
    return {lab: s[len(ids):] for lab, s in seqs.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    for c in ("ppl", "greedy"):
        a = sub.add_parser(c)
        a.add_argument("model")
        a.add_argument("--formats", required=True)
        a.add_argument("--wformat", default="fp4")
        a.add_argument("--modes", default="bf16,fp24")
        if c == "ppl":
            a.add_argument("--tokens", type=int, default=2000)
            a.add_argument("--fmodes", default="")
            a.add_argument("--out")
        else:
            a.add_argument("--n", type=int, default=8)
            a.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "check":
        return check()
    if a.cmd == "ppl":
        ppl(a.model, a.formats, a.wformat, a.tokens, a.modes.split(","), a.out,
            tuple(a.fmodes.split(",")))
        return 0
    res = greedy(a.model, a.formats, a.wformat, a.modes.split(","), a.n)
    print(json.dumps(res))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
