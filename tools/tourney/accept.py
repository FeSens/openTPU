"""Fitness and the accept rule. Pure functions, no I/O (tests: tests/test_tourney.py).

Area-equivalent (LUT-eq): one number for "how much of the xc7k480t this costs".

    area_eq = LUT + LUTRAM_LUT + 0.5 * FF + 40 * DSP + 80 * BRAM36 (+ 40 * BRAM18)

Rationale for the weights: the device has 298,600 LUTs, 597,200 FFs, 1,920 DSP48E1 and 955
BRAM36, so the "exchange rate" at full device is ~1 LUT = 2 FF, ~155 LUT per DSP, ~312 LUT
per BRAM36. The board build is LUT-bound (~70% LUTs, ~25% DSPs, ~65% BRAM), so DSPs and block
RAMs are cheaper than their device ratio: we charge them at about a quarter of it, which lets
a hypothesis move logic into DSPs / BRAM when that saves real LUTs, but not for free.

Estimated fmax from Yosys' logic-only arrival (no placement or routing):

    fmax_est = 1000 / (ROUTE * logic_ns + FIXED_NS)  MHz,  ROUTE = 1.6, FIXED_NS = 0.5

(routing typically adds 40-80% on 7-series; FIXED covers clock-to-out + setup). This ranks
candidates; it is not a signoff number (the vivado evaluator gives real post-route timing).

Accept rule (vs the champion; every gate must have passed):
  A. area:  fmax_new >= target  and  area_eq_new <= area_eq_old * (1 - AREA_GAIN)      (1%)
  B. speed: fmax_new >= fmax_old * (1 + FMAX_GAIN)  and  area_eq_new <= area_eq_old * (1 + AREA_SLACK)
            (3% faster, at most 1% bigger)
"""
from __future__ import annotations

W_FF, W_DSP, W_BRAM36, W_BRAM18 = 0.5, 40.0, 80.0, 40.0
ROUTE, FIXED_NS = 1.6, 0.5
AREA_GAIN, FMAX_GAIN, AREA_SLACK = 0.01, 0.03, 0.01


def area_eq(m: dict) -> float:
    """LUT-equivalent area of a synthesis result dict (keys: lut, lutram, ff, dsp, bram36,
    bram18; missing keys count 0)."""
    g = lambda k: float(m.get(k) or 0)
    return (g("lut") + g("lutram") + W_FF * g("ff") + W_DSP * g("dsp") +
            W_BRAM36 * g("bram36") + W_BRAM18 * g("bram18"))


def est_fmax(logic_ns: float | None) -> float:
    """Estimated fmax (MHz) from logic-only arrival (ns); inf for purely combinational-free."""
    if logic_ns is None or logic_ns <= 0:
        return float("inf")
    return 1000.0 / (ROUTE * logic_ns + FIXED_NS)


def combine(parts: list[tuple[dict, float]]) -> dict:
    """Weighted sum of several synthesis results (e.g. the fp operators times their
    instance counts); fmax is the minimum over the parts, logic_ns the maximum."""
    out: dict = {}
    for m, w in parts:
        for k in ("lut", "lutram", "ff", "dsp", "bram36", "bram18"):
            out[k] = out.get(k, 0) + w * float(m.get(k) or 0)
    lns = [m.get("logic_ns") for m, _ in parts if m.get("logic_ns") is not None]
    out["logic_ns"] = max(lns) if lns else None
    fm = [m["fmax"] for m, _ in parts if m.get("fmax") is not None]
    out["fmax"] = min(fm) if fm else est_fmax(out["logic_ns"])
    return out


def fitness(m: dict) -> dict:
    """Adds area_eq and fmax (if absent) to a synthesis result."""
    r = dict(m)
    r["area_eq"] = area_eq(m)
    if r.get("fmax") is None:
        r["fmax"] = est_fmax(m.get("logic_ns"))
    return r


def accept(old: dict, new: dict, target_mhz: float) -> tuple[bool, str]:
    """Decide whether `new` replaces the champion `old`. Both are fitness() dicts.
    Returns (accepted, reason)."""
    ao, an = old["area_eq"], new["area_eq"]
    fo, fn = old["fmax"], new["fmax"]
    da = (an - ao) / ao if ao else 0.0
    df = (fn - fo) / fo if fo and fo != float("inf") else 0.0
    if fn >= target_mhz and an <= ao * (1 - AREA_GAIN):
        return True, f"area {da:+.1%} at {fn:.0f} MHz (>= {target_mhz:.0f})"
    if fn >= fo * (1 + FMAX_GAIN) and an <= ao * (1 + AREA_SLACK):
        return True, f"fmax {df:+.1%} ({fo:.0f} -> {fn:.0f} MHz), area {da:+.1%}"
    why = []
    if fn < target_mhz:
        why.append(f"fmax {fn:.0f} < target {target_mhz:.0f}")
    why.append(f"area {da:+.1%} (need <= {-AREA_GAIN:.0%})")
    why.append(f"fmax {df:+.1%} (need >= {FMAX_GAIN:+.0%} with area <= {AREA_SLACK:+.0%})")
    return False, "; ".join(why)


def perf_ok(old_cycles: int | None, new_cycles: int | None, tol: float = 0.002) -> bool:
    """The performance proxy may not regress by more than `tol` (0.2%)."""
    if old_cycles is None or new_cycles is None:
        return True
    return new_cycles <= old_cycles * (1 + tol)
