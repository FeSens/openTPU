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
  C. pareto: fmax_new >= fmax_old  and  area_eq_new <= area_eq_old * (1 - AREA_GAIN)
            (smaller and no slower: below the target, an area win may not cost fmax, but it
            need not reach the target either)
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
    if fn >= fo and an <= ao * (1 - AREA_GAIN):
        return True, f"area {da:+.1%}, fmax {df:+.1%} ({fo:.0f} -> {fn:.0f} MHz, below target)"
    why = []
    if fn < target_mhz and fn < fo:
        why.append(f"fmax {fn:.0f} < target {target_mhz:.0f} and below the champion's {fo:.0f}")
    why.append(f"area {da:+.1%} (need <= {-AREA_GAIN:.0%})")
    why.append(f"fmax {df:+.1%} (need >= {FMAX_GAIN:+.0%} with area <= {AREA_SLACK:+.0%})")
    return False, "; ".join(why)


# ------------------------------------------------------------------------------ fmax objective
# The full-design tournament (--objective fmax) judges a candidate by the whole board built in
# Vivado at the tournament's clock: core_clk fmax = 1000 / (period - WNS).
FULL_GAIN, FULL_WNS_GAIN, OOC_GAIN, OOC_SLACK = 0.005, 0.05, 0.005, 0.01


def full_problems(new: dict) -> list[str]:
    """Why a full build cannot be accepted whatever its fmax (empty when it can)."""
    why = []
    if new.get("fmax") is None:
        why.append("no core_clk timing (build failed)")
    if new.get("whs") is not None and new["whs"] < 0:
        why.append(f"hold violated (WHS {new['whs']:+.3f} ns)")
    if new.get("congested"):
        why.append("router congestion (Route 35-447)")
    if new.get("collisions"):
        why.append(f"{new['collisions']} block RAM read-address collisions (Synth 8-6430)")
    return why


def accept_fmax(old: dict, new: dict) -> tuple[bool, str]:
    """The full-design rule: fmax +0.5% or better, or WNS +0.05 ns or better at the same clock,
    with hold met, no router congestion and no Synth 8-6430 memory. Area does not count here
    (it breaks ties between several winners, see rank_fmax)."""
    bad = full_problems(new)
    if bad:
        return False, "; ".join(bad)
    fo, fn = old["fmax"], new["fmax"]
    df = (fn - fo) / fo
    dw = (new["wns"] - old["wns"]) if (old.get("period") == new.get("period") and
                                       old.get("wns") is not None) else None
    msg = f"full fmax {fo:.2f} -> {fn:.2f} MHz ({df:+.2%})" + (
        f", WNS {old['wns']:+.3f} -> {new['wns']:+.3f} ns" if dw is not None else "")
    if df >= FULL_GAIN - 1e-12 or (dw is not None and dw >= FULL_WNS_GAIN - 1e-9):
        return True, msg
    return False, msg + f" (need >= {FULL_GAIN:+.1%} or WNS >= {FULL_WNS_GAIN:+.2f} ns)"


def ooc_promising(old: dict | None, new: dict, target_mhz: float) -> tuple[bool, str]:
    """Whether a component's OOC result earns a full build: its OOC fmax improves by >= 0.5%,
    or the component already clears the target on its own (so the gain must come from the paths
    between units, which OOC cannot see) and does not get more than 1% slower. No champion
    OOC number (e.g. the cross-unit component): always promising."""
    if new.get("collisions"):
        return False, f"{new['collisions']} Synth 8-6430 memories"
    if old is None or old.get("fmax") is None:
        return True, "no OOC baseline"
    fo, fn = old["fmax"], new["fmax"]
    if fn is None:
        return False, "no OOC timing"
    df = (fn - fo) / fo
    if df >= OOC_GAIN:
        return True, f"OOC fmax {fo:.1f} -> {fn:.1f} MHz ({df:+.1%})"
    if fo >= target_mhz and df >= -OOC_SLACK:
        return True, f"OOC fmax {fo:.1f} -> {fn:.1f} MHz ({df:+.1%}; unit already above target)"
    return False, f"OOC fmax {fo:.1f} -> {fn:.1f} MHz ({df:+.1%}; need >= {OOC_GAIN:+.1%})"


def rank_fmax(recs: list[dict]) -> list[dict]:
    """Accepted full-design candidates, best first: higher fmax, then smaller area."""
    return sorted(recs, key=lambda r: (-r["full"]["fmax"], area_eq(r["full"])))


def perf_ok(old_cycles: int | None, new_cycles: int | None, tol: float = 0.002) -> bool:
    """The performance proxy may not regress by more than `tol` (0.2%)."""
    if old_cycles is None or new_cycles is None:
        return True
    return new_cycles <= old_cycles * (1 + tol)
