"""Tournament report: per-slot table, acceptance and cost per model, progress plot.

    python3 -m tools.tourney.report [--comp otpu_coll ...]   (default: every component with a log)

Writes tools/tourney/runs/<comp>/REPORT.md and progress.png, and prints the tables.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "runs"


def load(comp: str) -> tuple[list[dict], dict | None]:
    d = RUNS / comp
    log = d / "log.jsonl"
    rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()] if log.exists() else []
    ch = d / "champion.json"
    return rows, (json.loads(ch.read_text()) if ch.exists() else None)


def _f(v, fmt="{:.0f}"):
    return "-" if v is None else fmt.format(v)


def slot_table(rows: list[dict]) -> str:
    out = ["| slot | outcome | title | area_eq | fmax | impl model @effort | cost $ | min | reason |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        m = r.get("metrics") or {}
        md, ef = r.get("models") or {}, r.get("efforts") or {}
        out.append(f"| {r['id']} | {r['outcome']} | {r.get('title', '')[:50]} | "
                   f"{_f(m.get('area_eq'))} | {_f(m.get('fmax'))} | "
                   f"{md.get('impl', '-')} @{ef.get('impl', '-')} | "
                   f"{_f(r.get('cost_usd'), '{:.2f}')} | "
                   f"{_f((r.get('seconds') or 0) / 60, '{:.0f}')} | "
                   f"{r.get('reason', '')[:80].replace('|', '/')} |")
    return "\n".join(out)


def model_summary(rows: list[dict]) -> dict:
    """Per implementation model and effort (the role that decides success): slots, accepted, broken,
    agent cost and wall time per role."""
    s: dict = defaultdict(lambda: {"slots": 0, "accepted": 0, "improvement": 0, "broken": 0,
                                   "no_gain": 0, "error": 0, "cost_usd": 0.0, "costed": 0,
                                   "role_seconds": defaultdict(float)})
    for r in rows:
        key = (r.get("models") or {}).get("impl", "default")
        eff = (r.get("efforts") or {}).get("impl")
        key += f" @{eff}" if eff else ""
        e = s[key]
        e["slots"] += 1
        e[r["outcome"]] = e.get(r["outcome"], 0) + 1
        if r.get("cost_usd") is not None:
            e["cost_usd"] += r["cost_usd"]
            e["costed"] += 1
        for role, info in (r.get("roles") or {}).items():
            e["role_seconds"][role] += info.get("seconds", 0)
    return s


def model_table(rows: list[dict]) -> str:
    out = ["| impl model @effort | slots | accepted | no gain | broken/error | accept rate | cost $ | "
           "$ / slot | $ / accept | agent min (hyp/impl/scribe) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for m, e in sorted(model_summary(rows).items()):
        acc = e["accepted"] + e["improvement"]
        rs = e["role_seconds"]
        out.append(f"| {m} | {e['slots']} | {acc} | {e['no_gain']} | {e['broken'] + e['error']} | "
                   f"{100 * acc / e['slots']:.0f}% | {e['cost_usd']:.2f} | "
                   f"{e['cost_usd'] / max(e['costed'], 1):.2f} | "
                   f"{(e['cost_usd'] / acc) if acc else float('nan'):.2f} | "
                   f"{rs['hyp'] / 60:.0f}/{rs['impl'] / 60:.0f}/{rs['scribe'] / 60:.1f} |")
    return "\n".join(out)


def plot(comp: str, rows: list[dict], champ: dict | None, path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    colors = {"accepted": "tab:green", "improvement": "tab:olive", "no_gain": "tab:gray",
              "broken": "tab:red", "error": "black"}
    best_a, best_f, xs = [], [], []
    for i, r in enumerate(rows):
        m = r.get("metrics") or {}
        if m.get("area_eq") is not None:
            a1.scatter(i, m["area_eq"], c=colors.get(r["outcome"], "gray"), s=18)
            a2.scatter(i, m["fmax"], c=colors.get(r["outcome"], "gray"), s=18)
        if r["outcome"] == "accepted":
            best_a.append(m["area_eq"]), best_f.append(m["fmax"]), xs.append(i)
    if champ is not None:
        a1.axhline(champ["area_eq"], color="tab:green", lw=0.8, ls="--", label="champion")
        a2.axhline(champ["fmax"], color="tab:green", lw=0.8, ls="--")
    if xs:
        a1.step(xs, best_a, where="post", color="tab:green")
        a2.step(xs, best_f, where="post", color="tab:green")
    a1.set_ylabel("area-equivalent")
    a2.set_ylabel("est fmax (MHz)")
    a2.set_xlabel("slot (chronological)")
    a1.set_title(f"{comp}: green accepted, grey no gain (red/black broken: no metrics)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def report(comp: str) -> str:
    rows, champ = load(comp)
    d = RUNS / comp
    parts = [f"# Tourney report: {comp}", ""]
    if champ:
        parts.append(f"Champion {champ['sha'][:9]}: LUT {champ.get('lut')}, LUTRAM "
                     f"{champ.get('lutram')}, FF {champ.get('ff')}, DSP {champ.get('dsp')}, BRAM36 "
                     f"{champ.get('bram36')}, logic {champ.get('logic_ns')} ns, est fmax "
                     f"{champ.get('fmax', 0):.1f} MHz, area-eq {champ.get('area_eq', 0):.0f}, "
                     f"perf proxy {champ.get('perf_cycles')} cycles ({champ.get('backend')}).")
        parts.append("")
    parts += ["## Slots", "", slot_table(rows), "", "## Per implementation model", "",
              model_table(rows), ""]
    if plot(comp, rows, champ, d / "progress.png"):
        parts += ["![progress](progress.png)", ""]
    txt = "\n".join(parts)
    (d / "REPORT.md").write_text(txt)
    return txt


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", nargs="*")
    a = ap.parse_args(argv)
    comps = a.comp or sorted(p.name for p in RUNS.glob("*") if (p / "log.jsonl").exists())
    for c in comps:
        print(report(c))
        print()


if __name__ == "__main__":
    main()
