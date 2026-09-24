"""Per-component architecture tournament (docs/tourney.md).

    python3 -m tools.tourney.orchestrator --comp otpu_coll --rounds 1 --slots 1 [--agent claude]
        [--eval yosys|vivado] [--base main] [--reset] [--keep] [--no-scribe] [--baseline-only]

Each component evolves on its own champion branch `tourney/<comp>` (created from --base, i.e.
main, the first time; --reset recreates it). A round runs K slots in parallel, each in its own
git worktree off the champion: a hypothesis agent writes HYPOTHESIS.md, an implementation
agent edits only the component's files, then the gates run (tools/tourney/gates.py). The best
accepted slot is committed and fast-forwarded onto the champion branch. Every slot is logged to
tools/tourney/runs/<comp>/log.jsonl and leaves a lesson in LESSONS.md. The orchestrator never
touches main and never pushes.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from . import accept as A
from . import agents as AG
from . import gates as G

HERE = Path(__file__).resolve().parent
os.environ["GIT_CONFIG_PARAMETERS"] = (os.environ.get("GIT_CONFIG_PARAMETERS", "") +
                                       " 'commit.gpgsign=false' 'tag.gpgsign=false'").strip()
_LOCK = threading.Lock()


def git(*args, cwd: Path, check: bool = True) -> str:
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout.strip()


def load_component(name: str) -> dict:
    p = HERE / "components" / f"{name}.yaml"
    if not p.exists():
        names = sorted(q.stem for q in (HERE / "components").glob("*.yaml"))
        raise SystemExit(f"unknown component {name!r}; one of {names}")
    return yaml.safe_load(p.read_text())


class Run:
    def __init__(self, a):
        self.a = a
        self.repo = Path(git("rev-parse", "--show-toplevel", cwd=Path.cwd()))
        self.comp = load_component(a.comp)
        self.branch = f"tourney/{a.comp}"
        self.dir = self.repo / "tools" / "tourney" / "runs" / a.comp
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log = self.dir / "log.jsonl"
        self.lessons = self.dir / "LESSONS.md"
        self.wtroot = self.repo / ".tourney" / a.comp
        self.shared_build = self.repo / "build"

    # ---- champion
    def ensure_branch(self) -> str:
        exists = git("rev-parse", "--verify", "--quiet", self.branch, cwd=self.repo,
                     check=False)
        if exists and not self.a.reset:
            return exists
        base = git("rev-parse", self.a.base, cwd=self.repo)
        git("branch", "-f", self.branch, base, cwd=self.repo)
        print(f"[tourney] champion branch {self.branch} <- {self.a.base} ({base[:9]})")
        return base

    def sync_base(self) -> None:
        """Merge the base branch into the champion when it has moved (other components'
        verified winners land there), so candidates are built and tested against the current
        design. A conflicting merge is abandoned: the round runs on the old champion."""
        behind = git("rev-list", "--count", f"{self.branch}..{self.a.base}", cwd=self.repo)
        if behind == "0":
            return
        wt = self.worktree("sync", self.branch, detach=True)
        try:
            r = subprocess.run(["git", "merge", "--no-edit", "-q", self.a.base], cwd=wt,
                               capture_output=True, text=True)
            if r.returncode != 0:
                git("merge", "--abort", cwd=wt, check=False)
                print(f"[tourney] sync: {self.a.base} does not merge into {self.branch} "
                      f"cleanly; staying on the old champion", flush=True)
                return
            try:                                   # a clean text merge can still be broken
                G.pytest(wt, self.comp["tests"]["fast"], None, "fast")
            except G.GateFailure as e:
                print(f"[tourney] sync: the merged champion fails its fast tests ({e.tail[-200:]}); "
                      f"staying on the old champion", flush=True)
                return
            sha = git("rev-parse", "HEAD", cwd=wt)
            git("branch", "-f", self.branch, sha, cwd=self.repo)
            print(f"[tourney] sync: merged {behind} commits of {self.a.base} -> "
                  f"{self.branch} {sha[:9]}", flush=True)
        finally:
            self.drop(wt, None)

    def champion(self) -> dict:
        """The champion's metrics, measured once per champion commit (cached)."""
        sha = git("rev-parse", self.branch, cwd=self.repo)
        cache = self.dir / "champion.json"
        if cache.exists():
            c = json.loads(cache.read_text())
            if c.get("sha") == sha and c.get("backend") == self.a.eval:
                return c
        print(f"[tourney] measuring champion {sha[:9]} ({self.a.eval})", flush=True)
        wt = self.worktree("champion", sha, detach=True)
        try:
            m = G.synthesize(wt, self.comp, self.a.eval,
                             self.shared_build / "tourney" / self.a.comp / f"champ-{sha[:9]}")
            m["perf_cycles"] = G.perf(wt) if self.comp.get("perf") else None
        finally:
            self.drop(wt, None)
        m["sha"] = sha
        m["critical"] = self.critical(m)
        cache.write_text(json.dumps(m, indent=1))
        return m

    def critical(self, m: dict, n: int = 24) -> str:
        """The deepest cells of the critical path from the Yosys sta report(s)."""
        out = []
        for part in self.comp["synth"]["parts"]:
            sta = self.shared_build / "tourney" / self.a.comp
            cands = sorted(sta.glob(f"*/{part['top']}/sta.txt"), key=lambda p: p.stat().st_mtime)
            if not cands:
                continue
            lines = cands[-1].read_text().splitlines()
            i = next((k for k, l in enumerate(lines) if "Latest arrival" in l), None)
            if i is not None:
                out.append(f"[{part['top']}]")
                out += [l for l in lines[i:i + 2 * n] if l.strip()]
        return "\n".join(out) or "(not available)"

    # ---- worktrees
    def slot_branch(self, sid: str) -> str:
        # not under tourney/<comp>/: a ref cannot be both a branch and a directory
        return f"tourney-slot/{self.a.comp}/{sid}"

    def worktree(self, name: str, ref: str, detach: bool = False) -> Path:
        wt = self.wtroot / name
        if wt.exists():
            self.drop(wt, None)
        wt.parent.mkdir(parents=True, exist_ok=True)
        if detach:
            git("worktree", "add", "--detach", str(wt), ref, cwd=self.repo)
        else:
            git("worktree", "add", "-B", self.slot_branch(name), str(wt), ref, cwd=self.repo)
        # share the Verilator build cache (keyed by source hash) and the model weights
        self.shared_build.mkdir(exist_ok=True)
        (wt / "build").symlink_to(self.shared_build)
        m = G.models_dir(self.repo)
        if m is not None and not (wt / "models").exists():
            (wt / "models").symlink_to(m.parent)
        return wt

    def drop(self, wt: Path, branch: str | None) -> None:
        if self.a.keep and branch:
            return
        git("worktree", "remove", "--force", str(wt), cwd=self.repo, check=False)
        shutil.rmtree(wt, ignore_errors=True)
        git("worktree", "prune", cwd=self.repo, check=False)
        if branch:
            git("branch", "-D", branch, cwd=self.repo, check=False)

    # ---- log
    def history(self, n: int = 8) -> str:
        if not self.log.exists():
            return "(none yet)"
        rows = [json.loads(l) for l in self.log.read_text().splitlines() if l.strip()][-n:]
        return "\n".join(f"- {r['id']}: {r['outcome']} -- {r.get('title', '')[:70]} "
                         f"({r.get('reason', '')[:90]})" for r in rows) or "(none yet)"

    def append(self, rec: dict) -> None:
        with _LOCK:
            with self.log.open("a") as f:
                f.write(json.dumps(rec) + "\n")

    def lesson(self, line: str) -> None:
        with _LOCK:
            if not self.lessons.exists():
                self.lessons.write_text(f"# Lessons: {self.a.comp}\n\n")
            with self.lessons.open("a") as f:
                f.write(f"- {line.strip()}\n")

    # ---- one slot
    def slot(self, rid: str, k: int, champ: dict) -> dict:
        sid = f"{rid}-s{k}"
        rec = {"id": sid, "comp": self.a.comp, "agent": self.a.agent, "eval": self.a.eval,
               "champion": champ["sha"], "start": dt.datetime.now().isoformat(timespec="seconds")}
        rec["slot"] = k
        rec["models"] = {r: AG.model_for(r, k, self.a.agent) or "default" for r in AG.ROLES}
        rec["efforts"] = {r: AG.effort_for(r, k) for r in AG.ROLES}
        rec["roles"] = {}
        wt = self.worktree(sid, self.branch)
        logs = self.dir / "agents"
        logs.mkdir(exist_ok=True)
        t0 = time.time()

        def agent(role: str, prompt: str) -> dict:
            try:
                info = AG.run(self.a.agent, prompt, wt, logs / f"{sid}.{role}.jsonl",
                              self.a.agent_timeout, f"{sid} {role}",
                              AG.model_for(role, k, self.a.agent), AG.effort_for(role, k))
            except AG.AgentError as e:
                rec["roles"][role] = {kk: v for kk, v in e.info.items() if kk != "text"}
                raise
            rec["roles"][role] = {kk: v for kk, v in info.items() if kk != "text"}
            return info

        try:
            lessons = self.lessons.read_text() if self.lessons.exists() else "(none yet)"
            cat = AG.CATEGORIES[k % len(AG.CATEGORIES)]
            rec["category"] = cat.split(":")[0]
            agent("hyp", AG.hypothesis_prompt(self.comp, wt, champ, lessons, self.history(),
                                              cat, champ["critical"]))
            hyp = (wt / "HYPOTHESIS.md").read_text() if (wt / "HYPOTHESIS.md").exists() else ""
            if not hyp.strip():
                raise G.GateFailure("hypothesis", "no HYPOTHESIS.md written")
            rec["title"] = hyp.strip().splitlines()[0].lstrip("# ").strip()
            rec["hypothesis"] = hyp
            if G.changed_paths(wt) != ["HYPOTHESIS.md"]:
                raise G.GateFailure("sandbox", f"hypothesis phase touched {G.changed_paths(wt)}")
            agent("impl", AG.implement_prompt(self.comp, wt))
            rec["implementation"] = ((wt / "IMPLEMENTATION.md").read_text()
                                     if (wt / "IMPLEMENTATION.md").exists() else "")
            # ---- gates
            gs = rec["gate_seconds"] = {}

            def gate(name, f, *args):
                t = time.time()
                try:
                    return f(*args)
                finally:
                    gs[name] = round(time.time() - t, 1)

            rec["files"] = gate("sandbox", G.sandbox, wt, self.comp["allowed"])
            rec["diff"] = git("diff", "--stat", cwd=wt)
            pdir = self.dir / "patches"                # the full change, kept for every outcome
            pdir.mkdir(exist_ok=True)
            (pdir / f"{sid}.patch").write_text(git("diff", cwd=wt) + "\n")
            gate("lint", G.lint, wt)
            rec["fast"] = gate("fast", G.pytest, wt, self.comp["tests"]["fast"], None, "fast")
            rec["board"] = gate("board", G.pytest, wt, self.comp["tests"]["board"], G.BOARD_ENV,
                                "board")
            if self.comp.get("perf"):
                cyc = gate("perf", G.perf, wt)
                rec["perf_cycles"] = cyc
                if not A.perf_ok(champ.get("perf_cycles"), cyc):
                    raise G.GateFailure("perf", f"{cyc} cycles vs champion "
                                                f"{champ.get('perf_cycles')} (> +0.2%)")
            m = gate("synth", G.synthesize, wt, self.comp, self.a.eval,
                     self.shared_build / "tourney" / self.a.comp / sid)
            rec["metrics"] = {q: m.get(q) for q in ("lut", "lutram", "ff", "dsp", "bram36",
                                                     "bram18", "logic_ns", "fmax", "area_eq")}
            ok, why = A.accept(champ, m, self.comp["target_mhz"])
            rec["outcome"], rec["reason"] = ("improvement" if ok else "no_gain"), why
            rec["gain"] = (champ["area_eq"] - m["area_eq"]) / champ["area_eq"] + \
                          (m["fmax"] - champ["fmax"]) / champ["fmax"]
        except G.GateFailure as e:
            rec["outcome"], rec["reason"] = "broken", f"{e.gate}: {e.tail}"
        except Exception as e:  # noqa: BLE001 -- agent / tool crashes are logged, not fatal
            rec["outcome"], rec["reason"] = "error", f"{type(e).__name__}: {str(e)[-600:]}"
        rec["seconds"] = round(time.time() - t0)
        rec["wt"] = str(wt)
        print(f"[tourney] {sid}: {rec['outcome']} -- {rec.get('reason', '')[:160]}", flush=True)
        return rec

    # ---- rounds
    def round(self, r: int) -> None:
        self.sync_base()
        champ = self.champion()
        print(f"[tourney] round {r}: champion {champ['sha'][:9]} area_eq {champ['area_eq']:.0f} "
              f"fmax {champ['fmax']:.0f} MHz perf {champ.get('perf_cycles')}", flush=True)
        rid = f"r{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        with ThreadPoolExecutor(self.a.slots) as ex:
            recs = list(ex.map(lambda k: self.slot(rid, k, champ), range(self.a.slots)))
        winners = sorted((x for x in recs if x["outcome"] == "improvement"),
                         key=lambda x: -x["gain"])
        if winners:
            w = winners[0]
            wt = Path(w["wt"])
            git("add", "--", *w["files"], cwd=wt)
            git("commit", "-q", "-m", f"tourney {self.a.comp}: {w.get('title', w['id'])}\n\n"
                f"{w['reason']}\n\nSlot {w['id']}, agent {self.a.agent}, eval {self.a.eval}.",
                cwd=wt)
            sha = git("rev-parse", "HEAD", cwd=wt)
            git("update-ref", f"refs/heads/{self.branch}", sha, champ["sha"], cwd=self.repo)
            w["outcome"], w["merged"] = "accepted", sha
            print(f"[tourney] accepted {w['id']} -> {self.branch} {sha[:9]}: {w['reason']}")
        for x in recs:
            if self.a.scribe and x.get("hypothesis"):
                try:
                    info = AG.run(self.a.agent, AG.scribe_prompt(
                        self.a.comp, x["hypothesis"], x["outcome"], x.get("reason", "")),
                        self.dir, self.dir / "agents" / f"{x['id']}.scribe.jsonl", 600,
                        f"{x['id']} scribe", AG.model_for("scribe", x["slot"], self.a.agent),
                        AG.effort_for("scribe", x["slot"]))
                    x["roles"]["scribe"] = {k: v for k, v in info.items() if k != "text"}
                    line = info["text"].strip().splitlines()
                    line = line[-1] if line else ""
                except Exception:  # noqa: BLE001 -- fall back to the template line
                    line = ""
            else:
                line = ""
            if not line:
                line = f"{x.get('title', x['id'])}: {x['outcome']} ({x.get('reason', '')[:120]})"
            x["lesson"] = line[:300]
            costs = [r["usage"].get("cost_usd") for r in x["roles"].values()]
            x["cost_usd"] = round(sum(c for c in costs if c), 4) if any(costs) else None
            x["end"] = dt.datetime.now().isoformat(timespec="seconds")
            self.lesson(f"[{x['id']}] {x['lesson']}")
            self.append({k: v for k, v in x.items() if k != "wt"})
            self.drop(Path(x["wt"]), self.slot_branch(x["id"]))

    def main(self) -> None:
        self.ensure_branch()
        champ = self.champion()
        print(f"[tourney] {self.a.comp} champion: " + json.dumps(
            {k: champ.get(k) for k in ("lut", "lutram", "ff", "dsp", "bram36", "logic_ns",
                                        "fmax", "area_eq", "perf_cycles")}))
        if self.a.baseline_only:
            return
        for r in range(self.a.rounds):
            self.round(r)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--comp", required=True)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--slots", type=int, default=1)
    ap.add_argument("--agent", default=os.environ.get("AGENT", "claude"))
    ap.add_argument("--eval", default=os.environ.get("EVAL", "yosys"))
    ap.add_argument("--base", default="main")
    ap.add_argument("--reset", action="store_true", help="recreate the champion from --base")
    ap.add_argument("--keep", action="store_true", help="keep slot worktrees and branches")
    ap.add_argument("--no-scribe", dest="scribe", action="store_false")
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--agent-timeout", type=int, default=3600)
    Run(ap.parse_args(argv)).main()


if __name__ == "__main__":
    sys.exit(main())
