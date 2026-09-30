"""Per-component architecture tournament (docs/tourney.md).

    python3 -m tools.tourney.orchestrator --comp otpu_coll --rounds 1 --slots 1 [--agent claude]
        [--eval yosys|vivado-remote] [--base main] [--reset] [--keep] [--no-scribe] [--baseline-only]
        [--objective area|fmax|unit] [--target-mhz 133.33]

Each component evolves on its own champion branch `tourney/<comp>` (created from --base, i.e.
main, the first time; --reset recreates it). A round runs K slots in parallel, each in its own
git worktree off the champion: a hypothesis agent writes HYPOTHESIS.md, an implementation
agent edits only the component's files, then the gates run (tools/tourney/gates.py). The best
accepted slot is committed and fast-forwarded onto the champion branch. Every slot is logged to
tools/tourney/runs/<comp>/log.jsonl and leaves a lesson in LESSONS.md. The orchestrator never
touches main and never pushes.

--objective fmax (docs/tourney.md, the fmax tournament): every component evolves ONE shared
champion, `tourney/fmax`, scored by the whole board built in Vivado on the build host at
--target-mhz. A slot that passes the correctness gates runs its component out of context in
Vivado (tools/tourney/remote.py); the most promising slot of the round (at most one) gets the full
build, and the accept rule is accept.accept_fmax. Logs in tools/tourney/runs/fmax/<comp>/, full
build results cached per commit in tools/tourney/runs/_full/.

--objective unit (docs/tourney.md, the unit tournament): each component evolves its own champion,
`tourney/unit/<comp>`, scored by the component alone out of context in Vivado on the build host
at --target-mhz (no full builds); the accept rule is accept.accept_unit (area at no fmax
regression, or speed). Logs in tools/tourney/runs/unit/<comp>/, winners in runs/unit/WINNERS.jsonl.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
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
# BUILD_ID of every tournament full build: one constant, so the champion and the candidates differ
# only in their RTL (the register's value is part of the netlist). These bitstreams are for timing
# numbers, not for the card.
STOP = Path(os.environ.get("OTPU_TOURNEY_STOP", "/tmp/otpu-tourney-stop"))
PAUSE = Path(os.environ.get("OTPU_TOURNEY_PAUSE", "/tmp/otpu-tourney-pause"))
FULL_BUILD_ID = "0f3a0000"
# the confirmation build of a winner: the same tree with another BUILD_ID constant, i.e. another
# netlist hash and so another placement (Vivado has no placer seed): the second draw of
# place-and-route noise, which must pass the accept rule too
CONFIRM_BUILD_ID = "0f3a0001"


def git(*args, cwd: Path, check: bool = True) -> str:
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout.strip()


class _Done(Exception):
    """A slot's gates finished early with an outcome already set (fmax objective)."""


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
        self.fmax = a.objective == "fmax"
        self.unit = a.objective == "unit"
        if a.objective not in self.comp.get("objectives", ["area", "fmax", "unit"]):
            raise SystemExit(f"{a.comp} has no {a.objective} objective")
        runs = self.repo / "tools" / "tourney" / "runs"
        # names of this objective's OOC runs on the build host (another loop may build the same
        # commit of the same component at once)
        self.tag = "unit-" if self.unit else ""
        if self.unit:
            if not self.comp["synth"]["parts"]:
                raise SystemExit(f"{a.comp} has no out-of-context part for the unit objective")
            if a.eval != "vivado-remote":
                print(f"[tourney] --objective unit evaluates with vivado-remote (not {a.eval})")
                a.eval = "vivado-remote"
            self.period = round(1000.0 / a.target_mhz, 3)
            self.branch = f"tourney/unit/{a.comp}"
            self.dir = runs / "unit" / a.comp
            self.wtroot = self.repo / ".tourney" / f"unit-{a.comp}"
        elif self.fmax:
            if a.eval != "vivado-remote":
                print(f"[tourney] --objective fmax evaluates with vivado-remote (not {a.eval})")
                a.eval = "vivado-remote"
            self.period = round(1000.0 / a.target_mhz, 3)
            self.branch = "tourney/fmax"
            self.dir = runs / "fmax" / a.comp
            self.fulldir = runs / "_full"
            self.fulldir.mkdir(parents=True, exist_ok=True)
            self.wtroot = self.repo / ".tourney" / f"fmax-{a.comp}"
        else:
            self.period = None
            self.branch = f"tourney/{a.comp}"
            self.dir = runs / a.comp
            self.wtroot = self.repo / ".tourney" / a.comp
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log = self.dir / "log.jsonl"
        self.lessons = self.dir / "LESSONS.md"
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
            if (c.get("sha") == sha and c.get("backend") == self.a.eval and
                    c.get("period") == self.period):
                if self.fmax:
                    c["full"] = self.full_result(sha, None)
                return c
        print(f"[tourney] measuring champion {sha[:9]} ({self.a.eval})", flush=True)
        wt = self.worktree("champion", sha, detach=True)
        try:
            out = self.shared_build / "tourney" / self.a.comp / f"{self.tag}champ-{sha[:9]}"
            if self.comp["synth"]["parts"]:
                m = G.synthesize(wt, self.comp, self.a.eval, out, self.period)
            else:
                m = {}
            m["perf_cycles"] = G.perf(wt) if self.comp.get("perf") else None
            if self.fmax:
                full = self.full_result(sha, wt)
        finally:
            self.drop(wt, None)
        m["sha"], m["period"], m["backend"] = sha, self.period, self.a.eval
        m["critical"] = self.critical(m) if not (self.fmax or self.unit) else ""
        if self.unit:          # measured here, not a winner's: the fmax reference starts anew
            m["anchor_fmax"] = m.get("fmax")
        cache.write_text(json.dumps(m, indent=1))
        if self.fmax:
            m["full"] = full
        return m

    # ---- the whole design (fmax objective)
    def full_key(self, sha: str) -> Path:
        """The cache file of a commit's full build. Keyed by the git trees of what the build
        reads (rtl/, boards/), not the commit: a champion that only took host or doc commits
        from main is not rebuilt. A result cached under the commit (older runs) is moved over."""
        from . import remote as R
        args = "".join(f"-{a.replace('=', '')}" for a in R.BUILD_ARGS)   # e.g. -FAST1
        trees = "".join(git("rev-parse", f"{sha}:{d}", cwd=self.repo)[:6] for d in ("rtl", "boards"))
        key = self.fulldir / f"t{trees}-{self.a.target_mhz:g}{args}.json"
        old = self.fulldir / f"{sha[:12]}-{self.a.target_mhz:g}{args}.json"
        if not key.exists() and old.exists():
            old.rename(key)
        return key

    def full_result(self, sha: str, wt: Path | None) -> dict:
        """The full board build of commit `sha` at the target clock: cached in runs/_full/ (all
        components share it); built on the host when missing (one waiter per commit, under a
        lock file, so two tournament processes never build the same tree)."""
        key = self.full_key(sha)
        with key.with_suffix(".lock").open("w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            if key.exists():
                return json.loads(key.read_text())
            own = wt is None
            if own:
                wt = self.worktree(f"full-{sha[:9]}", sha, detach=True)
            try:
                print(f"[tourney] full build of {sha[:9]} at {self.a.target_mhz:g} MHz on the "
                      f"build host", flush=True)
                t = time.time()
                r = G.full_design(wt, f"full-{sha[:12]}", self.a.target_mhz, FULL_BUILD_ID)
                r["seconds"], r["sha"] = round(time.time() - t), sha
            finally:
                if own:
                    self.drop(wt, None)
            key.write_text(json.dumps(r, indent=1))
            return r

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
        # not under tourney/<comp>/: a ref cannot be both a branch and a directory. Per objective
        # (the worktree root's name, e.g. fmax-otpu_vpu), as two loops may run the same component
        return f"tourney-slot/{self.wtroot.name}/{sid}"

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

    def dedicated(self) -> bool:
        """Whether this checkout is a linked worktree (a tournament's own, e.g. openTPU-tv), whose
        build/ only the tournament uses; the main checkout's build/ is everyone's."""
        return (Path(git("rev-parse", "--absolute-git-dir", cwd=self.repo)) !=
                (self.repo / git("rev-parse", "--git-common-dir", cwd=self.repo)).resolve())

    def clean_leftovers(self) -> None:
        """Drops the worktrees an earlier run of this component and objective left behind (killed
        or crashed mid-round), unless another run of it is live: every run holds a shared lock on
        <wtroot>.lock while it lives, the cleanup needs it exclusively first."""
        self.wtroot.parent.mkdir(parents=True, exist_ok=True)
        self._live = (self.wtroot.parent / f"{self.wtroot.name}.lock").open("w")
        try:
            fcntl.flock(self._live, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fcntl.flock(self._live, fcntl.LOCK_SH)
            return
        try:
            for wt in sorted(self.wtroot.iterdir()) if self.wtroot.is_dir() else []:
                if wt.is_dir():
                    print(f"[tourney] removing the leftover worktree {wt}", flush=True)
                    self.drop(wt, None if wt.name == "champion" or wt.name == "sync" or
                              wt.name.startswith("full-") else self.slot_branch(wt.name))
        finally:
            fcntl.flock(self._live, fcntl.LOCK_SH)

    def drop(self, wt: Path, branch: str | None) -> None:
        if self.a.keep and branch:
            return
        try:
            G.remote_clean(wt)
        except Exception as e:  # noqa: BLE001 -- a leftover tree is not worth failing a round
            print(f"[tourney] could not remove {G.remote_name(wt)} on the build host: {e}")
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

    def logged(self) -> int:
        """Slots logged so far."""
        return len([l for l in self.log.read_text().splitlines() if l.strip()]
                   ) if self.log.exists() else 0

    def lessons_text(self, n: int = 30) -> str:
        """This run's lessons; the unit objective adds the last `n` of the component's other
        tournaments (the Yosys component tournament's, the fmax tournament's)."""
        own = self.lessons.read_text() if self.lessons.exists() else "(none yet)"
        if not self.unit:
            return own
        runs = self.repo / "tools" / "tourney" / "runs"
        out = [own]
        for p, what in ((runs / self.a.comp / "LESSONS.md", "the component tournament (Yosys)"),
                        (runs / "fmax" / self.a.comp / "LESSONS.md",
                         "the whole-design fmax tournament")):
            lines = [l for l in p.read_text().splitlines()
                     if l.strip() and not l.startswith("#")][-n:] if p.exists() else []
            if lines:
                out.append(f"From {what} on this component:\n" + "\n".join(lines))
        return "\n\n".join(out)

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
            lessons = self.lessons_text()
            cats = (AG.FMAX_CATEGORIES if self.fmax else
                    AG.UNIT_CATEGORIES if self.unit else AG.CATEGORIES)
            # rotated over the slots and the logged slots before them, so a component run one
            # slot per round still takes each focus in turn
            cat = cats[(k + self.logged()) % len(cats)]
            rec["category"] = cat.split(":")[0]
            if self.fmax:
                agent("hyp", AG.hypothesis_prompt_fmax(self.comp, wt, champ, lessons,
                                                       self.history(), cat, self.a.target_mhz))
            elif self.unit:
                agent("hyp", AG.hypothesis_prompt_unit(
                    self.comp, wt, champ, lessons, self.history(), cat, self.a.target_mhz,
                    A.unit_floor(champ)))
            else:
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
            # the diff as git writes it: git() strips, which cuts a hunk's last context line
            # when it is empty (" "), and the patch no longer applies
            (pdir / f"{sid}.patch").write_text(subprocess.run(
                ["git", "diff"], cwd=wt, capture_output=True, text=True, check=True).stdout)
            gate("lint", G.lint, wt)
            rec["fast"] = gate("fast", G.pytest, wt, self.comp["tests"]["fast"], None, "fast")
            rec["board"] = gate("board", G.pytest, wt, self.comp["tests"]["board"], G.BOARD_ENV,
                                "board")
            if self.comp.get("perf"):
                cyc = gate("perf", G.perf, wt)
                rec["perf_cycles"] = cyc
                tol = A.PERF_TOL_FMAX if self.fmax else A.PERF_TOL
                if not A.perf_ok(champ.get("perf_cycles"), cyc, tol):
                    raise G.GateFailure("perf", f"{cyc} cycles vs champion "
                                                f"{champ.get('perf_cycles')} (> {tol:+.1%})")
            if self.fmax:
                self.ooc_step(rec, wt, sid, champ, gate)
                raise _Done
            m = gate("synth", G.synthesize, wt, self.comp, self.a.eval,
                     self.shared_build / "tourney" / self.a.comp / f"{self.tag}{sid}", self.period)
            rec["metrics"] = {q: m.get(q) for q in ("lut", "lutram", "ff", "dsp", "bram36",
                                                     "bram18", "logic_ns", "fmax", "area_eq")}
            if self.unit:
                rec["metrics"].update({q: m.get(q) for q in ("wns", "whs", "collisions")})
                rec["_m"] = m                      # the new champion's result if it wins
                ok, why = A.accept_unit(champ, m)
            else:
                ok, why = A.accept(champ, m, self.comp["target_mhz"])
            rec["outcome"], rec["reason"] = ("improvement" if ok else "no_gain"), why
            rec["gain"] = (champ["area_eq"] - m["area_eq"]) / champ["area_eq"] + \
                          (m["fmax"] - champ["fmax"]) / champ["fmax"]
        except _Done:
            pass
        except G.GateFailure as e:
            rec["outcome"], rec["reason"] = "broken", f"{e.gate}: {e.tail}"
        except Exception as e:  # noqa: BLE001 -- agent / tool crashes are logged, not fatal
            rec["outcome"], rec["reason"] = "error", f"{type(e).__name__}: {str(e)[-600:]}"
        rec["seconds"] = round(time.time() - t0)
        rec["wt"] = str(wt)
        print(f"[tourney] {sid}: {rec['outcome']} -- {rec.get('reason', '')[:160]}", flush=True)
        return rec

    # ---- fmax objective: the out-of-context step and the round's full build
    def ooc_step(self, rec: dict, wt: Path, sid: str, champ: dict, gate) -> None:
        """Sets rec's outcome to `candidate` (earns a full build) or `no_gain`."""
        if self.comp["synth"]["parts"]:
            m = gate("ooc", G.synthesize, wt, self.comp, "vivado-remote", Path(sid), self.period)
            rec["ooc"] = {q: m.get(q) for q in ("lut", "lutram", "ff", "dsp", "bram36", "bram18",
                                                 "wns", "whs", "fmax", "area_eq", "collisions")}
            ok, why = A.ooc_promising(champ, m, self.a.target_mhz)
            rec["ooc_area"] = A.area_delta(champ, m)
            rec["ooc_gain"] = A.ooc_score(champ, m)       # fmax change - area_eq change
        else:
            ok, why = True, "cross-unit component: no out-of-context step"
            rec["ooc_gain"] = 0.0
        rec["outcome"], rec["reason"] = ("candidate", why) if ok else ("no_gain", f"ooc: {why}")

    def full_step(self, recs: list[dict], champ: dict) -> None:
        """The round's full builds: the candidates with the best OOC gains (ties: slot order),
        as many at once as the build hosts have free slots (at least one; the others are logged
        `not_built`, their patches stay in patches/). The builds that pass the rule are
        confirmed in score order until one confirms; the others that passed are `runner_up`."""
        cands = sorted((x for x in recs if x["outcome"] == "candidate"),
                       key=lambda x: -x.get("ooc_gain", 0.0))
        if not cands:
            return
        n = min(len(cands), G.free_build_slots())
        build, rest = cands[:n], cands[n:]
        for x in rest:
            x["outcome"] = "not_built"
            x["reason"] += (f"; the round's full build{'s' if n > 1 else ''} went to "
                            + ", ".join(w["id"] for w in build))
        if n > 1:
            print(f"[tourney] {n} free build slots: full builds of "
                  + ", ".join(w["id"] for w in build), flush=True)
            with ThreadPoolExecutor(n) as ex:
                list(ex.map(lambda w: self.full_build(w, champ), build))
        else:
            self.full_build(build[0], champ)
        passed = sorted((w for w in build if w["outcome"] == "improvement"),
                        key=lambda w: -w["gain"])
        for k, w in enumerate(passed):
            if self.confirm(w, champ):
                for x in passed[k + 1:]:
                    x["outcome"] = "runner_up"
                    x["reason"] += f"; the round's winner is {w['id']} (confirmed)"
                return

    def full_build(self, w: dict, champ: dict) -> None:
        """A candidate's full build, judged by the full-design rule: outcome `improvement`
        (to be confirmed), `no_gain` or `broken`."""
        t = time.time()
        try:
            full = G.full_design(Path(w["wt"]), f"{self.a.comp}-{w['id']}", self.a.target_mhz,
                                 FULL_BUILD_ID)
        except G.GateFailure as e:
            w["outcome"], w["reason"] = "broken", f"{e.gate}: {e.tail}"
            return
        finally:
            w.setdefault("gate_seconds", {})["full"] = round(time.time() - t, 1)
        full["seconds"] = round(time.time() - t)
        w["full"] = {q: full.get(q) for q in ("period", "wns", "whs", "fmax", "wns_design", "lut",
                                               "lutram", "ff", "dsp", "bram36", "bram18",
                                               "collisions", "congested", "host")}
        w["_full"] = full
        ad = A.area_delta(champ["full"], full)
        w["full_area"] = ad                              # per-resource deltas, every candidate
        cyc = (champ.get("perf_cycles"), w.get("perf_cycles"))
        dc = A.cycle_delta(*cyc)
        w["cycles_delta"] = dc
        ok, why = A.accept_fmax(champ["full"], full, *cyc)
        w["outcome"], w["reason"] = ("improvement" if ok else "no_gain"), why
        w["gain"] = A.score((full["fmax"] - champ["full"]["fmax"]) / champ["full"]["fmax"],
                            ad["area_eq"], dc)

    def confirm(self, w: dict, champ: dict) -> bool:
        """The confirmation: a second placement of the same tree must pass the rule as well.
        False leaves w `unconfirmed`."""
        full, why = w["_full"], w["reason"]
        cyc = (champ.get("perf_cycles"), w.get("perf_cycles"))
        t = time.time()
        try:
            conf = G.full_design(Path(w["wt"]), f"{self.a.comp}-{w['id']}-c", self.a.target_mhz,
                                 CONFIRM_BUILD_ID)
        except G.GateFailure as e:
            w["outcome"], w["reason"] = "unconfirmed", why + f"; confirmation build broken: {e.tail[-300:]}"
            return False
        finally:
            w.setdefault("gate_seconds", {})["confirm"] = round(time.time() - t, 1)
        w["confirm"] = {q: conf.get(q) for q in ("period", "wns", "whs", "fmax", "lut", "lutram",
                                                  "ff", "dsp", "bram36", "congested", "host")}
        ok2, why2 = A.accept_fmax(champ["full"], conf, *cyc)
        both = f"WNS {full['wns']:+.3f} / {conf['wns']:+.3f} ns (build / confirmation)"
        if not ok2:
            w["outcome"], w["reason"] = "unconfirmed", f"{why}; confirmation failed: {why2}; {both}"
            return False
        w["reason"] = f"{why}; confirmed: {both}"
        # rank on the worse of the two draws
        w["gain"] = min(w["gain"], A.score((conf["fmax"] - champ["full"]["fmax"]) /
                                           champ["full"]["fmax"],
                                           A.area_delta(champ["full"], conf)["area_eq"],
                                           w["cycles_delta"]))
        return True

    # ---- rounds
    def round(self, r: int) -> None:
        self.sync_base()
        champ = self.champion()
        if self.fmax:
            f = champ["full"]
            print(f"[tourney] round {r}: champion {champ['sha'][:9]} full fmax {f['fmax']:.2f} "
                  f"MHz (WNS {f['wns']:+.3f} ns at {f['period']} ns), OOC fmax "
                  f"{champ.get('fmax') or 0:.1f} MHz, perf {champ.get('perf_cycles')}", flush=True)
        elif self.unit:
            print(f"[tourney] round {r}: champion {champ['sha'][:9]} OOC area_eq "
                  f"{champ['area_eq']:.0f} fmax {champ['fmax']:.1f} MHz (area wins need >= "
                  f"{A.unit_floor(champ):.1f}) perf {champ.get('perf_cycles')}", flush=True)
        else:
            print(f"[tourney] round {r}: champion {champ['sha'][:9]} area_eq "
                  f"{champ['area_eq']:.0f} fmax {champ['fmax']:.0f} MHz perf "
                  f"{champ.get('perf_cycles')}", flush=True)
        rid = f"r{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        with ThreadPoolExecutor(self.a.slots) as ex:
            recs = list(ex.map(lambda k: self.slot(rid, k, champ), range(self.a.slots)))
        if self.fmax:
            self.full_step(recs, champ)
        winners = sorted((x for x in recs if x["outcome"] == "improvement"),
                         key=lambda x: -x["gain"])
        if winners:
            w = winners[0]
            wt = Path(w["wt"])
            git("add", "--", *w["files"], cwd=wt)
            git("commit", "-q", "-m", f"tourney {self.tag.replace('-', ' ')}{self.a.comp}: "
                f"{w.get('title', w['id'])}\n\n{w['reason']}\n\nSlot {w['id']}, agent "
                f"{self.a.agent}, eval {self.a.eval}.", cwd=wt)
            sha = git("rev-parse", "HEAD", cwd=wt)
            git("update-ref", f"refs/heads/{self.branch}", sha, champ["sha"], cwd=self.repo)
            w["outcome"], w["merged"] = "accepted", sha
            if self.fmax:                  # the committed tree is the one just built
                self.full_key(sha).write_text(json.dumps(dict(w["_full"], sha=sha), indent=1))
                with (self.dir.parent / "WINNERS.jsonl").open("a") as f:
                    f.write(json.dumps({
                        "time": dt.datetime.now().isoformat(timespec="seconds"),
                        "comp": self.a.comp, "slot": w["id"], "sha": sha,
                        "old": champ["sha"], "title": w.get("title"), "reason": w["reason"],
                        "score": w["gain"], "full": w["full"], "confirm": w.get("confirm"),
                        "area": w.get("full_area"), "cycles_delta": w.get("cycles_delta"),
                        "perf_cycles": w.get("perf_cycles"),
                        "perf_old": champ.get("perf_cycles")}) + "\n")
            if self.unit:                  # the committed tree is the one just measured
                new = dict(w["_m"], sha=sha, period=self.period, backend=self.a.eval,
                           perf_cycles=w.get("perf_cycles"), critical="",
                           anchor_fmax=champ.get("anchor_fmax") or champ["fmax"])
                (self.dir / "champion.json").write_text(json.dumps(new, indent=1))
                keys = ("lut", "lutram", "ff", "dsp", "bram36", "bram18", "wns", "fmax",
                        "area_eq")
                with (self.dir.parent / "WINNERS.jsonl").open("a") as f:
                    f.write(json.dumps({
                        "time": dt.datetime.now().isoformat(timespec="seconds"),
                        "comp": self.a.comp, "slot": w["id"], "sha": sha,
                        "old": champ["sha"], "title": w.get("title"), "reason": w["reason"],
                        "gain": w["gain"], "ooc_old": {q: champ.get(q) for q in keys},
                        "ooc": {q: w["_m"].get(q) for q in keys},
                        "perf_cycles": w.get("perf_cycles"),
                        "perf_old": champ.get("perf_cycles"),
                        "tests": {g: w.get(g) for g in ("fast", "board")}}) + "\n")
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
            self.append({k: v for k, v in x.items() if k not in ("wt", "_full", "_m")})
            self.drop(Path(x["wt"]), self.slot_branch(x["id"]))
        try:
            G.remote_prune()
        except Exception as e:  # noqa: BLE001
            print(f"[tourney] could not prune the build host's Verilator cache: {e}")
        if self.dedicated():                   # the agents' quick-test builds on this machine
            gone = G.local_prune(self.shared_build)
            if gone:
                print(f"[tourney] pruned {len(gone)} local Verilator builds", flush=True)
        try:
            self.wtroot.rmdir()                # the slot trees are gone: so is their empty root
        except OSError:
            pass

    def main(self) -> None:
        # the same control files as tools/tourney/forever.sh: a planned downtime or a quiet
        # window holds every new round, also the ones a `make tourney-fmax` loop starts
        if STOP.exists():
            print(f"[tourney] {STOP} exists: not starting {self.a.comp}", flush=True)
            raise SystemExit(3)
        waited = False
        while PAUSE.exists():
            if not waited:
                print(f"[tourney] {PAUSE} exists: {self.a.comp} waits", flush=True)
                waited = True
            time.sleep(60)
        self.clean_leftovers()
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
    ap.add_argument("--objective", choices=("area", "fmax", "unit"),
                    default=os.environ.get("OBJECTIVE", "area"))
    ap.add_argument("--target-mhz", type=float, default=float(os.environ.get("TARGET_MHZ",
                                                                             "133.33")))
    Run(ap.parse_args(argv)).main()


if __name__ == "__main__":
    sys.exit(main())
