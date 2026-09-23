"""Agent runtime (claude / codex CLIs, headless) and the three prompts: hypothesis,
implementation, scribe. Agents run with their working directory in the slot worktree."""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

CATEGORIES = [
    "timing: shorten the critical path (retime, split a stage, precompute, one-hot, "
    "register an input/output) without adding latency the protocol cannot absorb",
    "area: share or time-multiplex logic, narrow datapaths to the bits actually needed, "
    "remove redundant state",
    "memories: move register arrays or wide muxes into LUT RAM / block RAM / SRLs, or use "
    "DSP48 cascades or pre-adders instead of LUT arithmetic",
    "structure: restructure the control or datapath (e.g. a different queue/scoreboard/tree "
    "organisation) that removes whole blocks of logic",
]

# Per-role model and reasoning effort. MODEL_<ROLE> / EFFORT_<ROLE> (ROLE = HYP, IMPL, SCRIBE)
# may be comma lists: slot k of a round uses entry k mod len, so one round can compare settings.
# Every role runs Opus 5.5; the scribe (a one-line summary) at low effort.
ROLES = ("hyp", "impl", "scribe")
ROLE_DEFAULTS = {"hyp": "claude-opus-5-5", "impl": "claude-opus-5-5", "scribe": "claude-opus-5-5"}
EFFORT_DEFAULTS = {"hyp": "high", "impl": "high", "scribe": "low"}
EFFORTS = ("low", "medium", "high", "xhigh", "max")
ALIASES = {"opus": "claude-opus-5-5"}


def _pick(raw: str | None, slot: int) -> str | None:
    names = [n.strip() for n in (raw or "").split(",") if n.strip()]
    return names[slot % len(names)] if names else None


def model_for(role: str, slot: int, provider: str, env: dict | None = None) -> str | None:
    """The model for `role` in slot `slot` (None: the CLI's own default)."""
    env = os.environ if env is None else env
    m = _pick(env.get(f"MODEL_{role.upper()}"), slot)
    if m is None:
        return ROLE_DEFAULTS[role] if provider == "claude" else None
    return ALIASES.get(m, m) if provider == "claude" else m


def effort_for(role: str, slot: int, env: dict | None = None) -> str:
    """The reasoning effort for `role` in slot `slot` (claude --effort; codex
    model_reasoning_effort, where xhigh/max map to high)."""
    env = os.environ if env is None else env
    e = _pick(env.get(f"EFFORT_{role.upper()}"), slot) or EFFORT_DEFAULTS[role]
    if e not in EFFORTS:
        raise ValueError(f"EFFORT_{role.upper()} must be one of {EFFORTS}, not {e!r}")
    return e


def build_cmd(provider: str, prompt: str, cwd: Path, last_msg: Path,
              model: str | None = None, effort: str | None = None) -> list[str]:
    if provider == "claude":
        cmd = ["claude", "-p", prompt, "--dangerously-skip-permissions",
               "--output-format", "stream-json", "--verbose"]
        return (cmd + (["--model", model] if model else []) +
                (["--effort", effort] if effort else []))
    if provider == "codex":
        cmd = ["codex", "--ask-for-approval", "never", "exec", "-C", str(cwd),
               "--sandbox", "workspace-write", "--skip-git-repo-check", "--json",
               "--output-last-message", str(last_msg)]
        if effort:
            ce = {"xhigh": "high", "max": "high"}.get(effort, effort)
            cmd += ["-c", f"model_reasoning_effort={ce}"]
        return cmd + (["--model", model] if model else []) + [prompt]
    raise ValueError(f"AGENT must be claude or codex, not {provider!r}")


def _summary(provider: str, line: str) -> str | None:
    try:
        ev = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(ev, dict):
        return None
    if provider == "claude" and ev.get("type") == "assistant":
        for c in ev.get("message", {}).get("content", []) or []:
            if isinstance(c, dict) and c.get("type") == "tool_use":
                inp = c.get("input") or {}
                t = inp.get("file_path") or inp.get("command") or inp.get("pattern") or ""
                return f"{c.get('name')}: {str(t)[:90]}"
    if provider == "claude" and ev.get("type") == "result":
        return f"result: {str(ev.get('result', ''))[:120]}"
    if provider == "codex" and ev.get("type") == "item.completed":
        it = ev.get("item") or {}
        t = it.get("type")
        if t == "command_execution":
            return f"shell: {str(it.get('command'))[:90]}"
        if t == "file_change":
            return f"file: {it.get('path', '')}"
    return None


def usage_of(provider: str, ev: dict, acc: dict) -> None:
    """Accumulates token usage / cost from one stream event into `acc`, as far as the CLI reports
    it: claude's final `result` event carries usage and total_cost_usd; codex reports tokens on
    each `turn.completed` (no cost)."""
    if provider == "claude" and ev.get("type") == "result":
        u = ev.get("usage") or {}
        acc["input_tokens"] = (u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
                               + u.get("cache_read_input_tokens", 0))
        acc["cache_read_tokens"] = u.get("cache_read_input_tokens", 0)
        acc["output_tokens"] = u.get("output_tokens", 0)
        if ev.get("total_cost_usd") is not None:
            acc["cost_usd"] = float(ev["total_cost_usd"])
        if ev.get("num_turns") is not None:
            acc["turns"] = ev["num_turns"]
    if provider == "codex" and ev.get("type") == "turn.completed":
        u = ev.get("usage") or {}
        for k, a in (("input_tokens", "input_tokens"), ("cached_input_tokens", "cache_read_tokens"),
                     ("output_tokens", "output_tokens")):
            acc[a] = acc.get(a, 0) + int(u.get(k, 0))


class AgentError(RuntimeError):
    def __init__(self, msg: str, info: dict):
        super().__init__(msg)
        self.info = info


def run(provider: str, prompt: str, cwd: Path, log: Path, timeout: int, tag: str,
        model: str | None = None, effort: str | None = None) -> dict:
    """Runs the agent, streaming its events to `log`. Returns {text (final message), model,
    effort, seconds, usage}; raises AgentError (carrying the same dict as .info) on failure."""
    last = log.with_suffix(".last")
    cmd = build_cmd(provider, prompt, cwd, last, model, effort)
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    timer = threading.Timer(timeout, proc.kill)
    timer.start()
    final, usage = "", {}
    try:
        with log.open("w") as f:
            for line in proc.stdout:
                f.write(line)
                f.flush()
                s = _summary(provider, line)
                if s:
                    print(f"    [{tag}] {s}", flush=True)
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(ev, dict):
                    usage_of(provider, ev, usage)
                    if ev.get("type") == "result":
                        final = str(ev.get("result", ""))
        proc.wait()
    finally:
        timer.cancel()
    if last.exists():
        final = last.read_text()
    info = {"text": final, "model": model or "default", "effort": effort, "seconds": round(time.time() - t0, 1),
            "usage": usage}
    if proc.returncode != 0:
        raise AgentError(f"{provider} exited {proc.returncode} (log {log})", info)
    return info


# ------------------------------------------------------------------------------ prompts
def _src(wt: Path, files: list[str]) -> str:
    return "\n\n".join(f"=== {f} ===\n{(wt / f).read_text()}" for f in files)


def hypothesis_prompt(comp: dict, wt: Path, champ: dict, lessons: str, recent: str,
                      category: str, critical: str) -> str:
    files = comp["allowed"]
    return f"""You are a hardware architecture research agent working on openTPU, an FPGA
accelerator (Kintex-7 xc7k480t, 100 MHz target). Your job: propose ONE concrete change to the
component `{comp['name']}` that improves its area and/or clock frequency, without changing
what it computes.

## Component
{comp['description']}

Files you may change (only these): {', '.join(files)}
Synthesis parameters (board build): {json.dumps([p['params'] for p in comp['synth']['parts']])}

## Current champion (Yosys synth_xilinx, out of context)
LUT {champ.get('lut')}, LUT-RAM {champ.get('lutram')}, FF {champ.get('ff')}, DSP {champ.get('dsp')},
BRAM36 {champ.get('bram36')}, logic {champ.get('logic_ns')} ns, est fmax {champ.get('fmax', 0):.0f} MHz,
area-equivalent {champ.get('area_eq', 0):.0f} (LUT + LUTRAM + 0.5 FF + 40 DSP + 80 BRAM36).
Target: est fmax >= {comp['target_mhz']} MHz.

Critical path (Yosys sta):
{critical}

## How a change is judged (the orchestrator does all of this, you do not)
1. Only the files above changed. 2. Verilator lint of the whole design. 3. Bit-exact RTL tests
against the instruction-set simulator (tests/test_rtl.py subsets, also on the board's AXI memory
path) -- results must stay IDENTICAL: same fp32 rounding, same order of operations (docs/isa.md),
same port protocols (timing may change; grant/handshake contracts may not break). 4. A Qwen3
decode-token proxy on the RTL may not get more than 0.2% slower. 5. Accepted if est fmax >= target
and area-equivalent drops >= 1%, or est fmax rises >= 3% with area at most +1%.

## Focus for this slot
{category}

## History
Recent outcomes:
{recent}

Lessons from earlier rounds (tools/tourney/runs/{comp['name']}/LESSONS.md):
{lessons}

## Source
{_src(wt, files)}

## Instructions
Read the code (and docs/isa.md, the unit's neighbours in rtl/ if you need the protocols). Then
WRITE a file HYPOTHESIS.md at the repository root (your current directory) with:
  # <short title>
  Category: <timing|area|memories|structure>
  Motivation: why this is the bottleneck (cite the numbers above)
  Change: exactly what to change, where, and why results stay bit-identical
  Expected: estimated LUT / FF / fmax effect
  Risks: what could break
Do not modify any other file in this phase."""


def implement_prompt(comp: dict, wt: Path) -> str:
    return f"""You are implementing an approved hardware change in openTPU.

Read HYPOTHESIS.md in your current directory and implement it by editing ONLY these files:
{', '.join(comp['allowed'])}

Rules:
- Results must stay bit-identical to the instruction-set simulator (opentpu/isasim.py,
  docs/isa.md); keep module ports and handshake/grant contracts intact.
- Keep the code style of the file (comment density, naming). Keep `ifndef SYNTHESIS` checks.
- Check your work: `verilator --lint-only -Wno-fatal -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC
  -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM --timing --top-module otpu_top` over the files of
  opentpu/rtlsim.py RTL_SOURCES (prefix rtl/) plus sim/verilator/otpu_axi_mem.sv, and run a quick
  test such as `PYTHONPATH=. python3 -m pytest -q -x "tests/test_rtl.py::test_mlp_rtl[1]"`.
  The orchestrator runs the full gates afterwards; do not run long suites.
- Do not edit tests, tools, docs or any other RTL file. Do not commit.
- When done, write IMPLEMENTATION.md at the repository root: what you changed and anything the
  reviewer should know (3-10 lines)."""


def scribe_prompt(comp: str, hyp: str, outcome: str, detail: str) -> str:
    return f"""One experiment on the openTPU component {comp} just finished.

Hypothesis:
{hyp[:3000]}

Outcome: {outcome}
Detail: {detail[:2000]}

Write ONE line (max 200 characters) for LESSONS.md capturing what future experiments on this
component should know (what worked or failed and why). Reply with that line only, no preamble.
Do not modify any files."""
