"""Agent runtime (claude / codex CLIs, headless) and the three prompts: hypothesis,
implementation, scribe. Agents run with their working directory in the slot worktree."""
from __future__ import annotations

import json
import os
import re
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
    """The model for `role` in slot `slot` (None: the CLI's own default). With claude every
    role runs Opus: a MODEL_<ROLE> naming another family is an error, not a silent downgrade."""
    env = os.environ if env is None else env
    m = _pick(env.get(f"MODEL_{role.upper()}"), slot)
    if m is None:
        return ROLE_DEFAULTS[role] if provider == "claude" else None
    if provider != "claude":
        return m
    m = ALIASES.get(m, m)
    if "opus" not in m:
        raise ValueError(f"MODEL_{role.upper()}={m!r}: the tournament runs Opus for every role")
    return m


def effort_for(role: str, slot: int, env: dict | None = None) -> str:
    """The reasoning effort for `role` in slot `slot` (claude --effort; codex
    model_reasoning_effort, where xhigh/max map to high)."""
    env = os.environ if env is None else env
    e = _pick(env.get(f"EFFORT_{role.upper()}"), slot) or EFFORT_DEFAULTS[role]
    if e not in EFFORTS:
        raise ValueError(f"EFFORT_{role.upper()} must be one of {EFFORTS}, not {e!r}")
    return e


ALLOWED_TOOLS = [
    "Read", "Edit", "Write", "Glob", "Grep",
    "Bash(verilator:*)", "Bash(python3 -m pytest:*)", "Bash(PYTHONPATH=. python3 -m pytest:*)",
    "Bash(git diff:*)", "Bash(git status:*)", "Bash(git log:*)", "Bash(ls:*)",
    "Bash(grep:*)", "Bash(wc:*)", "Bash(head:*)", "Bash(tail:*)",
]


def build_cmd(provider: str, prompt: str, cwd: Path, last_msg: Path,
              model: str | None = None, effort: str | None = None) -> list[str]:
    if provider == "claude":
        # Least privilege: edits are auto-accepted inside the slot's worktree only (cwd),
        # shell access is limited to lint/test/read-only commands, no network tools; any
        # other permission request is denied (headless -p mode cannot prompt).
        cmd = ["claude", "-p", prompt, "--permission-mode", "acceptEdits",
               "--allowedTools", *ALLOWED_TOOLS,
               "--disallowedTools", "WebFetch", "WebSearch",
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
    return "\n\n".join(f"=== {f} ===\n" + ((wt / f).read_text() if (wt / f).exists() else
                                             "(does not exist yet: you may create it)")
                       for f in files)


def _src_paths(files: list[str]) -> str:
    """For a component whose files are too many to quote (the whole RTL): where to read."""
    return ("The files are not quoted here (the whole RTL is too large): read the ones on the "
            "paths you target, using the unit map under the path summary above (the unit's file, "
            "then the slice / board top that wires it to the other unit). Files you may change: "
            + ", ".join(files) + ".")


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


FMAX_CATEGORIES = [
    "pipeline: cut the worst path with a register stage the protocol absorbs (a FIFO or skid "
    "stage, a registered block RAM output, an extra stage in a fixed-latency pipe matched on "
    "every branch)",
    "logic depth: precompute, one-hot decode, move a mux or compare before the register, split a "
    "wide reduction or carry chain",
    "fanout and placement: replicate high-fanout registers (per bank / per lane copies), keep "
    "broadcast enables local, move logic next to the block RAM or DSP it drives",
    "memories and DSPs: use the block RAM / DSP48 output and input registers, cascade DSPs, "
    "keep LUT RAM reads registered",
]


def expand(wt: Path, globs: list[str]) -> list[str]:
    """The files the allowed globs name in the worktree (a literal path stays as is)."""
    out = []
    for g in globs:
        hits = sorted(str(p.relative_to(wt)) for p in wt.glob(g)) if any(
            c in g for c in "*?[") else [g]
        out += [h for h in hits if h not in out]
    return out


# the units of the board build (instance names under otpu_fpga_top_ld/u_sys/u_board, the
# LiteDRAM build's; otpu_fpga_top/u_board, the MIG build's) and their RTL
UNIT_FILES = {
    "u_seq": "rtl/seq/otpu_seq.sv", "u_tmem": "rtl/mem/otpu_tmem.sv",
    "u_act": "rtl/mem/otpu_actram.sv", "u_dma": "rtl/dma/otpu_dma.sv",
    "u_vpu/u_tail": "rtl/vpu/otpu_se_tail.sv", "u_mxu": "rtl/mxu/otpu_mxu.sv",
    "u_quant": "rtl/vpu/otpu_quant.sv", "u_vpu": "rtl/vpu/otpu_vpu.sv",
    "u_mem": "rtl/mem/otpu_axi_dram.sv", "u_nmem": "rtl/mem/otpu_native_dram.sv",
    "u_coll": "rtl/top/otpu_coll.sv",
    "u_ctrl": "rtl/boards/ypcb-00338/otpu_ctrl.sv", "u_trace": "rtl/boards/ypcb-00338/otpu_trace.sv",
    "u_slice": "rtl/top/otpu_slice.sv (its own logic)",
    "u_board": "rtl/boards/ypcb-00338/otpu_board.sv (its own logic)",
    "u_ch": "rtl/boards/ypcb-00338/otpu_mem_ch.sv (u_ch0 / u_ch1: a channel's bridge)",
    "u_split": "rtl/boards/ypcb-00338/otpu_axi_split2.sv",
    "u_sys": "rtl/boards/ypcb-00338/otpu_native_sys.sv (its own logic)",
    "u_ld": "the LiteDRAM core (boards/ypcb-00338/litedram/otpu_litedram.v, generated by "
            "tools/litedram/gen_core.py): not hand-written RTL",
    "u_bd": "the block design (XDMA IP; the MIG build's MIGs): not RTL",
}
_PATH_RE = re.compile(r"^\s*(-?[\d.]+) ns \S+ (\S+) -> (\S+) levels (\d+), data ([\d.]+) ns "
                      r"\(logic ([\d.]+), route ([\d.]+)\)")


def unit_of(cell: str) -> str:
    """The unit of a timing-path pin (`u_board/u_slice/u_dma/cleft_reg[28]/C` -> `u_dma`):
    the first instance under the slice (or the board); `u_vpu/u_tail` for the stream engine's
    tail; `u_ch` for either channel's bridge (`u_sys/u_ch1/...`);
    the slice / board itself for their own registers. The LiteDRAM build's otpu_native_sys
    (`u_sys/`) around the board is looked through, and a generate block's name
    (`g_native.u_nmem`) is left out."""
    parts = [p.split(".")[-1] for p in cell.split("/")[:-1]]      # the pin name goes
    if len(parts) > 1 and parts[0] == "u_sys":
        parts = parts[1:]
        if parts[0] in ("u_ch0", "u_ch1"):
            return "u_ch"
        if len(parts) == 1:                          # a register of otpu_native_sys itself
            return "u_sys"
        if parts[0] != "u_board":
            return parts[0]
    if parts and parts[0] == "u_board":
        parts = parts[1:]
        top = "u_board"
        if parts and parts[0] == "u_slice":
            parts, top = parts[1:], "u_slice"
    else:
        top = parts[0] if parts else "?"
        parts = parts[1:] if parts else []
    if len(parts) < 2 or not parts[0].startswith("u_"):   # a register of the top itself
        return top
    if parts[0] == "u_vpu" and len(parts) > 2 and parts[1].split(".")[-1] == "u_tail":
        return "u_vpu/u_tail"
    return parts[0]


def path_summary(timing: str) -> str:
    """The worst paths grouped by source and destination unit: how many, the worst slack, the
    logic levels and the share of the data delay that is routing."""
    groups: dict = {}
    for line in (timing or "").splitlines():
        m = _PATH_RE.match(line)
        if not m:
            continue
        slack, src, dst, lv, data, _logic, route = m.groups()
        g = groups.setdefault((unit_of(src), unit_of(dst)), [0, 0.0, 0, 0.0, 0.0])
        g[0] += 1
        g[1] = min(g[1], float(slack)) if g[0] > 1 else float(slack)
        g[2] = max(g[2], int(lv))
        g[3] += float(data)
        g[4] += float(route)
    if not groups:
        return "(no paths parsed)"
    rows = sorted(groups.items(), key=lambda kv: kv[1][1])
    out = [f"- {a} -> {b}: {n} of the paths, worst {w:+.3f} ns, up to {lv} levels, "
           f"route {r / d:.0%} of the data delay" for (a, b), (n, w, lv, d, r) in rows]
    units = sorted({u for k in groups for u in k})
    out.append("Units: " + "; ".join(f"{u} = {UNIT_FILES.get(u, '?')}" for u in units))
    return "\n".join(out)


def hypothesis_prompt_fmax(comp: dict, wt: Path, champ: dict, lessons: str, recent: str,
                           category: str, target_mhz: float) -> str:
    """The hypothesis prompt of the fmax objective: the whole board's post-route timing is the
    score, so the agent sees the full design's worst paths and congestion, and (if the component
    has one) its own out-of-context result."""
    files = expand(wt, comp["allowed"])
    full = champ.get("full") or {}
    ooc = champ if champ.get("fmax") is not None and comp["synth"]["parts"] else None
    ooc_txt = (f"""## This component alone (Vivado out of context, placed and routed at {1000 / target_mhz:.3f} ns)
LUT {ooc.get('lut')}, LUT-RAM {ooc.get('lutram')}, FF {ooc.get('ff')}, DSP {ooc.get('dsp')}, BRAM36 {ooc.get('bram36')}
WNS {ooc.get('wns') if ooc.get('wns') is not None else 'n/a'} ns -> fmax {ooc['fmax']:.1f} MHz.
Worst paths inside the component:
{ooc.get('timing', '(not available)')}
""" if ooc else "## This component alone\nNo out-of-context run: its paths only exist in the "
            "whole design, so every candidate goes straight to the full build.\n")
    return f"""You are a hardware timing-closure agent working on openTPU, an FPGA accelerator on a
Kintex-7 xc7k480t-2 (Vivado 2026.1). The goal of this tournament is a faster clock for the WHOLE
design, and a smaller one: propose ONE concrete change to the component `{comp['name']}` that
raises the post-route fmax of the full board build (core clock) or shrinks its area at the same
timing, without changing what the design computes.

## Component
{comp['description']}

Files you may change (only these): {', '.join(files)}

## The whole design now (full board build, `make bit DDR=1066 CORE_MHZ={target_mhz:g}`)
core clock {full.get('core_clock', '?')}: period {full.get('period', '?')} ns, WNS {full.get('wns', '?')} ns
-> fmax {full.get('fmax') or 0:.2f} MHz (target {target_mhz:g} MHz); WHS {full.get('whs', '?')} ns.
Utilization: LUT {full.get('lut')}, LUT-RAM {full.get('lutram')}, FF {full.get('ff')}, DSP {full.get('dsp')},
BRAM36 {full.get('bram36')} (xc7k480t: 298,600 LUT, 1,920 DSP, 955 BRAM36).

The 30 worst setup paths of the whole design (slack, start -> end, logic levels, data delay):
{full.get('timing') or '(not available)'}

The same paths by source and destination unit:
{path_summary(full.get('timing') or '')}

Congestion (report_design_analysis -congestion):
{full.get('congestion_report') or '(not available)'}

{ooc_txt}
## How a change is judged (the orchestrator does all of this, you do not)
1. Only the files above changed; constraint files may not add timing exceptions
   (set_false_path, set_multicycle_path, set_max_delay, ...) or change clocks.
2. Verilator lint; bit-exact RTL tests against the instruction-set simulator, also on the
   board's AXI memory path -- results must stay IDENTICAL (same fp32 rounding, same order of
   operations, docs/isa.md; latency may change, handshake/grant contracts may not break).
3. A Qwen3 decode-token proxy on the RTL may not get more than 0.5% slower (in cycles);
   cycles count in the score below (a clock gain spent on cycles is no gain).
4. {"The component alone, out of context in Vivado: its fmax must rise by 0.5% or more, or its area_eq shrink by 1% or more at no more than 1% lower fmax, or it must stay within 1% if it already clears the target (then the gain must come from the paths around it)." if ooc else "(no out-of-context step for this component)"}
5. The full board build at {target_mhz:g} MHz, with hold met, no router congestion (Route 35-447)
   and no block RAM read-address collision (Synth 8-6430), is accepted for
   - timing: core clock fmax +0.5% or more (or WNS +0.05 ns or more) at no more than +1% area, or
   - area: area_eq -0.5% or more with WNS no worse than -0.05 ns,
   where area_eq = LUT + LUTRAM + 0.5 FF + 40 DSP + 80 BRAM36 of the whole board, and the score
   (fmax change - area_eq change - cycle change) is positive. Timing wins count fmax minus
   cycles. A winner is confirmed by a second full build with another placement, which must
   pass too. One or two full builds run per round (one to two hours each), so aim at the
   worst paths above or at a large block of logic, not at small wins.

## Focus for this slot
{category}

## History
Recent outcomes:
{recent}

Lessons from earlier rounds:
{lessons}

## Source
{_src(wt, files) if comp.get("source") != "paths" else _src_paths(files)}

## Instructions
Read the code (and docs/isa.md, the neighbouring units in rtl/ if you need the protocols). Then
WRITE a file HYPOTHESIS.md at the repository root (your current directory) with:
  # <short title>
  Category: <pipeline|logic depth|fanout and placement|memories and DSPs>
  Motivation: which of the worst paths above this removes and why (cite them)
  Change: exactly what to change, where, and why results stay bit-identical, and its cycle
          cost (the perf proxy may not get more than 0.5% slower; cycles count in the score)
  Expected: estimated effect on the full design's WNS / fmax, and on area
  Risks: what could break
Do not modify any other file in this phase."""


def implement_prompt(comp: dict, wt: Path) -> str:
    return f"""You are implementing an approved hardware change in openTPU.

Read HYPOTHESIS.md in your current directory and implement it by editing ONLY these files:
{', '.join(expand(wt, comp['allowed']))}

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
