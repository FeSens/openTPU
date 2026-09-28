"""One card session measuring the host's per-token path (docs/host.md, "Host time per token"):
streamed logits against reading them after the run, for each model, with the host's critical
path split into its steps.

    python3 tools/host_path_card.py [--models lfm2,qwen3] [--tokens 96] [--out DIR]
                                    [-- extra decode_profile arguments, e.g. --wformat fp4]

For each model it runs tools/decode_profile.py four times on the card (holding the device lock
for each; OTPU_LOCK_WAIT applies): greedy with and without streamed logits -- the two replies
must be the same tokens (streaming changes when the host reads the logits, not what it reads)
-- then sampled (the model's chat defaults) with and without. It prints one table per model:
the host's critical path (HALTED seen -> next RUN) per item, the poll overshoot, wall and
device tokens/s; the JSON of every run goes to DIR (default build/host_path_<time>).
Exit status 1 when a greedy pair differs or a run fails.

The first run is the streamed greedy one with 16 tokens: streaming reads and writes the card's
DRAM while the accelerator runs, which no earlier tool did; if it fails, the script stops
there (the card then needs the usual recovery: docs/host.md).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ITEMS = ["io-write", "compile-wait", "prog-upload", "imem-load", "start", "counters",
         "logits-tail", "logits-read", "status", "sample", "detok", "ui"]


def profile(model, out: Path, name: str, args: list[str]) -> dict:
    j = out / f"{model}_{name}.json"
    cmd = [sys.executable, str(ROOT / "tools/decode_profile.py"), "--model", model,
           "--json", str(j)] + args
    print("$ " + " ".join(cmd[1:]), flush=True)
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    (out / f"{model}_{name}.txt").write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        print(r.stdout[-3000:] + r.stderr[-3000:])
        raise SystemExit(f"{model} {name}: decode_profile failed (exit {r.returncode})")
    print("\n".join(l for l in r.stdout.splitlines() if l.startswith(("host critical",
                                                                       "streamed", "wall"))))
    return json.loads(j.read_text())


def table(model: str, runs: dict) -> str:
    names = list(runs)
    w = max(len(n) for n in names) + 2
    lines = [f"{model}: ms per token on the host's critical path (HALTED seen -> next RUN)",
             f"{'':<14}" + "".join(f"{n:>{w}}" for n in names)]
    for it in ITEMS:
        v = [runs[n]["ms"].get(it, {}).get("critical", 0.0) for n in names]
        if any(v):
            lines.append(f"{it:<14}" + "".join(f"{x:>{w}.3f}" for x in v))
    for key, label in (("critical_ms", "critical path"), ("overshoot_ms", "poll overshoot"),
                       ("device_ms", "device"), ("wall_tok_s", "wall tok/s"),
                       ("dev_tok_s", "device tok/s")):
        lines.append(f"{label:<14}" + "".join(f"{runs[n][key]:>{w}.3f}" for n in names))
    return "\n".join(lines)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", default="lfm2,qwen3")
    ap.add_argument("--tokens", type=int, default=96)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    out = Path(a.out or ROOT / "build" / time.strftime("host_path_%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    ok = True
    for i, m in enumerate(a.models.split(",")):
        if i == 0:                                      # the first streamed run: short
            profile(m, out, "smoke", ["--greedy", "--tokens", "16"] + extra)
        tk = ["--tokens", str(a.tokens)] + extra
        runs = {"greedy": profile(m, out, "greedy", ["--greedy"] + tk),
                "greedy-nostream": profile(m, out, "greedy-nostream",
                                           ["--greedy", "--no-stream"] + tk),
                "sampled": profile(m, out, "sampled", tk),
                "sampled-nostream": profile(m, out, "sampled-nostream", ["--no-stream"] + tk)}
        same = runs["greedy"]["reply_ids"] == runs["greedy-nostream"]["reply_ids"]
        ok &= same
        rep = table(m, runs) + ("\ngreedy replies identical with and without streaming: "
                                f"{'yes' if same else 'NO'} ({len(runs['greedy']['reply_ids'])} "
                                "tokens)")
        (out / f"{m}_table.txt").write_text(rep + "\n")
        print(rep, flush=True)
    print(f"results in {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
