"""Card check of a chat's KV cache across turns (docs/host.md, "The KV cache across turns"): a
scripted conversation through otpu-chat's Chat on --backend board, greedy, each turn rolled
back to where its rendering parts from what the device holds (Engine.rollback; Qwen3.5, LFM2,
Gemma 4 past its rings' slack: from the snapshot of their state each turn takes at its reply's
start, copied on the device) and fed the rest; then each turn's rendering again from position
0, as the chat fed it before (the restart), whose logits must equal the turn's first logits bit
for bit (int8 weights: any split of a prompt into runs gives the same rows), and whose greedy
reply the turn's. Per turn: the rendering's tokens, the tokens reused and fed, the prefill's
runs, where a restore took the context (restored at), device Mcycles and wall seconds against
the restart's, the snapshot's and the restore's device cycles (the ISA simulator counts no
cycles: tokens and runs). --mtp: the turns' prefill is MTP's (its rows and
the MTP layer's runs), the restart a plain prefill of the rows.

    otpu-lock -- python3 tools/qual/chatkv.py --model qwen35 [--mtp] [--dev /dev/otpu0]
    otpu-lock -- python3 tools/qual/chatkv.py --model gemma4 --think --max-new 300 --turns 3
    python3 tools/qual/chatkv.py --model qwen3 --backend isa --turns 3 --max-new 8   (dry run)

Gemma 4 E2B-it (int8 layers, the fp4 PLE table) continues each turn's context thinking off; with
--think the template drops a reply's thinking channel (a reply cut at --max-new, its channel
open, entirely), so the turn rolls back to the reply's start, through the snapshot (restored)
once that is more than its rings' slack (257) back.

Prints one line per turn and "CHATKV PASS" (exit 0) or "CHATKV FAIL" (exit 1)."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from opentpu.host.chat import SNAPSHOTS, Chat, make_backend, sampler   # noqa: E402
from opentpu.llm import MODELS, load_spec, model_dir                  # noqa: E402
from opentpu.llm import snapshot as SN                                # noqa: E402
from opentpu.llm.qwen3 import PREFILL_ROWS, Engine, load_weights      # noqa: E402

MESSAGES = ["What is 1 + 32 * 9?",
            "And why is the ocean blue?",
            "Give me three facts about it as a short list.",
            "Shorter, please.",
            "Thanks! Summarize this chat in one sentence."]


def recording(rec: list, eng=None):
    """A greedy chat.sampler that keeps the logits of its picks (each turn's first token), and
    the engine's runs so far (len(eng.stats): the prefill's runs end there). Gemma 4's: with its
    soft cap, as Chat takes (the same argmax)."""
    base = sampler(0, 0, 1.0, None, softcap=getattr(getattr(eng, "spec", None), "softcap", None))

    def pick(logits, context=()):
        rec.append((np.array(logits, np.float32), len(eng.stats) if eng is not None else 0))
        return base(logits, context)
    for k in ("stream", "warm", "greedy", "params", "softcap", "rng"):
        setattr(pick, k, getattr(base, k))
    return pick


def sim_config(spec, cap: int, wformat: str, head_format: str | None):
    """The simulators' configuration: the card's (one slice, MCOLS 4, PAIR, DSTEP, STREAM), its
    DRAM cut to the image and the program area (board.sim_config), doubled where the
    snapshots' area does not fit beside them."""
    from dataclasses import replace

    from opentpu.host.board import sim_config as cut
    from opentpu.isasim import board_config
    cfg = cut(spec, cap, board_config(PAIR=True, DSTEP=True, STREAM=True), lookup=True,
              wformat=wformat, head_format=head_format)
    img = spec.image(replace(cfg, DRAM_BYTES=1 << 40), cap, 1, PREFILL_ROWS, wformat,
                     head_format, lookup=True)
    if SN.stateful(img) and SN.Snapshots.area(img, cfg, SNAPSHOTS) is None:
        cfg = replace(cfg, DRAM_BYTES=2 * cfg.DRAM_BYTES)
    return cfg


def _cycles(stats) -> int:
    return sum((s or {}).get("cycles", 0) for s in stats)


def _mc(c: int, on: bool) -> str:
    return f" {c / 1e6:.1f} Mcycles" if on else ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="qwen35",
                    help=f"{' or '.join(MODELS)} (models/<name>), or a checkpoint directory")
    ap.add_argument("--backend", default="board", choices=["board", "board-sim", "isa"])
    ap.add_argument("--dev", help="the card's device prefix (default: /dev/otpu0 if present, "
                    "else /dev/xdma0)")
    ap.add_argument("--cap", type=int, default=1024)
    ap.add_argument("--wformat", default="int8", choices=["int8", "fp4", "int4", "nvfp4", "mix"],
                    help="int8 (default): bit-exact in any split; with PAIR a 4-bit run of "
                         "more than MCOLS / 2 rows sums in another order (docs/prefill.md)")
    ap.add_argument("--head-format", default=None, choices=["int8", "fp4", "int4", "nvfp4"])
    ap.add_argument("--turns", type=int, default=len(MESSAGES))
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--mtp", action="store_true", help="Qwen3.5: the MTP decoder")
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    path = model_dir(a.model)
    tok, spec = AutoTokenizer.from_pretrained(path), load_spec(path)
    if a.mtp:
        from dataclasses import replace
        spec = replace(spec, mtp=True)
    if a.backend == "board":
        backend, cfg = make_backend(a.backend, spec, a.cap, a.dev, path.name, lookup=True,
                                    wformat=a.wformat, head_format=a.head_format)
    else:
        backend, cfg = "isa", sim_config(spec, a.cap, a.wformat, a.head_format)
        if a.backend == "board-sim":
            from opentpu.host.board import BoardBackend, SimTransport
            tr = SimTransport(ch_bytes=cfg.DRAM_BYTES // 2)
            backend = (lambda c, imgs: BoardBackend(c, imgs, transport=tr, model=path.name))
    kw = dict(cap=a.cap, cfg=cfg, backend=backend, wformat=a.wformat,
              head_format=a.head_format, prompt_runs=True, snapshots=SNAPSHOTS)
    if a.mtp:
        from opentpu.llm.mtp import mtp_engine
        eng = mtp_engine(spec, load_weights(path, mtp=True), **kw)
    else:
        eng = Engine(spec, load_weights(path), resident=True, **kw)
    khz = (getattr(eng.backend, "info", None) or {}).get("core_khz")
    mhz = khz / 1e3 if khz else 0.0
    cyc_on = a.backend != "isa"            # the ISA simulator counts no cycles
    rec: list = []
    chat = Chat(eng, tok, a.think, recording(rec, eng), a.max_new, clock_mhz=mhz)
    print(f"{path.name} {eng.image.wformat} head {eng.image.head_format}, {a.backend}"
          f"{f' {mhz:g} MHz' if mhz else ''}, cap {a.cap}, think {a.think}, mtp {chat.use_mtp}, "
          f"stateful {eng.stateful}, snapshots "
          f"{'none' if eng._snaps is None else 'device' if eng.snapshot_bytes else 'host'}"
          f" ({0 if eng._snaps is None else eng._snaps.nbytes / 2**20:.2f} MiB)", flush=True)
    turns = []
    for msg in MESSAGES[:a.turns]:
        k0, s0 = len(eng.stats), len(eng.snap_stats)
        reply, t = chat.ask(msg)
        ids = chat._template(history=chat.history[:-1])
        ops = {op: _cycles(x for x in eng.snap_stats[s0:] if x["op"] == op)
               for op in ("snapshot", "restore")}
        runs = rec[-1][1] - k0 if not chat.use_mtp else None
        turns.append(dict(ids=ids, t=t, logits=rec[-1][0], reply=list(chat._reply), ops=ops,
                          runs=runs))
        back = [s["pos"] for s in eng.snap_stats[s0:] if s["op"] == "restore"]
        print(f"turn {len(turns)}: {len(ids)} tokens, reused {t.reused}, fed {t.prefill_tokens}"
              f"{'' if runs is None else f' in {runs} runs'}"
              f"{' (restarted)' if t.restarted else ''}"
              f"{f' (restored at {back[0]})' if back else ''}; prefill {t.prefill_s:.3f} s"
              f"{_mc(t.prefill_cycles, cyc_on)}, TTFT {t.ttft_s or 0:.3f} s; "
              + (f"snapshot {ops['snapshot'] / 1e6:.2f} restore {ops['restore'] / 1e6:.2f} "
                 f"Mcycles; " if cyc_on else "")
              + f"reply {t.gen_tokens} tokens ({t.end})", flush=True)
    ok, tot = True, np.zeros(6)
    for i, x in enumerate(turns):                   # the restart: the rendering from 0
        eng.reset()
        k0, w0 = len(eng.stats), time.perf_counter()
        lg = eng.prefill(x["ids"])
        wall, cyc, runs = time.perf_counter() - w0, _cycles(eng.stats[k0:]), len(eng.stats) - k0
        same = np.array_equal(lg.view(np.uint32), x["logits"].view(np.uint32))
        diff = float(np.max(np.abs(lg - x["logits"]))) if not same else 0.0
        out, want = [], x["reply"]
        while len(out) < len(want):                 # the greedy reply after it
            out.append(int(np.argmax(lg)))
            if len(out) < len(want):
                lg = eng.step(out[-1])
        t = x["t"]
        good = same and out == want and not (i and t.restarted)
        ok &= good
        cyc_rb = t.prefill_cycles + sum(x["ops"].values())
        tot += [len(x["ids"]), t.prefill_tokens, cyc, cyc_rb, runs, x["runs"] or 0]
        lgs = "equal" if same else f"differ (max {diff:.3g})"
        rb_runs = "" if x["runs"] is None else f" in {x['runs']} runs"
        print(f"[{'PASS' if good else 'FAIL'}] turn {i + 1}: logits {lgs}, reply "
              f"{'equal' if out == want else 'differs'}; restart: {len(x['ids'])} tokens in "
              f"{runs} runs {wall:.3f} s{_mc(cyc, cyc_on)}, rollback: {t.prefill_tokens} "
              f"tokens{rb_runs} {t.prefill_s:.3f} s{_mc(cyc_rb, cyc_on)}", flush=True)
    runs = "" if chat.use_mtp else f"{int(tot[5])} "       # MTP: its rows and layer runs
    print(f"tokens fed {int(tot[1])} (restart {int(tot[0])}), runs {runs}(restart "
          f"{int(tot[4])})" + (f", device Mcycles {tot[3] / 1e6:.1f} (restart "
                               f"{tot[2] / 1e6:.1f})" if cyc_on else ""))
    print("CHATKV PASS" if ok else "CHATKV FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
