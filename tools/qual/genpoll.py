"""Card A/B of the generate loop's poll (BoardBackend.run_generate's GEN_POLL_FAR, opentpu/host/
board.py): the host's longest sleep while the next token is not due. The card runs the loop by
itself, so the device's tokens/s must not move with it; what can move is the wall rate (the
last token's hand-over), the first token's time and the host's CPU a token.

One process, one engine (otpu-chat's own, through its main: any card, any model the chat
takes), the same greedy turn each time from a reset context: a warm-up, then A B B A (the first
run after a load is slower; the pairs' order cancels a drift). A is --a seconds (default 0.001,
the cap before the change), B is --b (default the tree's GEN_POLL_FAR).

    otpu-lock -- python3 tools/qual/genpoll.py --model qwen35-2b [--dev /dev/xdma1]
    OTPU_TARGET=bcu1525-wg otpu-lock -- python3 tools/qual/genpoll.py --model qwen35-2b
    python3 tools/qual/genpoll.py --install DIR -- [the chat's arguments]
                    DIR holds chat35.py or chat35x2.py (the 35B's set-up of otpu-chat: its
                    install() first, its FIXED arguments before the chat's); run it as the
                    wrapper runs the chat: the card held in RUN by its bring-up, the wrapper's
                    environment (otpu-chat35's header)
    python3 tools/qual/genpoll.py --model qwen3 --backend isa --max-new 8      (dry run: the ISA
                    simulator has no run_generate, so both sides are the same code)

Prints a line per run and the pairs' means, then "GENPOLL PASS" (exit 0: the five replies equal
token for token, the device's rate of B within --tol of A's, default 1 %) or "GENPOLL FAIL"
(exit 1). The wall rate, the first token's time and the CPU are reported, not judged: they are
what the change is for."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--a", type=float, default=1e-3, help="side A's GEN_POLL_FAR, seconds")
    ap.add_argument("--b", type=float, help="side B's (default: the tree's)")
    ap.add_argument("--max-new", type=int, default=192)
    ap.add_argument("--prompt", default="Write a short story about a lighthouse keeper.")
    ap.add_argument("--tol", type=float, default=0.01, help="the device rate's tolerance")
    ap.add_argument("--install", metavar="DIR", help="chat35.py's or chat35x2.py's directory")
    a, rest = ap.parse_known_args(argv)
    rest = [x for x in rest if x != "--"]
    import opentpu.host.chat as C
    from opentpu.host import board as BD
    fixed = []
    if a.install:
        sys.path.insert(0, a.install)
        mod = __import__("chat35x2" if (Path(a.install) / "chat35x2.py").is_file() and
                         "+" in os.environ.get("CHAT35_DEV", "") else "chat35")
        if mod.install() is None:
            return 0
        fixed = list(mod.FIXED)
        dev = os.environ.get("CHAT35_DEV", "")
        if "+" in dev:
            fixed += ["--dev", dev]
    got = {}

    class Opened:                           # otpu-chat's main opens the engine; no turn there
        use_mtp = False

        def __init__(self, eng, tok, think, pick, max_new, clock_mhz=0.0):
            got.update(eng=eng, tok=tok, think=think, clock_mhz=clock_mhz)

        def attach(self, src):
            pass

        def ask_plain(self, text):
            pass
    chat_cls, C.Chat = C.Chat, Opened
    try:
        C.main(fixed + rest + ["--greedy", "--prompt", "-"])
    finally:
        C.Chat = chat_cls
    if "eng" not in got:
        raise SystemExit("genpoll: otpu-chat opened no engine")
    pick = C.sampler(0, 1, 1.0, 0, 1.0, getattr(got["eng"].spec, "softcap", None))
    chat = C.Chat(got["eng"], got["tok"], got["think"], pick, a.max_new, clock_mhz=got["clock_mhz"])
    b_far = BD.GEN_POLL_FAR if a.b is None else a.b
    print(f"genpoll: on_card {chat.on_card}; A = {a.a * 1e3:g} ms, B = {b_far * 1e3:g} ms; "
          f"{a.max_new} tokens a run")
    runs, replies = [], []
    for side in ("warm-up", "A", "B", "B", "A"):
        BD.GEN_POLL_FAR = a.a if side == "A" else b_far
        chat.reset()
        cpu = time.process_time()
        reply, t = chat.ask(a.prompt)
        cpu = time.process_time() - cpu
        n = max(1, t.gen_tokens)
        r = dict(side=side, tokens=t.gen_tokens, wall=t.decode_tok_s, dev=t.decode_dev_tok_s,
                 ttft=t.ttft_s, cpu_ms=1e3 * cpu / n)
        runs.append(r)
        replies.append(reply)
        f = lambda x, p=2: "n/a" if x is None else f"{x:.{p}f}"     # noqa: E731
        print(f"  {side:<8} {r['tokens']:4d} tokens  decode {f(r['wall'])} tok/s wall, "
              f"{f(r['dev'])} device  TTFT {f(r['ttft'])} s  host CPU {r['cpu_ms']:.2f} ms/token "
              f"(the turn's, prefill included)", flush=True)

    def mean(side, k):
        v = [r[k] for r in runs if r["side"] == side and r[k] is not None]
        return sum(v) / len(v) if v else None
    ok = len(set(replies)) == 1
    if not ok:
        print("genpoll: the replies differ between runs")
    for k, what in (("dev", "device tok/s"), ("wall", "wall tok/s"), ("ttft", "TTFT s"),
                    ("cpu_ms", "host CPU ms/token")):
        x, y = mean("A", k), mean("B", k)
        if x is None or y is None:
            print(f"  {what}: n/a")
            continue
        print(f"  {what}: A {x:.3f}  B {y:.3f}  B/A {y / x:.4f}" if x else f"  {what}: A 0")
        if k == "dev" and x and abs(y / x - 1) > a.tol:
            ok = False
            print(f"genpoll: the device's rate moved by more than {a.tol:.1%}")
    print("GENPOLL PASS" if ok else "GENPOLL FAIL")
    return 0 if ok else 1


if __name__ == "__main__":      # (the Engine's spawned compile workers import this module)
    sys.exit(main())
