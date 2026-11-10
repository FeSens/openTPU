"""serve35x2.py: otpu-serve35's server (serve35.py: the OpenAI-style API, /metrics, the dashboard) over otpu-chat35x2's
engine (chat35x2.py: Qwen3.6-35B-A3B on both BCU1525s as one machine, bcu1525-x2:st16, four slices in tensor parallel
over the QSFP28 link, card 0 and card 1: CHAT35_DEV "/dev/xdmaA+/dev/xdmaB"). The same worker, KV store
and dashboard: each context's pages and state are every slice's (two a card), saved and written back through the
staging area at the same addresses on both cards (kvio over CardsTransport). The cards' telemetry (telemetry.
CardsSampler: smi's default reads on each card, its own BMC bridge) feeds /metrics and the dashboard as one machine (the
hottest die, the governors' RUN together) and each card by name. Run from the tree's root with PYTHONPATH=. and the
entry's environment, both cards up and held: otpu-serve35x2 does all of it."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
NAMES = os.environ.get("CHAT35_NAMES") or "card0,card1"      # card 0, card 1 (otpu-serve35x2's: BCU_ENV's names)


def open_engine(argv=()):
    """chat35x2's engine, for the server: chat35x2.open_engine when it has one; else its main() with FIXED, `argv` and
    --prompt - over a stand-in for otpu-chat's Chat that takes what main opens and asks nothing (chat35.open_engine's
    way). Returns (eng, tok, pick, clock_mhz) with warm_layers None (at S > 1 the layer runs come from the program
    cache: warm35x2.py) and the stand-in's class (stub)."""
    sys.path.insert(0, str(HERE))
    import chat35x2 as X2
    if hasattr(X2, "open_engine"):
        return X2.open_engine(argv)
    import opentpu.host.chat as C
    got = {}

    class Opened:
        use_mtp = False

        def __init__(self, eng, tok, think, pick, max_new, clock_mhz=0.0):
            got.update(eng=eng, tok=tok, pick=pick, clock_mhz=clock_mhz)

        def ask(self, text, *a, **kw):  # (chat35x2 wraps Chat.ask: its link recovery)
            pass

        def attach(self, src):
            raise ValueError("no images: the 35B has no vision side")

        def ask_plain(self, text):
            pass
    chat, argv0 = C.Chat, sys.argv
    C.Chat = Opened
    sys.argv = [argv0[0], *argv, "--prompt", "-"]
    try:
        X2.main()
    finally:
        C.Chat, sys.argv = chat, argv0
    if "eng" not in got:
        raise SystemExit("serve35x2: chat35x2 opened no engine")
    return SimpleNamespace(**got, warm_layers=None, stub=Opened)


def main(argv=None) -> int:
    sys.path.insert(0, str(HERE))
    import serve35
    dev = os.environ.get("CHAT35_DEV", "")
    if "+" not in dev and not {"-h", "--help"} & set(sys.argv[1:] if argv is None else argv):
        raise SystemExit("otpu-serve35x2: CHAT35_DEV must name both cards, card 0 first: /dev/xdmaA+/dev/xdmaB")
    return serve35.main(argv, open_engine=open_engine, dev=dev, label="otpu-serve35x2", card_names=NAMES,
                        description=__doc__.split("\n")[0])


if __name__ == "__main__":     # (spawned workers import this module)
    sys.exit(main())
