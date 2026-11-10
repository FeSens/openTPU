"""serve35.py: otpu-serve (opentpu/serve/api.py: the OpenAI-style API, /metrics, the dashboard) over the engine
otpu-chat35 chats with (chat35.open_engine: Qwen3.6-35B-A3B resident on one BCU1525, its whole 262,144-token
context, the card image from the cache, prompts layer by layer with CHAT35_LM). One sequence: requests run in turns
(the scheduler's worker), each context saved to the KV store in host RAM (DeltaNet states, convolution windows and
the attention layers' KV pages) and written back for its next turn; a chat's next turn feeds only its new tokens. The
layer runs of the context's bucket are warmed while the queue is empty (chat35's warm_layers, as after a chat's
turn). The card's telemetry (otpu-smi's default reads, every --telemetry seconds) feeds /metrics and the dashboard.
Run from the tree's root with PYTHONPATH=. and the entry's environment, the card up and held: otpu-serve35 does all of
it (docs/dashboard.md)."""
import argparse
import os
import signal
import socket
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = os.environ.get("CHAT35_MODEL") or "qwen36-35b"      # (chat35x2.py's: the entry's environment)
KV_DISK_GB = 64.0               # the KV store's disk tier (the contexts and prefixes kept across restarts)
# ... on the data disk (BCU_DATA, otpu-serve35's; SERVE35_KV_DIR names it outright)
KV_DIR = os.environ.get("SERVE35_KV_DIR") or os.path.join(
    os.environ.get("BCU_DATA") or os.path.expanduser("~/otpu-build/data"), "kv35")
KV_FREE_GB = 32.0               # ... at most what leaves this much free there


def parse(argv=None, prog: str = "otpu-serve35", description: str = __doc__.split("\n")[0]):
    ap = argparse.ArgumentParser(prog=prog, description=description)
    ap.add_argument("--host", default="127.0.0.1",
                    help="the address to listen on (0.0.0.0: every interface; default this host "
                         "only: ssh -L 8000:127.0.0.1:8000 HOST)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--api-key", default=os.environ.get("SERVE35_API_KEY") or None,
                    help="require 'Authorization: Bearer <key>' (the dashboard: /#key=<key>; "
                         "default $SERVE35_API_KEY: otpu-serve35 passes an OTPU_API_KEY there, out "
                         "of the image cache's key)")
    ap.add_argument("--served-model-name", default=MODEL,
                    help="the model's name in the API")
    ap.add_argument("--kv-ram-gb", type=float, default=4.0,
                    help="host RAM for saved contexts (a 28k-token context is ~0.6 GB)")
    ap.add_argument("--kv-disk-gb", type=float, default=KV_DISK_GB,
                    help="disk for saved contexts past the RAM, kept across restarts (0: none; "
                         f"at most what leaves {KV_FREE_GB:g} GB free on its filesystem)")
    ap.add_argument("--kv-dir", default=KV_DIR,
                    help="the disk tier's directory (default $SERVE35_KV_DIR, else kv35 on $BCU_DATA)")
    ap.add_argument("--prefix-block", type=int, default=2048, metavar="N",
                    help="a prompt's prefix kept in the KV store every N positions and at its "
                         "last message boundary as it is fed: another conversation sharing it "
                         "(opencode's tools and system prompt) restores it (0: none)")
    ap.add_argument("--max-queue", type=int, default=64)
    ap.add_argument("--max-running", type=int, default=4,
                    help="requests that run in turns at once (the others queue)")
    ap.add_argument("--quantum", type=int, default=32,
                    help="tokens a request decodes per turn while others wait")
    ap.add_argument("--slots", type=int, default=1, metavar="N",
                    help="sequences on the card (otpu-chat --slots: each its KV and DeltaNet state): N > 1 "
                         "decodes up to N requests together in batched steps (opentpu/serve/batching35.py; an "
                         "image of its own in the image cache)")
    ap.add_argument("--cap", type=int, metavar="TOKENS",
                    help="a sequence's context at most (default 262,144; with --slots N 262,144 / N: every slot's "
                         "KV cache below 4 GiB, 2 x 262,144 needs 5,965 MiB of the 3,840 there)")
    ap.add_argument("--think", action="store_true",
                    help="thinking mode by default (a request's chat_template_kwargs override it)")
    ap.add_argument("--reasoning-parser", choices=["qwen3"])
    ap.add_argument("--preserve-thinking", action=argparse.BooleanOptionalAction, default=True,
                    help="every assistant turn rendered with its thinking, as the model wrote it (Qwen3.6's "
                         "preserve_thinking): a chat's or an agent's context only grows, its next message fed "
                         "alone (--no-preserve-thinking: the earlier turns' thinking dropped, as the template's "
                         "default; a request's chat_template_kwargs override it)")
    ap.add_argument("--telemetry", type=float, default=2.0, metavar="SEC",
                    help="the card's telemetry every SEC seconds (0: off)")
    ap.add_argument("--telemetry-i2c", action="store_true")
    ap.add_argument("--no-warm-idle", action="store_true",
                    help="no layer runs warmed while the queue is empty")
    return ap.parse_args(argv)


def kv_disk(a) -> None:
    """--kv-disk-gb within the disk: what leaves KV_FREE_GB free on --kv-dir's filesystem,
    counting the store's files there already (an earlier server's); 0 under 1 GB."""
    if a.kv_disk_gb <= 0:
        return
    import shutil
    d = Path(a.kv_dir)
    probe = next((p for p in (d, *d.parents) if p.exists()), Path("/"))
    have = sum(f.stat().st_size for f in d.rglob("*.bin")) if d.exists() else 0
    room = (shutil.disk_usage(probe).free + have) / 2**30 - KV_FREE_GB
    if room < a.kv_disk_gb:
        print(f"otpu-serve35: {probe} has room for {max(room, 0):.1f} GB of saved contexts "
              f"(leaving {KV_FREE_GB:g} GB free): --kv-disk-gb {a.kv_disk_gb:g} -> "
              f"{max(room, 0) if room >= 1 else 0:.1f}", flush=True)
        a.kv_disk_gb = round(room, 1) if room >= 1 else 0.0


def _port_free(host: str, port: int) -> None:
    """Refuse a port in use now, not after the 35B's minutes of loading."""
    s = socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
    except OSError as e:
        raise SystemExit(f"otpu-serve35: {host}:{port}: {e.strerror}") from None
    finally:
        s.close()


def main(argv=None, open_engine=None, dev: str | None = None, label: str = "otpu-serve35",
         card_names: str | None = None, description: str | None = None) -> int:
    """otpu-serve35's server; serve35x2.py's with its engine (open_engine: chat35.open_engine's
    result, or None to exit), its cards (dev: "/dev/xdmaA+/dev/xdmaB", card_names theirs to show)
    and its label. dev None: the engine's card (OTPU_DEV, otpu-serve35's: BCU_CARD's)."""
    dev = dev or os.environ.get("OTPU_DEV") or "/dev/xdma0"
    a = parse(argv, label, description or __doc__.split("\n")[0])
    _port_free(a.host, a.port)
    kv_disk(a)
    # SIGTERM (otpu-serve35's stop) as Ctrl-C, once: the server's finally runs (the worker stopped, the store persisted)
    def term(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, term)
    sys.path.insert(0, str(HERE))
    import chat35
    import opentpu.host.chat as C
    from opentpu.host.runstate import busy_exits
    from opentpu.serve import api

    def run():
        cap = a.cap or (chat35.CAP // a.slots if a.slots > 1 else None)
        o = (open_engine or chat35.open_engine)((["--slots", str(a.slots)] if a.slots > 1 else [])
                                                + (["--cap", str(cap)] if cap else []))
        if o is None:
            return 0
        eng = o.eng
        path = C.model_dir(MODEL)
        ns = argparse.Namespace(temperature=None, top_k=None, top_p=None, repetition_penalty=None)
        defaults = C.sampling(eng.spec, ns, path)
        weights = C.weights_label(eng.image, eng.image.wformat)
        sa = api.parser().parse_args([])
        sa.model, sa.backend, sa.dev, sa.cap = MODEL, "board", dev, eng.cap
        sa.card_names = card_names
        for k in ("host", "port", "api_key", "served_model_name", "kv_ram_gb", "kv_disk_gb", "kv_dir",
                  "max_queue", "max_running", "quantum", "think", "reasoning_parser", "telemetry",
                  "telemetry_i2c", "prefix_block", "preserve_thinking"):
            setattr(sa, k, getattr(a, k))
        warm = None
        if eng.layer_major and not a.no_warm_idle and getattr(o, "warm_layers", None) is not None:
            def warm():
                o.warm_layers(eng)
        return api.run(sa, eng, o.tok, eng.spec, path, defaults, weights, on_idle=warm, label=label)
    try:
        return busy_exits(run)() or 0
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":     # (the Engine's spawned compile workers import this module)
    sys.exit(main())
