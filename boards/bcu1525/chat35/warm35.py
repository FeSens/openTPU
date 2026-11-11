"""warm35.py [N] [LB]: the 35B chat's programs for every attention bucket of its image into the program cache, ahead of the
chats: each bucket's generate programs (gen_warm.warm: greedy and the chat's sampler, with their wait entries) and its
resident decode step (the prompt's token steps: Engine._decode's, as its compile workers write it) with its wait
entries, in N processes (default 4) at nice 19, bucket b in process (b - 1) % N; bucket 1's process also a new
context's first runs' wait entries (start_waits). The arguments are gen_warm.start's, kept by chat35.py at a chat's
start (warm35.pkl beside this file). Host compiles only, no card: run it between chats from the tree's root with
PYTHONPATH=. and the chat's environment (OTPU_PROG_CACHE: the chat's program cache). With LB, warm35x2.py's phases:
buckets 1 .. LB, then their layer-major runs (CHAT35_LM, CHAT35_LOOP; the engine then loads them instead of compiling
them in its workers), then the other buckets."""
import multiprocessing as mp
import os
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def decode_waits(layout, what, cfg) -> None:
    """The wait entries of the resident decode step `what` as the engine reads it back from the disk (decoded), kept
    there (waits.entries: gen_warm.warm's for the generate programs)."""
    from opentpu import isa as I
    from opentpu import progcache as PC
    from opentpu import waits as WC
    k = PC.key(layout, what)
    v = PC._load(PC.cache_dir() / k[:2] / f"{k}.npz")
    if v is None:
        return
    for part in v[0] if isinstance(v[0], tuple) else (v[0],):
        for p in part:
            WC.entries(I.assemble(p), cfg.D, cfg.S, prog=p, aw=cfg.AW, win=cfg.WC_WIN)


def prefix_tokens(conv_lo: int, path=None) -> list:
    """The tokens before the first run-time position (conv_lo) of a new context as the chat template renders it: a first
    user message, and one after a system message (otpu-serve35's requests with one). From the tokenizer at `path`
    (default: CHAT35_MODEL's, else chat35's model, FIXED); none when it does not load here."""
    import chat35 as C35
    from opentpu.llm import model_dir
    try:
        from transformers import AutoTokenizer
        model = os.environ.get("CHAT35_MODEL") or C35.FIXED[C35.FIXED.index("--model") + 1]
        tok = AutoTokenizer.from_pretrained(path or model_dir(model))
    except Exception as e:
        print(f"warm35: no tokenizer ({e}): a context's first steps not warmed", flush=True)
        return []
    out = []
    for msgs in ([{"role": "user", "content": "x"}],
                 [{"role": "system", "content": "x"}, {"role": "user", "content": "x"}]):
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        out.append(tuple(list(ids["input_ids"] if hasattr(ids, "keys") else ids)[:conv_lo]))
    return out


def start_waits(image, block: int, conv_lo: int, prefixes, snapshots: int) -> int:
    """The wait entries of a new context's first runs, kept (waits.entries; else BoardBackend.prepare or start computes
    each at run time, 2-3 s on the 35B): the steps before conv_lo (Engine.step's per-position programs: an image with
    lookup tables compiles the token into its program, so each prefix's tokens, prefix_tokens) and the copies of
    Engine(snapshots=)'s device snapshots (a reply's start saved, the next turn's rollback restoring it). Returns the
    programs."""
    from opentpu import isa as I
    from opentpu import waits as WC
    from opentpu.llm import snapshot as SN
    c = image.cfg
    steps = sorted({(p, t[p]) for t in prefixes for p in range(min(conv_lo, len(t)))}) if image.lookup else []
    runs = [image.compile_step(p, block, tok=int(t)) for p, t in steps]
    if snapshots and SN.stateful(image):
        sn = SN.Snapshots(image, c, snapshots, SN.Snapshots.area(image, c, snapshots))
        if sn.base is not None:
            runs += [ps for j in range(snapshots) for save in (True, False) for ps in sn._programs(save, j, 0)]
    for progs in runs:                  # (BoardBackend._entries': each slice's program)
        for p in progs[:c.S]:
            WC.entries(I.assemble(p), c.D, c.S, prog=p, aw=c.AW, win=c.WC_WIN)
    return len(runs)


def run(k: int, n: int, args: tuple, lo: int = 1, hi: int | None = None) -> None:
    """A process's part of buckets lo .. hi (default every bucket): lo + k, lo + k + n, ..."""
    (spec, cfg, cap, batch, rows, image_kw, stream_fill, layout, block, conv_lo, chain, debug, split, waitc,
     samps) = args
    os.nice(19)
    from opentpu import progcache as PC
    from opentpu.host import gen_warm as GW
    from opentpu.llm import generate as G
    from opentpu.llm import qwen3 as Q
    Q._worker_init(spec, cfg, cap, batch, rows, block, image_kw, stream_fill)
    image = Q._WORKER[0]
    nb = G.buckets(cap, block)
    hi = nb if hi is None else min(hi, nb)
    t0, done = time.time(), 0
    if waitc and lo + k == 1:       # bucket 1's process: a new context's first runs too
        from opentpu.host import chat as C
        made = start_waits(image, block, conv_lo, prefix_tokens(conv_lo), C.SNAPSHOTS)
        print(f"warm35[{k}]: a context's first runs: {made} programs' wait entries {time.time() - t0:.0f} s",
              flush=True)
    for b in range(lo + k, hi + 1, n):
        made = 0
        for samp in samps:
            made += GW.warm(image, layout, block, conv_lo, b, chain, samp, debug, split, waitc)
        lo = max((b - 1) * block, conv_lo)
        what = ("decode", b, lo) + (("fill",) if stream_fill else ())
        if not PC.has(layout, what):
            # at S > 1 every slice's programs, as the engine compiles them itself (no compile workers there):
            # _worker_decode's words are slice 0's, which the cards run with HALT on the other slices
            PC.get(layout, what, (lambda: image.compile_decode(b, lo, block)) if image.cfg.S > 1 else
                   (lambda: Q._worker_decode(b, lo)))
            made += 1
        if waitc:                   # its wait entries too (BoardBackend.prepare's: the step's 2.6 s)
            decode_waits(layout, what, image.cfg)
        done += 1
        if done % 16 == 1 or b + n > hi:
            print(f"warm35[{k}]: bucket {b}/{nb} ({made} compiled) {time.time() - t0:.0f} s", flush=True)


def main() -> int:
    if len(sys.argv) > 2:           # [N] LB: warm35x2.py's phases on this image (the layer runs: the engine loads them)
        sys.path.insert(0, str(HERE))
        import warm35x2 as WX
        return WX.warm(HERE / "warm35.pkl", "warm35")
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    args = pickle.loads((HERE / "warm35.pkl").read_bytes())
    from opentpu import progcache as PC
    print(f"warm35: {n} processes, cap {args[2]}, block {args[8]}, chain {args[10]}, waitc {args[13]}, "
          f"samplers {[None if s is None else s.key for s in args[14]]}, cache {PC.cache_dir()}", flush=True)
    t0 = time.time()
    ps = [mp.get_context("spawn").Process(target=run, args=(k, n, args), name=f"warm35-{k}") for k in range(n)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    bad = [p.exitcode for p in ps if p.exitcode]
    print(f"warm35: done in {time.time() - t0:.0f} s, exit codes {[p.exitcode for p in ps]}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
