"""warm35x2.py [N] [LB]: warm35.py for chat35x2.py's image (both BCU1525s as one machine): every attention bucket's generate
programs and resident decode step, with their wait entries, into the program cache (OTPU_PROG_CACHE: the chat's), in N
processes (default 4) at nice 19; with LB, buckets 1 .. LB's first, then the layer-major prompt's runs of buckets 1 ..
LB (CHAT35_LM rows a run, default 2: Engine._layer_run's, through the program cache at S > 1, which has no compile
workers; CHAT35_LOOP=1: their loops, Engine.layer_loop) and their wait entries, then the other buckets'; an image of
N sequences (otpu-serve35x2 --slots N: batch35.prefill_slot's runs at each slot) every slot's runs. A dense model
(CHAT35_MODEL=lfm2-2.6b): after the buckets every bucket's prompt runs (warm_prompts), the prompt's only programs but
decode's, and the buckets' loop for a request's temperature 0 (dense_samplers); CHAT35_NOEVICT=1: no cache scan on each write, a warm below the cap's (progcache._evict globs every file). The
arguments are gen_warm.start's, kept by chat35x2.py at a chat's start (warm35x2.pkl beside this file; CHAT35_AHEAD=1:
without a card). Host compiles only, no card: run it from the tree's root with PYTHONPATH=. and the chat's environment
(w35x2.sh)."""
import multiprocessing as mp
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import warm35 as W5             # noqa: E402  (its run: a process's buckets)


def layer_keys(spec, b: int, R: int, loop: bool = False) -> list:
    """Engine._prefill_layers' runs in bucket b (a part of R-row runs: the embed run, every layer's R- and 1-row runs;
    bucket 1 also the rows before conv_k - 1 at compile-time positions), as warm_layers' in chat35.py. loop
    (CHAT35_LOOP, Engine.layer_loop): those runs' loops instead (prefill_loop.loop_key; a lone run a loop of one)."""
    L, K = spec.layers, getattr(spec, "conv_k", 1)
    keys = [(-1, b, 1, True)] + [(li, b, n, True) for li in range(L) for n in dict.fromkeys((R, 1))]
    if loop and R > 1:
        keys = [("L",) + k[:3] for k in keys]
    if b == 1:
        i = 0
        while i < K - 1:
            n = min(R, K - 1 - i)
            keys += [(li, None, n, True, i) for li in range(L)]
            i += n
    return keys


def run_layers(k: int, n: int, args: tuple, nb: int, R: int, loop: bool = False) -> None:
    """A process's part: buckets k + 1, k + 1 + n, ... of 1 .. nb."""
    os.nice(19)
    warm_layers(args, range(1 + k, nb + 1, n), R, f"[{k}]", loop)


def warm_layers(args: tuple, buckets, R: int, tag: str = "", loop: bool = False) -> None:
    """The layer-major runs of `buckets` (layer_keys, and the head's) into the program cache, with their entries; at each
    of the image's `batch` slots (prefill_slot's runs: Engine.prefill_layers(slot=), the slot in the cache's what)."""
    (spec, cfg, cap, batch, rows, image_kw, stream_fill, layout, block, conv_lo, chain, debug, split, waitc,
     samps) = args
    from opentpu import isa as I
    from opentpu import progcache as PC
    from opentpu import waits as WC
    from opentpu.llm import qwen3 as Q
    Q._worker_init(spec, cfg, cap, batch, rows, block, image_kw, stream_fill)
    image = Q._WORKER[0]
    t0, done, buckets = time.time(), 0, list(buckets)
    for b in buckets:
        made = 0
        for key, slot in [(k, s) for s in range(batch) for k in layer_keys(spec, b, R, loop) + (["head"] if b == 1
                                                                                                    else [])]:
            what = Q.layer_what(key, block, slot=slot)          # (Engine._layer_run's)
            if not PC.has(layout, what):
                made += 1
            progs, _ = PC.get(layout, what, lambda key=key, slot=slot: Q.layer_programs(image, key, block, slot=slot))
            if waitc:               # (BoardBackend.prepare's, a slice's program at a time)
                for p in progs:
                    WC.entries(I.assemble(p), cfg.D, cfg.S, prog=p, aw=cfg.AW, win=cfg.WC_WIN)
        done += 1
        if done % 8 == 1 or b == buckets[-1]:
            print(f"warm35x2 layers{tag}: bucket {b} ({made} compiled) {time.time() - t0:.0f} s", flush=True)


def run_prompts(part: list, args: tuple) -> None:
    """A process's part of the buckets (prompt_parts)."""
    os.nice(19)
    warm_prompts(args, part, f"[{part[0]}]")


def prompt_parts(args: tuple, size: int = 8) -> list:
    """Every bucket's prompt runs in parts of `size` buckets, a new process a part: an engine keeps what prefill.programs
    loaded (~130 MB a bucket at every R). Bucket 0: a new context's compile-time runs."""
    from opentpu.llm import generate as G
    nb = G.buckets(args[2], args[8])
    return [[0]] + [list(range(b, min(b + size, nb + 1))) for b in range(1, nb + 1, size)]


def prompt_engine(args: tuple):
    """What prefill's programs ask of the Engine, without a card: the image and layout the chat's compile (gen_warm.start's
    arguments), the program cache, and on a card that takes them the wait entries of each program as
    BoardBackend.prepare's."""
    from types import SimpleNamespace as NS
    from opentpu import isa as I
    from opentpu import progcache as PC
    from opentpu import waits as WC
    from opentpu.llm import qwen3 as Q
    (spec, cfg, cap, batch, rows, image_kw, stream_fill, layout, block, conv_lo, chain, debug, split, waitc,
     samps) = args
    Q._worker_init(spec, cfg, cap, batch, rows, block, image_kw, stream_fill)

    def prepare(progs):
        for p in progs if waitc else ():
            WC.entries(I.assemble(p), cfg.D, cfg.S, prog=p, aw=cfg.AW, win=cfg.WC_WIN)
    return NS(image=Q._WORKER[0], spec=spec, cfg=cfg, cap=cap, block=block, layout=layout, prog_cache=True, tile_runs=None,
              backend=NS(prepare=prepare), _host_rows=None, _tokens_kw=lambda part: {"tokens": [int(t) for t in part]},
              cached=lambda what, compile: PC.get(layout, what, compile))


def warm_prompts(args: tuple, buckets, tag: str = "") -> None:
    """prefill's runs of `buckets` into the program cache, with their entries: the bucket's fit (its fact) and the P and L
    runs of every size a prompt can ask for (the split cuts a run at the prompt's and the bucket's end). Bucket 0: a new
    context's first runs, at compile-time positions."""
    from opentpu.llm import prefill as PF
    eng, t0 = prompt_engine(args), time.time()
    K, buckets = PF.conv_k(eng), list(buckets)
    for i, b in enumerate(buckets):
        R, covered = PF._fit(eng, b)
        for r in range(R, 0, -1):
            for kind, hidden in PF._kinds(eng):
                for p in range(K - 1) if not b else [max((b - 1) * eng.block, K - 1)]:
                    PF.programs(eng, p, r, kind, hidden)
        if i % 16 == 0 or b == buckets[-1] or not covered:
            print(f"warm35x2 prompts{tag}: bucket {b}: R {R}{'' if covered else ', not covered'}; {time.time() - t0:.0f} s",
                  flush=True)


def phase(target, *a) -> None:
    """A phase's process: `target`, with CHAT35_NOEVICT no eviction."""
    if os.environ.get("CHAT35_NOEVICT") == "1":
        from opentpu import progcache as PC
        PC._evict = lambda d: None
    target(*a)


def main() -> int:
    return warm(HERE / "warm35x2.pkl", "warm35x2")


def dense_samplers(args: tuple) -> tuple:
    """args with one more of the server's samplers: a request's temperature 0 under its default penalty takes the
    device's loop at top-k 1 (Worker's G.Sampling(**params), its own key), or each bucket's program would compile on the
    chat's thread. Any other sampler a request asks for (another top-k, no penalty) still does."""
    from opentpu.llm import generate as G
    samps = list(args[14])
    for s in filter(None, args[14]):
        z = G.Sampling(0.0, 1, 1.0, s.penalty)
        if s.pen and all(o is None or o.key != z.key for o in samps):
            samps.append(z)
    return args[:14] + (samps,)


def warm(pkl: Path, name0: str) -> int:
    """main's: [N] [LB] from the command line, gen_warm.start's arguments from pkl (warm35.py's too, with LB)."""
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    lb = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    R = int(os.environ.get("CHAT35_LM") or 2)
    loop = os.environ.get("CHAT35_LOOP") == "1"
    args = pickle.loads(pkl.read_bytes())
    if not getattr(args[0], "moe", None):
        args = dense_samplers(args)
    from opentpu import progcache as PC
    print(f"{name0}: {n} processes, S {args[1].S}, cap {args[2]}, block {args[8]}, chain {args[10]}, waitc {args[13]}, "
          f"samplers {[None if s is None else s.key for s in args[14]]}, layer runs of buckets 1-{lb} ({R} rows"
          f"{', loops' if loop else ''}), cache {PC.cache_dir()}", flush=True)
    t0, bad = time.time(), []
    ctx = mp.get_context("spawn")
    phases = [(W5.run, (1, lb), name0), (run_layers, (lb, R, loop), f"{name0}-layers"),
              (W5.run, (lb + 1,), name0)] if lb else [(W5.run, (), name0)]
    for target, extra, name in phases:
        ps = [ctx.Process(target=phase, args=(target, k, n, args) + extra, name=f"{name}-{k}") for k in range(n)]
        for p in ps:
            p.start()
        for p in ps:
            p.join()
        bad += [p.exitcode for p in ps if p.exitcode]
        print(f"{name}: done in {time.time() - t0:.0f} s, exit codes {[p.exitcode for p in ps]}", flush=True)
    if not getattr(args[0], "moe", None):
        with ProcessPoolExecutor(n, mp_context=ctx, max_tasks_per_child=1) as pool:
            jobs = [pool.submit(phase, run_prompts, part, args) for part in prompt_parts(args)]
            errs = [j.exception() for j in jobs]
        bad += list(filter(None, errs))
        print(f"{name0}-prompts: done in {time.time() - t0:.0f} s, {len(jobs)} parts, failed {[repr(e) for e in errs if e]}",
              flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
