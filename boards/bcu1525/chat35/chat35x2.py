"""chat35x2.py: chat35.py's chat (otpu-chat with Qwen3.6-35B-A3B resident, its whole 262,144-token context) on both
BCU1525s as one machine: bcu1525-x2:st16, four slices, two a card, in tensor parallel over the QSFP28 link (card 0
and card 1: opentpu.host.cards.CardsTransport, --dev "/dev/xdmaA+/dev/xdmaB" from CHAT35_DEV, card 0
first; otpu-chat35x2 finds them). The Engine's configuration is the target's (T.config, which the image was built with;
the cards' registers checked against it: Board.info's check_target and S / MCOLS / D here), each slice's image in its
card's DRAM. The image is built on the build host (CHAT35_BUILD=1: c35x2.sh) as chat35's, never on the card host: a cache miss
is refused. The rest is chat35.py's, its helpers imported from it: the entry as a mapped raw image (CHAT35_RAW) whose
data runs alone are written after the scrub, the lazy checkpoint, the warm pickle (warm35x2.pkl beside this file, for
warm35x2.py; CHAT35_AHEAD=1 writes it with no card and exits: Ahead), CHAT35_TIMES (the cards' link set-up timed
too) and CHAT35_PROF. CHAT35_SNAP=DIR: each reply's context in the cards' DRAM saved before and after it (snapping).
A prompt part of CHAT35_LM_MIN (64) tokens or more goes layer by layer, CHAT35_LM rows a run
(each slice's layer run its part of the tensor-parallel step's; at S > 1 there are no compile workers, so the runs
come from the program cache: warm35x2.py LB, and go to the cards as its words, opentpu.host.runwords; CHAT35_RUNWORDS=0
as the engine reads them; CHAT35_LOOP=1, chat35.py's: a layer's runs in a bucket as one run, the loops' keys warmed by
warm35x2.py with CHAT35_LOOP=1), shorter parts and every part without CHAT35_LM a token a run. A run
the link fails (cards.LinkError) ends the turn: the link is reset (CardsTransport.recover) and the chat goes on from
its last snapshot (Chat's rollback), the turn asked again once. Run from the tree's root with PYTHONPATH=. and the
entry's environment (otpu-chat35x2 sets both)."""
import os
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chat35 as C35            # noqa: E402  (its helpers: Null, _timed, _LazyWeights, raw_images, RUNS)

TARGET = "bcu1525-x2:st16"
CAP = C35.CAP
# the model (CHAT35_MODEL): the 35B, or LFM2-2.6B, a dense model whose prompts go as run-time prompt runs (docs/prefill.md),
# in every bucket (dense_prompts), CHAT35_LM 0, its one sequence's whole 128,000-token context
MODELS = {"qwen36-35b": ["--cap", str(CAP), "--wformat", "fp4", "--head-format", "int8", "--no-prompt-runs"],
          "lfm2-2.6b": ["--cap", "128000", "--wformat", "mix", "--head-format", "int8"]}
MODEL = os.environ.get("CHAT35_MODEL") or "qwen36-35b"
FIXED = ["--backend", "board", "--model", MODEL] + MODELS[MODEL]


def dense_prompts() -> None:
    """A dense model's prompts as prompt runs in every bucket, from the warmed program cache: at S > 1 there are no compile
    workers, so prefill.covers' other route (today's runs, where they would take more rows: LFM2-2.6B's bucket 250) would
    compile each run on the chat's thread, for the whole prompt."""
    from opentpu.llm import prefill as PF
    PF.covers = lambda eng, p0, P: True


def target_config():
    """The x2 machine's configuration: what the image is built with and the chat runs."""
    from opentpu import target as T
    return T.config(T.load(TARGET))


class LayerExperts:
    """W (chat35's _LazyWeights) with a layer's fused expert tensors (its mlp.experts.gate_up_proj and down_proj, each
    converted whole by qwen3.Weights, which keeps one tensor that big) kept while the build reads that layer's
    experts: at S = 4 the pool's build (moe.pool_build, global expert g = layer g // E) reads each expert once a
    slice, and every read converted both tensors again (2.8 s an expert, 8 h for the 35B's 10,240). Other tensors,
    attributes and the key's model_dir are W's; the values are W's own arrays."""

    FUSED = ("mlp.experts.gate_up_proj", "mlp.experts.down_proj")

    def __init__(self, W):
        self._W, self._keep = W, {}

    def __getitem__(self, k):
        if not k.endswith(self.FUSED):
            return self._W[k]
        v = self._keep.get(k)
        if v is None:
            layer = k[:-len(next(f for f in self.FUSED if k.endswith(f)))]
            self._keep = {n: x for n, x in self._keep.items() if n.startswith(layer)}
            v = self._keep[k] = self._W[k]
        return v

    def __iter__(self):
        return iter(self._W)

    def __len__(self):
        return len(self._W)

    def __contains__(self, k):
        return k in self._W

    def __getattr__(self, name):
        return getattr(self._W, name)


class Ahead:
    """No hardware, for CHAT35_AHEAD (the build host, before a card session): the Engine's backend as the cards' is (run
    arguments, the generate loop, wait entries: the x2 bitstream's CAPS and SEQ_WC), so that the chat's start keeps
    warm35x2.pkl as a card session's chat would (gen_warm.start's arguments) and warm35x2.py warms ahead of it."""
    args = generates = chains = True

    def __init__(self, c, images):
        self.cfg, self.waitc, self.info = c, bool(c.SEQ_WC), {}

    def close(self):
        pass


def ahead_images(cfg) -> list:
    """The image's stand-ins for CHAT35_AHEAD: a slice's DRAM of zeros each (calloc'd: no pages until written; the
    Engine's areas past the image fit), never written or read."""
    import numpy as np
    return [np.zeros(cfg.DRAM_BYTES, np.uint8) for _ in range(cfg.S)]


def back_to_snapshot(chat) -> int:
    """After a run that failed: the chat's context back to its latest snapshot at or before what it fed (the failed
    run may have written the recurrent state, so not Engine.rollback's slack), or a new context with none; returns
    the tokens kept."""
    eng = chat.eng
    snaps = getattr(eng, "_snaps", None)
    j = None if snaps is None else snaps.best(len(chat.fed))
    if j is None:
        eng.reset()
        chat.fed.clear()
        return 0
    t = snaps.at[j]
    eng._drain()
    snaps.restore(eng, j)
    snaps.drop_above(t)
    eng.pos = t
    del chat.fed[t:]
    return t


def recovering(ask):
    """Chat.ask whose turn, when a run fails on the link (cards.LinkError), resets the link (CardsTransport.recover:
    the cores stopped, LRST), takes the context back to its snapshot (back_to_snapshot) and is asked once more."""
    from opentpu.host.cards import LinkError

    def ask_recovering(self, text, *a, **kw):
        n = len(self.history)
        try:
            return ask(self, text, *a, **kw)
        except LinkError as e:
            t = getattr(getattr(self.eng.backend, "board", None), "t", None)
            if not hasattr(t, "recover") or self.use_mtp:
                raise
            print(f"chat35x2: a run failed on the link ({e}): the link reset, the turn again", file=sys.stderr)
            t.recover()
            del self.history[n:]                # (the turn's message: asked again)
            self._next = None
            kept = back_to_snapshot(self)
            print(f"chat35x2: the context back to {kept} tokens", file=sys.stderr)
            return ask(self, text, *a, **kw)
    return ask_recovering


def snapping(gen, d: Path):
    """Engine.generate_card with each slice's context in DRAM saved before and after it (CHAT35_SNAP=DIR:
    snapdiff.py's snapshot, the KV caches up to the position and the DeltaNet states, as DIR/gK{a,b}.snap.npz for the
    chat's K-th reply): two runs' compared region by region (snapdiff.py A B)."""
    for p in (HERE, HERE / "x2win", HERE / "tiny"):     # (beside this file, or a deployed directory's)
        if (p / "snapdiff.py").exists():
            sys.path.insert(0, str(p))
            break
    import snapdiff
    d.mkdir(parents=True, exist_ok=True)
    n = [0]

    def generate_card(self, *a, **kw):
        k, n[0] = n[0], n[0] + 1
        snapdiff.save(self, d / f"g{k}a.snap.npz")
        out = gen(self, *a, **kw)
        snapdiff.save(self, d / f"g{k}b.snap.npz")
        print(f"chat35x2: reply {k}'s context before and after it in {d}/g{k}a.snap.npz, g{k}b.snap.npz",
              file=sys.stderr)
        return out
    return generate_card


def main():
    from opentpu import imgcache as IC
    import opentpu.host.chat as C
    from opentpu.llm import qwen3 as Q
    build_mode, key_mode = os.environ.get("CHAT35_BUILD") == "1", os.environ.get("CHAT35_KEY") == "1"
    ahead = os.environ.get("CHAT35_AHEAD") == "1"   # (warm35x2.pkl without a card: Ahead)
    entry = os.environ.get("CHAT35_ENTRY", "")
    raw_dir = os.environ.get("CHAT35_RAW", "")  # the entry as a raw image there, mapped (raw_images)
    dev = os.environ.get("CHAT35_DEV", "")      # the cards, card 0 first: "/dev/xdmaA+/dev/xdmaB"
    if os.environ.get("OTPU_TARGET") != TARGET:
        raise SystemExit(f"chat35x2: OTPU_TARGET must be {TARGET} (the entry's environment)")
    cfg = target_config()
    build, make_backend, load_spec = IC.build, C.make_backend, C.load_spec

    def guarded_build(image, W, layout):
        d = IC.cache_dir()
        ck = None if d is None else IC.checkpoint(getattr(W, "model_dir", None))
        k = None if ck is None else IC.key(image, layout, ck)
        have = k is not None and (d / k[:2] / f"{k}.img").is_file()
        print(f"chat35x2: image key {k} ({'in the cache' if have else 'not in the cache'}; low area "
              f"{getattr(image, 'low_end', image.nbytes) / 2**20:.0f} MiB, {image.nbytes / 2**30:.2f} GiB a slice, "
              f"S {cfg.S})", flush=True)
        if key_mode:
            os._exit(0)
        if ahead:
            IC.last = ("hit", 0.0)
            return ahead_images(cfg)
        if build_mode:
            t0 = time.time()
            build(image, W, layout)
            print(f"chat35x2: built {IC.last} in {time.time() - t0:.0f} s: {k}", flush=True)
            os._exit(0 if IC.last[0] in ("hit", "miss") else 1)
        if not have or (entry and k != entry):
            raise SystemExit(f"chat35x2: the image (key {k}) is not in the cache{f' (want {entry})' if entry else ''}: "
                             "it is built on the build host (CHAT35_BUILD=1: c35x2.sh), never here")
        if raw_dir:
            t0 = time.perf_counter()
            out = C35.raw_images(d / k[:2] / f"{k}.img", Path(raw_dir), image, W)
            IC.stats["hit"] += 1
            IC.last = ("hit", time.perf_counter() - t0)
            return out
        out = build(image, W, layout)
        if IC.last[0] != "hit":
            raise SystemExit(f"chat35x2: image cache {IC.last}, not a hit")
        return out

    def backend_with_cfg(name, spec, cap, dev, model=None, **kw):
        if build_mode or key_mode:
            return C35.Null, cfg
        if ahead:
            return Ahead, cfg
        backend, got = make_backend(name, spec, cap, dev, model, **kw)
        if (got.S, got.MCOLS, got.D) != (cfg.S, cfg.MCOLS, cfg.D):
            raise SystemExit(f"chat35x2: the cards are S {got.S} MCOLS {got.MCOLS} D {got.D}, the image's "
                             f"{TARGET} S {cfg.S} MCOLS {cfg.MCOLS} D {cfg.D}")

        def with_runs(c, imgs):         # (the raw image's data runs: the scrub leaves them, only they are written)
            if C35.RUNS.get("runs") is not None:
                imgs.runs = C35.RUNS["runs"]
            return backend(c, imgs)
        return with_runs, cfg           # the configuration the entry was built with

    def spec_with_formats(path):
        from opentpu.host.streamed import defaults
        spec = load_spec(path)
        return replace(spec, formats=defaults(spec)[2]) if spec.moe else spec   # the chat's mix for the MoE family

    if "--no-prompt-runs" not in FIXED:
        dense_prompts()
    prefill_chunks = Q.Engine.prefill_chunks
    lm = int(os.environ.get("CHAT35_LM") or 0)  # (a long prompt part layer by layer, lm rows a run)
    lm_min = int(os.environ.get("CHAT35_LM_MIN") or 64)
    loop = {"layer_loop": True} if os.environ.get("CHAT35_LOOP") == "1" else {}     # (chat35.py's)
    if lm:
        from opentpu.host import runwords
        engine = C.Engine

        def engine_lm(*a, **kw):
            eng = engine(*a, layer_major=lm, **loop, **kw)
            if os.environ.get("CHAT35_RUNWORDS") != "0":
                runwords.install(eng)   # (the prompt's runs to the cards as the program cache's words)
            return eng
        C.Engine = engine_lm

    def token_steps(self, tokens, seq=0, chunk=None, **kw):
        tokens = list(tokens)
        if chunk is None and self.spec.moe is not None and (not self.layer_major
                                                             or len(tokens) < lm_min):
            chunk = 1
        return prefill_chunks(self, tokens, seq, chunk, **kw)

    from opentpu.host import gen_warm as GW
    gw_start = GW.start

    def start_saving(eng, sampling=None):
        # gen_warm's arguments kept for warm35x2.py (every bucket's programs ahead, between chats)
        be = eng.backend
        args = (eng.spec, eng.cfg, eng.cap, eng.batch, eng.rows, eng._image_kw,
                eng.image.stream_fill, eng.layout, eng.block, eng._conv_lo,
                bool(getattr(be, "chains", False)), eng.gen_debug, eng.gen_split,
                bool(getattr(be, "waitc", False)), GW.samplings(sampling))
        try:
            tmp = HERE / f".warm35x2.{os.getpid()}.pkl"
            tmp.write_bytes(pickle.dumps(args))
            os.replace(tmp, HERE / "warm35x2.pkl")
        except OSError as e:
            print(f"chat35x2: warm35x2.pkl not written: {e}", file=sys.stderr)
            if ahead:
                os._exit(1)
        if ahead:
            print(f"chat35x2: warm35x2.pkl written (samplers {[s and s.key for s in args[-1]]}, chain {args[10]}, "
                  f"waitc {args[13]})", flush=True)
            os._exit(0)
        return gw_start(eng, sampling)

    if os.environ.get("CHAT35_EXPAND"):          # the entry named into CHAT35_RAW, then exit
        f = Path(os.environ["CHAT35_EXPAND"])
        C35.expand(f, Path(raw_dir) / f"{f.stem}.raw")
        return 0
    IC.build, C.make_backend, C.load_spec = guarded_build, backend_with_cfg, spec_with_formats
    load_weights = C.load_weights
    C.load_weights = lambda path, **kw: LayerExperts(C35._LazyWeights(load_weights, path, kw))
    GW.start = start_saving
    Q.Engine.prefill_chunks = token_steps
    C.Chat.ask = recovering(C.Chat.ask)
    C.ST.streams = lambda spec: False   # the 35B resident: the cards' DRAM holds it whole
    if os.environ.get("CHAT35_TIMES"):          # a timeline of the decode's host calls (stderr)
        C35._timed(Q.Engine, "generate_card", "_generate_prog", "_generate_chain", "step",
                   "prefill_layers", "_layer_run")
        from opentpu.host import board as BD
        from opentpu.host import cards as CD
        C35._timed(BD.BoardBackend, "run_generate", "prepare", "start", "wait")
        C35._timed(BD.BoardBackend, "__init__")
        C35._timed(BD.Board, "scrub", "info")
        C35._timed(CD.CardsTransport, "__init__", "set_cards", "recover")
        C35._timed(Q.Engine, "__init__")
        C35._timed(C, "load_tokenizer", "load_weights", "make_backend", "load_spec")
    if os.environ.get("CHAT35_SNAP"):           # (outside the timed call: the snapshots not in its time)
        Q.Engine.generate_card = snapping(Q.Engine.generate_card, Path(os.environ["CHAT35_SNAP"]))
    args = list(FIXED)
    if build_mode or key_mode or ahead:
        args += ["--prompt", "-"]
    elif "+" not in dev and not {"-h", "--help"} & set(sys.argv[1:]):
        raise SystemExit("chat35x2: CHAT35_DEV must name both cards, card 0 first: /dev/xdmaA+/dev/xdmaB")
    else:
        args += ["--dev", dev]
    args += sys.argv[1:]
    prof = os.environ.get("CHAT35_PROF")        # a cProfile of the chat into this file (pstats)
    if prof:
        import cProfile
        pr = cProfile.Profile()
        try:
            return pr.runcall(C.main, args)
        finally:
            pr.dump_stats(prof)
    return C.main(args)


if __name__ == "__main__":     # (the Engine's spawned compile workers import this module)
    sys.exit(main())
