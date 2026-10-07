"""ISA-simulator references for the card qualification's token-exact checks, cached by content.

    python3 tools/qual/refs.py cfg OUT.pkl [--name NAME]    the card's configuration (registers)
    python3 tools/qual/refs.py compute CFG.pkl [--runs R ...] [--jobs N] [--ntok 32]
                                         references for the runs (default all eight), in
                                         parallel as memory allows; exit 1 if any job failed or
                                         died
    python3 tools/qual/refs.py card CFG.pkl MODEL WF HF NTOK [--long] [--resident [--card-loop]]
                                         [--prompt-runs]
                                         the card's greedy tokens and logits against the cached
                                         reference (--card-loop: after the prompt's token, the
                                         decode loop on the card, Engine.generate_card, CAPS
                                         bit30; --prompt-runs: the prompt in prompt runs,
                                         docs/prefill.md, resident; --long: the long prompt)
    python3 tools/qual/refs.py key CFG.pkl MODEL WF HF NTOK [--long] [--prompt-runs]
                                         the cache file of one reference
    python3 tools/qual/refs.py one CFG.pkl MODEL WF HF NTOK [--long] [--prompt-runs]
                                         (worker: one reference)

Run from a host tree (the repo root; PYTHONPATH set). A run is MODEL:WF[:HF[:FLAGS]], HF '-' =
the LM head in WF; FLAGS, comma-separated: long (the long prompt), pr (prompt runs).

A reference is the ISA simulator's NTOK greedy tokens after the prompt, past EOS (no stop ids:
a model's one-sentence answer ends after about 8 tokens, and the decode goes on), and a digest
of every logits vector a token was picked from (the prefill's last row's, then each step's: the
fp32 bits' sha256, with the top 5 for the FAIL line). The card check compares the tokens and,
bit for bit, the logits: all NTOK per position and in resident decode (which gives per-position
decoding's logits bit for bit, tests/test_lfm2.py), the prefill's on the card's decode loop
(whose logits stay on the device). The prompts:
- short: otpu-selftest's model check ("What is the capital of France?"), about 22 tokens;
- long: LONG_TEXT cut to at most LONG_TOKENS tokens, so that NTOK = 32 tokens decode across
  position 256, the attention bucket's end (a run per bucket, the decode loop's HALT CHAIN).
With pr (--prompt-runs) the reference prefills in prompt runs too (resident, at run-time
positions: their split can round a 4-bit MM's sums in another order than compile_rows' runs).

The cache ($REFCACHE, default ~/otpu-build/refcache) is keyed by what decides the ISA
simulator's tokens and logits: the sources of the opentpu package minus the card's host code
(opentpu/host, which only drives the card; the simulator configuration it derives, sim_config,
is hashed as a value) but with the host code the models import (HOST_IN_REF), the ISA
configuration, the model checkpoint (config and file sizes), the formats (WF, HF), the OTPU_*
environment those sources read (env_key: OTPU_FORMATS, OTPU_PLE_FORMAT, ...; not the runtime
ones), the prompt, its token ids (the chat template's output, with the date pinned), the token
count, prompt runs and the reference's format (VERSION). A host-only change (a poll fix) reuses
the references; any compiler, kernel or simulator change recomputes them. The key has no
machine or path in it, so references computed on another box (omarchy, or the Mac under its
load rules) can be copied into the cache directory: `rsync -a ~/otpu-build/refcache/ ...`.

Hang-proofing: a reference being computed has a `.pending` file (host, pid, a heartbeat every
15 s); a failure leaves a `.failed` file. The card check waits only while a live job is
computing the reference, and fails at once when none is (never started, died, OOM-killed).
`compute` starts jobs only while /proc/meminfo's MemAvailable covers them (their recorded peak
RSS, with a reserve) and runs one fewer per Vivado process on the box; killed (SIGTERM), it
kills its jobs and marks their references failed.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pickle
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

VERSION = 2                                   # the reference's format: tokens past EOS, logits
CAP = 512                                     # KV positions: the long prompt decodes past 256
NTOK = 32
RUNS = ["qwen35:fp4:int8", "qwen3:fp4:int8", "qwen35:int8:-", "lfm2:fp4:int8", "qwen3:int8:-",
        "lfm2:int8:-", "lfm2:int8:-:long", "lfm2:int8:-:long,pr"]      # longest first
PROMPT = "What is the capital of France? Answer in one sentence."
LONG_TEXT = (
    "Read this history of timekeeping, then summarize it in three sentences and name the "
    "invention you consider the most important. Mechanical clocks first appeared in the "
    "towers of European cathedrals and town halls near the end of the thirteenth century. They "
    "had no faces at first: a falling weight turned a train of gears, a verge and foliot "
    "escapement let the train advance in small steps, and a bell struck the hours for the "
    "people below. Such clocks could drift by a quarter of an hour in a day, so their keepers "
    "set them each noon against a sundial. The pendulum changed that. In 1656 Christiaan "
    "Huygens built a clock whose escapement was governed by a swinging pendulum, and its error "
    "fell to under a minute a day. Minute hands became worth having, and soon the anchor "
    "escapement allowed long pendulums with small swings, which kept even better time. At sea, "
    "though, a pendulum is useless, and navigators needed the time at their home port to find "
    "their longitude: every four minutes of error in the clock meant a degree of error in the "
    "position. After a prize was offered in 1714, John Harrison spent decades building a series "
    "of marine timekeepers, the last of them a large watch that kept time to within a few "
    "seconds on a voyage to Jamaica. In the nineteenth century the railways spread standard "
    "time across whole countries, and telegraph signals carried the time from observatories to "
    "stations. Quartz crystals, which vibrate at a steady frequency in an electric field, "
    "replaced pendulums in the best clocks of the 1930s, and cheap quartz watches reached "
    "everyone in the 1970s. Atomic clocks, which count the oscillations of caesium atoms, have "
    "defined the second since 1967; the best of them would neither gain nor lose a second in "
    "many millions of years. Today satellites carry such clocks, and a phone finds its place on "
    "Earth by comparing the times at which their signals arrive.")
LONG_TOKENS = 240                             # the long prompt's tokens, at most
ATTN_BLOCK = 256                              # the attention bucket (opentpu.llm.qwen3's)
DATE = datetime.date(2026, 9, 29)             # "today" for chat templates that state it
EXCLUDE = ("host/", "lens.py", "lens_app.html", "rtlsim.py", "hwtrace.py")
# host code the models import into the reference's path (opentpu.llm.*, kernels/mailbox.py:
# LINE, Layout, RowLayout, ExpertServer: the mailbox, PLE rows, MoE serving)
HOST_IN_REF = ("host/offload.py",)
# OTPU_* variables the hashed sources read that do not change a reference: the caches' and the
# card tools' (opentpu.progcache.RUNTIME_ENV), the simulators' scratch space and build jobs, and
# board_config's defaults (the card's configuration is in the key by value)
RUNTIME_ENV = ("OTPU_SIM_TMP", "OTPU_BUILD_JOBS", "OTPU_MCOLS", "OTPU_LANES", "OTPU_PAIR",
               "OTPU_DSTEP", "OTPU_STREAM", "OTPU_ACT_ROWS")
HEARTBEAT = 15                                # s between .pending updates
STALE = 120                                   # s without a heartbeat: the job is dead
RESERVE = 4 << 30                             # bytes of MemAvailable kept free
MEM_GUESS = {"lfm2": 2.0, "qwen3": 3.0, "qwen35": 4.0, "lfm2-2.6b": 12.0, "smollm3": 13.0,
             "phi4-mini": 16.0, "qwen35-2b": 10.0, "qwen35-4b": 14.0,
             "gemma4": 9.0}  # GiB, before a peak


def cache_root() -> Path:
    return Path(os.environ.get("REFCACHE", Path.home() / "otpu-build/refcache"))


def sources(pkg: Path | None = None):
    """(name, path) of the files the key hashes, in order."""
    pkg = pkg or ROOT / "opentpu"
    for p in sorted(pkg.rglob("*")):
        rel = p.relative_to(pkg).as_posix()
        if not p.is_file() or "__pycache__" in rel or p.suffix not in (".py", ".json", ".txt",
                                                                         ".html"):
            continue
        if rel.startswith(EXCLUDE) and rel not in HOST_IN_REF:
            continue
        yield rel, p


def source_hash(pkg: Path | None = None) -> str:
    h = hashlib.sha256()
    for rel, p in sources(pkg):
        h.update(rel.encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()[:16]


def env_key(pkg: Path | None = None) -> dict:
    """The OTPU_* variables set in the environment that the hashed sources read (lfm2's
    OTPU_MLP_UNROLL_BODIES, gemma4's OTPU_PLE_FORMAT and OTPU_PLE_HOST, OTPU_FORMATS), but not
    the runtime ones (RUNTIME_ENV): {name: value}; set to "" is not unset."""
    from opentpu.progcache import RUNTIME_ENV as PC_RUNTIME
    names = set()
    for _, p in sources(pkg):
        names |= {m.decode() for m in re.findall(rb"OTPU_[A-Z0-9_]+", p.read_bytes())}
    skip = set(PC_RUNTIME) | set(RUNTIME_ENV)
    return {k: os.environ[k] for k in sorted(names - skip) if k in os.environ}


def model_fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(path.iterdir()):
        if p.suffix == ".json":
            h.update(p.name.encode() + p.read_bytes())
        elif p.suffix == ".safetensors":
            h.update(f"{p.name}:{p.stat().st_size}".encode())
    return h.hexdigest()[:16]


def parse_run(r: str) -> tuple:
    """MODEL:WF[:HF[:FLAGS]] -> (model, wf, hf, long, pr)."""
    f = r.split(":")
    if not 2 <= len(f) <= 4:
        raise ValueError(f"{r}: a run is MODEL:WF[:HF[:FLAGS]]")
    flags = set(filter(None, f[3].split(","))) if len(f) > 3 else set()
    if flags - {"long", "pr"}:
        raise ValueError(f"{r}: the flags are long and pr")
    return f[0], f[1], f[2] if len(f) > 2 else "-", "long" in flags, "pr" in flags


def key(cfg, model: str, wf: str, hf: str, n: int, long: bool = False,
        pr: bool = False) -> tuple[Path, dict]:
    from opentpu.host.board import sim_config
    from opentpu.llm import load_spec, model_dir
    path = model_dir(model)
    spec = load_spec(path)
    from transformers import AutoTokenizer
    ids = prompt_ids(AutoTokenizer.from_pretrained(path), long)
    fmt = {"wformat": wf, "head_format": None if hf == "-" else hf}
    parts = {"v": VERSION, "src": source_hash(),
             "sim_cfg": repr(sim_config(spec, CAP, cfg, lookup=pr, **fmt)),
             "model": path.name, "ckpt": model_fingerprint(path), "wf": wf, "hf": hf,
             "ntok": n, "cap": CAP, "prompt": f"{LONG_TOKENS}: {LONG_TEXT}" if long else PROMPT,
             "ids": hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16], "pr": pr,
             "env": env_key()}
    k = hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:20]
    tag = ("_long" if long else "") + ("_pr" if pr else "")
    return cache_root() / f"{model}_{wf}_{hf}_{n}{tag}_{k}.pkl", parts


def _chat(tok, text: str) -> list:
    """`text` as a user turn of the model's chat template, with today's date pinned (DATE:
    SmolLM3's system header states the date, and a reference must not expire); a base model
    without a template (Gemma 4 E2B) gets the plain text."""
    if not tok.chat_template:
        return list(tok(text)["input_ids"])
    msgs = [{"role": "user", "content": text}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False,
                                  tokenize=True, strftime_now=lambda fmt: DATE.strftime(fmt))
    return list(ids["input_ids"] if hasattr(ids, "keys") else ids)


def prompt_ids(tok, long: bool = False) -> list:
    """The prompt's token ids: PROMPT's, or (long) those of the most words of LONG_TEXT that
    make a prompt of at most LONG_TOKENS tokens."""
    if not long:
        return _chat(tok, PROMPT)
    words = LONG_TEXT.split(" ")
    lo, hi = 1, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(_chat(tok, " ".join(words[:mid]))) <= LONG_TOKENS:
            lo = mid
        else:
            hi = mid - 1
    return _chat(tok, " ".join(words[:lo]))


def load(model):
    from transformers import AutoTokenizer

    from opentpu.llm import load_spec, model_dir
    from opentpu.llm.qwen3 import load_weights
    path = model_dir(model)
    return path, load_spec(path), load_weights(path), AutoTokenizer.from_pretrained(path)


def side(kp: Path, ext: str) -> Path:
    return kp.with_suffix(ext)


def digest(lg) -> tuple[str, list]:
    """A logits vector's fp32 bits' sha256 (16 hex digits) and its top 5 [(id, value)]."""
    import numpy as np
    lg = np.ascontiguousarray(lg, np.float32)
    top = np.argsort(-lg, kind="stable")[:5]
    return (hashlib.sha256(lg.view(np.uint32).tobytes()).hexdigest()[:16],
            [(int(i), float(lg[i])) for i in top])


def greedy(eng, ids, n: int, loop: bool = False) -> tuple[list, list]:
    """n greedy tokens after the prompt, past EOS (no stop ids), and the digests of the logits
    each was picked from: the prefill's last row's, then each step's; with loop (the decode
    loop on the card, Engine.generate_card) only the prefill's (the loop's stay on the
    device)."""
    import numpy as np
    lg = eng.prefill(ids)
    toks, digs = [int(np.argmax(lg))], [digest(lg)]
    if loop:
        return toks + [int(t) for t in eng.generate_card(toks[0], n - 1, stop_ids=[])], digs
    while len(toks) < n:
        lg = eng.step(toks[-1])
        toks.append(int(np.argmax(lg)))
        digs.append(digest(lg))
    return toks, digs


def prompt_runs_taken(eng, ids) -> bool:
    """The engine prefills `ids` in prompt runs (prefill.supported and covers)."""
    from opentpu.llm import prefill as PF
    return PF.supported(eng) and PF.covers(eng, 0, len(ids))


# ---- one reference (the worker)
def one(cfg, model, wf, hf, n, long: bool = False, pr: bool = False) -> int:
    kp, parts = key(cfg, model, wf, hf, n, long, pr)
    pend = side(kp, ".pending")
    if kp.exists():
        pend.unlink(missing_ok=True)            # (compute's mark of a queued job)
        print(f"ref {model} {wf}/{hf}: cached ({kp.name})")
        return 0
    kp.parent.mkdir(parents=True, exist_ok=True)
    side(kp, ".failed").unlink(missing_ok=True)
    me = {"host": socket.gethostname(), "pid": os.getpid()}
    stop = threading.Event()

    def beat():
        while not stop.is_set():
            pend.write_text(json.dumps({**me, "t": time.time()}))
            stop.wait(HEARTBEAT)
    th = threading.Thread(target=beat, daemon=True)
    th.start()
    try:
        from opentpu.host.board import sim_config
        from opentpu.llm.qwen3 import Engine
        path, spec, W, tok = load(model)
        t0 = time.time()
        hf_ = None if hf == "-" else hf
        ref = Engine(spec, W, cap=CAP, cfg=sim_config(spec, CAP, cfg, lookup=pr, wformat=wf,
                                                     head_format=hf_),
                     wformat=wf, head_format=hf_, resident=pr, prompt_runs=pr)
        ids = prompt_ids(tok, long)
        if pr and not (ref.resident and prompt_runs_taken(ref, ids)):
            raise RuntimeError(f"{model} {wf}/{hf}: the ISA simulator's engine takes no prompt "
                               "runs (prefill.supported, covers)")
        toks, digs = greedy(ref, ids, n)
        want = {"v": VERSION, "tokens": toks, "logits": [d for d, _ in digs],
                "top": [t for _, t in digs], "prompt": len(ids), "pr": pr}
        tmp = kp.with_name(f"{kp.name}.{me['host']}.{me['pid']}.tmp")   # this writer's own
        tmp.write_bytes(pickle.dumps(want))
        tmp.rename(kp)
        side(kp, ".json").write_text(json.dumps({**parts, "tokens": len(toks),
                                                 "prompt_tokens": len(ids),
                                                 "seconds": round(time.time() - t0),
                                                 "host": me["host"]}, indent=1))
        print(f"ref {model} {wf}/{hf}: {len(toks)} tokens in {time.time() - t0:.0f} s")
        return 0
    except BaseException as e:                  # noqa: BLE001 (recorded, then re-raised)
        side(kp, ".failed").write_text(f"{type(e).__name__}: {e}")
        raise
    finally:
        stop.set()
        th.join()
        pend.unlink(missing_ok=True)


# ---- many references, memory-aware
def mem_available() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def vivado_count() -> int:
    """Vivado builds on this box: the top-level process run_vivado.sh starts (-nojournal); the
    synthesis and implementation runs it spawns belong to the same build."""
    n = 0
    for d in Path("/proc").glob("[0-9]*"):
        try:
            c = (d / "cmdline").read_bytes()
            if b"unwrapped/lnx64.o/vivado" in c and b"-nojournal" in c:
                n += 1
        except OSError:
            pass
    return n


def peak_rss(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def mem_db() -> dict:
    p = cache_root() / "mem.json"
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def run_key(cfg, r: str, n: int) -> Path:
    m, wf, hf, long, pr = parse_run(r)
    return key(cfg, m, wf, hf, n, long, pr)[0]


def compute(cfgf: Path, runs: list[str], jobs: int | None, n: int) -> int:
    cfg = pickle.loads(cfgf.read_bytes())
    todo = []
    for r in runs:
        kp = run_key(cfg, r, n)
        if kp.exists():
            print(f"ref {r}: cached ({kp.name})", flush=True)
        else:
            todo.append((r, kp))
    host = socket.gethostname()

    def mark_queued():                          # queued jobs count as being computed
        for _, kp in todo:
            kp.parent.mkdir(parents=True, exist_ok=True)
            side(kp, ".failed").unlink(missing_ok=True)
            side(kp, ".pending").write_text(json.dumps(
                {"host": host, "pid": os.getpid(), "t": time.time(), "queued": True}))
    mark_queued()
    linux = mem_available() is not None
    maxj = jobs or (3 if linux else 1)
    db = mem_db()
    env = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
           "MKL_NUM_THREADS": "2"}
    running: dict = {}                          # pid -> (run, Popen, kp, t0, peak)
    failed = 0
    # killed (qual.sh: its bounded wait, an early exit): the jobs go too, and say so below
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(128 + signal.SIGTERM))
    try:
        while todo or running:
            for pid, (r, pr, kp, t0, peak) in list(running.items()):
                running[pid] = (r, pr, kp, t0, max(peak, peak_rss(pid)))
                rc = pr.poll()
                if rc is None:
                    continue
                peak = running.pop(pid)[4]
                if peak:
                    db[r] = max(peak, db.get(r, 0))
                if rc == 0 and kp.exists():
                    continue
                failed += 1
                why = (f"killed by signal {-rc}" + (" (the OOM killer?)" if rc == -signal.SIGKILL
                                                     else "")) if rc < 0 else f"exit {rc}"
                side(kp, ".pending").unlink(missing_ok=True)
                if not side(kp, ".failed").exists():
                    side(kp, ".failed").write_text(why)
                print(f"ref {r}: FAILED, {why}; see {kp.with_suffix('.log')}", flush=True)
            cap = max(1, maxj - (vivado_count() if linux else 0))
            while todo and len(running) < cap:
                r, kp = todo[0]
                need = db.get(r) or int(MEM_GUESS.get(r.split(":")[0], 16.0) * (1 << 30))
                avail = mem_available()
                if running and avail is not None and avail - RESERVE < need:
                    break                               # wait for memory (always start one)
                todo.pop(0)
                m, wf, hf, long, pr = parse_run(r)
                kp.parent.mkdir(parents=True, exist_ok=True)
                log = open(kp.with_suffix(".log"), "w")
                p = subprocess.Popen(
                    ["nice", "-n", "5", sys.executable, __file__, "one", str(cfgf), m, wf, hf,
                     str(n)] + ["--long"] * long + ["--prompt-runs"] * pr,
                    stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
                running[p.pid] = (r, p, kp, time.time(), 0)
                mem = "" if avail is None else (f", MemAvailable {avail / 2**30:.1f} GiB, "
                                                f"needs ~{need / 2**30:.1f}")
                print(f"ref {r}: started (pid {p.pid}, {len(running)} running{mem})",
                      flush=True)
            mark_queued()
            time.sleep(2)
    finally:                                    # stopped early: no job left behind
        for r, p, kp, _, _ in running.values():
            p.kill()
            p.wait()
            side(kp, ".pending").unlink(missing_ok=True)
            side(kp, ".failed").write_text("killed: refs.py compute stopped before it ended")
            print(f"ref {r}: killed, refs.py compute stopped", flush=True)
        for r, kp in todo:
            side(kp, ".pending").unlink(missing_ok=True)
            side(kp, ".failed").write_text("never started: refs.py compute stopped")
    (cache_root() / "mem.json").write_text(json.dumps(db, indent=1))
    for r in runs:
        kp = run_key(cfg, r, n)
        meta = side(kp, ".json")
        if kp.exists() and meta.exists():
            s = json.loads(meta.read_text()).get("seconds")
            print(f"ref {r}: ok ({s} s when computed)")
    return 1 if failed else 0


# ---- the card check
def wait_ref(kp: Path) -> str | None:
    """None when the reference exists; otherwise why it never will (fails fast)."""
    while not kp.exists():
        fail = side(kp, ".failed")
        if fail.exists():
            return f"the reference job failed: {fail.read_text()[:300]}"
        pend = side(kp, ".pending")
        if not pend.exists():
            return ("no reference and no job computing it: run tools/qual/refs.py compute "
                    f"(or copy {kp.name} into {kp.parent})")
        try:
            p = json.loads(pend.read_text())
        except ValueError:
            time.sleep(1)
            continue
        age = time.time() - p["t"]
        alive = True
        if p["host"] == socket.gethostname():
            try:
                os.kill(p["pid"], 0)
            except ProcessLookupError:
                alive = False
            except PermissionError:
                pass
        if not alive or age > STALE:
            return (f"the reference job (pid {p['pid']} on {p['host']}) died: last heartbeat "
                    f"{age:.0f} s ago")
        time.sleep(5)
    return None


def compare(toks: list, digs: list, want, tok=None) -> str | None:
    """None when the card's tokens and logits digests are the reference's (all its tokens; the
    logits the card gave: the card loop's prefill's only), else the first difference."""
    if not isinstance(want, dict) or want.get("v") != VERSION:
        return f"the reference is not of format {VERSION} (an older refs.py's token list)"
    name = (lambda t: f"{t} {tok.decode([t])!r}") if tok is not None else str
    for i, (a, b) in enumerate(zip(toks, want["tokens"])):
        if a != b:
            return f"token {i} is {name(a)}, the ISA simulator's {name(b)}"
    if len(toks) != len(want["tokens"]):
        return f"{len(toks)} tokens, the ISA simulator's {len(want['tokens'])}"
    for i, ((d, top), w) in enumerate(zip(digs, want["logits"])):
        if d != w:
            return (f"token {i}'s logits differ (sha {d}, the ISA simulator's {w}); top 5 "
                    f"{top}, the ISA simulator's {want['top'][i]}")
    return None


def card(cfgf: Path, model, wf, hf, n, resident: bool, loop: bool = False,
         prompt_runs: bool = False, long: bool = False) -> int:
    import numpy as np

    from opentpu.host.board import BoardBackend, XdmaTransport
    from opentpu.llm.qwen3 import Engine
    resident = resident or loop or prompt_runs
    cfg = pickle.loads(cfgf.read_bytes())
    kp, _ = key(cfg, model, wf, hf, n, long, prompt_runs)
    label = f"model {model} {wf}/{hf}" + (" long" if long else "") + \
        (" resident" if resident else "") + (" card loop" if loop else "") + \
        (" prompt runs" if prompt_runs else "")
    early = None if side(kp, ".pending").exists() else wait_ref(kp)
    if early:
        print(f"  [FAIL] {label}: {early}")
        return 1
    path, spec, W, tok = load(model)
    ids = prompt_ids(tok, long)
    tr = XdmaTransport("/dev/xdma0")
    hf_ = None if hf == "-" else hf
    dev = Engine(spec, W, cap=CAP, cfg=cfg, wformat=wf, head_format=hf_, resident=resident,
                 backend=lambda c, imgs: BoardBackend(c, imgs, transport=tr, model=path.name),
                 prompt_runs=prompt_runs)
    why = None                          # a check that would not test what it says: a FAIL
    if resident and not dev.resident:
        why = "resident decode not engaged (CAPS bit25, the image's lookup tables)"
    elif loop and not dev.can_generate:
        why = "the card does not run the decode loop (CAPS bit30, resident)"
    elif prompt_runs and not prompt_runs_taken(dev, ids):
        why = "prompt runs not taken (prefill.supported, covers)"
    elif long and not len(ids) < ATTN_BLOCK <= len(ids) + n - 2:
        why = (f"the decode, positions {len(ids)} .. {len(ids) + n - 2}, does not cross "
               f"position {ATTN_BLOCK}")
    if why:
        print(f"  [FAIL] {label}: {why}")
        dev.backend.close()
        return 1
    t0 = time.time()
    got, digs = greedy(dev, ids, n, loop)
    dt = time.time() - t0
    pre = [s for s in dev.stats if "rows" in s]
    pre_cyc = sum(s["cycles"] for s in pre) / max(1, sum(s["rows"] for s in pre))
    one_ = [s["cycles"] for s in dev.stats if "rows" not in s]
    if loop:                            # a run per bucket reached: cycles per token
        one_ = [sum(one_) / max(1, len(got) - 1)] * (len(got) - 1)
    dev._drain()
    dev.backend.close()
    why = wait_ref(kp)
    if why:
        print(f"  [FAIL] {label}: {why}")
        return 1
    bad = compare(got, digs, pickle.loads(kp.read_bytes()), tok)
    print(f"  [{'FAIL' if bad else 'PASS'}] {label}: "
          + (f"{bad}; " if bad else f"{len(got)} tokens and the logits of "
             f"{'the prefill' if loop else f'all {len(digs)}'} bit-exact; ")
          + f"{tok.decode(got[:12], skip_special_tokens=True)!r}...; prompt of {len(ids)} "
          f"tokens in {len(pre)} runs, {pre_cyc / 1e6:.2f} Mcycles/token; {len(one_)} "
          f"{'tokens on the card loop' if loop else 'one-token runs'} at positions "
          f"{len(ids)} .. {len(ids) + n - 2}, {np.mean(one_) / 1e6:.2f} Mcycles/token; "
          f"{dt:.1f} s wall")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["cfg", "compute", "card", "key", "one"])
    ap.add_argument("cfg")
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--name", help="cfg: also keep it as $REFCACHE/configs/NAME.pkl")
    ap.add_argument("--runs", nargs="*", default=RUNS)
    ap.add_argument("--jobs", type=int)
    ap.add_argument("--ntok", type=int, default=NTOK)
    ap.add_argument("--resident", action="store_true")
    ap.add_argument("--card-loop", action="store_true")
    ap.add_argument("--prompt-runs", action="store_true")
    ap.add_argument("--long", action="store_true", help="the long prompt")
    a = ap.parse_args()
    cfgf = Path(a.cfg)
    if a.mode == "cfg":
        from opentpu.host.board import Board, XdmaTransport, device_config
        t = XdmaTransport("/dev/xdma0", dma=False)
        cfg = device_config(Board(t, check=False, lock=False).info())
        blob = pickle.dumps(cfg)
        cfgf.write_bytes(blob)
        if a.name:
            (cache_root() / "configs").mkdir(parents=True, exist_ok=True)
            (cache_root() / "configs" / f"{a.name}.pkl").write_bytes(blob)
        print(f"config {hashlib.sha1(repr(cfg).encode()).hexdigest()[:12]}: {cfg}")
        return 0
    if a.mode == "compute":
        return compute(cfgf, a.runs, a.jobs, a.ntok)
    model, wf, hf, n = a.rest[0], a.rest[1], a.rest[2], int(a.rest[3])
    cfg = pickle.loads(cfgf.read_bytes())
    if a.mode == "key":
        kp, parts = key(cfg, model, wf, hf, n, a.long, a.prompt_runs)
        print(kp)
        print(json.dumps(parts, indent=1))
        return 0
    if a.mode == "one":
        return one(cfg, model, wf, hf, n, a.long, a.prompt_runs)
    return card(cfgf, model, wf, hf, n, a.resident, a.card_loop, a.prompt_runs, a.long)


if __name__ == "__main__":
    sys.exit(main())
