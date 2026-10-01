"""ISA-simulator reference tokens for the card qualification, cached by content.

    python3 tools/qual/refs.py cfg OUT.pkl [--name NAME]    the card's configuration (registers)
    python3 tools/qual/refs.py compute CFG.pkl [--runs R ...] [--jobs N] [--ntok 32]
                                         references for the runs (default all six), in parallel
                                         as memory allows; exit 1 if any job failed or died
    python3 tools/qual/refs.py card CFG.pkl MODEL WF HF NTOK [--resident [--card-loop]]
                                         the card's greedy tokens against the cached reference
                                         (--card-loop: after the prompt's token, the decode
                                         loop on the card, Engine.generate_card, CAPS bit30)
    python3 tools/qual/refs.py key CFG.pkl MODEL WF HF NTOK  the cache file of one reference
    python3 tools/qual/refs.py one CFG.pkl MODEL WF HF NTOK  (worker: one reference)

Run from a host tree (the repo root; PYTHONPATH set). A run is MODEL:WF:HF, HF '-' = the LM
head in WF. The prompt is otpu-selftest's model check ("What is the capital of France?").

The cache ($REFCACHE, default ~/otpu-build/refcache) is keyed by what decides the ISA
simulator's tokens: the sources of the opentpu package minus the card's host code
(opentpu/host, which only drives the card; the simulator configuration it derives,
sim_config, is hashed as a value), the ISA configuration, the model checkpoint (config and
file sizes), the formats (WF, HF and OTPU_FORMATS), the prompt, its token ids (the chat
template's output, with the date pinned) and the token count. A host-only change (a poll fix)
reuses the references; any compiler, kernel or simulator change recomputes them. The key has
no machine or path in it, so references computed on another box (the Mac, under its load
rules) can be copied into the cache directory: `rsync -a ~/otpu-build/refcache/ omarchy:...`.

Hang-proofing: a reference being computed has a `.pending` file (host, pid, a heartbeat every
15 s); a failure leaves a `.failed` file. The card check waits only while a live job is
computing the reference, and fails at once when none is (never started, died, OOM-killed).
`compute` starts jobs only while /proc/meminfo's MemAvailable covers them (their recorded peak
RSS, with a reserve) and runs one fewer per Vivado process on the box.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pickle
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CAP = 256
RUNS = ["qwen35:fp4:int8", "qwen3:fp4:int8", "qwen35:int8:-", "lfm2:fp4:int8", "qwen3:int8:-",
        "lfm2:int8:-"]                        # longest first (under load: 10, 6.5, 3, 2, 2, 0.5 min)
PROMPT = "What is the capital of France? Answer in one sentence."
DATE = datetime.date(2026, 9, 29)             # "today" for chat templates that state it
EXCLUDE = ("host/", "lens.py", "lens_app.html", "rtlsim.py", "hwtrace.py")
HEARTBEAT = 15                                # s between .pending updates
STALE = 120                                   # s without a heartbeat: the job is dead
RESERVE = 4 << 30                             # bytes of MemAvailable kept free
MEM_GUESS = {"lfm2": 2.0, "qwen3": 3.0, "qwen35": 4.0, "lfm2-2.6b": 12.0, "smollm3": 13.0,
             "phi4-mini": 16.0, "qwen35-2b": 10.0, "qwen35-4b": 14.0,
             "gemma4": 9.0}  # GiB, before a peak


def cache_root() -> Path:
    return Path(os.environ.get("REFCACHE", Path.home() / "otpu-build/refcache"))


def source_hash(pkg: Path | None = None) -> str:
    h = hashlib.sha256()
    pkg = pkg or ROOT / "opentpu"
    for p in sorted(pkg.rglob("*")):
        rel = p.relative_to(pkg).as_posix()
        if not p.is_file() or "__pycache__" in rel or rel.startswith(EXCLUDE):
            continue
        if p.suffix not in (".py", ".json", ".txt", ".html"):
            continue
        h.update(rel.encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()[:16]


def model_fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(path.iterdir()):
        if p.suffix == ".json":
            h.update(p.name.encode() + p.read_bytes())
        elif p.suffix == ".safetensors":
            h.update(f"{p.name}:{p.stat().st_size}".encode())
    return h.hexdigest()[:16]


def key(cfg, model: str, wf: str, hf: str, n: int) -> tuple[Path, dict]:
    from opentpu.host.board import sim_config
    from opentpu.llm import load_spec, model_dir
    path = model_dir(model)
    spec = load_spec(path)
    from transformers import AutoTokenizer
    ids = prompt_ids(AutoTokenizer.from_pretrained(path))
    fmt = {"wformat": wf, "head_format": None if hf == "-" else hf}
    parts = {"src": source_hash(), "sim_cfg": repr(sim_config(spec, CAP, cfg, **fmt)),
             "model": path.name, "ckpt": model_fingerprint(path), "wf": wf, "hf": hf,
             "ntok": n, "cap": CAP, "prompt": PROMPT,
             "ids": hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16],
             "formats": os.environ.get("OTPU_FORMATS")}     # per-kind formats (Spec.formats)
    k = hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:20]
    return cache_root() / f"{model}_{wf}_{hf}_{n}_{k}.pkl", parts


def prompt_ids(tok):
    """The prompt as a user turn of the model's chat template, with today's date pinned
    (DATE: SmolLM3's system header states the date, and a reference must not expire); a base
    model without a template (Gemma 4 E2B) gets the plain prompt."""
    if not tok.chat_template:
        return list(tok(PROMPT)["input_ids"])
    msgs = [{"role": "user", "content": PROMPT}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False,
                                  tokenize=True, strftime_now=lambda fmt: DATE.strftime(fmt))
    return list(ids["input_ids"] if hasattr(ids, "keys") else ids)


def load(model):
    from transformers import AutoTokenizer

    from opentpu.llm import load_spec, model_dir
    from opentpu.llm.qwen3 import load_weights
    path = model_dir(model)
    return path, load_spec(path), load_weights(path), AutoTokenizer.from_pretrained(path)


def side(kp: Path, ext: str) -> Path:
    return kp.with_suffix(ext)


# ---- one reference (the worker)
def one(cfg, model, wf, hf, n) -> int:
    kp, parts = key(cfg, model, wf, hf, n)
    if kp.exists():
        print(f"ref {model} {wf}/{hf}: cached ({kp.name})")
        return 0
    kp.parent.mkdir(parents=True, exist_ok=True)
    pend = side(kp, ".pending")
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
        ref = Engine(spec, W, cap=CAP, cfg=sim_config(spec, CAP, cfg, wformat=wf,
                                                     head_format=hf_),
                     wformat=wf, head_format=hf_)
        want = ref.generate(prompt_ids(tok), max_new=n)
        tmp = side(kp, ".tmp")
        tmp.write_bytes(pickle.dumps(want))
        tmp.rename(kp)
        side(kp, ".json").write_text(json.dumps({**parts, "tokens": len(want),
                                                 "seconds": round(time.time() - t0),
                                                 "host": me["host"]}, indent=1))
        print(f"ref {model} {wf}/{hf}: {len(want)} tokens in {time.time() - t0:.0f} s")
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


def compute(cfgf: Path, runs: list[str], jobs: int | None, n: int) -> int:
    cfg = pickle.loads(cfgf.read_bytes())
    todo = []
    for r in runs:
        m, wf, hf = r.split(":")
        kp, _ = key(cfg, m, wf, hf, n)
        if kp.exists():
            print(f"ref {m} {wf}/{hf}: cached ({kp.name})", flush=True)
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
            m, wf, hf = r.split(":")
            kp.parent.mkdir(parents=True, exist_ok=True)
            log = open(kp.with_suffix(".log"), "w")
            pr = subprocess.Popen(
                ["nice", "-n", "5", sys.executable, __file__, "one", str(cfgf), m, wf, hf,
                 str(n)], stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
            running[pr.pid] = (r, pr, kp, time.time(), 0)
            mem = "" if avail is None else (f", MemAvailable {avail / 2**30:.1f} GiB, "
                                            f"needs ~{need / 2**30:.1f}")
            print(f"ref {r}: started (pid {pr.pid}, {len(running)} running{mem})", flush=True)
        mark_queued()
        time.sleep(2)
    (cache_root() / "mem.json").write_text(json.dumps(db, indent=1))
    for r in runs:
        m, wf, hf = r.split(":")
        kp, _ = key(cfg, m, wf, hf, n)
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


def card(cfgf: Path, model, wf, hf, n, resident: bool, loop: bool = False) -> int:
    import numpy as np

    from opentpu.host.board import BoardBackend, XdmaTransport
    from opentpu.llm.qwen3 import Engine
    cfg = pickle.loads(cfgf.read_bytes())
    kp, _ = key(cfg, model, wf, hf, n)
    label = f"model {model} {wf}/{hf}" + (" resident" if resident else "") + \
        (" card loop" if loop else "")
    early = None if side(kp, ".pending").exists() else wait_ref(kp)
    if early:
        print(f"  [FAIL] {label}: {early}")
        return 1
    path, spec, W, tok = load(model)
    ids = prompt_ids(tok)
    tr = XdmaTransport("/dev/xdma0")
    hf_ = None if hf == "-" else hf
    dev = Engine(spec, W, cap=CAP, cfg=cfg, wformat=wf, head_format=hf_, resident=resident,
                 backend=lambda c, imgs: BoardBackend(c, imgs, transport=tr, model=path.name))
    if loop and not dev.can_generate:
        print(f"  [FAIL] {label}: the card does not run the decode loop (CAPS bit30, resident)")
        dev.backend.close()
        return 1
    t0 = time.time()
    if loop:                            # Engine.generate's tokens, the ones after the first
        got = [int(np.argmax(dev.prefill(ids)))]    # picked and fed back on the card
        if got[0] not in spec.eos:
            got += dev.generate_card(got[0], n - 1)
    else:
        got = dev.generate(ids, max_new=n)
    dt = time.time() - t0
    engaged = dev.resident
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
    want = pickle.loads(kp.read_bytes())
    ok = got == want
    note = "" if not resident else ("" if engaged else " (resident decode not engaged)")
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{note}: "
          f"{tok.decode(got, skip_special_tokens=True)!r}; prompt of {len(ids)} tokens in "
          f"{len(pre)} runs, {pre_cyc / 1e6:.2f} Mcycles/token; {len(one_)} "
          f"{'tokens on the card loop' if loop else 'one-token runs'}, "
          f"{np.mean(one_) / 1e6:.2f} Mcycles/token; {dt:.1f} s wall"
          + ("" if ok else f"; ISA simulator says {tok.decode(want)!r}"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["cfg", "compute", "card", "key", "one"])
    ap.add_argument("cfg")
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--name", help="cfg: also keep it as $REFCACHE/configs/NAME.pkl")
    ap.add_argument("--runs", nargs="*", default=RUNS)
    ap.add_argument("--jobs", type=int)
    ap.add_argument("--ntok", type=int, default=32)
    ap.add_argument("--resident", action="store_true")
    ap.add_argument("--card-loop", action="store_true")
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
        kp, parts = key(cfg, model, wf, hf, n)
        print(kp)
        print(json.dumps(parts, indent=1))
        return 0
    if a.mode == "one":
        return one(cfg, model, wf, hf, n)
    return card(cfgf, model, wf, hf, n, a.resident or a.card_loop, a.card_loop)


if __name__ == "__main__":
    sys.exit(main())
