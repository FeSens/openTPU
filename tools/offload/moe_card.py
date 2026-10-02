"""A MoE model (LFM2.5-8B-A1B, Qwen3.5-35B-A3B) end to end on the ISA simulator with the card's
4 GiB of DRAM (docs/offload.md, phase 2): the card routes and computes every expert, its
experts stream into per-layer slots through the host's server (opentpu.host.offload), and the
decode loop runs on the card (autodecode's generate program). Greedy tokens against Hugging
Face's.

    python3 tools/offload/moe_card.py MODEL --hf out.json       # HF's greedy tokens (bf16, CPU)
    python3 tools/offload/moe_card.py MODEL --check out.json    # the simulator's, compared
    python3 tools/offload/moe_card.py MODEL --check out.json --cfg dev.pkl --card   # the card's

The HF run and the simulator run are separate processes: each needs most of a 32 GB host.
--cfg takes the bitstream's configuration (tools/qual/refs.py cfg): the simulator's run with it
is the card's reference bit for bit (its prefill logits' sha256 and its tokens); --card runs on
the card (/dev/xdma0) with the host's server polled while each run is in flight, and adds the
device's cycles, the tok/s and where the host's time went (pool reads, DRAM writes over PCIe).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

PROMPT = "What is the capital of France? Answer in one sentence."


def hf_greedy(model: str, n: int, max_memory: str | None, prompt: str = PROMPT) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    if tok.chat_template is not None:
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      add_generation_prompt=True, tokenize=True)
        ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
    else:                                       # a checkpoint without one (the 26B's download):
        ids = list(tok(prompt)["input_ids"])    # the prompt as plain text, after BOS
    kw = {}
    if max_memory:                              # the rest offloaded to disk (accelerate)
        kw = dict(device_map="auto", max_memory={"cpu": max_memory},
                  offload_folder=str(Path(model) / ".." / ".offload-moe-card"))
    m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16, **kw).eval()
    t = time.time()
    with torch.no_grad():
        out = m.generate(torch.tensor([ids]), max_new_tokens=n, min_new_tokens=n,
                         do_sample=False, output_scores=True, return_dict_in_generate=True)
    new = [int(x) for x in out.sequences[0, len(ids):]]  # n tokens: past an end of turn too
    top = []                                    # per step HF's 8 best (id, logit): a card's
    for sc in out.scores:                       # different pick, by how much it lost
        v, i = torch.topk(sc[0].float(), 8)
        top.append([[int(a), round(float(b), 4)] for a, b in zip(i, v)])
    return dict(model=model, prompt=prompt, ids=ids, tokens=new, text=tok.decode(new),
                top=top, seconds=round(time.time() - t, 1))


def host_mem() -> dict | None:
    """The host's memory now, GB (Linux /proc): this process's resident (rss, its peak hwm;
    rss_file the mapped files' pages, which the page cache holds and may drop) and swapped, its
    children's resident and swapped, the system's available, page cache, anonymous and swap
    used."""
    def kb(path, keys):
        try:
            with open(path) as f:
                return {k: int(v.split()[0]) for k, _, v in (ln.partition(":") for ln in f)
                        if k in keys}
        except OSError:
            return {}
    me = kb("/proc/self/status", ("VmRSS", "VmHWM", "VmSwap", "RssFile"))
    if not me:
        return None
    kids = {"VmRSS": 0, "VmSwap": 0}
    for d in Path("/proc").iterdir():           # (the Engine's compile worker)
        try:
            if d.name.isdigit() and int((d / "stat").read_text().rsplit(")", 1)[1].split()[1]) \
                    == os.getpid():
                for k, v in kb(d / "status", ("VmRSS", "VmSwap")).items():
                    kids[k] += v
        except (OSError, ValueError, IndexError):
            continue
    m = kb("/proc/meminfo", ("MemTotal", "MemAvailable", "Cached", "AnonPages", "SwapTotal",
                             "SwapFree"))
    gb = lambda v: round(v / 1e6, 2)            # noqa: E731 (kB)
    return dict(rss=gb(me.get("VmRSS", 0)), hwm=gb(me.get("VmHWM", 0)),
                rss_file=gb(me.get("RssFile", 0)), swap=gb(me.get("VmSwap", 0)),
                children_rss=gb(kids["VmRSS"]),
                children_swap=gb(kids["VmSwap"]), total=gb(m.get("MemTotal", 0)),
                available=gb(m.get("MemAvailable", 0)), cached=gb(m.get("Cached", 0)),
                anon=gb(m.get("AnonPages", 0)),
                swap_used=gb(m.get("SwapTotal", 0) - m.get("SwapFree", 0)))


def fit_experts(spec, cfg, cap: int, **kw) -> int:
    """The expert slots per MoE layer that fill the card's DRAM beside the rest of the image."""
    from dataclasses import replace
    probe = spec.image(replace(cfg, DRAM_BYTES=1 << 40), cap, rows=1, experts=spec.moe.k, **kw)
    L = probe.offload
    return min(spec.moe.E, (cfg.DRAM_BYTES - L.slots[0][0]) // (L.layers * L.pitch))


def card(model: str, ref: dict, n: int, experts: int, cap: int, pool: str | None,
         host_loop: bool = False, embed: str | None = None, trace: str | None = None,
         cfg_file: str | None = None, on_card: bool = False, policy: str = "lfu",
         embed_host: bool | None = None, hints: bool | None = None,
         hint_part: int | None = None, hint_drop: bool = False,
         hint_trace: str | None = None, wformat: str = "fp4", head_format: str = "int8",
         formats: str | None = None, layer_major: int = 0, pooled: bool = True,
         release_weights: bool = True, willneed: bool = True, pool_map: bool = True,
         legacy_serve: bool = False, embed_runs: bool = False) -> dict:
    import hashlib
    import pickle
    from dataclasses import replace
    from opentpu.isasim import board_config
    from opentpu.llm import load_spec
    from opentpu.llm.qwen3 import Engine, LazyWeights
    if formats is not None:                     # per-kind formats (Gemma 4: "experts=fp4"
        os.environ["OTPU_FORMATS"] = formats    # beside int8 layers; opentpu/llm/formats.py)
    spec = load_spec(model)                     # a MoE's Spec.embed: "int8" (from_hf)
    if embed is not None:
        spec = replace(spec, embed=embed)
    if hints is not None:                       # (default: the Spec's, Qwen3.5-MoE's on)
        spec = replace(spec, moe=replace(spec.moe, hint=hints))
    W = LazyWeights(model)
    cfg = pickle.loads(Path(cfg_file).read_bytes()) if cfg_file else board_config()
    ekw = {} if embed_host is None else {"embed_host": embed_host}   # (default: the image's)
    if not experts:
        experts = fit_experts(spec, cfg, cap, wformat=wformat, head_format=head_format,
                              lookup=True,
                              **ekw)
    backend = "isa"
    if on_card:
        from opentpu.host.board import BoardBackend, XdmaTransport
        tr = XdmaTransport("/dev/xdma0")
        backend = lambda c, imgs: BoardBackend(c, imgs, transport=tr,      # noqa: E731
                                               model=Path(model).name)
    t = time.time()
    if layer_major:                             # the prompt a layer at a time, runs of R rows
        ekw.update(layer_major=layer_major, pooled=pooled,  # (docs/offload.md 13)
                   embed_runs=embed_runs)           # (13.6: a bitstream with port A's fix)
    eng = Engine(spec, W, cap=cap, cfg=cfg, rows=1, wformat=wformat, head_format=head_format,
                 resident=True, experts=experts, pool_file=pool, backend=backend,
                 release_weights=release_weights, pool_map=pool_map,  # (10.6, 10.7)
                 **ekw)
    load_s = time.time() - t
    srv = eng.server
    srv.policy = policy                         # the slots' replacement (ExpertServer)
    srv.history, per_req = [], []               # each request's ids and misses
    serve, pool_of, mem = srv.serve, srv.pool, srv.mem
    tm = dict(serve=0.0, pool=0.0, write=0.0, read=0.0, stage=0.0, flush=0.0,   # host's s
              poll=0.0, hint=0.0)

    calls = dict.fromkeys(tm, 0)                # and how many calls

    def timed(part, f):
        def g(*a):
            t0 = time.perf_counter()
            try:
                return f(*a)
            finally:
                tm[part] += time.perf_counter() - t0
                calls[part] += 1
        return g

    def counted(ids_):
        m0 = srv.misses
        timed("serve", serve)(ids_)
        per_req.append(srv.misses - m0)
    srv.serve = counted
    srv.step = timed("hint", srv.step)          # a hinted expert's part on idle time (polls')
    if hint_part:                               # the hints' handling (ExpertServer)
        srv.part = hint_part
    srv.drop = hint_drop
    srv.pool = timed("pool", pool_of)
    mem.write, mem.read = timed("write", mem.write), timed("read", mem.read)
    if hasattr(mem, "write_slot"):              # BoardDram: staging (the main thread), and
        mem.write_slot = timed("stage", mem.write_slot)     # waiting for the DMA thread
        mem.flush = timed("flush", mem.flush)
    if callable(getattr(eng.backend, "host", None)):
        polled = eng.backend.host

        def served():                           # the polls that served a request: the card
            t0 = time.perf_counter()            # waits from its post to served (with the
            r = polled()                        # poll's own latency)
            if r:
                tm["poll"] += time.perf_counter() - t0
            return r
        eng.backend.host = served
    warm, pf = getattr(srv, "pool_warm", None), getattr(srv, "pool_file", None)
    if willneed and pf is not None:             # a request's misses read from the disk at once
        srv.ahead = pf.willneed
    if legacy_serve:                            # the serving of docs/offload.md 10.7 (A/B, 10.8):
        from opentpu.host import board as brd   # halves, buffer lists, the victims' entries
        mem.lead, srv.clear_late = None, False  # first, a 50 us sleep between polls, out[] read
        if pf is not None:                      # every poll
            pf.iov = False
        t_ = getattr(getattr(eng.backend, "board", None), "t", None)
        if t_ is not None:
            t_.host_idle = brd.HOST_IDLE
        brd.HOST_TAKE = 0.0

    def warm_at():                              # the pool file's packed experts: read by the
        if warm is None:                        # warm thread, and in the page cache
            return None
        r = pf.resident(pf.ids)
        return dict(read_gb=round(warm.bytes / 1e9, 2), done=not warm.is_alive(),
                    resident_gb=None if r is None else round(r / 1e9, 2))
    warm_load = warm_at()
    ids = ref["ids"]
    t = time.time()
    lg = eng.prefill(ids if host_loop else ids[:-1])
    prefill_s = time.time() - t
    pre = len(per_req) if layer_major else 0    # layer-major: a request per layer run
    warm_decode = warm_at()
    lg_sha = hashlib.sha256(np.asarray(lg, np.float32).tobytes()).hexdigest()[:16]
    tm0, b0, st0, calls0 = dict(tm), srv.bytes, len(eng.stats), dict(calls)
    if hint_trace:                              # the decode's timeline of hints and requests
        srv.events = []                         # (and BoardDram's DMA calls)
        if hasattr(mem, "calls"):
            mem.calls = []
    dma0 = (getattr(mem, "dma_s", 0.0), getattr(mem, "dma_bytes", 0))
    direct0 = getattr(mem, "direct", 0)         # experts read from the file into their runs
    wait0 = getattr(mem, "wait_s", 0.0)         # staging's waits for the DMA thread
    if pf is not None:                          # the decode's pool reads: page cache or disk
        pf.io = {}
    board = getattr(eng.backend, "board", None)     # the card's free-running counters
    snap = getattr(board, "snapshot", None)
    snap0, mem_decode = snap() if snap else None, host_mem()
    t = time.time()
    top = []                    # host loop: the device's 8 best (id, logit) per step

    def best(v):
        i = np.argsort(-v, kind="stable")[:8]
        top.append([[int(j), round(float(v[j]), 4)] for j in i])
        return int(i[0])
    if host_loop:               # the resident decode programs, the argmax on the host
        got = [best(lg)]
        for _ in range(n - 1):
            got.append(best(eng.step(got[-1])))
    else:                       # the prompt's last token fed by the card's loop: every pick
        got = eng.generate_card(ids[-1], n, stop_ids=[])        # on the card
    gen_s = time.time() - t
    snap1, mem_end, warm_end = snap() if snap0 else None, host_mem(), warm_at()
    if hint_trace:
        Path(hint_trace).write_text(json.dumps(srv.events))
        srv.events = None
        if getattr(mem, "calls", None) is not None:
            Path(hint_trace).with_suffix(".calls.json").write_text(json.dumps(mem.calls))
            mem.calls = None
    host = {k: round(tm[k] - tm0[k], 3) for k in tm}          # the decode's
    host.update(bytes=srv.bytes - b0, polls_read_s=host.pop("read"),
                polls_reads=calls["read"] - calls0["read"])     # (each a card read's DMA)
    if hasattr(mem, "dma_s"):                   # the DMA thread's own time and rate
        ds, db = mem.dma_s - dma0[0], mem.dma_bytes - dma0[1]
        host.update(dma_s=round(ds, 3), dma_gbs=round(db / ds / 1e9, 3) if ds else None,
                    memory=type(mem).__name__, direct=mem.direct - direct0)
    if hasattr(mem, "wait_s"):                  # stage = these waits + the pool's reads + copies
        host.update(stage_wait=round(mem.wait_s - wait0, 3))
    if pf is not None:
        host.update(reads={k: [v[0], round(v[1], 3), v[2], v[3]] for k, v in pf.io.items()})
        pf.io = None
    khz = (getattr(eng.backend, "info", None) or {}).get("core_khz")
    cyc = sum(s.get("cycles", 0) for s in eng.stats[st0:])       # (the simulator: none)
    dev_s = cyc / (khz * 1e3) if khz else None
    L = eng.image.offload
    J, E = L.layers, L.E
    T = (len(per_req) - pre) // J               # whole tokens (the last request may be unread)
    mpt = np.array(per_req[pre:pre + T * J]).reshape(T, J).sum(1)  # misses per token
    prompt = (0 if host_loop else 1) if pre else len(ids)   # the prompt's tokens among them
    dec = mpt[prompt:]
    if trace:                                   # the card's routes, as router_trace.py's
        req = np.array(srv.history[pre:pre + T * J]).reshape(T, J, -1)
        first = spec.moe.first
        z = {f"L{first + j}_idx": (req[:, j] - j * E).astype(np.int16) for j in range(J)}
        z.update({f"L{first + j}_ok": np.float64(1.0) for j in range(J)})
        meta = dict(model=model, moe_layers=[first + j for j in range(J)], experts=E,
                    k=spec.moe.k, source="moe_card: the card's routes")
        np.savez(trace, meta=json.dumps(meta), **z)
    diff = next((i for i, (a, b) in enumerate(zip(got, ref["tokens"])) if a != b), None)
    at = None
    if diff is not None and "top" in ref:       # HF's 8 best at the first different pick
        at = dict(hf=ref["tokens"][diff], card=got[diff], hf_top=ref["top"][diff],
                  card_top=top[diff] if top else None)
    return dict(tokens=got, match=got == ref["tokens"][:len(got)], first_diff=diff,
                at_first_diff=at,
                experts_per_layer=experts, slots=L.layers * experts, pool=L.layers * L.E,
                policy=policy, embed_host=bool(getattr(eng.image, "embed_host", False)),
                row_requests=eng.row_server.seq if eng.row_server is not None else None,
                hint=spec.moe.hint,
                hints=dict(served=srv.hints, prefetched=srv.prefetched, promoted=srv.promoted,
                           dropped=srv.dropped, withdrawn=srv.withdrawn, part=srv.part,
                           drop=srv.drop) if spec.moe.hint else None,
                image_mib=round(eng.image.nbytes / 2**20), slot_mb=round(L.slot_bytes / 1e6, 2),
                requests=len(srv.history), hits=srv.hits, misses=srv.misses,
                misses_per_token_decode=round(float(dec.mean()), 2) if len(dec) else None,
                expert_uses_per_token=J * spec.moe.k,
                misses_per_token_decode_2nd_half=round(float(dec[len(dec) // 2:].mean()), 2)
                if len(dec) else None,
                misses_per_token=mpt.tolist(), layer_major=layer_major,
                pooled=pooled if layer_major else None,
                embed_runs=embed_runs if layer_major else None,
                release_weights=release_weights, willneed=willneed, pool_map=pool_map,
                legacy_serve=legacy_serve,
                prefill_requests=pre or None, prefill_misses=sum(per_req[:pre]) if pre else None,
                load_s=round(load_s), prefill_s=round(prefill_s), generate_s=round(gen_s),
                loop="host" if host_loop else "card", embed=spec.embed,
                backend="card" if on_card else "isa", cfg=cfg_file, prefill_logits_sha=lg_sha,
                decode_cycles=cyc, core_khz=khz,
                tok_s_wall=round(len(got) / gen_s, 2) if gen_s else None,
                tok_s_device=round(len(got) / dev_s, 2) if dev_s else None,
                host_decode_s=host,
                pool_warm=dict(packed_gb=round(len(pf.ids) * pf.slot / 1e9, 2),
                               split=pf.split, resident_gb_at_open=None
                               if pf.resident_at_open is None else
                               round(pf.resident_at_open / 1e9, 2),
                               at_load=warm_load, at_decode=warm_decode, at_end=warm_end)
                if warm else None,
                host_mem=dict(at_decode=mem_decode, at_end=mem_end),
                device_counters={k: v - snap0[k] for k, v in snap1.items()} if snap1 else None,
                misses_per_request_decode=per_req[pre + prompt * J:pre + T * J],
                bytes_per_token_decode=round(host["bytes"] / max(1, len(got))),
                top=top or None)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model")
    ap.add_argument("--hf", help="write HF's greedy tokens here")
    ap.add_argument("--check", help="HF's tokens (--hf's output) to compare the card's with")
    ap.add_argument("-n", type=int, default=16, help="tokens to generate")
    ap.add_argument("--prompt", default=PROMPT, help="HF: the user turn")
    ap.add_argument("--experts", type=int, default=0,
                    help="expert slots per MoE layer (0: as many as fit)")
    ap.add_argument("--cap", type=int, default=4096, help="KV capacity")
    ap.add_argument("--pool", help="the expert pool file (each expert packed once, reused)")
    ap.add_argument("--max-memory", help="HF: host RAM for weights, the rest to disk (e.g. 10GiB)")
    ap.add_argument("--out", help="the card's result as JSON")
    ap.add_argument("--embed", choices=["int8", "f32"], default=None,
                    help="Spec.embed (default the model's: int8 for a MoE, the rows gathered on "
                         "the device from the int8 head or table), or an fp32 table (1 GiB for "
                         "LFM2.5-8B-A1B)")
    ap.add_argument("--trace", help="save the card's routes as a router trace (cachesim.py)")
    ap.add_argument("--host-loop", action="store_true",
                    help="decode with the resident step programs and the argmax on the host, "
                         "the device's 8 best logits per step in the result (a diagnostic: "
                         "where a run parts from HF's)")
    ap.add_argument("--cfg", help="the configuration, pickled (tools/qual/refs.py cfg: the "
                                  "bitstream's); default isasim.board_config()")
    ap.add_argument("--card", action="store_true",
                    help="run on the card (/dev/xdma0) instead of the ISA simulator")
    ap.add_argument("--policy", choices=("lru", "lfu"), default="lfu",
                    help="the expert slots' replacement (ExpertServer): least recently used, or "
                         "least decayed use")
    ap.add_argument("--embed-table", choices=("host", "card"), default=None,
                    help="an untied int8 embedding table on the host (embed_host: the card asks "
                         "for each token's row; a MoE's default) or on the card")
    ap.add_argument("--wformat", default="fp4", help="the weights' format (the experts' too)")
    ap.add_argument("--head-format", default="int8", help="the LM head's format")
    ap.add_argument("--formats", help="per-kind formats, OTPU_FORMATS (Gemma 4: experts=fp4)")
    ap.add_argument("--hint-part", type=int, default=0,
                    help="KiB of a hinted expert per idle poll (default: ExpertServer's 512)")
    ap.add_argument("--hint-drop", action="store_true",
                    help="a request withdraws its layer's hinted experts it does not name")
    ap.add_argument("--hint-trace", help="the decode's hint and request timeline as JSON")
    ap.add_argument("--hints", choices=("on", "off"), default=None,
                    help="the router's prefetch hints before each mixer (docs/offload.md 12; "
                         "default: the model's, on for Qwen3.5-MoE)")
    ap.add_argument("--layer-major", type=int, default=0, metavar="R",
                    help="prefill a layer at a time in runs of R rows (docs/offload.md 13; "
                         "default 0: token by token)")
    ap.add_argument("--per-layer-slots", action="store_true",
                    help="--layer-major with each layer's own slots (default: pooled)")
    ap.add_argument("--embed-runs", action="store_true",
                    help="--layer-major with the embed runs and compile-time-position runs "
                         "for an embedding table on the host (Engine embed_runs; needs a "
                         "bitstream whose port A drops its beats at RUN: docs/offload.md 13.6)")
    ap.add_argument("--keep-weights", action="store_true",
                    help="keep the checkpoint mapped after the image is built (by default "
                         "LazyWeights.release gives its pages back: page cache for the pool; "
                         "docs/offload.md 10.6)")
    ap.add_argument("--no-willneed", action="store_true",
                    help="read a request's misses from the pool one after another (by default "
                         "PoolFile.willneed queues those not in the page cache at once)")
    ap.add_argument("--no-pool-map", action="store_true",
                    help="read the pool through read() only (by default each read is also "
                         "touched through a read-only map of the pool: its page cache's standing "
                         "under MGLRU, docs/offload.md 10.7)")
    ap.add_argument("--legacy-serve", action="store_true",
                    help="serve as before docs/offload.md 10.8 (A/B): a request's first miss in "
                         "halves, its reads through buffer lists, each victim's entry cleared "
                         "before its slot is written, a 50 us sleep between empty polls")
    ap.add_argument("--release-weights", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--willneed", action="store_true", help=argparse.SUPPRESS)  # (the default)
    a = ap.parse_args()
    if a.hf:
        r = hf_greedy(a.model, a.n, a.max_memory, a.prompt)
        Path(a.hf).write_text(json.dumps(r, indent=1))
        print(json.dumps(r))
        return
    ref = json.loads(Path(a.check).read_text())
    r = card(a.model, ref, a.n, a.experts, a.cap, a.pool, a.host_loop, a.embed, a.trace,
             a.cfg, a.card, a.policy,
             None if a.embed_table is None else a.embed_table == "host",
             None if a.hints is None else a.hints == "on", a.hint_part << 10, a.hint_drop,
             a.hint_trace, a.wformat, a.head_format, a.formats, a.layer_major,
             not a.per_layer_slots, not a.keep_weights, not a.no_willneed,
             not a.no_pool_map, a.legacy_serve, a.embed_runs)
    print(json.dumps(r))
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
