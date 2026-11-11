#!/usr/bin/env python3
"""lfm_check.py: the checks of otpu-serve35x2 on LFM2-2.6B (CHAT35_MODEL=lfm2-2.6b), a client of its API (no card, tokenizer or model of its own):

  lfm_check.py refs URL OUT.json    the replies of a server over the ISA simulator: greedy at the device's loop, greedy
                                    under the default penalty, the default sampling with a seed
  lfm_check.py check URL REFS.json  the same requests on the cards, each reply word for word against its reference
  lfm_check.py perf URL             decode on a short prompt and after a long one, the long prompt's time to its first
                                    token, its next turn's (the reply's start restored)
  lfm_check.py replay URL CAP.jsonl a coding client's captured bodies as sent: the main prompt, 10 ms later its title
                                    request, then the next turn: each one's usage, what the worker fed and restored

Each prints the programs compiled and wait entries computed over its requests (a warmed server: 0). Exit 1: a reply
differs or a request failed."""
from __future__ import annotations

import http.client
import json
import sys
import threading
import time
from urllib.parse import urlparse

MAX_NEW = 24
LONG_N = 594
PERF = (100, 400, (2560, 12900), 100)   # (the short prompt, its tokens; the long prompts; the next turn's new tokens)
LONG = "The history of computing is a long story of machines that got faster and smaller. "
RAW = ["The capital of France is", "def fibonacci(n):"]
CHAT = ["Write a Python function that reverses a linked list.",
        "What does the 'git rebase' command do? Answer in two sentences."]
SAMPLING = {"temperature": 0.1, "top_k": 50, "top_p": 1.0, "repetition_penalty": 1.05}


def request(url: str, method: str, path: str, body=None, timeout: float = 14400.0):
    u = urlparse(url)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
    c.request(method, path, None if body is None else json.dumps(body),
              {"Content-Type": "application/json"})
    return c.getresponse()


def call(url: str, path: str, body=None):
    r = request(url, "POST" if body is not None else "GET", path, body)
    data = r.read()
    if r.status != 200:
        raise RuntimeError(f"{path}: HTTP {r.status}: {data[:400]!r}")
    return json.loads(data)


def counters(url: str) -> dict:
    d = call(url, "/v1/stats")
    pc = d.get("prog_cache") or {}
    return {"saves": d.get("saves"), "restores": d.get("restores"), "device_reuses": d.get("device_reuses"),
            "prefill_tokens": d.get("prefill_tokens"), "compiled": pc.get("programs_compiled"),
            "entries_computed": pc.get("entries_computed")}


def show_counters(url: str, a: dict) -> dict:
    b = counters(url)
    print("counters", {k: (b[k] or 0) - (a[k] or 0) for k in b}, flush=True)
    return b


def prompts(url: str) -> list:
    """[(label, ids)]: lfmtok's five, as Hugging Face's reference run took them."""
    def tok(**body):
        return call(url, "/tokenize", body)["tokens"]
    out = [(f"raw {p[:24]!r}", tok(prompt=p)) for p in RAW]
    out += [(f"chat {p[:24]!r}", tok(messages=[{"role": "user", "content": p}])) for p in CHAT]
    long_ids = tok(prompt=LONG * 37)[:LONG_N]
    return out + [(f"long {len(long_ids)}", long_ids)]


def cases(url: str) -> list:
    """[(label, ids, sampling_params)]."""
    ps = prompts(url)
    greedy = {"temperature": 0, "repetition_penalty": 1.0}
    out = [(f"greedy {label}", ids, greedy) for label, ids in ps]
    out.append((f"t0 penalty {ps[2][0]}", ps[2][1], dict(SAMPLING, temperature=0)))
    out += [(f"sampled {label}", ids, dict(SAMPLING, sampling_seed=7 + i))
            for i, (label, ids) in enumerate(ps[2:4])]
    return out


def generate(url: str, ids: list, sp: dict) -> dict:
    t0 = time.time()
    r = call(url, "/generate", {"input_ids": ids, "sampling_params": dict(
        sp, max_new_tokens=MAX_NEW, ignore_eos=True, skip_special_tokens=False)})
    r["seconds"] = round(time.time() - t0, 2)
    return r


def refs(url: str, out: str) -> int:
    got, a = [], counters(url)
    for label, ids, sp in cases(url):
        r = generate(url, ids, sp)
        got.append({"label": label, "ids": ids, "sampling_params": sp, "output_ids": r["output_ids"],
                    "text": r["text"]})
        print(f"{label}: {len(ids)} + {len(r['output_ids'])} tokens, {r['seconds']:.0f} s: {r['text'][:80]!r}",
              flush=True)
        with open(out, "w") as f:
            json.dump(got, f)
    show_counters(url, a)
    return 0


def check(url: str, path: str) -> int:
    bad, want, a = 0, json.load(open(path)), counters(url)
    for c in want:
        r = generate(url, c["ids"], c["sampling_params"])
        same = r["output_ids"] == c["output_ids"]
        bad += not same
        first = next((i for i, (x, y) in enumerate(zip(r["output_ids"], c["output_ids"])) if x != y), None)
        print(f"{'SAME' if same else 'DIFF at ' + str(first)} {c['label']}: {r['seconds']:.2f} s, "
              f"{r['text'][:70]!r}" + ("" if same else f" ref {c['text'][:70]!r}"), flush=True)
    show_counters(url, a)
    print(f"lfm_check check: {len(want) - bad} of {len(want)} replies equal", flush=True)
    return 1 if bad else 0


def stream(url: str, tag: str, path: str, body: dict, out: dict | None = None) -> dict:
    """A streamed request: the seconds to its first delta and its last, its deltas' count, usage, text."""
    t0, first, last, n, text, usage = time.time(), None, None, 0, [], None
    r = request(url, "POST", path, body)
    if r.status != 200:
        raise RuntimeError(f"{tag}: HTTP {r.status}: {r.read()[:400]!r}")
    d = {}
    for ln in r:
        ln = ln.decode().strip()
        if not ln.startswith("data: {"):
            continue
        d = json.loads(ln[6:])
        usage = d.get("usage") or usage
        for ch in d.get("choices", []):
            dl = ch.get("delta") or {}
            part = (dl.get("content") or "") + (dl.get("reasoning_content") or "") + (ch.get("text") or "")
            if dl.get("tool_calls"):
                part += json.dumps(dl["tool_calls"])
            if part:
                last = time.time()
                first = first or last
                n += 1
                text.append(part)
    res = dict(first=None if first is None else first - t0, last=None if last is None else last - t0, deltas=n,
               usage=usage, text="".join(text), model=d.get("model"))
    print(f"{tag}: first delta {res['first'] or 0:.2f} s, last {res['last'] or 0:.2f} s, {n} deltas, model field "
          f"{res['model']!r}, usage {json.dumps(usage)}, text {res['text'][:160]!r}", flush=True)
    if out is not None:
        out[tag] = res
    return res


def perf(url: str) -> int:
    a = counters(url)

    def run(tag, ids, n):
        r = stream(url, tag, "/v1/completions", {
            "prompt": ids, "max_tokens": n, "ignore_eos": True, "stream": True, **SAMPLING,
            "stream_options": {"include_usage": True}})
        k = r["usage"]["completion_tokens"]
        print(f"  {tag}: {len(ids)} tokens, cached {(r['usage'].get('prompt_tokens_details') or {}).get('cached_tokens')}: "
              f"first token {r['first']:.2f} s, {(len(ids) - 1) / r['first']:.1f} tok/s; decode "
              f"{(k - 1) / (r['last'] - r['first']):.2f} tok/s over {k} tokens", flush=True)
        return r
    short, n, longs, more = PERF
    ids = call(url, "/tokenize", {"prompt": LONG * 760})["tokens"]
    run("short", ids[:short], n)
    for p in longs:
        run(f"prompt {p}", ids[:p], n // 2)
    p = longs[-1]
    r = run(f"again {p} + {more}", ids[:p + more], more)
    show_counters(url, a)
    return 0 if r["first"] is not None else 1


def replay(url: str, path: str) -> int:
    bodies = sorted((json.loads(ln)["body"] for ln in open(path)), key=lambda b: len(json.dumps(b)))
    title, main = bodies[0], bodies[-1]
    a, out = counters(url), {}
    print("models", json.dumps(call(url, "/v1/models")), flush=True)
    ts = [threading.Thread(target=stream, args=(url, "main", "/v1/chat/completions", main, out))]
    ts[0].start()
    time.sleep(0.01)
    ts.append(threading.Thread(target=stream, args=(url, "title", "/v1/chat/completions", title, out)))
    ts[1].start()
    for t in ts:
        t.join()
    a = show_counters(url, a)
    msgs = main["messages"] + [{"role": "assistant", "content": out["main"]["text"] or "ok"},
                               {"role": "user", "content": "Now say it again."}]
    stream(url, "next", "/v1/chat/completions", dict(main, messages=msgs), out)
    show_counters(url, a)
    print(f"last_io {json.dumps(call(url, '/v1/stats').get('last_io'))[:600]}", flush=True)
    return 0 if len(out) == 3 else 1


def main(argv: list) -> int:
    if len(argv) < 2 or argv[0] not in ("refs", "check", "perf", "replay"):
        print(__doc__)
        return 2
    cmd, url, rest = argv[0], argv[1].rstrip("/"), argv[2:]
    return refs(url, *rest) if cmd == "refs" else check(url, *rest) if cmd == "check" else \
        perf(url) if cmd == "perf" else replay(url, *rest)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
