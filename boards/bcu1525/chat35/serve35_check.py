"""serve35_check.py: otpu-serve35's first card check, from any machine that reaches its port (stdlib only):
  python3 serve35_check.py [--url http://127.0.0.1:8000] [--key KEY]
/health and /v1/models (max_model_len 262144); conversation A's first turn streamed with include_usage (TTFT, decode
tok/s, the usage chunk last); conversation B in between (A's context saved to the store); A's second turn (written
back: its cached tokens at least the common prefix of A1's and A2's renders (/tokenize), the request's swaps or the
store's restores); /flush_cache, then A's second turn again from nothing (its greedy text the restored one's: the
35B's contexts through the staging area bit for bit); /api/stats' card (the governor's die, RUN, power), each request's
TTFT by phase (a save's and a restore's by part, with the bytes moved; the last idle save's too) and /metrics'
names. Greedy (temperature 0) throughout.
It forgets the server's saved contexts (the flush): run it on a fresh server. A line a check, PASS or FAIL;
exit 1 on a FAIL. About 2-4 minutes on the card (a few short prompts, ~250 tokens)."""
import argparse
import http.client
import json
import sys
import time
from urllib.parse import urlparse

FAILS = []


def check(ok: bool, what: str) -> bool:
    print(f"{'PASS' if ok else 'FAIL'}  {what}", flush=True)
    if not ok:
        FAILS.append(what)
    return ok


class Client:
    def __init__(self, url: str, key: str | None):
        u = urlparse(url)
        self.host, self.port = u.hostname, u.port or 80
        self.h = {"Content-Type": "application/json"}
        if key:
            self.h["Authorization"] = f"Bearer {key}"

    def call(self, method: str, path: str, body=None, timeout: float = 900):
        c = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        c.request(method, path, None if body is None else json.dumps(body), self.h)
        r = c.getresponse()
        data = r.read()
        c.close()
        ct = r.getheader("Content-Type", "")
        return r.status, json.loads(data) if ct.startswith("application/json") else data.decode()

    def stream(self, body, timeout: float = 900):
        """(text, usage, ttft_s, total_s, chunks) of a streamed chat completion."""
        c = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        t0 = time.perf_counter()
        c.request("POST", "/v1/chat/completions", json.dumps({**body, "stream": True}), self.h)
        r = c.getresponse()
        if r.status != 200:
            raise SystemExit(f"stream: HTTP {r.status}: {r.read()[:400]!r}")
        text, usage, ttft, chunks = [], None, None, []
        while True:
            line = r.readline()
            if not line:
                break
            line = line.decode().rstrip("\n")
            if not line.startswith("data: "):
                continue
            if line == "data: [DONE]":
                chunks.append("DONE")
                break
            ch = json.loads(line[6:])
            chunks.append(ch)
            for cc in ch.get("choices") or ():
                d = cc.get("delta", {}).get("content")
                if d:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    text.append(d)
            if ch.get("usage"):
                usage = ch["usage"]
        c.close()
        return "".join(text), usage, ttft, time.perf_counter() - t0, chunks


PARTS = ("store", "host", "prog", "link", "load", "run", "adopt")    # a save's or restore's (scheduler._io_parts)


def print_parts(label: str, total, sub: dict, io: dict | None) -> None:
    """A save's or restore's seconds by part (sub: part -> s) and the bytes it moved (io)."""
    if not sub:
        return
    io = io or {}
    print(f"        {label} {total or 0:.4f} s: " + " + ".join(f"{k} {sub[k]:.4f}" for k in PARTS if k in sub)
          + f" (within: gc {sub.get('gc', 0):.4f}, a card sample {sub.get('telemetry', 0):.4f}); link "
          f"{io.get('link_bytes', 0) / 1e6:.2f} MB in "
          f"{io.get('link_calls', 0)} calls, {io.get('copies', 0)} copy runs of "
          f"{io.get('copied_bytes', 0) / 1e6:.2f} MB")


def common(a, b) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--key")
    ap.add_argument("--model", default="qwen36-35b")
    a = ap.parse_args()
    cl = Client(a.url, a.key)
    st, h = cl.call("GET", "/health")
    check(st == 200 and h.get("status") == "ok", f"/health {st} {h}")
    st, m = cl.call("GET", "/v1/models")
    mm = (m.get("data") or [{}])[0] if st == 200 else {}
    check(mm.get("id") == a.model and mm.get("max_model_len") == 262144,
          f"/v1/models {mm.get('id')} max_model_len {mm.get('max_model_len')}")

    # A's first turn, streamed
    a1 = [{"role": "system", "content": "You are a concise assistant."},
          {"role": "user", "content": "Name three uses of an FPGA in a data centre, one line each."}]
    text1, u1, ttft, total, chunks = cl.stream({"model": a.model, "messages": a1, "max_tokens": 96,
                                                "temperature": 0,
                                                "stream_options": {"include_usage": True}})
    n1 = (u1 or {}).get("completion_tokens", 0)
    dec = (n1 - 1) / (total - ttft) if ttft and n1 > 1 and total > ttft else None
    print(f"      A1: {n1} tokens, TTFT {ttft:.2f} s, decode {dec or 0:.2f} tok/s: {text1[:120]!r}")
    check(bool(text1) and chunks[-1] == "DONE" and isinstance(chunks[-2], dict) and
          chunks[-2].get("choices") == [] and u1 is not None, "A1 streamed: the usage chunk last, then [DONE]")
    check(dec is not None and dec > 25, f"A1 decode {dec or 0:.2f} tok/s (chat35: 33)")

    # B in between: A's context goes to the store
    st, rb = cl.call("POST", "/v1/chat/completions",
                     {"model": a.model, "max_tokens": 32, "temperature": 0,
                      "messages": [{"role": "user", "content": "What is 17 times 23? Answer with the number."}]})
    check(st == 200 and rb["usage"]["completion_tokens"] > 0,
          f"B {st}: {rb['choices'][0]['message']['content'][:60]!r}" if st == 200 else f"B {st} {rb}")

    # A's second turn: written back from the store
    a2 = a1 + [{"role": "assistant", "content": text1},
               {"role": "user", "content": "Which of the three needs the most memory bandwidth?"}]
    body2 = {"model": a.model, "messages": a2, "max_tokens": 64, "temperature": 0}
    t0 = time.perf_counter()
    st, r2 = cl.call("POST", "/v1/chat/completions", body2)
    t2 = time.perf_counter() - t0
    c2 = r2["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0) if st == 200 else 0
    text2 = r2["choices"][0]["message"]["content"] if st == 200 else ""
    p1 = u1["prompt_tokens"] if u1 else 0
    # what A2 can reuse: the common prefix of A1's and A2's renders (A1's generation prompt ends with the empty
    # thinking block that the history's rendering of its reply leaves out: 4 tokens on the Qwen template)
    sa, k1 = cl.call("POST", "/tokenize", {"model": a.model, "messages": a1})
    sb, k2 = cl.call("POST", "/tokenize", {"model": a.model, "messages": a2})
    pre = common(k1["tokens"], k2["tokens"]) if sa == sb == 200 else None
    check(sa == sb == 200 and k1["count"] == p1, f"/tokenize: A1 {k1.get('count') if sa == 200 else sa} tokens "
          f"(A1's usage {p1}), A2 {k2.get('count') if sb == 200 else sb}, common prefix {pre}")
    check(st == 200 and pre is not None and c2 >= pre,
          f"A2 restored: {c2} of {r2['usage']['prompt_tokens'] if st == 200 else '?'} prompt tokens cached "
          f"(A1's and A2's common prefix {pre}, A1's prompt {p1}), {t2:.2f} s")
    time.sleep(1.5)                     # (/api/stats: the monitor's last tick, a second apart)
    st, s = cl.call("GET", "/api/stats")
    row = next((x for x in s.get("recent", ()) if x.get("completion") == r2["usage"]["completion_tokens"]), {}) \
        if st == 200 else {}
    kv = s.get("kv", {}) if st == 200 else {}
    check(kv.get("restores", 0) >= 1 or row.get("swaps", 0) >= 1,
          f"the store wrote a context back: restores {kv.get('restores')}, saves {kv.get('saves')}, "
          f"A2 swaps {row.get('swaps')}")

    # the same turn from nothing: its greedy text the restored one's
    st, _ = cl.call("POST", "/flush_cache")
    check(st == 200, "/flush_cache")
    st, r3 = cl.call("POST", "/v1/chat/completions", body2)
    c3 = r3["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0) if st == 200 else -1
    text3 = r3["choices"][0]["message"]["content"] if st == 200 else None
    check(st == 200 and c3 == 0, f"A2 fresh: {c3} cached tokens (0 after the flush)")
    check(text3 == text2, "A2 restored = A2 fresh (greedy text)" + ("" if text3 == text2 else
          f":\n      restored {text2!r}\n      fresh    {text3!r}"))

    # the card's telemetry and /metrics
    st, s = cl.call("GET", "/api/stats")
    c = (s.get("card") or {}) if st == 200 else {}
    check(c.get("die_src") == "governor" and isinstance(c.get("die_c"), (int, float)),
          f"card die {c.get('die_c')} C from {c.get('die_src')} (card_note {s.get('card_note')!r}, "
          f"card_error {s.get('card_error')!r})")
    check((c.get("gov") or {}).get("state") == "RUN", f"governor {(c.get('gov') or {}).get('state')}")
    p = c.get("power") or {}
    check(p.get("w") is not None, f"power {p.get('w')} W ({p.get('source')}: {p.get('label')})")
    u = c.get("util") or {}
    print(f"      utilization {u}  DRAM {c.get('dram_gbs')} GB/s  monitor {s.get('monitor')}")
    # each request's time to its first token by phase (the worker's: queue, switch, prompt)
    # (the first token: what is left of the TTFT, the decode loop's first run with the prompt's last token)
    for x in reversed(s.get("recent") or [] if st == 200 else []):
        ph = x.get("phases") or {}
        q, sw, pf = x.get("queue_s") or 0, ph.get("switch", 0), x.get("prefill_s") or 0
        first = x["ttft_s"] - q - sw - pf if x.get("ttft_s") is not None else None
        print(f"      {x.get('rid')}: prompt {x.get('prompt')} cached {x.get('cached')} TTFT {x.get('ttft_s')} s = "
              f"queue {q} (idle save {ph.get('idle_save', 0)}) + switch {sw} (save {ph.get('save', 0)}, restore "
              f"{ph.get('restore', 0)}, rollback {ph.get('rollback', 0)}) + prompt {pf} (snapshot "
              f"{ph.get('snapshot', 0)}) + first token {'?' if first is None else f'{first:.3f}'}")
        for kind in ("save", "restore"):
            sub = {k: ph[f"{kind}.{k}"] for k in PARTS + ("gc", "telemetry") if f"{kind}.{k}" in ph}
            print_parts(kind, ph.get(kind), sub, (x.get("io") or {}).get(kind))
    kv = s.get("kv") or {}
    for kind, v in sorted((kv.get("last_io") or {}).items()):     # (an idle save's too)
        print_parts(f"the last {kind}", v.get("s"), {k: v[k] for k in PARTS + ("gc", "telemetry") if k in v}, v)
    pc = kv.get("prefix") or {}
    print(f"      store: saves {kv.get('saves')} (at idle {kv.get('idle_saves')}, idle saves pre-empted "
          f"{kv.get('idle_preempts')}) restores {kv.get('restores')} device reuses {kv.get('device_reuses')} snapshots "
          f"{kv.get('snapshots')} rollbacks {kv.get('rollbacks')}; prefixes kept {pc.get('prefix_saves')}, restored "
          f"{pc.get('prefix_hits')} ({pc.get('prefix_hit_tokens')} tokens)")
    st, text = cl.call("GET", "/metrics")
    names = {ln.split("{")[0] for ln in text.splitlines() if ln and not ln.startswith("#")} if st == 200 else set()
    want = {"vllm:num_requests_running", "vllm:generation_tokens_total", "vllm:time_to_first_token_seconds_bucket",
            "vllm:kv_cache_usage_perc", "otpu:die_temp_celsius", "otpu:power_watts", "otpu:governor_run",
            "otpu:utilization_ratio"}
    check(st == 200 and want <= names, f"/metrics: {len(names)} names" +
          (f", missing {sorted(want - names)}" if want - names else ""))
    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILED'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
