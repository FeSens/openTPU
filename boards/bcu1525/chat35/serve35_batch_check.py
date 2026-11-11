"""serve35_batch_check.py: otpu-serve35 --slots N's card check (batched decode: opentpu/serve/batching35.py), from any
machine that reaches its port (stdlib only, serve35_check.py's client):
  python3 serve35_batch_check.py [--url http://127.0.0.1:8000] [--key KEY] [--max-tokens 96] [--chats N]
/v1/stats' slots (2 or more); /flush_cache; prompts A, B, C and D alone, one at a time (each one's greedy reply, its
reference, and its decode tok/s alone); A and B at once (each reply its reference's word for word: a batched step's
rows are single sequences' bit for bit; each one's decode tok/s and the rows' summed); then A, B and C, then A, B, C
and D at once (past the slots, the rest wait for one; all their references); after each, the batched steps it ran and
their cycles a step from /v1/stats' step timing; the worker's step timing at the end (programs compiled ahead or
inline, reused, the host's waits for them). Greedy (temperature 0), then production's sampling (temperature 0.7, top_k
20, top_p 0.8) with a seed a request: as many of A, B, C and D as there are slots (2 at least) alone, then at once,
each reply its alone's (a sampled row picks as the single sequence's sampler for the same seed; at S > 1 too).
--chats N (up to 4): N conversations of three turns (the first one's opening a ~1.6k-token ledger, --paste-lines
lines; --chat-tokens a reply), each alone after a /flush_cache, then all at once after another: every turn's reply its
alone's, each turn's TTFT and decode tok/s. It forgets the server's saved contexts (the flushes): run it on a fresh
server. A line a check, PASS or FAIL; exit 1 on a FAIL. A few minutes on the card. --replies FILE: the replies alone (a
JSON of greedy, sampled and chats), for a card A/B's comparison of two servers word for word."""
import argparse
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from serve35_check import FAILS, Client, check        # noqa: E402

PROMPTS = {
    "A": "List the planets of the solar system in order from the Sun, one line each with one fact about it.",
    "B": "Write a short Python function that returns the n-th Fibonacci number iteratively, then explain it.",
    "C": "Explain in a few sentences why the sky is blue and why sunsets are red.",
    "D": "Name four rivers of Europe and the countries each one flows through.",
}
NAMES = ("Ada", "Ben", "Cleo", "Dev", "Eli", "Fay")


def paste(lines: int) -> str:
    """A ledger of `lines` entries (120: ~1.6k tokens), the first conversation's opening."""
    return "\n".join(f"entry {i}: {NAMES[i % len(NAMES)]} paid {(i * 37) % 500 + 10} euros"
                     for i in range(1, lines + 1))


CHATS = [                                       # --chats' conversations, three turns each ("{paste}": the ledger)
    ["{paste}\n\nWhich entry has the highest amount, and who paid it?", "What did entry 42 pay?",
     "Add the amounts of entries 7 and 99."],
    ["Suggest a name for a small bakery and explain it in two sentences.", "Now suggest two more in the same style.",
     "Which of the three would you pick, and why?"],
    ["What is a prime number? Give the first ten.", "Is 91 prime? Show why.",
     "Write a short Python check for primality."],
    ["Describe the water cycle in four steps.", "Which step does the Sun drive most directly?",
     "Sum all of it up in one sentence."],
]


def find(d, key):
    """The first value of `key` anywhere in the JSON d."""
    if isinstance(d, dict):
        if key in d:
            return d[key]
        d = list(d.values())
    if isinstance(d, list):
        for v in d:
            got = find(v, key)
            if got is not None:
                return got
    return None


SEEDS = {"A": 11, "B": 3, "C": 7, "D": 5}      # the sampled requests' (production's sampling)


def body(messages: list, n: int, model: str, seed: int | None = None) -> dict:
    out = {"model": model, "messages": messages, "max_tokens": n,
           "temperature": 0, "stream_options": {"include_usage": True},
           "chat_template_kwargs": {"enable_thinking": False}}
    if seed is not None:
        out.update(temperature=0.7, top_k=20, top_p=0.8, seed=seed)
    return out


def ask(c: Client, b: dict) -> tuple:
    """(text, completion tokens, ttft s, decode tok/s, total s) of one streamed request."""
    text, usage, ttft, total, _ = c.stream(b)
    k = (usage or {}).get("completion_tokens") or 0
    dec = (k - 1) / (total - ttft) if ttft is not None and k > 1 and total > ttft else 0.0
    return text, k, ttft or 0.0, dec, total


def run(c: Client, names, n: int, model: str, sampled: bool = False) -> dict:
    """names' requests at once: name -> (text, completion tokens, ttft s, decode tok/s, total s)."""
    out = {}

    def one(name):
        out[name] = ask(c, body([{"role": "user", "content": PROMPTS[name]}], n, model,
                                SEEDS[name] if sampled else None))
    ts = [threading.Thread(target=one, args=(x,)) for x in names]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return out


def show(label: str, got: dict, t: float) -> None:
    toks = sum(v[1] for v in got.values())
    print(f"      {label}: " + "; ".join(f"{k} {v[1]} tokens, TTFT {v[2]:.2f} s, decode {v[3]:.1f} tok/s"
                                         for k, v in sorted(got.items()))
          + f"; {toks} tokens in {t:.1f} s ({toks / t:.1f} tok/s in all, decode "
          f"{sum(v[3] for v in got.values()):.1f} tok/s summed)", flush=True)


def timing(c: Client) -> dict:
    st, stats = c.call("GET", "/v1/stats")
    tm = find(stats, "step_timing") if st == 200 else None
    return tm if isinstance(tm, dict) else {}


def show_steps(label: str, t0: dict, t1: dict) -> None:
    """The batched steps between two step timings: their count (programs run, compiled or reused), cycles a step."""
    d = {k: t1.get(k, 0) - t0.get(k, 0) for k in ("programs", "reused", "cycles", "device_s", "run_s")}
    n = d["programs"] + d["reused"]
    print(f"      {label} steps: {n} ({d['programs']} programs new), "
          f"{d['cycles'] / n if n else 0:,.0f} cycles a step, device {d['device_s']:.2f} s, "
          f"run {d['run_s']:.2f} s", flush=True)


def converse(c: Client, turns: list, n: int, model: str) -> list:
    """One conversation's turns in order, each with the replies so far: [(text, tokens, ttft, decode, total)]."""
    msgs, out = [], []
    for t in turns:
        msgs.append({"role": "user", "content": t})
        out.append(ask(c, body(list(msgs), n, model)))
        msgs.append({"role": "assistant", "content": out[-1][0]})
    return out


def chats(c: Client, k: int, n: int, lines: int, model: str) -> list:
    """--chats: k conversations alone (the references), then at once, every turn its reference's word for word.
    Returns the replies alone."""
    convs = [[t.replace("{paste}", paste(lines)) for t in ts] for ts in CHATS[:k]]
    got = {}
    for label, at_once in (("alone", False), ("at once", True)):
        st, _ = c.call("POST", "/flush_cache")
        check(st == 200, f"/flush_cache before the chats {label}")
        out, t0, tm0 = {}, time.perf_counter(), timing(c)

        def one(i):
            out[i] = converse(c, convs[i], n, model)
        if at_once:
            ts = [threading.Thread(target=one, args=(i,)) for i in range(k)]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
        else:
            for i in range(k):
                one(i)
        dt = time.perf_counter() - t0
        for i in range(k):
            print(f"      chat {i} {label}: " + "; ".join(
                f"turn {j} {r[1]} tokens, TTFT {r[2]:.2f} s, decode {r[3]:.1f} tok/s" for j, r in enumerate(out[i])),
                flush=True)
        toks = sum(r[1] for i in range(k) for r in out[i])
        print(f"      {k} chats {label}: {toks} tokens in {dt:.1f} s", flush=True)
        show_steps(f"{k} chats {label}", tm0, timing(c))
        got[label] = out
    for i in range(k):
        for j, (r, ref) in enumerate(zip(got["at once"][i], got["alone"][i])):
            check(r[0] == ref[0] and r[1] == ref[1] and r[1] > 1,
                  f"chat {i} turn {j} beside {k - 1} other chats: its reply alone's "
                  f"({r[1]} tokens{'' if r[0] == ref[0] else ', the text differs'})")
    return [[r[0] for r in got["alone"][i]] for i in range(k)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--key")
    ap.add_argument("--model", default="qwen36-35b")
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--chats", type=int, default=0, choices=range(len(CHATS) + 1))
    ap.add_argument("--chat-tokens", type=int, default=64)
    ap.add_argument("--paste-lines", type=int, default=120)
    ap.add_argument("--replies")
    a = ap.parse_args()
    c = Client(a.url, a.key)
    st, stats = c.call("GET", "/v1/stats")             # (a batching worker's usage: its slots)
    slots = find(stats, "slots") if st == 200 else None
    n_slots = len(slots) if isinstance(slots, list) else slots
    if not check(st == 200 and isinstance(n_slots, int) and n_slots >= 2, f"/v1/stats: {n_slots} slots (2 or more)"):
        return 1
    st, _ = c.call("POST", "/flush_cache")
    check(st == 200, "/flush_cache")
    ref = {}
    for name in PROMPTS:                        # each alone: its reference
        t0 = time.perf_counter()
        ref.update(run(c, [name], a.max_tokens, a.model))
        show(f"{name} alone", {name: ref[name]}, time.perf_counter() - t0)
        check(ref[name][1] > 1, f"{name} alone: {ref[name][1]} tokens")
    for names in [tuple(PROMPTS)[:k] for k in range(2, len(PROMPTS) + 1)]:
        t0, tm0 = time.perf_counter(), timing(c)
        got = run(c, names, a.max_tokens, a.model)
        show(" + ".join(names) + " at once", got, time.perf_counter() - t0)
        show_steps(" + ".join(names), tm0, timing(c))
        for x in names:
            check(got[x][0] == ref[x][0] and got[x][1] == ref[x][1],
                  f"{x} beside {'+'.join(y for y in names if y != x)}: its reply alone's "
                  f"({got[x][1]} tokens{'' if got[x][0] == ref[x][0] else ', the text differs'})")
    sref, seeds = {}, [x for x in SEEDS if x in PROMPTS][:max(2, n_slots)]
    for name in seeds:                          # sampled, each alone: its reference for its seed
        t0 = time.perf_counter()
        sref.update(run(c, [name], a.max_tokens, a.model, sampled=True))
        show(f"{name} sampled alone", {name: sref[name]}, time.perf_counter() - t0)
        check(sref[name][1] > 1, f"{name} sampled alone: {sref[name][1]} tokens")
    t0, tm0 = time.perf_counter(), timing(c)
    got = run(c, seeds, a.max_tokens, a.model, sampled=True)
    show(" + ".join(seeds) + " sampled at once", got, time.perf_counter() - t0)
    show_steps(" + ".join(seeds) + " sampled", tm0, timing(c))
    for x in seeds:
        check(got[x][0] == sref[x][0] and got[x][1] == sref[x][1],
              f"{x} sampled beside {'+'.join(y for y in seeds if y != x)}: its reply alone's for seed {SEEDS[x]} "
              f"({got[x][1]} tokens{'' if got[x][0] == sref[x][0] else ', the text differs'})")
    conv = chats(c, a.chats, a.chat_tokens, a.paste_lines, a.model) if a.chats else []
    if a.replies:
        Path(a.replies).write_text(json.dumps({"greedy": {k: v[0] for k, v in ref.items()},
                                               "sampled": {k: v[0] for k, v in sref.items()}, "chats": conv}, indent=1))
    st, stats = c.call("GET", "/v1/stats")             # (the worker's usage: the steps' timing)
    tm = find(stats, "step_timing") if st == 200 else None
    print(f"      step timing: {tm}", flush=True)
    check(isinstance(tm, dict) and tm.get("programs", 0) > 0, "batched steps ran (step_timing programs > 0)")
    print("ALL PASS" if not FAILS else f"{len(FAILS)} FAIL", flush=True)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
