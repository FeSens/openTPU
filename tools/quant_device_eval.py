"""4-bit weights through the device code path: the ISA simulator runs the model's decode
programs (the ones the card runs, bit for bit), and this compares a weight format with int8.

    python3 tools/quant_device_eval.py --model qwen3 --config fp4 --json build/quant/dev-fp4.json
    python3 tools/quant_device_eval.py --report build/quant/dev-*.json

A config is the Engine's wformat / head_format ("int8", "fp4", "fp4+head8", "int4", ...);
OTPU_PAIR=1 (or --pair) compiles 4-bit MMs with column reuse (MM PAIR / QACT DUP), as a PAIR
bitstream runs them. Each run feeds, token by token with the decode kernel (Engine.step):

  texts    the first --text-tokens tokens of tools/data/austen_pp_ch1.txt and of docs/isa.md
           (quant_eval.py's texts; it takes 640), teacher forced: perplexity, and the logits;
  greedy   quant_eval.py's 8 prompts, each followed by the int8 run's greedy continuation of
           --gen tokens (teacher forced, so every config sees the same tokens).

The int8 config also records its greedy continuations, which the other configs are fed. The
report compares every config with int8 over the same positions: perplexity per text, KL(int8 ||
config) of the next-token distributions, and top-1 agreement (argmax equal to int8's) on the texts
and on int8's greedy continuations.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PROMPTS = ["A prime number larger than 100 is", "The capital of France is", "def fibonacci(n):",
           "Water boils at", "The quick brown fox",
           "In 1969, the first person to walk on the moon was",
           "The largest planet in the solar system is", "import numpy as np\n"]


def texts(tok, n: int) -> dict:
    book = (ROOT / "tools/data/austen_pp_ch1.txt").read_text()
    isa = re.sub(r"`|\|", "", (ROOT / "docs/isa.md").read_text())    # as quant_eval.py
    return {name: tok(t)["input_ids"][:n] for name, t in (("book", book), ("isa", isa))}


def logsoftmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


def feed(eng, ids) -> np.ndarray:
    """Logits after each token of `ids`, from position 0 (the engine is reset)."""
    eng.reset()
    return np.stack([eng.step(int(t)) for t in ids])


def run(a) -> dict:
    from transformers import AutoTokenizer

    from opentpu.isasim import board_config
    from opentpu.llm import load_spec, model_dir
    from opentpu.llm.qwen3 import Engine, load_weights
    path = model_dir(a.model)
    tok = AutoTokenizer.from_pretrained(path)
    spec = load_spec(path)
    wf, _, head = a.config.partition("+")
    head_format = {"head8": "int8", "": None}.get(head, head)
    kw = {} if a.pair is None else {"PAIR": a.pair}
    cfg = board_config(DRAM_BYTES=1 << 32, **kw)
    t0 = time.time()
    eng = Engine(spec, load_weights(path), cap=a.cap, cfg=cfg, wformat=wf,
                 head_format=head_format)
    out = {"model": path.name, "config": a.config, "pair": bool(getattr(cfg, "PAIR", False)),
           "text_tokens": a.text_tokens, "gen": a.gen, "texts": {}, "greedy": []}
    for name, ids in texts(tok, a.text_tokens).items():
        lg = feed(eng, ids)
        lp = logsoftmax(lg)
        nll = -lp[np.arange(len(ids) - 1), ids[1:]]
        out["texts"][name] = {"ids": list(map(int, ids)), "ppl": float(np.exp(nll.mean())),
                              "argmax": lg.argmax(-1).tolist()}
        np.save(Path(a.json).with_suffix(f".{name}.npy"), lg.astype(np.float32))
        print(f"{a.config} {name}: ppl {out['texts'][name]['ppl']:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    ref = json.loads(Path(a.reference).read_text()) if a.reference else None
    for i, prompt in enumerate(PROMPTS):
        ids = tok(prompt)["input_ids"]
        if ref is None:                    # the int8 run: greedy continuation
            eng.reset()
            lg = [eng.step(t) for t in ids]
            cont = []
            for _ in range(a.gen):
                cont.append(int(np.argmax(lg[-1])))
                lg.append(eng.step(cont[-1]))
            lg = np.stack(lg)
        else:                              # the reference's continuation, teacher forced
            cont = ref["greedy"][i]["cont"]
            lg = feed(eng, ids + cont[:-1])
        out["greedy"].append({"prompt": prompt, "n_prompt": len(ids), "cont": cont,
                              "argmax": lg.argmax(-1).tolist()})
        np.save(Path(a.json).with_suffix(f".greedy{i}.npy"),
                lg[len(ids) - 1:len(ids) - 1 + len(cont)].astype(np.float32))
    out["seconds"] = time.time() - t0
    Path(a.json).write_text(json.dumps(out))
    print(f"{a.config}: done in {out['seconds']:.0f}s", flush=True)
    return out


def report(files) -> str:
    runs = {json.loads(Path(f).read_text())["config"]: Path(f) for f in files}
    ref = runs.get("int8")
    if ref is None:
        raise SystemExit("--report needs the int8 run")
    R = json.loads(ref.read_text())

    def logits(p: Path, key: str) -> np.ndarray:
        return np.load(p.with_suffix(f".{key}.npy"))

    rows = []
    for name, p in runs.items():
        d = json.loads(p.read_text())
        kl, agree, n = 0.0, {"text": [0, 0], "greedy": [0, 0]}, 0
        for key in ("book", "isa"):
            a, b = logsoftmax(logits(ref, key)), logsoftmax(logits(p, key))
            kl += float((np.exp(a) * (a - b)).sum(-1).sum())
            n += len(a)
            agree["text"][0] += int((a.argmax(-1) == b.argmax(-1)).sum())
            agree["text"][1] += len(a)
        for i in range(len(PROMPTS)):
            a, b = logsoftmax(logits(ref, f"greedy{i}")), logsoftmax(logits(p, f"greedy{i}"))
            kl += float((np.exp(a) * (a - b)).sum(-1).sum())
            n += len(a)
            agree["greedy"][0] += int((a.argmax(-1) == b.argmax(-1)).sum())
            agree["greedy"][1] += len(a)
        rows.append((name, d["pair"], d["texts"]["book"]["ppl"], d["texts"]["isa"]["ppl"],
                     kl / n, agree["text"][0] / agree["text"][1],
                     agree["greedy"][0] / agree["greedy"][1]))
    head = (f"{R['model']}, the decode programs on the ISA simulator; texts of "
            f"{R['text_tokens']} tokens, greedy: 8 prompts x {R['gen']} tokens of int8's "
            "continuation\n\n| config | PAIR | ppl book | ppl isa.md | KL vs int8 | top-1 = int8, "
            "texts | top-1 = int8, greedy |\n|---|---|---:|---:|---:|---:|---:|")
    body = [f"| {n} | {'yes' if pr else 'no'} | {pb:.2f} | {pi:.2f} | {k:.3f} | {t:.3f} | "
            f"{g:.3f} |" for n, pr, pb, pi, k, t, g in sorted(rows, key=lambda r: r[0] != "int8")]
    return "\n".join([head] + body)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="qwen3")
    ap.add_argument("--config", default="fp4",
                    help="wformat[+head format]: int8, fp4, fp4+head8, int4, ...")
    ap.add_argument("--pair", type=int, choices=[0, 1], default=None,
                    help="PAIR column reuse (default: OTPU_PAIR)")
    ap.add_argument("--text-tokens", type=int, default=320)
    ap.add_argument("--gen", type=int, default=40)
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--reference", help="the int8 run's JSON (its greedy continuations)")
    ap.add_argument("--json")
    ap.add_argument("--report", nargs="+")
    a = ap.parse_args(argv)
    if a.report:
        print(report(a.report))
        return
    if not a.json:
        raise SystemExit("--json is required")
    if a.config != "int8" and not a.reference:
        raise SystemExit("--reference (the int8 run) is required for a non-int8 config")
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    run(a)


if __name__ == "__main__":
    main()
