#!/usr/bin/env python3
"""MTP phase 0 (docs/mtp.md section 8): how often a k = 1 draft is accepted, offline, on the
Hugging Face model (no card, no simulator).

Per prompt (chat, code, summarization of a given text) the model generates up to --tokens tokens,
greedy or sampled with its defaults. One forward pass over the whole sequence then gives each
position's logits (and Qwen3.5's final hidden). For every generated token after the first, the
drafters propose it from what comes before it:

- n-gram, prompt lookup with g = 2 and 3 (docs/mtp.md 6.2): the token after the latest earlier
  match of the last g tokens of the history (the prompt and the tokens so far), if there is one;
- Qwen3.5's MTP head (6.1; --mtp), teacher-forced: the MTP layer over (embedding of token t + 1,
  the model's hidden at t after model.norm) at position t drafts token t + 2. Its LM head is
  the full float head, its fp4 copy, or the fp4 rows of the N lowest ids (a BPE vocabulary
  ranks its merges by frequency: a draft head over the most frequent tokens), with the
  activations quantized to int8 as the card's QACT.

Greedy: the draft is accepted when it equals the model's token. Sampled: speculative sampling
with a one-hot draft accepts with probability p(d) under the processed distribution
(temperature, top-k, top-p, the repetition penalty; docs/mtp.md 5.2), and the study sums p(d).

Writes a JSON with, per prompt, the ids and per drafter one value per drafted token (None:
no draft); --summary prints acceptance and the projected k = 1 speedup from such files (and
"ngram3>2": g = 3 where it matches, else g = 2). A checkpoint without a chat template (a
pre-trained one) gets a plain "User: ... Assistant:" dialogue. docs/mtp.md 7.1 has the results.

    tools/mtp_accept.py --model Qwen3.5-0.8B --mtp --mode greedy --out q35-greedy.json
    tools/mtp_accept.py --summary q35-greedy.json [--c2 1.21]"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]

DRAM_TEXT = (
    "Dynamic memory keeps each bit as charge in a tiny capacitor, and the charge leaks away. "
    "To keep the data, the memory controller refreshes every row before its charge falls too "
    "far: it issues a refresh command about every 7.8 microseconds, and the chip refreshes a "
    "group of rows internally. While a refresh runs, the banks it touches cannot serve reads "
    "or writes, so a refresh costs bandwidth and adds latency to the requests that arrive "
    "during it. Larger chips have more rows to refresh, so each refresh command takes longer. "
    "Hot chips leak faster: above 85 degrees Celsius the refresh interval halves. Controllers "
    "hide part of the cost by refreshing banks that are idle, by postponing a few refreshes "
    "while traffic is heavy and catching up later, and by scheduling reads to other banks "
    "while one bank refreshes. Some designs refresh only the rows that hold data, or measure "
    "how long each row really keeps its charge and refresh the strong rows less often. These "
    "techniques save power as well as time, because every refresh moves charge in thousands of "
    "cells at once.")

COUNCIL_TEXT = (
    "On Tuesday evening the city council voted six to three to build protected bike lanes on "
    "Harbor Street, the busiest road between the train station and the old town. The plan "
    "removes one lane of car traffic and forty parking spaces, and adds a planted barrier "
    "between cyclists and cars. Supporters said that two cyclists were badly injured on the "
    "street last year, and that a survey found many residents would cycle to work if they felt "
    "safe. Several shop owners objected, worried that customers who drive would go elsewhere "
    "once the parking disappears. The council agreed to add short-term loading zones for "
    "deliveries and to review sales figures with the shop owners after one year. Construction "
    "starts in the spring and should take about four months; during that time buses will be "
    "rerouted along Mill Road. The project costs 2.4 million, of which a regional transport "
    "grant pays half. The mayor called the vote a first step and said the council would study "
    "similar lanes on two other streets next year.")

PROMPTS = [
    ("chat", "What are three good habits for staying focused while working from home? Explain "
             "each one briefly."),
    ("chat", "A friend is visiting my city for a weekend. Suggest a relaxed plan for Saturday, "
             "from breakfast to dinner."),
    ("chat", "Explain how a rainbow forms, for a curious ten-year-old."),
    ("code", "Write a Python function that returns the n-th Fibonacci number iteratively, with a "
             "docstring and a short test."),
    ("code", "Write a Python class for a fixed-size ring buffer with push, pop and __len__, and "
             "show a short usage example."),
    ("code", "Here is a function:\n\n```python\ndef mean(xs):\n    total = 0\n    for x in xs:\n"
             "        total += x\n    return total / len(xs)\n```\n\nRewrite it so that it "
             "returns None for an empty list, add type hints, and keep the loop."),
    ("summary", "Summarize the following text in five sentences.\n\n" +
                (ROOT / "tools" / "data" / "austen_pp_ch1.txt").read_text().strip()),
    ("summary", "Summarize the following text in four sentences.\n\n" + DRAM_TEXT),
    ("summary", "Summarize the following text in four sentences.\n\n" + COUNCIL_TEXT),
]

HEAD_IDS = (16384, 32768, 65536)        # the draft heads over the N lowest ids


def sampling_of(path: Path) -> dict:
    """The model's sampled-decoding defaults: otpu-chat's (opentpu/host/chat.py SAMPLING) for
    its families, else the checkpoint's generation_config.json."""
    from opentpu.host.chat import SAMPLING
    t = json.loads((path / "config.json").read_text()).get("model_type", "")
    fam = {"qwen3": "qwen3", "lfm2": "lfm2", "qwen3_5": "qwen35", "qwen3_5_text": "qwen35"}
    if t in fam:
        return dict(SAMPLING[fam[t]])
    g = json.loads((path / "generation_config.json").read_text())
    return dict(temperature=g.get("temperature", 1.0), top_k=g.get("top_k", 0) or 0,
                top_p=g.get("top_p", 1.0), repetition_penalty=g.get("repetition_penalty", 1.0))


def processed(logits: np.ndarray, context, s: dict) -> np.ndarray:
    """The target's sampling distribution: the repetition penalty (Hugging Face's) over the
    context's ids, temperature, top-k, then top-p as otpu-chat's sampler keeps it."""
    lg = logits.astype(np.float64)
    if s["repetition_penalty"] != 1.0 and len(context):
        ix = np.unique(np.asarray(context))
        w = lg[ix]
        lg[ix] = np.where(w > 0, w / s["repetition_penalty"], w * s["repetition_penalty"])
    lg = lg / s["temperature"]
    k = s["top_k"] if 0 < s["top_k"] < len(lg) else len(lg)
    idx = np.argpartition(-lg, k - 1)[:k]
    idx = idx[np.argsort(-lg[idx], kind="stable")]
    p = np.exp(lg[idx] - lg[idx[0]])
    p /= p.sum()
    keep = min(len(p), int(np.searchsorted(np.cumsum(p), s["top_p"])) + 1)
    out = np.zeros(len(lg))
    out[idx[:keep]] = p[:keep] / p[:keep].sum()
    return out


def ngram_draft(ctx, g: int):
    """The token after the latest earlier occurrence of ctx's last g tokens, or None."""
    n = len(ctx)
    if n <= g:
        return None
    tail = ctx[n - g:]
    for j in range(n - g - 1, -1, -1):
        if ctx[j:j + g] == tail:
            return ctx[j + g]
    return None


class MTPHead:
    """Qwen3.5's MTP layer (mtp.* of the checkpoint) on the model's modules: the two input norms,
    fc over [embedding, hidden], a full-attention decoder layer with its own (causal) cache,
    mtp.norm; position t for the input of hidden t (mlx_vlm's drafter)."""

    def __init__(self, model, path: Path):
        import torch
        from safetensors import safe_open
        from transformers.models.qwen3_5 import modeling_qwen3_5 as M
        cfg = model.config
        H, eps = cfg.hidden_size, cfg.rms_norm_eps
        li = list(cfg.layer_types).index("full_attention")
        self.pre_e, self.pre_h = M.Qwen3_5RMSNorm(H, eps), M.Qwen3_5RMSNorm(H, eps)
        self.fc = torch.nn.Linear(2 * H, H, bias=False)
        self.layer = M.Qwen3_5DecoderLayer(cfg, li)
        self.norm = M.Qwen3_5RMSNorm(H, eps)
        mods = {"pre_fc_norm_embedding": self.pre_e, "pre_fc_norm_hidden": self.pre_h,
                "fc": self.fc, "layers.0": self.layer, "norm": self.norm}
        n = 0
        for f in sorted(path.glob("*.safetensors")):
            with safe_open(str(f), "pt") as st:
                for k in st.keys():
                    if not k.startswith("mtp."):
                        continue
                    name = k[4:]
                    pre = next(p for p in mods if name.startswith(p + "."))
                    mods[pre].get_parameter(name[len(pre) + 1:]).data.copy_(
                        st.get_tensor(k).float())
                    n += 1
        if n != sum(len(list(m.parameters())) for m in mods.values()):
            raise ValueError(f"{n} mtp tensors, the modules have "
                             f"{sum(len(list(m.parameters())) for m in mods.values())}")
        for m in mods.values():
            m.float().eval()
        self.model = model

    def __call__(self, ids, hidden):
        """ids [L], hidden [L, H] (the model's, after model.norm) -> the MTP's normed outputs
        [L - 1, H]: row t drafts token t + 2."""
        import torch
        m = self.model.model
        with torch.no_grad():
            e = m.embed_tokens(torch.tensor(ids[1:])[None]).float()
            h = torch.as_tensor(hidden[:-1])[None].float()
            x = self.fc(torch.cat([self.pre_e(e), self.pre_h(h)], -1))
            pos = torch.arange(x.shape[1])[None]
            pe = m.rotary_emb(x, pos[None].expand(3, 1, -1))
            x = self.layer(x, position_embeddings=pe, attention_mask=None, position_ids=pos)
            return self.norm(x)[0].numpy()


def heads(model) -> dict:
    """The draft heads: name -> (rows' ids or None, weight [N, H] float32, quantize x)."""
    from opentpu import quant as Q
    E = model.get_input_embeddings().weight.detach().float().numpy()
    t = time.time()
    _, _, deq = Q.quantize_w4(E, "fp4")
    print(f"fp4 head quantized in {time.time() - t:.0f} s", flush=True)
    deq = deq.astype(np.float32, copy=False)
    out = {"full": (None, E, False), "fp4": (None, deq, True)}
    for n in HEAD_IDS:
        out[f"fp4-{n // 1024}K"] = (np.arange(n), deq[:n], True)
    return out


def act_int8(x: np.ndarray) -> np.ndarray:
    """x as the card's QACT gives it to the MXU (int8 per row and 128-block), dequantized."""
    from opentpu.runtime import quantize_rows
    q, s = quantize_rows(np.ascontiguousarray(x, np.float32), 128)
    N, K = x.shape
    return (q.reshape(N, K // 128, 128).astype(np.float32) * s[..., None]).reshape(N, K)


def run(a) -> None:
    import torch
    import transformers
    path = ROOT / "models" / a.model if not Path(a.model).is_dir() else Path(a.model)
    tok = transformers.AutoTokenizer.from_pretrained(path)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[a.dtype]
    model = transformers.AutoModelForCausalLM.from_pretrained(path, dtype=dtype).eval()
    torch.set_num_threads(a.threads)
    samp = sampling_of(path)
    mtp = MTPHead(model, path) if a.mtp else None
    hd = heads(model) if a.mtp else {}
    cap = {}
    if a.mtp:
        model.model.norm.register_forward_hook(lambda mod, i, o: cap.__setitem__("h", o))
    res = {"model": path.name, "mode": a.mode, "sampling": samp if a.mode == "sampled" else None,
           "dtype": a.dtype, "tokens": a.tokens, "chat_template": tok.chat_template is not None,
           "prompts": []}
    for i, (kind, text) in enumerate(PROMPTS):
        if a.only is not None and i not in a.only:
            continue
        if tok.chat_template is None:       # a pre-trained checkpoint: a plain dialogue
            ids = tok(f"User: {text}\n\nAssistant:").input_ids
        else:
            ids = tok.apply_chat_template([{"role": "user", "content": text}],
                                          add_generation_prompt=True, enable_thinking=False,
                                          tokenize=True)
        ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
        t0 = time.time()
        torch.manual_seed(1000 + i)
        kw = (dict(do_sample=True, **samp) if a.mode == "sampled" else
              dict(do_sample=False, repetition_penalty=1.0))
        with torch.no_grad():
            seq = model.generate(torch.tensor([ids]), max_new_tokens=a.tokens, **kw)[0].tolist()
            logits = model(torch.tensor([seq])).logits[0].float().numpy()
        P, L = len(ids), len(seq)
        rec = {"kind": kind, "prompt_tokens": P, "generated": L - P, "ids": seq,
               "seconds": round(time.time() - t0, 1)}
        targets = range(P + 1, L)          # each iteration drafts the token after the last one

        probs = {}

        def score(i, d):
            if d is None:
                return None
            if a.mode == "greedy":
                return int(d == seq[i])
            if i not in probs:      # the distribution token i was sampled from
                probs[i] = processed(logits[i - 1], seq[:i] if samp["repetition_penalty"] != 1.0
                                     else (), samp)
            return float(probs[i][d])
        for g in (2, 3):
            rec[f"ngram{g}"] = [score(i, ngram_draft(seq[:i], g)) for i in targets]
        if a.mtp:
            x = mtp(seq, cap["h"][0].float().numpy())        # row t drafts t + 2
            xq = act_int8(x)
            for name, (ix, Wh, q) in hd.items():
                lg = (xq if q else x) @ Wh.T
                d = np.argmax(lg, axis=1)
                d = d if ix is None else ix[d]
                rec[f"mtp-{name}"] = [score(i, int(d[i - 2])) for i in targets]
            rec["in_ids"] = {f"{n // 1024}K": float(np.mean([seq[i] < n for i in targets]))
                             for n in HEAD_IDS}
        res["prompts"].append(rec)
        line = " ".join(f"{k} {np.mean([v for v in rec[k] if v is not None] or [0]):.2f}"
                        for k in rec if k.startswith(("ngram", "mtp")))
        print(f"{path.name} {a.mode} #{i} {kind}: {P}+{L - P} tokens, {rec['seconds']} s; {line}",
              flush=True)
    Path(a.out).write_text(json.dumps(res))


def loop_cost(vals, c: float) -> float:
    """The k = 1 loop's expected cost, in decode steps, over one sequence: vals[j] for the draft
    of its j-th drafted token (None: no draft, the plain step at cost 1, one token; else the
    probability q that the draft is accepted: the iteration costs c and emits 2 tokens with q,
    1 with 1 - q). Every token is emitted once, so the speedup is len(vals) / cost. Greedy q is 0
    or 1, so this walks the loop exactly; sampled it is the expectation over the acceptances."""
    n = len(vals)
    C = [0.0] * (n + 2)
    for j in range(n - 1, -1, -1):
        q = vals[j]
        if q is None:
            C[j] = 1 + C[j + 1]
        elif j == n - 1:
            C[j] = c
        else:
            C[j] = c + q * C[j + 2] + (1 - q) * C[j + 1]
    return C[0]


def summary(files, c2: float, cdraft: dict) -> None:
    """Acceptance per model, mode, drafter and prompt kind (drafted: the share of tokens with a
    draft; accept: the mean over those), and the projected k = 1 speedup: the tokens over the
    loop's expected cost (loop_cost), summed over the prompts."""
    rows = []
    for f in files:
        r = json.loads(Path(f).read_text())
        for p in r["prompts"]:      # g = 3 where it matches, else g = 2 (its matches include 3's)
            p["ngram3>2"] = [v3 if v3 is not None else v2
                             for v3, v2 in zip(p["ngram3"], p["ngram2"])]
        keys = [k for k in r["prompts"][0] if k.startswith(("ngram", "mtp"))]
        for k in keys:
            c = c2 + cdraft.get(k.split("-", 1)[1] if k.startswith("mtp") else "ngram", 0.0)
            for kind in ("chat", "code", "summary", "all"):
                ps = [p for p in r["prompts"] if kind in ("all", p["kind"])]
                vals = [v for p in ps for v in p[k]]
                got = [v for v in vals if v is not None]
                h = len(got) / max(1, len(vals))
                acc = float(np.mean(got)) if got else 0.0
                sp = len(vals) / sum(loop_cost(p[k], c) for p in ps)
                rows.append((r["model"], r["mode"], k, kind, len(vals), h, acc, c, sp))
    print(f"{'model':16s} {'mode':7s} {'drafter':12s} {'prompts':8s} {'steps':>6s} "
          f"{'drafted':>7s} {'accept':>6s} {'cost':>5s} {'speedup':>7s}")
    for m, mode, k, kind, n, h, acc, c, sp in rows:
        print(f"{m:16s} {mode:7s} {k:12s} {kind:8s} {n:6d} {h:7.2f} {acc:6.2f} {c:5.2f} "
              f"{sp:7.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", help="checkpoint directory, or its name under models/")
    ap.add_argument("--mode", choices=["greedy", "sampled"], default="greedy")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--mtp", action="store_true", help="Qwen3.5: the MTP head's drafts too")
    ap.add_argument("--only", type=lambda s: [int(x) for x in s.split(",")],
                    help="these prompts only (indices, e.g. 0,6)")
    ap.add_argument("--out", help="the JSON to write")
    ap.add_argument("--summary", nargs="+", help="print the summary of these JSON files")
    ap.add_argument("--c2", type=float, default=1.21, help="a 2-row verify run, in decode steps")
    a = ap.parse_args()
    if a.summary:
        # the drafter's own cost per iteration, in decode steps (docs/mtp.md 6.1, Qwen3.5-0.8B):
        # the MTP layer (about 2%) plus its head; n-gram's lookup is free
        summary(a.summary, a.c2, {"full": 0.46, "fp4": 0.25, "fp4-16K": 0.035,
                                  "fp4-32K": 0.05, "fp4-64K": 0.08, "ngram": 0.0})
        return
    run(a)


if __name__ == "__main__":
    main()
