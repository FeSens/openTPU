"""The LM head's dKL alone, on a Hugging Face model's own final states (bf16, CPU; router_trace.py's
loading: weights beyond --max-memory GiB offloaded to disk):

    python tools/head_proxy.py MODEL_DIR OUT.json [--tokens 2000] [--max-memory 8] [--hs HS.npy]

the first --tokens of docs/isa.md (formats_scan.text_ids) through the model once, its final norm's
output saved to HS.npy (OUT.json's name + .hs.npy; read back if present), then formats_scan._head
over it: the float head on the states, the int8 and fp4 heads on the states quantized as the
device's head input (int8, D 128), the config's logit soft cap where it has one (Gemma); rows as
formats_scan's (nll, ppl, kl_float per token), dKL of fp4 against int8 with its paired SE. The
layers stay float: a proxy of a scan's head row for the models formats_scan does not emulate
(Qwen3.5-35B-A3B's experts), low where the head's error grows on the layers' (Gemma 4 E2B: 26%
under its scan's row, Qwen3.5-4B within 1%; docs/formats.md). --max-memory 0: the weights mmapped
(run it in a memory-capped cgroup, as tools/offload/router_trace.py's traces); else accelerate
offloads the rest to disk."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _fs():
    import importlib.util
    s = importlib.util.spec_from_file_location("formats_scan", ROOT / "tools/formats_scan.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def states(model, ids, max_memory, off):
    import torch
    from transformers import AutoModelForCausalLM
    kw = dict(dtype=torch.bfloat16, low_cpu_mem_usage=True)
    if max_memory:
        kw.update(device_map="auto", max_memory={"cpu": f"{max_memory}GiB"}, offload_folder=off)
    t0 = time.time()
    m = AutoModelForCausalLM.from_pretrained(model, **kw).eval()
    print(f"loaded in {time.time() - t0:.0f} s", flush=True)
    t0 = time.time()
    with torch.no_grad():
        hs = m.model(input_ids=torch.tensor([ids]), use_cache=False).last_hidden_state
    print(f"forward {len(ids)} tokens in {time.time() - t0:.0f} s", flush=True)
    return hs[0].float().numpy()


def head_name(model):
    """The checkpoint tensor of the LM head (the embedding where tied) and its file."""
    d = Path(model)
    idx = d / "model.safetensors.index.json"
    keys = json.loads(idx.read_text())["weight_map"] if idx.exists() else None
    if keys is None:
        from safetensors import safe_open
        with safe_open(d / "model.safetensors", "np") as f:
            keys = {k: "model.safetensors" for k in f.keys()}
    for suffix in ("lm_head.weight", "embed_tokens.weight"):
        hit = [k for k in keys if k.endswith(suffix) and "mtp" not in k and "visual" not in k]
        if hit:
            return sorted(hit, key=len)[0], d / keys[sorted(hit, key=len)[0]]
    raise SystemExit("no LM head in the checkpoint")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--tokens", type=int, default=2000)
    ap.add_argument("--max-memory", type=float, default=8.0)
    ap.add_argument("--hs")
    a = ap.parse_args()
    FS = _fs()
    from opentpu.llm.qwen3 import _fake_q
    ids = FS.text_ids(a.model, a.tokens)
    hp = Path(a.hs or f"{a.out}.hs.npy")
    if hp.exists():
        hs = np.load(hp)
    else:
        hs = states(a.model, ids, a.max_memory, f"{a.out}.offload")
        np.save(hp, hs)
    name, f = head_name(a.model)
    print(f"head {name} ({f.name}); states {hs.shape}", flush=True)
    from safetensors import safe_open

    def chunks():
        with safe_open(f, "pt") as t:
            sl = t.get_slice(name)
            V = sl.get_shape()[0]
            for r0 in range(0, V, FS.HEAD_ROWS):
                yield r0, sl[r0:r0 + FS.HEAD_ROWS].float().numpy()
    h = np.asarray(hs, np.float64)
    hq = _fake_q(h, 128)
    c = json.loads((Path(a.model) / "config.json").read_text())
    cap = (c.get("text_config") or c).get("final_logit_softcapping")
    t0 = time.time()
    res = FS._head({"float": h, "int8": hq, "fp4": hq},
                   {"float": None, "int8": "int8", "fp4": "fp4"}, chunks, ids, 128,
                   kl_to=("float", "int8"), cap=cap)
    print(f"head pass {time.time() - t0:.0f} s", flush=True)
    rows = {}
    for lab, (nll, top1, kl) in res.items():
        rows[lab] = {"nll": float(nll.mean()), "ppl": float(np.exp(nll.mean())),
                     "kl_float": float(kl["float"].mean()), "kl_tok": kl["float"].tolist(),
                     "top1": top1.tolist()}
    d = np.asarray(rows["fp4"]["kl_tok"]) - np.asarray(rows["int8"]["kl_tok"])
    dn = np.asarray(res["fp4"][0]) - np.asarray(res["int8"][0])
    out = {"model": a.model, "tokens": len(ids), "head": name, "cap": cap, "dkl": float(d.mean()),
           "se_dkl": float(d.std(ddof=1) / math.sqrt(len(d))), "dppl": float(np.expm1(dn.mean())),
           "se_ppl": float(dn.std(ddof=1) / math.sqrt(len(dn))),
           "top1_fp4_int8": float(np.mean(np.equal(rows["fp4"]["top1"], rows["int8"]["top1"]))),
           "rows": {k: {x: v for x, v in r.items() if x not in ("kl_tok", "top1")}
                    for k, r in rows.items()}}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"{Path(a.model).name}: {len(ids)} tokens; int8 head KL "
          f"{100 * rows['int8']['kl_float']:.3f}%, fp4 {100 * rows['fp4']['kl_float']:.3f}%: dKL "
          f"{100 * out['dkl']:+.3f}% (SE {100 * out['se_dkl']:.3f}), ppl {100 * out['dppl']:+.2f}% "
          f"(SE {100 * out['se_ppl']:.2f}), top-1 fp4 = int8 {out['top1_fp4_int8']:.3f}",
          flush=True)


if __name__ == "__main__":
    main()
