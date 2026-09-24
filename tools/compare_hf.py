"""Greedy decoding of Qwen3-0.6B on the ISA simulator vs Hugging Face (fp32), prompt by prompt.

    python3 tools/compare_hf.py [--tokens 16] [--emulate] [prompt ...]

For each prompt: the two continuations and the first token where they differ, with Hugging
Face's logit gap there. --emulate also runs `emulated_logits` (float64 with openTPU's int8
quantization points and none of its rounding) at that point, to tell quantization effects
(the emulation agrees with the device) from kernel bugs (it agrees with Hugging Face).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PROMPTS = ["A prime number larger than 100 is", "The capital of France is", "def fibonacci(n):",
           "Water boils at", "The quick brown fox",
           "In 1969, the first person to walk on the moon was",
           "The largest planet in the solar system is", "import numpy as np\n"]


def main() -> None:
    import torch
    import transformers
    from opentpu.llm.qwen3 import Engine, Spec, emulated_logits, load_weights

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("prompts", nargs="*", default=PROMPTS)
    ap.add_argument("--model", default=str(ROOT / "models" / "Qwen3-0.6B"))
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--emulate", action="store_true")
    a = ap.parse_args()
    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    hf = transformers.AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    spec, W = Spec.from_hf(a.model), load_weights(a.model)
    eng = Engine(spec, W, cap=256)
    top2 = lambda l: [tok.decode([int(i)]) for i in np.argsort(-l)[:2]]
    same = 0
    for p in a.prompts:
        ids = tok(p).input_ids
        with torch.no_grad():
            want = hf.generate(torch.tensor([ids]), max_new_tokens=a.tokens,
                               do_sample=False)[0, len(ids):].tolist()
        eng.reset()
        got = eng.generate(ids, max_new=a.tokens)
        k = next((i for i, (x, y) in enumerate(zip(got, want)) if x != y), None)
        print(f"{p!r}")
        print(f"   HF    : {tok.decode(want)!r}\n   device: {tok.decode(got)!r}")
        if k is None:
            same += 1
            print(f"   same for all {a.tokens} tokens")
            continue
        pre = ids + want[:k]
        with torch.no_grad():
            lg = hf(torch.tensor([pre])).logits[0, -1].numpy()
        gap = lg.max() - lg[got[k]]
        rank = int((lg > lg[got[k]]).sum()) + 1
        print(f"   first difference at token {k + 1}: HF {tok.decode([want[k]])!r}, device "
              f"{tok.decode([got[k]])!r} (HF's choice #{rank}, {gap:.2f} logits below the top)")
        if a.emulate:
            emu = emulated_logits(spec, W, pre)[-1]
            print(f"   int8 emulation's top two there: {top2(emu)}")
    print(f"{same}/{len(a.prompts)} prompts identical for {a.tokens} tokens")


if __name__ == "__main__":
    main()
