"""Greedy decoding on the ISA simulator vs Hugging Face (fp32), prompt by prompt.

    python3 tools/compare_hf.py [--model qwen3|lfm2|qwen35|DIR] [--tokens 16] [--chat] [--emulate]
                                [prompt ...]

For each prompt: the two continuations, the first token where they differ with Hugging Face's
logit gap there, and the device's logit error against Hugging Face's logits over the steps
with the same context (max |difference| and the lowest cosine of the logit vectors). --chat
wraps each prompt in the model's chat template (a user turn). --emulate also runs the model's
`emulated_logits` (float64 with openTPU's int8 quantization points and none of its rounding)
at the first difference, to tell quantization effects (the emulation agrees with the device)
from kernel bugs (it agrees with Hugging Face).
"""
from __future__ import annotations

import argparse
import importlib
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
    from opentpu.llm import MODELS, load_spec, model_dir
    from opentpu.llm.qwen3 import Engine, load_weights

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("prompts", nargs="*", default=PROMPTS)
    ap.add_argument("--model", default="qwen3",
                    help=f"{' or '.join(MODELS)} (models/<name>), or a checkpoint directory")
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--chat", action="store_true",
                    help="prompts as user turns of the chat template")
    ap.add_argument("--emulate", action="store_true")
    a = ap.parse_args()
    path = model_dir(a.model)
    tok = transformers.AutoTokenizer.from_pretrained(path)
    hf = transformers.AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    spec, W = load_spec(path), load_weights(path)
    emulated_logits = importlib.import_module(type(spec).__module__).emulated_logits

    def encode(p):
        if not a.chat:
            return tok(p).input_ids
        ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                      enable_thinking=False, tokenize=True)
        return list(ids["input_ids"] if hasattr(ids, "keys") else ids)

    prompts = [(p, encode(p)) for p in a.prompts]
    need = max(len(ids) for _, ids in prompts) + a.tokens
    eng = Engine(spec, W, cap=max(256, -(-need // 128) * 128))
    top2 = lambda l: [tok.decode([int(i)]) for i in np.argsort(-l)[:2]]
    same, worst = 0, (0.0, 1.0)
    for p, ids in prompts:
        with torch.no_grad():
            # pure greedy: LFM2's generation_config adds a repetition penalty
            want = hf.generate(torch.tensor([ids]), max_new_tokens=a.tokens, do_sample=False,
                               repetition_penalty=1.0)[0, len(ids):].tolist()
        eng.reset()
        got, dev = [], [eng.prefill(ids)]            # dev[i]: the logits that chose got[i]
        while len(got) < a.tokens:
            got.append(int(np.argmax(dev[-1])))
            if got[-1] in spec.eos or len(got) == a.tokens:
                break
            dev.append(eng.step(got[-1]))
        k = next((i for i, (x, y) in enumerate(zip(got, want)) if x != y), None)
        n = min(len(got), len(want)) if k is None else k + 1   # steps with the same context
        with torch.no_grad():
            ref = hf(torch.tensor([ids + want[:n - 1]])).logits[0, len(ids) - 1:].numpy()
        d = np.array(dev[:n])
        err = float(np.abs(d - ref).max())
        cos = float(((d * ref).sum(-1) / np.linalg.norm(d, axis=-1)
                     / np.linalg.norm(ref, axis=-1)).min())
        worst = (max(worst[0], err), min(worst[1], cos))
        print(f"{p!r}")
        print(f"   HF    : {tok.decode(want)!r}\n   device: {tok.decode(got)!r}")
        print(f"   logits vs HF over {n} steps: max |diff| {err:.3f} (HF's logits span up to "
              f"{np.ptp(ref, axis=-1).max():.1f}), min cosine {cos:.5f}")
        if k is None:
            same += 1
            print(f"   same for all {len(got)} tokens")
            continue
        lg = ref[k]
        gap = lg.max() - lg[got[k]]
        rank = int((lg > lg[got[k]]).sum()) + 1
        print(f"   first difference at token {k + 1}: HF {tok.decode([want[k]])!r}, device "
              f"{tok.decode([got[k]])!r} (HF's choice #{rank}, {gap:.3f} logits below the top)")
        if a.emulate:
            emu = emulated_logits(spec, W, ids + want[:k])[-1]
            print(f"   int8 emulation's top two there: {top2(emu)}")
    print(f"{same}/{len(prompts)} prompts identical for {a.tokens} tokens; logits vs HF: "
          f"max |diff| {worst[0]:.3f}, min cosine {worst[1]:.5f}")


if __name__ == "__main__":
    main()
