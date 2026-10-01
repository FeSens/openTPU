"""Gemma 4's numpy reference (opentpu.llm.gemma4.reference_logits) against Hugging Face
transformers on a truncation of a real checkpoint: the chosen checkpoint layers only, both in
fp32, the logits of a prompt before the soft cap. Hugging Face runs in a child process first,
so the peak memory is one fp32 copy of the truncated model (26B-A4B: 3.1 GB per layer of
experts plus the 2.95 GB embedding).

  python tools/gemma4_hf_check.py models/gemma-4-26B-A4B --layers 5
  python tools/gemma4_hf_check.py models/gemma-4-26B-A4B --layers 0,5 --tokens 16
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROMPT = "The lighthouse keeper climbed the stairs at dusk, and"


def _hf_logits(model_dir: str, layers: list, toks: list, out: str) -> None:
    """The truncated model's logits [T, vocab] by transformers, into out (.npy): without the
    soft cap (a truncated model's logits saturate it: ties)."""
    import torch
    import transformers
    from safetensors import safe_open
    from transformers.models.gemma4 import modeling_gemma4 as M
    try:                                    # transformers >= 5.17
        from transformers.initialization import no_init_weights
    except ImportError:
        from transformers.modeling_utils import no_init_weights
    top = json.loads((Path(model_dir) / "config.json").read_text())
    c = dict(top.get("text_config", top))
    if c.get("num_kv_shared_layers"):
        raise SystemExit("checkpoints with KV-shared layers: truncate with Spec.truncated rules")
    c.update(num_hidden_layers=len(layers), layer_types=[c["layer_types"][i] for i in layers],
             final_logit_softcapping=None)
    hc = transformers.Gemma4TextConfig(**{k: v for k, v in c.items() if k != "model_type"})
    hc._attn_implementation = "eager"
    with no_init_weights():
        m = M.Gemma4ForCausalLM(hc)
    m = m.float().eval()
    files = {}
    for p in sorted(Path(model_dir).glob("*.safetensors")):
        f = safe_open(str(p), "pt")
        files.update({k: f for k in f.keys()})

    def ck(name):
        """The checkpoint name of a parameter of the truncated model."""
        parts = name.split(".")
        if name.startswith("model.layers."):
            parts[2] = str(layers[int(parts[2])])
        return "model.language_model." + ".".join(parts[1:])

    with torch.no_grad():
        for n, t in list(m.named_parameters()) + list(m.named_buffers()):
            if n == "lm_head.weight" or "rotary" in n or n.endswith("embed_scale"):
                continue
            t.copy_(files[ck(n)].get_tensor(ck(n)).float())
        m.tie_weights()
        logits = m(torch.tensor([toks])).logits[0].float().numpy()
    np.save(out, logits)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model_dir")
    ap.add_argument("--layers", default="5", help="checkpoint layers, comma-separated")
    ap.add_argument("--tokens", type=int, default=24, help="prompt tokens (at most)")
    a = ap.parse_args(argv)
    from tokenizers import Tokenizer
    from opentpu.llm import gemma4 as G
    layers = [int(x) for x in a.layers.split(",")]
    tok = Tokenizer.from_file(str(Path(a.model_dir) / "tokenizer.json"))
    spec = G.Spec.from_hf(a.model_dir).truncated(layers)
    toks = ([spec.bos] + tok.encode(PROMPT, add_special_tokens=False).ids)[:a.tokens]
    with tempfile.TemporaryDirectory() as d:
        out = str(Path(d) / "hf.npy")
        p = mp.get_context("spawn").Process(target=_hf_logits,
                                            args=(a.model_dir, layers, toks, out))
        p.start()
        p.join()
        if p.exitcode:
            return p.exitcode
        hf = np.load(out)
    ref = G.reference_logits(spec, G.load_weights(a.model_dir), toks)
    err = np.abs(ref - hf).max(axis=1)
    cos = (ref * hf).sum(1) / np.linalg.norm(ref, axis=1) / np.linalg.norm(hf, axis=1)
    same = ref.argmax(1) == hf.argmax(1)
    top2 = np.sort(hf, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    clear = margin > 2 * err
    print(f"layers {layers} tokens {len(toks)}: max |ref - hf| {err.max():.3g} "
          f"(per token {np.round(err, 5).tolist()}), cosine min {cos.min():.7f}, "
          f"logit scale {np.abs(hf).max():.3g}; argmax agree {same.mean():.3f}, "
          f"{same[clear].sum()} of the {clear.sum()} tokens whose top-2 margin exceeds "
          f"2 |ref - hf| (margins {np.round(margin, 5).tolist()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
