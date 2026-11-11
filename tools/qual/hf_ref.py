"""Hugging Face's greedy tokens for a checkpoint bigger than host RAM, in bf16: tools/qual/refs.py's
ISA references against Hugging Face (docs/bcu1525_wg.md).

    python3 tools/qual/hf_ref.py MODEL [--tokens N] [--long | --xl] [--ref REF.pkl]
                                 [--dtype bf16|fp32] [--layers N] [--json OUT]

The checkpoint's text model is built on the meta device and each decoder layer's weights are
read from the safetensors files just before the layer runs and dropped after it (Stream): a pass
holds the embedding table, the head and one layer (Qwen3.8-27B's 0.8 GB in bf16, Qwen3.6-35B-A3B's
1.7 GB) where from_pretrained holds the whole model (52 / 67 GB). The model's own forward runs
(hooks on its layers), so the logits are from_pretrained's bit for bit (tests/test_hf_ref.py).

The prompt is refs.py's (prompt_ids: PROMPT, or --long / --xl, in the chat template with its
date pinned). There is no KV cache: a pass reads the whole checkpoint. Without --ref, N greedy
tokens take N passes over the growing sequence. With --ref (a refs.py reference: the ISA
simulator's tokens) one pass over the prompt and the reference's tokens but its last gives
Hugging Face's choice at each step in the reference's context: where every choice is the
reference's token, Hugging Face's greedy continuation is the reference's (each step's context the
same); else the first that differs, with the reference token's rank and its gap below Hugging
Face's top there. Each step's top two and their gap are printed; --json writes them. --layers N:
the first N layers only (and the final norm and head), a quick check of the path.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "qual"))

TEXT = ("model.language_model.", "model.")      # a multimodal checkpoint's text model's names
FUSED = {"mlp.experts.gate_up_proj": ("gate_proj", "up_proj"),     # a MoE's fused experts
         "mlp.experts.down_proj": ("down_proj",)}


def mem() -> dict:
    """The process's resident memory in GB: VmRSS, its peak VmHWM, and the anonymous part."""
    d = {}
    for line in Path("/proc/self/status").read_text().splitlines() if \
            Path("/proc/self/status").exists() else ():
        k, _, v = line.partition(":")
        if k in ("VmRSS", "VmHWM", "RssAnon"):
            d[k] = round(int(v.split()[0]) / 2**20, 2)
    return d


class Checkpoint:
    """A checkpoint's tensors under the text model's names (model.language_model.* as model.*),
    each file opened for one read (its map goes with it: a pass reads tens of GB). A MoE's
    experts are the model's fused parameters (E, 2F, H) and (E, H, F), from the checkpoint's
    fused tensors or stacked from its experts' own (gate and up rows in that order, as
    transformers' conversion mapping), or GPT-OSS's dequantized from MXFP4 (NAME_blocks and
    NAME_scales) in `dtype` as transformers' (mxfp4.convert_moe_packed_tensors: from_pretrained's
    without a GPU; every value exact in bf16)."""

    def __init__(self, path: Path, dtype=None):
        from safetensors import safe_open
        self.dtype = dtype
        self.at: dict = {}
        for f in sorted(Path(path).glob("*.safetensors")):
            with safe_open(str(f), "pt") as h:
                for k in h.keys():
                    self.at[k.replace(*TEXT, 1) if k.startswith(TEXT[0]) else k] = (str(f), k)
        self.bytes = 0                          # read so far

    def _read(self, names: list) -> dict:
        from safetensors import safe_open
        by: dict = {}
        for n in names:
            f, k = self.at[n]
            by.setdefault(f, []).append((n, k))
        out = {}
        for f, nk in by.items():
            with safe_open(f, "pt") as h:
                for n, k in nk:
                    out[n] = h.get_tensor(k)
                    self.bytes += out[n].numel() * out[n].element_size()
        return out

    def tensors(self, names: list) -> dict:
        import torch
        out = self._read([n for n in names if n in self.at])
        for n in names:
            if n in out:
                continue
            if n + "_blocks" in self.at:            # MXFP4 (GPT-OSS's experts)
                from transformers.integrations.mxfp4 import convert_moe_packed_tensors
                q = self._read([n + "_blocks", n + "_scales"])
                out[n] = convert_moe_packed_tensors(q[n + "_blocks"], q[n + "_scales"],
                                                    dtype=self.dtype or torch.bfloat16)
                continue
            fused = next((s for s in FUSED if n.endswith(s)), None)
            if fused is None:
                raise KeyError(f"{n}: not in the checkpoint")
            base = n[:-len(fused)] + "mlp.experts."
            n_e = 1 + max(int(k[len(base):].split(".")[0]) for k in self.at if k.startswith(base))
            parts = self._read([f"{base}{e}.{p}.weight" for e in range(n_e) for p in FUSED[fused]])
            out[n] = torch.stack([torch.cat([parts[f"{base}{e}.{p}.weight"] for p in FUSED[fused]])
                                  for e in range(n_e)])
        return out


class Stream:
    """The checkpoint's text model with its decoder layers streamed: built on the meta device,
    the embedding, final norm, head and rotary embedding filled once, each layer's weights read
    by a forward pre-hook and moved back to the meta device by a forward hook."""

    def __init__(self, path, dtype: str = "bf16", layers: int | None = None):
        import torch
        import transformers
        self.torch = torch
        hc = transformers.AutoConfig.from_pretrained(path)
        tc = getattr(hc, "text_config", hc)
        if layers:
            tc.num_hidden_layers = layers
            if getattr(tc, "layer_types", None):
                tc.layer_types = tc.layer_types[:layers]
        if hasattr(tc, "quantization_config"):  # GPT-OSS's MXFP4: the model's own parameters,
            del tc.quantization_config          # the experts dequantized as read (Checkpoint)
        self.dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
        with torch.device("meta"):
            m = transformers.AutoModelForCausalLM.from_config(tc, dtype=self.dtype)
        m.eval()
        self.ck = Checkpoint(Path(path), self.dtype)
        rot = m.model.rotary_emb
        m.model.rotary_emb = type(rot)(config=tc)        # its inv_freq computed, on the CPU
        fixed = [n for n, _ in m.named_parameters() if not n.startswith("model.layers.")]
        tied = getattr(tc, "tie_word_embeddings", False) and "lm_head.weight" not in self.ck.at
        got = self.ck.tensors([n for n in fixed if not (tied and n == "lm_head.weight")])
        if tied:
            got["lm_head.weight"] = got["model.embed_tokens.weight"]
        m.load_state_dict({n: t.to(self.dtype) for n, t in got.items()}, strict=False,
                          assign=True)
        left = [n for n, t in list(m.named_parameters()) + list(m.named_buffers())
                if t.is_meta and not n.startswith("model.layers.")]
        if left:
            raise RuntimeError(f"not filled: {left}")
        for i, layer in enumerate(m.model.layers):
            layer.register_forward_pre_hook(self._fill(i))
            layer.register_forward_hook(self._drop)
        self.m, self.layer_s = m, []

    @staticmethod
    def _drop(mod, args, out) -> None:
        mod.to("meta")                          # (None: the layer's output stands)

    def _fill(self, i: int):
        def pre(mod, args):
            t = time.time()
            names = [n for n, _ in mod.named_parameters()]
            got = self.ck.tensors([f"model.layers.{i}.{n}" for n in names])
            mod.load_state_dict({n: got[f"model.layers.{i}.{n}"].to(self.dtype) for n in names},
                                strict=True, assign=True)
            self.layer_s.append(time.time() - t)
        return pre

    def logits(self, ids: list, keep: int) -> np.ndarray:
        """The last `keep` rows' logits (fp32) of one pass over `ids`."""
        with self.torch.no_grad():
            out = self.m(input_ids=self.torch.tensor([ids]), use_cache=False, logits_to_keep=keep)
        return out.logits[0].float().numpy()


def top(lg: np.ndarray, k: int = 5) -> list:
    i = np.argsort(-lg, kind="stable")[:k]
    return [(int(j), float(lg[j])) for j in i]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model", help="a short name (opentpu.llm.MODELS) or a checkpoint directory")
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--long", action="store_true", help="refs.py's long prompt")
    ap.add_argument("--xl", action="store_true", help="refs.py's xl prompt")
    ap.add_argument("--ref", help="a refs.py reference (.pkl): Hugging Face in its context")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--layers", type=int, help="the first N layers only")
    ap.add_argument("--json", help="write the steps here")
    a = ap.parse_args()
    import refs as R
    from transformers import AutoTokenizer

    from opentpu.llm import model_dir
    path = model_dir(a.model)
    tok = AutoTokenizer.from_pretrained(path)
    ids = R.prompt_ids(tok, a.long, a.xl)
    t0 = time.time()
    s = Stream(path, a.dtype, a.layers)
    print(f"{path.name}: {len(ids)} prompt tokens, {s.dtype}, built in {time.time() - t0:.1f} s; "
          f"{mem()}", flush=True)
    want = None
    if a.ref:
        want = [int(t) for t in pickle.loads(Path(a.ref).read_bytes())["tokens"][:a.tokens]]
    passes, rows, toks = [], [], []
    while len(rows) < (len(want) if want else a.tokens):
        t, b = time.time(), s.ck.bytes
        if want:
            rows = list(s.logits(ids + want[:-1], len(want)))
        else:
            rows.append(s.logits(ids + toks, 1)[-1])
        toks = [int(np.argmax(r)) for r in rows]
        passes.append({"s": round(time.time() - t, 1), "read_gb": round((s.ck.bytes - b) / 1e9, 1),
                       "layers_read_s": round(sum(s.layer_s), 1), **mem()})
        s.layer_s.clear()
        print(f"pass {len(passes)}: {passes[-1]}", flush=True)
    steps, first = [], None
    for j, r in enumerate(rows):
        tp = top(r)
        st = {"hf": toks[j], "top": tp, "gap12": tp[0][1] - tp[1][1]}
        if want:
            st["ref"] = want[j]
            st["ref_rank"] = int((r > r[want[j]]).sum()) + 1
            st["ref_gap"] = float(r.max() - r[want[j]])
            if first is None and toks[j] != want[j]:
                first = j
        steps.append(st)
        print(f"  step {j}: HF {toks[j]} {tok.decode([toks[j]])!r} (gap to #2 {st['gap12']:.3f})"
              + (f", ref {want[j]} {tok.decode([want[j]])!r} rank {st['ref_rank']}"
                 f" gap {st['ref_gap']:.3f}" if want else ""))
    print(f"HF: {tok.decode(toks)!r}")
    if want:
        print(f"ref: {tok.decode(want)!r}")
        print("SAME: Hugging Face's greedy tokens are the reference's" if first is None else
              f"DIFFER at step {first}: HF {toks[first]}, ref {want[first]}")
    if a.json:
        Path(a.json).write_text(json.dumps({"model": path.name, "dtype": a.dtype,
                                            "layers": a.layers, "prompt": ids, "hf": toks,
                                            "ref": want, "first_diff": first, "steps": steps,
                                            "passes": passes}, indent=1))
    return 0 if want is None or first is None else 1


if __name__ == "__main__":
    sys.exit(main())
