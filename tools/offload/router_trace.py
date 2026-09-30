"""Record a MoE model's expert choices on a text, for the expert-cache study (docs/offload.md).

    python3 tools/offload/router_trace.py --model DIR --text FILE [--text FILE ...]
                                          [--tokens 2048] [--out trace.npz] [--max-memory 8]

Runs the Hugging Face model (bf16, CPU) over each text once, teacher-forced (routing is causal:
a token's experts are the same in a prefill of the text as in a decode that produced it), with
weights beyond --max-memory GiB offloaded to disk, and records per MoE layer and token:

- `idx`: the experts the router picked (tokens x top-k) and `w` their routing weights;
- predictions of the same choice from earlier states of the residual stream (the router's 2 x
  top-k best, in its order), the router of the layer applied to (through the norm that feeds it):
  - `pre`: the layer's input (before its attention / convolution): known one mixer earlier;
  - `prev_r`: the previous layer's residual at its router: known one layer earlier;
  - `prev_in`: the previous layer's input: one layer and one mixer earlier.

The trace (`<out>`, one npz per text: `L<l>_idx`, `L<l>_w`, `L<l>_pre`, ...) feeds
tools/offload/cachesim.py. `ok` checks that the router applied to the recorded residual picks
the recorded experts (the norm mapping per architecture below is right).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

# the norm whose output the router reads (None: the router takes the residual and norms it itself)
ROUTER_NORM = {"lfm2_moe": "ffn_norm", "gemma4": None, "gemma4_text": None,
               **{a: "post_attention_layernorm" for a in (
                   "qwen3_moe", "qwen3_5_moe", "qwen3_5_moe_text", "qwen3_next", "olmoe",
                   "gpt_oss")}}


def layers_of(model):
    for path in ("model.language_model.layers", "model.layers", "language_model.model.layers",
                 "model.text_model.layers"):
        m = model
        try:
            for p in path.split("."):
                m = getattr(m, p)
            return m
        except AttributeError:
            continue
    raise SystemExit("no decoder layers found")


def router_of(layer):
    for name, m in layer.named_modules():
        if type(m).__name__.endswith("Router"):
            return name, m
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", action="append", required=True)
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--out", default="trace")
    ap.add_argument("--max-memory", type=float, default=8.0, help="GiB of weights kept in RAM")
    ap.add_argument("--offload-dir", default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--chat", action="store_true", help="wrap each text as one user turn")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    cfg = AutoConfig.from_pretrained(a.model)
    tc = getattr(cfg, "text_config", None) or cfg
    arch = tc.model_type
    norm_name = ROUTER_NORM[arch]
    tok = AutoTokenizer.from_pretrained(a.model)
    t0 = time.time()
    kw = dict(dtype=torch.bfloat16, low_cpu_mem_usage=True)
    if a.max_memory:
        off = a.offload_dir or str(Path(a.out).with_suffix("")) + ".offload"
        kw.update(device_map="auto", max_memory={"cpu": f"{a.max_memory}GiB"}, offload_folder=off)
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw).eval()
    print(f"loaded {arch} in {time.time() - t0:.0f} s", flush=True)
    layers = layers_of(model)

    rec: dict[str, np.ndarray] = {}
    state = {"guard": False, "h_in": {}, "r": {}, "prev": None}

    def topk_idx(out):
        return out[2].reshape(-1, out[2].shape[-1])

    def ranked(out, extra, n):
        """The n best experts in the router's selection order (its top-k first)."""
        score = out[0].reshape(-1, out[0].shape[-1]).float()
        if extra and extra[0] is not None:          # LFM2: sigmoid scores + the expert bias
            score = score.sigmoid() + extra[0].float()
        return torch.topk(score, n, dim=-1).indices
    def norm_of(layer):
        return getattr(layer, norm_name) if norm_name else (lambda x: x)

    moe = []
    for li, layer in enumerate(layers):
        rname, router = router_of(layer)
        if router is None:
            layer.register_forward_pre_hook(
                lambda m, args, kwargs, li=li: state["h_in"].__setitem__(li, _hs(args, kwargs)),
                with_kwargs=True)
            continue
        moe.append(li)

        def pre_layer(m, args, kwargs, li=li):
            state["h_in"][li] = _hs(args, kwargs)
        layer.register_forward_pre_hook(pre_layer, with_kwargs=True)
        if norm_name:
            getattr(layer, norm_name).register_forward_pre_hook(
                lambda m, args, li=li: state["r"].__setitem__(li, args[0].detach()))

        def post_router(m, args, kwargs, out, li=li, layer=layer):
            if state["guard"]:
                return
            state["guard"] = True
            try:
                x = args[0]
                if not norm_name:
                    state["r"][li] = x.detach()
                idx = topk_idx(out)
                w = out[1].reshape(idx.shape)
                rec[f"L{li}_idx"] = idx.to(torch.int16).numpy()
                rec[f"L{li}_w"] = w.float().to(torch.float16).numpy()
                nrm = norm_of(layer)
                extra = args[1:]

                k = idx.shape[-1]

                def route(h):
                    h = h.reshape(-1, h.shape[-1]).to(x.dtype)
                    return ranked(m(nrm(h), *extra, **kwargs), extra, 2 * k).to(torch.int16).numpy()
                rec[f"L{li}_ok"] = np.array(
                    (np.sort(route(state["r"][li])[:, :k], -1)
                     == np.sort(rec[f"L{li}_idx"], -1)).all(-1).mean(), np.float32)
                rec[f"L{li}_pre"] = route(state["h_in"][li])
                p = state["prev"]
                if p is not None:
                    rec[f"L{li}_prev_r"] = route(state["r"][p])
                    rec[f"L{li}_prev_in"] = route(state["h_in"][p])
                state["prev"] = li
            finally:
                state["guard"] = False
        router.register_forward_hook(post_router, with_kwargs=True)

    base = model.model
    for ti, path in enumerate(a.text):
        text = Path(path).read_text()
        if a.chat:
            text = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                           add_generation_prompt=False)
        ids = tok(text, return_tensors="pt").input_ids[:, :a.tokens]
        rec.clear()
        state.update(h_in={}, r={}, prev=None)
        t0 = time.time()
        with torch.no_grad():
            hs = base(input_ids=ids, use_cache=False).last_hidden_state
            # sanity: the model's loss on the last 128 tokens
            head = model.get_output_embeddings()
            n = min(128, ids.shape[1] - 1)
            logits = head(hs[0, -n - 1:-1]).float()
            soft = getattr(tc, "final_logit_softcapping", None)
            if soft:
                logits = torch.tanh(logits / soft) * soft
            nll = torch.nn.functional.cross_entropy(logits, ids[0, -n:]).item()
        dt = time.time() - t0
        out = f"{a.out}.{Path(path).stem}.npz"
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        meta = dict(model=a.model, arch=arch, text=path, tokens=int(ids.shape[1]), moe_layers=moe,
                    experts=int(getattr(tc, "num_experts", 0)
                                or getattr(tc, "num_local_experts", 0)),
                    nll_last128=nll, seconds=dt)
        np.savez_compressed(out, meta=json.dumps(meta), **rec)
        oks = [float(rec[f"L{li}_ok"]) for li in moe]
        print(f"{path}: {ids.shape[1]} tokens in {dt:.0f} s, nll(last {n}) {nll:.3f}, "
              f"router check min {min(oks):.4f} -> {out}", flush=True)


def _hs(args, kwargs):
    h = args[0] if args else kwargs["hidden_states"]
    return h.detach()


if __name__ == "__main__":
    main()
