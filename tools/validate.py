"""The device against a CPU golden: greedy tokens and logits of the card, the ISA simulator or
the RTL simulator next to Hugging Face transformers running the same checkpoint in fp32 on the
CPU with openTPU's quantization, and (--against) next to a second device.

    python3 tools/validate.py [--model qwen3|lfm2|qwen35|DIR] [--wformat int8|int4|fp4|mix]
                              [--head-format int8|int4|fp4] [--formats SPEC]
                              [--backend isa|rtl|board] [--against isa|rtl|board|RUN.npz]
                              [--cfg board|design|CFG.pkl] [--resident] [--tokens 16] [--chat]
                              [--weights-only] [--no-fp32] [--no-golden] [--min-top1 F]
                              [--kl-ratio R] [--max-kl NATS] [--save RUN.npz] [--json OUT.json]
                              [prompt ...]

The golden ("W+A"): the Hugging Face model with every matmul weight the device streams replaced
by the values the device multiplies (the model image's quantize_mxu in the format the image
gives that weight: --wformat, --head-format, --formats or OTPU_FORMATS; dequantized as the MXU
reads it), the embedding rows the device's, and the activations rounded where the device
rounds them, by its quantizer (fp32.quantize: int8 per 128-block, scale amax * (1/127),
rounding by 127 * recip(amax)): every nn.Linear's input, and in attention as _attend_heads
does, q * log2(e) / sqrt(d) and K per head and 128-block, V per head and token, and P V as V's
int8 values against the numerators 2^(s - max) times V's per-token scales, rounded per 128
keys. --weights-only quantizes the weights only ("W"); the fp32 golden (the checkpoint, Hugging
Face's eager attention) is there for context unless --no-fp32.

For each prompt: the device's greedy continuation and each golden's (text, ids, the first
token that differs); then, teacher forced on the device's own tokens (every step compares the
logits of the same context): top-1 agreement, KL(golden || device) mean and max, the largest
|logit difference| and the lowest cosine. Over all prompts also the golden against fp32 (the
quantization's own error) and against itself with its embedding rows moved by about an fp32
ulp: the floor. The golden cannot round exactly as the device does (its fp32 sums, norms and
exponentials differ in the last bits), and once an int8 value rounds the other way the
difference spreads through the later layers; a healthy device sits near that floor. The run
passes when the device's top-1 agreement with the golden is at least --min-top1 and its mean KL
at most --kl-ratio times the floor's (or --max-kl, whichever is larger); with --weights-only
only the top-1 bound holds.

--against runs a second device on the same prompts, or reads a run saved with --save: the
tokens must be the same and the logits bit-exact (the card and the simulators run the same
programs on the same arithmetic), else the run fails; it reports the largest difference in
ulps and the first step that differs. A card session can be the card's run alone (--backend
board --no-golden --save CARD.npz, seconds of the card); --against CARD.npz on a build host
then runs the ISA simulator in the configuration the card ran in, and the golden.

Devices: isa, the ISA simulator in --cfg's configuration (default isasim.board_config, with
OTPU_MCOLS etc. as there; a pickled Config; or "design"); rtl, the Verilator RTL
(opentpu.llm.rtl_backend: minutes per token on a real model, so a prompt and a few tokens of
LFM2.5-230M); board, the card through the host driver as otpu-chat --backend board (run it
under otpu-lock on the card host; a simulator then takes the card's configuration from its
registers). The device runs first and is freed before Hugging Face loads (Qwen3.5-0.8B's ISA
run peaks near 9 GB). Gemma 4 and the MoE models are not supported (unsupported()). Exit
status 0: PASS, 1: FAIL, 3: the card is in use (otpu-chat's).
"""
from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import pickle
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PROMPTS = ["A prime number larger than 100 is", "The capital of France is", "def fibonacci(n):",
           "Water boils at", "The quick brown fox",
           "In 1969, the first person to walk on the moon was",
           "The largest planet in the solar system is", "import numpy as np\n"]
# the bounds on the device against the quantized golden, teacher forced (measured on the ISA
# simulator, 8 prompts x 16 tokens, Qwen3-0.6B, LFM2.5-230M, Qwen3.5-0.8B in int8 and fp4: top-1
# 94.6-99.2%, KL mean 0.84-1.16x the floor's, per prompt 0.5-2.3x)
MIN_TOP1 = 0.8          # top-1 agreement (a fraction)
KL_RATIO = 3.0          # mean KL(golden || device) at most this times the floor's ...
MAX_KL = 1e-3           # ... or this (nats), whichever is larger (a floor near 0: no flips)
GOLDENS = {"quant": "W+A", "weights": "W", "fp32": "fp32"}


# ---------------------------------------------------------------------------- the golden
def quant_parts(x, g: int) -> tuple:
    """x (a torch tensor) quantized in groups of g along its last axis by the device's quantizer
    (fp32.quantize; the axis zero-padded to a multiple of g): (the int8 values as fp32 [..., n],
    the scales [..., n / g])."""
    import torch

    from opentpu import fp32 as F
    a = x.detach().to(torch.float32).cpu().numpy()
    n = a.shape[-1]
    if n % g:
        a = np.pad(a, [(0, 0)] * (a.ndim - 1) + [(0, -n % g)])
    q, s = F.quantize(a.reshape(*a.shape[:-1], -1, g), axis=-1)
    q = q.astype(np.float32).reshape(a.shape)[..., :n]
    return torch.from_numpy(np.ascontiguousarray(q)), torch.from_numpy(s)


def fake_quant(x, g: int):
    """x quantized as quant_parts and dequantized: q * s in fp32, the values the MXU multiplies."""
    q, s = quant_parts(x, g)
    n = q.shape[-1]
    return (q * s.repeat_interleave(g, -1)[..., :n]).to(x.dtype)


def device_weight(w, fmt: str, D: int = 128) -> np.ndarray:
    """The values the MXU multiplies for a weight matrix w [N, K] in `fmt` (int8, int4, fp4):
    the image's quantize_mxu (opentpu.qcache, as the images' builds), dequantized as the MXU
    reads it; fp32 [N, K]. Quantization is per row and K-block, so a whole checkpoint matrix
    gives the values of the row and column parts the images place."""
    from opentpu import qcache as QC
    from opentpu import quant as Q
    w = np.asarray(w, np.float32)
    N, K = w.shape
    if K % D:                                   # (zero columns: the image's padding)
        w = np.pad(w, ((0, 0), (0, -K % D)))
    Kp = w.shape[1]
    q, s = QC.quantize_mxu(w, fmt, D)
    out = np.empty((N, Kp), np.float32)
    for r in range(0, N, Q.QUANT_ROWS):
        e = min(N, r + Q.QUANT_ROWS)
        if fmt == "int8":
            qq = q[r:e].view(np.int8).reshape(e - r, Kp // D, D).astype(np.float32)
            out[r:e] = (qq * s[r:e].view(np.float32)[..., None]).reshape(e - r, Kp)
        else:
            out[r:e] = Q.dequantize_w4(q[r:e, :Kp // 2], s[r:e], fmt, D)
    return out[:, :K]


def embed_rows(table, spec, D: int = 128) -> np.ndarray:
    """The embedding rows the device computes with (qwen3.Embedding, for the whole table):
    the fp32 rows, or with spec.embed "int8" the int8 rows as the device's gather dequantizes
    them (kernels.gather.dequant_blocks, bit for bit)."""
    from opentpu import quant as Q
    from opentpu.kernels.gather import dequant_blocks
    t = np.asarray(table, np.float32)
    if getattr(spec, "embed", "f32") != "int8":
        return t
    out = np.empty_like(t)
    V, H = t.shape
    for r in range(0, V, Q.QUANT_ROWS):
        e = min(V, r + Q.QUANT_ROWS)
        q, s = Q.quantize_mxu(t[r:e], "int8", D)
        out[r:e] = dequant_blocks(q.view(np.int8).reshape(-1, D), s.view(np.float32).reshape(-1),
                                  "int8", D).reshape(e - r, H)
    return out


def unsupported(spec) -> str | None:
    """Why the golden cannot stand for this model's device, or None. Gemma 4's device gathers
    its embedding rows from the LM head in the head's format and its PLE rows from records of
    their own, which the golden does not model (and E2B is about 20 GB in fp32, beside the ISA
    run's 9); a MoE's routers are int8 in every format and its experts stream."""
    if type(spec).__module__.endswith(".gemma4"):
        return "Gemma 4 is not supported: its embedding and PLE rows are the device's own gathers"
    if getattr(spec, "moe", None) is not None:
        return "MoE models are not supported (the routers' formats, the experts' streaming)"
    return None


class Golden:
    """The CPU golden: a Hugging Face causal LM (fp32, CPU) whose matmuls see what the device's
    see. set("quant"): the device's weight values, embedding rows and activation quantization;
    set("weights"): its weight values and embedding rows only; set("fp32"): the checkpoint.

    The weight formats are the model image's: the port's weight_kind and formats.resolver
    (wformat, head_format, OTPU_FORMATS or spec.formats), as Image and emulated_logits."""

    def __init__(self, model, spec, wformat: str = "int8", head_format: str | None = None,
                 D: int = 128):
        import torch
        from transformers import AttentionInterface
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, AttentionMaskInterface

        from opentpu.llm import formats as FM
        why = unsupported(spec)
        if why:
            raise ValueError(why)
        self.model, self.spec, self.D, self.act = model.eval(), spec, D, False
        port = importlib.import_module(type(spec).__module__)
        wf, formats = FM.named(spec, wformat, None)
        fmt = FM.resolver(formats, port.KINDS, spec.formats, wf, head_format)
        self.formats = {}                   # checkpoint weight -> its format
        self._lin = []                      # (module, the checkpoint's weight, the device's)
        for n, m in model.named_modules():
            if not isinstance(m, torch.nn.Linear):
                continue
            name = n.replace("model.language_model.", "model.") + ".weight"
            f = fmt(*port.weight_kind(name))
            dw = device_weight(m.weight.detach().numpy(), f, D)
            self._lin.append((m, m.weight, torch.nn.Parameter(torch.from_numpy(dw), False)))
            self.formats[name] = f
            m.register_forward_pre_hook(self._quant_input)
        emb = model.get_input_embeddings()
        self._emb = (emb, emb.weight, emb.weight)
        if getattr(spec, "embed", "f32") == "int8":
            rows = embed_rows(emb.weight.detach().numpy(), spec, D)
            self._emb = (emb, emb.weight, torch.nn.Parameter(torch.from_numpy(rows), False))
        self.noise = False                  # the embedding rows times (1 + 2^-23 N(0, 1))
        emb.register_forward_hook(self._noisy)
        # the attention: this golden's function (and the eager mask) under a name of its own
        key = f"otpu_golden_{id(self):x}"
        AttentionInterface.register(key, self._attention)
        AttentionMaskInterface.register(key, ALL_MASK_ATTENTION_FUNCTIONS["eager"])
        model.set_attn_implementation(key)
        self._eager = {}

    def set(self, mode: str) -> None:
        for m, w32, wd in self._lin:
            m.weight = w32 if mode == "fp32" else wd
        emb, w32, wd = self._emb
        emb.weight = w32 if mode == "fp32" else wd
        self.act = mode == "quant"

    def _noisy(self, mod, args, out):
        """With `noise`, the embedding rows perturbed by about an fp32 ulp, the same noise in
        every pass (a fixed seed): the golden against itself so perturbed is its noise floor,
        the divergence that rounding differences alone cause once a quantized value flips."""
        import torch
        if not self.noise:
            return None
        gen = torch.Generator().manual_seed(1)
        return out * (1 + 2.0 ** -23 * torch.randn(out.shape, generator=gen, dtype=out.dtype))

    def _quant_input(self, mod, args):
        if self.act:
            return (fake_quant(args[0], self.D),) + tuple(args[1:])
        return None

    def _attention(self, module, q, k, v, mask, scaling=None, dropout=0.0, **kw):
        """softmax(q k^T * scaling + mask) v; with the activations quantized, the device's
        flash attention (kernels/attention.py _attend_heads): q * scaling * log2(e) and K in
        int8 per head and min(d, D)-block, V per head with one scale per token, and P V as the
        int8 values of V against the numerators 2^(s - max) times V's scales (folded in, QACT
        CSCALE), quantized per D keys, divided by the numerators' unquantized sum.
        Otherwise the model's own eager attention."""
        import torch
        if not self.act:
            t = type(module).__module__
            if t not in self._eager:
                self._eager[t] = importlib.import_module(t).eager_attention_forward
            return self._eager[t](module, q, k, v, mask, scaling=scaling, dropout=0.0, **kw)
        from opentpu.language import LOG2E
        d = q.shape[-1]
        scaling = d ** -0.5 if scaling is None else scaling
        g = min(d, self.D)
        # the device's scores are in log2 units: q times log2(e) * scaling (one fp32 product),
        # then exp2
        q = fake_quant(q * torch.tensor(LOG2E * scaling, dtype=torch.float32), g)
        k = fake_quant(k, g)
        vq, vs = quant_parts(v, d)                          # [B, Hkv, T, d], [B, Hkv, T, 1]
        G = q.shape[1] // k.shape[1]
        k, vq, vs = (t.repeat_interleave(G, 1) for t in (k, vq, vs))
        s = q @ k.transpose(2, 3)
        if mask is not None:
            s = s + mask[..., :k.shape[2]]
        p = torch.exp2(s - s.amax(-1, keepdim=True))
        o = (fake_quant(p * vs[..., 0][:, :, None, :], self.D) @ vq) / p.sum(-1, keepdim=True)
        return o.transpose(1, 2).contiguous(), None

    def logits(self, ids) -> np.ndarray:
        """Logits [len(ids), vocab] of the whole sequence (causal), fp32."""
        import torch
        with torch.no_grad():
            return self.model(torch.tensor([list(ids)]), use_cache=False).logits[0].float().numpy()

    def greedy(self, ids, n: int, eos, start=()) -> list:
        """Greedy continuation of ids (up to n tokens, or eos), from `start` (its first tokens,
        known): one forward of the whole sequence a token (no cache: every step is the same
        computation as the teacher-forced pass)."""
        got = list(start)
        while len(got) < n and not (got and got[-1] in eos):
            got.append(int(np.argmax(self.logits(list(ids) + got)[-1])))
        return got


# ---------------------------------------------------------------------------- the device
def device_runs(eng, prompts, n: int, eos) -> list:
    """Greedy decoding of each prompt (token ids) on an Engine: [{"tokens", "logits"}], logits[i]
    the device's logits [vocab] that chose tokens[i] (the prompt's prefill, then a step a
    token; up to n tokens, or eos)."""
    out = []
    for ids in prompts:
        eng.reset()
        lg = [np.array(eng.prefill(ids), np.float32)]
        got = []
        while True:
            got.append(int(np.argmax(lg[-1])))
            if got[-1] in eos or len(got) == n:
                break
            lg.append(np.array(eng.step(got[-1]), np.float32))
        out.append({"tokens": got, "logits": np.stack(lg)})
    return out


def sim_cfg(spec, cap: int, a, card=None):
    """The simulators' configuration: the card's (`card`: from its registers, or the one a saved
    card run ran in), else --cfg's ("board" or none: isasim.board_config; a pickled Config;
    "design": None, the Engine's default), with the DRAM cut to what the model needs
    (host.board.sim_config, as tools/qual/refs.py)."""
    from opentpu.host.board import sim_config
    from opentpu.isasim import board_config
    if card is None and a.cfg == "design":
        return None
    base = card or (board_config() if a.cfg in (None, "board") else
                    pickle.loads(Path(a.cfg).read_bytes()))
    return sim_config(spec, cap, base, lookup=a.resident, wformat=a.wformat,
                      head_format=a.head_format)


def run_device(name: str, spec, path: Path, cap: int, prompts, a, card=None, W=None) -> tuple:
    """(runs, configuration) of device `name` on the prompts; card: the card's (backend
    factory, configuration) when the card is one of the devices (opened once); W: the
    weights (default: the checkpoint's at `path`)."""
    from opentpu.llm.qwen3 import Engine, load_weights
    kw = dict(cap=cap, wformat=a.wformat, head_format=a.head_format, resident=a.resident)
    W = load_weights(path) if W is None else W
    if name == "board":
        eng = Engine(spec, W, cfg=card[1], backend=card[0], **kw)
    else:
        cfg = sim_cfg(spec, cap, a, card and card[1])
        if name == "rtl":
            from opentpu.llm.rtl_backend import RtlBackend
            eng = Engine(spec, W, cfg=cfg, backend=RtlBackend, **kw)
        else:
            eng = Engine(spec, W, cfg=cfg, **kw)
    cfg = eng.cfg
    print(f"device {name}: {cfg}, weights {a.wformat}, head {eng.image.head_format}"
          f"{', resident decode' if eng.resident else ''}", flush=True)
    try:
        runs = device_runs(eng, prompts, a.tokens, spec.eos)
    finally:
        eng._drain()
        if hasattr(eng.backend, "close"):
            eng.backend.close()
    del eng, W
    gc.collect()
    return runs, cfg


def open_card(spec, path: Path, cap: int, dev: str):
    """The card's (backend factory, configuration), as otpu-chat --backend board."""
    from opentpu.host.board import ConfigMismatch
    from opentpu.host.chat import make_backend
    try:
        return make_backend("board", spec, cap, dev, path.name)
    except (OSError, ConfigMismatch) as e:
        raise SystemExit(f"board: {e} (the card host, under otpu-lock)") from None


def arch(cfg) -> str:
    """A configuration without its DRAM size (the simulators' is cut to the model's image)."""
    return repr(replace(cfg, DRAM_BYTES=0))


def save_run(f: str, runs, meta: dict, prompts) -> None:
    arrs = {f"ids{i}": np.array(p, np.int64) for i, p in enumerate(prompts)}
    for i, r in enumerate(runs):
        arrs[f"tokens{i}"] = np.array(r["tokens"], np.int64)
        arrs[f"logits{i}"] = r["logits"]
    np.savez(f, meta=json.dumps(meta), **arrs)


def load_run(f: str, meta: dict, prompts) -> tuple:
    """(runs, its meta) of a saved run; SystemExit unless it is of the same model, weights,
    prompts and token count."""
    z = np.load(f)
    m = json.loads(str(z["meta"]))
    for k in ("model", "wformat", "head_format", "formats", "tokens"):
        if m.get(k) != meta.get(k):
            raise SystemExit(f"{f}: {k} {m.get(k)!r}, this run's is {meta.get(k)!r}")
    if len(prompts) != m["prompts"] or any(z[f"ids{i}"].tolist() != list(p)
                                           for i, p in enumerate(prompts)):
        raise SystemExit(f"{f}: other prompts (or another chat template / tokenizer)")
    return [{"tokens": z[f"tokens{i}"].tolist(), "logits": z[f"logits{i}"]}
            for i in range(len(prompts))], m


# ---------------------------------------------------------------------------- comparisons
def _logsoftmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def divergence(dev: np.ndarray, gold: np.ndarray) -> dict:
    """Teacher-forced metrics of a device's logits against a golden's on the same contexts
    ([steps, vocab] each): top-1 agreement, KL(golden || device) per step, the largest
    |difference| and the lowest cosine."""
    d, g = np.asarray(dev, np.float64), np.asarray(gold, np.float64)
    lg, ld = _logsoftmax(g), _logsoftmax(d)
    kl = (np.exp(lg) * (lg - ld)).sum(-1)
    cos = (d * g).sum(-1) / np.linalg.norm(d, axis=-1) / np.linalg.norm(g, axis=-1)
    return {"steps": len(d), "top1": int((d.argmax(-1) == g.argmax(-1)).sum()),
            "kl": kl.tolist(), "max_abs": float(np.abs(d - g).max()), "min_cos": float(cos.min())}


def summary(divs: list) -> dict:
    """divergence() over all prompts: top-1 agreement (fraction), mean and max KL, ..."""
    kl = [x for v in divs for x in v["kl"]]
    steps = sum(v["steps"] for v in divs)
    return {"steps": steps, "top1": sum(v["top1"] for v in divs) / steps,
            "kl_mean": float(np.mean(kl)), "kl_max": float(np.max(kl)),
            "max_abs": max(v["max_abs"] for v in divs),
            "min_cos": min(v["min_cos"] for v in divs)}


def first_diff(a, b):
    """The first index where two token lists differ (one ending early counts), else None."""
    k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    return k if k is not None or len(a) == len(b) else min(len(a), len(b))


def ulps(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """|a - b| in units in the last place of fp32 (ordered bit patterns; -0 == +0)."""
    def ordered(x):
        i = np.asarray(x, np.float32).view(np.int32).astype(np.int64)
        return np.where(i < 0, -(i & 0x7FFFFFFF), i)
    return np.abs(ordered(a) - ordered(b))


def agreement(runs_a, runs_b) -> dict:
    """Two devices' runs of the same prompts: per prompt the first token that differs and, over
    the steps with the same context (up to and including that token), whether the logits are
    bit-exact, their largest difference in ulps and the first step that differs."""
    per = []
    for ra, rb in zip(runs_a, runs_b):
        k = first_diff(ra["tokens"], rb["tokens"])
        n = min(len(ra["logits"]), len(rb["logits"]), len(ra["tokens"]) if k is None else k + 1)
        u = ulps(ra["logits"][:n], rb["logits"][:n])
        bad = np.nonzero(u.max(-1))[0]
        per.append({"token_diff": k, "steps": int(n), "max_ulp": int(u.max()),
                    "first_step": int(bad[0]) if len(bad) else None,
                    "max_abs": float(np.abs(ra["logits"][:n].astype(np.float64)
                                            - rb["logits"][:n]).max())})
    ok = all(p["token_diff"] is None and p["first_step"] is None for p in per)
    return {"pass": ok, "prompts": per,
            "same_tokens": sum(p["token_diff"] is None for p in per),
            "bit_exact": sum(p["first_step"] is None for p in per),
            "steps": sum(p["steps"] for p in per), "max_ulp": max(p["max_ulp"] for p in per)}


def against_golden(golden, prompts, runs, modes, n: int, eos) -> list:
    """Per prompt: for each golden mode the teacher-forced divergence() of the device on its own
    sequence and the golden's greedy continuation (up to n tokens or eos), {"goldens": {mode:
    {...}}}; and "pairs": the first golden against fp32 on the same contexts (the error of its
    quantization alone) and against itself with 1-ulp input noise ("floor", Golden.noise)."""
    out = [{"goldens": {}, "pairs": {}} for _ in prompts]
    keep = [{} for _ in prompts]                # the goldens' logits, for the pairs
    for mode in modes + ["floor"]:
        golden.set(modes[0] if mode == "floor" else mode)
        golden.noise = mode == "floor"
        for i, (ids, r) in enumerate(zip(prompts, runs)):
            got = r["tokens"]
            gl = golden.logits(list(ids) + got[:-1])[len(ids) - 1:]
            v = min(gl.shape[1], r["logits"].shape[1])
            gl = gl[:, :v]
            if mode == "floor":
                out[i]["pairs"]["floor"] = divergence(gl, keep[i][modes[0]])
                continue
            keep[i][mode] = gl
            # the golden's continuation is the device's up to the first step where the golden
            # picks another token, then its own
            k = next((j for j, t in enumerate(gl.argmax(-1)) if t != got[j]), None)
            want = got if k is None else golden.greedy(ids, n, eos,
                                                       start=got[:k] + [int(gl[k].argmax())])
            out[i]["goldens"][mode] = {"tokens": want, "first_diff": first_diff(got, want),
                                       **divergence(r["logits"][:, :v], gl)}
        if mode == "fp32" and modes[0] != "fp32":
            for i in range(len(prompts)):
                out[i]["pairs"]["fp32"] = divergence(keep[i][modes[0]], keep[i]["fp32"])
    golden.noise = False
    golden.set("fp32")
    return out


# ---------------------------------------------------------------------------- the report
def fmt_div(s: dict) -> str:
    return (f"{100 * s['top1']:6.1f}%  {s['kl_mean']:9.2e}  {s['kl_max']:9.2e}  "
            f"{s['max_abs']:8.3f}  {s['min_cos']:.6f}")


HEAD = f"{'top-1':>7s}  {'KL mean':>9s}  {'KL max':>9s}  {'max|dl|':>8s}  {'min cos':>8s}"


def report(tok, texts, prompts, runs, gold, modes, dev_name, other, agree, limits) -> dict:
    """Prints the comparison; returns it (the --json document) with "pass". modes: the goldens
    in `gold` (none: the device and --against only)."""
    dec = (lambda t: tok.decode(t)) if tok else str
    g0 = GOLDENS[modes[0]] if modes else None
    rows = [(f"device vs {GOLDENS[m]}", lambda e, m=m: e["goldens"][m]) for m in modes]
    pairs = [(f"{g0} vs fp32", "fp32", "its quantization's error"),
             (f"{g0}~ vs {g0}", "floor", "1-ulp input noise: the floor")]
    pairs = [p for p in pairs if gold and p[1] in gold[0]["pairs"]]
    w = max([len(r[0]) for r in rows + pairs] + [len(f"vs {other}") - 3, 14]) + 3
    res = {"prompts": [], "goldens": {}, "pairs": {}}
    for i, (text, ids, r) in enumerate(zip(texts, prompts, runs)):
        print(f"\n{text!r} ({len(ids)} tokens)")
        print(f"   {'device':{w - 3}s}: {dec(r['tokens'])!r}  {r['tokens']}")
        e = {"prompt": text, "ids": list(map(int, ids)), "device": r["tokens"], "goldens": {},
             "pairs": gold[i]["pairs"] if gold else {}}
        for m in modes:
            g = gold[i]["goldens"][m]
            k = g["first_diff"]
            same = f"same {len(r['tokens'])} tokens" if k is None else \
                f"first difference at token {k + 1}: {dec(g['tokens'][k:k + 1])!r} vs the " \
                f"device's {dec(r['tokens'][k:k + 1])!r}; {dec(g['tokens'])!r}  {g['tokens']}"
            print(f"   {'golden ' + GOLDENS[m]:{w - 3}s}: {same}")
            e["goldens"][m] = g
        if modes:
            print(f"   {'teacher forced':{w}s}{HEAD}")
        for lab, get in rows:
            print(f"   {lab:{w}s}{fmt_div(summary([get(gold[i])]))}")
        if agree:
            p = agree["prompts"][i]
            print(f"   {'vs ' + other:{w - 3}s}: " + (
                "same tokens" if p["token_diff"] is None else
                f"tokens differ at {p['token_diff'] + 1}") + "; logits " + (
                f"bit-exact over {p['steps']} steps" if p["first_step"] is None else
                f"differ from step {p['first_step'] + 1}: max {p['max_ulp']} ulp, "
                f"|d| {p['max_abs']:.3g}"))
        res["prompts"].append(e)
    steps = sum(len(r["tokens"]) for r in runs)
    ok, lines = True, []
    if modes:
        print(f"\nall {len(runs)} prompts, {steps} tokens from {dev_name}, teacher forced on the "
              f"device's sequences:")
        print(f"   {'':{w}s}{HEAD}")
    for m in modes:
        s = summary([g["goldens"][m] for g in gold])
        same = sum(g["goldens"][m]["first_diff"] is None for g in gold)
        print(f"   {'device vs ' + GOLDENS[m]:{w}s}{fmt_div(s)}   greedy tokens the same in "
              f"{same}/{len(runs)} prompts")
        res["goldens"][m] = {**s, "same_tokens": same}
    for lab, key, what in pairs:
        s = summary([g["pairs"][key] for g in gold])
        print(f"   {lab:{w}s}{fmt_div(s)}   ({what})")
        res["pairs"][key] = s
    if modes:
        first = res["goldens"][modes[0]]
        min_top1, ratio, max_kl = limits
        ok = first["top1"] >= min_top1
        line = f"device vs golden {g0}: top-1 {100 * first['top1']:.1f}% (at least " \
            f"{100 * min_top1:g}%), KL mean {first['kl_mean']:.2e} "
        if modes[0] == "quant":             # the floor: what rounding alone does to the golden
            fl = res["pairs"]["floor"]["kl_mean"]
            cap = max(max_kl, ratio * fl)
            ok = ok and first["kl_mean"] <= cap
            line += (f"= {first['kl_mean'] / fl:.2f}x the floor's {fl:.2e}" if fl > 0 else
                     "(the floor 0)") + f" (at most {cap:.2e})"
        else:                               # (the device rounds activations; this golden not)
            line += "(no KL bound without the activations' rounding)"
        lines.append(line + f": {'ok' if ok else 'FAIL'}")
    if agree:
        res["against"] = {"device": other, **agree}
        a = agree
        lines.append(f"{dev_name} vs {other}: tokens the same in {a['same_tokens']}/{len(runs)}"
                     f" prompts, logits bit-exact in {a['bit_exact']}/{len(runs)} ({a['steps']} "
                     f"steps, max {a['max_ulp']} ulp): {'ok' if a['pass'] else 'FAIL'}")
        ok = ok and a["pass"]
    if not lines:
        lines.append(f"{steps} tokens from {dev_name}; nothing to compare (--no-golden)")
    print("\n" + "\n".join(lines))
    print("PASS" if ok else "FAIL")
    res["pass"] = ok
    return res


# ---------------------------------------------------------------------------- main
def main(argv=None) -> int:
    from opentpu.host.runstate import busy_exits
    return busy_exits(_main)(argv)


def _main(argv=None) -> int:
    from opentpu.llm import MODELS, load_spec, model_dir

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("prompts", nargs="*", default=PROMPTS)
    ap.add_argument("--model", default="qwen3",
                    help=f"{' or '.join(MODELS)} (models/<name>), or a checkpoint directory")
    ap.add_argument("--wformat", default="int8", choices=["int8", "int4", "fp4", "mix"])
    ap.add_argument("--head-format", default=None, choices=["int8", "int4", "fp4"])
    ap.add_argument("--formats", help="per-kind weight formats (OTPU_FORMATS, "
                    "opentpu/llm/formats.py), e.g. mlp=fp4")
    ap.add_argument("--backend", default="isa", choices=["isa", "rtl", "board"])
    ap.add_argument("--against", help="a second device (isa, rtl, board) or a run saved with "
                    "--save: tokens and logits must be identical")
    ap.add_argument("--cfg", help="the simulators' configuration: board (the default, "
                    "isasim.board_config), design, or a pickled Config (tools/qual/refs.py cfg);"
                    " with the card, or --against a saved card run, the card's")
    ap.add_argument("--dev", default="/dev/xdma0", help="the card's XDMA device")
    ap.add_argument("--resident", action="store_true", help="the resident decode program")
    ap.add_argument("--tokens", type=int, default=16, help="greedy tokens per prompt")
    ap.add_argument("--chat", action="store_true", help="prompts as user turns of the chat "
                    "template")
    ap.add_argument("--weights-only", action="store_true", help="the golden quantizes the "
                    "weights only (fp32 activations)")
    ap.add_argument("--no-fp32", action="store_true", help="no fp32 golden")
    ap.add_argument("--no-golden", action="store_true", help="no golden: the device's run "
                    "(--save) and --against only (a card session's part)")
    ap.add_argument("--min-top1", type=float, default=MIN_TOP1,
                    help="the device's top-1 agreement with the golden, at least")
    ap.add_argument("--kl-ratio", type=float, default=KL_RATIO,
                    help="its mean KL at most this times the golden's floor ...")
    ap.add_argument("--max-kl", type=float, default=MAX_KL, help="... or this, in nats")
    ap.add_argument("--save", help="save the device's run (.npz) for a later --against")
    ap.add_argument("--json", help="write the results here")
    a = ap.parse_args(argv)
    if a.formats is not None:
        os.environ["OTPU_FORMATS"] = a.formats
    import torch
    import transformers

    path = model_dir(a.model)
    spec = load_spec(path)
    if unsupported(spec):                       # (before the device's run)
        raise SystemExit(f"{path.name}: {unsupported(spec)}")
    tok = transformers.AutoTokenizer.from_pretrained(path)

    def encode(p):
        if not a.chat:
            return tok(p).input_ids
        ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                      enable_thinking=False, tokenize=True)
        return list(ids["input_ids"] if hasattr(ids, "keys") else ids)

    prompts = [encode(p) for p in a.prompts]
    need = max(len(ids) for ids in prompts) + a.tokens
    cap = max(256, -(-need // 128) * 128)
    meta = {"model": path.name, "wformat": a.wformat, "head_format": a.head_format,
            "formats": os.environ.get("OTPU_FORMATS"), "tokens": a.tokens,
            "prompts": len(prompts), "resident": a.resident}
    card = open_card(spec, path, cap, a.dev) if "board" in (a.backend, a.against) else None
    other = name = None
    if a.against and a.against not in ("isa", "rtl", "board"):     # a saved run
        other, m = load_run(a.against, meta, prompts)
        name = f"{m.get('backend', 'saved')} ({Path(a.against).name})"
        if card is None and a.cfg is None and m.get("backend") == "board":
            from opentpu.isasim import Config         # the simulators in the card's configuration
            card = (None, Config(**m["config"]))
    runs, cfg = run_device(a.backend, spec, path, cap, prompts, a, card)
    if a.save:
        save_run(a.save, runs, {**meta, "backend": a.backend, "cfg": repr(cfg),
                                "arch": arch(cfg), "config": asdict(cfg)}, prompts)
        print(f"saved the {a.backend} run in {a.save}")
    agree = None
    if other is not None:
        if m.get("arch") != arch(cfg):              # (the DRAM size aside)
            print(f"note: {a.against} ran in {m.get('cfg')}, this device in {cfg}: other "
                  f"programs, bit-exact logits not expected")
    elif a.against:
        other, _ = run_device(a.against, spec, path, cap, prompts, a, card)
        name = a.against
    if other is not None:
        agree = agreement(runs, other)
    gold, modes, formats = None, [], None
    if not a.no_golden:             # Hugging Face after the devices (freed): fp32 on the CPU
        torch.set_grad_enabled(False)
        hf = transformers.AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
        golden = Golden(hf, spec, a.wformat, a.head_format)
        modes = ["weights" if a.weights_only else "quant"] + ([] if a.no_fp32 else ["fp32"])
        gold = against_golden(golden, prompts, runs, modes, a.tokens, spec.eos)
        formats = golden.formats
    res = report(tok, a.prompts, prompts, runs, gold, modes, a.backend, name, agree,
                 (a.min_top1, a.kl_ratio, a.max_kl))
    if a.json:
        res.update(meta=meta, cfg=repr(cfg), backend=a.backend, formats=formats)
        Path(a.json).write_text(json.dumps(res, indent=1))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
