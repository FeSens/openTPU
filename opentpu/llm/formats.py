"""Weight formats per kind and layer: the rules every model's image reads (Gemma 4's first;
docs/gemma4_e4b.md). A model's weights are in its image's `wformat`, except where a formats
string says otherwise:

    "kind=fmt,kind@a-b=fmt,..."     fmt: int8, int4 or fp4; a..b: checkpoint layers (a: one)

A rule with a range wins over one without, and among equals the first one in the string. Each
model names the kinds it has (Gemma 4: attn, mlp, gateup, down, ple, head, experts; Qwen3 and
the Llama-likes: attn, mlp, gateup, down, head; LFM2 also conv, Qwen3.5 also delta); "mlp" is
the default of "gateup" and "down", and "head" takes no range. The string comes from the
caller, else the OTPU_FORMATS environment variable, else the model's own default
(Spec.formats).

The Qwen3 / Llama-like, LFM2 and Qwen3.5 models also have a named choice, wformat "mix": int8
with the model's recommended formats string (Spec.mix, from MIXES: docs/formats.md), unless
the caller or OTPU_FORMATS gives one. wformat int8, int4 and fp4 stay uniform. wformat "auto"
(otpu-chat's default, auto()) is "mix" where the model has a mix, else int8.
"""
from __future__ import annotations

import os

FORMATS = ("int8", "int4", "fp4")
MIX = "mix"                 # the named choice: int8 with the model's mix (Spec.mix)
AUTO = "auto"               # "mix" where the model has one, else int8 (otpu-chat's default)

# the recommended mixes over int8 (wformat "mix", docs/formats.md), by (model_type, layers,
# hidden size, vocabulary) of the checkpoint's config
MIXES: dict = {
    ("phi3", 32, 3072, 200064): "mlp@4-29=fp4",                 # Phi-4-mini
    ("qwen3_5_text", 32, 2560, 248320): "delta=fp4,mlp=fp4",    # Qwen3.5-4B
    ("smollm3", 36, 2048, 128256): "gateup@9-35=fp4",           # SmolLM3-3B
    ("lfm2", 30, 2048, 65536): "conv=fp4,mlp=fp4",              # LFM2-2.6B
    ("gemma4_text", 35, 1536, 262144): "attn@15-24=fp4,mlp@15-34=fp4",  # Gemma 4 E2B
}


def mix_for(config: dict) -> str:
    """The recommended mix of a checkpoint's (text) config, "" for none."""
    return MIXES.get((config.get("model_type"), config.get("num_hidden_layers"),
                      config.get("hidden_size"), config.get("vocab_size")), "")


def auto(spec, wformat: str) -> str:
    """wformat "auto": "mix" where the model has a recommended mix (Spec.mix), else int8; any
    other wformat is itself."""
    if wformat != AUTO:
        return wformat
    return MIX if getattr(spec, "mix", "") else "int8"


def named(spec, wformat: str, formats: str | None) -> tuple:
    """(wformat, formats) of a weight choice: "mix" is int8 with spec.mix (unless `formats`
    or OTPU_FORMATS gives a formats string); any other wformat is itself."""
    if wformat != MIX:
        return wformat, formats
    if formats is None:
        formats = os.environ.get("OTPU_FORMATS", getattr(spec, "mix", ""))
    if not formats:
        raise ValueError("wformat mix: this model has no recommended mix (Spec.mix, "
                         "formats.MIXES)")
    return "int8", formats


def rules(formats: str | None, kinds, default: str = "", unranged=("head",)) -> list:
    """The rules of a formats string (None: OTPU_FORMATS, else `default`): a list of (kind,
    first layer, last layer, ranged, format). ValueError names a malformed item."""
    if formats is None:
        formats = os.environ.get("OTPU_FORMATS", default)
    out = []
    for item in formats.replace(" ", "").split(","):
        if not item:
            continue
        key, _, fmt = item.partition("=")
        kind, _, span = key.partition("@")
        lo, _, hi = span.partition("-")
        if kind not in kinds or fmt not in FORMATS or (span and not (lo + hi).isdigit()) or \
                (span and kind in unranged):
            raise ValueError(f"weight format {item!r}: kind[@a-b]=int8|int4|fp4")
        out.append((kind, int(lo) if span else 0, int(hi or lo) if span else 1 << 30,
                    bool(span), fmt))
    return out


def pick(rules_: list, kind: str, layer: int, default):
    """The format of `kind` in checkpoint layer `layer`: the first ranged rule that covers it,
    else the first rule for the whole model, else `default`."""
    hit = sorted((not r, i) for i, (k, lo, hi, r, _) in enumerate(rules_)
                 if k == kind and lo <= layer <= hi)
    return rules_[hit[0][1]][4] if hit else default


def plain(rules_: list) -> dict:
    """{kind: format} of the rules without a range (the first one of each kind)."""
    return {r[0]: r[4] for r in reversed(rules_) if not r[3]}


SUB = {"gateup": "mlp", "down": "mlp"}      # a kind whose default is another kind's format


def resolver(formats: str | None, kinds, default: str, wformat: str,
             head_format: str | None = None):
    """fmt(kind, layer): the format of a weight of `kind` in checkpoint layer `layer` (rules
    of `formats` as rules(); else the kind's SUB parent's, else `wformat`); the head's is
    `head_format`, else the formats' head rule, else `wformat`."""
    r = rules(formats, kinds, default)
    head = head_format or plain(r).get("head") or wformat

    def fmt(kind: str, layer: int = 0) -> str:
        if kind == "head":
            return head
        base = pick(r, SUB[kind], layer, wformat) if kind in SUB else wformat
        return pick(r, kind, layer, base)
    return fmt


def uniform(fmt, layers: dict) -> dict:
    """{kind: format} of a model whose layers share one layout (Qwen3, LFM2, Qwen3.5: each kind
    one format in all the layers that have it, `layers` {kind: those layers}); ValueError when a
    rule's range splits a kind."""
    out = {}
    for k, ls in layers.items():
        fs = {fmt(k, i) for i in ls}
        if len(fs) > 1:
            raise ValueError(f"weight formats: {k} in {sorted(fs)} by layer, this model's "
                             f"layers hold one format per kind")
        out[k] = fs.pop() if fs else fmt(k)
    return out
