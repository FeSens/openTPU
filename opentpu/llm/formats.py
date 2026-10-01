"""Weight formats per kind and layer: the rules every model's image reads (Gemma 4's first;
docs/gemma4_e4b.md). A model's weights are in its image's `wformat`, except where a formats
string says otherwise:

    "kind=fmt,kind@a-b=fmt,..."     fmt: int8, int4 or fp4; a..b: checkpoint layers (a: one)

A rule with a range wins over one without, and among equals the first one in the string. Each
model names the kinds it has (Gemma 4: attn, mlp, gateup, down, ple, head, experts); "head"
takes no range. The string comes from the caller, else the OTPU_FORMATS environment variable,
else the model's own default (Spec.formats).
"""
from __future__ import annotations

import os

FORMATS = ("int8", "int4", "fp4")


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
