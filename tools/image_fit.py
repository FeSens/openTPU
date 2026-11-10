"""A Qwen3.5 image's DRAM per slice, by part, and the bytes a decode token reads and writes: does
the model fit a slice's DRAM (the KV cache at `cap` and the DeltaNet states included)?

    python3 tools/image_fit.py MODEL [--S 4 --mcols 8 --wg 8 ...] [--cap 4096]
        [--wformat fp4] [--formats F] [--head-format int8] [--lookup] [--embed-host]
        [--embed f32|int8 (GPT-OSS)] [--ctx N] [--json FILE]

MODEL: a short name (opentpu.llm.MODELS) or a checkpoint directory (only its config.json is
read: a layout-only image, no weights). The configuration is the target's (--target, else
OTPU_TARGET's) with the fields the options name over it.

Parts per slice (bytes, from the image's layouts and shapes):
  weights  every projection's data and block scales (in groups of Config.WG rows), by kind
           (attn, delta, mlp), the LM head, the int8 embedding table when the image holds one
  qk-rep   of delta: the q and k rows a value head repeats from its key head (Image: value head h
           takes key head h // (value / key heads)'s rows; none with Spec.qk_share or qk_once)
  consts   norms, DeltaNet A_log / dt_bias / norm, the convolution taps
  kv       the attention layers' K / V caches at `cap` tokens
  state    the DeltaNet states and convolution windows
  io       the I/O area (x, cos, sin, logits, gates, scratch), the lookup tables (RoPE, masks,
           the generate area, the embedding slot), the 4 KiB rounding of the layer blocks and
           the groups' alignment

Above AW 32 (bcu1525-wg) an image that does not fit below 4 GiB has its weights from 4 GiB on
(qwen35.Image.high): the parts are those of its low area (to low_end: I/O, KV caches, states,
tables) and its high one (from 4 GiB: weights, the resident expert pool), the bytes between
them free for the run-time areas (programs, snapshots).

A decode token at context `ctx` (estimated from the shapes, not a ledger run: tools/dataflow.py
measures one) reads every weight byte once, the KV cache of its context in whole 256-token
blocks, each DeltaNet state and window once, and writes them back, its K / V row, its logits.

GPT-OSS (model_type gpt_oss, opentpu/llm/gptoss.py; the embedding table always the image's
own, fp32, or with --embed int8 the MXU's int8 rows): the parts attn, router, experts (the
pool: every layer's slots, their tags and alignment), head, embed, consts (norms, biases,
sinks), kv (the sliding layers' 256-slot rings, the full layers' caches at `cap`), io; a
token reads every attention and router byte, its k experts a layer, the head, the rings whole,
the full layers' caches in whole blocks (K rows at head_dim 64 zero-padded to D), and the
table's row. Each part and kind also as the model's own bytes in the image's formats (scales
included): no padding (hidden 2880 -> 2944, the experts' 2880 -> 3072, 4-bit rows to whole 2 D
elements, K's 64 -> 128, the rings' slots past the window, the cache's last block past the
context), the measure's bytes (docs/bcu1525_wg.md).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opentpu import quant as Q                                          # noqa: E402
from opentpu.compiler import gall_wg, k_scale_bytes                     # noqa: E402
from opentpu.kernels import gather as GA                                # noqa: E402
from opentpu.llm import model_dir                                       # noqa: E402
from opentpu.llm.qwen35 import ATTN, LIN, Spec                          # noqa: E402
from opentpu.llm import gptoss as GO                                    # noqa: E402

MiB = 1 << 20
PEAK = 76.8e9                   # the BCU1525's 4 DDR4-2400 DIMMs (the 86% measure's peak)
BLOCK = 256                     # the attention block: decode reads whole blocks of the cache


def weight_bytes(n: int, k: int, fmt: str, D: int, R: int = 1, kt: bool = False) -> int:
    """n rows of k weights in format fmt, grouped for a machine of WG R (kt: in MM KT's pairs):
    their data and block scales."""
    return sum(Q.grp_bytes(n, k, fmt, D, R, kt))


def parts(img) -> dict:
    """{part: bytes} of one slice of image `img` (a qwen35.Image); the parts sum to its bytes
    (used_bytes: img.nbytes, split at 4 GiB its low and high areas')."""
    spec, cfg = img.spec, img.cfg
    D, H, d, K, R = cfg.D, spec.hidden, spec.head_dim, spec.conv_k, cfg.WG
    dk, dv, nl, og = spec.lin_dk, spec.lin_dv, img.nl, img.og
    out = dict.fromkeys(("attn", "delta", "qk-rep", "mlp", "head", "embed", "consts", "kv",
                         "state", "io"), 0)
    # the q and k rows a pair holds beyond its key heads' own (one pair's, then the slice's)
    rk = spec.lin_heads // spec.lin_nk
    nk_loc = nl // rk if nl % rk == 0 else nl           # key heads a slice holds
    qk_rows = 0 if img.qkonce else (2 if img.shared else 4) * dk * (nl // 2) - 2 * dk * nk_loc
    for li, (kind, g) in enumerate(img.keys[:spec.layers]):
        lay = img.layouts[g]
        mf = lay.mf
        for name in ("wg", "wu"):
            out["mlp"] += weight_bytes(img.f_loc, H, mf[name], D, R)
        out["mlp"] += len(lay.lofs[kind]["wd"]) * weight_bytes(img.h_loc, cfg.S * lay.dchunk,
                                                               mf["wd"], D, R)
        out["consts"] += 2 * 4 * H
        if kind == LIN:
            fl = mf["wh"]
            out["delta"] += weight_bytes(nl // 2 * img.RP, H, fl, D, R)
            out["delta"] += weight_bytes(2 * nl, H, fl, D, R)
            out["delta"] += weight_bytes(nl // og * H, og * dv, fl, D, R)
            out["delta"] += weight_bytes(img.QC, H, fl, D, R)       # (qkonce: the key heads')
            out["qk-rep"] += weight_bytes(qk_rows, H, fl, D, R)
            out["consts"] += 4 * (2 * nl + dv) + 4 * K * (nl // 2 * img.CP + img.QC)
            out["state"] += img.slots * (4 * nl // 2 * (K - 1) * img.CP + img.se * nl * dv * dk)
            out["state"] += 4 * (K - 1) * img.QC
            if img.grouped:                             # the per-token gates in the groups
                out["state"] += 4 * 4 * (nl // 2)
        else:
            for name, (n, k) in img.mats[ATTN].items():
                if name not in ("wg", "wu"):
                    out["attn"] += weight_bytes(n, k, mf[name], D, R)
            out["consts"] += 2 * 4 * d
            out["kv"] += img.nkv_loc * (img.cap * d + img.cap * k_scale_bytes(d, D) +
                                        d * img.cap + 4 * img.cap)
    out["head"] = weight_bytes(img.v_loc, H, img.head_format, D, R)
    lk = img.lookup
    if lk.get("own"):                               # (split: this slice's columns)
        hs = H // lk.get("split", 1)
        out["embed"] = spec.vocab * (hs + Q.gather_scale_bytes(hs, "int8", D))
    elif lk.get("head_gsc"):
        out["embed"] = spec.vocab * Q.gather_scale_bytes(H, "int8", D)
    elif "embed" in lk:
        out["embed"] = 4 * spec.vocab * H
    pool = getattr(img, "pool", None)
    if pool is not None:                            # (resident: every expert, from 4 GiB on)
        out["experts"] = pool.experts * pool.ebytes
    out["delta"] -= out["qk-rep"]
    out["io"] = used_bytes(img) - sum(out.values())
    return out


def used_bytes(img) -> int:
    """The bytes the image holds: nbytes, or split at 4 GiB (Image.high) its low area's and
    its high one's (from HIGH); with the bank-half window (Image.mix) its low area's, the
    DeltaNet mixers' (from wlo to wskip's start) and the other weights' (from wskip's end)."""
    from opentpu.llm.qwen3 import HIGH
    if getattr(img, "mix", False):
        return img.low_end + (img.wskip[0] - img.wlo) + (img.nbytes - img.wskip[1])
    return img.low_end + img.nbytes - HIGH if getattr(img, "high", False) else img.nbytes


def token_bytes(img, ctx: int, p: dict) -> dict:
    """A decode token's DRAM bytes on one slice at context ctx (estimated): {kind: (read,
    written)}."""
    spec, cfg = img.spec, img.cfg
    d, H = spec.head_dim, spec.hidden
    nkv, n_attn = img.nkv_loc, spec.kinds.count(ATTN)
    blocks = -(-(ctx + 1) // BLOCK) * BLOCK
    kv_row = d + k_scale_bytes(d, cfg.D) + d + 4         # K, its scales, V^T's column, V's scale
    w = sum(p[k] for k in ("attn", "delta", "qk-rep", "mlp", "head"))
    return {"weights": (w, 0), "kv": (n_attn * nkv * blocks * kv_row, n_attn * nkv * kv_row),
            "state": (p["state"], p["state"]),
            "io": (spec.hidden * (1 if spec.embed == "int8" else 4), 4 * img.v_loc)}


def report(img, ctx: int) -> dict:
    p = parts(img)
    t = token_bytes(img, ctx, p)
    return {"S": img.cfg.S, "cap": img.cap, "ctx": ctx, "wformat": img.wformat,
            "formats": img.formats, "head": img.head_format, "lookup": bool(img.lookup),
            "embed_host": img.embed_host, "dram": img.cfg.DRAM_BYTES, "nbytes": img.nbytes,
            "high": bool(getattr(img, "high", False)), "low_end": getattr(img, "low_end", None),
            "used": used_bytes(img), "parts": p, "token": t}


def text(r: dict) -> str:
    S, MB = r["S"], 1e6
    lines = [f"formats {r['wformat']} {r['formats'] or '-'}, head {r['head']}, KV cap "
             f"{r['cap']}, lookup {r['lookup']}, embedding on the host {r['embed_host']}",
             "", "| part | MiB a slice | MiB, all slices |", "|---|--:|--:|"]
    for k, v in r["parts"].items():
        lines.append(f"| {k} | {v / MiB:.1f} | {S * v / MiB:.1f} |")
    free = r["dram"] - r["used"]
    lines += [f"| **total** | **{r['used'] / MiB:.1f}** | {S * r['used'] / MiB:.1f} |",
              f"| free of {r['dram'] / MiB:.0f} | {free / MiB:.1f} | |"]
    if r.get("high"):
        lines += [f"| low area (I/O, KV, states, tables; below 4 GiB) | {r['low_end'] / MiB:.1f}"
                  f" | |", f"| high area (weights, from 4 GiB) | "
                  f"{(r['nbytes'] - (1 << 32)) / MiB:.1f} | |",
                  f"| image's end | {r['nbytes'] / MiB:.1f} | |"]
    lines += ["",
              f"A decode token at context {r['ctx']} (estimated):", "",
              "| kind | MB read a slice | MB written a slice | MB read, all slices |",
              "|---|--:|--:|--:|"]
    rd = wr = 0
    for k, (a, b) in r["token"].items():
        lines.append(f"| {k} | {a / MB:.2f} | {b / MB:.3f} | {S * a / MB:.1f} |")
        rd, wr = rd + a, wr + b
    lines.append(f"| **total** | **{rd / MB:.1f}** | {wr / MB:.2f} | **{S * rd / MB:.1f}** |")
    return "\n".join(lines)


def model_bytes(n: int, k: int, fmt: str, D: int) -> int:
    """n rows of k weights in format fmt as the model's own bytes: k elements a row at the
    format's bits, its Sw and each D-block's scale data (no padding)."""
    return n * ((k if fmt == "int8" else k // 2) + 4 + Q.SCALE_DATA[fmt] * -(-k // D))


def gptoss_report(img, ctx: int) -> dict:
    """A GPT-OSS image's parts (bytes of one slice; they sum to what it holds: img.nbytes, or
    with Image.high its low area and what is from HIGH on) and a decode token's bytes at
    context ctx, each as the image holds them and as the model's own bytes."""
    spec, cfg = img.spec, img.cfg
    D, R, Hp, H0, d, mo = cfg.D, cfg.WG, img.H, spec.hidden, spec.head_dim, spec.moe
    nq, nkv, wf, hf, E = spec.n_q * d, spec.n_kv * d, img.wformat, img.head_format, mo.E
    fmt, Lo, P = img.fmt, img.offload, img.pool
    p = dict.fromkeys(("attn", "router", "experts", "head", "embed", "consts", "kv", "io"), 0)
    u = dict(p)                                         # the model's own bytes
    real = {"wq": (nq, H0), "wk": (nkv, H0), "wv": (nkv, H0), "wo": (H0, nq)}
    kvt = img.dk + k_scale_bytes(img.dk, D) + d + 4     # a token's K, K scale, V^T column, V scale
    ukv = d + 4 + d + 4                                 # the model's: K and V at head_dim, scales
    for kind in spec.kinds:
        for name, (n, k) in img.mats.items():
            p["attn"] += weight_bytes(n, k, img.mf[name], D, R, img.kt[name])
            u["attn"] += model_bytes(*real[name], img.mf[name], D)
        p["router"] += weight_bytes(E, Hp, "int8", D, R, img.kt["router"])
        u["router"] += model_bytes(E, H0, "int8", D)
        p["consts"] += 4 * (2 * Hp + img.mats["wo"][0] + nq + 2 * nkv + spec.n_q + E + 1)
        u["consts"] += 4 * (3 * H0 + nq + 2 * nkv + spec.n_q + E)
        ck = img.ring if kind == GO.SLIDE else img.cap
        p["kv"] += spec.n_kv * ck * (img.dk + k_scale_bytes(img.dk, D) + img.dk + 4)
        u["kv"] += spec.n_kv * (spec.window if kind == GO.SLIDE else img.cap) * ukv
    p["consts"] += 4 * Hp
    u["consts"] += 4 * H0
    p["experts"] = P.experts * P.ebytes if P is not None else Lo.end - Lo.slots[0][0]
    gu_read = sum(weight_bytes(fmt.Fg, Hp, wf, D, R, t) for t in fmt.kts[:2])  # an expert's:
    dn_read = sum(weight_bytes(fmt.Hd, w, wf, D, R, t)                     # W_down in its parts
                  for w, t in zip(fmt.cw, fmt.kts[2:]))
    ex_read = gu_read + dn_read + 4 * (2 * fmt.Fg + fmt.Hd)
    ex_own = 3 * model_bytes(mo.ffn, H0, wf, D) + 4 * (2 * mo.ffn + H0)
    u["experts"] = spec.layers * E * ex_own
    p["head"] = weight_bytes(spec.vocab, Hp, hf, D, R, img.kt["head"])
    u["head"] = model_bytes(spec.vocab, H0, hf, D)
    if spec.embed == "int8":            # a table row: int8, its gather scales (the gathers' groups)
        row, urow = sum(GA.table_bytes(Hp, "int8", D, gall_wg(cfg))), H0 + 4 * -(-H0 // D)
    else:
        row, urow = 4 * Hp, 4 * H0
    p["embed"] = spec.vocab * row
    u["embed"] = spec.vocab * urow
    held = img.low_end + img.nbytes - GO.HIGH if img.high else img.nbytes
    p["io"] = held - sum(p.values())
    u["io"] = 0
    nf, ns = spec.kinds.count(GO.FULL), spec.kinds.count(GO.SLIDE)
    blocks = -(-(ctx + 1) // BLOCK) * BLOCK
    L, k = spec.layers, mo.k
    rd = {}
    for name, kind in (("wq", "attn q"), ("wk", "attn k"), ("wv", "attn v"), ("wo", "attn o")):
        rd[kind] = (L * weight_bytes(*img.mats[name], img.mf[name], D, R, img.kt[name]),
                    L * model_bytes(*real[name], img.mf[name], D))
    rd["router"] = (p["router"], u["router"])
    rd["norms, biases, sinks"] = (p["consts"], u["consts"])
    rd["experts gate, up"] = (L * k * gu_read, L * k * 2 * model_bytes(mo.ffn, H0, wf, D))
    rd["experts down"] = (L * k * dn_read, L * k * model_bytes(H0, mo.ffn, wf, D))
    rd["experts biases"] = (L * k * 4 * (2 * fmt.Fg + fmt.Hd), L * k * 4 * (2 * mo.ffn + H0))
    rd["head"] = (p["head"], u["head"])
    # the KV slots read (the rings whole, the full caches in whole blocks) and the model's (the
    # window, the context); K at dk with its scale against head_dim's, V^T columns at head_dim
    slots = spec.n_kv * (ns * img.ring + nf * blocks)
    used = spec.n_kv * (ns * min(ctx + 1, spec.window) + nf * (ctx + 1))
    rd["kv K"] = (slots * (img.dk + k_scale_bytes(img.dk, D)), used * (d + 4))
    rd["kv V"] = (slots * (d + 4), used * (d + 4))
    rd["embedding row"] = (row, urow)
    wr = {"kv": (L * spec.n_kv * kvt, L * spec.n_kv * ukv), "logits": (4 * spec.vocab,) * 2}
    return {"model": "gpt_oss", "S": cfg.S, "cap": img.cap, "ctx": ctx, "wformat": wf,
            "formats": img.formats, "head": hf, "embed": spec.embed, "dram": cfg.DRAM_BYTES, "nbytes": held,
            "parts": p, "own": u, "read": rd, "written": wr, "expert": (ex_read, ex_own),
            "pool": (p["experts"], u["experts"], spec.layers * E)}


def gptoss_text(r: dict) -> str:
    MB = 1e6
    lines = [f"GPT-OSS: formats {r['wformat']} {r['formats'] or '-'}, head {r['head']}, "
             f"router int8, the embedding table {r['embed']} (the image's), KV cap {r['cap']}",
             "",
             "| part | MiB | the model's own MiB |", "|---|--:|--:|"]
    for k, v in r["parts"].items():
        lines.append(f"| {k} | {v / MiB:.1f} | {r['own'][k] / MiB:.1f} |")
    free = r["dram"] - r["nbytes"]
    lines += [f"| **total** | **{r['nbytes'] / MiB:.1f}** | "
              f"{sum(r['own'].values()) / MiB:.1f} |",
              f"| free of {r['dram'] / MiB:.0f} | {free / MiB:.1f} | |", "",
              f"Expert pool: {r['pool'][2]} experts in {r['pool'][0] / MiB:.1f} MiB (the model's "
              f"{r['pool'][1] / MiB:.1f}); an expert reads {r['expert'][0] / MB:.3f} MB (its own "
              f"{r['expert'][1] / MB:.3f}).", "",
              f"A decode token at context {r['ctx']} (estimated): its reads as the image holds "
              f"them and the model's useful bytes (the checkpoint's shapes in the image's "
              f"formats, the window's and the context's KV):", "",
              "| kind | MB read | useful MB | padding MB | padding % |", "|---|--:|--:|--:|--:|"]
    ta = tb = 0
    for kind, (a, b) in r["read"].items():
        lines.append(f"| {kind} | {a / MB:.2f} | {b / MB:.2f} | {(a - b) / MB:.2f} | "
                     f"{100 * (a - b) / a:.1f} |")
        ta, tb = ta + a, tb + b
    lines.append(f"| **total** | **{ta / MB:.1f}** | **{tb / MB:.1f}** | "
                 f"**{(ta - tb) / MB:.1f}** | **{100 * (ta - tb) / ta:.1f}** |")
    wa, wb = (sum(v[i] for v in r["written"].values()) for i in (0, 1))
    bw = 0.86 * PEAK
    lines += ["", f"It writes {wa / MB:.3f} MB ({wb / MB:.3f} useful). At 86% of {PEAK / 1e9:.1f} "
              f"GB/s on the useful bytes ({bw / (tb + wb):.1f} tok/s) the DRAM moves "
              f"{(ta + wa) / (tb + wb) * bw / 1e9:.1f} GB/s, "
              f"{100 * (ta + wa) / (tb + wb) * 0.86:.1f}% of the peak."]
    return "\n".join(lines)


def config(a):
    from opentpu import target as T
    cfg = T.config(a.target) if a.target else T.config()
    kw = {k: v for k, v in (("S", a.S), ("MCOLS", a.mcols), ("ACT_BLOCKS", a.act_blocks),
                            ("DRAM_BYTES", a.dram), ("TMEM_WORDS", a.tmem),
                            ("IMEM_WORDS", a.imem), ("WG", a.wg)) if v is not None}
    return dataclasses.replace(cfg, **kw)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model")
    ap.add_argument("--target")
    ap.add_argument("--S", type=int)
    ap.add_argument("--mcols", type=int)
    ap.add_argument("--act-blocks", type=int)
    ap.add_argument("--dram", type=int)
    ap.add_argument("--tmem", type=int)
    ap.add_argument("--imem", type=int)
    ap.add_argument("--wg", type=int)
    ap.add_argument("--cap", type=int, default=4096)
    ap.add_argument("--ctx", type=int)
    ap.add_argument("--wformat", default="fp4")
    ap.add_argument("--formats")
    ap.add_argument("--head-format", default="int8")
    ap.add_argument("--lookup", action="store_true")
    ap.add_argument("--embed-host", action="store_true")
    ap.add_argument("--embed", choices=("f32", "int8"))
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    md = model_dir(a.model)
    cfg = config(a)
    # measured against the slice's DRAM, built in a large one (an image that does not fit is
    # reported, not refused)
    big = dataclasses.replace(cfg, DRAM_BYTES=1 << 40)
    if json.loads((Path(md) / "config.json").read_text()).get("model_type") == "gpt_oss":
        gs = GO.Spec.from_hf(md)
        gs = dataclasses.replace(gs, embed=a.embed) if a.embed else gs
        img = gs.image(big, a.cap, wformat=a.wformat, formats=a.formats,
                       head_format=a.head_format, lookup=True)
        img.cfg = cfg
        r = gptoss_report(img, a.ctx if a.ctx is not None else a.cap - 1)
        print(gptoss_text(r))
        if a.json:
            Path(a.json).write_text(json.dumps(r, indent=1))
        return 0 if r["nbytes"] <= cfg.DRAM_BYTES else 1
    spec = Spec.from_hf(md)
    img = spec.image(big, a.cap, wformat=a.wformat, formats=a.formats,
                     head_format=a.head_format, lookup=a.lookup,
                     embed_host=True if a.embed_host else None)
    img.cfg = cfg
    r = report(img, a.ctx if a.ctx is not None else a.cap - 1)
    print(text(r))
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=1))
    return 0 if r["nbytes"] <= cfg.DRAM_BYTES else 1


if __name__ == "__main__":
    sys.exit(main())
