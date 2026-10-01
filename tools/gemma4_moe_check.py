"""Gemma 4 26B-A4B's MoE block on the ISA simulator against float, layer by layer, on real
activations: moe.moe_ffn alone (one program per token and layer, the experts served into the
image's slots by the host's server, the production flags from the environment), on the inputs
the float model gives its MoE blocks over the first tokens of a text (tools/gemma4_quant_eval.py's
batched emulation, float, cached in --inputs).

Per layer, the relative error of the block's output (mean over tokens of |a - b| / |b|):
  device vs emulation   the float64 emulation of the same formats on the device's routes (its
                        weights: the int8 router's softmax over them): the device's rounding
  vs float experts      float experts and activation on the device's routes (the float
                        router's softmax over them): the formats' error
  vs float              the float block on its own routes: with the routes' flips

  python tools/gemma4_moe_check.py models/gemma-4-26B-A4B REF.npz fp4 [--layers 10,20,29]
                                   [--tokens 64] [--slots 64] [--inputs moe_inputs_64.npz]

REF.npz: ids0, the text's token ids (hf_long-style). The experts' format is the third argument;
the router and activations are int8 as on the device. 26B-A4B on omarchy (2026-10-01): int8
experts 0.016 / 0.019 / 0.015 against float experts at layers 10 / 20 / 29, fp4 0.094 / 0.122 /
0.104 (docs/gemma4.md, "26B-A4B"); about 4 GB of memory with --slots 64 (2 GiB of DRAM).
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from opentpu import language as ol  # noqa: E402
from opentpu.llm import moe as MO  # noqa: E402


@ol.jit
def moe_block(m, li):
    """The MoE block of the image's layer li on the row at m.x, its output back into m.x."""
    x = ol.load(m.x[0:1, :])
    acc = MO.moe_ffn(x, m.layer(li), m.spec.moe, m.moe_dev, m.spec.eps, residual=False,
                     y_first=True)
    ol.store(m.x[0:1, :], acc)


def inputs(spec, W, ids, path) -> dict:
    """{layer prefix: the float model's MoE input rows [T, H]}, from path when it holds them."""
    import gemma4_quant_eval as QE
    if Path(path).exists():
        return dict(np.load(path))
    caps, moe = {}, QE._moe

    def keep(spec_, W_, p, x, *a):
        caps[p] = np.array(x)
        return moe(spec_, W_, p, x, *a)
    QE._moe = keep
    try:
        QE.emulate(spec, W, ids, wformat="none", rows=[len(ids) - 1])
    finally:
        QE._moe = moe
    np.savez(path, **caps)
    return caps


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model_dir")
    ap.add_argument("ref")
    ap.add_argument("experts", choices=["int8", "fp4", "int4"])
    ap.add_argument("--layers", default="10,20,29", help="checkpoint layers, comma-separated")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--slots", type=int, default=64, help="expert slots per layer")
    ap.add_argument("--inputs", help="the float MoE inputs' cache (default moe_inputs_T.npz)")
    a = ap.parse_args(argv)
    from opentpu.isasim import board_config
    from opentpu.llm import gemma4 as G
    from opentpu.llm.qwen3 import Engine, _fake_q, _fake_w
    layers = [int(v) for v in a.layers.split(",")]
    T, EF = a.tokens, a.experts
    full = G.Spec.from_hf(a.model_dir)
    W = G.load_weights(a.model_dir)
    E, k, F, H, eps = full.experts, full.top_k, full.expert_ffn, full.hidden, full.eps
    ids = [int(t) for t in np.load(a.ref)["ids0"]][:T]
    caps = inputs(full, W, ids, a.inputs or f"moe_inputs_{T}.npz")
    spec = replace(full.truncated(layers), formats=f"experts={EF}")
    t0 = time.time()
    eng = Engine(spec, W, cap=256, cfg=board_config(DRAM_BYTES=1 << 31), rows=1,
                 wformat="int8", experts=a.slots, pipeline=False)
    io, m = eng.image.io, eng.image.descriptors(0)
    progs = [moe_block.trace(eng.cfg, 0, {"m": m, "li": j}).finish() for j in range(len(layers))]
    print(f"engine {time.time() - t0:.0f} s: experts {eng.image.efmt}, {a.slots} slots per layer, "
          f"programs {[len(p) for p in progs]} instructions", flush=True)

    def fw(w, f):
        w = np.pad(np.asarray(w, np.float32), ((0, 0), (0, -w.shape[1] % 128)))
        return w.astype(np.float64) if f == "none" else _fake_w(w, 128, f)

    def blocks(p, x, cases):
        """The block on rows x for each case (expert sets per row, logits whose softmax over a
        row's set weighs it, weight format, int8 activations), one expert at a time."""
        g2 = np.asarray(W[p + "pre_feedforward_layernorm_2.weight"], np.float64)
        pes = np.asarray(W[p + "router.per_expert_scale"], np.float32)
        xe = G._norm(x, None, eps) * g2
        prep, outs = [], []
        for sets, lg, wf, act in cases:
            fq = (lambda v: _fake_q(np.asarray(v, np.float64), 128)) if act else \
                (lambda v: np.asarray(v, np.float64))
            wts = {}
            for t, st in enumerate(sets):
                z = lg[t, st]
                w = np.exp(z - z.max())
                for e, we in zip(st, w / w.sum()):
                    wts.setdefault(e, []).append((t, we))
            prep.append((fq, fq(xe), wts, wf))
            outs.append(np.zeros((len(x), H)))
        for e in sorted(set().union(*(c[2] for c in prep))):
            gu = G._rows(W, p + "experts.gate_up_proj", [e])[0]
            dn = G._rows(W, p + "experts.down_proj", [e])[0] * pes[e]
            mats = {}
            for (fq, xq, wts, wf), out in zip(prep, outs):
                if e not in wts:
                    continue
                if wf not in mats:
                    mats[wf] = (fw(gu[:F], wf), fw(gu[F:], wf), fw(dn, wf))
                gq, uq, dq = mats[wf]
                t = np.array([i for i, _ in wts[e]])
                we = np.array([w for _, w in wts[e]])
                u = G._gelu(xq[t] @ gq.T) * (xq[t] @ uq.T)
                u = fq(np.pad(u, ((0, 0), (0, dq.shape[1] - u.shape[1]))))
                out[t] += we[:, None] * (u @ dq.T)
        return outs

    def rel(u, v):
        return float(np.mean(np.linalg.norm(u - v, axis=1) / np.linalg.norm(v, axis=1)))

    print("layer | routes = float's | device vs emulation | vs float experts | vs float")
    for j, li in enumerate(layers):
        p = f"model.layers.{li}."
        x = caps[p][:T]
        eng.server.history = []
        dev = np.zeros((T, H))
        for t in range(T):
            eng.backend.write(0, io["x"], x[t:t + 1].astype(np.float32))
            eng.backend.run([progs[j]])
            dev[t] = eng.backend.read(0, io["x"], 4 * H).view(np.float32)
        eng.server.poll()
        sets = [sorted(g - j * E for g in r) for r in eng.server.history]
        if len(sets) != T or not all(0 <= e < E for s in sets for e in s):
            raise RuntimeError(f"layer {li}: the server saw {len(sets)} requests for {T} runs")
        xn = G._norm(x, None, eps)
        r = p + "router."
        wr = np.asarray(W[r + "proj.weight"], np.float32) * (
            np.asarray(W[r + "scale"], np.float32) * np.float32(H ** -0.5))[None, :]
        lg8 = _fake_q(xn, 128) @ _fake_w(wr, 128, "int8").T
        lgf = xn @ wr.T
        fsets = [sorted(s) for s in np.argsort(-lgf, axis=1, kind="stable")[:, :k].tolist()]
        emu, fdev, flt = blocks(p, x, [(sets, lg8, EF, True), (sets, lgf, "none", False),
                                       (fsets, lgf, "none", False)])
        same = np.mean([u == v for u, v in zip(sets, fsets)])
        print(f"{li} | {same:.3f} | {rel(dev, emu):.5f} | {rel(dev, fdev):.4f} | "
              f"{rel(dev, flt):.4f}", flush=True)
    return 0


if __name__ == "__main__":          # (the image build's spawn workers import this file)
    sys.exit(main())
