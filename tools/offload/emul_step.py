"""LFM2-MoE's logits at one decode step in lfm2.emulated_logits' float64 emulation (openTPU's
quantization points, none of its rounding) of the card run's formats: fp4 body, int8 head and
router, the embedding rows gathered from the int8 head. The dense weights are fake-quantized
once and kept; each expert's are fake-quantized when used and dropped (the 8B's experts in
float64 would not fit). Separates fp4 quantization from a kernel bug at a card run's first
different token.

    python3 tools/offload/emul_step.py MODEL HF.json CARD.json [DIAG.json] > out.json
"""
import json, math, sys, time
import numpy as np
from opentpu.llm import load_spec
from opentpu.llm import moe as MO
from opentpu.llm.lfm2 import ATTN, CONV, _norm
from opentpu.llm.qwen3 import LazyWeights, _fake_q, _fake_w, _pv, _v_parts, rope_tables

model, hf, cardf = sys.argv[1:4]
spec = load_spec(model)
W = LazyWeights(model)
ref, card = json.load(open(hf)), json.load(open(cardf))
k = card["first_diff"]
tokens = list(ref["ids"]) + list(ref["tokens"][:k])
D, wformat, hfmt = 128, "fp4", "int8"
d, G, H, K = spec.head_dim, spec.n_q // spec.n_kv, spec.hidden, spec.conv_k
head = "model.embed_tokens.weight" if spec.tied else "lm_head.weight"
Wq = {}


def w(n):
    if ".experts." in n:
        return _fake_w(W[n], D, wformat)
    if n not in Wq:
        Wq[n] = _fake_w(W[n], D, hfmt if n == head else wformat)
    return Wq[n]


def norm(v, g):
    return _norm(v, g, spec.eps)


Kc = {i: [] for i, kd in enumerate(spec.kinds) if kd == ATTN}
Vc = {i: [] for i in Kc}
state = {i: [np.zeros(H)] * (K - 1) for i, kd in enumerate(spec.kinds) if kd == CONV}
routes = []
t0 = time.time()
for pos, tk in enumerate(tokens):
    x = w(head)[tk].copy() if spec.tied else np.asarray(W[head][tk], np.float64)
    c, s = rope_tables(spec, pos)

    def rot(v):
        v1, v2 = v[..., :d // 2], v[..., d // 2:]
        return np.concatenate([v1 * c - v2 * s, v2 * c + v1 * s], -1)

    for i, kind in enumerate(spec.kinds):
        p = f"model.layers.{i}."
        h = _fake_q(norm(x, W[p + "operator_norm.weight"]), D)
        if kind == CONV:
            B, C, xx = np.split(w(p + "conv.in_proj.weight") @ h, 3)
            win = state[i] + [B * xx]
            state[i] = win[1:]
            wc = W[p + "conv.conv.weight"][:, 0, :]
            y = C * sum(win[j] * wc[:, j] for j in range(K))
            x = x + w(p + "conv.out_proj.weight") @ _fake_q(y, D)
        else:
            a = p + "self_attn."
            q = (w(a + "q_proj.weight") @ h).reshape(spec.n_q, d)
            kk = (w(a + "k_proj.weight") @ h).reshape(spec.n_kv, d)
            v = (w(a + "v_proj.weight") @ h).reshape(spec.n_kv, d)
            q = rot(norm(q, W[a + "q_layernorm.weight"]))
            kk = rot(norm(kk, W[a + "k_layernorm.weight"]))
            Kc[i].append(_fake_q(kk, min(d, D)))
            Vc[i].append(_v_parts(v))
            Kh = np.stack(Kc[i], 1)
            Vq, Vs = (np.stack(z, 1) for z in zip(*Vc[i]))
            o = np.zeros((spec.n_q, d))
            for hq in range(spec.n_q):
                sc = Kh[hq // G] @ _fake_q(q[hq] / math.sqrt(d), min(d, D))
                pp = np.exp(sc - sc.max())
                o[hq] = _pv(pp, Vq[hq // G], Vs[hq // G], D) / pp.sum()
            x = x + w(a + "out_proj.weight") @ _fake_q(o.reshape(-1), D)
        h = _fake_q(norm(x, W[p + "ffn_norm.weight"]), D)
        if spec.is_moe(i):
            f = p + "feed_forward."
            if f + "gate.weight" not in Wq:             # the router is int8 in every format
                Wq[f + "gate.weight"] = _fake_w(W[f + "gate.weight"], D, "int8")
            lg = Wq[f + "gate.weight"] @ h
            ids, wts = MO.route(lg, W[f + "expert_bias"], spec.moe)
            sel = np.sort(1 / (1 + np.exp(-lg)) + W[f + "expert_bias"])[::-1]
            km = spec.moe.k
            routes.append((pos, i, [int(e) for e in ids], float(sel[km - 1] - sel[km])))
            for e, we in zip(ids, wts):
                ep = f"{f}experts.{e}."
                g, u = w(ep + "w1.weight") @ h, w(ep + "w3.weight") @ h
                x = x + we * (w(ep + "w2.weight") @ _fake_q((g / (1 + np.exp(-g))) * u, D))
            continue
        g = w(p + "feed_forward.w1.weight") @ h
        u = w(p + "feed_forward.w3.weight") @ h
        x = x + w(p + "feed_forward.w2.weight") @ _fake_q((g / (1 + np.exp(-g))) * u, D)
    print(f"pos {pos} {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
lo = w(head) @ _fake_q(norm(x, W["model.embedding_norm.weight"]), D)
top = np.argsort(-lo)[:8]
out = dict(step=k, emulated_top=[[int(i), round(float(lo[i]), 4)] for i in top],
           hf_token=ref["tokens"][k], hf_top=ref.get("top", [None] * (k + 1))[k],
           near_routes=[r for r in routes if r[3] < 1e-3],
           seconds=round(time.time() - t0))
if len(sys.argv) > 4:                       # a --host-loop run's result: the device's top 8
    dg = json.load(open(sys.argv[4]))
    assert dg["tokens"][:k] == ref["tokens"][:k]
    out["card_top"] = dg["at_first_diff"]["card_top"]
for i in (278, 358, 600):
    out[f"emulated_{i}"] = round(float(lo[i]), 4)
print(json.dumps(out))
