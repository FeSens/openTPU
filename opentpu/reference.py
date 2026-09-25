"""float64 numpy references for the kernels (the math, not the hardware numerics)."""
import numpy as np


def rmsnorm(x, gamma, eps):
    x = np.asarray(x, np.float64)
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * gamma


def silu(x):
    return x / (1.0 + np.exp(-x))


def mlp(h, gamma, w_gate, w_up, w_down, eps):
    xn = rmsnorm(h, gamma, eps)
    a = silu(xn @ w_gate.T) * (xn @ w_up.T)
    return h + a @ w_down.T


def attention(q, k, v, n_kv_heads):
    """q [Hq, d]; k, v [Hkv, T, d] -> [Hq, d]."""
    Hq, d = q.shape
    G = Hq // n_kv_heads
    out = np.zeros((Hq, d))
    for h in range(Hq):
        kh, vh = k[h // G].astype(np.float64), v[h // G].astype(np.float64)
        s = kh @ q[h] / np.sqrt(d)
        p = np.exp(s - s.max())
        out[h] = (p / p.sum()) @ vh
    return out


def rope(x, cos, sin):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def attention_layer(h, gamma, wq, wk, wv, wo, cos, sin, k_cache, v_cache, n_q_heads,
                    n_kv_heads, pos, eps):
    """One decode step. k_cache/v_cache [Hkv, >=pos, d] hold positions < pos."""
    d = k_cache.shape[-1]
    xn = rmsnorm(h, gamma, eps)[0]
    q = (wq @ xn).reshape(n_q_heads, d)
    k = (wk @ xn).reshape(n_kv_heads, d)
    v = (wv @ xn).reshape(n_kv_heads, d)
    q, k = rope(q, cos, sin), rope(k, cos, sin)
    K = np.concatenate([k_cache[:, :pos], k[:, None]], axis=1)
    V = np.concatenate([v_cache[:, :pos], v[:, None]], axis=1)
    o = attention(q, K, V, n_kv_heads)
    return h + (wo @ o.reshape(-1))[None, :], k, v


def gated_deltanet_step(S, q, k, v, a, b, A_log, dt_bias, eps=1e-6):
    """One token of the Gated DeltaNet recurrence (as HF's torch_recurrent_gated_delta_rule
    with the q/k L2 norm in the kernel). S: [H, dk, dv]; returns (new S, o [H, dv])."""
    S, q, k, v = (np.asarray(t, np.float64) for t in (S, q, k, v))
    q = q / np.sqrt((q * q).sum(-1, keepdims=True) + eps) * q.shape[-1] ** -0.5
    k = k / np.sqrt((k * k).sum(-1, keepdims=True) + eps)
    a = np.asarray(a, np.float64) + dt_bias
    g = -np.exp(np.asarray(A_log, np.float64)) * np.logaddexp(0.0, a)
    beta = 1.0 / (1.0 + np.exp(-np.asarray(b, np.float64)))
    S = S * np.exp(g)[:, None, None]
    kv = np.einsum("hkv,hk->hv", S, k)
    S = S + np.einsum("hk,hv->hkv", k, (v - kv) * beta[:, None])
    return S, np.einsum("hkv,hk->hv", S, q)
