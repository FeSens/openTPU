"""Host-side weight layouts matching the kernels' slice ownership rules."""
import numpy as np


def head_parallel_attention_weights(wq, wk, wv, wo, n_q_heads, n_kv_heads, d, S):
    """Reorder attention weights so that the contiguous row shard of slice s holds the heads
    that slice owns (KV heads s, s+S, ... and their query groups), and the columns of wo follow
    the slice-major order in which `attention_layer` all-gathers the head outputs. Fewer KV
    heads than slices (S a multiple of them): each KV head's rows on S / n_kv_heads slices, the
    slices of its query group (KVDesc.kv_heads: S heads, slice s's KV head s; its K / V
    computed and cached on every one of them)."""
    if n_kv_heads % S and S % n_kv_heads == 0:
        r = S // n_kv_heads
        wk, wv = (np.repeat(w.reshape(n_kv_heads, d, -1), r, axis=0).reshape(S * d, -1)
                  for w in (wk, wv))
        n_kv_heads = S
    G = n_q_heads // n_kv_heads
    if n_kv_heads % S:
        raise ValueError("n_kv_heads must be a multiple of the slice count, or divide it")
    kv_order = [h for s in range(S) for h in range(s, n_kv_heads, S)]
    q_order = [h * G + g for h in kv_order for g in range(G)]
    qrows = np.concatenate([np.arange(h * d, (h + 1) * d) for h in q_order])
    kvrows = np.concatenate([np.arange(h * d, (h + 1) * d) for h in kv_order])
    return wq[qrows], wk[kvrows], wv[kvrows], wo[:, qrows]
