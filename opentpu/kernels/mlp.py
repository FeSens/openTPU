"""SwiGLU MLP block with pre-RMSNorm and residual, sharded over slices along F."""
from .. import language as ol
from .lib import rmsnorm, silu


def _chunk(f_loc: int, D: int, q: int | None = None) -> int:
    """F-chunk size: about 8 chunks per slice, a multiple of q (default D; 2D for 4-bit W_down,
    whose column slices must start on whole D-byte chunks)."""
    q = q or D
    if f_loc % q:
        raise ValueError(f"F per slice {f_loc} is not a multiple of {q}")
    c = max(q, (f_loc // 8) // q * q)
    while f_loc % c:
        c -= q
    return c


def swiglu_down(xs, w_gate, w_up, w_down, chunk=None, act=silu):
    """This slice's output columns of W_down( act(W_gate x) * (W_up x) ), x quantized in `xs`
    (act: SiLU, or e.g. lib.gelu_tanh for Gemma's GeGLU).

    Weights are sharded by rows: slice s owns F range s of gate/up and rows (output columns)
    s of the down projection. See `mlp` for the pipelining.
    """
    S = ol.num_programs()
    f_loc = w_gate.shape[0]
    D = ol.block_size()
    C = chunk or _chunk(f_loc, D, D if w_down.wf == 0 else 2 * D)
    starts = list(range(0, f_loc, C))

    def gate_up(c0):
        return ol.dot(xs, w_gate[c0:c0 + C, :]), ol.dot(xs, w_up[c0:c0 + C, :])

    y = None
    nxt = gate_up(starts[0])
    for i, c0 in enumerate(starts):
        g, u = nxt
        if i + 1 < len(starts):
            nxt = gate_up(starts[i + 1])                # MXU streams ahead while the VPU works
        a = ol.all_gather(act(g) * u)                   # [M, S*C]: chunk c of every slice
        for t in ol.static_range(S):
            cols = w_down[:, t * f_loc + c0:t * f_loc + c0 + C]
            at = a[:, t * C:(t + 1) * C]
            if y is None:
                y = ol.dot(at, cols)
            else:
                ol.dot(at, cols, acc=y)                 # y += a_t . W_down[:, chunk of slice t]
    return y


@ol.jit
def mlp(h, gamma, w_gate, w_up, w_down, out, eps, chunk=None):
    """out = h + W_down( silu(W_gate xn) * (W_up xn) ),  xn = rmsnorm(h) * gamma.

    h: Input [M, H]; gamma: Input [H]; w_gate, w_up: Weight [F, H] and w_down: Weight [H, F],
    all sharded by rows (shard=0): slice s owns F range s of gate/up and output columns s of
    the down projection, which it stores itself (no final reduction).

    F is processed in chunks, software-pipelined: the MXU streams gate/up of chunk c+1 while
    the VPU computes a_c = silu(g_c) * u_c and the collective unit all-gathers it; then every
    slice accumulates its W_down rows against all slices' a_c. Every weight byte is streamed
    exactly once and every collective overlaps the weight stream.
    """
    sid = ol.program_id()
    x = ol.load(h)
    xs = ol.quantize(rmsnorm(x, ol.load(gamma), eps))   # stationary, reused by all gate/up MMs
    y = swiglu_down(xs, w_gate, w_up, w_down, chunk)
    h_loc = w_down.shape[0]
    mine = slice(sid * h_loc, (sid + 1) * h_loc)
    ol.store(out[:, mine], x[:, mine] + y)
