"""SwiGLU MLP block with pre-RMSNorm and residual, sharded over slices along F."""
from dataclasses import replace

from .. import language as ol
from .lib import rmsnorm, silu


def _chunk(f_loc: int, D: int, q: int | None = None) -> int:
    """F-chunk size: about 8 chunks per slice, a multiple of q (default D; 2D for 4-bit W_down,
    whose column slices must start on whole D-byte chunks), and at least 2D where F allows:
    each chunk's down MM is a QACT of its own (one Sa per MM row), and an MM of 2 blocks or
    more keeps the MX MXU at full rate with MCOLS/2 epilogue lanes (rows / E <= KB; PAIR's
    rows are MCOLS/2: KB/2), docs/mx_mxu.md 4. The activations' quantization follows the
    chunks: emulations take them from down_chunk."""
    q = q or D
    if f_loc % q:
        raise ValueError(f"F per slice {f_loc} is not a multiple of {q}")
    lo = min(f_loc, -(-2 * D // q) * q)
    c = max(lo, (f_loc // 8) // q * q)
    while f_loc % c and c > lo:
        c -= q
    while f_loc % c:
        c += q
    return c


def down_chunk(f_loc: int, D: int, fmt: str) -> int:
    """The F-chunk of a W_down in format fmt (_chunk): the columns each down MM's activations
    are quantized over (one QACT)."""
    return _chunk(f_loc, D, D if fmt == "int8" else 2 * D)


def f_chunk(w_gate, w_down, chunk=None) -> int:
    """swiglu_down's F-chunk (its each(i) calls: w_gate's rows / it)."""
    D = ol.block_size()
    return chunk or (w_down.pw // w_down.gs if w_down.parts is not None else
                     _chunk(w_gate.shape[0], D, D if w_down.wf == 0 else 2 * D))


def swiglu_down(xs, w_gate, w_up, w_down, chunk=None, act=silu, loop=False, each=None,
                glu=None, out=None, rscale=None):
    """This slice's output columns of W_down( act(W_gate x) * (W_up x) ), x quantized in `xs`
    (act: SiLU, or e.g. lib.gelu_tanh for Gemma's GeGLU); glu(g, u, c0): a chunk's
    activations from its gate and up outputs (its columns c0 .. of the slice's F) instead of
    act(g) * u (GPT-OSS's experts: moe.oai_glu).

    Weights are sharded by rows: slice s owns F range s of gate/up and rows (output columns)
    s of the down projection. See `mlp` for the pipelining. `loop`: the chunks run as a
    hardware loop (_swiglu_down_loop), the same operations in the same order in fewer
    instructions (a model whose program would not fit IMEM with every chunk unrolled).
    `each(i)`: called as chunk i's work begins, once the next chunk's gate and up MMs are
    issued (instructions of the caller's that wait while the MXU works: moe_prologue_rows'
    need lines), not with `loop`. `out`: a [M, N] tile the first down MM overwrites (ol.dot's
    out=) and the result, instead of a new one (not with `loop`). `rscale` [M]: each row's
    activations times its factor in their QACTs (QACT RSCALE: moe.EM_GSCALE's router weights;
    not with `loop`).

    A W_down whose columns are in the all-gather's order (w_down.gs = S, quant.gather_order:
    chunk c of every slice's F part, stored in parts of S C) takes chunk c in one MM over the
    gathered [M, S C], its activations one QACT; else S MMs of C columns, a QACT each.

    W_gate's rows need not be whole chunks (one slice: moe.ExpertFormat tight, GPT-OSS's 2880
    in chunks of 256): the last chunk's gate and up MMs take the rows left, and its activations
    are zero-filled to the width of W_down's last part (its whole D-blocks: 128 columns), which
    changes no QACT range's amax.
    """
    S = ol.num_programs()
    f_loc = w_gate.shape[0]
    if w_down.gs not in (1, S):
        raise ol.CompileError(f"W_down in the gather's order of {w_down.gs} slices, on {S}")
    C = f_chunk(w_gate, w_down, chunk)
    short = f_loc % C
    if short and (S > 1 or w_down.gs > 1 or loop):
        raise ol.CompileError(f"swiglu_down: {f_loc} rows in chunks of {C} on one slice only, "
                              f"unrolled")
    if loop and f_loc // C >= 6:                # two iterations at least
        if each is not None or glu is not None or out is not None or rscale is not None:
            raise ol.CompileError("swiglu_down: each=, glu=, out= or rscale= with the chunks "
                                  "as a hardware loop")
        return _swiglu_down_loop(xs, w_gate, w_up, w_down, C, act)
    starts = list(range(0, f_loc, C))

    def gate_up(c0):
        n = min(C, f_loc - c0)
        return ol.dot(xs, w_gate[c0:c0 + n, :]), ol.dot(xs, w_up[c0:c0 + n, :])

    y = None
    nxt = gate_up(starts[0])
    for i, c0 in enumerate(starts):
        g, u = nxt
        if i + 1 < len(starts):
            nxt = gate_up(starts[i + 1])                # MXU streams ahead while the VPU works
        if each is not None:
            each(i)
        if c0 + C > f_loc:                              # the short chunk, to its part's
            a = ol.zeros((g.rows, w_down.shape[1] - c0))  # width
            a[:, 0:g.cols].set(act(g) * u if glu is None else glu(g, u, c0))
        else:
            a = ol.all_gather(act(g) * u if glu is None else glu(g, u, c0))   # [M, S*C]: chunk
                                                                              # c of every slice
        if w_down.gs > 1:                               # y += a . W_down's part i
            cols = w_down[:, S * c0:S * (c0 + C)]
            if rscale is not None:
                a = ol.quantize(a, rscale=rscale)
            if y is None:
                y = ol.dot(a, cols, out=out)
            else:
                ol.dot(a, cols, acc=y)
            continue
        cw = a.cols // S
        for t in ol.static_range(S):
            cols = w_down[:, t * f_loc + c0:t * f_loc + c0 + cw]
            at = a[:, t * cw:(t + 1) * cw]
            if rscale is not None:
                at = ol.quantize(at, rscale=rscale)
            if y is None:
                y = ol.dot(at, cols, out=out)
            else:
                ol.dot(at, cols, acc=y)                 # y += a_t . W_down[:, chunk of slice t]
    return y


def _down_cols(w_down, j, C: int):
    """Columns j C .. (j + 1) C of W_down (j may be a loop expression): a part of a matrix
    stored in column parts of C, equally spaced, or a column slice."""
    if w_down.parts is None:
        return w_down[:, j * C:(j + 1) * C]
    p = w_down.parts
    if w_down.pw != C:
        raise ol.CompileError(f"W_down is stored in parts of {w_down.pw}, not {C}")
    dd, ds = (p[1].data - p[0].data).static(), (p[1].scale - p[0].scale).static()
    if any((q.data - p[0].data).static() != i * dd or (q.scale - p[0].scale).static() != i * ds
           for i, q in enumerate(p)):
        raise ol.CompileError("W_down's parts are not equally spaced")
    return replace(p[0], data=p[0].data + j * dd, scale=p[0].scale + j * ds)


def _swiglu_down_loop(xs, w_gate, w_up, w_down, C: int, act=silu):
    """swiglu_down with its chunks in a hardware loop, software-pipelined as the unrolled
    chunks are (gate / up of chunk k + 1 stream while the VPU computes a_k): two buffers A, B
    hold gate / up of alternate chunks, and each iteration runs two chunks, 2 j + 1 from B and
    2 j + 2 from A. Chunk 0 comes before the loop (it starts y), the last one or two after it.
    Per chunk the operations and their order are the unrolled ones', so y is bit-identical."""
    S = ol.num_programs()
    f_loc = w_gate.shape[0]
    n = f_loc // C

    def gate_up(k, buf):
        c0 = k * C
        ol.dot(xs, w_gate[c0:c0 + C, :], out=buf[0])
        ol.dot(xs, w_up[c0:c0 + C, :], out=buf[1])

    def down(k, buf, y):
        a = ol.all_gather(act(buf[0]) * buf[1])         # [M, S*C]: chunk k of every slice
        if w_down.gs > 1:                               # W_down in the gather's order
            if y is None:
                return ol.dot(a, _down_cols(w_down, k, S * C))
            ol.dot(a, _down_cols(w_down, k, S * C), acc=y)
            return y
        for t in ol.static_range(S):
            at = a[:, t * C:(t + 1) * C]
            if y is None:
                y = ol.dot(at, _down_cols(w_down, t * n + k, C))
            else:
                ol.dot(at, _down_cols(w_down, t * n + k, C), acc=y)
        return y

    A = (ol.dot(xs, w_gate[0:C, :]), ol.dot(xs, w_up[0:C, :]))
    B = (ol.empty(A[0].shape), ol.empty(A[1].shape))
    gate_up(1, B)
    y = down(0, A, None)
    reps = (n - 2) // 2                     # the loop's last gate / up is chunk 2 reps + 1
    for j in ol.range(reps):
        gate_up(2 * j + 2, A)
        down(2 * j + 1, B, y)
        gate_up(2 * j + 3, B)
        down(2 * j + 2, A, y)
    k = 2 * reps + 1                        # in B
    if k + 1 < n:
        gate_up(k + 1, A)
    down(k, B, y)
    if k + 1 < n:
        down(k + 1, A, y)
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
