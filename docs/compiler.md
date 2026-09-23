# The `ol` language and how it lowers to data movement

`ol` (`opentpu/language.py`) is a Triton/Gluon-style kernel language. A kernel is a Python
function under `@ol.jit`. It is traced once per slice, and each call in it lowers to one or a
few ISA instructions. Nothing is scheduled behind the programmer's back: where a tensor lives
is its layout, and changing layout is an instruction.

## Layouts

| Layout | Where | Created by | Consumed by |
|---|---|---|---|
| `Tensor` | DRAM fp32, row-major with a row stride | runtime `Input`/`Output` | `ol.load`, `ol.store` |
| `QTensor` | DRAM int8 rows plus fp32 block scales | runtime `Weight`, KV cache views | the streamed operand of `ol.dot` |
| `Tile` | TMEM fp32, base + shape + row stride | load, VPU ops, dot | VPU ops, quantize, store |
| `Stationary` | ACT RAM int8 rows plus scales | `ol.quantize` | the stationary operand of `ol.dot` |
| `Bcast` | a view of a 1-D tile as a column or row | `v[:, None]`, `v[None, :]` | VPU B-operand modes |

Layout transitions are the data movement:

| Call | Instruction | Movement |
|---|---|---|
| `ol.load(t)` | `LD` | DRAM to TMEM, burst |
| `ol.store(t, x)` | `ST` | TMEM to DRAM, burst |
| `ol.quantize(x)` | `QACT` | TMEM fp32 rows to ACT RAM int8 |
| `ol.dot(a, w, acc, out, rowmax, acc_scale)` | `MM` (plus `QACT` if `a` is a tile) | streams `w` from DRAM once, result to TMEM |
| `x + y`, `ol.exp2`, `ol.max`, ... | `VOP` | TMEM to TMEM |
| `ol.all_gather(x)` | `GATHER` | TMEM of every slice to TMEM of every slice |
| `ol.all_reduce(x)` | `GATHER` + `VOP` adds | every slice gets the sum of all slices' `x` |
| `ol.kv_append(kv, h, pos, k, v)` | two `QST` | TMEM fp32 to DRAM int8 in the cache layouts |

`dot(a[M, K], w[N, K])` computes `a @ w.T`. The stationary side has at most 8 rows, so a
decode token, a GQA query group or a softmax probability block fits in one pass. Each weight
byte is read from DRAM exactly once per `dot`. The optional arguments map to MM epilogue flags:

| Argument | Flag | Effect |
|---|---|---|
| `out=t` | | write the result into an existing tile, e.g. a double buffer |
| `rowmax=True` | `RMAX` | also write the row maxima; read them as `result.rowmax` |
| `acc=t, acc_scale=alpha` | `ACC` + `ASCALE` | `t[i, j] = t[i, j] * alpha[i] + (a @ w.T)[i, j]` in the drain |

`ol.max(dot(...), axis=1)` on an unnamed result is rewritten to `rowmax` automatically.

## Broadcasting

VPU ops take a second operand in one of four modes, and the compiler picks the mode from the
operand's shape:

| Expression | Mode |
|---|---|
| `a * b`, same shape | FULL |
| `a * v[:, None]` | ROW: one value per row |
| `a * v[None, :]` | COL: one vector shared by all rows |
| `a * 0.5` | SCALAR: an immediate in the instruction |

## Loops

- `ol.static_range(n)` is a plain Python range, so the body is unrolled.
- `ol.range(n)` emits a hardware `LOOP`. The body is traced once, so values that change from
  one iteration to the next must be written with `tile.set(value)`.

Addresses inside a hardware loop are affine in the induction variables. `i * block` is an
`Affine` value, and slicing a descriptor with it (`K[i*block : i*block + n, :]`) produces an
address with a constant part and loop coefficients. The compiler allocates one address
register per distinct coefficient pattern, steps it by `ADDI` at the end of the body, resets it
after the loop and frees it when every loop that uses it has ended. Instructions add the
register to their base field, which is the ISA's `w1 += R[ra]` rule.

## The flash-attention loop, source to instructions

`opentpu/kernels/attention.py` is an online-softmax (flash) attention, software-pipelined in the
style of FlashAttention-3: while the VPU and quantizer finish block b, the MXU is already
streaming q.K^T of block b+1 into the other score buffer. The loop body covers two blocks, so
every temporary is double-buffered.

```python
def scores(t0, n, out):
    ol.dot(qs, K[t0:t0 + n, :], out=out, rowmax=True)   # s and max(s, axis=1) in one MM

def finish(s, t0, n):
    vs = ol.load(VS[t0:t0 + n])
    m_new = ol.maximum(m, s.rowmax)
    p = ol.exp2(s - m_new[:, None])                      # EXP2SUB
    alpha = ol.exp2(m - m_new)
    pq = ol.quantize(p * vs[None, :])                    # QACT CSCALE folds V's token scales
    ol.dot(pq, VT[:, t0:t0 + n], acc=acc, acc_scale=alpha)   # acc = acc*alpha + P.V
    l.set(l * alpha + ol.sum(p, axis=1))
    m.set(m_new)

scores(0, block, sA)
for i in ol.range(pairs):
    t0 = i * 2 * block
    scores(t0 + block, block, sB);      finish(sA, t0, block)
    scores(t0 + 2 * block, block, sA);  finish(sB, t0 + block, block)
```

For 4 query heads, d=32, 32-token blocks and T=160, the loop body is 22 instructions plus
3 address steps, as printed by `opentpu.isa.disassemble`:

```
 9: MM     +rowmax         ; prologue: s_A = q.K^T of block 0
10: LOOP   x2
11: MM     R1,R2 +rowmax   ; s_B = q.K^T of block 2i+1   (streams while 12-21 run)
12: LD     R2              ; V scales of block 2i
13: VOP    max             ; m_new = max(m, rowmax_A)
14: VOP    exp2sub         ; p = exp2(s_A - m_new[:, None])
15: VOP    exp2sub         ; alpha = exp2(m - m_new)
16: QACT   CSCALE          ; P * vs -> ACT RAM
17: MM     R3 ACC ASCALE   ; acc = acc*alpha + P.V
18-21: VOP mul rsum add copy ; l and m updates, off the MXU's critical path
22: MM     R1,R2 +rowmax   ; s_A = q.K^T of block 2i+2
23-32:                     ; finish(s_B), same as 12-21
33-35: ADDI R1, R2, R3     ; step the address registers
```

The MXU never waits for the VPU on the softmax: the scoreboard lets MM 22 start streaming as
soon as its DRAM operands are ready, and the rescale `acc * alpha` happens in the MXU drain.
A tail block that is not a multiple of `block` is traced after the loop, with P padded to a
multiple of D with zeros.

## Fusion peepholes

All peepholes fire only on unnamed temporaries (checked with the Python reference count), so a
named intermediate is never rewritten, and a dead-tile check catches any later misuse.

| Source | Becomes |
|---|---|
| `ol.exp2(a - b)` | one `VOP EXP2SUB` |
| `ol.quantize(x * v[None, :])` | `QACT CSCALE` (column scale applied while quantizing) |
| `ol.quantize((x * r[:, None]) * v[None, :])` | `QACT CSCALE RSCALE` |
| `ol.sum(x * x, axis=1)` | `VOP RSSQ` (used by rmsnorm) |
| `ol.max(ol.dot(...), axis=1)` | `MM RMAX` |
| `t.set(expr)` | the producer of `expr` writes straight into `t` |

## Memory allocation

- **TMEM** uses liveness: each allocation is a buffer object, freed when the last tile viewing it
  is garbage-collected. New buffers are placed next-fit from a moving cursor, which spreads
  consecutive temporaries over different addresses so the scoreboard sees no false conflicts.
- **ACT RAM** blocks are freed when their `Stationary` dies and are reused round-robin, so a
  stationary operand that is still live is never overwritten.

## Bank-aware strides

TMEM has LANES banks selected by the low address bits. The MXU writes the M results of one
streamed row in a single cycle, to addresses `out + j * row_stride`. The compiler therefore
gives MXU outputs and constructed 2-D tiles an odd row stride, so those M addresses fall in
distinct banks. The RTL stops with a fatal error on any bank conflict, so a layout mistake
cannot pass silently.

## Multi-slice programs

`ol.program_id()` and `ol.num_programs()` are compile-time constants inside each slice's trace.
Kernels use them for ownership, as in `kv.owned_heads(n)` or `if ol.program_id() == 0`. The
runtime shards `Weight(array, shard=0)` by rows across slices, and `ol.all_gather` concatenates
the slices' tiles along the last axis.

The MLP (`opentpu/kernels/mlp.py`) splits F into about eight chunks per slice. Gate and up
for chunk c+1 stream while the VPU computes `silu(g) * u` for chunk c. The chunk is all-gathered,
and each slice multiplies it by its own rows of `W_down` (`Weight(w_down, 0)`), accumulating
`y` for the H columns it owns. No reduction across slices is needed, and every weight byte is
streamed exactly once.

## Profiling

`opentpu/profile.py` runs a kernel on the RTL with tracing and returns per-instruction dispatch,
ready, start and end cycles, unit busy time, DRAM port counters and TMEM arbitration losses.
Lens (`python -m opentpu.lens`) records these into profile files and explores them; see docs/lens.md.

## Errors the compiler reports

- A `dot` whose K is not a multiple of D, or whose shapes or accumulator do not match.
- A stationary operand that does not fit ACT RAM, or that was overwritten before a later use.
- Running out of TMEM or address registers, or nesting loops deeper than 4.
- Reading a tile that `.set()` has already moved into another tile.
- Shape mismatches in stores, `set` and broadcasts, and unsupported strided views.
- A KV head accessed from a slice that does not own it.
