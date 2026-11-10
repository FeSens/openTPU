# The `ol` language and how it lowers to data movement

`ol` (`opentpu/language.py`) is a Triton/Gluon-style kernel language. A kernel is a Python
function under `@ol.jit`. It is traced once per slice, and each call in it lowers to one or a
few ISA instructions. Nothing is scheduled behind the programmer's back: where a tensor lives
is its layout, and changing layout is an instruction.

## Layouts

| Layout | Where | Created by | Consumed by |
|---|---|---|---|
| `Tensor` | DRAM fp32, row-major with a row stride | runtime `Input`/`Output` | `ol.load`, `ol.store` |
| `QTensor` | DRAM rows in an MXU format (`wf`: int8, int4, fp4, nvfp4) plus a scale row per row: its fp32 Sw, then its blocks' offsets (docs/quant.md "The MX formats"); `k0` the first block a column slice reads, `rsc` Sw alone (the K cache at d = D) | runtime `Weight`, KV cache views | the streamed operand of `ol.dot` |
| `Tile` | TMEM fp32, base + shape + row stride | load, VPU ops, dot | VPU ops, quantize, store |
| `Stationary` | ACT RAM int8 rows plus a scale per row and QACT range and offsets per D/4 (`qranges`) | `ol.quantize` | the stationary operand of `ol.dot` |
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

The KV cache (`compiler.KVDesc`): K token-major `[cap, d]` with a scale row per token
(`k_scale_bytes`: its fp32 scale at d = D, read by MM RSC; else an int8 scale row); V transposed,
`[d, cap]`, stored in tiles of 256 tokens (`VT_TILE`, the attention block) when the capacity is
a multiple of 256: tile *i* holds tokens 256*i .. 256*i + 255 of every row, `[d, 256]`
contiguous. P.V of one attention block then streams one contiguous 32 KB tile in full bursts
(untiled, each of its 128 rows was a separate 256-byte piece, cap bytes apart), and a token's
append writes bytes 256 apart at any capacity. An attention block must divide the tile; a
V^T column slice stays within one tile (`QTensor.padded_cols` lets an MM's zero-weighted
padding run past it).

`dot(a[M, K], w[N, K])` computes `a @ w.T`. The stationary side has at most 8 rows, so a
decode token, a GQA query group or a softmax probability block fits in one pass. Each weight
byte is read from DRAM exactly once per `dot`. The optional arguments map to MM epilogue flags:

| Argument | Flag | Effect |
|---|---|---|
| `out=t` | | write the result into an existing tile, e.g. a double buffer |
| `rowmax=True` | `RMAX` | also write the row maxima; read them as `result.rowmax` |
| `acc=t, acc_scale=alpha` | `ACC` + `ASCALE` | `t[i, j] = t[i, j] * alpha[i] + (a @ w.T)[i, j]` in the drain |

`ol.max(dot(...), axis=1)` on an unnamed result is rewritten to `rowmax` automatically.

An MM's rows read one activation scale (docs/isa.md "MM"), so `dot` runs one MM per QACT range
of the stationary operand (`Stationary.qranges`: one range unless parts were quantized on
their own), the first as asked and the rest `ACC`, `RMAX` on the last. A column slice of a
scaled weight moves the MM's `k0` (an immediate): at a run-time or loop-dependent column the
weight must be stored in column parts (`QTensor.parts`, each with its own scale rows).

## Broadcasting

VPU ops take a second operand in one of four modes, and the compiler picks the mode from the
operand's shape:

| Expression | Mode |
|---|---|
| `a * b`, same shape | FULL |
| `a * v[:, None]` | ROW: one value per row |
| `a * v[None, :]` | COL: one vector shared by all rows |
| `a * 0.5` | SCALAR: an immediate in the instruction |

`ol.load(desc, out=t)` loads into an existing tile, so a hardware loop can refill one of two
buffers while the other is in use (Qwen3.5's DeltaNet head pairs).

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

A freed register is 0 again, but the loops live when it was freed may still use it earlier in
their body. So it takes no address with terms of those loops (their step would leave it
non-zero for that use in the next iteration), and no address that starts at a run-time value
(a `RunVar`'s argument, a `DevVar`'s register, the generate loop's `RLD MUL` of a state word:
its init leaves `c * var` in the register after its loops), unless the loop's body zeroes the
register before any address takes it (a scratch register given back at the body's head, as
generate.py's token count). A loop with a device-computed count (a guard) may run 0 times and
skip the inits and zeroings inside it: after it, a register whose value was set inside it is
free again, taken last, and zeroed when an address or `scratch()` takes it. A body that would
end on the last instruction of an inner body (a `LoopBlock` or a guard emitted as
instructions) ends with a `NOP` (docs/isa.md, `LOOP`).

### Addresses above 4 GiB

At AW > 32 (docs/isa.md "Address bits": the BCU1525's 36 bits, 32 GiB) an instruction's
offset is a 32-bit word, zero-extended, so `Builder.addr` returns an address at or above 4 GiB
(or a loop's below 0) as a register that holds a base plus an offset below 4 GiB:

- **The window register** for a static address: one register, taken for good the first time,
  `LI Rw, B` (LI's `{w2, w1}`) where the address leaves `[B, B + 4 GiB)`, B the address rounded
  down to 256 MiB. The weights stream in order, so one LI serves the MMs of 4 GiB of them. Its
  value counts only from the LI on in the same loop body: a loop's start or end forgets it (the
  body runs again after its end, and an LI inside it is the window at the body's start).
- **A biased key** for a loop's address: its register's key gains the term `(BIAS, B)`, it
  starts at `LI r, B` (`ADDI r, R_arg, B` with a run-time term, the RLD MUL of the generate
  loop's word then `ADDI r, r, B`) before the outermost of its loops, and the loops step and
  reset it as any; B is the base of a live or spare register of the same terms that holds the
  address within 4 GiB, else the address rounded down to 256 MiB. When its loops end it holds B
  (a spare): the next run of the same base takes it with no LI (a layer run's weights after
  another's); with no free register left it is taken back, zeroed or set where taken.
- A bare run-time argument's address (`c * var` and a constant, no loop) at or above 4 GiB:
  a biased key too, `ADDI r, R_arg, B` where it is used.
- **A negative argument.** The host's argument words are zero-extended above AW 32, so
  `c * var` with `c < 0` (a row counted down from a bucket's end: RunPos.bucket's
  `-4 * tpos`) is the argument `-c * var~`, `var~ = bound - 1 - var` (`compiler.Comp`, named
  `var~` in run_args and the program cache), and the address's constant less
  `-c * (bound - 1)`; the variable needs a bound. The program's own loads of its arguments
  (the generate loop's `RLD MUL` of a state word, `isa.rld_mul`) form `c * var` in AW bits
  and keep `c`. At AW 32 every argument is as before.
- A device value's address (a `DevVar` or a `Counter`: the program's own register) with a
  constant at or above 4 GiB cannot be reached: the device value holds its base (an expert's
  address `ADDI(RLD MUL(id, expert bytes), pool base)`).

Steps and resets are `isa.wrap`'d into AW bits (a reset of a stride times a count of 4 GiB or
more); at AW 32 every word is as before (LI and ADDI with `w2 = 0`). At AW 32 an address of 4
GiB or more is an error (isa._w: an LD's word), where before AW it wrapped.

A split image (qwen3 / qwen35 `Image(high)`: a unit's weights from 4 GiB, its states below)
steps two registers in a layer loop where its units' one stride served. bcu1525-wg's programs
fit (Qwen3.5-4B, 27B and 35B-A3B high, fp4: decode and generate at every bucket of a 4K
context, at most 3947 instructions, the 27B's with its 24 pairs of DeltaNet heads unrolled and
each key head's q and k projected once, qwen35.PAIR_LOOP_QK; the 27B's tile runs of 4 rows, at
most 4045); on the m4 card's arch at AW 36 (board_config: group-major DeltaNet pairs, 7 or 8
arguments) the resident decode of Qwen3.5-4B's dims from bucket 8 is one address register
short ("out of address registers: R9 is taken, argument 6 needs it": the retired spares hold
the registers an argument found late needs; test_4b_resident_decode_fits_imem). Open: moving
an argument into a spare retired before its register was taken, as `_move_arg` does into a
released argument's.

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
| `ol.sum(S * k[None, :], axis=1)` | `VOP RDOT` (any broadcast of the second factor; `S @ k` is the same without the product tile) |
| `t.set(S @ k)` | the RDOT writes the 1-D tile `t` directly |
| `ol.max(ol.dot(...), axis=1)` | `MM RMAX` |
| `t.set(expr)` | the producer of `expr` writes straight into `t` |

## Linear recurrences

`x @ v` (a 2-D tile times a 1-D tile) is the row dot products, one RDOT pass with no product
tile; `ol.sum(x * v[None, :], axis=1)` on an unnamed product gives the same RDOT but allocates
the product tile while tracing. `ol.outer(x, y)` is the rank-1 tile `x[:, None] * y[None, :]`
(one MUL). With `acc=S` it is the state update of a linear recurrence, in place in one pass
(VOP OUTER): `S = S * decay + x[:, None] * y[None, :]`, where `decay` is a `[1]` tile (one
factor), a `[cols]` tile (per column, e.g. GLA, KDA) or omitted (1.0). It mirrors
`dot(acc=, acc_scale=)`. A `[1]` tile used as an operand broadcasts as a scalar read from TMEM
(`x * a[h:h+1]`). `ol.log2` and `lib.softplus` (`max(x, 0) + ln2 * log2(1 + 2^(-|x| log2 e))`)
complete the gates. The Gated DeltaNet step (`opentpu/kernels/deltanet.py`) keeps each head's
state transposed, `St[v, k]`, so every contraction is a row reduction:

```python
w.set(St @ k)                             # RDOT: kv
w.set((v - w * decay) * beta)             # 3 small VOPs: d
ol.outer(w, k, acc=St, decay=decay)       # OUTER: St = decay * St + d k^T
o[h].set(St @ q)                          # RDOT
```

Three passes over the 128 x 128 fp32 state instead of seven (MUL + RSUM, MUL, MUL, ADD,
MUL + RSUM), with no 16K-word temporaries, so two state buffers fit and the DMA loads the next
head's state during the passes. The schedule matters: a TMEM bank takes one write per cycle
and the DMA's writes come first, so a state LD (8 writes per cycle) stalls an OUTER for as long
as they overlap, while an RDOT (it holds its row sums until the last row) runs beside it. The
work tile `w` shared by all heads makes head h+1's first RDOT wait for head h's OUTER, which
puts the next LD under an RDOT (see `head_step`).

Measured on the RTL at the board configuration (LANES = 8, AXI memory path, 80% bandwidth,
`tools/perf_deltanet.py`): one head takes 9,173 cycles (old ISA: 17,360), and 16 heads
108,710 (6.6K per additional head; old ISA: 16.8K per head, from 1 and 8 heads).

## Memory allocation

- **TMEM** uses liveness: each allocation is a buffer object, freed when the last tile viewing it
  is garbage-collected. New buffers are placed next-fit from a moving cursor, which spreads
  consecutive temporaries over different addresses so the scoreboard sees no false conflicts.
  In a small TMEM next-fit can leave the free words in gaps too small for a later tile; a
  decode step, which has no smaller form to fall back to, is then traced again with best-fit
  placement (`compiler.trace_fit`). The compiles that probe (a prefill run's rows, a prompt
  run, a generate bucket's single program) keep next-fit's answer and fall back to their
  smaller forms, so a program that fits next-fit is placed as before.
- A hardware loop's body is traced once and runs again on the same words, so next-fit can put
  a body tile on the words the body's first load writes: the next iteration's load then waits
  for that tile's last use (a false WAR across iterations) and holds the in-order DMA queue
  behind it. `ol.load_home(desc, key)` loads into the program's tile for `key`, allocated at its
  first use and kept for the whole program, so no later tile takes its words (Qwen3.5's
  DeltaNet layers load their input norm's gain so). `opentpu.profile`'s `Profile.loop_waits`
  totals such waits from a trace.
- **ACT RAM** blocks are freed when their `Stationary` dies and are reused round-robin, so a
  stationary operand that is still live is never overwritten.

## Sizes come from the Config

Nothing in the compiler, the kernels or the models assumes the card's sizes; they read
`opentpu.isasim.Config` (a target's: docs/targets.md):

- TMEM: the LM head's chunks, the prefill runs' rows (`qwen3.fit_chunk` shrinks a run that does
  not fit), the snapshot copies' chunks, Gemma 4's MLP chunk (768 at the card's 64K words,
  proportionally less below).
- IMEM: the prefill runs and generate buckets that must fit it, the generate loop's program
  slots (`generate.prog_bytes`: two programs of IMEM's instructions); the ISA simulator refuses
  a program over IMEM, as the board's loader does.
- ACT RAM and TMEM: the one-hot operand of the device's gathers (`kernels/gather.py`) is
  quantized once where it fits, else each MM's blocks on their own.
- DRAM: Gemma 4's formats and PLE place follow the fit to the configuration's DRAM (at most the
  4 GiB a slice addresses).
- D: attention heads and DeltaNet heads of any multiple of D (LFM2's heads at most D, or a
  multiple), the emulations at the configuration's D.

A model that does not map onto a configuration is refused by its `Spec.check` with the limit
named (an inner dimension over ACT RAM, a head pair's DeltaNet state over TMEM without DSTEP,
one slice for Gemma 4 and MoE). `tests/test_config_matrix.py` runs the kernels and the tiny
models on every target's architecture, the design point and grown versions of the small one.

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
- Running out of TMEM (the tile's words and the words in use) or address registers, or nesting
  loops deeper than 4.
- At AW > 32, an address no register can reach ("Addresses above 4 GiB"): a device value's
  plus a constant at or above 4 GiB, or one past 2^AW; a negative argument of a variable
  with no bound.
- Reading a tile that `.set()` has already moved into another tile.
- Shape mismatches in stores, `set` and broadcasts, and unsupported strided views.
- A KV head accessed from a slice that does not own it.
