# openTPU ISA v1

This document is the contract between the compiler (`opentpu/compiler.py`), the bit-exact
instruction-set simulator (`opentpu/isasim.py`) and the RTL (`rtl/`). All three must agree on
every bit written to TMEM or DRAM.

## Machine model

- `S` slices. Each slice has: a sequencer with 16 x 32-bit registers (`R0` reads as 0), an
  instruction memory, a private DRAM, a TMEM, an ACT RAM, an MXU, a VPU and a quantizer.
  Slices share nothing except the collective unit (`GATHER`, `BAR`).
- `D` = MXU depth = quantization block size (bytes / int8 elements). Default 32 in tests,
  128 in the design. `MCOLS` = MXU columns (8). `ACT_ROWS` = ACT RAM rows =
  max stationary rows of one MM (at least MCOLS; default MCOLS). With `ACT_ROWS > MCOLS` the MXU
  consumes each streamed chunk once per group of MCOLS rows ("replay"): one DRAM pass serves all
  M rows, at ceil(M / MCOLS) cycles per chunk. Results do not depend on MCOLS or ACT_ROWS.
- **DRAM**: byte addressed, little endian, accessed as 32-bit words. All word accesses and all
  MXU chunk reads must be 4-byte aligned. Only `QST` writes single bytes.
- **TMEM**: 32-bit word addressed. Holds fp32 values (raw IEEE bits).
- **ACT RAM**: `ACT_ROWS` rows x `ACT_BLOCKS` blocks x `D` int8, plus one fp32 scale per
  (row, block). Written only by `QACT`, read only by `MM`.

Execution is in order. Every instruction completes (all its writes are visible) before the
next one starts. The MXU prefetches its streamed operand from DRAM internally.

### Arguments

A run starts with `R0..R7` = 0 and `R8..R15` = the run's arguments `ARG0..ARG7`: words the
host writes before RUN (board: control registers 0x060 + 4k, announced by CAPS bit25,
docs/observability.md; ISA simulator: `Machine(..., args)` / `load(programs, args)`; RTL
simulator: `+arg0=..+arg7=`; unwritten arguments are 0). The same program can then serve
different values: an address is `R[x] + imm`, and `LOOP` runs `R[ra] + w2` times.

The compiler's run-time values (`compiler.RunVar`) use them: an address that adds `c * var`
reads the argument register holding `c * var` (the host computes the product: there is no
multiply), or, when it also has loop terms, a register that `ADDI r, R_arg, 0` initializes
before the outermost of those loops. A program's k-th distinct (var, c) is in `R15 - k`
(`compiler.arg_reg`, `arg_words`), so address registers grow from `R1` and arguments from
`R15`. The resident decode programs (qwen3.compile_decode, docs/host.md) take the token id
and the position this way: LFM2.5-230M uses 6 arguments (token x 4096; the position within
its 256-token bucket x 128, x 4, x 1 and x -4; the convolution ring's row x 2048) and at most
8 address registers (up to 16 attention blocks).

A kernel done with a run-time value gives its argument registers back (`ol.release(var)`,
`Builder.release_arg`; the resident decode releases the token after its embedding row). Any
later use of the value is a compile error. The compiler takes a released register for an
address only when no other register is free, and zeroes it before that address's first loop
(it holds the argument until the release). A later argument whose own register `R15 - k` an
address took after the release is copied, at the release, into a released register that is
still free (`ADDI r, R15 - k, 0`), and used from there. Both only happen where the program
would otherwise run out of registers, so programs that fit without them do not change:
Qwen3.5-4B's resident decode with the int8 embedding gather (two arguments of the token,
eight in all) needs them from bucket 6 on.

## Arithmetic (fp32)

IEEE-754 binary32, round to nearest even, **flush to zero**: denormal inputs are treated as
signed zero and denormal results are replaced by signed zero. A result is denormal if IEEE
rounding (to the subnormal grid) gives a denormal: a product rounding up to `2^-126` is kept
(`0.5 * 0x00FFFFFF = 2^-126`; the multiplier must not round at 24 bits first). No NaN inputs are expected;
any NaN produced is the canonical `0x7FC00000` (a final sign flip, as in `recip`, may set its sign).
NaN operands are defined all the same (the hardware's behaviour, which opentpu/fp32.py models): `add`
and `mul` give the canonical NaN; flushing, max / min, abs, COPY and FILL keep a NaN's bits; the
compares order raw sign-magnitude bits (a NaN with the sign set is below `-inf`); `exp2(NaN) = +inf`,
`recip(NaN)` is a zero with its sign, `rsqrt` of a NaN with the sign set is `+0` (others: NaN),
`log2(NaN)` the canonical NaN, `q8(NaN) = 0`. Composite functions are defined as fixed sequences of
`add`/`mul` so that every implementation is bit exact:

- `i2f(i)`: int32 to fp32, RNE.
- `exp2(x)`: if `x < -126` return `+0`; if `x >= 128` return `+inf`. `i = floor(x)`,
  `f = x - i2f(i)`, `p = C0 + f*(C1 + f*(C2 + ... + f*C7))` (Horner, fp32, Taylor coefficients
  `ln2^k/k!` rounded to fp32), result = `p` with `i` added to its exponent field.
- `recip(x)`: `x == 0` returns `+0`; `|x| >= 2^126` (including infinity) returns a zero with
  the sign of `x` (the result would be subnormal and flush). Otherwise on `a = |x|`:
  `y = bits(0x7EF311C3 - bits(a))`, three times `y = y * (2 - a*y)`; the sign of `x` is
  applied at the end. If `|x| >= 2^123` (exponent field >= 250), `a = |x|/16` (the field minus
  4) and the result is `y/16` (y's field minus 4; y >= 2^-122 there, so it stays normal):
  unscaled, the seed flushes for `|x| > 0x7E7311C3`. This keeps `silu(x) = x * recip(1 + exp2(-x*log2e))` exact at `-0` for
  very negative `x`, where `exp2` overflows to infinity.
- `rsqrt(x)`: `x <= 0` (any sign bit set) and `x = +inf` return `+0`. With `x'` = `16x` if x's
  exponent field is <= 2, `x/16` if it is 250..254, else `x` (a change of the field by 4):
  `y = bits(0x5F3759DF - (bits(x') >> 1))`, `h = 0.5*x'`, three times `y = y * (1.5 - h*(y*y))`;
  the result is `4y`, `y/4` or `y` (y's field plus 2, minus 2, unchanged). Unscaled, `h`
  flushes at field 1 and `y*y` at fields >= 252 (NaN or errors up to 4x). The steps are exact
  under the scaling, so every other input gives the unscaled bits.
- `log2(x)`: `x = +-0` returns `-inf`, `x < 0` (and NaN) the canonical NaN, `x = +inf` `+inf`.
  Otherwise, with `f` the 23 fraction bits of `x` and `ex` its exponent field: `ge = f >= 0x3504F3`
  (the mantissa is at least sqrt(2)), `e = ex - 127 + ge`, `m = bits((ge ? 126 : 127) << 23 | f)`
  (so `x = 2^e * m`, `m` in [sqrt(1/2), sqrt(2))), `t = m + (-1)` (exact), `q = C9`, then
  `q = q*t + Ck` for k = 8, 7, ..., 1, and the result is `q*t + i2f(e)` (every `a*b + c` is a
  rounded `mul` then a rounded `add`). `C1..C9` are a minimax fit of `log2(1+t)/t` on
  [sqrt(1/2)-1, sqrt(2)-1], as fp32 bits: `3FB8AA3B BF38AA38 3EF639EB BEB8AE27 3E9369C2
  BE74ADF2 3E5CE48E BE543E8E 3E00DB73`. The result is within 2.3 ulp of the exact value
  (every fp32 in [0.5, 4), and samples of every exponent: `tests/test_vops.py`).
- `a > b` compares flushed values in a total order: `-inf < ... < -0 < +0 < ... < +inf < NaN`
  (sign-magnitude bits). `max(a, b)` = `a if a > b else b`; `min(a, b)` = `a if b > a else b`.
  Because the order is total, a max/min reduction gives the same bits in any order.
- **Sums** (MM accumulation and the VOP row sums) are defined so that a pipelined adder can
  compute them at full rate. `isum_P(x[0..n-1])` with P a power of two: partial
  `p[q] = +0 + x[q] + x[q+P] + x[q+2P] + ...` (left to right, each `+` an fp32 add), then a
  folding tree over the partials: `n = P/2, P/4, ..., 1: p[i] = p[i] + p[i+n]` for `i < n`.
  (For MM that is `(p0 + p2) + (p1 + p3)`.)
  MM uses P = 4 over the K blocks; RSUM, RSSQ and RDOT use P = 64 over the columns. RSUM, RSSQ
  and RDOT pad a row with +0 terms to a multiple of 64 columns (each added: a -0 partial, from a
  sum that flushed, becomes +0). MM has no pad terms: a partial is the chain of its own blocks
  only (one with no block is the +0 it starts from).
- `q8(x)`: round half to even to an integer, saturate to `[-127, 127]`; NaN gives 0 (a zero
  times an infinite `inv`, below).

Quantization of a group `x[0..n)` (a block of `D`, or a whole row in row mode):
`amax = max |x|`; if `amax == 0`: `s = 0`, `inv = 0`; else `s = amax * f32(1/127)`,
`inv = 127 * recip(amax)`; `q[i] = q8(x[i] * inv)`.

## Encoding

Every instruction is 8 x 32-bit words `w0..w7`.
`w0 = opcode[7:0] | ra[11:8] | rb[15:12] | rc[19:16] | rd[23:20] | flags[31:24]`.
`R[x]` is the register value. Addresses below are "register + immediate".

| op | name | semantics |
|---|---|---|
| 0x00 | NOP | |
| 0x01 | HALT | stop this slice; flag bit0 CHAIN: then start the program at `R[ra]` (see "HALT CHAIN") |
| 0x02 | LI | `R[rd] = w1` |
| 0x03 | ADDI | `R[rd] = R[ra] + w1` |
| 0x04 | LOOP | body = next `w1` instructions, executed `R[ra] + w2` times (0: skipped). Loops nest (depth 4); a body must not end on the same instruction as an enclosing body. |
| 0x05 | BAR | wait until every slice has reached a `BAR` |
| 0x06 | RLD | `R[rd] = f2i(T[R[ra]+w1])`, flag bit0 RAW: the word's bits, bit1 MUL: times `R[rb]+w2` (see "RLD") |
| 0x07 | WAITW | DMA: wait until `M32[R[ra]+w1] & w4` compares (flags[1:0]: EQ, NE, GE) with `R[rc]+w3`, then `T[R[rb]+w2]` = the word's bits (see "WAITW") |
| 0x10 | LD | DRAM -> TMEM, `n = w3` words: `T[R[rb]+w2+i] = M32[R[ra]+w1+4i]` |
| 0x11 | ST | TMEM -> DRAM: `M32[R[ra]+w1+4i] = T[R[rb]+w2+i]` for `i < w3` |
| 0x12 | DSTEP | one Gated DeltaNet head step on a DRAM state, run by the DMA (see below; `Config.DSTEP`, CAPS bit6) |
| 0x20 | MM | see below |
| 0x21 | QACT | quantize TMEM rows into ACT RAM |
| 0x22 | QST | quantize TMEM rows into DRAM bytes |
| 0x30 | VOP | vector op on a TMEM tile |
| 0x40 | GATHER | all-gather over slices |

### MM

Fields: `sa = R[ra]+w1` (streamed int8 rows, bytes), `ssa = R[rb]+w2` (streamed scales, bytes),
`out = R[rc]+w3` (TMEM words), `N = w4[15:0]`, `KB = w4[31:16]`, `rs = w5` (row stride,
bytes), `ors = w6[15:0]` (output row stride, words), `M = w6[23:16]`, `ab = w6[31:24]` (first
ACT RAM block), `srs = w7` (scale row stride, bytes). Flags: bit0 `UNIT` (streamed scales are
1.0 and not read), bit1 `ACC` (accumulate into `out`), bit2 `RMAX` (also write each output
row's maximum: `T[out + M*ors + j] = fold(max, y[j][0..N-1])`, folded from `n = 0` like the
`RMAX` VOP; the softmax row max comes for free from the MXU epilogue), bit3 `ASCALE` (needs
`UNIT` and `ACC`; `ssa` is then the TMEM address of M per-row factors and the old accumulator
is rescaled first: `y = T[out + j*ors + n] * T[ssa + j] + acc[j]` -- the flash-attention
correction step, done in the MXU epilogue), bits 5:4 `WF`, the streamed weights' format (below;
0 = int8), bit6 `PAIR` (4-bit weights at full rate, "Column reuse"). The streamed rows are
D-byte aligned (`sa` and `rs` are multiples of D): the MXU streams whole D-byte DRAM chunks.
`0 < M <= ACT_ROWS`; RMAX, ASCALE and PAIR need `M <= MCOLS` (PAIR: `2*M <= MCOLS`), and an MM
with `M > MCOLS` needs its streamed row to fit the MXU's chunk FIFO (`KB` chunks for int8,
`ceil(KB/2)` for 4-bit; the board: 1024; a replayed row stays in the FIFO until its last group).

```
for n in 0..N-1:
  for k in 0..KB-1:
    w  = M8[sa + n*rs + k*D + i], i < D          (int8)
    ws = UNIT ? 1.0 : M32[ssa + n*srs + 4k]
    for j < M:
      isum      = sum_i ACT[j][(ab+k)*D + i] * w[i]  (exact integer)
      t[j][k]   = (i2f(isum) * ws) * ASCALE[j][ab+k]
  for j < M:
    acc[j] = isum_4(t[j][0..KB-1])              (see "Sums")
    y = acc[j];  if ACC: y = T[out + j*ors + n] + acc[j]
    T[out + j*ors + n] = y
```

#### Weight formats

`WF = 0` (int8) is the loop above. `WF = 1` (int4) and `WF = 2` (FP4, E2M1) stream 4-bit
elements, two per byte: block `k` of row `n` is the `D/2` bytes at `sa + n*rs + k*D/2`
(two blocks per D-byte chunk; a row's last chunk is half used when KB is odd), element `i` in
the low nibble of byte `i/2` for even `i`, the high nibble for odd `i`. A nibble `c` is the
integer `w = c - 16*c[3]` (int4, -8..7) or, for E2M1, twice its value: `c[2:0]` in
{0, 1, 2, 3, 4, 6, 8, 12} with `c[3]` the sign (so a stored E2M1 matrix carries half its
scale). Each block has one scale word `sw = M32[ssa + n*srs + 4k]`, two-level: `ws = bf16(sw[15:0])`
(the fp32 with bits `sw[15:0] << 16`) and four unsigned multipliers `m_b = sw[16+4b +: 4]`, one
per sub-block `b` of `D/4` elements. The block's integer is the exact

```
isum = sum_b m_b * sum_{i in b} ACT[j][(ab+k)*D + i] * w[i]      (|isum| < 2^22 at D = 128)
```

and everything after it (`i2f`, `* ws`, `* ASCALE`, the sums, ACC, RMAX, ASCALE) is as for int8.
With `UNIT`, `ws = 1.0` and every `m_b = 1`. opentpu/quant.py makes these matrices
(`quantize_w4`: `ws` a bf16 block scale, `m_b` in 1..15 chosen to minimize the squared error) and
docs/quant.md compares the formats. The stationary operand (ACT RAM), the KV cache and QST stay
int8.

#### Column reuse

A 4-bit MM takes one block per cycle, half a streamed chunk. Flag bit6 `PAIR` (4-bit `WF` only,
`2*M <= MCOLS`) lets it take a whole chunk per cycle when the operand has at most MCOLS/2 rows:
the idle columns `M..2M-1` take the odd blocks. Output row `j` gets its even blocks from ACT
row `j` and its odd blocks from ACT row `j + M` (which `QACT DUP` fills with the same row), and
the two terms of a chunk are added before the sums:

```
t[r][k]  as above, for ACT rows r < 2M
p[j][c]  = t[j][2c] + (2c+1 < KB ? t[j+M][2c+1] : +0)       c < ceil(KB/2)
acc[j]   = isum_4(p[j][0..ceil(KB/2)-1])
```

The streamed rows and scales are the same as without `PAIR`: chunk `c` holds blocks `2c` and
`2c+1`, whose scale words are adjacent (`ssa + n*srs + 8c`) and are read as one 8-byte pair, so
without `UNIT` both `ssa` and `srs` are multiples of 8 (with KB odd the scale rows are padded; the
last chunk's second word is read and ignored). ACC, RMAX and ASCALE are unchanged and write M
rows. The results differ from a `PAIR`-less MM only in fp32 rounding (the sum order).

### QACT

`src = R[ra]+w1` (TMEM words), `rows = w2[7:0]` (at most ACT_ROWS), `ab = w2[15:8]`,
`KB = w2[31:16]`, `srs = w3` (source row stride, words). Flag bit0 `ROW`: one scale per row instead of per block.
Flag bit1 `CSCALE`: every element is first multiplied by a per-column scale,
`x = T[src + r*srs + c] * T[w4 + c]` (this folds V's per-token scale into P for free).
Flag bit2 `RSCALE`: every element is first multiplied by a per-row factor `T[w5 + r]`. With
both, `x = (T[src + r*srs + c] * T[w5 + r]) * T[w4 + c]` (RMSNorm's `x * r * gamma`).
For each row `r < rows` and block `k < KB`: quantize `T[src + r*srs + k*D + i]`, write
`ACT[r][(ab+k)*D + i] = q[i]`, `ASCALE[r][ab+k] = s`.
Flag bit3 `DUP` (`2*rows <= MCOLS`): row `r + rows` receives the same bytes and scales as row `r`
in the same cycles, the operand layout of `MM PAIR` ("Column reuse").

### QST

`src = R[ra]+w1` (TMEM), `dst = R[rb]+w2` (DRAM bytes), `sdst = R[rc]+w3` (DRAM bytes),
`rows = w4[15:0]`, `KB = w4[31:16]`, `srs = w5` (words), `drs = w6` (bytes), `es = w7`
(element stride, bytes). Flag bit0 `ROW`.
Element `c` of row `r` goes to byte `dst + r*drs + c*es`. Scales: per block to
`sdst + (r*KB + k)*4`; in `ROW` mode one scale per row to `sdst + r*4`. The data and scale ranges of
one QST must not overlap. Flag bit1 `HALF` (`ROW` mode only): the scale is still the whole row's,
but only elements `c < KB*D/2` are written (a V^T append of a head half as wide as its padded row,
LFM2's 64 of 128, then writes its 64 real rows instead of 128 byte-strided ones; a quantizer
without `HALF` writes the zero padding too, which nothing reads).

### VOP

`dst = R[ra]+w1`, `a = R[rb]+w2`, `b = R[rc]+w3` (TMEM words), `rows = w4[15:0]`,
`cols = w4[31:16]`, `drs = w5[15:0]`, `ars = w5[31:16]`, `brs = w6[15:0]`,
`func = w6[23:16]`, `bmode = w6[25:24]`, `imm = R[rd] + w7` (fp32 bits; OUTER: a TMEM address,
register-relative like the others; rd = 0 for an immediate).

`B(r,c)` is `T[b + r*brs + c]` (bmode 0, full), `T[b + r*brs]` (1, per row; with `brs = 0`, one
TMEM scalar for the whole tile), `T[b + c]` (2, per column) or `imm` (3, scalar).
`A(r,c) = T[a + r*ars + c]`.
Elementwise functions write `T[dst + r*drs + c] = f(A, B)`; reductions write
`T[dst + r*drs] = fold(A(r, 0..cols-1))` sequentially from `c = 0`.

| func | name | result |
|---|---|---|
| 0 | ADD | A + B |
| 1 | SUB | A - B |
| 2 | RSUB | B - A |
| 3 | MUL | A * B |
| 4 | MAX | max(A, B) |
| 5 | MIN | min(A, B) |
| 6 | OUTER | A * Dv(c) + B(r) * Cv(c), in place (see below) |
| 8 | COPY | A |
| 9 | EXP2 | exp2(A) |
| 10 | RECIP | recip(A) |
| 11 | RSQRT | rsqrt(A) |
| 12 | ABS | abs(A) |
| 13 | FILL | B (A is not read) |
| 14 | EXP2SUB | exp2(A - B) (the softmax step, fused) |
| 15 | LOG2 | log2(A) |
| 16 | RSUM | isum_64(A(r, 0..cols-1)) (see "Sums") |
| 17 | RMAX | max over A(r, c) (total order: any evaluation order) |
| 18 | RSSQ | isum_64(A(r,c) * A(r,c)) (sum of squares, for RMSNorm: RDOT with B = A) |
| 19 | RDOT | isum_64(A(r,c) * B(r,c)), B in any bmode (row dot products: `S @ k` is B per column) |
| 20 | ARGMAX | the RMAX of the row at `T[dst + r*drs]` and the index of its first column at `T[dst + r*drs + 1]` (see below) |

In RSSQ and RDOT each product is rounded, then added into the isum_64 partials.

**ARGMAX** writes a pair per row: `m = ` the row's RMAX, and `i2f(c + base)` for `c` the first
column with `A(r, c) == m` (flushed bits; in the total order `-0 < +0`, so ties are between equal
bits only, and the first wins as in `np.argmax`). `base = R[rd] + w7` is a signed integer here, not
fp32 bits (an LM head chunk passes its first vocabulary row, so the index is the token id). The
pair of row `r` is `T[dst + r*drs]`, `T[dst + r*drs + 1]`: rows > 1 need `drs >= 2`. B is not read.

**OUTER** (the state update of linear recurrences: Gated DeltaNet, Mamba2/SSD, GLA, RWKV,
linear attention) is elementwise and in place, `T[dst + r*drs + c] = add(mul(T[dst + r*drs + c],
Dv(c)), mul(B(r), Cv(c)))`: two rounded products, then a rounded add. B must be per row
(bmode 1): `B(r) = T[b + r*brs]`. The column vector is `Cv(c) = T[imm + c]`, and the decay
`Dv(c)` is `T[a + c]`, or `T[a]` for every column with flag bit 0 (DSCALAR), or 1.0 with flag
bit 1 (DONE; `a` is not read). The `a` field holds the decay address: A is dst itself (`ars` is
not used). `cols <= 256`. `Cv` and `Dv` are read before anything is written, so they may overlap
dst; `B(r)` is read with every element and must not be written by an earlier element.

### DSTEP

One Gated DeltaNet head step, run by the DMA on a fp32 state that stays in DRAM: the DMA
streams the state `St [rows, cols]` (row-major, the transposed state: rows are the value
dimension) from DRAM through its datapath and writes it back in place, so the state never
enters TMEM. `dram = R[ra]+w1` (bytes, chunk aligned), `qk = R[rb]+w2`, `v = R[rc]+w3`,
`rows = w4[15:0]` (1..256), `cols = w4[31:16]` (64, 128, 192 or 256), `g = w5`, `o = w6`,
`gs = w7`; flag bit 0 (ZERO): the state is taken as 0 and not read (position 0). With
`q = T[qk + c]`, `k = T[qk + cols + c]`, `v(r) = T[v + r]`, `e = T[g]`, `beta = T[g + gs]`:

```
kv = RDOT(St, k)                       isum_64 row dots, as the VOP
d  = MUL(SUB(v, MUL(kv, e)), beta)
St = OUTER(St, e, d, k)                add(mul(St, e), mul(d(r), k(c))), in place in DRAM
o  = RDOT(St, q)                       T[o + r]
```

bit for bit the VOP sequence RDOT, MUL, SUB, MUL, OUTER (DSCALAR), RDOT it replaces
(`tests/test_vops.py::test_dstep_is_the_vop_sequence`). Every input is read before anything is
written; `o` is written last. The scoreboard footprint: DRAM `[dram, dram + 4 rows cols)` written,
TMEM `[qk, qk + 2 cols)`, `[v, v + rows)` and `{g, g + gs}` read, `[o, o + rows)` written.
A bitstream without it leaves CAPS bit6 clear; the compiler then emits the VOP sequence
(`Config.DSTEP = False`, the default of `board_config`; the host takes it from CAPS through
`device_config`). Its slice takes DSTEP and STREAM as illegal instructions: the run stops with
ERROR (otpu_seq STREAMS; before fix-board the DMA waited for the missing stream engine forever).

On the board DSTEP runs on the stream engine (`docs/stream.md`): the DMA moves the state, and
the VPU's slot-0 partial loop and tree plus the tail (`rtl/vpu/otpu_se_tail.sv`) compute. It
takes 8 state words per cycle: a head of 128 x 128 is 2,048 cycles of datapath plus about 200
of fill and pipeline, against 1,024 cycles of port-B chunks (64 KiB read, 64 KiB written).
The DMA reads the state 16 chunks at a time and writes it back in runs of 16 gathered chunks
(DRAM bursts; timing only). The VOPs wait while a DSTEP holds the engine. Only an 8-lane
build (W = 8) has DSTEP.

### STREAM

`STREAM` (0x13) generalizes DSTEP. A descriptor of float-safe TMEM words (written by FILLs)
programs a pass over the rows of a stream:
- **A:** a row dot;
- **SCALAR:** a few scalar ops per row;
- **U:** `Y = S*G + D*B`;
- **Q:** a dot of Y.

Fields:
- `w1` = desc [15:0] | ks [31:16]
- `src = R[ra]+w2`, `dst = R[ra]+w3`
- column slots `vec = R[rb]+w4`, row scalars `x = R[rc]+w5`
- constants `K_j = T[R[rd]+w6 + j*ks]`
- row outputs `o = w7`
- flags: ZERO, SRC_T, DST_T, NODST

The full semantics, the descriptor format and the model mappings (Gated DeltaNet, DeltaNet,
KDA, GLA, RetNet, Mamba2, mLSTM, RWKV-7, RMSNorm, attention's reductions) are in
[stream.md](stream.md). The ISA simulator runs all of it.

The board runs the subset `opentpu.isa.stream_hw_cfg` accepts, announced by **CAPS bit26 =
STREAM** (`regs.CAP_STREAM`; bit26 is taken; the full CAPS list is the register table in
[observability.md](observability.md)):
- DRAM state in place, rows ≤ 256, cols 64..256;
- the state-step modes.

The compiler (`ol.state_step`) falls back to VOPs otherwise. DSTEP is STREAM with
`isa.gdn_desc` and ks = gs.

### RLD

`R[rd] = f2i(T[R[ra] + w1])`: a TMEM word into a register, the only way a value the device
computed reaches an address, a loop count (`LOOP R[ra] + w2`) or a condition (a `LOOP` of count
0 or 1 around the instructions it guards). `f2i` truncates toward zero: `|x| < 1` (with +-0 and
the flushed denormals) gives 0, `|x| >= 2^31`, infinities and NaN give `0x80000000`; integers of
up to 24 significant bits come out exact. Flag bit0 RAW: the word's bits instead. Flag bit1
MUL: the value (f2i's or RAW's) times `R[rb] + w2`, the low 32 bits of the product (signed and
unsigned alike): an integer the device computed scaled into a byte offset past fp32's 24 bits,
e.g. a token's row in a 2.4 GB table (`tok * 9344`). `rd = 0` writes nothing.

RLD reads its word after every older instruction that writes it (the scoreboard, like any TMEM
read), and no younger instruction is issued until `R[rd]` holds the value: they may use it. The
cost is the wait for the writer plus a few cycles; the decode loop (`opentpu/llm/generate.py`)
uses a handful per token.

### WAITW

Wait for a word the host writes. The DMA reads `v = M32[R[ra] + w1]` from DRAM until
`cmp(v & w4, R[rc] + w3)` holds, then writes `v` to `T[R[rb] + w2]` (the word's bits, as LD).
`RLD` with RAW takes it into a register; the value never passes the VPU, so an address comes
through whole.
- flags[1:0] `cmp`: 0 EQ, 1 NE, 2 GE. GE means the 32-bit difference `(v & w4) - ref` is >= 0
  as a signed number: counters, and positive fp32 values, which order as their bits.
- `w5`: cycles between reads, the first at once. `w6`: a timeout in cycles (0: none), at which
  the slice stops with an error the host sees.
- The scoreboard footprint: all of DRAM read, `T[R[rb] + w2]` written. Older stores land before
  its first read; younger instructions that read or write DRAM, and those that use the word,
  wait until it completes.
- Every read is a fresh DRAM read. Once WAITW has seen a word the host wrote after an h2c DMA
  completed, every younger MM or LD reads that DMA's data: the host orders its data before its
  flag, the card its flag before its reads. The XDMA and the core meet in `otpu_mem_ch` and
  LiteDRAM, which must keep this order. The board's DRAM adapter keeps port A's last beat and
  read runs (the MXU's scales), which the host's writes do not reach: a WAITW that holds drops
  them, as a run's start and a program load do (`otpu_native_dram` `a_flush`). Bitstreams before
  the port-A flush (production up to g2fix 0885d436) did not: an MM whose first scale read after
  a WAITW, or in a new run, fell in the beat of the last scale read, or in the next channel beat
  of its read run, took the old data (the 35B's back-to-back embed runs). A flag in the last
  beat of the DMA that carries the data (a slot's tag, docs/offload.md 10.11) also needs that
  DMA's writes on its channel to land in order: its other channel's DMA has completed before
  it starts.

The MoE expert streaming of docs/offload.md uses it for a fence (`served >= seq`: the host has
finished the card's earlier requests), for each present expert's directory entry and each
missing one's word of the request's answer (`!= 0`: the expert's slot address), and for a
missing expert's slot tag (`!= 0`: its DMA has landed; docs/offload.md 10.11). The ISA simulator runs it in order (the slice
waits) and calls the host (`Machine.host`) when every slice that can run waits; a WAITW that
still does not hold is the timeout (SimError).

In the RTL (otpu_dma, CAPS bit31) it is an LD of one word whose TMEM write waits for the
compare: the chunk is read, the word taken, compared a cycle later and written through lane 0,
or, if it does not hold, read again after `w5` cycles. The timeout is taken between two reads:
a read in flight when `w6` cycles have passed still completes the WAITW if its word holds, so a
WAITW either completes (the word written, younger instructions go on) or times out (nothing
written). At the timeout the slice stops: the sequencer and the units are held in reset (no
instruction starts or ends after it; TMEM, the DRAM and ICOUNT stay as they were), and STATUS
shows HALTED (once the units are stopped and WR_IDLE holds), ERROR and WAIT_TO (bit8; the first
WAITW bitstream, be824d5, shows HALTED and ERROR only), until RUN falls. Bitstreams before
fix-board showed HALTED at once while the other units went on, and a read in flight at the
timeout could still complete the WAITW after ERROR rose. The scoreboard sees all of DRAM as
written (older DRAM readers and writers complete first, younger ones wait) and the TMEM word.

On the card, `tools/qual/waitw.py` (qual.sh, and otpu-diag's `waitw-host` group) checks that
order (opentpu/host/checks.py `waitw_host`). In each round the host:
1. writes old data and a flag the compare fails on;
2. starts the card and checks that it waits;
3. writes new data, then the flag.

The card's LD after the WAITW must read the new data (1 to 32768 words from any word offset; EQ,
NE, GE and a masked EQ; poll intervals 0, 64 and 1000 cycles; flags in both channels). Then a
WAITW that never holds must stop at its timeout with ERROR, and the next run halt normally.

### HALT CHAIN

`HALT` with flag bit0 (CHAIN) halts the slice as `HALT` does (the window drains, every store has
landed), then the board loads `R[rb]` instructions from DRAM byte address `R[ra]` (chunk aligned,
at most the IMEM) into IMEM and starts them: `R0..R7 = 0`, `R8..R15` the run's arguments as the
host wrote them; TMEM, ACT RAM and DRAM keep their contents. On the card the run goes on (RUN
stays set, CYCLES counts, ICOUNT adds the programs' instructions up); the host sees one run.
The reload is the program loader's DRAM read of the new program, once per chained program. The decode loop's program of an attention
bucket chains to the next bucket's.

### GATHER

All slices must execute a `GATHER` with the same `dst`, `rows`, `cols`, `drs`, `seg`.
`src = R[ra]+w1`, `dst = R[rb]+w2`, `rows = w3[15:0]`, `cols = w3[31:16]`, `srs = w4`,
`drs = w5`, `seg = w6` (words). For every slice `s`, every `r < rows`, `c < cols`:
`T_all[dst + s*seg + r*drs + c] = T_s[src + r*srs + c]` is written into every slice's TMEM.
The source and destination ranges must not overlap.
