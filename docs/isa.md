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
- **ACT RAM**: `ACT_ROWS` rows x `ACT_BLOCKS` blocks x `D` int8, plus a scale record per
  (row, block): the row's fp32 scale `Sa[r][k]` and four 8-bit offsets `da[r][k][b]`, one per
  sub-block `b` of `D/4` elements (the sub-block's scale is `Sa * 2^-da`). Written only by
  `QACT`, read only by `MM`.

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

IEEE-754 binary32 values; `add`, `sub` and `mul` round as IEEE 754's roundTiesToEven, with
**flush to zero**: denormal inputs are treated as signed zero and denormal results are replaced by
signed zero. A result is denormal if IEEE
rounding (to the subnormal grid) gives a denormal: a product rounding up to `2^-126` is kept
(`0.5 * 0x00FFFFFF = 2^-126`; the multiplier must not round at 24 bits first). A zero sum has
IEEE's sign under roundTiesToEven: an exact cancellation is `+0` (`x - x`, `x + (-x)`,
`(+0) + (-0)`), `(-0) + (-0) = (-0) - (+0) = -0`, so `x + (-0) = x` for every `x` (the units use
`v*1 + -0` as the identity); a sum that flushes is a zero with the sign of the exact sum
(tests/test_fp.py and test_fp_rtl.py check these on the RTL against the host's IEEE adder).
This is not a conforming IEEE 754 implementation, by design: roundTiesToEven is the only rounding
direction (no roundTowardZero / Positive / Negative, no rounding-mode register), denormals flush,
there are no exception flags (an overflow is `+-inf`, an invalid operation the canonical NaN),
and divide, square root and the other functions are the approximations below, not correctly
rounded operations. No NaN inputs are expected;
any NaN produced is the canonical `0x7FC00000` (a final sign flip, as in `recip`, may set its sign).
NaN operands are defined all the same (the hardware's behaviour, which opentpu/fp32.py models): `add`
and `mul` give the canonical NaN; flushing, max / min, abs, COPY and FILL keep a NaN's bits; the
compares order raw sign-magnitude bits (a NaN with the sign set is below `-inf`); `exp2(NaN) = +inf`,
`recip(NaN)` is a zero with its sign, `rsqrt` of a NaN with the sign set is `+0` (others: NaN),
`log2(NaN)` the canonical NaN, `q8(NaN) = 0`. Composite functions are defined as fixed sequences of
`add`/`mul` so that every implementation is bit exact:

- `i2f(i)`: a signed integer to fp32, RNE (MM's accumulators have at most 48 bits).
- `exp2(x)`: if `x < -126` return `+0`; if `x >= 128` return `+inf`. `i = floor(x)`,
  `f = x - i2f(i)`, `p = C7`, then `p = p*f + Ck` for k = 6, 5, ..., 0 (each a rounded `mul`
  then a rounded `add`), result = `p` with `i` added to its exponent field. `C0..C7` are a
  minimax fit of `2^f` on [0, 1] (relative error, degree 7) with `C0 = 1`, rounded to fp32 and
  then searched in fp32 for the smallest error through these steps, as fp32 bits: `3F800000
  3F317218 3E75FDE9 3D63593E 3C1D8534 3AAFE2F6 3916C07C 37B3C7D6`. The result is within
  1.26 ulp of the exact value where `f` is exact (`x >= 0` and `x <= -0.5`) and within 1.61 ulp
  where `x + 1` rounds (`-0.5 < x < 0`), measured over every fp32 `f` in [0, 1] and every fp32
  `x` in (-0.5, 0); it is exact at every integer and non-decreasing in `x` (`p` is
  non-decreasing over every fp32 `f` in [0, 1], and `p(1) = 2`): `tests/test_fp.py`. (Taylor
  coefficients `ln2^k/k!` were up to 13 ulp off near `f = 1`.)
- `recip(x)`: `x == 0` returns `+0`; `|x| >= 2^126` (including infinity) returns a zero with
  the sign of `x` (the result would be subnormal and flush). Otherwise, with `ex` the exponent
  field of `x`, on its significand `a = bits(127 << 23 | (bits(x) & 0x7FFFFF))` (in [1, 2)):
  `y = bits(0x7EF311C3 - bits(a))`, twice `y = y * (2 - a*y)`, then `e = 1 - a*y` and
  `y = y*e + y` (a rounded `mul` then a rounded `add`); the result is `y` (in [0.5, 1]) with
  `127 - ex` added to its exponent field (it stays in 1..254) and the sign of `x`. The last
  step is in correction form because `y * (2 - a*y)` without a fused multiply-add has a fixed
  point an ulp below `1/a` (`recip(1)` was `0.99999994`). The result is within 1.16 ulp of the
  exact value (every fp32 in [1, 2), and samples of every exponent) and exact at every power of
  two: `tests/test_fp.py`. Returning a zero for `|x| >= 2^126` keeps
  `silu(x) = x * recip(1 + exp2(-x*log2e))` exact at `-0` for very negative `x`, where `exp2`
  overflows to infinity.
- `rsqrt(x)`: `x <= 0` (any sign bit set) and `x = +inf` return `+0`. With `x'` = `16x` if x's
  exponent field is <= 2, `x/16` if it is 250..254, else `x` (a change of the field by 4):
  `y = bits(0x5F3759DF - (bits(x') >> 1))`, `h = 0.5*x'`, twice `y = y * (1.5 - h*(y*y))`,
  then `e = 0.5 - h*(y*y)` and `y = y*e + y`; the result is `4y`, `y/4` or `y` (y's field plus
  2, minus 2, unchanged). Unscaled, `h` flushes at field 1 and `y*y` at fields >= 252 (NaN or
  errors up to 4x). The steps are exact under the scaling, so every other input gives the
  unscaled bits. The result is within 1.03 ulp of the exact value (every fp32 in [1, 4), and
  samples of every exponent) and exact at every power of four: `tests/test_fp.py`.
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
- **Sums** (the VOP row sums) are defined so that a pipelined adder can compute them at full
  rate. `isum_P(x[0..n-1])` with P a power of two: partial
  `p[q] = +0 + x[q] + x[q+P] + x[q+2P] + ...` (left to right, each `+` an fp32 add), then a
  folding tree over the partials: `n = P/2, P/4, ..., 1: p[i] = p[i] + p[i+n]` for `i < n`.
  RSUM, RSSQ and RDOT use P = 64 over the columns and pad a row with +0 terms to a multiple
  of 64 columns (each added: a -0 partial, from a sum that flushed, becomes +0). MM sums
  exactly: an integer ("MM").
- `q8(x)`: round half to even to an integer, saturate to `[-127, 127]`; NaN gives 0 (a zero
  times an infinite `inv`, below).

Quantization of a row `x[0..n)` (QACT and QST: the `KB*D` elements of one row) with its groups
(QACT: the sub-blocks of `D/4` elements; QST: the D-blocks; `ROW` mode: every `d = 0`):

```
amax = max |x| over the row            (flushed bits, as RMAX)
S    = amax == 0 ? 0 : amax * f32(1/127)                  the row's scale
inv  = amax == 0 ? 0 : 127 * recip(amax)
a_g  = max |x| over group g
d_g  = (a_g == 0 or ROW) ? 0 : min(31, e(amax) - e(a_g) - (f(a_g) > f(amax) ? 1 : 0))
q[i] = q8(x[i] * (inv * f32(2^d_g)))                      i in group g; two rounded mul
```

`e` is the exponent field and `f` the fraction bits: `d_g` is the largest `d <= 31` with
`a_g * 2^d <= amax`, so group `g` is quantized with the scale `S * 2^-d_g`, within a factor
of 2 of its own `a_g / 127`. (Before the MX MXU, docs/mx_mxu.md, every D-block had its own
fp32 `s = amax_block / 127`.)

## Encoding

Every instruction is 8 x 32-bit words `w0..w7`.
`w0 = opcode[7:0] | ra[11:8] | rb[15:12] | rc[19:16] | rd[23:20] | flags[31:24]`.
`R[x]` is the register value. Addresses below are "register + immediate": the fields written
`R[x] + w`. Counts, shapes and strides are immediates, whatever the register fields hold (LD's
and ST's `n`, QACT's `w2` and `w3`, GATHER's `w3`).

| op | name | semantics |
|---|---|---|
| 0x00 | NOP | |
| 0x01 | HALT | stop this slice; flag bit0 CHAIN: then start the program at `R[ra]` (see "HALT CHAIN") |
| 0x02 | LI | `R[rd] = w1` |
| 0x03 | ADDI | `R[rd] = R[ra] + w1` |
| 0x04 | LOOP | body = next `w1` instructions, executed `R[ra] + w2` times (0: skipped). Loops nest (depth 4); a body must not end on the same instruction as an enclosing body. |
| 0x05 | BAR | wait until every slice has reached a `BAR` |
| 0x06 | RLD | `R[rd] = f2i(T[R[ra]+w1])`, flag bit0 RAW: the word's bits, bit1 MUL: times `R[rb]+w2`, bit2 BOUND: `w3 <= R[rd] <= w4` (see "RLD") |
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

Fields: `sa = R[ra]+w1` (streamed rows, bytes), `ssa = R[rb]+w2` (the weights' scale rows,
bytes), `out = R[rc]+w3` (TMEM words), `N = w4[15:0]`, `KB = w4[23:16]`, `k0 = w4[31:24]` (the
first block of the MM within the scale rows), `rs = w5` (row stride, bytes), `ors = w6[15:0]`
(output row stride, words), `M = w6[23:16]`, `ab = w6[31:24]` (first ACT RAM block),
`srs = w7` (scale row stride, bytes). Flags: bit0 `UNIT` (no scale rows: `Sw = 1.0`, every
`dw = 0` and `m = 1`; nothing is read at `ssa`), bit1 `ACC` (accumulate into `out`), bit2 `RMAX`
(also write each output row's maximum: `T[out + M*ors + j] = fold(max, y[j][0..N-1])`, folded
from `n = 0` like the `RMAX` VOP; the softmax row max comes for free from the MXU epilogue), bit3
`ASCALE` (needs `UNIT` and `ACC`; `ssa` is then the TMEM address of M per-row factors and the old
accumulator is rescaled first: `y = T[out + j*ors + n] * T[ssa + j] + y` -- the
flash-attention correction step, done in the MXU epilogue), bits 5:4 `WF`, the streamed
weights' format ("Weight formats"), bit6 `PAIR` (4-bit weights at full rate; on a DUAL target int8 weights at twice the rate,
"Column reuse"),
bit7 `RSC` (row scales only: each scale row is the one word `Sw[n]`, every `dw = 0` and `m = 1`;
not with `WF = 3`). The streamed rows are D-byte aligned (`sa` and `rs` are multiples of D):
the MXU streams whole D-byte DRAM chunks. `N >= 1`, `KB >= 1` and `0 < M <= ACT_ROWS`; RMAX,
ASCALE and PAIR need `M <= MCOLS` (PAIR: `2*M <= MCOLS`), and an MM with `M > MCOLS` needs its
streamed row to fit the MXU's chunk FIFO (`KB` chunks for int8, `ceil(KB/2)` for 4-bit; the
board: 1024; a replayed row stays in the FIFO until its last group).

Each output is one exact integer: every product of a block's group (a sub-block of `D/4`
elements; NVFP4's weights: of `D/8`) is scaled by a power of two, `B`, that the weights' and
the activations' offsets give, and the fp32 epilogue runs once per output:

```
W = 16 (WF 0, int8), 11 (WF 1, 2: int4, fp4), 13 (WF 3, nvfp4)       the shift window
for n in 0..N-1:
  for j < M:
    acc = 0                                              an exact integer (|acc| < 2^47)
    for k in 0..KB-1:
      r = PAIR and k odd ? j + M : j                     the ACT row of block k
      for each group g of block k, b its sub-block (WF 3: g in 0..7, b = g/2; else g = b in 0..3):
        c   = sum_{i in g} ACT[r][(ab+k)*D + i] * w[i]   (exact; w: the elements of block k of row n)
        sh  = W - dw(n, k, g) - da[r][ab+k][b]
        acc = acc + (sh >= 0 ? m(n, k, g) << sh : 0) * c
    y = (i2f(acc) * Sw[n]) * (Sa[j][ab] * f32(2^-W))      three rounded mul (FTZ)
    if ASCALE: y = T[out + j*ors + n] * T[ssa + j] + y    (a rounded mul, then add)
    elif ACC:  y = T[out + j*ors + n] + y
    T[out + j*ors + n] = y
```

A group whose two offsets add up to more than `W` (`sh < 0`) does not count: its scale is under
`2^-W` of the row's largest weight scale times the row's largest activation scale (the formats'
scans measured at most 7e-5 of the groups, docs/mx_mxu.md 3.3). The bounds keep `acc` exact:
a group's `|c| < 2^19` (int8; 4-bit `< 2^16`), `B < 2^17`, at most `4 * 255` groups for int8.
An MM's rows `j` read `Sa[j][ab]` only: its blocks `ab .. ab+KB-1` (and, with PAIR, row `j + M`'s)
must hold the same `Sa`, as one QACT writes them (the ISA simulator raises otherwise).

#### Weight formats

`WF = 0` (int8) streams int8 elements, block `k` of row `n` the `D` bytes at `sa + n*rs + k*D`.
`WF = 1` (int4), `WF = 2` (FP4, E2M1) and `WF = 3` (NVFP4: E2M1 elements) stream 4-bit elements,
two per byte: block `k` of row `n` is the `D/2` bytes at `sa + n*rs + k*D/2` (two blocks per
D-byte chunk; a row's last chunk is half used when KB is odd), element `i` in the low nibble of
byte `i/2` for even `i`, the high nibble for odd `i`. A nibble `c` is the integer
`w = c - 16*c[3]` (int4, -8..7) or, for E2M1, twice its value: `c[2:0]` in
{0, 1, 2, 3, 4, 6, 8, 12} with `c[3]` the sign (so a stored E2M1 matrix carries half its scale).

The scale row of weight row `n` starts at `ssa + n*srs` with the row's fp32 scale `Sw[n]` (a
word), followed by the blocks' offsets and multipliers (little endian; the MM's block `k` is
the row's block `k0 + k`):

| WF | scale row | block `k`'s `dw` and `m` |
|---|---|---|
| 0 int8 | `Sw`, then one byte per block | `dw` = byte `4 + k0 + k`; `m = 1` |
| 1 int4, 2 fp4 | `Sw`, a reserved word, then a word per block | word at byte `8 + 4*(k0+k)`: `dw` = bits 7:0, `m` of sub-block `b` = bits `8+6b +: 6` (0..63) |
| 3 nvfp4 | `Sw`, three reserved words, then two words per block | byte `16 + 8*(k0+k) + g`, g < 8: an E4M3 scale `s` (s[7] = 0) of the group of `D/8` elements `g`: `E = s[6:3]`, `M = s[2:0]`, `m = E ? 8 + M : M`, `dw = 15 - max(E, 1)` |

So a weight's value is `w * m * 2^-dw * Sw[n]` (NVFP4: the E4M3 value `m * 2^(max(E,1) - 10)`
times `2^-5 Sw[n]`): a per-row fp32 scale and power-of-two offsets under it, MX style
(docs/mx_mxu.md). With `RSC` the scale row is `Sw[n]` alone (the K cache, one scale per token).
`ssa` and `srs` are multiples of 4 (int8, RSC), 8 (int4, fp4) or 16 (nvfp4). opentpu/quant.py
makes these rows (`quantize_mxu`): `int8` (Sw = the row's amax / 127, `dw` per block as the
quantizer's `d_g`), `fp4` / `int4` (the `m` that minimize the squared error), `nvfp4` (E4M3 per
16 under a per-row Sw); docs/quant.md and docs/mx_mxu.md compare them. OCP MXFP4 (E8M0 per 32)
is `fp4` with `m = 2^(e_b - e_min)` where a block's four exponents span at most 5, else `nvfp4`.

#### Column reuse

A 4-bit MM takes one block per cycle, half a streamed chunk. Flag bit6 `PAIR` (4-bit `WF` only,
`2*M <= MCOLS`) lets it take a whole chunk per cycle when the operand has at most MCOLS/2 rows:
the idle columns `M..2M-1` take the odd blocks. Output row `j` gets its even blocks from ACT
row `j` and its odd blocks from ACT row `j + M` (which `QACT DUP` fills with the same row), as
the loop above says: the result is the same integer, so PAIR changes no bit of the output.
Chunk `c` holds blocks `2c` and `2c+1`, whose scale words (WF 1, 2: 8 bytes; WF 3: 16) are read
together: `k0` is even and `ssa` and `srs` are multiples of 8 (WF 3: 16); with KB odd the last
chunk's second block word is read and ignored. ACC, RMAX and ASCALE are unchanged and write M
rows.

On a DUAL target (Config.DUAL: the slice's DRAM on four channels, port B reading two adjacent
chunks a cycle, rtl/mem/otpu_native_dram2.sv) `PAIR` also takes int8 weights (`WF = 0`): an
advance takes two chunks, block `2c` (columns `0..M-1`, ACT row `j`) and block `2c+1` (columns
`M..2M-1`, ACT row `j + M`), so an MM of at most MCOLS/2 rows (decode) streams 256 bytes a
cycle. The loop above is unchanged (the same integer); `k0` is even (the two blocks' offset
bytes are adjacent in a scale line); with KB odd the last advance takes one chunk. Elsewhere an
int8 `PAIR` is an error (the ISA simulator's; the encoder takes it).

On a QUAD target (Config.QUAD, with DUAL, MCOLS a multiple of four) a 4-bit `PAIR` of
`4*M == MCOLS` rows (not `UNIT`, not `RSC`) takes four blocks an advance, two chunks (the DUAL
port's pair): block `4a+g` in columns `g*M..(g+1)*M-1` from ACT row `j + g*M`, so a decode MM
streams 256 bytes of 4-bit weights a cycle, as int8's `PAIR`; the last advance of a row takes the
`KB mod 4` blocks left. The advance's block data (16 bytes, NVFP4 32, at an even `k0` as `PAIR`'s;
NVFP4's 16-byte aligned) may straddle two 32-byte scale lines: the MXU reads both (a row whose
first advance does costs a cycle of its own). Any other 4-bit `PAIR` takes two blocks (one
chunk); int8 `PAIR` takes two chunks as DUAL's. The result is the same integer. `QACT DUP` of `4*rows <= MCOLS` rows fills rows `r + rows`,
`r + 2*rows` and `r + 3*rows` (of more rows, `r + rows` alone), so the operand of either `PAIR`
is in place.

### QACT

`src = R[ra]+w1` (TMEM words), `rows = w2[7:0]` (at most ACT_ROWS), `ab = w2[15:8]`,
`KB = w2[31:16]`, `srs = w3` (source row stride, words). Flag bit1 `CSCALE`: every element is
first multiplied by a per-column scale, `x = T[src + r*srs + c] * T[w4 + c]` (this folds V's
per-token scale into P for free). Flag bit2 `RSCALE`: every element is first multiplied by a
per-row factor `T[w5 + r]`. With both, `x = (T[src + r*srs + c] * T[w5 + r]) * T[w4 + c]`
(RMSNorm's `x * r * gamma`). Each row `r < rows` of `KB` blocks is quantized as one row
("Quantization", its groups the sub-blocks of `D/4`): `ACT[r][(ab+k)*D + i] = q[k*D + i]`,
`Sa[r][ab+k] = S`, `da[r][ab+k][b] = d` of sub-block `b` of block `k`. Flag bit0 `ROW`: every
`d = 0` (the row's one scale). Flag bit3 `DUP` (`2*rows <= MCOLS`): row `r + rows` receives the
same bytes and scale records as row `r` in the same cycles, the operand layout of `MM PAIR`
("Column reuse"); on a QUAD target with `4*rows <= MCOLS` rows `r + 2*rows` and `r + 3*rows` too.

### QST

`src = R[ra]+w1` (TMEM), `dst = R[rb]+w2` (DRAM bytes), `sdst = R[rc]+w3` (DRAM bytes),
`rows = w4[15:0]`, `KB = w4[31:16]`, `srs = w5` (words), `drs = w6` (bytes), `es = w7`
(element stride, bytes). Flag bit0 `ROW`. Each row is quantized as one row ("Quantization",
its groups the D-blocks). Element `c` of row `r` goes to byte `dst + r*drs + c*es`. Scales: in
`ROW` mode the word `S` to `sdst + r*4` (an `RSC` scale row); otherwise row `r`'s int8 scale row
(WF 0, `k0 = 0`) to `sdst + r*(4 + 4*ceil(KB/4))`: the word `S`, then the bytes `d_k` (`k < KB`;
the pad bytes 0). The data and scale ranges of one QST must not overlap. On the board a 64-byte
DRAM beat a QST writes only part of is read, merged and written whole (`otpu_native_dram`): the
scoreboard counts every word of the QST's beats as written by it, and the host must not write
the other bytes of those beats while it runs. Flag bit1 `HALF` (`ROW` mode only): the scale is
still the whole row's, but only elements `c < KB*D/2` are written (a V^T append of a head half
as wide as its padded row, LFM2's 64 of 128, then writes its 64 real rows instead of 128
byte-strided ones; a quantizer without `HALF` writes the zero padding too, which nothing reads).

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
- DRAM state, in place or into a chunk-aligned `dst` apart from `src`, rows ≤ 256, cols 64..256;
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
e.g. a token's row in a 2.4 GB table (`tok * 9344`). `rd = 0` writes nothing. Flag bit2
BOUND: the value lies in `[w3, w4]` (unsigned), what the kernel knows of the word (an expert's
index below E, a slot of the expert pool, a row's place in a scratch); the compiler's wait
entries (`opentpu.waits`, "Wait counts") bound the footprints computed from the register by it
instead of taking any value, the ISA simulator stops on a value outside it, and the hardware
ignores the field.

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
- The scoreboard footprint: all of DRAM and `T[R[rb] + w2]` written. Older stores land before
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

### Wait counts (WAITC builds)

A build with `otpu_seq` WAITC (`SEQ_WC`, EXPERIMENTAL, docs/seq_vpu.md) does not find the
dependencies between instructions itself: each instruction carries a 64-bit **wait entry** that
says which older instructions it waits for. The machine model does not change: a program whose
wait entries are correct runs as if every instruction completed before the next started, the
units overlapping wherever the waits allow. A wrong wait entry is a wrong program (its results
are undefined), like an out-of-range address.

**Semantics.** An entry is two words: the counts' word, seven 4-bit counts `k_j = w0[4j+3:4j]`
(bits 28..31 are 0), and the VPU mask's word, `vmask = w1[30:0]` (bit 31 is 0). The units
are U0 DMA (`LD ST DSTEP STREAM WAITW`), U1 MXU (`MM`), U2 quantizer (`QACT QST`), U3 VPU
(`VOP`), U4 collective (`BAR GATHER RLD`); control instructions (`NOP HALT LI ADDI LOOP`) have
no entry that matters.
- `k_j > 0`, j = 0, 1, 2, 4: the instruction does not start (an `MM`: is not released to read
  the ACT RAM and write TMEM) until the `k_j`-th last instruction of unit Uj before it in program
  order has completed, and with it every older instruction of Uj. These units start their
  instructions in program order and complete them in order.
- `k_5`, `k_6`: the same for an `MM`'s weight stream (its DRAM reads), on U0 and U2.
- `k_3` is 0: the VPU's waits are the mask. Bit `r - 1` of `vmask` (r = 1..31): the instruction
  does not start (an `MM`: is not released) until the r-th last `VOP` before it has completed.
  The VPU starts the oldest of its instructions whose waits are met, so `VOP`s complete out of
  order: the mask names each `VOP` the instruction depends on (a `VOP`'s own included), not the
  nearest and all older.
- `k_j = 0`: no wait on that unit; an instruction's own unit needs none, except the VPU's (its
  mask): the DMA, the MXU, the quantizer and the collective unit start their instructions in
  order and finish one's memory accesses before the next one's begin.
- Nothing past 15 (a count) or 31 (the mask) is ever needed. On a unit that starts in order, an
  instruction 16 or more instructions of its unit back has completed: the sequencer's window
  holds 16 instructions, and that one and its 15 younger ones of the unit would all hold one. A
  `VOP` is dispatched only once the `VOP` 31 before it has completed, so every older `VOP` has
  too (a `VOP`'s window slot is freed when it completes, so younger ones can pass a `VOP`
  blocked on its producers; the hold binds only at 31 apart).

A correct wait entry waits, for every older instruction that may still run and whose footprint
(the TMEM, DRAM and ACT RAM ranges it may read or write: `otpu_pkg` `footprint()`) conflicts
with this instruction's (a read after a write, a write after a read, a write after a write,
across units too), for that instruction (a `VOP`) or that one or a younger one of its unit
(the other units). `BAR` conflicts with everything; `WAITW` writes all of DRAM.

**Encoding.** The wait entries are not part of the 8-word instruction. A program image for a
WAITC build is its instructions, padded to the next chunk (`D` bytes), then one 64-bit entry
per instruction (the counts' word, then the mask's); the loader (RUN, `HALT CHAIN`) reads them
into the sequencer's sidecar RAM after the instructions (`ld_n`, `R[rb]` count instructions,
as before).

**Host.** A WAITC build says so in CFG_UARCH bit24 (docs/observability.md), and its CFG_HASH
holds `SEQ_WC` (`opentpu.target.arch_hash`: in the hash only when on, so every other build's
hash stays and a host that writes no entries refuses the build). `device_config` sets the
Config's `SEQ_WC` from the bit; the host then loads every program with its entries
(`Board.load_program(addr, words, entries)`, `Board.load(addr, instructions)`;
`waits.loadable`) and refuses a program without them (`ConfigMismatch`); on a build without
the bit nothing changes. A Config with `SEQ_WC` gives each program its entries' room: the
program area after the image and each chain slot's half are `waits.area` (5/4 of IMEM's bytes),
and the chained programs are written with their entries (`generate.slot_words`). The entries
are computed once per program (`waits.entries`: its words, its arguments' bounds, `D`, `S` and
the compiler's source, which holds this pass) and kept beside the program cache's programs. The
window must be 16 or less (a target with `SEQ_WC` and a larger `WIN` is an error; the host
refuses such a bitstream).

**Compiler.** `opentpu/waits.py` `wait_counts(program)` computes them from the program alone:
an abstract execution with the registers as intervals (the run's arguments bounded by their
variables where the program says so, an RLD's value within its BOUND or else unknown), each instruction's footprint
over every value its registers may hold, and per unit the last 15 instructions' footprints
(the VPU's last 31), joined where paths can meet (a loop's exits, a count the program does not know); a pc's count
is the smallest over all its visits, its mask the union. It is conservative: an unknown address
conflicts with every address of its space.

**ISA simulator.** `isasim` runs programs as before and ignores wait entries (they cannot change
a correct program's result). With `OTPU_WAITS_CHECK=1` it checks each program's
`wait_counts()` against the run: every conflict with an older instruction that may still be
running must be waited for (`waits.Checker`). The RTL simulation checks the same in
`otpu_seq` (`$fatal` on a conflict on the window that no count or mask bit covers).

### GATHER

All slices must execute a `GATHER` with the same `dst` (the address `R[rb]+w2`, through any
register), `rows`, `cols`, `drs`, `seg`.
`src = R[ra]+w1`, `dst = R[rb]+w2`, `rows = w3[15:0]`, `cols = w3[31:16]`, `srs = w4`,
`drs = w5`, `seg = w6` (words). For every slice `s`, every `r < rows`, `c < cols`:
`T_all[dst + s*seg + r*drs + c] = T_s[src + r*srs + c]` is written into every slice's TMEM.
The source and destination ranges must not overlap.

GATHER is optional at S = 1, where it is a TMEM copy the compiler never emits
(`opentpu.language.all_gather` returns its input): a one-slice bitstream built with
`COLL_GATHER=0` runs BAR alone and clears CAPS2 bit0 (docs/observability.md). There a GATHER
moves no data and raises ERROR, and the host refuses to load a program with one
(`Board.load_program` raises `NoGather`).
