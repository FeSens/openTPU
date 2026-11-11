# openTPU ISA v1

This document is the contract between the compiler (`opentpu/compiler.py`), the bit-exact
instruction-set simulator (`opentpu/isasim.py`) and the RTL (`rtl/`). All three must agree on
every bit written to TMEM or DRAM.

## Machine model

- `S` slices. Each slice has: a sequencer with 16 x AW-bit registers (`R0` reads as 0; AW 32
  but on the BCU1525's 36, "Address bits"), an
  instruction memory, a private DRAM, a TMEM, an ACT RAM, an MXU, a VPU and a quantizer.
  Slices share nothing except the collective unit (`GATHER`, `BAR`).
- `D` = MXU depth = quantization block size (bytes / int8 elements). Default 32 in tests,
  128 in the design. `MCOLS` = MXU columns (8). `ACT_ROWS` = ACT RAM rows =
  max stationary rows of one MM (at least MCOLS; default MCOLS). With `ACT_ROWS > MCOLS` the MXU
  consumes each streamed chunk once per group of MCOLS rows ("replay"): one DRAM pass serves all
  M rows, at ceil(M / MCOLS) cycles per chunk. Results do not depend on MCOLS or ACT_ROWS.
- **DRAM**: byte addressed (`DRAM_BYTES`, a power of two of at most 2^AW), little endian,
  accessed as 32-bit words. All word accesses and all
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

### Address bits

`Config.AW` (a target's `[arch] AW`, docs/targets.md) is the width of the registers and of
every address the sequencer resolves: 32 on every target but the BCU1525 at WG 8 (36: 32 GiB
now, 64 GiB later, the same programs).

- **Registers** hold AW bits; `LI` and `ADDI` take a 64-bit immediate `{w2, w1}` (`LI R[rd] =
  {w2, w1}`, `ADDI R[rd] = R[ra] + {w2, w1}`, both mod 2^AW). At AW 32 the hardware ignores
  `w2` and the compiler leaves it 0, so a 32-bit machine's programs are as before; above it a
  negative immediate has `w2 = 0xFFFFFFFF` (`isa.imm64`). Every other immediate is one 32-bit
  word: `isa._w` refuses a value that does not fit instead of truncating it.
- **Addresses**: a field written `R[x] + w` in `w1`, `w2` or `w3` (every DRAM address: LD / ST,
  WAITW, DSTEP's state, STREAM's src and dst, MM's `sa` and `ssa`, QST's `dst` and `sdst`; and
  the TMEM addresses there) is `R[x] + w` mod 2^AW, `w` zero-extended: an address at or above
  4 GiB is a register's (a base register: docs/compiler.md "Addresses above 4 GiB") plus an
  offset below 4 GiB; a negative offset is no longer the register minus it. The fields in
  `w4..w7` that add a register (STREAM's `vec`, `x`, `k`; VOP's `w7`; QST GRP's first row) and
  LOOP's count stay 32 bits, and WAITW compares the low 32 bits of `R[rc] + w3`.
- **RLD**'s word goes into `R[rd]` zero-extended (a negative f2i, -8, is 2^32 - 8 there), and
  **RLD MUL**'s product is AW bits ("RLD"): a resident expert's address on the card is `ADDI
  (RLD MUL(id, expert bytes), pool base)`, no directory (docs/offload.md's host-served pool
  keeps its 32-bit directory, its slots below 4 GiB).
- **Below 4 GiB**: the programs (the host's PROG_ADDR, HALT CHAIN's `R[ra]`), the run's
  arguments (`R8..R15` from 32-bit host registers, zero-extended), the mailboxes, the I/O area,
  the KV caches and the recurrent states, so their addresses stay `R0 + w` or an argument's
  register plus `w`; the weights and the expert pools go above (docs/compiler.md).
- **The wait entries** (`opentpu.waits`, "Wait counts") take their intervals mod 2^AW: with
  SEQ_WC there is no hardware footprint to catch a wrapped range.

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

The split's lo part (flag `LO` of QACT and QST, a machine with SPLITQ: CAPS2 bit17,
`Config.SPLITQ`; refused without it): the same `S`, `inv` and `d_g`, the same product `p`, and

```
p     = x[i] * (inv * f32(2^d_g))
lo[i] = sat127(rint(256 * p) - 256 * q8(p))     (rint half to even, of the exact 256 p; NaN 0)
S_lo  = S * 2^-8                                (a rounded mul: 0 for S below 2^-118)
```

written in place of `q` and `S` (the offsets `d_g` unchanged), so an operand is two int8
halves, `x ~ (q + lo / 256) * S * 2^-d_g` to `2^-9` of `S * 2^-d_g` (`2^-8` where `lo`
saturates: `p` within `2^-9` of a half): an MM over the hi blocks and one over the lo blocks
(ACC; MM's blocks share their `Sa`) gives the product of the split operand (chunked DeltaNet's
matmuls, [deltanet_chunk.md](deltanet_chunk.md)). `fp32.q8lo`, `fp32.quantize_mx(lo=True)`.

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
| 0x02 | LI | `R[rd] = {w2, w1}` mod 2^AW (AW 32: `w1`, see "Address bits") |
| 0x03 | ADDI | `R[rd] = R[ra] + {w2, w1}` mod 2^AW (AW 32: `R[ra] + w1`) |
| 0x04 | LOOP | body = next `w1` instructions, executed `R[ra] + w2` times (0: skipped). Loops nest (depth 4); a body must not end on the same instruction as an enclosing body. |
| 0x05 | BAR | wait until every slice has reached a `BAR` |
| 0x06 | RLD | `R[rd] = f2i(T[R[ra]+w1])`, flag bit0 RAW: the word's bits, bit1 MUL: times `R[rb]+w2`, bit2 BOUND: `w3 <= R[rd] <= w4` (see "RLD") |
| 0x07 | WAITW | DMA: wait until `M32[R[ra]+w1] & w4` compares (flags[1:0]: EQ, NE, GE) with `R[rc]+w3`, then `T[R[rb]+w2]` = the word's bits (see "WAITW") |
| 0x10 | LD | DRAM -> TMEM, `n = w3` words: `T[R[rb]+w2+i] = M32[R[ra]+w1+4i]` |
| 0x11 | ST | TMEM -> DRAM: `M32[R[ra]+w1+4i] = T[R[rb]+w2+i]` for `i < w3` |
| 0x12 | DSTEP | one Gated DeltaNet head step on a DRAM state, run by the DMA (see below; `Config.DSTEP`, CAPS bit6) |
| 0x14 | LDA | DMA: rows a QACT `ST` stored back into ACT RAM, unchanged (see "Quantize once"; `Config.QONCE`, CAPS2 bit24) |
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
`srs = w7` (scale row stride, bytes; with `w7` bit31 `SPL` its bits 19:0, "Split terms"). Flags: bit0 `UNIT` (no scale rows: `Sw = 1.0`, every
`dw = 0` and `m = 1`; nothing is read at `ssa`), bit1 `ACC` (accumulate into `out`), bit2 `RMAX`
(also write each output row's maximum: `T[out + M*ors + j] = fold(max, y[j][0..N-1])`, folded
from `n = 0` like the `RMAX` VOP; the softmax row max comes for free from the MXU epilogue), bit3
`ASCALE` (needs `UNIT` and `ACC`; `ssa` is then the TMEM address of M per-row factors and the old
accumulator is rescaled first: `y = T[out + j*ors + n] * T[ssa + j] + y` -- the
flash-attention correction step, done in the MXU epilogue), bits 5:4 `WF`, the streamed
weights' format ("Weight formats"), bit6 `PAIR` (4-bit weights at full rate, "Column reuse"),
bit7 `RSC` (row scales only: each scale row is the one word `Sw[n]`, every `dw = 0` and `m = 1`;
not with `WF = 3`). The streamed rows are D-byte aligned (`sa` and `rs` are multiples of D):
the MXU streams whole D-byte DRAM chunks. `N >= 1`, `KB >= 1` and `0 < M <= ACT_ROWS`; RMAX,
ASCALE and PAIR need `M <= MCOLS` (PAIR: `2*M <= MCOLS`; not with weight groups; RMAX and
ASCALE with weight groups only under GALL, "Every MM in groups"), and an MM
with `M > MCOLS` needs its streamed row to fit the MXU's chunk FIFO (`KB` chunks for int8,
`ceil(KB/2)` for 4-bit, `KB` a group; the board: 1024; a replayed row stays in the FIFO until
its last group).

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
rows. A grouped 4-bit operand ("Weight groups") takes a chunk a cycle without it: PAIR is for
rows of one.

#### Weight groups

`Config.WG` (default 1) is the MXU's weight lanes: a WG > 1 machine's MXU reads a group chunk of
`P = WG x D` bytes a step, one `D`-byte lane to each of its WG column pairs' inputs
([weight_groups.md](weight_groups.md)). MM's rd field (MM has no rd register, and its flag bits are all taken)
holds `LG`: the streamed operand is in groups of `R = 2^LG` rows, `R` 1 (the layout above), `WG`
(int8, `WF = 0`) or `2 x WG` (4-bit, `WF` 1 to 3: two rows a lane). Row `n` is row
`j = n mod R` of group `g = n div R`, and a group's chunk `c` is its rows' K-block `c`, at
`A = sa + g*rs + c*P`:

```
int8: lane l of chunk c   A + l*D: row l's block c, D bytes          rs: the group stride
4-bit: lane l of chunk c  A + l*D: byte i = row 2l's element i of block c (bits 3:0) and
                          row 2l+1's (bits 7:4)
group g's scale area      a = ssa + g*srs                            srs: the group scale stride
  row j's Sw              the word at a + 4j
  its block kk's data     a + SWP + (kk*R + j)*SB                    kk = k0 + k, as above
```

`SB` is a block's data bytes (int8 1, int4 / fp4 4, NVFP4 8; RSC: none) and `SWP` the Sw words'
part, padded: `4R` (int8, RSC, int4, fp4), `8R` (NVFP4); at `R = 1` the scale row's header (4,
8 or 16 bytes). Each field holds what it holds above, and at `R = 1` every formula is the one
above, byte for byte. A chunk is one K-block of the group's rows whatever the format: 4-bit
groups have twice int8's rows and the same chunks, so a step gives every column two rows'
elements and the MXU takes a 4-bit chunk a cycle at any M (no PAIR). With `R > 1`: `R` is the
machine's (`WG` int8, `2 WG` 4-bit); `sa` and `rs` are multiples of `P`, `ssa` and `srs` of
`SWP`; the `ceil(N / R)` groups the MM touches lie in DRAM whole (rows past N in the last group
are read and not computed: `opentpu/quant.py` `group_rows` pads a matrix with zero rows, N to a
multiple of 16 for 4-bit at WG 8); `k0` is any block; PAIR is not taken, nor RMAX and ASCALE
(but under GALL). ACC, UNIT and RSC are unchanged, and so are the results: the grouping moves
bytes, not a bit of the output. Everything an instruction writes (the KV cache, V^T, tables) is
in rows of one (but QST GRP's, under GALL); `quant.group_rows` and `group_scales` make the
images' grouped matrices (`quant.grp_r` the rows of a format's group).

With `Config.CHASH` (the memory's channel hash, CAPS bit7, as a `WG > 1` machine has it:
`opentpu.target.config`) each group chunk's lanes are stored in the hash's order: the chunk at
`A` holds lane `l`'s `D` bytes at `A + (l ^ h)*D`, `h` the XOR fold of `A div P` into
`log2 WG` bits (bit `i` into bit `i mod log2 WG`: `quant.chunk_fold`). A group step is one
`D`-byte beat on each of `WG` channels and the hash puts the chunk's beat `l` on channel
`l ^ h`, so channel `c` holds lane `c` of every chunk: the MXU's lane `c` takes its channel's
beats as they arrive, with nothing to permute them ([native_channels.md](native_channels.md)). Without CHASH
`h = 0`. `quant.hash_rows` makes the order (`grouped(..., at=)`); it is its own inverse at one
address, and a grouped matrix's bytes depend on where it is (moving one is `hash_rows` at the
old address, then at the new). The scale areas are not permuted. The footprint of a grouped MM
(`waits.footprint`, `otpu_pkg`) is the rows' with `N - 1` replaced by `(N - 1) div R` and both
lengths times `R` (a 4-bit group's data is half that: the footprint covers it).

#### Every MM in groups (GALL, EXPERIMENTAL)

`Config.GALL` (default off; in `arch_hash` only when on at `WG > 1`) at `WG > 1`: every MM is in
groups (`R > 1`; the ISA simulator refuses an MM of rows of one), so the MXU streams only P-byte
group chunks (port W) and its scales (port A): no operand uses the natural D-byte chunks (port
B). The operands written at run time are grouped too: the KV caches (`compiler.KVDesc` `wg`: K
in groups of WG tokens, V^T's tiles in groups of WG dimensions, their scales in group areas),
appended by QST GRP; the tables and records `kernels.gather` reads (each row's blocks a grouped
`[K / D, D]` matrix, one chunk a group: int8 groups of WG blocks, the row's bytes as they are,
gather scales in areas of WG words; 4-bit groups of 2 WG blocks, lane l blocks 2l and 2l + 1,
each block's scale entry its chunk's Sw and its own word; `gather.table_bytes`, `record_bytes`),
gathered with int8's one-hot operand in every format; the experts' slots (`moe.ExpertFormat`
`grp`, their lanes hashed as the host writes them). Grouped MMs take RMAX and ASCALE (the
attention's scores and P.V): the results are the formulas above, in `n` order. The RTL at
`WG > 1` implements GALL alone (otpu_mxu has no rows of one there, otpu_quant takes QST GRP): a
target's Config and the board's derive `GALL` from `WG > 1`, and a WG 1 machine has none
([weight_groups.md](weight_groups.md) section 6).

#### K tail (KT)

`Config.KT` (CAPS2 bit5; default off, in `arch_hash` only when on): a grouped MM (`R > 1`) whose
K ends in the middle of its last K-block streams that block's halves of two groups as one chunk.
Under GRP `rs` is a multiple of `P`, so its bit 0 is free: `rs[0] = KT`, and `rs - 1` is then the
stride of pairs of groups. Pair `p` (groups `2p`, `2p + 1`) at `A = sa + p*(rs - 1)`:

```
A + c*P,              c < KB-1    group 2p's chunk c
A + (KB-1)*P                      the tail chunk T_p: lane l bytes 0 .. D/2-1 group 2p's lane l
                                  bytes 0 .. D/2-1 of block KB-1, bytes D/2 .. D-1 group 2p+1's
A + (KB+c)*P,         c < KB-1    group 2p+1's chunk c
```

Group `2p` takes its chunks as without KT: block `k` against ACT block `ab + k`, T_p as block
`KB-1`. Group `2p + 1` takes T_p first, against ACT block `ab + KB-1` with its halves swapped
(element `i` at `i ^ D/2`, its sub-block offsets `da[b]` at `b ^ 2`), then its chunks `c` against
ACT blocks `ab + c`; its scale entries are read in that order (entry `k` for its `k`-th chunk):
the image stores the tail block's first, its multipliers of sub-blocks 0 and 1 also at 2 and 3
(`quant.kt_scales`). The upper half of ACT block `ab + KB-1` is zero (QACT of a row of K
elements padded with zeros: the ISA simulator raises otherwise), so each half of T_p adds zero
to the other group's sums, and every output is the MM's over the padded groups bit for bit
(`quant.kt_rows` makes the pairs from `group_rows`' groups; the channel hash permutes T_p as any
chunk at its address). A pair is `2 KB - 1` chunks, one less than its groups: the half blocks of
zeros are not streamed. KT needs `k0 = 0`, an int8, int4 or fp4 operand and `rs - 1 >= (2 KB - 1)
P` (so the footprint above, with `rs` as written, covers the reads); an odd group count's last
pair is group `2p` alone. A machine without KT raises at `rs[0]`. The compiler pairs a grouped
operand whose K is `D*n + D/2` only on a `Config.KT` machine (the images of every other machine,
and of every other K, are as before).

#### Split terms (SPL)

`Config.SPLMM` (CAPS2 bit25, otpu_mxu SPL; default off, in `arch_hash` only when on; needs the
wait entries, SEQ_WC, and groups, WG > 1): `w7` bit31 `SPL` takes a split product's terms in one
MM. The operands are split ("Quantization", SPLITQ): the activations' hi part in ACT block `ab`
and lo in `ab + 1` (QACT HL), the weights' hi rows at `sa` and lo rows `HLD = w7[30:20]` KiB
after them (QST HL), each one K-block (`D`) a row; `srs` is then `w7[19:0]`. A lo part is
`q8lo` under its hi part's scale times `2^-8` with hi's offsets, so the MM reads hi's scales
alone (`Sa[j][ab]`, the weights' `Sw` at `ssa`) and adds 8 to each lo term's offsets. `KB` is
the advances, each a block of the weights' rows against an ACT block:

```
KB = 3 (a split's three terms: hi.hi + lo.hi + hi.lo)    KB = 2 (hi + lo against one block)
  k 0: ACT ab,     rows at sa                              k 0: ACT ab, rows at sa
  k 1: ACT ab + 1, rows at sa,       da + 8                k 1: ACT ab, rows at sa + HLD, da + 8
  k 2: ACT ab,     rows at sa + HLD, da + 8
```

Each output is the formula above with those blocks (`ACT[r][(ab + a_k)*D + i]`, the rows' block
at `sa` or `sa + HLD`: group `g`'s chunk at `sa + g*rs` and `sa + HLD + g*rs`, lanes by their own
address's fold with CHASH), `sh = W - dw - (da[r][ab + a_k][b] + e_k)` (`e_k` 8 on a lo term,
else 0; `dw = 0` under RSC) and one fp32 epilogue with `Sa[j][ab]`. A term whose `sh < 0` adds
nothing (a lo term at `da >= 9`: under `2^-17` of its row's largest). SPL needs `KB` 2 or 3,
groups (`R > 1`), int8 `RSC` weights, `k0 = 0`, `HLD > 0` and a multiple of `P`, `ab + KB - 1
<= ACT_BLOCKS`, and not `UNIT`, `PAIR`, KT, `RMAX` or `ASCALE`. Its footprint is the two parts'
`[sa, sa + (n-1)/R*rs + P)` and the same `HLD` after, and ACT blocks `ab .. ab + KB - 2`
(opentpu.waits; the sequencer's footprint scoreboard has the MM's as without SPL, so a machine
without the wait entries has no SPL). It is not the three (two) MMs' results bit for bit: their
sum is rounded once, and a lo term drops where its offset reaches `W - 8` rather than `W`.

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
("Column reuse"). Flag bit4 `LO` (SPLITQ): the split's lo part (`lo`, `S_lo`; "Quantization").
Flag bit5 `HL` (a machine with QHL: CAPS2 bit21, `Config.QHL`, which needs SPLITQ and the wait
entries, SEQ_WC; refused without it): both halves from one read of the rows, `q` and `S` into
blocks `ab .. ab+KB-1` and `lo` and `S_lo` into `ab+KB .. ab+2*KB-1` (the same offsets `d`), the
bytes and records of a QACT and a QACT `LO` at `ab + KB`; `KB*D <= QHL_ELEMS` (8192: the
quantizer's row buffer), `ab + 2*KB <= ACT_BLOCKS`, not with `LO` or `DUP`. Its footprint is the
`2*KB` blocks (opentpu.waits; the sequencer's footprint scoreboard has `KB`, so a machine
without the wait entries has no HL). Flag bit6 `ST` (a machine with QONCE: CAPS2 bit24,
`Config.QONCE`; refused without it; not with `HL`): each row's bytes and records also go to DRAM
("Quantize once"); `w3` is then the DRAM row base `R[rc] + w3`, the source row stride moves to
`w6` and `w7` is the DRAM row stride.

### Quantize once

A machine with QONCE (`[arch] QONCE = true`, `Config.QONCE`, CAPS2 bit24; it needs the wait
entries, SEQ_WC) keeps a quantized operand in DRAM and loads it back into ACT RAM without
quantizing it again: the expert-major prefill's router QACT stores each token's row once, and
every expert run that takes the token loads it (opentpu.llm.moe em_q). A row in DRAM is
`qrow_bytes(KB, D) = KB*D + 8*KB` bytes: its `KB` blocks' `D` bytes as ACT RAM holds them, then
`KB` records of `QREC = 8` bytes, block `k`'s at `KB*D + 8k`: its scale `S` (fp32, little-endian)
and its four sub-block offsets `d` (a byte each).

- **QACT flag `ST`**: as QACT, and row `r`'s bytes and records to DRAM at `R[rc] + w3 + r*w7` in
  the same pass (`DUP`'s copies are not stored).
- **LDA** (0x14, DMA): `R[ra] + w1` the first DRAM row, `rows = w2[7:0]`, `ab = w2[15:8]`,
  `KB = w2[31:16]`, `w3` the DRAM row stride, `w4` the first ACT row: DRAM row `r` into ACT row
  `w4 + r` at blocks `ab .. ab+KB-1`, its bytes, scales and offsets exactly the QACT's.
  `w4 + rows <= ACT_ROWS`, `ab + KB <= ACT_BLOCKS`.
- **Alignment** (`QALIGN = 64`): QACT `ST`'s and LDA's first DRAM row and, with more than one
  row, their row stride are multiples of 64 bytes, the stride at least a row's bytes (the ISA
  simulator refuses others).
- **Footprints** (the wait entries, opentpu.waits / otpu_pkg): QACT `ST` also writes DRAM from
  its first row for `(rows-1)*w7 + qrow_bytes`; LDA reads that much DRAM from `R[ra] + w1` (`w3`
  the stride) and writes ACT RAM blocks `ab .. ab+KB-1`.

The RTL: otpu_quant `QONCE` writes the bytes through the quantizer's SW port as pass 1 makes
them (16 bytes a cycle: `QUANT_SWW` 4 with `4 x SWW = ULANES x QUANT_PW`) and the records from a
32-entry FIFO in the port's free cycles (pass 1 holds while it is full); otpu_dma `LDA` reads the
rows through LD's path, row after row, and writes ACT RAM through a 4-entry queue at its second
write port (otpu_actram `DPORT`), which yields to the quantizer: the DMA writes only in a cycle
with no quantizer write at the RAM's write stage, and its reads stall while the queue is full.

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
Flag bit7 `LO` (SPLITQ): the split's lo part (`lo`, `S_lo`; "Quantization"), in every mode.

Flag bit2 `GRP` (GALL at `WG > 1` only, "Every MM in groups"): the rows go into a grouped int8
matrix at `dst` (a multiple of `P`; groups of `R = 2^LG` rows, `LG` the flags' bits 6:4 and `R`
the machine's `WG`, chunks of `P = R x D` bytes, lane `l` at `(l ^ h)*D` with CHASH, as MM reads
it): rows `n = R[rd] + w6 + r` (`drs` is the first row; rd is the register). Element `e` of
row `n` goes to group `n div R`'s chunk `e div D`, lane `n mod R`, byte `e mod D`, the group
stride `KB*P`; row `n`'s scales into its group's area at
`sdst + (n div R)*es` (`es`: the areas' stride, a multiple of 4): `S` the word at `4*(n mod R)`,
the bytes `d_k` at `4R + k*R + (n mod R)` (`ROW`: `S` alone, an RSC area). Flag bit3 `COL`
(with `GRP` and `ROW`): the rows are the matrix's columns (a V^T append): element `e` of row `n`
goes to the matrix's row `e` (group `e div R`, lane `e mod R`), column `n` (chunk `n div D`,
byte `n mod D`), the group stride `es` (a multiple of `P`); `S` to `sdst + r*4` as in `ROW` mode.
The data and scale bytes must not overlap, nor two elements' bytes. The footprint
(`waits.footprint`) of a GRP QST is every group (COL: every chunk of columns) and scale area its
rows can reach from the first row's, as `R[rd]`'s values bound it.

`w5` bit31 `HL` (a machine with QSTHL: CAPS2 bit22, `Config.QSTHL`, which needs SPLITQ, the wait
entries SEQ_WC, `WG > 1` and the quantizer's row buffer for QST rows, QUANT_QSTB; refused
without it): both halves of `GRP` rows in `ROW` mode from one read, the hi part (`q`, `S`) as
above and the lo part (`lo`, `S_lo`) `HLD = w5[30:20]` KiB after it, its bytes at `dst + HLD`
and its scale words at `sdst + HLD`: the bytes and words of a QST and a QST `LO` whose `dst` and
`sdst` are `HLD` past the first's (with CHASH, lo's chunks take the lanes of their own chunk
index's `h`). `srs` is then `w5[19:0]`; `HLD > 0` and a multiple of `P`; `KB*D <= QHL_ELEMS`
(8192: the row buffer); not with `COL`, `HALF` or `LO`. Its footprint is the two parts'
(`waits.footprint`). The chunked DeltaNet (`kernels.dn_chunk`) stores its split operands and
states this way where lo's bytes and scales sit the same multiple of `P` after hi's (its
scratch's layout), one read of the rows instead of two.

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
| 21 | SIGMOID | recip(add(exp2(mul(A, B)), 1.0)), B in any bmode (`Config.VSIG`, see below) |
| 22 | SILU | mul(A, SIGMOID's result) (`Config.VSIG`) |
| 23 | MAC | T[dst + r*drs + c] + A * B, in place; B per column or scalar (see below) |

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
dst; `B(r)` is read with every element and must not be written by an earlier element. A target
may build its VPU without OUTER (`[arch] OUTER = false`, Config.OUTER: the BCU1525's weight-group targets', whose DeltaNet
runs DSTEP; CAPS2 bit1, docs/observability.md): the compiler and the ISA simulator refuse it
there, and the RTL stops a simulation on one. Such a VPU with MAC may still run OUTER of decay
one (DONE) alone (`[arch] OUTER1 = true`, `Config.OUTER1`, CAPS2 bit23; needs VMAC; default off,
in `arch_hash` only when on): on MAC's datapath, `add(mul(B(r), Cv(c)), dst)`, which is the
OUTER's `add(mul(dst, 1.0), mul(B(r), Cv(c)))` bit for bit (the product by 1.0 flushes dst as the
add flushes its addend, and the add commutes). The chunked DeltaNet core's `T = (I - NL)^-1` takes
its rank-one steps so (`kernels/dn_chunk.py`); an OUTER with a decay word is refused there as
without OUTER.

**SIGMOID** and **SILU** (`Config.VSIG`, `[arch] VSIG = true`, CAPS2 bit19; default off, in
`arch_hash` only when on; DSTEP, 8 lanes, D >= 32): the logistic function and SiLU in one
VOP, bit for bit the VOPs MUL (A, B), EXP2, ADD 1.0 and RECIP (then MUL by A for SILU), each step
rounded and flushed as that VOP's. B is -log2(e) for `lib.sigmoid` and `lib.silu` (B_SCALAR), 1.0 for
`lib.silu_l2` (an input already -log2(e) times: SILU's `A * 1 + -0` is EXP2's own first step) and for
`lib.gelu_tanh`'s exponent, -2 log2(e) / c for `lib.softcap`. On the VPU each is four passes of the
composite loop at VPU_CNS 5 and six at 3 (otpu_se_comp SIG: EXP2's eight steps from the product,
then RECIP's), where the chain takes seven (SILU) or six (SIGMOID) at VPU_CNS 5. Without VSIG the
compiler emits the chain and the ISA simulator refuses them; the RTL stops a simulation on one.

**MAC** (`Config.VMAC`, `[arch] VMAC = true`, CAPS2 bit20; default off, in `arch_hash` only when
on) is elementwise and in place, `T[dst + r*drs + c] = add(T[dst + r*drs + c], mul(A(r, c),
B(c)))`: the rounded product, then the rounded add, so a MAC is bit for bit the MUL into a temp
and the ADD of dst and it (`isa.mac`; the compiler's `ol.mac(acc, a, v[None, :])`: the DeltaNet
convolution's taps in a pass each). B is per column (bmode 2, `cols <= 256`: `T[b + c]`, read
before anything is written, so it may overlap dst) or the scalar `imm` (bmode 3); bmode 0 and 1
are refused, and so are flags. A is read with every element and must not be written by an
earlier element (A may be dst itself, element for element). The footprint has dst read and
written, A and B's columns read. On the VPU a per-column MAC first reads B into OUTER's column
buffer (at least two chunks), then reads A on port A and dst on port B. On a configuration
without MAC the compiler and the ISA simulator refuse it, and the RTL stops a simulation on one.

**FSUM** (flag bit 2; `Config.FSUM`, `[arch] FSUM = true`, CAPS2 bit16; default off, in
`arch_hash` only when on): an EXP2SUB also computes the row sums of its results,
`T[imm + r] = isum_64(Y(r, 0..cols-1))` for `Y(r, c)` the result written at
`T[dst + r*drs + c]`: RSUM's sums of those words, bit for bit (`isa.exp2sub_sum`; a softmax's
denominator `sum(p)` without a pass of its own: attention's `l = l * alpha + sum(p)`, the compiler's
`ol.exp2(s - m, rowsum=True)`). `imm = R[rd] + w7` is the sums' address here (rows in consecutive
words), so B is not a scalar (bmode 0, 1 or 2); `rows <= 256` (the VPU buffers them, as RDOT's
row sums) and `cols >= 64` (every partial takes a term; the compiler sums with RSUM below). The
sums must not overlap the results. EXP2's results are never negative, -0 or NaN, so the +0 terms
that pad RSUM's chunks change no partial: the VPU adds each result into its partial as it is
written, folds a row's 64 partials on RSUM's tree once its last result is in, and writes the
rows' sums after the last one, while the VOPs after it run.

The EXP2SUB is done at its last result, as without the flag; its sums are written by the time its
**take** is done: an RSUM with flag FSUM (`isa.fsum_take`: dst the sums' words, `T[dst + R[ra] +
r]` with drs 1, A the EXP2SUB's results), which reads and computes nothing. Takes go in the order
of their EXP2SUBs (an FSUM may come before the take of the one before it), each after its own, and
every FSUM has one before the HALT. Between an FSUM and its take the sums' words are undefined:
nothing may read or write them (the sequencer fences them only at the take: its footprint is
RSUM's, the sums written; the EXP2SUB's has both ranges). The ISA simulator writes NaNs there at
the EXP2SUB and the sums at the take, and refuses a take with no FSUM's sums pending or at other
words and a HALT with sums not taken. On the VPU, no reduction, FSUM, take or stream begins while
an FSUM's sums are pending; the elementwise VOPs do. On a configuration without FSUM the compiler
and the ISA simulator refuse the flag, and it is clear on every other function.

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
written; `o` is written last. The scoreboard footprint: DRAM `[dram, dram + 4 rows cols)`
written (2 rows cols with flag bit 3, ST16: the state in 16 bits, "ST16" below), TMEM
`[qk, qk + 2 cols)`, `[v, v + rows)` and `{g, g + gs}` read, `[o, o + rows)` written.
A bitstream without it leaves CAPS bit6 clear; the compiler then emits the VOP sequence
(`Config.DSTEP = False`, the default of `board_config`; the host takes it from CAPS through
`device_config`). Its slice takes DSTEP and STREAM as illegal instructions: the run stops with
ERROR (otpu_seq STREAMS; before fix-board the DMA waited for the missing stream engine forever).

Flag bit 1 (PAIR; `Config.DSTEP2`, CAPS2 bit2, `[arch] DSTEP2 = true` in the BCU1525's weight-group targets): two heads in
one DSTEP, one stream for a pair of value heads. Rows `r < rows / 2` are head a's and the rest
head b's, whose state follows head a's in DRAM; `rows` is a multiple of 16 (each head's rows
whole row groups of the stream engine) and `cols` 128. Head b's `e = T[g + 1]` and
`beta = T[g + gs + 1]`; its `q | k` is head a's, or with flag bit 2 (QKB) `T[qk + 2 cols + c]`
(two value heads on two key heads). Bit for bit head a's DSTEP and then head b's
(`tests/test_vops.py::test_dstep_pair_is_two_dsteps`, `test_dstep_pair_rtl_bit_exact`). Its
footprint reads `[qk, qk + 4 cols)` with QKB and `[g, g + gs + 2)`. The compiler emits it with
`Config.DSTEP2` alone (`ol.deltanet_step2`: else a DSTEP a head) and the ISA simulator refuses it
without; a build without it (the cards') clears CAPS2 bit2, and its DMA in simulation stops on a
pair. The fill reads head b's q, k, e and beta after head a's (twice q and k: 32 more TMEM reads
of 8 words), and the pair saves the second stream's setup, fill and pipeline drain.

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

### ST16

A build with `[arch] ST16` (`Config.ST16`: `"e6m9"` or `"bf16"`; CAPS2 bits 14:13 = 1 or 2, 0
without; otpu_dma `ST16`) keeps DSTEP's and STREAM's DRAM state in 16 bits a value; the engine
computes in fp32 as before. DSTEP flag bit 3 (ST16, `isa.F_DST16`) and STREAM flag bit 4
(`isa.F_SST16`) choose it per instruction: the DMA widens each value to fp32 as it reads the
state, and rounds the step's new state to the format as it writes it, in place or to `dst`. Bit
for bit the fp32 instruction on the widened state, then its written state rounded
(`tests/test_st16.py::test_dstep_st16_is_the_fp32_step_rounded`,
`test_stream_st16_is_the_fp32_stream_rounded`); `o` and the row outputs come from the fp32 values,
before the rounding. Value (r, c) is the 16 bits at `dram + 2 (r cols + c)`, so a row is 2 cols
bytes, whole DRAM chunks, and the footprint is `[dram, dram + 2 rows cols)`.

On a build with `[arch] ST16S` as well (`Config.ST16S`; CAPS2 bit18; otpu_dma `ST16S`; refused
without it), STREAM's flag bit 5 (ST16R, `isa.F_SST16R`) keeps the source alone in 16 bits and
writes `dst` in fp32; bit 6 (ST16W, `isa.F_SST16W`) reads an fp32 source and writes `dst` in 16
bits. Each
end has its own size (2 or 4 rows cols bytes), and the two must not overlap: in place is
refused, as is a `dst` within the other end's bytes. Bits 5 and 6 together mean bit 4. A
chunked prefill uses them at a pass's ends: decode's 16-bit state widened to an fp32 copy for
the pass, and the pass's fp32 state rounded back (`ol.state_copy`: a pass-through STREAM over
zeros, so the values widen exactly or round as ST16 stores them; -0 becomes +0 and fp32
subnormals flush). The DMA reads by the source's size and writes by the destination's
(otpu_dma `ds_hr`, `ds_hw`; at SE_R > 1 each side has its own chunks a row). The tests:
`tests/test_st16.py::test_stream_st16_one_side`, `test_state_copy_converts`, and in
`test_st16_rtl_bit_exact` a round trip through an fp32 copy.

The formats, rounded to nearest even from the fp32 bits (`isasim.st16_pack` / `st16_f32`,
otpu_dma `st16_pack` / `st16_x`):
- **e6m9** (code 1): sign, 6 exponent bits with bias 31, 9 mantissa bits. A magnitude below
  2^-30 is stored as a signed zero (no subnormals). A finite magnitude above (2 - 2^-9) 2^31, or
  one that rounds up into exponent 63, saturates to `0x7DFF`. Inf is stored as `{s, 63, 0}` and a
  NaN as `s | 0x7F00`. Widening: exponent 0 is ±0, 63 is Inf or NaN, else the fp32 exponent is
  e + 96.
- **bf16** (code 2): the fp32 value's high 16 bits, rounded to nearest even.

The DeltaNet states of the 35B and the 2B stay below 40 in magnitude (tools/dn_state.py's
largest value by layer: 36.5 and 19.4), so e6m9 gives up range they do not use for 2 more
mantissa bits than bf16. The quality study (tools/dn_state.py on docs/isa.md's text; the KL from
the fp32-state model's distribution, 2,000 tokens unless noted):

| model, weights | state | KL vs fp32 state | greedy 8 tokens |
| --- | --- | --- | --- |
| 2B, float | e6m9 / bf16 | 0.013% / 0.254% (4,000 tokens) | |
| 2B, float | e6m9 / bf16 | 0.050% / 0.678% (8,000), 0.145% / 1.603% (16,000) | |
| 27B, float | e6m9 / bf16 / fp24 | 0.005% / 0.111% / 0.000% | |
| 27B, fp4 / int8 head | e6m9 / bf16 | -0.005% (±0.025) / +0.047% (±0.034) | |
| 35B, float | e6m9 / bf16 / fp24 | 0.347% / 0.498% / 0.133% | |
| 35B, fp4 / int8 head | e6m9 / bf16 | (bf16 +0.131% ±0.073) | HF's |

The quantized rows give the change in KL from the float model. The 35B's float row is mostly its
router: there even fp24 moves 0.13%, where the dense-FFN 27B's fp24 moves nothing (most likely
a rounded state flipping an expert choice now and then). At the production formats the rounding
sits inside the quantization's spread: the 35B at fp4 / int8 is 7.1% KL from float. The
rounding's KL grows with the position, faster than linearly: on the 2B, e6m9's KL per range of
positions is 0.009% at 1,000-2,000, 0.021% at 2,000-4,000, 0.086% at 4,000-8,000 and 0.241% at
8,000-16,000 (bf16's 0.196%, 0.382%, 1.101%, 2.529%).

Rules: ST16 needs DSTEP and STREAM (the DMA's state path) and D ≤ 128 (`opentpu.target`); the
STREAM form keeps both ends in DRAM (no SRC_T, DST_T or NODST). There is no VOP fallback. The
compiler's LD and ST refuse a 16-bit tensor (`Tensor.esize` 2), and `ol.stream` refuses one
outside `stream_hw_cfg`. The ISA simulator and the DMA refuse the flag on a configuration
without ST16. A target's ST16 joins arch_hash and rtl_params only when set, so every target and
variant before it keeps its CFG_HASH (`bcu1525-wg:bhw16` is `:bhw` with ST16 e6m9: 4d5cbf8d;
`bcu1525-s2:st16` e331f79c). A Qwen3.5 image (`opentpu.llm.qwen35`) stores its states in 16 bits
wherever the configuration has ST16 and DSTEP (`Spec.state16`; set it to False to keep a model in
fp32). The state slots, `kv_bytes` and chat-KV snapshots then take half the bytes. A state
saved off the device (otpu-serve's KV store) carries the format in its key
(`snapshot.state_format`: `dn_fold-e6m9`), so it goes back only into an image whose states have
the same format.

### RLD

`R[rd] = f2i(T[R[ra] + w1])`: a TMEM word into a register, the only way a value the device
computed reaches an address, a loop count (`LOOP R[ra] + w2`) or a condition (a `LOOP` of count
0 or 1 around the instructions it guards). `f2i` truncates toward zero: `|x| < 1` (with +-0 and
the flushed denormals) gives 0, `|x| >= 2^31`, infinities and NaN give `0x80000000`; integers of
up to 24 significant bits come out exact. Flag bit0 RAW: the word's bits instead. Flag bit1
MUL: the value (f2i's or RAW's) times `R[rb] + w2`, the low AW bits of the product (at AW 32
the low 32 bits, signed and unsigned alike; above it the value is zero-extended, so a negative
multiplier comes from a register: `isa.rld_mul`, LI `R[rd]` the AW-bit multiplier, then RLD MUL
by `R[rd] + 0`, its multiplier read before `R[rd]` is written): an integer the device computed
scaled into a byte offset past fp32's 24 bits, e.g. a token's row in a 2.4 GB table
(`tok * 9344`), or a resident expert's place in its pool (`id * expert bytes`, "Address
bits"). `rd = 0` writes nothing. Flag bit2
BOUND: the value lies in `[{w5, w3}, {w6, w4}]` (unsigned; w5 and w6 the bounds' high words, 0
below 2^32, so a 32-bit bound's words are w3 and w4 alone; a lower bound below 2^AW with an
upper one past it wraps: the value from the lower bound to 2^AW - 1, or the value plus 2^AW up
to the upper one, a negative multiplier's product, e.g. a sliding layer's mask row at
`-4 * tpos`), what the kernel knows of the word (an expert's index below E, a slot of the
expert pool, a row's place in a scratch, a resident expert's place in its pool: `id * expert
bytes` up to the pool's size, past 32 bits above AW 32); the compiler's wait entries
(`opentpu.waits`, "Wait counts") bound the footprints computed from the register by it instead
of taking any value (unbounded, a resident expert's MMs waited
for the DMA's older work and the next stores for them), the ISA simulator stops on a value
outside it, and the hardware ignores the fields. w7 VARIANT (bit 31, `isa.rld`'s variant:
`K << 16 | k` below it): the word is variant k's count of a K-word table at `w1 - k`, exactly
one of its words 1 at a run, each 0 or 1 (attention's sub-block variants: the loop of 0 or 1
iterations of the variant the run's tpos takes); the wait entries take one pass a combination
of the tables' variants (at most 16) and join their entries, instead of one pass with each
count 0 or 1 apart, which paired a VOP with the ones of a path no run takes. The hardware
ignores w7.

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
(bit 28 FOLLOW, below; bits 29..31 are 0), and the VPU mask's word, `vmask = w1[30:0]` (bit 31
is 0). The units are U0 DMA (`LD ST DSTEP STREAM WAITW`), U1 MXU (`MM`), U2 quantizer (`QACT
QST`), U3 VPU (`VOP`), U4 collective (`BAR GATHER RLD`); control instructions (`NOP HALT LI ADDI
LOOP`) have no entry that matters.
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
- FOLLOW (bit 28; a build with wait entries at S > 1, `otpu_seq` FOLLOW): a `VOP` with `k_4 = 1`
  waits for that collective instruction, a `GATHER`, to start instead of to complete, and the
  VPU reads its A operand only below the top of the GATHER's writes into its slice's TMEM
  (`otpu_slice` / `otpu_vpu`: the whole VPU waits, as for the TMEM grant). The compiler sets it
  only where that is exact (`opentpu.waits.follows`): the `VOP` is an `RSUM`, `RSSQ`, `RMAX` or
  `ARGMAX` of one row (A alone, read upwards), the `GATHER` writes one run (`rows` 1, `seg` =
  `cols`: upwards from `dst` in every slice), and the `VOP`'s only conflict with it is reading
  what it writes. A build without FOLLOW ignores the bit (the `VOP` waits for the `GATHER` to
  complete): the same results. A norm after its all-gather then sums its squares as the row
  arrives ([seq_vpu.md](seq_vpu.md), the collective wait's FOLLOW).
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
is the smallest over all its visits, its mask the union (over each pass, with RLD VARIANT tables one a combination of
their variants). It is conservative: an unknown address
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
