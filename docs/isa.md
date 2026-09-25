# openTPU ISA v1

This document is the contract between the compiler (`opentpu/compiler.py`), the bit-exact
instruction-set simulator (`opentpu/isasim.py`) and the RTL (`rtl/`). All three must agree on
every bit written to TMEM or DRAM.

## Machine model

- `S` slices. Each slice has: a sequencer with 16 x 32-bit registers (`R0` reads as 0), an
  instruction memory, a private DRAM, a TMEM, an ACT RAM, an MXU, a VPU and a quantizer.
  Slices share nothing except the collective unit (`GATHER`, `BAR`).
- `D` = MXU depth = quantization block size (bytes / int8 elements). Default 32 in tests,
  128 in the design. `MCOLS` = MXU columns = max stationary rows (8).
- **DRAM**: byte addressed, little endian, accessed as 32-bit words. All word accesses and all
  MXU chunk reads must be 4-byte aligned. Only `QST` writes single bytes.
- **TMEM**: 32-bit word addressed. Holds fp32 values (raw IEEE bits).
- **ACT RAM**: `MCOLS` rows x `ACT_BLOCKS` blocks x `D` int8, plus one fp32 scale per
  (row, block). Written only by `QACT`, read only by `MM`.

Execution is in order. Every instruction completes (all its writes are visible) before the
next one starts. The MXU prefetches its streamed operand from DRAM internally.

## Arithmetic (fp32)

IEEE-754 binary32, round to nearest even, **flush to zero**: denormal inputs are treated as
signed zero and denormal results are replaced by signed zero. No NaN inputs are expected;
any NaN produced is the canonical `0x7FC00000` (a final sign flip, as in `recip`, may set its sign). Composite functions are defined as fixed sequences of
`add`/`mul` so that every implementation is bit exact:

- `i2f(i)`: int32 to fp32, RNE.
- `exp2(x)`: if `x < -126` return `+0`; if `x >= 128` return `+inf`. `i = floor(x)`,
  `f = x - i2f(i)`, `p = C0 + f*(C1 + f*(C2 + ... + f*C7))` (Horner, fp32, Taylor coefficients
  `ln2^k/k!` rounded to fp32), result = `p` with `i` added to its exponent field.
- `recip(x)`: `x == 0` returns `+0`; `|x| >= 2^126` (including infinity) returns a zero with
  the sign of `x` (the result would be subnormal and flush). Otherwise on `|x|`:
  `y = bits(0x7EF311C3 - bits(|x|))`, three times `y = y * (2 - |x|*y)`; the sign of `x` is
  applied at the end. This keeps `silu(x) = x * recip(1 + exp2(-x*log2e))` exact at `-0` for
  very negative `x`, where `exp2` overflows to infinity.
- `rsqrt(x)`: `x <= 0` and `x = +inf` return `+0`. `y = bits(0x5F3759DF - (bits(x) >> 1))`, `h = 0.5*x`,
  three times `y = y * (1.5 - h*(y*y))`.
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
  MM uses P = 4 over the K blocks; RSUM, RSSQ and RDOT use P = 64 over the columns. Missing terms
  are +0 (a partial is never -0, so they do not change it).
- `q8(x)`: round half to even to an integer, saturate to `[-127, 127]`.

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
| 0x01 | HALT | stop this slice |
| 0x02 | LI | `R[rd] = w1` |
| 0x03 | ADDI | `R[rd] = R[ra] + w1` |
| 0x04 | LOOP | body = next `w1` instructions, executed `R[ra] + w2` times (0: skipped). Loops nest (depth 4); a body must not end on the same instruction as an enclosing body. |
| 0x05 | BAR | wait until every slice has reached a `BAR` |
| 0x10 | LD | DRAM -> TMEM, `n = w3` words: `T[R[rb]+w2+i] = M32[R[ra]+w1+4i]` |
| 0x11 | ST | TMEM -> DRAM: `M32[R[ra]+w1+4i] = T[R[rb]+w2+i]` for `i < w3` |
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
correction step, done in the MXU epilogue). The streamed rows are D-byte aligned (`sa` and `rs`
are multiples of D): the MXU streams whole D-byte DRAM chunks.

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

### QACT

`src = R[ra]+w1` (TMEM words), `rows = w2[7:0]`, `ab = w2[15:8]`, `KB = w2[31:16]`,
`srs = w3` (source row stride, words). Flag bit0 `ROW`: one scale per row instead of per block.
Flag bit1 `CSCALE`: every element is first multiplied by a per-column scale,
`x = T[src + r*srs + c] * T[w4 + c]` (this folds V's per-token scale into P for free).
Flag bit2 `RSCALE`: every element is first multiplied by a per-row factor `T[w5 + r]`. With
both, `x = (T[src + r*srs + c] * T[w5 + r]) * T[w4 + c]` (RMSNorm's `x * r * gamma`).
For each row `r < rows` and block `k < KB`: quantize `T[src + r*srs + k*D + i]`, write
`ACT[r][(ab+k)*D + i] = q[i]`, `ASCALE[r][ab+k] = s`.

### QST

`src = R[ra]+w1` (TMEM), `dst = R[rb]+w2` (DRAM bytes), `sdst = R[rc]+w3` (DRAM bytes),
`rows = w4[15:0]`, `KB = w4[31:16]`, `srs = w5` (words), `drs = w6` (bytes), `es = w7`
(element stride, bytes). Flag bit0 `ROW`.
Element `c` of row `r` goes to byte `dst + r*drs + c*es`. Scales: per block to
`sdst + (r*KB + k)*4`; in `ROW` mode one scale per row to `sdst + r*4`. The data and scale ranges of
one QST must not overlap.

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

In RSSQ and RDOT each product is rounded, then added into the isum_64 partials.

**OUTER** (the state update of linear recurrences: Gated DeltaNet, Mamba2/SSD, GLA, RWKV,
linear attention) is elementwise and in place, `T[dst + r*drs + c] = add(mul(T[dst + r*drs + c],
Dv(c)), mul(B(r), Cv(c)))`: two rounded products, then a rounded add. B must be per row
(bmode 1): `B(r) = T[b + r*brs]`. The column vector is `Cv(c) = T[imm + c]`, and the decay
`Dv(c)` is `T[a + c]`, or `T[a]` for every column with flag bit 0 (DSCALAR), or 1.0 with flag
bit 1 (DONE; `a` is not read). The `a` field holds the decay address: A is dst itself (`ars` is
not used). `cols <= 256`. `Cv` and `Dv` are read before anything is written, so they may overlap
dst; `B(r)` is read with every element and must not be written by an earlier element.

### GATHER

All slices must execute a `GATHER` with the same `dst`, `rows`, `cols`, `drs`, `seg`.
`src = R[ra]+w1`, `dst = R[rb]+w2`, `rows = w3[15:0]`, `cols = w3[31:16]`, `srs = w4`,
`drs = w5`, `seg = w6` (words). For every slice `s`, every `r < rows`, `c < cols`:
`T_all[dst + s*seg + r*drs + c] = T_s[src + r*srs + c]` is written into every slice's TMEM.
The source and destination ranges must not overlap.
