# Weight groups (WG): the MXU's wide weight stream

A slice whose memory delivers more than one D-byte chunk a cycle needs an MXU that takes more
than one weight row a step. `WG` is that width: at WG > 1 a step reads **P = WG x D bytes**
(the BCU1525 builds: 8 x 128 = 1 KiB, one 128-byte beat on each of 8 channel lanes) holding one
K-block of **WG consecutive weight rows**, and each of the MXU's MCOLS columns takes one of those
rows against one activation row that all columns share. The results do not change: grouping
changes where the bytes live and the schedule, never a bit of an MM's output.

This page is the reference for the layout, the ISA's side, the MXU's schedule and the compiler's
rules. The encoding itself is in [isa.md](isa.md) ("Weight groups", "K tail"); the memory side
(port W, the channel hash, the posted beats) is [native_channels.md](native_channels.md); the
card's numbers are in [bcu1525_wg.md](bcu1525_wg.md).

**Status.** WG 8 is the BCU1525's machine (`targets/bcu1525-wg.toml`, `bcu1525-s2`,
`bcu1525-x2`); every other target is WG 1, which is the machine before the field, bit for bit
and hash for hash (WG is in `arch_hash` only when > 1). At WG > 1 every MM is grouped (GALL,
section 6). Bit-exact against WG 1 on the ISA simulator and on Verilator for every model path in
the repository (`tests/test_wg.py`, `tests/test_wg_rtl.py`).

## 1. Definitions

- `MCOLS`: the slice's physical MXU columns. The WG builds: 8.
- `WG`: weight lanes, a group step's D-byte lanes, one a column. A power of two, `WG <= MCOLS`,
  and `WG x D = NCH x CB` (the slice's channels times their beat). `Config.WG`, default 1.
- `P = WG x D`: a group step's bytes (1,024).
- `R`: rows a group. `R = WG` for int8, `R = 2 WG` for the 4-bit formats (two rows a lane),
  `R = 1` ungrouped (`quant.grp_r`). Every address formula below is written for R, and **at
  R = 1 each one is the natural layout's**.
- A **group chunk** is P bytes: one K-block of the group's R rows, in every format.
- `sb`: a block's scale bytes a row (int8 1, int4 / fp4 4, NVFP4 8; RSC none). `E = R x sb`.
- `SwP`: a group's Sw words, padded: `4R` (int8, RSC, int4, fp4), `8R` (NVFP4).

## 2. The layout

Row n of a streamed operand is row `j = n mod R` of group `g = n div R`. Group g's chunk c is
the P contiguous bytes at `sa + g*rs + c*P` (rs: the group stride), lane l's D bytes at `+ l*D`:

```
int8  (R = WG)     lane l: row l's block c, its D elements
4-bit (R = 2 WG)   lane l, byte i: element i of block c of row 2l (bits 3:0) and of row 2l + 1
                   (bits 7:4)
```

So a 4-bit chunk is one block of 2 WG rows: the MXU takes it in one advance, each column's packed
PE multiplying its lane's two rows against the same activation block, at any M and any k0 (no
PAIR, no DUP rows). `sa` and `rs` are multiples of P, `rs >= KB * P`, and the groups an MM
touches lie whole in DRAM: the image pads N up to a multiple of R with zero rows.

With the memory's channel hash (`CHASH`, every FPGA target) a chunk's rows are stored in the
hash's order, row j at lane `j ^ h(chunk)`, so that channel c's beat is column c's row and
nothing permutes a chunk on the way in (`quant.hash_rows`).

**Scales.** Group g's scale area starts at `a = ssa + g*srs`:

| WF | row j's Sw | row j's data of the area's block kk |
|---|---|---|
| RSC (int8, int4, fp4) | word at `a + 4j` | none (dw 0, m 1) |
| int8 | word at `a + 4j` | byte at `a + SwP + kk*R + j` |
| int4 / fp4 | word at `a + 4j` | word at `a + SwP + 4*(kk*R + j)` |
| nvfp4 | word at `a + 4j` | 8 bytes at `a + SwP + 8*(kk*R + j)` |

The fields' contents are the natural layout's. Scale bytes a P-byte step: int8 8 (0.8%), fp4 /
int4 64 (6.25%), NVFP4 128 (12.5%), plus SwP once a group. A grouped 4-bit scale area is smaller
than its rows' natural scale rows (no per-row header): an fp4 [2048, 512] matrix's 49,152 ->
40,960 bytes; images are 0.6-0.8% (fp4) and 1.1-1.5% (NVFP4) smaller than with 8-row 4-bit
groups.

## 3. The ISA

- **MM**: its rd field (MM has no rd register) holds `LG = log2 R`; 0 is the natural MM. The
  instruction carries its own R, so the footprint (`otpu_pkg`, `opentpu/waits.py`) needs no
  Config. An R other than 1 and the format's is an error. A grouped MM never has PAIR (a grouped
  4-bit MM is at full rate without it). k0, KB, M, ab, ACC, UNIT, RSC are unchanged; an MM with
  M > MCOLS needs `KB <= DEPTH` group chunks in the MXU's FIFO, as a replayed row does.
- **Footprint**: DRAM read `[sa, sa + (ceil(N/R) - 1)*rs + chunks*R*D)`, scales `[ssa, ssa +
  (ceil(N/R) - 1)*srs + SwP + (k0 + KB)*E)` (RSC: `+ 4R`); at R = 1 both are the natural MM's.
- **QST GRP** (flag bit 2): the appends of grouped run-time operands. A K row is KB pieces of D
  bytes, P apart, in its group's lane; a V^T append is QST GRP COL, each dimension's byte into
  its row's lane at the token's column.
- **KT**, the K tail (section 5): `rs[0]`, free under GALL since `rs` is a multiple of P.

## 4. The MXU's schedule

A grouped step takes one P-byte chunk from port W: column j takes lane j's D bytes (int8 row j;
4-bit rows 2j and 2j + 1), and every column takes the same ACT block of one activation row m.
The ACT RAM reads row m into column 0; column j > 0 takes it from its neighbour a cycle later
(the activation hop).

```
for each group g of R rows:                  (its chunks: once from DRAM)
  for each activation row m < M:             (replay: the chunks re-read from the FIFO)
    for each chunk c < KB:  advance
  -> one result row: R integers (rows gR .. gR+R-1 against row m), R Sw, one Sa'
```

A group's first advance pops only once all of its chunks and scale entries are in the FIFOs, so
the array never stalls inside a group. Only G in {1, WG} exists: with one activation row a step
and replay, G = WG keeps every column busy at any M, where G = MCOLS / M would idle columns when
M does not divide MCOLS and need several ACT banks a step. So decode's weight MMs (M = 1) run at
P bytes a cycle, and batch or prefill MMs at P / M, each group read once from DRAM for all M
rows.

**Ports.** Port W carries the group chunks (P-byte reads, their beats lane by lane straight into
the chunk FIFO, the lowest priority on a channel); port A the scales (a group's Sw line, then one
request a chunk answered with a whole channel beat); port B nothing at WG > 1 (section 6).

**Datapath.** Column j's PEs take row j's slice of the chunk through its own skew, in place of
the natural mode's hop from column j - 1, so every timing downstream (column ends, deskew, row
assembly) is one mode's. A result FIFO row is MCOLS integers, MCOLS Sw factors and one Sa'; the
lanes compute `(i2f(x) * Sw) * Sa'`, the ISA's multiply order. A 4-bit group's 16 integers are
two FIFO rows. The drain writes result (r, m) to `out + m*ors + gR + r`: consecutive words,
distinct TMEM banks.

**The packed PE.** A column multiplies D elements a cycle and a 4-bit chunk holds 2D, so each
DSP position takes both of its byte's rows in one multiply (the two 4-bit weights packed into
one operand); 4-bit weights then stream at the same P bytes a cycle as int8.

## 5. The K tail (KT)

An MM's K is whole D-blocks. A K of `D n + D/2` (GPT-OSS's hidden size 2,880 = 22.5 blocks of
128) is padded with a zero half block that is streamed: at WG 8, half of a 1,024-byte chunk,
32 bytes a row in 4-bit and 64 in int8. With KT the image stores such an operand's groups in
pairs that share the tail chunk:

```
pair p at sa + p*rs:  [group 2p: chunks 0 .. KB-2] [T_p] [group 2p+1: chunks 0 .. KB-2]
T_p lane l:           bytes 0 .. D/2-1   group 2p's half block
                      bytes D/2 .. D-1   group 2p+1's half block
```

A pair is `2 KB - 1` chunks instead of `2 KB`. Group 2p consumes its chunks and then T_p against
activation block KB - 1, whose upper half is zero (the compiler asserts it), so the other group's
half adds 0. Group 2p + 1 consumes T_p first, against the same activation block with its two
halves swapped, then its own chunks. Every sum is one exact integer sum, so the results are the
padded MM's bit for bit. In the MXU it is one swap of the activation block's halves at column
0's input and the issuer's and consumer's chunk counts: about 0.5K LUTs of multiplexer and a few
hundred of control (estimate), no DSP or RAM. Scales need no RTL: the image stores the odd
group's area in consumption order.

Only images with a K of `D n + D/2` change; every other model's images and programs stay bit for
bit. Measured on the BCU1525's board model (GPT-OSS-20B's 2-layer cut, position 60, `bcu1525-wg`
against `bcu1525-wg:kt`):

| | a token (cycles) | useful share of the DIMMs' 76.8 GB/s |
|---|---:|---:|
| KT off | 5,750,992 | 85.25% |
| KT on | 5,654,363 | **86.71%** |

-96.6K cycles a token (-1.68%): the experts' gate, up and last down part -53.8K, the int8 head
(K 2,944 -> 2,880) -26.9K, QKV and the router -15.9K; the cut's image 3,722.19 -> 3,692.41 MiB.
Qwen3.8-27B and Qwen3.6-35B-A3B gain nothing (every K whole blocks). Not covered: the KV cache's
K at head_dim 64 (its append address is not affine in the position), and a half chunk at an
arbitrary position of a run-time V^T.

## 6. Every MM in groups (GALL)

At WG > 1 no MM streams rows of one: `Config.GALL` is derived from WG > 1 (`opentpu.target`
DERIVED), the ISA simulator refuses a natural MM there, and the MXU's natural mode and its port
B slot are not built. What is grouped beyond the image's weights:

- **The KV caches** (`compiler.KVDesc` wg): K in groups of WG tokens, each V^T tile in groups of
  WG dimensions; the same bytes as rows of one, appended by QST GRP. RMAX and ASCALE are taken
  in groups (the attention's scores and P.V).
- **The experts' slots** (`moe.ExpertFormat` grp): each matrix in groups, P-aligned in the slot,
  the data regions lane-hashed by their slot address as the expert server writes them.
- **The gathers**: a table row's blocks are a grouped [K / D, D] matrix, one chunk a group; the
  4-bit gathers and the records (Gemma 4's head table and PLE records) the same, gathered with
  int8's one-hot operand. The values are the natural gathers'.
- **serve's KV pages** (`serve/kvio.Layout`): a partial page rounds to whole groups of tokens.
- **MiniCPM-V's vision tower**: each piece's blocks grouped, its KV scratch in groups.

Costs (computed): tables whose rows are not whole P-byte chunks pad (a 2,048-column int8 table
at four slices 512 -> 1,024 bytes a slice row; Gemma 4's PLE records int8 +9.6%, fp4 +20%); the
KV caches and the experts' data keep their sizes. Gain: grouped scores and P.V at M query rows
take `M x KB` advances a group of 8 rows where natural MMs take `8 x KB` (-25% of the
attention's advances at M = 6); most of it sits under the VPU's softmax, so the token moves by a
fraction of a percent. GALL's case is area: one mode in the MXU.

## 7. Compiler, layout, images

- `QTensor.grp` = R; `QTensor.weight(..., grp=WG)`. For a grouped matrix `rs` and `srs` are group
  strides; a row slice starts at a multiple of R (`CompileError` otherwise); a column slice moves
  data by `(c0/D)*P`. `dot()` emits the grouped MM for R > 1 and checks alignment and the FIFO
  bound. `language.weight_group(fmt)` is R.
- `opentpu/quant.py`: `group_rows` / `group_scales` (natural rows -> a grouped matrix's bytes),
  `grp_row_bytes`, `grp_scale_bytes`, `grp_bytes`, `grouped`, their inverses for tests, `grp_r`,
  `swp`, `hash_rows`.
- Every image builder lays each streamed weight matrix out grouped at `cfg.WG`, per slice (each
  slice's rows are a matrix of their own, padded to R).
- A WG 8 model's tokens are the WG 1 model's: the same MMs (N, KB, k0, parts), only addresses,
  strides and LG differ.

## 8. Tests

`tests/test_wg.py` (the ISA: every model path against WG 1, QST GRP against `quant.group_rows`,
KT against the padded MM, the waits' footprints), `tests/test_wg_rtl.py` (the MXU on Verilator
against the ISA: every format, replay, RMAX / ASCALE in groups, KT, the packed PE's extremes,
posted lanes with stalls), `tests/test_quant.py` (the grouped gathers and records).
