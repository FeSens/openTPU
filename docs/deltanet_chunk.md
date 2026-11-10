# The chunked DeltaNet: the state's work as MXU matmuls (EXPERIMENTAL)

A Gated DeltaNet layer's prefill runs its recurrence a row at a time on the stream engine: DSTEP
is 2,048 cycles a value head and row ([isa.md](isa.md) "DSTEP"), so 16,384 a row on a slice
holding 8 of Qwen3.6-35B-A3B's 32 value heads, whatever the MXU's width. The chunked form runs
the same recurrence over **chunks of C rows as matmuls** (the WY / UT form), on the MXU, with the
state in TMEM for a whole pass. It is what takes the DeltaNet layers off the stream engine in a
long prompt's prefill ([pf500.md](pf500.md)).

**Status: experimental.** `Config.DNCHUNK = C` (and `DNMERGE`) are compile choices, never in the
CFG_HASH, and **no target file sets them**: they are given as a target modifier
(`bcu1525-x2:b2+DNCHUNK=64`) by `tools/pf500.py` and the tests. The chunked form is **not
bit-exact** with the recurrence (the sums are taken in another order and the operands are
quantized); it is judged by its distance to the float model, below. The RTL it needs is in
`bcu1525-x2:wmx` (`SPLITQ`, `QHL`, `ST16S`, `QUANT_SWW`) and `:b2` (those and `QUANT_QSTB`,
`QSTHL`, `SPLMM`) only, bit-exact against the ISA simulator in the Verilator tests. On the board
model the merged prep and post programs leave every byte the ISA simulator's; the core's run at
R 512 there was checked by its instruction counts only. None of it is in a production image or
has run on a card.

## 1. The chunk's math

Per value head, a chunk of C rows. `gc = cumsum(g)` (the log2 decays), `Gam[r, s] = 2^(gc_r -
gc_s)`.

| # | op | MM shape | role |
|---|----|----------|------|
| M1 | L = (beta K K^T) . Gam, strictly lower | [C, C], K = d | the in-chunk interaction |
| - | T = (I + L)^-1 | forward substitution on the VPU | |
| M2 | U0 = T . (beta V) | [C, d], K = C | |
| M3 | Wc = T . (beta e^gc K) | [C, d], K = C | |
| M5 | U = U0 - Wc . S | [C, d], K = d (ACC into U0) | the state's correction |
| M4 | A = (Q K^T) . Gam, lower with the diagonal | [C, C], K = d | |
| M6 | O = (e^gc Q) . S | [C, d], K = d | the state's output |
| M7 | O += A . U | [C, d], K = C (ACC) | |
| M8 | S' = e^gl S + (e^(gl - gc) K)^T U | [d, d], K = C | the state update |

On the units (`opentpu/kernels/dn_chunk.py`):

- **Masks without new VOPs.** `EXP2SUB` of a FILL of gc by row and by column, then MIN against
  the 0/1 strict and lower masks, gives Gam's two tiles exactly.
- **Scales in the quantizer and the epilogue**: -beta and e^gc as QACT's RSCALE and CSCALE; the
  state's decay e^gl as ASCALE.
- **The solve**: T from I by columns, a MUL and an ADD a column (or one VOP OUTER a column where
  the target has it).
- **Streamed operands** go to a DRAM scratch through QST (K rows; K^T, V^T and U^T as transposes
  by an identity MM from the rows). A pair of value heads shares its key head's token
  contractions (C <= D / 2: one head's tokens in columns 0 .. C - 1, the other's in C .. 2C - 1),
  which saves two D x D QSTs a pair and chunk.
- **The state** stays fp32 in TMEM for the pass and is converted from and to the decode's 16-bit
  slots at the pass's ends (STREAM's ST16R / ST16W, `ST16S`).

**Precision.** An int8 activation times an int8 weight is not enough for the matmuls that carry
the state. A *split* operand is quantized to a hi and a lo int8 part (lo = the residual at 2^-8
of hi's scale), and a split MM is three passes: hi.hi + hi.lo + lo.hi. A one-sided split keeps
int8's error (about 1.1e-2 relative) and is not an option. The kernel's pick is **C = 64, M1 one
int8 pass, M2-M8 split**.

## 2. In the layer

With `DNCHUNK = C > 0` a DeltaNet layer of a prefill chunk of R rows runs as four steps
(`opentpu/llm/tiled.py` `dn_parts`, `Engine.prefill_layers`):

| # | run | per | does |
|---|-----|-----|------|
| 1 | prep tile (`part="dn-prep"`) | T rows | the previous layer's MoE combine, the norm, the projections, the convolution, SiLU, the l2 norms, the gates -> the scratch rows `dnq` |
| 2 | core run (`Image.compile_dn_run`) | the chunk | per key head, chunks of C rows: the table above, `dnq` -> `dno`, the state slot read at the start and written at the end |
| 3 | post tile (`part="dn-post"`) | T rows | the gated norm, out_proj, the residual, the MoE prologue |
| 4 | the expert run (unchanged) | the chunk | |

The tail (the last `n mod C` rows) and a prompt's first rows run through "dn-core" tiles: the
recurrence by DSTEPs on the same scratch, which is also the bit-exact reference for prep and
post (`tests/test_em_resident.py` test_deltanet_by_chunks_is_the_recurrent_tiles_bits).

`DNMERGE = N` (`opentpu/stage.py`) merges 8 tiles of prep, or of post, into one program with N
in flight: each tile's kernel is a stream, slice 0 picks the earliest start among the N oldest
streams' heads and the other slices replay its picks. It is bit-exact against the serial runs
(`tests/test_dn_stage.py`; every DRAM byte equal on four slices on the ISA simulator and the
board model, the real 35B's dense weights included).

## 3. The RTL it needs

| piece | key | what | a slice (yosys, UltraScale+) |
|---|---|---|---|
| the quantizer writes a lo part | `SPLITQ` (QACT / QST flag LO) | lo = q8(256 (x inv - q)), scale S / 256, as an instruction of its own | +1,164 LUT, +129 FF |
| hi and lo from one read | `QHL`, `QSTHL` (QACT / QST flag HL) | both parts in one pass over the operand | |
| QST pass 1 at 16 bytes a cycle | `QUANT_SWW` 4 (rows mode; COL 8) | was 4 and 1 | about +0.3K LUT, +0.3K SRL, +0.4K FF over the quantizer, the stages and the adapter |
| QST rows through the row buffer | `QUANT_QSTB` | pass 1 under the next rows' pass 0 | |
| the state's 16-bit <-> fp32 copies | `ST16S` (STREAM ST16R / ST16W) | at a pass's ends | +25 LUT, +12 FF |
| split MMs folded | `SPLMM` (MM SPL) | a triple (lo.hi, hi.hi, hi.lo) or a transpose pair as one MM of KB 3 / 2 | about 0.2-0.3K LUT (estimate) |

**MM SPL** ([isa.md](isa.md) "Split terms"): the lo advances add 8 to their ACT record's offsets,
so the 2^-8 of the lo parts comes from the shift and their own scales are never read; hi.hi goes
last so the row's Sa' is the hi one. It needs RSC, int8 and GALL, and stays inside the MXU's
47-bit accumulator (|acc| < 2^40 at 12 groups a row). A folded pass takes 3 cycles and yields 2
rows under the pump, so the 8-lane drain keeps up ([wide_mxu.md](wide_mxu.md) "The pass rule"):
the fold equals two result rows a cycle on the DeltaNet core without its estimated 14.6K LUTs. It changes
bits (the three products are one integer sum), so it carries its own references.

Not needed: a transposed MM output, a cumsum VOP (six Hillis-Steele ADDs), a mask VOP, a solve
unit.

## 4. Cycles

Unit cycles a row on a slice (8 value heads; the ISA run's ledger at `tools/pf500.py`'s rates):

| | MXU | VPU | quantizer | DRAM | TMEM peak (words) |
|---|---:|---:|---:|---:|---:|
| DSTEP (the recurrence) | - | held | - | - | stream engine 16,384 |
| chunked, C 32 | 2,152 | 1,429 | ~7.6K | 1,297 | 78K |
| chunked, C 64 | 1,808 | 2,580 | ~5.1K | 1,078 | 97K |

As first built the core is quantizer-bound at about 5.1K cycles a row, 3.2x DSTEP: half of the
quantizer's time is the lo passes re-reading their operand. With hi and lo from one read (HL) it
is about 2.6K and VPU-bound. C 64 is the pick: C 32 halves the solve's VPU work but costs 1.5x
the quantizer. On the board model (`bcu1525-x2:b2+DNCHUNK=64`, R 512, four slices) the core's
run is 2,999,468 cycles a chunk with MM SPL, 175,750 a token; the prefill it gives is in
[pf500.md](pf500.md).

## 5. Numerics

Real weights: Qwen3.6-35B-A3B, 2,000 tokens, production formats (`tools/dn_chunk.py ppl`). dKL
is KL against fp32 Hugging Face minus int8's (1.010%); the difference is against production's
recurrence with the state stored e6m9 after every row, +- its standard error over the tokens.

| mode | dKL % | vs production (pt) |
|------|------:|-------------------:|
| the recurrence in fp32 (formats alone) | 4.596 | -0.193 +- 0.062 |
| the recurrence, state e6m9 a row (production) | 4.790 | 0 |
| chunked C 64, all split, fp32 state | 4.700 | -0.090 +- 0.067 |
| chunked C 64, all split, e6m9 at the chunk ends | 4.866 | +0.076 +- 0.065 |
| chunked C 32 / C 128, all split, e6m9 | 4.697 / 4.759 | -0.093 / -0.031 |
| chunked C 64, M1 int8, the rest split, e6m9 | 4.656 | -0.134 +- 0.066 |
| the same, M4 int8 too | 4.917 | +0.127 +- 0.073 |
| the state's matmuls (M5, M6, M8) int8 | 5.378 | +0.589 +- 0.108 |
| **the pick as the kernel runs it** (transposes by identity MMs) | 4.566 | **-0.224 +- 0.068** |

- Every mode with the state's matmuls split is within 2 standard errors of production; by the
  formats rule (1 point of dKL for 10% of speed, [formats.md](formats.md)) that is at most 1.3%.
- M1 takes one int8 pass at no cost; M4 stays split; M5, M6 and M8 need the split.
- C 32, 64 and 128 agree within noise: the chunk size is a cycles choice.
- The recurrence pays 0.19 +- 0.06 for rounding its state to e6m9 every row; the chunked path
  rounds once a chunk. The pick equals the fp32 recurrence (-0.03 +- 0.06).

**On tiny random weights the logits are a weak test.** Replaying the chunked core against the
recurrent tiles with every MoE route forced (24 cases at four slices), 2 cases exceed a KL of
2e-4 (the highest 3.35e-4), but so do the tiles against themselves with their state moved by 1-2
units in the last place (17 of 128, the highest 3.59e-4): a head with a long memory amplifies
last-bit differences, and nothing under about 1.3e-4 to 2.4e-4 is visible in the logits. Against
the float model the core and the tiles are level on average (mean KL 2.563e-4 against 2.570e-4),
not case by case. One case stays 1.06x above its control and is untraced; it is recorded, not
explained. A KL near 1e-2 on such a model is a flipped expert route, not a core error. The gate
is therefore `tests/test_dn_chunk.py` (the kernel against the float64 recurrence: 1.2-1.4e-4
relative over 128 rows, within 2.5e-4 of the mode's emulation) and a sharp wiring test (every
head given a set memory, layer 0's rows against float at most 1.015x the tiles'); the real-weights
evidence is the table above.

## 6. Tests

`tests/test_dn_chunk.py` (the kernel against float and against `tools/dn_chunk.py`'s emulation),
`tests/test_em_resident.py` (prep / core / post against the recurrent tiles, the core in the
engine), `tests/test_dn_stage.py` (DNMERGE on the ISA and on Verilator), `tests/test_splitq.py`
and `tests/test_quant_rtl.py` (LO, HL, SWW, QSTB), `tests/test_wg_rtl.py` (MM SPL against the
ISA), `tests/test_st16.py` (ST16S).
