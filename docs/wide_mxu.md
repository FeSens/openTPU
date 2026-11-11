# The wide MXU: the pumped array, two TMEM writes a bank, and the VPU beside it

At long context a decode token is no longer only weights at the DRAM's rate: every attention
layer scores the whole KV cache, and those MMs and their softmax grow with the position (on the
BCU1525, 4.5 -> 7.4 M cycles a token from a short context to 31k: about 9.3 cycles a position an
attention layer, linear to 262k). Prefill is MXU work at any context. This page is the set of
changes that widen the core for both without a second array: the MXU's columns on the 2x clock
(`MXU_PUMP`), a second TMEM write a bank (`WPB 2`), and the VPU's composites and softmax
(`VPU_CNS`, `FSUM`, `VSIG`, `VMAC`). All of them are timing only or new VOPs with exact
definitions: tokens are the unpumped machine's.

**Status.** RTL, bit-exact on Verilator against the ISA simulator, and measured on the board
model. The variants that carry it are `bcu1525-wg:wmx` and `bcu1525-x2:wmx` (and `x2:b2`, the
staging set of [pf500.md](pf500.md)). The production images (`bcu1525-wg:bhw16`,
`bcu1525-x2:st16`) do not, and no pumped bitstream exists: the one build of the set, a 100 MHz
development build, failed timing after routing, and the 150 MHz build was not launched
(section 7).

## 1. Bounds first

The board model at 31,000 positions (Qwen3.6-35B-A3B's 4-layer cut, one card, `bcu1525-wg:bhw16`),
with oracles for each unit (`tools/vpu_hack.py`):

| | cycles | cycles a position a layer |
|---|---:|---:|
| today | 638,949 | 9.35 |
| MXU 2x | 637,173 | ~9.3 |
| VPU 2x (EXP2 in two passes + the fused row sum) | 598,772 | 8.05 |
| MXU 2x + VPU 2x | 564,487 | 6.94 |
| the same, TMEM two writes a bank | 512,411 | ~5.3 |

The MXU and the VPU are co-bound at long context (each about 85% of the cycles in work), and
TMEM's writes are next: no one of the three pays alone. GPT-OSS-20B at 31k is VPU-bound (EXP2SUB
74% of the VPU's work, RSUM 24%). In prefill the MXU binds: the 27B at 4 rows -30.8% with the MXU
at 2x, the 35B's attention layer at 31k -33.1% with both.

Ruled out: MCOLS 16, D 256 or a second array (+1,024 DSPs where a VU9P die has 1,505 of its
2,280 in use) and 16 VPU lanes (+28K LUTs and 16-word TMEM ports).

## 2. The pumped array (`MXU_PUMP`)

`[uarch] MXU_PUMP` (`otpu_mxu` PUMP; WG > 1 only). Each column is an `otpu_mxpc`: its 128
positions' DSP48E2s run on `clk_f = 2 x clk`, each position taking activation row 0 on one fast
edge and row 1 on the other against the same packed weight, the cascade carrying both rows' sums
interleaved. A grouped pass is then rows 2p and 2p + 1 (the even ACT bank's and the odd one's at
one address): `ceil(M / 2)` passes a group instead of M. Column 0 skews both rows and interleaves
them per position; the columns past it take the previous column's interleaved stream two fast
edges later, so only column 0 has a multiplexer. Each column has two rows' column ends. The
result FIFO's rows are pushed in the unpumped passes' order from one row-end queue, so the drain
and everything after it are unchanged; a pass's first advance waits while the last passes' rows
could stall a row in its middle.

Decode's weight MMs have M = 1, so the pump changes nothing there: decode at the DRAM's rate
stays. It pays where M > 1: the attention's scores and P.V over a query group, batched decode
and every prefill MM.

**Area** (yosys, `otpu_mxu` alone at `bcu1525-x2`'s parameters, UltraScale+, the DSP positions
as boxes): 56,284 -> 63,098 LUTs and 57,791 -> 76,486 flip-flops, **+6.8K LUTs and +18.7K
flip-flops a slice** (+1.7 and +2.4 points of a VU9P die), no DSP. A first form with a
multiplexer in every column was 70,111 / 83,654.

**Timing.** The weights' registers load at the even fast edges, so a weight launched at a core
edge has a core period; the chain ends' holds load at an even edge and are read at the next core
edge. `otpu_clk2x.tcl`'s pumped-MXU section sets those two multicycle pairs (setup 2 fast
periods, hold 1) on the columns' own clocks, and the build's check stops unless every column was
constrained. The rest of a column keeps one fast period (3.33 ns at 150 MHz).
`rtl/exp/pump_us/` times `otpu_mxu` alone at 150 / 300 MHz with the same section.

**The pass rule.** A pumped pass yields two result rows, and the drain (`MXU_E` = 8 epilogue
lanes) takes one row of 8 words a cycle. An MM whose rows are short (KB 1) is then bound by its
drain, not by the array: a DeltaNet-core MM of 16 rows x 128 results drains in 256 cycles, 2.0x
its 128 advances. Two result rows a cycle would need 16 lanes end to end (16 scale multipliers,
16 ACC adders, 16 TMEM writes a cycle): about +14.6K LUTs a slice as Vivado would count them
(estimated from yosys differences of +18-19K), +6 points of a die. Not built. The cheaper form is the split MMs folded into one of KB 3 (MM SPL,
[deltanet_chunk.md](deltanet_chunk.md)): a pass then takes 3 cycles and yields 2 rows, and the
8-lane drain keeps up.

## 3. TMEM: two writes a bank (`WPB 2`)

The one-copy TMEM's pumped RAMs ([tmem_nocopy.md](tmem_nocopy.md), TMEM_IMPL 3 with TMEM_PMP)
read on both fast edges at port B and write on the even one at port A; port B is idle on the odd
edge. WPB 2 writes a bank's second word there, so two units' writes to one bank land in one core
cycle. The slice's mask arbiter admits a write group unless two groups admitted before it write a
bank it writes.

Cost (yosys, the slice with its units as black boxes: TMEM, the arbiter and the glue): 23,858 ->
25,283 LUTs (+1,425, +6%), +376 flip-flops, the same 16 UltraRAMs. An earlier form left the input
stage's address and data registers enabled by the grant (672 clock-enable pins from the
arbiter's output, +4.7K LUTs); the stage now loads every cycle and the selection takes a port's
lanes only under its grant (25 pins). The arbiter's longest path grows from 10 to 15 logic levels
(grant -> the units' read enables -> TMEM's lost reads -> the VPU's hold): the first place to
look in a build's timing report.

On the board model WPB 2 is worth -0.36% at 31k and -0.83% at position 60 on the 35B cut, and
within 0.1% on the 27B: small in decode. In prefill it matters more, where every DMA LD (8 words
a cycle into every bank) otherwise blocks the VPU's and the MXU's writes.

## 4. The VPU

- **`VPU_CNS` 5** (`otpu_vpu` CNS; default 3): the composites' loop with five stages a lane.
  EXP2, EXP2SUB, RSQRT and LOG2 take two passes instead of three or four, four columns a cycle,
  bit-exact.
- **`FSUM`** (`[arch]`, `otpu_vpu` FSUM): EXP2SUB's fused row sums. The softmax's `RSUM` after
  an EXP2SUB becomes a no-work fence that takes the sums the EXP2SUB already folded. Area (yosys
  with 7-series cells, the VPU at 8 lanes): +3,789 LUTs, +2,229 flip-flops.
- **`VSIG`** (`[arch]`, `otpu_se_comp` SIG): VOP SIGMOID and SILU as composites, four passes at
  CNS 5 and six at CNS 3, in place of the 6 / 7 separate VOPs. At CNS 3 EXP2's finish moves onto
  the loop register and RECIP's setup into the stage after S0; the ring that holds SILU's operand
  is unchanged.
- **`VMAC`** (`[arch]`, `otpu_vpu` MAC): VOP MAC, the DeltaNet's convolution a pass a tap.

FSUM alone is a case for bounds before code. On today's MXU it is -3.07% at 31k on the 35B cut
(the board model, FSUM's first form, which ended an EXP2SUB after its flush), but with VPU_CNS 5
it is **+0.75%**: the attention is then MXU-bound, removing VPU work there does not help, and the fused sum's tail sits on the path from
EXP2SUB through the quantizer to P.V. With the pumped MXU it pays again (GPT-OSS at 31k -5.2%).

**CNS 3 or 5.** On the prefill model's first calibration ([pf500.md](pf500.md)) CNS 5 and FSUM
together are worth 1.5% of prefill for about 13K LUTs a slice, where VSIG off costs 6.2%, VMAC
off 5.1%, the pump off 17.9%; that cut could not keep VSIG, which needed CNS 5. `x2:b2` builds
CNS 3 with a VSIG at six passes and FSUM off: about 8.6K LUTs a slice less than CNS 5 with VSIG
(estimated from a yosys out-of-context difference of the VPU), for 3.2% of prefill on the later
class fits (4.4% with LDA and the DMA's second command, [dma_ldq.md](dma_ldq.md)).

## 5. The split of a copied KV head (`KVSPLIT`)

With fewer KV heads than slices (the 35B on four slices: two KV heads, each copied on two
slices), each slice of a pair reads the whole head's cache every token, and DRAM binds again at
long context. `KVSPLIT = N` (`[arch]`, `OTPU_KVSPLIT`; a compile choice, not in the CFG_HASH)
splits a head's blocks between its copies in decode buckets of N blocks or more: each copy
attends its half, the pair exchanges the partial softmax state (m, l and the accumulator rows)
over the collective hub, and each slice merges its own queries. Both copies still append every
token, so the cache stays whole. Two-card co-simulation, the 35B cut at 31k: 263,983 -> 254,777
cycles (-3.5%), with the MXU and VPU at 2x 261,539 -> 237,586 (-9.2%); break-even at about 2,750
positions. It changes the attention's bits past the split's first bucket (position 2,816 at
KVSPLIT 12), so references there are regenerated. [x2.md](x2.md) has the two-card side.

## 6. Measured (board model, one card, cycles a token)

`bcu1525-wg:wmx` (the pump, VPU_CNS 5) against `bhw16` at 31,000 positions: **-9.4%** on the 35B
cut, **-4.2%** on the 27B's, **-9.3%** on GPT-OSS-20B's; the 35B cut at position 60 is within
0.2% (decode at the DRAM's rate stays).

| point | wmx, WPB 1 | wmx, WPB 2 | wmx + FSUM (WPB 1) |
|---|---:|---:|---:|
| 35B cut, r31000 | 579,061 | 576,994 (-0.36%) | 576,079 (-0.52%) |
| 27B cut, r31000 | 2,181,754 | 2,183,944 (+0.10%) | |
| GPT-OSS cut, r31000 | 2,479,888 | 2,470,891 (-0.36%) | 2,350,780 (-5.2%) |
| 35B cut, r60 | 348,949 | 346,054 (-0.83%) | |
| 27B cut, r60 | 1,830,110 | 1,828,703 (-0.08%) | |
| GPT-OSS cut, r60 | 1,580,650 | 1,581,143 (+0.03%) | |

The pumped RTL matches the "MXU 2x" oracle of section 1 within 0.4% (board model, the one-card
target; from the study's log).

## 7. Building it

`bcu1525-x2:wmx` = `st16` + MXU_PUMP, VPU_CNS 5, TMEM WPB 2, FSUM, VSIG, VMAC and the instruction
cache ([icache.md](icache.md): 64 lines, programs to 16,384 instructions). Density, from the
routed two-card build without the pump: the slice dies at 62% of their LUTs, 33% of their
flip-flops, 66% of their DSPs, 49% of their UltraRAM, with 94-95% of the CLB sites touched (the
placer's packing, not resources).

**No bitstream yet.** The one build of the wmx set (100 MHz, the development directives) routed
and failed the build's gate: setup WNS -2.949 ns (78,832 failing endpoints), hold WHS -0.860 ns,
pulse width -0.177 ns. Placement screens from its checkpoint found one setup-clean start (each
slice's channel memories moved to the middle die, with the placer's `AltSpreadLogic_high`: WNS
+0.048 ns after placement); the build from that start was paused before it routed, and the
150 MHz build of `x2:b2` was not launched. The same RTL without the pump, synthesized at 150 MHz
with the full directives against 100 MHz with the development ones, is 11.8% more LUTs after
synthesis and, placed, 7.4-7.8% more LUTs and 1.7-1.8 points more CLB sites in the slice dies:
projections from a 100 MHz development build have to add that.

## Tests

`tests/test_wg_rtl.py` test_pumped_* (every format, RMAX / ASCALE, the packed PE's extremes,
posted UltraRAM lanes with stalls and four queued commands, KT), `tests/test_rtl.py`
test_one_copy_two_writes_* (WPB 1 and 2 bit-exact against the ISA, two writers, replays),
`tests/test_se_vpu.py` (the composites at CNS 3 and 5, FSUM's take, VSIG), `tests/test_vops.py`.
