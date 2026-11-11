# The state-step engine (SE_R): the streams beside the VPU

DSTEP and STREAM, the instructions that step a Gated DeltaNet head's state, run on a stream
engine: the DMA streams the state's rows from DRAM, the engine computes each row's two dot
products and its rank-one update, and the DMA streams the rows back
([isa.md](isa.md) "DSTEP", "STREAM"; [seq_vpu.md](seq_vpu.md) section 9). By default that engine is part
of the VPU, and a state step holds the VPU for its whole length. `SE_R` moves the streams onto an
engine of their own, `otpu_se_eng` (`rtl/vpu/otpu_se_tail.sv`), with `SE_R` row groups:

| `SE_R` | the streams run on | rows a cycle | VOPs during a state step |
|---|---|---|---|
| 0 (default) | the VPU's engine | 1 | wait |
| 1 | `otpu_se_eng` | 1 | run |
| 2 | `otpu_se_eng` | 2 | run |

It is a `[uarch]` key (`OTPU_SE_R` in the simulators): timing only. The ISA, the compiler's
programs and every result are the same bit for bit at any value; SE_R 0 builds today's RTL,
cycle-exact, and none of the engine. `targets/bcu1525-wg.toml`, `bcu1525-s2` and `bcu1525-x2` set
SE_R 2; the YPCB-00338 and the ASIC targets keep 0.

## Why an engine of its own, and why row groups

On a decode token of a DeltaNet model the state traffic is small next to the weights, but each
DSTEP is long (8 state words a cycle: 2,048 cycles a head of 128 x 128), and while it runs the
VPU can do nothing else: the mixer's VOPs (its convolution, norms and gates) and the next layer's
queue up behind it. Three other forms were bounded in co-simulation and ruled out:

- **Wider DSTEP lanes (16 or 32 a row) are not exact.** A row's dot products are partial-sum
  loops, and splitting a row's columns over more lanes changes the order of the adds. The exact
  form of "2x" steps two *rows* a cycle: row groups in lockstep, each row's operations the ones
  it has today.
- **A schedule that overlaps the steps with the VOPs** needs the VPU free during a step, which
  is the engine of its own; with it, the chain of the DMA's steps bounds the mixer.
- **Wider requests for the state's DRAM traffic** buy nothing: with the state's traffic removed
  altogether a token moves by a fraction of a percent.

So the design is both: an engine beside the VPU (no turns between streams and VOPs) with two
rows a cycle (the rate).

## The design

- **`otpu_se_eng`** is the VPU's stream engine as a unit, with R row groups. Per group: dot A's
  eight multiply-adds in a loop of 8 cycles (the VPU's RDOT path, bit for bit), a folding tree
  (`otpu_vtree`), and the tail's group (`otpu_se_tail`: the U update's eight multiply-adds, dot
  Q's multipliers and adders, the d stage). A cycle takes the same segment of rows `R i .. R i +
  R - 1` (R x 8 state words). The column buffers (k, q, the gate, B, x), the modes and the
  control are shared, since every group is at the same column in the same cycle. Each row's
  operations are those of R = 1, so the results are exact by construction, and the tests check
  them word for word.
- **`otpu_dma` SR = R** is the DMA's stream path at R rows a cycle: reads and writes go per row
  step, chunk column, row. The chunk buffer stays a FIFO; its chunks move through R chunk
  registers into the engine (two sets, ping-pong), and the updated segments collect into R chunk
  registers a bank and go into the write gather a chunk a cycle. The run structure is unchanged:
  reads in runs of 64 chunks, writes once 64 are gathered.
- **The VPU** keeps its composites' tail (EXP2, RECIP, RSQRT, the SiLU chain) and loses its
  stream half. Sharing the engine's multipliers with the composites would make them wait on the
  streams again.
- **Limits.** R is a power of two up to D / 32 (the segments in a chunk: 4 at D 128) with 8
  lanes. A row's segments are whole chunks. The last row step's missing rows are fed +0 and
  their outputs dropped.
- **No SRAM copies.** The engine's column buffers are filled from TMEM once a stream, as the
  VPU engine's are; the chunk registers hold data in transit.

`DMA_DEPTH` (`[uarch]`, `otpu_dma` DEPTH: the LD chunk buffer, a DSTEP's read-ahead; a power of
two, default 128) goes with it: at 512 it costs the same block RAMs (29 RAMB18 at any depth up
to 512) and lets a DSTEP's reads run further ahead of the engine.

`Config.SE_OWN` is derived from SE_R > 0 with DSTEP (`opentpu.target` DERIVED): a program-only
field that lets the compiler choose layouts that assume VOPs run beside a step (Qwen3.5's
`pair_loop`, the unrolled pairs of `qwen35.PAIR_LOOP_OWN`).

## Area

Vivado out-of-context synthesis on a larger UltraScale+ part than the BCU1525's VU9P (measured;
synthesis, not routed, and not the BCU builds' own counts):

| unit | LUTs | flip-flops | DSPs | block RAM tiles |
|---|---:|---:|---:|---:|
| `otpu_se_eng` R 1 | 19,311 | 17,391 | 86 | 4 |
| `otpu_se_eng` R 2 | 36,825 | 31,544 | 172 | 7.5 |
| `otpu_dma` SR 1 (SE_R 0 and 1) | 10,089 | 4,459 | 0 | 14.5 |
| `otpu_dma` SR 2 | 13,380 | 14,190 | 0 | 14.5 |

In a slice (a slice's synthesis alone on the same part, WG 8 parameters): SE_R 1 adds about 15K LUTs and 82
DSPs over SE_R 0, SE_R 2 about 39K LUTs, 38K flip-flops and 164 DSPs (the engine +35.6K LUTs,
the DMA +3.1K LUTs and +9.8K flip-flops, most of them its eight 1,024-bit chunk registers; the
VPU -2.3K, its stream half trimmed). An R 1 engine by part: dot A's multiply-adds 4.3K LUTs and
32 DSPs, the folding tree 3.6K, the tail's U 4.5K and 32 DSPs, the tail's Q 3.4K and 16 DSPs,
buffers and control 1.4K and 4 RAMB36; 35 fp32 adders at about 307 LUTs each are 60% of it. One
cut was free: dot A as `a*b + c` (the partial loop's own value) in place of `a*b + c*1.0`, bit
for bit the same under flush-to-zero, -0.8K LUTs and -16 DSPs.

## Results on the BCU1525

Qwen3.5-4B (fp4, int8 head) at 150 MHz on one card, decode at position 60
([bcu1525_wg.md](bcu1525_wg.md) has the runs):

| build | a DeltaNet layer (cycles) | a token | tokens/s |
|---|---:|---:|---:|
| SE_R 1, DMA_DEPTH 128 | 197.1K | 7,344.8K | 20.43 |
| SE_R 2, DMA_DEPTH 128 | 177.7K | 6,840.4K | 21.93 |
| SE_R 2, DMA_DEPTH 512 | 175.4K | 6,770.8K | 22.15 |
| SE_R 2, DMA_DEPTH 512, group-major pairs (`pair_loop=1`) | 172.2K | 6,741.4K | **22.26** |

+9.0% tokens/s from the engine's second row group and the deeper read-ahead, 90% of the token's
floor at the DIMMs' measured stream rate (24.7 tokens/s). A DSTEP alone on the card: 2,303 cycles
a head at SE_R 1, 1,448 at SE_R 2, 1,273 with DMA_DEPTH 512 (the board model: 2,275 / 1,389 /
1,264: 1.2%, 4% and 0.7% under the card, with no weight stream beside it).

SE_R 1 cannot reach the floor: its DSTEPs take 4.6K cycles a pair of heads where the DIMMs need
3.8K for the bytes they move. What is left at SE_R 2 is not the engine: beside a weight stream a
head's 128 KiB of state traffic costs the card about 935 cycles against about 300 for its bytes,
because the state's writes reach the DDR4 controller a few at a time and each group pays the
bus's read-to-write turnarounds. The lever there is the write path, not a third row group.

## Tests

`tests/test_stream_rtl.py`, `tests/test_rtl.py` and `tests/test_qwen35.py` run at `OTPU_SE_R` 1
and 2 against the ISA simulator (every DSTEP and STREAM word); `tests/test_se_vpu.py`
(`sim/verilator/tb_se_vpu.sv`) drives the engine and the VPU's composites as the DMA does, with
independent VOPs and random grants. SE_R 0 is cycle-exact against the RTL before the parameter.
