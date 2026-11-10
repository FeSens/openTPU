# The card's infrastructure: what it provisions, what the work uses, and what cutting it buys

Question (2026-10-12): outside the MXU, TMEM and the sequencer / VPU, what does the YPCB build
provision that its workloads do not use? This covers the host link (XDMA), the DRAM controllers
(LiteDRAM with its calibration CPU), the native DRAM adapter and the channel bridges, the DMA,
the quantizer, the trace buffer, the collective unit, the control block, the ACT RAM and IMEM.
And for each cut: what does it cost in cycles, and what would the freed resources buy?

The study works as follows:
- **Inventory.** The production build's units (91a9c2d3: xc7k480t-2, 133.33 MHz) and the MX
  column probe (999b0bc0).
- **Hypotheses.** Ranked by how much each unit provisions against how much of it is used.
- **Demand.** Measured in the RTL co-simulation on four workloads: decode, prefill, batched decode
  and MoE. New `+memstat` instrumentation records the memory path's occupancy cycle by cycle.
- **Bounds.** Timing-only what-ifs: depth parameters in the co-sim, the scoreboard.
- **Area.** Out-of-context Vivado runs of the candidates on the card's part, and the full build's
  checkpoint broken down per unit.
- **Report.** Section 9's table, with each change's cost and the decisions it needs.

Every number says whether it is **measured** (Vivado), **simulated** (the RTL co-sim) or
**estimated**.

**Short answer: the infrastructure the workloads touch is sized to what they use. The slack is
in hardware they never touch: diagnostics, calibration and the collective unit. For the ASIC
that openTPU is the architecture of, the bigger finding is data that moves more than once
(section 10).**

- **The memory path is full because the DRAM is.**
  - In decode the controller moves 93-94% of the peak.
  - The adapter's and bridges' queues hold the ~86 reads per channel that takes: the B
    FIFO's reservations at p99 106 of 128, the bridge's credits at p50 61 of 64.
  - Every weight-streaming workload sits at that ceiling: decode, batched decode, MoE rows and
    layer runs, two-row prefill.
  - Every depth cut tried costs 0.25-1.0% of decode and lowers DRAM efficiency: the B FIFO to
    64, the credits to 32, the A FIFO to 32. **None is taken** (2026-10-12).
- **The DMA's stream runs at 32 chunks** (-2.1K LUT) cost Qwen3.5 decode 0.27% (at 16: -2.7K
  LUT, 0.93%), and a DeltaNet layer 0.7-2% in MoE decode. Rejected.
- **The credits doubled (ARD 128)** gain 0.03-0.24%: the B FIFO becomes the limit. Not taken.
- **XDMA's configuration is already minimal.** The documented options free 73 LUT and a RAMB36;
  the rest of its 16.8K LUT are the DMA engines and the 7-series PCIe adapter. No change; the
  hidden `enable_resource_reduction` (-544 LUT) stays off.
- **LiteDRAM**: about a third of the core (~7K LUT, ~5K FF, 4 RAMB36 + 3 RAMB18) is
  calibration hardware that idles after configuration. Each channel has its own copy. Sharing it
  is core surgery on the PHY's tightest paths, not a parameter: a future lever, not now.
- **The ready levers**, both on this branch (the collective's default changed in release r2):
  - **The trace buffer** (32 RAMB36, ~2.3K LUT, ~3.9K FF; no workload cycles): the build knob
    `TRACE_DEPTH=0`. Production keeps 16,384 records, the card's debugging tool; the MCOLS 8
    build takes 0 if it is short of block RAM.
  - **The collective unit at S = 1** (~2.0K LUT, ~0.9K FF): GATHER is never compiled at
    S = 1. `COLL_GATHER=0` builds a BAR-only unit, announced by CAPS2 bit0 (register map 5); the
    host refuses a GATHER program on it (an ISA decision, for the MCOLS 8 build). Since
    release r2 the card builds it by default (section 5).
- **The quantizer, the ACT RAM and IMEM are not over-provisioned.**
  - The ACT RAM is width-bound, and ACT_ROWS 16 fills it for free.
  - IMEM binds long-context prefill. An 8K IMEM would cost what the trace frees; it is parked
    until the looped attention is costed.
- **Data movement** (section 10):
  - every DRAM read is reserved and buffered two or three times on its way;
  - a V^T append reads and writes a 64-byte beat per byte: 73% of a Qwen3-0.6B token's writes,
    1.0-1.4% of its time (measured), 0.2-0.3% of the 3-4B models';
  - the DeltaNet state goes to DRAM and back every token (6.6% of Qwen3.5-0.8B's DRAM beats);
  - the host copies every upload once to interleave the channels;
  - the S > 1 collective is a serial, blocking all-gather through one hub. PUT / PUTADD link
    engines replace it (10.4, the approved direction): Qwen3-0.6B's collectives would
    take ~8K cycles a token at S = 2 instead of ~175K (estimated).

**Recommendation:**
- Change nothing in the next production build unless it needs room.
- If the MX column's MCOLS 8 build is short, take `TRACE_DEPTH=0` (block RAM) and
  `COLL_GATHER=0` (LUTs).
- For the ASIC: one landing buffer per channel sized to bandwidth x latency, credits against the
  consumer's buffer, and the slice-to-slice link of section 10.3.

Section 9 has the table.

## 1. Inventory: the production build (91a9c2d3)

The production image is 177,292 LUT (59.4%), 134,913 FF, 585 RAMB36 + 35 RAMB18 (63%), 716 DSP
(37%) and 76.6% of the slices. The study's units total about 91.6K LUT, 52% of the design
(**measured**, routed `util_hier.rpt`):

| unit | LUT | of which LUT RAM | FF | block RAM | DSP | what it is |
|---|---|---|---|---|---|---|
| XDMA (`xdma_0`) | 16,832 | 1,725 | 18,730 | 23 RAMB36 + 1 RAMB18 | | the PCIe Gen2 x8 DMA IP. Of this, `udma_wrapper` is 12,712 LUT and the `pcie2_to_pcie3` shim 4,120 |
| SmartConnect `sc_ctl` + XADC | 438 | | 1,227 | | | BAR0's AXI-Lite: the control registers, the LiteDRAM CSRs, temperature |
| LiteDRAM (`u_ld`) | 20,583 | 324 | 17,335 | 4 RAMB36 + 3 RAMB18 | | two DDR3-1066 controllers with ECC, their PHY, BIST, CSRs, and the calibration CPU (VexRiscv: 1,065 LUT, 730 FF) |
| native DRAM adapter (`u_mem`) | 14,328 | 6,596 | 3,767 | | | `otpu_native_dram`: the slice's port A / B / SW requests onto the two channels, their read FIFOs |
| channel bridges (`u_ch0`, `u_ch1`) | 10,450 | 5,332 | 4,602 | | | `otpu_mem_ch`: the accelerator and XDMA onto each channel's two native ports, clock crossings |
| XDMA's split (`u_split`) | 1,352 | | 2,200 | | | `otpu_dma_split`: XDMA's AXI by address bit 31, three AXI register slices at 250 MHz |
| `otpu_native_sys` glue | 1,466 | | | | | |
| DMA (`u_dma`) | 11,025 | 3,080 | 3,882 | 14 RAMB36 + 1 RAMB18 | | LD / ST / DSTEP / STREAM. The line buffer `u_lb` (1024 bits x 128) shows 3,932 LUT |
| quantizer (`u_quant`) | 7,924 | 439 | 6,969 | | 66 | QACT with the RMSNorm gain prescale (`u_sc` 2,673 LUT, 16 DSP) |
| trace buffer (`u_trace`) | 2,291 | 856 | 3,901 | 32 RAMB36 | | 16,384 records x 64 bits |
| collective unit (`u_coll`) | 2,343 | | 906 | | | BAR and GATHER |
| control block (`u_ctrl`) | 866 | | 2,434 | | | registers, performance counters |
| ACT RAM (`u_act`) | 958 | | 1,605 | 64 RAMB36 + 4 RAMB18 | | |
| IMEM (`u_imem`) | 601 | | | 28 RAMB36 + 1 RAMB18 | | 4,096 instructions x 256 bits |

The finer breakdowns are in each unit's section: leaf cells by name from a routed image's
checkpoint (`breakdown.tcl` on 542fc43a, the same units as 91a9c2d3's), and the LiteDRAM core
alone.

**The same units on UltraScale+** (**measured**, synthesis out of context on xcku5p-2 at 200 MHz,
`run_ooc.sh ku-*`; against xc7k480t-2's out-of-context numbers):

| unit | LUT (of which LUT RAM) | FF | block RAM | xc7k480t-2 |
|---|---|---|---|---|
| `otpu_native_dram` | 13,928 (5,756) | 3,774 | | 15,520 (6,660) LUT, 3,777 FF |
| `otpu_mem_ch` (one channel) | 4,478 (2,294) | 2,257 | | 5,232 (2,680), 2,243 |
| `otpu_dma`, RUN 64 | 10,154 (2,800) | 4,159 | 14 RAMB36 + 1 RAMB18 | 10,375 (3,096), 4,162 |
| `otpu_dma`, RUN 32 | 7,984 (1,520) | 4,021 | the same | 8,239 (1,688), 3,989 |
| `otpu_trace`, 16,384 records | 2,224 (750) | 3,959 | 28 RAMB36 + 1 RAMB18 | 2,314 (862), 3,921, 32 RAMB36 |
| `otpu_coll`, S = 1 | 2,080 | 987 | | 2,081, 987 |

The units come out 0-15% smaller in LUTs (the LUT RAM most), with the same flip-flops; the cuts
free the same amounts (RUN 32: -2,170 LUT against -2,136). Nothing in the study depends on the
7-series.

## 2. The memory path: the depths decode uses

The path from the slice to the DRAM, per channel:
- **The adapter** (`otpu_native_dram`):
  - request queues: B and A at QD 16, the SW gather at WQD 32;
  - the B read FIFO: RD 128 beats of 512 bits, its room reserved per read at issue;
  - the A read FIFO: AD 64, with APF 32-beat runs;
  - the tags of the reads in flight.
- **The bridge** (`otpu_mem_ch`, LiteDRAM's sys clock, 133.33 MHz from another MMCM):
  - the command and write-data FIFOs into that clock (32);
  - the read credits and read-data FIFO back (ARD 64, in order by slot);
  - per port (bank parity): the output command queue (OQ 32), write-data FIFO (OD 32) and
    read tags (128).

Nearly all of it is LUT RAM, which comes in 32- and 64-deep pieces, so only depths that cross
128 -> 64 or 64 -> 32 free LUTs.

**What the cuts free** (**measured**, out of context on xc7k480t-2):

| unit | depths | LUT | LUT RAM | FF | against the board's |
|---|---|---|---|---|---|
| `otpu_native_dram` | the board's | 15,520 | 6,660 | 3,777 | |
| | RD 64 | 13,937 | 5,244 | 3,701 | -1,583 LUT |
| | AD 32 / APF 16 | 14,857 | 5,980 | 3,761 | -663 LUT |
| | RD 64, AD 32 / APF 16 | 13,110 | 4,524 | 3,678 | -2,410 LUT |
| | RD 32 | 13,174 | 4,512 | 3,654 | -2,346 LUT |
| `otpu_mem_ch` (one channel) | the board's | 5,232 | 2,680 | 2,243 | |
| | ARD 32 | 4,769 | 2,332 | 2,199 | -463 LUT a channel |

Placed and routed (**measured**, 7.5 ns), every variant closes with margin: the adapter +0.32 ns
(the board's) and +0.63 ns (RD 64, AD 32), the bridge +0.58 / +0.42 ns.

**What decode uses** (**simulated**: the co-sim at DDR3-1066 and 133.33 MHz, `+memstat`, one
token of Qwen3-0.6B at position 544, fp4; per channel):

| queue / FIFO | depth | p50 | p99 | max |
|---|---|---|---|---|
| adapter: reads in flight (tags) | 256 | 85 | 93 | 96 |
| adapter: B read FIFO, reserved | 128 | 86 | 106 | 112 |
| adapter: B read FIFO, filled | 128 | 1 | 45 | 112 |
| adapter: A read FIFO, reserved | 64 | 18 | 34 | 47 |
| adapter: B request queue | 16 | 13 | 16 | 16 |
| adapter: A request queue | 16 | 0 | 1 | 1 |
| adapter: SW gather queue | 32 | 0 | 25 | 32 |
| bridge: read credits in use | 64 | 61 | 63 | 64 |
| bridge: command FIFO into sys | 32 | 27 | 32 | 32 |
| bridge: a port's command queue | 32 (+1) | 2 | 33 | 33 |
| bridge: a port's write-data FIFO | 32 | 0 | 3 | 32 |
| bridge: a port's read tags | 128 | 15 | 46 | 51 |

The other workloads (**simulated**, the board's depths, channel 0):

| workload | B reserved p50 / p99 (of 128) | A reserved p50 / p99 (of 64) | credits p50 / p99 (of 64) |
|---|---|---|---|
| Qwen3.5-0.8B decode | 86 / 106 | 18 / 33 | 61 / 63 |
| LFM2 decode | 87 / 105 | 18 / 34 | 61 / 63 |
| 4-slot batched decode (Qwen3-0.6B) | 84 / 106 | 17 / 32 | 61 / 63 |
| MoE decode rows (Qwen3.5-35B-A3B) | 88 / 106 | 17 / 33 | 62 / 63 |
| MoE layer run, 2 rows (Qwen3.5-35B-A3B) | 74 / 106 | 16 / 33 | 61 / 63 |
| MoE layer run, 2 rows (Gemma 4 26B-A4B) | 77 / 106 | 21 / **63** | 61 / 63 |
| prefill, 2 rows, zero image (Qwen3.5-4B) | 75 / 106 | 18 / 34 | 61 / 63 |
| prefill, 4 rows (Qwen3-0.6B / Qwen3.5-0.8B / LFM2) | 11 / 85-94 | 9-16 / 32-34 | 9-10 / 62 |

- **Everything weight-streaming sits at the same ceiling.** Decode, batched decode, MoE, and
  prefill at one or two rows on the larger models all hold B reservations at p99 106 and
  credits at 61-63.
- **Gemma 4's MoE layer runs use the whole A FIFO** (p99 63 of 64).
- **Four-row prefill on the small models** is compute-bound: the path idles at p50, but its
  bursts still reach the same limits.

- **The path is full because the DRAM is.** The controller moves 92.8% of the core port's
  peak, and the queues hold the reads it needs in flight: about 86 B beats per channel, ~0.93
  beats a cycle times the ~90-cycle round trip through the clock crossing and the controller.
- **Where it binds:**
  - the bridge's credits are what limit the reads at the controller (p50 61 of 64);
  - the reads beyond them wait in its command FIFO (p50 27 of 32).
- **The B FIFO's 128** covers the reservations' p99 of 106.

**What the cuts cost** (**simulated**, cycles per token and the DRAM efficiency):

| candidate | Qwen3-0.6B decode | Qwen3.5-0.8B decode | 4-slot batched | DRAM efficiency (Qwen3 / Qwen3.5) |
|---|---|---|---|---|
| the board's | 3,623,644 | 4,760,853 | 1,671,279 | 92.8% / 94.1% |
| RD 64 | +0.96% | +0.80% | | 91.9% / 93.4% |
| ARD 32 | +0.44% | +0.41% | | 92.4% / 93.8% |
| AD 32 / APF 16 | +0.25% | +0.31% | | 92.5% / 93.9% |
| all three | +1.00% | +1.06% | +0.70% | 91.9% / 93.2% |

**Every cut lowers DRAM efficiency, so none is taken.**
- The ~3.3K LUT they would free cost 0.25-1% of decode.
- The memory path's ~26K LUT (adapter, bridges, split) are mostly these FIFOs, and they are
  used: they are the bandwidth-latency product of a controller that is 93-94% busy.
- docs/litedram.md ("Decode after the fused build: the controller's share") reached the same
  place from the other side: the controller spends ~94% of decode's cycles on data, and 0.6%
  with nothing asked.

**The other direction: the credits doubled (ARD 128)**, which `otpu_mem_ch`'s tag FIFO now
allows (TGD, sized from ARD + XRD; the board's netlist and cycles unchanged at ARD 64: the
`basechk` runs equal the base to the cycle). Simulated:

| workload | ARD 64 | ARD 128 | change |
|---|---|---|---|
| Qwen3-0.6B decode | 3,623,644 | 3,618,573 | -0.14% |
| Qwen3.5-0.8B decode | 4,760,853 | 4,757,718 | -0.07% |
| MoE decode rows, Qwen3.5-35B-A3B (4-layer prefix) | 5,490,770 | 5,480,351 | -0.19% |
| MoE layer runs, Gemma 4 26B-A4B (two layers) | 969,765 / 993,612 | 969,468 / 991,259 | -0.03% / -0.24% |

In Qwen3.5 decode the controller takes more reads in flight (credits p50 82, p99 101 of 128,
against 61 / 63 of 64), the B FIFO becomes the limit (reservations p50 106, p99 127 of 128),
and DRAM efficiency moves from 94.1% to 94.2%. Doubling the credits, and the B FIFO with them,
would cost ~0.7K LUT a channel and more for 0.03-0.24%: the DRAM is the limit, not the
credits. **Not taken.**

## 3. The DMA

`otpu_dma` (11,025 LUT, 3,080 LUT RAM, 3,882 FF, 14 RAMB36 + 1 RAMB18 routed).

**The line buffer** (`u_lb`, 1024 bits x 128 chunks, 14.5 block RAM tiles) is width-bound like
the ACT RAM: it uses 128 of the RAMB36's 512 rows. The 3.9K LUT the report puts under it (3,766
and no flip-flops out of context too) are not the RAM. They are the selects of everything that
reads the line, which synthesis groups with it. Out of context, by output net (**measured**,
3,897 LUT):
- port B's write data (`b_wdata`, 1,024: the line or a stream's gather);
- LD's TMEM write data (`t_wdata`, 448);
- the stream engine's input (`ss_in_d`, 514);
- WAITW's word (256);
- ~900 on the block RAMs' outputs.

In a routed image's leaf cells (542fc43a, `breakdown.tcl`) the same logic goes by other names:
- 2,291 LUT drive the adapter's B queue (`g_mem.qbm`): a chunk's two 64-byte beats onto their
  channels by the hashed interleave;
- 1,079 sit on the line's read data;
- 948 hold ST's chunk gather (`cb`).

They are what one 1024-bit buffer serving LD, ST, the streams and WAITW costs. There is nothing
to cut short of separate buffers.

**The stream runs** (`RUN`, now a parameter): a DSTEP or STREAM reads and writes the DRAM in
runs of RUN chunks, and its write gather holds 2 x RUN chunks of 128 bytes in LUT RAM. RUN went
from 16 to 64 in 63260142 (one read/write turnaround per run): Qwen3.5 decode -3.4% then, on
the old DDR3 model, together with ST's SRUN.

| RUN | LUT | LUT RAM | FF | routed slack (7.5 ns) |
|---|---|---|---|---|
| 64 (the board's) | 10,375 | 3,096 | 4,162 | +1.18 ns |
| 32 | 8,239 | 1,688 | 3,989 | +1.30 ns |
| 16 | 7,658 | 1,048 | 3,926 | |

(**measured**, out of context on xc7k480t-2.) RUN 32 frees 2.1K LUT, RUN 16 2.7K.

Cost (**simulated**, cycles):

| workload | RUN 32 | RUN 16 |
|---|---|---|
| Qwen3-0.6B decode (no streams) | 0.00% | |
| Qwen3.5-0.8B decode (DSTEP) | +0.27% (DRAM efficiency 94.1% -> 93.9%) | +0.93% (94.1% -> 93.3%) |
| LFM2.5-230M decode | 0.00% | 0.00% |
| Qwen3.5-0.8B prefill, 4 rows (4 layers) | +0.10% | +0.11% |
| LFM2.5-230M prefill, 4 rows | 0.00% | 0.00% |
| MoE decode rows, Qwen3.5-35B-A3B (4-layer prefix) | +0.14% (a DeltaNet layer +0.66%) | +0.35% (+1.8%) |
| MoE layer run, Qwen3.5-35B-A3B, 2 rows: its DeltaNet layer / attention layer | +0.71% / 0.00% | +2.04% / 0.00% |

The cost follows DSTEP's share: nothing where no stream runs, 0.7-2% of a DeltaNet layer.
**Rejected** (with the other depth cuts): the 2.1-2.7K LUT do not pay for slower
Qwen3.5 and MoE decode.

## 4. The trace buffer

`otpu_trace` (docs/observability.md) keeps the slice's trace events as 64-bit records in a ring
of 16,384 (32 RAMB36). The host reads it through two registers per record.

**Who uses it.** On the card it has three users:
- otpu-lens's hardware lens (`opentpu/host/hwlens.py`);
- `tools/batch_bench.py --trace`;
- `tools/q35_trace.py card`.

The qualification (`tools/qual`), serving and chat never turn it on. `Board.run(trace=...)`
raises a clear error on a bitstream without it (CAPS bit0 = 0), and the RTL has supported
`TRACE_DEPTH = 0` since the trace was added (`otpu_board` g_trace, `otpu_ctrl` CAPS).

**What it covers.** A Qwen3-0.6B decode token is ~200K records (**simulated**, the co-sim's
trace lines), so the ring holds ~8% of a token: its last layers, or its first ones with
STOP_WHEN_FULL.

**Area** (**measured**, out of context on xc7k480t-2, `run_ooc.sh k7-trace`):

| depth | LUT | LUT RAM | FF | block RAM |
|---|---|---|---|---|
| 16,384 (the card's) | 2,314 | 862 | 3,921 | 32 RAMB36 |
| 4,096 | 2,297 | 862 | 3,917 | 7 RAMB36 + 1 RAMB18 |
| 0 (none) | 0 | 0 | 0 | 0 |

Depth frees block RAM only. The LUTs and flip-flops are the capture pipeline:
- the slice's event bundle (`perf_t`, ~1,190 bits), registered three times (`pr`, `ev` and the
  unpacker's head: ~3.6K of the 3.9K FF);
- the bundle queue (32 in LUT RAM);
- the unpacker.

The production routed image agrees: 2,291 LUT, 3,901 FF, 32 RAMB36.

**The cut.** This branch passes `TRACE_DEPTH` to the YPCB build:
- `run_vivado.sh TRACE_DEPTH=0`, defaulting to the target's 16,384;
- through `create_project.tcl`, `otpu_fpga_top_ld` and `otpu_native_sys`.

The default netlist is unchanged; `tests/test_target.py` checks the defaults. A build without the
trace frees 32 RAMB36, ~2.3K LUT and ~3.9K FF. It costs no cycles and has no ISA or compiler
impact. What is lost is the card-side lens: profiles then come from the co-sim only, which
matches the card within ~1% (docs/board.md).

The lead's decision (2026-10-12): the knob merges with the default unchanged. Production keeps
16,384 records, the card's debugging tool. The MCOLS 8 build takes `TRACE_DEPTH=0` if it is
short of block RAM.

**A cheaper whole-token trace (estimate, not built).** The records could go to DRAM through
the SW port:
- a 64-byte beat per 8 records;
- at ~200K records a token, that is ~1.6 MB a token, about 0.35% of decode's DRAM traffic at
  1066 MT/s.

Profiling would then cost that bandwidth while on, and nothing while off. It would also need
only a small staging FIFO, not 32 RAMB36. It changes the host decoder and the control block,
so it is a design item, not a parameter.

## 5. The collective unit at S = 1

`otpu_coll` runs BAR and GATHER across slices. The card has one slice:
- `language.all_gather` returns its input at S = 1, so no compiled program emits GATHER;
- no compiled program emits BAR either;
- both appear only in the card's operation checks (`opentpu/host/opchecks.py`).

**Area** (**measured**): out of context 2,081 LUT / 987 FF at S = 1 (2,487 LUT at S = 2); in the
routed image 2,343 LUT / 906 FF. The routed image's leaf cells under it include ~0.9K LUT of the
TMEM's write data and address muxing for the unit's write port (W_COLL: `pw_d`, `pw_a`),
grouped there by synthesis. The rest is its own:
- the strided addresses and counters;
- the read-data register (256 FF) and the lanes' write addresses.

The TMEM read port is shared with RLD and stays.

The TMEM uses only the address's low 16 bits in synthesis, so Vivado already drops the
address adders' upper halves. A narrower `otpu_coll` would save little.

**The cut: a BAR-only unit at S = 1.** The lead approved it on 2026-10-12, as an ISA call under
the user's grant: GATHER only at S > 1, with a capability bit so the host knows, behind a
parameter, for the MCOLS 8 build. It is built on this branch:
- **`otpu_board` `COLL_GATHER`** (the parameter's default 1: the netlist's unit as before; the
  simulations keep it). `run_vivado.sh` passes it through `create_project.tcl`,
  `otpu_fpga_top_ld` and `otpu_native_sys`. It is S = 1 only; elsewhere it is a `$fatal`.
- **The card's default is 0 since release r2** (`run_vivado.sh`; `COLL_GATHER=1 ./run_vivado.sh`
  builds the GATHER unit). The reason is timing, not only area. In r2's m4 build (r2m4_d352e53d,
  133.33 MHz) the GATHER unit's read select (`ssel`) and the write-bank mask fed the slice's
  TMEM arbiter, so they reached the MXU's grant and the MXU's pipeline enables. That group is
  ~330 endpoints, -0.261 ns worst and 12% of core_clk's TNS, plus 2% of clk -> clk2x through
  the pumped banks' read pins. With the unit BAR-only, the collective's port is tied off, and
  the only reader left on P_COLL is RLD, which reads from flip-flops.
  - Nothing the card runs changes: no compiled program has a GATHER at S = 1.
  - The ISA references hold: GATHER is not a Config field. The host's `device_config` of such a
    card equals r2's m4 configuration, and so do the references' keys (`tools/qual/refs.py`).
  - CFG_HASH holds too: `run_vivado.sh` hashes the arch knobs, and `COLL_GATHER` is not one.
- **`otpu_coll_bar`** replaces the unit:
  - BAR is acknowledged two cycles after the request, then a cycle's cool-down, as before;
  - the TMEM read and write ports are tied off, so the collective's write port and its muxing
    leave the TMEM's arbiter;
  - a GATHER is acknowledged, moves nothing and raises STATUS ERROR until the core reset. In
    simulation it is also a `$fatal` unless `+coll_gerr_ok`.
- **Register map 5.** CAPS is full, so the bit lives in CAPS2 (0x5C), bit0 GATHER: set at
  S > 1 and with `COLL_GATHER` 1 (docs/observability.md). On a bitstream before map 5 the host
  takes it as set.
- **The host** (`opentpu/host`):
  - `Board.load_program` raises `NoGather` on a program with a GATHER when the bit is clear,
    naming the slice and the instruction;
  - the operation checks and `otpu-diag` leave GATHER out on such a bitstream;
  - `otpu-diag` accepts map 5 and checks CAPS2's reserved bits.
- **Tests:**
  - `test_caps2_gather_and_the_load_refusal` (the fake transport: maps 3-5, both program
    forms, the operation checks);
  - `test_caps2_coll_bar` (the board model at `COLL_GATHER` 0: CAPS2 reads 0, a BAR program
    copies its data, the host refuses the GATHER program, and loaded past the host the GATHER
    sets ERROR and moves nothing);
  - `test_register_map` and `test_cfg_registers` read map 5.
  - the gate: the full suite's RTL half passed on the pre-merge tree (400); r2's gate replaces the rest.
- **Instance names.** The unit is now `u_sys/u_board/g_coll.u_coll` (or `g_coll_bar.u_coll`).
  The BCU1525 floorplan, its elaboration check and `tools/synth/board_equiv.sh` follow.

What it frees, about 2.0K LUT and 0.9K FF in all:
- the unit's own ~1.4K LUT and ~0.9K FF (**measured**: the routed image's unit, less the TMEM
  muxing grouped under it);
- the collective's write port in the TMEM's arbiter: 601 LUT (**measured**, out of context on
  xc7k480t-2: `otpu_tmem` as the slice builds it, 12,955 LUT with the port live and 12,354 with
  it tied off; the same 4,608 FF and 384 RAMB36).

The stub itself is a 2-bit state and a flag. The full build is the MCOLS 8 build's to measure.

## 6. XDMA: what the host leaves unused

The card's XDMA (`bd_native.tcl`, Gen2 x8, 128 bits at 250 MHz) has:
- one H2C and one C2H channel;
- `xdma_rnum_rids` 8 (the H2C overrun fix: it stays);
- the AXI-Lite master on BAR0;
- MSI with one vector and one user interrupt;
- no descriptor bypass, no AXI-Lite slave, no AXI-Stream.

The host uses the H2C and C2H engines (`/dev/xdma0_h2c_0`, `c2h_0`), the user BAR and the
interrupts (the driver's default mode). Everything the IP offers beyond that is already off.

**The IP's own options**, synthesized out of context at the card's configuration and at each
variant (**measured**, `run_xdma.sh` on xc7k480t-2):

| variant | LUT | FF | block RAM | against the card's |
|---|---|---|---|---|
| the card's | 18,904 | 21,013 | 23 RAMB36 + 1 RAMB18 | |
| `xdma_wnum_rids` 16 -> 8 (C2H) | 18,831 | 20,980 | 22 + 1 | -73 LUT, -33 FF, -1 RAMB36 |
| `c_m_axi_num_write` 8 -> 4 | | | | not allowed: 8 is the minimum |
| `en_axi_slave_if` off | 18,904 | 21,013 | 23 + 1 | 0: not built in DMA mode |
| `axi_id_width` 4 -> 2 | 18,899 | 21,007 | 23 + 1 | -5 LUT |
| `cfg_mgmt_if` off | 18,929 | 21,013 | 23 + 1 | +25 LUT (noise) |
| `enable_resource_reduction` | 18,360 | 20,996 | 23 + 1 | -544 LUT: a hidden parameter, with no display name or documentation in the IP; the HDL is encrypted |
| all of them (lean) | 18,309 | 20,955 | 22 + 1 | -595 LUT (-3.1%), -58 FF, -1 RAMB36 |

(Out-of-context synthesis keeps logic that the routed image trims: the image's XDMA is 16,832
LUT.)

**The card's XDMA configuration is already minimal** (2026-10-12: no change).
- The one documented saving is 73 LUT and a RAMB36 (C2H write IDs). It is not worth a card
  qualification.
- The hidden option's 0.5K LUT would buy an undocumented change in a vendor DMA engine whose
  H2C ordering the host depends on (docs/host.md, "XDMA's H2C overrun").
- The rest of XDMA is structural:
  - the DMA engines (`udma_wrapper`, 12.7K LUT);
  - the 7-series PCIe block's adapter to the PCIe3-style interface the engines use
    (`pcie2_to_pcie3`, 4.1K LUT).

**A different DMA (estimate, not proposed).**
- A LitePCIe-style engine on the 7-series block directly (one H2C and one C2H channel,
  128 bits) is commonly ~6-8K LUT. That would free ~9-11K LUT.
- It replaces the kernel driver, the host transport (`opentpu/host`), the ordering that WAITW
  relies on (docs/host.md), the flash and boot flow, and the card's qualification.
- That is a project of its own, not a cut.

**The bridges around it** (`u_split` and the two bridges' XDMA sides; section 2):
- XDMA's split and register slices cost 1,352 LUT and 2,200 FF;
- each channel's XDMA side ~2.5K LUT, of which ~0.8K is LUT RAM.

Both channels' bridges run in the same clock (LiteDRAM's sys), so one XDMA side could serve
both channels, with the channel chosen per burst in that clock (estimate):
- it frees ~3.5K LUT and ~1.8K FF: one XDMA side, the split's two channel slices, its routing;
- it moves the build's worst path (in u_ch1's XDMA side at 250 MHz, +0.013 ns);
- XDMA addresses one channel per 2 GiB (bit 31), so a burst never spans both, and H2C at
  ~3 GB/s is far below one channel's rate.

It must keep each channel's write counts (`n_wdone`, which WAITW's ordering rests on) and
XDMA's response order. It needs the card's H2C / C2H rates, `selftest`, the H2C sweep and the
WAITW tests on the card. Not built.

## 7. LiteDRAM

The production core (`boards/ypcb-00338/litedram/otpu_litedram.v`, generated by
`tools/litedram/gen_core.py`):
- two DDR3-1066 channels, each with its PHY and controller;
- two user ports per channel, on bank parity (docs/litedram.md 11, "Two user ports per
  channel": +6 to +13% decode), each port with its ECC frontend;
- a BIST per channel;
- the CSR banks;
- the calibration CPU (docs/litedram.md 10).

Synthesized alone (**measured**, `ld_ooc.tcl`, xc7k480t-2) it is 20,742 LUT, 18,114 FF,
4 RAMB36 and 3 RAMB18; in the routed image, 20,583 LUT. The core is one flat module, so its
leaf cells are grouped by name. Flip-flops are exact. LUTs are the leaf cells scaled to the
report's total: two cells can share a LUT, and the leaf count is 15% over.

| function | LUT | FF | block RAM | in use after calibration |
|---|---|---|---|---|
| PHY (`wl7ddrphy`, the serializers' input logic, bitslips, delays) | ~6.3K | 5,490 | | yes |
| ECC frontends (one decoder and two encoders a channel) | ~4.2K | 4,495 | | yes |
| DFI phase injectors (the calibration's command and read-data CSRs) | ~2.5K | 2,520 | | no |
| BIST (generator, checker, per-lane error bits) | ~2.3K | 1,441 | | the qualification's checks |
| CSR banks and bus | ~1.9K | 616 | | mostly not |
| controller (bank machines, command mux, refresh) | ~1.9K | 1,533 | | yes |
| calibration CPU (VexRiscv, its memory and mailbox) | 1,025 | 976 | 4 RAMB36 + 3 RAMB18 | no |
| FIFOs (LUT RAM) | ~1.0K | 96 | | yes |
| other (clocking, resets, the CSR AXI bridge) | ~0.3K | 947 | | |

**What is over-provisioned: the calibration hardware.**
- About 7.3K LUT, 5.3K FF, 4 RAMB36 and 3 RAMB18: the phase injectors, BIST, the calibration
  CPU and most of the CSR banks.
- That is ~35% of the core, and it works for a few seconds per channel at configuration. BIST
  also runs the qualification's checks.
- The two channels each have a full set, though they share the sys clock and the CPU
  calibrates them one after the other.

The cuts, none recommended now. The lead (2026-10-12): the shared calibration engine is not for
now; the ~7K LUT of calibration hardware is a future lever.

| cut | frees | costs |
|---|---|---|
| one calibration engine for both channels: a BIST and a phase-injector capture muxed onto either channel's DFI | ~2.4K LUT, ~2K FF (estimate) | regenerating the core (`gen_core.py`); the firmware and the host's `ddrcal` taking a channel select; the DFI injector's paths, already the write clocks' tightest (`tools/litedram/dfii_q.py`), get a mux; a calibration warm soak and qualification on the card |
| the calibration CPU removed | 1.0K LUT, 1.0K FF, 4 RAMB36 + 3 RAMB18 (measured, docs/litedram.md 10) | the host calibrates again, 25 s per channel at every bitstream load |
| ECC off (64-bit lanes) | ~4.2K LUT, 4.5K FF (docs/litedram.md: ~1.8K LUT and 3.2K FF a channel) | no error detection, which the qualification reads (the ECC counters must be 0); read-modify-write stays (no DM pins). A reliability decision; not recommended |

**Not over-provisioned:**
- the second user port per channel (bought +6 to +13% decode);
- the bank machines;
- the PHY.

DDR3 stays at 1066 MT/s.

## 8. The quantizer, the ACT RAM and IMEM

**The quantizer** (`otpu_quant`: 7,924 LUT, 6,969 FF, 66 DSP). Its multipliers do real work:
- the prescale (`g_pre`) carries the fused RMSNorm gain (cscale / rscale), which the compiler
  uses (`fuse_cscale`);
- the scale path (`u_sc`) computes the per-row and per-block scales.

How busy it is (**simulated**, the co-sim's unit counters, board configuration):

| workload | QUANT busy | for comparison |
|---|---|---|
| Qwen3-0.6B decode | 9% | VPU 16% |
| Qwen3.5-2B decode | 2% | DMA 20% |
| LFM2 decode | 5% | |
| Qwen3-0.6B prefill, 4 rows | 20% | VPU 36% |
| Qwen3-0.6B prefill, 8 rows | 27% | VPU 52% |
| LFM2 prefill, 4 rows | 13% | |
| batched decode, 4 / 8 slots (Qwen3-0.6B) | 5% / 8% | |

At half its lanes, prefill's 20-27% would become 40-54%, enough to put it on prefill's path
next to the VPU. The DSPs it would free are not short (37% used). The MX column (dsp-pump's
mx-column, probe 999b0bc0) rebuilds it anyway: 7,635 LUT, 64 DSP, 4 RAMB18. **No cut here**;
its size should follow the MX column's rate, not this study.

**The ACT RAM** (`otpu_actram`: 64 RAMB36 + 4 RAMB18). It reads MCOLS x D bytes a cycle (4,096
bits), so it is width-bound: 16 memories of 256 bits, four 64-bit RAMB36 each. At ACT_ROWS 4
it uses 128 of each RAMB36's 512 rows (25%). **ACT_ROWS 16, which comes with the MX column,
fills the same 64 RAMB36 for free** (one group per 128 rows). There is nothing to cut, and the
growth costs no block RAM.

**IMEM** (4,096 instructions x 256 bits: 28 RAMB36 + 1 RAMB18, fully used). It binds in a few
places:
- **long-context prefill** (docs/prefill.md 10): in bucket 16 (positions 3840-4095),
  Phi-4-mini's mix and LFM2-2.6B prefill one row a run, because a two-row run is 4,538 / 5,116
  instructions. That is 3-4x the weight passes of a three- or four-row run;
- **tiled prefill past ~1K tokens**: 4,282 instructions at 1,000 tokens, 7,424 at 2,000 at
  MCOLS 4 (docs/tiled_prefill.md);
- **Gemma 4 E2B's mix in bucket 2**: 4,106, fixed by the additive mask (3,975);
- **decode buckets that do not fit**: these run as two chained programs per token, at a
  program load's cost.

The scalable fix is the looped attention (docs/prefill.md 10, a compiler item): it makes a
run's program independent of the context. The scoreboard with IMEM unlimited moves prefill by
+0.2% (**estimated**). That figure misses long contexts, where the row count per run is the
lever.

**An 8K-instruction IMEM** (estimate):
- +28.5 RAMB36: almost exactly what the trace buffer frees (32);
- an arch change (`IMEM_WORDS` 65536: the arch hash, the compiler's fit checks, the program
  caches);
- a deeper block RAM column on the sequencer's fetch path. That is seq-vpu's timing to check.

Parked (2026-10-12) until the looped attention is costed. It is the cheapest
stopgap if long prompts on Phi / LFM2 matter first.

## 9. Report

Device: xc7k480t-2 (298,600 LUT, 597,200 FF, 955 RAMB36). Cycles are the co-sim's (DDR3-1066,
133.33 MHz): decode / prefill / batched / MoE, "0" where the change touches no path a workload
uses.

| opportunity | resources freed | cycle cost (decode / prefill / batched / MoE) | what it enables | ISA / numerics / compiler | risk | decisions | next production build |
|---|---|---|---|---|---|---|---|
| **Trace buffer out** (`TRACE_DEPTH=0`, this branch) | 32 RAMB36, ~2.3K LUT, ~3.9K FF (measured) | 0 / 0 / 0 / 0 | the block RAM of an 8K IMEM (28.5 RAMB36); slack in a full build | none: CAPS bit0 = 0; the hardware lens and `batch_bench --trace` refuse such a bitstream | low: a generic, default unchanged, lint at 0 | **taken as a knob**: production keeps 16,384, the card's debugging tool | the MCOLS 8 build, if it is short of block RAM |
| **Trace at 4K records** | 25 RAMB36 (measured) | 0 | as above, less | none (CAPS reports the depth) | low | the lens sees 2% of a token, not 8% | no: the BRAM is not short |
| **Collective unit BAR-only at S = 1** (`COLL_GATHER=0`, this branch) | ~2.0K LUT, ~0.9K FF (measured: the unit's ~1.4K, the TMEM write port's 601) | 0 / 0 / 0 / 0 | slack | **ISA**: GATHER only at S > 1; CAPS2 bit0 (register map 5); the host refuses GATHER programs on such a bitstream | low: a parameter (the card's build default 0 since release r2, section 5); lint at 0 and 1; board-model test | **approved** (the maintainer, an ISA call under the user's grant); the card's default in r2 | the MCOLS 8 build; r2's m4 card (timing) |
| Memory path depths (RD 64, ARD 32, AD 32) | up to ~3.3K LUT (measured) | +0.25 to +1.0% decode, +0.7% batched (simulated) | | none | | **rejected** | no: each lowers DRAM efficiency |
| DMA stream runs (RUN 32 / 16) | 2.1K / 2.7K LUT (measured) | Qwen3.5 decode +0.27% / +0.93%; prefill +0.10% / +0.11%; batched 0; MoE +0.14% / +0.35% (simulated) | | none (timing only) | low | **rejected** | no: lowers Qwen3.5's DRAM efficiency |
| XDMA options (C2H IDs 8; the hidden resource reduction) | 73 LUT + 1 RAMB36; 0.6K LUT with the hidden option (measured) | 0 | | none | the hidden option is undocumented | **no change** | no |
| One XDMA side for both channels | ~3.5K LUT, ~1.8K FF (estimate) | 0 (estimate: H2C ~3 GB/s against one channel's 8.5) | moves the worst path (u_ch1's XDMA side, +0.013 ns) | none | medium: H2C ordering, WAITW's write counts | card qualification: H2C / C2H rates, selftest, the H2C sweep, WAITW | not now: a design item |
| LiteDRAM: one calibration engine for both channels | ~2.4K LUT, ~2K FF (estimate); ~7K LUT of calibration hardware in all | 0 | slack | none | medium-high: core regeneration, the DFI injector's tight paths, calibration | **not now; a future lever**. Card: calibration warm soak, qualification | not now |
| LiteDRAM: no calibration CPU | 1.0K LUT, 1.0K FF, 4 RAMB36 + 3 RAMB18 (measured) | 0 | | none | low | the host calibrates, 25 s a channel at every load | no |
| LiteDRAM: no ECC | ~4.2K LUT, 4.5K FF | 0 | | reliability: no error detection; the qualification reads the ECC counters | | the user's | no |
| Core on LiteDRAM's clock (no crossing) | ~0.5-1K LUT, a few cycles of read latency (estimate) | ~0 (decode is DRAM-bound) | | none | | **one-way**: the core clock is the DRAM's (no 100 MHz builds at DDR3-1066, no core above 133.33) | no |
| Quantizer at half rate | ~3.5K LUT, ~32 DSP (estimate) | prefill: QUANT 20-27% busy -> 40-54% | | none | | | no: the MX column sizes it |
| **IMEM 8K** (a cost: reinvestment) | -28.5 RAMB36 | long-context prefill: 2-3 rows a run instead of 1 in bucket 16 (Phi-4-mini, LFM2-2.6B) | fewer weight passes at long context | **arch**: `IMEM_WORDS` 65536 (the arch hash, fit checks, program caches) | the sequencer's fetch path (seq-vpu to check) | **parked** until the looped attention is costed | no |
| ACT_ROWS 16 (with the MX column) | 0 block RAM: fills the ACT RAM's unused 75% | (the MX column's) | MM replay over 16 rows | (the MX column's) | | | yes, with the MX column |
| Bridge credits doubled (ARD 128: the other direction) | costs ~0.7K LUT a channel (estimate) | -0.03 to -0.24% decode / MoE (simulated) | | none | | | **no**: the DRAM is the limit |
| *ASIC:* V^T append from an on-chip tail (section 10.2) | 28.7K beats read and 28.7K written a Qwen3-0.6B token (73% of its writes; estimated) | decode: QST busy 4.9% | | compiler: the KV append; numerics none | SRAM: tb x d bytes a KV head | the ASIC's | no |
| *ASIC:* DeltaNet state resident in the slice | 6.6% of Qwen3.5-0.8B decode's DRAM beats, 88% of its writes (simulated) | decode: DSTEP busy 14.8% | | numerics: fp32 kept, or bf16 (half the SRAM) | SRAM: 18 MiB / S a sequence for this model | the ASIC's | no |
| *ASIC:* one landing buffer per channel | two of three reservation stages per read (section 10.2) | 0 | | none | | the ASIC's | no |
| *ASIC:* PUT / PUTADD link engines (sections 10.3, 10.4) | S > 1: 1.75x fewer bytes into each slice, no hub, no barrier | S > 1 decode: Qwen3-0.6B's collectives ~175K cycles a token today, ~8K (S = 2) / ~14K (S = 4) proposed (estimated) | slices added without a central unit | **ISA**: PUT / PUTADD and WAITC; the compiler's sharding | | **approved as the direction**; no RTL until a multi-slice target is next | no: S > 1 only |

**For the card's qualification list** (none is proposed for the next build):
- a shared XDMA side: H2C / C2H rates, `selftest`, the H2C sweep, the WAITW tests;
- any XDMA option change: the same;
- a shared calibration engine: calibration on both channels, a warm soak, BIST, the ECC
  counters;
- `TRACE_DEPTH=0` itself needs nothing beyond the usual qualification: the trace is off in it.

**Reinvestment.** In the MCOLS 8 build the two levers free ~4.3K LUT, ~4.8K FF and 32 RAMB36:
- the trace buffer and the collective stub;
- about 1.4% of the device's LUTs, 2.4% of the design's;
- in slices, a point or two of the 76.6%.

What that buys:
- **Timing headroom** at 133.33 MHz for the MX column's build (WNS +0.013 ns today). The
  trace's capture registers sit next to the slice's units.
- **An 8K IMEM**, paid by the trace's block RAM, if long-context prefill matters before the
  looped attention.
- **Not MCOLS 8 or 128K TMEM.**
  - 128K TMEM needs ~384 more RAMB36 at TMEM_IMPL 0: the device has ~352 free, so it is
    tmem-pump's banked TMEM, not these 32.
  - MCOLS 8 needs the MXU's resources doubled, an order of magnitude above what this study
    frees; dsp-pump's MX column is the route to that rate.

## 10. Data movement: what moves more than once, and the link between slices

openTPU is the architecture of a future inference ASIC; the card is its sanity check. The rule
that follows from that: a byte should cross each memory level once, on its way to the unit that
uses it, and no SRAM should hold a copy of another SRAM's data. This section lists the platform
and DMA paths that break the rule, with what each costs on the card, and what the slice-to-slice
path should be when slices are NUMA regions added side by side. The sizes are for one decode
token unless named otherwise; **simulated** numbers are the co-sim's (section 2's runs).

### 10.1 What already moves once

- **Weights and the KV cache never pass through TMEM.** MM weights, K and V^T stream from DRAM
  through port B into the MXU's chunk FIFO; their block scales come through port A. In decode
  every weight byte is read once per token. At S > 1 each slice streams only its own rows from
  its own channels (docs/compiler.md, "Multi-slice programs").
- **The DMA reads each chunk of its line buffer once.** An LD's chunk goes out as one segment a
  cycle, an ST's chunk is gathered once and written once, and a stream's chunk is fed to SE once.
  The DMA's TMEM traffic is the activations and small per-token tensors (Qwen3-0.6B decode: 788
  LDs and 19 STs a token, busy 3.4% of its cycles, **simulated**).

### 10.2 What moves more than once

**Host to card (the platform):**

| path | what happens | cost on the card | what an ASIC's platform should do |
|---|---|---|---|
| the channel interleave | the accelerator sees one space beat-interleaved over its channels (CHASH), but XDMA addresses each channel by itself (bit 31). So the host CPU rearranges every byte it uploads into the two channels' runs (`Board.write`; `offload.BoardDram` with `np.take`) before the DMA | one host-memory copy of every image, expert and KV restore | the device's DMA writes through the same address map as the core, so the host DMAs from the source buffer |
| the staging bounce | a host buffer not placed for XDMA (d % 64 for writes, d % 4096 in [32, 4064] for reads) goes through a staging copy (`XdmaTransport._stage`) | a second host copy: H2C at 2.2-2.5 GB/s instead of 3.0 (docs/host.md) | a scatter-gather DMA engine without alignment penalties |
| sub-beat host writes | a write that is not whole 64-byte beats is widened with a read of the card first (ECC: the controller cannot take a partial beat) | a PCIe round trip per edge beat; `BoardDram` keeps shadows of its own words to avoid it | byte-masked writes in the memory controller |
| offloaded experts | an expert went from the pool through four host copies before `BoardDram` (544 MB/s against the link's 1.37 GB/s, docs/offload.md 10.1); now one pass into page-aligned runs, then card DRAM, then the MXU | the DRAM write and read are the expert cache, which pays when experts repeat | the same cache. What goes is the host copy (above) |

**On the card (the memory path and the DMA):**

1. **Every DRAM read is reserved and buffered two or three times.**
   - A weight chunk lands in three buffers on its way to the MXU: the bridge's read-data FIFO
     (`u_ar`, ARD 64), the adapter's B FIFO (`rbm`, RD 128), and the MXU's chunk FIFO (1K
     chunks, 128 KB of block RAM).
   - An LD chunk lands in the bridge's and the adapter's FIFOs, then in the DMA's chunk buffer
     (`u_lb`, 16 KB), then in TMEM.
   - Each stage reserves its room per read because read data cannot be refused. `+memstat`
     shows two of them full at once for the same reads: the credits at p50 61 of 64 and the B
     reservations at p50 86 of 128 (section 2). The bridge's FIFO also crosses the clock, since
     LiteDRAM runs from its own MMCM.
   - On an ASIC: one landing buffer per channel, in the controller's clock or behind a single
     crossing, its room sized to the bandwidth-delay product, with credits issued against the
     consumer's buffer. The data is then written once, into the MXU's chunk FIFO or TMEM.
     **Measured here:** about 86 beats of 64 B in flight per channel at 93-94% DRAM efficiency,
     about 5.5 KB at DDR3-1066. It scales with bandwidth x latency, not with the buffers'
     count.
2. **The transposed V append: a 64-byte read and a 64-byte write per byte stored.**
   - `kv_append` writes V^T one byte per dimension, each in its own beat (docs/isa.md QST:
     "each transposed byte is a separate DRAM write").
   - The adapter's SW queue fills each beat with a read, then writes it whole. That avoids the
     controller's read-modify-write but not the traffic.
   - Qwen3-0.6B: 8 KV heads x 28 layers x 128 bytes = 28,672 beats read and written a token
     (**estimated** from the layout). That is about 73% of the token's 39,064 written beats
     (**simulated**) and 0.85% of its DRAM beats, for 28 KB of V. QST is busy 4.9% of the
     token's cycles.
   - On an ASIC: keep the open V^T tile (its last tb tokens, tb x d bytes per KV head) on
     chip and write it whole once it fills; or append several tokens per write. Batched decode,
     MTP and prefill already write n bytes per beat.
3. **The DeltaNet state goes to DRAM and back every token.**
   - DSTEP reads each head's fp32 state (128 x 128 x 4 B) and writes it back.
   - Qwen3.5-0.8B: 18 layers x 16 heads x 64 KiB = 18 MiB each way, 294,912 beats read and as
     many written a token (**estimated** from the shapes, against the co-sim's totals). That
     is 6.6% of the token's DRAM beats and 88% of its 334,360 written beats (**simulated**).
     DSTEP is busy 14.8% of the token.
   - The state does not grow with the context, and at S > 1 its heads split over the slices.
     That makes it the first candidate to stay resident in a slice's SRAM: 18 MiB / S per
     sequence for this model, or half that in bf16 (a numerics decision).
4. **WAITW polls DRAM.**
   - The DMA re-reads its chunk every w5 cycles until the host's word arrives.
   - That is DRAM bandwidth and a poll interval's latency spent on a flag the host itself wrote.
   - On an ASIC: a doorbell, either a host write to a register or the write path noticing the
     address.
5. **The device's embedding gather streams the row D / M times.**
   - `kernels/gather.py` extracts a table row through the MXU against one-hot operands: 32 MMs
     at MCOLS 4 for an int8 row.
   - Qwen3-0.6B, when the decode loop gathers its embedding on the card: a 1 KB row read 32
     times, 32 KB a token. Negligible in bytes, but a DMA gather with the quantizer's
     dequantization would move it once.
6. **Prefill reads every weight once per run, not once per prompt.**
   - A run holds R = 3-4 rows today, so a 1,000-token prompt streams each weight about 250
     times.
   - Runs of 8-32 rows (MM replay) cut that weight traffic 2.8x at about the same cycles a row
     (docs/tiled_prefill.md 0 and 4.1). On the card prefill is MXU-bound, so the bytes do not
     show as cycles. On an ASIC they are energy, and the tile size should follow SRAM
     capacity, not the cycle count.

**Inside the slice**, for the data-movement ledger (other studies own these units):
- Every MM's input is staged twice: TMEM fp32, then the quantizer, then ACT RAM int8.
- `VOP COPY` reshapes TMEM tiles: 672 copies a Qwen3-0.6B token, 24 a layer.
- TMEM_IMPL 0 writes every word into one block-RAM copy per read-port group (6 on the card).
  That is the largest SRAM replication; tmem-pump's banked TMEM (IMPL 1) is the alternative.

### 10.3 The slice-to-slice path at S > 1: the cross-NUMA link

**What it is today** (`otpu_coll`, and `otpu_coll_hub` / `otpu_coll_port` on the BCU1525):
- **One central unit, one collective: GATHER.** It is an all-gather that writes every slice's
  segment into every slice's TMEM. `all_reduce` is a GATHER followed by S - 1 VOP adds, so it
  moves S times the data.
- **Serial.** It reads one slice at a time, LANES words a cycle at S = 1. Through the hub it
  moves one beat per ~7-cycle round trip (docs/bcu1525.md). Its time is the total size, so it
  does not shrink as slices are added.
- **Blocking.** Every GATHER is a barrier, and it holds every slice's TMEM grant (the grant is
  the AND of the slices').
- **It copies SRAM.** The source and destination must not overlap, so a slice copies even its
  own segment. At S = 1, GATHER is nothing but a TMEM copy, which is why the stub (section 5)
  drops it.

**What the kernels move with it** (**estimated** from `qwen3.py` and `kernels/mlp.py`):
- Qwen3-0.6B gathers 7,168 words per layer at S = 2:
  - the attention outputs, 2,048;
  - the output projection, 1,024;
  - the SwiGLU activations, 3,072, chunked;
  - the residual, 1,024.
- That is 784 KB a token, written into every slice. At one beat per round trip it is about
  175K cycles a token, ~10% of an S = 2 token's weight stream, plus a barrier per GATHER.
- The scheme gathers the wide activations and keeps every sum on one slice, which keeps the
  results bit-exact without a cross-slice reduction.

**What the link should be:**
- **Move only activations, at the fewest points.**
  - Weights and KV already stay with their owner.
  - Tensor parallel needs two reductions of H per layer: after the output projection and after
    the down projection, both sharded by input. Today's scheme moves 7H at Qwen3-0.6B's shapes.
  - With a reduction on the link, each slice receives 4H(S-1)/S words a layer on a ring,
    against 7H(S-1)/S today: 1.75x fewer. More important, no hub serializes them (10.4).
  - A reduction in a fixed order (ring order) stays bit-exact against the ISA simulator.
- **Point to point, one engine per slice, no hub.**
  - Each slice gets a link engine: a DMA queue whose PUT sends a local TMEM, ACT RAM or DRAM
    range to a neighbour's address, optionally adding into it on arrival.
  - Credit flow control.
  - Completion as a counter at the receiver, which its sequencer waits on, as WAITW does, but on
    an on-chip counter, not a DRAM poll.
  - Bandwidth then grows with every slice added (a ring at small S, a 2D mesh or torus beyond),
    and nothing is global: no TMEM grant AND, no barrier.
- **Land once, where the consumer reads.**
  - A received activation goes straight into the TMEM tile or ACT RAM rows of the next
    instruction.
  - A slice's own segment stays in place, with the remote segments laid around it.
  - Nothing is staged and copied, and nothing is replicated to a slice that does not read it.
- **Overlap.**
  - The link engine runs beside the weight stream.
  - The compiler issues a PUT as soon as a chunk is ready, as the MLP's chunked gather already
    pipelines.
  - It waits on the counter only where the data is used.
- **MoE needs all-to-all, not all-gather.** Expert parallelism sends each token's hidden vector
  to the slices that own its experts and combines the results back. Destinations are known only
  at run time, so a PUT takes its slice and address from registers.
- **Scale by groups.**
  - Tensor parallelism within a small group of neighbouring slices.
  - Pipeline or expert parallelism across groups, which crosses only R x H per group boundary.
  - The link's demand then stays local as slices are added.

**Sizing** (first principles, dense layers of ~12 H^2 MACs a token, b bytes an activation, M
MACs/s a slice):
- Prefill (compute-bound, R rows): the two reductions take 4 R H b (S - 1)/S bytes per layer
  per slice. The layer computes in 12 R H^2 / (S M) seconds. To hide the transfer a slice needs
  about S b M / (3 H) of link bandwidth. That is independent of R.
  - The card's slice (68 GMAC/s, H = 1,024, bf16): ~44 MB/s x S. Trivial.
  - A frontier slice (10^15 MAC/s, H = 8,192, S = 8): ~650 GB/s. That is comparable to its HBM,
    so the link is a first-class resource there.
- Decode (latency-bound, ~2H b a layer): what matters is hops x hop latency, twice a layer,
  against the layer's weight-stream time per slice. Few hops (a group of neighbours, a tree)
  matter more than width.

**What it needs:**
- ISA: PUT / PUTADD and a counter wait, and a CAPS2 bit each.
- Compiler: `all_gather` / `all_reduce` lowered onto them, and W_o and W_down sharded by input.
- Simulator: the ISA simulator's fixed reduction order.
- Tests: the board model at S = 2 and 4.

The S = 1 card needs none of it: with the stub it keeps BAR alone.

### 10.4 Proposal: PUT / PUTADD link engines

The lead approved this on 2026-10-12 as the direction for the cross-slice path. No RTL until a
multi-slice target is next. The S = 1 card keeps BAR alone (section 5).

**The engine.** One per slice, beside its DMA.
- It owns the slice's links to its neighbours: two at S = 2, a ring or a full mesh at S = 4, a
  2D mesh beyond.
- A queue of PUTs, issued by the sequencer like any unit instruction.
- A landing path at the receiver: the collective's TMEM write port (W_COLL), which multi-slice
  builds keep. For PUTADD, also the collective's read port and an fp32 adder.
- 16 completion counters, 32 bits each.

**The instructions** (eight words, as every instruction):

| instruction | fields | what it does |
|---|---|---|
| `PUT` | `src = R[ra]+w1` (local), `dst = R[rb]+w2` (the receiver's), `rows = w3[15:0]`, `cols = w3[31:16]`, `srs = w4`, `drs = w5`, `w6[7:0]` the destination slice (flag `REL`: `(sid + w6[7:0]) mod S`; or from a register, for MoE), `w6[15:8]` the receiver's counter, `w6[17:16]` the source space (TMEM words, ACT RAM rows, DRAM bytes) | sends rows x cols words; at the receiver they land at `dst + r*drs + c`, then the counter goes up by rows x cols. Locally it completes when the source has been read, so the source may be reused |
| `PUTADD` | `PUT` with flag `ADD` | at the receiver, `T[dst...] += data` in fp32, in arrival order. One sender per destination range at a time, so the order is the schedule's and the ISA simulator reproduces it bit for bit |
| `WAITC` | `w1` the counter, `w2 + R[ra]` the target, `w5` / `w6` as WAITW's poll and timeout | the slice waits until the counter reaches the target, then subtracts it (so a loop reuses the counter); past the timeout, ERROR |

- Data from one sender to one receiver arrives in order. Each link has credit flow control.
- A remote write is not seen by the receiver's scoreboard; WAITC is the synchronization. The
  compiler gives each collective its landing slots and counter.
- BAR becomes a zero-length PUT to the neighbours plus a WAITC.
- The register map announces the engine in a CAPS2 bit, next to bit0 GATHER.

**How it replaces the hub's GATHER** (the compiler: `language.all_gather` / `all_reduce`, and
the kernels' sharding):
- **all_gather** of a segment of n words over a ring:
  - S - 1 steps. At step k each slice PUTs the segment it received at step k - 1 (its own at
    k = 0) to its right neighbour's slot for it, then WAITCs (k + 1) n.
  - A slice's own segment is produced in its slot, so it is never copied.
  - At S <= 4 a full mesh sends every segment directly, in one step.
- **all_reduce** = reduce-scatter (S - 1 PUTADD steps of n / S words) plus all-gather (S - 1 PUT
  steps of n / S words). Each slice sends and receives 2 (S - 1) n / S words.
- **The kernels:**
  - W_o and W_down are sharded by their input dimension, so attention and the MLP each end in
    one all_reduce of H.
  - The attention-output and SwiGLU gathers go: the heads and the F chunks stay on their
    slice.
  - The residual stays replicated. Every slice adds the reduced y itself.

**Estimated bytes and time** (decode, one token; fp32 activations; a link of 8 words a cycle
each way, 10 cycles a hop, both assumed. Today's figure is the hub's one 8-word beat per ~7-cycle
round trip, which does not change with S):

| model | S | words into each slice a layer, today (7H-class gathers) | proposed (two all-reduces) | collective cycles a token, today | proposed | of the token's weight stream at that S |
|---|---|---|---|---|---|---|
| Qwen3-0.6B (H 1,024, 28 layers) | 2 | 3,584 | 2,048 | ~175K | ~8K | ~10% -> ~0.5% |
| | 4 | 5,376 | 3,072 | ~175K | ~14K | ~19% -> ~1.5% |
| Phi-4-mini (H 3,072, 32 layers) | 2 | 8,704 | 6,144 | ~487K | ~26K | ~5% -> ~0.3% |
| | 4 | 13,056 | 9,216 | ~487K | ~41K | ~9% -> ~0.8% |

- **Today's words a layer:**
  - Qwen3-0.6B gathers 7,168: o 2,048, W_o's output 1,024, the SwiGLU activations 3,072, the
    residual 1,024.
  - Phi-4-mini gathers 17,408: 3,072, 3,072, 8,192 and 3,072.
  - Each slice receives (S - 1) / S of them. The hub reads all of them in series.
- **Proposed:** a ring all-reduce takes 2 (S - 1) x (H / (8 S) + 10) cycles, two of them a
  layer.
- **The token's weight stream at S** is the S = 1 token over S: Qwen3-0.6B 3.62M cycles at
  position 544, Phi-4-mini ~20.8M at 2K context (`tools/scoreboard.py`).
- **The proposed cycles also overlap:** a PUT goes out while the next weights stream, and only
  the WAITC before the consumer waits.
- **Barriers:** today's ~11 GATHERs a layer (o, y, ~8 MLP chunks, the residual) are each a
  barrier with every slice's TMEM granted. The proposal has two counter waits a layer.
- **Prefill and batched decode** multiply the words by the rows. On the card the link stays far
  below what hides under compute (10.3's sizing: ~44 MB/s x S for the card's slice).

### 10.5 The V^T append in time, and its layouts

**Per decode token** (one fill read and one whole-beat write per appended byte, L x Hkv x d
partial beats; the two channels share them evenly, so the DRAM time is at least one channel
cycle per partial beat):

| model | partial beats a token | DRAM bytes moved a token | DRAM cycles a token, lower bound | DRAM cycles a token, at 2K / 4K | token at 2K context (cycles) | share at 2K | token at 4K | share at 4K |
|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B (28 x 8 x 128) | 28,672 | 3.67 MB (28 KB stored) | 28,672 | **59.9K / 56.2K** (49.0K at position 544) | 4.37M | **1.37%** | 5.37M | **1.05%** |
| SmolLM3-3B (36 x 4 x 128) | 18,432 | 2.36 MB | 18,432 | ~37K | 15.5M | ~0.24% | 16.1M | ~0.23% |
| Phi-4-mini (32 x 8 x 128) | 32,768 | 4.19 MB | 32,768 | ~66K | 20.8M | ~0.32% | 21.9M | ~0.30% |

- **Qwen3-0.6B is measured** (**simulated**). It is the co-sim at positions 544, 2,048 and 4,096
  against the same run with every V^T append dropped: `--hack novt`, `tools/vpu_hack.py` on
  `infra-hunt2`. Dropping them removes 224 QSTs a token:
  - written beats fall from 39,064 to 10,168 a token (at 2K);
  - read beats fall by the same 28,896;
  - DRAM efficiency rises from 92.9% to 94.2% at 2K.
- **A partial beat costs 1.7-2.1 cycles, not 1.** The scattered beats open DDR3 rows (8.1-9.4K
  more row opens a channel a token) and turn the bus around.
- **SmolLM3-3B and Phi-4-mini are estimated:** 2.0 cycles a partial beat (with the scale
  beats), against `tools/scoreboard.py`'s token times.
- **dataflow's ISA-simulator ledger agrees on the bytes:** 1.86 MB read and 1.89 MB written a
  Qwen3-0.6B token at 544.
- **In time it is small.** The append is 73% of the written beats but 1.0-1.4% of a Qwen3-0.6B
  token, and 0.2-0.3% of the 3-4B models': they stream far more weight bytes per KV head.
- **Batched decode multiplies it by the users while the weights are shared.** Sixteen Qwen3-0.6B
  users at 2K append ~0.9M cycles a step, ~2.3% of the scoreboard's step (16 tokens at 54.2
  tok/s).

**The layouts, costed in bytes** (one KV head, one layer, one appended token; d = 128 int8 and
64-byte beats; today: 16 KB of DRAM moved, 8 KB read and 8 KB written, to store 128 bytes).

- **(a) V row-major, which the MXU consumes for P.V.**
  - The append writes 128 bytes, two whole beats, plus its scale.
  - But the MXU streams the operand whose rows are P.V's dot products, V^T's rows. Row-major V
    needs either a D x D transpose stage in front of the MXU, or a new MXU mode that
    accumulates P[g, t] x V[t, :] (an outer-product dataflow, the MXU study's).
  - A transpose stage moves every V byte of the context through SRAM twice per decode step:
    2 T d bytes a head. At 4K context that is 1 MB a head-layer, 235 MB a Qwen3-0.6B token of
    on-chip traffic, to save 3.67 MB of DRAM traffic. It moves the cost from the write, paid
    once, to the reads, paid T times.
  - Only the native MXU mode makes (a) pay.
- **(b) Blocked V^T, written in whole beats every N tokens** (N = 64 at int8, 32 at bf16: a
  beat holds N tokens of one dimension). DRAM traffic falls to 128 bytes a token, amortized,
  and no reads. The open tile is N x d = 8 KB a KV head a layer: per sequence L x Hkv x 8 KB,
  which is 1.84 MB (Qwen3-0.6B), 1.18 MB (SmolLM3-3B) and 2.10 MB (Phi-4-mini). Where it lives:
  - **On chip**, in the slice's SRAM: a TMEM-class buffer of 1-2 MB per sequence (the card's
    TMEM is 256 KB: it does not fit), times the users in batched decode. P.V must read the
    open tile's tokens from it, so the MXU needs a second streamed source, SRAM beside DRAM.
  - **In DRAM, token-major** (a V tail ring of N tokens: two whole beats an append). Every N
    tokens the DMA transposes the tail into the V^T tile through a D x D transpose buffer
    (16 KB of SRAM).
    - DRAM traffic: 128 + 128 + 128 = 384 bytes a token a head, 42x less than today.
    - P.V reads the open tail's < N tokens row-major, so they go through the transpose buffer
      too: N x d = 8 KB a head a token, 1.8 MB of on-chip traffic a Qwen3-0.6B token.
- **(c) A write-combining buffer in the DMA** (or the adapter's SW gather, which merges within
  one QST today).
  - It holds each open V^T beat until its 64 bytes fill: (b)'s on-chip tile, organized by
    beats.
  - It is all or nothing. With fewer than L x Hkv x d open beats (28,672 for Qwen3-0.6B, 1.84
    MB), each token evicts beats before they fill, which is today's traffic again.
  - P.V's port B reads of V^T must see the bytes still in it: a snoop on the read path, or
    the attention reading the open tile from it.
  - DRAM traffic: 128 bytes a token a head.
- **Smaller levers:**
  - byte-masked writes (DM pins, no ECC read-modify-write) drop the fill read but keep the
    64-byte burst: 8 KB a head-token, 2x better;
  - appending k tokens a step (MTP, speculative decode, prefill) writes k bytes a beat: k x
    fewer.

### 10.6 The opportunities, ranked by bytes

Bytes are a decode token's unless named otherwise (**estimated**: shapes and `tools/scoreboard.py`
at 2K context, fp4 weights with an int8 head). None of these is being worked on now; the first
two are ASIC design rules more than card changes.

| rank | opportunity | bytes moved that need not be | what fixes it | who |
|---|---|---|---|---|
| 1 | prefill re-reads every weight once per run (R rows) | weight bytes per prompt token = layer weights / R: Qwen3-0.6B ~254 MB / 8 = 32 MB, SmolLM3-3B ~1.5 GB / 4 = 378 MB, Phi-4-mini ~1.75 GB / 2 = 874 MB | tiles of more rows (docs/tiled_prefill.md: 2.8x less at 8-32 rows); on an ASIC, as many rows as SRAM holds | tiled prefill (merged, opt-in) |
| 2 | the DRAM read path's staging: two buffers (`u_ar`, `rbm`) before the consumer's own | two extra on-chip writes and reads of every byte read: Qwen3-0.6B ~1.06 GB a token (531 MB read), Phi-4-mini ~5.0 GB (2.5 GB read) | one landing buffer per channel sized to bandwidth x latency (~86 beats per channel here), credits against the consumer's buffer (10.2 item 1) | the ASIC's memory system |
| 3 | the host's interleave copy | one host-memory copy of every uploaded byte. Offloaded MoE: Qwen3.5-35B-A3B ~104 misses x 1.67 MB = ~174 MB a token, Gemma 4 26B-A4B ~87 x 3.44 MB = ~299 MB. Plus every image load once | the device's DMA writes through the core's address map (the interleave in hardware), so the host DMAs from the pool's pages | the ASIC's host interface |
| 4 | the DeltaNet state's round trip | Qwen3.5-0.8B 37.7 MB a token (18 MiB each way); 6.6% of its DRAM beats | the state resident in the slice's SRAM (18 MiB / S a sequence; bf16 halves it) | dataflow / the ASIC's SRAM budget |
| 5 | the V^T append | Qwen3-0.6B 3.67 MB a token, SmolLM3-3B 2.36 MB, Phi-4-mini 4.19 MB, for 1/128 of it stored; in time 1.0-1.4% of a Qwen3-0.6B token (measured), 0.2-0.3% of the 3-4B models' | 10.5: a blocked V^T with its open tile on chip, or a V tail in DRAM | dataflow |
| 6 | the slice-to-slice gathers at S > 1 | Qwen3-0.6B 7,168 words a layer through one hub, written into every slice | PUT / PUTADD (10.4) | approved direction, no RTL |
| 7 | WAITW's DRAM polls; the embedding gather's 32 re-reads | KB a token | a doorbell; a DMA gather | |

## 11. Tools

The build knobs `TRACE_DEPTH` and `COLL_GATHER` (sections 4 and 5) and `otpu_mem_ch`'s tag FIFO
sized from ARD (section 2) are merged with this document. The study's instrumentation and
scripts below stay on branch `infra-hunt2` (kept, not merged), to be merged when a study needs
them. Everything is opt-in: the board's defaults are unchanged, and no build changes unless a
parameter is set.

- **`+memstat=FILE`** (simulation only: `otpu_native_dram`, `otpu_mem_ch`). It appends
  per-cycle histograms (`NATSTAT` / `CHSTAT` lines) to FILE at the end of the run:
  - the adapter's reads in flight (its tags);
  - the B and A read FIFOs' reservations and fills;
  - the request queues (B, A, SW) and the write-gather queue;
  - the bridge's read credits, command FIFO, per-port output command queues, write-data FIFOs
    and read tags.

  Next to them it records counters: the cycles a depth held a request back, and the command
  slots that cost (a cycle with a request, no room, and nothing else issued).
- **Depths as parameters**, through `otpu_top` / `tb_top` / `otpu_ldc_mem`, and set in the
  co-sim with `OTPU_MEMPATH=NAT_RD=64,CH_ARD=32,...` (`opentpu/rtlsim.py`; only `NAT_`, `CH_`
  and `DMA_` keys). They cover:
  - the adapter's `NAT_QD`, `NAT_WQD`, `NAT_RD`, `NAT_AD`, `NAT_APF`;
  - the bridge's `CH_ARD`, `CH_OQ`, `CH_OD` and `CH_AQD`. `CH_AQD` is the depth of
    `otpu_mem_ch`'s new `AQD` parameter (its command and write-data clock-crossing FIFOs,
    default 32);
  - the DMA's stream run `DMA_RUN` (`otpu_dma` `RUN`, default 64).

  `OTPU_SIM_PLUSARGS` passes more simulator arguments.
- **`rtl/exp/infra_hunt/`**:
  - `run_cosim.sh`: workloads x candidates, each with `+memstat`. The lists are `lists/all.txt`
    (the four workloads), `lists/ds.txt` (the stream workloads) and `cands.txt`;
  - `memstat.py`: cycles against the first candidate, and the histograms' max / p99 / p99.9;
  - `run_ooc.sh` / `ooc.tcl`: the units out of context;
  - `breakdown.tcl`: a build checkpoint's per-unit reports and leaf cells;
  - `ld_ooc.tcl`: the LiteDRAM core alone;
  - `run_xdma.sh` / `xdma_ooc.tcl`: the XDMA IP at the card's configuration and at variants.
  - `tmem_wc.sv`: `otpu_tmem` as the slice builds it, with the collective's write port tied off
    or not (section 5).
- **`TRACE_DEPTH`** and **`COLL_GATHER`** for `boards/ypcb-00338/run_vivado.sh` (sections 4
  and 5; merged).
