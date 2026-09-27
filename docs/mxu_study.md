# MXU study: width (MCOLS) and dot-product implementation for the board

Question: which MXU should the YPCB-00338 build (xc7k480t, XDMA + 2 x MIG, 100 MHz) use, given
that prefill and batched decode are weight-stream-shared across the MXU's columns while
single-stream decode is DRAM-bound?

## What the MXU is

A broadcast-weight engine, not a systolic array: every cycle one D = 128-byte int8 weight chunk
is broadcast to MCOLS columns, each holding one stationary activation row (ACT RAM); each column
reduces the 128 int8 x int8 products exactly (adder tree, 4 register levels), converts to fp32 and
scales by the weight and activation block scales. MCOLS is the number of activation rows (decode
sequences or prefill tokens) that share one pass over the weights.

`otpu_mxu` now has `IMPL` (0 = adder tree, default; 1 = DSP cascade) and `CL` (cascade chain
length), plumbed as `MXU_IMPL` / `MXU_CL` through slice, top, board and testbench, and selectable in
simulation with `OTPU_MXU=cascade [OTPU_MXU_CL=16]`. MCOLS may now exceed LANES (the ASCALE alpha
load and the RMAX write-back loop over LANES lanes at a time; the drain already took several
cycles per row).

### Cascade (systolic-style) variant, IMPL=1

The 128 positions form D/CL chains of CL (default 16). Position k of a chain sees its weight and
activation bytes delayed k cycles (SRL shift registers; the weight skew is shared by all columns),
so a new chunk enters every cycle and each stage is exactly DSP48E1's `M` register followed by
`P = PCIN + M`: one DSP per product with the running sum on the dedicated cascade, no fabric
adders. The D/CL chain outputs (8 at CL=16) go through a 2-level registered adder tree. Latency is
CL + 1 + tree levels (19 cycles at CL=16) against 4 for the tree; the activation-scale and the
weight-scale delays follow it (`LDOT`). Results are bit-identical: the sums are exact integers.

Packing two int8 x int8 products per DSP48E1 (the "INT8 packing" trick) cannot be kept exact on a
cascade: the 25-bit A port leaves an 18-bit field for the low product, whose signed sum overflows
into the high product after at most 3 accumulations (4 if -128 never occurs). Correction logic per
short chain costs more fabric than the DSPs it saves, so the variant uses one product per DSP.

## Area and logic delay (yosys synth_xilinx, MXU alone, board parameters: D=128, LANES=8, FIFO 1024)

| MCOLS | impl | LUT | FF | DSP | BRAM36 (MXU) | ACT RAM BRAM36 | logic delay | est. fmax |
|---|---|---|---|---|---|---|---|---|
| 2 | tree | 14.3K | 10.3K | 270 | 1 + 57 BRAM18 | 33 | 6.19 ns | ~96 MHz |
| 2 | cascade | 15.7K | 15.3K | 270 | 1 + 57 BRAM18 | 33 | 6.19 ns | ~96 MHz |
| 4 | tree | 29.3K | 20.3K | 539 | 1 + 57 BRAM18 | 65 | 6.19 ns (7.65 before `rx`) | ~96 MHz |
| 4 | cascade | 31.9K | 30.0K | 539 | 1 + 57 BRAM18 | 65 | 7.65 ns (before `rx`) | ~78 MHz |
| 8 | tree | 59.6K | 39.5K | 1078 | 1 + 57 BRAM18 | 129 | 8.20 ns (10.43 before `rx`) | ~72 MHz |
| 8 | cascade | 65.7K | 58.6K | 1078 | 1 + 57 BRAM18 | 129 | 10.63 ns (before `rx`) | ~57 MHz |
| 16 | tree | ~120K (2 x MCOLS 8) | ~79K | ~2150 | 1 + 57 BRAM18 | 257 | not completed (yosys stalled) | - |

- DSP = ~134 per column (128 products + the fp32 scale multipliers): yosys already maps each tree product onto
  a DSP48E1 multiplier; the tree's adders are in fabric (~2.4K LUT per column).
- yosys does not absorb the cascade's `P = PCIN + M` into the DSP (it keeps a fabric adder and
  maps the skew to SRLs), so its cascade LUT count is not what Vivado will produce. Vivado infers
  this pattern as a PCIN cascade; the expected cascade MXU is the tree's LUT count minus the adder
  tree (~2.4K LUT/column) plus the activation skew (~1K SRL/column): ~1.4K LUT/column less, and
  the dot product leaves the fabric timing entirely.
- The logic delay grew with MCOLS (7.6 ns at 4, 10.4 ns at 8) because of one control path: the
  drain's bank-conflict lane selection (serial over min(MCOLS, LANES) lanes) feeding the RMAX
  compare and a dynamically indexed write. This study registers the RMAX compare one cycle after
  the drain (`rx`, bit-exact, `c_drained` waits for it): MCOLS 4 drops to 6.19 ns (the common fp
  datapath, as at MCOLS 2). At MCOLS 8 one 8.2 ns path remains, the 8-lane selection itself
  feeding `dj`; all other endpoints are under 6.6 ns.
- est. fmax = 1 / (1.6 x logic + 0.5 ns) (routing allowance used by the synth workstream).
- ACT RAM (outside the MXU) is MCOLS x 1024 bits per read, 128 blocks deep: 16 x MCOLS BRAM36
  plus one (33 at MCOLS 2, matching the synthesized `otpu_actram`).

### Whole-board estimate

Current slice per unit at MCOLS 2 (synth workstream, same flow): VPU 52.9K LUT, seq 21.1K,
TMEM 17.6K, quant 16.0K, MXU 14.3K, ACT 5.2K, DMA/AXI/coll/ctrl 6.8K: ~134K LUT, 416 DSP,
~603 BRAM36 (TMEM 512). XDMA + 2 x MIG add roughly 45K LUT and ~40 BRAM36. xc7k480t: 298.6K LUT,
1920 DSP, 955 BRAM36.

| MCOLS | impl | board LUT | board DSP | board BRAM36 | fits? |
|---|---|---|---|---|---|
| 2 | tree | ~179K (60%) | 416 (22%) | ~643 (67%) | yes (today's build) |
| 4 | tree | ~199K (67%) | 685 (36%) | ~675 (71%) | yes |
| 8 | tree | ~240K (80%) | 1224 (64%) | ~739 (77%) | tight on LUT/routing |
| 8 | cascade (Vivado) | ~221K (74%) | 1224 (64%) | ~739 (77%) | yes, if Vivado absorbs the cascade |
| 16 | any | ~320K (107%) | ~2300 (120%) | ~867 (91%) | no: DSP and LUT |

## Performance (RTL-measured MLP layer, projected to Qwen3-0.6B)

`tools/mxu_bench.py` runs one Qwen3-0.6B-sized MLP layer (H=1024, F=3072, 9.4 MB int8) on the RTL
at the board configuration (AXI DRAM model, 100% bandwidth, latency 30) with M = 1..16 rows.
Cycles per weight pass (a pass serves up to MCOLS rows):

| MCOLS | M=1 | M=2 | M=4 | M=8 | M=16 | roofline (M=1) |
|---|---|---|---|---|---|---|
| 2 | 80,516 | 80,900 | 80,770 | 80,806 | 80,851 | 74,112 |
| 4 | 80,516 | 80,900 | 81,861 | 81,892 | 81,907 | 74,112 |
| 8 | 80,516 | 80,900 | 81,861 | 84,517 | 84,246 | 74,112 |
| 16 | 80,516 | 80,900 | 81,861 | 84,517 | 118,647 | 74,112 |

A pass costs ~8% over the weight roofline up to 8 rows. At 16 rows the pass is 47% longer: the
per-row vector work (2 composite VPU lanes' exp2/recip for SiLU, quantization) no longer hides
under the weight stream. The cascade adds its 15 extra latency cycles once per MM chain
(80,533 vs 80,516 cycles at M=1; 84,531 vs 84,517 at M=8): negligible.

Full model: `tools/bench_llm.py --mcols M --bw 80 --ctx 512 --prompts 512` (batched decode and
8-row chunked prefill, qwen3_rows; RTL, 80% DRAM bandwidth, 100 MHz, context 512). MCOLS 16 cannot
be measured there (prefill chunks are at most 8 rows); its row is `tools/mxu_bench.py`'s
projection scaled to the measured MCOLS 8 result. The projection model (weight passes + per-sequence
KV, LM head once per prompt) is within ~10% of the measurement at MCOLS 2 but optimistic at 8
(prefill 156 vs 109, b=8 105 vs 77): attention and the per-row vector work grow with the rows.

| MCOLS | impl | LUT (board) | DSP | BRAM36 | est. fmax | prefill@512 tok/s | decode b1 | b2 | b4 | b8 | TTFT@512 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | tree | ~179K | 416 | ~643 | ~96 MHz | **38.9** | 15.5 | 28.1 | 28.2 | 28.3 | **13.2 s** |
| 4 | tree | ~199K | 685 | ~675 | ~96 MHz | **68.4** | 15.5 | 28.1 | 48.9 | 49.1 | **7.5 s** |
| 8 | tree | ~240K | 1224 | ~739 | ~72 MHz | **109.1** | 15.5 | 28.1 | 48.9 | 76.7 | **4.7 s** |
| 16 | tree | ~320K | ~2300 | ~867 | - | ~157 (proj.) | 15.5 | 28.1 | 48.9 | 76.7 | ~3.3 s (proj.) |

Decode at b=1 is the same for every width (DRAM-bound). b=2..8 scale up to min(b, MCOLS)
because the rows share the weight stream; prefill scales with MCOLS (1.76x at 4, 2.8x at 8).
The cascade implementation has the same cycle counts (see above).

## Recommendation

**Build MCOLS = 4 with the adder tree (IMPL=0) for the board.** It fits the xc7k480t with XDMA +
2 x MIG with margin (~67% LUT, 36% DSP, ~71% BRAM36), leaves single-stream decode unchanged
(15.5 tok/s at 80% bandwidth is DRAM-bound at every width), and gives 1.76x prefill (68 vs
39 tok/s), 1.74x decode at b=4 (49 vs 28 tok/s) and TTFT@512 7.5 s instead of 13.2 s. With the RMAX
compare registered its MXU logic delay is 6.19 ns, the same as MCOLS 2 and the other slice units
(VPU 6.06, quantizer 6.32 ns).

- **MCOLS = 8** is the stretch goal: 2.8x prefill (109 tok/s), b=8 decode 77 tok/s, TTFT@512
  4.7 s, still fits (1224 DSP), but ~80% LUT and one remaining 8.2 ns control path: the drain's
  serial bank-conflict lane selection over 8 lanes feeding `dj`. Pipeline that selection (compute
  the next cycle's lane group a cycle ahead) and see the MCOLS 4 build's utilization / timing in
  Vivado before moving to it.
- **MCOLS = 16 does not fit**: ~2150 DSP for the MXU alone (1920 on the part) and ~107% LUT, and
  at 16 rows the per-row vector work already costs 47% more per pass.
- **Cascade (IMPL=1)**: bit-exact and cycle-neutral, but only worth it if LUTs become the limit
  (MCOLS 8 under Vivado, est. -19K LUT). With yosys it is larger (no PCIN absorption). Keep the
  tree as the default; the cascade stays behind `MXU_IMPL=1` for a Vivado trial at MCOLS 8.
- Two products per DSP48E1 is not usable exactly on a cascade (see above), so the DSP ceiling is
  one product per DSP: MCOLS <= 14 on this part, before the rest of the slice's 146 DSPs.

## The core-side ceiling at DDR3-1066: port width, clock, VPU (2026-09-27)

Question: with 4-bit full-rate weights (MM PAIR), what keeps decode from running at the speed of
the DRAM, and which change lifts it? Targets: 80% DRAM efficiency, LFM2.5-230M at 88+ tok/s and
Qwen3.5-0.8B at 24+ tok/s.

**The ceilings.** DDR3-1066 on two x64 channels peaks at 17.07 GB/s. The slice takes at most one
128-byte chunk per core cycle (two 512-bit AXI beats, one per channel, both at core_clk; the
smartconnect crosses to the MIG's 133 MHz ui_clk). So the core port carries 128 B x f: 12.8 GB/s
(75% of the DRAM peak) at 100 MHz, 14.8 GB/s (87%) at 116 MHz, and matches the DRAM only at
133 MHz. At 100 MHz, 80% DRAM efficiency is out of reach whatever else is done.

**What the DRAM gives a streaming phase.** With the core as fast as the controller (the model's
`+axi_tpc=1 +axi_tpu=1`), the weight-streaming phases reach 88-92% of the DRAM peak (LM head
92.0%, MLP 88.7-90.4%, conv 88.4%): refresh, row switches and the port-A scale reads cost the
rest. That is the most a wider or faster core port can buy.

### Measured on the RTL (simulated)

`tools/perf_qwen.py --layers 0 --pos 128 --dram rbc --lat 38 --arc 4` with the DDR3-1066 timings
(`+axi_trp=3 +axi_trcd=3 +axi_tras=5 +axi_trc=7 +axi_trfc=22 +axi_trefi=1040 +axi_trmw=29`),
`OTPU_PAIR=1`, fp4 layers; the core clock enters as the ratio `+axi_tpc/+axi_tpu` (4/3 = 100 MHz,
23/20 = 116 MHz, 1/1 = 133 MHz). tok/s = cycles at that clock, no host time. DRAM efficiency =
useful bytes per token / (token time x 17.07 GB/s). L16: `OTPU_LANES=16 OTPU_ULANES=8
OTPU_VPU_CL=2`. Branch `port` at 152f9d4 (fp4-rebase with main's ddr-attn).

| model, weights / LM head | core | Mcycles/token | tok/s | DRAM eff. | core port busy |
|---|---|---:|---:|---:|---:|
| LFM2, fp4 / int8 | 100 MHz | 1.480 | 67.5 | 62.4% | 83.3% |
| LFM2, fp4 / int8 | 116 MHz | 1.491 | 77.8 | 71.9% | 82.6% |
| LFM2, fp4 / int8 | 133 MHz | 1.610 | 82.8 | 76.6% | 76.6% |
| LFM2, fp4 / fp4 | 100 MHz | 1.218 | 82.1 | 59.8% | 79.7% |
| LFM2, fp4 / fp4 | 116 MHz | 1.231 | **94.2** | 68.6% | 78.8% |
| LFM2, fp4 / int8, L16 | 116 MHz | 1.480 | 78.4 | 72.4% | 83.3% |
| Qwen3.5, fp4 / int8 | 100 MHz | 6.236 | 16.0 | 53.7% | 71.6% |
| Qwen3.5, fp4 / int8 | 116 MHz | 6.271 | 18.5 | 62.0% | 71.3% |
| Qwen3.5, fp4 / int8 | 133 MHz | 6.589 | 20.2 | 67.8% | 67.8% |
| Qwen3.5, fp4 / fp4 | 116 MHz | 5.285 | 21.9 | 57.2% | 65.7% |
| Qwen3.5, fp4 / int8, L16 | 116 MHz | 5.486 | 21.1 | 70.9% | 81.5% |
| Qwen3.5, fp4 / fp4, L16 | 100 MHz | 4.405 | 22.7 | 59.2% | 78.9% |
| Qwen3.5, fp4 / fp4, L16 | 116 MHz | 4.500 | **25.8** | 67.2% | 77.2% |

Where the cycles go (116 MHz):

- **LFM2**: the MM phases (MLP, conv, LM head) run at 97-99% of the core port. The 6 attention
  layers take 328 K cycles for 11.7 MB, 28% of their bytes: small MMs in a dependency chain,
  with MXU gaps after each QACT (one of 28 K cycles at the first `attention.py` block). That is
  the loss that keeps LFM2 under 80%: without it the token would be ~1.0 M cycles (~85%).
- **Qwen3.5**: the DeltaNet mixer takes 2.78 M cycles for 142 MB (40% of its bytes) with the VPU
  busy 97% of it: at fp4 its weights halve but the state passes do not. With 16 lanes it drops
  to 2.00 M (56%). Then it is no longer VPU-throughput-bound: removing the second RDOT pass
  altogether (a timing-only experiment, wrong results) saves 344 K cycles at 8 lanes but 8 K
  at 16. At 16 lanes one DeltaNet layer (110 K cycles, roofline 62 K) spends most of its VPU
  time in the latency of ~200 small ops per layer (VOP.mul: 51.6 K busy cycles for 4.2 K of
  work), and OUTER runs at half rate beside the state loads' TMEM writes.

### Options, ranked

Gains are projections from the table above (simulated cycles), resources from yosys
(`tools/synth/sta.sh`, logic only) or the fp4pair Vivado synthesis (191.6 K LUT, 283 DSP,
550 RAMB36 + 22 RAMB18: 64% / 15% / 59% of the xc7k480t).

1. **fp4 LM head** (d, fewer bytes; no hardware). LFM2 -34 MB/token: 77.8 -> 94.2 tok/s at
   116 MHz, 82.1 at 100. Qwen3.5 -127 MB: 18.5 -> 21.9 (25.8 with L16). The only change that
   reaches both tok/s targets; its cost is accuracy (docs/quant.md: the head is the most
   sensitive matrix), which is a product decision.
2. **Core clock >= 114 MHz** (a; the fmax branch). Needed for 80% DRAM efficiency at all with a
   128-byte port (87% ceiling at 116, 75% at 100); +15% tok/s over 100 MHz on every model.
3. **VPU 16 lanes** (existing `make bit LANES=16`, the MXU and quantizer stay on 8). Qwen3.5
   -12.5% cycles (fp4 layers, either head); LFM2 -0.7%. yosys: VPU 29.5 K -> 58.3 K LUT, 68 ->
   132 DSP, logic 4.39 -> 5.02 ns; TMEM (board ports) 12.1 K -> 35.7 K LUT; DMA and coll +1.5 K:
   about +54 K LUT in yosys terms (the earlier "+21 K" estimate is too low). On today's 64% that
   is ~80% of the part: a routing and fmax risk at 114 MHz.
4. **The LFM2 attention chain and the Qwen3.5 DeltaNet small ops** (compiler scheduling). They
   are what separates both models from 80% once the port is fast enough: ~240 K cycles of
   LFM2's 1.23 M and ~0.9 M of Qwen3.5's 4.5 M (at L16).
5. **A 256-byte weight path** (b). Two chunks per cycle: a 1024-bit AXI port per channel at
   core_clk (the smartconnect downsizes to the MIG's 512 bits and crosses clocks), port-B read
   data and the chunk FIFO 2048 bits wide (same RAMB36 count at half the depth), and an MXU
   whose two DSP pairs each take their own chunk at M = 1 (MCOLS = 4 hardware; 4-bit: the
   column-reuse split in each pair, four K-blocks per cycle and a 4-way fp32 sum; int8: one
   block per pair, two per cycle), QACT writing each row four times, and a 16-byte scale read.
   It moves the MM phases from the core's 128 B/cycle to the DRAM's 88-92%: at 116 MHz that is
   +4-5% on the MM phases (86% -> ~90% of peak); at 100 MHz, +21% (LFM2 fp4 ~94 tok/s at
   100 MHz, Qwen3.5 fp4 / L16 ~25). Cost, estimated: MXU +13-15 K LUT, +130 DSP (the second DSP
   pair per column pair; docs/quant.md's proxy: +6.1 K LUT per extra block datapath), ACT RAM
   +32 RAMB36, smartconnect 1024-bit ports +5-10 K LUT, adapter +3 K: ~+25-35 K LUT. Worth it
   only if the clock stays near 100 MHz: at 114+ it is the smallest gain on this list per LUT.
6. **A split clock** (c): the adapter or the MXU stream at the MIG's 133 MHz. On its own it
   gains nothing: the ceiling is the chunks the MXU consumes per core cycle, not the clock of the
   AXI side. It only makes sense as the CDC half of option 5, and the smartconnect already
   provides that crossing.

**Recommendation.** Keep the 128-byte port and get the clock to 114-116 MHz (2); decide the fp4
LM head (1); build LANES=16 only after a Vivado run shows it fits at that clock (3); put the
next compiler effort into LFM2's attention chain and Qwen3.5's DeltaNet small-op latency (4).
The 256-byte path (5) is the fallback if the clock cannot leave 100 MHz.
