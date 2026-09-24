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
