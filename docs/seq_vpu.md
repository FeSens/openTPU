# The sequencer and the VPU: where the area goes, and what can give it back

Question (the user, 2026-10-12): where is the design bottlenecked, and is there an architecture
change that frees as much as the TMEM study did (5 of 6 TMEM copies, 320 of 585 RAMB36, for
about 1% of cycles)? Two candidates:
- `rtl/seq/otpu_seq.sv`, the sequencer: ~16K LUT, ~11K FF, its dynamic hazard check the worst
  timing offender of the banked-TMEM builds and the MX column build;
- `rtl/vpu`, the vector unit: ~38K LUT, ~29K FF and 86 DSP in the full design.

Every number says whether it is measured (Vivado out of context, OOC), simulated (the RTL
co-simulation, Verilator) or estimated.

**Results** (section 8 has the table and the per-board recommendations):
- **Neither unit sets the throughput.** The MXU is 94-100% busy in every workload but
  Qwen3.5's prefill (decode and MoE at the DRAM's rate, prefill at the MXU's); the other units
  wait on true dataflow. Qwen3.5's prefill is the exception: DSTEP's streams keep the VPU busy
  and the MXU at 56-74% (simulated).
- **The sequencer's hazard check is 70% of its LUTs, 56% of its flip-flops and all 7 DSPs**
  (measured), and its footprint is the worst path of the banked-TMEM and MX builds and of the
  ASIC. The compiler knows those dependencies: **WAITC v2**, a 64-bit wait entry per instruction
  (a count for each in-order unit, a 31-bit mask of the VOPs for the out-of-order VPU), replaces
  the check: **6,797 LUT, 5,722 FF, 0 DSP, 2,682 slices against 14,375 / 10,729 / 7 / 5,429
  (-53% / -47% / -100% / -51%) for +7 RAMB36 of per-instruction metadata beside IMEM**
  (measured), at +0.06% of today's cycles on average over 18 workloads, -0.60% to +1.52% (simulated,
  bit-exact). Adopted for the
  MCOLS 8 build; on `waitc-mx`, off by default, with the host's side done.
- A cheaper conservative check (option b) costs +0.9% to +61% of the cycles: out.
- **The VPU is 61% DSTEP's datapath**, which stays. Its composite path at 4 lanes saves 2.9K LUT
  for up to +8.9% of prefill, at 2 lanes up to +25%: keep 8 lanes. The VPU's lever is fewer
  passes over TMEM (work folded into the units' input and output paths), not narrower lanes;
  with the data-movement study's fold the composites leave the VPU and 1-2 lanes suffice.
- 30-38% of v2's LUTs are one trace field (the start's ready cycle), which the host can derive
  from the trace and the entries.
- **Qwen3.5's prefill: the DeltaNet state once per prompt run** (section 9; designs and RTL on
  `dstep-mr`, docs/dstep_mr.md). The card's state step is bound by the stream engine's 8 words a
  cycle, not by DRAM: stepping a run's R rows with the state read and written once cuts its DRAM
  bytes by R (37.7 -> 9.4 MB a 0.8B row at R = 4) for -1.8% to -3.7% of the cycles (built,
  bit-exact); two chained engines take -19% of Qwen3.5-0.8B's run (bound; -18.6% as built).
  The chain is parked (2026-10-12): the ASIC's prefill step goes to the MXU (the
  chunkwise delta rule, docs/dstep_mr.md 9) in the release after r2.

## 1. Method

- **Area**: Vivado 2026.1 out of context on the YPCB's part (xc7k480tffg1156-2) at 7.5 ns
  (133.33 MHz), the card's parameters (`IMEM_WORDS` 32768, `TMEM_WORDS` 65536, `D` 128, `WIN`
  16; the VPU: 8 lanes, the stream engine, the write buffer). `rtl/exp/seq_vpu/run_ooc.sh`:
  the module as built (placed and routed), ablations in synthesis
  (`rtl/exp/seq_vpu/abl/otpu_seq.sv`: the hazard check pruned piece by piece), and the flat
  cells grouped by the RTL name they carry (`ooc.tcl`). UltraScale+ (the BCU1525's and F2's
  family): xcku5p-2 at 4.0 ns as the stand-in.
- **Cycles**: the RTL co-simulation of the card's configuration (PAIR, DSTEP, STREAM, LiteDRAM's
  controller at DDR3-1066, the core at 133.33 MHz), `rtl/exp/seq_vpu/run_cosim.sh` over
  `lists/all.txt` and `lists/resident.txt`:
  - decode, one token, whole model: Qwen3-0.6B (`d_q3`), Qwen3.5-0.8B (`d_q35`), LFM2.5-230M
    (`d_lf`); the resident decode (`dr_q3`, `dr7_q3`: one program, positions 544 and 700 as
    run arguments);
  - prefill at MCOLS 4, 4 rows: `p_q3`, `p_q35`, `p_lf`; Qwen3.5-4B and Phi-4-mini layers on zero
    images (`z_q4_r2`, `z_ph_r3`); at MCOLS 8: `p8_q3`, `p8_q35`;
  - batched decode: 4 slots (`b4_q3`), 8 slots at MCOLS 8 (`b8_q3`);
  - MoE (every expert resident): the layer-major prefill's layer runs (`l_q35m`, `l_g26`) and
    a decode row through the MoE block (`m_q35m`, `m_g26`).
  Every perf_qwen run is checked bit-exact against the ISA simulator (`--check`).
- **Options in simulation before RTL**: `otpu_seq` and `otpu_vpu` take simulation-only knobs
  (`+hz_*`, `+vt_*`; nothing synthesized) that emulate each option exactly at the cycle level:
  coarser footprints, smaller windows, in-order units, per-instruction wait counts from an
  oracle (the compiler's insertion emulated), and a narrower VPU path at today's latency
  (`cands.txt`). The finalist, explicit waits (WAITC), is real RTL, and its counts come from a
  real compiler pass (`opentpu/waits.py`).

## 2. Where the area is (measured, OOC)

### 2.1 The sequencer

`otpu_seq` as built, placed and routed: **14,375 LUT (+912 LUT RAM), 10,729 FF, 7 DSP48E1,
5,429 slices, 28.5 RAMB36 (IMEM)**, WNS +0.363 ns at 7.5 ns. What it does per instruction: fetch
(IMEM rows of 4), resolve the register fields, then four pipeline stages that turn the
instruction into its footprint (R -> P -> S -> Q: the multiplications on 7 DSPs, then the
ranges, as otpu_pkg `footprint()`), and at C compare that footprint with the 16 window slots'
(per slot: one DRAM pair, a TMEM write and four TMEM reads, an ACT range; `conf_s`), keeping
each slot's dependency row (`sdep`, and `sdepd` for the MXU's stream start), the age matrix,
per-unit start selection, the MXU's two-phase release, per-unit command queues and the trace
events.

Ablations (synthesis, the same settings; `seq_s` = 14,404 LUT / 10,601 FF is their reference):

| variant | LUT | FF | DSP | what it removes |
|---|---|---|---|---|
| as built | 14,404 | 10,601 | 7 | |
| no hazard check (`ABL_NOHAZ`: no footprints, no compares) | 4,270 | 4,622 | 0 | **10,134 LUT, 5,979 FF, 7 DSP (70% / 56%)** |
| footprints computed, no compares or slot storage (`ABL_NOCMP`) | 8,235 | 4,958 | 7 | the compares and the slots' footprints: 6,169 LUT, 5,643 FF |
| no DRAM compares (`ABL_NODRAM`) | 11,582 | 8,418 | 7 | 2,822 LUT, 2,183 FF |
| no TMEM compares (`ABL_NOTMEM`) | 11,245 | 7,762 | 7 | 3,159 LUT, 2,839 FF |
| an 8-slot window | 9,564 | 7,111 | 8 | 4,840 LUT, 3,490 FF |
| TMEM 128K words (TAW 17) | 14,595 | 10,781 | 7 | +191 LUT, +180 FF |
| TMEM 256K words (TAW 18) | 14,740 | 10,944 | 7 | +336 LUT, +343 FF |

So **70% of the sequencer's LUTs and 56% of its flip-flops are the hazard check**: ~6.2K LUT and
~5.6K FF compare and store footprints, ~4.0K LUT and 7 DSPs compute them. The rest (~4.3K LUT:
fetch, registers, loops, the window's bookkeeping, start selection, the command queues, the
trace) is what any sequencer needs. The check is also where its timing is: the worst paths of
the synthesized runs end in `c_fp` (the footprint register) or start at `q_pr` / `q_cmd`; the
MX column build's `u_seq q_cmd -> c_fp` and the banked-TMEM builds' sequencer paths are these.
A larger TMEM costs the check little (TAW 18: +2.3% LUT): the ranges were already 32-bit
until the C stage.

### 2.2 The VPU

`otpu_vpu` as built (8 lanes, the stream engine, the write buffer), placed and routed: **30,797
LUT + 2,901 LUT RAM/SRL, 27,415 FF, 86 DSP48E1, 12,219 slices, 16 RAMB18**, WNS +0.261 ns (its
worst paths: the write buffer's enable fan-out, `en_r` / `en_q`). Vivado's hierarchy:

| part | total LUT | FF | DSP | what it is |
|---|---|---|---|---|
| `g_se.u_tail` (otpu_se_tail) | 11,610 | 9,562 | 54 | the stream engine's tail: DSTEP's d stage, U (8 `otpu_fmma`) and Q (8 fmul + fadd); composites also run through U and Q |
| `g_comp.u_comp` (otpu_se_comp) | 10,885 | 6,978 | 0 | the composite functions' microcode, state, operand muxes and delay lines (EXP2, EXP2SUB, RECIP, RSQRT, LOG2) |
| `g_slane[0..7].u_ma` | 4,789 | 2,472 | 32 | slot 0: 8 `otpu_fmma` (every elementwise op, the reductions' partial loop, DSTEP's row dots) |
| `u_vt` (otpu_vtree) | 4,187 | 3,541 | 0 | the row-sum trees (RSUM, RSSQ, RDOT, and DSTEP's dots) |
| the rest | 2,168 | 4,678 | 0 | decode, operand buffers, the write buffer, enables |

DSTEP (Qwen3.5's DeltaNet recurrence, 8 state words a cycle) uses slot 0, the tail and the trees:
**61% of the VPU's LUTs belong to DSTEP's datapath**, which the brief says must stay whole. A
"half VPU" (4 lanes) cannot shrink them without halving DSTEP. What the VPU has besides DSTEP's
datapath is the composite path (`se_comp`, 32% of its LUTs). Standalone at 7.5 ns (synthesis):

| se_comp lanes | LUT | LUT RAM / SRL | FF | |
|---|---|---|---|---|
| 8 (today) | 6,020 | 2,195 | 7,009 | |
| 4 | 3,102 | 1,141 | 3,696 | -2.9K LUT, -1.1K LUT RAM, -3.3K FF |
| 2 | 1,645 | 615 | 2,052 | -4.4K LUT, -1.6K LUT RAM, -5.0K FF |

(Standalone counts omit the operand selection and delay lines the VPU's hierarchy adds around
it: 10.9K in the VPU against 6.0K + 2.2K alone; those scale with the lanes too.)

## 3. What the hazard check does on real programs (simulated)

`+hz_log` statistics of today's sequencer on every workload (`report.py hazards`): of each
unit's cycles, busy, idle with nothing to start, and blocked by a dependency (on which unit);
the instructions with no dependency at all when they dispatch; the window full (dispatch held).

| run | cycles | instr | no dep | window full | RLD wait | MXU busy | DMA busy / blocked | QUANT busy / blocked | VPU busy / not ready / blocked |
|---|---|---|---|---|---|---|---|---|---|
| d_lf.0 | 1,339,842 | 4,651 | 21% | 58% | 0% | 100% | 3% / 52% | 5% / 51% | 12% / 0% / 46% |
| d_q35.0 | 4,760,853 | 11,585 | 18% | 89% | 0% | 100% | 20% / 46% | 2% / 31% | 15% / 14% / 26% |
| d_q3.0 | 3,623,644 | 17,071 | 21% | 85% | 0% | 100% | 3% / 35% | 9% / 49% | 16% / 0% / 48% |
| dr7_q3.0 | 3,697,238 | 17,295 | 21% | 85% | 0% | 100% | 4% / 37% | 9% / 49% | 17% / 0% / 47% |
| dr_q3.0 | 3,694,791 | 17,295 | 21% | 85% | 0% | 100% | 4% / 37% | 9% / 49% | 17% / 0% / 47% |
| p8_q35.0 | 942,492 | 2,901 | 28% | 99% | 0% | 56% | 62% / 5% | 4% / 32% | 37% / 46% / 18% |
| p8_q3.0 | 502,925 | 6,077 | 24% | 96% | 0% | 95% | 9% / 0% | 27% / 68% | 52% / 0% / 48% |
| p_lf.0 | 1,630,055 | 10,975 | 24% | 99% | 0% | 99% | 5% / 11% | 13% / 84% | 33% / 0% / 67% |
| p_q35.0 | 1,137,275 | 2,901 | 28% | 99% | 0% | 74% | 50% / 6% | 3% / 42% | 30% / 38% / 32% |
| p_q3.0 | 372,383 | 3,293 | 23% | 96% | 0% | 97% | 6% / 2% | 20% / 74% | 36% / 0% / 63% |
| z_ph_r3.0 | 1,677,624 | 2,403 | 22% | 91% | 0% | 100% | 1% / 1% | 4% / 85% | 9% / 0% / 91% |
| z_q4_r2.0 | 2,225,099 | 4,469 | 30% | 97% | 0% | 100% | 25% / 3% | 2% / 61% | 15% / 20% / 65% |
| b4_q3.0 | 1,671,279 | 3,812 | 26% | 98% | 2% | 99% | 1% / 0% | 5% / 17% | 13% / 0% / 85% |
| b8_q3.0 | 1,827,502 | 7,734 | 24% | 99% | 0% | 97% | 2% / 1% | 8% / 20% | 24% / 0% / 73% |
| l_g26.0 | 969,765 | 1,777 | 27% | 89% | 17% | 96% | 3% / 16% | 3% / 72% | 11% / 0% / 83% |
| l_g26.1 | 993,612 | 1,504 | 25% | 90% | 17% | 95% | 3% / 36% | 2% / 66% | 10% / 0% / 84% |
| l_q35m.0 | 382,905 | 1,963 | 32% | 76% | 26% | 96% | 52% / 27% | 1% / 27% | 26% / 38% / 26% |
| l_q35m.1 | 296,391 | 1,111 | 27% | 57% | 48% | 94% | 6% / 34% | 5% / 53% | 22% / 0% / 66% |
| m_g26.0 | 758,072 | 1,178 | 27% | 86% | 23% | 98% | 1% / 2% | 2% / 69% | 7% / 0% / 92% |
| m_g26.1 | 1,514,737 | 2,353 | 27% | 87% | 23% | 98% | 1% / 2% | 2% / 69% | 7% / 0% / 92% |
| m_g26.2 | 1,613,257 | 2,220 | 27% | 88% | 22% | 98% | 1% / 14% | 2% / 67% | 6% / 0% / 93% |
| m_q35m.0 | 4,579,588 | 1,279 | 26% | 76% | 2% | 100% | 3% / 94% | 0% / 3% | 1% / 2% / 4% |
| m_q35m.1 | 5,227,461 | 3,695 | 27% | 75% | 6% | 100% | 6% / 85% | 0% / 7% | 3% / 4% / 11% |
| m_q35m.2 | 5,490,770 | 4,366 | 27% | 74% | 8% | 100% | 6% / 81% | 0% / 9% | 4% / 4% / 15% |

(`.n`: the MoE tool's runs, one per layer prefix. "Busy" is from start to completion: an MM
streaming its weights from DRAM is busy.)

What this says:
- **The MXU is the bottleneck almost everywhere** (94-100% busy): decode and MoE stream weights
  at the DRAM's rate, prefill and batched decode are MXU-bound at MCOLS 4. The exception is
  Qwen3.5's prefill (56-74%: the VPU is "not ready" 38-46% of the cycles, busy with DSTEP
  streams).
- **The other units wait on true dataflow**, not on false conflicts: the blocked cycles are
  QACT waiting for the VPU's normalization, the VPU waiting for the MM's result, the ST waiting
  for the MM (`report.py hazards`, "blocked cycles by (waiting, producer)": QACT<-VOP,
  VOP<-MM, ST<-MM in every workload; the 35B's MoE decode row: the DMA's expert fetches).
- **The window is full most of the time** (57-99%): the sequencer runs ahead until 16
  instructions wait, mostly behind the MXU. What it buys is the overlap of the next MMs' weight
  streams with the VPU / quantizer work. Today the DMA, the quantizer and the VPU start their
  oldest *ready* instruction, not their oldest: only the VPU's use of that freedom matters
  (section 4: the VPU in program order costs 5-8% of prefill; the DMA and the quantizer cost
  0-0.2%).
- **The dependencies are short in each unit's own order**: of a VOP's dependencies on an older
  VOP, 52-94% are on the VPU's last instruction and 94-99% within its last 4; QACT's on the
  VPU are within the VPU's last 2 (100%); a VOP's on the MXU within its last 8 in 65-94% and
  always within 15. 18-32% of the instructions have no dependency when they dispatch.

## 4. The sequencer's options, in cycles (simulated)

Each option against today's sequencer (`i0`), cycles of the whole run, every perf_qwen run
bit-exact against the ISA simulator. Section 4.1 is option (b), the ISA unchanged and a cheaper
conservative check; 4.2-4.4 are option (a), explicit dependencies.

### 4.1 (b) A cheaper check: coarser footprints, a smaller window

The cheapest conservative checks keep the footprints but compare fewer or coarser ranges: the
reads' hull (`rh`), one DRAM range (`dh`), TMEM / DRAM granules (`g*`, `t*`, `d*`), or fewer
slots (`w8`). Granules of 256 TMEM words and 1 MB of DRAM on top of the hulls (`i0g8`, which
would save roughly the compares' LUTs: section 2.1) cost (`#n`: a MoE run's layer prefix):

| workload | i0 cycles | `i0g8` |
|---|---|---|
| d_lf | 1,339,842 | +11.89% |
| d_q3 | 3,623,644 | +15.51% |
| d_q35 | 4,760,853 | +10.22% |
| dr7_q3 | 3,697,238 | +17.63% |
| dr_q3 | 3,694,791 | +17.66% |
| p8_q3 | 502,925 | +60.95% |
| p8_q35 | 942,492 | +13.88% |
| p_lf | 1,630,055 | +37.29% |
| p_q3 | 372,383 | +46.08% |
| p_q35 | 1,137,275 | +14.95% |
| z_ph_r3 | 1,677,624 | +10.61% |
| z_q4_r2 | 2,225,099 | +8.95% |
| b4_q3 | 1,671,279 | +10.50% |
| b8_q3 | 1,827,502 | +18.36% |
| l_g26#0,1 | 969,765 | +4.38% |
| l_g26#0,5 | 993,612 | +3.62% |
| l_q35m#1 | 382,905 | +16.32% |
| l_q35m#4 | 296,391 | +12.04% |
| m_g26#0 | 758,072 | +2.54% |
| m_g26#0,1 | 1,514,737 | +2.53% |
| m_g26#0,5 | 1,613,257 | +2.16% |
| m_q35m#1 | 4,579,588 | +0.86% |
| m_q35m#3 | 5,227,461 | +2.10% |
| m_q35m#4 | 5,490,770 | +2.35% |

`i0g8` decides option (b); its finer variants were stopped to free the slot for v2's last
workloads. The hulls alone (`rhdh`: the reads' hull and one DRAM range per instruction, no
granule), which finished on nine workloads before that, already cost +3.9% to +30% (decode
+4.0-6.9%, prefill +3.9-30%: p8_q3 +30.4%, p_q3 +23.2%, p_lf +16.5%). The 8-slot window (`w8`)
and the one-sided granules (`d32`, `t16`) were not run.

The granules make unrelated tiles of one buffer conflict (the K/V cache's blocks, an MM's
output tiles against the next MM's inputs), and the prefill runs, where the VPU and the
quantizer overlap the MXU on neighbouring tiles, lose the most. **Option (b) is out**: no
granule that saves area is free, and the window is already full most of the time (section 3).

### 4.2 (a) Explicit dependencies: what in-order units cost

With explicit dependencies the sequencer does not compare footprints at dispatch, so a unit can
no longer pick a ready instruction among its waiting ones by looking at them: it starts its
oldest, or the compiler must say which are ready. The exact dependencies with a unit started
in program order (`+hz_inom`, today's check otherwise):

| workload | all in order (`ino`) | DMA (`inod`) | QUANT (`inoq`) | VPU (`inov`) | QUANT + VPU |
|---|---|---|---|---|---|
| p_q3 | +4.98% | +0.00% | -0.33% | +5.28% | +4.98% |
| p8_q3 | +8.67% | +0.01% | +0.20% | +7.98% | +8.67% |
| b4_q3 | +1.36% | +0.00% | +0.02% | +1.20% | +1.36% |

Only the VPU's order matters: prefill interleaves independent VPU chains (a tile's softmax
behind its MM, the next tile's normalization ahead of it), and the VPU in order waits on the
first. The DMA and the quantizer in order cost nothing.

### 4.3 (a) Wait counts: three designs

Cycles against `i0`. `wcr` is v1 as built with the oracle's counts (every unit in order),
`wcc` v1 with the compiler's counts; `wcv` and `wcvx1` are simulations (below); `wcm` is v2 as
built with the compiler's entries (`opentpu/waits.py`), the design adopted. Every RTL run is
bit-exact against the ISA simulator; a blank was not run.

| workload | i0 cycles | v1 (`wcr`) | v1, compiler (`wcc`) | watermark (`wcv`) | VPU exact (`wcvx1`) | **v2 (`wcm`)** |
|---|---|---|---|---|---|---|
| d_lf | 1,339,842 | +0.45% | | +0.17% | | **+0.18%** |
| d_q3 | 3,623,644 | +0.18% | +0.33% | +0.14% | | **+0.13%** |
| d_q35 | 4,760,853 | +0.64% | | | | **+0.05%** |
| dr7_q3 | 3,697,238 | +2.81% | +4.15% | +2.62% | | **-0.32%** |
| dr_q3 | 3,694,791 | +2.80% | +4.23% | +2.69% | -0.41% | **-0.38%** |
| p8_q3 | 502,925 | +8.69% | | +7.83% | +0.30% | **+0.79%** |
| p8_q35 | 942,492 | +1.42% | | | | **+0.69%** |
| p_lf | 1,630,055 | +3.28% | | +3.21% | | **-0.02%** |
| p_q3 | 372,383 | +5.07% | +4.95% | +4.63% | -0.33% | **-0.14%** |
| p_q35 | 1,137,275 | +1.24% | | +1.13% | | **+0.23%** |
| z_ph_r3 | 1,677,624 | +1.25% | | | | **+0.08%** |
| z_q4_r2 | 2,225,099 | +0.21% | | | | **+0.12%** |
| b4_q3 | 1,671,279 | +1.36% | | +1.21% | | **+0.09%** |
| b8_q3 | 1,827,502 | +2.32% | | | | **+0.15%** |
| l_g26#0,1 | 969,765 | +1.05% | | | | **+0.78%** |
| l_g26#0,5 | 993,612 | +0.90% | | | | **+0.56%** |
| l_q35m#1 | 382,905 | +1.06% | | | | **+1.51%** |
| l_q35m#4 | 296,391 | +4.54% | | | | **+2.15%** |
| m_g26#0 | 758,072 | +0.41% | | | | **+0.30%** |
| m_g26#0,1 | 1,514,737 | +0.39% | | | | **+0.27%** |
| m_g26#0,5 | 1,613,257 | +0.29% | | | | **+0.23%** |
| m_q35m#1 | 4,579,588 | +0.02% | | | | **+0.03%** |
| m_q35m#3 | 5,227,461 | +0.06% | | | | **+0.09%** |
| m_q35m#4 | 5,490,770 | +0.07% | | | | **+0.10%** |

- **v1** (every unit in order, a 4-bit count per unit: "the k-th last instruction of unit u
  and every older one"): the VPU's in-order start, +1.2-8.7% on prefill (Qwen3's +5-9%), +2.8%
  on the resident decode (+4.2% with the compiler's counts).
- **The VPU out of order with counts** (`wcv`): a count on the VPU now means "the k-th last VOP
  and every older one" (a completion watermark), and a consumer of one VOP waits behind older
  VOPs that are blocked on an MM. It gets back almost nothing.
- **The VPU's dependencies exact** (`+hz_wcvx=1`, simulation: today's range check for the VPU's
  producers, the oracle's counts for the other units, the DMA and the quantizer in order): within
  today's (dr_q3 -0.41%, p_q3 -0.33%, p8_q3 +0.30%). Exact only for VOP consumers (`wcvx2`):
  +1.2% to +3.7%. Every consumer of a VOP needs it.
- **v2** (built): a VPU mask per instruction, naming each VOP it waits for. The VPU keeps
  starting its oldest ready VOP, and a consumer waits for exactly its producers: v2 lands where
  the exact simulation did (p_q3 -0.14% against -0.33%, dr_q3 -0.38% against -0.41%, p8_q3
  +0.79% against +0.30%). With the first mask, 15 bits (the column above), the mean over the 18
  workloads is +0.25% (+0.32% counting each MoE prefix as a run), and every run is within +0.8%
  but the 35B's layer runs (l_q35m: +1.51% and +2.15%). Section 4.5 splits that cost; its
  largest part was the 15-VOP hold, and the final mask is 31 bits (`wcm31`, below).

**The final mask** (job V: `wcm` on 38a6342b, the 31-bit mask and the hold at 31, against `i0`
with WAITC off on its parent e274fc4a; both on `waitc-mx`, the MX freeze with WAITC, so `i0`'s
cycles are not the column above's). The 12 runs that compare against the ISA simulator
(perf_qwen's) are bit-exact; the zero-image and layer tools count cycles:

| workload | i0 cycles | **v2 final (`wcm`, 31-bit)** |
|---|---|---|
| d_lf | 1,354,920 | **+0.02%** |
| d_q3 | 3,663,067 | **-0.13%** |
| d_q35 | 4,786,435 | **+0.02%** |
| dr7_q3 | 3,738,260 | **-0.60%** |
| dr_q3 | 3,736,639 | **-0.51%** |
| p8_q3 | 506,472 | **-0.37%** |
| p8_q35 | 952,135 | **+0.05%** |
| p_lf | 1,645,954 | **-0.00%** |
| p_q3 | 374,524 | **+0.22%** |
| p_q35 | 1,141,254 | **+0.01%** |
| z_ph_r3 | 1,683,871 | **+0.04%** |
| z_q4_r2 | 2,251,563 | **-0.00%** |
| b4_q3 | 1,647,306 | **-0.06%** |
| b8_q3 | 1,798,097 | **+0.24%** |
| l_g26#0,1 | 967,330 | **+0.49%** |
| l_g26#0,5 | 988,619 | **+0.35%** |
| l_q35m#1 | 388,844 | **+1.03%** |
| l_q35m#4 | 302,998 | **+1.52%** |
| m_g26#0 | 755,866 | **+0.38%** |
| m_g26#0,1 | 1,510,335 | **+0.36%** |
| m_g26#0,5 | 1,605,752 | **+0.30%** |
| m_q35m#1 | 4,484,132 | **+0.03%** |
| m_q35m#3 | 5,148,870 | **+0.02%** |
| m_q35m#4 | 5,414,569 | **+0.08%** |

The mean over the 18 workloads is **+0.06%** (+0.15% counting each MoE prefix as a run), from
-0.60% (dr7_q3) to +1.52%; every run is within +0.5% but the 35B's layer runs (+1.03% and
+1.52%), whose remaining cost section 4.5 splits (the in-order DMA and quantizer, the compiler's
counts). The 31-bit mask took 0.43-0.45 points off them and 0.82 off p8_q3 (+0.45% -> -0.37%),
the hold's share as predicted.

### 4.4 v2 as built: area and timing (measured, OOC)

`otpu_seq` WAITC 1 (`SEQ_WC`): the dispatch reads the instruction's entry from a sidecar RAM
beside IMEM (59 bits an entry, the counts' 28 and the mask's 31; 1,024 rows of 4 for IMEM's
4,096 instructions) and keeps per slot a target count for each wait (`wt`: the producer's index
+ 1 in its unit's order; a wait holds until the unit's completion counter `wc_n` reaches it) and
the VOP slots it waits for (`vpend`, cleared as each VOP completes). A VOP is dispatched only
once the VOP 31 before it has completed (the hold), so a 31-bit mask names every VOP a later
instruction may wait for. No footprint is computed; the R -> P -> S -> Q stages carry only the
command. The same flow as section 2.1 (xc7k480t-2 at 7.5 ns, placed and routed; xcku5p-2 at
4.0 ns as the UltraScale+ stand-in):

| build | LUT | LUT RAM | FF | DSP | slices | RAMB36 | CARRY4 | WNS | worst path |
|---|---|---|---|---|---|---|---|---|---|
| today (`i0`) | 14,375 | 912 | 10,729 | 7 | 5,429 | 28.5 | 2,066 | +0.363 ns | `sv` -> a unit's command copy |
| v1 (counts, all in order) | 5,609 | 912 | 5,245 | 0 | | 32.5 | | | |
| v2, 15-bit mask | 5,944 | 916 | 5,402 | 0 | 2,410 | 33.5 | 344 | +0.466 ns | `sp` -> IMEM's read address (fetch) |
| **v2, 31-bit mask (final, 38a6342b)** | **6,797** | 916 | **5,722** | **0** | **2,682** | 35.5 | 344 | +0.494 ns | `wc_n` -> a unit's command copy (13 levels) |
| final - today | **-7,578 (-53%)** | +4 | **-5,007 (-47%)** | **-7** | **-2,747 (-51%)** | +7 | -1,722 | +0.131 ns | |
| today, xcku5p 4.0 ns | 15,166 | 794 | 10,754 | 7 | | 28.5 | | +0.027 ns | `c_fp` (the footprint, 14 levels) |
| v2 15-bit, xcku5p 4.0 ns | 5,935 | 798 | 5,402 | 0 | | 33.5 | | +0.068 ns | `wc_n` -> a unit's command copy (9 levels) |
| final, xcku5p 4.0 ns | not run (dropped for release r2, 2026-10-13) | | | | | | | | |

- **The +7 RAMB36 are per-instruction metadata, not a data buffer**: the sidecar holds each
  instruction's wait entry (236 Kbit for IMEM's 4,096 instructions), written by the host with
  the program (`waits.loadable`) and read once per dispatch beside the instruction's IMEM row.
  It is instruction bytes, like IMEM's own 28.5 RAMB36: no operand passes through it and no
  data is copied into it, so it adds no data movement (the study's rule is about data movement
  and buffer copies, not the program's encoding).
- The VPU mask's own logic at 15 bits (the pending set `vpend`, the hold and the mask's sidecar
  bits) is **619 LUT and 324 FF** (synthesis: 6,002 LUT with it, 5,383 without); v1 -> v2 costs
  +335 LUT, +157 FF and 1 RAMB36 and gets the VPU's out-of-order start back (prefill: +1.2-8.7%
  with v1, -0.1% to +0.8% with v2). 15 -> 31 bits costs **+853 LUT, +320 FF, +272 slices and +2
  RAMB36** (the mask's pending sets and indices, 32 wide) and removes the hold's cost (section
  4.5).
- The footprint path is gone from both families' worst paths: on the K7 the final build's worst
  is the wait test into a unit's command copy (+0.494 ns at 7.5 ns), on the KU+ at 250 MHz the
  same path in the 15-bit build. On the 7-series device (74,650 slices) the final build frees
  3.7% of the slices, 7 DSPs and the timing offender of the banked-TMEM and MX column builds
  (`u_seq q_cmd -> c_fp`).
- **The wait test, registered (release r2).** In the full build, r2's m4 at 133.33 MHz
  (`r2m4_d352e53d`), that worst path failed:
  - `wc_n` -> `g_cmd q`: 358 endpoints, -0.482 ns, 11.5 levels, 16% of core_clk's TNS.
  - The fix registers the per-slot results as `w_all_q` / `w_dram_q`. The starts, the MXU's
    release and the ready stamp read those, and the per-slot compares end at flip-flops.
  - A wait releases a cycle after its count is reached. A slot's readiness only rises while it
    holds its instruction, so last cycle's value is never wrong. The slot written this cycle
    clears its flag, so an instruction starts two cycles after its dispatch at the earliest.
  - Co-simulated on m4 (LiteDRAM DDR3-1066, against d352e53d), the cost is noise:

    | run | change |
    |---|---|
    | Qwen3.5-0.8B tiled P, b1, T4 | +0.011% |
    | Qwen3.5-0.8B decode at 544 | -0.013% |
    | Qwen3-0.6B decode at 544 | -0.012% |
    | Qwen3-0.6B decode at 1800 | -0.008% |

  - The same runs with the scoreboard (`SEQ_WC` 0) are -0.57% to +0.19% against WAITC.
  - WAITC stays for its area and for the hazard check it removed.
- **FOLLOW (S > 1).** With wait entries across slices (`otpu_seq` FOLLOW, `otpu_slice`: at
  S > 1), bit 28 of an entry's counts' word (docs/isa.md "Wait counts") ends a VOP's collective
  wait at its GATHER's start: the count is met when the unit runs the producer (`c_run`, `wc_n`
  one short of it). The VPU then starts with `ufol` and reads behind the GATHER's writes
  (`otpu_vpu` FOLLOW). The sidecar entry is 60 bits there (240 a row of 4). Its use:
  a norm's sum of squares after its all-gather, taken as the row arrives.

### 4.5 Where v2's remaining cycles go (simulated)

v2 with the 15-bit mask against today on the runs where it cost the most, built up one
restriction at a time (`split`: the MX freeze's tree, e274fc4a, so `i0` differs from 4.3's by up
to 1.6%; every run bit-exact; `#n` a MoE run's layer prefix):
- `inodq`: today's check with the DMA and the quantizer started in program order, as v2 starts
  them (the MXU and the collective unit already are): exact dependencies, v2's order;
- `wcmx0`: the compiler's counts for the in-order units, the VPU's producers exact (today's
  range check, `+wc_vx`), no hold;
- `wcmx`, `wcmx24`, `wcmx31`: the same with the hold at 15, 24 and 31 VOPs;
- `wcm`: v2 with the 15-bit mask; `wcm8`: the hold at 8; `wcma`: v2 with the entries computed
  for the run's own arguments (`OTPU_WAITS=args`) instead of their bounds.

| workload | i0 cycles | `inodq` | `wcmx0` | `wcmx` (15) | `wcmx24` | `wcmx31` | **`wcm`** | `wcm8` | `wcma` |
|---|---|---|---|---|---|---|---|---|---|
| p8_q3 | 506,472 | -0.22% | -0.37% | +0.45% | -0.37% | -0.37% | **+0.45%** | +0.69% | +0.45% |
| l_g26#0,1 | 967,330 | +0.32% | +0.45% | +0.63% | +0.46% | +0.45% | **+0.68%** | +0.80% | +0.68% |
| l_g26#0,5 | 988,619 | +0.12% | +0.31% | +0.46% | +0.40% | +0.31% | **+0.51%** | +0.74% | +0.51% |
| l_q35m#1 | 388,844 | +0.18% | +0.93% | +1.37% | +1.02% | +0.93% | **+1.46%** | +1.92% | +1.46% |
| l_q35m#4 | 302,998 | +0.69% | +1.37% | +1.84% | +1.38% | +1.37% | **+1.97%** | +3.14% | +1.97% |

So on the 35B's layer runs (#1 / #4, points):
- **the in-order DMA and quantizer**: +0.18 / +0.69 (`inodq`). The layer runs' expert
  weight streams (LD) and the attention's QST would pass an older blocked one today.
- **the compiler's counts over the exact dependencies**: +0.75 / +0.68 (`wcmx0` - `inodq`;
  the 26B: +0.13 / +0.19). Knowing the run's arguments changes nothing (`wcma` = `wcm`), so the
  conservative counts do not come from the arguments' bounds, nor from the RLDs' (the RLD
  bounds below).
- **the 15-VOP hold**: +0.44 / +0.47 (`wcmx` - `wcmx0`), and on p8_q3 all of v2's cost (+0.82
  points of +0.45%: without the hold v2's order gains 0.37%). At 24 it costs +0.09 / +0.01, at
  31 nothing on these runs; at 8 (`wcm8`) +0.46 / +1.17 more than at 15.
- **the mask** over the exact VPU dependencies: +0.09 / +0.13 (`wcm` - `wcmx`).

Hence the final mask: 31 bits, the hold at 31 (38a6342b), +2 RAMB36 and +853 LUT (section 4.4)
for the hold's +0.4-0.8 points. Measured (job V, section 4.3's final table): l_q35m #1 / #4
at +1.03% / +1.52% (from +1.46% / +1.97%), the hold's 0.43 / 0.45 points; what is left is the
in-order DMA and quantizer (+0.18 / +0.69) and the compiler's counts (+0.75 / +0.68).

**Expert-major MoE** (a proxy, not the expert-major schedule: `moe: expert` runs the 35B's
expert FFN over M = 4 tokens, the kernel an expert-major prefill would run per expert; M = 16
does not fit TMEM in this tool): 28,337 cycles, v2 -0.01% (15 and 31 alike).

**The RLD bounds** (the compiler item this section left open; branch `waitc-rld`): the MoE
layer runs address each expert's weights from the routing (an RLD loads the expert's index at
run time), and the pass took an RLD's value as any 32-bit value, so every footprint computed
from it covered the whole space it addresses. `isa.rld(..., bound=(lo, hi))` now carries what
the kernel knows of the word (flag bit 2, w3..w4; the hardware ignores it, the ISA simulator
stops on a value outside it), and the pass uses it: the experts' indices (0 .. E-1), the
directory's and the answers' offsets, the slots (the pool's first and last), a row's place in
the scratch, the present / missing flags. On the 35B's layer programs 13 of 14 RLDs are
bounded, 116 / 124 entries change, the VPU masks' bits fall from 1,473 to 985 and 1,570 to
1,064, and no loop of unknown count is widened any more (28 -> 0). Cycles (simulated, job R,
wcm against 38a6342b's, the same i0):

| row | i0 cycles | wcm (38a6342b) | wcm with the RLD bounds |
|---|---|---|---|
| l_q35m#1 | 388,844 | +1.03% | +0.93% |
| l_q35m#4 | 302,998 | +1.52% | +1.37% |
| l_g26#0,1 | 967,330 | +0.49% | +0.46% |
| l_g26#0,5 | 988,619 | +0.35% | +0.31% |
| m_g26#0 / #0,1 / #0,5 | 755,866 / 1,510,335 / 1,605,752 | +0.38 / +0.36 / +0.30% | +0.32 / +0.31 / +0.27% |
| m_q35m#1 / #3 / #4 | 4,484,132 / 5,148,870 / 5,414,569 | +0.03 / +0.02 / +0.08% | +0.01 / +0.00 / +0.05% |

The bounds take 0.09 / 0.14 points off the 35B's layer runs, which is the VPU mask's share
over the exact producers (38a6342b's `wcm` against `wcmx31`: +0.10 / +0.15): l_q35m is now at `wcmx31`'s
+0.93% / +1.37%. The counts' +0.75 / +0.68 over the exact dependencies stay, so the
RLD-derived addresses were not their cause; what is left is the in-order DMA and quantizer
(+0.18 / +0.69) and counts that must hold for every iteration of a loop (a loop body's entry
is the worst over its iterations; not measured further).

**The entries' ASIC form.** The dense sidecar grows with IMEM's depth (section 8: +23% of
IMEM's bits, ~2.6x the hazard check at 8K instructions on nangate45), so how compressible are
the entries? Measured on the programs themselves (static, one entry per instruction, the
production pass with the arguments' bounds; no simulation):

| program | instructions | (a) no wait | (b) = previous | (c) entry changes | distinct entries | count words / masks |
|---|---|---|---|---|---|---|
| dr_q3 (Qwen3-0.6B resident decode) | 680 | 11.2% | 18.7% | 81.0% | 134 | 61 / 31 |
| p_q3 (4-row prefill, 2 layers) | 1,653 | 15.1% | 8.4% | 91.5% | 267 | 145 / 75 |
| l_q35m#1 (35B layer run, DeltaNet) | 765 | 10.9% | 23.6% | 75.2% | 221 | 72 / 125 |
| l_q35m#4 (35B layer run, attention) | 627 | 6.0% | 9.9% | 88.5% | 337 | 135 / 169 |
| 35B decode, bucket 1 | 1,893 | 2.1% | 18.9% | 79.3% | 690 | 229 / 182 |
| 35B generate (chained), bucket 1 | 1,948 | 1.5% | 17.7% | 80.2% | 751 | 232 / 231 |

((a) of the unit instructions; (b) the entry equal to the previous instruction's; (c) the
instructions where a sticky entry would change, a loop's first instruction and the one after its
body counted whenever their entry is nonzero; p_q3's entries computed for its run's arguments,
an older dump; the programs as perf_qwen, moe_prefill_cosim and the pass's cost
measurement (section 6) build them, not simulated.)
The IMEM bits of each encoding against the dense sidecar's 59 an instruction:

| encoding | bits against dense | instructions added |
|---|---|---|
| explicit: a 256-bit SYNC before each instruction that waits | 3.7-4.1x | +85-95% |
| delta: a 256-bit SYNC where the entry changes | 3.3-4.0x | +75-92% |
| delta as a 64-bit prefix word (a variable-length IMEM) | 0.82-0.99x | +75-92% prefixes |
| a table of the program's entries, an index per instruction (8-10 bits) | 0.31-0.69x | |
| two tables (the counts' words, the masks), two indices (11-16 bits) | 0.25-0.51x | |

The entries are not sparse: 85-98% of the instructions wait on something and 76-92% wait on
something else than the instruction before, because the counts and the mask are relative to
each unit's latest instructions, which move with every instruction. **Explicit and delta
encodings are out**: a SYNC op costs 3.3-4.1x the dense bits and nearly doubles the
instructions fetched and dispatched; a short prefix where the entry changes is the dense bits
again. A dictionary halves to quarters the bits, but its tables grow with the program (the
35B's decode: 690-751 entries, 229-232 count words, 182-231 masks), a lookup joins the fetch,
and programs sharing an IMEM share a table only as their union (1,604 distinct entries over
these six programs); a full table needs the compiler to merge entries into stronger ones
(each count the younger, the masks OR'd: conservative, cycles).

**Recommendation: dense entries, sized by the instruction store rather than the program.**
With a resident IMEM (the card's), the dense sidecar is +23% of IMEM's bits; a two-table
dictionary is the fallback if IMEM must grow past ~3K instructions. With a streamed
instruction buffer (programs in DRAM, a small on-chip buffer that the sequencer refills), the
entries stream with their instructions and the sidecar is 59 bits per buffer slot, a fixed
cost however long the program: a 512-instruction buffer holds 30 Kbit of entries. Streaming
them costs +25% of the instruction bytes (64 bits on 256), ~0.14 MB a token for the 35B's
decode (~17.5K instructions dispatched a token: its 1,750-instruction body of four layers 10
times) against ~1.7 GB of weights a token (3B active parameters at fp4, estimated), under
0.01% of DRAM's bytes. That is the ASIC's form: the
hazard check's logic gone, and the waits' metadata scaling with the instruction buffer.

## 5. The VPU's composite path at 4 and 2 lanes (simulated)

The only part of the VPU that is not DSTEP's datapath is the composite path (`se_comp`, section
2.2). `+vt_cw=W` runs the composite functions W lanes wide at today's latency (the elementwise
ops, the reductions, DSTEP and the stream engine unchanged); cycles against `i0`:

| workload | i0 cycles | 4 lanes (`v_cw4`) | 2 lanes (`v_cw2`) |
|---|---|---|---|
| d_lf | 1,339,842 | +1.33% | +4.31% |
| d_q3 | 3,623,644 | +0.82% | +5.38% |
| d_q35 | 4,760,853 | +0.83% | +4.79% |
| dr7_q3 | 3,697,238 | +1.60% | +8.73% |
| dr_q3 | 3,694,791 | +1.68% | +8.79% |
| p8_q3 | 502,925 | +3.62% | +25.31% |
| p8_q35 | 942,492 | +8.93% | +24.65% |
| p_lf | 1,630,055 | +3.75% | +14.49% |
| p_q3 | 372,383 | +2.15% | +17.10% |
| p_q35 | 1,137,275 | +5.74% | +18.94% |
| z_ph_r3 | 1,677,624 | +1.43% | +4.99% |
| z_q4_r2 | 2,225,099 | +0.25% | +1.65% |
| b4_q3 | 1,671,279 | +0.50% | +3.71% |
| b8_q3 | 1,827,502 | +0.99% | +6.96% |
| l_g26#0,1 | 969,765 | +0.02% | +0.10% |
| l_g26#0,5 | 993,612 | +0.02% | +0.05% |
| l_q35m#1 | 382,905 | +1.09% | +8.52% |
| l_q35m#4 | 296,391 | +1.94% | +6.97% |
| m_g26#0 | 758,072 | +0.01% | -0.01% |
| m_g26#0,1 | 1,514,737 | +0.01% | +0.01% |
| m_g26#0,5 | 1,613,257 | +0.00% | +0.01% |
| m_q35m#1 | 4,579,588 | +0.00% | +0.03% |
| m_q35m#3 | 5,227,461 | +0.04% | +0.03% |
| m_q35m#4 | 5,490,770 | +0.03% | +0.13% |

- **4 lanes** (-2.9K LUT, -1.1K LUT RAM, -3.3K FF standalone; the VPU's hierarchy adds operand
  muxes and delay lines that scale too) costs **+8.93% on Qwen3.5's prefill at MCOLS 8**, +5.74%
  at MCOLS 4, +2.2-3.8% on Qwen3's and LFM2's prefill, +0.8-1.7% on decode, +0.5-1.0% on
  batched decode, +1.1-1.9% on the 35B's layer runs: the composites (the softmax's EXP2SUB, the
  norms' RSQRT, RECIP) sit between MMs that wait for them, and Qwen3.5's gates and DeltaNet
  normalizations add composites per head.
- **2 lanes** costs +14-25% on the 4-row prefill runs, +1.7-5% on the zero-image layers,
  +4.3-5.4% on decode (+8.7-8.8% on the resident decode), +3.7-7.0% on batched decode.
- The MoE decode rows and the 26B's layer runs do not notice (at most +0.13% at 2 lanes): their
  time is the DMA's.

Which work the composite path does (the data-movement study's ledger: the ISA simulator with
the VPU's issue model, the MX build, zero images; share of the VPU's and stream engine's issue
cycles, the composite passes' part in brackets): Qwen3-0.6B's decode at position 544 spends
40.1% (27.6%) in the softmax's P and 28.8% (16.0%) in the GLU, at 4096 80.7% (59.8%) in the
softmax's P; SmolLM3's decode at 4096 68.2% (50.6%) in the softmax's P; LFM2.5's 4-row prefill
41.3% (23.0%) in the GLU and 29.5% (20.4%) in the softmax's P; Qwen3.5-0.8B's decode 57.5% in
DSTEP. The composite cycles are almost all the softmax's EXP2SUB and the GLU's SiLU (EXP2 and
RECIP), and the softmax's grow with the context: the decode runs above (position 544) understate
what a narrower path would cost at 4K-token contexts and beyond.

4 lanes would free 2.9K of the VPU's 30.8K LUTs (standalone; perhaps ~5K with the hierarchy's
muxes, estimated) for up to 9% of prefill. **Keep 8 lanes.** The
first-principles lever on the VPU is not its width but its passes over TMEM: each VOP reads a
tile from TMEM and writes it back, and the softmax, the norms and the gates are each such a
pass behind an MM. Work folded into the MXU's output path (as `RMAX` and `ASCALE` already are)
removes the pass and the TMEM traffic; that is the data-movement study's subject, and the VPU's
width should be revisited after it, against what is left.

What is left, by that study's fused design (docs/dataflow.md on its branch; the same ledger,
simulated, not built): the GLU, the gates, the norms, RoPE, the attention's output scaling and
the softmax's numerators move to the quantizer's input, with the composite unit (MCOLS / 2 lanes,
3 stages) moved there; the residual to the MXU's drain; DeltaNet's convolution, L2 norms and
SiLU to the stream engine's input. The VPU keeps 1-5% of today's issue cycles on Qwen3, Qwen3.5,
SmolLM3 and LFM2.5 and 11% on Gemma 4 E2B (its sandwich norm), with at most ~1% in composites
(flash attention's row scalars), plus sampling in batched steps (5.4% at 8 users); DSTEP stays on
the stream engine. In its co-simulation at MCOLS 4 a VPU that costs nothing gains nothing over
the fold (Qwen3-0.6B -3.1% / -7.2% for a row / 4 rows either way). So the composite path's width
is a question for the fold's composite unit, not for the VPU: with the fold, the VPU's own
composite path can go to 2 lanes or 1 at no measurable cost; without it, it stays at 8.

## 6. WAITC in production (EXPERIMENTAL, off by default)

On the branch `waitc-mx` (the MX freeze merged under it, every target `SEQ_WC = false`):
- `SEQ_WC` is a Config field and an `[arch]` key of every target, a parameter through
  `otpu_board` to the slices, and `CFG_UARCH` bit 24. `arch_hash` holds it only when on, so no
  existing build's hash changes and a host that writes no entries refuses a WAITC bitstream.
- The host loads every program with its entries on a WAITC bitstream (`Board.load_program`,
  `Board.load`, the chain slots via `generate.slot_words`, the diagnostics' images) and refuses
  one without them; on a bitstream without the bit nothing changes. The program area, the
  chain slots, the snapshot and the KV staging take the entries' room (`waits.area`: 5/4 of
  IMEM's bytes). `WIN > 16` with `SEQ_WC` is an error in the targets, the RTL simulation and the
  host.
- The entries are computed once per program (`waits.entries`, keyed by the words, the
  arguments' bounds, `D`, `S` and the compiler's source) and kept in memory and beside the
  program cache's programs on the disk; the compile workers compute them with the programs
  (`qwen3.worker_words`, `compiler.Words` carries them), the generate chain's buckets in
  parallel (`waits.entries_many`), and prebuild computes those a card session loads first.

**The pass's cost** (measured on a laptop, one core; the compile itself is 10-40 ms a program):

| program (buckets b1, b8) | instructions | cold (several runs) |
|---|---|---|
| Qwen3.5-35B-A3B decode | 1,893 / 2,235 | 4.2-4.9 s |
| Qwen3.5-35B-A3B generate (the chained program) | 1,948 / 2,290 | 8.0-9.1 s |
| Phi-4-mini decode | | 0.3 / 0.8 s |
| Phi-4-mini generate | | 0.9 / 2.8 s |

Once computed, an entry comes back from memory in under 0.1 ms and from the disk in under 1 ms.

With the final mask (31 VOPs of history, 38a6342b) the pass takes the same abstract steps and
each costs more (the simulation host, one core, both trees in one run, buckets 1 / 8):

| program | 15 VOPs (e274fc4a) | 31 VOPs (38a6342b) | |
|---|---|---|---|
| Qwen3.5-35B-A3B decode | 3.83 / 3.72 s | 4.26 / 4.24 s | +11% / +14% |
| Qwen3.5-35B-A3B generate | 7.55 / 7.73 s | 8.37 / 8.79 s | +11% / +14% |
| Phi-4-mini decode | 0.28 / 0.72 s | 0.36 / 0.92 s | +27% / +28% |
| Phi-4-mini generate | 0.80 / 2.45 s | 0.99 / 2.74 s | +25% / +12% |

Still off the critical path (the workers, prebuild and the caches); the instructions that wait
on something go from 92.1% to 94.6% of the 35B's decode (the mask names VOPs further back).

The 35B's cost is one loop: the outer loop over its layers (count 10, a body of 1,750
instructions) holds 95% of the abstract steps, and each iteration advances R1, R2 and R4-R7, so
the state never repeats. What was tried, entries bit-identical in every case: an exact
steady-state shortcut (a loop's state equal to one p <= 8 iterations back: 0 steps saved), a
prefilter on each footprint's hull (-5-10% time), the conflict test split by space (-5-15%), a
smaller cap on the states a loop keeps (4 -> 2: up to -15%). None reaches the ~1 s goal, so the
pass stays as it is, off the critical path (the workers, prebuild and the caches above).

- **If card sessions ever show the cost** (a cold 35B session's first tokens): the pass's
  inner loop (the abstract step and the conflict test) in C through ctypes, with a test that its
  entries are bit-identical to the Python pass's on the workloads' programs.
- **An idea, not tried**: an affine abstraction, a register as base + k x stride in the loop's
  counter, makes the state periodic modulo the induction variable, and stays exact when the
  footprints are affine in it (the weight streams, the K/V blocks, the tiles): one symbolic
  iteration per loop instead of `count`.

**MX freeze with WAITC**: job V's runs (section 4.3's final table) are on the merged tree
(`waitc-mx`): `i0` and `wcm`, the 12 perf runs bit-exact against the ISA simulator, mean +0.06%.

## 7. The trace's share of the sequencer (measured)

One trace field takes **1.8-2.2K LUT and 532 FF**, 30-38% of v2's LUTs: the cycle each started
instruction became ready (`seq_ev_t.s_rdy`, the trace's `R` record, `otpu_trace`'s wait time
`s_dt`). It is a 32-bit stamp per slot (`srdy_c`, 512 FF) and, per unit, its read through the
started slot (`ucs`, the event's select). v2's cells grouped by name (placed and routed) put
2,243 LUT there (`srdy_c` 624 with 128 CARRY4, `ucs` 797, the select 822); today's put 1,786
(`srdy_c` 84, `ucs` 882, the select 820), so part of v2's `srdy_c` group may be the ready test
that the starts need anyway, and the lower figure is the safe one. With
WAITC it is redundant: an instruction is ready when its waits are met, the latest of the
completions its entry names (the trace's `E` records) and its dispatch (`D`), all of which the
trace records and the host has the entries for. Deriving it on the host (`opentpu/hwtrace.py`)
and dropping the stamp from WAITC builds would bring v2 to ~3.7-4.2K LUT and ~4.9K FF
(estimated), 26-29% of today's sequencer's LUTs. Not done: it changes the trace format.

## 8. Summary and recommendations

| option | sequencer / VPU area (measured, OOC, K7) | cycles (simulated, against today) | verdict |
|---|---|---|---|
| today's dynamic hazard check | 14,375 LUT, 10,729 FF, 7 DSP, 5,429 slices, 28.5 RAMB36 | | the worst path of the banked-TMEM and MX builds |
| (b) a cheaper check (`i0g8`) | est. -6K LUT at most | **+0.9% to +61%** | no |
| (a) v1: counts, every unit in order | 5,609 LUT, 5,245 FF, 0 DSP, 32.5 RAMB36 | +0.0% to +8.7% (prefill +1.2-8.7%) | superseded |
| (a) v2 with a 15-bit VPU mask | 5,944 LUT, 5,402 FF, 0 DSP, 2,410 slices, 33.5 RAMB36 | -0.38% to +2.15%, mean +0.25% | superseded (the hold) |
| **(a) v2: counts + 31-bit VPU mask (WAITC, 38a6342b)** | **6,797 LUT, 5,722 FF, 0 DSP, 2,682 slices, 35.5 RAMB36** (+7: per-instruction metadata) | -0.60% to +1.52%, mean +0.06% (+0.15% per MoE prefix) | **adopt** |
| v2 without the ready stamp | est. ~3.7-4.2K LUT, ~4.9K FF | none (the host derives it) | follow-up |
| VPU composite path at 4 lanes | -2.9K LUT, -1.1K LUT RAM, -3.3K FF (standalone) | +0.0% to +8.9% | no |
| VPU composite path at 2 lanes | -4.4K LUT, -1.6K LUT RAM, -5.0K FF (standalone) | 0.0% to +25% | no |

**Where the design is bottlenecked**: neither unit sets the throughput. The MXU is 94-100% busy
in every workload but Qwen3.5's prefill: decode and the MoE runs at the DRAM's rate (the MXU
streams the weights), prefill and batched decode at the MXU's compute (section 3). The
exception is Qwen3.5's prefill, where the VPU's DSTEP streams hold the MXU at 56-74%. The
sequencer costs area and timing, not cycles; the VPU's area is 61% DSTEP's datapath, which
Qwen3.5 needs whole.

**The TMEM-like change** is the sequencer's: what the hazard check finds at run time the
compiler knows at compile time (the programs are static, their run-time arguments bounded).
Moving it there frees 53% of the sequencer's LUTs, 47% of its flip-flops, all its DSPs and 51%
of its slices for +0.06% of the cycles; 3.7% of the K7's slices, where the TMEM study
freed a third of its block RAM. What it costs instead is instruction metadata (59 bits an
instruction, read once per dispatch), not data movement.

Per board:
- **YPCB-00338** (xc7k480t, one slice): WAITC v2 in the MCOLS 8 build (adopted once the MX
  gates and the remaining four workloads pass; section 6). The VPU stays at 8 lanes. The ready
  stamp (section 7) is the next 1.8-2.2K LUT if the build needs them.
- **BCU1525** (xcvu9p, two slices): the same RTL and compiler, twice the saving (-18.5K LUT
  with the 15-bit mask on the xcku5p, the final's not run; -14 DSP, +14 RAMB36). The device has room; the reason is timing: at 250 MHz on the
  UltraScale+ stand-in the footprint path is today's worst (+0.027 ns) and v2's worst is the
  wait test (+0.068 ns with the 15-bit mask; the final's not run). Take it with the YPCB's validation.
- **F2** (VU47P, UltraScale+ with HBM; no target yet): v2 as on the BCU1525. HBM raises the DRAM
  rate that bounds decode by an order of magnitude or more, so the MXU's and the VPU's share of
  decode grows (and the softmax's composites with the context): rerun section 5's
  composite-width runs at HBM's rate and at long contexts before any VPU change; today's answer
  (8 lanes) is for DDR3/DDR4, and those two only strengthen it.
- **ASIC** (nangate45, docs/asic.md): explicit waits, no hazard check at all. The case is timing
  and scaling more than area: the sequencer's logic is 0.224 mm² of the card's 10.7 mm²
  (synthesis, measured), of which the check is ~0.15 mm² (estimated by the FPGA's ratio), and its
  dispatch footprint (`q_pr -> c_fp`, 50 levels) is nangate45-small's worst register-to-register
  path. What the sidecar trades, per slice (the card's WIN 16, 64K-word TMEM, 4,096-instruction
  IMEM; from the RTL, areas estimated):

  | | the hazard check (removed) | WAITC (added) |
  |---|---|---|
  | storage | per window slot its footprint (`sfp`: two 64-bit DRAM ranges, five 33-34-bit TMEM ranges, the ACT range, 322 bits) and two dependency rows (32 bits): **354 flops a slot, 5.7K** (the ablation: 5,643 FF) | **59 SRAM bits an instruction** (7.4 per 32-bit IMEM word, +23% of IMEM's bits): 236 Kbit; per slot 7 8-bit targets, a 32-bit pending set and its index, ~100 flops (1.6K) |
  | comparators | at each dispatch, against each slot, 28 range comparators (4 DRAM pairs: 8 of 32 bits; 9 TMEM and 1 ACT pair: 20 of 17 bits): **448**, plus the footprint's multiplications and range arithmetic (7 DSPs on the FPGA) | per slot 7 8-bit counter compares and a 32-bit AND: 112 small compares |
  | worst path | the footprint into the compares, 50 levels | the wait test into a unit's command copy |
  | area, nangate45 (est.) | ~0.15 mm² | ~0.20 mm² of SRAM at IMEM's density (0.858 mm² a Mbit), ~0.01 mm² of flops |

  How each scales:
  - **more slices**: both are per slice (each slice's sequencer has its own window and IMEM), so
    S slices cost S times either; the trade per slice does not change.
  - **a deeper IMEM**: the sidecar grows by 59 bits an instruction, the check not at all. At the
    estimates above they break even near 3K instructions; at 8K the sidecar is ~0.39 mm², ~2.6x
    the check. The longest program measured here (the 35B's chained generate) is 2,290 instructions.
  - **a wider window**: the check grows linearly in its slots (354 flops and 28 comparators each)
    and quadratically in its dependency rows (2 x WIN² bits); the entries grow by a bit per
    count (7 bits an instruction) per doubling, the waits' slots by ~100 flops each.
  - **a larger TMEM or DRAM, more ranges per instruction, more units**: the check widens every
    stored range and comparator (TMEM: 2 bits a range per doubling) and adds a comparator per
    slot for each new range pair; the entry adds 4 bits per in-order unit and nothing for the
    memories.

  So on the ASIC the waits cost SRAM bits that grow with the program, where the check costs
  logic and a deep path that grow with the machine. The entries do not fit today's 256-bit
  encoding: MM and QST use all seven immediate words (MM in 59-79% of its instances) and VOP
  often six, so 33-43% of the instructions have fewer than two zero words (the resident decode
  and the 4-row prefill, `dr_q3` and `p_q3`, measured); nor are they narrow (counts of 5-15 and
  mask ranks of 17-31 occur in both programs; counts within 7 and the mask within 16 cover 81-85%
  of the instructions). They stay 59 bits beside the instruction, as a narrow SRAM read at the
  same address as IMEM (or an instruction word that much wider: the same bits), and they are
  not sparse either (section 4.5: 85-98% of the instructions wait, 76-92% on something else
  than the instruction before; a SYNC op or a delta encoding costs 3.3-4.1x the dense bits). The
  ASIC form: dense entries in a streamed instruction buffer, so the metadata scales with the
  buffer and not with the program; a two-table dictionary (0.25-0.51x) only if a resident IMEM
  must grow.
  The VPU's width follows the data-movement study's fold (section 5): with the fold, the
  composites (and their unit) move to the quantizer's input, the VPU keeps 1-11% of its work and
  its own composite path can be 1-2 lanes; without it, 8. The stream engine's DSTEP datapath
  stays either way.

## 9. Multi-row DSTEP: the DeltaNet state once per prompt run (simulated)

The lead's follow-up (2026-10-12, from the data-movement study's ledger): a Qwen3.5 prefill row
moves each DeltaNet head's fp32 state from DRAM through the stream engine and back, once per
row: 37.7 MB a row for Qwen3.5-0.8B and 2B (34% and 17% of a row's DRAM bytes), 100.7 MB for
4B, 125.8 MB for the 35B (from the shapes: layers x heads x 128 x 128 x 4 B, read and
written). The recurrence is per state row, so a run of R rows can step each state row through
the R tokens while it is on chip, bit for bit. Designs (docs/dstep_mr.md 3): (A) one engine,
a block of state rows circulating for T tokens (DRAM bytes / T); (B) P engines in a chain,
each stepping the next token (datapath x P); (C) both. (A) and (C) are built on branch
`dstep-mr` (STREAM_T, STREAM_P; RTL tests against the ISA simulator word for word).

The bound (RTL co-simulation, the card's configuration: LiteDRAM at DDR3-1066, 133.33 MHz,
MCOLS 4; the first 4 layers, one prompt run of R rows, R the longest the 64K-word TMEM
compiles; `mr8`: (A)'s DRAM bytes on today's engine; `st4`: (A) as built at STREAM_T 4;
`mr16`: two engines' datapath, the state once; `st4p2`: (C) as built; `mrfree`: one DSTEP a
run; `nose`: no DSTEP, the floor; `#n` a 35B layer prefix, #4 an attention layer):

| case | R | today | `mr8` | `st4` | `mr16` | `st4p2` | `mrfree` | `nose` |
|---|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | 4 | 1,141,254 | -0.1% | -2.0% | -19.0% |  | -28.4% | -30.1% |
| Qwen3.5-2B | 4 | 2,093,091 | -0.8% | -1.8% | -7.9% |  | -7.9% | -8.4% |
| Qwen3.5-4B | 2 | 2,251,563 | -5.0% | -3.7% | -5.0% |  | -5.0% | -10.2% |
| Qwen3.5-35B-A3B #1 | 2 | 388,844 | -3.5% | -3.5% | -10.2% |  | -10.2% | -20.1% |
| Qwen3.5-35B-A3B #4 | 2 | 302,998 | +0.0% | +0.0% | +0.0% |  | +0.0% | +0.0% |

At MCOLS 8 and 16 (4 rows; "of the floor": what two and four engines take of `nose`'s saving):

| model | MCOLS | today | `mr8` | `st4` | `mr16` | `st4p2` | `mrfree` | `nose` | P = 2 of the floor | P = 4 of the floor |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | 4 | 1,141,254 | -0.1% | -2.0% | -19.0% |  | -28.4% | -30.1% | 63% | 95% |
| Qwen3.5-0.8B | 8 | 952,135 |  |  | -22.9% |  | -34.2% | -45.1% | 51% | 76% |
| Qwen3.5-0.8B | 16 | 952,353 |  |  | -22.9% |  | -34.2% | -45.0% | 51% | 76% |
| Qwen3.5-2B | 4 | 2,093,091 | -0.8% | -1.8% | -7.9% |  | -7.9% | -8.4% | 93% | 94% |
| Qwen3.5-2B | 8 | 1,476,449 |  |  | -14.8% |  | -20.8% | -24.4% | 61% | 85% |
| Qwen3.5-2B | 16 | 1,476,822 |  |  | -14.8% |  | -20.7% | -24.3% | 61% | 85% |

- **The state step is engine-bound on the card**: 8 words a cycle are 64 B a cycle of state
  traffic, half of DDR3-1066's 128 B, so the bytes alone (`mr8`, (A)) give 0.8B nothing; on 4B
  and the 35B, whose runs are DRAM-bound, they give -3.5% to -5.0%. (A) cuts the state's DRAM
  bytes by R whatever the cycles: the data-movement half, the one that counts where compute
  outruns DRAM.
- **Cycles need datapath, and more of it as the MXU widens**: the floor grows from 30% / 8% of
  the run (0.8B / 2B) at MCOLS 4 to 45% / 24% at MCOLS 8 (MCOLS 16 = 8 at R = 4: four rows fill
  8 columns with PAIR, and 8 rows do not fit the TMEM). Two engines take 51-93% of it, four
  76-95%. The rule from the shapes: P = MCOLS / 2 chained engines (the MXU's rows a pass) hold the
  step at 32 Z / W of the MXU's time (Z the state words, W the weight MACs of a row: 0.30 for
  0.8B, 0.11 for 2B and 4B, 0.21 for the 35B; estimated).
- **Area** (docs/dstep_mr.md 6): an engine is the DSTEP datapath of section 2.2, ~20.6K LUT and
  86 DSP on the FPGA, ~0.7 mm² on nangate45 (estimated; the OOC of one engine against two was
  not run): compute, not a copy of SRAM.
- **Decision** (2026-10-12; docs/dstep_mr.md 8, 9): the chain (C) is parked, its
  bit-exact RTL a card-only option; the ASIC's prefill state step goes to the MXU, the chunkwise
  delta rule (D) with its products in three int8 MMs, in the release after r2 (a first
  co-simulation at C = 4 puts it at 1.8-2.4x today's run on the card, bound by the quantizer: it
  needs runs of C >= 16 and the bound still to run). Decode and short runs keep today's single
  engine; (A) at STREAM_T 4 is built and exact.
- **One streaming engine** (attention decode on a chain, docs/dstep_mr.md 5) stays a separate
  question, with its numerics decision (the online softmax) first.
