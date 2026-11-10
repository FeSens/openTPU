# The VPU: where it binds, and whether pumping or widening it pays

Question (the user, 2026-10-11): prefill, multi-user and MoE now count as much as decode. Where
does the vector unit (VPU) bind in each, and does it pay to run its floating-point units at twice
the core clock ("pumping") or to widen it? This page answers with:
- the RTL co-simulation, using timing-only program transforms as upper bounds before any RTL;
- an out-of-context Vivado experiment: a pumped fp32 multiply-add, bit-exact with today's;
- out-of-context runs of `otpu_vpu` at 8 and 16 lanes;
- a recommendation per board.

Every number says whether it is measured (Vivado), simulated (the RTL co-sim) or estimated.

**Short answer: no, not on the YPCB. Pumping works, and the VPU is not what binds.**
- **Decode, prefill, multi-user and MoE.** On the models the card runs, an infinitely fast VPU
  would save:
  - at most 0.8% of a decode token (2.4% at 4K context);
  - 1-6.5% of a prefill run at today's MCOLS 4;
  - 2-3% of a batched step;
  - 2-6% of a MoE layer, with every expert resident.

  A VPU of twice the throughput saves about half of each figure.
- **With a pumped MXU.** At MCOLS 8 (dsp-pump's rate) the VPU's share grows. 2x saves 2.6-7%
  of prefill and 4.7% of an 8-user step.
- **The larger vector lever is the stream engine's DSTEP.** It matters for the Qwen3.5 family's
  prefill: removing its time entirely saves 9-10% at MCOLS 4 and 24% at MCOLS 8 (2B). Doubling
  its datapath saves at most ~5-6% and ~12%.
- **The pumped unit works.** A pumped fp32 multiply-add closes at 266.67 MHz on the card's part,
  is bit-exact, and has today's latency, once its two long stages are split by hand. But it
  saves only the units: 16 DSPs and ~1.7K LUT for every 8 lanes. A 2x VPU's cost is in
  everything around them, and the units are ~14% of the VPU's LUTs. The 16-word TMEM ports
  alone are tmem-pump's subject.
- **The VPU as a whole does not run at 266.67 MHz:** -1.39 ns out of context.

Do not build a 2x VPU or a pumped VPU into the YPCB. If Qwen3.5 prefill matters once the MXU has
MCOLS 8's rate, the vector-side change worth designing is a 16-word DSTEP datapath with a
multi-row DSTEP; it needs no TMEM bandwidth. Section 7 gives the recommendation per board.

## 1. The VPU today

`rtl/vpu/otpu_vpu.sv` on the board: LANES 8, the stream engine (HAS_SE), the write buffer (WBUF),
at the core clock (133.33 MHz on the YPCB). Rates per instruction class (docs/stream.md 11):

| class | functions | words a cycle | latency (cycles) | units |
|---|---|---|---|---|
| elementwise | ADD, SUB, RSUB, MUL, OUTER | 8 | 7 | slot 0 of each lane: `otpu_fmma`, y = a·b + c·e (two `otpu_fmul`, LM 2, and an `otpu_fadd`, LA 4) |
| moves | MAX, MIN, COPY, ABS, FILL | 8 | 1 | fabric |
| row sums | RSUM, RSSQ, RDOT | 8 (a row padded to 64 columns) | 7 + the tree | slot 0 in the isum_64 partial loop, then `otpu_vtree`; run alone |
| row maxima | RMAX, ARGMAX | 8 | the tree | fabric |
| composites | EXP2, EXP2SUB / RECIP / RSQRT, LOG2 | 2.67 / 4 / 2 | 75 / 50 / 100 | `otpu_se_comp`: each chunk loops 3 / 2 / 4 times through slot 0, the tail's U and Q (25 cycles a pass) |
| stream engine | DSTEP, STREAM | 8 state words | ~200 to fill | slot 0 and the tail (d, U, Q); VOPs wait while a stream holds it |

A DeltaNet head's DSTEP (a 128 x 128 state) takes 2,048 datapath cycles and ~200 of fill,
against 1,024 port-B chunks of state traffic (docs/prefill.md 8). It is bound by its datapath.
Its state streams from DRAM through the DMA, not through TMEM.

**Area.** Out of context at 7.5 ns (measured here, section 6): `otpu_vpu` as built is 30,802 LUT,
27,415 FF, 86 DSP48E1 and 12,341 slices. Routed in the full design it is 38,700 LUT
(docs/stream.md 12.3). The production image is 176,357 LUT (59.1%), 60,694 slices (81.3%) and
713 DSPs (37%) (e698dcd, docs/board.md). The slices are short; the DSPs are not.

**What maps to DSPs.** Only the 24 x 24-bit significand products. Each one is two DSP48E1 (25 x 18
signed, so 24 x 17 unsigned per DSP), with the product registered in the DSP. `otpu_fmma` has two
products, so 4 DSPs. The 86 are:
- slot 0: 8 x 4 = 32;
- the tail's U: 32;
- the tail's Q, one product: 8 x 2 = 16;
- the d stage: 6.

Everything else is fabric:
- the multipliers' exponent and rounding stage;
- the whole adder: alignment, the 28-bit add, the normalizing shift, rounding (four stages);
- the row-sum trees, the composites' operand muxes, state and control, the write buffer.

So a pumped DSP in the VPU means a pumped multiply-add, most of which is fabric. The MXU is
different: one DSP48E1 per integer MAC (docs/dsp_pumping.md). The 8 slot-0 units are 4,378 LUT
and 32 DSPs out of context (section 5): 14% of the VPU's LUTs.

**TMEM.** The VPU reads operands A and B on two read ports (VA, VB), 8 words a cycle each from 8
banks, and writes 8 words a cycle; on the FPGA each read port has its own copy of the banks. A VPU
of twice the element rate needs 16 words a cycle on VA, VB and the write side. That means 16
banks, or banks pumped at 2x (tmem-pump's subject). The composites do not need it: they consume at
most 4 words a cycle (RECIP), so a faster composite path fits today's 8-word ports. DSTEP does not
need it either.

## 2. Method

**The co-simulation.** `tools/perf_qwen.py` runs the RTL (Verilator, the board's
micro-architecture) with LiteDRAM's controller co-simulated (`--ldc`): DDR3-1066, a 133.33 MHz
core, and the bitstream's features (`OTPU_PAIR=1 OTPU_DSTEP=1 OTPU_STREAM=1`). Its decode tokens
are within 0.8-1.4% of the card's (docs/litedram.md 11). Every run here uses fp4 weights and an
int8 LM head at position 544 (or 4000 where marked), on tree `vpu-pump` (main 23b68089 plus this
page's tools). Most ran on the simulation host, the first three on a laptop.
- **Whole small models, real weights:** Qwen3-0.6B, LFM2.5-230M, Qwen3.5-0.8B.
  - decode: one token (`perf_qwen --layers 0`);
  - prefill: a 4-row run (the card's R_max), no logits (`--rows 4 --logits none`);
  - multi-user: a batched decode step, B slots with the picks on the device (`--batch B`, new:
    `opentpu/llm/batch.py`'s step as otpu-serve runs it). Only Qwen3 and the Llama-likes
    batch; LFM2, Qwen3.5 and Gemma 4 images hold one sequence.
- **The card's larger models, zero images** (`tools/vpu_hack.py cosim`, new; the timing does
  not depend on the data, so only config.json is read): a layer prefix without the LM head.
  - Qwen3.5-2B and 4B: 4 layers (one period of 3 DeltaNet and 1 attention).
  - SmolLM3-3B and Phi-4-mini: 2 layers. LFM2-2.6B: 6 layers. Gemma 4 E2B: layers 0-4.
  - "R1" is a one-row run without logits: decode's layers without the LM head.
  - The batched step has the LM head; its per-layer cost is the 2- and 4-layer runs'
    difference.
- **MoE:** `tools/moe_prefill_cosim.py` on zero images, with every expert resident at its
  layer's first slot (so no host link: on the card the link adds time, which lowers the VPU's
  share further).
  - A decode row through the MoE block (`rows --variant moe`), a layer's cost as a prefix
    difference.
  - The layer-major prefill's program of one layer at 2 rows (`layer`; R x k <= 16 expert ids
    caps it at 2, the card's LAYER_MAJOR).
- **MCOLS 8:** dsp-pump's pumped-MXU rate (docs/dsp_pumping.md 3), where an 8-user step runs as
  one. Today's MCOLS 4 runs it as two steps of 4.

**The upper bounds: timing-only transforms** (`tools/vpu_hack.py`, new; `perf_qwen --hack`,
`moe_prefill_cosim --hack` and `vpu_hack.py cosim --hack` apply them to the compiled programs).
They change only the cycles; the results are wrong afterwards, as with
`tools/decode_gaps.py --drop`.

| mode | what it models |
|---|---|
| `novop` | every VOP a NOP: an infinitely fast VPU, its whole exposed share |
| `half` | every VOP's element work halved (columns, or the rows of a row sum of at most 64 columns): a VPU of twice the throughput with today's latencies (16 lanes, or 8 lanes pumped at 2x) |
| `comp1` | each composite as a one-slot elementwise op of its shape: composites at 8 words a cycle and 7 cycles (the bound of any composite speed-up) |
| `nose` | every DSTEP / STREAM a NOP: an infinitely fast stream engine, and also no state traffic |
| `halfse` | each DSTEP's rows halved: twice the datapath rate, and also half the state traffic, which a faster datapath alone would not give (it overstates) |

The module also has `nocomp`, `halfcomp` and `halfew` (the composites removed or halved, the
other VOPs halved), not used on this page.

`moe_prefill_cosim` keeps the routing's own small VOPs exact (`vpu_hack.moe_keep`): their
values become RLD addresses and loop counts, so a transformed MoE run still runs k experts.

**The static inventory** (`python3 tools/vpu_hack.py MODEL KIND N POS [MCOLS]`, compile only)
takes each instruction's issue cycles at the rates above, times its loop counts, and sets them
against the port-B chunks (the memory stream's cycles). It shows the work before any overlap;
the co-sim shows what the overlap leaves.

**A real 16-lane VPU**, for comparison with `half`, uses the existing knob `OTPU_LANES=16
OTPU_ULANES=8`:
- `otpu_vpu` at LANES 16 on 16 TMEM banks, with the MXU and the quantizer on 8 of them;
- no stream engine (it needs 8 lanes), so DSTEP and STREAM are illegal there
  (`otpu_slice.sv`: HAS_SS); composites run on CL = 4 long lanes;
- Qwen3 only, with `--check` (bit-exact against the ISA simulator).

## 3. Where the VPU binds

### 3.1 The work (static inventory)

VPU issue cycles against the port-B chunks of the same program (whole model, fp4, int8 head,
position 544 unless marked), before any overlap:

| model | decode | prefill, 4 rows | batched, 4 users | decode at 4,000 |
|---|---|---|---|---|
| Qwen3-0.6B | 10.4% (composites 4.7%) | 47.7% (21.3%) | 35.2% (14.9%) | 23.9% (15.5%) |
| LFM2.5-230M | 8.1% (3.6%) | 48.3% (21.2%) | | |
| Qwen3.5-0.8B | 10.1% (3.9%) + DSTEP 13.7% | 52.9% (20.1%) + DSTEP 72.0% | | |
| Qwen3.5-2B | 5.4% (2.1%) + DSTEP 6.1% | 30.9% (12.1%) + DSTEP 35.2% | | |
| Qwen3.5-4B | 5.2% (2.1%) + DSTEP 7.9% | (TMEM: 2 rows) 13.1% + DSTEP 20.0% (4 layers) | | |
| SmolLM3-3B | 5.4% (2.8%) | 24.5% (12.6%) | 21.4% (10.7%) | 12.2% (7.9%) |
| LFM2-2.6B | 5.1% (2.4%) | (TMEM: 3 rows) 17.4% (6 layers) | | |
| Gemma 4 E2B | 7.6% (2.6%) | 40.0% (13.5%) | | |
| Phi-4-mini | 3.7% (1.8%) | (TMEM: 3 rows) 14.4% (2 layers) | | 10.4% (6.8%) |

- Vector work grows with the rows; weights stream once a pass, so prefill and batching raise
  the VPU's share.
- The small models overstate it: vector work scales with the hidden size, weights with its
  square.
- The largest single items are EXP2SUB (the softmax), MUL, and the SiLU's EXP2 and RECIP.
- DSTEP holds the stream engine, so the VOPs wait: for Qwen3.5 the DSTEP and VOP shares add.
- MoE (Qwen3.5-35B-A3B's layer-major prefill at 2 rows): a DeltaNet layer is 28% VOPs + 60%
  DSTEP; Gemma 4 26B-A4B's layers 11-13%.

### 3.2 What the co-sim exposes

Change in cycles (simulated). novop is the VPU's whole exposed share, half is what twice the
throughput gives, and comp1 bounds the composites.

**The card's models, layers on zero images:**

| model | case | MCOLS | cycles | novop | half | comp1 | nose | halfse | half + halfse |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-2B, 4 layers | R1 (decode) | 4 | 1,071,522 | -0.8% | | | -5.4% | | -2.8% |
| Qwen3.5-4B, 4 layers | R1 (decode) | 4 | 2,082,844 | -0.6% | | | -5.6% | | |
| SmolLM3-3B, 2 layers | R1 at 4,000 | 4 | 768,306 | -2.4% | -2.2% | -1.2% | | | |
| Phi-4-mini, 2 layers | R1 at 4,000 | 4 | 1,026,301 | -0.8% | -0.6% | | | | |
| Qwen3.5-2B | R4 | 4 | 2,084,720 | -3.2% | -1.7% | -0.5% | -9.0% | -5.8% | -6.5% |
| Qwen3.5-4B | R2 | 4 | 2,225,099 | -1.1% | -0.4% | | -10.3% | -5.0% | -5.4% |
| SmolLM3-3B | R3 | 4 | 1,285,843 | -2.0% | -1.2% | -0.4% | | | |
| Phi-4-mini | R3 | 4 | 1,677,624 | -1.7% | -0.9% | | | | |
| LFM2-2.6B, 6 layers | R3 | 4 | 3,944,882 | -2.4% | -1.3% | -0.5% | | | |
| Gemma 4 E2B, layers 0-4 | R4 | 4 | 2,333,456 | -6.5% | -2.9% | +1.1% | | | |
| Qwen3.5-2B | R4 | 8 | 1,465,409 | -13.9% | -7.1% | | -24.3% | | -19.3% |
| SmolLM3-3B | R4 | 8 | 767,072 | -4.4% | -2.6% | | | | |
| Gemma 4 E2B | R4 | 8 | 1,299,814 | -9.0% | -4.2% | | | | |
| SmolLM3-3B, a layer | B4 | 4 | 656,146 | -2.9% | -1.8% | | | | |

**Whole small models, real weights:**

| model | case | MCOLS | cycles | VPU busy | novop | half | comp1 | nose | halfse | half + halfse |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | decode | 4 | 3,623,644 | 15.5% | -0.7% | | | | | |
| LFM2.5-230M | decode | 4 | 1,339,842 | 12.1% | -0.4% | | | | | |
| Qwen3.5-0.8B | decode | 4 | 4,760,853 | 14.9% | -0.4% | | | -7.5% | | -3.8% |
| Qwen3-0.6B | decode at 4,000 | 4 | 5,267,247 | 33.2% | -0.2% | -0.2% | +0.1% | | | |
| Qwen3-0.6B | R4 | 4 | 5,212,997 | 36.4% | -7.9% | -3.5% | -0.7% | | | |
| LFM2.5-230M | R4 | 4 | 1,630,055 | 32.7% | -8.5% | -4.0% | -0.7% | | | |
| Qwen3.5-0.8B | R4 | 4 | 6,821,095 | 30.3% | -22.2% | -9.3% | -4.3% | -31.5% | -17.4% | -26.7% |
| Qwen3-0.6B | R4 | 8 | 3,727,896 | 51.0% | -8.8% | -4.3% | | | | |
| Qwen3.5-0.8B | R4 | 8 | 5,653,030 | 36.5% | -22.1% | -10.9% | | -45.3% | -21.1% | |
| Qwen3-0.6B | B4 | 4 | 6,478,568 | 30.8% | -5.0% | -2.7% | -0.5% | | | |
| Qwen3-0.6B | B4 at 4,000 | 4 | 13,086,624 | 51.9% | -2.7% | -1.4% | | | | |
| Qwen3-0.6B | B8 | 8 | 8,611,275 | 44.8% | -9.5% | -4.7% | | | | |

**MoE, every expert resident** (per layer, as prefix differences; `layer` runs at 2 rows):

| model | case | layer | cycles | novop | half | comp1 | nose | halfse |
|---|---|---|---|---|---|---|---|---|
| Gemma 4 26B-A4B | decode row | 1 (sliding) | 756,665 | -1.8% | -0.7% | -0.04% | | |
| Gemma 4 26B-A4B | decode row | 5 (full) | 855,185 | -2.3% | -1.0% | -0.05% | | |
| Qwen3.5-35B-A3B | decode row | 1-2 (DeltaNet), each | 323,937 | -1.6% | -0.6% | | -11.8% | |
| Qwen3.5-35B-A3B | decode row | 3 (attention) | 263,309 | -2.3% | -1.2% | | -0.1% | |
| Gemma 4 26B-A4B | prefill layer, R2 | 1 | 969,765 | -3.3% | -1.4% | | | |
| Gemma 4 26B-A4B | prefill layer, R2 | 5 | 993,612 | -3.7% | -1.8% | | | |
| Qwen3.5-35B-A3B | prefill layer, R2 | 0 (DeltaNet) | 382,905 | -4.3% | -1.9% | | -20.8% | -10.8% |
| Qwen3.5-35B-A3B | prefill layer, R2 | 3 (attention) | 296,391 | -6.2% | -2.6% | | 0 | 0 |

**Per workload (the card's models; the small whole models in brackets):**
- **Single-user decode: the VPU does not bind.**
  - An infinitely fast VPU saves 0.6-0.8% (0.4-0.7%); a 2x one about half of that.
  - At 4K context the softmax grows: SmolLM3 -2.4% (half -2.2%), Phi-4 -0.8%. Qwen3-0.6B's
    whole token gains 0.2%, because the KV stream hides it.
  - The MXU's gaps in decode follow QACT, not VOPs (Qwen3: 48K gap cycles, 1.3%).
- **Prefill: bound by the MXU at MCOLS 4.**
  - The VPU is 1-6.5% on the card's models (half 0.4-2.9%); Gemma 4 E2B is the largest.
  - At MCOLS 8 the MXU stops binding and the VPU shows: Qwen3.5-2B -13.9% (half -7.1%), E2B
    -9.0% (-4.2%), SmolLM3 -4.4% (-2.6%).
  - For Qwen3.5 the stream engine's DSTEP is the larger share in both cases (section 4).
- **Multi-user:** SmolLM3 at 4 users -2.9% a layer (half -1.8%); Qwen3-0.6B -5.0% (-2.7%);
  8 users at MCOLS 8 -9.5% (-4.7%). At 4,000 tokens the KV stream dominates and the share falls
  (-2.7%).
- **MoE:** a decode row's MoE layers -1.6 to -2.3% (half -0.6 to -1.2%); a layer-major prefill
  layer -3.3 to -6.2% (half -1.4 to -2.6%), with every expert resident. On the card the experts'
  host-link time comes on top (docs/offload.md 13.7: the 35B's 18.5 s prefill at 2 rows was
  8.3 s of layer compute and 7.2 s of exposed misses), so these shares shrink further. The 35B's
  DeltaNet layers are the exception: DSTEP is 11.8% of a decode row's layer and 20.8% of a
  prefill layer.
- **Composites:** comp1 bounds any composite speed-up.
  - On the card's models it is at most -1.2%. Gemma 4 E2B's +1.1% is the overlap of reordered
    VOPs: comp1 changes their latencies too.
  - On the small whole models it is -0.5 to -0.7%, and -4.3% on Qwen3.5-0.8B, whose DeltaNet
    sigmoids are a larger share.
  - Their throughput is not worth hardware.

## 4. The stream engine (Qwen3.5's DeltaNet)

DSTEP holds the engine: slot 0 and the tail run it at 8 state words a cycle, and VOPs wait.
In the Qwen3.5 family's prefill runs it outweighs the VOPs:

| case | MCOLS | half (VPU x2) | nose (DSTEP free) | halfse (DSTEP x2, overstated) | half + halfse |
|---|---|---|---|---|---|
| Qwen3.5-2B R4 | 4 | -1.7% | -9.0% | -5.8% | -6.5% |
| Qwen3.5-2B R4 | 8 | -7.1% | -24.3% | | -19.3% |
| Qwen3.5-4B R2 | 4 | -0.4% | -10.3% | -5.0% | -5.4% |
| Qwen3.5-0.8B R4, whole | 4 | -9.3% | -31.5% | -17.4% | -26.7% |
| Qwen3.5-0.8B R4, whole | 8 | -10.9% | -45.3% | -21.1% | |
| Qwen3.5-35B-A3B prefill layer (DeltaNet) R2 | 4 | -1.9% | -20.8% | -10.8% | |

- `halfse` halves DSTEP's state traffic as well as its datapath time. In these prefill runs port
  B is 48-58% busy (Qwen3.5-0.8B at MCOLS 4 and 8), so the traffic half gives little: there
  halfse is close to a 2x datapath. In decode, which is bound by DRAM, it overstates.
- In decode, `nose` (-5.4 to -7.5%) is mostly the state's DRAM traffic: 2 MB a DeltaNet layer
  against ~32 MB of weights for the 2B. A faster datapath would not remove that traffic.
- Per DeltaNet layer and row (static), the 0.8B and the 2B issue 15.7K VOP cycles and 32,768
  DSTEP datapath cycles (16 heads x 2,048); the 4B issues 22.7K and 65,536.
- A multi-row DSTEP (one pass over the state for a run's R rows, docs/prefill.md 8) would cut
  the state traffic R-fold. At 8 words a cycle it saves only the fill (~2.5% on the 0.8B), because
  the datapath binds. docs/prefill.md's bound for it at no datapath cost: -28.6% of the 0.8B's
  4-row prefill. A 16-word datapath is what turns it into a gain.

The DSTEP datapath is fed from port B through the DMA, not from TMEM. A 16-word stream engine
needs the DMA's stream side widened to 16 words and slot 0 and the tail doubled or pumped; it
does not need the TMEM's banks.

## 5. Pumping: the experiment (`rtl/exp/vpu_pump/`)

**The design** (`vpu_pump_lane.sv`). `vp_pair_pump` runs one multiply-add unit at clk_f =
2 x clk_s for two lanes:
- a 2:1 operand mux sits in front of the unit's input registers: lane 0 on the mid edge, lane 1
  on the slow edge;
- the result goes into one of two holding registers by phase, which the slow clock reads;
- the phase is clk_s's toggle sampled by clk_f, with no reset; both clocks come from one MMCM
  at phase 0;
- the unit advances on the VPU's pipeline enable, a clk_s signal, so a frozen core cycle
  freezes both fast cycles.

The unit variants are:
- **flat:** today's `otpu_fmma`, 6 stages;
- **deep:** each stage followed by XR plain registers for synthesis retiming;
- **split** (`vp_fmma_split`): the two stages that are too long at 3.75 ns split by hand, plus a
  second product register (the DSP's M and P), 10 stages. The two split stages are the
  multiplier's rounding (`fp_mul_s2`: select, then the increment) and the adder's normalize
  (`fp_add_s3`: the 16/8 shift levels, then the 4/2/1 levels).

**Bit-exact** (simulated: `run_tb.sh`, Verilator). Random operands with zeros, denormals,
infinities and NaNs, a random pipeline enable, and 15,017 enabled core cycles. Every pumped
variant's two lanes equal two of today's units, bit for bit, at a fixed offset. The latency in
enabled core cycles is flat 4, deep 7, split 6; today's unit is 6.

**Out of context** (measured: Vivado 2026.1, xc7k480tffg1156-2, clk_s 133.33 MHz and clk_f
266.67 MHz from an MMCME2_ADV, all I/O registered at clk_s; `ooc_lane.tcl`, `run_ooc.sh`). Four
pumped units give the same element rate as eight of today's:

| variant | LUT | FF | DSP | slices | WNS at clk_f (failing endpoints) |
|---|---|---|---|---|---|
| lane0: 8 x `otpu_fmma` at 133.33 MHz (today) | 4,378 | 3,798 | 32 | 1,665 | +2.441 ns (clk_s) |
| lane1: 4 pumped, flat (6 stages) | 2,868 | 3,330 | 16 | 1,304 | -0.940 ns (729) |
| lane2: 4 pumped, deep (12 stages), retiming | 2,924 | 4,590 | 16 | 1,516 | -0.247 ns (75) |
| lane2n: the same without retiming | 2,924 | 4,590 | 16 | 1,516 | -0.247 ns (75) |
| lane3: 4 pumped, deep (18 stages), retiming | 2,900 | 5,035 | 16 | 1,673 | -0.237 ns (66) |
| **lane4: 4 pumped, split (10 stages)** | **2,714** | **4,043** | **16** | **1,396** | **+0.123 ns (0); WHS +0.088** |

- **Flat** fails on the product: the input register goes through the two-DSP cascade (PCIN)
  into the product register in one fast period.
- **Deep:** synthesis retiming moved nothing (lane2 and lane2n are identical). What remains is
  the rounding stage and the normalize, -0.25 / -0.18 ns.
- **Split:** the hand split closes. The slow-to-fast paths (the operands and the enable into
  the input registers) have +0.22 ns, and the fast-to-slow returns +1.54 ns.

**What pumping saves.** Per 8 lanes' rate it saves 1,664 LUT, 16 DSP and 269 slices, and costs
245 more FF (lane4 against lane0). That is for the multiply-add units only: the other ~86% of
the VPU is not pumped.

**The whole VPU at 266.67 MHz is not an option** (measured, `vpu8se2x`: `otpu_vpu` as built at
3.75 ns). WNS is -1.394 ns, with 12,778 failing endpoints of 78,782. The worst path is the
tail's Q multiplier cascade, and the failures run through the trees, the composites' control
and the write buffer as well.

## 6. Options

**Out of context** (measured, Vivado 2026.1, xc7k480t-2, 7.5 ns, `ooc_vpu.tcl`):

| `otpu_vpu` | LUT | LUTRAM | FF | DSP | slices | BRAM | WNS |
|---|---|---|---|---|---|---|---|
| vpu8se: LANES 8, the stream engine (as built) | 30,802 | 2,901 | 27,415 | 86 | 12,341 | 16 | +0.545 ns |
| vpu8: LANES 8, no stream engine (CL 2 long lanes) | 24,260 | 1,795 | 20,041 | 68 | 9,334 | 8 | +0.317 ns |
| vpu8cl4: LANES 8, no stream engine, CL 4 | 35,606 | 2,657 | 30,095 | 104 | 13,517 | 8 | +0.519 ns |
| vpu16: LANES 16, no stream engine, CL 4 | 59,043 | 3,910 | 44,058 | 168 | 21,787 | 0 | +0.291 ns |
| vpu8se2x: as built, at 3.75 ns (section 5) | | | | | | | -1.394 ns |

**(a) A 2x VPU: 16 lanes, or 8 lanes pumped.**
- Gain: `half` in section 3. On the card's models that is 0.4-2.9% of prefill at MCOLS 4,
  2.6-7.1% at MCOLS 8, ~1.8% of a batched layer, ≤1% of decode, and 0.6-2.6% of MoE layers.
- The real 16-lane VPU in the co-sim (Qwen3-0.6B, `OTPU_LANES=16 OTPU_ULANES=8`, no DSTEP) is
  bit-exact against the ISA simulator. Against 8 lanes:
  - 4-row prefill: 5,011,350 cycles against 5,212,997, -3.9% (`half` gave -3.5%);
  - a 4-user step: 6,273,939 against 6,478,568, -3.2% (`half` -2.7%);
  - decode at 4,000 tokens: -0.26% (`half` -0.2%).

  So `half` is a fair, slightly low model of 16 lanes. The CL 4 long lanes also run the
  composites at 4 words a cycle, faster than COMP8's EXP2.
- Cost, measured out of context: 16 lanes against 8 at the same CL 4 is +23.4K LUT, +14.0K FF,
  +64 DSP and +8.3K slices. Against today's VPU it is +28.2K LUT and +9.4K slices, and the
  stream engine is gone.
- A 16-lane VPU that keeps the engine is at least the 8 -> 16 delta on top of today's
  (estimated: ~54K LUT and ~20.6K slices, before the engine's own part doubles).
- On the xc7k480t's 74,650 slices, +8-9K slices takes the image from 81.3% to ~92-94%, before
  TMEM. TMEM needs 16 banks on every copy: docs/mxu_study.md (yosys, an older design) put the
  board's TMEM ports at 12.1K -> 35.7K LUT for 16 lanes. LANES = 16 already failed to route on
  this part at 212K placed LUT (docs/board.md).
- 16 lanes also drop the stream engine, which the RTL builds for 8 lanes only: DSTEP and STREAM
  become illegal, and Qwen3.5 falls back to the VOP path for DeltaNet. A 16-lane VPU that keeps
  DSTEP needs the tail and the RDOT partial loop rebuilt; that design does not exist.
- Pumping the units instead of doubling them saves ~1.7K LUT and 16 DSPs per 8 lanes (section
  5). Everything else of a 2x VPU still doubles: the TMEM ports, the write buffer, the trees,
  the composites' state. The second clock adds multicycle constraints, as in docs/dsp_pumping.md.

**(b) VPU_CL 4.** Unused since COMP8. With the stream engine, `otpu_vpu` issues composites
LANES columns a cycle into `otpu_se_comp` (`NCL = C8 ? LANES : CL`), so CL changes nothing in
the board build. docs/board.md's `make bit VPU_CL=4` note predates COMP8. Without the engine,
CL 4 against CL 2 costs +11.3K LUT, +36 DSP and +4.2K slices (vpu8cl4 against vpu8): the long
lanes' 9 extra slots each.

**(c) Composites.** `comp1`, composites at the elementwise rate and latency, is the bound of any
composite change: at most -1.2% on the card's models. More stages per lane (NS 6: EXP2 in two
passes, RECIP in one) or pumped stages would cost DSPs and fabric for that. No.

**(d) The stream engine's DSTEP datapath at 16 words.** It is worth up to ~5-6% of the Qwen3.5
family's prefill at MCOLS 4 (halfse) and ~12% at MCOLS 8 (the 2B's half + halfse less half).
halfse overstates by its traffic share; nose bounds the gain at 9-10% and 24%. With a multi-row
DSTEP the state traffic also drops R-fold (docs/prefill.md 8: -28.6% on the 0.8B at no datapath
cost).
- It needs no TMEM bandwidth: the state comes from port B through the DMA.
- Doubling slot 0, U and Q costs ~8.8K LUT and 64 DSPs of units, from lane0's measured units,
  plus the doubled partial loop, tree and delay line (estimated). Pumping them instead (lane4)
  saves ~1.7K LUT and 16 DSPs per 8 units, but only once they no longer share slot 0 with the
  VPU's elementwise path at the slow clock.
- This is the one vector-side change that reaches double digits, for one model family and
  only once the MXU has MCOLS 8's rate.

**(e) UltraScale+ (DSP48E2).** The 27 x 18 multiplier still needs two DSPs for a 24 x 24
significand product, and DSPs do not bind on either UltraScale+ board. DSP48E2 reaches 775 MHz
at -2 (docs/dsp_pumping.md 1), so the DSP would not limit pumping; the fabric would. Even on the
-2 Kintex at 266.67 MHz the rounding and normalize stages had to be split. At the BCU1525's
planned 200 MHz core, 2x would be 400 MHz fp32 fabric: not attempted. With LUTs to spare (a
slice uses 39% of a VU9P SLR's LUTs, docs/bcu1525.md 7), widening is simpler than pumping there.

**(f) TMEM.** A 2x VPU needs 16-word VA, VB and write ports, which is tmem-pump's subject. (c)
and (d) need none.

## 7. Recommendation per board

What a 2x VPU gives each workload, in two views:
- **The co-sim's `half`** on the card's models (section 3, simulated).
- **The workloads agent's scoreboard** (`--vpu 2`, an analytic model; branch
  worktree-agent-acdd3a9dfe2e66038 at 5465f08a, calibrated on this page's runs). Its `--vpu`
  also scales DSTEP. The ratios are speed-ups against each board as it is.

| board | decode | prefill | multi-user | MoE | recommendation |
|---|---|---|---|---|---|
| YPCB-00338 (xc7k480t-2, 81% slices) | co-sim ≤1%; scoreboard 1.000x | 0.4-2.9% (MCOLS 4), 2.6-7.1% (MCOLS 8); 1.045x | 1.8-2.7%; 1.016x | 0.6-2.6%; 1.016x | no 2x VPU, no pumping |
| BCU1525 (VU9P, 2 slices, DDR4) | 1.000x | 1.044x | 1.016x | 1.015x | 16 lanes only with MCOLS 8 and a 16-word stream engine |
| AWS F2 (VU47P, HBM, S 4 at 250 MHz) | 1.000x | 1.041x | 1.015x | 1.011x | no; spend LUTs on slices |
| ASIC (NanGate45 card architecture) | as the YPCB at the same MCOLS | | | | keep 8 lanes; a 16-word DSTEP path if Qwen3.5 prefill matters |

On top of MCOLS 8 with 4x TMEM (8-row prompt runs), the scoreboard puts `--vpu 2` at:
- prefill: 1.086x geomean over its 9 models;
- the Qwen3.5 family: 0.8B 1.34x (DSTEP-bound, 43%), 2B 1.15x, 4B 1.12x. Most of that is
  DSTEP, which the knob scales too;
- the others: 1.01-1.06x;
- 8-user batches: 1.032x.

That is the same picture as the co-sim: once the MXU is faster, the vector side matters for
Qwen3.5 through DSTEP, and little otherwise.

- **YPCB.** Do not build a 2x VPU or pump the VPU.
  - The best case is 2.6-7.1% of prefill, once the MXU has MCOLS 8's rate.
  - That would cost a second VPU's worth of fabric and a 16-bank TMEM on a part at 81% slices,
    where LANES 16 already failed to route.
  - Pumping is feasible (section 5) but saves only the units.
  - If the Qwen3.5 family's prefill matters after the MXU's rate is raised, prototype (d): a
    16-word DSTEP datapath with a multi-row DSTEP. Bound it in the co-sim first with a
    datapath-only hack (open item 1).
- **BCU1525.** LUTs are not short (one slice per SLR at 39%), and dsp-pump recommends MCOLS 8
  per slice with plain DSPs. With MCOLS 8 the VPU's share is 2.6-7.1% of prefill. A 16-lane VPU
  is affordable there, but the stream engine must come along, or Qwen3.5 regresses to the VOP
  path. So: plain 16 lanes and a 16-word stream engine, no pumping (the fabric would need 400
  MHz at a 200 MHz core). Only once the 16-lane engine exists.
- **AWS F2.** LUTs limit the slice count (~7 slices of today's size, docs/dsp_pumping.md 6). A
  16-lane VPU costs a share of a slice for a few percent of prefill, while another slice adds a
  whole slice's throughput to batched and MoE serving. Keep 8 lanes.
- **ASIC.** No DSPs, so pumping means a faster clock on the same logic: no gain over a faster
  core. Widening is linear in area: the VPU is 1.161 mm² of the NanGate45 card build
  (docs/asic.md), plus the TMEM's bank macros. Keep 8 lanes. The DSTEP path (d) is the only
  vector-side addition worth its area, and only for the Qwen3.5 family.

## 8. Risks and open items

1. **A datapath-only DSTEP bound.** `halfse` also halves the state traffic, and `nose` removes
   it. Before any RTL for (d), add a co-sim knob for DSTEP's words per cycle; a timing-only
   change in the tail's pace would do. Measure 16 words with today's traffic, and with a
   multi-row DSTEP.
2. **The zero-image co-sims** cover a few layers of each larger model, without the LM head and
   with every MoE expert resident. Whole-model shares are lower in decode, since the LM head
   has no VOPs. The card's MoE runs add host-link time.
3. **Pumping in a full build** adds a second clock with multicycle paths (slow -> fast with
   +0.22 ns out of context) to a design at 81% slices. Hold between the related clocks met out
   of context (+0.088 ns). It is unproven in a full build; docs/dsp_pumping.md 8 lists the same
   risks for the MXU.
4. **No card time is needed** for these conclusions: they are co-sim upper bounds and
   out-of-context measurements. A pumped or wider VPU would need the usual RTL tests, then a
   FAST=1 build (the dev-builds rule) and a qual, before any card claim.
5. **LANES 16 for Qwen3.5** is not a configuration today: DSTEP and STREAM are illegal without
   the stream engine, and the compiler does not reject `board_config(LANES=16, DSTEP=True)`.

## Files

- `tools/vpu_hack.py`: the transforms, the static inventory, and `cosim` (zero-image runs of
  any model's layer prefix).
- `tools/perf_qwen.py`: `--batch`, `--hack`, `--json`. `tools/moe_prefill_cosim.py`: `--hack`.
- `rtl/exp/vpu_pump/`: the pumped lane (`vpu_pump_lane.sv`), its testbench (`tb_vpu_pump.sv`,
  `run_tb.sh`), the OOC scripts (`ooc_lane.tcl`, `ooc_vpu.tcl`, `run_ooc.sh`) and the co-sim
  lists (`run_cosim.sh`, `lists/`).
