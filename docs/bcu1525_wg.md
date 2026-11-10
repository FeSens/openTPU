# The BCU1525 at WG 8: decode at the DIMMs' bandwidth

The BCU1525's one-card image: one openTPU slice with the MXU's weight groups (WG 8) on four
DDR4-2400 UDIMMs behind AMD's DDR4 MIG, the core at 150 MHz. This page is its reference: the
goal and its measure, the memory path, the clocks, the floorplan, the build, what closing
150 MHz took, the card's results, and what limits them. The board itself, its pins, the thermal
governor and the earlier LiteDRAM images are in [bcu1525.md](bcu1525.md); two slices on the card
in [bcu1525_s2.md](bcu1525_s2.md); how to build and bring up an image in
[boards/bcu1525/README.md](../boards/bcu1525/README.md).

**Short answer.**
- **The goal was decode at 86% or more of the four DIMMs' peak** (4 x 19.2 = 76.8 GB/s: 66.05
  GB/s of useful bytes) on three large models, with the tokens the reference gives. On the card
  (`bcu1525-wg:bhw16`, 150 MHz, 2026-11-01 and 11-02) with real weights: Qwen3.6-35B-A3B
  **88.5%**, Qwen3.8-27B **90.6%**, GPT-OSS-20B **~88.0%**; eight tokens and their logits
  bit-exact against the ISA simulator for all three (section 10 has each number's run).
- **The core is not the limit.** Port W reads 1 KiB a step: 153.6 GB/s at 150 MHz, twice the
  DIMMs. What a token loses is the stream's stops (reads and writes beside it, bank conflicts,
  gaps between instructions), so the levers were in the memory path and the program's order,
  not the MXU.
- **The first images read 68-73%** (small Qwen models, the only ones they could hold). Three
  things in the memory path came next: the MIG's bank bits moved off the stream's stride
  (`WG_MIGMAP=b15`), the SmartConnect on the stream's path replaced by a clock converter (it
  gave L beats in L + 1 cycles: 88.9% at 8-beat bursts), and the address space laid out so that
  what decode writes never shares a DRAM bank with the weights streaming beside it (the
  bank-half window, BHW 33). With those and the sequencer's wider window the 27B and GPT-OSS
  were over 86%; the 35B read about 85.2% until its DeltaNet states went to 16 bits (section 9).
- **150 MHz closes with one MMCM**, the slice in SLR2, at WNS +0.019 / WHS +0.007 ns on the
  `:bhw16` build. The margins are small: section 6 is what each failed build taught.

## 1. The goal and its measure

**Decode token bandwidth** is the useful bytes a decode step reads from DRAM (weights with their
scales, KV, states: what the model's arithmetic consumes, counted once) divided by the step's
time, as a share of the DIMMs' peak, 76.8 GB/s. A step's time is its cycle count on the card at
the core clock. 86% is 66.05 GB/s.

| Model | Shape | Image | Decode at 86% |
|---|---|---:|---:|
| Qwen3.8-27B | 64 layers (48 DeltaNet, 16 attention), hidden 5120, MLP 17408 | 14,930 MiB | ~4.5 tok/s |
| Qwen3.6-35B-A3B | 40 layers, 256 experts, top 8 | 17,903 MiB | ~34 tok/s |
| GPT-OSS-20B | 24 layers, 32 experts, top 4 | | ~26 tok/s |

(The last column is section 10's card rate scaled to 86%; GPT-OSS's from its 2,510 MB of useful
bytes a token.)

Why WG 8: with D = 128 the MXU's port B takes 128 bytes a step, 19.2 GB/s at 150 MHz, a quarter
of the card. With weight groups the MXU's port W reads **P = WG x D = 1 KiB a step**, one
128-byte beat on each of 8 channels: 153.6 GB/s at 150 MHz, 102.4 at 100 MHz, above the DIMMs at
either clock. In co-simulation (Qwen3.5-4B) the weight-bound phases take the same time at 100 and
150 MHz (the LM head 9.20 ms at both); the clock buys the compute-bound DeltaNet and attention
phases (+18% tokens a second at 150).

## 2. The memory path

`targets/bcu1525-wg.toml` (platform 5): S 1, D 128, MCOLS 8, WG 8, MXU_IMPL 2, TMEM pumped at 2x,
the MXU's chunk FIFO in UltraRAM, a trace buffer. From the core down:

- **Virtual channels.** The slice's 8 channels of 128-byte beats are two a DIMM, interleaved at
  4 KiB inside it: channel j = 2d + s is DIMM d's half s, its byte o at the DIMM's byte
  `{o[CL-1:12], s, o[11:0]}`. A channel is 4 GiB (CL 32), the card 32 GiB (section 8).
- **A channel's bridge** (`rtl/shell`'s `otpu_hbm_ch`): a core half, a register pipeline of N
  stages and an AXI half, so the two halves can sit on different dies. Its AXI3 side is 256 bits
  at aclk, 300 MHz: 9.6 GB/s, half a DIMM. Port W's reads are posted into the MXU's chunk FIFO
  (no wait in the core for a beat's return). A read is ordered against the core's writes by
  hazard buckets of 512-byte blocks: a read waits while a write to its bucket is in flight, and
  a write while a read is.
- **A pair** (`rtl/boards/bcu1525/otpu_mig_pair.sv`) joins a DIMM's two channels onto one 512-bit
  AXI4 port (4 ID bits: the channel and the bridge's own IDs), interleaving at the 4 KiB grain,
  with R FIFOs so that the MIG's reordering across IDs never stalls the other channel. The FIFOs
  are four lanes each (`otpu_mig_pair_rl`): the first build's single FIFO failed timing on its
  512-bit read data (section 6).
- **The DIMM's port** (`otpu_mig_hmux`): the pair and the host's DMA for that DIMM, muxed at
  aclk, into an AXI Clock Converter (`cc_mem<d>`) to the MIG's ui_clk. The host's DMA crosses
  from XDMA's clock through a SmartConnect of its own (`sc_h<d>`). There is no SmartConnect on
  the accelerator's stream (section 7 says why).
- **The MIG** (`boards/bcu1525/vivado/bd_wg.tcl`): AMD's DDR4 IP, AXI4 512 bits, DDR4-2400 at
  833 ps, CL 17 / CWL 12, with Micron's MTA8ATF1G64AZ-2G3 as the timing twin of this card's
  GOODRAM GR2400D464L17S/8G (one rank of x8 parts, DM, no ECC). Each MIG calibrates itself; no
  soft CPU. C1's MIG takes its system clock unbuffered from the IBUFDS that also feeds the clock
  wizard.
- **Power and the governor** as on the LiteDRAM top ([bcu1525.md](bcu1525.md) section 14):
  everything off after configuration, a DIMM's MIG in reset while off, the accelerator only while
  the governor grants it and all four DIMMs are calibrated.
- **Host.** XDMA Gen3 x16, 512 bits; BAR0: control 0x0, governor 0x10000, BMC 0x20000, SYSMON
  0x30000. The host's DMA reaches channel j at card address j x 4 GiB
  (`opentpu.host.board.channel_map`, from the CFG registers); the top permutes each burst's
  address into the DIMM's byte as the pair does.

The counter DRAM_RD does not see port W's posted reads; a count is 64 bytes on this target
(`board.counter_units`).

**Checked in simulation.** `tests/test_bcuwg_board.py` runs tiny Qwen3 (int8) and Qwen3.5 (fp4)
images on the board model (`sim/verilator/tb_bcuwg.sv`: the whole memory path in front of
`otpu_dimm_model`, four DIMMs' AXI4 ports with latency, jitter, a bandwidth cap, stalls and
out-of-order reads across IDs) bit for bit against the ISA simulator; `tests/test_migpair.py` the
pair alone at CL 29, 32 and 33; `tests/test_bcuwg_build.py` the build's generated parameters, the
pins file against the top's ports, the clock pair's netlist.

## 3. Clocks

One MMCM on the board's clk300_1 (VCO 1200 MHz): core_clk at VCO / CORE_DIV with CORE_DIV even,
100 MHz (12), 120 (10) or 150 (8); a 2x output for the pumped TMEM; aclk at 300 MHz for the
bridges' AXI halves. Each MIG has its own clocks from its DIMM's 300 MHz input.

The accelerator's clocks stop while the governor holds it off.

**As built.** Every image carded so far at 150 MHz, `:bhw16` among them, has acc_clk as a second
MMCM output behind a clock-enable buffer and acc_clk2x on a BUFGCE of the 2x output in the top.
The enable (the governor's sequencer, in core_clk) is registered on its own BUFG of the 2x
output before it reaches the buffers' CE. The first 150 MHz build had the enable into the CE
straight from core_clk and missed by 0.513 ns, its one failing endpoint (HARDSYNC, the 7-series
answer, is not valid on UltraScale+).

**In the tree, not yet built (2026-11-04).** The top now makes both clocks from the 2x output
with a BUFGCE_DIV pair on one enable and one clear: acc_clk2x at /1, acc_clk at /2. Each run of
the 2x clock starts at an even 2x edge and ends after an odd one, so acc_clk rises with core_clk
and both stop low; the clear resets the divider while they are stopped.
`boards/bcu1525/sim/tb_acc_clocks.sv` drives random stops and starts, and
`tests/test_bcuwg_build.py` checks the pair's netlist. It has not been through Vivado. Its
reason: with two MMCM outputs, every crossing between acc_clk and acc_clk2x carries the
outputs' phase error and jitter, 0.182 ns of hold uncertainty; on the placed 100 MHz
development build of a wide-MXU variant (docs/wide_mxu.md, a build that did not close) 94% of
the worst crossing's hold requirement was clock. The divided pair has one source.

## 4. Floorplan

The VU9P is three dies (SLRs). DIMM C0's pins are in SLR0, C1's and C2's in SLR1 with the PCIe
block, C3's in SLR2. A slice was sized before the first build from an out-of-context synthesis
for another UltraScale+ part: 141K LUTs, 299 block RAMs, 156 UltraRAMs and 1,438 DSPs, 36% /
41% / 49% / 63% of one of this card's SLRs (as placed on this card: Area, below).

`constraints/bcu1525_wg_floorplan.tcl` puts the accelerator and its bridges' core halves in one
SLR (`WG_SLR`, default 2; a soft pblock by default, `WG_FP=soft|hard|none`); the bridges' AXI
halves, the pairs and the port muxes stay with the placer beside their MIGs. Each channel's
bridge pipeline (RLY, the bridge's N, at aclk) follows from the distance: 4 k + 1 stages for a
DIMM k die crossings away, 2 for one on the slice's own die. With the slice in SLR2 that is 9
(C0), 5 (C1, C2), 2 (C3).

The first build had the slice in SLR1. With XDMA (54K LUTs), two MIGs (14K each) and the DMA's
interconnect beside it, SLR1's slices were 94.8% used (SLR0 7.4%, SLR2 37.7%), the router's
congestion level 6-7, and it missed 100 MHz by 0.884 ns, every failing path in SLR1 and nearly
all of its delay route. In SLR2 the slice shares its die with C3's MIG only.

**Area** (S = 1, 150 MHz, the build before the bank-half window; placed, of each SLR):

| | SLR0 | SLR1 | SLR2 |
|---|---:|---:|---:|
| CLBs used | 7.1% | 67.0% | 91.5% |
| LUTs | 3.8% | 39.2% | 56.1% |

The slice is 168,899 LUTs, 1,502 DSPs and 156 UltraRAMs, of which the MXU 48,495 LUTs and 1,218
DSPs. A 150 MHz build costs more area than a 100 MHz one of the same RTL: on the two-slice
image's pair of builds, 11.8% more LUTs out of synthesis, 7.4-7.8% more placed in the slices'
SLRs, 1.7-1.8 points more of their CLB sites.

## 5. The build

    cd boards/bcu1525
    IMAGE=wg WG_TARGET=bcu1525-wg:bhw16 CORE_MHZ=150 FAST=0 \
      WG_MIGMAP=b15 OTPU_IP_CACHE= STEP=synth ./run_vivado.sh
    IMAGE=wg WG_TARGET=bcu1525-wg:bhw16 CORE_MHZ=150 FAST=0 \
      WG_MIGMAP=b15 OTPU_IP_CACHE= STEP=impl  ./run_vivado.sh

(`make wg-synth` and `make wg-impl`; [boards/bcu1525/README.md](../boards/bcu1525/README.md) has
the prerequisites and the bring-up.) `run_vivado.sh` with `IMAGE=wg`:

1. `scripts/make_wg.py` writes the build's inputs into OUT_DIR (default
   `build/vivado_bcu1525_wg`) from the target: `otpu_board`'s parameters as a global include with
   the configuration's hash and the tree's commit, the RTL list, the clock divide, RLY, the
   channels' address bits.
2. `vivado/create_project_wg.tcl` builds the project and the block design (`bd_wg.tcl`: XDMA, the
   four MIGs, the clock wizard, the converters, the System Management Wizard).
3. `vivado/build.tcl` synthesizes, implements, and writes `reports/`, `otpu.bit`, `otpu.ltx` and
   `otpu.mcs`. `vivado/bcu1525_wg_check.tcl`, opt_design's pre-hook, stops the run if a late
   constraint missed a cell or a clock.

| Knob | What |
|---|---|
| `WG_TARGET` | `bcu1525-wg`, `bcu1525-s2` or `bcu1525-x2`, with an optional `:variant` ([targets.md](targets.md)) |
| `CORE_MHZ` | 100, 120 or 150 |
| `FAST` | 1: development strategies (a functional image); 0: the full ones |
| `STEP` | `bd`, `synth`, `impl`: one stage; unset: all |
| `IMPL_ALONE=1` | with `STEP=impl`: impl_1 runs with no Vivado holding the project (~2 GB less) |
| `IMPL_TO=opt_design` | with `STEP=impl`: stop after that step; `vivado/impl_resume.tcl` goes on |
| `WG_SLR`, `RLY` | the slice's SLR (default 2) and the bridges' pipeline stages, `N0,N1,N2,N3` |
| `WG_FP` | `soft` (default), `hard` or `none`: the slice's pblock |
| `WG_MIGMAP=b15` | the MIGs' address map with the bank bits at DIMM byte bits 16:15 (section 7); needs `OTPU_IP_CACHE=` (empty) |
| `WG_MIGRD=s62` | the MIGs' ordering Strict with 62 reads in flight (section 7: no gain; needs `OTPU_IP_CACHE=` too) |
| `WG_SET="K=V ..."` | parameter overrides for `make_wg.py --set` |
| `WG_PLAN`, `WG_MXU_COLS0/1`, `WG_MMCM`, `WG_STOP` | two slices ([bcu1525_s2.md](bcu1525_s2.md)) |
| `JOBS`, `SYNTH_THREADS` | Vivado's jobs and synthesis threads (memory, below) |
| `OUT_DIR`, `VIVADO`, `PREFLIGHT=0` | the output directory, the Vivado binary, skip the licence check |

**Directives and resuming.** `vivado/impl_resume.tcl OUT_DIR` continues an implementation that
stopped after opt_design, a step a Vivado, with markers so that a rerun skips what is done. Its
directives come from the environment: `OTPU_WG_PLACE`, `OTPU_WG_PHYS`, `OTPU_WG_ROUTE`,
`OTPU_WG_POSTPHYS` (each `Default` unless set), hooks `OTPU_WG_PREROUTE` and `OTPU_WG_PREPHYS`
(a Tcl file each), and `OTPU_WG_STOP=placed` for a placement trial with its reports. The
first 100 MHz build to meet timing set only phys_opt AggressiveExplore; the 100 MHz image that
went on the card used place ExtraPostPlacementOpt, phys_opt AggressiveExplore, route Explore
and a post-route phys_opt AggressiveExplore.

**Memory and time.** Peaks on a 32 GB build host: synthesis 20.4 GB (`JOBS=2`), placement
18.6 GB, routing 21.6 GB; a full S = 1 build about 1 h 50 min. Run synthesis and implementation
as two steps so that implementation starts in a fresh Vivado.

## 6. What closing 150 MHz took

In build order; each line is a failure and its fix, all of them in the flow now.

| Build | Result | What it taught |
|---|---|---|
| 100 MHz, slice in SLR1 | WNS -0.884 ns | SLR1 holds PCIe and two MIGs: the slice goes to SLR2 (section 4) |
| 100 MHz, slice in SLR2 | WNS +0.001, WHS 0.000 | the pair's R FIFOs in four lanes; RLY 9,5,5,2 |
| 100 MHz, directives as above | WNS +0.001, WHS +0.009; SLR2 81.1% of its CLBs; 22.3 W estimated | the first image on the card |
| 150 MHz, first | clock enable -0.513 ns | the stopped clocks' enable (section 3) |
| 150 MHz, second | WNS +0.035, WHS +0.010 | every clock met |
| 150 MHz, b15 map | WNS +0.008, WHS +0.010 | the map patch holds timing |
| 150 MHz, `:bhw16` | WNS +0.019, WHS +0.007 | |

Flow fixes the first builds needed, each of which would stop or silently spoil a build:

- **No bank INTERNAL_VREF in the pins file.** The MIG sets HPIO_VREF per byte lane; the
  INTERNAL_VREF the LiteDRAM image's constraints set on the twelve DDR4 banks gave DRC BIIVRC-1.
  The WG pins file has none, and `impl_resume.tcl` resets it on every POD12 bank.
- **DCI in the multifunction banks.** DRC DCIRST-1 asks for a DCIRESET where DCI pads share a
  bank with configuration pins. The top instantiates three, one a die (the VU9P has one
  configuration site in each SLR).
- **The accelerator's reset** is pipelined three core_clk cycles together with its enable; a
  single register driving the whole slice was the worst path at 100 MHz.
- **clk300_1's two MMCMs.** Its pad drives C1's MIG's MMCM and the clock wizard's through one
  IBUFDS and BUFG, two clock regions apart, which the I/O clock placer refuses; place_design's
  pre-hook (`vivado/bcu1525_wg_place_pre.tcl`) sets `CLOCK_DEDICATED_ROUTE BACKBONE` on the
  BUFG's output.
- **The MIGs' own block designs.** Each DDR4 IP holds a block design of its own; `build.tcl`
  skips them when it looks for the design's.
- **Board clocks.** Each MIG creates its own clock on its system clock pins, so the pins file
  defines none for them; clk300_1 and the PCIe reference clock, which no IP creates, are in
  `constraints/bcu1525_wg_clocks.xdc`.
- **proc_sys_reset's polarity** is read-only in a block design (it follows its source, the MIG's
  active-high reset): `bd_wg.tcl` checks it after validation instead of setting it.
- **A false path to the DCIRESETs' RST** is an error in opt_design (Constraints 18-512: the pin
  has no timing arc) and is gone.

From the denser builds since (two slices a card, the wide-MXU variants):

- **Density decides.** The two-slice images' slice SLRs are at 93.6% and 94.8% of their CLB
  sites at 150 MHz (62% of their LUTs: count sites, not LUTs) and end within 0.04 ns of zero
  slack either way ([bcu1525_s2.md](bcu1525_s2.md)); a wide-MXU variant at 98% missed 100 MHz
  by 2.9 ns.
- **A reset released on one die and used on another** needs its last stages in the destination
  clock, pinned to the die between ([bcu1525_s2.md](bcu1525_s2.md)).
- **Staged for the next dense build, not yet proven by one that closed:** the divided clock
  pair of section 3 (the wide variant's router turned hold fixing off over the two clocks'
  skew; `vivado/pulse_skew.tcl` reports it, `route_holdfix.tcl` is the pre-route hook); a soft
  exclusion pblock beside each DIMM's PHY (`WG_MIGKO`); high-fanout registers replicated by
  name before placement (`vivado/phys_screp.tcl`); register slices on the ports that cross
  dies (`WG_HRS`, `WG_PRS`).

## 7. What limited the stream, and the fixes

**The first images.** At 100 MHz the four DIMMs calibrated at DDR4-2400 on the first try;
otpu-diag's memory section (49 checks and a march C- over every byte) and otpu-selftest passed;
Qwen3-0.6B ran token for token at 1.08 Mcycles a token (92.6 tok/s), 56.0 GB/s: 73%. Qwen3.5-4B:

| Qwen3.5-4B on the card | 100 MHz | 150 MHz |
|---|---:|---:|
| cycles a token | 5,509.4k | 7,344.8k |
| tokens a second | 18.2 | 20.4 |
| useful bytes | 46.2 GB/s (60%) | 52.0 GB/s (68%) |

At 100 MHz by phase: a DeltaNet layer 155.7k cycles at 38.7 GB/s, an attention layer 94.5k at
60.7, the LM head 1,027.3k at 62.5. At 150 MHz the stream alone read 60.4-60.9 GB/s in fp4 (64.6
with its scales: 84%) and 64.2-64.9 in int8 (65.5 with scales: 85.3%); a pointer chase took 85.0
cycles a hop (567 ns); the head 64.3 GB/s (83.7%). For scale, AMD's own memory test on a
DDR4-2400 bank of a U250 through the same MIG reads 89.5% at best.

**The DeltaNet layers** were compute- and side-traffic-bound, not stream-bound: two row groups on
the stream engine (SE_R 2: a head's DSTEP 2,303 -> 1,448 cycles on the card, 2,275 -> 1,389 on
the board model), the convolution's
taps and windows a layer ahead, a shared pair in one DSTEP, and a deeper read-ahead (DMA_DEPTH
512) took the 4B from 20.4 to 22.15 tok/s (+9.0% with the pair loop).

**The bank map (`WG_MIGMAP=b15`).** The MIG's default map puts the bank bits where the 4 KiB
interleave makes the stream and the loads beside it collide in a bank. `vivado/mig_map.tcl`
patches the generated user-interface decode so that the bank comes from DIMM byte bits 16:15 and
the column from bits 14:8 and 5:3; the build checks the patch in all eight decodes after
synthesis, and the IP cache must be off. On the card: loads beside the stream 9,189 -> 6,092
cycles a MiB (-34%), a DeltaNet layer 175.4k -> 163.7k (-6.7%), the 4B 23.14 tok/s. A second
map with the bank group on higher bits was 12% slower on the card on the 35B's 4-layer cut (the
MIG spaces CASes of the same bank group at 12 tCK) and is not used.

**The MIG's reads in flight** (`WG_MIGRD=s62`: ordering Strict, the AXI read channel's FIFOs at
62 and 64) looked like the ceiling in the model (-7 to -8%) and gave -1% on the card: int8
256 MiB 66.0 GB/s, fp4 128 MiB 61.9. It is not in the build's recipe.

**The SmartConnect's R bubble.** The ceiling was the SmartConnect then between each pair and its
MIG. Its netlist, simulated between a master that is always ready and an ideal in-order memory at
the card's three clocks, returns a burst of L beats in L + 1 cycles:

| Burst | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|
| R beats a cycle | 0.800 | 0.889 | 0.941 | 0.970 |

That holds with one ID or two and with aclk above or below ui_clk. The pair's 8-beat bursts make
0.889, and with the DIMMs' refresh (4.5%) that is the ~86% the stream read. The MIG's own AXI
front end has no such bubble. In the same bench, between the pair and the MIG:

| | R beats a cycle |
|---|---:|
| SmartConnect as built | 0.889 |
| SmartConnect, no narrow bursts, 32 outstanding; the same with two threads | 0.889 |
| AXI Interconnect (crossbar for performance) | 0.616 |
| AXI Interconnect with register slices | 0.593 |
| **AXI Clock Converter** (asynchronous, 512 bits, 4 ID bits) | **1.000** |

Hence section 2's path: a clock converter a DIMM for the accelerator, the host's DMA brought to
aclk by a SmartConnect of its own, whose bubble does not matter, and muxed in our RTL. **Open:**
the host's DMA still pays the bubble, and any burst through a SmartConnect will; longer bursts
on the pair (16 or 32 beats) were not needed after the converter and are not built.

The board model has each of these (`otpu_dimm_model`: `+dimm_mc=1` the MIG's bank machine,
`+mc_ba=15` the b15 map, `+mc_rw` the reads in flight, `+mc_strict=1`, `+mc_sc=1` the bubble) and
matched the card within 0.4-1.9% on the stream benches once they were in.

## 8. 36-bit addresses: the whole 32 GiB

The first images addressed 4 GiB (512 MiB a channel). DRAM addresses are now `[arch] AW` bits
(36 on this target; docs/isa.md "Machine model"): LI and ADDI take a 64-bit immediate from two
words, RLD multiplies at AW bits, weights go through a window register. Below the core one
parameter, CAW (the channels' beat address bits, from the target's `channel_bytes`), sets the
memory path: a channel is 2^(CAW + 7) bytes.

| | CAW 22 (first images) | **CAW 25 (32 GiB)** | CAW 26 (64 GiB) |
|---|---:|---:|---:|
| a channel / the card | 512 MiB / 4 GiB | **4 GiB / 32 GiB** | 8 GiB / 64 GiB |
| the pair's CL / the DIMM's address | 29 / 30 bits | **32 / 33** | 33 / 34 |
| the MIG's AXI address | 33 | **33** | 34 |
| the host's DMA window a DIMM | 1 GiB | **8 GiB** | 16 GiB |
| channel j in the host's view | j x 512 MiB | **j x 4 GiB** | j x 8 GiB |

CAW 26 needs 16 GB DIMMs (another MIG part) and is not built: `bd_wg.tcl` and `make_wg.py` stop
above 25. No register reports AW; the host takes it from the card's CFG_HASH
([targets.md](targets.md), Limits).

## 9. The bank-half window, the wider sequencer window, 16-bit states

`bcu1525-wg:bhw16` is the target with four things on:

**The bank-half window (BHW 33).** A decode's writes (KV appends, the DeltaNet states, the
windows' stores) shared DRAM banks with the weights streaming beside them: a write held a weight
stream's bank open or closed it. The window splits the banks by address. The slice's first
2^BHW bytes take the map's bank-high bit from their own top bit (a swap of two address bits,
its own inverse; `otpu_mig_pair`, the top's host remap and the DIMM model alike):

| Region | Holds | Banks |
|---|---|---|
| [0, 4 GiB) | I/O, KV, states, windows, tables, programs | bank-high 0 (8 of 16) |
| [4, 8 GiB) | the DeltaNet mixers' weights | bank-high 1 (the other 8) |
| from 8 GiB | every other weight, the resident expert pool | all 16 |

Nothing the core writes lies at or above 4 GiB (the ISA simulator refuses a store there on a
windowed image), so the bridges also exempt reads from 4 GiB up from the hazard buckets. On the
board model, decode cycles of 2- and 4-layer cuts:

| Cut | BHW 0 | BHW 33 | with the exemption | against BHW 0 |
|---|---:|---:|---:|---:|
| 35B, 2 layers | 240,175 | 228,756 | 221,793 | -7.65% |
| 35B, 4 layers | 412,295 | 396,593 | 386,378 | -6.29% |
| 27B, 2 layers | 1,011,108 | 979,455 | 966,841 | -4.38% |
| 27B, 4 layers | 1,925,264 | 1,876,805 | 1,858,023 | -3.49% |
| GPT-OSS, 2 layers | 1,578,962 | 1,579,443 | 1,576,204 | -0.17% |

GPT-OSS has no mixers, so nothing moves but the low area's banks. CAPS2 bits 11:6 report the
window; it is in the CFG_HASH only when on.

**The sequencer's window of 32** (WIN 32: the wait entries' counts saturate at 15). More
instructions overlap the stream: 35B 84.97 -> 86.01%, 27B 89.61 -> 89.70%, GPT-OSS 86.99 ->
88.16% on the board model (projections from 2- and 4-layer cuts), for 5,923 LUTs and 3,330 flip-flops. A full scoreboard at that window
would cost about 25K LUTs and was not built ([bcu1525_s2.md](bcu1525_s2.md) section 4).

**MM's K tail (KT)**: an MM whose K ends in the middle of a block (GPT-OSS's hidden 2880), 1,264
LUTs; the other models' programs are the same word for word.

**16-bit states (ST16 e6m9).** A DSTEP or STREAM with the flag reads and writes its DeltaNet
state in a 16-bit float (6 exponent bits, bias 31, 9 mantissa bits; rounded to nearest even,
saturated), fp32 in the engine: half the state's bytes each way. On the 35B the state's path was
the decode's bound (bytes at the DIMMs' rate); the image chooses it per model by its divergence
from the float reference (docs/isa.md "ST16"). CAPS2 bits 14:13.

## 10. On the card

`bcu1525-wg:bhw16` at 150 MHz on one card, real weights. These come from the sessions' logs
(2026-11-01; the 35B's row from 11-02, with the embedding table on the card):

| Model | Cycles a token | Tokens a second | Share of 76.8 GB/s | Zero-weight image (11-01) |
|---|---:|---:|---:|---:|
| Qwen3.6-35B-A3B | 4,317,734 | 34.7 | **88.53%** (repeat 88.50%) | 88.39% (4,324,318) |
| Qwen3.8-27B | 31.47 M | 4.77 | **90.6%** | 90.74% (31,415,701) |
| GPT-OSS-20B | not recorded | | **~88.0%** | 88.60% (5,533,374: 27.1 tok/s) |

For each model eight tokens and their logits match the ISA simulator bit for bit (11-01), and
the ISA simulator's eight tokens match the reference implementation's. On 11-01, with the
embedding rows served by the host, the 35B's loop read 86.14% (its resident decode alone
88.32%); the table on the card took the loop to 88.53%. The board model's projections for the
variant without the 16-bit states were 35B 86.0%, 27B about 90%, GPT-OSS about 88%; the card
read the 35B at 85.17% on that variant's zero-weight image.

In use: the 35B as a chat with its whole 262,144-token context resident decodes at 33.3 tok/s
(4.5 Mcycles a token) with a first token after 1.1-1.6 s on short prompts.

**Open.**
- The stream's remaining loss is refresh (4.5%) and the stops between instructions; the
  sequencer's window and prefetch studies are in [bcu1525_s2.md](bcu1525_s2.md).
- Decode slows with context on the 35B (4.5 -> 7.37 Mcycles a token from a short context to
  31k positions): attention's KV, not the weight stream.
- GPT-OSS's attention needs int8 weights for quality (mean KL against the float reference 0.333
  in fp4, 0.049 in int8, on the format study's sample), which costs bytes a token.
- The SYSMON wizard's temperature bus can stop: [bcu1525_sysmon.md](bcu1525_sysmon.md).
