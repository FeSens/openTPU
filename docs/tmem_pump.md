# TMEM study: fewer copies, pumped copies, more words

Status: an experiment (2026-10-11 and 12), on branch `worktree-agent-ab3fe74f4336cefb9`. Nothing
here changes the card's build: the knobs default to today's TMEM. Numbers are marked by where
they come from: *measured* (a Vivado run, an RTL co-simulation, the compiler), or *estimate*
(arithmetic, the workloads scoreboard's analytic model). No number here comes from the card.

Questions:

1. Does the banked TMEM (`TMEM_IMPL 1`, three copies, the ASIC's) work on the FPGA, and does
   going from six copies to three relieve the YPCB build's routing and timing? What does it cost
   in cycles?
2. Can a block RAM copy run at twice the core clock and serve two read ports, so that the same
   behaviour takes half the block RAM?
3. Can one copy pumped at 3x or 4x replace the three (section 2, "One copy at 3x or 4x?")?
4. What would 2x, 4x or 6x the words buy (prefill rows a run, batched rows, MoE), and which board
   can afford it?
5. Which TMEM for each board (YPCB, BCU1525, F2, ASIC), and what does it do to decode, prefill,
   batched decode and MoE?

## TMEM today

A slice's TMEM is 64K fp32 words in LANES = 8 banks (bank = address mod 8). Eight read ports
(MXU, DMA, QUANT, VPU A, VPU B, QUANT 2, QUANT 3, COLL) and four write ports (DMA, MXU, COLL,
VPU) use it; a read returns the next cycle, a write lands through a registered stage
(`otpu_tmem`). On the FPGA (`TMEM_IMPL 0`) every read port has its own copy of the RAM, except
QUANT 3 and COLL, which share the DMA's copy (the arbiter keeps them apart): six copies of
8 banks x 8K x 32 bits, 8 RAMB36 a bank, 384 RAMB36. Every write goes to every copy.

On the card's production build (91a9c2d3, `BUILD_ID=91a9c2d3 CORE_MHZ=133.33 make bit DDR=1066`,
2026-10-11, WNS +0.013 ns) *measured*:

| | LUT | FF | RAMB36 | share of the design's block RAM |
|---|---|---|---|---|
| `u_tmem` | 6,158 | 4,060 | 384 | 64% |
| the design | 177,292 (59.4%) | 134,913 (22.6%) | 602.5 tiles of 955 (63.1%) | |
| IMEM (`u_seq/u_imem`, 4,096 instructions) | 601 | 0 | 28 + 1 RAMB18 | |

Slices are 76.6% used. The TMEM is the largest block RAM user by far, so any change to its
copies or its size moves the block RAM budget more than anything else.

## 1. The banked TMEM on the FPGA

`TMEM_IMPL 1` (`docs/asic.md`, "TMEM") keeps three copies of 1R1W banks, a read port on one copy
(`CPY`), and the slice's arbiter counts reads per copy and bank (`gnt_k`): a bank's address comes
from the port granted in it. The YPCB build now takes it as a knob:

    make -C boards/ypcb-00338 bit DDR=1066 CORE_MHZ=133.33 TMEM_IMPL=1 [TMEM_RQ=1]

(`create_project.tcl`'s generic, through `otpu_fpga_top_ld`, `otpu_native_sys` and `otpu_board`,
which sets RPB 1 for it; default 0, the card's build unchanged.)

### As the ASIC has it: does not close

The full YPCB build at production settings (`BUILD_ID=0924a7e1 CORE_MHZ=133.33 JOBS=2
TMEM_IMPL=1 make -C boards/ypcb-00338 bit DDR=1066`, the build host, 2026-10-11, 2 h 53 min) against
production (91a9c2d3, the same settings, 49 min), *measured*:

| | production (6 copies) | `TMEM_IMPL 1` (3 copies) |
|---|---|---|
| WNS / WHS | +0.013 / +0.012 ns | **-3.269** / +0.032 ns |
| block RAM tiles | 602.5 (63.1%) | 410.5 (43.0%) |
| `u_tmem` LUT / FF / RAMB36 | 6,158 / 4,060 / 384 | 5,830 / 3,228 / 192 |
| slice LUTs | 177,292 (59.4%) | 179,764 (60.2%) |
| slices | 57,198 (76.6%) | 57,467 (77.0%) |
| placer congestion windows (`report_design_analysis -congestion`) | North level 5 107%, South level 5 116% | North level 5 125%, East level 6 118%, West level 6 102% |
| initial router congestion above level 5 | none | East and West, Long, level 5, 37.8% of the tiles (around the stream engine's tail) |

The core clock's 30 worst paths (`timing_worst.rpt`) all start at the DMA's state (`u_dma/ackw`)
and run through the arbitration into a TMEM copy's bank address (`ADDRBWRADDR`, 27 of them) or
its bypass compare: 15-21 logic levels, 9.9-10.6 ns, 90% of it routing. Three copies free 192
block RAMs but no slices (the arbitration and the address multiplexers replace the copies'
fan-out), and the placer's congestion gets worse, not better. So on the FPGA the banked TMEM as
built for the ASIC does not relieve the routing; it moves the problem onto the grant.

### Request-priority addresses (`TMEM_RQ`)

The grant is the end of the slice's longest logic: every unit's request, the stalls, the
arbitration of all groups. In `TMEM_IMPL 0` it reaches only the RAMs' enables; in `TMEM_IMPL 1`
it also picks each bank's address, so it runs on through a wide multiplexer into 192 block RAMs'
address pins. `TMEM_RQ 1` takes it out of the address: a bank's address on a copy is the address
of the first port that *asks* for that bank, in the arbiter's priority order (DMA, COLL, VPU,
MXU, QUANT; `otpu_tmem` `RQ`, the ranks `PRANK`), whether or not it is granted; the arbiter
(`gnt_q`) grants a group only if no earlier group asks for any copy and bank it needs (asking,
granted or not), and otherwise as `gnt_k`. A granted port is then always the first asker in its
banks, so it reads its own address. The cost is a group held back by an earlier one that was not
granted itself. Results are the same as the ISA simulator's (the co-simulations below check every
run, `--check`); the testbench of section 2 checks `RQ` against `otpu_tmem` with the slice's
grants.

In the full build (`BUILD_ID=b383c919 CORE_MHZ=133.33 JOBS=2 TMEM_IMPL=1 TMEM_RQ=1 make -C
boards/ypcb-00338 bit DDR=1066`, the build host, 2026-10-12, 1 h 30 min) *measured*:

| | production (6 copies) | `TMEM_IMPL 1`, grants | `TMEM_IMPL 1`, `TMEM_RQ` |
|---|---|---|---|
| WNS / WHS | +0.013 / +0.012 ns | -3.269 / +0.032 ns | **-1.063** / +0.010 ns |
| failing endpoints (core clock), TNS | 0 | | 36,115, -11,711 ns |
| block RAM tiles | 602.5 | 410.5 | 410.5 |
| `u_tmem` LUT / FF / RAMB36 | 6,158 / 4,060 / 384 | 5,830 / 3,228 / 192 | 5,851 / 3,381 / 192 |
| slice LUTs | 177,292 (59.4%) | 179,764 (60.2%) | 178,784 (59.9%) |
| slices | 57,198 (76.6%) | 57,467 (77.0%) | 58,605 (78.5%) |
| placer congestion, level 5 | North 107%, South 116% | North 125%, East (level 6) 118% | North 111%, South 119% |

`TMEM_RQ` does what it was for: the grant no longer reaches the block RAMs, and none of the 100
worst paths ends at one. They are now spread over the slice: 60 end in the sequencer's command
queue (`u_seq` `sv` into `g_cmd`), 19 in the MXU, 13 in the TMEM's write stage (from the DMA's
state), and 5 in the VPU. All of them are 90-93% routing, 7.8-8.7 ns of wire. The banked TMEM still
misses 133.33 MHz by 1.06 ns. It frees 192 block RAMs but no slices (78.5% against production's
76.6%), and the placer's congestion around the VPU's stream-engine tail is above production's.
So, with the arbiter as it is, the banked TMEM does not fit the YPCB at the card's clock: both
banked builds place the slice worse than production does.

### Cycles

RTL co-simulation of the four workloads, *measured*: `tools/perf_qwen.py --ddr 1066 --mhz
133.33 --ldc --check --pos 544` (LiteDRAM co-sim, DDR3-1066, core 133.33 MHz; PAIR, DSTEP,
STREAM; fp4 weights with an int8 head) and `tools/moe_prefill_cosim.py` for the MoE models, each
with the card's TMEM and with the banked ones (`OTPU_TARGET=nangate45` is the card's arch with
`TMEM_IMPL 1`): the grant-addressed banked TMEM (`gnt_k`, 3 copies), the same with
request-priority addresses (`OTPU_TMEM_RQ=1`, 3 copies), and P2's four slots
(`OTPU_TMEM_RQ=1 OTPU_TMEM_NCP=4`). Every perf_qwen run is bit-exact against the ISA simulator.
Core cycles, and the banked TMEMs against the card's:

| workload | card's TMEM | banked, grants | banked, RQ | P2 (4 slots, RQ) |
|---|---|---|---|---|
| decode, Qwen3-0.6B (28 layers, head) | 3,623,644 | | +0.43% | +0.51% |
| decode, Qwen3.5-0.8B (24 layers, head) | 4,760,853 | | +0.08% | +0.01% |
| prefill, Qwen3-0.6B, 4 rows, 2 layers | 372,383 | +1.93% | +2.00% | +1.60% |
| prefill, Qwen3.5-0.8B, 4 rows, 4 layers | 1,137,275 | | +0.06% | +0.08% |
| batched decode, 4 users (Qwen3-0.6B, 2 layers, each row's logits) | 1,697,925 | +0.37% | +0.37% | +0.30% |
| batched decode, 6 users | 3,326,827 | | +0.32% | +0.25% |
| MoE decode, Qwen3.5-35B-A3B (4 layers, every expert present) | 5,490,770 | | +0.04% | +0.03% |
| MoE decode, gemma-4-26B-A4B-it (layers 0 and 5) | 1,613,257 | | +0.09% | -0.01% |
| MoE prefill layer run, 35B, 2 rows | 296,391 | | +1.05% | +0.66% |
| MoE prefill layer run, 26B, 2 rows | 993,612 | | +0.42% | +0.06% |
| MCOLS 8: prefill, Qwen3-0.6B, 4 rows | 267,269 | | +2.35% | +1.99% |
| MCOLS 8: prefill, Qwen3-0.6B, 8 rows | 502,925 | | +3.19% | +2.09% |
| MCOLS 8: batched decode, 6 users | 1,793,140 | | +0.41% | +0.36% |

(Eight users at once do not fit IMEM: 4,597 instructions at position 544.) The banked TMEM costs
little where one unit streams (decode, MoE decode: under 0.5%) and 1.6-3.2% in prefill, where the
MXU's reads, the quantizer's three and the VPU's two meet in the same banks, and 2.0-3.2% at
MCOLS 8. Request-priority addresses cost 0.07 points over the grant-addressed arbiter (+2.00%
against +1.93%). Four slots (P2) cost less than three copies: fewer ports share a slot.

## 2. A pumped TMEM

### The design

`rtl/exp/tmem_pump/otpu_tmem_pump.sv` (in no file list) is `otpu_tmem` with each block RAM
copy run at twice the core clock (`clk_f`, from the same MMCM, both at phase 0, so every core
edge is a fast edge). A pumped copy is LANES banks of one true dual-port RAM on `clk_f`. In each
core cycle a bank does:

- at the core edge (an even fast edge): two reads, port A and port B, from the read requests of
  the cycle that ends there, the same edge at which `otpu_tmem`'s copies read;
- at the fast edge in the middle of the core cycle (odd): port A writes the registered write
  stage's word, the write `otpu_tmem` lands at the next core edge.

So a pumped copy carries two read slots and the write, and the read data are RAM outputs that
hold for the whole next core cycle (port A is NO_CHANGE on writes, port B only reads), as
`otpu_tmem`'s do. A write lands half a core cycle earlier than in `otpu_tmem`, before the read at
that core edge, so the read-during-write bypass is no longer needed (`BYP 0`, the default: no
bypass registers, no address compare, no multiplexer between the RAM and the lanes); the read
data are the same bit for bit either way. Reads (even edges) and writes (odd edges) never meet in
one fast cycle, so the RAM's collision behaviour does not matter.

The phase: each bank has a toggle in the core domain and its copy in the fast domain; they differ
during the first half of a core cycle, which marks the next fast edge as odd. No reset is needed;
it aligns itself from the first core edge.

Timing contract (`rtl/exp/tmem_pump/ooc.tcl`): a bank's read address and enable inputs are
sampled on even fast edges only, a core period after they leave the core domain (a multicycle
path of 2 fast periods from `clk`); the RAM outputs change only after even edges (a multicycle
path of 2 into `clk`); the write stage and the toggle reach the RAM in one fast period (single
cycle at `clk_f`, the real constraint).

Why two reads and not three: a third read in the same copy would have to happen at the odd edge.
Issued there, its request would have to be ready half a cycle early (the requests settle during
the cycle and are captured at its end), or, issued at the next even edge with the others, its
data would land half a cycle late. Either changes the units' timing. Two reads and one write is
what a true dual-port RAM gives at 2x without that (and at 3x or 4x too: "One copy at 3x or 4x?"
below).

Variants (same ports as `otpu_tmem`, `IMPL` as there):

| variant | slots | pumped copies | RAMB36 at 64K | behaviour |
|---|---|---|---|---|
| P3: `IMPL 0` pumped | 6 (the hosts; QUANT 3 and COLL share the DMA's, as today) | 3 | 192 | cycle for cycle `TMEM_IMPL 0`'s |
| `TMEM_IMPL 1` (not pumped) | 3 copies of 1R1W | (3) | 192 | banked: cycles differ where ports meet in a bank |
| P2: `IMPL 1` pumped, NCP 4 | 4 banked slots | 2 | 128 | banked, 4 slots (`TMEM_NCP 4`) |
| P1: `IMPL 1` pumped, NCP 2 | 2 banked slots | 1 | 64 | the quantizer's three reads do not fit (below) |

### What the quantizer and the VPU need

The slice's arbitration groups read several ports at once: the quantizer QUANT, QUANT 2 and
QUANT 3 (the row factor, one lane), the VPU A and B. In a banked TMEM the ports of one group must
sit on different slots, or the group could never be granted. P3 keeps every port's slot (no
change). P2's four slots (`otpu_slice` `TMEM_NCP 4`): MXU and VPU A, VPU B and QUANT 3, DMA and
QUANT 2, QUANT and COLL share a slot; each group is spread, so it fits. P1's two slots cannot
hold the quantizer's three reads: QUANT 3 would have to come off the TMEM ports (the row factor
kept in a register by the quantizer, or read a cycle earlier), the change `docs/asic.md` names
for a two-copy ASIC TMEM. QUANT 3 is one word a row (`rsb + r`, the same for every chunk of the
row), so the register needs one read per row. The co-simulation of P1 (`TMEM_NCP 2`, below) puts
QUANT 3 on a third slot of its own to stand for it.

### Bit-exact

`rtl/exp/tmem_pump/tb_tmem_pump.sv` (`rtl/exp/tmem_pump/run_tb.sh`, Verilator, two clocks) drives
`otpu_tmem` and the pumped TMEM side by side with random traffic under the slice's rules (each
read port's lanes, the write ports' priorities and the registered write stage, the guests and
the banked grants as the slice's arbiter makes them, hot addresses to force read-during-write
and bank conflicts) and compares every read port's data every core cycle. *Measured*: 200,000
cycles each, 1.9-2.6 million lane reads and 0.93 million lane writes, no mismatch, for P3 at 4K
words, at 4K with the bypass kept, at 64K; the banked variants at NCP 3 (two pumped copies, 3 of
their 4 slots) and NCP 4, each with grant addresses and with `RQ` (request-priority addresses,
section 1). The testbench catches broken pumping: with the write data taken from the
wrong register it fails at cycle 8, with the phase inverted (writes at the core edge, reads in
the middle) at cycle 13; with the bypass multiplexer removed it passes (20,000 cycles, 255,800
lane reads, 92,484 lane writes), which is the check that the bypass is redundant here. On
2026-10-12 (44ed40d2, the simulation host) the same eleven configurations pass with the fast-domain write
stage (`FW`) and the one-copy 3x and 4x variants (`K`, compared with `otpu_tmem` a cycle
earlier, grant and `RQ` addresses, 4K and 64K words): 200,000 cycles each, 1.9-2.6 million lane
reads, no mismatch.

### Kintex-7: out of context

`rtl/exp/tmem_pump/tmem_pump_top.sv` wraps one TMEM with registered ports (the slice's port set:
MXU's lanes generic, QUANT 3 and COLL as guests) and both clocks from one MMCME2; `ooc.tcl` runs
synthesis, placement, phys_opt and routing out of context on xc7k480tffg1156-2 (the YPCB's part),
core 133.33 MHz, pumped copies 266.67 MHz, with the contract's multicycle paths. *Measured*
(Vivado 2026.1, the card host, 2026-10-11 for the first build, FW 0; 2026-10-12 for the fast-domain
write stage and phase, FW, c34116bc):

| run | RAMB36 | TMEM LUT | TMEM FF | slices | WNS, FW 0 | its path | WNS, FW |
|---|---|---|---|---|---|---|---|
| `otpu_tmem` IMPL 0, 6 copies (the card's) | 384 | 12,615 | 4,320 | 5,087 | +0.198 | write grant -> write stage address (core) | |
| `otpu_tmem` IMPL 1, 3 copies | 192 | 11,683 | 3,624 | 4,156 | +0.212 | the same | |
| P3: pumped IMPL 0, 3 copies | 192 | 9,879 | 2,720 | 3,915 | +0.037 | phase toggle (core) -> RAM WEA (fast) | **+0.267** |
| P2: pumped, 2 copies | 128 | 10,179 | 2,704 | 3,644 | +0.299 | write stage valid (core) -> RAM WEA (fast) | +0.311 |
| P1: pumped, 1 copy | 64 | 10,172 | 2,688 | 3,596 | +0.313 | RAM -> read data (fast -> core, 2 periods) | +0.366 |
| P3 at 128K words | 384 | 10,169 | 2,728 | 4,196 | +0.085 | write stage data (core) -> RAM DI (fast) | +0.214 |
| `otpu_tmem` IMPL 0 at 128K words | 768 | | | | +0.054 | write grant -> write stage address | |

All seven meet timing out of context; the card's TMEM at 128K takes the 768 RAMB36 the
arithmetic says (of 955 on the part). The pumped TMEM takes half the block RAM at the same
behaviour and fewer LUTs and flip-flops than the card's (P3 against IMPL 0: -22% LUTs, -37%
flip-flops in the TMEM, -23% slices), since the bypass registers and comparators go, and half
the copies take the write fan-out. The block RAM itself is far from its limit at 266.67 MHz (the
pulse-width check: RAMB36E1's minimum period 1.839 ns against 3.75, 1.91 ns of slack). The
tight paths are the ones that cross from the core clock into the fast one in a single fast
period: the write stage and the phase toggle reach the RAM's write port at the odd edge, half a
core cycle after they change. In the first build P3 at 64K had 37 ps there. Reads cross in two
fast periods (the contract), with 0.17-0.73 ns to spare.

Two RTL changes widen that margin, and both are in (`FW`, the default, c34116bc). The phase is
a fast-domain register in each bank, so no core-clock net ends at the RAMs' enables. The write
stage's registers are on `clk_f`, loaded at the even edges only, so their inputs cross from the
core clock in a core period, as the read addresses do, and only one-level fast-domain paths reach
the RAMs. The testbench passes with them (seven configurations, 200,000 cycles each), and fails
with the write data taken from the wrong register (cycle 8) or the phase inverted (cycle 12).
Out of context, P3 goes from +0.037 to **+0.267 ns** at 133.33 / 266.67 MHz. Its tightest path is
now the write stage's data register into the RAM's data pin: one fast period, no logic. With the
core at 150 MHz and the copies at 300, P3 goes from +0.041 to +0.151 ns. 133.33 MHz is not its
edge. (The FW column comes from a second run: the first run's constraint missed the write
stage's registers, whose names carry the generate block, `g_fw.pw_*`, so they were timed in one
fast period: +0.006 ns. `ooc.tcl` matches them now: `OTPU_MCP_PW 381`.)

What out of context does not show: the slice's arbitration and the units around the TMEM (the
full build, section 1), and placement pressure from the rest of the design. The 266.67 MHz clock
is one more MMCM output on the core's MMCM (the YPCB's `clk_wiz` already makes the core clock)
and a BUFG.

### UltraScale+: out of context

the build host's license has no Virtex UltraScale+ part, so these runs use the free xcku5p-ffvb676-2-e
(MMCME4, BUFGCE). It has the BCU1525's and F2's block RAM and UltraRAM primitives (RAMB36E2,
URAM288), and it is a -2 speed grade at VCCINT 0.85 V. The BCU1525's VU9P is -2L: from the data
sheet (DS923), -2L at 0.85 V has -2 timing and is slower at 0.72 V; which voltage the card runs
is to be checked. F2's VU47P is -2. Not modelled: the larger dies, the SLR crossings, and the
slices around the TMEM. *Measured* (Vivado 2026.1, the build host, 2026-10-12):

| run | clocks (core / copies) | RAM | WNS | what limits it |
|---|---|---|---|---|
| P3 in block RAM | 250 / 500 MHz | 192 RAMB36E2 | +0.031 | the write stage's address into the RAM (one fast period) |
| P3 in UltraRAM | 250 / 500 MHz | 48 URAM288 | -0.189 | URAM288's minimum period is 2.000 ns (500 MHz, 0 ps of pulse-width slack); its read data into the core clock |
| P3 in UltraRAM | 300 / 600 MHz | 48 URAM288 | -0.887 | above the URAM's 500 MHz (pulse width -0.333 ns) |
| P3 in block RAM | 200 / 400 MHz | 192 RAMB36E2 | +0.195 | the write stage's data into the RAM |
| P3 in UltraRAM | 200 / 400 MHz | 48 URAM288 | +0.180 | the write stage's data into the URAM's cascade |
| `otpu_tmem` IMPL 1, 3 copies | 250 MHz | 192 RAMB36E2 | -0.122 | RAM to read data through the lanes (one core period) |

RAMB36E2's minimum period is 1.355 ns (738 MHz). So on UltraScale+ -2, block RAM pumps at 2x up to
a ~250 MHz core (P3 meets 250 / 500), and UltraRAM is capped at 500 MHz by its own pulse width, so
a pumped UltraRAM copy needs a core at or below 250 MHz, and then misses by 0.19 ns there. At the
BCU1525's likely 200 MHz core, P3 meets 200 / 400 MHz in both block RAM (+0.195 ns) and UltraRAM
(+0.180 ns).

### One copy at 3x or 4x?

The question: pump a single copy fast enough to carry all the reads, instead of three copies at
2x.

**Slots.** A true dual-port block RAM at K times the core clock has 2K port edges a core cycle.
The write needs one, and any edge will do: it comes from the registered write stage. The reads
are the limit. The units expect a synchronous RAM read (`otpu_tmem`): the request settles during
core cycle t, and the word is used through the whole of cycle t + 1. Only a read at the core edge
gives both sides a full core period: the request has all of cycle t to settle, and the word all
of cycle t + 1. A RAM port's output holds one word, so each port gives one such read per core
cycle, whatever K is. A read at another fast edge takes time from one side. Issued j fast edges
before the core edge, its request has j/K of the core period to settle. Issued j edges after it,
its word arrives j/K late. At 133.33 MHz and 3x that leaves 2.5 or 5.0 ns of the 7.5 ns. The
request side runs through the units' address logic, the rotators and the routing out to the
block RAM columns, and it needs most of the core period today. Two such paths had to be cut to
close 133.33 MHz: VPU B's lane test into the bank address (+0.130 ns of slack before the cut,
`otpu_vpu`) and QUANT 2's bank enables (+0.258 ns, `otpu_quant`). The word side runs from the RAM
through the lanes' rotator into the units' first registers.

So at today's read latency, one copy has two read slots at 2x, 3x or 4x alike; pumping faster
adds only write edges. To get more read slots, the request must be registered. The copy takes
the requests at the core edge, reads them at the K edges of the next core cycle, and presents the
words at the core edge after that, so a read takes 2 cycles instead of 1. Then every port edge
but the write's is a read slot: 2K - 1, so 3, 5 or 7 at 2x, 3x or 4x
(`rtl/exp/tmem_pump/otpu_tmem_pump.sv` `K`, `otpu_pumpk_bank`).

| variant | copies | read slots | read latency | RAMB36 at 64K | what it needs |
|---|---|---|---|---|---|
| P3, 2x | 3 | 6 (QUANT 3 and COLL as the DMA's guests, as today) | 1 | 192 | nothing: cycle for cycle the card's |
| P2, 2x | 2 | 4, banked | 1 | 128 | the banked arbiter |
| P1, 2x | 1 | 2, banked | 1 | 64 | the banked arbiter; QUANT 3 off the ports |
| P1, 3x | 1 | 5, banked | 2 | 64 | the banked arbiter; every unit's read a cycle longer |
| P1, 4x | 1 | 7, banked (COLL on the DMA's) | 2 | 64 | every unit's read a cycle longer |

The worst-case demand in one cycle is every read stream at once: MXU, DMA (with QUANT 3 and COLL
on its copy), QUANT, VPU A, VPU B and QUANT 2, so 6 streams, 7 ports with QUANT 3 on its own,
plus the write. Only P3 and P1 at 4x cover that without a stall. P3 does it at today's latency;
P1 at 4x needs the 2-cycle read.

**How often one copy runs short.** Every co-simulation below counts the TMEM's read demand in
each cycle (`otpu_slice` `TMEMSTAT`, simulation only, commit 6e911192). On the card's TMEM, the
demand is what the units ask for when nothing stalls a read (P3 serves it cycle for cycle). For
each cycle we take the bank with the most read ports asking for it; a pumped copy runs short
when that number is above its slots. *Measured*, the same workloads as section 1, share of the
cycles from reset to the halt:

| workload | cycles | more than 2 (P1 at 2x short) | more than 4 (P2 short) | more than 5 (one copy at 3x short) | most on one bank |
|---|---|---|---|---|---|
| decode, Qwen3-0.6B (28 layers, head) | 3,623,644 | 1.79% | 10 cycles | 0 | 5 |
| decode, Qwen3.5-0.8B (24 layers, head) | 4,760,853 | 1.59% | 0 | 0 | 4 |
| prefill, Qwen3-0.6B, 4 rows, 2 layers | 372,383 | 4.57% | 0.05% | 0 | 5 |
| prefill, Qwen3.5-0.8B, 4 rows, 4 layers | 1,137,275 | 2.42% | 24 cycles | 0 | 5 |
| batched decode, 4 users | 1,697,925 | 1.04% | 0.01% | 0 | 5 |
| batched decode, 6 users | 3,326,827 | 0.81% | 276 cycles | 0 | 5 |
| MoE decode, Qwen3.5-35B-A3B | 5,490,770 | 0.24% | 1 cycle | 0 | 5 |
| MoE decode, gemma-4-26B-A4B-it | 1,613,257 | 0.27% | 0 | 0 | 4 |
| MoE prefill layer run, 35B, 2 rows | 296,391 | 1.55% | 0.22% | 0 | 5 |
| MoE prefill layer run, 26B, 2 rows | 993,612 | 0.88% | 27 cycles | 0 | 5 |
| MCOLS 8: prefill, 4 rows | 267,269 | 7.08% | 0.06% | 0 | 5 |
| MCOLS 8: prefill, 8 rows | 502,925 | 7.18% | 0.08% | 0 | 5 |
| MCOLS 8: batched decode, 6 users | 1,793,140 | 1.53% | 0.02% | 0 | 5 |

No cycle in any of the thirteen workloads has more than five read ports on one bank. So one copy
at 3x (5 slots) would never run short of slots, and neither would 4x (7). Two slots (P1 at 2x)
run short in 0.2-7.2% of the cycles: most in prefill and at MCOLS 8, where the MXU, the
quantizer's two ports and the VPU's two read together, least in MoE decode. Four slots (P2) run
short in at most 0.22% (the 35B's MoE layer run). Counted by stream rather than by bank (any of
the six streams asking, whatever its banks), more than two ask in 0.26-8.6% of the cycles and more
than four in at most 0.23%.

**Cycles.** RTL co-simulation, bit-exact (`--check`, the perf_qwen runs; the MoE runs as in
section 1), the banked arbiter with request-priority addresses (`OTPU_TARGET=nangate45
OTPU_TMEM_RQ=1`, the simulation host, 2026-10-12). P1 at 2x is `OTPU_TMEM_NCP=2`. QUANT 3 sits on a third
slot of its own there, as if the quantizer kept the row factor in a register, which is the change
P1 needs. That change's own cost is one more cycle per row the quantizer scales. The 2-cycle
read's column is an *estimate*, one cycle per instruction (the run's instruction count over its
cycles): a unit's first word comes a cycle later, and so does a dependent instruction's start.
The pipelines would hide some of it and an instruction that reads in two passes would pay twice.
Core cycles, and against the card's TMEM:

| workload | the card's TMEM (= P3) | P2, 2x: 4 slots | P1, 2x: 2 slots | 5 slots (3x), today's latency | 7 slots (4x), today's latency | the 2-cycle read, estimate |
|---|---|---|---|---|---|---|
| decode, Qwen3-0.6B (28 layers, head) | 3,623,644 | +0.51% | +0.43% | -0.03% | -0.03% | +0.47% |
| decode, Qwen3.5-0.8B (24 layers, head) | 4,760,853 | +0.01% | +0.19% | -0.03% | -0.03% | +0.24% |
| prefill, Qwen3-0.6B, 4 rows, 2 layers | 372,383 | +1.60% | +1.89% | +0.42% | +0.27% | +0.88% |
| prefill, Qwen3.5-0.8B, 4 rows, 4 layers | 1,137,275 | +0.08% | +0.31% | -0.04% | -0.03% | +0.26% |
| batched decode, 4 users | 1,697,925 | +0.30% | +0.41% | +0.00% | +0.04% | +0.22% |
| batched decode, 6 users | 3,326,827 | +0.25% | +0.30% | +0.02% | +0.01% | +0.17% |
| MoE decode, Qwen3.5-35B-A3B | 5,490,770 | +0.03% | +0.03% | +0.03% | +0.03% | +0.02% |
| MoE decode, gemma-4-26B-A4B-it | 1,613,257 | -0.01% | +0.00% | -0.01% | -0.01% | +0.08% |
| MoE prefill layer run, 35B, 2 rows | 296,391 | +0.66% | +1.13% | +0.03% | +0.03% | +0.21% |
| MoE prefill layer run, 26B, 2 rows | 993,612 | +0.06% | +0.18% | -0.02% | -0.02% | +0.09% |
| MCOLS 8: prefill, 4 rows | 267,269 | +1.99% | +2.43% | +0.01% | +0.11% | +1.23% |
| MCOLS 8: prefill, 8 rows | 502,925 | +2.09% | +2.44% | -0.19% | -0.18% | +1.21% |
| MCOLS 8: batched decode, 6 users | 1,793,140 | +0.36% | +0.42% | -0.01% | -0.01% | +0.31% |

P1 at 2x costs 0.0-0.4% in decode, batched decode and MoE decode, 0.2-1.9% in prefill and MoE layer
runs, and 2.4% at MCOLS 8. That is 0.0-0.5 points more than P2's two copies. The unit that waits is
the quantizer. With two slots it loses 15.6% of the cycles to the arbitration in Qwen3-0.6B's
prefill and 20.8-22.9% at MCOLS 8 (3.4-4.0% on the card's TMEM). The MXU loses 1-2.6 points
more. Most of those waits overlap the MXU's streaming. So in cycles, two read slots would be
nearly enough. What keeps P1 at 2x off the FPGA is its arbiter: it needs the banked arbiter,
whose full build misses 133.33 MHz by 1.06 ns (section 1), plus QUANT 3 off the ports. P3 needs
neither.

Five or seven slots at today's latency come within -0.19% to +0.42% of the card's TMEM. The
negative values are probably the banked arbiter's own order: QUANT 3 waits only where the DMA
reads its bank, not whenever the DMA reads (the guest rule of the card's TMEM). So the slots are
not what one copy at 3x or 4x would cost. The 2-cycle read is, and its estimate (+0.02% to +1.2%)
is larger than every slot cost except P1's and P2's in prefill.

**Timing.** `VAR 5` of `tmem_pump_top` is one copy pumped at 3x or 4x (`otpu_pumpk_bank`):
requests registered at the core edge, the next fast edge's address and enable registered one
fast edge ahead (a 2:1 multiplexer before the RAM), each read's word taken at the next fast
edge, and all words presented at the core edge. The testbench checks it against `otpu_tmem` a
cycle later (`run_tb.sh` `k3n5`, `k4n7` and their `RQ` cases). It fails with the bypass removed,
with a word taken at the wrong edge, or with the slots read in the wrong order. Out of context,
*measured* (Vivado 2026.1; Kintex-7 on the card host, xcku5p on the build host, 2026-10-12):

| part | core / copy clock | RAM | WNS | what limits it |
|---|---|---|---|---|
| xc7k480t-2 (YPCB) | 133.33 / 400 MHz, 3x | 64 RAMB36E1 | **-0.027** | the RAM's clock-to-out (1.800 ns, no output register) into the next register |
| xc7k480t-2 | 133.33 / 533.33 MHz, 4x | 64 RAMB36E1 | -0.657 | the same; RAM pulse width +0.036 ns |
| xcku5p-2 | 133.33 / 400 MHz, 3x | 64 RAMB36E2, cascaded | +0.064 | three cascaded RAMB36E2 in the RAM-to-register path |
| xcku5p-2 | 133.33 / 400 MHz, 3x | 64 RAMB36E2, uncascaded (`CASC 1`) | **+0.441** | the next edge's address into the RAM |
| xcku5p-2 | 133.33 / 400 MHz, 3x | 16 URAM288 | -0.518 | the UltraRAM's clock-to-out through its two-URAM cascade (2.32 ns of a 2.85 ns path) |
| xcku5p-2 | 133.33 / 533.33 MHz, 4x | 64 RAMB36E2, cascaded | -0.536 | the cascade |
| xcku5p-2 | 133.33 / 533.33 MHz, 4x | 64 RAMB36E2, uncascaded | **+0.097** | the RAM's clock-to-out (0.947 ns) into the next register |
| xcku5p-2 | 133.33 / 533.33 MHz, 4x | 16 URAM288 | -1.156 | above the UltraRAM's 500 MHz (pulse width -0.125 ns) |
| xcku5p-2 | 200 / 600 MHz, 3x | 64 RAMB36E2, cascaded | -0.680 | the cascade |
| xcku5p-2 | 200 / 600 MHz, 3x | 64 RAMB36E2, uncascaded | **+0.002** | the RAM's clock-to-out into the next register |
| xcku5p-2 | 200 / 600 MHz, 3x | 16 URAM288 | -1.285 | above the UltraRAM's 500 MHz (-0.333 ns) |
| xcku5p-2 | 200 / 800 MHz, 4x | 64 RAMB36E2, cascaded / uncascaded | -1.122 / -0.334 | above the block RAM's 738 MHz (pulse width -0.105 ns) |
| xcku5p-2 | 200 / 800 MHz, 4x | 16 URAM288 | -1.732 | above the UltraRAM's 500 MHz (-0.750 ns) |

On the YPCB's Kintex-7 -2, 3x at 133.33 / 400 MHz misses by 27 ps. All the failing paths (20
endpoints, -0.261 ns in total) are a block RAM's output into the register that takes its word
one fast period later. That path has no logic: the RAM's clock-to-out without its output register
is 1.800 ns, and routing to the next slice adds 0.56 ns, against the 2.5 ns period. Everything
else meets: the requests' crossing has +0.36 ns and the core-domain paths +1.4 ns. The RAM's own
minimum period is 1.839 ns (0.661 ns of slack at 400 MHz). 4x at 533.33 MHz cannot work on this
part. The same clock-to-out leaves 75 ps of the 1.875 ns period (-0.657 ns, 2,623 endpoints),
the RAM's pulse width has 36 ps, and even the phase's single-wire crossing from the core clock
misses (-0.085 ns). Using the RAM's output register would take the clock-to-out out of the path,
but then the word from the last fast edge arrives after the core edge, so 4x would give 5 read
slots, the same as 3x.

On UltraScale+ -2 the block RAM is twice as fast out: its clock-to-out without the output
register is 0.947 ns against Kintex-7's 1.800 ns, and its minimum period is 1.355 ns (738 MHz)
against 1.839 ns. Vivado's first choice cascades the bank's RAMB36E2, which puts the cascade
(0.25 ns a hop) in the one-fast-period path. Kept out of a cascade (`CASC 1`, `cascade_height 1`),
one copy meets 3x at 133.33 / 400 MHz (+0.441 ns) and 4x at 133.33 / 533.33 MHz (+0.097 ns). It
meets 3x at 200 / 600 MHz with 2 ps to spare, which is no margin for a full build. 4x at a 200 MHz
core (800 MHz) is above the block RAM's own limit. UltraRAM never gets there: its minimum period is
2.000 ns (500 MHz) and its clock-to-out through the two-URAM cascade of an 8K-word bank is
2.3 ns, so a pumped UltraRAM copy works at 2x only, with the core at or below 250 MHz (P3 in
UltraRAM meets 200 / 400 MHz, +0.180 ns, above).

So one copy at 4x is possible on the UltraScale+ boards, in block RAM, with a 133.33 MHz core,
and one copy at 3x is possible with a core up to about 200 MHz. On the YPCB's Kintex-7 neither
is. Wherever it is possible, it carries the 2-cycle read: every unit's read path one stage
longer, so the units are no longer the card's cycle for cycle.

## 3. More words

### What a run of R rows needs

`tools/tmem_capacity.py need` compiles a rows-run (`compile_rows`, the host's prefill and
batched-decode runs) from a model's `config.json` alone, and bisects the TMEM size in 1K-word
steps to the least that compiles; it also counts the program's instructions (IMEM holds 4,096).
Settings: fp4 weights with an int8 head, PAIR, DSTEP, STREAM (the card's), a prompt's last run
(logits for its last row), positions 300 (bucket 2) and 3,000 (bucket 12). *Measured* (the
compiler); the instruction counts barely move with the TMEM size (Qwen3-0.6B, R = 4..8 at
bucket 12: the same within 1% at 64K, 128K and 256K).

MCOLS 4 (the card). The 64K..384K columns are the most rows a run takes by TMEM alone, the
IMEM columns the most by the instruction count alone (rows probed: 1, 2, 3, 4, 5, 6, 8, 10, 12,
16):

| model | words a row | bucket | R = 1 needs | 64K | 128K | 256K | 384K | IMEM 4096 | IMEM 8192 |
|---|---|---|---|---|---|---|---|---|---|
| LFM2.5-230M | 6.7K | 2 | 12K | 8 | 16 | 16 | 16 | 10 | 16 |
| LFM2.5-230M | 6.3K | 12 | 13K | 8 | 16 | 16 | 16 | 4 | 10 |
| Qwen3-0.6B | 6.0K | 2 | 13K | 8 | 16 | 16 | 16 | 12 | 16 |
| Qwen3-0.6B | 6.3K | 12 | 14K | 8 | 16 | 16 | 16 | 5 | 10 |
| Qwen3.5-0.8B | 9.7K | 2 | 18K | 6 | 12 | 16 | 16 | 8 | 16 |
| Qwen3.5-0.8B | 8.3K | 12 | 22K | 6 | 12 | 16 | 16 | 6 | 16 |
| Qwen3.5-2B | 12.7K | 2 | 21K | 4 | 8 | 12 | 12 | 10 | 12 |
| Qwen3.5-2B | 11.7K | 12 | 24K | 4 | 8 | 12 | 12 | 6 | 12 |
| gemma-4-E2B-it | 10.3K | 2 | 20K | 5 | 8 | 12 | 12 | 6 | 12 |
| gemma-4-E2B-it | 8.7K | 12 | 24K | 5 | 8 | 12 | 12 | 4 | 10 |
| LFM2-2.6B | 13.7K | 2 | 24K | 3 | 8 | 12 | 12 | 3 | 8 |
| LFM2-2.6B | 13.0K | 12 | 25K | 4 | 8 | 12 | 12 | 1 | 3 |
| SmolLM3-3B | 9.3K | 2 | 17K | 6 | 10 | 12 | 12 | 12 | 12 |
| SmolLM3-3B | 8.7K | 12 | 23K | 5 | 10 | 12 | 12 | 8 | 12 |
| Qwen3.5-4B | 13.7K | 2 | 27K | 3 | 6 | 8 | 8 | 6 | 8 |
| Qwen3.5-4B | 13.3K | 12 | 31K | 3 | 6 | 8 | 8 | 4 | 8 |
| Phi-4-mini-instruct | 18.3K | 2 | 21K | 3 | 6 | 8 | 8 | 8 | 8 |
| Phi-4-mini-instruct | 15.3K | 12 | 23K | 3 | 6 | 8 | 8 | 5 | 8 |
| gemma-4-E4B-it | 13.0K | 2 | 21K | 4 | 6 | 12 | 12 | 6 | 12 |
| gemma-4-E4B-it | 11.7K | 12 | 25K | 4 | 6 | 12 | 12 | 3 | 8 |

A run takes the smallest of the TMEM column, the IMEM column and the ACT RAM's limit. The ACT
RAM caps MCOLS 4 at 8 rows (Qwen3.5-4B, Phi-4-mini: R = 10 fails with "ACT RAM full") or 12 (the
other models from 2B up: R = 16 fails), which is why the 256K and 384K columns stop growing. At
MCOLS 8 (the ACT RAM's rows double) the TMEM column reaches 16 rows at 256K for every model (12
for Qwen3.5-4B, Phi-4-mini and gemma-4-E4B-it), and 64K holds the same rows as at MCOLS 4
(gemma-4-E2B-it 4 and 2).

Three things follow:

- **64K holds 3 to 8 rows.** Each row past the first costs 6K (the 0.6B models) to 18K words
  (Phi-4-mini). The three largest dense models (LFM2-2.6B, Qwen3.5-4B, Phi-4-mini) fit 3 rows,
  less than one MCOLS 4 pass of int8 rows and less than a MCOLS 8 pass of fp4 rows.
- **IMEM binds at long contexts.** Attention is unrolled per row, head and block, so the program
  grows with the context: at bucket 12, 4,096 instructions hold 1 (LFM2-2.6B) to 8 rows, fewer
  than 64K of TMEM does for half the models. More TMEM without more IMEM pays at short contexts
  only. Doubling IMEM costs 28.5 RAMB36 tiles on the YPCB (*estimate*: today's 28 + 1 RAMB18,
  twice).
- **2x is where the step is.** 128K lifts every model to 6-16 rows by TMEM; past 128K the ACT
  RAM (MCOLS 4) or IMEM binds first, so 4x and 6x add rows only together with a larger ACT RAM
  and IMEM, or at MCOLS 8.

Batched decode (every row its logits) needs the same words a row (the head runs a row at a
time), and its programs are longer (each row's head and pick): IMEM 4,096 holds 3-12 rows at
bucket 2 (4-6 for most) and 1-6 at bucket 12 (`tools/tmem_capacity.py need ... --logits all`).
Eight users in one run fit 64K by TMEM only for the two smallest models, and IMEM for none at
bucket 12. At bucket 2, 128K and 8,192 instructions fit 8 users for seven of the ten models (not
Qwen3.5-4B, Phi-4-mini, gemma-4-E4B-it: 6 rows by TMEM); 256K fits all ten.

### What the rows buy

A pass of the MXU computes MCOLS rows of int8 weights, or MCOLS / 2 rows of PAIRed fp4 weights,
so at MCOLS 4 an fp4 run fills the MXU from 2 rows. Rows past a pass save weight traffic (each
weight chunk streams once a run and is replayed from the chunk FIFO) and the run's fixed costs,
not MXU time. RTL co-simulation (LiteDRAM co-sim, DDR3-1066, 133.33 MHz, PAIR/DSTEP/STREAM,
Qwen3-0.6B fp4 with an int8 head, 2 layers, position 544) *measured*, core cycles a row:

| rows a run | MCOLS 4 | MCOLS 8 |
|---|---|---|
| 2 | 100,131 | 100,481 |
| 3 | 113,363 | 77,719 |
| 4 | 93,096 | 66,817 |

(R = 3 costs a fourth row at MCOLS 4: PAIR runs rows in pairs.) At MCOLS 4 going from 2 to 4
rows saves 7% a row; at MCOLS 8 it saves 33%.

The same at larger TMEMs, for the models 64K holds to 3 rows, *measured* on the RTL with the
sequencer's ranges widened (4eca9a76, below: without it a co-sim above 64K serializes and reads
pessimistic): `tools/scoreboard_cosim.py run` (workloads branch, e769bf79; config.json only, zero
DRAM image, the card's TMEM), LiteDRAM co-sim at 133.33 MHz, fp4 with an int8 head, a prompt run
without logits at position 544, Qwen3.5-4B's first 4 layers (3 DeltaNet + 1 attention),
Phi-4-mini's and Qwen3-0.6B's first 2. Core cycles a row, at the most rows each TMEM takes (in
brackets the other row count tried):

| model | MCOLS | 64K | 128K | 256K |
|---|---|---|---|---|
| Qwen3.5-4B | 4 | R 2: 1,112,600 (R 3: 1,289,866) | R 4: 999,704 (-10.1%) | |
| Qwen3.5-4B | 8 | R 3: 802,354 (R 2: 1,113,048) | R 4: 677,628 (-15.5%) | R 8: 640,516 (-20.2%) |
| Phi-4-mini | 4 | R 2: 471,319 (R 3: 559,589) | R 4: 428,707 (-9.0%) | |
| Phi-4-mini | 8 | R 3: 327,076 (R 2: 471,440) | R 4: 254,619 (-22.2%) | R 8: 233,048 (-28.7%) |
| Qwen3-0.6B | 4 | R 8: 91,002 (R 4: 92,536) | R 16: 91,010 (0.0%) | |
| Qwen3-0.6B | 8 | R 8: 62,539 (R 4: 66,704) | R 16: 60,746 (-2.9%) | |

The TMEM's size alone changes nothing (Qwen3.5-4B at 2 rows: 1,112,600 at 64K, 1,112,553 at
128K; Qwen3-0.6B at 8 rows: 91,002 and 91,132). So 2x TMEM takes 9-10% of the cycles off a
prompt row at MCOLS 4 for the large models (+10-11% speed) and nothing for the small ones, and
15-22% at MCOLS 8 (+18-29%: a whole fp4 pass of 4 rows instead of 3); 4x takes another 5.5-8.5%
there. These are layer cycles: the LM head of a prompt's last run and the per-run host costs
come on top, so a whole prompt gains less (the scoreboard's +9.6% for Qwen3.5-4B at 1,024
tokens, MCOLS 4).

Batched decode: 6 users cost more a row than 4 at MCOLS 4 (554,471 against 424,481 cycles a
row, Qwen3-0.6B, 2 layers, each row's logits): the int8 head runs MCOLS rows a pass, so 6 rows
take two passes of the 155 MB head. Batches want multiples of MCOLS; at MCOLS 8, 6 users take
298,857 a row. So on the card's MXU the TMEM's capacity is worth
little for the models that already reach 4 rows; it matters for the three that stop at 3 (in
fp4 they run 2, since PAIR makes an odd R cost the next even one) and for a wider MXU.

The workloads scoreboard (`tools/scoreboard.py` on the workloads branch, an analytic model
calibrated on the card and the co-sim; *estimate*), prefill of a 1,024-token prompt, geometric
mean over nine dense models, against the board as it is:

| what-if | YPCB | BCU1525 | F2 |
|---|---|---|---|
| TMEM 128K | +4.4% | +6.0% | +9.5% |
| TMEM 256K | +5.6% | +8.0% | +13.4% |
| TMEM 384K | +9.2% | +14.3% | +26.5% |
| MCOLS 8 | +35.9% | +30.0% | +23.6% |
| MCOLS 8, TMEM 128K | +52.5% | +45.7% | +39.7% |
| MCOLS 8, TMEM 256K | +60.7% | +53.7% | +48.2% |

TMEM changes neither single-user decode (a decode step needs well under 64K) nor, in that model,
batched decode (it runs users in blocks of a pass); one 8-row batched run instead of two 4-row
runs is +4.6% at MCOLS 4 (`--batch-rows 8`), but eight users' programs need IMEM x2 (above).

### What more than 64K words takes besides the RAM

- **STREAM descriptors.** The compiler puts them at the top of TMEM (`STREAM_DESC_AREA`, the
  last 64 words), and the STREAM instruction's descriptor field is 16 bits (`isa.stream`,
  `otpu_dma`'s `sc.w1[15:0]`): above 64K the descriptors land out of its reach, and the programs
  that use STREAM (`state_step`, and `deltanet_step` into another state, MTP's) fail to compile.
  4eca9a76 keeps the area at the top of the first 64K when TMEM is larger (a region no tile
  takes; at 64K nothing moves, so today's programs are unchanged); widening the field instead
  would be an ISA change. The rows-runs probed above do not stream, so they compiled either way.
- **The sequencer's hazard ranges.** `otpu_seq` keeps an instruction's TMEM ranges in
  TAW = 16 bits; a range at or above word 65,536 sets `all`, and the instruction then waits like
  a BAR. A program with tiles above 64K runs correctly but serialized: the tiling study measured
  Qwen3-0.6B at R = 4 (2 layers, MCOLS 4) at 83.6K cycles a row with TMEM 64K and 112.7K with
  256K, the same program (branch `tiled-prefill`, `tools/perf_tiled.py`). 4eca9a76 sets
  TAW = max(16, log2 TMEM_WORDS) from a parameter `otpu_slice` passes (16 at 64K: the card's
  build unchanged); RTL co-sims above 64K need it too (section 3's ran with it).
- The register map's CFG_TMEM (0x084) tells the host the size, and `device_config` takes it from
  the bitstream, so programs follow the build.
- The bank index stays addr mod 8, so only the banks' depth changes.

### MoE

More TMEM does not change the MoE models' runs today (*measured*, the compiler):

- A layer-major prefill run takes at most 2 rows (`compile_layer_run` fails "3 rows a layer
  run" for both Qwen3.5-35B-A3B and gemma-4-26B-A4B-it at any TMEM size): the expert request line
  holds R x k <= 16 ids. Its programs are short (366-907 instructions at 1-2 rows).
- The expert-major prefill (dormant, `docs/offload.md` 13.11) keeps its 512-row chunk in DRAM and
  runs each expert's rows in passes of two; its TMEM holds the tables, not the rows.
- Decode is one row.

What more rows a run would buy, if the request line grew (*estimate*, from the 35B's card
routes, 37 tokens x 40 layers, and a Zipf popularity model fitted per model to the measured
per-layer union of a whole prompt, 144 of 256 experts over 134 tokens for the 35B and 75 of 128
over 124 for the 26B, `docs/offload.md` 13.7):

| rows a run | 35B experts streamed a row | 35B MXU passes a row, MCOLS 4 / 8 | 26B experts a row | 26B passes a row, MCOLS 4 / 8 |
|---|---|---|---|---|
| 1 | 8.0 | 8.0 / 8.0 | 8.0 | 8.0 / 8.0 |
| 2 | 6.3 | 6.3 / 6.3 | 5.7 | 5.7 / 5.7 |
| 4 | 5.3 | 5.9 / 5.3 | 4.1 | 5.3 / 4.1 |
| 8 | 4.2 | 5.4 / 4.4 | 2.9 | 4.9 / 3.5 |
| 16 | 3.3 | 5.1 / 3.9 | 2.0 | 4.7 / 3.1 |
| 32 | 2.5 | 4.8 / 3.4 | 1.4 | 4.5 / 2.7 |
| 512 (a chunk, expert-major) | 0.44 | 4.1 / 2.2 | 0.22 | 4.1 / 2.1 |

(35B rows 1-32 from the card routes, the chunk from the model; 26B from the model.) The weight
traffic a row (and, for experts not on the card, the PCIe traffic) falls with the rows, the MXU
passes much less at MCOLS 4. That is the expert-major prefill's case, which already has the
rows; a bigger TMEM does not add to it. For MoE the levers are the request line (more ids a
run) and MCOLS, not TMEM words.

## 4. Per board

### What each board can hold

TMEM block RAM by variant and size, 8K x 32 bits a RAMB36 column of 8 (*estimate*: the
arithmetic of the measured 384 RAMB36 at 64K x 6 copies):

| TMEM | 6 copies (today) | 3: P3 or `TMEM_IMPL 1` | 2: P2 | 1: P1 |
|---|---|---|---|---|
| 64K | 384 | 192 | 128 | 64 |
| 128K | 768 | 384 | 256 | 128 |
| 256K | 1,536 | 768 | 512 | 256 |
| 384K | 2,304 | 1,152 | 768 | 384 |

**YPCB (xc7k480t, 955 RAMB36 tiles).** Everything but the TMEM takes 218.5 tiles today. A
larger TMEM also wants IMEM x2 (+28.5, section 3); MCOLS 8 doubles the ACT RAM (+64, *estimate*:
today's 64 RAMB36 at ACT_ROWS 4). Totals (tiles, of 955):

| TMEM | at MCOLS 4: 6 copies | P3 | P2 | at MCOLS 8: 6 copies | P3 | P2 |
|---|---|---|---|---|---|---|
| 64K | 602.5 (today) | 410.5 | 346.5 | 666.5 | 474.5 | 410.5 |
| 128K | 1,015 (no) | 631 | 503 | 1,079 (no) | 695 | 567 |
| 256K | no | 1,015 (no) | 759 | no | 1,079 (no) | 823 |
| 384K | no | no | 1,015 (no) | no | no | 1,079 (no) |

So with six copies the YPCB cannot have a larger TMEM at all; with three (P3, or a banked
TMEM) it can have 128K at either MCOLS, and with two (P2) 256K.

**BCU1525 (VU9P, a slice per SLR: 720 RAMB36 and 320 URAM288 each).** Its build already moves
the TMEM to UltraRAM (`TMEM_URAM=1`, `docs/bcu1525.md`): 48 URAM288 a slice in yosys' packing
(two 32-bit words a 72-bit row), 96 if Vivado cascades two a bank. 128K takes 96 (192), 256K 192
(384), 384K 288 (576) with six copies: up to 384K fits an SLR's 320 URAM288 in yosys' packing
without pumping, 128K in Vivado's. UltraRAM pumps at 2x only: URAM288's minimum period is
2.000 ns at -2 (500 MHz, measured above), so P3 in UltraRAM needs the core at or below 250 MHz;
it meets 200 / 400 MHz.

**F2 (VU47P: 2,016 RAMB36 and 960 URAM288 in three SLRs, 672 and 320 each, from the data
sheet; not built).** The scoreboard's F2 runs S = 4 slices, so one SLR holds two: 160 URAM288 a
slice. Six copies fit 128K (96 a slice in yosys' packing), three fit 256K (96) and 384K (144).

**ASIC.** The TMEM is banked already (`TMEM_IMPL 1`, three copies of 1R1W macros). Pumping is an
FPGA trick: an SRAM macro at twice the core clock would cap the core at half the macro's speed.
Capacity is macro area: in the card's arch on nangate45 the TMEM is 4.05 mm2 (48 macros of
4096 x 32), 38% of the synthesized area (`docs/asic.md`), so 2x the words adds about 4 mm2
(*estimate*). Two copies with QUANT 3 off the ports would save a third: 128K at two copies is
about 5.3 mm2 (+1.3), against 8 mm2 (+4) at three.

### The options on the four workloads

Through the workloads scoreboard (commit e769bf79, *estimate*), each against its board as it is;
a banked TMEM's measured cycle cost (section 1) applied on top. Decode = single-user decode,
prefill = a 1,024-token prompt, batched = 4-16 users, MoE = the 26B and 35B (geometric means
over the scoreboard's models):

| option, YPCB | decode | prefill | batched | MoE |
|---|---|---|---|---|
| P3 at 64K (frees 192 RAMB36) | 1.000 | 1.000 | 1.000 | 1.000 |
| banked RQ or P2 at 64K (measured, section 1) | 0.995-0.999 | 0.980-0.999 | 0.996-0.997 | 0.990-1.000 |
| TMEM 128K (P3) | 1.000 | 1.042 | 1.000 (1.046 at 8 rows a run) | 1.000 |
| TMEM 256K (P2) | 1.000 | 1.053 | 1.000 (1.046) | 1.000 |
| MCOLS 8 at 64K | 1.001 | 1.359 | 1.733 | 1.003 |
| MCOLS 8, TMEM 128K (P3) | 1.001 | 1.525 | 1.733 | 1.003 |
| MCOLS 8, TMEM 256K (P2) | 1.001 | 1.607 | 1.733 | 1.003 |
| MXU pumped x2, TMEM 128K (P3) | 1.000 | 1.470 | 1.306 (1.379) | 1.000 |

The batched column is 4-16 users in runs of a pass (MCOLS rows); "8 rows a run" is one run of 8
users at MCOLS 4. The same what-ifs on the BCU1525 give prefill 1.057 (128K), 1.300 (MCOLS 8) and
1.457 (both), on the F2 1.095, 1.236 and 1.397. The RTL co-sims of section 3 agree in kind:
128K at MCOLS 4 is +10-11% for a large model's prompt rows and nothing for a small one; at MCOLS
8 it is +18-29%.

## Recommendation

**The fewest copies per board**, holding timing at a negligible cycle cost:

| board | fewest copies | why not fewer |
|---|---|---|
| YPCB (xc7k480t-2, 133.33 MHz) | **3, pumped at 2x (P3)**: the card's behaviour cycle for cycle; +0.267 ns out of context at 133.33 / 266.67 | Two copies or one at 2x are banked: 0.0-2.4% in cycles (measured), but the banked arbiter misses 133.33 MHz by 1.06 ns in the full build (b2). One copy at 3x misses by 27 ps out of context (the block RAM's 1.800 ns clock-to-out) and needs the 2-cycle read. 4x misses by 0.66 ns. |
| BCU1525 (VU9P -2L; measured on xcku5p-2) | **3 (P3)** at a 200 MHz core, in block RAM (+0.195 ns) or UltraRAM (+0.180 ns) at 200 / 400 | One copy at 3x and 200 MHz meets by 2 ps in uncascaded block RAM (no margin), needs the 2-cycle read, and needs block RAM, while the board's TMEM is in UltraRAM, which stops at 500 MHz. |
| F2 (VU47P -2) | **3 (P3)**, as the BCU1525; one copy at 4x only with a 133.33 MHz core | One copy at 4x meets 133.33 / 533.33 (+0.097 ns, uncascaded block RAM), but only with the 2-cycle read (estimate +0.02-1.2% in cycles, a change to every unit). |
| ASIC | **3 banked 1R1W copies** (as today); 2 with QUANT 3 off the ports | Pumping does not apply: an SRAM macro at K times the core clock caps the core at 1/K of the macro's speed. |

So the question "pump TMEM by 3x or 4x so it needs one copy instead of 3?" has these answers:
- **At the units' current read latency: no, at any K.** A copy gives two reads a core cycle (one
  per RAM port) however fast it is pumped, because each read's word must hold for the whole next
  core cycle. Faster pumping only adds write edges.
- **With a 2-cycle read**, one copy has 2K - 1 read slots. Five slots (3x) were never short in
  thirteen workloads, since no cycle has more than five ports on one bank. In timing, the YPCB
  cannot do it (3x misses by 27 ps out of context, so a full build would miss by more; 4x misses
  by 0.66 ns). The UltraScale+ parts can do 4x at a 133.33 MHz core and 3x at about 200 MHz, in
  block RAM. The cost is a read path one stage longer in every unit (MXU, VPU, quantizer, DMA,
  collective, stream engine); co-simulation cannot measure it until those units are changed.
  *Estimate*: up to one cycle per instruction, +0.02-1.2%.
- **Three pumped copies (P3)** are the fewest that keep the card's behaviour: half the block RAM
  of today's six, no cycle cost, no change outside the TMEM and the clock.

**YPCB, against today's MXU.** P3 frees 192 RAMB36 (20% of the part) and 22% of the TMEM's LUTs
at the same behaviour. On today's MXU that room buys little: a 128K TMEM is +4.2-4.4% prefill
(scoreboard), +10% a prompt row for the three models that stop at 3 rows (Qwen3.5-4B measured:
1,112,600 -> 999,704 cycles a row, 2 to 4 rows), and nothing for decode, batched decode or MoE.
Not worth a production change on its own.

**YPCB, against the MX column** (dsp-pump, approved for production: MCOLS 4 first, then 8 rows
with NVFP4 on the pumped DSP column; it frees about 31% of the MXU's 24K LUTs and 264 DSPs). At 8
rows the ACT RAM doubles (+64 RAMB36, *estimate*). With six copies that is 666.5 block RAM tiles
of 955 at 64K, and a 128K TMEM does not fit (1,079). With P3 it is 474.5 at 64K and 695 at 128K
with IMEM x2. MCOLS 8 with 128K is +52.5% prefill against +35.9% for MCOLS 8 alone (scoreboard),
and +18-29% a prompt row for the large models (co-simulation, section 3). The pumped DSP column
already needs a 2x clock (266.67 MHz), and P3 runs on the same one. So **fuse P3 and a 128K TMEM
into the MX column's MCOLS 8 build**. P3 is what makes 128K fit there, and the second clock is
paid once.

What P3 still needs to be production-worthy:
1. The two timing changes: **done** (`FW`, c34116bc). Out of context +0.037 -> +0.267 ns at
   133.33 / 266.67, +0.151 ns at 150 / 300.
2. Out of `rtl/exp` into the build: `otpu_slice` instantiates the pumped TMEM behind a knob
   (`TMEM_PUMP`); `clk_f` through `otpu_slice`, `otpu_top`, `otpu_board`, `otpu_native_sys` and
   `otpu_fpga_top_ld`; a 2x output on the block design's `clk_wiz` with its BUFG; `ooc.tcl`'s
   multicycle paths in the board's XDC (read addresses in, RAM outputs out, the write stage in).
   Not done.
3. Tests: `run_tb.sh` in the RTL gate (P3 against `otpu_tmem`; its eleven configurations pass,
   200,000 cycles each); a `tb_top` with the second clock for a bit-exact co-simulation smoke
   (its cycles must equal `TMEM_IMPL 0`'s); the RTL gate on the build host.
4. A full YPCB build at production settings that closes 133.33 MHz (not run: out of context
   only).
5. A card qualification: BIST, token-exact against production, and decode and prefill speed
   equal to production's (cycle for cycle).

P3 and the MX column are independent in RTL. P3 is the TMEM (`rtl/mem`, today
`rtl/exp/tmem_pump/otpu_tmem_pump.sv`) and its instance in `otpu_slice`. The MX column is
`otpu_mxu` (IMPL 3), `rtl/tech`'s DSP cells and `otpu_quant`'s X mode. They touch the same files
in one place: the 2x clock's plumbing, which is `otpu_slice`'s and `otpu_top`'s ports,
`otpu_board`, `otpu_native_sys`, the YPCB top, the block design's `clk_wiz` and the XDC. Whichever
lands first adds the clock and the other uses it. Conflicts are limited to those port lists.

The banked TMEM is not the FPGA's answer. As the ASIC has it the build misses by 3.27 ns (the grant
into the block RAMs' addresses). With request-priority addresses it misses by 1.06 ns, spread over
the slice. Both cost 0.0-3.2% in cycles.

**BCU1525.** Its TMEM is in UltraRAM already (`TMEM_URAM=1`). An SLR has room for 128K with six
copies in Vivado's cascade (192 of 320 URAM288) and up to 384K in yosys' packing, so the copies do
not bind until 256K. UltraRAM pumps at 2x only (500 MHz at -2), with the core at or below 250 MHz.
At a 200 MHz core P3 in UltraRAM meets 400 MHz (+0.180 ns) and halves the URAM. Take P3 when the
TMEM grows to 256K (with MCOLS 8: +54% prefill on the board, scoreboard). One copy at 3x would
have to move the TMEM back to block RAM and take the 2-cycle read.

**F2.** Two slices an SLR: 160 URAM288 a slice. Six copies hold 128K; 256K and 384K need P3 in
UltraRAM (a core at or below 250 MHz), or block RAM.

**ASIC.** Keep the banked TMEM (three copies of 1R1W macros; the measured cycle cost is
0.1-3%). Its listed critical paths are elsewhere (`docs/asic.md`: dispatch, and the grant into
the units' enables, which `TMEM_RQ` does not remove), so `TMEM_RQ` is optional there; check it in
the next ORFS run if the grant-to-address path shows. Capacity is area: 128K at three copies
adds ~4 mm2 (+38% of the card's synthesized area), so take QUANT 3 off the ports first. Two
copies give 128K for +1.3 mm2, and P1's co-simulation says two read slots cost 0.0-2.4% in cycles.

## Open risks

- P3 has not been in a full build: placement pressure and the second clock domain may cost more
  than out of context shows. Its tightest OOC paths at 133.33 / 266.67 MHz now have +0.214 (128K)
  to +0.267 ns (64K), the write stage into the RAM inside the fast domain, after the two timing
  changes.
- One copy at 3x or 4x was timed out of context and checked in the testbench, never in a
  co-simulation with the units: its 2-cycle read is a change to every unit's read pipeline, and
  its cycle cost here is an estimate (one cycle per instruction).
- P1 at 2x was co-simulated with QUANT 3 modelled as free (a third slot). The quantizer change it
  stands for is not written. Its own cost is estimated at one cycle per scaled row.
- The 266.67 MHz clock adds power (the RAMs toggle twice per core cycle; vectorless OOC power
  says 1.60 W for P3 against 2.14 W for the card's TMEM, a default-activity estimate) and a
  BUFG.
- TMEM above 64K needs 4eca9a76 (the sequencer's TAW, the STREAM descriptors) and IMEM growth
  to pay off at long contexts; the host's runs and caches key on the configuration already.
- The scoreboard's TMEM what-ifs come from the compiler's fit and an analytic model; the
  co-sims above check them at the points listed (section 3), not everywhere.
- Nothing here ran on the card.

## Reproducing

From the tree's root, `PYTHONPATH=.`:

```sh
# the pumped TMEM against otpu_tmem (Verilator; seconds to minutes each)
rtl/exp/tmem_pump/run_tb.sh /tmp/tmem_tb [cycles]
# out-of-context Vivado runs (the build host or the card host, through ~/otpu-build/slot.sh; one at a time)
rtl/exp/tmem_pump/run_ooc.sh <outdir> k7-v0 k7-v2 k7-v3 k7-v4 k7-v1 k7-v2x2 ku-v2b ku-v2u ku-v1b \
  ku-v2u300 ku-v2b-200 ku-v2u-200
# one copy at 3x / 4x (VAR 5): Kintex-7 at 133.33, xcku5p at 200 and 133.33, block RAM / UltraRAM
rtl/exp/tmem_pump/run_ooc.sh <outdir> k7-p1x3 k7-p1x4 ku-p1x3b-200 ku-p1x3u-200 ku-p1x4b-200 \
  ku-p1x4u-200 ku-p1x3b-133 ku-p1x3u-133 ku-p1x4b-133 ku-p1x4u-133 ku-p1x3c-200 ku-p1x3c-133
# the YPCB build with the banked TMEM (the build host)
make -C boards/ypcb-00338 bit DDR=1066 CORE_MHZ=133.33 TMEM_IMPL=1 TMEM_RQ=1
# cycles: the banked TMEM in the co-sim (OTPU_TARGET=nangate45 is the card's arch with
# TMEM_IMPL 1); OTPU_TMEM_RQ=1 the request-priority addresses, OTPU_TMEM_NCP=4 four slots (P2)
OTPU_PAIR=1 OTPU_DSTEP=1 OTPU_STREAM=1 OTPU_TARGET=nangate45 OTPU_TMEM_RQ=1 OTPU_TMEM_NCP=4 \
  python3 tools/perf_qwen.py --ddr 1066 --mhz 133.33 --ldc --check --pos 544 --model qwen3 \
  --wformat fp4 --head-format int8 --layers 2 --rows 4 --logits none
# P1 at 2x: OTPU_TMEM_NCP=2; the slot counts of one copy at 3x / 4x: OTPU_TMEM_NCP=5 / 7; any run
# appends the TMEM's read demand (TMEMSTAT lines) to OTPU_TMEMSTAT_LOG when it is set
OTPU_TMEMSTAT_LOG=/tmp/demand.txt OTPU_PAIR=1 OTPU_DSTEP=1 OTPU_STREAM=1 \
  python3 tools/perf_qwen.py --ddr 1066 --mhz 133.33 --ldc --check --pos 544 --model qwen3 \
  --wformat fp4 --head-format int8 --layers 2 --rows 4 --logits none
# the TMEM a rows-run needs (config.json only)
python3 tools/tmem_capacity.py need models/Qwen3-0.6B --mcols 4,8 --R 1,2,3,4,5,6,8,10,12,16 \
  --pos 300,3000 --logits last
python3 tools/tmem_capacity.py layer <Qwen3.5-35B-A3B dir> --tmem 65536 --R 1,2,3,4
```
