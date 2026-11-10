# openTPU as an ASIC

The ASIC target of the multi-target work (`docs/superpowers/specs/2026-10-10-multi-target-design.md`,
D5 and D6): the same core RTL as the FPGA card, built from SRAM macros and standard cells, through
OpenROAD on NanGate45 for PPA numbers and as a rehearsal for a manufacturable PDK. How to run it:
`asic/nangate45/README.md`.

## The block

`rtl/asic/otpu_asic_top.sv`: one slice (`otpu_slice`) and its collective unit, with the slice's own
memory ports at the boundary (port A words, the SW byte writes, port B chunks, the ready / valid /
idle handshakes), the run control (resets, the loader, the run's arguments) and the status. The
DRAM side (PHY, controller, the channel interleave of `otpu_native_dram`) is the platform's, as
LiteDRAM is the YPCB card's; the block is what every openTPU chip shares.
The platform's memory system must batch the host's writes per arbitration turn: each switch
between the accelerator's reads and a host write costs the controller a bus turnaround and its
open rows, so on the card single 256-byte host bursts cost the accelerator 0.20 s per GB written
(3.4x the data's own 0.059 s); queueing them into turns of up to 32 beats halves that (a
quarter to a third less at 2-3 GB/s), at no cost while the host is idle (`otpu_mem_ch` XT, XB;
docs/offload.md 13.19).

## Memories

Every large memory of the core goes through a technology RAM (`rtl/tech/<family>/otpu_tech_ram.sv`:
`otpu_ram_1r1w`, `otpu_ram_1r1w_be`, `otpu_ram_1rw`; registered reads with an enable): IMEM, the ACT
RAM's pieces and scales, the DMA's chunk buffer, the MXU's chunk and scale FIFOs, the stream
engine's delay buffer, and the banked TMEM's banks. `rtl/tech/generic` infers them (Verilator,
yosys, the FPGA, with the `ram_style` they had); `rtl/tech/asic` tiles them onto macros of at most
128 bits x 4096 words (depth tiles with a registered read select). Small asynchronous-read RAMs
(the dispatch window, the MXU's drain rows, the VPU's row sums, the quantizer's block buffer) stay
flip-flops.

**TMEM.** The FPGA keeps a copy of TMEM per read port (block RAM is cheap there; reads never
conflict). The ASIC's TMEM is banked (`otpu_tmem` IMPL 1, `otpu_slice` TMEM_IMPL 1): three copies of
LANES 1R1W bank RAMs, each bank one read and one write per cycle (RPB 1, WPB 1), writes to every
copy. A read port sits on one copy (`CPY`): copy 0 MXU, QUANT, VPU A; copy 1 VPU B, QUANT 2; copy 2
DMA, QUANT 3, COLL. The ports of one arbitration group are on different copies, so a group alone
always fits (the quantizer reads three ports at once, the VPU two; with fewer copies they could
never be granted). The slice's arbiter counts per copy and bank (`gnt_k`), and a bank's address
comes from the port granted in it. Results are the same as the FPGA's (the ISA simulator's); cycles
differ where ports meet in a bank: a Qwen3-0.6B decode token on the RTL (`tools/asic/tmem_cycles.py`,
the board's micro-architecture otherwise) takes 4,819,205 cycles banked against 4,815,735 with a
copy per port, +0.07%, with identical results. Three copies is 3x the TMEM's bits (the FPGA build
has six); a two-copy TMEM needs the quantizer's row-factor read off the TMEM ports. That is
nangate45-small's TMEM. The card point's since release r2 is one copy (`otpu_tmem` IMPL 3 in its
ASIC form, `docs/tmem_nocopy.md` 6.4: two RAMs a bank (NSB 2), lost reads replayed, two-port RAMs
with the write on port A, not pumped): 16 1RW1R macros of 4096 x 32, 1.33 mm² against the three
copies' 48 1R1W macros, 4.00 mm².

Verified: the RTL suite (`tests/test_rtl.py`, 133) and the model tokens on the RTL (Qwen3-0.6B
int8, fp4 with an int8 head, fp4 PAIR; Qwen3.5-0.8B with and without DSTEP) bit-exact against the
ISA simulator with the ASIC's TMEM (`OTPU_TARGET=nangate45 OTPU_UARCH=board`: 138 passed) and
with the card's (138 passed).

**Macros.** NanGate45 has no memory compiler. `tools/asic/fakeram.py` writes abstract macros in
bsg_fakeram's form (the pins, layers and straps of ORFS's fakeram45 macros) for 1RW, 1R1W and
1RW1R (bsg_fakeram makes 1RW only), any size, with area, clock-to-q, minimum period and clock
power fitted to ORFS's 24 fakeram45 macros (CACTI); a 1R1W or 1RW1R macro is 1.5x the 1RW's
area. The read data drives like NanGate45's BUF_X4 (its delay and transition over the load, its
242 fF limit; fakeram45's scalar clock-to-q let a macro drive millimetres of wire unbuffered: the
card, below). IHP SG13G2 and commercial nodes have 2-port (1R1W) macros; GF180 has 1RW only, so a
1RW TMEM is a later variant.

## Power-up state

A flip-flop's power-up value is not a specification: an FPGA's start at their INIT (0), an ASIC's
at whatever the silicon gives, a four-state simulation's (XSim, yosys `sim`) at X. Control state
(valids, grants, write enables, FSM states, counters, replay and age state) takes a synchronous
reset, to the value it is meant to start at; data (a register read only once a reset valid says it
was written: a command's fields, payload beside a valid, read data) and the RAMs' words do not.
The resets are written `x <= rst ? '0 : d` or `if (rst) ... else if (en)`, the flip-flop's R pin
(no LUT in front of it: TMEM's replay valids are 150 MHz replicas), never asynchronous.
Data is outside the reset altogether, not held through it: a register the reset holds takes the
reset in its enable. The sequencer's per-slot producer masks (`vpend`, `sdepd`) update every cycle,
and held through the reset they had it as the whole array's enable (a BCU1525 build at 150 MHz: the
sequencer's reset to `vpend`, 1,024 flip-flops behind a global buffer, 6.96 ns); they load at a
slot's dispatch and are read only for live slots. The MXU's XPOST read addresses, delayed to each
column's chain stage (`ar`, WASK), are data too: each column reads at its tap every cycle, and the
PEs take its bytes only at a pop's delays, as they take the skews' bytes without WASK (a BCU1525
build at 150 MHz: the MXU's reset copies into those 1,540 R pins its worst core paths, 5.7 ns of
route). XPOST (WG = MCOLS = 8, D 128) is past `tools/reset_audit.py`'s small sizes.

The RTL once zeroed some control state only in simulation-only `initial` blocks: a port of the RTL
to a ZU7EV simulated with `SYNTHESIS` defined in XSim stopped at 85 us of a 91.5 us demo, a TMEM
grant at X, the X circulating through the replay valids (`rpv`) and their ages, which Verilator's
`--x-initial 0` (`opentpu/rtlsim.py`) and the FPGA's INIT zeros had hidden. TMEM's write stage,
replays and ages and the pumped RAMs' phase, and the ACT RAM's write enables, now reset with the
slice's power-on reset (`otpu_slice` `rst_mem`, from `sys_rst`: a run's reset leaves the memories'
writes landing), on the ASIC and in the simulations; the FPGA builds leave it out
(`OTPU_FPGA_INIT`, below). The valids the units carry beside a delay line's payload are in reset
registers (`otpu_vdelay`: the VPU's lane taps and reductions, the MXU's row ends and ACC writes, the
quantizer's chunks and scales, the stream tail's d and Y), and the VPU's Q capture and RMAX meta and
the quantizer's stages and group RAM write enables reset.

A struct that carries a valid resets its valid alone (`if (rst) m.v <= 1'b0` after the enabled
load): the quantizer's chunks (`m0`, `mb1`, `mb2`, `mh`, `mc`) and write (`wq_r`), the VPU's RMAX
tree meta (`mxh_m`, read through `live()`: its valid and tag) and the MXU's ACC entry `r1` (whose
`xo` / `alr` then take no reset in their enable; its lanes go on into `u_rwv`'s reset stages behind
its valid, and `rw.m` is read only under `rw.v`); every use of their other fields is under the
valid (the BCU's quantizers too: QSTB's `m_ss` / `m_hl` / `baddr`, QONCE's byte writes, SW's
`m_ws`), so the reset loads the valids alone. HL's and QSTB's flags beside `mb1` / `mb2`
(`hb1`, `hb2`, `sb1`, `sb2`) keep theirs. The audit's class v checks that each such valid is known
after the reset. The quantizer's delay lines whose last stage reset (`otpu_qdly`: the scale unit's
Newton operands and flags, the write's fields behind its valid) carried no valid of their own, so
under the same rule they take no reset: they are `otpu_delay` lines, data beside the valids in
`otpu_vdelay` (`u_v`, `u_wqv`), which reset every stage and which the audit requires known. Their
last stage stays a flip-flop in Vivado (`shreg_extract = "no"`, as in every `otpu_delay`);
`otpu_qdly`'s reset did that for yosys's Xilinx mapping, which ignores the attribute (the quantizer
tournament's evaluations).

**The FPGA builds: `OTPU_FPGA_INIT`.** The ypcb-00338 and bcu1525 flows define it (their
`create_project.tcl`, bcu1525's `create_project_wg.tcl` for the WG, two-slice and two-card builds,
and their lints); `rst_mem` is then 0, so TMEM's and the ACT RAM's control state (`wi_en`,
`wi_gnt`, `pw_v`, `rpv` / `rpvb` and their ages, `bh` / `bh1`, each pumped RAM's `sv` / `v1` and
`tog` / `tq` / `evr`; `we_i` / `we_h`, `swe_i` / `swe_h`, `ssel`, `wsel`) has no reset there. What
keeps it right:

- *Configuration.* Every flip-flop starts at its INIT, 0: the state the reset gives, but for the
  pumped RAMs' `tq`, which the reset sets to 1. `tq` and `evr` only sample `tog` (`tq` its copy a
  fast edge later, `evr` whether they differ), so they hold the right phase from the first and
  second fast edges after configuration, whatever they started at; nothing asks TMEM for anything
  for milliseconds after it (PCIe enumeration, the host's program load).
- *A stopped run* (the host clears RUN, a CHAIN reload, a timeout: `core_rst` / `srst`) never
  reset this state, with or without the define: `rst_mem` follows `sys_rst` only, so that a run's
  accepted writes land. It is a pipeline of the units' requests: `wi_en` / `wi_gnt` load the
  write ports' requests and grants, `pw_v`, `sv` and `v1` the selected writes a cycle later, `we_i`
  -> `we_h` and `swe_i` -> `swe_h` the quantizer's ACT writes, `ssel` / `wsel` their selects, `bh` /
  `bh1` a read's bypass hit beside its data. With every unit in reset no request comes, and a few
  core cycles later every write valid is 0, the writes in flight landed. Replays (`rpv`) retry
  until they win their bank, the oldest first (their ages, read only while two replays run); with
  no new request they have all won a few cycles later. `tog` toggles every core cycle whatever it
  starts at (the pumped RAMs' phase: `tq` / `evr` follow it), so any known value is right. The
  units stay in reset far longer: until the host loads the next program and sets RUN.
- *`sys_rst` after configuration* (PERST#, the MMCM losing lock: `core_rst` follows it, so the
  units are in reset too) is the same drain, with the core clock running: the reset block holds it
  through its release sequence after the clock locks, far more cycles than the drain takes. Across
  an unlocked clock any flip-flop can take a wrong value and a RAM write can land wrong, with the
  reset or without it.

The ASIC keeps the reset (no INIT: its flip-flops start at whatever the silicon gives, and a
four-state `tog` from X stays X), and so do the simulations; `tests/test_reset_audit.py` reads the
slice both ways (on the pumped TMEM and on wmx's VPU and quantizer: `rst_mem` a register without
the define and gone with it, TMEM's and the ACT RAM's synchronous resets fewer, every other unit's
the same) and checks that only the FPGA flows set it.

Checks:

- `python3 tools/reset_audit.py [--tmem]` (yosys with slang, `YOSYS`; ~70 s, < 1.5 GB): the slice
  read as synthesis reads it (`-D SYNTHESIS`), every flip-flop and array (not the RAMs) at X, the
  resets held 3 cycles with the inputs idle (yosys `sim`). Every flip-flop still X must be in its
  ALLOW list with the reason it is data, no register a simulation-only `initial` sets may be X, and
  no ALLOW entry may be stale; the four configurations (the ASIC's small point, the card's TMEM at
  the core clock and pumped, a copy per read port). `--tmem`: TMEM under random conflicting traffic
  from an X power-up (`sim/yosys/tb_tmem_x.sv`, the card's IMPL 3 with replays and ages) keeps its
  lost, frz and blocks known, and without its reset port does not (the ZU7EV's fault, from the
  first cycle). `tests/test_reset_audit.py` runs both, and reads the slice with and without
  `OTPU_FPGA_INIT`.
- `OTPU_X_INITIAL=unique OTPU_X_SEED=N`: the Verilator builds with `--x-initial unique` (and
  `--no-assert-case`: a one-hot select is one-hot from its reset on, and Verilator checks `unique
  case` from time 0) and their runs with random two-state power-up values;
  `tests/test_rtl.py::test_random_power_up` runs random programs that way against the ISA (the
  design's and the card's micro-architecture, several seeds): DRAM, and TMEM at the words the
  program writes (a RAM's other words keep whatever the cycles before the reset held wrote). For
  it the behavioural DRAM (`otpu_dram`) takes no request in `sys_rst`, as a controller in reset,
  and the simulation-only checks start once their reset has held two edges (`rst_seen`: a unit's
  reset register is itself arbitrary until the slice's reset pipeline fills). On 91df4109's RTL,
  its checks made displays, 18 of 20 such runs failed (the card's micro-architecture: all 10 never
  halted; the design's: 2 never halted, 6 wrote other DRAM); on this one 40 of 40 passed (20 seeds
  each), no check tripping.

iverilog (through sv2v) is no four-state check here: it compiles TMEM but its IMPL 3 never leaves
the first clock edge (a zero-delay loop), so the four-state check is yosys's `sim`. The RAMs keep
their simulation-only zero `initial` (the dumps compare every word against the ISA's); a RAM word is
data, written before it is read.

## Results

NanGate45 with `tools/asic/fakeram.py`'s macros, 2.5 ns asked (the targets' `core_mhz` 400), the
ports with 20% of the period outside. Two clock figures: **register to register**
(`report_clock_min_period`: the core's own), and **with the ports** (the worst slack including the
paths into and out of the block's ports, whose budget is a guess until the platform around the
block exists; from CTS on the flow times them latency-neutral, the virtual clock taking the clock
tree's insertion delay, [asic_timing_loop.md](asic_timing_loop.md) 4; nangate45-small's and the
card before's figures are against the ideal virtual clock). Power is OpenSTA's vectorless estimate (default activities) at 400 MHz.
`tools/asic/flow.py report` and `tools/asic/paths.py` make these numbers from a run.
nangate45-small is the reference point: ORFS `openroad/orfs:latest` (pulled 2026-10-10) at 35%
utilisation; the same run with a native ORFS of the same date (f395e4b6, OpenROAD
26Q4-110-g91e5f5ff2f) gives the same within the tools' noise (422 / 341 MHz, 0.79 W, 0 DRC), and
its reports are in `asic/nangate45/results/small/`. The card is release r2's arch at ACT_ROWS 8
(`targets/nangate45.toml`) through that native ORFS on the simulation host, at 45% with the MXU's macros by
its columns and the TMEM's by its readers, with the ports' boundary registers and the VPU's grant
copies (main 1aad9195, r4 below); "before" is the previous card point (2026-10-11: the card's arch before r2,
ACT_ROWS 4 and three TMEM copies, at 35% on the macro placer's floorplan).

| | nangate45-small | card: r2, ACT_ROWS 8, the TMEM by its readers (r4) | card before (2026-10-11) |
|---|---|---|---|
| stage | routed, final report (signoff extraction) | CTS; global routing (clean at metal4-6's 7%), timed on its parasitics before its repair | CTS; global routing ends congested |
| die | 2.65 x 2.65 mm = 7.02 mm² | 4.52 x 4.52 mm = 20.4 mm² | 5.53 x 5.53 mm = 30.6 mm² |
| standard cells | 1.18 mm², 590 K instances (20.8% of the core): 94 K flip-flops 0.43 mm², 373 K gates 0.66 mm² | 4.59 mm², 2.07 M instances (29.7% of the core outside the macros; 1.83 M after synthesis) | 4.36 mm², 2.02 M instances (18.3% of the core; 1.76 M after synthesis) |
| macros | 27, 233 KiB, 1.31 mm² | 90, 758 KiB, 4.98 mm² | 113, 1126 KiB, 6.62 mm² |
| register to register | 2.39 ns: **419 MHz** (no violation at 2.5 ns) | 2.67 ns: **375 MHz** on the global route, before its repair (2.65 ns, 377 MHz, after CTS on the placement's estimate; clock skew 0.52 ns; the worst path one unrepaired flip-flop of the MXU's decoded weight scale, 2.55 ns without it; tm100, before the boundary registers and the grant copies, 2.59 ns) | 5.60 ns: 178 MHz (clock skew 2.7 ns) |
| with the ports | 2.95 ns: 339 MHz (one endpoint: `sw_req`) | 2.67 ns: 375 MHz, the ports latency-neutral and met (out +1.24 ns, in +0.08 ns) | 6.40 ns: 156 MHz |
| power at 400 MHz | 0.79 W: sequential 0.28, clock 0.20, combinational 0.15, macros 0.16; leakage 21 mW | 4.38 W: clock 1.75, sequential 1.22, macros 0.78, combinational 0.63; leakage 83 mW (vectorless; tm100 4.87 W, the difference the estimate's: r4 below) | 3.34 W: sequential 1.09, clock 0.93, macros 0.87, combinational 0.46; leakage 82 mW |
| route | 16.2 m of wire, 0 DRC violations, 0 antenna violations | global: 109.4 m, no overflow (29.9% of the tracks); not detail-routed | not routed |

### The card: release r2

r2 changed the card's two largest memories. The TMEM is one copy (16 1RW1R macros, 1.33 mm²,
against the three copies' 48, 4.00 mm²: Memories, above). The ACT RAM took ACT_ROWS 16: an MM of
up to 16 stationary rows, the MXU replaying each streamed weight chunk for ceil(M / MCOLS) groups
(`docs/isa.md` MM, `docs/tiled_prefill.md` 3.1), four times the previous card's rows (ACT_ROWS =
MCOLS = 4). On the FPGA that was free: each of the ACT RAM's pieces is block RAM 512 words deep,
and four rows used 128 of them (one group of 128 blocks); sixteen fill them. On the ASIC the macros
are paid by the bit: its 32 macros went from 128 x 128 (0.617 mm²) to 512 x 128 (2.038 mm², 3.3x),
the second unit after the MXU at 20% of the synthesized area. "ACT_ROWS" (below) measured what the
rows buy; the card point takes 8 (32 macros of 256 x 128, 1.136 mm²), and the area per unit after
synthesis, mm² (macros in brackets; yosys's per-unit figures, which overlap by about 0.07 mm²
against the block's total), is:

| unit | before (2026-10-11) | r2, ACT_ROWS 16 | r2, ACT_ROWS 8 |
|---|---|---|---|
| MXU (128 x 4 systolic, the weight FIFO) | 1.993 (0.951) | 2.383 (0.990) | 2.384 (0.990) |
| ACT RAM | 0.645 (0.617) | 2.063 (2.038) | 1.165 (1.136) |
| DMA with the stream engine | 1.420 (0.150) | 1.420 (0.150) | 1.420 (0.150) |
| TMEM | 4.053 (4.000: 48 x 4096x32 1R1W) | 1.405 (1.330: 16 x 4096x32 1RW1R) | 1.405 (1.330) |
| sequencer | 1.082 (0.858) | 1.185 (1.055) | 1.185 (1.055) |
| VPU | 1.161 (0.040) | 1.170 (0.040) | 1.170 (0.040) |
| quantizer | 0.364 | 0.508 (0.270) | 0.508 (0.270) |
| the block (ORFS's synthesis total) | 10.66 (6.62) | 10.06 (5.87) | 9.17 (4.98) |

A further 0.38 mm² is in the ACT RAM's ports: as 1RW macros (2/3 of a 1R1W's area) it would be
0.76 mm² at 8 rows (0.68 mm² less at 16). That needs the quantizer's writes and the MXU's reads to
share one port a macro, with a stall where both want it in a cycle; how often that happens is a
co-simulation question, not run.
### The card's floorplan: the MXU by its columns

The ACT RAM is the MXU's stationary operand: every cycle each of the MCOLS columns reads a block
(D bytes) of its row, 4096 bits at MCOLS 4. `otpu_actram` keeps one memory per column j and
32-byte piece p, tiled into 128-bit macros (g_c), so macro (j, k) (k = 2p + c) holds bytes
16k .. 16k+15 of column j's row and feeds only PE chain k of MXU column j (`otpu_mxu`: chains of
16 products, each PE one byte of its column's row); the weights move from column j to j+1 one hop a
cycle. The macro placer does not see that: on r2 it packs the ACT RAM's macros into one block in a
corner, and the 4096-bit read crosses the die to the array (the previous card run's congestion,
above). `asic/nangate45/mxu_macros.tcl` (ORFS `MACRO_PLACEMENT_TCL`, the card point's `macros` in
`tools/asic/flow.py`) places them by that structure, and the macro placer places the rest:

- two stacks of abutted ACT macros, columns 0-1 on the left (pins to the right) and 2-3 on the
  right (pins to the left), chain-major ((k, j) upward), facing a strip (800 um) where global
  placement puts the array: each macro's 128-bit read goes straight into its own chain, and the
  weights hop from column to column inside the strip without crossing a macro;
- the columns' scale RAMs (`u_asc`) on top of their side's stack;
- the weight FIFO (the chunk's macros in bit order, the scale FIFO's in the middle) a third stack
  at the core's left edge, pins to the right, 100 um from the left stack (the chunk crosses it in
  metal5 and up: the macros block metal1-4);
- the block centred on the core's height at its left edge, and the macro placer's fence right of
  it (ORFS's default fence is 0, no fence), so nothing else lands in the strip.

Global placement's routability estimate (weighted congestion of the worst tiles; the share of the
tiles overflowed), r2 at ACT_ROWS 16, `PLACE_DENSITY_LB_ADDON` 0.02, native ORFS on the simulation host:

| floorplan | die | first routability iteration | best | wire (global placement) |
|---|---|---|---|---|
| macro placer, 45% | 4.74 mm, 22.4 mm² | 2.61 (16.0%) | 2.61: no later iteration better | 92.2 m |
| macro placer, 40% | 5.03 mm, 25.3 mm² | 1.73 (4.4%) | 1.09 (0.44%) | 90.7 m |
| **by columns, 45%** | 4.74 mm, 22.4 mm² | 2.34 (6.7%) | **1.10 (0.48%)** | **87.5 m** |
| before (113 macros), 35%, addon 0.10 | 5.53 mm, 30.6 mm² | 2.07 (6.6%) | 2.07 | 98.6 m |
| before, 35%, addon 0.02 | 5.53 mm, 30.6 mm² | 1.85 (5.4%) | 1.01 (0.24%): the target | 98.6 m |

The column floorplan holds the 40% die's congestion on the 45% die with 5% less wire; neither
reaches global placement's target (1.01) exactly. (The 40% run's detailed placement then failed
to legalize: the negotiation legalizer left 2463 violations.) The clock tree follows: CTS (ORFS's
defaults) on the column placement leaves 0.71 ns of skew against the previous card run's 2.7 ns;
before the timing repair the macros' clock pins sit at 1.87-2.08 ns and the registers' at
1.91-2.13 ns (`-obstruction_aware` builds the same tree; `-macro_clustering_size 1` leaves the
macros without a clock).

### The card's clock and paths (ACT_ROWS 8, the TMEM in the corner, after CTS)

The macros' output drive matters here. With fakeram45's scalar clock-to-q (and a 500 fF limit) a
macro's read data reached its logic through millimetres of unbuffered wire and the resizer saw no
violation: on the ACT_ROWS 16 column floorplan after CTS the worst register-to-register path was
the weight FIFO's `rd_out` -> 3.4 ns of wire (485 fF) -> the MXU's S1 decode (-2.56 ns; 5.06 ns
register to register). With the read data driving like a BUF_X4 (Memories, above) 52 macro
outputs exceed the limit, the FIFO's, the DMA's line buffer's and the ACT RAM's (to 417 fF);
`repair_design` buffers them and that CTS result reaches 4.16 ns. The ACT_ROWS 8 placement ran on
the scalar models, so its CTS runs ORFS's design repair first (`PRE_CTS_TCL`:
`estimate_parasitics -placement; repair_design_helper`), then CTS with ORFS's defaults.

After CTS: 3.52 ns register to register (284 MHz), 0.96 ns of clock skew (a TMEM macro's clock
pin at 2.78 ns against 1.82 ns for the VPU's registers). By kind (`tools/asic/paths.py card 300
4_cts`, 78.5 K endpoints):

| kind | endpoints | worst slack | the worst path |
|---|---|---|---|
| io | 1227 | -2.50 ns | the DMA -> port B's write data (`b_wdata`) |
| sram | 4901 | -1.02 ns | a TMEM macro -> the DMA's `t_rq`; the TMEM's reads into the VPU -0.88 / -0.75, the DMA's `dsw` -0.72, the quantizer -0.62, the MXU -0.52 |
| dma | 32.4 K | -0.52 ns | the DMA's `dw` -> `ww_w` |
| control | 23.4 K | -0.02 ns | the quantizer's `bm` -> `dq` |
| systolic | 5517 | +0.10 ns | MXU `hi0` -> PE |
| fpipe | 7030 | +0.11 ns | the VPU's grant -> an fmul s1 |
| tmem | 4038 | +0.27 ns | the slice's reset -> the sequencer |

Everything in the MXU's block now meets 2.5 ns (the array, its fp lanes, the ACT RAM's paths
+0.31 ns); the limit is the TMEM's reads. The macro placer put its 16 macros in the core's
bottom-right corner, away from the units that read it (VPU, quantizer, DMA, MXU drain), and the
macro clock tree reaches them about a nanosecond after the registers; without that, the
register-to-register period would be about 2.6 ns. The ports' worst is port B's write data
straight from the DMA's state across the die, which a boundary register removes (the SW port
has one on every target; port B's is in work).

### The card's global route (ACT_ROWS 8, the TMEM in the corner)

ORFS's global routing on the CTS result, with nangate45's layer adjustments (`fastroute.tcl`:
metal2-3 50%, metal4-10 25%, set at the floorplan and kept in the database), ends congested
(GRT-0116 after 30 iterations: 426 overflowed edges at 31% usage; 60 min, 27 GB). All of it is
one spot: the DMA's line buffer, 8 macros of 128 x 128 that the macro placer packed 2 x 4 against
the core's top edge (x 1.66-2.35 mm), under the top edge's port B write-data pins (967 of
`b_wdata`'s 1024 along x 0.39-2.31 mm). The pins and the line buffer's outputs cross the block in
metal5 and up (of the 426: metal5 128, metal7 95, metal6 70, metal9 43, metal8 42; metal2-4 30).
The ACT RAM's edge, the previous card's congestion, has none. GRT's hint (GRT-0704: the layer
adjustment from 25% to 7%) routes it: with metal4-10 at 7% (`PRE_GLOBAL_ROUTE_TCL`
`set_global_routing_layer_adjustment`; metal7-10 stay reduced about 28-44% by the power grid's
straps, so the change is metal4-6's) the overflow goes 189 -> 39 -> 0 by the 14th extra iteration
(25%: 909 -> 742 -> 448), 99.2 m of wire (usage: metal2 51%, metal3 48%, metal4 38%, metal5 21%,
metal6 10%, metal7 and up under 5%), 40 min and 25 GB. The placement's fix is a channel between
the line buffer and the edge, or port B's pins on an edge without macros; 7% leaves the detailed
router less room on metal4-6. Not detail-routed (2.07 M nets: a decision of its own).

On the global route's parasitics (`estimate_parasitics -global_routing`, the CTS netlist: ORFS's
repair after global routing not run), register to register is 2.94 ns (340 MHz), against 3.52 ns
on the placement's estimate after CTS: the estimate gives every net one layer's resistance, the
router puts the long ones on metal4-6. The worst path is still a TMEM macro -> the DMA's `t_rq`;
the macro's clock pin is at 2.63 ns against 1.88 ns for the registers (0.74 ns of skew; its leaf
buffer 0.47 ns and the wire from it to the clock pin 0.35). The ports' worst is port B's write
data (-2.47 ns). Left for the repair: 1506 nets over their maximum capacitance and 126 pins over
their maximum slew (56 and 6 after CTS).

### The card's TMEM by its readers

On the column floorplan the macro placer puts the TMEM's 16 macros in a 4 x 4 block in the core's
bottom-right corner, 3-5 mm from the units that read it, and the clock tree reaches that corner's
last column late (2.68-2.78 ns against 1.80-2.17 for the other TMEM macros and 1.89 for the
registers' median). `mxu_macros.tcl` with `TMEM_GAP_UM` (the card point's `tmem_gap`, 100)
places them as a block of their own, that far right of the MXU's: two stacks of abutted macros,
banks 0-3 on the left (pins to the right) and 4-7 on the right (pins to the left), facing a strip
(`TMEM_STRIP_UM`, 200 um) for the banks' muxes and the slice's arbiter, centred on the core's
height; global placement puts the readers around it, and the macro placer's fence starts right
of it. On the same netlist, the TMEM block at x 1.70-2.47 mm, y 1.10-3.42 mm, after CTS
(placement parasitics for both):

| | TMEM in the corner (the macro placer) | TMEM by its readers |
|---|---|---|
| the TMEM macros' clock pins | 1.80-2.78 ns, mean 2.09 | 1.67-1.86 ns, mean 1.79 |
| all 90 macros / the registers' median | 1.71-2.78 / 1.885 ns | 1.67-2.03 / 1.872 ns |
| clock skew | 0.96 ns | 0.31 ns |
| register to register | 3.52 ns (284 MHz): a TMEM macro -> the DMA's `t_rq` | 2.69 ns (372 MHz): the VPU's grant -> an fmul s1 |
| with the ports | 5.00 ns: port B's write data | 4.70 ns: port B's write data |
| wire after detailed placement | 70.6 m | 83.7 m |
| global placement's weighted congestion: first / best | 2.67 (9.0%) / 1.125 | 2.80 (11.8%) / 1.122 |
| power at 400 MHz | 4.88 W | 4.79 W |

The TMEM's reads by reader (the worst 300 paths into each, the wire the Manhattan length along the
path's pins, the data delay from the macro's clock pin):

| reader | corner: worst slack | wire mean / max | data delay | by its readers: worst slack | wire mean / max | data delay |
|---|---|---|---|---|---|---|
| DMA | -1.02 ns | 5.5 / 7.3 mm | 1.77 ns | +0.75 ns | 3.3 / 4.5 mm | 1.18 ns |
| VPU | -0.88 ns | 5.6 / 6.9 mm | 1.75 ns | +0.67 ns | 2.9 / 4.1 mm | 1.05 ns |
| quantizer | -0.62 ns | 4.1 / 5.5 mm | 1.42 ns | +0.50 ns | 3.5 / 6.1 mm | 1.30 ns |
| MXU (drain) | -0.52 ns | 4.0 / 5.3 mm | 1.40 ns | +0.80 ns | 3.3 / 3.7 mm | 1.19 ns |
| the TMEM's own registers | -0.60 ns | 5.2 / 6.0 mm | 1.62 ns | +0.71 ns | 3.3 / 4.6 mm | 1.24 ns |
| the slice's arbiter | -0.66 ns | 4.8 / 6.0 mm | 1.60 ns | +0.63 ns | 3.8 / 4.2 mm | 1.32 ns |

The distance is the NUMA cost of a memory read here: over all 1488 paths, data delay = 0.47 ns +
0.220 ns per mm of wire (in the corner 0.55 ns + 0.213 ns per mm), so a reader 3 mm away pays
about 0.66 ns of a 2.5 ns cycle in wire. Of each reader's 1.1-1.8 ns of slack gained about 1 ns
is the clock (the worst paths launched from the corner's late macros, now with the registers) and
0.1-0.7 ns the wire (its mean data delay; 0.6-2.7 mm less on average). Every TMEM read meets 2.5 ns with 0.5-0.8 ns to spare, and the
register-to-register limit moves to the VPU's grant fanout (387 endpoints at -0.19 ns; +0.11 in
the corner run, the VPU's registers now spread around the block: "Grant fanout", Open problems).
The cost is wire: the block splits the core, 18.5% more after placement. It routes: global
routing as a8's (metal4-6 at 7%, no repair) ends clean after 7 extra iterations, 112.2 m against
99.2 m (usage 30.9% against 26.9%; metal4 47%, metal5 26%), 40 min, 27 GB; on its parasitics
register to register is 2.59 ns (**386 MHz**, against 2.94 ns, 340 MHz, with the TMEM in the
corner), 0.26 ns of skew, the VPU's grant the worst path, port B's write data -2.19 ns, 4.87 W
at 400 MHz (4.96), 2213 nets over their maximum capacitance and 203 pins over their maximum slew
left for the repair (1506 and 126). The macro placer then put port B's write-data pins on the
right edge (830, beside the sequencer's IMEM macros) and the bottom edge (194), the line buffer
in the bottom-right corner, and none of it congests.

### The card's boundary registers and the grant copies (r3, r4)

Two RTL changes, each a run of the card point from main through global routing (the same recipe:
the TMEM by its readers, metal4 and up at 7%, the ports latency-neutral from CTS; global routing
without ORFS's repair, timed on its parasitics: `estimate_parasitics -global_routing` from the
route's guides, every endpoint's worst path; tm100 is the card point above, its virtual clock at
its own CTS latency, 1.872 ns): r3 (main 7e42a626) registers port A's and port B's outputs and
port B's inputs at the block's boundary ("The ports", Open problems), r4 (main 1aad9195) gives
the VPU's stream grant per-lane copies ("Grant fanout"; its 24 `otpu_delay` copies come out of
synthesis as 24 instances). The same night's host-write batches (`otpu_mem_ch`) are the FPGA's
only: the ASIC block's sources leave out `rtl/shell`. Main since adds r2.2's two cycle-exact cuts
from the FPGA's m8 near-critical paths (`otpu_dma`: TMEM's read request straight from flip-flops,
the advance predicted a cycle ahead; `otpu_mxu`: `mx_go` a register of its own), not through this
flow.

| | tm100 (06c4be2d) | r3: the boundary registers | r4: and the grant copies |
|---|---|---|---|
| after synthesis | 1,784,512 cells, 9.17 mm² | 1,835,617 cells, 9.27 mm² | 1,834,493 cells, 9.27 mm² |
| after CTS | 2.00 M instances, 4.48 mm² of cells | 2.01 M instances, 4.48 mm² of cells | 2.07 M instances, 4.59 mm² of cells |
| wire after detailed placement | 83.7 m | 82.9 m | 78.2 m |
| register to register, ideal clocks (placement) | 2.66 ns | 2.43 ns | 2.69 ns |
| register to register after CTS | 2.69 ns (372 MHz) | 2.40 ns (416 MHz) | 2.65 ns (377 MHz) |
| global route | 112.2 m, no overflow | 112.0 m, no overflow | 109.4 m, no overflow |
| register to register on it | 2.59 ns (**386 MHz**) | 2.73 ns (367 MHz) | 2.67 ns (375 MHz) |
| reg2reg: worst, TNS, failing endpoints | -0.091 ns, -0.48 ns, 14 | -0.227 ns, -0.31 ns, 4 | -0.166 ns, -2.43 ns, 33 |
| in2reg: worst | +0.217 ns | +0.201 ns | +0.075 ns |
| reg2out: worst, TNS, failing | -0.322 ns, -71.09 ns, 634 | +1.173 ns, 0, 0 | +1.244 ns, 0, 0 |
| with the ports | 2.82 ns (port B's write data) | 2.73 ns (the ports meet) | 2.67 ns (the ports meet) |
| power at 400 MHz | 4.87 W | 4.98 W | 4.38 W |

The boundary registers (r3) took the ports off the clock: reg2out went from -0.322 ns (634
failing endpoints, port B's write data from the DMA's rotator) to +1.17 ns, in2reg stayed at
+0.2 ns, and with its ports the block runs at its register-to-register clock (2.82 -> 2.73 ns),
for 51 K cells and 0.10 mm². The grant copies (r4) took the VPU's stream grant off the critical
list: its worst path (from `ss_act` or a copy) went from -0.091 ns (tm100) and +0.268 (r3) to
+0.447 ns (a copy's complement into the tail's Q multiplier operands), and the VPU's worst register
path is now +0.410 ns (the tail's U buffer macro into U's operand registers). Neither moves the
register-to-register clock, which each run sets by a path its change did not touch:

- r3's 2.73 ns is the TMEM's replay freeze (`frz[0]`) into the VPU's read enable, -0.227 ns, 0.755 ns
  of it one AND4_X1 on a long net; r4's 2.67 ns is the MXU's decoded weight-block scale (`dhi1`,
  one flip-flop of it) into every column's aligned scale (`bm2`: the weight scale less the column's
  activation scale), -0.166 ns, 1.01 ns of it the DFF_X1's clock-to-Q into its load (32 of r4's 33
  failing endpoints). Each is a weak driver on a long net, what `repair_design` sizes or buffers
  (r4 leaves it 2,388 nets over their maximum capacitance and 179 pins over their maximum slew; r3
  2,248 and 111).
- Past those, r3's next is the MXU's `lo` -> `dal` at +0.233 ns (2.27 ns) and r4's the DMA's
  write-data rotator into port B's output register (`b_wdata[960]`, nine MUX2 levels from one
  select) at -0.053 ns (2.55 ns; +0.279 on r3), then port B's ready into the same registers
  (`b_rdy`, +0.075 ns; +0.201 on r3) and the TMEM's read ports into the VPU (+0.096 ns).
- The difference is placement. r4 runs on r3's floorplan, every macro where r3's are (for r4's
  netlist the macro placer had put the IMEM 0.95 mm to the left: 3.11 ns on ideal clocks), but
  global placement places the standard cells anew for each netlist: 5.7% less wire after detailed
  placement (78.2 against 82.9 m), CTS adding 228 K instances against 154 K (the tree one buffer
  level deeper), and the untouched paths 0.2-0.3 ns apart between the two runs (the rotator above;
  `b_rvalid` into the line buffer's macros +0.381 / +0.172 ns).

So the card's clock is now set by the flow, not by the arch: 2.67-2.73 ns before a repair, from
one weak driver in each run, and 2.27-2.55 ns past it, the spread placement's. What the RTL can
still take: the rotator in two stages, a registered `b_rdy` ("The ports", Open problems), and
the decoded weight scale copied per column as the grant was.

Power (vectorless: OpenSTA's default activities) reads 4.38 W against r3's 4.98, and neither
part of the difference is the RTL's. The clock tree takes 0.57 W more (1.75 W: CTS added 228 K
instances to r4 against 154 K), and the combinational logic 1.15 W less (0.63 against 1.78), all
of it in the VPU (1.91 -> 0.79 W already at placement: the SE tail 0.98 -> 0.31, the lanes' RDOT
multipliers 0.53 -> 0.14), the logic behind the grant's selects. The copies equal the one
flip-flop every cycle (the change is cycle-exact), and under the default activities the grant
never switches (its propagated activity is 0 in both runs): what moved is the estimate's
propagation through the new selects, not what the logic does. A VCD from a workload (Open
problems) would measure it.

ORFS's own repair after global routing (`repair_design` then `repair_timing` on the route's
parasitics, single-threaded) does not fit this block's size: on r3 `repair_design` went through
1.52 M of its 2.53 M nets in 25 minutes and then slowed to about 2,000 a minute (2,233 buffers,
46 resizes by then), more than 8 hours to go before `repair_timing` began; it was stopped. The
numbers here are before that repair (2,248 nets over their maximum capacitance and 111 pins over
their maximum slew left on r3), and a routed card point needs a faster repair: by region, or
limited to the failing paths' nets. The card point now has the latter (`asic/nangate45/grt_repair.tcl`,
`PRE_GLOBAL_ROUTE_TCL`: the GRT step's `repair_design` a no-op, its `repair_timing` on the failing
paths). On main's card (5873d298 on r3's floorplan) with a 0.5 ns setup margin it took 100 s, 41
buffers and 295 upsized cells, and register to register went from 2.497 to 2.000 ns on the
route's parasitics; with a 1.0 ns margin it stalled at 1.99 ns (the VPU's floating-point stages)
after ~3,000 moves (docs/timing_ffs.md section 4).

### Before: the previous card point (2026-10-11)

The card's arch before r2 (113 macros, three TMEM copies, ACT_ROWS 4), 35%, the macro placer's
floorplan. Its clock was a physical-design result, not the RTL's: CTS left the macros' clock pins
at 4.9-5.0 ns of insertion delay against 2.3 ns for the flip-flops (2.7 ns of skew on a 30 mm² die
with 113 macros; nangate45-small's skew is 0.13 ns), and the worst path paid it: a TMEM bank
macro (launched 2.6 ns late) -> a 1.1 ns wire across the die -> the TMEM read mux -> the DMA's
`dsw` (-3.10 ns). By kind, after CTS (`tools/asic/paths.py card 300 4_cts`, the worst 100 K
endpoints):

| kind | failing endpoints | worst slack | the worst path |
|---|---|---|---|
| io | 1227 | -3.90 ns | DMA `fs_k` -> port `sw_req` |
| sram | 4815 | -3.10 ns | TMEM bank macro -> DMA `dsw`; the TMEM's reads into every unit -2.2 to -2.7 (the macros' late clock) |
| dma | 73.6 K | -1.81 ns | DMA `fs_k` -> the arbitration -> the quantizer's block buffer (the enable fanout, now across a 5.5 mm die) |
| tmem | 491 | -1.77 ns | DMA `fs_k` -> TMEM bypass hit |
| control | 2572 | -1.69 ns | MXU `h` -> the quantizer's block buffer |
| fpipe | 13.1 K | -1.38 ns | DMA `fs_k` -> the quantizer's fadd s2 (an enable) |
| systolic | 0 | +0.41 ns | MXU `hi0` -> PE |

Without the skew the macro paths are about -0.5 ns; the enable fanout (-1.8 ns, no macro in it)
is then the card's limit, as it is the small point's second: on a 30 mm² die it is wire. The fp
stages' own logic still meets 2.5 ns on the card (fmul s1 -> s2 +0.76 ns, s2 -> the next s1
+0.46, fadd s1 -> s2 +0.95, s2 -> s3 +0.92), while every path into them fails through the
enable (-1.31 to -1.38 ns, 57-65 levels from the DMA's `fs_k`, `ds_fill`, `pl`); the stage-class
report stopped there (the container's memory).

**The card's global routing** (native ORFS, 62 GB) ends congested: GRT-0116 after its 30
congestion iterations, 176 K overflow left and not shrinking, 4 h 22 min at a 31.7 GB peak; the
whole run took 7 hours to there. The macro placer packs the ACT RAM's 32 macros (128 x 128,
136 x 137 um, about 270 pins on one edge each) into the die's bottom-left corner, 0.6 x 1.1 mm with
60 um channels. The router's congestion report puts nearly all the overflow along that block's
inner edge and just past it (x 0.4-1.0 mm, y 0-1.4 mm); the nets in its markers are the ACT RAM's
own (141 K), its 4096-bit read data to the MXU (`act_rdata`, 100 K), the MXU's (39 K) and the
block's port B data (16 K). Global placement saw it coming (routability mode: "could not reach
target", weighted congestion 1.60). Spacing the macros does not help: 40 um halos and 150 um
channels give 1.69, and with the macro placer's boundary weight off and the channels on every side
1.67. A lower placement density does: the target is 0.28 while the cells fill under a fifth of the
core, and with `PLACE_DENSITY_LB_ADDON` 0.02 (target 0.22, the macros where they were) global
placement reaches its routability target (weighted congestion under 1.01 with 0.24% of the tiles
overflowed; 1.18 at its end) at the same wire length. Global routing was not run on it: r2's card
point took the next steps instead (the ACT RAM by the MXU's columns, a balanced macro clock, a
smaller die: above).

The final power-grid check failed on one row segment (PSM-0069: 26 um boxed in by macros,
narrower than the M4 straps' pitch, its rails unconnected; 14 cells); the run finished with the
IR-drop analysis off. The flow now drops such rows (`asic/nangate45/tapcell.tcl`, checked on this
floorplan: no row narrower than 60 um is left) and no longer fixes hold at the ports (2188 input
endpoints, 12 K hold buffers here); the card's run has both.

Area per unit after synthesis (before placement's buffers), nangate45-small, mm²:

| unit | total | macros | logic |
|---|---|---|---|
| TMEM (banked, 12 x 4096x32) | 1.019 | 1.000 | 0.019 |
| VPU | 0.461 | | 0.461 |
| sequencer (IMEM 2 x 1024x128) | 0.438 | 0.214 | 0.224 |
| MXU (systolic, 32 x 2) | 0.267 | 0.030 | 0.237 |
| quantizer | 0.195 | | 0.195 |
| DMA | 0.072 | 0.037 | 0.035 |
| ACT RAM | 0.031 | 0.026 | 0.005 |
| collective unit | 0.010 | | 0.010 |

The banked TMEM's logic (bank muxes, copy selection, the write stage) is 19 K um² (small) and 53 K
um² (the card before r2): its cost is the three copies' macros, 38% of that card's synthesized
area (the card's area per unit: "The card: release r2", above).

### Critical paths

nangate45-small, final (`tools/asic/paths.py small 300 6_final`: 91 K endpoints, signoff
parasitics). The worst path of each kind:

| kind | endpoints | worst slack | the worst path |
|---|---|---|---|
| io | 435 | -0.452 ns | DMA stream state `fs_k` -> the slice's SW port arbitration -> port `sw_req` (27 levels) |
| control | 19 K | 0.114 ns | sequencer `q_pr` -> `c_fp` (50 levels): the dispatch footprint of the queued command, `fp_ranges` then `fp_seg` |
| dma | 52 K | 0.398 ns | DMA stream state `fs_k` -> the TMEM arbitration -> the quantizer's grant -> its enables, into its block buffer (39 levels); the same source reaches the VPU's registers (8.6 K endpoints at 0.57 ns) |
| tmem | 4.1 K | 0.533 ns | DMA `fs_k` -> TMEM bypass hit (`bh`) |
| fpipe | 8.1 K | 0.544 ns | DMA `fs_k` -> the quantizer's fadd s1 (37 levels): an enable again |
| sram | 3.9 K | 0.593 ns | DMA chunk buffer macro -> TMEM bank macro (write data) |
| systolic | 3.6 K | 0.765 ns | MXU PAIR column select (`hi0`) -> PE |

Per stage class, the worst path out of it (its own logic) and into it from its operands:

| stage class | out of it | into it, from its operands |
|---|---|---|
| otpu_fmul s1 (the significand product) | | 0.657 ns from an fmul's result, 0.692 from an fadd's (19-21 levels): about 1.8 ns |
| otpu_fmul s2 (normalise, round) | 1.353 ns (13 levels) | |
| otpu_fadd s1 -> s2 | 1.055 ns (29 levels) | |
| otpu_fadd s2 -> s3 | 0.831 ns (26 levels) | |
| otpu_fadd s3 -> s4 | 1.267 ns (21 levels) | |
| systolic PE -> PE | 1.102 ns (19 levels) | 1.017 from the weight data |
| systolic column end | 1.106 ns (25 levels) | |
| sequencer | 0.114 ns (50 levels) | |

Into every fp stage register the worst path is an enable (0.54-0.59 ns: the DMA's stream state, the
MXU's `al_i` or the VPU's issue test through the arbiter), not the arithmetic: the units stall by
enable, and the grant reaches thousands of registers. After CTS the order was the same except that
the VPU's issue test (`ic` -> the arbiter -> the quantizer's block buffer, 0.29 ns) came second;
routing's repair moved it behind the DMA's.

**LM / LA.** `otpu_fmul` is two stages and `otpu_fadd` four whatever LM / LA say: a larger LM or LA
appends delay registers after the last stage (`otpu_delay u_pad`), it moves no logic. Raising them
shortens no path unless synthesis retimes (ORFS's `SYNTH_RETIME_MODULES`, experimental), so a
lm3la6 run would time the same as the default and was not run. The fp stages are not the limit
either: their own logic has 0.66-1.35 ns of slack at 2.5 ns (the product stage about 1.8 ns,
fadd's second stage 1.67 ns), against 0.11 ns for the sequencer and 0.40-0.60 ns for the
arbitration and enable paths. A faster fp datapath (a split multiplier: the XLS pilot's +27..48%
clock on the operator) pays once those are fixed, from about 540 MHz.

### Open problems

- **The ports.** The SW and port B requests (`sw_req`, `sw_addr`, `sw_wdata`, `b_wmask`, ...) left
  the block combinationally from the DMA's and quantizer's state through the slice's arbitration
  (12-27 levels): with 20% of the period outside, 339 MHz. A register stage at the boundary (as the
  platform's adapter would add) frees them, and every port has one now, on every target (the
  card's too: one design):
  - The SW port's register took the small block at 3.0 ns from 21 routed port violators to none
    ([asic_timing_loop.md](asic_timing_loop.md)).
  - Port B's output register (`b_req` ... `b_par`) and input register (`b_rvalid`, `b_rtag`,
    `b_rdata`). At the card point (a8, after CTS, 2.5 ns) every one of port B's 1,092 output
    endpoints missed, worst `b_wdata` -2.50 ns from the DMA's write-data rotator (18 buffers and
    six MUX2 levels from one select flip-flop), `b_wmask` -1.28, `b_par` -1.25, `b_addr` -0.97,
    `b_req` -0.85. Those figures charge the core clock's 1.9 ns insertion delay to the outputs
    against the ideal virtual I/O clock; with it credited on both sides, `b_wdata` is -0.59 and
    the rest meet (+0.62 ... +1.16), and the inputs read `b_rdata` -> the DMA's line-buffer macros
    -0.04, `b_rvalid` +0.25, `b_rtag` +0.27. The routed small block met at 2.5 ns (`b_wmask`
    +0.09 worst): port B is a card-size problem. The a8 point's global routing overflowed only at
    the DMA's line-buffer macros on the top edge, where 967 `b_wdata` pins and the buffers' read
    data meet; the register also lets the pins' drivers sit by the pins, away from the DMA.
  - Port A's output register (`a_req`, `a_addr`: the MXU's scale reads), the register the card's
    DRAM adapter (`otpu_native_dram`) took them into, moved into the slice: the card has one
    register between the MXU and the adapter's queues as before. At the card point `a_req` was
    -1.62 ns against the ideal I/O clock (+0.25 credited), from the DMA's request through port
    B's grant and the MXU's issue, `a_addr` -0.93 (+0.94), and `a_rdy` and `b_rdy` crossed the
    block to `a_req` (input to output, +0.11 and -0.02); the routed small block met (+0.15). Its
    read data needs no register: the MXU registers `a_rdata` at the port (`a_rvalid` +0.41,
    `a_rdata` +1.73, credited).
  - The registers' valid bits are the contract: a request waits in a register while `sw_req` /
    `a_req` / `b_req` is high, and port B's read data is the block's a cycle after its
    `b_rvalid`. So a platform's "every write the block gave is taken" is `wr_idle && !sw_req &&
    !(b_req && b_we)`, and "no read of the block's in flight" is `rd_idle && !a_req && !(b_req &&
    !b_we)` with no `b_rvalid` the cycle before (as `otpu_board`'s STATUS WR_IDLE and run start;
    port A's requests are reads).
  - `a_rdy` and `b_rdy` still reach the slice's grant logic combinationally (-> the MXU's tag
    count: `b_rdy` -0.30 ns, `a_rdy` -0.18, with the insertion delay credited): a registered
    ready (a credit: the memory's ready a cycle ahead, or a two-entry skid buffer) waits for the
    card's routed timing with the TMEM placed by script.
- **Sequencer dispatch.** `q_pr -> c_fp` (the footprint of the queued command, 50 levels) is the
  worst register-to-register path; `fp_ranges` and `fp_seg` in two cycles would give it a full one.
- **Grant fanout.** The DMA's stream state -> the arbitration -> the quantizer's and the VPU's
  enables (39 levels), a unit's request -> the banked arbiter -> another unit's grant -> its
  enables (43-45): the stall-by-enable design puts a high-fanout enable behind the arbitration. A
  registered grant (a cycle from request to grant) or per-unit clock gating is the lever. On the
  card (the TMEM by its readers) the worst register-to-register path is the VPU's stream grant
  (`ss_act`, `ss_gnt`): one flip-flop (its `max_fanout` is Vivado's, synthesis here ignores it)
  whose Q and QN drive 2467 gates through 190 buffers, into 2692 registers. The worst of them is
  the SE tail's VOP-mode select (`cm` -> `uc`: each lane's `u_m` operands `ma` / `mb`, 512 mux
  selects) ahead of the fmul's significand product: 10 buffers across the VPU (0.50 ns), the mux,
  the product's carry-save tree into its s1 register (17 levels), -0.09 ns on the global route
  (504 endpoints; +0.17 ns with the TMEM in the corner, its registers then less spread). The
  lanes now take registered copies of the mode, three a lane, loaded with `ss_act`'s next value
  so each equals it in the same cycle (`otpu_vpu` `g_sar`: `sa_r` for the lane's RDOT term,
  `cm_u` / `cm_q` for the tail's U and Q operands; each a kept instance, since identical
  flip-flops merge, and ORFS's slang import keeps the attribute): the word-level selects on the
  one flip-flop go from 2181 to the control's 133, and each copy carries one lane's 64-96. On the
  card's global route (r4, above) the grant's worst path is +0.447 ns (-0.091 as one flip-flop),
  off the critical list.
- The macros are CACTI-fitted estimates (`tools/asic/fakeram.py`), not a compiler's; a real PDK's
  1R1W macros (IHP SG13G2, commercial) replace them through `rtl/tech/asic`'s leaves.
- Power is vectorless; a VCD from the Verilator RTL (a decode token) would give real activities.
- The TMEM's unpumped true dual-port RAMs (TDP without PMP, the card's) go through
  `otpu_ram_1rw1r` under synthesis (the ASIC's macros); on an FPGA that costs 703 LUTs (+4.9%:
  Vivado out of context, `otpu_tmem` 14,410 -> 15,113, with the same 64 RAMB36, flip-flops and
  READ_FIRST port A) against inferring them in the TMEM, the logic inside the RAMs' instances. No
  FPGA target uses that mode (theirs are pumped), so it is a note, not a fix.
- **The weight chunk's layout** (a data-movement fix, not made). The MXU's FIFO holds the D-byte
  chunk as DRAM brings it, and array position i takes byte i in int8 but nibble i of the chunk's
  low half and nibble i of its high half in the 4-bit formats (`otpu_mxu` `wsel` / `dec4`: int4,
  E2M1, NVFP4; the high half of a 4-bit advance, PAIR's odd block). So FIFO bit b feeds two
  positions, b/8 and b/4 (low half) or b/8 and (b - 512)/4 (high half), up to 64 positions apart:
  on the card's column floorplan (ACT_ROWS 8, after placement) the 1024 FIFO read nets carry 0.66 m
  of wire, 649 um each on average and up to 1.17 mm, nearly all of it along the FIFO's stack, where
  the placer puts the decode. A 4-bit layout with position i's two elements in byte i (the low
  half's element in its low nibble, the high half's in its high nibble) gives every position its
  own byte in every format: each FIFO bit has one reader, in the chain beside its macro, which
  saves most of that wire (about 0.5 m: a net from a macro to the chain level with it, about
  0.15 mm, against 0.65 mm) and the decode's nibble muxes. It changes the packer (the 4-bit
  weight images), the ISA simulator's decode and the RTL's: a decision of its own.
