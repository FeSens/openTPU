# Timing: the wire, and what flip-flops buy

The question (2026-10-16): the card's clock is limited by route delay, not by logic, so can added
flip-flops fix it, and what else in the flow can improve? This is the exploration's record: where
the delay is (the FPGA card, the ASIC card), what the tools do alone, three batches of
cycle-exact register copies (branch `worktree-timing-ffs`), FPGA builds at 133.33 MHz and at a
150 MHz equivalent, the ASIC flow's timing repair, (the follow-up question: reduce the
fanout) a census of what the card's nets fan out to and the batches that cut it, and (the third:
cut the MXU from TMEM, within 1% of the cycles) the MXU's TMEM interface, and (the fourth:
a real 150 MHz with comfortable margins, for production) the 150 MHz builds and what the
card's DRAM made of them.

**Short answer.**
- **On the FPGA, flip-flops help where the long wire carries a register that is already a cycle
  old.** A copy per consumer group, each loaded with the same next value, sits by its loads and
  changes no cycle (section 3). With the first batch the card closed 133.33 MHz without r22's ECO.
  But the copies must stay narrow (enables, indices, counters, slots). One wide copy, the ACT
  RAM's write stage per column (+9 k LUTs, +9.5 k flip-flops), congested the 87%-full device and
  made the 150 MHz equivalent worse everywhere (-0.70 -> -1.33 ns), so it is reverted. With the
  narrow copies only (batches 1-3 without it), the 150 MHz-equivalent build goes from main's
  -0.696 ns and 8,473 failing core endpoints to **-0.255 ns and 1,577**: ~129 -> ~140 MHz at the
  clock crossings, ~136 -> ~145 MHz core to core, for 0.3% more LUTs and 2.6% more flip-flops;
  at 133.33 MHz the core then closes without the production strategy too.
- **Flip-flops do not fix a same-cycle decision**: an arbiter's grant into the enables of
  thousands of registers, the TMEM's read arbitration, the sequencer's start selection, the MXU's
  stage enables (`h[acc]`, fanout 11.7 k). Those need another cycle in a protocol (a registered
  grant with a skid, a start a cycle later) or a different structure (section 5). Vivado alone
  (replication, re-placement) reaches ~137 MHz on E3; main's own 150 MHz-equivalent build ~136
  MHz core to core and ~129 MHz at the clk -> clk2x crossing.
- **Most of the card's fanout was LUT RAM addresses, and block RAMs read ahead remove it**
  (section 6): 56% of the loads on nets of 200 or more were LUT RAM address pins. The widest
  memories as block RAMs read at their pointer's next value (a cycle early, so the same cycles)
  and late pushes off LUT RAM write enables: at the 150 MHz equivalent the failing endpoints go
  2,066 -> 820, TNS -177 -> -40 ns, the clock it reaches ~140 -> ~145 MHz, for 7.9 k fewer LUTs
  and 90 more block RAM tiles. A second level of the VPU's enable copies did not help (reverted).
  The MXU's TMEM grant (11.3 k enables) was what was left.
- **The MXU's grant is off its epilogue, at no cost in cycles** (section 7). On the card TMEM
  replays any lost read, so the MXU's reads no longer wait for the grant; the grant takes only
  its writes, from a ring of three, and the epilogue advances on four flip-flops. The five
  co-simulations move by at most +0.004%. At the 150 MHz equivalent: core -0.202 -> **-0.057
  ns**, failing endpoints 820 -> 141, TNS -40 -> -2.5 ns, the clock crossings positive, ~145 ->
  **~149 MHz**; at 133.33 MHz the card meets timing with Vivado's default implementation (core
  +0.012 ns, every clock positive). A first version that queued the read with the writes behind a registered advance
  was worse (-0.489 ns): the grant's fanout only moved, onto 118 replicas, and the queue put a
  LUT RAM read in front of TMEM's read address.
- **On the ASIC the gain is in the flow, not the RTL.** The card's timing repair never ran: CTS's
  repair saw only ports (it runs before their clock latency is set), and the global route's
  `repair_design` does not finish on 2.5 M nets. With the global route's repair as
  `repair_timing` only (`grt_repair.tcl`), main's card goes from 2.497 to **2.000 ns** register
  to register (400 -> 500 MHz on the route's parasitics) in 100 s of repair; CTS's repair now runs
  after the latency (`cts_latency.tcl`: 1 real violator instead of 1,232 ports). Batch 1 (with
  the ACT RAM copy) cost the ASIC 1.2% of area and 11% of wire for 0.008 ns (section 4).
- **A real 150 MHz closes with margin, and the card runs at it since 2026-10-19** (section 8).
  With the units' TMEM reads by bank from flip-flops and a setup margin in the flow, r150fp met
  150 MHz with core +0.130 / clk2x +0.154 ns. On the card every 150 MHz bitstream with the MXU's
  array on DSP X1-X8 read ECC errors back under inference, up to hundreds of uncorrectable
  words, where the same RTL at 133.33 MHz reads clean: the array's switching beside the DDR3
  PHYs closes their read windows. With the array on X3-X10 (r150o, core +0.079 ns) and the read
  taps chosen under prefill's load, each lane's eye included (`opentpu.host.loadcal`), its
  deploy window and 5 h soak read no corrected word, and r150o is production; for 4% of prefill
  and under 1% of decode.

## 1. Where the delay is

**FPGA (E3, the production m8 build, 039b34d5, 133.33 MHz, WNS +0.007 ns).** The 260 core-clock
paths within 0.2 ns of failing: 0.81 ns of logic and 5.67 ns of route on the median (88% route),
45% through at most one LUT, 55% with a net of fanout 30 or more, 40% crossing units, spanning up
to 246 slice rows. The units alone are not much faster: every recent out-of-context unit run at
7.5 ns had 0.07-0.5 ns of margin (~134-142 MHz).

**ASIC (r4, nangate45, global route without repair, 2.67 ns register to register).** The worst
117 register-to-register paths: 28% of their delay the launching flip-flop's clock-to-Q (weak
flip-flops driving long nets: unrepaired), 37% buffers, 35% logic; 68 of the 117 start at one
register, the MXU's decoded weight scale (`dhi1`) into every column.

The paths by the fix they need:

| class | example | fix |
|---|---|---|
| A: a same-cycle value fanned out | the TMEM pump's write stage, the MXU's decoded scale | a register copy per consumer group (cycle-exact) |
| B1: a registered enable with a huge fanout | the VPU's `en_r` (21 k loads) | a copy per lane, loaded with `en_r`'s next value (cycle-exact) |
| B2: a combinational enable | the MXU's TMEM grant and `h[acc]` -> its stage enables (~11.7 k registers, 8 levels) | a registered grant and a skid buffer (protocol) |
| C: cross-unit arbitration | the MXU's `t_raddr` / `t_ren`, the TMEM's `rpv` (10-13 levels) | a request stage (protocol) |
| D: the sequencer's dispatch and start | `sv`, `sstarted`, `soh` -> `can_start` -> every unit's command register | the write side: a copy (batch 3); the start side: a protocol or a structure |
| E: hard-block hops | a DSP chain's end -> the column end, a block RAM / SRAM macro -> far registers | FPGA: +1 cycle; ASIC: the repair's buffers (section 4) |
| F: IP | XDMA, LiteDRAM, `u_split` at 250 MHz | not ours |

## 2. The tools alone (no RTL change)

On E3's checkpoint before the userclk1 reroute, with the setup checks tightened to a 150 MHz
equivalent (`set_clock_uncertainty -setup` +0.833 ns core to core, +0.417 ns on the clk2x paths;
150 MHz itself has no even MMCM divide for CLK2X at `bd_native.tcl`'s fixed 800 MHz VCO): the routed design at
-0.826 ns (9,485 failing endpoints); post-route phys_opt AggressiveExplore -0.802 (one cell).
Placing again under the tighter checks: placed -0.974, phys_opt -0.706 (~860 flip-flops
replicated), routed -0.665, post-route phys_opt -0.620 ns (13,038 failing): **~137 MHz is what
Vivado's replication and re-placement reach alone.** What fails there: the sequencer (5,787
endpoints), the VPU's `en_r` (1,079-1,164), the MXU's `t_raddr`; 84% with fanout >= 30, 86%
route-dominated. Vivado replicates the driver register but not the logic behind it, and its
replicas are placed by the loads it sees, not by the consumer groups of the design.

## 3. Flip-flops in the RTL: copies per consumer

Every copy is loaded with the same next value as the register it copies (so the design's cycles
are unchanged), kept (`(* keep_hierarchy = "yes" *) otpu_delay`, or a register per generate
slice: identical flip-flops would merge in synthesis), and checked in simulation against the
original where one stays.

**Batch 1 (bc92de2f).**
- The pumped TMEM's write stage a copy per block RAM (`otpu_tmem` `g_pmp` `g_sl`, `otpu_pump_bank`
  W): each RAM's slice of the write data, address and valid.
- The MXU's decoded scale a copy per column (`g_bd`, `dlo1` / `dhi1` / `vlo1` / `vhi1` arrays);
  the result FIFO's write counters a copy per row slice (`g_rfm` `g_s`).
- The VPU's fill index a copy per lane (`g_sfi`; `otpu_se_tail` `fi` per lane).
- The ACT RAM's second hop a copy per column (`g_row` `u_h`). **Reverted (aa3ec024):** the copy
  took the wide write stage with it (the byte enables and bytes `wbe` / `wd`, 1,152 bits, and the
  lanes' placement crossbar, per column): `u_act` 1,603 -> 10,648 LUTs and 1,863 -> 11,404
  flip-flops on the m8 card. Main already gives each RAM piece its own copy of the write address
  and select, which is the narrow part.

**Batch 2 (be935256).** The VPU's pipeline enable a copy per lane (`g_enl`: `en_l`, loaded with
`en_r`'s next value; checked `en_l == {LANES{en_r}}`): the lanes' stages, the SE tail's units
(`cen` per lane) and the composites' lanes (`otpu_se_comp` `en_l`) take their lane's.

**Batch 3 (ec27459e).** With wait counts (`SEQ_WC`, the card and the nangate45 card), the
sequencer's command store is written a cycle after the dispatch, from flip-flops: the enable and
the slot a kept copy per unit, the command one register (`otpu_seq` `g_cmd` `g_wq`). A slot starts
two cycles after its dispatch at the earliest (its ready flags `w_all_q` / `w_dram_q` clear in the
dispatch cycle), so its command is in the store by then; a simulation check stops any start of a
slot whose write is pending. At the 150 MHz equivalent (section 2's run) these were 3,036 LUT RAM
write-address pins (`sv` -> the free slot, -0.49 ns) and 868 write enables (`vout` -> `c_go`,
-0.34 ns) of the sequencer's 5,787 failing endpoints.

**Verification.** Every batch and the final tree (aa3ec024: batches 1-3 without the ACT RAM copy):
`tests/test_rtl.py` 176 passed, 1 skipped (as main); the SE, SE composite, stream, quantizer, fp,
wait-count, models-on-board, target and ASIC flow tests all passed; the m8 card co-simulations
(RTL with LiteDRAM, 133.33 MHz) cycle-exact against main itself: the tiled P run 950,732 cycles
with every busy counter equal, Qwen3.5-0.8B's decode at 544 2,539,392, Qwen3-0.6B's at 544
1,617,739 and at 1800 1,708,259, the MoE decode step (Qwen3.5-35B-A3B) 4,489,345.
`tests/test_target.py` `test_the_long_wire_drivers_are_copies_per_consumer` keeps the copies (and
the ACT RAM's write stage one register).

**FPGA at 133.33 MHz (b_t1: batch 1 with the ACT RAM copy, the production recipe).** Closed without
the ECO: WNS +0.002 ns, WHS +0.033, the clk -> clk2x crossing +0.067 and back +0.050 (r22, main's
RTL, failed clk -> clk2x at -0.202 and needed the block-RAM swap ECO). 195,926 LUTs (65.6%),
174,604 flip-flops, slices 89.8%. Under +0.2 ns: 730 core endpoints (r22: 713); the worst the
sequencer's `soh` into the MXU (+0.002), the DMA's `t_re_q` into the VPU (+0.007, fanout 464),
XDMA, the MXU's DSP chain to the column end (+0.026).

**FPGA at a 150 MHz equivalent** (the production recipe at 133.33 MHz with the uncertainty above
appended to `otpu_clk2x.tcl`; core to core the period it reaches is 6.667 ns - WNS, at the clk2x
crossings 2 x (3.333 ns - WNS)):

| build | main (b_t0_150, 5873d298) | batch 1 with the ACT RAM copy (b_t1_150, bc92de2f) | batches 1-3, narrow (b_t4_150, aa3ec024) |
|---|---|---|---|
| core -> core WNS, failing | -0.696 ns, 8,473 | -1.328 ns, 19,055 | **-0.255 ns, 1,577** |
| core TNS | -1,911 ns | -6,829 ns | -142 ns |
| core -> clk2x / clk2x -> core | -0.531 / -0.446 | -0.529 / -0.528 | -0.241 / -0.225 |
| clk2x -> clk2x | -0.369 | -0.418 | -0.150 |
| core to core reaches | 7.363 ns, 135.8 MHz | 7.995 ns, 125.1 MHz | **6.922 ns, 144.5 MHz** |
| with the crossings | 7.728 ns, 129.4 MHz | 7.724 ns, 129.5 MHz | **7.148 ns, 139.9 MHz** |
| LUTs / flip-flops | 185,403 / 161,219 | 196,565 / 174,503 | 186,027 / 165,387 |
| slices | 87.1% | 89.6% | 88.3% |
| level-5 congestion windows | none | North 111% | North 104% (u_mem 60%, u_mxu 21%) |

The narrow copies take main's 150 MHz-equivalent failures from 8,473 core endpoints to 1,577
and the clock it reaches from ~129 to ~140 MHz (the clk2x crossings, where the pumped TMEM's
write stage per RAM sits), for 624 LUTs and 4,168 flip-flops. One build per configuration, so a
difference of a tenth of a nanosecond could be the placement's; these are several times that.
What is left is flat (-0.15 to
-0.26 ns, every unit): the per-lane enable copies themselves (`g_enl`, 2,553 loads each, 6.2 ns of
route: a second level of copies), the pump RAMs' outputs (4 levels), and the protocol classes of
section 5 (the TMEM's `rpv`, the MXU's `lvm` with fanout 11,778, the sequencer's `w_all_q`).

**Fewer flow crutches (b_t4_def: aa3ec024 at 133.33 MHz, `impl_directives.tcl` empty).** Without
the production strategy (`Performance_ExtraTimingOpt`; build.tcl's own `Explore` directives for
opt, place, phys_opt and route, and its post-route phys_opt, stay) and without r22's ECO, the
core domain closes: core +0.008 ns, clk2x +0.065, the crossings +0.005 / +0.001, WHS +0.012;
186,027 -> 187,606 LUTs, slices 88.3 -> 81.4% (the strategy spreads the placement). The one
failing endpoint is the PCIe block's own 500 MHz path into its TX block RAM (`userclk1`, -0.004
ns): `userclk1_reroute.tcl` took it to +0.029 but another clock to -0.053, so it reopened the
routed design. The strategy is now a margin, not the closure: at 133.33 MHz the copies close the
core either way, and the PCIe IP's path is the reroute hook's (its acceptance rule could weigh a
-0.053 core slack against a failing PCIe clock).

## 4. The ASIC flow

**Main against batch 1 on the same floorplan** (r3's macro placement and blockages, the die
pinned at 4530 um so main's 0.06% smaller synthesis does not move a macro row off the core,
global route without repair, the route's parasitics, 2.5 ns):

| | r4 (archive, 1aad9195) | main (openTPU-t0, 5873d298) | batch 1 (openTPU-t1, bc92de2f) |
|---|---|---|---|
| synthesis | | 2115 s, 9.27 mm² | 3355 s, 9.36 mm² |
| after CTS (design area, macros in) | | 9.47 mm² (46%) | 9.58 mm² (47%) |
| global route wire | | 106.6 m | 118.4 m |
| endpoints | 156,535 | 156,410 | 176,498 |
| register to register | 2.666 ns, 32 failing (all `dhi1` -> `bm2`) | **2.497 ns**, none failing | **2.489 ns**, none failing |
| within 0.1 / 0.2 / 0.3 ns | 62 / 84 / 120 | 4 / 9 / 30 | 2 / 6 / 48 |
| the worst | `u_mxu:dhi1` -> `u_mxu:bm2` | the slice's SRAM -> the DMA's chunk buffer (+0.003) | the slice's SRAM -> a TMEM bank (+0.011) |
| ports | +1.244 | +1.169 | +1.094 |

Main already routes at the clock: the r2.2 cuts and the pinned floorplan took `dhi1` off the
list (r4's limit was one weak, unrepaired flip-flop of it), and what is left near 2.5 ns are SRAM
macro to SRAM macro hops. Batch 1 there (with the ACT RAM copy, since reverted; the narrow
copies alone were not run on the ASIC) costs 1.2% of area, 11% of wire and 20 k flip-flop
endpoints for 0.008 ns: the copies are the FPGA's fix (a register next to its loads where route,
not drive, is the delay); on the ASIC that wire is the repair's (buffering and sizing). Before
they reach the ASIC the copies want a target switch: a parameter, on for the FPGA targets.

**CTS's repair saw only ports.** ORFS's CTS step runs `repair_timing` and then `POST_CTS_TCL`
(`io_latency.tcl`: the ports' virtual clock takes the registers' median clock arrival, 1.95 ns).
Before that the ideal virtual clock charges the insertion delay to every output path: on main's
card the repair found 1,232 violating endpoints, every one a port (worst `b_addr[30]` -0.95 ns),
worked 125 s, resized 1,488 cells and inserted 355 buffers on those paths and stopped (RSZ-0062,
-0.788 ns); r4's did the same. With `io_latency.tcl` sourced first (now `asic/nangate45/cts_latency.tcl`,
`PRE_CTS_TCL`, every point): one violating endpoint (the sequencer's `ir_sync`, -0.12 ns), 61 s, 6
changes, the CTS result's register-to-register period 2.62 -> 2.59 ns (the placement's
parasitics), the same hold repair (786 buffers).

**The global route's repair, as `repair_timing` only.** ORFS's GRT step runs `repair_design`
(every net's slew, capacitance and fanout, single-threaded) before `repair_timing`; on r3 (2.5 M
nets) it ran 8 h before it was stopped, so the card's numbers were unrepaired. With the step's
`repair_design` a no-op (`asic/nangate45/grt_repair.tcl`, `PRE_GLOBAL_ROUTE_TCL`, the card point:
placement has run it) the repair works on the failing paths only. On main's card (t0's CTS
result) with ORFS's `SETUP_SLACK_MARGIN` 0.5 (the repair's target: 2.0 ns): 100 s of repair, 41
buffers, 295 cells upsized, and on the route's parasitics register to register **2.497 -> 2.000
ns** (every register-to-register slack at least +0.500 ns at 2.5), the ports unchanged
(+1.169). The SRAM hops that limited it (+0.003 ns) were unbuffered macro outputs, +0.501 after.
The whole step ~2,380 s (2,176 without the repair; its wall time less the 13 min it was paused
while t1's global route ran), peak 33.9 GB (27.5). With a 1.0 ns margin the repair reached
1.99 ns within ~3,000 moves and then stalled (+0.512-0.514 ns over the next 49,000; the VPU's
floating-point stages, the SE tail's `u_ma` / `u_c`; stopped after 2 h): the floor of this
placement without a pipeline change. These are global route numbers (the guides' parasitics);
detailed routing has not been run on the card.

## 5. What flip-flops do not fix

- **A combinational enable (B2).** The MXU's TMEM grant and its `h[acc]` reach ~11.7 k registers'
  enables in the cycle they are decided (8 levels, the worst core path of main's 150 MHz
  equivalent, -0.70 ns). A registered grant moves the decision a cycle ahead of its use: the unit
  must hold its request's data a cycle longer or take a skid buffer of the cycle's beats.
  Section 7 does it for the MXU: its reads off the grant (TMEM replays them), its writes in a
  ring the grant pops.
- **Cross-unit arbitration (C).** The TMEM's read selection takes every reader's address and
  enable in the cycle they are formed (10-13 levels; the MXU's `t_raddr` / `t_ren`, the TMEM's
  `rpv`): a request register in each reader costs a cycle of read latency and the arbiter's view
  of the requests.
- **The sequencer's start (D).** `can_start` into every unit's command register's enable and the
  start slot's read into its data (1,625 endpoints, -0.60 / -0.62 ns at the 150 MHz equivalent)
  are the start decision itself. A per-unit dispatch-order queue would give the in-order units
  (DMA, MXU, quantizer, collective) their next slot from a register (its head), so their command
  read starts at flip-flops; the VPU starts its oldest ready instruction and keeps the age-matrix
  selection.
- **Hard blocks on the FPGA (E).** A DSP chain's end or a block RAM's output into far registers
  takes a cycle of pipelining and its latency in the schedule.
- **Wide data.** A copy of a wide bus per consumer multiplies its registers and the logic that
  feeds them; on a device at 87% of its slices that is congestion, and congestion lengthens every
  route (section 3, the ACT RAM).

## 6. Fanout: what the nets reach, and what was cut

**The census** (`b_t4_def`'s opt_design checkpoint: aa3ec024 at 133.33 MHz; every signal net with 200
or more loads, its loads by pin kind). 562 nets carry 278,123 loads: **LUT RAM address pins
156,954 (56%)**, flip-flop enables 53,987 (19%), LUT inputs 25,723, resets 20,477 (most of them
XDMA's own), LUT RAM write enables 18,024 (6.5%). Over every LUT RAM of the design: 1,535 memories,
357,960 address pins, 39,880 write enables. A LUT RAM takes the whole address on every cell (a
RAM32M holds 6 bits of a 32-deep memory), so a 512-bit queue is a fanout of ~86 per address bit
from its pointer register, and a wide memory's pointer was the widest net of its unit: the DMA's
gather (39,424 address pins), the native adapter's B response FIFO (38,528), the MXU's result FIFO
(37,632), the native adapter's A response RAM (19,152), its B queue (12,960), its SW queue's data
and fill data (11,980 each), the channel bridges' output FIFOs (11,980 each), the sequencer's
command store (10,820). The enables: the MXU's TMEM grant (11,254 enables: its whole epilogue,
held while TMEM does not grant), the quantizer's pipeline enable (8,254), the VPU's per-lane
enable copies (~2,500 each).

**Batch 5 (0a2750a4): the widest LUT RAMs as block RAMs read ahead.** `otpu_ram_1r1w_ra`
(`rtl/tech`, both families) is a LUT RAM's behaviour on a block RAM: `rd` is the word at the read
address register, every write to the last edge in it, from the address's next value (`ra_n`) a
cycle early; generic: a block RAM read at `ra_n` and a registered bypass of the word written at
that edge; asic: flip-flops and a multiplexer at the registered address. `tb_tech_ram` kind 4
checks the two families against each other (a third of its reads the word written at that
edge). The DMA's gather (`g_gb`, at `gh_n`), the B response FIFO (`g_rb`, at `rb_hn`; its tail
copies `rb_tq` gone) and the MXU's result FIFO (`g_rfm`, three copies at `rf_qn + c`) use it; each
next value is checked in simulation against its register. And two enables that held for nothing:
port B's input register in the slice (its readers take it only under `b_rvalid_i`) and TMEM's
input stage address and data with one write port per bank (taken only under `wi_gnt`).

**Batch 6 (8bf19b12): late pushes off LUT RAM write enables.** A queue written on its push puts
the push, decided late, on every cell's write enable. The in-order channel FIFOs (`otpu_afifo` OOO
0: `u_aq`, `u_ad`, `u_xq`, `u_xd`) write the tail slot every cycle there is room (`wready`, the
pointers' registers), and the native adapter's A queue (`qam`, 2 QD slots) its tail every cycle,
as its B queue already did; a push only moves the tail. (A first version took the room a cycle
old: the push the synchronized read pointer had just made room for was then not written; the
board model's turnaround test caught it.) The SW queue's data and fill data (`u_qwd`, `u_wrm`) are
block RAMs read ahead at `qw_fn`. The MXU's ASCALE factors (`alpha`) load every cycle the words are
there, not only at the grant (TMEM holds them until it; checked against a grant-only copy at
every read, and a mutation loading early is caught): 90 of the grant net's 127 failing endpoints
in b_t4_150. `tools/memch_test.py` (the bridge's own bench, func and mutations) passes as before.

**Batch 7 (0c0b16c8), reverted (402bb3bd): the VPU's enable copies per lane and per consumer.**
`g_enl` kept four copies per lane, each loaded with `en_r`'s next value: the lanes' own stages,
the composites, the tail's units, and the tree's lanes (`otpu_vtree` took `en_l` for its row
levels). Each copy's loads went from ~2,700 to ~990, but their routes did not shorten (0 levels:
-0.186 ns in b_t6_150, -0.424 in b_t7_150): a lane's consumers spread as far as the lane does,
so a copy per consumer still crosses the same distance. And the build was worse everywhere, in
units the change does not touch too (core -0.517 ns, 3,794 failing, TNS -575; the sequencer
-0.150 -> -0.517). One build each, so part of that may be the placement's, but nothing in it
shows the second level helping. A copy helps where its consumer group is compact (the MXU's
columns, the pump's RAMs); the VPU's next step is its placement (a pblock per lane), not more
copies.

Every batch: `tests/test_rtl.py` 176 passed, 1 skipped; the SE, composite, tail, stream, quantizer,
fp, wait-count, vops, models-on-board, board, target, tech RAM and ASIC flow tests pass; the five
co-simulations are cycle-exact (950,732 / 2,539,392 / 1,617,739 / 1,708,259 / 4,489,345 cycles, as
main).

**FPGA at the 150 MHz equivalent** (the same recipe and constraints as section 3's table):

| build | b_t4_150 (aa3ec024) | + batch 5 (b_t5_150) | + batch 6 (b_t6_150) | + batch 7 (b_t7_150, reverted) |
|---|---|---|---|---|
| core -> core WNS, failing, TNS | -0.255 ns, 1,577, -142 | -0.236, 1,731, -124 | **-0.202, 732, -36** | -0.517, 3,794, -575 |
| core -> clk2x / clk2x -> core | -0.241 / -0.225 | -0.006 / -0.029 | -0.122 / -0.003 | -0.145 / -0.247 |
| clk2x -> clk2x | -0.150 | +0.075 | -0.050 | -0.106 |
| failing endpoints, all clocks | 2,066 | 1,738 | **820** | 4,136 |
| reaches | 7.148 ns, 139.9 MHz | 6.903 ns, 144.9 MHz | 6.910 ns, 144.7 MHz | 7.184 ns, 139.2 MHz |
| LUTs (as memory) | 186,027 (30,535) | 178,435 (22,204) | 178,168 (20,949) | 178,384 (20,947) |
| flip-flops | 165,387 | 169,067 | 171,043 | 171,105 |
| block RAM tiles | 436 (45.7%) | 496.5 (52.0%) | 526.5 (55.1%) | 526.5 |
| slices | 88.3% | 87.0% | 87.0% | 87.1% |

Batch 5 takes the clk2x crossings, which bounded b_t4_150, from -0.24 to -0.03 ns (~140 -> ~145
MHz), and 8.3 k LUTs of memory out (-27%), for 3.7 k flip-flops (the bypass registers) and 60
block RAM tiles. Batch 6 halves the failing endpoints again and cuts TNS by two thirds; the
worst path moves. In b_t6_150 no failing path touches a LUT RAM. What fails there: the VPU's
per-lane enable copies (1,803 endpoints within 0.2 ns, 0 levels; batch 7 did not help), TMEM's `rpv` into the
VPU (12 levels), the sequencer's dispatch (`sv`, `vout`), the MXU's TMEM grant (`u_rw` `E`, 11,301
loads: its epilogue's stage enables) and the ACT RAM's write stage.

**The census after** (`b_t6_def`'s opt_design checkpoint: 8bf19b12 at 133.33 MHz, against
`b_t4_def`'s): LUT RAM address pins over every memory 357,960 -> **218,406 (-39%)**, LUT RAM write
enables 39,880 -> 26,457 (-34%), LUT RAM cells 53,226 -> 39,803; the loads of nets of 200 or more
278,123 -> **231,299 (-17%)**, those nets 562 -> 425 (their LUT RAM address loads 156,954 ->
114,198, enables 53,987 -> 51,145, LUT RAM write enables 18,024 -> 15,370; LUT inputs +2,459, the
bypass multiplexers). The five memories converted have no address pins left; the widest nets
are now the MXU's grant (10,783 enables), the quantizer's enable (8,029), XDMA's reset and the
VPU's per-lane copies (~2,500 each).

**At 133.33 MHz without the production strategy** (`b_t6_def`, as `b_t4_def` in section 3): core
-0.026 ns at one endpoint (the MXU's `mxm` into TMEM's read selection `rpv`, 90% route: class C),
clk2x +0.420 (b_t4_def +0.065), the crossings +0.111 / +0.066 (+0.005 / +0.001), the PCIe block's
`userclk1` +0.040 (-0.004); 181,260 LUTs (187,606), slices 81.0% (81.4%). Both miss by one
endpoint and a few picoseconds, so at 133.33 MHz the fanout batches are a wash on the worst path
and margin everywhere else (the clk2x paths +0.36 ns).

**What fanout is left**: the MXU's grant reaches its whole epilogue (the lanes, `r0` / `r1`, the
fmadd, the delay lines: 11.3 k enables) in the cycle TMEM decides it. It needs the protocol change
of section 5: the epilogue running free into a skid of its depth (a pipeline's worth of entries,
popped one per granted cycle, so the requests TMEM sees stay the same cycles), or a registered
grant. The quantizer's enable (8,254) is on no failing path. The remaining LUT RAMs are the
channel bridges' FIFOs (asynchronous: their reads cross clocks, so a block RAM there is a CDC
change) and the native adapter's A response RAM (read at `ra_h + drop`, two registers' sum).
Section 7 takes the grant off the epilogue.

## 7. The MXU and TMEM: the grant off the epilogue (TQ)

The question (the third): cut the MXU from TMEM's arbitration, at a cost of at most 1% of the
cycles. The slice grants the MXU's TMEM read and write ports as one group, and the MXU's epilogue
(the drain's lanes, the read-modify-write's fmadd and delay lines, RMAX, the ASCALE loads)
advanced on that grant: 10,783 enables in the census, decided in the cycle they are used. On the
card the grant is `gnt_m` (ARB_MASK, which TMEM_IMPL 3 with replays requires): every group's write
banks against those before it, and a group held while its port replays a lost read. So the
MXU's enables hung at the end of the other units' write requests: in b_t6_150 the VPU's write
buffer head (`hrot`) into the MXU, 720 endpoints at -0.138 ns, and the DMA's `t_wen`, 176 at
-0.063.

**A first version (72cbb745), measured and replaced.** `ea`, the epilogue's advance, became a
flip-flop; a beat (the read in the request register and the cycle's writes) the grant did not take
waited in a queue of two, with a bypass when it was empty; `ea_q`'s next value came from the
grant, the queue and `t_lost`. The co-simulations stayed within 0.05% and every test passed,
but b_t8_150 came out worse everywhere: core -0.489 ns, 5,236 failing, TNS -858. Three things
did it, each in the path report. The grant's fanout moved instead of going away: `ea_q`
(`max_fanout` 128) became 118 replicas, and their D net, 120 loads, hung at the end of the
arbitration (the grant arrives at 9.3 ns of 10.46). The queue (2 x 784 bits) became LUT RAM,
and TMEM's read address came out of its read at `tq_h`: +1.1 ns in front of TMEM's conflict
logic, into `lost` and on to the VPU's hold (-0.485 ns) and into the pumped RAMs' addresses
(-0.186, 604 endpoints). And the push, from the comb write beat and the grant, enabled every
queue flip-flop (1,643 endpoints, -0.487). The lesson: the grant may reach only a few
flip-flops, and nothing may go in front of TMEM's read address (b_t6_150: the MXU's `t_raddr`
into TMEM at -0.117 and +0.010 ns). With the read inside the granted beat, a read the grant does
not take must be held, either by the request register changing on the grant (its fanout again)
or by a multiplexer in front of the address.

**The interface (9b6279fc, TQ; the slice's `MXU_TQ`: TMEM_IMPL 3 with replays).** TMEM already
replays any port's lost read (`RPLM`; its age logic counts the MXU among the replaying askers):
only the slice's grant had kept the MXU's reads behind the writes' arbitration. Now:
- TMEM reads what the MXU asks, at an advance (`r_en[P_MXU]` not gated by the grant, as the
  VPU's reads); a read that loses a RAM is replayed and the slice holds the MXU's grant for it,
  as before. TMEM's read address is the request register itself (`rq_ad`), as release r2 built
  it.
- The grant takes only the writes, from a ring of three slots of flip-flops written every cycle
  the tail is free (their enables from the counters' registers), with the writes taken as issued
  when the ring is empty.
- `ea = room_q && !lost_q && (tq_z || !rq_any)`, all flip-flops: the room for one more cycle's
  writes counted without this cycle's pop (the third slot makes that cost nothing), no replay
  pending (`lost_q`: the lost read's word comes the cycle after its replay), and with a read in
  the request register an empty ring (a read never passes an older write; it is read with its
  advance's own writes, as one granted beat was). The grant ends at the ring's counters, `t_lost`
  at `lost_q`.

TQ is 0 (the grant as the enable) for every other TMEM: their grant decides reads (IMPL 1's
copies serve the port granted) or never replays them. Checks: in simulation, the ring's overflow,
`rq_any` and `tq_z` against what they summarize, no read with a write waiting, no loss of a read
not asked; `tests/test_target.py` pins the structure (the grant only in `take` / `pop`, `t_lost`
only into `lost_q`, `ea`'s four inputs, `t_raddr = rq_ad`, the slots' enables, the slice's
ungated read). test_rtl's fuzzers run the default TMEM, where TQ is 0, so the card's TMEM got
its own sweep: test_rtl's random programs (20) and a variant heavy on MXU reads (40: MMs of a
few shapes, three in four accumulating, VPU ops alongside) on R2_TMEM and R2_ARB, against the
ISA model: 11,891 MXU reads, 463 of them lost and replayed, all the ISA's. test_rtl 176 passed,
1 skipped; the other RTL groups 327 passed, 10 skipped (slow only).

**Cycles** (the five co-simulations, the m8 card's uarch):

| co-simulation | before (b_t6's RTL) | first version (72cbb745) | TQ (9b6279fc) |
|---|---|---|---|
| tile, Qwen3.5-0.8B 4 layers | 950,732 | 950,732 | 950,771 (+0.004%) |
| Qwen3.5-0.8B decode at 544 | 2,539,392 | 2,538,200 | 2,537,235 (-0.08%) |
| Qwen3-0.6B decode at 544 | 1,617,739 | 1,617,971 | 1,617,480 (-0.02%) |
| Qwen3-0.6B decode at 1800 | 1,708,259 | 1,707,951 | 1,707,656 (-0.04%) |
| Qwen3.5-35B-A3B MoE step at 544 | 4,489,345 | 4,489,345 | 4,489,345 |

The epilogue no longer stops for a withheld write grant (the ring takes the writes), and a read
waits for an empty ring instead; net, no run moves by more than +0.004% (the 1% budget was not
needed).

**FPGA at the 150 MHz equivalent** (section 3's recipe):

| build | b_t6_150 (8bf19b12) | b_t8_150 (first version) | **b_t9_150 (TQ)** |
|---|---|---|---|
| core -> core WNS, failing, TNS | -0.202 ns, 732, -35.7 | -0.489, 5,236, -858 | **-0.057, 140, -2.5** |
| core -> clk2x / clk2x -> core | -0.122 / -0.003 | -0.205 / -0.190 | **+0.021 / +0.065** |
| clk2x -> clk2x | -0.050 | -0.110 | +0.050 |
| failing endpoints, all clocks | 820 | 5,601 | **141** (1: the PCIe block's `userclk1`) |
| endpoints within +0.2 ns | 6,400 | 12,075 | 2,509 |
| reaches | 6.910 ns, 144.7 MHz | 7.156 ns, 139.7 MHz | **6.724 ns, 148.7 MHz** |
| LUTs / flip-flops | 178,168 / 171,043 | 179,729 / 171,844 | 178,141 / 171,987 |
| block RAM tiles, slices | 526.5, 86.95% | 526.5, 87.01% | 526.5, 86.94% |

The grant's classes are gone, the clock crossings pass, and TMEM's `rpv` paths with them (into
the pumped RAMs' addresses +0.108 ns, -0.122 in b_t6_150; into the VPU's hold no longer within
0.2 ns, -0.202). What is left within 0.06 ns: the MXU's fmadd adder (`g_rmw` `u_y`, 10 levels)
and the DSP outputs (-0.057: arithmetic), the MXU's ACT RAM read control `ck` (-0.044), the
VPU's enable copies (-0.044: 0 levels, 2,702 loads, placement), TMEM's read selection into the
VPU's hold from the MXU's `room_q` (-0.041, 12 levels: `t_ren` is one LUT of flip-flops), the
sequencer's start (-0.028), the quantizer (-0.026).

**At 133.33 MHz without the production strategy** (`b_t9_def`, as `b_t6_def` in section 6):
**timing met**, every clock: core +0.012 ns (b_t6_def -0.026 at one endpoint, the MXU's `mxm`
into TMEM's `rpv`), clk2x +0.078 (+0.420), the crossings +0.093 / +0.025 (+0.111 / +0.066),
`userclk1` +0.052 (+0.040), hold +0.002; 177,872 LUTs (181,260), slices 79.6% (81.0%). The card
then closes its clock with Vivado's default implementation, without r22's ECO or the production
strategy.

A copy of `rpv` per bank (for the asker
multiplexers, ranks, data select and lanes; TMEM's tests pass with it) was prepared for the
`rpv` classes and is not committed: in b_t9_150 they pass.

## 8. A real 150 MHz, and the card's DRAM

The question (2026-10-17): build the m8 card at a real 150 MHz (not the equivalent), give it
comfortable margins, and promote it. **The margins came (r150fp: core +0.130 ns, clk2x +0.154,
every clock above production E3's at 133.33 MHz), but at first the card was not clean at 150
MHz: under inference its DRAM reads came back with ECC errors, corrected and uncorrectable, in
every 150 MHz bitstream with the MXU's array on DSP X1-X8 (the TMEM moved away from the PHYs
included: one load clean, the next not), while the same RTL at 133.33 MHz reads clean. The
array's switching closes the PHYs' read windows. r150o (2026-10-19: the array on X3-X10, core
+0.079 ns) with the read taps chosen under prefill's load, each lane's eye included, read no
corrected word in its deploy window and 5 h soak, and is production.**

**The clock.** `bd_native.tcl`'s fixed 800 MHz VCO has no even divider for 150 MHz with the pumped
TMEM's core_clk2x; `core_clock.tcl` (18168e94) picks the VCO an even divider needs (150: 900 MHz /
6, clk2x 300 MHz) and keeps 800 MHz where its dividers serve (133.33, 100). The memory controllers
keep their own 133.33 MHz sys clock.

**The builds** (m8, the production strategy; slacks real, ns; `MARGIN` the core group's setup
margin, below):

| build | RTL | margin | core | clk2x | userclk1 / userclk2 | crg (sys) | WHS | the change |
|---|---|---|---|---|---|---|---|---|
| r150a | 18168e94 | - | -0.087 | +0.006 | +0.039 / +0.035 | +0.109 | +0.028 | the clock |
| r150b | 198e535c | - | +0.008 | +0.021 | +0.024 / +0.003 | +0.137 | +0.012 | b_t9_150's last paths (d85d6572, eb17a348) |
| r150c | 6501e15e | - | -0.007 | +0.041 | +0.058 / +0.096 | +0.286 | +0.000 | r150a's paths: the pumped word's select (cb7b81e4), `c_lz`, `vnx_q` (2f19b650), the VPU's back end on flip-flops (6501e15e) |
| r150d | 671b4265 | 0.15 | +0.037 | +0.043 | +0.048 / +0.002 | +0.122 | +0.012 | XDMA's write packing (64f896c9); the margin |
| r150f | 1da7d3b9 | 0.15 | +0.033 | +0.113 | +0.099 / +0.046 | +0.106 | +0.013 | the MXU's read by bank, `rpv` per bank |
| **r150fp** | 1da7d3b9 | 0.15 | **+0.130** | **+0.154** | +0.040 / +0.041 | +0.106 | +0.013 | r150f before the userclk1 reroute |
| r150h | f5eb87c5 | 0.15 | -0.286 | -0.086 | +0.054 / +0.012 | +0.063 | +0.015 | the TMEM in a hard pblock, X0Y3:X1Y4 (`OTPU_TMEM_PBLOCK`, between the PHYs' rows) |
| r150i | f5eb87c5 | 0.15, sys 0.3 | -0.167 | -0.062 | +0.074 / -0.073 | +0.278 | +0.011 | r150h's pblock, `OTPU_SYS_MARGIN` 0.3 |
| r150m | 1da7d3b9 | 0.15 | -0.686 | -0.447 | +0.049 / -0.176 | +0.187 | +0.016 | the MXU's array on DSP X4-X11 (`OTPU_MXU_DSP_COLS`), away from the PHYs |
| **r150n** | 1da7d3b9 | 0.15 | **+0.066** | **+0.124** | +0.016 / +0.008 | +0.054 | +0.001 | the array on DSP X2-X9 |
| **r150o** | 1da7d3b9 | 0.15 | **+0.079** | **+0.151** | +0.032 / +0.023 | +0.208 | +0.019 | the array on DSP X3-X10, the placer's ExtraPostPlacementOpt (`OTPU_PLACE_DIRECTIVE`) |
| E3 (production) | 039b34d5 | - | +0.007 | +0.019 | +0.012 / +0.041 | +0.015 | +0.018 | 133.33 MHz |

- **Setup margins by clock group** (671b4265, 65f5f357): `OTPU_SETUP_MARGIN` adds that much setup
  uncertainty to every check between core_clk and core_clk2x, `OTPU_SYS_MARGIN` to the memory
  controller's sys clock to itself, as a late implementation-only constraint (`otpu_margin.tcl`
  in OUT_DIR); build.tcl reports each group's slack with it (SUMMARY.txt's MARGIN lines) and sets
  it back to 0 before the reroute, the reports and the bitstream.
- **The reads by bank** (1da7d3b9, bc7c33a2): r150c / r150d's worst classes ran from the units'
  request registers through TMEM's merge of their lanes (the MXU's crossbar, the rotators) into
  the ranks, the conflict compares and the pumped RAMs' read pins (8 levels, 5 ns of route). The
  units now decode their next request to TMEM's banks before the register (`otpu_tmem_breq`; the
  MXU's `rb_q` / `rb_a`) and TMEM takes the banks as they come (`PBR`); `rpv` is kept per bank
  (`rpvb`). Both cycle-exact, checked every cycle in simulation. r150f closed core -> clk2x at
  +0.13 ns real (r150d +0.014).
- **The userclk1 reroute** (a98c51b2): its re-entrant route, run for a hold failure after
  rerouting the PCIe block's 500 MHz nets, moves other nets, and was kept while the other clocks
  stayed above 0: in r150f it took userclk1 +0.040 -> +0.099 and the core +0.130 -> +0.033 (the
  sequencer's `ir` into its IMEM). A reroute is now kept only if the design's worst slack rises.
  r150fp is r150f's routed design before it (its `pre_userclk1.dcp`, written to a bitstream as
  the fixed rule would have).
- **The memory PHY, statically**: every build's I/O paths are within E3's (ISERDES out, OSERDES
  in, the output enables, which are untimed by construction: 0.4-5.5 ns of route in every build,
  E3 included), and the ECC decoders' read data >= +0.69 ns in r150f against E3's +0.687.

**The card.** r150fp's window (the r2.2 qual body with the bitstream loaded) passed the selftest
and its scrub, then the first inference run (`perf.py qwen3 int8`) ended on an uncorrectable ECC
error; over the fast qual channel 1 counted 1,747 corrected and 293 uncorrectable words, channel
0 one, and token-exact runs then diverged. The window was stopped and the bitstreams compared on
one sequence (the card host's `~/otpu-build/bisect150/pwr.sh`, `xadc.py` beside it: scrub; idle 10
s; the DRAM turnaround stress 30 s; `perf.py` qwen3 and lfm2 int8), with the XADC's minimum and
maximum of each supply over each step:

| bitstream | core | ECC after the stress | after qwen3 / lfm2 (ch0 ch1, corrected/uncorrectable) | VCCINT min |
|---|---|---|---|---|
| E3 (production, 039b34d5) | 133.33 | 0/0 0/0 | 0/0 0/0, 0/0 0/0 | 0.978 V |
| b_t9_def (9b6279fc) | 133.33 | 0/0 0/0 | 0/0 0/0, 0/0 0/0 | 0.980 V |
| r150b (198e535c) | 150 | 0/0 0/0 | 17/0 0/0, 28/0 0/0 (a first load: clean) | 0.978 V |
| r150d (671b4265) | 150 | - | 1/0 0/0, 1/0 0/0 | - |
| r150fp (1da7d3b9) | 150 | 0/0 0/0 | 25/1 0/0, 431/1 133/10 | 0.980 V |
| r150h (f5eb87c5), first load | 150 | 0/0 0/0 | 0/0 0/0, 0/0 0/0 (four sequences) | 0.978 V |
| r150h, second load | 150 | 19/0 0/0 | 489/5 0/0, 15563/293 0/0 | 0.978 V |

- The turnaround stress (the MXU streaming weights while the DMA stores and reloads 64 KiB
  tiles, 150-170 GB in 30 s) is clean on every bitstream; inference, both channels near their
  full 16 GB/s with every unit busy, is not, at 150 MHz only. The supplies sag the same (VCCINT
  0.978-0.981 V at the XADC's ~4 us sampling) on all of them, and the die runs at 66-73 C on
  all.
- It is not a calibration: r150fp recalibrated warm (`memcal selfcal` after a run) still counted
  110/9 on channel 1, and that calibration found channel 0's common write window 485 ps
  (cold: 703 ps; E3: 753 ps). The read windows stay 9-11 taps on every bitstream.
- It is not the RTL: b_t9_def is r150b's RTL less the 150 MHz commits, at 133.33 MHz; and
  `otpu_mem_ch` hands LiteDRAM whole beats only (`c_wdata_we` all ones, partial writes merged
  before), so logic upstream cannot produce a word whose check bits disagree. The errors are
  between LiteDRAM's encoder and its decoder: the PHY and the DRAM. With the core at 133.33 MHz
  its switching keeps one phase to the memory clocks; at 150 MHz (9/8 of them) it sweeps across
  their edges.
- It is not where the TMEM sits. The failing channel had been the one whose PHY rows held the
  pumped TMEM's 128 RAMs (r150b rows 0-3, channel 0; r150f rows 4-7, channel 1; the PHYs in rows
  0-2 and 5-7 of the left I/O column), so r150h put the TMEM in a hard pblock between them
  (X0Y3:X1Y4, f5eb87c5). It fails timing there (core -0.286, clk2x -0.086; r150i -0.167 /
  -0.062), and on the card (2026-10-18) its first load read clean over four sequences while a
  second load of the same bitstream read 15,563 corrected and 293 uncorrectable words on
  channel 0, the turnaround stress included (19 corrected, the first stress failure).
- Each load calibrates anew, and channel 0's read framing differs per load at 150 MHz. E3's
  channel 0 reads every bit at its lane's bitslip (5, lane 1 at 4); every 150 MHz load read back
  bits a CLK off their lane (r150h's first: lanes 5, 6 and 8, five bits; its second: lanes 4, 6
  and 8, seven bits, and lane 2 at bitslip 6 tap 27 against 5 / 14; r150fp: lanes 4 and 8).
  Channel 1's varies on every bitstream (E3: lane 5's bits 0, 2 and 5 at -2; r150h's first
  load none, its second lanes 3 and 8), and read clean on E3 and both r150h loads. The framing
  alone does not decide it (r150h's first load had misframed bits and read clean), and
  the write side did not move (channel 0's write window 703 ps on every 150 MHz load, E3 753).
- **It is the read capture under load** (docs/litedram.md section 10, "Read taps under the core's
  load"). LiteDRAM's BIST, which has no ECC and checks its own data per lane, read wrong beats on
  channel 1's lanes 0 and 1 beside `perf.py lfm2` while the channel's ECC counted. It read none
  with the core idle or running DRAM-free TMEM / VPU work. A scan of every lane's read taps under
  an MXU + DMA load shows the windows closing from 7-9 taps to 4-6 at 150 MHz, mostly from the
  right, and to 6-8 around the calibrated tap at 133.33 MHz. The idle centre is then a tap or two
  right of the loaded one. `opentpu.host.loadcal` centres every lane under that load on a
  configuration's first scrub when the core clock is not the controllers'. On three fresh r150fp
  loads that read ECC errors under inference at the calibrated taps, the BIST beside inference
  and the ECC over the stress and both models then read 0.
- **Under prefill it is not enough** (docs/litedram.md, "Not under prefill"). r150fp's window
  with loadcal (r150fpl) was clean until the heavy xl quals: phi4-mini's 1,400-token prompt read
  channel 1 1 corrected / 1 uncorrectable, gemma4-e4b 5 corrected more; no soak, not promoted.
  The windows under one core load at a time put it on the MXU's switching (the TMEM, the VPU and
  the DMA barely move them), worst under prefill's 16 rows beside the DMA: the eighteen windows
  sum to 76 taps at 150 MHz against E3's 105. loadcal now centres under that load (each load's
  windows lie inside the lighter ones'), and a core clock on the DDR3 bit clock's grid (h152:
  r150fp's routed design, its MMCM at 1066.67 / 7 = 152.38 MHz, core -0.013 ns) gives back part
  of the one-row MXU's loss but little of prefill's (83 taps). Centred under prefill, h152's heavy
  xl quals read 1, 1 and 0 corrected words on channel 1: thirty times fewer, not none. With the
  array on X1-X8, a faster core costs about 1.4 taps (110 ps) of read window a lane under prefill,
  which no choice of taps gives back.
- **The array's place decides it** (2026-10-18; docs/litedram.md "The array's place"). On X1-X8
  the array's column 0 is one DSP column from the left I/O column. Five of its eight columns sit
  in clock region X0 (DSP X0-X5), the PHYs' half of the die.
  - **Moving it.** `OTPU_MXU_DSP_COLS`, `OTPU_MXU_BRAM_COLS` and `OTPU_MXU_WRAM_COL` move the
    array. `otpu_mxu_place.tcl` now skips the block RAM sites that are missing beside the PCIe
    block (RAMB36_X11Y40-44) and those other cells hold (the XDMA's RAMB36_X10Y40/41/43/44).
  - **What prefill costs.** The read window that prefill beside the DMA takes from idle (sum over
    the 18 lanes): X1-X8 62 taps, X2-X9 (r150n) 50, X4-X11 (r150m) 35; E3 at 133.33 MHz 39.
  - **The heavy xl quals.** Centred under prefill, r150m and r150n both read ECC 0 / 0.
  - **X4-X11 fails timing.** r150m does not close 150 MHz. About 200 DSPs outside the array (the
    VPU's 86, the quantizer's 64, the MXU's own ~50) do not fit the rows the array leaves free in
    X4-X11. They land on X0-X3 beside the PHYs, away from their logic: the VPU's and the
    quantizer's own paths fail by 1-2 ns. A hard pblock that keeps the XDMA beside the PCIe
    block (`OTPU_XDMA_PBLOCK`; the placer had moved it to X0Y5-Y7) makes it worse.
- **Placement screens** (the build host's `~/otpu-build/tff/screen/`). Each screen starts from r150m's
  opt checkpoint, re-places the array and runs the build's place_design and phys_opt_design
  (about 20 minutes, no route). The estimate after phys_opt (with the 0.15 ns margin) predicts the
  route:
  - r150fp's -0.221 ns / TNS -35 met timing after the route;
  - r150m's -1.182 / -2,779 failed;
  - X2-X9's -0.105 / -29 became r150n's +0.066 real;
  - X3-X10's -0.053 / -2.8 with ExtraPostPlacementOpt became r150o's +0.079 real.

  | the array | the placer | core | clk2x | userclk2 | TNS |
  |---|---|---|---|---|---|
  | X1-X8 (r150fp) | ExtraTimingOpt | -0.221 | -0.061 | +0.120 | -35 |
  | X4-X11 (r150m) | ExtraTimingOpt | -1.182 | -0.936 | -0.359 | -2,779 |
  | X4-X11 on rows 23-159, the XDMA in X1Y3:X1Y6 | ExtraTimingOpt | -2.673 | -1.741 | +0.107 | -5,964 |
  | X4-X11 on rows 0-136, the same | ExtraTimingOpt | -2.068 | -1.364 | +0.095 | -6,612 |
  | X4-X11 | ExtraPostPlacementOpt | -1.907 | -1.475 | +0.042 | -3,189 |
  | X4-X11 on rows 23-159 | ExtraPostPlacementOpt | -1.553 | -0.619 | +0.078 | -2,168 |
  | X3-X10 | ExtraTimingOpt | -0.375 | -0.197 | +0.008 | -354 |
  | X3-X10 | EarlyBlockPlacement | -0.229 | -0.075 | -0.163 | -63 |
  | X3-X10 | ExtraNetDelay_high | -0.305 | -0.238 | +0.004 | -261 |
  | X3-X10 | AltSpreadLogic_high | -0.278 | -0.200 | -0.141 | -156 |
  | **X3-X10** | **ExtraPostPlacementOpt** | **-0.053** | **0.000** | +0.008 | **-2.8** |
  | X2-X9 (r150n) | ExtraTimingOpt | -0.105 | -0.056 | -0.148 | -29 |
  | X2-X9, the XDMA in X1Y3:X1Y6 | ExtraTimingOpt | -0.738 | -0.390 | +0.002 | -587 |

  `OTPU_PLACE_DIRECTIVE` (build.tcl) sets the placer's directive alone and keeps the rest of the
  production strategy.
- **r150o on the card** (2026-10-19; docs/litedram.md "Each lane's eye under load"). Its first
  deploy window read 56 corrected words on channel 1 (every token right, no soak): that load's
  calibration had put channel 1 lane 6 at tap 26, with 2 taps under prefill. loadcal now chooses
  each lane's eye under load as well as its tap (1c4b3bf8: the lane's read bitslip and the one
  either side). The second window, on production's host ec87de44 with loadcal 6b8d9d4c and
  1c4b3bf8, read ECC 0 / 0 on both channels in every qual (156 reads: the fast qual, E2B, the
  chats, MoE, the stresses, the heavy xl quals) and in the 5 h soak (20 quals, 1,161 reads).
  Promoted the same day (the card host's `deploy_r150o_1da7d3b9`; E3 production-prev): the
  promotion's own load windows under prefill beside the DMA summed 94 taps, every lane 4 or more
  (E3 105, its narrowest lane 4).
- What 150 MHz buys, on the same runs: prefill 4% (Qwen3-0.6B 173.1 -> 180.1 tok/s, LFM2.5
  569.1 -> 595.1), decode under 1% (24.70 -> 24.76 tok/s; 66.53 -> 67.08): decode waits on the
  DRAM, which is the same 17 GB/s at either clock.

So the card's clock is 150 MHz since 2026-10-19 (r150o). The same RTL also closes 133.33 MHz with
Vivado's default implementation (b_t9_def, section 7).

## 9. Next

- The FPGA: r150o's windows under prefill sum 94 taps against E3's 105. Its weakest lane is one the
  eye search cannot move: a lane calibrated at bitslip 0 whose lower eye is a bitslip below (the
  second window's channel 0 lane 4: bitslip 0, tap 20, 4 taps under load). The calibration itself
  (the core's CPU's and ddrcal's) could take the eye nearest the delay line's start rather than the
  first of equals. The DRAM's Rtt_Nom off (MR1 0x2, Rtt_WR kept: the DRAM terminating during writes
  only) is untried. X4-X11 (r150m, 98 taps) does not close 150 MHz with any placer directive
  screened; the read capture on regional clocks (BUFIO / BUFR, one MMCM per bank region) or on DQS
  instead of the BUFG-clocked sys4x would take a PHY change. Per-bit read delays (gating the
  IDELAYs' CE / LD with dly_sel_bits) would add at most half a tap: the bits of a lane open within a
  tap of each other under load.
- The ASIC: run the card's flow (both hooks are its default now) through detailed routing, with
  the clock at 2.0 ns rather than a margin (the global route's 2.0 ns is on the guides'
  parasitics); put the copies behind a target parameter (FPGA on, ASIC off); at 2.0 ns the VPU's
  floating-point stages are next (a deeper `LM` / `LA` moves no logic without retiming,
  docs/asic.md).
- The protocol changes of section 5: the MXU's grant is off its epilogue (section 7). At
  b_t9_150's -0.06 ns what is left is mostly arithmetic (the MXU's fmadd adder, the DSP outputs)
  and placement (the VPU's enable copies). TMEM's read selection into the VPU's hold is at
  -0.041 ns from the MXU's `room_q`; if it leads again, `t_ren` can be a register of its own
  (its next value from the grant and `t_lost`, eight flip-flops), and `rpv` a copy per bank
  (section 7: about 30 lines in `otpu_tmem`'s IMPL 3 block). Then the sequencer's dispatch-order
  queue.
