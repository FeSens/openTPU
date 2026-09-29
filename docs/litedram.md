# LiteDRAM instead of the MIG: an assessment (draft)

Draft, 2026-09-29. This is an investigation; the design is unchanged. Labels: **measured** means
a Vivado run or a simulation that was actually run (tool and version given). **estimate** means
arithmetic. Nothing here has run on the card.

The question: could [LiteDRAM](https://github.com/enjoy-digital/litedram) replace the two Xilinx
MIG controllers, the SmartConnect in front of them and the AXI side of `otpu_axi_dram`? Its
native port could be driven directly, skipping AXI. The question covers area, bandwidth and
DDR3 speed.

## Summary

- **Area.** Today's memory path costs 60.3K LUTs and 50.0K FFs, about 28% of the design's LUTs
  and 30% of its FFs. A LiteDRAM channel with a realistic set of ports measures 8.7K LUTs and
  7.8K FFs after place and route, test load included. The whole replacement would save roughly
  30K LUTs and 30K FFs (**estimate**).
- **Bandwidth.** LiteDRAM's controller reaches 90.9% of peak on sequential reads in simulation
  at DDR3-1066, and 95.0% when one stream is split over two ports. The fitted MIG model gives
  88.9%. But the core's 128-byte port at 120.755 MHz caps the whole path at 90.6% of the
  DDR3-1066 peak, and decode already reads at 83-85%. So a better controller buys at most a few
  percent of decode speed.
- **DDR3 speed.** The DDR3 sits on HR banks, and DDR3-1066 is the limit AMD characterizes for HR
  banks at this speed grade. That is a limit of the silicon, not a choice MIG makes, and
  LiteDRAM does not lift it.
- **Risk.** LiteDRAM's 7-series PHY cannot write-level on HR banks. The board has 9 x8 chips on
  a fly-by bus per channel. The upstream LiteX target for this exact board uses only 4 of the 9
  byte lanes of one channel, marked "FIXME: Get all modules working".
- **Recommendation: staged.**
  1. Keep the MIG (its PHY, calibration and ECC) but switch it to its native interface. Replace
     the SmartConnect with our own arbiter and clock crossing. This gets about half of the LUT
     savings and most of the FF savings with no bring-up risk.
  2. Then, as an experiment, build a one-channel LiteDRAM test image to answer the
     write-leveling question on the card.

## 1. What the memory path costs today

The routed champion build (the production RTL at 125.49 MHz, `tv-full-5a3238f96026`,
`util_hier.rpt`), **measured**:

| Block | Total LUTs | Logic | LUTRAM | SRL | FFs |
|---|---|---|---|---|---|
| MIG channel 0 (`mig_0`) | 14,141 | 11,630 | 1,974 | 537 | 10,267 |
| MIG channel 1 (`mig_1`) | 14,151 | 11,642 | 1,972 | 537 | 10,255 |
| SmartConnect `sc_mem` | 15,310 | 10,947 | 4,232 | 131 | 24,248 |
| `otpu_axi_dram` (`u_mem`) | 16,664 | 9,224 | 7,440 | 0 | 5,254 |
| **total** | **60,266** | 43,443 | 15,618 | 1,205 | **50,024** |

That is 27.9% of the design's 216,194 LUTs and 30.2% of its 165,499 FFs. In the tournament's
area measure (LUT + LUTRAM + 0.5 FF + ...) it is about 85K of 359K. No block RAM and no DSP.

What `sc_mem` does (`boards/ypcb-00338/vivado/bd.tcl`):

- It has 3 slaves:
  - XDMA's M_AXI: 128 bits at 125 MHz. This carries the host's loads and logits reads.
  - The accelerator's `m0` and `m1`: 512 bits at the core clock.
- It has 2 masters, one per MIG, at each MIG's `ui_clk` of 133 MHz.
- It does the 128-to-512 width conversion, the core-to-`ui_clk` and XDMA-to-`ui_clk` clock
  crossings, and the routing.

Whatever replaces it must keep the host's DMA path to both channels. The small `sc_ctl` (259
LUTs) also connects BAR0 to the MIGs' ECC control ports.

Inside one MIG, measured with an out-of-context synthesis of today's configuration (DDR3-1066,
72-bit, ECC, AXI 512-bit), `mig_axi_util_hier.rpt` (Vivado 2026.1):

| Part | Total LUTs | FFs |
|---|---|---|
| Calibration logic (`u_ddr_calib_top`: read leveling, write leveling, OCLK delay, PRBS, temperature) | 5,866 | 3,655 |
| PHY datapath (`u_ddr_mc_phy_wrapper`) | 2,514 | 1,163 |
| Controller (`mc0`, of which the ECC decode and fix take 1,481 LUTs) | 2,668 | 1,308 |
| User interface (`u_ui_top`) | 2,086 | 1,817 |
| AXI front end (`u_axi_mc`) | 1,525 | 1,423 |
| **MIG with the AXI interface** | **15,002** | **10,365** |
| **MIG with the native interface (same settings)** | **12,668** | **8,591** |

Most of the MIG is calibration done in hardware state machines. LiteDRAM does calibration in
software instead.

What the MIG gives us today:

- calibration, including write leveling with PHASER_OUT fine delays, which exist in HR banks;
- ECC with read-modify-write, which is how partial writes work without DM pins;
- temperature tracking;
- 83-85% of peak on decode reads (docs/board.md).

What it costs in speed, from the fitted model (docs/board.md section 4): each AXI read
transaction costs 1.22 controller cycles plus 0.06 per beat, each write transaction 2 cycles,
and each read/write turnaround 4 cycles.

## 2. LiteDRAM on this board

**Board support.** litex-boards already has this board:
`litex_boards/{platforms,targets}/ypcb_00338_1p1.py`, written by Florent Kermarrec in 2025
from TiferKing's pinout. Its DDR3 pins match ours pin for pin. The target:

- uses `A7DDRPHY` with the comment "DDR3 on HR Bank so no ODELAY, use A7DDRPHY as workaround";
- wraps the pads in `PHYPadsReducer(..., [0, 1, 2, 3])` with "FIXME: Get all modules working",
  so it uses 4 of the 9 byte lanes of one channel, 32 bits;
- runs at 125 MHz sys, which is DDR3-1000;
- sets `INTERNAL_VREF 0.750` on banks 11-18.

So full-width LiteDRAM operation on this board has not been shown by anyone.

**The PHY, and why write leveling is the main risk.**

- LiteDRAM's `K7DDRPHY` needs ODELAYE2, which exists only in HP banks. The DDR3 banks 11-18 are
  HR, so the only choice is `A7DDRPHY` (`S7DDRPHY(with_odelay=False)`).
- For DDR3 without ODELAY, LiteDRAM sets `write_leveling = False` ("DDR3 Write leveling is not
  possible on Artix7 due to the lack of ODELAYE2"). It only calibrates write latency in coarse
  bitslip steps.
- Each channel has 9 x8 chips on a fly-by address and clock bus. Each chip's DQS must arrive
  within ±0.25 tCK of its CK, which is ±0.47 ns at DDR3-1066.
- The MIG meets that with per-byte-group PHASER_OUT fine delays (`mb_wrlvl_inst` above), which
  exist in HR banks too. LiteDRAM's SERDES-based PHY does not use the phasers.
- The far chips of the fly-by are the likely reason the upstream target stops at 4 lanes. This is
  a hypothesis: only the card can confirm it.

**The DDR3 speed ceiling is the HR banks, not the controller.**

- For HR banks at speed grade -2, AMD characterizes DDR3 at 1.5 V up to 1066 Mb/s.
- MIG encodes this as `tmin_hr` = 1875 ps in `mig_7series_v4_2/.../time_periods.xml` (docs/board.md,
  "Faster DDR3"). The HP-bank figures (1600 and up) do not apply to this board.
- LiteDRAM does not change the I/O. Above 1066 it has less margin than the MIG, because it has
  no fine write deskew.
- On the card, the out-of-spec MIG at 1300 calibrated and passed diag and model checks, but was
  never qualified. 1333 failed (docs/board.md).
- Vivado itself would build LiteDRAM's PHY up to about a 700 MHz DDR clock. Its limit is the
  BUFG minimum period of 1.408 ns (**measured**, pulse-width report). But nothing checks the
  I/O timing of a calibrated interface, so a clean build says nothing about the card.

**Faster DDR3 would not help much anyway.** The MXU takes one 128-byte chunk per core cycle.
At 120.755 MHz that is 15.46 GB/s, or 90.6% of DDR3-1066's 17.07 GB/s peak. Decode already
reads at 83-85% of peak. Anything past the port needs the wider path in
[wide_dram.md](wide_dram.md).

**Clocking.** There are two options:

- **Own clock plus a crossing.** LiteDRAM's sys clock is its DDR clock / 4, so 133.33 MHz at
  DDR3-1066. It then needs a clock crossing to the core, like the MIG's `ui_clk` does today.
- **Run it from the core clock.** 120.755 MHz gives DDR3-966, and no crossing. But that peak is
  15.46 GB/s, exactly the port's cap, and at about 90% controller efficiency it delivers about
  13.9 GB/s. That is below the 14.2-14.5 GB/s decode reads today, a loss of about 3-4%
  (**estimate**).

So keep a separate clock and a crossing.

**Calibration without a CPU.** LiteDRAM's initialization and leveling run as C code in the LiteX
BIOS (`liblitedram/sdram.c`, 1.9K lines covering all PHYs) on a soft CPU. There are two options:

- **A small VexRiscv plus its ROM:** roughly 1-2K LUTs and some block RAM (**estimate**). One CPU
  could calibrate both channels.
- **The host does it:** the PHY and controller CSRs sit in a window on BAR0, and the host runs
  the init and leveling sequence in Python. The LiteDRAM generator already offers this: with
  `cpu: None` it exposes a Wishbone control bus. Only the S7 read leveling and write-latency
  sequence would need porting.

The host already exists and owns BAR0, so the host-driven path fits this design.

Calibration is well proven on Artix-7 boards and on HP-bank Kintex-7 boards. On this board it has
been shown on 4 lanes only. There is no equivalent of the MIG's temperature tracking, so the
qualification's warm soak is where drift would show.

## 3. The interface: LiteDRAM's native port

- **Classic Wishbone** allows one transaction in flight, so it would reach a small fraction of
  the channel. Not an option for the weight stream.
- **The AXI frontend** (`LiteDRAMAXI2Native`) exists and has a read-modify-write option for
  partial strobes. It is only needed for XDMA's path.

**The native port**, one per master and channel, has three streams: `cmd` (`we`, beat address),
`wdata` (`data`, `we` byte mask) and `rdata` (`data`). It behaves as follows (LiteDRAM 2026.8,
`core/crossbar.py`, `core/bankmachine.py`):

- **One command per beat.** A beat is one BL8 burst, 64 bytes on a 64-bit channel (576 bits
  with the ECC lane). Commands are pipelined, and there is no per-transaction cost.
- **Several commands in flight.** Each bank machine queues `cmd_buffer_depth` commands (8 by
  default).
- **One bank at a time per master.** The crossbar holds a master to one bank until that bank's
  queue has issued (`lock`). So each port's reads come back in order, but every change of bank
  exposes an activate plus tRCD. With the row-bank-column mapping that means every 8 KB. Masters
  working in different banks run in parallel.
- **No backpressure on read data.** The master must always accept it. `otpu_axi_dram` already
  reserves response room before it issues a read.
- **The byte mask cannot reach the DRAM**, because the board has no DM pins. LiteDRAM's ECC
  frontend rejects partial writes (`we_error`). So every partial write needs a read-modify-write
  in our adapter. `otpu_axi_dram` already does one for the quantizer's gathered stores (port SW).
  The A port's byte writes and B's masked writes would need the same.

**How our ports would map**, one native port per slice port per channel:

- **Port B:** a 32-beat weight read becomes 32 consecutive read commands. A DMA ST's or DSTEP's
  write run becomes a run of write commands with their data.
- **Port A:** the MXU's scale reads, with the prefetch runs as today.
- **Port SW:** the gathered quantizer stores.
- **XDMA:** through the AXI frontend with read-modify-write, and a 128-to-512 converter.

What goes away: the SmartConnect, the MIG AXI front end, and `otpu_axi_dram`'s AXI burst forming
(gather timers, burst lengths, AXI IDs and write responses). What stays: the channel interleave
(CHASH), the store gathering and read-modify-write, the A prefetch runs, the reserved response
room and `wr_idle`.

**The adapter as built** (2026-09-29): `rtl/mem/otpu_native_dram.sv`, one native master per
channel for `otpu_mem_ch`, selected in `otpu_board` by `MEM_NATIVE` (default 0: production keeps
`otpu_axi_dram`). Its header comment has the details; in short:

- **One command per beat, per core cycle and channel**, from the A queue (a run of APF read
  commands, or a byte-enabled write), the SW queue's fill reads and writes, and the B queue.
  Write data goes with its command; partial beats of ports A and B keep their byte mask (the
  channel module does the read-modify-write). The SW port keeps its gather and read-fill.
- **Read data** returns in command order, so a tag FIFO per channel routes each beat to B, A or
  an SW slot; room is reserved before a read goes out, and simulation checks for overflow.
- **Ordering by the command stream instead of write responses.** An SW fill read waits until
  the older SW writes of its beat have entered the stream, not until they are done. A write
  drops the A run or reused beat it touches when it goes out, not when it is answered.
  `wr_idle` waits for `n_wdone`.
- **Simulation:** `sim/verilator/otpu_native_mem.sv` (random command and write-data
  backpressure, in-order read data with jitter, late `n_wdone`, the DDR3 bank model of
  `otpu_axi_mem`); `OTPU_NATIVE=1` runs the AXI-path tests and the board model on it.
- **Area, yosys** (`synth_xilinx`, **measured**): 10,620 LUTs, 3,364 FFs, 1,492 LUT RAM cells,
  against `otpu_axi_dram`'s 13,464, 4,914 and 1,880. Logic depth 4.57 ns against 6.01 ns.
- **Cycles, simulation (measured,** `tools/litedram/path_bench.py`**):** the DDR3-1066 bank
  model at 120.755 MHz and 300 ns latency, the AXI path with its fitted per-transaction costs
  (as calibrated on the card) / without them / native:

  | Workload | AXI | AXI, no costs | native |
  |---|---|---|---|
  | Qwen3-0.6B decode, 2 layers, pos 300 | 1,736,120 | 1,534,272 | 1,548,978 (-10.8%) |
  | fp4 weight stream, 2 MB (`rw_bench mm`) | 20,068 | 17,760 | 17,900 (-10.8%) |
  | the same beside 64 KiB loads (`mm+ld`) | 140,487 | 140,566 | 142,222 (+1.2%) |
  | the same beside 64 KiB stores (`mm+st`) | 141,932 | 140,482 | 137,441 (-3.2%) |
  | transposed V append, 4 tokens | 15,375 | 10,593 | 11,706 (-23.9%) |
  | K append, 16 tokens | 13,624 | 13,063 | 12,047 (-11.6%) |

  The percentages are against the AXI path with its costs, the one the card has. `mm+ld` is the
  native model's lookahead: with 256 cycles of it instead of 16, 140,897 (+0.3%).
- **Limit:** one command per core cycle and channel is 90.6% of DDR3-1066's peak at 120.755 MHz,
  the same as port B's cap, but B, A and SW traffic now share it (AXI's read and write channels
  did not). Decode saturates it (0.98 read beats per cycle and channel), which is most of the 1%
  to the cost-free AXI model; a transposed V append needs two commands per beat (the fill read
  and the write). A second command per cycle, or a faster command clock, would lift it.

## 4. Measured

### Area and timing

All runs are Vivado 2026.1 on omarchy, place and route on xc7k480t-ffg1156-2, using channel 0's
real pins from the litex-boards platform. The core is LiteX and LiteDRAM 2026.8 (git), and the
scripts are in `tools/litedram/ooc_ypcb.py`.

Each design has:

- a sys clock at the DDR clock / 4;
- no CPU: the CSRs sit behind a JTAG bridge (JTAGBone), standing in for BAR0;
- a traffic generator on every port, so Vivado keeps the full datapath. Write data comes from an
  LFSR; read data is XOR-folded to a pin.

The generators are included in the counts below: about 0.7K LUTs and 0.6K FFs per native port
(**estimate** from their structure).

| Design (one channel) | sys clock | Total LUTs (LUTRAM) | FFs | BRAM | Worst slack (all clocks) |
|---|---|---|---|---|---|
| Controller + A7DDRPHY, 72-bit, one native port | 133.33 MHz (DDR3-1066) | 5,627 (142) | 5,746 | 0 | +1.428 ns setup, +0.075 ns hold |
| same | 150 MHz (DDR3-1200) | 5,631 | 5,752 | 0 | +1.169 / +0.057 ns |
| 64-bit (8 lanes, no ECC), 2 native ports through clock crossings into a 125 MHz "core" clock, plus XDMA's 128-bit AXI through the AXI frontend (read-modify-write on) in a 100 MHz clock | 133.33 MHz | 8,676 (626) | 7,758 | 40 RAMB36 + 6 RAMB18 | +0.715 / +0.034 ns (sys +1.043) |
| same, 72-bit with LiteDRAM's ECC frontend on all 3 ports | 133.33 MHz | 10,479 (628) | 10,997 | 40 RAMB36 + 6 RAMB18 | +0.383 / +0.029 ns (sys +0.383) |

Notes:

- **The block RAMs are the clock-crossing FIFOs.** They are 576 bits wide and 16 deep, and Vivado
  mapped them to block RAM. As LUT RAM they would cost roughly 400 LUTs each (**estimate**).
- **Clock margin.** The controller's sys-clock logic has over 1 ns of slack at 133.33 MHz, so it
  would run near 165 MHz. ECC takes most of that margin.
- **What a channel would really cost.** Without the test load and the JTAG bridge, a LiteDRAM
  channel with the ports we need is about 6.5-7K LUTs and 6K FFs (**estimate**). That compares
  with 14.1K LUTs and 10.3K FFs for one MIG today, before counting the SmartConnect.

**The whole path, estimated.**

- Two LiteDRAM channels: about 17K LUTs, 16K FFs and 86 block RAMs. The block RAMs become
  +6-8K LUTs if the FIFOs move to LUT RAM.
- Plus what stays of `otpu_axi_dram`: about 12-14K LUTs, since only its AXI burst forming goes.
- Total: about 30K LUTs and 20K FFs, against 60.3K and 50.0K today.
- That saves about 30K LUTs and 30K FFs, 8-10% of the design's area measure (**estimate**).

**Stage 1 alternative**, estimated:

- Keep the MIG with its native interface: -2.3K LUTs and -1.8K FFs per channel (**measured**, out of
  context).
- Replace the SmartConnect's 15.3K LUTs and 24.2K FFs with our own per-channel arbiter, the
  core-to-`ui_clk` crossing and an XDMA-to-native bridge: roughly 3-5K LUTs and 3-5K FFs
  (**estimate**).
- That saves about 15K LUTs and 23K FFs with no change to calibration or the PHY.

The tournament's SmartConnect-pruning build (`omarchy:~/otpu-build/scprune-12c0d212`: bd.tcl
only, with `m0` going only to `mig_0`, `m1` only to `mig_1`, and XDMA to both) is the
lowest-effort baseline. It was still building when this was written.

### Efficiency

**Setup, measured.** The LiteDRAM 2026.8 controller, bank machines, refresher and crossbar
(their own RTL) were simulated in migen. The PHY was a DFI stub with the S7 PHY's read latency.
One channel, DDR3-1066, sys 133.33 MHz, the MT41K256M8 geometry (8 KB rows), 16,384 beats per
run. The script is `tools/litedram/bench_native.py`. Efficiency = beats / sys cycles, and one beat per cycle is
the peak.

**MIG column, estimate.** This is arithmetic from the fitted model's parameters:

- 1.22 + 0.06 cycles per beat per read transaction;
- 2 cycles per write burst;
- 4 cycles per turnaround;
- a tRFC of 22 cycles every 1040, plus an activate.

The same model reproduces the card within about 1-2% (docs/board.md).

| Pattern | LiteDRAM | MIG fitted model |
|---|---|---|
| sequential reads, one port | 90.2% (90.9% with tRFC 160 ns) | 32-beat bursts: 88.9% |
| sequential writes, one port | 89.2% | 8-beat bursts: 78.1% |
| 8 read runs of 32 beats, then a 32-beat write run elsewhere, one port | 85.7% | 85.4% |
| the same reads on one port, writes on another | 84.3% | - |
| random 32-beat reads (a row miss each) | 74.0% | - |
| one read stream split over 2 ports by 8 KB row | 94.0% (95.0% with tRFC 160 ns) | - |

- **tRFC.** LiteDRAM's `MT41K256M8` sets tRFC to 128 clocks, 240 ns at 1066. The board's 2 Gb
  parts need 160 ns, the value the MIG project uses. A custom module class gains 0.7 points.
- **Command buffer.** Depth 16 changed nothing for these patterns: read 90.2%, mix 85.7%, two
  ports 84.3%.
- **The bank-switch bubble.** The one-bank-per-master lock costs about 4 points on a single
  sequential stream. Splitting the stream over two ports by row hides it.
- **Card anchor, measured.** The card's MIG does 86.9% of peak on `tools/rw_bench.py mm` at 32-beat
  bursts, which is at the port's cap.
- **Reading.** LiteDRAM's controller is about 2 points better than the MIG through AXI on reads and
  about 11 points better on writes, and even on the 8:1 mix. Both sit at or above the core
  port's 90.6% cap on sequential reads.
- **How much decode could gain.** Today's 83-85% has at most about 7-9% of headroom before the
  port (**estimate**). The MIG's native interface also removes the per-transaction costs that
  make up most of the MIG column's losses, so stage 1 should take most of that gain.

## 5. Risks and cost of a switch

1. **Write leveling and full width** are not shown on this board. See section 2. Only the card can
   answer it, and it may need a lower data rate or not work at 72 bits at all.
2. **Calibration software.** The init and leveling sequence has to be written for the host, or a
   soft CPU added. There is no temperature tracking: the warm soak must pass on every image.
3. **No DM pins.** Every partial write needs our own read-modify-write: for the A port's byte
   writes, B's masked writes and XDMA's partial beats. LiteDRAM's AXI frontend has one for XDMA.
4. **ECC.** It costs about 1.8K LUTs and 3.2K FFs per channel (**measured**, the difference of
   the two realistic runs, which also includes the ninth lane's PHY). It also cuts the sys-clock slack from +1.04 to +0.38 ns. Without it,
   the ninth lane goes unused and nothing corrects a flipped bit. The MIG's ECC counters were
   never read on the card, so it is not known whether ECC ever fires.
5. **The build flow.** The core is Migen-generated Verilog. It can be committed as a generated
   file, like the MIG's output, but it is flat and not readable. The tournament could tune its
   generation parameters and the ports around it, but not usefully edit the controller.
6. **Card time.** Bring-up means JTAG reloads (the link retrain in `otpu-rescan` has never been
   exercised) and a 38-minute qualification per image. There is no MIG fallback inside a
   LiteDRAM image.

## 6. Recommendation

1. **Stage 1: keep the MIG, drop the AXI (go).** Switch the MIG to its native interface. Replace
   the SmartConnect with our own per-channel arbiter, the core-to-`ui_clk` crossing and an
   XDMA-to-native bridge.
   - Expected savings, **estimate**: about 15K LUTs and 23K FFs.
   - It also removes the per-transaction costs, from the fitted model: 1.22 + 0.06/beat per
     read, 2 per write burst.
   - Calibration, write leveling, ECC and temperature tracking stay as they are, so the risk is
     low.
   - The first step is to compare against the tournament's SmartConnect-pruning build.
2. **Stage 2: a LiteDRAM test image on one channel (experiment).** Full 72-bit width on one
   channel, host-driven calibration through a CSR window on BAR0, and a memory test and
   bandwidth test from the host.
   - It answers the write-leveling question for this board: calibration results per lane,
     margins, and whether all 9 lanes write correctly when warm.
   - Pursue a full switch only if the answer is yes and the memory path is still the area
     problem after stage 1. It would save about another 15K LUTs, from dropping the MIG's
     hardware calibration.
3. **No faster DDR3.** Neither controller goes past the HR banks' 1066 in spec, and the core port
   caps the gain.

## 7. The one-channel test image (LiteDRAM first, 2026-09-29)

On 2026-09-29 the plan changed. LiteDRAM became the primary path and stage 1 became the
fallback: MIG native with our own arbiter, committed and unit-tested (`otpu_mig_ch`, `tb_mig`),
but not yet integrated. The key unknown for LiteDRAM is calibration and write timing on all 9
lanes without ODELAY. The test image measures exactly that. It is built by
`tools/litedram/ld_test.py` and driven by `tools/litedram/ld_host.py`.

**Design**
- **Memory:** DDR3 channel 0 at 72 bits (all 9 x8 chips). A7DDRPHY at DDR3-1066, with CL 7 /
  CWL 6 as in the MIG project. LiteDRAM's controller with MT41K256M8 and tRFC 160 ns. The pins
  are litex-boards', checked one by one against `constraints/ddr3_ch0.pins.xdc`.
- **PCIe:** XDMA as an RTL IP with the production settings, subsystem 10ee:4C44. BAR0 goes to a
  CPU-less LiteX SoC's CSR bus. XDMA's DMA master is answered by a stub.
- **Memory test:** a 576-bit BIST (LiteDRAM's BIST needs a power-of-two width). It covers
  sequential writes and read-checks over any range, with random data or address data. It keeps
  an error count per lane and a cycle count, from which bandwidth is computed.
- **Write DQS:** the write DQS clock comes from an MMCM output with fine phase shift, 1/56 of the
  1066.67 MHz VCO period, i.e. 16.7 ps per step. The host moves it (`phase_dqs_shift`). Without
  write leveling this one knob sets every lane's DQS-to-CK timing, so a scan of it measures the
  write margin.
- **Clocks:** the 50 MHz oscillator, as in production, drives an integer MMCM (sys, sys4x,
  sys4x_dqs) and a PLL for the 200 MHz IDELAYCTRL reference.

**Host calibration (`ld_host.py`)**
- Runs the JEDEC init through DFII, then write-latency calibration, then read leveling. The
  algorithm is the one in liblitedram's `sdram.c`, but every scan covers all 9 lanes at once
  (`dly_sel` = all lanes; each lane's bytes are checked separately).
- `all` first scans the DQS phase over one tCK. At each step it recalibrates the write latency and
  records, per lane, the widest read window. The lane's write margin is the run of phases where
  that window is at least 3 taps. It then centres DQS in the range common to all lanes,
  calibrates, and runs the BIST over the whole 2 GiB channel.
- `selftest` checks the algorithm against a simulated PHY.

### Results on the card (2026-09-29, opentpu, DDR3-1066, channel 0, 72 bits)

The logs are in `docs/data/litedram/card-*.log`.

**Build**
- Vivado 2026.1 met timing: WNS +0.857 ns, WHS +0.046 ns, with the GT lanes on X0Y23..16.
- Controller, PHY, BIST and CSR bus take 6.7K LUT and 6.1K FF. XDMA takes 15.2K LUT.

**Calibration** (host-driven over BAR0; about 0.1 s per full calibration)
- **Read leveling:** windows of 9 to 11 taps (703 to 859 ps of the 938 ps bit), bitslip 4 on
  every lane.
- **Read window under traffic:** 8 to 9 taps. This is a BIST read of 256 MiB at each tap
  (`rscan`).
- **Write latency:** follows the fly-by. Lanes 0-3 need bitslip 0 and lanes 4-8 need bitslip 6
  (one tCK earlier) at the chosen phase.

**Write margin** (DQS phase scan in 16.7 ps steps over one tCK = 112 steps; at each step: full
calibration, then a 64 MiB BIST write and read-back per lane)
- Each lane writes correctly over 39 to 48 steps, i.e. 650 to 800 ps.
- The lanes are skewed against each other by fly-by, so their windows only partly overlap. All
  nine pass together in two places:
  - window A: +7 to +20 steps from 90°, 14 steps, 234 ps;
  - window B: 10 steps, 167 ps.
- Between the two windows, lanes 2 and 3 cross their write-latency boundary: they fail at
  either latency. About 70 steps fail on every lane.
- **LiteX's default 90° phase fails lane 2 under traffic, although its DFI calibration check
  passes.** The single-burst check is not enough: the DQS phase must be chosen with a traffic
  test (`ld_host.py all` does this).

**Soak and bandwidth**
- At the centre of window A (+13), 266 passes in 300 s with 0 errors. Each pass writes and reads
  back 2 GiB of random data and 2 GiB of address data.
- At the window's edges (+7, +20), 40 passes each with 0 errors.
- BIST bandwidth (sequential, one port): write 7.69 GB/s (90.1% of 8.53 GB/s), read 7.76 GB/s
  (91.0%). This matches the controller simulation in section 4.

**PCIe**
- After each JTAG load, test image and production alike, `otpu-rescan`'s retrain did not bring
  the card back. It tried Retrain Link, Link Disable and secondary bus reset, 15 attempts in all.
  The root port stayed at LnkSta 0x1081 and nothing appeared on bus 01.
- A warm reboot brought each image up: the test image enumerated as 10ee:7028/4C44, and
  production passed its selftest afterwards.
- On opentpu, a reload therefore costs a reboot.

### Decision

**Go for LiteDRAM, with two conditions.**
1. **Two DQS clock groups.** The common write window (234 ps, about ±117 ps) is thin. The cause
   is that this PHY drives one DQS clock for all lanes: the HR banks have no ODELAY, and the PHY
   does not use the phasers that the MIG uses for write leveling. The fix: give lanes 0-3 and
   lanes 4-8 each their own MMCM output with fine phase shift (the MMCM has spare outputs). Each
   group should then get a window the size of a single lane's, 650 to 800 ps.
2. **Channel 1.** It is not tested yet, and its fly-by split may differ. It needs its own run of
   the test image.

In production, calibration must choose the DQS phase(s) with a traffic check.
