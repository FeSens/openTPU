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

### What the write scan measured (corrected reading)

The first reading above was wrong in one respect. The scan moves only DQS: DQ stays on the
unshifted sys4x. So each lane's result is the overlap of two separate limits (`card-all1.log`,
steps from 90°):

1. **The DQ-to-DQS eye, common to all lanes.**
   - Every lane fails from about +21..+27 to +93..+98, the ~70 steps quoted above. There DQS no
     longer sits inside its own DQ bits, or the DQS serializer's hand-over from its slow clock
     breaks; this scan cannot tell the two apart.
   - The eye's edges line up across lanes to within about 6 steps. It is about 44 steps
     (730 ps) wide.
2. **The fly-by limit (tDQSS), per lane.**
   - Each lane's write latency flips where its DQS is half a tCK off the CK at its chip, and the
     lane fails for 5 to 7 steps around that point.
   - The crossings are at: lane 0 +86..+92, lane 1 +93..+97, lane 2 +108..+1, lane 3 +1..+6,
     lane 8 +21..+24. Lanes 4-7 cross outside the eye.
   - So the chain order is 0, 1, 2, 3, 8, 4-7: the ECC byte sits mid-chain, as on a DIMM
     (**inferred** from the order of the crossings).
   - The crossings of lanes 2 and 3 fall inside the eye and cut it into window A (+7..+20) and
     window B (+98..+107).

So **two DQS-only clock groups would not help.** Lanes 0-3 would keep the crossings of lanes 2
and 3 inside the same eye, and stay capped at 14 steps. What would help is moving DQ and DQS
together per group, which is real write leveling. The eye would then move away from the group's
crossings: those of lanes 0-3 span about 27 steps, so a phase about 40 steps from all of them
would exist (**estimate**). That needs per-group 4x clocks for DQ and DQS, while the serializers'
slow clock stays put. It is a PHY rewrite, not a small change.

### Decision (revised 2026-09-29)

**Go for LiteDRAM, pending one more card session.**

**The test image:** both channels, one DQS phase per channel, each picked by a BIST scan. The
session answers two questions:
- **Channel 1:** its calibration and write window.
- **Temperature:** 30-40 min of continuous BIST on both channels, which heats the card towards
  its working temperature. The window is rescanned every 5 min on both channels, and the FPGA
  temperature is logged from the XADC.

**Rule for the result:**
- **Both channels keep a common write window of at least ~150 ps across the temperature
  sweep:** integrate with one BIST-picked DQS phase per channel, rechecked when the temperature
  changes.
- **Otherwise:** build the per-group DQ+DQS PHY. The MIG native path (stage 1) stays the fallback.

**Either way, production calibration must pick the DQS phase with a traffic check.**

**The parts every path needs are started (branch `litedram-int`):**
- `otpu_native_dram`: the slice ports on native ports, replacing `otpu_axi_dram`.
- `otpu_mem_ch`: one channel's accelerator and XDMA masters on a generic native port, with a
  read-modify-write for LiteDRAM's ECC port or wr_bytes for the MIG. It replaces the SmartConnect.
- The host calibration flow.

### The two-channel image on the card (2026-09-29, opentpu)

Logs: `docs/data/litedram/card2-*.log`. Both channels are built as in the one-channel image, with
one difference: channel 1's DQS clock comes from a **second MMCM** (fine phase shift moves every
fine-PS output of an MMCM together), while its DQ, CK and serializer CLKDIV stay on the first.

- **Channel 0** passes, as before. BIST at 7.69 / 7.76 GB/s (90.1 / 91.0% of peak) with 0 errors.
  The common write window is 12 steps (201 ps) at FPGA 53 C / board 47 C (234 ps in the cooler
  first session).
- **Channel 1** fails. Calibration passes, but the BIST has errors on all nine lanes, and no DQS
  phase works for every lane.
- **Channel 0's temperature run** (`card2-temp-ch0.log`, 35 min): 3578 back-to-back BIST passes
  of 2 GiB, 0 wrong beats. The window was rescanned 7 times and stayed at 12 steps (201 ps), with
  the running phase at -6 / +5 steps from its edges every time. But the BIST barely heats the
  card: FPGA 54.4 to 55.0 C, board 47.5 to 47.8 C. So this is a stability run at the card's idle
  temperature, not a temperature sweep. The first session measured 234 ps at a cooler, unlogged
  temperature.

**The DQS scan with the write latency forced** (`ld_host.py wscan`, both channels in the same
minute, FPGA 54.5 C, board 47.5 C). Each lane's latency is held at bitslip 0, then at 6, so a
lane's window at one latency shows whole instead of being cut where the calibration switches
latency. Steps of 16.7 ps; 64 MiB BIST per step; "edges" = steps from a lane's last clean phase
to the first with half its beats wrong.

| lane | ch0 wl 0 | ch0 wl 6 | ch1 wl 0 | ch1 wl 6 |
|---|---|---|---|---|
| 0 | -16..+24 (41) | - | +18..+32 (15) | - |
| 1 | -13..+23 (37) | -22 (1) | +19..+32 (14) | - |
| 2 | +3..+23 (21) | -21..-5 (17) | +7..+32 (26) | - |
| 3 | +9..+26 (18) | -22..-1 (22) | +28..+33 (6) | +6..+8 (3) |
| 4 | - | -19..+25 (45) | +25..+33 (9) | - |
| 5 | - | -21..+25 (47) | +32..+33 (2) | +6..+12 (7) |
| 6 | - | -22..+25 (48) | +21..+33 (13) | - |
| 7 | - | -19..+24 (44) | +24..+31 (8) | - |
| 8 | - | -19..+20 (40) | - | +6..+17 (12) |
| edges | 1-4 steps | | 8-12 steps | |

Reading (**measured** unless marked):
- Forcing the latency widens no lane. Every window is bounded by the two limits of section 7's
  corrected reading: the DQ-DQS eye, common to the channel's lanes, and each lane's tDQSS
  crossing, around which it fails at both latencies.
- **The fly-by is genuine and no larger on channel 1.** Its crossings, taken where the
  calibration switches latency, span 34 steps (570 ps): lane 2 -6, 0 +6, 1 +8, 6 +8, 7 +12,
  4 +13, 3 +19, 5 +21, 8 +28. On channel 0, lanes 0 to 8 already span at least 46 steps, and
  lanes 4-7 cross outside the eye.
- **What is worse on channel 1 is the eye and every edge.** The eye is 28 steps (+6..+33, 467 ps)
  against 48 (-22..+25, 800 ps). The zone around a crossing is 19 steps (lanes 3 and 5) against
  7-9 (lanes 2 and 3 of channel 0). Every edge is 8-12 steps soft against 1-4: the error counts
  fall off like Gaussian tails over 10-15 steps, where channel 0's go from none to all in 2-3.
  Reads look the same on both channels (windows of 9-11 taps).
- **Cause (inferred, not proven):** channel 1's DQS clock comes from its own MMCM, whose jitter
  is independent of the MMCM that clocks its DQ and CK. Its DQS serializer also has CLK from one
  MMCM and CLKDIV from the other, which the OSERDESE2 does not allow. Both would give soft,
  jittery edges on the write side only. The board is not implicated.
- **Consequence for the design:** separate MMCMs per clock group would repeat the penalty
  (**measured** on channel 1: an eye 20 steps narrower, crossing zones 10 steps wider). DQS-only
  groups cannot work either (section 7: the eye does not move with DQS). DQ has to move with its
  DQS, and all of a channel's interface clocks have to come from one MMCM. Section 8.

## 8. Write leveling by clock groups (WL7DDRPHY, 2026-09-29)

The goal: both channels at DDR3-1066 with at least 150 ps of write window per clock group,
through a temperature run. The route: real write leveling, meaning DQ moves together with its
DQS against CK, in two groups per channel. Built by `ld_test.py --phy wl`
(`tools/litedram/wl7ddrphy.py`), calibrated by `ld_host.py all` / `temp`
(`opentpu/host/ddrcal.py`: `WriteClocks`, `calibrate_groups`).

**The PHY (`WL7DDRPHY`).** It is LiteDRAM's A7DDRPHY with its clocks split. Each byte lane's
write side (its 8 DQ serializers, DQS serializer, and their CLKDIV) runs on its group's static
clocks, and so does the read capture (since d2c1ede; `ldtest3` captured on the shifted pair, see
below). CK and the commands run on the clocks that the fine phase shift moves:

| clock | MMCM output | frequency | phase | drives |
|---|---|---|---|---|
| `sysc` | CLKOUT0 | 133.33 MHz | fine PS | CLKDIV of CK and the commands |
| `sys4xc` | CLKOUT1 | 533.33 MHz | fine PS | CK, commands |
| `sysw` | CLKOUT2 | 133.33 MHz | 0 (sys's phase) | CLKDIV of every write serializer and read ISERDES |
| `sys4xw a` | CLKOUT3 | 533.33 MHz | 0 | group 0's DQ, every read ISERDES |
| `sys4xw a dqs` | CLKOUT4 | 533.33 MHz | 90 deg | group 0's DQS |
| `sys4xw b` | CLKOUT5 | 533.33 MHz | offset | group 1's DQ |
| `sys4xw b dqs` | CLKOUT6 | 533.33 MHz | offset + 90 deg | group 1's DQS |

- **DQ keeps its DQS at 90 deg**, which is the measured eye centre (channel 0's eye is centred
  within 2 steps of it). CK moves against the pair, so the eye no longer limits the phase; only
  the lanes' tDQSS crossings do. Every lane still picks its own write latency (bitslip).
- **Why CK moves and not DQ/DQS (measured, the first build):** Vivado's bitgen rejects an MMCM
  whose fine-phase outputs differ in their sub-VCO phase fraction (DRC 12-1117, FINE_PS_FRAC). So
  DQ at 0 and DQS at 90 deg cannot both be fine-phase outputs of one MMCM, and a DRP offset in
  1/8 VCO steps cannot go on one either. CK and its CLKDIV both sit at 0 deg, and moving CK
  against static DQ/DQS gives the same relative motion.
- **One MMCM per channel.** All of a channel's interface clocks, CK included, come from one VCO,
  so no clock pair in the DDR3 interface carries another MMCM's jitter. The MMCM has exactly the
  7 outputs this takes.
- **Group 1's offset from group 0 (dropped: measured on `ldtest3`, every lane of the group
  failed).** The host set it through the MMCM's DRP: CLKOUT5/6's PHASE_MUX and DELAY_TIME (static
  outputs), in 1/8 VCO steps (117 ps = 7 fine steps), with the MMCM held in reset. It moved only
  group 1's serializer CLK against its CLKDIV (`sysw`). Every offset tried lost every group 1
  lane: 1, 2, 3, 13, 14 and 15 eighths on channel 0, 6 on channel 1. The whole write side moved
  together (CLK and CLKDIV) worked at 4 eighths and failed at 1, 2, 3 and 5. The groups remain in
  the code; every lane is in group 0 by default, and the host leaves group 1 at 0.
- **Cascaded from `sys` at DIVCLK_DIVIDE 1**, with the feedback through a BUFG. The outputs keep a
  fixed phase to `sys`, so Vivado times the paths from `sys` (the controller and the PHY's
  logic) to `sysw` as synchronous. A cascade from the 50 MHz oscillator would need DIVCLK_DIVIDE 3
  for 1066.67 MHz, and the phase against the controller's clock would then be one of three
  values after each lock. The top MMCM (`sys` only) runs at 1200 MHz VCO with DIVCLK 1.
- **Commands cross from `sys` to the shifted `sysc` through registers on `sys`'s falling edge**
  (the read data did too until d2c1ede). The host keeps `sysc` within half a tCK (0.94 ns) of `sys`. Every phase
  has an equivalent a tCK away against CK, and the write latency calibration absorbs the tCK
  (`DqsPhase` wraps its targets into [-56, +56) steps). Registered half a cycle (3.75 ns) away
  from the other side's edge, the data keep about 2.8 ns of setup and hold. The build constrains
  both directions with 1.0 ns of clock uncertainty on setup and on hold. At phase 0 the PHY's
  timing is A7DDRPHY's, cycle for cycle. The tristate controls (TQ in BUF mode, not clocked by
  the serializer) stay as A7DDRPHY has them.

**Groups (by bank).** Each group's clocks then reach only its own banks' clock regions:
- channel 0: group 0 = bank 11 (lanes 0-3), group 1 = banks 12 and 13 (lanes 4-8);
- channel 1: group 0 = bank 16 (lanes 0, 1, 3, 8), group 1 = banks 17 and 18 (lanes 2, 4, 5, 6, 7).

**Calibration (`calibrate_groups`, since d2c1ede).** Group 1 at offset 0; the common CK phase is
scanned over a tCK, each step with a full calibration (per-lane write latency, read leveling) and
a 64 MiB BIST per lane; the phase goes to the centre of the longest run common to all lanes, and
the channel is calibrated there. (Until d2c1ede a second step set group 1's offset to bring its
run onto group 0's and scanned again; see above for why it is gone.)

The temperature run rescans every 5 minutes and logs each group's run.

**Expected windows** (**estimates**, from the crossing positions above; they assume channel 0's
7-9-step crossing zones, i.e. that one MMCM per channel restores its sharp edges on channel 1):
- **Channel 1, whose crossings span 34 steps:** about 67 steps (1.1 ns) even as one group, and
  more for each of its two groups.
- **Channel 0 as one group:** only the gap between lane 3's crossing (+4) and lane 8's (+25):
  12 steps, the 200 ps it has now. That holds unless lanes 4-7 cross far enough away, and the
  first image cannot see them.
- **Channel 0 in two groups:** group 0 (crossings -21..+4) about 78 steps; group 1 (lane 8 at
  +25, lanes 4-7 somewhere beyond +29) probably at least 40.

### Builds (Vivado 2026.1, omarchy)

- **First attempt (DQ/DQS shifted):** routed, but bitgen refused it (FINE_PS_FRAC, above).
- **CK shifted, falling-edge registers as a migen domain:** WNS -2.226 ns, 711 failing endpoints,
  all on the `sys` -> `sysc` crossings. Vivado absorbed none of the clock inversions: the
  registers were clocked through a LUT, with 3.3-3.6 ns of skew. Now they are FDREs with
  IS_C_INVERTED on the global `sys` (1572 of them).
- **CK shifted, FDREs (76c1e7e):** WNS +0.176 ns, WHS +0.018 ns, all constraints met, 24 of 32
  BUFGs. The crossings through the cascaded MMCMs, **measured** on the routed design (worst path
  of each; uncertainty = the 1.0 ns phase range plus 0.25 ns of jitter and phase error):

  | crossing | setup slack | hold slack | uncertainty |
  |---|---|---|---|
  | sys -> sysc0 (commands, resets) | +0.176 ns | +2.098 ns | 1.248 ns |
  | sysc0 -> sys (read data) | +0.506 ns | +2.577 ns | 1.253 ns |
  | sys -> sysw0 (write side, static) | +1.349 ns | +0.105 ns | 0.248 ns |
  | sys -> sysc1 | +0.300 ns | +1.931 ns | 1.248 ns |
  | sysc1 -> sys | +0.852 ns | +2.582 ns | 1.253 ns |
  | sys -> sysw1 | +1.107 ns | +0.104 ns | 0.248 ns |

- **Clock primitives:** the `sys` MMCM, each channel's MMCM, the IDELAYCTRL reference PLL, and
  XDMA's pipe-clock MMCM. Six IDELAYCTRLs, replicated by Vivado into the six DDR3 bank regions
  from the one 200 MHz reference. The banks are in clock regions X0Y0-2 (channel 0: banks 11, 12,
  13) and X0Y5-7 (channel 1: banks 16, 17, 18). The placer had put channel 0's MMCM at X0Y5 and
  channel 1's at X0Y0, each among the other channel's banks. `--mmcm-locs` now places them in
  their own regions: X0Y2 (bank 13) and X0Y6 (bank 17, the command bank). XDMA's MMCM is at X0Y1,
  the `sys` MMCM at X0Y4, the PLL at X0Y1. All the PHY clocks leave through BUFGs.
- **CK shifted, MMCMs placed (5cc8986, the card image `ldtest3`):** the same Verilog plus the two
  LOCs. WNS +0.096 ns, WHS +0.032 ns, all constraints met. ldmmcm0 is at X0Y2 and ldmmcm1 at X0Y6.
  The placer then moved the `sys` MMCM to X0Y5 and the PLL to X0Y7; XDMA's MMCM stays at X0Y1.
  The crossings, **measured** as above:

  | crossing | setup slack | hold slack | uncertainty |
  |---|---|---|---|
  | sys -> sysc0 | +0.096 ns | +2.191 ns | 1.248 ns |
  | sysc0 -> sys | +0.530 ns | +2.586 ns | 1.253 ns |
  | sys -> sysw0 | +1.701 ns | +0.081 ns | 0.248 ns |
  | sys -> sysc1 | +0.224 ns | +1.955 ns | 1.248 ns |
  | sysc1 -> sys | +0.492 ns | +2.578 ns | 1.253 ns |
  | sys -> sysw1 | +1.358 ns | +0.101 ns | 0.248 ns |

  Bitstream md5: 1ae9ef50d2ddec89f3282e29104178bf.

### `ldtest3` on the card (2026-09-29, opentpu, both channels, FPGA 54-55 C)

The session stopped after each channel's calibration and forced-latency scan: calibration failed
on both channels (one dead lane each), so the soak and the temperature run would have run
uncalibrated. The card went back to production (se-cand3, build 002569bc) afterwards; its
selftest passed (11 of 11). Logs: `docs/data/litedram/card3-*.log`.

**Every lane but one per channel: the CK shift works.** The forced-latency scan at group 1
offset 0, write bitslips 0, 2, 4 and 6, a 64 MiB BIST per fine step (`card3-wscan4-ch*.log`).
Each lane fails only where its write latency flips, 6-10 steps wide (both channels' zones as
sharp as channel 0's were on `ldtest2`); with each lane at its own latency, the steps where every
lane passes (**measured**, the complement of the zones):

| channel | lanes | failing zones (CK steps) | common run |
|---|---|---|---|
| 0 | 0-6, 8 | -42..-22, -7..+3, +14..+24 | **45 steps (750 ps)**, +25..-43 across the wrap; also 14, 14, 10 |
| 1 | 0-7 | -3..+3, +6..+17, +24..+29 | **79 steps (1.32 ns)**, +30..-4 across the wrap |

For comparison, `ldtest2` (DQS shifted alone) had 12 steps on channel 0 and none usable on
channel 1. One write group is enough: the offset is not needed (and did not work, above).

**The dead lanes: one DQ bit a tCK late on the read.** Channel 0's lane 7 and channel 1's lane 8
fail at every CK phase and every write bitslip. With marked data (beat k = 0x11 (k + 1)) at the
lane's least bad read setting, 7 of its 8 bits are right and one reads the pattern 2 beats (one
tCK) late: channel 0 dq61 (and dq60 at times), channel 1 dq65 (and dq71 at times). The same bits
at CK steps -45, -10, +20 and +40 and after every PHY reset (`card3-bits*.log`). A single 1
written at beat 2 or 6 and read from column 0 and from column 4 (DDR3 reads column 4 as beats
4-7, 0-3; writes ignore it) tells a late write from a late read: the 1 written at beat 6 shows
at position 4 from column 4, and the one at beat 2 does not show from column 4. So the write is
right and the bit comes out of its ISERDES a CLK after its lane (`card3-burst-order.log`). The
bits are the DQ pins nearest their bank's end: IOB Y51 and Y53 in bank 12, Y252 and Y253 in bank
16. `ldtest3`'s ISERDES were on the shifted pair (`sys4xc` / `sysc`); `ldtest2` captured on
static clocks and had no late bit on either channel, and `ldtest3`'s write serializers on the
static pair, the same pins among them, wrote every bit right. The reset's timing is not the
cause: the late bits did not change when it arrived 1.4 ns later against the capture clock
(the CK phase scan) or 470 ps later against the write clocks (below). d2c1ede moves the read
ISERDES onto the static pair and gives every lane's serializers their own reset register
(constrained to 1.2 ns).

**The DRP offsets (`card3-g1-offsets.log`, `card3-write-shift.log`).** Group 1 alone at 1, 2, 3,
13, 14, 15 eighths (channel 0, CK at +40): every group 1 lane lost its read window. The whole
write side (CLKOUT2-6 together, CK moved with it): 4 eighths as at 0 (the same late bit), 1, 2, 3
and 5 eighths every lane lost.

**Status:** d2c1ede (read capture static, one group) building; the checkpoint run follows.
