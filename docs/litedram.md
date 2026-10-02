# LiteDRAM instead of the MIG: an assessment (draft)

Draft, 2026-09-29. This is an investigation; the design is unchanged. Labels: **measured** means
a Vivado run, a simulation or a card run that was actually done (tool and version given).
**estimate** means arithmetic. Sections 1-6 predate the card; sections 7 and 8 are the card runs
(the test images, then the production image); section 9 is the board build; section 10 is
the calibration CPU in the core; section 11 is the controller co-simulated, where 133.33 MHz
loses, and the clock x DDR3 rate grid.

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
`otpu_axi_dram`; since `ld-default` it is the only adapter). Its header comment has the details;
in short:

- **One command per beat, per core cycle and channel**, from the A queue (a run of APF read
  commands, or a byte-enabled write), the SW queue's fill reads and writes, and the B queue.
  Write data goes with its command; partial beats of ports A and B keep their byte mask (the
  channel module does the read-modify-write). The SW port keeps its gather and read-fill.
- **Read data** returns in command order, so a tag FIFO per channel routes each beat to B, A or
  an SW slot; room is reserved before a read goes out, and simulation checks for overflow.
- **Ordering by the command stream instead of write responses.** An SW fill read waits until
  the older SW writes of its beat have entered the stream, not until they are done. A write
  drops the A run or reused beat it touches when it goes out, not when it is answered.
  `wr_idle` waits for `n_wdone`. The host's writes (XDMA, the channels' other master) reach
  neither: `a_flush` drops both between runs, at a program load and when a WAITW holds
  (2026-10-02, after the 35B's back-to-back embed runs read token 0's scales for tokens 1 and 2;
  tests/test_rtl.py `test_porta_*`, two runs in one simulation: tb_top `+runs`).
- **Simulation:** `sim/verilator/otpu_native_mem.sv` (random command and write-data
  backpressure, in-order read data with jitter, late `n_wdone`, the DDR3 bank model of
  `otpu_axi_mem`); `OTPU_NATIVE=1` runs the AXI-path tests and the board model on it (since
  `ld-default` the AXI-path runs always do; `otpu_axi_mem` is removed).
- **Area, yosys** (`synth_xilinx`, **measured**): 10,620 LUTs, 3,364 FFs, 1,492 LUT RAM cells,
  against `otpu_axi_dram`'s 13,464, 4,914 and 1,880. Logic depth 4.57 ns against 6.01 ns.
- **Cycles, simulation (measured,** `tools/litedram/path_bench.py`**, removed with the AXI path):** the DDR3-1066 bank
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
but not yet integrated. (Since generalized into `otpu_mem_ch`, which drives either controller's
native port, with `otpu_mig_native` for the MIG; tested by `tb_memch` via `tools/memch_test.py`.)
The key unknown for LiteDRAM is calibration and write timing on all 9
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

**Read framing per bit (8cbfd3b).** The ISERDES of one lane do not all frame the word alike: a
bit can come out a CLK (two bitslips) early or late against the other bits of its lane, with a
full-width window at its own framing (`ldtest3d` and `ldtest3e` below: different pins in each
build). The PHY's CSR `dly_sel_bits` (8 bits, reset 0xFF) masks the read and write bitslip
strobes per DQ bit of the lanes that `dly_sel` picks; DQS and DM follow only with every bit
selected. Read leveling (`ddrcal`: `read_scan`, `lane_best_bits`) counts wrong reads per DQ bit.
A lane's window is where every bit reads right at the lane's bitslip or two away. The widest
window wins, and among equal ones the one with the fewest bits off. Those bits then get their
own bitslip. The host finds the CSR by name, and cores without it are calibrated per lane as
before.

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
- **Reads on the static clocks, a reset register per lane (d2c1ede, `ldtest3d`)** and **per-bit
  read framing (8cbfd3b, `ldtest3e`, the checkpoint image):** the same LOCs. The read data now
  cross from `sysw`, so there is no `sysc` -> `sys` path. Every clock crossing meets its
  constraint. The serializer resets' `set_max_delay` does not: 1.2 ns on `ldtest3d`, with routes
  up to 1.66 ns (WNS -0.967 ns, 308 endpoints); 2.0 ns on `ldtest3e` (WNS -0.308 ns, 85
  endpoints). That constraint is hygiene (section `ldtest3` below: the reset's arrival does not
  move the framing), so neither image was rebuilt for it. **Measured**, worst path of each:

  | crossing | `ldtest3d` setup / hold | `ldtest3e` setup / hold |
  |---|---|---|
  | sys -> sysc0 | +0.271 / +2.469 ns | +0.340 / +2.466 ns |
  | sys -> sysc1 | +0.234 / +2.459 ns | +0.193 / +2.461 ns |
  | sys -> sysw0 | +0.840 / +0.101 ns | +1.516 / +0.109 ns |
  | sys -> sysw1 | +1.545 / +0.103 ns | +1.247 / +0.131 ns |
  | sysw0 -> sys (read data) | +3.726 / +0.093 ns | +3.623 / +0.081 ns |
  | sysw1 -> sys | +4.465 / +0.064 ns | +3.593 / +0.061 ns |

  `ldtest3e` bitstream md5: 79e977a4c8360812daa89a831f8cff4c.

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

### `ldtest3d` on the card (d2c1ede: reads on the static clocks, 2026-09-29)

Logs: `docs/data/litedram/card3d-*.log`.

**Channel 0: all nine lanes, calibrated.**
- **Calibration:** the first scan found a 28-step common run (469 ps). The second found 45 steps
  (753 ps) with CK at +50 (`card3d-all-ch0.log`, `card3d-all2-ch0.log`).
- **BIST:** clean.
- **300 s soak:** 268 passes of 2 GiB x 2 data modes, 0 errors (`card3d-soak-ch0.log`).
- **Temperature run:** stopped at 28:20 when `ldtest3e` was ready. It had 2,888 BIST passes and
  0 wrong beats, with the FPGA at 54.3-55.2 C. The rescans every 5 minutes found 44, 44, 25, 44
  and 45 steps; the 25 is -13..+11 around the running phase against -13..+30 in the others
  (`card3d-temp-ch0.log`).

**Channel 1: no common phase.** Two lanes had no read window at any CK phase
(`card3d-all-ch1.log`). A per-bit read scan, which shows each bit's passing taps at each read
bitslip (`card3d-diag6.log`), found:
- in lane 3, dq27 reads right two bitslips later than the other seven bits;
- in lane 8, dq64 and dq67 read right two earlier than the other six.

Each of these bits has a full window of 10-11 taps at its own framing. The misframed bits move
with the build: channel 0's lane 7, which `ldtest3` lost to dq61, calibrated on `ldtest3d`. The
reset reached those ISERDES in 0.58-0.72 ns, so its arrival did not decide the framing. 8cbfd3b
adds the per-bit read framing (above).

### `ldtest3e` on the card: the checkpoint (8cbfd3b, 2026-09-29)

The image: `ldtest3` with reads on the static clocks, a reset register per lane, and per-bit read
framing; one write group; group 1 at offset 0. The session (`task3.sh`) ran on both channels, per
channel:
- calibration and a BIST;
- the forced-latency scan;
- a 300 s soak.

Then came the 35 min temperature run on both channels together, and the BIST bandwidth. Logs:
`docs/data/litedram/card3e-*.log`. **Measured:**

| | channel 0 | channel 1 |
|---|---|---|
| common window (calibration scan) | **45 steps, 753 ps**, CK at +50 | **73 steps, 1222 ps**, CK at +64 (-48) |
| forced-latency scan, best latency per lane | +90..+134 (45 steps) | +76..+148 (73 steps) |
| write latency | every lane at bitslip 6 | every lane at bitslip 0 |
| read windows at the chosen taps | 9-11 taps (703-859 ps) | 9-10 taps (703-781 ps) |
| bits off their lane's read framing | none | lane 3 bit 3; lane 8 bits 1, 2, 5, 7 (+2 each) |
| BIST, 2 GiB x 2 passes x 2 patterns | 0 errors | 0 errors |
| 300 s soak | 268 passes, 0 errors | 268 passes, 0 errors |
| BIST bandwidth | write 7.69 GB/s (90.1%), read 7.76 GB/s (91.0%) | the same |

Channel 1 needs the per-bit framing: five bits in two lanes read right only at their lane's
bitslip + 2. Channel 0 needs none on this build. Lane 3's bit 3 (dq27) is the same pin as on
`ldtest3d`; lane 8's set differs.

**The temperature run** (`card3e-temp.log`): 35 min of back-to-back BIST on both channels at
once, with the FPGA at 54.4-55.2 C and the board at 47.5-48.0 C. Each channel ran 3,209 passes
of 2 GiB with 0 wrong beats. The windows were rescanned every 5 minutes, 7 scans in all:
- channel 0: 737-753 ps (-22..+21 or +22 around the running phase);
- channel 1: 1222 ps every time (-36..+36).

The running phase stayed inside every window. The BIST afterwards was clean on both channels
(`card3e-bist.log`).

**Verdict: GO.** Both channels calibrate at one common CK phase with per-lane latency; their
BIST, 300 s soak and 35 min temperature run are clean; and the common windows (753 and
1222 ps) are 5 and 8 times the 150 ps criterion. The production core (`gen_core.py --phy wl`)
is this PHY.

**`rd_reg` (86bee3c): an opt-in register after the read bitslip mux.**
- **What it is:** `gen_core.py --rd-reg` adds the register and lengthens the read latency by one
  sys cycle.
- **Why:** in the first fused image (14875bf, FAST=1) the mux's select fed the ECC error counters
  through 9 LUT levels and missed 133.33 MHz by 0.409 ns.
- **Effect:** OOC synthesis of the two cores (Vivado 2026.1, **measured**, before place and route)
  takes those paths from 9 levels (5.50 ns) to 6 or 8 (4.02 ns).
- **Status:** off in the qualified core. 5e5a58ab met timing without it under full-effort place
  and route.
- **On the card:** ldcpu's isolated test image (`ld_test --selfcal`) is the first card run with
  `rd_reg`. Both channels had 0 BIST errors and a 300 s soak each with 0 errors, at 58 C
  (**measured**, as ldcpu reported it).

**Reading the ldtest3e numbers:**
- **Window width:** the scans step the common CK phase in 16.7 ps steps. A lane fails only
  where its write latency flips, so the width is the gap between the lanes' tDQSS crossings,
  not a data eye.
- **Temperature coverage:** the run held the FPGA near 55 C. It does not cover a cold start or
  a hot card; the rescans give the host the drift to recalibrate on if the window moves.

### The production image on the card (ld-top 14875bf, 2026-09-29)

The first fused image: accelerator plus this PHY. It ran on opentpu with host tree ld-top
5e5a58a. Build facts:
- MEM=litedram, MCOLS=4 (systolic MXU), core 100 MHz, FAST=1;
- core at 8cbfd3b, without `rd_reg`;
- routed WNS -2.78 ns. The failing paths were the ECC counters, a reset strobe into the 50 MHz
  domain and the serializer resets; none are read-data paths.

**Measured** (logs in `~/otpu-build/fused-14875bf` on opentpu):

| check | channel 0 | channel 1 |
|---|---|---|
| `ld_host.py --fused all`: common window | 44 steps (737 ps), CK +50 | 73 steps (1222 ps), CK +64 |
| write latency | every lane at bitslip 6 | every lane at bitslip 0 |
| bits off their lane's read framing | none | lane 3 bits 1, 3; lane 8 bit 6 |
| BIST, 2 GiB x 2 x 2 | 0 errors | 0 errors |
| 300 s soak | 268 passes, 0 errors | 268 passes, 0 errors |
| `otpu-memcal cal --force` (HOSTCAL, the `memcal.ensure` path) | 25.1 s, same phase and window | 26.0 s, same phase and window |

- **Channel 1's misframed bits differ in each build.** dq27 (lane 3 bit 3) was off in all three
  builds (`ldtest3d`, `ldtest3e`, 14875bf); the other bits moved from build to build. The
  per-bit framing absorbed every case.
- **`otpu-selftest`, ALL PASS:** it ran after the calibration. ld-top's selftest did not yet
  calibrate a HOSTCAL image itself (fixed in 1b97d90).
  - scrub: 4 GiB in 2.8 s;
  - host to card 1.37 GB/s, card to host 1.02 GB/s;
  - kernel, vops and stream all pass.
- **Prefill and decode counters, 6 configurations (`tools/qual/perf.py`):** decode was 5.448,
  1.988 and 6.573 Mcycles/token for Qwen3-0.6B, LFM2.5-230M and Qwen3.5-0.8B in int8, and 3.730,
  1.357 and 4.673 in fp4.
  - DRAM ran at 11.9-12.5 GB/s while busy, 70-74% of the DDR3-1066 peak. This is an
    **estimate**, from arithmetic: at a 100 MHz core the 128-byte port caps the path at
    12.8 GB/s, so it ran at 93-98% of that cap.
  - For comparison, se-cand3 (MIG, MCOLS=2, 120.755 MHz) took 5.566, 2.049, 6.907, 3.792, 1.397
    and 4.950 Mcycles/token. The core clocks differ, so the cycle counts are not a like-for-like
    comparison.
- **Not run:** token-exact against the ISA simulator. The qual run was stopped for the next
  build (5e5a58ab: the same core, all timing met). Then the card's JTAG chain went empty and a
  cold power cycle brought back the factory image.

### The merge candidate on the card (ld-top 5e5a58ab, 2026-09-29): qualified

Changes from 14875bf:
- the same core (8cbfd3b), with the serializer resets' max delay at 3.0 ns;
- a scoped false path for the reset strobe into the 50 MHz domain;
- the SW hazard counts in `otpu_native_dram` without a reset;
- the strategy `Performance_ExplorePostRoutePhysOpt` at a 100 MHz core.

Routed: WNS +0.128 ns, WHS +0.016 ns, every constraint met. It ran on opentpu with host tree
ld-qual b719cd3, whose `otpu-selftest` calibrates a HOSTCAL image itself. Logs are in
`~/otpu-build/fused-5e5a58ab` on opentpu. **Measured:**

| check | channel 0 | channel 1 |
|---|---|---|
| `ld_host.py --fused all`: common window | 44 steps (737 ps), CK +49 | 72 steps (1205 ps), CK +64 |
| write latency | every lane at bitslip 6 | every lane at bitslip 0 |
| bits off their lane's read framing | none | lane 3 bit 1 |
| BIST, 2 GiB x 2 x 2 | 0 errors | 0 errors |
| 300 s soak | 268 passes, 0 errors | 268 passes, 0 errors |

**Channel 1's bits off their lane's framing, across four builds:**
- `ldtest3d`: lane 3 bit 3 at +2, lane 8 bits 0 and 3 at -2;
- `ldtest3e`: lane 3 bit 3, lane 8 bits 1, 2, 5 and 7, all at +2;
- 14875bf: lane 3 bits 1 and 3, lane 8 bit 6, all at +2;
- 5e5a58ab: lane 3 bit 1 at +2.

Channel 0 needed none in any build. The 3.0 ns reset constraint added no misframed bits.

**`tools/qual/qual.sh fast`, 38 min, 0 FAIL lines:**
- **`otpu-selftest` ALL PASS**, before and after. Its calib stage calibrated both channels
  through `memcal.ensure` in 50.7 s.
- **Token-exact against the ISA simulator:** all 12 pass (Qwen3-0.6B, LFM2.5-230M and
  Qwen3.5-0.8B, in int8 and in fp4 with an int8 head, each per-position and resident).
- **Decode:** 5.445, 1.988 and 6.573 Mcycles/token (int8); 3.731, 1.357 and 4.673 (fp4).
  DRAM ran at 11.9-12.5 GB/s while busy.
- **Streamed decode (`decode_profile`, fp4):** 28.9, 73.6 and 21.1 tok/s wall.
- **Warm soak and diag:** a 3 min warm soak, FPGA 64 to 66 C. Then `otpu-diag` with the quick
  memory test: 129 PASS, 0 FAIL (isa 93 / 93, mem 11 / 11).

Afterwards the card went back to se-cand3 (build 002569bc), whose selftest passed.

## 9. The board build (branch `ld-top`, 2026-09-29; the only build since `ld-default`)

The accelerator and XDMA reach the DDR3 channels through native ports, with no SmartConnect and
no AXI front end in the controllers: the committed LiteDRAM core with WL7DDRPHY (section 8;
`boards/ypcb-00338/litedram/`, `tools/litedram/check_core.sh`), top `otpu_fpga_top_ld`, block
design `bd_native.tcl`. `make bit` builds it.

Until `ld-default`, `create_project.tcl` also built the MIG controllers (`MEM=mig`: the MIGs'
AXI ports behind the SmartConnect, `otpu_fpga_top`, `bd.tcl`, the production images until this
build qualified; `MEM=mig_native`: the MIGs' native interface behind `otpu_mig_native`,
`otpu_fpga_top_mn`, parked as LiteDRAM's fallback and never built). Once this build qualified on
the card (ld-top 5e5a58ab, section 8) they were removed, with their adapter `otpu_axi_dram`,
their simulation models and tests (`git log` on `ld-default` has them).

`otpu_native_sys` holds:
- `otpu_board` (`otpu_native_dram` on the native masters);
- XDMA's DMA master, split by address bit 31 (`otpu_axi_split2`);
- one `otpu_mem_ch` per channel, in front of that channel's two controller ports (the even
  and the odd banks', section 11), in the controller's clock. LiteDRAM's ECC port takes whole
  beats only, so `otpu_mem_ch` read-modify-writes partial beats.

**`bd_native.tcl`**: XDMA, the XADC and the control SmartConnect (the MIG build's `bd.tcl` had
the same). XDMA's `M_AXI` is exported as `M_AXI_DMA`; the CSR window is at BAR0 0x10000. The
50 MHz clock sits on one BUFG shared with the core's MMCMs.

**The LiteDRAM core** is `gen_core.py --phy wl` (the arguments are in `core.json`, which
`check_core.sh` regenerates with). It has the same ports as the A7DDRPHY core it replaced, plus
since ld-2port a second user port per channel (section 11). Its CSRs add each channel's write
clock MMCM (`wclk` / `wclk1` at 0x7000 / 0x7800 of the window, for ddrcal's DRP and reset
access). Its XDC carries two sets of constraints:
- 1.0 ns of clock uncertainty on the sys <-> sysc crossings;
- the serializer resets' max delay: 3.0 ns (1.2 at d2c1ede, 2.0 at 8cbfd3b; at 2.0 ldtest3e and
  the first fused image, 14875bf, missed it by placement alone, 1.88 ns of route at 0 LUT
  levels, and ldtest3e calibrated and passed BIST on both channels).

`otpu_top_native.tcl` false-paths the paths from sys into the core's MMCM and PLL reset
chains (LiteX's FDCE chains in the 50 MHz clock) and no others; the standalone images do it
with `add_false_path_constraints`.

`otpu_top_ld.xdc` places the two write clock MMCMs at X0Y2 / X0Y6, as on the card images.

**Constraints:**
- `otpu_top_ld.xdc` (the board: the MIG builds' `otpu_top.xdc` less the MIG's lines) plus the
  core's XDC.
- Then, late and in implementation only, `otpu_mem_ch.tcl` (scoped to each `otpu_mem_ch`) and
  `otpu_top_native.tcl`. The latter has the report_cdc waivers for XDMA's read data, the TIMING-9
  waiver and max delays for LiteX's CSR crossing and the calibration flags.
- No `set_clock_groups` and no clock-level false paths anywhere: they would override
  `otpu_mem_ch`'s max delays.

**Checks:**
- Offline: `make lint` (`lint-ld` is the same).
- In Vivado without a build: `vivado/check_native.tcl`. It runs the project, validates the block
  design, then runs `synth_design -rtl` and checks the instances.
- Board model: `tb_board` with `MEM_NATIVE=2` (the default) puts `otpu_mem_ch` and a controller
  model (`sim/verilator/otpu_chmem.sv`) behind the native adapter, each channel in its own clock
  (`OTPU_NATIVE=ld`, the default; `OTPU_NATIVE=1`: the native memory model alone).

## 10. Calibration on the card: the core's own CPU (2026-09-29)

The host calibration (section 7, `ddrcal`) needs the host: the card's DDR3 is unusable until a
host tool has run it, 25 s per channel over PCIe. With `--selfcal` (`gen_core.py`, `ld_test.py`)
the LiteDRAM core carries a small CPU that runs the same algorithm at reset. The channels come
up calibrated, and the host only reads the result.

**The CPU (`tools/litedram/calcpu.py`).**
- VexRiscv, LiteX's `minimal` variant: RV32I, no caches, no bypassing, the prebuilt
  `VexRiscv_Min.v`. SERV would be about 200 LUTs, but at ~40 cycles per instruction the
  calibration would take minutes.
- **Area, measured** (Vivado 2026.1, the placed test image `ldcpu-d5eb5270`): the VexRiscv
  1104 LUTs, 726 FFs, 2 RAMB18 (its register file). The whole image against `ldtest3e`, the
  same design without the CPU: +1026 LUTs, +4 RAMB36 (the firmware memory), +3 RAMB18 (the
  CPU's two, the mailbox), +2128 FFs, of which about 1152 are `rd_reg`'s (on in this image, not
  in `ldtest3e`). CPU and glue: about 1.0K LUTs, 1.0K FFs, 4 RAMB36 and 3 RAMB18.
- **A hard FSM instead (estimate, not built):** the same algorithm (the per-bit pass masks over
  8 bitslips x 32 taps x 72 bits, the window search with the +-2 framing, the circular margins,
  the DRP and init sequences) would need 1.0-1.5K LUTs, 0.5-0.8K FFs and 1 RAMB18: about
  the CPU's LUTs, 4 RAMB36 fewer. Every change to it would be RTL and a new bitstream. The
  CPU runs `ddrcal`'s algorithm, checked write for write against it, and takes a new
  firmware from the host (below).
- It runs in the core's sys domain (133.33 MHz) and has its own memory: 16 KB, true dual-port
  (instructions on one port, data on the other; 4 BRAM36). The memory's initial content is the
  firmware. A read-ahead on the instruction port answers sequential fetches in one cycle. Both
  ports register their read data (the data port is READ_FIRST). LiteX writes a byte-writable
  WRITE_FIRST port as an asynchronous read of a registered address, and Vivado turned that into
  a block RAM whose collisions differ from the RTL (Synth 8-6430), which `build.tcl` stops on.
- A 256-word result mailbox, which the host reads through two CSRs, and a 64-bit cycle counter.
- Its window onto the core's CSR bus is a second master beside the host's, behind LiteX's
  round-robin arbiter. After each CPU access the window leaves the bus free for a cycle, so a
  waiting host access goes next. The SoC's memory map and every existing CSR address are
  unchanged: the core's `csr.csv` equals main's (14875bf's) plus the `selfcal` block, which is
  pinned at location 16 (0x8000).
- The CPU's Verilog is appended to `otpu_litedram.v`, its modules renamed `otpu_selfcal_*`, and
  the firmware is inlined as the memory's initial value, so the core stays one file.

**The firmware (`tools/litedram/selfcal_fw`).**
- C, a port of `ddrcal.calibrate_channel` and everything it calls: the write clock check and
  group 1 at 0, the CK phase scan over a tCK (at each step the JEDEC init, write latency, read
  leveling with per-bit framing, and a 64 MiB BIST), the centre of the run common to every
  lane, the calibration there, the controller given the PHY, `cal_ready`.
- It calibrates both channels, 0 then 1, as `memcal.ensure` does. A failed channel keeps
  `cal_ready` low; the other is still calibrated.
- It keeps the decisions of `ddrcal` exactly. The only change in kind: `ddrcal` counts wrong
  reads per DQ bit, and each of its decisions only asks whether a count is 0, so the firmware
  keeps one bit per DQ bit.
- It adds timeouts where `ddrcal` waits for ever (the phase shift's busy bit).
- `fw.py` builds it against the build's `csr.csv` and `sdram_init.py`: the CSR addresses per
  channel, the init sequence, the PHY settings, and the test patterns (Python's MT for seeds
  42, 84, 36, precomputed). 9.9 KB of code and constants, 3.4 KB of data, `-O2`.

**The CSRs (`opentpu/host/selfcal.py`).**

| CSR | |
|---|---|
| `selfcal_hold` | 1: the CPU is held in reset. It takes effect between two of the CPU's bus accesses (an access under way finishes), so the host's own accesses never collide with a half-done one. 0: the firmware starts over and recalibrates, dropping each channel's `cal_ready` as it reaches it. |
| `selfcal_config` | bits 1:0 the channels (reset 0b11), 15:8 the scan stride (reset 1), read when the firmware starts |
| `selfcal_status` | bits 31:16 `0x5CA1` (the core has the CPU; an older core reads 0 there), bits 4:1 the firmware memory's size (2^12 words), bit 0 held |
| `selfcal_state` | written by the firmware: 2 bits per channel (idle, running, ok, failed), bit 7 done, each channel's error code |
| `selfcal_mbox_adr` / `_dat` | the mailbox: per channel the CK phase, the common run, write latency per lane, read window / bitslip / tap per lane, per-bit read offsets, each lane's pass map over the scan, cycles |
| `selfcal_mem_adr` / `_dat` / `_rdat` | while the CPU is held, the firmware memory's data port is the host's: `_dat` writes the word at `_adr`, `_rdat` reads it (a write while the CPU runs is dropped) |

**A new firmware without a new bitstream.** `fw.py target BUILD_DIR OUT_DIR --fw-id ID` builds
the firmware for a core (its `csr.csv` and `sdram_init.py`), and `otpu-memcal selfcal --firmware
OUT_DIR/selfcal.bin` loads and runs it. `selfcal.load()` holds the CPU, writes every word of its
memory (the image, then zeros), reads the memory back, and the release restarts the CPU from
address 0 on the new code. The firmware has no initialised data (`link.ld` checks), so a restart
needs nothing but the image. The mailbox's word 1 gives the running firmware's ID. The loaded
firmware stays until the FPGA is configured again; the bitstream's own comes back then. A
permanent change is the firmware in the bitstream (the core regenerated, or `updatemem` on the
memory's BRAMs; not set up).

**The host (`opentpu/host/memcal.py`).**
- `ensure()` on a core with the CPU:
  - the STATUS bits set: nothing to do;
  - the CPU still running: it waits (up to 60 s) and saves the CPU's result;
  - a channel the CPU failed: the host holds the CPU and calibrates that channel.
- `otpu-memcal cal --force` holds the CPU and calibrates from the host. The CPU stays held, so
  it does not recalibrate behind the host's back.
- `otpu-memcal selfcal` has the CPU calibrate again (`--firmware`: a new firmware first).
- `otpu-memcal` (status) prints the CPU's state and result.
- `ld_host.py selfcal [--rerun] [--soak] [--compare]` does the same for the test image, with the
  BIST, the soak and a host calibration to compare against.

**Verification.**
- **Firmware against ddrcal** (`tests/test_selfcal.py`, **measured** on the Mac): the firmware,
  built for the Mac, and `ddrcal` run on two identical simulated PHYs (`FakeBoard`). Their CSR
  write sequences must be identical (name, value, order), and the firmware's mailbox, decoded,
  must equal `calibrate_channel`'s result. The cases:
  - the production core's map at strides 4 and 1 (the whole 113-step scan on both channels);
  - the WL and A7 test images;
  - ldtest3e's and ldtest3d's per-bit framing, six random PHYs with random misframed bits, and
    a map without `dly_sel_bits`;
  - no common phase, and a write clock MMCM whose phases are wrong (both must fail alike);
  - two write clock groups, and one channel alone.

  All identical: for example 1,063,658 writes at stride 8.
- **The CPU in simulation** (`tools/litedram/selfcal_sim.py`, Verilator 5.047, **measured**):
  - the LiteX RTL (CPU, memory, mailbox, hold logic, the SoC's arbiter and CSR banks) runs the
    RV32I image built for the production core;
  - every PHY, controller and BIST CSR access leaves the SoC on a port that the testbench
    serves from a `FakeBoard`;
  - the CSR write sequence equals `ddrcal`'s (643,724 writes at stride 16, 1,064,298 at
    stride 8; 6,951,674 at stride 1 with the binary in the card's test image, run from reset
    as on the card), `c0_ready` / `c1_ready` rise, and the decoded results equal `ddrcal`'s;
  - held mid-scan, the CPU makes no access in the next 2M cycles; released, it calibrates both
    channels again;
  - `--upload-test` (the core with the upload CSRs, stride 16): a write to the memory while the
    CPU runs is dropped; `selfcal.load` puts a second build of the firmware (another ID) in the
    held CPU's memory and reads it back (33,810 cycles); released, the CPU reports the new ID,
    and its 643,724 CSR writes and its result equal its first run's.
- **Time:** 2.9 s per channel at stride 1 in simulation, 25 ms per scan step (**measured**;
  the simulated BIST completes at once). On the card, with the BIST: 6.3 s and 5.8 s (below),
  against 25 s per channel from the host. The CPU spends most of its time on instructions (CPI
  about 2.5, no bypassing), not on the bus (a CSR access on the bus in 2% of the cycles).

### The test image on the card (`ldcpu-d5eb5270`, 2026-09-29, opentpu)

The isolated image: `ld_test.py --selfcal` (litedram-int dfa6a74 with the CPU at d5eb527, so
`rd_reg` on: its first card run), both channels, stride 1. Session: `card_session.sh selfcal`,
after a JTAG load and a warm reboot. FPGA 58.3 C, board 50.8 C at the start. All **measured**.

| | channel 0 | channel 1 |
|---|---|---|
| the CPU at configuration | 6.28 s (the whole run 12.10 s, the CPU's cycle counter) | 5.83 s |
| CK phase, common run | +50, 45 steps / 753 ps | -48 (= +64), 73 steps / 1222 ps |
| write latency | 6 on every lane | 0 on every lane |
| bits read off their lane's bitslip | none | lane 3 bit 3 +2, lane 8 bits 0 and 3 -2 |
| the host (`ddrcal`, the CPU held) | CK +50, 45 steps, write latency 6, no bit off | CK -48, 73 steps, write latency 0, the same three bits |
| BIST (2 GiB x 2 modes x 2 passes) | 0 errors after the CPU's, the host's and the rerun's calibration | the same |
| 300 s soak at the CPU's calibration | 267 passes, 0 errors | 267 passes, 0 errors |
| the CPU again (hold, release) | 6.3 s: CK +50, 44 steps / 737 ps | 5.84 s: CK -48, 73 steps / 1222 ps, the same three bits |

- The CPU and the host agree on the CK phase, the window (within a step), write latency and
  the per-bit framing. ldtest3e's host runs found the same windows (45 / 73 steps at CK +50 /
  +64); channel 1's bits off their lane are ldtest3d's set (section 8: each build frames its
  own).
- Read leveling: the same windows (9-11 taps). On channel 0, lanes 3, 5 and 6 sometimes take an
  equal-width window at the adjacent bitslip; that flips between the CPU's own two runs too.
- The host's calibration of the same channels: about 25 s each (the step's time less its BIST
  runs; not timed directly), as HOSTCAL on the fused image (section 8).
- After the session: se-cand3 loaded back, warm reboot, `otpu-selftest` ALL PASS.
- Logs: `ldcpu-d5eb5270/selfcal*.log` on opentpu.
- One host fix came out of it before the rerun ran: `selfcal.release()` clears `selfcal_state`
  first. The firmware writes the state only once it starts, a millisecond after the release, and
  until then `wait()` saw the last run's done bit.

### The production core with the CPU

`boards/ypcb-00338/litedram/otpu_litedram.v` is main's core (14875bf's, the serializer resets at
3.0 ns) regenerated with `--selfcal` (`core.json`: `--phy wl --selfcal --fw-id 4461c052`, the
firmware built with riscv64-elf-gcc 15.2.0). `check_core.sh` regenerates it byte for byte.
Against the core without the CPU:
- `csr.csv` is the same plus the `selfcal` block; `sdram_init.py`, the XDC and the ID ROM are
  byte-identical, and the port list is the same (`check_offline.py --mem litedram`);
- every line of the Verilog is in the new core (CSR banks matched by location) except the
  one-master bus arbiter and the CSR read-data OR, which now include the CPU;
- `gen_core.py --rd-reg` adds `wl7ddrphy.py`'s read register (off here, as in main's core).

memcal detects the CPU by the magic in `selfcal_status`. A bitstream built on an older core
reads 0 there, and the host calibrates it as before.

### The fused image with the CPU on the card (08b898d5, 2026-09-29): qualified

The production design with the CPU in its core: main f1f635a + ld-cpu2 (the core above, rd_reg
off), MEM=litedram, MCOLS=4, FAST=1 at 100 MHz. Routed on opentpu (Vivado 2026.1): WNS +0.043 ns
(LiteDRAM sys at 133.33 MHz; the 100 MHz core +0.171), every constraint met. On the card after
a JTAG load and a warm reboot, host tree = this branch at 08b898d5; logs in
`~/otpu-build/fused-sc-08b898d5` on opentpu. FPGA 59.9 C at the start, 64-67 C in the soak. All
**measured**.

| | channel 0 | channel 1 |
|---|---|---|
| the CPU at configuration | 6.27 s (the whole run 12.09 s) | 5.82 s |
| CK phase, common run | +50, 44 steps / 737 ps | -47 (= +65), 72 steps / 1205 ps |
| write latency | 6 on every lane | 0 on every lane |
| bits read off their lane's bitslip | none | none |
| BIST, 2 GiB x 2 x 2 | 0 errors | 0 errors |
| after a new firmware (below) | 6.29 s, CK +50, 44 steps, BIST 0 errors | 5.83 s, CK -47, 72 steps, BIST 0 errors |

- Within a step of the isolated image's run (45 / 73 steps) and equal to 5e5a58ab's host
  calibration (44 / 737 ps, 72 / 1205 ps).
- **A new firmware on the card:** `otpu-memcal selfcal --firmware` with the same source built
  as FW_ID 0xb0b0cafe (5 bytes differ: the ID): 9892 bytes loaded into the 16384 and read back,
  the CPU done 11.9 s later, its mailbox giving the new ID.
- **`otpu-selftest` ALL PASS:** its calib stage found both channels calibrated by the core's CPU
  (0.0 s, no host calibration).
- **`tools/qual/qual.sh fast` (LOAD=0), 37 min, 0 FAIL lines:** token-exact against the ISA
  simulator 12 / 12 (Qwen3-0.6B, LFM2.5-230M, Qwen3.5-0.8B; int8 and fp4 with an int8 head;
  per-position and resident), every Mcycles/token equal to 5e5a58ab's (e.g. fp4 decode 3.30 /
  1.27 / 4.64); int8 decode counters 5.444 / 1.988 / 6.573 Mcycles/token (5e5a58ab: 5.445 /
  1.988 / 6.573); streamed decode (`decode_profile`, fp4) 28.9 / 73.1 / 21.3 tok/s wall; a
  3 min warm soak; `otpu-diag` with the quick memory test ALL PASS; the final selftest ALL PASS.
- A first qual run from a host tree without its `models` link failed every model phase
  (FileNotFoundError) and still printed "0 FAIL lines": qual.sh counts `[FAIL]` lines only.
- Afterwards the card went back to se-cand3 (build 002569bc), whose selftest passed.

### At 133.33 MHz: the production image (main e698dcd, 2026-09-29, evening)

The Vivado tournament's full-effort champion of main e698dcd (the design above, with the core
clock at 133.33 MHz): WNS +0.032 ns, core 133.9 MHz and LiteDRAM sys 134.2 MHz. The bitstream is
written from its routed checkpoint. On the card after a JTAG load and a warm reboot; qualified
with `tools/qual/qual.sh fast` (LOAD=0) and promoted to production (`docs/board.md`, section 5).
All **measured**.

| | channel 0 | channel 1 |
|---|---|---|
| the CPU at configuration | 6.25 s | 5.81 s |
| CK phase, common run | 49, 43 steps / 720 ps | -48, 73 steps / 1222 ps |
| write latency | 6 on every lane | 0 on every lane |
| bits read off their lane's bitslip | none | lane 5 bit 2 at -2, lane 8 bit 6 at +2 |

- **qual.sh fast:** 41 min, 0 FAIL lines (34 PASS).
  - Token-exact 12 / 12.
  - `otpu-selftest` ALL PASS, before and after.
  - The warm diag passes; the FPGA ran at 65-68 C.
- **DRAM efficiency:** while decoding, the card read and wrote 13.9-14.5 GB/s, 82-85% of the
  DDR3-1066 peak (17.1 GB/s). At 100 MHz (5e5a58ab) it was 11.9-12.5 GB/s (70-74%): the 128-byte
  port's cap, 12.8 GB/s at that clock. At 133.33 MHz the port matches the two channels, and
  LiteDRAM runs as close to the peak as the MIG did (se-cand3: 83-85%).
- **Decode:** it takes 13-16% more cycles per token than at 100 MHz, and 13-15% less time. For
  example, Qwen3 4-bit goes from 3.731 to 4.256 Mcycles/token and from 26.8 to 31.3 device
  tok/s. It equals se-cand3's decode within 2.3%.
- **Prefill:** 1.3-2.0x faster than se-cand3.
- **Channel 1's per-bit read offsets** differ again from every earlier build. The CPU's per-bit
  framing found them on its own.

### Follow-up: temperature rescans (not built)

**Why.** The CPU calibrates once, at configuration.
- **Temperatures covered so far:** ldtest3e's 35 min temperature run held its windows (737-753
  and 1222 ps) at 54-55 C. The fused image passed its qual at 60-67 C.
- **Not covered:** a cold start, and a card that heats well past its calibration temperature.
- **The margin:** the CK phase sits in the middle of the common run, about 22 and 36 steps (370
  and 600 ps) from its edges on channels 0 and 1. A drift that moves a lane's edge that far
  would make the channel fail.

**What the hardware already gives.**
- The mailbox keeps each lane's pass map over the scan and the run the CPU chose.
- `selfcal_config` picks the channels.
- A release recalibrates as at reset, dropping and raising that channel's `cal_ready` (STATUS
  CALIB0 / CALIB1).
- A new firmware loads without a bitstream (`otpu-memcal selfcal --firmware`).

**Step 1: host-scheduled rescans, no new firmware.**
- **Trigger:** `memcal` records the XADC temperature with each calibration. When the card has
  moved more than a threshold from it (15 C, say; to be set from measurements), the next
  `Board()` open, or a between-runs hook, schedules a rescan while no program runs (the device
  lock held).
- **The rescan:** hold the CPU, set `selfcal_config` to the channel, release. This takes about 6
  s per channel, during which the channel's `cal_ready` is low and the accelerator must not use
  it. The firmware clears the whole mailbox when it starts, so afterwards the mailbox holds only
  that channel's result; `memcal` keeps the other channel's from its saved record.
- **The data:** the scan's BIST writes the channel's first 64 MiB. Either the host keeps that
  region out of its allocations (the weights and KV cache then live above it), or it saves and
  restores the region around a rescan. Either way, the scanned region is scrubbed again (ECC
  check bits) before use.
- **Logging:** `otpu-memcal` shows the old and new CK phase and run per channel. A run that
  moved or narrowed is the drift record that sets the threshold.

**Step 2: tracking firmware, loaded when step 1 shows drift.**
- **The scan:** a narrow scan around the current CK phase (plus or minus N steps), its BIST on a
  reserved region. CK moves and the channel is re-leveled only if the run's centre moved by
  more than M steps.
- **Selecting it:** through `selfcal_config`'s reserved bits (7:2), so the reset path stays
  `ddrcal`'s.
- **Checks:** the same as for the reset firmware:
  - a `ddrcal` counterpart first, with an identical CSR trace on the simulated PHY;
  - then the Verilator SoC sim;
  - then a card run from a cold start (after power-off) and from a hot card (after a long
    soak), comparing the runs and with BIST during and after.

**Cost (estimate).**
- **Step 1:** host code only (`memcal` rescan, the temperature trigger, the allocator's reserved
  region), plus a card session for the temperature sweep.
- **Step 2:** a firmware mode, and no RTL or bitstream change.

## 11. The card's controller in simulation (2026-09-29, night)

**Why.** The DDR3 bank model behind the adapter in simulation (`otpu_native_mem`'s
`+axi_dram`, `profile.ddr3_plusargs`) matches the card at 100 MHz within 1%. At 133.33 MHz it
is about 10% optimistic. At 100 MHz the core's 128-byte port caps the path at 12.8 GB/s, below
what the controller can do, so the controller's own losses do not show. At 133.33 MHz the port
matches the two channels (17.07 GB/s), and they do.

**The co-simulation.** The simulation now runs LiteDRAM's own controller instead of a model of
it. Setting `OTPU_LDC=1066` (rtlsim's `ldc=`; `tools/perf_qwen.py --ddr 1066 --ldc`) puts the
card's channels behind the adapter (`otpu_top` AXI = 2):

- **The controller:** `tools/litedram/gen_ldc.py` generates one channel of the production
  controller as Verilog (`sim/verilator/otpu_ldc_ch.v`). This is LiteDRAM's own RTL at
  `core.json`'s commits: the bank machines, the multiplexer, the refresher, the crossbar with
  the idle BIST port, and `LiteDRAMNativePortECC`. It has the core's settings: MT41K256M8 with
  tRFC 160 ns, 1:4, the core's `ControllerSettings` (`tools/litedram/ctl_settings.py`:
  LiteDRAM's defaults but, since memeff, refresh postponing 2 and read / write times 256 / 128),
  and WL7DDRPHY's latencies (read 8, write 1). The PHY is a DFI stub. It is DDR3-1066 only:
  the board's DDR3 never runs faster, the rate its HR banks are specified for. Since ld-2port
  it has the core's two user ports per channel (below); the checks in this section's first
  table were made with one.
- **The memory:** `sim/verilator/otpu_ldc_mem.sv` puts each channel behind the board's bridge
  (`otpu_mem_ch`, with its clock crossing and read-modify-write) in the controller clock. The
  data is kept in the model and moves in the controller's command order.
- **The clocks:** the core runs at `OTPU_LDC_MHZ` (`rtlsim.ldc_plusargs`) against the
  controller's 133.33 MHz.
- **The testbench:** `tb_top` now changes its signals on the falling clock edge, so they cannot
  race the memory model's own clock. Before the dump it waits 256 cycles for the last writes to
  land.

**Checks, measured** (Verilator 5.047 on omarchy; the card figures are sections 7 and 10):

| | simulation | card |
|---|---|---|
| sequential reads, one channel, the controller alone (the BIST's pattern; memeff: 90.4%, card 90.4%) | 90.9% of peak | 91.0% |
| sequential writes (memeff: 89.9%, card 89.8%) | 90.1% | 90.1% |
| Qwen3 4-bit decode, pos 544, int8 head: Mcycles/token at 100 / 133.33 MHz | 3.734 / 4.218 | 3.731 / 4.256 (+0.1 / -0.9%) |
| LFM2 4-bit, the same | 1.348 / 1.541 | 1.357 / 1.554 (-0.7 / -0.8%) |
| Qwen3.5 4-bit, the same | 4.636 / 5.360 | 4.673 / 5.435 (-0.8 / -1.4%) |

The decode figures are the resident decode program at the card's qual operating point (KV
capacity 2048, position 544, 4-bit layers, int8 LM head), the whole model's token built from
layer prefixes (`tools/perf_ddr.py`). The card's are its qual runs (5e5a58ab at 100 MHz, the
production e698dcd at 133.33 MHz). The old bank model gave 3.831 M for Qwen3 at 133.33 MHz,
10% under the card.

- **Bit-exact:** `tests/test_rtl.py::test_ldc_memory_path` runs the hazard, random and DMA
  programs on the co-simulated channels, with the core as fast as, slower than, and faster than
  the controller. All match the ISA simulator bit for bit.
- **BIST check:** `test_ldc_sequential_is_the_card_bist` checks the sequential-stream figures
  against the card's BIST numbers.
- **The production RTL:** main's RTL and the grid's tree (ld-qual b719cd3) give the same Qwen3
  4-bit decode at 133.33 MHz to within 3 cycles (4,218,386 against 4,218,389).

### Where 133.33 MHz loses, measured

**Method.** Qwen3 4-bit's whole decode step (28 layers, pos 544) was traced as the adapter
issues it (`otpu_native_mem +nat_trace`). One channel's trace has 3.45M reads and 19.5K writes.
It was then replayed open loop through the controller, which takes each command as soon as it
can (`sim/verilator/tb_ldc_replay.sv`). The replay has no compute stalls, so it bounds what the
memory alone can do. Its production figure, 4.15M controller cycles, is 2.5% under the card's
token of 4.256M cycles.

**Results** (controller cycles for the channel's trace; the reference row is the production
controller):

| variant | cycles | beats / cycle | change |
|---|---|---|---|
| **the production controller** | 4,150,608 | 0.837 | - |
| the crossbar without its bank lock (a bound: read order is not kept) | 3,694,870 | 0.940 | -11.0% |
| no refresh | 4,070,532 | 0.854 | -1.9% |
| command buffer depth 16 (instead of 8) | the same to the cycle (1- and 2-layer traces) | | 0.0% |
| bank index XORed with the row's low bits | 4,147,431 | 0.838 | -0.1% |
| the adapter's A runs of 32 beats (instead of 8) | 4,039,449 | 0.860 | -2.7% |
| A runs of 64 beats | 4,012,619 | 0.866 | -3.3% |
| **two user ports**, split on beat bit 7 (the bank's low bit), queues of 16 | 3,740,986 | 0.929 | **-9.9%** |
| two user ports, queues of 32 | 3,717,703 | 0.935 | -10.4% |
| two user ports, queues of 16, A runs of 32 | 3,691,501 | 0.941 | -11.1% |

LFM2 4-bit (16 layers) shows the same pattern:
- without the lock: -9.8%;
- two ports: -9.0% (queues of 32: -9.7%);
- A runs of 64 beats: -3.4%;
- no refresh: -1.9%;
- the XOR hash: -0.1%.

**The breakdown** (the production controller moves 0.837 beats per cycle; 1.0 is the peak):

- **The crossbar's bank lock: about 11%.** A master may have commands in only one bank at a
  time. A bank machine stays locked to the master while its command buffer or its lookahead
  holds one of the master's commands, so the next command to another bank waits until the
  previous bank's last column command has gone out. Each bank switch then leaves about 6 idle
  column slots: the next bank's activate and tRCD, and the pipeline from the crossbar through
  the bank machine to the multiplexer.

  Decode switches bank about every 66 commands. Its streams are 8 KiB rows (128 beats) per bank
  under ROW_BANK_COLUMN, and the A runs, B streams and writes interleave. The single-port
  sequential stream switches only every 128 beats, which is why the BIST loses only about 4
  points to the lock (section 4).
- **Refresh: 1.9%.** tRFC is 23 cycles every tREFI of 1042 cycles.
- **The rest: about 3.4%.** Row misses within a bank (tRP + tRCD), read/write turnarounds, and
  the pipeline's fill.

### Two user ports per channel in the co-simulation, measured

LiteDRAM's controller was regenerated with two user ports per channel (`gen_ldc.py --ports 2`,
each with its ECC frontend, and the idle BIST port). A behavioural split was put between the
unchanged bridge and the ports: each command goes, in order, to the port of its bank's parity
(beat bit 7), into that port's queue of 16, so neither port waits on the other. Reads go back
to the bridge in command order. The split is bit-exact on the hazard, random and DMA programs
at 100 to 250 MHz. Decode at 133.33 MHz, the fp4 trio:

| | one port | two ports | change | tokens/s | DRAM GB/s (% of peak) |
|---|---|---|---|---|---|
| Qwen3 | 4.218 M | 3.734 M | -11.5% | 31.6 -> 35.7 | 14.1 -> 15.9 (82 -> 93%) |
| LFM2 | 1.541 M | 1.417 M | -8.0% | 86.5 -> 94.1 | 14.2 -> 15.4 (83 -> 90%) |
| Qwen3.5 | 5.360 M | 5.042 M | -5.9% | 24.9 -> 26.4 | 14.3 -> 15.2 (84 -> 89%) |

- **Closed loop beats the replay.** The replay bounded Qwen3 at -10.4% (queues of 32). The
  full model does better because the adapter keeps issuing while a port waits.
- **Qwen3.5 gains least:** more of its token is the DeltaNet state's traffic and compute.
- **The per-port queue matters.** Qwen3 at 133.33 MHz, by the queue's depth:

  | depth | 2 | 4 | 8 | 16 | 32 | 64 |
  |---|---|---|---|---|---|---|
  | Mcycles/token | 4.058 | 4.075 | 3.875 | 3.734 | 3.686 | 3.759 |
  | change | -3.8% | -3.4% | -8.1% | -11.5% | -12.6% | -10.9% |

  - **Too short:** the queue fills with one bank's commands while that port waits on its lock,
    and the arbiter behind it stops.
  - **Too deep (64):** a queue runs further ahead of the other port, and the reads that come
    back out of order hold the adapter's read window longer (**estimate**).
  - **32 is the best measured.**

### Two user ports per channel: the build (ld-2port, 2026-09-30)

The split is now in the board's RTL. It takes a core regeneration plus bridge RTL; nothing else
in the accelerator changes.

- **The core (a regeneration: `gen_core.py`, `check_core.sh --update`).** Each channel gets a
  second native user port, `c0b_*` / `c1b_*`. It has no `ready` pin: `c0_ready` / `c1_ready` are
  the channels'.
  - **The crossbar:** each bank's arbiter now has three masters: the even banks' port, the odd
    banks' port, and the BIST, which moves from master 1 to master 2.
  - **The ECC (`tools/litedram/ecc_ports.py`):** each port keeps LiteDRAM's encoder and its
    register, but one decoder serves both ports' reads. The crossbar gives every master the
    controller's read bus, and only the valids are per master. A register per port follows the
    decoder, as in `LiteDRAMNativePortECC`. So the encoding and the latencies are the stock
    frontend's, and one CSR block per channel (`ecc`, `ecc1`) counts both ports' errors.
    - `tools/litedram/ecc_ports_check.py` runs it against two stock frontends in migen's
      simulator, cycle by cycle, on the same random traffic. The read words carry 0, 1 or 2
      flipped bits: SEC must correct, DED must be counted, and the counters must match. It
      passes 600 cycles (303 read beats, 479 words corrected, 477 uncorrectable), and it fails
      when the ports get the undecoded bus.
    - `gen_ldc.py` uses the same module, so the co-simulation has it too. Qwen3's decode there is
      the same to the cycle as with two stock frontends (3,693,026).
  - **The CSR map:** main's, byte for byte (`csr.csv` differs only in its date).
  - **Unchanged:** the XDC, the calibration CPU's firmware ROM (`otpu_litedram_mem.init`) and
    `sdram_init.py` are byte for byte the same.
  - `check_core.sh` reproduces main's core before the change and this one after it.
- **The bridge (RTL only: `otpu_mem_ch`, `otpu_afifo`; the wiring in `otpu_native_sys` and
  `otpu_fpga_top_ld`).**
  - **The split:** each command goes to the port of its beat's bit 7.
  - **Per port:** an output command queue of 32 behind a register, a write-data FIFO of 32 and a
    tag FIFO. A port whose bank waits holds back only its own commands.
  - **In-order return:** each read takes the next slot of its master's read-data FIFO when it
    issues; the read credits keep that slot free. Its data is written into the slot when its port
    returns it, and the FIFO (`otpu_afifo` OOO) passes the slots in order. It passes a slot the
    cycle after the slot is written, using the slots' flags alone, so the returning beat does not
    reach the credit counts in its own cycle.
  - **Unchanged:** the read-modify-write, the holds and `n_wdone`. They work per beat, and a beat
    is always on the same port.
  - **One read bus.** The core's two ports of a channel carry the same read data every cycle:
    one decoder feeds both ports' registers, and each register loads every cycle because
    `rdata_ready` is tied high. So the bridge reads port 0's data whichever port returns the beat,
    and Vivado prunes port 1's registers. Simulation checks this contract: `otpu_mem_ch` errors if
    port 1's data ever differs, and `otpu_ldc_mem` fails if the generated controller's two buses
    differ. The read-modify-write merge selects on `rm_busy`, a flip-flop, not on the returning
    beat's tag.
  - **The XDMA side's command and write-data FIFOs have a registered `wready`** (`otpu_afifo`
    RWR), so XDMA's write enables start from flip-flops rather than from the hold's synchronizer.
    The flag is the room left after this cycle's write, which is at most a cycle more
    conservative.
  - **Constraints:** `otpu_mem_ch.tcl`'s CDC waivers name the per-port endpoints
    (`g_port[*].u_oq`, `oc`, `oc_v`, the FIFOs' write pointers) and the new counters. The max
    delays are unchanged.
- **Simulation:**
  - `otpu_ldn_model` is two ports. It checks that each beat is on its bit-7 port and that at most
    one read beat per cycle comes back across the ports.
  - `gen_ldc.py` defaults to `--ports 2`, so the co-simulation has the new core's controller.
  - `tb_ldc_replay` drives the first port and leaves the second idle, as the one-port core did.
- **Checks in the RTL, in simulation:**
  - read data from both ports in one cycle;
  - read data without a tag;
  - a slot written twice or overrun;
  - no room for the merged write;
  - a full tag FIFO;
  - a hold that ends with a command of its master still queued.
- **The check on the board: `n_err`.** Read data from both ports in one cycle, or read data
  without a tag, sets a sticky bit in the controller's clock. It crosses into the core's clock
  and shows as STATUS bit 4 (AXI_ERR), which the host reports as a memory-path error. Only the
  controller's reset clears it. `memch_test`'s `doublebeat` scenario checks it: the model
  controller returns one beat on both ports.
- **A write is visible to both masters once counted.** XDMA's B and the accelerator's `n_wdone`
  count writes the controller has taken. A later read of the address goes to the write's port
  (its bit 7), and each port passes its commands in order, so the read follows the write in its
  bank's queue. `memch_test`'s `pubstall` scenario checks it: mostly shared operations, one
  master reading a beat as soon as the other's write is counted, while the controller stalls and
  the two ports' queues run apart.
- **`memch_test` mutations:** fourteen new ones (32 in all), all caught.
  - the second port's writes not counted;
  - reads without credits;
  - a slot freed on return;
  - every command on port 0;
  - the split on bit 8;
  - every read in slot 0;
  - every tag from port 0's FIFO;
  - a command bypassing a queue that holds older ones;
  - write data pushed into the other port's FIFO;
  - each port given the other port's write-data head;
  - the in-order release passing a slot before it is written (`otpu_afifo`);
  - XDMA's B sent once its beats are in the bridge, before the controller takes them;
  - the XDMA FIFOs' registered `wready` not counting this cycle's write;
  - `n_err` not set by a double beat.

**Decode, co-simulated** (the RTL above behind the two-port controller; fp4 layers, int8 head,
pos 544, 133.33 MHz, DDR3-1066; GB/s: bytes read and written per token over its time):

| | one port | two ports | change | tokens/s | DRAM GB/s (% of peak) |
|---|---|---|---|---|---|
| Qwen3 | 4.218 M | 3.693 M | -12.5% | 31.6 -> 36.1 | 14.06 -> 16.06 (82.4 -> 94.1%) |
| LFM2 | 1.541 M | 1.415 M | -8.2% | 86.5 -> 94.2 | 14.20 -> 15.46 (83.2 -> 90.6%) |
| Qwen3.5 | 5.360 M | 5.001 M | -6.7% | 24.9 -> 26.7 | 14.32 -> 15.35 (83.9 -> 89.9%) |

The bytes per token are the same. The RTL does a little better than the behavioural split above
and matches its queue-of-32 point.

**Correction (2026-09-30): the two-port column above is 5% short.** It was built from layer
prefixes (`tools/perf_ddr.py`'s first layer plus the second's increment for each further layer),
and on two ports the second layer costs less than the layers after it ("What is left in the
controller", below). The whole model, simulated: Qwen3 3.903 M (-7.5% from one port), LFM2
1.417 M (-8.1%), Qwen3.5 4.952 M (-7.6%). The card measured -7.4%, -7.9% and -7.9% (the fused
build c2830d6's qual). The one-port column holds: its layers all cost the same.

**Area and timing of the bridge** (per channel; Vivado 2026.1, `tools/memch_ooc.tcl`, routed out
of context at 7.5 / 7.5 / 8.0 ns for the core, the controller and XDMA):

| | LUTs (as RAM) | FFs | WNS uclk | WNS clk | clk -> uclk max delay slack |
|---|---|---|---|---|---|
| main (one port) | 4,256 (2,256) | 1,758 | +0.111 | +1.148 | +3.29 |
| ld-2port | 5,607 (2,680) | 1,929 | +0.081 | +0.594 | +1.64 |

- **The worst uclk path is the same in both:** the write-data FIFO's read pointer through its
  LUT RAM to `c_wdata_data`. That port carries the OOC's 30% output delay. In the build it goes
  into the ECC frontend's register.
- **Both builds are CDC-clean:** only Info rows after the waivers, the bus skews met, and every
  XDC query matched.
- **The whole board:** the bridges add 2,702 LUTs and 342 FFs for both channels.

**Area and timing of the core with the bridges** (Vivado 2026.1, `ldimpl`: the core and both
bridges placed and routed together on the board's pins, with random masters, using the production
build's strategy; 7.5 ns for the core and the controller, 8 ns for XDMA). WNS is within each
clock: the wrapper's unreplicated LiteX reset and the bridges' async crossings are left out
(`memch_ooc.tcl` times the crossings). The wrapper drives the core's CSR bus and stretches the
placement, so the numbers compare the runs rather than predict the board's.

| | main (one port) | ld-2port, first cut | ld-2port |
|---|---|---|---|
| core LUTs / FFs | 19,808 / 15,645 | 21,143 / 17,895 | 20,533 / 16,863 |
| bridges' LUTs (as RAM) / FFs | 9,819 (4,488) / 3,567 | 12,228 (5,332) / 3,923 | 12,154 (5,332) / 3,922 |
| slices | 9,898 | 11,194 | 11,018 |
| sys WNS, into or out of the bridges | -0.785 | -2.079 | -1.389 |
| sys WNS, all (the CSR bus in every run) | -2.500 | -2.479 | -3.274 |
| XDMA WNS | -0.140 | -0.951 | +0.125 |
| core clock WNS (the wrapper's output fold) | +1.322 | +1.036 | +1.675 |

- **The core's block RAMs are unchanged:** 4 RAMB36 and 3 RAMB18, the calibration CPU's.
- **Two stock frontends per channel cost more.** With them, the core grew by 2,467 LUTs, in a run
  with the default strategy.
- **The first cut's new paths, and their fixes:**
  - The read return reached the credit counts in its own cycle: the core's read valid -> the
    port's tag -> the FIFO's release -> `x_pend`, 6 levels. The release now waits a cycle.
  - XDMA's write enables followed the hold's synchronizer: `x_hs2` -> `wready` -> `u_xd`'s write
    enables, 4 levels. `wready` is now a flip-flop.
  - A 512-bit port mux sat on the read valid: the valid -> the mux -> the RMW merge -> `u_of`.
    The bridge now reads the one read bus, and Vivado prunes port 1's registers in the core
    (-1,033 FFs).
  - The arbiter's `go` needed the chosen head's port: `u_ad`'s rvalid -> the pick -> the heads'
    mux (fanout 680) -> `go` -> `a_pend`, 11 levels. Each master's room is now found from its own
    head, alongside the pick.
- **The worst bridge path left** is the write data: `u_ad`'s head -> the heads' mux -> the RMW
  merge -> port 1's write-data FIFO, 3 levels, mostly route (a FIFO per port). The one-port
  core's worst, at -0.785, was the hold -> its one FIFO.
- **Decode moves by under 0.3%:** Qwen3 3,693,026 -> 3,690,662 cycles/token, Qwen3.5
  5,001,431 -> 4,990,251 (co-simulated, fp4, 133.33 MHz, DDR3-1066).

### What would gain, ranked

Expected gains are for decode at DDR3-1066 and 133.33 MHz, the board's only rate.

1. **Two native ports per channel, split on beat bit 7: +6 to +13% decode tokens/s, measured in
   the co-simulation above.** Risk: medium. Built as ld-2port (above): +7 to +14% on the RTL.
   - **How it works.** Alternate banks go to different crossbar ports, so one port's lock no
     longer holds the other's commands.
   - **Core:** `gen_core.py` adds a second user port per channel, with its own ECC frontend.
     This needs a core regeneration.
   - **Bridge:** `otpu_mem_ch` gets two output ports, per-port command queues (32 deep) and
     write-data FIFOs, and in-order read return. The return can use the read-data FIFOs the
     bridge already has: each read reserves its slot at issue (the read credits), its port writes
     it there, and the FIFO passes slots in order. The two ports share the channel's data bus, so
     at most one beat comes back per cycle. In the replay, a separate reorder buffer would have
     held up to 75-126 beats per channel.
   - **Hazards stay within one port.** A beat always maps to the same port, and each port keeps
     its commands in order. `n_wdone` counts both ports' write handshakes. The XDMA path and
     the read-modify-write read keep the same rule.
   - **Costs:** one ECC frontend and crossbar port per channel in the core. In the bridge, a
     second write-data FIFO and command queue per channel. The bridge's and the controller's
     timing at 133.33 MHz also need checking: LiteDRAM's sys domain has +0.048 ns.
2. **The adapter's A runs from 8 to 32 beats: +2.7% alone, about +1.2 points on top of item
   1.** Risk: low. Done in memeff (below): 0.9 to 1.3% fewer cycles per token on two ports.
   - **Change:** `otpu_native_dram`'s APF, a wider run counter, and an A data FIFO of at least
     2 × APF - 1 = 63 beats (4 KiB per channel, block RAM).
   - **Caveat:** the trace rewrite assumed the merged runs were all used, because they were
     consecutive runs of one stream. A stream that ends early would fetch beats it does not use.
3. **No gain:**
   - refresh postponing: all of refresh is 1.9% (in the replay; on the whole model with two
     ports, postponing 2 is -0.2 to -0.4% on top of the other fixes, taken in memeff, below);
   - a deeper command buffer: 0.0%;
   - XOR bank hashing: 0.1%.
4. **A higher-risk alternative to item 1:** patch LiteDRAM's crossbar so the next bank's
   activate can go out while the previous bank still has column commands pending. This is
   bounded by the no-lock row at -11%. Read data would then return in column-command order, not
   command order, so the adapter would need the in-order return anyway, and we would carry a
   LiteDRAM fork.

### What is left in the controller, and three fixes (memeff, 2026-09-30)

**Simulate the whole model, not prefixes.** On two ports, the second layer's increment is not a
layer's cost. Qwen3 4-bit decode at 133.33 MHz (the fused build c2830d6's RTL), by the number of
layers simulated:

| layers | 1 | 2 | 3 | 4 | 8 | 16 | 28 |
|---|---|---|---|---|---|---|---|
| Mcycles/token | 1.424 | 1.508 | 1.601 | 1.699 | 2.067 | 2.801 | 3.903 |
| cycles per layer since the previous point | - | 83.9k | 93.0k | 97.7k | 92.0k | 91.8k | 91.8k |

- **One port:** every layer from the second on costs 99.4k to 99.6k, so the prefix build held
  (4.218 M against the whole model's 4.219 M).
- **Two ports:** the prefix build gave 3.691 M, 5.4% short of the whole model.
- **`tools/perf_ddr.py` now simulates the whole model** (2-3 minutes a point on the Mac).
  `--prefixes` keeps the old build.

The whole model against the card (c2830d6's qual), Mcycles/token:

| | co-simulated | card |
|---|---|---|
| Qwen3 4-bit | 3.903 | 3.939 (-0.9%) |
| LFM2 4-bit | 1.417 | 1.431 (-1.0%) |
| Qwen3.5 4-bit | 4.952 | 5.006 (-1.1%) |
| Qwen3 8-bit | 5.698 | 5.735 (-0.6%) |
| LFM2 8-bit | 2.083 | 2.096 (-0.6%) |

**Where the cycles go.** The runs were c2830d6, the card's operating point, 4-bit layers. The
figures are per channel, in % of the controller's cycles. They come from counters added to the
co-simulation for this study (not in the tree). Each cycle counts under the first of these that
holds:
- a column command on the DFI (data);
- refresh;
- a bank with a column command that was not issued (the multiplexer);
- a bank opening or closing a row;
- commands at the ports that no bank holds (the crossbar);
- nothing to do.

| | Qwen3 | LFM2 | Qwen3.5 |
|---|---|---|---|
| column commands (data) | 89.0 | 90.4 | 90.8 |
| refresh | 2.9 | 2.9 | 2.9 |
| rows opening and closing | 2.5 | 2.2 | 2.7 |
| the multiplexer: read / write turnarounds | 0.7 | 0.4 | 1.7 |
| the multiplexer: its chooser's grant on a bank with no column command | 0.6 | 0.5 | 0.8 |
| the crossbar (its lock and pipeline) | 0.4 | 0.4 | 0.5 |
| nothing asked | 3.7 | 3.2 | 0.6 |

- **The crossbar's lock** cost 11% on one port. The two ports took it under 0.5%.
- **The bridge's read credits and queues** limit under 0.1% of the cycles.
- **The idle time is the core's:** about 700 runs per token of 64 to over 256 cycles (norms,
  attention's tails). It is a lever for the program, not for the memory.

**The variants, measured** (the whole model, Qwen3 4-bit, cycles per token against c2830d6; the
controller variants from `gen_ldc.py`'s settings or small LiteDRAM subclasses):

| variant | change |
|---|---|
| A runs (APF) of 16 beats | -0.55% |
| A runs of 32 | -1.25% |
| A runs of 64 | -1.22% (0.26% more beats read) |
| no refresh (a bound) | -2.79% |
| refresh postponing 8 | -0.59% |
| with A runs of 32 and read / write times 256 / 128: postponing 1 / 2 / 4 / 8 | -1.29 / -1.50 / -1.59 / -1.50% |
| a refresher that refreshes when the controller is idle (8 postponed / 8 pulled in, or 4 / 4) | -0.17 to -0.38% |
| the same, a forced refresh draining all owed | -0.59 to -0.61% |
| read / write times 128 / 64 (Qwen3.5: 256 / 128 -0.59%, 512 / 256 -0.72%) | -0.03% |
| command buffer 16 | +0.13% |
| the multiplexer's chooser granting a valid request in the same cycle | -0.63% |

- **Refresh is structural.** Decode's idle time comes in a few clusters per layer. DDR3 lets
  only 8 refreshes be postponed or pulled in, so most of a layer's 88 refreshes land in its
  weight streams. Refreshing when idle gains nothing over LiteDRAM's postponing, which saves one
  wait for the banks and one row reopening per burst of postponed refreshes.
- **Postponing 2, not 8.** A burst of 8 refreshes blocks a channel for about 210 cycles. On
  DRAM-bound decode that costs nothing more than 2 or 4 (Qwen3.5 4-bit with the other two
  fixes: 4.858 / 4.838 / 4.844 / 4.837 M cycles at 1 / 2 / 4 / 8). But the core's buffers do not
  hide it when the MXU is the limit: perf_qwen's 2-layer Qwen3 4-bit run (pos 9, 99% of its
  roofline) takes 1,479,646 / 1,479,908 / 1,480,313 / 1,495,742 cycles at 1 / 2 / 4 / 8.
- **The chooser, and a shorter read-to-write turnaround,** were left for later: measured on
  top of these fixes in "The chooser and the turnarounds" below, and parked.

**The three fixes (memeff):**
1. **APF 8 -> 32** (`otpu_native_dram`). The run counters and the A order's drop field go from
   3 bits to log2(APF). The A read FIFO goes from 32 to 64 beats (at least 2 APF - 1).
2. **Refresh postponing 2.**
3. **Read / write times 256 / 128.**

Fixes 2 and 3 are `tools/litedram/ctl_settings.py`, which `gen_core.py` and `gen_ldc.py` both
use.
- **What changes in the core's Verilog:** per channel, only the refresher's two counters and the
  multiplexer's two timers.
- **What stays byte for byte** (`check_core.sh`): the XDC, the firmware ROM, `csr.csv` and
  `sdram_init.py`.
- **The co-simulated controller** is regenerated with the same settings. With postponing 8 its
  longest time between two refreshes was 8 tREFI (DDR3 allows 9); with 2 it is about 2 (the
  refresher waits 2 tREFI, then refreshes twice).

**Decode, measured** (the whole model, `tools/perf_ddr.py`, 133.33 MHz, DDR3-1066, the card's
operating point: pos 544, KV capacity 2048, int8 LM head; main f311476 against memeff):

| | main, cycles/token | memeff | change | tokens/s |
|---|---|---|---|---|
| Qwen3 4-bit | 3,902,777 | 3,844,073 | -1.50% | 34.2 -> 34.7 (+1.53%) |
| LFM2 4-bit | 1,416,696 | 1,398,641 | -1.27% | 94.1 -> 95.3 (+1.29%) |
| Qwen3.5 4-bit | 4,951,910 | 4,837,936 | -2.30% | 26.9 -> 27.6 (+2.36%) |
| Qwen3 8-bit | 5,697,906 | 5,617,507 | -1.41% | 23.4 -> 23.7 (+1.43%) |
| LFM2 8-bit | 2,082,922 | 2,054,882 | -1.35% | 64.0 -> 64.9 (+1.36%) |
| Qwen3.5 8-bit | 6,982,409 | 6,836,578 | -2.09% | 19.1 -> 19.5 (+2.13%) |

- **An MXU-bound run hardly moves:** perf_qwen's 2-layer Qwen3 4-bit run (`--wformat fp4 --ddr
  1066 --mhz 133.33 --ldc`, systolic, MCOLS 4) goes 1,478,546 -> 1,479,908 cycles (+0.09%).
- **Row changes** fall by about 30% (Qwen3 4-bit: 128,304 -> 89,923).
- **The longer A runs read a little more:** +3,584 beats per token on Qwen3 4-bit (0.05%).
- **Sequential streams on the controller alone** (`tb_ldc_replay2`, 256K beats):
  - **Split over the two ports, as the bridge splits them** (the path of decode and XDMA):
    reads 96.7% -> 97.1% of peak, writes 95.6% -> 96.5%.
  - **On one port** (the BIST's pattern, `tb_ldc_replay`): reads 90.9% -> 90.4%, writes
    90.1% -> 89.9%. Behind one port's crossbar lock, a single refresh costs about 13 cycles of
    throughput, since part of it overlaps the lock's bank-change gaps. In a burst, each costs
    more (about 25 cycles in bursts of 8).
  - So the card's BIST figure drops by about half a point. The card measured it (below):
    `test_ldc_sequential_is_the_card_bist` expects 90.4% / 89.8%.

**On the card** (build `f8c6c950`, `deploy_memeff_f8c6c950`: main 3ff7cfd + memeff, 133.33 MHz,
full effort, opentpu, 2026-10-01). Timing: WNS +0.104 ns, WHS +0.019 ns (build B, 79c5707a:
+0.017 / +0.016). The A read FIFO is 2 x 171 RAM64M; its worst path is +0.350 ns, the tail
register's fanout to the write addresses (0 levels, 97% route). Slice LUTs 173,379 (B 172,837),
FF 133,503 (134,197), slices 75.0% (75.3%); `otpu_native_dram` +1,268 LUTs (APF, AD 64). The
qualification (`qual.sh fast`) passes: 0 FAIL lines, 42 PASS. Both channels calibrate, with write
windows as wide as B's (770 / 1222 ps against 737 / 1222; the PHY is the same). The BIST (2 GiB,
2 passes, both channels): reads 90.4%, writes 89.8% of peak, no errors (the simulation: 90.4 /
89.9; B: 91.0 / 90.1). The scrub after it: no ECC errors.

Decode on the card, Mcycles/token (`qual.sh`'s prefill + decode counters, 64 tokens; the DRAM
bytes per token are B's):

| | build B | memeff | change | the simulation's |
|---|---|---|---|---|
| Qwen3 4-bit | 3.940 | 3.877 | -1.60% | -1.50% |
| LFM2 4-bit | 1.431 | 1.413 | -1.26% | -1.27% |
| Qwen3.5 4-bit | 5.006 | 4.895 | -2.22% | -2.30% |
| Qwen3 8-bit | 5.735 | 5.652 | -1.45% | -1.41% |
| LFM2 8-bit | 2.096 | 2.069 | -1.29% | -1.35% |
| Qwen3.5 8-bit | 7.036 | 6.884 | -2.16% | -2.09% |

- **The simulation's prediction holds** within 0.1 points on every model.
- **Bandwidth while decoding** goes from 88-92% of the 17.1 GB/s peak to 90-94%.
- **The streamed-logits decode** (`decode_profile`, 96 steps) gives the same picture: Qwen3 4-bit
  36.92 -> 37.47 tokens/s, LFM2 97.53 -> 98.84, Qwen3.5 26.76 -> 27.35. The on-card decode loop
  gives Qwen3 37.37 -> 37.89 and LFM2 97.65 -> 98.89.

### The chooser and the turnarounds (2026-10-01; in the core 2026-10-02, with the next build)

The two multiplexer items left after memeff, with a third of the same kind, as options of
`tools/litedram/fastmux.py`: a copy of LiteDRAM's multiplexer that `gen_core.py` and `gen_ldc.py`
build through. `ctl_settings.py`'s `MULTIPLEXER` is `FASTMUX`, all three on, since 2026-10-02
(the committed core and `otpu_ldc_ch.v`; their first build is the next full build). With all
three off the core and the co-simulated controller are LiteDRAM's, byte for byte: `gen_ldc.py
--rtw none --no-same-cycle --no-direct-wtr` writes the model main had before.
- **`same_cycle`: the choosers grant in the cycle a request is valid.** LiteDRAM's
  `_CommandChooser` keeps its round-robin grant in a register and moves it only when the granted
  request is taken or is not valid. So a grant on a bank with nothing to issue in the current state
  (a read while writing, a bank waiting on its own timers) costs a cycle. Here the grant is the
  first valid request from a round-robin pointer on: a priority encoder in front of the command
  mux, in LiteDRAM's sys domain.
- **`rtw`: the read-to-write turnaround as a command spacing.** LiteDRAM's RTW state lasts
  read_latency - 1 = 7 cycles and issues nothing, so its first write goes 9 cycles after the last
  read (8 when read_time runs out). With `rtw` the multiplexer goes from READ straight to WRITE,
  and a counter from the last read (a `tXXDController`, like tCCD's) holds the writes; activates
  and precharges go on meanwhile. `rtw 3` makes it 3 cycles.
- **`direct_wtr`: the write-to-read turnaround the same way.** The same tWTR counter (LiteDRAM's:
  tWTR + CWL's cycles + tCCD = 2 + 2 + 1) holds the reads in READ, so the WTR state's extra
  transition goes: 6 cycles to 5.

**The command spacing, checked** (`tb_ldc_replay2`'s spacing line: the DFI's column commands, both
ports, a trace of 60,000 commands in short read and write bursts over 4 rows of every bank,
`gen_ldc.py`'s options): from a read to the next write at least 9 cycles in LiteDRAM's and 3 with
`rtw 3`; from a write to the next read 6, and 5 with `direct_wtr`. That trace runs in 252,595
cycles in LiteDRAM's, 213,550 with `rtw 3`, 210,956 with `same_cycle`, and 171,609 with all
three. Bit for bit: `test_rtl.py -k ldc`'s three memory paths pass with all three.

**The PHY's side of the read-to-write turnaround** (DDR3-1066, CL 7, CWL 6, BL8; the reads and
writes go on phase 2 of their sys cycle):
- **DDR3** needs RL + tCCD + 2 tCK - WL = 7 + 4 + 2 - 6 = 7 tCK from a read to a write: 2 sys
  cycles.
- **The PHY needs more.** WL7DDRPHY (A7DDRPHY's write control) switches its DQ and DQS output
  enables a whole sys cycle at a time (`OSERDESE2` TRISTATE_WIDTH 1, TQ in BUF mode): a preamble
  cycle, the data's cycle and a postamble cycle. So it drives the bus 4 tCK before the write data
  where DDR3 asks for 1, and the data lands b tCK into its cycle, b being the write latency
  calibration's bitslip. On the card b is 0 or 1 tCK (memcal's write latency 0 / 6). The PHY keeps
  CK within half a tCK of sys, so b stays in {0, 1} on this board.
- **Counted in tCK from the last read command R,** with the write command at W = R + 4k (k sys
  cycles):
  - The read burst holds the bus from R + CL - 1 (the preamble) to R + CL + 4 + 0.5 (the
    postamble) = R + 11.5, plus tDQSCK (0.3 ns).
  - The PHY starts to drive at W + CWL - 4 - b = W + 2 - b.
  - So no overlap needs 4k >= 9.7 + b, and about 0.5 tCK more at the FPGA's pins (the round
    trip).
- **k = 2 is not safe:** the PHY would start to drive while the read's last beats are still on the
  bus.
- **k = 3 is safe for b <= 1,** with at least 0.8 tCK (1.5 ns) to spare at the FPGA's pins, plus
  the output enable's own path delay. A b of 2 or 3 tCK would need k = 4.
- **ODT is static** (DFII's control bit held high, Rtt_Nom and Rtt_WR at RZQ/4, one rank): no ODT
  pin switches between a read and a write. The DRAM's own switch to Rtt_WR comes CWL - 2 = 4 tCK
  after the write, at R + 16, after the read's postamble.
- **The write-to-read count is LiteDRAM's** (5 cycles = 20 tCK; DDR3 needs CWL + 4 + tWTR = 14),
  and the write's postamble cycle is long gone when the read's preamble comes.

**Decode, measured** (the whole model, `tools/perf_ddr.py`, the card's operating point; cycles per
token against main ff186b1, memeff):

| | main | `same_cycle` | `rtw 3` | `rtw 3`, `direct_wtr` | all three |
|---|---|---|---|---|---|
| Qwen3 4-bit | 3,844,073 | -0.33% | -0.35% | -0.42% | -0.74% |
| LFM2 4-bit | 1,398,641 | -0.26% | -0.14% | -0.17% | -0.35% |
| Qwen3.5 4-bit | 4,837,936 | -0.35% | -0.47% | -0.52% | -0.86% |
| Qwen3 8-bit | 5,617,507 | -0.27% | -0.27% | -0.34% | -0.50% |
| LFM2 8-bit | 2,054,882 | -0.20% | -0.14% | -0.15% | -0.31% |
| Qwen3.5 8-bit | 6,836,578 | -0.45% | -0.57% | -0.69% | -0.99% |

- **Why it is small:** decode reads in long runs. The turnarounds and the chooser's empty grants
  were 0.9 to 2.5% of the controller's cycles (the breakdown above), and the options take back
  a third to a half of that. The rest is DDR3's and the PHY's own spacing and banks still waiting on their timers.
- **One port, sequential** (the BIST's pattern): `same_cycle` takes reads from 90.4% to 91.7% of
  peak, writes from 89.9% to 91.2% (91.5% with all three). That is the BIST's figure, not decode.
- **No build of its own** (under the ~1% a build and a card session would have to pay for): it
  rides with the next full build, whatever brings it.

**The choosers' timing, out of context** (2026-10-02, omarchy: the generated core alone, both
channels, sys at 7.5 ns and the board's clock constraints, OBUFs on the PHY's single-ended outputs
as the board top's synthesis puts them; Explore):
- **The worst path through the multiplexer's cells** (`*choose*`, `*multiplexer*`,
  `*steerer*`) is +1.177 ns at 9 levels in LiteDRAM's core (the refresher's ZQCS timer into
  the bank machines), and +0.577 ns at 13 levels with `FASTMUX`. That path is the combinational
  grant: bank 10's row compare (CARRY4) -> `choose_cmd`'s priority encoder -> the command's
  accept and tFAW -> bank 12's `trascon` reset. 1.31 ns of it is logic and 5.18 ns is route.
- **Calibration:** the same filter on the production build's routed checkpoint (g2fix
  0885d436) gives +1.004 ns, 0.17 ns tighter than out of context. So `FASTMUX` should keep
  about +0.4 ns in a full build.
- **The core's other paths:** the sys domain's WNS out of context (-1.30 ns LiteDRAM's,
  -1.06 `FASTMUX`) is on paths only the out-of-context placement has (a CSR counter into the
  CPU's LUTRAM FIFOs, 92% route). In the full build that domain is +0.009 ns, on the
  `crg_rst1` fanout. `FASTMUX` takes 1,021 fewer LUTs: the registered grant goes.

**The fallback, `FASTMUX_SAFE`** (`rtw 3`, `direct_wtr`, no `same_cycle`): no new
combinational path, -0.42 / -0.17 / -0.52% decode on the 4-bit models (above). If a build
misses sys's timing on the `choose_cmd` / `choose_req` paths:
1. Set `MULTIPLEXER = FASTMUX_SAFE` in `ctl_settings.py`.
2. Run `check_core.sh --update` and `gen_ldc.py sim/verilator/otpu_ldc_ch.v`.
3. Rebuild. `test_rtl.py`'s BIST figures follow `MULTIPLEXER`.

**The BIST's figures in the model** (`test_ldc_sequential_is_the_card_bist`, one port,
sequential, % of peak):

| | read | write |
|---|---|---|
| LiteDRAM's multiplexer | 90.4 (card 90.4) | 89.9 (card 89.8) |
| `FASTMUX` | 91.65 | 91.49 |
| `FASTMUX_SAFE` | 90.38 | 90.17 |

The first fastmux build's qual compares the card's BIST with these.

**On the card,** `tools/qual/qual.sh` runs `tools/qual/turnaround.py` before its final selftest,
in every qual:
- **The traffic:** 30 s (TURN) of fp4 weight reads (port A) beside 64 KiB stores and loads of
  the tile stored just before (port B), so reads and writes take turns in both channels all the
  time.
- **The data:** the stored tiles and the MMs' results must equal the ISA simulator's.
- **The ECC counters:** both channels' sec / ded counts must be 0. A turnaround that comes too
  early corrupts a burst on the bus, and every beat carries its ECC byte, so the channel's
  decoder sees it. The loads of what was just stored put any write it corrupted through the
  decoder in the same run.
- **Checked so far:** on the board model (`test_board.py`). Its first card run is the next
  qual.

### What is left after memeff (2026-10-02)

The breakdown above, again on the production RTL (main 4e0b866: two ports, memeff, port A's
flush), now in the tree: `OTPU_LDC_BREAK=1 tools/perf_ddr.py ...` passes `+ldc_break` and
prints and caches each channel's `BRK` line (`sim/verilator/otpu_ldc_mem.sv`). The categories are
the same, in the same priority order; the line also counts the idle gaps and each bank's row
changes. The co-simulation must run the board's configuration
(`OTPU_MCOLS=4 OTPU_MXU=systolic OTPU_PAIR=1 OTPU_DSTEP=1 OTPU_STREAM=1`): without the last three
the program is not the card's (Qwen3 4-bit: 5.37 M cycles against the card's 3.88 M).

The whole model at 133.33 MHz, 4-bit layers, % of channel 0's cycles from its first command to
its last (channel 1 within 0.1 points):

| | Qwen3 | LFM2 | Qwen3.5 |
|---|---|---|---|
| Mcycles/token | 3.844 | 1.398 | 4.841 |
| column commands (data) | 90.4 | 91.7 | 92.9 |
| refresh | 2.7 | 2.7 | 2.7 |
| the multiplexer (turnarounds, empty grants) | 1.6 | 0.9 | 1.9 |
| rows opening and closing, a bank's timers | 1.5 | 1.3 | 1.3 |
| the crossbar | 0.2 | 0.2 | 0.3 |
| nothing asked | 3.6 | 3.2 | 0.9 |
| of which gaps of 8+ cycles that end in the next beat of one of the last 16 read streams (up to 256 a gap) | 1.5 | 0.8 | 0.1 |
| the same, of the last 4 streams | 1.0 | 0.4 | 0.1 |
| row changes per bank, in the ports' order (thousands) | 37.7 | 13.3 | 46.5 |
| of which back to the bank's previous row (two streams in one bank) | 17% | 14% | 19% |

- **memeff's A runs halved the rows' share** (2.2-2.7% before). Refresh is unchanged and
  structural: DDR3 has no per-bank refresh.
- **The layout.** In a layer's block every matrix's data is followed by its scales and then by
  the next matrix's data, in the order the layer runs them (q, k, v, o, gate, up, the down
  parts), so most matrices start on the next beat of the previous one's scale stream. The
  exceptions are attention (its KV reads come between v and o) and the next layer (after the
  layer's KV cache).
- **A sequential prefetch into the core's gaps** would read, in a gap, the next beats of the read
  stream the gap interrupted. Its bound is the gaps that end in such a beat: 0.1 to 1.5% with an
  oracle choosing among the last 16 streams, 0.1 to 1.0% among the last 4. The other gaps end
  in a new stream (the KV cache, the next layer's block), whose address only the program knows.
  Not taken.
- **Bank-aware placement** could only avoid the rows that two streams take turns at in one bank:
  14-19% of the row changes, so about 0.2-0.3% of the cycles at the rows' share. Under
  ROW_BANK_COLUMN every stream longer than 64 KiB crosses all 8 banks, so a placement cannot
  keep two streams' banks apart anyway (XOR bank hashing, above: 0.1%). Not taken.
- **The multiplexer's share** is what `fastmux` takes a third to a half of (-0.3 to -1.0%,
  above). The rows' and the crossbar's are under 1.5% together.
- **So the controller is within about 6% of its data rate on decode,** and no single fix is worth
  more than about 1%. `fastmux` is the one to take, with a build that happens anyway.

### The core's own gaps (2026-10-02)

The idle time above is the core's. `tools/decode_gaps.py` runs the same whole-model token
traced (`+trace`), on main d1cb669's RTL at 133.33 MHz:
- **Blame:** each gap between two MMs (the MXU not streaming) goes to the instruction the next
  MM waited for.
- **Bounds:** the token again with a group of instructions replaced by NOPs (`--drop`,
  `--drop-match`). The data come out wrong; the cycles are the bound for making that work free
  or hidden.
- **Micro-architecture variants:** the RTL with a timing-only knob changed (`--uarch`, over
  `rtlsim.BOARD_UARCH`).

Cycles per token against main (4-bit layers unless noted):

| | Qwen3 | LFM2 | Qwen3.5 |
|---|---|---|---|
| main | 3,844,073 | 1,398,439 | 4,841,349 |
| EXP2SUB free (a bound) | -3.59% | -3.00% | -0.07% |
| every VPU and QACT instruction of attention free (a bound) | -3.17% | | |
| attention's QACTs free | -0.48% | | |
| q / k norms and RoPE free | -0.17% | | |
| the norms before the weight MMs free | +0.04% | | |
| the QACTs before the weight MMs free | +0.05% | | |
| the DeltaNet output's QACT free (a bound) | | | -0.39% |
| MXU prefetch FIFO 2048 chunks (1024) | 0.00% | | |
| dispatch window 32 (16) | -0.72% | -0.57% | +0.03% |
| TMEM: 2 writes per bank and cycle (1) | -3.53% | | |
| **TMEM arbiter: the VPU ahead of the MXU drain** | **-2.97%** | **-2.41%** | **-0.41%** |
| the same, Qwen3 8-bit (main 5,617,507) | -2.01% | | |
| dispatch window 32 on top of VPU-first (against VPU-first) | -0.08% | -0.08% | |
| the next head's score MMs ahead of the PV MM (program order; ISA results unchanged) | -0.03% | +0.09% | |

- **The idle time is attention's softmax.** Making EXP2SUB free takes away about as much as the
  controller's idle (Qwen3: 137.8k cycles against 136.6k). A head's PV MM waits for its
  probabilities: EXP2SUB on 2 x 256 scores, then QACT. The MXU runs its MMs in order, so the
  next head's MMs wait behind that PV MM.
- **EXP2SUB ran at about half its rate.** It averaged 501 cycles per 2 x 256 op, where its
  three composite passes take about 270. The rest was the VPU frozen by the TMEM arbiter:
  395k frozen VPU cycles per Qwen3 token.
  - The VPU writes all its lanes' banks at once and was last in the priority order.
  - So an MXU drain write in any one bank cost it the whole cycle.
- **The fix is the arbiter's order:** DMA, COLL, VPU, MXU drain, QUANT. The drain holds instead,
  which costs it little.
  - It takes 83% of the EXP2SUB bound on Qwen3 and 80% on LFM2, and Qwen3.5 gains 0.4%.
  - Two write ports per bank take a little more, but every replicated TMEM copy would need a
    second write port (twice the copies, or a live-value table): not taken.
- **Not worth taking:**
  - The norm -> QACT -> MM chains cost nothing: the 1024-chunk MXU FIFO (128 KiB) streams the
    next MM's weights through them, and 2048 chunks change nothing.
  - Moving the next head's score MMs ahead of the PV MM changes nothing either: the VPU, not
    the MXU's order, was the slow part.
  - A 32-entry window is worth 0.6-0.7% alone but only 0.08% on top of VPU-first (the gaps it
    filled were the softmax's), and it costs the window's hazard logic at 133.33 MHz.

### The fused build: fastmux and VPU-first (2026-10-02)

One build at 133.33 MHz carries both: fastmux 5a088e5 (the core with `FASTMUX`, "The chooser and
the turnarounds" above) and vpu-first ed66aba (the arbiter's order, "The core's own gaps"), on
main. It was built from fused-fmvf 542fc43 (main a60df35 merged; BUILD_ID 542fc43a). The 32-entry
window stays out: on top of VPU-first it is worth 0.08%, not the 1% that would pay for its hazard
logic.

**The slice out of context** (`otpu_slice` alone at the board's generics: MCOLS 4, LANES 8,
WIN 16, RPB 64, WPB 1; 7.5 ns; the build's synthesis and implementation directives; omarchy).
Synthesis spreads the arbiter's LUTs into the units and the grant ports leave the hierarchy, so
the arbiter's paths are found by their ends: a unit's write request into another unit's grant.

| | main | VPU-first |
|---|---|---|
| WNS (neither is the arbiter: route-only paths) | +0.052 ns (MXU weight register -> DSP, 0 levels) | +0.081 ns (ACT RAM write index fanout, 1 level) |
| the VPU's write request (hen / hrot) -> worst | +1.077 ns (-> TMEM write address, 8 levels) | +0.669 ns (-> the MXU's grant enables, 9 levels) |
| DMA -> the MXU's grant enables | +0.506 ns (9 levels) | +0.861 ns (8 levels) |
| the MXU -> the VPU's grant (WBUF enable) | +1.175 ns (9 levels) | (no path: the MXU is after the VPU) |
| a unit -> the TMEM write ports (the grants gate them), worst | +0.182 ns | +0.478 ns |
| Slice LUTs | 105,500 | 104,281 |

- **The new path is the VPU's request into the MXU's grant:** +0.669 ns. It runs the VPU's
  bank flip-flops through the bank masks and the priority chain into the drain's clock enables:
  6.59 ns, 86% of it route.
- **Nothing near the arbiter got worse.** Every cross-unit path in the table has at least
  +0.48 ns in VPU-first, against +0.18 ns at worst in main. Placement moves these by a few
  tenths from run to run, and the whole design runs about 0.2 ns tighter than out of context,
  so the margin holds.

**The predictions** (whole-token co-simulation, the qualification's operating point: pos 544,
int8 head, 133.33 MHz, DDR3-1066). The card's column is production pa e4db91c9's qualification
(`qual.sh`'s prefill + decode counters), and the expected column is that times the co-simulated
change:

| | main, cycles | fused, cycles | change | the card now, Mcycles | expected | device tok/s |
|---|---|---|---|---|---|---|
| Qwen3 4-bit | 3,844,073 | 3,696,899 | -3.83% | 3.878 | 3.730 | 34.38 -> 35.75 |
| LFM2 4-bit | 1,398,439 | 1,355,035 | -3.10% | 1.413 | 1.369 | 94.36 -> 97.38 |
| Qwen3.5 4-bit | 4,841,349 | 4,765,281 | -1.57% | 4.895 | 4.818 | 27.24 -> 27.67 |
| Qwen3 8-bit | 5,617,507 | 5,471,471 | -2.60% | 5.650 | 5.503 | 23.60 -> 24.23 |
| LFM2 8-bit | 2,055,625 | 2,012,426 | -2.10% | 2.069 | 2.026 | 64.44 -> 65.83 |
| Qwen3.5 8-bit | 6,830,385 | 6,764,651 | -0.96% | 6.884 | 6.818 | 19.37 -> 19.56 |

- Together the two do a little better than their sum. On the 4-bit models, fastmux alone gives
  -0.74 / -0.35 / -0.86% and VPU-first alone -2.97 / -2.41 / -0.41%, which sum to
  -3.71 / -2.76 / -1.27%.
- memeff's build met its predictions within 0.1 points. Read more than 0.3 points short as a
  miss to explain.

**Qualification:** the usual `qual.sh`, which now carries `turnaround.py` (data against the ISA
simulator, and the ECC counters for fastmux's rtw 3). The decode table compares the six counters
with the expected column.

**The build** (omarchy, 2026-10-02 09:19-10:27, full effort; speculative while the models' RTL gate
ran, 64 / 64 passed):
- WNS +0.026 ns, WHS +0.014 ns. The core clock has +0.054 ns (pa +0.055), LiteDRAM's sys clock
  +0.257 ns (pa +0.116), and userclk1 +0.048 ns: the reroute ran on 7 paths and was kept.
- Slice LUTs 174,268 (pa 175,425), slices 76.3%; BRAM and DSPs as pa.
- otpu.bit sha256 3c91fa6a46ff8296c12420749c27285911fdff2c43aa2f2a000f11ae8c65a93a.

**On the card** (opentpu, 09:40-10:15, one otpu-lock; production pa restored after): `qual.sh fast`
gave 0 FAIL lines and 44 PASS. The decode counters land on the predictions:

| | pa, Mcycles | expected | measured | change, predicted | change, measured |
|---|---|---|---|---|---|
| Qwen3 4-bit | 3.878 | 3.730 | 3.730 | -3.83% | -3.82% |
| LFM2 4-bit | 1.413 | 1.369 | 1.369 | -3.10% | -3.11% |
| Qwen3.5 4-bit | 4.895 | 4.818 | 4.817 | -1.57% | -1.59% |
| Qwen3 8-bit | 5.650 | 5.503 | 5.506 | -2.60% | -2.55% |
| LFM2 8-bit | 2.069 | 2.026 | 2.025 | -2.10% | -2.13% |
| Qwen3.5 8-bit | 6.884 | 6.818 | 6.818 | -0.96% | -0.96% |

- **Every row is within 0.05 points of its prediction.** Bandwidth while decoding rises from
  90-94% of the 17.1 GB/s peak to 93-95%.
- **The streamed decode** (`decode_profile`, device tokens/s), pa to fused: Qwen3 4-bit
  37.48 -> 38.35, LFM2 98.86 -> 100.08, Qwen3.5 27.35 -> 27.80. The on-card decode loop gives
  Qwen3 37.89 -> 38.76, LFM2 98.89 -> 99.38 and Qwen3.5 27.71 -> 28.16.
- **Every model run is token-exact** against the ISA simulator: all six per position, resident,
  and on the card's decode loop.
- **The turnarounds:** 1765 runs and 181.3 GB in 30 s, every result equal to the ISA simulator's.
  The ECC counters are 0 / 0 on both channels.
- **The BIST** (2 GiB, 2 passes, both channels, 0 errors): reads 91.6% and writes 91.4% of peak,
  which is `FASTMUX`'s simulated 91.65 / 91.49. Production's stock multiplexer gives 90.4 / 89.8.
  The scrubs after it read ECC 0 / 0.
- **The rest also passes:**
  - WAITW's tag rounds: 2000 rounds of 1-4 MiB.
  - The H2C stress: acc_overlap 300 s, xmon overlap 300 s and serial 60 s, flags 0.
  - dma_bench: H2C 2.31-2.32 GB/s.
  - The selftests.

### Decode after the fused build: the controller's share (2026-10-02)

With VPU-first and fastmux in (production 542fc43a), decode is bound by the controller. The same
whole token on the fused RTL with the controller's breakdown (`OTPU_LDC_BREAK=1
tools/decode_gaps.py ...`), % of channel 0's cycles from its first command to its last:

| | Qwen3 | LFM2 | Qwen3.5 |
|---|---|---|---|
| cycles per token | 3,696,899 | 1,355,035 | 4,765,281 |
| column commands (data) | 94.0 | 94.6 | 94.4 |
| refresh | 2.7 | 2.7 | 2.7 |
| rows opening and closing, a bank's timers | 1.65 | 1.17 | 1.22 |
| the multiplexer | 0.82 | 0.52 | 0.85 |
| the crossbar | 0.23 | 0.22 | 0.26 |
| nothing asked | 0.58 | 0.81 | 0.60 |
| of which in gaps of 24+ cycles (room for a refresh) | 0.48 | 0.75 | 0.57 |

- **The controller is busy 99.2-99.4% of the token.** VPU-first took "nothing asked" from 3.6% to
  0.6% on Qwen3. The core still leaves its ports idle in 3.2-3.7% of 16-cycle windows, but the
  controller's queues absorb those windows. So a change in the core cannot shorten decode any
  more: the softmax's remainder, the QACTs, two writes per bank, the window and the FIFO all drop
  out.
- **Refresh is the largest share and structural.** DDR3 refreshes a whole rank (no per-bank
  refresh), and both channels already refresh at the same time. A model with no refresh
  (`gen_ldc.py --no-refresh`, a bound only) takes -2.01 / -2.24 / -2.40%.

**Refresh in the idle time** (`tools/litedram/idlerefresh.py`, `gen_ldc.py --idle-refresh
AHEAD,BEHIND,BURST,MIN_IDLE`):
- **How it works:** it refreshes once when no bank machine has had a request for MIN_IDLE
  cycles, and when the balance is full it refreshes BURST times back to back.
- **DDR3's limits:** it keeps the balance in a credit, the tREFI ticks minus the refreshes done,
  within DDR3's limits: at most 8 refreshes owed and 8 done ahead.
- **Checking the limits:** the BRK line counts the refresh commands on the DFI and the balance's
  extremes (`refs`, `owe`), as a check of those limits.

With 8, 8, 2, 4:

| | Qwen3 | LFM2 | Qwen3.5 |
|---|---|---|---|
| cycles per token | 3,688,542 | 1,351,317 | 4,746,662 |
| against LiteDRAM's refresher (postponing 2) | -0.23% | -0.27% | -0.39% |
| nothing asked, cycles (from) | 3,928 (21,462) | 6,979 (10,986) | 12,881 (28,635) |

- **The gain is the idle time it fills:** most of it already, so another setting would gain
  little.
- **The balance reaches DDR3's limits:** LFM2's ran from -8 to +8, so busy stretches do postpone
  all 8. LiteDRAM's refresher (postponing 2) stays within 0 to 2. A version for the card would
  keep a margin (BEHIND 7).
- **tREFI:** the core and the model refresh at LiteDRAM's DDR3 tREFI, 64 ms / 8192 = 7.8 us
  (1042 cycles at 133.33 MHz; ld_test.py's MT41K256M8_tRFC160). That is the normal temperature
  range, up to 85 C; the 3.9 us of the extended range is not configured. The idle refresher keeps
  the same average: one refresh per tick, at most 8 owed or ahead.
- **Before any card use:** a simulation test that counts the refreshes over a long window under a
  worst-case stream (no idle gaps), showing the balance never passes 8 either way.
- **Parked:** -0.2 to -0.4% is under the 1% that pays for a build, and it changes the refresh
  timing the card's DRAM sees. It could ride along with a build that happens anyway, after a
  card check of its own (BIST and the ECC counters through a warm soak).

**What else is left in the controller:**
- **Rows:** at most about 0.3%, the reopens two streams cause in one bank (21% of the row changes
  on Qwen3).
- **The multiplexer:** 0.5-0.85% after fastmux, mostly the turnarounds the PHY needs.
- **The crossbar:** 0.2%.

Together these are about 1% and none reaches it alone. Decode's remaining levers are its bytes
per token: the formats, the KV cache and the LM head.

### The core clock at DDR3-1066: the co-simulated grid

Decode, 4-bit layers, int8 head, pos 544, one port per channel (the production controller);
Mcycles/token, tokens/s (DRAM GB/s, % of the 17.07 GB/s peak):

| | 100 MHz | 133.33 MHz | 150 MHz | 200 MHz |
|---|---|---|---|---|
| Qwen3-0.6B | 3.734, 26.8 (11.9, 70%) | 4.218, 31.6 (14.1, 82%) | 4.660, 32.2 (14.3, 84%) | 6.230, 32.1 (14.3, 84%) |
| LFM2.5-230M | 1.348, 74.2 (12.2, 71%) | 1.541, 86.5 (14.2, 83%) | 1.712, 87.6 (14.4, 84%) | 2.259, 88.5 (14.5, 85%) |
| Qwen3.5-0.8B | 4.636, 21.6 (12.4, 73%) | 5.360, 24.9 (14.3, 84%) | 5.983, 25.1 (14.4, 85%) | - |

- **Above 133.33 MHz decode gains 1-2% at 150 MHz and nothing more at 200.** The controller
  saturates at 82-85% of peak.
- **Below 133.33 MHz the port is the limit.** At 100 MHz the port caps the path at 12.8 GB/s,
  and decode runs at 70-73% of peak.
- **Decode's lever is the controller's efficiency, not the clock.** At 133.33 MHz the port
  matches the two channels.
- **With two ports the knee stays at 133.33 MHz.** Qwen3.5 on the two-port design (ld-2port):
  5.001 Mcycles/token at 133.33 MHz, 26.7 tokens/s (15.35 GB/s, 89.9%); 5.576 at 150 MHz, 26.9
  (15.48, 90.7%). That is +0.9%, against +0.8% with one port.

The old bank model's grid (`docs/board.md`, "Faster DDR3") predicted 35.4 / 36.3 tokens/s for
Qwen3 at 150 / 200 MHz. It had no crossbar lock.

## 12. The core's timing at 133.33 MHz (ld-fmax, 2026-09-30)

The 812bb01 full build (main at 133.33 MHz: WNS +0.074) had three LiteDRAM sys-domain path
families between +0.077 and +0.127 ns. All three are the die's scale, not logic: the two channels
sit at its two ends (by their DDR3 banks), and one register drove loads at both.

- **The CSR bus** (`interface1_dat_w` -> `phaseinjector*_storage`, 0 levels, 97% route): the CSR
  bridge's registers drove about 140 loads each over 200 rows.
  - Now a register stage per group of banks (`tools/litedram/csr_pipe.py`): channel 0's banks,
    channel 1's, and the rest (`ctrl`, `identifier_mem`, `selfcal`). Each copy is placed by its
    banks, and each group's read data comes back through a register.
  - LiteX's registered Wishbone2CSR acks two cycles later: an access takes 5 sys cycles instead
    of 3. The host (AXI-Lite) and the calibration CPU both handshake, so nothing depends on it.
  - The CSR map, `csr.csv`, `sdram_init.py`, the XDC and the firmware ROM are unchanged.
- **The sys reset** (`FDPE_1` -> `dfi_p*_address` R, `selfcal_mem_dat_storage` R; 3,794 loads,
  6.8 ns of route): `FDPE_1` is the reset synchronizer's ASYNC_REG flip-flop, which Vivado does
  not replicate. `WLCRG(rst_reg=True)` (`ld_test.py`; the test images keep the stock reset)
  puts a plain register after it (initially 1, `max_fanout` 256). The whole sys reset releases
  a cycle later: the controllers, the PHYs' serializer resets (`wlrst`, from `sys_rst`) and the
  write-clock MMCMs' resets keep their order.
- **The ECC counters** (the PHY's read bitslip -> the decoder -> `ded_errors_status` CE, 11
  levels): `ecc_ports.py` counts from registered error flags, so the counters are
  `LiteDRAMNativePortECC`'s a cycle late. A beat's errors are dropped if a clear comes with it,
  as the stock counters drop them. The read data path, and so the two ports' identical read
  buses, are unchanged. The four ECC CSRs per channel stay.
- **Checks:**
  - `csr_pipe_check.py` (migen): LiteX's bridge and bus against the pipelined ones, 3,000 random
    accesses (reads and writes to every word and to unmapped offsets). The banks hold a 32-bit
    storage, a 64-bit `atomic_write` storage, a 40-bit status, a CSR's write / read strobes (as
    the PHYs' delay-line taps) and pulse fields. Every access returns the same data, and leaves
    the same storages; every strobe comes once, in order, with the same data. It fails with one
    wait cycle fewer, and without the read-data register.
  - `ecc_ports_check.py` (600 cycles, now comparing a cycle later): PASS at seeds 1, 2 and 3.
  - `selfcal_sim.py` on the regenerated core, its SoC on the same pipelined bridge and bus
    (stride 16, `--hold-test`): 643,724 CSR writes, identical to `ddrcal`'s; `c0_ready` /
    `c1_ready` rise; both channels' results equal `ddrcal`'s; the hold test passes.
  - The co-simulation (`gen_ldc.py` regenerated: `sim/verilator/otpu_ldc_ch.v`):
    `test_rtl.py -k ldc` 5 passed; `perf_qwen --wformat fp4 --ddr 1066 --mhz 133.33 --ldc`
    1,478,549 cycles before and after on d3d5f0f, and 1,478,546 on ld-2port's final eb5beed (its
    `otpu_ldc_mem` checks both ports' read buses equal) and with this on it: the whole profile
    identical both times.
  - `check_core.sh` reproduces the committed core.
- **ld-memch's ldimpl harness** (the core and both bridges on the board's pins, the board
  build's strategies; Vivado 2026.1; eb5beed -> this). The harness stretches these families far
  past the full build (its sys WNS is its own XDMA queues', -5.68 / -5.91), so compare within the
  clock: for each sys endpoint below +0.5 ns, the family of its worst path.
  - The sys reset: 2,295 endpoints below 0, worst -4.284 (`FDPE_1`, 0 levels) -> 1,085, worst
    -3.260 (1 level).
  - The CSR bus: 716 below 0, worst -2.479 -> 420, worst -2.534. Into the phase injectors'
    storages (the full build's family), worst -1.276 -> -0.148.
  - The ECC counters: 152 endpoints below +0.5, worst -1.845 (10 levels) -> none.
  - Unchanged: the write clocks, +0.43..+0.56 -> +0.38..+0.53. u_ld: 23,825 -> 23,917 LUTs,
    17,895 -> 18,139 flip-flops.
  - Left in the harness: `max_fanout` makes synthesis's copies of the reset register, and they
    are not placed by their loads. `crg_rst1_reg_rep__13` sits at Y173 and drives 181 loads at
    Y254-302 (6.8 ns of route to them), as does the CSR bus's write strobe for the PHY's bitslips
    (`wdly_dq_bitslip_rst`, 213 loads). The next full build measures both.

### The write clocks' paths (ld-phy)

After the sys families, the write clocks' worst paths were the PHY's own two hops to the
serializers, at +0.15 to +0.35 ns in 812bb01 and 110ec6d.

- **The serializer resets** (`wlrst` -> OSERDES / ISERDES RST, 3.0 ns `set_max_delay`): one
  register per lane drove its 9 serializers across the lane's 12 I/O rows (812bb01: +0.282, 1.99
  ns of route). Now there is one per DQ pad (its OSERDES and ISERDES, in one I/O tile) and one per
  DQS pad. The command pads keep one per eight; `reset_n`, in the next bank, gets its own, so no
  group spans two banks (812bb01: +0.347, a group over rows 325-388). That makes 86 registers per
  channel instead of 22, with the same reset and the same release.
- **The command pads' crossing** (`nreg`: an FDRE on sys's falling edge, half a cycle in, then
  half a cycle less the crossing's 1.0 ns of uncertainty out to the serializer). A query of
  110ec6d's routed checkpoint over all 402 of these registers found:
  - The worst are `cs_n` / `ras_n` / `cas_n` / `we_n`'s: +0.220..+0.29 on the way out.
  - Their input is the phase injector's command issue, which LiteDRAM decodes from the CSR bus's
    address register in the same cycle: 3-4 LUT levels from the die's centre, +0.33..+0.6 in.
    The placer parks the registers between the two (channel 0: rows 158-186, for pads at
    80-92).
  - The two halves together have at least +0.634 / +0.708 (channel 0 / 1). A Pblock by the pads
    would only move the failure to the input side, so there is none.
  - `dfii_q.py` registers the command issue instead. A software-injected command (the
    calibration's; the controller's commands do not pass there) leaves from a flip-flop a sys
    cycle later. The whole 4-phase word moves with it: the command, its write and read data
    enables, and commands on other phases. Its address, bank, data and command fields are CSR
    storages written by earlier accesses.
  - The command still goes out within its own CSR access (5 cycles). The selfcal firmware reads
    `rddata` a whole precharge command later (4 CSR writes after the read), and the host's
    `ddrcal` microseconds later.
- **Checks:**
  - `dfii_q_check.py` (migen): LiteDRAM's DFIInjector against one with this PhaseInjector, the
    same CSR writes, controller traffic and read data. There are 20,000 writes per seed: issues
    on any phase (activate, write with its data enable, read with its, precharge, a data enable
    alone), the storages, and the control register. The gaps are 0-4 cycles; the production
    bus's are 4 at least.
    - Under software control each phase's command signals equal LiteDRAM's a cycle earlier.
    - Every injected word comes out whole a cycle later, so the phases keep their alignment.
    - The storage-driven signals are the same in the same cycle, and under hardware control
      everything is; the read data registers are equal.
    - At seeds 1-3: PASS, 5,561-5,683 injected-command cycles each. Of those, 855-903 are data
      enables on another phase than a command at most 5 cycles before, and 1,096-1,122 come at
      most 2 cycles after the last command.
    - It fails with the issue not registered, or registered twice.
  - `selfcal_sim.py` (stride 16, `--hold-test`): PASS.
  - The co-simulation, with `otpu_ldc_ch.v` regenerated (its four injectors' registers):
    `test_rtl.py -k ldc` 5 passed; `perf_qwen --ldc` 1,478,546 cycles, as before.
  - The regenerated core differs from one without the register by exactly the eight injectors'
    registers, and from ld-fmax's by those and the resets. Every SERDES RST is driven by a
    `wlrst`: 144 of them at fanout 2 (a DQ's OSERDES and ISERDES), 22 at fanout 1, 6 at 8.
- **The card, before production:**
  - Calibration and BIST on both channels.
  - Write and read leveling results equal to the production image's: the injected commands'
    timing is the calibration's.
