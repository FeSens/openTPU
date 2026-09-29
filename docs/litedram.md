# LiteDRAM instead of the MIG: an assessment (draft)

Draft, 2026-09-29. This is an investigation; the design is unchanged. Labels: **measured** means
a Vivado run, a simulation or a card run that was actually done (tool and version given).
**estimate** means arithmetic. Sections 1-6 predate the card; sections 7 and 8 are the card runs
(the test images, then the production image); section 9 is the board build; section 10 is
the calibration CPU in the core.

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
  `wr_idle` waits for `n_wdone`.
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
- one `otpu_mem_ch` per channel, in front of that channel's controller port, in the
  controller's clock. LiteDRAM's ECC port takes whole beats only, so `otpu_mem_ch`
  read-modify-writes partial beats.

**`bd_native.tcl`**: XDMA, the XADC and the control SmartConnect (the MIG build's `bd.tcl` had
the same). XDMA's `M_AXI` is exported as `M_AXI_DMA`; the CSR window is at BAR0 0x10000. The
50 MHz clock sits on one BUFG shared with the core's MMCMs.

**The LiteDRAM core** is `gen_core.py --phy wl` (the arguments are in `core.json`, which
`check_core.sh` regenerates with). It has the same ports as the A7DDRPHY core it replaced; its CSRs
add each channel's write clock MMCM (`wclk` / `wclk1` at 0x7000 / 0x7800 of the window, for
ddrcal's DRP and reset access). Its XDC carries two sets of constraints:
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

**Temperature rescans (designed for, not built).** The mailbox keeps each lane's pass map over the
scan and the run the CPU chose, so a drift shows as a moved or narrowed run. A rescan needs the
channel's traffic stopped (the scan writes the first 64 MiB and moves CK), so the host schedules
it: it quiesces the channel, sets `selfcal_config` to that channel and releases the CPU, which
recalibrates it as at reset. Firmware that tracks drift in place (a narrow scan around the
current phase, the BIST on a reserved region) can be loaded later without a new bitstream.

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
