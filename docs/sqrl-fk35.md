# openTPU on the SQRL FK35

The SQRL Forest Kitten 35: Virtex UltraScale+ HBM xcvu35p-fsvh2104-2-e (two SLRs), 8 GB HBM2 in two stacks, PCIe
(GTY lanes of an x16 edge connector, 8 used), two MT25QU256 configuration flashes (SPI x8). It is the FK33's PCB
(docs/sqrl-fk33.md) with the bigger FPGA: twice the logic, so two slices and all 8 GB.

| | |
|---|---|
| target | `targets/sqrl-fk35.toml` (platform 4, S = 2, DUAL) |
| top | `rtl/boards/sqrl-fk35/otpu_fpga_top_fk35.sv` |
| block design | `boards/sqrl-fk35/vivado/bd.tcl`: XDMA (Gen4 x8 on PCIE4C_X1Y0, 512-bit AXI at 250 MHz; PCIE_GEN=3: Gen3 x8, 256-bit, the builds before f35d000c), clocks, SYSMON, BAR0 and DMA interconnects |
| memory | `otpu_hbm.sv` (the HBM IP, all 32 AXI ports as arrays) and one `otpu_hbm_port.sv` per channel |
| floorplan | `boards/sqrl-fk35/constraints/fk35_floorplan.tcl`: the memory side (and the slices' memory adapters: MEMX) in SLR0, the slices and the shell in SLR1; Gen4: XDMA in a pblock at the PCIe block |
| build | `boards/sqrl-fk35/run_vivado.sh` (Vivado 2019.2), with this board's `vivado/build.tcl`, `scripts/preflight.tcl` and `vivado/tmem_uram.xdc` |

## The design

Two slices (`otpu_fk35_sys` at S = 2, a copy of the BCU1525's `otpu_bcu_sys`: the accelerator and XDMA's DMA
meeting in one `otpu_mem_ch` per channel), each on four channels (DUAL, as the FK33's slice: docs/sqrl-fk33.md):
board channel j = 4s + c is HBM bytes j x 1 GiB .. + 1 GiB, the four pseudo-channels of mini-switch j, so slice 0
has the first stack and slice 1 the second, and the slices' 32-bit DRAM addresses (4 GiB each) reach all 8 GB. Each
slice's D-byte chunks alternate between its channel pairs (`otpu_native_dram2`), and an int8 `MM PAIR` takes two
chunks an advance, so a slice streams 256 bytes a cycle (64 GB/s) and the card 128 GB/s; 4-bit layers still take
one chunk an advance. A model's weights are split between the slices (tensor parallel), so a model whose image needs
up to about 8 GB runs. Each channel's `otpu_hbm_port` (the VCU128 build's HBM adapter, PCB 2) turns otpu_mem_ch's
two native controller ports into four AXI3 ports (NS = 2), port a = p + 2k on the IP's AXI_(4j + a): every port
of the HBM is used, each channel's accesses stay in its own mini-switch, and its 64 bytes a cycle use half of its
ports' 128. The global switches are on. XDMA reaches channel j at j x 2 GiB (the host's channel map; the channel's
1 GiB at the window's start). The HBM releases the adapters when both stacks have calibrated (`apb_complete_0/1`);
a catastrophic temperature trip clears CALIB (otpu_board then refuses RUN and LOAD).

Before DUAL (build f35e0001, 2026-11-01) each slice had two channels of 2 GiB (j = 2s + c, PCB 3, AXI_(8j + 4k + p):
ports 0, 1, 4, 5, .., 28, 29), 128 bytes a cycle a slice. otpu_fpga_top_fk35 keeps that map for its `DUAL = 0`;
such a build also needs the target's `DUAL = false` and bd.tcl's `NCH 4` (and the top's four DMA ports).

SLRs: the HBM and the PCIe block are in SLR0, so the memory side (the eight otpu_mem_ch, their HBM adapters) stays
there with XDMA, and otpu_board's slices and shell go to SLR1. MEMX (the default; `MEMX=0` to run_vivado.sh for the
other way) puts each slice's memory adapter in SLR0 too: `otpu_native_xdram2` wraps the slice's otpu_native_dram2
(u_dram) with a crossing of the slice's ports A, SW and B (u_s with the slice, u_m with the adapter, their sll_*
registers flip-flop to flip-flop in the Laguna sites: about 7,100 a slice). Each port crosses into a 16-entry queue
on the adapter's side, the slice's rdy a credit per entry; the read data come back through two registers; the
adapter's wr_idle / rd_idle come back with the takes they cover and count only with every credit home; the flush and
the adapter's reset cross with the requests. The native ports then stay in SLR0 (otpu_fk35_sys SLL 0), and the
board's DRAM beat counters take the beats registered on the adapter's side (a cycle late). Without MEMX every
channel's native port crosses through two `otpu_native_reg` (u_tx with the slice, u_rx with the channel). With the
four-channel adapters in SLR1 that build (f35d0001) filled SLR1's block RAMs (96%, 92.5% of its CLBs) and routed at
WNS -0.15 ns; with MEMX SLR1 holds 69% of its block RAMs and 81% of its CLBs, SLR0 35% and 93% (about 67% of its
LUTs). `QMEM` (run_vivado.sh, the top's parameter) is the other way tried: the adapters' 512-bit x 32 queues in LUT
RAM (1) and the B FIFOs in UltraRAM too (2) took SLR1 down to 69% of its block RAMs with the adapters in place, but
neither routed better than about -0.3 ns before it was stopped. The core reset crosses as a register per side.

Gen4 x8 (PCIE_GEN 4, from f35d000c) needs XDMA's 512-bit AXI: XDMA grows from 24k to 54k LUTs (pcie4c_ip 3k -> 10k,
the DMA engine 20k -> 44k), SLR0 from 62% to 70% of its LUTs and 91% to 94% of its CLBs. Unconstrained (f35d000c)
the placer spread XDMA over X64-232 Y0-239 of SLR0, mostly its top row and a fifth among channels 2-5 and the memory
adapters, and the memory side failed (WNS -0.524 ns core, -0.404 uclk, -0.250 xdma_aclk; congestion level 6).
fk35_floorplan.tcl now puts XDMA and sc_dma's 512-bit entry (its S00 entry pipeline and nodes, where it converts to
the channels' 128 bits) in a hard pblock pb_xdma: the two right columns of SLR0 (the PCIe block PCIE4CE4_X1Y0 is in
clock region X7Y0) and the top row's right half, X6Y0:X7Y3 + X3Y3:X5Y3. The HBM side's reset (the top's u_rst) is a
register with max_fanout, not a gate on rst_uclk fanned out to every adapter. f35d000d (pblock, register, and an
axi_dwidth_converter 512 -> 128 ahead of sc_dma) routed at WNS +0.012 / WHS +0.005 ns but its DMA did not work:
2019.2's SmartConnect kept generating its S00 for XDMA's 512 bits and 4-bit ID behind the converter, the block
design zero-padded the converter's 128 bits into it, and C2H read DECERR (status 0x200) while H2C hung at any size.
Keep XDMA's M_AXI straight into sc_dma. f35d000e (pblock with sc_dma's entry, the reset register) met timing:
WNS +0.002 ns (core_clk +0.002, uclk +0.007, xdma_aclk +0.017, pipe_clk +0.026), WHS +0.004 ns; SLR0 70.3% of its
LUTs, 95.2% of its CLBs. Flashed 2026-11-07: self-test ALL PASS (host -> card 2.09 GB/s, card -> host 2.83), Qwen3.5-9B
decode 24.48 tok/s fp4 and 21.97 mix, as f35d000b. The slot (PCH root port 0:1B.4, x4) trains Gen4 x4 after a full
restart; retrains while Windows runs (flash + rescan, root-port restarts, sleep / fast-startup) left it at Gen3 x4
from 2026-10-27 until then. At Gen4 x4 (2026-11-07, no AER errors under load): self-test 2.18 / 2.89 GB/s
(host -> card / card -> host, Gen3 x4: 2.09 / 2.83), the driver's direct DMA 2.49 / 3.19 GB/s, its loop test to one
HBM channel 2.63 / 3.39 GB/s. The link is not the limit: a transfer goes to one channel's 128-bit DMA port
(4 GB/s at 250 MHz) and through otpu_mem_ch's XDMA side; SQRL's own design moves 7.1 / 6.9 GB/s over the same link.

The host reaches channel j at j x 2 GiB, and the channel's 1 GiB repeats in the window's upper half (otpu_hbm_port
takes the low 30 bits): CFG_DRAM still reports 2^31 bytes a channel, the 4 GiB slice's map. Nothing addresses the
upper half (a slice reaches 1 GiB of each of its four channels); the self-test's address-line check covers the
bytes a slice reaches.

## Measured

Build `f35d0004` (MEMX, 2026-11-02): WNS +0.009 ns, WHS +0.003 ns (AggressiveExplore placement, physical
optimization and routing; 2 h 32 min of routing). Flashed, the self-test passes (every channel, the model stage on
Qwen3.5-0.8B). Decode on the device (tools/qual/perf.py: 64 greedy tokens after a 512-token prompt), tok/s:

| Model | int8 | 4-bit, int8 head | Before DUAL, int8 (f35e0001) |
|:--|--:|--:|--:|
| Qwen3.5-0.8B | 117.59 | 118.65 | 73.92 |
| Qwen3.5-2B | 53.99 | 56.76 | 31.12 |
| Qwen3.5-4B | 25.29 | 25.89 | 13.99 |
| Qwen3.5-9B | 14.28 | 14.66 | (13.34 in 4-bit) |

int8 streams 439-462 bytes a cycle on the two slices (86-90% of their 512; 116 GB/s for the 9B, a quarter of the
HBM's 460.8), 4-bit about 242: a 4-bit MM takes one chunk an advance, so on this card int8 decodes about as fast as
4-bit and the 4B mix (25.79 tok/s) gains 2% over int8. MXU starvation 0-1%, DRAM wait 0%.

Build `f35d000b` (QUAD, 2026-11-04): a 4-bit MM PAIR of one row takes four blocks, the DUAL port's two chunks, an
advance (otpu_mxu QUAD, docs/isa.md "Column reuse"; target arch QUAD, CFG_UARCH bit25). Its first build (f35d0009)
routed at -0.190 ns, the failing paths spread over SLR0's memory side and the MXU's drain; three register stages
(the crossing's idle registers and a reply stage before its SLR crossing, STATUS AXI_ERR), otpu_mem_ch BRAM (the
channels' read and order FIFOs in block RAM: SLR0's LUT RAM 85.7k -> 62.2k LUTs, its block RAM 238 -> 418 tiles) and
the Performance_ExtraTimingOpt strategy closed it: WNS +0.010 ns (uclk; core_clk +0.022), WHS +0.006 ns. (The same
without BRAM, f35d000a, routed at -0.032 ns, and +0.015 ns implemented again with a BUFG on u_rst, the QoR
suggestion of its routed design: run_vivado.sh RQS= / PRE_OPT=.) Flashed, the self-test passes. Decode on the
device, tok/s:

| Model | int8 | mix (formats.MIXES) | 4-bit, int8 head | 4-bit before QUAD (f35d0004) |
|:--|--:|--:|--:|--:|
| Qwen3.5-0.8B | 117.51 | | 164.18 | 118.65 |
| Qwen3.5-2B | 54.03 | | 86.02 | 56.76 |
| Qwen3.5-4B | 25.29 | 35.90 | 43.08 | 25.89 |
| Qwen3.5-9B | 14.29 | 21.96 | 24.48 | 14.66 |

4-bit now streams 404-424 bytes a cycle for the 4B and 9B (int8 439-463), so 4-bit decodes 1.4-1.7x faster than int8
and the card takes the MIXES of docs/formats.md (no `[mixes]` of its own since; f35d0004's made the 4B and 9B int8).
Prefill is unchanged (0.8B 242 tok/s, 9B 24-29).

Clocks: the card has no clock input besides the PCIe reference, so one MMCM on XDMA's axi_aclk (250 MHz) makes
core_clk (250 MHz), uclk (250 MHz: the HBM's AXI ports, the adapters, otpu_mem_ch's controller side) and clk100 (the
HBM's reference and APB clock). A PCIe link reset restarts the HBM too.

PCIe identity: SQRL's, vendor 1E24, device 1533, subsystem 1E24:0001 (rev A3), class 12 00 00. BAR0 (1 MB): the
control registers at 0x0, the SYSMON at 0x30000; the DMA BAR is XDMA's (64 KB). XDMA's DMA reaches channel j at
j x 2 GiB (16 GiB of windows, 8 GiB of HBM).

Implementation: the FK33's directives, placement spreading the logic (AltSpreadLogic_high) and physical optimization
and routing with AggressiveExplore. TMEM_URAM defaults to 1 here.

## Power

The card cannot report its own power: the bitstream has no I2C/PMBus master to the board's regulators (HAS_I2C=0),
and the SYSMON at 0x30000 gives temperature and supply voltages, no current. So far there is only Vivado's
vectorless estimate of f35d000b (`reports/power.rpt` of its build; default switching activity, confidence Medium),
on-chip only: 38.8 W, of it HBM 17.8, block RAM 4.2, signals 3.6, clocks 3.4, CLB 2.4, GTY 2.4, the rest
(URAM, DSP, PCIe, MMCM) 1.8 and static 3.2; VCCINT 0.85 V at 20.4 A. Divided by the decode speeds above, Qwen3.5-9B:

| 9B weights | tok/s | J/token | kWh per million tokens |
|:--|--:|--:|--:|
| 4-bit, int8 head | 24.48 | ~1.59 | ~0.44 |
| mix | 21.96 | ~1.77 | ~0.49 |
| int8 | 14.29 | ~2.72 | ~0.75 |

Unverified, and without the board's regulator losses, fans and the rest of the card.

Measured at the wall (2026-11-05, f35d000b, a socket meter on the whole PC): 130 W with the card configured and idle,
130 W with Qwen3.5-9B 4-bit loaded and idle, **153 W during its decode with the loop on the card**
(`Engine.generate_card`: 22.9 tok/s, the host at 6% of one core), so the decode costs the card **about 23 W over
idle, about 1.0 J/token (0.28 kWh per million tokens)**. A decode loop on the host (perf.py's `step` loop) does not
measure the card: it can hold a host core at 100%, which on that PC alone is +60 W at the wall. The card's idle power
is inside the 130 W (it needs a reading without the card, or a PCIe power riser). The 9B int8 decode loop on the card
stopped after its first token there (run_generate's 600 s timeout; the 4-bit loop and the 0.8B's int8 loop ran).

## Build and load

```sh
source <Vivado 2019.2>/settings64.sh      # the VU35P needs a license that covers it
boards/sqrl-fk35/run_vivado.sh            # build/vivado_sqrl-fk35/otpu.bit, otpu_primary.mcs, otpu_secondary.mcs
```

Load `otpu.bit` over JTAG (Vivado's hardware manager) and rescan PCIe (Linux: `echo 1 > /sys/bus/pci/devices/<bdf>/remove`,
then `echo 1 > /sys/bus/pci/rescan`; Windows: `pnputil /remove-device` and `/scan-devices`), or write the two .mcs to
the flashes (part `mt25qu256-spi-x1_x2_x4_x8`). Then `python -m opentpu.host.selftest`. A design with other BAR sizes
loaded before (the openTPU_FK builds' 128 KB BAR0) leaves the old windows on Windows until the card's root port is
restarted (`pnputil /restart-device` on it) before the rescan.
