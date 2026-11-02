# openTPU on the SQRL FK33

The SQRL Forest Kitten 33: Virtex UltraScale+ HBM xcvu33p-fsvh2104-2-e (one SLR), 8 GB HBM2 in two stacks, PCIe
(GTY lanes of an x16 edge connector, 8 used), two MT25QU256 configuration flashes (SPI x8). The FK35 is the same
PCB with the xcvu35p (docs/sqrl-fk35.md: two slices, all 8 GB).

| | |
|---|---|
| target | `targets/sqrl-fk33.toml` (platform 5, S = 1, DUAL) |
| top | `rtl/boards/sqrl-fk33/otpu_fpga_top_fk33.sv` |
| block design | `boards/sqrl-fk33/vivado/bd.tcl`: XDMA (Gen3 x8 on PCIE4C_X1Y0), clocks, SYSMON, BAR0 and DMA interconnects |
| memory | `otpu_hbm.sv` (the HBM IP, all 32 AXI ports as arrays) and one `otpu_hbm_port.sv` per channel |
| build | `boards/sqrl-fk33/run_vivado.sh` (Vivado 2019.2), with this board's `vivado/build.tcl`, `scripts/preflight.tcl` and `vivado/tmem_uram.xdc` |

## The design

One slice (`otpu_fk33_sys` at S = 1, a copy of the BCU1525's `otpu_bcu_sys`: the accelerator and XDMA's DMA
meeting in one `otpu_mem_ch` per channel) on four channels, both HBM stacks (DUAL): channel j is HBM bytes
j x 2 GiB .. + 2 GiB, 0 and 1 on the first stack, 2 and 3 on the second. The slice's 32-bit DRAM addresses reach
4 GiB of the 8 GB: its D-byte chunks alternate between the channel pairs (`otpu_native_dram2`: chunk m is chunk
m >> 1 of the two-channel adapter m & 1, on channels 2 (m & 1) and 2 (m & 1) + 1), 1 GiB of each channel. Each
channel's `otpu_hbm_port` (the VCU128 build's HBM adapter) turns otpu_mem_ch's two native controller ports into four
AXI3 ports (NS = 2), port a = p + 2k on the IP's AXI_(8j + 4k + p), each sub-port in its own mini-switch; AXI ports
0, 1, 4, 5, .., 28 and 29 are used, the rest tied off. The global switches are on. XDMA's DMA reaches channel j at
j x 2 GiB (the block design's SmartConnect, four masters); the host applies the chunk map (`opentpu/host/board.py`
`dual_at`; a bitstream whose CFG_DRAM reports four channels a slice is a DUAL one).

Decode streams the weights. The MXU takes one D = 128-byte chunk a cycle for every row it holds, and a decode MM
has one row, so three of the four columns idle; DUAL lets an int8 `MM PAIR` (docs/isa.md "Column reuse") take two
chunks an advance, block 2c in columns 0..M-1 and block 2c+1 in columns M..2M-1 (the operand's rows duplicated by
`QACT DUP`, as for 4-bit PAIR), and the port reads the pair in one request, a beat on each of the four channels. The
result is the same integer, bit for bit. The read path is sized for the HBM's round trip at 250 MHz: otpu_mem_ch ARD
256 read credits per channel, otpu_hbm_port RD 64 slots per AXI port (256 a channel) and otpu_native_dram RD 256
(otpu_board NRD). The HBM releases the adapters when both stacks have calibrated (`apb_complete_0/1`); a catastrophic
temperature trip clears CALIB (otpu_board then refuses RUN and LOAD).

Decode on the card (`tools/qual/perf.py`, 2026-11-01; device tok/s, prefill 512 tokens):

| build | Qwen3.5-0.8B int8 | 2B int8 | 4B mix | prefill 0.8B / 2B / 4B | DRAM while decoding |
|---|---|---|---|---|---|
| two channels, upstream read credits (ARD 64, RD 32, RD 128) | 30.76 | 12.87 | 8.53 | 132 / 63 / 21 | ~78% of 32 GB/s |
| two channels, ARD 128 | 33.53 | 13.92 | 9.43 | 137 / 69 / 23 | ~85% |
| two channels, ARD 256, RD 64, RD 256 | 38.07 | 15.99 | 10.50 | 144 / 78 / 26 | ~96% |
| four channels, DUAL (this target) | **68.71** | **30.98** | **14.05** | 146 / 79 / 31 | 88-95% of 64 GB/s (4B mix: 64%) |
| two channels, MCOLS 8 (experiment) | 38.14 | 16.00 | 10.52 | 157 / 78 / 35 | ~96% |

The 4B mix's 4-bit layers (DeltaNet, MLP) still take one chunk a cycle (4-bit PAIR fills the second column with the
chunk's other half), so it gains 1.34x where the int8 models gain 1.8-1.9x. Eight MXU columns (MCOLS 8) leave decode
where it is and help prefill only where TMEM (64K words) holds more rows a run.

Clocks: the card has no clock input besides the PCIe reference, so one MMCM on XDMA's axi_aclk (250 MHz) makes
core_clk (250 MHz), uclk (250 MHz: the HBM's AXI ports, the adapters, otpu_mem_ch's controller side) and clk100 (the
HBM's reference and APB clock). A PCIe link reset restarts the HBM too.

PCIe identity: SQRL's, vendor 1E24, device 1533, subsystem 1E24:0001 (rev A3), class 12 00 00. BAR0 (1 MB): the
control registers at 0x0, the SYSMON at 0x30000; the DMA BAR is XDMA's (64 KB). XDMA's DMA reaches channel j at
j x 2 GiB.

Implementation: the slice, the shell and the four channels share the one SLR and congest; placement spreads the
logic (AltSpreadLogic_high) and physical optimization and routing use AggressiveExplore. TMEM_URAM defaults to 1
here. At 250 / 250 MHz (CORE_MHZ, UCLK_MHZ) the core's pipeline registers in the sequencer, DMA, activation RAM,
control, collective, trace, quantizer and fp units carry it. The DUAL build (f33d0002, 2026-11-01): WNS +0.003 ns
(core), +0.055 ns (uclk), +0.165 ns (xdma), WHS +0.010 ns, 0 critical warnings; 294k LUTs (67%), 416 block RAM
tiles, 723 DSPs (the two-channel build: 221k, 341.5, 723). On an FK33: selftest ALL PASS with the model stage for
Qwen3.5-0.8B (the two-channel builds: also -2B int8 and -4B `--wformat mix`, whose int8 image needs 4142 MiB of the
slice's 4096).

Simulation (Verilator, tests/test_dual_rtl.py): int8 PAIR against the ISA simulator with the behavioural memory and
with otpu_native_dram2 on a four-channel model with random stalls (odd and even KB, pairs across the chunk parity,
unit / RSC / MX scales, ACC, RMAX, ASCALE, and the fuzzers' DMA, QST and scoreboard-hazard programs); the board model
(tests/test_board.py, OTPU_TARGET=sqrl-fk33: the host driver, the board shell and a tiny Qwen3 on four channels); and
the RTL suite with DUAL off (the other targets) unchanged.

## Build and load

```sh
source <Vivado 2019.2>/settings64.sh      # the VU33P needs a license covering it
boards/sqrl-fk33/run_vivado.sh            # build/vivado_sqrl-fk33/otpu.bit, otpu_primary.mcs, otpu_secondary.mcs
```

Load `otpu.bit` over JTAG (Vivado's hardware manager) and rescan PCIe (Linux: `echo 1 > /sys/bus/pci/devices/<bdf>/remove`,
then `echo 1 > /sys/bus/pci/rescan`; Windows: `pnputil /remove-device` and `/scan-devices`), or write the two .mcs to
the flashes (part `mt25qu256-spi-x1_x2_x4_x8`). Then `python -m opentpu.host.selftest`.

FK33 cards carry early VU33P silicon (JTAG IDCODE revision 0) for which Vivado 2019.2 has no part: Vivado's
hardware manager then refuses the bitstream (Labtools 27-3303) until `set_param xicom.use_bitstream_version_check
false`. The production -2-e bitstream runs on it.
