# openTPU on the SQRL FK33

The SQRL Forest Kitten 33: Virtex UltraScale+ HBM xcvu33p-fsvh2104-2-e (one SLR), 8 GB HBM2 in two stacks, PCIe
(GTY lanes of an x16 edge connector, 8 used), two MT25QU256 configuration flashes (SPI x8). The FK35 is the same
PCB with the xcvu35p.

| | |
|---|---|
| target | `targets/sqrl-fk33.toml` (platform 5, S = 1) |
| top | `rtl/boards/sqrl-fk33/otpu_fpga_top_fk33.sv` |
| block design | `boards/sqrl-fk33/vivado/bd.tcl`: XDMA (Gen3 x8 on PCIE4C_X1Y0), clocks, SYSMON, BAR0 and DMA interconnects |
| memory | `otpu_hbm.sv` (the HBM IP, all 32 AXI ports as arrays) and one `otpu_hbm_port.sv` per channel |
| build | `boards/sqrl-fk33/run_vivado.sh` (Vivado 2019.2), with this board's `vivado/build.tcl`, `scripts/preflight.tcl` and `vivado/tmem_uram.xdc` |

## The design

One slice (`otpu_fk33_sys` at S = 1, a copy of the BCU1525's `otpu_bcu_sys`: the accelerator and XDMA's DMA
meeting in one `otpu_mem_ch` per channel) on two channels of the first HBM stack: channel j is HBM bytes j x 2 GiB .. + 2 GiB, so
the slice's 32-bit DRAM addresses reach 4 GiB of the 8 GB. Each channel's `otpu_hbm_port` (the VCU128 build's HBM
adapter) turns otpu_mem_ch's two native controller ports into four AXI3 ports (NS = 2), port a = p + 2k on the IP's
AXI_(8j + 4k + p), each sub-port in its own mini-switch; AXI ports 0, 1, 4, 5, 8, 9, 12 and 13 are used, the rest
tied off. The global switches are on. The HBM releases the adapters when both stacks have calibrated
(`apb_complete_0/1`); a catastrophic temperature trip clears CALIB (otpu_board then refuses RUN and LOAD).

Clocks: the card has no clock input besides the PCIe reference, so one MMCM on XDMA's axi_aclk (250 MHz) makes
core_clk (250 MHz), uclk (250 MHz: the HBM's AXI ports, the adapters, otpu_mem_ch's controller side) and clk100 (the
HBM's reference and APB clock). A PCIe link reset restarts the HBM too.

PCIe identity: SQRL's, vendor 1E24, device 1533, subsystem 1E24:0001 (rev A3), class 12 00 00. BAR0 (1 MB): the
control registers at 0x0, the SYSMON at 0x30000; the DMA BAR is XDMA's (64 KB). XDMA's DMA reaches channel j at
j x 2 GiB.

Implementation: the slice, the shell and both channels share the one SLR and congest; placement spreads the logic
(AltSpreadLogic_high) and physical optimization and routing use AggressiveExplore. TMEM_URAM defaults to 1 here.
At 250 / 250 MHz (CORE_MHZ, UCLK_MHZ) the core's pipeline registers in the sequencer, DMA, activation RAM, control,
collective, trace, quantizer and fp units carry it: WNS +0.031 ns (core), +0.291 ns (uclk), WHS +0.010 ns. On an
FK33: selftest ALL PASS with the model stage for Qwen3.5-0.8B and -2B (int8) and -4B (`--wformat mix`: its int8 image
needs 4142 MiB of the slice's 4096).

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
