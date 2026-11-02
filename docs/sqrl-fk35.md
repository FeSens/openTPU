# openTPU on the SQRL FK35

The SQRL Forest Kitten 35: Virtex UltraScale+ HBM xcvu35p-fsvh2104-2-e (two SLRs), 8 GB HBM2 in two stacks, PCIe
(GTY lanes of an x16 edge connector, 8 used), two MT25QU256 configuration flashes (SPI x8). It is the FK33's PCB
(docs/sqrl-fk33.md) with the bigger FPGA: twice the logic, so two slices and all 8 GB.

| | |
|---|---|
| target | `targets/sqrl-fk35.toml` (platform 4, S = 2) |
| top | `rtl/boards/sqrl-fk35/otpu_fpga_top_fk35.sv` |
| block design | `boards/sqrl-fk35/vivado/bd.tcl`: XDMA (Gen3 x8 on PCIE4C_X1Y0), clocks, SYSMON, BAR0 and DMA interconnects |
| memory | `otpu_hbm.sv` (the HBM IP, all 32 AXI ports as arrays) and one `otpu_hbm_port.sv` per channel |
| floorplan | `boards/sqrl-fk35/constraints/fk35_floorplan.tcl`: the memory side in SLR0, the slices and the shell in SLR1 |
| build | `boards/sqrl-fk35/run_vivado.sh` (Vivado 2019.2), with this board's `vivado/build.tcl`, `scripts/preflight.tcl` and `vivado/tmem_uram.xdc` |

## The design

Two slices (`otpu_fk35_sys` at S = 2, a copy of the BCU1525's `otpu_bcu_sys`: the accelerator and XDMA's DMA
meeting in one `otpu_mem_ch` per channel), each on two channels: board channel j = 2s + c is HBM bytes j x 2 GiB ..
+ 2 GiB, so slice 0 has the first stack and slice 1 the second, and the slices' 32-bit DRAM addresses (4 GiB each)
reach all 8 GB. A model's weights are split between the slices (tensor parallel), so a model whose image needs up to
about 8 GB runs. Each channel's `otpu_hbm_port` (the VCU128 build's HBM adapter) turns otpu_mem_ch's two native
controller ports into four AXI3 ports (NS = 2), port a = p + 2k on the IP's AXI_(8j + 4k + p), each sub-port in its
own mini-switch; AXI ports 0, 1, 4, 5, .., 28 and 29 are used, the rest tied off. The global switches are on. The
HBM releases the adapters when both stacks have calibrated (`apb_complete_0/1`); a catastrophic temperature trip
clears CALIB (otpu_board then refuses RUN and LOAD).

SLRs: the HBM and the PCIe block are in SLR0, so the memory side (the four otpu_mem_ch, their HBM adapters) stays
there with XDMA, and otpu_board (both slices and the shell) goes to SLR1. Every channel's native port crosses
between them through two `otpu_native_reg` (SLL = 4'b1111: u_tx with the slice, u_rx with the channel), flip-flop to
flip-flop in the Laguna registers; the core reset crosses as a register per side.

Clocks: the card has no clock input besides the PCIe reference, so one MMCM on XDMA's axi_aclk (250 MHz) makes
core_clk (250 MHz), uclk (250 MHz: the HBM's AXI ports, the adapters, otpu_mem_ch's controller side) and clk100 (the
HBM's reference and APB clock). A PCIe link reset restarts the HBM too.

PCIe identity: SQRL's, vendor 1E24, device 1533, subsystem 1E24:0001 (rev A3), class 12 00 00. BAR0 (1 MB): the
control registers at 0x0, the SYSMON at 0x30000; the DMA BAR is XDMA's (64 KB). XDMA's DMA reaches channel j at
j x 2 GiB (8 GiB in all).

Implementation: the FK33's directives, placement spreading the logic (AltSpreadLogic_high) and physical optimization
and routing with AggressiveExplore. TMEM_URAM defaults to 1 here.

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
