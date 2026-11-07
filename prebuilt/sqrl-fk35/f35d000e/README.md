# SQRL FK35, openTPU build f35d000e (PCIe Gen4 x8, two slices, 250 MHz)

The FK35 (Virtex UltraScale+ HBM xcvu35p-fsvh2104-2-e, 8 GB HBM2) with two DUAL + QUAD slices at 250 MHz and XDMA at
PCIe Gen4 x8 (512-bit AXI at 250 MHz). It replaces the Gen3 x8 build f35d000b with the same accelerator: target
`targets/sqrl-fk35.toml`, architecture hash 0x0E066748. Built and tested 2026-11-07.

## Build

| | |
|---|---|
| Sources | this repository at dd636eb ("openTPU chat") plus the change committed with this folder: the HBM side's reset from a register (`rtl/boards/sqrl-fk35/otpu_fpga_top_fk35.sv`) and the XDMA pblock `pb_xdma` (`boards/sqrl-fk35/constraints/fk35_floorplan.tcl`) |
| Command | `PREFLIGHT=0 BUILD_ID=f35d000e JOBS=8 IMPL_STRATEGY=Performance_ExtraTimingOpt ./boards/sqrl-fk35/run_vivado.sh` (PCIE_GEN 4, the default), Vivado 2019.2.1 |
| BUILD_ID | 0xF35D000E |
| Clocks | core_clk 250 MHz, uclk (HBM AXI) 250 MHz, XDMA axi_aclk 250 MHz, clk100 |
| Memory | HBM2 8 GB, 8 channels of 1 GiB (DUAL: 4 a slice), 1800 MT/s |
| Layout | memory side and XDMA in SLR0 (XDMA + sc_dma's 512-bit entry in `pb_xdma`, clock regions X6Y0:X7Y3 + X3Y3:X5Y3, at the PCIe block PCIE4CE4_X1Y0), slices and shell in SLR1 |
| Timing | **met**: WNS +0.002 ns, WHS +0.004 ns. core_clk +0.002, uclk +0.007, xdma_aclk +0.017, pipe_clk +0.026 ns (`reports/SUMMARY.txt`, `reports/timing_summary_tables.rpt`) |
| Utilization | SLR0: 70.3 % of its LUTs, 95.2 % of its CLBs; SLR1: 51.2 % / 83.3 % (`reports/util*.rpt`) |
| Implementation | placement WNS +0.073 ns; routing reached -0.063 ns over its rip-up iterations and +0.002 ns after the router's clock-delay optimization |

The two builds before it, not released: f35d000c (Gen4 without the pblock) failed timing at -0.524 ns; f35d000d met
timing but put an `axi_dwidth_converter` ahead of sc_dma, which 2019.2's SmartConnect does not follow (its S00 stays
512 bits with an ID), so every DMA failed. See `docs/sqrl-fk35.md`.

## Files

| File | Bytes | SHA-256 |
|---|--:|---|
| `otpu.bit` | 47,931,303 | `43f49fb2ed3e0d083026085fc6ce95358e336f8943eea8f62dd39bceaa77e934` |
| `otpu_primary.mcs` | 67,409,592 | `07641ea62e733aaeea0440af860bce2d64ebc69393d47ef35d072adaa5cba2f5` |
| `otpu_primary.prm` | 394 | `a5c66aabd97fb17bbaca98cfb13644b6b5026573fccf29f4738b8b6e65e2141d` |
| `otpu_secondary.mcs` | 67,409,592 | `8ec0a5ef67fe4caf69af94be571042d27c074f236c5d0d3e9f53ae2514f1704f` |
| `otpu_secondary.prm` | 394 | `a5c66aabd97fb17bbaca98cfb13644b6b5026573fccf29f4738b8b6e65e2141d` |
| `otpu.ltx` | 1,040 | `1bfa8aa092987ad4bcc2634442a92bf2094206bd7d8442d86bb2d2d0bdf9a7f3` |
| `fk35_flash.ps1` | 5,623 | `8f1d3e91161f3474d45cdc684182f4b01774e009337c15f844e6bcf5c676ed5c` |
| `reports/` | | SUMMARY, utilization (flat and per SLR), congestion, clock interaction, power, the timing summary's tables |

- **Flash:** `otpu_primary.mcs` and `otpu_secondary.mcs`, each with its `.prm`: the board's two MT25QU256 in dual QSPI
  (SPIx8). One plain bitstream, no Tandem stages.
- **JTAG (volatile):** `otpu.bit`, with `otpu.ltx` for the debug hub.

## Use

1. As administrator: `.\fk35_flash.ps1 -Dir <this folder>`. It disables the card, programs both flashes, boots from
   flash over JTAG, rescans, restarts the root port if BAR0 reads stale and prints the ID registers (`ID 0x4F545055`,
   `BUILD_ID 0xF35D000E`). `-Vivado` and `-Tool` (sqrl_fk_mcap.exe of the SQRL_FK33 Windows driver, 2.1 or later)
   point at the installations.
2. **Restart Windows (Start > Restart, not Shut down or sleep) for Gen4.** After the flash the link comes back at
   Gen3 x4; only a full restart trains Gen4 x4 in the test PC. Retraining while Windows runs (rescan, root-port restart,
   sleep, fast startup) left it at Gen3.
3. The host: target `sqrl-fk35`, the Windows transport `SqrlFkTransport` (device `sqrlfk0`); `opentpu.host.selftest`,
   `opentpu.host.chat --backend board`, `otpu-smi --dev sqrlfk0`.

## Tested on the board (2026-11-07)

Test PC: the card in the x4 slot behind the PCH root port 0:1B.4 (Intel 7AC4).

| | Gen3 x4 (after the flash) | Gen4 x4 (after a full restart) |
|---|---|---|
| Self-test (Qwen3.5-0.8B model stage) | ALL PASS | ALL PASS |
| Self-test bandwidth, host -> card / card -> host | 2.09 / 2.83 GB/s | 2.18 / 2.89 GB/s |
| sqrl_fk_mcap bench, direct requests, 64 MB | | 2.49 / 3.19 GB/s |
| sqrl_fk_mcap loop, HBM stacks 0 and 1 | | 2.63 / 3.39 GB/s, data OK |
| Qwen3.5-9B fp4 decode (tools/qual/perf.py) | 24.48 tok/s | 24.48 tok/s |
| Qwen3.5-9B mix decode | 21.97 tok/s | |
| Corrected PCIe errors (WHEA, root port 0:1B.4) | none | none, through the tests |

- Chat (Qwen3.5-9B fp4, greedy, through the fk35-chat launcher with its caches): loaded in 15.4 s, 24.35 tok/s
  (24.6 on the device), time to first token 1.27 s.
- Decode speed is the same as f35d000b: it is set by the HBM stream (107 GB/s), not PCIe.
- Gen4 adds little DMA rate: one transfer goes to one channel's 128-bit DMA port (4 GB/s at 250 MHz) and through
  `otpu_mem_ch`'s XDMA side; the link carries more.
