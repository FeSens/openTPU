# openTPU on the Inspur YPCB-00338

The board build: one openTPU slice (D = 128, 2 MXU columns, 8 VPU lanes, 64K-word TMEM) on a
Kintex-7 xc7k480t-ffg1156-2, with both DDR3 channels (2 x 2 GiB) behind Xilinx MIG
controllers, and the host PC over PCIe Gen1 x8 (Xilinx XDMA). The host compiles each token's
program, loads it and runs it; `otpu-chat --backend board` chats with Qwen3-0.6B (or LFM2.5-230M,
or Qwen3.5-0.8B) on it.

```
 host PC ── PCIe Gen1 x8 ── XDMA ──┬── AXI-Lite (BAR0) ─────────────── control registers ┐
                                   └── AXI4 128b @125 MHz ──┐                             │
                                                            SmartConnect ── MIG0 ── DDR3 CH0 (2 GiB)
               otpu_board (core_clk 100 MHz) ── m0 512b ──┤            └── MIG1 ── DDR3 CH1 (2 GiB)
                 slice + otpu_axi_dram       ── m1 512b ──┘
```

Files: `rtl/boards/ypcb-00338/` (otpu_fpga_top, otpu_board, otpu_ctrl),
`boards/ypcb-00338/` (constraints, Vivado Tcl, MIG generator, build and program scripts,
self-test), `opentpu/host/` (driver and tools, [host.md](host.md)).

## 1. Build the bitstream

Vivado 2026.1 with a license that covers the xc7k480t. The free edition does not include this
device: use a paid license or AMD's 30-day evaluation license (see "License" below).

```sh
cd boards/ypcb-00338
make lint          # offline: MIG pin check, Tcl syntax, XDC vs top ports, Verilator lint
make bit           # = ./run_vivado.sh 800 -> build/vivado/otpu.bit, otpu.mcs, reports/
make bit DDR=1066  # DDR3-1066 (533 MHz, MIG ui_clk 133 MHz) once 800 works
make bit CORE_MHZ=80   # accelerator clock fallback when 100 MHz does not close (800/D MHz, D in 1/8 steps)
make bit MCOLS=4   # 4 MXU columns: ~1.7x prefill and batched decode, ~67% LUT (the host
                   # reads MCOLS and LANES from the bitstream's VERSION register)
make bit VPU_CL=4  # 4 VPU lanes with exp2/recip/rsqrt (2 by default): ~75% -> ~89% of the
                   # roofline on long-context attention; timing only, programs unchanged
make bit LANES=16  # 16 VPU lanes / TMEM banks (the MXU and quantizer stay on 8): Qwen3.5 -4%
                   # cycles at 80% bw, -13% at 100% (simulated). Does not route on the xc7k480t
                   # (measured, MCOLS=4 VPU_CL=2 with the r3-route area cuts: 212K LUT placed,
                   # route_design stops at global congestion level 6); kept for larger parts
```

`run_vivado.sh` runs `scripts/gen_mig_prj.py` (MIG configuration from the board pin lists),
`vivado/create_project.tcl` (project, block design `vivado/bd.tcl`, constraints) and
`vivado/build.tcl` (synthesis, implementation with post-route phys_opt, reports, bitstream,
BPI flash image). Expect 1.5-3 h. Look at `build/vivado/reports/SUMMARY.txt` first: WNS/WHS and
the achieved frequency per clock; then `timing_summary.rpt`, `util_hier.rpt`, `cdc.rpt`.

Measured (Vivado 2026.1, 2026-09-24; default build: MCOLS=2, core 100 MHz, DDR3-800, PCIe Gen1
x8): all timing constraints met, WNS +0.082 ns, WHS +0.016 ns. Utilization: 187,852 LUT
(62.9%), 126,679 FF (21.2%), 635 BRAM36 tiles (66.5%), 267 DSP48 (13.9%). Vivado's power
estimate is 8.75 W (low confidence: no switching activity supplied). About 3 h with `JOBS=1`
in Docker on a 16 GB Apple Silicon Mac (4 jobs ran out of memory).

`make bit MCOLS=4 VPU_CL=4` (measured, same tools and date): all timing constraints met, WNS
+0.003 ns, WHS +0.012 ns (no margin: expect some builds of this configuration to miss by a few
ps; try `IMPL_STRATEGY=Performance_Explore`). 211,629 LUT (70.9%), 144,722 FF (24.2%), 668 BRAM36
tiles (70.0%), 443 DSP48 (23.1%); power estimate 9.40 W (low confidence).

With the DMA chunk buffer (eb29dd3) and the RDOT / OUTER / LOG2 VPU ops (ddec900), the same
`make bit MCOLS=4 VPU_CL=4` (measured, 2026-09-25): all timing constraints met, WNS +0.028 ns,
WHS +0.016 ns; 213,391 LUT (71.5%), 147,522 FF (24.7%), 690.5 BRAM36 tiles (72.3%), 459 DSP48
(23.9%); power estimate 9.70 W (low confidence).

Default build of 3c270c9 (MCOLS=2, VPU_CL=2, LANES=8; rotator TMEM, drain fix; measured,
2026-09-25): all timing constraints met, WNS +0.065 ns, WHS +0.038 ns; 184,124 LUT (61.7%),
657.5 BRAM36 tiles (68.9%), 283 DSP48 (14.7%); power estimate 8.99 W (low confidence).

In all the builds above the MXU's 1K x 128-byte chunk FIFO was not in block RAM: Vivado had
absorbed its read register into the DSP inputs and built it from 5,472 RAM64M (~22K LUTs; synthesis
warning `Infeasible attribute ram_style = "block"`). Since 54045fa it is a block RAM module of its
own (29 BRAM36), and since 617fffb the TMEM keeps 6 read copies instead of 8.

Default build of 74d4859 (MCOLS=2, VPU_CL=2, LANES=8; chunk FIFO in block RAM, 6 TMEM copies;
measured, 2026-09-26): all timing constraints met, WNS +0.104 ns, WHS +0.038 ns; 155,965 LUT
(52.2%, -28K), 558 BRAM36 tiles (58.4%, -99.5), 283 DSP48 (14.7%); power estimate 8.89 W (low
confidence).

### Vivado on Apple Silicon (Docker + Rosetta)

Vivado is x86-64 Linux/Windows only. On an M-series Mac:

1. Docker Desktop -> Settings -> General: "Use Rosetta for x86_64/amd64 emulation"; give it
   >= 32 GB RAM, >= 8 CPUs, ~150 GB disk.
2. Build an image with Vivado installed (Ubuntu 22.04 amd64 base; the AMD unified installer in
   batch mode, `xsetup -b Install -a XilinxEULA,3rdPartyEULA -c install_config.txt`, edition
   "Vivado ML Standard", devices: Kintex-7 only to save space). Downloading the installer needs
   your AMD account login (do it yourself in the browser).
3. Install into a Docker volume so the image stays small, e.g. `xilinx-2026.1` mounted at
   `/opt/Xilinx`, then:

```sh
VIVADO_DOCKER=vivado:2026.1 VIVADO_MOUNT=xilinx-2026.1:/opt/Xilinx \
VIVADO_SETTINGS=/opt/Xilinx/2026.1/Vivado/settings64.sh \
VIVADO_MAC=02:42:0a:7b:00:01 XILINXD_LICENSE_FILE=$HOME/.Xilinx/otpu.lic \
JOBS=4 make bit
```

### License

Without a license Vivado 2026.1 stops at start-up ("a valid license was not found"). A node-locked
license is tied to a host ID, the Ethernet MAC address. Docker gives each container a new MAC, so
pick one and pass it as `VIVADO_MAC`; `run_vivado.sh` starts every container with it.

1. At AMD's Product Licensing site (your AMD login), generate a node-locked license for the
   Vivado Enterprise edition (the 30-day evaluation is enough for bring-up). Host ID: the MAC
   without colons, e.g. `02420a7b0001` for `02:42:0a:7b:00:01`.
2. Save the `.lic` file and pass its path as `XILINXD_LICENSE_FILE`.

Rosetta runs Vivado at roughly half native speed; a Linux x86 box is faster if one is at hand.
JTAG does not go through Docker: program from macOS with openFPGALoader (below).

## 2. Program the FPGA

### Which bitstream to load

The host needs no configuration for a bitstream: it reads D / MCOLS / LANES from the VERSION
register and builds its configuration from them (`opentpu.host.board.device_config`). VPU_CL is
timing only and not reported. What differs between builds is the instruction set: Qwen3.5 needs
the RDOT / OUTER / LOG2 VPU functions (commit ddec900); a bitstream built before them runs Qwen3
and LFM2 only, and the self-test's `vops` stage says so.

| Bitstream | Build | RTL | VERSION | RDOT / OUTER / LOG2 | Models | Timing |
|---|---|---|---|---|---|---|
| **`build/deploy_r3route_74d4859/otpu.bit`** (primary) | `make bit` (MCOLS=2, VPU_CL=2, LANES=8) | 74d4859 (chunk FIFO in block RAM, 6 TMEM copies) | D=128 MCOLS=2 LANES=8 | yes | Qwen3, LFM2, Qwen3.5 | met, WNS +0.104 ns, WHS +0.038 ns |
| `build/deploy_default_3c270c9/otpu.bit` (first fallback) | `make bit` (MCOLS=2, VPU_CL=2, LANES=8) | 3c270c9 | D=128 MCOLS=2 LANES=8 | yes | Qwen3, LFM2, Qwen3.5 | met, WNS +0.065 ns, WHS +0.038 ns |
| `build/vivado_100mhz_m4cl4_vops_met/otpu.bit` (fallback) | `make bit MCOLS=4 VPU_CL=4` | ddec900 (DMA chunk buffer + new VPU ops) | D=128 MCOLS=4 LANES=8 | yes | Qwen3, LFM2, Qwen3.5 | met, WNS +0.028 ns |
| `build/vivado_100mhz_gen1_met/otpu.bit` (fallback) | `make bit` (MCOLS=2, VPU_CL=2, LANES=8) | v0.4 (6587cb4) | D=128 MCOLS=2 LANES=8 | no | Qwen3, LFM2 | met, WNS +0.082 ns |

Start with the primary image; if it misbehaves where the 3c270c9 image does not, the chunk FIFO
/ TMEM change is the suspect (same programs, same cycles in simulation). The 4&4 build is the same instruction set with twice the MXU
columns (the host picks MCOLS=4 up from VERSION); the v0.4 build is the last resort. All are
100 MHz core, DDR3-800, PCIe Gen1 x8, register map 2. The RTL changes after ddec900
(TMEM rotators, LANES=16 option, MXU drain, chunk FIFO in block RAM, 6 TMEM copies) change timing or area only, not results. The
`.mcs` next to each `.bit` is the BPI flash image of the same build. The self-test's config stage and
`otpu-smi` print the loaded image's BUILD_ID (the first 8 hex digits of the commit checked out
at build time: `74d48591` for the primary, `3c270c93` for the first fallback), so you can tell which image is on the card.

### Load it over JTAG

```sh
cd boards/ypcb-00338
make program BIT=../../build/deploy_r3route_74d4859/otpu.bit         # openFPGALoader (5 retries)
make program-vivado BIT=$PWD/../../build/deploy_r3route_74d4859/otpu.bit   # Vivado hw_manager
make flash MCS=../../build/deploy_r3route_74d4859/otpu.mcs          # permanent: BPI flash
```

Without `BIT=` / `MCS=` the scripts take `build/vivado/otpu.bit` / `otpu.mcs` (the last build).
`program.sh` takes the same file as its argument (`./program.sh [--vivado|--flash] [file]`).

- **openFPGALoader** (macOS or Linux) with a Xilinx Platform Cable USB II: the JTAG chain has an
  Inspur CPLD (IDCODE 0x10931093) in front of the FPGA; `program.sh` declares it
  (`--misc-device`, `--index-chain 0`). The cable needs its FX2 firmware on every plug-in
  (`XUSB_FIRMWARE`, default the inspur-adventures copy). See the `ypcb-00338` / `xpcu-macos`
  skills for the macOS cable quirks; after "Unable to read constant" on every attempt, replug.
- **Vivado hardware manager** (`--vivado`): runs `vivado -mode batch` on the machine with the
  cable (hw_server local). Give it an absolute path.

Programming over JTAG does not survive a power cycle; the flash does.

### After programming: PCIe

A PCIe device must be up within ~100 ms of power; a JTAG load is much later, so the host has
to rescan after loading. With the card in the Linux PC (powered by it) and the JTAG cable on
it, program, then on the PC:

```sh
opentpu/host/setup_pcie.sh --rescan      # remove + rescan the device, (re)load the driver, ID check
# or by hand:
sudo sh -c 'echo 1 > /sys/bus/pci/rescan'
lspci -d 10ee: -nn -vv    # expect: [10ee:7028], LnkSta: Speed 2.5GT/s, Width x8
```

Expect the link at Gen1 x8 (2.5 GT/s, `LnkCap` also 2.5GT/s x8): the XDMA is configured for
Gen1 (section 5), so 2.5 GT/s is not a downtrained link. Device ID 7028 is set in the block
design (Xilinx's default for a 7-series Gen2 x8 core, 7018 would be Gen1 x8; both are in the
XDMA driver's table, so the driver binds either way).

If the device does not appear, warm-reboot the host (the FPGA keeps its configuration across a
warm reboot, as long as the slot power stays on) -- or write the flash (`make flash`), then
power-cycle: the FPGA configures from flash at power-up in time for enumeration. If the link
trains at a lower width, check `LnkSta` and the PCIe placement note in section 6.

## 3. Host driver (Linux PC)

`opentpu/host/setup_pcie.sh` does all of this ([host.md](host.md) sections 2-3). By hand:

```sh
git clone https://github.com/Xilinx/dma_ip_drivers
cd dma_ip_drivers/XDMA/linux-kernel/xdma && make && sudo make install   # to /lib/modules/$(uname -r)/xdma
sudo modprobe xdma                     # XDMA_POLL=1 setup_pcie.sh / modprobe xdma poll_mode=1: no interrupts
ls /dev/xdma0_*                        # xdma0_user, xdma0_h2c_0, xdma0_c2h_0, ...
```

`/dev/xdma0_user` is BAR0 (the control registers), `/dev/xdma0_h2c_0` / `_c2h_0` move data
to / from the DDR3 at the file offset = AXI address (MIG0 at 0, MIG1 at 0x8000_0000). The
accelerator's logical DRAM is interleaved over the two channels in 64-byte beats;
opentpu/host/board.py applies the map (never write the channels directly except in the
self-test).

Python on the host: `pip install -e . torch transformers safetensors` (the `otpu-*`
commands), and the checkpoints in `models/` (`Qwen3-0.6B`, `LFM2.5-230M`, `Qwen3.5-0.8B`).

## 4. Self-test, then chat

```sh
otpu-selftest                                 # stages link .. vops (docs/host.md section 4)
otpu-selftest --model qwen3 --tokens 8        # plus the model stage (lfm2, qwen35)
otpu-chat --backend board                     # chat with Qwen3-0.6B on the card
otpu-chat --backend board --model lfm2        # LFM2.5-230M; --model qwen35 needs the vops bitstream
otpu-smi                                      # the card's state (from another terminal)
```

No `OTPU_MCOLS` / `OTPU_LANES`: the tools follow the bitstream. If either is set in the shell
and disagrees with the bitstream, they stop with a message naming both.

Verified so far only on the Verilator board model (`otpu-selftest --sim`, the current RTL; no
card yet): every stage passes, and the model stage matches the ISA simulator token for token.
What the model cannot show: MIG calibration, the controllers' read-modify-write of partial
writes (the model applies byte strobes directly), PCIe, the DMA rate and the real DRAM latency.

## 5. Clocks and the roofline

| Clock | Frequency | Source | Drives |
|---|---|---|---|
| sys_clk_50 | 50 MHz | AA28 oscillator | MMCM (VCO 800 MHz) |
| core_clk | 100 MHz | MMCM /8 | accelerator, control registers, interconnect core side |
| clk_200 | 200 MHz | MMCM /4 | MIG reference (IDELAYCTRL); MIG system clock at DDR3-800 |
| clk_267 | 266.667 MHz | MMCM /3 | MIG system clock at DDR3-1066 |
| ui_clk0/1 | 100 MHz (133 MHz at 1066) | MIG | MIG AXI side, 512 bit |
| DDR3 CK | 400 MHz (533 MHz) | MIG PLL | memory |
| axi_aclk | 125 MHz | XDMA | PCIe AXI side, 128 bit (Gen1 x8) |

Decode is DRAM-bound: every token streams all weights once. The accelerator consumes one
128-byte chunk per core cycle at its peak; each DDR3 channel (x64) delivers 8 bytes per CK edge.

| DDR3 | channel peak | both channels | core clock to match (128 B/cycle) | MIG ui_clk |
|---|---|---|---|---|
| 800 (default) | 6.4 GB/s | 12.8 GB/s | >= 100 MHz | 100 MHz |
| 1066 | 8.5 GB/s | 17.1 GB/s | >= 133 MHz | 133 MHz |

So the fmax each part must clear to stay at the roofline at DDR3-800: core_clk >= 100 MHz
(accelerator, otpu_axi_dram, control), MIG ui_clk 100 MHz (fixed by the MIG), SmartConnect
paths at their own clocks (100 / 125 MHz), XDMA 125 MHz (fixed by the IP at Gen1 x8). Real DDR3
efficiency (refresh, row misses, read/write turnaround) is ~70-85 %, which the accelerator's
deep prefetch absorbs; a core clock above 100 MHz buys nothing at DDR3-800. Host transfers
per token are small (program ~40 KB, logits 600 KB): the one-time weight upload (~820 MB) takes
~0.5-0.7 s at Gen1 x8 (2 GB/s). Gen1 rather than Gen2: at Gen2 the PCIe block runs a
500 MHz user clock whose IP-placed paths missed timing by ~0.1 ns (Vivado 2026.1, 80 MHz build).

## 6. What to check on first build (assumptions made without Vivado)

1. **MIG configuration** (`vivado/mig/mig_ddr3_ch*.prj`, generated): 9 x MT41K256M8DA-125 per
   channel, 72-bit with **ECC enabled**, no data mask, DDR3-800, 4:1, AXI 512-bit, internal
   VREF (valid up to 800 Mb/s), DCI termination (default), address map BANK_ROW_COLUMN.
   - Why ECC: the board has **no DM pins** (none in the pin lists), and the accelerator does
     byte and masked-word writes (QST, DMA ST). Without DM, MIG ignores write strobes and would
     corrupt neighbouring bytes; with ECC, MIG handles partial writes by read-modify-write, and
     the ninth byte lane (present on the board) holds the ECC. If ECC must be turned off (a bad
     ninth lane), partial writes need a read-modify-write in otpu_axi_dram instead.
   - The byte groups were checked offline (`gen_mig_prj.py --check`): all 9 lanes of both
     channels have DQ and DQS in one byte group, DQS on the DQS-capable pair, 3 contiguous HP
     banks per channel, address/command in the middle bank, reset_n in a data bank.
   - If Vivado rejects the .prj (schema differences between MIG versions): open the MIG IP in
     the block design, "Create Design" with the settings above, "Fixed Pin Out" -> "Read
     XDC/UCF" -> `vivado/mig/mig_ddr3_ch<N>_pins.xdc` -> Validate -> finish; then export the
     .prj it writes over the generated one.
   - The DDR3 timing parameters in the .prj are the MT41K256M8-125 datasheet values; MIG's
     own part database (`MT41K256M8XX-125`) takes precedence.
2. **DDR3 calibration** (`STATUS` bits 5/6, or the green LED): the older in-house controller
   saw a stuck byte lane (CH0 physical lane 3) and capture trouble on CH1 lanes 6-7. MIG
   calibrates per lane; if a channel does not calibrate, read the MIG calibration status via
   the MIG debug signals (set `Debug_En` ON in the .prj) to find the lane.
3. **PCIe placement**: the lanes are on the GTX pins F2 H2 K2 M2 N4 P2 T2 U4 (TX, lane 0..7:
   banks 116 then 115), refclk on J8 (MGTREFCLK0_116). The XDMA's default GT sites are one quad
   lower, so `constraints/otpu_top.xdc` LOCs lane i to `GTXE2_CHANNEL_X0Y(23-i)`. If the link
   comes up narrower than x8 or not at all, suspect the lane order first (`lspci -vv`, LnkSta).
   PERST# is Y26 (LVCMOS18, pulled up).
4. **Reset**: the board reset pin R28 is not wired; the design resets from the MMCM lock.
5. **Configuration**: CFGBVS GND / 1.8 V, BPI x16 flash (A1..A25, 64 MB), compressed bitstream.
6. **LED polarity** is unverified: led[0] heartbeat, led[1] PCIe link up and both channels
   calibrated, led[2] the accelerator runs / halted cleanly.
7. **Timing**: the accelerator was timed with yosys only; if core_clk fails at 100 MHz,
   `reports/timing_worst.rpt` names the paths; the MIG and XDMA domains are fixed by the IPs.
   To get a working board first, rebuild with `make bit CORE_MHZ=80` (or 75): decode is
   DRAM-bound, so 80 MHz loses little (the adapter issues one 64-byte beat per channel per
   cycle, 80 MHz x 128 B = 10.2 GB/s against DDR3-800's 12.8 GB/s peak).
