# openTPU on the Inspur YPCB-00338

The board build: one openTPU slice (D = 128, 4 MXU columns with the systolic MXU, 8 VPU lanes,
64K-word TMEM) on a Kintex-7 xc7k480t-ffg1156-2, with both DDR3 channels (2 x 2 GiB) behind the
LiteDRAM core ([litedram.md](litedram.md)), and the host PC over PCIe Gen1 x8 (Xilinx XDMA). The host compiles each token's
program, loads it and runs it; `otpu-chat --backend board` chats with Qwen3-0.6B (or LFM2.5-230M,
or Qwen3.5-0.8B) on it.

```
 host PC ── PCIe Gen1 x8 ── XDMA ──┬── AXI-Lite (BAR0) ── control registers, LiteDRAM CSRs, XADC
                                   └── AXI4 128b @125 MHz ── otpu_axi_split2 ──┐ (bit 31: channel)
                                                                               │
    otpu_board (core_clk) ── native 512b ch0 ── otpu_mem_ch ── LiteDRAM ch0 ── DDR3 CH0 (2 GiB)
      slice + otpu_native_dram ── ch1 ───────── otpu_mem_ch ── LiteDRAM ch1 ── DDR3 CH1 (2 GiB)
```

Files: `rtl/boards/ypcb-00338/` (otpu_fpga_top_ld, otpu_native_sys, otpu_mem_ch, otpu_board,
otpu_ctrl), `boards/ypcb-00338/` (constraints, Vivado Tcl, the generated LiteDRAM core, build and
program scripts, self-test), `opentpu/host/` (driver and tools, [host.md](host.md)).

The MIG builds (the two MIGs' AXI ports behind a SmartConnect, `otpu_axi_dram`, `bd.tcl`; and
the parked native-MIG build, `otpu_mig_native`) were the production images until the LiteDRAM
build qualified on the card, and were then removed (their history below stays as it was
written; the images in the table of section 2 still load and run: the host follows CAPS).

## 1. Build the bitstream

Vivado 2026.1 with a license that covers the xc7k480t. The free edition does not include this
device: use a paid license or AMD's 30-day evaluation license (see "License" below).

```sh
cd boards/ypcb-00338
make lint          # offline: Tcl syntax, XDC vs top ports, Verilator lint of the top
make bit           # = MCOLS=4 ./run_vivado.sh 1066 -> build/vivado/otpu.bit, otpu.mcs, reports/.
                   # The LiteDRAM core (WL7DDRPHY, DDR3-1066, the speed it is generated for; host
                   # calibration; per channel behind otpu_mem_ch, which also takes XDMA's
                   # traffic; docs/litedram.md section 9), 4 MXU columns and the systolic MXU
                   # (docs/mxu_systolic.md; the host reads MCOLS and LANES from the bitstream's
                   # VERSION register)
make bit FAST=1    # the same as a development build at 100 MHz (below)
make bit MCOLS=2   # 2 MXU columns
make bit CORE_MHZ=80   # accelerator clock fallback when 100 MHz does not close (800/D MHz, D in 1/8 steps)
make bit VPU_CL=4  # 4 VPU lanes with exp2/recip/rsqrt (2 by default): ~75% -> ~89% of the
                   # roofline on long-context attention; timing only, programs unchanged
make bit LANES=16  # 16 VPU lanes / TMEM banks (the MXU and quantizer stay on 8): Qwen3.5 -4%
                   # cycles at 80% bw, -13% at 100% (simulated). Does not route on the xc7k480t
                   # (measured, MCOLS=4 VPU_CL=2 with the r3-route area cuts: 212K LUT placed,
                   # route_design stops at global congestion level 6); kept for larger parts
```

`run_vivado.sh` runs `vivado/create_project.tcl` (project, block design `vivado/bd_native.tcl`,
constraints) and `vivado/build.tcl` (synthesis, implementation with post-route phys_opt, reports, bitstream,
BPI flash image). Expect 1.5-3 h. Look at `build/vivado/reports/SUMMARY.txt` first: WNS/WHS and
the achieved frequency per clock; then `timing_summary.rpt`, `util_hier.rpt`, `cdc.rpt`.

`FAST=1` is the development build for new architecture and IP work, where the function comes
before the clock. It uses the default CORE_MHZ=100, Vivado's default synthesis (no retiming) and
default implementation (no post-route phys_opt, no `impl_directives.tcl`). The block design's IP
synthesis is cached per host in `~/.cache/otpu-vivado-ip` (`OTPU_IP_CACHE`; empty turns it off).
Every build used to be a fresh project, so the IP runs never hit a cache (on the MIG builds they
took about 9 minutes of each build: the two MIGs, xdma and sc_mem about 3 minutes each).

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

4&4 build of 550aa35 (MCOLS=4, VPU_CL=4, LANES=8; same RTL as 74d4859; measured, 2026-09-26):
all timing constraints met, WNS +0.086 ns, WHS +0.037 ns; 179,081 LUT (60.0%, -34K against the
ddec900 4&4 build), 591 BRAM36 tiles (61.9%), 459 DSP48 (23.9%); power estimate 9.53 W (low
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
| **`build/deploy_secand3_02569bc/otpu.bit`** (production, 2026-09-29) | `make bit DDR=1066 CORE_MHZ=120.755` (AXI_BL=32 and AXI_WBL=8 are the defaults; MCOLS=2, LANES=8; built as SE=v2, now the only configuration) | se-cand3 02569bc: the stream engine v2 in the VPU (docs/stream.md; CAPS bit26 STREAM) + main 5cd6c39 + the three timing cuts below, 32-beat reads, 8-beat writes | D=128 MCOLS=2 LANES=8 | yes | Qwen3, LFM2, Qwen3.5 (int8 and 4-bit) | met, WNS +0.031 ns, WHS +0.016 ns |
| `build/deploy_pnbl32_e2521032/otpu.bit` (production 2026-09-28 afternoon until 2026-09-29) | `make bit DDR=1066 CORE_MHZ=120.755` (AXI_BL=32 and AXI_WBL=8 are the defaults; MCOLS=2, LANES=8) | prod-next e252101 (be388a1 + Qwen3.5 resident decode + DSTEP / ST write runs + AXI write bursts + the port-A order register), 32-beat reads, 8-beat writes | D=128 MCOLS=2 LANES=8 | yes | Qwen3, LFM2, Qwen3.5 (int8 and 4-bit) | met, WNS +0.013 ns, WHS +0.021 ns |
| `build/deploy_bl32mx120_be388a32/otpu.bit` (production 2026-09-28 11:00 until the afternoon) | `make bit DDR=1066 CORE_MHZ=120.755 AXI_BL=32` (MCOLS=2, LANES=8) | be388a1 (tv-cand2: main + port B read bursts up to 64 beats + the MXU / adapter timing fixes), 32-beat reads | D=128 MCOLS=2 LANES=8 | yes | Qwen3, LFM2, Qwen3.5 (int8 and 4-bit) | met, WNS +0.066 ns, WHS +0.016 ns |
| `build/deploy_bl16mx120_be388a1f/otpu.bit` (production 2026-09-28 morning) | as above, `AXI_BL=16` | be388a1, 16-beat reads | D=128 MCOLS=2 LANES=8 | yes | Qwen3, LFM2, Qwen3.5 (int8 and 4-bit) | met, WNS +0.048 ns, WHS +0.013 ns |
| `build/deploy_prod120hp_aebb0bf0/otpu.bit` (production 2026-09-28 until the morning) | `make bit DDR=1066 CORE_MHZ=120.755` (MCOLS=2, LANES=8) | aebb0bf (host-path: 4-bit MXU with PAIR, r7 DRAM path, replay, resident decode run arguments (CAPS bit25), DSTEP, VPU WBUF) | D=128 MCOLS=2 LANES=8 | yes | Qwen3, LFM2, Qwen3.5 (int8 and 4-bit) | met, WNS +0.004 ns, WHS +0.016 ns |
| `build/deploy_prod120fp4_ea3bc560/otpu.bit` (production 2026-09-27 evening) | `make bit DDR=1066 CORE_MHZ=120.755` (MCOLS=2, LANES=8) | ea3bc56 (4-bit MXU with PAIR, fmax fixes, VPU WBUF) | D=128 MCOLS=2 LANES=8 | yes | Qwen3, LFM2, Qwen3.5 (int8 and 4-bit) | met, WNS +0.149 ns, WHS +0.016 ns |
| `build/deploy_prod120_b01b8acb/otpu.bit` (production 2026-09-27 afternoon, int8 only) | `make bit DDR=1066 CORE_MHZ=120.755` (MCOLS=2, LANES=8) | b01b8ac (fmax fixes; section 5, "Faster DDR3") | D=128 MCOLS=2 LANES=8 | yes | Qwen3, LFM2, Qwen3.5 | met, WNS +0.080 ns, WHS +0.016 ns |
| `build/deploy_prod1066_b2c7ce43/otpu.bit` (previous production) | `make bit DDR=1066` at 100 MHz | b2c7ce4 | D=128 MCOLS=2 LANES=8 | yes | Qwen3, LFM2, Qwen3.5 | met, WNS +0.085 ns |
| `build/deploy_burst_a691ea98/otpu.bit` (older primary) | `make bit` (MCOLS=2, VPU_CL=2, LANES=8) | a691ea98 (port-B AXI read bursts, MXU_STARVE counter; register map 3) | D=128 MCOLS=2 LANES=8 | yes | Qwen3, LFM2, Qwen3.5 | met, WNS +0.085 ns, WHS +0.040 ns (omarchy build) |
| `build/deploy_r3route_74d4859/otpu.bit` (single-beat reads, 2.4x slower decode) | `make bit` (MCOLS=2, VPU_CL=2, LANES=8) | 74d4859 (chunk FIFO in block RAM, 6 TMEM copies) | D=128 MCOLS=2 LANES=8 | yes | Qwen3, LFM2, Qwen3.5 | met, WNS +0.104 ns, WHS +0.038 ns |
| `build/deploy_default_3c270c9/otpu.bit` (first fallback) | `make bit` (MCOLS=2, VPU_CL=2, LANES=8) | 3c270c9 | D=128 MCOLS=2 LANES=8 | yes | Qwen3, LFM2, Qwen3.5 | met, WNS +0.065 ns, WHS +0.038 ns |
| `build/deploy_m4cl4_550aa35/otpu.bit` (faster prefill) | `make bit MCOLS=4 VPU_CL=4` | 550aa35 (as the primary, 4 MXU columns, 4 composite VPU lanes) | D=128 MCOLS=4 LANES=8 | yes | Qwen3, LFM2, Qwen3.5 | met, WNS +0.086 ns, WHS +0.037 ns |
| `build/vivado_100mhz_gen1_met/otpu.bit` (fallback) | `make bit` (MCOLS=2, VPU_CL=2, LANES=8) | v0.4 (6587cb4) | D=128 MCOLS=2 LANES=8 | no | Qwen3, LFM2 | met, WNS +0.082 ns |

Start with the primary image; if it misbehaves where the 3c270c9 image does not, the chunk FIFO
/ TMEM change is the suspect (same programs, same cycles in simulation). The 4&4 build is the same instruction set with twice the MXU
columns (the host picks MCOLS=4 up from VERSION); the v0.4 build is the last resort. The rows
from the burst image down are 100 MHz core, DDR3-800, PCIe Gen1 x8. The RTL changes after ddec900
(TMEM rotators, LANES=16 option, MXU drain, chunk FIFO in block RAM, 6 TMEM copies) change timing or area only, not results. The
`.mcs` next to each `.bit` is the BPI flash image of the same build. The self-test's config stage and
`otpu-smi` print the loaded image's BUILD_ID (the first 8 hex digits of the commit checked out
at build time: `74d48591` for the primary, `3c270c93` for the first fallback, `550aa355` for the 4&4 image), so you can tell which image is on the card.

### Load it over JTAG

```sh
cd boards/ypcb-00338
make program BIT=../../build/deploy_burst_a691ea98/otpu.bit         # openFPGALoader (5 retries)
make program-vivado BIT=$PWD/../../build/deploy_burst_a691ea98/otpu.bit   # Vivado hw_manager
make flash MCS=../../build/deploy_burst_a691ea98/otpu.mcs          # permanent: BPI flash
```

Without `BIT=` / `MCS=` the scripts take `build/vivado/otpu.bit` / `otpu.mcs` (the last build).
`program.sh` takes the same file as its argument (`./program.sh [--vivado|--flash] [file]`).

- **openFPGALoader** (macOS or Linux), with either cable:
  - an FTDI FT232H adapter (Digilent HS2 style, USB 0403:6014): `CABLE=digilent_hs2 make
    program`. On the development PC this cable's chain shows the FPGA alone (`openFPGALoader
    -c digilent_hs2 --detect`: index 0, IDCODE 0x23751093, xc7k480t); `program.sh`'s CPLD
    declaration is then unused and `--index-chain 0` is the FPGA. The loads at first light
    were done with this cable (`openFPGALoader -c digilent_hs2`).
  - a Xilinx Platform Cable USB II (the default): the JTAG chain has an Inspur CPLD
    (IDCODE 0x10931093) in front of the FPGA, which `program.sh` declares (`--misc-device`,
    `--index-chain 0`). The cable needs its FX2 firmware on every plug-in (`XUSB_FIRMWARE`,
    default the inspur-adventures copy). See the `ypcb-00338` / `xpcu-macos` skills for the
    macOS cable quirks; after "Unable to read constant" on every attempt, replug.

  On Linux, `otpu-setup` installs the udev rule that lets a normal user open either cable.
  Arch: the `openfpgaloader` package (1.1.1 on the development PC; when the mirror lacks it,
  from archive.archlinux.org). Ubuntu 24.04 packages 0.12.0 and 22.04 none: build it from
  source (github.com/trabucayre/openFPGALoader) for a current version with the BPI bridge.
- **Vivado hardware manager** (`--vivado`): runs `vivado -mode batch` on the machine with the
  cable (hw_server local). Give it an absolute path.

Programming over JTAG does not survive a power cycle; the flash does.

### Write the flash (boot without JTAG)

`make flash MCS=...` (`program.sh --flash`) writes the `.mcs` of a build into the card's BPI
flash through the FPGA: openFPGALoader first loads its `bpiOverJtag_xc7k480tffg1156` bridge
(shipped with openFPGALoader 1.1.1 on the development PC), which replaces the running design,
so stop every program using the card first. After a power cycle the FPGA configures from flash
in time for enumeration: no JTAG and no rescan. Until then the card boots the image already in
flash: on the development PC that image enumerates as 10ee:7028 with class 05 80 00 and a 2 MiB
64-bit BAR (from dmesg at boot), is not openTPU, and `otpu-setup --check` says so. Not done on
the card yet: the flash still holds that image. The first attempt (2026-09-28, openFPGALoader
1.1.1) stopped before touching the flash: without `-b ypcb003381p1` it takes its SPI bridge,
and `program.sh` now passes the board entry. [flash.md](flash.md) has the Vivado route
(`flash_jtag.sh`: backup, program with verify, boot from flash), the byte order, and a proposal
for writing the flash over PCIe with a golden image.

### After programming: PCIe

A PCIe device must be up within ~100 ms of power; a JTAG load is much later, so the host has
to rescan after loading. With the card in the Linux PC (powered by it) and the JTAG cable on
it, program, then on the PC:

```sh
sudo otpu-setup --rescan     # remove + rescan the card, bind the driver, read the ID register
otpu-setup --check           # the whole host setup, the card and the link
```

Expect the link at Gen1 x8 (2.5 GT/s, `LnkCap` also 2.5GT/s x8): the XDMA is configured for
Gen1 (section 5), so 2.5 GT/s is not a downtrained link. Device ID 7028 is set in the block
design (Xilinx's default for a 7-series Gen2 x8 core; 7018 would be Gen1 x8; both are in the
XDMA driver's table, so the driver binds either way). The block design also sets the class
(12 00 00, processing accelerator), subsystem 10ee:4f54 and revision 01; bitstreams built
before that show class 07 00 01 (serial), subsystem 10ee:0007, revision 00. Verified on the card
(build b11bb679, 2026-09-27): `lspci -nn` shows `Processing accelerators [1200]: Xilinx
Corporation Device [10ee:7028] (rev 01)`, subsystem `[10ee:4f54]`, and the driver binds it
without the serial-port override.

If the device does not appear, `--rescan` retrains the link of the card's upstream port and
rescans again ([host.md](host.md) section 3). If it is still missing, warm-reboot the host (the
FPGA keeps its configuration across a warm reboot, as long as the slot power stays on), or write
the flash, then power-cycle. If the
link trains at a lower width, check `LnkSta` and the PCIe placement note in section 6.

## 3. Host setup and bring-up checklist

`otpu-setup` does the host side ([host.md](host.md) sections 2-3: the XDMA driver with DKMS,
the udev rules, the driver options). On a new PC, in order:

1. `pip install -e . torch transformers safetensors` (the `otpu-*` commands) and the
   checkpoints in `models/` (`Qwen3-0.6B`, `LFM2.5-230M`, `Qwen3.5-0.8B`).
2. `sudo otpu-setup`: builds and installs the driver, installs the rules. Once per PC, and
   again after a kernel upgrade if DKMS is not installed.
3. Load the bitstream: `make program` (JTAG), then `sudo otpu-setup --rescan`; or, once the
   flash holds it, just power up.
4. `otpu-setup --check`: exits 0 with "all in place" when driver, rules, card, link, device
   nodes and the ID register are right; otherwise it names what is not.
5. `otpu-selftest`, then `otpu-diag` and chat (section 4).

`/dev/xdma0_user` is BAR0 (the control registers), `/dev/xdma0_h2c_0` / `_c2h_0` move data
to / from the DDR3 at the file offset = AXI address (MIG0 at 0, MIG1 at 0x8000_0000). The
accelerator's logical DRAM is interleaved over the two channels in 64-byte beats;
opentpu/host/board.py applies the map (never write the channels directly except in the
self-test).

### The card's host: opentpu (since 2026-09-28)

Until 2026-09-28 the card was in omarchy, the development PC (Intel Core i5-12600KF), where every
measurement before that date was taken. It is now in opentpu (Intel Core i7-4790, Haswell). The
link trains at 2.5 GT/s x8 as before. On opentpu, `otpu-selftest` measures 1.26-1.34 GB/s host
to card and 0.84-1.00 GB/s card to host; omarchy measured 1.42 and 1.16 GB/s.

- **Root access** is narrow. `/etc/sudoers.d/60-otpu-card` allows exactly three commands, with
  no password:
  - `sudo -n /usr/local/sbin/otpu-rescan`;
  - `sudo -n lsof -t /dev/xdma0_*`;
  - `sudo -n dmesg`.

  Nothing else runs as root: `otpu-setup` and driver installs need the PC's owner.
  `tools/qual/qual.sh` uses these three.
- **`otpu-rescan`** runs a root-owned copy of `opentpu/host/setup_pcie.sh --rescan` (with
  `pcie/relink.sh`) from `/usr/local/lib/otpu/host`. `~/otpu-card-setup.sh` on that PC
  refreshes the copy from the repository. When the plain remove and rescan does not bring the
  card back, it retrains the upstream port's link (section 2, "After programming"). The retrain
  has not yet been tried on a real reload.
- **A JTAG reload**: after one, a hot rescan once did not bring the link back on opentpu, and a
  warm reboot did. So for now a reload there is planned with a warm reboot, and `qual.sh` runs
  with `LOAD=0` (it qualifies the bitstream the card already runs).
- **Flash**: it still holds the factory image. A cold power cycle loses the design; a warm
  reboot keeps it.
- **Commands**: the `otpu-*` commands are on `PATH` through symlinks in `~/.local/bin` to
  `~/otpu-venv/bin`.

The host path is slower there. Wall numbers from the two PCs are not comparable; section 5
compares them per token.

## 4. Self-test, diagnostic, then chat

Run order on the card:

1. `otpu-selftest` -- the staged check, a few seconds; it stops at the first failure with a
   hint. All PASS: go to chat.
2. `otpu-diag --json diag.json` -- when anything is off, or straight away for the full
   picture (about a minute). It runs every check whose prerequisites passed instead of
   stopping, and ends with a works / does-not-work matrix and diagnosis hints (byte lane, address
   bit, unit). Keep the JSON: it is the record of the card's state.
3. `otpu-diag --mem full --soak 10` -- the whole 4 GiB (march C-) and repeated kernels, for
   intermittent faults (several minutes).
4. Chat.

```sh
otpu-selftest                                 # stages link .. vops (docs/host.md section 4)
otpu-selftest --model qwen3 --tokens 8        # plus the model stage (lfm2, qwen35)
otpu-diag --json diag.json                    # everything, no stopping (docs/host.md section 5)
otpu-chat --backend board                     # chat with Qwen3-0.6B on the card
otpu-chat --backend board --model lfm2        # LFM2.5-230M; --model qwen35 needs the vops bitstream
otpu-smi                                      # the card's state (from another terminal)
```

No `OTPU_MCOLS` / `OTPU_LANES`: the tools follow the bitstream. If either is set in the shell
and disagrees with the bitstream, they stop with a message naming both.

Verified so far only on the Verilator board model (the current RTL; no card yet):
`otpu-selftest --sim` passes every stage, and its model stage matches the ISA simulator token
for token (Qwen3, LFM2 and Qwen3.5 at MCOLS=2; Qwen3 at MCOLS=4 VPU_CL=4); `otpu-diag --sim`
passes every check that the model can run. What the model cannot show: MIG calibration, the
controllers' read-modify-write of partial writes (the model applies byte strobes directly),
PCIe, the DMA rate and the real DRAM latency.

### Qualifying a bitstream

A candidate bitstream becomes the resting image only after `tools/qual/qual.sh` passes on the
card. It runs from the host tree it lives in, under the card lock, and leaves the candidate on
the card (`REST=path/otpu.bit` leaves another; `LOAD=0` loads nothing and qualifies the
bitstream the card already runs: on opentpu a hot rescan after a JTAG reload once did not bring
the link back, and a warm reboot did):

```sh
otpu-lock --wait 3600 -- bash tools/qual/qual.sh deploy_bl32mx120_be388a32         # fast
otpu-lock --wait 3600 -- bash tools/qual/qual.sh deploy_bl32mx120_be388a32 full    # full
```

| Phase | fast | full |
|:--|:--|:--|
| load, `otpu-selftest` (with the RDOT / OUTER / LOG2 op checks) | yes | yes |
| ISA references (background, `tools/qual/refs.py compute`) | yes | yes |
| `otpu-diag --mem full --soak 20` cold (march C-, 2 x 2.6 min) | - | yes |
| prefill 512 tokens + decode counters (`tools/qual/perf.py`), 6 configurations | yes | yes |
| streamed decode (`tools/decode_profile.py`, 96 tokens) | 4-bit | all 6 |
| `tools/rw_bench.py` | - | yes |
| warm soak (continuous Qwen3 decode) | 3 min | 5 min |
| `otpu-diag --soak 20`, warm | quick memory test | march C- |
| after the soak: token-exact against the ISA simulator, 6 configurations, per-position and resident decode | yes | yes |
| final `otpu-selftest` (after loading `REST` if set) | yes | yes |

Every phase prints its duration; the table of phases is at the end and in `$OUT/phases.tsv`
(`/tmp/qual-<deploy>`), with every PASS / FAIL line in `$OUT/checks.txt`. The fast profile is
meant for images that change timing or the DRAM path; use `full` for a new production
candidate after RTL changes to the MXU, VPU or memory system, and whenever fast finds anything.

The ISA references (greedy tokens of the ISA simulator, 32 tokens per configuration) are what
costs time when they are not cached: 0.5 to 10 minutes each, 20 minutes for the six on a loaded
omarchy. `tools/qual/refs.py` caches them by content: the sources of the `opentpu` package
without `opentpu/host` (the card's host code; the simulator configuration it computes is hashed
as a value), the configuration, the checkpoint, the formats and the token count. A host-only
change (a poll fix, a new tool) reuses them; any compiler, kernel or simulator change computes
new ones. The key holds no path or machine, so they can be computed ahead on any box and
copied:

```sh
python3 tools/qual/refs.py compute ~/otpu-build/refcache/configs/deploy_bl16mx120_be388a1f.pkl
rsync -a ~/otpu-build/refcache/ omarchy:otpu-build/refcache/    # from another box
```

(`qual.sh` keeps each card's configuration as `$REFCACHE/configs/<deploy>.pkl`; a new build with
the same D, MCOLS, LANES, PAIR, DSTEP and ACT_ROWS has the same one.) On the Mac the six
compute in 6.4 minutes one at a time (Qwen3.5 4-bit 139 s, LFM2 int8 11 s), against about 20
minutes three at a time on a busy omarchy (Qwen3.5 4-bit 620 s), and give the same tokens
(checked for all six against omarchy's references of 2026-09-28).

A card check never waits for a reference nobody is computing: a job in progress keeps a
`.pending` file with its pid and a heartbeat, a failed job leaves a `.failed` file with the
reason, and `refs.py card` fails at once when the reference is missing, its job died (the OOM
killer: `killed by signal 9`) or its heartbeat stopped. `compute` starts a job only while
`MemAvailable` covers its recorded peak memory plus 4 GiB, and runs one job fewer per Vivado
build on the box. (Before this, a reference job killed by the OOM killer beside two Vivado
builds left the card check polling for an hour.) References computed during the session load
the host while the prefill and decode phases measure wall time: for wall numbers that go into
a table, compute the references first.

### First light (measured on the card, 2026-09-26)

Build 74d48591 (the primary image), Arch Linux 7.1 host, Xilinx dma_ip_drivers XDMA (poll mode),
Digilent FT232H JTAG cable (`openFPGALoader -c digilent_hs2`; this cable's chain shows the FPGA
alone). PCIe Gen1 x8; both DDR3 channels calibrate. `otpu-selftest` passes every stage and
`otpu-diag` every check (124, including the 93 instruction variants). Greedy decoding equals the
ISA simulator token for token for all three models ("The capital of France is Paris."):

| Model | device Mcycles / token | tok/s at 100 MHz | simulated (bw 80, ctx 128) |
|---|---|---|---|
| Qwen3-0.6B | 20.66 | 4.8 | 6.20 |
| LFM2.5-230M | 7.71 | 13.0 | 2.35 |
| Qwen3.5-0.8B | 25.91 | 3.9 | 8.41 |

Burst reads (build a691ea98, measured 2026-09-27; port-B reads as INCR bursts of up to 8 beats per
channel): every model still equals the ISA simulator token for token.

| Model | device Mcycles / token | tok/s at 100 MHz | selftest wall tok/s | before (74d48591) |
|---|---|---|---|---|
| Qwen3-0.6B | 8.58 | 11.65 | 9.75 | 20.66 |
| LFM2.5-230M | 3.14 | 31.8 | 20.40 | 7.71 |
| Qwen3.5-0.8B | 11.88 | 8.42 | 6.58 | 25.91 |

DRAM reads while running: 7.2 GB/s (Qwen3), 7.6 GB/s (LFM2), against 3.0 before. MXU_STARVE (cycles
the MXU waits for weight chunks) is still 37% (Qwen3) / 26% (LFM2): DRAM-800 efficiency and the
request pipeline are the next limit. LFM2 runs 65% of the token time; the rest is the host.

DRAM address map and gathered QST writes (build e58ecb65, DDR3-800, measured 2026-09-27): the
MIG address map is ROW_BANK_COLUMN (it was BANK_ROW_COLUMN, which put everything below 256 MB of a
channel in bank 0, so every switch between the weight stream, the scale reads and the KV cache
was a precharge + activate), and the AXI adapter gathers the quantizer's byte stores into whole
64-byte beats (a partial-strobe write is an ECC read-modify-write in the MIG). `otpu-selftest`
passes every stage for all three models, token for token, and `otpu-diag --mem full` passes
(126 checks). Counters over each model run (while RUNNING):

| Model | device Mcycles / token | before (a691ea98) | DRAM reads while running | MXU_STARVE | model projection |
|---|---|---|---|---|---|
| Qwen3-0.6B | 6.49 | 8.58 | 9.56 GB/s (75% of 12.8) | 20.5% (37%) | 6.13 (-25%) |
| LFM2.5-230M | 2.33 | 3.14 | 10.19 GB/s (80%) | 17.5% (26%) | 2.20 (-27%) |
| Qwen3.5-0.8B | 8.44 | 11.88 | 9.55 GB/s (75%) | 26.2% | not run |

The projection comes from the DDR3 bank model in the AXI memory simulation
(`sim/verilator/otpu_axi_mem.sv`, `+axi_dram=1`; `tools/perf_qwen.py --dram brc|rbc`): open rows
per bank, tRCD / tRP / tRAS / tRC, refresh, turnarounds and the ECC read-modify-write, with three
parameters fitted to the a691ea98 measurements (tRP = tRCD = 3 controller cycles, a
read-modify-write holds the channel 23 cycles, 4 cycles per AXI read transaction). Fitted, it
reproduces a691ea98 at 8.22 (Qwen3) and 3.00 (LFM2) Mcycles, 4% under the card; the -25% / -27%
it projected for this build came out -24% / -26% on the card.

Three more parameters fitted on 2026-09-28: production image deploy_prod120hp_aebb0bf0,
DDR3-1066, 120.755 MHz. All three are set in `ddr3_plusargs`.
- Every AXI read transaction costs the data bus 1.22 controller cycles (`+axi_tgap=122`), plus
  0.06 cycles per beat (`+axi_bgap=6`). This caps sequential 8-beat bursts at about 8 / 9.7 of
  the peak: ~111 B per core cycle. The split between the two comes from the card's 16-beat
  bursts (below). With the per-transaction cost alone (1.7, the first fit), 16-beat bursts
  simulated 2.5% faster than the card.
- Every AXI write transaction (the adapter's writes are single beats) costs 2 controller cycles
  (`+axi_wgap=200`).
- A read/write turnaround costs 4 controller cycles (`+axi_tturn=4`, was 2).

Without these the model gave ~126 B/cycle and 1-2% MXU starvation, where the card shows ~15%. The
write parameters come from `tools/rw_bench.py`, which puts an fp4 weight stream (`mm`) beside DMA
stores, loads or DSTEPs:

| rw_bench mode | card cycles | sim before | sim fitted |
|---|---|---|---|
| mm | 78,669 | 69,943 | 79,166 |
| st | 141,965 | 136,457 | 136,619 |
| ld | 137,852 | 136,472 | 137,276 |
| dstep | 91,218 | 74,460 | 74,706 |
| mm+st | 217,350 | 189,058 | 217,266 |
| mm+ld | 199,015 | 192,180 | 198,171 |
| mm+dstep | 158,457 | 112,794 | 154,713 |

The fitted model then checks against the decode runs. These were run with the first fit (tgap 170, no bgap); at 8-beat bursts the split fit gives the same cycles within 0.1%. The Qwen3.5 runs use DSTEP
(`OTPU_PAIR=1 OTPU_DSTEP=1`), as the image does:

| Decode run (4-bit + int8 head) | card (lens) | sim before | sim fitted |
|---|---|---|---|
| LFM2, pos 50 | 1.418 M | 1.262 M | 1.431 M (+0.9%) |
| Qwen3.5, 1 layer, pos 60 | 2.553 M | 2.203 M | 2.512 M (-1.6%) |
| Qwen3.5, 4 layers | 2.952 M | 2.529 M | 2.909 M (-1.5%) |
| Qwen3.5 DeltaNet + MLP layer ((3 - 1 layers) / 2) | 154.7 K | 120.8 K | 152 K |
| Qwen3.5 attention + MLP layer (4 - 3 layers) | 89.3 K | 84.4 K | 93 K |

Still missing: DSTEP alone runs 22% slower on the card (91.2K vs 74.7K cycles). It is not
bandwidth: the sim's DSTEP is far from the DRAM limit and does not respond to wgap. It is
probably latency (write responses, or read-after-write), which the model does not separate.

**16-beat read bursts (`make bit AXI_BL=16`), measured 2026-09-28.** This image is
unqualified: build fb2b6630, WNS -0.294 ns at 120.755 MHz, fmax 116.6. It passes selftest,
`otpu-diag --only isa` (93 / 93), and LFM2 is token-exact against the ISA simulator (32 tokens).
Same session, against production hp-wb:

| | hp-wb (BL 8) | BL 16 (unqualified, WNS -0.294) | sim BL 8 / 16 (fitted) |
|---|---|---|---|
| rw_bench mm | 78,670 cycles, 113.3 B/cycle | 74,578, 119.5 B/cycle (-5.2%) | 79,166 / 74,495 |
| rw_bench mm+dstep | 158,638 | 154,996 | 154,647 / 150,052 |
| LFM2 4-bit + int8 head decode, 96 tokens | 1.455 Mcycles, 83.0 device / 79.0 wall tok/s | 1.384 Mcycles, 87.3 / 83.5 tok/s (-4.9%) | pos 50: 1.430 / 1.347 M |

Before the burst fix: Decode runs at ~3.2x the simulated cycles: the counters show the MXU starved (MXU_BUSY 94%,
MXU_MAC 21%, DRAM_WAIT 0.1%) and DRAM reads at 3.0 GB/s (0.23 beats / cycle / channel). Port B
issues single-beat 64-byte AXI reads (SmartConnect ports MAX_BURST_LENGTH 1); the per-transaction
cost in SmartConnect and the MIG AXI front end, which the simulated DRAM does not charge, caps the
rate. Fix in progress: burst reads on port B. Host DMA: 0.78 GB/s host -> card, 1.12 GB/s back.

Found at bring-up, fixed in the host (558a4bf): the DRAM must be written once after configuration
(ECC: a read of a never-written beat hangs; `Board.scrub`, ~6 s, done by every tool), register
access must be single 32-bit loads / stores, and DMA reads must not use `preadv` (the XDMA driver's
asynchronous read_iter crashes kernel 7.1). The XDMA's PCI class code is "serial controller",
so the 8250 driver probes the card: a udev rule sets `driver_override=xdma` (now installed by
`otpu-setup`; bd.tcl sets class 12 00 00 from the next build on).

## 5. Clocks and the roofline

| Clock | Frequency | Source | Drives |
|---|---|---|---|
| sys_clk_50 | 50 MHz | AA28 oscillator | MMCM (VCO 800 MHz; 1000 MHz at DDR3-1333) |
| core_clk | 100 MHz | MMCM /8 (/10) | accelerator, control registers, interconnect core side |
| clk_200 | 200 MHz | MMCM /4 (/5) | MIG reference (IDELAYCTRL); MIG system clock at DDR3-800, 1300, 1600 |
| clk_mig | VCO / 3 | MMCM /3 | MIG system clock at DDR3-1066 (266.667 MHz) and 1333 (333.333 MHz) |
| ui_clk0/1 | 100 MHz (133 / 162.5 / 167 / 200 at 1066 / 1300 / 1333 / 1600) | MIG | MIG AXI side, 512 bit |
| DDR3 CK | 400 MHz (533 / 650 / 667 / 800) | MIG PLL | memory |
| axi_aclk | 125 MHz | XDMA | PCIe AXI side, 128 bit (Gen1 x8) |

Decode is DRAM-bound: every token streams all weights once. The accelerator consumes one
128-byte chunk per core cycle at its peak; each DDR3 channel (x64) delivers 8 bytes per CK edge.

| DDR3 | channel peak | both channels | core clock to match (128 B/cycle) | MIG ui_clk |
|---|---|---|---|---|
| 800 (default) | 6.4 GB/s | 12.8 GB/s | >= 100 MHz | 100 MHz |
| 1066 | 8.5 GB/s | 17.1 GB/s | >= 133 MHz | 133 MHz |
| 1300 (out of spec) | 10.4 GB/s | 20.8 GB/s | >= 163 MHz | 162.5 MHz |
| 1333 (out of spec) | 10.7 GB/s | 21.3 GB/s | >= 167 MHz | 167 MHz |
| 1600 (out of spec) | 12.8 GB/s | 25.6 GB/s | >= 200 MHz | 200 MHz |

So the fmax each part must clear to stay at the roofline at DDR3-800: core_clk >= 100 MHz
(accelerator, otpu_axi_dram, control), MIG ui_clk 100 MHz (fixed by the MIG), SmartConnect
paths at their own clocks (100 / 125 MHz), XDMA 125 MHz (fixed by the IP at Gen1 x8). Real DDR3
efficiency (refresh, row misses, read/write turnaround) is ~70-85 %, which the accelerator's
deep prefetch absorbs; a core clock above 100 MHz buys nothing at DDR3-800. Host transfers
per token are small (program ~40 KB, logits 600 KB): the one-time weight upload (~820 MB) takes
~0.5-0.7 s at Gen1 x8 (2 GB/s). Gen1 rather than Gen2: at Gen2 the PCIe block runs a
500 MHz user clock whose IP-placed paths missed timing by ~0.1 ns (Vivado 2026.1, 80 MHz build).

### DRAM efficiency

Two numbers describe how well a token uses the memory. `tools/perf_qwen.py` prints both:

- **DRAM efficiency** = the bytes a token moves to and from DRAM (weights with their block
  scales, the KV cache, the conv state, the I/O; reads and writes) / (its time x the DDR3
  peak). The peak is 16 bytes x the data rate: two x64 channels, ECC bits not counted, so
  17.07 GB/s at DDR3-1066. This is the number the hardware as a whole is judged by
  (`opentpu.profile.dram_efficiency`).
- **Port efficiency** = the same bytes / (cycles x 128 B): the fraction of the core's own
  path, one 64-byte beat per channel per core cycle.

The port caps DRAM efficiency at 128 B x f_core / peak: 75% at 100 MHz and DDR3-1066, 87% at
116 MHz, 94% at 125 MHz, 100% from 133 MHz. Above that clock, or with a wider port
([wide_dram.md](wide_dram.md)), the DDR3 is the limit. `perf_qwen.py --ddr 1066 --mhz F`
runs the DDR3 bank model calibrated on the card (section 4; `opentpu.profile.ddr3_plusargs`)
at that data rate and core clock and prints both efficiencies. Every figure it prints is
simulated. [lfm2.md](lfm2.md) has the LFM2 numbers.

### Faster DDR3

This section measured the MIG builds, since removed. The LiteDRAM core is generated for DDR3-1066
(`make bit` takes no other speed); a faster one means regenerating it (tools/litedram/gen_core.py,
docs/litedram.md).

The core takes at most 12.8 GB/s (one 128-byte chunk per 100 MHz cycle, a 512-bit port per
channel), so a faster DDR3 helps only up to that point. It lets the controllers keep the port
full despite refresh, row misses and ECC read-modify-write: the simulated benchmark gains
16.1 -> 20.0 tok/s going from 80 % to 100 % of the chunk rate (Qwen3-0.6B, batch 1). Going past
12.8 GB/s needs a wider path: [wide_dram.md](wide_dram.md).

**What MIG allows on this board.** The DDR3 channels are on banks 11-18, which on the
xc7k480t-ffg1156 are HR banks (no DCI; MIG terminates with `IN_TERM UNTUNED_SPLIT_50`). MIG's
limit for HR banks on a -2 part at 4:1 with single-rank components
(`mig_7series_v4_2/data/dlib/7series/ddr3_sdram/time_periods.xml`, `tmin_hr`) is:

| DDR3 voltage | min tCK | max data rate |
|---|---|---|
| 1.5 V | 1875 ps | DDR3-1066 |
| 1.35 V (DDR3L) | 2500 ps | DDR3-800 |

The same file gives 1500 ps (`tmin_hp_18`) and 1072 ps (`tmin_hp_20`), but those figures are
for HP banks, where VCCAUX_IO matters, and this board has none.

**DDR3 voltage: 1.5 V (measured).** The MT41K256M8 is DDR3L: it runs at 1.35 V or 1.5 V. The
board's DDR3 supply was measured at 1.5 V with a multimeter (2026-09-27), which matches every
build's SSTL15 setting and TiferKing's reverse-engineered MIG project (DDR3-1066 at 1.5 V). MIG's
limit for these HR banks is therefore DDR3-1066, and DDR3-1066 is in spec on this board.

MIG offers internal VREF only up to 800, so 1066 and up rely on an external VREF. The VREF pins
of the six DDR3 banks carry no DDR3 signals, which fits an external VREF, but it has not been
measured.

MIG's messages when it imports the generated .prj (Vivado 2026.1, mig_7series 4.2):

- 1066: `[Mig7series 79-144] Invalid Input Clock Period 266.667. Setting to nearest possible
  Input Clock Period value 266.666.` It is a rounding only, and the configuration is supported.
- 1300 (tCK 1538 ps), 1333 (1500 ps), 1600 (1250 ps): `CRITICAL WARNING: [Mig7series 79-155]
  Memory Time Period (1500 ps) (666.666687 Mhz) is not supported for MIG. There has been a change
  in the allowed frequency ranges as described in Answer Record 67179. Instantiate and customize
  a new instance of MIG for your design.` (the 1333 text; 1300 and 1600 differ only in the
  period).
- Input clocks. At 1300, 200 MHz is rounded to 200.06 (79-144), with the PLL at x13/2. At 1333,
  200 MHz is rejected (79-144, "nearest possible ... 205.128"), so the MMCM runs at a VCO of
  1000 MHz and gives MIG 333.333 MHz, accepted as is (PLL x4). At 1600, 200 MHz is accepted
  (PLL x8).

79-155 is a critical warning, not an error. MIG still generates the controller: CL 9 / CWL 7 at
1300 and 1333, CL 11 / CWL 8 at 1600, tRFC 160 ns.

**The PHY patch at 1333 and 1600.** For tCK <= 1500 ps, MIG's byte group
(`mig_7series_v4_2_ddr_byte_group_io.v`: `IDELAY_FINEDELAY_USE = (TCK > 1500) ? "FALSE" :
"TRUE"`) instantiates IDELAYE2_FINEDELAY. In this design that stays a black box, and the first
DDR3-1333 build stopped in opt_design with 146 x `DRC INBB-3 ... of type
otpu_bd_mig_1_0_IDELAYE2_FINEDELAY has undefined contents` (measured, 2026-09-27). So for those
two speeds, `vivado/build.tcl` rewrites that line in the generated MIG sources to use the plain
IDELAYE2, the primitive MIG itself uses above 1500 ps, and prints `CRITICAL WARNING: [openTPU]
... MIG PHY patched to IDELAYE2 (out of spec)`. The read-capture delay line then has the coarse
IDELAY taps only, without the fine-delay steps MIG expects at these speeds. DDR3-1300 needs no
patch, so it is the fastest setting MIG itself builds.

All three run the HR I/O and the PHY beyond what AMD characterizes. Nothing in them changes a
voltage, so the risk is only that calibration fails or that data goes bad, and a power cycle
undoes a JTAG load. Only the card can show whether they calibrate, and whether the data stays
correct as the die warms. Their deploy directories carry `_oos` (out of spec) in the name.

**Builds** (measured, Vivado 2026.1 on omarchy, 2026-09-27; MCOLS=2, core 100 MHz). All timing
constraints are met:

| DDR3 | commit | WNS / WHS | core_clk slack | MIG ui_clk (clk_pll_i) slack | deploy directory |
|---|---|---|---|---|---|
| 1300 (out of spec) | 819fee49 | +0.032 / +0.044 ns | +0.032 ns | +0.060 / +0.105 ns (6.154 ns) | `build/deploy_ddr1300_oos_819fee49` |
| 1333 (out of spec, PHY patched) | 254f8388 | +0.088 / +0.043 ns | +0.094 ns | +0.102 / +0.091 ns (6.000 ns) | `build/deploy_ddr1333_oos_254f8388` |

The tightest inter-clock paths are inside the MIG: ui_clk to the ISERDES clocks, +0.088 to
+0.094 ns. The SmartConnect crossings are not among the worst. In the 1333 build MIG adds its
own 400 MHz IDELAY reference (`clk_ref_mmcm_400`). Each deploy directory holds otpu.bit,
otpu.mcs, otpu.prm, reports/ and `mig_messages.txt`, which lists the MIG critical warnings and
the patch messages of that build.

**Production image (2026-09-29): `build/deploy_secand3_02569bc`** (omarchy and opentpu
`~/otpu-build/`; branch se-cand3 02569bc, built with `SE=v2` (now the only configuration), `AXI_BL=32`,
`AXI_WBL=8`, BUILD_ID 002569bc). The stream engine (docs/stream.md) replaces the VPU's two
composite chains and the DMA's DSTEP datapath. The composites run on the 8 lanes' three stages
(COMP8), and a stream's dot A and dot Q share the VPU's tree (ONE_TREE). STREAM is CAPS bit26.
On top of pn32's RTL, it also carries main since e252101, including 5cd6c39's MXU command queue
as head / next registers, and three timing cuts:
- `cleft`'s compares as registers (DMA → adapter B queue);
- a second reset stage for Q and the VPU;
- TMEM's read data registered before the DMA's chunk buffer.

Core clock 120.755 MHz, DDR3-1066.
- **Timing:** WNS +0.031 ns after the post-route phys_opt (-0.218 routed), TNS 0, WHS
  +0.016 ns.
- **Area:** 202,757 LUT, 162,837 FF, 301 DSP, 569 BRAM tiles, 62,639 slices (83.9%).
  pn32 was 214,900 LUT, 353 DSP and 94.8% of slices. VPU + DMA: 52.9K LUT / 86 DSP against
  66.7K / 138 (docs/stream.md 12.3).

Qualified on the card 2026-09-29 on opentpu with `LOAD=0 tools/qual/qual.sh
deploy_secand3_02569bc fast` (JTAG load, then a warm reboot; host tree se-cand3): 0 FAIL, 44 PASS.
- The selftest passes, with its new `stream` stage: STREAM in every mode, and DSTEP.
- The warm diag passes after the 180 s soak.
- All 6 configurations are token for token equal to the ISA simulator, per-position and
  resident.

Decode (`tools/decode_profile.py`, greedy, 96 tokens; device cycles; the wall numbers are
opentpu's):

| Model | Weights | Mcycles/token | device tok/s | wall tok/s | pn32 device tok/s |
|---|---|---|---|---|---|
| LFM2.5-230M | 4-bit, int8 head | 1.334 | 90.52 | 87.17 | 90.21 |
| Qwen3-0.6B | 4-bit, int8 head | 3.504 | 34.46 | 34.06 | 34.26 |
| Qwen3.5-0.8B | 4-bit, int8 head | 4.899 | 24.65 | 24.18 | 24.65 |

- **Prefill** (512 tokens, device), int8 / 4-bit:
  - LFM2 161.5 / 169.7 and Qwen3 55.0 / 58.5, equal to pn32;
  - Qwen3.5 46.3 / 48.8, against 44.1 / 46.3.
- **DRAM read per token** equals pn32's in all six configurations.
- The int8 decode profile (the full qual profile) has not been run on this image.

**Previous production image (2026-09-28 afternoon until 2026-09-29):
`build/deploy_pnbl32_e2521032`** (omarchy
`~/otpu-build/`; branch prod-next e252101, built with `AXI_BL=32`, `AXI_WBL=8`, the
ExtraTimingOpt implementation strategy, BUILD_ID e2521032). On top of be388a32's RTL:
- the DMA writes an ST's chunks in runs of 32 on consecutive cycles (they gather in the LD chunk
  buffer, idle during an ST), and DSTEP's state reads and write-backs in runs of 64 chunks
  instead of 16, so the DRAM turns between reads and writes once per run, not once per chunk
  between the MXU's weight reads (Qwen3.5 decode: 48K turnarounds per channel and token in the
  simulator before, 7.3K after);
- port B writes go out as AXI INCR bursts of up to 8 beats (AWLEN / WLAST, one write response
  per burst);
- a port B write drops the MXU's scale prefetch runs only when it writes into them;
- the adapter's oldest pending A read (aoh) is a register, which took the order FIFO's LUT RAM
  read off the port-A return path (the first build of this branch failed there at -0.469 ns).

Core clock 120.755 MHz, DDR3-1066. Post-route WNS -0.132 ns, +0.013 ns after the post-route
phys_opt; WHS +0.021 ns. The worst paths are the DMA's request (cleft) into the adapter's B
queue count and run marks (qb_n, qc), 16-18 levels, +0.013 to +0.028 ns; then TMEM -> the DMA's
chunk buffer at +0.048 ns. No Synth 8-6430 memory; congestion level 5 at most. The same RTL at
`AXI_BL=16` (`~/otpu-build/deploy_pnbl16_e2521016`) meets timing too, WNS +0.012 ns, WHS +0.038 ns
(worst: the write burst length FIFO's enable, 10 levels), and has not been on the card.

Qualified on the card 2026-09-28 10:54-11:32 with `tools/qual/qual.sh deploy_pnbl32_e2521032
full` (host tree prod-next e0315b4 = e252101 + main 27bdf62; the six ISA references computed
ahead on the Mac in 6.5 min and copied; omarchy's load 1.2-1.5, nothing else running): 0 FAIL,
32 PASS, 4 ALL PASS.
- selftest with the 12 op checks passes;
- `otpu-diag --mem full --soak 20` passes cold (54 °C) and after the 300 s warm soak;
- all 6 configurations are token for token equal to the ISA simulator after the soak, with
  per-position and resident decode.

| Phase (full profile) | seconds |
|---|---|
| load + selftest | 18 |
| ISA references (cached) | 3 |
| diag cold, full march | 309 |
| prefill + decode counters, 6 configurations | 371 |
| decode_profile, 6 configurations | 323 |
| rw_bench | 1 |
| warm soak | 302 |
| diag warm, full march | 306 |
| token-exact after the soak, 6 x per-position + resident | 616 |
| final selftest | 2 |
| total | 38 min |

Decode (`tools/decode_profile.py`, greedy, 96 tokens, streamed logits; "wall" is the whole run),
against be388a32's device cycles from its A/B below:

| Model | Weights | Mcycles/token | device tok/s | wall tok/s | be388a32 Mcycles | change |
|---|---|---|---|---|---|---|
| LFM2.5-230M | int8 | 1.992 | 60.61 | 59.30 | 2.008 | -0.8% |
| Qwen3-0.6B | int8 | 5.300 | 22.78 | 22.68 | 5.324 | -0.5% |
| Qwen3.5-0.8B | int8 | 6.876 | 17.56 | 17.47 | 7.415 | -7.3% |
| LFM2.5-230M | 4-bit, int8 head | 1.339 | 90.21 | 87.02 | 1.353 | -1.0% |
| Qwen3-0.6B | 4-bit, int8 head | 3.524 | 34.26 | 34.03 | 3.555 | -0.9% |
| Qwen3.5-0.8B | 4-bit, int8 head | 4.899 | 24.65 | 24.40 | 5.400 | -9.3% |

The host's critical path is 0.21-0.30 ms per token; HALTED is seen 0.005-0.22 ms after the run's
end. The fitted DDR3 model (lfm2-cycles 71b55e1, at BL16) projected 4.98 Mcycles for Qwen3.5
4-bit with this RTL.

Prefill (a 512-token prompt; wall includes compiling each chunk on the host) and DRAM (the
card's counters over 64 decode tokens, `tools/qual/perf.py`):

| Model | Weights | prefill device tok/s | prefill wall tok/s | DRAM read per token | DRAM write per token | DRAM while decoding |
|---|---|---|---|---|---|---|
| LFM2.5-230M | int8 | 161.5 | 97.4 | 245 MB | 0.54 MB | 14.45 GB/s (85%) |
| Qwen3-0.6B | int8 | 55.0 | 51.7 | 662 MB | 2.50 MB | 14.41 GB/s (84%) |
| Qwen3.5-0.8B | int8 | 44.1 | 39.5 | 810 MB | 21.40 MB | 14.50 GB/s (85%) |
| LFM2.5-230M | 4-bit, int8 head | 169.7 | 100.5 | 164 MB | 0.54 MB | 14.18 GB/s (83%) |
| Qwen3-0.6B | 4-bit, int8 head | 58.5 | 54.4 | 442 MB | 2.50 MB | 14.12 GB/s (83%) |
| Qwen3.5-0.8B | 4-bit, int8 head | 46.3 | 41.5 | 561 MB | 21.40 MB | 14.21 GB/s (83%) |

`tools/rw_bench.py` (cycles) against be388a32 and the fitted DDR3 model (the simulator ran this
RTL at 16-beat reads):

| Mode | this image | be388a32 | simulated |
|---|---|---|---|
| mm (8.91 MB of fp4 weights) | 72,594 | 72,559 | - |
| mm + 64 STs of 64 KB | 146,100 | 211,763 | 149,013 |
| mm + 32 DSTEPs | 110,642 | 151,974 | 120,789 |
| 32 DSTEPs alone | 75,706 | 89,037 | 74,706 |

The card charges an 8-beat write burst about as the model does (one wgap per AW), not per beat.

**Host path, measured 2026-09-28 12:14-12:23** on this image (host main 59a9f06, omarchy's load
1.8-3.8, a niced test run beside it; 96 greedy tokens per run; logs and JSON in the deploy
directory's `readme-run/`). The host's changes since e0315b4:
- the sampler is warmed when the Chat starts, so the first pick (2-10 ms in a new process) is off
  the decode;
- the next pick's buffers are set up after the run's start;
- the status file's rewrite runs on one writer thread, at least 2 ms after the token;
- streamed pieces are marked again after the next start;
- the expected run time, the poll's hint, is kept per program length.
The last change removed the "poll overshoot". A trace of the host's sleeps, register reads and
DMA (12:04) showed HALTED seen within 10 us of the run's end on 95 of 96 runs. The one late run
was the first decode run after the prompt: it was expected to take the prefill chunk's 25 ms
and was seen 14 ms late, which averaged to 0.15 ms per token.

LFM2.5-230M, 4-bit with an int8 head, 5 runs:

| run | wall tok/s | device tok/s | host critical path, ms/token | HALTED seen after the end (median / max), ms |
|---|---|---|---|---|
| 1 | 88.76 | 90.16 | 0.177 | 0.003 / 0.014 |
| 2 | 88.78 | 90.12 | 0.170 | 0.003 / 0.015 |
| 3 | 88.07 | 90.16 | 0.271 | 0.003 / 0.017 |
| 4 | 88.74 | 90.18 | 0.181 | 0.003 / 0.013 |
| 5 | 88.90 | 90.20 | 0.165 | 0.003 / 0.014 |
| median | **88.76** | 90.16 | 0.177 | 0.003 |

The five replies are token for token identical. The other configurations, one run each, with
the same host:

| Model | Weights | Mcycles/token | device tok/s | wall tok/s |
|---|---|---|---|---|
| LFM2.5-230M | int8 | 1.993 | 60.59 | 59.78 |
| Qwen3-0.6B | int8 | 5.291 | 22.82 | 22.73 |
| Qwen3.5-0.8B | int8 | 6.879 | 17.55 | 17.46 |
| Qwen3-0.6B | 4-bit, int8 head | 3.527 | 34.24 | 34.02 |
| Qwen3.5-0.8B | 4-bit, int8 head | 4.895 | 24.67 | 24.56 |

On the card's host (fine timestamps, 11:50, host c3db597), the median critical path from
HALTED to the next RUN is:
- the counters, 19 us;
- WR_IDLE, the last logits piece's read (32 KB) and its hand-over to the sampler, 89 us;
- the backend and chat code, 16 us;
- the argmax, 13 us;
- the run arguments, 21 us;
- CTRL, 11 us.

**On opentpu (measured 2026-09-28 16:56-19:20 UTC-3, like the times above; same image, host
main a739d08; load 0.6-2.7).**
`qual.sh fast` with `LOAD=0` passed in 37 min, with 0 FAIL. The logs and JSON are in the deploy
directory's `opentpu-run/` on opentpu. The device numbers are omarchy's within 0.3%; the host
takes longer per token. LFM2.5-230M, 4-bit with an int8 head, 5 runs:

| run | wall tok/s | device tok/s | host critical path, ms/token |
|---|---|---|---|
| 1 | 85.51 | 90.44 | 0.572 |
| 2 | 86.77 | 90.34 | 0.447 |
| 3 | 86.55 | 90.34 | 0.489 |
| 4 | 86.79 | 90.35 | 0.452 |
| 5 | 87.12 | 90.40 | 0.418 |
| median | **86.77** | 90.35 | 0.452 (omarchy: 0.177) |

The replies are identical token for token. They match the A/B runs below, qual's run, and
omarchy's 5 runs above. The other configurations, one run each (their replies match
omarchy's):

| Model | Weights | device tok/s | wall tok/s | host critical path, ms/token (omarchy) |
|---|---|---|---|---|
| LFM2.5-230M | int8 | 60.69 | 58.13 | 0.686 (0.231) |
| Qwen3-0.6B | int8 | 22.81 | 22.57 | 0.481 (0.193) |
| Qwen3.5-0.8B | int8 | 17.58 | 17.14 | 1.322 (0.296) |
| Qwen3-0.6B | 4-bit, int8 head | 34.31 | 33.69 | 0.524 (0.188) |
| Qwen3.5-0.8B | 4-bit, int8 head | 24.71 | 24.03 | 1.032 (0.193) |

A `watch otpu-smi` (it reads the card every 2 s) ran beside all of these except Qwen3.5 4-bit;
it stopped during the Qwen3 4-bit run.
Three LFM2 4-bit runs without it, in the A/B below, gave 86.96, 86.69 and 86.10: no visible
effect. These numbers are recorded as measured; the opentpu host path was not tuned.

**Shelved: a smaller last LM-head chunk (branch host-tail, 694237d, not merged).** The idea: split
the decode LM head's last chunk so that the chunk stored last has 2048 rows (LFM2: 7 x 8192 +
6144 + 2048 rows). The read after HALTED would then shrink from 32 KB to 8 KB. What we checked:
- In the simulator, the logits and the tokens stayed identical in all six configurations.
- The perf model charged 1,225 cycles (10 us) more per LFM2 token, for the extra MM.
- On opentpu, alternating with main (LFM2 4-bit), it lost: 84.76, 84.24 and 80.96 tok/s wall,
  against 86.96, 86.69 and 86.10.
  - The read after HALTED went from 89.5 to 196.5 us.
  - HALTED was seen later: the median overshoot went from 0.003-0.006 ms to 0.065-0.194 ms.

Our reading of the numbers (not traced): an MM of 2048 rows takes about 0.15 ms. So the
next-to-last piece (24 KB) now comes complete about 0.15 ms before HALTED, instead of about
0.57 ms. The slower host is still reading or probing that piece when the run ends. Do not retry
the split without reading the next-to-last piece earlier, or ending the stream loop at HALTED.

The same branch (b51ee4e) also kept a running greedy argmax, and read only STATUS and CYCLES
after a streamed step. In those runs the sampler and the counter reads moved by only 1-2 us.
They were never measured apart from the split, and were not tried on omarchy.
DSTEP alone, 22% slower on the card than simulated with 16-chunk runs, now matches the
simulator: that gap was the read / write turn per 2 KB run.

**Production image 2026-09-28 11:00 until the afternoon (BL32): `build/deploy_bl32mx120_be388a32`** (omarchy
`~/openTPU/build/`; be388a1 built with `AXI_BL=32`: port B reads in bursts of up to 32 beats,
the adapter's request queue 32 deep). Core clock 120.755 MHz, DDR3-1066, WNS +0.066 ns, WHS
+0.016 ns. The worst path is MXU q_h -> TMEM pw_d at +0.066 ns (11 levels). The adapter's
qc -> qb_n (the run / queue count) comes next at +0.097 ns (18 levels): QD 32 fits, with little
margin. `make bit` now builds it (`AXI_BL` defaults to 32).

BL16 vs BL32 A/B, 2026-09-28 08:44-08:55: the same session, host main 1654f70 (the poll fix),
omarchy's load 1.5-2.3 during BL16 and 1.6-4.1 during BL32. Decode is `tools/decode_profile.py`,
greedy, 96 tokens, streamed logits; "wall" is the whole run, the first window included.

| Model | Weights | BL16 Mcycles | BL16 device / wall tok/s | BL32 Mcycles | BL32 device / wall tok/s | BL32 vs BL16 (device) |
|---|---|---|---|---|---|---|
| LFM2.5-230M | int8 | 2.059 | 58.63 / 57.12 | 2.008 | 60.14 / 58.86 | +2.6% |
| Qwen3-0.6B | int8 | 5.459 | 22.12 / 21.66 | 5.324 | 22.68 / 22.20 | +2.5% |
| Qwen3.5-0.8B | int8 | 7.566 | 15.96 / 15.89 | 7.415 | 16.29 / 16.20 | +2.0% |
| LFM2.5-230M | 4-bit, int8 head | 1.384 | 87.22 / 84.11 | 1.353 | 89.27 / 85.41 | +2.3% |
| Qwen3-0.6B | 4-bit, int8 head | 3.620 | 33.36 / 32.31 | 3.555 | 33.97 / 33.75 | +1.8% |
| Qwen3.5-0.8B | 4-bit, int8 head | 5.501 | 21.95 / 21.82 | 5.400 | 22.36 / 22.22 | +1.9% |

`tools/rw_bench.py` mm: 122.9 B/cycle (BL32) against 119.5 (BL16) and 113.3 (BL8, aebb0bf0).
The fitted DDR3 model agrees with the card at all three burst lengths. The comparison has to be
like for like: the card's decode_profile averages positions 24-119 of the resident program, so
the simulation runs the resident program at pos 70 (`perf_qwen.py --resident --pos 70 --ddr 1066
--mhz 120.755`):

| LFM2 4-bit + int8 head | BL 8 | BL 16 | BL 32 |
|---|---|---|---|
| card, Mcycles/token | 1.455 | 1.384 | 1.353 |
| sim (tgap 1.22, bgap 0.06), Mcycles/token | 1.454 | 1.371 (-0.9%) | 1.336 (-1.3%) |
| card rw_bench mm, cycles | 78,670 | 74,591 | 72,559 |
| sim rw_bench mm, cycles | 79,161 (+0.6%) | 74,492 (-0.1%) | 72,144 (-0.6%) |

The per-position program at pos 50 (1.309 M at BL 32) is not comparable to the card's average.
A refit with a larger per-beat share (tgap 0.90 + bgap 0.10, or 0.60 + 0.14) moves rw_bench
away from the card at BL 16 / 32 (75,706 / 73,915 and 76,975 / 75,731 cycles), so the fit stays.

Qualified on the card 2026-09-28 08:58-10:54 (host tree main 1654f70; omarchy's load 6-16:
2 Vivado builds and a test suite):
- selftest with the 12 op checks passes;
- `otpu-diag --mem full --soak 20` passes cold (54 °C) and after a 315 s warm soak (61 -> 64
  °C): isa 93, system 5, mem 13;
- all 6 configurations are token for token equal to the ISA simulator after the soak. Before the
  soak, 5 of the 6 were checked: the Qwen3 4-bit reference job was killed by the OOM killer, and
  its card check timed out waiting for it;
- resident decode is equal to the per-position programs, 64 of 64 greedy tokens, for all 6
  configurations (Qwen3.5 is resident on main since 1654f70).

Prefill (a 512-token prompt; wall includes compiling each chunk on the host, on the loaded host)
and DRAM (the card's counters over 64 decode tokens):

| Model | Weights | prefill device tok/s | prefill wall tok/s | DRAM read per token | DRAM while decoding |
|---|---|---|---|---|---|
| LFM2.5-230M | int8 | 161.5 | 86.7 | 245 MB | 14.36 GB/s (84%) |
| Qwen3-0.6B | int8 | 55.0 | 50.1 | 663 MB | 14.33 GB/s (84%) |
| Qwen3.5-0.8B | int8 | 39.3 | 34.8 | 818 MB | 13.59 GB/s (80%) |
| LFM2.5-230M | 4-bit, int8 head | 169.7 | 58.7 | 164 MB | 14.07 GB/s (83%) |
| Qwen3-0.6B | 4-bit, int8 head | 58.5 | 52.2 | 443 MB | 14.04 GB/s (82%) |
| Qwen3.5-0.8B | 4-bit, int8 head | 41.9 | 36.7 | 570 MB | 13.11 GB/s (77%) |

rw_bench on it: mm 122.8 B/cycle, mm+st 211,763 cycles, mm+dstep 151,974, dstep 89,037.

**Production image 2026-09-28 morning (BL16): `build/deploy_bl16mx120_be388a1f`** (omarchy; branch
tv-cand2 be388a1: main 6e2605b + lfm2-cycles 6fb2851 (port B read bursts up to 64 beats) +
the tournament's MXU / adapter timing fixes (q_tz from its factors, dg1 registered, the port-A
return registered, DSTEP fill and RMAX drain registered), built with `AXI_BL=16`. Core clock
120.755 MHz, DDR3-1066, WNS +0.048 ns, WHS +0.013 ns. It was the production image from the
morning until BL32 replaced it at 11:00. A second BL16 image, `~/otpu-build/deploy_bl16_fb2b6630`
(lfm2-cycles fb2b663, AXI_BL=16 without the timing fixes, WNS +0.003 ns after an ExtraTimingOpt
re-implementation), also passed the checks below (05:44-06:44).

Qualified on the card 2026-09-28 06:46-07:43 (host tree host-8efceb7):
- selftest with the 12 op checks passes;
- `otpu-diag --mem full --soak 20` passes cold (57 °C) and after a 305 s warm soak (63 -> 64 °C):
  isa 93, system 5, mem 13;
- all 6 configurations are token for token equal to the ISA simulator, before and after the
  soak;
- resident decode is equal to the per-position programs (64 greedy tokens, Qwen3 and LFM2 at
  both formats).

Device numbers from the same session: decode Mcycles/token from `tools/decode_profile.py`
(greedy, 96 tokens, streamed logits); DRAM from the counters over 64 tokens; prefill from a
512-token prompt. The wall columns are left out: omarchy's load was 11-14 during the session
(2 Vivado builds and a test suite).

| Model | Weights | Mcycles/token | device tok/s | prefill device tok/s | DRAM read per token | DRAM while decoding | vs aebb0bf0 (device) |
|---|---|---|---|---|---|---|---|
| LFM2.5-230M | int8 | 2.060 | 58.63 | 157.5 | 245 MB | 14.02 GB/s (82%) | +5.2% |
| Qwen3-0.6B | int8 | 5.456 | 22.13 | 53.6 | 663 MB | 13.99 GB/s (82%) | +5.3% |
| Qwen3.5-0.8B | int8 | 7.515 | 16.07 | 38.1 | 811 MB | 13.27 GB/s (78%) | +4.6% |
| LFM2.5-230M | 4-bit, int8 head | 1.382 | 87.40 | 169.3 | 164 MB | 13.78 GB/s (81%) | +5.1% |
| Qwen3-0.6B | 4-bit, int8 head | 3.631 | 33.26 | 58.1 | 443 MB | 13.75 GB/s (81%) | +4.9% |
| Qwen3.5-0.8B | 4-bit, int8 head | 5.478 | 22.04 | 40.7 | 562 MB | 12.73 GB/s (75%) | +4.1% |

`tools/rw_bench.py` on it: the weight stream alone runs at 119.5 B/cycle (74,591 cycles; hp-wb
113.3), mm+st at 213,612 cycles, mm+dstep at 155,425, dstep at 91,438.

HALTED probe, LFM2 4-bit. With the normal poll, the host saw HALTED 0.35-0.86 ms after the run's
end on the loaded host, for 80.2 wall tok/s. With the poll reading back to back from the run's
start (`spin_probe`), it saw HALTED 0.023 ms after the end, for 84.1 wall tok/s with the host
equally loaded. So the overshoot is the host's sleep waking up late, not the card.
lfm2-cycles therefore starts the back-to-back reads earlier: POLL_EARLY 0.5 ms + 1% -> 1.5 ms + 3%.

**Production image (2026-09-28): `build/deploy_prod120hp_aebb0bf0`** (branch host-path-fx aebb0bf:
the host-path target (the 4-bit MXU with PAIR, r7-apf's DRAM path (64-entry store queue,
read-merge-write, port-A prefetch, QST word writes, CHASH), MM replay, vt-tile, the resident
decode's run-argument registers (CAPS bit25), DSTEP for Qwen3.5's DeltaNet and QST HALF) with
fmax c1b91dd (VPU WBUF, the RDOT row buffer in distributed RAM). Core clock 120.755 MHz,
DDR3-1066, WNS +0.004 ns (4 ps of margin), WHS +0.016 ns, no Synth 8-6430; 212.2K LUT (71.1%),
167.0K FF (28.0%), 577 BRAM36 (60.4%); Vivado's power estimate 10.1 W. The directory holds
otpu.bit, otpu.mcs, otpu.prm, reports/ and build.log; the flash has not been written with it.
The host needs main 8efceb7 or later (CHASH channel map, resident decode, streamed logits with
the WR_IDLE wait after HALTED).

Qualified on the card 2026-09-27 23:25 - 2026-09-28 00:29 (host tree host-path-fx aebb0bf):
selftest with the 12 op checks; `otpu-diag --mem full --soak 20` all pass cold (56 °C) and after
a 317 s warm soak (61 -> 63 °C, 20.98 device tok/s steady) (isa 93, system 5; RDOT / OUTER /
LOG2 all pass); Qwen3, LFM2 and Qwen3.5 token for token against the ISA simulator at int8 and
4-bit + int8 head; resident decode equal to the per-position programs (64 greedy tokens, Qwen3
and LFM2 at both formats). One host failure in that session: a streamed-logits tail read found
unwritten words once (LFM2 4-bit, sampled, host aebb0bf); it did not recur in 30 runs at hosts
7a016f1 and 4d2af30 (12 LFM2 + 3 Qwen3 sampled seeds and a greedy set each, 2026-09-28 01:25 -
02:06), and 4d2af30 waits for the memory adapter's writes (STATUS WR_IDLE) after HALTED.

Measured 2026-09-28 02:08-02:26, host main 8efceb7. Decode: `tools/decode_profile.py`, greedy,
96 tokens, logits streamed during the run, resident decode where marked. Prefill: a 512-token
prompt. DRAM: the card's counters over 64 decode tokens, bytes / running time (peak 17.1 GB/s).

| Model | Weights | Resident decode | Mcycles/token | device tok/s | wall tok/s | prefill device tok/s | prefill wall tok/s | DRAM read per token | DRAM while decoding |
|---|---|---|---|---|---|---|---|---|---|
| LFM2.5-230M | int8 | yes | 2.17 | 55.74 | 54.20 | 150.0 | 76.8 | 245 MB | 13.33 GB/s (78%) |
| Qwen3-0.6B | int8 | yes | 5.75 | 21.02 | 20.89 | 51.3 | 48.5 | 663 MB | 13.34 GB/s (78%) |
| Qwen3.5-0.8B | int8 | no | 7.86 | 15.37 | 15.21 | 36.7 | 23.1 | 811 MB | 12.68 GB/s (74%) |
| LFM2.5-230M | 4-bit, int8 head | yes | 1.45 | 83.12 | 80.84 | 169.3 | 80.2 | 164 MB | 13.14 GB/s (77%) |
| Qwen3-0.6B | 4-bit, int8 head | yes | 3.81 | 31.72 | 30.68 | 58.0 | 52.0 | 443 MB | 13.15 GB/s (77%) |
| Qwen3.5-0.8B | 4-bit, int8 head | no | 5.70 | 21.18 | 20.80 | 40.9 | 26.2 | 562 MB | 12.21 GB/s (72%) |

Qwen3.5 decodes with per-position programs: its image has no resident decode tables
(`has_lookup` is false for its Spec). Prefill wall includes compiling each chunk's program on the
host; Qwen3.5's (DSTEP rows) is host-bound. One run each.

The previous production image:

**Production image until 2026-09-28 (2026-09-27, evening): `build/deploy_prod120fp4_ea3bc560`** (branch fp4-fx
ea3bc56: the full-rate 4-bit MXU (fp4-rebase) with fmax c1b91dd: the fmax fixes of b01b8ac, the
VPU write buffer (WBUF) and the RDOT row buffer in distributed RAM, whose block RAM mapping made
vg125 / wb120 / fp4f125 return every RDOT one result late). Core clock 120.755 MHz, DDR3-1066,
WNS +0.149 ns, WHS +0.016 ns, no Synth 8-6430; 167.6K LUT (56.1%), 133.4K FF (22.3%), 561 BRAM36
(58.7%); Vivado's power estimate 9.1 W. CAPS: 4-bit MM and PAIR; no resident decode (bit25) or
DSTEP. The directory holds otpu.bit, otpu.mcs, otpu.prm, reports/ and build.log; the flash has
not been written with it. Qualified on the card 2026-09-27 20:06-21:14 (JTAG load, host tree
host-path-fx a75d2e8 + host-path 21297d0):

- selftest all pass, including the vops stage's 12 op checks; `otpu-diag --mem full --soak 20`
  all pass cold (48 °C) and again after the warm soak (platform 9, regs 7, i2c 4, mem 13, isa
  93, system 5; every RDOT / OUTER / LOG2 check passes);
- Qwen3, LFM2 and Qwen3.5 match the ISA simulator token for token at int8 and with 4-bit
  (fp4) layers and an int8 LM head (Qwen3.5's prompt through chunked prefill);
- warm soak: 326 s of continuous Qwen3 decode (12 replies of 256 tokens, 18.75 device tok/s
  throughout), board 53 -> 55 °C, then the diag above and Qwen3 token for token again.

Greedy decode (`tools/host_path_card.py`, 96 tokens, logits streamed during the run; the
greedy replies are the same tokens with and without streaming), prefill of a 512-token prompt,
and DRAM traffic from the card's counters over 64 decode tokens (bytes x the running time; peak
17.1 GB/s):

| Model | Weights | Mcycles/token | device tok/s | wall tok/s | prefill device tok/s | DRAM read per token | DRAM while running |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | int8 | 6.41 | 18.82 | 18.58 | 39.6 | 651 MB | 11.83 GB/s (69%) |
| LFM2.5-230M | int8 | 2.32 | 51.88 | 48.52 | 124.8 | 243 MB | 12.40 GB/s (73%) |
| Qwen3.5-0.8B | int8 | 8.44 | 14.31 | 14.17 | 26.1 | 814 MB | 11.87 GB/s (70%) |
| Qwen3-0.6B | fp4, int8 head | 4.52 | 26.75 | 24.92 | 44.2 | 431 MB | 10.95 GB/s (64%) |
| LFM2.5-230M | fp4, int8 head | 1.63 | 74.21 | 59.71 | 141.0 | 162 MB | 11.71 GB/s (69%) |
| Qwen3.5-0.8B | fp4, int8 head | 6.87 | 17.59 | 16.86 | 27.9 | 560 MB | 10.12 GB/s (59%) |

Mcycles/token are the selftest's one-token runs; the DRAM rows' own 64-token decode measured
6.67 / 2.38 / 8.49 / 4.78 / 1.68 / 6.92 (a longer reply, a longer context). One run each. At
4-bit the host is the gap on LFM2: 1.0-3.0 ms per token on the critical path against a 13.5 ms
run (the host-path branch works on it).

The previous production image:

**Production image until the evening (2026-09-27, afternoon): `build/deploy_prod120_b01b8acb`** (branch fmax
b01b8ac: main 32d900b plus the fmax fixes: fanout caps, the AXI adapter's request queues in LUT
RAM from r7-apf, registered MXU inputs and ACT RAM writes, a reset register per unit). Core
clock 120.755 MHz, DDR3-1066, WNS +0.080 ns, WHS +0.016 ns; 158.6K LUT (53.1%), 128.5K FF
(21.5%), 558 BRAM36 (58.4%); Vivado's power estimate 9.3 W. The directory holds otpu.bit,
otpu.mcs, otpu.prm, reports/ and build.log; the flash has not been written with it. On the card
(JTAG load, host of 79f07d7):

- selftest all pass (including RDOT / OUTER / LOG2); `otpu-diag --mem full --soak 20` all pass
  (platform 9, regs 7, i2c 4, mem 13, isa 93, system 5);
- Qwen3, LFM2 and Qwen3.5 match the ISA simulator token for token;
- warm soak: 307 s of continuous Qwen3 decode (12 replies of 256 tokens, 18.92 device tok/s
  throughout), board temperature 51 -> 55 °C, then `otpu-diag --mem full --soak 20` all pass
  again and Qwen3 still matches the simulator.

Greedy decode, 64 tokens at a short context (`tools/decode_profile.py`; the host converts cycles
with the bitstream's CORE_KHZ):

| Model | Mcycles/token | device tok/s | wall tok/s | 100 MHz image (b2c7ce43): Mcycles, device, wall |
|---|---|---|---|---|
| Qwen3-0.6B | 6.33 | 19.07 | 17.94 | 6.85, 14.6, 14.1 |
| LFM2.5-230M | 2.30 | 52.41 | 48.64 (48.15-48.94, 3 runs, host 01246cf) | 2.46, 40.7, 38.5 |
| Qwen3.5-0.8B | 8.38 | 14.41 | 13.39 | 9.75, 10.3, 9.9 |

The cycles per token also dropped (Qwen3 -7.6%): this image carries r5-dram (MIG
ROW_BANK_COLUMN and gathered QST writes) and the adapter queues, which b2c7ce43 did not. On LFM2
the host adds about 1.3 ms per token (logits read 0.4-0.6 ms, sampling 0.2-0.3 ms, the input
write 0.1-0.2 ms, poll overshoot 0.25 ms); a first single run read 42.8 wall tok/s, which the
three repeats did not reproduce.

At a long context the host matters more. LFM2 with a ~1,800-token prompt and 94 decode tokens
(context ~1,900) on this image: device 47.9 tok/s (2.52 Mcycles/token); wall 17.8 tok/s with the
host of 79f07d7 (the step waits 33.4 ms per token for its program to compile) and 33.3 tok/s with
the host of 01246cf (longctx; the wait drops to 6.1 ms).

**4-bit weights on the card (2026-09-27): `deploy_fp4f125_cf3b6093`** (branch fmax-fp4 cf3b609:
fp4-rebase 7439b0d with the full-rate 4-bit MXU (PAIR), the fmax fixes and VPU WBUF; 125.49 MHz,
DDR3-1066, WNS +0.067 ns). Not production: it has vg125's RDOT fault (the four RDOT diag checks
fail one result late), so Qwen3.5 was not run. Host e16f272 (the compile worker builds the image
in the engine's weight formats; before it, 4-bit decode programs were compiled as int8 and Qwen3
fp4 answered '!!!!'). Greedy, 64 tokens (`tools/decode_profile.py`); every configuration matches
the ISA simulator token for token (`otpu-selftest --model ... --wformat ...`):

| Model | Weights | Mcycles/token | device tok/s | wall tok/s |
|---|---|---|---|---|
| Qwen3-0.6B | int8 | 6.36 | 19.72 | 18.61 |
| Qwen3-0.6B | fp4, int8 LM head | 4.48 | 27.98 | 25.65 |
| Qwen3-0.6B | fp4 | 3.83 | 32.76 | 29.64 |
| LFM2.5-230M | int8 | 2.31 | 54.28 | 50.06 |
| LFM2.5-230M | fp4, int8 LM head | 1.62 | 77.57 | 59.89 |
| LFM2.5-230M | fp4 | 1.33 | 94.10 | 85.44 |

One run each (the LFM2 int8 row is from the same image an hour earlier, host bbd4886).
Accuracy of the 4-bit formats is in docs/quant.md.

**Not promoted: `deploy_vg125_4b9ab8ad`** (fmax-vg125 4b9ab8a, 125.49 MHz, DDR3-1066, WNS
+0.017 ns, WHS +0.045 ns; the fmax image plus a registered-ahead VPU TMEM grant, "VPU WBUF").
Selftest, the three models token for token (Qwen3.5 included) and decode (6.36 / 2.31 / 8.42
Mcycles/token, 19.7 / 54.3 / 14.9 device tok/s) all pass, but `otpu-diag` fails all four RDOT
programs 20 times out of 20, cold and warm, and after the warm soak also the RDOT / OUTER / LOG2
system program. Each RDOT result is the one the previous RDOT program should have stored: a
result one operation late, deterministic, so a logic fault in the WBUF change rather than a
timing margin. The selftest of that time passed it with a note (it took any RDOT mismatch for a
bitstream built before RDOT); it now fails wrong RDOT results on register map 3 or later.

The 100 MHz image it replaces:

**Production image (2026-09-27): `build/deploy_prod1066_b2c7ce43`** (main b2c7ce4, DDR3-1066,
WNS +0.085 ns, PCI class 12 00 00, I2C). On the card: calibration, selftest, `otpu-diag --mem
full --soak 20` all pass (127 checks), and Qwen3 / LFM2 / Qwen3.5 match the ISA simulator token
for token at 6.85 / 2.46 / 9.75 Mcycles/token.

**Measured on the card (2026-09-27, JTAG loads, host code of a691ea98).**

| Image | Calibration | selftest | diag memory | Qwen3 decode |
|---|---|---|---|---|
| DDR3-800, burst (a691ea98) | ok | all pass | all pass | 8.58 Mcycles/token, 11.65 tok/s, DRAM 7.2 GB/s, MXU_STARVE 37% |
| DDR3-1066, in spec (a691ea98) | ok (both channels) | all pass | 13 / 13 pass, `--mem full --soak 20` | 6.85 Mcycles/token (-20%) |
| DDR3-1300, out of spec (819fee49) | ok (both channels) | all pass | 11 / 11 pass | 6.50 Mcycles/token, 15.38 tok/s, DRAM 9.53 GB/s, MXU_STARVE 18% |
| DDR3-1333, out of spec, patched PHY (254f8388) | ok (both channels) | fails at the DMA bandwidth stage (H2C timeout), then the card leaves the PCIe bus (ID 0xffffffff) | not run | not run |

The 1333 failure followed the selftest's 200 sub-beat host writes, the trigger of the host-write
hang being bisected (docs/host.md), so it is not yet a clean DDR verdict; the loss of the PCIe
link is worse than that hang and makes 1333 suspect regardless.

DDR3-1300 then passed the model and soak checks (2026-09-27, one run, card at room temperature
after ~30 minutes of builds and tests): `otpu-diag --mem full --soak 20` all pass (platform 9,
regs 7, mem 13, isa 93, system 5), and the three models match the ISA simulator token for token:

| Model | Mcycles/token at 1300 | device tok/s | wall tok/s (host of a691ea98) |
|---|---|---|---|
| Qwen3-0.6B | 6.50 (800: 8.58) | 15.4 | 13.6 |
| LFM2-350M | 2.34 (800: 3.14) | 42.7 | 23.5 |
| Qwen3.5-0.8B | 9.08 (800: 11.88) | 11.0 | 7.5 |

DDR3-1066, inside MIG's range for these banks, passed the same checks the same day (a691ea98,
WNS +0.107 ns). It gets most of 1300's gain:

| Model | Mcycles/token at 1066 | device tok/s | wall tok/s (host of a691ea98) |
|---|---|---|---|
| Qwen3-0.6B | 6.85 | 14.6 | 12.6 |
| LFM2-350M | 2.46 | 40.7 | 21.8 |
| Qwen3.5-0.8B | 9.75 | 10.3 | 7.4 |

With the host code of 7f9cec1 (the next program compiles in a worker process after the card
starts; docs/host.md), `tools/decode_profile.py` on the same image (96-token reply) measures
wall 38.5 / 14.1 / 9.9 tok/s against device 40.5 / 14.5 / 10.2 for LFM2 / Qwen3 / Qwen3.5: the
host adds 1.3 / 1.9 / 2.7 ms per token, mostly the logits read and the sampling.

DDR3-1300 is still not qualified: MIG's ECC correction counters were not read (a marginal link corrects
silently), the warm soak (step 2) was not run, and it is outside MIG's range for these banks.

**Checklist per speed.** Status: *unmeasured* at every speed above 800 until the results are
filled in here. Load the bitstream over JTAG, not flash (section 2), so a bad one is gone at
the next power cycle.

1. **Calibration.** `otpu-diag` rows "DDR3 calibration channel 0/1" (STATUS bits 5 and 6), and
   `otpu-smi`. A channel that does not calibrate within 5 s fails there; everything after it is
   skipped.
2. **Memory tests.** `otpu-diag --json diag_ddr<speed>.json` (walking bits, address bits, random
   blocks, partial writes per channel, and the interleave). Then run
   `otpu-diag --mem full --soak 10` (march C- over 4 GiB and repeated kernels), once cold and once
   after 10+ minutes of `otpu-chat`, since timing margins shrink as the die warms.
   `otpu-smi` shows the temperature.
3. **ECC corrections.** Single-bit errors are corrected silently, so a marginal link can pass
   every test. Read MIG's correctable-error counter (UG586 AXI ECC registers: CE_CNT at offset
   0x0C, so BAR0 0x1000C for channel 0 and 0x2000C for channel 1; unverified on this card)
   before and after the soak. It should stay 0.
4. **Bandwidth.** The DMA test is PCIe-bound (~1 GB/s) and does not show DDR3 speed. Use the
   accelerator instead: `otpu-selftest --model qwen3 --tokens 32` and `otpu-smi` during
   `otpu-chat`. Compare device Mcycles/token, DRAM read GB/s and MXU_STARVE against the
   DDR3-800 bitstream of the same commit. The 800 figure is the baseline; the speed-up is
   bounded by 12.8 GB/s.
5. **Correctness.** `otpu-selftest --model qwen3` must still match the ISA simulator token for
   token.

| DDR3 | bitstream | MIG in range | calibration | diag / soak | ECC CE | decode vs 800 |
|---|---|---|---|---|---|---|
| 800 | default | yes | passes (2026-09-26) | diag passes (2026-09-26); soak not recorded | not read | baseline |
| 1066 | `make bit DDR=1066` | yes | passes (2026-09-27) | diag + `--mem full --soak 20` pass, cold only | not read | Qwen3 -20% cycles/token, LFM2 -22%, Qwen3.5 -18% |
| 1300 (out of spec) | `make bit DDR=1300` | no (79-155) | passes (2026-09-27) | diag + `--mem full --soak 20` pass, cold only | not read | Qwen3 -24% cycles/token, LFM2 -25%, Qwen3.5 -24% |
| 1333 (out of spec) | `make bit DDR=1333` | no (79-155, PHY patched) | passes (2026-09-27) | fails: H2C timeout, card leaves PCIe | not read | not run |
| 1600 (out of spec) | `make bit DDR=1600` | no (79-155, PHY patched) | not pursued: 1333 already fails | | | |

## 6. What to check on first build (assumptions made without Vivado)

The first MIG build's checklist (the MIG builds are removed; the board facts stay true: the DDR3
pins are `constraints/ddr3_ch*.pins.xdc`, LiteDRAM's in `litedram/otpu_litedram.xdc`).

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
