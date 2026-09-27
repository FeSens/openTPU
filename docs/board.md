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
make bit DDR=1300  # DDR3-1300 / 1333 / 1600: OUT OF SPEC (outside MIG's range for these HR
                   # banks; 1333 and 1600 also patch MIG's PHY). Experiments only, never a
                   # default; see "Faster DDR3" in section 5
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
| **`build/deploy_burst_a691ea98/otpu.bit`** (primary) | `make bit` (MCOLS=2, VPU_CL=2, LANES=8) | a691ea98 (port-B AXI read bursts, MXU_STARVE counter; register map 3) | D=128 MCOLS=2 LANES=8 | yes | Qwen3, LFM2, Qwen3.5 | met, WNS +0.085 ns, WHS +0.040 ns (omarchy build) |
| `build/deploy_r3route_74d4859/otpu.bit` (single-beat reads, 2.4x slower decode) | `make bit` (MCOLS=2, VPU_CL=2, LANES=8) | 74d4859 (chunk FIFO in block RAM, 6 TMEM copies) | D=128 MCOLS=2 LANES=8 | yes | Qwen3, LFM2, Qwen3.5 | met, WNS +0.104 ns, WHS +0.038 ns |
| `build/deploy_default_3c270c9/otpu.bit` (first fallback) | `make bit` (MCOLS=2, VPU_CL=2, LANES=8) | 3c270c9 | D=128 MCOLS=2 LANES=8 | yes | Qwen3, LFM2, Qwen3.5 | met, WNS +0.065 ns, WHS +0.038 ns |
| `build/deploy_m4cl4_550aa35/otpu.bit` (faster prefill) | `make bit MCOLS=4 VPU_CL=4` | 550aa35 (as the primary, 4 MXU columns, 4 composite VPU lanes) | D=128 MCOLS=4 LANES=8 | yes | Qwen3, LFM2, Qwen3.5 | met, WNS +0.086 ns, WHS +0.037 ns |
| `build/vivado_100mhz_gen1_met/otpu.bit` (fallback) | `make bit` (MCOLS=2, VPU_CL=2, LANES=8) | v0.4 (6587cb4) | D=128 MCOLS=2 LANES=8 | no | Qwen3, LFM2 | met, WNS +0.082 ns |

Start with the primary image; if it misbehaves where the 3c270c9 image does not, the chunk FIFO
/ TMEM change is the suspect (same programs, same cycles in simulation). The 4&4 build is the same instruction set with twice the MXU
columns (the host picks MCOLS=4 up from VERSION); the v0.4 build is the last resort. All are
100 MHz core, DDR3-800, PCIe Gen1 x8, register map 2. The RTL changes after ddec900
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
the card yet: the flash still holds that image.

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
before that show class 07 00 01 (serial), subsystem 10ee:0007, revision 00.

If the device does not appear, warm-reboot the host (the FPGA keeps its configuration across a
warm reboot, as long as the slot power stays on), or write the flash, then power-cycle. If the
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

### Faster DDR3

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

**DDR3 voltage (unknown).** The MT41K256M8 is DDR3L: it runs at 1.35 V or 1.5 V. The board's
VDDQ and the banks' VCCO have not been measured. Every build uses 1.5 V (SSTL15), as the
reverse-engineered MIG project (TiferKing's systest example, DDR3-1066 at 1.5 V, external VREF)
does. DDR3-800 calibrates with MIG's internal VREF at 0.75 V, VCCO / 2 for 1.5 V. At 1.35 V
that VREF would sit 75 mV high, and calibration at 800 might still pass. So the evidence for
1.5 V is only suggestive: a working calibration at 1066 would support it. The direct check is a
multimeter on a DDR3 decoupling capacitor (VDDQ 1.5 or 1.35 V) and on a bank VREF pin (half of
it). If the board is at 1.35 V, MIG's limit drops to DDR3-800 and the SSTL15 settings are wrong.

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
| 1600 (out of spec) | `make bit DDR=1600` | no (79-155, PHY patched) | unmeasured | unmeasured | unmeasured | unmeasured |

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
