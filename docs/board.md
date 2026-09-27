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
make bit DDR=1333  # DDR3-1333 / 1600: outside MIG's range for these (HR) banks, experiments
                   # only; see "Faster DDR3" in section 5
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
| **`build/deploy_r3route_74d4859/otpu.bit`** (primary) | `make bit` (MCOLS=2, VPU_CL=2, LANES=8) | 74d4859 (chunk FIFO in block RAM, 6 TMEM copies) | D=128 MCOLS=2 LANES=8 | yes | Qwen3, LFM2, Qwen3.5 | met, WNS +0.104 ns, WHS +0.038 ns |
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

Decode runs at ~3.2x the simulated cycles: the counters show the MXU starved (MXU_BUSY 94%,
MXU_MAC 21%, DRAM_WAIT 0.1%) and DRAM reads at 3.0 GB/s (0.23 beats / cycle / channel). Port B
issues single-beat 64-byte AXI reads (SmartConnect ports MAX_BURST_LENGTH 1); the per-transaction
cost in SmartConnect and the MIG AXI front end, which the simulated DRAM does not charge, caps the
rate. Fix in progress: burst reads on port B. Host DMA: 0.78 GB/s host -> card, 1.12 GB/s back.

Found at bring-up, fixed in the host (558a4bf): the DRAM must be written once after configuration
(ECC: a read of a never-written beat hangs; `Board.scrub`, ~6 s, done by every tool), register
access must be single 32-bit loads / stores, and DMA reads must not use `preadv` (the XDMA driver's
asynchronous read_iter crashes kernel 7.1). The XDMA's PCI class code is "serial controller",
so the 8250 driver probes the card: a udev rule sets `driver_override=xdma` (a class code in
bd.tcl is the real fix, pending).

## 5. Clocks and the roofline

| Clock | Frequency | Source | Drives |
|---|---|---|---|
| sys_clk_50 | 50 MHz | AA28 oscillator | MMCM (VCO 800 MHz; 1000 MHz at DDR3-1333) |
| core_clk | 100 MHz | MMCM /8 (/10) | accelerator, control registers, interconnect core side |
| clk_200 | 200 MHz | MMCM /4 (/5) | MIG reference (IDELAYCTRL); MIG system clock at DDR3-800 and 1600 |
| clk_mig | VCO / 3 | MMCM /3 | MIG system clock at DDR3-1066 (266.667 MHz) and 1333 (333.333 MHz) |
| ui_clk0/1 | 100 MHz (133 / 167 / 200 at 1066 / 1333 / 1600) | MIG | MIG AXI side, 512 bit |
| DDR3 CK | 400 MHz (533 / 667 / 800) | MIG PLL | memory |
| axi_aclk | 125 MHz | XDMA | PCIe AXI side, 128 bit (Gen1 x8) |

Decode is DRAM-bound: every token streams all weights once. The accelerator consumes one
128-byte chunk per core cycle at its peak; each DDR3 channel (x64) delivers 8 bytes per CK edge.

| DDR3 | channel peak | both channels | core clock to match (128 B/cycle) | MIG ui_clk |
|---|---|---|---|---|
| 800 (default) | 6.4 GB/s | 12.8 GB/s | >= 100 MHz | 100 MHz |
| 1066 | 8.5 GB/s | 17.1 GB/s | >= 133 MHz | 133 MHz |
| 1333 (out of MIG's range) | 10.7 GB/s | 21.3 GB/s | >= 167 MHz | 167 MHz |
| 1600 (out of MIG's range) | 12.8 GB/s | 25.6 GB/s | >= 200 MHz | 200 MHz |

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

The 1333 and 1600 figures in the same file (`tmin_hp_18` 1500 ps, `tmin_hp_20` 1072 ps) are for
HP banks, where VCCAUX_IO matters. Neither applies to this board. The .prj keeps 1.5 V (SSTL15),
the setting DDR3-800 calibrates with; the board's DDR3 supply is not known. Internal VREF is
MIG's option only up to 800, so 1066 and up rely on an external VREF. The VREF pins of the six
DDR3 banks are free of DDR3 signals, which fits an external VREF, but nobody has measured it.

MIG's messages when it imports the generated .prj (Vivado 2026.1, mig_7series 4.2):

- 1066: `[Mig7series 79-144] Invalid Input Clock Period 266.667. Setting to nearest possible
  Input Clock Period value 266.666.` It is a rounding only, and the configuration is supported.
- 1333: `CRITICAL WARNING: [Mig7series 79-155] Memory Time Period (1500 ps) (666.666687 Mhz) is
  not supported for MIG. There has been a change in the allowed frequency ranges as described in
  Answer Record 67179. Instantiate and customize a new instance of MIG for your design.` With a
  200 MHz input, MIG also gives `[Mig7series 79-144] Invalid Input Clock Period 200.000. Setting
  to nearest possible Input Clock Period value 205.128.`, so the 1333 build runs the MMCM at a
  1000 MHz VCO and feeds MIG 333.333 MHz, which it accepts as is (PLL x4).
- 1600: the same 79-155 for `1250 ps (800.000000 Mhz)`. A 200 MHz input is accepted (PLL x8).

These are critical warnings, not errors: MIG still generates the controller (CL 9 / CWL 7 at
1333, CL 11 / CWL 8 at 1600, tRFC 160 ns). So 1333 and 1600 bitstreams can be built, but they run
the HR I/O and the MIG PHY beyond what AMD characterizes. Whether they calibrate, and whether the
data stays correct over temperature, only the card can show.

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
| 1066 | `make bit DDR=1066` | yes | unmeasured | unmeasured | unmeasured | unmeasured |
| 1333 | `make bit DDR=1333` | no (79-155) | unmeasured | unmeasured | unmeasured | unmeasured |
| 1600 | `make bit DDR=1600` | no (79-155) | unmeasured | unmeasured | unmeasured | unmeasured |

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
