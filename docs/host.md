# Host PC: driving the openTPU card over PCIe

The YPCB-00338 card runs the accelerator behind a Xilinx XDMA PCIe bridge (Gen1 x8). This page
covers the host side: the PC the card is plugged into, its driver, and the tools that talk to
the card. Building and loading the bitstream is in [board.md](board.md); the registers,
counters and trace buffer the tools read are specified in [observability.md](observability.md).

What the host sees:

| Device node | What it is | Used for |
|---|---|---|
| `/dev/xdma0_user` | BAR0, the AXI-Lite control registers (`rtl/boards/ypcb-00338/otpu_ctrl.sv`) | start / load / status / counters |
| `/dev/xdma0_h2c_0` | DMA host -> card; file offset = card AXI address | writing DRAM |
| `/dev/xdma0_c2h_0` | DMA card -> host; file offset = card AXI address | reading DRAM |

The card's AXI address map: DDR3 channel 0 at `0x0000_0000`, channel 1 at `0x8000_0000`,
2 GiB each. The accelerator sees one 4 GiB logical DRAM interleaved over the two channels in
64-byte beats: logical beat *b* of chunk *m* = *b* // 2 is at offset *m* * 64 on channel
*b* % 2, or, on a bitstream with CAPS bit7 (CHASH), on channel (*b* % 2) XOR parity(*m*): each
128-byte chunk still has one beat on each channel, but the chunks with an odd number of set
bits have them swapped, so a column at a power-of-two stride (the transposed V cache: one byte
per cache row) is spread over both channels instead of loading one. `opentpu/host/board.py`
reads CAPS and applies the map, so everything above it uses logical addresses.

The host software is the package `opentpu/host` (userspace; the kernel side is the Xilinx
XDMA driver, installed by `otpu-setup`). `pip install -e .` installs its commands:

| Command | What it does |
|---|---|
| `otpu-setup` | the PC's setup: XDMA driver, udev rules, rescan after JTAG (sections 2-3) |
| `otpu-smi` | the cards' state, like nvidia-smi (section 8) |
| `otpu-selftest` | staged bring-up (section 4) |
| `otpu-diag` | the full hardware diagnostic: every check, no stopping, a works / does-not-work matrix (section 5) |
| `otpu-chat` | chat with Qwen3, LFM2 or Qwen3.5 on the card (section 6) |
| `otpu-lens` | Lens profiles from the card's hardware trace ([lens.md](lens.md)) |
| `otpu-i2c` | the card's I2C buses, read only: `scan`, `read`, `pmbus-dump`, `levels` ([observability.md](observability.md#i2c-pins)) |

Without installing, `python3 -m opentpu.host.<smi|selftest|chat|hwlens|i2c>` does the same.

## 1. Requirements

- A Linux x86-64 PC with a free x8 (or x16) PCIe slot. The card takes power from the slot;
  make sure the slot provides enough power and there is airflow over the heatsink.
- Kernel headers for the running kernel, `gcc`, `make`, `git`, `patch`, and preferably DKMS
  (section 2 lists the packages).
- Python 3.10+ with `numpy`; for the model also `torch`, `transformers`, `safetensors`.
- This repository, and the models in `models/` (Hugging Face checkpoints, e.g.
  `huggingface-cli download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B`; likewise
  `LiquidAI/LFM2.5-230M` -> `models/LFM2.5-230M`, `Qwen/Qwen3.5-0.8B` -> `models/Qwen3.5-0.8B`).
- No configuration: the tools read the bitstream's D / MCOLS / LANES from its VERSION register.
  Leave `OTPU_MCOLS` / `OTPU_LANES` unset (they configure the simulators); when set, they must
  match the bitstream or the tools stop, naming both values.

## 2. Install the driver: otpu-setup

`sudo otpu-setup` (the script `opentpu/host/setup_pcie.sh`, shipped in the package) sets up
the PC once; re-running it is safe, it changes only what differs:

- the Xilinx XDMA driver from `dma_ip_drivers` at a pinned commit (b846609, driver version
  2025.2.0) with `opentpu/host/pcie/xdma-otpu.patch`: a Makefile fix for kernels >= 6.13
  (`$(src)` is relative there, and `EXTRA_CFLAGS` is ignored) and an `otpu` module tag that
  `--check` reads. Source in `/usr/src/otpu-xdma-b846609.1`.
- with DKMS installed, the module is a DKMS package (`otpu-xdma`), rebuilt at every kernel
  upgrade; without DKMS it goes to `/lib/modules/$(uname -r)/updates/xdma.ko`, and after a
  kernel upgrade you re-run `sudo otpu-setup` (`otpu-setup --check` reports the missing module).
- the name clash: the kernel ships its own `xdma` module (`drivers/dma/xilinx`, AMD's platform
  DMA driver for Alveo cards, not a PCI driver for this card). Ours, in `updates/`, takes
  precedence (depmod's search order). DKMS also moves the kernel's file aside into
  `/var/lib/dkms/otpu-xdma/original_module` and puts it back at `--uninstall`.
- udev rules (`opentpu/host/pcie/*.rules`, installed in `/etc/udev/rules.d/`):
  `59-otpu-xdma.rules` sets `driver_override=xdma` on the card and loads the driver;
  `60-otpu.rules` gives non-root access to `/dev/xdma*_user`, `_h2c_*`, `_c2h_*`, `_events_*`
  and to the JTAG cables (FT232H 0403:6014, Platform Cable USB II 03fd:*). `/dev/xdma*_control`,
  `_xvc` and `_bypass` stay root-only: `_control` programs the DMA engines, which can then
  write anywhere in host memory.
- the driver's completion mode in `/etc/modprobe.d/otpu-xdma.conf` (see below).
- then it loads the driver, binds the card and reads the ID register ("OTPU").

```sh
pip install -e .                 # the otpu-* commands, otpu-setup included
sudo otpu-setup                  # install / update (otpu-setup asks for sudo itself)
otpu-setup --check               # what is installed and working; exit 1 if anything is not
sudo otpu-setup --rescan         # after a JTAG load (section 3)
sudo otpu-setup --uninstall      # remove it all again
otpu-setup --help                # the other options: --poll, --irq, --no-dkms, --src DIR
```

The driver is never unloaded under a running program: if the card is in use, the install
says so and the new module or mode takes effect at the next `--rescan` or reboot.
`otpu-setup --check` on the development PC after a fresh install and `--rescan` (Arch Linux,
kernel 7.1.4, 2026-09-27; kernel and JTAG lines left out):

```
== driver module
   ok    /lib/modules/7.1.4-arch1-1/updates/dkms/xdma.ko.zst (openTPU build b846609.1)
   ok    DKMS: otpu-xdma/b846609.1, 7.1.4-arch1-1, x86_64: installed (Original modules exist)
   ok    loaded, poll_mode=0 (interrupts)
== configuration
   ok    /etc/udev/rules.d/59-otpu-xdma.rules
   ok    /etc/udev/rules.d/60-otpu.rules
   ok    /etc/modprobe.d/otpu-xdma.conf: poll_mode=0
== card 0000:01:00.0: [10ee:7028] subsystem 10ee:0007 revision 00
   ok    class 0x070001: 16450 serial port (bitstream before the PCI identity change; ...)
   ok    link 2.5 GT/s PCIe x8 (the design: Gen1 x8)
   ok    bound to xdma
   ok    /dev/xdma0_{user,h2c_0,c2h_0} usable by bonetto
   ok    ID register 0x4f545055 (OTPU)

all in place
```

**Distributions.** Arch: `pacman -S base-devel linux-headers dkms` (the headers of your
kernel flavor, e.g. `linux-lts-headers`). Debian / Ubuntu: `apt install build-essential
linux-headers-$(uname -r) dkms`. With Secure Boot on, the kernel only loads signed modules:
DKMS signs with its own key (`/var/lib/dkms/mok.pub`), which you enroll once with
`sudo mokutil --import /var/lib/dkms/mok.pub` and a reboot. Tested: Arch Linux with kernel
7.1.4 (install, DKMS, card). Build only, in containers without a card: Ubuntu 24.04 (headers
6.8.0-142, gcc 13) and 22.04 (5.15.0-194, gcc 11), both the plain build and the DKMS install.

**PCI identity.** Bitstreams built before the PCI identity change in `bd.tcl` report class
07 00 01 (a 16450 serial port) with subsystem 10ee:0007 and revision 00. For those, the kernel's
8250_pci driver probes the card first (seen in dmesg as `serial 0000:01:00.0: enabling device`)
and fails, and the udev rule then hands the card to xdma. Newer bitstreams report class
12 00 00 (processing accelerator), subsystem 10ee:4f54 and revision 01, which no other driver
claims. The device ID stays 7028 in both, as in the driver's `pci_ids[]`.

**Poll or interrupt mode.** The driver waits for a DMA transfer to finish either by interrupt
(`poll_mode=0`, the default) or by polling the engine's write-back status from its completion
threads (`poll_mode=1`). Bring-up ran in poll mode because a 64-byte read timed out in
interrupt mode. That turned out to be the ECC scrub (a read of never-written DRAM hangs; see
board.md, "First light"), not the interrupts. Measured on the card (build 74d48591, Arch
Linux 7.1.4, 2026-09-27; the driver takes one MSI vector):

| | interrupts (`poll_mode=0`) | polling (`poll_mode=1`) |
|---|---|---|
| `otpu-selftest` | all PASS | all PASS |
| `otpu-diag` | 124 PASS, 1 SKIP (power: no report) | 124 PASS, 1 SKIP |
| `otpu-diag --mem full --soak 10` (5.6 min) | 127 PASS, 1 SKIP | not run |
| model stage, 32 tokens: LFM2.5-230M / Qwen3-0.6B | 10.26 / 4.57 tok/s wall | 10.28 / 4.60 tok/s wall |
| 64 B DMA write / read, median per call | 11.7 / 11.3 us | 6.7 / 6.2 us |
| 4 KiB write / read | 14.7 / 13.7 us | 9.8 / 8.7 us |
| 64 KiB write / read | 95.5 / 49.5 us | 89.7 / 44.4 us |
| 8 MiB write / read | 1.69 / 1.63 GB/s | 1.76 / 1.66 GB/s |
| 64 MiB write / read | 0.77 / 1.15 GB/s | 0.77 / 1.17 GB/s |
| CPU during 64 MiB transfers (process + driver threads) | 0.03 / 0.04 cores | 0.98 / 1.01 cores |
| XDMA timeouts / errors in dmesg | 0 | 0 |

Interrupts cost about 5 us more per transfer and nothing measurable in tokens/s; polling
keeps a CPU core busy for as long as a transfer runs. So the default is interrupts. If DMA calls
hang with XDMA timeouts in `dmesg` on some other PC (interrupts not delivered), switch with
`sudo otpu-setup --poll` (and back with `--irq`); it reloads the driver when the card is idle.
The table comes from `tools/dma_bench.py` (latency, bandwidth, CPU), `otpu-selftest` and
`otpu-diag` run in each mode.

The bandwidth rows were measured before the buffer placement fix below. Then, whether a
write ran at 1.7 or 0.77 GB/s depended on where the benchmark's buffer happened to land in
memory, not on the mode.

**DMA and buffer placement.** At PCIe Gen1 x8 the DMA runs at 1.7 GB/s for writes and 1.65 GB/s
for reads from 1 MiB per call up; below that the per-call cost dominates (64 KiB: 1.3 /
1.0 GB/s, 4 KiB: 0.27 / 0.28 GB/s). Whether a transfer gets that speed depends on where the
host buffer sits relative to the card address, d = host address - card address (measured
with 8 MiB transfers, 2026-09-27, interrupt mode, no IOMMU on this PC):

| d | write (h2c) | read (c2h) |
|---|---|---|
| d % 64 != 0 (e.g. 16, 32, 48) | 0.77 GB/s | 1.16 GB/s at d % 4096 = 16 or 4080; 1.6 GB/s at 32, 48 |
| d % 4096 = 0 (page-aligned buffer, page-aligned card address) | 1.70 GB/s | 1.16 GB/s |
| d % 64 = 0 and 32 <= d % 4096 <= 4064 | 1.70 GB/s | 1.62-1.69 GB/s |

The size, the chunking (one 8 MiB call vs 8 x 1 MiB) and the card offset alone make no
difference. The slow cases are what ordinary buffers hit: a large NumPy array starts 16 bytes
past a page boundary (glibc's mmap chunk header), so every write through it ran at 0.77 GB/s,
and so did `Board.scrub` and every weight upload. That was the anomaly: `tools/dma_bench.py`
showed 8 MiB writes at 1.7 GB/s and 1 or 64 MiB ones at 0.77 GB/s only because its 8 MiB
buffer happened to land differently. `XdmaTransport` now puts its buffers at d % 4096 = 2048
(`board.DMA_PLACE`): reads it allocates are placed that way; writes and reads from a buffer
that is placed badly go through a persistent staging buffer (a copy runs at ~20 GB/s).
`Board.write` / `Board.read` of more than 16 MiB split the data into the two channels' placed
buffers in pieces and run the DMA in a worker thread, overlapping the interleave copies with
the transfer. Measured with the same scripts before and after (build 74d48591; the "after"
column repeated on the burst build a691ea98 gave the same numbers within 2%):

| | before | after |
|---|---|---|
| `Board.scrub` (4 GiB) | 5.55 s (0.77 GB/s) | 2.55 s (1.68 GB/s) |
| `Board.write`, 263 MB (the LFM2.5-230M weight image) | 0.40 s (0.65 GB/s) | 0.18 s (1.50 GB/s) |
| `Board.read`, 263 MB | 0.31 s (0.84 GB/s) | 0.17 s (1.51 GB/s) |
| `tools/dma_bench.py` 64 MiB write / read (transport) | 0.77 / 1.16 GB/s | 1.50 / 1.45 GB/s |

The rest of the gap to 1.7 GB/s is host work that is not overlapped: first-touch page faults
of new buffers and the staging copy on the transport path. The PC has no IOMMU enabled (no
DMAR / IOMMU groups), so address translation plays no part in these numbers.

**Sub-beat writes.** A host DMA write shorter than a 64-byte beat, or not aligned to one,
can wedge the card's write path until the FPGA is reloaded: the XDMA host->card engine stays
BUSY, every later write times out (errno 110, `timed out` in dmesg) and a driver reload or
`otpu-setup --rescan` does not clear it. Reads and registers keep working. Bisected on the card
(burst build a691ea98, 2026-09-27, one JTAG reload + scrub per case) down to two writes:

| case (channel 0, then a 4 KiB write) | result |
|---|---|
| A = 20 B at 0x100c9d, then B = 2 B at 0x100caf (same beat 0x100c80) | hangs |
| B, then A | hangs |
| A, then A again | hangs |
| A alone (then two 4 KiB writes) | passes |
| A then B on channel 1 | hangs |
| 0x100cac 31 B, then B | passes |
| aligned start: 49 B at 0x100c80, then 48 B at 0x100c80 | passes |
| aligned start: 49 B at 0x100c80, then 17 B at 0x100ca0 | passes |
| six 4-byte writes to 16-byte-aligned words of the same beat | passes |
| the same sequences from a host buffer at another 16-byte phase | hang the same way |

So the trigger is two sub-beat writes to the same beat with unaligned start addresses (the
XDMA's 128-bit master issues them as short bursts from an unaligned AWADDR, which the
SmartConnect widens to 512 bits for the MIG), followed by any write. It does not depend on
the host buffer placement. The selftest's 200 random sub-beat writes hit it on some builds
and buffer layouts (it stopped the DDR3-1333 qualification and these bisection runs), not on
others.

The fix is on the host: `XdmaTransport.mem_write` sends only whole 64-byte beats. A range
that does not start and end on a beat boundary is widened, its edge beats read and merged on
the host (`Board.write` already widened to 128 bytes). The selftest's and diag's sub-beat
checks now test that merge. The accelerator's own partial writes (QST bytes, masked stores
through `rtl/mem/otpu_axi_dram.sv`) are single 512-bit beats from a 64-byte-aligned address
(AWLEN 0, AWSIZE 6) with only the strobes varying: the aligned-start cases above, which pass,
are the closest the host can come to that shape, and the kernel stage's masked writes and every
model run have never hung. The accelerator path is very likely safe, not proven.

**IOMMU.** If DMA transfers fail on a machine with the IOMMU on, boot with `iommu=pt` (Intel:
`intel_iommu=on iommu=pt`).

## 3. Check the card on the bus

The card must be configured before the PC enumerates the bus: either boot the PC with the
bitstream already in the card's configuration flash (`make flash`, [board.md](board.md)
section 2), or program over JTAG and then rescan:

```sh
sudo otpu-setup --rescan             # remove the card, rescan, bind xdma, read the ID register
lspci -d 10ee: -nn                   # the card: "... [10ee:7028]"
sudo lspci -d 10ee: -vv | grep -E "LnkCap|LnkSta|Region"
#   LnkCap/LnkSta: Speed 2.5GT/s, Width x8  <- Gen1 x8 is the design (not a downtrained link);
#                                             fewer lanes cost DMA bandwidth only
#   Region 0: Memory at ... [size=1M]   <- BAR0, the control registers (AXI-Lite master, 1 MiB)
#   Region 1: Memory at ... [size=64K]  <- the XDMA's own registers (the driver uses them)
```

`--rescan` refuses while a program has the card open (it would pull the device out from
under it). It re-adds the card with the kernel's automatic probing off, so the 8250 driver
never touches a serial-class bitstream, and binds xdma itself.

A quick register check without Python: `dd if=/dev/xdma0_user bs=4 count=1 2>/dev/null | xxd`
must print `55 50 54 4f` ("OTPU", the ID register at offset 0, little-endian).

## 4. Self-test

```sh
otpu-selftest                                 # stages link .. vops on /dev/xdma0
otpu-selftest --model qwen3                   # plus the model stage (lfm2, qwen35, or a directory)
otpu-selftest --model qwen3 --tokens 1        # fewer generated tokens (default 8)
otpu-selftest --sim                           # rehearsal on the Verilator board model
otpu-selftest --dev /dev/xdma1 --bw-mib 128   # another card; bandwidth test size
```

(`python3 -m opentpu.host.selftest ...` is the same without installing.)

Stages, in order (it stops at the first failure and prints a hint):

| Stage | Checks |
|---|---|
| link | the ID register reads `0x4F545055` |
| config | reads D / MCOLS / LANES from VERSION and builds the host configuration from them (`device_config`); fails when `OTPU_MCOLS` / `OTPU_LANES` are set to other values, or D is not 128 |
| calib | both DDR3 controllers calibrated (STATUS bits 5, 6) |
| regs | SCRATCH register write / read |
| i2c | with CAPS.i2c: scan the LM73 bus and the PCIe SMBus (one-byte read probes, 0x08-0x77), identify the LM73 (ID 0x0190, temperature), TI INA2xx current monitors and PMBus devices (revision, MFR_ID / MFR_MODEL, READ_* telemetry); a line that stays low FAILs. Read only (`opentpu/host/i2c.py`). The result is saved for `otpu-smi`'s measured power. SKIP on the board model and without the I2C pins |
| addr | walking address bits and random patterns on each channel (raw channel addresses) |
| pattern | random data through the channel interleave, unaligned edges, the top of DRAM; sub-beat host writes (partial byte strobes) |
| bandwidth | host -> card and card -> host DMA rate |
| kernel | a program using every unit (DMA, VPU, quantizer, MXU, QST), then one of partial DRAM writes (QST bytes, short stores); DRAM equals the ISA simulator bit for bit |
| vops | RDOT / OUTER / LOG2 (Qwen3.5's DeltaNet functions) against the ISA simulator. A bitstream built before them runs the program without an error but computes other values: the stage passes with a note ("Qwen3 and LFM2 only"), and fails only with `--model qwen35` |
| model | (with `--model`) greedy decoding of "What is the capital of France?" (`--tokens` tokens) equals the ISA simulator token for token |

The model stage also runs the ISA simulator's reference on the host (a few seconds per token).
The prompt (21-24 tokens) and every generated token are one step each; on the board model
(`--sim`) each step is a Verilator run of minutes, so use `--tokens 1` there.

### When a stage fails

The self-test prints a hint under the failing stage; in more detail:

| Stage | First things to check |
|---|---|
| link | `lspci -d 10ee:` lists the card? If not: rescan after JTAG (`sudo otpu-setup --rescan`) or reboot. Listed but `/dev/xdma0_user` missing: `lsmod \| grep xdma`, `dmesg \| grep -i xdma`. ID `0xffffffff`: the link dropped (the FPGA was reprogrammed after enumeration: rescan). Another ID: a bitstream without openTPU, or the AXI-Lite path in the block design |
| config | The message names the bitstream's value and the environment's: `unset OTPU_MCOLS OTPU_LANES`, or load the bitstream built for them (board.md, "Which bitstream to load") |
| calib | A DDR3 channel did not calibrate: STATUS bit 5 = channel 0, bit 6 = channel 1 (`otpu-smi` shows both). One channel only: its byte lanes / pinout (board.md section 6.2). Both: the 200 MHz reference clock, or the memory supply |
| regs | SCRATCH does not hold writes: the AXI-Lite write path, or core_clk / reset not running (the heartbeat LED) |
| addr | The message names the channel and the address bit that aliases or is stuck: MIG address width / pinout of that channel, or the interconnect map (channel 1 at 0x8000_0000) |
| pattern | Errors on one channel only: its byte lanes. Errors every other 64-byte beat: the host interleave vs `rtl/mem/otpu_axi_dram.sv`. Only the partial writes fail: the MIG ECC read-modify-write (board.md section 6.1) |
| bandwidth | Below 0.5 GB/s: `LnkSta` width / speed, the IOMMU (`iommu=pt`), or the driver in a slow mode. A DMA call that hangs: interrupts (reload with `poll_mode=1`) |
| kernel | `retired n of m instructions`: the core stopped early (illegal instruction, AXI error). DRAM differs: run `pytest tests/test_board.py` (the same program on the RTL model) and compare the counters with `otpu-smi -q` |
| vops | With `--model qwen35` only: the bitstream predates RDOT / OUTER / LOG2; load one that has them |
| model | Kernels pass but tokens differ: the image does not fit or a DRAM region is bad (pattern stage covers only samples), or a timing-dependent bug; compare per-token logits against the ISA simulator (`opentpu.llm.qwen3.Engine` with `backend="isa"`) |

Partial writes matter on this board: each DDR3 channel is 9 x8 devices (72-bit, ECC) without
data-mask pins, so the memory controller turns every write with partial byte strobes into a
read-modify-write. The pattern and kernel stages exercise that path from the host (sub-beat DMA)
and from the accelerator (QST byte writes, word-masked stores). The board model applies byte
strobes directly, so only the card proves the controller's read-modify-write.

## 5. Full diagnostic: otpu-diag

```sh
otpu-diag                                     # every check, about a minute
otpu-diag --json diag.json                    # the report as JSON as well
otpu-diag --mem full                          # plus a march C- over all 4 GiB
otpu-diag --soak 20                           # rerun the kernel set 20 times: intermittents
otpu-diag --model qwen3 --tokens 8            # plus the model check of otpu-selftest
otpu-diag --only mem,isa                      # platform plus some sections
otpu-diag --sim                               # the board model (memory tests scaled to it)
```

Where otpu-selftest stops at the first failure, otpu-diag runs everything it can. A check whose
prerequisite failed is marked SKIP with the reason (the memory tests of a channel need its
calibration; the programs need the ID, the configuration and both calibrations); everything
else runs. It prints a line per check, then a matrix (PASS / FAIL / SKIP / INFO per section),
the failing checks with their details and the diagnosis. Exit code 1 on any FAIL.

| Section | Checks |
|---|---|
| platform | PCIe link speed and width (sysfs; expected 2.5 GT/s x8), XDMA module and device nodes, ID, VERSION -> configuration, BUILD_ID and CORE_KHZ, calibration of each channel, STATUS ERROR / AXI_ERR (cleared with CLEAR if left by an earlier run), die temperature, the power estimate from `power.json` (an estimate, INFO) |
| regs | SCRATCH, PROG_ADDR, PROG_N, TRACE_ADDR: 68 write / read patterns each (walking 1, walking 0, all 0 / 1, checkerboards; stuck bits named); TRACE_CTRL bits; read-only registers: sane values (VERSION, REGMAP, CAPS, CORE_KHZ, 0xDEADBEEF on an undefined offset) and ignoring writes; SNAP and the free-running counters |
| mem | per channel (raw channel addresses): walking 1 and walking 0 over the 512 bits of a beat, walking address bits (aliasing named), 16 random blocks spread over the channel, 200 sub-beat updates (merged into whole beats on the host, see section 2), DMA bandwidth each way; the interleave through the accelerator's address map; with `--mem full` a march C- over every byte with address-in-address data (progress line; errors per byte lane, DQ bit and address bit) |
| isa | one program per instruction variant (`opentpu/host/opchecks.py`, 93 at MCOLS=2), each compared with the ISA simulator bit for bit: NOP, HALT, LI / ADDI, LOOP (nested, count from a register, count 0), BAR; LD / ST aligned, unaligned, short, register offsets; MM plain, UNIT, ACC, RMAX, ACC+RMAX, UNIT+ACC+ASCALE, M=1, another ACT block, a row stride, register operands; QACT ROW / CSCALE / RSCALE; QST dense, strided, ROW; GATHER; every VOP function under each legal broadcast mode (FULL / ROW / COL / SCALAR for the binary ones and RDOT), OUTER with each decay mode; the composite and simple functions on edge values (zeros, denormals, the largest floats, infinities) |
| system | the all-units demo, the masked-write and the RDOT / OUTER / LOG2 programs; the cycle counters (a NOP loop of n and 2n iterations: CYCLES grows, on the card agrees with the wall time at CORE_KHZ and with UPTIME); `--soak N` |
| model | `--model`: greedy decoding against the ISA simulator, as otpu-selftest |

Memory errors are counted per byte lane: on this board byte b of a channel offset travels on
DQ byte lane b % 8 (a 64-byte AXI beat is one BL8 burst of the 64-bit channel), so a failing
lane names DQ[8L+7:8L]. The diagnosis reads the pattern of failures, for example:

- `channel 1 byte lane 5 errors -> DQ[47:40] pinout / calibration of that lane` (and the one
  DQ bit when only one is wrong); errors on every lane of a channel point at the whole channel;
- `channel 0 address bit 27 aliases with bit 26 -> that address line or the MIG address width`;
- `all MXU rows fail but the VPU and DMA pass -> MXU / DSP path`; `QACT / QST fail while MM
  passes -> the quantizer`; composite functions against simple ones -> the composite lanes;
- `only RDOT / OUTER / LOG2 fail: a bitstream built before ddec900`;
- every program failing -> program load, sequencer, clock / reset or DRAM.

The JSON report (`--json`) holds every row (section, name, status, message, seconds, the
per-lane counts) and the hints. On the board model (`--sim`) the PCIe, driver and bandwidth
checks are SKIP; everything else passes (about 80 s). Known difference between the RTL and the
ISA simulator, left out of the edge values: NaN inputs (RECIP of a NaN is 0 in the RTL; MAX,
ABS and COPY pass a signalling NaN through where the simulator returns the canonical NaN).

## 6. Chat

```sh
otpu-chat --backend board                      # the full-screen interface
otpu-chat --backend board --plain              # a line-by-line REPL instead
otpu-chat --backend board --prompt "Why is the sky blue?"   # one-shot, plain output
otpu-chat --backend board --clock-mhz 100      # override the core clock (v1 bitstreams)
otpu-chat --backend board --model lfm2         # LFM2.5-230M instead of Qwen3-0.6B
otpu-chat --backend board --model qwen35       # Qwen3.5-0.8B (needs RDOT / OUTER / LOG2 in the bitstream)
```

The first call writes the model image (at the default `--cap 2048`: 0.69 GiB for Qwen3-0.6B,
0.27 GiB for LFM2.5-230M, 0.77 GiB for Qwen3.5-0.8B) to the card; every token then writes the
embedding row and the token's program (a few tens of KiB), runs, and reads the logits (0.58 MiB
for Qwen3, 0.25 MiB for LFM2, 0.95 MiB for Qwen3.5).

**The interface** (Textual, `opentpu/host/chat_tui.py`) keeps the terminal's own background
and one accent colour. The conversation is a single column: a header box with the model,
backend, device, bitstream and clock, then each prompt after a dim `>` and each reply after a
`⏺`, streamed token by token and rendered as Markdown. While a reply runs, a spinner line above
the input shows the phase (`Prefilling… 12/21 tok · 1.4s`, then `Decoding… 87 tok · 7.6
tok/s`). Under the input box one status line is always visible (LFM2.5-230M on the card,
second turn of a chat, build 74d48591, measured 2026-09-26):

```
LFM2.5-230M · board 100 MHz │ TTFT 2.66s │ prefill 10.2 tok/s (dev 13.0) │ decode 7.9 tok/s (dev 12.9) · 7.73 Mcyc/tok │ ctx 105/2048 ▱▱▱▱▱▱▱▱ 5%
```

TTFT is submit to the first generated token; prefill and decode are tokens/s on the wall clock
and, on the card, on the device (the CYCLES of the steps at the bitstream's CORE_KHZ, or
`--clock-mhz`); the context meter turns amber over 75 % and red over 90 %. In a narrow terminal
the line drops the model, then the prefill device rate and Mcycles/token, then the prefill; the
context stays. On the ISA backend the numbers are wall time only.

Enter sends, Esc stops the reply (what was generated stays in the history and the KV cache),
Ctrl-S shows or hides a side panel with the detail, Ctrl-C or Ctrl-D quits. Typing `/` opens
the commands (up / down, Tab completes, Enter runs): `/help`, `/continue` (a reply cut at
`--max-new`, default 1024, goes on where it stopped), `/reset` (forget the conversation and the
KV cache), `/stats` (the detail inline: the last turn, DRAM for the image and the KV cache, the
session's turns, tokens in and out and average decode tok/s, the sampling), `/think on|off`
(thinking mode; the history is re-fed on the next turn), `/quit`.

The KV cache holds `--cap` tokens (default 2048) and the engine has no sliding window. A reply
that fills the cache stops with "context full"; a message that no longer fits is refused
without touching the cache; `/reset` starts over. The plain REPL takes `/continue` and `/reset`
too.

Prefill here is the tokens a turn adds: the KV cache keeps every earlier turn, so a turn feeds
only what the chat template appended since (all of it again when the template rewrote the
history, e.g. after `/think`). Decode tok/s counts the tokens after the first, over the time
since the first. `--plain` and `--prompt` print the same numbers after each reply (LFM2.5-230M
on the card, build 74d48591, measured 2026-09-26):

```
[TTFT 2.14s; prefill 21 tokens, 9.90 (device 13.0) tok/s; decode 25 tokens, 9.57 (device 13.0) tok/s, 7.72 Mcycles/token at 100 MHz; context 46/2048]
```

Decode tokens/s of a 160-token reply (the same prompt and seed), measured on the card (build
a691ea98, 100 MHz), before (main at e639ecd) and after the host changes in section 7:

| | device | `--plain` before | after | interface before | after |
|---|---|---|---|---|---|
| LFM2.5-230M | 31.7 | 17.05 | 30.35 | 15.20 | 30.19 |
| Qwen3-0.6B | 11.4 | 10.03 | 11.11 | 9.53 | 11.10 |

While a chat runs, `otpu-smi` shows the process, the model, the DRAM in use and tokens/s.

`--backend board-sim` runs the same driver against the Verilator board model (bit-exact, but
minutes per token for the real model; use it with small models).

## 7. Control registers

[observability.md](observability.md) has the register map (version 2: the version 1 registers
plus REGMAP, CAPS, CORE_KHZ, BUILD_ID, TEMP, SNAP, the free-running counters, the trace
buffer and the I2C pins); `opentpu/host/regs.py` has the same as constants. The driver's sequence per token
(`opentpu/host/board.py`, `Board.load_program` and `Board.run`):

1. write the program into DRAM (logical address right after the model image), `CTRL = 0`;
2. `PROG_ADDR`, `PROG_N` (instructions), `CTRL = LOAD`; poll `STATUS.LOADING == 0`;
3. (profiling) `TRACE_CTRL = CLEAR`, then `ENABLE` (+ `STOP_WHEN_FULL`);
4. `CTRL = CLEAR` (zero the per-run counters; also holds the core in reset), `CTRL = RUN`;
5. poll `STATUS.HALTED`; read `STATUS`, `CYCLES`, `ICOUNT`, the DRAM port counters (and
   `TRACE_COUNT`, `TRACE_DROP`, the records);
6. `CTRL = 0`. `STATUS.ERROR` (illegal instruction) and `STATUS.AXI_ERR` (a DRAM access got an
   error response) make the driver raise.

**Version 1, 2 and 3 bitstreams.** `Board.info()` reads REGMAP. Version 3 adds the MXU_STARVE
counter (`otpu-smi`: "MXU-starve" in the stalls line); on a version 2 bitstream the snapshots,
`rates()` and `otpu-smi` leave it out. A version 1 bitstream has no such register (it reads `0xDEADBEEF`, or 0): the driver then never touches the version 2
offsets (version 1 decodes 8 address bits, so 0x100 and up alias onto the low registers).
Everything but the counters, the trace and the temperature works: `snapshot()` returns None,
`otpu-smi` shows utilization / power / temperature as n/a, `otpu-lens record` falls back to
counters-only profiles, tokens/s uses `--clock-mhz`.

**Polling.** `XdmaTransport.poll` reads the register back to back for the first 100 us (a
PCIe read is ~1 us: program loads finish in this phase, with no sleep latency), then sleeps
elapsed/32, at most 1 ms, between reads. A wait of T is noticed at most ~T/32 late, at most
1 ms, with ~32 ln(T / 100 us) + T / 1 ms reads instead of T / 1 us. A token's run is polled
with a hint instead: `BoardBackend` passes the previous run's device time (its CYCLES at
CORE_KHZ; the next token's is a few cycles longer), and the poll sleeps once until 0.5 ms + 1%
before it, then reads back to back. Measured on the card (build a691ea98), the run is seen
0.02-0.06 ms after the device time instead of 0.5-1.4 ms.

**DMA.** Reads go straight into one preallocated numpy buffer per channel (`os.preadv` into
memoryview slices; no concatenation), writes from memoryview slices; both in 8 MiB calls (the
XDMA driver pins each call's pages and builds one descriptor list: 8 MiB bounds that, and the
per-call cost stays under 1% of the transfer).

**Pipelining.** A token's program depends on its position only, so `Engine.step` compiles
position p + 1 while the card runs p (`Engine(..., pipeline=None)`: on for every backend but
the ISA simulator); on the board, three worker processes keep p + 1 .. p + 3 in flight
(`COMPILE_AHEAD`), so a compile may take up to three device runs (two until LFM2 fp4's 13.5 ms
run: 0.8-1.2 ms of compile wait per token on a card host running two Vivado builds). A precompile made for another
position (after `reset`, or `run_rows`) is waited for and dropped. Results are unchanged (`tests/test_host.py`, and the board-model tests
compare the logits with the ISA simulator's bit for bit).

On the board the compile runs in worker processes (spawned when the engine starts; until they
are up, steps compile in line) and sends back the assembled words, and `Engine.step` hands it
position p + 1 only after `BoardBackend.start` has copied program p to the card and started
it. Both matter because a compile is 10-30 ms of Python on the PC above (Qwen3-0.6B ~11 ms,
LFM2.5-230M ~20-30 ms), and in a thread of the same process it holds the GIL: started before
the program copy, it made the copy wait for the GIL (the copy's DMA call releases it and then
waits up to the 5 ms switch interval to get it back), measured 8.6 ms per token for Qwen3 and
20 ms for LFM2 on the card. Other backends (the RTL simulator, the ISA simulator with
`pipeline=True`) and `otpu-lens record` (which needs the programs) compile on a thread
(`pipeline="thread"`). A script that makes an engine on the board needs the usual
`if __name__ == "__main__":` guard (the worker is spawned and imports the main module).

**Host time per token.** `tools/decode_profile.py --model lfm2|qwen3` runs one chat turn on the
card with a timer around each piece of a token's host work. Measured on the card (build
a691ea98, 100 MHz, a 96-token reply; ms per decode token):

| | LFM2 before | LFM2 after | Qwen3 before | Qwen3 after |
|---|---|---|---|---|
| x / cos / sin write | 0.43 | 0.16 | 0.17 | 0.18 |
| wait for the compile | 0.01 | 0.08 | 0.01 | 0.01 |
| program upload (26 / 14 KiB) | 20.44 | 0.23 | 8.59 | 0.08 |
| run: device time | 31.47 | 31.47 | 87.92 | 87.92 |
| run: poll overshoot | 1.42 | 0.01 | 0.57 | 0.02 |
| status file | 0.34 | 0.04 | 0.17 | 0.18 |
| logits read (0.25 / 0.58 MiB) | 0.37 | 0.41 | 0.54 | 0.65 |
| sampling | 0.73 | 0.65 | 0.78 | 0.83 |
| detokenize | 0.07 | 0.04 | 0.06 | 0.04 |
| other | 0.74 | 0.16 | 0.42 | 0.20 |
| **tok/s wall (device)** | **17.84 (31.78)** | **30.07 (31.78)** | **10.08 (11.37)** | **11.09 (11.37)** |

"Before" is main at e639ecd. The x / cos / sin rows go in one DMA write now (they are
adjacent in the I/O area), the status file is rewritten at most every 0.25 s (a timer writes
the last tokens), and the reply is detokenized incrementally (`chat.Detok`: only the tokens
since the last emitted text, from one token earlier; an incomplete UTF-8 character is held back)
instead of decoding the whole reply every token. What is left, ~1.8 ms (LFM2) and ~2.3 ms
(Qwen3) per token, is mostly the logits read and the sampling over the vocabulary; reading
only an on-device argmax would save ~0.4-0.6 ms of it for greedy decoding only.

Sampling selects the top k on the float32 logits: the k-th largest of the 64-logit block
maxima bounds the k-th largest logit from below, so the selection runs on the few logits above
it, and only those are converted to float64. When the k largest are distinct and above the
next one the result is unique, so the picks are those of the float64 argpartition it replaces;
any tie falls back to that path (`tests/test_host.py` compares the picks). Measured on the
card (build a691ea98, burst image), ms per token for the sampling, before / after: LFM2
0.66 / 0.47, Qwen3 0.67 / 0.44, Qwen3.5 1.13 / 0.51 (wall tok/s 30.20 / 30.54, 11.14 / 11.14,
8.18 / 8.22). In isolation the new selection takes 0.10 / 0.16 / 0.27 ms on the PC above; in the
chat loop the logits are freshly read from the card, and the rest of the sampler (the
repetition penalty, the choice) adds to it.

**Streamed logits.** What the host does between seeing HALTED and writing RUN for the next token
is the token's critical path: the card idles meanwhile. Reading the whole logits vector and
sampling from it after the run were most of it (0.41 + 0.47-0.65 ms for LFM2 above). The LM
head writes the logits in chunks of 8,192 (32 KiB), one after the other, in the last part of
the run (25-40% of an LFM2 token, simulated), so the host now takes each chunk while the run
goes on (`BoardBackend.start(stream=...)` / `wait(feed)`, `Engine.step(sink=...)`):

- Before the run the logits region holds a word the device never stores, 0xFFFFFFFF (its NaN
  is canonical: 0x7FC00000, or 0xFFC00000 after a sign flip). A chunk is complete when none of
  its words is left: the host probes the chunk's last beat (one 64-byte read), then reads the
  chunk and checks every word, and writes the marker back into it (the run is past it).
- The probes start 0.3 ms before the time the same chunk was seen complete in the last token
  (every 0.5 ms on the first), and retry every 0.1 ms; a chunk already complete at its first
  probe moves its time 0.3 ms earlier, so the schedule follows the run both ways. The waits are
  sleeps of at most 1 ms (a long sleep overshoots: 38% on macOS). HALTED ends the loop; the last
  chunk (written just before the halt) and any other still pending are read in one read after
  it, and marked again right after the next token's RUN (its LM head is milliseconds away).
- The sampler takes each chunk as it comes (`pick.stream(context)`, chat.sampler): the
  repetition penalty and the 64-logit block maxima of the top-k prefilter are applied per chunk,
  so after the halt only the selection is left (21-31 us against 44-108 us for the whole
  vector on the Mac, the models' defaults). The picks are those of `pick(logits, context)`
  exactly, with the same random stream (tests/test_host.py).
- A run that leaves a word of the region unwritten is an error, not stale logits; a run that
  is not a streamed decode step (a prefill, a batch) makes the next step mark the whole region
  again (256 KiB for LFM2, before its RUN). Transports without DMA during a run (the board
  model) read the logits after the run as before; `Engine.stream_logits = False` turns it off.

This DMA during a run is new on the card: the whole path is tested against a fake card that
runs the ISA simulator and reveals the logits late, chunk by chunk (bit-exact logits and the
same picks over prefill and decode), but not yet on the card.

The token's text and the interface's drawing now go after the next RUN (`on_start`), and the
status file is rewritten only by its timer thread, never from the token's critical path.

**Measuring it.** `tools/decode_profile.py` measures the critical path directly: from HALTED
seen to the next RUN written, per token, split into its steps, every transport operation
(DMA write / read, register read / write, the HALTED poll) filed under the step it serves and
as critical or overlapped (`--no-stream` for the old read; `--json` adds the reply's token ids).
The tool's own timers add a few microseconds per operation. `tools/host_path_card.py` is the
card session: for each model, greedy with and without streaming (the replies must be the same
tokens) and sampled with and without, then one table per model:

```sh
OTPU_LOCK_WAIT=600 python3 tools/host_path_card.py --models lfm2,qwen3 --tokens 96
OTPU_LOCK_WAIT=600 python3 tools/host_path_card.py --models lfm2 -- --wformat fp4 --head-format int8
```

What is left on the critical path after streaming, by the measurements above (*estimate* until
the card run): the x / cos / sin write (0.16 ms), the program upload and IMEM load (0.23 ms, gone
with one position-independent program per model), the wait for the compile (0.08 ms; at a 12 ms
token the per-position compile of LFM2 no longer keeps up on a laptop: the same fix), the
counters (~0.02 ms), the last chunk's read and the selection (~0.1 ms), Python (~0.1 ms).

**Poll outliers** (card, host-path bc132ba: ~1 ms per Qwen3 token as poll overshoot in half
the runs, or as critical time in the counters in one streamed run per pair). The poll's
expected time was the last run's CYCLES / CORE_KHZ; a run a little longer than that (past the
0.1 ms spin) was seen only after the next 1 ms sleep, and one that ended before the wake-up was
seen at the wake-up; after the streamed wait had seen HALTED itself, the poll still slept until
the expected end. Now: the sleeps past the expected end are 1/32 of the lateness (at least
20 us, at most 1 ms); the expected time is CYCLES / CORE_KHZ times the smoothed ratio of the
seen run times to it (raised only by a halt seen by back-to-back reads that saw the run going
just before, lowered by any); a halt the streamed wait saw skips the sleep; the wake-up sleep
halves its way to the target (a 50 ms sleep overshoots 2-3 ms on macOS). Measured on
`FakeTransport` (`decode_profile.py --backend fake --fake-skew F [--fake-stream]`: CYCLES a
fraction F off the wall time; the fake writes its logits piece by piece), Qwen3's 53 ms run
on the loaded Mac, HALTED seen after the run's end: 0.00-0.02 ms without streaming (before:
3.2-3.7 ms, most of it the macOS sleep overshoot) and 0.07-0.18 ms streamed, at F = +0.25% and
-2%. Not yet on the card.

**Streamed wait vs a late piece time.** The streamed wait sleeps until each piece's time of
the last token; a time learned on a delayed token can lie past this run's end, and HALTED was
checked only at the piece's time: LFM2 fp4 on a busy card host (fp4fx120, two Vivado builds)
saw HALTED 2.2 ms after the run's end in one streamed run. HALTED is now read after every
sleep slice (at most 1 ms apart). Not yet re-measured on the card.

**HALTED before the last stores land.** HALTED rises once the last store has left the core;
the memory adapter may still have writes in flight (STATUS WR_IDLE clear), and a host read
through the XDMA may pass them. On the card (hp-wb aebb0bf0, resident decode, LFM2 fp4 + int8
head, sampled, streamed) a tail read found words of the last logits piece unwritten.
`Board.wait` now waits for WR_IDLE after HALTED (at most 2 ms; not on the board model, whose
register script replays in order), and the streamed tail reads again for up to 5 ms before it
calls a piece unwritten (the error names the words). `FakeTransport.logits_lag` lands the last
piece after HALTED, with WR_IDLE clear until then (tests/test_host.py). Not yet re-run on the
card.

**Resident decode.** A decode program now takes the position and the token as run arguments
(docs/isa.md "Arguments": ARG0..7, R8..R15 at the start; CAPS bit25), so one program serves
every position of an attention bucket (`Engine(resident=True)`, the default of `otpu-chat` and
`decode_profile.py`; `--per-position` for the old path):

- Attention spans `blocks` 256-token blocks, the last one masked: the program adds a row of a
  mask table (`+inf` for the first `cap` entries, then 256 of `-inf`) to its scores at an
  offset of -4 x position, `min(score, mask)`; a masked score is `-inf`, its exp2 is +0, so
  the row sums, P.V and the row maximum (VOP RMAX, the same bits as the MM epilogue) are the
  unmasked columns' (`kernels/attention.Bucket`). One program per 256 positions: compiled once
  (on the worker process, 16 positions before the bucket is reached) and kept in IMEM.
- The program sees the position as t0 + tpos, t0 the bucket's first position and tpos < 256
  the argument (so a V^T append stays in its 256-token tile). The KV append, the K / V scales
  and V^T columns, and the RoPE rows are at tpos x 128, x 4 and x 1; LFM2's convolution ring is mirrored (each row stored twice, in 2K rows) so the last K
  rows are contiguous at ((p + 1) mod K) x row: one argument instead of three
  (`lfm2._ring_rows`).
- The token's embedding row and the RoPE cos / sin rows come from tables in the image
  (`Image(lookup=True)`: the fp32 embedding, cos / sin of every position): the step writes no
  inputs. For LFM2.5-230M the tables are 256 MiB of the card's 4 GiB.
- Positions 0 and 1 of LFM2 (the convolution's taps before position 0) and Qwen3.5 (its
  convolution and DeltaNet programs use R1..R8) keep the per-position programs, and so does a
  bitstream without CAPS bit25 (`Engine.resident` falls back; the worker processes of
  `COMPILE_AHEAD` stay for it).

The logits are bit-identical to the per-position programs' (tests: `test_qwen3.py`,
`test_lfm2.py` from position 0 across the bucket boundaries 256 and 512 on the ISA
simulator, positions 255 and 256 on the Verilator RTL, LFM2.5-230M fp4 with an int8 head on
the RTL; the board model with the ARG registers, where each run is a fresh simulation, so it
loads the program every run; `FakeTransport` counts one load). A step's host work before RUN is then the
changed ARG words (up to 6 register writes, posted) instead of the x / cos / sin write
(0.16 ms), the program upload and IMEM load (0.23 ms) and the compile wait (0.08 ms, and at
long contexts the 16-33 ms trace that no longer keeps up): by the measurements above the
critical path would be ~0.25 ms per token (counters, the last chunk and the selection,
Python). This is an *estimate*: it needs a bitstream with the ARG registers, not built yet. On
`FakeTransport` (its DMA is a memory copy, so it shows the Python side only; LFM2.5-230M fp4 /
int8 head, 48 tokens, an 11 ms run, the Mac above) `decode_profile.py` measures 0.215 ms of
critical path per token against 0.245 with `--per-position`.

The price is on the device: the bucket's last block is computed in full, masked, so early in
a bucket attention does up to 255 columns more than a per-position program. LFM2.5-230M, fp4
with an int8 head, DDR3-1066 bank model at 116 MHz, position 9 (the worst case), simulated:
1,289,001 cycles against 1,260,820 (+2.2%, 90.0 against 92.0 tok/s device); at a bucket's end
the two meet. Against the ~0.5 ms of host time per token it removes (4% of an 11 ms token)
it is a gain at every position.

The chat interface draws each token while the card runs the next one: `Chat` hands a token to
`on_update` from `Engine.step`'s `on_start` hook (called once the run is started), so the
interface's work does not delay the host work that starts a run. Before, the full-screen
interface took ~5 ms per LFM2 token from the generation thread.

**Long contexts.** Two host costs grew with the context, and at ~1900 tokens LFM2 decoded at
14.8 tok/s on the wall against 37.2 on the device (reported on the card). Both reproduce
without the card, on `FakeTransport` with a 27 ms run (37.0 tok/s device) on the PC above
while two Vivado builds ran on it (load ~10 on 16 threads):

- The decode program grows with the position: attention's hardware loop covers whole groups of
  three 256-token blocks and the rest is unrolled, so LFM2's program goes from 837 to 1518
  instructions (26 to 47 KiB) at position 1900 and its trace from 16 to 30-33 ms, longer than
  the 27 ms run. With one worker process the step waited 11-21 ms for it (`decode_profile.py
  --backend fake`, plain mode: 20.4-25.9 tok/s); with two in flight, 3 ms (32.7-32.9 tok/s).
  Qwen3 (26 ms of trace against a 68 ms run) and Qwen3.5 (56 against 97 ms) had room.
- The interface's work per token grew with the reply. Textual's Markdown lays out and
  restyles its blocks on every append, and a style rule on `:first-child` / `:last-child`
  made each mounted block restyle all its siblings, so one long reply cost 20 ms of CPU per
  token at its start and 78 ms after 1600 tokens; sharing the GIL, the decode loop fell to
  11 tok/s. Now a reply is Markdown in parts of 16 blocks (a new part starts at a blank line
  outside a code fence, where the text goes on unindented, so the reply reads the same), the
  first and last margins are inline styles (Textual's style cache does not track those
  pseudo-classes, which also left stale margins), the one-line widgets update without a
  layout of the screen, and the interface thread's CPU is held to 10% of the time between
  updates (`ChatApp.UI_SHARE`): past it, an update waits and takes several tokens at once.

A 1000-token prompt and a 900-token reply in the interface (context 1898), fake card as above:
15.4 tok/s wall before, 31.8 after (CPU per token 48 ms, now 7 ms); at short context plain
mode goes 34.4 -> 35.7 tok/s. Not yet re-measured on the card.

## 8. Device lock, status file and otpu-smi

**Waiting for the card.** A runner that finds the card in use fails with `DeviceBusy` at once,
unless `OTPU_LOCK_WAIT=<seconds>` is set: then it waits for the lock (every openTPU tool: chat,
selftest, diag, scripts using `Board`). Steps that are not openTPU tools but must not overlap a
run (a JTAG reload, a driver reload, a rescan) go through `otpu-lock -- <command>`, which holds
the same lock around the command:

```sh
OTPU_LOCK_WAIT=3600 otpu-selftest --model lfm2
otpu-lock -- sh -c 'openFPGALoader -c digilent_hs2 build/deploy_burst_a691ea98/otpu.bit && sudo otpu-setup --rescan'
```

Do not wait for the card with `pgrep -f` loops: the pattern appears in the waiting shell's own
command line, and in other waiters', so they match each other and wait forever (seen at bring-up).


**Lock.** Anything that runs programs or writes the card's DRAM (`Board`, `BoardBackend`,
`otpu-selftest`, `otpu-chat`, `otpu-lens record`) takes an exclusive `flock` on
`/tmp/otpu/<dev>.lock` (`<dev>` = `xdma0` for `/dev/xdma0`; `OTPU_RUN_DIR` moves the directory)
and writes its pid there. A second runner fails at once with `xdma0 is in use by process <pid>
(<command>)`. The lock belongs to the open device (Boards on the same transport share it) and
goes away with the process, however it ends. Monitors (`otpu-smi`) never lock.

**Status file.** The runner publishes `/tmp/otpu/<dev>.json`, replaced atomically after every
token (`BoardBackend`: at most every 0.25 s, with a timer writing the last tokens) and removed
at exit: `pid`, `argv`, `start`, `dev`, `model`, `core_khz`, `dram` (bytes:
`total`, `image`, `weights`, `kv_capacity`, `kv_used`, `program`, `free`), `tokens`,
`last_cycles`, `tok_s_device` (CORE_KHZ / last cycles), `tok_s_wall` (over the last 8 tokens,
host work included), `updated`. A file whose pid is gone (a runner killed with SIGKILL) is
shown as stale.

**otpu-smi.**

```
$ otpu-smi
otpu-smi 0.1.0                                                       2026-09-24 08:08:46
+--------------------------------------------------------------------------------------+
| /dev/xdma0       openTPU D=128 MCOLS=2 LANES=8  100.0 MHz  build 1234abcd  regmap v2 |
| link ok (2.5 GT/s PCIe x8)   DDR3 calib ch0 ok ch1 ok   temp 34.5 C   running        |
| power 4.63 W est.   DRAM 900 / 4,096 MiB (KV 12 / 224)   DRAM bw 4.61 GB/s           |
+--------------------------------------------------------------------------------------+
| util  RUN 80%  MXU 60%  MAC 50%  VPU 12%  QNT 5%  DMA 3%                             |
| stall TMEM-deny 1%  DRAM-wait 20%   IPC 0.005   over 200 ms                          |
| pid 53814  otpu-chat --backend board                                                 |
|       model Qwen3-0.6B   tokens 57   12.10 tok/s wall   device 17.86 tok/s           |
+--------------------------------------------------------------------------------------+
```

| Option | |
|---|---|
| `-l SEC` | repeat every SEC seconds; utilization over each period |
| `--json` | one object per device (every field below, raw counters included) |
| `-q` | every detail as `key value` lines |
| `--dev /dev/xdma0` | one device (repeatable); default every `/dev/xdma*_user` |
| `-i SEC` | the sampling interval of a single report (default 0.2 s) |
| `--power-json FILE` | default `build/vivado/reports/power.json` |
| `--sim` | the board model (below); `--fake`: an in-memory card with synthetic counters |

Fields: the bitstream (VERSION, CORE_KHZ, BUILD_ID, REGMAP), the link (ID register; the PCIe
speed and width from sysfs), DDR3 speed and calibration (DDR_MTS when CAPS bit3 is set, else plain
"DDR3"; STATUS bits 5, 6), temperature (TEMP, measured
by the XADC), the DRAM used / total and the KV cache (from the status file), the DRAM bandwidth
((DRAM_RD + DRAM_WR) deltas x 64 B over the UPTIME delta / CORE_KHZ), utilization (the deltas of
RUNNING, MXU_BUSY, MXU_MAC -- MAC utilization --, VPU_BUSY, QNT_BUSY, DMA_BUSY, TMEM_DENY,
DRAM_WAIT and, with register map 3, MXU_STARVE over the UPTIME delta, between two SNAPs) and
the owning process.

**Power** is an estimate (the card cannot measure it): fixed + sum over units of the unit's
dynamic power x its utilization, from Vivado's `report_power` of the build. `build.tcl` writes
`reports/power.rpt`, `reports/power.xml` and, through `python3 -m opentpu.host.power`,
`reports/power.json` (format in `opentpu/host/power.py`: the summary, the on-chip components,
the power by hierarchy, and per unit -- found by instance name: `u_mxu` + `u_act` MXU, `u_vpu`,
`u_quant`, `u_dma`, `u_tmem`, `u_seq`, `u_mem` DRAM -- the dynamic watts and the counter that
scales them; everything else, static power included, is fixed). Without the file, power is n/a.
Copy the file next to the card's host at the same path, or pass `--power-json`.

**`--sim`** runs against the Verilator board model. SimTransport starts a fresh simulation per
flush, so both SNAPs happen inside one flush: the model loads the bring-up demo program, SNAPs,
runs it, SNAPs again when it halts (`--sim-idle N`: an idle window of N cycles instead). With
the register map 1 model the table shows the run's cycles and instructions and n/a for the
counters.

## Rehearsal without hardware

`opentpu/host/board.py`'s `SimTransport` replays the driver's register and DMA operations on
the Verilator model of the board (`sim/verilator/tb_board.sv`: control registers, program
loader, slice, DRAM adapter and a two-channel AXI memory with random stalls).
`tests/test_board.py` runs a program and a tiny Qwen3 through it, bit-exact against the ISA
simulator. Each flush of the transport is a fresh simulation: DRAM persists between flushes
(through the channel image files), registers, IMEM and TMEM do not -- the driver loads, runs
and reads the counters of a program within one flush.

Batched use, for work that must happen within one simulation: `queue_read(addr)` queues a
read and returns its index in the next flush's `results` (in order; an address may repeat),
`wait_cycles(n)` queues the testbench's existing `C n` command (wait n cycles; no RTL change).
`otpu-smi --sim` samples the counters twice this way, and `Board.run(trace=...)` reads the
whole trace buffer (CAPS depth) in the flush that ran the program. When
`rtl/boards/ypcb-00338/otpu_trace.sv` exists it is added to the model's sources; the
testbench's AXI-Lite address is 12 bits wide (register map 2).

The model is built with the configuration the environment selects -- `OTPU_MCOLS`,
`OTPU_LANES`, `OTPU_VPU_CL` (default 2, as `make bit`) -- and the MXU and quantizer on 8 TMEM
lanes as in the bitstream; the self-test and the tools then read it back from its VERSION
register, exactly as on the card.

`opentpu/host/fake.py`'s `FakeTransport` is an in-memory card with register map 2 (or 1):
synthetic counters, a trace buffer served from a list, runs that take a set wall time. The
host tests (`tests/test_host.py`) and `otpu-smi --fake` use it.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| no `10ee:` device in `lspci` | card not configured at enumeration: flash the bitstream, or program over JTAG and `sudo otpu-setup --rescan`; check the PERST# / refclk constraints of the bitstream |
| ID register reads `0xffffffff` right after a JTAG load | the PC still has the old configuration: `sudo otpu-setup --rescan` |
| after a kernel upgrade no `/dev/xdma*` | no DKMS: re-run `sudo otpu-setup` |
| `/dev/xdma0_*` missing | driver not loaded or not bound: `otpu-setup --check` says which |
| ID register reads `0xffffffff` | BAR not mapped / link down (`lspci -vv` shows `!` flags or `Region 0: ... [disabled]`) |
| ID reads something else | wrong bitstream, or the AXI-Lite interconnect in the block design does not reach `otpu_ctrl` |
| calib fails | MIG pinout, memory clock or DDR3 voltage; see board.md |
| DMA very slow | link trained at Gen1 or fewer lanes (`LnkSta`), or IOMMU without `iommu=pt` |
| `run` times out | the core clock or reset is not running, or a DRAM write never gets its response (`STATUS.WR_IDLE` stays 0) |
| `xdma0 is in use by process N` | another runner (chat, selftest, lens record) holds the card: stop it; `otpu-smi` shows it |
| otpu-smi shows n/a for utilization, temperature and power | a register map 1 bitstream (REGMAP reads 0xDEADBEEF), or no `power.json` for the power |
| kernel stage differs | reproduce with `pytest tests/test_board.py` (the same program on the RTL model), compare the counters |
