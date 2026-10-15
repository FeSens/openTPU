# LitePCIe instead of XDMA

The YPCB-00338's PCIe endpoint is [LitePCIe](https://github.com/enjoy-digital/litepcie)
(enjoy-digital), in place of Xilinx's XDMA IP, as its memory controllers are
[LiteDRAM](https://github.com/enjoy-digital/litedram) ([litedram.md](litedram.md)). Only the hard
block stays Xilinx's: the PCIE_2_1 integrated block and its GTX lanes, through the 7 Series
Integrated Block wrapper (the `pcie_7x` IP, no license). Everything above it is open source:
LitePCIe's PHY glue and pipe clock, its TLP layer, BAR0 master, scatter-gather DMA and MSI;
openTPU's DMA front end (`otpu_dma`, in the generated core), the stream-to-memory adapter
(`rtl/shell/otpu_pcie_mm.sv`) and the Linux driver (`opentpu/host/pcie/otpu`).

Status: branch `litepcie`. Labels as in [litedram.md](litedram.md): **measured** is a run that
was done (tool and version given), **estimate** is arithmetic.

## 1. The pieces

```
 host PC ── PCIe Gen2 x8 ── PCIE_2_1 (pcie_7x) ── S7PCIEPHY ── LitePCIe endpoint ──┬── BAR0 master
                            hard block, GTX x8    pipe clock   TLP layer, 16 tags │    0x00000-0x7FFFF: AXI-Lite to the block
                                                  MMCM                            │    design (control registers, LiteDRAM
                                                                                  │    CSRs, XADC); 0x80000-: LiteX CSRs
                                                                                  ├── DMA reader (H2C) ─┐
                                                                                  ├── DMA writer (C2H) ─┤ otpu_dma: command
                                                                                  └── MSI               │ queues, counts, record
                                                                                                        │
                       otpu_native_sys x_* (AXI4, 128 bits) ── otpu_pcie_mm ────────────────────────────┘
                       (otpu_dma_split, otpu_mem_ch per channel)    streams + card addresses -> bursts
```

| | XDMA (r2.2) | LitePCIe |
|---|---|---|
| hard block | PCIE_2_1, Gen2 x8 | the same, through `pcie_7x` |
| user clock | axi_aclk 250 MHz, 128 bits | userclk2 250 MHz, 128 bits (the top's `xclk`) |
| BAR0 | 1 MiB, AXI-Lite master | 1 MiB: 512 KiB user window (AXI-Lite master), LiteX CSRs above |
| DMA | XDMA's engines, an AXI4 memory-mapped master | LitePCIe's scatter-gather streams; card addresses from `otpu_dma`'s command queues; `otpu_pcie_mm` makes the AXI4 bursts |
| completion | XDMA's (poll mode or interrupts) | a 16-byte record the card writes to host memory behind the data, then an MSI |
| driver | Xilinx `dma_ip_drivers` XDMA + patch | `otpu_pcie` (openTPU's, GPL-2.0, 660 lines) |
| PCI ID | 10ee:7028, subsystem 10ee:4f54 | 10ee:5054, subsystem 10ee:4c50, revision 02, class 12 00 00 |

The two bitstream kinds have different PCI IDs, so each driver binds only its own and both stay
installed (`otpu-setup`). The driver's nodes keep XDMA's names and calls with another prefix
(`/dev/otpu0_user`, `_h2c_0`, `_c2h_0`), so the host software runs on either:
`opentpu/host/board.py`'s `card_transport` opens whichever driver has the card.

## 2. The core: `tools/litepcie/gen_core.py`

LiteX generates one Verilog module, `otpu_litepcie`
(`boards/ypcb-00338/litepcie/otpu_litepcie.v`, with the `pcie_7x` IP's creation in
`pcie_s7.tcl`, LiteX's synchronizer constraints in `otpu_litepcie.xdc` and the CSR map in
`opentpu/host/litepcie/csr.csv` and the driver's `csr.h`). The pinned migen / LiteX / LitePCIe
commits are in `boards/ypcb-00338/litepcie/core.json`; `tools/litepcie/check_core.sh` regenerates
the core and checks the committed files against it (`--setup DIR` makes the venv, `--update`
rewrites them).

- **PHY.** `S7PCIEPHY`, Gen2 x8, 128 bits at 250 MHz, its streams in its own clock domain (the
  core's `sys` is the PCIe user clock; naming another domain puts asynchronous FIFOs on both
  datapaths). Its MMCM makes the pipe clocks from lane 0's TXOUTCLK; `pclk` is a BUFGCTRL
  between 125 and 250 MHz (Gen1 / Gen2), constrained as two physically exclusive generated
  clocks (`boards/ypcb-00338/constraints/otpu_litepcie_clk.tcl`).
- **Endpoint.** 64-bit addressing, 16 read requests in flight (16 x 512 bytes), the completion
  buffers in LUTRAM.
- **DMA.** `LitePCIeDMA`: the reader (host -> card) and the writer (card -> host), each with a
  256-deep descriptor table in program mode (each descriptor runs once, the driver refills),
  64-bit descriptors {address, length, irq_disable, last_disable}. No loopback.
- **otpu_dma.** The driver's side: two command queues (64 deep) of {card address, length,
  flags}, `h2c_count` (a pulse per finished H2C command from `otpu_pcie_mm`), `c2h_count` (the
  writer's interrupt on a command's last descriptor), and the completion record: on any change
  of the counts, one 16-byte memory write {h2c_count, c2h_count, mm_status, seq} to the host
  address the driver gives, through its own crossbar port, so it leaves after the transfer's own
  writes (PCIe keeps posted writes in order: a record that shows a C2H command done comes after
  all of its data). The record's MSI follows its write by 32 cycles: the write still crosses the
  TLP layer and the PHY when it is accepted, and the co-simulation saw an undelayed MSI reach
  the host first.
- **MSI.** One vector: bit 0 a record, bit 1 an `otpu_pcie_mm` error.
- **Timing.** Register slices (`stream.Buffer`: valid, payload and ready from flip-flops) on the
  four streams to `otpu_pcie_mm`, on the PHY's two TLP streams (`RegisteredPHY`: the endpoint sees
  the PHY through them) and on each channel of the user window's AXI-Lite master
  (`axil_slice`); a registered CSR bridge; the DMA reader's descriptor retire registered
  (`RetireRegTable`: it only steps `loop_status`, which the driver does not read); LUTRAM for
  every memory of 64 words or fewer; MAX_FANOUT 32 on the FIFOs' pointers (section 5).

BAR0's CSRs (`csr.h`): `OTPU_DMA_*` at 0x81000 (ID 0x4f544c50 "OTLP", CAPS {version, log2 table
depth, tags, log2 queue depth}, the queues, counts, record address and control, `MM_CTRL` (a
reset of `otpu_pcie_mm` and the queues), `MM_STATUS`), LitePCIe's `PCIE_DMA0_*` (tables, enables)
at 0x81800, `PCIE_ENDPOINT_PHY_*` / `PCIE_PHY_PHY_*` (link status, negotiated sizes) at 0x82000 /
0x83000, `PCIE_MSI_*` at 0x82800.

## 3. `otpu_pcie_mm`

`rtl/shell/otpu_pcie_mm.sv`, in the PCIe user clock, between the streams and the memory port
(`otpu_native_sys`'s `x_*`, where XDMA's AXI master was).

- **H2C.** A command {addr, len} writes the next len bytes of the stream at card address addr, in
  INCR bursts of 16-byte beats that never cross 512 bytes. A burst's AW goes only when all its
  beats are in the data FIFO (256 beats), so W never waits on the stream and a channel is never
  held mid-burst. `h2c_done` pulses when the command's last write response is in: the data is
  then where every later access finds it, the accelerator's included. FILL (flag 0) writes zeros
  without stream data.
- **C2H.** A command reads len bytes at addr onto the stream (last on its final beat); a burst's
  AR goes only with room for all its beats in the read FIFO, so R is always taken.
- **Timing.** Register slices on both streams and on the AXI master (`otpu_skid`,
  `otpu_axi_slice`): every path to LitePCIe and to the memory port runs flip-flop to flip-flop.
  A burst's length is registered a cycle after the command position moves, and the FIFOs' room
  and promised beats are counters (a pushed beat counts a cycle later, from a flip-flop). Behind
  it `otpu_dma_split` runs at `REG` 2 (`otpu_native_sys` `XREG` 2): a second slice at each
  channel's bridge, since the PCIe block and the DDR3 banks are on opposite sides of the die
  (section 5).

`sim/verilator/tb_pcie_mm.sv` runs it through `otpu_dma_split` and both `otpu_mem_ch`s against
LiteDRAM port models at the card's clocks: random commands from any 16-byte offset, FILLs,
zero-length commands, stream gaps and stalls, a byte shadow of each channel checking every C2H
beat, `h2c_done` once per command and never before its last write response, plus assertions
(no burst across 512 bytes, W never starved, R never into a full FIFO). **Measured** (Verilator
5.047): every scenario passes; back to back 64 KiB commands move 0.998 H2C beats per cycle
(3.99 GB/s at 250 MHz) and 0.713 C2H (2.85 GB/s: one channel's reads, `otpu_mem_ch`'s 16
outstanding), the same with the split's second slices as without (0.712).
`tools/memch_test.py` builds `tb_memch`'s slice configurations at `XREG` 2.

## 4. The driver: `opentpu/host/pcie/otpu`

`otpu_pcie.c` binds 10ee:5054 subsystem 10ee:4c50, checks `OTPU_DMA` ID and CAPS against its
`csr.h`, and makes per card N:

| node | call | |
|---|---|---|
| `/dev/otpuN_user` | `mmap` | BAR0's user window (at most 512 KiB, uncached); the CSRs above it program DMA engines that can write anywhere in host memory, so no mapping reaches them |
| `/dev/otpuN_h2c_0` | `pwrite(fd, buf, n, card)` | host -> card; returns once the card's memory holds the data |
| `/dev/otpuN_c2h_0` | `pread(fd, buf, n, card)` | card -> host |

buf, n and card must be multiples of 16 bytes (EINVAL otherwise); card + n at most 4 GiB (bit 31
picks the DDR3 channel). A call pins the caller's pages (no copy) and runs in commands of 256 KiB,
up to 4 queued per direction: each is one queue entry plus the descriptors of its pages (none
across a 4 KiB boundary unless it starts on one: the DMA splits descriptors into requests from
their start, and a request must not cross 4 KiB). H2C pushes the command before its descriptors,
C2H after (the writer must have its descriptors when the data comes). The waiter spins on the
record for `spin_us` (20), then sleeps until an MSI, after which it spins up to `msi_spin_us` (2)
for a record still on its way, and looks again every 200 us whatever happens. A timeout
(`timeout_ms`, 10 s) logs the DMA's state, resets it (engines off, tables, `otpu_pcie_mm`,
counts re-read) and fails the call with EIO. sysfs on `otpuN_user`: `h2c_count`, `c2h_count`,
`mm_status`, `link_status`, `max_payload`, `max_read_request`, `irqs`, `resets`, `msi_early`,
`timer_wakes`.

`otpu-setup` installs it with DKMS (package `otpu-pcie`, version `1.<source hash>`) beside the
XDMA driver, and the udev rule opens the three nodes to users.

## 5. Verification and timing

**The co-simulation, `tools/litepcie/sim_core.py`.** The generated core's endpoint, DMA,
`otpu_dma`, MSI, register slices and BAR0 master against LitePCIe's host model (root complex,
host memory above 4 GiB so every TLP is 64-bit, completions reordered across tags and split at
64-byte boundaries as root complexes split them), with the card's side of `otpu_pcie_mm` as a
model, driven by a Python copy of the driver's register sequence through BAR0 at `csr.h`'s
addresses. It checks the CSR map against `csr.csv`, ID and CAPS (MMIO byte order), the user
window, every H2C command's data in card memory when its count is reached, every C2H command's
data in host memory at the cycle its record shows it, H2C and C2H at once, an MSI after every
record, and three timeouts (a lost `h2c_done`, an H2C stream the card stops taking, a C2H stream
it stops sending) with the recovery and transfers after each. **Measured** (LiteX fad9b37,
LitePCIe 586d7a4): the data, ordering and recovery checks all pass. Its runs turned up one
artifact of the model and two problems of the design. The model splits read completions at any
dword count, which real root complexes never do for 16-byte aligned reads (they split at 64-byte
boundaries; the bench now does). The record's MSI reached the host before the record: it is now
delayed in the core and covered in the driver. And the timing pass's stream slices held a beat
across the driver's recovery, so the first transfer after a timeout started with a stale beat:
they now reset with `otpu_pcie_mm`.

**The driver** builds against the hosts' kernel (7.2.5) with `W=1`, no warnings.

**Timing at 250 MHz.** The first m8 build (407a035b, FAST: Vivado's default strategies, no
retiming, 2026-10-15) missed the PCIe user clock by 1.195 ns, with more than 5000 failing paths:
LitePCIe's completion buffers in block RAM (1.8 ns clock-to-out) into the reader's data FIFO,
the descriptor tables' 1024-deep LUTRAM write fanout, the loopback mux, the PHY's CDC FIFOs, and
`otpu_pcie_mm`'s burst decision (11 logic levels). The second build (2df5b89a) has the fixes of
sections 2 and 3.

## 6. On the card

(The card session's results go here: selftest, memcal, DMA rates against XDMA's, the quals.)
