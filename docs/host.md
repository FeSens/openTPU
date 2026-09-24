# Host PC: driving the openTPU card over PCIe

The YPCB-00338 card runs the accelerator behind a Xilinx XDMA PCIe bridge (Gen2 x8). This page
covers the host side: the PC the card is plugged into, its driver, and the tools that talk to
the card. Building and loading the bitstream is in [board.md](board.md).

What the host sees:

| Device node | What it is | Used for |
|---|---|---|
| `/dev/xdma0_user` | BAR0, the AXI-Lite control registers (`rtl/boards/ypcb-00338/otpu_ctrl.sv`) | start / load / status / counters |
| `/dev/xdma0_h2c_0` | DMA host -> card; file offset = card AXI address | writing DRAM |
| `/dev/xdma0_c2h_0` | DMA card -> host; file offset = card AXI address | reading DRAM |

The card's AXI address map: DDR3 channel 0 at `0x0000_0000`, channel 1 at `0x8000_0000`,
2 GiB each. The accelerator sees one 4 GiB logical DRAM interleaved over the two channels in
64-byte beats (logical beat *b* is on channel *b* % 2 at offset (*b* // 2) * 64).
`host/board.py` applies that map, so everything above it uses logical addresses.

## 1. Requirements

- A Linux x86-64 PC with a free x8 (or x16) PCIe slot. The card takes power from the slot;
  make sure the slot provides enough power and there is airflow over the heatsink.
- Kernel headers for the running kernel (`linux-headers-$(uname -r)`), `gcc`, `make`, `git`.
- Python 3.10+ with `numpy`; for the model also `torch`, `transformers`, `safetensors`.
- This repository, and the model in `models/Qwen3-0.6B` (the Hugging Face checkpoint:
  `huggingface-cli download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B`).

## 2. Build and load the XDMA driver

`host/setup_pcie.sh` does sections 2 and 3 in one go (finds the card, builds and loads the
driver, adds the udev rule, reads the ID register); `host/setup_pcie.sh --rescan` after JTAG
programming. The manual steps:

```sh
git clone https://github.com/Xilinx/dma_ip_drivers
cd dma_ip_drivers/XDMA/linux-kernel/xdma
make
sudo make install            # or: sudo insmod xdma.ko
sudo modprobe xdma           # after make install
ls /dev/xdma0_*              # xdma0_user, xdma0_h2c_0, xdma0_c2h_0, xdma0_control, ...
```

The driver only binds to the PCI IDs in its table (`xdma/xdma_mod.c`, `pci_ids[]`). The block
design keeps Xilinx's default vendor ID `10ee` and the default XDMA device ID; if `lspci`
shows an ID that is not in the table, add it there and rebuild the driver.

Non-root access: a udev rule, e.g. `/etc/udev/rules.d/60-xdma.rules`:

```
KERNEL=="xdma[0-9]*", MODE="0666"
```

then `sudo udevadm control --reload && sudo udevadm trigger` (or reload the driver).

If DMA transfers fail on a machine with the IOMMU on, boot with `iommu=pt` (Intel:
`intel_iommu=on iommu=pt`).

## 3. Check the card on the bus

The card must be configured before the PC enumerates the bus: either boot the PC with the
bitstream already in the card's configuration flash, or program over JTAG and then rescan:

```sh
lspci -d 10ee:                       # the card, e.g. "Memory controller: Xilinx ... Device 7028"
sudo lspci -d 10ee: -vv | grep -E "LnkCap|LnkSta|Region"
#   LnkSta: Speed 5GT/s, Width x8     <- Gen2 x8; anything less costs DMA bandwidth only
#   Region 0: Memory at ... [size=...]  <- BAR0, the control registers (AXI-Lite; size set in the block design, >= 4 KiB)
# after JTAG programming, without a reboot:
echo 1 | sudo tee /sys/bus/pci/devices/0000:XX:00.0/remove
echo 1 | sudo tee /sys/bus/pci/rescan
sudo rmmod xdma; sudo modprobe xdma   # the driver must re-probe the new function
```

A quick register check without Python: `sudo dd if=/dev/xdma0_user bs=4 count=1 2>/dev/null |
xxd` must print `55 50 54 4f` ("OTPU", the ID register at offset 0, little-endian).

## 4. Self-test

```sh
python3 tools/board_selftest.py                       # stages 1-8 on /dev/xdma0
python3 tools/board_selftest.py --qwen models/Qwen3-0.6B   # plus the model, vs the ISA simulator
python3 tools/board_selftest.py --sim                 # rehearsal on the Verilator board model
```

Stages, in order (it stops at the first failure and prints a hint):

| Stage | Checks |
|---|---|
| link | the ID register reads `0x4F545055` |
| config | the bitstream's D / MCOLS / LANES equal `opentpu.isasim.board_config()` |
| calib | both DDR3 controllers calibrated (STATUS bits 5, 6) |
| regs | SCRATCH register write / read |
| addr | walking address bits and random patterns on each channel (raw channel addresses) |
| pattern | random data through the channel interleave, unaligned edges, the top of DRAM; sub-beat host writes (partial byte strobes) |
| bandwidth | host -> card and card -> host DMA rate |
| kernel | a program using every unit (DMA, VPU, quantizer, MXU, QST), then one of partial DRAM writes (QST bytes, short stores); DRAM equals the ISA simulator bit for bit |
| qwen | greedy decoding of "What is the capital of France?" equals the ISA simulator token for token |

The qwen stage needs the ISA simulator's reference too (about 3 s per token on the host).

Partial writes matter on this board: each DDR3 channel is 9 x8 devices (72-bit, ECC) without
data-mask pins, so the memory controller turns every write with partial byte strobes into a
read-modify-write. The pattern and kernel stages exercise that path from the host (sub-beat DMA)
and from the accelerator (QST byte writes, word-masked stores). The board model applies byte
strobes directly, so only the card proves the controller's read-modify-write.

## 5. Chat

```sh
python3 tools/chat.py --backend board                      # interactive
python3 tools/chat.py --backend board --prompt "Why is the sky blue?"
python3 tools/chat.py --backend board --clock-mhz 100      # the bitstream's core clock
```

The first call writes the model image (about 0.8 GiB for Qwen3-0.6B) to the card; every token
then writes the embedding row and the token's program (a few tens of KiB), runs, and reads the
logits (0.6 MiB). After each answer the tool prints wall-clock tokens/s and the device's own
cycles per token (from the CYCLES register), converted with `--clock-mhz`.

`--backend board-sim` runs the same driver against the Verilator board model (bit-exact, but
minutes per token for the real model; use it with small models).

## 6. Control registers

`rtl/boards/ypcb-00338/otpu_ctrl.sv` has the authoritative list. The driver's sequence per
token (`host/board.py`, `Board.load_program` and `Board.run`):

1. write the program into DRAM (logical address right after the model image), `CTRL = 0`;
2. `PROG_ADDR`, `PROG_N` (instructions), `CTRL = LOAD`; poll `STATUS.LOADING == 0`;
3. `CTRL = CLEAR` (zero the counters; also holds the core in reset), `CTRL = RUN`;
4. poll `STATUS.HALTED`; read `STATUS`, `CYCLES`, `ICOUNT`, the DRAM port counters;
5. `CTRL = 0`. `STATUS.ERROR` (illegal instruction) and `STATUS.AXI_ERR` (a DRAM access got an
   error response) make the driver raise.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| no `10ee:` device in `lspci` | card not configured at enumeration: flash the bitstream, or program over JTAG and rescan; check the PERST# / refclk constraints of the bitstream |
| `/dev/xdma0_*` missing | driver not loaded, or the device ID is not in `pci_ids[]` |
| ID register reads `0xffffffff` | BAR not mapped / link down (`lspci -vv` shows `!` flags or `Region 0: ... [disabled]`) |
| ID reads something else | wrong bitstream, or the AXI-Lite interconnect in the block design does not reach `otpu_ctrl` |
| calib fails | MIG pinout, memory clock or DDR3 voltage; see board.md |
| DMA very slow | link trained at Gen1 or fewer lanes (`LnkSta`), or IOMMU without `iommu=pt` |
| `run` times out | the core clock or reset is not running, or a DRAM write never gets its response (`STATUS.WR_IDLE` stays 0) |
| kernel stage differs | reproduce with `pytest tests/test_board.py` (the same program on the RTL model), compare the counters |

## Rehearsal without hardware

`host/board.py`'s `SimTransport` replays the driver's register and DMA operations on the
Verilator model of the board (`sim/verilator/tb_board.sv`: control registers, program loader,
slice, DRAM adapter and a two-channel AXI memory with random stalls). `tests/test_board.py`
runs a program and a tiny Qwen3 through it, bit-exact against the ISA simulator. Each flush of
the transport is a fresh simulation: DRAM persists between flushes (through the channel image
files), registers, IMEM and TMEM do not -- the driver loads, runs and reads the counters of a
program within one flush.
