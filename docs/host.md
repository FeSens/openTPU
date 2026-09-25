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
64-byte beats (logical beat *b* is on channel *b* % 2 at offset (*b* // 2) * 64).
`opentpu/host/board.py` applies that map, so everything above it uses logical addresses.

The host software is the package `opentpu/host` (userspace; the kernel side is the stock
Xilinx XDMA driver). `pip install -e .` installs its commands:

| Command | What it does |
|---|---|
| `otpu-smi` | the cards' state, like nvidia-smi (section 7) |
| `otpu-selftest` | staged bring-up (section 4) |
| `otpu-chat` | chat with Qwen3 on the card (section 5) |
| `otpu-lens` | Lens profiles from the card's hardware trace ([lens.md](lens.md)) |

Without installing, `python3 -m opentpu.host.<smi|selftest|chat|hwlens>` does the same.

## 1. Requirements

- A Linux x86-64 PC with a free x8 (or x16) PCIe slot. The card takes power from the slot;
  make sure the slot provides enough power and there is airflow over the heatsink.
- Kernel headers for the running kernel (`linux-headers-$(uname -r)`), `gcc`, `make`, `git`.
- Python 3.10+ with `numpy`; for the model also `torch`, `transformers`, `safetensors`.
- This repository, and the model in `models/Qwen3-0.6B` (the Hugging Face checkpoint:
  `huggingface-cli download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B`).

## 2. Build and load the XDMA driver

`opentpu/host/setup_pcie.sh` does sections 2 and 3 in one go (finds the card, builds and loads the
driver, adds the udev rule, reads the ID register); `opentpu/host/setup_pcie.sh --rescan` after JTAG
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
#   LnkSta: Speed 2.5GT/s, Width x8   <- Gen1 x8; anything less costs DMA bandwidth only
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
otpu-selftest                                 # stages 1-8 on /dev/xdma0
otpu-selftest --qwen models/Qwen3-0.6B        # plus the model, vs the ISA simulator
otpu-selftest --sim                           # rehearsal on the Verilator board model
```

(`python3 -m opentpu.host.selftest ...` is the same without installing.)

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
otpu-chat --backend board                      # interactive
otpu-chat --backend board --prompt "Why is the sky blue?"
otpu-chat --backend board --clock-mhz 100      # override the core clock (v1 bitstreams)
```

The first call writes the model image (about 0.8 GiB for Qwen3-0.6B) to the card; every token
then writes the embedding row and the token's program (a few tens of KiB), runs, and reads the
logits (0.6 MiB). After each answer the tool prints wall-clock tokens/s and the device's own
cycles per token (from the CYCLES register), converted with the bitstream's CORE_KHZ register
(register map 2) or `--clock-mhz` (default 100 on a register map 1 bitstream). While it runs,
`otpu-smi` shows the process, the model, the DRAM in use and tokens/s.

`--backend board-sim` runs the same driver against the Verilator board model (bit-exact, but
minutes per token for the real model; use it with small models).

## 6. Control registers

[observability.md](observability.md) has the register map (version 2: the version 1 registers
plus REGMAP, CAPS, CORE_KHZ, BUILD_ID, TEMP, SNAP, the free-running counters and the trace
buffer); `opentpu/host/regs.py` has the same as constants. The driver's sequence per token
(`opentpu/host/board.py`, `Board.load_program` and `Board.run`):

1. write the program into DRAM (logical address right after the model image), `CTRL = 0`;
2. `PROG_ADDR`, `PROG_N` (instructions), `CTRL = LOAD`; poll `STATUS.LOADING == 0`;
3. (profiling) `TRACE_CTRL = CLEAR`, then `ENABLE` (+ `STOP_WHEN_FULL`);
4. `CTRL = CLEAR` (zero the per-run counters; also holds the core in reset), `CTRL = RUN`;
5. poll `STATUS.HALTED`; read `STATUS`, `CYCLES`, `ICOUNT`, the DRAM port counters (and
   `TRACE_COUNT`, `TRACE_DROP`, the records);
6. `CTRL = 0`. `STATUS.ERROR` (illegal instruction) and `STATUS.AXI_ERR` (a DRAM access got an
   error response) make the driver raise.

**Version 1 and version 2 bitstreams.** `Board.info()` reads REGMAP. A version 1 bitstream has
no such register (it reads `0xDEADBEEF`, or 0): the driver then never touches the version 2
offsets (version 1 decodes 8 address bits, so 0x100 and up alias onto the low registers).
Everything but the counters, the trace and the temperature works: `snapshot()` returns None,
`otpu-smi` shows utilization / power / temperature as n/a, `otpu-lens record` falls back to
counters-only profiles, tokens/s uses `--clock-mhz`.

**Polling.** `XdmaTransport.poll` reads the register back to back for the first 100 us (a
PCIe read is ~1 us: program loads finish in this phase, with no sleep latency), then sleeps
elapsed/32, at most 1 ms, between reads. A wait of T is noticed at most ~T/32 late (a
Qwen3-0.6B token of ~50 ms: <= 1 ms, 2%) with ~32 ln(T / 100 us) + T / 1 ms reads instead of
T / 1 us, and the sleeps free the GIL for the compile thread (below).

**DMA.** Reads go straight into one preallocated numpy buffer per channel (`os.preadv` into
memoryview slices; no concatenation), writes from memoryview slices; both in 8 MiB calls (the
XDMA driver pins each call's pages and builds one descriptor list: 8 MiB bounds that, and the
per-call cost stays under 1% of the transfer).

**Pipelining.** A token's program depends on its position only, so `Engine.step` compiles
(and `BoardBackend.prepare` assembles) position p + 1 on a worker thread while the card runs
p (`Engine(..., pipeline=None)`: on for every backend but the ISA simulator). A precompile made
for another position (after `reset`, or `run_rows`) is waited for and dropped; the compiler's
tracing context is per thread. Results are unchanged (`tests/test_host.py`). Measured with a
fake card whose run takes a fixed time (FakeTransport), Qwen3-0.6B on the host above: the
compile of a step is 4.5-7 ms; per token, device 30 ms: 36.0 ms sequential -> 32.9 ms
pipelined; device 60 ms: 68.1 -> 64.2 ms (the compile is hidden; what remains is the
embedding/RoPE writes, the 0.6 MiB logits read and the poll's wake-up).

## 7. Device lock, status file and otpu-smi

**Lock.** Anything that runs programs or writes the card's DRAM (`Board`, `BoardBackend`,
`otpu-selftest`, `otpu-chat`, `otpu-lens record`) takes an exclusive `flock` on
`/tmp/otpu/<dev>.lock` (`<dev>` = `xdma0` for `/dev/xdma0`; `OTPU_RUN_DIR` moves the directory)
and writes its pid there. A second runner fails at once with `xdma0 is in use by process <pid>
(<command>)`. The lock belongs to the open device (Boards on the same transport share it) and
goes away with the process, however it ends. Monitors (`otpu-smi`) never lock.

**Status file.** The runner publishes `/tmp/otpu/<dev>.json`, replaced atomically after every
token and removed at exit: `pid`, `argv`, `start`, `dev`, `model`, `core_khz`, `dram` (bytes:
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
speed and width from sysfs), DDR3 calibration (STATUS bits 5, 6), temperature (TEMP, measured
by the XADC), the DRAM used / total and the KV cache (from the status file), the DRAM bandwidth
((DRAM_RD + DRAM_WR) deltas x 64 B over the UPTIME delta / CORE_KHZ), utilization (the deltas of
RUNNING, MXU_BUSY, MXU_MAC -- MAC utilization --, VPU_BUSY, QNT_BUSY, DMA_BUSY, TMEM_DENY and
DRAM_WAIT over the UPTIME delta, between two SNAPs) and the owning process.

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
`rtl/boards/ypcb-00338/otpu_trace.sv` exists it is added to the model's sources. The
testbench's AXI-Lite address is 8 bits wide today: the register map 2 offsets (0x100 and up)
need it widened to 12 bits (the RTL side of docs/observability.md).

`opentpu/host/fake.py`'s `FakeTransport` is an in-memory card with register map 2 (or 1):
synthetic counters, a trace buffer served from a list, runs that take a set wall time. The
host tests (`tests/test_host.py`) and `otpu-smi --fake` use it.

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
| `xdma0 is in use by process N` | another runner (chat, selftest, lens record) holds the card: stop it; `otpu-smi` shows it |
| otpu-smi shows n/a for utilization, temperature and power | a register map 1 bitstream (REGMAP reads 0xDEADBEEF), or no `power.json` for the power |
| kernel stage differs | reproduce with `pytest tests/test_board.py` (the same program on the RTL model), compare the counters |
