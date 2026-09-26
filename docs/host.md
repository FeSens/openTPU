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
| `otpu-smi` | the cards' state, like nvidia-smi (section 8) |
| `otpu-selftest` | staged bring-up (section 4) |
| `otpu-diag` | the full hardware diagnostic: every check, no stopping, a works / does-not-work matrix (section 5) |
| `otpu-chat` | chat with Qwen3, LFM2 or Qwen3.5 on the card (section 6) |
| `otpu-lens` | Lens profiles from the card's hardware trace ([lens.md](lens.md)) |

Without installing, `python3 -m opentpu.host.<smi|selftest|chat|hwlens>` does the same.

## 1. Requirements

- A Linux x86-64 PC with a free x8 (or x16) PCIe slot. The card takes power from the slot;
  make sure the slot provides enough power and there is airflow over the heatsink.
- Kernel headers for the running kernel (`linux-headers-$(uname -r)`), `gcc`, `make`, `git`.
- Python 3.10+ with `numpy`; for the model also `torch`, `transformers`, `safetensors`.
- This repository, and the models in `models/` (Hugging Face checkpoints, e.g.
  `huggingface-cli download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B`; likewise
  `LiquidAI/LFM2.5-230M` -> `models/LFM2.5-230M`, `Qwen/Qwen3.5-0.8B` -> `models/Qwen3.5-0.8B`).
- No configuration: the tools read the bitstream's D / MCOLS / LANES from its VERSION register.
  Leave `OTPU_MCOLS` / `OTPU_LANES` unset (they configure the simulators); when set, they must
  match the bitstream or the tools stop, naming both values.

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
`intel_iommu=on iommu=pt`). If DMA calls hang and `dmesg` shows XDMA timeouts, the interrupts
do not arrive: reload in poll mode, `sudo modprobe -r xdma; sudo modprobe xdma poll_mode=1`
(`XDMA_POLL=1 opentpu/host/setup_pcie.sh`).

## 3. Check the card on the bus

The card must be configured before the PC enumerates the bus: either boot the PC with the
bitstream already in the card's configuration flash, or program over JTAG and then rescan:

```sh
lspci -d 10ee: -nn                   # the card: "... Xilinx ... [10ee:7028]"
sudo lspci -d 10ee: -vv | grep -E "LnkCap|LnkSta|Region"
#   LnkCap/LnkSta: Speed 2.5GT/s, Width x8  <- Gen1 x8 is the design (not a downtrained link);
#                                             fewer lanes cost DMA bandwidth only
#   Region 0: Memory at ... [size=1M]   <- BAR0, the control registers (AXI-Lite master, 1 MiB)
#   Region 1: Memory at ... [size=64K]  <- the XDMA's own registers (the driver uses them)
# after JTAG programming, without a reboot:
echo 1 | sudo tee /sys/bus/pci/devices/0000:XX:00.0/remove
echo 1 | sudo tee /sys/bus/pci/rescan
sudo rmmod xdma; sudo modprobe xdma   # the driver must re-probe the new function
```

A quick register check without Python: `sudo dd if=/dev/xdma0_user bs=4 count=1 2>/dev/null |
xxd` must print `55 50 54 4f` ("OTPU", the ID register at offset 0, little-endian).

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
| link | `lspci -d 10ee:` lists the card? If not: rescan after JTAG (`setup_pcie.sh --rescan`) or reboot. Listed but `/dev/xdma0_user` missing: `lsmod \| grep xdma`, `dmesg \| grep -i xdma`. ID `0xffffffff`: the link dropped (the FPGA was reprogrammed after enumeration: rescan). Another ID: a bitstream without openTPU, or the AXI-Lite path in the block design |
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
| mem | per channel (raw channel addresses): walking 1 and walking 0 over the 512 bits of a beat, walking address bits (aliasing named), 16 random blocks spread over the channel, 200 partial (byte-strobe) writes, DMA bandwidth each way; the interleave through the accelerator's address map; with `--mem full` a march C- over every byte with address-in-address data (progress line; errors per byte lane, DQ bit and address bit) |
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
otpu-chat --backend board                      # interactive
otpu-chat --backend board --prompt "Why is the sky blue?"
otpu-chat --backend board --clock-mhz 100      # override the core clock (v1 bitstreams)
otpu-chat --backend board --model lfm2         # LFM2.5-230M instead of Qwen3-0.6B
otpu-chat --backend board --model qwen35       # Qwen3.5-0.8B (needs RDOT / OUTER / LOG2 in the bitstream)
```

The first call writes the model image (at the default `--cap 2048`: 0.69 GiB for Qwen3-0.6B,
0.27 GiB for LFM2.5-230M, 0.77 GiB for Qwen3.5-0.8B) to the card; every token then writes the
embedding row and the token's program (a few tens of KiB), runs, and reads the logits (0.58 MiB
for Qwen3, 0.25 MiB for LFM2, 0.95 MiB for Qwen3.5). After each answer the tool prints wall-clock tokens/s and the device's own
cycles per token (from the CYCLES register), converted with the bitstream's CORE_KHZ register
(register map 2) or `--clock-mhz` (default 100 on a register map 1 bitstream). While it runs,
`otpu-smi` shows the process, the model, the DRAM in use and tokens/s.

`--backend board-sim` runs the same driver against the Verilator board model (bit-exact, but
minutes per token for the real model; use it with small models).

## 7. Control registers

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

## 8. Device lock, status file and otpu-smi

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
