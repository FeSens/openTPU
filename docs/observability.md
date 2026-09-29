# Observability: registers, counters and the hardware trace

The contract between the board RTL (`rtl/boards/ypcb-00338/otpu_ctrl.sv`, `otpu_trace.sv`) and
the host (`opentpu/host`, `otpu-smi`, `otpu-lens`). Register offsets are bytes from BAR0; every
register is 32 bits. The control block decodes 12 address bits (4 KiB; the map repeats through
the 64 KiB BAR). Reads answer a few cycles after the address (the data path is registered);
undefined offsets read 0xDEADBEEF.

## Register map (version 3)

The version 1 registers keep their offsets. Version 3 adds one free-running counter,
MXU_STARVE (0x160); everything else is as in version 2. A version 2 bitstream reads 0xDEADBEEF
at 0x160, so the host reads the counters its REGMAP has (`opentpu/host/regs.py`, `counters()`).

DDR_MTS (0x054) came later without a version bump, because CAPS bit3 announces it. On a
bitstream built before it, the bit is clear and 0x054 reads 0xDEADBEEF, and the host reports the
DDR3 speed as unknown.

| Offset | Name | Access | Meaning |
|---|---|---|---|
| 0x000 | ID | RO | 0x4F545055 ("OTPU") |
| 0x004 | VERSION | RO | {D[15:0], MCOLS[7:0], LANES[7:0]} |
| 0x008 | CTRL | RW | bit0 RUN, bit1 LOAD, bit2 CLEAR (per-run counters only) |
| 0x00C | STATUS | RO | bit0 HALTED, bit1 ERROR, bit2 LOADING, bit3 WR_IDLE, bit4 AXI_ERR, bit5 CALIB0, bit6 CALIB1, bit7 RUN |
| 0x010 | PROG_ADDR | RW | program byte address |
| 0x014 | PROG_N | RW | program length (instructions) |
| 0x018, 0x01C | CYCLES lo, hi | RO | cycles of the current or last run (cleared by CLEAR) |
| 0x020 | ICOUNT | RO | instructions retired in the run |
| 0x024 | B_RD | RO | port B read requests taken (run) |
| 0x028 | B_WR | RO | port B write requests taken (run) |
| 0x02C | A_RD | RO | port A read requests taken (run) |
| 0x030 | SW_WR | RO | scalar (QST) write requests taken (run) |
| 0x034 | B_STALL | RO | cycles a port B request waited (run) |
| 0x038 | SCRATCH | RW | host bring-up test |
| 0x03C | REGMAP | RO | register map version (3; 2 before MXU_STARVE) |
| 0x040 | CAPS | RO | bit0 trace buffer present, bit1 temperature present, bit2 I2C pins present, bit3 DDR_MTS present, bit4 the MXU runs 4-bit weights (docs/quant.md), bit5 column reuse: MM PAIR / QACT DUP (docs/isa.md), bit6 DSTEP: the DMA's DeltaNet head step (docs/isa.md), bit7 CHASH: the hashed channel interleave, [15:8] log2(trace depth), [23:16] log2(P/Q window cycles), bit24 ACT_ROWS present, bit25 run arguments ARG0..7 (docs/isa.md), bit26 STREAM: the stream engine runs STREAM's hardware subset (docs/stream.md, docs/isa.md), bit27 reserved for HOSTCAL: LiteDRAM controllers, calibrated by the host through BAR0 0x10000 (branch litedram-int), bit28 / bit29 proposed for the flash controller and the reboot (docs/flash.md, a draft), bits 30..31 not assigned (a new flag takes the lowest free bit and is added here first) |
| 0x044 | CORE_KHZ | RO | accelerator clock in kHz (a build parameter; the host turns cycles into time with it) |
| 0x048 | BUILD_ID | RO | a build parameter: the first 8 hex digits of the git commit |
| 0x04C | TEMP | RO | bit31 valid, [11:0] the XADC die-temperature code (from the block design's XADC, shared with the MIGs; °C = code × 503.975 / 4096 − 273.15). Valid once channel 0 is calibrated and has reported a non-zero code |
| 0x050 | SNAP | W / R | write (any value): latch every free-running counter into its shadow at once; read: number of snapshots taken |
| 0x054 | DDR_MTS | RO | a build parameter: the DDR3 data rate the MIGs were generated for, in MT/s (800, 1066, 1300, ...; `make bit DDR=`). Valid when CAPS bit3 is set |
| 0x058 | ACT_ROWS | RO | a build parameter: the ACT RAM rows, the most rows of one MM (`make bit ACT_ROWS=`; docs/isa.md). Valid when CAPS bit24 is set; otherwise MCOLS |
| 0x060 + 4k | ARG0..ARG7 | R / W | the run's arguments: R8..R15 of the sequencer at the start of every run (docs/isa.md "Arguments"); reset 0, kept across runs. Present when CAPS bit25 is set |

Build parameters (`make -C boards/ypcb-00338 bit`): CORE_KHZ is computed from CORE_MHZ the way
the block design rounds it (the MMCM divides 800 MHz in steps of 1/8: 100 → 100000, 80 →
80000, 75 → 75294); BUILD_ID is the first 8 hex digits of the git commit the bitstream was
built from (`git rev-parse HEAD`; 0 if unknown; `make bit BUILD_ID=...` overrides). DDR_MTS is
the `DDR=` speed of the build (the nominal rate: 1066 for DDR3-1066). It tells apart two
bitstreams of one commit built at different DDR3 speeds, which BUILD_ID cannot. `otpu-smi`
shows it in the DDR3 row ("DDR3-1066 ch0 ok ch1 ok"), and the config lines of `otpu-selftest`
and `otpu-diag` show it too. The board model (`sim/verilator/tb_board.sv`) reports CORE_KHZ
100000, BUILD_ID 0x0B0A4D00, TEMP code 0xA1A (45 °C) and no DDR_MTS (parameter 0, CAPS bit3
clear).

### Free-running counters

These are never cleared by CLEAR, only by the core reset (the bitstream load; the core reset
follows the clock generator's lock, not PCIe PERST). They are 64 bits; the host reads the
shadows (low word at the offset, high word at +4) after writing SNAP. Utilization is the
difference between two snapshots divided by the UPTIME difference. All thirteen are latched in the
same cycle, so a snapshot is consistent. The units' signals reach the counters through a
register: an event is counted one or two cycles after it happens (a snapshot taken right at a
transition may see it in the next snapshot).

| Offset | Counter | Counts cycles (or events) where |
|---|---|---|
| 0x100 | UPTIME | always (cycles since reset) |
| 0x108 | RUNNING | RUN and not HALTED |
| 0x110 | MXU_BUSY | the MXU has an instruction (started and not completed: streaming, computing or draining) |
| 0x118 | MXU_MAC | the MXU consumes a weight chunk (a D×MCOLS MAC step; the P line's mx) |
| 0x120 | VPU_BUSY | the VPU has an instruction (started, not completed) |
| 0x128 | QNT_BUSY | the quantizer has an instruction |
| 0x130 | DMA_BUSY | the DMA has an instruction |
| 0x138 | TMEM_DENY | some unit with a TMEM request was not granted (any of the P line's fm fq fv fc) |
| 0x140 | DRAM_RD | events: 64-byte beats read from DRAM (R handshakes, both channels; a port B read is 2 beats, a port A read 1, an A read that reuses the previous beat 0) |
| 0x148 | DRAM_WR | events: 64-byte beats written to DRAM (W handshakes, both channels; a port B write is 1 beat per channel its word mask touches, a QST word write 1) |
| 0x150 | DRAM_WAIT | a slice DRAM request (port A, B or the QST port) was not accepted. Request backpressure only: the DRAM adapter queues up to RD reads per channel, so this stays near 0 even when the MXU starves for read data (then MXU_BUSY - MXU_MAC is large; measured on the card: 94% vs 21% for Qwen3) |
| 0x158 | INSTR | events: instructions retired (as ICOUNT: over one run the difference equals ICOUNT) |
| 0x160 | MXU_STARVE | (version 3) the MXU is streaming a command's weight chunks and none has arrived: its chunk FIFO is empty (the Q line's ms). It excludes the MXU's compute-only and drain cycles |

DRAM_WAIT counts only requests the memory has not yet accepted. The memory can take requests
and still return data too slowly (on the card, single-beat AXI reads reached about a quarter
of a channel's bandwidth, with DRAM_WAIT near 0). MXU_STARVE sees that case: MXU_BUSY minus
MXU_MAC minus MXU_STARVE is roughly the MXU's own time (TMEM arbitration, drain, waiting for
its operands).

The systolic MXU (MXU_IMPL 2) starts a row only once all of the row's chunks are in the FIFO, so
it waits for memory mostly with part of a row in the FIFO: blocked (the Q line's mb), not starved.
With late reads (the mlp kernel on the native path, a 400-cycle read latency:
tests/test_observability.py test_mxu_starve_counter) the tree MXU counts 53971 starved and 5
blocked cycles, and the systolic MXU 26811 and 27857; both wait for about 0.8 of their busy
time. (On the removed MIG AXI path, single-beat reads at 4 cycles per read transaction gave
34303 / 2051 and 10788 / 26258.) Follow-up: otpu-smi's
"MXU-starve" reads only MXU_STARVE and so reads low on a systolic card. It needs mb, or a
starve that includes a row still filling.

The program loader's reads (LOAD) count in DRAM_RD and DRAM_WAIT too; RUNNING over one run
equals the run's CYCLES.

### Trace buffer

The trace buffer records the slice's trace events -- the lines the simulator prints with `+trace`
(`opentpu/profile.py`) -- as 64-bit records in a ring of DEPTH records (board: 16384, CAPS
[15:8]). `opentpu/hwtrace.py` turns the records back into those lines
(`records_to_trace(records, sid=0)`, `ring_order(raw, count, depth)`).

| Offset | Name | Access | Meaning |
|---|---|---|---|
| 0x200 | TRACE_CTRL | RW | bit0 ENABLE (record while RUN), bit1 CLEAR (write 1: empty the buffer and zero the counters; reads 0), bit2 STOP_WHEN_FULL (1: keep the first records, 0: keep the last), bit3 BUSY (read only: events taken but not yet in the buffer) |
| 0x204 | TRACE_COUNT | RO | records written since the last clear (saturates at 2^32 − 1) |
| 0x208 | TRACE_DROP | RO | events (trace lines) lost because the capture queue was full |
| 0x20C | TRACE_ADDR | RW | index of the record to read (taken modulo DEPTH) |
| 0x210 | TRACE_LO | RO | bits [31:0] of the record at TRACE_ADDR |
| 0x214 | TRACE_HI | RO | bits [63:32]; reading it increments TRACE_ADDR |

### I2C pins

Two I2C buses reach the FPGA (pins from the vendor XDC): bus 0 is the LM73 temperature
sensor's (SCL N24, SDA N25, ALERT P25), bus 1 the PCIe edge connector's SMBus (SCL R26, SDA
R27), which the host's own SMBus controller may also drive. The RTL has no I2C master: the
host bit-bangs the lines through two registers (`opentpu/host/i2c.py`). Each line is open
drain: an IOBUF that drives 0 or floats, with the weak internal pull-up as a backup to the
board's.

| Offset | Name | Access | Meaning |
|---|---|---|---|
| 0x220 | I2C_CTRL | RW | 1 drives the line low, 0 releases it: bit0 SCL0, bit1 SDA0, bit2 SCL1, bit3 SDA1. Reset: 0 (every line released) |
| 0x224 | I2C_IN | RO | the pin levels, two flip-flops from the pins: bit0 SCL0, bit1 SDA0, bit2 SCL1, bit3 SDA1, bit4 ALERT0 (the LM73's ALERT, active low) |

A bitstream that never writes I2C_CTRL leaves the pins as an unconnected design would. The
host's master only reads: a transaction writes at most a command code (a register pointer)
before a repeated START, never data, so it cannot change a regulator's settings (the
module's docstring lists the rules). Processes share the buses through an flock on
`/tmp/otpu/<device>.i2c.lock`.

With STOP_WHEN_FULL = 0 the buffer is a ring. The oldest record is at index
TRACE_COUNT mod DEPTH when TRACE_COUNT > DEPTH, otherwise at index 0. With STOP_WHEN_FULL = 1
recording ends when TRACE_COUNT reaches DEPTH; later events are neither recorded nor counted
as dropped.

Reading a run's trace: write TRACE_CTRL = CLEAR, then ENABLE (| STOP_WHEN_FULL), then run; after
HALTED, poll TRACE_CTRL until BUSY is 0 (the last cycles' events drain for up to a few thousand
cycles), read TRACE_COUNT and TRACE_DROP, write TRACE_ADDR = 0 (or the oldest index), and read
TRACE_LO, TRACE_HI pairs (the pair is one record; HI moves to the next). The records of a
run keep the simulator's line order, so the ring's oldest records may be the tail of a group
(see below): the decoder skips it, as it skips a group cut at the end.

Capture: every cycle's events enter a queue of 32 cycles (bundles); each cycle's events become
one to 61 records, written one per cycle. A cycle whose events find the queue full is lost as a
whole and its events are added to TRACE_DROP. The steady rate is up to one record per cycle,
enough for decode kernels (an instruction is 6 to 11 records, a P/Q window 17); bursts (the
window filling at the program start) are absorbed by the queue.

### Trace records (64 bits)

```
[63:60] type   [59:0] payload (by type)
```

Cycles are the slice's cycle counter, the `c=` of the trace lines (low 32 bits): 0 in the first
cycle the slice runs, one cycle after RUN rises -- a cycle is CYCLES − 1 at that time, and the
H record's cycle + 1 is the run's CYCLES. A record may be followed by records that complete it
(its group); fields not listed are 0.

| Type | Name | Payload | Followed by | Rebuilt trace line |
|---|---|---|---|---|
| 1 | D | [55:52] slot, [47:32] pc, [31:0] cycle | 2 O | `T0 D c= s= pc= op= w1= w2= w3=` (op and the resolved w1..w3 from the O records) |
| 9 | O | [52] half (0, then 1), [51:0] that half of the 104-bit {op[7:0], w3, w2, w1} | | |
| 2 | S | [55:52] slot, [51:48] unit, [47:32] cycle − ready cycle, [31:0] cycle | 1 R if [47:32] = 0xFFFF | `T0 S c= s= u= r=` |
| 10 | R | [31:0] the ready cycle (the delay did not fit 16 bits) | | |
| 3 | G | [55:52] slot, [31:0] cycle | | `T0 G c= s=` |
| 4 | E | [55:52] slot, [31:0] cycle | | `T0 E c= s=` |
| 5 | U | [51:48] unit, [31:0] cycle | V per counter: unit 1 (MXU) starve bp frz deny; 2 (QUANT) frz; 3 (VPU) frz | `T0 U c= u= k=v ...` |
| 6 | P | [55:32] n (window cycles), [31:0] cycle (the window's last) | 9 V: bm bd am aq mx fm fq fv fc | `T0 P c= n= bm bd am aq mx fm fq fv fc` |
| 7 | Q | as P | 6 V: bs as ms mb ff ld | `T0 Q c= n= bs as ms mb ff ld` |
| 8 | H | [31:0] cycle | 4 V: bmxu bdma amxu aq | `T0 H c= bmxu= bdma= amxu= aq=` |
| 11 | V | [59:56] field index (0, 1, ... in the order above), [31:0] value | | |

Every field is exact: counter values are 32 bits (as the simulator's; a run is < 2^32 cycles),
a start delay that does not fit 16 bits comes with an R record, the resolved operands with two
O records. Slots are 4 bits (the board's 16-slot window), pc 16 bits (IMEM holds 4096
instructions).

Within a cycle the lines (and records) come in a fixed order: G, E (by slot), S (by unit), D,
U (MXU, QUANT, VPU), H, P, Q. The simulator prints them in that order from the same signals
(`rtl/top/otpu_slice.sv`, reported one cycle after the fact), so rebuilding the records gives
exactly what the simulator prints with +trace for the same run; `tests/test_observability.py`
runs kernels both ways on the board model and compares. P and Q lines come every P/Q window
(board: 1024 cycles, CAPS [23:16]; in simulation `+bucket=N` sets it) and at the halt.

## Host-side state

A process that holds the device (the runner) takes an exclusive lock
(`/tmp/otpu/<device>.lock`, flock). It publishes `/tmp/otpu/<device>.json`, which `otpu-smi`
reads:
- the owner: pid, command, start time;
- the model and the DRAM layout: weights, KV cache (capacity and used), program area, free;
- throughput: tokens, the last step's cycles, tok/s.

Monitors never take the lock. They only read registers, and use SNAP for consistent samples.

## Power

No public document names the card's regulators or says whether any of them can report its
power on a bus the FPGA reaches. `otpu-diag` (section i2c) and `otpu-i2c scan` look for
one on both I2C buses. `otpu-smi` reports:
- **Temperature**, measured: the die (TEMP) and, when the LM73 answers, the board ("Board").
- **Power, measured**, when a PMBus device on the card's I2C buses reports it: the sum over
  those devices of READ_PIN, else READ_POUT, else READ_VOUT × READ_IOUT, shown as
  "Power 21.5W measured". The first query scans the buses and saves what it found to
  `/tmp/otpu/<device>.i2c.json` (`otpu-i2c scan` and `otpu-diag` refresh it); later ones read
  only those devices. `--no-i2c` skips the buses.
- **Power, estimated** otherwise. On the YPCB-00338 it is always the estimate: the first
  card scan (2026-09-27, build b11bb679) found only the LM73 at 0x4a on bus 0 (43.00 C board
  temperature) and nothing on the SMBus, so the card has no readable power monitor. Shown as "Power ~5.1W estimate": static + Σ over units of (that unit's dynamic power × its measured utilization). The per-unit powers come from the Vivado build's `report_power`, written as JSON to `build/vivado/reports/power.json`. Without that file, power shows as n/a.
