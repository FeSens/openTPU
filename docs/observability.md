# Observability: registers, counters and the hardware trace

The contract between the board RTL (`rtl/boards/ypcb-00338/otpu_ctrl.sv`, `otpu_trace.sv`) and
the host (`opentpu/host`, `otpu-smi`, `otpu-lens`). Register offsets are bytes from BAR0; every
register is 32 bits. The control block decodes 12 address bits (4 KiB); the BAR is 64 KiB.

## Register map (version 2)

The version 1 registers keep their offsets.

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
| 0x03C | REGMAP | RO | register map version (2) |
| 0x040 | CAPS | RO | bit0 trace buffer present, bit1 temperature present, [15:8] log2(trace depth), [23:16] log2(P/Q window cycles) |
| 0x044 | CORE_KHZ | RO | accelerator clock in kHz (a build parameter; the host turns cycles into time with it) |
| 0x048 | BUILD_ID | RO | a build parameter (e.g. the low 32 bits of the git commit) |
| 0x04C | TEMP | RO | bit31 valid, [11:0] the XADC die-temperature code (from MIG channel 0's device_temp; °C = code × 503.975 / 4096 − 273.15) |
| 0x050 | SNAP | W / R | write: latch every free-running counter into its shadow at once; read: number of snapshots taken |

### Free-running counters

These are never cleared by CLEAR, only by reset (the bitstream load or PCIe PERST). They are
64 bits; the host reads the shadows (low word at the offset, high word at +4) after writing SNAP.
Utilization is the difference between two snapshots divided by the UPTIME difference.

| Offset | Counter | Counts cycles (or events) where |
|---|---|---|
| 0x100 | UPTIME | always (cycles since reset) |
| 0x108 | RUNNING | RUN and not HALTED |
| 0x110 | MXU_BUSY | the MXU has an instruction (in flight or draining) |
| 0x118 | MXU_MAC | the MXU consumes a weight chunk (a D×MCOLS MAC step) |
| 0x120 | VPU_BUSY | the VPU has an instruction |
| 0x128 | QNT_BUSY | the quantizer has an instruction |
| 0x130 | DMA_BUSY | the DMA has an instruction |
| 0x138 | TMEM_DENY | some unit with a TMEM request was not granted |
| 0x140 | DRAM_RD | events: 64-byte beats read from DRAM (both channels) |
| 0x148 | DRAM_WR | events: 64-byte beats written to DRAM |
| 0x150 | DRAM_WAIT | a DRAM request was waiting for the memory |
| 0x158 | INSTR | events: instructions retired |

### Trace buffer

| Offset | Name | Access | Meaning |
|---|---|---|---|
| 0x200 | TRACE_CTRL | RW | bit0 ENABLE (record while RUN), bit1 CLEAR (write 1: empty the buffer and zero the counters), bit2 STOP_WHEN_FULL (1: keep the first records, 0: keep the last) |
| 0x204 | TRACE_COUNT | RO | records written since the last clear (saturates at 2^32 − 1) |
| 0x208 | TRACE_DROP | RO | events lost because the capture queue was full |
| 0x20C | TRACE_ADDR | RW | index of the record to read |
| 0x210 | TRACE_LO | RO | bits [31:0] of the record at TRACE_ADDR |
| 0x214 | TRACE_HI | RO | bits [63:32]; reading it increments TRACE_ADDR |

With STOP_WHEN_FULL = 0 the buffer is a ring. The oldest record is at index
TRACE_COUNT mod DEPTH when TRACE_COUNT > DEPTH, otherwise at index 0.

### Trace records (64 bits)

```
[63:60] type   [59:0] payload (by type)   -- cycle = the run's CYCLES counter, low 32 bits
```

| Type | Name | Payload | Rebuilt trace line (docs: opentpu/profile.py) |
|---|---|---|---|
| 1 | D | [55:52] slot, [47:32] pc, [31:0] cycle | `T0 D c= s= pc=` (op and w1..w3 come from the program the host loaded) |
| 2 | S | [55:52] slot, [51:48] unit, [47:32] ready cycle − cycle (saturating), [31:0] cycle | `T0 S c= s= u= r=` |
| 3 | G | [55:52] slot, [31:0] cycle | `T0 G c= s=` |
| 4 | E | [55:52] slot, [31:0] cycle | `T0 E c= s=` |
| 5 | U | [51:48] unit, [47:32] a counter (saturating), [31:0] cycle; one record per counter the unit reports, in the RTL's order | `T0 U c= u= k=v ...` |
| 6 | P | [59:56] field index, [55:32] value, [31:0] window end cycle; one record per field | `T0 P c= n= bm bd am aq mx fm fq fv fc` |
| 7 | Q | as P | `T0 Q c= n= bs as ms mb ff ld` |
| 8 | H | [59:56] field index, [55:32] value, [31:0] cycle | `T0 H c= bmxu= bdma= amxu= aq=` |

Rebuilding the trace lines from the records must give exactly what the simulator prints with
+trace for the same run. The test runs the same program both ways on the board model and
compares.

## Host-side state

A process that holds the device (the runner) takes an exclusive lock
(`/tmp/otpu/<device>.lock`, flock). It publishes `/tmp/otpu/<device>.json`, which `otpu-smi`
reads:
- the owner: pid, command, start time;
- the model and the DRAM layout: weights, KV cache (capacity and used), program area, free;
- throughput: tokens, the last step's cycles, tok/s.

Monitors never take the lock. They only read registers, and use SNAP for consistent samples.

## Power

The card has no current sensing the accelerator can read. `otpu-smi` reports:
- **Temperature**, measured (TEMP).
- **Power, estimated.** static + Σ over units of (that unit's dynamic power × its measured utilization). The per-unit powers come from the Vivado build's `report_power`, written as JSON to `build/vivado/reports/power.json`. Without that file, power shows as n/a.
