# Native channels: the slice's memory adapter at any width

A slice reaches its DRAM through `rtl/mem/otpu_native_dram.sv`, the native adapter: per channel a
command stream, write data in write-command order, read data in read-command order with no
backpressure. The adapter, the boards, the host and the co-simulation take a slice's channels as
two parameters:

- **NCH**, channels a slice (a power of two, at least 2);
- **CB**, bytes a channel beat (64 or 128).

NCH 2 x CB 64 is the YPCB-00338's and the LiteDRAM BCU1525's, and builds the RTL those cards
built before, cycle for cycle. The BCU1525's weight-group builds (`bcu1525-wg`, `-s2`, `-x2`) are
**NCH 8 x CB 128**: a wide chunk of 1 KiB a cycle, which the MXU takes whole on port W
([weight_groups.md](weight_groups.md)). This page is the adapter's contract, its ports and every
`NAT_*` knob; the BCU1525's memory path behind the channels (the DDR4 controllers and the bridge
in front of them) is in [bcu1525_wg.md](bcu1525_wg.md).

None of the knobs here is in the Config or the `arch_hash`: programs and images do not depend on
them, and every setting is bit-exact against every other.

## 1. The map

Logical beat `b = addr / CB`, wide chunk `m = b / NCH`, channel `c = (b mod NCH) XOR h(m)`,
channel offset `m x CB + addr mod CB`. `h` is 0 without `CHASH`; with it (every FPGA target) the
XOR fold of m's bits into log2(NCH) bits, bit i of m into bit `i mod log2(NCH)` (at NCH 2: the
parity of m). The fold spreads a strided access over the channels; `otpu_slice`, the adapter and
the host's `chunk_hash` compute the same one. `CFG_DRAM` reports NCH ([15:8]), log2(NCH)
([23:16]) and log2(CB / 64) ([31:24]); `opentpu/host/regs.py` decodes them, and
`opentpu/host/board.py` splits and joins the host's reads and writes by them.

A target states `channels_per_slice`, `channel_bytes` and `beat_bytes`; `opentpu.target.load()`
checks them (NCH a power of two, CB 64 or 128, `D % CB = 0`, `NCH x CB % D = 0`, `NCH x
channel_bytes = DRAM_BYTES`; WG > 1 needs `beat_bytes = D`, a weight row a beat).

## 2. The ports

| port | who | what | priority on a channel |
|---|---|---|---|
| A | the MXU's scales | reads, in runs of channel-consecutive beats | 1 |
| SW | the quantizer's QST | writes, gathered into beats (a fill read first when `NAT_FILL`) | 2 (fill read), 3 (write) |
| B | the DMA (LD, ST, DSTEP, STREAM), the natural-mode MXU | D-byte chunks, reads and writes | 4 |
| W | the MXU's weight groups (WG > 1) | P-byte reads | 5 |

W is last because the weight stream at WG 8 is always backlogged and B's demand is bounded (a
D-byte chunk a cycle): a DSTEP's state on B never waits behind the weights. Ordering between the
ports is the slice's: `wr_idle` holds a read of what a QST or a B write wrote, `rd_idle` a write
over what a read (W's included: queued, in flight or in the output register) has still to read.

**Port B** stays D-wide at any NCH and CB: a D-byte chunk is `D / CB` beats of one wide chunk, so
the DMA is unchanged. Its reads in flight are `256 / NCH` beats a channel by default.

**Port W**: `w_rdy, w_req, w_addr` (a P-aligned word address) in; `w_bv[NCH]`, `w_bd` (NCH lanes
of CB bytes) out. A request is one command on every channel at the same channel beat. The chunk
is not assembled in the adapter: lane c is channel c's beat, each lane in its requests' order and
the lanes independent of each other, with no backpressure. The MXU reserves a chunk FIFO entry
as its request goes (the credit), each lane writes its beat into the entry at its own tail, and
the entry is in the FIFO once every lane has written it. A grouped operand stores a chunk's rows
in the hash's order, so lane c is row c, column c's part: the chunk is stored once, where the MXU
reads it, and nothing permutes it. Each channel has a W queue and a bound on its W reads in
flight (`NAT_XD`).

**Posted beats** (`NAT_XPOST`, `NAT_WPOST`): a memory controller with an AXI data channel of its
own clock can hand port W's read beats straight to the MXU's chunk FIFO lanes (XPOST: the
channels post them, with their counts, across the clock crossing; the adapter never carries the
weight stream's data), and take B's and SW's write data at the take, ahead of the command
(WPOST). XPOST needs WG > 1 and a D chunk of one beat.

## 3. The knobs

`[uarch]` keys of a target (`opentpu/target.py` NAT_KEYS; `otpu_board` / `otpu_top` parameters of
the same names). Absent: the adapter's defaults, the two-channel cards'.

| key | meaning | default | `bcu1525-wg` / `-s2` / `-x2` |
|---|---|---|---|
| `NAT_RD` | port B's reads in flight a channel, in beats | 256 / NCH | 16 |
| `NAT_AD` | port A's read FIFO a channel, in beats (>= 2 APF - 1) | 64 | 16 |
| `NAT_APF` | port A's read run: a miss fetches APF consecutive beats | 32 | 8 |
| `NAT_XD` | port W's reads in flight a channel | 64 | default |
| `NAT_WQD` | the SW queue's depth a channel, in gathered beats | 32 | default |
| `NAT_FILL` | a partial SW beat is read first and written whole (1), or written with its byte mask (0: a memory that takes masks) | 1 | false |
| `NAT_WPOST` | B's and SW's write data posted to the channel at the take (needs `NAT_FILL` 0) | off | true |
| `NAT_SWG` | with WPOST: one SW gather beat for every channel and one posted beat a cycle on one bus (an area cut) | off | true |
| `NAT_XPOST` | port W's beats posted into the MXU's chunk FIFO by the channels | off | true |
| `NAT_RHB` | read data in half beats (needs XPOST, CB 128) | off | true |
| `NAT_WHB` | write data in half beats (needs WPOST and SWG, CB 128) | off | off (no target) |
| `NAT_XSTG` | register stages each way between a slice and its adapter (`otpu_xstg`) | 0 (wired) | 2 (`x2:b2` 3) |
| `NAT_XSTG_P` | the same for the posted W beats and their counts | `NAT_XSTG`'s | 2 |

The simulators take `OTPU_NAT_XSTG` / `OTPU_NAT_XSTG_P` as overrides (the table in
[host.md](host.md), "Simulation overrides").

### Port A's read runs (NAT_APF, NAT_AD)

An A read that misses fetches a run of APF beats on its channel. A goes before W on every
channel, so a run's beats hold the weight stream off, and a run started near the end of an MM's
scale area fetches beats past it. APF 32 suits DDR3 behind two channels (fewer row and bank
changes, [litedram.md](litedram.md) section 11); with eight channels and a grouped MM's small
scale area it over-fetches: at 32 a grouped int8 MM's runs fetch nearly three times the scale
bytes it needs, at 8 about 1.45 times (the rounding of a run on each channel, under 1% of the
MM's bytes), and nothing is gained below 8. That was measured on Verilator with eight channels
of a faster memory model than the BCU1525's DIMMs (grouped MMs back to back, 2.5-3.5% faster
there, MXU-bound MMs unchanged); it was not measured again on the BCU's memory path. The wide
targets take APF 8 and AD 16, and A's FIFO is half the LUT RAM.

### The SW queue's fill reads (NAT_FILL)

A partial SW beat (a QST's bytes that leave part of a beat unwritten: a transposed V column, one
byte a beat) is by default read first and written whole, so that a DDR3 controller never runs a
read-modify-write. A memory that takes byte masks needs none of it: with `NAT_FILL = false` a
partial beat goes out as one masked write, with no fill read (half of a V^T column's commands),
no fill data and no hazard buckets. The adapter a slice at NCH 8, CB 128 (yosys, UltraScale+
cells): 42.2K LUTs + 3.3K LUT RAM cells and 26.0K flip-flops with the fill, 26.1K + 2.7K and
21.9K without: **-21K LUTs and -4.2K flip-flops a slice**. The BCU1525's DDR4 controllers take
write strobes (the DIMMs' DM pins), so its WG targets set it false.

### Half beats (NAT_RHB, NAT_WHB)

At CB 128 the paths between a slice's adapter and its channels are 1,024 bits wide each way, with
LUT RAMs at that width on both sides. Once port W no longer comes back through the adapter
(XPOST), the rest can go at half a beat a cycle and every LUT RAM on the way hold half beats.

- **RHB.** A read beat's halves arrive on consecutive strobes, the low one first. The adapter's B
  and A FIFOs hold half beats. Port B joins a chunk over two cycles, and takes the next read's
  low half in the cycle a chunk goes out, so reads on different channels still go a chunk a
  cycle (two on one channel: one every two cycles). Port A keeps a beat a cycle: each channel
  holds its head beat's low half in a register of its own, and the scale stream's consecutive
  beats are on different channels. Needs XPOST, `NAT_FILL` 0 and CB = D.
- **WHB.** A posted write beat goes out in two cycles on the shared bus. The bus is then one
  posted beat every two cycles for the whole slice, and a DeltaNet state's write-back (DSTEP)
  pays for it: several percent of a dense decode token in a co-simulation on that faster memory
  model, where RHB is free. In no target.

### Stages to the adapter (NAT_XSTG)

`rtl/mem/otpu_xstg.sv` puts `NAT_XSTG` register stages each way between a slice and its adapter,
for a floorplan with the adapters a die away from their slices; `NAT_XSTG_P` is the posted W
beats' and their counts'. A request crosses with the ready the slice saw and lands in a queue of
at least 2N + 2 entries on the adapter's side, whose room (for 2N + 1 more) is the slice's
ready N stages back: no handshake logic sits on the slice's side, and a stage adds latency, not
a bubble. The slice holds `wr_idle` and `rd_idle` low for the 2N cycles after each write or read
it sends. The BCU1525's WG builds use 2 (the adapters beside the slice, the channels' AXI halves
at their DIMMs).

## 4. The co-simulation's channel model

`sim/verilator/otpu_native_mem.sv`, per channel: `+nat_bpc=N` limits it to N / 1000 bytes a core
cycle, `+axi_lat` is the read latency in core cycles, `+nat_refi=P +nat_rfc=L` a refresh-like
duty loss (no command for L cycles every P, the channels staggered). `opentpu.target.mem_model(t,
mhz)` derives them from a target's platform keys `sim_ch_gbps`, `sim_lat_ns`, `sim_refi_ns`,
`sim_rfc_ns` (`bcu1525-wg`: 9.12 GB/s a channel, 250 ns, 7,800 / 350 ns), and `tb_top` runs take
them with `OTPU_MEM_MODEL=<core MHz>`. The board model of a multi-channel card
(`SimTransport(target=...)`) uses the card's own bridge and a DIMM model instead; its fidelity
against the cards is in [bcu1525_wg.md](bcu1525_wg.md).

## 5. Tests

`tests/test_native_dram.py` (the adapter against a reference at NCH 2 / 8 and CB 64 / 128, the
ordering rules, mutation hooks `OTPU_MUT_*`), `tests/test_wg_rtl.py` (port W into the MXU, posted
lanes with stalls), `tests/test_board.py` and `tests/test_wg_board.py` (the board model).
