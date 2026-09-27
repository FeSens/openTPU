# Design note: using faster DDR3 (a 256-byte-per-cycle weight path)

Status: a proposal, not implemented. Every number marked *estimate* or *projection* is
arithmetic or a simulation of the RTL, not a measurement on the card.

## Where the bandwidth goes today

The MXU consumes one D = 128-byte weight chunk per core cycle. `otpu_axi_dram` splits it into
one 64-byte beat per DDR3 channel, and each channel's accelerator port on the SmartConnect is
512 bits wide at core_clk. At 100 MHz that is 12.8 GB/s, whatever the DDR3 speed.

| DDR3 | CK | MIG ui_clk | peak, both channels | at 70-85 % efficiency (*estimate*) | what the core can take |
|---|---|---|---|---|---|
| 800 (default) | 400 MHz | 100 MHz | 12.8 GB/s | 9.0-10.9 GB/s | 12.8 GB/s |
| 1066 | 533 MHz | 133 MHz | 17.1 GB/s | 11.9-14.5 GB/s | 12.8 GB/s |
| 1333 | 667 MHz | 167 MHz | 21.3 GB/s | 14.9-18.1 GB/s | 12.8 GB/s |
| 1600 | 800 MHz | 200 MHz | 25.6 GB/s | 17.9-21.8 GB/s | 12.8 GB/s |

At DDR3-800 the channels deliver less than the core can consume, because refresh, row misses and
ECC read-modify-write take their share. From DDR3-1066 up, the controllers can keep the 128-byte
path full, and the core becomes the limit.

The simulated benchmark shows what that step is worth. `docs/benchmarks.md` runs Qwen3-0.6B at
batch 1 and ctx 128. At 80 % of the 128-byte peak it decodes 16.1 tok/s, and at 100 % it decodes
20.0 tok/s (*simulated*). So DDR3-1066 or 1333 with today's RTL is worth up to about +24 % on
decode (*projection*), provided the card itself reaches the simulated rate. It does not yet:
first light ran at 4.8 tok/s, with the MXU starved by single-beat reads, and the burst-read
change on r4-burst targets that. DDR3-1600 adds nothing over 1333 until the core takes more than
128 bytes per cycle.

## What a 256-byte path needs

The goal is 256 bytes per core cycle at 100 MHz: 25.6 GB/s, which matches DDR3-1600's peak.

### 1. AXI and interconnect

- Each channel must accept 128 bytes per core cycle. That takes either one 1024-bit AXI port per
  channel or two 512-bit ports. One 1024-bit port is simpler, since SmartConnect supports
  1024-bit data. SmartConnect then downsizes 1024 bits at 100 MHz to the MIG's 512 bits at
  200 MHz: the same bytes per second, so the clock converter never throttles.
- The channel interleave changes. A 256-byte chunk becomes one 128-byte beat per channel, and
  the host's address map in `opentpu/host/board.py` must follow. A cheaper option keeps the
  64-byte interleave and issues two beats per channel per cycle, but that doubles the request
  rate in `otpu_axi_dram`.
- Read data in flight doubles for the same latency. Today RD = 128 beats × 64 B = 8 KB per
  channel; the wide path needs 16 KB per channel, about +8 BRAM36 per channel (*estimate*).
- SmartConnect's 200 MHz side and the MIG's 512-bit ECC path must close timing. The DDR3-1600
  build on the `ddr` branch measures this; see docs/board.md, "Faster DDR3".

### 2. MXU: 256 multiply-accumulates per column per cycle

A 256-byte chunk of int8 weights means 256 products per column per cycle instead of 128. The
quantization block stays at D = 128 bytes, so the MXU takes two blocks per cycle:

- two 128-wide dot products per column, each with its own i2f and weight-scale multiply;
- one extra fp32 add to pair the two blocks before the 4-deep partial-sum loop, so the loop's
  timing does not change;
- the ACT RAM read doubles from MCOLS × 1024 bits to MCOLS × 2048 bits per cycle.

Cost at MCOLS = 2, from `docs/mxu_study.md`'s per-column figures (*estimate*):

| | today | 256 B/cycle | board total after |
|---|---|---|---|
| DSP48 | 267 | ~530 | ~530 of 1920 (28 %) |
| MXU LUT | 14.3K | ~20K | ~194K of 298.6K (65 %) |
| ACT RAM BRAM36 | 33 | ~65 | ~690 of 955 with the AXI buffers (72 %) |

At MCOLS = 4 the board would reach about 223K LUT (75 %, *estimate*), close to where routing
already struggles (LANES = 16 did not route at 212K placed LUT).

### 3. How this pairs with 4-bit weights

A 256-wide MXU is the same hardware that 4-bit weights need. A 128-byte chunk of 4-bit weights
carries 256 weights, so the fp4 branch's formats fill 256 multiply-accumulates per cycle from
today's 128-byte, DDR3-800 path. The weight decode is a 16-entry table per nibble to an int8
code, in front of the products. So one widened MXU serves two uses:

| weights | DRAM path | DDR3 needed | weights per cycle | decode vs today (*projection*) |
|---|---|---|---|---|
| int8 | 128 B/cycle (today) | 1066+ to fill it | 128 | up to 1.24x (bw 80 % → 100 %) |
| 4-bit | 128 B/cycle | 1066+ to fill it | 256 | ~2x |
| int8 | 256 B/cycle | 1600 | 256 | ~2x (at ~80 % of 25.6 GB/s) |
| 4-bit | 256 B/cycle | 1600 | 512 | ~4x, but needs a 512-wide MXU (does not fit at MCOLS = 2 with today's LUT budget) |

The ~2x rows assume decode stays DRAM-bound and that the per-token vector work (VPU, attention)
does not grow with it. At batch 1 that work is a small share of the cycles, which is what
`docs/benchmarks.md` implies.

## Suggested order

1. Measure DDR3-1066 and 1333 calibration and bandwidth on the card; checklist in docs/board.md.
   These work with no RTL change.
2. Get the 128-byte path to its simulated rate on the card: burst reads (r4-burst), then
   MXU_STARVE near zero.
3. Widen the MXU to 256 products per column, and use it first with 4-bit weights on the
   existing DRAM path.
4. Widen the AXI ports to 1024 bits for int8 at DDR3-1600, if 1600 calibrates on the card.
