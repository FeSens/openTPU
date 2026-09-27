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
| 1300 (out of spec) | 650 MHz | 162.5 MHz | 20.8 GB/s | 14.6-17.7 GB/s | 12.8 GB/s |
| 1333 (out of spec) | 667 MHz | 167 MHz | 21.3 GB/s | 14.9-18.1 GB/s | 12.8 GB/s |
| 1600 (out of spec) | 800 MHz | 200 MHz | 25.6 GB/s | 17.9-21.8 GB/s | 12.8 GB/s |

At DDR3-800 the channels deliver less than the core can consume, because refresh, row misses and
ECC read-modify-write take their share. From DDR3-1066 up, the controllers can keep the 128-byte
path full, and the core becomes the limit.

On this board MIG supports DDR3 up to 1066. The DDR3 is on HR banks, and MIG warns (79-155)
that 1300, 1333 and 1600 are not supported, though it still generates them. 1333 and 1600 also
need MIG's PHY patched (docs/board.md, "Faster DDR3"). The rows from 1300 up are out-of-spec
operation that the card may or may not sustain.

`docs/benchmarks.md` runs Qwen3-0.6B at batch 1 and ctx 128 in simulation. At 80 % of the
128-byte peak it decodes 16.1 tok/s; at 100 % it decodes 20.0 tok/s (*simulated*). So DDR3-1066
or 1300 with today's RTL is worth up to about +24 % on decode (*projection*), provided the card
reaches the simulated rate. It does not yet: first light ran at 4.8 tok/s, with the MXU starved
by single-beat reads, and the burst-read change on r4-burst targets that.

## What a 256-byte path needs

The goal is 256 bytes per core cycle at 100 MHz: 25.6 GB/s, which matches DDR3-1600's peak.
If the card tops out at 1300 or 1333, the DRAM limits the path to about 15-18 GB/s
(*estimate*).

### 1. AXI and interconnect

- Each channel must accept 128 bytes per core cycle. That takes either one 1024-bit AXI port per
  channel or two 512-bit ports. One 1024-bit port is simpler, since SmartConnect supports
  1024-bit data. SmartConnect then downsizes to the MIG's 512 bits at ui_clk. At 1600
  (200 MHz) the two sides carry the same bytes per second. At lower speeds the MIG side is
  narrower, which is expected, since the DRAM is then the limit.
- The channel interleave changes. A 256-byte chunk becomes one 128-byte beat per channel, and
  the host's address map in `opentpu/host/board.py` must follow. A cheaper option keeps the
  64-byte interleave and issues two beats per channel per cycle, but that doubles the request
  rate in `otpu_axi_dram`.
- Read data in flight doubles for the same latency. Today RD = 128 beats × 64 B = 8 KB per
  channel; the wide path needs 16 KB per channel, about +8 BRAM36 per channel (*estimate*).
- The MIG's 512-bit ECC path and SmartConnect's MIG-side logic must close timing at the faster
  ui_clk (200 MHz at 1600). The out-of-spec builds on the `ddr` branch measure this
  (docs/board.md).

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

At MCOLS = 4 the board would reach about 223K LUT (75 %, *estimate*). That is close to where
routing already struggles: LANES = 16 did not route at 212K placed LUT.

### 3. How this pairs with 4-bit weights

A 256-wide MXU is the same hardware that 4-bit weights need. A 128-byte chunk of 4-bit weights
carries 256 weights, so the fp4 branch's formats fill 256 multiply-accumulates per cycle from
today's 128-byte path. The weight decode is a 16-entry table per nibble to an int8 code, in front
of the products. One widened MXU serves both uses:

| weights | DRAM path | DDR3 | weights per cycle | decode vs today (*projection*) |
|---|---|---|---|---|
| int8 | 128 B/cycle (today) | 1066 or 1300 to fill it | 128 | up to 1.24x (the 80 % → 100 % bw gain) |
| 4-bit | 128 B/cycle | 1066 or 1300 to fill it | 256 | ~2x, up to ~2.5x |
| int8 | 256 B/cycle | 1300 | 256 | ~1.4x (~16 GB/s against ~10 GB/s today) |
| int8 | 256 B/cycle | 1600 (out of spec) | 256 | ~2x |

These projections assume decode stays DRAM-bound and that the per-token vector work (VPU,
attention) does not grow with it. At batch 1 that work is a small share of the cycles, which is
what `docs/benchmarks.md` implies.

### 4. What to build first

4-bit weights on today's 128-byte path give the largest gain, and they need only the MXU
widening. The AXI widening adds ~1.4x for int8 at DDR3-1300 / 1333 and ~2x at 1600, but only
if the card runs that speed reliably, out of spec.

## Suggested order

1. Measure DDR3-1066, 1300, 1333 and 1600 calibration and bandwidth on the card; the checklist is in
   docs/board.md. Neither needs an RTL change.
2. Get the 128-byte path to its simulated rate on the card: burst reads (r4-burst), then
   MXU_STARVE near zero.
3. Widen the MXU to 256 products per column, and use it first with 4-bit weights on the
   existing DRAM path.
4. Widen the AXI ports to 1024 bits for int8 weights, if 1333 or 1600 holds up on the card.
