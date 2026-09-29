# A 2D systolic MXU (`MXU_IMPL=2`)

Goal: an MXU that scales to MCOLS = 8 (4 and 12 must also build) on the xc7k480t without the
broadcast fan-outs and the fabric adder tree that make MCOLS = 4 fail timing today
([mxu_study.md](mxu_study.md); the MCOLS = 4 builds of 2026-09-28: fmax 101-119 MHz at 97% of the
slices, worst paths route-dominated). Results must stay bit-identical to the ISA simulator, and
decode must not lose cycles.

## What changes, what stays

The MXU is a D x MCOLS dot-product engine: each advance, one D = 128-byte weight chunk (one K-block
of one output row, or two 4-bit blocks under PAIR) meets MCOLS activation blocks (ACT RAM, one per
column = one activation row), and each column reduces its 128 int8 x int8 products exactly, then
runs a per-column fp32 epilogue. `IMPL=0` broadcasts the decoded chunk to every column and sums in
a fabric tree; `IMPL=1` (study) sums along D in DSP cascades but still broadcasts the (skewed)
weights, and supports neither 4-bit weights nor PAIR.

`IMPL=2` keeps the MXU's front and back unchanged:

- the issuer, the chunk / scale FIFOs, the pop logic, the ACT RAM interface;
- the fp32 epilogue per column, in the ISA's order: i2f (exact) -> x weight scale -> x activation
  scale -> [PAIR: + partner column] -> isum_4 partial loop -> (p0+p2)+(p1+p3) -> result FIFO ->
  drain (ACC / ASCALE read-modify-write, RMAX). Integer sums are order-free; every fp op here is
  not, so the epilogue is not touched.

and replaces the dot product (operands -> s4, the exact integer block sum) with a systolic array.

## The array

```
            column 0            column 1                 column MCOLS-1
w chunk -> decode -> skew -> [hop] -> ... -> [hop] -> ... -> [hop]
                      |          |                        |
position i (chain g = i/16, stage k = i%16), in each column:
    act[j][i] (ACT RAM, delayed 1 + k + j: one SRLC32E per bit)
    M  = act * w          (DSP48E1 M register, one product per DSP)
    P  = PCIN + M         (stage k > 0; stage 0: P = M)  -> the chain's running sum
chain ends (8 per column): sub-block b = chains 2b, 2b+1 (32 positions, the 4-bit sub-block)
    v[b] = (chain 2b + chain 2b+1) * m_b   (DSP pre-adder + multiplier, as IMPL=0)
    s4   = v0 + v1 + v2 + v3               (small fabric adder, registered)
deskew: column j's s4 delayed MCOLS-1-j cycles -> all columns aligned for the epilogue
```

- **Weights flow, not broadcast.** The chunk is decoded once (int8; int4 / E2M1 nibbles -> int8),
  skewed once per chain stage (shared SRLs, delay k), and then moves one register hop per column:
  column j sees it j cycles after column 0. Each hop register drives one DSP per position and the
  next hop: fan-out 2, not MCOLS.
- **PAIR (4-bit, 2M <= MCOLS):** columns j < M take the chunk's low block, columns j >= M its high
  block. Both decoded streams flow (2 x 8 bits per position per hop) and each column selects its
  stream with a per-command constant (`hi0[j]`). Phase 1 selects in fabric (a 2:1 mux per position
  per column); the DSP pre-adder can do it for free later (A = low, D = high, INMODE per column).
- **Activations** stay per column (ACT RAM rows). Their delay, 1 (the decode register) + k (chain
  stage) + j (column), is at most 1 + 15 + 11 = 27 cycles for MCOLS <= 12: one SRLC32E per bit,
  the same cost as IMPL=1's chain skew alone. The column delay is free.
- **One product per DSP.** Two int8 products per DSP48E1 is not exact on a cascade (the study:
  the low field overflows after 3 accumulations), so each column has 128 product DSPs, 4 sub-block
  multipliers and the epilogue's fp multipliers: ~136 DSP per column, ~1090 at MCOLS = 8 of the
  part's 1920 (the rest of the slice uses ~200).
- **The 4-bit sub-blocks** are 32 positions = two 16-stage chains, so the sub-block multipliers
  m_b apply at the chain ends, exactly as IMPL=0 applies them to its group sums. int8 uses
  m_b = 1. Chain length is fixed at 16 (`CL`).
- **Latency** S0 -> aligned s4: 1 (decode) + 16 (chain) + 1 (product) + 2 (v, s4) + MCOLS - 1
  (column skew, then deskew) = MCOLS + 19 cycles (23 at MCOLS 4, 27 at 8), against 7 for IMPL=0.
  It is paid once per MM command's drain (the study measured +17 cycles per MM chain for
  IMPL=1: negligible).

## Control travels with the data

Today every compute-pipeline register has the clock enable `en_c` (the pipeline freezes when a
row's next chunk has not arrived). Its fan-out and the control logic in front of it are the MXU's
worst paths on the board (the MXU `q_h` -> `mk` CE / tree / combine CE paths: 15-16 levels, ~88%
route).

- The per-chunk meta (valid, first, last, partial index, 4-bit half) already travels with the data
  (`cm_t`), and the epilogue already treats invalid slots (bubbles) between rows.
- **Row-gated pops (phase 2):** a row's first block pops only when all of the row's chunks and
  scales are in the FIFOs (the FIFO holds a whole row already: replay requires KB <= DEPTH). A row
  then never stalls in its middle, `en_c` is constant 1, and the compute pipeline (array and
  epilogue) has no clock enable at all: bubbles are invalid slots. The only cost is when the
  weight stream is slower than the MXU (DRAM-bound decode): a row starts once complete instead of
  chunk by chunk, so the last row of a command finishes up to one row (KB advances) later.
- The drain keeps its own enable (the TMEM grant); it is outside the compute pipeline.

## Area expectation (per column, MCOLS 4, Vivado OOC of otpu_mxu)

IMPL=0 today (with the DSP-cascade pair sums, `49ed2e0`): ~7.9K LUT per column, of which the
integer side (weight decode per DSP pair ~0.9K, group-sum tree and sub-block sums ~1.9K, pair
sums ~0.3K) is ~3.1K and the fp epilogue ~4K. IMPL=2 replaces the integer side with ~1K SRL (the
activation skew), ~0.5K LUT (the PAIR stream select, until INMODE does it) and ~0.1K (s4), plus
~2K flip-flops of weight hops: about -1.5K LUT and fewer CARRY4s per column, and none of the
broadcast. The fp epilogue (~4K LUT per column) is then the MXU's largest part at MCOLS 8.

## Plan and status

1. `IMPL=2` dot product with the unchanged epilogue; bit-exact on the MM / 4-bit / PAIR RTL tests
   and the MM fuzz at MCOLS 4 and 8.
2. Row-gated pops, `en_c` = 1 for `IMPL=2`; cycles per model at MCOLS 4 and 8.
3. OOC area and fmax of `otpu_mxu` alone at MCOLS 4 and 8 (Vivado, omarchy).
4. Later: PAIR select in the DSP pre-adder (INMODE); the fp epilogue's area.
