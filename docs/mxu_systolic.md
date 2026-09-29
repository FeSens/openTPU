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
  stream with a per-command constant (`hi0[j]`) in the DSP48E1 itself (`otpu_pe`: A = low, D =
  high, INMODE selects one through the pre-adder), so there is no fabric mux per position and
  column. `hi0[j]` drives the INMODE of the column's 128 DSPs (a 128 fan-out, the OOC's worst path
  at MCOLS 4: 152 MHz); it only changes between commands whose rows do not overlap.
- **Activations** stay per column (ACT RAM rows). Their delay, 1 (the decode register) + k (chain
  stage) + j (column), is at most 1 + 15 + 11 = 27 cycles for MCOLS <= 12: one SRLC32E per bit,
  the same cost as IMPL=1's chain skew alone. The column delay is free.
- **One product per DSP.** Two int8 products per DSP48E1 is not exact on a cascade (the study:
  the low field overflows after 3 accumulations), so each column has 128 product DSPs, 4 sub-block
  multipliers and the epilogue's fp multipliers: 138 DSP per column, 1,106 at MCOLS = 8 of the
  part's 1920 (measured; the rest of the design uses ~150).
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
route). IMPL 2 removes it:

- **Row-gated pops.** A row's first block pops only when all of the row's chunks and scales are
  in the FIFOs (the FIFO holds a whole row already: replay requires KB <= DEPTH). A row then never
  stalls in its middle, `en_c` is constant 1, and the compute pipeline (array and epilogue) has no
  clock enable at all: a cycle without a pop is a bubble (an invalid slot), which the epilogue
  already handles between rows. The simulation checks that no row stalls.
- **Commands overlap.** The consumer used to run one command at a time: the next command's pops
  waited for the previous command's full drain (pipeline + epilogue + drain), so every MM paid the
  pipeline's latency, and IMPL 2's longer dot product (MCOLS + 19 cycles against 7) made that 16-20
  cycles more per MM. Now there are two heads: the drain head `q_h` (the oldest command: its
  results, its ACC / ASCALE / RMAX state, its completion) and the pop head `q_p`, which moves to
  the next command as soon as the head's chunks have all popped, provided the pipeline treats both
  alike (the same PAIR and, under PAIR, the same M: `pr0`, `hi0` and `M0` are the only per-command
  state read after S0). The next command's rows follow the head's through the pipeline; the drain
  takes the head's rows (counted in `rows_live`, the pop head's in `rows_p`) and then moves on.
  IMPL 0 / 1 keep one head (`q_p = q_h`, compiled out).
- **The result FIFO** is 64 rows for IMPL 2 (32 before): a row pops only while fewer than RF rows
  are between the pop and the drain, and one-block rows (attention scores, K = 128) fill the longer
  pipeline one per cycle; 32 blocked the pops (MLP decode +18%, LFM2 decode +4.8%).
- The drain keeps its own enable (the TMEM grant); it is outside the compute pipeline.

## Measured: area and fmax of `otpu_mxu` alone (Vivado 2026.1 OOC, synthesis + P&R)

xc7k480t-2, `otpu_mxu` out of context at a 6.667 ns clock (150 MHz), D = 128, board FIFO depth.
IMPL 0 is the tree with the DSP-cascade pair sums (`49ed2e0`); "use_dsp" is `465d629` (Vivado
infers the product DSPs); "DSP48E1" is `0cb4a7d` (`otpu_pe` instances, the stream select in the
pre-adder), the version on this branch.

| MXU | MCOLS | LUT | of which SRL | FF | slices | DSP | WNS | fmax |
|---|---|---|---|---|---|---|---|---|
| IMPL 0 (tree) | 4 | 26,389 | 163 | 15,504 | 9,479 | 282 | +0.153 | 153.5 MHz |
| IMPL 2, use_dsp | 4 | 24,426 | 2,651 | 31,540 | 12,232 | 554 | +0.332 | 157.9 MHz |
| IMPL 2, use_dsp | 8 | 47,822 | 5,440 | 60,926 | 23,706 | 1,106 | +0.045 | 151.0 MHz |
| IMPL 2, DSP48E1 | 4 | 22,504 | 2,699 | 34,894 | 12,298 | 554 | +0.109 | 152.5 MHz |
| IMPL 2, DSP48E1 | 8 | 44,779 | 5,304 | 67,997 | 24,214 | 1,106 | +0.068 | 151.5 MHz |

Per column (MCOLS 4 -> 8, DSP48E1): +5.6K LUT, +8.3K FF, +2.5K slices, +138 DSP. IMPL 0 per
column (synthesis, MCOLS 2 -> 4): +7.3K LUT, +3.4K FF, +70 DSP; in the MCOLS 2 -> 4 full builds
+7.8K LUT and +5.1K FF per column.

What this says, and what it does not:

- **LUTs go down** (-15% at MCOLS 4 against the tree; ~1.2K LUT less per column than the tree's
  synthesis). **Flip-flops more than double** (the weight hops, 2 x 8 bits per position and column,
  and the skew registers) and **DSPs double** (one product per DSP: 1,106 at MCOLS 8, ~1,250 with
  the rest of the design, 65% of the part; MCOLS 12 ~1,800, 94%).
- **OOC slices are higher** (12.3K against 9.5K at MCOLS 4). OOC P&R does not pack densely (the
  tree's 26.4K LUT would fit in 6.6K slices, the systolic array's 22.5K LUT / 34.9K FF in ~5.6K),
  so this is not the in-context density, but it is not the slice saving that would settle the
  MCOLS = 4 density cliff on its own: the full build has to show it.
- **Standalone fmax does not separate them**: the tree also makes 153 MHz alone. Its MCOLS = 4
  failure is in context (97% of the slices, route-dominated broadcast and control paths); the
  systolic array has no broadcast (fan-out 2 per hop), no compute clock enable and no fabric
  adder tree, which is what the full build tests.
- The worst paths alone: IMPL 0, the pair-sum pack register into the DSP C port; IMPL 2 at
  MCOLS 4, `hi0[j]` into the INMODE of its column's 128 DSPs (fan-out; a register per chain would
  remove it); at MCOLS 8, the weight skew SRL output into a DSP's A port (route).

## Measured: cycles (perf_qwen, board configuration, DDR3-1066, 120.755 MHz, fp4 weights)

`tools/perf_qwen.py --layers 0 --pos 256 --ddr 1066 --mhz 120.755 --wformat fp4 --head-format
int8 --bl 32 --wbl 8 --check` with PAIR and DSTEP; every run bit-exact with the ISA simulator.
Decode is one token at position 256; prefill is 6 rows (positions 256..261, `--rows 6 --logits
none`).

| model | IMPL 0, MCOLS 2 | IMPL 0, MCOLS 4 | IMPL 2, MCOLS 4 | IMPL 2, MCOLS 8 |
|---|---|---|---|---|
| LFM2 decode | 1,319,029 | 1,319,029 | 1,320,453 (+0.11%) | 1,320,668 (+0.12%) |
| Qwen3.5 decode | 4,848,785 | 4,832,273 | 4,832,035 (-0.35%) | 4,832,472 (-0.34%) |
| Qwen3 decode | | | 3,541,285 | |
| LFM2 prefill, 6 rows | | | 2,949,361 | 1,736,596 (-41%) |
| Qwen3.5 prefill, 6 rows | | 12,059,908 | 11,998,949 (-0.5%) | 9,323,037 (-22%) |
| Qwen3 prefill, 6 rows | | | 8,979,279 | 5,837,044 (-35%) |

(Percentages: decode against IMPL 0 at MCOLS 2; prefill at MCOLS 8 against IMPL 2 at MCOLS 4.)

Decode does not regress (the gate: <= 1%). The overlap of commands pays IMPL 2's longer pipeline.

**The next bottleneck at MCOLS 8.** LFM2's prefill is still MXU-bound (MXU busy 95%, MAC 80.6% of
the cycles: 6 rows leave 2 of 8 columns idle). Qwen3.5's is not: DeltaNet is 65% of the cycles,
the DSTEP head steps (serial per token, 3.54M cycles of work in 9.32M, the same at any MCOLS) set
the pace, and the MXU waits 5.3M cycles in gaps after the QACT flush (`qwen35.py:1029`) and the
state window stores (`qwen35.py:1104`) in front of them. More columns do not help there; the
stream engine's state step does. Qwen3 sits between the two: MXU busy 90%, MAC 73.8%, but 9.7% of the
cycles the chunk FIFO is empty while streaming (the weight stream, not the array), and attention
is 63% of the cycles with the longest MXU gap (0.61M cycles) after the attention setup's QACT
(`attention.py:42`).

## Plan and status

1. Done: `IMPL=2` dot product with the unchanged epilogue, bit-exact on the MM / 4-bit / PAIR RTL
   tests, the MM fuzz and test_perf at MCOLS 4 and 8, and the full models (`perf_qwen --check`).
2. Done: row-gated pops (`en_c` = 1), command overlap, the 64-row result FIFO; cycles above.
3. Done: OOC area and fmax at MCOLS 4 and 8 (above); MCOLS 12 queued.
4. Done: the PAIR select in the DSP pre-adder (`otpu_pe`).
5. The combined build (branch `se-sys`: the stream engine v2 + `MXU_IMPL=2`, MCOLS 4). `make bit`
   selects the MXU through `create_project.tcl`'s MXU argument (systolic, the default on this
   branch, or tree). MCOLS 8 after the LiteDRAM memory path frees its area.
6. Later: a registered copy of `hi0` per chain (the INMODE fan-out); the fp epilogue's area (now
   the MXU's largest LUT consumer per column).
