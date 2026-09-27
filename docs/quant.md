# 4-bit weights

Decode is DRAM-bound: every token streams every weight once. Halving the weight bytes is the
largest single speedup available to this machine, so we looked at 4-bit weight formats, chose
one, and built it into the ISA, the simulator, the compiler and the MXU. 8-bit weights work as
before; 4-bit is an extra mode of the `MM` instruction.

In short:

- **Format.** `fp4`: E2M1 (FP4) elements with a two-level scale per 128 elements, a bf16 block
  scale times an unsigned 4-bit multiplier per 32 (4.25 bits per weight). On the three models
  it is about as accurate as NVFP4 (4.5 bits) and clearly better than MXFP4 (4.25 bits).
- **Hardware.** The MXU reads a 128-byte chunk as two 128-element blocks and consumes them in
  two cycles, at the same MAC width as int8. The scale word goes through the existing scale
  stream (one 32-bit word per block, as for int8). The multipliers are applied to exact integer
  sub-block sums, so the arithmetic stays bit exact. Cost in yosys: +2.2K LUTs, +8 DSPs, +1
  BRAM36 on the MXU (of 299K LUTs and 1,920 DSPs on the xc7k480t).
- **Speed (simulated).** Qwen3-0.6B decode on the RTL of the board configuration: 1.89x fewer
  cycles per token at a DRAM rate close to what the card delivers today (the model at 25% of
  peak bandwidth), 1.27x at 80% and 1.02x at 100%. At high DRAM rates the MXU's one block per
  cycle becomes the limit (21.5 tokens/s at an assumed 100 MHz).
- **Accuracy cost.** On these small models (0.2 to 0.8 B parameters) 4-bit weights cost
  noticeably more than int8: on Qwen3-0.6B perplexity goes from 23.5 to 28.3 on our book sample
  (int8: 23.5), and the next-token distribution moves by a mean KL of 0.19 nats (int8: 0.007).
  Keeping the LM head or the attention projections in int8 recovers part of it (below).

Everything here is measured in simulation or estimated with yosys. Nothing has run on the card.

## Formats

All formats quantize blocks of consecutive elements of a weight row (the K dimension the MXU
reduces over). "bits" counts element bits plus the block scales; a per-tensor scale is ignored.

| name | elements | block | scale | bits/w | notes |
|---|---|---:|---|---:|---|
| `int8` | int8, [-127, 127] | 128 | fp32 | 8.25 | today's format (the baseline) |
| `int4-gG-S` | int4, [-7, 7] | G = 32, 64, 128 | fp32, fp16, e4m3, e8m0 | 4.25 - 5 | symmetric |
| `e2m1-g128-fp32` | E2M1 | 128 | fp32 | 4.25 | FP4 with today's scale stream |
| `mxfp4` | E2M1 | 32 | E8M0 (power of two) | 4.25 | OCP Microscaling spec; `-ceil`: a scale that never clips |
| `nvfp4` | E2M1 | 16 | E4M3, x fp32 per tensor | 4.5 | NVIDIA's format |
| `int4k`, `e2m1k` | int4 / E2M1 | 128, sub-blocks of 32 | bf16 x u4 in 1..15 | 4.25 | two-level, one 32-bit word per 128 |

E2M1 has the values 0, 0.5, 1, 1.5, 2, 3, 4, 6 and a sign. Rounding is to nearest with ties to
the even code, saturating (opentpu/quant.py). `-s` means the block scale is chosen by
minimizing the block's squared error over a small grid of candidates instead of taking
amax / max-element (for the two-level formats: the best multiplier of each sub-block). This is
the cheap relative of GPTQ/AWQ-style error minimization; those use activation statistics,
which we did not try.

The two-level format is our addition. Like k-quants in llama.cpp, it spends the scale bits on a
coarse float scale plus small integer multipliers, so four 32-element sub-blocks get their own
scale for 32 bits per 128 elements, the same cost as MXFP4 and as today's fp32 scale per 128.

## Accuracy

`tools/quant_eval.py` runs each Hugging Face model in fp32 (PyTorch, CPU) with every weight the
MXU streams (every decoder `nn.Linear` whose K is a multiple of 128, and the LM head) replaced
by its quantize-dequantize copy. Every one of these matmuls also gets its input fake-quantized
to int8 per 128 elements, as `QACT` does on the device. Embeddings stay fp32 (the host gathers
them). The reference is the unquantized fp32 model. Measured on:

- two texts of 640 tokens: the opening of *Pride and Prejudice* (public domain,
  `tools/data/austen_pp_ch1.txt`) and the prose of `docs/isa.md`, which is less likely to be in
  the training data. Perplexity is measured on each text; "top-1 text" is the argmax
  agreement with the reference on both texts;
- the reference's greedy continuations of 8 prompts, 40 tokens each ("top-1 greedy", 320
  positions, so differences below about 0.04 are noise);
- KL: the mean KL(reference || quantized) of the next-token distributions over all 1,600
  positions. This is the steadiest single number.

```
python3 tools/quant_eval.py --model qwen3 --json build/quant/qwen3.json     # ~1 h on the Mac
python3 tools/quant_eval.py --report build/quant/qwen3.json                  # the table below
```

"+head8" keeps the LM head in int8; "+ends8" also the first and last layer; "+down8" the head
and the MLP down projections; "+attn8" the head and every attention (Qwen3.5: DeltaNet, LFM2:
convolution) projection. MB is the weight bytes streamed per token.

QWEN3_TABLE

LFM2_TABLE

QWEN35_TABLE

What the tables say:

- **Smaller blocks help most.** On Qwen3, int4 goes from a KL of 0.44 with a scale per 128 to
  0.36 per 64 and 0.28 per 32; FP4 from 0.30 per 128 to 0.19 per 16 (NVFP4).
- **FP4 elements beat int4 elements** at the same block and scale (`e2m1k` against `int4k`,
  `e2m1-g128` against `int4-g128`). Weights are roughly bell-shaped, and E2M1's uneven grid
  spends its codes near zero where the weights are.
- **MXFP4's power-of-two scale is its weak point.** With int4 elements an E8M0 scale is by far
  the worst option (the scale can be up to 2x too large). MXFP4's E2M1 elements soften this,
  but it stays behind NVFP4 and the two-level formats on every model. Rounding the scale up so
  that it never clips (`mxfp4-ceil`) is worse than the spec's rule; the error search helps a
  little.
- **NVFP4 and `e2m1k-s` are the most accurate**, NVFP4 slightly ahead at 4.5 bits and
  `e2m1k-s` at 4.25 bits.
- **Error-minimizing scales** lower the KL of every format (by 0.004 to 0.07 on Qwen3). They
  help least for MXFP4, whose power-of-two scale leaves little to choose, and they do not
  always lower the perplexity of one text.
- **Mixed precision** buys accuracy with bytes. On Qwen3-0.6B the tied LM head is a quarter of
  all weights, so "+head8" costs 25% more bytes for a modest gain. Per extra bit, "+attn8" and
  "+ends8" gain the most. The Engine supports one format for the layers and one for the LM
  head; per-matrix formats ("+attn8", "+down8") would be a small change to the images, and
  per-layer formats ("+ends8") would split the hardware layer loop.

Recommendation: `e2m1k-s`, called `fp4` in the code: nearly NVFP4's accuracy at MXFP4's size,
and (next section) the cheapest of the accurate formats to build. It is what `wformat="fp4"`
builds. Whether to keep the LM head in int8 is a speed/quality choice per model.

## Hardware options

The MXU is D = 128 deep and MCOLS columns wide. Each cycle it consumes one 128-byte weight
chunk from the DRAM stream (one int8 block) and one fp32 scale from the scale stream, and
reuses the block across MCOLS stationary rows. A 4-bit chunk holds 256 weights. We compared
three datapaths, all for the two-level format, with yosys (`tools/synth`: `synth_xilinx`, Xilinx
cell delays, no routing; the fmax is a rough estimate, T = 1.6 x logic + 0.5 ns):

| option | what | LUT | FF | DSP | BRAM36 | logic ns | decode speedup, simulated (bw 25 / 80 / 100%) |
|---|---|---:|---:|---:|---:|---:|---|
| baseline | int8 MXU (main) | 10,055 | 4,232 | 142 | 29.5 | 4.39 | 1 |
| **(a) half rate** | 2 cycles per 4-bit chunk, one block each | 12,222 | 5,484 | 150 | 30.5 | 4.67 | **1.89 / 1.27 / 1.02** |
| (b) full rate, DSP | (a) + a second block per cycle | +6,130 | +3,019 | +144 | | | about 1.9 at each (roofline) |
| (c) full rate, LUT | (b) with the second block's products in LUTs | +17,905 | +9,542 | +16 | | | same as (b) |

(a) is built, verified and measured. The (b) and (c) rows are the extra datapath alone (a
synthesis proxy, `tools/synth/proxy_mxu_x2.sv`: nibble decode, MCOLS x 128 products, the tree,
i2f, the two fp multiplies and one fp add per column), without the control and the second ACT
RAM read port that a real unit would also need. Their speedups are the DRAM roofline of the
bytes streamed, not simulations.

- **(a)** reuses everything. The chunk FIFO's registered read feeds a nibble decoder (the half
  and the element type select 4 bits per lane; E2M1 decodes to twice its value, an integer in
  {0, 1, 2, 3, 4, 6, 8, 12}), which is registered before the multipliers (without that
  register the decode sits in front of the DSPs: 5.1 ns of logic in yosys). The dot product
  keeps its exact integer tree; the sums of the four 32-element sub-blocks are multiplied by
  their 4-bit multipliers (eight small multipliers, which yosys maps to DSPs) in the stage that
  used to form the block sum, and the block sum moves one stage later; it is still exact
  (|sum| < 2^22). Then i2f, x scale, x activation scale and the accumulation are unchanged. The
  issuer requests a chunk every second block and a scale word every block; the scale FIFO is
  twice as deep (two words per chunk in flight). The logic depth grows from 4.39 to 4.67 ns (the
  sub-block sum in front of its multiplier; est. 133 -> 126 MHz, the board runs at 100). The MXU
  pipeline is two stages longer, also for int8: QWEN3_LAT on a two-layer Qwen3 token, +64 cycles
  (0.25%) on the small MLP kernels of tests/test_perf.py.
- **(b)** doubles the MXU's rate for 4-bit weights: two blocks per cycle, so a 4-bit chunk per
  cycle. It needs a second ACT RAM block per cycle (even/odd banks), 128 more DSPs for the
  products at MCOLS = 2, and a different accumulation: two terms per column per cycle cannot
  go through the 4-partial adder loop in order, so the ISA would sum the pair first. It only
  pays when DRAM delivers more than 64 bytes per cycle.
- **(c)** avoids the DSPs (E2M1 x int8 is a shift and add: a, 3a, shifted), but 256 LUT
  products cost three times the fabric of (b), and it only works for E2M1 elements.

Winner: **(a)**. The card's DRAM delivers about 30 bytes per cycle today (20.7 M cycles per
Qwen3 token measured; the simulator at 25% bandwidth, 32 bytes per cycle, gives 19.8 M), well
below the 64 bytes per cycle at which (a) saturates. There (a) gets the whole 2x of the bytes
for 2K LUTs. (b) is the upgrade once the DRAM path delivers more than half its peak; at 80% it
would take 4-bit decode from 1.27x to about 1.9x.

Why not the other scale formats in hardware:

- MXFP4 needs a power-of-two scale per 32: the four sub-block sums would be aligned by
  shifts of up to 254 bits. Exact, it needs a wide adder and i2f; approximate, it needs its
  own rounding rule. It is also the least accurate.
- NVFP4 needs 8 E4M3 scales per 128 (two scale words per block, twice the scale stream the
  MXU can fetch in one cycle) and a per-tensor fp32 scale the MM has no field for.
- The two-level scale uses the existing stream unchanged and keeps the integer tree exact.

## ISA and DRAM layout

`MM` flags bits 5:4 are the weight format `WF`: 0 int8, 1 int4, 2 E2M1 (docs/isa.md, "Weight
formats"). A 4-bit row stores block k at `sa + n*rs + k*64` bytes (two blocks per 128-byte
chunk, the element 2i in the low nibble); the scale word of block k is where the fp32 scale of an
int8 row would be, `ssa + n*srs + 4k`: bits 15:0 the bf16 scale, bits 16+4b the multiplier of
sub-block b. Rows are chunk aligned, so a column slice of a 4-bit matrix must start at an even
block (the compiler checks this; W_down is chunked in multiples of 256 columns). Activations
(`QACT`), the KV cache and `QST` stay int8: this is W4A8.

## Using it

```python
from opentpu.llm.qwen3 import Engine, Spec, load_weights
eng = Engine(spec, W, wformat="fp4")                       # every weight 4-bit
eng = Engine(spec, W, wformat="fp4", head_format="int8")   # LM head int8
```

`wformat` is `"int8"` (default), `"fp4"` or `"int4"`; the Qwen3, LFM2 and Qwen3.5 images all
take it. For kernels, `runtime.Weight(w, shard, fmt="fp4")`. `opentpu/quant.py` has the
quantizers (`quantize_w4`, `quantize_mxu`) and the reference formats of the survey.
`tools/bench_llm.py` and `tools/perf_qwen.py` take `--wformat` and `--head-format`.

## Measured speed (simulated)

Qwen3-0.6B decode, one sequence at context 128, from `tools/bench_llm.py --batches 1 --ctx 128`
(the RTL of the board configuration, MCOLS = 2, the AXI memory path at `bw` percent of peak
bandwidth; random weights, since the timing does not depend on them). Cycles per token; tokens
per second at an assumed 100 MHz, without host time.

| bw | int8 | fp4 | fp4, int8 LM head |
|---:|---:|---:|---:|
| 25% | 19,766,594 (5.06 tok/s) | 10,453,874 (9.57 tok/s, 1.89x) | 12,886,087 (7.76 tok/s, 1.53x) |
| 80% | 6,201,475 (16.13 tok/s) | 4,890,475 (20.45 tok/s, 1.27x) | 5,247,960 (19.06 tok/s, 1.18x) |
| 100% | 4,978,229 (20.09 tok/s) | 4,888,951 (20.45 tok/s, 1.02x) | 4,932,311 (20.27 tok/s, 1.01x) |

At 25% the 4-bit token is DRAM-bound (97% of the DRAM roofline); at 80% and 100% it runs at the
MXU's one block per cycle (95% of that bound). On the card, where the int8 token takes 20.7 M
cycles, we expect about the 25% column; this is a projection until it runs there.

## Tests

- `tests/test_quant.py`: the rounding rules, bits per weight, pack/unpack round trips, the 4-bit
  `MM` on the ISA simulator against float64 math on the dequantized weights (int4 and FP4, odd
  and even block counts, UNIT and ACC), the MLP kernel at 4 bits.
- `tests/test_rtl.py`: the random-program fuzzers and the scoreboard stress tests now draw
  int4 and FP4 MMs too (also on the AXI memory path with random stalls), and a 4-bit MLP at
  D = 32 and D = 128. RTL and ISA simulator agree bit for bit.
- `tests/test_qwen3.py`: a tiny Qwen3 at 4 bits follows its float64 emulation; Qwen3-0.6B with
  FP4 layers and an int8 head answers the France question correctly on the ISA simulator, each
  token the argmax of the emulation; and one Qwen3-0.6B FP4 token on the RTL is bit exact
  against the ISA simulator (weights, KV cache, logits).

## Not done

- No run on the card; the speeds above are simulated.
- The MXU's DSP cascade variant (`IMPL = 1`, a timing study option) does not support 4-bit
  weights; the simulation stops if it meets one.
- Option (b), for when the DRAM path delivers more than half of its peak.
- Calibration-based quantization (GPTQ, AWQ) and importing MXFP4 / NVFP4 checkpoints. An MXFP4
  block converts exactly only when its four scales span at most 2^3, and NVFP4 not in general.
