# 4-bit weights

Decode is DRAM-bound: every token streams every weight once. Halving the weight bytes is the
largest single speedup available to this machine, so we looked at 4-bit weight formats, chose
one, and built it into the ISA, the simulator, the compiler and the MXU. 8-bit weights work as
before; 4-bit is an extra mode of the `MM` instruction.

In short:

- **Format.** `fp4`: E2M1 (FP4) elements with a two-level scale per 128 elements, a bf16 block
  scale times an unsigned 4-bit multiplier per 32 (4.25 bits per weight). On the three models
  it comes close to NVFP4 (4.5 bits) and is clearly better than MXFP4 (4.25 bits): mean KL to
  the fp32 model 0.19 / 0.16 / 0.12 (Qwen3-0.6B / LFM2.5-230M / Qwen3.5-0.8B), NVFP4
  0.15 / 0.14 / 0.10, MXFP4 0.27 / 0.27 / 0.19.
- **Hardware.** The MXU reads a 128-byte chunk as two 128-element blocks. By default it
  consumes them in two cycles, at the same MAC width as int8. With column reuse (`MM PAIR`,
  decode and other MMs of at most MCOLS/2 rows) the idle MXU columns take the second block, so
  a 4-bit chunk goes through in one cycle: 256 MACs per cycle per row with the same 128 DSPs,
  twice today's rate. The multipliers are applied to exact integer sub-block sums, so the
  arithmetic stays bit exact. Cost on the MXU, Vivado synthesis (out of context, xc7k480t):
  column reuse is +4.6K LUTs, +1.2K FFs, +2 BRAM tiles and no DSPs over the half-rate MXU,
  which itself was +2.2K LUTs over int8 in yosys.
- **Speed (simulated).** Qwen3-0.6B decode on the RTL of the board configuration: 1.89x fewer
  cycles per token at a DRAM rate close to what the card delivers today (the model at 25% of
  peak bandwidth), 1.27x at 80% and 1.02x at 100%. At high DRAM rates the MXU's one block per
  cycle becomes the limit (21.5 tokens/s at an assumed 100 MHz).
- **Accuracy cost.** On these small models (0.2 to 0.8 B parameters) 4-bit weights cost
  noticeably more than int8: on Qwen3-0.6B the perplexity of our book sample is 28.3, against
  23.5 in fp32 and int8, and the next-token distribution moves by a mean KL of 0.19 nats (int8:
  0.007).
  Keeping the LM head or the attention projections in int8 recovers part of it (below).

Everything here is measured in simulation or in synthesis (yosys, Vivado). Nothing has run on
the card.

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

**Qwen3-0.6B** (fp32 reference: ppl 23.50 / 39.80)

| config | bits/w | MB | weight err | ppl book | ppl isa.md | KL | top-1 text | top-1 greedy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| int8 | 8.25 | 615 | 0.0000 | 23.54 | 40.12 | 0.007 | 0.943 | 0.984 |
| int4-g128-fp32 | 4.25 | 317 | 0.0146 | 36.07 | 54.50 | 0.444 | 0.630 | 0.822 |
| int4-g128-fp32-s | 4.25 | 317 | 0.0119 | 34.76 | 56.59 | 0.374 | 0.659 | 0.828 |
| int4-g64-fp16 | 4.25 | 317 | 0.0121 | 32.34 | 51.46 | 0.360 | 0.644 | 0.847 |
| int4-g32-fp16 | 4.50 | 335 | 0.0097 | 30.34 | 50.80 | 0.279 | 0.681 | 0.844 |
| int4-g32-fp16-s | 4.50 | 335 | 0.0086 | 31.67 | 49.55 | 0.246 | 0.703 | 0.872 |
| int4-g32-e8m0 | 4.25 | 317 | 0.0218 | 40.35 | 65.50 | 0.562 | 0.586 | 0.800 |
| int4-g32-e4m3 | 4.25 | 317 | 0.0098 | 30.86 | 48.55 | 0.293 | 0.684 | 0.856 |
| e2m1-g128-fp32 | 4.25 | 317 | 0.0120 | 32.73 | 50.39 | 0.295 | 0.696 | 0.847 |
| e2m1-g128-fp32-s | 4.25 | 317 | 0.0106 | 29.90 | 45.66 | 0.247 | 0.710 | 0.841 |
| mxfp4 | 4.25 | 317 | 0.0135 | 31.53 | 50.87 | 0.273 | 0.703 | 0.869 |
| mxfp4-ceil | 4.25 | 317 | 0.0137 | 34.83 | 56.28 | 0.320 | 0.671 | 0.872 |
| mxfp4-s | 4.25 | 317 | 0.0127 | 31.69 | 51.01 | 0.269 | 0.703 | 0.894 |
| nvfp4 | 4.50 | 335 | 0.0090 | 29.01 | 49.72 | 0.191 | 0.747 | 0.903 |
| nvfp4-s | 4.50 | 335 | 0.0074 | 28.18 | 45.82 | 0.154 | 0.764 | 0.872 |
| int4k | 4.25 | 317 | 0.0098 | 31.20 | 50.08 | 0.284 | 0.670 | 0.866 |
| int4k-s | 4.25 | 317 | 0.0089 | 30.20 | 49.48 | 0.236 | 0.704 | 0.891 |
| e2m1k | 4.25 | 317 | 0.0103 | 29.35 | 48.71 | 0.240 | 0.715 | 0.859 |
| e2m1k-s | 4.25 | 317 | 0.0086 | 28.34 | 45.66 | 0.186 | 0.738 | 0.884 |
| mxfp4-s+head8 | 5.29 | 394 | 0.0093 | 30.44 | 49.59 | 0.232 | 0.721 | 0.897 |
| nvfp4-s+head8 | 5.48 | 408 | 0.0054 | 27.23 | 44.87 | 0.128 | 0.778 | 0.900 |
| int4k-s+head8 | 5.29 | 394 | 0.0066 | 29.40 | 48.48 | 0.213 | 0.730 | 0.894 |
| e2m1k-s+head8 | 5.29 | 394 | 0.0063 | 27.84 | 45.20 | 0.158 | 0.764 | 0.897 |
| e2m1k-s+ends8 | 5.51 | 410 | 0.0058 | 26.57 | 43.42 | 0.139 | 0.792 | 0.894 |
| e2m1k-s+down8 | 5.89 | 438 | 0.0052 | 27.21 | 44.62 | 0.128 | 0.792 | 0.906 |
| e2m1k-s+attn8 | 6.48 | 482 | 0.0039 | 26.66 | 42.69 | 0.098 | 0.808 | 0.934 |

**LFM2.5-230M** (fp32 reference: ppl 29.12 / 62.46)

| config | bits/w | MB | weight err | ppl book | ppl isa.md | KL | top-1 text | top-1 greedy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| int8 | 8.25 | 237 | 0.0000 | 29.05 | 62.57 | 0.003 | 0.964 | 0.969 |
| int4-g128-fp32 | 4.25 | 122 | 0.0150 | 36.71 | 82.65 | 0.331 | 0.656 | 0.780 |
| int4-g128-fp32-s | 4.25 | 122 | 0.0123 | 37.82 | 80.82 | 0.297 | 0.646 | 0.793 |
| int4-g64-fp16 | 4.25 | 122 | 0.0124 | 35.31 | 81.02 | 0.276 | 0.685 | 0.824 |
| int4-g32-fp16 | 4.50 | 129 | 0.0099 | 34.07 | 74.59 | 0.217 | 0.719 | 0.850 |
| int4-g32-fp16-s | 4.50 | 129 | 0.0088 | 33.73 | 72.09 | 0.206 | 0.720 | 0.789 |
| int4-g32-e8m0 | 4.25 | 122 | 0.0225 | 46.79 | 103.30 | 0.539 | 0.585 | 0.727 |
| int4-g32-e4m3 | 4.25 | 122 | 0.0100 | 33.27 | 71.73 | 0.224 | 0.710 | 0.824 |
| e2m1-g128-fp32 | 4.25 | 122 | 0.0121 | 37.67 | 78.26 | 0.246 | 0.723 | 0.855 |
| e2m1-g128-fp32-s | 4.25 | 122 | 0.0107 | 34.70 | 73.79 | 0.204 | 0.731 | 0.833 |
| mxfp4 | 4.25 | 122 | 0.0137 | 37.15 | 73.19 | 0.271 | 0.689 | 0.806 |
| mxfp4-ceil | 4.25 | 122 | 0.0139 | 35.60 | 78.70 | 0.282 | 0.676 | 0.797 |
| mxfp4-s | 4.25 | 122 | 0.0127 | 35.77 | 74.31 | 0.250 | 0.699 | 0.833 |
| nvfp4 | 4.50 | 129 | 0.0090 | 35.87 | 73.64 | 0.171 | 0.742 | 0.877 |
| nvfp4-s | 4.50 | 129 | 0.0074 | 32.83 | 71.05 | 0.142 | 0.762 | 0.863 |
| int4k | 4.25 | 122 | 0.0100 | 33.83 | 74.29 | 0.217 | 0.718 | 0.841 |
| int4k-s | 4.25 | 122 | 0.0090 | 34.79 | 71.28 | 0.206 | 0.725 | 0.811 |
| e2m1k | 4.25 | 122 | 0.0103 | 34.95 | 74.90 | 0.195 | 0.732 | 0.841 |
| e2m1k-s | 4.25 | 122 | 0.0086 | 33.16 | 69.03 | 0.159 | 0.768 | 0.859 |
| mxfp4-s+head8 | 5.42 | 156 | 0.0079 | 35.37 | 72.74 | 0.230 | 0.711 | 0.850 |
| nvfp4-s+head8 | 5.60 | 161 | 0.0046 | 32.29 | 70.89 | 0.130 | 0.781 | 0.877 |
| int4k-s+head8 | 5.42 | 156 | 0.0057 | 34.48 | 71.07 | 0.194 | 0.742 | 0.833 |
| e2m1k-s+head8 | 5.42 | 156 | 0.0053 | 32.92 | 69.50 | 0.146 | 0.773 | 0.877 |
| e2m1k-s+ends8 | 5.84 | 168 | 0.0044 | 31.67 | 69.50 | 0.122 | 0.802 | 0.872 |
| e2m1k-s+down8 | 6.06 | 174 | 0.0044 | 31.11 | 67.20 | 0.110 | 0.804 | 0.903 |
| e2m1k-s+attn8 | 6.33 | 182 | 0.0035 | 32.11 | 67.92 | 0.082 | 0.819 | 0.907 |

**Qwen3.5-0.8B** (fp32 reference: ppl 12.60 / 30.29)

| config | bits/w | MB | weight err | ppl book | ppl isa.md | KL | top-1 text | top-1 greedy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| int8 | 8.25 | 775 | 0.0000 | 12.57 | 30.37 | 0.003 | 0.969 | 0.972 |
| int4-g128-fp32 | 4.25 | 399 | 0.0155 | 19.13 | 43.86 | 0.295 | 0.666 | 0.834 |
| int4-g128-fp32-s | 4.25 | 399 | 0.0127 | 17.94 | 40.90 | 0.235 | 0.700 | 0.863 |
| int4-g64-fp16 | 4.25 | 399 | 0.0127 | 18.85 | 39.42 | 0.238 | 0.667 | 0.853 |
| int4-g32-fp16 | 4.50 | 423 | 0.0101 | 16.97 | 37.02 | 0.171 | 0.732 | 0.887 |
| int4-g32-fp16-s | 4.50 | 423 | 0.0090 | 16.33 | 36.74 | 0.149 | 0.761 | 0.872 |
| int4-g32-e8m0 | 4.25 | 399 | 0.0222 | 21.97 | 48.73 | 0.382 | 0.625 | 0.791 |
| int4-g32-e4m3 | 4.25 | 399 | 0.0102 | 16.84 | 37.88 | 0.177 | 0.716 | 0.891 |
| e2m1-g128-fp32 | 4.25 | 399 | 0.0122 | 17.12 | 38.34 | 0.181 | 0.709 | 0.881 |
| e2m1-g128-fp32-s | 4.25 | 399 | 0.0107 | 15.52 | 34.70 | 0.143 | 0.761 | 0.884 |
| mxfp4 | 4.25 | 399 | 0.0135 | 16.64 | 39.59 | 0.194 | 0.725 | 0.869 |
| mxfp4-ceil | 4.25 | 399 | 0.0142 | 17.24 | 40.67 | 0.215 | 0.714 | 0.863 |
| mxfp4-s | 4.25 | 399 | 0.0128 | 16.82 | 38.90 | 0.190 | 0.730 | 0.872 |
| nvfp4 | 4.50 | 423 | 0.0090 | 15.56 | 33.55 | 0.119 | 0.768 | 0.909 |
| nvfp4-s | 4.50 | 423 | 0.0074 | 15.30 | 34.18 | 0.099 | 0.772 | 0.928 |
| int4k | 4.25 | 399 | 0.0102 | 17.13 | 35.79 | 0.170 | 0.725 | 0.900 |
| int4k-s | 4.25 | 399 | 0.0092 | 16.48 | 35.15 | 0.152 | 0.746 | 0.894 |
| e2m1k | 4.25 | 399 | 0.0103 | 16.38 | 34.04 | 0.140 | 0.766 | 0.909 |
| e2m1k-s | 4.25 | 399 | 0.0086 | 16.03 | 33.13 | 0.118 | 0.790 | 0.884 |
| mxfp4-s+head8 | 5.60 | 527 | 0.0064 | 16.31 | 37.10 | 0.155 | 0.764 | 0.887 |
| nvfp4-s+head8 | 5.77 | 542 | 0.0037 | 14.90 | 33.66 | 0.081 | 0.818 | 0.944 |
| int4k-s+head8 | 5.60 | 527 | 0.0046 | 16.05 | 34.49 | 0.132 | 0.761 | 0.891 |
| e2m1k-s+head8 | 5.60 | 527 | 0.0044 | 15.60 | 33.50 | 0.098 | 0.807 | 0.906 |
| e2m1k-s+ends8 | 5.81 | 547 | 0.0040 | 14.53 | 32.31 | 0.076 | 0.836 | 0.934 |
| e2m1k-s+down8 | 6.07 | 571 | 0.0041 | 15.03 | 32.72 | 0.080 | 0.828 | 0.928 |
| e2m1k-s+attn8 | 6.84 | 643 | 0.0014 | 14.06 | 31.87 | 0.047 | 0.863 | 0.956 |

What the tables say:

- **Smaller blocks help most.** On Qwen3, int4 goes from a KL of 0.44 with a scale per 128 to
  0.36 per 64 and 0.28 per 32; FP4 from 0.30 per 128 to 0.19 per 16 (NVFP4).
- **FP4 elements beat int4 elements** at the same block and scale (`e2m1k` against `int4k`,
  `e2m1-g128` against `int4-g128`). Weights are roughly bell-shaped, and E2M1's uneven grid
  spends its codes near zero where the weights are.
- **MXFP4's power-of-two scale is its weak point.** With int4 elements an E8M0 scale is by far
  the worst option (the scale can be up to 2x too large). MXFP4's E2M1 elements soften this,
  but it stays behind NVFP4 and the FP4 two-level format on every model. Rounding the scale up so
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
  per-layer formats ("+ends8") would split the hardware layer loop. Both exist now (formats
  per kind and layer range, and the recommended mixes: `docs/formats.md`).

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
  pipeline is two stages longer, also for int8. That is lost in the noise of a two-layer Qwen3
  token (1,901,860 cycles against 1,901,880 before, bw 80%), and costs 64 cycles (0.25%) on the
  small MLP kernels of tests/test_perf.py.
- **(b)** doubles the MXU's rate for 4-bit weights: two blocks per cycle, so a 4-bit chunk per
  cycle. It needs a second ACT RAM block per cycle (even/odd banks), 128 more DSPs for the
  products at MCOLS = 2, and a different accumulation: two terms per column per cycle cannot
  go through the 4-partial adder loop in order, so the ISA would sum the pair first. It only
  pays when DRAM delivers more than 64 bytes per cycle.
- **(c)** avoids the DSPs (E2M1 x int8 is a shift and add: a, 3a, shifted), but 256 LUT
  products cost three times the fabric of (b), and it only works for E2M1 elements.

First winner: **(a)**. When it was built the card's DRAM delivered about 30 bytes per cycle
(20.7 M cycles per Qwen3 token measured; the simulator at 25% bandwidth, 32 bytes per cycle,
gives 19.8 M), well below the 64 bytes per cycle at which (a) saturates. There (a) gets the whole
2x of the bytes for 2K LUTs. Once the DRAM path delivers more than half its peak (DDR3-1066 at
80% is 136 bytes per 100 MHz cycle), (a) is the limit, and the full-rate MXU below replaces (b).

### Full rate by column reuse (built)

Decode multiplies one activation row by every weight, so an MXU with MCOLS = 2 leaves its
second column idle. `MM PAIR` (docs/isa.md, "Column reuse") gives that column the chunk's second
block: column 0 takes block 2c against ACT row 0, column 1 block 2c+1 against ACT row 1, which
`QACT DUP` fills with the same activation row in the same cycles. The two terms are added in
fp32 and then accumulated as usual, so an MM of at most MCOLS/2 rows consumes a whole 4-bit chunk
per cycle. What changed in the MXU (rtl/mxu/otpu_mxu.sv):

- **The products.** One DSP48 makes both columns' products: today (a0*2^16 + a1) * w with a
  shared weight. Under PAIR the columns have different weights (the low and the high nibble), so
  the 18-bit B operand packs both: B = wl*2^13 + wh (|w| <= 12). The product holds a0*wl at bit
  29 and a1*wh at bit 0, and the two cross terms in between (under 2^28 for a pair of positions,
  so the fields separate exactly after the post-adder, with an offset of 2^28 + 2^12). The group
  sums pick their field by a mux. No new DSPs.
- **The scales.** Each column takes its own block's weight scale and sub-block multipliers. The
  two scale words of a chunk are adjacent in DRAM and port A now returns the other word of an
  8-byte pair with each read (`a_rdata2`), so there is still one scale request per chunk; the
  scale FIFO is 64 bits wide.
- **The ACT RAM.** Rows at or past M read block ab+2c+1, the others ab+2c: a per-row read
  address.
- **The pair adder.** One fp32 adder (4 stages) per low column before the partial loop; only
  PAIR MMs pass through it, so int8 and half-rate MMs keep their latency.

Vivado synthesis of the MXU alone (out of context, xc7k480t-2, the board's parameters, 10 ns
clock; post-synthesis, no placement):

| MXU | LUT | FF | DSP | BRAM tiles | WNS (post-synthesis) |
|---|---:|---:|---:|---:|---:|
| half rate (a) | 11,880 | 9,060 | 142 | 30.5 | +5.33 ns |
| (a) + column reuse | 16,437 | 10,224 | 142 | 32.5 | +5.11 ns |

The extra 4.6K LUTs are mostly the packed B operand, the field muxes and the wider post-adder
result; the pair adder is about 0.5K (yosys). The whole design uses about 53% of the LUTs.

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
(`QACT`), the KV cache and `QST` stay int8: this is W4A8. `MM` flag bit6 `PAIR` and `QACT` flag
bit3 `DUP` are column reuse (docs/isa.md, "Column reuse"); `PAIR` reads a chunk's two scale words
as one 8-byte pair, so `ssa` and `srs` are multiples of 8 (the compiler falls back to half rate
otherwise, e.g. a matrix with an odd block count and dense scale rows).

## Using it

```python
from opentpu.llm.qwen3 import Engine, Spec, load_weights
eng = Engine(spec, W, wformat="fp4")                       # every weight 4-bit
eng = Engine(spec, W, wformat="fp4", head_format="int8")   # LM head int8
```

`wformat` is `"int8"` (default), `"fp4"` or `"int4"`; the Qwen3, LFM2 and Qwen3.5 images all
take it. `Config.PAIR` (a bitstream with CAPS bit5; `OTPU_PAIR=1` for the simulators) makes
the compiler use column reuse for every 4-bit MM of at most MCOLS/2 rows. For kernels, `runtime.Weight(w, shard, fmt="fp4")`. `opentpu/quant.py` has the
quantizers (`quantize_w4`, `quantize_mxu`) and the reference formats of the survey.
`tools/bench_llm.py` and `tools/perf_qwen.py` take `--wformat` and `--head-format`.

## Measured speed (simulated)

Qwen3-0.6B decode, one sequence at context 128, from `tools/bench_llm.py --bw 25,80,100
--batches 1 --ctx 128 --wformat fp4 [--head-format int8]`
(the RTL of the board configuration, MCOLS = 2, the AXI memory path at `bw` percent of peak
bandwidth; random weights, since the timing does not depend on them). Cycles per token; tokens
per second at an assumed 100 MHz, without host time.

| bw | int8 | fp4 | fp4, int8 LM head |
|---:|---:|---:|---:|
| 25% | 19,766,322 (5.06 tok/s) | 10,454,570 (9.57 tok/s, 1.89x) | 12,886,700 (7.76 tok/s, 1.53x) |
| 80% | 6,198,609 (16.13 tok/s) | 4,888,972 (20.45 tok/s, 1.27x) | 5,245,838 (19.06 tok/s, 1.18x) |
| 100% | 4,977,860 (20.09 tok/s) | 4,887,335 (20.46 tok/s, 1.02x) | 4,930,877 (20.28 tok/s, 1.01x) |

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
  against the ISA simulator (weights, KV cache, logits), with an int8 LM head at half rate and
  all 4-bit with column reuse at MCOLS = 2.
- Column reuse: tests/test_pair.py (ddr) holds the ISA simulator to a scalar model of the
  definition; the RTL fuzzers draw PAIR MMs (with and without a DUP operand, odd and even block
  counts); a 4-bit MLP runs PAIR on the RTL at the board's D = 128, MCOLS = 2.

## Not done

- No run on the card; the speeds above are simulated.
- The MXU's DSP cascade variant (`IMPL = 1`, a timing study option) does not support 4-bit
  weights; the simulation stops if it meets one.
- Column reuse needs a free column: MMs of more than MCOLS/2 rows (prefill chunks, batched
  decode) run at half rate.
- Calibration-based quantization (GPTQ, AWQ) and importing MXFP4 / NVFP4 checkpoints. An MXFP4
  block converts exactly only when its four scales span at most 2^3, and NVFP4 not in general.
