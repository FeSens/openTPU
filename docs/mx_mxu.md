# MX MXU: accumulate in the column, scales as shifts, one epilogue per output

Question (the user, 2026-10-11): "pumped MXU with systolic and MX (verify how we could make
attention fast on this, and if we need to drop support for NVFP at all or if we can find a
pumped way to improve this) ... Prefill speed and multi user, and MOE are now as important as
decode speed." The design measured here: the MXU column accumulates a whole output in its column
end's DSP as an exact integer, with every block scale applied as a shift (power-of-two block
scales, as in OCP MX), so the fp32 epilogue that today runs per K-block in every column runs once
per output, and E lanes of it serve the whole array. Pumped DSP48E1s (2x clock,
docs/dsp_pumping.md) or packed DSP48E2s give each physical column two stationary rows.

Measured: the formats in float64 emulation (formats_scan's text and dKL, 2000 tokens, three
models); the shared epilogue in the RTL co-sim (Verilator, LiteDRAM's DDR3-1066 controller, the
card's 133.33 MHz) with a timing hack that caps the MXU's drain at E results a cycle; the column
and lanes in xsim (bit-exact) and Yosys (area; Vivado's timing is pending, section 5); the four
workloads through the workloads agent's scoreboard. Nothing here ran on the card.

**Short answer**

- **Accuracy: no loss, a small gain.** With block scales as powers of two under a per-row fp32
  and 6-bit sub-block multipliers (`fp4x6`: 4.25 bits a weight, today's bytes), the 4-bit path is
  at least as accurate as today's fp4 on all three models (paired dKL -0.47 +- 0.36,
  -0.15 +- 0.15, -1.96 +- 0.21 on Qwen3-0.6B, Qwen3.5-0.8B, LFM2.5-230M), and the int8 path is
  better than today's int8 (-0.41, -0.07, -0.12), because the activations get a power of two per
  32 under a per-row fp32 instead of an fp32 per 128. The terms the shift window drops (at most 7
  in 10^5) cost nothing measurable (-0.02 to 0). The fixed-point accumulation is exact; today's
  fp32 order is within ~1e-7 of exact. int8 weights get 2.3% smaller (8.06 bits).
- **NVFP4 does not have to be dropped.** It runs exactly on the MX column: its E4M3 scale's
  mantissa is the multiplier, its exponent the shift (9 column-end DSPs a row instead of 5, or a
  pumped column end). It is 2.6 / 2.0 points better than today's fp4 on Qwen3 / Qwen3.5 and 1.6
  worse on LFM2.5, where it degrades with position: there it helps the attention's weights (-2.6
  points in them alone) and hurts the MLP's (+1.0) (section 3.4). It costs 0.25 bit a weight
  (+5.9% of 4-bit weight bytes). By the formats rule (a point of dKL is worth 10% of speed) it
  pays on Qwen3 and Qwen3.5: a per-model, per-kind choice, as `formats.py` already makes. OCP
  MXFP4 (E8M0 per 32) is 6-8 points worse than today's fp4: not a candidate.
- **Attention stays as fast.** q.K^T gives an output per row per K row (one block at d = 128),
  but attention is bound by the softmax on the VPU, not by the MXU's results: E = MCOLS/2 lanes
  cost 0 to +0.7% in every co-sim (prefill at 4, 8 and 16 rows, at 200, 544 and 4000 tokens,
  decode at 544 and 4000, batched 4 and 8 users, MoE experts of 1-16 rows; Qwen3, Qwen3.5,
  LFM2.5, the 35B and 26B experts), E = MCOLS/4 +0.1 to +1.4% once the MLP's down-projection
  partials are long enough (a compiler rule; MoE experts on 16 rows still +9-11%), one lane +13 to
  +56%. No raw-score path or full-rate q.K^T epilogue is needed. Nor does the MX MXU make
  attention faster: that is the VPU's (pumped VPU, another agent's study).
- **Area** (synthesis estimate, section 5): at today's 4 rows the MX datapath has 31% fewer LUTs,
  42% fewer flip-flops and 264 fewer DSPs; at 8 rows 35% fewer LUTs, 47% fewer flip-flops and 528
  fewer DSPs than today's MCOLS 8. Going from 4 to 8 rows costs +2.2K LUTs and +1.8K flip-flops
  (plus 4.3K SRLs, 1.2K LUT RAM, 24 DSPs and the ACT RAM's width) instead of MCOLS 8's +12.0K and
  +14.2K. Vivado timing is pending (section 5).
- **Workloads** (co-sim, the YPCB's memory path, an MX array of 8 rows with 4 lanes against today's
  MCOLS 4): prefill -28% (Qwen3, 4 rows; a 16-row prefill -38%, -52% on a 16-row array), -17%
  (Qwen3.5), -39% (LFM2.5), -12% at 4000 tokens; batched decode of 8 users -29% a step (-41% on 16
  rows); MoE expert FFNs of 4-16 rows -45 to -48% (-68 to -72% on 16 rows); single-user decode 0.
  Scoreboard (YPCB): 8 rows 1.00x decode, 1.38x prefill, 1.74x batched, 1.00x MoE; 16 rows with 4x
  TMEM 1.00 / 2.09 / 2.78 / 1.00.
- **Every card model** (3.6: 13 models, 2000 tokens, the MoE models whole, the Gemma 4 -it
  models on their chat): the MX defaults (int8 kinds in `int8x`; a production 4-bit kind in
  NVFP4 where it pays its 5.9% more bytes, else `fp4x6`) are better than today on 12, by 0.04 to
  2.5 points of dKL, for -2.0 to +4.5% of decode time; the 26B, with NVFP4 experts by the user's
  choice (+6.3%), equals today on its chat and is 15 points better on docs/isa.md. NVFP4 pays in
  most 4-bit kinds (every non-Gemma gate / up, the MoE models' dense kinds, LFM2.5-8B-A1B's
  experts), not in the 35B's streamed experts. Without NVFP4, LFM2.5-8B-A1B would be 2.3 points
  worse than today.
- **The production finals** (3.7: `formats_scan.py` with the production quantizers and the
  device's ranges, each model in its production configuration): MX is better than today on all
  13 card models (16 rows: the -it models on both texts), by 0.06 to 12 points of dKL, and its
  tail is no longer but on Qwen3.5-35B-A3B (8 tokens over 0.5 against 6, within noise) and
  E4B-it on docs/isa.md (16 against 8, from its `fp4x6` head: accepted, formats.md). 11 rows are
  within 2 SE of 3.6's; the others are the models' sensitivity to the quantizer's realization
  (the device's tie rounding and ranges against 3.6's: one change moves E4B-it's docs/isa.md by
  0.6 points), not errors.
- **A side finding**: a TMEM above 64K words serializes the sequencer (16-bit TMEM ranges in its
  dependency window): +70% on an 8-row prefill at 256K words until the ranges are widened
  (section 4).

## 1. Today's numerics (the reference)

`MM` (docs/isa.md, `opentpu/isasim.py` `_mm`), per output row `j`, weight row `n`, K-block `k`
of D = 128 elements:

```
isum      = sum_i ACT[j][k*D + i] * w[i]                              exact integer
  4-bit:    sum_b m_b * sum_{i in b} ACT * w     (b: 4 sub-blocks of 32; m_b unsigned 4-bit)
t[j][k]   = (i2f(isum) * ws[n][k]) * ASCALE[j][k]                    two fp32 roundings
acc[j]    = isum_4(t[j][0..KB-1])                four interleaved fp32 partials, then combined
```

- int8 weights: `ws` fp32 per (row, block), amax / 127 (`runtime.quantize_rows`): 8.25 bits.
- fp4 (E2M1 stored as twice its value, {0, 1, 2, 3, 4, 6, 8, 12}) and int4: `ws` bf16 per (row,
  block) and `m_b` in 1..15 per 32, picked by squared error (`quant.quantize_w4`): 4.25 bits.
- Activations (QACT): int8 per (row, block), `s = amax * (1/127)` in fp32; ROW mode one scale a
  row; CSCALE / RSCALE fold per-column / per-row factors in first.
- KV cache: K int8 per token (QST, a scale per 128-block), V^T int8 with V's per-token scales
  folded into P (QACT CSCALE), so P.V streams V^T with UNIT scales.

Every K-block's integer is converted and scaled before it is added, so every column runs i2f, two
fp32 multipliers (6 DSPs), the PAIR adder, the partial loop and the combine each cycle, and the
drain's read-modify-write lanes run per column too: ~5.1K LUTs and ~5.1K flip-flops a column with
the ACT RAM's share, against ~0.44K LUTs and ~3.1K flip-flops for the 128-DSP column itself
(docs/dsp_pumping.md section 5). That per-result part is what MCOLS 8 costs on the YPCB.

## 2. The MX column

Every per-block scale becomes a power of two under a per-row fp32, so the column end's DSP applies
it as a shift of its B operand, and a whole output accumulates as an exact integer in the DSP's
48-bit P register:

```
weights   Sw[n] fp32 per row; per block k: dw[n][k] (uint8); 4-bit: m[n][k][b], 6 bits per 32
acts      Sa[j] fp32 per row; per 32 (sub-block b): da[j][k][b] (uint8)  (one 32-bit word a block)
c         = sum_{i in b} ACT[j][i] * w[i]                     (today's chains, exact)
sh        = Wf - dw[n][k] - da[j][k][b]                       Wf 11 (fp4x6), 13 (fp4x4), 16 (int8x)
B         = sh >= 0 ? m << sh : 0                              m << sh < 2^17: the 18-bit B port
acc[j][n] = sum_k sum_b B * c                                  exact in 48 bits
y         = rne(rne(rne_f32(acc) * Sw[n]) * (Sa[j] * 2^-Wf))   once per output, then ACC / ASCALE
```

- **Drops.** A term whose two offsets add to more than Wf is dropped (B would need a right shift):
  its scale is under 2^-Wf of the row's largest weight scale times the row's largest activation
  scale. **Bounds**: 4-bit, a sub-block's sum < 2^16, times B < 2^17, 4 sub-blocks and 128 blocks
  (K = 16384): < 2^42; int8 < 2^44. K = 65536 still fits (< 2^46).
- **Column end**: today's four `otpu_colend` DSPs (a sub-block's two chain ends pre-added, times
  the multiplier, PCIN cascade) with B = m << sh instead of m, and a fifth DSP that accumulates over
  K (P = P + PCIN; a row's first block P = PCIN). The shift is an 8-bit subtract, a clamp and a
  17-bit shift of the 6-bit m per sub-block, in fabric. PAIR's two columns accumulate separately;
  the lane adds the two integers.
- **Epilogue lanes**: E lanes shared by the array, each i2f48 (round to nearest even), two fp32
  multipliers and the RMW multiply-add (ACC, ASCALE), fed from a result FIFO of 48-bit integers.
  A lane takes an output a cycle; a row of KB blocks takes KB cycles in the array (PAIR KB/2), so
  an MM keeps full rate while rows / KB <= E.
- **Formats** (bits a weight; the per-row fp32 is negligible):
  - `int8x`: int8, 2^-d per 128 under Sw: 8.06 bits (today 8.25: the block's fp32 becomes 8 bits).
  - `fp4x6`: E2M1, 2^-d per 128 and a 6-bit multiplier per 32 under Sw: one 32-bit word a block,
    4.25 bits, as today. `fp4x4`: today's 4-bit multipliers (4.19 bits).
  - `nvfp4` on the column: E2M1 with an E4M3 scale per 16 under a per-row fp32 (`nvfp4r`; the
    standard's per-tensor fp32 works too, the row's largest exponent becoming a per-row byte).
    Each 16-long chain end gets its own DSP (B = the E4M3 mantissa 8..15, subnormals 0..7, << sh,
    Wf 13), 8 a row plus the accumulator: 4.5 bits.
  - `mxfp4` (OCP): E2M1, E8M0 per 32: m = 1, Wf 16.
  - Activations `a8x32`: int8, 2^-d per 32 under a per-row fp32 (OCP MXINT8's grouping); the ACT
    RAM's scale word holds the four 8-bit offsets, a per-row register the fp32. `a8x128`: one
    offset per block.
- **Single-block outputs.** q.K^T at d = 128 is one block, so dw = da = 0 by construction and the
  lane multiplies by the token's K scale and the q row's scale: today's arithmetic for a
  one-block MM, and the K cache keeps its per-token fp32. Heads of 256 / 512 (Qwen3.5, Gemma 4)
  store K per token as int8x (QST in QACT's X mode).

## 3. Formats and accuracy

### 3.1 The accumulation itself

On random int8 operands with log-normal block scales (K = 1024 / 3072 / 8192), today's order
(per-block fp32 products, four interleaved fp32 partials) is within 5-8e-8 (mean) and 1e-6 (max)
of the exact sum, relative to the mean |output|; the MX order (exact 48-bit integers, then i2f
and two fp32 multiplies) is within 3-4e-8 / 7e-7 of its own exact sum. Both are ~2^-24: the
fixed-point accumulation changes nothing measurable by itself. What moves the model is the
change of formats that shifts require, and the dropped terms.

### 3.2 Method

`tools/exp/mx_formats.py` runs `tools/formats_scan.py`'s float64 emulation (the same layers; the
first 2000 tokens of docs/isa.md; the same head; dKL = mean KL(float || variant) minus int8's, in
percent, SE paired over the tokens) with a quantizer per variant for the weights, the head and the
activations (every MM input: the layers', the head's, q, P and K as attention reads them; V stays
int8 per token), and the window's drops emulated exactly per 16 elements (`XAct` / `WX`: the
dropped sub-products are subtracted from the product). The simulation host, 2026-10-12, 13-36 min a model.
Variants are weights / head / activations [/ flush: the drops on].

### 3.3 Results (2000 tokens)

dKL in percent (SE); perplexity of the 2000 tokens. **MX** rows run on the proposed column, drops
included; the others need today's per-column epilogue.

| variant | Qwen3-0.6B | Qwen3.5-0.8B | LFM2.5-230M | bits / weight |
|---|---:|---:|---:|---:|
| float | -1.25 (0.05), ppl 42.23 | -0.40 (0.01), 30.96 | -0.52 (0.02), 60.05 | |
| int8 today (int8 / int8 / a8) | 0, ppl 41.87 | 0, 30.91 | 0, 59.19 | 8.25 |
| int8 / int8 / a8x128 | +0.17 (0.04) | +0.12 (0.01) | +0.10 (0.02) | 8.25 |
| int8 / int8 / a8x32 | -0.50 (0.04) | -0.14 (0.01) | -0.22 (0.02) | 8.25 |
| **MX int8: int8x / int8x / a8x32** | **-0.41 (0.04)**, 42.29 | **-0.07 (0.01)**, 31.05 | **-0.12 (0.02)**, 60.53 | 8.06 |
| fp4 today (fp4 / int8 / a8) | +15.99 (0.45), ppl 46.27 | +11.07 (0.23), 35.18 | +15.83 (0.28), 64.68 | 4.25 |
| fp4 / int8 / a8x32 | +15.67 (0.44) | +10.92 (0.23) | +15.59 (0.28) | 4.25 |
| fp4x6 / int8 / a8 | +15.69 (0.44) | +11.11 (0.24) | +14.19 (0.25) | 4.25 |
| fp4x4 / int8 / a8 | +16.44 (0.40) | +11.94 (0.27) | +15.77 (0.35) | 4.19 |
| **MX fp4: fp4x6 / int8x / a8x32** | **+15.52 (0.42)**, 47.28 | **+10.91 (0.23)**, 34.52 | **+13.87 (0.25)**, 66.71 | 4.25 |
| MX fp4x4 / int8x / a8x32 | +15.96 (0.39) | +11.80 (0.28) | +15.63 (0.35) | 4.19 |
| nvfp4 / int8 / a8 (per-tensor fp32) | +13.98 (0.43) | +9.22 (0.21) | +22.01 (0.45) | 4.5 |
| nvfp4r / int8 / a8 (per-row fp32) | +13.28 (0.35) | +9.25 (0.21) | +16.92 (0.32) | 4.5 |
| **MX NVFP4: nvfp4r / int8x / a8x32** | **+13.38 (0.35)**, 47.23 | **+9.07 (0.21)**, 34.34 | **+17.38 (0.33)**, 76.11 | 4.5 |
| mxfp4 / int8 / a8 (OCP) | +24.77 (0.62) | +17.12 (0.38) | +23.03 (0.44) | 4.25 |
| MX mxfp4 / int8x / a8x32 | +24.29 (0.60) | +17.05 (0.38) | +22.84 (0.44) | 4.25 |

Paired differences (points, SE over the tokens):

| | Qwen3-0.6B | Qwen3.5-0.8B | LFM2.5-230M |
|---|---:|---:|---:|
| MX int8 - int8 today | -0.41 (0.04) | -0.07 (0.01) | -0.12 (0.02) |
| MX fp4x6 - fp4 today | -0.47 (0.36) | -0.15 (0.15) | -1.96 (0.21) |
| MX fp4x4 - fp4 today | -0.03 (0.37) | +0.74 (0.21) | -0.20 (0.27) |
| MX NVFP4 - fp4 today | -2.61 (0.41) | -1.99 (0.20) | +1.55 (0.36) |
| MX NVFP4 - MX fp4x6 | -2.14 (0.37) | -1.84 (0.20) | +3.52 (0.33) |
| MX mxfp4 - fp4 today | +8.29 (0.56) | +5.99 (0.35) | +7.01 (0.40) |
| fp4: a8x32 - a8 | -0.33 (0.15) | -0.15 (0.05) | -0.24 (0.06) |
| int8x: drops on - off | -0.02 (0.02) | -0.01 (0.01) | 0 |

Share of (row, output, 16-element) terms dropped: int8x 0 to 7e-7, fp4x6 7e-9 to 7e-5, NVFP4
7e-10 to 2e-5, mxfp4 0 to 2e-7.

- **The activations are the free gain**: a power of two per 32 under a per-row fp32 beats today's
  fp32 per 128 on all three models (-0.14 to -0.50 points at int8): a block's outlier no longer
  sets the scale of the other 96 elements. A power of two per 128 alone loses 0.10-0.17.
- **fp4x6 replaces today's fp4 without loss**: the bf16 block scale's mantissa is not needed when
  the sub-block multiplier has 6 bits. With today's 4-bit multipliers (fp4x4) Qwen3.5 loses 0.74.
- **NVFP4** is the most accurate 4-bit format on Qwen3 and Qwen3.5 (-2.0 to -2.6 points against
  today's fp4), not on LFM2.5 (section 3.4).
- **MXFP4** is far worse everywhere (+6 to +8 points against today's fp4).
- dKL here is the whole model's (formats_scan's), 2000 tokens; per-kind mixes (formats.py) were
  not scanned in the MX formats.

### 3.4 NVFP4 on LFM2.5: the position

dKL per window of 250 tokens (points against int8, same tokens):

| | 0 | 250 | 500 | 750 | 1000 | 1250 | 1500 | 1750 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fp4 today | +13.7 | +14.6 | +19.7 | +13.7 | +18.6 | +15.9 | +16.4 | +14.0 |
| MX fp4x6 | +12.6 | +14.4 | +17.0 | +14.0 | +14.4 | +13.7 | +13.6 | +11.2 |
| nvfp4 (per tensor) | +9.8 | +12.3 | +20.2 | +21.2 | +24.8 | +31.3 | +33.0 | +23.5 |
| MX nvfp4r | +9.9 | +12.4 | +17.5 | +15.9 | +17.7 | +25.1 | +23.3 | +17.2 |

NVFP4 is the best format over the first 500 tokens on LFM2.5 too, then it falls behind from
~750 tokens on; on Qwen3 and Qwen3.5 it stays ahead in every window.

Which weights (per-tensor NVFP4 in one kind, today's fp4 in the others; paired against fp4
everywhere, same tokens):

| NVFP4 in | dKL | against fp4 | windows 0 .. 1750 |
|---|---:|---:|---|
| attention (q, k, v, o) | +13.23 (0.24) | **-2.60 (0.16)** | +12.0 +12.9 +16.0 +11.5 +14.6 +14.0 +14.0 +10.9 |
| conv (in, out) | +14.81 (0.28) | -1.02 (0.16) | +13.6 +14.8 +17.0 +12.7 +16.3 +15.3 +15.7 +13.0 |
| MLP (gate, up, down) | +16.81 (0.32) | **+0.98 (0.29)** | +11.3 +13.7 +22.2 +17.3 +18.1 +19.7 +19.0 +13.3 |
| all but attention | +15.52 (0.30) | -0.31 (0.31) | +11.5 +13.1 +17.8 +15.1 +17.0 +19.0 +18.5 +12.1 |
| all | +22.01 (0.45) | +6.18 | |

NVFP4 helps LFM2.5's attention and conv weights and hurts its MLP, and the kinds do not add: the
attention's NVFP4 gains 2.6 points over fp4 elsewhere but costs 6.5 with the MLP and conv in
NVFP4 too. So on LFM2.5 NVFP4 is a per-kind choice (attention), which `opentpu/llm/formats.py`
already expresses and the MX column supports per MM. (For scale: on the MX column, LFM2.5 with
the attention in int8x and the rest in fp4x6 is at +8.77, -5.10 points against fp4x6 everywhere.)

### 3.5 What the change means

The MM's accumulation order changes (exact integers, then three roundings), and so does every
format's scale, the weights' and the activations'. Every image is re-quantized and every reference
recomputed (refcache keys, validate.py's card comparisons, the token-exact tests' expected tokens).
The user decides (section 7).

### 3.6 The card models: NVFP4 per kind, and MX against today

Every card model at 2000 tokens of docs/isa.md, the Gemma 4 -it models also of a chat (below)
(`tools/exp/mx_formats.py run`, the simulation host, 2026-10-12; the checkpoints at the build host's Hugging Face
revisions, three shards' sha256 compared).
Each run holds, besides float and int8 (today's int8 weights, head and a8: the dKL reference):

- **today**: the production configuration in today's encodings (int8 / fp4, a8), as the card
  picks it (`otpu-chat --wformat auto`: `formats.MIXES`; Gemma 4 E4B-it's `fit_formats`;
  `streamed.FORMATS` for the MoE models): Qwen3-0.6B, LFM2.5-230M, Qwen3.5-0.8B / 2B and Gemma 4
  E2B-it int8 everywhere; Qwen3.5-4B `delta=fp4,mlp=fp4`; LFM2-2.6B `conv=fp4,mlp=fp4`; SmolLM3
  `gateup@9-35=fp4`; Phi-4-mini `mlp@4-29=fp4`; E4B-it `head=fp4,gateup@0-11=fp4`; LFM2.5-8B-A1B
  and Qwen3.5-35B-A3B fp4 everywhere with an int8 head; gemma-4-26B-A4B-it int8 layers, fp4
  experts and head. The LM head is int8 except where named; routers are int8.
- **mx:...**: the same configuration on the MX column: int8 kinds in `int8x` (routers too),
  `a8x32` activations, the drops on, and each of the production's 4-bit kinds in `fp4x6` (F) or
  `nvfp4r` (N), every combination (a 2^n factorial over the n 4-bit kinds; gate / up and down
  separately where the MLP is 4-bit).
- **opp:...**: an int8 kind of production in NVFP4 on top (a speed opportunity: reported, not
  applied).

What the emulation covers beyond 3.3's three models:

- **K as QST writes it.** The K cache is int8 per token and head with one fp32 and a 2^-d per
  128-block (docs/isa.md QST; d = 128 is one fp32 a token, RSC), not `a8x32`; q, P and the MM
  inputs are `a8x32`, V int8 per token as today. (3.3's rows had K in `a8x32`: MX int8 against
  today was -0.41 / -0.07 / -0.12 there on Qwen3-0.6B / Qwen3.5-0.8B / LFM2.5-230M, -0.25 /
  -0.04 / -0.11 here.)
- **MoE.** LFM2-MoE and Qwen3.5-MoE: every layer, every expert the 2000 tokens route to (the
  whole model, not a sample), each variant routing on its own activations through its own
  router (int8 / int8x; LFM2: sigmoid + bias, Qwen3.5: the softmax of the top 8 and the shared
  expert's gate); the shared expert is the MLP kind. Gemma 4 26B-A4B as
  `tools/gemma4_quant_eval.py`: the router on the unit norm, the experts on the norm times
  `pre_feedforward_layernorm_2`'s gain (not folded), `per_expert_scale` folded into the down
  projection, the width padded to 768. Each expert is read once and quantized once per format
  on worker processes.
- **Gemma 4** (E2B-it, E4B-it with per-layer embeddings, KV sharing, the soft cap): every variant
  runs layer after layer together, each with its own K / V for the shared layers. The PLE table
  is a gather's records, not an MM format, and stays as the image places it: fp4 on E2B-it's
  card, int8 on E4B-it's host (the int8 reference: int8).
- With today's formats the extended paths give `lfm2.emulated_logits`, `qwen35.emulated_logits`
  and `gemma4_quant_eval.emulate`'s logits to 1.6e-15 on tiny random models
  (`mx_formats.py check`).

**The rule's arithmetic.** Decode time a token is T(b) = T0 + k b in the weight bytes b, fitted
through each model's card int8 and 4-bit (int8 head) decode (README, docs/formats.md: Qwen3-0.6B
21.6 / 31.3 tok/s, LFM2.5-230M 59.0 / 85.8, Qwen3.5-0.8B 17.6 / 24.5, Qwen3.5-2B 8.02 / 12.09,
SmolLM3 5.00 / 8.74, LFM2-2.6B 6.13 / 11.15, Phi-4-mini 3.99 / 6.56) at the runs' bytes
(`mx_formats.row_bytes`: int8x 8.06 bits a weight, fp4x6 4.25, nvfp4 4.5 with their row
words); Qwen3.5-4B its fp4 5.88 tok/s and the 2B's slope; Gemma 4 E2B docs/formats.md's
T = 13.375 M + 8.04 K (b - 1412 MiB) cycles, E4B the same slope (an estimate). A kind in NVFP4
instead of fp4x6 costs s% of decode time, so it must gain at least 0.1 s points of dKL. Its
gain is the factorial's main effect: per token, the mean over the 2^(n-1) pairs of combinations
that differ in that kind alone of KL(NVFP4) - KL(fp4x6), with its SE over the tokens. A kind
takes NVFP4 when that gain beats 0.1 s by at least 1 SE (formats.md's margin for a default),
else fp4x6. The MoE models' experts also set the slots and the streamed bytes (below).

**The defaults.** Every int8 kind of production stays 8-bit in `int8x` (the head too, where it is
int8), the activations `a8x32`, and each 4-bit kind takes NVFP4 where its gain pays its bytes by 1
SE, else `fp4x6`. The Gemma 4 instruct checkpoints are decided on their chat (below), where the
float model is calibrated, with docs/isa.md as the cross-check:

| model | production today | MX default: 4-bit kinds (the rest int8x) | MX - today, points (SE) | decode time vs today |
|:--|:--|:--|--:|--:|
| Qwen3-0.6B | int8 | none | -0.25 (0.05) | -1.7% |
| LFM2.5-230M | int8 | none | -0.11 (0.01) | -1.8% |
| Qwen3.5-0.8B | int8 | none | -0.04 (0.01) | -1.7% |
| Qwen3.5-2B | int8 | none | -0.08 (0.01) | -2.0% |
| Gemma 4 E2B-it (chat) | int8 (fp4 PLE table) | none | -0.50 (0.14) | -1.9% |
| Qwen3.5-4B | `delta=fp4,mlp=fp4` | `delta=fp4x6, gateup=nvfp4, down=nvfp4` | -1.03 (0.14) | +2.1% |
| LFM2-2.6B | `conv=fp4,mlp=fp4` | `conv=nvfp4, gateup=nvfp4, down=fp4x6` | -1.69 (0.33) | +4.0% |
| SmolLM3-3B | `gateup@9-35=fp4` | `gateup@9-35=nvfp4` | -0.28 (0.06) | +0.2% |
| Phi-4-mini | `mlp@4-29=fp4` | `gateup@4-29=nvfp4, down@4-29=fp4x6` | -0.35 (0.09) | +0.2% |
| Gemma 4 E4B-it (chat) | `head=fp4,gateup@0-11=fp4` | `head=fp4x6, gateup@0-11=fp4x6` | -0.34 (0.08) | -1.6% |
| LFM2.5-8B-A1B | fp4, int8 head | `attn=nvfp4, conv=nvfp4, mlp=nvfp4, experts=nvfp4` | -2.47 (0.60) | +4.5% |
| Qwen3.5-35B-A3B | fp4, int8 head | `attn=nvfp4, delta=nvfp4, mlp=nvfp4, experts=fp4x6` | -0.76 (0.18) | +3.0% |
| gemma-4-26B-A4B-it (chat) | int8, `experts=fp4`, fp4 head | `head=fp4x6, experts=nvfp4` (the user's choice) | +0.18 (2.39) | +6.3% |

No model is worse on its MX default than today beyond its SE: every one but the 26B is better
beyond its SE, and the 26B with its NVFP4 experts equals today on the chat (+0.18 (2.39)) and is
15.2 (3.4) points better on docs/isa.md. Within its SE NVFP4 is at least as accurate as `fp4x6` in
every kind of every model but one, the 26B's head on docs/isa.md (+4.4 (2.0)); so no per-kind
default trades one model's accuracy for another's: where the defaults differ, the gain does not
pay the kind's 5.9% more bytes (Qwen3.5-4B's DeltaNet, LFM2-2.6B's and Phi-4-mini's down
projections, Phi-4-mini's at 0.96 SE, just under the margin; the 35B's streamed experts; E4B-it's
head and gate / up on the chat). The 26B's experts are NVFP4 by the user's choice (2026-10-12),
for robustness to text far from the chat, over the rule's `fp4x6` there (below); its head is
`fp4x6` as the rule picks. E4B-it's NVFP4 head would not fit its 4 GiB image anyway (4114.6 MiB at
cap 2048, the MX compiler's measure). Gemma's LM head is its tied embedding: the gather reads the
head's rows, so the head's format is the embedding's in GATHER too, as the emulation runs it.

**The dense models with 4-bit kinds.** NVFP4 pays in the gate / up projections of the four
non-Gemma models (2.1 to 3.3 SE over its bar), in LFM2-2.6B's conv and in Qwen3.5-4B's down
projections. It does not in Qwen3.5-4B's DeltaNet (-0.07 for a bar of 0.11), LFM2-2.6B's down
projections (-0.14 for 0.14), Phi-4-mini's (-0.12 for 0.07: 0.96 SE, just under the margin), or
anywhere in E4B-it on its chat (head -0.02, gate / up -0.005; on docs/isa.md -0.37 and -0.23,
both over their bars). The kinds interact a little (Qwen3.5-4B's down gains -0.10 with the
DeltaNet in `fp4x6`, -0.20 with it in NVFP4), which the main effect averages. Against today
every default is better, by 0.28 to 1.69 points, for -1.6 to +4.0% of decode time; `fp4x6`
alone is better or equal (-0.09 to -0.58; SmolLM3 +0.05, 0.8 SE).

**Qwen3.5-4B** (card fp4 + slope 8.15 K/MiB):

| 4-bit kind | NVFP4 - fp4x6, points (SE) | decode time | bar | margin | default |
|:--|--:|--:|--:|--:|:--|
| delta | -0.07 (0.08) | +1.11% | 0.112 | -0.47 SE | fp4x6 |
| gateup | -0.41 (0.08) | +1.69% | 0.169 | +2.85 SE | NVFP4 |
| down | -0.16 (0.05) | +0.78% | 0.078 | +1.62 SE | NVFP4 |

| row | dKL % (SE) | against today, points (SE) | MiB a token | decode time vs today |
|:--|--:|--:|--:|--:|
| int8 | +0.00 (0.00) | -5.25 (0.16) | 4135 | +53.4% |
| today (production) | +5.25 (0.16) | +0.00 (0.00) | 2573 | +0.0% |
| MX, every 4-bit kind fp4x6 | +4.67 (0.12) | -0.58 (0.13) | 2562 | -0.4% |
| **MX, NVFP4 in gateup, down (default)** | +4.23 (0.12) | -1.03 (0.14) | 2634 | +2.1% |
| MX, every 4-bit kind NVFP4 | +4.04 (0.10) | -1.22 (0.15) | 2667 | +3.2% |

**LFM2-2.6B** (card int8 / fp4 fit, k 8.43 K/MiB):

| 4-bit kind | NVFP4 - fp4x6, points (SE) | decode time | bar | margin | default |
|:--|--:|--:|--:|--:|:--|
| conv | -0.36 (0.15) | +0.83% | 0.083 | +1.90 SE | NVFP4 |
| gateup | -0.80 (0.21) | +3.00% | 0.300 | +2.37 SE | NVFP4 |
| down | -0.14 (0.11) | +1.35% | 0.136 | +0.03 SE | fp4x6 |

| row | dKL % (SE) | against today, points (SE) | MiB a token | decode time vs today |
|:--|--:|--:|--:|--:|
| int8 | +0.00 (0.00) | -7.48 (0.31) | 2527 | +76.9% |
| today (production) | +7.48 (0.31) | +0.00 (0.00) | 1406 | +0.0% |
| MX, every 4-bit kind fp4x6 | +6.93 (0.28) | -0.55 (0.30) | 1408 | +0.2% |
| **MX, NVFP4 in conv, gateup (default)** | +5.79 (0.20) | -1.69 (0.33) | 1465 | +4.0% |
| MX, every 4-bit kind NVFP4 | +5.73 (0.22) | -1.75 (0.35) | 1485 | +5.4% |

**SmolLM3-3B** (card int8 / fp4 fit, k 8.51 K/MiB):

| 4-bit kind | NVFP4 - fp4x6, points (SE) | decode time | bar | margin | default |
|:--|--:|--:|--:|--:|:--|
| gateup | -0.33 (0.05) | +1.62% | 0.162 | +3.27 SE | NVFP4 |

| row | dKL % (SE) | against today, points (SE) | MiB a token | decode time vs today |
|:--|--:|--:|--:|--:|
| int8 | +0.00 (0.00) | -2.14 (0.07) | 3024 | +22.7% |
| today (production) | +2.14 (0.07) | +0.00 (0.00) | 2444 | +0.0% |
| MX, every 4-bit kind fp4x6 | +2.19 (0.07) | +0.05 (0.06) | 2409 | -1.4% |
| **MX, every 4-bit kind NVFP4 (default)** | +1.86 (0.07) | -0.28 (0.06) | 2450 | +0.2% |

**Phi-4-mini** (card int8 / fp4 fit, k 8.52 K/MiB):

| 4-bit kind | NVFP4 - fp4x6, points (SE) | decode time | bar | margin | default |
|:--|--:|--:|--:|--:|:--|
| gateup | -0.27 (0.06) | +1.43% | 0.143 | +2.09 SE | NVFP4 |
| down | -0.12 (0.05) | +0.68% | 0.068 | +0.97 SE | fp4x6 |

| row | dKL % (SE) | against today, points (SE) | MiB a token | decode time vs today |
|:--|--:|--:|--:|--:|
| int8 | +0.00 (0.00) | -2.89 (0.15) | 3772 | +31.4% |
| today (production) | +2.89 (0.15) | +0.00 (0.00) | 2836 | +0.0% |
| MX, every 4-bit kind fp4x6 | +2.80 (0.15) | -0.09 (0.08) | 2801 | -1.2% |
| **MX, NVFP4 in gateup (default)** | +2.54 (0.14) | -0.35 (0.09) | 2843 | +0.2% |
| MX, every 4-bit kind NVFP4 | +2.41 (0.14) | -0.47 (0.10) | 2863 | +0.9% |

**Gemma 4 E4B-it, the chat** (E2B slope, fixed part x 42/35 (estimate)):

| 4-bit kind | NVFP4 - fp4x6, points (SE) | decode time | bar | margin | default |
|:--|--:|--:|--:|--:|:--|
| head | -0.02 (0.04) | +0.53% | 0.052 | -0.69 SE | fp4x6 |
| gateup | -0.00 (0.04) | +0.49% | 0.049 | -1.27 SE | fp4x6 |

| row | dKL % (SE) | against today, points (SE) | MiB a token | decode time vs today |
|:--|--:|--:|--:|--:|
| int8 | +0.00 (0.00) | -0.55 (0.07) | 4567 | +14.6% |
| today (production) | +0.55 (0.07) | +0.00 (0.00) | 3947 | +0.0% |
| **MX, every 4-bit kind fp4x6 (default)** | +0.21 (0.07) | -0.34 (0.08) | 3881 | -1.6% |
| MX, every 4-bit kind NVFP4 | +0.18 (0.06) | -0.36 (0.08) | 3923 | -0.6% |

**The int8 models.** MX int8 (`int8x`, `a8x32`) is better than today's int8 on every one and
1.7-2.0% faster (8.06 bits a weight against 8.25: the scale words):

**Qwen3-0.6B** (card int8 / fp4 fit, k 9.11 K/MiB):

| row | dKL % (SE) | against today, points (SE) | MiB a token | decode time vs today |
|:--|--:|--:|--:|--:|
| int8 (production) | +0.00 (0.00) | +0.00 (0.00) | 586 | +0.0% |
| **MX: int8x, a8x32 (default)** | -0.25 (0.05) | -0.25 (0.05) | 575 | -1.7% |

**LFM2.5-230M** (card int8 / fp4 fit, k 9.11 K/MiB):

| row | dKL % (SE) | against today, points (SE) | MiB a token | decode time vs today |
|:--|--:|--:|--:|--:|
| int8 (production) | +0.00 (0.00) | +0.00 (0.00) | 226 | +0.0% |
| **MX: int8x, a8x32 (default)** | -0.11 (0.01) | -0.11 (0.01) | 221 | -1.8% |

**Qwen3.5-0.8B** (card int8 / fp4 fit, k 8.99 K/MiB):

| row | dKL % (SE) | against today, points (SE) | MiB a token | decode time vs today |
|:--|--:|--:|--:|--:|
| int8 (production) | +0.00 (0.00) | +0.00 (0.00) | 739 | +0.0% |
| **MX: int8x, a8x32 (default)** | -0.04 (0.01) | -0.04 (0.01) | 725 | -1.7% |

**Qwen3.5-2B** (card int8 / fp4 fit, k 8.55 K/MiB):

| row | dKL % (SE) | against today, points (SE) | MiB a token | decode time vs today |
|:--|--:|--:|--:|--:|
| int8 (production) | +0.00 (0.00) | +0.00 (0.00) | 1850 | +0.0% |
| **MX: int8x, a8x32 (default)** | -0.08 (0.01) | -0.08 (0.01) | 1811 | -2.0% |

**Gemma 4 E2B-it, the chat** (formats.md E2B model):

| row | dKL % (SE) | against today, points (SE) | MiB a token | decode time vs today |
|:--|--:|--:|--:|--:|
| int8 | +0.00 (0.00) | -0.33 (0.11) | 2242 | +0.0% |
| today (production) | +0.33 (0.11) | +0.00 (0.00) | 2242 | +0.0% |
| **MX: int8x, a8x32 (default)** | -0.16 (0.14) | -0.50 (0.14) | 2196 | -1.9% |

**Speed opportunities: the rule's reference.** The user decided (2026-10-12) that the formats rule
on the MX column takes MX int8 (`int8x`, `a8x32`) as its reference: a mix qualifies if its dKL
against MX int8 is at most 0.1 x its decode gain over MX int8, by 1 SE for a default. Against
today's int8 (formats.md's reference until now, which banks MX's own gain) faster mixes would
qualify on five models, for information: Qwen3.5-4B its attention in 4-bit (NVFP4 everywhere +5.04
for +55%, 3.6 SE under its bar), SmolLM3-3B its down projections 9-26 (+3.69 for +41%, 3.8 SE),
Qwen3.5-2B `mlp=nvfp4r` (+2.64 for +27.5%, 2.1 SE; `gateup=nvfp4r` +1.49 for +18%, 8.0 SE), E4B-it
its down projections 0-11 with an NVFP4 head (the chat: +0.28 for +19%, 19.7 SE; even as a step
from its MX default +0.07 for +2.2%, 2.0 SE; but the head does not fit the image) and E2B-it its
NVFP4 head (the chat: +0.68 for +9.4%, 1.6 SE). Against MX int8 none of the int8 models'
opportunities qualifies (Qwen3-0.6B, LFM2.5-230M and Qwen3.5-0.8B by 13 to 49 SE over the bars);
the closest are Qwen3.5-2B's gate / up (+1.57 for +15%, bar 1.53, -1.2 SE) and E2B-it's head
(+0.84 for +7.4%, bar 0.74, -0.7 SE). The 4-bit models' runs hold no MX int8 row; it ran apart for
the two that pass against today's int8 (the same tokens and the same float model, so KL per token
pairs across the runs). Against MX int8, the rule's own test: SmolLM3-3B's
`gateup=nvfp4r,down@9-26=nvfp4r` qualifies, +3.60 (0.11) for +38.4% (bar 3.84, 2.1 SE), and
`gateup=fp4x6,down@9-26=fp4x6` inside 1 SE (0.6 SE); Qwen3.5-4B's attention in 4-bit qualifies
inside 1 SE only (NVFP4 everywhere +5.13 for +52.0%, 0.5 SE; `fp4x6,attn=nvfp4r` 0.3 SE), an
opt-in by formats.md's margin, not a default; both models' MX defaults qualify by 3.6 SE. As a
step from their MX default, a stricter test, all of these fail, by 3.5 to 7.1 SE. No default
changes here: SmolLM3's faster mix (+38% against MX int8, the default's +20%) is the user's call.
E4B-it has no MX int8 row, and its mix needs an NVFP4 head that does not fit. MX int8 against
today's int8: Qwen3.5-4B -0.09 (0.01), SmolLM3 +0.09 (0.02), the one model where it is worse (its
production is a 4-bit mix, and the MX default is still better than today, -0.28 (0.06)). The rest
fail against today's int8 by 2.2 SE or more, which MX int8 does not turn: it reads 2% fewer bytes,
lowering every bar (by about 0.3 points at a 40% gain), and is at most 0.09 points worse than
today's int8 where measured: LFM2-2.6B (its attention in 4-bit, 6 SE), Phi-4-mini (2.2 SE) and the
MoE models (8 to 10 SE). The rows within 3 SE of their bar or under it (the Gemmas on the chat;
all of them in `tools/exp/mx_acc.json`):

| model | opportunity (weights; head) | dKL % (SE) | gain over int8 | bar | margin | against MX int8: dKL, points (SE) | gain | bar | margin |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| Qwen3.5-2B | `int8x,gateup=nvfp4r` | +1.49 (0.03) | +17.7% | 1.77 | +8.0 SE | +1.57 (0.03) | +15.3% | 1.53 | -1.2 SE |
| Qwen3.5-2B | `int8x,mlp=nvfp4r` | +2.64 (0.05) | +27.5% | 2.75 | +2.1 SE | +2.71 (0.06) | +25.0% | 2.50 | -3.9 SE |
| Gemma 4 E2B-it | `int8x,down@25-34=nvfp4r` | +0.43 (0.15) | +5.3% | 0.53 | +0.7 SE | +0.59 (0.10) | +3.4% | 0.34 | -2.6 SE |
| Gemma 4 E2B-it | `int8x,attn@25-34=nvfp4r,mlp@25-34=nvfp4r` | +2.14 (0.27) | +14.5% | 1.45 | -2.6 SE | +2.30 (0.25) | +12.3% | 1.24 | -4.3 SE |
| Gemma 4 E2B-it | `int8x`; head nvfp4r | +0.68 (0.16) | +9.4% | 0.94 | +1.6 SE | +0.84 (0.15) | +7.4% | 0.74 | -0.7 SE |
| Qwen3.5-4B | `nvfp4r` | +5.04 (0.13) | +55.0% | 5.50 | +3.6 SE | +5.13 (0.13) | +52.0% | 5.20 | +0.5 SE |
| Qwen3.5-4B | `fp4x6,attn=nvfp4r` | +5.65 (0.14) | +60.9% | 6.09 | +3.1 SE | +5.73 (0.14) | +57.7% | 5.77 | +0.3 SE |
| SmolLM3-3B | `int8x,gateup=nvfp4r,down@9-26=nvfp4r` | +3.69 (0.12) | +41.3% | 4.13 | +3.8 SE | +3.60 (0.11) | +38.4% | 3.84 | +2.1 SE |
| SmolLM3-3B | `int8x,gateup=fp4x6,down@9-26=fp4x6` | +4.29 (0.12) | +45.7% | 4.57 | +2.4 SE | +4.20 (0.12) | +42.7% | 4.27 | +0.6 SE |
| SmolLM3-3B | `int8x,gateup=nvfp4r` | +2.98 (0.10) | +31.1% | 3.12 | +1.4 SE | +2.89 (0.09) | +28.5% | 2.85 | -0.4 SE |
| Phi-4-mini | `int8x,attn=nvfp4r,mlp@4-29=nvfp4r` | +5.09 (0.19) | +46.7% | 4.67 | -2.2 SE | +2.55 (0.12) | +11.9% | 1.19 | -11.8 SE (a step from the MX default) |
| Gemma 4 E4B-it | `int8x,mlp@0-11=nvfp4r`; head nvfp4r | +0.28 (0.08) | +19.0% | 1.90 | +19.7 SE | +0.07 (0.07) | +2.2% | 0.22 | +2.0 SE (a step from the MX default) |

**The MoE models.** What ran is the whole model, not a sample: every layer and every expert
that the 2000 tokens route to (Qwen3.5-35B-A3B 217-256 of 256 a layer, gemma-4-26B-A4B-it
100-123 of 128, LFM2.5-8B-A1B of its 32, 4 a token), each variant through its own router on its
own activations, so a format's routing changes are in its dKL. The text is the one limit: one
document's tokens reach each expert as often as its content routes there.

**The streamed MoE models' time.** Their experts live in DRAM slots filled over PCIe, so an
expert format's bytes cost twice: in the DRAM reads of the k routed experts a token, and in the
link's streaming of the misses, of which a larger expert brings more bytes a miss and, through
fewer slots, more misses. A dense kind's bytes also take slot space. The model, from
docs/offload.md's card runs: T = T_today (5.3 tok/s for Qwen3.5-35B-A3B, 3.68 for the 26B,
15.0 for LFM2.5-8B-A1B from /dev/shm) + 8.1 K cycles a MiB of DRAM reads more + T_today x link
share (the link busy over the wall: 0.40, 0.48; the 8B 0.01 at 98.5% hits) x (the expert's size
ratio - 1 + 0.7 x the share of slots lost: cachesim's 7% fewer slots gave 5% more misses), with
today's slots (1680, 540, 616) and the slot space they hold. NVFP4 experts so cost the 35B about
+5.9% of decode time where their DRAM reads alone say +1.2%: an estimate, so a choice close to
its bar is named as such. `fp4x6`'s row head (`Sw` and a reserved word, 8 bytes) makes a row 8
bytes longer than today's fp4: +2.9% on a 512-element row (the 35B's down projections), +1.5% on
the 35B's experts, which the misses pay again (+1.0% of decode for MX `fp4x6` against today).

**LFM2.5-8B-A1B** (MoE: today 15.0 tok/s, link 1% of it, 616 slots, misses x0.7 the slots lost):

| 4-bit kind | NVFP4 - fp4x6, points (SE) | decode time | bar | margin | default |
|:--|--:|--:|--:|--:|:--|
| attn | -0.73 (0.33) | +0.19% | 0.019 | +2.16 SE | NVFP4 |
| conv | -0.93 (0.33) | +0.91% | 0.091 | +2.53 SE | NVFP4 |
| mlp | -1.92 (0.25) | +0.26% | 0.026 | +7.66 SE | NVFP4 |
| experts | -1.19 (0.25) | +3.06% | 0.306 | +3.52 SE | NVFP4 |

| row | dKL % (SE) | against today, points (SE) | MiB a token | decode time vs today |
|:--|--:|--:|--:|--:|
| int8 | +0.00 (0.00) | -24.99 (0.61) | 1658 | +63.1% |
| today (production) | +24.99 (0.61) | +0.00 (0.00) | 980 | +0.0% |
| MX, every 4-bit kind fp4x6 | +27.25 (0.64) | +2.26 (0.60) | 980 | +0.0% |
| **MX, every 4-bit kind NVFP4 (default)** | +22.51 (0.54) | -2.47 (0.60) | 1027 | +4.5% |

**Qwen3.5-35B-A3B** (MoE: today 5.3 tok/s, link 40% of it, 1680 slots, misses x0.7 the slots lost):

| 4-bit kind | NVFP4 - fp4x6, points (SE) | decode time | bar | margin | default |
|:--|--:|--:|--:|--:|:--|
| attn | -0.30 (0.08) | +0.36% | 0.036 | +3.30 SE | NVFP4 |
| delta | -0.39 (0.10) | +1.35% | 0.135 | +2.53 SE | NVFP4 |
| mlp | -0.18 (0.04) | +0.19% | 0.019 | +3.78 SE | NVFP4 |
| experts | -0.10 (0.06) | +5.88% | 0.588 | -8.59 SE | fp4x6 |

| row | dKL % (SE) | against today, points (SE) | MiB a token | decode time vs today |
|:--|--:|--:|--:|--:|
| int8 | +0.00 (0.00) | -4.76 (0.18) | 2897 | +91.9% |
| today (production) | +4.76 (0.18) | +0.00 (0.00) | 1745 | +0.0% |
| MX, every 4-bit kind fp4x6 | +4.86 (0.18) | +0.11 (0.13) | 1746 | +1.0% |
| **MX, NVFP4 in attn, delta, mlp (default)** | +4.00 (0.16) | -0.76 (0.18) | 1794 | +3.0% |
| MX, every 4-bit kind NVFP4 | +4.01 (0.14) | -0.74 (0.17) | 1831 | +9.0% |

**gemma-4-26B-A4B-it, the chat** (MoE: today 3.68 tok/s, link 48% of it, 540 slots, misses x0.7 the slots lost):

| 4-bit kind | NVFP4 - fp4x6, points (SE) | decode time | bar | margin | default |
|:--|--:|--:|--:|--:|:--|
| head | -0.99 (1.22) | +0.97% | 0.097 | +0.73 SE | fp4x6 |
| experts | +1.75 (1.98) | +6.62% | 0.662 | -1.22 SE | fp4x6 |

| row | dKL % (SE) | against today, points (SE) | MiB a token | decode time vs today |
|:--|--:|--:|--:|--:|
| int8 | +0.00 (0.00) | -15.18 (1.95) | 3807 | +88.6% |
| today (production) | +15.18 (1.95) | +0.00 (0.00) | 2753 | +0.0% |
| MX, every 4-bit kind fp4x6 | +12.30 (2.37) | -2.87 (2.27) | 2729 | -0.3% |
| **MX, NVFP4 in experts (default, the user's choice)** | +15.35 (2.59) | +0.18 (2.39) | 2781 | +6.3% |
| MX, every 4-bit kind NVFP4 | +13.07 (2.41) | -2.11 (2.31) | 2805 | +7.3% |

On Qwen3.5-35B-A3B NVFP4 pays in the dense kinds (attention, DeltaNet, the shared expert: 2.5 to
3.8 SE over their bars) and not in the experts (-0.10 (0.06) for a bar of 0.59: their bytes are
paid twice; the choice holds at the DRAM share alone, bar 0.12). MX in `fp4x6` alone equals
today there (+0.11 (0.13)). LFM2.5-8B-A1B streams nearly nothing (98.5% of its experts resident),
so its experts cost only their DRAM reads, and NVFP4 pays in every kind, the experts by 3.5 SE.
Its kinds interact strongly (the same kind's pairs range from -4.2 to +1.9 points), and today's
fp4 sits 25 points from int8, the largest of the card models: there MX in `fp4x6` alone is
**worse than today**, +2.26 (0.60) points: the one model where the MX column without NVFP4
clearly loses accuracy (the 26B's +4.9 (3.4) on docs/isa.md is 1.4 SE).

gemma-4-26B-A4B-it is the outlier: on docs/isa.md the instruct checkpoint is far from calibrated
(float perplexity 682), and under int8 alone 321 of the 2000 tokens are more than a nat from
float (dKL -68 for float); NVFP4 experts cut that damage broadly (-20.0 (2.9) points against
`fp4x6`, still -13.2 without the 50 most extreme tokens each side), where an NVFP4 head
costs +4.4 (2.0). On the chat below, where the float model is calibrated (perplexity 25.3), the
experts' format does not matter (+1.75 (1.98); +0.24 without the 50 extreme tokens each side)
and the rule's default is `fp4x6` everywhere. The user chose NVFP4 experts (2026-10-12) for
robustness to text far from the chat, at about +6.6% of decode time (estimated); the head stays
`fp4x6`. The table above is the chat's, both texts side by side below.

**Gemma 4 -it on their chat.** docs/isa.md is plain prose to the instruct checkpoints (float
perplexity 150, 98 and 682 for E2B-it, E4B-it and the 26B-it), on which even int8 moves the 26B
far from float. [gemma4.md](gemma4.md) also scores them on a chat (`tools/chat_text.py`:
E2B-it's own replies to 16 questions in one conversation, rendered by its template; the same
file, the build host's `gemma-it/chat-it.json`, the first 2000 of its 2637 tokens, `mx_formats.py run
--ids`), and so does this, every variant again. Decided (2026-10-12): the Gemma -it
defaults follow the chat, docs/isa.md is the cross-check.

| model | | docs/isa.md | the chat |
|:--|:--|--:|--:|
| Gemma 4 E2B-it | float perplexity (int8's) | 149.6 (153.3) | 3.54 (3.62) |
| | float against int8, dKL | -4.69 (0.24) | -1.46 (0.12) |
| | MX int8 - today | -1.78 (0.19) | -0.50 (0.14) |
| | `int8x,down@25-34=nvfp4r`: margin against int8 / as a step from MX | +9.1 / -3.5 SE | +0.7 / -2.6 SE |
| | `int8x,attn@25-34=nvfp4r,mlp@25-34=nvfp4r`: margin against int8 / as a step from MX | +5.0 / -5.0 SE | -2.6 / -4.3 SE |
| | `int8x,mlp@15-34=nvfp4r`: margin against int8 / as a step from MX | +3.1 / -4.4 SE | -4.1 / -5.7 SE |
| | `int8x`, head nvfp4r: margin against int8 / as a step from MX | +8.4 / -2.2 SE | +1.6 / -0.7 SE |
| Gemma 4 E4B-it | float perplexity (int8's) | 98.0 (95.0) | 4.34 (4.40) |
| | float against int8, dKL | -1.89 (0.16) | -0.83 (0.07) |
| | head: NVFP4 - fp4x6 (bar 0.05) | -0.37 (0.10), +3.3 SE: NVFP4 | -0.02 (0.04), -0.7 SE: fp4x6 |
| | gateup: NVFP4 - fp4x6 (bar 0.05) | -0.23 (0.11), +1.6 SE: NVFP4 | -0.00 (0.04), -1.3 SE: fp4x6 |
| | MX default - today | -2.04 (0.38) | -0.34 (0.08) |
| | MX in `fp4x6` alone - today | -1.44 | -0.34 |
| | `int8x,mlp@0-11=nvfp4r`, head nvfp4r: margin against int8 / as a step from MX | +5.5 / -2.6 SE | +19.7 / +2.0 SE |
| gemma-4-26B-A4B-it | float perplexity (int8's) | 681.9 (912.4) | 25.32 (25.96) |
| | float against int8, dKL | -68.07 (3.58) | -18.03 (1.94) |
| | head: NVFP4 - fp4x6 (bar 0.10) | +4.43 (2.02), -2.2 SE: fp4x6 | -0.99 (1.22), +0.7 SE: fp4x6 |
| | experts: NVFP4 - fp4x6 (bar 0.66) | -19.96 (2.89), +6.7 SE: NVFP4 | +1.75 (1.98), -1.2 SE: fp4x6 |
| | MX default - today | -15.21 (3.36) | +0.17 (2.39) |
| | MX in `fp4x6` alone - today | +4.90 | -2.87 |
| | `int8x,attn=nvfp4r,experts=nvfp4r`, head nvfp4r: margin against int8 / as a step from MX | -9.9 / -14.2 SE | -8.8 / -7.8 SE |
| | `int8x,mlp=nvfp4r,experts=nvfp4r`, head nvfp4r: margin against int8 / as a step from MX | -8.3 / -12.1 SE | -8.2 / -7.7 SE |

The chat lowers every format's cost (today's E4B-it mix +0.55 against int8 there, +2.30 on
docs/isa.md) and NVFP4's gains with it: E4B-it keeps `fp4x6` in its head and gate / up, the 26B
in its experts and head. On the 26B the two texts disagree (above): on docs/isa.md NVFP4 experts
are worth 20 points and MX in `fp4x6` alone is +4.9 (3.4) against today; on the chat they are
worth nothing. The user chose NVFP4 experts for the 26B (2026-10-12), for text far from the
chat, at about +6.6% of decode time (estimated).

**Drops.** The share of (row, output, 16-element) terms the windows drop is 0 to 7e-7 in the rows
with no 4-bit kind and at most 7.3e-5 in any MX row (Qwen3.5-0.8B in `fp4x6` everywhere), the
range 3.3 measured at no cost (drops on - off: -0.02 to 0 points).

**What ran where.** Every run on the simulation host (2026-10-12; 2000 tokens; 6 BLAS threads and 6
quantizer processes; 3 min for LFM2.5-230M to 82 min for Qwen3.5-35B-A3B), no proxy model: the
checkpoints at the build host's Hugging Face revisions (Qwen3.5-35B-A3B 59d61f3c, gemma-4-26B-A4B-it
4d7ae498, Phi-4-mini cfbefacb, ...), shard sha256s equal to the build host's (Phi-4-mini, Qwen3.5-4B,
Qwen3.5-35B-A3B). The -it checkpoints read docs/isa.md as plain text after `<bos>`, as
formats.md's Gemma 4 rows do, and the chat above. Qwen3.5-4B's and SmolLM3's MX int8 rows ran
apart (`mx_formats.py run` with that variant, 10 and 7 min; int8's KL per token equal to the first
runs'). The decode-time model is the card's for the dense models, an estimate for E4B-it (E2B's
slope) and for the MoE models (above). The runs' numbers (every row's dKL, SE, bytes and drops;
each kind's effect, cost and bar; the opportunities against both references; the defaults and the
decisions) are `tools/exp/mx_acc.json`.

### 3.7 The production finals

Every card model at 2000 tokens in its production configuration on the MX column as mx-column
(3095bb02) sets it, through the production path: `tools/formats_scan.py ppl` with `opentpu.quant`'s
quantizers (int8 = `int8x`, fp4 = `fp4x6`, nvfp4 = `nvfp4r`; QACT `a8x32`; the weight's exact ties
rounded as the device's fp32 quantizer rounds them; MM's window drops) and the device's ranges: K
as QST writes it (ROW at d = D, else an offset per D-block), P per attention block of 256
positions, the down projection's input per F chunk (`kernels.mlp.down_chunk`), the DeltaNet
out_proj's per head group, the int8 embedding as the gather's rows, a MoE expert's down input per
slot range (`moe.ExpertFormat.ranges`). The configurations: `formats.MIXES` for the dense 4-bit
models, `streamed.FORMATS` for the MoE models (LFM2.5-8B-A1B nvfp4 with an int8 head;
Qwen3.5-35B-A3B fp4 with `attn, delta, mlp=nvfp4` and an int8 head; gemma-4-26B-A4B-it int8 with
`experts=nvfp4` and an fp4 head), `Spec.fit_formats` for E4B-it; Gemma's PLE table as the image of
cap 2048 places it. For the MoE models formats_scan gains the LFM2-MoE and Qwen3.5-MoE FFN (this
branch: the router in int8, `moe.route`, Qwen3.5's shared expert and its gate, each routed expert
read once and quantized once per format), held by `formats_scan.py check` to `lfm2` /
`qwen35.emulated_logits` on tiny MoE models (logits to 1.9e-15). Today is 3.6's today rows, the
pre-MX formats_scan's exactly: the float NLL per token is equal in every run, and ee8f118d's
formats_scan gives 3.6's today KL per token to the last digit on Qwen3-0.6B (int8), LFM2-2.6B
(`conv=fp4,mlp=fp4`) and E2B-it on the chat. MX - today is paired per token (points of dKL, SE over
the tokens).

| model | text | production on MX | MX - today, points (SE) | phase 1's MX - today | finals - phase 1's MX, points (SE) |
|:--|:--|:--|--:|--:|--:|
| Qwen3-0.6B | docs/isa.md | int8 | -0.25 (0.04) | -0.25 (0.05) | -0.008 (0.027), 0.3 SE |
| LFM2.5-230M | docs/isa.md | int8 | -0.12 (0.01) | -0.11 (0.01) | -0.015 (0.009), 1.7 SE |
| Qwen3.5-0.8B | docs/isa.md | int8 | -0.06 (0.01) | -0.04 (0.01) | -0.022 (0.005), 4.2 SE |
| Qwen3.5-2B | docs/isa.md | int8 | -0.10 (0.01) | -0.08 (0.01) | -0.024 (0.006), 4.3 SE |
| Gemma 4 E2B-it | the chat | int8 (fp4 PLE table) | -0.55 (0.13) | -0.50 (0.14) | -0.055 (0.086), 0.6 SE |
| Gemma 4 E2B-it | docs/isa.md | int8 (fp4 PLE table) | -2.11 (0.21) | -1.78 (0.19) | -0.330 (0.105), 3.1 SE |
| Qwen3.5-4B | docs/isa.md | `delta=fp4x6, mlp=nvfp4` | -1.04 (0.14) | -1.03 (0.14) | -0.014 (0.025), 0.6 SE |
| LFM2-2.6B | docs/isa.md | `conv=nvfp4, gateup=nvfp4, down=fp4x6` | -1.69 (0.33) | -1.69 (0.33) | -0.005 (0.038), 0.1 SE |
| SmolLM3-3B | docs/isa.md | `gateup@9-35=nvfp4` | -0.32 (0.06) | -0.28 (0.06) | -0.042 (0.030), 1.4 SE |
| Phi-4-mini | docs/isa.md | `gateup@4-29=nvfp4, down@4-29=fp4x6` | -0.28 (0.10) | -0.35 (0.09) | +0.064 (0.036), 1.8 SE |
| Gemma 4 E4B-it | the chat | `head=fp4x6, gateup@0-11=fp4x6` | -0.35 (0.07) | -0.34 (0.08) | -0.016 (0.038), 0.4 SE |
| Gemma 4 E4B-it | docs/isa.md | `head=fp4x6, gateup@0-11=fp4x6` | -0.80 (0.32) | -1.44 (0.34) | +0.639 (0.149), 4.3 SE |
| LFM2.5-8B-A1B | docs/isa.md | nvfp4, int8 head | -2.14 (0.61) | -2.47 (0.60) | +0.333 (0.326), 1.0 SE |
| Qwen3.5-35B-A3B | docs/isa.md | fp4x6, `attn, delta, mlp=nvfp4`, int8 head | -0.70 (0.17) | -0.76 (0.18) | +0.058 (0.053), 1.1 SE |
| gemma-4-26B-A4B-it | the chat | int8, `experts=nvfp4`, fp4x6 head | -3.75 (2.29) | +0.18 (2.39) | -3.928 (1.664), 2.4 SE |
| gemma-4-26B-A4B-it | docs/isa.md | int8, `experts=nvfp4`, fp4x6 head | -11.98 (3.54) | -15.21 (3.36) | +3.232 (1.808), 1.8 SE |

**Consistency with 3.6.** The production path and phase 1's emulation run the same layers (on a
tiny Qwen3.5 the MX tree's layer code with phase 1's quantizers gives phase 1's layer outputs
exactly) and differ in four places: an int8x weight's exact tie (the device's fp32 rounding against
half to even: 3.5e-4 of the values, unbiased, the same squared error), the ranges (P per attention
block, the down projection's input per F chunk, the DeltaNet out_proj per head group, against one
range a row), the int8 embedding (the gather's rows) and MM's drops in q.K and P.V (phase 1 summed
the attention exactly). 11 of the 16 rows are within 2 SE of phase 1's MX row of the same
configuration; the others are the two Qwen3.5 int8 models (-0.022 (0.005) and -0.024 (0.006): 4 SE
of a tiny paired SE, in MX's favour), the two Gemma -it models on docs/isa.md, the cross-check text
(E2B-it -0.33 (0.11), E4B-it +0.64 (0.15)), and gemma-4-26B-A4B-it on the chat -3.93 (1.66), whose
KL is large everywhere (p99 near 6 on the chat). With phase 1's choices put back in the production
path the rows go back to phase 1's: E4B-it -0.02 (0.08) from its row and E2B-it -0.18 (0.10) with
the ties and the ranges put back, Qwen3.5-0.8B -0.011 (0.005) with all but the attention's drops.
They are not errors but the models' sensitivity to the quantizer's realization: on E4B-it any one
of the changes alone (the ties only: -0.59 (0.14) from the final; P and down only: -0.54 (0.15))
brings the final back, though they touch disjoint code, and E2B-it moved the other way. On
docs/isa.md an instruct model's dKL moves by up to ~0.6 points with one rounding step on a few
weights, more than its token SE; on the chat, the text the -it defaults are decided on, E2B-it's
and E4B-it's finals are within 0.6 SE of phase 1's rows. The 26B, whose KL is an order larger than
the others' on both texts, moves by points between neighbouring configurations already in phase 1's
factorial (its experts NVFP4 against `fp4x6` on the chat: +1.75 (1.98)); its finals are not ablated
(50 min a run). formats.md's Gemma 4 E4B-it section has the decomposition.

**The tail.** The mean can hide a few bad tokens (MM's window dropping a group on a row with an
outlier, mx-compiler's finding on validate), so each final's KL per token against today's, on the
same tokens:

| model | text | p99, MX / today | max, MX / today | tokens > 0.1 | tokens > 0.5 | window drops (layer weights' MMs): inputs, tokens |
|:--|:--|--:|--:|--:|--:|:--|
| Qwen3-0.6B | docs/isa.md | 0.060 / 0.067 | 0.42 / 0.61 | 5 / 8 | 0 / 1 | not counted |
| LFM2.5-230M | docs/isa.md | 0.016 / 0.021 | 0.15 / 0.13 | 1 / 1 | 0 / 0 | not counted |
| Qwen3.5-0.8B | docs/isa.md | 0.013 / 0.018 | 0.08 / 0.04 | 0 / 0 | 0 / 0 | not counted |
| Qwen3.5-2B | docs/isa.md | 0.010 / 0.016 | 0.04 / 0.05 | 0 / 0 | 0 / 0 | not counted |
| Gemma 4 E2B-it | the chat | 0.197 / 0.328 | 0.62 / 1.23 | 58 / 89 | 4 / 9 | 4.1e-05, 887 (every MM kind) |
| Gemma 4 E2B-it | docs/isa.md | 0.284 / 0.552 | 0.75 / 1.74 | 93 / 196 | 4 / 24 | 8.9e-06, 271 (every MM kind) |
| Qwen3.5-4B | docs/isa.md | 0.257 / 0.331 | 0.61 / 1.20 | 174 / 254 | 3 / 7 | 2.8e-04, 1906 |
| LFM2-2.6B | docs/isa.md | 0.337 / 0.442 | 2.61 / 4.64 | 288 / 421 | 8 / 16 | 4.0e-05, 871 (every MM kind) |
| SmolLM3-3B | docs/isa.md | 0.127 / 0.144 | 0.50 / 0.72 | 33 / 53 | 1 / 1 | not counted |
| Phi-4-mini | docs/isa.md | 0.173 / 0.216 | 0.93 / 0.99 | 89 / 103 | 3 / 4 | 2.2e-03, 1959 (every MM kind) |
| Gemma 4 E4B-it | the chat | 0.152 / 0.206 | 0.62 / 0.77 | 38 / 51 | 1 / 3 | 1.5e-05, 604 (every MM kind) |
| Gemma 4 E4B-it | docs/isa.md | 0.381 / 0.292 | 2.74 / 6.10 | 102 / 178 | 16 / 8 | 5.8e-07, 19 |
| LFM2.5-8B-A1B | docs/isa.md | 1.517 / 1.426 | 3.14 / 3.58 | 1589 / 1658 | 366 / 393 | 4.6e-05, 1312 (every MM kind) |
| Qwen3.5-35B-A3B | docs/isa.md | 0.268 / 0.293 | 1.49 / 2.12 | 207 / 270 | 8 / 6 | not counted |
| gemma-4-26B-A4B-it | the chat | 5.762 / 6.051 | 16.11 / 19.39 | 444 / 495 | 216 / 251 | 1.6e-07, 2 |
| gemma-4-26B-A4B-it | docs/isa.md | 9.147 / 9.237 | 17.82 / 19.48 | 1190 / 1246 | 542 / 628 | 6.3e-08, 0 |

MX has no more tokens over 0.5 than today on every model but two, and a max no higher but on
LFM2.5-230M (0.15 against 0.13) and Qwen3.5-0.8B (0.08 against 0.04), both far under 0.5; its p99
is lower everywhere but on E4B-it's docs/isa.md and LFM2.5-8B-A1B (1.52 against 1.43, a model whose
KL is large on that text in every row). Qwen3.5-35B-A3B has 8 tokens over 0.5 against today's 6 (a
bootstrap over the tokens puts the difference within -4 to +8; over 0.3 and over 1.0 MX has fewer,
16 against 19 and 1 against 3, and a lower max, 1.49 against 2.12): its high tokens are the `fp4x6`
experts' (MX int8 under 0.15 on them; phase 1's identical row has the same tokens high, and its
NVFP4-experts row 7 over 0.5). Gemma 4 E4B-it on docs/isa.md has 16 against 8, from its `fp4x6`
head (2 with an NVFP4 head, which does not fit its image; the decision, and the chat's 1 against 3,
in formats.md's E4B-it section). The 26B's KL is large on both texts and its tail too, shorter on
MX: the 26B on the chat 216 against 251 over 0.5 (max 16.1 against 19.4); the 26B on docs/isa.md
542 against 628 over 0.5 (max 17.8 against 19.5).

**Drops.** `formats_scan.py ppl` now counts what MM's window drops in every variant's MMs
(`_counted_mm` around `quant.mx_matmul`; "drops" in a row of the JSON: the MMs, the (row, 16
columns) inputs and (row, output, group) terms dropped, per MM kind: the layers' weights, q.K, P.V,
the head; the dropped terms per token; the first runs, marked "every MM kind" above, have them
together). The layer weights' drops are few and fall on tokens no worse than today: Qwen3.5-4B's
`fp4x6` DeltaNet (W 11) drops 2.8e-4 of its inputs' groups, on 1906 tokens (MX int8: 142 groups on
30 tokens), and on those tokens MX - today is -0.010 points; E4B-it's on docs/isa.md 80 groups on
19 tokens. P.V drops a group where an attention probability is under 2^-16 of its block's largest
(Qwen3.5-4B 3.5e-4 of the groups, the same in MX int8); q.K and the head drop nothing in any run
counted. Where every kind is counted together (Phi-4-mini 2.2e-3 of the inputs, as much in its MX
int8 row: not the 4-bit kinds') MX - today on the tokens with a drop is -0.002 to -0.031.

**What ran where.** the simulation host (the dense models, the MoE models, the ablations; the priority slot
from 17:40) and the build host (Gemma 4 E2B-it and E4B-it), 2026-10-12, 2000 tokens, 6 BLAS threads, one
process a run: 45 s for LFM2.5-230M to 82 min for Qwen3.5-35B-A3B (25.5 GB); the 26B 49 and 38 min
(33 and 40 GB, the chat and docs/isa.md). The today rows' identity: ee8f118d's formats_scan on the
same text (docs/isa.md as the MX tree has it) and the same chat ids. The runs' numbers (each row's
MX - today, phase 1's, the difference, dKL from float, the tails and the drops) are
`tools/exp/mx_finals.json`.

## 4. Attention on shared lanes

**The rate.** A lane takes one output a cycle. q.K^T streams one K row (a token) a cycle and gives
an output per stationary row per K row when d = 128 (one block): G rows in decode (Qwen3-0.6B 2;
Qwen3.5-0.8B 4 with d = 256 = 2 blocks; Gemma 4 26B 2 / 8 with d 256 / 512; Qwen3.5-35B 8 with d
256), up to MCOLS in prefill (R x G rows). P.V's outputs take two blocks (256-token attention
blocks). So decode needs ~2 lanes for full rate, prefill MCOLS.

**The co-sim.** `OTPU_EPI_RATE = E` caps the MXU's drain at E results a cycle
(`rtl/mxu/otpu_mxu.sv`, `drain_run`; the rows wait in the result FIFO and the pops stall once it
is full, as for today's bank-conflict runs), passed through `OTPU_VDEFS` (`opentpu/rtlsim.py`).
Qwen3-0.6B fp4 with the int8 head, 2 layers, the YPCB memory path (LiteDRAM DDR3-1066, 133.33 MHz),
PAIR / DSTEP / STREAM on, `tools/perf_qwen.py` (`--batch B`: one row per KV slot); cycles and the
change against E = 0 (today's drain, MCOLS results a cycle):

| run | MCOLS | E = 0 | E = MCOLS/2 | E = MCOLS/4 | E = 1 |
|---|---:|---:|---:|---:|---:|
| prefill 4 rows, pos 544 | 4 | 372,383 | 373,700 (+0.4%) | | 467,233 (+25.5%) |
| prefill 4 rows, pos 544 | 8 | 267,269 | 267,269 (0) | 293,307 (+9.7%) | 404,693 (+51%) |
| the same, chunk 1024 | 8 | 268,146 | | 270,366 (+0.8%) | |
| prefill 8 rows, pos 544 | 8 | 502,925 | 503,044 (+0.02%) | 558,336 (+11.0%) | 786,573 (+56%) |
| prefill 4 rows, pos 4000 | 4 | 842,590 | 842,967 (+0.04%) | | |
| prefill 4 rows, pos 4000 | 8 | 740,566 | 740,566 (0) | 764,535 (+3.2%) | |
| the same, chunk 1024 | 8 | 741,125 | | 742,003 (+0.1%) | |
| batched 4 users + head, pos 544 | 4 | 1,696,116 | 1,694,384 (-0.1%) | | |
| batched 4 users + head | 8 | 1,593,159 | | 1,614,799 (+1.4%) | |
| batched 8 users, no head | 8 | 518,581 | 521,472 (+0.6%) | 570,702 (+10.1%) | |
| batched 8 users, no head | 16 | 429,052 | 429,052 (0: E >= rows) | 453,374 (+5.7%) | |
| batched 8 users, chunk 1024 (1) | 8 | 525,211 | | 528,844 (+0.7%) | |
| batched 8 users, chunk 1024 (1) | 16 | 425,265 | | 427,909 (+0.6%) | |
| prefill 16 rows, pos 200 (1) | 8 | 785,826 | 790,945 (+0.7%) | 900,573 (+14.6%) | |
| the same, chunk 1024 (1) | 8 | 810,064 | | 821,526 (+1.4%) | |
| prefill 16 rows, pos 200 (1) | 16 | 613,106 | 613,106 (0) | 660,953 (+7.8%) | |
| the same, chunk 1024 (1) | 16 | 607,149 | | 611,637 (+0.7%) | |
| decode + head, pos 544 | 4 | 1,471,170 | 1,471,170 (0) | | 1,482,874 (+0.8%) |
| decode + head, pos 4000 | 4 | 1,589,677 | 1,589,677 (0) | | 1,670,786 (+5.1%) |
| Qwen3.5-0.8B decode + head, 4 layers | 4 | 2,571,917 | 2,572,854 (+0.04%) | | 2,578,205 (+0.2%) |
| Qwen3.5-0.8B prefill 4 rows, 4 layers | 8 | 942,492 | 942,492 (0) | 1,006,839 (+6.8%) | |
| the same, chunk 1024 | 8 | 941,973 | | 950,872 (+0.9%) | |
| LFM2.5 prefill 4 rows, 4 layers | 8 | 271,253 | 271,253 (0) | 312,863 (+15.3%) | |
| the same, chunk 1024 | 8 | 269,796 | | 272,042 (+0.8%) | |

(1) A TMEM of 256K words (`OTPU_TMEM_WORDS`, 16 ACT rows), which 16 rows and chunks of 1024 at 8
rows need, with the sequencer's TMEM ranges widened (`OTPU_TAW=18`, below).

**A side finding: a TMEM above 64K words serializes.** `rtl/seq/otpu_seq.sv` keeps 16-bit TMEM
ranges in its dependency window (TAW = 16) and makes any access above 2^16 conflict with
everything. With 256K words the compiler places tiles there, so the same 8-row prefill took
853,599 cycles instead of 502,925 (bit-exact with the ISA, only serialized; 839,408 at 128K), and
502,398 with TAW = 18 (`OTPU_TAW`, an experimental define). A bigger TMEM (the TMEM study, the
scoreboard's TMEM what-ifs) needs TAW = clog2(TMEM_WORDS) in the sequencer.

**Where the E = MCOLS/4 cost is.** Not in attention: at MCOLS 8, 4 rows, E = 2 the attention phase
goes 182,531 -> 184,544 cycles (+1.1%), at 4000 tokens 655,852 -> 655,764 (0). It is in the MLP
(83,897 -> 107,922): `kernels/mlp.swiglu_down` accumulates the down projection in partial MMs over
F chunks of 256 (`_chunk`), two K-blocks each, one PAIR advance, so 4 outputs a cycle. An MM keeps
full rate while rows / E <= KB (KB / 2 with PAIR), so the partial MMs need chunks of at least
D x rows / E (twice that with PAIR). With E = MCOLS/2 today's 256 always meets it (rows <= MCOLS;
PAIR rows <= MCOLS/2); with E = MCOLS/4 it takes 512 or more. With chunks of 1024
(`OTPU_MX_CHUNK=1024`, an experimental override) the MLP's cost falls to +1.0% and the run's to
+0.8% (the "chunk 1024" rows). Bigger chunks need TMEM: batched 8 users and the MoE experts' 8
rows at chunk 1024 do not fit today's 64K words (section 6.4 runs them at 256K).

**Attention is VPU-bound.** One lane (E = 1) costs attention +28% (prefill) to +40% (decode at
4000 tokens: 202,137 -> 283,103 cycles); two or more cost nothing: the softmax (EXP2SUB, the row
sums, P's quantization) sets attention's pace, and q.K^T's results wait for it anyway (the
VPU-first arbiter finding). MCOLS 8 itself takes attention only from 219,275 to 182,531 (-17%) at
544 tokens and 689,473 to 655,852 (-5%) at 4000.

**So:**
- E = MCOLS/2 shared lanes: no measurable cost (0 to +0.7%).
- E = MCOLS/4: +0.1 to +1.4% against the same chunks with the down projection's chunks at
  D x rows / E or more (a compiler rule; it needs TMEM at 8 rows and more), +3 to +15% without.
  The bigger chunks are not free by themselves (16 rows at MCOLS 8: +3.1%; at 16: -1.0%).
- E = 1: no.
- A raw int32 score path (scores without scales, the scale folded into the softmax's pre-scale)
  or full-rate lanes for q.K^T buy nothing on these models: attention's MXU side is not the
  limit. K keeps its per-token fp32 (one block a token at d = 128).

## 5. Area and timing

`rtl/exp/mx_mxu/` (experimental, in no file list): `otpu_mx_x7.sv` (pumped DSP48E1 columns of two
rows, their column ends with the shift and the accumulating DSP, the 266.67 MHz clock),
`otpu_mx_us.sv` (the same on packed DSP48E2s, one clock), `otpu_mx_lane.sv` (a shared lane:
i2f48, the two fp32 multiplies, the RMW multiply-add), `mx_top.sv` (NCOL physical columns, 2 NCOL
rows, PAIR's integer add, the result FIFO, E lanes), `base_top.sv` (today's columns and per-column
epilogue, copied from `otpu_mxu`, with the same surroundings), `tb_mx.sv` (both families against a
reference model), `ooc.tcl` / `run_ooc.sh` (Vivado), `run_xsim.sh`.

**Bit-exact**: `tb_mx.sv` in xsim on the unisim models (the build host, 2026-10-11): 260 rows, 880 lane
outputs, 416 conversions, 0 mismatches, for the DSP48E1 (pumped) and the DSP48E2 (packed) columns.

**Synthesis estimate** (Yosys 0.64 `synth_xilinx -family xc7 / xcup -flatten` on a laptop, the
sources through sv2v; synthesis only, no placement or timing, and Yosys's LUT counts run above
Vivado's: use the differences). Both tops hold the columns, the column ends, the epilogue, the
result FIFO and the read-modify-write lanes; neither holds the ACT RAM, the issuer, the chunk and
scale FIFOs or TMEM (the same for both, except the ACT RAM's width, which follows the rows):

| family | design | rows | LUT | SRL | LUT RAM | FF | DSP |
|---|---|---:|---:|---:|---:|---:|---:|
| DSP48E1 (7-series) | today, MCOLS 4 | 4 | 15,713 | 6,300 | 576 | 12,049 | 552 |
| | **MX, 2 pumped columns, E 2** | 4 | **10,838** | 6,065 | 960 | **6,993** | **288** |
| | today, MCOLS 8 | 8 | 27,726 | 10,958 | 1,092 | 26,285 | 1,104 |
| | **MX, 4 pumped columns, E 4** | 8 | **17,942** | 10,593 | 1,728 | **13,855** | **576** |
| | MX, 4 pumped columns, E 2 | 8 | 14,865 | 10,499 | 1,728 | 12,478 | 564 |
| DSP48E2 (UltraScale+) | today, MCOLS 4 | 4 | 16,587 | 6,300 | 504 | 12,421 | 552 |
| | **MX, 2 packed columns, E 2** | 4 | **11,204** | 6,129 | 840 | **6,946** | **288** |
| | today, MCOLS 8 | 8 | 28,941 | 10,958 | 936 | 27,029 | 1,104 |
| | **MX, 4 packed columns, E 4** | 8 | **18,529** | 10,705 | 1,488 | **13,700** | **576** |

One lane: 1,167 LUTs, 100 SRLs, 546 flip-flops, 6 DSPs.

- **At today's rate** (4 rows) the MX datapath is 31% fewer LUTs, 42% fewer flip-flops and 48%
  fewer DSPs (-4.9K, -5.1K, -264): the per-column epilogue and drain lanes go, two shared lanes,
  the shifts and the accumulating DSPs come. Pumping alone accounts for -256 DSPs and ~-1.7K
  flip-flops of that (docs/dsp_pumping.md section 5, Vivado); the rest is the epilogue.
- **At MCOLS 8's rate** (8 rows, E 4) it is 35% fewer LUTs, 47% fewer flip-flops and 528 fewer DSPs
  than today's MCOLS 8 (DSP48E2: 36%, 49%, 528). Going from today's 4 to 8 rows costs +2.2K LUTs,
  +4.3K SRLs, +1.2K LUT RAM, +1.8K flip-flops and +24 DSPs on MX (the second activation skew and the
  column ends), against +12.0K, +4.7K, +0.5K, +14.2K and +552 for today's MCOLS 8.
- **NVFP4** adds 4 DSPs and 4 shifts a row to the column end (+16 DSPs at 4 rows, +32 at 8).
- The ACT RAM's read width doubles with 8 rows either way (64 -> ~128 BRAM36, docs/dsp_pumping.md
  4.3).

**Vivado out of context** (`run_ooc.sh`: Vivado 2026.1 synthesis, opt, place, phys_opt and
route. clk_s is 133.33 MHz. On the 7-series the pumped columns run at 266.67 MHz from the same
MMCM, with their DSPs' operand pins as two-fast-cycle multicycle paths. Inputs and outputs are
registered. The build host, 2026-10-12, 3 to 8 minutes a run):

| part | design | rows | LUT | LUT RAM | SRL | FF | DSP | slices | WNS clk_s | clk_f | clk_s -> clk_f | clk_f -> clk_s | WHS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| xc7k480t-2 | today, MCOLS 4 | 4 | 14,141 | 576 | 3,091 | 24,654 | 552 | 6,983 | +0.265 | | | | +0.020 |
| | **MX, 2 pumped columns, E 2** | 4 | **11,041** | 960 | 2,888 | **18,038** | **288** | **4,929** | +1.287 | +0.162 | +0.446 | +0.333 | +0.054 |
| | today, MCOLS 8 | 8 | 25,400 | 1,092 | 5,978 | 45,773 | 1,104 | 13,292 | +0.079 | | | | +0.050 |
| | **MX, 4 pumped columns, E 4** | 8 | **17,922** | 1,728 | 5,053 | **35,794** | **576** | **8,854** | +0.329 | +0.135 | +0.359 | +0.103 | +0.050 |
| | MX, 4 pumped columns, E 2 | 8 | 16,139 | 1,728 | 5,017 | 34,136 | 564 | 8,310 | +1.183 | +0.171 | +0.315 | +0.158 | +0.052 |
| | MX, 4 pumped columns, E 1 | 8 | 15,241 | 1,728 | 5,001 | 33,257 | 558 | 7,868 | +1.008 | +0.155 | +0.634 | +0.087 | +0.053 |
| xcku5p-2 | today, MCOLS 4 | 4 | 15,951 | 504 | 4,985 | 22,729 | 552 | 3,998 | +2.154 | | | | +0.010 |
| | **MX, 2 packed columns, E 2** | 4 | **12,722** | 828 | 4,897 | **17,324** | **288** | **2,890** | +2.997 | | | | +0.010 |
| | today, MCOLS 8 | 8 | 28,752 | 936 | 9,355 | 47,711 | 1,104 | 8,145 | +1.373 | | | | +0.010 |
| | **MX, 4 packed columns, E 4** | 8 | **22,084** | 1,488 | 9,317 | **32,686** | **576** | **5,225** | +1.517 | | | | +0.010 |
| | MX, 4 packed columns, E 2 | 8 | 20,506 | 1,488 | 9,218 | 32,900 | 564 | 4,985 | +1.400 | | | | +0.010 |

Slices are CLBs on the xcku5p. FF counts include the registered inputs (1,024 per row plus the
chunk) in both designs, so compare differences, not totals.

- **Every run meets timing, hold included, with 0 critical warnings.**
  - On the 7-series the pumped domain is again the tightest. Two paths set it, the same ones and
    the same margins as the single column of docs/dsp_pumping.md section 5:
    - the fast-cycle INMODE replicas, +0.135 to +0.171 ns;
    - the one-fast-period return from the chain ends into the column end's A / D registers,
      +0.087 to +0.333 ns.
  - The MX slow-clock paths have at least +0.33 ns. At 8 rows with 4 lanes the worst is the
    result FIFO's write enable fanning out to its three LUT RAM copies, which a replica fixes;
    the other MX runs have at least +1.0 ns. Today's MCOLS 8 has +0.079 ns (hi0 to the PEs'
    INMODE).
  - The new slow paths have room (`paths.tcl` on the routed checkpoints, 2 / 4 columns, ns):

    | path | 7-series | xcku5p |
    |---|---:|---:|
    | the column end's shift into the B port (6 levels) | +2.351 / +2.143 | +3.814 / +3.413 |
    | the accumulating DSP's OPMODE | +5.852 / +5.181 | +6.451 / +6.161 |
    | its P to the row registers | +6.012 / +5.374 | +6.360 / +6.382 |
    | the lane's i2f48 (13 levels on the 7-series) | +2.551 / +2.390 | +3.777 / +3.624 |
    | PAIR's 48-bit add | +4.802 / +3.373 | +6.139 / +5.601 |
    | the result FIFO's write | +2.168 / +0.329 | +4.695 / +3.593 |
    | the chain ends into the column end (7-series: one fast period; xcku5p: the packed split) | +0.334 / +0.103 | +4.592 / +4.190 |
  - On the xcku5p every path has at least +1.37 ns.
- **Against today, on the 7-series:**
  - 4 rows: -22% LUTs, -27% flip-flops, -48% DSPs, -29% slices.
  - 8 rows, 4 lanes: -29%, -22%, -48%, -33%.
  - 8 rows, 2 lanes: -36% LUTs and -37% slices.
  - On the xcku5p: -20%, -24%, -48%, -28% CLBs at 4 rows; -23%, -31%, -48%, -36% at 8 rows with 4
    lanes.
- **What goes:**
  - today's per-column epilogue (the two fmuls, PAIR's fadd, the loop fadd, the combine fadds and
    their delays): 5,879 LUTs, 3,432 flip-flops and 16 DSPs at 4 rows; 11,800, 6,734 and 32 at 8;
  - its MCOLS read-modify-write lanes: 1,730, 1,330 and 8 at 4 rows; 3,441, 2,646 and 16 at 8.
- **What comes:**
  - E lanes of ~920 LUTs, ~720 flip-flops and 6 DSPs each;
  - a fifth DSP in each row's column end. The shift adds no visible LUTs: the xcku5p's column end
    has 64 to 68 LUTs against today's 73, and the 7-series flattens it into the column.
- **Going from 4 to 8 rows on the 7-series:** MX with 4 lanes adds +6.9K LUTs, +3.9K slices and
  +288 DSPs; today's design adds +11.3K, +6.3K and +552.
- **Against the Yosys estimate above:** Vivado's LUT savings are smaller (-22% and -29% against
  -31% and -35%), its DSP savings the same, and its flip-flop totals higher on both sides (it
  keeps the input registers).

## 6. Workloads

### 6.1 Co-sim, the YPCB's memory path

MX arrays of 8 rows with 4 lanes and of 16 rows with 8, against today's MCOLS 4:

| workload | today, MCOLS 4 | MX 8 rows, E 4 | change | MX 16 rows, E 8 | change |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B fp4 prefill, 4 rows, pos 544 (2 layers) | 372,383 | 267,269 | -28.2% | | |
| Qwen3-0.6B fp4 prefill, 4 rows, pos 4000 | 842,590 | 740,566 | -12.1% | | |
| Qwen3-0.6B fp4 prefill, 16 rows, pos 200 (2) | 1,266,469 | 790,945 | -37.5% | 613,106 | -51.6% |
| Qwen3.5-0.8B fp4 prefill, 4 rows (4 layers) | 1,137,275 (1) | 942,492 | -17.1% | | |
| LFM2.5-230M fp4 prefill, 4 rows (4 layers) | 443,238 (1) | 271,253 | -38.8% | | |
| Qwen3-0.6B batched decode, 4 users + head | 1,696,116 | 1,593,159 | -6.1% | | |
| Qwen3-0.6B batched decode, 8 users, no head | 730,652 | 521,472 | -28.6% | 429,052 | -41.3% |
| Qwen3-0.6B decode + head, pos 544 / 4000 | 1,471,170 / 1,589,677 | unchanged | 0 | | |

(1) docs/dsp_pumping.md section 3 (the same tree and settings). (2) TMEM 256K words, TAW 18
(section 4). Prefill at 8 rows: section 6.2.

### 6.2 An ideal port

The BCU1525's slices: a 128-byte chunk a core cycle, 60 cycles of latency:

| workload | MCOLS 4 | MX 8 rows | change | MX 16 rows | change |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B prefill 4 rows | 365,469 | 256,787 (E 4) | -29.7% | | |
| Qwen3-0.6B prefill 8 rows | 719,211 | 493,946 (E 4) | -31.3% | | |
| Qwen3-0.6B batched 4 users + head | 1,652,489 | 1,543,794 | -6.6% | | |
| Qwen3-0.6B batched 8 users, no head | 715,821 | 504,995 (E 4) | -29.5% | 418,047 (E 0), 442,100 (E 4) | -41.6%, -38.2% |
| Qwen3-0.6B decode + head | 1,433,117 | 1,433,117 (E 2) | 0 | | |

### 6.3 Scoreboard

`tools/scoreboard.py` (the workloads agent's tree, e769bf79 and its edits since), each option
against its own board and clock:

| board | option | decode | prefill | batched | MoE | score |
|---|---|---:|---:|---:|---:|---:|
| YPCB | MX 8 rows | 1.00x | 1.38x | 1.74x | 1.00x | 1.25x |
| YPCB | MX 8 rows, TMEM x4, 16 ACT rows | 1.00x | 1.62x | 1.84x | 1.00x | 1.32x |
| YPCB | MX 16 rows, TMEM x4 | 1.00x | 2.09x | 2.78x | 1.00x | 1.55x |
| BCU1525 DDR4-2400, 200 MHz | MX 8 rows | 1.00x | 1.32x | 1.66x | 1.00x | 1.22x |
| BCU1525 DDR4-2400, 200 MHz | MX 16 rows, TMEM x4 | 1.00x | 1.93x | 2.50x | 1.00x | 1.48x |
| BCU1525 DDR4-2400, 200 MHz | port 256 B + MXU x2 | 1.41x | 1.50x | 1.63x | 1.28x | 1.45x |
| BCU1525 DDR4-2400, 200 MHz | port 256 B + MXU x2, 8 rows | 1.42x | 1.79x | 2.54x | 1.28x | 1.70x |

The scoreboard has no lane limit (E = MCOLS); the co-sim puts E = MCOLS/2 within 0.6% of that.
"MXU x2 / x4" is where the MX column matters beyond the row count: two products per DSP (pumped
on the 7-series, packed on DSP48E2) behind an epilogue that does not grow with them. The wider
port is docs/wide_dram.md's separate change; on the YPCB (DDR3-1066) it buys nothing, so its rows
are left out. Decode and MoE on the YPCB are bound by DRAM and the PCIe link (1.00x); MoE's
expert FFNs gain where they run (6.4).

### 6.4 MoE expert FFNs (`tools/moe_prefill_cosim.py expert`: one expert's FFN on M rows, cycles)

| model, M rows | MCOLS 4 | MX 8 rows, E 4 | 8 rows, E 2 | MX 16 rows, E 8 | 16 rows, E 4 | 16 rows, E 4, chunk 1024 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-35B-A3B, 1 | 13,454 | 13,458 | 13,458 | 13,466 | 13,466 | 13,446 |
| Qwen3.5-35B-A3B, 2 | 13,298 | 13,302 | 13,302 | 13,310 | 13,310 | 13,368 |
| Qwen3.5-35B-A3B, 4 | 25,895 | 13,850 | 17,795 | 13,858 | 13,858 | 14,510 |
| Qwen3.5-35B-A3B, 8 | 49,505 | 27,350 (2) | 35,624 | 16,052 | 20,451 | 17,816 |
| Qwen3.5-35B-A3B, 16 (1) | 104,784 | 56,406 | | 32,578 | 41,090 | 35,515 |
| Gemma 4 26B-A4B, 1 | 27,964 | 27,968 | 27,968 | 27,976 | 27,976 | 27,898 |
| Gemma 4 26B-A4B, 2 | 28,024 | 28,028 | 28,028 | 28,036 | 28,036 | 27,879 |
| Gemma 4 26B-A4B, 4 | 52,407 | 28,813 | 35,923 | 28,821 | 28,821 | 28,806 |
| Gemma 4 26B-A4B, 8 | | 56,408 (2, 3) | 57,167 (3) | | | 32,813 |
| Gemma 4 26B-A4B, 16 (1) | 209,931 | 109,506 | | 58,839 | 76,315 | 65,484 |

(1) TMEM 256K words, TAW 18 (section 4). (2) E 0 (not run at E 4; E = rows or more is no cap
up to 4 rows). (3) Chunks of 1024 (today's do not fit 64K words at 8 rows).

An expert's FFN streams its slot once (1.67 MB for the 35B, 3.4 MB for the 26B): 1-2 rows are bound
by the port at any row count; 4 rows halve at 8 rows (-47% / -45%); 8 rows take -45% at 8 rows and
-68% at 16; 16 rows -46% / -48% at 8 rows and -69% / -72% at 16 (E 8 = E 0 on all of them). On 16
rows, E = MCOLS/4 costs the experts +26% / +30% (+9% / +11% with chunks of 1024, which the 35B's F =
512 caps at 512): E = MCOLS/2. MoE decode (one row) does not change.

## 7. Recommendation and decisions

**Recommendation** (Vivado out of context closes the datapath at 133.33 MHz and the pumped
columns at 266.67 MHz, section 5):

1. **Make the MX column the MXU's next datapath**, with `int8x` and `fp4x6` weights and `a8x32`
   activations, exact accumulation in the column end, and E = MCOLS/2 shared epilogue lanes. It is
   at least as accurate as today on all three models (section 3), it costs nothing measurable in
   any workload (section 4), and it removes the per-column epilogue that makes MCOLS 8 expensive.
2. **Keep NVFP4, as a per-kind option, not as the default.** The column end with a DSP per
   16-long chain (8 + 1 a row, +4 DSPs a row against fp4x6's 5) runs NVFP4 exactly, and the same
   column end runs every other format (two chain ends with equal B). It pays where the formats
   scan says so (Qwen3, Qwen3.5: -1.8 to -2.1 points against fp4x6 for +5.9% of 4-bit bytes; on
   LFM2.5 only in the attention). A pumped column end (the two 16-chains of a sub-block in the
   two fast cycles) would do it with today's DSP count; not built. Drop OCP MXFP4.
3. **No special attention path.** E >= 2 hides q.K^T's one-block outputs behind the VPU-bound
   softmax. Attention's speed is the VPU's (the pumped-VPU study).
4. **Rows**: on the YPCB, the MX datapath first at today's 4 rows (it gives back ~4.9K LUTs and
   ~5.1K flip-flops of the 81%-full part), then 8 rows: the prefill / batched-decode / MoE-expert
   lever (-28 to -48% of those runs' cycles, decode unchanged) for a fifth of the LUTs (under half
   with the SRLs and LUT RAM) and an eighth of the flip-flops that MCOLS 8 costs today. 16 rows with
   a bigger TMEM (and the sequencer's wider ranges) on the bigger boards (scoreboard 1.48x on the
   BCU1525, before a wider port). Pumped DSP48E1s on the 7-series, packed DSP48E2s
   on UltraScale+.

**Decisions for the user** (not guessed here):

- **The numerics change.** MX changes MM's accumulation order and every format's scale; every image
  and reference is rebuilt and the token streams change (dKL equal or better on the three models
  measured). Today's fp4 (bf16 block scale) cannot run exactly on the MX column: its 8-bit
  mantissa times the 4-bit multiplier would leave Wf = 5 and drop terms, so fp4x6 replaces it
  (same bytes). Accept or not?
- **NVFP4 in hardware** (+4 DSPs a row in the column end, the formats scan choosing per kind) or
  not (then fp4x6 alone).
- **E**: MCOLS/2 (no measured cost; 2 lanes at 4 rows, 4 at 8) or MCOLS/4 (one lane, ~1.3K LUTs,
  fewer; needs the chunk rule and TMEM, and the MoE experts still lose 9-11% on 16 rows).
- **Card time**: a qualification on the card needs a session the maintainer schedules (none used here).
- **Decided (the user, 2026-10-12)**: the formats rule on the MX column takes MX int8 (`int8x`,
  `a8x32`) as its reference; the per-kind defaults are 3.6's table, with the 26B's experts in
  NVFP4 by the user's choice (+6.6% decode, estimated) for text far from the chat (3.6). Open:
  against MX int8, SmolLM3's `gateup=nvfp4r,down@9-26=nvfp4r` qualifies by 2.1 SE (+38% decode
  against MX int8, the default's +20%); a faster default, or not?

## 8. Implementation plan

Staged so that each step is checked before the next one costs Vivado time.

**1. ISA, simulator, compiler (no RTL).**
- `docs/isa.md`, `opentpu/isa.py`: MM flag bit 7 `MX` (free today) reinterprets WF: 0 int8x, 1 int4x
  (int4 with fp4x6's scales; not measured here), 2 fp4x6, 3 nvfp4. Scale words: fp4x6 / int4x one
  word a block (`dw` in [7:0], four 6-bit m above it); int8x one byte a block (4 blocks a word);
  nvfp4 two words a block (eight E4M3 bytes) and the row's reference exponent; Sw a word per row
  ahead of the row's scales. QACT flag bit 4 `X`: int8 with four 8-bit offsets in the block's ACT
  scale word and Sa in a per-row register (ASCALE's row); QST `X` for K of d > 128. MM semantics as
  in section 2: exact accumulation, the drop rule, y = rne(rne(i2f(acc) x Sw) x (Sa x 2^-Wf)), then
  ACC / ASCALE / RMAX as today; PAIR's two terms added as integers; UNIT = Sw 1, dw 0, m 1.
- `opentpu/isasim.py` (`_mm`, `_qact`, `_qst`) and `opentpu/fp32.py` (i2f of a 48-bit integer,
  round to nearest even).
- `opentpu/quant.py`: `quantize_int8x`, `quantize_fp4x6` (the multiplier search with m in 1..63),
  `quantize_nvfp4r` (from `tools/exp/mx_formats.py`), `quantize_mxu` / `mxu_wf` / `row_bytes` for
  them; `opentpu/llm/formats.py` names; `runtime.quantize_rows` X mode; every family's
  `emulated_logits` and formats_scan / quant_eval take the MX quantizers.
- `opentpu/kernels/mlp.py`: the chunk rule (D x rows / E, twice with PAIR) when E < MCOLS/2.
- Tests: `tests/test_isa.py` MX MMs against a numpy model (exact accumulation, sh = 0 / -1 at the
  drop boundary, the 48-bit bound at K = 65536, PAIR, UNIT, ACC / ASCALE / RMAX); `test_quant.py`
  and `test_formats.py` for the new formats; the formats gate (dKL on Qwen3-0.6B, Qwen3.5-0.8B,
  LFM2.5 at 2000 tokens no worse than today's); the ISA-level token-exact images rebuilt with new
  refcache keys.
- Before RTL: dKL on bigger models (Qwen3.5-4B, Gemma 4 E2B / E4B, the 26B / 35B MoE experts):
  their scale ranges set how often the window drops.

**2. RTL at today's 4 rows** (`rtl/mxu/otpu_mxu.sv` IMPL 3, the experiment's cells).
- The column end with the shift and the accumulating DSP (`rtl/tech/{xilinx7,xilinxus,generic}`),
  the 48-bit result FIFO, E shared lanes (`otpu_mx_lane`), PAIR's integer add, Sw from the scale
  stream, Sa per row; `rtl/vpu/otpu_quant.sv` X mode (per-32 amax, the offsets against the row's
  largest, the row's scale).
- `tests/test_rtl.py` / `test_quant_rtl.py` (Verilator): IMPL 3 against isasim, bit-exact, random
  MMs at MCOLS 4 / 8, E 1 / 2 / 4, every format; the `*_token_on_rtl` tests on the new refs.
- OOC of `otpu_mxu` IMPL 3 against IMPL 2 (the build host / the card host), then a FAST=1 board build at 100
  MHz (dev-builds rule) for function, then 133.33 MHz; card qualification in a session the maintainer
  schedules.

**3. 8 rows** on the YPCB: the pumped column (`rtl/exp/dsp_pump`, a second clock) or 8 physical
columns, ACT RAM 8 rows a cycle, E = 4; on UltraScale+ the packed DSP48E2 column (`otpu_mx_us`).
16 rows with 4x TMEM where the board has the RAM (BCU1525), with `otpu_seq`'s TAW widened
to clog2(TMEM_WORDS).

**Risks**
- **QACT's X mode** needs the row's largest scale before the offsets: ROW mode's two passes over
  the row instead of one. The co-sim did not model it (QACT timing unchanged); attention is
  VPU-bound, so P's quantization must not get slower. Alternative: the row scale from the previous
  MM's RMAX.
- **Drops on other models**: measured at most 7e-5 of the terms on three small models; a model
  with a wider spread of block scales loses more. Wf is set by the 18-bit B port on both DSP
  families.
- **MoE experts at 16 rows with E = MCOLS/4**: +26-30%, +9-11% with the chunk rule (the 35B's
  F = 512 caps its chunks); E = MCOLS/2 has no such case.
- **A TMEM above 64K words** needs the sequencer's TMEM ranges widened (TAW = clog2(TMEM_WORDS) in
  `otpu_seq`; `OTPU_TAW` here is a co-sim define only), or every access above 2^16 serializes.
- **The pumped clock domain** on the 7-series (docs/dsp_pumping.md section 5: +0.16 ns at 266.67
  MHz out of context; a full build not run).


## 9. The RTL (branch `mx-rtl-prod`, 2026-10-12)

The MX column replaces the MXU's previous datapath (no second IMPL: `MXU_IMPL` must be 2), at
MCOLS 4 on the YPCB and parametric in MCOLS (4 and 8 tested) and E. It is bit-exact against
`opentpu/isasim.py` (the ISA of b370889d) in Verilator.

**`rtl/mxu/otpu_mxu.sv`**
- **The array** is the systolic one unchanged (chains of CLS = min(16, D/8) products, weights
  hopping column to column, PAIR's select in `otpu_pe`), so each chain end is one group of D/8
  elements (a 4-bit sub-block's half at D = 128 is a chain: two chain ends a sub-block, CPG = 2
  at D = 256).
- **The column end** (`otpu_mxce` in `rtl/tech/{generic,xilinx7,xilinxus}`, replacing
  `otpu_colend`): eight multiplier DSPs, B = m << sh per group (the shift in fabric two cycles
  ahead, `bm2`), their products summed down the PCIN cascade in two halves, and a ninth DSP that
  accumulates the row (OPMODE P + PCIN, or PCIN at the row's first block) in 48 bits: 9 DSPs a
  column (the decision's "+4 a row against fp4x6's 5": NVFP4's 16-long groups each get one). sh =
  W - dw - da comes from the scale entry and the ACT RAM's record (11-bit signed; B = 0 when
  negative, the drop).
- **The scale reader.** The issuer asks port A for the 32-byte line holding each advance's block
  data (int8x a byte, int4x / fp4x6 a word, PAIR two; NVFP4 eight bytes, PAIR sixteen; RSC Sw) and
  for the row's Sw. Port A now returns the aligned 32-byte line (256 bits, `otpu_dram` /
  `otpu_native_dram`, the ASIC top's pins) instead of a word, so Sw shares a line with the row's
  first advance or the previous row's last when the layout allows; otherwise one request cycle
  without a chunk fetches it (a cycle a row: rows of KB >= 8 lose under 1/KB). A tag per request
  (128 in flight, LUT RAM) says where the data sits in the line; the response fills one entry a
  chunk ({Sw, the low block's data} and {the high block's data}: two block RAM FIFOs of 96 and 64
  bits).
- **The ACT RAM** record is 64 bits a (row, block): {da3, da2, da1, da0, Sa} (`otpu_actram`); one
  Sa a row per MM is the ISA's rule, so each column carries Sa' = Sa x 2^-W of the row's last
  block to the result FIFO (an exponent subtract, `fp_pow2m`, with fp_mul's flush; the simulation
  checks it against the multiply).
- **Row assembly**: the column ends' row sums are deskewed, PAIR's partner integer added
  (columns j and j + M), and the row (MCOLS x 48-bit integers, MCOLS Sa', Sw) written to the
  result FIFO: RF = 32 rows at MCOLS 4 (64 at 8), three LUT RAM copies with their own write
  pointer and enable replicas (the write fan-out was the experiment's tightest path).
- **E shared lanes** (`otpu_mx_lane`, E = MCOLS/2 by default, `MXU_E` / `OTPU_MXU_E`; above MCOLS
  it is MCOLS): i2f48 (round to nearest even), x Sw, x Sa', then the existing TMEM write or the
  ACC / ASCALE read-modify-write and RMAX. The drain picks up to E results of the head row a
  cycle (fewer at a TMEM bank conflict). The card builds E = 4, full rate at MCOLS 4
  (`targets/ypcb-00338.toml` [uarch] MXU_E: the board top, `run_vivado.sh` / `make bit MXU_E=`,
  the board model and `rtlsim.BOARD_UARCH` follow it; timing only, so the CFG_HASH stays
  89417856): see "The card's lanes" below.

**`rtl/vpu/otpu_quant.sv`**: QACT and QST in the MX form, two passes a row (section 8's risk):
pass 0 computes the row's amax and each group's (QACT D/4, QST D) into four group RAMs, the
scale unit (reciprocal) gives S and inv, QST writes its scale row (S, then four d bytes a cycle),
and pass 1 reads the row again and writes q8(x x (inv x 2^d)) with d from the group RAMs (inv x
2^d an exponent add). The ACT RAM gets {d, S} at each block's first chunk. The slice sizes the
group RAMs from the configuration (the power of two at or above max(4 ACT_BLOCKS, TMEM_WORDS /
D): 512 groups on the card), so every QACT within the ACT RAM's bounds and every QST whose row
fits TMEM fits: no new ISA bound.

**The row buffer.** The two passes cost a 4-row prefill +0.45% in co-simulation (below), so QACT
rows go through a FIFO of prescaled chunks (`otpu_quant` XD, the slice's `QUANT_XD`, rtlsim's
`OTPU_QUANT_XD`; default 1024 chunks of LANES x 32 bits, rows of up to 8,192 elements at LANES
8): pass 0 also pushes each chunk x 2^-e(amax) into it and the reader goes on to the next row's
pass 0 at once; pass 1 pops the row's chunks once its scale is out of the scale unit (no second
TMEM read or prescale). Up to four rows are between the passes: the group maxes and the rows' S,
inv and amax are kept by row in four banks (row r in r % 4; the scale unit carries the bank as
its tag), and a read waits for a free FIFO entry (credits: the counter's top bit) and for its
row's banks (pass 1 of row r - 4 done). The S of a block's ACT RAM record travels with its
write. QST, and QACT rows over XD x LANES elements, keep the two passes over TMEM; XD 0 leaves
the buffer out.

**Identity.** CAPS bit29 (MX) is set by every MX bitstream (`opentpu/host/regs.py` CAP_MX), and
`opentpu/target.py` hashes the numerics (`NUMERICS = "mx"`) with the Config. The card target's
ACT RAM now holds 16 rows (`targets/ypcb-00338.toml` ACT_ROWS = 16; the board Makefile and
`run_vivado.sh` take it from the target, the fake and the board model follow it): its CFG_HASH is
89417856 (ff270e6e at ACT_ROWS 4; e170f967, the previous numerics at 4 rows, is production
91a9c2d3's), so a host of the previous numerics refuses an MX bitstream wherever it checks the
hash (`board.device_config`: ConfigMismatch), and the reverse; `BoardBackend` and otpu-serve
check only that the program fits (the MX compiler adds the hash check).

**Port A** returns the aligned 32-byte line (256 bits) in `otpu_dram`, `otpu_native_dram`, the
slice, the board shell and the ASIC top's pins.

**Scale-read statistic** (simulation): `otpu_mxu` counts its port A line requests and those that
fetch a row's Sw alone; rtlsim's `stats["mxu_scale"]` has the totals, and `tools/perf_qwen.py`
prints them and adds both to its MM-by-source table (per MM from the `MXS` lines under +trace).

**Tests** (Verilator against isasim, every one bit-exact):
- `tests/test_quant_rtl.py`: `test_mm_mx_rtl` over 12 cases (int8x, int4x, fp4x6, NVFP4; D 32 /
  128; MCOLS 4 / 8; E 1 / 2 / 4; PAIR, RSC, k0, sh = 0 and -1 forced at the window's edge, Sw
  corners), each run plain, UNIT + RMAX, ACC + RMAX and ASCALE; `test_mm_mx_replay_rtl` (M >
  MCOLS, up to ACT_ROWS 16 at D 128); `test_quantizer_modes_rtl` (QACT and QST in every mode at
  D 32 / 128, LANES 8 / 16, read back by UNIT and int8x MMs; with the row buffer, without it,
  and with a buffer of one row and a third, where the reads wait for entries); the ties tests
  for QACT and QST.
- `tests/test_rtl.py`: the fuzzers, the scoreboard stress, replay and count tests on MX MMs
  (every format, PAIR from DUP rows, RSC, k0), the hazard program's QST scale rows.
- `tests/test_tech.py`: the xilinx7 `otpu_mxce` on Yosys's DSP48E1 model equals the generic one
  at CPG 1 and 2.

**Out of context** (Vivado on the build host, xc7k480t-2 at 7.5 ns, D 128, MCOLS 4, DEPTH 1024, LANES
8): `otpu_mxu` LUT 12,311, LUT RAM 4,071, FF 22,472, DSP 562, BRAM36 33, WNS +0.164 against
b370889d's 15,704 / 3,656 / 24,545 / 554 / 32, WNS +0.051. Both are limited by the same pop_q ->
`otpu_pe` INMODE fan-out. The new paths' slacks: shift into B +4.69, chain end into A +0.98, the
accumulator's OPMODE +4.62, shift registers +3.49, scale decode +2.47, row add +3.50, the lanes'
i2f48 +3.61, result FIFO write +2.35, drain pick +1.61, scale line +4.69, scale FIFO +2.79.
`otpu_actram` at 4 and 16 rows: 9,361 / 9,629 LUTs, 68 BRAM36 both, WNS +0.879 / +1.341.

**Board builds** (YPCB-00338, DDR3-1066, ACT_ROWS 16, CFG_HASH 89417856, post-route):

| | dev dd372036 (FAST, 100 MHz) | 999b0bc0 (133.33 MHz) | MX production 8b33206e (133.33 MHz) | production 91a9c2d3 |
|---|---|---|---|---|
| WNS / WHS (ns) | +0.025 / +0.012 | +0.059 / +0.023 | +0.007 / +0.013 | +0.013 / +0.012 |
| core_clk WNS (ns) | +0.074 | +0.070 | +0.007 | |
| LUT | 161,221 | 172,276 | 178,529 | 177,292 |
| FF | 128,929 | 134,996 | 138,466 | 134,913 |
| BRAM tiles (RAMB36 + RAMB18 / 2) | 607.5 | 607.5 | 615 | 602.5 |
| DSP | 722 | 722 | 734 | 716 |
| `u_mxu` LUT / FF / DSP | 17,553 / 18,444 / 562 | 19,889 / 19,965 / 562 | 23,766 / 22,838 / 574 | 25,082 / 21,823 / 554 |
| `u_quant` LUT / FF / RAMB18 | 7,184 / - / 4 | 7,635 / 8,511 / 4 | 8,621 / 8,765 / 5 (+ 7 RAMB36) | 7,924 / 6,969 / 0 |
| `u_act` LUT / RAMB36 | 722 / 68 | 1,321 / 68 | 1,338 / 68 | 958 / 64 (+4 RAMB18) |
| `u_mem` LUT / FF | 12,715 / 3,935 | 14,101 / 3,954 | 13,625 / 3,953 | 14,328 / 3,767 |

The 133.33 MHz build's core paths at the limit are the sequencer's (`u_seq` q_cmd -> c_fp,
+0.070, unchanged code), port A's line (`u_mem` aoh -> a_rdata, +0.077, 96% route) and the ACT
RAM's write data (+0.081, route). Port A's 256-bit line costs about +170 FF in `u_mem`. Neither
build has the row buffer (two passes); it adds 15 RAMB18 (7.5 BRAM36: Yosys synth_xilinx of
`otpu_quant` at D 128, LANES 8, XD 1024 against XD 0, the four group RAMs unchanged at one
RAMB18 each), +561 FF and +371 LUT.

**The production build, 8b33206e** (2026-10-12: E 4, the row buffer at XD 1024, 133.33 MHz,
DDR3-1066, Gen2 x8, full effort; BUILD_ID 0x8B33206E, CFG_HASH 89417856; 58,358 slices, 78.2%).
Against production 91a9c2d3: LUT +0.7%, FF +2.6%, BRAM +12.5 tiles (the row buffer's 9.5 in
`u_quant`, the ACT RAM's 16 rows), DSP +18 (the column ends' ninth DSP and E 4's lanes against the
previous MXU's 554). The first route missed by 4 ps (WNS -0.004 on one endpoint: `u_seq` q_cmd
w4 -> c_fp, 20 levels with 10 CARRY4, 74% route), the sequencer's footprint of an MM, which MX
lengthened: KB in w4[23:16] with k0 beside it, and the scale rows' bytes by format. It closed
without a netlist change (same BUILD_ID): the routed checkpoint reopened, the userclk1 reroute
first, then post-route phys_opt (AggressiveExplore) replicating the path's high-fanout nets: WNS
+0.007, WHS +0.013 (userclk1 +0.030, userclk2 +0.081, LiteDRAM sys +0.111). phys_opt before the
reroute did not close it (core WNS stayed at -0.048 on IMEM's address into its BRAM, a route the
userclk1 reroute fixes). The fallback, not built, is branch `mx-fpcut` (15da5b56): each footprint
product carries its addend into the sequencer's sum stage, so the queue stage does one base +
length add; same cycles, and `footprint()` equal to 8b33206e's by a Yosys SAT miter at (D, S) =
(128, 1), (32, 1), (128, 2). Sign-off on the frozen tree in Verilator: the token_on_rtl tests (11)
and the six new family tests, `test_models_on_board_rtl` 10/10, and `test_rtl` / `test_board` /
`test_quant_rtl` at `OTPU_MXU_E=4` (283 passed, the MM fuzzers included).

**In production** (2026-10-12, 17:49 on the card host): `deploy_mx_8b33206e` (build 8b33206e,
host e37709e1: this branch with tiling's default-on tiled prompt runs and expert-major MoE
prefill), promoted after its qualification: selftest all pass at Gen2 x8; `qual.sh` fast 53 checks
and Gemma 4 E2B-it's token-exact run 33, no FAIL; otpu-chat on 8 models and chatmoe on the 3
streamed MoE models (LFM2.5-8B-A1B, Qwen3.5-35B-A3B, gemma-4-26B-A4B-it, the MX pools)
token-exact against the ISA simulator; the H2C stress with no flag. Against the previous
production (mt4, 91a9c2d3): decode cycles a token -1.4% to -1.9% on the int8 models, +3.5% on
Qwen3.5-4B and +5.8% on LFM2-2.6B (their NVFP4 kinds' bytes, section 3.6); the streamed MoE
models' decode tok/s +7% on the 35B (fp4x6 experts), -6% to -8% on the 26B and -9% to -10% on
LFM2.5-8B, whose NVFP4 experts are bigger: fewer slots fill the DRAM, so more experts miss. The
deploy directory's README has the full table.

**QACT's cost** (co-simulation, `tools/perf_qwen.py`, Qwen3-0.6B, 2 layers and the LM head,
against b370889d's per-block streaming QACT, the MM unchanged):

| | decode token | 4-row prefill |
|---|---|---|
| b370889d | 1,537,293 | 1,577,203 |
| two passes (XD 0) | 1,537,739 (+0.029%) | 1,584,347 (+0.453%) |
| row buffer (XD 1024) | 1,537,477 (+0.012%) | 1,577,931 (+0.046%) |

QACT start to end, b370889d / two passes / row buffer: 2x128 96 / 158 / 101 cycles, 4x384 272 /
558 / 295, 4x1024 632 / 1,198 / 695, 4x2048 1,208 / 2,222 / 1,335; a 1-row QACT keeps the second
read's latency (1x1024 200 / 310 / 308). The MMs asked port A for 1,461,408 scale lines in the
decode token, none for a row's Sw alone (today's compiler's layouts).

**The card's lanes: E 4.** A single-block MM (q.K^T at d = 128: KB 1) gives G outputs a K row,
G a KV head's query rows, and the drain takes E of the head row's a cycle: at E 2 a row of 3 or
4 takes two cycles against the array's one. With ACT_ROWS 16 at MCOLS 4 a head of G > 4 rows
replays each chunk ceil(G/4) times, so the array's time a K row is KB x ceil(G/4) and the
G 8 models (Qwen3.5-35B, Gemma 4 E2B / E4B at d 256: KB 2) balance at E 2; P.V's 256-token blocks
(KB 2) balance at G 2 to 8. On the card the drain bounds q.K^T for G 3 (Phi-4-mini) and G 4
(SmolLM3) at d 128. Decode co-simulations (`tools/perf_qwen.py`, 2 and 4 layers and the LM head,
LiteDRAM DDR3-1066 at 133.33 MHz, PAIR / DSTEP / STREAM; the previous numerics: ee8f118d's RTL
and compiler; a full token = the head + the layers x the 4- minus 2-layer difference):

| model (G, d) | | layer at 2000 | layer at 4000 | full token at 2000 | at 4000 |
|---|---|---:|---:|---:|---:|
| Phi-4-mini (3, 128) | previous | 881,660 | 916,436 | 33.35M | 34.46M |
| | MX E 2 | +0.30% | +2.94% | -0.12% | +2.13% |
| | MX E 3 / E 4 | -2.09% | -1.78% | -2.15% | -1.88% |
| SmolLM3-3B (4, 128) | previous | 676,358 | 696,982 | 26.55M | 27.29M |
| | MX E 2 / E 3 | +0.29% | +1.68% | +0.06% | +1.34% |
| | MX E 4 | -1.12% | -1.10% | -1.23% | -1.21% |

The 2-layer runs at 2000 tokens of Qwen3-0.6B (G 2: 1,613,955 cycles at E 2 and E 4, 1,644,467
before), Qwen3.5-0.8B (G 4 at d 256: 2,487,917 at E 2 and E 4, 2,540,395 before) and Gemma 4
E2B (G 8 at d 256: 3,986,760 at E 2, 3,986,572 at E 4, 4,077,912 before) do not move with E.
Phi-4-mini's and SmolLM3's layers without the drain's excess are faster than the previous
numerics' (and the LM head 2.5% smaller), but at E 2 the excess, about T x Hkv cycles a layer,
grows with the position, and both models' decode is slower than before by 4000 tokens. E 3
(bit-exact with the ISA simulator, `perf_qwen --check`) fixes G 3 only. Out of context (`otpu_mxu`, MCOLS 4,
ACT_ROWS 16, 7.5 ns): E 2 12,311 LUT / 22,472 FF / 562 DSP (WNS +0.164), E 3 14,116 / 23,824 /
568 (+0.206), E 4 16,171 / 25,113 / 574 (+0.067, the same pop_q -> otpu_pe INMODE path), about
1,930 LUT, 1,320 FF and 6 DSP a lane. The card builds E 4. Packing the drain across rows (G 3:
1.5 cycles a row) or lanes without ACC / ASCALE would cost less area but fix less (G 4 needs four
results a cycle), and splitting a head's rows across MMs keeps G x T / E.

**Attention decode at MCOLS 8: which E.** A K row of q.K^T costs the array KB = d / 128 cycles
(no replay while G <= MCOLS), the drain ceil(G / E), and the KV cache (d + 4) bytes (int8 and the
row's Sw) at about 120 bytes a cycle (94% of DDR3-1066's 128 bytes a 133.33 MHz cycle); P.V's
rows (V^T, 256-token blocks: KB 2) cost the drain the same ceil(G / E) against 2. At MCOLS 8, E =
MCOLS / 2 = 4 (from the shapes; no runs):

| model | G (sliding / global) | d (sliding / global) | q.K^T a K row: drain / array / KV | bound at E 4 | cost at 4K |
|---|---|---|---|---|---|
| Qwen3-0.6B | 2 | 128 | 1 / 1 / 1.1 | KV | 0 |
| Phi-4-mini | 3 | 128 | 1 / 1 / 1.1 | KV | 0 |
| SmolLM3-3B | 4 | 128 | 1 / 1 / 1.1 | KV | 0 |
| LFM2-2.6B, LFM2.5-8B-A1B (230M: G 2) | 4 | 64 | 1 / 1 / 0.6 | the array (a half block, any E) | 0 |
| Qwen3.5-0.8B / 2B / 4B | 4 | 256 | 1 / 2 / 2.2 | KV | 0 |
| Qwen3.5-35B-A3B | 8 | 256 | 2 / 2 / 2.2 | KV (the drain level with the array) | 0 |
| Gemma 4 E2B | 8 / 8 | 256 / 512 | 2 / 2 / 2.2, 2 / 4 / 4.3 | KV | 0 |
| Gemma 4 E4B | 4 / 4 | 256 / 512 | 1 / 2 / 2.2, 1 / 4 / 4.3 | KV | 0 |
| gemma-4-26B-A4B | 2 / 8 | 256 / 512 | 1 / 2 / 2.2, 2 / 4 / 4.3 | KV | 0 |
| G 5 to 8 at d 128 (the test's G 6; Llama-3-70B and Qwen2.5-72B shapes: G 8) | 5-8 | 128 | 2 / 1 / 1.1 | the drain, at every context | +0.9 T Hkv cycles a layer |

No card model's attention decode is drain-bound at MCOLS 8 / E 4, at any context: the drain and
the KV cache both grow with T, so the bound is the shape's, not the position's. The shapes that
are, G 5 to 8 at d 128, run q.K^T at 2 cycles a K row against the cache's 1.1: at 4K that is
+3,700 x Hkv cycles a layer, about +40% on the attention's cache time (q.K^T and P.V 2.2 cycles
a K row become 3.1); `tests/test_perf.py` keeps the design point's G 6 case at E = MCOLS / 2 as a
strict xfail. E 4 also costs prefill at d 128: an 8-row q.K^T drains in 2 cycles a K row against
the array's 1 (section 10's table at 16 rows: 4x at E 4). The MCOLS 8 integration tree should
build E = MCOLS (8): every G up to the column count and every prefill q.K^T at full rate, as the
card at MCOLS 4 does, for 4 lanes more (about 7.7K LUTs, 5.3K FFs, 24 DSPs; this section's
1,930 / 1,320 / 6 a lane). E = 4 is the area fallback, at no card model's decode cost but at 2x
on prefill q.K^T at d 128 and on decode of G 5 to 8 at d 128.

## 10. MCOLS 16 on the YPCB, out of context

Sixteen rows a pass on eight pumped DSP48E1 columns: section 5's experiment (`rtl/exp/mx_mxu`
`mx_top` on `otpu_mx_x7`, `run_ooc.sh k7-mx8e4 k7-mx8e8`: clk_s 133.33 MHz, the pumped columns at
266.67 MHz, inputs and outputs registered) and the production tree's `otpu_actram`
(`tools/synth/ooc_rows.py --mcols 16 --rows 16 --tops otpu_actram`), on 8b33206e with
`rtl/exp/dsp_pump` (the build host, 2026-10-12, 5 to 7 minutes a run). The 4-column run again
reproduces section 5's row to the LUT.

| design (xc7k480t-2) | rows | LUT | LUT RAM | SRL | FF | DSP | slices | WNS clk_s | clk_f | clk_s -> clk_f | clk_f -> clk_s | WHS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MX, 4 pumped columns, E 4 (section 5) | 8 | 17,922 | 1,728 | 5,053 | 35,794 | 576 | 8,854 | +0.329 | +0.135 | +0.359 | +0.103 | +0.050 |
| **MX, 8 pumped columns, E 4** | 16 | **29,402** | 3,264 | 9,710 | **62,611** | **1,128** | **15,527** | +0.198 | +0.165 | +0.315 | +0.122 | +0.047 |
| MX, 8 pumped columns, E 8 | 16 | 33,041 | 3,264 | 9,653 | 68,618 | 1,152 | 17,554 | +0.217 | +0.160 | +0.409 | +0.031 | +0.024 |

- **Every run meets timing, hold included.** Eight columns do not tighten the pumped domain: its
  worst path is still a fast-cycle INMODE replica (+0.165 / +0.160 ns against +0.135 at four
  columns). The one-fast-period return from the chain ends into the column end has +0.122 at E 4
  and +0.031 at E 8 (the denser placement). The slow clock's worst is the result FIFO's write (the
  write enable at E 4, a write-address replica at E 8), which `otpu_mxu` already splits into one
  enable and address a copy (section 9).
- **From 8 to 16 rows** (E 4): +11.5K LUTs, +26.8K flip-flops (about 8.7K of them the eight more
  rows' registered inputs), +552 DSPs (eight columns of 128 pumped PEs, the column ends' five DSPs
  a row: 1,024 + 80 + 4 x 6) and +6.7K slices.
- **A lane at 16 rows** (E 4 -> E 8, divided by four): 910 LUTs, 1,502 flip-flops, 6 DSPs and 507
  slices (section 5's 8-row lane: ~920 LUTs, ~720 flip-flops). Its pick is a 16:1 multiplexer
  of the head row's results.
- **The ACT RAM at MCOLS 16** (ACT_ROWS 16, the 64-bit record, D 128, 128 blocks): 272 RAMB36,
  9,702 LUTs, 2,207 flip-flops, WNS +0.203 ns at 7.5 ns, against 68 RAMB36 and 9,629 LUTs at
  MCOLS 4 (section 9): the read width, four times the rows, takes four times the RAMs.
- **The fit, estimated from 8b33206e** (734 DSPs, 615 block RAM tiles, 58,358 slices, 78%): with
  this 16-row E 4 datapath in place of today's MXU, about 1,290 DSPs (67% of 1,920) and 615 - 68 +
  272 = 819 block RAM tiles (86% of 955). The slices are for the full build to show, with fmax:
  the 16-row datapath alone takes 15.5K here (8.9K at 8 rows), while today's MXU is a 4-row,
  unpumped one with its issuer, FIFOs and TMEM ports, which these tops leave out.

**What the 272 RAMB36 hold** (`otpu_actram`). One copy of distinct rows: row r is only in column
r mod MCOLS (group r div MCOLS), a column being a bank of D / 32 pieces of 32 bytes (four 64-bit
RAMB36 a piece) and a 64-bit record RAM. At ACT_ROWS 16 the store is 16 rows x 128 blocks x (128
+ 8) bytes, 256 KiB of int8 and 16 KiB of records, at MCOLS 4 and at MCOLS 16 alike. What grows
is the read: the array takes MCOLS x D bytes a cycle (each column its own row's block) and a
RAMB36 port gives 8, so the banks are MCOLS x D / 8 data RAMB36 (64 at MCOLS 4, 256 at 16) plus
MCOLS record RAMs. At MCOLS 4 they are full (4 groups x 128 blocks = 512 entries); at MCOLS 16 a
quarter full (128 of 512): the 4x is the port's width, not copies. The one row held twice is QACT
DUP for PAIR (M <= MCOLS / 2: row r also in column r + M, which reads the odd blocks), a second
write into columns the MM would leave idle and no RAM of its own (`tools/dataflow.py` counts its
bytes twice in act_wr; docs/tiled_prefill.md 11.2). So:
- **The idle depth is free capacity.** The same 272 RAMB36 hold ACT_ROWS 64 at MCOLS 16 (four
  groups, full depth), and replay streams each weight row from DRAM once for all the MM's rows
  (group 0 pops a row's chunks, the later groups read the same FIFO entries): a quarter of the
  weight bytes a prompt row against ACT_ROWS 16, with no more block RAM (the array's cycles and
  the replay rule KB <= DEPTH unchanged; dataflow's ledger, fp4x6 layers: Qwen3-0.6B 14.9 ->
  3.7 MB a row, SmolLM3-3B ~94 -> ~23, Qwen3.5-4B ~120 -> ~30). It pays only with a 64-row tile,
  and today's TMEM (64K words) stops the tile kernel at 8 to 16 rows at MCOLS 16 (compile checks:
  Qwen3-0.6B 16 rows at position 544, 12 at 4000; SmolLM3 8): it needs about 4x the TMEM a row,
  or S > 1 slices splitting the outputs (docs/dataflow.md 4.12).
- **Fewer RAMs take a faster port, not fewer copies.** Reading at the pumped clock (266.67 MHz,
  two half-rows a slow cycle into a register stage a bank; not built) gives 16 bytes a RAMB36 a
  slow cycle: 128 data and 8 record RAMB36 at MCOLS 16, 136 freed (half depth at ACT_ROWS 16,
  full at 32). The records (16 x 128 x 64 bits) could also leave block RAM for LUT RAM (16 freed,
  a few thousand LUTs).
- **32 columns:** 4,096 bytes a cycle, 512 + 32 RAMB36 at the slow clock (256 + 16 pumped), full
  at ACT_ROWS 128. On an ASIC the store is ACT_ROWS x K bytes in macros MCOLS x D bytes wide, so
  its area follows the bits held (256 KiB at ACT_ROWS 16, for any MCOLS) and the port width sets
  the macros' aspect, not their count.
- **S slices:** when slices split an MM's outputs, each gathers the activations and quantizes them
  into its own ACT RAM, so the rows are held S times. One store would feed the slices' arrays in
  lockstep from one MCOLS x D-byte read a cycle over a registered broadcast tree (16 Kbit wide at
  MCOLS 16), freeing S - 1 stores at the price of one slice's stall stalling all. The YPCB builds
  S 1.

**The drain at 16 rows.** An output costs the array KB cycles (a 128-element chunk a block) and
the drain ceil(M / E) (the head row's M live results, E a cycle; bank conflicts aside): drain /
array, at ACT_ROWS = MCOLS = 16 (no replay up to 16 rows):

| MM | rows M | KB | E 4 | E 8 |
|---|---:|---:|---:|---:|
| decode q.K^T, G 2 to 4 at d 128 (Qwen3-0.6B, Phi-4-mini, SmolLM3) | 2-4 | 1 | 1.00 | 1.00 |
| decode q.K^T, G 4 at d 256 (Qwen3.5-0.8B, Gemma 4 E4B) | 4 | 2 | 0.50 | 0.50 |
| decode q.K^T and P.V, G 8 at d 256 (Gemma 4 E2B, Qwen3.5-35B-A3B) | 8 | 2 | 1.00 | 0.50 |
| decode P.V (256-token blocks), G <= 4 | <= 4 | 2 | 0.50 | 0.50 |
| a 16-row q.K^T at d 128 (prefill) | 16 | 1 | **4.00** | 2.00 |
| a 16-row q.K^T at d 256 or P.V (prefill) | 16 | 2 | **2.00** | 1.00 |
| a 16-row projection, d_in >= 1024 | 16 | >= 8 | <= 0.50 | <= 0.25 |

Decode is never drain-bound at E 4 on the card's models; full 16-row attention passes are (4x at
d 128, 2x at d 256 and in P.V), and E 8 halves that. The projections are not, at any E.

**A lane's parts** (production `otpu_mxu`, MCOLS 4: Vivado's E 2 -> E 4 difference per lane,
1,930 LUTs, 1,320 flip-flops and 6 DSPs, split by Yosys `synth_xilinx` with the hierarchy kept
(its LUT counts run above Vivado's) and, for the logic in `otpu_mxu`'s top, by counting bits):

| part | LUT | FF | DSP | SRL |
|---|---:|---:|---:|---:|
| i2f48 (sign / magnitude, 48-bit leading zeros, normalize, round to nearest even) | 319 | 188 | 0 | 2 |
| Sw / Sa' delays to the multiplies | 0 | 0 | 0 | 64 |
| x Sw and x Sa' (two `otpu_fmul`) | 250 | 128 | 4 | 0 |
| ACC / ASCALE's multiply-add (`otpu_fmadd`: fmul 124 / 64 / 2, fadd 484 / 180, the addend's delay 64 FF) | 608 | 308 | 2 | 1 |
| the old value and the factor registers, ASCALE's factor select | ~32 | 64 | 0 | 0 |
| the read-modify-write's address, value and column through its 6 stages | ~0 | ~75 | 0 | ~72 |
| RMAX: the value's key against each column's (2 x MCOLS 32-bit compares), the candidate OR | ~250-280 | ~75 | 0 | 0 |
| the pick (the head row's integer, Sa', column and TMEM address, MCOLS:1) and its registers | ~120 | ~121 | 0 | 0 |
| the address and column alongside the lane (7 stages) | ~0 | 40 | 0 | 40 |
| the lane's TMEM port multiplexer | ~64-100 | 0 | 0 | 0 |

The pick and RMAX grow with MCOLS (at 16: a 16:1 pick, ~350-450 LUTs; RMAX ~1K LUTs a lane unless
it leaves the lane); the result FIFO is shared (MCOLS x 80 + 32 bits a row: 32 rows in three LUT
RAM copies, ~700 LUT RAM at MCOLS 4; 64 rows of 1,312 bits, ~5.2K, at 16). A lane without ACC /
ASCALE saves the multiply-add, its registers and its bookkeeping: ~640 LUTs, ~450 flip-flops,
2 DSPs.

The pumped PEs' multicycle constraint: these runs source `boards/ypcb-00338/constraints/
otpu_clk2x.tcl` (tmem-pump's 2x clock, 07297ddd) with an MX section for `otpu_pump_pe` (from the
core clock through x0 / x1 / w: two fast cycles, -end); in the out-of-context netlist its
core-clock lookup (the register driving a PE's x1[0]) found none, so `ooc.tcl` set the
experiment's own multicycle on the DSPs' A / B / D pins (OTPU_C2X 0), the constraint of section
5's rows.

**Next release (a pumped MXU), not r2:** r2 builds an unpumped MXU (MCOLS 8, E 8), so the 2x
clock's MX section is left for the release that pumps the columns. It is not on any branch:
`otpu_clk2x.tcl` exists only on tmem-pump's line (07297ddd to 930b3db7, not on main), with the MX
section still a placeholder. What it needs: find `otpu_pump_pe`'s cells by module and their
`clk_f` clock as above, but take the core clock as the start clock of a timing path through the
first PE's x1 pins (`get_timing_paths -through`, falling back to the clocks of their fan-in's
start points, `all_fanin -startpoints_only`) instead of following x1[0]'s net to its driver;
then the multicycle from that clock through x0 / x1 / w to `clk_f` (2 -setup -end, 1 -hold
-end). The proof is section 5's 4-column run with the section sourced instead of `ooc.tcl`'s
own multicycle: OTPU_C2X 1, and clk_f +0.135, clk_s -> clk_f +0.359, clk_f -> clk_s +0.103.
`rtl/exp/dsp_pump` (the pumped column the experiment reads) is not on main either.
