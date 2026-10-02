# Gemma 4 on openTPU (gemma-4-E2B)

openTPU runs a fourth model family: Google's Gemma 4, checked with
[gemma-4-E2B](https://huggingface.co/google/gemma-4-E2B) (the text decoder; the vision and
audio encoders are not loaded). It uses the same ISA, bitstream, W8A8 / W4A8 numerics and host
driver as Qwen3, LFM2 and Qwen3.5. No new hardware op: everything Gemma 4 needs beyond them
is composed from the existing ones, including the per-layer embeddings, which are gathered on
the card by the MXU (`opentpu/kernels/gather.py`). The model code is `opentpu/llm/gemma4.py`.

```sh
hf download google/gemma-4-E2B --local-dir models/gemma-4-E2B
python3 -m pytest -q tests/test_gemma4.py                 # tiny random model (+ --runslow: RTL)
python3 tools/perf_qwen.py --model gemma4 --layers 0,1,2,3,4,5,6,7,8,9,15,16,17,18,19 --pos 600 \
    --wformat fp4 --head-format int8 --resident --ddr 1066 --mhz 133.33 --check
```

## The model

Gemma 4 E2B has 35 layers, hidden size 1536, a 262,144-token vocabulary and a tied LM head
(the embedding table, scaled by sqrt(1536) on input).

- **Attention.** 8 query heads on 1 KV head, RMSNorm on each q, k and v head (v without a
  weight), attention scaling 1 (the norms set the scale). 28 layers are sliding (the last 512
  tokens, the current one included; 256-wide heads, RoPE theta 1e4); every fifth layer is
  global (all tokens, 512-wide heads, "proportional" RoPE with theta 1e6 that rotates only the
  pairs (i, i + 256), i < 64).
- **KV sharing.** The last 20 layers compute no K or V: they attend to the cache of the last
  earlier layer of their kind (layer 13, sliding; layer 14, global).
- **MLP.** GeGLU, `gelu_tanh(W_gate x) * (W_up x)`, 6144 wide, 12288 in the 20 KV-shared layers.
- **Per-layer embeddings (PLE).** A second table, 262,144 x (35 x 256) (4.7 GB in bf16), gives
  each layer a 256-vector per token: `pli = (RMSNorm(W_proj x0 / sqrt(H)) + PLE[t] * 16) /
  sqrt(2)`, and after the MLP each layer adds `RMSNorm(W_p (gelu_tanh(W_g x) * pli_l))`.
- **Norms.** Plain RMSNorm `x * w` (not Gemma 3's `1 + w`) before and after every block, then
  the per-layer input, then `x *= layer_scalar` (0.018 to 0.88).
- **Final soft cap.** `30 tanh(z / 30)` on the logits: monotonic, so greedy decoding picks the
  same token; the host applies it (`gemma4.softcap`).

## How it maps

**The embedding and PLE rows are gathered on the card.** A resident decode program takes only
the token id and the position as run arguments (6 of the 8 argument words); nothing per token
comes from the host. The gather is the MXU's own dequantization: a table row streams as a
matrix of its D-blocks against a one-hot stationary operand (row j of one-hot block k is 1 at
element MCOLS k + j), so one MM gives element MCOLS k + j of every block, `(i2f(127 w) ws) *
f32(1/127)`, and D / MCOLS MMs give them all (4-bit: twice as many, the one-hot blocks
interleaved with zero blocks to pick a chunk's first or second block). The operand is a
constant in DRAM, quantized into ACT RAM once per token.
- The embedding is a row of the tied int8 LM head (32 MMs of 12 blocks into a transposed
  [128, 12] tile, then one COPY VOP per block).
- The PLE table is stored for gathering: one 9,344-byte record per token (int8), element i in
  block i % 70 at position i // 70, so the MMs write the row in order with no copies (70 is 2
  mod 4: the MMs' output rows fall in distinct TMEM banks). The table's scale (16 / sqrt(2)) is
  folded into the records, the 1 / sqrt(H) and 1 / sqrt(2) of the projection into W_proj and
  the norm weight.
- At the token's start, one pass computes every layer's per-layer input (the 8960 x 1536
  projection, a norm per layer, plus the gathered row) into a DRAM area that each layer loads
  its row of.

Prefill runs and per-position programs gather their rows on the device too: the host compiles
the run's token ids into the program (`compile_rows(tokens=)`, `compile_step(tok=)`), which
gathers each row's embedding and PLE row (the PLE rows through the I/O area, which the layers
load a group at a time) and loads the RoPE rows of its positions from the table. The host does
no model math. The gathers read only the token's rows, each once per one-hot block (32 MMs):
the embedding row with its scales 32 x 1,584 B = 51 KB, the PLE record 32 x 9,344 B = 299 KB,
and the int8 one-hot operand once (64 KB), about 0.41 MB a token against 1.43 GB of weights
(0.03%). `dequant_row` / `dequant_records`
are the gathers' host twins, bit for bit, for tests (`host_inputs`: the host-written rows of the
same programs without tokens). The final soft cap is monotonic, so greedy decoding (argmax)
needs none and the stored logits are the raw ones; a sampler on the card (autodecode's
`m.lm_sink`) gets the capped logits, `kernels.lib.softcap` applied to each LM-head chunk before
the sink when the spec has a `softcap`.

**The sliding window over a KV ring.** Sliding layers keep K / V in a ring of window + one
attention block, 768 slots, position p in slot p mod 768. A token at position p attends over
three 256-token blocks of the ring: the window's first block is masked at its start, its last
at its end, the middle one is full (`attention.Blocks`). The masks at a run-time position are
computed on the card: the position within the block as a float from an iota table, then
`((tpos - c) + 0.5) * 2^100 * 2^100` is +inf where token c counts and -inf where not, and its
negation for the start block. The extra block means a prefill chunk's appends never overwrite
a slot an earlier row of the chunk still reads. Global layers keep every position (Bucket, as
Qwen3), and the KV-shared layers read their source layer's cache.

**Few argument words.** K rows (with each row's block scales right after its data) are 1,536
bytes apart, as are the rows of the RoPE table (sliding cos, sin, global cos, sin), so a
run-time position moves both with one argument word: the token x 1536 (embedding row), x 48
(its scales), x 9344 (PLE record), the position x 1536 (K row, RoPE row), x 4 and x 1.

**Hardware loops.** The layers run as loops over their repeated unit (`lfm2.plan` over each
layer's attention kind, own K / V and MLP width): (4 sliding + 1 global) x 3 with their own K /
V, then the same unit x 4 sharing it. Each of the four layer kinds has a DRAM block of its own
size; a loop steps one address register by its unit's size.

**Program size and prefill.** Inside each unit's loop the four sliding layers loop again (nested
hardware loops), so a program holds 4 layer bodies, not 10: decode is 1,412 instructions at
position 600 and 1,581 at 1500 (IMEM holds 4,096). A prefill run takes 4 rows, the ACT RAM's
rows: an MM with more rows than that streams every weight again (`Image.fit_rows`, which
`qwen3.Engine` takes as its limit). 4 rows fit TMEM and IMEM up to position 2048: the queries
are normed and RoPE'd in place and the attention output is written over them, the per-layer
inputs are made a row at a time, and the MLP's F chunk is at most 768 (`MLP_CHUNK`, so the
12288-wide MLPs run in 16 chunks). A 512-token prompt is 128 runs of 4 (2,087 to 2,493
instructions), and a 2048-token prompt 512 runs (3,840 at the end).

**The DRAM image** (fp4 layers, int8 LM head, int8 PLE, 4096 tokens, 8 prefill rows, the
resident decode's tables): 3.651 GiB of the card's 4 GiB.

| Area | Address | Size |
|---|---|---:|
| I/O (8 rows: inputs, logits 1 MiB a row, per-layer inputs, mask rows) | 0x000000000 | 8.6 MiB |
| PLE projection (8960 x 1536, fp4) | 0x00089dc00 | 7.0 MiB |
| layers, run by run (K / V caches 39.8 MiB) | 0x000f97000 | 984.9 MiB |
| LM head (int8, the embedding table too) | 0x03e874000 | 396.0 MiB |
| PLE records (262,144 x 9,344 B) | 0x057474000 | 2,336.0 MiB |
| lookup tables (RoPE rows, iota, one-hot operands) | 0x0e9474000 | 6.1 MiB |
| program area (after the image) | 0x0e9a85000 | 0.125 MiB |
| free | | 357 MiB |

At 2048 tokens the image is 3.636 GiB (372 MiB free). The on-card decode loop's area (about
5 to 7 MiB with its three 1 MiB logit vectors) fits in what is free. With an fp4 PLE table
(`OTPU_PLE_FORMAT=fp4`) the records are 4,864 bytes and int8 layers fit too (3.424 GiB); the
default picks int8 whenever the image fits 4 GiB. The build
quantizes the matrices in worker processes (`OTPU_BUILD_JOBS`, default 4), the 4-bit ones through
the image caches' disk cache (`opentpu/qcache.py`, [board.md](board.md); int8 matrices and the
PLE records are quantized at each build): 324 s cold on omarchy. An image with the fp4 PLE table
(explicit int8 layers) spends most of its build on the table's 4-bit search: 697 of 773 s on one
core, about 10 minutes with four jobs on the card host, at every build. Caching those records
(branch qcache-ple: `gather.pack_records` through `opentpu/qcache.py`, rows stored without their
padding, prebuild covering such runs) would take 1.27 GB of the card host's disk; it is parked
while that disk is at its floor, as only an explicit `--wformat int8` E2B pays it.

## Accuracy

The numpy reference (`gemma4.reference_logits`) matches Hugging Face's fp32 model (transformers
5.17) to 6.5e-5 in the logits on three prompts, with the same argmax.

Greedy decoding of the real model on the ISA simulator (board configuration, resident decode,
fp4 layers, int8 LM head, int8 PLE) against Hugging Face's greedy decoding (fp32, the PLE table
in bf16): **all 72 tokens equal**, 3 prompts x 24 tokens (2026-09-30):

| Prompt (after `<bos>`) | Tokens | Same | Continuation |
|---|---:|---|---|
| `The capital of France is` | 5 + 24 | all 24 | ` Paris.\n\nThe capital of France is Paris.\n\n...` |
| `def fibonacci(n):` | 6 + 24 | all 24 | `\n    if n == 0:\n        return 0\n    elif n == 1:\n        return` |
| `The quick brown fox` | 4 + 24 | all 24 | ` jumps over the lazy dog.\n\nThe quick brown fox ...` |

The ISA simulator takes 11 to 16 s a token.

On the card (production image `deploy_champ_e698dcd7`, 133.33 MHz, 2026-09-30), the prompt fed
by resident decode steps (the device gathers every input): the same 72 tokens with the int8 head
and with the fp4 head, and refs.py's prompt (13 tokens) + 32 greedy tokens equal the ISA
simulator's with both heads. With prefill runs of 4 rows (ISA simulator, the tokens compiled in)
the int8 head gives the same 72 tokens; the fp4 head takes HF's third choice at the first
prompt's third token, a three-way near tie (`\n\n` 24.62, ` It` 24.52, ` The` 24.39 after the
cap).

On build B (`deploy_fused133c_79c5707a`, production since 2026-09-30), 2026-10-01, with both
heads, the card equals the ISA simulator on every decode path: after a prefill run of the
11-token prompt `The lighthouse keeper climbed the stairs at dusk, and`, its first token and 24
tokens each by resident decode steps, per-position steps, and the card's generate loop, greedy
and sampled (temperature 0.8, top-k 40, top-p 0.95; `Engine.generate_card`).

**Long context.** After 900 tokens of prose (past the 512-token window and the 768-slot ring;
ISA simulator, fp4 layers), the device follows HF's greedy tokens for 5 tokens with either head,
then takes ` your` for HF's ` Lizzy`. HF's gap there is 3.46 before the soft cap (0.44 after). A
float64 emulation of the whole sequence with the quantization points switched on one at a time
(`tools/gemma4_quant_eval.py step`, the batched twin of `emulated_logits`) says which point moves
it, in logits before the cap:

| Weights | Activation points | ` Lizzy` | ` your` | Top |
|---|---|---:|---:|---|
| float | none | 52.34 | 48.89 | ` Lizzy` (HF's 28.2233 after the cap, exactly) |
| float | all: matmul inputs, K / V, P | 51.88 | 48.80 | ` Lizzy` |
| int8 layers, int8 head and PLE | all | 51.64 | 49.03 | ` Lizzy` |
| fp4 layers, int8 head and PLE | none | 39.95 | 48.01 | ` your` (` Lizzy` 8th) |
| fp4 layers, int8 head (the device) | all | 39.40 | 47.75 | ` your` (` Lizzy` 8th) |
| fp4 layers, fp4 head | all | 39.87 | 47.65 | ` your` (` Lizzy` 8th) |

The fp4 layers move it; the activation quantization, the int8 K / V and P and the head's format
do not. The device's pick is the emulation's: a quantization error of the 4-bit layers, not a
kernel's. (int8 layers do not fit the card: 4.65 GB.) Over the whole vocabulary at that step the
emulated logits have cosine 0.974 (int8 head) and 0.971 (fp4 head) with the float64 model's. By
kind: the attention projections in int8 with the MLPs and PLE projections in fp4 put ` Lizzy` back
on top (46.80 against 46.50); the MLPs in int8 with the attention in fp4 do not (` Lizzy` 6th).

Over the whole text, the next-token NLL of the soft-capped logits at its 899 positions
(`tools/gemma4_quant_eval.py nll`; every activation point on):

| Layers | Head | Weight bytes / token | ppl | Top-1 = float64's |
|---|---|---:|---:|---:|
| float64 | float64 | | 4.236 | 1 |
| int8 (4.65 GB: does not fit) | int8 | 2.37 GB | 4.351 | 0.952 |
| fp4 (the default) | int8 | 1.43 GB | 5.629 | 0.770 |
| fp4 | fp4 | 1.23 GB | 5.852 | 0.746 |
| fp4, attention int8 | int8 | 1.57 GB | 5.129 | 0.818 |

As on the other models ([quant.md](quant.md)), the int8 head is the default: the fp4 head costs 4%
in perplexity for 14% of decode speed. Attention in int8 would recover a third of the 4-bit
loss for 9% more bytes.

Per layer ([formats.md](formats.md), 2000 tokens of docs/isa.md): the recommended mix
(`wformat="mix"`, otpu-chat's default) is `attn@15-24=fp4,mlp@15-34=fp4`, the KV-shared layers'
MLP and the attention of layers 15-24 in fp4, 30.0% faster than int8 on the card (8.30 tok/s on
fmvf, session mix5) for dKL +2.60% (SE 0.06), 6.7 SE under its bar; fp4 layers are about +21%.
The first pick, `attn@15-34=fp4,mlp@15-34=fp4`, measured 8.47 tok/s on the card (32.5%) for
+3.34%: 1.1 SE over its bar. The PLE table and its lookup stay on the card: int8 where the image
leaves room, else fp4. The mix keeps the int8 table at 2048 and 4096 tokens (3.955 and 3.973 GiB,
46 and 28 MiB spare); int8 layers leave none (4.52 GiB with the int8 table), so an explicit
`wformat="int8"` takes the fp4 table: perplexity 19.15 against 18.62 with the int8 table on the
host (float 18.60), dKL +2.42% (SE 0.08). The host table (`OTPU_PLE_HOST=1`, 0.3% slower on fmvf)
is the accurate int8 the mixes are measured against, an opt-in reference, not a default.

The mix runs three layer runs (format boundaries at 15 and 25) with six layer bodies, where the
first pick and int8 run two with four: more than the two layouts the other families allow,
taken because its programs compile (resident decode 2579 instructions, 20,632 of the IMEM's
32,768 words at bucket 16). Its prompt runs ([prefill.md](prefill.md)) hold fewer rows; R_max
by bucket at a 4096-token cap (main 1f9e69e; `!` where today's route fits more rows, so a prompt
touching that bucket takes today's route):

| Weights | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|:--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|
| int8 | 4 | 4 | 4 | 4 | 4 | 3! | 3 | 3 | 3! | 3 | 3 | 3! | 3 | 3 | 3! | 3 |
| `attn@15-34=fp4,mlp@15-34=fp4` | 4 | 4 | 4 | 4 | 4 | 4 | 3! | 3 | 4 | 3! | 3 | 4 | 3! | 3 | 4 | 3! |
| the mix | 4 | 3! | 3 | 3 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |

On fp4 MMs (PAIR) a run of R = 2 rows costs the same per row as R = 4; int8 MMs stream their
weights once a pass of up to 4 rows, so at R = 2 they take half the rows per stream.

The limit is IMEM: a prompt run's program grows with its bucket (attention is unrolled per
block), and the mix's three runs add about 1160 instructions to the first pick's at bucket 2
(4106 at 4 rows, 10 over the 4096; the first pick's 2946). Bucket 2 is uncovered, so a prompt
past 256 tokens takes today's route, which compiles every run's program on the host. On the card
(session mix5, fmvf, cap 2048) a 1500-token prompt takes 536 runs: 100.3-101.4 s to the first
token warm (98.7 cold), 69.3 s of it on the device (a run 129-146 ms at 4 rows in buckets 1-2,
149-152 at 3 in 3-4, 109-110 at 2 in 5-6) and 58 ms a run on the host, where today's compile of
the next run outlasts the device's. The first pick would take prompt runs, 375 of 4 rows:
estimated 62 s (59.4 on the device by the weights model above, the rest of a row as the mix's
measured runs; 1.9-3.3 s for the prompt runs' whole-bucket attention; 0.7 ms a run on the host).
Prompts within bucket 1 (up to 256 tokens) take prompt runs of 4 rows, as the first pick's. With
bucket 2 covered the mix would take prompt runs too (estimated 72 s); with attention looped over
blocks instead of unrolled ([prefill.md](prefill.md) 7), 4 rows in every bucket. The fix, in the
prompt runs, is in progress.

## 26B-A4B: accuracy

gemma-4-26B-A4B is the family's mixture of experts: 30 layers ((5 sliding + 1 global) x 5),
hidden 2816, a dense GeGLU MLP (2112 wide) beside 128 routed experts (704 wide, top 8) in every
layer, global layers with K = V and 2 KV heads of 512, no per-layer embeddings. Its experts
stream from the host into slots on the card ([offload.md](offload.md), section 11). Numbers here
are from omarchy, 2026-10-01.

**The reference.** `reference_logits` against Hugging Face in fp32 on a truncation of the
checkpoint (`tools/gemma4_hf_check.py --layers 0,5`, both before the soft cap): max |diff|
0.0017 at logits of ~583, cosine >= 0.9999999, the same argmax at 11 of 11 tokens. In float the
900 tokens of *Pride and Prejudice* give NLL 0.2109 (ppl 1.235): the model knows the text.

**The experts' input.** The MoE block's norm gain, pre_feedforward_layernorm_2's g2, is ~0 on
the residual's outlier channels and large elsewhere (layer 10: the three largest channels of
the unit-norm input average 17.3, 10.2 and 9.5, where g2 is -0.002, 0.0 and 0.24; g2's max is
92.5, 7.6x its rms). The first device design quantized the unit norm once, for the router and
the experts, with g2 folded into the experts' gate and up columns: every int8 / fp4 block that
holds an outlier channel's column then quantizes the other columns coarsely, and the
outlier's activation meets that error. moe.moe_ffn now quantizes the norm times g2 for the
experts (`g_exp`, QACT with a scale), the unit norm for the router (router.scale is flat:
30.5-33.75, so its fold costs nothing; top 8 = float's 0.929 folded, 0.930 unfolded, over 30
layers x 64 tokens).

The block's relative error against float, on the float model's MoE inputs (64 tokens), the
float routes; the emulation (`tools/gemma4_quant_eval.py`), g2 folded / on the activation, and
the device (`tools/gemma4_moe_check.py`: moe_ffn alone on the ISA simulator, against float
experts on its own routes):

| Experts, activations | Layer 10 | Layer 20 | Layer 29 |
|---|---:|---:|---:|
| int8, float: folded / on the activation | 0.176 / 0.007 | 0.448 / 0.009 | 0.317 / 0.007 |
| int8, int8: folded / on the activation | 0.263 / 0.015 | 0.506 / 0.018 | 0.354 / 0.014 |
| fp4, int8: folded / on the activation | 0.684 / 0.094 | 0.911 / 0.121 | 0.873 / 0.104 |
| int8, int8: the device | 0.0163 | 0.0189 | 0.0145 |
| fp4, int8: the device | 0.0942 | 0.1216 | 0.1038 |

The dense MLP's error is 0.013-0.042 in int8 and 0.068-0.135 in fp4. The device matches the
emulation on its routes to 0.003-0.004 (int8) and 1e-4 (fp4). The whole model on the ISA
simulator (all 30 layers, int8 everywhere, 8 expert slots a layer, the experts streamed by the
host's server, the first 16 tokens of the text fed in): mean NLL over those 16 positions 1.593
against float's 1.601, the argmax float's at 15 of them; with g2 folded 2.508, 9 of 16.

**Weight formats.** The next-token NLL over the 899 positions of the text
(`gemma4_quant_eval.py nll`, every activation point on; the experts' format with
`--formats experts=...`), ΔNLL against float:

| Dense layers | Experts | Head | NLL | ΔNLL | ppl | Top-1 = float's |
|---|---|---|---:|---:|---:|---:|
| float | float | float | 0.2109 | | 1.235 | 1 |
| int8 | int8 | int8 | 0.2107 | -0.0001 | 1.235 | 0.996 |
| int8 | int8 | fp4 | 0.2227 | +0.0119 | 1.249 | 0.993 |
| int8 | fp4 | int8 | 0.2352 | +0.0243 | 1.265 | 0.982 |
| int8 | fp4 | fp4 | 0.2439 | +0.0331 | 1.276 | 0.984 |
| fp4 | int8 | int8 | 0.2611 | +0.0503 | 1.298 | 0.971 |
| fp4 | fp4 | int8 | 0.3101 | +0.0992 | 1.364 | 0.958 |

The model knows this text (ppl 1.235), which can hide a format's cost. The first 900 tokens of
docs/offload.md, written for this repository in September 2026 (a text no model has seen;
float ppl 10.02):

| Dense layers | Experts | Head | NLL | ΔNLL | ppl | Top-1 = float's |
|---|---|---|---:|---:|---:|---:|
| float | float | float | 2.3045 | | 10.02 | 1 |
| int8 | int8 | int8 | 2.3179 | +0.0134 | 10.15 | 0.951 |
| int8 | fp4 | int8 | 2.3380 | +0.0334 | 10.36 | 0.939 |
| int8 | fp4 | fp4 | 2.3404 | +0.0359 | 10.39 | 0.920 |

- int8 costs nothing on Austen and +0.013 on the new text.
- fp4 experts cost +0.024 over int8 on Austen and +0.020 on the new text; the fp4 head on top
  of them +0.009 and +0.002; fp4 dense layers (attention and the dense MLP) +0.050 (Austen),
  and with fp4 experts +0.099, more than the sum.

**The choice: int8 dense layers, fp4 experts, the fp4 head.** The dense layers and the head
set the card's room for expert slots: with fp4 experts (3.45 MB a slot; cap 4096, the lookup
tables) int8 dense layers leave 420 slots (14 a layer) with the int8 head and 540 with the fp4
head, fp4 dense layers 660 and 780; a decode token reads 1725 MB of int8 dense layers or 911 MB
of fp4, and 761 MB of int8 head or 392 MB of fp4. offload's event model (`cachesim.py`, four
2048-token texts, decayed-use slots, the host of [offload.md](offload.md) section 11.3), tok/s
(relative: the 35B ran about 15% under the same model on the card):

| Dense layers / head | Slots | Misses / token (of 240) | Gen1 | Gen2 (2.8 GB/s) | All resident |
|---|---:|---:|---:|---:|---:|
| int8 / int8 | 420 | 83.8 | 2.66 | 3.54 | 4.26 |
| int8 / fp4 | 540 | 67.5 | 3.19 | 4.11 | 4.79 |
| fp4 / int8 | 660 | 55.5 | 3.70 | 4.77 | 5.64 |
| fp4 / fp4 | 780 | 45.9 | 4.45 | 5.67 | 6.62 |

Against the bar of about +0.01 NLL per +10% decode rate: the fp4 head buys +20% for +0.009
(Austen) / +0.002 (new text); fp4 dense layers +39% for +0.075, twice the bar. fp4 experts
about double the rate of int8 experts (offload.md section 11.3) for +0.020-0.024; int8 experts
stay an opt-in for accuracy.

Two earlier runs with g2 folded are void: int8 layers, fp4 experts and the int8 head gave ppl
127.3, fp4 layers 138.4.

## Performance

One resident decode token on the Verilator RTL of the board configuration (PAIR, DSTEP, STREAM:
the production image's), through the board's memory path with the DDR3-1066 bank model at
133.33 MHz (`tools/perf_qwen.py --model gemma4 --layers 0,...,9,15,...,19 --pos 600 --wformat
fp4 --head-format int8 --resident --ddr 1066 --mhz 133.33 --check`, 2026-09-30). The full model's
image (3.6 GiB) does not fit the simulated memory (2 GiB), so the run takes 15 of the 35 layers:
two own units and one KV-shared unit, the nested loops included, with the full LM head. DRAM
after the token is bit-identical to the ISA simulator's.

- **6,827,804 cycles**, 88.3% of the core-port roofline; the useful bytes (806,978,416: weights,
  scales, KV, I/O) at 92.3% of DDR3-1066's peak, 15.76 GB/s. MXU MAC 88.1% of the cycles,
  starved 0.4%, blocked 9.4%; 436 s of simulation.
- By phase: LM head 3.354 M cycles (97.0% of its roofline), MLP 2.450 M (96.0%), attention
  0.903 M (69.1%), per-layer inputs 0.111 M (66.8%), the gathers 9,161 cycles.

The whole model moves 1.43 GB a token at position 600: the MLPs 827 MB, the attention
projections 148 MB, the LM head 415 MB (int8 with its scales), the per-layer input projections
22 MB, K / V 17 MB. At the phases' measured efficiencies that is about 12.2 M cycles, **about
10.9 tokens/s at 133.33 MHz** (a projection).

On the card (build B, `deploy_fused133c_79c5707a`, 2026-10-01; tools/qual/perf.py's method: 64
greedy tokens after the 512-token Austen prompt, the host's argmax in the loop; the prompt in
prefill runs, 4 rows each where they fit TMEM; cap 2048):

| Head | Image | Decode, device | Decode, wall | Mcycles / token | DRAM read / token | DRAM while running | Prefill (runs) |
|---|---:|---:|---:|---:|---:|---:|---:|
| int8 | 3.643 GiB | 10.57 tok/s | 10.53 tok/s | 12.611 | 1475 MB | 15.61 GB/s (92%) | 32.1 tok/s (128) |
| fp4 | 3.456 GiB | 12.14 tok/s | 12.09 tok/s | 10.981 | 1274 MB | 15.49 GB/s (91%) | 29.9 tok/s (139) |

The MXU starved 1% of the running cycles. The card's generate loop (`tools/decode_profile.py
--card-loop`, 64 tokens; the card samples and goes on, the host only reads the tokens):

| Head | Greedy, device / wall | Mcycles / token | Sampled, device / wall | Mcycles / token |
|---|---:|---:|---:|---:|
| int8 | 11.01 / 10.83 tok/s | 12.105 | 10.92 / 10.74 tok/s | 12.206 |
| fp4 | 12.73 / 12.52 tok/s | 10.476 | 12.61 / 12.23 tok/s | 10.576 |

With the PLE table on the host instead (`OTPU_PLE_HOST=1`, E4B's slot and fence:
[gemma4_e4b.md](gemma4_e4b.md)), the int8 head's greedy loop takes 12.173 M cycles a token:
the wait for the host's row costs 0.56% (68K cycles, 0.51 ms). Wall is 10.83 tok/s both ways.
(The int8 head's sampled run ended after 58 tokens.)

The projection above (12.2 M cycles) is 3% optimistic on build B. On the previous image,
`deploy_champ_e698dcd7` (2026-09-30), it was 12% optimistic: the int8 / fp4 heads ran 13.842 /
12.106 M cycles, 9.63 / 11.01 tok/s at 83 / 82% of the peak. On that image LiteDRAM's
controller was the limit, where the DDR3 bank model is about 10% optimistic ([board.md](board.md);
`perf_qwen.py --ldc` co-simulates the controller).

## Tests

`tests/test_gemma4.py`, with a tiny random Gemma 4 (9 layers: (two own sliding, an own
global) x 2, a shared sliding and two shared globals, so a loop inside a loop and a loop over
shared layers; 128 / 256-wide heads, 8 query heads on 1 KV head, a double-wide MLP, a 128-wide
per-layer input):

- The KV sources, MLP widths, rotated pairs and layer loops from the config (written by
  transformers 5.14 or 5.17); the numpy reference against Hugging Face (1e-4).
- The device against Hugging Face (cosine > 0.985) and against the quantization emulation
  (> 0.99; the emulation itself is at 0.991 against Hugging Face: int8 noise over 9 random
  layers).
- PLE records round trip (int8 and fp4).
- Resident decode (gathers and run-time masks) bit-exact against per-position programs (int8 /
  int8 and fp4 / fp4), across the first bucket boundary, and across the ring's wrap
  (positions 760 to 775).
- Chunked prefill bit-exact against token-by-token decoding (logits and caches).
- Device inputs: a 4-row prefill run and a per-position program with their tokens compiled in
  (the device gathers the rows and loads the RoPE rows) bit-exact against the same programs with
  host-written inputs (`Image.host_inputs`, the gathers' host twins).
- `--runslow`: a token at position 600 on the Verilator RTL through the board's memory path,
  per-position and resident, bit-exact against the ISA simulator (logits and DRAM).
- With models/gemma-4-E2B: the real model's resident decode programs fit IMEM with 6 argument
  words.

On omarchy (transformers 5.17, PAIR / DSTEP / STREAM) all 14 pass, in 16 minutes.
