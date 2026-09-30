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
quantizes the matrices in worker processes (`OTPU_BUILD_JOBS`, default 4) into a cache keyed by
the checkpoint and the quantizer's sources (`OTPU_QCACHE`, default `~/otpu-build/qcache`): 324 s
cold on omarchy, 8 s cached.

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
simulator's with both heads.

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

On the card (production image, 2026-09-30; tools/qual/perf.py's method: 64 greedy tokens after
the 512-token Austen prompt, the host's argmax in the loop; the prompt by resident decode steps):

| Head | Image | Decode, device | Decode, wall | Mcycles / token | DRAM read / token | DRAM while running | Prompt (steps) |
|---|---:|---:|---:|---:|---:|---:|---:|
| int8 | 3.629 GiB | 9.63 tok/s | 9.60 tok/s | 13.842 | 1475 MB | 14.22 GB/s (83%) | 9.93 tok/s |
| fp4 | 3.441 GiB | 11.01 tok/s | 10.98 tok/s | 12.106 | 1274 MB | 14.05 GB/s (82%) | 11.40 tok/s |

The projection was 12% optimistic: the DDR3 bank model it rests on is about 10% optimistic at
133.33 MHz, where LiteDRAM's controller is the limit ([board.md](board.md); `perf_qwen.py --ldc`
co-simulates the controller). The MXU starved 1-2% of the running cycles.

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
- `--runslow`: a token at position 600 on the Verilator RTL through the board's memory path,
  per-position and resident, bit-exact against the ISA simulator (logits and DRAM).
- With models/gemma-4-E2B: the real model's resident decode programs fit IMEM with 6 argument
  words.

On omarchy (transformers 5.17, PAIR / DSTEP / STREAM) all 13 pass, in 15 minutes.
