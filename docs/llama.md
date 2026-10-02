# SmolLM3 and Phi-4-mini on openTPU

openTPU runs two Llama-style decoders of 3-4B parameters:
[SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) and
[Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct). They are Qwen3
without the RMSNorm on each q and k head, so they run on Qwen3's code (`opentpu/llm/qwen3.py`):
its DRAM image, kernels, resident decode and chunked prefill. `opentpu/llm/llama.py` only reads
their configs. The ISA, the RTL, the bitstream and the W8A8 numerics are the same as for the
other models.

```sh
hf download HuggingFaceTB/SmolLM3-3B --local-dir models/SmolLM3-3B
hf download microsoft/Phi-4-mini-instruct --local-dir models/Phi-4-mini-instruct
otpu-chat --model smollm3 --backend board            # its mix, gateup@9-35=fp4 (docs/formats.md)
otpu-chat --model smollm3 --backend board --wformat int8   # closest to Hugging Face (Accuracy)
otpu-chat --model phi4-mini --backend board --wformat fp4 --head-format int8
python3 tools/compare_hf.py --model phi4-mini --cfg board --tokens 16   # ISA simulator vs HF
python3 tools/perf_qwen.py --model smollm3 --layers 4 --pos 128 --ddr 1066 --mhz 133.33
```

## The models

- **SmolLM3-3B**: 36 layers, hidden size 2048, 16 query heads and 4 KV heads of 128, a SwiGLU
  MLP 11008 wide, a 128,256-token vocabulary and a tied LM head. RoPE (theta 5e6) runs in 27
  layers; every fourth layer (3, 7, ..., 35) has no positional encoding ("NoPE",
  `no_rope_layers`).
- **Phi-4-mini-instruct** (3.8B): 32 layers, hidden size 3072, 24 query heads and 8 KV heads of
  128, a SwiGLU MLP 8192 wide, a 200,064-token vocabulary and a tied LM head. The checkpoint
  fuses q, k and v into `qkv_proj` and gate and up into `gate_up_proj`. RoPE covers the first 96
  of each head's 128 dimensions (`partial_rotary_factor` 0.75, theta 1e4) and is LongRoPE: up to
  `original_max_position_embeddings` (4096) positions Hugging Face uses the short factors, which
  divide the frequencies (all 1.0 in this checkpoint), and scales cos and sin by the attention
  factor `sqrt(1 + ln(131072 / 4096) / ln(4096))` = 1.1902.

Both have no biases and plain RMSNorm, and neither uses a sliding window.

## How they map

Qwen3's `Spec` gained the fields in which they differ. Each is off for Qwen3, whose programs
assemble to the same words as before.

| Spec field | SmolLM3-3B | Phi-4-mini | What it does |
|:--|:--|:--|:--|
| `qk_norm` | False | False | no q_norm / k_norm (`qwen3._norm_heads`) |
| `nope` | layers 3, 7, ..., 35 | | the rope gate below |
| `rotary` | | 96 | RoPE on the first 96 dimensions; the others pass through, as in Qwen3.5 |
| `rope_div`, `rope_scale` | | short factors, 1.1902 | LongRoPE's frequencies and scale, in the RoPE tables |
| `ctx` | | 4096 | the largest KV capacity: past it Hugging Face switches to the long factors |
| `embed` | int8 | int8 | the embedding is gathered from the int8 LM head |

**One hardware loop over all layers, with and without RoPE.** A SmolLM3 layer without RoPE
could be a second loop body, but that would double the program. Instead each layer block holds
a rope gate g: (1, 0) with RoPE and (0, 1) without. The layer rotates with `cos * g0 + g1` and
`sin * g0`, which gives the tables bit for bit, or the identity rotation (1, +-0)
(`qwen3._rope_gate`). The cost is one LD and three VOPs of 64 per layer, and the program is the
same for any pattern of NoPE layers (`tests/test_llama.py`).

**LongRoPE is a table.** The RoPE rows of every position are computed once, when the image is
built (`qwen3.rope_tables`), and every program reads its rows from that table in the image, so
the short factors and the attention factor cost nothing per token, on the device or the host. The KV capacity stays within 4096 positions, where Hugging Face uses the
short factors: `Image` refuses a larger one.

**Fused projections.** `qwen3.Weights` reads Phi-3's `qkv_proj` and `gate_up_proj` as separate
q / k / v and gate / up matrices, by row slices of the tensor (safetensors `get_slice`).

**The embedding is gathered from the int8 LM head.** An fp32 embedding table would be 1.0 GB
(SmolLM3) or 2.3 GB (Phi-4-mini): beside int8 weights it does not fit the card's 4 GiB. The tied
LM head is already in DRAM as int8 with one scale per 128-block, so the device reads the token's
row out of it with the shared table gather (`kernels/gather.py`, `gather_row`): the row streams
as rows of one block each against a one-hot operand (row j of one-hot block k is 1 at element
4k + j at MCOLS 4), so each MM gives four elements of every block, dequantized by the MXU's own
arithmetic, `f32(f32(i2f(127 q) * s) * f32(1/127))`, within an ulp of `q * s`; one copy per
block puts the row in order. The one-hot operand sits in the image's lookup tables. The resident
decode, the per-position programs and the prefill runs all gather their rows on the device, with
the token ids as run arguments or compiled into the program, so the host never dequantizes a row
(`Engine.device_inputs`, [host.md](host.md)). `qwen3.gathered_rows` (`gather.dequant_row`) is
the host's reference for the tests, and the float64 emulation quantizes the embedding row too.
With a 4-bit LM head the image holds an int8 table of its own. The gather adds 49 (SmolLM3) or
57 (Phi-4-mini) instructions to the resident decode program, and a token id costs it two run
arguments (the row's byte offset and its scales' offset) instead of one.

**Weights are read lazily.** `load_weights` returns a mapping that converts each tensor to fp32
when it is first used (`qwen3.Weights`), so a 3-4B checkpoint (12-15 GB in fp32) is never all in
memory: building the image quantizes one tensor at a time.

**Sizes.** At a 2048-token KV capacity, with the resident decode's tables:

| Model | int8 | 4-bit, int8 head | 4-bit (own int8 embedding) |
|:--|--:|--:|--:|
| SmolLM3-3B | 3104 MiB | 1763 MiB | 1896 MiB |
| Phi-4-mini | 3912 MiB | 2376 MiB | 2687 MiB |

The resident decode program is one hardware loop over the layers: SmolLM3 685 instructions in
the first 256-position bucket and 1019 in the eighth (2048 positions), Phi-4-mini 486 and 1148.
Prefill runs 4 rows (SmolLM3) or 3 rows (Phi-4-mini) per device run. TMEM limits it: the
MLP's fp32 tiles of 2048- and 3072-wide rows, and the gathers' temporaries. SmolLM3 fits 5 rows
but runs 4 (`qwen3._whole_passes`): at MCOLS 4 a run of 5 rows streams each weight twice, and
without the LM head a 4-layer run of 5 rows takes 5.04 M cycles on the RTL, one of 4 rows
2.58 M.

## On the card

The production image, build B (`deploy_fused133c_79c5707a`, 133.33 MHz, DDR3-1066), 2026-09-30,
`tools/qual/perf.py` (a 512-token prompt, then 64 greedy decode tokens with the host's argmax in
the loop) and `tools/decode_profile.py` (96 tokens, the logits streamed), device / wall tok/s:

| Model | Weights | Decode | Streamed decode | Prefill, device | DRAM while decoding |
|:--|:--|--:|--:|--:|--:|
| SmolLM3-3B | int8 | 5.00 / 4.99 | | 21.1 | 16.0 GB/s (94%), 3202 MB/token |
| SmolLM3-3B | 4-bit, int8 head | 8.74 / 8.72 | 8.92 / 8.89 | 22.8 | 15.7 GB/s (92%), 1796 MB/token |
| Phi-4-mini | int8 | 3.99 / 3.98 | | 13.8 | 16.0 GB/s (94%), 4011 MB/token |
| Phi-4-mini | 4-bit, int8 head | 6.56 / 6.55 | 6.69 / 6.67 | 15.0 | 15.8 GB/s (92%), 2400 MB/token |

Decode streams every weight once per token at 92-94% of the DDR3 peak. The previous image
(`deploy_champ_e698dcd7`) was 7-9% slower at 84-87%: 4.62 and 7.99 tok/s (SmolLM3), 3.70 and
6.04 (Phi-4-mini). SmolLM3's prefill is 1.69x e698dcd7's in int8 (12.5 tok/s) and 1.56x in
4-bit (14.6). Phi-4-mini's prefill, 3 rows on both images, gains only 1.08x and 1.01x, so most of
it is the 4-row runs: whole MXU passes instead of 5 rows (`qwen3._whole_passes`). The card gives the ISA
simulator's tokens, bit for bit, per-position and with the resident decode program
(`tools/qual/refs.py card`, 32 tokens of otpu-selftest's prompt; int8 on e698dcd7, 4-bit on
c2830d6a).

## Accuracy

The numpy reference (`qwen3.reference_logits`) matches Hugging Face's fp32 models to 1e-4 on the
tiny test models. On the device both run in W8A8, so they drift from Hugging Face where two
tokens are nearly tied.

Greedy decoding on the ISA simulator in the card's configuration vs Hugging Face fp32
(`tools/compare_hf.py --cfg CFG.pkl`, int8 weights, 16 tokens). The logit error is over the
steps where both have the same context.

| Model | Prompt | Same tokens | First difference: HF's rank of the device's token, logit gap | Max logit error | Min cosine |
|:--|:--|:--|:--|--:|--:|
| SmolLM3-3B | The capital of France is | 2 | #3, 0.107 | 1.31 | 0.9998 |
| | def fibonacci(n): | all 16 | | 3.60 | 0.9934 |
| | Water boils at | all 16 | | 4.16 | 0.9342 |
| | import numpy as np | all 16 | | 1.93 | 0.9930 |
| Phi-4-mini | The capital of France is | all 16 | | 2.27 | 0.9993 |
| | def fibonacci(n): | 0 | #2, 0.584 | 1.19 | 0.9999 |
| | Water boils at | all 16 | | 1.97 | 0.9998 |
| | The quick brown fox | all 16 | | 2.15 | 0.9997 |

SmolLM3's logit errors are larger than the other models': its first token's hidden state has a
few channels near 400 (a "massive activation"), and the int8 quantization of those rows costs
about 2.5% per element. The float64 emulation with the same quantization points is as far from
Hugging Face as the device.

With 4-bit weights and the int8 head (`--wformat fp4 --head-format int8`), SmolLM3 leaves Hugging
Face's path sooner. The same four prompts give 0 of 4 identical (int8: 3 of 4), and the logit
error grows 3-4 times:

| Prompt | Same tokens | First difference: HF's rank of the device's token, logit gap | Max logit error | Min cosine |
|:--|:--|:--|--:|--:|
| The capital of France is | 2 | #2, 0.011 | 3.88 | 0.9963 |
| def fibonacci(n): | 11 | #2, 0.233 | 13.46 | 0.7252 |
| Water boils at | 1 | #2, 0.590 | 12.73 | 0.8966 |
| import numpy as np | 7 | #2, 0.308 | 14.04 | 0.8276 |

Each first difference is HF's second choice, and the device's continuations are fluent (" 100 °C
at standard atmospheric pressure" for HF's " 212°F (100°C) at 1 atmosphere").

The MLP's 4-bit weights cause most of the error, not the attention's (Gemma 4's case). This was
measured with the float64 emulation, with a format per weight kind and the int8 head,
teacher-forced on HF's 16 tokens. The max logit error over the 16 steps, per prompt:

| Weights | Max logit error | Tokens before the first difference |
|:--|:--|:--|
| all 4-bit (the device's) | 9.7-16.3 | 2, 11, 3, 7 |
| attention int8, MLP 4-bit | 10.1-14.9 | 2, 11, 3, 16 |
| attention 4-bit, MLP int8 | 3.9-10.1 (three prompts under 5) | 2, 16, 1, 7 |
| all int8 | 1.9-4.1 | 2, 16, 1, 16 |

The MLP is 87% of SmolLM3's layer weights, so an int8 MLP costs nearly as much as int8 weights.
The 4-bit emulation agrees with the device on three prompts: the same first difference and the
same token. "Water boils at" is a near-tie at token 2 ('100' 0.59 below HF's '212'). The device takes
'100' there in 4-bit, and so does the int8 emulation. The 4-bit emulation and the int8 device
take '212'. So the device's 4-bit path is quantization, not a kernel error.

**For SmolLM3, int8 is the format that stays with Hugging Face** (`--wformat int8`, 5.0 tok/s).
4-bit (8.7 tok/s) gives fluent text, but it leaves HF's greedy path after 1 to 11 tokens on
the four prompts. No cheaper mix helps, because the error is in the MLP, which holds most of the
weights. The recommended mixes (`wformat="mix"`, `docs/formats.md`) are SmolLM3's
`gateup@9-35=fp4` (23% faster than int8, dKL 2.1%) and Phi-4-mini's `mlp@4-27=fp4` (5.20 tok/s
on the card, 28% over int8).

## Tests

`tests/test_llama.py`, on tiny random models saved as checkpoints and read back through
`load_spec` / `load_weights`: a SmolLM3 with 2 of its 8 layers without RoPE, and a Phi-3 with
query groups of 3, RoPE on 96 of 128 dimensions and LongRoPE short factors 1.0 .. 3.35 (Phi-4's
are all 1).

- The specs and the fused projections' row slices.
- The reference against Hugging Face (< 1e-4); the device against Hugging Face (cosine > 0.998)
  and the emulation (> 0.9995), design and board configurations; 4-bit weights with an int8
  head.
- The resident decode, bit-exact against the per-position programs across the bucket boundary
  256 after a chunked prefill, with the gather from the int8 LM head, and with an own int8 table
  under a 4-bit head.
- Chunked prefill bit-exact against decoding token by token, the KV cache included.
- One loop body for any pattern of NoPE layers.
- The device's gather from an int8 table equals `gathered_rows` bit for bit.
