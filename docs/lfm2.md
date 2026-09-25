# LFM2 on openTPU (LFM2.5-230M)

openTPU runs a second model family next to Qwen3: Liquid AI's LFM2, checked with
[LFM2.5-230M](https://huggingface.co/LiquidAI/LFM2.5-230M). It uses the same ISA, RTL, W8A8
numerics and host driver as Qwen3; only the model code is new (`opentpu/llm/lfm2.py`).

```sh
hf download LiquidAI/LFM2.5-230M --local-dir models/LFM2.5-230M
otpu-chat --model lfm2                          # ISA simulator
otpu-chat --model lfm2 --backend board          # the card
otpu-selftest --sim --model lfm2 --tokens 2     # the board model, vs the ISA simulator
python3 tools/compare_hf.py --model lfm2 --chat --tokens 48 "Describe the water cycle."
python3 tools/perf_qwen.py --model lfm2 --layers 0 --pos 128 --bw 80
```

## The model

LFM2.5-230M has 14 layers, hidden size 1024, a 65536-token vocabulary and a tied LM head. Each
layer is either a short convolution (c) or GQA attention (A), followed by a SwiGLU MLP (2560
wide), in the order `c c A c A c A c A c A c A c`.

- **Conv layer.** `in_proj` (3072 x 1024) gives B, C and x; then `y = C * conv(B * x)`, a causal
  depthwise convolution with 3 taps per channel, and `out_proj` (1024 x 1024). The state
  between tokens is the last two rows of `B * x`.
- **Attention layer.** 16 query heads and 8 KV heads of 64, RMSNorm on each q and k head, then
  RoPE (theta 1e6). This is Qwen3's attention with narrower heads.
- A final RMSNorm (`embedding_norm`) before the LM head.

## How it maps

**Conv state.** Each conv layer keeps `B * x` in fp32 in a 3-slot ring in its DRAM layer block.
Position p writes slot p % 3 and reads the slots of p - 1 and p - 2. Programs are compiled per
position, so the slots are constant addresses. At positions 0 and 1 the missing rows are simply
not read, so a new conversation (`Engine.reset`) needs no clearing.

**64-wide heads on a 128-deep MXU.** `q . K^T` contracts over whole MXU blocks of D = 128. The
queries and the cached K rows are therefore padded with zeros to 128, which leaves the scores
unchanged. A quantized store (QST) writes whole blocks, so V's rows are stored padded too, but
`P . V` reads only V's 64 real rows (`KVDesc(dv=64)`). The projections are not padded: the
weights stream exactly their real bytes. The padding costs KV-cache bandwidth only: half of the
K stream, about 1.3% of the token at a 1024-token context.

**Hybrid layer loop.** All layer blocks in DRAM have the same size, so layer i starts at
`layer0 + i * LS` whatever its kind. The kernel runs the repeated `(conv, attn)` unit as one
hardware loop of 6 iterations, and the first and last conv layers unrolled (`lfm2.plan`). The
program is 819 instructions at position 0 and 1414 at position 4095, well inside the board's
4K-instruction IMEM.

Nothing new was needed in the ISA or the RTL. The shared kernel code gained two small
generalizations: `KVDesc` takes a V width (`dv`), and `qwen3._attention` pads heads narrower
than D. For Qwen3 both are no-ops: its programs are bit-identical to before.

**Limit:** LFM2 runs one token per device run (`Engine.step`). Batched decode and chunked
prefill (`qwen3_rows`) are not implemented for it; the prompt is fed token by token.

## Accuracy

The numpy reference (`lfm2.reference_logits`) matches Hugging Face's fp32 LFM2.5-230M to 5e-5.
On the device the model runs in W8A8, like Qwen3, so it drifts from Hugging Face where two
tokens are nearly tied.

Pure greedy decoding on the ISA simulator vs Hugging Face fp32 (`tools/compare_hf.py --model
lfm2 --emulate`; `repetition_penalty=1.0`, because LFM2.5's generation config sets 1.05). The
logit error is measured over the steps where both have the same context.

Chat prompts (`--chat`, 48 tokens):

| Prompt | Same tokens | First difference: HF's rank of the device's token, logit gap | Max logit error | Min cosine |
|---|---|---|---:|---:|
| What is the capital of France? Answer in one sentence. | all 8 (to EOS) | | 1.41 | 0.9980 |
| Explain in two sentences why the sky is blue. | 14 | #2, 0.084 | 0.92 | 0.9990 |
| Write a Python function that checks whether a number is prime. | all 48 | | 1.67 | 0.9943 |
| Give me a short definition of photosynthesis. | 21 | #2, 0.055 | 0.85 | 0.9987 |
| List the first five prime numbers. | all 48 | | 1.19 | 0.9975 |
| Translate 'good morning' into French and Spanish. | 22 | #3, 0.278 | 1.11 | 0.9986 |
| What is the capital of Japan, and what is it famous for? | 46 | #2, 0.009 | 0.87 | 0.9991 |
| Describe the water cycle in one paragraph. | all 48 | | 1.85 | 0.9963 |

Raw prompts (the eight of the README's Qwen3 table, 16 tokens): 5 of 8 identical. The other
three differ at tokens 1, 5 and 13, each where the device's token is HF's second choice, 0.02
to 0.10 logits below the top. The largest logit error is 1.57, and the lowest cosine 0.9962.

HF's logits span about 30 to 60 across the vocabulary, so an error of 1 to 2 moves only
near-ties. In 4 of the 7 differences, the float64 emulation with the same int8 quantization
points picks the device's token. In the other 3 it picks HF's (`Water boils at`, `The quick
brown fox`, the translation prompt). There the device's fp32 rounding, not only the
quantization, decides the tie. That is within what Qwen3 shows too: over the same 16 tokens of
one prompt, the device's logits agree with the emulation to a lowest cosine of 0.9988 for
LFM2.5-230M and 0.9881 for Qwen3-0.6B. On a tiny random LFM2, the device agrees with the
emulation to 0.99969 (`tests/test_lfm2.py`, the same bound as Qwen3's test).

## Performance

One decode token of the full model (14 layers and the LM head), measured on the Verilator RTL
of the board configuration: 1 slice, D=128, MCOLS=2, LANES=8, the AXI memory path with the
program booted from DRAM, and 30 cycles of AXI latency. The RTL is the committed one (HEAD
5a1b5d2). `bw` is the fraction of peak DRAM bandwidth (one 128-byte chunk per cycle).
Tokens/s are projections: the measured cycles at an assumed 100 MHz clock, without host time.

| bw | context (pos) | cycles/token (measured) | DRAM roofline | % of roofline | tok/s at 100 MHz (projected) |
|---:|---:|---:|---:|---:|---:|
| 80% | 128 | 2,366,994 | 2,281,850 | 96.4% | **42.2** |
| 80% | 1023 | 2,455,759 | 2,365,250 | 96.3% | 40.7 |
| 100% | 128 | 1,901,551 | 1,825,480 | 96.0% | 52.6 |
| 100% | 1023 | 1,996,829 | 1,892,200 | 94.8% | 50.1 |

The roofline is every DRAM chunk the token reads or writes (weights, their block scales, the KV
cache including its zero padding, the I/O), at bw. The token is DRAM-bound: the DRAM port is
busy 96% of the cycles at 100% bandwidth. The RTL's DRAM and the ISA simulator's agree bit for bit at
position 128 (`--check`).

For comparison, Qwen3-0.6B decodes at 16.1 tok/s under the same conditions (80%, context 128;
[benchmarks.md](benchmarks.md)). LFM2.5-230M streams about 234 MB per token (1,825,480 chunks),
Qwen3-0.6B about 600 MB.

DRAM: the image is 233 MiB at a 256-token KV capacity and 276 MiB at 2048, of which the KV
cache and conv state are 25 MiB (Qwen3-0.6B: 703 MiB at 2048).

## Tests

`tests/test_lfm2.py`:

- A tiny random LFM2 (`c A c A c`) against Hugging Face over 140 tokens, and against the int8
  emulation.
- `Engine.reset` gives bit-identical logits, so no conv state leaks between sequences.
- The layer plan, and the program at position 4095 fitting the board IMEM.
- The tiny model on the Verilator board model through the host driver, over four tokens (a full
  turn of the conv ring), bit-identical to the ISA simulator.
- LFM2.5-230M: 8 greedy tokens equal to HF's ("The capital of France is Paris."), and one real
  token on the RTL, bit-exact against the ISA simulator.
