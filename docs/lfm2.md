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
Position p writes slot p % 3 and reads the slots of p - 1 and p - 2. The ring is mirrored: 6
rows, slot s at rows 1 + s and 4 + s (slot 2 has no mirror), row 0 scratch, so rows w + 1 ..
w + 3, w = (p + 1) % 3, hold positions p - 2 .. p in order (`_ring_rows`). A per-position
program uses constant rows; the resident decode program (docs/host.md) reaches them all with
one argument register, w x row (storing its row to w and w + 3). At positions 0 and 1 the
missing rows are simply not read, so a new conversation (`Engine.reset`) needs no clearing.

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

**Chunked prefill** (`lfm2_rows`, run by `Engine.prefill_chunks`): a prompt runs up to 8
tokens per device run, and every weight streams once for the run's rows.
- A conv layer projects B and x for all rows, then convolves each row over the rows before it
  in the chunk and over the ring (the positions before the chunk). C is projected after the
  convolution: TMEM does not hold all three for 8 rows. The ring ends up holding the chunk's
  last rows, each in its slot.
- An attention layer uses Qwen3's row attention (`qwen3._attention_rows`), which pads the
  64-wide heads as `_attention` does.
- Per row the operations and their order are the decode kernel's, so the KV cache, the ring and
  the logits are bit-identical to feeding the tokens one by one (`tests/test_lfm2.py`, and the
  real model on the ISA simulator).

**Limit:** batched decode (several sequences) is not implemented for LFM2.

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

### 4-bit weights on the DDR3 bank model

Everything in this section is **simulated**: one decode token of the full model at position 128
(cache capacity 256), on the Verilator RTL of the board configuration with column reuse
(`OTPU_PAIR=1`: every 4-bit layer MM a full-rate PAIR MM, [quant.md](quant.md)), against the
DDR3 bank model calibrated on the card at DDR3-1066 (`tools/perf_qwen.py --model lfm2 --ddr
1066 --mhz F`; the model came within 4% of the card at DDR3-800, [board.md](board.md) section
4). Tokens/s are the simulated cycles at the given clock, without host time. DRAM and port
efficiency are defined in [board.md](board.md) ("DRAM efficiency"). "fp4 + int8 head" is 4-bit
layers with an int8 LM head (`--wformat fp4 --head-format int8`, `otpu-chat --wformat fp4
--head-format int8`): 158 MB moved per token, of which the head is 69 MB.

| weights | RTL | core clock | Mcycles/token | port eff. | DRAM eff. | tok/s (device) |
|---|---|---:|---:|---:|---:|---:|
| int8 | this branch | 100 MHz | 2.071 | 90.2% | 67.6% | 48.3 |
| fp4 | this branch | 100 MHz | 1.180 | 82.3% | 61.7% | 84.8 |
| fp4 + int8 head | this branch, before the V^T move | 100 MHz | 1.442 | 85.5% | 64.1% | 69.3 |
| fp4 + int8 head | this branch | 100 MHz | 1.428 | 86.3% | 64.7% | 70.0 |
| fp4 + int8 head | + r6/r7 adapter (r7-apf a10203a), before the V^T move | 100 MHz | 1.321 | 93.3% | 70.0% | 75.7 |
| fp4 + int8 head | + r6/r7, before the V^T move | 116 MHz | 1.330 | 92.7% | 80.7% | 87.2 |
| fp4 + int8 head | + r6/r7, before the V^T move | 125 MHz | 1.334 | 92.4% | 86.6% | 93.7 |
| fp4 + int8 head | + r6/r7 | 116 MHz | 1.312 | 94.0% | 81.7% | 88.4 |
| fp4 | + r6/r7, before the V^T move | 116 MHz | 1.067 | 90.9% | 79.1% | 108.7 |

"The V^T move": `qwen3._attention` (which LFM2's attention layers use) appends every KV head's
K first and each head's V^T together with its queries, so the quantizer's slow V^T appends (a
byte into each of the head's cache rows, one ECC read-modify-write each) overlap the heads
before it instead of holding all queries back. The values are the same, so decode stays bit
exact; it also takes Qwen3-0.6B from 5.792 to 5.739 Mcycles (int8) and 3.465 to 3.410 (fp4),
same conditions at 100 MHz.

Where the cycles go (fp4 + int8 head, r6/r7, 116 MHz, 1.312 Mcycles): the LM head 548K (99%
of its byte roofline), the MLPs 461K (99%), the conv blocks 146K (97%), attention 157K (58%:
91K of bytes). The core-port roofline of the token is 1.233 Mcycles; the DDR3 peak alone would
allow 1.073. Nearly all the loss is in the attention layers: the quantizer runs the cache
appends (8 K appends of ~350 cycles and 8 V^T appends of 1.3-2K cycles per layer) in order,
ahead of each head's query and P quantizations, and the MXU waits on them (MXU gaps after
the attention QACTs: ~29K cycles per token). The rest is the dependency chain at each layer
boundary and conv out_proj (~30K together).

**What 88 tokens/s (wall) needs.** At 88 tok/s a token has 11.36 ms. The host adds 1.3 ms per
token today (measured on the card for the int8 image, [board.md](board.md) section 4: mostly
the logits read and the per-token register and DMA traffic; the host's own computation,
sampling included, is ~0.1 ms, measured on a Mac). With the int8 head at 116 MHz the device
alone takes 11.31 ms (simulated), so the host would have to add nothing. What closes it, largest
first (all *estimates*):

1. RTL: faster V^T appends. The adapter's write queue at 64 entries was measured on r7-apf
   (simulated): an append 7.7K -> 4.9K cycles, with QST's own floor at ~2.35K cycles per KiB, so
   the rest needs the V^T channel layout and the QST path. Attention at its byte roofline would
   save ~65K cycles (-5%): ~1.25 Mcycles, ~93 tok/s device at 116 MHz (*estimate*).
2. Clock: 116 -> 125 MHz scales tokens/s with the clock (the cycles grow 0.3%): ~99 tok/s
   device with item 1.
3. Host per-token time (1.3 ms -> <= 0.5 ms): the logits are now read and sampled while the
   run goes on, all but the last chunk ([host.md](host.md), "Streamed logits"); one program
   per model instead of one per position removes the program copy, the IMEM load and the
   compile wait. `tools/host_path_card.py` measures what is left on the card.
4. Compiler + layout: one QST for all heads' K appends (and V^T), with the per-head scale
   arrays interleaved by position (a QST writes its scales contiguously): about -2K cycles
   per attention layer, -1% per token.
5. Format: a 4-bit LM head instead of int8 moves 34 MB less per token: 1.067 Mcycles and 108.7
   tok/s device at 116 MHz (before the V^T move), which reaches 88 wall with today's host.
   Accuracy ([quant.md](quant.md), emulation, book text): ppl 33.16 (KL 0.159) against 32.92
   (KL 0.146) with the int8 head, int8 everywhere 29.05 (0.003).

With items 1 and 3 the int8-head configuration projects to ~93 tok/s device and ~88 wall at
116 MHz (0.5 ms of host per token), and ~100 device / ~95 wall at 125 MHz, where DRAM efficiency
is ~92% of the DDR3-1066 peak. Past that the core port, not the DDR3, is the limit.

### Decode attention at DDR3-1066 (2026-09-27)

Simulated on the RTL with the DDR3-1066 bank model (`tools/perf_qwen.py --model lfm2 --layers 0
--pos 128 --dram rbc --lat 38 --arc 4`, DDR3-1066 timings, `OTPU_PAIR=1`, fp4 layers, int8 LM
head; core clock as the `+axi_tpc/+axi_tpu` ratio; no host time). Before, the six attention
layers took 328 K cycles for 11.7 MB (28% of their bytes): each V^T append (a byte per
dimension, stride = the cache capacity) cost ~4.8 K cycles of controller read-modify-writes, and
the query QACTs queued behind all 16 appends on the quantizer.

| | 100 MHz | 116 MHz | attention (116 MHz) |
|---|---:|---:|---:|
| fp4-rebase + ddr-attn (152f9d4) | 1,480,447 (67.5 tok/s) | 1,491,387 (77.8 tok/s) | 328,087 |
| + r7-apf (adapter RMW, 64-beat SW queue, fast QST passes) | 1,303,606 (76.7) | 1,307,072 (88.7) | 143,795 |
| + V appends per head, QST `HALF` | 1,284,746 (77.8) | 1,287,048 (**90.1**) | 124,757 |
| same, fp4 LM head | 1,022,385 (97.8) | 1,024,680 (113.2) | |

- **QST `HALF`** (docs/isa.md): a V^T append of a 64-wide head writes its 64 real rows, not the
  128 of its padded tile (P.V never reads the padding): half the byte-strided writes.
- **The V^T appends move into the head pipeline** (`_attention`): the K appends go first, and
  head j's V append is emitted with its queries, so head j's scores wait for V_0..V_j, not for
  all eight. Qwen3 gains 0.5% (int8) / 0.9% (fp4); Qwen3.5 is unchanged.
- Tried and dropped: one Q MM for all heads instead of one per KV head (+1.9 K cycles), the
  pipeline's `ahead` 1..7 (within 0.4%).

At 116 MHz the token is now at 95.8% of the core port's roofline (157.8 MB); the DRAM
efficiency is 83%. The attention layers still take 125 K cycles for 91 K of bytes: the Q
projections stream at ~70% while the V^T read-modify-writes share the DRAM, and each head's
softmax chain adds ~0.7 K cycles after them.

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
