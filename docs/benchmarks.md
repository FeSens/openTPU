# LLM benchmarks: Qwen3-0.6B on the board configuration

Prefill throughput, decode throughput at batch 1/2/4/8, and time to first token (TTFT), all
measured on the Verilator RTL of the YPCB-00338 board configuration. Reproduce with:

    python3 tools/bench_llm.py --validate --jobs 8 [--cache rtl_cache.json] [--json out.json]

Configuration: `board_config()` (1 slice, D=128, MCOLS=2, LANES=8, 64K-word TMEM, 4K-instruction
IMEM), `rtlsim.BOARD_UARCH`, the AXI memory path with the program booted from DRAM, 30 cycles of
AXI latency, no random stalls. Throughput is at 100 MHz. `bw` is the fraction of peak DRAM
bandwidth, where peak is one 128-byte chunk per cycle: 80% is realistic for the board's two DDR3
channels, and 100% is the peak. Weights are int8 with fp32 block scales (W8A8). Host time is not
included.

## Results

Decode. Each step feeds one token to each of b sequences; all b sequences have the same context
length.

| bw | b | ctx | ms/step | tok/s (total) | roofline tok/s | bound | achieved | MXU-stream bound | achieved |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 80% | 1 | 128 | 62.2 | **16.1** | 16.5 | DRAM | 98% | 16.5 | 98% |
| 80% | 2 | 128 | 66.3 | **30.2** | 32.5 | DRAM | 93% | 32.5 | 93% |
| 80% | 4 | 128 | 132.0 | 30.3 | 41.9 | compute | 72% | 32.5 | 93% |
| 80% | 8 | 128 | 263.5 | 30.4 | 41.9 | compute | 72% | 32.5 | 93% |
| 80% | 1 | 512 | 64.7 | 15.5 | 15.9 | DRAM | 97% | 15.9 | 97% |
| 80% | 2 | 512 | 71.2 | 28.1 | 30.3 | DRAM | 93% | 30.3 | 93% |
| 80% | 4 | 512 | 141.7 | 28.2 | 39.1 | compute | 72% | 30.3 | 93% |
| 80% | 8 | 512 | 282.7 | 28.3 | 39.1 | compute | 72% | 30.3 | 93% |
| 80% | 1 | 1024 | 67.7 | 14.8 | 15.2 | DRAM | 97% | 15.2 | 97% |
| 80% | 2 | 1024 | 77.4 | 25.9 | 27.8 | DRAM | 93% | 27.8 | 93% |
| 80% | 4 | 1024 | 154.1 | 26.0 | 35.9 | compute | 72% | 27.8 | 93% |
| 80% | 8 | 1024 | 309.3 | 25.9 | 35.9 | compute | 72% | 27.8 | 93% ¹ |
| 100% | 1 | 128 | 49.9 | **20.0** | 20.6 | DRAM | 97% | 20.6 | 97% |
| 100% | 2 | 128 | 54.3 | **36.8** | 40.6 | DRAM | 91% | 40.6 | 91% |
| 100% | 4 | 128 | 107.9 | 37.1 | 41.9 | compute | 88% | 40.6 | 91% |
| 100% | 8 | 128 | 215.4 | 37.2 | 41.9 | compute | 89% | 40.6 | 91% |
| 100% | 1 | 512 | 52.5 | 19.0 | 19.9 | DRAM | 96% | 19.9 | 96% |
| 100% | 2 | 512 | 59.1 | 33.8 | 37.9 | DRAM | 89% | 37.9 | 89% |
| 100% | 4 | 512 | 117.5 | 34.1 | 39.1 | compute | 87% | 37.9 | 90% |
| 100% | 8 | 512 | 234.3 | 34.2 | 39.1 | compute | 87% | 37.9 | 90% |
| 100% | 1 | 1024 | 55.4 | 18.1 | 19.0 | DRAM | 95% | 19.0 | 95% |
| 100% | 2 | 1024 | 65.3 | 30.6 | 34.8 | DRAM | 88% | 34.8 | 88% |
| 100% | 4 | 1024 | 130.0 | 30.8 | 35.9 | compute | 86% | 34.8 | 88% |
| 100% | 8 | 1024 | 260.1 | 30.8 | 35.9 | compute | 86% | 34.8 | 88% ¹ |

Prefill and TTFT. Prefill starts from an empty cache and runs chunks of 8 prompt rows. TTFT is
the time until the first generated token's logits are in DRAM.

| bw | prompt P | prefill tok/s | TTFT | roofline tok/s | bound | achieved | MXU-stream bound | achieved |
|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 80% | 32 | **40.1** | **0.80 s** | 57.3 | compute | 70% | 44.1 | 91% |
| 80% | 128 | 40.2 | 3.18 s | 57.0 | compute | 71% | 44.7 | 90% |
| 80% | 512 | 38.9 | 13.2 s | 54.5 | compute | 71% | 44.7 | 87% |
| 100% | 32 | 49.1 | 0.65 s | 57.3 | compute | 86% | 55.1 | 89% |
| 100% | 128 | 49.2 | 2.60 s | 57.0 | compute | 86% | 55.9 | 88% |
| 100% | 512 | 47.2 | 10.8 s | 54.5 | compute | 87% | 54.2 | 87% |

¹ At b=8 with a 1024-token context, the program has 4738 instructions, which is more than the
board's 4K-instruction IMEM. It was run with an 8K-instruction IMEM; see the limits below.

LM head (151936 × 1024) cycles, by number of rows M:

| bw | M=1 | M=2 | M=4 | M=8 |
|---:|---:|---:|---:|---:|
| 80% | 1.59 M | 1.60 M | 3.20 M | 6.39 M |
| 100% | 1.27 M | 1.29 M | 2.58 M | 5.17 M |

Validation: a full 28-layer run at 80% bandwidth and a 128-token context matched the
extrapolation.

| b | measured cycles | extrapolated | error |
|---:|---:|---:|---:|
| 1 | 6,216,779 | 6,216,776 | −0.00005% |
| 2 | 6,628,714 | 6,628,647 | −0.001% |

## What the numbers say

- **b=1 decode is DRAM-bound and runs at 95–98% of the roofline.** The roofline is every weight
  byte plus the KV cache, read once per token. The b=1 decode kernel is `qwen3_step`, the
  optimized kernel from the perf work.
- **b=2 nearly doubles throughput, from 16.1 to 30.2 tok/s at 80% bandwidth.** With MCOLS=2,
  one weight stream serves both rows. The remaining 7% is the second sequence's attention and
  its KV stream.
- **b=4 and b=8 add nothing at MCOLS=2.** An MM holds only MCOLS stationary rows, and each MM
  streams its weights from DRAM. So R rows stream every weight ceil(R/MCOLS) times: step time
  grows linearly with b beyond 2.
  - The "MXU-stream bound" column models exactly this, and the design reaches 88–93% of it.
  - Against the machine's ideal compute roofline (MCOLS·D = 256 MACs/cycle, with weights read
    once), b≥4 achieves only 72% at 80% bandwidth.
  - Closing that gap needs either a wider MXU (MCOLS ≥ b) or weight reuse across MMs. Both
    belong to the MXU study.
- **Prefill is bound the same way, at about 2 tokens per weight pass.** It gets 40 tok/s at 80%
  bandwidth and 49 tok/s at 100%. That is 87–91% of the MXU-stream bound and 70–87% of the
  compute roofline (57 tok/s).
  - With MCOLS=2, chunks larger than 2 only amortise per-run overhead.
  - Prefill is still 2.5× faster than feeding the prompt token by token: 40 vs 16 tok/s.
- **TTFT for a 32-token prompt is 0.80 s at 80% bandwidth.** It is almost entirely the four
  8-row chunks; the LM head is 1.6 M cycles, 2%.

## Method

### Kernels

**`qwen3_rows(m, rows, logit_rows)`** in `opentpu/llm/qwen3.py` runs R ≤ 8 token rows in one
device run.
- Each row is a (sequence, position) pair.
- Every projection, MLP and LM-head `dot` takes all R rows. The compiler emits ceil(R/MCOLS)
  MMs per weight chunk.
- Each sequence has its own KV cache: `Image(batch=b)` gives every layer block b caches.

**Attention in `qwen3_rows`:**
- All rows' K/V are appended first. Each (row, KV head) pair then attends over positions
  0..pos of its own sequence.
- For a prefill chunk, whose rows are consecutive positions of one sequence, this is exactly
  the causal mask, flash-style over the cache plus the chunk.
- The (row, head) pairs form one software-pipelined flash-attention stream: `_attend_heads`
  now takes per-entry caches and lengths, plus an `emit` callback that frees each finished
  head's state.

**Prefill:** `Engine.prefill(tokens, seq=, chunk=)` runs the prompt in chunks. Only the last
chunk runs the LM head, and only for its last row.

**Batched decode:** `Engine.step_batch(tokens)` and `Engine.generate_batch(prompts)` step b
sequences together, each at its own position.

**b=1 decode:** keeps `qwen3_step`, the per-token kernel with per-head lazy Q.

**Correctness** (`tests/test_qwen3.py`):
- The tiny random model's chunked prefill is bit-identical to token-by-token decode: same
  logits, and decoding continues identically.
- The tiny model is also checked against Hugging Face.
- Batched decode of 3 different prompts equals 3 separate runs, bit for bit.
- Real Qwen3-0.6B: chunked prefill (chunk 8) followed by batched decode of 2 prompts gives HF's
  greedy tokens, and matches a separate run.

### Measurement (`tools/bench_llm.py`)

**Timing is data-independent,** so the image holds random weights with the real shapes and an
empty KV cache.

**A full 28-layer run takes 2–3 minutes of simulation per token,** so the tool measures
1-layer and 2-layer proxies instead. It keeps the real layer shapes and the full LM head:

| Quantity | How it is measured |
|---|---|
| layer(ctx) | c(2 layers) − c(1 layer), LM head skipped |
| head(M) | c(1 layer + head) − c(1 layer) |
| fixed | c(1) − layer: boot, I/O, pipeline fill |
| step | fixed + 28·layer + head(b) |

- **b=1** uses `qwen3_step`, which always includes the head.
- **Prefill** chunk costs are measured at the first and last chunk of every prompt, and on
  both sides of each 256-token attention-block boundary. Costs in between are interpolated
  linearly, then summed over the chunks, plus head(1).
- **`--validate`** runs the full 28-layer model for b=1 and b=2 and compares it with the
  extrapolation.

**Rooflines** (per run, at bw):
- **DRAM:** weights + LM head + each sequence's KV cache, read once, at 128·bw/100 bytes per
  cycle.
- **Compute:** MACs / (MCOLS·D). MACs are rows × parameters, plus 2·n_q·head_dim·ctx per row
  and layer for attention.
- **"Roofline"** is max(DRAM, compute). For prefill it is the whole prompt in one pass.
- **"MXU-stream"** is this MXU's own bound: every weight is streamed ceil(R/MCOLS) times. For
  prefill it is summed over the chunks.
- **"Achieved"** is bound / measured.

### Options for MXU studies

| Option | Effect |
|---|---|
| `--mcols N`, `--lanes N` | Override the configuration. MCOLS must be ≤ LANES. |
| `--chunk` | Prefill rows per run, ≤ 8. TMEM limits a run to 8 rows. |
| `--batches`, `--ctx`, `--prompts`, `--bw`, `--lat` | Select the measurement points. |
| `--cache` | Keeps RTL results between invocations. Delete it after changing the RTL or the kernels. |

The full default sweep is 88 RTL runs plus the validation, about 10 minutes with 8 jobs.

## Limits

- **A device run holds at most 8 rows,** because TMEM is 64K words.
- **Attention is unrolled** per row, KV head and 256-token block. At b=8 the program reaches
  about 3.7K instructions at a 128-token context and 4.7K at 1024. The board IMEM holds 4K
  instructions, so b=8 at contexts above about 768 needs a larger IMEM or a looped attention; those
  rows are marked ¹.
- **Prefill of long prompts in chunks of 8** fits the IMEM up to about 1024 tokens.
- **All sequences in a batch run at the same position here.** The kernel itself allows
  independent positions.

## LFM2.5-230M

LFM2 runs one token per device run (no batched decode or chunked prefill), so only b=1 decode
is measured, on the full model with `tools/perf_qwen.py --model lfm2 --layers 0` (same
configuration as above). Tokens/s are projections at 100 MHz.

| bw | ctx | cycles/token (measured) | tok/s | % of DRAM roofline |
|---:|---:|---:|---:|---:|
| 80% | 128 | 2,366,994 | **42.2** | 96.4% |
| 80% | 1024 | 2,455,759 | 40.7 | 96.3% |
| 100% | 128 | 1,901,551 | 52.6 | 96.0% |
| 100% | 1024 | 1,996,829 | 50.1 | 94.8% |

Details, accuracy and the mapping are in [lfm2.md](lfm2.md).

## Qwen3.5-0.8B

Qwen3.5 runs one token per device run, so only b=1 decode is measured, on the full model with
`tools/perf_qwen.py --model qwen35 --layers 0` (same configuration as above; the roofline is
every byte the token moves, KV read once). Tokens/s are projections at 100 MHz.

| bw | ctx | cycles/token (measured) | tok/s | % of DRAM roofline | without RDOT/OUTER (measured) |
|---:|---:|---:|---:|---:|---:|
| 80% | 128 | 9,130,829 | **11.0** | 87.6% | 11,535,257 (69.3%) |
| 80% | 1024 | 9,231,246 | 10.8 | 87.2% | 11,636,450 (69.2%) |
| 100% | 128 | 8,022,432 | 12.5 | 79.8% | 10,419,105 (61.4%) |
| 100% | 1024 | 8,099,652 | 12.3 | 79.5% | 10,495,642 (61.4%) |

The Gated DeltaNet recurrence runs on the vector unit with RDOT and OUTER (3 passes over each
head's state instead of 7) and takes 37% of the token (3.41 M cycles, at either bandwidth); the
MLPs, attention and LM head run at their rooflines. VPU_CL=4 gains under 1%. Details,
accuracy and the cycle breakdown are in [qwen35.md](qwen35.md).
