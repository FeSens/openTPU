# Qwen3.5 on openTPU (Qwen3.5-0.8B)

openTPU runs a third model family: Qwen3.5, checked with
[Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) (the text decoder; the vision tower
and the multi-token-prediction layer are not loaded). It uses the existing ISA and RTL
unchanged, and the same W8A8 numerics and host driver as Qwen3 and LFM2. This port is the
baseline of the architecture tournament: what the current hardware does with a model whose
main layer is a linear-attention recurrence, measured on the RTL.

```sh
hf download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B
otpu-chat --model qwen35                          # ISA simulator
otpu-chat --model qwen35 --backend board          # the card
otpu-selftest --sim --model qwen35 --tokens 2     # the board model, vs the ISA simulator
python3 tools/compare_hf.py --model qwen35 --chat --tokens 48 "Describe the water cycle."
python3 tools/perf_qwen.py --model qwen35 --layers 0 --pos 128 --bw 80
```

## The model

Qwen3.5-0.8B has 24 layers, hidden size 1024, a 248,320-token vocabulary and a tied LM head.
The layers repeat `(DeltaNet, DeltaNet, DeltaNet, attention)` six times, and every layer ends
with a SwiGLU MLP 3584 wide. All RMSNorms but one are zero-centered, `x * (1 + w)`.

- **Gated DeltaNet** (18 layers). `in_proj_qkv` (6144 x 1024) gives q, k and v for 16 heads of
  128; a causal depthwise convolution (4 taps) and SiLU run over them, then q and k are
  L2-normalized and q is scaled by 1/sqrt(128). Two tiny projections give per head
  `beta = sigmoid(b)` and a decay `g = -exp(A_log) * softplus(a + dt_bias)`. Each head keeps a
  128 x 128 state S and, for every token:

  ```
  S     = exp(g) S
  delta = beta (v - S^T k)
  S     = S + k delta^T
  o     = S^T q
  ```

  The output is `RMSNorm(o) * w * silu(z)` (z from `in_proj_z`, 2048 x 1024) and `out_proj`
  (1024 x 2048).
- **Gated attention** (6 layers). 8 query heads and 2 KV heads of 256, RMSNorm on each q and k
  head, RoPE on the first 64 of the 256 dimensions (theta 1e7). `q_proj` also yields a gate per
  query dimension: the attention output is multiplied by `sigmoid(gate)` before `o_proj`.

## How it maps

**The DeltaNet state streams through TMEM.** The state is 16 heads x 128 x 128 fp32 = 1 MiB per
layer, 16 times TMEM. It stays in fp32 in the layer's DRAM block and moves through TMEM one
head (64 KiB) at a time. The heads run in pairs in a hardware loop with two sets of buffers:
while the vector unit (VPU) updates head h, the DMA loads head h+1's state and the MXU streams
head h+1's projections (its 128 q, k, v and z rows of the input projections, stored head by
head). Head h's output is multiplied into the residual by its own 1024 x 128 column block of
`out_proj` (an accumulating MM), so no projection waits for all heads.

**The recurrence is VPU passes.** The state is stored transposed, `St = S^T` (rows are the
value dimension), so both reads of S are row sums and the update is one outer product:

```
kS = St k,  qS = St q           2 products (column broadcast) + 2 row sums
delta = beta (v - exp(g) kS)    vectors of 128
o = exp(g) qS + (k . q) delta   (= S_new^T q, from the old state)
St = exp(g) St + delta k^T      a scale, an outer product, an add
```

Per head that is 7 passes over 16K fp32 values: about 14,300 VPU cycles at 8 lanes. The rows
are independent, so each pass runs on blocks of 64 rows, which keeps the temporaries at 8K
words (TMEM also holds the two state buffers of 16K).

**Everything else.** The convolution state is a 4-slot ring of the pre-convolution q, k, v rows
in DRAM (LFM2's scheme, `docs/lfm2.md`). The VPU has no logarithm, so softplus is
`max(x, 0) + log1p(2^(-|x| log2 e))` with log1p a degree-8 polynomial (within 2e-7). The `a`
and `b` projections run in int8 like every other matrix: in the float64 emulation, keeping them
in full precision did not reduce the logit error (max 0.98 vs 1.10 and 1.42 vs 1.20 over two
38- and 42-token texts, argmax agreement unchanged within one token). Attention reuses Qwen3's
kernel: a 256-wide head is two MXU blocks, the RoPE tables cover 64 dimensions and the other
192 pass through, and the gate is a fifth projection whose sigmoid is applied to the head
outputs. A query group of 4 heads does not fit a 2-column MXU; there each KV head is streamed
twice, once per pair of query heads (with `OTPU_MCOLS=4`, once).

**Language additions**, no ISA change: `u[:, None] * v[None, :]` is one VOP (the A operand reads
`v` with row stride 0), a length-1 tile broadcasts over a whole tile (ROW mode with row stride
0), `t += x` and `t *= x` update in place without a temporary, `ol.load(desc, out=t)` refills a
buffer, and `ol.mxu_columns()` returns MCOLS. The Qwen3 and LFM2 programs assemble to exactly
the same words as before (checked at several positions, both configurations, MCOLS 2 and 4).

**Program size.** The six-fold layer unit is one hardware loop and the head pairs another: the
program is 1,781 instructions at position 0 and 2,181 at position 4095 (board configuration,
MCOLS=2; 1,739 and 1,993 with MCOLS=4), within the 4K-instruction IMEM. The DRAM image is
788 MiB at any KV capacity up to 4096 tokens (the DeltaNet layer blocks set the block size), of
which 21 MiB are KV cache, convolution ring and DeltaNet state at a 256-token capacity.

## Accuracy

The numpy reference (`qwen35.reference_logits`) matches Hugging Face's fp32 Qwen3.5-0.8B to
4e-5 in the logits on a short prompt. On the device the model runs in W8A8 like Qwen3, with the
DeltaNet state, convolution, gates and norms in fp32, so it drifts from Hugging Face where two
tokens are nearly tied.

Greedy decoding on the ISA simulator vs Hugging Face fp32 (`tools/compare_hf.py --model qwen35
--emulate`, design configuration; measured). The logit error is over the steps where both have
the same context. Hugging Face's generation config stops only at `<|endoftext|>`; the device
also stops at `<|im_end|>`, so a chat answer is compared up to its end.

Chat prompts (`--chat`, 48 tokens):

| Prompt | Same tokens | First difference: HF's rank of the device's token, logit gap | Max logit error | Min cosine |
|---|---|---|---:|---:|
| What is the capital of France? Answer in one sentence. | all 8 (to EOS) | | 2.01 | 0.9932 |
| Explain in two sentences why the sky is blue. | 28 | #2, 0.012 | 1.03 | 0.9987 |
| Write a Python function that checks whether a number is prime. | 35 | #2, 0.036 | 1.45 | 0.9925 |
| Give me a short definition of photosynthesis. | all 46 (to EOS) | | 1.10 | 0.9986 |
| List the first five prime numbers. | all 48 | | 1.02 | 0.9980 |
| Translate 'good morning' into French and Spanish. | all 20 (to EOS) | | 1.08 | 0.9976 |
| What is the capital of Japan, and what is it famous for? | all 48 | | 1.25 | 0.9978 |
| Describe the water cycle in one paragraph. | 28 | #2, 0.276 | 1.07 | 0.9981 |

Raw prompts (the README's eight, 32 tokens): 5 of 8 identical for all 32 tokens. The other
three differ at tokens 4, 5 and 6, where the device's token is HF's #3 (0.155 logits below the
top) and #2 (0.012, 0.079). The largest logit error is 1.45, the lowest cosine 0.9957.

HF's logits span about 30 to 55, so an error of 1 to 2 moves only near-ties. At the three raw
differences the float64 emulation with the same int8 quantization points picks the device's
token: quantization decides those ties. At the three chat differences it picks HF's: there the
device's fp32 rounding decides. On a tiny random Qwen3.5 the device agrees with Hugging Face to
a cosine above 0.998 over 48 tokens and with the emulation above 0.999 (`tests/test_qwen35.py`;
the emulation also rounds the weights to int8 slightly differently, dividing by the scale
where the device multiplies by 127/amax).

## Performance

One decode token of the full model (24 layers and the LM head), measured on the Verilator RTL
of the board configuration: 1 slice, D=128, 8 VPU lanes, the AXI memory path with the program
booted from DRAM, 30 cycles of AXI latency (`tools/perf_qwen.py --model qwen35 --layers 0
--check`, RTL of commit eb29dd3, whose DMA requests each DRAM chunk once). `bw` is the fraction of peak DRAM bandwidth (one
128-byte chunk per cycle). Every run ended with DRAM bit-identical to the ISA simulator's.

The **DRAM roofline** is every byte the token must move, at bw: the int8 weights and their fp32
block scales (LM head included), the convolution taps, the DeltaNet state read and written
(2 MiB per layer), the convolution ring, the KV cache read once and appended, and the I/O
(embedding in, logits out). That is 819.1 MB at context 128 and 824.4 MB at 1023, the bytes
the MCOLS=4 program moves; with MCOLS=2 the KV cache is read twice (+1.2 and +6.4 MB), which
counts against it. Tokens/s are projections: measured cycles at an assumed 100 MHz, no host
time.

| MCOLS / VPU_CL | bw | context (pos) | cycles/token (measured) | DRAM roofline | % of roofline | tok/s at 100 MHz (projected) |
|---|---:|---:|---:|---:|---:|---:|
| 2 / 2 (default board) | 80% | 128 | 11,535,257 | 7,999,152 | 69.3% | **8.7** |
| 2 / 2 | 80% | 1023 | 11,636,450 | 8,050,300 | 69.2% | 8.6 |
| 2 / 2 | 100% | 128 | 10,419,105 | 6,399,322 | 61.4% | 9.6 |
| 2 / 2 | 100% | 1023 | 10,495,642 | 6,440,240 | 61.4% | 9.5 |
| 4 / 4 | 80% | 128 | 11,504,083 | 7,999,152 | 69.5% | 8.7 |
| 4 / 4 | 80% | 1023 | 11,549,952 | 8,050,300 | 69.7% | 8.7 |
| 4 / 4 | 100% | 128 | 10,353,455 | 6,399,322 | 61.8% | 9.7 |
| 4 / 4 | 100% | 1023 | 10,386,998 | 6,440,240 | 62.0% | 9.6 |
| 2 / 4 | 80% | 128 | 11,513,884 | 7,999,152 | 69.5% | 8.7 |
| 4 / 2 | 80% | 128 | 11,525,809 | 7,999,152 | 69.4% | 8.7 |

(`OTPU_MCOLS=4 OTPU_VPU_CL=4` is the 4&4 bitstream of [board.md](board.md).) For comparison,
at 80% and context 128 (before eb29dd3): Qwen3-0.6B 16.1 tok/s and LFM2.5-230M 42.2 tok/s, both
at 96% or more of their rooflines.

### Where the cycles go

The same runs split into phases: the instructions, in program order, grouped by the kernel
function that emitted them (`perf_qwen.py` prints this table); a phase is charged the cycles
from the end of the previous phase to its own last completion, so the phases add up to the
token. The bytes column is what the phase moves; its roofline is those bytes at bw.
Default board (MCOLS=2, VPU_CL=2), context 128, measured:

| phase | cycles at 80% | share | its bytes (MB) | its roofline at 80% | cycles at 100% | its roofline at 100% |
|---|---:|---:|---:|---:|---:|---:|
| DeltaNet mixer, 18 layers | 5,834,217 | 50.6% | 236.6 | 39.6% | 5,856,438 | 31.6% |
| MLP, 24 layers | 2,649,504 | 23.0% | 272.6 | 100.5% | 2,113,202 | 100.8% |
| LM head | 2,574,507 | 22.3% | 263.2 | 99.8% | 2,060,090 | 99.8% |
| attention, 6 layers | 459,940 | 4.0% | 47.9 | 102% | 372,432 | 100% |

(Above 100%: a phase's first weights stream while the phase before it finishes.)

- **The DeltaNet recurrence is VPU-bound, and the rest of the token is at its roofline.** The
  mixer takes 5.83 to 5.86 M cycles at either bandwidth: it does not wait for DRAM. Its 236.6 MB (the
  projections, out_proj, 37.7 MB of state traffic) need 2.31 M cycles at 80%. Run at its
  roofline, the token would take about 8.01 M cycles at 80% and 6.41 M at 100%, i.e. the
  roofline.
- **Per head** (288 per token) the mixer spends about 20,300 cycles. The 7 state passes are
  14,300 VPU cycles of issue (8 lanes); the rest is the state load competing with the VPU for
  TMEM writes (one write per bank per cycle: the 16K-word load costs the VPU about 2,000
  cycles), the per-head vector work (convolution, SiLU, L2 norms, gated RMSNorm: ~50 small
  VOPs, whose composite functions run on 2 lanes) and dependency stalls in the chain
  row sums -> delta -> outer product. Meanwhile the MXU streams the next head's 512 projection
  rows and this head's out_proj block (5,120 chunks, 6,400 cycles at 80%) and then idles.
- **VPU_CL=4** (composite functions on 4 lanes) saves 21 K cycles per token (0.2%); the
  recurrence uses only multiplies, adds and row sums. **MCOLS=4** reads each KV head once
  instead of twice: 9 K cycles at context 128 (80%). At context 1023 the 4&4 build saves 86 K
  cycles over the default, about 21 K of them from VPU_CL.
- Attention, MLP and LM head are what they were for Qwen3: at 99-100% of their bytes.


## What limits it

These are the ISA's and the RTL's limits for this model, found while mapping it, measured
where a number is given. An entrant can change them; this port does not.

- **No multiply-reduce.** `S^T k` is a product pass (writes 8K words per 64-row block) and a
  row-sum pass. A VOP that sums `A * B` along rows would remove two of the 7 passes and a third
  of the TMEM writes.
- **No three-operand update.** `exp(g) St + delta k^T` needs a scale, an outer product and an
  add; a fused multiply-add VOP (or a per-row scale on the outer product) would make it one.
- **TMEM writes, one per bank per cycle (board RPB 64, WPB 1).** The VPU writes 8 words per
  cycle and the DMA's state load competes for the same write slots.
- **The DMA delivers 32 bytes per cycle to TMEM** (8 words; the MXU streams 128): the 2 MiB of
  state per layer take 65 K DMA cycles, hidden here behind the VPU, but a faster recurrence would meet
  them.
- **8 VPU lanes.** Everything above scales with lanes: at 16 lanes the 7 passes would take
  7,200 cycles per head, about the MXU's time for the head's weights at 80%.
- **The VPU has no logarithm**: softplus is a polynomial (about 25 VOPs on 16 values, once per layer;
  negligible time).
- **MCOLS=2** splits the 4-head query groups: each KV head is read twice (1.2 MB at context 128,
  6.4 MB at 1023).

Not attempted: batched decode and chunked prefill (Qwen3's `qwen3_rows`). Qwen3.5 runs one
token per device run (`Engine.step`); the prompt is fed token by token. The state is kept per
sequence in the layer block, so batching would need a state per sequence.

## Tests

`tests/test_qwen35.py`:

- A tiny random Qwen3.5 (`lin lin attn` x 2, 8 DeltaNet heads, a 4-head query group) against
  Hugging Face over 48 tokens and against the int8 emulation, on the design configuration (2
  slices, 8 MXU columns) and the board configuration (1 slice, 2 columns: query groups split).
- `Engine.reset` gives bit-identical logits: position 0 does not read the previous sequence's
  DeltaNet state or convolution rows.
- The layer plan, and the programs at position 4095 fitting the board IMEM (MCOLS 2 and 4).
- The tiny model on the Verilator board model through the host driver, over five tokens (a
  full turn of the convolution ring), bit-identical to the ISA simulator.
- Qwen3.5-0.8B: 8 greedy tokens equal to HF's ("The capital of France is Paris."), and one real
  token on the RTL, bit-exact against the ISA simulator (DRAM: weights, KV cache, convolution
  ring, DeltaNet state; logits).

`otpu-selftest --sim --model qwen35 --tokens 2` also passes: the chat prompt and two generated
tokens (26 tokens) on the Verilator board model through the host driver, identical to the ISA
simulator, 11.8 M cycles per token (72 minutes of simulation; run before rebasing onto eb29dd3).
