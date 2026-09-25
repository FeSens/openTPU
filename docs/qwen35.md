# Qwen3.5 on openTPU (Qwen3.5-0.8B)

openTPU runs a third model family: Qwen3.5, checked with
[Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) (the text decoder; the vision tower
and the multi-token-prediction layer are not loaded), with the same W8A8 numerics and host
driver as Qwen3 and LFM2. Its main layer is a linear-attention recurrence, which runs on the
VPU ops RDOT, OUTER and LOG2 (commit ddec900); the port was first written for the ISA without
them, and both are measured on the RTL below.

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

**The recurrence is three VPU passes.** The layers call `kernels/deltanet.py`'s `head_step`
(the kernel of `gated_deltanet_step`), which uses the VPU ops RDOT (a row dot product against
one vector), OUTER (a rank-1 update with a decay, in place) and LOG2 (commit ddec900). The state
is stored transposed, `St = S^T` (rows are the value dimension), so both reads of S are row
dots:

```
kv = St k                       RDOT
delta = beta (v - exp(g) kv)    vectors of 128
St = exp(g) St + delta k^T      OUTER, in place
o  = St q                       RDOT
```

That is 3 passes over the 16K fp32 values of a head, about 6,100 VPU issue cycles at 8 lanes,
with no temporary the size of the state. The two RDOTs write TMEM only at their end, so the
next head's state load (which takes TMEM write slots first) overlaps them without stalling
the VPU.

**Everything else.** The convolution state is a 4-slot ring of the pre-convolution q, k, v rows
in DRAM (LFM2's scheme, `docs/lfm2.md`). softplus is `max(x, 0) + ln2 log2(1 + 2^(-|x| log2 e))` with the LOG2 op. The `a`
and `b` projections run in int8 like every other matrix: in the float64 emulation, keeping them
in full precision did not reduce the logit error (max 0.98 vs 1.10 and 1.42 vs 1.20 over two
38- and 42-token texts, argmax agreement unchanged within one token). Attention reuses Qwen3's
kernel: a 256-wide head is two MXU blocks, the RoPE tables cover 64 dimensions and the other
192 pass through, and the gate is a fifth projection whose sigmoid is applied to the head
outputs. A query group of 4 heads does not fit a 2-column MXU; there each KV head is streamed
twice, once per pair of query heads (with `OTPU_MCOLS=4`, once).

**Language additions**, no ISA change: `ol.load(desc, out=t)` refills an existing buffer, and
`ol.mxu_columns()` returns MCOLS. The Qwen3 and LFM2 programs assemble to exactly the same
words as before (checked at several positions, both configurations, MCOLS 2 and 4).

**Program size.** The six-fold layer unit is one hardware loop and the head pairs another: the
program is 1,496 instructions at position 0 and 1,896 at position 4095 (board configuration,
MCOLS=2; 1,454 and 1,708 with MCOLS=4), within the 4K-instruction IMEM. The DRAM image is
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
| What is the capital of France? Answer in one sentence. | all 8 (to EOS) | | 1.57 | 0.9970 |
| Explain in two sentences why the sky is blue. | all 38 (to EOS) | | 1.09 | 0.9985 |
| Write a Python function that checks whether a number is prime. | 35 | #2, 0.036 | 1.37 | 0.9946 |
| Give me a short definition of photosynthesis. | 39 | #2, 0.239 | 1.15 | 0.9973 |
| List the first five prime numbers. | all 48 | | 1.00 | 0.9979 |
| Translate 'good morning' into French and Spanish. | all 20 (to EOS) | | 1.23 | 0.9978 |
| What is the capital of Japan, and what is it famous for? | all 48 | | 1.02 | 0.9977 |
| Describe the water cycle in one paragraph. | 8 | #2, 0.114 | 0.81 | 0.9990 |

Raw prompts (the README's eight, 32 tokens): 4 of 8 identical for all 32 tokens. The other
four differ at tokens 4 to 6, where the device's token is HF's #2 or #3, 0.012 to 0.300 logits
below the top. The largest logit error is 1.34, the lowest cosine 0.9972.

HF's logits span about 30 to 55, so an error of 1 to 2 moves only near-ties. At three of the
four raw differences the float64 emulation with the same int8 quantization points picks the
device's token: quantization decides those ties. At the fourth, and at the three chat
differences, it picks HF's: there the device's fp32 rounding decides. (Measured with the LOG2
softplus of ddec900; with the earlier polynomial softplus the counts were 5 of 8 chat and 5 of
8 raw, with the same kind of near-tie differences.) On a tiny random Qwen3.5 the device agrees with Hugging Face to
a cosine above 0.998 over 48 tokens and with the emulation above 0.999 (`tests/test_qwen35.py`;
the emulation also rounds the weights to int8 slightly differently, dividing by the scale
where the device multiplies by 127/amax).

## Performance

One decode token of the full model (24 layers and the LM head), measured on the Verilator RTL
of the board configuration: 1 slice, D=128, 8 VPU lanes, the AXI memory path with the program
booted from DRAM, 30 cycles of AXI latency (`tools/perf_qwen.py --model qwen35 --layers 0
--check`). `bw` is the fraction of peak DRAM bandwidth (one 128-byte chunk per cycle). Every
run ended with DRAM bit-identical to the ISA simulator's.

The **DRAM roofline** is every byte the token must move, at bw: the int8 weights and their fp32
block scales (LM head included), the convolution taps, the DeltaNet state read and written
(2 MiB per layer), the convolution ring, the KV cache read once and appended, and the I/O
(embedding in, logits out). That is 819.1 MB at context 128 and 824.4 MB at 1023, the bytes
the MCOLS=4 program moves; with MCOLS=2 the KV cache is read twice (+1.2 and +6.4 MB), which
counts against it. Tokens/s are projections: measured cycles at an assumed 100 MHz, no host
time.

| MCOLS / VPU_CL | bw | context (pos) | cycles/token (measured) | DRAM roofline | % of roofline | tok/s at 100 MHz (projected) |
|---|---:|---:|---:|---:|---:|---:|
| 2 / 2 (default board) | 80% | 128 | 9,130,829 | 7,999,152 | 87.6% | **11.0** |
| 2 / 2 | 80% | 1023 | 9,231,246 | 8,050,300 | 87.2% | 10.8 |
| 2 / 2 | 100% | 128 | 8,022,432 | 6,399,322 | 79.8% | 12.5 |
| 2 / 2 | 100% | 1023 | 8,099,652 | 6,440,240 | 79.5% | 12.3 |
| 2 / 4 | 80% | 128 | 9,060,803 | 7,999,152 | 88.3% | 11.0 |
| 2 / 4 | 80% | 1023 | 9,162,124 | 8,050,300 | 87.9% | 10.9 |
| 2 / 4 | 100% | 128 | 7,949,977 | 6,399,322 | 80.5% | 12.6 |
| 2 / 4 | 100% | 1023 | 8,029,275 | 6,440,240 | 80.2% | 12.5 |
| 4 / 4 | 80% | 128 | 9,050,858 | 7,999,152 | 88.4% | 11.0 |

(`OTPU_MCOLS=4 OTPU_VPU_CL=4` is the 4&4 bitstream of [board.md](board.md).)

**Without RDOT and OUTER.** The first version of this port ran on the ISA before ddec900: the
recurrence was 7 VPU passes per head (products, row sums, a scale, an outer product through a
row-stride-0 operand, an add) on 64-row blocks, and softplus a log1p polynomial. Measured the
same way (RTL of eb29dd3, commit 2cc24c1 of this port):

| MCOLS / VPU_CL | bw | context 128 | % of roofline | context 1023 | % of roofline |
|---|---:|---:|---:|---:|---:|
| 2 / 2 | 80% | 11,535,257 | 69.3% | 11,636,450 | 69.2% |
| 2 / 2 | 100% | 10,419,105 | 61.4% | 10,495,642 | 61.4% |
| 4 / 4 | 80% | 11,504,083 | 69.5% | 11,549,952 | 69.7% |
| 4 / 4 | 100% | 10,353,455 | 61.8% | 10,386,998 | 62.0% |

The new ops take 2.40 M cycles off the token at 80% (21%), all of it in the DeltaNet mixer.

### Where the cycles go

The same runs split into phases: the instructions, in program order, grouped by the kernel
function that emitted them (`perf_qwen.py` prints this table); a phase is charged the cycles
from the end of the previous phase to its own last completion, so the phases add up to the
token. The bytes column is what the phase moves; its roofline is those bytes at bw.
Default board (MCOLS=2, VPU_CL=2), context 128, measured:

| phase | cycles at 80% | share | its bytes (MB) | its roofline at 80% | cycles at 100% | its roofline at 100% | before RDOT/OUTER, 80% |
|---|---:|---:|---:|---:|---:|---:|---:|
| DeltaNet mixer, 18 layers | 3,411,051 | 37.4% | 236.6 | 67.7% | 3,397,901 | 54.4% | 5,834,217 |
| MLP, 24 layers | 2,650,991 | 29.0% | 272.6 | 100.4% | 2,121,082 | 100.4% | 2,649,504 |
| LM head | 2,574,494 | 28.2% | 263.2 | 99.8% | 2,060,087 | 99.8% | 2,574,507 |
| attention, 6 layers | 459,927 | 5.0% | 47.9 | 101.6% | 372,432 | 100.4% | 459,940 |

(Above 100%: a phase's first weights stream while the phase before it finishes.)

- **The DeltaNet mixer is still VPU-bound, and the rest of the token is at its roofline.** The
  mixer takes 3.40 to 3.41 M cycles at either bandwidth, with the VPU busy 96% of it; its
  236.6 MB (the projections, out_proj, 37.7 MB of state traffic) need 2.31 M cycles at 80%.
  Run at its roofline, the token would take about 8.03 M cycles at 80%, i.e. the roofline.
- **Per head** (288 per token) the mixer spends about 11,800 cycles. The 3 state passes are
  about 6,100 VPU cycles of issue (16K values at 8 lanes; estimated from the op sizes). The rest
  is the per-head vector work on 128- to 512-wide vectors: the convolution, SiLU and L2 norms
  before the recurrence and the gated RMSNorm after it, some 50 small VOPs, each paying its
  pipeline latency, whose composite functions (exp2, recip, rsqrt) run on VPU_CL lanes and
  which wait on each other in a chain. Meanwhile the MXU streams the next head's 512
  projection rows and this head's out_proj block (5,120 chunks, 6,400 cycles at 80%).
- **VPU_CL=4** saves 70 K cycles per token at 80% (0.8%), 66 K of them in the mixer's
  composite functions. **MCOLS=4** reads each KV head once instead of twice: 10 K cycles at
  context 128 (80%).
- Attention, MLP and LM head are what they were for Qwen3: at 99-100% of their bytes.

## What limits it

These are the limits of the ISA and the RTL for this model, found while mapping it; the numbers
are measured where given.

- **Small vector ops per head.** With the recurrence at 3 passes, the per-head vectors
  (convolution, SiLU, L2 norms, gated norm) cost about as much as the state passes. They are
  independent across heads; issuing all 16 heads' short vectors as [16, 128] tiles once per
  layer, before the head loop, would amortize their latency (not attempted: it needs the
  projections of all heads first, i.e. a TMEM buffer of 16 x 512 words, or a second pass).
- **TMEM writes, one per bank per cycle (board RPB 64, WPB 1).** OUTER writes 8 words per
  cycle and the DMA's state load takes the same write slots first; `head_step` places the load
  beside the RDOTs, which write only at their end.
- **The DMA delivers 32 bytes per cycle to TMEM** (8 words; the MXU streams 128): the 2 MiB of
  state per layer take 65 K DMA cycles per layer, 1.2 M per token, hidden behind the VPU.
- **8 VPU lanes.** The state passes and the small ops scale with lanes.
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
simulator, 11.8 M cycles per token (72 minutes of simulation; run with the first version, without
RDOT and OUTER, before eb29dd3).
