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
head (64 KiB) at a time, in two buffers: while the vector unit (VPU) updates one head, the DMA
stores the previous head's state and loads the next one's.

**The heads run in pairs.** Everything small is done for two heads at once, on [2, n] tiles:
the convolution, SiLU, L2 norms and silu(z) before the recurrence ("prep") and the gated
RMSNorm after it ("post"). The input projections are stored pair by pair (the q, k, v rows of
head a, of head b, then their z rows), so one MM gives a pair's 1,024 projections. `out_proj`
is multiplied in per group of 4 heads (K = 512, its natural column block) and accumulated into
the residual. The pairs run in a hardware loop over two pairs, with two sets of the per-pair
buffers (by pair parity); the MXU streams the projections two pairs ahead.

**The recurrence is three VPU passes**, as in `kernels/deltanet.py`'s `head_step` (the kernel of
`gated_deltanet_step`), with the VPU ops RDOT (a row dot product against one vector), OUTER (a
rank-1 update with a decay, in place) and LOG2 (commit ddec900). The state
is stored transposed, `St = S^T` (rows are the value dimension), so both reads of S are row
dots:

```
kv = St k                       RDOT
delta = beta (v - exp(g) kv)    vectors of 128
St = exp(g) St + delta k^T      OUTER, in place
o  = St q                       RDOT
```

That is 3 passes over the 16K fp32 values of a head, about 6,100 VPU issue cycles at 8 lanes,
with no temporary the size of the state.

**The schedule** (`_deltanet` in `llm/qwen35.py`) is built around the TMEM write port: a bank
takes one write per cycle, the DMA and the MXU go first, and a VPU op that writes TMEM stalls
whenever they write one of its banks. The state load is the big writer (16K words per head),
and an RDOT writes nothing until its last row. So a pair's segment runs, in program order,

```
VPU  RDOT1(a)  d(a) OUTER(a)  post(p-1)  RDOT2(a)  RDOT1(b)  d(b) OUTER(b)  prep(p+1)  RDOT2(b)
DMA  load b             store a     load pair p+1's a               store b     taps, ring (p+2)
MXU  projections of pair p+2; out_proj of the group pair p-1 completes
```

with OUTER, RDOT2 and the state loads and stores split in halves of 64 rows: a buffer's store
starts after the first half of its OUTER, and the next head's load into it after the first half
of its RDOT2, so the load lands beside RDOT2 and the next RDOT1. The first pair's projections
are split so that head a starts before head b's rows arrive, and the last group's out_proj is
split in pairs. The sequencer (16-instruction window, each unit starts its oldest ready
instruction) overlaps the rest; the order above is the one that measured best among the
variants tried (explicit fences that force RDOT pairs back to back measured worse).

**Everything else.** The convolution state is a 4-slot ring of the pre-convolution q, k, v rows
in DRAM (LFM2's scheme, `docs/lfm2.md`), stored per pair next to the pair's taps so that one
load brings both (it reads all 4 slots, one more than needed: 0.4 MB per token). softplus is `max(x, 0) + ln2 log2(1 + 2^(-|x| log2 e))` with the LOG2 op. The `a`
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
program is 2,021 instructions at position 0 and 2,475 at position 4095 (board configuration,
MCOLS=2; 1,979 and 2,287 with MCOLS=4), within the 4K-instruction IMEM. The DRAM image is
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
| What is the capital of France? Answer in one sentence. | all 8 (to EOS) | | 1.47 | 0.9976 |
| Explain in two sentences why the sky is blue. | all 38 (to EOS) | | 1.16 | 0.9982 |
| Write a Python function that checks whether a number is prime. | 35 | #2, 0.036 | 1.25 | 0.9946 |
| Give me a short definition of photosynthesis. | 39 | #2, 0.239 | 0.94 | 0.9988 |
| List the first five prime numbers. | all 48 | | 1.02 | 0.9981 |
| Translate 'good morning' into French and Spanish. | 0 | #3, 0.614 | 0.92 | 0.9982 |
| What is the capital of Japan, and what is it famous for? | all 48 | | 1.40 | 0.9975 |
| Describe the water cycle in one paragraph. | 8 | #2, 0.114 | 0.76 | 0.9989 |

Raw prompts (the README's eight, 32 tokens): 4 of 8 identical for all 32 tokens. The other
four differ at tokens 5, 6, 14 and 31, where the device's token is HF's #2, 0.012 to 0.392
logits below the top. The largest logit error is 1.45, the lowest cosine 0.9970.

HF's logits span about 30 to 55, so an error of 1 to 2 moves only near-ties. At two of the
four raw differences the float64 emulation with the same int8 quantization points picks the
device's token: quantization decides those ties. At the other two, and at the four chat
differences, it picks HF's: there the device's fp32 rounding decides.

These are the numbers of the paired-head schedule. Its out_proj sums 4 heads in one MM (K = 512)
instead of adding each head's product into the residual in turn, a different fp32 summation
order; everything else is bit-identical to 891fde4 (checked: with out_proj per head the new
schedule reproduces 891fde4's logits bit for bit over a 25-token chat prefill). The different
rounding changes the device's logits by 0.12 on average (0.66 at most) after that prefill, i.e.
it re-draws which near-ties the int8 quantization noise tips: the translation prompt's first
token, a three-way near-tie on the device ('Bon' 17.63, '**' 17.54, 'Pour' 17.05), now goes the
other way. With 891fde4 the chat prompts gave 5 of 8 identical (max error 1.57, min cosine
0.9946) and the raw prompts 4 of 8 (1.34, 0.9972), with the same kind of near-tie differences.
Summing out_proj per head costs 223 K cycles per token at 80% (8,630,306 instead of 8,407,188;
2.6%), because the MXU then writes 4 times as many outputs into TMEM, each stalling a writing
VPU op. (Measured with the LOG2 softplus of ddec900; with the earlier polynomial softplus the
counts were 5 of 8 chat and 5 of 8 raw.)

On a tiny random Qwen3.5 the device agrees with Hugging Face to a cosine above 0.998 over 48
tokens and with the emulation above 0.999 (`tests/test_qwen35.py`; the emulation also rounds the
weights to int8 slightly differently, dividing by the scale where the device multiplies by
127/amax).

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
| 2 / 2 (default board) | 80% | 128 | 8,407,188 | 7,999,152 | 95.1% | **11.9** |
| 2 / 2 | 80% | 1023 | 8,509,044 | 8,050,300 | 94.6% | 11.8 |
| 2 / 2 | 100% | 128 | 7,446,693 | 6,399,322 | 85.9% | 13.4 |
| 2 / 2 | 100% | 1023 | 7,523,772 | 6,440,240 | 85.6% | 13.3 |
| 2 / 4 | 80% | 128 | 8,365,927 | 7,999,152 | 95.6% | 12.0 |
| 2 / 4 | 80% | 1023 | 8,469,064 | 8,050,300 | 95.1% | 11.8 |
| 2 / 4 | 100% | 128 | 7,302,209 | 6,399,322 | 87.6% | 13.7 |
| 2 / 4 | 100% | 1023 | 7,381,718 | 6,440,240 | 87.2% | 13.5 |
| 4 / 4 | 80% | 128 | 8,356,727 | 7,999,152 | 95.7% | 12.0 |

(`OTPU_MCOLS=4 OTPU_VPU_CL=4` is the 4&4 bitstream of [board.md](board.md). Context 1023 runs
with `--cap 1024`. One run, 2/4 at 80% and context 1023, first reported a DRAM mismatch against
the ISA simulator while another job hit "no space left on device" on the same disk; the same run
repeated three times (on a disk with 28 GiB free) was bit-identical at the same cycle count.
A truncated simulator dump used to read as zeros; `rtlsim` now rejects a short DRAM or TMEM
dump, so a full disk fails loudly instead of looking like a mismatch. `perf_qwen.py` also
refuses `--pos` at or past `--cap`, which writes the KV cache out of bounds and does not match
the ISA simulator; its default `--cap` is now the multiple of 256 above `--pos`.)

**Before pairing the heads** (commit 891fde4: one head at a time, the small vector work per head,
out_proj per head; same measurement):

| MCOLS / VPU_CL | bw | context 128 | % of roofline | context 1023 | % of roofline |
|---|---:|---:|---:|---:|---:|
| 2 / 2 | 80% | 9,130,829 | 87.6% | 9,231,246 | 87.2% |
| 2 / 2 | 100% | 8,022,432 | 79.8% | 8,099,652 | 79.5% |
| 2 / 4 | 80% | 9,060,803 | 88.3% | 9,162,124 | 87.9% |
| 2 / 4 | 100% | 7,949,977 | 80.5% | 8,029,275 | 80.2% |
| 4 / 4 | 80% | 9,050,858 | 88.4% | | |

Pairing takes 0.72 M cycles off the token at 80% (7.9%) and 0.58 M at 100%, all of it in the
DeltaNet mixer.

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

| phase | cycles at 80% | share | its bytes (MB) | its roofline at 80% | cycles at 100% | its roofline at 100% | per head (891fde4), 80% | before RDOT/OUTER, 80% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DeltaNet mixer, 18 layers | 2,704,642 | 32.2% | 237.1 | 85.6% | 2,876,145 | 64.4% | 3,411,051 | 5,834,217 |
| MLP, 24 layers | 2,650,991 | 31.5% | 272.6 | 100.4% | 2,121,082 | 100.4% | 2,650,991 | 2,649,504 |
| LM head | 2,574,511 | 30.6% | 263.2 | 99.8% | 2,060,091 | 99.8% | 2,574,494 | 2,574,507 |
| attention, 6 layers | 459,941 | 5.5% | 47.9 | 101.6% | 372,432 | 100.4% | 459,927 | 459,940 |

(Above 100%: a phase's first weights stream while the phase before it finishes.)

- **The DeltaNet mixer is still VPU-bound, and the rest of the token is at its roofline.** The
  mixer takes 2.70 M cycles at 80% and 2.88 M at 100%, with the VPU busy 94% of it at 80%; its
  237.1 MB (the projections, out_proj, 37.7 MB of state traffic) need 2.32 M cycles at 80%.
- **Per head** (288 per token) the mixer spends about 9,400 cycles at 80% (11,800 before).
  One DeltaNet layer alone, profiled on the RTL (MLP and LM head removed, 80%, context 128:
  150.9 K cycles), splits as follows. The VPU's issue work, from the op sizes (8 columns per
  cycle, 2 for exp2, recip, rsqrt, log2; reductions padded to 64 columns), is 118.9 K cycles:
  about 99 K for the 3 state passes of 16 heads, 8.3 K for the composite functions (SiLU of
  1,024 values per pair: exp2 and recip at 2 lanes), about 12 K for the rest. The VPU lost 14.5 K cycles to
  TMEM write conflicts (the MXU's projection outputs while OUTER runs, about 10% of OUTER's
  cycles, and loads that overrun an RDOT). The first RDOT starts 7.6 K cycles into the layer
  (the first pair's projections), and the last out_proj ends 3 K after the last RDOT. The
  layer's port-B bytes need 128 K cycles at 80%.
- **VPU_CL=4** saves 41 K cycles per token at 80% (0.5%), 40 K of them in the mixer's
  composite functions. **MCOLS=4** reads each KV head once instead of twice: 9 K cycles at
  context 128 (80%).
- Attention, MLP and LM head are what they were for Qwen3: at 99-100% of their bytes.

## What limits it

These are the limits of the ISA and the RTL for this model, found while mapping it; the numbers
are measured where given.

- **VPU issue work.** Per pair of heads the VPU has about 14.9 K cycles of issue work, 12.4 K
  of them the state passes, against 16.0 K cycles of port-B traffic at 80% (the projections,
  out_proj and the state): at 80% the mixer could reach its roofline if nothing else were lost,
  at 100% (12.8 K) it stays VPU-bound. Pairing the small ops removed most of their latency, not their issue cycles:
  the composite functions (SiLU's exp2 and recip) issue at VPU_CL = 2 lanes.
- **TMEM writes, one per bank per cycle (board WPB 1; reads never conflict).** A VPU op that writes stalls on
  any cycle the DMA or the MXU writes one of its banks; an RDOT writes only at its end (and that
  final write waits for a running load). The state loads are placed beside RDOTs, but the MXU's
  projection outputs (one per 8 chunks streamed) still cost OUTER about 10%. The DMA's state
  load itself runs at about 5.5 words per cycle while the MXU streams (1.5 K cycles per 64-row
  half), not the 8 of its TMEM port. A third state buffer, which would free the loads from the
  stores, does not fit TMEM (3 x 16K words plus the pair buffers exceed 64K).
- **The sequencer's window** (16 instructions; each unit starts its oldest ready instruction)
  decides the overlap: with ~35 small VOPs per pair in flight, placing an instruction early
  in program order is the only control. Explicit fences (a one-element op that makes an RDOT
  wait for the end of prep) measured 1-10% worse per layer than leaving the order to the
  sequencer.
- **8 VPU lanes.** The state passes and the small ops scale with lanes. With 16 lanes / TMEM
  banks (the MXU and quantizer on 8 of them; `OTPU_LANES=16 OTPU_ULANES=8 OTPU_VPU_CL=2`),
  measured: 8,068,007 cycles at 80% (99.3% of the useful bytes, -4.0%) and 6,510,050 at 100%
  (98.5%, -12.6%). Qwen3 and LFM2 gain 0.1-0.2%. Not in the default bitstream (`make bit
  LANES=16`: an estimated +21K LUT, not built).
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
