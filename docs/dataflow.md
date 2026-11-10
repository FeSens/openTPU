# Dataflow: where the bytes go, and folding the VPU into the MXU

Question (the user, 2026-10-12): "Can we fold the VPU into the MXU without compromising on the
accelerator performance / decode performance / prefill performance? What would be the gain of
that?" With the user's rules: maximize the architecture, not one FPGA; no SRAM copies (of TMEM or
otherwise); data movement is the name of the game; slices are NUMA regions; first principles
win. The lead's additions: rank by total share (not share of writes), DRAM bytes per token first
(decode is DRAM-bound), then TMEM round trips; the V^T append and the DeltaNet state round trip
as first-class items; per-family tables by context and by batch; KV options in bytes; a KV
format change is a numerics decision (dKL over >= 2000 tokens).

Base: the MX build (`mx-column` 8b33206e: MX column, E = 4 shared epilogue lanes, ACT_ROWS 16,
PAIR, DSTEP, STREAM), the card's arch (D 128, MCOLS 4, LANES 8, TMEM 64K words), fp4x6 layers
and the int8 LM head. Every number says how it was obtained:

- **simulated (ledger)**: `tools/dataflow.py`, an instrumented ISA run of the compiled program on
  a zero image (section 1); exact counts of what the program moves, no timing.
- **co-sim**: the RTL in Verilator with LiteDRAM's DDR3-1066 controller and the core at 133.33
  MHz (`rtlsim.BOARD_UARCH`), cycles; the upper bounds of section 5 are co-sim runs of
  timing-only program transforms.
- **estimated**: arithmetic on the above, or published figures (labeled with their source).

Nothing here ran on the card.

**Short answer**

- **Yes: the fold costs neither decode nor prefill, and it is what lets the array grow.** The
  VPU's elementwise work moves into the MXU's datapath at its two ends: the quantizer's input
  (QACT / QST modes: EXP + SUM for the softmax, GLU, NORM, RoPE, the mask), the drain's ACC (the
  residual, today), and the stream engine's input for DeltaNet (section 7.1). Every fused op is
  today's fp32 work in today's order (two one-rounding moves aside, a dKL check), the drain's
  lanes stay as they are, and no co-sim bound is slower than its base by more than 0.2%. The one
  condition: QUANT's composite unit needs MCOLS / 2 lanes, the floor at long context (~75% busy
  in a 4000-token tile at MCOLS 16), and MCOLS of them for 16-row tiles there.
- **The gain is small on today's card and grows with the columns and the context** (co-sim
  upper bounds, section 5): decode 0.6-3.1% at MCOLS 4, 8 and 16 (decode stays DRAM-bound);
  prompt runs of 3-4 rows 3-12%; MoE layers 1.6-3.3%; Qwen3-0.6B's 16-row prompt tile -21% at
  MCOLS 8 and **-37% at MCOLS 16** (-26% / -49% at 4000). Without the fold MCOLS 8 -> 16 buys 16%
  of that tile; with it, 47%. **At MCOLS 16 the VPU is the prefill bottleneck**: the project adopted
  the fold as the precondition for scaling MCOLS (2026-10-12). Beside the time: TMEM's words
  -53 to -75%, the single pumped TMEM copy's cost (+1.1-1.7% against six copies, co-sim) gone,
  area about neutral (the composite unit moves from the VPU to QUANT), and under 2% of a
  token's data-movement energy: the fold moves on-chip words, not DRAM bytes.
- **Data movement is DRAM, and DRAM is weights and KV.** A decode token moves 0.16-2.8 GB of
  DRAM (simulated); everything on chip (TMEM, ACT RAM writes, IMEM) is 2-11% of those bytes in
  decode and 17-29% in prefill and batched steps, and at published energy per bit (section 6)
  DRAM is ~75% (HBM-class) to ~93% (DDR-class) of a decode token's data-movement energy.
- **KV is the lead item for long context and multi-user**: 38.6% of a Qwen3-0.6B token at 4K,
  83% per user at batch 8 (simulated). The format decides it, and the accuracy limits it (dKL
  over 2000 tokens on 10 card models, section 4.8): **4-bit K fails on every model** (+0.3 to
  +14.8%), 4-bit V passes only Qwen3-0.6B's batched decode. **6-bit V** (int6 with a token's
  scale: today's V format at 6 bits) passes at batch 8 / 4K on every model but Gemma 4 and
  single-user on Qwen3-0.6B from 2K: -12% of the KV bytes (Qwen3-0.6B 7.2 KB a context token,
  -4.7% of a 4K token, -10.1% per user at batch 8), ~3-4K LUT; 6-bit K still fails where K
  matters, so K stays int8 (a next-release proposal, section 4.8). **N-pair** (two d 64 K rows
  a chunk, int8, no numerics change) recovers LFM2's padded K: -6.4% of an LFM2.5-230M token at
  4K. The V^T append (1.05-1.37% of a decode token, co-sim, infra-hunt) is second-tier.
- **The fold, by class** (section 3): 95% of Qwen3-0.6B's VPU issue cycles can leave the VPU
  (36% to the MXU's output path now, 19% more with weight-layout changes, 40% to QUANT's input
  path: the softmax numerators); 5% stays (row scalars, init). At 4K, softmax P is 81% of the
  VPU's cycles and the QUANT path takes it. Qwen3.5 keeps its DSTEP (57% of its VPU + SE time)
  in the stream engine; multi-row DSTEP (seq-vpu) cuts its state's bytes, and its time with a
  wider engine (-19% of a Qwen3.5-0.8B prompt run with two chained engines, co-sim bound).
- **ACT RAM reads gated to the MM's columns: done** (20627927, section 4.10): cycle- and
  bit-exact; a decode MM reads 1 of MCOLS columns instead of all (-52 to -88% of the reads).
- **Slices (S > 1, section 4.12)**: fan-out MMs split by output, fan-in MMs by K, one
  all-reduce of H a block (adopted): 0.56-0.64x the output split's link bytes, 0.23-0.33x the K
  split's. It is also how prompt tiles grow past the TMEM: at MCOLS 16, 64-row tiles need ~4x
  the TMEM a row, or S > 1.
- **Staged** (section 7.3). Release r2 carries the ACT gating (20627927) and the ledger and
  co-sim tools; everything else is a next-release item. For the MCOLS 8 build: the compiler-only
  stage T, the quantizer's GLU and EXP + SUM modes (GE: -10% / -21% of the 16-row tile at 544 /
  4000), N-pair and, if approved, int6 V. For MCOLS 16: the rest of the fold (NORM, RoPE, the
  mask, the drain into QUANT's row buffer, a 16-word quantizer read, E 8, the narrowed VPU).
  Later: S > 1, the corner-turning V FIFO, multi-row DSTEP's engines, k-major weight tiles.

## 1. Method: the ledger

`tools/dataflow.py MODEL KIND N POS` compiles a case as `tools/vpu_hack.compile_case` does
(decode at a position, `rows` R prompt rows, `batch` B users, MoE decode on layer prefixes with
every expert resident; the card's PAIR / DSTEP / STREAM), runs it on `isasim` with counting
slices (`LedgerSlice`), and writes a JSON per run; `tools/dataflow.py report` tables them. Per
instruction, attributed to its opcode class and its source site (the compiler's call chain):

- DRAM bytes read and written by kind: weights and their scale rows, the KV cache (K, V^T and
  their scales, MM-streamed or LD), the KV appends (QST: useful bytes, whole 64-byte beats
  written, beats read for the merge), the DeltaNet state, LD / ST, programs (HALT CHAIN);
- TMEM words per read port (MXU, DMA, Q, Q2, Q3, VA, VB, COLL, SE) and write port (DMA, MXU,
  VPU, COLL), with the chunk cycles each takes;
- ACT RAM bytes written (QACT, with DUP) and read: logically (each output row's blocks) and
  physically (MCOLS rows a replay group, what the RTL reads), and with the read enables gated to
  the MM's columns (section 4.10);
- IMEM words fetched, VPU / stream engine issue cycles (vpu_hack's issue model), MXU advances,
  port A lines and port B chunks;
- producer -> consumer flows: every TMEM word carries its writer and its first reader, so a
  read is a hand-off (read once) or a reuse; DRAM beats likewise.

Each VOP gets a role from its site and function (residual, norm statistics / apply, sandwich
post-norm, qk-norm, RoPE, layout pad, GLU, gate, attention output, softmax numerators / row
scalars / init, mask, short conv, DeltaNet front / gates / post, PLE, sampling, MoE) and a
fusion verdict (section 3). Zero images keep the runs fast (an MM's math is skipped; its
counters are exact). Weights fp4x6 (`OTPU_FORMATS` unset), the head int8.

Models: all dense card models whole (Qwen3-0.6B, SmolLM3-3B, Phi-4-mini, Qwen3.5-0.8B / 2B / 4B,
LFM2.5-230M, LFM2-2.6B, Gemma 4 E2B); the big MoE models on layer prefixes (Qwen3.5-35B-A3B
layers 1-4, Gemma 4 26B-A4B layers 0, 0-1, 0-5), per layer by difference.

## 2. The ledger (task 1)

### 2.1 DRAM per decode token, by context (MB; simulated)

| model | pos | weights | scales | KV read | KV append (RMW) | DeltaNet state | other | total | KV + append |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 544 | 375.8 | 20.9 | 35.0 | 3.76 | | 0.9 | 436.2 | 8.9% |
| | 2048 | | | 124.8 | | | | 526.1 | 24.4% |
| | 4096 | | | 245.9 | | | | 647.2 | 38.6% |
| | 8192 | | | 488.1 | | | | 889.4 | 55.3% |
| SmolLM3-3B | 544 | 1668.8 | 123.6 | 22.5 | 2.41 | | 1.1 | 1818.4 | 1.4% |
| | 2048 | | | 80.2 | | | | 1876.2 | 4.4% |
| | 4096 | | | 158.1 | | | | 1954.1 | 8.2% |
| | 8192 | | | 313.8 | | | | 2109.8 | 15.0% |
| Phi-4-mini | 544 | 2225.2 | 118.8 | 39.9 | 4.29 | | 1.6 | 2389.9 | 1.9% |
| | 2048 | | | 142.6 | | | | 2492.6 | 5.9% |
| | 4000 | | | 273.5 | | | | 2623.5 | 10.6% |
| Qwen3.5-0.8B | 544 | 503.1 | 24.9 | 3.7 | 0.40 | 37.7 | 5.7 | 575.5 | 0.7% |
| | 4096 | | | 26.1 | | | | 597.8 | 4.4% |
| | 8192 | | | 51.7 | | | | 623.5 | 8.4% |
| Qwen3.5-2B | 544 | 1194.9 | 56.4 | 3.7 | 0.40 | 37.7 | 5.9 | 1299.0 | 0.3% |
| | 8192 | | | 51.7 | | | | 1347.0 | 3.9% |
| Qwen3.5-4B | 544 | 2420.2 | 135.7 | 9.9 | 1.06 | 100.7 | 9.6 | 2677.2 | 0.4% |
| | 8192 | | | 137.9 | | | | 2805.2 | 5.0% |
| LFM2.5-230M | 544 | 148.4 | 8.0 | 5.5 | 0.41 | | 0.6 | 162.9 | 3.6% |
| | 4096 | | | 39.7 | | | | 197.1 | 20.4% |
| | 8192 | | | 79.0 | | | | 236.4 | 33.6% |
| LFM2-2.6B | 544 | 1351.6 | 91.2 | 7.4 | 0.55 | | 2.0 | 1452.8 | 0.5% |
| | 8192 | | | 105.4 | | | | 1550.6 | 6.8% |
| Gemma 4 E2B | 544 | 1341.1 | 75.4 | 25.9 | 0.60 | | 2.3 | 1445.3 | 1.8% |
| | 4096 | | | 77.2 | | | | 1496.6 | 5.2% |
| | 8192 | | | 136.6 | | | | 1556.0 | 8.8% |

(Blank cells: as the row above. Phi-4-mini's range ends at 4096, so 4000. Gemma's KV grows
slower than its context: sliding layers keep 512 tokens and 20 of its 35 layers share KV.)

- **Weights dominate at short context; KV takes over with position, at a rate set by KV heads x
  head dim x attention layers against the parameter count.** Qwen3-0.6B (28 layers x 8 KV heads
  x 128, 0.6B parameters) passes half its token in KV by 8K; the hybrids (Qwen3.5: 6 of 24
  layers attend, 2 KV heads; LFM2: a few attention layers among short convolutions; Gemma's
  sliding window and shared KV) stay under 9%, except the small LFM2.5-230M (34% at 8K).
- **Scales are 5.6-7.4% of every token** (fp4x6's 4.25 bits; the int8 head's per-row scales).
- **The int8 LM head is 155 MB of Qwen3-0.6B's 376 MB of weights** (151,936 x 1,024): 36% of a
  short-context token. The head is a decode-only cost (prefill runs it on the last row).
- **The DeltaNet state round trip** (DSTEP: fp32 state in and out of DRAM each token) is
  constant: 37.7 MB (6.6% of Qwen3.5-0.8B's token, 2.9% of 2B's), 100.7 MB (3.8% of 4B's).
- **The V^T append**: each token writes one byte into each of d V^T rows per KV head, each byte
  its own 64-byte beat, read-merged and written back (`otpu_native_dram`'s ECC read-modify-write):
  3.76 MB of DRAM traffic a Qwen3-0.6B token to store 28.7 KB (131x). 73% of the token's written
  beats at 544 (infra-hunt), but 0.9% of its bytes, and 0.6% at 4K. Co-sim (infra-hunt, the
  same model, every append QST dropped): 1.35% of the token's cycles at 544, 1.37% at 2K, 1.05%
  at 4K; each partial beat costs 1.7-2.1 cycles (row opens, turnarounds). Ranked by total share,
  second-tier; per user it grows with batch (2.1 below).
- **Programs**: 21-58 KB a token, nothing.

### 2.2 Per user, batched decode (MB per user; simulated)

| model | pos | B | weights | scales | KV read | KV append | other | total per user | KV + append |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 544 | 1 | 375.8 | 20.9 | 35.0 | 3.76 | 0.87 | 436.2 | 8.9% |
| | | 4 | 93.9 | 5.2 | 35.0 | 3.76 | 0.07 | 137.9 | 28.1% |
| | | 8 | 47.0 | 2.6 | 35.0 | 3.76 | 0.04 | 88.3 | 43.8% |
| | 4096 | 1 | 375.8 | 20.9 | 245.9 | 3.76 | 0.87 | 647.2 | 38.6% |
| | | 4 | 93.9 | 5.2 | 245.9 | 3.76 | 0.07 | 348.9 | 71.6% |
| | | 8 | 47.0 | 2.6 | 245.9 | 3.76 | 0.04 | 299.3 | 83.4% |
| SmolLM3-3B | 544 | 1 | 1668.8 | 123.6 | 22.5 | 2.41 | 1.12 | 1818.4 | 1.4% |
| | | 4 | 417.3 | 30.9 | 22.5 | 2.41 | 0.22 | 473.3 | 5.3% |
| | 4096 | 1 | 1668.8 | 123.6 | 158.1 | 2.41 | 1.12 | 1954.1 | 8.2% |
| | | 4 | 417.3 | 30.9 | 158.1 | 2.41 | 0.22 | 608.9 | 26.4% |

Batched steps share the weights and keep each user's KV, so KV's share grows with B as fast as
the weights' falls. (SmolLM3 at batch 8 and Phi-4-mini at batch 4 and 8 do not fit 64K TMEM
words in one step program; the card splits them.) The V^T append is per user: 4.3% of a
Qwen3-0.6B user's bytes at batch 8 / 544.

### 2.3 Prefill, per row (MB; simulated; compile_rows, 4 rows at 544)

| model | R | weights | scales | KV read | KV append | state | other | total per row | / decode token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 4 | 55.1 | 4.8 | 35.0 | 0.96 | | 0.07 | 95.8 | 0.22 |
| SmolLM3-3B | 4 | 351.5 | 30.3 | 22.5 | 0.62 | | 0.16 | 405.1 | 0.22 |
| Qwen3.5-0.8B | 4 | 62.2 | 5.5 | 3.7 | 0.10 | 37.7 | 1.17 | 110.4 | 0.19 |
| Qwen3.5-2B | 4 | 171.6 | 12.9 | 3.7 | 0.10 | 37.7 | 1.22 | 227.2 | 0.17 |
| LFM2.5-230M | 4 | 20.3 | 1.8 | 5.5 | 0.11 | | 0.12 | 27.9 | 0.17 |
| Gemma 4 E2B | 4 | 234.6 | 17.8 | 26.0 | 0.60 | | 0.40 | 279.4 | 0.19 |
| Phi-4-mini | 3 | 536.9 | 37.7 | 40.0 | 1.45 | | 0.27 | 616.3 | 0.26 |
| LFM2-2.6B | 3 | 405.8 | 30.0 | 7.4 | 0.19 | | 0.77 | 444.1 | 0.31 |
| Qwen3.5-4B | 2 | 892.3 | 64.9 | 9.9 | 0.54 | 100.7 | 4.30 | 1072.5 | 0.40 |

(Weights per row are 1 / R of decode's minus the head. Phi-4-mini, LFM2-2.6B and Qwen3.5-4B do
not fit TMEM at 4 rows in this path: run at 3 / 3 / 2.)

Two per-row terms do not shrink with R: each compile_rows row streams the KV again (Qwen3-0.6B
36% of its row), and each row does its own DeltaNet state round trip (Qwen3.5-0.8B 34% of its
row, 2B 17%). The tile kernel (tiling, `docs/tiled_prefill.md`) shares K / V^T across a tile's
rows (Qwen3-0.6B at 16 rows: 1.11 MB a row of K/V re-stream for the first layers, against 2.20
at 8); nothing shares the DeltaNet state yet: multi-row DSTEP (section 4.9).

### 2.4 MoE decode, per layer (MB; simulated; prefix differences, every expert resident)

| model | layer | weights (experts + dense) | scales | KV | state | other | total |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-35B-A3B | 1-2 (DeltaNet), each | 31.52 | 2.56 | | 4.19 | 0.35 | 38.62 |
| Qwen3.5-35B-A3B | 3 (attention) | 28.31 | 2.20 | 1.24 | | 0.02 | 31.83 |
| Gemma 4 26B-A4B | 1 (sliding) | 79.30 | 2.93 | 2.47 | | 0.09 | 85.06 |
| Gemma 4 26B-A4B | 5 (full) | 93.72 | 3.05 | 2.45 | | 0.09 | 99.44 |

The routed experts' weights are 80-95% of a MoE layer; KV and state are small at 544. The
offloaded case adds the link (`docs/offload.md`), which this ledger does not model.

### 2.5 On chip (KB per token / row / user; simulated)

| run | DRAM | TMEM r | TMEM w | VPU's share of TMEM words | TMEM / DRAM | ACT w | ACT r logical | ACT r physical | IMEM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B decode @544 | 436,228 | 14,592 | 11,573 | 55% | 6.0% | 1,160 | 705,433 | 1,741,479 | 547 |
| Qwen3-0.6B decode @4096 | 647,197 | 40,358 | 30,815 | 48% | 11.0% | 4,572 | 1,140,215 | 2,611,043 | 1,732 |
| Qwen3-0.6B rows 4 | 95,838 | 13,724 | 10,777 | 59% | 25.6% | 944 | 540,218 | 612,509 | 369 |
| Qwen3-0.6B batch 8 | 88,327 | 14,293 | 11,349 | 59% | 29.0% | 945 | 705,433 | 777,633 | 359 |
| SmolLM3-3B decode @544 | 1,818,440 | 38,422 | 33,744 | 45% | 4.0% | 1,786 | 3,359,962 | 7,185,265 | 765 |
| SmolLM3-3B rows 4 | 405,073 | 37,453 | 32,792 | 47% | 17.3% | 1,126 | 3,080,994 | 3,080,994 | 340 |
| Phi-4-mini decode @544 | 2,389,908 | 27,981 | 22,865 | 55% | 2.1% | 1,817 | 4,199,332 | 9,622,168 | 537 |
| Qwen3.5-0.8B decode @544 | 575,467 | 21,169 | 17,786 | 65% | 6.8% | 440 | 814,359 | 2,153,592 | 383 |
| Qwen3.5-0.8B rows 4 | 110,407 | 17,628 | 14,300 | 75% | 28.9% | 242 | 544,207 | 544,207 | 142 |
| Qwen3.5-4B decode @544 | 2,677,161 | 48,396 | 40,407 | 66% | 3.3% | 1,381 | 4,508,846 | 10,327,296 | 736 |
| LFM2.5-230M decode @544 | 162,922 | 4,616 | 3,861 | 54% | 5.2% | 326 | 255,284 | 653,174 | 149 |
| LFM2-2.6B decode @544 | 1,452,774 | 23,870 | 20,510 | 61% | 3.1% | 1,290 | 2,759,690 | 5,774,477 | 306 |
| Gemma 4 E2B decode @544 | 1,445,329 | 36,172 | 28,645 | 70% | 4.5% | 1,551 | 2,530,240 | 5,807,818 | 376 |
| Gemma 4 E2B rows 4 | 279,377 | 34,307 | 26,742 | 74% | 21.9% | 915 | 2,102,558 | 2,102,558 | 188 |

(TMEM words x 4 bytes. ACT reads include the 8-byte scale record a block.)

- **TMEM traffic is 2-11% of the DRAM bytes in decode and 17-29% in prefill and batched steps;
  the VPU's two read ports and its write port carry 45-75% of it.**
- **The largest on-chip stream is the ACT RAM's read**: every MXU advance reads MCOLS rows of D
  bytes and their records, M = 1 or not (and the RTL read every cycle, popping or not): 1.7 GB a
  Qwen3-0.6B decode token at 544, 4x the token's DRAM bytes, for 0.7 GB of useful reads.
  Section 4.10 gates it.
- **IMEM is small** (0.1-0.3% of DRAM bytes).

**TMEM hand-offs** (Qwen3-0.6B decode at 544, K words a token; a hand-off is a word read exactly
once):

| writer -> reader | words read | hand-offs | what |
|---|---:|---:|---|
| MXU -> MXU | 430.1K | 430.1K | ACC partials (the down projection's F chunks, flash P.V blocks) |
| VPU:glu -> VPU:glu | 430.1K | 430.1K | the SiLU chain's intermediates (exp2, 1 + e, recip, products) |
| VPU:softmax.p -> QACT | 286.7K | 57.3K | the numerators to P's quantization |
| VPU:softmax.p -> VPU:softmax.p | 258.9K | 0 | EXP2SUB's passes, the row sums |
| MXU -> VPU:glu | 258.0K | 86.0K | gate / up outputs |
| MXU -> VPU:softmax.p | 244.2K | 244.2K | the scores (round trip: MXU drain -> TMEM -> VPU) |
| LD -> QACT | 174.1K | 173.1K | the embedding row, V's per-token scales (CSCALE) |
| MXU -> VPU:qknorm | 172.0K | 0 | q / k out of their projections |
| VPU:qknorm -> VPU:rope -> VPU:rope | 172.0K + 172.0K | 0 + 172.0K | normalized q / k through RoPE's halves |
| MXU -> ST | 151.9K | 151.9K | the logits to DRAM |
| VPU:glu -> QACT | 86.0K | 86.0K | the GLU product to the down projection's operand |
| VPU:rope -> QACT, VPU:attn.out -> QACT | 57.3K each | 57.3K each | q to the scores MM; the attention output to W_o |

3.59M words read a token, 2.31M of them hand-offs (64%): most of TMEM's traffic is a value
written by one unit and read once by the next. Every MXU -> VPU -> QACT chain is a round trip
the fold removes (MXU drain -> TMEM -> VPU -> TMEM -> QUANT becomes MXU drain -> QUANT).

## 3. The fusion map (task 2)

Every VOP class by role: its **class** (the bound's grouping, section 5: **a** foldable with
today's numerics, **a\*** with a weight-layout or rounding-order change, **b** needing the
quantizer's input path, **c** stays, **se** the stream engine, **x** a layout copy the compiler
drops) and its **home** in the design of section 4: the MXU's **drain** (the epilogue lanes),
the **QUANT** input (QACT / QST: between TMEM and the ACT RAM or the KV cache), the **SE** input
(DSTEP's operands), or the remaining **VPU**. Shares: of the run's VPU + stream-engine issue
cycles (simulated; composite passes, i.e. EXP2 / RECIP / RSQRT / LOG2's extra passes, in
brackets), and of the run's TMEM words (r + w).

| role | class | home | what, and why there | Q3 @544 | Q3 @4K | Smol rows 4 | Q3.5-0.8B | E2B | TMEM words (Q3 / Q3.5 / E2B) |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| residual | a | drain | x + o: ACC into the residual stream on W_o's / W_down's drain (exists) | 2.1 | 0.6 | 2.6 | 0.6 | | 2.6 / 1.5 / - |
| norm.stat | a | QUANT | the residual row's sum of squares beside the max the next QACT finds anyway; the row scale absorbs 1 / rms (NORM) | 2.3 (0.1) | 0.7 | 2.6 | 0.6 | 3.4 (0.1) | 0.9 / 0.5 / 1.3 |
| norm.apply | a | QUANT | the gain: CSCALE (exists) | | | | | 3.1 | - / - / 2.6 |
| qknorm | a\* | QUANT / QST | q's and k's per-head RMS: the gain as CSCALE before RoPE, 1 / rms into q's row scale (QACT) and k's per-token scale (QST); one rounding moves | 10.1 (0.3) | 2.9 | | 0.6 | | 7.2 / 0.8 / - |
| rope | a\* | QUANT / QST | rotate-half pairs (i, i + d/2) within the head's row the quantizer reads, cos / sin as two operands (Q2, Q3) | 9.6 | 2.8 | 5.0 | 0.1 | 3.5 | 10.2 / 0.3 / 3.4 |
| glu | a | QUANT | act(g) x u from the gate's and up's raw outputs (two operands) into the down projection's operand | 28.8 (16.0) | 8.4 (4.7) | 62.3 (34.6) | 9.4 (5.2) | 62.2 (25.9) | 18.4 / 12.4 / 45.9 |
| gate | a | QUANT | Qwen3.5's attention output gate (sigmoid), as the GLU, into W_o's operand | | | | 1.0 (0.7) | | - / 1.0 / - |
| attn.out | a | QUANT | the flash finish (o / l): W_o's QACT row scale (RSCALE, exists) | 2.3 (0.1) | 0.7 | 1.3 | 0.3 | 1.3 | 1.8 / 0.6 / 1.1 |
| ple | a | QUANT | Gemma's per-layer input: GELU times the layer's embedding, as the GLU | | | | | 3.4 (0.7) | - / - / 3.0 |
| softmax.p | b | QUANT | exp2(s c - m) of the scores (m: RMAX on q.K^T's drain, exists), V's per-token scale (CSCALE, exists), the row sums; P quantized in the same pass (EXP + SUM) | 40.1 (27.6) | 80.7 (59.8) | 24.2 (16.7) | 1.4 (1.0) | 10.3 (7.1) | 12.7 / 0.9 / 3.2 |
| mask | b | QUANT | the additive mask before the exp2 | | | | | 1.8 | - / - / 1.2 |
| dn.post | b | QUANT | DeltaNet's gated RMSNorm (o / rms x g x SiLU(z)): NORM with a GLU-like operand | | | | 5.5 (2.3) | | - / 7.4 / - |
| conv | a\* | SE | the 4-tap short conv of in_proj's outputs ahead of DSTEP: the stream engine's input (its state ring) | | | | 9.4 | | - / 23.8 / - |
| dn.front | a\* | SE | q / k L2 norms and SiLU ahead of DSTEP: the engine's input, which has the composites | | | | 12.8 (6.9) | | - / 14.8 / - |
| pad | x | (none) | zero rows / copies for layouts | | | | 0.1 | | - / 0.2 / - |
| postnorm | c | VPU | Gemma's sandwich norm: norm(o) is added to the residual, not quantized: a second pass | | | | | 9.1 (0.1) | - / - / 8.0 |
| softmax.row | c | VPU | flash's per-row scalars (max, sum, rescale factors) | 1.4 (0.6) | 2.3 (1.0) | 0.4 | 0.0 | 0.2 | 0.3 / 0.0 / 0.1 |
| softmax.init | c | VPU | FILLs of the flash accumulators (an MM without ACC on the first block drops most) | 3.3 | 1.0 | 1.7 | 0.2 | 1.7 | 1.3 / 0.2 / 0.7 |
| dn.gates | c | VPU | a few scalars a head (softplus, sigmoid) | | | | 0.1 | | - / 0.1 / - |
| sample, moe, misc | c | VPU | sampling (batched steps: 5.4%), routing, bookkeeping | | | | 0.3 | | |
| dn.state | se | SE | DSTEP: the state stream | | | | 57.5 | | - / 1.5 / - |
| **a** | | | | **36%** | **10%** | **69%** | **12%** | **73%** | |
| **a + a\* + x** | | | | **55%** | **16%** | **74%** | **35%** | **77%** | |
| **+ b** | | | | **95%** | **97%** | **98%** | **42%** | **89%** | |
| **c** | | | | **5%** | **3%** | **2%** | **1%** | **11%** | |
| **se** | | | | 0 | 0 | 0 | **57%** | 0 | |

(Q3: Qwen3-0.6B decode; Smol: SmolLM3-3B; Q3.5-0.8B and E2B decode at 544. Other runs: LFM2.5
4 rows a 50%, a + a\* + x 68%, + b 97%, c 3%; Qwen3.5-0.8B 4 rows as its decode.)

**The rule behind the homes: each value is transformed on its way into its consumer.** Nearly
every VPU chain ends in a quantizer (QACT into an MM's operand, QST into the KV cache) or in
DSTEP. The quantizer already reads every element of the row, finds its max and multiplies
per-column and per-row factors; the elementwise work of the chain (norms, RoPE, activations,
the softmax numerators) can ride that one read. Only the residual add belongs on the drain
(ACC, as today); only values that are neither quantized nor streamed next (Gemma's sandwich
norms, flash's scalars, sampling) need the VPU.

**What the classes say**

- **Two roles are most of the VPU: the GLU and the softmax numerators** (together 69-89% of the
  attention models' VPU cycles: the GLU 8-62%, the softmax 10-81%, the softmax's share growing
  with context). They are also the composite passes (SiLU's exp2 and recip, EXP2SUB): 33-65% of
  all VPU cycles are composites in those two roles (`docs/seq_vpu.md` section 5 uses this: a narrower composite
  path costs decode more at long context). The fold's value is mostly these two.
- **Softmax goes to QUANT, not the drain.** The scores leave q.K^T's drain at G per cycle (d 128,
  KB 1: one K row an advance), up to MCOLS in prefill tiles; exp2 per score on the drain would
  need G to MCOLS composite results a cycle, and P needs the row's max first anyway (RMAX on the
  drain gives it, already). QUANT reads the scores once after the MM, at LANES a cycle, which is
  the read P's quantization does today. Rate check in 4.3.
- **What stays is small**: row scalars, init, Gemma's sandwich norms (11% of E2B), DeltaNet's
  gate scalars, sampling, routing: 1-11% of today's VPU cycles. DSTEP stays in the stream engine
  (57% of Qwen3.5's VPU + SE time; its prefill fix is multi-row, section 4.9).
- **TMEM words**: the same roles carry the VPU's TMEM traffic (GLU 12-46%, softmax 1-31%, conv /
  DeltaNet front 39% of Qwen3.5's), so the fold takes 45-75% of TMEM's words with it (2.5).

## 4. Design sketch (task 3)

The design in one line: **the MXU's own datapath absorbs the VPU's work at its two ends.** On the
way in, QUANT (QACT into the ACT RAM, QST into the KV cache) gets the elementwise transforms of
its operand: norms, RoPE, activations, the softmax numerators. On the way out, the drain keeps
the residual add (ACC, today). The stream engine's input does the same for DSTEP's operands. The
VPU keeps what is neither quantized nor streamed next.

### 4.1 Rates: what the drain gives, what QUANT must take

The array finishes an output row (its M outputs, M <= MCOLS a replay group) every advances-a-row
cycles: KB (K / 128) without PAIR, KB / 2 with PAIR (4-bit, 2M <= MCOLS). Outputs a cycle
(estimated from the formats; D = 128):

| MM | K | decode (M 1, PAIR) | MCOLS 4 prefill (M 4) | MCOLS 8 prefill (M 8) | MCOLS 16 prefill (16 rows) |
|---|---:|---:|---:|---:|---:|
| q / k / v / o, gate / up at H 1024 | 1024 | 0.25 | 0.5 | 1 | 2 |
| the same at H 2048-3072 | 2048-3072 | 0.08-0.13 | 0.17-0.25 | 0.33-0.5 | 0.67-1 |
| W_down's F chunk (C 256) | 256 | 1 | 2 | 4 | 8 |
| q.K^T (d 128: KB 1, int8) | 128 | G (2-4; G 8: 4 with replay) | 4 | 8 | 16 |
| P.V (256-token blocks, int8 V^T) | 256 | G / 2 | 2 | 4 | 8 |

- **The drain's lanes are sized by the drain alone** (dsp-pump's rule: E lanes keep a pass of r
  rows over KB blocks at the array's rate iff ceil(r / E) <= KB, KB / 2 with PAIR): at E =
  MCOLS / 2 every MM keeps it but prefill q.K^T at d 128 (2x). The fold adds nothing to them
  (4.2).
- **A pipelined fp32 composite is big**: exp2 is 9 multiply-adds after the range reduction,
  recip 6 (`otpu_se_comp`'s op lists, bit-exact with `opentpu/fp32.py`); a lane's fmadd alone is
  ~0.6K LUT and 2 DSP (mx-rtl-prod). Composites belong in one place, microcoded over a few
  stages per lane as `otpu_se_comp` does today: QUANT's input, where the values that need them
  (the softmax's numerators, the GLU's gate, the gates, GELU) all go next.

### 4.2 The drain's lanes: unchanged by the fold (one lane design at MCOLS 16)

Today's lane (`otpu_mx_lane`, one output a cycle): a 48-bit integer from the result FIFO -> i2f48
(RNE) -> x Sw -> x (Sa x 2^-W) -> a TMEM write, or ACC / ASCALE (read the old y, fmadd y_old x
alpha + y_new), with RMAX. Measured (mx-rtl-prod / dsp-pump, Vivado OOC of otpu_mxu at MCOLS 4,
xc7k480t-2, 7.5 ns): E 2 / 3 / 4 = 12,311 / 14,116 / 16,171 LUT, 562 / 568 / 574 DSP: **~1.93K
LUT, 1.32K FF, 6 DSP a lane**. Split (Yosys and bit counts, mx-rtl-prod): i2f48 319 LUT; the two
multiplies 250 LUT, 4 DSP; the ACC / ASCALE fmadd 608 LUT, 2 DSP; RMAX ~250-280 (MCOLS 4; ~1K at
16 rows unless it leaves the lane); the pick ~120 (~350-450 at 16 rows); the TMEM port mux
~64-100. The result FIFO is shared (5.2K LUTRAM at MCOLS 16). **At MCOLS 16** (mx-rtl-prod
1d9b82f7, `docs/mx_mxu.md` section 10; Vivado OOC of mx_top, 8 pumped DSP48E1 columns, 16 rows,
133.33 / 266.67 MHz, timing met): E 4 = 29,402 LUT, 62,611 FF, 1,128 DSP; E 8 = 33,041 LUT,
68,618 FF, 1,152 DSP: **a lane at 16 rows 910 LUT, 1,502 FF, 6 DSP** (the 16:1 pick adds the
flip-flops). The drain's cycles an output (ceil(M / E) / KB): decode's attention <= 1 at E 4
and 8; projections <= 0.5; a full 16-row q.K^T at d 128 4.0 at E 4, 2.0 at E 8 (d 256 or P.V:
2.0 / 1.0).

| lane | at MCOLS 16, E 8 | what | cost a lane |
|---|---|---|---|
| full lane (today's) | 8 | i2f48, x Sw, x Sa', write, ACC / ASCALE, RMAX | 910 LUT, 1,502 FF, 6 DSP at 16 rows (measured, mx_top OOC); ~1.93K LUT in production otpu_mxu at MCOLS 4 |
| score lane (optional) | up to 8 more | no ACC / ASCALE: q.K^T and P.V's first block | the full lane less ~640 LUT, 450 FF, 2 DSP (mx-rtl-prod's split, estimated) |

- **The fold adds nothing here**: the residual's ACC exists; every other fused op lives at
  QUANT's input (4.3). An earlier sketch put the norms' statistic and RoPE in "projection lanes"
  (~1.3K LUT, 6 DSP each); QUANT reads the same rows anyway, with the pairs of a head in one
  row, so it does them without TMEM reads of partners or tables through the drain's port.
- **Score lanes are the drain's question**, not the fold's: up to 8 lanes (8 x 0.3-1.3K LUT,
  32 DSP, estimated) to bring a full 16-row q.K^T at d 128 from 2.0 drain cycles an output to the
  array's rate at E 8; decode and the projections do not need them. The fold makes it more pressing: with the softmax
  off the VPU, q.K^T's drain is the next limit of prefill attention (section 5, MCOLS 16).

### 4.3 QUANT's input path: the activation engine

QACT today reads a row (Q port), optionally multiplies a per-column (CSCALE, Q2) and a per-row
(RSCALE, Q3) factor, finds the row's max, and quantizes into the ACT RAM; QST does the same into
the KV cache. The fold gives the quantizer's input a composite unit (`otpu_se_comp`'s, moved:
NS multiply-add stages a lane, a function's passes looping) and these modes:

| mode | the quantizer computes on its way in | replaces |
|---|---|---|
| EXP (+ SUM) | p = exp2(s c - T[m + r]) after an optional additive mask operand, times V's per-token scale (CSCALE); the row's sum of p to T[l + r] | EXP2SUB's 3 passes, RSUM, the mask's add, P's write and read |
| GLU | act(g) x u (act SiLU / GELU / sigmoid), u on the second operand port | the SiLU chain's 8 VOPs and the product's write and read (GLU, gates, PLE, DeltaNet's gated norm with NORM) |
| NORM | the row's (or each head's) sum of squares beside its max; 1 / rms folded into the row scale (the quantized values of y / rms are y's) | RMSNorm's statistic and scaling passes (qk-norm per head) |
| ROPE | (x_lo, x_hi) -> (x_lo c - x_hi s, x_hi c + x_lo s) over a head's two halves in the row, cos / sin on the second and third operand ports | RoPE's VOPs and their round trips |

Flash's rescale stays ASCALE on P.V's drain and 1 / l is W_o's RSCALE (both exist). qk-norm's
gain is CSCALE, applied before the rotation (today: after the normalization; one rounding
moves: a\*); NORM's 1 / rms moves from the values to the scale (one rounding: a\*). Everything
else is today's fp32 operations in today's order.

**Rates** (estimated). The softmax needs ~MCOLS / 2 exp2 a cycle to keep pace with q.K^T and P.V
(per KV head and 256-token block: decode at MCOLS 4, G 2: 256 + 256 advances for 512 scores; a
16-row tile at MCOLS 8, G 2: 1,024 + 1,024 advances for 8,192 scores), the GLU M / 16 SiLU a
cycle in prefill (0.5 at MCOLS 8, 1 at 16). A composite unit of MCOLS / 2 lanes x 3 stages
(EXP2 in 3 passes, as `otpu_se_comp` with NS 3) gives MCOLS / 2 exp2 a cycle: 12 multiply-add
stages at MCOLS 8, 24 at MCOLS 16 (today's se_comp: 24). The quantizer's read width must follow:
LANES 8 words a cycle matches MCOLS 8; MCOLS 16 tiles at long context need 16. Per element it
is one pass over the scores instead of today's five, and one over the GLU's operands instead of
nine.

**The stream engine's input** gets the same treatment for DSTEP's operands (Qwen3.5's short
conv, the q / k L2 norms, SiLU): the engine already has the composites (`otpu_se_comp` is its
part) and reads those operands; its input stage applies them.

### 4.4 What remains of the VPU

The c classes (row scalars, FILLs, Gemma's sandwich norms, DeltaNet's gate scalars, sampling,
routing: 1-11% of today's VPU cycles). Its width is seq-vpu's call with these shares
(`docs/seq_vpu.md` 5: today's composite width matters at long context only because of the
softmax, which leaves). Two loads keep it from shrinking to a scalar unit: sampling over the
vocabulary (151,936 logits a token: 19K cycles at 8 lanes, 76K at 2, against a 3.6M-cycle
token) and Gemma's sandwich norms (9% of E2B's VPU today). Its composites (a row's rsqrt, LOG2
for sampling) are rare: QUANT's composite unit can serve them (it is idle outside quantizer
passes) or the VPU keeps a 1-2 lane path.

### 4.5 TMEM ports, single copy

TMEM words a token by port, today and with the fused classes' VPU reads and writes removed
(simulated today; the fused rows estimated: the VPU's words of each fused class removed, the
operands the fused path reads added; QUANT's added operand reads, GLU's u and RoPE's cos / sin,
not yet counted):

| run | mode | r MXU | r Q | r Q2 | r VA | r VB | w MXU | w VPU | all ports |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B decode @544 | today | 489 | 660 | 202 | 1,718 | 427 | 1,229 | 1,475 | 6,541 |
| | ASB | 621 | 660 | 202 | 7 | 5 | 1,143 | 94 | 3,073 (-53%) |
| | ASB + Q | 621 | 402 | 202 | 7 | 5 | 885 | 94 | 2,557 (-61%) |
| SmolLM3-3B rows 4 | today | 3,319 | 1,134 | 129 | 3,689 | 1,092 | 4,685 | 3,395 | 17,561 |
| | ASB + Q | 3,536 | 516 | 129 | 9 | 7 | 3,671 | 102 | 8,088 (-54%) |
| Qwen3.5-0.8B rows 4 | today | 449 | 240 | 22 | 2,626 | 876 | 856 | 2,503 | 7,982 |
| | ASB + Q | 943 | 86 | 22 | 6 | 1 | 617 | 77 | 2,162 (-73%) |
| Gemma 4 E2B rows 4 | today | 881 | 879 | 84 | 4,996 | 1,727 | 1,984 | 4,558 | 15,262 |
| | ASB + Q | 926 | 325 | 84 | 597 | 151 | 1,091 | 544 | 3,872 (-75%) |
| LFM2.5-230M rows 4 | today | 148 | 177 | 38 | 505 | 160 | 337 | 464 | 1,884 |
| | ASB + Q | 223 | 102 | 38 | 1 | 1 | 227 | 14 | 662 (-65%) |

(K words; DMA, Q3, COLL and SE ports unchanged and in the total. "+ Q": the outputs that go
straight to a QACT pass through QUANT's row buffer instead of TMEM: the MXU -> TMEM -> Q round
trip of the GLU's operands, q and the attention output.)

- **VA and VB fall to 0.1-15% of today's**, the VPU's write port to 2-12%: TMEM's words fall
  53-75%. With one pumped TMEM copy (tmem-pump's r2pgRs, NSB 2: the pick) the replays come
  from requesters meeting on a bank in the same cycle; the fold removes the two busiest and,
  with the row buffer, half of Q's reads. Measured (5.3): the single copy's replays fall 98-99%
  and its cost against six copies goes from +1.1-1.7% to ~0.
- **QUANT's operand ports carry more**: the GLU's u, RoPE's cos / sin (Q2, Q3); the drain's (MXU)
  port carries nothing new.
- **"+ Q" needs the whole row before its scale** (QACT's per-row fp32 Sa is the row's max): the
  drain writes QUANT's row buffer (XD 1024 words, `docs/mx_mxu.md` "row buffer"), and rows longer
  than the buffer quantize in parts. Block-local activation scales (OCP MX's E8M0 a 32) would
  remove the condition at a numerics cost; not proposed.

### 4.6 ISA and compiler

ISA (a proposal; bit-exact except where marked):
- QACT / QST: input modes EXP with a SUM output (the row's sum to an address), GLU (a function
  code; the second operand's address), NORM (a sum-of-squares span: the row or the head; a\*),
  ROPE (the head size; cos / sin addresses), an additive mask operand. They compose with CSCALE /
  RSCALE (which exist) in a fixed order: mask, EXP / GLU, ROPE, CSCALE, NORM's scale, quantize.
- MM: none. Optional: MM into QUANT's row buffer ("+ Q").
- DSTEP: its operands' transforms (conv taps, L2 norm, SiLU) as input modes of the stream engine.

Compiler: a VOP chain from an MM's output to a QACT / QST (`lib.silu` / `gelu_tanh` / `sigmoid`
products, `rmsnorm`, `rope`, the softmax in `attention.finish`) becomes the quantizer's modes; the
flash init FILLs a first MM without ACC; the residual's VOP ADD the MM's ACC. WAITC (seq-vpu) is
unaffected: the fused work completes with its QACT.

### 4.7 Slices and columns

- **NUMA-local**: a slice transforms the operands it quantizes; nothing crosses a slice to be
  activated. Rows that are gathered from all slices before a QACT today (`kernels/mlp`'s
  `all_gather(act(g) * u)`) are gathered raw instead (the same bytes) and transformed by each
  slice's quantizer.
- **Pairs stay in a row**: RoPE's pairs are within a head's row; the GLU's g and u of an F range
  are in the same slice (`kernels/mlp.swiglu_down`).
- **Columns**: E = MCOLS / 2 lanes for the drain (4.2); QUANT's composite lanes and read width
  at MCOLS / 2 and up to MCOLS (4.3).

### 4.8 KV: the layout and the format (the lead item)

Today (`opentpu/llm/qwen3.py` `_qk` / `_pv`): K rows [T, d] int8 with a per-token fp32 Sw (RSC),
streamed by q.K^T as the weight operand (KB = d / 128 blocks a row); V as V^T rows [d, T] int8,
streamed UNIT by P.V, with V's per-token scale folded into P by QACT's CSCALE. Two costs follow:
a K row narrower than a chunk takes a whole chunk (d 64: LFM2's K stream reads 2x its bytes),
and the V^T append writes one byte into each of d beats (2.1).

**The options, in bytes and MXU / drain cycles** (`tools/dataflow.py kv`: estimated from the
ledger's runs at 4K: the KV
bytes by MM and kind, the attention MMs' advances; per user for batch B; "step" = the change of
the whole step's DRAM bytes; MCOLS 8, MCOLS 4 in brackets where it differs):

| model (G, d) | B | KV + append today | V rows | N-pair int8 K | fp4 V | fp4 K | fp4 K + V | fp4 K + V, V rows | q.K^T outputs an advance today -> fp4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B (2, 128) | 1 | 249.7 MB (38.6%) | -0.6% | 0 | -9.4% | -7.9% | -17.3% | -17.9% | 2 -> 4 |
| | 4 | (71.6%) | -1.0% | 0 | -17.4% | -14.7% | -32.1% | -33.1% | |
| | 8 | (83.4%) | -1.2% | 0 | -20.2% | -17.2% | -37.4% | -38.6% | |
| SmolLM3-3B (4, 128) | 1 | 160.5 MB (8.2%) | -0.1% | 0 | -2.0% | -1.7% (+0.2%) | -3.7% (-1.8%) | -3.8% (-1.9%) | 4 -> 8 |
| | 4 | (26.4%) | -0.4% | 0 | -6.4% | -5.4% (+0.8%) | -11.8% (-5.6%) | -12.2% (-6.0%) | |
| Phi-4-mini (3, 128), 4000 | 1 | 277.8 MB (10.6%) | -0.2% | 0 | -2.6% | -2.2% (+0.3%) | -4.7% (-2.2%) | -4.9% (-2.4%) | 3 -> 6 |
| Qwen3.5-0.8B (4, 256) | 1 | 26.5 MB (4.4%) | -0.1% | 0 | -1.1% | -0.9% | -2.0% | -2.1% | 2 -> 4 |
| Qwen3.5-4B (4, 256) | 1 | 70.5 MB (2.6%) | 0 | 0 | -0.6% | -0.5% | -1.2% | -1.2% | 2 -> 4 |
| LFM2.5-230M (2, 64) | 1 | 40.1 MB (20.4%) | -0.2% | **-6.4%** | -3.3% | -5.6% | -8.9% | -9.1% | 2 -> 4 |
| LFM2-2.6B (4, 64) | 1 | 53.5 MB (3.6%) | 0 | -1.1% (0) | -0.6% | -1.0% (+0.1%) | -1.6% (-0.4%) | -1.6% (-0.5%) | 4 -> 8 |
| Gemma 4 E2B (8, 256 / 512) | 1 | 77.8 MB (5.2%) | 0 | 0 | -1.3% | -1.1% | -2.4% | -2.5% | 4 -> 4 |

Qwen3-0.6B at 8K: KV 55.3% of the token, fp4 K + V -25.0%, fp4 V -13.4%; at 544 -3.7% (B 1),
-18.1% (B 8).

- **In bytes, the format is the lever: fp4 K and V halve the KV bytes**, -17% of a Qwen3-0.6B
  token at 4K and -25% at 8K, -37% per user at batch 8 / 4K, and halve the attention MMs'
  advances where pairing applies. The time follows the bytes in decode (DRAM-bound).
  - **fp4 K** (fp4x6, the weights' MX format: per-token Sw, a scale word a 128-element block): d/2
    + 12 bytes a row against d + 4. At d <= D a fp4 row is half a chunk, so the bytes come only
    with **N-pair**: one chunk holds two tokens' rows, columns j < M take the low row and
    columns j + M the high one (PAIR's mechanism, with the two halves drained as two outputs
    instead of summed; 2M <= MCOLS). It doubles q.K^T's outputs an advance (2G): the drain
    needs E >= 2G to keep the halved advances (Qwen3 G 2 at E 4 yes; SmolLM3 / Phi G 3-4 at E 4
    no: their q.K^T keeps its time but its bytes halve; at MCOLS 4 they cannot pair at all,
    hence the brackets).
  - **fp4 V**, two forms: token-major E2M1 values with one scale a token (max |v| / 6; UNIT, the
    scale in P's CSCALE as today's), or V^T rows in a weight format (fp4x6 / NVFP4: one Sw a
    channel and attention block, the format's scales along the tokens), whose scales a token
    appended alone cannot set: it needs sealing (option 2 or 1 below). d/2 bytes a token either
    way; P.V pairs (PAIR) where 2G <= MCOLS.

**The accuracy decides it, and it says no to 4-bit K** (the rule: dKL against the MX int8
KV, 2000 tokens of `docs/isa.md`, int8x weights, every other point as the device; `tools/formats_scan.py
ppl MODEL ';kv=MODE'`, the float64 emulations with the KV format under test, Gemma 4's through
`tools/gemma4_quant_eval.py`; dKL in % with its paired SE; simulated. The gains: fp4 V alone,
the step's DRAM bytes at MCOLS 8 from the table's ledger runs; the formats rule: dKL <= 0.1 x
gain, a default 1 SE under):

| model | K fp4x6 | K NVFP4 | V E2M1 (token) | V^T fp4x6 | V^T NVFP4 | fp4 V's gain: 544 / 2K / 4K / 8K (B 1); batch | best V passes |
|---|---:|---:|---:|---:|---:|---|---|
| LFM2-2.6B | +2.29 (0.11) | +1.99 (0.09) | +1.02 (0.03) | +0.76 (0.04) | +1.45 (0.09) | 0.1 / 0.3 / 0.6 / 1.1 | V^T fp4x6: none |
| LFM2.5-230M | +2.94 (0.07) | +2.92 (0.08) | +0.76 (0.02) | +0.66 (0.02) | +0.64 (0.03) | 0.6 / 1.9 / 3.3 / 5.4 | V^T NVFP4: none |
| Phi-4-mini-instruct | +5.12 (0.20) | +4.28 (0.20) | +1.17 (0.06) | +2.12 (0.06) | +1.71 (0.06) | 0.4 / 1.4 / 2.6 (4000) | V E2M1 (token): none |
| Qwen3-0.6B | +14.76 (0.46) | +11.65 (0.37) | +1.72 (0.07) | +2.09 (0.08) | +1.90 (0.07) | 2.1 / 5.9 / 9.4 / 13.4; B4@544: 6.7, B8@544: 10.4, B4@4096: 17.4, B8@4096: 20.2 | V E2M1 (token): B8@4096 |
| Qwen3.5-0.8B | +0.39 (0.01) | +0.33 (0.01) | +1.25 (0.03) | +0.31 (0.01) | +0.29 (0.01) | 0.2 / 0.6 / 1.1 / 2.0 | V^T NVFP4: none |
| Qwen3.5-2B | +0.30 (0.01) | +0.28 (0.01) | +0.76 (0.02) | +0.21 (0.01) | +0.19 (0.01) | 0.1 / 0.3 / 0.5 / 0.9 | V^T NVFP4: none |
| Qwen3.5-4B | +0.34 (0.01) | +0.36 (0.03) | +0.54 (0.03) | +0.17 (0.01) | +0.14 (0.01) | 0.1 / 0.3 / 0.6 / 1.2 | V^T NVFP4: none |
| SmolLM3-3B | +5.18 (0.18) | +4.09 (0.13) | +1.30 (0.05) | +1.73 (0.07) | +1.51 (0.07) | 0.3 / 1.1 / 2.0 / 3.6; B4@544: 1.2, B4@4096: 6.4 | V E2M1 (token): none |
| gemma-4-E2B-it | +9.23 (0.54) | +8.79 (0.54) | +23.49 (1.39) | +2.74 (0.23) | +1.85 (0.18) | 0.5 / 0.8 / 1.3 / 2.2 | V^T NVFP4: none |
| gemma-4-E4B-it | +3.23 (0.22) | +3.28 (0.21) | +3.90 (0.34) | +1.43 (0.17) | +1.52 (0.15) | 0.3 / 0.5 / 0.7 / 1.2 | V^T fp4x6: none |

- **4-bit K fails on every model.** Qwen3-0.6B +14.8% even with its K smoothing (`qk_gains`),
  SmolLM3 and Phi-4-mini +5.1-5.2%, Gemma 4 E2B-it +9.2% and E4B-it +3.2%, LFM2 +2.3-2.9%;
  Qwen3.5's few attention layers +0.3-0.4%.
  NVFP4's finer scales take up to a quarter of the loss off, not enough: no K variant meets its
  bar at any measured context or batch (fp4 K's gains, 0.1-17% of a step, set bars of 0.01-1.7%).
  One scale a token and a block is too coarse for the scores: the softmax exponentiates their
  errors. Per-channel K scales (KIVI's form) vary along q.K^T's reduction: they fold into q only
  by quantizing q again for each block of tokens (a QACT a block and head), not costed here. Not
  proposed.
- **4-bit V passes only where the KV dominates the step.** Token-major E2M1 is the better V on
  Qwen3 and SmolLM3, V^T NVFP4 / fp4x6 on Qwen3.5, LFM2 and Gemma 4 (their V channels vary more
  than their tokens; E2B-it's token-major E2M1 V costs +23.5%). Against the rule, single-user
  decode passes on no model up to 8K: the best V costs 0.1-1.9% of dKL for 0.1-13% of bytes. **Qwen3-0.6B's batched decode passes at 4K** (batch 8: -20.2% per
  user against E2M1's +1.72 (0.07), a bar of 2.02: 4 SE under), with the cheapest hardware of
  the options (token-major E2M1 needs no sealing).
- **N-pair alone, with int8 K, needs no numerics change** and recovers LFM2's padded K: -6.4% of
  an LFM2.5-230M token at 4K, -10.6% at 8K, -1.1% of LFM2-2.6B's at 4K (d 64: two K rows a chunk;
  MCOLS 8 for LFM2-2.6B's G 4). The free item of this section.
- **The layout (V token-major instead of V^T) is -0.1 to -1.2% of the bytes**, more in time
  (infra-hunt's co-sim: the appends cost 1.05-1.37% of a Qwen3-0.6B token, partial beats being
  slow), and per user it grows with batch. Three ways to keep P.V's operand transposed without
  the read-merge-writes, costed per KV head, layer and token at d 128:
  1. *Token-major V in DRAM, a corner-turning MXU FIFO.* The append writes 2 whole beats. The
     MXU's chunk FIFO stores each D-byte chunk across D byte-wide banks with the bank rotated by
     the entry index, so a "transposed" read takes byte n of D consecutive entries in one cycle:
     V^T's chunk (n, k) out of D V rows. Every byte is written and read once, as today (no extra
     SRAM traffic: infra-hunt's objection to a separate transposer, 2 T d bytes a head a step,
     does not apply when the transposer *is* the FIFO). A 256-token attention block of d 128 is
     256 entries of the 1024-entry FIFO, so the MXU keeps its row order (row n's KB chunks are
     the block's bytes n) and the bits are today's. Cost: D byte-banks with their own addresses
     and two D-byte rotators (FPGA: ~10K LUT and ~2x the FIFO's block RAMs, estimated; an ASIC
     takes banked SRAM cheaply). Gemma's d 512 fills the FIFO (block 128 there).
  2. *Blocked V^T sealed every 64 tokens* (infra-hunt (b)): the open block in a token-major tail,
     transposed into whole beats when it fills; 128-384 B a token, but P.V over the open tail
     needs option 1's transposer or a second MM form.
  3. *Byte-masked partial writes* (no ECC merge): halves today's traffic; needs ECC off or per
     byte for the KV region. Smallest change, smallest gain.
  Option 1 also makes a V^T format's sealing natural (a 16-token NVFP4 group is 16 FIFO
  entries) and costs nothing in time. It is a next-generation item.

**6 bits: the point between 4 and 8** (2026-10-12: with no 4-bit K and almost no
4-bit V, the next numerics point is between). Two forms, emulated as above (`;kv=int6k`, `fp6k`,
`int6v`, `fp6vt`): **int6**, today's formats with values in +-31 (K: int8x's quantizer, a
per-token Sw and the D-blocks' power-of-two offsets; V: one scale a token, folded into P as
today's), and **MXFP6** (OCP MX: E2M3 values, an E8M0 scale a 32-element block; K along d, V^T
along an attention block's tokens, sealed as 4.8's V^T formats). Both store 3/4 of int8's value
bytes (a d-128 row in 96 bytes), the scales as today's. dKL against the MX int8 KV, 2000 tokens,
simulated; the gains as above (the step's DRAM bytes at MCOLS 8 from the ledger runs; batch 8 at
4K estimated from the B 1 run with the step's non-KV bytes shared by 8 users, which reproduces
Qwen3-0.6B's batch-8 ledger to 0.1 point; LFM2's 6-bit K counted past N-pair, which packs its
d 64 K at int8 for free):

| model | K int6 | K MXFP6 | V int6 (token) | V^T MXFP6 | K + V^T MXFP6 | 6-bit V's gain: 544 / 2K / 4K / 8K (B 1) | passes (a default: 1 SE under 0.1 x gain) |
|---|---:|---:|---:|---:|---:|---|---|
| LFM2-2.6B | +0.19 (0.01) | +0.52 (0.02) | +0.13 (0.02) | +0.12 (0.02) | +1.00 (0.03) | 0.0 / 0.2 / 0.3 / 0.5; B8@4K (est.) V 2.0, K 1.9 | V int6 (token): B8@4K (est.); V^T MXFP6: B8@4K (est.) |
| LFM2.5-230M | +0.49 (0.03) | +0.54 (0.02) | +0.14 (0.01) | +0.06 (0.01) | +0.57 (0.02) | 0.3 / 0.9 / 1.6 / 2.7; B8@4K (est.) V 6.9, K 6.7 | K int6: B8@4K (est.); K MXFP6: B8@4K (est.); V int6 (token): B8@4K (est.), 4096, 8192; V^T MXFP6: B8@4K (est.), 2048, 4096, 8192; K + V^T MXFP6: B8@4K (est.) |
| Phi-4-mini-instruct | +0.78 (0.04) | +1.06 (0.05) | +0.13 (0.02) | +0.18 (0.02) | +1.25 (0.05) | 0.2 / 0.7 / 1.3 (4000); B8@4K (est.) V 5.9, K 5.7 | V int6 (token): B8@4K (est.); V^T MXFP6: B8@4K (est.) |
| Qwen3-0.6B | +4.53 (0.18) | +2.36 (0.09) | +0.08 (0.03) | +0.29 (0.04) | +2.65 (0.09) | 1.1 / 3.0 / 4.7 / 6.7; B8@4K (est.) V 10.1, K 9.8 (ledger: 10.1, 9.8) | V int6 (token): B8@4K (est.), 2048, 4096, B4@4096, B8@4096, B4@544, B8@544, 8192; V^T MXFP6: B8@4K (est.), 4096, B4@4096, B8@4096, B4@544, B8@544, 8192 |
| Qwen3.5-0.8B | +0.08 (0.01) | +0.05 (0.01) | +0.20 (0.01) | +0.00 (0.01) | +0.06 (0.01) | 0.1 / 0.3 / 0.5 / 1.0; B8@4K (est.) V 3.3, K 3.2 | K int6: B8@4K (est.), 8192; K MXFP6: B8@4K (est.), 8192; V int6 (token): B8@4K (est.); V^T MXFP6: B8@4K (est.), 2048, 4096, 544, 8192; K + V^T MXFP6: B8@4K (est.), 4096, 8192 |
| Qwen3.5-2B | +0.07 (0.01) | +0.06 (0.01) | +0.11 (0.01) | +0.01 (0.00) | +0.06 (0.01) | 0.0 / 0.1 / 0.2 / 0.5; B8@4K (est.) V 1.7, K 1.7 | K int6: B8@4K (est.); K MXFP6: B8@4K (est.); V int6 (token): B8@4K (est.); V^T MXFP6: B8@4K (est.), 2048, 4096, 8192; K + V^T MXFP6: B8@4K (est.), 8192 |
| Qwen3.5-4B | +0.07 (0.01) | +0.07 (0.01) | +0.07 (0.00) | +0.01 (0.00) | +0.07 (0.01) | 0.0 / 0.2 / 0.3 / 0.6; B8@4K (est.) V 2.1, K 2.1 | K int6: B8@4K (est.); K MXFP6: B8@4K (est.); V int6 (token): B8@4K (est.); V^T MXFP6: B8@4K (est.), 2048, 4096, 8192; K + V^T MXFP6: B8@4K (est.), 8192 |
| SmolLM3-3B | +1.18 (0.06) | +0.65 (0.05) | +0.17 (0.02) | +0.21 (0.02) | +0.81 (0.04) | 0.2 / 0.5 / 1.0 / 1.8; B8@4K (est.) V 5.1, K 4.9 | V int6 (token): B8@4K (est.), B4@4096; V^T MXFP6: B8@4K (est.), B4@4096; K + V^T MXFP6: B8@4K (est.) |
| gemma-4-E2B-it | +1.70 (0.24) | +1.49 (0.38) | +5.82 (0.35) | -0.04 (0.11) | +0.94 (0.35) | 0.2 / 0.4 / 0.7 / 1.1; B8@4K (est.) V 3.9, K 3.6 | V^T MXFP6: B8@4K (est.), 8192 |
| gemma-4-E4B-it | +0.60 (0.06) | +0.47 (0.07) | +0.38 (0.09) | +0.17 (0.07) | +0.62 (0.11) | 0.1 / 0.2 / 0.4 / 0.6; B8@4K (est.) V 2.4, K 2.2 | V^T MXFP6: B8@4K (est.) |

**The bytes, a context token** (what each decode step reads per token of context, at 4K and B 1;
the cache's size for all but Gemma 4, whose KV-sharing layers read their source's cache again and
whose sliding layers stop at the window; LFM2's base is N-pair's int8 K; estimated from the
ledger's runs, `tools/dataflow.py kv`):

| model | KV a context token, int8 today (KB, 4K) | 6b V saves | 6b K saves | 6b K+V saves | fp4 V saves | step at 4K: 6b V / 6b K / both (B 1) |
|---|---:|---:|---:|---:|---:|---|
| LFM2-2.6B (N-pair K) | 8.6 | 1.0 (12%) | 1.0 (12%) | 2.0 (24%) | 2.1 (24%) | 0.3% / 0.3% / 0.6% |
| LFM2.5-230M (N-pair K) | 6.5 | 0.8 (12%) | 0.8 (12%) | 1.5 (24%) | 1.5 (24%) | 1.6% / 1.6% / 3.2% |
| Phi-4-mini-instruct | 66.8 | 8.2 (12%) | 8.0 (12%) | 16.2 (24%) | 16.4 (25%) | 1.3% / 1.2% / 2.5% |
| Qwen3-0.6B | 58.6 | 7.2 (12%) | 7.0 (12%) | 14.2 (24%) | 14.4 (25%) | 4.7% / 4.5% / 9.2% |
| Qwen3.5-0.8B | 6.2 | 0.8 (12%) | 0.8 (12%) | 1.5 (25%) | 1.5 (25%) | 0.5% / 0.5% / 1.1% |
| Qwen3.5-2B | 6.2 | 0.8 (12%) | 0.8 (12%) | 1.5 (25%) | 1.5 (25%) | 0.2% / 0.2% / 0.5% |
| Qwen3.5-4B | 16.6 | 2.1 (12%) | 2.0 (12%) | 4.1 (25%) | 4.1 (25%) | 0.3% / 0.3% / 0.6% |
| SmolLM3-3B | 37.7 | 4.6 (12%) | 4.5 (12%) | 9.1 (24%) | 9.3 (25%) | 1.0% / 1.0% / 2.0% |
| gemma-4-E2B-it | 18.4 | 2.4 (13%) | 2.2 (12%) | 4.5 (25%) | 4.7 (26%) | 0.7% / 0.6% / 1.3% |
| gemma-4-E4B-it | 19.4 | 2.5 (13%) | 2.3 (12%) | 4.8 (25%) | 5.0 (26%) | 0.4% / 0.3% / 0.7% |

(Phi-4-mini at 4000.)

- **6-bit V passes where 4-bit V did not.** The best 6-bit V costs -0.04 to +0.17% of dKL
  against the best 4-bit V's 0.14-1.85%. With int6, batch 8 at 4K passes on every model but
  Gemma 4, which passes with V^T MXFP6 only (E2B-it at batch 8 and single-user at 8K, E4B-it at
  batch 8 by 1 SE). Single-user, int6 passes on Qwen3-0.6B from 2K (+0.08 (0.03) against -3.0%
  of a 2K token, -4.7% at 4K) and on LFM2.5-230M from 4K, V^T MXFP6 on Qwen3.5 from 2K (+0.00 to
  +0.01). Elsewhere a single user's V is too small a share of the step (6-bit V saves 0.5-1.8%
  at 8K) for its +0.12 to +0.38%.
- **Neither 6-bit V form wins everywhere**, the same split as 4-bit's: int6 with a token's scale
  is the better V on Qwen3-0.6B (+0.08 against +0.29), SmolLM3 (+0.17 / +0.21) and Phi-4-mini
  (+0.13 / +0.18); V^T MXFP6 on Qwen3.5 (+0.00-0.01 against +0.07-0.20), LFM2.5-230M (+0.06 /
  +0.14) and Gemma 4 (E2B-it -0.04 (0.11) against +5.82 (0.35), E4B-it +0.17 / +0.38); LFM2-2.6B
  ties (+0.12 / +0.13). On Qwen3.5 the KV is a small share of the step and at batch 8 int6
  passes too; Gemma 4 is the family that needs a channel-scaled V at all (its V channels differ
  far more than its tokens: E2B-it's token-scaled 4-bit V was +23.5%).
- **6-bit K fails where K matters.** At batch 8 and 4K (the case asked about; estimated gains):
  Qwen3-0.6B int6 +4.53 (0.18) / MXFP6 +2.36 (0.09) against a bar of 0.98, SmolLM3 +1.18 /
  +0.65 against 0.49, Phi-4-mini +0.78 / +1.06 against 0.57, LFM2-2.6B +0.19 / +0.52 against
  0.19, E2B-it +1.70 / +1.49 against 0.36, E4B-it +0.60 / +0.47 against 0.22. It passes only
  where K is a small share and cheap: Qwen3.5 (+0.05-0.08 at batch 8 and 4K, Qwen3.5-0.8B also
  single-user at 8K; 1.0-3.2% of a step) and LFM2.5-230M at batch 8 (+0.49 (0.03) against
  0.67). MXFP6's 32-element scales halve int6's
  loss on Qwen3 and SmolLM3, not enough. K + V^T MXFP6 passes on SmolLM3 at batch 8 only because
  V carries K: K's own share (+0.60 for 4.9%) fails.
- **The bytes**: 6-bit V saves 12% of the KV (a quarter of V), 6-bit K the same, both 24%; fp4
  V would save 25% but fails. On Qwen3-0.6B 7.2 KB a context token of 58.6 (-4.7% of a 4K token,
  -10.1% per user at batch 8 / 4K), SmolLM3 4.6 KB of 37.7, Phi-4-mini 8.2 of 66.8, Qwen3.5-4B
  2.1 of 16.6, Gemma 4 2.4-2.5 of 18-19.

**The proposal** (deferred to the next release by the r2 consolidation): **V in
int6 with a token's scale** (today's format and layout at 6 bits), chosen per model and serving
point by the formats rule, Gemma 4 excepted; **K stays int8** (N-pair for d 64), no 6-bit K
path.

- *Format: int6, a token's scale.* It is today's V format at 6 bits: the per-token scale folds
  into P (QACT CSCALE) as now, an append is one token, nothing is sealed, and P.V stays one MM a
  head. It is the better form on the models where V matters most (Qwen3-0.6B, SmolLM3,
  Phi-4-mini, whose KV is the largest share). MXFP6 V^T is more accurate on Qwen3.5, LFM2.5 and
  Gemma 4, but its 32-token scales need sealing and a second P.V a head (below), and where it
  wins the bytes are few: 6-bit V saves 0.0-1.0% of a Qwen3.5 single-user step.
- *Policy: the formats rule, per model and serving point.* On where 0.1 x the step's bytes saved
  (%) exceeds the dKL by 1 SE, the bytes from the model's shapes at the cache's length and the
  slot count (6-bit V saves a quarter of V: Delta / (W / B + KV) a user, W the weight bytes a
  step; it reproduces the batch-8 ledger), the dKL from this table, `opentpu/llm/formats.py`'s
  way for the weights. The format is fixed when the cache is allocated (no mixed formats in one
  cache), so a chat session decides at its context limit and a server at its slots. From the
  measurements:

  | model | int6 V on | off |
  |---|---|---|
  | Qwen3-0.6B | single-user from 2K; batch 4 and 8 from 544 | single-user at 544 |
  | LFM2.5-230M | single-user from 4K; batch 8 at 4K | single-user under 4K |
  | SmolLM3-3B | batch 4 and 8 at 4K | single-user |
  | Phi-4-mini, LFM2-2.6B, Qwen3.5-0.8B / 2B / 4B | batch 8 at 4K | single-user (Qwen3.5: V^T MXFP6 would pass from 2K, not proposed) |
  | Gemma 4 E2B-it, E4B-it | none (V^T MXFP6 would pass at batch 8 / 4K, E2B-it also single-user at 8K) | everywhere: int6 V costs +5.82% (E2B-it), +0.38% (E4B-it) |

  (batch 8 at 4K estimated from the B 1 ledger for all but Qwen3-0.6B; SmolLM3 measured at batch
  4.)
- *K stays int8 everywhere* (with N-pair for d 64). 6-bit K passes only on Qwen3.5 and
  LFM2.5-230M at batch 8, for 1.7-6.7% per user, and its path costs more than V's: a 96-byte K
  row (48 at d 64) breaks the MM's D-aligned streamed rows (one RSC row a chunk), so it needs
  N-pair-like row packing at 3/4 besides QST's 6-bit append. Not worth a second path; with a
  token-major K layout (rows packed in the stream) it could be revisited.

**What int6 V costs** (today's V^T layout; FPGA figures estimated, nothing built):

- *Packing*: plain bit packing along a V^T row, token t's value at bits 6t .. 6t+5 (little
  endian): 16 values in three 32-bit words, 32 in three 64-bit words, a 128-token block in 96
  bytes. Rows stay D-byte aligned (the row stride 3/4 of today's, rounded up to D).
- *QST* (`otpu_quant`): a 6-bit element mode for the append: the ROW scale at amax / 31 and the
  clamp at +-31 (constants), each element written as a 6-bit field at its bit offset, with a bit
  mask. ISA: QST's flag bit2 (free) and the token's bit offset (0, 2, 4 or 6) in a free field.
- *DRAM writes* (`otpu_native_dram`): QST's writes arrive as byte-enabled words on port SW and
  merge per 64-byte beat (the ECC read-modify-write every append already pays). A 6-bit field
  needs bit enables: the port's 4-bit byte enable becomes a 32-bit mask, each channel's gather
  buffer keeps a 512-bit mask instead of 64 bits, and the merge selects per bit (~1-1.5K LUT,
  ~1K FF). An append still writes one partial beat a row; one token in 85 straddles two beats.
- *MM* (`otpu_mxu`): a 6-bit element format, UNIT (V^T's scales stay in P): block k of row n is
  the 96 bytes at sa + n rs + 96 k, so the D-byte chunk stream carries 4 blocks in 3 chunks. A
  gearbox between the chunk FIFO and the element path (a 128-byte carry register and a 4-phase
  96-byte select) and the 6-to-8-bit sign extension into the int8 path; the issuer's and pop's
  chunk count a row becomes ceil(3 KB / 4) (~1.5-2.5K LUT, ~1.1K FF). The products, the scale
  path (UNIT: Sw 1, dw 0, m 1, W 16) and the epilogue are int8's. ISA: WF is two bits with all
  four codes used (int8, int4, fp4, NVFP4) and the MM's eight flag bits are taken, so the format
  needs a third WF bit outside the flags (e.g. w7's top bit: srs is not read under UNIT) or a new
  opcode.
- *PAIR and DUP*: PAIR takes two blocks a cycle (192 bytes) from a FIFO that pops one D-byte
  chunk a cycle, so a 6-bit P.V runs without PAIR, one block a cycle: int8 V's advances for 3/4
  of its bytes. In decode P.V is DRAM-bound (64 bytes a cycle at DDR3-1066 and 133.33 MHz
  against 96 a block), so its time follows its bytes. DUP is PAIR's companion in QACT:
  unaffected (P's QACT with CSCALE as today).
- *Also touched*: the sequencer's decode of the new fields; `isasim`, `emulated_logits` and the
  quantizer emulations (the 31-level rounding); the compiler's KV allocation and append.
- *Total*: ~3-4K LUT, ~2K FF, no DSP, no block RAM on the FPGA. On an ASIC ~30K gate
  equivalents (~0.03 mm^2 in NanGate45, estimated) and a mux a bit of energy, for 12% of the KV
  bytes: on Qwen3-0.6B at 4K, 30 MB a token, ~1.0 mJ at HBM-class 4 pJ / bit (~4.8 mJ at DDR's
  20). The logic is paid once; the bytes every token.
- *MXFP6 V^T would cost more*: its E8M0 scale spans 32 tokens of a channel, so a token cannot be
  appended alone. It needs an int8 tail of up to 31 tokens sealed into MXFP6 every 32 (a pass:
  read, per-channel amax, E2M3 rounding, a scale byte each), two P.V MMs a head (sealed blocks,
  then the tail with P's per-token CSCALE), P quantized twice, an E2M3 decode (~0.8K LUT) and a
  32-element scale layout in the MM. The extra MM and QACT a KV head and layer cost decode
  time (Qwen3-0.6B: 224 more MMs a token) that its accuracy edge does not buy back.

### 4.9 DeltaNet: multi-row DSTEP (prefill)

DSTEP streams the fp32 state of a head (dk x dv) from DRAM through the stream engine and back
for every token: 37.7 MB a token for Qwen3.5-0.8B and 2B, 100.7 MB for 4B (simulated), 589,824 SE
cycles a token for 0.8B (8 state elements a cycle, 64 bytes of state traffic a cycle). In
prefill each row repeats it (34% of a Qwen3.5-0.8B row at 4 rows, 17% of 2B's), and a tile of T
rows shares the weights and K / V but not the state: at T = 12 the state would be ~60% of a
0.8B row.

**Multi-row DSTEP**: one instruction for R consecutive rows of a head: the state read once,
the R updates applied in order to each state row while it is in the engine (S_t = g_t S_{t-1} +
beta_t (v_t - S_{t-1} k_t) k_t^T, o_t = S_t q_t: a state row's R updates need only the R rows'
k, q, v[i], beta, g, a few KB), written once. Bytes a row: 37.7 / R MB (3.1 MB at R = 12,
against the tile kernel's K / V sharing, which takes Qwen3-0.6B's re-stream from 2.20 to 1.11 MB
a row at 8 -> 16 rows: tiling's ledger). Bit-exact (the same operations in the same order).
Cycles: the engine's work stays R x 589,824 a token-row (Qwen3.5-0.8B), and on the card the
step is already bound by the engine's 8 words a cycle, not by DRAM. seq-vpu's bound (co-sim,
Qwen3.5-0.8B, 4 prompt rows, MCOLS 4, 4 layers; R 8 does not compile in 64K TMEM words, so the
card's R is 4):

| case | cycles | change |
|---|---:|---:|
| today | 1,141,254 | |
| mr8: the state read and written once a run, today's datapath (built, bit-exact: dstep-mr 5ad9b4fc) | 1,140,648 | -0.05% |
| mr16: two chained 8-word engines | 924,164 | -19.0% |
| mrfree: the step free | 816,817 | -28.4% |
| nose: no stream engine work | 798,208 | -30.1% |

So multi-row DSTEP alone saves the state's DRAM bytes (37.7 MB a row / R: energy, and the
port's time for the other streams) but no cycles on the card; the time comes with the engine's
width (chained engines, or the chunked form: the WY / chunkwise delta rule, the R-row update as
MXU matmuls, the state once a chunk).
**Owner: seq-vpu** (assigned on 2026-10-12, after its WAITC merge candidate). Its
design (branch dstep-mr, local): a multi-token STREAM (T tokens an instruction, per-token
strides in the instruction, in place; every token's inputs loaded before the first step), the
state rows circulating through the engine for T tokens (one engine, the FPGA's form) or P
chained engines (the ASIC's); compiler and bit-exact tests against T single steps done, RTL
and its bound next. The fold's SE input modes (conv taps, SiLU, L2) become flags on the same
instruction, applied as each token's raw rows load into the engine's per-token buffers (a
token's whole head row loads before its first step, so L2's statistic is ready: today's
operations in today's order), and the composite stages the tail lends the VPU today go with
the fold: the engine can be pure stream.

#### 4.9.1 The chunked delta rule on the MXU (2026-10-14: accuracy and cost, no kernel)

The chunked form (Hugging Face's `torch_chunk_gated_delta_rule`, the WY representation) takes a
chunk of C rows of a head in eight matmuls and a triangular solve, and reads and writes the
state once a chunk instead of once a row. With gc the decays' cumulative sums in the chunk and
Gamma[r, s] = exp(gc_r - gc_s):
- M1 L = (beta k . k^T) * Gamma, strictly lower; T = (I + L)^-1 by forward substitution (VPU,
  fp32, C^3 / 6 MACs);
- M2 T . (beta v), M3 T . (beta k * exp(gc)); M5 U = M2 - M3 . S (S the state at the chunk's
  start);
- M4 A = (q . k^T) * Gamma, lower with the diagonal; M6 O = (q * exp(gc)) . S, M7 + A . U;
- M8 S' = exp(gc_C) S + (k * exp(gc_C - gc))^T . U.

**Accuracy.** `tools/dn_chunk.py` (tools/formats_scan.py's Qwen3.5 emulation with its
recurrence replaced; `check`: the chunked form against the sequential one on random gates) on
MiniCPM-V's card mix (delta@0-19=int8, attention and DeltaNet nvfp4,
shared expert int8, fp4 experts, int8 head), 2000 tokens of docs/isa.md, every DeltaNet layer
chunked over the whole text (an upper bound: decode stays DSTEP). int8: today's MXU operands
(QACT rows against int8 ROW rows over d; over the chunk's tokens, one int8 scale a token folded
into the other side, as P.V), with MM's window drops; "split": each operand its int8 part plus
the int8 of its remainder, three MXU passes (hi.hi + hi.lo + lo.hi: ~16-bit operands on today's
MXU). Values under fp32's normal range are flushed to zero before a quantization, and the decay
matrix's upper triangle is zeroed before its exp, as Hugging Face's (it would overflow). KL
against the float model:

| recurrence | KL to float | dKL against int8 (SE) | against today's, paired (SE) |
|---|---:|---:|---:|
| today's: sequential, fp32 state (DSTEP) | 4.667% | +3.680 (0.141) | 0 |
| chunked, C = 64, exact | 4.667% | +3.680 | +0.0000 |
| C = 16 / 32 / 64, all eight int8 | 49.8 / 50.3 / 51.1% | +48.8 / +49.3 / +50.1 | +45.1 / +45.6 / +46.4 (2.0) |
| C = 64, M1-M4 and M7 int8 (in the chunk) | 49.4% | +48.5 (2.0) | +44.8 (1.9) |
| C = 64, M5, M6, M8 int8 (the state's) | 5.178% | +4.192 (0.147) | +0.512 (0.093) |
| C = 16, the state's int8 | 5.375% | +4.388 (0.148) | +0.708 (0.117) |
| C = 64, M1 int8 (L's k . k^T) | 4.731% | +3.745 (0.141) | +0.065 (0.068) |
| C = 64, M2 and M3 int8 (T's products) | 48.3% | +47.3 (2.0) | +43.6 (2.0) |
| C = 64, M4 and M7 int8 (the outputs') | 15.9% | +14.9 (0.79) | +11.2 (0.77) |
| C = 64, the state's split | 4.634% | +3.648 (0.140) | -0.032 (0.060) |
| C = 64, the chunk's split, the state's int8 | 5.300% | +4.313 (0.154) | +0.633 (0.102) |
| C = 64, all eight split | 4.798% | +3.811 (0.147) | +0.131 (0.071) |
| C = 16, all eight split | 4.785% | +3.798 (0.137) | +0.118 (0.066) |

- The chunked form itself is exact (to float64's rounding).
- What breaks in int8 is not the solve's input: L's k . k^T (M1) takes one int8 pass within its
  SE. It is the products with the solve's output T = (I + L)^-1 (M2, M3: T . beta v and
  T . beta k exp(gc)), which go on into U and the state, so their error adds up chunk after
  chunk: +43.6 points. The output path (M4, M7: A = q . k^T * Gamma, signed and unnormalized,
  against U) costs +11.2, without reaching the state. The state's matmuls take int8 for +0.5
  (C = 64) to +0.7 (C = 16).
- All eight split costs +0.13 at C = 64 (+0.12 at 16): the candidate, against a bar of about
  +0.5. The chunk's split with the state's int8 misses it (+0.63).

**Cost.** Issue cycles a row and head from the op sizes (d = 128; the card's MXU, MCOLS 4: a
D-block a column a cycle; VPU 8 lanes, exp2 2), against DSTEP's 2,048 (its datapath: the state's
16K words once through, 8 a cycle). "split": an operand as its int8 part plus the int8 of the
remainder, three MXU passes (~16-bit operands on today's MXU). MiniCPM-V's 488-row image prompt
(30 DeltaNet layers x 32 heads) takes 7.2 s of DSTEP, 57% of it exposed (4.1 s, below).

| C | matmuls | MXU | VPU | DRAM | x DSTEP (units overlapped) | the prompt's recurrence |
|---:|---|---:|---:|---:|---:|---:|
| 16 | all int8 | 424 | 382 | 64 | 4.8 | 1.5 s |
| 64 | all int8 | 256 | 294 | 16 | 7.0 | 1.0 s |
| 16 | the chunk's split, the state's int8 | 632 | 542 | 64 | 3.2 | 2.2 s |
| 16 | all split | 1,272 | 974 | 64 | 1.6 | 4.5 s |
| 64 | the chunk's split, the state's int8 | 512 | 454 | 16 | 4.0 | 1.8 s |
| 64 | all split (the candidate: +0.13) | 768 | 598 | 16 | 2.7 | 2.7 s |
| 16 | the chunk's on the VPU in fp32, the state's int8 | 320 | 1,022 | 64 | 2.0 | 3.6 s |
| 64 | the chunk's on the VPU in fp32, the state's int8 | 128 | 2,854 | 16 | 0.7 | 10.0 s |

**At MCOLS 8 (r2.1).** The mixer's MXU work shrinks there and DSTEP does not. Co-simulated, the
m8 variant (ypcb-00338:m8: MCOLS 8, MXU_E 8, 128K TMEM words) against m4 on docs/tiled_prefill.md
14.2's layers (MiniCPM-V's formats, tiles of 8 rows at position 256, ms a row), each also with
DSTEP's cost removed (a timing hack) for its exposed part; DSTEP itself is 0.49 ms a row and
layer (32 heads x 2,048 cycles):

| layer | m4 | m4, no DSTEP | DSTEP exposed | m8 | m8, no DSTEP | DSTEP exposed |
|---|---:|---:|---:|---:|---:|---:|
| DeltaNet, int8 (layers 0-19) | 1.027 | 0.722 | 0.305 (62%) | 1.020 | 0.722 | 0.298 (61%) |
| DeltaNet, nvfp4 (layers 20-39) | 0.975 | 0.720 | 0.255 (52%) | 0.963 | 0.509 | 0.454 (92%) |
| attention, nvfp4 | 0.710 | | | 0.488 | | |

- MCOLS 8 takes the nvfp4 DeltaNet layer's other work from 0.720 to 0.509 ms a row, and DSTEP's
  exposed part grows by about as much (0.255 -> 0.454): the layer gains 1%. Over the 40 layers
  (15 + 15 DeltaNet, 10 attention) the mixer goes from 37.1 to 34.6 ms a row, nearly all of it
  the attention layers', and DSTEP's exposed part from 8.4 ms a row (57% of its 14.7; 23% of the
  mixer) to 11.3 (76%; 33% of the mixer): 4.1 -> 5.5 s of the image prompt.
- The int8 DeltaNet layers do not move, for their tile program's sake, not the MXU's: their
  projections run in pieces of 4 rows (the image's gr / on rows, `grows`: RUN_ROWS 4) and the MoE
  prologue (the router and the shared expert) in groups of 2 rows (a need line's), so an 8-row
  tile streams the projections twice and the shared expert four times, and int8 weights go at
  port B's chunk a cycle whatever MCOLS (docs/tiled_prefill.md 14.3).
- The candidate (all split, C = 64) at MCOLS 8: the MXU's 768 cycles a row and head halve to 384
  and the VPU's 598 stay (8 lanes): VPU-bound, 3.4x DSTEP, 2.1 s of the prompt (2.7 s on m4).
  Against DSTEP's exposed 5.5 s that is ~3.4 s back (~20% of the mixer's 16.9 s), against ~1.4 s
  on m4. Half of its VPU work is the split's residuals (304 of 598 cycles), which an MM taking
  ~16-bit operands itself would remove with two of the three passes.

**ASIC architecture item: a short-K MM, and the state's traffic.** MM contracts whole D-blocks,
so a contraction over the chunk's C < D rows pads to D: M8 (Kd^T . U, d x d over C rows) takes
4,096 cycles a chunk at MCOLS 4 whatever C is, and M2, M3 and M7 pad the same way. An MM mode
that takes a short K at C / D of a block's cycles cuts the small chunks' MXU cost about 3x (C = 16,
all int8: 424 -> 116 cycles a row and head; C = 32: 304 -> 136). The data movement is the case
for the chunked form: MiniCPM-V's image prompt moves 61 GB of DeltaNet state with DSTEP (128 KB
a row and head, read and written), ~1 GB at C = 64 (0.96 GB; 3.8 GB at C = 16), about 10% of the
request's ~590 GB, where the card gains only ~1.4 s of its 90 s with the candidate (all split at
C = 64: 2.7 s of the units' issue against DSTEP's 7.2 s, 4.1 s of it exposed; ~3.4 s at MCOLS
8, above). Not built: no kernel until the arch decision.

### 4.10 ACT RAM reads gated to the MM's columns (done)

`otpu_mxu`'s act_ren was 1: every column's ACT block RAMs (the D-byte pieces and the 8-byte scale
record) read every cycle, so a decode MM (M = 1) read MCOLS rows for 1 (2 with PAIR), and nothing
stopped the reads while the MXU waited. Now `act_ren[j] = p_act && pop_e.cm[j]`, cm set when a
command is accepted (every column with replay; else j < M, PAIR j < 2M), p_act the pop head's
release: a register and one LUT before each block RAM's enable, pop itself off the path. A
column that does not read holds its last block, which only its own (undrained) products take.
Cycle- and bit-exact by construction; a simulation check stops if an output row's column does not
read at a pop. Test: `tests/test_rtl.py::test_act_reads_follow_the_mm_rows` (MCOLS 4 and 8, M 1
to 9, int8 / int4 / fp4 / NVFP4, PAIR; with the MM replay and single-slice fuzzers: 26 passed,
the simulation host). Commit 20627927.

**Per token, whole models** (simulated, ledger: physical = MCOLS rows read every advance, as the
RTL did at each pop; gated = the MM's columns; GB, each block D + 8 bytes):

| run | DRAM | ACT reads before, MCOLS 4 | after | MCOLS 8 before | after |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B decode @544 | 0.44 | 1.74 | 0.71 (-59%) | 3.48 | 0.71 (-80%) |
| SmolLM3-3B decode @544 | 1.82 | 7.19 | 3.36 (-53%) | 14.37 | 3.36 (-77%) |
| Qwen3.5-0.8B decode @544 | 0.58 | 2.15 | 0.81 (-62%) | 4.31 | 0.81 (-81%) |
| Gemma 4 E2B decode @544 | 1.45 | 5.81 | 2.53 (-56%) | 11.51 | 2.53 (-78%) |
| Qwen3-0.6B, 4 prompt rows (a row) | 0.096 | 0.61 | 0.54 (-12%) | 0.76 | 0.54 (-29%) |
| Qwen3-0.6B, batch 8 (a user) | 0.088 | 0.78 | 0.71 (-9%) | 0.92 | 0.71 (-23%) |
| Qwen3-0.6B, 16-row tile at MCOLS 8 (a row, no head) | 0.025 | | | 0.54 | 0.54 (0) |

**Measured** (co-sim, column reads including the cycles between pops, which the RTL also read):

| run | cycles | before (cycles x MCOLS) | after | at pops (the minimum) | change |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B one row, 2 layers, MCOLS 4 | 174,249 | 698,020 | 331,472 | 283,680 | -52.5% |
| Qwen3-0.6B 4 rows, 2 layers, MCOLS 4 | 374,524 | 1,499,120 | 1,209,484 | 1,134,912 | -19.3% |
| SmolLM3-3B one row at 4000, 2 layers, MCOLS 4 | 789,968 | 3,160,896 | 1,714,450 | 1,479,712 | -45.8% |

*(section 5's runs report the same counters for every workload)*

**What remains is the logical read, and it is large**: every advance reads the activation block
of every row it serves, so a row's activation reads are about twice its 4-bit weight bytes. In
decode that is 1.4-1.9x the token's DRAM bytes after the gating; in a 16-row tile, whose weight
bytes a row are a sixteenth, it is 22x the row's DRAM bytes (0.54 GB against 25 MB). The
"stationary" operand is re-read for every weight row. Section 6.1 costs it and 7.3 stages the
fix (a k-major order over tiles of weight rows with an accumulator per row in the column end:
integer sums are order-free, so it is bit-exact).

An ASIC goes further at no risk: gate the unused columns' multiplier clocks with the same cm
(their products are discarded), and gate the reads on the pop itself through a clock-gating
cell (the FPGA keeps pop off the enable for timing).

### 4.11 The stationary operand, made stationary (next generation)

Section 4.10's ledger shows what the gating cannot remove: the MXU streams a weight row's KB
blocks and reads, for every one, the activation block of each of its rows from the ACT RAM, so
an activation block is read once per weight row: N times an MM. The ACT RAM read register
already holds its block when its enable is low (4.10). So:

- **Issue order k-major over tiles of NT weight rows**: for each block k, read the activation
  block once, hold it (enable low) while the NT rows' chunks of block k pass, then the next k.
  ACT reads fall NT-fold (NT 128: decode's 0.71 GB a Qwen3-0.6B token to 5.5 MB; a 16-row tile's
  0.54 GB a row to 4.2 MB), and the multipliers' activation inputs stop toggling.
- **The column end accumulates NT rows instead of one**: an NT-entry accumulator (48-bit) per
  column instead of the DSP's P register, double-buffered so a tile drains while the next
  accumulates: 2 x 128 x 48 bits a column (~12 Kbit; ~0.2-0.3K LUT a column as LUTRAM plus a
  48-bit adder, estimated). The block scales are applied as shifts before accumulation, as
  today, so the integer sums are the same in any order: **bit-exact**.
- **The weights stored tile-major** ([tile][k][row in tile][D bytes], scale entries likewise),
  so the stream stays sequential in DRAM: an image layout change, no numerics. The drain sees
  NT x M outputs at a tile's end instead of M every KB advances: the same average rate, in
  bursts the result FIFO and E lanes already absorb (or a tile-sized output buffer).

It is the same structure as 4.8's corner-turning FIFO (k-major with per-row accumulators), so
the two share one MXU change. An energy item (section 6.1), not a time item: the reads happen
under the stream today.

### 4.12 Slices at S > 1: which side of an MM to split

At S > 1 each MM is split over the slices one of two ways. **(a) By output** (N): each slice
computes N / S of the outputs over all of K, so every slice holds the whole input: S copies of
the activation, broadcast to all. This is today's scheme (`kernels/mlp`'s and `attention`'s
`all_gather`: the residual, the attention output, W_o's output and the SwiGLU product, 7H words
a layer at Qwen3-0.6B's shapes, infra-hunt 10.3). **(b) By K**: each slice holds K / S of the
input (no copies) and computes partial sums of all N outputs, reduced over the link (PUTADD,
infra-hunt's proposal 10.4, approved as the direction). (a) moves the input, (b) moves the output
as fp32 partials, so the cheaper side is the narrower tensor.

**The per-MM rule (c)**: split a fan-out MM (N >= K: q / k / v and Qwen3.5's gate, DeltaNet's
in_proj, gate / up, the router and the experts' gate / up, the LM head) by output and a fan-in MM
(K > N: W_o, out_proj, down, the experts' down) by K, over the fan-out's local outputs. The wide
intermediates (heads, F, DeltaNet's q / k / v / z, the experts' F) never leave their slice; per
block one all-reduce of H (a PUTADD reduce-scatter, then a PUT all-gather) and the block's input
norm(x), H wide, the narrowest activation, in every slice. It is tensor parallelism's (Megatron)
pairing, and it is what the minimum of both costs gives from the shapes. **Adopted
(2026-10-12) as the S > 1 split**, with the attention and MoE rules below.

Per decode token (with the head) and per prompt row (without), S = 2 and 4 (estimated from the
models' shapes; link: what a slice receives, fp32 words on a ring; copies: activation bytes held
beyond one copy, the ACT RAM's int8 + records and TMEM's fp32; syncs: collectives a layer):

| model | S | per | (a) link | (a) copies | (b) link | (c) link | (c) copies | (c)'s link / the slice's DRAM bytes |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 2 | token | 403 KB | 1021 KB | 992 KB | 231 KB | 295 KB | 0.11% |
| Qwen3-0.6B | 2 | prompt row | 401 KB | 1016 KB | 688 KB | 229 KB | 290 KB | 3.34% |
| Qwen3-0.6B | 4 | token | 605 KB | 3064 KB | 1488 KB | 347 KB | 886 KB | 0.32% |
| Qwen3-0.6B | 4 | prompt row | 602 KB | 3048 KB | 1032 KB | 344 KB | 871 KB | 10.01% |
| Qwen3.5-4B | 2 | token | 1185 KB | 2999 KB | 2761 KB | 660 KB | 842 KB | 0.05% |
| Qwen3.5-4B | 2 | prompt row | 1180 KB | 2986 KB | 2264 KB | 655 KB | 829 KB | 0.82% |
| Qwen3.5-4B | 4 | token | 1777 KB | 8997 KB | 4141 KB | 991 KB | 2527 KB | 0.15% |
| Qwen3.5-4B | 4 | prompt row | 1769 KB | 8958 KB | 3396 KB | 983 KB | 2488 KB | 2.46% |
| Qwen3.5-35B-A3B | 2 | token | 1028 KB | 2602 KB | 2528 KB | 659 KB | 840 KB | 0.08% |
| Qwen3.5-35B-A3B | 2 | prompt row | 1024 KB | 2592 KB | 2031 KB | 655 KB | 829 KB |  |
| Qwen3.5-35B-A3B | 4 | token | 1542 KB | 7807 KB | 3792 KB | 989 KB | 2519 KB | 0.23% |
| Qwen3.5-35B-A3B | 4 | prompt row | 1536 KB | 7776 KB | 3047 KB | 983 KB | 2488 KB |  |

(collectives a layer: (a) 4 gathers, (b) 4 reduce-scatters and 4 row-scalar all-reduces, (c)
2 all-reduces. DRAM: the ledger's decode token at 544 over S; a prompt row: a 16-row pass's
weight bytes a row over S, Qwen3-0.6B's from the tile kernel's ledger, Qwen3.5-4B's from its
decode weights.)

- **(c) moves 0.56-0.64x (a)'s link bytes and 0.23-0.33x (b)'s, holds 0.28-0.32x (a)'s copies,
  and has half (a)'s collectives and a quarter of (b)'s.** (b) avoids copies but sends the fan-out MMs'
  wide outputs (q / k / v, 2F, DeltaNet's 12K, the head's vocabulary) as fp32 partials, and
  needs a row-scalar all-reduce before every QACT (its row scale) and every norm. Its copies-free
  property is worth little: (c)'s copies are norm(x) and x, one H-wide row each a block (5-13
  KB a row and block).
- **Decode and prefill take the same rule.** In decode the link is 0.05-0.3% of a slice's DRAM
  bytes a token: what counts is the collectives' latency, 2 a layer in (c). In prefill the link
  becomes a first-class resource: Qwen3-0.6B's 16-row tile at S = 4 receives 10% of its DRAM
  bytes a row over the link (3.3% at S = 2; Qwen3.5-4B 0.8-2.5%), and the share grows as tiles
  grow (the weight bytes a row fall as 1 / T, the link's do not). Prefill can split the residual
  by rows (sequence parallel: the all-reduce's two halves become a reduce-scatter and an
  all-gather of rows, the same bytes), so x is not held S times either.
- **Attention splits by KV head where n_kv >= S** (Qwen3-0.6B's 8, Qwen3.5-4B's 4): the KV and
  DeltaNet's state stay with their heads, NUMA-local. The 35B's 2 KV heads at S = 4 split by
  tokens instead (each slice 1 / S of the KV blocks; the partial m, l and acc combined over the
  link: n_q (d + 2) words a row and attention layer, ~16.5 KB) rather than holding a KV head
  twice in DRAM.
- **MoE: split each expert's F** (gate / up by output, down by K): every slice works on every
  routed expert, balanced at batch 1, and the combine is the block's all-reduce. Whole experts per
  slice (expert parallelism) leave a decode token's 8 experts unevenly spread over 4 slices; it
  is the choice for many slices or many rows, with an all-to-all (PUT to a register-named slice).
- **The TMEM a row shrinks toward 1 / S** under (c) (the wide intermediates split; norm(x), x
  and the fan-in's partial sums stay H wide: ~0.65x at S = 2 for Qwen3-0.6B, estimated), so a prompt pass
  holds more rows in the same SRAM and the weight bytes a prompt row fall with it: most of (b)'s
  capacity argument without its link bytes.
- **The TMEM caps the prompt tile, not the ACT RAM.** At MCOLS 16 the ACT RAM's BRAMs hold
  ACT_ROWS 64 as they hold 16 (dsp-pump, `docs/mx_mxu.md` 10: the bank count is the read port's
  width), and 64 rows would cut a prompt row's weight bytes 4x (Qwen3-0.6B 14.9 -> 3.7 MB a row,
  SmolLM3 ~94 -> ~23, Qwen3.5-4B ~120 -> ~30; estimated, the layers' weights of a pass over T).
  But the tile kernel's TMEM footprint stops it at 8-16 rows in 64K words (compile checks at
  MCOLS 16: Qwen3-0.6B 16 rows at 544, 12 at 4000, none of 16 at 2048; SmolLM3 8). **64-row
  tiles need ~4x the TMEM a row, or S > 1 under (c).**

## 5. Upper bounds (task 4): the fold in co-sim

**The transforms** (`tools/dataflow.py cosim ... --modes`): each VOP of a fused role becomes a
two-element ADD whose operands and destination span the original's first and last words, so
the sequencer still orders every producer and consumer it waited for (WAITC's spans are
conservative the same way, seq-vpu), while the work and the TMEM traffic go: an upper bound of
the fold, timing only (results are wrong afterwards). Modes, cumulative by class (section 3):
**A** (a and x), **AS** (+ a\*: qk-norm, RoPE, DeltaNet's conv and front), **ASB** (+ b: the
softmax numerators, the mask, DeltaNet's gated norm), **ASB,Q** (+ every QACT a one-row, one-block
QACT: its TMEM read and pass gone, the bound of quantize-on-drain through QUANT's row buffer), **C**
(every VOP: the VPU free, dependencies kept; DSTEP stays). And by stage (7.3): **T** (what
today's ISA takes with compiler changes alone: the residual as MM ACC, o / l as W_o's RSCALE, the
gain as CSCALE, the pad copies), **T,GE** (+ the quantizer's GLU and EXP + SUM modes: the GLU, the
gates, PLE, the softmax numerators, the mask). E = 4 at MCOLS 4 (the card) and MCOLS 8, E = 8 at
MCOLS 16 (E = MCOLS / 2, the sizing chosen; the drain's rate is in the co-sim). The RTL of
`mx-column` 8b33206e with the ACT gating (20627927), LiteDRAM DDR3-1066, 133.33 MHz; zero images;
2 layers unless noted.

### 5.1 Cycles

Base's cycles and the change against them:

| workload (2 layers unless noted) | MCOLS / E | base | T | T,GE | A | AS | ASB | ASB,Q | C |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B rows 1 @544 | 4 /  | 174,249 |  |  | -0.4% | -1.8% | -3.0% | -3.1% | -3.1% |
| Qwen3-0.6B rows 4 @544 | 4 /  | 374,524 |  |  | -1.7% | -5.1% | -7.0% | -7.2% | -7.2% |
| Qwen3.5-2B rows 1 @544 (layers 4) | 4 / 4 | 1,087,407 |  |  | -0.3% | -0.4% | -0.4% | -0.6% | -0.6% |
| Qwen3.5-2B rows 4 @544 (layers 4) | 4 / 4 | 2,093,091 |  |  | -1.4% | -2.6% | -2.7% | -3.2% | -3.0% |
| SmolLM3-3B rows 1 @4000 | 4 / 4 | 789,968 |  |  | -0.3% | -0.5% | -2.3% | -2.3% | -2.3% |
| SmolLM3-3B rows 4 @544 | 4 / 4 | 1,319,910 |  |  | -1.4% | -1.8% | -3.0% | -3.4% | -3.0% |
| gemma-4-E2B-it rows 4 @544 (layers 0,1,2,3,4) | 4 / 4 | 2,356,838 |  |  | -3.8% | -3.8% | -3.8% | -4.9% | -6.0% |
| Qwen3-0.6B batch 4 @544 | 4 / 4 | 1,647,306 |  |  | -0.5% | -1.1% | -1.5% | -1.8% | -1.5% |
| Qwen3.5-35B-A3B moelayer 2 @544 (layers 4) | 4 / 4 | 311,313 |  |  | -1.2% | -2.5% | -2.5% | -3.3% | -5.1% |
| gemma-4-26B-A4B-it moelayer 2 @544 (layers 0,5) | 4 / 4 | 1,059,021 |  |  | -1.1% | -1.1% | -1.1% | -1.6% | -2.9% |
| LFM2-2.6B rows 3 @544 (layers 6) | 8 / 4 | 2,271,652 | -0.4% | -1.7% | -0.8% | -2.1% | -3.4% | -3.5% | -3.4% |
| Phi-4-mini-instruct rows 3 @544 | 8 / 4 | 990,631 | -0.7% | -1.5% | -1.1% | -1.5% | -2.5% | -3.0% | -2.5% |
| Qwen3-0.6B rows 1 @544 | 8 / 4 | 173,866 | -0.0% | -1.0% | -0.3% | -1.7% | -2.8% | -2.9% | -2.8% |
| Qwen3-0.6B rows 4 @544 | 8 / 4 | 270,988 | -0.7% | -2.7% | -1.2% | -5.8% | -8.3% | -8.0% | -8.1% |
| Qwen3.5-2B rows 4 @544 (layers 4) | 8 / 4 | 1,476,449 | -0.7% | -0.9% | -1.5% | -10.2% | -11.2% | -11.7% | -11.5% |
| SmolLM3-3B rows 1 @4000 | 8 / 4 | 789,966 | -0.3% | -2.1% | -0.3% | -0.4% | -2.3% | -2.4% | -2.4% |
| SmolLM3-3B rows 4 @544 | 8 / 4 | 788,728 | -0.8% | -2.8% | -1.3% | -2.1% | -4.1% | -4.7% | -4.1% |
| gemma-4-E2B-it rows 4 @544 (layers 0,1,2,3,4) | 8 / 4 | 1,313,924 | -1.2% | -2.7% | -4.2% | -4.3% | -4.4% | -6.0% | -7.9% |
| Qwen3-0.6B batch 4 @544 | 8 / 4 | 1,540,634 | -0.1% | -0.5% | -0.3% | -0.7% | -1.1% | -1.2% | -1.1% |
| Qwen3-0.6B tile 16 @544 | 8 / 4 | 823,684 | -1.3% | -9.9% | -2.5% | -9.8% | -18.3% | -21.0% | -19.1% |
| Qwen3-0.6B tile 16 @4000 | 8 / 4 | 1,876,158 | -0.6% | -20.7% | -1.1% | -4.8% | -24.8% | -25.8% | -25.1% |
| SmolLM3-3B batch 4 @544 | 8 / 4 | 2,954,584 | -0.3% | -0.7% | -0.4% | -0.7% | -1.1% | -1.4% | -1.2% |
| SmolLM3-3B tile 12 @544 | 8 / 4 | 2,734,502 |  |  | -1.2% | -1.8% | -4.2% | -5.5% | -4.4% |
| Qwen3-0.6B rows 1 @544 | 16 / 8 | 173,930 | -0.2% | -0.8% | +0.2% | -1.6% | -2.8% | -2.6% | -2.8% |
| Qwen3-0.6B batch 4 @544 | 16 / 8 | 1,542,699 | -0.1% | -0.5% | -0.3% | -0.8% | -1.2% | -1.3% | -1.2% |
| Qwen3-0.6B tile 12 @4000 | 16 / 8 | 1,326,631 | -1.2% | -42.9% | -2.3% | -5.7% | -46.7% | -48.6% | -47.0% |
| Qwen3-0.6B tile 16 @544 | 16 / 8 | 695,359 | -2.8% | -20.3% | -7.3% | -17.0% | -31.2% | -37.2% | -31.7% |
| SmolLM3-3B tile 8 @544 | 16 / 8 | 867,372 | -0.8% | -6.7% | -1.7% | -3.0% | -8.8% | -11.5% | -9.2% |

(rows N: N prompt rows of the bucket kernel at a position, rows 1 a decode step without the head;
tile N: the tile kernel's N packed rows, `opentpu/llm/tiled.py`, Qwen3 / SmolLM3 only; batch B:
B users a step; moelayer: MoE decode on a layer prefix. What does not fit: a 16-row tile at 4000
and SmolLM3's 12-row tile exhaust TMEM at MCOLS 16; batch 8's program overflows the 32K-word IMEM
at MCOLS 8 and 16; Phi-4-mini's batches and SmolLM3's batch 8 exhaust TMEM. The stage modes T
and T,GE ran at MCOLS 8 and 16 only, after the MCOLS 4 runs.)

### 5.2 What the bounds say

- **On the card (MCOLS 4) the fold is worth little**: 0.6-3.1% of a decode step, 3-7% of a
  4-row prompt run, 1.6-3.3% of a MoE layer (co-sim bounds). Decode is DRAM-bound and MCOLS 4's
  MXU leaves the VPU the time it needs; the VPU's work overlaps the weight stream.
- **Class c is off the critical path in the dense runs**: C (every VOP fused, the QACTs as
  today) is within 0.8 point of ASB in all of them. Gemma (E2B: ASB -3.8 / -4.4%, C -6.0 /
  -7.9% at MCOLS 4 / 8; its sandwich norms) and the MoE layers (35B -2.5 vs -5.1%, 26B -1.1 vs
  -2.9%; routing and combine) are the exceptions: their c classes are on it. The quantizer's row
  path (",Q") is worth more than class c wherever QACTs are many (6 points of the MCOLS 16 tile).
- **The fold grows with the array and with the context.** Qwen3-0.6B's prompt tiles:

  | tile | MCOLS 8 (E 4) | MCOLS 16 (E 8) |
  |---|---:|---:|
  | 16 rows at 544 | -21.0% | **-37.2%** |
  | at 4000 (16 rows at MCOLS 8, 12 at 16) | -25.8% | **-48.6%** |

  Without the fold, MCOLS 8 -> 16 buys 16% at 544 (823,684 -> 695,359 cycles); with it, MCOLS 16's
  tile takes ~437K, 47% under MCOLS 8 without it. **At MCOLS 16 the VPU is the prefill bottleneck,
  and the fold is the precondition for scaling the array** (adopted by 2026-10-12).
  SmolLM3's tiles gain less (-5.5% for 12 rows at MCOLS 8, -11.5% for 8 rows at MCOLS 16): its
  wider projections (H 2048, F 11008) keep the MXU the limit longer.
- **The hybrids gain through the stream engine's input**: Qwen3.5-2B's 4-row run -11.7% at MCOLS 8
  (-3.2% at 4), nearly all of it a\* (DeltaNet's conv taps and q / k front: -10.2 points).
- **By class** (the 16-row tile at MCOLS 16): A 7 points, a\* (qk-norm, RoPE) 10, b (softmax
  numerators, mask) 14, the quantizer's row path (Q) 6; class c 0.5. In decode, a\* is most of
  it for Qwen3 (qk-norm and RoPE run per head on small rows).
- **By stage**: T (compiler only) 0-1.3% everywhere (-2.8% for the MCOLS 16 tile); T,GE (+ GLU and
  EXP + SUM) a third to a half of the fold in the dense models' decode and 4-row runs (most of
  SmolLM3's at 4000, little of the hybrids', whose fold is the stream engine's input). In the
  prompt tiles T,GE is half the fold or more, the more so at long context (Qwen3-0.6B: -9.9 of
  -21.0% at MCOLS 8 / 544, -20.3 of -37.2% at MCOLS 16 / 544, -20.7 of -25.8% at MCOLS 8 / 4000,
  -42.9 of -48.6% at MCOLS 16 / 4000): the softmax's EXP + SUM and the GLU are the two modes that
  pay first.
- **Decode stays DRAM-bound at MCOLS 8 and 16**: one row is 174,249 / 173,866 / 173,930 cycles at
  MCOLS 4 / 8 / 16; the fold's -2.3 to -3.1% (Qwen3-0.6B, SmolLM3) is the same at all three.
  Batched steps (batch 4) gain 1.2-1.8%. No bound is slower than its base by more than 0.2% (A
  at MCOLS 16 in decode: +0.2%, the schedule's noise).

**QUANT's rate, checked** (estimated). The bounds make a fused op free; the real quantizer must
keep up. A 16-row tile at 544 has ~393K padded scores (16 rows x 16 q heads x 768 x 2 layers):
at MCOLS / 2 = 8 exp2 a cycle (4.3's composite unit) ~49K cycles, plus the SiLU's ~98K at 8 a
cycle ~12K: ~14% of the fused run's ~437K cycles, spread over the layer. At 4000 the 12-row tile
at MCOLS 16 has ~1.57M padded scores: ~197K cycles of exp2 at 8 a cycle, 29% of the fused run's
~682K. Per KV head and 256-token block QUANT must take ~6 scores a cycle against the attention
MMs' ~1,024 advances for 24 score rows x 256 (two replay groups of q.K^T, two of P.V): 8
composite lanes keep pace at ~75% load. **MCOLS / 2 lanes is the floor, not a margin**; 16-row
tiles at long context, or score lanes that speed q.K^T's drain, need MCOLS of them.

### 5.3 TMEM replays on the single copy

The single pumped TMEM copy (tmem-pump's r2pgRs: NSB 2, sides s, the pick) against
today's six copies (i0), under the same transforms (co-sim on tmem-pump's tree b5d6bcef, the
pre-MX MXU; Qwen3-0.6B, 2 layers; replays: TMEMSTAT's lost reads by requester):

| workload | mode | cycles, six copies (i0) | one pumped copy (r2pgRs) | the copy's cost | replays (MXU / Q / VA / VB / Q2) |
|---|---|---:|---:|---:|---|
| Qwen3-0.6B, 4 rows, MCOLS 4 | base | 372,383 | 376,555 | +1.12% | 12,632 (3084 / 3115 / 1484 / 409 / 4540) |
| Qwen3-0.6B, 4 rows, MCOLS 4 | A | 367,645 | 371,856 | +1.15% | 12,438 (3025 / 3048 / 1441 / 349 / 4575) |
| Qwen3-0.6B, 4 rows, MCOLS 4 | AS | 354,748 | 360,393 | +1.59% | 10,039 (3045 / 1594 / 466 / 350 / 4584) |
| Qwen3-0.6B, 4 rows, MCOLS 4 | ASB | 350,081 | 351,035 | +0.27% | 2,966 (471 / 152 / 18 / 3 / 2322) |
| Qwen3-0.6B, 4 rows, MCOLS 4 | ASB,Q | 344,742 | 344,737 | -0.00% | 200 (98 / 86 / 14 / 2 / 0) |
| Qwen3-0.6B, 4 rows, MCOLS 4 | C | 349,339 | 351,035 | +0.49% | 4,056 (409 / 725 / 265 / 228 / 2429) |
| Qwen3-0.6B, 8 rows, MCOLS 8 | base | 502,925 | 511,272 | +1.66% | 29,631 (6552 / 8944 / 3582 / 1146 / 9407) |
| Qwen3-0.6B, 8 rows, MCOLS 8 | A | 486,433 | 495,021 | +1.77% | 29,176 (6249 / 9091 / 3584 / 839 / 9413) |
| Qwen3-0.6B, 8 rows, MCOLS 8 | AS | 461,836 | 470,028 | +1.77% | 20,510 (6264 / 3126 / 871 / 833 / 9416) |
| Qwen3-0.6B, 8 rows, MCOLS 8 | ASB | 452,188 | 453,287 | +0.24% | 6,252 (1108 / 250 / 19 / 18 / 4857) |
| Qwen3-0.6B, 8 rows, MCOLS 8 | ASB,Q | 444,438 | 443,767 | -0.15% | 237 (115 / 106 / 12 / 4 / 0) |
| Qwen3-0.6B, 8 rows, MCOLS 8 | C | 452,186 | 454,117 | +0.43% | 8,323 (839 / 1442 / 583 / 423 / 5036) |

- **The fold removes the single copy's cost.** Without it one copy costs 1.1-1.8% of a prompt run
  (requesters meeting on a bank and replaying); with ASB the cost is 0.24-0.27%, with ASB,Q it
  is 0 (-0.00%, -0.15%), and the replays fall 98-99%.
- **The softmax numerators are most of it** (AS -> ASB: VA / VB and most of Q's and the MXU's
  replays go), and **QACT's CSCALE operand (Q2) goes with quantize-on-drain** (",Q").
- C (the VPU free, QACTs as compiled) keeps +0.4-0.5%: the quantizer's own reads stay.
- tmem-pump records it in `docs/tmem_nocopy.md` 6.6.

### 5.4 ACT reads in the same runs (co-sim counters)

| workload | MCOLS | cycles | ACT reads before (cyc x MCOLS) | after | at pops | change |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B rows 1 @544 | 4 | 174,249 | 698,020 | 331,472 | 283,680 | -52.5% |
| Qwen3-0.6B rows 4 @544 | 4 | 374,524 | 1,499,120 | 1,209,484 | 1,134,912 | -19.3% |
| Qwen3.5-2B rows 1 @544 | 4 | 1,087,407 | 4,350,652 | 2,150,898 | 1,806,352 | -50.6% |
| Qwen3.5-2B rows 4 @544 | 4 | 2,093,091 | 8,373,388 | 8,055,360 | 7,225,504 | -3.8% |
| SmolLM3-3B rows 1 @4000 | 4 | 789,968 | 3,160,896 | 1,714,450 | 1,479,712 | -45.8% |
| SmolLM3-3B rows 4 @544 | 4 | 1,319,910 | 5,280,664 | 5,129,952 | 5,034,304 | -2.9% |
| gemma-4-E2B-it rows 4 @544 | 4 | 2,356,838 | 9,428,376 | 8,849,160 | 6,390,720 | -6.1% |
| Qwen3-0.6B batch 4 @544 | 4 | 1,647,306 | 6,590,248 | 6,287,832 | 5,996,672 | -4.6% |
| Qwen3.5-35B-A3B moelayer 2 @544 | 4 | 311,313 | 1,246,276 | 1,127,500 | 974,880 | -9.5% |
| gemma-4-26B-A4B-it moelayer 2 @544 | 4 | 1,059,021 | 4,237,108 | 2,891,814 | 2,066,432 | -31.8% |
| LFM2-2.6B rows 3 @544 | 8 | 2,271,652 | 18,175,264 | 12,951,088 | 11,520,384 | -28.7% |
| Phi-4-mini-instruct rows 3 @544 | 8 | 990,631 | 7,927,096 | 5,549,907 | 4,889,376 | -30.0% |
| Qwen3-0.6B rows 1 @544 | 8 | 173,866 | 1,392,976 | 330,694 | 283,680 | -76.3% |
| Qwen3-0.6B rows 4 @544 | 8 | 270,988 | 2,169,952 | 1,401,546 | 1,134,912 | -35.4% |
| Qwen3.5-2B rows 4 @544 | 8 | 1,476,449 | 11,813,640 | 9,102,056 | 7,225,504 | -23.0% |
| SmolLM3-3B rows 1 @4000 | 8 | 789,966 | 6,321,776 | 1,712,072 | 1,479,712 | -72.9% |
| SmolLM3-3B rows 4 @544 | 8 | 788,728 | 6,311,872 | 5,798,320 | 5,034,304 | -8.1% |
| gemma-4-E2B-it rows 4 @544 | 8 | 1,313,924 | 10,513,440 | 9,318,784 | 6,390,720 | -11.4% |
| Qwen3-0.6B batch 4 @544 | 8 | 1,540,634 | 12,327,120 | 6,448,168 | 5,996,672 | -47.7% |
| Qwen3-0.6B tile 16 @544 | 8 | 823,684 | 6,591,520 | 5,534,120 | 4,543,488 | -16.0% |
| Qwen3-0.6B tile 16 @4000 | 8 | 1,876,158 | 15,011,312 | 13,422,976 | 8,082,432 | -10.6% |
| SmolLM3-3B batch 4 @544 | 8 | 2,954,584 | 23,638,720 | 14,403,004 | 13,250,688 | -39.1% |
| SmolLM3-3B tile 12 @544 | 8 | 2,734,502 | 21,878,064 | 20,598,944 | 15,104,640 | -5.8% |
| Qwen3-0.6B rows 1 @544 | 16 | 173,930 | 2,786,976 | 330,822 | 283,680 | -88.1% |
| Qwen3-0.6B batch 4 @544 | 16 | 1,542,699 | 24,687,280 | 6,459,388 | 5,996,672 | -73.8% |
| Qwen3-0.6B tile 12 @4000 | 16 | 1,326,631 | 21,230,192 | 9,950,288 | 6,061,568 | -53.1% |
| Qwen3-0.6B tile 16 @544 | 16 | 695,359 | 11,129,840 | 5,884,688 | 4,544,512 | -47.1% |
| SmolLM3-3B tile 8 @544 | 16 | 867,372 | 13,882,048 | 11,734,624 | 10,069,504 | -15.5% |

The gating's share follows the M of the MMs: decode -52 to -76%, 4-row runs -3 to -35%, the
16-row tile -16% at MCOLS 8 and -47% at MCOLS 16 (its attention MMs use G of 16 columns).

## 6. Cost model (task 5)

### 6.1 Energy per byte by level (published estimates)

| level | energy | source (estimates, not measured here) |
|---|---|---|
| DRAM, DDR3 / DDR4 (off-chip, with I/O) | ~15-25 pJ/bit | Horowitz, ISSCC 2014 (1.3-2.6 nJ a 64-bit access at 45 nm: 20-40 pJ/bit); DDR4 interface studies ~15-20 |
| DRAM, LPDDR5 | ~4-5 pJ/bit | vendor / survey figures |
| DRAM, HBM2 / HBM3 | ~3-4 pJ/bit | O'Connor et al., MICRO 2017 (HBM2 3.9 pJ/bit) |
| SRAM read, 8 KB / 32 KB / 1 MB array, 45 nm | 0.16 / 0.31 / 1.6 pJ/bit | Horowitz 2014 (10 / 20 / 100 pJ a 64-bit read) |
| SRAM at 7-5 nm | ~0.05-0.3 pJ/bit | scaled, order of magnitude |
| on-chip wire | ~0.1 pJ/bit/mm | order of magnitude, 16-7 nm |
| int8 multiply-add / fp32 multiply-add, 45 nm | 0.23 pJ / 4.6 pJ | Horowitz 2014 (0.2 + 0.03; 3.7 + 0.9) |

**A Qwen3-0.6B decode token at 544** (the ledger's bytes x the table; estimated, 45 nm-class
SRAM, two DRAM classes):

| item | bytes / ops a token | DDR-class (20 pJ/bit) | HBM-class (4 pJ/bit) |
|---|---:|---:|---:|
| DRAM | 436 MB | 70 mJ | 14 mJ |
| ACT RAM reads, physical (ledger: MCOLS rows an advance) | 1.74 GB | 2.2 mJ | 2.2 mJ |
| ... gated to the MM's columns (M 1, PAIR: 2 of 4) | ~0.87 GB | 1.1 mJ | 1.1 mJ |
| the weight FIFO (written and read once) | 0.87 GB | ~2.1 mJ | ~2.1 mJ |
| TMEM (r + w, one copy; today's six copies multiply the writes) | 26 MB | ~0.1 mJ (x6 writes: ~0.4) | same |
| int8 products (MCOLS columns computing, 1-2 useful) | 1.64 G | 0.38 mJ (0.14 useful) | same |
| VPU fp32 work | ~27 M ops | ~0.12 mJ | same |

- **DRAM is ~93% of this token's data-movement energy with DDR, ~75% with HBM.** The ACT RAM's
  physical reads are the next item (3% / 12%): the gating halves them at MCOLS 4 and cuts 3/4 at
  MCOLS 8 in decode, worth more than the whole VPU fold on an HBM-class ASIC.
- **The VPU fold saves ~0.1-0.4 mJ a token** (its TMEM traffic and fp32 work): under 1% of the
  DDR-class token, ~2% of the HBM-class one. Its value is time and ports, not energy.
- **The KV format outweighs everything on chip**: fp4 K + V at 4K would save ~120 MB a
  Qwen3-0.6B token, ~19 mJ (DDR) / ~4 mJ (HBM), but 4-bit K fails the accuracy rule (4.8); the
  6-bit V that passes saves ~30 MB, ~4.8 mJ / ~1.0 mJ, still more than the whole fold.
- **In prefill the activation re-read rivals DRAM.** A 16-row tile of Qwen3-0.6B moves 25 MB of
  DRAM a row (4 mJ at 20 pJ/bit, 0.8 mJ at 4 pJ/bit) and reads 0.54 GB of ACT RAM
  (0.7 mJ): on an HBM-class ASIC the re-read is ~45% of the row's data-movement energy, and
  4.11's k-major tiles remove it (NT 128: 4.2 MB, ~0.005 mJ).

### 6.2 SRAM bytes and ports

- **No new SRAM.** The fold moves the composite unit to QUANT and adds QUANT's mode registers;
  the corner-turning FIFO (4.8 option 1) rebanks the existing FIFO. No copies. (At MCOLS 16 the
  ACT RAM's 272 RAMB36, against 68 at MCOLS 4, are one copy whose bank count is the read port's
  width: dsp-pump, `docs/mx_mxu.md` 10.)
- **TMEM requesters**: VA / VB traffic falls 85-99% (4.5), so the VPU's two read ports can merge
  into one narrow port; Q2 carries the GLU's second operand; the MXU's (ACC) port carries the
  epilogue's operands. On the single pumped copy (r2pgRs) that removes the copy's replays and
  its time (5.3: +1.1-1.7% against six copies today, ~0 with the fold, co-sim).
- **ACT RAM**: reads -52.5% (co-sim, decode row at MCOLS 4), the minimum (pops x the MM's
  columns) 14% below that.

### 6.3 FPGA area (estimated)

| item | LUT | DSP | note |
|---|---:|---:|---|
| out: otpu_se_comp's composite path | -10.9K | (its stages are the VPU's slot 0 and the tail's U / Q) | seq-vpu's OOC breakdown of otpu_vpu (30.8K LUT) |
| in: QUANT's composite unit, MCOLS/2 lanes x 3 stages, own multiply-add stages | +5.5K (MCOLS 8) / +10.9K (MCOLS 16) | +24 / +48 | se_comp's structure at half / full width; an fmadd stage ~0.6K LUT, 2 DSP (mx-rtl-prod) |
| in: QUANT's modes (EXP's mask and sum, GLU's operand, NORM's square-sum tree, ROPE's half-row pairing) | ~+2K | ~+4 | |
| in: the stream engine's input modes (conv taps, L2 statistic) | ~+1K | | the composites are already the engine's |
| the drain's lanes | 0 | 0 | 4.2: nothing added |
| the VPU's elementwise width (seq-vpu's call) | 0 to -2K | | the stream engine's datapath (61% of otpu_vpu) stays for DSTEP |
| **net** | **~-2.5K (MCOLS 8) / ~+3K (MCOLS 16)** | **+28 / +52** | about area-neutral: the composite work moves rather than grows |

At MCOLS 16 a 16-word quantizer read (4.3) adds the row path's width; the E 8 lanes and any score
lanes are the MX column's own cost (E 4 -> 8 at 16 rows: +3.6K LUT, +6K FF, +24 DSP, measured
OOC), with or without the fold. The ACT gating costs ~MCOLS LUTs (an AND a
column; cm is MCOLS flip-flops a queue entry).

## 7. Recommendation (task 6)

**Fold the VPU's elementwise work into the MXU's datapath at its two ends, staged with the
array.** It does not compromise decode or prefill: every fused op is today's fp32 work in today's
order (two one-rounding moves, a\*, aside), the drain's lanes are unchanged, and no bound is
slower than base by more than 0.2%. Its gain is small on the card and grows with the columns and
the context (co-sim bounds, section 5): decode 0.6-3.1% at MCOLS 4, 8 and 16; prompt runs 3-12%;
Qwen3-0.6B's prompt tile -21% (-26% at 4000) at MCOLS 8 and **-37% (-49% at 4000) at MCOLS 16**,
where without it the second 8 columns buy 16%. Beside the time: TMEM's words -53 to -75% (4.5),
the single pumped TMEM copy's replays and cost gone (5.3), about the same area (6.3), under 2% of a
token's data-movement energy (6.1). The lead adopted it on 2026-10-12 as the precondition for
scaling MCOLS. Release r2 (the consolidation) carries the ACT gating (20627927) and the
tools that measured all this (`tools/dataflow.py`, the KV formats in `tools/formats_scan.py`);
the fold and every proposal below are next-release items.

### 7.1 What to fold, what stays

- **Fold into the quantizer's input (QACT / QST)**: the softmax numerators (EXP + SUM), the GLU's
  act(g) x u and the gates, the pre-MM RMSNorms and qk-norm (NORM), RoPE, the mask. These are the
  composite passes and most of the VPU's TMEM traffic; the quantizer reads each operand once on
  its way into the ACT RAM or the KV cache, and se_comp's composite unit moves there (MCOLS / 2
  lanes x 3 stages: the floor at long context, 5.2).
- **Keep on the drain**: the residual (ACC, today). The E lanes gain nothing; E = MCOLS / 2.
- **Fold into the stream engine's input**: DeltaNet's conv taps, L2 norms, SiLU, applied as each
  token's rows load into multi-row DSTEP's buffers (seq-vpu's design; interface agreed, 4.9).
- **Stays**: flash's row scalars, Gemma's sandwich norms, DeltaNet's gate scalars, sampling,
  routing on a narrow VPU (1-2 composite lanes, seq-vpu); DSTEP on the stream engine.

### 7.2 Proposals and decisions (for the next release unless marked)

1. **The fold's ISA** (4.6): QACT / QST input modes EXP (+ SUM), GLU, NORM, ROPE, a mask operand;
   DSTEP input modes; MM unchanged. Bit-exact except qk-norm's gain before RoPE and NORM's 1 / rms
   in the row scale (one rounding each: a dKL check with the first build of the modes). Adopted
   as the precondition for MCOLS 16 (2026-10-12); the ISA comes with its RTL.
2. **ACT reads gated to the MM's columns**: done (20627927), in r2; accepted for the MCOLS 8
   tree.
3. **KV** (4.8, dKL against MX int8 over 2000 tokens on 10 card models): **no 4-bit K** (+0.3
   to +14.8% on every model; agreed); 4-bit V only where KV dominates (Qwen3-0.6B's batched
   decode at 4K). **Proposed, deferred to the next release: V in int6 with a token's scale**
   (today's V^T format and layout at 6 bits: +0.07-0.20% of dKL outside Gemma 4, where it costs
   +0.38-5.82%), on per model and serving point by the formats rule: at batch 8 / 4K on every
   model but Gemma 4, single-user on Qwen3-0.6B from 2K and LFM2.5-230M from 4K; -12% of the KV
   bytes (Qwen3-0.6B: 7.2 KB a context token, -4.7% of a 4K token, -10.1% per user at batch 8 /
   4K), ~3-4K LUT, no DSP or block RAM (estimated). **K stays int8**:
   6-bit K fails at batch 8 / 4K on the dense models (Qwen3-0.6B +4.5 / +2.4% against a 1.0%
   bar) and passes only where K is a small share (Qwen3.5, LFM2.5-230M at batch 8).
4. **N-pair for LFM2's d 64 K** (int8, numerics-neutral): -6.4% of an LFM2.5-230M token at 4K,
   -10.6% at 8K, -1.1% of LFM2-2.6B's at 4K (MCOLS 8). A free item: PAIR's column split with the
   two halves drained as two outputs.
5. **Multi-row DSTEP** (4.9): assigned to seq-vpu (2026-10-12), after its WAITC merge
   candidate; bit-exact; the state's DRAM bytes / R, and the time only with a wider engine
   (seq-vpu's bound: mr8 -0.05%, two chained engines -19.0% of a Qwen3.5-0.8B prompt run).
6. **S > 1: fan-out MMs split by output, fan-in MMs by K** (4.12), one all-reduce of H a block
   over PUT / PUTADD; attention by KV head (by tokens where n_kv < S), MoE experts by F. Adopted
   (2026-10-12).
7. **k-major weight tiles** (4.11): recorded as a next-generation energy item.
8. **The chunked delta rule and a short-K MM** (4.9.1, 2026-10-14): recorded as an ASIC
   architecture item. The DeltaNet state once a chunk instead of once a row: MiniCPM-V's image
   prompt's 61 GB of state traffic to ~1 GB at C = 64, for +0.13 points of dKL with every
   matmul's operands split (three int8 passes); an MM mode for a contraction shorter than D cuts
   the small chunks' MXU cost about 3x. The card gains ~1.4 s of a 90 s request (~3.4 s at
   MCOLS 8, where the MXU work around DSTEP shrinks and DSTEP does not); no kernel.

### 7.3 Staging

In r2: the ACT gating (20627927) and the tools. The rest, for the next release on:

| item | next release: the MCOLS 8 build | needed for MCOLS 16 | later generations |
|---|---|---|---|
| ACT reads gated to the MM's columns | **in r2** (20627927) | | clock-gate idle columns, reads on pop (ASIC) |
| T: compiler only (residual ACC, o / l RSCALE, gain CSCALE, no pad copies) | yes, no RTL (0-1.3%; -2.8% of the MCOLS 16 tile) | | |
| GE: QUANT's GLU and EXP + SUM modes, se_comp's composite unit at QUANT (MCOLS / 2 lanes x 3 stages) | proposed: -1.0% of decode, -2.7% of 4 rows, -9.9% / -20.7% of a 16-row tile at 544 / 4000 | yes (-20.3% / -42.9% of the tile) | |
| NORM, ROPE, the mask operand (a\*: dKL with the build); the SE's input modes | | yes | |
| "+Q": the drain into QUANT's row buffer (no TMEM round trip before a QACT) | | yes (6 of the tile's 37 points; the single TMEM copy's last replays) | |
| QUANT's read width 16 words, MCOLS composite lanes for 16-row tiles at long context | | yes | |
| E = MCOLS / 2 lanes | E 4 (as today) | E 8 (the MX column's: +3.6K LUT, +24 DSP, measured OOC) | |
| score lanes (no ACC / ASCALE) | | optional: a full 16-row q.K^T at d 128 | |
| the VPU narrowed (1-2 composite lanes) | | yes, with the fold (seq-vpu) | |
| N-pair (LFM2's int8 K) | yes (dsp-pump / mx-rtl-prod) | | |
| 64-row prompt tiles (ACT_ROWS 64 fits MCOLS 16's ACT BRAMs; weights a row / 4) | | ~4x TMEM a row, or S > 1 under 4.12's rule | |
| S > 1 (PUT / PUTADD, 4.12's split rule) | | | the multi-slice target (adopted direction) |
| int6 V (per model and serving point; QST's 6-bit append, the MM's 6-bit stream, bit-masked beat merges) | if approved: independent of MCOLS (~3-4K LUT) | | 4-bit V where KV dominates; 6-bit K only with a token-major K layout |
| multi-row DSTEP; chained SE engines for its time | | | seq-vpu (assigned) |
| V token-major, corner-turning FIFO | | | yes (second-tier: 1.05-1.37% of decode; whole-beat appends, also for 6-bit V) |
| k-major weight tiles, NT-row accumulators | | | yes (ASIC energy) |

### 7.4 Risks

- **QUANT becomes the hot unit**: softmax, GLU and norms all pass through it; at MCOLS / 2
  composite lanes it runs at ~75% during a long-context tile's attention (5.2's check), so it is
  the floor, and 16-row tiles at long context want MCOLS lanes, or attention waits on QUANT
  instead of the VPU.
- **The bounds are bounds**: a fused op is free in them; the real quantizer modes read up to
  three operands (Q, Q2, Q3) and run the composite passes.
- **Numerics**: two one-rounding changes (7.2 item 1); the KV formats per model (dKL).
- **Compiler**: pattern rules in `kernels/lib`, `attention`, `mlp`, `tiled`; the reference models
  (`isasim`, `emulated_logits`) must model every mode exactly for the bit-exact tests.
- **Gemma and the MoE layers keep VPU work on their critical path** (sandwich norms, routing):
  1-2 points of their bound; the narrow VPU must keep those rates.
