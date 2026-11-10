# Batched decode for Qwen3.6-35B-A3B (sequence slots)

The 35B's image can hold several sequences, each in a slot of its own (KV caches, DeltaNet
states, convolution windows), and one decode step can advance every busy slot: the weights are
streamed once a step and each row adds its own work. This page is the layout, the step's program,
what bounds it, and what two BCU1525s measured (docs/x2.md is the machine; docs/serve.md the
server that uses it). Software only: no RTL is particular to it. Labels: **card** is measured on
the cards, **co-sim** the board's RTL model, **ISA** unit busy cycles on the instruction-set
simulator's ledger, **estimate** arithmetic.

## Summary

- **The image.** `Image(batch=N)`: N slots of an equal context. On two cards four slots of
  131,072 tokens fit; on one card two.
- **The step.** One program per set of slots and sampling kinds, each row's position a run-time
  word, so nothing compiles while requests decode. Every row's logits, greedy pick and sampled
  pick are the single-sequence decode's bit for bit.
- **On two cards at 150 MHz (card).** One sequence decodes at 65.3 tok/s. With four requests
  the device steps at about 103 tok/s in all (103 / 4 is about 26 a sequence): its rate while
  stepping, on one 64-token leg at 3.88 rows a step, with no prefill or host time. Before the
  DeltaNet mixer was pipelined over the rows that leg read about 88: the pipeline takes
  13.5-14.7 % of a step's cycles on every like-for-like pair, every reply unchanged. By the
  wall clock, on one schedule of four 256-token requests, the summed decode rate went from 75.6
  to 81.3 tok/s.
- **What bounds it.** A row's own work (its routed experts at the DRAM's rate, its DeltaNet
  states on the DMA, its softmax on the VPU) does not hide under the shared weight stream as the
  unit bounds assume: the ISA bound at four rows is 138 tok/s, DRAM alone 160.
- **One card** binds earlier: on the DeltaNet state engine from two rows, then on the MXU. Two
  slots are the useful size there.

## 1. Bounds

`tools/batch35_bound.py` splits a decode row's work, from one-row layer runs on the ISA ledger,
into:

- **shared bytes**, read once a step: the dense weights, the shared expert and the router
  (21.6 MB a DeltaNet layer and 17.8 MB an attention layer on one card; 5.4 and 4.7 MB a slice on
  two), and the head (514 MB on one card);
- **routed bytes**: one expert for each expert in the rows' union, 1.69 MB on one card, 0.42 MB a
  slice on two. Uniform routing's expectation of the union is 15.8 / 30.5 / 57.4 / 102 experts at
  2 / 4 / 8 / 16 rows; real routing is skewed, so the real union is smaller;
- **per-row bytes**: the DeltaNet state read and written (2 MiB a layer on one card), the KV
  caches (1,044 B a position an attention layer on one card, 522 a slice on two), activations;
- **per-row unit cycles**: MXU, VPU, the state engine (DSTEP), the quantizer. The MXU does not
  batch on grouped weights: an MM costs one advance per activation row, so B rows cost B times
  one row;
- **collectives** on two cards: each one's latency once a step, its words per row.

A layer's bound is its busiest unit, with DRAM at 88 %; a step's is the sum over the 30 DeltaNet
and 10 attention layers plus the head. These are unit bounds: no stalls, every unit overlapped.
At one row on one card the bound is 4.363 Mcycles against the card's 4.32 (an earlier one-card
image; the chat's image in docs/x2.md reads 4.51).

Tok/s a sequence / in all, at 150 MHz (ISA bound):

| target | position | B = 1 | B = 2 | B = 4 | B = 8 | B = 16 | binds at B >= 8 |
|---|---|---|---|---|---|---|---|
| one card (`bcu1525-wg:bhw16`) | 1,024 | 34.4 | 24.5 / 49.1 | 14.1 / 56.3 | 7.3 / 58.1 | 3.6 / 58.1 | DSTEP |
| one card | 8,192 | 32.7 | 21.6 / 43.1 | 11.7 / 47.0 | 5.9 / 47.5 | 3.0 / 47.5 | MXU |
| one card | 32,768 | 22.9 | 13.8 / 27.6 | 7.3 / 29.1 | 3.7 / 29.3 | 1.8 / 29.3 | MXU |
| two cards (`bcu1525-x2:st16`) | 1,024 | 67.7 | 51.5 / 102.9 | 35.2 / 141.0 | 22.3 / 178.0 | 13.1 / 209.4 | DRAM |
| two cards | 8,192 | 63.0 | 46.2 / 92.4 | 30.5 / 121.9 | 18.6 / 148.7 | 10.6 / 170.0 | DRAM |
| two cards | 32,768 | 50.8 | 34.2 / 68.4 | 20.8 / 83.3 | 11.9 / 95.1 | 6.5 / 103.3 | DRAM |

- **One card.** Two rows give each sequence twice what taking turns gives it (24.5 against about
  12 tok/s) and 1.4x the total. Past four rows the state engine binds: 65,536 cycles a row a
  DeltaNet layer (the 1 MiB state at 8 words a cycle), 1.97 Mcycles a token a row, a ceiling of
  76 tok/s. At long contexts attention's per-row MXU and VPU take over.
- **Two cards** are DRAM-bound at every size; from four rows the union's routed bytes dominate
  the DeltaNet layers.

The step's program itself, run on the ISA ledger at position 1,024 (`--isa`), meets the bound:

| B | one card Mcycles | tok/s a sequence / all | two cards Mcycles | tok/s a sequence / all |
|---|---|---|---|---|
| 1 | 4.364 | 34.4 | 2.215 | 67.7 |
| 2 | 6.131 | 24.5 / 48.9 | 2.922 | 51.3 / 102.7 |
| 4 | 10.952 | 13.7 / 54.8 | 4.337 | 34.6 / 138.3 |
| 8 | | | 7.166 | 20.9 / 167.4 |

## 2. Slots and memory

Everything a decode writes lies in the image's low area, 3,840 MiB a slice; the weights, the head
and the expert pool lie above it and are shared. A slice's low area:

| a slice | one card | two cards |
|---|---|---|
| other than KV and state (I/O, prefill rows, lookup tables) | 592 MiB | 289 MiB |
| DeltaNet state and windows, a slot | 32.9 MiB | 8.2 MiB |
| KV, a slot a token (10 attention layers) | 10.48 KB | 5.24 KB |
| sampler words, a sampled slot | 2.84 MiB | 2.84 MiB |

So the slots of a context C:

| C | one card | two cards |
|---|---|---|
| 262,144 | 1 | 2 |
| 131,072 | 2 | 5 |
| 65,536 | 4 | 10 |
| 32,768 | 9 | 20 |
| 16,384 | 16 | 39 |

The servers open N slots at 262,144 / N tokens each unless `--cap` says otherwise. The slots'
count and context are part of the image's key: an image is built for them.

## 3. The step

`opentpu/llm/batch35.py`, `qwen35_batch_step`. Per row every value is the single-sequence
decode's operations in the same order.

1. **Inputs.** Each row's token is embedded on the card from the image's lookup tables; its
   rotary rows come at its own position.
2. **DeltaNet layer.** The projections for all rows in one weight pass; each row's convolution
   over its slot's window and its DSTEPs on its slot's state; out_proj over the rows in one
   pass (`_deltanet_slots`; pipelined, section 5).
3. **Attention layer.** Each row over its own KV cache (`qwen3._attention_rows`).
4. **MoE.** The router and the shared expert once over the rows, then each row's routed experts
   (`moe.moe_ffn_tp_rows` on more than one slice; on one slice `moe.moe_ffn_rows`, the union
   with each expert on every row of its group). Reading the router and shared expert once took
   the two-card step from 4 / 9 / 15 % short of the bound at 2 / 4 / 8 rows to 0.2 / 1.9 / 6 %.
5. **Head.** One weight pass over the rows, each chunk handed to every row's greedy or sampled
   sink. On more than one slice a greedy row gathers the slices' maxima and a sampled row their
   top k, selected again, so every slice picks the same id.

**One program per (slots, kinds).** A program's rows are named by their slots and their kinds
(greedy, sampled, sampled with a penalty). Each row's position is five run-time words: the
position, its attention groups, its V^T tile and column, and a flat offset. Attention runs its
blocks in a loop of the groups word, each block masked by its row, so a block wholly past the
position leaves the online softmax as it was and the result is the exact blocks' bit for bit.
`BatchWorker35.warm_steps` compiles every slot set's programs before the server listens (15 at
most at two slots with three kinds; 80 at four slots for greedy and the default sampling), so a
request joining or leaving waits for no compile.

Keying a program by its rows' positions instead does not scale: at two slots of 131,072 tokens,
262,144 programs even with the position inside a block at run time.

**Four rows.** Two things did not fit at first on `bcu1525-x2:st16`:

- *Address registers.* Each row held two registers stepped by the layer loop. The V^T append now
  takes its row as a run-time operand, leaving one a row; five rows run out by one, and eight
  need a transient form of the V scale's address.
- *IMEM.* The step unrolled does not fit st16's 4,096 instructions. A unit's DeltaNet layers run
  in a loop instead (the prefill's nesting, docs/tiled_prefill.md), where they do not fit unrolled:

| four rows, greedy / sampled with a penalty | unrolled | nested | IMEM |
|---|---|---|---|
| `bcu1525-x2:st16` | 4,611 / 5,253 | 2,466 / 3,108 | 4,096 |
| `bcu1525-x2:wmx` (ICACHE) | 3,888 / 4,530 | | 16,384 |

**Sampling.** A sampled row's picks at four slices equal one slice's sampler on the same logits.
One difference stays, the single-sequence sampler's too: exact fp32 ties among the top k are
ordered by slice rather than by block (on random logits with six of the top 30 copied elsewhere,
344 of 19,200 picks differ; with no ties, none).

## 4. On two cards (card)

Two BCU1525s, `bcu1525-x2:st16`, 150 MHz, DDR4-2400; four slots of 131,072 tokens; fp4 weights
with an int8 head; requests of 256 tokens.

**Tokens.** Four prompts alone, then two, three and four at once, greedy and sampled (T 0.7,
top-k 20, top-p 0.8, a seed each), then four three-turn chats at once (one opens with a
1.6k-token paste): every reply equals its reply alone, 38 checks of 38, about 2,000 greedy
tokens among them.

**The rate, before the pipeline.** One sequence runs the card's own decode loop: 65.3 tok/s,
2.30 Mcycles a token. The batched steps' cycles fit a + b x rows with a = 1.119 and b = 1.401
Mcycles (three greedy groups, within 0.06 %). With every row busy:

| B | Mcycles a step | tok/s a slot | tok/s in all | DRAM share | ISA bound: Mcycles a step (tok/s in all) |
|---|---|---|---|---|---|
| 1, the decode loop (measured) | 2.30 | 65.3 | 65.3 | 85 % | 2.215 |
| 1, a step | 2.52 | 59.5 | 59.5 | 77 % | |
| 2 | 3.92 | 38.3 | 76.5 | 65 % | 2.922 (102.7) |
| 3 | 5.32 | 28.2 | 84.6 | 60 % | |
| 4 | 6.72 | 22.3 | 89.2 | 56 % | 4.337 (138.3) |

Only the decode loop is measured with every row busy; the other rows are the fit. The DRAM share
is the step's bytes (uniform routing's union: an upper bound) over 1,024 B a cycle.

A row costs 1.40 Mcycles where the bound gives it about 0.5 of DRAM: its state engine and MXU
work run after the shared stream, not under it.

## 5. The pipelined mixer

**Where a row's cycles go (co-sim, position 1,024; the model's step is within 1.3 % of the
card's at four rows and 3.7 % at one):**

| layer | B | cycles | mixer | MoE |
|---|---|---|---|---|
| DeltaNet | 1 | 53,648 | 32,753 | 20,564 |
| DeltaNet | 4 | 166,876 | 89,748 | 76,026 |
| attention | 1 | 43,457 | 22,487 | 20,639 |
| attention | 4 | 121,811 | 47,193 | 73,516 |

- The attention mixer at four rows is near the VPU's bound (busy 43.7K of 47.2K: each row's
  softmax over its own cache).
- The MoE is at the DRAM's rate: a row's 8 experts, 3.36 MB a slice, in 15.3K cycles, about 83 %
  of a slice's 256 B a cycle. The routing before them is 13.0K, serial on the VPU.
- **The DeltaNet mixer was DMA-bound and unpipelined.** Of 89.7K cycles the DMA was busy 66.6K
  (32 DSTEPs 44.8K; windows and taps 15.2K in 100 loads and stores; the gates' round trip
  through DRAM 3.5K) and idle 23.1K waiting on a pair's projections; the MXU was active 19.6K.
  A pair ran its projections, then its heads, then its DSTEPs, in program order on single
  buffers.

`_deltanet_slots_ahead` gives the rows the single-sequence decode's schedule: the MXU projects a
pair ahead into buffers of its own while the DMA runs this pair's DSTEPs and the VPU prepares
the next; a (pair, slot) window is one load and one store (16 each, not 48); every load goes
before the first DSTEP and every store after the last, because a load among the DSTEPs lengthens
the DMA's in-order chain one for one; the gates stay in TMEM; a row's two heads go in one DSTEP.
It runs where a slice's pairs unroll and its buffers fit half of TMEM (the 35B at four rows:
59.5K of 131K words).

| DeltaNet layer at four rows (co-sim; the mixer ends at the residual's all-gather, another split than the table above's) | before | pipelined |
|---|---|---|
| layer | 166,876 | 137,617 |
| mixer | 92,100 | 62,799 |
| MoE (after the mixer's last record) | 74,747 | 74,778 |
| DSTEPs | 32, chain 72,172 (busy 44,788) | 16 double DSTEPs, chain 39,677, no gaps |
| DMA busy (load / store / DSTEP) | 64.7K (12.0 / 7.8 / 44.8) | 49.9K (6.2 / 4.1 / 39.7) |
| MXU active | 20.7K | 24.8K |
| VPU | 33.7K | 32.7K |

30 layers at -29.3K is -0.88 Mcycles a step: the co-sim projected the card's four-row step at
about 5.83 Mcycles, about 103 tok/s.

**On the cards.** An A/B of the two images (only the programs differ), legs in the order
A, B, B, A, each with a server of its own; every reply of every leg equal word for word. A
group's step count follows how the prompts land between steps, so only legs of an equal count
are compared:

| group | steps (rows a step) | before, Mcycles a step | pipelined | change |
|---|---|---|---|---|
| two requests | 257 (1.74) | 3.551 | 3.069, 3.068 | -13.6 % |
| three | 257 (2.02) | 3.951 | 3.417 | -13.5 % |
| three | 194 (2.68) | 4.416 | 3.791 | -14.2 % |
| four | 257 (2.83) | 5.082 | 4.370, 4.370 | -14.0 % |
| four, sampled | 239 (3.04) | 5.324 | 4.564 | -14.3 % |
| four, sampled | 257 (2.83) | 5.124 | 4.400 | -14.1 % |
| four chats | 170 | 6.546 | 5.584 | -14.7 % |
| four chats | 174 | 6.420 | 5.481 | -14.6 % |

Where a count repeats, the cycles repeat to about 0.01 %. The warm-up legs (64 tokens, every
request ending together, one leg an image) are the nearest to every row busy:

| group | rows a step | before: Mcycles, tok/s | pipelined: Mcycles, tok/s | change |
|---|---|---|---|---|
| two | 1.94 | 3.853, 75.5 | 3.303, 88.0 | -14.3 % |
| three | 2.91 | 5.185, 84.1 | 4.460, 97.8 | -14.0 % |
| four | 3.88 | 6.619, 87.9 | 5.633, 103.2 | -14.9 % |
| four, sampled | 3.88 | 6.681, 87.0 | 5.694, 102.1 | -14.8 % |

tok/s is rows a step x 150 MHz / the step's cycles: the device while stepping, no prefill or
host time. The measured result is the ratio, -13.5 to -14.7 % over every like-for-like pair. A
fit of a measured leg extrapolated to four rows moves 7-9 % with the legs it is given (89 to 97
tok/s for one image), so those are not quoted.

What did not move: decode alone (65.1-65.2 tok/s on every leg; it runs the decode loop, not the
steps). Four chats at once are bound by their prefills (647 tokens in 50-54 s on every leg). On
one schedule of four greedy requests the device time fell from 8.71 to 7.49 s and the summed
decode rate rose from 75.6 to 81.3 tok/s.

## 6. Bugs the cards found

- **Sampled rows at three rows.** The sampler keeps one maximum per block of 64 logits, written
  at the chunk's first logit / 64, which is right only for a chunk starting on a block boundary.
  The head's chunk is sized to TMEM by the rows: 8,192 at one and two rows, 4,096 at four, and
  at three 5,456, 85 blocks and 16. From the second chunk on each maximum was 16, 32, ... logits
  off its block, the gather missed some of the top 20, and the row sampled from another
  distribution. Greedy rows take their own argmax and were exact. Tiny test vocabularies fit
  one chunk and hid it. `head_rows_chunk` now keeps whole blocks (5,440 at three rows), the
  sampler refuses a chunk off a boundary at compile time, and `tests/test_batch_steps.py` has
  the case (a 336-logit chunk, top-k 3).
- **A halted decode loop's uniforms.** The card's decode loop draws a run's uniforms ahead; when
  another request arrived and halted it, the unused ones were lost, so a seeded request that went
  on in steps used later uniforms than it did alone. The generator is now restored to one
  uniform a token picked.
- **The batch area on a mapped image.** The engine writes the batch area's first contents into
  the image's arrays; a raw image mapped read-only refused. The maps are copy-on-write and the
  area's ranges are added to the runs written.

## 7. Limits and no-gos

- **The windows and taps a layer ahead: no-go (co-sim).** Loading the next DeltaNet layer's
  windows beside this layer's MoE, as the single-sequence step does, gains 2,361 cycles a layer
  even when the loads are free (the mixer's head is bound by the activation quantize, pair 0's
  MM and its prepare, not the loads), and every place tried costs that or more: beside the first
  row's experts -604 a layer in all (the MoE is DRAM-bound, so the loads cost their bytes); once
  the shared MMs are issued +2,464 (they collide with the residual's all-gather); held behind
  the all-gather +1,470 (they delay the norm's VPU writes). Over a step, about even.
- **After the pipeline** a DeltaNet layer at four rows is 137.6K cycles (co-sim): a mixer of
  62.8K, 39.7K of it the DSTEP chain, and 74.8K of MoE. They use different units but do not overlap: the MoE needs the mixer's output.
  Overlapping one row's experts with another's DSTEPs would need out_proj by row groups.
- **Floors at four rows** (uniform routing): DRAM alone 3.75 Mcycles a step (160 tok/s), 88 % of
  DRAM 4.26 (141), the ISA bound 4.34 (138). The card's pipelined step is about 5.63 at 3.88 rows a step (one leg).
- **One card** past four rows is capped by the state engine's 8 words a cycle; only RTL lifts
  that.
- **Rows past four** on two cards need the V scale's address in a transient form (section 3).
- **Start-up.** A server computes its step programs' wait entries at start unless the program
  cache can store them (it keeps 4 GiB of its disk free, `OTPU_PROG_CACHE_FREE_GB`, and skips saves below that): 80
  programs in about 97 s with the entries stored, about 260 s computed.
- **Not done:** the real routing's union at 4-16 rows from router logs (it lowers the routed
  bytes); a paged KV pool in place of N equal slots, so long and short sequences share the
  budget; MTP with slots.

## 8. In the server

docs/serve.md "In sequence slots" is the worker (`BatchWorker35`, `opentpu/serve/batching35.py`):
a request takes the slot whose context it continues, else the store's longest match restored into
a free slot, else position 0; prompts go in by layer runs at the slot, up to 512 tokens an
iteration, between steps; one sequence alone runs the card's decode loop. A run the two-card
link fails resets the link, drops the slots' contexts and resumes each request from the store
and its output so far, once a request (docs/x2.md section 3).

## 9. Tests

- `tests/test_batch35.py`: logits at two rows on one and two slices and at three with top-8
  groups, greedy and sampled picks on the device, prompts into slots, the step over slots across
  block and group ends, four rows on both two-card variants (nested where it must), the 35B's
  program sizes, the pipelined mixer against the unpipelined; all bit for bit against
  single-sequence engines.
- `tests/test_batch_steps.py`: the head's chunks and sampler blocks.
- `tests/test_serve_batching35.py`: requests in slots, next turns from a slot and from the
  store, sampled requests alone and at once, a step the link fails, a halted decode loop's
  uniforms, nothing compiled after `warm_steps`.
- `tools/batch35_bound.py [--target T] [--B ...] [--pos ...]` prints section 1's bound; `--isa`
  runs the step's program on the ledger and `--sim` on the board's model.
