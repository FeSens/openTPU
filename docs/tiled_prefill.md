# Tiled prefill: more rows per weight pass, and the same machinery for MoE and many users

The question (the user, 2026-10-11): prefill, multi-user and MoE are now as important as decode.
Today a prefill run holds its R rows in TMEM, R_max 3-4 a bucket (docs/prefill.md 7), so each
weight chunk read from DRAM serves 3-4 tokens. Would a run of 32-256 rows, every weight chunk
streamed once for all of them, pay; what do the compiler and the ISA need for it; and how does
the same machinery serve MoE prefill and batched decode?

Measured here means an RTL co-simulation (tools/perf_tiled.py: the board's configuration and
memory path, LiteDRAM's controllers at DDR3-1066 with the core at 133.33 MHz, or a memory at the
core port's full rate; zero images, the timing does not depend on the data) or an ISA-simulator
run; estimated means arithmetic on measured parts, named as such.

## 0. Short answer

(Section 10 has the tile in the engine for every card model family, dense and MoE, and its
measurements, section 11 the tile under the MX numerics, where the dense models' prompt runs
take it by default; this section is the prototype's.)

- **Rows a weight pass are not what prefill lacks.** A pass needs about MCOLS rows (4-bit) or
  2 MCOLS (int8) to reach the MXU's multiply rate (section 2), and the card's prompt runs are
  already 96% MXU-busy at 4 rows. Today's kernel at 8, 16 and 32 rows with MM replay is within
  -1.2% / +1.3% a row of 4 rows at MCOLS 4 (4-bit), while its weight traffic falls 2.8x
  (section 4.1). 32-256-row tiles buy DRAM traffic, not cycles.
- **What pays is the restructured run** (opentpu/llm/tiled.py, bit for bit today's results):
  attention packed MCOLS / G rows to a K / V stream, the next KV head's projections streaming
  while a head attends, the MLP in pipelined pieces, and nothing in TMEM that grows with the
  model's widths. Measured (RTL co-simulation, 2 layers from position 256, cycles a row):
  - Qwen3-0.6B on the card (MCOLS 4): 83.6K today -> 79.5K at T = 4 (-4.9%, today's bitstream:
    no replay) -> 76.4K at T = 16 (-8.6%, ACT_ROWS 16); int8 89.2K -> 76.5K at T = 8 (-14%);
    at position 1024, 110.8K -> 92.4K (-17%);
  - Qwen3-0.6B at MCOLS 8: 57.7K today (4 rows), 53.8K (8 rows) -> 44.4K at T = 16 (-23% /
    -17%); position 1024: 82.0K -> 58.4K (-29%). With the tile MCOLS 8's prefill is 1.88x the
    card's today, against 1.45x with today's kernel;
  - a Qwen3-4B shape: -1.1% at MCOLS 4 (its MLP is 69% of a row and at its floor; its G = 4
    heads already fill the columns), -10.7% at MCOLS 8; Qwen3.5-0.8B is 61-73% DeltaNet, which
    no tile touches (about -4% at MCOLS 8, estimated, not ported);
  - a memory at the port's full rate (the BCU1525's DDR4 at 1600 or 2400, the F2's HBM)
    changes these by 0-4%: prefill is MXU-bound on every board.
- **It fits today's TMEM**: T = 16-24 on Qwen3-0.6B and 8-12 on a 4B shape in 64K words, with
  no DRAM spill; a bigger TMEM adds nothing to the tile (77.0K at 256K words against 76.4K).
  Past ~1K tokens the unrolled packed attention needs the looped attention (IMEM).
- **MoE**: expert-major prefill is already this structure, and the tiled mixer runs belong on
  its path (layer-major runs keep R = 2 for their expert union). Tiling them is worth an
  estimated -2.6 to -3.1 s of gemma-4-26B's 16.25 s pooled prompt on the card (its int8 dense
  weights stream at 2 rows a pass today; -2.2 to -2.6 s at T = 4 on today's bitstream), little
  on Qwen3.5-35B (fp4, DeltaNet-bound). Expert passes: at MCOLS 4 a pair is the floor
  (measured), at MCOLS 8 four rows cost what two do. The dormant expert-major path's own
  predicted -1.7 / -1.9 s is the link side; its card A/B (pfB, staged) comes first and needs
  card time, which the maintainer schedules.
- **Many users**: one pass of 2 MCOLS users against two runs of MCOLS: -1% (fp4) to -6% (int8).
  Users' attention does not pack; the aggregate's lever is the MXU's rate (MCOLS 8: -36% a row
  for 8 users).
- **Recommendation.** The tile is the prefill half of a faster MXU: build it with the MCOLS 8 /
  pumped-MXU work (+12% to +30% prefill throughput on top) and for int8 layers once
  ACT_ROWS >= 8 (it holds 8-12 rows of a 4B-class layout in 64K words, today's kernel 3-4).
  On today's card alone it is small (Qwen3-0.6B -5%, more at long context; 4B-class fp4 ~0),
  so it should not displace other card work; its card-side win now is gemma-4-26B's mixer
  runs, after the pfB A/B.
- **Found on the way**: an RTL co-simulation with a TMEM over 64K words serialized every
  instruction touching the upper words (otpu_seq's 16-bit hazard ranges; 83.6K -> 112.7K a
  row); the fix is on this branch and on tmem-pump's (the same change, 3.5).

## 1. Where a prefill row's cycles go today

Measured (tools/perf_tiled.py --kernel rows: compile_rows, today's qwen3_rows; Qwen3-0.6B, its
first 2 layers, 4 rows at positions 256-259, fp4 with no logits; the card's configuration:
ACT_ROWS = MCOLS, TMEM 64K words; the card's memory path at DDR3-1066, core 133.33 MHz; a phase
is the cycles from the last instruction end before it to the last end within it, perf_qwen's
split; "attn-proj" is the Q / K / V / O projections and the queries' norms and RoPE, "attn-core"
the flash attention, "mlp" the norm, gate / up, SiLU and down):

| MCOLS | cycles a row (2 layers) | attn-core | attn-proj | mlp | MXU busy | the linear layers' MXU floor |
|---|---|---|---|---|---|---|
| 4 | 83.6K | 14.6K (17%) | 30.7K (37%) | 38.1K (46%) | 80.5K (96%) | 61.4K |
| 8 (virtual, ACT_ROWS 8) | 57.7K | 14.6K (25%) | 21.9K (38%) | 21.0K (36%) | 54.6K (95%) | 30.7K |

- **The MXU is busy 96% of a row.** The MLP runs at 1.03x its weights' floor (38.1K against
  36.9K); the attention block's projections at 1.25x theirs (30.7K against 24.6K: the per-row
  query norms and RoPE between the MMs). What the MXU does past the linear floor (19.1K) is the
  attention's q.K^T and P.V MMs: each (row, KV head) pair is a flash-attention entry of its own,
  M = G = 2 query rows on 4 columns, half the MXU idle.
- **Attention grows with blocks, slowly.** 3.7K a row and layer at position 100 (one block),
  7.3K at 256 and 267 (two): about 0.46K a (row, KV head, block). At position 1024 (five
  blocks) it is 41.1K of 110.8K a row (37%).
- **A run's fixed cost is nil**: 1 layer 41.8K a row, 2 layers 83.6K (the layers are a
  hardware loop).
- **The card agrees within 6%.** int8, 4 rows: the co-simulation gives 81.8K a row at position
  100 and 89.2K at 256 and 267, 1.145M and 1.249M cycles a token over 28 layers. The card's
  prompt runs (docs/prefill.md 7, fmvf 2026-10-02) took 288.0M cycles for 236 tokens at 0-235
  in 59 runs of 4 (1.214M a token, less the last run's logits) and 21.4M for 13 tokens at 267
  in three runs of 4 and one of 1 with logits (~1.33M a token in the 4-row runs, taking the
  1-row run as an int8 decode step's ~5.4M). Prompt runs cost ~6% more than compile_rows'
  programs (run-time positions: RunWords, the mask tiles' LDs, V^T appended a row at a time).
- The whole model at that rate: 1.17M cycles a token (fp4), 114 tokens/s at 133.33 MHz.

## 2. How many rows a weight pass needs

A pass of T rows over one 128-byte weight chunk costs max(stream, MXU) cycles (docs/isa.md MM,
docs/dsp_pumping.md 2 on dsp-pump's branch):
- the stream: 1 / eta, eta the chunks a cycle the memory delivers to the port (at most one: the
  port is 128 bytes a cycle). The card's DDR3-1066 through LiteDRAM: 0.82-0.85 while decoding
  (docs/board.md); the BCU1525's DDR4 (two channels a slice: 25.6 GB/s at 1600, 38.4 at 2400)
  and the F2's HBM deliver more than one slice's port takes (12.8 GB/s at 100 MHz, 17.1 at
  133.33), so eta = 1 there;
- the MXU: int8, ceil(T / MCOLS) cycles (replayed groups); 4-bit, two blocks a chunk: 1 cycle
  with PAIR (T <= MCOLS / 2), else 2 ceil(T / MCOLS).

Cycles a chunk and row, max(stream, MXU) / T (`tools/tile_model.py knee`; estimated from the ISA
rates, the co-simulations of section 4 measure them):

| weights, eta | MCOLS | T = 1 | 2 | 4 | 8 | 16 | 32 | the knee (fewest rows at the floor) | the floor |
|---|---|---|---|---|---|---|---|---|---|
| 4-bit, 0.85 (YPCB) | 4 | 1.176 | 0.588 | **0.500** | 0.500 | 0.500 | 0.500 | 4 | 0.500 |
| | 8 | 1.176 | 0.588 | 0.294 | **0.250** | 0.250 | 0.250 | 8 | 0.250 |
| | 16 | 1.176 | 0.588 | 0.294 | 0.147 | **0.125** | 0.125 | 16 | 0.125 |
| 4-bit, 1.0 (port-bound) | 4 | 1.000 | **0.500** | 0.500 | 0.500 | 0.500 | 0.500 | 2 | 0.500 |
| | 8 | 1.000 | 0.500 | **0.250** | 0.250 | 0.250 | 0.250 | 4 | 0.250 |
| | 16 | 1.000 | 0.500 | 0.250 | **0.125** | 0.125 | 0.125 | 8 | 0.125 |
| int8, 0.85 (YPCB) | 4 | 1.176 | 0.588 | 0.294 | **0.250** | 0.250 | 0.250 | 8 | 0.250 |
| | 8 | 1.176 | 0.588 | 0.294 | 0.147 | **0.125** | 0.125 | 16 | 0.125 |
| | 16 | 1.176 | 0.588 | 0.294 | 0.147 | 0.074 | **0.062** | 32 | 0.062 |
| int8, 1.0 (port-bound) | 4 | 1.000 | 0.500 | **0.250** | 0.250 | 0.250 | 0.250 | 4 | 0.250 |
| | 8 | 1.000 | 0.500 | 0.250 | **0.125** | 0.125 | 0.125 | 8 | 0.125 |
| | 16 | 1.000 | 0.500 | 0.250 | 0.125 | **0.062** | 0.062 | 16 | 0.062 |

- **The linear layers need about MCOLS rows a pass (4-bit) or 2 MCOLS (int8), not 32-256.**
  Past the knee the MXU's multiply rate is the bound and more rows change nothing but the DRAM
  traffic. The knee scales with the MXU's rate per chunk of stream: MCOLS 8, or a 2x-pumped
  MCOLS 4 (docs/dsp_pumping.md), doubles it; MCOLS 16 doubles it again.
- **What today's R = 4 leaves** (at the card's eta): 4-bit at MCOLS 4, nothing; int8 at MCOLS 4,
  15% of the linear layers; 4-bit at MCOLS 8, 15%; int8 at MCOLS 8, 58% (0.294 -> 0.125: four
  rows use half of eight columns). Section 4.1 measures it on whole rows: -6%, -7% and -36%
  from 4 to 8 rows.
- **A wider port** (2 chunks a cycle, which DDR4-2400 and HBM could feed) halves the stream term;
  the knee in rows falls, the floor stays. It speeds decode and below-knee runs, not a pass at
  the floor.
- **What the knee leaves out** is everything that is not a weight stream: attention (its K / V
  streams once per row and KV head today, at G of MCOLS columns), the vector work per row, and
  the dependency chains between them. Those are what the tile restructures.

For a Qwen3-0.6B layer (61.4K chunks of 4-bit weights, 122.9K int8) the floor is 30.7K cycles a
row at MCOLS 4 and 15.4K at MCOLS 8; a Qwen3-4B-shaped layer (394K / 788K chunks) 197K and 99K
(`tools/tile_model.py layers`).

## 3. Design

### 3.1 What "weight-stationary over token tiles" is on this MXU

The MXU streams the weights (one 128-byte chunk a cycle at most) against a stationary operand
of activation rows in the ACT RAM. So a weight chunk is used by every row the ACT RAM holds:
with ACT_ROWS > MCOLS the MXU keeps each streamed chunk in its FIFO and replays it for
ceil(M / MCOLS) groups of MCOLS rows (docs/isa.md, MM: one DRAM pass serves all M rows). A tile
of T <= ACT_ROWS rows is therefore one MM per weight slice, each chunk fetched once and held for
T / MCOLS cycles. The alternative orientation (weights in the ACT RAM, activations streamed)
does not exist on this MXU: the ACT RAM is int8 only (no 4-bit weights), it is written only by
QACT from fp32 TMEM rows, and the scales would multiply in the other order (not the ISA's bits).

What limits T today is not the MXU but where the rows live:
- **TMEM.** Today's kernel (qwen3_rows) holds every row's activations at once: 6.7-7.6K words a
  row on Qwen3-0.6B (compiled: 30.6K words at 4 rows, 55.8K at 8, 107.9K at 16, 215.6K at 32),
  so a 64K-word TMEM takes 8 rows, and 3-4 on the larger layouts.
- **ACT RAM rows.** The card's bitstream has ACT_ROWS = MCOLS = 4. MM replay is in the RTL
  (otpu_actram's groups, otpu_mxu's replay; `make bit ACT_ROWS=8` was built and measured on
  2026-09-27 at MCOLS 2, docs/benchmarks.md) but not in production.
- **IMEM.** Attention is unrolled per row, KV head and block, so a run's program grows with
  its rows and context (docs/prefill.md 10).

### 3.2 The tiled run

`opentpu/llm/tiled.py` (qwen3_tile, compile_tile). For a tile of T rows at positions p0 ..
p0 + T - 1, every layer (the layers' hardware loop as today):

1. **Attention input.** RMSNorm and quantization of all T rows in one QACT (RSCALE / CSCALE,
   the instruction today's peephole makes, emitted directly so no [T, H] product is traced).
2. **Per KV head j** (the next head's projections stream while this head attends):
   - K_j, V_j = MM(xs, W_k[j]), MM(xs, W_v[j]): one weight pass for T rows (W split by rows:
     each output element is the same MM sum, docs/isa.md);
   - K norm and RoPE per row, all T rows' K / V appended (QST) before any row attends;
   - Q_j = MM(xs, W_q[j]) [T, G d], normed and rotated into [T G, d] (row (r, g) = r G + g);
   - the attention in packed tiles of P = MCOLS / G rows: one MM of P G query rows streams the
     head's K (and P.V its V^T) once for P rows, where today each row streams them alone. The
     blocks every row of the packed tile sees whole run in the attention's hardware loop as
     today; the last block(s) take a mask per row group (RowMasks: the score buffer filled with
     -0, then -inf past each row's token, then q.K^T accumulated into it with the row maxima,
     MM ACC + RMAX). A row group that sees none of a block (a packed tile across a block
     boundary) gets all -inf: m stays, p = +0, alpha = 1, acc and l pass unchanged;
   - the head's outputs acc / l quantized into its columns of W_o's operand (ol.stationary +
     ol.quantize_into: per row and block, so the same bytes and scales as quantizing [T, n_q d]
     at once).
3. **Output projection and residual** by parts of the output columns: y_p = MM(o, W_o[p]),
   x[:, p] += y_p. TMEM holds [T, H / parts] of y, parts chosen so that is at most 1/8 of TMEM.
4. **MLP.** xs as in 1; gate and up in pieces of `sub` columns (software-pipelined: the next
   piece streams while the VPU computes silu(g) u), each piece quantized into its chunk's
   operand, which stays in the ACT RAM (F / D blocks of T rows: 24 on the 0.6B, 76 on a 4B);
   then the down projection by output parts, each part summing the chunks in swiglu_down's
   order (dot, then dot(acc=)), so each element is today's sum.

Bit for bit: every row's arithmetic is today's (section 7). What TMEM holds no longer grows
with H, n_q d or F, only with T times a KV head's width, a gate / up piece and an output part.
At T <= MCOLS the tile needs no replay: T = 4 runs on the card's bitstream as it is.

### 3.3 Activations: in TMEM, or through DRAM

Compiled for the card's 64K-word TMEM and 4K-instruction IMEM (the first 2 layers; fp4, int8
head, position 256): Qwen3-0.6B fits T = 16 (2,874 instructions) and T = 24 (3,994), not 32 (the
[32, 1024] residual is half of TMEM); a Qwen3-4B-shaped layout (H 2560, F 9728, 32 / 8 heads)
fits T = 12 (3,956 instructions; today's kernel fits 3-4 rows there, and none at 4 rows past
position ~1000 without fit_chunk's shrinking). So on the dense models up to 4B the tile stays in
TMEM at the T that matters (section 2), with no spill.

Past that, the residual moves to DRAM. A design that needs no new instruction:
- x lives in a DRAM scratch in D-column blocks, [H / D][T][D] fp32 (each block contiguous);
- the norm's sum of squares is accumulated over the blocks in the 64 partials of the ISA's
  isum_64 (partial q gets columns q, q + 64, ... in order: two MUL + ADD passes of [T, 64] per
  block) and folded at the end as RSSQ folds: bit for bit RSSQ's sum (docs/isa.md, Sums);
- the QACT then runs per D-column block (rows 0 .. T - 1, KB = 1, RSCALE / CSCALE): per-block
  quantization, so the same operand;
- the residual adds load and store a block column at a time.
Spill traffic: about 6 x 4H bytes a row and layer (two norms' reads, two residual
read-modify-writes): 24 KB on the 0.6B against 7.9 MB of fp4 weights a layer, i.e. 0.19K
chunks a row against 61.4K / T. At T = 64 that is 1.15K chunks of port traffic a row and layer
in 30.7K MXU cycles (4% of the port): the spill rides in the headroom the MXU-bound phases
leave. Its cost is instructions and latency, not bandwidth. Not built: section 4 shows no T
past 16 that would need it.

### 3.4 What the compiler and the ISA need

| need | today | for tiles |
|---|---|---|
| MM over more rows than MCOLS | replay in the ISA, ISA simulator and RTL (M <= ACT_ROWS <= 255; a replayed row stays in the 1024-chunk FIFO until its last group) | T = MCOLS needs none. T = 16: a bitstream with ACT_ROWS 16: the ACT RAM's RAMs are a 256-bit word per column and 32-byte piece, GROUPS x BLOCKS deep; at MCOLS 4, 4 groups x 128 blocks = 512 words fill the 64 BRAM36 that 128 words use today (estimated from the RAM shapes, not synthesized). ACT_ROWS 32: 128 BRAM36 (+64) |
| MM over more rows than ACT_ROWS | no | not needed: T <= ACT_ROWS (16) is past the knee (section 2) |
| an operand built in parts | no | `ol.stationary(rows, cols)` + `ol.quantize_into(st, x, col)` (prototype) |
| an operand that does not pin its TMEM source | no | `ol.quantize(x, free=True)` (prototype) |
| norms without [T, H] products | the peephole after tracing the products | `ol.quantize(x, rscale=, cscale=)`, `ol.sumsq(x)` (prototype; the same instructions) |
| attention rows of several tokens in one MM | no: RMAX / ASCALE need M <= MCOLS | per-row-group masks (RowMasks, TileSeq; prototype): P = MCOLS / G rows a K stream |
| attention of a whole tile in one MM | no | RMAX and ASCALE over replayed groups (M row maxima and M rescale factors in the epilogue: an RTL change). The MXU's cycles stay those of packed tiles (replay runs MCOLS rows a cycle too); it would cut MMs, their fills and the K / V streams: at most the ~9K a row (2 layers) the tile's attention takes over its MXU work at MCOLS 4 (estimated) |
| address registers | 8 arguments, up to 15 registers | the tile at compile-time positions takes 1 (the layer loop's); at a run-time position its rows share one tpos (RunPos.offset), so the prompt run's 6 coefficients serve T rows |
| IMEM | 4K instructions | T = 16 at MCOLS 4: 2,874 instructions at position 256, 4,282 at 1000, 7,424 at 2000; MCOLS 8 (P = 4): 1,946 / 2,650 / 4,224. Past ~1K tokens the packed tiles need the looped attention of docs/prefill.md 10 (a loop over a head's packed tiles: affine q tile, output rows and mask rows) |
| DMA | LD / ST / QST | T rows' K / V appends (QST of [T, d]; V^T byte-strided as today), the I/O rows; with the spill, block-column LD / ST |

### 3.5 Against a bigger TMEM

A bigger TMEM alone lets today's kernel run more rows: 108K words for 16 rows of Qwen3-0.6B,
216K for 32. Measured (section 4.1), that buys at most 1-2% a row: its linear layers are at the
MXU's rate from MCOLS rows (4-bit) or 2 MCOLS (int8, which 64K words already fit), and its
attention stays per row. The restructured run fits today's 64K words at T = 16-24, and a
TMEM of 256K words gives it nothing (T = 16: 77.0K a row against 76.4K at MCOLS 4, 44.6K
against 44.4K at MCOLS 8; larger MLP pieces pipeline less). A bigger TMEM is worth something
to the tile only past T = 24 (no spill), which section 2 says no MXU here needs below MCOLS 16.

**A co-simulation trap.** Every TMEM above 64K words was co-simulated with otpu_seq tracking
TMEM ranges in 16 bits (TAW): an instruction touching a word past 65,535 conflicted with
everything, as a barrier does. Today's kernel at 4 rows took 112.7K a row with a 256K-word
TMEM against 83.6K with 64K, the same program. This branch makes TAW follow TMEM_WORDS
(otpu_seq's TMEM_WORDS parameter; 16 bits up to 64K words, so the card's build does not
change); tmem-pump's branch has the same change (4eca9a76, with the STREAM descriptors kept in
the first 64K). Every TMEM-256K figure here is with the change.

## 4. Measured: the tiled run against today's

`tools/perf_tiled.py` (section 1's setup): "ldc" is the card's memory path (DDR3-1066 through
LiteDRAM's controllers, the core at 133.33 MHz), "port" a memory that delivers a chunk every
cycle with a 40-cycle latency, which stands in for the BCU1525's DDR4 at 1600 or 2400 and the
F2's HBM (each delivers more than a slice's port takes, section 2: their cycles are the same).
MCOLS 8 is virtual (the RTL at MCOLS 8; the card's part does not fit it). Cycles a row for the
first 2 layers, positions from 256, no logits; tokens/s are the whole model at that rate.

### 4.1 Today's kernel with replay

compile_rows over R rows in one run, ACT_ROWS = max(MCOLS, R), TMEM 64K words up to 8 rows
and 256K past that:

| weights | MCOLS | memory | R = 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| fp4 | 4 | ldc | 83.6K | 82.6K | 83.1K | 84.7K |
| | 4 | port | 82.7K | | | |
| | 8 | ldc | 57.7K | 53.8K | 53.0K | 54.0K |
| | 8 | port | 55.2K | 53.0K | | |
| int8 | 4 | ldc | 89.2K | 83.8K | 83.8K | 85.0K |
| | 8 | ldc | 89.2K | 56.8K | 54.5K | 54.4K |

- **Past the knee more rows a pass give nothing.** The port-B chunks a row fall 2.8x (fp4
  41.2K -> 14.5K, 4 -> 32 rows) and the cycles stay within 2%.
- **Up to the knee they give what section 2 predicts**: int8 at MCOLS 4 -6% (4 -> 8 rows),
  fp4 at MCOLS 8 -7%, int8 at MCOLS 8 -36%. Each fits 64K words of TMEM at 8 rows; what it
  needs is ACT_ROWS 8.
- **The memory is not the bound**: the port's full rate is 1-4% faster.

### 4.2 The tiled run

compile_tile, TMEM 64K words, ACT_ROWS 16 (T = 4 also on ACT_ROWS 4, the card's: 79.5K, the
same). The best of today's kernel at the same configuration for comparison:

| weights | MCOLS | memory | T = 4 | 8 | 16 | today's best (rows) | tile at 16 against it |
|---|---|---|---|---|---|---|---|
| fp4 | 4 | ldc | 79.5K | 76.5K | 76.4K | 82.6K (8) / 83.6K (4) | -7.5% / -8.6% |
| | 4 | port | 78.5K | 76.5K | 76.5K | 82.7K (4) | -7.5% |
| | 8 | ldc | 49.1K | 45.1K | 44.4K | 53.0K (16) / 57.7K (4) | -16% / -23% |
| | 8 | port | 47.9K | 44.6K | 44.4K | 53.0K (8) | -16% |
| int8 | 4 | ldc | 84.7K | 76.5K | 76.4K | 83.8K (8) / 89.2K (4) | -8.8% / -14% |
| | 8 | ldc | 82.5K | 48.8K | 44.4K | 54.4K (32) / 89.2K (4) | -18% / -50% |

Where the cycles go (MCOLS 4, fp4; the tile's attention phase includes the next head's
projections, which it overlaps, so the attention block is compared whole: input norm,
projections, attention, output projection):

| run | attention block | mlp | MXU busy |
|---|---|---|---|
| today, R = 4 | 45.5K | 38.1K | 80.5K |
| tile, T = 4 | 41.5K | 37.9K | 76.5K |
| tile, T = 16 | 38.7K | 37.7K | 74.4K |
| MCOLS 8: today, R = 8 | 33.6K | 20.3K | 50.5K |
| MCOLS 8: tile, T = 16 | 25.1K | 19.3K | 42.4K |

- **The gain is the attention block's.** Packed tiles put MCOLS / G rows' queries on the
  columns that one row's G = 2 left idle (at MCOLS 4: the attention MMs at full width, -4.0K at
  T = 4); at T = 8-16 its projections pass the weights once for all rows and overlap the
  attention (-2.8K more). The MLP is at its weights' floor either way (1.02x).
- **MCOLS 8 is where it matters**: four rows share a K / V stream and the projections reach the
  floor that 4-row runs cannot (PAIR's pairs stream-bound at 0.294 a chunk): -23% against 4-row
  runs, -17% against today's kernel at its best (16 rows with replay).
- **What is left over the floor** at T = 16: 15.0K a row at MCOLS 4 (76.4K against 61.4K),
  13.7K at MCOLS 8: the attention's MXU work (estimated ~5K a row at MCOLS 4: K and V^T take
  ~650 cycles a packed tile and KV head over two blocks) and its MMs' fills and dependency
  chains.

**Longer context** (position 1024, five blocks; fp4):

| MCOLS | today, R = 4 | today, R = 8 | tile, T = 16 | change |
|---|---|---|---|---|
| 4 | 110.8K (attn-core 41.1K) | | 92.4K | -17% |
| 8 | 84.1K | 82.0K | 58.4K | -29% |

The tile's T = 16 program is 4,986 instructions there at MCOLS 4 (over the card's 4K: the
looped attention first, 3.4) and 3,002 at MCOLS 8.

**Larger and hybrid models** (fp4, from position 256; a Qwen3-4B-shaped layout's 2 layers: H
2560, F 9728, 32 / 8 heads (G = 4), d 128; Qwen3.5-0.8B's first 4 layers: 3 DeltaNet, 1 gated
attention):

| model | MCOLS | today, R = 4 | tile, T = 4 | T = 8 | T = 12 | best change |
|---|---|---|---|---|---|---|
| Qwen3-4B shape | 4 | 428.7K | 428.2K | 423.9K | 434.2K | -1.1% |
| | 8 | 254.2K | 249.3K | 226.9K | (TMEM) | -10.7% |
| Qwen3.5-0.8B (today's kernel only) | 4 | 282.4K | | | | |
| | 8 | 233.7K | | | | |

- **The 4B shape is its MLP**: 294.4K of 428.7K a row (69%) at 1.01x its weights' floor; the
  attention is 5% (20.4K) and its G = 4 query heads already fill MCOLS 4's columns (P = 1: no
  packing). So the tile has nothing to take at MCOLS 4. At MCOLS 8, 8 rows reach the 4-bit
  floor that 4-row paired runs miss (stream-bound at 0.294 a chunk): -10.7%, the MLP at its
  floor (147.9K against 145.9K).
- **Qwen3.5-0.8B is its DeltaNet**: the DeltaNet rows (perf_tiled's deltanet phase and the
  time outside the named phases: the recurrence, convolutions and gates) take 173K of 282.4K a
  row (61%; 170K of 233.7K at MCOLS 8, 73%), per row on the DSTEP and the VPU. MCOLS 8 gives
  -17% (docs/prefill.md 8 measured -17.2% on the whole model); a tile (not ported) would add at
  most the linear layers' 15% stream share at MCOLS 8, about -4% (estimated), nothing at
  MCOLS 4. Its prefill lever is the DSTEP (16 lanes, docs/prefill.md 8).

So the tile's gain is the attention's share and the MXU's appetite for rows: largest on small
models with many KV heads (Qwen3-0.6B, G = 2), at long context and at MCOLS 8; about nothing on
a 4B-class fp4 model on the card.

### 4.3 The whole model, by board

Qwen3-0.6B, fp4, prefill tokens a second of device time at positions ~256 (28 layers from the
2-layer cycles; no host time, no logits):

| board | today (R = 4) | today at its best | tile, T = 4 | tile, T = 16 |
|---|---|---|---|---|
| YPCB (MCOLS 4, DDR3-1066, 133.33 MHz) | 114 | 115 (R = 8, ACT_ROWS 8) | 120 (today's bitstream) | 125 (ACT_ROWS 16) |
| the same at position 1024 | 86 | | | 103 |
| MCOLS 8, DDR3-1066, 133.33 MHz | 165 | 180 (R = 16) | 194 | 214 |
| BCU1525 / F2, one slice, MCOLS 4, 100 MHz | 86 | | 91 | 93 |
| BCU1525 / F2, one slice, MCOLS 8, 100 MHz | 129 | 135 (R = 8) | 149 | 161 |

int8 weights (MCOLS 4 at 133.33 MHz): 107 today, 114 at R = 8, 125 with the tile at T = 8;
at MCOLS 8: 107 / 175 (R = 16) / 214 (T = 16).

## 5. MoE

### 5.1 Expert-major is a tiled prefill already

Layer-major prefill with expert-major MoE (docs/offload.md 13.11-13.13, merged, off by default)
runs each MoE layer of a chunk (up to 512 rows, the rows in DRAM records) as mixer runs of R = 2
rows, then one expert run that buckets the chunk's R k entries by expert and makes passes of up
to C rows of one expert (C = 4 on the card: docs/offload.md 13.18). That is the tiled run's structure with the MoE layer's tile being "the
rows that chose this expert": the rows live in DRAM between the phases (3.3's spill), each pass
loads its rows, quantizes them and streams the expert once. Tiling generalizes it in two places:

- **The mixer runs become tiles.** Their dense weights (attention or DeltaNet projections, the
  shared expert or Gemma 4's dense MLP, the router) are streamed once per run, so at R = 2 a
  run's weight stream is shared by 2 rows. With the tile layout (3.2) a mixer run takes 8-16
  rows, its dense weights once for all of them, its attention packed.
- **The expert passes take MCOLS / 2 rows (4-bit experts, PAIR) or up to ACT_ROWS (replay).**
  A 4-bit expert at MCOLS 4 is at the floor with pairs already (two rows paired cost one
  row's cycles; 4 rows cost two pairs'); at MCOLS 8 a pass of 4 paired rows costs what 2 do
  now. Replay past that saves the expert's re-reads (a 26B expert used by 13 rows is streamed
  7 times in pairs, once in a 16-row pass) but not cycles: it matters only where the stream is
  shared (the host's H2C writes during compute: 0.17 s a GB, docs/offload.md 12.6).

One expert's FFN on M rows, co-simulated (tools/moe_prefill_cosim.py expert: the card's memory
path, ACT_ROWS 16, TMEM 64K; the expert's cycles less the same program without it; ms at
133.33 MHz):

| expert | MCOLS | M = 1 | 2 | 4 | 8 | a row at the floor |
|---|---|---|---|---|---|---|
| Qwen3.5-35B-A3B (1.67 MB fp4) | 4 | 0.101 | 0.100 | 0.194 | 0.389 | 0.049 |
| | 8 | 0.101 | 0.100 | 0.104 | 0.205 | 0.024 |
| gemma-4-26B-A4B (3.45 MB fp4) | 4 | 0.210 | 0.210 | 0.393 | (TMEM) | 0.101 |
| | 8 | 0.210 | 0.210 | 0.216 | (TMEM) | 0.050 |

- **At MCOLS 4 a pair is the floor**: 4 rows cost two pairs (0.194 against 2 x 0.100), 8 rows
  four. Replay saves the stream, not cycles, as section 2 has it.
- **At MCOLS 8, 4 rows cost what 2 do** (0.104 against 0.100): the expert passes of the
  expert-major run should take MCOLS / 2 rows, not 2.
- The 26B's 8 rows (and either model's 16) do not fit 64K words with this kernel (swiglu_down
  keeps the [M, H] input and [M, F] products); the tile's MLP (3.2: pieces, parts) would fit
  them, for the stream's sake only.

### 5.2 What it gives the two MoE models

Each layer's dense weights (from the config.json shapes):

| model | dense weights a layer | format (as run) | a token's dense MACs | expert | experts a token |
|---|---|---|---|---|---|
| gemma-4-26B-A4B | 52.8M (sliding) / 70.2M (global) | int8 | 1.67G (30 layers) | 5.95M (3.45 MB fp4) | 8 of 128 |
| Qwen3.5-35B-A3B | 37.4M (DeltaNet) / 30.9M (attention), with the shared expert | fp4 | 1.43G (40 layers) | 3.15M (1.67 MB) | 8 of 256 |

**YPCB (experts offloaded, Gen2 link).** The measured split of a pooled R = 2 prompt
(docs/offload.md 13.7: 35B 134 tokens, 26B 124 tokens): layer runs' compute 8.29 s (35B) and
13.72 s (26B), of which the mixer ("pre", up to the request) is 1.94 / 1.18 ms a run (35B
DeltaNet / attention) and 3.33 / 3.85 ms (26B sliding / global), plus 3.46 / 2.12 s of exposed
misses.

- **26B: the mixer runs are its int8 dense weights at 2 rows a pass.** 52.8 MB a sliding layer is
  412K chunks: at 3.33 ms a run that is the stream at 93% of the port, so the mixer is
  stream-bound at 1.67 ms a row. In tiles of T >= 8 at MCOLS 4 the same weights take the int8
  floor, 0.25 cycles a chunk and row: 0.77 ms (sliding), 1.03 ms (global) a row. Over 124
  tokens and 30 layers: **about -3.1 s** (estimated) of the pooled prompt's 16.25 s after the
  compile fix (-19%), or -2.6 s if the tile runs these layers at 1.25x the floor as it does
  Qwen3-0.6B's (section 4.2). T = 4 on today's bitstream (stream-bound at 0.294 a chunk and
  row) gets -2.2 to -2.6 s of it; MCOLS 8 (int8 floor 0.125) -4.1 to -4.8 s.
- **The tiled mixer needs expert-major's split.** In today's layer-major runs each union expert
  runs on all R rows of the run, so more rows mean a larger union, and the request is one line
  of R k <= 16 ids (docs/offload.md 13.2: R = 2 is the point for 4-bit experts). Expert-major
  runs the mixer apart from the experts, so its mixer runs can take any T; the tile goes on
  that path.
- **35B: little.** Its dense layers are fp4, where 2 paired rows are at the stream's rate and the
  floor is 15% below; its DeltaNet recurrence (32 value heads, 2,048 cycles each a row on the
  8-lane DSTEP: 0.49 ms a row and layer) does not tile. About -0.3 to -0.5 s (estimated).
- **Expert-major (dormant) is the other half.** Its predicted gain (docs/offload.md 13.11's model,
  needs streamed during the mixer runs, against today's pooled base) is -1.7 s (35B) and -1.9 s
  (26B): the link side, which tiling does not touch. For the 35B it is most of what is left
  without new RTL; for the 26B the tiled mixer is the larger half, and the two add (the tiled
  mixer runs shorten the phase that the need-streaming hides under, so the overlap shrinks:
  the sum is an upper bound).
- **The card A/B of expert-major needs card time** (session pfB, staged on the card host in
  ~/otpu-build/offload/card2/pfB/README.md with its predicted column).
  Nothing in this note changes that session's arms.

**BCU1525 / F2 (experts resident, no link).** Estimated per token and slice at the floor, MCOLS
4, the port's rate (lower bounds: the tile ran Qwen3-0.6B at 1.25x its floor):

| model | dense | experts | DeltaNet steps | today's R = 2 | tiled |
|---|---|---|---|---|---|
| 26B | 3.26M cycles (int8; R = 2 at the port: 6.5M) | 2.79M (fp4: pairs are at the floor) | | ~9.3M + attention | ~6.1M + attention (-35%) |
| 35B | 2.80M (fp4; R = 2 paired: at the floor) | 1.96M | 1.97M (30 layers x 32 heads x 2,048) | ~6.7M | ~6.7M |

At 100 MHz (the BCU1525's development clock) that is ~10.7 -> ~16 tokens/s a slice for the 26B
and ~15 for the 35B. At MCOLS 8 the MXU parts halve and the 35B's DeltaNet steps (1.97M) become
its bound: its prefill lever is the recurrence (a 16-lane DSTEP, docs/prefill.md 8, or the
chunked delta rule as MXU work: an algorithm change, not bit-exact, needing its own accuracy
check), not more rows.

## 6. Many users

Batched decode (opentpu/llm/batch.py, docs/serve.md) runs a step's rows in blocks of MCOLS
slots, one run each, every run streaming every weight. With replay a whole step takes one pass:
rows of different sequences are rows of one stationary operand (up to ACT_ROWS), each row's
attention an entry of its own over its own cache (rows of different sequences share no K / V,
so they do not pack: the tile's main gain does not apply). What one pass of 2 MCOLS rows gives
against two runs of MCOLS is what today's kernel measures at 8 and 16 rows against 4 and 8
(section 4.1; 2 layers at position 256, cycles a row):

| MCOLS | weights | runs of MCOLS rows | one pass of 2 MCOLS rows | change |
|---|---|---|---|---|
| 4 | fp4 | 83.6K | 82.6K | -1.2% |
| 4 | int8 | 89.2K | 83.8K | -6.1% |
| 8 | fp4 | 53.8K | 53.0K | -1.5% |
| 8 | int8 | 56.8K | 54.5K | -4.0% |

- **Longer passes buy 1-6%**, int8 most (its knee is 2 MCOLS). A decode step's int8 LM head
  (18% of a Qwen3-0.6B decode row) is in the same case: about 3% more for a step with logits
  (estimated). It needs ACT_ROWS 8 at MCOLS 4 and nothing else (64K words of TMEM hold 8 rows).
- **What raises the aggregate is the MXU's rate**: 8 users' rows at MCOLS 8 in one run cost
  53.8K a row against 83.6K in two runs at MCOLS 4 (-36%; fp4). So 4-8 users gain from MCOLS 8
  or a pumped MCOLS 4, not from longer passes.
- **The attention is per user.** At context 1024 the card's 4-row step took 60.6 ms against
  46.3 ms at 64 (docs/serve.md): per-row attention, which no tiling across users removes.
- **Mixed runs** (a prompt's rows with other users' decode rows, docs/serve.md) are where the
  tile helps a multi-user server: a prompt's rows ride the decode step's weight pass with their
  attention packed, instead of a feed run of MCOLS rows per pass.

## 7. The prototype

- `opentpu/llm/tiled.py`: qwen3_tile and compile_tile (Qwen3 and the Llama-likes of this code
  path; one slice; compile-time positions). Qwen3.5's gated attention and DeltaNet rows, LFM2
  and Gemma 4 are not ported.
- `opentpu/kernels/attention.py`: RowMasks (a packed tile's masked block) and TileSeq (whole
  blocks looped, then the listed ones); the existing paths are unchanged.
- `opentpu/compiler.py`, `opentpu/language.py`: ol.stationary / ol.quantize_into,
  ol.quantize(free=, rscale=, cscale=), ol.sumsq. Defaults leave every existing program as it
  was.
- `rtl/seq/otpu_seq.sv`, `rtl/top/otpu_slice.sv`: the TMEM hazard ranges' width follows
  TMEM_WORDS (3.5; no change at 64K words).
- `tools/perf_tiled.py` (the co-simulations of section 4), `tools/tile_model.py` (section 2's
  table, the MoE pass counts).
- `tests/test_tiled.py`:
  - a tile against today's prefill runs of MCOLS rows over the same tokens (Engine's cache of
    the tokens before them; the ISA simulator): every layer block (weights and KV cache) and
    the last row's logits bit for bit, at MCOLS 4 int8 (16 rows), MCOLS 4 fp4 (16 rows across
    the attention block at 256), MCOLS 2 int8 (6 rows: one row a packed tile, a partial MXU
    group) and MCOLS 8 fp4 (14 rows: 4 a packed tile, a partial last one);
  - the packed tiles' blocks and masks (tile_seq);
  - the RTL against the ISA simulator (DRAM and TMEM equal), on the board's memory path: a
    12-row tile across a block at MCOLS 4 fp4 and MCOLS 8 int8 (ACT RAM of 16 rows), and a
    4-row tile on the card's own configuration (MCOLS 4, ACT_ROWS 4, fp4).

**The order differences, exactly.** None in the tile itself: an MM's results do not depend on
how many rows share it (docs/isa.md), splitting W_q / W_k / W_v / W_o / W_down by rows gives each
output element the same MM sum, the down projection's parts keep swiglu_down's order of
chunks, per-block QACTs give per-block bytes and scales, and a masked or fully masked block is
exact (section 3.2). The one difference against today's runs is PAIR: a run of at most
MCOLS / 2 rows of 4-bit weights is paired (two blocks a cycle), and PAIR adds a chunk's two
blocks before the partial sums (docs/isa.md, Column reuse), so its sums round differently from
an unpaired MM. A tile pairs exactly when today's run would (2T <= MCOLS); a tile of more rows
matches today's runs of more than MCOLS / 2 rows bit for bit, and differs from a paired run
(e.g. the 2-row runs of the Qwen3.5-4B, docs/prefill.md 7) by that rounding only. The test's
MCOLS 8 case takes runs of 8 and 6 rows for that reason (a run of 4 would pair).

## 8. Plan, and what it gives each workload on each board

In order; each step stands on its own.

1. **The tile kernel in the engine's prompt runs** (software; at T = MCOLS it runs on today's
   bitstream). Scheduled with the MXU work below rather than ahead of other card work: on the
   card alone it is worth -5% on Qwen3-0.6B (-17% at position 1024 with T = 16) and about
   nothing on 4B-class fp4 models.
   - run-time positions: the tile as a prompt run (RunPos: the rows share one tpos), so one
     program serves every tile of a bucket, cached as today's prompt runs. Its masks come from
     a table as the prompt runs' additive masks do (docs/prefill.md 9), a second one: per
     position q, row group i (G rows) masked at q + i, so a packed tile's mask is one LD at
     the table + tpos x its stride (1 MiB at 256 x MCOLS 4, as the first);
   - the looped attention of docs/prefill.md 10 over a head's packed tiles, so the program
     stays under 4K instructions past ~1K tokens;
   - the planner's T: the knee (MCOLS for 4-bit, 2 MCOLS for int8) bounded by what trace_fit
     fits in TMEM and by ACT_ROWS; Engine.prefill takes the tile where its cost a row is lower
     than today's (as prefill.covers decides prompt runs today);
   - Qwen3 and the Llama-likes first, then LFM2, Gemma 4 and Qwen3.5's attention layers (the
     DeltaNet rows stay per row).
2. **With MCOLS 8 or a pumped MXU** (dsp-pump's and mcols' work) the tile is the prefill
   kernel: 4-bit's knee is then 8 rows (no replay at MCOLS 8; ACT_ROWS 8 for a 2x-pumped MCOLS
   4), and it takes Qwen3-0.6B from 1.45x to 1.88x the card's prefill today, the 4B shape from
   1.69x to 1.89x.
3. **ACT_ROWS 8-16 in a card bitstream that has room** (`make bit ACT_ROWS=16`; a FAST=1 build
   at 100 MHz first for the function: tests/test_tiled.py's RTL cases on the card, then a tile
   against compile_rows on a real prompt, token for token; then the timing build): int8 layers
   reach their knee (Qwen3-0.6B int8 -14% with the tile at T = 8), batched decode a pass of 8
   (int8 -6%).
4. **MoE**: the pfB A/B of expert-major first (card time), since the
   tiled mixer goes on its path (5.2); then its mixer runs as tiles (Gemma 4 26B first:
   section 5.2's largest estimate), expert passes of MCOLS / 2 rows (2 at MCOLS 4, 4 at 8).
5. **Later, measured first in co-simulation**: RMAX / ASCALE over replayed groups (fewer
   attention MMs; at most ~9K of the 76.4K at MCOLS 4, estimated); the DRAM spill only if a
   tile must outgrow TMEM (MCOLS 16).

What it gives each workload, by board (YPCB: the card, MCOLS 4, DDR3-1066, 133.33 MHz; MCOLS 8:
the faster MXU of dsp-pump's and mcols' work, same memory; BCU1525 and F2: one slice at the
port's rate, 100 MHz; device tokens a second of prefill at positions ~256, fp4; measured = the
co-simulations above, estimated = arithmetic on them):

| workload | YPCB today | YPCB, tile on today's bitstream | YPCB, + ACT_ROWS 16 | MCOLS 8: today's kernel -> tile | BCU1525 / F2 (MCOLS 4 today -> tile; MCOLS 8 tile) |
|---|---|---|---|---|---|
| single-user decode | unchanged (the tile is a prefill kernel) | unchanged | unchanged | unchanged | unchanged |
| prefill, Qwen3-0.6B | 114 | 120 (+5%) | 125 (+9%) | 165 -> 214 (+30%) | 86 -> 93; 161 |
| prefill, Qwen3-0.6B at ~1024 | 86 | | 103 (+20%) | 113 -> 163 (+44%) | |
| prefill, Qwen3-4B shape | 17.3 | 17.3 | 17.5 (-1.1%) | 29.1 -> 32.6 (+12%) | |
| prefill, Qwen3.5-0.8B | 78.7 | not ported (~0) | ~0 | 95.1 -> ~99 (estimated) | |
| multi-user 4 | unchanged | unchanged | unchanged | MCOLS 8: -31% a row (57.7K / 83.6K) | as YPCB |
| multi-user 8 | two runs of 4 | unchanged | one pass of 8: -1% (fp4), -6% (int8) | one run of 8: -36% a row | as YPCB |
| MoE prefill, 26B (offloaded; expert-major's mixer runs as tiles) | 16.25 s (124 tokens, pooled, after compiles) | -2.2 to -2.6 s (estimated) | -2.6 to -3.1 s (estimated) | -4.1 to -4.8 s (estimated) | resident: ~9.3M -> ~6.1M cycles a token (estimated) |
| MoE prefill, 35B | 13.32 s (134 tokens, likewise) | -0.3 to -0.5 s (estimated) | same | DeltaNet-bound | ~6.7M a token either way (estimated) |

Expert-major's -1.7 s (35B) / -1.9 s (26B) adds to the MoE rows once the pfB A/B confirms it.

## 9. Risks and open points

- **No card measurement of a tile.** Every cycle count here is a co-simulation of the card's
  memory path (DDR3-1066 through LiteDRAM's controllers) or of a memory at the port's rate
  standing in for the BCU1525's DDR4 and the F2's HBM. The co-simulation of today's kernel is
  ~6% below the card's prompt runs (section 1); a tile as a prompt run pays the same run-time
  position costs, so its gain is expected within a few points of section 4's.
- **The 2-layer proxy at position 256** has two attention blocks a row; a short prompt's rows
  have one (less attention, so less tile gain), a long one's more (section 4's position 1024:
  more gain).
- **ACT_ROWS 16 on the card is unbuilt.** Replay is in the RTL and ACT_ROWS 8 was built once
  (2026-09-27, at MCOLS 2: docs/benchmarks.md); the BRAM count (unchanged at MCOLS 4) is read
  off the RAM shapes, not synthesized, and the replay mux's timing at 133.33 MHz is unknown. A
  dev build answers both. Step 1 of the plan does not need it.
- **PAIR rounding.** A tile of more than MCOLS / 2 rows matches today's unpaired runs bit for bit
  but not today's paired runs of 2 rows (the Qwen3.5-4B's at MCOLS 4): a model whose reference
  tokens were made with paired runs gets new reference tokens (refcache keys), not a failure.
- **IMEM past ~1K tokens.** The prototype unrolls the packed tiles; a 16-row tile at position
  1024 is 5.0K instructions at MCOLS 4, over the card's 4K. The looped attention of
  docs/prefill.md 10 is needed before long prompts take tiles of 16 (today's 4-row run is
  2,357 instructions there).
- **Models not ported.** The prototype is Qwen3's code path; Qwen3.5 (gated attention with G = 4,
  so no packing at MCOLS 4, and the DeltaNet's per-row DSTEP), LFM2's convolutions and Gemma 4's
  sliding / global layers and per-layer embeddings each need their tile kernel. The MoE numbers
  of section 5 are estimates on measured parts, the expert-major gain a model's prediction
  until the pfB session.
- **Co-simulations with a TMEM over 64K words** made before otpu_seq's fix (3.5) are pessimistic
  wherever the compiler placed tiles past word 65,535; that includes this note's first runs (not
  used here) and any other branch's that did not carry the change.
- **A user decision:** none needed for the software steps. An ACT_ROWS 16 bitstream and its card
  qualification, and the pfB A/B, take card time, which the maintainer schedules.

## 10. In the engine (2026-10-12)

The tile now serves every card model family's prefill, opt-in: `Engine(tile_runs=True)` or
`OTPU_TILE=1` (prefill.tile_flag; with the MX numerics the dense models' default: 11.3). Dense models take tiled prompt runs (prefill.py's tile mode:
tiled.compile_tile_run at a run-time position, a program per bucket and kind as today's prompt
runs); MoE models take tiled expert-major layer runs (Engine.layer_tile:
tiled.compile_layer_tile). Branch tiled-prefill.

### 10.1 What runs tiled

| family | prompt runs (dense) | layer runs (MoE, expert-major) |
|---|---|---|
| Qwen3 and the Llama-likes (Phi-4-mini, SmolLM3) | qwen3_tile | |
| Qwen3.5 | qwen35_tile: gated attention tiled, DeltaNet rows in runs of the rows that fit (their pair buffers grow with the rows) | qwen35_layer_tile (35B): the same; the route, need lines and shared expert a need line's rows at a time |
| LFM2 / LFM2.5 | lfm2_tile: attention, convolutions by channel parts | |
| Gemma 4 (E2B, E4B with PLE) | gemma4_tile: sliding / global spans, KV-shared layers, PLE | gemma4_layer_tile (26B): attention, then the route and need lines, then the dense MLP over the T rows |

- **Forms** (compile_tile_run / compile_layer_tile try them until one fits TMEM, ACT RAM and
  IMEM: `_first_fit`): as it is; x spilled to DRAM (the I/O input rows, or the expert-major
  records' X) while the heads attend and loaded back into its own TMEM words (x is carried
  across the layers' hardware loops); Qwen3.5's DeltaNet in runs of fewer rows; each at the
  attention's pipeline depth 3, then, over IMEM, at depth 2 (SHALLOW).
- **Query parts.** A KV head's query group runs in parts of at most MCOLS heads whose
  [T, heads d] queries take at most TMEM / 8 words (QPART); each part's rows pack MCOLS query
  rows to an MM, so K / V stream G / MCOLS times a row whatever the part. This is what fits
  Gemma 4's 8-head groups, Qwen3.5-35B's G = 8 and Qwen3.5's 16-row tiles.
- **The engine's rows.** With prompt runs and tiling on, the image's I/O rows grow to ACT_ROWS
  (an L run of T rows writes row T - 1's logits). A MoE image built with OTPU_TILE=1 holds the
  mask tiles its layer runs need (2 MiB at MCOLS 4; the probe that sizes the expert slots sees
  them too). Layer 0's runs that gather their rows from the host's slot stay untiled.
- **References.** tools/qual/refs.py's key holds OTPU_TILE (and OTPU_ACT_ROWS): a tiled qual's
  references are its own.

### 10.2 Correctness

- **ISA: every family and form bit for bit against today's runs** (tests/test_tiled.py):
  prompt runs of Qwen3 (MCOLS 2, 4, 8), SmolLM3, Phi-4-mini (G = 3), LFM2, Qwen3.5 (8 and 16
  DeltaNet heads, query parts of 2 heads, x spilled, DeltaNet in runs of 4), Gemma 4 (1 and 2
  KV heads, int8 and fp4, PLE from the host, query parts of 1 and 2 heads), the shallow
  pipeline over 1,030 tokens; expert-major layer runs of the tiny Gemma 4 MoE and Qwen3.5-MoE
  (8 and 6 rows against token by token in int8, and against today's expert-major runs of 4
  rows in 4-bit): logits, KV caches, DeltaNet states and windows, and the decode steps after.
- **PAIR, exactly.** A 4-bit MM of at most MCOLS / 2 rows pairs (two K blocks a cycle, summed
  before the partial sums: a different rounding). A tile and today's run give the same bits
  wherever both pair or neither does; they differ (last bits; max |diff| / max |y| < 1e-3 on
  the logits) exactly where one of them pairs, e.g. today's 2-row or 1-row runs at a chunk's
  end against a tile's 4 or more rows (test_tiled_prompt_runs_pair_rounding; a Qwen3.5-MoE
  chunk of 13 rows by 4 left a 1-row run: 4 words of the layers' 8.4 MB differed, the logits
  not). int8 never pairs. A DeltaNet split runs its rows as evenly as they go (6 by 4 is
  3 + 3, not 4 + 2, whose 2 rows would pair).
- **RTL = ISA** (DRAM and TMEM, the board's memory path), on the card's configuration (MCOLS
  4, ACT_ROWS 4) and on ACT_ROWS 16 / MCOLS 8: prompt runs of Qwen3 (4 rows, ACT RAM of 4),
  LFM2 (16), Qwen3.5 (16; and x spilled with the DeltaNet in runs), Gemma 4 (the sliding
  window's start-masked block, 4 rows at position 600; MCOLS 8, 8 rows); expert-major layer
  runs of the Gemma 4 MoE (a sliding layer's 4 rows on ACT_ROWS 4, a global layer's 8 on 16)
  and of Qwen3.5-MoE (an attention layer's 4 rows on ACT_ROWS 4, the DeltaNet layer after it,
  8 rows on 16). (With the LANES 16 design-point configuration a Qwen3.5-MoE DeltaNet layer
  run, today's untiled one too, trips otpu_tmem.sv's copy check at the dump: a word written
  to one TMEM copy only. The card's LANES 8 passes.)

### 10.3 Rows a tile holds

Compile only (tools/perf_prompt.py --compile-only, fp4 layers, int8 head, the card's 64K-word
TMEM and 4K-instruction IMEM, bucket 2), against today's prompt runs:

| model | today | MCOLS 4, ACT_ROWS 4 | MCOLS 4, ACT_ROWS 16 | MCOLS 8, ACT_ROWS 16 |
|---|---|---|---|---|
| Qwen3-0.6B | 8 | 4 | 16 | 16 |
| Qwen3.5-0.8B | 4 | 4 | 16 | 16 |
| Qwen3.5-2B | 4 | 4 | 12 | 12 |
| Qwen3.5-4B | 3 | 4 | 8 | 8 |
| LFM2-2.6B | 3 | 4 | 8 | 8 |
| LFM2.5-230M | 8 | 4 | 12 | 16 |
| Phi-4-mini | 3 | 4 | 8 | 12 |
| SmolLM3-3B | 4 | 4 | 12 | 16 |
| Gemma 4 E2B | 4 | 4 | 8 | 12 |
| Gemma 4 E4B | 4 | 4 | 8 | 8 |
| gemma-4-26B-A4B (expert-major layer runs) | 2 | 4 | 8 | |
| Qwen3.5-35B-A3B (expert-major layer runs) | 2 | 4 | 8 | |

(The survey's IMEM check is at the attention's depth 3; the engine retries an over-IMEM size
at depth 2, which the survey did not: Qwen3.5-2B's 16 rows were 5,790 instructions at depth 3.)

**Long prompts: IMEM, not TMEM, holds the tile past bucket ~3.** A packed tile's attention
leaves up to ~3 x depth block bodies unrolled (the pipeline's prime, its hardware loop's body,
the blocks after it), the most at bucket 8. Instructions at buckets 2 / 8 / 15 (MCOLS 4,
ACT_ROWS 16, depth 3): Phi-4-mini T = 8 3,176 / 7,726 / 6,318; LFM2-2.6B T = 4 2,331 / 4,609 /
3,905; E2B T = 8 2,994 / 4,355 / 4,003; SmolLM3 T = 8 2,211 / 4,489 / 3,785; Qwen3-0.6B T = 16
3,066 / 7,616 / 6,208. At depth 2 (bucket 8): Phi-4-mini T = 4 2,604, LFM2-2.6B T = 4 3,201,
E2B T = 8 3,651, SmolLM3 T = 8 3,081, Qwen3-0.6B T = 8 2,582, Qwen3.5-0.8B T = 12 3,576. The
engine takes the largest T that fits each bucket (prefill._probe_tile, a program cache fact),
at depth 2 where depth 3 is over IMEM (the depth costs 0.0-0.14% a row: 10.4). Bucket 2's
tiles hold to about bucket 3; past that T drops to 4-8 on the larger models.

The looped attention of docs/prefill.md 10 (A: an entry loop) is what lifts this: a KV head's
packed tiles as a hardware loop (its queries', outputs' and mask tiles' addresses affine in
the tile, K / V the same each iteration, the last, partial tile peeled), its body one packed
tile's attention with its block loop inside; loop depth layers, entries, blocks (3 of the
ISA's 4). The program is then about one packed tile's attention a KV head and part, so the
tile's T is TMEM's at every bucket. It matters most at MCOLS 8 and on long prompts; its gates
are docs/prefill.md 10's (an ISA count per layout and bucket, an RTL co-simulation of the
looped run against the unrolled one, then bit for bit on the ISA and an RTL test).

### 10.4 Measured (RTL co-simulation, the card's memory path)

Prompt runs (tools/perf_prompt.py --phases: one P run at the start of a bucket, the model's
first layers: 2 for Qwen3-0.6B and Phi-4-mini, 4 for Qwen3.5 and SmolLM3, 5 for Gemma 4, 6 for
LFM2; fp4 layers, int8 head; cycles a row). Today's rows are the most that fit (3-4, 8 on the
small ones) and, where that is 3, also 2, which the planner's PAIR rule takes for 4-bit
(docs/prefill.md 7: 2 paired rows cost about one):

Bucket 2 (position 256), MCOLS 4 (the card):

| model | today (rows) | tile, T = 4 (ACT_ROWS 4: today's bitstream) | tile, larger T (ACT_ROWS 16) |
|---|---|---|---|
| Qwen3-0.6B | 88.8K (8) | 82.9K (-6.6%) | 81.1K (16: -8.7%) |
| Qwen3.5-0.8B | 283.8K (4) | 282.7K (-0.4%) | 281.6K (12: -0.8%) |
| Qwen3.5-4B | 1,120.6K (2; 3: 1,296.8K) | 1,004.7K (-10.3%) | 1,001.1K (8: -10.7%) |
| LFM2-2.6B | 1,092.7K (2; 3: 1,313.8K) | 990.9K (-9.3%) | 987.2K (8: -9.7%) |
| LFM2.5-230M | 168.6K (8) | 163.2K (-3.2%) | 159.7K (12: -5.3%) |
| Phi-4-mini | 477.9K (2; 3: 564.9K) | 433.3K (-9.3%) | 430.6K (8: -9.9%) |
| SmolLM3-3B | 658.9K (4) | 656.1K (-0.4%) | 652.9K (12: -0.9%) |
| Gemma 4 E2B | 574.3K (4) | 574.9K (+0.1%) | 578.6K (8: +0.8%) |
| Gemma 4 E4B | 1,126.3K (4) | 1,127.7K (+0.1%) | 1,133.7K (8: +0.7%) |

Bucket 2, MCOLS 8 (today's runs on ACT_ROWS 8; 4-bit runs of up to 4 rows pair):

| model | today (rows) | tile, T = 8 (ACT_ROWS 8) | tile, T = 16 (ACT_ROWS 16) |
|---|---|---|---|
| Qwen3-0.6B | 62.4K (8) | 50.7K (-18.7%) | 51.5K (-17.5%) |
| Qwen3.5-0.8B | 235.3K (4) | 233.4K (-0.8%) | |
| Qwen3.5-4B | 808.2K (3) | 650.7K (-19.5%) | |
| LFM2-2.6B | 746.7K (3) | 511.5K (-31.5%) | |
| LFM2.5-230M | 103.8K (8) | 92.0K (-11.4%) | 91.9K (-11.5%) |
| Phi-4-mini | 331.3K (3) | 230.3K (-30.5%) | |
| SmolLM3-3B | 385.6K (4) | 344.5K (-10.7%) | 344.6K (-10.7%) |
| Gemma 4 E2B | 323.1K (4) | 309.2K (-4.3%) | |
| Gemma 4 E4B | 711.7K (4) | 596.7K (-16.2%) | |

(Qwen3.5-2B's image needs a 4 GiB DRAM, past what the simulator models: not run. 12 rows at
MCOLS 8 lose: Phi-4-mini 301.8K, Qwen3.5-0.8B 254.8K, E2B 372.2K, the replay's second pass of
4 rows costing a whole one; the engine takes whole passes of MCOLS above MCOLS rows and never
picks 12 there.)

Bucket 8 (position 1792: seven whole blocks and the masked one), MCOLS 4:

| model (layers) | today (rows) | tile, T = 4 | tile, larger T | change |
|---|---|---|---|---|
| Qwen3-0.6B (2) | 140.3K (4) | | 112.1K (8; 112.1K at depth 2) | -20.1% |
| Phi-4-mini (2) | 531.2K (2; 3: 619.4K) | 487.4K (488.1K at depth 2) | | -8.2% |
| LFM2-2.6B (6) | 1,155.3K (2; 3: 1,377.4K) | 1,055.2K (depth 2) | | -8.7% |
| Qwen3.5-4B (4) | 1,146.0K (2) | 1,030.7K | | -10.1% |
| SmolLM3-3B (4) | 720.9K (4) | 720.2K | 716.4K (8, depth 2) | -0.6% |
| Gemma 4 E2B (5) | 741.2K (4) | 742.3K | 745.4K (8, depth 2) | +0.2% / +0.6% |

- **On today's bitstream (T = 4, ACT_ROWS 4)** the tile takes -9 to -10% a row off the models
  whose prompt runs hold 3 rows today (Phi-4-mini, LFM2-2.6B, Qwen3.5-4B: 4 rows reach MCOLS
  4's 4-bit knee, where today runs 2 paired rows), -6.6% off Qwen3-0.6B (G = 2: two rows a
  K / V stream) and -3% off LFM2.5-230M; the rest are within +-0.4%. ACT_ROWS 16 adds 0.4-2.1
  points at MCOLS 4 (Gemma 4 aside).
- **G >= MCOLS** (SmolLM3 and Qwen3.5 G = 4, Gemma 4 E2B G = 8) packs nothing at MCOLS 4, and
  4-bit layers are at their floor at 4 rows: no change there (as section 4.2's 4B shape);
  Gemma 4 at 8 rows is +0.7-0.8% (replay), which a cost-aware choice of T would avoid (the
  engine takes the largest T that fits).
- **MCOLS 8 is where the tile is the prefill kernel**: -11% to -32% a row on every model with
  G <= 4 (two or more rows a K / V stream, and 8 rows at the 4-bit knee that 3-4-row paired
  runs miss), -4% on E2B (G = 8), ~-1% on Qwen3.5-0.8B (DeltaNet-bound).
- **Long prompts** gain more where the attention is a larger share: Qwen3-0.6B -20% at bucket
  8; the 3-row models keep their -8 to -10% there.

Expert-major layer runs of gemma-4-26B-A4B (the mixer: the layer before's end, the attention,
the route and need lines, the dense MLP; int8 dense weights, every need served; ms a row at
133.33 MHz, on tools/moe_prefill_cosim.py's setup):

| layer | position | today, R = 2 | tile, T = 4 (ACT_ROWS 4) | tile, T = 8 (ACT_ROWS 16) |
|---|---|---|---|---|
| sliding | 256 | 2.766 | 1.498 (-45.8%) | 1.411 (-49.0%) |
| global | 256 | 2.850 | 1.812 (-36.4%) | 1.725 (-39.5%) |
| sliding | 900 | 3.665 | 1.949 (-46.8%) | 1.866 (-49.1%) |
| global | 900 | 3.387 | 2.346 (-30.7%) | 2.251 (-33.5%) |

- **The 26B's mixer runs halve on today's bitstream.** Its int8 dense weights (the attention
  projections and the 2,112-wide dense MLP) stream once for 4 rows instead of 2, and the
  sliding layers' G = 2 packs two rows a K / V stream. Over its 30 layers (25 sliding, 5
  global) that is about -37 ms a prompt row of the layer runs' device time at T = 4 (-40 ms at
  T = 8) against ~83 ms today: for a 124-token pooled prompt about -4.6 s (-4.9 s) of device
  time (estimated from the co-simulated layers; section 5.2's estimate from the weights alone
  was -2.2 to -3.1 s). How much the card shows depends on the link: expert-major hides the
  experts' streaming under these runs, so shorter runs can expose misses (docs/offload.md
  13.11); a card A/B measures it.
- **Its bits are today's.** The 26B's dense weights and router are int8 (no PAIR) and its
  experts run in the layer's expert run as today: the tiled runs give today's expert-major
  tokens, so chatmoe's references stand.

### 10.5 ACT_ROWS 16

tools/synth/ooc_rows.py (Vivado OOC of otpu_actram and otpu_mxu by ROWS on the card's part):
today's MXU at ACT_ROWS 16 costs +259 LUT, +72 FF and no BRAM (the ACT RAM's 512-deep groups x
blocks fit the same BRAM36s; 32 rows would double them), WNS +0.875 ns against +0.773 at 4;
dsp-pump's MX column the same (+268 LUT, +72 FF). The MX build carries ACT_ROWS 16
(dsp-pump's decision, 2026-10-12).

### 10.6 Next

- **Card, today's bitstream (the tree before MX, bebf375f: after the MX
  merge the host refuses a bitstream without the MX numerics):** an OTPU_TILE=1 qual of the
  dense models (prompt runs -9 to -10% a row on Phi-4-mini, LFM2-2.6B and Qwen3.5-4B, -6.6% on
  Qwen3-0.6B, -3% on LFM2.5-230M in co-simulation; the rest within +-0.4%) and the 26B's
  expert-major prefill with tiled layer runs of 4 rows (Engine layer_major=4; chatmoe
  token-exact against today's; a TTFT A/B against expert-major at R = 2). New images: the
  dense ones' I/O rows, the MoE ones' mask tiles (OTPU_TILE=1 as they are built). The dense
  references move only where today's runs paired (4-bit runs of 1-2 rows).
- **With the MX build (ACT_ROWS 16):** T = 8-16; references recomputed with OTPU_TILE=1 and
  OTPU_ACT_ROWS=16 (MX changes them anyway).
- **Looped attention** (10.3) for long prompts, with the MCOLS 8 / MX work.

## 11. Under MX (2026-10-12)

The MX stack (mx-column 761f7ccc, merged into this branch as c0e11bbd: docs/mx_mxu.md) changes
two things the tile leaned on:

- **One Sa a QACT.** QACT gives each row one scale Sa over the blocks it writes (power-of-two
  offsets under it a sub-block), and an MM's rows read the blocks of one Sa (docs/isa.md MM).
  Under the previous numerics a part's quantize_into (a KV head's outputs, an MLP piece, a
  convolution channel part) wrote the bytes one QACT of the whole tile would (an fp32 scale a
  block); under MX each part is a QACT range of its own (Builder.stationary's qranges, the
  merge's fix), so W_o's dot is an MM a range added up with ACC: not today's bits, and slower,
  each range MM reading every W_o row's elements and scale row in short pieces (the MX scale
  reader fetches a row's Sw in a cycle of its own where it shares no line: an MM of KB = 2
  takes about twice its chunks' cycles), and a 4-bit range cannot start at an odd block
  (Phi-4-mini's G = 3 at d = 128).
- **An MM's output is one exact integer**, replayed or paired, so PAIR changes no bit: the
  PAIR exception of 10.2 is gone.

**The tile quantizes each operand over today's columns** (3fd2cc78, 7b0237f8, 5e0eaa08):

- the attention's outputs are quantized at once into W_o's operand (_attention_tile,
  _gemma_attention_tile: `whole`, as _attention_rows' dot(o_all, wo)). Each part's outputs go
  to the image's I/O scratch as the part ends (io "o", ACT_ROWS x n_q d words, the descriptors'
  `oscr`; Gemma 4: its widest heads; a row a store) and come back as one load once the heads
  are done. Gathered in TMEM instead ([T, n_q d] beside the heads' work, the forms with
  `oscr=False`, tried after: no stores, a shorter program) they took 16-32K words, which cost
  rows (Qwen3.5's 16 rows were 12, E2B's 8 were 4) and time (Qwen3-0.6B's 16-row tile: 94.9K
  cycles a row against 86.5K in the range forms);
- the MLP's activation is one QACT a down chunk (_mlp_tile: pieces of `sub` columns into a
  [T, C] tile when sub < C, C the MX chunk of mlp._chunk; mlp_sub counts the tile). Where the
  chunks' activations do not all fit ACT RAM beside the operand (Qwen3.5-27B on one slice: F
  17408, 17 chunks of 8 blocks with the operand's 40, against ACT_BLOCKS 128) they go in
  groups that do (tiled.act_groups: 9 and 8), each group's gate / up MMs and then its down MMs
  into every part's sum, the sums in TMEM across the groups ([T, H] more: fresh MM outputs,
  odd row strides, so their MM ACC has no bank conflict) and added to x after the last: each
  element sums the chunks in today's order, the same bits (test_tiled_mlp_past_act_ram_in_
  groups_is_todays: F 2560 at ACT_BLOCKS 18);
- LFM2's gated product goes whole into out_proj's operand (_conv_tile);
- each with quantize(free=True): a stationary otherwise keeps its source tile in TMEM (the [T,
  C] tiles of every chunk stayed until the down projection: Qwen3-0.6B 8 rows instead of 16).

So **a tiled run is today's bits on the MX ISA**, every family and form, PAIR on or off
(tests/test_tiled.py: the prompt runs of 10.2's families and the expert-major layer runs bit
for bit; test_tiled_prompt_runs_pair_alike: today's 2-row runs pair, the same bits). A dense
model's tiled references equal today's; refs.py still keys OTPU_TILE (a MoE run's range
forms, below, are not today's bits), and OTPU_ACT_ROWS by value (RUNTIME_ENV).

**Placement: next-fit first** (82e150a8). The load back needs [T, n_q d] words in one piece
(Qwen3-0.6B at 16 rows: 32K of the 64K-word TMEM), and two long-lived tiles left mid-TMEM
split it: the attention's q / k norm gains (loaded after x at the next-fit cursor, alive to
the layer's end) and Gemma 4's RoPE rows (loaded after x's gather, whose temporaries had moved
the cursor 26K words on). Next-fit failed there, trace_fit's best-fit fit, and best-fit runs
slower: it reuses the words just freed, which work still running reads. Qwen3-0.6B's 16-row
tile was 94.9K cycles a row placed best-fit, 87.4K next-fit; E2B's 8-row tile 585.3K against
578.8K. Now the gains go when the heads are done, gemma4_tile loads the RoPE rows before x,
and _first_fit tries every form placed next-fit before any best-fit.

**The range forms stay** (`whole=False`, tiled.ranges). A layer run's rows are the engine's
(layer_major), so compile_layer_tile tries them after the whole forms; a prompt run whose
outputs whole do not fit takes fewer rows instead (prefill._probe_tile over variants: whole
forms only). Their accuracy (tests/test_tiled.py, the tiny models' logits after two prompts,
KL divergence from the fp32 forward, reference_logits; ranges / today's):

| model | 1st prompt | 2nd prompt | ranges from today's |
|---|---|---|---|
| Qwen3 (G = 2, 16 rows) | 0.0128 / 0.0126 | 0.0156 / 0.0173 | 0.0028, 0.0033 |
| Phi-4-mini's shape (G = 3, int8) | 1.90e-4 / 1.69e-4 | 2.31e-4 / 2.30e-4 | 5.9e-5, 6.0e-5 |
| Qwen3.5 (16 rows) | 0.0188 / 0.0179 | 0.0177 / 0.0147 | 0.0029, 0.0014 |

The range forms are as close to the fp32 forward as today's runs (each part's Sa is its own
row maximum: a finer scale), only not the same bits; the Gemma 4 MoE's 8-row layer runs in
ranges are 6.3e-5 from its 4-row runs. RTL = ISA in both forms (test_tiled_prompt_run_on_rtl,
test_tiled_expert_major_layer_run_on_rtl).

### 11.1 Rows a tile holds under MX

Compile only, MCOLS 4, ACT_ROWS 16, the card's 64K-word TMEM, fp4 layers, int8 head, the
model's first layers (11.2's), whole forms: the most rows of 16 / 12 / 8 / 4 whose forms fit
TMEM (instructions at depth 3), and the rows the engine takes on the card, whose IMEM holds 4K
instructions (P and L at depth 3, else 2: SHALLOW; prefill._probe_tile's rule at T; d2: at
depth 2; best: only best-fit placement fits). Today's rows: 10.3's.

| model | today | bucket 2: TMEM | bucket 2: card | bucket 8: TMEM | bucket 8: card |
|---|---|---|---|---|---|
| Qwen3-0.6B | 8 | 16 (3,176) | 16 | 16 (7,726) | 12 (d2) |
| Qwen3.5-0.8B | 4 | 16 (4,105) | 12 | 16 (6,441) | 12 (d2) |
| Qwen3.5-2B | 4 | 16 (5,772) | 12 (best) | 16 (8,108) | 8 |
| Qwen3.5-4B | 3 | 8 (4,355) | 4 | 8 (6,691) | 4 |
| LFM2-2.6B | 3 | 12 (6,853) | 4 | 12 (13,675) | 4 (d2) |
| LFM2.5-230M | 8 | 16 (4,453) | 12 | 16 (9,003) | 8 (d2) |
| Phi-4-mini | 3 | 12 (4,798) | 8 | 12 (11,620) | 4 |
| SmolLM3-3B | 4 | 16 (4,267) | 12 | 16 (8,817, best) | 8 (d2) |
| Gemma 4 E2B | 4 | 12 (5,185) | 8 | 12 (7,250) | 8 (d2) |
| Gemma 4 E4B | 4 | 8 (2,364) | 8 | 8 (2,556) | 8 |

**On the card IMEM, not TMEM, sets the rows of most models**: through the I/O scratch the
whole forms fit TMEM at as many rows as before MX (10.3) or more, but a packed tile's unrolled
attention makes their programs longer than 4K instructions at 12-16 rows (10.3's long-prompt
note, now at bucket 2 too). Where that leaves 4 rows (Qwen3.5-4B, LFM2-2.6B) the gain is that
of 10.4's T = 4 column, within 0.4% of 8 rows before MX. The looped attention (10.3) is what
gives the card TMEM's rows.

Expert-major layer runs (MCOLS 4, ACT_ROWS 16, OTPU_TILE=1; every program within IMEM; the
form the search took):

| model | layer | 4 rows | 8 rows |
|---|---|---|---|
| gemma-4-26B-A4B (int8) | sliding | whole | whole |
| | global | whole | ranges (x spilled) |
| Qwen3.5-35B-A3B (fp4) | DeltaNet | (no W_o) | (no W_o) |
| | attention | whole | whole (x spilled) |

With the I/O scratch only the 26B's global layers at 8 rows ([8, 16 x 512] outputs and the
heads' work beside them) take the range forms.

### 11.2 Measured (RTL co-simulation, MX): cycles and DRAM bytes a row

Prompt runs, as 10.4's (tools/perf_prompt.py --phases: one P run at the start of bucket 2, the
model's first layers, fp4 layers, int8 head; MCOLS 4, ACT_ROWS 16, the card's memory path), on
the shipping tree (mx-column 8b33206e with the default: the MX RTL with the card's 4 epilogue
lanes, MXU_E 4). Today's: the MX build's prompt runs at the rows they take (3-row runs pair at
2, docs/prefill.md 7); the tile's: the whole forms through the I/O scratch, placed next-fit, at
the rows the engine takes on the card (11.1). Cycles a row, and the DRAM bytes a row on port B
(the weights' stream, K / V, every LD and ST: the profile's port-B transfers x D = 128 bytes;
port A's scale words not counted):

| model (layers) | today: cycles (rows) | tile: cycles (rows) | cycles | today: MB a row | tile: MB a row | bytes |
|---|---|---|---|---|---|---|
| Qwen3-0.6B (2) | 89.8K (8) | 82.6K (16) | -8.0% | 4.14 | 2.12 | -49% |
| Phi-4-mini (2) | 483.3K (2) | 433.1K (8) | -10.4% | 52.7 | 15.0 | -72% |
| Qwen3.5-4B (4) | 1,133.0K (2) | 1,007.4K (4) | -11.1% | 125.9 | 69.9 | -44% |
| LFM2-2.6B (6) | 1,109.0K (2) | 996.0K (4) | -10.2% | 123.0 | 62.4 | -49% |
| Qwen3.5-0.8B (4) | 285.0K (4) | 286.3K (12) | +0.5% | 17.4 | 13.1 | -24% |
| LFM2.5-230M (6) | 170.4K (8) | 164.7K (12) | -3.4% | 6.06 | 3.80 | -37% |
| SmolLM3-3B (4) | 661.4K (4) | 655.1K (12) | -0.9% | 41.4 | 15.4 | -63% |
| Gemma 4 E2B (5) | 580.7K (4) | 574.1K (8) | -1.1% | 27.1 | 15.3 | -44% |

By level (dataflow's ledger: tools/dataflow.py at 46609834 on branch dataflow, the ISA
simulator on a zero image, the same runs; a row, over the model's first layers as above; DRAM
reads and writes, the K / V re-stream its attention's part; TMEM and ACT RAM as words x 4 and
bytes; the scratch: the outputs' ST and LD at tiled.py's sites):

| model | DRAM a row: today -> tile | K / V re-stream | TMEM read | TMEM written | ACT RAM written | scratch round trip |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | 4.61 -> 2.49 MB (-46%) | 2.20 -> 1.11 MB | +3% | -1% | -37% | 32 KB (1.3% of DRAM) |
| Phi-4-mini | 56.5 -> 16.2 MB (-71%) | 2.21 -> 2.21 MB | +2% | 0% | -30% | 48 KB (0.3%) |
| Qwen3.5-4B | 134.1 -> 74.1 MB (-45%) | 1.09 -> 1.09 MB | -3% | -4% | -42% | 32 KB (0.04%) |
| LFM2-2.6B | 132.1 -> 67.0 MB (-49%) | 1.70 -> 1.70 MB | -2% | -2% | -42% | 32 KB (0.05%) |
| Qwen3.5-0.8B | 18.4 -> 13.7 MB (-25%) | 0.54 -> 0.54 MB | 0% | 0% | 0% | 16 KB (0.1%) |
| LFM2.5-230M | 6.62 -> 4.21 MB (-36%) | 1.67 -> 0.85 MB | 0% | -2% | -27% | 16 KB (0.4%) |
| SmolLM3-3B | 45.1 -> 16.8 MB (-63%) | 2.23 -> 2.23 MB | +2% | +1% | 0% | 64 KB (0.4%) |
| Gemma 4 E2B | 29.2 -> 16.5 MB (-44%) | 3.24 -> 3.24 MB | +6% | +3% | -4% | 101 KB (0.6%) |

The weights' stream is what the tile cuts (each weight read once for T rows); the K / V
re-stream halves where G <= MCOLS / 2 packs two rows a stream (Qwen3-0.6B, LFM2.5-230M) and is
unchanged elsewhere; TMEM traffic a row stays within -4% to +6% (the scratch's round trip and
the staging of the whole operands against fewer per-run tiles); the ACT RAM's writes a row
fall by up to 42% where today's runs pair: PAIR's DUP writes each row of a 4-bit layer's
operand twice at M <= MCOLS / 2 (Qwen3.5-4B, 2 rows: 187 KB a row, 109 KB with PAIR off, the
same as the 4-row tile's; dataflow's reading, checked); its reads (the stationary for every
weight chunk) are the same a row.

- **Bytes first.** The tile streams each weight once for its T rows: a row's DRAM traffic falls
  by a quarter to three quarters on every model, cycles by 8-11% where today's runs hold 2 rows
  (the 4-bit layers' MXU knee at MCOLS 4 rows) or Qwen3-0.6B's G = 2 packs two rows a K / V
  stream, and by 1-3% where the MXU, not the weights' stream, sets the time (G >= MCOLS,
  DeltaNet). Qwen3.5-0.8B is the one +0.5% (DeltaNet-bound, 12 rows in runs of 8); its bytes
  still fall 24%.
- **The epilogue lanes matter to the tile.** Before the E 4 merge (2 lanes, the same tree
  otherwise) Qwen3-0.6B's 16-row tile was 87.4K (-3.0% against 90.1K), now 82.6K; the 3-row
  models' gains were within 0.5 points of these (Phi-4-mini 443.5K, -10.1%; Qwen3.5-4B 8 rows
  1,006.4K; LFM2-2.6B 8 rows 997.6K). Before MX (10.4, the 2-lane MXU) the tile's gains were
  the same: -9.7% to -10.7% on the 3-row models, -8.7% on Qwen3-0.6B.
- **The I/O scratch's round trip**: a whole form stores the attention's outputs (fp32) a part
  at a time and loads them back whole, 2 x 4 x n_q d bytes a row a layer through DRAM and the
  same through TMEM: 16 KiB on Qwen3-0.6B (n_q d = 2,048), 24 KiB on Phi-4-mini, 32 KiB on
  E2B's global layers; the ledger finds exactly that, 0.04-1.3% of the tile's DRAM bytes a
  row (Qwen3-0.6B's 1.3% against ~0.5 MB of fp4 weights a row a layer at 16 rows).
  It buys today's bits (one Sa over W_o's operand); the range forms, which skip it, are within
  ~1% in cycles (Qwen3-0.6B at 16 rows 86.5K against 87.4K, 2 lanes; Qwen3.5-0.8B 285.0K,
  SmolLM3 666.8K, E2B at 12 rows 573.6K, Qwen3.5-4B at 12 1,011.2K) and as close to the fp32
  forward as today's (11), not the same bits.
- **More rows is not always cheaper** (SmolLM3: 12 rows 664.1K against 16 668.2K, 2 lanes);
  the engine takes the largest T that fits (a cost-aware T: 10.4's note on Gemma 4).

### 11.3 On by default

The tiled prompt runs are the default with the MX numerics (0a50b82d, 7f92a846):
prefill.tile_flag takes OTPU_TILE unset (or empty) as on, OTPU_TILE=0 off, and an engine's
tile_runs True / False first; one slice only (a two-slice configuration keeps today's runs:
prefill.tiling). tests/test_prefill.py and test_prefill_gemma4.py test today's prompt runs
(OTPU_TILE=0 there), tests/test_tiled.py the tiled ones against them (MTP's drafter after tiled
runs too). An engine with prompt runs then grows its image's I/O rows to ACT_ROWS (16 on the MX
target) and a bucket's runs are the tile's where one fits it, today's elsewhere; the bits are
today's. A MoE model's tiled expert-major layer runs stay opt-in (prefill.layer_tile_flag:
tile_runs True, or OTPU_TILE set and not 0): they go with the expert-major switch, which ships
on its own, and without them a MoE image holds no mask tiles (its layout and expert slots as
before). tools/qual/refs.py keys OTPU_TILE as it is set: the default's references are the card's
default's; OTPU_TILE=0 and =1 have keys of their own.

### 11.4 Next

- **Card, the MX build (ACT_ROWS 16):** the dense models' qual with the
  default, token-exact against today's (the same bits), and the prompt runs' gain where 11.2
  shows it: Phi-4-mini, LFM2-2.6B and Qwen3.5-4B -10 to -11% a row, Qwen3-0.6B -8%, LFM2.5-230M
  -3%; then, with the expert-major switch, the 26B's expert-major prefill with tiled layer
  runs (10.6's A/B; its global layers' 8-row runs may take the range forms: chatmoe's tokens
  checked against today's, not assumed).
- **IMEM** holds most models to fewer rows than TMEM on the card (11.1: Qwen3.5-4B and
  LFM2-2.6B at 4); the looped attention (10.3) lifts it.
- **A cost-aware T** (Qwen3.5-0.8B +0.5% at 12 rows; SmolLM3 12 rows cheaper than 16).
- **The scratch's round trip** (11.2) is the price of one Sa over W_o's operand; dataflow's
  epilogue fold (the row scale applied in the MXU's epilogue lanes) may let the parts be
  quantized as they end without it.

## 12. The card's routes, and two loops that lift IMEM (2026-10-13)

### 12.1 What the card took

`tools/prefill_routes.py` (90b17fd7) runs prefill.py's own rules on a weightless image of the
whole model as otpu-chat makes it (resident, prompt runs, the image's rows grown to ACT_ROWS, the
model's formats: otpu-chat's auto) on the card's configuration (m4: MCOLS 4, ACT_ROWS 16, 64K
TMEM words, 4K instructions of IMEM; PAIR, DSTEP, STREAM, SEQ_WC), bucket by bucket over a 2K
context: the tile's T (_tile_fit), today's prompt runs' R and whether they cover the bucket
(_fit), and the form prefill.tiled takes. Release r2's tree (the simulation host); t: tiled, p: today's
prompt runs, buckets 1-8:

| model | rows a run |
|---|---|
| Qwen3-0.6B | 16t 16t 16t 12t 12t 12t 8t 12t |
| Qwen3.5-0.8B, Qwen3.5-2B | 4t everywhere |
| Qwen3.5-4B | 4t (1-4), 2t (5-8) |
| LFM2-2.6B | 4p, 2p (2-4), 1p (5-8) |
| LFM2.5-230M | 16t, 12t (2-3), 8t (4-8) |
| SmolLM3-3B | 8t (1-2), 4t (3-8) |
| Phi-4-mini | 4t 3t 3p 2t 2p 1t 1t 1t |
| Gemma 4 E2B | 4t everywhere |
| Gemma 4 E4B | 3t 2t 2t 2t 2p 2p 2p 1t |

IMEM, not TMEM, set these on most models (11.1 compiled the first layers; the whole models' tile
programs are longer): it unrolls each layer of an irregular stack (LFM2-2.6B's 30 layers: 7,250
instructions at 4 rows in bucket 2, 14,076 in bucket 8) and each kind's packed attention (rows x
KV heads x blocks), which grows with the context. Co-simulated (tools/perf_prompt.py --phases, the
card's memory path, the models' first layers, otpu-chat's formats), cycles a row:

- LFM2-2.6B (6 layers): bucket 5, today's 1-row runs 2,402.8K against a 4-row tile 1,039.9K
  (-56.7%); bucket 2, today's 2 rows 1,212.6K against 4 rows 1,007.7K (-16.9%).
- Qwen3.5-4B (4 layers), bucket 5: the tile's 2 rows 1,272.9K against 4 rows 1,029.6K (-19.1%).

The tile's rows at other IMEM sizes (--imem; TMEM and ACT RAM the card's): data for the IMEM
question, not a proposal (buckets in parentheses where they differ):

| model | 4K instructions (the card) | 8K | 16K | unbounded |
|---|---|---|---|---|
| Qwen3-0.6B | 16 (1-3), 12, 8 (7) | 16 | 16 | 16 |
| Qwen3.5-0.8B | 4 | 12 | 16 | 16 |
| Qwen3.5-4B | 4 (1-4), 2 | 8 (1-2), 4 | 8 | 8 |
| LFM2-2.6B | none: today's 4, 2, 1 | 4 (1-2), 2 | 8 (1-3), 4 | 12 |
| LFM2.5-230M | 16 (1), 12 (2-3), 8 | 16 | 16 | 16 |
| SmolLM3-3B | 8 (1-2), 4 | 16 (1-2), 12, 8 (5) | 16 | 16 |
| Gemma 4 E2B | 4 | 8 (1-4), 4 | 12 | 12 |

### 12.2 Qwen3.5: a unit's DeltaNet layers in a loop (0fff0c44)

Qwen3.5's stack is (3 DeltaNet layers, 1 attention layer) x n: plan's one loop over the unit with
its three DeltaNet layers unrolled. lfm2.subs plans each run's unit again and run_layers(nest)
loops its repeated part inside the unit's loop (the part's blocks at jt x its bytes), as
gemma4's subs do its sliding layers: one copy of a DeltaNet layer in the program instead of
three (qwen35.NEST: qwen35_rows and tiled.qwen35_tile; MTP's verify rows, decode-time programs,
stay unrolled). The same arithmetic in the same order, so today's bits by construction
(test_qwen35_nested_layers_are_the_unrolled_ones, tiled and today's prompt runs; the RTL cases of
test_tiled_prompt_run_on_rtl). Loop depth: the unit, its part, the layer's own.

Cycles, nested against unrolled at equal rows (perf_prompt, 8 layers: two units): Qwen3.5-4B's
4-row tile in bucket 5 8,197,653 / 8,198,988 (-0.02%), Qwen3.5-0.8B's 12-row tile 7,100,076 /
7,097,265 (+0.04%), the 4B's 2-row prompt run in bucket 2 5,000,603 / 5,001,703 (-0.02%): the
joins at the inner loop's exit cost nothing measurable.

### 12.3 LFM2: the stack in one loop

LFM2-2.6B's 30 layers (ccA ccA cccA x 4 ccA ccA cc) repeat no unit long enough for plan, so its
tile unrolled nearly every layer. tiled._lfm2_stack runs a dense stack of both kinds in one
formats group (lfm2.Image.stack) as one hardware loop over its layers, whatever the pattern:

- a TMEM table, a word per layer and kind (1.0 where the layer is of that kind: ol.zeros and a
  FILL per run of layers of a kind at the start), RLD (bounded: 0 or 1) into two counts;
- a loop of 0 or 1 iterations around each kind's mixer (LOOP runs R[ra] + w2 times and skips its
  body at 0), then the MLP: LFM2's blocks start with the common part (norms, MLP) at the same
  offsets in both kinds;
- the layer's block from a compiler.Counter, a register the program steps itself
  (Builder.counter / step / end_counter): 0 before the layer loop, ADDI of the kind's block size
  in a loop of 0 or 1 iterations after the MLP, freed after the loop. An address adds 1 x the
  counter and with it at most one run-time value of the host's (the KV rows at tpos, a conv
  layer's ring row): a register of their sum, which starts at c x var before the layer loop
  (an RLD MUL of the word, bounded by the variable's bound) and steps with the counter
  (Builder._derived). The run-word fix f1c025a1 is what makes the ring word one the host writes
  before each run: it is read only through such a register.

The image is today's: its blocks lie one after the other in layer order, each its kind's size,
with no padding (a uniform stride would add 110.7 MB, 4.2%, to LFM2-2.6B's 2,664 MB) and no
change to the image, its key or today's programs. Blocks grouped by kind (a region a kind, a
counter each, the common parts in a region of their own) were tried first: three strides in
every plan loop, and today's prompt runs and the decode loop, which already take 14 of the 15
address registers, ran out of them ("generate b1: R10 is taken, argument 5 needs it").

Waits (opentpu.waits): the analysis runs the layer loop's iterations with the counter as an
interval, at the counts' 0 / 1 loops the join of both outcomes, so layer i's addresses span the
blocks from the stack's base to i times a conv and an attention block: a run's writes (KV rows,
conv rings) and the weight streams after them are conservative over that window, inside the
layers' region and the head and tables after it (the I/O area, which the runs write, lies before
the base).
They cost nothing measurable (below). Loop depth: the layers, the 0 / 1 loop, the layer's own (3;
4 where an attention layer's block loop nests: the tile's depth-2 forms take bucket 8).

Bits: today's (test_lfm2_stack_in_one_loop_is_the_plans_runs: an irregular 7-layer stack, two
prompts, the logits and every slice's DRAM bit for bit against the plan's runs, the wait counts
checked against every run, OTPU_WAITS_CHECK; the tiny LFM2's tiled-against-today's and RTL cases,
whose (conv, attn) x 2 + conv now runs this way; test_a_counter_steps_where_the_program_says).
Programs, LFM2-2.6B on the card, 4 rows, P / L: bucket 1 1,604 / 1,626 instructions (the plan's
6,194 / 6,216), bucket 5 3,012 / 3,034 (10,418 / 10,440), bucket 8 2,824 / 2,846 (14,076 /
14,098): test_lfm2_tiles_fit_the_cards_imem. Cycles, the stack against the plan's runs at equal
rows (perf_prompt, LFM2-2.6B, the card): 6 layers, 4 rows in bucket 5 4,158,517 / 4,159,466
(-0.02%), 8 rows in bucket 2 8,003,181 / 8,007,873 (-0.06%); 12 layers, 4 rows in
bucket 5 8,227,065 / 8,229,002 (-0.02%) and bucket 8 8,415,530 / 8,417,819 (-0.03%; 30 layers
overflow the co-simulation's DRAM model); LFM2.5-230M (6 layers), 12 rows in bucket 2 2,054,026 /
2,054,702 (-0.03%); port B's bytes the same. The prototype on a uniform layout (every block padded to the larger kind's)
measured the same: 4,158,523 and 8,003,814.

Rows a run on the card now (prefill_routes; before, in parentheses):

| model | buckets 1-8 |
|---|---|
| Qwen3.5-0.8B | 16t, 12t (2-8) (4t) |
| Qwen3.5-2B | 8t (4t) |
| Qwen3.5-4B | 8t, 4t (2-8) (4t, then 2t from 5) |
| LFM2-2.6B | 12t, 8t (2-3), 4t (4-8) (4p, 2p, 1p) |
| LFM2.5-230M | 16t (1-3), 12t 8t 12t 8t 12t (16t, 12t, 8t) |

### 12.4 Ties

prefill.tiled took the tile where its T equals today's R. At equal rows the tile was the slower
on every production dense model under the card's formats (co-sim as 12.1, cycles a row, tile
against today's prompt runs; port B's bytes equal within 0.2%):

| model | rows | bucket | tile against today's |
|---|---|---|---|
| Qwen3.5-0.8B | 4 | 2 / 5 | +0.50% / +0.48% |
| Qwen3.5-4B | 2 | 5 | +0.36% |
| Gemma 4 E2B | 4 | 2 / 5 | +1.24% / +1.08% |
| SmolLM3-3B | 4 | 5 | +0.63% |
| Phi-4-mini | 3 / 2 / 1 | 2 / 4 / 7 | +0.83% / +0.96% / +0.59% |
| Gemma 4 E4B | 2 / 1 | 3 / 8 | +0.60% / +0.53% |

So a tie takes today's prompt runs (prefill.tiled: tiled unless today's runs cover the bucket
with R >= T; test_a_bucket_takes_the_form_with_more_rows_a_run). With 12.2 Qwen3.5's ties are
gone (the tile takes more rows); E2B (buckets 1-6), SmolLM3 (3-6), Phi-4-mini (2, 4, 6-8) and E4B
(2-4, 8) now run today's runs there, at the same rows. A tile of the probe's largest rows is
still taken unprobed (no production model's today's runs reach it).

### 12.5 Next

- **Phi-4-mini and E4B's 1-row runs** (from bucket 5-6): one kind of layer, so neither loop above
  helps; their packed attention's entries are what fills IMEM. The looped attention
  (docs/prefill.md 10: an entry loop over a KV head's packed tiles) is the next step, with no
  IMEM growth: done in 13.

## 13. The looped attention (2026-10-14)

12.5's next step. A tiled run's attention was unrolled, one entry for each KV head, query part
and packed tile (rows x heads x blocks), and past the first buckets that filled IMEM: Phi-4-mini's
whole-model tile took 3,666 instructions at 4 rows in bucket 1 and 11,552 in bucket 8, so on the
card it ran 1 to 3 rows a run. tiled._heads_loop runs it in hardware loops instead:

- **K and V first.** Every KV head's K and V are projected and appended first, unrolled. These
  are their run-time rows: in a loop each would take another address register (the K row, its
  scales, the V^T column, V's scale).
- **A loop over the KV heads.** Their cache blocks are evenly spaced, so K, its scales, V^T and
  V's scales sit at head 0's address plus j steps, in one register. In the loop, each part of
  the head's query group is unrolled (one part for G <= MCOLS) and projected into one TMEM
  buffer, with W_q's rows at the loop's index. Then comes a loop over the part's packed tiles of
  P rows. A partial last tile follows the loop, unrolled, with its own operand, state and score
  buffers.
- **The loop body is one packed tile's attention.** It has the same instructions as the
  unrolled entry (attention._scores and _finish, now module functions: the tile's blocks, its
  masked blocks' row tiles, acc and l), so it gives the same bits, at addresses affine in the
  loops' indices. Two TMEM operands take a loop's register, and both share one register:
  - the tile's queries (compiler.Builder.quantize_into's `at`, QACT's src);
  - its outputs, which are written over the queries (.vop's `at`, VOP's dst).

  The mask tiles and the part's outputs (to their columns of the I/O scratch) are DRAM
  addresses as before.
- **A pipeline across a part's tiles.** A tile's body ends with the next tile's queries
  quantized (into the same ACT blocks, once the tile's last scores are issued) and that tile's
  first depth - 1 blocks' scores in flight, in buffers of their own, so every iteration
  addresses the same buffers. The last tile primes a phantom tile that is never used. That
  prime reads only the program's own words: the query buffer's zeroed tile past T, and the mask
  tiles of the table's padding. The padding (qwen3.AMASK_PAD, amask_bytes) is MCOLS positions
  after the mask table: 16 KB at MCOLS 4 (64 KB at 8). It moves everything after the table,
  so every image's layout and key change. tiled.PEEL runs the last tile after the loop
  instead. The loop drains between parts and between heads.
- **Loop depth.** The levels are the layers' loop(s), the heads and the tiles. The blocks inside
  a tile's body stay unrolled, about 10 instructions a block. LFM2's stack uses all 4: layers,
  the kind's 0/1 loop, heads, tiles. Its 0/1 loops' count registers (12.3) are now taken
  before each loop and freed in its body, so the attention's loops can use them.
- **Fallback levels** (tiled.LOOPS, _first_fit). Each form is traced with the heads' and the
  tiles' loops first. If it runs out of address registers, it is traced with the tiles' loop
  alone, then unrolled.

Where it applies (_loopable): prompt runs at run-time positions with the image's mask tiles, the
outputs whole through the I/O scratch, whole query parts, one slice, and whole blocks in a
tile's span. Qwen3.5's gated attention stays unrolled, because its gates multiply the part's
outputs after its last tile.

### 13.1 Bits

- **Looped against unrolled.** test_looped_attention_is_the_unrolled_one compares the looped
  runs with the same runs unrolled (tiled.LOOP off): two prompts across a bucket's end, the
  last logits and every slice's DRAM, bit for bit. The wait counts are checked against every
  run (OTPU_WAITS_CHECK), and every MM's drain is checked to keep up. Cases:
  - tiny Phi-3 (G = 3, a row a tile) at MCOLS 4 and 8 (a partial last tile);
  - Qwen3 (a partial 7th tile; the last tile peeled);
  - SmolLM3 (layers without RoPE);
  - LFM2 (64-wide heads, the stack's loops);
  - Qwen3.5 kh8, unrolled.
- **Gemma 4.** test_tiled_gemma4_prompt_runs_are_todays now runs looped, at 16 rows where it
  took 8-12.
- **On the RTL.** The RTL prompt runs (test_tiled_prompt_run_on_rtl: Qwen3, LFM2, Gemma 4) are
  looped. The Gemma 4 runs now take 16 rows; before, IMEM held them to 4 and 8.
- **IMEM fit at real dims.** test_looped_tiles_fit_the_cards_imem builds the real dims as a
  weightless layout on the card's m4 with otpu-chat's formats. Phi-4-mini fits 12 rows in every
  bucket of a 2K context (P and L). Gemma 4 E4B fits 4 rows in buckets 1-7 and 3 in bucket 8.
- **Unrolled programs unchanged.** With the loop off and no padding (AMASK_PAD 0), the unrolled
  programs are word for word the previous tree's on Phi-4-mini, Qwen3-0.6B, SmolLM3, E2B and
  Qwen3.5-0.8B. LFM2's differ only by the count registers.

### 13.2 Programs

Instructions of the whole model's tile program: P runs (L adds the LM head's rows: +56 on
Phi-4-mini, +70 on Gemma 4), the card's m4, otpu-chat's formats.

| model | rows | bucket 1: looped / unrolled | bucket 8: looped / unrolled |
|---|---|---|---|
| Phi-4-mini | 4 | 1,432 / 3,666 | 1,662 / 11,552 |
| Phi-4-mini | 12 | 3,634 / 9,684 | 3,864 / 33,314 |
| Gemma 4 E4B | 4 | 3,741 / 4,734 | 4,044 / 7,317 |
| Gemma 4 E2B | 4 | 2,312 / 2,701 | 2,716 / 4,422 |
| Gemma 4 E2B | 8 | 5,208 / 6,117 | 5,912 / 9,614 |

The looped program grows by 33-58 instructions a bucket: each attention body gains a block. What
is left is the layers' other work:

- **E4B:** its MLP alone is 1,200 instructions. Its L program at 4 rows is 4,114 in bucket 8,
  18 over IMEM, so that bucket takes 3 rows. Accepted (2026-10-14): no Gemma layer
  loops and no IMEM change for it.
- **E2B:** at 8 rows it is over 4K in every bucket.
- **LFM2-2.6B** (the stack's loops and the heads' loop): 677-753 instructions at 4 rows,
  1,131-1,207 at 8.

### 13.3 Cycles

Phi-4-mini (2 layers; the card's memory path, otpu-chat's formats; tools/perf_prompt.py
--phases, the tile at forced rows), cycles a row, instructions in parentheses:

| bucket | rows | looped | unrolled | last tile peeled |
|---|---|---|---|---|
| 1 | 4 | 456.5K (660) | -1.00% (1,400) | -0.25% (675) |
| 4 | 4 | 480.5K (692) | +0.17% (2,456) | -0.28% (741) |
| 8 | 4 | 519.6K (736) | -0.11% (4,030) | -0.43% (829) |
| 1 | 8 | 439.3K (1,386) | -1.25% (2,762) | -0.14% (1,401) |
| 4 | 8 | 462.3K (1,418) | +0.13% (4,874) | -0.23% (1,467) |
| 8 | 8 | 501.0K (1,462) | +0.08% (8,016) | -0.16% (1,555) |
| 8 | 12 | 501.2K (1,806) | -0.28% (11,620) | |

- **The loop's cost.** From bucket 4 the loop is within 0.3% of the unrolled form. In bucket 1
  it is 1.0-1.25% slower, not yet explained. One block a tile leaves less to overlap: each
  tile's scores, softmax and PV, and each head's drain, sit between iterations. That has not
  been measured apart. Accepted (2026-10-14): a route special case for bucket 1 is
  not worth it against the gain below.
- **The phantom prime** costs 0.14-0.43%: its first blocks' scores, with 4.5-8.2 KB of K on
  port B a run. That is under the 0.5% bar, so the loop keeps the prime and does not peel; a
  peel costs a tile body of instructions. At 12 rows the prime's share is a third of its share
  at 4 rows.
- **The gain.** Against today's 1-row prompt runs (Phi-4-mini's buckets 6-8 after the tie
  rule), cycles a row:

| bucket | today's 1 row | looped 4 rows | looped 12 rows |
|---|---|---|---|
| 6 | 1,725.1K | 500.9K (-71.0%) | 482.6K (-72.0%) |
| 7 | 1,733.9K | 510.2K (-70.6%) | 491.8K (-71.6%) |
| 8 | 1,743.1K | 519.6K (-70.2%) | 501.2K (-71.2%) |

### 13.4 Routes

prefill_routes (as 12.1, the card's m4, a 2K context, the tie rule of 12.4), rows a run, buckets
1-8 (t: tiled, p: today's prompt runs). "Before" is this branch before the loop (12.1 with
12.2-12.4):

| model | now | before |
|---|---|---|
| Qwen3-0.6B | 16t | 16t 16t 16t 12t 12t 12t 8t 12t |
| Qwen3.5-0.8B | 16t, 12t (2-8) | the same |
| Qwen3.5-2B | 8t | the same |
| Qwen3.5-4B | 8t, 4t (2-8) | the same (gated attention: unrolled) |
| LFM2-2.6B | 12t | 12t, 8t (2-3), 4t (4-8) |
| LFM2.5-230M | 16t | 16t (1-3), 12t 8t 12t 8t 12t |
| SmolLM3-3B | 16t | 8t 8t, 4p (3-6), 4t (7-8) |
| Phi-4-mini | 12t | 4t, 3p (2-3), 2p (4-5), 1p (6-8) |
| Gemma 4 E2B | 4p (1-6), 4t (7-8) | the same (8 rows: over 4K) |
| Gemma 4 E4B | 4t (1-7), 3t | 3t, 2p (2-7), 1p |

### 13.5 Next

- **The card's qual** comes with the next host batch, which the maintainer schedules: token-exact
  against today's runs. The prompt runs' gain on Phi-4-mini, SmolLM3, LFM2-2.6B, E4B and
  Qwen3-0.6B's later buckets is to be measured there.
- **Qwen3.5's gated attention, looped.** The gates would be applied in the part's store after
  the tiles. Qwen3.5-4B is IMEM-bound at 4 rows from bucket 2, while TMEM holds 8 (11.1).
- **Gemma's other work.** E4B's bucket 8 and E2B at 8 rows are bound by their layers' other
  work, not the attention. The levers are the same loops over their layers that 12.2-12.3 gave
  Qwen3.5 and LFM2, or an 8K IMEM (docs/infra_hunt.md, parked until the looped attention was
  costed).

## 14. MoE expert-major runs in tiles (2026-10-14)

A Qwen3.5-MoE prompt (MiniCPM-V's language model and production's qwen35-35b, the same backbone)
runs layer by layer, and in a call of EM_ROWS rows or more expert-major ([offload.md](offload.md)
13.11-13.18). Each layer's mixer runs over the chunk's rows R = 2 at a time (`layer_major`),
then one expert run takes every row's experts. The mixer's weights stream once a run: at R = 2 a
DeltaNet layer's projections are read once for every two rows of the prompt.

The expert-major calls' layer runs now go in tiles of T rows (`tiled.compile_layer_tile`,
sections 10-11), on by default as the dense models' prompt tiles are:
- A Qwen3.5-MoE image holds the mask tiles and the I/O scratch unless `OTPU_TILE=0` as it is
  built (`prefill.layer_tile_flag`, the family's default: `tiled.LAYER_ROWS`), and its engine
  runs tiles of `tiled.LAYER_ROWS` = 8 rows, at most ACT_ROWS (`Engine.tile_rows`; otpu-chat's,
  otpu-serve's and chatmoe's engines through `streamed.engine`); `chatmoe.py --tile-rows T`
  overrides T. With `OTPU_TILE=0` the image is today's (the same expert slots) and the runs are
  today's. Gemma 4's tiled layer runs stay opt-in (`OTPU_TILE=1`): its global layers take range
  forms at 8 rows.
- The image's extra 1.25 MiB costs no expert slot at otpu-chat's and otpu-serve's 2048-token
  cache (qwen35-35b 41, MiniCPM-V 36 a layer, `streamed.fit_experts` on the card's 4 GiB); at
  4096 tokens qwen35-35b has 40 instead of 41.
- The calls gate_expert_major runs layer-major (a chat turn's short ones) keep today's runs of R
  rows, untiled. Before, `tile_runs` tiled them too, at R rows: today's run in a slower form.
- Layer 0's runs that gather the host's rows, and the image rows from the vision side, stay
  today's runs of R rows.

### 14.1 Bits

Under MX a tile in its whole forms gives today's bits (section 11): a 4-bit MM's output is one
exact integer whether it pairs or not, so the fp4 and nvfp4 layers keep their bits as the int8
ones do. On the card's configuration (MCOLS 4, ACT_ROWS 16) compile_layer_tile takes whole forms
for every 35B layer kind at T = 8 in buckets 2, 8 and 16 (positions 256, 2047 and 4095 of a
4096-token cache; the attention's program 1,980, 2,926 and 3,982 instructions of IMEM's 4K). At
T = 16 the attention layers fall to range forms, which are not today's bits and are slower
(0.791 against 0.710 ms a row, below). So T = 8.

- `tests/test_em_tile.py`: a tiny Qwen3.5-MoE in the 35B's shape (a DeltaNet pair sharing its q
  and k), a 43-row call expert-major in tiles of 8 rows and a 5-row call layer-major, against the
  same calls in today's runs of R = 2: the logits, the layers' blocks and two decode steps after
  them, bit for bit, in int8 and in fp4. A default engine runs tiles of 8 rows (4 at ACT_ROWS
  4), and one made with `OTPU_TILE=0` today's runs of 2 rows on an image without the mask tiles,
  with the default's logits, blocks and decode steps bit for bit.
- `tests/test_minicpmv.py` test_image_rows_run_in_expert_major_tiles: MiniCPM-V's image (Qwen3.5's,
  subclassed: `tiled.layer_family` by the class's bases) with 20 image rows from vis among 31 text
  rows, tiles of 8 rows against today's runs: logits and three decode steps bit for bit.
- Tokens against today's runs, on the ISA simulator with the card's configuration (b10245f5),
  `chatmoe.py --check` (the simulation host, 2026-10-14): qwen35-35b, a 260-token text prompt (its 256-row
  call expert-major in 1,418 runs) and 16 decoded tokens; MiniCPM-V's two image requests of session
  hostr21, thumb (88 rows expert-major in 548 runs, and a second turn of 23 rows layer-major; 16
  tokens each) and sliced (488 rows in 2,893 runs, against today's ~10,000; 32 tokens). The same
  tokens, all three. No dKL run: the tiles take whole forms only, today's bits.

### 14.2 Co-simulated

One layer at position 256, the card's configuration and memory path (b10245f5: PAIR, DSTEP,
STREAM, SEQ_WC; 133.33 MHz, DDR3-1066), ms a row:

| layer | formats | today's R = 2 | tile T = 4 | tile T = 8 |
|---|---|---:|---:|---:|
| DeltaNet, layers 0-19 | MiniCPM-V: int8 | 1.541 | 1.028 | 1.027 |
| DeltaNet, layers 20-39 | MiniCPM-V: nvfp4, shared expert int8 | 1.111 | 0.980 | 0.975 |
| attention | MiniCPM-V: nvfp4 | 0.784 | 0.716 | 0.710 |
| DeltaNet | qwen35-35b: nvfp4, shared expert nvfp4 | 1.070 | 0.940 | 0.936 |
| attention | qwen35-35b: nvfp4 | 0.742 | 0.675 | 0.670 |

Over the 40 layers (30 DeltaNet, 10 attention) the mixer takes 47.6 -> 37.1 ms a row on MiniCPM-V
(-22%) and 39.5 -> 34.8 on qwen35-35b (-12%). The int8 DeltaNet layers gain the most: their
projections are nearly twice the bytes of the 4-bit ones'. MiniCPM-V's image request of session
hostr21 (488 prompt rows; its mixer runs 23.2 s on the card, the co-sim's 47.6 ms x 488) would take
about 18.1 s: -5.1 s of its 90.4 s.

### 14.3 Next

- The card A/B (chatmoe.py --tile-rows 8 against today's) in the next host batch.
- The expert run after the tiles. Co-simulated (Qwen3.5-35B-A3B's layer 3 at MiniCPM-V's 488
  rows: 210 experts used, 3,904 row-expert entries in 1,054 passes, 59 slots), it takes 238.1 ms
  against a DRAM bound of 130.4 ms for its 1,721 MiB of experts: MXU-bound, at 1.8x its bytes
  (the passes' MMs 78% of its 31.7 M cycles; the MXU busy 27.2 M). r2.1's MCOLS 8 halves those
  MMs' cycles, so it should cut the expert runs a lot, toward their bytes: to check in the r2.1
  window's chatmoe.
- The tiles at MCOLS 8 (r2.1). Co-simulated on the m8 variant (docs/dataflow.md 4.9.1), the
  int8 DeltaNet layer at T = 8 did not move (1.027 -> 1.020 ms a row; 0.722 both without DSTEP's
  cost), while the nvfp4 one's other work did (0.720 -> 0.509) and the attention layer (0.710 ->
  0.488). The tile program was why: a DeltaNet layer's rows ran in pieces of 4 (`layer_variants`'
  drows: the image's gr / on rows, `grows`, RUN_ROWS 4), and the MoE prologue (the router and the
  shared expert, `moe_prologue_rows`) in groups of 2 rows (a need line's LINE / 4 / k ids). So an
  8-row tile streamed the DeltaNet projections twice and the shared expert and the router four
  times, and int8 weights go at port B's chunk a cycle. Both are lifted, compile-side, with the
  same bits (an MM's row groups leave its integer sums as they are):
  - A tile's MoE prologue takes its rows at once: the router one MM of the T rows, the shared
    expert in whole column groups of min(T, MCOLS) rows (T would not fit m4's TMEM). Its need
    lines stay `moe.need_rows(k)` rows each: the first goes out before the shared expert, the
    others as its passes begin (`swiglu_down(each=)`). A post waits for the host to serve the one
    before, so back to back they would add the host's poll (~150 us with parts in flight) two or
    three times a tile; the co-simulation serves at once and cannot show it, the card's A/B can.
  - A tiled MoE image's gr / on rows are its tile's (16-20 KB more; no expert slot lost at 2048
    or 4096 tokens), and a DeltaNet layer runs up to MCOLS rows at a time (more only replays).
  - Co-simulated, ms a row (T = 8, position 256), before / the prologue / both:

    | layer | m4 | m8 |
    |---|---|---|
    | DeltaNet, MiniCPM-V's int8 | 1.026 / 0.967 / 0.968 | 1.020 / 0.936 / 0.871 |
    | DeltaNet, MiniCPM-V's nvfp4 | 0.976 / 0.918 / 0.917 | 0.964 / 0.877 / 0.864 |
    | attention, MiniCPM-V's | 0.710 / 0.651 / 0.651 | 0.488 / 0.401 / 0.401 |
    | DeltaNet, qwen35-35b's | 0.936 / - / 0.917 | 0.923 / - / 0.863 |
    | attention, qwen35-35b's | 0.670 / - / 0.650 | 0.448 / - / 0.400 |

    MiniCPM-V's mixer: m4 37.1 -> 34.8 ms a row (-6.3%: -1.15 s of the image prompt), m8 34.6 ->
    30.0 (-13.3%: -2.25 s); qwen35-35b's -2.2% and -7.1% (its shared expert 4-bit: half the
    bytes, and paired at 2 rows). On m4 the gr / on rows change nothing (the same program).
  - `tests/test_em_tile.py` test_a_tiles_moe_prologue_takes_its_rows_at_once: a tiny model at
    k = 4 (two need lines a tile): the router's and the shared expert's rows, the posts, and the
    prompt, blocks and decode steps against today's runs bit for bit, every need landed.
  - On the card (session emprol, `tools/offload/sessions/emprol.sh`, 2026-10-14; the card host
    `~/otpu-build/emprol/run-1014-1320`): r2.1's m8 bitstream 039b34d5, production's host d57a3568
    as A against these commits as B, a warm-up run in each arm, then A B B A, every run's tokens
    the warm-up's (A's):

    | request | A: TTFT (device) | B: TTFT (device) | B - A, mean |
    |---|---|---|---|
    | MiniCPM-V, doc.png at 9 slices (488 prompt rows) | 53.07 / 52.85 s (25.17 / 25.13) | 51.36 / 50.60 s (23.03 / 22.87) | TTFT -1.98 s (-3.7%), device -2.20 s (-8.7%) |
    | qwen35-35b, a 260-token text prompt | 15.60 / 15.33 s (13.93 / 13.73) | 15.16 / 15.07 s (13.57 / 13.50) | TTFT -0.35 s (-2.2%), device -0.30 s (-2.1%) |

    The device's -2.20 s on the image request is the co-simulation's -2.25 s. The host's share of
    its TTFT moved +0.2 s, within the pairs' 0.5 s spread: the need lines' posts between the
    shared expert's passes cost the card no more than that. Decode unchanged (4.28-4.37 and
    4.42-4.50 tokens a second).

## 15. Chunked DeltaNet (proposal, parked, 2026-10-14)

Not built: r2.2's prefill ends at section 14. This section sizes the kernel for a later decision.

A Qwen3.5 prompt row's DeltaNet recurrence is DSTEP's: 2,048 cycles a row and value head (the
head's 16K fp32 state words through the stream engine at 8 a cycle, read from DRAM and written
back: MiniCPM-V's 488-row image prompt moves 61 GB of state). The chunked form (docs/dataflow.md
4.9.1: eight matmuls and a triangular solve a chunk of C rows, the state once a chunk) with every
matmul split (an operand's int8 part and the int8 of its remainder: three MXU passes) at C = 64 is
the candidate: +0.13 dKL (SE 0.07) on MiniCPM-V's card mix, ~1 GB of state traffic on that
prompt, and issue cycles a row and head of MXU 768 / VPU 598 / DRAM 16 at MCOLS 4 (MXU 384 at
MCOLS 8; the VPU's 8 lanes either way).

### 15.1 What it would gain

DSTEP's exposed part, a row's cycles with and without it (tools/vpu_hack.py's nose: every DSTEP /
STREAM a NOP): MiniCPM-V's from section 14.3's layers (T = 8, position 256; docs/dataflow.md
4.9.1); the dense models' from tools/perf_tiled.py's run kernel (4 layers: 3 DeltaNet and the
gated attention, position 256, fp4, runs of 4 rows; the 4B at MCOLS 4 in runs of 2, as its prompt
runs go today; the simulation host, `~/mcpv/emprol/K2-*.jsonl` and `K3-*.jsonl`):

| | MCOLS 4: exposed (of DSTEP's cost) | MCOLS 8 |
|---|---:|---:|
| MiniCPM-V's image prompt (DSTEP 7.2 s) | 4.1 s (57%) | 5.5 s (76%) |
| Qwen3.5-0.8B, of a row | 30.6% (89%) | 45.3% (110%) |
| Qwen3.5-2B | 8.6% (46%) | 24.3% (91%) |
| Qwen3.5-4B | 10.2% (58%) | 24.2% (84%) |

The chunked recurrence would be a run of its own (15.2), hidden by nothing: 0.29 (MXU and VPU
overlapped) to 0.48 (one after the other) of DSTEP's cycles at MCOLS 8, 0.375 to 0.67 at MCOLS 4.
The gain is the exposed part less that (estimated; negative is faster):

| | MCOLS 8 | MCOLS 4 |
|---|---|---|
| MiniCPM-V's image prompt | -2.0 to -3.4 s | -1.4 to +0.7 s |
| qwen35-35b / MiniCPM-V text, a row (30 DeltaNet layers: DSTEP 14.7 ms, taken as exposed as MiniCPM-V's nvfp4 layers', 92% / 52%) | -6.5 to -9.2 ms (-22 to -31% of the mixer's ~30) | -2.1 to +2.2 ms |
| Qwen3.5-0.8B, a row | -25 to -33% | -8 to -18% |
| Qwen3.5-2B | -12 to -17% | -2 to +4% |
| Qwen3.5-4B | -10 to -16% | -4 to +2% |

So MCOLS 8 (r2.1's m8 and later; the 0.8B also at 4), and a compile choice by MCOLS and the
prompt's rows, not a bitstream's: DSTEP stays for decode, the MTP verify and short prompts. LFM2
and Gemma 4 have no DeltaNet.

### 15.2 The phase split

C = 64 needs 64 rows of a layer at once, and a tile holds 8 (MoE, section 14) or 4-16 (dense).
So a DeltaNet layer would run three kinds of run for each chunk j of 64 rows:
- A(j), the tile without DSTEP, in tiles of T rows: the projections, the convolution and the
  gates; q, k, g and beta to a DRAM scratch, with k^T and v^T written transposed (the V^T store
  path);
- R(j), the chunk's recurrence over every head, one run;
- C(j): the output's norm and gate, out_proj, the residual and the MoE prologue (its need lines
  posted as today).

A layer runs A(0)'s tiles, R(0), the tiles of C(0) fused with A(1), R(1), and so on, C's last
tiles, then (MoE) the expert run: one run more for each 64 rows and layer (~240 on MiniCPM-V's
image prompt, ~10-20 ms). The last chunk is padded with rows of k = 0, beta = 0 and g = 0 (no
decay), their outputs dropped. The state keeps DSTEP's layout (St, fp32), so decode follows a
chunked prompt as it is.

No TMEM copies: four of the eight matmuls contract over the chunk's rows and take an operand
transposed. A writes k^T and v^T; R forms U^T itself (U^T = (beta v)^T . T^T - St . W^T, St
already transposed), which M7 (A . U) and M8 (St' = U^T . Kd) take as it is; the beta and
exp(gc) scalings fold into the TMEM side's operand. `ol.dot`'s w is a DRAM QTensor, so R stores
T, W, U^T and St's int8 parts and streams them back: ~4 store-and-stream round trips a chunk and
head (~50 KB), and A's transposed stores are ~12 KB of strided bytes a row and layer. Neither is
in the cycles above; step 0 (15.5) measures them.

The dense models' prompt runs take every layer a tile at a time, so for them the split means a
layer-major DeltaNet prefill. Without it, a chunk of the tile's own rows (C = 16, the 0.8B's and
2B's tiles) takes MXU 636 / VPU 974 cycles a row and head at MCOLS 8 (1,272 / 974 at 4): the
0.8B -13 to -26% a row and the 2B -3 to -12% at MCOLS 8, the 0.8B -9 to +7% at MCOLS 4; the 4B's
tiles of 8 rows are too short (a contraction over C < D pads to D).

### 15.3 Fits

- IMEM (4K instructions): R for a value head is 8 matmuls x 3 passes in MMs of at most 16 rows
  (M8: 128), ~110 MMs, with the splits, the exp / Gamma masks, a looped forward-substitution solve
  and the state's load and store ~170 instructions. Unrolled over 32 heads that is ~5.4K, which
  does not fit; a hardware loop over the 16 key heads (two value heads a body, q . k^T and
  k . k^T shared, DRAM addresses affine in the loop) ~400, in a run of its own. The C + A tile is
  today's (DeltaNet 1,249 instructions at MCOLS 8, 1,675 at 4) and the scratch stores, ~+10%.
- TMEM: R's working set at C = 64 is ~72K words a value head (q, k, W, U^T and O 8K each, T and
  A 4K each, St 16K). MCOLS 8's 128K holds it; MCOLS 4's 64K needs the value dimension in halves
  (~56K) or C = 32.

### 15.4 Numerics

Not today's bits: the chunk's sums round otherwise than DSTEP's sequential fp32. Every Qwen3.5
token-exact reference changes, and the gate becomes the dKL against the float model (docs/
dataflow.md 4.9.1's method). Measured on MiniCPM-V's card mix only (2,000 tokens: +0.13, SE 0.07,
against a bar of about +0.5); the 0.8B, 2B, 4B and qwen35-35b's mix are not.

### 15.5 Plan, when taken up

1. Step 0, the go / no-go (2-3 days): tools/dn_chunk.py on the 0.8B, 2B, 4B and qwen35-35b's mix
   at C = 64 and 16; one head pair's chunk of R co-simulated at MCOLS 4 and 8 (the solve, the
   splits, the round trips and the transposed layout as they would be); A's strided stores. Go if
   R takes at most ~0.5x DSTEP's cycles at MCOLS 8 and the dKL is at most ~0.3 on every model.
2. The R kernel (5-7 days): the head-pair loop, the chunk's rows at run time and the padding,
   St's layout, an MTP image's slots; tests against tools/dn_chunk.py's emulation; the fits at
   MCOLS 4 and 8.
3. The MoE phases (5-7 days): `tiled.qwen35_layer_tile` split into A and C, the fused C + A
   tile, the engine's run order; tests/test_em_tile.py against a chunked reference.
4. The dense models (4-6 days): a layer-major DeltaNet prefill, or C = 16 in the 0.8B's and 2B's
   tiles.
5. References for every Qwen3.5 model, the dKL gate, the card's qualification and an A/B
   (MiniCPM-V's image, qwen35-35b's text, dense prompts): 2-3 days and card windows.

About 3.5-4.5 weeks; the MoE part (1-3 and 5) 2.5-3.

### 15.6 Other ways

- DSTEP at 16 lanes (RTL): today's bits and no new references; but the state's 128 bytes a cycle
  are all of DDR3-1066's, so it gains less than 2x (~-2.2 s of the image prompt at MCOLS 4,
  estimated), and the 61 GB stay.
- An MM taking ~16-bit operands in one pass (a QACT / QST mode, RTL), with the chunked form: the
  VPU's 598 cycles a row and head fall to ~294 (the split's residuals go), MXU-bound at 384 on
  MCOLS 8: ~-0.75 s more of the image prompt.
- A short-K MM (ASIC, docs/dataflow.md 7.2 item 8): a contraction over C < D rows at C / D of a
  block's cycles, small chunks ~3x cheaper on the MXU, so a tile's own rows would pay without the
  phase split.
