# Multi-token prediction on the card (design note)

Status: **design only, 2026-09-30.** There is no RTL or compiler work yet. **Measured** means a
full-model simulation of today's production configuration with LiteDRAM's controller
co-simulated (within 1.1% of the card, docs/board.md). **Projected** means arithmetic on those
measurements; nothing projected has run.

The goal: more than one token per weight pass in the decode loop on the card
([autodecode.md](autodecode.md)), for every model on one bitstream, with no host work per token
and no model math on the host. It runs in three steps:

1. A **drafter** proposes k tokens after the current one.
2. A **verify run**, the multi-row prefill run at a run-time position, computes the model's own
   next token at each of the k + 1 positions.
3. The card **accepts** the longest prefix of the drafts that matches, emits it plus the model's
   next token, rolls the recurrent state back to the last accepted row, and drafts again.

Greedy decode gives exactly the tokens of plain decode. Sampled decode gives the same
distribution (section 5.2).

## 1. What a verify run costs

`tools/perf_qwen.py --layers 0 --wformat fp4 --head-format int8 --pos 544 --cap 2048 --ddr 1066
--mhz 133.33 --ldc --rows R --logits all` gives the cycles of one run of R rows with logits for
every row. It uses the production configuration: `OTPU_MXU=systolic OTPU_MCOLS=4 OTPU_PAIR=1
OTPU_DSTEP=1 OTPU_STREAM=1`. R = 1 is the decode step (`--rows 0`). This is the card's qual
operating point. Measured on omarchy, main 16e578c:

| model | decode step (R = 1) | R = 2 | R = 3 | R = 4 | LM head share of a decode step |
|---|---|---|---|---|---|
| Qwen3-0.6B | 3,772,943 (35.3 tok/s) | 4,387,572 (1.16x) | 6,414,864 (1.70x) | 6,955,127 (1.84x) | 35% (161 MB, int8) |
| LFM2.5-230M | 1,387,786 (96.1 tok/s) | 1,509,286 (1.09x) | 2,185,300 (1.57x) | 2,306,814 (1.66x) | 41% (69 MB) |
| Qwen3.5-0.8B | 4,943,542 (27.0 tok/s) | 6,101,487 (1.23x) | 8,290,097 (1.68x) | 9,392,558 (1.90x) | 44% (263 MB) |

Cycles per run; the factor is against the decode step.

Measured later, for phase 0 (section 7.1), on main c2d2975 (the pipelined rows kernel with
its fused VOPs). The 4B's full image overflows the LiteDRAM model, so its numbers are from 4-
and 8-layer runs, extrapolated to 32 layers (each 4 layers is three DeltaNet and one
attention). Those runs take 7,429,621 and 9,524,096 cycles at R = 1, and 7,593,481 and
9,845,932 at R = 2. The extrapolated 6.0 tok/s is within 3% of the card's 5.88.

| model | decode step (R = 1) | R = 2 |
|---|---|---|
| Qwen3.5-0.8B | 4,943,542 | 5,962,906 (1.21x) |
| Qwen3.5-2B | 10,740,340 (12.4 tok/s) | 11,327,655 (1.055x) |
| Qwen3.5-4B | 22,090,946 (6.0 tok/s) | 23,360,638 (1.057x) |
| SmolLM3-3B | 14,840,079 (9.0 tok/s) | 15,400,586 (1.038x) |
| LFM2-2.6B | 11,893,504 (11.2 tok/s) | 12,141,498 (1.021x) |

The 2B has the same 16 DeltaNet heads as the 0.8B, so the same per-row DSTEP and VPU work, but
about 2.2x the weights. The MXU still bounds its 2-row run. The 4B has 32 DeltaNet value heads
and 32 layers. Its second row costs 2.2x the 2B's (1.27 M against 0.59 M cycles), but its
decode step grows 2.1x as well, so its c_2 is the 2B's.

What sets these costs:

- **Two rows ride free on the weights; three do not.** fp4 MMs run at full rate through column
  reuse (`MM PAIR`, docs/isa.md), which needs `2 M <= MCOLS`. On MCOLS = 4 that means M <= 2
  rows. With 3 or 4 rows the MM takes one block per cycle instead of a whole chunk, so every
  weight MM becomes MXU-bound at half rate. That is the jump from 1.16x to 1.70x (Qwen3's MLP:
  1.17 M -> 2.11 M cycles; "port A 5.2 M, MXU 5.5 M" in the run).
  - **Consequence:** on today's bitstream the verify run is 2 rows wide: one draft token per
    iteration (k = 1).
  - Wider verify runs need more MXU columns: MCOLS = 8 keeps PAIR up to 4 rows. That carries an
    area and fmax cost: MCOLS = 4 is already near the device's slice limit.
- **The per-row cost is what does not share weights:**
  - **Attention** reads its row's KV. For Qwen3 at position 544: +0.68 M cycles per row
    (attention 1.20 M -> 1.88 M).
  - **DeltaNet** reads and writes the whole state once per row with DSTEP. For Qwen3.5-0.8B:
    +1.01 M per row (DeltaNet 1.36 M -> 2.37 M; +38 MB of traffic per row, which is 2 x 18 MiB of
    state).
    - Only about 0.3 M of that is the state traffic itself (38 MB at 128 B per cycle).
    - The rest is the stream engine. DSTEP runs on it and the VOPs wait while a DSTEP holds it
      (docs/isa.md, DSTEP), so a pair's VPU work (convolution, SiLU, norms, the gated RMSNorm)
      and its 2 R DSTEPs (2,248 cycles per head step) run one after the other.
    - With one row (a 4-layer co-sim of the rows kernel) the DeltaNet layers are MXU-bound:
      the MXU is busy 97% of their cycles, the engine 93%. With two rows the engine is the
      bound: busy 95.5% (the DSTEPs 72% of the cycles), the MXU 77%.
    - The VPU's part grows with the rows: its throughput bounds it, not the VOPs' latency. The
      SiLUs' EXP2 and RECIP run on the long lanes at 2 columns per cycle, the other VOPs at 8
      (docs/stream.md). That is about 0.33 M cycles per row in the DeltaNet layers.
    - The rows kernel now pipelines the pairs two ahead, as the decode kernel `_deltanet_dstep`
      does (`_rows_pipelined`: the MXU projects pair p + 2 while the engine runs pair p's
      DSTEPs). The 2-row run went from 6,101,487 to 5,984,177 cycles: 1.21x a decode step,
      -1.9%. Prefill runs gain about as much (4 rows -2.3%, 6 rows -2.2%).
    - This note projected 1.05x before. That projection set the two rows' DSTEPs (2 x 0.65 M)
      against the MXU's 1.1 M for DeltaNet's weights and left out the VPU work sharing the
      engine.
    - Each of the kernel's VOPs now takes both heads of a pair, as `_deltanet_dstep`'s do. That
      saves only 0.4% (5,962,906 cycles for 2 rows), because the elements stay the same. A
      timing experiment that dropped head 1's VPU work altogether gave 5,643,600 (-5.7%): that
      is the VPU's whole share.
    - What would cut the rest is hardware, and the MXU bounds it at about 1.12x (estimated: the
      DeltaNet layers at the MXU's busy time). Either the VOPs run beside a DSTEP (the
      engine's slot-0 loop is the DSTEP's datapath), or the long lanes get wider
      (docs/stream.md section 9).
  - LFM2's convolutions and short attention are cheap: +0.12 M per row.
- **The LM head is 35-44% of a decode token.** The verify run pays it once: 2 rows share the
  head's weights. But a drafter that runs the full head for every draft token pays that share
  again per draft. Section 6.1 deals with this for the MTP head.

## 2. The loop on the card (greedy, k = 1)

The generate loop's state block (generate.py `S_*`) gains the draft `d`, the accepted count
`n` and the recurrent-state slot `b` (section 4). One iteration at position p, with t the token
at p:

```
verify run, rows (t, d) at positions p, p + 1:
    a0 = argmax logits(row 0)          the model's token at p + 1
    a1 = argmax logits(row 1)          its token at p + 2, valid only if d == a0
n  = (d == a0)                         accepted drafts, 0 or 1 (k = 1)
n  = min(n, stop: a0 is a stop id -> 0, tokens left - 1)
out[p + 1] = a0;  if n: out[p + 2] = a1          (stores guarded by n; the host reads out[])
commit: p += n + 1, tok = a_n, b = slot of row n (section 4), ring and tpos as their rules
draft:  d = drafter(...)                         (section 6)
stop:   a stop id emitted, the host's stop word, or no tokens left -> HALT
```

For larger k, n is the length of the run of leading matches: c_i = prod_{j <= i} (d_{j+1} ==
a_j), n = sum c_i. That is k scalar VOPs. The tokens are a_0 .. a_n.

The pieces, against what the generate loop has today:

- **Guarded stores and data-dependent counts:**
  - RLD of n.
  - `LOOP R` of count 0 or 1 around each optional store (the stop HALT's pattern, generate.py
    `_token_end`).
  - RLD MUL for the slot's address (section 4).
- **A data-dependent loop length.** Today's count is `min(left, block - tpos)`, fixed at the
  loop's start. With MTP an iteration advances 1..k + 1 positions. So the loop takes a large
  count, and a guarded HALT CHAIN leaves it:
  - with no tokens left, a stop, or the host's stop word: HALT;
  - when `tpos + k + 1 > block` (the verify rows would leave the bucket): ST the state, then
    CHAIN to the bucket's **plain** generate program. That program runs the last <= k
    positions one at a time and chains to the next bucket's MTP program at the wrap.
- **CHAIN is cheap enough to use per iteration.** A reload of a 2,000-instruction program is
  64 KB, about 500 cycles of port plus latency: 0.01% of a token.
  - So an iteration can also take the plain step when there is no draft (the n-gram drafter
    finds no match). It CHAINs between the verify program and the plain program, rather than
    paying a 2-row run for one token.
  - The two bodies need not fit IMEM together. Qwen3.5-0.8B's generate program alone is 2,020
    instructions.
- **The host is unchanged.** `run_generate` already reads tokens from `out[]` as they land; they
  just land 1..k + 1 at a time. The host writes the prompt's ids once per turn for the n-gram
  drafter (section 6.2).

## 3. The verify run at a run-time position

Today `qwen3_rows` / `qwen35_rows` / `lfm2_rows` are compiled per row list: positions and
tokens are compile-time values, as prefill needs. The verify run is the same kernel at a
`RunPos`:

- **Positions** p + r = t0 + tpos + r: one argument register (c * tpos) serves every row with a
  static offset r. The bucket's attention mask covers rows up to tpos + k (the run stays in its
  bucket, section 2).
- **Tokens** t and d are device values. The embedding rows (and Gemma 4's PLE rows) are gathered
  at run-time token ids, as the generate loop does today for t.
  - Each extra token would hold one argument register for the whole program (two with the int8
    embedding), yet it is used at one gather.
  - An RLD into a scratch register at the gather holds none. Compiler item: a RunVar used at
    one place gets a scratch RLD there, not an argument register.
- **V^T appends** at a run-time position go one row per append (language.kv_append: "V^T rows at
  a run-time token: one row per append"). That costs R small stores.
- **Logits for every row:** `_lm_head_rows` over rows 0..k, then ARGMAX per row. ARGMAX takes a
  row stride (`tests/test_autodecode.py::test_argmax_index_base_and_row_stride`). Greedy needs
  only the per-row maximum's index. Sampled needs each row's logits through the sampler
  (section 5.2).
- **The hidden rows** go to DRAM for the MTP drafter after the final norm (the LM head's
  input), as `m.lm_split` stores x for the split generate programs.
- **Programs per bucket:** the verify program, the plain program and (for MTP) the draft
  kernel, chained as today's split programs are (`ptab2`). The chain area holds 8K instructions
  per bucket and mode.

## 4. Rolling the state back

A rejected row has already written its state. Every piece of state either does not need
restoring or keeps one copy per row:

| state | after a rejection | how |
|---|---|---|
| KV cache (all attention layers, the MTP layer's) | nothing to undo | positional: the next run appends at p + n + 1 and overwrites; rows attend only to positions <= their own. K has per-token scales, V^T one scale per token, so overwriting one position is exact. |
| LFM2's conv ring (rows of position p in slot p mod K, mirrored) | nothing to undo | ring modulus K -> K + k. A rejected row lands in a slot no kept row needs. With K = 3 and k = 1 the ring has 4 slots: `ring` = (p + 1) mod 4 in `generate.rules`. |
| Qwen3.5's conv window (K - 1 = 3 rows per pair, stored after the taps) | keep one per row | the verify run stores the window after each row into slot r: 3 rows of the pair's convolved channels (768 words for 0.8B), 1.3 MB in all for 0.8B per row. The next run loads slot n. |
| DeltaNet state S (per head, [dv, dk] fp32; 18 MiB for 0.8B) | keep one per row | k + 2 slots per head. Base b. Row r reads slot (b + r) mod (k + 2) and writes (b + r + 1) mod (k + 2). The new base is b + n + 1. The slot is one more run-time variable (RLD MUL into the state's address register). |
| The sampler's penalty factors (`pa`, `pb`) | tentative per row | row i applies the drafts d_1 .. d_i as if generated (section 5.2). Commit applies a_0 .. a_n. |

**The DeltaNet slots cost no bandwidth with DSTEP.** DSTEP already reads and writes the whole
state once per row (that is the +38 MB per row of section 1). Writing it to the next slot
instead of in place moves the same bytes. The only cost is k + 1 more copies of the state in
DRAM: 36 MiB more for 0.8B at k = 1, and more for the 4B / 9B / 35B states. DRAM has room
(4 GiB card); offload's slot planning has to count it.

**One RTL item.** DSTEP writes in place. STREAM's ISA has `src` and `dst`, but the board's
hardware subset (`isa.stream_hw_cfg`) takes only "DRAM state in place". A DSTEP / STREAM with
`dst != src` is a DMA change: the write address stream gets its own base. It is the only
hardware change k = 1 needs.

- Without DSTEP (the VOP path) the rows kernel loads a head's state into TMEM once. It then
  stores it once per row instead of once at the end: k more state writes per run.
- **Rejected work is the price.** A rejected row's attention and DeltaNet step are wasted. That
  is what the per-row cost of section 1 measures, and why the break-even acceptance below is
  the per-row cost over a decode step.

## 5. Accepting on the card

### 5.1 Greedy

Section 2's compare runs on the VPU in a few VOPs on k words. With stop ids, n is cut at the
first emitted stop id (`_token_end`'s check per emitted token). With tokens left, n <= left - 1.
Greedy MTP gives bit-identical tokens to plain decode, because each emitted token is the
argmax of a logits row the plain step would also compute. That holds only if the rows run is
bit-identical to R decode steps, which `_attention_rows` and `_deltanet_rows` promise and their
tests check. **This is the test for every phase: tokens equal to the plain generate loop's.**

One caveat bounds the width. An fp4 MM with PAIR sums a chunk's two blocks before the
accumulation, so its fp32 rounding differs from a PAIR-less MM (docs/isa.md, "Column reuse").
The decode step runs with PAIR. So a verify run is bit-identical to it only while it runs with
PAIR too, which means R <= MCOLS / 2: 2 rows today. A wider run gives logits that differ in the
last bits, and an argmax tie could then flip.

### 5.2 Sampled

The drafts are deterministic (greedy drafts, or n-gram lookups), so the draft distribution q is
one-hot. The standard speculative sampling rule then reduces to:

- Accept draft d at row i with probability p_i(d). p_i is the target's processed distribution
  at row i: temperature, top-k, top-p and the penalty, exactly what `generate.Sampler` computes.
- On rejection, sample from p_i with d removed, renormalized.

The output follows p exactly. On the card:

- **Two uniforms per position** instead of one. The host already writes them per run (`uni[]`).
- **p_i(d)** is one gather from the sampler's distribution of row i.
- **The residual draw** is the sampler's draw with d's mass zeroed: one store of 0 into the
  sorted candidates before the cumulative sum.
- **The penalty** at row i must see d_1 .. d_i. Their factors are applied to row i's logits as
  i gathers and scales. They are committed only for a_0 .. a_n.

`reference_pick` grows a `verify` mode that the ISA simulator's runs are checked against, as
the sampled generate loop is today (`test_sampled_generate_matches_the_reference_pick`).

## 6. Drafters

### 6.1 Qwen3.5's MTP head

Qwen3.5-0.8B, 4B and 35B-A3B ship one MTP layer (`mtp_num_hidden_layers: 1`,
`mtp_use_dedicated_embeddings: false`). Our loader skips `mtp.*` today
(opentpu/llm/qwen3.py:130); transformers does the same (`_keys_to_ignore_on_load_unexpected =
[r"^mtp.*"]`). Qwen3.5-0.8B's tensors, from the checkpoint's header (bf16):

| tensor | shape |
|---|---|
| `mtp.pre_fc_norm_embedding.weight`, `mtp.pre_fc_norm_hidden.weight` | [1024] each |
| `mtp.fc.weight` | [1024, 2048] |
| `mtp.layers.0.input_layernorm`, `.post_attention_layernorm` | [1024] each |
| `mtp.layers.0.self_attn.q_proj` (q and its output gate, as the model's attention layers) | [4096, 1024] |
| `.k_proj`, `.v_proj` | [512, 1024] each |
| `.o_proj` | [1024, 2048] |
| `.q_norm`, `.k_norm` | [256] each |
| `mtp.layers.0.mlp.gate_proj`, `.up_proj` / `.down_proj` | [3584, 1024] / [1024, 3584] |
| `mtp.norm.weight` | [1024] |

So the MTP layer is one full-attention layer (gated, 8 query heads and 2 KV heads of 256, its
own KV cache) with the dense MLP. It has 20.4 M parameters, about 10 MB in fp4. The embedding
and the LM head are the model's (tied).

The dataflow follows the one installed reference, mlx_vlm 0.6.8's drafter
(`mlx_vlm/speculative/drafters/qwen3_5_mtp/qwen3_5_mtp.py`, `_forward_hidden`). transformers
and mlx_lm drop the weights, and vLLM and SGLang are not installed:

```
e = pre_fc_norm_embedding(embed(token at t + 1))
h = pre_fc_norm_hidden(hidden of the main model at t, after its final norm model.norm)
x = fc(concat(e, h))                       [2H] -> [H], the embedding first
x = the MTP decoder layer (gated full attention + SwiGLU), RoPE position t, its own KV cache
draft for t + 2 = argmax(lm_head(mtp.norm(x)))     lm_head = the tied embedding
```

- Every MTP norm is zero-centred, x * (1 + w), as the model's are (qwen35's `g1`).
- The position is the hidden's (mlx_vlm). transformers' generic MtpLayer uses the token's
  instead. Both work: the layer attends only to its own cache, and RoPE is relative.
- mlx_vlm drafts 2 tokens per round by feeding the layer's normed output back as the next
  hidden. After the verify, it pushes the accepted tokens through the MTP layer with the main
  model's verify hiddens to get the next draft, as below.

On the card, per iteration:

- **The MTP layer runs over the verify run's rows**, (normed x_i, a_i) for i = 0..k. That keeps its KV
  cache filled at every accepted position. It is a (k + 1)-row run of one layer, so its weights
  stream once (about 2% of a 0.8B token).
- **The draft comes from row n** (a run-time row: RLD of n into the address of the row read).
- **The draft's LM head is the problem.** The full int8 head is 44% of a 0.8B decode token
  (263 MB). Running it per draft would cost more than the k = 1 verify run itself.
  - Options, cheapest first:
    1. a **draft head over the N most frequent token ids**: fp4, N = 32K rows of 1024 is
       16.8 MB, about 3% of a token. ARGMAX over it, then an id map gather. A draft outside
       the N ids is impossible, which costs some acceptance.
    2. the fp4 copy of the full head (130 MB, 23%);
    3. the full int8 head.
  - Phase 0 (section 8) measures the acceptance of each.
  - The verify run keeps the full int8 head, so the emitted tokens do not depend on this
    choice.
- **For k > 1** the MTP layer runs again on its own output and draft: DeepSeek-V3's chained
  use. On MCOLS = 4 fp4, k > 1 needs 3-row verify runs (section 1), so it waits for wider
  hardware.

### 6.2 Prompt lookup (n-gram), for every model

Gemma 4, LFM2, Qwen3, SmolLM3, Phi-4-mini and Llama-likes have no MTP head. The model-agnostic
drafter matches the last g tokens (g = 2 or 3) against the context and proposes the tokens
that followed the latest match:

- **The history** is the prompt's ids (written by the host once per turn) followed by `out[]`
  (the generated ids, already on the card).
- **The match** is VOPs over the history: eq(h[i], t_{-1}) * eq(h[i - 1], t_{-2}) ... times a
  position ramp, ARGMAX for the latest match, RLD of its index, then a load of the k tokens
  after it. That is a few passes over at most cap words, under 0.1% of a token for 2K
  contexts.
- **No match** means no draft: the iteration CHAINs to the plain step (section 2) and pays
  nothing.
- **Acceptance depends on the text.** It is high for code edits, quoting, structured output and
  long replies that repeat the prompt, and low for free chat. It is cheap enough to leave on,
  because a no-match iteration costs nothing.

### 6.3 A draft model

A small model with the same tokenizer is the general drafter, for example Qwen3-0.6B for a
larger Qwen3. It is out of scope for v1. The verify and accept machinery is the same, but the
draft model's own decode steps cost about its size per draft token.

## 7. Projected gains (k = 1)

With acceptance a (the probability that the draft equals the model's token), an iteration
emits 1 + a tokens and costs c_2 + c_draft decode steps:

speedup = (1 + a) / (c_2 + c_draft)

c_2 comes from section 1 (Qwen3.5-0.8B: 1.21, with the pipelined rows kernel). c_draft is 0
for n-gram and about 0.05 for MTP with a 32K fp4 draft head (10 MB layer + 17 MB head over
574 MB per 0.8B token).

| model, drafter | c_2 + c_draft | break-even a | a = 0.5 | a = 0.7 | a = 0.85 |
|---|---|---|---|---|---|
| Qwen3-0.6B, n-gram | 1.16 | 0.16 | 1.29x | 1.46x | 1.59x |
| LFM2.5-230M, n-gram | 1.09 | 0.09 | 1.38x | 1.56x | 1.70x |
| Qwen3.5-0.8B, MTP + 32K head | 1.26 | 0.26 | 1.19x | 1.35x | 1.47x |
| Qwen3.5-0.8B, MTP + 32K head, VOPs beside the DSTEPs (hardware; c_2 = 1.12, estimated) | 1.17 | 0.17 | 1.28x | 1.45x | 1.58x |
| Qwen3.5-0.8B, n-gram | 1.21 | 0.21 | 1.24x | 1.40x | 1.53x |

These are projections; section 7.1 measures a.
- The one published number is DeepSeek-V3's: 85-90% acceptance of its MTP's second token.
- n-gram's a counts only iterations that found a draft. The others cost exactly a plain step.
  So its rows above overstate it: most tokens get no draft (section 7.1).
- Qwen3.5's per-row DeltaNet cost makes it the least favourable model per row. The rows
  kernel's overlap is in (section 1). What is left is the stream engine's work, which the
  DSTEPs and the VOPs share. A multi-token DSTEP (docs/stream.md 5.2: T tokens per state
  pass) would also cut the state traffic: the state is read once for both rows and written
  once per row. Its datapath time stays per row.

### 7.1 Measured acceptance (phase 0)

`tools/mtp_accept.py` measures a offline, on the Hugging Face model on omarchy (float32;
bfloat16 for Qwen3.5-4B and Gemma 4 E2B). It uses no card and no simulator.

- **Prompts.** There are nine, each through the model's chat template with thinking off:
  - three chat questions;
  - three code tasks, one of them an edit of given code;
  - three summaries of a given text: Pride and Prejudice's first chapter (tools/data) and two
    short texts in the tool.

  Each runs for up to 256 tokens, greedy and sampled. Sampling uses otpu-chat's defaults;
  Gemma's and SmolLM3's come from their generation_config.json. omarchy's Gemma 4 E2B is the
  pre-trained checkpoint, which has no chat template, so its prompts are a plain
  "User: ... Assistant:" dialogue.
- **Drafts.** Every generated token after the first gets drafts from the tokens before it:
  - n-gram (6.2) at g = 2, at g = 3, and "3>2": g = 3 where it matches, else g = 2;
  - Qwen3.5's MTP head (6.1), teacher-forced over the sequence, with five LM heads: the full
    float head, its fp4 copy, and fp4 heads over the 16K, 32K and 64K lowest ids. A BPE
    vocabulary numbers its tokens by merge order, which follows frequency: the 32K lowest ids
    cover 94% of the tokens the models generate here (16K: 88%, 64K: 97%). The activations
    go through int8, as the card's QACT.
- **Acceptance.** Greedy: the draft equals the model's token. Sampled: the probability p(d)
  under the processed distribution (5.2), averaged.
- **Speedup.** The k = 1 loop's expected cost over each sequence, with every token emitted
  once:
  - a plain step costs 1;
  - an iteration with a draft costs c_2 + c_draft and emits two tokens when the draft is
    accepted;
  - the token right after an accepted draft gets no iteration of its own. n-gram's hits come in
    runs, so per-token rates overstate its gain.

  c_2 is 1.21 for every model (Qwen3.5-0.8B's, section 1). c_draft is the MTP layer plus its
  head, as a share of a 0.8B token: 0.035 / 0.05 / 0.08 for the 16K / 32K / 64K heads, 0.25
  for the full fp4 head, 0.46 for the full int8 head. It is 0 for n-gram.

**Qwen3.5's MTP head** (all nine prompts; acceptance, then speedup):

| draft LM head | c_draft | 0.8B greedy | 0.8B sampled | 2B greedy | 2B sampled | 4B greedy |
|---|---|---|---|---|---|---|
| full (int8 on the card) | 0.46 | 0.76, 1.04x | 0.71, 1.01x | 0.79, 1.06x | 0.76, 1.04x | 0.81, 1.07x |
| full fp4 | 0.25 | 0.76, 1.19x | 0.71, 1.15x | 0.78, 1.20x | 0.75, 1.19x | 0.80, 1.22x |
| fp4, 16K ids | 0.035 | 0.71, 1.37x | 0.65, 1.31x | 0.72, 1.38x | 0.69, 1.36x | 0.73, 1.37x |
| **fp4, 32K ids** | 0.05 | 0.73, 1.36x | 0.67, 1.32x | 0.74, 1.38x | 0.72, 1.36x | 0.76, 1.38x |
| fp4, 64K ids | 0.08 | 0.75, 1.34x | 0.69, 1.29x | 0.76, 1.36x | 0.74, 1.34x | 0.79, 1.37x |

By prompt kind, 32K head, greedy (0.8B / 2B / 4B):
- chat: 0.64, 1.29x / 0.66, 1.31x / 0.70, 1.33x;
- code: 0.87, 1.49x / 0.88, 1.49x / 0.89, 1.50x;
- summaries: 0.68, 1.32x / 0.70, 1.37x / 0.70, 1.34x.

The 2B and 4B columns use the 0.8B's c_2 (1.21) and c_draft. Their own c_2 is lower (section
1: 1.055 for the 2B, 1.057 for the 4B), so their real gain is larger; the ranking at the end
of this section uses it.

**n-gram, 3>2** (all nine prompts; greedy / sampled):

| model | tokens with a draft | accepted | speedup | summaries, greedy | code, greedy |
|---|---|---|---|---|---|
| Qwen3-0.6B | 0.36 / 0.35 | 0.60 / 0.60 | 1.08x / 1.09x | 1.18x | 1.08x |
| LFM2.5-230M | 0.26 / 0.22 | 0.53 / 0.53 | 1.05x / 1.04x | 1.09x | 1.09x |
| LFM2-2.6B | 0.24 / 0.22 | 0.47 / 0.46 | 1.04x / 1.03x | 1.04x | 1.07x |
| SmolLM3-3B | 0.28 / 0.26 | 0.50 / 0.49 | 1.05x / 1.05x | 1.07x | 1.08x |
| Gemma 4 E2B (pre-trained) | 0.61 / 0.38 | 0.80 / 0.55 | 1.23x / 1.09x | 1.52x | 1.18x |
| Qwen3.5-0.8B | 0.34 / 0.27 | 0.54 / 0.49 | 1.07x / 1.05x | 1.09x | 1.08x |
| Qwen3.5-2B | 0.27 / 0.25 | 0.47 / 0.46 | 1.04x / 1.04x | 1.08x | 1.06x |

g = 2 alone is within 0.01x of 3>2. g = 3 alone drafts less often (11-22% of the tokens) and
gains less (1.03-1.07x).

What follows:

- **Qwen3.5 gets the MTP drafter, with the 32K fp4 draft head.** It gives 1.36x / 1.38x greedy
  on the 0.8B / 2B, 1.32x / 1.36x sampled, and 1.49x on code.
  - Between 16K and 64K the head size hardly matters. 32K is the middle. 64K's extra
    acceptance does not pay for its larger head. 16K covers only 88% of the tokens and loses
    about as much acceptance as its smaller head saves.
  - The full heads cost more per draft than their acceptance returns. The int8 one comes to
    1.04x.
  - This matches section 7's a = 0.7 column (1.35x).
- **MTP beats n-gram even where n-gram matches.** At the tokens where g = 3 matches, the 32K
  MTP head is accepted 0.78 (0.8B) and 0.82 (2B), against n-gram's 0.58 and 0.54. Qwen3.5
  needs no n-gram.
- **On the chat checkpoints, n-gram gives 1.03-1.09x on average** and up to 1.18x on
  summaries.
  - Only 22-36% of tokens get a draft, and about half of those are accepted.
  - A token without a draft costs a plain step, so it never loses (1.00x on chat at worst).
    That makes it safe to leave on, but it is a small win at c_2 = 1.21.
  - At Qwen3's measured c_2 of 1.16 it gives 1.10x, and at LFM2.5's 1.09 it gives 1.07x.
  - Gemma 4 E2B's numbers are higher because its pre-trained checkpoint copies: greedy, it
    repeats the request and the text it was asked to summarize. That makes 1.52x on summaries
    a copy rate, not a summary. A chat checkpoint (gemma-4-E2B-it) needs its own run.
- **With VOPs beside the DSTEPs** (c_2 about 1.12, section 1), the 0.8B's 32K MTP comes to
  1.47x greedy and 1.42x sampled.
- **Caveats:**
  - The float model stands in for the card's fp4 one.
  - Nine prompts give a few hundred drafts per kind.
  - The tables above use c_2 = 1.21 for every model. The ranking below uses each model's own,
    where section 1 measures it.

**What to build, ranked, with the card's tok/s.** Each model at its own c_2 (section 1; 1.21
where it is not measured). The MTP's c_draft is its layer plus the 32K fp4 head, as a share of
that model's token: 0.05 for the 0.8B and 2B, 0.04 for the 4B. "Today" is the card's device
tok/s with 4-bit weights and the int8 head (README, production build B).

| model | drafter | c_2 | today | greedy | sampled |
|---|---|---|---|---|---|
| Qwen3.5-2B | MTP, 32K fp4 head | 1.055 | 12.09 tok/s | 1.57x, 19.0 tok/s | 1.55x, 18.7 tok/s |
| Qwen3.5-4B | MTP, 32K fp4 head | 1.057 | 5.88 tok/s | 1.59x, 9.4 tok/s | not run |
| Qwen3.5-0.8B | MTP, 32K fp4 head | 1.21 | 24.5 tok/s | 1.36x, 33.3 tok/s | 1.32x, 32.3 tok/s |
| SmolLM3-3B | n-gram 3>2 | 1.038 | 8.74 tok/s | 1.08x, 9.4 tok/s | 1.08x, 9.4 tok/s |
| LFM2-2.6B | n-gram 3>2 | 1.021 | 10.96 tok/s | 1.07x, 11.7 tok/s | 1.06x, 11.6 tok/s |
| Qwen3-0.6B | n-gram 3>2 | 1.16 | 31.3 tok/s | 1.10x, 34.4 tok/s | 1.10x, 34.4 tok/s |
| LFM2.5-230M | n-gram 3>2 | 1.09 | 85.8 tok/s | 1.07x, 91.8 tok/s | 1.06x, 90.9 tok/s |
| Gemma 4 E2B (pre-trained) | n-gram 3>2 | 1.21 (not measured) | 9.6 tok/s | 1.23x, 11.8 tok/s | 1.09x, 10.5 tok/s |

**n-gram with a g = 1 fallback ("3>2>1").** Where neither g = 3 nor g = 2 matches, the token
after the latest earlier copy of the last token still makes a draft. It is accepted only 0.16-
0.26 of the time. But a 2-row run costs just c_2 - 1 more than a plain step, so the draft pays
where that is well below its acceptance. These numbers are greedy only. The study's ids give
greedy acceptance for any drafter (the next id), but sampled acceptance would need a rerun.

| model | c_2 | 3>2, greedy | 3>2>1, greedy | tok/s |
|---|---|---|---|---|
| LFM2-2.6B | 1.021 | 1.07x | 1.13x | 12.4 |
| SmolLM3-3B | 1.038 | 1.08x | 1.13x | 9.9 |
| Qwen3-0.6B | 1.16 | 1.10x | 1.12x | 35.1 |
| LFM2.5-230M | 1.09 | 1.07x | 1.10x | 94.5 |
| Gemma 4 E2B (pre-trained) | 1.21 | 1.23x | 1.23x | 11.8 |
| Qwen3.5-0.8B (MTP's model) | 1.21 | 1.07x | 1.06x | - |

1. **The verify, accept and roll-back machinery** (plan items 2 and 5) comes first. Both
   drafters need it.
2. **MTP for Qwen3.5** (plan item 3) is the payoff: 1.3-1.6x. It needs the one RTL item,
   DSTEP / STREAM with dst != src, for the DeltaNet state's per-row slots (section 4).
3. **n-gram for the other models** is the generic fallback: 1.06-1.10x on chat checkpoints
   with g = 3 and 2. Falling back to g = 1 gives 1.10-1.13x greedy where c_2 is at most about
   1.16. The g = 1 fallback is chosen per model by its c_2. It needs no hardware change: those
   models have no DeltaNet state, and their KV caches and LFM2's conv ring need no copies
   (section 4). That makes it a software feature on today's bitstream, and a natural first
   drafter for testing the machinery.

## 8. Plan

0. **Acceptance study, no card:** done (section 7.1, `tools/mtp_accept.py`).
   - Qwen3.5's MTP head with a 32K fp4 draft head: accepted 0.73 / 0.67 (0.8B, greedy /
     sampled), 1.36x / 1.32x at c_2 = 1.21. At their own c_2 the 2B gets 1.57x greedy and the
     4B 1.59x.
   - n-gram for the other models: 1.03-1.09x at c_2 = 1.21. At their own c_2 it gives
     1.06-1.10x, and 1.10-1.13x greedy with the g = 1 fallback.
1. **The rows kernel's DSTEP overlap for Qwen3.5** (section 1): done. `_rows_pipelined` runs
   two pairs ahead, as `_deltanet_dstep` does: the 2-row run -1.9% (1.21x), prefill -2.2% at
   6 rows. The rest is the stream engine's work (section 1).
2. **ISA simulator: verify, accept and roll back (greedy, n-gram).**
   - Rows kernels at a RunPos.
   - Scratch-RLD tokens.
   - The slots for DeltaNet S and Qwen3.5's window, and LFM2's K + k ring.
   - The accept / commit / CHAIN logic.
   - Test: tokens equal to the plain generate loop's on tiny Qwen3, LFM2, Qwen3.5 and Gemma 4,
     across a bucket boundary and stops. That includes iterations with every draft rejected,
     forced by a deliberately wrong drafter.
3. **The MTP drafter for Qwen3.5.**
   - The loader keeps `mtp.*`, and an image region holds the MTP layer and the draft head.
   - The kernel, checked against the reference of phase 0 bit for bit in the ISA simulator.
4. **RTL:**
   - DSTEP / STREAM with dst != src (the one hardware change);
   - the RTL tests against the ISA simulator;
   - a dev build (FAST=1, 100 MHz), then the card qual's new phase: tokens equal to the plain
     loop's, plus tokens per iteration.
5. **Sampled verify** (section 5.2), checked against `reference_pick`'s verify mode.
6. **Later:**
   - wider verify runs (MCOLS = 8, or a PAIR mode for 4 rows);
   - k > 1 for MTP;
   - the multi-token DSTEP;
   - MoE: offload's experts per verify row. Two rows route to up to twice the experts, so
     streaming costs grow with k. The slot cache's hit rate decides whether MTP pays on the
     35B-A3B, and that needs offload's measurements.

## 9. Open questions

- The MTP dataflow (section 6.1) is **confirmed against mlx_vlm 0.6.8**'s Qwen3.5 drafter:
  - the concat order is the embedding first, then the hidden;
  - the hidden is taken after `model.norm`;
  - the MTP layer has its own KV cache and its own `mtp.norm`.
  **Still to check against vLLM's `qwen3_next_mtp.py`.** Phase 0 also checks it against the
  checkpoint's acceptance: a wrong detail shows up as near-zero acceptance.
- Whether the 35B-A3B's MTP layer is dense or MoE. The 0.8B's, 2B's and 4B's are dense: phase
  0 loads them as plain decoder layers.
- The draft-head frequency table: phase 0 takes the token ids' merge order (the 32K lowest
  ids) and that is enough. On the 4B, greedy, the 32K head is accepted 0.76 against the full
  fp4 head's 0.80. A corpus count might close part of that gap.
- Whether the plain-step fallback per iteration (no n-gram match) should instead keep a 2-row
  run with a guessed draft. Answered in section 7.1: yes, where c_2 is low. The g = 1 guess is
  accepted 0.16-0.26, against c_2 - 1 = 0.02-0.16 on LFM2-2.6B, SmolLM3, LFM2.5 and Qwen3. At
  the 0.8B's 1.21 it does not pay. The steps no n-gram covers (33-41% of the tokens) remain
  plain.
