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
    - The rest is serialization: the 2-row run's MXU sits idle for 1.59 M cycles, against
      0.07 M in the decode step. The gaps are after the QACTs of `_deltanet_rows`' `flush`
      (3 x 227K) and after `store_window`'s STs (2 x 133K).
    - The decode kernel `_deltanet_dstep` pipelines two pairs ahead: the MXU projects pair
      p + 2 while the DMA runs pair p's DSTEPs.
    - `_deltanet_rows` is one pair ahead (`NB` = 2 buffer sets) and runs 2 R DSTEPs per pair.
      Its out_proj flushes and window stores wait on them.
    - The two rows' DSTEPs take 2 x 0.65 M cycles (288 head steps of 2,248 cycles), about the
      MXU's 1.1 M for DeltaNet's weights. So with the same overlap, a 2-row run would cost
      about 1.05x a decode step (projected). That would also speed up Qwen3.5's prefill.
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
- **The hidden rows x** (before the final norm) go to DRAM for the MTP drafter, as `m.lm_split`
  stores x for the split generate programs.
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

The dataflow, in the published Qwen3-Next / DeepSeek-V3 form:

```
e = pre_fc_norm_embedding(embed(token at t + 1))
h = pre_fc_norm_hidden(hidden of the main model at t)
x = fc(concat(e, h))                       [2H] -> [H]
x = the MTP decoder layer at position t + 1 (its own KV cache)
draft for t + 2 = argmax(lm_head(norm(x)))
```

No reference implementation is installed here (transformers drops the weights; vLLM and
SGLang are not installed). **To confirm against vLLM's `qwen3_next_mtp.py` before building:**
the concat order (embedding first is the published form), whether h is taken before or after
the main model's final norm, the norm the draft's head uses (mtp.norm), and the MTP layer's
position.

On the card, per iteration:

- **The MTP layer runs over the verify run's rows**, (x_i, a_i) for i = 0..k. That keeps its KV
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

c_2 comes from section 1; c_draft is 0 for n-gram and about 0.05 for MTP with a 32K fp4 draft
head (10 MB layer + 17 MB head over 574 MB per 0.8B token).

| model, drafter | c_2 + c_draft | break-even a | a = 0.5 | a = 0.7 | a = 0.85 |
|---|---|---|---|---|---|
| Qwen3-0.6B, n-gram | 1.16 | 0.16 | 1.29x | 1.46x | 1.59x |
| LFM2.5-230M, n-gram | 1.09 | 0.09 | 1.38x | 1.56x | 1.70x |
| Qwen3.5-0.8B, MTP + 32K head | 1.28 | 0.28 | 1.17x | 1.32x | 1.44x |
| Qwen3.5-0.8B, MTP + 32K head, rows kernel overlapped (c_2 = 1.05, projected) | 1.10 | 0.10 | 1.36x | 1.55x | 1.68x |
| Qwen3.5-0.8B, n-gram | 1.23 | 0.23 | 1.22x | 1.38x | 1.50x |

All projected; a is unknown for our models until phase 0 measures it.
- The one published number is DeepSeek-V3's: 85-90% acceptance of its MTP's second token.
- n-gram's a counts only iterations that found a draft. The others cost exactly a plain step.
- Qwen3.5's per-row DeltaNet cost makes it the least favourable model per row today. Most of
  that cost is the rows kernel's missing overlap (section 1), which is a compiler change. A
  multi-token DSTEP (docs/stream.md 5.2: T tokens per state pass) would also cut the state
  traffic: the state is read once for both rows and written once per row.

## 8. Plan

0. **Acceptance study, no card.**
   - For Qwen3.5-0.8B and 4B, run the MTP head (from the checkpoint, in numpy or torch on
     omarchy) with the full head, the fp4 head and 16K / 32K / 64K draft heads.
   - For every model, run the n-gram drafter at g = 2 and 3.
   - Use the qual prompts plus a chat / code / summarization set, at the greedy and sampled
     defaults.
   - Output: a per model and drafter, and the choice of draft head.
1. **The rows kernel's DSTEP overlap for Qwen3.5** (section 1): `_deltanet_rows` pipelined
   as deep as `_deltanet_dstep`. It is a compiler change, with cycles measured as in section 1.
   It pays for prefill even without MTP.
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

- The reference details of the MTP dataflow (section 6.1): concat order, pre- or post-norm
  hidden, the MTP layer's position.
- Whether the 4B / 35B-A3B MTP layers are dense or MoE (the 0.8B's MLP is dense). Their
  checkpoints are on omarchy and opentpu, not on the Mac.
- The draft-head frequency table: from a tokenizer-level corpus count, or from the token ids'
  merge order (byte-level BPE ids roughly follow merge frequency). Phase 0 decides.
- Whether the plain-step fallback per iteration (no n-gram match) should instead keep a 2-row
  run with a guessed draft. The per-row cost (section 1) says no for every model measured.
