# Multi-token prediction on the card (design note)

Status, 2026-10-01:
- Phase 0, the offline acceptance study, is done (section 7.1).
- Phase 2, the k = 1 MTP loop for Qwen3.5 on the ISA simulator, is done (sections 9 and 9.1):
  greedy tokens equal plain decode's on the real 0.8B and 2B.
- The loop on the card (phase 3) is next. No RTL change is needed.

**Measured** means a full-model simulation of today's production configuration with LiteDRAM's
controller co-simulated (within 1.1% of the card, docs/board.md). **Projected** means
arithmetic on those measurements.

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
| DeltaNet state S (per head, [dv, dk] fp32; 18 MiB for 0.8B) | keep one per row | at k = 1, two slots per head, the parity c naming the committed one: row 0 steps slot c in place, row 1 steps from slot c into slot 1 - c; commit c ^= n (section 9). |
| The sampler's penalty factors (`pa`, `pb`) | tentative per row | row i applies the drafts d_1 .. d_i as if generated (section 5.2). Commit applies a_0 .. a_n. |

**The DeltaNet slots cost no bandwidth with DSTEP.** DSTEP already reads and writes the whole
state once per row (that is the +38 MB per row of section 1). Writing it to the next slot
instead of in place moves the same bytes. The only cost is k + 1 more copies of the state in
DRAM: 36 MiB more for 0.8B at k = 1, and more for the 4B / 9B / 35B states. DRAM has room
(4 GiB card); offload's slot planning has to count it.

**No RTL change.** DSTEP writes in place, but STREAM has `src` and `dst`, and the DMA already
writes a stream to its `dst` (`w3`; the RTL's subset check forbids only an overlap with `src`).
`tests/test_mtp.py::test_mtp_verify_and_draft_on_rtl` runs a verify run whose row 1 steps are
STREAMs into the other slot on the RTL, its DRAM equal to the ISA simulator's. The catch: src
and dst share one address register, so dst - src is a compile-time constant (section 9).

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

**Section 11 replaces this with an equivalent that is simpler.** Sample row i's token from p_i
with the position's one uniform, as the plain loop does, and accept when it equals d. The
acceptance probability is p_i(d), and a rejection's token follows the residual. It needs one
uniform per position, no residual draw, and no separate reference: the tokens equal the
plain sampled loop's.

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
2. **MTP for Qwen3.5** (plan item 3) is the payoff: 1.3-1.6x. Its DeltaNet state slots need a
   STREAM with dst != src, which the RTL already runs (section 4).
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
3. **The MTP drafter for Qwen3.5:** done in phase 2 (sections 9 and 9.1), host-driven.
   - The loader keeps `mtp.*` (`load_weights(path, mtp=True)`), and the image (`Spec.mtp`)
     holds the MTP layer and the draft head.
   - The kernel (`qwen35_mtp`) is checked against a numpy reference (`mtp_reference`).
   - The verify run with its state slots, and the loop itself, are `opentpu/llm/mtp.py`.
     Greedy tokens equal plain decode's on the 0.8B and 2B.
   - Still to do (phase 3): the programs at a `RunPos` per bucket, and the loop on the card
     (items 2 and 4).
4. **RTL:**
   - STREAM with dst != src: already in the RTL (section 4), no change;
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

## 9. Phase 2 design: the MTP loop on the ISA simulator

The decision after phase 0: Qwen3.5 gets the MTP drafter with the 32K fp4 draft head (v1).
n-gram waits. Phase 2 runs the k = 1 loop on the ISA simulator, driven by the host: each run is
a program compiled at its position, as `Engine.step`'s are. The same kernels become per-bucket
programs at a `RunPos` in phase 3 (section 3), with the loop on the card (section 2).

**Two programs per iteration**, at position p with the token t (at p) and the draft d (for
p + 1):

1. **verify(p, c)** is `qwen35_rows` over the rows (t, d) at p and p + 1, with logits for both
   rows. It also stores the final norm's output of both rows, the LM head's input, to `hid`
   for the MTP layer. c is the state parity (below).
2. **mtp(p)** runs over the rows (hid[0], a0) and (hid[1], a1) at positions p and p + 1:
   - e = `pre_fc_norm_embedding`(embed(a_r)) and h = `pre_fc_norm_hidden`(hid[r]);
   - x = fc([e, h]), one MM with K = 2H;
   - the MTP decoder layer, which is one more attention layer block in the image: gated
     attention with its own KV cache, then the dense MLP;
   - `mtp.norm`, then the draft head: fp4 rows of the 32K lowest ids, chunked as the LM head;
   - ARGMAX per row gives draft[r].

**The loop.**
- n = (d == a0). The iteration emits a0, then a1 if n is 1.
- A stop id among the emitted tokens ends the loop there.
- The next iteration's draft is draft[n], its position p + 1 + n and its token a_n.
- The MTP rows always run in pairs. When n = 0, row 1 sits on a rejected hidden. Its KV entry
  at p + 1 is overwritten by the next run's row 0 before anything attends to it.

**Rolling back: checkpoint, not recompute.**
- **The KV caches** (the model's and the MTP layer's) are positional and need nothing undone.
- **The DeltaNet state** gets two slots per head; c names the one holding the committed state.
  - The verify run's row 0 steps it in place in slot c. Row 1 is always the one that can be
    rejected. Its step is a STREAM from slot c into slot 1 - c, so slot c keeps S after row 0.
  - The commit is c ^= n.
  - This moves exactly the bytes of today's 2-row run: no extra cycles. It costs DRAM: 18 MiB
    more on the 0.8B and 2B, 48 MiB on the 4B.
- **The conv window** also gets two slots. The run stores the window after row 0 into W_c and
  the one after row 1 into W_(1-c). That is 3 more rows per pair: 1.3 MB per run on the 0.8B,
  about 10K cycles (0.2% of a token).
- **Recompute costs more.** A delta-rule step has no bit-exact inverse, so getting S after
  row 0 back needs the state from before the run kept anyway. On every rejection it also
  needs row 0's step run again:
  - 0.8B and 2B: 288 head steps of about 2,250 cycles each on the stream engine, 0.65 M
    cycles. At the 0.27 rejection rate that is 0.17 M per iteration: 3.5% of a 0.8B token,
    1.6% of a 2B token.
  - 4B: 768 steps, about 0.4 M per iteration.
- **Copying on rejection** costs about 0.08 M per iteration. That design keeps one fixed slot
  pair (row 0 B -> A, row 1 A -> B) and copies A -> B through the port after each rejection.
- **No RTL change.** Row 1's step needs dst != src. STREAM has both fields. The DMA already
  writes a stream to `w3`; the RTL's subset check only forbids src and dst overlapping. DSTEP
  stays in place.
- **One catch: two variants per program.** src and dst share one address register, so
  dst - src is a compile-time constant. The parity is therefore a compile-time choice: two
  verify programs, one per c. In phase 3 the loop CHAINs to the one c names (`HALT CHAIN`
  takes its target from a register).
- The decode step and prefill run at either parity through the same descriptors (the image's
  `slot`).

**Prefill.**
- The prompt runs in plain prefill's runs (`fit_chunk`), each storing `hid` for every row
  (row by row: one row of TMEM, so the runs fit as plain prefill's do).
- mtp then runs over the prompt's rows, (h_i, x_(i+1)) at positions i, in runs of up to 4
  rows. This fills the MTP layer's KV cache.
- The last row, (h_(P-1), a_0), gives the first draft.

**PAIR and exactness.** With PAIR and 4-bit weights, a decode step's MMs pair (column
reuse). A run's MMs pair only when 2R <= MCOLS. A paired MM adds each even K-block to the
odd one before the partial sums: the same products, summed in another order. So a prefill
run of more than MCOLS / 2 rows differs from the steps in the last bit of a sum. An int8
activation rounding downstream can turn that into a quantization step: about 1e-2 of the
largest logit on a tiny fp4 model (`test_prefill_pair_sum_order`).

Greedy MTP therefore gives plain greedy's tokens bit for bit only when two conditions hold:
- **The 2-row verify pairs as the steps do.** That needs MCOLS >= 4, the card's. MTPDecoder
  refuses PAIR with 4-bit weights below that.
- **The prompt runs in plain prefill's runs.** Plain prefill takes 4 rows per run on the
  0.8B and 2B and 3 on the 4B, at every position up to 4K, with or without the `hid`
  store.

**Tests.** Greedy tokens must equal plain greedy decode's, bit for bit:
- On tiny models (Mac) with three drafters:
  - the MTP head;
  - forced-right drafts (every draft accepted, so the parity flips every iteration);
  - forced-wrong drafts.
- After the loop, the committed slot equals plain decode's state, word for word.
- With int8 weights, and fp4 with PAIR at MCOLS 4. The prompt runs as plain prefill's
  runs.
- On the real 0.8B and 2B on omarchy, a few prompts.

**Measurements.**
- perf_qwen gets `--mtp verify` and `--mtp draft` at position 544. The verify run's c_2 is
  checked with `--check`, which also exercises the RTL's dst != src STREAM.
- The end-to-end speedup combines those cycles with the loop's own acceptance on the
  simulator.
- The 4B's verify run is extrapolated from 4 and 8 layers, as in section 1.

### 9.1 Phase 2 results

**Bit-exact on the real models.** `tools/mtp_decode.py` runs the loop on the ISA simulator.
- Setup: fp4 weights with the int8 head and the production flags (MCOLS 4, PAIR, DSTEP,
  STREAM), against plain greedy decode on the same simulator.
- Prompts: phase 0's prompts 0 (chat), 3 (code) and 7 (a summary of 237 tokens), 48 tokens
  each.
- The tokens are equal on every prompt.

| model | tokens equal | verify runs for 3 x 48 tokens | acceptance (chat / code / summary) | tokens per verify |
|---|---|---|---|---|
| Qwen3.5-0.8B | 3 of 3 | 84 | 0.66 / 0.74 / 0.71 | 1.68 |
| Qwen3.5-2B | 3 of 3 | 80 | 0.62 / 0.88 / 0.81 | 1.76 |

- **Against phase 0 on the same prompts.** Phase 0's float model accepted these shares of
  its first 46 drafts:
  - 0.8B: 0.76 / 0.87 / 0.65;
  - 2B: 0.80 / 0.89 / 0.74.

  The card's fp4 model writes its own text and drafts about as well: 0.70 on the 0.8B
  against 0.76, and 0.76 on the 2B against 0.81.
- **The 4B's loop was not run on the simulator.** The 2B's three prompts took 1.4 h of plain
  decode and 0.9 h of MTP decode there, and the 4B is twice the work. Its speedup below takes
  phase 0's greedy acceptance.

**The runs, co-simulated** (`perf_qwen --mtp verify / draft`, position 544, the LiteDRAM
controller at DDR3-1066, 133.33 MHz). The decode steps are section 1's: those programs are
byte-identical on this branch.

| model | decode step | verify run (2 rows, forked) | c_2 | draft run (MTP, 2 rows) | c_draft |
|---|---|---|---|---|---|
| Qwen3.5-0.8B | 4,943,542 | 5,914,147 | 1.196 | 255,753 | 0.052 |
| Qwen3.5-2B | 10,740,340 | 11,325,903 | 1.055 | 581,117 | 0.054 |
| Qwen3.5-4B | 22,090,946 | 23,494,861 | 1.064 | 932,708 | 0.042 |

- **The 4B's verify run is extrapolated** from 4 and 8 layers: 7,610,300 and 9,879,523
  cycles.
- **The fork costs almost nothing.** The verify run against section 1's plain 2-row run:
  - 0.8B: 5,914,147 against 5,962,906;
  - 2B: 11,325,903 against 11,327,655;
  - 4B: +0.2% at 4 layers and +0.3% at 8, so c_2 1.064 against 1.057.

  Row 1's steps into the other slot, the second window store and the hidden rows' store cost
  at most a fraction of a percent.
- **RTL check.** The 0.8B verify run passes `--check`: its DRAM after the RTL run equals the
  ISA simulator's, bit for bit, with the dst != src STREAMs.
- **c_draft is as section 6.1 estimated**: 0.05 of a token.

**End to end, in cycles.**
- The formula: (tokens - 1) x decode step / (verify runs x verify + MTP runs x draft). The
  first token comes from the prefill either way.
- Each iteration runs one verify and one draft, except the last, which needs no draft.

| model | speedup in cycles | phase 0 projected (greedy) | card today | projected on the card |
|---|---|---|---|---|
| Qwen3.5-0.8B | 1.347x (chat 1.30, code 1.40, summary 1.35) | 1.36x | 24.5 tok/s | 33.0 tok/s |
| Qwen3.5-2B | 1.593x (chat 1.46, code 1.70, summary 1.63) | 1.57x | 12.09 tok/s | 19.3 tok/s |
| Qwen3.5-4B (phase 0's acceptance) | 1.58x | 1.59x | 5.88 tok/s | 9.3 tok/s |

- **The phase 0 column**: section 7.1's loop over all nine prompts (256 tokens each), at that
  section's c_2 and c_draft.
- **With the measured costs instead**, the same loop gives 1.37x on the 0.8B, 1.57x on the 2B
  and 1.58x on the 4B (`mtp_accept.py --summary --c2 --cdraft`).
- **The card column** is today's device tok/s times the speedup in cycles.

**On the card** (2026-10-01; build B 79c5707a at 133.33 MHz, DDR3-1066; `tools/mtp_decode.py
--card`; the same three prompts, 48 tokens each).
- Plain greedy decode is the production loop (`generate_card`). The MTP loop is phase 2's,
  driven by the host.
- **Tokens:** equal on all six prompts, to plain greedy on the card and to the ISA simulator's.
- **Acceptance:** the simulator's, run for run.
- **Parities:** both exercised (verify runs per slot: 2B 37 / 43, 0.8B 40 / 44).

| model | plain, device tok/s | MTP, device tok/s | speedup (chat / code / summary) | co-simulated | c_2 card (co-sim) | c_draft card |
|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | 27.13 | 36.26 | 1.336x (1.29 / 1.39 / 1.33) | 1.347x | 1.207 (1.196) | 0.051 |
| Qwen3.5-2B | 12.19 | 19.41 | 1.593x (1.47 / 1.70 / 1.63) | 1.593x | 1.055 (1.055) | 0.054 |

Device tok/s counts the runs' CYCLES at the core clock, for the tokens after the prefill's.

**Wall tok/s is host-bound.** Each iteration compiles two programs on the host.
- 2B: plain 11.95 against MTP 7.09. The compiles are 11.8 s of the loop's 19.9 s; without
  them, 17.47.
- 0.8B: plain 25.80 against MTP 8.10. The compiles are 12.6 s of 17.4 s; without them, 29.42.

Phase 3's loop on the card removes both the compiles and the host round trips.

**Fit for phase 3** (compiled at positions 544 and 4094, board configuration, rows = 8):

| model | verify (instructions, TMEM words) | draft | decode step |
|---|---|---|---|
| Qwen3.5-0.8B | 2,362-2,562; 25.9K-29.7K | 464-666; 24.7K-29.7K | 1,798-1,900 |
| Qwen3.5-2B | 2,131-2,331; 32.1K-33.8K | 404-606; 28.8K-33.8K | 1,567-1,669 |
| Qwen3.5-4B | 2,694-3,100; 39.1K-44.1K | 730-1,132; 39.1K-44.1K | 1,720-1,930 |

IMEM holds 4,096 instructions and TMEM 64K words. The phase 3 programs are these kernels at a
`RunPos`, with the loop's few hundred instructions on top.

## 10. Phase 3 design: the loop on the card

The phase 2 kernels run at a run-time position and chain on the card. The host then only
prefills, writes the state block and reads `out[]`, as `generate_card` does.

**Programs per attention bucket b** (positions [t0, t0 + 256); compiled once per bucket,
in the chain area):

| program | rows | what it does | chains to |
|---|---|---|---|
| V[b][c] | (t, d) at p, p + 1, tpos <= 254 | the verify (`qwen35_rows` fork + hidden, slot c), an ARGMAX per row on the card: a0, a1; the emission; out[]; stop | D[b], or HALT |
| D[b] | (hid_r, a_r) at p, p + 1 | the MTP layer, draft[r]; the commit | V or E, by (bucket, parity) |
| E[b][c] | t at p = t0 + 255 (static) | the decode step (slot c), hidden, ARGMAX a0; out[]; stop | D1[b], or HALT |
| D1[b] | (hid_0, a0) at p | the MTP layer, one row; the commit | V[b + 1][c] |

- **Rows are kept in one bucket.** A verify at tpos = 255 would put row 1 in the next
  bucket. That position takes E instead.
  - E steps one row and keeps the MTP layer's KV cache whole: D1 runs its row.
  - Its draft d is dropped (one position in 256).
- **The parity is compile-time** (section 9), so V and E have one program per c.
- **Emission:** k = 1 + n, where n = (d == a0), but n counts only when tokens are left and
  a0 is not a stop id.
  - out[p + 1] = a0. With k = 2 also out[p + 2] = a1 (a LOOP of count k - 1).
  - Then left -= k. At a stop, the host's stop word, or no tokens left, the program commits
    the state and HALTs.
- **The commit** (D, D1; V or E on a halt):
  - tok = a_n, d = draft_n, c ^= n.
  - tpos += 1 + n. 256 wraps to the next bucket.
  - The chain offset into the table is 8 * (6 * wrap + 2 * [tpos == 255] + c), all fp32
    arithmetic on the state block.

**Kernels at a run-time position** (`qwen3.RunRows`, R rows at t0 + tpos + r):
- **Inputs:** the embeddings at run-time tokens: tok, and tok1 for row 1 (the draft in V,
  a1 in D). The RoPE rows come at pos.pos .. + R.
- **Attention:** the KV appends go one row per append (V^T tiles). Row r's mask is the
  bucket's, shifted by r positions.
- **DeltaNet:** `_deltanet_rows` needs nothing. At p >= K - 1 its window does not depend on
  the position (section 9).
- **The LM head rows** feed one Greedy sink per row. `qwen35_mtp` already ends in one.

**State block** (generate.py words, plus): d, a0, a1, n, c. Each program loads it and its
run-time arguments from it (`run_words`), and stores it before the CHAIN.

**Host:**
- `MTPDecoder.generate_card`: the prefill (phase 2: plain prefill's runs, then the MTP over
  them; a0 and the first draft).
- The state block, the chain area for the buckets reached, then a run from V or E.
- `run_generate` reads the tokens, 1 or 2 per iteration.
- The final state gives the slot and the position.

**Milestones:**
1. On the ISA simulator, the 2B's three prompts give plain greedy's tokens. Tiny models
   cross buckets with a small block and hit both parities, E, stops and max tokens.
2. A card session against `generate_card`: tokens equal, and tok/s on the 2B, 0.8B and,
   if it fits, the 4B.

### 10.1 Milestone 1: the loop on the ISA simulator

`tools/mtp_decode.py --loop device --no-plain --want <phase 2's tokens>` (fp4, int8 head,
cap 1024, the card's MCOLS 4 / PAIR / DSTEP / STREAM), 48 tokens per prompt. Every prompt
gives plain greedy's tokens:

| model | prompt | iterations / accepted | phase 2 (host loop) |
|---|---|---|---|
| 2B | 0 (chat, 30 tokens) | 29 / 18 | 29 / 18 |
| 2B | 3 (code, 35) | 25 / 22 | 25 / 22 |
| 2B | 7 (summary, 237) | 27 / 20 | 26 / 21 |
| 0.8B | 0 | 29 / 18 | 29 / 19 |
| 0.8B | 3 | 27 / 20 | 27 / 20 |
| 0.8B | 7 | 28 / 19 | 28 / 20 |

- Prompt 7 runs past position 256, so it covers E, D1 and the wrap into the next bucket.
- The loop accepts less than phase 2 in two places, both by design:
  - With one token left, n is 0 on the card. Phase 2 counted that iteration's draft.
  - On the 2B's prompt 7, E at position 255 verifies no draft, which costs one iteration.

The first run found a compiler bug: the verify's row 1 embedded a wrong row (at 16 DeltaNet
heads, the 0.8B's and 2B's). A run-time argument whose register an address had taken
moved into a released register at the first release, which was before row 1's token was
used. A moved argument now goes into a register at that register's own release
(`Builder._move_arg`). The tiny tests' 8 heads never move an argument, so
`test_mtp_loop_on_the_device_is_plain_greedy[kh16]` runs 16.

Programs at cap 4096 (instructions; every one fits its 8K slot):

| model | V | E | D | D1 |
|---|---|---|---|---|
| 0.8B | 2395-2688 | 1979-2125 | 423-715 | 336-482 |
| 2B | 2165-2458 | 1749-1895 | 363-655 | 276-422 |
| 4B | 2648-3240 | 2133-2431 | 613-1197 | 424-716 |

### 10.2 Milestone 2: the loop on the card

Card session 2026-10-01 19:50-19:59 (opentpu), build 72256074 (xfix: Gen1, 133.33 MHz,
DDR3-1066), tree dfac215. Setup as 10.1: phase 0's prompts 0, 3 and 7, 48 tokens each.
- Plain greedy is the production loop on the card (`Engine.generate_card`).
- MTP is `MTPDecoder.generate_card`: one device run per generation.

**Tokens:** every prompt on all three models gives plain greedy's tokens on the card. The 0.8B's and
2B's also equal the ISA simulator's (10.1).

| model | device tok/s, plain | device tok/s, MTP | speedup | acceptance | best prompt (code) |
|---|---|---|---|---|---|
| Qwen3.5-0.8B | 27.71 | 36.64 | 1.322x | 0.68 | 1.372x |
| Qwen3.5-2B | 12.44 | 19.62 | 1.577x | 0.74 | 1.703x |
| Qwen3.5-4B | 6.04 | 9.91 | 1.641x | 0.81 | 1.777x (acceptance 0.96) |

- The rates count decode after the prefill's token: 141 tokens per model, in the runs'
  cycles at the core clock.
- **The device speedup is phase 2's** (9.1, on build B): 1.336x on the 0.8B and 1.593x on
  the 2B. The loop's own work costs nothing measurable: the state block, the emission, the
  chain and E's one-row steps at the bucket ends.
- **Wall tok/s follows the device's.** The host's one cost is compiling a bucket's programs,
  once per engine (0.5-0.7 s).
  - The code prompt, with its programs already compiled: 37.72 / 21.13 / 10.72 tok/s, against
    the device's 38.03 / 21.19 / 10.73.
  - Over the three prompts: 28.67 / 16.73 / 8.97 tok/s, and 36.42 / 19.57 / 9.89 without the
    compiles. Plain greedy: 26.45 / 12.16 / 5.97.
  - Phase 2's host-driven loop was host-bound (a compile per iteration).

## 11. Phase 4 design: sampled decoding on the card

**The plain loop already samples on the card.** `generate.Sampler` (docs/autodecode.md,
"Sampling") picks each token with temperature, top-k (1 .. 64), top-p and the repetition
penalty. The host is out of the loop: before a run it writes one uniform per position
(`uni[]`, from the chat's seeded generator) and the penalty vectors, then only reads `out[]`.
It costs +0.7% a token at k 20, top-p 0.95, and +2.0% with a penalty (measured on the RTL). So
what phase 4 adds is the MTP loop's sampled mode: sampled chat on Qwen3.5 at MTP's speed.

### 11.1 Speculative sampling with a one-hot draft is "sample, then compare"

The MTP head's draft d is its argmax: a deterministic draft, so its distribution q is one-hot.
For a one-hot q the speculative sampling rule (section 5.2) is:
- accept d with probability p(d);
- on a rejection, emit a token from p with d removed, renormalized.

Here p is the target's processed distribution at that row.

Sampling a0 from p and accepting when a0 = d gives the same joint law:
- P(accept) = P(a0 = d) = p(d).
- Given a rejection, a0 follows p conditioned on a0 ≠ d, which is the residual.

So the verify's sampled mode is the greedy V of section 10 with `generate.Sampler` in place
of the `Greedy` sink per row:
- **Row 0** (input t at p): a0 = Sampler's pick with u[p + 1]. Accept n = (a0 == d), cut
  as today by the tokens left and a stop id.
- **Row 1** (input d at p + 1): a1 = Sampler's pick with u[p + 2]. It is emitted only when
  n = 1.
- **E** (the bucket's last position, one row): a0 with u[p + 1], n = 0.
- **D, D1**: unchanged. The MTP head's greedy draft from the rows' hidden and tokens, now
  a0 / a1 sampled.

**The tokens equal the plain sampled loop's for the same uniforms, bit for bit.**
- The token at position q is always `Sampler`'s pick of q's logits with u[q].
- With an accepted draft, row 1's logits are q = p + 2's logits after a0 = d. That is what
  plain decode computes.
- After a rejection the next iteration's row 0 recomputes position p + 2 from a0, with the
  same u[p + 2].
- The rows are bit-identical to decode steps (sections 5.1 and 9: the 2-row verify pairs as
  the step does).

So phase 4 keeps phase 3's test: tokens equal the plain sampled loop's
(`Engine.generate_card(sampling=...)`) on the ISA simulator, then on the card. That is a
stronger check than matching a distribution. Section 5.2's two-uniform scheme would need its
own numpy reference and a statistical test.

**Acceptance** is E[p(d)], the share phase 0 measured sampled (section 7.1, 32K head): 0.67 on
the 0.8B and 0.72 on the 2B, against 0.73 / 0.74 greedy. The 4B was not run sampled.

### 11.2 What the card needs

Every step maps to an instruction the ISA already has. **No RTL change.**

| step | instructions | where |
|---|---|---|
| each row's logits, chunk by chunk | the rows LM head feeding one sink per row | `_lm_head_rows` (`m.lm_sinks`, phase 3) |
| repetition penalty per chunk | LD `pa`, `pb`; VOP MUL, MUL, MIN | `Sampler.__call__` |
| logits to DRAM, block maxima | ST; VOP RMAX | `Sampler.__call__` |
| the top blocks, then the top k | VOP ARGMAX, RLD, FILL (knock-out), a register-addressed LD gather, in a LOOP of RLD count | `Sampler.token` |
| softmax | VOP SUB, MUL, EXP2 | `Sampler.token` |
| cumulative sums | VOP RDOT with a triangular ones matrix | `Sampler.token` |
| top-p and the pick | VOP MIN / MAX (step), SUM; LD of `uni[q]` | `Sampler.token` |
| the id | RLD of the pick's index, `_pick` | `Sampler.token` |
| accept | the greedy V's compare of a0 with d | `_verify_gen` |
| the penalty's context | ST of 1/R, R at the id's address (`Sampler.after`) for a0, and a1 when n = 1 | the commit |

**Two things are new, and both are small.**
- **Row 1's penalty must count d.** Both rows share `pa` / `pb`, the context up to t, and row
  0 must not see d.
  - After the head, row 1's stored logit x of d stays when d is in the context already
    (pb[d] = R), else becomes min(x / R, x * R). A select by MIN / MAX against +-2^127
    picks between the two exactly; scaling x by R and back would round (section 11.5).
  - Then the maximum of d's 64-wide block is taken again.
  - In all: an RLD of d's address, two LDs, about ten one-word VOPs, an ST, an LD of the
    block and an RMAX.
- **Each row needs its own logits buffer.** `lg` becomes two, or a second one beside it in
  the MTP area (1 MB each for a 248K vocabulary).

**The uniforms stay on the host.** The run writes u[p + 1 .. p + n] once, as the plain loop
does: 4 bytes a position, 16 KB for a 4096-token reply. A device RNG would not take the host
out of anything more. The sequencer could run a 32-bit LCG in a register (RLD MUL, then a
1-wide ARGMAX writes i2f of the register into TMEM). But its state does not survive a chain
(an fp32 word keeps 24 bits), and a counter-based generator wants integer XOR and shifts that
the VPU does not have. A good device RNG means an RTL op. It is not needed for chat, and the
host's uniforms keep every run reproducible against the ISA simulator.

### 11.3 Cost

Per verify iteration, against greedy MTP:
- **The serial part:** the top-k loops run twice instead of the two ARGMAX picks. That is
  about 17K cycles a row at k 20 (docs/autodecode.md).
- **The per-chunk work:** the stores and block maxima hide under the LM head's weight
  stream, as they do for one row.
- **The logits:** 2 MB of stores (two rows of 248K fp32).
- **With the penalty:** `pa` and `pb` are read once per chunk for both rows (one load, two
  uses), 2 MB.

| model | MTP iteration (cycles, M2) | sampler, 2 rows | with the penalty |
|---|---|---|---|
| Qwen3.5-0.8B | 6.1 M | +34K (+0.6%) | about +70K (+1.1%) |
| Qwen3.5-2B | 11.8 M | +34K (+0.3%) | about +70K (+0.6%) |

The plain sampled loop pays about 17K a token itself, so against plain sampled decode the
difference is under 0.1%.

**Projected device speedup, sampled.** Take M2's iteration cost in decode steps, C = (1 + a) /
speedup (0.8B 1.27, 2B 1.10, 4B 1.10). Then speedup = (1 + a_sampled) / C, with a_sampled =
M2's greedy acceptance times phase 0's sampled / greedy ratio:

| model | greedy (M2) | sampled, projected |
|---|---|---|
| Qwen3.5-0.8B | 1.32x, a 0.68 | about 1.28x, a 0.62 |
| Qwen3.5-2B | 1.58x, a 0.74 | about 1.56x, a 0.72 |
| Qwen3.5-4B | 1.64x, a 0.81 | about 1.6x (no sampled phase 0) |

### 11.4 Plan

1. **ISA simulator.**
   - Changes: `_verify_gen` and E with `Sampler` sinks; the second `lg`; row 1's penalty
     fix-up.
   - The chain area: one table per mode. A sampled bucket holds V, E, D and D1 compiled for
     the run's `Sampling.key`, as the plain loop's chain area keeps greedy and sampled apart.
   - `MTPDecoder.generate_card(sampling=, rng=)` writes the uniforms and the penalty vectors.
   - Tests:
     - tokens equal `Engine.generate_card(sampling=...)` for the same uniforms, on the tiny
       Qwen3.5;
     - at 16 heads (kh16);
     - with and without the penalty (a penalty case where d is new to the context, and one
       where it is not);
     - across a bucket end;
     - at a stop id.
   - Use the varied tiny models (test_autodecode's `_tiny`, initializer_range 0.2): the tied
     8-head one mostly echoes its input token (phase 3's lesson).
2. **RTL:** one sampled loop across a bucket end on the board configuration, DRAM equal to the
   ISA simulator's, as `test_mtp_loop_on_rtl` does for greedy.
3. **Card session:**
   - tokens equal the plain sampled loop's on the card, for the same seed: 0.8B, 2B, 4B, three
     prompts;
   - device tok/s against plain sampled decode;
   - acceptance against phase 0's sampled shares.
4. **otpu-chat:** `Chat` runs Qwen3.5 turns through `MTPDecoder` (greedy and sampled), with
   TTFT and tok/s on the status line.

The n-gram drafter (section 6.2) is one-hot too, so it would take the same sampled mode.

### 11.5 Phase 4, step 1: the sampled loop on the ISA simulator and the RTL

**What changed.**
- `_verify_gen(samp=)` gives each row its own `generate.Sampler`:
  - row 0 uses u[p + 1] and the loop's `lg`;
  - row 1 uses u[p + 2] and the second `lg`, in the MTP area (`gen_alloc`; `gen_build` pads
    it with -inf as `lg` is);
  - `after(a0)` runs at once, and `after(a1)` inside the accepted branch;
  - E has one Sampler.
- **Row 1 reuses row 0's penalty factors** (`Sampler(pen_from=)`). Each chunk loads `pa` and
  `pb` once. Each row then writes l * pa to its own tile and l * pb in place over the head's
  output, then takes the MIN. That is 8 instructions a chunk for both rows instead of 10, so
  a 4B verify is 122 instructions shorter (61 chunks at 2 rows). The plain loop's Sampler is
  unchanged: the plain sampled generate programs hash the same before and after (0.8B, three
  samplings, buckets 1 and 3).
- **Row 1's draft penalty is exact** (`_penalize_draft`). With the stored logit x of d and
  y = min(x / R, x * R):
  - s = +2^127 when pb[d] > 1 (d in the context already), else -2^127;
  - the result is max(min(y, -s), min(x, s)), which is x or y exactly for finite logits.
  - 11.2's first form, min(x * pb[d] / R, x * pa[d] * R), rounds when d is in the context:
    x * R / R is not always x in fp32.
- **The host side.**
  - `MTPDecoder.generate_card(sampling=, rng=, context=)` draws a0 =
    `reference_pick(prefill logits)` with the generator's first uniform.
  - `Engine._generate_inputs` then writes u[P + 1 .. P + n] and the penalty vectors over the
    context and a0.
  - u[P + n + 1] = 0.5 is a stand-in for the last verify's row 1. That row is never emitted,
    and the stand-in is not drawn from the generator, which stays aligned with plain decode's.
- **Programs.** The chain area is rewritten when the mode (forced, `Sampling.key`) changes,
  and the program cache keys V and E by `Sampling.key`.
- **The tool.** `tools/mtp_decode.py --loop device --sample T,K,P,R --seed S` runs plain
  sampled decode and the sampled MTP loop from the same uniforms and checks that their
  tokens are equal.

**Tests on the ISA simulator** (`test_mtp_sampled_loop_is_plain_sampling`, 8 cases, all
pass). Each case checks two things against plain sampled decode for the same uniforms
(`reference_pick` on resident decode's logits): the tokens are equal, and the committed
DeltaNet states and windows are equal word for word. The cases:
- across bucket 256 with the right drafts, every third wrong, and the MTP's own drafts (E at
  255);
- with no penalty (T 1.5, top-k 20);
- greedy with penalty 3.0;
- the varied model with penalty 1.1;
- a stop id;
- 16 DeltaNet heads.

**The model choice matters for the penalty.** The varied tiny model (initializer range 0.2)
never let the penalty of the token before decide a pick in 16 tokens, at penalties 1.1 to
3.0, because its next token rarely repeats the last. The tied model (0.02) mostly echoes its
input, so the draft's penalty decides 7 picks in 16 tokens at penalty 1.5 (T 0.8, top-k 5,
top-p 0.9) and 14 at greedy 3.0. Those are the penalty cases. With `_penalize_draft` turned off,
both fail; with it, they pass.

**RTL.** `test_mtp_sampled_loop_on_rtl` (Verilator, the board's memory path) starts from a
251-token prefill with a0 sampled. It runs one sampled loop with penalty 1.5 across the
bucket end, every third draft wrong. The tokens and the image's DRAM equal the ISA
simulator's. The greedy `test_mtp_loop_on_rtl` passes too.

**IMEM on the real layouts** (fp4, cap 4096). Each cell is V's instructions at bucket 1 / 16;
IMEM holds 4096.

| model | greedy | T 0.7, top-k 20, top-p 0.8 (chat) | top-k 64, penalty 1.1 |
|---|---|---|---|
| Qwen3.5-0.8B | 2395 / 2687 | 2625 / 2917 | 3150 / 3442 |
| Qwen3.5-2B | 2165 / 2457 | 2395 / 2687 | 2920 / 3212 |
| Qwen3.5-4B | 2648 / 3239 | 2879 / 3469 | 3404 / 3994 |

- E is shorter in every case: at most 2679, the 4B at top-k 64 with the penalty.
- Before row 1 shared the penalty factors, the 4B's bucket-16 V at top-k 64 with the
  penalty was 4116 instructions, so it did not fit.
- A V that does not fit raises `CompileError` (`_gen_programs`). A caller then falls back to
  plain sampled decode.

### 11.6 Phase 4, step 4: otpu-chat (ISA simulator)

A chat needs the context to go on after a reply: the next turn's prefill and `/continue`
start where the loop stopped. Two changes make that work.

**The loop's last D runs before the HALT.**
- V no longer halts. It records the end in a state word (S_END: a stop id, the host's stop
  word, or no tokens left) and chains to D or D1 as always.
- D halts instead of chaining when S_END is set.
- The MTP layer's KV cache then holds every position before Engine.pos, the pair (h, the last
  token) included. `MTPDecoder.draft` holds the draft after the last token.
- Before, the last iteration's D did not run. Its KV position kept an older iteration's
  values, so a next turn drafted from a wrong cache: still the same tokens, but fewer
  accepted drafts.
- The cost is one D run a reply.

**`MTPDecoder.prefill` continues from Engine.pos.**
- It feeds the new tokens with the committed state slot and pairs them in the MTP layer
  (h_q, x_(q + 1)) as a first prefill does.
- `on_run` reports each run and can stop the prefill between runs. The pairs then end at the
  next prompt token, so a later prefill continues.

**Chat.**
- `Chat` on an MTP engine (`otpu-chat --mtp`, `mtp_engine` with `load_weights(mtp=True)`)
  prefills through the MTP decoder. The host's sampler picks the first token inside it,
  because the draft after that token needs it.
- Each reply is one `loop_card` run from that token and `MTPDecoder.draft`, polled with the
  chat's stop.
- `/continue` resumes from the cut reply's last token and the kept draft.
- The plain status line gains the iterations' acceptance.
- Settings the device sampler does not take decode on the host without the drafter, as they
  do on a plain engine.

**Tests on the ISA simulator.**
- `test_mtp_continues_its_context` (greedy and sampled, penalty 1.1): two turns on one
  decoder against one fresh decoder over the whole conversation. Equal: the tokens, the final
  draft, and the MTP layer's KV cache below Engine.pos, bit for bit.
- Negative control: without the last D, the KV cache's K differs at the position before the
  first turn's last token (position 50), and the draft after the turn differs (229 against
  463).
- `test_chat_with_mtp` (greedy, and sampled with seed 3 and penalty 1.1): a turn cut at
  max_new, `/continue`, then a second turn. The replies, the fed tokens, the context and the
  token counts equal plain Chat's, and the second turn does not restart the context.
- The phase 3 and 4 loop tests pass with the last D (test_mtp: 36 passed).

**RTL.** With the last D, `test_mtp_loop_on_rtl` and `test_mtp_sampled_loop_on_rtl` pass
(omarchy, Verilator).

**Real models on the ISA simulator.** `tools/mtp_decode.py --loop device --sample ...`, seed 0,
prompts 0, 3 and 7, 48 tokens: sampled MTP's tokens equal plain sampled decode's.
- The 2B at chat's default (T 0.7, top-k 20, top-p 0.8): 3/3.
- The 2B with penalty 1.1: 2/2. Prompt 7 was stopped by omarchy's memory guard.
- The 0.8B at chat's default: 3/3. Its acceptance per prompt, 0.62 / 0.68 / 0.47, equals the
  card's below.

### 11.7 Phase 4 on the card

Production build e4db91c9 (Gen2 x8, pa), 133.33 MHz, fp4 with an int8 head, cap 1024, phase
0's prompts 0, 3 and 7, 48 tokens, seed 0, one otpu-lock window (2026-10-02 02:58-03:17
opentpu).

**Sampled, chat's default (T 0.7, top-k 20, top-p 0.8): 9/9 prompts give plain sampled
decode's tokens.** The 2B with penalty 1.1 gives them too, 3/3. Device tok/s over the 141
tokens after the first:

| model | plain sampled | MTP sampled | speedup | projected (11.3) | acceptance |
|---|---|---|---|---|---|
| Qwen3.5-0.8B | 27.55 | 34.21 | 1.242x | about 1.28x | 0.58 |
| Qwen3.5-2B | 12.41 | 19.49 | 1.571x | about 1.56x | 0.74 |
| Qwen3.5-4B | 6.03 | 9.74 | 1.616x | about 1.6x | 0.78 |
| Qwen3.5-2B, penalty 1.1 | 12.37 | 19.45 | 1.572x | | 0.74 |

- The 0.8B lands under its projection because its sampled acceptance on these prompts is
  0.58, against the 0.62 projected (phase 0's sampled share applied to M2's 0.68). The 2B and
  4B meet or pass theirs.
- The sampler costs little. Plain sampled decode runs within 0.6% of plain greedy (27.55 /
  12.41 / 6.03 against 27.72 / 12.44 / 6.04 tok/s). The penalty costs the 2B's MTP loop 0.2%.

**The program cache (1a): first and second token times,** greedy, prompt 0, the card's wall
clock in seconds. Cold is a fresh `OTPU_PROG_CACHE` directory; warm is the next process with
it.

| model | plain TTFT | plain t2 - TTFT, cold / warm | MTP TTFT | MTP t2 - TTFT, cold / warm |
|---|---|---|---|---|
| Qwen3.5-0.8B | 1.17 | 0.179 / 0.058 | 2.33 | 0.586 / 0.113 |
| Qwen3.5-2B | 1.40 | 0.211 / 0.099 | 2.79 | 0.612 / 0.146 |
| Qwen3.5-4B | 3.23 | 0.277 / 0.175 | 5.41 | 0.909 / 0.252 |

- The cache takes the bucket compiles out of the second token: 0.10-0.12 s for plain decode's
  program, and 0.47-0.66 s for MTP's six.
- Warm, the second token comes one device iteration after the first (a step: 36 / 80 / 166 ms;
  an MTP iteration: about 46 / 88 / 183 ms), plus 0.01-0.02 s for plain and 0.06-0.07 s for
  MTP. MTP's extra is the host setting up the run, which includes writing its six programs to
  the card.
- TTFT does not change; it is the prefill's. The MTP prefill compiles its runs on the host one
  after another (1.85-2.9 s, against plain prefill's pipelined compile), which puts MTP's TTFT
  1.2-2.2 s above plain's. (1b), run-time-position prefill programs, takes that out.

**otpu-chat on the card** (the 2B, fp4, sampled at chat's default, seed 0, a 23-token prompt,
64 tokens). `--mtp` gives plain chat's reply, compared byte for byte.

| | TTFT | decode tok/s, wall (device) |
|---|---|---|
| plain | 1.32 s | 12.15 (12.4) |
| `--mtp` | 2.23 s | 15.73 (18.1), acceptance 0.62 |

The wall rate trails the device's by MTP's bucket compiles. otpu-chat does not turn the program
cache on, so a cold process pays them in its first reply.

## 12. Open questions

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
