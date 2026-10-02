# Prefill at run-time positions

**Goal:** a prompt's prefill from programs compiled once per attention bucket, not once per
prompt. Each prefill run currently bakes its positions and token ids into its program, so the
host compiles every run of every prompt. With the prompt in DRAM and the position as a run-time
value, the programs repeat across prompts, the program cache keeps them, and TTFT falls to the
device's prefill time. This is (1b) in the MTP plan (docs/mtp.md 11.7).

## 1. Where TTFT goes today

The card, e4db91c9, 133.33 MHz, fp4 with an int8 head, phase 0's prompts, wall seconds
(docs/mtp.md 11.7):

| model | prompt | runs | device prefill | plain TTFT | MTP TTFT | MTP's host compile |
|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | 30 | 8 | 0.40 | 1.14 | 2.31 | 1.82 |
| Qwen3.5-0.8B | 237 | 60 | 3.09 (+0.20 MTP layer) | 4.67 | 16.92 | 13.22 |
| Qwen3.5-2B | 30 | 8 | 0.74 | 1.44 | 2.81 | 1.95 |
| Qwen3.5-2B | 237 | 60 | 5.63 (+0.47) | 6.15 | 20.04 | 13.55 |
| Qwen3.5-4B | 30 | 10 | 2.36 | 3.24 | 5.49 | 2.91 |
| Qwen3.5-4B | 237 | 79 | 18.39 (+0.99) | 18.73 | 39.32 | 19.33 |

- **The device's part.** A prefill run streams every weight once for its R rows. One MXU pass
  takes MCOLS = 4 rows, so a run costs about a decode step: 6.9 / 12.5 / 31 M cycles. The
  0.8B and 2B take R = 4, and the 4B R = 3: R = 4 does not fit its TMEM (fit_chunk).
- **Plain prefill** compiles each run while the device runs the one before (Engine's worker
  pipeline). Short prompts still wait 0.6-0.9 s: the first run's compile, and R = 8 tried
  first, which fails TMEM.
- **MTP's prefill** compiles its runs one after the other on the host: a rows run and an MTP
  layer run per chunk, 0.1-0.25 s each on the card host. For a 237-token prompt that is
  13-19 s.

## 2. What exists

- `RunPos` / `RunRows` (qwen3.py): rows at a run-time position of a bucket, with RunPos.offset
  (gemma4: row r at tpos + r with its own mask row) and `bucket_row`.
- `Image.compile_rows_run` / `compile_mtp_run` (qwen35.py, gemma4): qwen35_rows and
  qwen35_mtp over RunRows, with the tokens as run arguments.
- `compile_layer_run` / `qwen35_embed_run` / `qwen35_prefill_head`: layer-major runs for the
  MoE models. Those stay as they are.
- The generate loop's state block, `run_words` (arguments from state words, RLD MUL) and
  HALT CHAIN; the program cache.

**But `compile_rows_run` does not compile on the real layouts.** On the 0.8B (cap 4096,
buckets 1, 2 and 16, `hidden=True`), R = 1 or 2 runs out of address registers, and R = 4 or 8
needs more than 8 run-time argument values. A decode bucket's arguments are already 7 (0.8B):
the token (x 4096, its embedding row) and tpos at six coefficients (128, 256, 8, 1, 4, -4: K
and its scales, V^T's column, V's scale, RoPE, the mask). R tokens as arguments make that
R + 6, so only R = 1 or 2 could ever fit, and even those leave the 16 DeltaNet heads no
registers.

## 3. Design

**The prompt's tokens go in DRAM, not in the arguments.** `out[]`, the generate loop's
token-per-position array, holds the token at each position q. The host writes the turn's new
tokens to out[p0 .. P) before the prefill. Each run then loads its R tokens, out[tpos + r], into
TMEM with one LD at a run-time address. Each embedding gather takes its row address from a
scratch register (RLD MUL, the row's bytes), released after the gather. The arguments are then
tpos's six coefficients for every R: R rows share one tpos, at r-row offsets as immediates
(RunPos.offset). The port-A rule is kept: out[] and the state are read by LD (port B), and since
4e0b866 port A is flushed at every run start anyway.

**Programs per bucket** (one image; R_max = 4, or 3 on the 4B):

| kind | rows | logits | count |
|---|---|---|---|
| P_r | r = 1 .. R_max, at tpos <= block - r | none | R_max |
| L_r | r = 1 .. R_max, the prompt's last run | last row's, then the first pick | R_max |
| (MTP image) M_r | the MTP layer over the run's rows (h_q, x_(q+1)) | the draft | R_max |

- That is 8 programs per bucket (12 with MTP), compiled when first used and then kept by the
  program cache. A typical prompt uses P_Rmax, one L_r, and a P_r where a run would cross the
  bucket's end. At cap 4096: 16 buckets, at most 128 / 192 programs, each 0.1-0.25 s once.
- **Positions 0 .. conv_k - 2** (a new conversation's first rows) stay at compile time:
  `_deltanet_rows`' taps take K - 1 as their run-time stand-in, as gemma4 noted. That is one
  head run H_r per R at p0 = 0, with run-time tokens, cached like the rest.
- **The runs' split** depends only on (p0, P, R_max, block): runs of R_max rows, cut at the
  bucket's end, the last one L. Plain and MTP engines get the same runs, so they get the
  same logits bit for bit, as today's fit_chunk gives both.

**MTP shares it.** The MTP image's P_r / L_r are the same kernels with `hidden=True` (the
rows' hidden to `hid`). After each rows run, M_r runs the MTP layer over the same rows. Its
tokens x_(q + 1) come from out[q + 1]. For the last run, x_P is the first token: the device's
pick, or the host's, written to out[P] before M_r. The last M_r's draft is the loop's first d,
as MTPDecoder.prefill gives today.

**Chaining (the target): one device run per turn.**
- Like the generate loop, every prefill program loads a state block (tpos, the rows left,
  the run's kind) and takes its arguments from it (`run_words`).
- At its end it advances tpos by R and HALT CHAINs to the next run's program: in this bucket
  or the next, P or L by the rows left, R = min(R_max, rows left, block - tpos). The bucket's
  table (like MTP's chain table) holds the programs' addresses.
- L_r picks the first token on the device: ARGMAX, or the Sampler with u[P]. It writes it to
  out[P], then chains into the generate loop's program at P (plain), or into M_r, then V or E
  (MTP).
- So a turn is one run, prefill and reply together. TTFT is the device's prefill plus the
  start, and the host streams out[] from the first token on, as it does today.
- otpu-chat's progress line reads the state's tpos while the run goes.
- **Checkpoint.** The same programs started one run at a time by the host (arguments per
  run), which is what `compile_rows_run`'s interface already is. This comes first; the chain
  goes on top once the programs are bit-exact. The first token then stays the host's pick.

**Models.** First the Qwen3.5 dense models, 0.8B / 2B / 4B, plain and MTP. Then Qwen3 and
LFM2 (qwen3_rows, lfm2's rows: RunPos with K - 1 static rows for LFM2's convolutions as well),
and Gemma 4 E2B (its rows kernel and per-row PLE rows from the gather the generate loop
already does).

The MoE models keep gemma4's layer-major runs, which are already at run-time positions
(`prefill_layers`, `compile_layer_run`, compiled in the worker processes ahead of their runs).
The new route takes `prefill_chunks` for dense images only (`spec.moe is None`). A MoE prompt's
rows below conv_k - 1 go through `prefill_chunks` as token steps (docs/offload.md 13.6, the
port-A workaround's default), and those stay as they are.

**Shared code, as agreed with gemma4.**
- This work changes: `_inputs_rows`' RunRows branch (the token source), `RunRows` (an out[]
  source beside the toks run variables), a helper for `_embed` / `_gather` with a token held in
  TMEM, the dense route of `prefill_chunks`, and `compile_rows_run` / `compile_mtp_run` /
  `_run_rows`.
- It leaves alone: RunPos.offset, `bucket_row`, `_attention_rows`, `prefill_layers` and its
  workers, `compile_layer_run`, and `qwen35_layer_run` / `_embed_run` / `_prefill_head`.

## 4. Expected TTFT

Device prefill as measured (the same runs, the same programs' arithmetic), plus about 0.05 s
of start-up per turn when chained; once per process and bucket, the cold compiles of 2-4
programs come on top (0.2-0.8 s, then the disk cache):

| model | prompt | plain now | plain (1b) | MTP now | MTP (1b) |
|---|---|---|---|---|---|
| Qwen3.5-0.8B | 30 | 1.14 | ~0.45 | 2.31 | ~0.48 |
| Qwen3.5-0.8B | 237 | 4.67 | ~3.1 | 16.92 | ~3.3 |
| Qwen3.5-2B | 30 | 1.44 | ~0.8 | 2.81 | ~0.85 |
| Qwen3.5-2B | 237 | 6.15 | ~5.7 | 20.04 | ~6.2 |
| Qwen3.5-4B | 30 | 3.24 | ~2.4 | 5.49 | ~2.5 |
| Qwen3.5-4B | 237 | 18.73 | ~18.4 | 39.32 | ~19.4 |

- A chat's later turns feed only what the template added, often 10-40 tokens: their TTFT
  becomes 0.2-0.8 s (0.8B, 2B).
- The 4B stays device-bound. Fitting R = 4 rows (one more MXU column's worth of TMEM) would
  take its prefill from 79 runs to 60 for 237 tokens, about -24%. That is a separate item.

## 5. Plan

1. **Tokens from out[]** in qwen35_rows / qwen35_mtp over RunRows. Tests on the tiny Qwen3.5
   at 16 heads (kh16) and on the real layouts' compile: P_r / L_r / M_r for r = 1 .. R_max,
   buckets 1, 2 and 16, within 8 arguments and IMEM.
2. **Host-started runs** (Engine.prefill and MTPDecoder.prefill through them, cached). The
   logits and the KV cache, DeltaNet states and windows must equal today's fit_chunk prefill
   bit for bit, plain and MTP, from p0 = 0 and from a later p0 (a chat's next turn), across a
   bucket's end.
3. **The chain**: state block, the table, L into the generate loop or MTP's V. Tokens equal to
   step 2's runs, greedy and sampled; RTL across a bucket's end.
4. **Card**: TTFT for 0.8B / 2B / 4B, plain and MTP, prompts 30 and 237, against section 4;
   otpu-chat's TTFT on its line.
5. Qwen3, LFM2, Gemma 4 E2B.

Effort: steps 1 and 2 (the checkpoint, which already takes the compiles out of TTFT) about 1.5
days; the chain about 1 more day, then RTL and a card session.

**Risks.**
- The address registers of the 16-head rows kernel at a run-time position: MTP's V fit only
  with `_move_arg` (docs/mtp.md 10.1). With the token arguments gone it gains two or more
  registers, but this needs checking on the real layouts first (step 1).
- R_max is per bucket (its L program's fit; prefill.r_max), since a bucket's programs are all
  at one R and a later bucket's attention makes a longer program. The Qwen3.5 models take the
  same R in every bucket (4 / 4 / 3; the 4B's R = 4 runs out of TMEM, a separate item);
  Phi-4-mini's mix takes 3 rows in bucket 1 and 1 in bucket 16, as fit_chunk's runs shrink
  with the context today. Plain and MTP share it, so they split alike.
- The gate keeps test_qwen35_moe's layer-major tests (test_layer_major_prefill_is_bit_exact,
  test_layer_major_runs_compile_in_the_worker_processes), so the MoE path is shown untouched.

## 6. Measured: host-started runs, and the chain parked

Steps 1 and 2 are in (`opentpu/llm/prefill.py`, `qwen35_prompt_run`, tests/test_prefill.py).
The engine keeps each prompt program (on the card assembled once); `Engine(prompt_runs=True)`
and `MTPDecoder.prefill` use them, otpu-chat by default (`--no-prompt-runs`: the old route).
On the ISA simulator the logits and every slice's DRAM equal compile-time runs of the same split
word for word, and with int8 weights today's prefill too. Real layouts (cap 4096, fp4 and int8):
every program fits IMEM; R_max 4 / 4 / 3.

**Card** (pa e4db91c9, 133.33 MHz, 2026-10-02 07:18-07:37 opentpu; fp4, int8 head; tree
76ef74b; `mtp_decode --loop device --prompt-runs`, prompts 0 and 7; MTP's tokens equal plain's in
every run; with prompt runs on and off the tokens are equal for all three models, both prompts,
plain and MTP). Wall seconds to the first token, warm program cache; "device" is the prefill
runs' cycles (MTP: rows and MTP layer runs):

| model | prompt | plain old | plain now | device | MTP old | MTP now | device |
|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | 30 | 1.280 | 0.604 | 0.405 | 2.410 | 0.557 | 0.431 |
| Qwen3.5-0.8B | 237 | 4.594 | 3.146 | 3.096 | 16.622 | 3.435 | 3.302 |
| Qwen3.5-2B | 30 | 1.400 | 0.959 | 0.743 | 2.767 | 0.898 | 0.803 |
| Qwen3.5-2B | 237 | 6.084 | 5.689 | 5.645 | 20.029 | 6.247 | 6.114 |
| Qwen3.5-4B | 30 | 3.224 | 2.905 | 2.364 | 5.442 | 3.016 | 2.488 |
| Qwen3.5-4B | 237 | 18.643 | 18.478 | 18.425 | 38.705 | 19.577 | 19.413 |

- **Per run, the host adds 0.5-0.8 ms (plain) and 1.0-1.1 ms (MTP: its rows and MTP layer
  runs alternate, each loading its program).** On the 237-token prompt (60 / 79 runs, MTP twice
  that) that is 0.3-1.6% of TTFT plain and 0.8-3.9% MTP (0.13 s, the 0.8B's MTP).
- **A process's first prompt pays 0.1-0.5 s more** (prompt 0 above: 0.20 / 0.22 / 0.54 s
  plain): its programs read from the disk cache, the engine's compile workers still starting
  beside it, and on the 4B R_max's probe of R = 4 (traced until TMEM runs out, every process).
  The chain would not remove it: its programs load the same way. A warm-up of bucket 1's
  prompt programs when the engine starts (or a kept R_max) would.
- Cold (a new program cache), the first prompt also compiles its 4-6 programs: 0.87-1.35 s.
- otpu-chat on the 2B (23 tokens, cold cache): plain and `--mtp` replies byte-equal, TTFT 1.41 /
  1.68 s.

**The chain is parked.** Host-started runs cost at most ~4% of TTFT (the 0.8B's MTP), at most
0.16 s on any prompt measured, and under 2% for plain prefill; the chain (state block, chain
table, the first pick on the device, RTL and a card session) would save only that. Its design
above stays the plan if prefill runs get much shorter (more rows per weight pass, a faster
link to the host's pick) or the host's per-run cost grows. Next: the other dense models
(step 5).

## 7. Prompt runs against today's rows

A run's device time is about its weight passes, ceil(rows / MCOLS): each pass streams every
weight. A bucket where prompt runs took fewer rows than today's prefill (fit_chunk's
compile_rows runs) at the same positions would stream the weights more often. Compiled on
the real layouts (omarchy, compile only; cap 4096, block 256, MCOLS 4, image rows 8; fp4 and
the mixes with an int8 head), the rows of a run in buckets 1 and 16 and the weight passes of
three prompts (P tokens from p0), prompt runs / today's:

| model | format | rows, bucket 1 | rows, bucket 16 | p0 0, P 30 | p0 3850, P 30 | p0 0, P 237 |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | int8 | 4 / 8 | 4 / 4 | 8 / 8 | 8 / 8 | 60 / 60 |
| SmolLM3-3B | int8 | 4 / 4 | 4 / 4 | 8 / 8 | 8 / 8 | 60 / 60 |
| SmolLM3-3B | mix | 4 / 4 | 3 / 3 | 8 / 8 | 10 / 10 | 60 / 60 |
| Phi-4-mini | int8 | 3 / 3 | 3 / 3 | 10 / 10 | 10 / 10 | 79 / 79 |
| Phi-4-mini | mix | 3 / 3 | 1 / 1 | 10 / 10 | 30 / 30 | 79 / 79 |
| LFM2.5-230M | int8 | 4 / 8 | 4 / 4 | 8 / 8 | 8 / 8 | 60 / 60 |
| LFM2-2.6B | int8 | 4 / 4 | 1 / 1 | 8 / 8 | 30 / 30 | 60 / 60 |
| Qwen3.5-0.8B | fp4, int8 | 4 / 4 | 4 / 4 | 8 / 8 | 8 / 8 | 60 / 60 |
| Qwen3.5-2B | fp4 | 4 / 4 | 4 / 4 | 8 / 8 | 8 / 8 | 60 / 60 |
| Qwen3.5-4B | fp4, mix | 3 / 3 | 3 / 3 | 10 / 10 | 10 / 10 | 79 / 79 |

- **The passes are equal everywhere.** In bucket 16 Phi-4-mini's mix and LFM2-2.6B take one
  row a run with both routes: today's two-row run at 3840 is over IMEM too (4538 and 5116
  instructions), since attention is unrolled per row, head and block and so grows with the
  context. A long context there costs 3-4x the passes of bucket 1 either way. A looped
  attention would lift both routes; that is a separate item.
- **Runs.** Qwen3-0.6B and LFM2.5-230M fit 8 rows (two passes) in one run of today's in
  bucket 1, where a prompt run takes one pass's 4: the same passes, more runs (30 tokens 8 / 5,
  237 tokens 60 / 31). The host adds 0.5-0.8 ms a run (section 6), about 2 and 20 ms; today's
  route spent 0.6-0.9 s compiling on short prompts (section 1).
- **A bucket's end.** A run-time run's rows stay in its bucket, so a prompt that crosses one
  can take one pass more than today's (3 tokens from 255: runs of 1 and 2 rows, today's one of
  3), at most one per bucket crossed.

**The rule (prefill.covers).** A prompt takes prompt runs when in every bucket it touches
R_max(bucket) >= min(today's rows there, MCOLS), so no bucket streams the weights more often;
otherwise Engine.prefill_chunks and MTPDecoder.prefill take today's route for the whole prompt.
Today's rows are checked only where R_max is below one pass's rows (MCOLS, the image's rows, the
cache's end): compile_rows of R_max + 1 rows at the bucket's first run-time position, which
must not fit. The answer is kept with R_max (progcache.fact: once per layout and bucket).
Every layout above is covered in buckets 1 and 16.

**Gemma 4 E2B** (`gemma4_prompt_run`): gemma4_step's rows over a RunRows, each row a `_RowPos`
(its slot, sliding window and mask pair as a layer run's), the tokens from out[]; each token's
embedding row and PLE record are gathered at addresses from scratch registers (RLD MUL of its
word, `_gathered_out`), the records into the I/O area's rows as compile-time runs store them.
With the PLE table on the host (OTPU_PLE_HOST=1) the run reads the slot's rows, which
prefill.run writes first (Image.prompt_host_rows, Engine._write_host_rows). An image with
lookup tables holds a mask pair per I/O row. MoE images keep the layer-major prefill.
tests/test_prefill_gemma4.py: int8, fp4, the host's PLE table and the sliding ring's wrap at
768, bit-exact against compile-time runs of the same split.
