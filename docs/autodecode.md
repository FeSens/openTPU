# The decode loop on the card

One device run generates a whole reply. The host starts it with the first token, the position,
the sampling settings and the stop ids. The card then does the rest: it picks each token,
feeds it back through the embedding gather, advances the position and writes the token to
`out[]`. It stops on a stop id or after N tokens. The host only streams the tokens and
detokenizes them. It never reads the logits.

The same bitstream runs it for every model: Qwen3, LFM2 and Qwen3.5 today. The loop wraps
the model's resident decode step, which is unchanged. The ISA gains three pieces: `RLD`,
`VOP ARGMAX` and `HALT CHAIN` (docs/isa.md).

```python
eng.generate_card(tok, n, stop_ids=None, on_token=None, stop=None,     # opentpu/llm/qwen3.py
                  sampling=None, context=(), rng=None)
```

`otpu-chat` uses it for every turn the device can take (`Chat.on_card`):

- greedy turns;
- sampled turns with `top_k` 1..64 and `repetition_penalty` >= 1, which covers every model's
  default (`chat.SAMPLING`).

## The gap it removes

Production image `0f3a0000` at 133.33 MHz, measured on opentpu with `tools/decode_profile.py`
(96 greedy tokens, 4-bit weights, streamed logits). "Host path" is the host's critical path
per token: from HALTED seen to RUN written, while the card idles.

| model | device ms/token | wall tok/s | device tok/s | host path ms/token |
|---|---|---|---|---|
| LFM2.5-230M | 11.18 | 84.50 | 89.46 (-5.5%) | 0.656 (0.334 steady + one 29 ms bucket compile) |
| Qwen3-0.6B | 29.69 | 33.32 | 33.68 (-1.1%) | 0.333 (0.303 steady) |
| Qwen3.5-0.8B | 40.61 | 24.19 | 24.63 (-1.8%) | 0.685 (0.338 steady + 29 ms compile) |

The steady part splits into:

| item | ms/token |
|---|---|
| Python and the rest | 0.18-0.21 |
| logits tail | 0.05-0.08 |
| sampling | 0.05-0.07 |
| counters | 0.02 |
| status | 0.01 |
| run arguments | 0.004 |

Without streaming (tools/qual/perf.py, int8) the gap is 0.5-4.3 ms per token. The host's DMA
moves 0.5, 1.2 and 1.9 MB per token for the three models. With the loop on the card, the only
per-token host work is reading one 64-byte beat of `out[]`, while the card runs.

## The loop

`generate.compile_generate` builds one program per attention bucket, like the resident
decode: positions [(b-1)·256, b·256).

```
LD the state block (tok, tpos, ring, tokens left, sampling words, stop ids)
LOOP min(left, 256 - tpos) times            (RLD of the device-computed count)
    each run-time argument c * var: RLD MUL of the state word into its register
    the step (the model's resident decode step, unchanged); the LM head hands each logits
        chunk to the sampler instead of storing it (m.lm_sink)
    the sampler's token -> out[p + 1]
    stop: the token is a stop id, or the host set the stop word -> LOOP stop {HALT}
    the state: tok, tpos + 1 mod 256, ring + 1 mod K, left - 1
ST the state block
tokens left: LD the next bucket's [address, instructions] from the chain table, HALT CHAIN
```

**Run-time arguments.** These are the values the host wrote as ARG registers before each run
in resident decode (docs/host.md): for example `4096 * tok` for LFM2's embedding row, and
`tpos` and `ring` multiples. Here `RLD MUL` forms each from the state word: the integer
times c, mod 2^32, as the host's `arg_words`.

- There is no bound on c. Gemma 4's per-layer embedding row, `9344 * tok`, passes 2^31 at
  vocab 262144, which no fp32 product holds exactly.
- The variables themselves stay below 2^24 (`check_args`).
- A register that starts at `c * var` and steps with the step's loops (a K/V append at the
  layer loop's layer) starts with its own `RLD MUL`, before the outermost of those loops
  (compiler `Builder.run_words`). Only an address of `c * var` alone keeps an argument
  register (R15 down), loaded at the top of the token loop. At Qwen3.5-35B-A3B's dimensions
  that is 4 argument registers instead of 7 (`256`, `8`, `1` and `4 * tpos` only seeded the
  K/V append registers), which leaves R9-R11 to the step; the resident decode's programs do
  not change.

**Chaining.** IMEM holds one program (4096 instructions on the board). A bucket's program is
563-2524 instructions (fp4, cap 4096). At a bucket's end, `HALT CHAIN` loads the next
bucket's program from the chain area in DRAM and restarts it. TMEM, DRAM and the arguments
are kept. So a reply is one run whatever its length, and the host writes each bucket's
program once.

**Split programs.** A bucket whose loop does not fit IMEM or TMEM runs in the split form
(`generate.compile_bucket`, `Engine.gen_split`: None when needed, True always):

- two programs per token, which chain to each other through their own table (`ptab2`,
  entries [address, instructions] per bucket and part);
- the first part runs the step's layers and stores x to DRAM (`xs`; the model's LM head
  stores it there when `m.lm_split` is set), then chains to the second;
- the second part loads x and runs the final norm, the LM head into the sampler, the token,
  the stop and the state update (the state block back to DRAM), then chains to this bucket's
  first part, or at its end (tpos back to 0) the next bucket's;
- a mixed chain works both ways: each table names a bucket's first program, whatever its form.

Each token then pays two program loads (at most 4096 instructions each, one DRAM read of
128 KB). The second part is 367 instructions at Qwen3.5's dimensions (greedy or sampled), so
the split frees the head, the sampler and its TMEM, and no more. The layers of one plan unit
(Qwen3.5's lin, lin, lin, attn; the plan loops it) must fit one program. At the 4B and 9B
dimensions they do not from bucket 8 on (4197 and 4157 instructions), and neither does the
resident decode there (4254, 4214). The fix for that is in the kernel: the DeltaNet head
pairs in a loop at a run-time position.

## Greedy

`generate.Greedy` handles the logits chunk by chunk:

1. As each chunk arrives, one `ARGMAX` of the chunk runs while the next chunk streams.
2. At the end, an `ARGMAX` over the chunk maxima picks the chunk.
3. A register-relative copy then fetches that chunk's id.

Ties go to the first id, as `np.argmax`. With S > 1 slices, the slices `all_gather` their best
(max, id), then take the ARGMAX over the gathered maxima.

## Sampling

`generate.Sampler` is chat.sampler's pick on the card, built from existing VPU operations plus
ARGMAX and RLD. No new unit is needed.

**Each chunk, while the next one streams:**

- **Soft cap.** With a spec that has one (Gemma's final_logit_softcapping), the LM head caps
  the chunk first: `c tanh(l / c)` by `kernels.lib.softcap`, 5 VOPs; `generate.softcap_ref`
  is the same in the ISA's fp32 for `reference_pick`. Greedy takes the raw logits' argmax
  (`Greedy.raw`: the cap keeps their order), and the stored logits stay raw.
- **Repetition penalty.** `min(l * pa, l * pb)` with the DRAM vectors `pa`, `pb`. For the
  context's ids these hold 1/R and R, and 1 for every other id. The result is l/R for l > 0
  and l·R for l < 0, as Hugging Face computes it. The card marks each token it generates
  itself.
- **Store and block maxima.** The chunk's logits go to DRAM (`lg`), and the maxima of its
  64-wide blocks go to TMEM.

**Then the token:**

1. **The top blocks.** The k blocks with the largest maxima. An `ARGMAX` finds each one,
   `RLD` reads its index, the knock-out sets it to −inf, and an `LD` gathers the block's 64
   logits with their ids. They hold the top k: the k-th largest block maximum is a lower bound
   of the k-th largest logit.
2. **The top k logits.** The k largest among the candidates, in descending order, by the same
   ARGMAX / knock-out loop. With S > 1, the k largest of all slices' candidates, after an
   `all_gather`.
3. **Softmax.** `p = exp2((l − l0) · log2(e)/T)`.
4. **Cumulative sums.** One RDOT with a triangular ones matrix.
5. **Top-p.** Keep element i while cum[i−1] < P · cum[last].
6. **The pick.** For the position's uniform u, pick = #{cum ≤ u · cum[last kept]}, capped at
   the last kept.

**The uniforms.** The host draws them from the sampler's own generator and writes them, one
per position, with the run's state: n words per run. A run that stops early drops the draws it
did not use.

**Exactness.** `generate.reference_pick` is the sampler in numpy with the ISA's fp32
arithmetic. The device's picks match it bit for bit.

- Taken over u in [0, 1), the pick's distribution matches chat.sampler's within 1e-4.
- Each id's probability is its share of u.
- The device only picks ids that chat.sampler keeps.

This is tested for Qwen3's and LFM2's default settings, for k = 64, and for k = 3 with S = 1
and S = 2.

**Cost.** Measured on the RTL (`tools/perf_qwen.py --resident --generate N`, Qwen3.5-0.8B
with 2 layers, fp4, board memory model; its LM head, 248K ids, is the largest of the three
models'), per token against the resident step (2,477,675 cycles):

| loop | cycles per token | against the step |
|---|---|---|
| greedy | 2,472,342 | −5.3K (−0.2%): the logits are not stored |
| sampled, k 20, top-p 0.95 | 2,494,388 | +16.7K (+0.7%) |
| sampled, k 20, top-p 0.95, penalty 1.1 | 2,528,122 | +50.4K (+2.0%) |

- On the whole 24-layer model (40.6 ms a token) these are −0.04 ms, +0.13 ms and +0.38 ms at
  133.33 MHz. The host path they replace is 0.34-0.69 ms.
- The per-chunk work hides under the LM head's weight stream. It works in place in two tiles
  held for the whole head: fresh tiles per chunk were freed under the next chunk's MM output,
  which then waited for the chunk's work (7.2K cycles a chunk, +224K a token).
- What remains is the token's serial part (the ARGMAX loops, about 17K cycles at k = 20) and,
  with the penalty, 3 MB more DRAM traffic a token (`pa`, `pb`, `lg`) beside the weights.
- Greedy turns use the Greedy sink and pay neither.

## The host

**Before each run**, `Engine.generate_card` writes:

- the state block;
- `OUT_MARK` over the run's `out[]` words;
- with sampling: the uniforms and the penalty vectors;
- once per bucket and mode (greedy or sampled): the chain area and its table.

**On the card**, `BoardBackend.run_generate` starts the program and reads new beats of `out[]`
as the tokens land. It sleeps between reads on the expected token gap. If `stop()` returns
true, it writes the state's stop word, and the card halts after the token in flight.

**On the ISA simulator**, `IsaBackend` runs the whole loop, then reads `out[]` at the end.

**Where a reply ends.** The last token of a run is picked but not fed. The next run feeds it:
`Chat.resume` continues a reply cut at max_new from there. A stop id is returned, but not fed.

## Status

- **isasim.** `tests/test_autodecode.py` covers:
  - the ISA pieces;
  - greedy generation matching the host's resident loop token for token on tiny Qwen3, LFM2
    and Qwen3.5, at S = 1 and 2, across the bucket boundary (HALT CHAIN) and at a stop id;
  - the sampled loop matching `reference_pick` in every mode, including greedy with the
    penalty;
  - chat turns, greedy and sampled, with resume and EOS;
  - `run_generate` on a fake card that computes: tokens streamed during the run, and stop.
- **rtlsim.** The same file on the Verilator RTL:
  - RLD (with MUL), ARGMAX and HALT CHAIN against the ISA simulator, and otpu-diag's gen
    checks in the board's configuration;
  - 12 tokens from a 248-token prefill across the bucket boundary in one run, greedy and
    sampled (top-k, top-p, the penalty), on tiny Qwen3, LFM2 and Qwen3.5 at S = 1 and 2: the
    tokens and the whole DRAM equal the ISA simulator's.
- **Board model.** `tests/test_board.py`, tb_board through `SimTransport`: the same run through
  `BoardBackend.run_generate` (CAPS bit30), and the gen checks.
- **Existing programs** are unchanged: perf_qwen (Qwen3.5 fp4, 2 layers) 2,477,721 cycles and
  2,477,675 resident, main 812bb01's.
- **The RTL:**

  | file | change |
  |---|---|
  | otpu_pkg | `OP_RLD` on U_COLL (footprint: one TMEM word read); `V_ARGMAX` (writes 2 words per row) |
  | otpu_seq | the R stage holds until RLD's value is in R[rd] (a flip-flop, off the fetch path); HALT CHAIN's address and count out |
  | otpu_slice | RLD read locally on the collective unit's TMEM port (not sent to otpu_coll), f2i and MUL (two partial products, 3 DSPs) in 3 stages; the CHAIN reload: once the writes are idle, hold the slice's units in reset and run the program loader from the chain address, then release (TMEM, DRAM and the arguments stay; HALTED stays low, ICOUNT adds up) |
  | otpu_vpu | ARGMAX through the RMAX tree with the lane index and the chunk's column; ties keep the older value; the pair written through lanes 0 and 1 four cycles after the row (index add, leading zeros, i2f) |
  | otpu_ctrl | CAPS bit30 |

- **Area and timing**, Vivado out of context at 7.5 ns (xc7k480t-2, place and route; the
  tournament's parts), against main (812bb01):

  | unit | LUT | FF | LUTRAM | DSP | WNS |
  |---|---|---|---|---|---|
  | otpu_seq | 15,835 (+497) | 10,655 (+65) | 912 (+0) | 7 (+0) | +0.642 ns (main +0.541) |
  | otpu_vpu | 24,026 (+431) | 19,414 (+276) | 1,200 (+39) | 68 (+0) | +0.961 ns (main +1.044) |
  | otpu_dma (WAITW) | 8,822 (+778) | 12,290 (+259) | 2,824 (+0) | 0 | +0.865 ns (main +0.593) |

  About 0.6% of the device's LUTs. otpu_slice's RLD reader and CHAIN FSM (about 100 FF,
  3 DSPs) are not in a part.
- **Dev build** (waitw be824d5: this and WAITW, FAST=1 at 100 MHz, omarchy 2026-09-30,
  `~/otpu-build/deploy_adw100_be824d5`): timing met, WNS +0.091 ns, WHS +0.016 ns (core
  +0.096, LiteDRAM +0.091); LUT 162,441 (54.4%), FF 132,924, slices 71.9%, BRAM 607.5, DSP
  716.
- **Card** (the dev build on opentpu, 2026-09-30, `tools/qual/qual.sh fast`, identity
  BUILD_ID be824d57, 100 MHz, CAPS bits 30 and 31): 42 PASS, 0 FAIL in 61 min. The decode
  loop on the card gave the ISA simulator's tokens in 6/6 runs (Qwen3, LFM2 and Qwen3.5, int8
  and fp4). WAITW on the host's writes: 200 rounds, 16..32768 words, 30,927..2,124,512 cycles;
  its timeout: ERROR, and the next run halted normally (be824d5 predates STATUS bit8
  WAIT_TO). `decode_profile --card-loop`, 96 tokens (95 on the card in one run), wall against
  device tok/s, beside the same bitstream's host loop (per-position picks, streamed logits):

  | model (fp4, int8 head) | host loop | card loop, greedy | card loop, sampled |
  |---|---|---|---|
  | Qwen3-0.6B | 28.96 / 29.21 (-0.9%) | 28.95 / 29.45 (-1.7%) | 28.85 / 29.30 (-1.5%) |
  | LFM2.5-230M | 73.94 / 77.39 (-4.5%) | 74.32 / 77.18 (-3.7%) | 71.62 / 75.15 (-4.7%) |
  | Qwen3.5-0.8B | 21.30 / 21.49 (-0.9%) | 21.03 / 21.70 (-3.1%) | 21.03 / 21.58 (-2.6%) |

  The card loop's wall counts from the host's pick of the first token, so it includes the
  run's start once per reply: the bucket's compile (on the Mac, bucket 1: 17-27 ms for Qwen3,
  35-53 ms for LFM2, 54-101 ms for Qwen3.5), the program's upload and the sampler's inputs.
  The differences above are 47-140 ms per reply, the same order. decode_profile now also
  prints the rate from the card's first token to its last, and the start with its compile, to
  tell the two apart.
- **133.33 MHz:** production has been the fused build c2830d6 since 2026-09-30 (it carries
  GEN and WAITW). Its qual passed: the card loop gave the ISA simulator's tokens in 6/6 runs,
  WAITW passed 200 rounds, the timeout set ERROR and WAIT_TO, and decode ran 7-8% faster.
- **Next:** more than one token per weight pass (multi-token prediction): the design is
  [mtp.md](mtp.md).
