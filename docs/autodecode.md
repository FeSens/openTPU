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
    each run-time argument c * var: VOP MUL of the state word, RLD into its register
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
`tpos` and `ring` multiples. Here the card computes each one as an fp32 product of the state
word, then `RLD` truncates it into the argument register.

- The product must be exact in fp32: `check_args` requires (bound − 1) × odd(c) < 2^24.
- Every model today uses 6 or 7 arguments, and the loop nests 2 deep.

**Chaining.** IMEM holds one program (4096 instructions on the board). A bucket's program is
563-2524 instructions (fp4, cap 4096). At a bucket's end, `HALT CHAIN` loads the next
bucket's program from the chain area in DRAM and restarts it. TMEM, DRAM and the arguments
are kept. So a reply is one run whatever its length, and the host writes each bucket's
program once.

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

**Cost.** The per-chunk work hides under the LM head's weight stream. The token's serial part
is the two ARGMAX loops of k iterations each. At LFM2's k = 50 that is about 18K cycles, about
1% of its token. At Qwen3's k = 20 it is about 3K cycles. These are estimates until rtlsim
gives the cycles. Greedy turns use the Greedy sink and pay neither loop.

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
- **Next: the RTL.**

  | file | change |
  |---|---|
  | otpu_pkg | `OP_RLD` on U_COLL (footprint: one TMEM word read); `V_ARGMAX` (writes 2 words per row) |
  | otpu_seq | the R stage holds until RLD's value is in R[rd]; HALT CHAIN's address and count out |
  | otpu_slice | RLD read locally on the collective unit's TMEM port (not sent to otpu_coll); the CHAIN reload: once the writes are idle, hold the slice's units in reset and run the program loader from the chain address, then release (TMEM, DRAM and the arguments stay; HALTED stays low) |
  | otpu_vpu | ARGMAX through the RMAX tree with the lane index and the chunk's column; ties keep the older value; the pair written through lanes 0 and 1 with i2f |
  | otpu_ctrl | CAPS bit28 |

  After that: rtlsim token-exact, a FAST=1 100 MHz build, the card, then 133.33 MHz.
- **Area estimate** before RTL: about 0.7K LUT and 0.25K FF, about 0.3% of the slices.
