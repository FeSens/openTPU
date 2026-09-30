# Gemma 4 E4B: the per-layer embeddings from the host

Status: design. Nothing here has run yet. Numbers are computed from the checkpoint's config and
the image allocator (`gemma4.Image`, main dc4181f), or marked *estimate*.

[gemma-4-E4B](https://huggingface.co/google/gemma-4-E4B) is the first model of the roadmap that
does not fit the card's 4 GiB. It is dense, and its body at fp4 fits with room to spare. What
does not fit is its per-layer embedding (PLE) table. A token uses one row of that table: 42 x 256
values, 11 KB as the card's int8 record. So the table stays on the host. The host only reads a
row from storage and DMAs it into a slot on the card. The card dequantizes it and does all the
math, as for E2B (docs/gemma4.md). This is offload's mechanism for a row that is never on the
card (docs/offload.md section 7, branch offload-p2).

## The model

| | E2B | E4B |
|---|---:|---:|
| layers (sliding : global) | 35 (4 : 1) | 42 (5 : 1) |
| hidden | 1536 | 2560 |
| query / KV heads | 8 / 1 | 8 / 2 |
| head size, sliding / global | 256 / 512 | 256 / 512 |
| KV-shared layers | 20 (of 35) | 18 (of 42) |
| MLP | 6144 (12288 shared) | 10240 |
| PLE per token | 35 x 256 | 42 x 256 |
| PLE table | 2.35 B params | 2.82 B params |
| parameters without PLE | 2.28 B | 4.64 B |

Everything else is E2B's: the tied 262,144-token head, GeGLU, the norms, the soft cap 30, the
RoPE kinds. `gemma4.Spec.from_hf` reads E4B's config as it is. The layers run as two hardware
loops: 4 x (5 sliding + 1 global) with their own K / V, then 3 x the same unit KV-shared.

## What fits

Image sizes at a 2048-token KV capacity, fp4 layers (`Image.nbytes`):

| LM head | PLE records | Image with the PLE table | Without it |
|---|---|---:|---:|
| int8 | int8 (11,264 B / token) | 5.457 GiB | **2.707 GiB** |
| int8 | fp4 (5,888 B) | 4.144 GiB | 2.707 GiB |
| fp4 | int8 | 5.144 GiB | 2.394 GiB |
| fp4 | fp4 | 3.832 GiB | 2.394 GiB |

- The default formats (fp4 layers, int8 head, int8 PLE) need the table off the card: the body
  is 2.707 GiB.
- With it on the card, E4B fits only with both the head and the PLE records in fp4
  (3.832 GiB). That is the fallback. `tools/gemma4_quant_eval.py nll` can measure its
  accuracy cost against the host-table default before anyone picks it.
- Off the card, the spare 1.3 GiB is room for a longer KV cache. A global layer's K / V is
  about 2 KB per token per layer, so the four own global layers at 32K tokens take about
  270 MB (*estimate*).

## Per token

- **Bytes on the card:** 2.803 GB of weights per decode token with the int8 head. That is
  attention 312 MB, the MLPs 1,755 MB, the per-layer inputs 29 + 15 MB and the head 692 MB.
  With the fp4 head it is 2.467 GB. KV and I/O add a few MB.
- **Speed:** E2B streams 1,475 MB a token at 9.63 tok/s on the card (docs/gemma4.md), 14.2
  GB/s. At that rate E4B gives **about 5.1 tok/s** (int8 head) and 5.8 (fp4 head)
  (*estimate*).
- **PCIe:** one 11,264-byte record per token (5,888 fp4), host to card. At the measured Gen1
  rate (1.3 GB/s, docs/offload.md section 1) that is 9 us, plus the DMA's fixed cost, about
  30-40 us in all (*estimate*, offload's figure for a 22-43 KB row). That is 0.02% of a
  200 ms token.
- **Host:** the table as the card's records, 262,144 x 11,264 B = 2.95 GB (int8) or 1.54 GB
  (fp4), in RAM or memory-mapped from a file. It is written once when the image is built, like
  every quantized weight (the build already quantizes on the host). Per token the host reads
  one record at token id x 11,264 and DMAs it. It does no arithmetic.

## The slot

The card's DRAM holds a **PLE slot** in place of the table: R records (R = the image's prefill
rows), the table's layout with R rows. The gather that reads E2B's table at the token id
(`gather_record`) reads the slot at row r (row r of a prefill run; row 0 in decode). Nothing
else in the programs changes: the per-layer inputs, the layer loops and the head are E2B's.

The row reaches the slot in one of two ways.

1. **Runs the host starts** (prefill runs, per-position and resident decode steps: the Engine
   today). The host knows the tokens before the run. It writes their records into the slot, then
   starts the run: `Image.host_rows(tokens) -> [(address, bytes)]`, which the Engine writes before
   each run. This is the same kind of write as the stop word or a program. There is no protocol
   and no wait. It works on main now, on the ISA simulator, the RTL and the card.
2. **The card's generate loop** (autodecode). Here the card samples the token and goes straight
   on to the next one, so the host learns the token from the card:
   - After sampling, the card writes the token id to the mailbox's row, then `seq + 1` to
     `seq`, the same way offload's MoE layers post expert ids.
   - The host's server (polled by `BoardBackend.host` during the run, like offload's
     `ExpertServer`) reads the id, writes the record into slot row 0, then sets
     `served = seq`.
   - The next token's program first gathers its embedding from the head and runs the first
     PLE projection MM (32 of the 42 layers' rows, 11 MB of weights, about 0.8 ms). Only then
     does it wait, with `WAITW served GE seq`, before its gather. So the round trip (about 0.1 ms) hides behind
     work the card has to do anyway.
   - **Race-free:** the card reads the slot only between its wait and the end of the per-layer
     input pass, early in the token. It posts the next token only after sampling, at the
     token's end. So the host never writes a slot the card is reading, and one slot is enough.

The mailbox words (`seq`, the id row, `served`) and the server loop are offload's: a
`RowServer` next to `ExpertServer` in `opentpu/host/offload.py`. It is a fixed slot with no
directory; a request is one id and is served by one DMA. `BoardBackend.host` polls both. To be
agreed with offload.

## Programs

- **Resident decode:** 1,202 instructions in bucket 1 and 1,647 in bucket 8, with 6 argument
  words, under the 4K IMEM (production configuration).
- **Prefill:** runs of 1 and 2 rows compile today (1,449 and 2,172 instructions). 3 and 4
  rows run out of TMEM in the attention's output projection at hidden 2560 (E2B fits 4 rows,
  the ACT RAM's rows, so a run streams its weights once). Until the attention frees its
  buffers earlier or splits the output projection, prefill streams the weights once per
  2 tokens.

## Plan

1. **The slot option.** `Spec.image(ple="host")` or an `Image` flag, `Image.host_rows`, the
   Engine's write before each run, and the host record store (built once, then
   memory-mapped).
   - Tests on the tiny model: logits with the table on the host bit-identical to the table
     on the card, for resident decode, per-position steps and prefill runs.
2. **E4B on the ISA simulator against Hugging Face.** Greedy, 3 prompts x 24 tokens as for
   E2B, then the 900-token text. The HF reference needs about 11 GB if its PLE rows are read on
   demand (hf_lean.py with a lazy per-layer embedding), so it runs on omarchy.
3. **4-row prefill** (TMEM), then RTL cycles for a subset of layers on the DDR3-1066 bank model
   and the LiteDRAM co-simulation.
4. **The generate loop's wait**, once autodecode's loop and `WAITW` are on main: the post, the
   `RowServer`, and the wait, first on the ISA simulator's host hook, then the RTL.
   `compile_generate` for Gemma comes with it.
5. **The card**, through team-lead: token-exactness and tok/s against the 5.1 above.
