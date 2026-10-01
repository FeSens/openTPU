# Gemma 4 E4B: the per-layer embeddings from the host

Status: runs on the card (build B, 2026-10-01): token for token the ISA simulator's on every
decode path, the card's generate loop included, at 3.78 tok/s (tools/qual/perf.py) and 3.83
tok/s in the generate loop (section "On the card"). The weight formats are in section
"Accuracy". Numbers are measured (with how), computed from the checkpoint's config and the
image allocator (`gemma4.Image`), or marked *estimate*.

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

Image sizes at a 2048-token KV capacity (`Image.nbytes`):

| Layers | LM head | PLE records | Image with the PLE table | Without it |
|---|---|---|---:|---:|
| fp4 | int8 | int8 (11,264 B / token) | 5.457 GiB | 2.707 GiB |
| fp4 | int8 | fp4 (5,888 B) | 4.144 GiB | 2.707 GiB |
| fp4 | fp4 | int8 | 5.144 GiB | 2.394 GiB |
| fp4 | fp4 | fp4 | 3.832 GiB | 2.394 GiB |
| int8 | int8 | int8 | | 4.557 GiB |
| int8 | fp4 | int8 | | 4.244 GiB |
| **int8, down 0-23 fp4** | **fp4** | **int8** | | **3.951 GiB** |

- Off the card, the PLE table leaves the body. With fp4 layers it is 2.7 GiB, but fp4 layers
  cost E4B too much accuracy (next section). So the chosen formats are int8 layers with
  the down projections of the first 24 layers in fp4, and the fp4 head: 3.951 GiB.
- With the table on the card, E4B fits only with fp4 layers, head and PLE records (3.832 GiB).
- A global layer's K / V is about 2.3 KB per token per layer. The four own global layers at
  4096 tokens add 19 MB to the 2048-token image (*estimate*).

## Accuracy: which weights in fp4

E4B in fp4 loses much more than E2B. `tools/gemma4_quant_eval.py nll` runs the float64 emulation
with every activation point (int8 matmul inputs, K / V, P) over the first 900 tokens of
*Pride and Prejudice* (docs/gemma4.md's long-context text). It gives the next-token NLL of the
soft-capped logits:

| Weights | ppl | Top-1 = float64's |
|---|---:|---:|
| float64 | 1.577 | 1 |
| int8 layers, int8 head (4.557 GiB: does not fit) | 1.714 | 0.944 |
| fp4 layers, int8 head | 10.142 | 0.484 |
| fp4 layers, attention int8 | 6.253 | 0.586 |
| fp4 layers, attention and down int8 | 5.286 | 0.631 |

E2B in fp4 on the same text: ppl 5.629, against 4.351 in int8. E4B's weights are not the
problem. They look like E2B's: Gaussian blocks with max / rms 2.9, and a relative error of
0.093 in the MXU's fp4 for every tensor sampled, with NVFP4 no better. E4B is more sensitive,
and the errors compound over its 42 layers.

**One unit in fp4.** One 6-layer unit of one kind in fp4, the rest int8, ΔNLL against int8
(0.539):

| Kind (MB saved per unit) | 0-5 | 6-11 | 12-17 | 18-23 | 24-29 | 30-35 | 36-41 |
|---|---:|---:|---:|---:|---:|---:|---:|
| gate / up (157) | +0.002 | +0.021 | +0.049 | +0.072 | +0.134 | +0.135 | +0.110 |
| down (79) | -0.012 | -0.000 | +0.014 | +0.045 | +0.056 | +0.054 | +0.025 |
| attention (46) | +0.000 | +0.042 | +0.093 | +0.254 | +0.074 | +0.060 | +0.062 |

The fp4 head costs +0.025 for 335 MB (measured on top of gate / up 0-23).

- The attention has to stay in int8: it costs the most per byte.
- The cheap savings are the head and the first layers' MLPs.
- In a mix the costs add up to 1.5-2x their sum.

On the ISA simulator with fp4 layers and the int8 head, none of the 3 prompts x 24 greedy
tokens matched HF; they diverged at tokens 4, 2 and 0. The device follows the fp4 emulation
(cosine 0.9987 over the vocabulary at prompt 2's first token). At that token HF's ` ran` is at
6.46 before the cap, the emulation's at 9.92 and the device's at 10.68. So the loss is the
format's, not a kernel's.

In the default formats (below), 2 of the 3 prompts match HF for all 24 tokens. Prompt 0
matches for 12 tokens. At token 13 HF takes ` a` and the device ` also`. That is a near-tie:
in the emulation of these formats ` a` leads by 0.028 before the cap, against 0.98 in float
and 1.17 in int8.

After the 900 tokens of *Pride and Prejudice* (ISA simulator, cap 1024, these formats), the
device follows HF for 4 tokens. At the 5th, HF takes ` such`, the device ` that`. The model is
unsure there: in float ` such` leads ` that` by only 0.83, at logits near 7. Even int8 layers
and head pick ` it` (` such` 0.18 below). The emulation of these formats has ` it` 4.90,
` that` 4.42, ` such` 4.29.

**Mixes that fit 4 GiB.** int8 everywhere is 4.557 GiB, so a mix must save 598 MB. Each of
these does:

| fp4 (the rest int8) | Loops | ppl | Top-1 | Image |
|---|---:|---:|---:|---:|
| **head, down 0-23** | 2 | **1.928** | 0.893 | 3.951 GiB |
| gate / up 0-23 | 2 | 2.007 | 0.882 | 3.971 GiB |
| head, gate / up 0-23 | 2 | 2.057 | 0.870 | 3.658 GiB |
| head, gate / up 0-11 | 3 | 1.851 | 0.912 | 3.951 GiB |
| head, gate / up and down 0-11 | 3 | 1.878 | 0.902 | 3.805 GiB |
| gate / up 0-11, down 0-23 | 3 | 1.985 | 0.880 | 3.971 GiB |
| head, gate / up 0-5, down 0-11 | 6 | 1.836 | 0.909 | 3.951 GiB |
| head, gate / up 0-5, down 0-23 | 4 | 2.022 | 0.877 | 3.805 GiB |

**A text no model has seen.** The first 900 tokens of docs/offload.md, written for this
repository in September 2026. Same emulation, ppl:

| Weights | E4B | Top-1 | E2B | Top-1 |
|---|---:|---:|---:|---:|
| float64 | 11.697 | 1 | 13.669 | 1 |
| int8 layers, int8 head | 11.986 | 0.938 | | |
| **int8; head, down 0-23 fp4** | **12.122** | 0.892 | | |
| int8; head, gate / up 0-11 fp4 (3 loops) | 12.414 | 0.908 | | |
| int8; gate / up 0-23 fp4 | 12.621 | 0.879 | | |
| fp4 layers, int8 head | 27.704 | 0.611 | 15.717 (E2B's default) | 0.811 |

- The default mix costs 1.1% in ppl over int8 here.
- On this text, the split three-loop mix is worse than the default, not better as on
  *Pride and Prejudice*.
- E4B in fp4 is far behind E2B in fp4. In the default mix it beats E2B's default, and E2B in
  float as well.

**The choice.**

- A layer loop runs one format: the format is in the MM instructions. So a format change
  inside the 24 own-KV layers splits their loop, and every extra loop costs IMEM.
- With three loops, decode is 2,298 / 2,967 instructions (bucket 1 / 8). That fits the 4K
  IMEM. A 4-row prefill run is 4,439 instructions even at position 0, though, so prefill falls
  back to 2-row runs, and those pass 4,096 near position 2044.
- The default, `down@0-23=fp4` with the fp4 head, keeps today's two loops. Its prefill runs
  4 rows up to about position 1000, like the fp4 image.
- Formats are set per layer and kind by `Spec.formats`, `Image(formats=)` or `OTPU_FORMATS`
  (`gemma4.layer_formats`: "kind[@a-b]=fmt", over `wformat`; `head=fmt` for the LM head).
- **The default follows the fit, not the model.** `Spec.from_hf` stores a mix,
  `Spec.fit_formats`: the head and the own-KV layers' down projections in fp4 (E4B:
  `head=fp4,down@0-23=fp4`). An int8 image (`wformat="int8"`, nothing else asked) takes it
  only when it fits 4 GiB beside no PLE choice. E2B's int8 image fits (3.442 GiB, fp4 PLE
  records on the card), so it stays as it is, and E2B's default fp4 layers never take it.
  E4B's int8 image takes it: 3.958 GiB at 2048 tokens with the lookup tables and the
  generate area, the PLE table on the host. `Image.choices` holds what was chosen, so the
  Engine's compile worker builds the same image (`Engine._image_kw`).

## Per token

- **Bytes on the card:** 4.140 GB of weights per decode token in the default formats (int8
  layers, down 0-23 and the head in fp4). With fp4 layers and the int8 head it is 2.803 GB:
  attention 312 MB, the MLPs 1,755 MB, the per-layer inputs 29 + 15 MB and the head 692 MB.
  KV and I/O add a few MB.
- **Speed:** E2B streamed 1,475 MB a token at 9.63 tok/s on the card's previous image
  (e698dcd7; docs/gemma4.md), 14.2 GB/s. At that rate E4B gives **about 3.4 tok/s** in the
  default formats, and 5.1 with fp4 layers (*estimate*). The RTL's 42-layer scaling below
  gives 3.5 and 5.0. Measured on build B: 3.78 tok/s, 4,215 MB a token at 15.95 GB/s
  (section "On the card").
- **PCIe:** one 11,264-byte record per token (5,888 fp4), host to card. At the measured Gen1
  rate (1.3 GB/s, docs/offload.md section 1) that is 9 us, plus the DMA's fixed cost, about
  30-40 us in all (*estimate*, offload's figure for a 22-43 KB row). That is 0.02% of a
  200 ms token.
- **Host:** the table as the card's records, 262,144 x 11,264 B = 2.95 GB (int8) or 1.54 GB
  (fp4), in RAM or memory-mapped from a file. It is written once when the image is built, like
  every quantized weight (the build already quantizes on the host). Per token the host reads
  one record at token id x 11,264 and DMAs it. It does no arithmetic.

## On the RTL

One resident decode token on the Verilator RTL at position 600:
- the production configuration;
- the board's memory path at 133.33 MHz;
- `OTPU_PLE_HOST=1`, with the slot written as the Engine writes it;
- `tools/perf_qwen.py --model models/gemma-4-E4B --layers 0,1,2,3,4,5,24,25,26,27,28,29 --pos
  600 --resident --ddr 1066 [--ldc] --mhz 133.33 --check`, with the formats below.

The full image does not fit the simulated memory, so the run takes 12 of the 42 layers: an own
unit, a KV-shared unit, and the full LM head. In every run, DRAM after the token is
bit-identical to the ISA simulator's.

| Formats | Memory model | Cycles | MLP | Head | Attention | PLE | Gathers |
|---|---|---:|---:|---:|---:|---:|---:|
| default (`--wformat int8 --head-format fp4`, OTPU_FORMATS=down@0-23=fp4) | DDR3-1066 bank model | 12,057,622 | 7.230 M | 2.897 M | 1.681 M | 0.229 M | 20,580 |
| | LiteDRAM's controller (`--ldc`) | 13,349,697 | 8.010 M | 3.253 M | 1.831 M | 0.236 M | 19,941 |
| fp4 layers, int8 head | DDR3-1066 bank model | 10,830,271 | 4.081 M | 5.583 M | 1.019 M | 0.136 M | 10,526 |
| | LiteDRAM's controller | 12,022,156 | 4.598 M | 6.163 M | 1.111 M | 0.140 M | 10,309 |

The co-simulated controller is within 1% of the card on E2B (docs/board.md). Its phases are
scaled to 42 layers by bytes: the MLP x 3.456 (24 layers with the fp4 down projection, 18
without), attention x 3.556, PLE x 3.5, the head as it is.

- Default formats: MLP 27.68 M, attention 6.51 M, PLE 0.82 M and head 3.25 M. That is
  **about 38.3 M cycles, 3.5 tok/s at 133.33 MHz** (*estimate*).
- fp4 layers: 26.6 M cycles, 5.0 tok/s.
- E2B ran 13.84 M cycles on the card's previous image (e698dcd7), which the co-simulation
  matches. Build B runs it in 12.61 M (-8.9%), and E4B in 35.24 M, 8.0% under the estimate.

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
2. **The card's generate loop** (`Engine.generate_card`). Here the card samples the token and
   goes straight on to the next one, so the host learns the token from the card. The image
   has a PLE mailbox in its I/O area, in offload's format: `seq`, the id row and `served`, each
   on its own 64-byte line.
   - Before the run, the host writes the first token's record into the slot.
   - After sampling, the card posts the token. It fences (`WAITW served GE seq`), writes the
     id to the row, then `seq + 1` to `seq` (`kernels/mailbox.post`, the pattern of
     offload's MoE layers, `moe.moe_ffn`).
   - The host's `RowServer` (`opentpu/host/offload.py`, next to `ExpertServer`) reads the id,
     writes the record into slot row 0, then sets `served = seq`. It is polled by the ISA
     simulator's WAITW hook, and on the card by `BoardBackend.host` while the run is in
     flight. The Engine polls both servers.
   - The next token's step gathers its embedding from the head, then waits with
     `WAITW served GE seq` (`kernels/mailbox.wait_served`) before its PLE gather. That costs 4
     instructions, and the wait holds at its first read when the host is faster. The
     host's write could also hide behind the first PLE projection MM (11 MB, about 0.8 ms).
     It is not done: the round trip is about 0.1 ms of a 285 ms token (*estimate*).
   - **Race-free:** the card reads the slot only after its wait, early in the token. It posts
     the next token only after sampling, at the token's end. So the host never writes a slot
     the card is reading, and one slot is enough.
   - Before the host writes rows for a run it starts itself, it first serves a request the
     last generate run left. Otherwise that request's record would land after its own.

   On the ISA simulator the tokens equal those with the table on the card, greedy (one
   program per bucket, or split in two) and sampled. Over a fake card that runs in a thread
   on the host's DRAM, with `BoardBackend` polling the server during the runs, the generate
   loop gives the ISA simulator's tokens (`tests/test_gemma4.py`).

## Programs

(Production configuration: PAIR, DSTEP, STREAM; cap 2048, the PLE table on the host.)

| Formats | Decode, bucket 1 / 8 | 4-row run at 0 / 1024 / 2044 | 2-row run |
|---|---:|---:|---:|
| default (int8, down 0-23 and head fp4) | 1,692 / 2,137 | 3,261 / 4,365 / 4,866 | 2,076 / 2,628 / 2,957 |
| fp4 layers, int8 head | 1,762 / 2,207 | 3,237 / 4,341 / 4,842 | 2,116 / 2,668 / 2,997 |
| head and gate / up 0-11 fp4 (3 loops) | 2,298 / 2,967 | 4,439 / 6,095 / 6,848 | 2,818 / 3,646 / 4,141 |

- **Resident decode** takes 5 argument words and fits the 4K IMEM. The MLP runs in chunks
  of 512 (4-bit) or 640 (int8) columns. `_mlp_chunk` takes the largest multiple of the
  format's chunk up to MLP_CHUNK that divides 10240, where it used to take 1280. Four rows'
  gate and up tiles in flight then fit TMEM.
- **Prefill:** runs of 4 rows fit TMEM. That needs three changes:
  - the chunk above;
  - the residual adds done in place, a row at a time, when a tile is over an eighth of TMEM
    (`_add_norm`);
  - the K / V temporaries freed before the output projection.

  From about position 1000 a 4-row run is over the IMEM, so the Engine falls back to 2-row
  runs.
- **E2B:** its programs keep their instruction counts (decode 1,334 / 1,779, 4-row runs 2,472
  to 4,077 with the int8 head). Only their TMEM addresses move. The tiny model's logits are
  bit-identical to main's, per-position and resident, int8 and fp4. With the fp4 head its 4-row
  runs now fit TMEM too (main: TMEM exhausted, so 2-row runs).

## On the card

Build B (`deploy_fused133c_79c5707a`, 133.33 MHz, DDR3-1066), 2026-10-01. The default formats
(int8 layers; the head and down 0-23 in fp4), the PLE table on the host as int8 records, cap
2048: a 3.958 GiB image, the Engine built in 158 s on opentpu.

- **Token-exact** against the ISA simulator. After prefill runs of the 11-token prompt `The
  lighthouse keeper climbed the stairs at dusk, and`: its first token, then 24 tokens each by
  resident decode steps, per-position steps, and the card's generate loop, greedy and sampled
  (temperature 0.8, top-k 40, top-p 0.95). In the loop, the `RowServer`'s history (the tokens
  the card asked rows for: 23 greedy, 24 sampled) equals the tokens.
- **tools/qual/perf.py** (64 greedy tokens after the 512-token prompt, the host's argmax in
  the loop): 35.242 M cycles a token, 3.78 tok/s device, 3.75 wall. 4,215 MB read a token,
  15.95 GB/s while running (94% of the peak); the MXU starved 0%. The prompt in 128 4-row
  prefill runs: 14.8 tok/s device, 14.1 wall.
- **The card's generate loop** (`tools/decode_profile.py --card-loop --greedy`, 64 tokens,
  one PLE row served per token): 34.784 M cycles a token, 3.83 tok/s device, 3.81 wall; 343 ms
  before the first token, 80.5 ms of it compiling.
- **The wait for the row** costs 0.56%. E2B, whose table fits the card, ran the generate loop
  both ways (greedy, int8 head): 12.173 M cycles a token with the table on the host
  (`OTPU_PLE_HOST=1`) against 12.105 M on the card, 68K cycles (0.51 ms). Wall is 10.83 tok/s
  both ways.

## Plan

1. **The slot option** (done): `Image(ple_host=)` / OTPU_PLE_HOST, `Image.host_rows`, the
   Engine's write before each run. Tests on the tiny model: logits and caches with the table
   on the host are bit-identical to the table on the card, for resident decode, per-position
   steps and prefill runs.
2. **Weight formats per layer** (done): `gemma4.layer_formats`, and the scan and mixes above.
3. **E4B on the ISA simulator against Hugging Face in the default formats.** Greedy, 3
   prompts x 24 tokens as for E2B, then the 900-token text (done, above). The HF reference needs 12-13 GB if
   its PLE rows are read on demand (hf_lean.py with a lazy per-layer embedding), so it runs on
   omarchy.
4. **RTL cycles** (done, above), on the DDR3-1066 bank model and the LiteDRAM co-simulation.
5. **The generate loop's wait** (done): the post, the `RowServer` and the wait, on the ISA
   simulator's host hook and on a fake card that the host serves during its runs. On the RTL,
   the fence holds at its first read in a resident step. E4B's generate programs fit IMEM
   as one program: greedy 1,753 / 2,197 instructions (bucket 1 / 8), sampled
   2,035 / 2,478.
6. **The card** (done, section "On the card"): token-exact on every decode path; 3.78 tok/s
   (perf.py) and 3.83 in the card's generate loop, against the 3.5 estimated.
