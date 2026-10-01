# Bigger than DRAM: MoE expert offloading

Status: phase 1, a design study, revised for path (a). Nothing here has run on the card yet.
Every number is marked as one of: measured (with where), simulated from real router traces, or
an *estimate*.

The goal (the user's) is to run models that do not fit in the card's 4 GiB, streaming from the
host and its SSD, starting with mixture-of-experts (MoE) models. An MoE decode token reads only
its top-k experts per layer, a few percent of the weights. So the card can keep the dense part
and a cache of experts on board, and fetch the rest.

**The rule: openTPU does all the computing.** The host CPU is not a compute resource. It stores
the model (SSD, page cache, RAM) and is the source of the DMA; nothing else. The card:

- routes (the router MM, and the top-k by autodecode's ARGMAX knock-out);
- asks for the experts it lacks;
- waits for them inside its own run;
- computes every expert.

A host daemon only moves bytes. An earlier version of this study also modelled a hybrid, where
the host computed the missing experts, and host-side routing. Both are gone, along with the
host fp4 kernel.

**Summary.**

- **Models (section 2).**
  - Gemma 4 E4B is dense. Its per-layer-embedding (PLE) table is what exceeds 4 GiB, and the
    host can write the token's row (section 7).
  - The Gemma 4 MoE is gemma-4-26B-A4B.
  - The best fits for the card are LFM2.5-8B-A1B (just over 4 GiB), gemma-4-26B-A4B and
    Qwen3.5-35B-A3B (3-4x the card, inside host RAM), and Qwen3-Next-80B-A3B (larger than host
    RAM: the SSD tier).
- **The hierarchy (section 1, measured on opentpu).** Card DRAM runs at 14 GB/s, PCIe Gen1 at
  1.3 GB/s and the SSD at 0.51 GB/s. A missing expert costs about 10x its compute time to bring
  over Gen1.
- **Router traces (section 3; 4 texts x 2048 tokens, CPU).** At the card's cache size, a
  per-layer LRU hits:
  - 98.6% of LFM2.5-8B-A1B's picks (85% of its experts fit);
  - 76% of Gemma 4 26B-A4B's (19% fit);
  - 66% of Qwen3.5-35B-A3B's (13% fit).

  Fixed profiles transfer badly across texts: Gemma 4 gets 22-41%, Qwen3.5 11-23%.
- **Tokens per second, path (a) (section 4, simulated).**

  | Model | Gen1 | Gen2 | All resident |
  |:--|--:|--:|--:|
  | LFM2.5-8B-A1B | 13.0 | 13.5 | 13.7 |
  | gemma-4-26B-A4B | 3.9 | 5.0 | 5.9 |
  | Qwen3.5-35B-A3B | 4.2 | 5.8 | 7.8 |

  - The card computes the experts it has while the missing ones stream. That is worth +12-14%
    over waiting for the last one on Gemma 4 and Qwen3.5.
  - Prefetch from router predictions is at best neutral (Qwen3.5 at Gen2). Otherwise it loses
    1-14% with the prediction's k best, and more with its 2k best: cache pollution, plus the
    link and DRAM time of wrong guesses.
  - 4-bit experts are worth 2.3-2.9x over int8.
  - Request and flag latency (10-100 us) moves results by under 1%.
- **Design (section 5).** One new instruction, `WAITW` (wait on a DRAM word), agreed with
  autodecode. Everything else is built from autodecode's `ARGMAX`, `RLD`, the register-count
  `LOOP` and `CHAIN`. Per MoE layer, the card:
  1. posts its k expert ids to a mailbox in DRAM;
  2. computes the experts a directory in DRAM says it has;
  3. `WAITW`s on each missing expert's directory entry, which the host writes after that
     expert's DMA.

  One run per reply; no halts.
- **PCIe Gen2 (section 6).** +30% for Gemma 4, +39% for Qwen3.5-35B and +3.5% for LFM2.5.
  Parked until phase 2 runs. It is then the biggest single lever for models 3-4x the card.
- **SSD tier (section 8).** Models larger than host RAM run at the link's and the SSD's pace:
  - Qwen3-Next-80B-A3B: ~2.4 tok/s at Gen1, ~3.5 at Gen2 (*extrapolated* from Qwen3.5-35B's
    traces);
  - gpt-oss-120b and Qwen3.5-122B-A10B: under ~1 tok/s at Gen1. Their experts are large and the
    card caches only 1-4% of them.
- **Next (section 9).** LFM2.5-8B-A1B end to end on the ISA simulator, with the card's 4 GiB,
  the router on the card, per-layer LRU slots and a host DMA daemon.

The tools:

- `tools/offload/survey.py`: the model survey, from the checkpoints' safetensors headers.
- `router_trace.py`: the expert choices of a Hugging Face model on a text, on CPU.
- `cachesim.py`: expert-cache hit rates on those traces, and path (a)'s event model.

## 1. The memory hierarchy

| Level | Size | Bandwidth | Source |
|:--|--:|--:|:--|
| Card DRAM (2 x DDR3-1066) | 4 GiB | 13.9-14.5 GB/s while decoding | card counters, README |
| PCIe Gen1 x8 (XDMA), host -> card | | 1.26-1.34 GB/s (opentpu); 1.42 (omarchy); 1.7 in best-placed 8 MiB calls (omarchy) | `otpu-selftest`, docs/host.md |
| card -> host | | 0.84-1.00 GB/s (opentpu) | `otpu-selftest` |
| one small DMA call (64 B - 4 KiB) | | 11-15 us (omarchy, interrupt mode) | docs/host.md |
| Host RAM (opentpu, DDR3, i7-4790) | 31 GB (23-28 free) | 22.6 GB/s (4-thread fp32 GEMV), 13.6 (1-thread sum) | measured 2026-09-29 |
| SSD (Crucial BX500 240 GB, SATA, LUKS + btrfs zstd) | 123-140 GB free | 512 MB/s (O_DIRECT, 4 MiB, 1 reader), 515-524 (2-4 readers), 227 MiB/s (128 KiB x 4) | measured 2026-09-29 |

Everything is measured on opentpu, the card's host, unless marked otherwise.

- The SSD figures read a 4 GiB file of random bytes written for the test
  (`dd iflag=direct`). Reads of the models' own files go through btrfs compression, are not
  O_DIRECT, and come from the page cache.
- opentpu's other disk is an HDD that is not ours.
- The card sits in the CPU's x16 slot (`max_link_speed` 8 GT/s x16 on the root port, 00:01.0),
  so a Gen2 card would train at 5 GT/s.

The ratios decide the design. Card DRAM : PCIe : SSD is 14 : 1.3 : 0.5. An expert that misses
the card's cache costs about 10x its compute time over Gen1, and 28x from the SSD. The design
has to miss rarely: size the cache, choose the eviction policy, keep experts in 4 bits. It must
also hide what it can of each miss behind the card's own work.

## 2. Models

`python3 tools/offload/survey.py --json survey.json` reads the safetensors headers of each
checkpoint, leaving out vision, audio and MTP weights. The columns:

- **Bytes:** our formats: fp4 blocks (4.25 bits per weight, docs/quant.md), with the LM head in
  int8.
- **Bytes / token:** what a decode token streams: dense, shared, top-k experts and head. The
  embedding row is looked up, not streamed.
- **On-card part:** what must stay on the card: dense, shared, head, and the embedding in int8
  when it is not tied.
- **Fits card:** allows 0.3 GB for KV / state, I/O and programs.
- **Fits host RAM:** the fp4 model in 26 GB.

| Model | Layers (MoE) | Experts, top-k | One expert | Params, total / active | Bytes / token | On-card part | Expert pool | Fits card / host RAM |
|:--|--:|--:|--:|--:|--:|--:|--:|:--|
| gemma-4-26B-A4B | 30 (30) | 128, top-8 | 3.16 MB | 25.2 / 3.82 B | 2.40 GB | 1.64 GB | 12.1 GB | no / yes |
| gemma-4-E4B | 42 (0) | dense | - | 7.5 / 4.70 B | 2.83 GB | 2.83 GB | 0.0 GB | no (yes without the PLE) / yes |
| LFM2.5-8B-A1B | 24 (22) | 32, top-4 | 5.85 MB | 8.5 / 1.69 B | 1.03 GB | 0.51 GB | 4.1 GB | no / yes |
| LFM2-8B-A1B | 24 (22) | 32, top-4 | 5.85 MB | 8.3 / 1.56 B | 0.89 GB | 0.38 GB | 4.1 GB | no / yes |
| LFM2-24B-A2B | 40 (38) | 64, top-4 | 5.01 MB | 23.8 / 2.33 B | 1.30 GB | 0.54 GB | 12.2 GB | no / yes |
| Qwen3-30B-A3B | 48 (48) | 128, top-8 | 2.51 MB | 30.5 / 3.04 B | 1.77 GB | 1.13 GB | 15.4 GB | no / yes |
| Qwen3.5-35B-A3B | 40 (40) | 256, top-8 + shared | 1.67 MB | 34.7 / 2.95 B | 1.82 GB | 1.81 GB | 17.1 GB | no / yes |
| Qwen3-Next-80B-A3B-Instruct | 48 (48) | 512, top-10 + shared | 1.67 MB | 79.7 / 3.56 B | 2.05 GB | 1.57 GB | 41.1 GB | no / no |
| Qwen3.5-122B-A10B | 48 (48) | 256, top-8 + shared | 5.01 MB | 122.1 / 9.01 B | 5.17 GB | 4.03 GB | 61.6 GB | no / no |
| gpt-oss-20b | 24 (24) | 32, top-4 | 13.22 MB | 20.9 / 3.61 B | 2.21 GB | 1.53 GB | 10.2 GB | no / yes |
| gpt-oss-120b | 36 (36) | 128, top-4 | 13.22 MB | 116.8 / 5.13 B | 3.02 GB | 1.71 GB | 60.9 GB | no / no |
| OLMoE-1B-7B-0125 | 16 (16) | 64, top-8 | 3.34 MB | 6.9 / 1.18 B | 0.68 GB | 0.36 GB | 3.4 GB | yes / yes |
| granite-3.1-3b-a800m-instruct | 32 (32) | 40, top-8 | 1.25 MB | 3.3 / 0.88 B | 0.51 GB | 0.19 GB | 1.6 GB | yes / yes |
| granite-4.0-h-tiny | 40 (40) | 64, top-6 + shared | 1.25 MB | 6.9 / 1.47 B | 0.86 GB | 0.55 GB | 3.2 GB | yes / yes |
| granite-4.0-h-small | 40 (40) | 72, top-10 + shared | 5.01 MB | 32.2 / 8.80 B | 4.88 GB | 2.88 GB | 14.4 GB | no / yes |
| DeepSeek-V2-Lite | 27 (26) | 64, top-6 + shared | 4.60 MB | 15.7 / 2.45 B | 1.41 GB | 0.91 GB | 7.6 GB | no / yes |
| Moonlight-16B-A3B-Instruct | 27 (26) | 64, top-6 + shared | 4.60 MB | 16.0 / 2.58 B | 1.54 GB | 1.17 GB | 7.6 GB | no / yes |
| SmallThinker-21BA3B-Instruct | 52 (52) | 64, top-6 | 3.13 MB | 21.5 / 3.33 B | 1.96 GB | 1.39 GB | 10.4 GB | no / yes |
| SmallThinker-4BA0.6B-Instruct | 32 (32) | 32, top-4 | 1.88 MB | 4.3 / 0.86 B | 0.58 GB | 0.58 GB | 1.9 GB | yes / yes |
| Phi-mini-MoE-instruct | 32 (32) | 16, top-2 | 6.27 MB | 7.6 / 2.23 B | 1.25 GB | 0.99 GB | 3.2 GB | no / yes |
| Mixtral-8x7B-v0.1 | 32 (32) | 8, top-2 | 93.59 MB | 46.7 / 12.75 B | 6.84 GB | 0.98 GB | 24.0 GB | no / yes |

Notes:
- **Gemma 4 E4B is dense** (`enable_moe_block` false). Its 2.82 B per-layer-embedding
  parameters (1.5 GB in fp4) are what push it over 4 GiB: 2.83 GB without them, 4.33 GB with.
- **The Gemma 4 MoE is gemma-4-26B-A4B.** It has 30 layers of 128 experts (width 704, top-8).
  Every layer also has a dense 2112-wide MLP beside the MoE, and the two outputs are added. The
  router reads the post-attention residual through its own RMSNorm, then applies softmax,
  renormalizes the top-8, and multiplies by a learned per-expert scale.
- The ones that already fit (OLMoE, Granite 3.1 3B, Granite 4.0 H Tiny, SmallThinker-4B) need
  no offloading, only the MoE block.
- Layer types:
  - Already covered: LFM2-MoE (convolution + attention: `opentpu/llm/lfm2.py`), Qwen3-MoE
    (`qwen3.py`), and Qwen3.5-MoE / Qwen3-Next (Gated DeltaNet + gated attention: `qwen35.py`).
  - Gemma 4 comes with the `gemma4` port (E2B).
  - New layers needed: gpt-oss (attention sinks, clamped SwiGLU with biases, MXFP4), DeepSeek-V2
    (MLA), Granite 4 (Mamba2).
- gpt-oss ships its experts in MXFP4 (E2M1, a power-of-two scale per 32). Our two-level format
  holds a 128-block of it exactly when its four sub-block exponents span at most 3 (a bf16 base
  times multipliers 1, 2, 4, 8). A conversion can check that per block (not measured).

## 3. Router traces and the expert cache

`tools/offload/router_trace.py` runs the Hugging Face model in bf16 on CPU, on omarchy. The
weights are memory-mapped from the checkpoint under a cgroup memory limit, or offloaded to disk.
It runs over four texts of 2048 tokens each, teacher-forced:

- prose: the first chapters of *Pride and Prejudice*;
- wiki: the Wikipedia article "Roman Empire";
- code: `opentpu/isasim.py`;
- technical markdown: `docs/host.md`.

Routing is causal, so a token's experts in this prefill are the ones a decode producing that
text would pick. For every MoE layer and token, the tool records the top-k and three predictions
of it. Each prediction applies the layer's router, through its norm, to an earlier state:

- `pre`: the layer's input, before the mixer;
- `prev_r`: the previous layer's router input;
- `prev_in`: the previous layer's input.

A check re-runs each router on the recorded residual. It gets back the recorded experts for this
share of the tokens:

- LFM2.5: 98.9-99.1%;
- Gemma 4: 92.8-98.4%;
- Qwen3.5-35B-A3B: 83-89%, whose 8 of 256 experts leave more near-ties.

The rest are ties between the k-th and (k+1)-th expert. bf16 logits make those common, and a
second top-k (of 2k) breaks them the other way. The recorded choice is the model's own.

`tools/offload/cachesim.py` replays each trace in decode order: token by token, layer by layer,
a layer's k experts at once. It plays them against a cache of C expert slots, under six
policies:

- `static`: the C experts most picked in the other three texts, never replaced;
- `lru`: shared by all layers;
- `lru_layer`: the slots split evenly over the layers;
- `lfu_layer`: per layer too, the victim the cached expert of least decayed use (section 5.5);
- `lfu`: counts, with the profile as a prior;
- `opt`: Belady, the upper bound.

Every policy starts from the static set, because the host fills the card from a profile at
load. Hit rates are the mean over the four texts, each scored with the profile of the other
three.

Hit rates, and a token's misses under the per-layer LRU (out of L x k requests):

| Model | Cache | static | lru | lru_layer | lfu | opt | misses / token (lru_layer: mean, p90) |
|:--|:--|--:|--:|--:|--:|--:|--:|
| LFM2.5-8B-A1B | 70 slots (10%) | 0.174 | 0.000 | 0.475 | 0.223 | 0.578 | 46.2, 59 of 88 |
| LFM2.5-8B-A1B | 141 slots (20%) | 0.321 | 0.554 | 0.586 | 0.381 | 0.768 | 36.4, 50 of 88 |
| LFM2.5-8B-A1B | 211 slots (30%) | 0.425 | 0.691 | 0.704 | 0.504 | 0.861 | 26.0, 38 of 88 |
| LFM2.5-8B-A1B | 352 slots (50%) | 0.642 | 0.874 | 0.862 | 0.722 | 0.954 | 12.1, 20 of 88 |
| LFM2.5-8B-A1B | 528 slots (75%) | 0.872 | 0.983 | 0.966 | 0.924 | 0.994 | 3.0, 6 of 88 |
| LFM2.5-8B-A1B | 595 slots (85%), the card | 0.942 | 0.996 | 0.986 | 0.971 | 0.998 | 1.2, 3 of 88 |
| gemma-4-26B-A4B | 192 slots (5%) | 0.084 | 0.000 | 0.459 | 0.144 | 0.576 | 129.8, 171 of 240 |
| gemma-4-26B-A4B | 384 slots (10%) | 0.168 | 0.536 | 0.573 | 0.261 | 0.756 | 102.4, 143 of 240 |
| gemma-4-26B-A4B | 744 slots (19%), the card | 0.293 | 0.755 | 0.761 | 0.437 | 0.885 | 57.3, 91 of 240 |
| gemma-4-26B-A4B | 768 slots (20%) | 0.302 | 0.765 | 0.770 | 0.448 | 0.890 | 55.3, 89 of 240 |
| gemma-4-26B-A4B | 1152 slots (30%) | 0.437 | 0.874 | 0.869 | 0.606 | 0.947 | 31.4, 55 of 240 |
| gemma-4-26B-A4B | 1920 slots (50%) | 0.670 | 0.971 | 0.963 | 0.827 | 0.988 | 8.9, 17 of 240 |
| gemma-4-26B-A4B | 2880 slots (75%) | 0.904 | 0.997 | 0.995 | 0.977 | 0.999 | 1.2, 3 of 240 |
| Qwen3.5-35B-A3B | 512 slots (5%) | 0.072 | 0.433 | 0.471 | 0.164 | 0.659 | 169.4, 226 of 320 |
| Qwen3.5-35B-A3B | 1024 slots (10%) | 0.138 | 0.598 | 0.608 | 0.278 | 0.776 | 125.4, 179 of 320 |
| Qwen3.5-35B-A3B | 1307 slots (13%), the card | 0.177 | 0.650 | 0.656 | 0.331 | 0.815 | 110.2, 162 of 320 |
| Qwen3.5-35B-A3B | 2048 slots (20%) | 0.264 | 0.748 | 0.748 | 0.451 | 0.882 | 80.8, 124 of 320 |
| Qwen3.5-35B-A3B | 3072 slots (30%) | 0.386 | 0.845 | 0.835 | 0.587 | 0.933 | 52.8, 84 of 320 |
| Qwen3.5-35B-A3B | 5120 slots (50%) | 0.610 | 0.943 | 0.929 | 0.801 | 0.977 | 22.6, 37 of 320 |
| Qwen3.5-35B-A3B | 7680 slots (75%) | 0.862 | 0.989 | 0.982 | 0.956 | 0.996 | 5.9, 12 of 320 |

Prediction accuracy: the share of a layer's top-k that the prediction's top-k names.

| Model | pre | prev_r | prev_in |
|:--|--:|--:|--:|
| LFM2.5-8B-A1B | 0.82 | 0.67 | 0.72 |
| gemma-4-26B-A4B | 0.79 | 0.72 | 0.65 |
| Qwen3.5-35B-A3B | 0.80 | 0.64 | 0.71 |

- **Adapt, do not profile.** At the card's size, a fixed set of the experts most picked in the
  other texts catches:
  - 0.92-0.96 of LFM2.5's picks (85% of the pool);
  - only 0.22-0.41 of Gemma 4's (19%), whose own-text profile would catch 0.80;
  - 0.11-0.23 of Qwen3.5-35B-A3B's (13%).

  Which experts are popular depends on the text. LRU adapts within a few tokens (0.74-0.78 on
  every text). LFU seeded with the profile adapts too slowly (0.40-0.51).
- **Global LRU thrashes below one token's sweep.** A token asks every layer in turn, a loop of
  L x k experts. At 5% of Gemma 4's pool (192 slots, fewer than the 240 a token asks for) it
  hits nothing. Split per layer, it degrades gracefully. At the card's sizes the two are within
  a point. The runtime splits its slots per layer, which also makes eviction race-free on the
  card (section 5.2).
- **Decayed use beats recency.** Per layer, replacing the expert whose uses, each halving every
  32 requests of its layer, sum least (`lfu_layer`) misses 9-15% less than `lru_layer` at the
  card's sizes (misses a token):

  | Model | slots | lru_layer | lfu_layer |
  |:--|:--|--:|--:|
  | LFM2.5-8B-A1B | 28 a layer | 0.83 | 0.73 |
  | gemma-4-26B-A4B | 682, fp4 experts and int8 head (section 11.3) | 62.9 | 53.6 |
  | gemma-4-26B-A4B | 789, fp4 head | 53.5 | 45.3 |
  | Qwen3.5-35B-A3B | 1280, the card's (32 a layer) | 111.4 | 100.7 |

  Half-lives of 16 to 128 requests are within a point. Allotting the slots unevenly over the
  layers (from the other texts' curves) gains nothing. Unlike the undecayed `lfu` above, it
  forgets a text's old experts within some 100 tokens.
- **Room above LRU.** Belady's bound misses half as often as LRU: 27.6 against 57.3 per token
  on Gemma 4, and 59 against 110 on Qwen3.5. A policy that knew reuse better could find up to
  2x fewer misses, but the predictions below do not (section 5.4).
- **Predictions.**
  - They name 79-82% of a layer's experts one mixer ahead (`pre`), and 64-72% one layer ahead
    (`prev_r`).
  - Of the per-layer LRU's misses at the card's size, the prediction's 2k best hold 90-92%
    (`pre`) and 71-81% (`prev_r`).

## 4. Path (a): tokens per second

`cachesim.py --survey survey.json` replays each trace through an event model of path (a)
(`linksim`): the card computes everything, and the host moves the experts it lacks over PCIe.
Per token, and per MoE layer in order:

1. **The card runs the layer's mixer and router** (`d_pre` bytes of weights). It then posts its
   k expert ids: an ST to a mailbox in DRAM.
2. **The host sees the request `t_req` later.** It DMAs the missing experts one after another
   into LRU victim slots of that layer's cache. Each costs x / B_pcie + t_call. The link is one
   queue across layers.
3. **The card, meanwhile, computes what needs no missing expert:**
   - the part of the layer that needs no expert (`d_post`: Gemma 4's dense MLP, Qwen3.5's shared
     expert);
   - then the experts it has.
4. **It computes each missing expert as soon as that expert lands**, plus `t_done` for the
   directory write the host makes after the DMA and the card's `WAITW` seeing it (section 5.2).
   Each transfer's DRAM writes are charged to the card's time.
5. **After the last layer, the LM head.**

| Constant | Value | Source |
|:--|--:|:--|
| B_dram, card DRAM while decoding | 14.1 GB/s | measured (README) |
| B_pcie, host -> card | 1.3 GB/s (Gen1), 2.6 GB/s (Gen2) | measured / *estimate* x2 |
| t_call, per DMA call | 30 us | *estimate* (11-15 us measured per small call) |
| t_req, the card's ST to the host's first DMA | 30 us | *estimate*: the host polls the mailbox, one small c2h read |
| t_done, a DMA's end to the card's WAITW seeing its flag | 15 us | *estimate*: one small h2c write |
| B_ssd | 0.51 GB/s | measured |

The cache is the per-layer LRU at the card's size from the survey: 4 GiB less the on-card part
and 0.3 GB. It is warmed from the profile of the other texts, and all four texts are averaged.

| Model | card cache | all resident (bound) | Gen1, wait for the last | **Gen1** | **Gen2** | misses / token | link busy, Gen1 |
|:--|:--|--:|--:|--:|--:|--:|--:|
| LFM2.5-8B-A1B | 595 slots (85%) | 13.73 | 12.78 | **13.02** | **13.48** | 1.2 of 88 | 7% |
| gemma-4-26B-A4B | 744 slots (19%) | 5.88 | 3.38 | **3.86** | **5.00** | 57 of 240 | 54% |
| Qwen3.5-35B-A3B | 1307 slots (13%) | 7.75 | 3.69 | **4.15** | **5.77** | 110 of 320 | 60% |

(tok/s, *simulated*. "wait for the last": the card computes a layer's k experts only after its
last missing one lands.)

- **Nearly fits (LFM2.5-8B-A1B).** 85% of the experts stay on the card, and a token misses 1.2
  of its 88. Streaming them over Gen1 runs at 95% of the all-resident bound; offloading is close
  to free.
- **Three times the card (gemma-4-26B-A4B, 13.8 GB at fp4).**
  - A token misses 57 of its 240 experts: 180 MB, 138 ms of Gen1, against 170 ms of card time.
  - Waiting for each layer's last expert gives 3.38 tok/s. Computing the ones the card has while
    the others stream gives 3.86 (+14%). Gen2 gives 5.00 (+30%).
  - What is left is the link. It is 54% busy at Gen1 and in series with the card on every
    layer that misses.
- **Four times the card (Qwen3.5-35B-A3B, 18.9 GB at fp4).**
  - A token misses 110 of its 320 experts: 184 MB, 141 ms of Gen1, against 129 ms of card time.
  - Its experts are the smallest here (1.67 MB), so it misses more of them than Gemma 4 does,
    but each costs less.
  - It gets 4.15 tok/s at Gen1 (+12% from hits first) and 5.77 at Gen2 (+39%), against a
    bound of 7.75.
- **Latency does not matter.**
  - With t_req / t_done / t_call at 10 / 5 / 15 us: LFM2.5 13.03, Gemma 4 3.89.
  - At 100 / 50 / 30 us: 13.00 and 3.83.
  - The bytes are what count, so a card-side DMA engine that saves the host's reaction time
    (the XDMA descriptor bypass) is not worth building.
- **What limits Gemma 4 beyond the link is its int8 LM head:** 761 MB of the 2.4 GB a token
  reads (262,144 x 2816). An fp4 head would raise the bound from 5.9 to ~6.9 tok/s (*estimate*).

Other levers (same model, Gen1 / Gen2):

| Lever | LFM2.5-8B-A1B | gemma-4-26B-A4B |
|:--|--:|--:|
| path (a) as above: fp4 experts, card cache | 13.02 / 13.48 | 3.86 / 5.00 |
| int8 experts (8.25 bits: twice the bytes, half the slots) | 4.56 / 6.64 (306 slots; bound 9.33) | 1.68 / 2.75 (383 slots; bound 4.53) |
| smaller cache (more KV reserve) | 10.60 / 12.40 (450 slots); 7.48 / 10.34 (300) | 3.19 / 4.52 (500 slots) |
| best prefetch (section 5.4) | 12.90 / 13.45 (`pre`, k best) | 3.39 / 4.68 (`prev_r`, k best) |
| prefetch of the prediction's 2k best | 12.34 / 13.09 | 2.61 / 3.76 |

On Qwen3.5-35B-A3B the best prefetch (`pre`, k best) gives 3.89 / 5.78, against 4.15 / 5.77.

- **4-bit experts** are the largest lever after the cache itself. They halve the bytes per
  miss and double the slots in the same DRAM.
- **The cache size** matters most for the models that nearly fit. LFM2.5 loses 19% going from
  595 to 450 slots, so its KV reserve should stay small.

## 5. Design

### 5.1 Where things live

- **Card DRAM** (4 GiB):
  - **the resident part:** dense layers, shared experts, norms, the LM head, the KV cache and
    convolution / DeltaNet state, I/O, and programs;
  - **the expert cache:** fixed-size slots, one expert per slot, split per MoE layer (section
    5.5). Every part of an expert is contiguous in its slot, with the gate, up and down rows and
    their scale words at fixed offsets, so one base address names an expert. All experts of a
    model have one size, so the slots need no allocator;
  - **the directory:** one entry {slot address, present} per global expert id (the j-th MoE
    layer's expert e is j x E + e). The host writes it; the card reads it;
  - **a mailbox:** `seq`, the card's last request, and one row of its k ids; `served`, the last
    request the host has finished.
- **Host RAM:** the **expert pool**, every expert already in the card's slot format (fp4 rows
  and scale words), in 4 KiB-aligned buffers placed for fast DMA (`board.DMA_PLACE`,
  docs/host.md). The host also keeps its own copy of the directory and the LRU state. A miss is
  one DMA write, with no conversion. Every model in the table except Qwen3-Next-80B,
  Qwen3.5-122B and gpt-oss-120b fits in 26 GB at fp4.
- **SSD:** the pool as one file of slot-sized records, for models larger than host RAM
  (section 8).

### 5.2 Control: the card routes, the host serves

The card runs the whole reply as one run: autodecode's decode loop, in which the card samples
and feeds back each token. The MoE layers wait inside it. Per MoE layer, as built in phase 2
(`opentpu/llm/moe.py`, branch offload-p2):

1. **Route.**
   - The router MM gives E scores, with the model's rule on the VPU: LFM2 takes sigmoid plus
     the expert bias for the choice; Gemma 4 takes softmax.
   - A `LOOP` of k `ARGMAX` with knock-out (`RLD` of the position, a `FILL` of -3e38 there)
     gives the k ids in order and their scores; ties go to the first, as `np.argmax`.
   - The combine weights are the model's: renormalized and scaled. The global ids are the
     ids plus the layer's j x E (a word in the layer's block).
2. **Fence.** `WAITW served GE seq`, with `seq` the card's last request, read from the
   mailbox: the host has finished every earlier request. From here on, the host's only work on
   this layer's slots is the request about to be posted, which never evicts an expert it
   names. The fence is normally immediate: the host served the previous layer while the card
   computed it.
3. **Post.** The k ids to the mailbox's row, then `seq + 1` in a second `ST`, so the host never
   reads a torn request. The fence comes first, so one row is enough. `seq` is a float, and
   positive floats compare by their bits, so it works directly in `WAITW`'s GE compare.
4. **Look up.** A `LOOP` over the ids: `RLD` of the entry's offset (id x 8, one `VOP` for
   all), `LD` of its present flag.
5. **Hits first.** A second `LOOP` over the ids with `LOOP R[present]` inside:
   - `WAITW NE 0` on the entry's address word copies the slot address to a TMEM word at once,
     and an `RLD` (RAW) puts it into a register;
   - the expert's MMs run at that register plus fixed offsets (`kernels.mlp.swiglu_down`);
   - its weighted output goes to its own row of a [k, H] tile.
6. **Misses.** A third `LOOP` with count 1 - present: the same code. There, `WAITW` waits until
   the host has written the entry after the expert's DMA, so each missing expert is computed
   as soon as it lands while the others still stream.
7. **Combine.** The k rows summed in the router's order, then added to the residual. The logits
   therefore do not depend on what the cache held.

A slot address never passes through the VPU, which would flush it as a denormal: `WAITW`
copies it from the directory to TMEM and `RLD` (RAW) takes its bits into the register. The MoE
block holds one register in all. The
resident decode's run arguments and the layer loops hold the others: LFM2.5-8B-A1B's generate
program had 2 of 15 left. It needs no index register: the per-expert values are the columns of
a small tile that each loop rotates by one, so expert i is always at column 0.

The host daemon (`opentpu/host/offload.py`, phase 2) only moves data:

1. It polls `seq` with a small c2h read.
2. It reads the k ids.
3. For each id missing from its directory copy, it:
   - picks the layer's LRU victim among the experts the request does not name;
   - writes {0, 0.0} to the victim's entry;
   - DMAs the expert into the slot;
   - writes {slot address, 1.0} to the entry.
4. It writes `served = seq`.

Hits need nothing from the host but its bookkeeping. The mailbox, directory and `served` are
ordinary DRAM words: the host reads and writes them the way `run_generate` already reads
tokens and writes the stop word during a run.

**Why it is race-free.** The only hazard would be the host evicting an expert the card is
using. Each layer has its own slots. Once the fence has passed, the only host work on a
layer's slots is the request the card then posts, which protects its own ids. The card uses a
layer's slots only between its fence and its next fence for that layer, a token later. With
S slices, only slice 0 would post, and every slice would run the same fence and waits.
Phase 2 runs one slice, as the board does.

**The instruction: `WAITW` (0x07), reserved by autodecode, run by the DMA.** Its fields follow
`LD`'s. It waits until `cmp(M32[R[ra] + w1] & w4, R[rc] + w3)` holds, then writes the word to
TMEM at `R[rb] + w2`, and an `RLD` (RAW) takes it into a register.

- Flags bits 1:0 choose the compare: 0 EQ, 1 NE, 2 GE. GE means the 32-bit difference is
  >= 0 as signed.
- `w5`: cycles between polls. The first read is immediate.
- `w6`: a timeout in cycles, 0 for none. On timeout the DMA stops the slice with an error the
  host sees, so a dead daemon cannot hang the card silently.
- It runs on the DMA unit (autodecode's form), polling through the DMA's DRAM read.
  - Its footprint is all of DRAM (read) plus the one TMEM word. Every younger instruction that
    reads or writes DRAM waits for it, and older stores (the mailbox's) land before it polls.
  - The sequencer needs nothing beyond RLD's hold.
- One requirement on the RTL: after `WAITW` sees a word the host wrote after a completed h2c
  DMA, every younger MM or LD reads that DMA's data. The host orders its data before its flag;
  the card orders its flag before its reads. The XDMA and the core meet in each channel's
  `otpu_mem_ch` and LiteDRAM, whose ordering must give this. team-lead has made this a gate of
  ld-memch's two-port work.
- In the ISA simulator, the host hook is called when every slice that can run is waiting: the
  simulated daemon, which copies experts and writes the directory. A `WAITW` that still does not
  hold is the timeout.

Everything else is built from instructions autodecode already added: `ARGMAX` (func 20), `RLD`,
`LOOP R[ra]` (predication), and `HALT CHAIN` when a program outgrows the 4096-instruction IMEM.

The hits-first split costs, per expert, an RLD of the entry's offset, an LD of its flag, an
RLD of the flag, one `WAITW` read and an RLD of the slot address: a few DRAM round trips per layer, ~0.2% of a token
(*estimate*).

**Not needed.**
- Halts per MoE layer. The first version of this study had one per MoE layer, 1.3-2.9 ms per
  token.
- Baking run-time values into immediates.
- A TOPK instruction: `ARGMAX` knock-out does it at k <= 8.
- The XDMA descriptor bypass (the card fetching by itself): latency does not matter
  (section 4), and the bypass would need a kernel driver pinning the pool.

### 5.3 Overlap

DMA into the card's DRAM runs while the card computes. The XDMA shares LiteDRAM's ports with
the core through `otpu_mem_ch`, and the streamed logits are already read during runs. On a
layer with misses, the card computes in this order while the link works:

1. the work that needs no expert (Gemma 4's dense MLP, Qwen3.5's shared expert);
2. its resident experts;
3. each missing expert as it lands.

On Gemma 4 that is +14% over waiting for the last expert, and on LFM2.5 +2% (it misses
rarely). The DMA's writes cost the card DRAM time, x / B_dram per expert, which the model
charges. That is 9% of the expert's transfer time at Gen1 and 18% at Gen2.

### 5.4 Prefetch: measured not to pay

The traces measure three early predictions of a layer's experts (section 3). In the model, a
prediction posts prefetches as a hint in the mailbox:

- it takes a slot at once (an LRU victim);
- it uses the link only when no demand transfer needs it (a demand preempts it at the next
  512 KiB chunk, each chunk one DMA call);
- its DRAM writes are charged to the card.

Every variant loses:
- with the k best: 1% on LFM2.5, and 12-14% on Gemma 4 at Gen1 (`pre` 3.34, `prev_r` 3.39,
  against 3.86);
- with the 2k best: 5-6% on LFM2.5 and 32-35% on Gemma 4.

The reasons, on Gemma 4 with `prev_r`, per token:
- 32 prefetches land, but 25 of them are evicted unused;
- the slots they take raise the demand misses from 57 to 66;
- the link goes from 54% to 84% busy.

The correct predictions are mostly experts the LRU already holds. The misses are the experts
that are hard to predict.

A perfect predictor would at most start each transfer one mixer earlier: Gemma 4's `d_pre` is
1.4 ms against 2.4 ms per expert over Gen1.

Prefetch does help when the card waits for a layer's last expert: +7% on Qwen3.5 at Gen2 (5.35
against 4.99). Computing the hits first already hides that time (5.77). The design therefore
carries no hints, and the ISA needs nothing for them.

With the slots replaced by decayed use (section 5.5), wrong guesses cost less: a prefetched
expert holds no use until a request names it, so the next miss replaces it first, not a used
expert. At 1.4 GB/s the k best of `pre` then pay on Qwen3.5-35B-A3B: 4.42 against 4.15 tok/s
at 1280 slots (10 of 23 prefetches a token wasted, against 33 of 49 under LRU). On Gemma 4
they break even (3.79 against 3.78; `prev_r` 3.86). It would need the card to run each layer's
router before its mixer and post the hint; not built.

### 5.5 Eviction

Per layer, warmed at load from a profile, with every resident weight pinned outside the cache
(dense, shared experts, norms, head). The victim is the cached expert, not in the request, of
least decayed use (`ExpertServer` policy "lfu", the default since card session 4, section
10.3):
- each use counts 2^(-age / 32), its age in requests of its layer;
- a warm expert counts one use at load;
- ties go to the least recently used.

The server keeps each expert's sum as log2 + t / 32, one update per use. It misses 9-15% less
than per-layer LRU (section 3), and on the card 2.4% less over the 35B's first 16 tokens (6.4%
over their second half).

- Undecayed LFU and static profiles lose to LRU, by 30-50 points on Gemma 4.
- Belady's bound (0.885 against 0.761 on Gemma 4) is the headroom, but predictions do not
  reach it (section 5.4).
- The per-layer split is within a point of a global LRU at the card's sizes. It is what makes
  eviction race-free without extra synchronization (section 5.2).

### 5.6 Prefill

A prefill chunk touches nearly every expert of every layer (the union of k picks over hundreds
of tokens). So it streams a layer's missing experts once per chunk, not per token.

- The card computes prefill at its MXU rate. Qwen3-0.6B 4-bit measures 103.4 tok/s: 2 x
  parameters x tok/s, about 124 G operations per second.
- For gemma-4-26B-A4B's 3.8 B active parameters that is ~16 tok/s (*estimate*). A 512-token
  chunk takes ~1 s per layer, against ~0.3 s to bring a whole layer's 128 experts (405 MB)
  over Gen1.
- Fetching layer l + 1 while layer l computes hides the transfer for chunks above ~150 tokens.

The card-side protocol covers this too: a prefill layer posts the union of its chunk's picks
and computes the resident ones first. Prefill stays compute-bound, as it is today.

### 5.7 Correctness and tests

Every model stays token-exact against the Hugging Face reference (greedy): first in the ISA
simulator, then the RTL, then the card. Offloading changes where an expert's bytes come from,
never which bytes, so the expert cache is invisible in the outputs.

The ISA simulator runs with a DRAM the size of the card's (4 GiB) and with the daemon as its
host hook. Tests:
- the logits are bit-identical to an all-resident run, on a model that fits, with a small cache
  that forces misses;
- the protocol's corner cases: all hits, all misses, a request naming an expert that another
  request's DMA is evicting, a lagging host (the fence), and the timeout.

The router runs on the card in every backend, so the three agree bit for bit. The one new risk
is routing near-ties. The card's router logits differ from bf16's, so the k-th and (k+1)-th
experts can swap where HF's would not. The check against HF will show whether greedy tokens
move, as the argmax can today.

### 5.8 What path (a) needs from autodecode

With the router on the card there is no tension left with autodecode: the card samples, feeds
back its tokens and waits for experts within one run. From autodecode's loop, path (a) needs:

1. `WAITW` in the ISA, the simulator (with the host hook) and the RTL, plus the ordering
   requirement above.
2. A host-side daemon beside `BoardBackend.run_generate`'s token reader. It shares the DMA
   device files, with one lock around each call. The token reader and the daemon are the only
   host work during a reply.
3. Room in the program. An MoE layer's expert part is three short loops. LFM2.5-8B-A1B's
   generate programs are 2577-3722 instructions of IMEM's 4096. If a model's decode step
   outgrows IMEM, it `CHAIN`s between parts, as autodecode's buckets already do.
4. Registers. The resident decode's run arguments (6 for LFM2.5: R10-R15) and the address
   registers of its layer loops leave 2-3 of R1-R15 free inside autodecode's loop. That is why
   the MoE block holds exactly one (section 5.2). A second MoE-sized feature in the same step
   would need registers back from the run arguments or the layer loops (autodecode's and
   models3's IMEM and loop work).
5. Nothing from sampling: the MoE is inside the decode step, which autodecode's loop runs
   unchanged.

### 5.9 The token's embedding row

In autodecode's loop the card feeds back its own token, so the next token's embedding row must
come from the card's DRAM. The resident decode keeps the whole table there in fp32 (vocab x H
x 4 bytes, `Image(lookup=True)`); the survey's "on-card part" (section 2) did not count it.

- **LFM2.5-8B-A1B:** 1.05 GB. With it, the 4 GiB hold 20 expert slots per layer (440 of 704,
  62%) instead of ~27. At 450 slots the model above gives 10.6 tok/s at Gen1, against 13.0 at
  595.
- **gemma-4-26B-A4B:** 262,144 x 2816 x 4 = 2.95 GB, which cannot fit beside its 1.64 GB
  on-card part.

**The fix: the card gathers the row from the tied int8 LM head.** Both models tie their
embedding to the head. An MM against a one-hot operand reads the token's row out of the head
the card already holds.
- It uses no extra DRAM and no host, and gets LFM2.5 back to ~27 slots per layer (~13 tok/s at
  Gen1, by the model above).
- The shared `kernels/gather.py` (main 77405e5) does this, and phase 2 uses it: LFM2.5-8B-A1B
  gets 28 slots per layer, and Qwen3.5-35B-A3B 32 instead of 9. The 35B's embedding is untied,
  so its rows come from an int8 table of 0.5 GB. The int8 embeddings of SmolLM3,
  Phi-4-mini and Qwen3.5-4B use it too.
- It is `Spec.embed` "int8" (main 2a04beb, `qwen3._embed`), which a MoE checkpoint's `from_hf`
  sets at any vocabulary size: beside a MoE's layers the DRAM is expert slots. The prompt's
  per-position programs and the resident decode then read every input from the image, so the
  host writes no embedding or RoPE row.

A host-written row fetched by `WAITW` (section 7) stays for rows that are not in the card at
all, such as E4B's per-layer embeddings. Phase 2's first 8B run kept the fp32 table (20 slots
per layer).

**An untied table on the host (`embed_host`).** A model whose int8 embedding table is its own
(untied, as Qwen3.5-35B-A3B's 0.5 GB) can keep it on the host instead, the way E4B keeps its
PLE records:
- The card holds a slot of the image's rows (one for a MoE engine), each row a record: its
  H int8 values, then its H / D scale words (the head's format), in whole MXU chunks
  (`qwen3.embed_record`), and a row mailbox (`RowLayout`).
- Before a run the host writes the run's rows into the slot (`Image.host_rows`, as for E4B).
  In the card's generate loop each sampled token is posted to the mailbox (`MB.post`), the
  host's `RowServer` writes its row into slot row 0 (one DMA call through the expert server's
  `BoardDram`, in order with its writes), and the next token's gather waits for `served`
  first (`MB.wait_served`, then `gather_row` of row 0). Data movement only: the host keeps
  the table as the card would hold it (`qwen3.embed_store`).
- The default: on for a MoE model with an int8 table of its own (its DRAM is expert slots),
  off for a dense one unless its image does not fit with the table (the Engine then retries
  with it on the host). A tied int8 head's rows stay the gather's. `Engine(embed_host=...)`
  and `moe_card --embed-table host|card` choose.
- Qwen3.5-35B-A3B at the board's 4 GiB: 34-35 slots a layer fit with the table on the card,
  42 (1680) with it on the host (cap 512-4096). The model of section 10.3 (decayed use, Gen1)
  gives 4.41 tok/s at 1560 slots and 4.45 at 1600, against 4.15 at the card runs' 1280
  (*projection*). Its programs change (the gather's wait and the
  post); every other model's are the same (sha256: E2B, E4B, LFM2.5-8B-A1B, the tiny MoEs;
  the 35B with `embed_host=False`).
- Tests: tests/test_qwen35_moe.py (prefill, the generate loop and the resident steps bit for
  bit against the table on the card; the live fake card with both servers through one
  `BoardDram`), test_lfm2_moe.py (the generate loop), test_qwen3.py (a dense model moved to
  the host when its image does not fit, the same logits).

## 6. PCIe Gen2

| Model | tok/s, Gen1 -> Gen2 | link busy, Gen1 -> Gen2 |
|:--|--:|--:|
| LFM2.5-8B-A1B | 13.02 -> 13.48 (+3.5%) | 7% -> 4% |
| gemma-4-26B-A4B | 3.86 -> 5.00 (+30%) | 54% -> 35% |
| Qwen3.5-35B-A3B | 4.15 -> 5.77 (+39%) | 60% -> 43% |

(tok/s, *simulated* as in section 4, with Gen2 taken as twice the measured Gen1 rate.)

Gen2 x8 doubles the link: 5 GT/s, with XDMA's AXI side at 128 bits and 250 MHz instead of 125.
- It pays in proportion to the bytes that cross PCIe. For a model 3x the card it is the largest
  single lever left after 4-bit experts and hits-first overlap.
- It also halves a prefill chunk's per-layer expert stream (405 MB: 0.31 -> 0.16 s) and the
  image load (4 GiB: ~3.3 -> ~1.7 s).

The cost is a build that may not close:
- Everything in xdma_aclk goes from 125 to 250 MHz: otpu_axi_split2, the XDMA side of
  otpu_mem_ch, the LiteDRAM CSR crossing and the AXI-Lite SmartConnect.
- The earlier Gen2 attempt missed by ~0.1 ns in the IP's own placed paths (an 80 MHz build,
  docs/board.md section 5).

**Verdict: parked until phase 2 runs LFM2.5-8B-A1B end to end, then the first build for
models 3-4x the card.** LFM2.5 gains 3.5%, which does not justify a Vivado slot. Gemma 4 26B
gains 30% and Qwen3.5-35B 39%, which do once they are the target. The build to try then is a
timing-only FAST=1 CORE_MHZ=100 build with `pl_link_cap_max_link_speed {5.0_GT/s}` and
`axisten_freq {250}` in `bd_native.tcl`.

## 7. Gemma 4 E4B: the per-layer embeddings from the host

E4B is dense, and misses 4 GiB only by its PLE table (262,144 x 42 x 256 = 2.82 B parameters).
A token uses one row of it: 42 x 256 values, 21.5 KB in bf16 (43 KB in fp32).

- **Where it lives.** The host keeps the table: 5.6 GB in bf16 in RAM, or 1.5 / 2.8 GB in
  fp4 / int8, in the format the card reads. The rest of E4B (2.83 GB) fits on the card, with
  ~1.1 GB to spare for its KV cache.
- **How the card gets the row.** The card asks for it like an expert. After sampling a token it
  posts the token id to the mailbox. The host DMAs that row into a fixed buffer (one write of
  ~22-43 KB, ~30-40 us at Gen1) and bumps a flag.
- **Why it costs nothing.** The card's first use of the row `WAITW`s on the flag. It is used
  only when layer 0 combines its per-layer input, after that layer's attention and MLP, so the
  ~0.1 ms round trip hides behind layer 0's ~4.8 ms of weights (*estimate*).
- **The host's part is a table lookup, i.e. data movement.** The E4B decode stays at ~5.0 tok/s
  (2.83 GB streamed at 14.1 GB/s, all resident, fp4 with an int8 head; *estimate*).

## 8. The SSD tier

Three models in the survey are larger than the host's RAM at fp4:

- Qwen3-Next-80B-A3B (41 GB of experts);
- gpt-oss-120b (61 GB);
- Qwen3.5-122B-A10B (62 GB).

The host RAM then becomes the second cache level: an LRU over the pool, about 24 GB of it, fed
by the card's misses. The SSD holds the pool as one file of slot-sized records. A host miss is
on the token's path: an SSD read at 0.51 GB/s (3.3 ms for a 1.67 MB expert, 26 ms for
gpt-oss's 13.2 MB), then the DMA.

The model below is `cachesim.py --host-frac`: the host RAM limited to part of the pool, the
SSD a queue of its own ahead of the link, and the card as in section 4. Two models that fit in
host RAM stand in for the larger ones:

| Model | host RAM holds | SSD reads / token | Gen1 | Gen2 |
|:--|--:|--:|--:|--:|
| gemma-4-26B-A4B | the whole pool | 0 | 3.86 | 5.00 |
| gemma-4-26B-A4B | 50% | 6.9 | 3.50 | 4.31 |
| gemma-4-26B-A4B | 25% | 34.3 | 2.44 | 2.68 |
| Qwen3.5-35B-A3B | the whole pool | 0 | 4.15 | 5.77 |
| Qwen3.5-35B-A3B | 50% | 18.4 | 3.67 | 4.63 |
| Qwen3.5-35B-A3B | 25% | 63.5 | 2.67 | 2.95 |

(tok/s, *simulated*.) At 25% the SSD is the limit, and Gen2 barely helps.

The three larger models:

- **Qwen3-Next-80B-A3B** is the SSD tier's best case. It has Qwen3.5's layers and expert size
  (1.67 MB), 512 experts and top-10.
  - Its card caches 1452 slots (5.9% of the pool), and host RAM holds ~58%.
  - Qwen3.5-35B-A3B's traces at those two fractions give 3.18 tok/s at Gen1 and 4.49 at Gen2,
    with 158 card misses and 11.6 SSD reads per token.
  - Qwen3-Next asks 1.5x as many experts per token (480 against 320). Scaling the time beyond
    the resident bound by that, and taking Qwen3-Next's own bound (2.05 GB per token), gives
    **~2.4 tok/s at Gen1 and ~3.5 at Gen2** (*extrapolated*).
- **gpt-oss-120b and Qwen3.5-122B-A10B** are link-bound.
  - Their experts are 13.2 and 5.0 MB. Their card caches are 172 slots (3.7%) and 150 slots
    (1.2%). Qwen3.5-122B's on-card part leaves room for those 150 only if its embedding rows
    come from the host, like E4B's PLE (section 7).
  - At Gemma 4's and Qwen3.5's per-layer LRU hit rates for such small caches (0.2-0.45), a
    token sends 1.1-1.5 GB over the link: **under ~1 tok/s at Gen1, ~1.5-2 at Gen2**
    (*estimate*), before any SSD reads.
  - They need a card with more DRAM, not a better policy.

Prediction cannot hide an SSD read (3-26 ms) behind a layer (3-9 ms). A faster disk moves the
host-RAM-limited rows directly; an NVMe drive at 2.5 GB/s would make the 25% rows link-bound.

Details for the pool file:
- fp4 does not compress, so write the file with btrfs compression off (`chattr +m`).
  Compressed extents fall back to the page cache, so this keeps O_DIRECT reads direct.
- Requests of one expert (1.7-13 MB) are in the 4 MiB regime measured above.

## 9. Phase 2

The target is LFM2.5-8B-A1B end to end on the ISA simulator: the card's 4 GiB DRAM, the
router on the card, per-layer LRU expert slots and host DMA. It is built on autodecode's ISA
(ARGMAX, RLD, CHAIN), with no Vivado and no card time.

Status (branch offload-p2, on autodecode's branch): items 1-4 are built. On a tiny LFM2 MoE
with 2 slots per layer, the logits are bit-identical to an all-resident run, in prefill and in
the card's generate loop. The MoE block also runs on the RTL (autodecode's WAITW, the Verilator
board model through the host driver) with every expert resident, bit-identical to the ISA
simulator. A live daemon is not tested there: the board model runs a fixed script, with no host
during a run.

LFM2.5-8B-A1B on the ISA simulator with the card's 4 GiB (fp4 experts, int8 head, the card's
generate loop), against HF's bf16 greedy tokens (`tools/offload/moe_card.py`):

| run | slots per layer | tokens | against HF | hit rate | misses per decode token |
|---|---|---|---|---|---|
| fp32 embedding table | 20 of 32 | 16 | token-exact | 86% | 9.9 |
| gathered embedding (section 5.9) | 28 of 32 | 144 | the first 19 | 97.8% | 1.54 (2.11 in the second half) |
| the same, a longer answer ("Describe the water cycle ...") | 28 of 32 | 160 | the first 4 | 98.5% | 0.87 (0.67) |

- Both first different picks are fp4's, not the device's. At token 19 HF's bf16 logits tie four
  ways (within 0.25). At token 4 of the longer run HF ties 600 and 358 at 46.75, with 278 at
  44.05. The card ranks 278, 358 and 600 at 46.65, 46.35 and 46.12. `lfm2.emulated_logits`'
  float64 decode of the same fp4 weights, with none of the device's rounding, gives 46.74,
  46.32 and 46.17 (`tools/offload/emul_step.py`, the experts fake-quantized per use):
  quantizing the weights moves 278 up 2.7 logits, and the device's arithmetic adds nothing.
- The misses are out of 88 expert uses per token.
- On the card's own routes, cachesim's per-layer LRU at 28 slots gives 98.0% and 1.7 misses per
  token (the card: 97.8%, 1.9 over all 163 tokens). The card warms its slots with experts 0..27,
  cachesim from a profile.
- By section 4's model, those routes run at 12.7 tok/s at Gen1 (13.4 at Gen2). The resident
  bound is 13.7.

Qwen3.5-35B-A3B (item 5) has:
- the softmax rule and the shared expert (`MoESpec.rule`, `shared`);
- 16 key heads for 32 value heads (main 4ab54a4);
- 32 slots per layer with the gathered embedding.

Its generate program fits IMEM with the DeltaNet pair loop (docs/qwen35.md, group-major
DeltaNet blocks): 2,282 instructions in bucket 1. On the ISA simulator with the card's 4 GiB
(4,084 MiB image, 1,280 slots of 1.67 MB, a pool of 10,240 experts) its 16 tokens are HF's
exactly, every one picked by the card's generate loop. Of 11,840 expert requests 62.4% hit;
decode missed 95.7 experts per token (113.4 in its second half), against 111 predicted.
cachesim's per-layer LRU on the card's routes gives 64.7% and 113.1 misses per token (37
tokens), its optimum 80.5% and 62.3. That run wrote the prompt's first three embedding rows
from the host. Run again with `Spec.embed` "int8" (above), every input comes from the image:
the same 16 tokens, and the same requests, hits and misses, since the gathered rows are the
host's bit for bit.

1. **The MoE block in `ol` kernels:**
   - the router MM;
   - the selection (autodecode's `_select`), with LFM2's sigmoid + bias rule and its
     renormalized, scaled weights;
   - the expert FFN (SwiGLU) at a slot base register;
   - the combine in a fixed order.

   Also the slot layout, the directory, the mailbox and the fence, and the pool in the card's
   format (`quant` per expert).
2. **`WAITW` in the ISA simulator**, in autodecode's DMA-side form, with its host hook.
3. **The daemon** (`opentpu/host/offload.py`):
   - the pool in host RAM;
   - the directory copy and per-layer LRU;
   - DMA of misses into slots.

   The same code serves the simulator (through the hook) and the card: `BoardBackend.host`
   polls it while a run is in flight, in `wait` and in `run_generate`'s token loop, on the
   one host thread. A fake card that computes in a thread over the host's DRAM
   (`tests/test_lfm2_moe.py`) checks it: with 2 slots per layer, misses are served while the
   card waits, and the logits and tokens are the ISA simulator's bit for bit.
4. **Tests:**
   - bit-identical logits against an all-resident run, on a small MoE with a tiny cache;
   - the protocol's corner cases (section 5.7);
   - LFM2.5-8B-A1B token-exact against HF (greedy), with the card's 4 GiB DRAM.
5. **Then** Gemma 4 26B-A4B on the `gemma4` port's layers, and Qwen3.5-35B-A3B on Qwen3.5's.
6. **The card** (through team-lead), done (section 10):
   - `WAITW` in RTL (autodecode) and its ordering check;
   - the misses' DMA while the core runs;
   - tok/s against the model above.

## 10. On the card

The production bitstream (build B, 79c5707a, 133.33 MHz, DDR3-1066, WAITW and the generate
loop) ran path (a) on 2026-09-30. The host side ran on opentpu, with no reload. The models were
LFM2.5-8B-A1B and Qwen3.5-35B-A3B, fp4 with an int8 head, and every token was picked by the
card's generate loop. The experts streamed from a packed pool file (each expert in the card's
slot format, packed once) into the per-layer LRU slots. The host only read the file and wrote
DRAM:

    python3 tools/offload/moe_card.py MODEL --check hf.json --cfg dev.pkl --pool pool.bin --card

The reference for each run is the ISA simulator's run with the bitstream's configuration (`--cfg`:
PAIR, DSTEP and STREAM on). The configuration changes the arithmetic: the 8B's 160-token run
with `isasim.board_config()` leaves these tokens at token 111. The card gave the simulator's
tokens, the same prefill logits (sha256) and the same misses per token:

| run | against the simulator | against HF | tok/s (wall / device) | hit rate | misses per decode token | MB streamed per decode token | host serving per token |
|---|---|---|---|---|---|---|---|
| 8B, 16 tokens, 28 slots | bit for bit | token-exact | 8.61 / 9.38 | 96.4% | 1.47 | 8.4 | 18.8 ms |
| 8B, 160 tokens ("Describe the water cycle ...") | bit for bit | the first 4 (fp4, section 9) | 10.03 / 10.10 | 98.5% | 0.89 (0.65 in the second half) | 5.2 | 10.1 ms |
| 35B-A3B, 16 tokens, 32 slots | bit for bit | token-exact | 2.02 / 2.04 | 62.4% | 95.9 (113.9) | 155 | 370 ms |

- The device rate is the token count over the run's cycles, which include the WAITW stalls on
  misses. Wall time is within 1% of device time: the host's polling loop costs nothing visible.
- The 8B decodes at 99 ms per token, of which the host's serving is about 10 ms. Section 4's
  model gives 12.7 tok/s on these routes; models3's dense runs measured about 13% under the bank
  model too.
- The 35B decodes at 490 ms per token, of which the serving is 370 ms: the run is streaming-bound.
  - The PCIe writes take 285 ms. 155 MB goes at 544 MB/s, against the 1.37 GB/s selftest
    bandwidth: each expert is one synchronous 1.67 MB write, then its 8-byte directory entry.
  - The pool reads take 89 ms (the 17 GB file, from the page cache and the disk).
  - Compute is the remaining 120 ms, so about 8 tok/s is the ceiling with the streaming
    hidden. At the selftest's write rate the writes alone would take about 115 ms per token.
- On the simulator, the last request of an 8B run (all hits) is posted but never served: the
  simulator's host hook runs only while a WAITW blocks. On the card the host serves it, so the
  card counts one more request (4 more hits).
- The checkpoints on the card's host are the routed experts' complement (1.4 GB for the 8B, 4.6
  GB for the 35B). The image build reads no expert; the pool file holds them.

The first run found a bug. The Engine's compile worker process, which the card's backend
compiles ahead in, built its image without the MoE's `experts`: every expert resident, over
DRAM (4568 MiB for 4096). It failed at the first prefill step, before the card ran anything.
The worker now builds the image with the Engine's own keywords (`Engine._image_kw`), and
tests/test_lfm2_moe.py checks it. The ISA simulator compiles in-process and never ran into it.

### 10.1 The expert server's DMA at the link's rate

`Board.write`, which the server used through `BackendDram`, sends an expert to the card in
three steps:
- The bytes are copied out of the pool, and CHASH's swaps and the two channel runs copy them
  three more times. A numpy buffer sits 16 bytes past a page, so each channel run then bounces
  through the staging buffer for the DMA's alignment.
- Two DMA calls follow, synchronously.
- Each 8-byte directory entry and the `served` word is widened to 128 bytes, with a read of the
  card first.

On omarchy's i5-12600KF, with the DMA dropped (`tools/offload/slot_bench.py --null`), that is
1.3 ms of host CPU per 1.67 MB expert. On the card's slower host, plus the DMA, the 35B paid
about 3 ms per miss. `BoardDram` (opentpu/host/offload.py, used on any transport that DMAs from
a worker thread, `dram_of`) does three things instead:
- It writes an expert's two channel runs in one pass, `np.take` of 64-byte beats with CHASH's
  swaps, from the pool's bytes into page-aligned staging buffers, then makes one DMA call per
  channel. No bounce.
- It keeps the host's own words in a shadow (`served` and the directory: the card only reads
  them) and writes them as whole 128-byte blocks, without a read.
- One worker thread makes every DMA call in order while the server stages the next expert, so
  an expert's data lands before its entry and every entry before `served`. `ExpertServer.poll`
  flushes before it returns.

tests/test_offload_server.py checks that the card's two channel memories end up byte for byte as
`Board.write` leaves them, with and without CHASH. tests/test_lfm2_moe.py runs the fake card that
computes beside the host with `BoardDram`.

The card ran it on 2026-09-30 (build B, the same references as section 10, all bit for bit):

| | before (`Board.write`) | `BoardDram` |
|---|---|---|
| `slot_bench`, 1.67 MB experts from RAM, 3 misses a request | 3.46 ms an expert, 483 MB/s | 1.74 ms, 958 MB/s (the DMA thread 1.40 GB/s) |
| `slot_bench`, 5.85 MB experts from RAM | 11.1 ms, 528 MB/s | 5.28 ms, 1109 MB/s (1.48 GB/s) |
| 35B-A3B, 16 tokens | 2.02 / 2.04 tok/s | 2.75 / 2.80 tok/s |
| 8B, 16 tokens | 8.61 / 9.38 tok/s | 9.34 / 10.04 tok/s |

The 35B's token took 357 ms. The host spent 242 ms of it answering requests, of which the DMA
thread wrote for 111 ms at 1.405 GB/s. The rest was staging, on the critical path because it was
slower than the DMA it fed: 1.69 ms an expert on the card's host (an i7-4790) against 1.19 ms
of DMA. Two things made it slow:
- The pool's pages: `np.take` read the pool file's memmap. A page not yet in the process's map
  costs a fault, and a page not in the page cache a read of the disk (about 115 MB/s on scattered
  reads), with the GIL held: the DMA thread waits too. `posix_fadvise` only asks for the read:
  nothing waits for it, or keeps the pages.
- The gather itself: about 1 ms for 1.67 MB on the i7-4790, from a cold source.

So now:
- The pool file has a split format (`opentpu.host.offload.split_order`). Each 4 KiB of an
  expert is stored as its two channel runs under CHASH, as they are when the block lands where
  the card's chunk index has even parity above its low 5 bits. Where that parity is odd, the
  two runs trade channels. `BoardDram` reads an expert with one `os.preadv` whose buffers are
  the 2 KiB pieces of its staging runs, in file order and routed by each block's parity. There
  is no gather, and the GIL is released for the copy. A slot that is not page-aligned, or a
  card without CHASH, takes the slot's bytes as before. `MO.serve` writes new pool files in
  this format (`<pool>.format`), and tools/offload/pool_split.py converts a file packed before.
- A thread reads every packed expert of the pool file once. This is the host's RAM tier in the
  page cache: for the 35B, 4333 experts, 7.2 GB. The Engine opens the pool (`moe.open_pool`)
  before it builds the image, so the read runs during the build (9 minutes for the 35B). The
  pages stay the kernel's to reclaim: nothing is pinned or locked, so other jobs on the host
  cannot run out of memory because of it. `moe_card` logs the packed experts' bytes in the page
  cache (mincore) when the pool opens, at load and at decode, and how much the thread had read.
- When no DMA is in flight (a request's first miss), `BoardDram` reads the expert in parts
  (`pieces`, 2 by default) and queues each part's DMA as soon as it is read, so the link starts
  after the first part. Each part costs one more DMA call per channel, about 50 us each (session
  2's two expert sizes: 1.67 MB at 1.395 GB/s and 5.85 MB at 1.484 GB/s fit 1.52 GB/s plus 50 us
  a call). A later miss's read overlaps the DMA ahead of it, so it goes whole.
- `BoardDram` keeps each slot's beat indices for the gather of the slot format, and writes the
  host's own words without `Board.write`'s general path.

On the card's host, with the DMA dropped (`slot_bench --null`, 1.67 MB experts, 3 misses a
request; another job's card session ran beside it):

| experts from | host ms per expert | of which staging |
|---|---|---|
| RAM, slot format, session 2's code | 1.38 | 1.00 |
| RAM, slot format | 1.16 | 1.01 |
| the pool file, slot format, in the page cache | 1.46 | 0.97, plus the read |
| the pool file, split format, in the page cache | 0.67 | 0.57 |

The 35B's projection. The DMA, at 1.19 ms an expert (session 2's 1.405 GB/s), is now the bound:
111 ms per token for its 93 misses. Each request's first staging (about 0.6 ms, about 40
requests a token) adds about 25 ms, and the entries and `served` about 5 ms. That is about 141
ms of serving per token instead of 242, so a token of about 256 ms: 3.9 tok/s, against 2.75
measured with session 2's code. The first staging in two parts exposes about 0.3 ms of it plus
two DMA calls (0.1 ms): about 7 ms a token less, so about 4.0 tok/s. The 8B, at 0.89 misses
per token, changes little.

tests/test_offload_server.py checks the split format: the order is a permutation (a short last
block included), and the records read back give the slot's bytes. `BoardDram` reading a
split-format file leaves the card's channel memories as `Board.write` does, at slots of either
page parity and off a page, with a request's first miss in 1 to 3 parts; a staging pair goes back
only after its last part (one pair, a slow link). `preadv` resumes a short read. The conversion
tool is covered too, and `PoolFile.resident` (on Linux: a file dropped from the cache, then
warmed).
tests/test_lfm2_moe.py runs the fake card with CHASH's map and a split-format pool file,
bit for bit against the ISA simulator.

### 10.2 Session 3: the split pool on the card

The card ran 4ae8dac on 2026-10-01, 02:37-02:50, on build B (79c5707a, reloaded after another
session's qualification run; the selftest passed before and after), against the same references:

| run | session 2 (`BoardDram`, slot-format pool) | session 3 (split pool, RAM tier, parts) |
|---|---|---|
| 35B-A3B, 16 tokens | 2.75 / 2.80 tok/s | **3.79 / 3.87** tok/s (+38%) |
| 8B, 160 tokens | 10.03 / 10.10 (session 1) | **10.64 / 10.71** (+6%) |

(wall / device). Both runs give the simulator's tokens and prefill logits bit for bit. The 35B
gives HF's 16 tokens; the 8B differs from HF at token 4, as before (fp4, section 9).

The 35B's decode token is 258 ms, and the host answers requests for 153 ms of it (242 in
session 2):
- 92 ms waiting for the DMA queue;
- 53 ms in the main thread: staging (0.53 ms an expert, overlapped but for each request's first
  part) and the directory;
- the rest, the mailbox reads and `served`.

The DMA thread wrote 107 ms of it, at 1.455 GB/s: the host is now bound by the link. Section
11.3's event model, with this host, predicted 3.94 tok/s.

The RAM tier: after the card host's reboot, 0.41 GB of the 35B's 7.24 GB of packed experts was
in the page cache when the pool opened. The warm thread had all of it in by the end of the image
build (552 s), and all of it was still resident at decode. The 8B's 4.12 GB went from 0 to all.

`slot_bench` on the card (3 misses a request, 1.67 MB experts):

| experts from | ms per expert | the DMA thread |
|---|---|---|
| RAM, slot format (the gather) | 1.72 | 1.44 GB/s |
| the split pool as found after the reboot (the disk) | 3.84 | 1.33 GB/s |
| the split pool warm, the first miss whole | 1.66 | 1.42 GB/s |
| the split pool warm, in 2 parts (the default) | 1.54 | 1.39 GB/s |
| the split pool warm, in 4 parts | 1.58 | 1.35 GB/s |
| the slot-format pool warm (preadv, then the gather) | 1.84 | 1.41 GB/s |

What is left:
- **Small DMA calls (host).** Each directory entry and `served` changes one 64-byte beat, but
  goes to the card as a whole 128-byte block: two DMA calls of about 50 us. One call each
  would save about 7 ms a token (*estimate*).
- **The misses themselves.** 93 a token at 1.19 ms each is 111 ms of link time. The next
  levers are a better replacement policy and more slots.

### 10.3 Session 4: one-beat writes and the decayed-use policy

The card ran 929cf9e on 2026-10-01, 04:12-04:31, on build B (79c5707a, as loaded; the selftest
passed before and after), against session 3's references. Both changes are host-only:
- each directory entry and `served` goes to the card as its one 64-byte beat, one DMA call on
  its channel, not as its 128-byte block in two calls;
- the slots' replacement by decayed use (section 5.5), run beside LRU.

| Qwen3.5-35B-A3B, 16 tokens | session 3 | session 4, LRU | session 4, decayed use |
|---|--:|--:|--:|
| tok/s (wall / device) | 3.79 / 3.87 | **3.90 / 3.98** | **3.95 / 4.04** |
| decode cycles | 551.5 M | 535.6 M | 528.7 M |
| misses a decode token (second half) | 95.9 (113.9) | 95.9 (113.9) | 93.5 (106.6) |
| the host's flush / DMA (s) | 1.47 / 1.71 | 1.28 / 1.76 | 1.34 / 1.75 |

All three give the simulator's tokens and prefill logits bit for bit, and HF's 16 tokens.

- The one-beat writes save 7.4 ms a token (the estimate was 7).
- Decayed use misses 2.4% less over the 16 decode tokens and 6.4% less over their second half;
  the traces' 2048 tokens give 9.6%. It misses more in the prompt (4508 against 4457 in all),
  before its counts build up.

`slot_bench` (1.67 MB experts, 3 misses a request): from RAM 1.61 ms an expert (1.72 in session
3), the split pool warm 1.47 ms (1.54).

The event model (section 11.3's, `--stream-policies lru,lfu`) for more slots on the 35B, tok/s
at 1.4 GB/s:

| Qwen3.5-35B-A3B | LRU | decayed use |
|:--|--:|--:|
| 1280 slots, int8 head (the card's) | 3.94 | 4.15 |
| 1440 slots | 4.08 | 4.30 |
| 1600 slots | 4.22 | 4.45 |
| fp4 head, 1280 slots | 4.24 | 4.48 |
| fp4 head, 1440 slots (the head's freed bytes) | 4.40 | 4.66 |

The fp4 head pays twice, as on Gemma 4: 254 MB fewer a token and some 150 more slots, but it
costs the 35B 2.4% perplexity. Hugging Face's final hidden states (bf16) over 900 tokens of
docs/isa.md's prose, through the head in float and in openTPU's formats (its input int8 per
block, as QACT):

| Qwen3.5-35B-A3B LM head | perplexity | KL(float, head) | top-1 as float's |
|:--|--:|--:|--:|
| float | 22.62 | 0 | 1 |
| int8 | 22.63 | 0.0003 | 0.990 |
| fp4 | 23.16 (+2.4%) | 0.018 | 0.889 |

The model has memorized the opening of Pride and Prejudice (perplexity 1.05 in every format),
so that text says nothing here. The head stays int8 by default; fp4 is an opt-in
(`head_format`), as on Gemma 4 E2B. With decayed use, Gemma 4 26B-A4B's rows in section 11.3
become 3.78 (fp4 experts, int8 head), 4.52 (fp4 head), 1.68 (int8 experts, int8 head) and
1.92 (int8 experts, fp4 head) tok/s.

## 11. Gemma 4 26B-A4B: design note

This is the next MoE target: Gemma 4's MoE, with its experts offloaded to host storage. The
note covers the model, how its layer maps onto `gemma4.py` and `moe.py`, the fit, and the
expected rate. The code waits for review, and is split with the gemma4 agent (below).

### 11.1 The model

From `config.json` and transformers' `modeling_gemma4.py`:
- 30 layers, hidden 2816, vocabulary 262,144, tied embedding, logits soft-capped at 30. No
  per-layer embeddings (`hidden_size_per_layer_input` 0) and no KV-shared layers.
- Attention:
  - 25 sliding layers: window 1024, 16 query heads and 8 KV heads of 256.
  - 5 full layers (every sixth): 16 query heads and 2 KV heads of 512, K = V. A full layer has
    no `v_proj`. V is `v_norm` (unit RMSNorm) of the K projection's raw output; K is
    `k_norm` then RoPE. The cache still holds both.
- Every layer has a MoE block beside a dense MLP. With r the residual after attention:

      dense = post_ffn_norm_1(mlp(pre_ffn_norm(r)))                  2112 wide, GELU-tanh
      w, ids = router(r)
      moe   = post_ffn_norm_2(sum_i w_i expert_ids[i](pre_ffn_norm_2(r)))
      x     = (r + post_ffn_norm(dense + moe)) * layer_scalar

  - The router: unit RMSNorm of r, times `router.scale` and H^-0.5, then `router.proj`
    [128, H], a softmax over all 128, the top 8 renormalized, and each weight times
    `router.per_expert_scale[id]`.
  - Experts: 128 per layer, top 8, width 704, GELU-tanh. The checkpoint stores them fused per
    layer: `experts.gate_up_proj` [128, 1408, 2816] (gate rows first) and `experts.down_proj`
    [128, 2816, 704].
- The checkpoint is 51.6 GB in bf16. omarchy has it whole. opentpu's copy in
  ~/openTPU/models is an interrupted download: the first shard (49.9 GB) is missing, with a
  17.9 GB `.incomplete` file in `.cache` dated 2026-09-30 00:07.

### 11.2 How the layer maps

On `moe.py`'s side (offload) no new card mechanism is needed:
- **The expert slot.** `ExpertFormat` with F padded from 704 to 768. fp4 blocks run 128
  along K, and gemma4's `ffn % 2D` check applies. The padding is zero rows of gate and up and
  zero columns of down, which is exact: gelu(0) * 0 = 0. The expert is 3.45 MB instead of 3.16.
- **Folds at packing:**
  - `per_expert_scale[e]` into expert e's W_down (it scales the expert's output linearly);
  - `router.scale` * H^-0.5 into `router.proj`'s columns.
  The router reads the unit RMSNorm of r, quantized, as `moe_ffn` does for every model.
- **The experts' own input.** `pre_ffn_norm_2`'s gain is not folded: the experts read the
  unit norm times the gain, quantized (`moe_ffn`'s `g_exp`; one `QACT` with column scale,
  8 more instructions a program). With the gain folded into the gate and up columns, the
  experts would read the unit norm quantized, whose blocks are scaled by the residual's
  outlier channels, which the gain all but zeroes (the 26B's layer 10: |x| 17 where the gain
  is 0, the gain up to 92 elsewhere), and the gain's own outliers would set the weights'
  column blocks. On the real model that gave perplexity 127 against float's 1.235 (int8 dense
  layers, fp4 experts; gemma4's emulation), the MoE block's relative error 0.27-0.51 from
  layer 10 on with int8 experts (the dense MLP, which quantizes norm times gain: 0.01-0.04).
  The router keeps the fold: a flipped route costs little. In tests/test_gemma4_moe.py's tiny
  model with 4 channels of the gain at 8x, the int8 device against HF goes from a median
  cosine of 0.977 (folded) to 0.994.
- **The rule.** The softmax rule as written (`MoESpec.rule` "softmax"): the softmax of the 8
  largest logits is the renormalized top 8 of the full softmax. The order is the same, ties to
  the first.
- **`MoESpec.act`.** GELU-tanh in the expert (`swiglu_down(act=gelu_tanh)`, as gemma4's dense
  MLP).
- **`moe_ffn` split into its parts.** It now returns `x + acc`. Gemma needs the experts' sum
  alone (`post_ffn_norm_2` is an RMSNorm of the sum), and wants its dense MLP to run after
  the request is posted, while the host streams. That is section 4's `d_post`, worth +14% in
  the model. So `moe_ffn` takes `beside` (code to emit after the post, before the expert
  loops) and can return the weighted sum without the residual. LFM2's and Qwen3.5's programs
  stay word for word the same.

On `gemma4.py`'s side (gemma4):
- `Spec.from_hf` accepts `enable_moe_block` and `attention_k_eq_v`.
- Global layers get their own KV head count (2 against 8) and K = V. ACT RAM holds 16 x 512
  = 8192 (board: 128 blocks of 128).
- The dense MLP is padded from 2112 to 2304 (`ffn % 2D`), or the check is relaxed to D (2176).
- The layer calls `moe_ffn` with the dense MLP as `beside` and combines the two norms. The
  image places the router weights and norms, and `Layout`'s slots after the rest.
- The generate loop polls the expert server too (qwen3.Engine already wires every server).

### 11.3 The fit and the rate

On the card: the non-expert weights are 1.64 GB (survey, fp4 with the int8 head and
embedding: 761 MB of it) plus 26 MB of dense padding. KV is about 0.2 GB: the 25 sliding
rings of 1024 + a block, 8 x 256, and the 5 full layers at 4096 positions, 2 x 512. With
0.3 GB kept for KV, I/O and programs, the rest holds the slots.

`cachesim.py` replays the four 2048-token traces through the event model, with:
- the per-layer LRU, warmed from the other texts' profile;
- the expert at 3.45 MB (`--expert-bits 4.636`) and the dense MLP's 9.5 MB after the router;
- the link at session 2's 1.4 GB/s, 50 us a DMA call, and a 450 us host lead (the first
  staging in parts plus the poll).

Calibration: the same model with session 2's effective host (staging-bound: 0.99 GB/s, 1.8
ms lead) gives the 35B 2.84 tok/s, against 2.80 measured on the card (device). With the host
as in 10.1 it gives 3.94.

| gemma-4-26B-A4B | slots (per layer) | misses / token (of 240) | MB / token streamed | tok/s | all resident (bound) |
|:--|--:|--:|--:|--:|--:|
| int8 head | 682 (22.7) | 62.5 | 216 | 3.51 | 5.71 |
| fp4 head (`--head-bits 4.25`) | 789 (26.3) | 53.2 | 184 | 4.19 | 6.72 |
| int8 experts (6.69 MB: `--expert-bits 9.0`), int8 head | 351 (11.7) | 107.5 | 719 | 1.55 | |
| int8 experts, fp4 head | 406 (13.5) | 97.0 | 649 | 1.75 | |

int8 experts give half the slots and twice the bytes a miss: under half the rate. The choice
waits for the perplexity of fp4 experts (gemma4's 900-token runs, both heads); a split by layer
range is the middle way (per-layer slot sizes in `Layout`).

- The fp4 head pays twice: 369 MB fewer bytes a token, and 107 more slots. On E2B it is an
  opt-in (cosine 0.974 -> 0.971, +14% decode), and its accuracy on the 26B is to be measured.
- Prefetch still loses here (3.0-3.7 against 3.5-4.2 without).

### 11.4 Host side and plan

- **Host files.** opentpu gets a stripped checkpoint (the non-expert weights, about 7.4 GB in
  bf16, 4.8 GB of it the text model's) and a split-format pool (3840 experts x 3.45 MB = 13.2 GB fp4), both made on omarchy
  from the whole checkpoint. The RAM tier then reads 13.2 GB into a 31 GB host's page cache.
- **Plan.**
  1. Emulation on omarchy (float64 + fake quantization against HF bf16, a few prompts): the
     folds and padding, fp4 against int8 head, greedy tokens.
  2. `moe.py`'s parts (offload) and gemma4's attention and from_hf (gemma4), meeting at a
     tiny random Gemma4-MoE model in tests. The layer composition is checked against HF and
     the ISA simulator bit for bit, the card's side with the live fake card and a split pool.
  3. The full model's ISA-simulator reference on omarchy, then a card session.
