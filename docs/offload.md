# Bigger than DRAM: MoE expert offloading

Status: phase 1, a design study. Nothing here has run on the card. Every number says whether
it was measured (and where), simulated from real router traces, or is an *estimate*.

The goal (the user's): run models that do not fit in the card's 4 GiB, by streaming from the
host and its SSD, with mixture-of-experts (MoE) models first. An MoE decode token reads only
its top-k experts per layer, a few percent of the weights, so the card can keep the dense part
and a cache of experts, and fetch the rest.

**Summary.**

- **Models (section 2).** Gemma 4 E4B is dense: its 2.8 B-parameter per-layer-embedding table
  is what exceeds 4 GiB, and the host can supply the token's row (22-43 KB per token). The
  Gemma 4 MoE is gemma-4-26B-A4B (128 experts, top-8). Of the 21 checkpoints surveyed, the
  best fits for the card: LFM2.5-8B-A1B (just over 4 GiB), gemma-4-26B-A4B and
  Qwen3.5-35B-A3B (3-4x the card, inside host RAM; Qwen3.5's layers already run here), and
  Qwen3-Next-80B-A3B (larger than host RAM: the SSD tier).
- **The hierarchy (section 1, measured on opentpu):** card DRAM 14 GB/s, about equal to a host
  fp4 kernel's 11-16 GB/s, well above PCIe Gen1's 1.3 GB/s and the SSD's 0.51 GB/s.
- **Router traces (section 3; 4 texts x 2048 tokens, CPU):** at the card's cache size a
  per-layer LRU hits 98.6% of LFM2.5-8B-A1B's picks (85% of its experts fit) and 76% of
  Gemma 4 26B-A4B's (19% fit). Fixed profiles transfer badly across texts (Gemma 4: 22-41%).
  A layer's router applied before its mixer names ~80% of its experts.
- **Tokens per second (section 4, simulated):** LFM2.5-8B-A1B 12.6 streaming its misses over
  Gen1 (13.7 all resident). Gemma 4 26B-A4B 3.2 streaming, 3.7 with prefetch, 4.1-4.8 with
  Gen2, and **6.2 with the hybrid**: the host computes the card's misses from its own RAM, with
  the card's arithmetic, while the card streams its hits. PCIe then carries activations, not
  experts.
- **Design (section 5):** Tier A needs no new hardware. The host routes at a halt after each
  MoE router (~60 us per layer, 2-4% of a token), binds the experts' cache slots into the next
  program segment, and streams or computes the misses. Tier B, shared with the card's own
  sampling, adds `TOPK`, `LDR` (a register from memory) and `WAITW` (wait on a DRAM word), so
  the card runs a whole token and the host only serves misses.
- **PCIe Gen2 (section 6): not worth a Vivado slot now.** It adds 28% to streaming and nothing
  to the hybrid.
- **SSD tier (section 8):** models larger than host RAM run at the SSD's pace. Extrapolating
  from Gemma 4's traces: ~6 tok/s for Qwen3-Next-80B-A3B (fine-grained experts, 98% of its
  picks in host RAM), 1.4-1.8 for gpt-oss-120b and Qwen3.5-122B-A10B.
- **Next (section 9):** LFM2.5-8B-A1B end to end on the ISA simulator with the card's 4 GiB,
  then the card. The Qwen3.5-35B-A3B trace is still running (its 72 GB download is slow).

The tools: `tools/offload/survey.py` (the model survey, from the checkpoints' safetensors
headers), `router_trace.py` (the expert choices of a Hugging Face model on a text, on CPU),
`cachesim.py` (expert-cache hit rates on those traces and the tokens-per-second model),
`hostkern.c` (the host's fp4 kernel with the card's MM arithmetic).

## 1. The memory hierarchy

| Level | Size | Bandwidth | Source |
|:--|--:|--:|:--|
| Card DRAM (2 x DDR3-1066) | 4 GiB | 13.9-14.5 GB/s while decoding | card counters, README |
| PCIe Gen1 x8 (XDMA), host -> card | | 1.26-1.34 GB/s (opentpu); 1.42 (omarchy); 1.7 best placed 8 MiB calls (omarchy) | `otpu-selftest`, docs/host.md |
| card -> host | | 0.84-1.00 GB/s (opentpu) | `otpu-selftest` |
| one small DMA call (64 B - 4 KiB) | | 11-15 us (omarchy, interrupt mode) | docs/host.md |
| Host RAM (opentpu, DDR3, i7-4790) | 31 GB (23-28 free) | 22.6 GB/s (4-thread fp32 GEMV), 13.6 (1-thread sum) | measured 2026-09-29 |
| Host fp4 kernel, card arithmetic | | 10.7-15.8 GB/s (4 threads, other load on the host) | `tools/offload/hostkern.c`, section 5.4 |
| SSD (Crucial BX500 240 GB, SATA, LUKS + btrfs zstd) | 123-140 GB free | 512 MB/s (O_DIRECT, 4 MiB, 1 reader), 515-524 (2-4 readers), 227 MiB/s (128 KiB x 4) | measured 2026-09-29 |

Measured on opentpu (the card's host) unless marked. The SSD figures read a 4 GiB file of random
bytes written for the test (`dd iflag=direct`); reads of the models' own files through btrfs
compression are not O_DIRECT and come from the page cache. opentpu's other disk is an HDD that
is not ours. The card sits in the CPU's x16 slot (`max_link_speed` 8 GT/s x16 on the root port,
00:01.0), so a Gen2 card would train at 5 GT/s.

The ratios decide the design: card DRAM : PCIe : SSD is 14 : 1.3 : 0.5, and the host's RAM,
read by a good 4-bit kernel, is as fast as the card's DRAM. An expert that misses the card's
cache costs ~10x its compute time to bring over PCIe, but only ~1x to compute on the host,
where it already is.

## 2. Models

`python3 tools/offload/survey.py --json survey.json` (safetensors headers of each checkpoint;
vision, audio and MTP weights left out). Bytes are for our formats: fp4 blocks (4.25 bits per
weight, docs/quant.md), the LM head in int8. "Bytes / token" is what a decode token streams
(dense + shared + top-k experts + head; the embedding row is looked up). "On-card part" is what
must stay on the card (dense, shared, head, and the embedding in int8 when it is not tied).
"Fits card" allows 0.3 GB for KV / state, I/O and programs; "fits host RAM" means the fp4 model
in 26 GB.

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
- **Gemma 4 E4B is dense** (`enable_moe_block` false); its 2.82 B per-layer-embedding (PLE)
  parameters (1.5 GB fp4) are what push it over 4 GiB (2.83 GB without them, 4.33 GB with).
  The Gemma 4 MoE is **gemma-4-26B-A4B**: 30 layers, 128 experts of 704, top-8, and a dense
  2112-wide MLP beside the MoE in every layer (the two outputs are added); the router reads the
  post-attention residual through its own RMSNorm, softmax, top-8 renormalized, times a learned
  per-expert scale.
- The ones that already fit (OLMoE, Granite 3.1 3B, Granite 4.0 H Tiny, SmallThinker-4B) need
  no offloading, only the MoE block.
- Our layer types cover LFM2-MoE (convolution + attention: `opentpu/llm/lfm2.py`), Qwen3-MoE
  (`qwen3.py`), and Qwen3.5-MoE / Qwen3-Next (Gated DeltaNet + gated attention: `qwen35.py`).
  Gemma 4 comes with the `gemma4` port (E2B). gpt-oss (attention sinks, clamped SwiGLU with
  biases, MXFP4), DeepSeek-V2 (MLA) and Granite 4 (Mamba2) need new layers.
- gpt-oss ships its experts in MXFP4 (E2M1, a power-of-two scale per 32). Our two-level format
  holds a 128-block of it exactly when its four sub-block exponents span at most 3 (a bf16 base
  times multipliers 1, 2, 4, 8); a conversion can check that per block (not measured).

## 3. Router traces and the expert cache

`tools/offload/router_trace.py` runs the Hugging Face model (bf16, CPU, on omarchy; the
weights memory-mapped from the checkpoint under a cgroup memory limit, or offloaded to disk)
over four texts of 2048 tokens each, teacher-forced: the first chapters of *Pride and Prejudice*
(prose), the Wikipedia article "Roman Empire", `opentpu/isasim.py` (code) and `docs/host.md`
(technical markdown). Routing is causal, so a token's experts in this prefill are the ones a
decode producing that text would pick. For every MoE layer and token it records the top-k and
three predictions of it (the layer's router applied, through its norm, to the layer's input,
before the mixer: `pre`; to the previous layer's router input: `prev_r`; to the previous layer's
input: `prev_in`). A check re-runs each router on the recorded residual and gets the recorded
experts for 98.9-99.1% (LFM2.5) and 92.8-98.4% (Gemma 4) of the tokens: the rest are ties
between the k-th and (k+1)-th expert, which bf16 logits make common and a second top-k (of 2k)
breaks the other way; the recorded choice is the model's own.

`tools/offload/cachesim.py` replays each trace in decode order (token by token, layer by layer,
a layer's k experts at once) against a cache of C expert slots shared by all layers, for five
policies: `static` (the C experts most picked in the other three texts, never replaced), `lru`,
`lru_layer` (the slots split evenly over the layers), `lfu` (counts, with the profile as a
prior) and `opt` (Belady, the upper bound). Every policy starts from the static set: the host
fills the card from a profile at load. Hit rates are the mean over the four texts, each scored
with the profile of the other three.

Hit rates (the mean over the four texts) and a token's misses under the per-layer LRU (of
L x k requests):

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

Prediction accuracy, the share of a layer's top-k the prediction's top-k names:

| Model | pre | prev_r | prev_in |
|:--|--:|--:|--:|
| LFM2.5-8B-A1B | 0.82 | 0.67 | 0.72 |
| gemma-4-26B-A4B | 0.79 | 0.72 | 0.65 |

- **Adapt, do not profile.** A fixed set of the experts most picked in the other texts catches
  0.92-0.96 of LFM2.5's picks at the card's size (85% of the pool), but only 0.22-0.41 of
  Gemma 4's (19%), whose own-text profile would catch 0.80: which experts are popular depends on
  the text. LRU adapts within a few tokens (0.74-0.78 on every text); LFU seeded with the
  profile adapts too slowly (0.40-0.51).
- **Global LRU thrashes below one token's sweep** (a token asks every layer in turn, a loop of
  L x k experts): at 5% of Gemma 4's pool (192 slots < 240 per token) it hits nothing. Split per
  layer it degrades gracefully; at the card's sizes the two are within a point. The runtime
  uses per-layer LRU.
- **Room above LRU.** Belady's bound misses half as often as LRU on Gemma 4 (27.6 against 57.3
  per token): a policy that knew reuse better (the predictions below, or reuse distances) has
  up to 2x fewer misses to find.
- **Predictions** name 79-82% of a layer's experts one mixer ahead (`pre`) and 67-72% one layer
  ahead (`prev_r`); of the per-layer LRU's misses at the card's size, the prediction's 2k best
  hold 90-92% (`pre`) and 73-81% (`prev_r`).

## 4. The model: tokens per second

`cachesim.py --survey survey.json` turns each replay into a time per token, from the per-layer
misses of every token (not their mean). With x an expert's bytes, d a MoE layer's dense bytes
(everything but its experts: attention or mixer, norms, router, shared expert, Gemma's dense
MLP), h the LM head's bytes, k the experts per layer, m(l) a token's misses in layer l, and:

| Constant | Value | Source |
|:--|--:|:--|
| B_dram, card DRAM while decoding | 14.1 GB/s | measured (README) |
| B_pcie, host -> card | 1.3 GB/s (Gen1), 2.6 GB/s (Gen2) | measured / *estimate* x2 |
| t_call, per DMA call | 30 us | *estimate* (11-15 us measured per small call) |
| t_sync, one halt, host step and restart (Tier A) | 60 us | *estimate* (5.2) |
| B_host, host fp4 kernel | 12 GB/s | measured 10.7-15.8 (hostkern.c) |
| B_ssd | 0.51 GB/s | measured |

the strategies are, per token:

- `resident`, the bound if everything fit (and the card routed): sum over layers of
  (d + k x) / B_dram, plus h / B_dram;
- `stream` (Tier A): the same plus a halt per MoE layer and every miss over PCIe in series,
  sum_l (t_sync + m(l) (x / B_pcie + t_call));
- `prefetch` (Tier A, 5.3): a predicted miss moves during the window before it is needed (one
  layer's dense time for `pre`, plus the previous layer's experts for `prev_r`); what does not
  fit in the window, the unpredicted misses and the wrong predictions (at the miss rate) cost
  link time;
- `hybrid` (5.4): per layer, d / B_dram + t_sync + max(hits x / B_dram, t_sync + m(l) x / B_host)
  (the card's hits against the host's misses); `hybrid/static` the same with a fixed cache, no
  inserts;
- `host_only`: the whole active model on the host, (L (d + k x) + h) / B_host, the baseline a
  4-bit CPU runtime would reach on opentpu (attention and sampling ignored).

The host's per-token work outside the device run (sampling, ~0.5-1.3 ms on opentpu today) is
left out of all of them.

| Model | card cache | resident (bound) | stream Gen1 | best prefetch Gen1 | stream Gen2 | best prefetch Gen2 | hybrid | hybrid/static | host only |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| LFM2.5-8B-A1B | 595 (85%) | 13.73 | 12.58 | 12.85 | 13.01 | 13.32 | 13.58 (link 0.07) | 13.86 | 11.69 |
| gemma-4-26B-A4B | 744 (19%) | 5.88 | 3.20 | 3.70 | 4.11 | 4.84 | 6.16 (link 0.86) | 6.01 | 5.00 |

*Simulated* from the traces with the constants above (the card's cache at its size from the
survey: 4 GiB less the on-card part and 0.3 GB; per-layer LRU; `link`: the share of Gen1 the
hybrid's cache inserts need beside it). Read with care: the time per halt and per DMA call are
estimates, and the host kernel's rate was measured alone.

- **Nearly fits (LFM2.5-8B-A1B).** 85% of the experts stay on the card and a token misses 1.2
  of its 88: streaming the misses over Gen1 runs at 92% of the all-resident bound. Offloading
  is close to free.
- **Three times the card (gemma-4-26B-A4B, 13.8 GB at fp4).** A token misses 57 of its 240 experts (180 MB):
  over Gen1 that is 140 ms of transfer against 170 ms of card time, 3.2 tok/s. Prefetch from
  the previous layer's prediction gives 3.7, Gen2 4.1, both 4.8. **The hybrid gives 6.2**,
  above the all-resident bound (5.9): the host computes the misses (15 ms of its time) while
  the card streams its hits, so the two DRAMs work in parallel, and PCIe carries activations
  (KB per layer) instead of experts (MB). With a fixed cache (no inserts at all) it gives 6.0;
  with the host kernel at a quarter of its measured rate (3 GB/s, a loaded host) 5.1, or 3.4
  with the fixed cache, whose 71% misses then load the host.
- **The host alone** would run these models at 5.0 and 11.7 tok/s at its kernel's rate: the
  card is worth it for what it adds beside the host, not instead of it.
- **What limits Gemma 4 now** is its int8 LM head: 761 MB of the 2.4 GB a token reads
  (262,144 x 2816). An fp4 head, or half the vocabulary computed on the host in parallel, is
  worth ~20% (*estimate*).

## 5. Design

### 5.1 Where things live

- **Card DRAM** (4 GiB): the resident part (dense layers, shared experts, norms, the LM head,
  KV cache / convolution / DeltaNet state, I/O, programs), then the **expert cache**: fixed-size
  slots, one expert per slot, every part of the expert contiguous in its slot (gate, up, down
  rows and their scale words at fixed offsets), so one base address names an expert. All
  experts of a model have one size, so the slots need no allocator.
- **Host RAM**: the **expert pool**, every expert already in the card's format (fp4 rows and
  scale words, the slot layout), in 4 KiB-aligned buffers placed for fast DMA
  (`board.DMA_PLACE`, docs/host.md), plus the cache directory (expert -> slot, the policy's
  state). A miss is one DMA write, no conversion. Every model in the table but Qwen3-Next-80B,
  Qwen3.5-122B and gpt-oss-120b fits in 26 GB at fp4.
- **SSD**: the pool as one file of slot-sized records, read with O_DIRECT in 4 MiB requests
  (the rate measured above) into host-RAM slots, for models larger than host RAM (section 8).

### 5.2 Control: three tiers

**Tier A: the host routes (no new hardware).** The token's program is split at each MoE
router into segments; the card halts after a router, the host routes and runs the next
segment. Per MoE layer l:

1. the card runs segment l: the experts of layer l-1 and their combine, the mixer of layer l
   (attention, convolution or DeltaNet), the router of layer l (an MM to DRAM), HALT;
2. the host sees HALTED, reads the router's logits (E words, one small DMA read), applies the
   model's rule (softmax / sigmoid + bias, top-k, renormalize, per-expert scale) in numpy,
   looks the k experts up in the directory, and fetches the misses (DMA writes into victim
   slots) or, in the hybrid, computes them (5.4);
3. the host binds the k slot addresses into segment l + 1 and starts it.

The binding uses the compiler's run-time values (`compiler.RunVar`): an expert's weights are
a descriptor at `slot_base + slot * SLOT_BYTES` with `slot` a RunVar. There are only 8 ARG
registers (a decode already uses 6 for the position), so instead of ARGs the host **bakes** the
values into the immediates of the assembled segment (the instruction's base register becomes
R0 and the value is added to its immediate; an `ADDI r, R_arg, 0` becomes `ADDI r, R0, v`), a
generic pass with no limit on the number of values, then writes it (a few KB) and loads it into
IMEM (4096 instructions on the board). Everything that is not an expert keeps its compile-time
addresses; segments depend on the position only through the existing ARGs.

The cost is one halt per MoE layer: HALTED seen (back-to-back register reads, ~1 us each), the
logits read (~12 us), the routing in numpy, the segment write and LOAD (~15-25 us), RUN:
*estimate* 40-100 us, 60 us in the model below (to be measured on the card in phase 2). For
the models here that is 1.3-2.9 ms per token (22-48 MoE layers), 2-4% of their token time.
The routing is on the host in every backend (ISA simulator, RTL, card), so all three agree bit
for bit, and the logits it reads are the card's own.

**Tier B: the card routes (new, general instructions).** Three additions, each useful beyond
MoE, which the `autodecode` work (the card sampling its own tokens) needs as well:

- `TOPK`: the indices and values of the k largest of a row (MoE routing over E <= 512; greedy
  and top-k sampling over the vocabulary);
- `LDR rd, [ra + imm]`: a register loaded from DRAM or TMEM. The token id becomes an
  embedding-row address; an expert id becomes its slot address through a directory table in
  DRAM that the host keeps; a loaded count in `LOOP R[ra] + w2` skips an expert (predication);
- `WAITW [addr], v`: stall until a DRAM word equals v (with a timeout that raises an error), so
  the card can wait for a host-served miss or a host input without halting.

The card then runs a whole token. A layer's experts are `LDR` of their directory entries; a
missing expert's entry points at a request: the card writes (layer, expert, sequence) to a
mailbox and waits on the slot's flag, while a host thread polls the mailbox (a 64-byte read,
as the streamed logits do: docs/host.md), writes the expert and sets the flag. Hits never
involve the host. This removes the 1.3-2.9 ms of Tier A halts and keeps the resident decode's
single program.

**Tier C: the card fetches (later, if ever).** The XDMA's descriptor bypass lets card logic
start host-to-card transfers from host bus addresses: with the pool in pinned host memory and
its bus addresses in a card table, a miss needs no host CPU at all. It needs a kernel driver
that pins ~20 GB and exports the addresses (root, the user's install), the bypass wired in the
block design and a request engine in RTL. It saves the miss latency (tens of us), not
bandwidth, which the tables above show is what limits; not worth it before Tiers A and B.

### 5.3 Prefetch and overlap

DMA into the card's DRAM runs while the card computes (the XDMA shares LiteDRAM's ports with
the core through `otpu_mem_ch`; the streamed logits already read during runs). A fetch of the
next layer's experts can therefore overlap the current layer if the choice is known early.
The traces measure three predictions: the layer's router applied to the layer's input (before
the mixer) and to the previous layer's states. They name 65-82% of the experts (section 3),
but the window they open is about one layer's compute: ~0.8 ms for LFM2.5-8B-A1B (11 MB of
dense weights at 14.1 GB/s) against 4.5 ms for one of its experts over Gen1, ~2.1 ms for
Gemma 4 (29 MB) against 2.4 ms. A prefetch hides at most about one transfer per layer, and a
wrong one costs link time. In the model the best prefetch (the previous layer's prediction,
its k best) gains 2% on LFM2.5 and 16% on Gemma 4; taking the prediction's 2k best loses. It
is worth having when experts stream, but the hybrid (5.4) does far more; prefetch is a
refinement for later.

### 5.4 Hybrid: the host computes the misses

The host holds every expert in RAM and computes fp4 as fast as the card
(`tools/offload/hostkern.c`: 10.7-15.8 GB/s on four cores of the i7-4790 with Gemma 4's and
LFM2.5's expert shapes, 150-250 us per matrix; eight threads collapse to 0.55 GB/s on this
4-core host with other load). An expert
that misses costs its bytes at ~14 GB/s on the host instead of 1.3 GB/s over PCIe, and the host
works while the card computes the hits: in Tier A the card runs the layer's hits while the host
computes the misses from the MoE input the card stored beside the logits; the host writes each
missed expert's output and the card combines all k in the fixed order, so the sum is the same
as an all-card run. A layer with misses takes a second halt (the combine waits for the host);
Tier B's `WAITW` removes it.

Bit-exactness: hostkern.c computes each 128-block's exact integer and then the card's fp32
sequence, `isum_4` over blocks of `(i2f(isum) * ws) * ascale`, with every multiply and add
rounded on its own (`-ffp-contract=off`) and denormals flushed (SSE FTZ + DAZ, the ISA's
flush-to-zero), and matches its scalar transcription of docs/isa.md bit for bit on random data
(all rows, three shapes). Phase 2 checks it against the ISA simulator's MM and adds the
activation (SiLU / GELU composites) and QACT between the projections, also in the ISA's exact
sequences.

The hybrid still inserts misses into the card's cache in the background when the link has
room (`link` in the tables: the share of PCIe a caching policy's inserts need); with a full
link it degrades to a fixed cache (`hybrid/static`).

### 5.5 Prefill

A prefill chunk touches nearly every expert of every layer (the union of k picks over hundreds
of tokens), so it streams the layer's missing experts once per chunk, not per token. The card
computes prefill at its MXU rate (Qwen3-0.6B 4-bit: 103.4 tok/s measured, 2 x parameters x
tok/s = ~124 G operations per second), which for gemma-4-26B-A4B's 3.8 B active parameters is
~16 tok/s (*estimate*): a 512-token chunk takes ~1 s per layer against ~0.3 s to bring a whole layer's
128 experts (405 MB) over Gen1. Fetching layer l + 1 while layer l computes hides it for chunks
above ~150 tokens. Prefill stays compute-bound, as it is today; the offload does not change it.
(The host's four AVX2 cores have more int8 multiply rate than the MXU, but a host prefill
would have to reproduce every kernel of the model bit for bit, not only the experts.)

### 5.6 Correctness and tests

Every model stays token-exact against the Hugging Face reference (greedy) in the ISA
simulator, then the RTL, then the card. Offloading changes where an expert's bytes come from,
never which bytes, so the expert cache is invisible in the outputs: the ISA simulator runs
with a DRAM the size of the card's (`DRAM_BYTES` = 4 GiB) and the host runtime of the card, and
a test compares its logits with an all-resident run (a small DRAM limit on a small MoE model
forces misses). Routing on the host from the card's logits keeps the three backends identical.
A new risk is routing near-ties: the card's router logits differ from bf16's, so the k-th and
(k+1)-th experts can swap where HF's would not; the check against HF will show whether greedy
tokens move (as the argmax can today).

## 6. PCIe Gen2

| Model | stream, Gen1 -> Gen2 | best prefetch, Gen1 -> Gen2 | hybrid | hybrid's inserts, share of the link |
|:--|--:|--:|--:|--:|
| LFM2.5-8B-A1B | 12.58 -> 13.01 (+3%) | 12.85 -> 13.32 (+4%) | 13.58 (no change) | 0.07 -> 0.04 |
| gemma-4-26B-A4B | 3.20 -> 4.11 (+28%) | 3.70 -> 4.84 (+31%) | 6.16 (no change) | 0.86 -> 0.43 |

(tok/s, *simulated* as in section 4, Gen2 taken as twice the measured Gen1 rate.)

Gen2 x8 doubles the link (5 GT/s; XDMA's AXI side 128 bits at 250 MHz instead of 125). On
decode it pays only where experts cross PCIe: +28-31% for Gemma 4 without the host's help,
nothing for the hybrid, which moves activations. It would also halve the hybrid's cache-insert
load (86% of Gen1 at Gemma 4's card size, so its LRU inserts fit either way), a prefill
chunk's per-layer expert stream (405 MB: 0.31 -> 0.16 s) and the image load (4 GiB: ~3.3 ->
~1.7 s).

The cost is a build that may not close: everything in xdma_aclk (otpu_axi_split2, the XDMA side
of otpu_mem_ch, the LiteDRAM CSR crossing, the AXI-Lite SmartConnect) goes from 125 to 250 MHz,
and the earlier Gen2 attempt missed by ~0.1 ns in the IP's own placed paths (80 MHz build,
docs/board.md section 5).

**Verdict: not worth a Vivado slot now.** The hybrid gets +93% on Gemma 4 over Gen1 streaming;
Gen2 gets +28%, and nothing on top of the hybrid. Revisit if the card shows the inserts or
prefill link-bound. When it comes: a timing-only FAST=1 CORE_MHZ=100 build with
`pl_link_cap_max_link_speed {5.0_GT/s}` and `axisten_freq {250}` in `bd_native.tcl`.

## 7. Gemma 4 E4B: the per-layer embeddings from the host

E4B is dense and misses 4 GiB only by its PLE table (262,144 x 42 x 256 = 2.82 B parameters).
A token uses one row of it: 42 x 256 values, 21.5 KB in bf16 (43 KB fp32). The host keeps the
table (5.6 GB bf16 in RAM, or fp4 / int8 at 1.5 / 2.8 GB) and writes the token's row with its
run arguments, one DMA write of ~22-43 KB, ~30-40 us at Gen1: under 0.2% of E4B's token
(2.83 GB streamed at 14.1 GB/s: 5.0 tok/s *estimate*, all resident, fp4 with an int8 head).
With the host in the loop (today's decode: the host samples) this costs nothing new; for the
card's own sampling (autodecode) it is a host input per token, the case `WAITW` covers. The
rest of E4B (2.83 GB) fits with ~1.1 GB to spare for its KV cache.

## 8. The SSD tier

Three models in the survey are larger than the host's RAM at fp4: Qwen3-Next-80B-A3B (41 GB of
experts), gpt-oss-120b (61 GB) and Qwen3.5-122B-A10B (62 GB). The host RAM then becomes the
second cache level (a global LRU over the pool, about 24 GB of it; the card's cache inside it)
and the SSD holds the pool as one file of slot-sized records. A host miss is on the token's
path: an SSD read at 0.51 GB/s, 3.3 ms for a 1.67 MB expert, 26 ms for gpt-oss's 13.2 MB.

Gemma 4's traces, with the host RAM limited to part of its pool, stand in for them (`cachesim.py
--host-frac`; the card's cache and the hybrid as in section 4, the SSD reads added on the
token's path):

| Host RAM holds | host hit rate | SSD reads / token | SSD ms / token | tok/s |
|--:|--:|--:|--:|--:|
| 30% of the pool | 0.874 | 30.4 | 188 | 2.87 |
| 40% | 0.937 | 15.1 | 94 | 3.93 |
| 50% | 0.971 | 7.1 | 44 | 4.87 |
| 60% | 0.987 | 3.1 | 19 | 5.51 |

Applied to the three models (*extrapolated*: Gemma 4's hit-rate curves at their fractions, the
per-layer mean misses; a card cache under 5% of the pool counted as no hits; Qwen3.5-122B's
on-card part leaves no room, so its LM head moves to the host):

| Model | pool (fp4) | host RAM holds | host hit | SSD per token | card cache | card + host | tok/s |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Qwen3-Next-80B-A3B-Instruct | 41 GB | 58% | 0.98 | 10 x 1.7 MB = 32 ms | 1452 (5.9%) | 129 ms | 6.2 |
| gpt-oss-120b | 61 GB | 39% | 0.92 | 12 x 13.2 MB = 302 ms | 172 (3.7%) | 242 ms | 1.8 |
| Qwen3.5-122B-A10B | 62 GB | 39% | 0.92 | 32 x 5.0 MB = 313 ms | 150 (1.2%) | 406 ms | 1.4 |

The SSD tier works best for fine-grained experts with few active bytes: Qwen3-Next-80B (512
experts of 1.67 MB, 3.6 B active, its layers those of Qwen3.5) stays within host RAM for 98%
of its picks and runs at ~6 tok/s; the two others are SSD-bound at 1.4-1.8 tok/s. A faster
disk moves the last two directly (an NVMe drive at 2.5 GB/s would give them ~2.1-3.3 tok/s,
*estimate*); prediction cannot hide an SSD read (3-26 ms) behind a layer (3-9 ms).

Details for the pool file: fp4 does not compress, so it should be written with btrfs
compression off (`chattr +m`) to keep O_DIRECT reads direct (compressed extents fall back to
the page cache); requests of one expert (1.7-13 MB) are in the 4 MiB regime measured above.

## 9. Phase 2

Phase 2 builds Tier A on the host side first, then the card through team-lead:

1. **The MoE block** in `ol` kernels: the router (an MM to logits in DRAM), the expert FFN over
   slot descriptors (SwiGLU; GeGLU for Gemma), the combine (routing weights as fp32 words the
   host writes, the k outputs summed in a fixed order), the shared expert (Qwen3.5) and the
   parallel dense MLP (Gemma). The expert slot layout and the host-side pool in the card's
   format (`quant.quantize_mxu` per expert).
2. **Segments and baking**: the decode program split at MoE routers (the hidden state stored at
   each halt and reloaded), and the compiler pass that bakes run-time values into immediates.
3. **The runtime** (`opentpu/host/offload.py`): routing on the host, the directory and the
   per-layer LRU, DMA of misses, the hybrid path through `hostkern` (C, loaded with ctypes,
   bit-exact against the ISA simulator's MM, activation and QACT), the SSD tier.
4. **The ISA simulator with the card's DRAM** (`DRAM_BYTES` = 4 GiB) driven by the runtime
   between segments; tests: logits bit-identical to an all-resident run on a model that fits
   (forced misses from a small cache), and token-exact against HF.
5. **First model: LFM2.5-8B-A1B** (LFM2's layers plus the MoE block; the card's cache holds 85%
   of its experts). Then Gemma 4 26B-A4B on the `gemma4` port's layers, and Qwen3.5-35B-A3B on
   Qwen3.5's.
6. **The card** (through team-lead): the per-layer halt cost, the misses' DMA while the core
   runs, tok/s against the model above.

Tier B's instructions (`TOPK`, `LDR`, `WAITW`) are shared with the card's own sampling
(`autodecode`) and are designed with it.
