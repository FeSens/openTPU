# MiniCPM-V 4.7 (35B-A3B) on openTPU

[openbmb/MiniCPM-V-4.7-35B-A3B](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B) (Apache-2.0,
BF16 safetensors, transformers >= 5.18 `minicpmv4_7`, `AutoModelForImageTextToText`) is a
vision-language model:
- a SigLIP2 vision tower with LLaVA-UHD v4 token compression inside the ViT;
- a projector into the language model;
- a Qwen3.5-MoE language model (`qwen3_5_moe_text`), the architecture openTPU already runs as
  `qwen35-35b`.

This document covers:
- the model as Hugging Face's code defines it: transformers v5.18.0's `minicpmv4_7`, which
  inherits `minicpmv4_6`; the checkpoint has no remote code;
- how it maps onto openTPU: one bitstream, no new hardware op, the vision tower's math on the card;
- the validation and card plan.

Revision checked: f8ddcae6f94778efa90dca75d7f2714916a8132d (2026-10-13). The model is `minicpmv-35b`
in `opentpu.llm.MODELS`.

## 1. The checkpoint

16 safetensors shards, 70.43 GB (1,152 tensors):

| part | tensors | parameters | bf16 |
|---|---|---|---|
| LLM experts (`mlp.experts.gate_up_proj`, `down_proj`, 40 layers x 256) | 80 | 32.21 B | 64.4 GB |
| LLM embedding + LM head (untied, 248,144 x 2048 each) | 2 | 1.02 B | 2.03 GB |
| LLM layers without experts (attention, DeltaNet, shared expert, router, norms) | ~1,000 | 1.43 B | 2.86 GB |
| vision tower + projector (`model.vision_tower.*`, `model.merger.*`) | 61 | 0.553 B | 1.11 GB |

The language model's tensors are `model.language_model.*` and `lm_head.weight`. Under
`load_weights`' rename to `model.*`, these are the names Qwen3.5-35B-A3B's checkpoint gives.

There are no `mtp.*` tensors, although the config says `mtp_num_hidden_layers: 1`. So MTP
speculative decode (docs/mtp.md) is not available for this model. OpenBMB's DSpark draft model
is a separate checkpoint, left for later.

## 2. The language model against qwen35-35b

The text configs differ in one field that matters: `vocab_size` is 248,144 here against
Qwen3.5-35B-A3B's 248,320. Everything else is the same:
- 40 layers, `(linear x 3, full) x 10`, hidden 2048;
- attention: 16 query heads and 2 KV heads of 256, an output gate, RoPE on 64 of the 256
  dimensions at theta 1e7;
- DeltaNet: 16 key heads and 32 value heads of 128, a conv of 4 taps;
- 256 experts top-8 of width 512, normalized softmax top-k, plus a sigmoid-gated shared expert
  of width 512;
- untied embeddings, RMSNorm eps 1e-6.

The other config differences are `dtype`, `eos_token_id`, `mlp_only_layers: []`,
`output_router_logits` and `tie_word_embeddings`. None of them changes the math: Spec.from_hf
defaults eos to (248046, 248044), which matches `generation_config.json`'s `[248044, 248046]`.
HF's MoE block (`Qwen3_5MoeSparseMoeBlock`) and router (softmax, top-8, renormalized) are the
ones openTPU's `moe.route` (`rule="softmax"`, `norm=True`) follows.

**Vocabulary.** 248,144 = 1,938 x 128 + 80, so the LM head's last 128-row block is partial.
The head streams vocabulary rows (`_lm_head`: chunks of `HEAD_CHUNK` rows, the last one short).
A row count that is not a multiple of D is not a K dimension: MM's K is the hidden size, 2048.
The tiny Qwen3.5 tests already run a 1,000-token vocabulary (1,000 = 7 x 128 + 104). The text
path's test checks it at the partial block: the device logits of the real vocabulary must be
bit-identical to those of the same head padded with zero rows to a multiple of D (the last real
id included). If it ever fails, the fallback is that padded head plus a -inf mask on the padding
ids, so they can never be sampled.

**M-RoPE.** `rope_parameters` has `mrope_interleaved: true`, `mrope_section [11, 11, 10]`
(section 4). The rotary dimension is 64, i.e. 32 frequencies. Frequency i takes the position of
channel T when i mod 3 = 0, of H when i mod 3 = 1, and of W when i mod 3 = 2. In HF's
`recomposition_frequencies`, H takes `slice(1, 33, 3)` and W takes `slice(2, 30, 3)`. A text
token has T = H = W = p, so text-only prompts are ordinary 1-D RoPE at p, the qwen35 tables.

**Chat template** (`chat_template.jinja`): Qwen3.5's `<|im_start|>` / `<|im_end|>` turns.
- The generation prompt opens `<think>\n`, or `<think>\n\n</think>\n\n` with
  `enable_thinking=false`. A past reply's thinking is dropped unless it follows the last user
  query.
- An image part of a message renders as `<|image_pad|>` (248056), joined to the text parts with
  `\n`. A video part renders as `<|video_pad|>` (248057). A system message may not hold images.
- Tools follow Qwen3.5's XML format.

## 3. The vision side

### 3.1 The processor (host)

`MiniCPMV4_6ImageProcessor` with `preprocessor_config.json`: max_slice_nums 9, mean = std = 0.5;
defaults scale_resolution 448, patch 14, slice_mode on, downsample "16x", use_image_id on.
Images are converted to RGB uint8 [3, H, W].

- **Grid.** `get_sliced_grid`: ratio = H W / 448^2 and m = min(ceil(ratio), 9). With m <= 1
  there are no slices. Otherwise the grid (rows, cols), rows x cols in {m - 1, m, m + 1} minus
  {1} minus > 9, minimizes |log(W / H) - log(cols / rows)|; on a tie the one with more rows wins.
- **Thumbnail ("source").** `find_best_resize(H, W)`. When H W > 448^2, or when there is no
  grid (`allow_upscale`):
  - h = int(448 / sqrt(W / H)), w = int(h W / H);
  - then each side goes to `ensure_divide(x, 56) = max(round(x / 56) * 56, 56)`: a multiple of
    4 patches, because the two 2x2 merges need it.

  So the thumbnail is always about 448^2 pixels: about 1,024 patches, 64 LLM tokens.
- **Slices.** `get_refine_size`:
  - the image is first resized to the slice size times the grid;
  - the slice size is `find_best_resize` of (ensure_divide(H, rows) / rows,
    ensure_divide(W, cols) / cols), with upscaling allowed;
  - then the image is cut into rows x cols slices, row-major.

  Each slice is also about 448^2 pixels.
- **Resample.** torchvision `resize`, bicubic, antialias on, on the uint8 tensor (uint8 out).
  Then x / 255, then (x - 0.5) / 0.5.
- **Patches** (`reshape_by_patch`). Each crop's 14 x 14 blocks are taken row-major, each block
  flattened (channel, row, column): 588 values. The crops are concatenated, thumbnail first.
  `target_sizes` holds each crop's (h, w) in patches. The patch embedding (a 14 x 14, stride-14
  Conv2d) is then a matrix product of these rows with `patch_embedding.weight` reshaped to
  [1152, 588].
- **Placeholder text** (`replace_image_token`, `_prepend_local_ids`). Each `<|image_pad|>` of
  the prompt becomes `<image_id>N</image_id>` (N the image's index in its sample), then
  `<image>`, then thumbnail_h thumbnail_w / 16 copies of `<|image_pad|>`, then `</image>`.
  With slices it is followed by the slice rows joined by `\n`, each row cols x (`<slice>` +
  slice_h slice_w / 16 pads + `</slice>`).
- **Ids:** `<image>` 248078, `</image>` 248079, `<slice>` 248088, `</slice>` 248089,
  `<image_id>` 248090, `</image_id>` 248091, `\n` 198. `mm_token_type_ids` is 1 exactly on
  248056.

A 448 x 448 image is one crop of 32 x 32 patches, 64 image tokens and 69 prompt tokens with its
markers. A large image with 9 slices is about 640 image tokens.

### 3.2 The vision tower (`MiniCPMV4_7VisionModel`, per crop: attention never crosses crops)

SigLIP2: hidden 1152 (16 heads of **72**), MLP 4304, 27 layers, gelu_pytorch_tanh, LayerNorm
with bias, eps 1e-6.

1. `x = P Wp^T + bp + E_pos[pid]`. P is the crop's patch rows [N, 588]. `E_pos` is a 70 x 70
   table (image_size 980 / 14). The NaViT position id is `pid = 70 bucket(i / h) +
   bucket(j / w)`, where bucket(f) counts the boundaries 1/70, 2/70, ... at or below f, in
   float32 (`get_vision_nearest_position_ids`).
2. Layers 0 .. 6 on the N tokens. Each layer is
   `x += out(attn(LN1(x)))`, then `x += fc2(gelu_tanh(fc1(LN2(x))))`. Every linear has a bias.
   The attention is bidirectional over the crop's tokens, scale 72^-0.5.
3. **The window merger** after layer 6 (`insert_layer_id`, `MiniCPMV4_7ViTWindowAttentionMerger`):
   - y = LN1(x);
   - attention within each 2x2 window of the patch grid (4 tokens, same q/k/v/out shapes as a
     layer);
   - x = x + attn;
   - each window's 4 rows, in row-major order within the window, are concatenated:
     z [N/4, 4608];
   - out = linear_2(gelu_tanh(linear_1(LN(z)))) + mean of the window's 4 rows, with linear_1
     4608 -> 17216 and linear_2 17216 -> 1152.

   The output is the half-resolution grid, row-major.
4. Layers 7 .. 26 on the N / 4 tokens, then `post_layernorm`.
5. **The merger** (`MiniCPMV4_7Merger`, merger_times 1, `DownsampleMLP`):
   - 2x2 windows of the half grid are concatenated as above, into [N/16, 4608];
   - LN(4608, eps 1e-6);
   - linear 4608 -> 4608, then `nn.GELU()` (the **exact erf** GELU, not tanh);
   - linear 4608 -> 2048.

   That gives one LLM embedding per 4x4 patches: the rows that replace the crop's
   `<|image_pad|>` tokens, in order.

"4x" mode skips step 3 (and the merger merges 2x2 once), 4 times the tokens. openTPU runs
the default 16x.

Parameters:
- 27 layers of 15.24 M each;
- patch embedding and position table 6.3 M;
- window merger 104.5 M;
- merger 30.7 M.

That is 553 M in all, 0.55 GB in int8.

### 3.3 Canvas M-RoPE (`get_rope_index`, `mrope_mode: "canvas"`)

Positions are triples (t, h, w).
- Text tokens take p, p, p, continuing a 1-D counter.
- A *group* is a run of frames separated only by markers. A *frame* is one image: its
  thumbnail, its slices and the markers around them.
- A frame starts at its `<image>` token with o, the counter's value: the canvas origin.
- The canvas size, in LLM tokens:
  - without slices, the thumbnail's grid / 4;
  - with slices, rows x slice_h by cols x slice_w (slice grid / 4).
- Every token of the frame first gets (o, o, o). Its markers, including a trailing `\n` (id 198)
  that closes the frame, keep that unless named below.

| token | t | h | w |
|---|---|---|---|
| `<image>` | o | max(o - 1, 0) | max(o - 1, 0) |
| thumbnail (i, j) | o | o + round(linspace(0, canvas_h - 1, th)[i]) | o + round(linspace(0, canvas_w - 1, tw)[j]) |
| `</image>` | o | o + canvas_h | o + canvas_w |
| slice k = (r, c), token (i, j) | o | o + r sh + i | o + c sw + j |
| `<slice>` of k | o | o + r sh | o + c sw |
| `</slice>` of k | o | o + r sh + sh - 1 | o + c sw + sw - 1 |
| `\n` after slice row r | o | o + (r + 1) sh - 1 | o + cols sw |

After the frame, the counter is o + max(canvas_h, canvas_w) + 1.

Notes:
- Without slices the linspace is an arange.
- The rounding is torch's, float32 linspace, round half to even.
- The text after the image continues 1-D. Decode adds `rope_deltas = max position + 1 -
  sequence length`, which is at most 0: the k-th generated token is at slot n + k and position
  n + k + delta.
- The causal mask is by slot, unchanged.

The host port of this is a direct transcription of `get_rope_index` (integer index arithmetic,
like a tokenizer's). It is checked against HF's function on prompts with and without slices,
and with several images.

## 4. Mapping onto openTPU

### 4.1 Text path

- `load_spec`: `model_type` `minicpmv4_7` gives `minicpmv.Spec`. It is qwen35's Spec (its
  `from_hf` reads `text_config`) plus the vision side (4.3); `vision=None` is the language
  model alone.
- `load_weights` and `LazyWeights` also skip `model.vision_tower.` and `model.merger.` (they skip
  `model.visual.` today). The vision side has its own loader (`minicpmv.load_vision`).
- The formats are MiniCPM-V's own (`streamed.FORMATS["minicpmv"]`; a Spec's family is its
  module, else its first base in the table): fp4 experts, DeltaNet int8 in layers 0-19 and NVFP4
  after them, attention NVFP4, the shared expert int8, int8 head, int8 embedding on the host.
  Section 5.1 has the scan that chose them.
- The pool is MX, built like `pool-q35-mx` (tools/offload/pack_pool.py) on the simulation host:
  - `pool-mcpv35-mx.split.bin`, 17,532,190,720 bytes: 10,240 experts in 1.71 MB slots, with
    its .packed (all 10,240), .format and .key;
  - its key holds this checkpoint's identity, the hash of its config.json.
- The slim checkpoint for the card is every tensor but the routed experts
  (tools/offload/strip_experts.py keeps the vision tower and the merger): 1,072 tensors, 5.59 GiB
  of bf16, 6,021,430,313 bytes with the tokenizer and configs. Its config.json is the original's,
  so the pool's key matches.

With these, text-only prompts are qwen35-35b's programs over another checkpoint: same kernels,
same bitstream, nothing new on the card.

### 4.2 Vision tower on the card

Rule: the host only decodes, resizes, slices, normalizes and cuts patches (3.1), and computes
integer indices (4.5). Every multiply-add of the tower and projector is on the card.

**The card's window order.** The host cuts a crop's patch rows in this order
(`minicpmv.card_order`, a permutation, as a tokenizer orders its pieces): each 4 x 4 block of
the patch grid row-major, within it its four 2 x 2 windows row-major, within those their 4
patches row-major. The position-table rows go in the same order.
- Attention and the row-wise parts (LayerNorm, the linears, the MLPs) do not see the order.
- A 2 x 2 window is 4 consecutive rows. So its [4C] merge row is those rows' 4608 words as they
  lie in DRAM: no gather and no strided loads.
- The window merger's output rows come out in the projector's window order (4 consecutive
  rows a window, one level up).
- The projector's rows come out row-major over the 4 x 4 blocks: HF's order, the order of the
  crop's `<|image_pad|>` tokens.
- The programs depend on the patch count N alone, not on (h, w).

**Layout** (`minicpmv.VisionImage`).
- The weights keep the checkpoint's shapes:
  - q, k, v, out are [1152, 1152], unpadded;
  - the MLP widths are padded to their chunks with zero rows and columns: 4304 -> 4352,
    17216 -> 17408;
  - the patch rows are padded 588 -> 640.
- Each LayerNorm's bias is folded into the next linear's bias on the host, at build time like
  the quantization: W (y + b) = W y + W b, with W b computed in float64.
- Each encoder layer is a block of `lstride` bytes, so a run-time layer index addresses any of
  them; block 27 is the window merger's attention.
- **Formats** per kind (`VisionImage.formats`: attn, mlp, wm, proj, embed): int8 by default,
  543 MB.
- The work area of a crop of up to 2048 patches is 74.5 MB:
  - P, the patch rows; E, the position rows; X and X2, the residual rows; Q head-major; O;
  - the K/V scratch: K token-major, V^T rows `nmax` tokens wide, untiled;
  - `vis`: the projector's output rows, the prompt's image rows (4.3).

**Kernels** (`opentpu/kernels/vit.py`; generic for a pre-norm ViT with 2 x 2 merges). No new
op and no ISA change.
- **LayerNorm before a linear** (`layernorm_q`): the mean (RSUM / n) and the centering on the
  VPU, r = rsqrt(RSSQ / n + eps). Then QACT applies r as its row scale and the gain as its
  column scale (RSCALE / CSCALE), so no normalized tile is ever written to TMEM; the bias is
  the next linear's (above). The post LayerNorm, whose rows are stored, runs in place by column
  parts.
- **Q, K, V** (`vit_qkv`): each head's q, k and v land in the first 72 columns of zero-padded
  [R, 128] tiles. Each starts as its bias (a fill), and the MM over the head's 72 weight rows
  accumulates into it. The MXU streams no padding rows, and the K/V cache and Q get D-wide rows
  (the scores are the 72-wide heads'; scale 72^-0.5). Q is stored head-major, one store per
  head.
- **Bidirectional attention** (`vit_attend`): `attention._attend_heads` with an int length N,
  groups of MCOLS query rows, every head's output columns gathered in one [G, 1152] tile and
  stored once.
- **Window attention** (`vit_window_attend`): G = 16 rows (ACT_ROWS), i.e. 4 windows, per MM.
  - The scores [16, 16] are computed against the group's 16 keys, plus a block-diagonal mask
    (-inf outside the windows).
  - The softmax is one block (no running maximum).
  - Then P V^T. The MXU streams D-aligned rows, so P goes into a [16, 128] tile at the group's
    offset within its 128 tokens (zero elsewhere; the band is cleared after its QACT), and V^T
    streams from the aligned token.
  - The 8 groups of a 128-token stretch are unrolled, and the stretches are a hardware loop.
    The last stretch is run whole: its rows past N are scratch, each of their windows sees only
    itself, and nothing reads them back.
- **Out + MLP** (`vit_out_mlp`): O Wo^T accumulated into the residual rows, the MLP's
  F = 4352 in chunks of 256 (the activations of chunk c quantized while chunk c + 1 streams).
- **Merges** (`vit_merge`: the window merger's MLP and the projector):
  - The window's [4608] row is never held whole. Its LayerNorm takes three passes over pieces
    of 768 (sum, centered sum of squares, then normalize and QACT into a stationary built in
    parts).
  - The window mean is accumulated from the first pass.
  - The MLP runs in chunks (512 for 17408, 256 for 4608) into the output tile.
- **GELU.** The ViT MLPs and the window merger use gelu_tanh, as HF does. The projector's is
  erf in HF; the card uses tanh there too, since the measurement (5.2) finds no difference.

**Programs.** A crop is 13 programs; host-orchestrated runs, as the MoE prefill's are.
- embed; then qkv / attend / out at a run-time layer index for layers 0..6; then the window
  merger's qkv / window attention / out (no MLP) / merge; then qkv2 / attend2 / out2 (last:
  with the post LayerNorm) for layers 7..26 on N / 4 rows; then the projector, at a run-time
  output row.
- Row groups of R = 16 rows (ACT_ROWS; the merges 8) are hardware loops, so every program fits
  the card's IMEM at any N. At 1024, 1056 and 2048 patches the largest is the window attention,
  1944 instructions of 4096; at MCOLS 8 / ACT_ROWS 8, 3734.
- They compile in about 0.1 s each and are cached by patch count. The thumbnail's N follows the
  aspect ratio; the slices of an image share one N.

**Cost on the card** (MCOLS 4 x D 128, 512 MAC a cycle, MX replay so weights stream once per
16 rows, 133 MHz), per crop of 1,024 patches:

| part | MAC |
|---|---|
| patch embedding (640 -> 1152) | 0.8 G |
| layers 0-6: linears 15.3 M a token, attention 16 x N^2 x (128 + 72) | 133 G |
| window merger: attention, its 4608 -> 17408 -> 1152 MLP on N / 4 rows | 31 G |
| layers 7-26 at 256 tokens | 83 G |
| projector | 2 G |

That is about 250 G MAC, about 3.7 s per crop at full MXU rate. The card measured 5.77 s for a
crop of 1,008 patches (session mcpv2, 6). The unpadded projections save
20% against the first estimate's padded heads (312 G). A 448^2 image is 1 crop and 64 tokens;
a large image with 9 slices is 10 crops, about 37 s, and about 640 tokens. The m8 build
(MCOLS 8) halves it. On the simulation host's ISA simulator a crop takes 5-7 minutes (a simulator,
not the card).

`--max-slices` (HF's `max_slice_nums`) bounds the cost: 1 gives the thumbnail alone.

**Pieces and their table** (`opentpu/kernels/pieces.py`).
- Every matrix is cut into pieces of at most an expert slot's bytes (1,712,128 at the 35B's
  fp4 slots), each its element rows, then its scale rows:
  - q, k, v: whole heads;
  - out and the patch embedding: stretches of D rows, one MM a piece, each into its output
    columns;
  - the MLPs' W1: whole chunks (the window merger's 512-row chunks span two pieces, one MM a
    piece);
  - the W2s: whole column parts.
- The real tower is 446 pieces, 544.5 MiB. The programs read 438 of them: the window merger's
  attention block has the layers' layout but no MLP.
- The kernels read each piece's address from a table in DRAM: RLD into a register, the last 4
  kept, given back at the end of each row group's body. So the same programs run with the
  pieces resident or placed in expert slots.
- The table's runs equal the static addresses' bit for bit (tests/test_minicpmv.py, int8 and
  fp4, pieces small enough to split every kind).

**Residence** (`Spec.vision_staged`).
- **Resident**: the pieces after each layer's vectors. 621 MiB of the image, 543 of them
  weights.
- **Staged** (the default since session mcpv2, 6): the pieces are the expert pool's entries
  past the experts (entry L x E + i is piece i; L the pool's layers, 41 with an MTP layer).
  - For an image's runs the expert server places them in expert slots (`ExpertServer.place`):
    free slots first, then the experts of least decayed use across the layers, each layer
    keeping k. The victims' directory entries are cleared before the pieces are written.
  - The host writes the table, runs the crops, then releases the slots to their layers, free;
    decode's misses refill them.
  - The image keeps 76 MiB: the vectors and the work area.
  - The pool grows by 446 entries (763.6 MB), and its key's "vision" field names them.
    `moe.open_pool` grows an experts-only pool in place and keeps its experts; a text-only run
    opens a pool with them as its experts'.
  - The ISA test: a staged engine's rows equal a resident one's bit for bit, with every slot
    warm. Its prefill afterwards, refetching the evicted experts, and its decode steps give the
    same logits.

The resident vision side costs expert slots. `streamed.fit_experts` at a 4096-token cache,
1.637 MiB slots of 40 layers:

| vision side | MiB | slots a layer | slots |
|---|---|---|---|
| none (staged or streamed) | 0 | 41 | 1640 |
| int8 resident | 617 (543 weights) | 31 | 1240 |
| int8, the ViT MLPs fp4 | 498 | 33 | 1320 |
| fp4 everywhere | 379 | 35 | 1400 |

The measurements and the choice: section 5.3.

### 4.3 Image rows into the prompt

**The model's image.** `load_spec` gives `minicpmv.Spec`: qwen35's Spec with `vision` (the
VisionSpec), `vision_fmt`, and the work area's patches and image rows. Its `image()` is
`minicpmv.Image`, which is qwen35's image plus a `VisionImage`.
- The VisionImage is allocated after the model's DRAM and before the expert slots (qwen35's
  `before_slots` hook), so the slots fill what is left (`fit_experts`).
- `build` writes qwen35's image, then the vision weights from the checkpoint (`load_vision`).
  The position table the host gathers from is an entry of an attribute of the image's own
  (`vis_tables`), so an image cache hit puts it back.
- The text path's programs are qwen35's: a text-only prompt runs exactly qwen35-35b's.

**Rows.** The 35B prefills layer-major (`Engine.prefill_layers`: a chunk's residual rows in
`xbuf`, or expert-major's records).
- `prefill_layers(tokens, vis=...)` takes each token's vis row (-1 for text).
- Layer 0 splits a chunk into runs of text and runs of image rows (consecutive vis rows, at
  most R, within an attention block). An image run is two runs:
  1. `vis_copy_run`: `vis[src:src + n] -> X[row:row + n]`, both run-time; xbuf rows or the
     expert-major records. It is a DMA through TMEM, no arithmetic.
  2. The layer run with `embedded=True`.
- Text runs, layers 1 .. 39 and the head are unchanged.
- The test gives image rows whose values an int8 embedding row holds exactly. Their logits and
  the next three decode steps are bit for bit those of the same prompt with text ids whose
  embedding rows are those values, layer-major and expert-major.

**M-RoPE.** Every kernel reads its RoPE rows from the image's `cos_t` / `sin_t` at the token's
slot. For an image prompt the host rewrites those rows (`minicpmv.Vision.positions`):
- Slot s < T gets `rope_index`'s (t, h, w).
- Slot s >= T gets s + delta, the decode's position.
- Each row's frequency i is the table's entry at the position of section i mod 3
  (`mrope_rows`): a gather of entries the image already holds, no arithmetic. A text
  position's row is the table's row itself.
- Only the rows that differ from what the table holds are written. Text after text writes
  none, and a text-only conversation keeps the table as built, bit-exact.
- A chat's rollback or new turn recomputes the positions of the whole conversation and writes
  the difference. `/reset` writes the table's own rows back (`pristine`).
- The rows written equal HF's `Qwen3_5MoeTextRotaryEmbedding` (interleaved, sections
  [11, 11, 10]) at the canvas positions within 1e-4 (the table's float32 against HF's).

The first design had two device runs build those rows (an RLD of the positions and masked
selects). The host's gather writes the same values without a program: indexing and DMA, no
arithmetic, the same class as a tokenizer's position ids (approved 2026-10-13). It
is 256 bytes per slot over PCIe (cos and sin, 32 frequencies each), once per image prompt: up to
1 MB at a 4096-token cache.

### 4.4 Host

- `opentpu/llm/minicpmv.py` holds:
  - the processor port: torch/torchvision resize as HF does it, normalization, patches, the
    NaViT position ids, the placeholder text and the canvas positions;
  - the numpy references for the vision tower and projector (fp32, checked against HF);
  - the card's tower (`VisionImage`), the model's image (`Spec`, `Image`) and the host's side of
    images on an engine (`Vision`: encode, positions, prompt).
- **otpu-chat**:
  - `--image PATH` (repeatable) attaches images to the first message; in `--plain`,
    `/image PATH` attaches one to the next message (`Chat.attach`).
  - `--max-slices N` is the latency knob. The default is the processor's 9, so card outputs
    stay comparable with HF's.
  - A turn's message carries its images before its text, as HF's processor does. The template
    renders `<|image_pad|>` for each image, `expand` puts each image's placeholder text in its
    place, and the tokenizer gives the ids: bit for bit HF's processor's.
  - `Vision.prompt`: the RoPE rows for the whole conversation, the crops of the images with
    tokens in the new part through the card's tower into the vis rows, then the prompt is
    prefilled with its vis rows.
- **otpu-serve** (docs/serve.md, Images):
  - `image_url` (or `input_image`) content parts, in any order with the text parts;
  - `data:` URLs by default; http(s) only with `--image-fetch` (20 s, 32 MB); never a file of
    the server's;
  - a request's `max_slices` (1 .. 36), `--max-slices` the server's default;
  - a context's key holds each image token's content key (`minicpmv.keyed`: a negative int32,
    a hash of the image's sha256 over its crops' sizes and patch rows and the token's index),
    so the device's context, the prefix cache and the KV store reuse a context past an image
    only for the same image at the same slices;
  - the worker feeds and picks with the token ids, writes the running request's M-RoPE rows at
    each switch, and encodes the images with tokens in the part it feeds. The model decodes one
    sequence at a time, so the table is the running request's.

### 4.5 The host / device split

| | host | card |
|---|---|---|
| image | decode, resize (bicubic, antialiased), slices, (x - 127.5) / 127.5, 14 x 14 patches | - |
| integer indices | NaViT position ids, the card's window order, canvas M-RoPE positions | - |
| tables | gathers rows the image holds: the position table's rows at the ids, the RoPE table's entries at the positions | - |
| text | the chat template, the placeholder text, the tokenizer, the image tokens' content keys | - |
| math | none | the patch embedding, 27 layers, the window merger, the projector, the language model |

Every gather is indexing plus DMA of values the image already holds, as a tokenizer's position
ids are: no host arithmetic on model values.

## 5. Validation

### 5.1 Text

References:
- openTPU's numpy `qwen35.reference_logits` (fp32, LazyWeights per tensor);
- HF's own model on a host without the RAM for its experts (`tools/minicpmv_hf.py`,
  `LazyExperts`: HF's eager expert loop reading each routed expert from the checkpoint; 0 logit
  difference against `from_pretrained` on a tiny model).

Results (the simulation host, ISA simulator):
- Greedy: the card's 16 tokens equal HF's through the end of the turn.
- The vocabulary's partial head block gives the logits of the head padded with zero rows, bit
  for bit (tests/test_minicpmv.py).
- Formats (tools/formats_scan.py ppl, 2000 tokens, dKL against int8 with its paired SE). Every
  row but float and int8 is the 35B's mix (fp4 experts, attn / delta / mlp nvfp4, head int8)
  with the named kinds or layer ranges int8. Slots: expert slots a layer on the card (4 GiB, a
  4096-token cache, `streamed.fit_experts`), with the vision side staged / resident int8.

| formats | ppl | dKL | resident MiB | slots | + ms a token: bytes + slots | net |
|---|---|---|---|---|---|---|
| float | 15.218 | -0.99% (SE 0.04) | - | - | - | - |
| int8 | 15.276 | 0 | 1846 | - | - | - |
| the 35B's mix | 16.102 | +5.68% (SE 0.22) | 1255 | 39 / 31 | 0 | 0 |
| attn int8 | 16.087 | +5.14% (SE 0.23) | 1370 | 38 / 29 | 12.3 + 0.7 (+6.9%) | -1.5 |
| mlp (the shared expert) int8 | 16.037 | +5.41% (SE 0.22) | 1308 | 39 / 30 | 5.7 + 0 (+3.0%) | -0.3 |
| delta int8 (all 30 DeltaNet layers) | 15.851 | +3.26% (SE 0.13) | 1680 | 33 / 25 | 45.5 + 4.4 (+26.4%) | -2.2 |
| delta int8 in layers 0-9 | 16.215 | +4.79% (SE 0.17) | 1369 | 38 / 29 | 12.2 + 0.7 (+6.8%) | +2.1 |
| delta int8 in layers 10-19 | 16.037 | +4.99% (SE 0.19) | 1354 | 38 / 30 | 10.6 + 0.7 (+6.0%) | +0.9 |
| delta int8 in layers 20-29 | 16.023 | +5.23% (SE 0.23) | 1369 | 38 / 29 | 12.2 + 0.7 (+6.8%) | -2.3 |
| delta int8 in layers 30-39 | 15.957 | +5.26% (SE 0.21) | 1354 | 38 / 30 | 10.6 + 0.7 (+6.0%) | -1.8 |
| delta int8 in layers 0-19 | 16.069 | +4.19% (SE 0.16) | 1468 | 36 / 28 | 22.8 + 2.1 (+13.2%) | +1.7 |
| delta int8 in layers 0-29 | 16.027 | +3.67% (SE 0.13) | 1581 | 34 / 26 | 34.9 + 3.6 (+20.3%) | -0.2 |
| **delta int8 in layers 0-19, mlp int8** | 15.976 | +3.68% (SE 0.14) | 1520 | 35 / 27 | 28.4 + 2.8 (+16.5%) | **+3.5** |

  The decode time a token is against the 35B's mix, with the vision side staged:
  - bytes: the resident weights every token reads, at decode's ~9.8 GB/s;
  - slots: the slot count's misses, interpolated from 5.3's cachesim points (~0.7 ms a slot a
    layer).

  Percentages are of the 35B's ~189 ms a token on the card. "net" is the formats rule's balance
  (1 point of dKL is worth 10% of speed): the dKL points gained against the 35B's mix x 10,
  less the decode time's percent.

  qwen35-35b ships at +4.00% (SE 0.16) on its own text (docs/mx_mxu.md). The first rule (the
  lead's, 2026-10-13: the most slots with a dKL point estimate at most +4.00%) picked DeltaNet
  int8 in layers 0-19 with the shared expert int8. Decode time confirms it (2026-10-13):
  it is also the best net (+3.5) and meets the quality bar.
  - It is MiniCPM-V's default on the card (`streamed.FORMATS["minicpmv"]`: fp4 experts, head
    int8, formats `delta@0-19=int8,attn=nvfp4,delta=nvfp4,mlp=int8`).
  - The experts stay fp4, so the pool is unchanged.
  - It reads 265 MiB more int8 weights every token than the 35B's mix and has 4 fewer slots a
    layer (a layer's 4 x 1.637 MiB slots in 40 layers).
  - From now on, format variants are ranked by dKL against the whole decode time (the bytes a
    token reads plus the slot count's misses from cachesim), not by slots alone.
  - The DeltaNet layers carry most of the loss: all of them int8 takes 2.4 points off.
  - The early ranges are worth more than the late ones (0-9 -0.89, 30-39 -0.42 points).
  - On the card the layer ranges are a layer block layout per formats group
    (tests/test_qwen35_moe.py test_formats_by_layer_range_stream_bit_exact: the layer-major
    prefill, expert-major and not, equals token by token's bit for bit with this mix's ranges).

### 5.2 Vision

- The processor port: pixel_values, target_sizes and placeholder ids equal to HF's processor's,
  bit for bit, on 5 images at max-slices 1, 9 and 36. The canvas positions equal HF's
  `get_rope_index`.
- `vision_reference` against HF's `get_image_features` (fp32) on a tiny random MiniCPM-V; the
  card's tower against the reference on a tiny tower (cosine >= 0.999 every row, 4 crop sizes)
  and against HF's tiny model (two crops).
- The projector's GELU (cats.jpg, refract.png, real weights): tanh against erf, row cosine
  >= 0.99987, the first reply token's dKL 1e-7 .. 6e-7, the same top token. The card uses tanh.
- The card's tower on the ISA simulator against HF's features (fp32), real weights:

| image (crops) | format | row cosine mean | min | first token dKL |
|---|---|---|---|---|
| cats.jpg (3) | int8 | 0.9946 | 0.776 | 3.3e-7 |
| cats.jpg (3) | int8, ViT MLPs fp4 | 0.9699 | 0.632 | 2.1e-7 |
| refract.png (3) | int8 | 0.9936 | 0.957 | 7.8e-6 |
| refract.png (3) | int8, ViT MLPs fp4 | 0.9611 | 0.673 | 3.6e-6 |

  The ViT MLPs stay int8. The first token's dKL hardly moves with the rows; the reply's dKL
  follows.
- **The reply's dKL** (`tools/minicpmv_hf.py vision --rows`, the simulation host). HF's greedy reply to
  "Describe this image in one sentence." is cut at its first `<|im_end|>` (21, 32 and 32
  tokens). The reply is then teacher-forced with each variant's rows in place of HF's, and the
  table gives the KL from HF's next-token distribution in nats per reply token, plus the share
  of reply tokens whose top token is HF's.
  - card int8: the ISA simulator's rows, which equal the card's.
  - emulated: the fp32 reference with the MXU's numerics (scratch vit_emul.py), int8, or
    two-pass int8 weights (a second int8 pass on the first pass's remainder) in layers 0-1,
    0-3 or 0-6.
  - fp32 tanh: no quantization, the projector's GELU tanh as on the card.

| variant | cats.jpg (3 crops) | refract.png (3) | doc.png (7) | mean | top token | tower s a crop | weights MB |
|---|---|---|---|---|---|---|---|
| card int8 | 7.6e-3 | 3.7e-3 | 4.6e-3 | 5.3e-3 | 0.95 / 1 / 0.97 | 5.77 (card) | 571 |
| emulated int8 | 2.5e-2 | 2.6e-3 | 9.8e-3 | 1.2e-2 | 0.95 / 1 / 1 | 5.77 | 571 |
| two-pass 0-1 | 2.4e-2 | 2.2e-3 | 7.6e-3 | 1.1e-2 | 0.95 / 1 / 0.97 | ~6.5 (+12% MAC) | +31 |
| two-pass 0-3 | 4.4e-3 | 6.7e-3 | 1.8e-2 | 9.6e-3 | 0.95 / 1 / 0.97 | ~7.2 (+24%) | +61 |
| two-pass 0-6 | 1.1e-2 | 3.3e-3 | 1.7e-2 | 1.1e-2 | 0.95 / 1 / 0.97 | ~8.3 (+43%) | +107 |
| fp32 tanh | 2.8e-3 | 1.0e-3 | 1.6e-3 | 1.8e-3 | 1 / 1 / 1 | - | - |

  - The two-pass times scale layers 0-6's measured linear phases (qkv and out, 364 ms a layer).
  - At this sample size the reply dKL does not separate the int8 variants: the emulation and
    the card differ 3x on cats.jpg. No two-pass range lowers it on all three images.
  - The card's int8 tower costs ~5e-3 nats a reply token. That is ~3x the floor of tanh alone,
    half the language model's int8 dKL (9.9e-3) and a ninth of its card formats' (4.7e-2 from
    float, 5.1). Every variant's top token disagrees with HF's at most once per reply.
  - Decided (2026-10-13): the tower stays int8, with no fix. The variants cannot be
    told apart at 3 images; the card's 5.3e-3 is half the language model's own int8 dKL; the
    top token differs from HF's at most once a reply.
- **The 0.776 row** is one row of 203: cats.jpg's second crop (40 x 28), output row 68, its
  bottom edge (refract.png's worst is 0.957; the next worst of cats.jpg 0.978). Stage by stage
  on the ISA simulator against the fp32 reference (scratch vit_stages2.py, the row's 16 patches
  and 4 merged rows tracked):
  - through layer 6 and the window merger every row stays at cosine >= 0.998, the row's own
    16 and 4 at >= 0.9993;
  - from layer 9 the residual holds a massive activation (|x| ~810, one row's channel 514), and
    from layer 12 a few rows drift, the row's merged row from 0.9988 (layer 11) to 0.979
    (12), 0.953 (15), 0.916 (24) and 0.718 (25); the error sits in its largest channels
    (875: -30.6 against -19.9 at layer 25);
  - the post LayerNorm brings it to 0.945 and the projector's 2 x 2 merge to 0.776.

  It is int8's precision, not a fault of the card's: the fp32 reference with the MXU's numerics
  (quant.fake_w, fake_act, mx_matmul; scratch vit_emul.py, a laptop, torchvision 0.28's crop)
  gives the row 0.737 at layer 25 and 0.820 out, the worst row too. One operand at a time
  (layer 25 / out): the weights int8 0.831 / 0.956, the LayerNorm outputs 0.928 / 0.994, the
  MLP's hidden 0.928 / 0.994; Q, K, V, P and the attention output 0.990 or more. Of the
  weights, fc2 (0.939) and layers 0-6 (0.855 alone; layers 18-26 0.9998) count most: a
  perturbation of the row before the merge grows through the late layers. bf16 weights and
  inputs (HF's own dtype) keep it at 0.998 / 0.9999. The language model hardly sees it (the
  first token's dKL 3.3e-7); the reply's dKL is the measure.
- End to end on a tiny MiniCPM-V with the real tokenizer: otpu-chat's prompt ids equal HF's
  processor's in a 3-turn conversation with 2 images.

### 5.3 The vision side's residence

Method: cachesim on qwen35-35b's router traces (4 texts of 2048 tokens), lfu_layer (the card's
policy), with its link model at the placed Gen2 rate of 3.0 GB/s, at the slot counts of 4.2.
The model reads about 15% above the card in absolute tok/s, so the differences are what count.

| option | slots | misses / token | model tok/s | ms / token against none |
|---|---|---|---|---|
| none / staged (between images) | 1640 | 85.5 | 6.39 | 0 |
| streamed per crop (a ~115 MB buffer and the work area) | 1560 | 88.6 | 6.33 | +1.5 (+1.0%) |
| resident, fp4 everywhere | 1400 | 95.3 | 6.22 | +4.3 (+2.7%) |
| resident, int8 with the ViT MLPs fp4 | 1320 | 98.8 | 6.16 | +5.8 (+3.7%) |
| resident int8 | 1240 | 102.6 | 6.09 | +7.7 (+4.9%) |

- Accuracy rules out the fp4 ViT MLPs (5.2: row cosine minimum 0.63 / 0.67).
- TTFT. The tower computes about 3.7 s a crop at MCOLS 4 (the card: 5.8 s, 6), 37 s for a
  10-crop image.
  - Resident: nothing more.
  - Staged: 571 MB of pieces over the link once per prompt with images, about 0.2 s, then
    refills of the evicted least-used experts (at most ~440 misses spread over the next tokens,
    about 0.2 s of link).
  - Streamed per crop: 543 MB a crop overlapped with compute. H2C contention (about 0.17 s/GB
    of card time) is about +0.09 s a crop: +0.9 s, 2.5%, on 10 crops.
- Data moved per 10-crop image: staged 0.6 GB plus at most 0.6 GB of refills; streamed 5.4 GB.

The choice (2026-10-13): staged through the expert server (4.2). Decode keeps every
slot between images, and it moves the fewest bytes. The card measured it in session mcpv2 (6):
decode +8.4% (36 slots a layer against 27), an image +0.3-0.9 s. It has been the default since
then; resident is `--vision-resident`.

## 6. Card plan

On the card host, the data disk's `minicpmv/` holds:
- the slim checkpoint: 6,021,430,313 bytes (sha256 sums in `SUMS-slim`, all checked there on
  2026-10-13);
- the MX pool: 17,532,190,720 bytes with its .packed, .format and .key (`SUMS-pool`, checked).
  Staged vision adds its 446 entries after the experts (763.6 MB): packed by the first staged
  run, or in the one repack that also adds the MTP layer (les's pool-extend: the experts byte
  for byte, then layer 40, then the vision pieces from 41 x E);
- `img/` (the sessions' images) and `refs/` (their ISA references);
- `pylib/`: torchvision 0.29.1+cpu, pillow 12.3.0 and transformers 5.18.0 over the venv's torch
  2.14.0 (the simulation host's `~/mcpv/tf518`, file for file). Neither host's venv has torchvision or
  pillow (pyproject's `minicpmv` extra names them; the tests that need pillow skip without it),
  and without torchvision the resize falls back to torch's (not bit for bit HF's): the
  card's image rows would not be the references'. The processor's crops of the sessions'
  images hash the same on the simulation host and the card host with it.

Relays go the simulation host -> a laptop -> the card host, between card sessions. Nothing large goes on /home.

Sessions (scripts on this branch):
1. **mcpv1** (`tools/offload/sessions/mcpv1.sh`; production's bitstream and configuration):
   through otpu-chat's engine and Chat (`tools/qual/chatmoe.py --image N:PATH`), each run
   token-exact against the ISA simulator's (REF=1 on the simulation host):
   - text: chatmoe.py's two text turns;
   - thumb: cats.jpg at `--max-slices 1` (one crop), two turns;
   - sliced: doc.png at the default 9 slices (7 crops), one turn;
   - serve: otpu-serve with `tools/serve_bench.py`'s images part (data: URLs, max_slices 1):
     the first reply must be thumb's; the repeat and the next turn reuse the context past the
     image, another image does not.
   Each turn reports the crops' seconds through the card's tower, TTFT, prefill and decode
   tok/s and the slots' hits.

   PASSED 2026-10-13 16:36 (the card host): text, thumb and serve token-exact; serve's first reply
   was thumb's, its repeat had 87 of 88 prompt tokens cached (TTFT 0.63 s) and its next turn
   103. Its times are the cold ones explained under mcpv2.
2. **mcpv2** (`tools/offload/sessions/mcpv2.sh`): where the time goes, and staged vision.
   - tower: `tools/minicpmv_tower.py time`;
   - text again, warm;
   - staged: thumb with `--vision-staged`, after a warm-up run that builds its image and packs
     the 446 pieces into the pool;
   - stower: the tool with the staged tower.

   Each run follows a page-cache reset: every other big file is dropped and the pool read in
   with dd.

   PASSED 2026-10-13 17:22 (the card host; RUNS tower text staged stower, sliced waits for its
   reference); text and staged are token-exact.

   Findings:
   - **The tower.** A crop of 1,008 or 1,024 patches takes 5.73-5.81 s of device time. Its
     wall time is device time plus 0.07 s: 87 runs, 7.2 MB of rows written, the projector's
     rows read.
     - By phase (1,024 patches): out (o-proj and MLP of layers 0-6) 2.05 s, out2 (layers
       7-26, 256 rows) 1.39 s, attend 0.63 s, qkv 0.50 s, wmerge 0.42 s, qkv2 0.36 s,
       attend2 0.15 s, the rest 0.27 s.
     - Against 512 MAC a cycle: qkv 84%, wmerge 91%, out and out2 58%, attend 54%. The RTL
       co-simulation of each phase is in tools/minicpmv_tower.py cosim.
     - **Where out's and out2's 42% went: TMEM banks.** Measured with `cosim --profile` at
       28 x 36: the MXU's stall counters and the slice's activity windows. out has out2's
       cycles per 16-row group.
       - out2 took 9.60M cycles. The MXU popped 5.59M (58.2%, its nominal pops) and was
         blocked 3.65M (38.0%): chunks in its FIFO, no pop.
       - Nearly all of that is W_down's MMs (16x256 . 1152x256^T, ACC into the residual):
         5.82M cycles for 2.35M pops, back-pressured 3.44M. W_up's MMs ran at 92% and W_o's
         at 82%.
       - The cause: vit_out_mlp loaded the residual x with ol.load, which keeps its DRAM row
         stride, 1,152 words, a multiple of TMEM's 8 banks. A streamed row's 16 results then
         sit in one bank, and the drain's lanes (a greedy pick up to the first bank conflict)
         write one a cycle: 16 cycles an output column.
       - W_down's array takes 8 (KB 2 at CHUNK 256, times 4 replay groups), the fewest pops
         per column of any MM here, so only it was drain-bound.
       - The drain also lost TMEM arbitration to the VPU (the next chunk's GELU, VPU-first):
         12.9% of the cycles.
       - The rest: gaps between MMs 3.6% (the QACTs of the LayerNorm operand and of the O rows
         at each group's start). DRAM cost nothing: never starved, port B 15% busy.
       - wmerge runs the same MLP into `ol.zeros`, whose row stride is odd (ol.empty, zeros
         and full give 2-D tiles one, as does a fresh MM output), and it reaches 91%.
     - **The fix** (`kernels/vit.py`): the residual is loaded into an `ol.empty` tile, whose
       row stride (C + 1) is odd. `compiler.load` takes such a padded destination as one LD a
       row.
       - out2 went from 9.60M to 6.87M cycles (-28.5%, co-simulation); pops are 81.4% of it.
       - The tower's rows are bit-identical: same arithmetic, another TMEM layout. Checked on
         the tiny tower on the ISA at 4 crop sizes.
       - vit_embed keeps its one dense LD: its MM is KB 5, 20 cycles a column against the
         drain's 16, and the per-row LDs made it 4% slower.
       - Left in out2: the drain still loses TMEM arbitration to the VPU on W_down (18% of
         the cycles), and 5.3% of gaps (the LayerNorm operand's QACT, the 16 per-row LDs).
       - Test: test_the_towers_mms_drain_at_the_arrays_rate. Every MM whose array takes fewer
         cycles a column than its rows must spread them over distinct banks.
     - **attend** (13.34M cycles at 28 x 36) is VPU-bound: VPU busy 97%, MXU pops 47.9%.
       - The MXU's drain lost TMEM arbitration 27.5% of the cycles.
       - The gaps between its MMs, 24%, are blamed on the P operand's QACTs and the row
         maxima.
       - The softmax's EXP2SUB leads the VPU, followed by ~60K one-word-a-row VOPs (fill,
         copy, recip, max).
       - On the efficiency list.
   - **The first crop of a patch count.** It takes ~23 s more on the card host (doc.png's
     first 32 x 32: 28.9 s). BoardBackend.start computes each new program's WAITC wait
     entries (waits.entries) the first time it is started: attend 14.1 s, wmerge 2.5 s, qkv
     1.4 s, the rest 4.7 s. On a laptop it is 6-17 s, depending on load; attend's 214,534 visits
     dominate it.
     - The analysis is linear in the instructions a run executes: it walks every iteration
       of each loop of known count (attend at 1,024 patches: 214,534 visits). Each visit
       checks ~68 footprints of the units' histories (75% of the time).
     - The checks are now split by address space (waits._MemoPass, the default). A run's
       TMEM and ACT footprints repeat while its DRAM addresses advance: attend has 977
       distinct histories of local footprints in 214,534 visits.
       - So a visit's local checks are kept by those histories, and the DRAM ranges are
         checked only where an instruction has some.
       - The entries are the same (tests: random programs over every unit, the tower's
         programs resident and staged).
       - Speed on a laptop: the tower's 13 programs at 1,024 patches 9.5 -> 2.4 s (attend
         5.3x); Qwen3.5-0.8B's, Qwen3-0.6B's and LFM2.5-230M's decode programs 2.3-4.2x.
     - Since then (decision 9) `opentpu/host/vision_warm.py` computes them ahead. After the
       engine's load, a spawned process at nice 19 builds the image's layout and computes
       every count's programs' entries into the program cache's disk, most frequent first.
       Each count is done once per tree: a marker lists its entries' keys.
     - The entries are kept in the program cache's disk, keyed by the tree's sources, so this
       comes once per host tree and patch count.
     - It is mcpv1's thumb's vision_s, 30.3 s: 5.8 device + ~24. serve's first image TTFT,
       16.6 s, is 10.8 s of prefill and 5.8 s of tower, its entries kept from thumb.
     - The processor gives 17 patch counts (800 to 1,200). Over thumbnails and 9-slice
       grids of 200-4000 px images, 1,008, 1,024, 960 and 1,056 are 78% of the crops.
   - **mcpv1's cold text turns** (0.5 tok/s decode, load 200.8 s). Its image's first build
     peaked at 28.4 GB of the host's 31 GB and took the pool out of the page cache. The pool
     is on /data, a 7,200 rpm disk (sdb, WD10EZEX), so 2,979 misses became random disk
     reads. "pool read in 0 s" was the relay's sha256 check's cache, before the build.
     - Warm (mcpv2): turn 1 decode 3.71 tok/s (device 4.16), turn 2 4.44 (device 4.46).
   - **Staged against resident**, thumb, both token-exact:
     - slots: 1,440 a layer staged (36 a layer) against 1,080 resident (27);
     - vision: 6.72 s against 5.8 s (the placement 0.24 s; device time +0.7% from the
       table's RLDs);
     - decode, device tok/s: turn 1 4.56 against 4.49, turn 2 4.80 against 4.43 (+8.4%);
     - MB a token: 204.8 / 182.4 against 221.3 / 212.9;
     - hits: 0.626 / 0.667 against 0.596 / 0.612.
     - stower (the tool): the placement takes 0.22-0.24 s an image and the release 0.001 s;
       a crop's device time is +0.7% (5.81 against 5.77 s). doc.png's 7 crops take
       41.0-41.4 s against 40.5-40.9 s resident.
     - The warm-up's first build of the staged image took 103 s; the pool grew by 763.6 MB.
   - **Decode against qwen35-35b** (~5.3 tok/s at 41 slots). MiniCPM-V's formats (5.1, 7)
     read 265 MiB more int8 weights every token (1,520 against 1,255 MiB), ~28 ms a token at
     decode's ~9.8 GB/s. They also have fewer slots.
3. **mcpv3** (`tools/offload/sessions/mcpv3.sh`): this tree's defaults (staged vision, the
   tower's wait entries computed ahead, waits._MemoPass, the odd-stride residual), the sliced
   reference's first run, and what the pool's disk costs.
   - **First run (2026-10-13, ~19:25 on the card host's clock): FAILED in 3 minutes, on the host only.**
     - Every run died opening the engine: "staged vision: 438 pieces, 0 expert slots past
       k = 8".
     - 3779dc22 checked staged vision's room in every minicpmv.Image's constructor (the
       pieces against the expert slots past k).
     - streamed.engine first builds a probe image, the layout alone, at k slots a layer.
       mcpv2's staged runs came before that check.
   - **The fix:** the room is Image.check_room, run on an engine's image (streamed.engine once
     it opens, and Vision).
   - **Tests:**
     - test_streamed_engine_opens_staged_vision_past_its_probe runs the tiny model through
       streamed.engine.
     - test_the_real_staged_layout_has_room_at_the_cards_slots builds the real layout on the
       card's 4 GiB: the probe at 8 slots a layer, the engine's 36.
   - **mcpv3b PASSED** (2026-10-13 19:56 on the card host's clock; tree 3e3190ab). ECC 0/0, selftest all pass.
     - Token-exact: warmup, text and thumb. Also sliced, for the first time: doc.png's 7
       crops, 488 prompt tokens, 32 reply tokens.
   - **The tower, as co-simulated.**
     - A crop's device time is 4.78-4.83 s (mcpv2: 5.73-5.81 s, -17%).
     - out: 206.1 ms a run at 1,008 patches and 209.3 at 1,024 (co-simulation 206.0 /
       209.3). out2: 51.5 / 52.3. last: 53.2 / 54.0.
     - doc.png's 7 crops encode in 34.5 s (mcpv2 staged: 41.0-41.4); the thumbnail in 5.7 s.
     - A new patch count's first crop builds its programs in 0.42-0.67 s, with no
       wait-entry analysis at its start (vision_warm and waits._MemoPass). mcpv1's took
       ~23 s more.
   - **Decode** (staged, device tok/s): 4.4-4.7 in the chat turns, 5.1-5.2 in pooldisk's
     longer turn.
   - **The cold pool, once more.**
     - The warm-up's first build of the image (load 102.6 s, a 30.2 GB peak) took the pool
       out of the page cache.
     - Its turn 1 decoded at 1.25 tok/s (device 4.42), its prompt in 15.7 s. The same turn
       warm (text): 3.88 tok/s, 4.9 s, with the same 2,950 misses.
   - **pooldisk: the dropped arms are INVALID.**
     - Both reported the whole pool still in the page cache: the engine's PoolFile maps the
       pool and touches every page it reads, and posix_fadvise DONTNEED does not drop a
       mapped page.
     - The cached arms stand: 2,523-2,605 misses at 0.52-0.55 ms of reads each (0.18-0.19
       ms a read, about 2.9 reads a miss), decode 5.1-5.2 tok/s.
     - Fixed in pool_disk.py: the engine's maps of the pool are released first (madvise
       DONTNEED), then posix_fadvise. An arm whose page cache is not what it says (mincore)
       is marked INVALID.
     - Checked on a Linux host, with a mapped, touched PoolFile of 512 MB: plain DONTNEED
       left all 537 MB cached; the fix left 0.
     - For the next session: the pool on /data and an SSD pool, the same arms.
4. **pooldisk** (`tools/offload/sessions/pooldisk.sh`, tree c20e4ea6): what the pool's disk
   costs a warm process.
   - MiniCPM-V's pool on /data (sdb: the 7,200 rpm WD10EZEX, ext4), and qwen35-35b's on /home
     (sda: a Crucial BX500 SATA SSD, btrfs on dm-crypt). The latter is production's file, read
     only: its size, mtime and its .packed's and .key's sums are checked before and after.
   - Per pool: a warm-up process, then pool_disk.py's arms cached, dropped, dropped, cached.
     Each arm asks the same text turn, 26 prompt tokens and 32 reply tokens, greedy.

   PASSED 2026-10-13 20:20 (the card host). Every arm valid by mincore; q35's files unchanged;
   selftest all pass before and after.

   | pool | arm | prefill tok/s (device) | decode tok/s | misses | ms of reads a miss | disk reads |
   |:--|:--|:--|:--|:--|:--|:--|
   | MiniCPM-V, HDD | cached | 5.27 (6.17) / 5.24 (6.16) | 5.06 / 5.10 | 2,605 / 2,523 | 0.52 / 0.56 | none |
   | | dropped | 0.39 (0.40) / 0.39 (0.40) | 0.99 / 0.99 | 2,544 / 2,529 | 31.9 / 32.5 | 2,785 at 28.1 ms / 2,737 at 29.0 |
   | qwen35-35b, SSD | cached | 5.32 (6.15) / 5.26 (6.10) | 5.60 / 5.63 | 2,511 / 2,429 | 0.57 / 0.63 | none |
   | | dropped | 0.68 (0.73) / 0.69 (0.74) | 1.82 / 1.50 | 2,456 / 2,442 | 14.6 / 16.8 | 2,489 at 13.0 ms / 2,626 at 14.5 |

   - **Warm, the disk does not matter.** A miss takes 0.52-0.63 ms of reads on both: about 2.9
     reads of 0.18-0.20 ms each, from the page cache. MiniCPM-V decodes ~10% slower than
     qwen35-35b because of its formats and slots (mcpv2), not the disk.
   - **Cold, a miss reads its expert (1.71 MB) from the disk:** ~32 ms on the HDD (54 MB/s),
     15-17 ms on the SSD (100-120 MB/s).
     - Prefill is 13.5x slower on the HDD and 7.7x on the SSD; decode 5.1x and 3.1-3.7x.
     - The turn took ~99 s cold against ~11 s warm on the HDD, ~57 s against ~11 s on the SSD.
   - **Reading the whole pool in order costs about one cold turn.** The last cached arm read it
     in with 16 MB reads: 79.5 s on the HDD (230 MB/s), 36.7 s on the SSD (478 MB/s). A cold
     turn's reads took about as long, 81-82 s and 36-41 s, for about a quarter of the pool
     (~2,500 of its 10,240 experts).
   - **The SSD read 3-4x slower than on 2026-10-01** (docs/offload.md 10.6: 4.2 ms an expert
     one at a time, 3.4-3.6 queued). Not explained yet. Candidates:
     - q35's warm-up had just rebuilt the 35B's image under this tree's keys (load 97.4 s)
       and written it to the image cache on the same SSD, a DRAM-less SATA drive: its
       writeback, and btrfs discarding the cache's evicted files.
     - The MX pool's extents: written 2026-10-12 onto a btrfs 84% full.
     - The in-order read after the dropped arms still ran at 478 MB/s.
     - `tools/offload/pool_reads.py` tells them apart, on the host alone and read only: the
       file's extents (filefrag) and its mount, 2 x 200 random experts dropped and read back
       (one at a time, then 8 queued at once), 512 MB in order. The simulation host's NVMe, a 2 GB test
       pool: 0.61 ms an expert one at a time, 0.52 queued.
     - **pool_reads on the card host** (2026-10-13 21:29 on the card host's clock, host idle, under the lock;
       q35's size, mtime and sums unchanged):

       | pool | file | one at a time | 8 queued | 512 MB in order |
       |:--|:--|:--|:--|:--|
       | MiniCPM-V, HDD (ext4) | 16 extents | 20.2 ms (85 MB/s) | 18.0 ms (95 MB/s) | 145 MB/s |
       | qwen35-35b, SSD (btrfs compress=zstd:3 on dm-crypt) | 2,214 extents, mean 7.7 MB; 1,156 compressed | 7.3 ms (235 MB/s) | 4.7 ms (364 MB/s) | 467 MB/s |

     - So the file is not the cause: one extent boundary every 7.7 MB, and the compressed
       extents (btrfs's at most 128 KB each, ~150 MB in all) are under 1% of it. The idle SSD
       reads an expert in 7.3 / 4.7 ms, 1.4-1.7x 2026-10-01's, against the session's 13-14.5;
       the HDD in 18-20 ms against the session's 28-29.
     - The rest is the session's: in both, a read in the engine took ~1.4x (HDD) to ~2-3x
       (SSD) the idle disk's. Candidates: the image's write just before (the SSD only), and the
       engine's reads (a miss in ~2.9 parts, a request's misses queued at once, the DMA thread
       beside them). Telling them apart needs the card: pooldisk's q35 arms again with the
       image already in the image cache.
   - **For serving: never serve from a cold pool.** The disk then sets only how long the wait
     is.
     - The engine already reads every packed expert into the page cache at each open
       (open_pool's PoolFile.warm: in file order, in a thread, from before the image build).
     - A 35B image's build (a 28-30 GB peak of the host's 31 GB) takes the pool back out of
       the page cache (mcpv1, mcpv3b). An open from the image cache (6-8 s) leaves the warm
       running: the first turn's misses go to the disk, competing with it.
     - Waiting for the warm costs ~80 s on the HDD or ~37 s on the SSD, once, and every turn
       after it is warm. For the turn above: 91 s against 99 s served cold on the HDD, 47 s
       against 57 s on the SSD; the cold turn also leaves three quarters of the pool on the
       disk.
     - Not there yet: otpu-serve's /health says ok and the server takes requests as soon as
       the engine opens; the warm's own seconds are not logged; nothing warms the pool again
       after a build that evicted it.
     - The pool must stay in RAM. One 35B pool fits in the host's 31 GB (18.3 or 17.5 GB,
       beside otpu-serve's 4 GB KV store and the process); both together (35.8 GB) do not.
       Alternating MiniCPM-V and qwen35-35b reads a pool again at every switch.
     - So moving MiniCPM-V's pool to the SSD shortens the wait of an open with a cold pool
       (a reboot, a build, a switch from the other 35B) from ~80 to ~37 s. It stays with the
       production rotation, as decided; it would count per switch if the two alternated.

## 7. Decisions

1. Vocabulary 248,144: the partial head block, tested bit-identical to a zero-padded head.
2. The projector's GELU: tanh (5.2).
3. No MTP head in the checkpoint; DSpark later.
4. The card host's storage: the data disk's `minicpmv/`.
5. `--max-slices`: default 9, in otpu-chat and as otpu-serve's request field and option.
6. M-RoPE rows written by the host's gather (4.3).
7. The vision weights: int8, without a fix for the 0.776 row (the reply's dKL, 5.2; the fp4
   ViT MLPs fail on accuracy). They are staged through the expert server's slots (4.2, 5.3), the
   default since session mcpv2 (6).
8. The language model's formats: the 35B's mix with DeltaNet int8 in layers 0-19 and the shared
   expert int8, +3.68% dKL at 35 slots a layer. It is also the best by dKL against decode time
   (+31 ms a token, 5.1), the rule for format variants from now on.
9. A new crop size's tower programs: their wait entries are computed ahead at the engine's load
   (opentpu/host/vision_warm.py, 6). Next, with a keyed change, the tower's programs take N as
   a run argument: one program set for every crop size.
