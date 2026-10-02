# Weight formats per layer

Decode streams every weight once per token, so a model's decode speed follows its weight bytes:
fp4 layers run 1.5-1.6x faster than int8 on the card (Qwen3.5-2B 12.09 against 8.02 tok/s,
Phi-4-mini 6.56 against 3.99), and they cost accuracy. A mix keeps most layers in int8 and puts
the weights that cost the least accuracy per byte in fp4. This page covers the mechanism, the
measure, the rule we choose by, and the recommended mixes (wformat "mix").

In short:

- **Mechanism.** A formats string sets the format per kind of weight and per layer range over
  the image's `wformat` (`opentpu/llm/formats.py`): `OTPU_FORMATS`, a caller's `formats=`, or
  the named choice `wformat="mix"`, which is int8 with the model's recommended string
  (`Spec.mix`, from `formats.MIXES`). int8, int4 and fp4 stay uniform.
- **Measure.** dKL: the mean KL divergence of the variant's next-token distribution from the
  float model's, beyond int8's, with its standard error paired over the tokens
  (`tools/formats_scan.py`). It is about ΔNLL against int8 (its % is about the perplexity
  change in %), without the noise of the sampled tokens.
- **Rule.** A mix qualifies if dKL% <= 0.1 x its decode gain in % over int8, with at most two
  layer layouts (Qwen3.5: one) unless the compile check passes (decode and prompt runs, buckets
  1 and 16: Gemma 4; E2B's 4-row prompt run of bucket 2 does not fit, [gemma4.md](gemma4.md)),
  and the LM head in int8. The default is the fastest mix at least
  1 SE under its bar; a dKL gap under 2 SE is a tie, which goes to the mix with fewer runs.
  Finals are at 2000 tokens.
- **Choices.** Phi-4-mini: `mlp@4-29=fp4`, 31% faster than int8 for dKL 2.8% (fp4: 64% for
  16%). Qwen3.5-4B: `delta=fp4,mlp=fp4` (the attention layers int8), 56% for 5.3% (fp4: 64% for
  6.5%, on its bar). SmolLM3-3B: `gateup@9-35=fp4`, 23% for 2.1%. LFM2-2.6B:
  `conv=fp4,mlp=fp4` (the attention layers int8), 77% for 7.4% (fp4: 82% for 11.8%). Gemma 4
  E2B: `attn@15-24=fp4,mlp@15-34=fp4` (the KV-shared layers' MLP, layers 15-24's attention),
  30% for 2.6% against int8 with the int8 PLE table (fp4 layers: 56% for about 21%).
  Qwen3.5-2B: none qualifies. On the card
  the mixes run at the predicted speed (within 0.1%), and their images give the ISA simulator's
  tokens (Phi's and SmolLM3's through a proxy).

## Formats strings

`"kind=fmt,kind@a-b=fmt,..."`, fmt one of int8, int4 and fp4, a..b checkpoint layers. A rule
with a range wins over one without; among equals the first one wins. The kinds: attn, mlp
(the default of gateup and down), gateup, down and head (no range); LFM2 also conv, Qwen3.5
also delta (the DeltaNet mixer). The string comes from the caller, else `OTPU_FORMATS`, else
the model's own default.

`wformat="mix"` (`--wformat mix` in `tools/compare_hf.py`, `tools/perf_qwen.py` and
`tools/decode_profile.py`) is int8 with `Spec.mix` unless the caller or `OTPU_FORMATS` gives a
string. `formats.mix_for` finds a checkpoint's mix in `formats.MIXES` by its text config's
(model_type, num_hidden_layers, hidden_size, vocab_size); a model without one refuses "mix".
`otpu-chat`'s default, `--wformat auto` (`formats.auto`), is "mix" where the model has one and
int8 elsewhere; it prints the formats it chose (`weights: mix: int8 + gateup@9-35=fp4, head
int8`), and `otpu-smi` shows them. `Engine`'s own default stays int8.

## Layouts, runs and IMEM

Each layer's formats (one per kind) are its formats group. Each group has its own layer block
layout (`Image.layouts`), and a hardware loop runs one layout: the layer plan's runs are the
maximal stretches of layers with the same (kind, group) key, and each run is one loop body in
the program. So the IMEM (4096 instructions) bounds how a mix may split the layers, and each
family bounds it differently. Program sizes, resident generate at the board configuration,
bucket 1 / 16:

| Model | One layout | Two layouts |
|:--|--:|--:|
| Phi-4-mini | 538 / 1110 (1 run) | 900 / 2044 (`gateup@0-23`, 2 runs), 1262 / 2978 (`mlp@4-27`, 3 runs) |
| Qwen3.5-4B | 2059 / 2356 (1 run) | 3972 / 4565 (`mlp@0-15`, 2 runs) |
| Qwen3.5-2B | 2308 / 2461 (1 run) | 4484 / 4789 (`mlp@0-11`, 2 runs) |
| LFM2-2.6B | 1806 / 3510 (5 runs) | 1974 / 3678 (`mlp@7-20`, 5 runs), 2358 / 4630 (`mlp@0-14`, 8 runs) |

Gemma 4 E2B, resident decode at bucket 16: 1769 instructions in int8, 1735 with
`attn@15-34=fp4,mlp@15-34=fp4` (2 runs), 2579 with `attn@15-24=fp4,mlp@15-34=fp4` (3 runs), 2418
with a boundary at layer 10 too (4 runs). Its prompt runs ([prefill.md](prefill.md)) take fewer
rows with more bodies: [gemma4.md](gemma4.md) has R_max by bucket.

- **Qwen3 and the Llama-likes** (SmolLM3, Phi-4-mini): a body is one layer, so two layouts in
  up to three runs fit.
- **Qwen3.5**: a body is four layers (three DeltaNet, one attention), about 1,900-2,300
  instructions, so a second run does not fit: one layout only.
- **LFM2**: the plan already has five runs (`lfm2.plan`); a second layout fits where its
  boundary falls on a run boundary, and adds runs elsewhere.
- **Gemma 4**: a body per (attention kind, own K / V, MLP width, formats): E2B has four (sliding
  and global, in the layers with their own K / V and in the KV-shared ones from 15) in two runs.
  A format boundary at 15 adds none; one at 10 or 25 adds two bodies and a run or two, which
  still fit. For Gemma 4 the compile check replaces the two-layout limit: decode and the prompt
  runs compile and fit the IMEM in buckets 1 and 16.

In DRAM each layer block has its kind's size in its formats group (`Image.loc`,
`Image._off`), so an fp4 block takes fp4's bytes. The Qwen3.5-4B image fits the card's 4 GiB
only with some fp4: at a 2048-token capacity int8 is 4237 MiB, `attn=fp4` 4097, `gateup=fp4`
3517, `delta=fp4,mlp=fp4` 2675, fp4 2535 (a 4096-token capacity adds about 37 MiB).

## The measure

`tools/formats_scan.py` runs float64 emulations of the device's quantization (the families'
`emulated_logits` with the loops swapped: layers outer, every variant at once) over the first
tokens of `docs/isa.md`, prose the models have not memorized. For each variant it records the
NLL per token and the KL divergence of its next-token distribution from the float model's and
from int8's.

- **dKL** = mean KL(float || variant) - mean KL(float || int8), its standard error paired over
  the tokens. Where the float model is calibrated it estimates the log perplexity ratio to
  int8.
- **Why not perplexity.** At 900 tokens the perplexity ratio's paired SE is 1-2% (SmolLM3's
  mixes), as large as the differences between mixes; dKL's is 0.03-0.6% for the mixes near
  their bars. Mixes are ranked by dKL, and the finals run 2000 tokens.
- `scan MODEL OUT.json`: pass 1 runs float, int8, fp4 (int8 head), each kind in fp4, each kind
  in each quarter of the layers, and the head; pass 2 every set of whole kinds (and the head),
  and the quarter groups added in order of dKL per byte saved (the scans before Gemma 4's: of
  NLL). `ppl MODEL FORMATS... --tokens 2000`: the given strings.

## The rule

Decode time per token is linear in the weight bytes b: T(b) = T4 + k (b - B4) cycles at 133.33
MHz, with T4 and B4 the card's fp4 (int8 head) decode and its bytes, and k from the card's
int8 and fp4 pair of the same model (Phi-4-mini 8.13 K cycles per MB) or family (Qwen3.5: the
2B's, 8.15; the 4B's int8 image does not fit the card; Gemma 4 E2B: session mix4's int8 and
fp4 on fmvf, 8.04). The gain is T(int8) / T(b) - 1.

A mix qualifies if dKL% <= 0.1 x gain%: 0.1% of perplexity for each 1% of speed. Among the
qualifying mixes with at most two layouts (Qwen3.5: one; more where the compile check passes,
Gemma 4) the fastest wins; a dKL gap under 2 SE is a tie, and the tie goes to fewer runs. The
LM head stays int8 in every mix.

A default needs at least 1 SE of margin under its bar, so that the noise of one text cannot
flip it. A faster mix that qualifies inside 1 SE is not the default; it is listed below with
its numbers, as an opt-in through `OTPU_FORMATS` (with `wformat="mix"` or int8).

## Choices

### Phi-4-mini

int8 3.99 tok/s, fp4 6.56 (card, build B). At 900 tokens fp4 is dKL +16.1% (SE 1.3) and no whole
kind qualifies: gateup in every layer +3.27% for +24% (bar 2.4), attention +3.95% for +11%. The
down projections of the first and last quarters are costly (+7.4% and +5.0%), the middle
quarters cheap (+0.3% each). Finals, 2000 tokens:

| Formats | Layouts / runs | dKL % (SE) | Perplexity vs int8 % (SE) | MiB | tok/s (est.) | Gain | Bar |
|:--|:-:|--:|--:|--:|--:|--:|--:|
| `mlp@2-29=fp4` | 2 / 3 | +9.48 (0.59) | +8.76 (1.09) | 2764 | 5.37 | +34.6% | 3.46 |
| `mlp@4-31=fp4` | 2 / 2 | +8.26 (1.46) | +8.91 (1.24) | 2764 | 5.37 | +34.6% | 3.46 |
| `mlp@4-30=fp4` | 2 / 3 | +5.66 (1.42) | +5.86 (1.11) | 2800 | 5.31 | +33.0% | 3.30 |
| `mlp@3-28=fp4` | 2 / 3 | +9.24 (0.57) | +8.98 (1.11) | 2836 | 5.24 | +31.4% | 3.14 |
| **`mlp@4-29=fp4`** | 2 / 3 | **+2.81 (0.14)** | +2.19 (0.70) | 2836 | 5.24 | +31.4% | 3.14 |
| `mlp@4-28=fp4` | 2 / 3 | +2.63 (0.16) | +1.95 (0.69) | 2872 | 5.18 | +29.8% | 2.98 |
| `mlp@3-27=fp4` | 2 / 3 | +9.26 (0.58) | +8.83 (1.11) | 2872 | 5.18 | +29.8% | 2.98 |
| `mlp@4-27=fp4` | 2 / 3 | +2.54 (0.15) | +1.72 (0.68) | 2908 | 5.12 | +28.3% | 2.83 |
| `mlp@6-25=fp4` | 2 / 3 | +1.91 (0.13) | +1.60 (0.59) | 3052 | 4.89 | +22.5% | 2.25 |
| `gateup@0-27=fp4` | 2 / 2 | +1.71 (0.15) | +0.83 (0.64) | 3100 | 4.82 | +20.7% | 2.07 |
| `gateup@0-23=fp4` | 2 / 2 | +1.57 (0.15) | +0.61 (0.61) | 3196 | 4.68 | +17.2% | 1.72 |
| `mlp@8-23=fp4` | 2 / 3 | +1.53 (0.12) | +1.31 (0.52) | 3196 | 4.68 | +17.2% | 1.72 |
| `attn=fp4` | 1 / 1 | +3.93 (0.28) | +2.56 (0.77) | 3388 | 4.42 | +10.9% | 1.09 |

`mlp@4-29=fp4` is the fastest that qualifies (2.3 SE under its bar). Layer 3's MLP adds 6.7
points of dKL, layers 28 and 29 together 0.27, layer 30 2.85 and layer 31 2.6 more (their SE
1.4: a few tokens). Until the probe of these edges the default was `mlp@4-27=fp4` (5.20 tok/s
on the card, below); the two run the same programs (three runs, generate 1262 / 2978).

### Qwen3.5-4B

fp4 5.88 tok/s on the card (build B); the int8 image does not fit (4237 MiB at a 2048-token
capacity), so its 3.58 tok/s is projected with the 2B's slope. At 900 tokens four single layouts
qualified: fp4 (+6.27%, bar 6.42), `delta=fp4,mlp=fp4`, `mlp=fp4` and `gateup=fp4`. The others
miss their bars (`attn=fp4,mlp=fp4` by 1.8 SE, `delta=fp4,gateup=fp4` by 1.3 SE, the rest by
more) and are slower than `delta=fp4,mlp=fp4` anyway. Finals, 2000 tokens:

| Formats | dKL % (SE) | Perplexity vs int8 % (SE) | MiB | tok/s (est.) | Gain | Bar |
|:--|--:|--:|--:|--:|--:|--:|
| fp4 (`attn=fp4,delta=fp4,mlp=fp4`) | +6.48 (0.20) | +5.96 (0.97) | 2433 | 5.88 | +64.2% | 6.42 |
| **`delta=fp4,mlp=fp4`** | **+5.30 (0.17)** | +4.42 (0.90) | 2573 | 5.59 | +55.9% | 5.59 |
| `mlp=fp4` | +3.01 (0.08) | +2.98 (0.69) | 3055 | 4.76 | +33.0% | 3.30 |
| `gateup=fp4` | +1.77 (0.05) | +2.31 (0.55) | 3415 | 4.29 | +19.8% | 1.98 |

fp4 sits on its bar (0.3 SE over; the bar itself rests on the projected int8 speed), so the
mix is the conservative `delta=fp4,mlp=fp4`, 1.7 SE under its bar: the eight attention layers
stay int8, 5% slower than fp4 for 1.2 points less dKL. fp4 stays one choice away: wformat fp4
with an int8 head, or "mix" with `OTPU_FORMATS=attn=fp4,delta=fp4,mlp=fp4`. The mix's image is
2675 MiB at a 2048-token capacity (2713 at 4096), one layout, generate 2099 / 2396
instructions.

### Qwen3.5-2B

int8 8.02 tok/s, fp4 12.09 (card, build B). No single layout qualifies at 900 tokens: the
closest, `gateup=fp4`, is +2.06% (SE 0.07) for +17% (bar 1.74), 4.6 SE over; `down=fp4` +1.35%
for +8% (bar 0.79); fp4 +7.79% for +51% (bar 5.07). The 2B has no mix: int8 or fp4.

### SmolLM3-3B

int8 5.00 tok/s, fp4 8.74 (card, build B; k 8.12 K cycles per MB). At 900 tokens no whole kind
qualifies (fp4 +12.2% for +75%, `gateup=fp4` +3.30% for +33%, bar 3.28). The gate / up
projections of the later half are the cheapest groups (+0.52% and +0.63% for a quarter, bar
0.66), the first quarter's the costliest (+1.49%); the down projections of the middle quarters
are cheap (+0.37%, +0.64%), the first quarter's not (+3.6%), and the attention costs +0.4-1.6%
a quarter for +1.5% of speed. Finals, 2000 tokens:

| Formats | Layouts / runs | dKL % (SE) | Perplexity vs int8 % (SE) | MiB | tok/s (est.) | Gain | Bar | Margin |
|:--|:-:|--:|--:|--:|--:|--:|--:|--:|
| `gateup=fp4,down@9-26=fp4` | 2 / 3 | +4.45 (0.14) | +2.80 (0.90) | 2057 | 7.23 | +44.7% | 4.47 | 0.1 SE |
| `gateup=fp4` | 1 / 1 | +3.55 (0.13) | +2.82 (0.81) | 2250 | 6.64 | +32.8% | 3.28 | over |
| **`gateup@9-35=fp4`** | 2 / 2 | **+2.13 (0.07)** | +2.01 (0.63) | 2444 | 6.14 | +22.7% | 2.27 | 2.0 SE |
| `mlp@18-35=fp4` | 2 / 2 | +2.44 (0.07) | +1.87 (0.60) | 2444 | 6.14 | +22.7% | 2.27 | over |
| `mlp@9-26=fp4` | 2 / 3 | +2.52 (0.08) | +1.24 (0.75) | 2444 | 6.14 | +22.7% | 2.27 | over |
| `gateup@18-35=fp4` | 2 / 2 | +1.24 (0.05) | +1.18 (0.47) | 2637 | 5.70 | +14.1% | 1.41 | 3.6 SE |
| `mlp@18-26=fp4` | 2 / 3 | +1.00 (0.04) | +0.50 (0.45) | 2734 | 5.51 | +10.2% | 1.02 | 0.5 SE |

The default is `gateup@9-35=fp4`, 2.0 SE under its bar (generate 1324 / 1900 instructions).
`gateup=fp4,down@9-26=fp4` gains twice as much (+44.7%) for twice the dKL, but only 0.1 SE
under its bar: an opt-in, `OTPU_FORMATS=gateup=fp4,down@9-26=fp4` (generate 1911 / 2775).

### LFM2-2.6B

int8 6.13 tok/s, fp4 11.15 (card, build pa; k 8.04 K cycles per MB). At 900 tokens fp4 is
+11.8% for +82% (bar 8.19), and the eight attention layers in fp4 cost +3.8% for +1.6% of
speed. The first quarter's layers are the costly ones (gate / up +1.08%, down +1.48%, conv
+1.47%); the other quarters' groups cost +0.3-1.2% each. A second layout fits only where it
starts on a run boundary of `lfm2.plan` (layers 2, 3, 23, 29); those below keep the five runs
and the program size (generate 1806 / 3510). Finals, 2000 tokens:

| Formats | Layouts | dKL % (SE) | Perplexity vs int8 % (SE) | MiB | tok/s (est.) | Gain | Bar | Margin |
|:--|:-:|--:|--:|--:|--:|--:|--:|--:|
| **`conv=fp4,mlp=fp4`** | 1 | **+7.35 (0.29)** | +3.65 (1.08) | 1406 | 10.84 | +76.9% | 7.69 | 1.2 SE |
| `conv@3-29=fp4,mlp=fp4` | 2 | +7.05 (0.39) | +0.32 (1.01) | 1422 | 10.73 | +75.0% | 7.50 | 1.2 SE |
| `conv@2-29=fp4,mlp@2-29=fp4` | 2 | +6.66 (0.37) | +0.19 (0.97) | 1485 | 10.29 | +67.8% | 6.78 | 0.3 SE |
| `conv@3-29=fp4,mlp@3-29=fp4` | 2 | +6.21 (0.33) | +0.30 (0.93) | 1516 | 10.08 | +64.4% | 6.44 | 0.7 SE |
| `mlp=fp4` | 1 | +5.33 (0.22) | +1.43 (0.89) | 1582 | 9.68 | +57.8% | 5.78 | 2.1 SE |
| `mlp@3-29=fp4` | 2 | +4.42 (0.15) | +1.83 (0.85) | 1676 | 9.15 | +49.2% | 4.92 | 3.3 SE |

The default is `conv=fp4,mlp=fp4`, one layout, 1.2 SE under its bar: everything but the
attention layers in fp4, 3% slower than fp4 for 4.1 points less dKL (900 tokens: +7.73%
against +11.80%).

### Gemma 4 E2B

`formats_scan.py`'s Gemma 4 family runs `tools/gemma4_quant_eval.py`'s emulation a variant at a
time up to the head's input (its KV-shared layers read earlier layers' K / V), then one pass
over the soft-capped head for every variant. The kinds are attn, gateup, down and ple (a
layer's PLE gate and projection; unranged, the PLE model projection too), the groups the layers
with their own K / V and the KV-shared ones, each halved at a multiple of five (0-9, 10-14,
15-24, 25-34). The speeds are session mix4's on fmvf (below): measured for int8, the first
mix and fp4 layers, else the session's model, T(b) = 13.375 + 8.04 (b - 1412 MB) M cycles at
133.33 MHz (it gives the measured mix within 0.1%). The first estimates took build B's fp4
point ([gemma4.md](gemma4.md): 12.611 M cycles, 1475 MB a token), on which every gain was about
3 points high.

**The PLE table** stays on the card, with its lookup: in int8 where the image leaves room for
it, else in fp4 (the image tries int8 on the card, then fp4 on the card; the host only with
`OTPU_PLE_HOST=1`). int8 layers leave no room (4.52 GiB with the int8 table), so an explicit
`wformat="int8"` takes the fp4 table, and that table is most of int8's error: at 2000 tokens
int8 with the int8 table on the host has perplexity 18.62 (float 18.60), with the fp4 table on
the card 19.15, dKL +2.42% (SE 0.08). The mixes are measured against the accurate int8, the
int8 table on the host (6.56 tok/s), so that the degraded table does not loosen their bar; the
host table is that reference only, not a default. A variant's table in the scan is the one its
image takes at 2048 tokens (otpu-chat's capacity).

At 900 tokens fp4 layers (int8 head) are dKL +18.7% against int8 with the fp4 table (about +21%
against the int8 table) for +60% (bar 6.0). Per byte, the KV-shared layers' attention and MLP
cost least (down@15-24 +0.32%, gate / up@15-24 +0.71%, attention@15-24 and @25-34 +0.36% and
+0.32%); the attention of layers 10-14 (+5.5%) and the PLE kinds (ple@10-14 +3.1%) the most.
Finals, 2000 tokens:

| Formats | Runs | Image (PLE table) | dKL % (SE) | Perplexity vs int8 % (SE) | MB | tok/s | Gain | Bar | Margin |
|:--|:-:|:--|--:|--:|--:|--:|--:|--:|--:|
| int8, `OTPU_PLE_HOST=1` (the reference) | 2 | 2.24 GiB (int8, host) | 0 | 0 | 2351 | 6.39 (card) | | | |
| int8 | 2 | 3.42 GiB (fp4) | +2.42 (0.08) | +2.82 (0.61) | 2351 | 6.37 (card) | -0.3% | | fails |
| `mlp@15-34=fp4` | 2 | 3.99 GiB (int8; fp4 at 4096) | +2.25 (0.05) | +2.86 (0.56) | 1784 | 8.15 | +27.5% | 2.75 | 9.1 SE |
| **`attn@15-24=fp4,mlp@15-34=fp4`** | 3 | 3.95 GiB (int8) | **+2.60 (0.06)** | +3.08 (0.59) | 1747 | 8.30 (card) | +30.0% | 3.00 | 6.7 SE |
| `attn@15-34=fp4,gateup@15-34=fp4,down@0-9=fp4,down@15-24=fp4` | 5 | 3.96 GiB (int8) | +3.20 (0.08) | +3.99 (0.70) | 1756 | 8.26 | +29.3% | 2.93 | -3.3 SE |
| `attn@15-34=fp4,mlp@15-34=fp4` | 2 | 3.92 GiB (int8) | +3.34 (0.08) | +3.90 (0.67) | 1709 | 8.47 (card) | +32.5% | 3.25 | -1.1 SE |
| `mlp=fp4` | 2 | 3.79 GiB (int8) | +5.90 (0.14) | +5.61 (0.94) | 1572 | 9.09 | +42.4% | 4.24 | -11.9 SE |
| `attn@15-34=fp4,gateup=fp4,down@0-9=fp4,down@15-34=fp4` | 4 | 3.74 GiB (int8) | +6.01 (0.14) | +6.23 (0.93) | 1520 | 9.36 | +46.6% | 4.66 | -9.7 SE |
| `attn@15-34=fp4,mlp=fp4` | 2 | 3.72 GiB (int8) | +7.01 (0.16) | +7.56 (1.01) | 1497 | 9.49 | +48.5% | 4.85 | -13.6 SE |
| `attn@0-9=fp4,attn@15-34=fp4,gateup=fp4,down@0-9=fp4,down@15-34=fp4` | 4 | 3.70 GiB (int8) | +7.41 (0.20) | +8.54 (1.00) | 1478 | 9.59 | +50.2% | 5.02 | -12.0 SE |
| `attn@0-9=fp4,attn@15-34=fp4,mlp=fp4` | 4 | 3.68 GiB (int8) | +8.57 (0.23) | +9.69 (1.08) | 1454 | 9.72 | +52.2% | 5.22 | -14.7 SE |
| fp4, int8 head (900 tokens) | 2 | 3.64 GiB (int8) | about +21 | | 1412 | 9.97 (card) | +56.1% | 5.6 | fails |

Image: at 2048 tokens, of the card's 4 GiB, with the PLE table the image takes. Gain over the
reference's 20.875 M cycles.

The default is `attn@15-24=fp4,mlp@15-34=fp4`, 6.7 SE under its bar (session mix5, below): the
KV-shared layers' MLP and the attention of layers 15-24 in fp4; the layers with their own K / V,
the attention of layers 25-34 and every PLE weight in int8. It runs three layer runs (boundaries
at 15 and 25; six layer bodies, resident decode 2579 instructions), and its image keeps the int8
PLE table on the card at 2048 and 4096 tokens (3.955 and 3.973 GiB, 46 and 28 MiB spare). Its
prompt runs hold fewer rows in IMEM, which costs long prompts time to the first token
([gemma4.md](gemma4.md)). The attention of layers 25-34 in fp4 as well
(`attn@15-34=fp4,mlp@15-34=fp4`, 2% faster, two runs) costs 0.74 points and puts that mix 1.1 SE
over its bar; `mlp@15-34=fp4` alone (9.1 SE under, 2% slower) keeps the int8 table only up to
about 2048 tokens (10 MiB spare) and takes the fp4 table at 4096, so it is not a default. Against
int8 with the fp4 table the faster mixes would qualify (`attn@15-34=fp4,mlp=fp4` +4.59% for
+48.9%); the bar is the accurate int8's, so the degraded table does not loosen it.

## On the card

Production build pa (`deploy_pa_e4db91c9`, Gen2, 133.33 MHz, DDR3-1066), 2026-10-02, sessions
mix1 and mix2 (tree main 4e0b866, the mixes through `OTPU_FORMATS`). `tools/qual/perf.py`: a
512-token prompt, then 64 greedy decode tokens with the host's argmax in the loop; device
tok/s:

| Model | Weights | Decode | Mcycles / token | Prefill | DRAM while decoding |
|:--|:--|--:|--:|--:|--:|
| Phi-4-mini | int8 | 4.05 | 32.936 | 14.0 | 95%, 4011 MB/token |
| Phi-4-mini | **mix** `mlp@4-27=fp4` | **5.20** | 25.640 | 14.5 | 95%, 3105 MB/token |
| Phi-4-mini | fp4, int8 head | 6.68 | 19.971 | 15.0 | 94%, 2401 MB/token |
| SmolLM3-3B | int8 | 5.07 | 26.309 | 21.4 | 95%, 3203 MB/token |
| SmolLM3-3B | **mix** `gateup@9-35=fp4` | **6.23** | 21.409 | 22.0 | 95%, 2594 MB/token |
| SmolLM3-3B | fp4, int8 head | 8.90 | 14.980 | 22.8 | 94%, 1796 MB/token |
| Qwen3.5-4B | **mix** `delta=fp4,mlp=fp4` | **5.70** | 23.384 | 12.8 | 94%, 2770 MB/token |
| Qwen3.5-4B | fp4, int8 head | 6.00 | 22.206 | 12.9 | 94%, 2623 MB/token |
| Qwen3.5-2B | fp4, int8 head | 12.35 | 10.800 | 42.3 | 94%, 1282 MB/token |
| LFM2-2.6B | int8 | 6.13 | 21.764 | 21.7 | 96%, 2661 MB/token |
| LFM2-2.6B | **mix** `conv=fp4,mlp=fp4` | **10.84** | 12.298 | 20.5 | 95%, 1486 MB/token |
| LFM2-2.6B | fp4, int8 head | 11.15 | 11.958 | 20.6 | 94%, 1444 MB/token |

- **The tok/s model holds.** With the session's own int8 and fp4 points (Phi, SmolLM3, LFM2)
  or its fp4 point and the 2B's slope (Qwen3.5-4B) it predicts each mix within 0.1%: Phi
  25.643 M cycles (measured 25.640), SmolLM3 21.404 (21.409), LFM2 12.296 (12.298), the 4B
  23.403 (23.384). The estimates in the tables above came from build B's points, 1.5-2% slower
  (Phi 3.99 / 6.56, SmolLM3 5.00 / 8.74), so they are low by as much; the gains are the same.
- **Token-exact.** `tools/qual/refs.py card` (the ISA simulator's tokens for otpu-selftest's
  prompt, per-position and resident) passes for the 4B's and LFM2-2.6B's mixes, the uniform
  LFM2-2.6B (int8, fp4) and Qwen3.5-2B / 4B (fp4) images with per-kind block sizes, and
  Qwen3-0.6B with `mlp@7-20=fp4`. Phi-4-mini's and SmolLM3's mixes were checked through that
  proxy: the same Qwen3 image code places their two layouts in two or three runs, and their own
  ISA references need 17-26 GB of host memory. They ran perf only.
- **The decode loop on the card** (session mix3, tree chat-auto, `refs.py card --card-loop` with
  wformat "mix"): the 4B's and LFM2-2.6B's mixes and the Qwen3-0.6B proxy give the ISA
  simulator's tokens with every decode step in the card's generate loop.

**Session mix4** (production fmvf `deploy_fmvf_542fc43a`, 2026-10-02, tree g4-formats e4d6b73;
`tools/qual/perf.py` as above; the E2B runs after a discarded warm-up, in ABBA order, each
configuration's two runs within 0.01%):

| Model | Weights | Decode | Mcycles / token | Prefill | DRAM while decoding |
|:--|:--|--:|--:|--:|--:|
| Gemma 4 E2B | int8, the PLE table int8 on the host (`OTPU_PLE_HOST=1`) | 6.39 | 20.875 | 28.3 | 94%, 2496 MB/token |
| Gemma 4 E2B | int8 (the fp4 PLE table on the card) | 6.37 | 20.929 | 28.3 | 93%, 2497 MB/token |
| Gemma 4 E2B | `attn@15-34=fp4,mlp@15-34=fp4` (int8 table), the first pick | 8.47 | 15.751 | 29.1 | 92%, 1855 MB/token |
| Gemma 4 E2B | fp4, int8 head (int8 table) | 9.97 | 13.375 | 29.4 | 91%, 1558 MB/token |
| Phi-4-mini | **mix** `mlp@4-29=fp4` | **5.39** | 24.759 | 14.7 | 96%, 3030 MB/token |

- **E2B.** With the session's int8 and fp4 points (k 8.04) the tok/s model predicts the first
  pick at 15.764 M cycles (measured 15.751). Its estimate had come from build B's fp4 point
  (12.611 M cycles, 1475 MB a token); on this tree and build the same image reads 1558 MB at
  13.375, so every gain was about 3 points high: that mix is 32.5% faster than int8 with the
  host table (estimated 35.4%), which puts it 1.1 SE over its bar. The E2B section's speeds are
  this session's, and its default the next mix that qualifies. The host table costs nothing
  measurable in perf.py's per-token loop. `refs.py card` passes for the first pick, resident
  and with the decode loop on the card.
- **Phi-4-mini.** The new mix, 24.759 M cycles: the pa session's model gives 25.03 on pa, and
  fmvf decodes 1-4% faster.

**Session mix5** (fmvf `deploy_fmvf_542fc43a`, 2026-10-02, tree g4-formats 96ff6c8 on main
1f9e69e; perf.py after a discarded warm-up, the two runs identical): E2B's
`attn@15-24=fp4,mlp@15-34=fp4` decodes at **8.30** tok/s, 16.058 M cycles a token (the model:
16.068), 30.0% faster than int8 with the host table; perf.py's 512-token prefill 29.0 tok/s
(128 runs of 4 rows on today's route). `refs.py card --prompt-runs` passes, resident and with the
decode loop on the card (a 13-token prompt in four prompt runs). A 1500-token prompt at a
2048-token cap takes 100.3 s to the first token (warm) on today's route: 536 runs, 31 s of them
host compiles. The first pick would take prompt runs, an estimated 62 s. The cause is an IMEM
limit, not the mix's arithmetic: bucket 2's prompt run of 4 rows is 4106 instructions of 4096,
so prompts past 256 tokens leave prompt runs; prompts within bucket 1 are unaffected. A fix in
the prompt runs is in progress ([gemma4.md](gemma4.md), [prefill.md](prefill.md) 7).

**otpu-chat's default** (`--wformat auto`, session mix3; one prompt, 18-85 tokens in each
model's chat template, greedy, 64 tokens; device / wall tok/s): Phi-4-mini 5.3 / 5.26 (`mix:
int8 + mlp@4-27=fp4`), SmolLM3 6.3 / 6.26 (`gateup@9-35=fp4`), LFM2-2.6B 10.9 / 10.75
(`conv=fp4,mlp=fp4`), Qwen3.5-4B 5.7 / 5.67 (`delta=fp4,mlp=fp4`), and Qwen3.5-2B, which has no
mix, 8.2 / 8.09 in int8. The wall rate is within 1.5% of the device's.

**The 4B's mix under MTP** (`--mtp`, the chat prompt above, 160 tokens, greedy): 16.04 M cycles
per token against 23.27 without the drafter (1.45x, 8.3 device tok/s), acceptance 0.61; fp4 on
the same prompt 14.56 M cycles (1.53x over fp4's plain 22.21 from mix1, 9.2 device tok/s),
acceptance 0.67. A single-prompt observation: at 160 tokens the acceptance's SE is about 0.05,
so 0.61 against 0.67 is about 1 SE, and fp4's own 1.53x here against 1.62x in [mtp.md](mtp.md)
shows how much the prompt moves it. The MTP layer takes the mix's unranged formats (attention
int8, MLP fp4). The wall rates of these MTP runs overlapped another job on the card host and are
not quoted; the device cycles are the card's own count.
