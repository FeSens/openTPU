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
  layer layouts (Qwen3.5: one) and the LM head in int8. The fastest qualifying mix is chosen; a
  dKL gap under 2 SE is a tie, which goes to the mix with fewer runs. Finals are at 2000
  tokens.
- **Choices.** Phi-4-mini: `mlp@4-27=fp4`, 28% faster than int8 for dKL 2.5% (fp4: 64% for
  16%). Qwen3.5-4B: `delta=fp4,mlp=fp4` (the attention layers int8), 56% for 5.3% (fp4: 64% for
  6.5%, on its bar). Qwen3.5-2B: none qualifies. SmolLM3 and LFM2-2.6B: open.

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

- **Qwen3 and the Llama-likes** (SmolLM3, Phi-4-mini): a body is one layer, so two layouts in
  up to three runs fit.
- **Qwen3.5**: a body is four layers (three DeltaNet, one attention), about 1,900-2,300
  instructions, so a second run does not fit: one layout only.
- **LFM2**: the plan already has five runs (`lfm2.plan`); a second layout fits where its
  boundary falls on a run boundary, and adds runs elsewhere.

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
  and the quarter groups added in order of NLL lost per byte saved. `ppl MODEL FORMATS...
  --tokens 2000`: the given strings.

## The rule

Decode time per token is linear in the weight bytes b: T(b) = T4 + k (b - B4) cycles at 133.33
MHz, with T4 and B4 the card's fp4 (int8 head) decode and its bytes, and k from the card's
int8 and fp4 pair of the same model (Phi-4-mini 8.13 K cycles per MB) or family (Qwen3.5: the
2B's, 8.15; the 4B's int8 image does not fit the card). The gain is T(int8) / T(b) - 1.

A mix qualifies if dKL% <= 0.1 x gain%: 0.1% of perplexity for each 1% of speed. Among the
qualifying mixes with at most two layouts (Qwen3.5: one) the fastest wins; a dKL gap under 2
SE is a tie, and the tie goes to fewer runs. The LM head stays int8 in every mix.

## Choices

### Phi-4-mini

int8 3.99 tok/s, fp4 6.56 (card). At 900 tokens fp4 is dKL +16.1% (SE 1.3) and no whole kind
qualifies: gateup in every layer +3.27% for +24% (bar 2.4), attention +3.95% for +11%. The
down projections of the first and last quarters are costly (+7.4% and +5.0%), the middle
quarters cheap (+0.3% each). Finals, 2000 tokens:

| Formats | Layouts / runs | dKL % (SE) | Perplexity vs int8 % (SE) | MiB | tok/s (est.) | Gain | Bar |
|:--|:-:|--:|--:|--:|--:|--:|--:|
| `mlp@2-29=fp4` | 2 / 3 | +9.48 (0.59) | +8.76 (1.09) | 2764 | 5.37 | +34.6% | 3.46 |
| **`mlp@4-27=fp4`** | 2 / 3 | **+2.54 (0.15)** | +1.72 (0.68) | 2908 | 5.12 | +28.3% | 2.83 |
| `mlp@6-25=fp4` | 2 / 3 | +1.91 (0.13) | +1.60 (0.59) | 3052 | 4.89 | +22.5% | 2.25 |
| `gateup@0-27=fp4` | 2 / 2 | +1.71 (0.15) | +0.83 (0.64) | 3100 | 4.82 | +20.7% | 2.07 |
| `gateup@0-23=fp4` | 2 / 2 | +1.57 (0.15) | +0.61 (0.61) | 3196 | 4.68 | +17.2% | 1.72 |
| `mlp@8-23=fp4` | 2 / 3 | +1.53 (0.12) | +1.31 (0.52) | 3196 | 4.68 | +17.2% | 1.72 |
| `attn=fp4` | 1 / 1 | +3.93 (0.28) | +2.56 (0.77) | 3388 | 4.42 | +10.9% | 1.09 |

`mlp@4-27=fp4` is the fastest that qualifies (1.9 SE under its bar). Two more layers at each
end (2-3 and 28-29) add 6.9 points of dKL.

### Qwen3.5-4B

fp4 5.88 tok/s on the card; the int8 image does not fit (4237 MiB at a 2048-token capacity), so
its 3.58 tok/s is projected with the 2B's slope. At 900 tokens four single layouts qualified:
fp4 (+6.27%, bar 6.42), `delta=fp4,mlp=fp4`, `mlp=fp4` and `gateup=fp4`. The others miss their
bars (`attn=fp4,mlp=fp4` by 1.8 SE, `delta=fp4,gateup=fp4` by 1.3 SE, the rest by more) and are
slower than `delta=fp4,mlp=fp4` anyway. Finals, 2000 tokens:

| Formats | dKL % (SE) | Perplexity vs int8 % (SE) | MiB | tok/s (est.) | Gain | Bar |
|:--|--:|--:|--:|--:|--:|--:|
| fp4 (`attn=fp4,delta=fp4,mlp=fp4`) | +6.48 (0.20) | +5.96 (0.97) | 2433 | 5.88 | +64.2% | 6.42 |
| **`delta=fp4,mlp=fp4`** | **+5.30 (0.17)** | +4.42 (0.90) | 2573 | 5.59 | +55.9% | 5.59 |
| `mlp=fp4` | +3.01 (0.08) | +2.98 (0.69) | 3055 | 4.76 | +33.0% | 3.30 |
| `gateup=fp4` | +1.77 (0.05) | +2.31 (0.55) | 3415 | 4.29 | +19.8% | 1.98 |

fp4 sits on its bar (0.3 SE over; the bar itself rests on the projected int8 speed), so the
mix is `delta=fp4,mlp=fp4`, 1.7 SE under its bar: the eight attention layers stay int8, 5%
slower than fp4 for 1.2 points less dKL. Its image is 2675 MiB at a 2048-token capacity (2713
at 4096), one layout, generate 2099 / 2396 instructions.

### Qwen3.5-2B

int8 8.02 tok/s, fp4 12.09 (card). No single layout qualifies at 900 tokens: the closest,
`gateup=fp4`, is +2.06% (SE 0.07) for +17% (bar 1.74), 4.6 SE over; `down=fp4` +1.35% for
+8% (bar 0.79); fp4 +7.79% for +51% (bar 5.07). The 2B has no mix: int8 or fp4.

### SmolLM3-3B and LFM2-2.6B

Open: SmolLM3's dKL scan, and LFM2's two layouts on run boundaries (its card int8 speed is not
measured yet).
