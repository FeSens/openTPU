# The workload scoreboard

`tools/scoreboard.py` estimates what a hardware configuration gives on four workloads:

- single-user decode;
- prefill (TTFT);
- batched decode;
- MoE.

It uses an analytic model calibrated on the card and on the RTL co-sim. It is pure Python and
needs no checkpoint, RTL or card. `tools/scoreboard_cosim.py` runs the RTL co-sim and the
compiler for the points the model is calibrated on.

## Usage

```sh
# from the repository root; the tool imports nothing from opentpu/
python3 tools/scoreboard.py report                         # the YPCB in production
python3 tools/scoreboard.py report --board bcu1525 --mts 2400 --mhz 200
python3 tools/scoreboard.py report --board f2
python3 tools/scoreboard.py compare --mcols 8 --tmem-words 262144 --rows 16 --board bcu1525
python3 tools/scoreboard.py whatifs                        # the standard what-ifs on the YPCB
python3 tools/scoreboard.py whatifs --board bcu1525        # ... against another board
python3 tools/scoreboard.py validate                       # the model against the card and co-sim
python3 tools/scoreboard.py cosim pts.jsonl                # a co-sim campaign's layers against the model
python3 tools/scoreboard.py boards                         # the presets
python3 tools/scoreboard.py models                         # the model shapes and their overhead fits
python3 tools/scoreboard.py report --markdown --json out.json --models qwen3,phi4-mini
```

- **`report`**: prints the tables for one configuration.
- **`compare`**: prints the ratio of a configuration to its board preset for each workload.
  The overall score is the geometric mean of the four ratios.
- **`whatifs`**: runs the list below and prints one row of ratios for each what-if.

Each option overrides the board preset:

| option | what it changes |
|---|---|
| `--board ypcb\|bcu1525\|f2` | the preset (below) |
| `-S N` | slices, tensor-parallel: weights, KV heads and experts are split |
| `--mcols N` | MXU columns: `MCOLS x D` MACs a cycle and MCOLS rows a pass (int8; a PAIRed 4-bit row takes two) |
| `--act-rows N` | rows a weight pass (MM replay, "virtual MCOLS"): a run of R rows streams the weights `ceil(R / N)` times |
| `--mxu-pump X` | MACs a cycle x X at the same rows a pass (a multi-pumped MXU) |
| `--tmem-words N` | TMEM words: sets the prefill rows a run from the compiler's fit (`tools/scoreboard_rmax.json`) |
| `--rows N` | prefill rows a run, forced (tiling); `--imem 0` lifts the IMEM limit (looped attention) |
| `--batch-rows N` | rows in one batched-decode run (default: rows a pass) |
| `--vpu X` | vector-unit and DSTEP throughput x X |
| `--mhz F` | core clock |
| `--mem ddr3\|ddr4\|hbm --mts N --channels N --mem-eff E --gbps G` | a slice's memory |
| `--port-bytes N` | core port bytes a cycle a slice (128: two 64-byte channel beats) |
| `--pcie-gbps G` | host-link bandwidth for offloaded experts |
| `--fast-gb G` | memory a slice holds and addresses (4: a 32-bit address) |
| `--moe-rows all\|gather` | a MoE layer run computes its union of experts on all R rows, or each row only its own |
| `--wformat fp4\|int8\|auto` | the dense models' weights (fp4 with an int8 head; `auto` = the recommended mix) |
| `--ctx N` | the decode context (default 512) |

The API returns plain dicts, so other tools and the what-if branches can call it directly:

```python
import sys; sys.path.insert(0, "tools")
import scoreboard as SB
hw = SB.board("ypcb", mcols=8)                 # a preset with overrides (an HW dataclass)
SB.decode("qwen3", hw)                         # {"tok_s", "mcycles", "mb_token"}
SB.prefill("phi4-mini", hw, P=1024)            # {"ttft_s", "tok_s", "rows", "runs", ...}
SB.batch("qwen3", hw, users=8)                 # {"total_tok_s", "user_tok_s", ...}
SB.moe_decode("qwen35-35b", hw); SB.moe_prefill("gemma4-26b", hw, P=256)
SB.compare(SB.board("ypcb"), hw)               # {"decode", "prefill", "batch", "moe", "score"}
```

The co-sim driver needs a model's `config.json` only, because it runs zero DRAM images:

```sh
PYTHONPATH=. python3 tools/scoreboard_cosim.py run models/Qwen3-0.6B --layers 1:2 --rows 0,4 \
    --mcols 8 --lanes 16 --tmem-words 262144 --act-rows 16 --mem ldc --out pts.jsonl
PYTHONPATH=. python3 tools/scoreboard_cosim.py probe models/Qwen3-0.6B --tmem 65536,262144 \
    --act-rows 16 --buckets 1,4                # compile only: the rows a prefill run fits
```

Run the co-sim on the build host through `~/otpu-build/slot.sh`. A two-layer Qwen3 run needs about
4 GB and 10-40 minutes.

## 1. The workloads

All weights are fp4 with an int8 LM head (`--wformat`); the MoE models use their production
formats.

- **Decode**: one user's tokens a second at context 512. This is device time; the card's
  decode loop picks each token itself.
- **Prefill**: a 256- and a 1024-token prompt from position 0, in prompt runs of R rows.
  - A run stays inside its 256-token bucket.
  - R is what the TMEM and IMEM fit allow under the host's rule (section 2).
  - TTFT is the runs' device time plus 0.6 ms of host time a run.
- **Batched decode**: 1, 4, 8 and 16 users at context 512.
  - A step runs the users in blocks of one weight pass (MCOLS rows, or `--batch-rows`), one
    run a block, and picks every row's logits on the card.
  - The table gives the total and the per-user tokens a second.
  - Batched kernels exist today for Qwen3 and the Llama-likes (SmolLM3, Phi-4-mini); the
    table shows those.
- **MoE**: Gemma 4 26B-A4B and Qwen3.5-35B-A3B.
  - Expert slots fill the memory the slices address beside the rest of the image. The other
    experts come over PCIe from the host, or from a second card tier (F2's DDR4).
  - Decode is the card's time with the experts present, plus each layer's link time that the
    card does not hide.
  - Prefill is layer-major, R = 2, with pooled slots (docs/offload.md 13).

The summary line and `compare` reduce each workload to one number, a geometric mean over the
models:

- decode tok/s;
- prefill tok/s at 1024 tokens;
- batched total tok/s at 16 users;
- for MoE, sqrt(decode x prefill 1024) tok/s.

`score` is the geometric mean of the four ratios, so the four workloads weigh the same.

**What binds.** Every result carries each unit's share of its time:

- the weight stream, named `port` or `memory` after whichever is slower;
- `MXU`;
- `attention`, the rows' KV streams;
- `DSTEP`;
- `overhead`, meaning vector unit, control and runs;
- `host link` and `host`.

The unit with the largest share binds. `report` shows it for each model and workload.
`whatifs` and `compare` show, for each option, what binds each workload after the change.

## 2. The model

A run of R rows costs, in each layer and on each slice:

    cycles = max(memory / S, MXU / S) + attention + recurrence + overhead

- **memory** is the layer's DRAM bytes over the slice's stream rate. The rate is the lower of
  the port (`port_bytes` a cycle) and the memory (`GB/s x mem_eff / MHz`). The bytes are:
  - **Weights**: `ceil(R / rows a pass)` times their bytes. A weight takes 8.25 bits (int8)
    or 4.25 bits (fp4), block scales included.
  - **DeltaNet**: the state, read and written once a run.
  - **LFM2**: the conv window.
  - **MoE**: the routed experts, the union of the run's rows.
- **attention** is a serial phase between the layer's matmuls: q, k and v before it, o after.
  - Each row streams its KV cache: whole 256-token blocks, a sliding layer at most its window
    plus a block, at `2d + 4d/128 + 4` bytes a token and KV head.
  - It streams them once for each group of MCOLS query heads of a KV head, as
    `gemma4._attention` and `qwen3._attend_heads` part them.
  - The card's prompt runs from 1 to 2 blocks (A0, B0, B1) and its batched steps at contexts
    64 and 1024 fit this stream alone, with no latency term (`att_blk` 0). The tiling agent's
    co-sim breakdown agrees: 0.46K cycles a row, KV head and block.
  - The YPCB's DDR3 (2 x 1066 MT/s x 0.97: 124.2 B a cycle at 133.33 MHz) sits 3% below its
    port (128 B a cycle). The co-sim at the port's own rate decodes 4% faster.
- **MXU**:
  - int8: `params / D x ceil(R / MCOLS)` cycles.
  - 4-bit with PAIR: `params / 2D x ceil(2R / MCOLS)` cycles.
  - Both are divided by `--mxu-pump`.

  The MXU takes D weights a cycle (2D at 4 bits), which is the port's rate. A wider port
  therefore needs a wider MXU (section 6).
- **recurrence**: a DeltaNet layer's first row of DSTEPs hides behind the weight stream.
  Every further row adds `v_heads / S x (dk dv / (8 x vpu) + 200)` cycles of DSTEP, plus
  0.4 of the row's own DeltaNet VOPs. The VOP count comes from vpu-pump's inventory, about
  `1.75 conv_dim + 112 v_heads + 1.3 v_heads dv`: 15.7K a row on the 0.8B and 2B, 22.7K on
  the 4B. The gap between the stream and the MXU hides 0.24 of itself from the sum, up to
  half of the sum.
  - The campaign's layers hid at most 37%.
  - The cap binds on the card's Qwen3.5-4B at R 3 (MXU-bound) and on long runs on a deep
    ACT RAM. Neither is measured layer by layer.
  - The two constants are fitted together with the kappas on six DeltaNet layers of the
    campaign, at MCOLS 4 and 8, and on the card.
  - Without them, the model put the 2B's DeltaNet layer at MCOLS 8 at 0.665x its MCOLS 4
    time; the co-sim gives 0.734x and the model now gives 0.720x.
  - Decode (one row) is unchanged.
- **overhead**:

      (kappa_pass + kappa_row + kappa_row x (R - 1) x (0.8 / vpu + 0.2)) x 7000 x hidden / 1024 / S

  - kappa_pass is a run's cost and kappa_row a row's. Both are fitted for each model on its
    card points (`scoreboard.py fit`).
  - The cost a row is flat in R and MCOLS. The tiling agent's co-sim of compile_rows gives
    83.6K / 82.6K / 83.1K / 84.7K cycles a row at R = 4 / 8 / 16 / 32. The campaign's one
    pass of 8 rows (ACT RAM 8) costs what two passes of 4 do, 363.1K against 361.9K cycles a
    layer.
  - **Vector unit.** 80% of the rows after the first scales with it, and a decode row not at
    all. vpu-pump's co-sim halves every VOP's work:
    - decode: -0.35% (the model 0%);
    - Qwen3 prefill at R = 4: -3.5% (the model -3.1%);
    - Qwen3 batch of 4: -2.7% (the model -3.1%);
    - Qwen3 at MCOLS 8: -4.3% (the model -4.3%);
    - SmolLM3: -1.2% (the model -1.1%).

    The campaign's LANES 16 (Qwen3) gives -3.8% at R = 4 and +0.5% at decode.
  - **S > 1.** The vector work splits with the heads. The collectives add 1000 cycles a layer
    and row: at the port's rate the campaign's S = 2 runs a layer in 0.506x (decode) and
    0.534x (R = 4) of S = 1's time, against the model's 0.512x and 0.520x.

A run adds 5000 cycles. Logits add the LM head, which streams once a pass of logits rows, plus
2000 cycles a row for argmax or sampling. Gemma 4 adds its PLE projection.

**The tile kernel** (`--kernel tiled`) is the tiling agent's restructured prompt run
(branch tiled-prefill):

- MCOLS / G rows share each KV head's entry (packed attention), with its K and V products on
  the MXU.
- The next head's projections stream meanwhile, and gate/up pieces pipeline against SiLU.
- An attention layer costs `max(memory, linear MXU + attention MXU)`, plus
  `4.9K x R x hidden / 1024` cycles outside the MXU.
  - This is fitted on 14 of its co-sim points (TMEM 64K; MCOLS 4 and 8; T 4-16; int8 and
    fp4; positions 256 and 1024).
  - Mean error is 2.9%. The worst is MCOLS 8 at 1024 (-12.6%), where the model's attention
    is too cheap.
  - On the card it costs x 1.07 over that co-sim. That is the gap between the model of
    today's kernel, fitted on the card's prompt runs, and tiling's co-sim of compile_rows:
    4-11% at every R and MCOLS, mostly in the attention layers.
    - The card's prompt runs add RunWords, mask loads and per-row V^T appends.
    - The card's own compile_rows route (B0 at R = 8) also comes in 4.7% under the model.
- It applies to prompt runs only. Batched users do not share KV, and MoE layer runs are
  their own kernel. The host keeps today's kernel wherever that is faster.

**Prefill rows.** Prefill takes the rows a run fits from the compiler itself:
`tools/scoreboard_cosim.py probe` compiles each model at each TMEM size and ACT RAM depth
until TMEM, the ACT RAM or IMEM overflows, giving `tools/scoreboard_rmax.json`. Of the rows
that fit, the host's rule takes whole passes above one pass, and an odd R's R - 1 at 4 bits
(PAIR). Of those and of fewer whole passes, the model takes the R it finds fastest a row.

The probe agrees with tmem-pump's own bisection of the TMEM a run needs (Qwen3.5-4B: 54K,
70K and 140K words at R = 3, 4 and 8). The probe gives 3 rows at 64K and 7 at 128K, with IMEM
over 4096 instructions from R = 8.

**MoE:**

- **Misses a decode token**: the per-layer LRU over four 2048-token router traces
  (docs/offload.md 3) at the slots' share of the pool, x 0.88 (decayed use), x 1.2 (the
  card's chats miss more than the traces).
- **Exposure**: a layer's link time is exposed past what the card computes meanwhile, namely
  the dense MLP or shared expert and the hit experts. Each layer with misses adds 0.15 ms of
  host lead.
- **Prompt misses**: each layer's distinct experts over the prompt (`k R^0.6`, at most a
  skewed 75 of 128 (26B) or 145 of 256 (35B)) minus the slots' share, of which 85% is
  exposed.
- **Host time**: 0.2 ms a layer run. A DeltaNet model runs its first `conv_k - 1` rows as
  decode steps.

## 3. Validation

Against the 51 dense card points (decode, prefill, batched), the mean |error| is 0.7% and the largest 4.7%. `scoreboard.py validate` lists every point:

| points | mean \|error\| | max | leave-one-out mean | leave-one-out max |
|---|---:|---:|---:|---:|
| decode (21) | 0.3% | 1.6% | 0.4% | 2.2% |
| prefill (24) | 1.1% | 4.7% | 1.5% | 7.0% |
| batched (6) | 0.5% | 1.6% | 0.5% | 1.8% |
| co-sim (8) | 1.0% | 1.3% | 1.2% | 2.0% |
| co-sim 2-layer (19) | 7.6% | 11.1% | 7.6% | 11.1% |
| co-sim tile kernel (14) | 2.9% | 12.6% | 2.9% | 12.6% |
| co-sim layer (10) | 1.0% | 2.2% | 1.1% | 2.9% |
| co-sim layer (compile_rows) (4) | 5.6% | 7.6% | 5.6% | 7.6% |
| MoE layer run ms (6) | 2.9% | 5.4% | - | - |
| MoE (4) | 2.1% | 3.1% | - | - |
| co-sim campaign (30) | 4.9% | 10.9% | - | - |

- **Card points**: production builds at 133.33 MHz.
  - Decode: the decode counters (`tools/qual/perf.py`).
  - Prompt runs (docs/prefill.md 7): a first prompt (A0), a 224-291-token prompt (B0) and a
    next turn of 13 tokens at 224-322 (B1), for six models.
  - Batched: otpu-serve's steps.
  - MoE: the streamed MoEs' sessions (docs/offload.md 13.7, 14.4).
- **Co-sim points**: the RTL with LiteDRAM's DDR3-1066 controller.
  - `co-sim`: whole-model decode and prefill runs. It includes a Qwen3.5 prefill run at
    MCOLS 8 that the fit never saw.
  - `co-sim layer`: the per-layer slopes of docs/prefill.md 8. Phi-4-mini's at position 128
    (1 block), SmolLM3's at 3968 (16 blocks), Qwen3.5-4B's at 544.
  - `co-sim 2-layer` and `co-sim tile kernel`: the tiling agent's runs, with TMEM at 64K or
    otpu_seq's TAW fixed. One set is today's compile_rows at R 4-32, MCOLS 4 and 8, int8 and
    fp4; the other is the tile kernel. They are not in the kappa fit. The model of today's
    kernel sits 4-11% above them, the prompt runs' extra cost. The tile kernel's row
    constant is fitted on its own points.
  - `co-sim campaign`: this tool's own runs (`tools/scoreboard_cosim_runs.jsonl`, 2-1 layer
    slopes). The table below compares their what-if ratios.
- **Leave-one-out**: refits the point's model without that point. The largest is Qwen3's B0
  at R = 8 (7.0%). That point is the compile_rows route, which runs below prompt runs (see
  below). Next are LFM2's and Qwen3.5-0.8B's at 2.9%.
- **MoE**: its constants (host lead, exposure, run time) are fitted on the same four MoE
  points, so its 2-3% is in-sample. The zero-miss layer runs (2.9%) check its compute part
  independently.
- **Decode bytes a token**: within 2% of the card's counters, except Gemma 4 E2B. The model
  gives ~5-7% fewer bytes than its counters read (~100 MB a token), and the cause is
  unexplained.

**The campaign's what-if ratios** (section 8): each what-if's layer cycles over the same layer's on the YPCB, in the co-sim and in the model. The model's absolute layer cycles run 4-11% above compile_rows' co-sim in attention layers (see above); its ratios are what the what-ifs use:

| what-if (Qwen3-0.6B, a layer) | co-sim | model |
|---|---:|---:|
| LANES 16 (VPU x2), decode | 1.005x | 1.000x |
| LANES 16 (VPU x2), R 4 | 0.962x | 0.969x |
| one pass of 8 rows (ACT 8) / two of 4 | 1.003x | 1.000x |
| MCOLS 8, R 8 | 0.683x | 0.687x |
| MCOLS 8 batch of 8 / MCOLS 4 batch of 4, a row | 0.673x | 0.677x |
| memory at the port's rate, decode | 0.959x | 0.972x |
| memory at the port's rate, R 4 | 0.983x | 0.992x |
| S 2 / S 1 (port rate), decode | 0.506x | 0.512x |
| S 2 / S 1 (port rate), R 4 | 0.534x | 0.520x |

## 4. Baselines

| configuration | stream GB/s | decode tok/s | prefill 1024 tok/s | batched 16 users tok/s | MoE | score vs YPCB |
|---|---:|---:|---:|---:|---:|---:|
| YPCB (production) | 16.6 | 15.4 | 41.9 | 29.1 | 6.88 | 1.00x |
| BCU1525 DDR4-1600, 100 MHz | 25.6 | 23.5 | 61.8 | 43.1 | 12.18 | 1.56x |
| BCU1525 DDR4-1600, 200 MHz | 48.6 | 44.8 | 119.8 | 84.5 | 22.36 | 2.98x |
| BCU1525 DDR4-2400, 100 MHz | 25.6 | 23.5 | 61.8 | 43.1 | 12.18 | 1.56x |
| BCU1525 DDR4-2400, 200 MHz | 51.2 | 47.1 | 121.8 | 85.5 | 23.17 | 3.06x |
| BCU1525 DDR4-2400, 200 MHz, port 256 + MXU x2 | 73.0 | 65.6 | 186.1 | 144.6 | 29.66 | 4.49x |
| F2 (HBM, S 4, 250 MHz) | 128.0 | 116.4 | 286.6 | 205.1 | 52.11 | 7.25x |
| F2, port 256 + MXU x2 | 256.0 | 219.1 | 468.9 | 363.7 | 81.28 | 12.38x |
| F2, port 512 + MXU x4 | 414.0 | 330.8 | 663.4 | 575.8 | 104.52 | 17.88x |

The presets (`scoreboard.py boards`):

- **ypcb**: the production card. xc7k480t, 2 x DDR3-1066 through LiteDRAM, 133.33 MHz,
  MCOLS 4, TMEM 64K words, PCIe Gen2 x8 (2.2 GB/s for an expert miss).
- **bcu1525**: VU9P with S = 2 slices, each on 2 DDR4 channels. Each slice addresses 4 GiB of
  its 2 x 16 GB.
  - Bring-up is DDR4-1600 at 100 MHz, with mem_eff 0.95 assumed.
  - PCIe Gen3 x16 at 9 GB/s is assumed, not measured.
- **f2**: AWS F2, VU47P. 16 GB of HBM (~460 GB/s, taken as 115 GB/s a slice at 0.90), S = 4
  at 250 MHz.
  - Its 64 GB of DDR4 is a second tier for experts, assumed to be one 19.2 GB/s channel.
  - Its PCIe is taken as 20 GB/s.

A slice streams at most 128 B a core cycle through its port.
- **YPCB**: its DDR3 sits 3% under that.
- **BCU1525 and F2**: port-bound. On the BCU1525 at 100 MHz the port is half its DDR4, and
  on F2 at 250 MHz it is 128 of 460 GB/s.
- **DDR4-2400** buys the BCU1525 nothing until the port widens.

### YPCB (production)

**ypcb: S 1, MCOLS 4 (rows a pass 4, MXU x1), TMEM 64K words, VPU x1, 133.33 MHz, DDR3-1066.67 x 2 a slice: 16.6 GB/s, 124.2 B a cycle (memory-bound), 16.6 GB/s in all; host link 2.2 GB/s**

summary (geometric means, fp4 weights, context 512): decode 15.4 tok/s, prefill 1024 41.9 tok/s, batched 16 users 29.1 tok/s in all, MoE 6.88 (sqrt of decode x prefill 1024 tok/s)

bound (the unit taking the largest share of the time, averaged over the models): decode memory 93%, prefill MXU 52%, batch MXU 69%, moe memory 69%

| model | decode tok/s | MB/token | prefill 256 tok/s | TTFT 256 s | prefill 1024 tok/s | TTFT 1024 s | rows a run | decode bound | prefill 1024 bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 35.6 | 440 | 116.3 | 2.20 | 100.6 | 10.18 | 8 | memory 85% | MXU 65% |
| lfm2 | 97.5 | 165 | 348.9 | 0.73 | 311.6 | 3.29 | 8..4 | memory 92% | MXU 74% |
| qwen35 | 27.7 | 571 | 77.2 | 3.32 | 76.6 | 13.36 | 4 | memory 95% | MXU 56% |
| qwen35-2b | 12.4 | 1,301 | 41.9 | 6.11 | 41.8 | 24.51 | 4 | memory 97% | MXU 85% |
| qwen35-4b | 6.1 | 2,671 | 14.7 | 17.44 | 14.6 | 70.05 | 2 | memory 98% | memory 88% |
| smollm3 | 8.9 | 1,794 | 22.3 | 11.49 | 21.9 | 46.80 | 4 | memory 95% | MXU 90% |
| phi4-mini | 6.7 | 2,397 | 17.8 | 14.37 | 17.4 | 58.97 | 2 | memory 95% | memory 90% |
| lfm2-2.6b | 11.1 | 1,446 | 26.7 | 9.58 | 24.5 | 41.73 | 4..2 | memory 97% | memory 72% |
| gemma4 | 10.0 | 1,446 | 29.9 | 8.57 | 29.2 | 35.12 | 4 | memory 85% | MXU 80% |

| batched decode, total / each tok/s | 1 users | 4 users | 8 users | 16 users | 16 users bound |
|---|---:|---:|---:|---:|---:|
| qwen3 | 35.6 / 35.6 | 77.2 / 19.3 | 77.2 / 9.6 | 77.2 / 4.8 | MXU 50% |
| smollm3 | 8.9 / 8.9 | 20.0 / 5.0 | 20.0 / 2.5 | 20.0 / 1.2 | MXU 82% |
| phi4-mini | 6.7 / 6.7 | 16.0 / 4.0 | 16.0 / 2.0 | 16.0 / 1.0 | MXU 76% |

| MoE | experts on the card | decode tok/s | misses a token | prefill 256 tok/s | TTFT 256 s | prefill 1024 tok/s | TTFT 1024 s | decode bound | prefill 1024 bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemma4-26b | 14% (535 slots) | 3.56 | 87 | 8.3 | 30.9 | 8.6 | 118.8 | memory 64% | memory 78% |
| qwen35-35b | 16% (1615 slots) | 5.40 | 104 | 12.0 | 21.4 | 13.6 | 75.5 | memory 64% | memory 70% |

### BCU1525 DDR4-1600, 100 MHz

**bcu1525: S 2, MCOLS 4 (rows a pass 4, MXU x1), TMEM 64K words, VPU x1, 100 MHz, DDR4-1600 x 2 a slice: 24.3 GB/s, 128.0 B a cycle (port-bound), 25.6 GB/s in all; host link 9 GB/s**

summary (geometric means, fp4 weights, context 512): decode 23.5 tok/s, prefill 1024 61.8 tok/s, batched 16 users 43.1 tok/s in all, MoE 12.18 (sqrt of decode x prefill 1024 tok/s)

bound (the unit taking the largest share of the time, averaged over the models): decode port 92%, prefill MXU 51%, batch MXU 68%, moe port 79%

| model | decode tok/s | MB/token | prefill 256 tok/s | TTFT 256 s | prefill 1024 tok/s | TTFT 1024 s | rows a run | decode bound | prefill 1024 bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 54.0 | 440 | 166.0 | 1.54 | 145.1 | 7.06 | 8 | port 83% | MXU 62% |
| lfm2 | 146.7 | 165 | 482.0 | 0.53 | 431.2 | 2.37 | 8..4 | port 89% | MXU 68% |
| qwen35 | 42.3 | 571 | 112.2 | 2.28 | 111.4 | 9.19 | 4 | port 94% | MXU 55% |
| qwen35-2b | 19.0 | 1,301 | 61.8 | 4.14 | 61.7 | 16.59 | 4 | port 96% | MXU 83% |
| qwen35-4b | 9.4 | 2,671 | 22.3 | 11.48 | 22.2 | 46.12 | 2 | port 98% | port 87% |
| smollm3 | 13.6 | 1,794 | 33.0 | 7.76 | 32.4 | 31.60 | 4 | port 94% | MXU 89% |
| phi4-mini | 10.3 | 2,397 | 27.2 | 9.42 | 26.5 | 38.64 | 2 | port 95% | port 89% |
| lfm2-2.6b | 17.1 | 1,446 | 39.5 | 6.47 | 37.1 | 27.61 | 4..2 | port 96% | port 70% |
| gemma4 | 15.3 | 1,446 | 44.1 | 5.81 | 43.0 | 23.79 | 4 | port 84% | MXU 79% |

| batched decode, total / each tok/s | 1 users | 4 users | 8 users | 16 users | 16 users bound |
|---|---:|---:|---:|---:|---:|
| qwen3 | 54.0 / 54.0 | 112.6 / 28.1 | 112.6 / 14.1 | 112.6 / 7.0 | MXU 48% |
| smollm3 | 13.6 / 13.6 | 29.7 / 7.4 | 29.7 / 3.7 | 29.7 / 1.9 | MXU 81% |
| phi4-mini | 10.3 / 10.3 | 24.0 / 6.0 | 24.0 / 3.0 | 24.0 / 1.5 | MXU 75% |

| MoE | experts on the card | decode tok/s | misses a token | prefill 256 tok/s | TTFT 256 s | prefill 1024 tok/s | TTFT 1024 s | decode bound | prefill 1024 bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemma4-26b | 44% (1694 slots) | 7.30 | 16 | 13.0 | 19.6 | 12.9 | 79.1 | port 85% | port 76% |
| qwen35-35b | 39% (4005 slots) | 11.85 | 41 | 19.0 | 13.5 | 19.7 | 52.0 | port 91% | port 66% |

### BCU1525 DDR4-2400, 200 MHz

**bcu1525: S 2, MCOLS 4 (rows a pass 4, MXU x1), TMEM 64K words, VPU x1, 200 MHz, DDR4-2400 x 2 a slice: 36.5 GB/s, 128.0 B a cycle (port-bound), 51.2 GB/s in all; host link 9 GB/s**

summary (geometric means, fp4 weights, context 512): decode 47.1 tok/s, prefill 1024 121.8 tok/s, batched 16 users 85.5 tok/s in all, MoE 23.17 (sqrt of decode x prefill 1024 tok/s)

bound (the unit taking the largest share of the time, averaged over the models): decode port 92%, prefill MXU 50%, batch MXU 68%, moe port 76%

| model | decode tok/s | MB/token | prefill 256 tok/s | TTFT 256 s | prefill 1024 tok/s | TTFT 1024 s | rows a run | decode bound | prefill 1024 bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 108.1 | 440 | 328.0 | 0.78 | 287.1 | 3.57 | 8 | port 83% | MXU 62% |
| lfm2 | 293.4 | 165 | 930.4 | 0.28 | 816.3 | 1.25 | 8..4 | port 89% | MXU 65% |
| qwen35 | 84.6 | 571 | 220.8 | 1.16 | 219.2 | 4.67 | 4 | port 94% | MXU 54% |
| qwen35-2b | 38.0 | 1,301 | 122.6 | 2.09 | 122.3 | 8.37 | 4 | port 96% | MXU 82% |
| qwen35-4b | 18.8 | 2,671 | 44.3 | 5.78 | 44.1 | 23.21 | 2 | port 98% | port 86% |
| smollm3 | 27.3 | 1,794 | 65.6 | 3.90 | 64.5 | 15.88 | 4 | port 94% | MXU 89% |
| phi4-mini | 20.7 | 2,397 | 53.9 | 4.75 | 52.6 | 19.48 | 2 | port 95% | port 88% |
| lfm2-2.6b | 34.3 | 1,446 | 78.6 | 3.26 | 73.5 | 13.94 | 4..2 | port 96% | port 70% |
| gemma4 | 30.6 | 1,446 | 87.5 | 2.92 | 85.5 | 11.97 | 4 | port 84% | MXU 78% |

| batched decode, total / each tok/s | 1 users | 4 users | 8 users | 16 users | 16 users bound |
|---|---:|---:|---:|---:|---:|
| qwen3 | 108.1 / 108.1 | 221.4 / 55.4 | 221.4 / 27.7 | 221.4 / 13.8 | MXU 48% |
| smollm3 | 27.3 / 27.3 | 59.1 / 14.8 | 59.1 / 7.4 | 59.1 / 3.7 | MXU 81% |
| phi4-mini | 20.7 / 20.7 | 47.8 / 11.9 | 47.8 / 6.0 | 47.8 / 3.0 | MXU 75% |

| MoE | experts on the card | decode tok/s | misses a token | prefill 256 tok/s | TTFT 256 s | prefill 1024 tok/s | TTFT 1024 s | decode bound | prefill 1024 bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemma4-26b | 44% (1694 slots) | 14.60 | 16 | 24.6 | 10.4 | 24.8 | 41.3 | port 85% | port 73% |
| qwen35-35b | 39% (4005 slots) | 22.06 | 41 | 34.0 | 7.5 | 36.1 | 28.3 | port 84% | port 60% |

### F2 (HBM, S 4, 250 MHz)

**f2: S 4, MCOLS 4 (rows a pass 4, MXU x1), TMEM 64K words, VPU x1, 250 MHz, HBM 115 GB/s a slice: 103.5 GB/s, 128.0 B a cycle (port-bound), 128.0 GB/s in all; host link 20 GB/s**

summary (geometric means, fp4 weights, context 512): decode 116.4 tok/s, prefill 1024 286.6 tok/s, batched 16 users 205.1 tok/s in all, MoE 52.11 (sqrt of decode x prefill 1024 tok/s)

bound (the unit taking the largest share of the time, averaged over the models): decode port 91%, prefill MXU 47%, batch MXU 66%, moe port 69%

| model | decode tok/s | MB/token | prefill 256 tok/s | TTFT 256 s | prefill 1024 tok/s | TTFT 1024 s | rows a run | decode bound | prefill 1024 bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 265.2 | 440 | 756.5 | 0.34 | 668.7 | 1.53 | 8 | port 82% | MXU 58% |
| lfm2 | 711.7 | 165 | 1,983.4 | 0.13 | 1,669.5 | 0.61 | 8..4 | port 87% | MXU 53% |
| qwen35 | 208.8 | 571 | 512.2 | 0.50 | 508.9 | 2.01 | 4 | port 92% | MXU 50% |
| qwen35-2b | 94.4 | 1,301 | 293.8 | 0.87 | 293.2 | 3.49 | 4 | port 96% | MXU 79% |
| qwen35-4b | 46.8 | 2,671 | 107.7 | 2.38 | 107.3 | 9.54 | 2 | port 97% | port 84% |
| smollm3 | 67.8 | 1,794 | 159.8 | 1.60 | 157.1 | 6.52 | 4 | port 93% | MXU 86% |
| phi4-mini | 51.4 | 2,397 | 130.4 | 1.96 | 127.3 | 8.05 | 2 | port 94% | port 85% |
| lfm2-2.6b | 85.1 | 1,446 | 190.8 | 1.34 | 176.4 | 5.80 | 4..2 | port 95% | port 67% |
| gemma4 | 76.0 | 1,446 | 211.3 | 1.21 | 206.6 | 4.96 | 4 | port 84% | MXU 76% |

| batched decode, total / each tok/s | 1 users | 4 users | 8 users | 16 users | 16 users bound |
|---|---:|---:|---:|---:|---:|
| qwen3 | 265.2 / 265.2 | 510.5 / 127.6 | 510.5 / 63.8 | 510.5 / 31.9 | MXU 44% |
| smollm3 | 67.8 / 67.8 | 144.1 / 36.0 | 144.1 / 18.0 | 144.1 / 9.0 | MXU 79% |
| phi4-mini | 51.4 / 51.4 | 117.2 / 29.3 | 117.2 / 14.7 | 117.2 / 7.3 | MXU 74% |

| MoE | experts on the card | decode tok/s | misses a token | prefill 256 tok/s | TTFT 256 s | prefill 1024 tok/s | TTFT 1024 s | decode bound | prefill 1024 bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemma4-26b | all | 35.56 | 0 | 55.0 | 4.7 | 53.9 | 19.0 | port 83% | port 63% |
| qwen35-35b | 86% (8786 slots) + 1453 far | 55.27 | 4 | 68.8 | 3.7 | 69.6 | 14.7 | port 84% | port 47% |

### MoE on one slice (today's code) against S slices

| board | model | slices | experts on the card | decode tok/s | prefill 1024 tok/s |
|---|---|---:|---:|---:|---:|
| BCU1525 DDR4-2400, 200 MHz | gemma4-26b | 1 | 14% | 7.40 | 13.3 |
| BCU1525 DDR4-2400, 200 MHz | gemma4-26b | 2 | 44% | 14.60 | 24.8 |
| BCU1525 DDR4-2400, 200 MHz | qwen35-35b | 1 | 16% | 11.09 | 20.7 |
| BCU1525 DDR4-2400, 200 MHz | qwen35-35b | 2 | 39% | 22.06 | 36.1 |
| F2 | gemma4-26b | 1 | 14% + 3304 far | 8.75 | 16.5 |
| F2 | gemma4-26b | 4 | all | 35.56 | 53.9 |
| F2 | qwen35-35b | 1 | 16% + 8624 far | 14.48 | 25.6 |
| F2 | qwen35-35b | 4 | 86% + 1453 far | 55.27 | 69.6 |


## 5. What-ifs

`scoreboard.py whatifs [--board B]` gives each what-if's ratio to the board.

**Against ypcb** (S 1, MCOLS 4 (rows a pass 4, MXU x1), TMEM 64K words, VPU x1, 133.33 MHz, DDR3-1066.67 x 2 a slice: 16.6 GB/s, 124.2 B a cycle (memory-bound), 16.6 GB/s in all; host link 2.2 GB/s):

| what-if (against ypcb) | decode | prefill | batched | MoE | score | bound after: decode / prefill / batched / MoE |
|---|---:|---:|---:|---:|---:|---|
| (ypcb itself) | 1.000x | 1.000x | 1.000x | 1.000x | 1.000x | memory 93% / MXU 52% / MXU 69% / memory 69% |
| MCOLS 8 | 1.001x | 1.376x | 1.743x | 1.003x | 1.246x | memory 93% / memory 49% / MXU 62% / memory 69% |
| TMEM x2 (8 ACT RAM rows) | 1.000x | 1.038x | 1.016x | 1.000x | 1.013x | memory 93% / MXU 76% / MXU 84% / memory 69% |
| TMEM x4 (16 ACT RAM rows) | 1.000x | 1.041x | 1.020x | 1.000x | 1.015x | memory 93% / MXU 76% / MXU 84% / memory 69% |
| 64 rows a run, today's kernel | 1.000x | 1.053x | 1.020x | 1.000x | 1.018x | memory 93% / MXU 82% / MXU 84% / memory 69% |
| tile kernel, 16 rows a run (64K TMEM) | 1.000x | 1.032x | 1.000x | 1.000x | 1.008x | memory 93% / MXU 86% / MXU 69% / memory 69% |
| tile kernel, 16 rows, MCOLS 8 | 1.001x | 1.758x | 1.743x | 1.003x | 1.325x | memory 93% / MXU 76% / MXU 62% / memory 69% |
| 8 users in one pass (replay) | 1.000x | 1.002x | 1.016x | 1.000x | 1.004x | memory 93% / MXU 53% / MXU 84% / memory 69% |
| VPU x2 | 1.000x | 1.049x | 1.016x | 1.018x | 1.020x | memory 93% / MXU 55% / MXU 70% / memory 70% |
| MXU pumped x2 | 1.000x | 1.310x | 1.462x | 1.010x | 1.179x | memory 93% / memory 73% / memory 77% / memory 70% |
| MCOLS 8 + TMEM x4 + VPU x2 | 1.001x | 1.766x | 1.828x | 1.021x | 1.348x | memory 93% / MXU 65% / MXU 76% / memory 71% |
| MXU x2 + TMEM x4 + VPU x2 (pumped) | 1.000x | 1.784x | 1.828x | 1.021x | 1.351x | memory 93% / MXU 63% / MXU 76% / memory 71% |
| port 256 B a cycle (memory permitting) | 1.000x | 1.000x | 1.000x | 1.000x | 1.000x | memory 93% / MXU 52% / MXU 69% / memory 69% |
| port 256 B + MXU x2 | 1.000x | 1.310x | 1.462x | 1.010x | 1.179x | memory 93% / memory 73% / memory 77% / memory 70% |
| MoE runs of 8 rows (64-id requests) | 1.000x | 1.000x | 1.000x | 0.942x | 0.985x | memory 93% / MXU 52% / MXU 69% / MXU 42% |
| MoE runs of 8 rows, each row its experts | 1.000x | 1.000x | 1.000x | 1.227x | 1.053x | memory 93% / MXU 52% / MXU 69% / memory 51% |

**Against bcu1525** (S 2, MCOLS 4 (rows a pass 4, MXU x1), TMEM 64K words, VPU x1, 100 MHz, DDR4-1600 x 2 a slice: 24.3 GB/s, 128.0 B a cycle (port-bound), 25.6 GB/s in all; host link 9 GB/s):

| what-if (against bcu1525) | decode | prefill | batched | MoE | score | bound after: decode / prefill / batched / MoE |
|---|---:|---:|---:|---:|---:|---|
| (bcu1525 itself) | 1.000x | 1.000x | 1.000x | 1.000x | 1.000x | port 92% / MXU 51% / MXU 68% / port 79% |
| MCOLS 8 | 1.001x | 1.371x | 1.726x | 1.004x | 1.242x | port 92% / port 47% / MXU 60% / port 80% |
| TMEM x2 (8 ACT RAM rows) | 1.000x | 1.033x | 1.013x | 1.000x | 1.011x | port 92% / MXU 74% / MXU 83% / port 79% |
| TMEM x4 (16 ACT RAM rows) | 1.000x | 1.036x | 1.018x | 1.000x | 1.013x | port 92% / MXU 74% / MXU 83% / port 79% |
| 64 rows a run, today's kernel | 1.000x | 1.049x | 1.018x | 1.000x | 1.017x | port 92% / MXU 80% / MXU 83% / port 79% |
| tile kernel, 16 rows a run (64K TMEM) | 1.000x | 1.026x | 1.000x | 1.000x | 1.006x | port 92% / MXU 74% / MXU 68% / port 79% |
| tile kernel, 16 rows, MCOLS 8 | 1.001x | 1.715x | 1.726x | 1.004x | 1.313x | port 92% / MXU 73% / MXU 60% / port 80% |
| 8 users in one pass (replay) | 1.000x | 1.003x | 1.013x | 1.000x | 1.004x | port 92% / MXU 51% / MXU 83% / port 79% |
| VPU x2 | 1.000x | 1.047x | 1.016x | 1.017x | 1.020x | port 92% / MXU 54% / MXU 69% / port 81% |
| MXU pumped x2 | 1.000x | 1.314x | 1.479x | 1.011x | 1.184x | port 92% / port 70% / port 75% / port 80% |
| MCOLS 8 + TMEM x4 + VPU x2 | 1.001x | 1.734x | 1.805x | 1.021x | 1.337x | port 92% / MXU 62% / MXU 74% / port 81% |
| MXU x2 + TMEM x4 + VPU x2 (pumped) | 1.000x | 1.753x | 1.805x | 1.021x | 1.341x | port 92% / MXU 61% / MXU 74% / port 81% |
| port 256 B a cycle (memory permitting) | 1.078x | 1.062x | 1.052x | 1.077x | 1.067x | MXU 93% / MXU 81% / MXU 85% / MXU 78% |
| port 256 B + MXU x2 | 1.814x | 1.688x | 1.821x | 1.609x | 1.731x | memory 88% / MXU 43% / MXU 63% / memory 68% |
| MoE runs of 8 rows (64-id requests) | 1.000x | 1.000x | 1.000x | 0.945x | 0.986x | port 92% / MXU 51% / MXU 68% / port 44% |
| MoE runs of 8 rows, each row its experts | 1.000x | 1.000x | 1.000x | 1.235x | 1.054x | port 92% / MXU 51% / MXU 68% / port 62% |

**Against f2** (S 4, MCOLS 4 (rows a pass 4, MXU x1), TMEM 64K words, VPU x1, 250 MHz, HBM 115 GB/s a slice: 103.5 GB/s, 128.0 B a cycle (port-bound), 128.0 GB/s in all; host link 20 GB/s):

| what-if (against f2) | decode | prefill | batched | MoE | score | bound after: decode / prefill / batched / MoE |
|---|---:|---:|---:|---:|---:|---|
| (f2 itself) | 1.000x | 1.000x | 1.000x | 1.000x | 1.000x | port 91% / MXU 47% / MXU 66% / port 69% |
| MCOLS 8 | 1.001x | 1.352x | 1.718x | 1.003x | 1.236x | port 91% / port 45% / MXU 58% / port 69% |
| TMEM x2 (8 ACT RAM rows) | 1.000x | 1.052x | 1.029x | 1.000x | 1.020x | port 91% / MXU 70% / MXU 80% / port 69% |
| TMEM x4 (16 ACT RAM rows) | 1.000x | 1.059x | 1.042x | 1.000x | 1.025x | port 91% / MXU 71% / MXU 81% / port 69% |
| 64 rows a run, today's kernel | 1.000x | 1.101x | 1.042x | 1.000x | 1.035x | port 91% / MXU 78% / MXU 81% / port 69% |
| tile kernel, 16 rows a run (64K TMEM) | 1.000x | 1.052x | 1.000x | 1.000x | 1.013x | port 91% / MXU 80% / MXU 66% / port 69% |
| tile kernel, 16 rows, MCOLS 8 | 1.001x | 1.738x | 1.718x | 1.003x | 1.316x | port 91% / MXU 69% / MXU 58% / port 69% |
| 8 users in one pass (replay) | 1.000x | 1.009x | 1.029x | 1.000x | 1.009x | port 91% / MXU 47% / MXU 80% / port 69% |
| VPU x2 | 1.000x | 1.043x | 1.015x | 1.012x | 1.017x | port 91% / MXU 49% / MXU 66% / port 70% |
| MXU pumped x2 | 1.000x | 1.281x | 1.452x | 1.007x | 1.170x | port 91% / port 64% / port 70% / port 70% |
| MCOLS 8 + TMEM x4 + VPU x2 | 1.001x | 1.726x | 1.816x | 1.015x | 1.336x | port 91% / MXU 58% / MXU 72% / port 70% |
| MXU x2 + TMEM x4 + VPU x2 (pumped) | 1.000x | 1.748x | 1.816x | 1.015x | 1.340x | port 91% / MXU 57% / MXU 72% / port 70% |
| port 256 B a cycle (memory permitting) | 1.078x | 1.059x | 1.050x | 1.073x | 1.065x | MXU 92% / MXU 75% / MXU 81% / MXU 68% |
| port 256 B + MXU x2 | 1.882x | 1.636x | 1.773x | 1.560x | 1.709x | port 86% / MXU 38% / MXU 59% / port 56% |
| MoE runs of 8 rows (64-id requests) | 1.000x | 1.000x | 1.000x | 1.030x | 1.007x | port 91% / MXU 47% / MXU 66% / port 42% |
| MoE runs of 8 rows, each row its experts | 1.000x | 1.000x | 1.000x | 1.308x | 1.070x | port 91% / MXU 47% / MXU 66% / port 58% |


What each what-if sets:

- **MCOLS 8**: `--mcols 8`, which also means rows a pass 8.
- **TMEM x2 / x4**: `--tmem-words 131072 / 262144` with ACT RAM rows 8 / 16, giving the
  probe's rows a run.
- **8 users in one pass**: `--batch-rows 8 --act-rows 8`, with the MXU still at MCOLS 4.
- **VPU x2**: `--vpu 2`.
- **MXU pumped x2**: `--mxu-pump 2`.
- **Port 256 B**: `--port-bytes 256`, alone or with an MXU x2.
- **64 rows a run, today's kernel**: `--rows 64 --act-rows 64 --imem 0`.
- **Tile kernel**: `--kernel tiled --rows 16`, which fits 64K TMEM for Qwen3; also with
  MCOLS 8.
- **MoE runs of 8 rows**: `--moe-ids 64`, a request line of 64 expert ids, so R x 8 <= 64.
  It runs either with the run's union of experts on all 8 rows (today's layer run) or with
  `--moe-rows gather`, where each expert computes only the rows that picked it.

## 6. What the numbers say

- **The YPCB's MXU is matched to its port, and prefill and batching are MXU-bound.**
  - The MXU takes one D-weight block a cycle for each group of MCOLS rows (two at 4 bits),
    the rate the port streams them. Decode therefore streams at the port's rate.
  - Prefill and batched decode become MXU-bound from 2 rows (4-bit) or 4 rows (int8).
  - So more rows a run (TMEM x2 / x4, 64 rows of today's kernel, 8-user passes on MCOLS 4)
    gain only their per-run costs: 4-5% on prefill, 1-2% on batching. Tiling's co-sim shows
    the same: compile_rows' cost a row is flat from R = 4 to 32.
    - Most of the 64-row gain is in the models TMEM holds to R = 2 today: Qwen3.5-4B +13%,
      Phi-4-mini +12%, LFM2-2.6B +9%, Gemma 4 +8%.
    - Qwen3.5-4B's share assumes a long run hides up to half its DeltaNet chain under the
      MXU. The campaign measured that only to R = 4.
  - The tile kernel gains 3% on prefill at MCOLS 4, where today's kernel is already about 96%
    MXU-busy (tiling's co-sim). Its gain is on long prompts of Qwen3, LFM2 and Gemma 4,
    whose G = 2 packs 2 rows an entry. With MCOLS 8 it packs 4 and reaches 1.76x, against
    MCOLS 8's 1.38x alone.
  - More MACs a cycle gain directly: MCOLS 8 gives prefill 1.38x and batched 1.74x; an MXU
    pumped x2 gives 1.32x and 1.46x.
  - Decode moves with neither.
- **More memory bandwidth needs a wider weight path.**
  - A wider port alone gives at most ~8%, the 4-bit stream's scale bytes, because the MXU
    then bounds decode.
  - A 256-byte port with an MXU x2 gives the BCU1525 1.8x decode and 1.6-1.8x on the other
    workloads.
  - At DDR4-2400 and 200 MHz that is 4.5x the YPCB's score.
- **On a fast board the per-row costs start to show.**
  - F2 at 512 B a cycle with an MXU x4 streams 3.2x F2's bytes but decodes 2.9x faster.
  - The vector unit stays a small lever: VPU x2 is worth nothing on decode (vpu-pump's
    co-sim: the VPU is 0.6-0.8% of it) and about 5% on prefill.
  - For Qwen3.5, DSTEP is the bigger lever.
  - An MXU x4 needs 4 x 716 DSPs a slice (11.5K for S = 4) against the VU47P's 9,024, so it
    has to be pumped (2x clock) rather than replicated.
- **MoE.**
  - On the YPCB only 14-16% of the experts fit, so a decode token misses 87-104 experts over
    PCIe Gen2.
  - The exposed link takes 26-32% of decode. The rest is mostly the card streaming the
    experts it holds (64%).
  - The model's expert cost matches tiling's expert co-sim: 0.101 ms (35B) and 0.210 ms (26B)
    at M = 1-2, twice that at M = 4 on MCOLS 4. The expert is stream-bound until its rows
    double the MXU's PAIR pass.
  - The BCU1525's two 4 GiB slices hold 39-44% and hide the link entirely. F2 holds the whole
    26B and 86% of the 35B, with the rest in its DDR4.
  - MoE prefill is bound by the port re-streaming each run's union of experts. A request
    line of 16 ids caps a layer run at 2 rows (k = 8), so a 1024-token prompt streams about
    12 experts a run, 512 runs a layer.
  - Wider requests (64 ids, 8 rows) only pay if each expert computes just the rows that
    picked it (`--moe-rows gather`): MoE 1.23-1.32x. With today's run, which computes the
    union on all rows, they lose (0.93-1.06x).
  - The MCOLS, TMEM and VPU what-ifs barely move MoE (1.00-1.06x). Memory, the host link, S
    and the request width do.

## 7. Caveats and decisions for the user

- **4 GiB a slice.** Addresses are 32 bits, so a slice holds 4 GiB of model (`--fast-gb`).
  The BCU1525's 2 x 16 GB a slice and F2's DDR4 tier need address windows or far-memory
  DMA, which is an ISA decision.
- **Gemma 4 and the MoE models run on one slice in today's code.** Their S > 1 rows assume
  tensor-parallel slices like the dense models. Section 4 gives MoE at S = 1 for comparison.
- **TMEM above 64K words** needs two RTL fixes before its what-ifs hold.
  - STREAM descriptors hold a 16-bit address, so the STREAM area must stay below word 65536
    (from tmem-pump).
  - `rtl/seq/otpu_seq.sv` tracks TMEM ranges in 16 bits (TAW = 16). Every access at or above
    word 65536 conflicts with every instruction and serializes the pipeline (from tiling).
    The tile kernel at T = 16 takes 104.8K cycles a row at TMEM 256K against 76.4K at 64K.
  - The model assumes both are fixed. Every co-sim above 64K words is pessimistic until TAW
    follows TMEM_WORDS, and the campaign dropped its TMEM 256K arm for this reason.
- **MoE layer runs are capped at R = 2** by the request line (R x k <= 16 expert ids). A
  wider line is `--moe-ids` (section 5).
- **A deep ACT RAM** (`--act-rows` above MCOLS, MM replay) costs nothing by itself.
  - At TMEM 64K an ACT RAM of 32 rows runs R = 4 exactly as 4 rows do (83.6K a row,
    tiling).
  - One pass of 8 rows costs what two passes of 4 do (the campaign).
  - It saves weight streams, which matters only where the stream binds (int8, wide MCOLS).
- **compile_rows against prompt runs**: the model is fitted on the card's prompt runs and
  decode. compile_rows' co-sim, today's route for some prompts, runs 4-11% below it in the
  attention layers and within 2% in the conv and DeltaNet layers.
- **The tile kernel at T = 16** fits 64K TMEM for Qwen3. For the larger models it is
  assumed, not probed.
- **The vector unit's share** (0.8 of the rows after the first) is fitted on Qwen3 and
  SmolLM3.
  - The conclusion does not hang on it: across shares from 0.25 to 1.0, VPU x2 is worth
    1.3-2.1% of the YPCB's score.
  - **For Qwen3.5 at MCOLS 8 the model still takes too little of it.** It gets the
    DeltaNet layer's total within 0.4-7% at MCOLS 4 and 8, and the DSTEP share right.
    vpu-pump's co-sim of the 2B at R 4 against the model, both without the LM head:

    | change | MCOLS 4: co-sim / model | MCOLS 8: co-sim / model |
    |---|---|---|
    | DSTEP free | -9.0% / -10.0% | -24.3% / -22.1% |
    | DSTEP at half time | -5.8% / -5.9% | |
    | VOPs halved | -1.7% / -1.3% | -7.1% / -1.9% |
    | VOPs and DSTEP halved | | -19.3% / -11.9% |

    The 2B's other VOPs (MLP SiLU, norms: 9.2K a row) sit in a kappa fitted at MCOLS 4,
    where the MXU hides them. So with MCOLS 8, the model understates what a faster VPU
    gives Qwen3.5 by about 5 points.
  - **LANES 16 has no stream engine.** The RTL builds DSTEP and STREAM at 8 lanes only
    (`otpu_slice.sv`: HAS_SS = HAS_DSTEP && LANES == 8, vpu-pump). The campaign's
    Qwen3.5 run at 16 lanes kept DSTEP on, so the slice stopped at its first DSTEP
    (error 1 after 2,965 cycles). That is an unsupported configuration, not an RTL bug.
    - `scoreboard_cosim.py` now builds 16 lanes without DSTEP and STREAM, so DeltaNet runs
      on the VOP fallback. A 16-lane VPU that keeps the engine needs otpu_se_tail and the
      RDOT partial loop rebuilt for 16 lanes.
    - The compiler accepts `board_config(LANES=16, DSTEP=True)` without complaint, a
      usability gap.
    - The model's `--vpu 2` is a VPU and stream engine of twice the rate (8 lanes pumped),
      not today's 16-lane VPU.
- **The S > 1 collective cost** (1000 cycles a layer and row) comes from one co-sim at S = 2.
  Between 0 and 6000 cycles it moves F2's decode by -7% to +2% and its prefill by -16% to
  +4%.
- **Context-dependent vector work** (softmax over the context) enters only through the KV
  bytes. That fits the card's batched steps at contexts 64 and 1024 to 0.4%.
- **Assumed, not measured**: the BCU1525's PCIe and DDR4 efficiency, F2's DDR4 tier and PCIe,
  and the S > 1 collective cost.
- **Batched decode** exists for Qwen3 and the Llama-likes only. Gemma 4, LFM2 and Qwen3.5
  batching is modelled the same way but has no kernel yet.

## 8. Calibration data and the co-sim campaign

The card and co-sim points are in `tools/scoreboard.py` (`CARD_DECODE`, `CARD_PREFILL`,
`CARD_BATCH`, `COSIM`, `CARD_MOE`, `MOE_LAYER_RUNS`). After adding a point:

1. Run `scoreboard.py fit` to refit the kappas.
2. Copy them into `KAPPA`.
3. Run `scoreboard.py validate`.

The model shapes come from `scoreboard.py gen-models DIR...`, which reads the config.json
and the safetensors headers, into `tools/scoreboard_models.json`. Refresh the rows a run fits
with `tools/scoreboard_cosim.py probe`.

The co-sim campaign (`tools/scoreboard_campaign.sh PART`, on the simulation host as four slot jobs) runs a model's first one
and two layers. The difference between them is a layer's cycles, with the run's own cost
removed. In order, it covers:

1. One pass of 8 rows (ACT RAM 8) against two passes of 4, both at TMEM 64K.
2. LANES 16 (decode and R 4, Qwen3; Qwen3.5's run had DSTEP on and stopped, see section 7).
3. MCOLS 8 (prefill R 8, an 8-row int8 batch).
4. S = 2 and S = 1 at the port's memory rate.
5. The rest of the YPCB baseline.
6. Qwen3.5's DeltaNet layers at MCOLS 4 and 8: the 0.8B and 2B at R 4, and the 4B at R 2
   and R 3 (parts 3 and 4).

It ran on the simulation host on 2026-10-12: parts 1 and 2 took about 25 minutes each, parts 3 and 4
about 20. Its results are in
`tools/scoreboard_cosim_runs.jsonl`, which `validate` and `scoreboard.py cosim FILE` read.
