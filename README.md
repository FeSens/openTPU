# openTPU

openTPU is a small inference accelerator for LLM decode, written to be read and changed. It
covers the whole stack: SystemVerilog RTL, an instruction set and a bit-exact simulator for
it, a small kernel language and compiler, and host software for a Kintex-7 PCIe card. The goal
is to learn about hardware, software and ISA performance work by changing one layer and
measuring what happens to the cycles.

It runs on real hardware: an Inspur YPCB-00338 PCIe card (Xilinx Kintex-7 xc7k480t, two DDR3
SODIMM channels) in a Linux PC, decoding Qwen3-0.6B, LFM2.5-230M and Qwen3.5-0.8B with their
real weights, token for token identical to the bit-exact simulator.

![otpu-chat on LFM2.5-230M with otpu-smi watching the card](docs/img/card-chat-smi.gif)

*`otpu-chat` (left) running LFM2.5-230M on the card, and `otpu-smi` (right) watching it. While
the reply decodes the card is running 84% of the time and reads 8.15 GB/s from DRAM; the status
line ends at 35.4 tok/s on the wall clock, 40.5 tok/s on the device. Screen recording,
2026-09-27, production image `b2c7ce43`. (The prompt's typo "ROme" is answered as a company by
this 230M-parameter model; the accelerator reproduces the model, mistakes included.)*

## Measured on the card

Production image `deploy_prod1066_b2c7ce43` (2026-09-27): core clock 100 MHz, DDR3-1066 on both
channels, PCIe Gen1 x8, int8 weights, 2 MXU columns, 8 vector lanes. Greedy decode at a short
context, measured with `tools/decode_profile.py`:

| Model | Mcycles / token | Device tok/s | Wall tok/s | DRAM read per token | DRAM bandwidth used (derived) |
|---|---|---|---|---|---|
| LFM2.5-230M | 2.46 | 40.7 | 38.5 | ~234 MB | ~9.5 GB/s |
| Qwen3-0.6B | 6.85 | 14.6 | 14.1 | ~600 MB | ~8.8 GB/s |
| Qwen3.5-0.8B | 9.75 | 10.3 | 9.9 | ~819 MB | ~8.4 GB/s |

"Device" counts only the cycles the accelerator runs; "wall" adds the host (the next token's
program compiles in a worker process while the card runs, so the host costs 1.3 to 2.7 ms per
token, mostly reading the logits and sampling). The bandwidth column is bytes per token times
device tok/s. DDR3-1066's peak is 17.1 GB/s over both channels, and the core takes at most one
128-byte chunk per cycle (12.8 GB/s at 100 MHz), so decode runs at about half the DRAM's peak
and 65-74% of what the core can take in: that gap is what the current work goes after.

On the same image, the three models match the ISA simulator token for token, `otpu-selftest`
passes, and `otpu-diag --mem full --soak 20` passes all 127 checks. Prefill runs the prompt in
chunks of rows that share each weight pass (bit-identical to token-by-token decode); the
recording above shows a short LFM2 prompt at TTFT 0.26 s and 54.3 prompt tok/s on the wall
clock (one observation, not a benchmark).

The build: Vivado 2026.1, all timing constraints met at 100 MHz (WNS +0.085 ns). 163K LUT
(54.6%), 147K FF (24.6%), 558 BRAM36 (58.4%), 283 DSP48 (14.7%) of the xc7k480t, including the
XDMA and two MIG DDR3 controllers. Vivado's power estimate is 9.0 W (low confidence; the card
has no power monitor). Bring-up, clocking and the DDR3 speed qualification are in
[docs/board.md](docs/board.md); the host side in [docs/host.md](docs/host.md).

Not done yet: the image is loaded over JTAG (the flash still holds an older image, so a power
cycle loads that); DDR3 ECC error counters are not read; clocks above 100 MHz and 4-bit weights
are being built and tested but are not in the production image, and no number above depends
on them.

## How it's built

```
 kernels (opentpu/kernels)          mlp, attention, a full Qwen3 layer      <- written in `ol`
        |  @ol.jit, traced once per slice (SPMD)
 language + compiler                opentpu/language.py, opentpu/compiler.py
        |  layouts, affine loop addressing, fusion peepholes, bank-aware strides
 ISA (docs/isa.md)                  opentpu/isa.py         8 x 32-bit words per instruction
        |                                       \
 bit-exact ISA simulator            RTL (SystemVerilog, rtl/) under Verilator
 opentpu/isasim.py      <== same DRAM + TMEM bits ==>   rtl/top/otpu_top.sv
        |                                                       |
 host runtime + CLI (opentpu/host)  ---- PCIe (XDMA) ---->  board: rtl/boards/ypcb-00338
```

Each layer is tested against the one below it. Kernels are checked against float64 numpy
references. The simulator's fp32 arithmetic is checked against a Python model of the RTL's
fp units. The RTL has to produce the same DRAM and TMEM bits as the simulator.

The machine is deliberately simple: a matrix unit that streams weights from DRAM, a vector
unit, a quantizer, a DMA engine and a collective unit for multi-slice runs. The on-chip
memories are explicit, and there are hardware loops. Every data movement is an instruction
you can see in the trace, so a profile can usually tell you why something is slow.

```
            host: program images, DRAM images in, DRAM out
                                  |
 +--------------------------------v-------------------------------------+  x S slices
 | SEQ  1 instr/cycle into a 16-slot scoreboard; 16 regs, LOOP, addr regs|
 |   +--> DMA (LD/ST)      DRAM burst port                              |
 |   +--> MXU (MM)         1 streamed D-byte int8 row/cycle from DRAM   |
 |   |      x M <= 8 stationary rows from ACT RAM, block scales, fp32   |
 |   +--> QUANT (QACT/QST) TMEM fp32 -> ACT RAM int8 | DRAM int8 (KV)   |
 |   +--> VPU (VOP)        fp32: add mul max, exp2 recip rsqrt          |
 |   +--> COLL (GATHER/BAR) ---------------- shared by all slices ------+--> other slices
 |  TMEM: LANES banks, arbitrated per cycle                             |
 |  ACT RAM: int8 activations + scales; DRAM: private per slice         |
 +----------------------------------------------------------------------+
```

- **Numerics.** Weights, MXU activations and the KV cache are int8 with one fp32 scale per D
  elements. Weights can also be 4-bit (FP4 or int4 elements with a two-level scale per D
  elements), about half the DRAM bytes per token; see [docs/quant.md](docs/quant.md). Everything else is fp32 with round-to-nearest-even and flush-to-zero. exp2, recip
  and rsqrt are fixed sequences of adds and multiplies, so Python, the simulator and the RTL
  agree bit for bit (they do not agree bit for bit with PyTorch).
- **Concurrency.** The sequencer issues one instruction per cycle into a 16-entry window and
  tracks what each instruction reads and writes. An instruction starts when nothing older
  conflicts with it, so the units overlap without the compiler scheduling them.
- **Attention** is flash attention with an online softmax, software-pipelined so q·Kᵀ for the
  next block streams while the softmax of the current one runs. **MLP** streams gate and up
  for the next chunk while the VPU computes SiLU.

Details: [design spec](docs/superpowers/specs/2026-09-23-opentpu-design.md),
[docs/isa.md](docs/isa.md), [docs/compiler.md](docs/compiler.md).

## Writing a kernel

```python
from opentpu import language as ol

@ol.jit
def mlp(h, gamma, w_gate, w_up, w_down, out, eps):   # simplified; see kernels/mlp.py
    x = ol.load(h)
    xs = ol.quantize(rmsnorm(x, ol.load(gamma), eps))   # ACT RAM, reused by both matmuls
    g = ol.dot(xs, w_gate)                              # this slice's columns
    u = ol.dot(xs, w_up)
    a = ol.all_gather(silu(g) * u)
    y = ol.all_gather(ol.dot(a, w_down))
    if ol.program_id() == 0:
        ol.store(out, x + y)
```

```python
from opentpu import Config
from opentpu.runtime import Input, Output, Weight, launch

res = launch(mlp, Config(S=2), backend="rtl",            # or "isa"
             h=Input(h), gamma=Input(g), w_gate=Weight(Wg, 0), w_up=Weight(Wu, 0),
             w_down=Weight(Wd, 0), out=Output((M, H)), eps=1e-6)
```

## Simulated kernel numbers

The card numbers are above. These come from Verilator RTL simulation of the board configuration (1 slice, 128-deep MXU with 2
columns, 8 vector lanes, 16-entry window) on Qwen3-0.6B's shapes (hidden 1024, MLP 3072, 16
query heads, 8 KV heads of 128). "Of roofline" is the kernel's cycles compared with the cycles
needed just to move its bytes over the simulated DRAM port, which is idealized; it says
nothing about real DDR3 behaviour.

| Workload | Cycles | Of roofline |
|---|---|---|
| MLP decode | 75,282 | 98.4% |
| Flash attention, context 1024 | 27,835 | 64.4% |
| Flash attention, context 4096 | 104,571 | 67.1% |
| Full attention layer (norm, QKV, RoPE, KV append, attention, output), position 1023 | 81,362 | 82.3% |

The MLP is limited by DRAM. Attention is not: the matrix unit and the vector unit are both
busy almost every cycle (100% and 99% at context 1024) while DRAM streams only 64% of the
time. The matrix unit spends that time on per-row work between streams, and the vector unit
on the softmax. That is the obvious next thing to improve.

Per-model breakdowns (which phase of the token spends the cycles, and against which roofline)
are in [docs/lfm2.md](docs/lfm2.md) and [docs/qwen35.md](docs/qwen35.md). Qwen3.5's DeltaNet
recurrence runs on the vector unit and does not keep up with DRAM; it is the largest single
gap left in that model.

## Accuracy

The kernels are checked against float64 numpy references, and the full model against Hugging
Face's fp32 Qwen3-0.6B. Greedy continuations of 16 tokens from eight short raw prompts, ISA
simulator vs Hugging Face (`python3 tools/compare_hf.py --emulate`):

| Prompt | Same for 16 tokens? | First different token | HF's rank of the device's token | HF logit gap |
|---|---|---|---|---|
| `A prime number larger than 100 is` | no | 2 | 2 | 0.15 |
| `The capital of France is` | no | 6 | 2 | 0.02 |
| `def fibonacci(n):` | yes | | | |
| `Water boils at` | no | 6 | 3 | 0.74 |
| `The quick brown fox` | no | 11 | 2 | 0.40 |
| `In 1969, the first person to walk on the moon was` | no | 3 | 2 | 0.49 |
| `The largest planet in the solar system is` | yes | | | |
| `import numpy as np` | no | 2 | 2 | 0.08 |

2 of 8 match exactly; the others drift apart after 1 to 10 tokens. That is expected: the
model runs in W8A8, meaning weights (W) and the activations fed to the matrix unit (A) are
stored as 8-bit integers instead of 32-bit floats. The rounding shifts the scores slightly, so
when two candidate tokens are nearly tied the device can pick the other one. A float64 model
with the same 8-bit rounding picks the same tokens as the device where we checked, which shows the differences
come from the quantization and not from a bug.

## Lens

![Lens replaying a Qwen3 decode step on the floorplan, 100 cycles per second](docs/img/lens-floorplan.gif)

*Lens replaying part of a Qwen3 decode step from an RTL simulation of the board configuration,
at 100 cycles per second. The dashes are data moving. The colours show what each unit is doing
in that cycle: busy, stalled on DRAM, lost TMEM arbitration, or waiting on a dependency.*

Lens is the profiler. It records a run (an RTL cycle trace, a simulator run, or, on the card,
the hardware trace buffer) and opens it in the browser: an overview with the roofline and where
the DRAM cycles went, a zoomable timeline, the floorplan replay, and per-instruction and
per-source-line tables.

```
python3 -m opentpu.lens record mlp attn -o run.otpuprof
python3 -m opentpu.lens open run.otpuprof
```

On the card (`otpu-lens`) and on the board model, the hardware trace buffer rebuilds the same trace lines the simulator
prints, and a test checks that they match. See [docs/lens.md](docs/lens.md) and
[docs/observability.md](docs/observability.md).

## The auto-arch tournament

Part of the RTL was tuned by an automated hill climb, `tools/tourney`, in the style of
[auto-arch-tournament](https://github.com/FeSens/auto-arch-tournament). It works on one
component at a time. Each round:

1. A few LLM agents read the current version of the component, its critical path and a log
   of what earlier rounds tried.
2. Each writes a hypothesis ("register the TMEM read data in front of the prescale
   multiplier"), and another agent implements it in its own worktree.
3. Each candidate must pass lint, the bit-exact RTL-vs-simulator tests on two
   micro-architectures, a Qwen3 decode cycle-count check and the kernel performance tests.
4. It is synthesized with yosys and kept only if it is smaller at the same estimated speed,
   or faster at the same size.

The first overnight run tried 96 changes across six components and kept 38. With some manual
fixes between units, the yosys estimate for the accelerator logic went from 41 MHz to 106 MHz
and from 132K to 82K LUT. These are yosys estimates for the accelerator and its control logic
only. Vivado then closed the whole board at 100 MHz on its first builds (see
[Measured on the card](#measured-on-the-card)). The logs and patches
are in `tools/tourney/runs/`, and [docs/tourney.md](docs/tourney.md) explains how to run it.

```
make tourney COMP=otpu_vpu N=5 K=2     # 5 rounds, 2 candidates per round
make tourney-report COMP=otpu_vpu      # REPORT.md and a progress plot
```

## The board

The card is a YPCB-00338 with an xc7k480t, two DDR3 SODIMM channels through MIG, and PCIe
through XDMA. [docs/board.md](docs/board.md) has the build (`make bit` in
`boards/ypcb-00338`), JTAG loading and bring-up steps.

The host software runs on top of the stock Xilinx XDMA driver (`sudo otpu-setup` installs it
and rescans the bus after a JTAG load). The tools that run programs take a lock on the card, so a
second one waits (`OTPU_LOCK_WAIT`) or names the holder; `otpu-smi` only reads counters and
runs alongside them:

```
otpu-selftest            registers, DRAM patterns, kernels, then a model (--sim for the board model)
otpu-diag                every hardware check without stopping: a works / does-not-work matrix
otpu-chat                chat with Qwen3-0.6B (or --model lfm2: LFM2.5-230M, qwen35: Qwen3.5-0.8B)
otpu-smi                 temperature, estimated power, DRAM use and bandwidth, per-unit utilization and stalls
otpu-lens                record a hardware trace and open it in Lens
```

Temperature comes from the FPGA's XADC and the board's LM73 sensor over I2C. Power is only an estimate: Vivado's per-unit power
report scaled by the utilization counters.

## Running the tests

You need Python 3.11+, numpy and pytest. The RTL tests also need Verilator 5 and skip
themselves without it. The Qwen3, LFM2 and Qwen3.5 tests need `torch` and `transformers`;
their real-model tests need the checkpoints in `models/Qwen3-0.6B`, `models/LFM2.5-230M` and
`models/Qwen3.5-0.8B`.

```
python3 -m pytest -q
```

| Suite | What it checks |
|---|---|
| `test_fp.py` | RTL fp units vs the Python fp32 model on 245K vectors |
| `test_isa.py` | Instruction semantics, loops, collectives, hazard detection |
| `test_compiler.py` | Layouts, loop addressing, broadcasts, fusion safety |
| `test_kernels.py` | MLP and attention on the simulator vs float64 references |
| `test_rtl.py` | Identical DRAM and TMEM on the RTL and the simulator, kernels and random programs |
| `test_perf.py` | Lower bounds on kernel efficiency on the RTL (MLP > 94.5%, attention > 60%) |
| `test_board.py`, `test_host.py`, `test_observability.py` | The board model through the host driver |
| `test_qwen3.py` | Qwen3 vs Hugging Face (tiny random model; one prompt on the real one) and one real token on the RTL |
| `test_lfm2.py` | The same for LFM2, plus a tiny LFM2 on the board model |
| `test_qwen35.py` | The same for Qwen3.5 (DeltaNet and gated attention), plus a tiny Qwen3.5 on the board model |

## Known simplifications

- QST writes one byte per cycle.
- The MXU dot product is behavioural in simulation; on the FPGA it maps to DSP48 cascades.
- MAX and MIN on NaN inputs are undefined.

## License

Apache License 2.0. See [LICENSE](LICENSE).
