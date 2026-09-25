# openTPU

openTPU is a small inference accelerator for LLM decode, written to be read and changed. It
covers the whole stack: SystemVerilog RTL, an instruction set and a bit-exact simulator for
it, a small kernel language and compiler, and host software for a Kintex-7 PCIe card. The goal
is to learn about hardware, software and ISA performance work by changing one layer and
measuring what happens to the cycles.

It is a prototype. It runs in simulation. It has not run on the FPGA yet: the Vivado build
(synthesis, place and route, timing) has not been done, and nothing below has been checked on
hardware.

![Lens replaying a Qwen3 decode step on the floorplan, 100 cycles per second](docs/img/lens-floorplan.gif)

*Lens replaying part of a Qwen3 decode step from an RTL simulation of the board configuration,
at 100 cycles per second. The dashes are data moving. The colours show what each unit is doing
in that cycle: busy, stalled on DRAM, lost TMEM arbitration, or waiting on a dependency.*

## What works, and what doesn't yet

Works, in simulation:

- **Qwen3-0.6B with its real weights** on the instruction-set simulator. It runs in int8, so
  its output differs slightly from Hugging Face's fp32 model because of quantization error
  (see [Accuracy](#accuracy)).
- **LFM2.5-230M**, Liquid AI's hybrid of short-convolution and attention layers, the same way
  (`otpu-chat --model lfm2`). Its 64-wide heads are zero-padded to the 128-deep matrix unit and
  its convolution state lives in DRAM; no ISA or RTL change was needed. See
  [docs/lfm2.md](docs/lfm2.md).
- **Qwen3.5-0.8B**, whose main layer is a Gated DeltaNet (linear attention with a 128 x 128
  fp32 state per head), the same way (`otpu-chat --model qwen35`). The 1 MiB of state per layer
  streams through the scratchpad head by head and the recurrence runs on the vector unit; no
  ISA or RTL change. It is the baseline of the architecture tournament: see
  [docs/qwen35.md](docs/qwen35.md).
- **RTL vs simulator.** The Verilator RTL ends with exactly the same memory contents as the
  simulator on the kernel tests, on a full Qwen3-0.6B token (6.38 M cycles), and on random
  programs where instructions keep conflicting over the same memory, which checks that the
  hardware keeps them in the right order while running units in parallel.
- **Board model.** A Verilator testbench of the board (`sim/verilator/tb_board.sv`) runs the
  bring-up and Qwen3 decode through the same host driver the card will use.

Not done or not verified:

- **No hardware run.** No bitstream has been built.
- **Clock speed is an estimate.** 106 MHz is a rough estimate from yosys, not from Vivado; it
  still has to be validated by a real build.
- **Tokens per second are projections.** They are simulated cycles divided by an assumed
  clock (100 MHz) and DRAM efficiency (80%), without host time.
- **The board model skips the hardest parts of the physical integration.** DDR3 calibration
  always succeeds, the DDR3 controllers are replaced by an AXI memory model, and PCIe, the
  clocks and the resets are not simulated. Expect problems to show up when it first runs on
  a real board.
- **The host tools** (`otpu-smi`, `otpu-lens`, ...) have only run against the board model and
  a fake device.

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
  elements. Everything else is fp32 with round-to-nearest-even and flush-to-zero. exp2, recip
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

## Measured numbers

All from Verilator RTL simulation of the board configuration (1 slice, 128-deep MXU with 2
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

One Qwen3-0.6B decode token at the board configuration (1 slice, 2 MXU columns, 8 lanes, AXI
memory path) takes about 6.4 M cycles. At an assumed 100 MHz that would be about 15 tokens/s
before host overhead. That is a projection, not a measurement.

LFM2.5-230M streams less than half as many bytes per token. Measured on the same RTL
configuration at 80% DRAM bandwidth and a 128-token context, a token takes 2.37 M cycles (96%
of the DRAM roofline), which would be about 42 tokens/s at 100 MHz, again a projection
([docs/lfm2.md](docs/lfm2.md)).

Qwen3.5-0.8B streams about 820 MB per token, but its DeltaNet recurrence runs on the vector
unit, which cannot keep up with DRAM: measured on the same configuration (80%, context 128) a
token takes 11.5 M cycles, 69% of the DRAM roofline, about 8.7 tokens/s at 100 MHz (a
projection). Half of it is the recurrence ([docs/qwen35.md](docs/qwen35.md)).

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

Lens is the profiler. It records a run (an RTL cycle trace, a simulator run, or, on the card,
the hardware trace buffer) and opens it in the browser: an overview with the roofline and where
the DRAM cycles went, a zoomable timeline, the floorplan replay above, and per-instruction and
per-source-line tables.

```
python3 -m opentpu.lens record mlp attn -o run.otpuprof
python3 -m opentpu.lens open run.otpuprof
```

On the board model, the hardware trace buffer rebuilds the same trace lines the simulator
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
only. They leave out the PCIe and DDR3 controllers, which add roughly 45K LUT more, and they
say nothing certain about what Vivado will achieve after place and route. The logs and patches
are in `tools/tourney/runs/`, and [docs/tourney.md](docs/tourney.md) explains how to run it.

```
make tourney COMP=otpu_vpu N=5 K=2     # 5 rounds, 2 candidates per round
make tourney-report COMP=otpu_vpu      # REPORT.md and a progress plot
```

## The board

The target is a YPCB-00338 card with an xc7k480t, two DDR3 SODIMM channels through MIG, and
PCIe through XDMA. [docs/board.md](docs/board.md) has the build and bring-up steps, and
[docs/status.md](docs/status.md) has the current state.

Current yosys estimate for the accelerator and control logic (with the hardware trace buffer):
85.6K LUT, 42.4K FF, 267 DSP, 635 BRAM36. Adding the vendor IP gives roughly 130K LUT, about
44% of the part. The first Vivado run will replace these estimates.

The host software runs on top of the stock Xilinx XDMA driver:

```
otpu-selftest            registers, DRAM patterns, kernels, then a model (--sim for the board model)
otpu-chat                chat with Qwen3-0.6B (or --model lfm2: LFM2.5-230M, qwen35: Qwen3.5-0.8B)
otpu-smi                 temperature, estimated power, DRAM use, per-unit utilization
otpu-lens                record a hardware trace and open it in Lens
```

Temperature comes from the FPGA's XADC. Power is only an estimate: Vivado's per-unit power
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
