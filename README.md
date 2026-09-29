# openTPU

**An open-source AI accelerator, developed by AI.**

openTPU brings the lessons of [auto-arch-tournament](https://github.com/FeSens/auto-arch-tournament)
to AI accelerators. It asks two questions: how far can AI agents go at hardware design, and can
they build the chip that runs their own inference?

![otpu-chat on LFM2.5-230M with otpu-smi watching the card](docs/img/card-chat-smi.gif)

*`otpu-chat` running LFM2.5-230M on the FPGA card (left), with `otpu-smi` showing the card's
utilization and DRAM bandwidth (right).*

## A place to learn

openTPU is also a learning project. The whole accelerator lives in one small monorepo that you
can read end to end: the hardware design (SystemVerilog), the instruction set, a bit-exact
simulator, a kernel language and its compiler, and the host software that drives a real PCIe
card. If you want to understand how an AI accelerator works, from a matmul in Python down to
the wires, this is a good place to start.

## Results

The design runs three modern models with their real weights on an Inspur YPCB-00338 card
(Xilinx Kintex-7 xc7k480t, two DDR3 channels), and the card produces the same tokens as the
simulator, bit for bit.

| Model | Weights | Decode, device | Decode, wall | Prefill, device | DRAM reads while decoding |
|:--|:--|--:|--:|--:|--:|
| LFM2.5-230M | int8 | 60.6 tok/s | 59.8 tok/s | 161.5 tok/s | 14.5 GB/s (85% of peak) |
| LFM2.5-230M | 4-bit, int8 head | 90.2 tok/s | 88.8 tok/s | 169.7 tok/s | 14.2 GB/s (83%) |
| Qwen3-0.6B | int8 | 22.8 tok/s | 22.7 tok/s | 55.0 tok/s | 14.4 GB/s (84%) |
| Qwen3-0.6B | 4-bit, int8 head | 34.2 tok/s | 34.0 tok/s | 58.5 tok/s | 14.1 GB/s (83%) |
| Qwen3.5-0.8B | int8 | 17.6 tok/s | 17.5 tok/s | 44.1 tok/s | 14.5 GB/s (85%) |
| Qwen3.5-0.8B | 4-bit, int8 head | 24.7 tok/s | 24.6 tok/s | 46.3 tok/s | 14.2 GB/s (83%) |

*Measured on the card with the production image (`deploy_pnbl32_e2521032`: 120.755 MHz,
32-beat DRAM read bursts, 8-beat write bursts, DDR3-1066 with a 17.1 GB/s peak) and the host on
main, with the card in the omarchy PC (Intel Core i5-12600KF), before it moved to opentpu (see
below). Decode is greedy, 96 tokens; "device" counts only the cycles the accelerator runs and
"wall" adds the host. LFM2 4-bit wall is the median of 5 runs (88.1 to 88.9); the others are one
run each. Prefill is a 512-token prompt. DRAM reads come from the card's own counters while it
runs. Every configuration matches the simulator token for token. More detail in
[docs/board.md](docs/board.md).*

Since 2026-09-28 the card sits in another PC, opentpu, with an older CPU (Intel Core i7-4790).
There, the same image and host code give the same device numbers (within 0.3%), but the host
takes about 0.45 ms per token between the end of one run and the start of the next, against
about 0.18 ms on omarchy. So wall is lower: LFM2 4-bit decodes at 86.8 tok/s (median of 5 runs,
85.5 to 87.1). The single runs of the others: LFM2 int8 58.1, Qwen3 int8 22.6, Qwen3 4-bit
33.7, Qwen3.5 int8 17.1, Qwen3.5 4-bit 24.0 tok/s.

Since 2026-09-29 the production image is `deploy_secand3_02569bc`, measured on opentpu. It
carries the stream engine ([docs/stream.md](docs/stream.md)), one programmable unit that
replaces the VPU's composite chains and the DMA's DeltaNet step: 21% fewer LUTs and 52 fewer
DSPs in the VPU and DMA, at the same device speed.
- 4-bit decode: LFM2 90.5, Qwen3 34.5 and Qwen3.5 24.65 device tok/s, against the table's
  90.2, 34.2 and 24.7.
- Prefill: equal, and 5% faster for Qwen3.5.
- Every configuration is still token for token equal to the simulator.

The table above was measured with the previous image.

4-bit weights ([docs/quant.md](docs/quant.md)) use FP4 values with two-level block scales, 4.25
bits per weight, and keep the LM head in int8 for accuracy. They cut the bytes per token by about
a third and raise decode speed by 40% (Qwen3.5) to 50% (Qwen3), at a measurable cost in
perplexity that docs/quant.md reports per model.

The host is nearly out of the way. For LFM2 and Qwen3 the card runs one decode program compiled
once, which reads the position from a register and looks up its own embedding and RoPE rows, and
the logits stream back while the card is still running: the host adds 0.17 to 0.30 ms per
token on omarchy (0.45 to 1.3 ms on opentpu).
Qwen3.5 runs the same way for decode; its prefill still compiles each chunk's program on the host,
ahead of the card.

## How it works

```
  Kernels in ol              mlp, attention, full model layers
        |  @ol.jit
  Language + compiler        layouts, affine loop addressing, fusion
        |
  ISA                        8 x 32-bit words per instruction
        |
  ISA simulator  <======>  RTL          same bits, checked by the tests
  (Python)                 (SystemVerilog)
                            |  Vivado bitstream
                           FPGA card    Kintex-7 xc7k480t
                            |  PCIe
                           Host         otpu-chat, otpu-smi, otpu-lens
```

The machine is deliberately simple. A sequencer issues one instruction per cycle to a few
units: DMA moves data, the matrix unit multiplies int8 weights streamed from DRAM, the vector
unit does fp32 math, and a quantizer turns results back into int8. There is no cache and no
hidden scheduling: every data movement is an instruction, so a trace shows exactly where the
cycles go. [docs/isa.md](docs/isa.md) describes the whole instruction set.

A kernel looks like this:

```python
from opentpu import language as ol

@ol.jit
def mlp(h, gamma, w_gate, w_up, w_down, out, eps):   # simplified; see kernels/mlp.py
    x = ol.load(h)
    xs = ol.quantize(rmsnorm(x, ol.load(gamma), eps))
    g = ol.dot(xs, w_gate)
    u = ol.dot(xs, w_up)
    a = ol.all_gather(silu(g) * u)
    y = ol.all_gather(ol.dot(a, w_down))
    if ol.program_id() == 0:
        ol.store(out, x + y)
```

Because every data movement is an instruction, a trace of a run explains its speed. Lens, the
profiler, records a run from the RTL, the simulator or the card and opens it in the browser,
with a roofline, a timeline and per-instruction tables ([docs/lens.md](docs/lens.md)).

![Lens replaying a Qwen3 decode step on the floorplan](docs/img/lens-floorplan.gif)

*Lens replaying part of a Qwen3 decode step. Colours show what each unit is doing in each
cycle: busy, waiting on DRAM, or waiting on another instruction.*

## Try it

Everything except the card runs on a laptop.

```sh
pip install -e .
pip install pytest torch transformers
python3 -m pytest -q          # RTL tests also need Verilator 5

hf download LiquidAI/LFM2.5-230M --local-dir models/LFM2.5-230M
otpu-chat --model lfm2 --backend isa    # chat on the simulator
```

With a card, build the bitstream (`make bit` in [`boards/ypcb-00338`](boards/ypcb-00338)), load
it over JTAG, then run `sudo otpu-setup` and `otpu-chat --backend board`.
[docs/board.md](docs/board.md) walks through the bring-up.

| Command | What it does |
|:--|:--|
| `otpu-chat` | chat with Qwen3-0.6B, LFM2.5-230M (`--model lfm2`) or Qwen3.5-0.8B (`--model qwen35`) |
| `otpu-smi` | temperature, power, DRAM bandwidth and per-unit utilization |
| `otpu-lens` | record a run and open it in the profiler |
| `otpu-selftest`, `otpu-diag` | check that the card works |

## Where to start reading

1. [docs/isa.md](docs/isa.md): the instruction set. Everything else is built on it.
2. [`opentpu/kernels`](opentpu/kernels) and [docs/compiler.md](docs/compiler.md): how a kernel
   becomes instructions.
3. [`opentpu/isasim.py`](opentpu/isasim.py): the simulator, which is the spec.
4. [`rtl/`](rtl): the hardware, starting from [`rtl/top/otpu_top.sv`](rtl/top/otpu_top.sv).
5. [docs/lfm2.md](docs/lfm2.md), [docs/qwen35.md](docs/qwen35.md),
   [docs/benchmarks.md](docs/benchmarks.md): whole models and where their cycles go.
6. [docs/board.md](docs/board.md): the physical card, from clocks to PCIe.

## What's next

- **Faster prefill.** Prefill is limited by the matrix unit's multiply rate. A four-column matrix
  unit halves that cost in simulation (Qwen3.5 prefill from 2.5 to 1.8 Mcycles per token); the
  question is whether it closes timing at 120 MHz in the space left on the chip.
- **A faster clock and a smaller design.** A tournament of Vivado runs keeps working on the paths
  that stop the design at 125 MHz and on its area. Decode is bound by DRAM, so the clock mostly
  helps prefill.
- **The last few percent of DRAM.** Decode reads 83 to 85% of the DDR3 peak.

## Contributing

Issues and pull requests are welcome, and most of the work needs only Python and Verilator, not
an FPGA. Changes to the ISA, the simulator or the RTL must keep `python3 -m pytest -q` passing,
and performance claims should say how they were measured.

## License

Apache License 2.0. See [LICENSE](LICENSE).
