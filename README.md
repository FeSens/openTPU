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
| LFM2.5-230M | int8 | 51.9 tok/s | 48.5 tok/s | 124.8 tok/s | 12.4 GB/s (73% of peak) |
| LFM2.5-230M | 4-bit, int8 head | 74.2 tok/s | 60-67 tok/s | 141.0 tok/s | 11.7 GB/s (69%) |
| Qwen3-0.6B | int8 | 18.8 tok/s | 18.6 tok/s | 39.6 tok/s | 11.8 GB/s (69%) |
| Qwen3-0.6B | 4-bit, int8 head | 26.8 tok/s | 24.9 tok/s | 44.2 tok/s | 11.0 GB/s (64%) |
| Qwen3.5-0.8B | int8 | 14.3 tok/s | 14.2 tok/s | 26.1 tok/s | 11.9 GB/s (70%) |
| Qwen3.5-0.8B | 4-bit, int8 head | 17.6 tok/s | 16.9 tok/s | 27.9 tok/s | 10.1 GB/s (59%) |

*Measured on the card with the production image (`deploy_prod120fp4_ea3bc560`: 120.755 MHz,
DDR3-1066 with a 17.1 GB/s peak, full-rate 4-bit matrix unit). Decode is greedy, 96 tokens;
"device" counts only the cycles the accelerator runs and "wall" adds the host. Prefill is a
512-token prompt. DRAM reads come from the card's own counters while it runs. Every
configuration matches the simulator token for token. More detail in
[docs/board.md](docs/board.md).*

4-bit weights ([docs/quant.md](docs/quant.md)) use FP4 values with two-level block scales, 4.25
bits per weight, and keep the LM head in int8 for accuracy. They cut the bytes per token by about
a third and raise decode speed by 23% (Qwen3.5) to 43% (LFM2, Qwen3), at a measurable cost in perplexity that
docs/quant.md reports per model.

LFM2 with 4-bit weights is limited by the host, not the card: the host still compiles a program
for every position, and at 13.5 ms per token it cannot always keep up. Programs that take the
position from a register (compiled once, reused for every token) and streaming the logits while
the card runs are built and tested in simulation, and wait for the next bitstream.

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

- **Use more of the DRAM bandwidth.** Decode reaches 59 to 73% of the DDR3 peak on the card. In
  simulation, a faster memory adapter and a DeltaNet state engine (`DSTEP`) bring LFM2 and
  Qwen3.5 above 80%; they are waiting for a bitstream that routes at 120 MHz.
- **Host time.** Resident decode programs and streamed logits, built and tested in simulation,
  bring the host's share of a token under half a millisecond.
- **A faster clock.** The core runs at 120.755 MHz; builds at 125 MHz meet timing but need more
  margin on the card.
- **Faster prefill.** Prefill is limited by the matrix unit's multiply rate, so the next step is
  more multipliers per cycle.

## Contributing

Issues and pull requests are welcome, and most of the work needs only Python and Verilator, not
an FPGA. Changes to the ISA, the simulator or the RTL must keep `python3 -m pytest -q` passing,
and performance claims should say how they were measured.

## License

Apache License 2.0. See [LICENSE](LICENSE).
