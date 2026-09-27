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

| Model | Decode speed | Cycles per token | DRAM reads | Share of DDR3 peak |
|:--|--:|--:|--:|--:|
| LFM2.5-230M | 52.4 tok/s | 2.30 M | 12.4 GB/s | 73% |
| Qwen3-0.6B | 19.1 tok/s | 6.33 M | 11.8 GB/s | 69% |
| Qwen3.5-0.8B | 14.4 tok/s | 8.38 M | 11.6 GB/s | 68% |

*Greedy decode on the card at 120.755 MHz with int8 weights and DDR3-1066 (17.1 GB/s peak),
measured with `tools/decode_profile.py`. DRAM reads are estimated: bytes read per token, counted
by the card's counters on an earlier image, times the decode speed above. More detail in
[docs/board.md](docs/board.md).*

4-bit weights (FP4 with two-level scales, [docs/quant.md](docs/quant.md)) have run on the card
too, on a test image that is not production (`fp4f125`, 125.49 MHz, full-rate 4-bit MXU): Qwen3
at 28.0 tok/s with an int8 LM head and 32.8 tok/s all 4-bit, LFM2 at 77.6 and 94.1 tok/s, each
token for token equal to the simulator. That image computes RDOT wrong, so Qwen3.5 was not run
on it.

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

- **Use more of the DRAM bandwidth.** Decode is memory-bound, and there is still headroom.
- **4-bit weights.** Already in the simulator and RTL; they roughly halve the bytes per token
  once a bitstream with them meets timing.
- **A faster clock.** The core runs at 120.755 MHz; a 125 MHz build is close.
- **Faster prefill and Qwen3.5's recurrence**, both limited by compute today.

## Contributing

Issues and pull requests are welcome, and most of the work needs only Python and Verilator, not
an FPGA. Changes to the ISA, the simulator or the RTL must keep `python3 -m pytest -q` passing,
and performance claims should say how they were measured.

## License

Apache License 2.0. See [LICENSE](LICENSE).
