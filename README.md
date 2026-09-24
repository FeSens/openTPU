# openTPU

A thin, Triton/Gluon-programmable inference accelerator for decode-time **attention and MLP**,
built for a Kintex-7 (xc7k480t) first and HBM (VU33P / FK33) later. The hardware exposes only
what a compiler needs: a streamed-weight matmul unit, a vector unit, explicit on-chip memories
and hardware loops. Every data movement is one visible instruction.

This repository has the whole stack, and every layer is tested against the one below it:

```
 kernels (opentpu/kernels)          mlp, attention_decode, attention_layer   <- written in `ol`
        |  @ol.jit, traced once per slice (SPMD)
 language + compiler                opentpu/language.py, opentpu/compiler.py
        |  layouts, affine loop addressing, fusion peepholes, bank-aware strides
 ISA (docs/isa.md)                  opentpu/isa.py         8 x 32-bit words per instruction
        |                                       \
 bit-exact ISA simulator            RTL (SystemVerilog, rtl/) under Verilator
 opentpu/isasim.py      <== same DRAM + TMEM bits ==>   rtl/top/otpu_top.sv
```

## Architecture

```
            host: program images, DRAM images in, DRAM out
                                  |
 +--------------------------------v-------------------------------------+  x S slices
 | SEQ  1 instr/cycle into a 32-slot scoreboard; 16 regs, LOOP, addr regs|
 |   |                                                                  |
 |   +--> DMA (LD/ST)      DRAM burst port, min(D/4, LANES) words/cycle |
 |   +--> MXU (MM)         1 streamed D-byte int8 row/cycle from DRAM   |
 |   |      FIFO prefetch, x M <= 8 stationary rows from ACT RAM,       |
 |   |      block scales, fp32 accumulate, optional ACC into TMEM       |
 |   +--> QUANT (QACT/QST) TMEM fp32 -> ACT RAM int8 | DRAM int8 (KV)   |
 |   +--> VPU (VOP)        LANES x fp32: add mul max exp2 recip rsqrt   |
 |   |                     row sum/max, broadcasts by row/col/scalar    |
 |   +--> COLL (GATHER/BAR) ---------------- shared by all slices ------+--> other slices
 |                                                                      |
 |  TMEM: LANES banks, 4 reads + 2 writes per bank per cycle, arbitrated |
 |  ACT RAM: MCOLS rows x ACT_BLOCKS blocks x D int8 (+ scales)         |
 |  DRAM: private per slice (weights shard, KV heads, I/O)              |
 +----------------------------------------------------------------------+
```

- **Numerics.** Weights, activations fed to the MXU and the KV cache are block-scaled int8 with
  one fp32 scale per D elements (D = MXU depth, 128 in the design, 32 in most tests).
  Everything else is fp32 with round-to-nearest-even and flush-to-zero. exp2, recip and rsqrt
  are fixed add/mul sequences, so Python, the ISA simulator and the RTL agree bit for bit.
- **Slices.** S is a parameter (S=2 planned on the K7). Kernels are SPMD. MLP is
  column-parallel with an all-gather. Attention is head-parallel, with each slice owning its KV
  heads and shards.
- **Concurrency.** The sequencer dispatches one instruction per cycle into a 32-slot window and
  tracks each instruction's TMEM, DRAM and ACT RAM footprint. An instruction starts when no
  older instruction in the window conflicts with it, so the DMA, MXU, quantizer, VPU and
  collectives all run at once. The MXU starts streaming as soon as its DRAM operands are safe
  and only waits for TMEM/ACT dependencies before consuming. TMEM banks are shared through a
  per-cycle arbiter (DMA > COLL > MXU drain > QUANT > VPU); a unit that loses simply stalls.
- **Attention.** Flash attention with online softmax, software-pipelined FA3-style: q.K^T of
  block b+1 streams while the softmax of block b runs. K is token-major with per-block scales.
  V is transposed with one scale per token, folded into P by the quantizer. Row maxima and the
  `acc * alpha` rescale are MXU epilogues, so the VPU is off the critical path.
- **MLP.** Row-sharded gate/up/down with chunked all-gather; gate/up of the next chunk stream
  while the VPU computes SiLU. Every weight byte is read exactly once.

The full rationale is in [the design spec](docs/superpowers/specs/2026-09-23-opentpu-design.md),
the instruction contract is in [docs/isa.md](docs/isa.md), and the programming model is in
[docs/compiler.md](docs/compiler.md).

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

# Weight(w, 0) shards w by rows across slices; Weight(w) replicates it.
res = launch(mlp, Config(S=2), backend="rtl",            # or "isa"
             h=Input(h), gamma=Input(g), w_gate=Weight(Wg, 0), w_up=Weight(Wu, 0),
             w_down=Weight(Wd, 0), out=Output((M, H)), eps=1e-6)
res.outputs["out"]
```

## Layout

| Path | Contents |
|---|---|
| `opentpu/fp32.py` | Bit-exact fp32 reference arithmetic (RNE, FTZ, exp2, recip, rsqrt, q8) |
| `opentpu/isa.py` | Encoder, assembler and disassembler |
| `opentpu/isasim.py` | Bit-exact multi-slice ISA simulator with hazard checks |
| `opentpu/compiler.py` | Tracer, tiles and layouts, affine addressing, loops, peepholes |
| `opentpu/language.py` | The `ol` kernel language |
| `opentpu/runtime.py` | Host arguments, DRAM placement, sharding, launch on either backend |
| `opentpu/kernels/` | MLP, attention decode, full attention layer, shared helpers |
| `opentpu/reference.py` | float64 numpy references |
| `opentpu/rtlsim.py` | Verilator build and run driver |
| `rtl/` | SystemVerilog: seq, dma, mxu, vpu, quant, mem, collectives, slice, top |
| `sim/verilator/` | Testbenches for the top level and the fp units |
| `tests/` | pytest suites (see below) |
| `opentpu/profile.py` | Parses the RTL cycle trace into per-instruction, per-unit and roofline data |
| `opentpu/lens.py`, `opentpu/lens_app.html` | Lens: profile files, recorder CLI and the browser app (docs/lens.md) |
| `opentpu/host/` | Card driver (XDMA over PCIe) and its tools: `otpu-smi`, `otpu-selftest`, `otpu-chat`, `otpu-lens` (docs/host.md) |

## Running

Requirements are Python 3.11 or newer, numpy, pytest and Verilator 5 for the RTL tests. RTL
tests skip themselves when Verilator is missing.

```
python3 -m pytest -q          # 96 tests: fp, ISA, compiler, kernels, RTL, fuzz, roofline
python3 -m opentpu.lens record mlp attn -o run.otpuprof   # profile workloads on the RTL
python3 -m opentpu.lens open run.otpuprof                 # explore them in the browser
```

| Suite | What it proves |
|---|---|
| `test_fp.py` | The RTL fp units match the Python fp32 model on 245K vectors |
| `test_isa.py` | Each instruction's semantics, loops, collectives and hazard detection |
| `test_compiler.py` | Layouts, loop addressing, broadcasts, fusion safety and error cases |
| `test_kernels.py` | MLP and attention on the ISA simulator match float64 references across S, M, T and D |
| `test_rtl.py` | Kernels and random hazard-heavy programs give identical DRAM and TMEM on RTL and the ISA simulator |
| `test_perf.py` | MLP and attention stay within a few percent of the DRAM roofline on the RTL |

## Lens: the profiler

Lens records runs into profile files (`.otpuprof`: RTL cycle traces, ISA-simulator runs, or
board counter snapshots) and opens them in a browser app with an overview (roofline and where
every DRAM cycle went), a zoomable timeline, a floorplan of the machine that replays the run
instruction by instruction with data movement and unit states, and per-instruction and
per-source-line tables. See [docs/lens.md](docs/lens.md).

```
python3 -m opentpu.lens list
python3 -m opentpu.lens record qwen-tiny --board --axi -o qwen.otpuprof
python3 -m opentpu.lens open qwen.otpuprof
```

## Results (RTL, design configuration S=2, D=128, 8 MXU columns, 16 lanes)

Roofline = cycles to move every byte the kernel must read or write over DRAM port B.

| Workload | Of roofline |
|---|---|
| MLP decode M=1, H=1024, F=4096 (49,547 cycles) | 99.5% |
| MLP M=4 | 98.5% |
| MLP M=8 | 97.2% |
| Flash attention 16 q / 4 kv heads, d=128, T=1024 | 97.4% |
| Same, T=2048 | 97.9% |
| Same, T=4096 | 98.2% |
| Attention G=6, T=2048 | 97.2% |
| Full attention layer at pos=1023 (norm, QKV, RoPE, KV append, attention, W_o) | 95.8% |

Before the scoreboard and the fusions, MLP ran at 93% and attention at about 30%. Short
sequences (G=6, T=512: 91%) are limited by the fixed prologue and epilogue.

## Known simplifications

- QST writes one byte per cycle. The DRAM model is an ideal two-port memory with a fixed latency.
- The fp functions are combinational and not yet pipelined for timing closure.
- The MXU dot product is behavioural. On the FPGA it maps to DSP48 cascades.
- MAX and MIN on NaN inputs are undefined.
