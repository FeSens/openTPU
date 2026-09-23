# openTPU — architecture design v3 (as built)

Date: 2026-09-23. Status: v3 describes what is implemented in `rtl/`, `opentpu/` and verified by
`tests/`. Sections 0 to 8 below are the v2 draft kept for rationale; where they disagree, this
section wins.

## v3: what was built, and how it differs from v2

| Topic | v2 draft | As built |
|---|---|---|
| Slices | One slice on the K7 | `S` is a parameter everywhere (RTL, ISA simulator, compiler). Tests run S = 1, 2 and 4. The K7 plan is S = 2, one per DDR3 channel. `GATHER` and `BAR` exist in hardware. |
| Block scale | fp16 per 128 | fp32 per D. D = 128 in the design and 32 in most tests. |
| Sync | Two queues with tile FIFOs | A scoreboard: one instruction per cycle into a WIN = 32 slot window with TMEM/DRAM/ACT footprints (see Concurrency below). |
| ISA | Five instructions | NOP, HALT, LI, ADDI, LOOP, BAR, LD, ST, MM, QACT, QST, VOP, GATHER. Eight 32-bit words each. See `docs/isa.md`. |
| TMEM | Monolithic | LANES = 16 banks (design), bank = address mod LANES, RPB = 4 reads and WPB = 2 writes per bank per cycle, private per-port data registers, arbitrated per cycle. The compiler uses odd row strides for MXU outputs. |
| VPU | 8 lanes fp32 | 8 lanes fp32, plus fused `EXP2SUB` and in-order chained row reductions so results are bit-exact. |
| DMA | Descriptors with strides | `LD`/`ST` of n contiguous words, min(D/4, LANES) words per cycle over the DRAM burst port. |
| Quantizer | Implicit in the MM stationary load | Explicit `QACT` (TMEM to ACT RAM, LANES wide) and `QST` (TMEM to int8 DRAM, used for KV append). |
| KV | V^T per 256-token block with an on-chip tail buffer | K token-major with per-block scales. V^T stored column-per-token with capacity stride and one scale per token. `QST` appends one token in place, so no tail buffer is needed. |
| Numerics | Unspecified | IEEE fp32 RNE with flush-to-zero and canonical NaN. exp2, recip and rsqrt are fixed add/mul sequences, identical in `opentpu/fp32.py`, `isasim.py` and `rtl/vpu/otpu_fp.sv`. |

### Data path per slice

```
 SEQ (16 regs, 4-deep LOOP, address registers) -- issues one instruction at a time
  |-- DMA    LD/ST      DRAM burst port <-> TMEM lanes
  |-- MXU    MM         DRAM chunk port -> FIFO -> D x MCOLS dot products x block scales
  |                     -> fp32 acc -> two result slots -> bank-aware TMEM drain (opt. ACC)
  |-- QUANT  QACT/QST   TMEM -> amax -> scale -> q8 -> ACT RAM | DRAM bytes
  |-- VPU    VOP        TMEM x (TMEM | row | col | imm) -> TMEM, LANES per cycle
  '-- COLL   GATHER/BAR shared across slices
```

### Verification layers

1. The fp units in RTL match `opentpu/fp32.py` on 245K vectors.
2. The ISA simulator running compiled kernels matches float64 numpy references within
   quantization tolerance.
3. RTL and the ISA simulator produce identical DRAM and TMEM bits for every kernel test and for
   random fuzzed programs on one and two slices.

### Concurrency (v3.1)

- **Scoreboard.** The sequencer dispatches one instruction per cycle into a WIN-slot window.
  Each slot holds the instruction's footprint (TMEM, DRAM and ACT RAM intervals, read and
  write). An instruction depends on every older slot it conflicts with; a dependency on an
  already-started instruction of the same unit counts as met, since units finish in order.
  DMA, QUANT and VPU start their oldest ready instruction; MXU and COLL start strictly in order.
  HALT drains the window.
- **Two-phase MM.** The MXU starts streaming (`ustart`) once its DRAM dependencies are clear and
  consumes (`urel`) once all dependencies are clear. It has a 2-entry command queue, so the
  next MM streams into a 128-deep FIFO while the current one drains.
- **TMEM arbiter.** Each cycle, units request bank slices; grants are all-or-nothing in the
  order DMA > COLL > MXU drain > QUANT > VPU, and a unit that is refused freezes for the cycle.
- **DRAM.** Port B serves DMA bursts and MXU streams (DMA first); port A serves QST writes and
  MXU scale reads.
- **Knobs.** `WIN`, `RPB`, `WPB` (`opentpu.rtlsim.UARCH`) and `FIFO_DEPTH` are RTL parameters.

New ISA flags (docs/isa.md): MM `RMAX` (row-max epilogue) and `ASCALE` (acc = acc*alpha + a.w),
QACT `CSCALE` and `RSCALE` (per-column and per-row scale while quantizing), VOP `RSSQ`
(sum of squares).

### Measured (RTL, design config S=2, D=128, MCOLS=8, LANES=16)

| Case | Of DRAM roofline |
|---|---|
| MLP M=1, H=1024, F=4096 | 99.5% |
| MLP M=8 | 97.2% |
| Flash attention 16q/4kv, d=128, T=1024 / 4096 | 97.4% / 98.2% |
| Full attention layer, pos=1023 | 95.8% |

Both decode kernels are memory-bound at the DRAM stream rate. At 8 lanes the G=6, T=512
attention drops to 79%, where the VPU softmax becomes the limit; Lens shows this case.

### Known simplifications

- QST writes one byte per cycle.
- The DRAM model is ideal and two-ported with a fixed latency. The board needs a DDR3 controller.
- fp functions are combinational. They must be pipelined for 200 MHz timing.
- The MXU dot product is behavioural. On the FPGA it maps to DSP48E1 cascades.
- RPB = 4 read ports per bank implies replicated BRAM or LUTRAM banks on the FPGA.

---

# v2 draft (kept for rationale)

## 0. What changed from v1 and why

- **Bug fixed — MXU fed 8x faster than memory.** v1 streamed 8 weight rows/cycle (1 KB/cycle =
  200 GB/s at 200 MHz) into an array on a 10–25 GB/s DDR3 system. v2 streams ONE 128-byte row per
  cycle and reuses it across up to 8 activation rows. 128 B x 200 MHz = 25.6 GB/s = DDR3-1600 x2
  peak, so the array is exactly balanced at M = 8 and memory-bound below it, as it should be.
- **Bug fixed — "no transposes" was false for P·V.** With token-major V the P·V reduction runs
  down a tile column. v2 stores V dim-major (V^T) per 256-token block; new tokens sit in an
  on-chip tail buffer and a full block is written out transposed once per 256 tokens.
- **One slice on the K7.** Multi-slice buys nothing when one array already saturates DRAM. The
  collective bus, global sync and GATHER disappear from the hardware. The slice/layout
  abstraction stays in the compiler so the HBM version can bring it back.
- **One dataflow for everything.** Matmul (decode, prefill, verify), q·K and P·V all use the same
  "stream a row, reuse it across M stationary rows" operation. No weight-stationary mode.
- **Adder tree -> DSP cascades.** 8 columns x 128 DSP48E1 in PCIN cascade = 1,024 int8 MACs with
  essentially zero fabric for the reduction. Trit mode is dropped: trits decode to int8 {-1,0,1}.
- **Semaphores -> tile FIFOs.** Two queues (DMA, compute) connected by named ring buffers with
  hardware full/empty. No WAIT/SIG in the common path; no deadlock-by-miscount class of bugs.
- **No KV page table.** Single tenant: each (layer, KV head) gets a contiguous region sized for
  max context. A `seq_len` register drives loop trip counts and addresses.
- **VPU: 8 lanes fp32** instead of 32 lanes fp16. Vector work is <1% of cycles; fp32 removes
  every fp16 accuracy question.
- **Bring-up model: dense Qwen3-0.6B/1.7B-class (head_dim 128 = array depth)** instead of the
  27B hybrid, which does not fit the K7 card anyway. GDN layers move to v2.

## 1. Goals (unchanged)

Golden ops: block-scaled matmul and attention. Thin hardware, everything visible to a
Triton/Gluon-style compiler with explicit layouts. K7 (xc7k480t, YPCB-00338) first, FK33 (VU33P,
HBM) later with the same programming model.

## 2. Decisions

| # | Decision |
|---|---|
| D1 | int8 mantissas, one fp16 scale per 128 along K, for weights, activations, K and V. fp32 everywhere else. (FP8 E4M3 would add alignment shifters to every accumulation; block-scaled int8 gives FP8-class accuracy without them.) |
| D2 | K7 v1 = one slice. Slices remain a compiler concept. |
| D3 | One MXU dataflow: streamed row (128 int8/cycle) x M <= 8 stationary rows. |
| D4 | Persistent program; host writes tokens, reads logits. |
| D5 | No speculation in hardware. Verify = the same program with M = draft+1 <= 8. Drafting is compiler/host work. |
| D6 | KV contiguous per (layer, head); K token-major, V dim-major per 256-token block; tail buffer on chip. |

## 3. Architecture (K7 v1)

```
   host (JTAG now, PCIe later) -- program, token in, logits out
          |
   +------v--------------------------------------------------------------+
   | SEQ: in-order, 2 queues, loop + address registers (layer, seq_len)  |
   |     |                                  |                             |
   |  DMA queue                         COMPUTE queue (MXU, VPU in order) |
   |     |  LD/ST descriptors               |                             |
   |     v                                  v                             |
   |  DMA engine --> tile FIFOs (rings) --> MXU 128 x 8 DSP cascades      |
   |   (stripes over     |                   |  streamed: 1 row/cycle     |
   |    both DDR3        |                   |  stationary: ACT RAM       |
   |    channels)        |                   |  (128 x RAMB18, M<=8 rows) |
   |     ^               |                   v                            |
   |     |            TMEM (BRAM tiles) <-- scale+fp32 accumulate (8)     |
   |     |               ^   |                                            |
   |     +---- ST -------+   +--> VPU 8 x fp32: exp2 max sum rsqrt silu   |
   |                              rope hadamard quantize residual         |
   |                           KV tail buffer (<=256 tokens per head)     |
   +--------------------+-----------------------+------------------------+
                        v                       v
                 DDR3 channel 0          DDR3 channel 1   (addresses interleaved)
```

### 3.1 MXU

- 8 columns x 128 rows of DSP48E1. DSP (i, j): P = PCIN + A * B, with A = stationary element
  x_j[k*128 + i] and B = streamed element w[n, k*128 + i]. The column sum leaves the bottom as
  the dot product of stationary row j with streamed row n over one 128-block.
- Streamed row: 128 bytes/cycle from a tile FIFO, broadcast to all 8 columns, skewed with SRL
  delay lines (~2K LUTs).
- Stationary operand: ACT RAM = one RAMB18 per row i in 72-bit mode, holding x_0..7[i] for up to
  256 K-blocks (K <= 32768). The block index k is the RAM address, so for each streamed row n the
  sequencer sweeps k and the stationary values change every cycle at no cost.
- Epilogue: per column, acc_fp32 += int_sum * s_w[n,k] * s_x[j,k]; after the last k, write
  y[j, n] to TMEM. Streaming row-major over (n, k) means one accumulator register per column,
  no accumulator memory.
- Throughput: 1 streamed row per cycle. Useful MACs = 128 x M per cycle. At M = 1 the array is
  memory-bound (as decode always is); at M = 8 it matches DDR3-1600 x2.

General form for later targets: 128 x (C x M) with C streamed rows per cycle. K7: C = 1, M <= 8.
FK33 decode: C ~ 10 (1.3 KB/cycle at 300 MHz ~ 400 GB/s), M <= 4 with DSP48E2 int8 packing.

### 3.2 Everything else

- **DMA**: descriptor {addr, bytes, stride, count, fifo}. 4 KB bursts, >= 2 in flight, addresses
  interleaved over both channels. The load path has an optional decoder (5-trits/byte -> int8).
- **Tile FIFOs**: named rings in BRAM (weights ring, K ring, V ring, spill). DMA pushes, compute
  pops; hardware full/empty handles synchronisation.
- **TMEM**: ~1 MB of BRAM tiles for activations, attention scores, KV tail, softmax state.
- **VPU**: 8-lane fp32, row-at-a-time over TMEM tiles.
- **ISA**: `LD`, `ST`, `MM`, `VOP`, `LOOP`, plus register writes (`seq_len`, layer base,
  strides). Five instructions.

## 4. Compiler: IR -> data movement

Layouts are the API (Gluon idea). On one slice the useful set is:
`Streamed(DRAM, row-major, block-scaled)` for weights, K and V^T; `Stationary(ACT RAM)` for
activation rows, query heads or softmax probabilities; `Tile(TMEM)` for everything in flight;
`KV(layer, head)` for the cache. Layout transitions are the only data movement: DRAM -> FIFO is
an `LD`, TMEM -> ACT RAM is the MM's stationary load, TMEM -> DRAM is an `ST` that only the KV
flush may use.

Rules:
- `matmul(x[M,K], W[N,K])`: `x` -> ACT RAM once; `LD` W rows as one sequential stream;
  `MM(n = 0..N, k = 0..K/128)`. Weights are read exactly once per M tokens.
- Fusion is mandatory: norms, RoPE, SiLU*up, Hadamard, quantize and residual are VOPs on tiles
  already in TMEM. An activation store is a compile error.
- Prefill = matmul with M = 8 token chunks. Verify (future speculation) = same with M = K+1.
- The program for one layer is identical across layers except for base registers, so the
  persistent program is one layer body inside `LOOP layers`.

SwiGLU MLP (M tokens):
```
x  = quantize(hadamard(rmsnorm(h)))        # VOPs, TMEM -> ACT RAM
g  = MM(x, Wgate) ; u = MM(x, Wup)          # stream Wgate, Wup once
a  = quantize(silu(g) * u)                  # VOPs -> ACT RAM
h += MM(a, Wdown)                           # stream Wdown once; residual VOP
```

## 5. Attention

Per layer, per KV head (sequential; each one streams at full bandwidth):
```
q  = stationary(quantize(rope(q_heads[6, 128])))       # 6 of 8 columns
for blk in kv(layer, head).blocks(seq_len):             # 256 tokens; last block = tail buffer
    s  = MM(q, K[blk])            * (s_k[blk] / sqrt(d))  # stream 256 K rows (token-major)
    m2 = max(m, rowmax(s)); p = exp2(s - m2)
    l  = l*exp2(m - m2) + rowsum(p); o = o*exp2(m - m2)
    o += MM(stationary(quantize(p * s_v[blk])), V^T[blk]) # stream 128 V^T rows (dim-major)
    m  = m2
o = o / l   ->  quantize  ->  out-proj MM
```
- q·K: streamed K row = one token, reduction over the 128 dims.
- P·V: streamed V^T row = one dimension over 256 tokens, reduction over tokens. V's per-token
  scale varies along the reduction, so it is folded into p before quantising p.
- Append: new K, V rows go to the tail buffer (VOP). When 256 tokens accumulate, the block is
  written out: K as-is, V transposed (one `ST` per block, 1/256 of tokens pay it).
- The tail block is attended from TMEM, so decode never waits on its own write-back.

## 6. Sizing (K7 v1, estimates)

| Block | Estimate |
|---|---|
| MXU | 1,024 DSP48E1 (of 1,920), ~3K LUTs of skew + control |
| ACT RAM | 128 RAMB18 |
| TMEM + FIFOs + KV tail | ~250 BRAM36 |
| VPU (8 x fp32) | ~16 DSP, ~15K LUTs |
| DMA, SEQ, host link | ~10K LUTs |
| DDR3 controller x2 | per the bonetto-soc controller |
| Clock target | 200 MHz (DSP cascades reach far higher; DDR3 user clock is the limit) |

Expected on Qwen3-1.7B int8 (~1.7 GB of weights): about 6–13 tok/s decode at 10–22 GB/s,
prefill ~0.2 TMAC/s. Qwen3-0.6B: roughly 3x that.

## 7. Open questions

- DDR3 controller maturity is still the critical path; bring-up starts from BRAM-resident layers.
- DSP48E1 has no int8 packing headroom (25x18); one MAC per DSP is assumed.
- The HBM version: lock-step lanes (one sequencer, C streamed rows) vs independent slices with a
  gather network. Decide with measurements after K7 v1.

## 8. Bring-up order (input to writing-plans)

1. Golden model in Python (numpy) for block-scaled matmul and attention with this exact layout.
2. MXU + epilogue in simulation against the golden model (Verilator).
3. DMA + FIFOs + TMEM with BRAM-resident weights over JTAG on the card.
4. One MLP layer, then one attention layer with the tail buffer, end to end.
5. Compiler: layouts, lowering, the persistent program; a cycle-approximate Python model runs
   the same instruction stream.
6. DDR3-backed full model.
