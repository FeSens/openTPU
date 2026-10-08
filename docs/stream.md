# The stream engine (SE): one programmable unit for the VPU and DSTEP

Status: **draft spec, M1 (2026-09-29 01:30)**. It is the contract for tonight's RTL. The
reference semantics are the ISA simulator's `STREAM` (`opentpu/isasim.py`); the encoding is
`opentpu/isa.py` (`StreamDesc`, `stream`). Figures marked *measured* come from the champion's
routed build (tv-full-5a3238f96026, util_hier). Everything else is an estimate or a projection,
and is labeled as such.

## 1. Decisions

1. **One engine, SE, in the VPU's place.**
   - It runs every VOP and every DSTEP-style fused kernel.
   - Old encodings keep working through a decoder: VOP (0x30) and DSTEP (0x12) are unchanged
     instructions.
   - A new `STREAM` (0x13) takes a descriptor from TMEM.
2. **The DMA stays the stream mover.** It reads and writes the DRAM rows (port B, runs of 16
   chunks), does the TMEM fills and writes the row outputs, exactly as it does for DSTEP today.
   The arithmetic moves out of the DMA into SE.
3. **The DSTEP pipeline's dot A runs on the VPU's own reduction hardware.**
   - That hardware is lane slot 0's partial loop and the folding tree `u_vt`, the RDOT path.
   - DSTEP's A lanes (`g_la`) and its tree `u_ta` are removed.
   - The rest of DSTEP becomes SE's *tail*: the d stage, the delay line, the update lanes, the
     Q lanes and tree.
4. **The tail is programmable** through descriptor modes: the vector for dot A, the gate (a
   scalar, a column or 1), the per-row scalar d, and dot Q on or off. With these modes, one
   datapath serves the whole linear-recurrence family (section 5).
5. **Attention decode stays on the MXU.** Its scores and P·V dot products run there, in v1
   and v2.
   - At 8 fp32 lanes the engine would run int8 KV **16x slower** than the MXU path, which is
     already at 99-100% of its byte roofline (docs/qwen35.md, docs/benchmarks.md).
   - SE runs attention's softmax VOPs as today.
   - `STREAM` can express attention (section 5.3), and the simulator runs it, but it isn't a
     performance path at this lane width.
6. **Tonight's hardware implements a subset of the descriptor space.**
   - The subset is DRAM streams and the tail modes of section 4.3. The ISA sim implements all
     of it.
   - `CAPS` announces the subset. `opentpu.isa.stream_hw_cfg` decides whether a descriptor
     runs on the board, and the compiler falls back to VOPs otherwise.

## 2. Stages

A stream is a matrix S[rows, cols] read row by row, 8 fp32 words (one *segment*) per cycle.
Per row r:

| stage | computes | hardware in SE |
|---|---|---|
| **REDUCE A** | `A = isum_64(S[r] * opA)`, opA a column slot (a vector over c), S itself (square), or a constant | VPU lane slot 0 (partial loop, RL = 8) + `u_vt` |
| **SCALAR** | a short program over registers r0..r7, A, X[r] (a row scalar), constants K0..K3, +0, +1: `d = f(...)` | tail d stage: 3 multiply-add slots with operand muxes (v1); later a scalar sequencer with composites |
| **DELAY** | S[r] held until the row's d is known (T_U segment cycles) | tail block-RAM delay line (UBD = 128) |
| **MAP U** | `Y = S*G + D*B` (two rounded products, one rounded add: the VOP OUTER), G a constant, a column slot or 1; D a scalar register; B a column slot | tail: 8 x `otpu_fmma` |
| **MAP/REDUCE Q** | `O[r] = isum_64(Y * Q)`, Q a column slot | tail: 8 x (fmul + fadd) partial loops + `u_tq` |
| write | Y back to the stream destination (in place), O[r] to TMEM | DMA (unchanged) |

**ACC** (a column accumulator, e.g. Σ p·V) is expressed with what exists:
- as dot Q over V^T rows (p in a slot), for attention;
- as MAP U with a TMEM stream updated in place (the VOP OUTER, `MM ASCALE`).

No separate unit is needed. **Composites** (exp2, recip, rsqrt, log2, exp2sub) stay on the
VPU's two long lanes, as today: 10 slots, 2 columns per cycle. See section 9 for widening them.

Rounding is always that of the VOP of the same name: add, sub and mul round once each; FMMA is
`add(mul(a,b), mul(c,e))`; isum_64 is 64 interleaved partials followed by a folding tree;
everything is FTZ. **Every stage configuration is bit-exact with a documented VOP sequence**
(section 5).

## 3. The STREAM instruction and the descriptor

### 3.1 Instruction (opcode 0x13, `isa.stream`)

| field | meaning |
|---|---|
| `w1` | `desc` [15:0]: TMEM word address of the descriptor; `ks` [31:16]: the constants' stride (static, so the scoreboard knows K's range) |
| `w2 + R[ra]` | `src`: the stream, in DRAM bytes, or TMEM words with `SRC_T` |
| `w3 + R[ra]` | `dst`: where Y goes (in place: dst = src; or another state) |
| `w4 + R[rb]` | `vec`: column slots, slot i at `vec + i*cols` (i = 0..3) |
| `w5 + R[rc]` | `x`: row scalars, `X[r] = T[x + r]` |
| `w6 + R[rd]` | `k`: constants, `K_j = T[k + j*ks]` (j = 0..3, ks from w1) |
| `w7` | `out`: row outputs, `O[r] = T[out + r]` |
| flags | bit0 `SZERO` (S reads as +0, nothing is read: position 0); bit1 `SRC_T`; bit2 `DST_T`; bit3 `NODST` |

DSTEP is exactly `STREAM` with `gdn_desc(rows, cols)`, `ks = gs` and `vec = qk` (q is slot 0, k is
slot 1), `x = v`, `k = g`, `out = o`. The hardware decodes DSTEP to that configuration.

### 3.2 Descriptor (`isa.StreamDesc`)

Descriptors are **float-safe words**:
- the exponent field is 0x80 (a normal fp32 in [2, 4));
- the 24-bit payload is the sign bit plus the 23 mantissa bits.

So a descriptor is written with ordinary `FILL` VOPs and survives FTZ. The compiler keeps the
descriptors in a small reserved TMEM area and fills them once at the start of the program.

| word | payload bits |
|---|---|
| d0 | rows[11:0], cols[23:12] |
| d1 | a_en[0], a_op[2:1] (slot, self, const, max), a_idx[4:3], u_mode[6:5] (pass, fmma, mul, add), g_src[8:7] (reg, slot, const, one), g_idx[11:9], b_src[13:12] (slot, const, reg), b_idx[16:14], d_reg[19:17], q_en[20], q_idx[22:21], out_p2[23] |
| d2 | nops[3:0], rinit[7:4] (r4..r7 = K0..K3 at the start), rsave[11:8] (r4..r7 written back to K0..K3 at the end), q_self[12], o_reg_en[13], o_reg[16:14] |
| d3 | reserved (payload 0; ks is in the instruction) |
| d4 | srs[11:0], drs[23:12] (TMEM stream row strides; 0 = cols) |
| d5.. | one scalar op per word: op[3:0], dst[6:4], a[10:7], b[14:11] |

- **Scalar ops:** add, sub, mul, max, min, mov, rsqrt, recip, exp2, log2 and exp2sub, each
  exactly `fp32.py`'s function.
- **Scalar operands:** 0-7 are r0..r7, 8 is A, 9 is X[r], 10-13 are K0..K3, 14 is +0 and 15
  is +1.
- **Order:** every input vector, row scalar and constant is read before any output is written.
  Row r is written after row r is read. O and the saved registers are written last. The same
  rules hold for DSTEP.

## 4. The hardware (v1, tonight)

### 4.1 Dataflow

- **Stream entry.** The DMA reads the stream in DRAM runs of 16 chunks, 4 segments per 128-byte
  chunk. It drives `ss_pe`, a registered pipeline advance, as it does for DSTEP today: a segment
  is available or the pipeline is draining, and there is room in the write gather.
- **Advance.** In stream mode SE advances on `ss_pe` alone. It makes no TMEM access in that
  mode: the fills and the O writes go through the DMA's ports.
- **X.** The tail registers the segment (`xd`), the dot-A vector at its columns (`xa`, from
  slot 1 or slot 2) and its row meta (`first`, `final`, `row_last`, `sub`, `j`).
- **A.** In every lane, VPU slot 0 in RMA partial-loop mode computes `pacc = S*a + prev`, the
  RDOT path: the same bits as DSTEP's g_la, since the add is commutative bit for bit. `u_vt`
  folds a row's final partials into `kv`, `RD_MAX` = 28 cycles later.
- **D.** Three multiply-add slots with operand muxes (4.3) turn kv, X[r], K0 and K1 into d. d
  goes into an 8-entry FIFO.
- **U.** The segment leaves the delay line T_U cycles after X, with
  T_U = (CBD - 1) + TA + RD_MAX + 3 SL + 3, where TA is the A-stage latency (VPU slot 0: 1 + LM +
  LA = 7). There `Y = S*G + d*B`, and Y leaves to the DMA's gather.
- **Q.** `O[r] = isum_64(Y * q)` goes to the DMA, which writes it to TMEM.

**Throughput** is 8 words per cycle, the same as DSTEP: a 128x128 head takes 2,048 cycles plus
about 200. **Clock:** every arithmetic block already closes at 120.755 MHz in production. What
is new is one 2:1 mux on VPU slot 0's `a` and `b` inputs, and the configuration muxes in the
tail (section 7, risks).

### 4.2 Arbitration

- **One stream at a time.** SE runs either VOPs or one stream.
- **Requesting.** The DMA raises `ss_req` for the whole of a DSTEP or STREAM.
- **Draining.** SE stops taking VOPs (`rdy` low), drains the ones in flight (issue done, write
  buffer empty), then raises `ss_gnt`.
- **Filling.** The DMA fills and streams only after `ss_gnt`, and drops `ss_req` after the
  last O write.
- **No sequencer change.** DSTEP and STREAM stay U_DMA instructions, and VOPs stay U_VPU.
- **No deadlock.** A stream never waits for a VOP that depends on it, because the scoreboard
  issues nothing out of dependency order. A VOP issued before the stream and queued in SE
  finishes first.

### 4.3 The tail's modes (`ss_cfg_t`, the hardware subset)

| field | values | datapath |
|---|---|---|
| `ns` | segments per row: cols/8, cols ∈ {64, 128, 192, 256} | |
| `rows` | 1..256 | |
| `a_sel` | 0: dot A with slot 1 (k); 1: with slot 2 (a); `a_en` = 0 zeroes kv | X: the `xa` mux |
| `dmode` | 0 DELTA `d = (x - kv*K0)*K1`; 1 DELTA1 `d = (x - kv)*K1`; 2 SCALE `d = x*K1`; 3 DOT `d = kv*K1` | slot 1 b ∈ {K0, 1}, slot 1 a ∈ {kv, +0}; slot 2 a ∈ {x, p1}, c ∈ {-p1, -0}; slot 3 unchanged. Each is bit-exact to its op list (4.4) |
| `g_src` | 0 K0 (scalar); 1 slot 3 column; 2 one | U: the `G` mux (a column buffer at the U index, like k) |
| `q_en` | 0: no O (the DMA counts no O) | Q |

B is always slot 1 and Q always slot 0. The fill order is:
1. slot 0 (q);
2. slot 1 (k/b);
3. slot 2 (a, when `a_sel` = 1);
4. slot 3 (g, when `g_src` = 1);
5. x (the rows);
6. K0;
7. K1.

The fill takes one step per 8 words, and the DMA sequences it.

### 4.4 Descriptor to hardware (`isa.stream_hw_cfg`)

A descriptor runs on the board when:
- the stream is in DRAM, written in place or to a chunk-aligned dst apart from src (the
  compiler checks both), with rows 1..256 and cols a multiple of 64 up to 256;
- A is off or a slot (1 or 2);
- U is FMMA with B = slot 1 and G ∈ {K0, slot 3, one};
- Q is off or slot 0;
- no register init, save, q_self or o_reg is used;
- its scalar program is one of:

| dmode | ops (d_reg = the last dst) |
|---|---|
| DELTA | `r0 = A*K0; r1 = X - r0; r2 = r1*K1` |
| DELTA1 | `r1 = X - A; r2 = r1*K1` |
| SCALE | `r2 = X*K1` |
| DOT | `r2 = A*K1` |

- **The trick behind DELTA1 and SCALE.** The hardware keeps DELTA's three slots and passes
  through the identities `mul(z, 1) = z` and `add(x, -0) = x`, which are exact for every fp32
  value under FTZ, -0 included. DELTA1 sets slot 1's b = 1. SCALE sets slot 1's a = +0, b = 1
  and slot 2's c = -0. DOT sets slot 2's a = p1 and c = -0.
- **Otherwise** the compiler falls back to the VOP sequence of section 5.

## 5. Mappings

### 5.1 Every VOP as a descriptor (ISA-level semantics; the v1 hardware keeps the VOP decoder)

A VOP is a stream over the A tile's rows (TMEM, `SRC_T`, row stride ars) with the result in
dst (`DST_T`, stride drs). Its B operand becomes:
- a column slot (`B_COL`);
- a row scalar X[r] (`B_ROW`);
- a constant (`B_SCALAR`);
- or a second TMEM row stream S2 (`B_FULL`: the VPU's read port B; a descriptor extension, v2).

| VOP | configuration |
|---|---|
| ADD, SUB, RSUB, MUL (any bmode) | MAP U with G = one: Y = S + 1*B / S - B / B - S / S*G |
| MAX, MIN | MAP (a max/min stage op; composite-lane slot 0 bypass in the VPU today) |
| COPY, ABS, FILL | MAP pass / abs / constant |
| EXP2, RECIP, RSQRT, LOG2, EXP2SUB | MAP composite (the long lanes, 2 columns/cycle) |
| RSUM | REDUCE A with opA = const +1 (S*1 = S exact) → O |
| RSSQ | REDUCE A with opA = self → O |
| RDOT (b2/b1/b3/b0) | REDUCE A with opA = slot / X[r] / const / S2 → O |
| RMAX | REDUCE A max (the VPU's max tree) → O |
| OUTER (dmode scalar/column/one) | TMEM in-place stream: Y = S*G + X[r]*B, G = K0 / slot / one, B = slot (Cv) |
| DSTEP | `gdn_desc`, DRAM stream |

### 5.2 DSTEP and the linear-recurrence family on one datapath

State rows are value dimensions and columns are key dimensions (DSTEP's transposed layout). The
precomputed vectors come from the VPU (e.g. `a = alpha*k`).

| model family | per-row step | descriptor (hardware v1?) | VOP fallback |
|---|---|---|---|
| Gated DeltaNet (Qwen3.5, Qwen3-Next) | `d = beta(v - e S k)`, `S' = eS + d k`, `o = S'q` | DELTA, G = K0 (yes) | RDOT, MUL, SUB, MUL, OUTER(sc), RDOT |
| DeltaNet | the same with e = 1 | DELTA1, G = one (yes) | RDOT, SUB, MUL, OUTER(one), RDOT |
| Kimi Delta Attention (per-channel gate) | `d = beta(v - S(alpha*k))`, `S' = S*alpha + d k` | DELTA1, A slot 2 = alpha*k, G = slot 3 alpha (yes) | MUL, RDOT, SUB, MUL, OUTER(col), RDOT |
| GLA / HGRN2 | `S' = S*alpha + v k` | SCALE (K1 = 1), no A, G = slot 3 (yes) | OUTER(col), RDOT |
| RetNet | `S' = gamma S + v k` | SCALE, G = K0 (yes) | OUTER(sc), RDOT |
| Mamba2 / SSD (decode) | `h' = a h + (dt x) B`, `y = h' C` | SCALE (K1 = dt), G = K0 = a, slot 1 = B, slot 0 = C (yes); `+ D x` on the VPU | MUL, OUTER(sc), RDOT |
| mLSTM (xLSTM) | `C' = f C + (i v) k`, `n' = f n + i k` (rows = 1, X = 1) | SCALE (K1 = i), G = K0 = f (yes) | as above |
| RWKV-7 (generalized delta) | `S' = S*w + (S kappa)(-a*kappa)... + v k~` (rank 2) | two passes: DOT (A slot 2 = kappa, K1 = -1, B = a*kappa, G = slot 3 w), then SCALE (G = one, B = k~, Q = r) (yes, 2x state traffic) | as the two passes |
| Mamba1 (S6) | per-element decay matrix exp(Δ A) | **no**: the decay is a second full stream, and cols = 16 < 64 | VOPs |

Chunked prefill uses one stream per row and head, as DSTEP does today. Every row of a state
evolves independently. So a future multi-token mode can take T tokens per row pass and pay the
state traffic once per chunk instead of once per row. It is datapath-bound at 8 lanes, and is
v2 work.

### 5.3 Attention decode

- **Expressible:** scores `s[t] = K[t]·q` as REDUCE A over the K rows, and `o[j] = V^T[j]·p` as
  REDUCE A over the V^T rows with p in a slot. The softmax runs on VOPs between them. The ISA
  sim runs it bit-exactly against the VOP path (`tests/test_stream.py`).
- **Not a performance path.**
  - The KV cache is int8.
  - The MXU does 128 x MCOLS int8 MACs per cycle, at DRAM rate. The engine does 8 fp32 MACs per
    cycle.
  - At 8K context on Qwen3, attention on the engine would take ~0.49 s per token, against
    ~0.03 s on the MXU (projected).
- **The generic attention levers are on the MXU side** (projections in section 6):
  - one KV read for any query group (MM replay with RMAX/ASCALE for M > MCOLS: Qwen3.5 reads
    its KV twice at MCOLS = 2);
  - a 4-bit KV cache.

## 6. Performance (projected unless marked)

- **VOPs:** unchanged. The same pipelines and rates, since the VOP path is untouched.
- **Streams:** DSTEP's rate: 8 words per cycle plus about 200 cycles of fill and tail per head.
  The one new latency is TA: +1 cycle per stream.
- **Lost overlap (the cost of sharing).**
  - Today a DSTEP in the DMA runs beside VOPs. In SE they take turns.
  - Qwen3.5 decode: about 648K cycles of streams (288 heads x 2.25K) and about 522K of VOP
    issue per token (inventory estimate) against ~4.9M cycles per token. The worst case, if they
    had overlapped fully, is +11%.
  - Most VOP work, like the MLP's SiLU, doesn't overlap with the streams anyway. **Expect 0-5%;
    the Verilator run (perf_qwen --model qwen35) decides.**
  - LFM2 and Qwen3 are unaffected: they run no streams.
- **Attention levers, device tok/s** (W = the measured per-token bytes at short context; KV
  bytes per context token include the scales and LFM2's padded 64-dim K rows):

| model | KV B/token | 256 | 2K | 8K | 8K, KV read once | 8K, 4-bit KV |
|---|---|---|---|---|---|---|
| LFM2 4-bit (W 157 MB) | 9.6 KB | 88.8 | 80.2 | 60.1 | — | 71.6 |
| Qwen3 4-bit (W 412 MB) | 59.1 KB | 33.2 | 26.6 | 15.8 | — | 21.3 |
| Qwen3.5 4-bit (W 575 MB) | 6.3 KB, read twice | 24.5 | 23.6 | 20.9 | 22.6 | 23.6 |

## 7. Area (target ≤ VPU + DSTEP = 55.7K LUT, measured today)

| block | today (measured) | SE v1 (estimate) |
|---|---|---|
| VPU | 29.4K LUT / 68 DSP | 29.4K + slot-0 stream mux and stream control ≈ +0.6K |
| DSTEP A lanes `g_la` + tree `u_ta` | 9.4K / 16 DSP | **0** (VPU slot 0 and `u_vt`) |
| rest of DSTEP (d, delay, U, Q, Q tree, buffers) | 16.9K / 54 DSP | 16.9K + modes (G mux, slots 2/3, d muxes) ≈ +1.0K |
| **total** | **55.7K LUT / 138 DSP** | **≈ 47.9K LUT (-14%) / 122 DSP** |

**Roadmap below v1** (v2+, not tonight):
- composites at full width on the three per-lane stages (A, U and Q are 24 fp units): the two
  10-slot chains (~12K LUT, 44 DSP) go and SiLU runs faster;
- then one tree for A and Q in two phases. The estimated floor is about 35K.

**Risks for timing:**
- the slot-0 input mux on the VPU's composite lanes (their first slot);
- the `ss_*` wires between u_dma and u_vpu, registered on both sides;
- placement: the tail moves from the DMA's region next to the VPU.

## 8. RTL decomposition (3 pieces, exact interfaces)

`otpu_pkg.sv` gains:

```systemverilog
typedef struct packed {
  logic [5:0] ns;      // segments per row (cols / 8)
  logic [8:0] rows;    // 1..256
  logic       a_en;    // REDUCE A on (else kv = +0)
  logic       a_sel;   // A's vector: 0 slot 1 (k), 1 slot 2 (a)
  logic [1:0] dmode;   // 0 DELTA, 1 DELTA1, 2 SCALE, 3 DOT
  logic [1:0] g_src;   // 0 K0, 1 slot 3 column, 2 one
  logic       q_en;    // O on
} ss_cfg_t;
typedef struct packed {  // the X-stage meta of a segment (otpu_dstep's sm_t)
  logic v, first, final_, row_last; logic [7:0] sub; logic [4:0] j;
} ss_meta_t;
localparam logic [2:0] SF_NONE=0, SF_Q=1, SF_K=2, SF_X=3, SF_K0=4, SF_K1=5, SF_A=6, SF_G=7;
```

### Piece 1: `rtl/vpu/otpu_vpu.sv` (SE front and arbitration), owner R-A

New ports:

```systemverilog
input  logic        ss_req,          // DMA: a stream holds SE (level, until its last O)
output logic        ss_gnt,          // SE is idle of VOPs and in stream mode (level)
input  ss_cfg_t     ss_cfg,          // valid while ss_req
input  logic        ss_pe,           // advance (registered in the DMA)
input  logic        ss_in_v,
input  f32_t        ss_in_d [LANES],
input  logic [2:0]  ss_fk,  input logic [4:0] ss_fi,  input f32_t ss_fd [LANES],   // fills
output logic        ss_y_v, output f32_t ss_y_d [LANES],                           // Y
output logic        ss_o_v, output f32_t ss_o_d                                    // O
```

- **Arbitration** (4.2): `rdy` goes low while `ss_req`. `ss_gnt` is `ss_req && !busy`, where
  busy covers issue, the reductions, elementwise ops in flight and the write buffer. It holds
  until `ss_req` falls.
- **Stream mode** (`ss_gnt`):
  - `en`, `ren` and the lanes' enables become `ss_pe`;
  - no TMEM read or write enable is raised;
  - slot 0 of every lane runs the RMA partial loop with `sa = xd[l]` and `sb = xa[l]` from the
    tail (masked with `xm.v`);
  - its `first` comes from `xm.first` through the same delay as `mi.first`;
  - `u_vt` gets `cap = xm.v && xm.final_`, `row_last`, `sub` and `rst || init`, delayed like
    `mt` is today;
  - `root` and `root_v` go to the tail's `kv` and `kv_v` instead of the row buffer.
- **Tail.** The tail is instantiated inside `otpu_vpu`. It takes `ss_*` and hands back
  `xd`/`xa`/`xm`, and it receives `kv`/`kv_v`.
- **Latency.** The A-stage latency TA is exported as a localparam for the tail's T_U: X to
  pacc is 1 + LM + LA.

### Piece 2: `rtl/vpu/otpu_se_tail.sv` (new, from otpu_dstep.sv), owner R-B

```systemverilog
module otpu_se_tail import otpu_pkg::*; import otpu_fp::*;
#(parameter int LANES = 8, parameter int TA = 7) (
  input  logic clk, rst, init,          // init: a stream starts (counters cleared)
  input  ss_cfg_t cfg,
  input  logic [2:0] fk, input logic [4:0] fi, input f32_t fd [LANES],
  input  logic pe, input logic in_v, input f32_t in_d [LANES],
  output f32_t xd [LANES], output f32_t xa [LANES], output ss_meta_t xm,   // X registers
  input  f32_t kv, input logic kv_v,
  output logic y_v, output f32_t y_d [LANES],
  output logic o_v, output f32_t o_d);
```

- **What it is:** otpu_dstep without `g_la` and `u_ta`.
- **X:** `xa` is `kb` (slot 1) or `ab` (slot 2) at `jn`.
- **D:** the dmode muxes of 4.3.
- **U:** `G` comes from `e` (K0), `gb` (slot 3 at `uj`) or 1.0.
- **Q:** with `!q_en`, `o_v` = 0.
- **T_U** = (CBD - 1) + TA + RD_MAX + 3 SL + 3, with the d FIFO as today.
- **Unit test:** tail plus a copy of VPU slot 0 and `u_vt`, bit-exact against the ISA sim.

### Piece 3: `rtl/dma/otpu_dma.sv`, `otpu_seq.sv`, `otpu_pkg.sv`, `otpu_slice.sv`, owner R-B after piece 2 (or me)

- **DMA:** remove `g_ds.u_ds` and route its former ports to `ss_*`.
- **Handshake:** raise `ss_req` from a DSTEP/STREAM's start until `ds_flushed`, and hold the
  fill and the first read until `ss_gnt`.
- **DSTEP:** `ss_cfg` = GDN (ns, rows, a_en = 1, a_sel = 0, dmode = DELTA, g_src = K0,
  q_en = 1). The fill runs as today.
- **STREAM:** the fill starts with one TMEM read of `desc`, 8 words. The DMA decodes the
  float-safe payloads into `ss_cfg` and the addresses (slots at vec + i*cols, x, K_j = k +
  j*ks, out), then fills the slots the modes need (4.3). `o` is counted only with `q_en`.
- **pkg / seq:** `OP_STREAM` = 0x13 is a U_DMA instruction. The sequencer cannot read the
  descriptor, so the footprint takes the subset's largest shape (rows, cols ≤ 256):
  - DRAM `[src, src + 256 KiB)` and `[dst, dst + 256 KiB)` written (the same range in place);
  - TMEM reads of desc `[desc, desc + 8)`, vec `[vec, vec + 1024)`, x `[x, x + 256)` and K
    `[k, k + ks + 1)` (ks from w1, which is why it is in the instruction);
  - a TMEM write of `[out, out + 256)`.
  - The extra false conflicts only order a STREAM against its neighbours. SE runs streams and
    VOPs in turn anyway. DSTEP keeps its exact footprint, and it is what Qwen3.5 uses.
- **CAPS:** bit26 `STREAM` (the subset). bit7 is CHASH; bit6 DSTEP stays set.
- **slice:** wire u_dma and u_vpu `ss_*`. `HAS_DSTEP` now gates the tail inside u_vpu.

### Piece 0 (me): ISA sim, compiler, tests, integration and builds

- `STREAM` in `isasim.py` (full semantics) and `isa.stream_hw_cfg` (the subset).
- `deltanet_step` emits STREAM on a `Config(STREAM=True)` machine, and the generic
  `ol.state_step`.
- Tests: see 9.
- Then the integration runs on omarchy (Verilator), the OOC area on omarchy, the full build on
  opentpu, and the card.

## 9. Test plan

1. **ISA level (Mac).** `tests/test_stream.py`:
   - STREAM(gdn) against DSTEP against the VOP sequence, bit for bit (shapes 7..256 x
     64..256, specials, ZERO);
   - each family of 5.2 against its VOP fallback, plus float64 sanity;
   - RMSNorm as a TMEM stream against `lib.rmsnorm`'s VOPs;
   - attention (5.3) against the VOP path;
   - descriptor float-safety (FILL writes it exactly) and `stream_hw_cfg` coverage;
   - tiny Qwen3.5 with STREAM against DSTEP against VOPs, bit-identical.
2. **RTL op level** (omarchy, `tools/omarchy_test.sh`):
   - `test_vops.py -k rtl` (every VOP, unchanged path) and `test_rtl.py -k fuzz`;
   - `test_fp.py -k rtl`;
   - `test_dstep_rtl_bit_exact` (now through SE);
   - new `test_stream_rtl_bit_exact` (random hardware-subset descriptors, all dmodes and g_src,
     with VOPs interleaved to exercise the arbitration);
   - new direct RTL tests for EXP2SUB, MIN, RSUB, ABS, FILL and COPY (the inventory's gap).
3. **Tiny-model tokens on the RTL:**
   - `test_qwen35.py::test_tiny_qwen35_on_board_model`, `test_tiny_prefill_rows_dstep_on_rtl`;
   - `test_lfm2.py::test_tiny_resident_decode_on_rtl`;
   - `test_qwen3.py::test_tiny_prefill_chunk_with_mm_replay_on_rtl`.
4. **Performance:** `perf_qwen --model qwen35 --layers 4 --ddr 1066 --mhz 120` against main,
   and LFM2/Qwen3 1-layer (unchanged).
5. **Area and timing:**
   - OOC: `u_vpu` with the tail, against VPU + DSTEP;
   - full build at 120.755 MHz;
   - qual.sh on the card.

## 10. Honest limits of tonight's v1

- **The VOP datapath is reused, not re-expressed.** Descriptors are the ISA contract for new
  kernels. VOPs map onto them semantically (5.1) but keep their decoder.
- **No composites in SCALAR, no TMEM streams in hardware.** So RMSNorm-as-one-pass and
  attention on SE are ISA-level only.
- **The sharing that saves the area costs the VOP/stream overlap** (6). It is measured before
  anything is merged.

## 11. v2: composites on the eight lanes, one tree (the engine as built)

**Since 2026-09-29, v2 is the only configuration.** After the card qualified it, v1 was removed
rather than kept as a fallback, so that nothing is left to test and maintain.
- The parameters are gone: `SE_COMP8` and `SE_ONE_TREE` on the slice and the tops, and `COMP8`
  and `ONE_TREE` on otpu_vpu, otpu_se_tail and otpu_dma. So are `make bit SE=`, `OTPU_SE`, and
  the tail's own Q tree.
- With the stream engine (`HAS_SE`, LANES = 8), the composites run on otpu_se_comp and the
  streams on the one tree.
- The long lanes' composite chains remain for the VPUs without it (LANES != 8).
- The rest of this section is the design as it was planned and built that night. Its "v1"
  and "both 0" rows are history.

**v2 is parameters on v1's code, not a second engine.** The user's decision at 02:30: go to v2,
fall back to v1 if it doesn't pay. v1 (pieces 1-3) is nearly done, so v2 adds two build
parameters to it. Both 0 is v1, and the 05:00 checkpoint picks the values.

| parameter | where | 0 (v1) | 1 (v2) |
|---|---|---|---|
| `COMP8` | otpu_vpu | EXP2/EXP2SUB/RECIP/RSQRT/LOG2 on the CL = 2 long lanes (slots 1..9, ~12K LUT, 36 DSP) | the long lanes go. `otpu_se_comp` loops each composite chunk through the three per-lane stages S0 → RR → U → Q → S0 on all 8 lanes |
| `ONE_TREE` | otpu_vpu + otpu_se_tail | u_vt (A) and the tail's u_tq (Q) | only u_vt: A's windows and Q's windows alternate (below) |

- **Stages.** Each lane l has three fp stages.
  - S0 (core) and U (tail) are `otpu_fmma`, `y = a*b + c*e`.
  - Q (tail) is `otpu_fmul` then `otpu_fadd`, `y = a*b + c`.
  - Each stage's result comes SL = 7 cycles after its operands sit at its input mux: the input
    register, LM = 2, LA = 4.
  - In VOP mode every stage advances on the VPU's `en` (the tail's U/Q as well). In stream mode
    they advance on the registered `ss_pe`.
- **Bit-exactness.** Each composite runs its existing op list: fp32.py, the same order and
  rounding as today's slots. Op k runs on stage k mod 3 of pass k / 3. An idle stage passes
  `v*1 + -0`, which is exact. The range reductions (EXP2's clamp/floor/i2f, LOG2's split) are
  RR, 3 cycles in otpu_se_comp between S0 and U.
- **Throughput (se-v2's count, estimated).** In columns per cycle, against 2 today:

| composite | cols/cycle | gain |
|---|---|---|
| EXP2 | 2.67 | 1.33x |
| EXP2SUB | 2.67 | 1.33x |
| RECIP | 4.0 | 2x |
| RSQRT | 2.0 | 1x, with latency ~100 cycles against 70 |
| LOG2 | 2.0 | 1x |

  VPU issue cycles drop 15-18% on LFM2, Qwen3 and Qwen3.5 (inventory mix).
- **Area (estimated).**
  - COMP8: -1 to -3K LUT and -36 DSP. The chains go, and per-lane RR, state delays and
    operand muxes come in.
  - ONE_TREE: -5K LUT.
  - v2 ≈ 40-42K LUT against v1's ~46-48K. The checkpoint threshold is ≤ ~40K for both.

### 11.1 otpu_se_comp (owner se-v2), inside otpu_vpu

```systemverilog
module otpu_se_comp import otpu_pkg::*; import otpu_fp::*;
#(parameter int LANES = 8, parameter int MW = 64,    // MW: the core's opaque chunk meta
  parameter int NS = 3, parameter int HA = 2,        // stages per lane; hold's lead
  parameter bit EXT = 1'b1) (                        // 1: the stage units are the owner's
  input  logic clk, rst, en,                         // en: the VPU's enable (VOP mode)
  input  logic             in_v,                     // a composite chunk enters (at S0's mux)
  input  logic [2:0]       in_c,                     // f_cc(func) of V_EXP2, V_EXP2SUB, V_RECIP,
                                                     // V_RSQRT, V_LOG2 (otpu_pkg), from a flip-flop
  input  f32_t             in_a [LANES], in_b [LANES], // in_b: EXP2SUB's B
  input  logic [LANES-1:0] in_m,
  input  logic [MW-1:0]    in_meta,
  output logic             hold,     // S0 is taken HA en-cycles from now: issue no chunk then
  output logic [2:0]       st_sel,   // stage s (0 S0, 1 U, 2 Q) takes comp's operands this cycle
  output f32_t             st_a [3][LANES], st_b [3][LANES], st_c [3][LANES], st_e [3][LANES],
  input  f32_t             st_y [3][LANES],          // stage s's result, SL en-cycles later
  output logic             out_v,                    // P(f)*T en-cycles after in_v
  output f32_t             out_d [LANES],
  output logic [LANES-1:0] out_m,
  output logic [MW-1:0]    out_meta);
```

- **Stages.** S0 and U are otpu_fmma, with st_e = 1.0 (y = a*b + c). Q is fmul + fadd, which
  ignores st_e.
- **Timing.** T = 3*SL + RR (3) + 1 = 25 en-cycles from S0 to S0. The number of passes P
  depends on the function:

| functions | P |
|---|---|
| EXP2, EXP2SUB | 3 |
| RECIP | 2 |
| RSQRT, LOG2 | 4 |

- **Core side.**
  - S0's input registers take st_*[0] when st_sel[0].
  - Issue holds when `hold` is set: read address → mi → m0 → S0 is HA = 2.
  - out_* joins the write path in start order. The latency rule is unchanged: L(f) = P*T.
  - st_*[1] and st_*[2] go to the tail's U and Q.
- **Standalone test (se-v2).** tests/test_se_comp.py checks every function bit-exact against
  fp32.py and otpu_fp: edge values, random words, en stalls, NS 1..4 and HA 1..3.
- **Area, yosys.** 8.6K LUT + 2.2K SRL without the units, being cut toward ~8-9K LUT-eq.
  COMP8 is then about -2 to -3K LUT and -36 DSP net, and ONE_TREE carries most of the area
  target.

### 11.2 otpu_se_tail additions

The owner is se-tail (R-B). se-v2 may make the COMP8 part on its branch, and stream merges.

```systemverilog
  // COMP8: the tail's U and Q as generic stages for otpu_se_comp in VOP mode
  input  logic        cm,                        // VOP mode (!ss_gnt): U and Q advance on cen
  input  logic        cen,                       // the VPU's en
  input  logic        u_sel, input f32_t u_a [LANES], u_b [LANES], u_c [LANES], u_e [LANES],
  output f32_t        u_y [LANES],               // SL cen-cycles after u_sel
  input  logic        q_sel, input f32_t q_a [LANES], q_b [LANES], q_c [LANES],
  output f32_t        q_y [LANES],               // a*b + c, SL cen-cycles after q_sel
  // ONE_TREE: Q's final partials to the core's u_vt, its root back
  output logic        qp_cap, qp_row_last, output logic [7:0] qp_sub, output f32_t qp_d [LANES],
  input  logic        qo_v,   input  f32_t qo_d,
```

### 11.3 One tree, two phases (ONE_TREE; R-A in the core, R-B in the tail)

- **The windows.** otpu_vtree takes one 8-cycle window per row (64 partials, sub 0..7), and
  rows must start a multiple of RL = 8 cycles apart. At ns = cols/8 segments per row, A's
  window is the last 8 pe-cycles of each ns-cycle row period. Q's window is A's shifted by Δ
  pe-cycles.
- **Choosing Δ.** The windows never collide, and stay aligned mod 8, iff Δ = 8·o with o odd
  and not a multiple of 3. So Δ ∈ {8, 40, 56, 88, 104, 136, …}, for every ns ∈ {16, 24, 32}.
  The tail pads its Q path to the smallest valid Δ at or above its natural offset (104 if
  that offset is in (88, 104]). T_U grows by the pad, and the delay line (UBD = 128) holds it.
- **ns = 8 (cols = 64).** Both dots need 16 tree cycles per 8-cycle row, so the stream runs
  at half rate. The DMA presents each row as 16 segments, the 8 real ones then 8 of +0 (no
  DRAM read), and drops the Y of the padded segments.
  - The tail reads its column buffers as +0 for j ≥ 8, with `cfg.pad64`.
  - This is exact: a partial is never -0, so +0 terms change nothing. Q's products with q = +0
    are ±0, so they change nothing either.
- **In the core.** u_vt's inputs are `mux(A's partials, qp_*)` by which one has `cap`. In
  simulation it's an assertion that both never do. A tag delayed with the tree's latency
  routes the root to `kv` (A) or `qo_*` (Q). VOP-mode reductions use phase A only.

### 11.4 Checkpoint 05:00 (the coordinator's criteria)

v2 goes to the full build if:
- every VOP and STREAM RTL test is bit-exact with COMP8 = ONE_TREE = 1;
- the OOC area is ≤ ~40K LUT for VPU + DSTEP;
- the OOC timing is plausible at 120.755 MHz;
- no model is slower in Verilator.

Otherwise v1 (both 0) builds ~07:00.

### 11.5 COMP8 as built (se-v2's notes)

- **The op lists on the loop.** Op k runs at stage k mod 3 of pass k / 3. An op's factors may
  come in either order, since the multiply is commutative bit for bit.

| function | ops | pass 0 (S0, U, Q) | later passes | P | latency |
|---|---|---|---|---|---|
| EXP2 / EXP2SUB | 9 | x·1 + (-0 or -y); RR; f = xf·1 - i2f(i); C7·f + C6 | the 6 Horner steps C5 .. C0 | 3 | 75 |
| RECIP | 6 | 2 - ay; t·y; 2 - ay (a: x's significand) | t·y; e = 1 - ay; y·e + y → v | 2 | 50 |
| RSQRT | 10 | -(x·0.5); y·y; 1.5 - (y·y)h | t·y; a Newton step; y·y; e = 0.5 - (y·y)h; y·e + y → v | 4 | 100 |
| LOG2 | 10 | m·1 - 1; RR (i2f(e)); C9·t + C8; q·t + C7 | C6 .. C1, then q·t + i2f(e) | 4 | 100 |

  - The first Horner coefficient (C7 or C9) is written into v by the op before it.
  - At elaboration, each stage keeps only the mux inputs its ops use (`st_use`). At NS = 3,
    Q's c is a constant except in RECIP's last op (y·e + y: k2), and Q writes only v.
  - RR's floor is ffloor on its domain (|x| < 128). Its i2f is a 9-bit exact one.
  - EXP2's range flags don't clamp x. A flagged result is +0 or +inf whatever the steps
    compute.
- **`hold`** is a tap of a T-bit shift register ("a continuing chunk was presented T - 2
  cycles ago"). `ret` and S0's control are registered a cycle early, so S0's selects leave
  flip-flops.
- **A bug fixed on the way (v1 too).** otpu_vpu's count of elementwise VOPs in flight (`ew_n`)
  was 4 bits. One VOP may start every other cycle, so up to latency / 2 can be in flight:
  about 33 in v1 and about 50 with COMP8's RSQRT/LOG2. The count wrapped, and busy, the
  latency rule and ss_gnt went wrong. It is now 7 bits.
  `tests/test_vops.py::test_many_small_composites_rtl_bit_exact` tests runs of 20-40
  independent 1-element composites.
- **NaN inputs** (isa.md: none are expected). The RTL gives a signed zero for recip(±NaN) and
  +0 for rsqrt(-NaN). That holds for the chains, `otpu_fp` and COMP8 alike, and COMP8 keeps
  those bits. fp32.py gives NaN for both.
- **Area, yosys** (synth_xilinx, flattened; otpu_vpu with the tail, HAS_SE, WBUF; LUT + SRL,
  DSP):

| build | LUT | SRL | DSP |
|---|---|---|---|
| v1 | 45,945 | 2,701 | 120 |
| ONE_TREE | 40,715 | 2,696 | 120 |
| COMP8 | 41,396 | 3,446 | 84 |
| v2 (both) | 36,558 | 2,924 | 84 |

  - These were measured before otpu_se_comp's last cut (513776f), which took it from 7,208
    to 6,347 LUT (EXT, MW 21).
  - otpu_se_comp's own path: 8,576 → 7,231 → 6,347 LUT, with 2.2K SRL16 (the k1/k2/ii/f
    delays: 76 bits × 3 stages × 8 lanes).
- **Timing** (yosys `sta`, logic only; Vivado OOC below).
  - Each stage's control is a ROM over (function, pass), built from `ucode` with constant
    arguments. Before, `k = pass·NS + s`, `k mod 2` and `(k - 1) mod 3` sat in front of the
    decode. `n_pass` is a constant per function.
  - The output side's control (destination, negate, the constant into v) and RR's flags are
    registered from a copy of (function, pass) delayed SL - 1. So the result's path (y, the
    destination mux, RR's range compare) starts at flip-flops.
  - Vivado OOC of otpu_vpu v2 at 8576057 (before both changes; the lead's run, 120.755 MHz):
    WNS +0.080 ns. Second worst: `g_sel[0].u_md` (the meta after S0) → `g_rr.q0`, 23
    levels.
  - yosys on otpu_vpu (tail, HAS_SE, WBUF) at 202856d (the ROM, without the registered
    output control): v2 5.67 ns against v1 5.43 ns (6.19 ns before the ROM). v2's worst path
    is now a DSP cascade in the tail's Q multiplier; v2 is 36,184 LUT + 2,912 SRL, 84 DSP;
    v1 45,892 + 2,700, 120 DSP.
- **Cycles per token** (RTL, tools/se_v2_perf.sh at se-int 202856d, removed with v1: perf_qwen with
  OTPU_DSTEP=1 OTPU_PAIR=1, DDR3-1066, 120 MHz; v2 with `--check`, bit-exact with the ISA
  simulator on all three):

| model | main | v1 | v2 | v2 - v1 |
|---|---|---|---|---|
| qwen35, 4 layers | 3,154,675 | 3,154,667 | 3,155,016 | +349 (+0.011%) |
| lfm2, 2 layers | 830,139 | 830,139 | 830,139 | 0 |
| qwen3, 2 layers | 1,716,608 | 1,716,624 | 1,716,721 | +97 (+0.006%) |

  - The decode tokens are MXU/port bound (VPU busy 2-5%), so the composite throughput doesn't
    reach the token time.
  - v2's longer RSQRT/LOG2 (100 cycles against about 70) shows where it's on the dependency
    chain: qwen35's DeltaNet +90 cycles a layer, attention +32; qwen3's attention +65 a layer.
  - VPU busy per phase drops with v2. qwen35 MLP 35,572 → 30,253, other 50,146 → 32,761;
    exp2's busy -22%, recip's -32%; rsqrt's +23%.

## 12. Results (2026-09-29; every number says measured or simulated)

"Simulated" means Verilator or the ISA sim. "Measured" means Vivado 2026.1 on omarchy or the
card. Rows still marked "pending" were not in when this was written.

**Outcome.** v2 is the build candidate se-cand3 (02569bc, branch `se-cand3`: se-int with main
merged in, both comp cuts, main's 5cd6c39 MXU queue, and three small timing cuts).
- It closes at 120.755 MHz (WNS +0.031 ns).
- It is qualified on the card: 0 FAIL, and all 12 token-exact checks pass.
- It decodes at production speed.
- VPU + DMA take 20.7% fewer LUTs and 52 fewer DSPs than production (pn32), with 83.9% of
  slices against 94.8%.
- It is merged into main (5fca9c9) and is the only configuration: v1 and its selectors are
  removed (11).

### 12.1 Bit-exactness (simulated)

| tree | config | result |
|---|---|---|
| se-int 2deead3 | v1 | 30 RTL tests pass (stream, DSTEP, VOPs, fuzz), plus the board-model stream and VOP tests |
| se-int 8576057 | v2 | R-A's full suite: 37 + 31 passed; R-B's stream/DSTEP 10/10 (v1 and v2), tiny models 10/10 |
| se-v2 | v2 | VOPs 15, fuzz + stream 16, tiny models 9 |
| 96d4a24 (= se-int 202856d) | unit | test_se_comp + test_se_vpu + test_se_tail: 29 passed |
| se-cand 44dfeb2 | v1, v2 | omarchy: 120 passed per config (stream + stream_rtl + se_comp/vpu/tail + vops/fp rtl 52; qwen35 board model / on_rtl / dstep 11; test_rtl fuzz / mlp / attention / mm_replay / scoreboard 57) |
| se-cand2 0d1f3ed | v1, v2 | omarchy: 151 passed per config (the above plus test_rtl dma / axi / run_arguments: 88); Mac: 122 per config |
| se-cand3 02569bc | v1, v2 | omarchy: 151 passed per config |
| se-cand3 on the card | v2 | measured: 12/12 token-exact against the ISA simulator (Qwen3, LFM2, Qwen3.5; int8 and 4-bit; per-position and resident), 12.5 |

- `test_stream_rtl_bit_exact` found one real RTL bug: with Q off (`q_en = 0`), rows still
  entered Q and shifted the next stream's O. Fixed in ab1f03d.
- **ew_n, in production too.** The 4-bit count of elementwise VOPs in flight (11.5) wraps on
  main. `test_many_small_composites_rtl_bit_exact` hangs on main at the sim window (no halt
  in 50M cycles) and passes at the board's WIN 16. The 7-bit fix is on main as its own merge
  (7a3bca1 test, cda1402 fix).

### 12.2 Cycles (simulated: `perf_qwen --ddr 1066 --mhz 120`, OTPU_DSTEP=1 OTPU_PAIR=1)

| model | main | SE v1 (202856d) | SE v2 (202856d, --check bit-exact) | v2 - main |
|---|---|---|---|---|
| Qwen3.5, 4 layers | 3,154,675 | 3,154,667 | 3,155,016 | +341 (+0.011%) |
| LFM2, 2 layers | 830,139 | 830,139 | 830,139 | 0 |
| Qwen3, 2 layers | 1,716,608 | 1,716,624 | 1,716,721 | +113 (+0.007%) |

- At 2deead3, v1 was 3,154,611 on Qwen3.5 (-64 against main). The lost VOP/stream overlap
  from 6 does not show up.
- **v2 is slower than main by 0.011% at most.**
  - Verilator is deterministic, so the extra cycles are real, not noise.
  - The coordinator accepted v2 at the checkpoint (03:13). They counted the extra cycles as
    within tolerance next to -16.3K LUT and -52 DSP.
  - The cause is COMP8's RSQRT latency (100 against ~70) where it sits on the dependency
    chain:
    - VOP.rsqrt busy on Qwen3.5 goes 11,541 → 14,196;
    - Qwen3.5 DeltaNet +90 cycles per layer (the gate is 2.7K), attention +32;
    - Qwen3 attention +65 per layer.
- **The candidates are cycle-neutral.** se-cand2 (0d1f3ed, v2) gives the same three numbers
  as 202856d. se-cand3's store register (TMEM read data registered before the chunk buffer) is
  +1 cycle per ST: Qwen3.5 -182, LFM2 +2, Qwen3 +7.
- **Prefill** (6 rows, MCOLS 2 and 4; Qwen3.5 and LFM2), v1 against v2: pending. The
  norm → quantize chain is where RSQRT's latency could show. Above ~0.5%, an RSQRT latency fix
  goes on the list.
- **Composite throughput doubles, but the decode doesn't see it.** exp2 busy -22%, recip -32%,
  exp2 wait_unit 6,835 → 3. These decodes are MXU/port bound, with the VPU 2-5% busy.

### 12.3 Area (measured, Vivado)

OOC at 120.755 MHz (LUT / FF / DSP):

| block | main | SE v2 (8576057) |
|---|---|---|
| VPU | 24,209 / 19,395 / 68 | 31,195 / 26,285 / 86 (COMP8 + ONE_TREE + tail) |
| DMA | 29,194 / 28,141 / 70 (with DSTEP) | 8,536 / 11,273 / 0 (no datapath) |
| **VPU + DMA** | **53,403 / 47,536 / 138** | **39,731 / 37,558 / 86 (-25.6% LUT, -38% DSP)** |

- ONE_TREE alone (u_vpu) was 34,291 LUT / 122 DSP.
- The SE DMA still carries ~1.9K of plain DMA, so the engine itself is ~37.8K.

Full build, post-synthesis hierarchy (se-int 600c087, SE=v2, against production pn32):

| block | pn32 | SE v2 |
|---|---|---|
| u_vpu | 33,513 / 22,061 / 68 | 43,538 / 31,319 / 86 |
| u_dma | 40,567 / 30,317 / 70 | 12,340 / 11,011 / 0 |
| **VPU + DMA** | **74,080 / 52,378 / 138** | **55,878 / 42,330 / 86 (-24.6% LUT)** |
| whole design | 234,977 LUT / 353 DSP / 554 BRAM | 218,701 / 301 / 554 |

- pn32 is not built from the same main commit, so the small deltas elsewhere (u_seq +1.5K
  for STREAM's footprint, u_quant -1.6K) are not all SE.
Full build, routed (se-int 600c087, SE=v2, against pn32; LUT / FF / DSP):

| block | pn32 | SE v2 |
|---|---|---|
| u_vpu | 29,287 / 21,642 / 68 | 39,701 / 30,998 / 86 |
| u_dma | 37,438 / 30,150 / 70 | 12,172 / 11,011 / 0 |
| **VPU + DMA** | **66,725 / 51,792 / 138** | **51,873 / 42,009 / 86 (-22.3% LUT, -52 DSP)** |
| whole design | 214,900 LUT / 577 BRAM | 206,064 / 569 |
| slices | 70,764 (94.8%) | 64,441 (86.3%) |

The routed design is 8.5 points less dense. The MCOLS = 4 density cliff was at ~97% slices.

**se-cand3 (02569bc), routed: the shipped v2 image** (against pn32; LUT / FF / DSP):

| block | pn32 | se-cand3 |
|---|---|---|
| u_vpu | 29,287 / 21,642 / 68 | 38,700 / 31,687 / 86 |
| u_dma | 37,438 / 30,150 / 70 | 14,204 / 11,300 / 0 |
| **VPU + DMA** | **66,725 / 51,792 / 138** | **52,904 / 42,987 / 86 (-20.7% LUT, -17.0% FF, -52 DSP)** |
| whole design | 214,900 LUT / 577 BRAM / 353 DSP | 202,757 / 569 / 301 |
| slices | 70,764 (94.8%) | 62,639 (83.9%) |

yosys (logic only, flattened) for VPU + DMA: main 63.7K, v1 55.1K, ONE_TREE 50.2K, v2 46.1K.

### 12.4 Timing (120.755 MHz, 8.281 ns)

- **Measured OOC:** main VPU WNS +0.357. v2 u_vpu at 8576057: WNS +0.080.
  - The worst paths were the `ss_act` → `en_r` replicas and comp RR.
  - 202856d cuts both: the enable goes through two stages, and RR is cut.
  - The OOC of 202856d was stopped for memory. It was not rerun.
- **yosys logic-only STA** (se-v2, 202856d): v2 5.67 ns, v1 5.43 ns. v2's worst path is now
  a DSP cascade in the tail's Q multiplier.
- **Full build, se-int 600c087** (v2, without either comp cut): **fails, WNS -0.199 ns**.
  - TNS -8.944 ns over 102 endpoints; hold met (+0.038). Post-route physopt did not help.
  - The 50 worst paths, by group:
    - comp, ~36 of them, worst -0.199: `u_md` (stage-0 meta) → `g_rr.q0` reset pins, 22
      levels (12 CARRY4, the k arithmetic in the decode). 202856d's ROM and bc05dd5's
      registered output control remove this path's logic.
    - u_mem, not SE: -0.177, the AXI_BL=32 run-length chain `qc` → `qb_n`, 17-19 levels,
      80% route.
    - seq → MXU: -0.137, the command total into the `q_total` LUTRAM, 7 levels. main's
      5cd6c39 replaces the LUTRAM with registers.
    - u_dma `cleft` → u_mem `lw_nx`: -0.118.
    - The reset fanout to u_vpu: -0.107, 0 levels.
  - pn32 closed with the same u_mem code at +0.013, so the non-SE paths are placement-dependent.
- **se-cand3 02569bc closes: WNS +0.031 ns, TNS 0 (0 of 610,014 endpoints), WHS +0.016 ns,
  WPWS +0.093.** The routed WNS was -0.218 ns (TNS -0.555); post-route physopt closed it.
  After placement it was +0.050, against 600c087's -0.295. On top of se-int, se-cand3 carries:
  - both comp cuts: the ROM over (function, pass), and bc05dd5's registered output control;
  - main's 5cd6c39: the MXU queue as head / next registers, with no `q_total` LUTRAM;
  - `cleft != 0` / `cleft == 1` as registers (`cl_nz`, `cl_one`). cleft → b_req goes from 5
    LUT levels to none, with the DMA's worst logic path unchanged at 4.23 ns;
  - a second reset stage for Q and the VPU, so their replicas load next to the units. They
    leave reset two cycles after the sequencer, and a simulation check guards every start;
  - qwen35's TMEM read data registered (`t_rq`) before the chunk buffer. yosys: t_rdata → lb
    was 1 LUT into the BRAM, and is now a register. This was the class that failed main's own
    07:00 build (-0.426).

### 12.5 Card (measured)

Measured on opentpu, 2026-09-29:
- **How it ran:** `deploy_secand3_02569bc` loaded over JTAG, then a warm reboot, then
  `LOAD=0 tools/qual/qual.sh ... fast` (42 min).
- **Result: 0 FAIL, 44 PASS.**
- **Selftest, first and final: ALL PASS.** That includes `stream` (STREAM in every mode, plus
  DSTEP) and the VOP op checks.
- **Warm diag** (quick memory test) passes after the 180 s soak, at 58-61 °C.
- **Token-exact:** 12/12 against the ISA simulator (6 configurations, each per-position and
  resident). Qwen3.5's DeltaNet layers run on SE.

Decode (`tools/decode_profile.py`, greedy, 96 tokens; device cycles, independent of the host),
against pn32's qualification (docs/board.md):

| model, 4-bit, int8 head | se-cand3 device | pn32 device | change | se-cand3 wall (opentpu) |
|---|---|---|---|---|
| Qwen3-0.6B | 34.46 tok/s (3.504 Mcycles) | 34.26 (3.524) | +0.6% | 34.06 |
| LFM2.5-230M | 90.52 tok/s (1.334) | 90.21 (1.339) | +0.3% | 87.17 |
| Qwen3.5-0.8B | 24.65 tok/s (4.899) | 24.65 (4.899) | 0.0% | 24.18 |

- **Prefill** (512 tokens, device), int8 / 4-bit:
  - Qwen3 55.0 / 58.5 and LFM2 161.5 / 169.7, both equal to pn32;
  - Qwen3.5 46.3 / 48.8, against pn32's 44.1 / 46.3.
- **DRAM read per token** equals pn32's in all six configurations.
- **Attribution.** se-cand3 also carries main's commits since pn32 (5cd6c39 among them), so the
  small decode gains and Qwen3.5's +5% prefill are not SE's alone.

### 12.6 Limits, as shipped

- **Attention decode stays on the MXU.** SE runs the DeltaNet recurrence (DSTEP and STREAM)
  and the VOPs. Attention on SE is ISA-level only (5.3).
- **NaN.** The RTL gives a signed zero for recip(±NaN) and +0 for rsqrt(-NaN). fp32.py gives
  NaN (11.5). No model feeds NaNs.
- **Mamba1** is not covered by the descriptor subset the hardware accepts (4.4).
- **cols = 64 streams run at half rate** under ONE_TREE (pad64: 16 segments, the padded ones
  dropped).
- **Latency.** COMP8's RSQRT and LOG2 take 100 cycles against the chains' 70. EXP2 takes 75,
  RECIP 50.
- **Footprint.** The sequencer compares a 5th read range (t4, K) for STREAM.
- **The SE DMA** is +6.6K LUT over the plain DMA; the datapath moved into u_vpu.
