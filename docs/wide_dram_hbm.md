# Design note: one slice on more than 4 GiB (VU33P 8 GiB, VU47P 16 GiB, BCU1525 up to 64 GiB)

Status: a proposal, not implemented (2026-11-01). Numbers marked *estimate* are arithmetic, not measurements.

It builds on two pieces that already run on cards:
- the DUAL memory path of this fork: a slice on four channels (docs/sqrl-fk33.md);
- the DAW = 33 design of the openTPU_FK port: one address bit more per DRAM field, validated on the FK35 and the FK33 with
  layers above 4 GiB, token-exact against the ISA simulator (`M:\openTPU_FK\docs\DRAM33_spec.md`; its code is uncommitted
  in `M:\openTPU_FK\external\opentpu`, on an upstream commit that is an ancestor of this fork).

## Targets

| part | memory | today | proposed |
|---|---|---|---|
| VU33P (SQRL FK33) | 8 GB HBM2, 2 x 4-high stacks, 256 MiB pseudo-channels | 1 slice, DUAL, 4 GiB | 1 slice, DUAL, **8 GiB** (DAW 33) |
| VU47P | 16 GB HBM2, 2 x 8-high stacks, 512 MiB pseudo-channels | no board | 1 slice, DUAL, **8 or 16 GiB** (DAW 33 / 34) |
| VU9P (BCU1525) | 4 x DDR4 RDIMM, MTA18ASF2G72PZ, 16 GiB each, LiteDRAM, one channel a DIMM | 2 slices, 2 GiB of each DIMM: 4 GiB a slice | 2 slices x **32 GiB** (DAW 35), or 1 slice, DUAL, **64 GiB** (DAW 36) |

The DAW is the target's: the same board files build any DAW up to what the part holds. A smaller DAW uses part of each
channel.

## Where the 4 GiB limit is

| layer | today | limit |
|---|---|---|
| ISA | a DRAM byte address is `R[x] + imm`, 32 bits (registers and immediates) | **4 GiB** |
| slice ports | `a_addr`, `b_addr`, `sw_addr`: 32-bit word addresses, 30 bits used | 16 GiB without widening |
| native adapter | `otpu_native_dram`: channel beat address 25 bits (`n_caddr`) | 2 GiB a channel |
| channel bridge | `otpu_mem_ch` `c_cmd_addr` 25 bits; `otpu_hbm_port` BAW 25 / PCB 3 with 256 MiB pseudo-channels | 2 GiB a channel |
| memory | HBM IP AXI 33 bits (8 GB parts); the BCU1525's LiteDRAM ports **28 bits** (16 GiB a DIMM) | 8 GiB / 16 GiB a channel |
| shell, host | CFG_DRAM DRAM_LOG2 32, CH_LOG2 31; `target.py` refuses DRAM_BYTES > 4 GiB; the block designs map 2 GiB a channel | 4 GiB |

## Proposal

### 1. The ISA: a 4 GiB region per instruction (DAW 33 .. 36)

A target key `DAW` (32 default, at most 36) and the RTL macro `` `OTPU_DAW ``. A DRAM byte address is
**`{REG, (R[x] + imm) mod 2^32}`**:
- `REG` is a region index of DAW - 32 bits (0..4 bits), one per instruction, from spare bits of the opcode word;
- all the instruction's DRAM operands lie in that 4 GiB region.

Registers, immediates, strides and lengths stay 32-bit. There are no wider adders on the core's paths, and DAW = 32
programs run unchanged (REG is then empty, its bits zero).

| op | DRAM operands | REG bits (w0 = `op[7:0] ra[11:8] rb[15:12] rc[19:16] rd[23:20] flags[31:24]`) |
|---|---|---|
| LD, ST, DSTEP, WAITW | w1 | `rd[DAW-33:0]` |
| MM | w1 `sa`, w2 `ssa` (DRAM scale rows) | `rd[DAW-33:0]`, both operands |
| QST | w2 `dst`, w3 `sdst` | `rd[DAW-33:0]`, both |
| STREAM | w2 `src`, w3 `dst` (when in DRAM) | `flags[DAW-29:4]` (STREAM's flags use bits 3:0), both |
| HALT CHAIN, program loads | | region 0 |

Why one region per instruction, not DRAM33's bit per field: two DRAM fields x 4 bits do not fit MM's or QST's spare
bits, and the compiler can always co-locate an instruction's operands:
- a matrix and its scale rows go in one region;
- a K or V cache goes with its scale rows;
- a DeltaNet state is read and written in place.

At DAW 33 DRAM33's per-field encoding is the special case where both bits are equal, so its compiler and RTL port with
that one change.

The rules:
1. One instruction's DRAM operands, with all their rows and strides, stay inside its region. The ISA simulator raises
   `SimError` and the compiler `CompileError`; the run planners split a LOOP-walked range at 4 GiB, and the layout starts
   at a region boundary any tensor that would straddle one.
2. Region 0 holds everything a 32-bit register or table word names as an absolute DRAM address: program text, HALT
   CHAIN slots, the pointer table, DevVar addresses (MoE slots, token rows) and the WAITW words the host writes.
3. The hazard scoreboard compares (REG, lo, hi). Two different regions never conflict.
4. `arch_hash` drops `DAW` at 32 (like `DUAL` and `SEQ_WC`): every existing target keeps its hash.

### 2. The slice and its ports

REG rides with the command into the units: DMA, MXU issuer, quantizer, stream engine and the sequencer's scoreboard.
- **Up to DAW 34**, it leaves as the word address's top bits (`a_addr` / `b_addr` / `sw_addr` bits 31:30), and no port
  widens.
- **At DAW 35 and 36**, those three buses widen to DAW - 2 bits (`AWW = max(32, DAW - 2)`, a parameter of
  `otpu_slice`, the adapters and the boards). `b_par` covers the new bits. The ports' boundary registers just get one to
  four flops wider.

### 3. The memory path

The beat address of a channel, `BAW` (25 today), becomes a parameter through `otpu_native_dram` (`n_caddr`),
`otpu_native_dram2`, `otpu_mem_ch` (`c_cmd_addr`), `otpu_native_reg` and the simulation models (DRAM33's R2 did this up
to 26). Each adapter's queue entries grow by its address bits.

| configuration | DAW | channels a slice | channel bytes (CH_LOG2) | BAW | channel path |
|---|---|---|---|---|---|
| FK33 8 GiB | 33 | 4 (DUAL) | 2 GiB (31) | 25 | **unchanged** (the DUAL build's) |
| VU47P 8 GiB | 33 | 4 (DUAL) | 2 GiB (31) | 25 | `otpu_hbm_port` with 512 MiB pseudo-channels |
| VU47P 16 GiB | 34 | 4 (DUAL) | 4 GiB (32) | 26 | the same, HBM IP at 16 GB, AXI 34 bits |
| BCU1525 2 x 32 GiB | 35 | 2 | 16 GiB (34) | 28 | `otpu_mem_ch` 28 bits = LiteDRAM's ports as generated |
| BCU1525 1 x 64 GiB | 36 | 4 (DUAL) | 16 GiB (34) | 28 | the same, one slice on all four DIMMs |

Board by board:
- **HBM boards:** `otpu_hbm_port` takes the pseudo-channel's beat bits as a parameter. `PCW` is 22 for 256 MiB and 23 for
  512 MiB, so BAW = PCW + PCB, where the port assumes 2^22 today; its amap follows.
- **BCU1525:** the memory side already reaches 16 GiB a DIMM (`boards/bcu1525/litedram`, `beat_address_bits` 28). The
  limit is `otpu_mem_ch`'s narrowing and the shell's 2 GiB map.
- **BCU1525, one slice:** the DIMMs sit in three SLRs (C0 SLR0, C1 and C2 SLR1, C3 SLR2), so the slice in SLR1 reaches C0
  and C3 through `otpu_native_reg` stages on the native ports, as the FK35 crosses its SLL.

### 4. Shell, host and tools

- **`otpu_ctrl`:** CFG_DRAM reports DRAM_LOG2 = DAW, NCH and CH_LOG2. Its 8-bit fields hold 36 and 34.
- **Block designs:** XDMA's DMA maps channel j at j x channel bytes (16 GiB on the BCU1525: card addresses to 64 GiB, so
  `sc_dma` decodes bits 35:34). The address segments take explicit offsets, since 2019.2's Tcl 8.5 `format %X` truncates
  to 32 bits, as in the FK35's bd.tcl.
- **`opentpu/host/board.py`:**
  - `channel_map` reads channel_bytes from CFG_DRAM;
  - `beat_at`, `split`, `join`, `dual_at` and `dual_parts` are integer arithmetic, already width-free;
  - the transports' file offsets are 64-bit;
  - `scrub` writes the whole DRAM once per configuration: 64 GiB at the BCU1525's Gen3 x16 is *~7-10 s*;
  - `load_program` / `prog_at` stay in region 0;
  - `device_config` takes DAW from DRAM_LOG2.
- **`target.py`:**
  - DRAM_BYTES up to 2^DAW;
  - `DAW` in [arch];
  - `channel_bytes` in [platform] as the map's;
  - `waits.py` footprints carry REG.
- **Compiler and layout** (DRAM33's C1 plus co-location):
  - the address builders set REG;
  - the layout puts each matrix with its scale rows, and each cache with its scales, in one region, and starts a tensor
    that would straddle at the next region;
  - the relocation test runs a tiny model with every layer above 4 GiB (and above 32 GiB at DAW 36), bit for bit.

### 5. What it buys (*estimates*)

| model on one slice | bytes a token | VU33P 8 GiB | VU47P 16 GiB | BCU1525 64 GiB |
|---|---|---|---|---|
| Qwen3.5-4B int8 (4142 MiB) | ~4.3 GB | fits | fits | fits |
| Qwen3.5-9B mix (4-bit DeltaNet and MLP) | ~5.5 GB | fits | fits | fits |
| Qwen3.5-9B int8 | ~10 GB | no | fits | fits |
| a ~27-35B model, 4-bit (e.g. the 35B MoE's experts on the card) | ~15-20 GB resident | no | tight | fits |

Speed follows each part's port, not its capacity:
- **HBM parts with DUAL:** 4 x 64 B a cycle at 250 MHz is 64 GB/s, about 13-14 tok/s for the 4B int8 and 7-8 tok/s for
  the 9B mix.
- **BCU1525:** four DDR4-1600 channels give 51 GB/s of DRAM, and 4 x 64 B a cycle at its 100 MHz development clock gives
  25.6 GB/s of port. It is a capacity target first (large models, MoE experts resident) until its core clock rises.

The 4-bit layers stream at 128 B a cycle even with DUAL. **4-bit QUAD** (a 4-bit PAIR over two chunks: four blocks an
advance on the four columns when 4M <= MCOLS; a QACT copy into four ACT rows, a four-way block select in the PEs, four
scale entries an advance) is the follow-on that doubles them on every part.

## Status

W1, W2, W3 and W4 are done; every target stays at `DAW = 32` until it is switched.

- **Simulation:** the DAW 32 suites are unchanged (RTL 182, DUAL / DAW / STREAM / quantizer / wait RTL 135, FK33 board and
  target 183, compiler 712, models 355, MoE 359); `tests/test_rtl_daw.py` (46) and `tests/test_daw_compile.py` (78) cover
  DAW 33, 34 and 36. The remaining failures fail on the tree before this work too.
- **FK33 at DAW 33** (build f33e0001, the DUAL build with `DAW = 33`, `DRAM_BYTES` 8 GiB): timing met at +0.013 ns (WHS
  +0.010); selftest ALL PASS (its pattern test on both sides of 4 GiB). Card results:

| model | image | decode (device) | DRAM read a token | port while running |
|---|---|---|---|---|
| Qwen3.5-4B int8 | 7.5 GiB, the layers in region 1 | 13.58 tok/s | 4326 MB | 59.5 GB/s |
| Qwen3.5-9B fp4 | 7.6 GiB, the layers in region 1 | 7.72 tok/s | 4318 MB | 33.8 GB/s |

  Greedy answers from both are coherent. The 9B has no recommended mix yet (`formats.MIXES`), and its int8 (11.3 GiB)
  needs a 16 GiB part.
- **Not done:** the board model (rtlsim `SimTransport`) at DAW > 32; LFM2, Gemma 4, MTP, MoE expert slots and MiniCPM-V
  are not spread over regions (they must fit region 0); W5, W6, W7.

## Work packages

Order: W1 and W2 in parallel, then W3, then W4, W5, W6 one board at a time.

- **W1. Python:**
  - `Config.DAW` (32..36), the target key and hash;
  - `isa` builders and checks with REG;
  - `isasim` with a sparse 2^DAW DRAM and the region check;
  - `waits`;
  - gate: every existing target's golden programs unchanged.
- **W2. RTL:**
  - REG decode and carry (`otpu_seq`, `otpu_dma`, `otpu_mxu` with DUAL, `otpu_quant`, the stream engine), the (REG, lo, hi)
    scoreboard;
  - AWW-wide ports above 34;
  - BAW through `otpu_native_dram` / `otpu_native_dram2` / `otpu_mem_ch` / `otpu_native_reg`;
  - `otpu_board` / `otpu_ctrl` CFG_DRAM;
  - the Verilator models (sparse DRAM; the four-channel model at BAW 28);
  - gate: today's suites unchanged at DAW 32 (RTL 229, DUAL 34, board 183) and yosys equivalence at DAW 32.
- **W3. Compiler and host:**
  - layout co-location and region splits, loaders, `Board` and `channel_map`;
  - relocation tests at DAW 33, 34 and 36 in simulation (a tiny Qwen3.5 above 4 GiB, and above 32 GiB, on the board
    model).
- **W4. FK33 at DAW 33:**
  - target `sqrl-fk33` with `DAW = 33`;
  - the DUAL build unchanged otherwise; timing is the risk, it closed at +0.003 ns;
  - card: selftest, relocation, Qwen3.5-4B int8, 9B mix.
- **W5. BCU1525 at DAW 35 (S = 2) or 36 (S = 1, DUAL):**
  - `otpu_mem_ch` 28 bits, the block design at 16 GiB channels, the floorplan for one slice on four DIMMs;
  - the BCU1525 build has never been run on a board (docs/bcu1525.md: its bring-up comes first, at 4 GiB, today's
    target).
- **W6. VU47P at DAW 33 / 34:** board files for the actual card, `otpu_hbm_port` PCW, the HBM IP at 16 GB, a floorplan.
- **W7 (optional). 4-bit QUAD.**

## Open questions and risks

- **BCU1525:**
  - it is not built yet, so its 4 GiB bring-up (LiteDRAM DDR4 calibration on the four RDIMMs) should come before DAW 36;
  - at DAW 36 one slice drives four DIMMs in three SLRs; at DAW 35 each slice keeps its two DIMMs (C0 + C1, C3 + C2), as
    today's floorplan does.
- **VU47P:** which card (PCIe lanes, clocks, flash)? Does the 2019.2 license cover the xcvu47p, and does 2019.2's HBM IP
  take 8-high stacks?
- **Region co-location:** it is a compiler rule. Two DRAM operands that cannot share a region (none known today) would
  need a DMA copy first or DRAM33's per-field bits back.
- **Timing:**
  - the scoreboard's range compare and the DMA / MXU command registers grow by REG;
  - at DAW 35/36 the port and queue entries grow by a few bits each;
  - the FK33 has no slack to spare; the VU9P and VU47P have room, but cross SLRs.
