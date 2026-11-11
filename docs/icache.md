# The instruction cache (ICACHE)

With ICACHE the sequencer's IMEM is a direct-mapped instruction cache over the program, and the
program itself stays in DRAM. Without it, IMEM holds a copy of the whole program (4,096
instructions on the BCU1525), and the compiler refuses longer programs. With it, a
program's length is bounded only by the bitstream's address bits (CFG_IMEM): 16,384 instructions
on bcu1525-wg:ic. "Why a cache, and how big" below is the study behind the design; the numbered
sections are the implementation.

ICACHE is off by default. At ICACHE 0, otpu_seq and otpu_slice are today's: every change is
behind the parameter, and at 0 it folds to a constant (section 6).

## Why a cache, and how big

A study before the RTL (2026-10-31; bcu1525-wg, 4 x DDR4-2400, 150 MHz) replayed a cache model
over the full models' instruction streams, with dispatch timing from the board model's traces.
Its numbers are that replay's unless marked.

- **The sequencer has slack.** It dispatches on 0.1-0.9 % of cycles, and an instruction waits a
  median of 600-2,450 cycles in the window before its unit starts it (board model). A fetch
  costs time only when it is later than the window's next free slot, so a prefetcher a line or
  two ahead hides DRAM.
- **The design it chose:** 16 KiB of instructions in 1 KiB lines (32 instructions and their
  256 B of wait entries), direct-mapped, filled on port B where the loader was, with a
  prefetcher that follows the loop stack 2 lines ahead (an active loop's back edge goes to the
  loop's start), 4 where the loaded read latency is above ~300 cycles.
- **Stalls.** With that prefetcher every decode and prefill program studied stalls only for a
  run's first line, about 200 cycles: 0.0006 % of a token on Qwen3.8-27B, 0.004 % on
  Qwen3.6-35B-A3B and GPT-OSS-20B, 0.05 % of a single 35B prefill layer run (less than loading
  that run into IMEM costs). With no prefetch it is 0.3-2.2 %; with a plain next-2-lines
  prefetch 0.001-0.08 %.
- **DRAM traffic.** 2.2 MB a token on the 27B, 1.0 MB on the 35B, 0.56 MB on GPT-OSS. Injected
  into the weight stream on the board model, fills cost 4,144 +- 830 cycles a MiB (ten runs):
  0.026 %, 0.079 % and 0.043 % of a token, under one run's noise (+-0.5 %). At two slices each
  fetches from its own DIMMs and the cost is about twice that, 0.05-0.16 %. Decode's rate is
  unchanged by the cache within about 0.1 %.
- **Size.** The big loop bodies (the layer units, 1,127-7,945 instructions) are larger than any
  cache considered, so they are fetched again every iteration at every size. 32 KiB cuts only
  GPT-OSS's fills (436 to 128) and a 27B prefill's (3,403 to 3,019); 64 KiB cuts the 35B's from
  796 to 355 and holds all of GPT-OSS, worth 0.04 % of the 35B's token; 8 KiB costs GPT-OSS's
  generate loop 0.18 % of stall.
- **What it buys.** No 4,096-instruction ceiling: a 27B prefill program of 8,971 instructions
  replays at 0.0001 % of stall, and programs stay resident in DRAM, so a switch or a HALT CHAIN
  is a jump and not a reload. And on-chip memory: on the ASIC card point the IMEM and wait-entry
  macros are 0.91 of 1.06 mm2 of SRAM, 9.9 % of the 9.17 mm2 block. On the FPGA the study's
  estimate of 23.5 RAMB36 a slice was too high; section 4 has what the RAMs take as built.

As built, bcu1525-wg:ic takes 64 lines: the tags cost LUTs but no block RAM and no timing
(section 4).

## 1. What a bitstream with it does

- **Program in DRAM.** The program is laid out as the loader has always read it. Line L, a line
  being 32 instructions (1 KiB), sits at PROG_ADDR + 1 KiB x L. At WAITC the wait entries follow
  the last instruction's chunk, and line L's entries sit 256 B x L into them.
- **LOAD** takes PROG_ADDR and PROG_N and drops every line. Nothing is copied: STATUS LOADING
  clears once the port is idle. **HALT CHAIN** also drops the lines and jumps, so chaining to a
  program is a jump, not a reload. **RUN** keeps the lines, so a second run of the same program
  starts with the lines the first run left.
- **Registers.** CAPS2 bit12 is ICACHE. CFG_UARCH bit28 is ICACHE. CFG_IMEM is the longest
  program in words (instructions x 8), from IMEM_WORDS (docs/observability.md).

## 2. The cache (rtl/seq/otpu_seq.sv g_ic, rtl/top/otpu_slice.sv g_icrd)

- **Lines.** IC_LINES lines of 32 instructions and their wait entries: 64 (64 KiB) on
  bcu1525-wg:ic, 16 (16 KiB) the RTL's default.
  They are direct-mapped by the instruction's line in the program (pc / 32), and each slot holds
  a tag, a valid bit and a pending bit. The RAMs are IMEM's own (rows of IPR = D / 32
  instructions at fa_n), only shallower: IC_LINES x 32 / IPR rows. The wait entries' RAM is
  IC_LINES x 8 rows.
- **Fetch.** fa_n is chosen as before. Each of its candidates also gets the presence of its line,
  looked up from flip-flops: fa's line, the next line, the two loop starts on the stack, and a
  LOOP's skip and enter targets. rv says whether the row read at fe belongs to a line that was
  present. R takes the instruction (ld) only then. When R's instruction leaves and fa's row is
  not good, R empties instead of executing its instruction again. A missing line is filled, and
  fa's row is read again once the line is valid (ic_rf).
- **Fills.** A line comes in on port B. Its 1024 / D instruction chunks arrive and, at WAITC,
  its 256 / D entry chunks: each entry chunk first, then four instruction chunks, so that the
  entry chunk's IPR rows go in one a cycle from a staging register before the next one can
  arrive. The slice issues a fill's reads ahead of the units' (otpu_slice's port B arbiter). It
  tells a fill's returns from the DMA's by the order of its reads, because the memory returns
  port B's reads in order. At most NF = 4 fills are under way, and a demand fill goes first. A
  slot's tag changes when its fill starts, and the slot becomes valid after its last chunk is
  written.
- **Prefetch.** Each time fa enters a line, a walker follows the program from there along the
  top two loops on the loop stack. A loop end with iterations left goes back to the loop's start,
  a loop that fits inside the line is left, and a loop that is left is popped. The walker takes
  the first IC_PF distinct lines other than fa's (IC_PF 1 to 4, default 2) and fills those that
  are missing. It never fills into fa's own slot and never walks past the program's end. A step
  takes two cycles: first the next position into flip-flops, then the lookup and the fill.
- **Simulation.** Each slice prints `ICACHE sN lines= pf= fills= demand= prefetch= stall=` at
  the end of the simulation, and rtlsim.run's stats["icache"] holds those counts.

## 3. Host, compiler and targets

- **Host.** `Board.icache` reads CAPS2 bit12. `load_program` then writes the program, and at
  WAITC its entries, to DRAM only, and LOAD sets the slices' base. `device_config` and
  `_reg_config` set Config.ICACHE. The fake board models CAPS2 and CFG_UARCH.
  `SimTransport.icache` builds the board model with ICACHE, IC_LINES, IC_PF and IMEM_WORDS from
  the target. OTPU_ICACHE, OTPU_IC_LINES and OTPU_IC_PF override them in simulation, and
  rtlsim's tb_top takes the same keys.
- **Compiler.** Config.ICACHE enters the arch hash only when it is on, so every existing
  target's hash is unchanged. On an ICACHE target, IMEM_WORDS is the longest program. Every fit
  check (isasim, compiler, prefill, BoardBackend) uses cfg.IMEM_WORDS, so on such a target the
  4,096 cap becomes the program-size limit, and targets without ICACHE keep 4,096.
- **Targets.** bcu1525-wg variant `ic` sets ICACHE = true, IMEM_WORDS = 131072 (16,384
  instructions), IMEM_WC_STYLE = "block" and IC_LINES = 64 (section 4). Its ARCH_HASH is
  0xc4849fa6 (IC_LINES is a build parameter, not an arch key). bcu1525-x2's `wmx` and `b2`
  variants set the same keys.

## 4. Area

otpu_seq alone, measured with yosys `synth_xilinx -family xc7 -flatten -abc9`
(tools/synth/sta.sh) at S 1, D 128, WIN 16, WAITC 1. Logic arrival is the STA's worst
register-to-register arrival in ps; RAMs are RAMB36 equivalents:

| | LUT | FF | RAMB36 eq | logic arrival |
|---|---:|---:|---:|---:|
| base (before the change, IMEM_WORDS 32768) | 7,852 | 6,273 | 35.5 | 4,270 |
| ICACHE 0 | 7,861 | 6,273 | 35.5 | 4,686 |
| ICACHE 1, 16 lines, PF 2 | 9,342 | 7,411 | 18 | 4,740 |
| ICACHE 1, 64 lines, PF 2 | 10,029 | 7,623 | 18 | 4,740 |
| ICACHE 1, 64 lines, one-cycle walker step | 10,038 | 7,604 | 18 | 5,123 |

At ICACHE 0 the cell counts by type are the same as the base's, apart from abc9's LUT mapping:
the logic is the same (section 6), and abc9's result depends on the netlist's order. ICACHE adds
about 1,500 LUTs and 1,140 FFs: tags, the candidates' lookups, the walker and the fill
counters. Its critical path is the sequencer's own (stk_rem -> fa_n -> the RAMs' address). The
one-cycle walker step was 5.28 ns, and the two-cycle step took it off the critical path. 64 lines
add 687 LUTs and 212 FFs to 16's (the tags and the seven lookups, 64:1 each) at the same arrival
(on xcup, `synth_xilinx -family xcup` with LUT mapping: 10,352 -> 10,774 LUTs, +422). Their tag
lookups stay off the critical path, so bcu1525-wg:ic takes 64.

**Block RAM on the BCU1525 (VU9P), per slice, estimated** from the RAMs' widths and depths
(RAMB36 SDP 512 x 72, RAMB18 512 x 36, URAM288 4K x 72; Vivado has not mapped it):

| RAM | today | ICACHE (16 lines; 64) |
|---|---|---|
| IMEM | 1,024 rows x 1,024 bits, block: 28.5 RAMB36 | 128 x 1,024 (512 x 1,024), block: 14.5 RAMB36 |
| wait entries | 1,024 x 240, ultra: 4 URAM288 | 128 x 240 (512 x 240), block: 3.5 RAMB36 |
| total | 28.5 RAMB36 + 4 URAM | 18 RAMB36 |

That saves about 10.5 RAMB36 and 4 URAM a slice: on bcu1525-wg (S 1) 10.5 + 4, and on
bcu1525-s2 (S 2) 21 RAMB36 + 8 URAM. yosys's xc7 mapping agrees, at 35.5 -> 18 RAMB36
equivalents (xc7 has no URAM, so there the entries are block RAM), and so does its xcup mapping:
57 RAMB18E2 + 4 URAM288 -> 36 RAMB18E2, at 16 lines and at 64. The study's 23.5 RAMB36 a
slice assumed the cache was bounded by capacity (16 KiB = 3.6 RAMB36). It is bounded by width
instead: a fill writes a 1,024-bit chunk a cycle, so the rows are 1,024 bits wide whatever the
depth, and 64 lines cost the same 14.5 RAMB36 as 16. Rows of one instruction (256 bits, about
4 RAMB36) would need fills written over four cycles a chunk, which in turn needs port B's
returns held back or buffered. That would save another ~10.5 RAMB36 a slice. It is a follow-up,
not done.

## 5. Verification

tests/test_icache_rtl.py, 23 tests, bit for bit against the ISA simulator (DRAM, TMEM and
instruction counts in every slice):

- nested loops that fit or thrash, PF 4, an inner loop inside a line, a loop count from a
  register;
- D 32 and 64;
- WAITC's entries;
- the board's memory path with stalls;
- random programs at S 1 and S 2;
- HALT CHAIN to a program elsewhere in DRAM, with and without WAITC;
- a second RUN that starts warm, and a LOAD that drops the lines;
- BAR and GATHER at S 2 with the slices' collectives in other lines.

The rest of the suites and the model-level runs: section 7.

## 6. ICACHE 0 is today's IMEM

Outside g_ic (otpu_seq) and g_icrd (otpu_slice), every change takes ICACHE as a constant:

- the RAMs' write ports are `ICACHE ? cache's : loader's`;
- NROW and NSROW are today's;
- `ld` gains `ic_ok` (1);
- `fe` gains `ic_rf` (0);
- R's emptying is `ICACHE && ...`;
- the port B arbiter gains `!ic_req` (0) and the loader `!ic_busy` (0);
- g_noic ties the rest to 0.

The simulation's IMEM preload is today's at ICACHE 0. The suites run at ICACHE 0 and 1 (section
7).

## 7. Status

On the board model (Verilator 5.046), measured:

| suite | ICACHE 0 | ICACHE 1 |
|---|---|---|
| test_rtl, test_scoreboard, test_waitw_rtl, test_coll_hub | 290 passed, 1 skipped | 290 passed, 1 skipped |
| test_board, test_observability | 202 passed | 202 passed |
| test_kt | 22 passed | 22 passed |
| test_host | 103 passed, 1 skipped | |
| test_icache_rtl | 23 passed | |
| test_target, test_isa, test_migpair, test_bcuwg_build | 174 passed | |
| test_bcuwg_board (bcu1525-wg, :ic, :bhw) | 14 passed | |

The last three rows ran with IC_LINES 64. At ICACHE 1, OTPU_ICACHE=1 builds every board model with the cache.

Model-level runs on bcu1525-wg:ic, bit for bit against the ISA simulator (DRAM, logits,
instruction counts), with each run's fills from the RTL's own counters:

| run | instructions | cycles | fills at 16 lines | at 64 |
|---|---:|---:|---:|---:|
| 27B prompt run, T8 tile (one P pass) | 7,480 | 3,612,895 | 234 | 234 |
| 35B-A3B decode cut (a DeltaNet and an attention layer, 256 experts resident), steps 1-3 | 1,542 / 1,576 / 1,610 | 151,799 / 159,134 / 159,275 | 19 / 20 / 20 | 19 / 20 / 20 |

Each fill is the line's first (the programs' lines: 7,480 / 32 = 234; 607 to 635 words a step),
one demand fill a run (61 cycles of stall), the rest prefetched. These cuts' programs pass over
their lines once, so 64 lines change nothing here; the study's replay counts the full models'
looped programs, where they do ("Why a cache, and how big"). The cycles are the same at 16 and 64.

The 35B-A3B generate loop on the board model (bcu1525-wg, without the cache too) runs its first
token and then waits at the next token's embedding fence (WAITW served >= seq on the embedding
mailbox: the int8 table is on the host, Engine's MoE default embed_host): the card's
BoardBackend.run_generate serves the mailbox (backend.host, the RowServer) while the run goes on,
and the board model, a script replayed per flush, cannot. The run polls the word until the
WAITW's timeout (2^30 cycles). A board-model generate of a MoE takes Engine(embed_host=False), or
one token a run.
