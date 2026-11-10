# TMEM without copies

Status: an experiment (2026-10-12), on branch `worktree-agent-ab3fe74f4336cefb9`, after
`docs/tmem_pump.md`. Nothing here changes the card's build: every knob defaults to today's TMEM
(`TMEM_IMPL 0`, six copies). Numbers are marked by where they come from: *measured* (an RTL
co-simulation bit-exact against the ISA simulator, a Vivado run, a trace of a co-simulation) or
*estimate* (arithmetic, a no-feedback replay of a trace). No number here comes from the card.

The question (the user, 2026-10-12): can the ISA or the microarchitecture change so that TMEM
needs no copies at all, or at least two? Explore in simulation first.

**Short answer.** One copy, no ISA change: 64 RAMB36 at 64K words instead of today's
384 (six copies) or P3's 192 (three, pumped). Each lane-bank is one true dual-port copy pumped at
2x (two reads and the write a core cycle), split in 2 RAMs; the board's arbiter stays, a read that
loses its RAM is replayed the next cycle while its unit holds, QUANT 3 and COLL are the DMA's
guests, and the read ports sit on the RAM ports by measured conflict (QUANT and VPU A on port A).
On the FPGA screen it makes 2.8-8.2% extra TMEM reads (the replays) and costs +0.70% cycles
geomean (+1.66% at worst, p8_q3), for 13.3K LUTs routed at 133.33 / 266.67 MHz, +0.232 ns
(today's TMEM 11.9K). With WAITC v2 on the full model list: +0.66% geomean against today's TMEM
without WAITC (section 6.5). A board builds it with `TMEM_IMPL=3`, the 2x clock from the block
design's MMCM (section 10). On the ASIC there is no pump: the copy is two-port macros at the core
clock, and a write takes one of the two accesses. That form costs +2.75% in cycles and 25-42%
extra TMEM reads, five times the pumped copy's (section 6.4). The user chose the
single copy over P3 (2026-10-12), and copies are out for the ASIC this core is for. What is left is
one RAM port's traffic: QUANT's data reads against the MXU's. Placement has done what it can;
fewer consumers remove the rest. In `dataflow`'s co-simulation, with the VPU's softmax numerators
fused away and QUANT fed from the drain, the replays fall 98-99% and the single copy costs nothing:
-0.00% at 4 rows, -0.15% at 8 rows and MCOLS 8 (section 6.6).

## 1. Why the TMEM has copies

A slice's TMEM is 64K fp32 words in 8 lane-banks (bank = word address mod 8). Eight read ports
(MXU, DMA, QUANT, VPU A, VPU B, QUANT 3, QUANT 2, COLL) and four write ports (DMA, MXU, COLL,
VPU) use it. A unit is granted all-or-nothing each cycle and holds when it is not. On the FPGA
(`TMEM_IMPL 0`) every read port has its own copy, except QUANT 3 and COLL, which share the DMA's:
six copies of 8 banks x 8K x 32 bits, 384 RAMB36, 64% of the card's block RAM
(`docs/tmem_pump.md`, "TMEM today"). Writes are already one per bank and cycle (the arbiter's
write masks), so only the reads need the copies.

What the reads need, *measured* (traces of the card's TMEM in 16 co-simulations,
`OTPU_SIM_PLUSARGS=+tmtrace=...`, `tools/tmem_trace.py`):

- **Bandwidth is not the problem.** The read ports are busy 9-43% of cycles together; two or
  more read streams overlap in 3-20% of cycles, three or more in 0.4-6.3%, and no cycle had more
  than five ports on one bank.
- **Addressing is.** A copy serves one address per bank per cycle. One copy with its banks split
  into sub-banks (8 or 16 RAMs a lane-bank, each with its own address) has the bandwidth, but two
  streams that walk the same sub-bank sequence in lockstep collide every cycle. Those are almost
  all *one unit's own* operands: VPU A and VPU B, and QUANT's data and its scales (section 3).
- **The grant is the hard part on the FPGA.** `docs/tmem_pump.md` built the banked TMEM (three
  copies, the arbiter counting reads per copy and bank) on the YPCB: with the grant in the RAM
  address path it missed 133.33 MHz by 3.27 ns, with request-priority addresses (`TMEM_RQ`) by
  1.06 ns, the failing paths spread over the slice. Any one-copy design that puts a RAM conflict
  into a unit's grant inherits that.

So the design question is not how to get more bandwidth but how to resolve a rare address
conflict without putting it into the grant, and without a second clock.

## 2. Method

1. **Traces.** The card's TMEM (`TMEM_IMPL 0`) writes every read, write, instruction start and
   end (`+tmtrace`). `tools/tmem_trace.py` measures overlap and streams, where each port's words
   come from (`flow`: the write port that produced them, how long ago, whether they are read
   again), and replays the reads through candidate organizations without feedback (`screen`:
   the cycles a conflict would add if nothing else moved; an upper bound in kind, not in size).
2. **The grant model** (`TMEM_IMPL 2`, `rtl/exp/tmem_nocopy/otpu_tmem_model.sv`): the data stay
   in a copy per port, so any grant reads the right words, and the model decides the grants of a
   candidate (sub-banks, hashes, reads per RAM, true dual-port RAMs, copies, lines kept per port,
   request-priority addresses, placement skews). The units feel the conflicts (feedback), and a
   run stays bit-exact by construction. Every port its own copy through the model (`x8`) is
   +0.00% against the card's TMEM, so the model's own path costs nothing.
3. **RTL** (`TMEM_IMPL 3`, `rtl/mem/otpu_tmem.sv`): one copy in fact. Its runs are checked
   against the ISA simulator (`--check`), so a wrong word shows.
4. **Vivado out of context** (`rtl/exp/tmem_nocopy/tmem_nc_top.sv`, `ooc.tcl`, `run_ooc.sh`):
   the TMEM and the slice's arbiter for it, every input and output registered, on xc7k480t-2 at
   133.33 MHz (the YPCB) and xcku5p-2 at 200 MHz (the UltraScale+ stand-in, `docs/tmem_pump.md`).

Workloads (`rtl/exp/tmem_nocopy/lists/`; the board's configuration: PAIR, DSTEP, STREAM, LiteDRAM
at DDR3-1066, core 133.33 MHz, 2 to 4 layers from position 544, fp4 weights, int8 head):

| tag | what | kind |
|---|---|---|
| d_q3, d_q35, d_lf | decode, Qwen3-0.6B, Qwen3.5-0.8B, LFM2.5 | decode |
| p_q3, p_q35, p_lf | a 4-row prompt run, MCOLS 4 | prefill |
| z_q4_r2, z_ph_r3 | Qwen3.5-4B at 2 rows, Phi-4-mini at 3 (zero images) | prefill, large |
| p8_q3, p8_q35, z8_q2_r4 | prompt runs at MCOLS 8 | prefill, MCOLS 8 |
| b4_q3, b8_q3 | batched decode, 4 and 8 users (MCOLS 4 / 8) | batched |
| l_q35m, l_g26 | a MoE layer run (35B-A3B, 26B-A4B), 2 rows | MoE |
| m_q35m, m_g26 | MoE expert rows, 1 row | MoE |

`screen.txt` (p_q3, p_q35, p8_q3, b4_q3, l_q35m, d_lf: one of each kind) carries the
candidates; `all.txt` the finalists.

## 3. Where the conflicts come from

### 3.1 Who produces what each port reads

*Measured* (`tools/tmem_trace.py flow`, 16 traces, word-weighted):

| read port | words | read again | produced by |
|---|---:|---:|---|
| MXU | 4.2M | 0% | MXU 92% (its accumulation), VPU 8% |
| DMA | 1.8M | 4% | MXU 59%, VPU 26%, DMA 16% |
| QUANT | 3.9M | 7% | VPU 94%, MXU 5% |
| VPU A | 19.3M | 18% | VPU 59%, MXU 34%, DMA 7% |
| VPU B | 7.5M | 28% | VPU 45%, DMA 36% (96% of those read again: gains, constants), MXU 19% |
| QUANT 3 | 0.07M | 99% | VPU 100% (the row factor) |
| QUANT 2 | 1.9M | 54% | DMA 84% (scales), VPU 16% |
| COLL | 469 | 0% | VPU 88% |

The distances are long: the median word reaches VPU A 94-7,600 cycles after it was written,
QUANT 140-860, the MXU 3,400-9,000. Producers do not hand words to consumers in flight; TMEM is
the buffer between them, which bounds what a direct path (B8 below) could take away.

### 3.2 One copy without feedback

Extra cycles in % of the run if every conflict cost its unit a cycle (`screen`, *estimate*):

| workload | 2+ streams | 3+ streams | one RAM a bank | 8 sub-banks | 16 | 16 hashed | 8, two reads a RAM | 8, true dual-port | 16 hashed, true dual-port |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| b4_q3 | 3.1% | 0.92% | 4.17 | 1.17 | 0.95 | 0.43 | 0.04 | 0.13 | 0.02 |
| b8_q3 | 5.6% | 1.72% | 7.63 | 1.94 | 1.46 | 0.76 | 0.06 | 0.25 | 0.03 |
| d_lf | 3.6% | 0.99% | 4.64 | 2.30 | 1.95 | 0.48 | 0.12 | 0.22 | 0.01 |
| d_q35 | 4.5% | 1.03% | 5.51 | 3.03 | 2.80 | 0.63 | 0.08 | 0.64 | 0.05 |
| d_q3 | 5.2% | 1.51% | 6.72 | 2.64 | 2.27 | 0.63 | 0.12 | 0.31 | 0.03 |
| l_g26 | 5.0% | 0.63% | 5.44 | 1.61 | 1.16 | 0.64 | 0.03 | 0.09 | 0.03 |
| l_q35m | 9.3% | 0.87% | 10.33 | 4.24 | 3.66 | 1.04 | 0.14 | 0.23 | 0.04 |
| p8_q35 | 14.4% | 1.06% | 15.58 | 7.04 | 6.11 | 1.53 | 0.04 | 0.22 | 0.11 |
| p8_q3 | 20.2% | 6.33% | 27.51 | 8.25 | 6.24 | 2.62 | 0.33 | 1.15 | 0.09 |
| p_lf | 12.0% | 2.95% | 14.90 | 5.29 | 4.24 | 1.35 | 0.34 | 0.60 | 0.04 |
| p_q35 | 11.8% | 1.06% | 12.96 | 5.86 | 5.10 | 1.29 | 0.03 | 0.15 | 0.09 |
| p_q3 | 13.6% | 4.02% | 18.30 | 4.77 | 3.87 | 1.59 | 0.22 | 0.62 | 0.05 |
| z_ph_r3 | 3.3% | 0.95% | 4.43 | 1.25 | 1.07 | 0.59 | 0.10 | 0.21 | 0.04 |
| z_q4_r2 | 5.2% | 0.43% | 5.60 | 2.04 | 1.74 | 0.52 | 0.03 | 0.45 | 0.04 |

(`c1`, `s8i`, `s16i`, `s16h2`, `s8i2`, `s8t`, `s16h2t` in `tools/tmem_trace.py`; "true
dual-port": two accesses a RAM, a write taking one.) More sub-banks help slowly (32 interleaved:
0.8-5.7%); a second access per RAM helps at once. Which ports' reads, taken away, remove the
conflicts (8 sub-banks / 16 hashed / 8 true dual-port, mean of 15 traces):

| reads taken away | 8 sub-banks | 16 hashed | 8 true dual-port |
|---|---:|---:|---:|
| none | 3.46 | 0.94 | 0.35 |
| QUANT 3 (the row factor in a register) | 3.39 | 0.90 | 0.34 |
| MXU (its accumulation kept in the MXU) | 3.29 | 0.71 | 0.31 |
| MXU and QUANT 3 | 3.23 | 0.67 | 0.29 |
| QUANT, QUANT 2, QUANT 3 | 2.30 | 0.60 | 0.15 |
| VPU B | **1.28** | 0.43 | 0.19 |

So the conflicts are a unit against itself: the VPU's two operands, streamed in lockstep over
the same sub-bank sequence, and the quantizer's data against its scales.

## 4. The design space

Eight directions, A for the microarchitecture alone, B for those that change the ISA or the
compiler's contract. Section 6 has their measurements, section 8 ranks them.

**A1. A wide line and stream buffers.** Read two words a lane a RAM access (RAMB36 in 512 x 72)
into a small buffer per port; a streaming port then needs its RAM every other cycle, and two
lockstep streams can take turns. In the screen this is "two reads a RAM" (0.03-0.34%, section
3.2) plus the buffers' latency: starts and completions two cycles late (`L2`, `L2i`: the cycles
of a longer read path, any TMEM). The MXU's scattered drain rows and the scalar ports gain
nothing, and every unit's read path gets the buffer's stage. A true dual-port RAM gives the
second access with no buffer and no latency (A1's result without A1's cost), so A1 is dominated
on the FPGA. On an ASIC with 1R1W macros it is the way to a second read (a 64-bit macro).

**A2. Interleaved banks, deterministic schedules.** Sub-banks interleaved by 8-word groups or
hashed (section 3.2) are the probabilistic half. The deterministic half, a rotating schedule
in which port p may read sub-bank (p + t) mod 8 at cycle t, never conflicts but makes a stream
wait up to 7 cycles for its phase at every start and after every hold, and does not fit the
MXU's scattered rows. *Estimate*: 3.5 cycles a start on average, more than `L2`'s 2, for what
the replays (section 5) get for a cycle per rare conflict. Not built.

**A3. Compiler placement.** The colliding pairs are one instruction's operands; a placement pass
could put them in different sub-bank phases. The grant model's skews (`s8ip`: VPU B 32 words,
QUANT 2 24, QUANT 3 40 off VPU A / QUANT's phase, the most a pass could give) halve the one-copy
cost (+0.78% against +1.57%, geomean, section 6.1). Its cost is the compiler's (below); with true
dual-port RAMs the pairs no longer share a RAM port, and A3 has little left to do.

**A4. Fewer readers per unit.** The MXU's accumulation kept in the MXU (92% of its port's words)
and the row factor (QUANT 3) in a register. They hardly matter to conflicts (3.46 -> 3.23,
section 3.2), because the MXU and the row factor do not stream in lockstep with another port.
The readers that matter are VPU B and QUANT 2, and taking them away means B5 or B6.

**B5. Purpose memories.** Split TMEM into memories by role (accumulators, activations,
constants and scales), each one copy with fewer readers. Section 3.1 says the roles mix: VPU A
reads the MXU's results (34%) and the VPU's own (59%), QUANT reads the VPU's (94%), VPU B reads
DMA constants and VPU results; every split moves data between memories (copies, or units
writing to two). It is an ISA change (address spaces), a compiler change (allocation per
memory, fragmentation of 64K words into fixed parts), and RTL in every unit, for what the
dual-port split of section 5 gives without them.

**B6. A register file in the VPU.** Re-reads: VPU A 18%, VPU B 28% (the DMA's constants: 96% of
them read again), QUANT 2 54%, QUANT 3 99%. A VRF or an operand cache would take those reads
away; the lines kept per port of the grant model (`s8iqB`: one line of the last word at QUANT 3,
VPU B and COLL) are its hardware-only form. The ISA form (register operands, the compiler's
register allocation) is a large change to the VPU's instruction set and the kernels for at most
the re-read share.

**B7. Operand-layout rules in the ISA.** A3 made a contract: an instruction's operands must sit
in different sub-bank phases (a rule the compiler keeps and the ISA simulator checks), so the
hardware needs no conflict logic for a unit's own ports. With the dual-port split the colliding
pairs are on different RAM ports already; the rule would only save the replay logic of the
remaining cross-unit conflicts, which are rare. Not worth an ISA rule.

**B8. The MXU's drain into its consumer.** The MXU's results reach VPU A (34% of its words) and
the VPU's reach QUANT (94% of QUANT's) only thousands / hundreds of cycles later (section 3.1):
TMEM is the buffer, the producer and the consumer do not run in step. A direct path would need
that buffer elsewhere. Without QUANT's three ports the one-copy cost falls by a third (3.46 ->
2.30, section 3.2); with the dual-port split, from 0.35 to 0.15. Fused producer-consumer
instructions are a compiler and kernel change for that remainder.

**The hardware answer.** One copy of sub-banked RAMs whose conflicts are settled by a replay
behind the board's arbiter (section 5), and, on true dual-port RAMs, with a unit's two operand
ports on different RAM ports. No ISA change. The ISA options above each take away a part of what
is left, at a compiler cost:

| option | compiler / ISA work | what it can take away (one copy, true dual-port, screen) |
|---|---|---|
| A3 placement | a phase-aware allocation pass (`opentpu/compiler.py` alloc / `_tmem_fit`), ~200-400 lines, and a checker; more fragmentation | little: the pairs it separates are on different RAM ports already |
| B5 purpose memories | address spaces in the ISA, allocation per memory, copies between them, every unit's RTL | the split's result, which the dual-port split has |
| B6 VRF | register operands in the VPU's ISA, register allocation in the kernels | at most the re-read share; 0.35 -> 0.19 without VPU B at all |
| B7 layout rule | an ISA rule, the compiler keeping it, the ISA simulator checking it | the replay logic of a unit's own ports |
| B8 drains | fused instructions, kernels restructured | 0.35 -> 0.15 without any QUANT read |

## 5. Replays: one copy behind the board's arbiter (`TMEM_RPL`)

`TMEM_IMPL 3 TMEM_RPL 1` (`otpu_tmem` RPL, `otpu_slice` TMEM_RPL; 1cb7d4a7, c074abab, 774b3163).
The slice keeps the board's arbiter (`ARB_MASK`: write-bank masks only, the VPU's write buffer),
so no RAM conflict ever reaches a grant: every unit reads as if it had its own copy. The conflict
is settled after the fact.

- **The RAMs.** One copy; each lane-bank is NSB RAMs (8: 1K words each at 64K, one RAMB36), the
  sub-bank = bank-row bits [0 +: 3] (8-word groups interleaved over the RAMs), optionally XORed
  with the higher row fields (SBH 2). A RAM's address is that of its first asker in rank, taken
  from the requests (no grant on the RAM's address path, as `TMEM_RQ`). Two askers of one word
  share the read.
- **The askers.** A port asks with its request, or with its replay while one is pending (the unit
  then holds, so its request does not ask): one asker a port. The DMA's request ranks first (it
  cannot hold: its data arrive from DRAM), then the replays, then the other requests (the
  groups' order: COLL, VPU A, VPU B, MXU, QUANT, QUANT 2, QUANT 3). A replay ranks before every
  request but the DMA's, so it wins the next cycle unless the DMA or an older replay takes the
  same RAM for another word. Among replays the oldest ranks first; those that began in one cycle
  keep the groups' order (section 5.3).
- **A lost read.** An asker loses when an asker before it takes one of its RAMs for another word.
  Its port's banks, rows and lanes are registered with the read (`rpe`, `rpa`, `rpl`); the next
  cycle they ask again (`frz`), and the cycle after the replay wins its words reach the port's
  lanes (`rpf`: the lanes fresh again, the rotation and the crossbar's selects kept, the bypass
  off).
- **The unit holds** for the cycle the replay takes: the MXU and QUANT through their grant
  (`gnt & !frz`, `frz` a register), the VPU through its registered enable (`otpu_vpu rhold` into
  `en_r`, the cycle after a lost read). The collective is not replayed (its far side cannot
  hold): it reads only where its request does not lose (`tm_bx`), as the banked arbiter would.
- **True dual-port** (`TMEM_TDP`, 774b3163). Each RAM's two ports, every read port on one of them:
  the MXU, the DMA, QUANT and VPU A on port B; VPU B, QUANT 3, QUANT 2, COLL and the writes on
  port A. Askers rank among their side only. A write takes its RAM's port A; a port-A asker of a
  RAM being written loses unless it reads the written word (a read takes the bypass, a replay
  port A's read-first word). So the pairs that collide in lockstep (VPU A / VPU B, QUANT /
  QUANT 2) are on different RAM ports and never meet. A 7-series true dual-port RAMB36 is 1K x 36
  at most, which is what 8 sub-banks of a 64K TMEM need: 64 RAMB36, as the simple dual-port.
- **Guests** (`TMEM_GST`, ced34f1b). QUANT 3 and COLL read only while the DMA, and the guests
  before them, ask nothing (today's rule: the slice's `q3_blk` / `coll_blk`), so they need no
  asker of their own: the DMA's asker asks for the first reading guest, and the guests take the
  DMA's data. Six askers instead of eight, one data select for three ports, and the same cycles
  as today's TMEM for those two ports. The DMA ranks first, so a guest's read is never lost. A
  guest asks only when that rule lets it read: while QUANT holds for a replay, QUANT 3's request
  asks nothing but still keeps COLL from reading, so COLL does not ask either (section 5.3).
- **Pumped** (`TMEM_PMP`, 30cbe6e0). Each RAM is P3's pumped bank (`otpu_pump_bank` on a 2x
  clock, `docs/tmem_pump.md`): both RAM ports read at the core edge (port B side B, port A side
  A), and the write lands at the fast edge between. A write takes no read port, so there are no
  write conflicts, and a read sees the landing write in the RAM, so there is no bypass (no
  address compares, the data select alone). The premise of section 5.1 then also covers the
  replay's own cycle t' (a write landing then would be seen); the ordering keeps it out as well.
  `TMEM_PMP 2` is its model at the core clock (a write port of its own, the bypass kept), the
  same words and cycles without the second clock, for co-simulation.
- **One RAM a bank** (`TMEM_NSB 1`). No sub-banks: each side of a lane-bank is one read, so two
  askers of one bank's side for different words conflict. No data select and no sub-bank
  compares; with the pump and the guests, the conflict logic is a compare of bank masks per side.

### 5.1 The ordering a replay needs

A replay reads the RAM later than the read it repeats. It returns that read's word only if the
word did not change in between. Exactly: for a read of word w at cycle t whose replay wins at
t' > t, **no write of w may land in cycles t+1 .. t'-1**. (A write landing at t is one the read
was to see: it took the bypass, and the RAM has it by t'. A write landing at t' is fine: the
RAM reads first.)

Who could write w in that window:

- **The reading unit itself.** It is frozen from t+1 until the replay wins. Its pipeline's
  writes still landing are of elements it read earlier (a stream reads forward; an instruction
  does not write a word it has still to read), and its next instruction has not started.
- **Another unit's instruction.** That is a write-after-read hazard between instructions, and
  it is what the hazard logic orders: the write's instruction must not start before the reading
  instruction is done. The reading instruction's done (`udone`) comes after its last replay (the
  unit is held until the replay wins), so "after the reader's done" covers the replays.
  - Today's sequencer (`otpu_seq`'s scoreboard) holds an instruction whose writes overlap an
    unfinished older instruction's reads until that one is done.
  - WAITC (seq-vpu, otpu_seq `WAITC` with the compiler's wait counts, `opentpu/waits.py`,
    8f6b749e) gives the same ordering: a wait gates the instruction's start; the counts count
    `udone`; `conflict()` covers RAW, WAR and WAW across units; same-unit waits are dropped
    except the VPU's. seq-vpu confirmed each point (2026-10-12). The caveat is theirs: the
    oracle counts (`KOUT`/`SYNC`) are a minimum for one run; `waits.py`'s static counts are the
    sound ones and are the deployable version.
- **The DMA's reads are never replayed** (it ranks first), so streams from DRAM need nothing.

The simulation checks the premise: `otpu_tmem` stops (`$fatal`) when a write lands on a word whose
replay is pending while that replay loses again (c074abab). No run here hit it, with the
scoreboard or with WAITC (section 6.4).

### 5.2 Tests

- `tests/test_rtl.py::test_tmem_random_traffic` (21 cases): `tb_tmem` drives random traffic
  against a reference memory, most reads at the words written the cycle before (the bypass),
  with the scoreboard's rule (no write to a word read that cycle or awaiting a replay); a port's
  lanes are checked every cycle except those its read or replay lost; at least 1,000 replays a
  run. Cases: today's TMEM, the banked one, IMPL 3 at three sub-bank layouts and hashed, the
  replays at three, true dual-port with and without replays, guests (a guest's group waits
  while its host asks; at least 1,000 guest reads), and pumped at 4, 2 and 1 RAMs a bank on a
  real 2x clock (`PMP 1`) and its model (`PMP 2`).
- Mutations: the lanes not refreshed by a replay, the replay's data select not loaded, the
  write's port-A conflict dropped, the host ignoring its guest's request, the core-clock model
  without its bypass: each fails within 12 cycles.
- The co-simulations of section 6 are checked against the ISA simulator.

### 5.3 Forward progress

A replay must win in the end. Two things can keep it losing: an asker ranked before it on its
RAM side that asks every cycle for a read that never happens, and younger replays that keep
coming back.

**The hang of release r2 (2026-10-13).** Under the r2 target (`OTPU_UARCH=board`),
`test_fuzz_mm_replay[2-8-1]` and `[2-6-3]` ran to 50M cycles without halting. TMEMSTAT showed
QUANT 2's replay losing every cycle, and VPU B's too in [2-8-1]. The cause was the guests:

1. QUANT holds while QUANT 2 replays. Its requests then ask nothing (the slice's `off` mask), but
   QUANT 3's request stays up.
2. The slice's guest rule (`coll_blk`, from the requests) keeps the collective from reading while
   QUANT 3 requests, so a pending collective read (`coll_ren`) is never granted.
3. `otpu_tmem`'s guest chain works from the asks. QUANT 3 asks nothing, so the DMA's asker, first in
   rank, asks for the collective's words, every cycle.
4. Those words sat in a RAM the replays needed, on port B: the DMA's side is also QUANT 2's and
   VPU B's. The replays lost to it every cycle, so QUANT never resumed and QUANT 3's request
   stayed up. A deadlock.

A debug print at the hang: [2-8-1], cycle 2,288, QUANT 2 (word 660) and VPU B (word 2,748)
replaying; the collective's words 4,568-4,575 in the same RAMs of banks 0-3; `coll_blk` 1, the
collective never granted. WAITC only moves the schedule: without it [2-6-3] still hangs and [2-8-1]
does not. The simulators' default TMEM (`TMEM_IMPL 0`) has no replays, so only the board's uarch
showed it.

**The fix: a guest asks only when it may read.** The slice keeps the collective's request out of the
asks while `coll_blk` holds. QUANT 3 already asks only when the DMA does not, which is `q3_blk`. So
every ask of the DMA's asker is a read that happens, unless that read waits on something no replay
holds up:
- the DRAM's write room for a QST (`sw_rdy`);
- another slice's grant of the collective;
- the DMA's write banks.

**The age order: a guarantee among the replays.** The replays had a fixed order (PRANK). Two ports
taking turns could then hold off a third. A replay that loses elsewhere still takes the RAMs it
asks first, so with QUANT 2's replay pending:
1. Cycle t: VPU B's replay beats QUANT 2's in one bank, and the MXU's read loses to QUANT 2's
   replay in another.
2. Cycle t+1: the MXU's replay beats QUANT 2's, and VPU B's next read loses to QUANT 2's replay.
3. Cycle t+2 is cycle t again.

Such a run ends only when one of the two units finishes its instruction; nothing bounds it. Now,
of two replays the older ranks first: a register per pair of replay ports on one RAM side, set as
a replay begins, and replays that begin in one cycle keep PRANK's order. With two replay askers on a
side, the fixed order cannot starve one without the DMA's asker every other cycle, so the age is kept
only on a side with three or more: the pick's port B (the MXU, VPU B and QUANT 2; the guests have no
asker). A replay then loses only to two things:
- the DMA's asker;
- older replays.

The oldest replay wins as soon as the DMA's asker leaves its RAMs. In the unpumped ASIC form a write
on port A also takes its RAM. None of these waits on the replaying unit. The simulation stops when
a replay loses 4M cycles in a row (`otpu_tmem`, the `RPLW` check).

**Measured** (the simulation host, the r2 target through `OTPU_UARCH=board`; the build host, Vivado 2026.1):

| | release-2 (5107b74f) | the guest fix alone | the fix (guests + age) |
|---|---|---|---|
| test_rtl's fuzzers' generators, 296 programs (seeds 0-39 of five mm shapes, 0-63 one slice, 0-31 two) | 34 hang | | none hangs, all bit-exact |
| the same at more seeds, 856 programs (0-119, 0-191, 0-63) | | | all bit-exact (one run failed before its simulation started; bit-exact on a rerun) |
| the mm fuzzers, 160 programs | | all bit-exact | |
| `test_fuzz_mm_replay[2-8-1]`, `[2-6-3]` | hang (50M cycles) | | 2,637 and 3,537 cycles, bit-exact |
| the age order alone (no guest fix), those two | | | still hang: the 4M-cycle check stops them |
| the 17 co-simulations of `all.txt` (the r2 target, real 2x clock), cycles | | | geomean +0.00% (12 identical, -0.03% to +0.05%); every checked run bit-exact |
| ... extra TMEM reads (replays / reads) | 1.39-6.94% | | 1.39-6.94% (each within 2.5% of release-2's count) |
| `k7-r2pgRs` out of context, LUTs | 13,271 | 13,314 | 13,551 |
| WNS / WHS at 7.5 / 3.75 ns | +0.232 / +0.071 | +0.071 / +0.068 | +0.281 / +0.056 |

The worst path of all three is the write stage into a RAM at the fast edge (0 or 1 level), as
section 6.3's; it moves with placement. The age costs about 240 LUTs, the guest fix about 40.

## 6. Measurements

Cycles are RTL co-simulations against the card's TMEM (`i0`, the same tree), in percent; every
`perf_qwen` run is bit-exact against the ISA simulator (`--check`; the MoE layer run `l_q35m` has
zero images and is not checked). Out-of-context runs are Vivado 2026.1 on the build host, `tmem_nc_top`
(the TMEM and the slice's arbiter for it, registered around), xc7k480t-2.

### 6.1 The grant model and the first RTL

The grant model (`TMEM_IMPL 2`: the units feel the conflicts, the data stay in copies), on
screen.txt:

| candidate | p_q3 | p_q35 | p8_q3 | b4_q3 | l_q35m | d_lf | geomean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `x8`: a copy per port, through the model | +0.27 | -0.03 | -0.18 | -0.03 | +0.02 | -0.02 | +0.00 |
| `s8i`: one copy, 8 sub-banks, the grant | +1.21 | +4.94 | +1.49 | +0.33 | +0.87 | +0.66 | +1.57 |
| `s8iq`: ... request-priority addresses | +1.03 | +4.93 | +1.69 | +0.32 | +0.85 | +0.70 | +1.58 |
| `s8ip`: ... operands placed apart (A3) | +0.68 | +1.67 | +2.15 | +0.17 | -0.16 | +0.18 | +0.78 |
| `s16i`: 16 sub-banks | +0.81 | +4.55 | +1.26 | +0.25 | +0.71 | +0.47 | +1.33 |

With the grant in the RAM's address path (`TMEM_IMPL 3` without replays) the timing goes the way
of `docs/tmem_pump.md`'s banked arbiter: `k7-r8i` +0.019 ns, `k7-r16i` -0.715 ns out of context
(21.7K and 38.4K LUTs). The replays (section 5) take the conflict out of the grant.

The replays at 8 sub-banks (`R`, simple dual-port, no guests) on the model list: p_q3 +1.01,
p_q35 +5.08, p8_q3 +2.16, b4_q3 +0.40, d_lf +0.88, d_q3 +0.26; geomean +1.62%. Out of context
`k7-r8iR` is 23.5K LUTs at -0.534 ns, `k7-r8xR` (true dual-port) 21.4K at +0.498.

### 6.2 What the LUTs are

Synthesis only (`s-*`, opt_design), each ablation removing one part (`OTPU_TMEM_ABL_*`; the
parts overlap, so the savings do not add up exactly):

| design | LUT | data select | conflict compares | replay buffers | address priority mux |
|---|---:|---:|---:|---:|---:|
| today (`v0`, 6 copies) | 11,909 | | | | |
| P3 (3 copies pumped) | 10,297 | | | | |
| `r8iR`: 8 sub-banks, simple dual-port | 23,070 | -4,288 | -2,885 | -849 | -3,963 |
| `r8xR`: 8 sub-banks, true dual-port | 21,384 | -3,340 | -766 | -619 | -1,578 |
| `r2pgR`: 2 RAMs a bank, pumped, guests | 13,475 | -1,079 | -1,178 | -576 | -542 |
| `r1pgR`: 1 RAM a bank, pumped, guests | 11,362 | (none) | -872 | -192 | -351 |

The sub-banks are what cost: an 8:1 data select per port and lane, a sub-bank compare per pair
of askers, an address multiplexer per RAM. True dual-port halves the compares (askers rank on
their side only) but adds the write's port-A conflict. Pumping removes the write conflict and
the bypass; guests remove two askers; one RAM a bank removes the data select and leaves a bank
compare per pair.

### 6.3 Out of context

Placed and routed, xc7k480t-2, 133.33 MHz (and 266.67 MHz for the pumped RAMs, from an MMCM in
the harness, P3 the same):

| design | RAMB36 | LUT | FF | WNS (ns) | the worst path |
|---|---:|---:|---:|---:|---|
| today (`k7-v0m`, 6 copies) | 384 | 11,885 | 7,989 | +0.587 | |
| P3 (`k7-p3`) | 192 | 10,079 | 6,432 | +0.328 | the fast write stage's enable (0 levels) |
| `k7-r8iR` | 64 | 23,459 | 7,621 | -0.534 | |
| `k7-r8xR` | 64 | 21,363 | 7,708 | +0.498 | |
| `k7-r4xR` | 64 | 16,763 | 7,526 | +0.210 | |
| `k7-r2xR` | 64 | 15,583 | 7,492 | +0.234 | |
| `k7-r2pgR` | 64 | 13,355 | 7,053 | +0.173 | a replay's rank into a RAM's address (7 levels) |
| `k7-r1pgR` | 64 | 11,258 | 6,986 | +0.280 | a RAM's word into QUANT's lanes (2 levels) |
| **`k7-r2pgRs`** (the pick) | **64** | **13,271** | **7,050** | **+0.232** | the write stage's address into a RAM at the fast edge (1 level) |
| `k7-r1pgRs` | 64 | 11,447 | 6,983 | +0.344 | the write stage's word into a RAM at the fast edge (0 levels) |
| `k7-r1pgR-128k` | 128 | 11,711 | 7,031 | +0.097 | the write stage's row into the RAM at the fast edge (1 level) |
| `k7-r2pgR-128k` | 128 | 13,325 | 7,126 | +0.254 | |
| `k7-r1pgR-256k` | 256 | 11,801 | 7,133 | +0.009 | |
| P3 at 128K (`k7-p3-128k`) | 384 | 10,610 | 6,471 | +0.032 | |

On xcku5p-2 at 200 / 400 MHz: `ku-p3` +0.131 ns (180 block RAM tiles), `ku-r1pgR` +0.056 (60),
`ku-r2pgR` +0.148 (64). The sides of section 6.4 change only constants: `k7-r2pgRs` against
`k7-r2pgR` -84 LUTs and +0.059 ns, `k7-r1pgRs` against `k7-r1pgR` +189 LUTs and +0.064 ns (place
and route's spread). Both meet the bars' +0.1 ns; the worst paths are now the write stage into
the RAMs at the fast edge, a fast period by design.

### 6.4 The one copy: the ASIC form, then the pumped FPGA form

**The ASIC form (the headline).** On the ASIC the copy is two-port SRAM macros at the core clock
(`TMEM_TDP` without `TMEM_PMP`). A bank's two accesses in a cycle are both reads only when no
write lands. A write takes port A, and a port-A read of another word in that RAM loses and is
replayed. The runs below use 2 RAMs a bank (the two 4K x 32 macros a 64K bank is anyway), replays,
QUANT 3 and COLL as the DMA's guests, on the screen against `i0`, without WAITC:

| workload | `i0` cycles | `r2xgR` | extra TMEM reads | `r2xgRs` | extra TMEM reads | (FPGA: `r2pgRs`) |
|---|---:|---:|---:|---:|---:|---:|
| p_q3 | 372,383 | +2.48 | 33.6% | +7.93 | 54.1% | +1.12 |
| p_q35 | 1,137,275 | +6.05 | 24.7% | +8.39 | 36.5% | +0.67 |
| p8_q3 | 502,925 | +4.26 | 42.2% | +12.70 | 67.6% | +1.66 |
| b4_q3 | 1,671,279 | +0.87 | 26.3% | +1.99 | 42.3% | +0.25 |
| l_q35m | 679,296 | +1.81 | 24.7% | +3.51 | 34.4% | +0.24 |
| d_lf | 1,339,842 | +1.12 | 31.3% | +2.32 | 38.2% | +0.25 |
| geomean | | **+2.75** | | +6.07 | | +0.70 |

Every run is bit-exact.
- **Cost.** The ASIC form costs +2.75% in cycles (p_q35 +6.05%, p8_q3 +4.26%). By the measure that
  counts on the ASIC it costs 25-42% extra TMEM reads, five times the pumped copy's 3-8%. A write
  takes one of a two-port macro's two accesses, so whenever a write lands the bank has one read
  slot instead of the pump's two.
- **Sides.** The sides run opposite to the pump's. Port A, with the writes, should carry the light
  ports: VPU B, QUANT 3, QUANT 2 and COLL (`r2xgR`, the unpumped default, `TM_SIDED`). Moving QUANT
  and VPU A there (`r2xgRs`, the pump's sides) more than doubles the cost: +6.07%, 34-68% extra
  reads.
- **Not measured on the ASIC form.** Two levers would take the write's port away from the reads:
  - fewer consumers, which removed 98-99% of the pumped copy's replays (section 6.6);
  - a write stage that lands writes in cycles where port A reads nothing (a write can wait; a read
    cannot).

**The pumped FPGA form (the board build).** One copy pumped at 2x (`TMEM_PMP`; co-simulated with
its core-clock model, `PMP 2`), true dual-port, replays, QUANT 3 and COLL the DMA's guests. Without
WAITC, against `i0`:

| candidate | p_q3 | p_q35 | p8_q3 | b4_q3 | l_q35m | d_lf | geomean | extra TMEM reads |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `r1pgR`: 1 RAM a bank, old sides | +1.48 | +1.25 | +2.89 | +0.50 | +0.90 | +0.65 | +1.28 | 10.8-25.3% |
| `r1pgRs`: 1 RAM, sides s | +0.95 | +0.87 | +2.53 | +0.43 | +0.64 | +0.55 | +0.99 | 6.3-19.7% |
| `r1pgRsm`: 1 RAM, sides s, MXU first | +1.22 | +0.99 | +2.23 | +0.44 | +0.73 | +0.64 | +1.04 | 6.1-19.7% |
| `r1pgRm`: 1 RAM, old sides, MXU first | +1.15 | +1.38 | +2.32 | +0.47 | +1.05 | +0.74 | +1.19 | 9.9-23.4% |
| `r2pgR`: 2 RAMs a bank, old sides | +1.86 | +0.89 | +2.50 | +0.39 | +0.28 | +0.27 | +1.03 | 5.3-14.5% |
| **`r2pgRs`: 2 RAMs, sides s** | **+1.12** | **+0.67** | **+1.66** | **+0.25** | **+0.24** | **+0.25** | **+0.70** | **2.8-8.2%** |
| `r2pgRsm`: 2 RAMs, sides s, MXU first | +0.89 | +0.69 | +0.73 | +0.22 | +0.37 | +0.28 | +0.53 | 3.8-9.5% |
| `r2pgRm`: 2 RAMs, old sides, MXU first | +1.08 | +0.91 | +1.07 | +0.28 | +0.61 | +0.34 | +0.71 | 6.1-17.1% |

*Extra TMEM reads*: the replays as a share of the reads, each a second read of a RAM (the slice's
TMEMSTAT `rd=` and `rpl=`: the cycles each read port reads and those it loses; `r1pgR`'s range
takes in the model list's runs of the same TMEM, `P1`, whose cycles are identical). *Old sides*:
VPU B, QUANT 3, QUANT 2 and COLL on RAM port A. *Sides s* (the screen's best of all 32 splits,
`tools/tmem_trace.py`-style replay of 15 traces): QUANT and VPU A on port A; the MXU, the DMA and
its guests, VPU B and QUANT 2 on port B. *MXU first*: the MXU's request ranked right after the
DMA's (`OTPU_TMEM_PRK=6014785`; default: COLL, VPU A, VPU B, the MXU, QUANT).

Each port's replays, as a share of its own reads:

| read port | `r1pgR` p_q3 | `r1pgR` p8_q3 | `r2pgRs` p_q3 | `r2pgRs` p8_q3 |
|---|---:|---:|---:|---:|
| MXU | 29.8% | 40.3% | 6.5% | 9.1% |
| QUANT (data) | 46.6% | 54.5% | 9.8% | 14.1% |
| VPU A | 18.0% | 21.7% | 2.4% | 2.9% |
| VPU B | 0.5% | 0.3% | 1.1% | 1.6% |
| QUANT 2 (scales) | 1.8% | 1.2% | 37.0% | 38.3% |
| DMA, QUANT 3, COLL | 0 | 0 | 0 | 0 |

(The DMA ranks first and never loses; its guests ride its reads.)

**Where the replays come from.** Most of the replays come from one RAM port (QUANT's data and the
MXU on port B), so one copy does not fail for want of read slots; the operands are badly placed
across them. Moving QUANT's data and VPU A to port A (QUANT 2, few reads, to
port B, away from QUANT) takes 40-65% of the replays away (2 RAMs a bank: 5.3-14.5% -> 2.8-8.2% of
the reads) and a third of the cycles, for no RAM and no logic: the sides are constants. Ranking the
MXU first moves the losses from the MXU to VPU B and QUANT 2: fewer cycles at MCOLS 8 (p8_q3 +1.66
-> +0.73) but more re-reads, so the rank stays (`TMEM_PRK` keeps it as a knob). A second RAM a bank
halves what is left. What placement cannot remove is the traffic of the units themselves on
port B: QUANT's data against the MXU's accumulation, the one pair still losing 6-14% of its reads.

### 6.5 With WAITC

WAITC v1 (in-order, `wcr`) and the replays at 8 sub-banks (`R`), against `i0` on the model list:

| workload | R (replays) | wcr (WAITC v1) | wcrR (both) |
|---|---:|---:|---:|
| p_q3 | +1.01 | +5.07 | +7.46 |
| p8_q3 | +2.16 | +8.69 | +12.34 |
| p_q35 | +5.08 | +1.24 | +6.29 |
| b4_q3 | +0.40 | +1.36 | +1.89 |
| d_lf | +0.88 | +0.45 | +1.57 |
| d_q3 | +0.26 | +0.18 | +0.08 |
| geomean | +1.62 | +2.79 | +4.85 |

So p_q3's +7.5% was +1.0 for the replays and +5.1 for WAITC v1 (+1.4 more together). WAITC v2
(`wcm`, ffd1140c: 64-bit wait entries with a VPU mask) costs +0.13% alone; with the pumped one
copy at 1 RAM a bank (`wcmP1`, default sides):

| workload | P1 vs i0 | wcm vs i0 | wcmP1 vs i0 | wcmP1 vs wcm |
|---|---:|---:|---:|---:|
| p_q3 | +1.48 | -0.14 | +1.60 | +1.75 |
| p_q35 | +1.25 | +0.23 | +1.42 | +1.19 |
| p8_q3 | +2.89 | +0.79 | +4.37 | +3.55 |
| b4_q3 | +0.50 | +0.09 | +0.68 | +0.59 |
| d_lf | +0.65 | +0.18 | +0.90 | +0.72 |
| d_q3 | +0.68 | +0.13 | +0.53 | +0.40 |
| dr_q3 | -0.05 | -0.38 | +0.41 | +0.79 |
| geomean | +1.05 | +0.13 | +1.41 | +1.28 |

The pick under WAITC v2 (`wcmP2s`: `r2pgRs` in the WAITC v2 tree, b5d6bcef + ffd1140c) on the full
model list (`all.txt`'s 17 workloads and wr.txt's dr_q3), against WAITC v2 with today's TMEM (`wcm`) and against today's
TMEM without WAITC (`i0`):

| workload | `i0` cycles | wcm vs i0 | wcmP2s vs i0 | **wcmP2s vs wcm** | extra TMEM reads |
|---|---:|---:|---:|---:|---:|
| p_q3 | 372,383 | -0.14 | +0.59 | +0.73 | 3.9% |
| p_q35 | 1,137,275 | +0.23 | +0.56 | +0.33 | 1.9% |
| p_lf | 1,630,055 | -0.02 | +0.64 | +0.66 | 4.9% |
| p8_q3 | 502,925 | +0.79 | +2.52 | +1.71 | 5.6% |
| p8_q35 | 942,492 | +0.69 | +0.95 | +0.25 | 1.6% |
| b4_q3 | 1,671,279 | +0.09 | +0.38 | +0.28 | 4.1% |
| b8_q3 | 1,827,502 | +0.15 | +0.58 | +0.43 | 3.8% |
| d_q3 | 3,623,644 | +0.13 | +0.37 | +0.24 | 4.0% |
| d_q35 | 4,760,853 | +0.05 | +0.36 | +0.31 | 4.7% |
| d_lf | 1,339,842 | +0.18 | +0.54 | +0.36 | 3.2% |
| z_q4_r2 | 2,225,099 | +0.12 | +0.34 | +0.22 | 1.6% |
| z_ph_r3 | 1,677,624 | +0.08 | +0.29 | +0.21 | 6.7% |
| z8_q2_r4 | (none) | | | +0.38 | 2.0% |
| l_q35m | 679,296 | +1.79 | +2.01 | +0.22 | 2.6% |
| l_g26 | 1,963,377 | +0.67 | +0.83 | +0.17 | 3.1% |
| m_q35m | 15,297,819 | +0.08 | +0.10 | +0.02 | 3.1% |
| m_g26 | 3,886,066 | +0.26 | +0.41 | +0.15 | 1.3% |
| dr_q3 | 3,694,791 | -0.38 | -0.27 | +0.10 | 3.9% |
| geomean | | +0.28 | +0.66 | **+0.38** | |

On top of WAITC v2 the pick costs +0.38% geomean. `wcmP1` cost +1.28% on the seven workloads of
the table above; the pick has two RAMs a bank and the screen's sides. The worst case is p8_q3 at
+1.71% (the 8-row prefill at MCOLS 8, 5.6% extra TMEM reads), then p_q3 +0.73% and p_lf +0.66%. The
MoE layers and models (`l_`, `m_`) cost +0.02 to +0.22%. With WAITC v2 the total is +0.66% against
today's TMEM without WAITC: l_q35m's +2.01% is WAITC's (+1.79%), not the TMEM's. The extra TMEM
reads are 1.3-6.7% of the reads. Every run is bit-exact. z8_q2_r4 has no `i0` run (`i0` ran
z8_q4_r4, whose 4 rows at MCOLS 8 do not fit 64K words).

### 6.6 Fewer consumers (no feedback)

The same replay of 15 traces with read ports taken off TMEM (their words from elsewhere: the
MXU's accumulation kept in the MXU, QUANT fed from the MXU's drain or the VPU's write, the scales
from a buffer), the best sides for each; extra cycles in % (mean / worst over the traces) and the
extra RAM reads in % of the reads:

| consumers off TMEM | 1 RAM a bank | extra reads | 2 RAMs a bank | extra reads |
|---|---:|---:|---:|---:|
| none | 1.66 / 6.14 | 5.0% | 0.84 / 2.79 | 1.4% |
| QUANT's data | 0.74 / 2.52 | 2.0% | 0.46 / 1.45 | 0.8% |
| the MXU's accumulation | 1.39 / 5.09 | 4.9% | 0.58 / 1.53 | 1.3% |
| VPU B | 1.42 / 5.74 | 4.8% | 0.67 / 2.57 | 1.2% |
| QUANT's data and scales | 0.28 / 0.86 | 1.1% | 0.19 / 0.57 | 0.5% |
| the MXU's accumulation and QUANT's data | 0.28 / 0.77 | 1.6% | 0.20 / 0.53 | 0.6% |
| the MXU's accumulation, QUANT's data and scales | 0.14 / 0.70 | 1.0% | 0.11 / 0.49 | 0.4% |

(No feedback: a conflict costs its cycle and nothing moves, which overstates the co-simulated cost
two to four times, section 6.4 against 3.2; the ranking holds. The extra reads here count lane
reads, the co-simulation's port-cycles.) QUANT's inputs are the lever: without them a single RAM a
bank comes within 0.3% on average. That is the `dataflow` study's quantize-on-drain and fused
VPU-to-QUANT path; it takes consumers off TMEM, which is the direction the ASIC wants anyway (each
TMEM read is SRAM energy and port time).

**Measured with feedback (`dataflow`, `docs/dataflow.md` 5.3).** The `dataflow` study co-simulated
the pick (`r2pgRs`, b5d6bcef) against today's TMEM (`i0`) on Qwen3-0.6B, 2 layers, with the same
programs under its fusion transforms. The modes:
- *base*: the programs as compiled;
- *AS*, *ASB*: the fused VOPs as spans, by class;
- *ASB,Q*: also every QACT as a 1-row, 1-block QACT, standing in for quantize-on-drain;
- *C*: every VOP spanned.

| run | mode | `i0` cycles | the copy's cost | replays (MXU / Q / VA / VB / Q2) |
|---|---|---:|---:|---|
| 4 rows, MCOLS 4 | base | 372,383 | +1.12% | 12,632 (3,084 / 3,115 / 1,484 / 409 / 4,540) |
| 4 rows, MCOLS 4 | ASB | 350,081 | +0.27% | 2,966 (471 / 152 / 18 / 3 / 2,322) |
| 4 rows, MCOLS 4 | ASB,Q | 344,742 | -0.00% | 200 (98 / 86 / 14 / 2 / 0) |
| 8 rows, MCOLS 8 | base | 502,925 | +1.66% | 29,631 (6,552 / 8,944 / 3,582 / 1,146 / 9,407) |
| 8 rows, MCOLS 8 | ASB | 452,188 | +0.24% | 6,252 (1,108 / 250 / 19 / 18 / 4,857) |
| 8 rows, MCOLS 8 | ASB,Q | 444,438 | -0.15% | 237 (115 / 106 / 12 / 4 / 0) |

The single copy then costs nothing. With the fold and quantize-on-drain (*ASB,Q*) the replays fall
98-99%. Fusing the softmax numerators out of the VPU (*AS* to *ASB*) removes VPU A's and VPU B's
replays and most of the MXU's and QUANT's. QUANT 2's go with quantize-on-drain: they are QACT's
scale operand, QUANT's own input. Spanning every VOP but leaving the QACTs (*C*) keeps +0.4-0.5%,
the QUANT traffic. So the screen's lever holds with feedback: take QUANT's inputs off TMEM and port
B has nothing left to lose.

### 6.7 With the MX column

The pick in the MX column's tree: mx-column 4406fbed (the MX MXU, E 4) merged with this branch at
b5d6bcef, its parameter-list conflicts resolved by keeping both sides (section "Merging"). It runs against
that tree's own TMEM (`i0n`), no WAITC, on six workloads, three of them at MCOLS 8:

| workload | `i0n` cycles | `r2pgRs` vs `i0n` | extra TMEM reads |
|---|---:|---:|---:|
| p_q3 (4 rows) | 373,923 | +0.91% | 5.7% |
| p_q35 (4 rows) | 1,148,266 | +0.69% | 2.4% |
| p8_q3 (8 rows, MCOLS 8) | 504,388 | +1.60% | 7.8% |
| p8_q35 (4 rows, MCOLS 8) | 952,090 | +0.68% | 2.5% |
| b8_q3 (batch 8, MCOLS 8) | 1,798,064 | +0.57% | 4.6% |
| d_lf (decode) | 1,355,181 | +0.18% | 2.9% |
| geomean | | +0.77% | |

All runs are bit-exact. The MX column changes the MXU's lanes and its drain, not which units meet on
port B, so the cost is the screen's: +0.77% against +0.70%, and the worst case is again the 8-row
prefill at MCOLS 8 (p8_q3 +1.60% against +1.66%).

## 7. Interactions

**WAITC** (seq-vpu's wait counts instead of the scoreboard). The replays need the ordering of
section 5.1, which WAITC gives as the scoreboard does; the combined runs are in section 6.5. Two
cautions from them: the first WAITC versions ran every unit in order and cost prefill 4-12% by
themselves (seq-vpu's own finding, which our oracle runs reproduce: p_q3 +7.5%, p8_q3 +12.3% with
the replays, +1.0% for the replays alone on p_q3), so the combined cost has to be read against
WAITC alone, not against the card; and WAITC v2 (ffd1140c) changes the VPU's issue order, so a
TMEM's conflicts move with it. The ordering argument does not depend on the version.

**The MX column** (4406fbed: MCOLS 4, then 8 rows with ACT_ROWS 16). The TMEM's ports and their
widths do not change with the MX column (its MXU reads and drains through the same port; the
quantizer's X mode adds no TMEM port), so a TMEM option carries over; what changes is the traffic:
8 rows double the MXU's drain and the quantizer's reads per weight pass. The screen's MCOLS 8 runs
(p8_q3, p8_q35) are the heaviest for every one-copy candidate (section 3.2), so MCOLS 8 is where a
one-copy TMEM costs most; section 6.7 has the MX column's own runs.

**A 128K or 256K TMEM.** The RAM scales with the words (64 RAMB36 a copy per 64K), the address
bits with them, the conflict logic not at all: one copy at 128K is 11.7K LUTs at +0.097 ns
(`k7-r1pgR-128k`, the write stage's row into the RAM at the fast edge), 13.3K at +0.254 at 2 RAMs a
bank, 256K 11.8K at +0.009. Whether the words pay is a question of bytes. On today's kernel a
bigger TMEM buys rows a weight pass, and a row's share of the weights' DRAM bytes falls with them:
`docs/tmem_pump.md` measured -9 to -22% cycles a prompt row at 128K for the models that stop at 2 or
3 rows (Qwen3.5-4B, Phi-4-mini; 3 -> 4 rows at MCOLS 8 is a quarter fewer weight bytes a row), and
nothing for decode, batched decode or MoE. The tile kernel (below) gets more rows than that out of
64K: T = 8-24 rows by kernel structure, not TMEM size, so fewer weight bytes a row than today's
kernel at 256K, and 128K or 256K adds nothing to it. So with the tile kernel a larger TMEM is SRAM
without a byte saved; on the ASIC that is 256 KB (128K) or 768 KB (256K) more a slice.

**The tile kernel** (`docs/tiled_prefill.md`, branch `tiled-prefill`). It restructures prefill so
that nothing in TMEM grows with the model's widths: T = 16-24 rows of Qwen3-0.6B and 8-12 of a
4B shape fit 64K words, and a bigger TMEM adds nothing to it (77.0K a row at 256K against 76.4K).
If the tile kernel ships with MCOLS 8, the case for 128K or 256K rests on batched decode at 8+
users and the MoE experts' larger chunks, not on dense prefill; and the case for one copy then
rests on block RAM for something else. Its access pattern (the next head's projections
streaming while a head attends, the MLP in pipelined pieces) overlaps the units more than today's
kernel, so a one-copy TMEM's conflicts should be measured on it before a build that has both; not
done here (its compiler is on its branch).

**Scaling across slices and columns.** The TMEM is per slice, so one copy saves the same five
sixths at every slice count; nothing in the replays crosses slices (the collective reads as a
guest, never replayed). Per slice, the read traffic that conflicts grows with the MXU's columns:
at MCOLS 8 the MXU's drain and QUANT's reads double per weight pass, and p8_q3 is every candidate's
worst workload (section 6.4: 2.5x the replays of the 4-row run). A wider MXU makes the port-B pair
(the MXU's accumulation, QUANT's data) heavier, which is why fewer consumers (section 6.6) matter
more as the columns grow, and placement less.

### 7.1 The input stage (release r2)

**Why.** r2's m4 build (`r2m4_d352e53d`: MCOLS 4, one pumped copy, 133.33 MHz) missed on two
kinds of path, both from the units' request logic straight into the TMEM:
- clk -> clk2x: 926 endpoints, TNS -146.2 ns, all into the pumped banks' pins. By source: the
  VPU's chunk address (cols / ic, 55%), the MXU's ASCALE address (31%), QUANT (4%), the DMA (2%),
  the collective (2%), and the write stage pw_* (4%). Worst -0.39 ns.
- core_clk: 65% of its TNS was one path, worst -0.536 ns at 16 levels. It runs from the MXU's
  ASCALE adder through the TMEM's address compare and rank to `lost`, then to the VPU's enables.
- The words (RAM -> unit) pass at m4: +0.02 to +0.26 ns.

**What.** Every read request goes to the TMEM from flip-flops, a cycle after it forms, and the
writes are registered at the TMEM's input. Each unit gets the request register, so its own
arithmetic ends there; the grants still come from the requests the TMEM sees.
- **MXU:** t_ren/t_raddr advance with t_gnt; the RMW gains a stage (rA) before r0; ASCALE takes
  its words a cycle later (al_st 3).
- **VPU:** ta_/tb_* advance with the enable; the chunk waits in mi2; the composite path's hold
  lookahead goes from HA 2 to 3.
- **QUANT:** all three ports advance with the enable; the chunk waits in m0a; the byte address
  is delayed by QL + 2; an unbuffered row's group word is delayed by a cycle (grd_d).
- **DMA:** the request advances with adv. The ST tags gain a stage (pa0). The stream fill's tags
  gain a stage (fr_k0), and the run starts a cycle after the fill. A STREAM's descriptor is
  captured a cycle later.
- **RLD:** asked from rl_pend's second cycle.
- **The collective:** on its own ports (otpu_top; otpu_board at S = 1), otpu_coll RQ: a request
  register (stage a) before stage p. Through the hub (S > 1), otpu_coll_port asks a held read
  from its second cycle (hq). The held read waits there until granted, with no replay: the guest
  rule `coll_blk` is evaluated on the same staged requests, and the DMA and QUANT never wait on
  the collective, so the hold adds no wait edge.
- **TMEM `IQ`:** the granted writes (lanes, addresses, data, grants) are registered, and the
  routing, selection and write stage run on them a cycle later.
- **PMP 1:** each pumped RAM gets its own copy of the write stage (`pk_*`, 16 at 64K words with
  NSB 2). The sub-bank decode and the RAM row are formed before the register, so the bank's
  write pins come straight from flip-flops. The bank-level `pw` stays for the bypass, the
  replays' check and the dump.

**Order.** Every read and every write moves exactly one cycle, so what a read sees is what it saw
before. This holds for the units', RLD's and the collective's reads, and for every write port's
writes. The registered write stage, its bypass and the replays' premise (section 5.1) are
unchanged relative to the reads. The scoreboard and the wait counts are completion-based and see
nothing. No new bypass is needed.

**Cost** (co-simulated on m4, LiteDRAM DDR3-1066; the reads +1 in every unit, the hack `rq3`,
against d352e53d):

| run | d352e53d | rq3 | change |
|---|---|---|---|
| Qwen3.5-0.8B tiled P (b1, T4, 4 layers) | 1,140,615 | 1,141,252 | +0.056% |
| Qwen3.5-0.8B decode at 544 | 2,538,092 | 2,536,681 | -0.056% |
| Qwen3-0.6B decode at 544 | 1,617,977 | 1,618,224 | +0.015% |
| Qwen3-0.6B decode at 1800 | 1,710,331 | 1,710,177 | -0.009% |

A second cycle for the words (a data register beside the TMEM, `rq4`) cost +0.138% on the P run.
m4 does not need it.

**With it in r2.** The other core_clk groups of r2m4 are not the TMEM's:
- The sequencer's wait test is registered (`otpu_seq` w_all_q / w_dram_q; docs/seq_vpu.md), at
  -0.013 to +0.011% on the same four runs.
- The card builds the collective unit without GATHER (`COLL_GATHER` 0, docs/infra_hunt.md 5).

## 8. Ranking

By SRAM first, then the bytes the TMEM moves (its reads, the replays among them), then logic,
and FPGA cycles last, as a sanity check (the user's rule for this core, 2026-10-12: copies are
out even when they buy 1-3% in cycles):

| option | SRAM (RAMB36 at 64K) | extra TMEM reads | LUTs (out of context) | cycles, screen geomean |
|---|---:|---:|---:|---:|
| today: six copies | 384 | 0 | 11.9K | 0 |
| P3: three copies pumped | 192 | 0 | 10.1K | 0 |
| two copies pumped (4 read slots) | 128 | (screened only) | | |
| one copy, 8 sub-banks, replays (`r8iR` / `r8xR`) | 64 | (not counted) | 23.5K / 21.4K | +1.62 (`R`) |
| one copy pumped, 1 RAM a bank, sides s (`r1pgRs`) | 64 | 6.3-19.7% | 11.4K | +0.99 |
| **one copy pumped, 2 RAMs a bank, sides s (`r2pgRs`)** | **64** | **2.8-8.2%** | **13.3K** | **+0.70** |
| ... the MXU ranked first (`r2pgRsm`) | 64 | 3.8-9.5% | 13.4K | +0.53 |
| ... and QUANT's inputs off TMEM (screen, section 6.6) | 64 | ~0.5% | | (0.19 no-feedback) |
| **ASIC form: two-port macros at the core clock, 2 a bank, old sides (`r2xgR`)** | **64 (16 macros)** | **24.7-42.2%** | (not built) | **+2.75** |
| ... with the pump's sides (`r2xgRs`) | 64 | 34.4-67.6% | | +6.07 |

Copies are rejected whatever they buy (P3, two copies). Among the single copies the sub-banks are
what costs logic (section 6.2): the pumped copy needs none for bandwidth and at most two for
conflicts. 2 RAMs a bank against 1: 2.5x fewer extra reads and a third fewer cycles for +2.1K LUTs
on the FPGA (a 2:1 data select and a sub-bank compare per pair of askers) and no SRAM: on the ASIC
a 64K-word bank is two 4K x 32 macros anyway (`docs/asic.md`).

The directions of section 4 by the same measure:

| direction | SRAM | bytes | what it does here |
|---|---|---|---|
| A1 wide line, stream buffers | = | more (a line read for a word) | dominated by the dual-port RAM on the FPGA; on an ASIC a second read |
| A2 deterministic schedules | = | = | waits up to 7 cycles a start; not built |
| A3 placement | = | fewer replays | done at the RAM-port level (sides s): -40-65% replays; within the sub-banks it halves the grant model's cost (`s8ip`) |
| A4 / B8 fewer consumers | = | fewer reads | QUANT's inputs off TMEM: replays to a quarter (6.6); the `dataflow` study |
| B5 purpose memories | split | copies between them | rejected (section 3.1: the roles mix) |
| B6 register file | +RF | fewer reads (re-reads) | at most the re-read share; VPU B off TMEM helps least (6.6) |
| B7 operand-layout rule | = | fewer replays | what sides s does without an ISA rule |

## 9. Per board, and the MX column's MCOLS 8 build

**YPCB, the MX column's MCOLS 8 build: one copy, `r2pgRs`, 64K words.** TMEM_IMPL 3, TMEM_NSB 2,
TMEM_SBP 3, TMEM_RPL 1, TMEM_TDP 1, TMEM_GST 1, TMEM_PMP 1 (the RAMs on the block design's
`core_clk2x`, section 10, the clock a pumped DSP column would share; co-simulated as PMP 2), the
default sides and rank: `TMEM_IMPL=3 ./run_vivado.sh`. 64 RAMB36 instead of 384:
the build's block RAM at MCOLS 8 (everything but the TMEM 282.5 tiles at 64K by
`docs/tmem_pump.md`'s count, with the ACT RAM doubled; the quantizer's row buffer 7.5 more) comes to
about 354 of 955 tiles (37%), against 666.5 with six copies. 128K words would be 128 RAMB36 (about
447 tiles with IMEM x2) and fits, but it buys rows only on today's kernel; with the tile kernel it
buys nothing (section 7). So 64K; 128K only if today's kernel has to ship with MCOLS 8.

**BCU1525 (UltraScale+).** The same single copy in block RAM pumped at 2x with a 200 MHz core
(`ku-r2pgR` +0.148 ns at 200 / 400 MHz out of context). Its TMEM is in UltraRAM today, which
pumps at 2x only up to a 250 MHz core (URAM288's 500 MHz); one copy needs 64 RAMB36 a slice,
which the part has, so the TMEM can leave the UltraRAM for capacity elsewhere.

**ASIC.** No pumping (an SRAM macro at twice the core clock caps the core at half the macro's
speed): the one copy's two reads and write a bank come from two-port macros at the core clock,
the write taking one port (`TMEM_TDP` without `TMEM_PMP`: the write's port-A conflict comes back,
`wk` / `wm`). 2 RAMs a bank are the two 4K x 32 macros a 64K bank needs anyway. Co-simulated
(section 6.4): +2.75% cycles and 25-42% extra TMEM reads with the writes' port A carrying the
light ports, the unpumped default. The write's port is the ASIC form's cost, and fewer consumers or
deferred writes are its levers. The macro
area is a third of P3's banked three copies (`docs/asic.md`: 48 macros of 4096 x 32, 4.05 mm2 on
nangate45, 38% of the core): 16 macros, about 1.35 mm2, before the conflict logic.

## 10. Decisions for the coordinator

- **The single copy is the TMEM** (the user, 2026-10-12: "2.9% on worst cases for 50% less RAM is a
  really good compromise"; and for the ASIC this core is for, copies are out whatever they buy).
  P3 and two copies are measured above and rejected; two copies were screened only, never built.
- **The pick: `r2pgRs`.** One pumped true dual-port copy, 2 RAMs a bank, replays, QUANT 3 and COLL
  the DMA's guests, sides s (now the default for a pumped TMEM, b5d6bcef), the default rank. 2.8-8.2%
  extra TMEM reads, +0.70% cycles geomean on the screen (p8_q3 +1.66%), 13.3K LUTs and +0.232 ns
  routed at 133.33 / 266.67 MHz (`k7-r2pgRs`).
- **The ASIC form** (two-port macros at the core clock) costs more than the pumped FPGA copy: +2.75%
  and 25-42% extra TMEM reads (section 6.4). Its sides are the unpumped default (the writes' port A
  for the light ports). The next ASIC step is to measure fewer consumers and deferred writes on it.
- **64K words** for the MCOLS 8 build, with the tile kernel; 128K only if today's kernel ships.
- **The port-B finding** is what is left to solve: most of the replays come from one RAM port
  (QUANT's data and the MXU), so one copy does not fail for want of read slots; the operands are
  badly placed across them. Sides s placed them as well as two ports allow; the rest is fewer
  consumers. The no-feedback screen leaves a quarter of the conflicts with QUANT's inputs off TMEM.
  `dataflow`'s co-simulation of its fold with quantize-on-drain cuts the replays 98-99%, and the
  copy then costs nothing (section 6.6).
- **Integration** (the MCOLS 8 tree: WAITC v2, MX E 4, the collective stub at S = 1):
  1. **The 2x clock (on this branch).** `clk_f` runs from both board tops through `otpu_native_sys`
     / `otpu_bcu_sys`, `otpu_board` (and `otpu_top` in simulation) and `otpu_slice` to `otpu_tmem`;
     it is used only with `TMEM_PMP 1` (the tops tie it to 0 otherwise), and `otpu_asic_top` ties it
     off: the ASIC does not pump. The block design makes it with `CLK2X 1` (`bd_native.tcl`, the
     BCU1525's `bd.tcl`): the MMCM's second output `core_clk2x`, from the same VCO by integer
     dividers and no phase shift, so it rises with `core_clk` (whose divider must then be even:
     133.33 MHz is 6, 100 MHz 8; 120.755 MHz stops the build). With `CLK2X 0` the port is
     `core_clk` itself, so the wrapper keeps one form and production's MMCM is unchanged.
     `create_project.tcl` takes the TMEM's knobs (`TMEM_IMPL=3 [TMEM_X=NSB=1,...] ./run_vivado.sh`,
     `vivado/tmem_knobs.tcl`: by default the pick), sets `CLK2X` from `PMP 1` and reads
     `constraints/otpu_clk2x.tcl`: the pumped banks' multicycle contract, the banks found by their
     module (`otpu_pump_bank`, moved into `rtl/mem/otpu_tmem.sv`), one section per user of the 2x
     clock. The out-of-context runs of the one copy read the same file. A pumped MX column at
     MCOLS 8, if chosen, takes the same `clk_f` and adds its section there (agreed with the MX
     column's lead: one MMCM output, one constraint set). `tb_top` makes `clk_f` with
     `TMEM_PMP 1` (`OTPU_TMEM_PMP=1`), so a co-simulation runs the real pumped RAMs. Checked
     (the build host, Vivado 2026.1): the block design at 133.33 MHz with `CLK2X` makes M 16, O0 6, O1 3
     (core_clk2x 266.67 MHz), at 100 MHz O1 4, validated, the wrapper with `core_clk2x`; without
     it the MMCM is production's (M 16, O 6, one output); 120.755 MHz stops; the BCU1525's at
     100 / 200 MHz (on xcku5p) validated. `k7-r2pgRs` with `otpu_clk2x.tcl` instead of the inline
     contract: the same 16 banks and 416 read-request pins (and 1,024 word pins), 13,271 LUTs,
     +0.232 / +0.071 ns, as before. `tb_tmem` 21 cases and the dump tests pass with the bank in
     `otpu_tmem.sv`; Verilator lint of both board tops and `tb_top` at the defaults and at the
     pick (PMP 1) clean. With the real 2x clock in `tb_top` (`r2pgRs1`: PMP 1, the default sides)
     p_q3 and d_lf run the model's cycles exactly (376,555 and 1,343,208, the same TMEMSTAT reads
     and replays as `r2pgRs`), bit-exact against the ISA.
  2. WAITC v2 gives the replays' ordering (section 5.1); the combined cost on the full list is in
     section 6.5. The trial merges with seq-vpu ffd1140c and mx-column 4406fbed conflict only in
     the tops' parameter lists (both sides kept) and `otpu_vpu`'s read enable (WAITC's line with
     `rhold`).

## Open risks

- **The pumped RAM has not run in a full build.** `TMEM_PMP 1` is checked in `tb_tmem` with a real
  2x clock, in `tb_top` (the PMP 1 candidate, `r2pgRs1`) and out of context with the boards'
  constraints; the block design's `CLK2X` is validated in Vivado, not yet placed and routed with
  the rest of the card. The co-simulations use the core-clock model (`PMP 2`, the same words and
  cycles by construction, a mutation without its bypass fails at cycle 12).
- **The replays' premise** (section 5.1) rests on the scoreboard's or WAITC's ordering. The
  simulation stops on a violation and no run hit one; WAITC's oracle counts are a minimum for one
  run, its compiler's static counts (`opentpu/waits.py`) are the deployable ones.
- **The tile kernel** overlaps the units more than today's kernel; one copy's replays have not been
  measured on it.
- **The ASIC form** (two-port macros at the core clock) is co-simulated only, never through the
  ASIC flow. Its write takes a port: +2.75% cycles and 25-42% extra TMEM reads (section 6.4), five
  times the pumped copy's. The levers (fewer consumers, deferred writes) are not yet measured on
  it.
- **Replay counts are port-cycles**, not bytes: a replay re-reads the lost read's lanes, from one
  word (QUANT 3) to eight.
- `TMEM_IMPL 3` without replays (the grant in the RAM's address path) at 16 RAMs a bank was not
  bit-exact in its one co-simulation (`r16i`, an early tree); it is not a candidate and was not
  pursued.

## Reproducing

```sh
# the traces and the no-feedback screens (section 3; the side screen of 6.4 and 6.6 are scratch
# scripts over tools/tmem_trace.py's segments/lanes, one RAM port a slot)
OTPU_SIM_PLUSARGS=+tmtrace=/tmp/p_q3.tt PYTHONPATH=. python3 tools/perf_qwen.py --model qwen3 ...
PYTHONPATH=. python3 tools/tmem_trace.py flow /tmp/p_q3.tt
# co-simulations (the simulation host; <outdir>/<workload>__<candidate>.log and .stat)
rtl/exp/tmem_nocopy/run_cosim.sh out rtl/exp/tmem_nocopy/lists/screen.txt \
  rtl/exp/tmem_nocopy/cands.txt i0 r2pgRs r1pgRs
python3 rtl/exp/tmem_nocopy/collect.py out --ref i0
# the pick by hand: the board's configuration plus
OTPU_TMEM_IMPL=3 OTPU_TMEM_NSB=2 OTPU_TMEM_SBP=3 OTPU_TMEM_RPL=1 OTPU_TMEM_TDP=1 OTPU_TMEM_GST=1 \
  OTPU_TMEM_PMP=2 OTPU_TMEMSTAT_LOG=/tmp/stat.txt PYTHONPATH=. python3 tools/perf_qwen.py ...
# out of context (a Vivado host): k7-r2pgRs, k7-r1pgR, k7-p3, k7-r1pgR-128k, ku-r2pgR, s-r2pgR-CMP ...
rtl/exp/tmem_nocopy/run_ooc.sh out k7-r2pgRs k7-r1pgR-128k
# the card with the pick (core_clk2x from the block design, constraints/otpu_clk2x.tcl)
TMEM_IMPL=3 boards/ypcb-00338/run_vivado.sh
# the TMEM's own tests
PYTHONPATH=. python3 -m pytest -q tests/test_rtl.py -k tmem_random_traffic
```

## Merging onto the MCOLS 8 tree

The branch is main 810632e2 plus the TMEM studies: the earlier one, `docs/tmem_pump.md` (the
`TMEM_IMPL 1` / `TMEM_RQ` / `TMEM_NCP` knobs, `rtl/exp/tmem_pump`), and this one. It shares base
810632e2 with mx-column (02d0fb68), waitc-mx (e274fc4a), infra-merge (7469c18d), tile-default and
dataflow. Trial merges (`git merge-tree`, 2026-10-12) conflict only in parameter lists, port
connections and build arguments. Each one resolves by keeping both sides:

1. **mx-column** (tile-default and dataflow conflict in the same places):
   - **Parameter lists and slice instances.** `otpu_top`, `tb_top`, `otpu_board` (both `u_slice`),
     `otpu_native_sys` and `otpu_fpga_top_ld`: keep `MXU_E` / `QUANT_XD` and the `TMEM_*` list,
     and `.clk_f` in the port connections.
   - **Build arguments.** This one is a renumbering, not only a text conflict. mx-column's 13th
     positional argument to `create_project.tcl` is `MXU_E`, the same position as this branch's
     `TMEM_IMPL` (then `TMEM_RQ`, `TMEM_X`). Put the TMEM's three after `MXU_E` (`lindex` 13, 14,
     15), in the same order in `run_vivado.sh`'s call. The generic string takes `MXU_E=$MXU_E` and
     `$TMEM_GEN`. The Makefile's `bit` line passes `MXU_E` and `TMEM_IMPL` / `TMEM_RQ` / `TMEM_X`.
2. **waitc-mx.** The resolutions below are the ones the WAITC co-simulations ran (b5d6bcef +
   ffd1140c: `wcmP1`, `wcmP2s`):
   - **`otpu_slice`'s parameter list.** `SEQ_WC` goes after `TMEM_SDM`, which then needs a comma.
   - **`otpu_slice`'s `u_seq`.** Use `.WAITC(SEQ_WC)` with this branch's
     `.ucmd(ucmd_s), .ustart(ustart_s), .urdy(urdy_s), .udone(udone_s)` (TMEM_IMPL 2's delayed
     starts and completions).
   - **`otpu_vpu`'s read enable.** Keep WAITC's `red_thr` and the replays' `!rhold`:
     `en_q <= ss_act_d ? ss_pe : (...) && red_thr; en_r <= en_q && !rhold;`, else
     `en_r <= (...) && red_thr && !rhold;`.
   - **`rtlsim.py`.** Keep both blocks (`OTPU_SEQ_WC`, the `OTPU_TMEM_*` knobs) and either
     `OTPU_SIM_PLUSARGS` comment.
   - **Board tops, systems and `create_project.tcl`** (both boards). Keep both `SEQ_WC` and the
     TMEM generics.
3. **infra-merge:**
   - **`create_project.tcl`.** Keep the `TRACE_DEPTH` / `COLL_GATHER` block next to the TMEM block.
     Both sides' fields go in the `puts` line and the generic string.
   - **Tops' parameter lists.** Keep both sides.
   - **`tb_board`'s `dut`.** Keep `.COLL_GATHER(COLL_GATHER)` and `.clk_f(1'b0)`.

Files only this branch changes, to keep as they are:
- `rtl/mem/otpu_tmem.sv`, with `otpu_pump_bank` at its end; its copy is gone from
  `rtl/exp/tmem_pump/otpu_tmem_pump.sv`.
- `boards/ypcb-00338/vivado/tmem_knobs.tcl`.
- `boards/ypcb-00338/constraints/otpu_clk2x.tcl`.
- The block designs' `core_clk2x` port: `bd_native.tcl`, the BCU1525's `bd.tcl`, and both lint stubs
  of the wrapper.
- `tests/test_target.py`, where the grant model is simulation-only.

After the merge, run:
- `make lint` of both boards, at the defaults and at `TMEM_IMPL 3` with the pick's generics;
- `test_tmem_random_traffic`;
- one co-simulation at the pick (`r2pgRs`, or `r2pgRs1` on the real 2x clock).

The MCOLS 8 build is `TMEM_IMPL=3` (the default knobs) at `CORE_MHZ` 133.33 or 100. Both are even
dividers; 120.755 is not. A pumped MX column widens the tops' `clk_f` condition (today
`TMEM_PMP == 1`) and adds its section to `otpu_clk2x.tcl`.
