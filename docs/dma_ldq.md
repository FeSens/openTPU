# The DMA's second command (LDQ 2) and quantize-once's LDA

Two changes to `rtl/dma/otpu_dma.sv` for prefill at long chunks ([pf500.md](pf500.md)), where the
expert-major run is DMA-bound: a slice's run at R 512 is some 8,700 short LDs and 4,096 STs, each
paying the DRAM's first-beat latency with the DMA holding one command at a time.

- **LDQ 2** (`[uarch] DMA_LDQ = 2`, the DMA's parameter `LDQ`): a second command in flight. The
  next command's DRAM reads go out under the last one's TMEM writes, or under an ST's wait for
  its writes.
- **LDA** (`[arch] QONCE`, opcode 0x14): a row the quantizer stored once comes back from DRAM
  straight into ACT RAM, in place of an LD of its fp32 words and a QACT.

**Status.** Both are in `bcu1525-x2:wmx` (LDQ 2) and `bcu1525-x2:b2` (LDQ 2 and QONCE) only:
bit-exact against the ISA simulator in the unit testbenches and the tests of section 3 on
Verilator. The staged design's board-model runs ([pf500.md](pf500.md) section 3) carry both, and
check instruction counts and the expert run's records against the ISA simulator, not every
output. No production image has them and neither has run on a card. The speeds below are
**simulated** (the sequencer model with each instruction class's service fitted to board-model
traces, [pf500.md](pf500.md)'s class fits), not measured.

LDQ is a `[uarch]` key: not in the Config, programs and images do not change, and LDQ 1 is the
logic every card built so far has. The simulators take `OTPU_DMA_LDQ` as an override
([host.md](host.md) "Simulation overrides").

## 1. LDQ 2

### Front and back

The DMA is two sides, each holding one command:

- the **front**, the DRAM request side: the command whose chunk reads are being issued;
- the **back**, the delivery and completion side: the command whose segments go into TMEM, or
  whose writes the DRAM has still to acknowledge.

With one command they are the same command. LDQ 2 adds one command register (the slot) in one of
three states: empty; *queued* (accepted, not launched); *front* (launched: its reads issuing, its
delivery waiting). `rdy = !(queued || front)`, from flip-flops: a start on an idle DMA launches
as at LDQ 1, a start on a busy one is latched into the slot. Two commands in flight, never three.

### What launches early

| queued | behind (the back) | when | condition |
|---|---|---|---|
| LD | LD | the back's requests have all gone out and it is not completing | - |
| LD | ST | the ST is in its wait for `wr_idle` (its TMEM reads done, its chunks written, the buffer empty) | the LD reads none of the ST's DRAM words (compared at accept) |
| LDA | LD, LDA | as an LD behind an LD | - |
| LD | LDA | the LDA's requests have all gone out | - |
| ST | ST | the back is in its wait for `wr_idle` | the front reads TMEM into the chunk buffer; its DRAM writes start only once the back is done (the write hold) |

Never early: an LDA behind an ST, an ST behind an LD, a zero-length LD, and WAITW, DSTEP and
STREAM, which overlap nothing either way. Those launch once the DMA is idle, from the slot.

**The handoff.** When the back completes (an LD's last TMEM write, an LDA's last ACT write, an
ST's `wr_idle`) the front's command takes the back in the same cycle, decoded again from the
slot. Dones stay in issue order and never share a cycle.

**The chunk buffer** is shared and never reset at a launch or a handoff: port B returns reads in
order and each command eats exactly the chunks it requested, so it is a FIFO across the two
commands and the returns carry no tag.

### Why nothing is observable

| back | hazard | covered by |
|---|---|---|
| LD | the front's TMEM writes after the back's (WAW) | the front's delivery starts after the back's last write (the handoff) |
| LD | the front's DRAM reads beside the back's | none needed (read / read) |
| ST | the front reads DRAM words the back writes (RAW) | no early launch on an overlap |
| ST | the front writes TMEM words the back read (WAR) | the back's TMEM reads are done in its wait; the front's delivery comes after its done |
| ST | a front ST's DRAM writes passing the back's | the write hold: `wr_idle` stays the back's own writes' |

The write hold is there because port B has no per-command acknowledgement: `wr_idle` is "every
accepted write gone out on every channel", the channels complete independently
([native_channels.md](native_channels.md)), and a quantizer write may overtake an older B write
on a channel. A front ST's writes behind the back's done lose nothing while its TMEM reads (4
cycles a chunk at D 128) are longer than the wait.

**The waits do not change.** The DMA still completes in order, and a done still means the
command's TMEM writes, or its acknowledged DRAM writes, are in place; so no ISA field, footprint
or wait entry ([seq_vpu.md](seq_vpu.md)) knows about LDQ.

### Results

Unit testbench (`sim/verilator/tb_dma.sv`: random LD / LDA / ST sequences against a reference
model with random port-B grants, latencies and ACT RAM contention; writes on independent
channels), measured on Verilator:

| case (read latency 100 cycles) | LDQ 2 / LDQ 1 cycles |
|---|---:|
| 40 LDs of 256 words | under 0.6 |
| LDA, LDA, LD | under 0.75 |
| 20 STs of 8 chunks, write wait 100 | 0.755 |
| 20 STs of 2 chunks, write wait 200 | 0.939 |

The testbench's mutations (the overlap compare never set, the front's requests not counted under an ST,
the handoff keeping the back's fields, a launch under a back that still requests, an LDA behind
an ST, ...) each fail the testbench.

Qwen3.6-35B-A3B prefill at R 512 on two cards (simulated; the class fits with the DeltaNet core
on a borrowed fit, VPU_CNS 3 with VSIG and LDA in every row. Read the differences: with the
core's own fits every row is about 8% lower, below):

| | tok/s |
|---|---:|
| one command, with LDA | 184.3 |
| LD behind LD and behind an ST's wait, LDA behind LD | 186.8 |
| + LDA behind LDA | 190.8 |
| + ST behind ST at any time (an upper bound for the write hold) | 197.1 |
| any pair, STREAM included | 197.2 |

With the DeltaNet core on its own fits the same design is 177.4 tok/s without an ST behind an
ST, 179.2 with the write hold as built and 181.6 with no hold: the hold keeps 45% of what an ST
behind an ST at any time would give. (The pumped MXU's pass rule then takes the as-built design
to 177.1, and the board model's own runs give 177.2: [pf500.md](pf500.md) section 3.) An LD behind an LDA is
worth nothing on this program (its row-id LDs run back to back).

Area a slice: LDQ 2 itself was estimated, not synthesized, at about +1K LUTs and +0.4K
flip-flops (a second command register and the write side's counters). By yosys with 7-series
cells on the DMA without its stream engine (run-to-run noise about 0.25K LUTs): the LDA cases
about +0.4K LUTs and +115 flip-flops more, and nothing measurable for the ST behind an ST (2
flip-flops).

## 2. Quantize once and LDA

The expert-major run ([prefill.md](prefill.md)) quantizes each token's input row again for every
(row, expert) entry: 80% of its quantizer time (1.05 M of 1.31 M cycles a layer at R 512) is
re-reading rows it has already quantized, and a slice's run loads 33.6 MB of fp32 rows and
1 / rms words to do it. With `QONCE` the router's QACT stores each row's quantized bytes and
records to DRAM as it makes them (QACT flag `ST`), and every expert run that takes the token
loads them back with **LDA**: DRAM to ACT RAM through the DMA's LD path, the bytes, scales and
offsets exactly the QACT's. The instruction forms, the row layout (`qrow_bytes`: 2,176 bytes a
row at a hidden size of 2,048, 2.6% over the bytes for the records) and the footprints are in
[isa.md](isa.md) "Quantize once".

- A row arrives at 16 bytes a cycle: about 144 cycles an entry against the QACT's 256, and the
  entry's 1 / rms LD and its gate / up QACT are gone. The run loads 8.9 MB.
- LDA writes ACT RAM at a second write port through a 4-entry queue and yields to the quantizer.
- Area a slice (yosys A/B, UltraScale+ cells): +2,277 LUTs, +696 flip-flops, no block RAM.
- Simulated, the 35B at R 512 (the class fits before the DeltaNet core had its own, VPU_CNS 5):
  164.7 to 191.4 tok/s alone (+16.2%), 207.6 with any two DMA commands in flight (a bound: the
  rule set as built is section 1's).

**Why from DRAM and not from TMEM.** The other form (the quantizer writing its rows to TMEM and
a raw QACT copying them into ACT RAM) is cheaper to build (about 1.0K LUTs) and gives less
(158.3 against 177.4 tok/s, each alone on an earlier, more pessimistic fit of the expert run's
LDs): it keeps a second SRAM copy of data that
already has a home, and it leaves the gate / up QACTs' quantizer time in place. The design rule
here is to move data less, not to copy it between on-chip memories.

## 3. Tests

`tests/test_dma_ldq.py` (LDQ 1 and 2: random, directed, buffer full, overlap, the ST-behind-ST
cases on 1 and 4 channels), `tests/test_dma_lda.py` (LDA in the queue), `tests/test_qonce.py`
(QACT `ST` and LDA against the ISA simulator, on the ISA and on Verilator at LDQ 1 and 2).
