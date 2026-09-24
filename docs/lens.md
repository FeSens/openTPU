# openTPU Lens

Lens is the profiler of openTPU: it records runs into profile files and explores them in a
browser app. It answers two questions: how close did the run get to the DRAM roofline, and
what kept it off.

## Quick start

```
python3 -m opentpu.lens list                                   # the workloads
python3 -m opentpu.lens record mlp attn -o run.otpuprof        # RTL traces (Verilator)
python3 -m opentpu.lens record qwen-tiny --board --axi --pos 64 -o qwen.otpuprof
python3 -m opentpu.lens open run.otpuprof                      # opens the app in the browser
python3 -m opentpu.lens html run.otpuprof -o run.html          # one standalone page
python3 -m opentpu.lens info run.otpuprof
```

With the package installed (`pip install -e .`) the same commands are available as `lens`.

`record` options:

| option | meaning |
|---|---|
| `--board` | the board configuration (`board_config()`) and micro-architecture (`BOARD_UARCH`) |
| `--axi` | the board's memory path: AXI adapter and the two-channel DDR model (D = 128) |
| `--stall N` | percent of AXI handshakes the DDR model withholds |
| `--isa` | the ISA simulator instead of the RTL (analytic timing, see below) |
| `--pos N` | qwen workloads: the token position (attention length N + 1) |
| `--bucket N` | cycles per counter bucket (default 64; smaller = finer strips, bigger files) |
| `--open` | open the app when done |

`qwen` runs one token of the real Qwen3-0.6B (`models/Qwen3-0.6B`): the tokens before `--pos`
run on the ISA simulator, the traced one on the RTL -- several minutes. `qwen-tiny` is a small
random Qwen3 with the same structure and runs in seconds.

## The app

`lens open` starts a local server (127.0.0.1) and opens the app. The page also works on its own:
**Open profile…** or drag and drop any `.otpuprof` file onto it. Deep links select a view and a
moment: `#view=floor&t=12000` (also `profile=` and `slice=`).

**Overview**

- cycles, time at the board clock, fraction of the DRAM roofline, DRAM port B busy, MACs per
  cycle, instruction count;
- *Where the cycles went*: every cycle of DRAM port B, either streaming or idle, with the idle
  cycles attributed in order to: DRAM not ready (backpressure), MXU starved (chunk FIFO empty:
  DRAM latency), MXU not consuming (drain / row credits), TMEM bank arbitration, MXU computing
  between streams, waiting on VPU / quantizer / collective / DMA work (the units in flight then),
  or nothing in flight (dispatch or program order);
- unit utilisation and TMEM arbitration losses; findings in plain sentences (including the
  instructions and source lines the MXU waited for); a per-instruction-class table with busy
  cycles, nominal work, and the time spent waiting for dependencies and for the unit.

**Timeline** -- one row per unit (instructions that overlap on a unit stack), the dispatch-to-
start wait drawn as a thin line (amber while dependencies are pending), and counter strips:
DRAM port B, MXU compute, DRAM stalls, TMEM losses and the MXU chunk-FIFO level. Wheel zooms,
drag pans, double-click resets; hover or click an instruction for its timing, counters and
source line.

**Floorplan** -- the machine as blocks: DRAM channels, AXI ports, DMA, ACT RAM, MXU (chunk FIFO
level and columns), quantizer, TMEM banks, VPU (composite and simple lanes), collective, and the
sequencer's dispatch window. Play / pause (space), step (arrows), speed, scrub. Two modes:
continuous (cycles per second) and instruction by instruction (jumps from one instruction start
to the next). At every moment:

- moving dashes show the data paths in use (LD: DRAM → AXI → DMA → TMEM; MM: DRAM → MXU,
  ACT RAM → MXU, MXU → TMEM; QACT: TMEM → quantizer → ACT RAM; QST: TMEM → quantizer → DRAM;
  VOP: TMEM → VPU → TMEM; GATHER: TMEM → collective → TMEM);
- each unit is outlined by its state: busy, stalled on DRAM, lost TMEM arbitration, blocked,
  waiting on a dependency (an MXU prefetching but not yet released counts as waiting), queued,
  idle; the side panel names the running instruction and its source line;
- the *roofline gap* bar shows DRAM port B in the current bucket and names what is holding it
  back right now, plus the fraction of peak streaming achieved so far;
- window slots are filled when running, outlined in amber while waiting for dependencies.

*Explanations.* Hover any element of the floorplan -- every unit, the dispatch-window slots,
the MXU's FIFO and columns, each TMEM bank, each VPU lane, the DRAM ports and channels, the
control block, every data path, each state colour in the legend, the roofline-gap bar, the
unit list and the playback controls -- for a card with: what it is, how it works in this design
(numbers from the profile's configuration, under "How it works here"), its live state at the
current cycle (the instruction it runs with operand addresses and source line, what it waits
for and why, occupancy, stalls so far by cause) and why it matters for the roofline, with what
would improve it. Click (or tap) to pin the card; it keeps updating while the run plays; click
elsewhere, × or Escape to close. Hovering the timeline's counter strips explains them too.
**? How to read** opens a short guide; **Tour** walks through the main units (shown once
automatically; remembered in the browser). Deep links: `#view=floor&t=9000&pin=mxu`,
`#tour`, `#notour`. `#selftest` hovers and pins every element and reports failures (used by
`tests/test_lens.py` in headless Chrome).

**Instructions** -- every dynamic instruction, filterable and sortable; clicking one jumps the
floorplan (and the timeline) to it. **Source** -- the same attributed to the kernel source lines.

## Profile files

`.otpuprof` is gzip-compressed JSON (plain JSON is accepted):

```
{"format": "openTPU-profile", "version": 1, "created": "...", "tool": "openTPU Lens",
 "profiles": [ {...}, ... ]}
```

Each profile has `kind` (`rtl`, `isa` or `board`), `name`, `cycles`, `clock_mhz`, `config`
(S, D, MCOLS, LANES, CL, ACT_BLOCKS, TMEM_WORDS, DRAM_BYTES, WIN, RPB, WPB), `roofline`
(`bound`, `efficiency`, per slice port-B/port-A transfers), `programs` (per slice
`[pc, name, detail, comment, source]`), `instrs` (per dynamic instruction
`[slice, idx, pc, unit, name, detail, dispatch, ready, release, start, end, source, counters,
nominal work, port-B, port-A]`), `slices` (DRAM port totals, unit busy cycles, TMEM losses and
the bucketed counters), `sources`, `notes`, and for board runs `board` (the card's counters).
Readers must reject a newer `version`.

Bucketed counters (one entry per bucket, from the RTL's P and Q trace lines; see
`opentpu/profile.py`): `c` bucket end cycle, `n` cycles, `bm`/`bd` port B busy for MXU / DMA,
`am`/`aq` port A for MXU / QST, `mx` MXU consuming, `fm`/`fq`/`fv`/`fc` cycles lost to TMEM
arbitration (MXU drain, quantizer, VPU, collective), `bs`/`as` port B / A requests waiting for
the memory, `ms` MXU starved, `mb` MXU not consuming although chunks are there, `ff` summed MXU
FIFO level, `ld` program-loader traffic.

**ISA profiles** (`--isa`) come from the ISA simulator, which has no cycle model: every executed
instruction is placed back to back for its nominal work (no overlap between units). The
timeline is an upper bound; use it for instruction mix and DRAM traffic, the RTL for timing.

**Board profiles**: `opentpu.lens.board_data(name, cfg, programs, stats)` turns the counters
returned by `opentpu.host.board.Board.run()` into a profile (totals only, no timeline).

**Hardware profiles** (kind `hw`, `otpu-lens record`, below) are the RTL profile built from the
card's trace buffer: the same fields, plus `board` (the run's counters) and `hwtrace`:
`count` (TRACE_COUNT), `drop` (TRACE_DROP), `depth`, `keep` (`first` / `last`), `wrapped`,
`lost` (records that did not fit), `records` (read back), `open_at_end` (instructions still in
flight where the trace stops), `traced_to` (the last traced cycle) and `complete`.

From Python:

```
from opentpu import lens
from opentpu.profile import profile
d = lens.to_data(profile(kernel, cfg, "name", **args))
lens.save([d], "run.otpuprof")
```

## Hardware profiling (otpu-lens)

The board's control block has a trace buffer (register map 2, [observability.md](observability.md)):
while enabled it records the same events the RTL prints with `+trace` (dispatch, start,
release, end, unit counters, the P/Q counter windows, the halt line) as 64-bit records.
`otpu-lens` (installed with `pip install -e .`; `opentpu/host/hwlens.py`) turns them into
Lens profiles:

```
otpu-lens record --dev /dev/xdma0 --model models/Qwen3-0.6B --prompt "Why is the sky blue?" \
                 --tokens 2 -o card.otpuprof                 # 2 decode steps on the card
otpu-lens record --dev /dev/xdma0 --pos 300 --tokens 1 -o late.otpuprof   # a long context
otpu-lens record --sim -o sim.otpuprof                       # mlp-small on the board model
otpu-lens record --sim --workload qwen-tiny --pos 8 -o q.otpuprof
otpu-lens open card.otpuprof                                 # open / html / info / list: as lens
```

`record` feeds the prompt (chat template; `--prompt-ids 1,2,3` skips the tokenizer), then
generates greedily; the steps at positions `--pos` .. `--pos` + `--tokens` - 1 (default: from
the prompt's last token, whose step yields the first new token) run with `TRACE_CTRL` on. After
each traced step the host reads the records back, rebuilds the trace text with
`opentpu.hwtrace.records_to_trace` (the D lines take their opcode from the program the host
loaded) and builds the profile with `opentpu.profile.parse` and `opentpu.lens.to_data`, exactly
as `lens record` does from an RTL simulation: one profile per traced step, the timeline at the
card's cycle resolution, at the bitstream's CORE_KHZ.

The buffer holds CAPS-depth records. `--keep first` (default, STOP_WHEN_FULL) keeps the start
of the run; `--keep last` makes it a ring that keeps the end (the host reorders it, oldest
first, and drops events of instructions whose dispatch was overwritten). Records that did not
fit, and events the capture queue dropped (TRACE_DROP), are reported on stdout, in the
profile's `hwtrace` metadata and as the profile's first notes. On a register map 1 bitstream
(no trace buffer) `record` writes counters-only `board` profiles; without `opentpu.hwtrace` it
stops with an error and saves the raw records next to the output (`.records.npy`).

`--sim` runs the same on the Verilator board model: a kernel workload of `lens list` (default
`mlp-small`) placed as on the card, or `qwen-tiny` steps; the model's trace buffer is read out
in the simulation that ran the program.
