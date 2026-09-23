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
returned by `host.board.Board.run()` into a profile (totals only, no timeline).

From Python:

```
from opentpu import lens
from opentpu.profile import profile
d = lens.to_data(profile(kernel, cfg, "name", **args))
lens.save([d], "run.otpuprof")
```
