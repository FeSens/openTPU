# The simulators in Bend

Status: **branch `bend-sim`, 2026-10-23.** Two of openTPU's Python simulators are ported to Bend
2 ([bend-lang.com](https://bend-lang.com), 2.0.38): the ISA simulator (`opentpu/isasim.py`,
with the fp32 arithmetic of `opentpu/fp32.py`) and the expert cache's replays
(`tools/offload/cachesim.py`'s `simulate`). Both give the Python's results bit for bit. Laws
state what they must satisfy, and Bend's checker accepts a proof of each.
Differential runs against the Python check everything else. Times are on a laptop (Apple M5,
4 performance + 6 efficiency cores), except where the simulation host (Linux, 20 cores, shared) is named.

## 1. Summary

| What | Python | Bend | Speedup | Bend, one thread | Checked |
|---|---|---|---|---|---|
| ISA sim, Qwen3-0.6B decode (s/token) | 1.339 | 0.089 | **15.0x** | 0.181 (7.4x) | logits bit-identical |
| ISA sim, Qwen3-0.6B prefill (16 tokens) | 9.75 s | 0.94 s | **10.4x** | 2.32 s (4.2x) | logits bit-identical |
| ISA sim, Qwen3.5-0.8B decode / prefill | 2.462 / 26.32 s | 0.198 / 2.60 s | **12.4x / 10.1x** | 6.2x / 5.0x | logits bit-identical |
| ISA sim, LFM2.5-230M decode / prefill | 0.527 / 2.90 s | 0.035 / 0.38 s | **15.1x / 7.6x** | 7.5x / 3.4x | logits bit-identical |
| ISA sim under test_qwen3 (1,102 runs) | 316.1 s | 23.9 s | **13.2x** | | every run's state identical |
| Expert-cache sweep, Qwen3.5-35B (200 replays) | 40.9 s | 0.69 s | **60x** | 2.26 s (18x) | every token's misses identical |
| Expert-cache sweep, Gemma 4 26B (200 replays) | 29.3 s | 0.51 s | **58x** | 1.66 s (18x) | identical |
| Expert-cache sweep, LFM2.5-8B (200 replays) | 12.7 s | 0.20 s | **65x** | 0.71 s (18x) | identical |

The Bend column is the ISA simulator at 4 threads (its best), and the cache sweep at the
runtime's default thread count (all cores). Section 6 has every thread count.

- **Laws:**
  - 79 about the fp32 arithmetic and the machine's words, with 10 lemmas. 19 of these 89 hold
    for every word or control array, proved by case splits and induction on the word's bits.
  - 7 about whole programs.
  - 15 about the cache, with 1 lemma.

  All of them check. The 89 word laws check again in BendTT's kernel, which has a proof in
  Lean, and so do 12 of the 15 cache laws.
- **Differential runs:** 15,463 isasim runs from 18 test files ran again on the Bend machine.
  15,444 ended in the same state and 19 on the same error. 5 more differed only in the
  shadow's count of HALT CHAINs, a harness artifact (section 5). Another 862,910 fp32
  vectors, 2,300 random STREAM programs and 935 random cache replays also agree. The MoE tests
  pass with Bend as their backend, host hooks included.
- **Bugs found in the Python:**
  - isasim's unary STREAM ops read an operand they do not use (81ac39dd).
  - cachesim's `lfu` held up to k − 1 experts more than its slots (8259401f). docs/offload.md
    section 4's lfu column is recomputed: it moved by −0.007 .. +0.001.

## 2. Why Bend

- **Laws on the code that runs.** In Bend a law is a type and its proof is a term. The proofs
  are about `sim.bend`, `fp32.bend` and `cache.bend` themselves, not about a model of them:
  the binary the tests run is built from the same files.
- **A second checker.** `--verdict` checks the proofs again with BendTT's kernel, which has its
  own proof in Lean.
- **Purity.** Bend functions are total and have no side effects. `test_pure` keeps `@unsafe`
  out of every simulator file but `simpar.bend`, so each run is a function of its inputs, and a
  law about a function holds for every call to it.
- **Parallelism without threads in the code.** Bend's runtime runs independent calls on all
  cores. The MXU's rows and outputs and a sweep's replays are written as fork-join, and the
  same code scales from one thread to ten.
- **Native speed.** `bend -o` compiles the program to C, with no interpreter in the loop.

## 3. What runs in Bend

### 3.1 The ISA simulator

| File | Lines | What |
|---|---|---|
| `bend/fp32.bend` | 456 | `opentpu/fp32.py` as a U32 softfloat: add, mul, max/min, exp2, log2, recip, rsqrt, int conversions, the int8 quantizer. Round to nearest even, flush to zero in and out, the canonical NaN. |
| `bend/word.bend` | 66 | the machine's word helpers: sign extension, the range check at the 32-bit wrap, the 64-bit product's high word |
| `bend/sim.bend` | 3,313 | one slice: every instruction of `opentpu/isa.py` (NOP, HALT and HALT CHAIN, LI, ADDI, LOOP, BAR, RLD, WAITW, LD, ST, MM, QACT, QST, VOP, GATHER, DSTEP, STREAM) with isasim's checks and error codes |
| `bend/simpar.bend` | 118 | the MXU's rows and output blocks in parallel (`Array.fork` / `join`; the only `@unsafe` file) |
| `bend/machine.bend` | 497 | the machine: rounds of the slices, the collectives (BAR, GATHER), the WAITW host's turn, the timeout and deadlock |
| `bend/server.bend`, `bend/srv.c` | 303, 100 | a server process that keeps the machine between runs |
| `opentpu/bend_sim.py` | 386 | the driver: `BendMachine` (isasim.Machine's API), `BendBackend` (an Engine backend) |

MM covers int8, int4, fp4 and NVFP4 weights, with PAIR, RSC, UNIT, ACC, ASCALE and RMAX, and
the exact 64-bit sum. VOP covers all 20 functions and every operand mode.

**State.** A slice is six `Array<U32>`:

- `c`: the registers, the status words, the loop stack, the run's arguments and IMEM;
- `t`: TMEM;
- `a`: the ACT RAM and its scale record;
- `d`: DRAM;
- `h`: the VPU's hazard stamps;
- `x`: scratch.

Each instruction is a function from a slice to a slice. The arrays are updated in place, since
each one has a single owner.

**The machine.** A round runs every slice until it halts, waits on a collective or polls a
WAITW that does not hold. When every slice that can run polls, the server hands the turn to
the host: it replies HOSTS with each slice's status words. `BendMachine.slices` shows those
slices as isasim's do (the WAITW each one polls, its registers, its DRAM words and bytes), so
the Engine's hook (`qwen3._isa_host`) and the tests' hooks (`tests/beat_link.py`) run
unchanged. HOST then reads every polled word again. If none holds, the run ends in isasim's
timeout.

**The server.** `build/otpu_bend` is built from `bend/server.bend` and rebuilt whenever a
`bend/*.bend` or `*.c` file is newer. The host talks to it over two pipes; their fds are passed
in `OTPU_BEND_IN` and `OTPU_BEND_OUT`. Each command is a 16-word header, and some carry data:

| Command | Does |
|---|---|
| INIT | the machine, from `isasim.Config` |
| DATA | one slice's DRAM, TMEM, ACT RAM or control words in or out |
| LOAD | a program into IMEM, and its eight run arguments |
| RUN | run the machine (sequential or parallel MXU) |
| HOST | after the host's turn: poll again, and run on |
| QUIT | stop the server |

The machine stays in the server between runs, so an Engine's weights are written once.
`BendBackend` is a drop-in Engine backend: `Engine(..., backend=bend_backend())`, or
`--backend bend` in otpu-chat, otpu-serve, tools/validate.py and tools/compare_hf.py.

**The MXU** is a template parameter (`~mm`). The server picks `sim.bend`'s `mm.seq` (each row's
outputs in order) or `simpar.bend`'s `mm.par`. `mm.par` splits the rows and then the outputs
in 4,096-word blocks over the cores, and gives the same result. With 8 threads, the parallel
MXU takes Qwen3-0.6B's prefill from 1.24 to 1.00 s. On a lone 1x8192x1024 MM it is 17.4x
isasim's NumPy, against 8.3x for the sequential one.

### 3.2 The expert cache's replays

| File | Lines | What |
|---|---|---|
| `bend/cache/cache.bend` | 1,062 | one replay of a router trace, as `cachesim.simulate`: `static`, `lru`, `lru_layer`, `lfu`, `opt`, with the warm list and the profile's prior |
| `bend/cache/main.bend` | 155 | a sweep: every job's replay at once, fork-join over the cores |
| `tools/offload/cache_bend.py` | 85 | `sweep(req, jobs)`: writes the trace and jobs, runs the binary, reads each job's misses per token and its cache's peak |

- **Layout.** A replay's layout (slots, tokens, layers, k, experts, the warm list) is one
  record.
- **Policies.**
  - `lru` and `lru_layer` keep queues of (expert, stamp) pairs: one queue for `lru`, one per
    layer for `lru_layer`. An entry is current while its stamp is its expert's.
  - `lfu` and `opt` keep their victims in a binary heap of three-word entries. `lfu`'s are
    (count, step, expert); `opt`'s put the furthest next use on top, after a backward pass
    has found each request's next use.
- **The sweep.** Each replay gets its own copy of the shared arrays (`Array.clone`). The halves
  of the job list are forked, so 200 replays run on every core.
- **The CLI.** `tools/offload/cachesim.py --bend` runs the sweep on Bend. It has no prediction
  coverage, since Bend returns misses per token, not the miss sets. Qwen3.5-35B's whole CLI run
  went from 57.0 to 2.2 s with the same `--json` rows (the Python run also computes the
  coverage).

### 3.3 What stays in Python, and why

| Simulator | Why not in Bend |
|---|---|
| `opentpu/rtlsim.py` (Verilator) | It simulates the RTL itself, as compiled C++. |
| `opentpu/profile.py`, `hwtrace.py`, `lens.py`, `opentpu/waits.py` | They read and analyze traces. The time goes to IO and NumPy, not to simulation. |
| `tools/tile_model.py`, `tools/scoreboard.py` | These are closed-form cost models that already run in milliseconds. |
| `tools/scoreboard_cosim.py`, `tools/moe_prefill_cosim.py` | They drive rtlsim. |
| `cachesim.py`'s `lfu_layer` and prefetch, `tools/offload/lesim.py` | Decayed use is a chain of float64 log2 and exp2, which Bend's 32-bit floats do not reproduce. |
| `tools/offload/em_slotsim.py` | Both models' traces take 2.4 s in Python, so starting a process and the file IO would be most of a port's time. |
| the board model (`SimTransport`, test_board's `*_board_model`) | It is the RTL under Verilator. |

## 4. Laws

A law is a type, and a proof is a term of that type. `{==}` proves an equality whose two sides
compute to the same value: the checker runs them. A `match` splits cases, a recursive call is
the induction hypothesis, and `%e : P` rewrites with an equality. Laws that hold for every word
go down to Base's `Word(32n)`, and lemmas prove the bit facts by induction on the word's length.

| File | Laws | `--check-only` | `--verdict` (Lean-proven kernel) |
|---|---|---|---|
| `bend/LAWS.bend` + `PROOF.bend` | 79 + 10 lemmas | ALL PROOFS CHECK, 0.9 s, 168 MB | ALL PROOFS CHECK, 2.9 s |
| `bend/prog/LAWS.bend` + `PROOF.bend` | 7 | ALL PROOFS CHECK, 1.9 s | does not fit: `sim.bend` in the kernel took >14 GB in 80 s |
| `bend/cache/LAWS.bend` + `PROOF.bend` | 15 + 1 lemma | ALL PROOFS CHECK, 22 s, 195 MB | 12 of 15 check; `skew_lfu`, `skew_opt` and `opt_bound` run out of the kernel's fuel |

**For every 32-bit word or value:**

- `ftz_idem`, `canon_idem`: flushing to zero and canonicalizing are idempotent;
- `fabs_sign`: |x| has no sign bit;
- `fmax_self`, `fmin_self`: the max and the min of a word with itself are the word;
- `r0_stays_zero`: writing R0 leaves every control array as it was;
- `waitw_eq`, `waitw_ne`, `waitw_ge`: WAITW's three compares, for every word, reference and
  mask, are isa.py's `waitw_holds` (GE: the masked word minus the reference, signed, is not
  negative);
- `max_self` (the cache).

**The IEEE-754 corners,** each value derived by hand and equal to fp32.py's:

- add: exact sums, ties down and up, the carry, overflow, inf − inf, NaN, cancellation, signed
  zeros, flush to zero in and out, the sticky bit;
- mul: rounding, ties, the sticky bit, overflow, 0 × inf, signs, the smallest normal;
- max and min of zeros, NaNs and denormals;
- int conversions at their ties and limits;
- exp2 and log2 at exact points and at their ranges' edges, including `log2_sqrt2`, the
  boundary of log2's range reduction;
- recip and rsqrt;
- the int8 quantizer's half-even rounding and saturation;
- every byte's sign extension; the range check at the 32-bit wrap; the 64-bit product's high
  word.

**Programs** (`bend/prog`): scalar code, a LOOP of 10, R0 reading 0, a program that runs past
its end (error 1), VOP FILL/ADD, and ST then LD through DRAM. Each law states the registers,
TMEM, DRAM, instruction count and state that isasim gives for the same program.

**The cache:**

- LRU thrashes on a loop one longer than the cache;
- OPT keeps the right expert;
- static never changes;
- lfu's ties;
- compulsory misses only, when the cache holds every expert;
- no misses when every expert is warm;
- `lfu_fits`: lfu holds no more experts than its slots. It failed on the old cachesim.
- `layer_sweep`: one LRU thrashes on a three-layer sweep, where an LRU per layer does not;
- the misses of each policy on a skewed trace;
- `opt_bound`: OPT misses no more than any policy that takes every miss in. Static takes no
  misses in, and here it beats OPT.

**How strong are they?** Of 15 hand-made mutants of `fp32.bend` and `word.bend`, the laws
caught 14. The one left swaps the operands of an add at equal magnitudes, which gives the same
result: it is equivalent to the original. One earlier survivor was a mul with a broken sticky
bit; it led to the `mul_sticky` law. Another, log2's boundary `>=` changed to `>`, led to
`log2_sqrt2`.

The laws do not prove that the Bend simulator equals isasim on every program; that equivalence
rests on the differential runs (section 5). The laws pin down the arithmetic's corners and the
machine's invariants, and they keep holding as the code changes. `tests/test_bend.py` checks
all three proof files on every run.

## 5. Differential checks

- **`pytest -p bend_shadow`** (`tests/bend_shadow.py`): every isasim run in a test suite runs
  again on the Bend machine from the same state. Both machines must end with the same DRAM,
  TMEM, ACT RAM, registers, instruction counts, stalls and chains, or stop on the same error.
  Runs whose WAITW host changes the host's state cannot be replayed and are skipped.

  | Where | Suites | Runs identical | Same error | Skipped (host hook) | Mismatch |
  |---|---|---|---|---|---|
  | a laptop | test_qwen3 (sequential MXU) | 1,049 | | 5 | 0 |
  | a laptop | test_qwen3 (parallel MXU) | 1,102 | | 5 | 0 |
  | a laptop | test_isa, test_kernels, test_mx, test_stream, test_vops, test_gemma4 | 275 | 18 | 68 | 0 |
  | the simulation host A | test_qwen35, test_qwen35_moe, test_lfm2, test_lfm2_moe | 2,908 | | 10,994 | 0 |
  | the simulation host B | test_prefill, test_prefill_gemma4, test_llama, test_gemma4, test_gemma4_moe, test_tiled, test_waits, test_chatkv | 10,110 | 1 | 10,275 | 5 (harness) |
  | a laptop | B's five mismatching tests again, with the fix | 531 | | 68 | 0 |

  B's five mismatches differ only in the count of HALT CHAINs (7 vs 0, 9 vs 0). They come
  from two of gemma4's PLE-on-host tests and three of chatkv's MTP rollback tests. isasim
  counts chains across runs, and the simulation host's copy of the shadow compared that total with one
  run's count (fixed in 97ee4630). With the fix, the same five tests pass on a laptop and all
  531 of their runs are identical. The
  "same error" runs come from tests that check isasim's errors (an RLD outside its bound, bad
  addresses and the like), and both machines stop on the same one. Under the shadow, A's tests had 199 passed and 3 skipped. B's had 361 passed and 1 skipped;
  its 5 failures are the five above, failed by the shadow's mismatch.
- **`pytest -p bend_backend`** (`tests/bend_backend.py`): every Engine that would run on isasim
  runs on the Bend machine instead, host hooks included. Each test then checks its own
  references. On test_lfm2_moe and test_qwen35_moe, with the MoE expert server and the
  beat-by-beat link (`tests/beat_link.py`) as hooks, 85 passed and 1 failed, in 460 s.
  - The failure is `test_device_follows_hf_and_routes_on_the_card`. Its median cosine to HF
    is 0.9968 against a threshold of 0.997, and it fails the same way on isasim on a laptop (an
    environment issue).
  - The host's turn is isasim's: it polls until a waiting WAITW holds. An earlier version
    polled until nothing was left to do, and that gave the hint tests different promotion
    counts.
- **`tests/test_bend.py`**:
  - the three proof files, and the word proofs under `--verdict`;
  - `@unsafe` kept out of fp32, word, sim, machine and cache;
  - fp32.bend against fp32.py: 16 ops × 4,096 random words, special values mixed in
    (862,910 vectors offline, 0 mismatches);
  - every VOP function in every operand mode;
  - 300 random STREAM programs, some malformed (2,300 offline);
  - the sequential and parallel MXU against isasim's MM;
  - the WAITW host's turn against isasim's;
  - random cache traces against `cachesim.simulate` (935 replays offline, 0 mismatches).

**Bugs found:**

- isasim's unary STREAM ops read the K operand, which they do not use. An out-of-range K could
  raise on a valid program (81ac39dd).
- cachesim's `lfu` made room for one miss per request, then took them all in (8259401f).
- The port's own bugs were caught by the shadow before they landed:
  - QST's HALF took QACT's CSCALE flag (24a487ee);
  - the shadow's chains count (97ee4630).

## 6. Speed

`tools/bend_bench.py` runs a model's prompt (16 tokens, 8 rows a run) and four greedy decode
steps on isasim and on Bend at each thread count. It times only the backends' `run()`; the
compiler is the same for both. Every run's logits agree bit for bit. Config: S=2, D=128,
MCOLS=8. The table gives seconds, with the speedup against isasim in brackets; the default
thread count is the runtime's choice.

| Model | | isasim | 1 thread | 2 | 4 | 8 | default |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | prefill | 9.75 | 2.32 (4.2x) | 1.22 (8.0x) | 0.94 (10.4x) | 1.00 (9.8x) | 0.99 (9.9x) |
| | decode / token | 1.339 | 0.181 (7.4x) | 0.093 (14.4x) | 0.089 (15.0x) | 0.099 (13.5x) | 0.104 (12.9x) |
| Qwen3.5-0.8B | prefill | 26.32 | 5.26 (5.0x) | 2.74 (9.6x) | 2.60 (10.1x) | 2.65 (9.9x) | 2.69 (9.8x) |
| | decode / token | 2.462 | 0.398 (6.2x) | 0.206 (12.0x) | 0.198 (12.4x) | 0.209 (11.8x) | 0.207 (11.9x) |
| LFM2.5-230M | prefill | 2.90 | 0.86 (3.4x) | 0.46 (6.3x) | 0.38 (7.6x) | 0.37 (7.8x) | 0.46 (6.3x) |
| | decode / token | 0.527 | 0.070 (7.5x) | 0.038 (13.9x) | 0.035 (15.1x) | 0.040 (13.2x) | 0.042 (12.5x) |

- **Where it comes from.** A decode step is many small instructions. isasim pays Python's
  dispatch for each one, so decode gains more (12-15x) than prefill (6-10x), whose MMs isasim
  already runs in NumPy.
- **Four threads is the knee.** Past it, the M5's efficiency cores and Bend's scheduler give
  nothing more. The scheduler hands each task to a core once, so a fork-join tree is only as
  balanced as its split.
- **The shadow's suite times.** On a laptop, test_qwen3's 1,102 runs took 316.1 s on isasim and
  23.9 s on Bend (13.2x). On the simulation host, the server was built before the parallel MXU and ran
  with 4 threads on a shared host (load 5-11). There the suites' runs took 885 vs 224 s (A,
  4.0x) and 3,224 vs 1,015 s (B, 3.2x). DSTEP's tests: 39.9 vs 6.9 s.

**The expert cache.** The full sweep of offload's router traces covers 4 texts of 2,048 tokens
× 10 sizes (`--frac`) × 5 policies, 200 replays a model. Each text is warmed from the other
three's profile. Bend's times include starting the process and the file IO. Every replay is
identical to Python's. The table gives seconds, with the speedup in brackets.

| Model (L, k, experts) | Python | 1 thread | 2 | 4 | 8 | default |
|---|---|---|---|---|---|---|
| LFM2.5-8B (22, 4, 704) | 12.65 | 0.71 (18x) | 0.40 (32x) | 0.25 (51x) | 0.20 (63x) | 0.20 (65x) |
| Gemma 4 26B (30, 8, 3,840) | 29.30 | 1.66 (18x) | 0.94 (31x) | 0.60 (49x) | 0.50 (59x) | 0.51 (58x) |
| Qwen3.5-35B (40, 8, 10,240) | 40.90 | 2.26 (18x) | 1.28 (32x) | 0.81 (50x) | 0.68 (60x) | 0.69 (60x) |

The replays are independent, so the sweep scales further than the ISA simulator. On one
thread, 18x is what compiled code with heap and queue structures gains over the Python's
dicts and heapq.

## 7. Running it

Bend 2 installs to `~/.bend/bin/bend`; `OTPU_BEND` points elsewhere. Set
`BEND_NO_TELEMETRY=1`. `--verdict` needs Lean (elan) on the PATH.

```sh
bend bend/PROOF.bend --check-only            # the word laws (and --verdict)
bend bend/prog/PROOF.bend --check-only       # the program laws
bend bend/cache/PROOF.bend --check-only      # the cache laws
PYTHONPATH=.:tests python3 -m pytest tests/test_bend.py
PYTHONPATH=.:tests python3 -m pytest -p bend_shadow tests/test_qwen3.py     # shadow every run
PYTHONPATH=.:tests python3 -m pytest -p bend_backend tests/test_qwen35_moe.py   # Bend as backend
python3 tools/bend_bench.py --model qwen3 --threads 0,1,4
python3 tools/offload/cachesim.py --bend ...
otpu-chat --backend bend ...                  # (also otpu-serve, validate.py, compare_hf.py)
```

| Variable | Does |
|---|---|
| `OTPU_BEND` | the bend binary |
| `OTPU_BEND_BIN` | a server built elsewhere (`bend bend/server.bend -o x.c`, then clang), for a host without bend |
| `OTPU_BEND_THREADS` | the runtime's threads (default: its choice) |
| `OTPU_BEND_SEQ=1` | the sequential MXU |
| `OTPU_BEND_LOG` | the shadow's per-run JSON lines |
| `OTPU_BEND_DUMP` | the shadow's state for a mismatching run, pickled |

## 8. Limits and next steps

- **The kernel's limits.** `--verdict` cannot load `sim.bend` (it runs out of memory) and runs
  out of fuel on the longest cache replays. Those laws are checked only by Bend's own checker.
- **Equivalence is tested, not proved.** A proof that `sim.bend` refines isasim, instruction by
  instruction, would need isasim's semantics written in Bend first. The laws cover the
  arithmetic and the invariants.
- **Host hooks** run in Python between rounds. A run with many host turns pays a pipe round
  trip for each WAITW read.
- **Scaling.** Beyond four threads the runtime gains nothing on these workloads. A finer split
  of the VPU's rows, like the MXU's, is the next lever for prefill.
- **More simulators.** `lfu_layer`, and with it `lesim`, needs its decayed use in fixed point,
  with Python's ties reproduced.
