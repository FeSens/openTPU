# Targets: one design, many boards and chips

openTPU builds for several targets from one source tree: the YPCB-00338 card in production, the
BCU1525 (Virtex UltraScale+ VU9P, four DDR4 DIMMs: four targets, below), ASICs (NanGate45 with OpenROAD now), and
whatever comes next. A target is one file, `targets/<name>.toml`; everything that depends on it
is derived from that file by `opentpu/target.py`. The design rationale is in
`docs/superpowers/specs/2026-10-10-multi-target-design.md`.

```
python3 -m opentpu.target list                  the targets
python3 -m opentpu.target show bcu1525          one, resolved
OTPU_TARGET=bcu1525 python3 ...                 simulators and tools use that target
OTPU_TARGET=ypcb-00338:m8 python3 ...           a target's variant (below)
```

A target file may hold variants, `[variants.<name>]` tables of `[arch]` and `[uarch]` values over
the file's, selected as `NAME:VARIANT` (OTPU_TARGET; the board flow's `make bit TARGET=`); the
file's `[target] variant` is the default, and its `[arch]` / `[uarch]` state that variant's values
(`load` checks them). Each variant has its own CFG_HASH. The card's: m4 (release r2: 4 MX columns
at E 4, 64K TMEM words) and m8 (8 at E 8, 128K words).

## The BCU1525's targets

One board, four target files, because the memory system and the number of slices change what
the compiler sees ([bcu1525.md](bcu1525.md) for the board):

| Target | Platform id | What | Memory | Docs |
|---|---:|---|---|---|
| `bcu1525` | 2 | two slices at MCOLS 4, two 64-byte channels a slice | LiteDRAM, DDR4-1200 | [bcu1525.md](bcu1525.md) |
| `bcu1525-wg` | 5 | one slice at MCOLS 8 with weight groups (WG 8), 8 channels of 128-byte beats, two a DIMM; 36-bit DRAM addresses, 32 GiB | AMD's DDR4 MIG, DDR4-2400 | [bcu1525_wg.md](bcu1525_wg.md) |
| `bcu1525-s2` | 6 | two such slices, a slice a die, two DIMMs (16 GiB) each, four channels a DIMM | the same | [bcu1525_s2.md](bcu1525_s2.md) |
| `bcu1525-x2` | 7 | two cards of `bcu1525-s2` joined by QSFP28: the compiler sees S = 4, each card reports its own two slices (`[platform] cards = 2`) | the same, 8 DIMMs | [bcu1525_s2.md](bcu1525_s2.md) section 6 |

Their variants. *Production* is what the cards run and what the documents' numbers are for;
*experimental* variants build and simulate, and their programs and results may change:

| Variant | Adds | State |
|---|---|---|
| `bcu1525-wg` (no variant) | | the first images; CFG_HASH 4b35ec47 |
| `bcu1525-wg:kt` | MM's K tail (docs/isa.md "K tail") | superseded by `bhw` |
| `bcu1525-wg:bhw` | `kt`, the bank-half window (BHW 33), the sequencer's window of 32 | superseded by `bhw16` |
| **`bcu1525-wg:bhw16`** | `bhw` and the DeltaNet states in 16 bits (ST16 e6m9) | **production**, one card; CFG_HASH 4d5cbf8d |
| `bcu1525-wg:ic` | IMEM as an instruction cache over the program in DRAM (docs/icache.md) | experimental: RTL and simulation |
| `bcu1525-wg:bhw16fs` | `bhw16` and EXP2SUB's fused row sums (FSUM) | experimental |
| `bcu1525-wg:wmx` | `bhw16`, the MXU's array pumped at 2x, the instruction cache, a second TMEM write a bank, new vector ops | experimental: `bcu1525-x2:wmx`'s slice on one card |
| `bcu1525-s2` (no variant) | BHW 33, K tail, window 32, SEXACT and the pipelined collective hub are the target's own | on the card at 100 MHz; CFG_HASH f2f66919 |
| `bcu1525-s2:kt` | nothing: an alias from before K tail was the default | alias |
| **`bcu1525-s2:st16`** | the 16-bit states | on the card at 150 MHz; the image each card of the pair runs (CFG_HASH e331f79c) |
| `bcu1525-x2:kt` | nothing | alias |
| **`bcu1525-x2:st16`** | the 16-bit states | **production**, two cards |
| `bcu1525-x2:wmx` | `st16`, the pumped MXU, the instruction cache, the VPU's composites in two passes, the chunked DeltaNet's quantizer | experimental: its 100 MHz build missed timing at its density (97-98% of the slice dies' CLB sites) |
| `bcu1525-x2:b2` | `wmx` with the composites back at three stages, quantize-once, the DMA's second command, more | experimental: not built |

A production variant's `[arch]` values are fixed by the bitstreams in use: change one and the
CFG_HASH changes, and the host refuses programs compiled for the other.

## The four layers of a target

| Section | What | Who depends on it |
|---|---|---|
| `[arch]` | the `opentpu.isasim.Config` fields: S, D, MCOLS, ACT_BLOCKS, ACT_ROWS, TMEM / IMEM words, DRAM bytes per slice, LANES, PAIR / DSTEP / STREAM, SEQ_WC (EXPERIMENTAL: the sequencer's wait entries, which the host loads with every program; docs/isa.md "Wait counts"), OUTER (false: a VPU without VOP OUTER, otpu_board OUTER 0 and CAPS2 bit1; the WG targets', whose DeltaNet runs DSTEP), DSTEP2 (true: DSTEP's pair, otpu_board DSTEP2 1 and CAPS2 bit2; the WG targets') | the compiler and the programs. Two targets with the same arch run the same programs |
| `[uarch]` | timing-only RTL parameters: MXU_IMPL, MXU_E (optional), VPU_CL, ULANES, TMEM_IMPL, RPB / WPB, WIN, FIFO_DEPTH, the trace buffer | the RTL, the co-simulation |
| `[tech]` | the cell family (`rtl/tech/<family>`: generic, xilinx7, xilinxus, asic), the part or the PDK | the RTL file list |
| `[platform]` | the board: memory type and rate, channels per slice, bytes a slice addresses per channel, the channel beat (an FPGA's: the native adapter's NCH and CB, `rtl/mem/otpu_native_dram.sv`; NCH x channel bytes = DRAM bytes), the host link, the core clock | the board flow, the host runtime |

Every `[arch]` field is stated (but WG, OUTER and DSTEP2, which default to 1, true and false:
the machine before them): the file is the hardware's truth. `board_config()`, which the
simulators and tests use, takes the target's sizes, WG, OUTER and DSTEP2 but keeps its optional
features (PAIR, DSTEP, STREAM,
SEQ_WC) off unless `OTPU_PAIR` / `OTPU_DSTEP` / `OTPU_STREAM` / `OTPU_SEQ_WC` turn them on, as
it did before targets; `opentpu.target.config(name)` is the full truth. A VPU without OUTER is
in the CFG_HASH (`arch_hash`: the field only when off, so every build with it hashes as before);
there the compiler refuses `outer(acc=...)` and the ISA simulator a VOP OUTER. DSTEP2 is in it
only when on (DSTEP's pair; with DSTEP alone: `build_config`).

## What is derived

- `target.config(t)`: the `Config`.
- `target.sim_uarch(t)`: the Verilator builds' micro-architecture (`rtlsim.BOARD_UARCH`), with
  an FPGA target's memory channels (`NCH`, `CB`) where they are not the two 64-byte channels the
  simulation tops default to.
- `target.mem_model(t, mhz)`: the co-simulation's channel model from the `sim_*` platform keys
  (per channel the sustained rate, the read latency, a refresh-like duty loss), as arguments of
  `sim/verilator/otpu_native_mem.sv` in core cycles; none for the DDR3 cards.
- `target.rtl_params(t)`: the parameters of a board or ASIC top, including the identity the
  control block reports.
- `target.build_config(t, MCOLS=, LANES=, ACT_ROWS=, DSTEP=)`: the Config of an FPGA build
  with the board flow's knobs applied.
- `target.arch_hash(cfg)`: a 32-bit hash of a Config, in the bitstream's CFG_HASH register.
- `python3 -m opentpu.target env NAME`: shell assignments the board flows read
  (`boards/ypcb-00338/run_vivado.sh` takes its knobs' defaults, the platform id and the arch
  hash from them).

## The device reports its configuration

Register map 4 (`docs/observability.md`) adds the CFG_* registers at 0x80..0x9C: slices, TMEM,
IMEM, ACT RAM, DRAM per slice, channels, the micro-architecture, the platform id and the arch
hash. `Board.info()` reads them; `device_config(info)` builds the host's Config from them (map
3 and older bitstreams: the YPCB-00338's values, as before) and refuses a CFG_HASH that is not
the hash of the Config the registers describe. `check_fits(cfg, info)` (BoardBackend) refuses
programs compiled for another MXU, lanes or ACT RAM rows, or needing more slices, TMEM, IMEM,
ACT RAM or DRAM than the device has.

## The RTL's layers

- The core, `rtl/top`, `rtl/mxu`, `rtl/vpu`, `rtl/mem`, `rtl/seq`, `rtl/dma`: no vendor cell.
- `rtl/tech/<family>/`: the same modules in each family, the file list picks one. Today the
  systolic MXU's cells `otpu_pe` and the MX column end `otpu_mxce` (`otpu_tech_dsp.sv`; docs/
  mx_mxu.md section 9): `generic` (behavioural:
  the simulators and the ASIC flows), `xilinx7` (DSP48E1: the YPCB card), `xilinxus` (DSP48E2:
  UltraScale+). `tests/test_tech.py` checks that the xilinx7 cells equal the generic ones cycle
  for cycle on yosys's DSP48E1 model.
- `rtl/shell/`: what the PCIe + native-memory FPGA boards share (control block, trace, the
  channel bridges, XDMA's split, `otpu_board`). The native DRAM adapter is `rtl/mem/otpu_native_dram.sv`.
- `rtl/boards/<board>/`: the board's top and its clock / memory / host wiring.

`opentpu.target.sources(t)` is the file list (`python3 -m opentpu.target files NAME`); the board
Makefile, `create_project.tcl` and `rtlsim.RTL_SOURCES` hold the same lists, which
`tests/test_target.py` checks.

## Adding a target

1. Copy the closest `targets/*.toml`, give it a new name and a new `platform_id`.
2. A new board: its top under `rtl/boards/<board>/`, its flow under `boards/<board>/`
   (constraints, the memory and host IP), reading `python3 -m opentpu.target env <name>`.
3. A new cell family: `rtl/tech/<family>/` with the same module names as `rtl/tech/generic/`.
4. Run the target's arch through the config matrix (`tests/test_config_matrix.py`) and its RTL
   through the token-exact tests (`OTPU_TARGET=<name>`).

## Limits

DRAM addresses are `[arch] AW` bits per slice (Config.AW: 32 to 40, default 32; docs/isa.md
"Machine model"), and DRAM_BYTES a power of two of at most 2^AW. Every target but the BCU1525 at
WG 8 leaves AW out: 32 bits, 4 GiB a slice, its hash and RTL parameters as before the field
(arch_hash and rtl_params leave AW out at 32). bcu1525-wg states AW 36: its 32 GiB (64 GiB
later) as 8 virtual channels of 4 GiB, two a DIMM (rtl_params' AW, DRAM_LOG2 35, CH_LOG2 32;
[bcu1525_wg.md](bcu1525_wg.md) section 8).
No register says AW; the host takes it from the card's CFG_HASH (opentpu.host.board device_aw:
the first of its platform's target's AW, 32 and the current target's whose hash it is), so a
bcu1525-wg build from before the field (AW 32, 4 GiB) still runs, a host of any target compiles
a card's programs at its AW, and a card with more DRAM than its AW addresses (a platform this
tree has not) is refused. The board model (SimTransport) builds otpu_board at the target's AW.
Offsets 0xA0 + 4s stay reserved for per-slice DRAM windows the host would set.
