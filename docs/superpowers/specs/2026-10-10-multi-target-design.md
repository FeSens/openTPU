# Multi-target openTPU: boards, other vendors and ASICs

Date: 2026-10-10. Status: approved direction (approach A), being implemented on branch `targets`
and the branches it fans out to.

## Goal

One source tree, one ISA, one compiler, many targets: the YPCB-00338 card (Kintex-7, 2 x DDR3,
in production), the BCU1525 (Virtex UltraScale+ VU9P, 4 x DDR4 DIMM up to 64 GB, PCIe Gen3 x16;
boards arrive the week of 2026-10-12, Vivado license pending), other FPGA vendors later, and
ASICs: NanGate45 with OpenROAD now (PPA numbers, a rehearsal for tapeout), a manufacturable open
PDK (IHP SG13G2 / GF180) and a commercial node later.

Success:

- A target is one file. Everything that depends on it (the RTL's parameters and file list, the
  Vivado / OpenROAD settings, the compiler's `Config`, the simulators' micro-architecture, the
  host's expectations) is derived from that file.
- A device reports its full configuration; the host builds its `Config` from the device and
  refuses programs compiled for another architecture.
- The core RTL (`rtl/top`, `mxu`, `vpu`, `mem`, `seq`, `dma`) has no vendor primitive and no
  memory structure that only an FPGA can build; vendor cells live in `rtl/tech/<family>`.
- Every target's RTL matches the ISA simulator bit for bit (the existing token-exact tests), so
  the simulator stays the single reference.
- The YPCB card keeps working unchanged: its bitstream's netlist is unchanged apart from the new
  registers, and the host still runs older bitstreams.

## What varies between targets

| Layer | Examples | Who sees it |
|---|---|---|
| arch (ISA-visible) | S, D, MCOLS, LANES, ACT_BLOCKS / ACT_ROWS, TMEM / IMEM words, DRAM bytes per slice, PAIR / DSTEP / STREAM | the compiler and the programs (`opentpu.isasim.Config`) |
| uarch (timing only) | MXU_IMPL, VPU_CL, ULANES, TMEM_IMPL, RPB / WPB, WIN, FIFO_DEPTH | the RTL, the co-simulation and cost models |
| tech | RAMs (block RAM inference, NanGate45 fake macros, foundry macros), the MXU's MAC cells (DSP48E1 / DSP48E2 / generic), clock gates, synchronizers | nothing above the RTL |
| platform | memory (DDR3 LiteDRAM, DDR4, an ASIC's port), channels per slice, host link (XDMA Gen2 x8, Gen3 x16), clocks, pins | the host runtime (address map, calibration) |

Programs depend on the arch only. Two targets with the same arch run the same compiled programs
(the YPCB card and an ASIC built with the card's arch; the ASIC's RTL simulation then reuses the
card's token-exact tests unchanged).

## Decisions

### D1. Target files

`targets/<name>.toml`, one per target:

```toml
[target]
name = "ypcb-00338"
kind = "fpga"                # fpga | asic | sim
platform_id = 1              # CFG_PLATFORM register
[arch]                       # opentpu.isasim.Config fields, the hardware's truth
S = 1
D = 128
...
[uarch]                      # timing-only RTL parameters
MXU_IMPL = 2
...
[tech]
family = "xilinx7"           # generic | xilinx7 | xilinxus | asic
part = "xc7k480tffg1156-2"
[platform]
board = "ypcb-00338"
core_mhz = 133.33
mem = "ddr3"
mem_mts = 1066
channels_per_slice = 2
channel_bytes = 2147483648
host = "pcie"
pcie_gen = 2
pcie_lanes = 8
```

Initial targets: `ypcb-00338` (production), `bcu1525` (S = 2, two DDR4 channels per slice),
`nangate45` (the card's arch, the ASIC uarch: banked TMEM, generic MAC), `nangate45-small` (a
small arch for fast flow iterations and the future shuttle die).

### D2. `opentpu/target.py`, the one loader

`load(name)`, `current()` (`OTPU_TARGET`, default `ypcb-00338`), `config(t, **kw)` (a
`Config`), `uarch(t)` (the simulators' timing knobs), `rtl_params(t)` (the top's generics),
`sources(t)` (the RTL file list: core + tech + shell + board), `arch_hash(cfg)` (a stable hash
of the `Config` fields), and a CLI (`python3 -m opentpu.target show|env|json|files NAME`) that
the shell / Tcl flows read.

`Config` keeps its fields (they are in every cache key); platform and uarch values are not added
to it. `board_config()` keeps its meaning and defaults ("the card's sizes, the optional features
from `OTPU_PAIR` / `OTPU_DSTEP` / `OTPU_STREAM`") but takes the sizes from `current()`;
`target_config(name)` is the full hardware truth. The five duplicated RTL file lists (rtlsim,
the board Makefile, `create_project.tcl`, `tools/synth`, the tourney) come from `sources()`.

### D3. Config ROM: register map 4

Free offsets 0x80..0xFC of the control block hold the rest of the configuration:

| Offset | Name | Contents |
|---|---|---|
| 0x80 | CFG_SLICES | S |
| 0x84 | CFG_TMEM | TMEM words |
| 0x88 | CFG_IMEM | IMEM words |
| 0x8C | CFG_ACT | ACT_BLOCKS |
| 0x90 | CFG_DRAM | [7:0] log2 DRAM bytes per slice, [15:8] channels per slice, [23:16] log2 bytes per channel |
| 0x94 | CFG_UARCH | [3:0] MXU_IMPL, [7:4] TMEM_IMPL, [15:8] VPU_CL, [23:16] ULANES |
| 0x98 | CFG_PLATFORM | the target's platform_id |
| 0x9C | CFG_HASH | low 32 bits of arch_hash at build time |

REGMAP reads 4. `Board.info()` reads them on map 4; `device_config()` builds the `Config` from
them (map 3 and older: the YPCB values, as today). `BoardBackend` compares the whole arch (not
only D / MCOLS / LANES / ACT_ROWS) and the DRAM a program needs against the device's.

### D4. RTL layout

- `rtl/top`, `rtl/mxu`, `rtl/vpu`, `rtl/mem`, `rtl/seq`, `rtl/dma`: the core, unchanged paths.
- `rtl/tech/<family>/`: the same module names in each family. `otpu_pe` and `otpu_colend` (the
  systolic MXU's cells) leave `otpu_mxu.sv`: `generic` holds today's behavioural models,
  `xilinx7` today's DSP48E1 code (the YPCB netlist stays the same), `xilinxus` a DSP48E2 port.
  The RAM wrappers (`otpu_ram_sdp`, and the 1R1W / 1RW wrappers the ASIC memories need) live here
  too. `ifdef SYNTHESIS` no longer selects cells; the file list does.
- `rtl/shell/`: what every PCIe + native-memory FPGA board shares, moved from
  `rtl/boards/ypcb-00338`: `otpu_board`, `otpu_ctrl`, `otpu_trace`, `otpu_afifo`, `otpu_mem_ch`,
  `otpu_axi_split2`.
- `rtl/boards/<board>/`: the board's top, clocking and its memory / host wiring.
- `rtl/asic/`: the ASIC top (slice + control + memory adapter, ports to the pads or the macro
  boundary).
- `boards/<board>/` (Vivado / Quartus flows, constraints, generated memory cores) and
  `asic/<pdk>/` (the ASIC flows).

### D5. Memories that an ASIC can build

The FPGA TMEM keeps one copy per read port (`TMEM_IMPL = repl`). The ASIC TMEM (`TMEM_IMPL =
banked`) has LANES banks, each one 1R1W macro (RPB = 1, WPB = 1: the slice's arbiter already
enforces per-bank read and write limits). Large RAMs (ACT RAM, IMEM, the MXU and DMA chunk
FIFOs, the stream engine's buffers) go through the tech RAM wrappers; small asynchronous-read
LUT RAMs become flip-flops. NanGate45 has no memory compiler: `tools/asic/fakeram.py` (after
bsg_fakeram, whose macros are 1RW only) generates 1RW and 1R1W abstract macros (LEF, Liberty,
Verilog model). Foundry 2-port macros exist on IHP and commercial nodes; GF180 has 1RW only, so
a 1RW TMEM variant is a later option.

### D6. ASIC flow

`asic/nangate45/`: yosys (slang front end) synthesis against the NanGate45 Liberty with the fake
macros as black boxes, then OpenROAD (floorplan, macro placement, place, CTS, route), the clock
from the target. Output: area, Fmax (worst slack), power estimate, per target, in
`build/asic/<target>/`. The ASIC target's RTL simulation (banked TMEM, generic cells) runs the
token-exact tests; gate-level simulation comes later. Tapeout-rehearsal rules for the core:
synchronous reset style kept consistent, clock gates and synchronizers only through the tech
layer, no initial-value reliance in synthesis (already true), no inferred multi-port RAM.

### D7. Size-general compiler

Constants that bake in this card (`ALIGN = 128`, `HEAD_CHUNK`, `MLP_CHUNK`, `PREFILL_ROWS`,
`PROG_SLOT`, `CARD_BYTES = 1 << 32`, the snapshot `CHUNK`, ...) derive from `Config`. A config
matrix (the small ASIC arch, the card's, the BCU1525's S = 2, the simulator design point) runs
the kernels and a small model token-exact on the ISA simulator, and on the RTL for small ones.

### D8. BCU1525

S = 2 slices, each on two DDR4 channels through the existing two-channel adapter
(`otpu_native_dram`, unchanged), so each slice is the YPCB slice's memory system twice over;
LiteDRAM's UltraScale+ DDR4 PHY (the YPCB flow already generates LiteDRAM cores) or the MIG;
XDMA Gen3; DSP48E2 (`rtl/tech/xilinxus`); development builds at 100 MHz first. The multi-slice
board (`otpu_board` with S slices and the collective unit, one control block: RUN and LOAD
shared, per-slice ICOUNT) and the host's multi-slice backend are new. Pins come from LiteX's
`sqrl_xcu1525` platform (BSD-2; the BCU1525 / XCU1525 are VCU1525 derivatives). Before the
boards and the license: Verilator lint and a board-model simulation of the S = 2 build, yosys
resource estimates for `xcup`, and Vivado runs on an installed Kintex UltraScale+ part for the
DSP48E2 cells and the DDR4 / Gen3 IP.

Each slice addresses 4 GiB (32-bit DRAM addresses), so S = 2 uses 8 of 64 GB at a time. The
rest needs an ISA decision later (a per-slice window register the host sets, or far-memory DMA
for MoE expert pools held on the card instead of the host); the window register's offset is
reserved now (0xA0 + 4s).

## Sub-projects and order

1. Targets and config plumbing (D1-D3). Foundation; no datapath change.
2. Core / shell split and tech layer (D4), unified file lists.
3. ASIC memories, flow and PPA (D5-D6).
4. Size-general compiler (D7).
5. BCU1525 platform (D8), as far as possible before the hardware arrives.

1 first; then 2; 3, 4 and 5 in parallel once 2 is in. Each lands on `main` after its gates
(a laptop's subset, the build host full gate with RTL), the YPCB build checked by lint and, where the netlist
can change, by a Vivado build compared against production.
