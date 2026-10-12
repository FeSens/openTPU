# openTPU on NanGate45 (OpenROAD)

The ASIC block (`rtl/asic/otpu_asic_top.sv`: one slice and its collective unit, the slice's memory
ports at the boundary) through OpenROAD-flow-scripts on the NanGate45 library (FreePDK45, a
predictive 45 nm kit: not manufacturable, the usual open PPA reference). `docs/asic.md` has the
design choices and the results.

## Tools

OpenROAD-flow-scripts (ORFS: OpenROAD, OpenSTA, yosys with slang, KLayout, and the `nangate45`
platform with its standard cells and fakeram45 macros), built natively (recommended) or as ORFS's
Docker image.

**Native** (Linux; the reference run: Ubuntu 26.04 on x86-64). Build ORFS once and point `ORFS_HOME` at it; `flow.py` uses its OpenROAD
and its yosys, nothing else is needed on `PATH`:

```
git clone https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts.git
cd OpenROAD-flow-scripts
git checkout f395e4b6ba30355564ab26997a01cb844678a65e   # the revision of the reference results
git submodule update --init --recursive
sudo ./etc/DependencyInstaller.sh -base                    # system packages
./etc/DependencyInstaller.sh -common -prefix=$PWD/dependencies
./build_openroad.sh --local --threads $(nproc)             # yosys + OpenROAD into tools/install
export ORFS_HOME=$PWD
```

The build takes about 20 minutes on 20 cores. ORFS's `flow/designs/nangate45/gcd` example checks
it (`make DESIGN_CONFIG=designs/nangate45/gcd/config.mk` in `flow/` after `source env.sh`: about
30 s). `-base` supports Ubuntu 20.04-24.04 (and RHEL-family systems); on a newer Ubuntu it stops at
KLayout, after the system packages. There, install KLayout and the flow's Python packages from the
distribution (`sudo apt-get install klayout python3-pandas python3-numpy python3-click
python3-yaml`; KLayout 0.30.0 works). The steps `-base` then skips only install KLayout's release
package and Docker, which the native flow does not use.

**Docker** (any host with Docker; Apple silicon runs the amd64 image under Rosetta, several times
slower): `docker pull --platform linux/amd64 openroad/orfs:latest` (6.5 GB) and leave `ORFS_HOME`
unset. `run` starts the container with the tree mounted at its own path. `prep` still runs on the
host: it needs yosys with the slang plugin there (`yosys -m slang`, e.g. from the OSS CAD Suite;
`YOSYS` names another binary).

**Resources.** nangate45-small: 15 GB of memory at its peak and 81 minutes on 20 cores natively
(the reference run below). The card's point needs more than 32 GB (global routing) and many hours.

## Run

```
python3 tools/asic/flow.py points                  # the design points, their parameters, clock
python3 tools/asic/flow.py all small               # prep + ORFS + report  (build/asic/small/)
python3 tools/asic/flow.py all card                # the card's arch
ORFS_TARGET=cts python3 tools/asic/flow.py run card            # stop after CTS (make target)
ORFS_CLEAN=floorplan python3 tools/asic/flow.py run small      # redo from the floorplan on
python3 tools/asic/flow.py report small            # report.json from the latest stage's reports
python3 tools/asic/paths.py small 300 6_final      # the critical paths by unit and stage
python3 tools/asic/paths.py small 300 4_cts        # the same, after CTS (placement parasitics)
```

`ORFS_CPUS` is ORFS's NUM_CORES (and the container's CPUs): every core natively, 6 in Docker by
default. `ORFS_IMAGE` is the Docker image.
`make` resumes from the stage files that exist: an interrupted `run` continues where it stopped
(after a config change, clean from the stage it affects). The sources are the worktree's own: an
RTL file newer than the run's synthesis makes `make` start over from synthesis; to run one stage
on the results there are, name its `do-` target (`ORFS_TARGET=do-finish`, `do-grt`, ...).
`ORFS_VARS` adds ORFS variables to a run (`ORFS_VARS="SKIP_INCREMENTAL_REPAIR=1"`: no repair in
global routing, an hour less on nangate45-small).

Design points (`tools/asic/flow.py` POINTS, from the targets: arch, uarch, `core_mhz`):

| point | target | D | MCOLS | LANES | TMEM words | IMEM words | ACT blocks | FIFO | streams | clock asked |
|---|---|---|---|---|---|---|---|---|---|---|
| small | nangate45-small | 32 | 2 | 4 | 16K | 8K | 32 (2 rows) | 64 | no | 2.5 ns |
| card | nangate45 | 128 | 4 | 8 | 64K | 32K | 128 (4 rows) | 1024 | yes | 2.5 ns |

Both: TMEM_IMPL 1 (banked: three copies of 1R1W bank RAMs), RPB 1, WPB 1, the systolic MXU
(MXU_IMPL 2) with the generic cells (`rtl/tech/generic/otpu_tech_dsp.sv`), WIN 16. `*-lm3la6`:
the same with otpu_fpipe's LM 3 and LA 6 (see `docs/asic.md`: without retiming they move no logic).

## What prep makes (build/asic/POINT/)

- `src/tiles.json`: the RAM tiles (`rtl/tech/asic/otpu_tech_ram.sv` tiles every technology RAM
  onto macros of at most 128 bits x 4096 words).
- `macros/`: one 1R1W (or 1RW) macro per tile size (`tools/asic/fakeram.py`: LEF, Liberty,
  behavioural model, black box).
- `src/otpu_sram_map.sv`: the leaves (`otpu_sram_1r1w`, `otpu_sram_1rw`), one branch per macro
  size.
- `config.mk`, `constraint.sdc`: the ORFS design: the slang front end, hierarchical synthesis
  (the macro placer groups by it), utilisation 35%, one clock (`core_clock`), the ports timed
  against a virtual clock with 20% of the period outside.

## Settings that matter (and why)

- `SYNTH_MEMORY_MAX_BITS 16384`: the remaining flip-flop memories (the quantizer's block buffer,
  the dispatch window, the MXU's drain rows) are above ORFS's 4096-bit default; they stay flops.
- `RTLMP_MIN_CHANNEL_SIZE 60 60`: the channels between macros must be wider than the M4 power
  straps' pitch (56 um), or PDN fails (PDN-0179, "Unable to repair all channels").
- `MACRO_PLACE_HALO 10 10`, `PLACE_DENSITY_LB_ADDON 0.10`, `TNS_END_PERCENT 100`.
- The macros' pins sit on the 0.14 um track grid (tools/asic/fakeram.py `GAP`): an off-grid pin
  stops the macro placer (MPL-0002).

## Outputs

`build/asic/POINT/report.json` (`flow.py report`): the stage reported, `wns_ns` and `fmax_io_mhz`
(every path, the ports' budgets included), `period_min_ns` and `fmax_mhz` (register to register:
`report_clock_min_period`), `power_w` (vectorless, OpenSTA's default activities; by group),
`units` (area per unit after synthesis, with its macros' share), the macros and their area, and
ORFS's metrics per stage. `paths_<stage>.txt*` / `.json` (`paths.py`): the worst paths, every
endpoint's worst slack, the stage classes. ORFS's own logs, reports and results are in `work/`.

## Reference results

`results/small/`: nangate45-small through this flow with the ORFS revision above (OpenROAD
26Q4-110-g91e5f5ff2f, yosys 0.68+): `report.json` (`flow.py report`), ORFS's final report
`6_finish.rpt` and the routed layout. A run of the same sources and tools should match it within
the tools' run-to-run noise (a few MHz). `docs/asic.md` has the analysis.

## A denser floorplan with planned macro placement

`asic/nangate45/openlayout/`: the same block at 60% utilization with the macros placed by plan
(OpenLayout's planner and checker skills) and plain Yosys + OpenROAD scripts: routed, timing met at
3.3 ns, no DRC violations on a 4.20 mm² die. Its README has the flow, the candidates and the results.

## Where to start

`docs/asic.md` ("Open problems") lists the known limits: the ports' combinational paths, the
sequencer's dispatch path, the grant fanout behind the stall-by-enable design, power from real
activities instead of vectorless, real 1R1W macros instead of the fitted ones, and the card point's
floorplan (its global routing ends congested). For a change to the flow (`config_mk` in
`tools/asic/flow.py`, `tapcell.tcl`) or to the RTL, run nangate45-small and compare its
`report.json` with `results/small/`.
