# bcu1525-x2: one card's image of the two-card machine

The build and bring-up of `bcu1525-x2:st16`: two BCU1525s, two slices each, joined by two QSFP28
cables into a four-slice tensor-parallel machine. The system is docs/x2.md and the link
docs/x2_link.md; this page is the files here, the build's command, what it checks, and the order
of a bring-up.

Both cards load the same bitstream. Which card is which is a register written at run time
(X2_CARD), never a build option.

## Files

| file | what it holds |
|---|---|
| `build_x2.sh` | the launcher: generated inputs, synthesis and implementation in separate Vivado processes |
| `sources_x2.f`, `project_x2.tcl` | the link's sources and the project's additions to the single-card wide-gather image |
| `aurora_ip.tcl` | the two Aurora 64B/66B cores: GTY, four lanes, 25.78125 Gb/s, 161.1328125 MHz reference, streaming, the core's own CRC off (the link has its own, docs/x2_link.md) |
| `pins_x2.xdc` | the QSFP28 cages' transceiver and reference pins. The pin mapping is Corundum's for the AU200 / VCU1525 (BSD-2-Clause-Views, THIRD_PARTY.md) |
| `clocks_x2.xdc`, `cdc_x2.tcl`, `constraints_x2.tcl` | the link's clocks; each crossing bounded on its own (`set_max_delay -datapath_only` and `set_bus_skew` per Gray count, FIFO pointer and held bus) rather than by a clock group, which would override those bounds |
| `floorplan_x2.tcl` | the link's pblock in SLR2; the adapters' register stages either side of the SLR2 / SLR1 boundary |
| `refclk_check.tcl`, `post_x2.tcl` | the checks after routing and the bitstream's gate |

## The card's configuration

| | |
|---|---|
| compiler target | `bcu1525-x2:st16`: S = 4, two cards (machine hash D5239726) |
| one card | S = 2, NCARDS = 2, CFG_PLATFORM 7, CFG_HASH E331F79C, PCIe x16 |
| floorplan | plan c: slice 0 in SLR0, slice 1 in SLR2, the hub, control and PCIe in SLR1, the link in SLR2 |
| QSFP0 | GTY X1Y48-51, common X1Y12 |
| QSFP1 | GTY X1Y44-47, common X1Y11 |
| reference | one buffer for both quads, from QSFP0's MGTREFCLK1 pair (K11 / K10) |

The image has no ports on the board's clock generator's reset and frequency-select pins: an
earlier link image that drove them from a clock that could stop deadlocked, and the generator's
power-on output is the reference the link needs.

## Build

```sh
bash boards/bcu1525/x2/build_x2.sh OUT             # a new directory
bash boards/bcu1525/x2/build_x2.sh --prepare OUT   # the generated inputs only, no FPGA tools
RESUME=1 bash boards/bcu1525/x2/build_x2.sh OUT    # implementation again on a finished synthesis
```

- The launcher builds the development image: `FAST=1`, `CORE_MHZ=100`. It refuses another clock,
  `FAST=0`, a `WG_SET`, a directory that already holds a project (without `RESUME=1`), and a
  resume whose saved clock, topology, hash or build id differ.
- `--prepare` runs `boards/bcu1525/scripts/make_wg.py` (the card's defines and parameters, the
  hash) and stops; it needs no Vivado.
- It prints the hash and topology before Vivado starts, and after routing the slacks, the GT
  sites, the reference's path and each leaf's SLR (`OUT/reports/x2_summary.txt`).
- **The gate.** No bitstream unless WNS >= 0, WHS >= 0 and the pulse-width slack >= 0, both
  commons and all eight channels are at the sites above, both QPLLs trace to the one reference
  pad, and the link's and the hub's leaves are in their SLRs. `OTPU_X2_WNS_MIN` (-0.1 to 0 ns,
  default 0) is a development image's setup allowance; hold and pulse width have none.
- Outputs: `OUT/otpu.bit`, `OUT/otpu.mcs`, `OUT/status.txt`, and under `OUT/reports/`
  `SUMMARY.txt`, `x2_summary.txt`, `timing_summary.rpt`, `util_slr.rpt`, `cdc.rpt`,
  `x2_bus_skew.rpt` among others.
- `JOBS` and `SYNTH_THREADS` (default 2 each) and `NICE` (15) size the run; `VIVADO` names the
  binary.

The measurements in docs/x2.md are from a 150 MHz image, which this launcher does not build: it
came from the same sources through `boards/bcu1525/run_vivado.sh` with the full directives, then
a post-route pass (phys_opt AggressiveExplore and a route Explore) that left a worst setup
slack of -0.035 ns, inside the gate's 0.1 ns allowance, and one failing pulse-width check
(-0.173 ns, Min Skew on a MIG's TX bitslice inputs), for which the gate has no allowance: a
development image (docs/x2.md section 6).

## What to review in a first build

1. Both generated cores' properties and port widths; no unbound primitive, no second STARTUPE3,
   no Tcl command silently ignored (a missing constraint selector stops the run before
   opt_design).
2. Setup, hold and pulse width; the CDC and methodology reports, `x2_bus_skew.rpt` and the
   exceptions' coverage. A passing timing summary alone does not qualify the crossings.
3. The eight GT sites, both commons and the reference's select paths.
4. Leaf placement by SLR, the four crossing reports (`x2_p{0,1}_{up,down}.rpt`), and no long
   direct path from the hub to the QSFP adapters.
5. Use by SLR. SLR2 holds slice 1 and the link.

## Bring-up

1. Load both cards with the same bitstream.
2. Each card on its own, as a single card: thermal governor, DDR4 calibration, PCIe, identity
   and build id, the configuration words above, memory.
3. The pair. `opentpu.host.cards.CardsTransport` does the link's part when it opens the two
   devices: X2_CARD on each, a link reset on both stopped cards, both ports up on both cards,
   the hellos and the cable map (docs/x2.md section 3). Straight and crossed cables both pass.
4. The qualification:

```sh
python3 tools/x2_card.py --dev /dev/xdmaA+/dev/xdmaB --expect EXPECT.json --json qual.json
python3 tools/x2_card.py --dev /dev/xdmaA+/dev/xdmaB --json perf.json --steps a,b \
    --cuts PACKS/q36-4,PACKS/q38-4,PACKS/goss-2
```

   Card 0 is the first device. The steps: **a** both cards' identity; **b** the cable map, a
   ping on every port and clean link counters; **c** BARs and a GATHER checked byte for byte;
   **d** Qwen3-0.6B (int8) decode and device generation against the token ids in `EXPECT.json`
   (`{"qwen3_0_6b": {"prompt": [...], "next_tokens": [...], "generated": [...]}}`, computed away
   from the card host); **f**, with `--cuts`, the decode cuts, `--cut-reps` times each (3). Every
   launch records both hubs' sequences and every link counter; nothing retries or resets a link
   behind a fault. The tool keeps both cards' thermal governors armed while it runs and leaves
   the cards in their safe state.

5. The cuts' packs are made away from the card host, so the card only copies bytes, runs and
   checks:

```sh
python3 tools/x2_cutpack.py --out PACKS             # q36, q38, goss: programs, wait entries, ISA results
python3 tools/x2_cutcheck.py PACKS/q36-4 --json check.json   # a pack through the two-board RTL model
python3 tools/x2.py compile POINT                   # a point's programs and image bytes by card
```

## Simulation

`sim/verilator/tb_x2.sv` is two of these boards and two cables; `tools/x2_cosim.py` times a cut
on it, and `tests/test_cards.py`, `tests/test_coll_x2.py`, `tests/test_x2_link.py` and
`tests/test_x2_fpga.py` are the gates (docs/x2.md section 9). `tests/test_x2_fpga.py` needs no
FPGA tools: it checks the generated inputs, lints the full top at 100 and 150 MHz and elaborates
the hierarchy.
