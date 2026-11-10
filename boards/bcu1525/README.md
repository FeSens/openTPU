# openTPU on the BCU1525

The BCU1525 (and the XCU1525: builds of Xilinx's VCU1525) is a Virtex UltraScale+ VU9P
(`xcvu9p-fsgd2104-2L-e`) with four DDR4 DIMM slots, PCIe Gen3 x16 and two QSFP28 cages. This
directory builds openTPU for it and brings the image up on the card. The reference pages:

| | |
|---|---|
| [docs/bcu1525.md](../../docs/bcu1525.md) | the board, its pins, the LiteDRAM images, the BMC, the thermal governor, the fan |
| [docs/bcu1525_wg.md](../../docs/bcu1525_wg.md) | the production image: one slice at WG 8 on AMD's DDR4 MIG, 150 MHz |
| [docs/bcu1525_s2.md](../../docs/bcu1525_s2.md) | two slices on a card, and two cards over QSFP28 |
| [docs/bcu1525_sysmon.md](../../docs/bcu1525_sysmon.md) | the SYSMON wizard's temperature bus: do not poll its DRP registers |
| [docs/targets.md](../../docs/targets.md) | the targets and their variants |

## What you need

- **Vivado with a license for the VU9P.** The images here were built with Vivado 2026.1. The
  VU9P is not in the free tier: the Virtex UltraScale+ device files install without a license,
  but Vivado lists only the parts a license enables. `run_vivado.sh` checks first (its preflight,
  about a minute) and stops with what is missing. `make ipcheck` and `make synthcheck` run the
  block design and a whole-project synthesis on the free `xcku5p` instead.
- **A build host with 32 GB of RAM or more.** One slice peaks at 21.6 GB (routing); two slices
  are expected near 28-31 GB with one job (an estimate). A full one-slice build takes about 1 h 50 min.
- **Four DDR4 UDIMMs** for the WG images. They are generated for 8 GB DDR4-2400 CL17 UDIMMs of
  one rank of x8 parts without ECC (GOODRAM GR2400D464L17S/8G, with Micron's MTA8ATF1G64AZ-2G3
  as the MIG's part). All four slots must be populated: the accelerator runs only with every
  DIMM calibrated. Other DIMMs need the MIG's part and the address widths changed in
  `vivado/bd_wg.tcl`. The LiteDRAM images take the same UDIMMs (`DIMM=udimm`, the default) or
  the VCU1525's 16 GB x4 ECC RDIMMs (`DIMM=rdimm`, not run on a card here).
- **Cooling.** These cards came without a fan. Every image starts with everything off behind a
  thermal governor (docs/bcu1525.md section 14), which grants the accelerator only below its
  thresholds; a fan on the heatsink and `scripts/fan_pid.py` (section 15) are what let it run.
- **The card host:** Linux with the XDMA driver (`sudo otpu-setup`, docs/host.md), `pip install
  -e .` of this repository, and for the JTAG steps Vivado or Vivado Lab's hardware server and
  the card's USB cable (its on-board FT4232H). `python3` on the build host too: the flow reads
  the target through `opentpu.target`.

## Building

    make wg          # the production image: bcu1525-wg:bhw16 at 150 MHz -> build/vivado_bcu1525_wg
    make wg-dev      # the same target at 100 MHz with development strategies (a functional image)
    make s2          # two slices: bcu1525-s2:st16, floorplan c         -> build/vivado_bcu1525_s2
    make x2          # one card's image of the two-card machine, 100 MHz -> build/vivado_bcu1525_x2
    make bit         # the LiteDRAM image (bcu1525: two slices at MCOLS 4, DDR4-1200)
    make bit IMAGE=mem   # the memory-test image, no accelerator

`make wg` is two Vivado runs, synthesis and then implementation (`wg-synth`, `wg-impl`), so that
implementation starts in a fresh process. What it runs:

    IMAGE=wg WG_TARGET=bcu1525-wg:bhw16 CORE_MHZ=150 FAST=0 WG_MIGMAP=b15 OTPU_IP_CACHE= \
      OUT_DIR=<repo>/build/vivado_bcu1525_wg STEP=synth ./run_vivado.sh       # then STEP=impl

`WG_TARGET`, `WG_MHZ` and `WG_OUT` override the target (any `bcu1525-wg` variant), the clock
(100, 120, 150) and the directory. `WG_MIGMAP=b15` patches the MIGs' generated address decode,
which is why the IP cache is off (`OTPU_IP_CACHE=` empty). Every knob of `run_vivado.sh` is in
its header and in docs/bcu1525_wg.md section 5.

The outputs, in the build directory: `otpu.bit`, `otpu.ltx` (the debug probes, which the bring-up
uses to read each MIG's calibration report), `otpu.mcs` (the flash image) and `reports/`
(`SUMMARY.txt`, `timing_summary.rpt`, utilization per SLR). **Check `reports/SUMMARY.txt` for
WNS and WHS at or above zero before loading an image**: the one-card flows write a bitstream
either way (the two-card script refuses negative slack).

If a 150 MHz build misses timing by a little, rerun the implementation step by step with other
directives; a finished opt_design is the starting point:

    $(WG_ENV as above) STEP=impl IMPL_ALONE=1 IMPL_TO=opt_design ./run_vivado.sh
    OTPU_WG_PLACE=ExtraPostPlacementOpt OTPU_WG_PHYS=AggressiveExplore \
    OTPU_WG_ROUTE=Explore OTPU_WG_POSTPHYS=AggressiveExplore \
      vivado -mode batch -source vivado/impl_resume.tcl -tclargs <build dir>

`impl_resume.tcl` keeps a marker for each finished step, so a rerun continues where it stopped,
and ends with the same reports and bitstream.

**Two cards.** `make x2` runs `x2/build_x2.sh`, which builds `bcu1525-x2:st16` at 100 MHz with
development strategies and refuses anything else: it is the functional image (links, collectives,
bit-exactness), and both cards load the same bitstream (a card's number is a register the host
writes). A 150 MHz two-card image is a `run_vivado.sh` build with `WG_TARGET=bcu1525-x2:st16
WG_PLAN=c WG_FP=hard` and the directives above; it has no wrapper here.

`make lint` is the offline check (Tcl syntax, pins against the tops, Verilator lint with
stand-ins for the vendor IP): no Vivado needed.

## Loading and bring-up (WG images)

Two steps, with a warm reboot between them (`scripts/wg_bringup.py`; its docstring has every
stage):

    python3 scripts/wg_bringup.py load --bit <build dir>/otpu.bit --out runs/a --serial <FT4232H serial>
    sudo systemctl reboot            # warm: the image stays, the BIOS enumerates the card
    python3 scripts/wg_bringup.py up --out runs/a --ltx <build dir>/otpu.ltx --mem full \
        --serial <FT4232H serial> --selftest-args "--model qwen3"

- `load` programs the image over JTAG (volatile: it stays until the card loses power) and checks
  DONE. `--serial` is the JTAG serial of the card's FT4232H; **pass your own** (the default in
  the script is one installation's).
- `up` works through PCIe (`--dev`, default `/dev/xdma0`). In order: the governor's RESUME
  threshold (`--resume`, default 65 C), SAFE as configured, the guard (heartbeat, die,
  regulator), a forced thermal trip and a heartbeat trip (`--no-trips` skips them), the four
  DIMMs on (each MIG calibrates itself), each MIG's calibration report over JTAG (`--ltx`;
  `--no-jtag` skips what needs the cable), the accelerator on, otpu-diag's memory section
  (`--mem quick`, the default; `full` adds a march C- over every byte; `none`), otpu-selftest
  (`--selftest-args "--model qwen3"` adds a model stage: Qwen3-0.6B decoded on the card and
  compared with the simulator token for token; `--no-selftest`), then each `--run "MODULE
  ARGS"`, and SAFE at the end. Everything it writes is under `--out`.
- `--hold MIN` keeps the card in RUN under the guard for that long, for other processes.

A bring-up that ends `ALL PASS` with the model stage is the image qualified on that card. The
LiteDRAM images use `scripts/memtest_jtag.py` (image M) and `scripts/fullimage_jtag.py` (image
F); docs/bcu1525.md sections 12 and 14 walk through them.

`make flash` writes the SPI flash and is permanent; nothing here needs it. A JTAG-loaded image
survives warm reboots.

## Running a model

While a bring-up holds the card in RUN (`up ... --hold 600` in one terminal), the usual tools
run against it from another, with the target named so that programs are compiled for this
image:

    export OTPU_TARGET=bcu1525-wg:bhw16
    otpu-smi                                        # the card: governor, die, power, link
    otpu-selftest --model qwen3
    otpu-chat --backend board --model qwen3         # the one card up; --dev /dev/xdmaN to name one

The large models (Qwen3.6-35B-A3B with its whole 262,144-token context resident, on one card
or two) need their card images built ahead of time on a host with enough RAM, and run through
the wrappers in `chat35/` (its README), which do the bring-up, hold RUN, serve the chat or an
OpenAI-style API, and return the card to SAFE.

## Things that will bite

- **Do not poll the SYSMON's registers over PCIe.** A read in BAR0's 0x30400-0x307FC can stop the
  governor's temperature for good (docs/bcu1525_sysmon.md). The tools here do not; a monitor of
  your own should read the governor's registers instead.
- **A chipset slot trains at Gen2 x4.** The image works there (host-to-card about 1.4 GB/s);
  otpu-diag's PCIe check fails there on the link's width and speed, and nothing else.
- **One Vivado at a time on a 32 GB host**, and `JOBS=1 SYNTH_THREADS=1` at two slices.
- **The image starts SAFE.** Registers behind the governor read 0xDEAD0FF0 until a bring-up or a
  wrapper asks for RUN and the die is below RESUME.
