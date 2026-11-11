#!/usr/bin/env bash
# Build the openTPU bitstream for the BCU1525 (xcvu9p-fsgd2104-2L-e) in Vivado batch mode
# (docs/bcu1525.md). The knobs, as boards/ypcb-00338/run_vivado.sh's:
#   S=2 MCOLS=4 VPU_CL=2 LANES=8 ACT_ROWS=MCOLS DSTEP=1   the accelerator (S slices, two DDR4 channels each)
#   CORE_MHZ=100 (the default: development builds at 100 MHz until the design runs; FAST=1: Vivado's
#   default strategies)   PCIE_LANES=16|8 (Gen3)   TMEM_URAM=0|1 (the TMEM in UltraRAM)   BUILD_ID
#   TMEM_IMPL=0|3 TMEM_X=NAME=value,... (3: one copy, by default pumped at 2x on core_clk2x; the
#   YPCB's vivado/tmem_knobs.tcl)
#   DIMM=udimm|rdimm (udimm: x8 UDIMMs without ECC, the lab card's; rdimm: the card's
#   original x4 ECC RDIMMs; create_project.tcl OTPU_DIMM)
#   OUT_DIR   JOBS   STEP=bd|synth|impl   IMPL_ALONE=1   IMPL_STRATEGY   IMPL_TO=STEP   PREFLIGHT=1|0
#   SYNTH_THREADS=N
#   IMAGE=mem: the memory-test image instead (vivado/create_project_mem.tcl: the four UDIMMs' cores,
#   the thermal governor, JTAG-to-AXI; no PCIe, no accelerator; docs/bcu1525.md section 14)
#   IMAGE=wg: the BCU1525 at WG 8 (targets/bcu1525-wg.toml: the WG targets' one slice on the four DIMMs
#   through AMD's MIG at DDR4-2400; scripts/make_wg.py, vivado/create_project_wg.tcl): CORE_MHZ
#   (an even divide of 1200: 100, 120, 150), WG_SLR=0|1|2 (the slice's SLR, make_wg.py --slr: the
#   default 2), RLY=N0,N1,N2,N3 (each DIMM's bridge pipelines, make_wg.py --rly: the default
#   follows WG_SLR), WG_SET="K=V ..." (make_wg.py --set), WG_FP=soft|hard|none (the slice's
#   pblock), WG_MIGMAP=b15 (the MIGs' bank from the DIMM address's bits 16:15, vivado/mig_map.tcl:
#   with OTPU_IP_CACHE= ), WG_MIGRD=s62 (the MIGs' ORDERING Strict and their AXI read channels'
#   FIFOs at 62 / 64: more reads in flight; mig_map.tcl, with OTPU_IP_CACHE= ); the other knobs
#   are the target's. WG_TARGET=bcu1525-s2: two slices, two DIMMs each (make_wg.py --target),
#   WG_PLAN=a|b|c their floorplan (make_wg.py --plan, scripts/bcu_fp.py; default c, the plan: a and
#   b overfill SLR1), WG_FP the slices' pblocks as for one. WG_MXU_COLS<s>="3 4 ..." slice s's MXU
#   DSPs on those DSP48E2 columns of its SLR (pblock_mxu<s>; empty: the placer's). WG_STOP=synth:
#   synthesis, its reports and checks (the floorplan's cells, each pblock's share:
#   vivado/bcu1525_wg_synth_check.tcl, at two slices always), then stop (exit 1); WG_STOP=link:
#   on into the implementation, stopped after its link and constraint checks, before opt_design
#   (vivado/bcu1525_wg_check.tcl: the linked netlist's memory). CLK_HOOK=clk133.tcl (a Tcl file
#   at the end of those checks, before opt_design; a bare name vivado/'s): a 150 MHz synthesis
#   implemented at 133.33 MHz (the MMCM's 2x and core divides 4.5 and 9, the fallback constants'
#   selects; it stops the run if a property does not read back)
# STEP=bd (IMAGE=wg) stops after create_project_wg.tcl: the block design, validate_bd_design and
# bd_wg.tcl's checks (WG_HRS's rs_h<d> among them), in minutes and without synthesis.
# STEP=synth stops after synthesis and its reports, STEP=impl implements a finished synthesis: on a
# 32 GB host run them as two calls (the full flow's Vivado keeps the synthesized design's ~6 GB
# into the implementation's ~10+ GB). SYNTH_THREADS caps synth_design's helper processes (its
# general.maxThreads: up to 7 with JOBS 8, each with its share of the design: ~19 GB for the VU9P
# top at S 2); JOBS is the runs launched at once (the IPs' and the top's synthesis). IMPL_ALONE=1
# (with STEP=impl) runs impl_1's runme.sh without a Vivado holding the project meanwhile (~2 GB),
# then the reports and the bitstream in a new one. IMPL_TO=STEP (opt_design, ...; with STEP=impl)
# runs impl_1 to that step and stops, no reports: vivado/impl_resume.tcl goes on from its
# checkpoint (a killed step then costs only itself).
# The VU9P is outside Vivado's free tier: the license must cover it, and the Virtex UltraScale+
# device files must be installed (Vivado's installer, "Virtex UltraScale+"). The preflight
# (scripts/preflight.tcl, about a minute) checks both first and stops with what is missing.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
image="${IMAGE:-full}"
case "$image" in full|mem|wg) ;; *) echo "run_vivado: IMAGE=$image (full, mem or wg)" >&2; exit 2 ;; esac
# (an `[[ ]] && echo` substitution would end the script under set -e for the full image)
sfx=""; if [[ $image != full ]]; then sfx=_$image; fi
tname=bcu1525; if [[ $image == wg ]]; then tname="${WG_TARGET:-bcu1525-wg}"; fi
# a target or its variant (bcu1525-wg:bhw: opentpu.target and make_wg.py --target take it)
case "${tname%%:*}" in bcu1525|bcu1525-wg|bcu1525-s2|bcu1525-x2) ;; *) echo "run_vivado: invalid WG_TARGET=$tname" >&2; exit 2 ;; esac
out="${OUT_DIR:-$root/build/vivado_bcu1525$sfx}"
jobs="${JOBS:-8}"
vivado_bin="${VIVADO:-vivado}"
# The target (targets/bcu1525.toml through opentpu/target.py): the knobs' defaults, the platform
# id, the DDR4 rate and the arch hash the CFG_* registers report (without python3: the same
# built-in values, CFG_HASH 0)
if tenv="$(PYTHONPATH="$root" python3 -m opentpu.target env "$tname" 2>/dev/null)"; then
  eval "$tenv"
elif [[ $image == wg ]]; then
  echo "run_vivado: python3 -m opentpu.target env $tname failed" >&2; exit 2
else
  echo "run_vivado: python3 -m opentpu.target env bcu1525 failed: built-in defaults, CFG_HASH 0" >&2
fi
s="${S:-${T_S:-2}}"
mcols="${MCOLS:-${T_MCOLS:-4}}"
vpu_cl="${VPU_CL:-${T_VPU_CL:-2}}"
lanes="${LANES:-${T_LANES:-8}}"
act_rows="${ACT_ROWS:-$mcols}"
dstep="${DSTEP:-${T_DSTEP:-1}}"
# otpu_seq WAITC (EXPERIMENTAL, docs/seq_vpu.md; CFG_UARCH bit24, in CFG_HASH when on): the
# target's [arch] SEQ_WC, or SEQ_WC=1. The host writes each program's wait entries
seq_wc="${SEQ_WC:-${T_SEQ_WC:-0}}"
core_mhz="${CORE_MHZ:-${T_CORE_MHZ:-100}}"
platform="${T_PLATFORM:-2}"
ddr_mts="${T_DDR_MTS:-1200}"
tmem_impl="${TMEM_IMPL:-0}"
tmem_x="${TMEM_X:-}"
arch_hash="$(PYTHONPATH="$root" python3 -m opentpu.target hash bcu1525 MCOLS="$mcols" \
  LANES="$lanes" ACT_ROWS="$act_rows" DSTEP="$dstep" SEQ_WC="$seq_wc" 2>/dev/null || echo 0)"
# the tree's commit: COMMIT (a copy shipped without .git), else
# git's (a failed git under pipefail ended the script)
build_id="${BUILD_ID:-$(cat "$root/COMMIT" 2>/dev/null || git -C "$root" rev-parse HEAD 2>/dev/null || true)}"
build_id="${build_id:0:8}"
build_id="${build_id:-0}"
mkdir -p "$out"
out="$(cd "$out" && pwd)"             # absolute: the runs' Vivados work in their own directories
export OTPU_IP_CACHE="${OTPU_IP_CACHE-$HOME/.cache/otpu-vivado-ip}"
export OTPU_FAST="${FAST:-0}"
export OTPU_SEQ_WC_GEN="$seq_wc"
export PCIE_LANES="${PCIE_LANES:-16}"
export TMEM_URAM="${TMEM_URAM:-0}"     # 1: the TMEM copies in UltraRAM (vivado/tmem_uram.xdc)
export OTPU_DIMM="${DIMM:-udimm}"
export OTPU_WG_FP="${WG_FP:-soft}"     # IMAGE=wg: the slice's pblock (constraints/bcu1525_wg_floorplan.tcl)
export OTPU_IMPL_TO="${IMPL_TO:-}"     # build.tcl: impl_1 to this step only
export OTPU_WG_OUT="$out"              # IMAGE=wg: the floorplan's wg_params.tcl (make_wg.py's) is here
export OTPU_WG_ROOT="$root"            # IMAGE=wg: the tree (bcu1525-x2: x2/constraints_x2.tcl sources its files from here)
export OTPU_WG_MIGMAP="${WG_MIGMAP:-${OTPU_WG_MIGMAP:-}}"  # IMAGE=wg: the MIGs' address map (vivado/mig_map.tcl; b15 needs OTPU_IP_CACHE=)
export OTPU_WG_MIGRD="${WG_MIGRD:-${OTPU_WG_MIGRD:-}}"     # IMAGE=wg: the MIGs' reads in flight (bd_wg.tcl, mig_map.tcl; s62 needs OTPU_IP_CACHE=)
export OTPU_WG_STOP="${WG_STOP:-${OTPU_WG_STOP:-}}"        # IMAGE=wg: synth / link stops the build after synthesis's checks / the implementation's link
export OTPU_WG_MXU_COLS0="${WG_MXU_COLS0:-${OTPU_WG_MXU_COLS0:-}}"  # IMAGE=wg: slice 0's MXU DSP columns (bcu1525_wg_floorplan.tcl)
export OTPU_WG_MXU_COLS1="${WG_MXU_COLS1:-${OTPU_WG_MXU_COLS1:-}}"  # IMAGE=wg: slice 1's (two slices)
export OTPU_WG_MMCM="${WG_MMCM:-${OTPU_WG_MMCM:-}}"          # IMAGE=wg, two slices: the clock wizard's MMCM site (bcu1525_wg_floorplan.tcl; default MMCM_X1Y5, none: the placer's)
export OTPU_WG_CDG="${WG_CDG:-${OTPU_WG_CDG:-}}"            # IMAGE=wg: 1 = CLOCK_DELAY_GROUP on the accelerator's clock pair (bcu1525_wg_top.tcl)
export OTPU_WG_UMEM="${WG_UMEM:-${OTPU_WG_UMEM:-}}"         # IMAGE=wg, two slices: 1 = umem, the adapters and the bridges' core halves into SLR1 (bcu1525_wg_place_pre.tcl, wg_umem.tcl)
export OTPU_WG_MIGKO="${WG_MIGKO:-${OTPU_WG_MIGKO:-}}"       # IMAGE=wg: 1 (or DIMM numbers) = an EXCLUDE_PLACEMENT soft pblock beside each DIMM's PHY (bcu1525_wg_place_pre.tcl, wg_migko.tcl)
export OTPU_WG_MIGKO_W="${WG_MIGKO_W:-${OTPU_WG_MIGKO_W:-}}" # IMAGE=wg: its SLICE columns (default 12)
export OTPU_WG_PRS="${WG_PRS:-${OTPU_WG_PRS:-}}"             # IMAGE=wg: 0 / 1 = the DIMMs' ports S_AXI_P<d> through an AXI register slice (bd_wg.tcl rs_p<d>; unset: on at two slices)
export OTPU_WG_HRS="${WG_HRS:-${OTPU_WG_HRS:-}}"             # IMAGE=wg: 1 (or DIMM numbers) = the host DMA ports of C0 and C3 through an SLR-crossing AXI register slice (bd_wg.tcl rs_h<d>)
export OTPU_WG_X2PORT="${WG_X2PORT:-${OTPU_WG_X2PORT:-}}"   # IMAGE=wg, x2: 1 = the link's core-clock ports and the hub adapters' SLR2 leaves into SLR1 (bcu1525_wg_place_pre.tcl, wg_x2port.tcl)
export OTPU_WG_CPORT="${WG_CPORT:-${OTPU_WG_CPORT:-}}"       # IMAGE=wg, two slices: 1 = each slice's collective port and its crossing's slice-side stages into SLR1 (bcu1525_wg_place_pre.tcl, wg_cport.tcl)
export OTPU_CLK_HOOK="${CLK_HOOK:-${OTPU_CLK_HOOK:-}}"     # IMAGE=wg: a Tcl file before opt_design (bcu1525_wg_check.tcl; clk133.tcl: the 150 MHz synthesis at 133.33)
if [[ -n "${SYNTH_THREADS:-}" ]]; then export OTPU_SYNTH_THREADS="$SYNTH_THREADS"; fi
run() { local script="$1"; shift
  "$vivado_bin" -mode batch -nojournal -log "$out/$(basename "$script" .tcl).log" -source "$script" -tclargs "$@"; }
if [[ "${PREFLIGHT:-1}" != 0 ]]; then
  part="${T_PART:-xcvu9p-fsgd2104-2L-e}"
  if ! command -v "$vivado_bin" > /dev/null; then
    echo "run_vivado: no vivado on PATH (source <Vivado>/settings64.sh)" >&2; exit 2
  fi
  if ! (cd "$out" && "$vivado_bin" -mode batch -nojournal -nolog -source "$here/scripts/preflight.tcl" \
          -tclargs "$part") > "$out/preflight.log" 2>&1; then
    grep -E "PREFLIGHT|ERROR" "$out/preflight.log" >&2 || tail -20 "$out/preflight.log" >&2
    echo "run_vivado: preflight failed ($out/preflight.log); PREFLIGHT=0 skips it" >&2
    exit 2
  fi
  grep "PREFLIGHT ok" "$out/preflight.log" >&2
fi
if [[ "${STEP:-}" == impl && "${IMPL_ALONE:-0}" == 1 ]]; then
  # impl_1's runme.sh on its own: no Vivado holding the project (~2 GB) while it runs
  run "$here/vivado/build.tcl" "$out" "$jobs" impl-scripts "${IMPL_STRATEGY:-}"
  echo "impl_1: $out/otpu.runs/impl_1/runme.log"
  (. "$out/impl_env.sh" && cd "$out/otpu.runs/impl_1" && ./runme.sh > /dev/null 2>&1) || true   # post checks it
  if [[ -n "$OTPU_IMPL_TO" ]]; then
    if [[ ! -e "$out/otpu.runs/impl_1/.$OTPU_IMPL_TO.end.rst" ]]; then
      echo "run_vivado: impl_1 did not finish $OTPU_IMPL_TO ($out/otpu.runs/impl_1/runme.log)" >&2; exit 1
    fi
    echo "done: impl_1 to $OTPU_IMPL_TO (vivado/impl_resume.tcl $out next)"; exit 0
  fi
  run "$here/vivado/build.tcl" "$out" "$jobs" post
elif [[ "${STEP:-}" == impl ]]; then
  run "$here/vivado/build.tcl" "$out" "$jobs" impl "${IMPL_STRATEGY:-}"
elif [[ "$image" == mem ]]; then
  run "$here/vivado/create_project_mem.tcl" "$out"
  run "$here/vivado/build.tcl" "$out" "$jobs" "${STEP:-full}" "${IMPL_STRATEGY:-}"
elif [[ "$image" == wg ]]; then
  wg_args=(--core-mhz "$core_mhz" --target "$tname")
  if [[ -n "${WG_PLAN:-}" ]]; then wg_args+=(--plan "$WG_PLAN"); fi
  if [[ -n "${WG_SLR:-}" ]]; then wg_args+=(--slr "$WG_SLR"); fi
  if [[ -n "${RLY:-}" ]]; then wg_args+=(--rly "$RLY"); fi
  for kv in ${WG_SET:-}; do wg_args+=(--set "$kv"); done
  PYTHONPATH="$root" python3 "$here/scripts/make_wg.py" "$out" "${wg_args[@]}"
  run "$here/vivado/create_project_wg.tcl" "$out"
  if [[ "${STEP:-}" == bd ]]; then
    grep -q "^bd_wg: four DIMMs" "$out/create_project_wg.log" || { echo "run_vivado: STEP=bd: bd_wg.tcl did not finish ($out/create_project_wg.log)" >&2; exit 1; }
    echo "done: block design ($out/create_project_wg.log)"; exit 0
  fi
  run "$here/vivado/build.tcl" "$out" "$jobs" "${STEP:-full}" "${IMPL_STRATEGY:-}"
else
  run "$here/vivado/create_project.tcl" "$out" "$mcols" "$core_mhz" "$build_id" "$vpu_cl" "$lanes" \
    "$act_rows" "$dstep" "$s" "$platform" "$arch_hash" "$ddr_mts" "$tmem_impl" "$tmem_x"
  run "$here/vivado/build.tcl" "$out" "$jobs" "${STEP:-full}" "${IMPL_STRATEGY:-}"
fi
if [[ "${STEP:-}" == synth ]]; then echo "done: synthesis ($out/reports/synth_*.rpt; STEP=impl next)"; exit 0; fi
if [[ -n "$OTPU_IMPL_TO" ]]; then echo "done: impl_1 to $OTPU_IMPL_TO (vivado/impl_resume.tcl $out next)"; exit 0; fi
echo "done: $out/otpu.bit"
