#!/usr/bin/env bash
# Build the openTPU bitstream for the SQRL FK35 (xcvu35p-fsvh2104-2-e, 8 GB HBM2) in Vivado batch mode
# (docs/sqrl-fk35.md). The knobs, as boards/sqrl-fk33/run_vivado.sh's (this board's own build.tcl, preflight.tcl and
# tmem_uram.xdc, after the FK33's):
#   MCOLS=4 VPU_CL=2 LANES=8 ACT_ROWS=MCOLS DSTEP=1   the accelerator (two slices, four HBM channels)
#   CORE_MHZ=250 UCLK_MHZ=250 (FAST=1: Vivado's default strategies)   TMEM_URAM=1|0   BUILD_ID   OUT_DIR   JOBS   STEP=impl
#   IMPL_STRATEGY   PREFLIGHT=1|0   QMEM=0|1|2 (the memory adapters' queues: block RAM, LUT RAM, + UltraRAM B FIFOs)
#   MEMX=1|0 (1: the adapters in SLR0, the SLR crossing on the slices' ports; 0: on the channels' native ports)
#   RQS=<file.rqs> (STEP=impl: QoR suggestions of an earlier routed run, read before opt_design: build.tcl)
#   PRE_OPT=<file.tcl> (sourced before opt_design: suggestions done by hand, e.g. CLOCK_BUFFER_TYPE on a net)
#   QUAD=1|0 (the target's arch QUAD: a 4-bit MM PAIR four blocks an advance; in the CFG_HASH)
#   PCIE_GEN=4|3 (the XDMA link: Gen4 x8, 512-bit AXI at 250 MHz, the default; or Gen3 x8, 256-bit: bd.tcl)
# Vivado 2019.2 builds it (its HBM and XDMA IPs; the VU35P needs a license that covers it). The bitstream is loaded
# over JTAG (Vivado's hardware manager) and the card's PCIe device rescanned; otpu_primary.mcs / otpu_secondary.mcs
# are the flash image (SPI x8, the card's two MT25QU256).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
out="${OUT_DIR:-$root/build/vivado_sqrl-fk35}"
jobs="${JOBS:-8}"
if tenv="$(PYTHONPATH="$root" python3 -m opentpu.target env sqrl-fk35 2>/dev/null)"; then
  eval "$tenv"
else
  echo "run_vivado: python3 -m opentpu.target env sqrl-fk35 failed: built-in defaults, CFG_HASH 0" >&2
fi
mcols="${MCOLS:-${T_MCOLS:-4}}"
vpu_cl="${VPU_CL:-${T_VPU_CL:-2}}"
lanes="${LANES:-${T_LANES:-8}}"
act_rows="${ACT_ROWS:-$mcols}"
dstep="${DSTEP:-${T_DSTEP:-1}}"
seq_wc="${SEQ_WC:-${T_SEQ_WC:-0}}"
core_mhz="${CORE_MHZ:-${T_CORE_MHZ:-250}}"
platform="${T_PLATFORM:-4}"
quad="${QUAD:-${T_QUAD:-0}}"
arch_hash="$(PYTHONPATH="$root" python3 -m opentpu.target hash sqrl-fk35 MCOLS="$mcols" \
  LANES="$lanes" ACT_ROWS="$act_rows" DSTEP="$dstep" SEQ_WC="$seq_wc" QUAD="$quad" 2>/dev/null || echo 0)"
build_id="${BUILD_ID:-$(git -C "$root" rev-parse HEAD 2>/dev/null | cut -c1-8)}"
build_id="${build_id:-0}"
mkdir -p "$out"
export OTPU_IP_CACHE="${OTPU_IP_CACHE-$HOME/.cache/otpu-vivado-ip}"
export OTPU_FAST="${FAST:-0}"
export OTPU_SEQ_WC_GEN="$seq_wc"
export OTPU_QMEM="${QMEM:-0}"
export OTPU_MEMX="${MEMX:-1}"
export OTPU_RQS="${RQS:-}"
export OTPU_PRE_OPT="${PRE_OPT:-}"
export OTPU_QUAD="$quad"
export TMEM_URAM="${TMEM_URAM:-1}"
run() { local script="$1"; shift
  vivado -mode batch -nojournal -log "$out/$(basename "$script" .tcl).log" -source "$script" -tclargs "$@"; }
if [[ "${PREFLIGHT:-1}" != 0 ]]; then
  if ! command -v vivado > /dev/null; then
    echo "run_vivado: no vivado on PATH (source <Vivado 2019.2>/settings64.sh)" >&2; exit 2
  fi
  if ! (cd "$out" && vivado -mode batch -nojournal -nolog -source "$here/scripts/preflight.tcl" \
          -tclargs "${T_PART:-xcvu35p-fsvh2104-2-e}") > "$out/preflight.log" 2>&1; then
    grep -E "PREFLIGHT|ERROR" "$out/preflight.log" >&2 || tail -20 "$out/preflight.log" >&2
    echo "run_vivado: preflight failed ($out/preflight.log); PREFLIGHT=0 skips it" >&2
    exit 2
  fi
  grep "PREFLIGHT ok" "$out/preflight.log" >&2
fi
if [[ "${STEP:-}" == impl ]]; then
  run "$here/vivado/build.tcl" "$out" "$jobs" impl "${IMPL_STRATEGY:-}"
else
  run "$here/vivado/create_project.tcl" "$out" "$mcols" "$core_mhz" "$build_id" "$vpu_cl" "$lanes" \
    "$act_rows" "$dstep" "$platform" "$arch_hash"
  run "$here/vivado/build.tcl" "$out" "$jobs" full "${IMPL_STRATEGY:-}"
fi
echo "done: $out/otpu.bit"
