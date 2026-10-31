#!/usr/bin/env bash
# Build the openTPU bitstream for the SQRL FK33 (xcvu33p-fsvh2104-2-e, 8 GB HBM2) in Vivado batch mode
# (docs/sqrl-fk33.md). The knobs, as boards/bcu1525/run_vivado.sh's (this board's own build.tcl, preflight.tcl and
# tmem_uram.xdc, after the BCU1525's):
#   MCOLS=4 VPU_CL=2 LANES=8 ACT_ROWS=MCOLS DSTEP=1   the accelerator (one slice, two HBM channels)
#   CORE_MHZ=250 UCLK_MHZ=250 (FAST=1: Vivado's default strategies)   TMEM_URAM=1|0   BUILD_ID   OUT_DIR   JOBS   STEP=impl
#   IMPL_STRATEGY   PREFLIGHT=1|0
# Vivado 2019.2 builds it (its HBM and XDMA IPs; the VU33P needs a license that covers it). The bitstream is loaded
# over JTAG (Vivado's hardware manager) and the card's PCIe device rescanned; otpu_primary.mcs / otpu_secondary.mcs
# are the flash image (SPI x8, the card's two MT25QU256).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
out="${OUT_DIR:-$root/build/vivado_sqrl-fk33}"
jobs="${JOBS:-8}"
if tenv="$(PYTHONPATH="$root" python3 -m opentpu.target env sqrl-fk33 2>/dev/null)"; then
  eval "$tenv"
else
  echo "run_vivado: python3 -m opentpu.target env sqrl-fk33 failed: built-in defaults, CFG_HASH 0" >&2
fi
mcols="${MCOLS:-${T_MCOLS:-4}}"
vpu_cl="${VPU_CL:-${T_VPU_CL:-2}}"
lanes="${LANES:-${T_LANES:-8}}"
act_rows="${ACT_ROWS:-$mcols}"
dstep="${DSTEP:-${T_DSTEP:-1}}"
seq_wc="${SEQ_WC:-${T_SEQ_WC:-0}}"
core_mhz="${CORE_MHZ:-${T_CORE_MHZ:-250}}"
platform="${T_PLATFORM:-5}"
arch_hash="$(PYTHONPATH="$root" python3 -m opentpu.target hash sqrl-fk33 MCOLS="$mcols" \
  LANES="$lanes" ACT_ROWS="$act_rows" DSTEP="$dstep" SEQ_WC="$seq_wc" 2>/dev/null || echo 0)"
build_id="${BUILD_ID:-$(git -C "$root" rev-parse HEAD 2>/dev/null | cut -c1-8)}"
build_id="${build_id:-0}"
mkdir -p "$out"
export OTPU_IP_CACHE="${OTPU_IP_CACHE-$HOME/.cache/otpu-vivado-ip}"
export OTPU_FAST="${FAST:-0}"
export OTPU_SEQ_WC_GEN="$seq_wc"
export TMEM_URAM="${TMEM_URAM:-1}"
run() { local script="$1"; shift
  vivado -mode batch -nojournal -log "$out/$(basename "$script" .tcl).log" -source "$script" -tclargs "$@"; }
if [[ "${PREFLIGHT:-1}" != 0 ]]; then
  if ! command -v vivado > /dev/null; then
    echo "run_vivado: no vivado on PATH (source <Vivado 2019.2>/settings64.sh)" >&2; exit 2
  fi
  if ! (cd "$out" && vivado -mode batch -nojournal -nolog -source "$here/scripts/preflight.tcl" \
          -tclargs "${T_PART:-xcvu33p-fsvh2104-2-e}") > "$out/preflight.log" 2>&1; then
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
