#!/usr/bin/env bash
# Build the openTPU bitstream for the YPCB-00338 in Vivado batch mode, natively or in Docker.
#
#   ./run_vivado.sh [800|1066|1300|1333|1600] # native: `vivado` on PATH (x86-64 Linux / Windows WSL)
#   MCOLS=4 ./run_vivado.sh               # 4 MXU columns (faster prefill / batched decode)
#   VPU_CL=4 ./run_vivado.sh              # 4 VPU lanes with exp2/recip/rsqrt (faster softmax)
#   LANES=16 ./run_vivado.sh              # 16 VPU lanes / TMEM banks
#   CORE_MHZ=80 ./run_vivado.sh           # slower core clock when 100 MHz does not close
#   FAST=1 ./run_vivado.sh                # development build: default strategies, no retiming or post-route
#                                         # phys_opt (with the default CORE_MHZ=100); about half the time
#   VIVADO_DOCKER=image ./run_vivado.sh   # Docker (e.g. Apple Silicon with Rosetta), see docs/board.md
#   VIVADO_AS_USER=1                      # Docker on Linux: run as the calling user (see run())
#   STEP=impl ./run_vivado.sh             # rerun implementation only (keeps project and synthesis)
#   IMPL_STRATEGY=Performance_Explore     # a stronger implementation strategy (with bit or impl)
#   MIG_ADDR_MAP=BANK_ROW_COLUMN          # the MIG address map (default ROW_BANK_COLUMN; gen_mig_prj.py)
#   MIG_BANK_MACHINES=8 MIG_ORDERING=Strict # the MIG controllers' bank machines (4) and ordering (Normal)
#   The xc7k480t needs a paid or 30-day evaluation license, node-locked to a MAC address. In Docker
#   set VIVADO_MAC (the MAC the license was issued for) and XILINXD_LICENSE_FILE (path to the .lic).
#
# Output: build/vivado/otpu.bit, build/vivado/otpu.mcs, build/vivado/reports/.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
speed="${1:-800}"
out="${OUT_DIR:-$root/build/vivado}"
jobs="${JOBS:-8}"
mcols="${MCOLS:-2}"            # MXU columns (the host reads them from the VERSION register)
vpu_cl="${VPU_CL:-2}"          # VPU lanes with the composite functions (timing only)
lanes="${LANES:-8}"            # VPU lanes / TMEM banks (likewise in VERSION)
act_rows="${ACT_ROWS:-$mcols}" # ACT RAM rows (> MCOLS: MM replay; the ACT_ROWS register)
dstep="${DSTEP:-1}"            # 0: no DSTEP datapath in the DMA (CAPS bit6 = 0)
axi_bl="${AXI_BL:-32}"         # port B read burst, beats (32 default, the production image; up to 64)
core_mhz="${CORE_MHZ:-100}"   # accelerator clock; lower it (80, 75) if timing does not close
# BUILD_ID register: the git commit's first 8 hex digits, taken here (Vivado may run in Docker)
build_id="${BUILD_ID:-$(git -C "$root" rev-parse HEAD 2>/dev/null | cut -c1-8)}"
build_id="${build_id:-0}"

python3 "$here/scripts/gen_mig_prj.py" --speed "$speed" ${MIG_ADDR_MAP:+--addr-map "$MIG_ADDR_MAP"} \
  ${MIG_BANK_MACHINES:+--bank-machines "$MIG_BANK_MACHINES"} ${MIG_ORDERING:+--ordering "$MIG_ORDERING"}

run() {  # run a Vivado Tcl script with arguments
  local script="$1"; shift
  if [[ -n "${VIVADO_DOCKER:-}" ]]; then
    # VIVADO_AS_USER=1: run as the calling user (a Linux host whose files Docker does not remap:
    # the .lic is often mode 600), with the image's entrypoint bypassed
    local as_user=() shell=(bash)
    if [[ -n "${VIVADO_AS_USER:-}" ]]; then
      as_user=(--user "$(id -u):$(id -g)" -e "HOME=$out/.home" --entrypoint bash); shell=()
      mkdir -p "$out/.home"
    fi
    docker run --rm \
      -v "$root:$root" -w "$root" \
      ${VIVADO_MAC:+--mac-address "$VIVADO_MAC"} \
      ${XILINXD_LICENSE_FILE:+-v "$XILINXD_LICENSE_FILE:$XILINXD_LICENSE_FILE:ro" -e "XILINXD_LICENSE_FILE=$XILINXD_LICENSE_FILE"} \
      ${VIVADO_MOUNT:+-v "$VIVADO_MOUNT"} \
      ${as_user[@]+"${as_user[@]}"} \
      "$VIVADO_DOCKER" \
      ${shell[@]+"${shell[@]}"} -lc "source ${VIVADO_SETTINGS:-/tools/Xilinx/Vivado/2026.1/settings64.sh} && \
                vivado -mode batch -nojournal -log $out/$(basename "$script" .tcl).log \
                -source $script -tclargs $*"
  else
    vivado -mode batch -nojournal -log "$out/$(basename "$script" .tcl).log" \
      -source "$script" -tclargs "$@"
  fi
}

mkdir -p "$out"
# the IP synthesis cache shared by this host's builds (create_project.tcl); OTPU_IP_CACHE= turns it off
export OTPU_IP_CACHE="${OTPU_IP_CACHE-$HOME/.cache/otpu-vivado-ip}"
export OTPU_FAST="${FAST:-0}"
if [[ "${STEP:-}" == impl ]]; then
  run "$here/vivado/build.tcl" "$out" "$jobs" impl "${IMPL_STRATEGY:-}"
else
  run "$here/vivado/create_project.tcl" "$speed" "$out" "$mcols" "$core_mhz" "$build_id" "$vpu_cl" "$lanes" "$act_rows" "$dstep" "$axi_bl"
  run "$here/vivado/build.tcl" "$out" "$jobs" full "${IMPL_STRATEGY:-}"
fi
echo "done: $out/otpu.bit"
