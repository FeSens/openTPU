#!/usr/bin/env bash
# Build the openTPU bitstream for the YPCB-00338 in Vivado batch mode, natively or in Docker.
#
#   ./run_vivado.sh [800|1066]            # native: `vivado` on PATH (x86-64 Linux / Windows WSL)
#   MCOLS=4 ./run_vivado.sh               # 4 MXU columns (faster prefill / batched decode)
#   CORE_MHZ=80 ./run_vivado.sh           # slower core clock when 100 MHz does not close
#   VIVADO_DOCKER=image ./run_vivado.sh   # Docker (e.g. Apple Silicon with Rosetta), see docs/board.md
#
# Output: build/vivado/otpu.bit, build/vivado/otpu.mcs, build/vivado/reports/.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
speed="${1:-800}"
out="${OUT_DIR:-$root/build/vivado}"
jobs="${JOBS:-8}"
mcols="${MCOLS:-2}"            # MXU columns; the host needs OTPU_MCOLS set to the same value
core_mhz="${CORE_MHZ:-100}"   # accelerator clock; lower it (80, 75) if timing does not close
# BUILD_ID register: the git commit's first 8 hex digits, taken here (Vivado may run in Docker)
build_id="${BUILD_ID:-$(git -C "$root" rev-parse HEAD 2>/dev/null | cut -c1-8)}"
build_id="${build_id:-0}"

python3 "$here/scripts/gen_mig_prj.py" --speed "$speed"

run() {  # run a Vivado Tcl script with arguments
  local script="$1"; shift
  if [[ -n "${VIVADO_DOCKER:-}" ]]; then
    docker run --rm --platform linux/amd64 \
      -v "$root:$root" -w "$root" \
      ${VIVADO_MOUNT:+-v "$VIVADO_MOUNT"} \
      "$VIVADO_DOCKER" \
      bash -lc "source ${VIVADO_SETTINGS:-/tools/Xilinx/Vivado/2026.1/settings64.sh} && \
                vivado -mode batch -nojournal -log $out/$(basename "$script" .tcl).log \
                -source $script -tclargs $*"
  else
    vivado -mode batch -nojournal -log "$out/$(basename "$script" .tcl).log" \
      -source "$script" -tclargs "$@"
  fi
}

mkdir -p "$out"
run "$here/vivado/create_project.tcl" "$speed" "$out" "$mcols" "$core_mhz" "$build_id"
run "$here/vivado/build.tcl" "$out" "$jobs"
echo "done: $out/otpu.bit"
