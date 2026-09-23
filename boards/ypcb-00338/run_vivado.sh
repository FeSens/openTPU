#!/usr/bin/env bash
# Build the openTPU bitstream for the YPCB-00338 in Vivado batch mode, natively or in Docker.
#
#   ./run_vivado.sh [800|1066]            # native: `vivado` on PATH (x86-64 Linux / Windows WSL)
#   VIVADO_DOCKER=image ./run_vivado.sh   # Docker (e.g. Apple Silicon with Rosetta), see docs/board.md
#
# Output: build/vivado/otpu.bit, build/vivado/otpu.mcs, build/vivado/reports/.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
speed="${1:-800}"
out="${OUT_DIR:-$root/build/vivado}"
jobs="${JOBS:-8}"

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
run "$here/vivado/create_project.tcl" "$speed" "$out"
run "$here/vivado/build.tcl" "$out" "$jobs"
echo "done: $out/otpu.bit"
