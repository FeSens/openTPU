#!/usr/bin/env bash
# One card's development x2 image, launched on the build host itself; no SSH, programming
# or flash action. Build synthesis and implementation in separate Vivado processes. Both cards
# use the resulting bitstream; CARD_ID is a runtime CSR, never a build knob.
#   bash boards/bcu1525/x2/build_x2.sh OUT
#   bash boards/bcu1525/x2/build_x2.sh --prepare OUT   # generated inputs only, no FPGA tools
#   RESUME=1 bash boards/bcu1525/x2/build_x2.sh OUT    # reuse a finished synthesis
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../.." && pwd)"
prepare=0
if [[ ${1:-} == --prepare ]]; then prepare=1; shift; fi
[[ $# == 1 ]] || { echo "usage: $0 [--prepare] OUT" >&2; exit 2; }
out=$1
mkdir -p "$out"
out="$(cd "$out" && pwd)"
# Fixed first-build contract: environment tuning must not silently turn it into a 150 MHz or
# altered-architecture build. Later frequency qualification should use a separately named OUT.
[[ ${CORE_MHZ:-100} == 100 && ${FAST:-1} == 1 && -z ${WG_SET:-} ]] || {
  echo "x2 dev requires FAST=1 CORE_MHZ=100 and no WG_SET" >&2; exit 2;
}
export IMAGE=wg WG_TARGET=bcu1525-x2:st16 WG_PLAN=c FAST=1 CORE_MHZ=100
export OUT_DIR="$out" JOBS="${JOBS:-2}" SYNTH_THREADS="${SYNTH_THREADS:-2}"
export TMPDIR="$out/tmp"
export VIVADO="${VIVADO:-$(command -v vivado || echo "$HOME/.local/bin/vivado")}"
build_sha="$(cat "$root/COMMIT" 2>/dev/null || git -C "$root" rev-parse HEAD 2>/dev/null || true)"
build_sha="${build_sha:-00000000}"     # offline --prepare on a source tarball without COMMIT
mkdir -p "$TMPDIR"
[[ ! -e "$out/otpu.xpr" || ${RESUME:-0} == 1 ]] || {
  echo "x2: $out already has a project; choose a new OUT or RESUME=1" >&2; exit 2;
}
# Readiness is concrete even on a host without Vivado. --prepare stops before the preflight.
if [[ ${RESUME:-0} == 1 ]]; then
  # Never relabel an old synthesis by overwriting its generated inputs with a new commit.
  test -f "$out/otpu.xpr"
  for line in 'set CORE_KHZ 100000' 'set WG_S 2' 'set WG_NCARDS 2' \
              'set ARCH_HASH e331f79c' "set BUILD_ID ${build_sha:0:8}"; do
    grep -Fxq "$line" "$out/wg_params.tcl" || { echo "x2 resume mismatch: $line" >&2; exit 2; }
  done
else
  PYTHONPATH="$root" python3 "$root/boards/bcu1525/scripts/make_wg.py" "$out" \
    --target "$WG_TARGET" --plan c --core-mhz 100
fi
grep -E '^set (CORE_KHZ|WG_S |WG_NCARDS|ARCH_HASH|BUILD_ID)' "$out/wg_params.tcl"
grep -q '^set ARCH_HASH e331f79c$' "$out/wg_params.tcl"
echo "x2 dev: FAST=1 CORE_MHZ=100, two local slices, NCARDS=2, PLATFORM=7, CFG_HASH=E331F79C"
echo "outputs: $out/otpu.bit, otpu.mcs; reports/{SUMMARY.txt,x2_summary.txt,timing_summary.rpt,util_slr.rpt,cdc.rpt,x2_bus_skew.rpt}"
echo "gate: WNS >= 0 AND WHS >= 0; expected QSFP0 X1Y48-51/COMMON X1Y12, QSFP1 X1Y44-47/COMMON X1Y11, both refs on U9"
if [[ $prepare == 1 ]]; then echo "prepared only: no Vivado invoked"; exit 0; fi
command -v "$VIVADO" >/dev/null || { echo "x2: no Vivado at $VIVADO" >&2; exit 2; }
[[ $build_sha != 00000000 ]] || { echo "x2: build needs git identity or a COMMIT file in the source tree" >&2; exit 2; }
# Do not inherit a partial-run knob or a different topology from the caller's shell.
unset IMPL_TO IMPL_STRATEGY WG_STOP OTPU_IMPL_TO OTPU_WG_STOP WG_SLR RLY
export WG_FP=hard PCIE_LANES=16
st="$out/status.txt"
trap 'rc=$?; echo "FAILED rc=$rc $(date -u)" | tee -a "$st"; exit "$rc"' ERR
echo "START x2 $(date -u) BUILD_ID=${build_sha:0:8}" | tee -a "$st"
cd "$out"
if [[ ${RESUME:-0} != 1 ]]; then
  STEP=synth nice -n "${NICE:-15}" bash "$root/boards/bcu1525/run_vivado.sh" 2>&1 | tee "$out/synth_stdout.log"
fi
STEP=impl IMPL_ALONE=1 nice -n "${NICE:-15}" bash "$root/boards/bcu1525/run_vivado.sh" 2>&1 | tee "$out/impl_stdout.log"
test -s "$out/otpu.bit"
cat "$out/reports/x2_summary.txt"
echo "DONE x2 $(date -u) bitstream=$out/otpu.bit (functional dev image; card qualification pending)" | tee -a "$st"
