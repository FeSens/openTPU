#!/usr/bin/env bash
# Load the bitstream into the YPCB-00338 over JTAG.
#
#   ./program.sh [bitfile]                 # openFPGALoader (macOS/Linux), Xilinx Platform Cable
#   ./program.sh --vivado [bitfile]        # Vivado hardware manager (hw_server on this machine)
#   ./program.sh --flash [mcsfile]         # write the BPI flash (permanent; loads at power-up)
#
# The card's JTAG chain has an Inspur CPLD (IDCODE 0x10931093) before the FPGA, hence
# --misc-device / --index-chain. The Platform Cable USB II needs its firmware (xusb_xp2.hex)
# loaded on every plug-in: set XUSB_FIRMWARE (default: the bonetto-soc/inspur-adventures copy).
# openFPGALoader must be a build with the xilinxPlatformCableUsb driver (see the ypcb-00338
# skill / inspur-adventures flake). The macOS cable wedges now and then: retries are built in;
# after 'Unable to read constant' on every attempt, replug the USB cable.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
mode=ofl
case "${1:-}" in
  --vivado) mode=vivado; shift ;;
  --flash) mode=flash; shift ;;
esac
if [[ $mode == flash ]]; then bit="${1:-$root/build/vivado/otpu.mcs}"
else bit="${1:-$root/build/vivado/otpu.bit}"; fi
fw="${XUSB_FIRMWARE:-$HOME/bonetto/inspur-adventures/firmware/xusb_xp2.hex}"
cable="${CABLE:-xilinxPlatformCableUsb}"
# shellcheck disable=SC2054  # the commas are part of the --misc-device argument
ofl=(openFPGALoader --cable "$cable" --probe-firmware "$fw"
     --misc-device 0x10931093,8,inspur_cpld --index-chain 0)

retry() {
  for i in 1 2 3 4 5; do
    echo ">> attempt $i: $*"
    "$@" && return 0
    sleep 2
  done
  echo "failed after 5 attempts (replug the JTAG cable if every attempt said 'Unable to read constant')"
  return 1
}

case $mode in
  ofl)
    retry "${ofl[@]}" --freq 6000000 "$bit" ;;
  flash)
    # BPI flash through the FPGA (openFPGALoader loads its bpiOverJtag bridge first)
    retry "${ofl[@]}" --freq 6000000 --fpga-part xc7k480tffg1156 -f "$bit" ;;
  vivado)
    tcl="$(mktemp "${TMPDIR:-/tmp}/otpu_prog.XXXXXX")"   # BSD mktemp: the X's must end the name
    trap 'rm -f "$tcl"' EXIT
    cat > "$tcl" <<EOF
open_hw_manager
connect_hw_server -allow_non_jtag
open_hw_target
set dev [lindex [get_hw_devices xc7k480t*] 0]
current_hw_device \$dev
set_property PROGRAM.FILE {$bit} \$dev
program_hw_devices \$dev
close_hw_manager
EOF
    vivado -mode batch -nojournal -nolog -source "$tcl" ;;
esac
echo "programmed. PCIe: rescan on the host now (docs/board.md, 'After programming')."
