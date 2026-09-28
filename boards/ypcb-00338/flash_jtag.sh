#!/usr/bin/env bash
# The card's BPI flash over JTAG with the Vivado hardware manager (docs/flash.md, "Over JTAG").
# Run it on the PC with the JTAG cable (opentpu: native Vivado 2026.1), with no program using
# the card:
#
#   ./flash_jtag.sh detect              the JTAG chain and the FPGA's configuration registers
#                                       (BOOTSTS: where the running image came from); reads only
#   ./flash_jtag.sh backup OUT.bin      read the whole 64 MB flash into OUT.bin (flash order,
#                                       python -m opentpu.host.bpi_image info OUT.bin); the flash
#                                       is only read
#   OTPU_FLASH_WRITE=yes ./flash_jtag.sh program FILE.mcs
#                                       erase, program and verify the range FILE.mcs covers
#                                       (build.tcl writes otpu.mcs next to otpu.bit, image at 0)
#   ./flash_jtag.sh verify FILE.mcs     compare the flash with FILE.mcs
#   ./flash_jtag.sh boot                JPROGRAM: the FPGA configures from the flash now
#
# backup, program and verify first load Vivado's flash programming core into the FPGA, which
# replaces the running design (the card leaves the PCIe bus). Afterwards load a bitstream
# (program.sh) or boot from the flash (boot), then `sudo otpu-setup --rescan`, as after any
# JTAG load. Environment: FLASH_PART (Vivado's cfgmem part; default mt28gu512aax1e-bpi-x16, the
# Micron MT28GU512AAA that openFPGALoader's ypcb003381p1 BPI support is written for: 512 Mb,
# x16, 1.8 V, 256 KB blocks; check it against the chip's marking), VIVADO (default vivado).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
mode="${1:-}"; file="${2:-}"
part="${FLASH_PART:-mt28gu512aax1e-bpi-x16}"
vivado="${VIVADO:-vivado}"
py="${PYTHON:-python3}"
die() { echo "flash_jtag: $*" >&2; exit 1; }

case $mode in
  detect|boot) ;;
  backup) [[ -n $file ]] || die "backup OUT.bin"; [[ ! -e $file ]] || die "$file exists" ;;
  program|verify)
    [[ -f $file && $file == *.mcs ]] || die "$mode needs the .mcs of a build (build.tcl writes otpu.mcs)"
    file="$(cd "$(dirname "$file")" && pwd)/$(basename "$file")"
    info="$(PYTHONPATH="$root" "$py" -m opentpu.host.bpi_image info "$file")" || die "cannot read $file"
    echo "$info"
    # a bitstream for this FPGA (xc7k480t IDCODE) at address 0, where the FPGA looks at power-up
    grep -q "offset 0x0000000 .*IDCODE 03751093" <<<"$info" || die "$file has no xc7k480t image at 0"
    if [[ $mode == program && ${OTPU_FLASH_WRITE:-} != yes ]]; then
      die "program writes the flash (permanent): set OTPU_FLASH_WRITE=yes"
    fi ;;
  *) sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 2 ;;
esac
if [[ $mode == backup ]]; then
  out="$(cd "$(dirname "$file")" && pwd)/$(basename "$file")"
fi

# the card must be idle: nothing holds its device nodes, no other JTAG tool has the cable
if [[ $mode != detect ]]; then
  holder="$(sudo -n lsof -t /dev/xdma0_* 2>/dev/null | head -1 || true)"
  [[ -z $holder ]] || die "process $holder has the card open: stop it first"
fi
! pgrep -x openFPGALoader >/dev/null || die "openFPGALoader is running (the cable is taken)"

tcl="$(mktemp "${TMPDIR:-/tmp}/otpu_flash.XXXXXX")"
trap 'rm -f "$tcl"' EXIT
cat > "$tcl" <<EOF
open_hw_manager
connect_hw_server -allow_non_jtag
open_hw_target
puts "JTAG chain: [get_hw_devices]"
set dev [lindex [get_hw_devices xc7k480t*] 0]
if {\$dev eq ""} { error "no xc7k480t on the JTAG chain" }
current_hw_device \$dev
refresh_hw_device -update_hw_probes false \$dev
EOF
# shellcheck disable=SC2016   # Tcl variables
if [[ $mode == detect ]]; then
  echo 'report_property $dev REGISTER.*' >> "$tcl"
elif [[ $mode == boot ]]; then
  echo 'boot_hw_device $dev' >> "$tcl"
else
  cat >> "$tcl" <<EOF
set part [lindex [get_cfgmem_parts {$part}] 0]
if {\$part eq ""} { error "Vivado has no cfgmem part $part" }
create_hw_cfgmem -hw_device \$dev \$part
set cfg [get_property PROGRAM.HW_CFGMEM \$dev]
set_property PROGRAM.BPI_RS_PINS {none} \$cfg
set_property PROGRAM.UNUSED_PIN_TERMINATION {pull-none} \$cfg
# the flash programming core: replaces the running design
create_hw_bitstream -hw_device \$dev [get_property PROGRAM.HW_CFGMEM_BITFILE \$dev]
program_hw_devices \$dev
refresh_hw_device \$dev
EOF
  case $mode in
    backup)
      echo "readback_hw_cfgmem -all -format bin -file {$out} \$cfg" >> "$tcl" ;;
    program|verify)
      e=0; [[ $mode == program ]] && e=1
      cat >> "$tcl" <<EOF
set_property PROGRAM.FILES [list {$file}] \$cfg
set_property PROGRAM.ADDRESS_RANGE {use_file} \$cfg
set_property PROGRAM.BLANK_CHECK 0 \$cfg
set_property PROGRAM.ERASE $e \$cfg
set_property PROGRAM.CFG_PROGRAM $e \$cfg
set_property PROGRAM.VERIFY 1 \$cfg
set_property PROGRAM.CHECKSUM 0 \$cfg
program_hw_cfgmem \$cfg
EOF
      ;;
  esac
fi
echo "close_hw_manager" >> "$tcl"

t0=$(date +%s)
"$vivado" -mode batch -nojournal -nolog -notrace -source "$tcl"
echo "flash_jtag $mode: $(( $(date +%s) - t0 )) s"
case $mode in
  backup)
    sha256sum "$out" 2>/dev/null || shasum -a 256 "$out"
    PYTHONPATH="$root" "$py" -m opentpu.host.bpi_image info "$out" || true ;;
esac
case $mode in
  backup|program|verify)
    echo "the FPGA runs Vivado's flash programming core now: load a bitstream (program.sh) or"
    echo "boot from the flash (./flash_jtag.sh boot), then: sudo otpu-setup --rescan" ;;
  boot)
    echo "configuring from the flash: sudo otpu-setup --rescan (or sudo -n otpu-rescan on opentpu)" ;;
esac
