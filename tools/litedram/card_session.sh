#!/bin/bash
# The LiteDRAM test image on the card (docs/litedram.md, section 7). On the card host (opentpu)
# a JTAG load comes back on the bus only after a warm reboot (otpu-rescan's retrain did not
# bring it back after any of the three loads so far), so a session is four steps around
# two reboots, each under the device lock:
#
#   otpu-lock --wait 3600 -- bash card_session.sh load DIR/ld_test.bit DIR   # otpu-smi of
#   sudo -n systemctl reboot                                                 #   production first
#   otpu-lock --wait 3600 -- bash card_session.sh test DIR                   # the test image's runs
#   otpu-lock --wait 3600 -- bash card_session.sh load PRODUCTION_BIT DIR
#   sudo -n systemctl reboot
#   otpu-lock --wait 3600 -- bash card_session.sh after DIR                  # production: selftest,
#                                                                            #   otpu-smi
# DIR holds ld_test.bit, csr.csv, sdram_init.py and ld_host.py; every step's output goes to a
# log there. Never reboot while a Vivado job runs on the host.
#
# An image with the calibration CPU (ld_test.py --selfcal, docs/litedram.md section 9): the step
# `selfcal DIR` instead of `test` (LDHOST: the ld_host.py to run, e.g. a staged tree's, whose
# opentpu package it then imports; SOAK: seconds per channel, default 300).
set -u
STEP=${1:?load BIT DIR | test DIR | selfcal DIR | after DIR}; shift
P=${PYTHON:-$HOME/otpu-venv/bin/python}
V=$(dirname "$P")
log() { echo "=== $* $(date +%T)"; }
others() { sudo -n lsof -t /dev/xdma0_* 2>/dev/null | head -1; }
# production answers "OTPU" at BAR0 offset 0 (its ID register)
production() { [ -e /dev/xdma0_user ] && $P -c 'import mmap, os, sys
fd = os.open("/dev/xdma0_user", os.O_RDONLY | os.O_SYNC)
m = mmap.mmap(fd, 4096, mmap.MAP_SHARED, mmap.PROT_READ)
sys.exit(0 if int.from_bytes(m[0:4], "little") == 0x4F545055 else 1)'; }

case $STEP in
load)
  BIT=${1:?bitstream}; DIR=${2:?dir}
  cd "$DIR" || exit 1
  if production; then
    log "otpu-smi (production, before the load)"; "$V/otpu-smi" 2>&1 | tee -a smi.log
  fi
  for _ in $(seq 60); do [ -z "$(others)" ] && break; sleep 5; done
  [ -n "$(others)" ] && { echo "device open by pid $(others)"; exit 1; }
  log "load $BIT"
  openFPGALoader -c digilent_hs2 --freq 10000000 "$BIT" 2>&1 | tail -2
  log "loaded: a warm reboot brings it onto the bus"
  ;;
test)
  DIR=${1:?dir}
  cd "$DIR" || exit 1
  [ -e /dev/xdma0_user ] || { echo "no /dev/xdma0_user"; exit 2; }
  production && { echo "the card runs production, not the test image"; exit 2; }
  log "info"; $P ld_host.py . info 2>&1 | tee info.log
  log "channel 1: DQS scan, calibration, BIST"; $P ld_host.py . all --ch 1 --stride 1 2>&1 | tee all_ch1.log
  log "channel 1: read taps under traffic"; $P ld_host.py . rscan --ch 1 2>&1 | tee rscan_ch1.log
  log "temperature run"; $P ld_host.py . temp --ch both --stride 1 --minutes "${MINUTES:-35}" \
    --every "${EVERY:-300}" 2>&1 | tee temp.log
  log "BIST bandwidth, both channels"; $P ld_host.py . bist --ch both 2>&1 | tee bist.log
  log "done"
  ;;
selfcal)
  DIR=${1:?dir}
  H=${LDHOST:-ld_host.py}
  cd "$DIR" || exit 1
  [ -e /dev/xdma0_user ] || { echo "no /dev/xdma0_user"; exit 2; }
  production && { echo "the card runs production, not the test image"; exit 2; }
  log "info"; $P "$H" . info 2>&1 | tee info.log
  log "the CPU's calibration since configuration: result, BIST, soak"
  $P "$H" . selfcal --soak --seconds "${SOAK:-300}" 2>&1 | tee selfcal.log
  log "the host's calibration (the CPU held) against the CPU's"
  $P "$H" . selfcal --compare 2>&1 | tee selfcal_compare.log
  log "the CPU calibrates again (released)"; $P "$H" . selfcal --rerun 2>&1 | tee selfcal_rerun.log
  log "done"
  ;;
after)
  DIR=${1:?dir}
  cd "$DIR" || exit 1
  production || { echo "the card does not run production"; exit 2; }
  log "selftest"
  timeout 1800 $P -m opentpu.host.selftest 2>&1 | grep -E "\[(PASS|FAIL)\]|config|ALL|stopped" | tee selftest.log
  log "otpu-smi (production, after)"; "$V/otpu-smi" 2>&1 | tee -a smi.log
  log "done"
  ;;
*) echo "unknown step $STEP"; exit 1 ;;
esac
