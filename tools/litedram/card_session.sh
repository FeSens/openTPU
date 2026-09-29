#!/bin/bash
# The LiteDRAM test image on the card, then production back (docs/litedram.md, "Test image").
# On the card host, under the device lock:
#
#   otpu-lock --wait 3600 -- bash card_session.sh DIR PRODUCTION_BIT [ld_host.py args]
#
# DIR holds ld_test.bit, csr.csv, sdram_init.py and ld_host.py. Loads the test image over JTAG,
# rescans (every otpu-rescan attempt is kept in DIR/rescan.log: whether the link came back and
# after how many), runs ld_host.py DIR all, then loads PRODUCTION_BIT, rescans until the ID
# register reads OTPU and runs the selftest. Stops (production not reloaded) only if the test
# image's link does not come back: then the card host needs a warm reboot, and production is
# loaded after it. LOAD=0: the card already runs the test image (after a warm reboot): no load.
set -u
DIR=${1:?dir}; PROD=${2:?production bitstream}; shift 2
P=${PYTHON:-$HOME/otpu-venv/bin/python}
cd "$DIR" || exit 1
log() { echo "=== $* $(date +%T)"; }
rescan() { sudo -n /usr/local/sbin/otpu-rescan 2>&1; }
others() { sudo -n lsof -t /dev/xdma0_* 2>/dev/null | head -1; }
load() {
  for _ in $(seq 60); do [ -z "$(others)" ] && break; sleep 5; done
  [ -n "$(others)" ] && { echo "device open by pid $(others)"; return 1; }
  openFPGALoader -c digilent_hs2 --freq 10000000 "$1" 2>&1 | tail -2
}
# up to 10 rescans; succeeds when the device's BAR0 answers (want: an ID register value, or "")
bring_up() {
  local want=$1
  for i in $(seq 10); do
    rescan > rescan.out; { echo "--- $2 attempt $i $(date +%T)"; cat rescan.out; } >> rescan.log
    if [ -e /dev/xdma0_user ]; then
      if [ -z "$want" ] || grep -q "ID register 0x$want" rescan.out; then
        echo "link back after $i rescan(s)"; tail -3 rescan.out; return 0; fi
    fi
    sleep 10
  done
  echo "no link after 10 rescans"; tail -5 rescan.out; return 1
}

if [ "${LOAD:-1}" = 0 ]; then
  log "LOAD=0: the card already runs the test image (e.g. after a warm reboot)"
  [ -e /dev/xdma0_user ] || { echo "no /dev/xdma0_user"; exit 2; }
else
  log "load the LiteDRAM test image"
  load ld_test.bit || exit 1
  bring_up "" "test image" || exit 2
fi
log "ld_host.py $DIR all $*"
$P ld_host.py "$DIR" info && $P ld_host.py "$DIR" all "$@" 2>&1 | tee host.log
log "restore production: $PROD"
load "$PROD" || exit 3
bring_up 4f545055 production || exit 4
timeout 1800 $P -m opentpu.host.selftest 2>&1 | grep -E "\[(PASS|FAIL)\]|config|ALL|stopped" | tee selftest.log
log "done"
