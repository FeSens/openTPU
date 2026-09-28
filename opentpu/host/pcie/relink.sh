# shellcheck shell=bash disable=SC2154,SC2086   # pci and state come from the caller; port lists split
# Link recovery for setup_pcie.sh --rescan, sourced by it (and by tests/test_setup_pcie.py, with
# a fake sysfs and a fake setpci). The caller sets pci (the sysfs PCI bus directory) and state
# (the file that keeps the cards' upstream ports between runs) and provides cards, note and ok.
#
# A JTAG load or an IPROG reconfiguration takes the card off the link; it comes back with the new
# bitstream. Most root ports retrain by themselves and a plain rescan finds the card. On opentpu
# (the Haswell PEG root port 00:01.0, 2026-09-28, seen once) rescans found nothing for 5 minutes
# and a warm reboot brought the card back. So rescan_cards first does the plain remove + rescan,
# and only when no card is back does it work on the cards' upstream ports, rescanning after each
# step: Retrain Link (Link Control bit 5), then Link Disable (bit 4) set and cleared, then a
# secondary bus reset (Bridge Control bit 6). A port that retrains by itself is never touched.
# The ports are held in D0 meanwhile: with runtime PM on (power/control "auto", as on opentpu) a
# port suspends 100 ms after its last device is removed, and a suspended port trains no link.

relink_tries=8 relink_step=0.25   # after each step: up to 8 rescans, 0.25 s apart
relink_hold=0.1                   # Link Disable and the secondary bus reset held for 0.1 s
bdf_re='^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$'
held=""                           # "port=power/control" pairs to restore
saved_ap=""                       # drivers_autoprobe while a rescan has it off

port_of() {  # the upstream port (bridge) of a device; nothing for a device on a root bus
  local up
  up="$(cd -P "$pci/devices/$1/.." 2>/dev/null && basename "$PWD")" || return 0
  if [[ $up =~ $bdf_re && -e "$pci/devices/$up" ]]; then echo "$up"; fi
}

card_ports() {  # the upstream ports of the cards on the bus, else those recorded by the last rescan
  local b p out=""
  for b in $(cards); do
    p="$(port_of "$b")"
    if [[ -n $p && " $out " != *" $p "* ]]; then out+=" $p"; fi
  done
  if [[ -z $out && -f $state ]]; then
    # shellcheck disable=SC2013   # one port per word
    for p in $(cat "$state"); do
      if [[ $p =~ $bdf_re && -e "$pci/devices/$p" && " $out " != *" $p "* ]]; then out+=" $p"; fi
    done
  fi
  echo $out
}

hold_ports() {  # keep the ports in D0 (runtime PM off) until release_ports
  local p f v
  for p; do
    f="$pci/devices/$p/power/control"
    [[ -w $f ]] || continue
    v="$(cat "$f")"
    if [[ $v != on ]]; then held+=" $p=$v"; echo on > "$f"; fi
  done
}

release_ports() {  # power/control back to what hold_ports found
  local h
  for h in $held; do echo "${h#*=}" > "$pci/devices/${h%%=*}/power/control"; done
  held=""
}

relink_cleanup() {  # on any exit: automatic probing and the ports' power control as they were
  if [[ -n $saved_ap ]]; then echo "$saved_ap" > "$pci/drivers_autoprobe"; saved_ap=""; fi
  release_ports
}

rescan_bus() {  # rescan with the kernel's automatic probing off, so that 8250_pci cannot take a
  # serial-class bitstream before the driver override is set (bind_cards binds it)
  saved_ap="$(cat "$pci/drivers_autoprobe")"
  echo 0 > "$pci/drivers_autoprobe"
  echo 1 > "$pci/rescan"
  echo "$saved_ap" > "$pci/drivers_autoprobe"
  saved_ap=""
}

remove_dev() { echo 1 > "$pci/devices/$1/remove"; }

wait_card() {  # up to relink_tries rescans, relink_step s apart, until a card answers
  local i
  for ((i = 0; i < relink_tries; i++)); do
    sleep "$relink_step"
    rescan_bus
    if [[ -n "$(cards)" ]]; then return 0; fi
  done
  return 1
}

link_state() {  # the port's Link Status, for the log
  local sta cap
  sta="$(setpci -s "$1" CAP_EXP+12.w 2>/dev/null)" || sta=""
  cap="$(setpci -s "$1" CAP_EXP+0c.l 2>/dev/null)" || cap=""
  if [[ -z $sta || -z $cap ]]; then echo "no PCIe capability"; return; fi
  printf 'LnkSta 0x%s: width x%d, training %d, ' "$sta" $(((0x$sta >> 4) & 0x3f)) $(((0x$sta >> 11) & 1))
  if (((0x$cap >> 20) & 1)); then echo "data link $( (((0x$sta >> 13) & 1)) && echo up || echo down)"
  else echo "data link state not reported by the port"; fi
}

relink() {  # no card after a plain rescan: retrain the ports' links, then disable / re-enable
  # them, then reset their secondary buses, rescanning after each step
  local p step what
  command -v setpci >/dev/null || { note "setpci missing (pciutils): cannot retrain the link"; return 1; }
  [[ $# -gt 0 ]] || { note "no upstream port known (the card was not on the bus at this or the last rescan)"; return 1; }
  for step in retrain disable reset; do
    case $step in
      retrain) what="retrain the link (Link Control: Retrain Link)" ;;
      disable) what="disable and re-enable the link (Link Control: Link Disable)" ;;
      reset)   what="secondary bus reset (Bridge Control)" ;;
    esac
    for p; do
      note "$p: $(link_state "$p")"
      note "$p: $what"
      case $step in
        retrain) setpci -s "$p" CAP_EXP+10.w=0020:0020 ;;
        disable) setpci -s "$p" CAP_EXP+10.w=0010:0010; sleep "$relink_hold"
                 setpci -s "$p" CAP_EXP+10.w=0000:0010 ;;
        reset)   setpci -s "$p" BRIDGE_CONTROL=0040:0040; sleep "$relink_hold"
                 setpci -s "$p" BRIDGE_CONTROL=0000:0040 ;;
      esac
    done
    if wait_card; then ok "the card is back after: $what"; return 0; fi
  done
  for p; do note "$p: $(link_state "$p")"; done
  return 1
}

rescan_cards() {  # remove the cards and rescan; when none is back, relink. 0: a card is on the bus
  local ports b
  ports="$(card_ports)"
  if [[ -n $ports ]]; then echo "$ports" > "$state"; fi
  hold_ports $ports
  for b in $(cards); do note "removing $b"; remove_dev "$b"; done
  rescan_bus
  if [[ -z "$(cards)" ]]; then note "no card after the rescan"; relink $ports || true; fi
  release_ports
  [[ -n "$(cards)" ]]
}
