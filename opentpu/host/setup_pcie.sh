#!/usr/bin/env bash
# Host PC setup for the openTPU card (docs/host.md section 2). `otpu-setup` runs this with sudo.
#
#   setup_pcie.sh              install or update the XDMA driver, the udev rules and the driver
#                              options; then load the driver and read the card's ID register
#   setup_pcie.sh --check      report what is installed and working, change nothing
#                              (exit 1 when something is missing or wrong)
#   setup_pcie.sh --rescan     after a JTAG load: remove the card from the bus, rescan, bind the
#                              driver, wait for /dev/xdma0_user, read the ID register
#   setup_pcie.sh --uninstall  remove everything the install put on the system
#   --poll | --irq             the driver's completion mode (poll_mode=1 | 0), kept in
#                              /etc/modprobe.d/otpu-xdma.conf; without either, an existing
#                              setting stays and a new install takes DEFAULT_POLL
#   --no-dkms                  install the module for the running kernel only, even with DKMS
#   --src DIR                  take the driver source from a dma_ip_drivers git clone that has
#                              the pinned commit, instead of fetching it from GitHub
#
# The driver is Xilinx's dma_ip_drivers XDMA at a pinned commit plus pcie/xdma-otpu.patch (a
# Makefile fix for kernels >= 6.13 and an "otpu" module tag this script checks). It goes to
# /lib/modules/<kernel>/updates, which takes precedence over the kernel's own module of the same
# name (drivers/dma/xilinx/xdma, AMD's platform driver for Alveo cards, which does not drive
# this card). With DKMS it is rebuilt at every kernel upgrade; without it, re-run this script
# after a kernel upgrade (--check says when the module is missing for the running kernel).
#
# Linux only. Safe to re-run: every step checks first and changes only what differs. The driver
# is never unloaded under a running program: a pending reload waits for --rescan or a reboot.
set -euo pipefail

XDMA_REPO=https://github.com/Xilinx/dma_ip_drivers
XDMA_COMMIT=b8466090b4e812e191da9e9305ffb11cb7ace768   # 2026-08-20, driver version 2025.2.0
VER=b846609.1              # <commit>.<patch level>: the module's "otpu" tag (pcie/xdma-otpu.patch)
PKG=otpu-xdma
DEFAULT_POLL=0             # interrupts; docs/host.md section 2 has poll vs interrupts, measured
OTPU_ID=4f545055           # "OTPU", the ID register at BAR0 offset 0

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
files="$here/pcie"
kver="$(uname -r)"
moddir="/lib/modules/$kver"
src_dir="/usr/src/$PKG-$VER"
cache="/var/cache/otpu/dma_ip_drivers.git"
rules=(59-otpu-xdma.rules 60-otpu.rules)
modconf=/etc/modprobe.d/otpu-xdma.conf

mode=install poll="" dkms_ok=1 src=""
args=("$@")
while (($#)); do
  case "$1" in
    --check) mode=check ;;
    --rescan) mode=rescan ;;
    --uninstall) mode=uninstall ;;
    --poll) poll=1 ;;
    --irq) poll=0 ;;
    --no-dkms) dkms_ok=0 ;;
    --src) src="$(cd "${2:?--src needs a directory}" && pwd)"; shift ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option $1 (--help)" >&2; exit 2 ;;
  esac
  shift
done

say()  { printf '\033[1m== %s\033[0m\n' "$*"; }
ok()   { printf '   ok    %s\n' "$*"; }
note() { printf '   note  %s\n' "$*"; }
nfail=0
bad()  { printf '\033[31m   FAIL  %s\033[0m\n' "$*"; nfail=$((nfail + 1)); }
die()  { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

[[ "$(uname -s)" == Linux ]] || die "run this on the Linux PC that holds the card"
if [[ $mode != check && $EUID != 0 ]]; then
  exec sudo -- bash "$0" "${args[@]}"
fi
[[ -d "$files" ]] || die "$files missing: run the script from the openTPU package"

# ---------------------------------------------------------------- probes (no side effects)

cards() {  # the openTPU cards on the bus: 10ee:7028, or any device with subsystem 10ee:4f54
  local d
  for d in /sys/bus/pci/devices/*; do
    [[ "$(cat "$d/vendor")" == 0x10ee ]] || continue
    if [[ "$(cat "$d/device")" == 0x7028 || "$(cat "$d/subsystem_device")" == 0x4f54 ]]; then
      basename "$d"
    fi
  done
}

describe_class() {
  case "$1" in
    0x120000) echo "processing accelerator" ;;
    0x070001) echo "16450 serial port (bitstream before the PCI identity change; the udev rule keeps 8250_pci off it)" ;;
    0x058000) echo "memory controller: not an openTPU bitstream (the card's flash image?)" ;;
    *) echo "unexpected" ;;
  esac
}

mod_path()    { modinfo -k "$kver" -n xdma 2>/dev/null || true; }
mod_tag()     { modinfo -k "$kver" -F otpu xdma 2>/dev/null || true; }
intree()      { find "$moddir/kernel" -name 'xdma.ko*' 2>/dev/null | head -1; }
loaded()      { [[ -d /sys/module/xdma ]]; }
loaded_ours() { [[ -e /sys/module/xdma/parameters/poll_mode ]]; }
have_dkms()   { ((dkms_ok)) && command -v dkms >/dev/null; }
attr()        { cut -c3- "$1/$2"; }   # a sysfs hex attribute without 0x
driver_of()   { local l; l="$(readlink "/sys/bus/pci/devices/$1/driver" 2>/dev/null)" || true; echo "${l##*/}"; }

conf_poll() {  # poll_mode in the installed modprobe.d file ("" when none)
  [[ -f $modconf ]] && sed -n 's/^options xdma .*poll_mode=\([0-9]\).*/\1/p' $modconf | head -1
  return 0
}

users() {  # processes holding /dev/xdma* open or mapped
  local p f hit
  for p in /proc/[0-9]*; do
    hit=0
    for f in "$p"/fd/*; do [[ "$(readlink "$f" 2>/dev/null)" == /dev/xdma* ]] && { hit=1; break; }; done
    grep -qs '/dev/xdma' "$p/maps" && hit=1
    if ((hit)); then
      printf '%s %s\n' "${p#/proc/}" "$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null | cut -c1-100)"
    fi
  done
}

busy() {  # the driver is loaded and something holds it (an open or mapped /dev/xdma* node)
  loaded && [[ "$(cat /sys/module/xdma/refcnt 2>/dev/null || echo 0)" != 0 ]]
}

node_of() {  # the xdma node prefix of a card (xdma<N>), from its sysfs device
  ls "/sys/bus/pci/devices/$1/xdma" 2>/dev/null | sed -n 's/^\(xdma[0-9]*\)_user$/\1/p' | head -1
}

read_id() {  # the ID register (BAR0 offset 0) through /dev/<node>_user, 8 hex digits
  dd if="/dev/$1_user" bs=4 count=1 2>/dev/null | od -An -tx4 | tr -d ' \n'
}

# ---------------------------------------------------------------- --check

check_card() {
  local bdf="$1" d="/sys/bus/pci/devices/$1" cls drv node id spd wid f miss=""
  cls="$(cat "$d/class")"
  say "card $bdf: [$(attr "$d" vendor):$(attr "$d" device)] subsystem \
$(attr "$d" subsystem_vendor):$(attr "$d" subsystem_device) revision $(attr "$d" revision)"
  case "$cls" in
    0x120000|0x070001) ok "class $cls: $(describe_class "$cls")" ;;
    *) bad "class $cls: $(describe_class "$cls")" ;;
  esac
  spd="$(cat "$d/current_link_speed" 2>/dev/null)"; wid="$(cat "$d/current_link_width" 2>/dev/null)"
  if [[ "$spd" == "2.5 GT/s"* && "$wid" == 8 ]]; then ok "link $spd x$wid (the design: Gen1 x8)"
  else note "link $spd x$wid: the design is 2.5 GT/s x8 (it works; DMA is slower)"; fi
  drv="$(driver_of "$bdf")"
  if [[ "$drv" == xdma ]]; then ok "bound to xdma"
  else bad "bound to '${drv:-nothing}', not xdma (driver_override: $(cat "$d/driver_override"))"; return; fi
  node="$(node_of "$bdf")"
  [[ -n "$node" ]] || { bad "no xdma device nodes under $d/xdma"; return; }
  for f in user h2c_0 c2h_0; do [[ -c /dev/${node}_$f ]] || miss+=" ${node}_$f"; done
  [[ -z "$miss" ]] || { bad "missing /dev/:$miss"; return; }
  if [[ -r /dev/${node}_user && -w /dev/${node}_h2c_0 && -r /dev/${node}_c2h_0 ]]; then
    ok "/dev/${node}_{user,h2c_0,c2h_0} usable by $(id -un)"
  else bad "/dev/${node}_* not readable / writable by $(id -un) (60-otpu.rules)"; fi
  id="$(read_id "$node" || true)"
  if [[ "$id" == "$OTPU_ID" ]]; then ok "ID register 0x$id (OTPU)"
  elif [[ "$id" == ffffffff ]]; then bad "ID register 0xffffffff: the card was reprogrammed after enumeration (--rescan)"
  else bad "ID register 0x${id:-unreadable}, want 0x$OTPU_ID"; fi
}

do_check() {
  local p t it want st r c="" u bdfs b
  say "kernel $kver"
  if [[ -d "$moddir/build" ]]; then ok "headers in $moddir/build"
  else note "no kernel headers for $kver (needed to build the driver)"; fi

  say "driver module"
  p="$(mod_path)"; t="$(mod_tag)"; it="$(intree)"
  if [[ -z "$p" ]]; then bad "no xdma module for $kver (run otpu-setup)"
  elif [[ "$t" == "$VER" ]]; then ok "$p (openTPU build $t)"
  elif [[ -n "$t" ]]; then bad "$p is openTPU build $t, this script installs $VER (run otpu-setup)"
  elif [[ "$p" == "$moddir/kernel/"* ]]; then
    bad "modprobe xdma finds only the kernel's own xdma ($p: AMD's platform driver, not for this card): run otpu-setup"
  else bad "$p is not an openTPU build (no otpu tag: built by hand?): run otpu-setup to replace it"; fi
  [[ -z "$it" || "$p" == "$it" ]] || note "the kernel's own xdma ($it) is shadowed (updates/ comes first)"
  if have_dkms; then
    st="$(dkms status -m $PKG -k "$kver" 2>/dev/null | grep -F "$VER" || true)"
    if [[ "$st" == *installed* ]]; then ok "DKMS: $st"
    else note "DKMS present, $PKG/$VER not installed for $kver (run otpu-setup)"; fi
  else note "no DKMS: re-run otpu-setup after each kernel upgrade"; fi

  want="$(conf_poll)"
  if ! loaded; then note "xdma not loaded"
  elif ! loaded_ours; then bad "the loaded xdma is not the Xilinx driver (no poll_mode parameter): the kernel's own?"
  else
    local lp; lp="$(cat /sys/module/xdma/parameters/poll_mode)"
    ok "loaded, poll_mode=$lp ($([[ $lp == 1 ]] && echo polling || echo interrupts))"
    [[ -z "$want" || "$lp" == "$want" ]] || note "$modconf says poll_mode=$want: takes effect at the next reload (--rescan)"
    [[ -z "$p" || "$(cat /sys/module/xdma/srcversion)" == "$(modinfo -F srcversion "$p" 2>/dev/null)" ]] || \
      note "the loaded module is not the installed one: the installed one loads at --rescan or reboot"
  fi

  say "configuration"
  for r in "${rules[@]}"; do
    if cmp -s "$files/$r" "/etc/udev/rules.d/$r"; then ok "/etc/udev/rules.d/$r"
    else bad "/etc/udev/rules.d/$r missing or different from $files/$r"; fi
  done
  if [[ -n "$want" ]]; then ok "$modconf: poll_mode=$want"; else bad "$modconf missing"; fi

  bdfs="$(cards)"
  if [[ -z "$bdfs" ]]; then
    say "card"
    bad "no openTPU card on the bus (10ee:7028): load the bitstream (JTAG, then --rescan), or flash it"
  else
    for b in $bdfs; do check_card "$b"; done
  fi

  say "JTAG (only for loading bitstreams)"
  for u in /sys/bus/usb/devices/*; do
    [[ -f $u/idVendor ]] || continue
    case "$(cat "$u/idVendor"):$(cat "$u/idProduct")" in
      0403:6014) c+=" FT232H (0403:6014, openFPGALoader -c digilent_hs2)" ;;
      03fd:0008) c+=" Platform Cable USB II (03fd:0008)" ;;
      03fd:0013) c+=" Platform Cable USB II (03fd:0013, firmware not loaded)" ;;
    esac
  done
  if [[ -n "$c" ]]; then note "cable:$c"; else note "no JTAG cable seen"; fi
  if command -v openFPGALoader >/dev/null; then note "openFPGALoader: $(command -v openFPGALoader)"
  else note "openFPGALoader not installed"; fi

  echo
  if ((nfail)); then echo "$nfail problem(s)"; exit 1; fi
  echo "all in place"
}

# ---------------------------------------------------------------- install steps

need_build_tools() {
  [[ -d "$moddir/build" ]] || die "kernel headers for $kver missing. Arch: pacman -S linux-headers \
(the headers of your kernel flavor, e.g. linux-lts-headers); Debian / Ubuntu: \
apt install linux-headers-$kver"
  command -v make >/dev/null || die "make missing (Arch: base-devel; Debian / Ubuntu: build-essential)"
  if grep -qs '^CONFIG_CC_IS_CLANG=y' "$moddir/build/.config"; then
    command -v clang >/dev/null || die "this kernel was built with clang: install clang and llvm"
  else
    command -v gcc >/dev/null || die "gcc missing (Arch: base-devel; Debian / Ubuntu: build-essential)"
  fi
  command -v patch >/dev/null || die "patch missing"
}

kbuild_llvm() { grep -qs '^CONFIG_CC_IS_CLANG=y' "$moddir/build/.config" && echo LLVM=1 || true; }

fetch_source() {  # /usr/src/otpu-xdma-<VER>: the pinned driver source with the patch applied
  if [[ -f "$src_dir/.complete" ]]; then ok "source $src_dir"; return; fi
  local tmp gitdir
  command -v git >/dev/null || die "git missing"
  if [[ -n "$src" ]]; then
    gitdir="$(git -c safe.directory='*' -C "$src" rev-parse --absolute-git-dir)" || die "--src $src: not a git clone"
  else
    gitdir="$cache"
  fi
  g() { git -c safe.directory='*' --git-dir="$gitdir" "$@"; }
  if ! g cat-file -e "$XDMA_COMMIT^{commit}" 2>/dev/null; then
    [[ -z "$src" ]] || die "commit $XDMA_COMMIT is not in $src (git -C $src fetch origin)"
    say "fetching $XDMA_REPO at ${XDMA_COMMIT:0:12}"
    [[ -d "$gitdir" ]] || { mkdir -p "$(dirname "$gitdir")"; git init -q --bare "$gitdir"; }
    g fetch -q --depth 1 "$XDMA_REPO" "$XDMA_COMMIT"
  fi
  tmp="$(mktemp -d)"
  g archive "$XDMA_COMMIT" XDMA/linux-kernel/xdma XDMA/linux-kernel/include | tar -x -C "$tmp"
  (cd "$tmp" && patch -p1 -s < "$files/xdma-otpu.patch") || die "xdma-otpu.patch does not apply"
  rm -rf "$src_dir"
  mkdir -p "$src_dir"
  mv "$tmp/XDMA/linux-kernel/xdma" "$tmp/XDMA/linux-kernel/include" "$src_dir/"
  sed "s/@VERSION@/$VER/" "$files/dkms.conf" > "$src_dir/dkms.conf"
  # DKMS cleans with `make clean` in the tree's top directory
  printf 'clean:\n\trm -rf xdma/*.o xdma/*.ko xdma/*.mod xdma/*.mod.c xdma/.*.cmd xdma/modules.order xdma/Module.symvers\n' \
    > "$src_dir/Makefile"
  cp "$files/xdma-otpu.patch" "$src_dir/"
  touch "$src_dir/.complete"
  rm -rf "$tmp"
  ok "source $src_dir (dma_ip_drivers ${XDMA_COMMIT:0:7} + xdma-otpu.patch)"
}

dkms_versions() { dkms status -m $PKG 2>/dev/null | sed -n "s|^$PKG/\([^,:]*\).*|\1|p" | sort -u; }

drop_other_builds() {  # other openTPU driver versions in DKMS and /usr/src
  local v
  if command -v dkms >/dev/null; then
    for v in $(dkms_versions); do
      if [[ "$v" != "$VER" ]] || ! have_dkms; then
        dkms remove -m $PKG -v "$v" --all >/dev/null 2>&1 || true
        note "removed DKMS $PKG/$v"
      fi
    done
  fi
  for v in /usr/src/$PKG-*; do
    [[ -e "$v" && "$v" != "$src_dir" ]] || continue
    rm -rf "$v"; note "removed $v"
  done
}

install_module() {
  local plain="$moddir/updates/xdma.ko" tmp p t it
  if have_dkms; then
    if [[ -f $plain ]]; then rm -f "$plain"; note "removed $plain (DKMS installs the module now)"; fi
    if [[ "$(dkms status -m $PKG -v $VER -k "$kver" 2>/dev/null)" == *installed* ]]; then
      ok "DKMS $PKG/$VER installed for $kver"
    else
      say "DKMS: building $PKG/$VER for $kver"
      dkms status -m $PKG -v $VER 2>/dev/null | grep -q . || dkms add -m $PKG -v $VER >/dev/null
      dkms install -m $PKG -v $VER -k "$kver" --force \
        || die "dkms install failed: see /var/lib/dkms/$PKG/$VER/build/make.log"
      ok "DKMS $PKG/$VER installed (rebuilt automatically at kernel upgrades)"
    fi
  else
    if [[ -f $plain && "$(modinfo -F otpu "$plain" 2>/dev/null)" == "$VER" && \
          "$(modinfo -F vermagic "$plain" | awk '{print $1}')" == "$kver" ]]; then
      ok "$plain (openTPU build $VER)"
    else
      say "building the driver for $kver"
      tmp="$(mktemp -d)"
      cp -r "$src_dir/xdma" "$src_dir/include" "$tmp/"
      make -s -C "$moddir/build" M="$tmp/xdma" $(kbuild_llvm) -j"$(nproc)" modules >/dev/null \
        || die "driver build failed (make -C $moddir/build M=$tmp/xdma modules)"
      if [[ -f $plain && -z "$(modinfo -F otpu "$plain" 2>/dev/null)" ]]; then
        note "replacing $plain (no otpu tag: an earlier build by hand)"
      fi
      install -D -m 644 "$tmp/xdma/xdma.ko" "$plain"
      rm -rf "$tmp"
      ok "installed $plain (no DKMS: re-run otpu-setup after a kernel upgrade)"
    fi
  fi
  depmod -a "$kver"
  p="$(mod_path)"; t="$(mod_tag)"; it="$(intree)"
  [[ "$t" == "$VER" ]] || die "modprobe xdma resolves to $p, not the openTPU build: check the depmod \
search order (/etc/depmod.d, /usr/lib/depmod.d: 'updates' must come before 'built-in')"
  ok "modprobe xdma -> $p"
  if [[ -n "$it" ]]; then
    note "the kernel also ships a module named xdma ($it: AMD's platform DMA driver"
    note "for Alveo cards, not a PCI driver); ours in updates/ takes precedence"
  fi
}

install_config() {
  local r changed=0 cur want
  for r in "${rules[@]}"; do
    if cmp -s "$files/$r" "/etc/udev/rules.d/$r"; then ok "/etc/udev/rules.d/$r"
    else install -m 644 "$files/$r" "/etc/udev/rules.d/$r"; changed=1; ok "installed /etc/udev/rules.d/$r"; fi
  done
  if ((changed)); then
    udevadm control --reload
    udevadm trigger --subsystem-match=xdma --subsystem-match=usb --action=change || true
  fi
  cur="$(conf_poll)"
  want="${poll:-${cur:-$DEFAULT_POLL}}"
  if [[ "$cur" != "$want" ]]; then
    printf '# written by opentpu/host/setup_pcie.sh (--poll / --irq)\noptions xdma poll_mode=%s\n' \
      "$want" > $modconf
  fi
  ok "$modconf: poll_mode=$want ($([[ $want == 1 ]] && echo polling || echo interrupts))"
}

bind_cards() {  # bind every card to xdma (the override matters for serial-class bitstreams)
  local b d drv
  for b in $(cards); do
    d="/sys/bus/pci/devices/$b"
    [[ "$(cat "$d/driver_override")" == xdma ]] || echo xdma > "$d/driver_override"
    drv="$(driver_of "$b")"
    if [[ -n "$drv" && "$drv" != xdma ]]; then
      echo "$b" > "/sys/bus/pci/drivers/$drv/unbind"; note "$b: unbound from $drv"
    fi
    [[ "$drv" == xdma ]] || echo "$b" > /sys/bus/pci/drivers_probe
  done
}

refuse_if_busy() {
  if busy; then
    echo "the card is in use:" >&2
    users | sed 's/^/   /' >&2
    die "$1"
  fi
}

load_driver() {  # (re)load when the loaded module is not the installed one or the mode differs
  local want why=""
  want="$(conf_poll)"
  if loaded; then
    if ! loaded_ours; then why="the kernel's own xdma is loaded"
    elif [[ "$(cat /sys/module/xdma/srcversion)" != "$(modinfo -k "$kver" -F srcversion xdma)" ]]; then
      why="the loaded xdma is not the installed build"
    elif [[ "$(cat /sys/module/xdma/parameters/poll_mode)" != "$want" ]]; then
      why="poll_mode $(cat /sys/module/xdma/parameters/poll_mode) -> $want"
    fi
    if [[ -n "$why" ]]; then
      if busy; then
        note "$why; the card is in use, so this waits for the next --rescan or reboot:"
        users | sed 's/^/         /'
        return 0
      fi
      say "reloading xdma ($why)"
      local n   # a monitor (otpu-smi) may hold the register node for a moment
      for n in 1 2 3 4 5 6 7 8 9 10; do modprobe -r xdma 2>/dev/null && break; sleep 0.5; done
      ! loaded || die "cannot unload xdma: in use (lsmod | grep xdma)"
    fi
  fi
  loaded || modprobe xdma
  bind_cards
  udevadm settle -t 10 || true
}

wait_nodes() {  # up to 10 s for every card's /dev/xdma*_user
  local b n all
  for _ in $(seq 50); do
    all=1
    for b in $(cards); do n="$(node_of "$b")"; [[ -n "$n" && -c /dev/${n}_user ]] || all=0; done
    ((all)) && return 0
    sleep 0.2
  done
}

report_cards() {
  local b node id n=0
  for b in $(cards); do
    n=$((n + 1))
    node="$(node_of "$b")"
    if [[ -z "$node" ]]; then
      bad "$b: no /dev/xdma* nodes (dmesg | grep -i xdma; with the IOMMU on, boot with iommu=pt)"
      continue
    fi
    id="$(read_id "$node" || true)"
    if [[ "$id" == "$OTPU_ID" ]]; then
      ok "$b -> /dev/${node}_*, ID 0x$id (OTPU), class $(cat /sys/bus/pci/devices/$b/class), poll_mode=$(cat /sys/module/xdma/parameters/poll_mode)"
    elif [[ "$id" == ffffffff ]]; then
      bad "$b: ID 0xffffffff: the card was reprogrammed after enumeration: otpu-setup --rescan"
    else
      bad "$b: ID register 0x${id:-unreadable}, want 0x$OTPU_ID: not an openTPU bitstream, or the AXI-Lite path to the accelerator is broken"
    fi
  done
  ((n)) || note "no openTPU card on the bus: load the bitstream over JTAG, then otpu-setup --rescan"
}

do_install() {
  say "XDMA driver ($PKG $VER, kernel $kver)"
  need_build_tools
  fetch_source
  drop_other_builds
  install_module
  say "udev rules and driver options"
  install_config
  say "driver"
  load_driver
  wait_nodes || true
  report_cards
  ((nfail == 0)) || exit 1
  echo "done -- next: otpu-selftest"
}

do_rescan() {
  local b ap
  refuse_if_busy "stop it first: the rescan removes the device under it"
  say "rescan"
  for b in $(cards); do note "removing $b"; echo 1 > "/sys/bus/pci/devices/$b/remove"; done
  # re-add without automatic probing, so that 8250_pci cannot take a serial-class bitstream
  # before the driver override is set (bind_cards binds it)
  ap="$(cat /sys/bus/pci/drivers_autoprobe)"
  trap "echo $ap > /sys/bus/pci/drivers_autoprobe" EXIT
  echo 0 > /sys/bus/pci/drivers_autoprobe
  echo 1 > /sys/bus/pci/rescan
  echo "$ap" > /sys/bus/pci/drivers_autoprobe
  trap - EXIT
  [[ -n "$(cards)" ]] || die "no openTPU card after the rescan: is the bitstream loaded? A JTAG \
load needs the card powered by this PC; if it still does not appear, warm-reboot (the FPGA keeps \
its configuration while the slot stays powered)"
  load_driver
  wait_nodes || true
  report_cards
  ((nfail == 0)) || exit 1
}

do_uninstall() {
  local v k f b
  refuse_if_busy "stop it first"
  say "uninstall"
  if loaded && loaded_ours; then modprobe -r xdma; ok "unloaded xdma"; fi
  if command -v dkms >/dev/null; then
    for v in $(dkms_versions); do
      dkms remove -m $PKG -v "$v" --all >/dev/null && ok "removed DKMS $PKG/$v"
    done
  fi
  for k in /lib/modules/*; do
    f="$k/updates/xdma.ko"
    if [[ -f $f && -n "$(modinfo -F otpu "$f" 2>/dev/null)" ]]; then
      rm -f "$f"; depmod -a "${k##*/}"; ok "removed $f"
    elif [[ -f $f ]]; then note "kept $f (no otpu tag: not installed by this script)"; fi
  done
  rm -rf /usr/src/$PKG-* "${cache%/*}"
  ok "removed /usr/src/$PKG-* and ${cache%/*}"
  for f in "${rules[@]}"; do
    if [[ -f /etc/udev/rules.d/$f ]]; then rm -f "/etc/udev/rules.d/$f"; ok "removed /etc/udev/rules.d/$f"; fi
  done
  udevadm control --reload
  if [[ -f $modconf ]]; then rm -f $modconf; ok "removed $modconf"; fi
  for b in $(cards); do echo > "/sys/bus/pci/devices/$b/driver_override"; done
  echo "done"
}

case $mode in
  check) do_check ;;
  install) do_install ;;
  rescan) do_rescan ;;
  uninstall) do_uninstall ;;
esac
