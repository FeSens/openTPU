#!/usr/bin/env bash
# Host PC setup for the openTPU card (docs/host.md section 2). `otpu-setup` runs this with sudo.
#
#   setup_pcie.sh              install or update both PCIe drivers (XDMA's, for the XDMA
#                              bitstreams, and openTPU's LitePCIe driver, for the LitePCIe ones),
#                              the udev rules and the driver options; then load the drivers and
#                              read the card's ID register
#   setup_pcie.sh --check      report what is installed and working, change nothing
#                              (exit 1 when something is missing or wrong)
#   setup_pcie.sh --rescan     after a JTAG load: remove the card from the bus, rescan, bind the
#                              driver its bitstream takes, wait for its /dev/xdma0_user or
#                              /dev/otpu0_user, read the ID register. When the card is not back,
#                              retrain its upstream port's link first
#                              (pcie/relink.sh: Retrain Link, Link Disable, secondary bus reset)
#   setup_pcie.sh --uninstall  remove everything the install put on the system
#   --poll | --irq             the XDMA driver's completion mode (poll_mode=1 | 0), kept in
#                              /etc/modprobe.d/otpu-xdma.conf; without either, an existing
#                              setting stays and a new install takes DEFAULT_POLL
#   --no-dkms                  install the modules for the running kernel only, even with DKMS
#   --src DIR                  take the XDMA driver source from a dma_ip_drivers git clone that
#                              has the pinned commit, instead of fetching it from GitHub
#
# The card's bitstream says which driver it takes, by its PCI ID: an XDMA bitstream is 10ee:7028
# (or subsystem 10ee:4f54, whatever the device ID), which the udev rule pcie/59-otpu-xdma.rules
# gives to xdma; a LitePCIe bitstream is 10ee:5054 (subsystem 10ee:4c50), which otpu_pcie binds
# by its own ID table. Neither takes the other's card, so both stay installed and a JTAG load
# of either kind needs only --rescan.
#
# The XDMA driver is Xilinx's dma_ip_drivers XDMA at a pinned commit plus pcie/xdma-otpu.patch (a
# Makefile fix for kernels >= 6.13 and an "otpu" module tag this script checks). It goes to
# /lib/modules/<kernel>/updates, which takes precedence over the kernel's own module of the same
# name (drivers/dma/xilinx/xdma, AMD's platform driver for Alveo cards, which does not drive
# this card). With DKMS it is rebuilt at every kernel upgrade; without it, re-run this script
# after a kernel upgrade (--check says when the module is missing for the running kernel).
#
# The LitePCIe driver (module otpu_pcie, DKMS package otpu-pcie) is openTPU's own, in pcie/otpu.
# Its version is <its MODULE_VERSION>.<a hash of its source> (dkms.conf is a template, as XDMA's),
# so a changed source is a new version; the source goes to /usr/src/otpu-pcie-<version>, then
# DKMS builds and installs it, or without DKMS it is built into /lib/modules/<kernel>/updates.
#
# Linux only. Safe to re-run: every step checks first and changes only what differs. A driver
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
lp_files="$files/otpu"     # the LitePCIe driver's source
# what a source tree may hold besides the source (lp_find): build products, a copy's marker, and
# a Mac's file metadata (a tree copied from one: ._*, .DS_Store)
lp_junk=('*.o' '*.ko' '*.mod' '*.mod.c' '.*.cmd' modules.order Module.symvers .complete '._*' .DS_Store)

lp_conf() {  # a variable of the LitePCIe driver's dkms.conf, unquoted ("" when it has none)
  sed -n "s/^$1=\"\{0,1\}\([^\"]*\)\"\{0,1\}\$/\1/p" "$lp_files/dkms.conf" 2>/dev/null | head -1 || true
}
lp_find() {  # the LitePCIe driver's source files, relative to its directory, in a fixed order
  local a=() j
  for j in "${lp_junk[@]}"; do a+=(! -name "$j"); done
  (cd "$lp_files" 2>/dev/null && find . -type f "${a[@]}" | LC_ALL=C sort) || true
}
lp_hash() {  # 8 hex digits of the LitePCIe driver's source: its files' names and contents
  local f
  { while read -r f; do echo "$f"; cat "$lp_files/$f"; done < <(lp_find); } \
    | { if command -v sha256sum >/dev/null; then sha256sum; else shasum -a 256; fi; } | cut -c1-8 || true
}
LP_PKG="$(lp_conf PACKAGE_NAME)"; LP_PKG="${LP_PKG:-otpu-pcie}"
LP_MOD="$(lp_conf 'BUILT_MODULE_NAME\[0\]')"; LP_MOD="${LP_MOD:-otpu_pcie}"
# The version: dkms.conf's, or for its template (@VERSION@, which the source copy gets filled in)
# <the driver's MODULE_VERSION>.<the source's hash>, as XDMA's <commit>.<patch level>: a changed
# source is another version, which DKMS builds anew and the old one's builds give way to.
LP_VER="$(lp_conf PACKAGE_VERSION)"
if [[ -z $LP_VER || $LP_VER == @VERSION@ ]]; then
  LP_VER="$(sed -n 's/^MODULE_VERSION("\([^"]*\)");.*/\1/p' "$lp_files"/*.c 2>/dev/null | head -1 || true)"
  LP_VER="${LP_VER:-1}.$(lp_hash)"
fi
lp_src_dir="/usr/src/$LP_PKG-$LP_VER"
lp_stamp="$moddir/updates/$LP_MOD.version"   # a module installed without DKMS: its version
lp_changed=0               # this run installed a new build of the module: load it anew

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
pci=/sys/bus/pci
state=/run/otpu-card-ports     # the cards' upstream ports, kept by --rescan (pcie/relink.sh)
# shellcheck source=pcie/relink.sh
source "$files/relink.sh"

# ---------------------------------------------------------------- probes (no side effects)

cards() {  # the openTPU cards on the bus: XDMA bitstreams (10ee:7028, or any device with
  # subsystem 10ee:4f54) and LitePCIe ones (10ee:5054)
  local d dev
  for d in "$pci"/devices/*; do
    [[ "$(cat "$d/vendor")" == 0x10ee ]] || continue
    dev="$(cat "$d/device")"
    if [[ $dev == 0x7028 || $dev == 0x5054 || "$(cat "$d/subsystem_device")" == 0x4f54 ]]; then
      basename "$d"
    fi
  done
}

card_driver() {  # the driver a card's bitstream takes: otpu_pcie for LitePCIe's 10ee:5054, else xdma
  if [[ "$(cat "$pci/devices/$1/device")" == 0x5054 ]]; then echo "$LP_MOD"; else echo xdma; fi
}

node_class() {  # the sysfs class (and /dev node prefix) of a driver's nodes: xdma, otpu
  if [[ $1 == xdma ]]; then echo xdma; else echo otpu; fi
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
lp_mod_path() { modinfo -k "$kver" -n "$LP_MOD" 2>/dev/null || true; }
lp_loaded()   { [[ -d /sys/module/$LP_MOD ]]; }
lp_params()   {  # the loaded otpu_pcie's parameters, " name=value" each
  local f; for f in /sys/module/"$LP_MOD"/parameters/*; do [[ -f $f ]] && printf ' %s=%s' "${f##*/}" "$(cat "$f")"; done; return 0
}
have_dkms()   { ((dkms_ok)) && command -v dkms >/dev/null; }
attr()        { cut -c3- "$1/$2"; }   # a sysfs hex attribute without 0x
driver_of()   { local l; l="$(readlink "/sys/bus/pci/devices/$1/driver" 2>/dev/null)" || true; echo "${l##*/}"; }

conf_poll() {  # poll_mode in the installed modprobe.d file ("" when none)
  [[ -f $modconf ]] && sed -n 's/^options xdma .*poll_mode=\([0-9]\).*/\1/p' $modconf | head -1
  return 0
}

users() {  # processes holding /dev/xdma* or /dev/otpu* (either driver's nodes) open or mapped
  local p f l hit
  for p in /proc/[0-9]*; do
    hit=0
    for f in "$p"/fd/*; do
      l="$(readlink "$f" 2>/dev/null)" || continue
      [[ $l == /dev/xdma* || $l == /dev/otpu* ]] && { hit=1; break; }
    done
    grep -qsE '/dev/(xdma|otpu)' "$p/maps" && hit=1
    if ((hit)); then
      printf '%s %s\n' "${p#/proc/}" "$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null | cut -c1-100)"
    fi
  done
}

busy_mod() {  # module $1 is loaded and something holds it (an open or mapped node of its)
  [[ -d /sys/module/$1 && "$(cat "/sys/module/$1/refcnt" 2>/dev/null || echo 0)" != 0 ]]
}

busy() { busy_mod xdma || busy_mod "$LP_MOD"; }   # either driver's nodes held

node_of() {  # the node prefix of a card (xdma<N> or otpu<N>), from its driver's class directory
  # under its sysfs device
  ls "/sys/bus/pci/devices/$1/xdma" "/sys/bus/pci/devices/$1/otpu" 2>/dev/null \
    | sed -En 's/^((xdma|otpu)[0-9]+)_user$/\1/p' | head -1
}

read_id() {  # the ID register (BAR0 offset 0) through /dev/<node>_user, 8 hex digits: a read
  # (XDMA's node takes one), else a 32-bit load from a mapping of the node (LitePCIe's)
  local id
  id="$(dd if="/dev/$1_user" bs=4 count=1 2>/dev/null | od -An -tx4 | tr -d ' \n')"
  if [[ -z "$id" ]] && command -v python3 >/dev/null; then
    id="$(python3 -c 'import mmap, os, sys
m = mmap.mmap(os.open(sys.argv[1], os.O_RDONLY | os.O_SYNC), 4096, mmap.MAP_SHARED, mmap.PROT_READ)
v = memoryview(m).cast("I")
print("%08x" % v[0])
v.release()
m.close()' "/dev/$1_user" 2>/dev/null)" || id=""
  fi
  echo "$id"
}

drv_state() {  # a card's driver's settings, for the report: XDMA's poll_mode, otpu_pcie's parameters
  if [[ $1 == xdma ]]; then echo "poll_mode=$(cat /sys/module/xdma/parameters/poll_mode 2>/dev/null)"
  else echo "$1$(lp_params)"; fi
}

# ---------------------------------------------------------------- --check

check_card() {
  local bdf="$1" d="/sys/bus/pci/devices/$1" cls drv want node id spd wid msp mwd f miss=""
  cls="$(cat "$d/class")"
  say "card $bdf: [$(attr "$d" vendor):$(attr "$d" device)] subsystem \
$(attr "$d" subsystem_vendor):$(attr "$d" subsystem_device) revision $(attr "$d" revision)"
  case "$cls" in
    0x120000|0x070001) ok "class $cls: $(describe_class "$cls")" ;;
    *) bad "class $cls: $(describe_class "$cls")" ;;
  esac
  spd="$(cat "$d/current_link_speed" 2>/dev/null)"; wid="$(cat "$d/current_link_width" 2>/dev/null)"
  msp="$(cat "$d/max_link_speed" 2>/dev/null)"            # the card's own: its bitstream's Gen
  mwd="$(cat "$d/max_link_width" 2>/dev/null)"            # and lanes (YPCB-00338 x8, BCU1525 x16)
  mwd="${mwd:-8}"
  if [[ -n "$spd" && "$spd" == "$msp" && "$wid" == "$mwd" ]]; then ok "link $spd x$wid (the card's own speed and width)"
  else note "link $spd x$wid: the card can do ${msp:-?} x$mwd (it works; DMA is slower; --rescan or a warm reboot retrains)"; fi
  drv="$(driver_of "$bdf")"; want="$(card_driver "$bdf")"
  if [[ "$drv" == "$want" ]]; then
    ok "bound to $want ($([[ $want == xdma ]] && echo "an XDMA" || echo "a LitePCIe") bitstream)"
  else bad "bound to '${drv:-nothing}', not $want (driver_override: $(cat "$d/driver_override"))"; return; fi
  node="$(node_of "$bdf")"
  [[ -n "$node" ]] || { bad "no $want device nodes under $d/$(node_class "$want")"; return; }
  for f in user h2c_0 c2h_0; do [[ -c /dev/${node}_$f ]] || miss+=" ${node}_$f"; done
  [[ -z "$miss" ]] || { bad "missing /dev/:$miss"; return; }
  if [[ -r /dev/${node}_user && -w /dev/${node}_h2c_0 && -r /dev/${node}_c2h_0 ]]; then
    ok "/dev/${node}_{user,h2c_0,c2h_0} usable by $(id -un)"
  else bad "/dev/${node}_* not readable / writable by $(id -un) (60-otpu.rules)"; fi
  id="$(read_id "$node" || true)"
  if [[ "$id" == "$OTPU_ID" ]]; then ok "ID register 0x$id (OTPU)"
  elif [[ "$id" == ffffffff ]]; then bad "ID register 0xffffffff: the card was reprogrammed after enumeration (--rescan)"
  else bad "ID register 0x${id:-unreadable}, want 0x$OTPU_ID"; fi
  [[ $want == xdma ]] || lp_dma_state "/sys/class/otpu/${node}_user"
}

lp_dma_state() {  # otpu_pcie's DMA state, from its attributes on the card's user node ($1)
  local a=() f v r
  for f in max_payload max_read_request irqs mm_status; do
    v="$(cat "$1/$f" 2>/dev/null)" || v="?"
    [[ $f == mm_status && $v != "?" ]] && v="$(printf '%#x' "$v")"
    a+=("${f//_/ } $v")
  done
  r="$(cat "$1/resets" 2>/dev/null)" || r="?"
  if [[ $r == 0 ]]; then ok "DMA: ${a[*]/%/,} no resets"
  else note "DMA: ${a[*]/%/,} resets $r (each after a transfer that timed out: dmesg | grep $LP_MOD)"; fi
}

check_lp_module() {  # --check's LitePCIe driver section
  local p st a b
  say "LitePCIe driver module ($LP_MOD: LitePCIe bitstreams)"
  if [[ ! -f "$lp_files/dkms.conf" ]]; then
    note "$lp_files/dkms.conf missing: this package has no LitePCIe driver"; return
  fi
  p="$(lp_mod_path)"
  if [[ -z "$p" ]]; then bad "no $LP_MOD module for $kver (run otpu-setup)"
  elif [[ -f $lp_stamp && "$p" == "$moddir/updates/$LP_MOD.ko" ]]; then
    if [[ "$(cat "$lp_stamp")" == "$LP_VER" ]]; then ok "$p ($LP_PKG $LP_VER, no DKMS)"
    else bad "$p is $LP_PKG $(cat "$lp_stamp"), this package's is $LP_VER (run otpu-setup)"; fi
  else ok "$p"; fi
  [[ -f "$lp_src_dir/.complete" ]] || bad "$lp_src_dir missing: this package's $LP_PKG $LP_VER is not installed (run otpu-setup)"
  if have_dkms; then
    st="$(dkms status -m "$LP_PKG" -k "$kver" 2>/dev/null || true)"
    if [[ "$(grep -F "$LP_PKG/$LP_VER" <<< "$st")" == *installed* ]]; then ok "DKMS: $(grep -F "$LP_PKG/$LP_VER" <<< "$st")"
    elif [[ -n "$st" ]]; then bad "DKMS has ${st//$'\n'/; }, this package's is $LP_PKG/$LP_VER (run otpu-setup)"
    else note "DKMS present, $LP_PKG/$LP_VER not installed for $kver (run otpu-setup)"; fi
  fi
  if ! lp_loaded; then note "$LP_MOD not loaded (it loads for a LitePCIe card, by its PCI ID)"
  else
    ok "loaded$(lp_params)"
    a="$(cat "/sys/module/$LP_MOD/srcversion" 2>/dev/null || true)"
    b="$(modinfo -F srcversion "$p" 2>/dev/null || true)"
    [[ -z "$a" || -z "$b" || "$a" == "$b" ]] || \
      note "the loaded $LP_MOD is not the installed one: the installed one loads at --rescan or reboot"
  fi
}

do_check() {
  local p t it want st r c="" u bdfs b
  say "kernel $kver"
  if [[ -d "$moddir/build" ]]; then ok "headers in $moddir/build"
  else note "no kernel headers for $kver (needed to build the driver)"; fi

  say "XDMA driver module (xdma: XDMA bitstreams)"
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

  check_lp_module

  say "configuration"
  for r in "${rules[@]}"; do
    if cmp -s "$files/$r" "/etc/udev/rules.d/$r"; then ok "/etc/udev/rules.d/$r"
    else bad "/etc/udev/rules.d/$r missing or different from $files/$r"; fi
  done
  if [[ -n "$want" ]]; then ok "$modconf: poll_mode=$want"; else bad "$modconf missing"; fi

  bdfs="$(cards)"
  if [[ -z "$bdfs" ]]; then
    say "card"
    bad "no openTPU card on the bus (10ee:7028 XDMA, 10ee:5054 LitePCIe): load the bitstream (JTAG, then --rescan), or flash it"
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

dkms_versions() { dkms status -m "$1" 2>/dev/null | sed -n "s|^$1/\([^,:]*\).*|\1|p" | sort -u; }

drop_other_builds() {  # other versions of DKMS package $1 (this script installs version $2,
  # source $3) in DKMS and /usr/src
  local v
  if command -v dkms >/dev/null; then
    for v in $(dkms_versions "$1"); do
      if [[ "$v" != "$2" ]] || ! have_dkms; then
        dkms remove -m "$1" -v "$v" --all >/dev/null 2>&1 || true
        note "removed DKMS $1/$v"
      fi
    done
  fi
  for v in /usr/src/"$1"-*; do
    [[ -e "$v" && "$v" != "$3" ]] || continue
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

lp_fetch_source() {  # /usr/src/<package>-<version>: the LitePCIe driver's source, its dkms.conf
  # with the version filled in (the version names the source: lp_hash)
  if [[ -f "$lp_src_dir/.complete" ]]; then ok "source $lp_src_dir"; return; fi
  local f
  rm -rf "$lp_src_dir"
  while read -r f; do install -D -m 644 "$lp_files/$f" "$lp_src_dir/$f"; done < <(lp_find)
  sed -i "s/@VERSION@/$LP_VER/" "$lp_src_dir/dkms.conf"
  touch "$lp_src_dir/.complete"
  ok "source $lp_src_dir (from $lp_files)"
}

lp_install_module() {  # the LitePCIe driver built for the running kernel: by DKMS, else into
  # updates/ with its version beside it (lp_stamp)
  local plain="$moddir/updates/$LP_MOD.ko" tmp p
  if have_dkms; then
    if [[ -f $lp_stamp ]]; then   # (ours, from a run without DKMS: not one DKMS put there)
      rm -f "$plain" "$lp_stamp"; note "removed $plain (DKMS installs the module now)"
    fi
    if [[ "$(dkms status -m "$LP_PKG" -v "$LP_VER" -k "$kver" 2>/dev/null)" == *installed* ]]; then
      ok "DKMS $LP_PKG/$LP_VER installed for $kver"
    else
      say "DKMS: building $LP_PKG/$LP_VER for $kver"
      dkms status -m "$LP_PKG" -v "$LP_VER" 2>/dev/null | grep -q . || dkms add -m "$LP_PKG" -v "$LP_VER" >/dev/null
      dkms install -m "$LP_PKG" -v "$LP_VER" -k "$kver" --force \
        || die "dkms install failed: see /var/lib/dkms/$LP_PKG/$LP_VER/build/make.log"
      lp_changed=1
      ok "DKMS $LP_PKG/$LP_VER installed (rebuilt automatically at kernel upgrades)"
    fi
  else
    if [[ -f $plain && "$(cat "$lp_stamp" 2>/dev/null)" == "$LP_VER" && \
          "$(modinfo -F vermagic "$plain" | awk '{print $1}')" == "$kver" ]]; then
      ok "$plain ($LP_PKG $LP_VER)"
    else
      say "building $LP_MOD for $kver"
      tmp="$(mktemp -d)"
      cp -r "$lp_src_dir"/. "$tmp/"
      make -s -C "$moddir/build" M="$tmp" $(kbuild_llvm) -j"$(nproc)" modules >/dev/null \
        || die "$LP_MOD build failed (make -C $moddir/build M=$tmp modules)"
      install -D -m 644 "$tmp/$LP_MOD.ko" "$plain"
      echo "$LP_VER" > "$lp_stamp"
      rm -rf "$tmp"
      lp_changed=1
      ok "installed $plain ($LP_PKG $LP_VER; no DKMS: re-run otpu-setup after a kernel upgrade)"
    fi
  fi
  depmod -a "$kver"
  p="$(lp_mod_path)"
  [[ -n "$p" ]] || die "modprobe $LP_MOD finds no module after the install (depmod -a $kver)"
  ok "modprobe $LP_MOD -> $p"
}

lp_install() {  # the LitePCIe driver: its source into /usr/src, then the module
  if [[ ! -f "$lp_files/dkms.conf" ]]; then
    note "$lp_files/dkms.conf missing: this package has no LitePCIe driver to install"; return 0
  fi
  lp_fetch_source
  drop_other_builds "$LP_PKG" "$LP_VER" "$lp_src_dir"
  lp_install_module
}

install_config() {
  local r changed=0 cur want
  for r in "${rules[@]}"; do
    if cmp -s "$files/$r" "/etc/udev/rules.d/$r"; then ok "/etc/udev/rules.d/$r"
    else install -m 644 "$files/$r" "/etc/udev/rules.d/$r"; changed=1; ok "installed /etc/udev/rules.d/$r"; fi
  done
  if ((changed)); then
    udevadm control --reload
    udevadm trigger --subsystem-match=xdma --subsystem-match=otpu --subsystem-match=usb --action=change || true
  fi
  cur="$(conf_poll)"
  want="${poll:-${cur:-$DEFAULT_POLL}}"
  if [[ "$cur" != "$want" ]]; then
    printf '# written by opentpu/host/setup_pcie.sh (--poll / --irq)\noptions xdma poll_mode=%s\n' \
      "$want" > $modconf
  fi
  ok "$modconf: poll_mode=$want ($([[ $want == 1 ]] && echo polling || echo interrupts))"
}

bind_cards() {  # bind every card to the driver its bitstream takes (card_driver): an XDMA one to
  # xdma by the override (it matters for serial-class bitstreams), a LitePCIe one to otpu_pcie by
  # its ID table (an override left on the device is cleared)
  local b d drv want
  for b in $(cards); do
    d="/sys/bus/pci/devices/$b"
    want="$(card_driver "$b")"
    if [[ $want == xdma ]]; then
      [[ "$(cat "$d/driver_override")" == xdma ]] || echo xdma > "$d/driver_override"
    elif [[ "$(cat "$d/driver_override")" != "(null)" ]]; then
      echo > "$d/driver_override"
    fi
    drv="$(driver_of "$b")"
    if [[ -n "$drv" && "$drv" != "$want" ]]; then
      echo "$b" > "/sys/bus/pci/drivers/$drv/unbind"; note "$b: unbound from $drv"
    fi
    [[ "$drv" == "$want" ]] || echo "$b" > /sys/bus/pci/drivers_probe
  done
}

refuse_if_busy() {
  if busy; then
    echo "the card is in use:" >&2
    users | sed 's/^/   /' >&2
    die "$1"
  fi
}

load_xdma() {  # (re)load xdma when the loaded module is not the installed one or the mode differs
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
      if busy_mod xdma; then
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
}

load_lp() {  # (re)load otpu_pcie when the loaded module is not the installed one (none
  # installed: nothing to load)
  local why="" a b n
  [[ -n "$(lp_mod_path)" ]] || return 0
  if lp_loaded; then
    a="$(cat "/sys/module/$LP_MOD/srcversion" 2>/dev/null || true)"
    b="$(modinfo -k "$kver" -F srcversion "$LP_MOD" 2>/dev/null || true)"
    if ((lp_changed)) || [[ -n "$a" && -n "$b" && "$a" != "$b" ]]; then
      why="the loaded $LP_MOD is not the installed build"
    fi
    if [[ -n "$why" ]]; then
      if busy_mod "$LP_MOD"; then
        note "$why; the card is in use, so this waits for the next --rescan or reboot:"
        users | sed 's/^/         /'
        return 0
      fi
      say "reloading $LP_MOD ($why)"
      for n in 1 2 3 4 5 6 7 8 9 10; do modprobe -r "$LP_MOD" 2>/dev/null && break; sleep 0.5; done
      ! lp_loaded || die "cannot unload $LP_MOD: in use (lsmod | grep $LP_MOD)"
    fi
  fi
  lp_loaded || modprobe "$LP_MOD"
}

load_drivers() {  # both drivers loaded (the installed builds), every card bound to its own
  load_xdma
  load_lp
  bind_cards
  udevadm settle -t 10 || true
}

wait_nodes() {  # up to 10 s for every card's /dev/xdma*_user or /dev/otpu*_user
  local b n all
  for _ in $(seq 50); do
    all=1
    for b in $(cards); do n="$(node_of "$b")"; [[ -n "$n" && -c /dev/${n}_user ]] || all=0; done
    ((all)) && return 0
    sleep 0.2
  done
}

report_cards() {
  local b node id want n=0
  for b in $(cards); do
    n=$((n + 1))
    node="$(node_of "$b")"; want="$(card_driver "$b")"
    if [[ -z "$node" ]]; then
      bad "$b: no /dev/$(node_class "$want")* nodes from $want (dmesg | grep -i $want; with the IOMMU on, boot with iommu=pt)"
      continue
    fi
    id="$(read_id "$node" || true)"
    if [[ "$id" == "$OTPU_ID" ]]; then
      ok "$b -> /dev/${node}_*, ID 0x$id (OTPU), class $(cat /sys/bus/pci/devices/$b/class), $(drv_state "$want")"
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
  drop_other_builds "$PKG" "$VER" "$src_dir"
  install_module
  say "LitePCIe driver ($LP_PKG $LP_VER, kernel $kver)"
  lp_install
  say "udev rules and driver options"
  install_config
  say "drivers"
  load_drivers
  wait_nodes || true
  report_cards
  ((nfail == 0)) || exit 1
  echo "done -- next: otpu-selftest"
}

do_rescan() {
  refuse_if_busy "stop it first: the rescan removes the device under it"
  say "rescan"
  trap relink_cleanup EXIT
  rescan_cards || die "no openTPU card after the rescan and the link retrain: is the bitstream \
loaded? A JTAG load needs the card powered by this PC; if it still does not appear, warm-reboot \
(the FPGA keeps its configuration while the slot stays powered)"
  trap - EXIT
  load_drivers
  wait_nodes || true
  report_cards
  ((nfail == 0)) || exit 1
}

do_uninstall() {
  local v k f b
  refuse_if_busy "stop it first"
  say "uninstall"
  if loaded && loaded_ours; then modprobe -r xdma; ok "unloaded xdma"; fi
  if lp_loaded; then modprobe -r "$LP_MOD"; ok "unloaded $LP_MOD"; fi
  if command -v dkms >/dev/null; then
    for v in $(dkms_versions $PKG); do
      dkms remove -m $PKG -v "$v" --all >/dev/null && ok "removed DKMS $PKG/$v"
    done
    for v in $(dkms_versions "$LP_PKG"); do
      dkms remove -m "$LP_PKG" -v "$v" --all >/dev/null && ok "removed DKMS $LP_PKG/$v"
    done
  fi
  for k in /lib/modules/*; do
    f="$k/updates/xdma.ko"
    if [[ -f $f && -n "$(modinfo -F otpu "$f" 2>/dev/null)" ]]; then
      rm -f "$f"; depmod -a "${k##*/}"; ok "removed $f"
    elif [[ -f $f ]]; then note "kept $f (no otpu tag: not installed by this script)"; fi
    f="$k/updates/$LP_MOD.ko"           # (the name is openTPU's alone)
    if [[ -f $f ]]; then rm -f "$f" "$k/updates/$LP_MOD.version"; depmod -a "${k##*/}"; ok "removed $f"; fi
  done
  rm -rf /usr/src/$PKG-* /usr/src/"$LP_PKG"-* "${cache%/*}"
  ok "removed /usr/src/$PKG-*, /usr/src/$LP_PKG-* and ${cache%/*}"
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
