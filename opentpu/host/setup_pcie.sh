#!/usr/bin/env bash
# Host PC setup for the openTPU card (docs/host.md, sections 2-3), in one command:
#
#   opentpu/host/setup_pcie.sh            # find the card, build/load the XDMA driver, udev rule, ID check
#   opentpu/host/setup_pcie.sh --rescan   # after JTAG programming: remove/rescan the card, reload the driver
#   XDMA_POLL=1 opentpu/host/setup_pcie.sh # load the driver in poll mode (DMA timeouts: no interrupts)
#
# Linux x86-64 only; asks for sudo where needed. Safe to re-run.
set -euo pipefail

DRV_DIR="${XDMA_DRIVERS:-$HOME/dma_ip_drivers}"
rescan=0
[[ "${1:-}" == "--rescan" ]] && rescan=1

say() { printf '\033[1m== %s\033[0m\n' "$*"; }
die() { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

[[ "$(uname -s)" == Linux ]] || die "run this on the Linux PC that holds the card"

# ---- 1. the card on the bus
say "PCIe: looking for the card (vendor 10ee)"
bdf="$(lspci -D -d 10ee: | awk '{print $1}' | head -1 || true)"
if [[ -n "$bdf" && $rescan == 1 ]]; then
  say "rescan: removing $bdf and rescanning the bus"
  sudo modprobe -r xdma 2>/dev/null || true
  echo 1 | sudo tee "/sys/bus/pci/devices/$bdf/remove" >/dev/null
  sleep 1
fi
if [[ -z "$bdf" || $rescan == 1 ]]; then
  echo 1 | sudo tee /sys/bus/pci/rescan >/dev/null
  sleep 1
  bdf="$(lspci -D -d 10ee: | awk '{print $1}' | head -1 || true)"
fi
[[ -n "$bdf" ]] || die "no Xilinx device on the bus: is the bitstream loaded (flash or JTAG)? \
After JTAG programming run: $0 --rescan (or reboot with the bitstream in flash)"
lspci -s "$bdf" -nn
sta="$(sudo lspci -s "$bdf" -vv | grep -E 'LnkSta:' | head -1 || true)"
echo "   $sta"
[[ "$sta" == *"2.5GT/s"* && "$sta" == *"Width x8"* ]] || \
  echo "   warning: expected Gen1 (2.5GT/s) x8; the link works but DMA bandwidth is lower"

# ---- 2. the XDMA driver (built and installed once; --rescan unloads it, so test for the module
# file, not for a loaded module)
if ! modinfo xdma >/dev/null 2>&1; then
  say "XDMA driver: building from $DRV_DIR"
  if ! command -v make >/dev/null || ! command -v gcc >/dev/null; then die "install gcc and make"; fi
  [[ -d "/lib/modules/$(uname -r)/build" ]] || \
    die "kernel headers missing: sudo apt install linux-headers-$(uname -r)"
  [[ -d "$DRV_DIR" ]] || git clone --depth 1 https://github.com/Xilinx/dma_ip_drivers "$DRV_DIR"
  did="$(lspci -s "$bdf" -n | awk '{print $3}' | cut -d: -f2)"
  if ! grep -qi "0x$did" "$DRV_DIR/XDMA/linux-kernel/xdma/xdma_mod.c"; then
    die "device ID 10ee:$did is not in the driver's pci_ids[] ($DRV_DIR/XDMA/linux-kernel/xdma/xdma_mod.c): add it and re-run"
  fi
  make -C "$DRV_DIR/XDMA/linux-kernel/xdma" -j"$(nproc)"
  sudo make -C "$DRV_DIR/XDMA/linux-kernel/xdma" install
  sudo depmod -a
fi
# XDMA_POLL=1: poll_mode=1 (no interrupts; for DMA calls that time out in dmesg)
if [[ "${XDMA_POLL:-0}" == 1 ]]; then sudo modprobe -r xdma 2>/dev/null || true; fi
sudo modprobe xdma ${XDMA_POLL:+poll_mode=$XDMA_POLL}
sleep 1

# ---- 3. device nodes, non-root access
rule=/etc/udev/rules.d/60-xdma.rules
if [[ ! -f $rule ]]; then
  say "udev: $rule (non-root access to /dev/xdma*)"
  echo 'KERNEL=="xdma[0-9]*", MODE="0666"' | sudo tee $rule >/dev/null
  sudo udevadm control --reload
  sudo udevadm trigger
  sleep 1
fi
ls /dev/xdma0_user /dev/xdma0_h2c_0 /dev/xdma0_c2h_0 >/dev/null 2>&1 || \
  die "no /dev/xdma0_* nodes: dmesg | grep -i xdma (IOMMU on? boot with iommu=pt)"

# ---- 4. the accelerator answers
say "registers: ID at BAR offset 0"
id="$(dd if=/dev/xdma0_user bs=4 count=1 2>/dev/null | od -An -tx4 | tr -d ' ')"
if [[ "$id" == "4f545055" ]]; then
  echo "   ID 0x$id (OTPU) -- next: otpu-selftest"
else
  die "ID register reads 0x$id, want 0x4f545055: the PCIe link is up but the AXI-Lite path \
to the accelerator is not (check the block design address map / core_clk / reset)"
fi
