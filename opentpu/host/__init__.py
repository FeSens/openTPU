"""The host side of openTPU: the userspace driver for the YPCB-00338 card (XDMA or LitePCIe over
PCIe) and its tools.

    board      transports (XDMA, LitePCIe, the Verilator board model), Board, BoardBackend (Engine)
    regs       the control register map (versions 1 and 2; docs/observability.md)
    runstate   the device lock and the runner's status file (/tmp/otpu)
    fake       FakeTransport: an in-memory card with register map 2 (tests, demos)
    power      Vivado report_power -> power.json, and the power estimate
    checks     bring-up checks
    ddrcal     LiteDRAM DDR3 calibration from the host (the LiteDRAM test image; docs/litedram.md)
    smi        otpu-smi         selftest   otpu-selftest
    hwlens     otpu-lens        chat       otpu-chat
    pcie_setup otpu-setup (setup_pcie.sh and its files in pcie/: drivers, udev rules)

The kernel side is, for the XDMA bitstreams, the Xilinx XDMA driver at a pinned commit plus
pcie/xdma-otpu.patch, and for the LitePCIe bitstreams openTPU's own driver otpu_pcie
(pcie/otpu), both installed by otpu-setup (docs/host.md).
"""
