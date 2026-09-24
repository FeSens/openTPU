"""The host side of openTPU: the userspace driver for the YPCB-00338 card (XDMA over PCIe) and
its tools.

    board      transports (XDMA, the Verilator board model), Board, BoardBackend (Engine)
    regs       the control register map (versions 1 and 2; docs/observability.md)
    runstate   the device lock and the runner's status file (/tmp/otpu)
    fake       FakeTransport: an in-memory card with register map 2 (tests, demos)
    power      Vivado report_power -> power.json, and the power estimate
    checks     bring-up checks
    smi        otpu-smi         selftest   otpu-selftest
    hwlens     otpu-lens        chat       otpu-chat

The kernel side is the stock Xilinx XDMA driver (docs/host.md).
"""
