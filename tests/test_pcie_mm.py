"""otpu_pcie_mm (rtl/shell: LitePCIe's DMA streams onto the card's memory port) in its bench,
sim/verilator/tb_pcie_mm.sv: through otpu_dma_split and both channels' otpu_mem_ch against
LiteDRAM port models at the card's clocks (docs/litepcie.md section 3). Each scenario's bench
checks every C2H beat against a byte shadow of the channels, h2c_done once per command in order
and never before its last write response, and its assertions (no burst across 512 bytes, W
never starved, R never into a full FIFO); it prints PASS or the first mismatches."""
import re
import subprocess

import pytest

from opentpu import rtlsim

SOURCES = [rtlsim.RTL / "shell" / f for f in ("otpu_afifo.sv", "otpu_axi_split2.sv", "otpu_mem_ch.sv")] + [
    rtlsim.TB / "otpu_ldn_model.sv", rtlsim.RTL / "shell" / "otpu_pcie_mm.sv", rtlsim.TB / "tb_pcie_mm.sv"]


@pytest.fixture(scope="module")
def bench(have_verilator):
    return rtlsim.build("tb_pcie_mm", SOURCES)


def _run(exe, *args):
    r = subprocess.run([str(exe), *args], capture_output=True, text=True, timeout=600)
    return r.stdout + r.stderr


@pytest.mark.parametrize("args", [
    ("+seed=1",),
    ("+seed=3", "+hgap=60", "+cstall=60"),          # streams held back most of the time
    ("+seed=4", "+fill=40", "+zero=10"),            # FILL and zero-length commands
    ("+seed=5", "+al64=0", "+hgap=0", "+cstall=0"),  # never 64-byte aligned, streams never held
    ("+seed=6", "+mix=0"),                          # reads not overlapping the next writes
    ("+seed=7", "+al64=100"),
], ids=["default", "stalls", "fill-zero", "unaligned", "nomix", "aligned"])
def test_pcie_mm_scenarios(bench, args):
    out = _run(bench, *args)
    assert re.search(r"^PASS", out, re.M) and "fatal" not in out.lower(), out[-3000:]


# Throughput with the streams never held: the H2C side takes a beat a cycle (4 GB/s at 250 MHz,
# the PCIe Gen2 x8 link's raw rate), the C2H side one channel's reads (otpu_mem_ch's 16
# outstanding: 0.71 beats a cycle, 2.85 GB/s, above what XDMA's C2H reached on the card)
def test_pcie_mm_throughput(bench):
    out = _run(bench, "+seq=1")
    m = re.search(r"seq: H2C ([0-9.]+) beats/cycle .* C2H ([0-9.]+) beats/cycle", out)
    assert m and re.search(r"^PASS", out, re.M), out[-3000:]
    assert float(m[1]) >= 0.99 and float(m[2]) >= 0.70, out[-3000:]
