"""The host side of the two PCIe drivers (opentpu/host/board.py): Xilinx XDMA's /dev/xdmaN nodes
and openTPU's LitePCIe driver's /dev/otpuN (opentpu/host/pcie/otpu). Which node a tool opens
(card_dev, card_transport, card_nodes, card_sysfs), the lock it takes, LitePcieTransport's
calls (no RUN_H2C limit during a run; every DMA call's card address, length and host buffer on
the driver's 16-byte grid) against files in place of the device nodes, and setup_pcie.sh's view
of the card's PCI ID and the LitePCIe driver's dkms.conf, and 60-otpu.rules' non-root nodes.
"""
import fnmatch
import mmap
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from opentpu.host import board as B
from opentpu.host import runstate as rs

ROOT = Path(__file__).resolve().parents[1]
HOST = ROOT / "opentpu" / "host"


@pytest.fixture(autouse=True)
def run_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("OTPU_RUN_DIR", str(tmp_path / "run"))


def _dev_nodes(monkeypatch, *names):
    """Of the cards' register nodes (/dev/xdmaN_user, /dev/otpuN_user) these exist, no other
    (every other path as on disk)."""
    real, node = os.path.exists, re.compile(r"/dev/(otpu|xdma)\d+_user")
    have = {f"/dev/{n}_user" for n in names}
    monkeypatch.setattr(B.os.path, "exists",
                        lambda p: str(p) in have if node.fullmatch(str(p)) else real(p))


# ------------------------------------------------------------------------------ resolution
@pytest.mark.parametrize("nodes,want", [((), "/dev/xdma0"), (("xdma0",), "/dev/xdma0"),
                                        (("otpu0",), "/dev/otpu0"),
                                        (("otpu0", "xdma0"), "/dev/otpu0")])
def test_the_default_is_card_0_of_whichever_driver_has_it(monkeypatch, nodes, want):
    _dev_nodes(monkeypatch, *nodes)
    assert B.card_dev() == B.card_dev(None) == want


def test_an_explicit_device_wins_when_it_exists_else_the_same_card_under_the_other_driver(
        monkeypatch):
    _dev_nodes(monkeypatch, "otpu0", "xdma1")
    assert B.card_dev("/dev/xdma1") == "/dev/xdma1"         # exists: itself, otpu0 or not
    assert B.card_dev("/dev/otpu0") == "/dev/otpu0"
    assert B.card_dev("/dev/xdma0") == "/dev/otpu0"         # a script's old default
    assert B.card_dev("/dev/otpu1") == "/dev/xdma1"
    assert B.card_dev("/dev/xdma2") == "/dev/xdma2"         # neither: as given (the open names it)
    assert B.card_dev("/dev/nonexistent_xdma") == "/dev/nonexistent_xdma"


def test_card_dev_on_paths_outside_dev(tmp_path):
    (tmp_path / "otpu3_user").write_bytes(b"")
    assert B.card_dev(str(tmp_path / "otpu3")) == str(tmp_path / "otpu3")
    assert B.card_dev(str(tmp_path / "xdma3")) == str(tmp_path / "otpu3")
    assert B.card_dev(str(tmp_path / "xdmaT")) == str(tmp_path / "xdmaT")   # not a driver's name


def test_card_nodes_lists_both_drivers_nodes(monkeypatch):
    found = {"/dev/otpu[0-9]*_user": ["/dev/otpu1_user", "/dev/otpu0_user"],
             "/dev/xdma[0-9]*_user": ["/dev/xdma0_user"]}
    monkeypatch.setattr(B.glob, "glob", lambda pat: found.get(pat, []))
    assert B.card_nodes() == ["/dev/otpu0", "/dev/otpu1", "/dev/xdma0"]
    from opentpu.host import smi
    assert smi.find_devices() == B.card_nodes()


def test_node_kind_and_sysfs_path():
    assert B.node_kind("/dev/otpu0") == "otpu" and B.node_kind("/dev/xdma1") == "xdma"
    assert B.node_kind("/tmp/fake0") == "xdma"
    assert B.card_sysfs("/dev/otpu2") == Path("/sys/class/otpu/otpu2_user/device")
    assert B.card_sysfs("/dev/xdma0") == Path("/sys/class/xdma/xdma0_user/device")
    assert B.DRIVERS == {"otpu": "otpu_pcie", "xdma": "xdma"}


# ------------------------------------------------------------------------------ the transport
def _nodes(d: Path, name: str, size: int = (1 << 31) + (1 << 21)) -> str:
    """Files in place of a card's three nodes: _user as BAR0 (1 MiB of zeros), _h2c_0 a sparse
    file over both channels, _c2h_0 the same file (what a write puts there a read gets back)."""
    (d / f"{name}_user").write_bytes(bytes(1 << 20))
    h2c = d / f"{name}_h2c_0"
    h2c.write_bytes(b"")
    os.truncate(h2c, size)
    (d / f"{name}_c2h_0").symlink_to(h2c.name)
    return str(d / name)


def test_card_transport_picks_the_class_by_the_node(tmp_path):
    lp, xd = _nodes(tmp_path, "otpu0"), _nodes(tmp_path, "xdma5")
    t = B.card_transport(lp, dma=False)
    assert type(t) is B.LitePcieTransport and t.devname == "otpu0" and t._otpu_lock is None
    t.close()
    t = B.card_transport(xd, dma=False)
    assert type(t) is B.XdmaTransport and t.devname == "xdma5"
    t.close()
    t = B.card_transport(str(tmp_path / "xdma0"), dma=False)    # no xdma0: the card's otpu0
    assert type(t) is B.LitePcieTransport and t.dev == lp
    t.close()


def test_the_lock_is_the_resolved_nodes(tmp_path):
    _nodes(tmp_path, "otpu0")
    t = B.card_transport(str(tmp_path / "xdma0"))
    try:
        assert t._otpu_lock.name == "otpu0" and t.devname == "otpu0"
        with pytest.raises(rs.DeviceBusy):
            rs.DeviceLock("otpu0", wait=0)
    finally:
        t.close()
    rs.DeviceLock("otpu0", wait=0).release()


def test_otpu_lock_defaults_to_the_resolved_card(tmp_path, monkeypatch):
    out = tmp_path / "held"
    cmd = [sys.executable, "-c", "import os, sys; open(sys.argv[1], 'w').write("
           "os.environ.get('OTPU_LOCK_HELD', ''))", str(out)]
    for nodes, want in (((), "xdma0"), (("otpu0",), "otpu0")):
        _dev_nodes(monkeypatch, *nodes)
        assert rs.hold_main(["--wait", "0", "--", *cmd]) == 0
        assert out.read_text() == want


def test_litepcie_has_no_run_h2c_limit():
    assert B.LitePcieTransport.run_h2c == 0 and B.XdmaTransport.run_h2c == B.RUN_H2C
    assert B.LitePcieTransport.dma_align == B.DMA_ALIGN == 16 and B.XdmaTransport.dma_align == 1


def _calls(monkeypatch):
    """Every DMA call the transport makes: (kind, host address, bytes, card address)."""
    calls, pw, ri = [], os.pwrite, B._readinto

    def pwrite(fd, mv, off):
        calls.append(("w", np.frombuffer(mv, np.uint8).ctypes.data, len(mv), off))
        return pw(fd, mv, off)

    def readinto(fd, mv, off):
        calls.append(("r", np.frombuffer(mv, np.uint8).ctypes.data, len(mv), off))
        return ri(fd, mv, off)
    monkeypatch.setattr(B.os, "pwrite", pwrite)
    monkeypatch.setattr(B, "_readinto", readinto)
    return calls


def _off16(n: int, at: int = 8) -> np.ndarray:
    """n bytes of host memory `at` bytes past a 16-byte boundary, 1 KiB + `at` into a page: off
    the 16-byte grid, but placed well enough for XDMA's direct reads (board._read_ok) from a
    page-aligned card address."""
    return np.frombuffer(mmap.mmap(-1, n + 8192), np.uint8, n, 1024 + at)


@pytest.mark.parametrize("cls,name", [(B.LitePcieTransport, "otpuT"), (B.XdmaTransport, "xdmaT")])
def test_writes_during_a_run(tmp_path, monkeypatch, cls, name):
    """Board.write while a run is in flight: XDMA's in calls of RUN_H2C, LitePCIe's in one call
    per channel; LitePCIe's every call on the 16-byte grid (host buffer, length, card address),
    from a host buffer off it too (the staging buffer)."""
    t = cls(_nodes(tmp_path, name))
    calls = _calls(monkeypatch)
    try:
        b = B.Board(t, check=False, lock=False)
        data = np.random.default_rng(1).integers(0, 256, 1 << 20, dtype=np.uint8)
        b.in_run = True
        b.write(1 << 16, data)
        sizes = [n for k, _, n, _ in calls if k == "w"]
        if cls is B.LitePcieTransport:
            assert sizes == [1 << 19, 1 << 19]
            assert all(p % 16 == 0 and n % 16 == 0 and o % 16 == 0 for _, p, n, o in calls)
        else:
            assert max(sizes) <= B.RUN_H2C and sum(sizes) == len(data)
        b.in_run = False
        assert np.array_equal(b.read(1 << 16, len(data)), data)
        calls.clear()
        src = _off16(4096)
        src[:] = data[:4096]
        t.mem_write(1, 64, src)                                 # (host - card) % 64 == 8
        assert np.array_equal(t.mem_read(1, 64, 4096), src)
        if cls is B.LitePcieTransport:
            assert all(p % 16 == 0 and n % 16 == 0 and o % 16 == 0 for _, p, n, o in calls)
    finally:
        t.close()


@pytest.mark.parametrize("n", [4096, 256 << 10])
def test_litepcie_reads_into_a_buffer_off_the_grid_go_through_the_staging_buffer(
        tmp_path, monkeypatch, n):
    t = B.LitePcieTransport(_nodes(tmp_path, "otpuT"))
    try:
        ref = np.random.default_rng(n).integers(0, 256, n, dtype=np.uint8)
        t.mem_write(0, 4096, ref)
        calls = _calls(monkeypatch)
        out = _off16(n, 3)
        assert t.mem_read(0, 4096, n, out) is out and np.array_equal(out, ref)
        assert calls and all(p % 16 == 0 and k == "r" for k, p, _, _ in calls)
        assert all(p != out.ctypes.data for _, p, _, _ in calls)
        calls.clear()
        new = t.mem_read(0, 4096, n)                        # a buffer of its own: direct
        assert np.array_equal(new, ref) and [p for _, p, _, _ in calls] == [new.ctypes.data]
    finally:
        t.close()


def test_xdma_reads_into_any_buffer_directly(tmp_path, monkeypatch):
    """XDMA takes any host buffer: a small read into one off the grid goes straight into it, as
    before LitePCIe."""
    t = B.XdmaTransport(_nodes(tmp_path, "xdmaT"))
    try:
        t.mem_write(0, 0, np.arange(4096, dtype=np.uint32).view(np.uint8)[:4096])
        calls = _calls(monkeypatch)
        out = _off16(4096, 3)
        t.mem_read(0, 0, 4096, out)
        assert [p for _, p, _, _ in calls] == [out.ctypes.data]
    finally:
        t.close()


@pytest.mark.parametrize("off,n", [(70, 5), (64, 17), (100, 300), (4095, 1), (16, 32)])
def test_litepcie_reads_off_the_grid_are_widened(tmp_path, monkeypatch, off, n):
    t = B.LitePcieTransport(_nodes(tmp_path, "otpuT"))
    try:
        ref = np.random.default_rng(off).integers(0, 256, 8192, dtype=np.uint8)
        t.mem_write(1, 0, ref)
        calls = _calls(monkeypatch)
        assert np.array_equal(t.mem_read(1, off, n), ref[off:off + n])
        out = np.zeros(n, np.uint8)
        assert t.mem_read(1, off, n, out) is out and np.array_equal(out, ref[off:off + n])
        assert all(o % 16 == 0 and k % 16 == 0 and p % 16 == 0 for _, p, k, o in calls)
    finally:
        t.close()


# ------------------------------------------------------------------------------ the tools
def test_diag_names_the_driver_bound_to_the_card(tmp_path, monkeypatch):
    from opentpu.host import diag
    dev = _nodes(tmp_path, "otpu0")
    sysfs = tmp_path / "sys"
    sysfs.mkdir()
    (sysfs / "driver").symlink_to("../bus/pci/drivers/otpu_pcie")
    monkeypatch.setattr(diag, "card_sysfs", lambda d: sysfs)
    ok, msg = diag.driver_state(dev)
    assert ok and "the card bound to otpu_pcie" in msg and f"{dev}_c2h_0" in msg
    ok, msg = diag.driver_state(str(tmp_path / "otpu9"))
    assert not ok and "missing" in msg


# ------------------------------------------------------------------------------ setup_pcie.sh
SETUP = HOST / "setup_pcie.sh"


def _fn(name: str) -> str:
    """A shell function of setup_pcie.sh, from its `name() {` line to its closing `}` line."""
    lines = SETUP.read_text().splitlines()
    i = next(k for k, s in enumerate(lines) if s.startswith(f"{name}() {{"))
    return "\n".join(lines[i:next(k for k in range(i, len(lines)) if lines[k] == "}") + 1])


def test_setup_tells_the_bitstreams_apart_by_pci_id(tmp_path):
    """cards(): the XDMA bitstreams (10ee:7028, or subsystem 10ee:4f54) and the LitePCIe ones
    (10ee:5054); card_driver(): otpu_pcie for the LitePCIe ones, xdma for the others."""
    devs = {"0000:01:00.0": ("0x10ee", "0x5054", "0x4c50"),     # LitePCIe
            "0000:02:00.0": ("0x10ee", "0x7028", "0x4f54"),     # XDMA
            "0000:03:00.0": ("0x10ee", "0x9038", "0x4f54"),     # XDMA, another device ID
            "0000:04:00.0": ("0x8086", "0x5054", "0x4c50"),     # not Xilinx's
            "0000:05:00.0": ("0x10ee", "0x9031", "0x0007")}     # some other Xilinx device
    for bdf, (v, d, s) in devs.items():
        p = tmp_path / "devices" / bdf
        p.mkdir(parents=True)
        for f, x in (("vendor", v), ("device", d), ("subsystem_device", s)):
            (p / f).write_text(x + "\n")
    script = "\n".join(["set -euo pipefail", f"pci={tmp_path}", "LP_MOD=otpu_pcie",
                        _fn("cards"), _fn("card_driver"), _fn("node_class"),
                        'for b in $(cards); do echo "$b $(card_driver "$b") '
                        '$(node_class "$(card_driver "$b")")"; done'])
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)
    assert out.stdout.split("\n")[:-1] == ["0000:01:00.0 otpu_pcie otpu", "0000:02:00.0 xdma xdma",
                                           "0000:03:00.0 xdma xdma"]


def test_setup_names_the_litepcie_package_from_its_dkms_conf(tmp_path):
    """The LitePCIe driver's package and module from pcie/otpu/dkms.conf, and its version: for
    the template (@VERSION@) <MODULE_VERSION>.<the source's hash>, which a change to the source
    changes and a build product beside it does not; a literal version as given."""
    text = SETUP.read_text()
    block = text[text.index('lp_files="$files/otpu"'):text.index("lp_changed=0")]
    script = ("set -euo pipefail\nfiles=$1 moddir=/lib/modules/k\n" + block
              + 'echo "$LP_PKG $LP_VER $LP_MOD $lp_src_dir"')
    src = tmp_path / "otpu"
    src.mkdir()
    conf = ('PACKAGE_NAME="otpu-pcie"\nPACKAGE_VERSION="{}"\nBUILT_MODULE_NAME[0]="otpu_pcie"\n'
            'DEST_MODULE_LOCATION[0]="/updates"\nAUTOINSTALL="yes"\n')
    (src / "dkms.conf").write_text(conf.format("@VERSION@"))
    (src / "otpu_pcie.c").write_text('int x;\nMODULE_VERSION("7");\nMODULE_INFO(otpu, "otpu-pcie");\n')

    def names(files=tmp_path):
        return subprocess.run(["bash", "-c", script, "-", str(files)], capture_output=True,
                              text=True, check=True).stdout.split()
    pkg, ver, mod, usr = names()
    assert (pkg, mod) == ("otpu-pcie", "otpu_pcie") and re.fullmatch(r"7\.[0-9a-f]{8}", ver)
    assert usr == f"/usr/src/otpu-pcie-{ver}"
    (src / "otpu_pcie.o").write_bytes(b"\0" * 64)                  # a build product: the same
    (src / ".otpu_pcie.o.cmd").write_text("cmd")
    assert names()[1] == ver
    (src / "otpu_pcie.c").write_text('int y;\nMODULE_VERSION("7");\n')
    assert names()[1] != ver and names()[1].startswith("7.")
    (src / "dkms.conf").write_text(conf.format("0.3"))
    assert names()[1] == "0.3"
    assert names(tmp_path / "none")[:3:2] == ["otpu-pcie", "otpu_pcie"]   # (no source: defaults)


def test_setup_script_still_parses_and_documents_both_drivers():
    subprocess.run(["bash", "-n", str(SETUP)], check=True)
    r = subprocess.run(["bash", str(SETUP), "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "otpu_pcie" in r.stdout and "10ee:5054" in r.stdout


def test_udev_rules_open_the_data_nodes_of_both_drivers_and_no_others():
    """60-otpu.rules: user, h2c and c2h (and XDMA's events) for everyone, under either driver;
    XDMA's control, xvc and bypass nodes stay root-only."""
    pats = []
    for line in (HOST / "pcie" / "60-otpu.rules").read_text().splitlines():
        m = re.search(r'KERNEL=="([^"]+)"', line)
        if m and 'MODE="0666"' in line:
            pats += m[1].split("|")
    opened = lambda n: any(fnmatch.fnmatchcase(n, p) for p in pats)   # noqa: E731
    for n in ("otpu0_user", "otpu0_h2c_0", "otpu12_c2h_0", "xdma0_user", "xdma0_h2c_0",
              "xdma0_c2h_1", "xdma0_events_3"):
        assert opened(n), n
    for n in ("xdma0_control", "xdma0_xvc", "xdma0_bypass", "otpu0_control", "otpu0",
              "otpu_user"):
        assert not opened(n), n
