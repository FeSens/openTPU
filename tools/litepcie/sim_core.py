#!/usr/bin/env python3
"""The PCIe core's DMA against LitePCIe's host model, driven as opentpu/host/pcie/otpu/otpu_pcie.c
drives it (gen_core.py's core_logic and bar0_logic; docs/litepcie.md).

    ~/eda/litepcie-venv/bin/python tools/litepcie/sim_core.py [--seed 1] [--quick]

LitePCIe's host model (its test/model, from the checkout --litepcie-src: the root complex, host
memory, read completions reordered across tags; the completions are split here, at 64-byte
boundaries) takes the PHY's place. Host memory is a pool of
4 KiB pages above 4 GiB (64-bit TLPs) and the completion record's page. The card's side of
otpu_pcie_mm is a Python model: it takes the command queues' commands, the H2C stream into a card
memory and the C2H stream out of it with random stalls, and pulses h2c_done some cycles after a
command's last beat. The driver is a Python copy of otpu_pcie.c's register sequence, through
BAR0 (memory TLPs at csr.h's addresses, the bytes as an x86 host's iowrite32 puts them): the
DMA's init, transfers in commands of CHUNK bytes with up to INFLIGHT queued, each buffer on random
host pages (adjacent ones merged into one segment, as the kernel's scatterlists merge them),
descriptors split at 4 KiB boundaries, the waits on the record, a timeout's recovery.

Checks: the sim SoC's CSR map against the committed csr.csv; ID and CAPS (MMIO byte order); the
user window through the AXI-Lite master; H2C data in card memory when its count is reached; C2H
data in host memory when the record shows its count (the record's write behind the data); H2C and
C2H at once; an MSI after every record; timeouts (a lost h2c_done, an H2C stream the card stops
taking, a C2H stream it stops sending), the recovery, and transfers after each.
"""
import argparse
import csv
import os
import random
import re
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def _src_arg():
    for i, a in enumerate(sys.argv):
        if a == "--litepcie-src" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith("--litepcie-src="):
            return a.split("=", 1)[1]
    return os.path.expanduser("~/eda/src/litepcie")


sys.path.insert(0, _src_arg())      # test.model
sys.path.insert(0, HERE)            # gen_core

from migen import *                                     # noqa: E402,F401
from litex.gen import *                                 # noqa: E402,F401
from litex.gen.sim import passive, run_simulation       # noqa: E402
from litex.build.xilinx import XilinxPlatform           # noqa: E402
from litex.soc.interconnect import stream               # noqa: E402
from litex.soc.interconnect.axi import AXILiteSRAM      # noqa: E402
from litex.soc.interconnect.csr import CSRStatus       # noqa: E402
from litex.soc.integration.soc_core import SoCMini      # noqa: E402
from litepcie.common import msi_layout                  # noqa: E402

from test.model.host import Host                        # noqa: E402
from test.model.tlp import RD32, RD64, WR64, CPLD, CPL, parse_dwords  # noqa: E402

import gen_core as gc                                   # noqa: E402

# Migen's SyncFIFO (the pinned e19524c9) writes its storage through a port without a read data
# signal, which Migen's own simulator then fails to lower (MemoryToArray reads dat_r): give such
# ports an unused one
from migen.fhdl import simplify as _simplify            # noqa: E402
from migen.fhdl.specials import Memory as _Memory       # noqa: E402
_mta = _simplify.MemoryToArray.transform_fragment


def _mta_dat_r(self, ms, f):
    for sp in f.specials:
        if isinstance(sp, _Memory):
            for p in sp.ports:
                if p.dat_r is None:
                    p.dat_r = Signal(sp.width)
    return _mta(self, ms, f)


_simplify.MemoryToArray.transform_fragment = _mta_dat_r

ROOT_ID = 0x100
ENDPOINT_ID = 0x400
PAGE = 4096
HOST_BASE = 0x1_0000_0000           # host memory above 4 GiB: 64-bit TLPs
DESC_MAX = 8 << 20
DESC_IRQ_DIS = 1 << 24
DESC_LAST_DIS = 1 << 25
MSI_REC, MSI_ERR = 1, 2


def bswap(v):
    return int.from_bytes(v.to_bytes(4, "little"), "big")


def load_csr_h(path):
    regs = {}
    for ln in open(path):
        m = re.match(r"#define (\w+) (0x[0-9a-fA-F]+)L?\b", ln)
        if m:
            regs[m[1]] = int(m[2], 16)
    return regs


class SimSoC(SoCMini):
    """The core above the PHY as gen_core builds it, on the host model."""
    def __init__(self, platform):
        SoCMini.__init__(self, platform, clk_freq=250e6, csr_data_width=32, csr_ordering="big",
                         ident="openTPU LitePCIe", ident_version=False)
        host = Host(128, ROOT_ID, ENDPOINT_ID, bar0_size=gc.BAR0_SIZE,
                    chipset_split=True, chipset_reordering=True)
        phy = host.phy
        phy.endianness = "big"          # S7PCIEPHY's
        # S7PCIEPHY's CSRs, which the real core has twice: in its own region pcie_phy and, through
        # the endpoint's reference, in pcie_endpoint (whose region sets later regions' places)
        for n in ("link_status", "msi_enable", "msix_enable", "bus_master_enable",
                  "max_request_size", "max_payload_size"):
            setattr(phy, "_" + n, CSRStatus(32, name=n))
        self.pcie_phy = host            # the region pcie_phy, as the real core's
        phy.msi = stream.Endpoint(msi_layout())
        # pcie_endpoint, pcie_dma0, otpu_dma, pcie_msi; otpu_pcie_mm's streams past their slices
        _, self.user_ports, _ = gc.core_logic(self, phy)
        axi = gc.bar0_logic(self, self.pcie_endpoint, phy)
        self.user_ram = AXILiteSRAM(1024, bus=axi)


@passive
def root_generator(self):
    """The host model's chipset, with read completions kept for the MMIO reads (its own
    generator hands every received TLP to the host)."""
    while True:
        if self.en:
            dwords = self.phy.receive()
            if dwords is not None:
                msg = parse_dwords(dwords)
                if isinstance(msg, (RD64, WR64)):
                    # the model decodes a 64-bit address's low dword only (its header[2] after
                    # its swap; the high dword is header[3])
                    msg.address = msg.header[3] << 32 | msg.header[2]
                if os.environ.get("SIM_DEBUG") and not isinstance(msg, (CPLD, CPL)):
                    print(f"  TLP {type(msg).__name__} {getattr(msg, 'address', 0):#x} len {getattr(msg, 'length', 0)} tag {getattr(msg, 'tag', 0)}")
                if isinstance(msg, (CPLD, CPL)):
                    self.cpls.append(msg)
                else:
                    self.host_callback(msg)
            self.cmp_callback()
        yield


class Bench:
    def __init__(self, soc, regs, rng, a):
        self.soc, self.r, self.rng, self.a = soc, regs, rng, a
        self.host = soc.pcie_phy
        self.root = soc.pcie_phy.chipset
        self.root.cpls = []
        self.root.generator = root_generator.__get__(self.root)
        self.cycle = 0
        self.mmio_busy = False
        self.npages = a.pages
        self.rec = HOST_BASE + self.npages * PAGE
        self.pool = [HOST_BASE + i * PAGE for i in range(self.npages)]
        self.host.malloc(HOST_BASE, (self.npages + 1) * PAGE)
        self.rec_writes = []            # cycles of the record's writes
        self.msis = []                  # cycles of the MSIs
        rm = self.host.read_mem

        def read_mem(adr, length=1):
            if not HOST_BASE <= adr <= adr + length <= HOST_BASE + len(self.host.buffer) * 4:
                raise RuntimeError(f"cycle {self.cycle}: a read of host memory at {adr:#x}+{length}")
            return rm(adr, length)
        self.host.read_mem = read_mem
        cb = self.host.callback
        bench = self

        def callback(msg):
            cb(msg)
            if getattr(msg, "address", None) == bench.rec:
                bench.rec_writes.append(bench.cycle)
        self.host.chipset.set_host_callback(callback)
        # the card
        self.mem = {}                   # card byte address (16-aligned) -> 16 bytes
        self.stall_done = False         # the next H2C command's h2c_done never comes
        self.stall_h2c = False          # the card stops taking the H2C stream mid-command
        self.stall_c2h = False          # the card stops sending the C2H stream mid-command
        self.h2c = Dir(True, regs)
        self.c2h = Dir(False, regs)
        self.fails = []
        self.card_state = {}
        self.stats = {"h2c_bytes": 0, "c2h_bytes": 0, "descs": 0, "cmds": 0, "recoveries": 0}

    def check(self, ok, what):
        if not ok:
            self.fails.append(f"cycle {self.cycle}: {what}")
            print("FAIL", self.fails[-1])

    # host memory: the model's dwords hold the TLP's bytes, the first in bits 31:24
    def host_rd(self, addr, n):
        return b"".join(w.to_bytes(4, "big") for w in self.host.read_mem(addr, n))

    def host_wr(self, addr, data):
        self.host.write_mem(addr, [int.from_bytes(data[i:i + 4], "big")
                                   for i in range(0, len(data), 4)])

    def buf_io(self, pages, off, n, data=None):
        """A buffer at byte off of its pages: read n bytes, or write data."""
        out, pos = [], 0
        while pos < n:
            v = off + pos
            p, o = pages[v // PAGE], v % PAGE
            k = min(n - pos, PAGE - o)
            if data is None:
                out.append(self.host_rd(p + o, k))
            else:
                self.host_wr(p + o, data[pos:pos + k])
            pos += k
        return b"".join(out)

    def card_rd(self, a, n):
        return b"".join(self.mem.get(a + i, bytes(16)) for i in range(0, n, 16))

    # BAR0
    def wr(self, off, v):
        yield from self.root.wr32(off >> 2, [bswap(v)])

    def rd(self, off):
        while self.mmio_busy:
            yield
        self.mmio_busy = True
        rd = RD32()
        rd.fmt, rd.type, rd.length, rd.first_be, rd.last_be = 0, 0, 1, 0xF, 0
        rd.address, rd.requester_id, rd.tag = off, ROOT_ID, 0
        yield from self.host.phy.send_blocking(rd.encode_dwords())
        t = 0
        while not self.root.cpls:
            t += 1
            if t > 5000:
                self.check(False, f"MMIO read of {off:#x}: no completion")
                self.mmio_busy = False
                return 0xFFFFFFFF
            yield
        v = bswap(self.root.cpls.pop(0).data[0])
        self.mmio_busy = False
        return v

    def wait(self, n):
        for _ in range(n):
            yield

    # -- generators -------------------------------------------------------------------------
    @passive
    def clock(self):
        while True:
            self.cycle += 1
            yield

    @passive
    def host_reads(self):
        """The root complex's read completions (in place of the model's, which splits a read at
        any dword count): split at the read completion boundary (64 bytes) into pieces of 64,
        128 or 256 bytes, each with its lower address, as root complexes do; the chipset model
        reorders them across tags."""
        h, rng = self.host, self.rng
        while True:
            if h.rd_queue:
                msg = h.rd_queue.pop(0)
                addr, n = msg.address, msg.length * 4
                data = h.read_mem(addr, n)
                pos = 0
                while pos < n:
                    a = addr + pos
                    k = min(n - pos, rng.choice([64, 128, 256]) - a % 64)
                    h.chipset.cmp(msg.requester_id, data[pos // 4:(pos + k) // 4], byte_count=n - pos,
                                  lower_address=a & 0x7F, tag=msg.tag)
                    pos += k
            yield

    @passive
    def msi(self):
        m = self.host.phy.msi
        yield m.ready.eq(1)
        while True:
            if (yield m.valid):
                self.msis.append(self.cycle)
            yield

    @passive
    def card_h2c(self):
        od, s, rng = self.soc.otpu_dma, self.soc.user_ports["h2c_dat"], self.rng
        cmd = self.soc.user_ports["h2c_cmd"]
        q, cur, dones, last_due, stuck_done = [], None, [], 0, False
        while True:
            if (yield od.mm_rst):
                q, cur, dones, stuck_done = [], None, [], False
            else:
                if (yield cmd.valid) and (yield cmd.ready):
                    c = [(yield cmd.addr), (yield cmd.len), (yield cmd.flags), 0,
                         False, False]
                    self.check(c[2] == 0 and c[1] % 16 == 0 and c[0] % 16 == 0, f"H2C command {c}")
                    if self.stall_done:
                        c[4], self.stall_done = True, False
                    if self.stall_h2c:
                        c[5], self.stall_h2c = True, False
                    q.append(c)
                if cur is not None and (yield s.valid) and (yield s.ready):
                    self.mem[cur[0] + cur[3]] = (yield s.data).to_bytes(16, "little")
                    cur[3] += 16
                    if cur[3] == cur[1]:
                        stuck_done = stuck_done or cur[4]
                        if not stuck_done:
                            last_due = max(last_due + 1, self.cycle + rng.randint(4, 80))
                            dones.append(last_due)
                        cur = None
            pulse = bool(dones) and dones[0] <= self.cycle
            if pulse:
                dones.pop(0)
            yield od.h2c_done.eq(pulse)
            if cur is None and q:
                cur = q.pop(0)
            stuck = cur is not None and cur[5] and cur[3] >= cur[1] // 2
            self.card_state = {"h2c_cur": cur and cur[:4], "h2c_q": len(q), "dones": len(dones)}
            yield s.ready.eq(cur is not None and not stuck and rng.random() < 0.85)
            yield cmd.ready.eq(len(q) < 2)
            yield

    @passive
    def card_c2h(self):
        od, k, rng = self.soc.otpu_dma, self.soc.user_ports["c2h_dat"], self.rng
        cmd = self.soc.user_ports["c2h_cmd"]
        q, cur, valid = [], None, False
        while True:
            if (yield od.mm_rst):
                q, cur, valid = [], None, False
            else:
                if (yield cmd.valid) and (yield cmd.ready):
                    c = [(yield cmd.addr), (yield cmd.len), (yield cmd.flags), 0,
                         False]
                    self.check(c[2] == 0 and c[1] % 16 == 0 and c[0] % 16 == 0, f"C2H command {c}")
                    if self.stall_c2h:
                        c[4], self.stall_c2h = True, False
                    q.append(c)
                if valid and (yield k.ready):
                    cur[3] += 16
                    valid = False
                    if cur[3] == cur[1]:
                        cur = None
            if cur is None and q:
                cur = q.pop(0)
            stuck = cur is not None and cur[4] and cur[3] >= cur[1] // 2
            if not valid and cur is not None and not stuck and rng.random() < 0.85:
                valid = True
            if valid:
                yield k.data.eq(int.from_bytes(self.card_rd(cur[0] + cur[3], 16), "little"))
                yield k.last.eq(cur[3] + 16 == cur[1])
            yield k.valid.eq(valid)
            yield cmd.ready.eq(len(q) < 2)
            yield

    # -- the driver (otpu_pcie.c) -----------------------------------------------------------
    def dma_init(self):
        r, wr = self.r, self.wr
        for k, v in (("CSR_PCIE_DMA0_READER_ENABLE_ADDR", 0), ("CSR_PCIE_DMA0_WRITER_ENABLE_ADDR", 0),
                     ("CSR_OTPU_DMA_WB_CTRL_ADDR", 0), ("CSR_OTPU_DMA_MM_CTRL_ADDR", 1),
                     ("CSR_PCIE_DMA0_READER_TABLE_LOOP_PROG_N_ADDR", 0),
                     ("CSR_PCIE_DMA0_WRITER_TABLE_LOOP_PROG_N_ADDR", 0),
                     ("CSR_PCIE_DMA0_READER_TABLE_RESET_ADDR", 1),
                     ("CSR_PCIE_DMA0_WRITER_TABLE_RESET_ADDR", 1)):
            yield from wr(r[k], v)
        yield from self.rd(r["CSR_OTPU_DMA_ID_ADDR"])
        yield from self.wait(50)
        yield from wr(r["CSR_OTPU_DMA_MM_CTRL_ADDR"], 0)
        self.h2c.issued = yield from self.rd(r["CSR_OTPU_DMA_H2C_COUNT_ADDR"])
        self.c2h.issued = yield from self.rd(r["CSR_OTPU_DMA_C2H_COUNT_ADDR"])
        self.host_wr(self.rec, struct.pack("<2I", self.h2c.issued, self.c2h.issued))
        yield from wr(r["CSR_OTPU_DMA_WB_LO_ADDR"], self.rec & 0xFFFFFFFF)
        yield from wr(r["CSR_OTPU_DMA_WB_HI_ADDR"], self.rec >> 32)
        yield from wr(r["CSR_OTPU_DMA_WB_CTRL_ADDR"], 1)
        yield from wr(r["CSR_PCIE_DMA0_READER_ENABLE_ADDR"], 1)
        yield from wr(r["CSR_PCIE_DMA0_WRITER_ENABLE_ADDR"], 1)
        yield from self.rd(r["CSR_OTPU_DMA_ID_ADDR"])

    def dma_off(self):
        r = self.r
        for k in ("CSR_PCIE_DMA0_READER_ENABLE_ADDR", "CSR_PCIE_DMA0_WRITER_ENABLE_ADDR",
                  "CSR_OTPU_DMA_WB_CTRL_ADDR", "CSR_PCIE_MSI_ENABLE_ADDR"):
            yield from self.wr(r[k], 0)
        yield from self.rd(r["CSR_OTPU_DMA_ID_ADDR"])

    def recover(self, q):
        self.stats["recoveries"] += 1
        yield from self.dma_off()
        yield from self.wait(400)           # msleep(2)
        yield from self.dma_init()
        yield from self.wr(self.r["CSR_PCIE_MSI_ENABLE_ADDR"], MSI_REC | MSI_ERR)

    def push_desc(self, q, addr, ln, irq):
        if os.environ.get("SIM_DEBUG"):
            print(f"  {'H2C' if q.h2c else 'C2H'} desc {addr:#x} {ln} irq {int(irq)} @{self.cycle}")
        yield from self.wr(q.value, ln | (0 if irq else DESC_IRQ_DIS) | DESC_LAST_DIS)
        yield from self.wr(q.value + 4, addr & 0xFFFFFFFF)
        yield from self.wr(q.we, addr >> 32)

    def segments(self, pages, off, n):
        """The scatterlist of a buffer's range: physically adjacent pages merged."""
        segs, pos = [], 0
        while pos < n:
            v = off + pos
            a = pages[v // PAGE] + v % PAGE
            k = min(n - pos, PAGE - v % PAGE)
            if segs and segs[-1][0] + segs[-1][1] == a and segs[-1][1] + k <= DESC_MAX:
                segs[-1][1] += k
            else:
                segs.append([a, k])
            pos += k
        return segs

    def cmd_descs(self, q, segs):
        descs = []
        for a, ln in segs:
            while ln:
                k = min(ln, PAGE - (a & (PAGE - 1))) if a & (PAGE - 1) else min(ln, DESC_MAX)
                descs.append((a, k))
                a += k
                ln -= k
        for i, (a, k) in enumerate(descs):
            yield from self.push_desc(q, a, k, irq=(not q.h2c and i == len(descs) - 1))
        return len(descs)

    def cmd_issue(self, q, pages, off, n, card):
        segs = self.segments(pages, off, n)
        if q.h2c:
            yield from self.wr(q.cmd_addr, card)
            yield from self.wr(q.cmd_len, n)
            nd = yield from self.cmd_descs(q, segs)
        else:
            nd = yield from self.cmd_descs(q, segs)
            yield from self.wr(q.cmd_addr, card)
            yield from self.wr(q.cmd_len, n)
        q.table += nd
        q.issued = (q.issued + 1) & 0xFFFFFFFF
        q.cmds.append((q.issued, nd, pages, off, n, card))
        self.stats["descs"] += nd
        self.stats["cmds"] += 1

    def reached(self, q, target):
        h, c = struct.unpack("<2I", self.host_rd(self.rec, 8))
        cnt = h if q.h2c else c
        return ((cnt - target) & 0xFFFFFFFF) < 0x80000000

    def cmd_wait(self, q, target):
        t = 0
        while not self.reached(q, target):
            t += 1
            if t > self.a.timeout:
                return False
            yield
        return True

    def transfer(self, q, pages, off, n, card, expect_timeout=False):
        """otpu_dma(): returns True when done, False on a timeout (after the recovery)."""
        a = self.a
        done = 0
        while done < n or q.cmds:
            if done < n and len(q.cmds) < a.inflight:
                k = min(n - done, a.chunk)
                worst = (off + done + k - 1) // PAGE - (off + done) // PAGE + 2
                if q.table + worst <= a.table:
                    yield from self.cmd_issue(q, pages, off + done, k, card + done)
                    done += k
                    continue
            target, nd, cp, co, cn, cc = q.cmds[0]
            ok = yield from self.cmd_wait(q, target)
            if not ok:
                if not expect_timeout:
                    r = self.r
                    st = {}
                    for k in ("CSR_OTPU_DMA_H2C_COUNT_ADDR", "CSR_OTPU_DMA_C2H_COUNT_ADDR",
                              "CSR_OTPU_DMA_QUEUE_ADDR", "CSR_PCIE_DMA0_READER_TABLE_LEVEL_ADDR",
                              "CSR_PCIE_DMA0_WRITER_TABLE_LEVEL_ADDR",
                              "CSR_PCIE_DMA0_READER_TABLE_LOOP_STATUS_ADDR"):
                        st[k[13:-5]] = hex((yield from self.rd(r[k])))
                    print("  state", st, "record", struct.unpack("<4I", self.host_rd(self.rec, 16)),
                          "card", self.card_state, "cmds", [(c[0], c[1], c[4], hex(c[5])) for c in q.cmds])
                yield from self.recover(q)
                q.cmds, q.table = [], 0
                return False
            # the command's data where its count says it is, at the cycle the count shows
            if q.h2c:
                self.check(self.card_rd(cc, cn) == self.buf_io(cp, co, cn),
                           f"H2C {cn} bytes to {cc:#x}: card memory differs at count {target}")
            else:
                self.check(self.buf_io(cp, co, cn) == self.card_rd(cc, cn),
                           f"C2H {cn} bytes from {cc:#x}: host memory differs at record {target}")
            q.table -= nd
            q.cmds.pop(0)
        self.check(not expect_timeout, f"{'H2C' if q.h2c else 'C2H'} transfer expected to time out")
        self.stats["h2c_bytes" if q.h2c else "c2h_bytes"] += n
        return True

    # -- scenarios --------------------------------------------------------------------------
    def buffer(self, n, off, pool):
        """Pages for a buffer of n bytes at byte off: from pool, scattered or in runs."""
        np_ = (off + n - 1) // PAGE + 1
        rng = self.rng
        if rng.random() < 0.4:
            start = rng.randrange(0, len(pool) - np_ + 1)
            return pool[start:start + np_]
        return rng.sample(pool, np_)

    def h2c_c2h(self, n, card, pool, off=None):
        rng = self.rng
        off = rng.randrange(0, PAGE, 16) if off is None else off
        pages = self.buffer(n, off, pool)
        data = rng.randbytes(n)
        self.buf_io(pages, off, n, data)
        ok = yield from self.transfer(self.h2c, pages, off, n, card)
        self.check(ok, "H2C failed")
        off2 = rng.randrange(0, PAGE, 16)
        pages2 = self.buffer(n, off2, pool if len(pool) > 8 else self.pool)
        self.buf_io(pages2, off2, n, bytes(n))
        ok = yield from self.transfer(self.c2h, pages2, off2, n, card)
        self.check(ok and self.buf_io(pages2, off2, n) == data, f"round trip {n} bytes at {card:#x}")

    def main(self):
        r, rng, a = self.r, self.rng, self.a
        self.root.enable()
        yield from self.wait(20)
        ident = yield from self.rd(r["CSR_OTPU_DMA_ID_ADDR"])
        caps = yield from self.rd(r["CSR_OTPU_DMA_CAPS_ADDR"])
        self.check(ident == r["OTLP_ID"], f"ID {ident:#x}")
        self.check(caps & 0xFF == r["OTLP_VERSION"] and (caps >> 8) & 0xFF == 10
                   and (caps >> 16) & 0xFF == gc.PENDING, f"CAPS {caps:#x}")
        # the user window
        yield from self.wr(0x100, 0x12345678)
        yield from self.wr(0x104, 0x9ABCDEF0)
        v0 = yield from self.rd(0x100)
        v1 = yield from self.rd(0x104)
        self.check((v0, v1) == (0x12345678, 0x9ABCDEF0), f"user window read back {v0:#x} {v1:#x}")
        # the driver's probe
        yield from self.dma_init()
        yield from self.wr(r["CSR_PCIE_MSI_CLEAR_ADDR"], MSI_REC | MSI_ERR)
        yield from self.wr(r["CSR_PCIE_MSI_ENABLE_ADDR"], MSI_REC | MSI_ERR)
        pool = [HOST_BASE + i * PAGE for i in range(self.npages)]

        print("sequential")
        sizes = [16, 48, 4096, 4096 + 32, a.chunk, 2 * a.chunk + 16 * rng.randrange(1, 64)]
        if not a.quick:
            sizes += [rng.randrange(16, 3 * a.chunk, 16) for _ in range(4)] + [5 * a.chunk]
        for i, n in enumerate(sizes):
            card = (i % 2) << 31 | rng.randrange(0, 1 << 20, 16)
            yield from self.h2c_c2h(n, card, pool)
        # page-aligned buffers (one descriptor per contiguous run)
        yield from self.h2c_c2h(3 * PAGE, 0x40000, pool[:3], off=0)

        print("concurrent")
        self.conc = 2
        return

    def worker_h2c(self, pool):
        rng = self.rng
        while not hasattr(self, "conc"):
            yield
        for i in range(2 if self.a.quick else 5):
            n = rng.randrange(16, 2 * self.a.chunk, 16)
            off = rng.randrange(0, PAGE, 16)
            pages = self.buffer(n, off, pool)
            data = rng.randbytes(n)
            self.buf_io(pages, off, n, data)
            ok = yield from self.transfer(self.h2c, pages, off, n, 0x100000 + i * 0x10000)
            self.check(ok, "concurrent H2C")
        self.conc -= 1

    def worker_c2h(self, pool):
        rng = self.rng
        while not hasattr(self, "conc"):
            yield
        for i in range(2 if self.a.quick else 5):
            n = rng.randrange(16, 2 * self.a.chunk, 16)
            card = 0x80100000 + i * 0x10000
            for b in range(0, n, 16):
                self.mem[card + b] = rng.randbytes(16)
            off = rng.randrange(0, PAGE, 16)
            pages = self.buffer(n, off, pool)
            ok = yield from self.transfer(self.c2h, pages, off, n, card)
            self.check(ok and self.buf_io(pages, off, n) == self.card_rd(card, n), "concurrent C2H")
        self.conc -= 1

    def tail(self):
        """After the concurrent phase: the timeouts and their recoveries."""
        rng, a = self.rng, self.a
        while not hasattr(self, "conc") or self.conc:
            yield
        pool = [HOST_BASE + i * PAGE for i in range(self.npages)]
        for what in ("stall_done", "stall_h2c", "stall_c2h"):
            print(what)
            n = 3 * a.chunk
            off = rng.randrange(0, PAGE, 16)
            pages = self.buffer(n, off, pool)
            self.buf_io(pages, off, n, rng.randbytes(n))
            setattr(self, what, True)
            q = self.c2h if what == "stall_c2h" else self.h2c
            ok = yield from self.transfer(q, pages, off, n, 0x200000, expect_timeout=True)
            self.check(not ok, f"{what}: no timeout")
            # and the DMA works after the recovery
            yield from self.h2c_c2h(a.chunk + 16 * rng.randrange(1, 64), 0x300000, pool)
            yield from self.h2c_c2h(48, 0x80300000, pool)
        yield from self.wait(300)
        # every record has an MSI at or after it
        for t in self.rec_writes:
            self.check(any(t <= m <= t + 200 for m in self.msis), f"record at cycle {t}: no MSI")
        self.done = True


class Dir:
    def __init__(self, h2c, r):
        self.h2c = h2c
        self.issued = 0
        self.cmds = []
        self.table = 0
        p = "H2C" if h2c else "C2H"
        t = "READER" if h2c else "WRITER"
        self.cmd_addr = r[f"CSR_OTPU_DMA_{p}_ADDR_ADDR"]
        self.cmd_len = r[f"CSR_OTPU_DMA_{p}_LEN_ADDR"]
        self.value = r[f"CSR_PCIE_DMA0_{t}_TABLE_VALUE_ADDR"]
        self.we = r[f"CSR_PCIE_DMA0_{t}_TABLE_WE_ADDR"]


def check_csr_map(soc, path):
    """The sim SoC's CSR addresses against the committed core's csr.csv (the sim has no PHY
    CSRs: pcie_phy, the last region, is left out)."""
    from litex.soc.integration.export import get_csr_csv

    def regs(rows):
        return {r[1]: int(r[2], 16) for r in rows if r and r[0] == "csr_register"}
    want = regs(csv.reader(open(path)))
    got = regs(csv.reader(get_csr_csv(soc, soc.csr_regions).splitlines()))
    # the PHY's CSRs through the endpoint take the model's path in their names
    got = {re.sub(r"^pcie_endpoint_.*?phy_(?!phy_)", "pcie_endpoint_phy_", k): v for k, v in got.items()}
    bad = [k for k in want if not k.startswith("pcie_phy") and got.get(k) != want[k]]
    return bad, len(want)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--quick", action="store_true", help="fewer and shorter transfers")
    ap.add_argument("--chunk", type=int, default=8192, help="bytes a command (the driver's CHUNK)")
    ap.add_argument("--inflight", type=int, default=4, help="commands queued per direction")
    ap.add_argument("--table", type=int, default=12,
                    help="descriptors the driver keeps in a table (1024 on the card)")
    ap.add_argument("--pages", type=int, default=96, help="host memory pages")
    ap.add_argument("--timeout", type=int, default=6000, help="cycles a command's wait")
    ap.add_argument("--litepcie-src", default=os.path.expanduser("~/eda/src/litepcie"),
                    help="LitePCIe's checkout (its test/model)")
    ap.add_argument("--vcd", help="write a VCD")
    a = ap.parse_args()
    random.seed(a.seed)                 # the host model's completion splits and reordering
    rng = random.Random(a.seed)
    platform = XilinxPlatform("xc7k480tffg1156-2", io=[], toolchain="vivado")
    soc = SimSoC(platform)
    regs = load_csr_h(os.path.join(REPO, "opentpu/host/pcie/otpu/csr.h"))
    b = Bench(soc, regs, rng, a)
    pool = [HOST_BASE + i * PAGE for i in range(b.npages)]
    half = len(pool) // 2
    gens = [b.main(), b.worker_h2c(pool[:half]), b.worker_c2h(pool[half:]), b.tail(),
            b.clock(), b.msi(), b.card_h2c(), b.card_c2h(),
            b.host_reads(), b.root.generator(), b.host.phy.phy_sink.generator(),
            b.host.phy.phy_source.generator()]
    run_simulation(soc, {"sys": gens}, {"sys": 10}, vcd_name=a.vcd)
    bad, nwant = check_csr_map(soc, os.path.join(REPO, "opentpu/host/litepcie/csr.csv"))
    b.check(not bad, f"CSR map differs from csr.csv: {bad}")
    b.check(getattr(b, "done", False), "the scenarios did not finish")
    print(f"cycles {b.cycle}, {b.stats}, records {len(b.rec_writes)}, MSIs {len(b.msis)}, "
          f"CSRs checked {nwant}")
    if b.fails:
        print(f"FAILED ({len(b.fails)})")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
