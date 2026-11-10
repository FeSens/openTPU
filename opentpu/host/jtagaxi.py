"""A register transport over JTAG for the BCU1525's memory-test image (docs/bcu1525.md section 14),
which has no PCIe: its JTAG-to-AXI master (boards/bcu1525/vivado/bd_mem.tcl) at the full image's
BAR0 offsets, driven by one Vivado kept open in batch mode on jtagaxi.tcl (beside this file) through
the hardware server. reg_read / reg_write / reg_read_many as the PCIe transports', so ddr4, ddrcal,
bmc and gov work on it unchanged; and what only JTAG reaches: the SYSMON without the fabric (mon),
DONE, programming, and boot (JPROGRAM: the configuration from the flash, the card's way out of a
design that does not answer). One command at a time (a lock: the guard's thread shares it).

The full image (bd.tcl) has a second, AXI4 master on XDMA's DMA interconnect: mem_read / mem_write
reach the card's DRAM through it at the DMA addresses (it needs XDMA's clock and reset, the PCIe
link up), and JtagTransport is opentpu.host.board's transport over the two (otpu-selftest --jtag).
"""
from __future__ import annotations

import contextlib
import os
import queue
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path

TCL = Path(__file__).with_name("jtagaxi.tcl")
HW_SERVER_PORT = 3121
CHUNK = 64                      # words per run_hw_axi
BURST = 256                     # words per AXI4 burst (32-bit data: 1 KiB; AXI4's INCR limit)
MEM_CHUNK = 4096                # words per mr / mw command (16 KiB)
PRIO_WAIT_S = 30.0              # the longest a DRAM transfer waits on a priority section


def bursts(addr: int, nwords: int, burst: int = BURST) -> list[tuple[int, int]]:
    """(address, words) for nwords 32-bit words from addr: at most `burst` words each, none across
    a 4 KiB boundary (AXI's rule)."""
    if addr % 4:
        raise ValueError(f"address {addr:#x} not 32-bit aligned")
    out = []
    while nwords > 0:
        n = min(burst, nwords, (4096 - addr % 4096) // 4)
        out.append((addr, n))
        addr, nwords = addr + 4 * n, nwords - n
    return out


class JtagAxiError(OSError):
    pass


def _hw_server_up(port: int = HW_SERVER_PORT) -> bool:
    try:
        socket.create_connection(("127.0.0.1", port), timeout=1).close()
        return True
    except OSError:
        return False


def start_hw_server(hw_server: str | None = None, log: Path | None = None) -> None:
    """The hardware server on localhost:3121, unless one listens there."""
    if _hw_server_up():
        return
    exe = hw_server or shutil.which("hw_server") or os.path.expanduser(
        "~/AMD/2026.1/Vivado/bin/hw_server")
    out = open(log, "ab") if log else subprocess.DEVNULL
    subprocess.Popen([exe, "-s", f"TCP:127.0.0.1:{HW_SERVER_PORT}", "-p0", "-I600"],
                     stdout=out, stderr=out, stdin=subprocess.DEVNULL, start_new_session=True)
    for _ in range(100):
        if _hw_server_up():
            return
        time.sleep(0.2)
    raise JtagAxiError("hw_server did not come up on 127.0.0.1:3121")


class JtagAxi:
    """One card by its FT4232H's serial (the JTAG target's name holds it)."""

    def __init__(self, serial: str, vivado: str = os.environ.get("OTPU_VIVADO") or "~/.local/bin/vivado", log: Path | None = None,
                 timeout: float = 30.0):
        self.serial, self.timeout = serial, timeout
        self.devname = f"bcu1525-{serial}"
        self.lock = threading.Lock()
        # priority sections (the guard's regulator reading, tens of small commands): the DRAM
        # transfers wait between their mr / mw commands while one is open (2026-10-26 full4:
        # the regulator went unread for 60 s behind run_demo's 7.7 MB, the guard SAFE on it)
        self.prio, self.prio_cv = 0, threading.Condition()
        start_hw_server(log=log.with_suffix(".hw_server.log") if log else None)
        self.log = open(log, "a") if log else None
        self.p = subprocess.Popen(
            [os.path.expanduser(vivado), "-mode", "batch", "-nolog", "-nojournal", "-notrace",
             "-source", str(TCL)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            bufsize=1, start_new_session=True)
        self.q: queue.Queue[str | None] = queue.Queue()
        self.late = 0                               # answers still to come of commands timed out
        threading.Thread(target=self._reader, daemon=True).start()
        self._answer(120.0)                         # READY: Vivado and the hardware manager up
        self.target = self.cmd(f"open {serial}", 60.0)

    def _reader(self) -> None:
        for line in self.p.stdout:
            line = line.rstrip("\n")
            if self.log:
                self.log.write(line + "\n")
                self.log.flush()
            if line.startswith("@@ "):
                self.q.put(line[3:])
        self.q.put(None)

    def _answer(self, timeout: float) -> list[str]:
        """The answer to the command just sent. One that timed out still gets its answer later
        (Vivado answers every line, in order): `late` counts those, skipped here, so that a
        command never takes an earlier one's answer."""
        deadline = time.monotonic() + timeout
        while True:
            try:
                a = self.q.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                self.late += 1
                raise JtagAxiError(f"JTAG: no answer in {timeout:.0f} s") from None
            if a is None:
                raise JtagAxiError("JTAG: Vivado exited")
            if not self.late:
                break
            self.late -= 1
        w = a.split()
        if w[0] != "OK":
            raise JtagAxiError("JTAG: " + a)
        return w[1:]

    def cmd(self, line: str, timeout: float | None = None) -> list[str]:
        with self.lock:
            if self.p.poll() is not None:
                raise JtagAxiError("JTAG: Vivado exited")
            self.p.stdin.write(line + "\n")
            self.p.stdin.flush()
            return self._answer(timeout or self.timeout)

    @contextlib.contextmanager
    def priority(self):
        """A section whose commands go before the DRAM transfers': mem_read / mem_write wait
        between their commands while one is open (up to PRIO_WAIT_S each)."""
        with self.prio_cv:
            self.prio += 1
        try:
            yield
        finally:
            with self.prio_cv:
                self.prio -= 1
                self.prio_cv.notify_all()

    def _bulk_turn(self) -> None:
        """A DRAM transfer's next command: once no priority section is open, or after
        PRIO_WAIT_S (a section that hangs holds no transfer for ever)."""
        with self.prio_cv:
            self.prio_cv.wait_for(lambda: not self.prio, PRIO_WAIT_S)

    # ---- the transport
    def reg_read(self, a: int) -> int:
        return int(self.cmd(f"r {a:x}")[0], 16)

    def reg_write(self, a: int, v: int) -> None:
        self.cmd(f"w {a:x} {v & 0xFFFFFFFF:x}")

    def reg_read_many(self, addrs) -> list[int]:
        addrs, out = list(addrs), []
        for i in range(0, len(addrs), CHUNK):
            out += [int(x, 16) for x in self.cmd("rm " + " ".join(f"{a:x}" for a in addrs[i:i + CHUNK]))]
        return out

    def reg_write_many(self, pairs) -> None:
        pairs = list(pairs)
        for i in range(0, len(pairs), CHUNK):
            self.cmd("wm " + " ".join(f"{a:x} {v & 0xFFFFFFFF:x}" for a, v in pairs[i:i + CHUNK]))

    # ---- the card's DRAM (the full image's AXI4 master)
    def axis(self) -> dict:
        """The JTAG-to-AXI cores: {"lite", "mem", "lite_protocol", "mem_protocol"} ("-": none)."""
        return dict(zip(("lite", "mem", "lite_protocol", "mem_protocol"), self.cmd("axis") + ["-"] * 4))

    def set_order(self, lsb_right: bool) -> None:
        """A burst's DATA with its first beat on the right (True, the default) or the left."""
        self.cmd(f"order {int(lsb_right)}")

    def mem_read(self, addr: int, n: int) -> bytes:
        """n bytes (a multiple of 4) at the DMA address addr (32-bit aligned)."""
        if n % 4:
            raise ValueError(f"{n} bytes: not whole 32-bit words")
        out = []
        bl = bursts(addr, n // 4)
        i = 0
        while i < len(bl):
            j, words = i, 0
            while j < len(bl) and words + bl[j][1] <= MEM_CHUNK:
                words += bl[j][1]
                j += 1
            self._bulk_turn()
            r = self.cmd("mr " + " ".join(f"{a:016x} {k}" for a, k in bl[i:j]), 120.0)
            if len(r) != words:
                raise JtagAxiError(f"JTAG: {len(r)} words read for {words}")
            out += r
            i = j
        return b"".join(int(w, 16).to_bytes(4, "little") for w in out)

    def mem_write(self, addr: int, data: bytes) -> None:
        """data (a multiple of 4 bytes) to the DMA address addr (32-bit aligned)."""
        if len(data) % 4:
            raise ValueError(f"{len(data)} bytes: not whole 32-bit words")
        words = [int.from_bytes(data[k:k + 4], "little") for k in range(0, len(data), 4)]
        bl = bursts(addr, len(words))
        i = pos = 0
        while i < len(bl):
            j, n, parts = i, 0, []
            while j < len(bl) and n + bl[j][1] <= MEM_CHUNK:
                a, k = bl[j]
                parts.append(f"{a:016x} {k} " + " ".join(f"{w:08x}" for w in words[pos + n:pos + n + k]))
                n += k
                j += 1
            self._bulk_turn()
            self.cmd("mw " + " ".join(parts), 120.0)
            pos += n
            i = j

    # ---- JTAG only
    def mon(self) -> dict:
        """The SYSMON over JTAG (no fabric): {"t", "vccint", "vccaux", "vccbram"} (C, V)."""
        t, vi, va, vb = (float(x) for x in self.cmd("mon"))
        return {"t": t, "vccint": vi, "vccaux": va, "vccbram": vb}

    def migs(self, ltx: str | Path, out: str | Path) -> dict:
        """Every DDR4 MIG core's status properties by its cell ({cell: {property: value}}), the
        full report_property of each (after refresh_hw_mig) into `out` (the probes file `ltx`
        names the cores; the debug hub needs its clock)."""
        w = self.cmd(f"migs {Path(ltx).resolve()} {Path(out).resolve()}", 300.0)
        cores = {}
        for e in w[1:1 + int(w[0])]:
            name, _, props = e.partition("|")
            cores[name] = dict(kv.split("=", 1) for kv in props.split(",") if "=" in kv)
        return cores

    def done(self) -> list[int]:
        return [int(x) for x in self.cmd("done")]

    def program(self, bit: str | Path) -> int:
        """Volatile; returns the JTAG-to-AXI cores found after it."""
        return int(self.cmd(f"prog {Path(bit).resolve()}", 600.0)[0])

    def boot(self) -> None:
        """boot_hw_device: JPROGRAM, the flash's configuration (the factory image)."""
        self.cmd("boot", 120.0)

    def close(self) -> None:
        try:
            self.cmd("quit", 30.0)
        except OSError:
            pass
        try:
            self.p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.p.kill()
        if self.log:
            self.log.close()


class JtagTransport:
    """opentpu.host.board's transport over JTAG: the full image's registers through JtagAxi's AXI-Lite
    master and its DRAM through the AXI4 one, at a few hundred KB/s. For bring-up with the bitstream
    loaded over JTAG and the PCIe link not enumerated (otpu-selftest --jtag). The JtagAxi is the
    caller's (opened, programmed, the governor kicked by a gov.Guard on it); close() leaves it open.
    The UDIMMs have no ECC (no scrub); DMA calls run in the caller's thread."""
    ecc = False
    threaded = False
    streams = False
    host_idle = 1e-3

    def __init__(self, jt: JtagAxi):
        from .board import BASE
        self.jt, self.devname, self.dev = jt, jt.devname, None
        self.bases = BASE
        self._otpu_lock = None

    def mem_write(self, ch: int, off: int, data) -> None:
        """Whole 64-byte beats, as XdmaTransport's: a range off them is widened, its edge beats
        read and merged here."""
        import numpy as np
        from .board import BEAT
        a = np.ascontiguousarray(data, np.uint8).reshape(-1)
        if len(a) == 0:
            return
        if off % BEAT or len(a) % BEAT:
            o0, o1 = off // BEAT * BEAT, -(-(off + len(a)) // BEAT) * BEAT
            buf = np.empty(o1 - o0, np.uint8)
            buf[:BEAT] = self.mem_read(ch, o0, BEAT)
            buf[-BEAT:] = self.mem_read(ch, o1 - BEAT, BEAT)
            buf[off - o0:off - o0 + len(a)] = a
            a, off = buf, o0
        self.jt.mem_write(self.bases[ch] + off, a.tobytes())

    def mem_read(self, ch: int, off: int, n: int, out=None):
        import numpy as np
        o0, o1 = off // 4 * 4, -(-(off + n) // 4) * 4
        b = np.frombuffer(self.jt.mem_read(self.bases[ch] + o0, o1 - o0), np.uint8)
        b = b[off - o0:off - o0 + n]
        if out is None:
            return b.copy()
        out[:n] = b
        return out

    def reg_write(self, off: int, val: int) -> None:
        self.jt.reg_write(off, val)

    def reg_read(self, off: int) -> int:
        return self.jt.reg_read(off)

    def reg_read_many(self, offs) -> list[int]:
        return self.jt.reg_read_many(offs)

    def poll(self, off: int, mask: int, val: int, timeout: float = 600.0,
             expect: float = 0.0) -> int:
        t0 = time.monotonic()
        while True:
            r = self.reg_read(off)
            if r & mask == val:
                return r
            if time.monotonic() - t0 > timeout:
                raise TimeoutError(f"register {off:#x} = {r:#x}, waiting for {val:#x}/{mask:#x}")
            time.sleep(0.005)

    def close(self) -> None:
        pass
