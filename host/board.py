"""Host driver for the openTPU board (YPCB-00338 over PCIe, Xilinx XDMA).

The card exposes, through the XDMA bridge:
  - its two DDR3 channels on the bridge's AXI master: channel c at BASE[c] (2 GiB each),
    written with /dev/xdma0_h2c_0 and read with /dev/xdma0_c2h_0 (file offset = AXI address);
  - the control registers (rtl/boards/ypcb-00338/otpu_ctrl.sv) on BAR0, /dev/xdma0_user.

The accelerator addresses one logical DRAM interleaved over the channels in 64-byte beats
(logical beat b lives on channel b % 2 at BASE[b % 2] + (b // 2) * 64; rtl/mem/otpu_axi_dram.sv).
This driver applies the same map, so the host works with logical addresses only.

BoardBackend implements the Engine backend interface (write / read / run), so
`Engine(..., cfg=board_config(), backend=BoardBackend)` runs Qwen3 on the card. With
transport=SimTransport the identical protocol runs against the Verilator model of the board
(sim/verilator/tb_board.sv) -- the bring-up rehearsal.
"""
from __future__ import annotations

import mmap
import os
import struct
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

BEAT = 64                       # bytes per interleave beat
BASE = (0x0000_0000, 0x8000_0000)
CH_BYTES = 1 << 31              # 2 GiB per channel

# control registers (otpu_ctrl.sv)
R_ID, R_VERSION, R_CTRL, R_STATUS = 0x00, 0x04, 0x08, 0x0C
R_PROG_ADDR, R_PROG_N, R_CYCLES, R_CYCLES_HI, R_ICOUNT = 0x10, 0x14, 0x18, 0x1C, 0x20
R_B_RD, R_B_WR, R_A_RD, R_A_WR, R_B_STALL, R_SCRATCH = 0x24, 0x28, 0x2C, 0x30, 0x34, 0x38
CTRL_RUN, CTRL_LOAD, CTRL_CLEAR = 1, 2, 4
ST_HALTED, ST_ERROR, ST_LOADING, ST_WR_IDLE, ST_AXI_ERR = 1, 2, 4, 8, 16
ST_CALIB0, ST_CALIB1, ST_RUN = 32, 64, 128
ID_OTPU = 0x4F545055


# ------------------------------------------------------------------------------ address map
def split(addr: int, data: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    """Logical bytes at `addr` -> [(channel, channel offset, bytes)], beat-aligned pieces
    merged into one contiguous run per channel. `addr` and len(data) must be multiples of
    2 * BEAT (the caller widens unaligned ranges)."""
    assert addr % (2 * BEAT) == 0 and len(data) % (2 * BEAT) == 0
    v = data.reshape(-1, 2, BEAT)
    off = addr // 2
    return [(c, off, np.ascontiguousarray(v[:, c, :]).reshape(-1)) for c in (0, 1)]


def join(parts: list[np.ndarray]) -> np.ndarray:
    """Inverse of split: the two channels' contiguous runs -> logical bytes."""
    a, b = (p.reshape(-1, BEAT) for p in parts)
    return np.stack([a, b], axis=1).reshape(-1)


# ------------------------------------------------------------------------------ transports
class XdmaTransport:
    """The card through the Xilinx XDMA driver (dma_ip_drivers/XDMA/linux-kernel)."""

    def __init__(self, dev: str = "/dev/xdma0"):
        self.h2c = os.open(f"{dev}_h2c_0", os.O_WRONLY)
        self.c2h = os.open(f"{dev}_c2h_0", os.O_RDONLY)
        fd = os.open(f"{dev}_user", os.O_RDWR | os.O_SYNC)
        self.regs = mmap.mmap(fd, 4096, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        os.close(fd)

    def mem_write(self, ch: int, off: int, data: np.ndarray) -> None:
        mv = memoryview(np.ascontiguousarray(data, np.uint8))
        pos, step = 0, 64 << 20
        while pos < len(mv):
            n = os.pwrite(self.h2c, mv[pos:pos + step], BASE[ch] + off + pos)
            if n <= 0:
                raise IOError("XDMA h2c write failed")
            pos += n

    def mem_read(self, ch: int, off: int, n: int) -> np.ndarray:
        out = bytearray()
        while len(out) < n:
            b = os.pread(self.c2h, min(n - len(out), 64 << 20), BASE[ch] + off + len(out))
            if not b:
                raise IOError("XDMA c2h read failed")
            out += b
        return np.frombuffer(bytes(out), np.uint8)

    def reg_write(self, off: int, val: int) -> None:
        self.regs[off:off + 4] = struct.pack("<I", val & 0xFFFFFFFF)

    def reg_read(self, off: int) -> int:
        return struct.unpack("<I", self.regs[off:off + 4])[0]

    def reg_read_many(self, offs: list[int]) -> list[int]:
        return [self.reg_read(o) for o in offs]

    def poll(self, off: int, mask: int, val: int, timeout: float = 600.0) -> int:
        t = time.time()
        while True:
            r = self.reg_read(off)
            if r & mask == val:
                return r
            if time.time() - t > timeout:
                raise TimeoutError(f"register {off:#x} = {r:#x}, waiting for {val:#x}/{mask:#x}")


class SimTransport:
    """The Verilator model of the board (sim/verilator/tb_board.sv): the memory lives here as
    the two channels' physical images, register operations are queued and replayed by the
    testbench when a result is needed (a flush).

    Every flush is a fresh simulation: DRAM persists (through the channel image files), but the
    control registers, IMEM and TMEM start from reset. The Board protocol is built for that --
    a program load, its run and the reads of its counters happen in one flush -- and the Qwen3
    step programs do not rely on TMEM surviving between runs."""

    def __init__(self, ch_bytes: int = 1 << 24, stall: int = 20, seed: int = 1,
                 params: dict | None = None, plusargs: list | None = None):
        self.ch = [np.zeros(ch_bytes, np.uint8) for _ in range(2)]
        self.stall, self.seed = stall, seed
        self.params = params or {}
        self.plusargs = list(plusargs or [])     # extra simulator arguments (e.g. "+trace")
        self.script: list[str] = []
        self.regs_seen: dict[int, int] = {}
        self.reads: list[int] = []               # the last flush's reads, in order
        self.out = ""                            # the last flush's simulator output
        self.cycles = 0

    def mem_write(self, ch: int, off: int, data: np.ndarray) -> None:
        self.flush()
        self.ch[ch][off:off + len(data)] = data

    def mem_read(self, ch: int, off: int, n: int) -> np.ndarray:
        self.flush()
        return self.ch[ch][off:off + n].copy()

    def reg_write(self, off: int, val: int) -> None:
        self.script.append(f"W {off:x} {val & 0xFFFFFFFF:x}")

    def reg_read(self, off: int) -> int:
        return self.reg_read_many([off])[0]

    def reg_read_many(self, offs: list[int]) -> list[int]:
        """All reads in one simulation, in order (a later flush starts a fresh machine)."""
        for o in offs:
            self.script.append(f"R {o:x}")
        self.flush()
        return self.reads[len(self.reads) - len(offs):]

    def poll(self, off: int, mask: int, val: int, timeout: float = 0) -> int:
        self.script.append(f"P {off:x} {mask:x} {val:x}")
        return val

    def flush(self) -> None:
        if not self.script:
            return
        from opentpu import rtlsim
        root = Path(__file__).resolve().parent.parent
        srcs = [rtlsim.RTL / s for s in rtlsim.RTL_SOURCES if not s.endswith("otpu_top.sv")]
        srcs += [root / "rtl/boards/ypcb-00338/otpu_ctrl.sv",
                 root / "rtl/boards/ypcb-00338/otpu_trace.sv",
                 root / "rtl/boards/ypcb-00338/otpu_board.sv",
                 rtlsim.TB / "otpu_axi_mem.sv", rtlsim.TB / "tb_board.sv"]
        from opentpu.isasim import board_config
        p = {"WORDS": 2 * len(self.ch[0]) // 4, "MCOLS": board_config().MCOLS}  # OTPU_MCOLS
        p.update(self.params)
        exe = rtlsim.build("tb_board", srcs, p)
        with tempfile.TemporaryDirectory(prefix="otpu_board_") as d:
            d = Path(d)
            for c in (0, 1):
                self.ch[c].view("<u4").astype(">u4").tofile(d / f"ch{c}.bin")
            (d / "host.txt").write_text("\n".join(self.script) + "\n")
            self.script = []
            r = subprocess.run([str(exe), f"+dir={d}", f"+axi_stall={self.stall}",
                                f"+axi_seed={self.seed}", *self.plusargs],
                               capture_output=True, text=True)
            out = r.stdout + r.stderr
            self.out, self.reads = out, []
            if "DONE" not in out:
                raise RuntimeError(f"board simulation failed:\n{out[-3000:]}")
            for line in out.splitlines():
                if line.startswith("REG "):
                    _, a, v = line.split()
                    self.regs_seen[int(a, 16)] = int(v, 16)
                    self.reads.append(int(v, 16))
                elif line.startswith("DONE"):
                    self.cycles += int(line.split("=")[1])
            for c in (0, 1):
                self.ch[c] = np.fromfile(d / f"ch{c}_out.bin", np.uint8)


# ------------------------------------------------------------------------------ the board
class Board:
    """Logical-address access to the card's DRAM, program loading and runs."""

    def __init__(self, transport=None, check: bool = True):
        self.t = transport or XdmaTransport()
        if check:
            ident = self.t.reg_read(R_ID)
            if ident != ID_OTPU:
                raise RuntimeError(f"no openTPU on the card (ID register {ident:#x})")

    def info(self) -> dict:
        v, st = self.t.reg_read_many([R_VERSION, R_STATUS])
        return {"D": v >> 16, "MCOLS": (v >> 8) & 0xFF, "LANES": v & 0xFF,
                "calibrated": bool(st & ST_CALIB0) and bool(st & ST_CALIB1), "status": st}

    def write(self, addr: int, data) -> None:
        data = np.ascontiguousarray(data).view(np.uint8).reshape(-1)
        if len(data) == 0:
            return
        a0 = addr // (2 * BEAT) * (2 * BEAT)
        a1 = -(-(addr + len(data)) // (2 * BEAT)) * (2 * BEAT)
        if a0 != addr or a1 != addr + len(data):        # widen: read-modify-write the edges
            buf = self.read(a0, a1 - a0)
            buf[addr - a0:addr - a0 + len(data)] = data
            data = buf
        for c, off, part in split(a0, data):
            self.t.mem_write(c, off, part)

    def read(self, addr: int, n: int) -> np.ndarray:
        a0 = addr // (2 * BEAT) * (2 * BEAT)
        a1 = -(-(addr + n) // (2 * BEAT)) * (2 * BEAT)
        parts = [self.t.mem_read(c, a0 // 2, (a1 - a0) // 2) for c in (0, 1)]
        return join(parts)[addr - a0:addr - a0 + n].copy()

    def load_program(self, addr: int, words: np.ndarray) -> None:
        """Copy a program into DRAM at `addr` (chunk aligned) and into IMEM."""
        words = np.asarray(words, "<u4")
        self.write(addr, words.view(np.uint8))
        t = self.t
        t.reg_write(R_CTRL, 0)
        t.reg_write(R_PROG_ADDR, addr)
        t.reg_write(R_PROG_N, len(words) // 8)
        t.reg_write(R_CTRL, CTRL_LOAD)
        t.poll(R_STATUS, ST_LOADING, 0)

    def run(self, timeout: float = 600.0) -> dict:
        """Run the loaded program until it halts; returns the counters."""
        t = self.t
        t.reg_write(R_CTRL, CTRL_CLEAR)
        t.reg_write(R_CTRL, CTRL_RUN)
        t.poll(R_STATUS, ST_HALTED, ST_HALTED, timeout)
        st, lo, hi, ic, brd, bwr, ard, awr, bst = t.reg_read_many(
            [R_STATUS, R_CYCLES, R_CYCLES_HI, R_ICOUNT, R_B_RD, R_B_WR, R_A_RD, R_A_WR,
             R_B_STALL])
        stats = {"cycles": lo | hi << 32, "instructions": [ic], "b_reads": brd,
                 "b_writes": bwr, "a_reads": ard, "a_writes": awr, "b_stall": bst,
                 "status": st}
        t.reg_write(R_CTRL, 0)
        if st & ST_ERROR:
            raise RuntimeError("the program stopped on an illegal instruction")
        if st & ST_AXI_ERR:
            raise RuntimeError("a DRAM access got an AXI error response")
        return stats


def sim_config(spec, cap: int):
    """board_config with the DRAM cut to what the model needs (power of two), for the board
    model: the image, then the program area."""
    from opentpu.isasim import board_config
    from opentpu.llm.qwen3 import Image
    probe = Image(spec, board_config(), cap)
    need = -(-probe.nbytes // 4096) * 4096 + 4 * board_config().IMEM_WORDS
    return board_config(DRAM_BYTES=1 << max(22, (need - 1).bit_length()))


class BoardBackend:
    """Engine backend on the card: images are written once, each token's program is copied to
    the program area right after the image and loaded into IMEM, then run."""

    def __init__(self, cfg, images: list, transport=None):
        from opentpu import isa as I
        self.I = I
        if cfg.S != 1:
            raise ValueError("the board has one slice: use opentpu.isasim.board_config()")
        self.cfg = cfg
        self.board = Board(transport)
        info = self.board.info()
        if (info["D"], info["MCOLS"], info["LANES"]) != (cfg.D, cfg.MCOLS, cfg.LANES):
            raise RuntimeError(f"bitstream is D={info['D']} MCOLS={info['MCOLS']} "
                               f"LANES={info['LANES']}, the configuration differs")
        img = np.asarray(images[0], np.uint8)
        self.prog_at = -(-len(img) // 4096) * 4096
        if self.prog_at + 4 * cfg.IMEM_WORDS > cfg.DRAM_BYTES:
            raise MemoryError("no room for the program area")
        self.board.write(0, img)

    def write(self, s: int, addr: int, data: np.ndarray) -> None:
        self.board.write(addr, data)

    def read(self, s: int, addr: int, nbytes: int) -> np.ndarray:
        return self.board.read(addr, nbytes)

    def run(self, programs: list) -> dict:
        words = np.asarray(self.I.assemble(programs[0]), np.uint32)
        if len(words) > self.cfg.IMEM_WORDS:
            raise ValueError("program does not fit IMEM")
        self.board.load_program(self.prog_at, words)
        return self.board.run()

    @staticmethod
    def config(**kw):
        from opentpu.isasim import board_config
        return board_config(**kw)
