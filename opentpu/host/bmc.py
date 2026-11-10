"""The BCU1525's board management controller (BMC, an MSP432) through the FPGA's pins to its UART:
the bridge otpu_bmc_uart (rtl/boards/bcu1525/otpu_bmc_uart.sv; BAR0 R_BMC, CAPS2 bit4) and the
BMC's I2C pass-through to the VCCINT regulator, an LTC3884 at 0x5B, read only. otpu-smi reads
VCCINT's power and temperatures from here (measured); otpu-bmc prints them, or every read of
the controller the tables below name (docs/bcu1525.md section 13).

    otpu-bmc                 the BMC's firmware and VCCINT: the regulator's input (V, A, W),
                             each page's output (V, A, W, temperature), the controller's die
    otpu-bmc --all           also the LTC3884's identification, status, settings and peaks
    otpu-bmc --json          as JSON
    otpu-bmc --fake          an in-memory card with a model BMC (fake_bmc; tests, demo)

The protocol is the card vendor's tools' (recovered from them, not published by the vendor):
115200 baud 8E1; frames 5C 02 ... 5C 03 with 5C doubled inside; inside, the command, 0, the
payload's length, the payload and the 16-bit sum of those bytes (little endian). 0x0C asks the
firmware's version (reply 0x8C: 4 bytes; 0x10002..0x10005 are the versions the vendor's tools
accept, and no I2C request goes to another); 0x0A is an I2C transaction (the address, the
write's length | 0x80, the read's length, the bytes written; reply 0x8A: 1 for success, then
the bytes read). Each read ends in the SMBus packet error code (PEC: CRC-8 over the address
bytes, the bytes written and the bytes read), checked. 00 5C 03 ends any partial frame at the
BMC (each session starts with it). The vendor's tools wait 20 ms after a request; this polls the
RX FIFO for the reply instead.

Safety. Every transaction is a read. The bytes written are one command code of the tables
(READS), or PAGE_PLUS_READ (0x06) of one, which reads a page's register without changing PAGE;
no other address, command or BMC request is ever sent (exchange refuses them). The regulator's
settings (OPERATION, VOUT_COMMAND, the limits) are read, never written.

Scope. The LTC3884 regulates VCCINT alone. READ_PIN is its input power on the 12 V side
(READ_VIN x READ_IIN through its input sense resistor; one value for both pages, never summed
from pages), the regulator's loss included; READ_POUT is each page's output. Neither covers the
rest of the card (the DIMMs, the transceivers, the other rails), so otpu-smi names the figure
"VCCINT in". Nor perhaps all of VCCINT: the VCU1525's design has two LTC3874 slave controllers
off PMBus beside it (docs/bcu1525.md section 13); the figures are lower bounds until checked.

Sharing. Processes serialize their sessions with an flock on <run dir>/<device>.bmc.lock. A
bridge with the fabric's watch (ID "BMC2": otpu_bmc_watch reads the regulator itself for the
thermal governor) lends its line: a session asks for it (HOLD) and waits for HELD before its
sync, gives it back at its end, and asks again when the watch has taken it back (REVOKED, its
hold's limit: the request then fails and is retried). watch() reads the watch's readings, no
request sent: the guard's (opentpu/host/gov.py) and the quickest look at the regulator.
"""
from __future__ import annotations

import fcntl
import json
import os
import struct
import time

from . import regs as R
from .i2c import linear11, linear16
from .runstate import ensure_run_dir, open_shared, run_dir

# otpu_bmc_uart's registers (R_BMC + offset)
RX, TX, STATUS, CTRL, LEVEL, ID, DIV, HOLD = 0x00, 0x04, 0x08, 0x0C, 0x10, 0x14, 0x18, 0x1C
W_HOT, W_P0, W_P1, W_CTL, W_PIN, W_VIN, W_FW, W_CNT = range(0x20, 0x40, 4)   # the watch's
ID_BMC, ID_BMC2 = 0x424D4331, 0x424D4332    # "BMC1", "BMC2" (with the watch)
BAUD = 115200                               # the line's (DIV: core clock cycles a bit)
DEPTH = 256                                 # each FIFO's bytes (otpu_bmc_uart DEPTH)
RX_VALID = 1 << 31
ST_TX_IDLE, ST_HELD, ST_STOPPED = 1 << 9, 1 << 11, 1 << 13
ST_ERRORS = {5: "RX overrun", 6: "framing error", 7: "parity error", 8: "a TX byte dropped",
             12: "the watch took the line back"}
CTRL_TX_RESET, CTRL_RX_RESET = 1, 2
HOLD_ASK, HOLD_HELD = 1, 2
AGE_NONE = 0xFFFF                           # a watch reading's age before its first

ADDR = 0x5B                                 # the LTC3884 (VCCINT) on the BMC's bus
FIRMWARE = range(0x10002, 0x10006)
SYNC = b"\x00\x5c\x03"
PAGE_PLUS_READ = 0x06
PAGES = (0, 1)
CMD_VERSION, CMD_I2C = 0x0C, 0x0A

# The LTC3884's reads (its data sheet's command summary): name -> (command, format); "b" a byte,
# "w" a word (a bit field), "l11" LINEAR11, "l16" LINEAR16 (the page's VOUT_MODE exponent).
# MFR_SPECIAL_ID (0x4Cxx for the LTC3884 family), MFR_ID and MFR_MODEL (blocks of 3 and 7 bytes)
# identify it.
MFR_SPECIAL_ID = 0xE7
IDENT = {"mfr_id": (0x99, 3), "mfr_model": (0x9A, 7)}
GLOBAL = {   # unpaged: the input side and the controller
    "vin": (0x88, "l11"), "iin": (0x89, "l11"), "pin": (0x97, "l11"), "die_c": (0x8E, "l11"),
    "capability": (0x19, "b"), "pmbus_revision": (0x98, "b"), "mfr_address": (0xE6, "b"),
    "mfr_config_all": (0xD1, "b"), "mfr_common": (0xEF, "b"), "status_input": (0x7C, "b"),
    "status_cml": (0x7E, "b"), "pin_accuracy": (0xAC, "b"), "peak_vin": (0xDE, "l11"),
    "peak_iin": (0xE1, "l11"), "peak_die_c": (0xF4, "l11"), "ichip": (0xE4, "l11"),
    "rsense_mohm": (0xE8, "l11"), "mfr_pads": (0xE5, "w"), "vin_ov_fault": (0x55, "l11"),
    "vin_uv_warn": (0x58, "l11"), "iin_oc_warn": (0x5D, "l11"),
}
PAGED = {    # each page (output channel), through PAGE_PLUS_READ
    "vout": (0x8B, "l16"), "iout": (0x8C, "l11"), "pout": (0x96, "l11"), "temp_c": (0x8D, "l11"),
    "vout_mode": (0x20, "b"), "operation": (0x01, "b"), "on_off_config": (0x02, "b"),
    "status_byte": (0x78, "b"), "status_word": (0x79, "w"), "status_vout": (0x7A, "b"),
    "status_iout": (0x7B, "b"), "status_temperature": (0x7D, "b"),
    "status_mfr_specific": (0x80, "b"), "mfr_chan_config": (0xD0, "b"),
    "mfr_pwm_mode": (0xD4, "b"), "freq_khz": (0x95, "l11"), "vout_command": (0x21, "l16"),
    "vout_margin_high": (0x25, "l16"), "vout_margin_low": (0x26, "l16"),
    "vout_ov_fault": (0x40, "l16"), "vout_ov_warn": (0x42, "l16"), "vout_uv_warn": (0x43, "l16"),
    "vout_uv_fault": (0x44, "l16"), "peak_vout": (0xDD, "l16"), "peak_iout": (0xD7, "l11"),
    "peak_temp_c": (0xDF, "l11"), "iout_cal_gain_mohm": (0x38, "l11"),
    "iout_oc_fault": (0x46, "l11"), "iout_oc_warn": (0x4A, "l11"), "ot_fault_c": (0x4F, "l11"),
    "ot_warn_c": (0x51, "l11"), "ton_delay_ms": (0x60, "l11"), "ton_rise_ms": (0x61, "l11"),
    "toff_delay_ms": (0x64, "l11"), "toff_fall_ms": (0x65, "l11"),
}
READS = frozenset({MFR_SPECIAL_ID, *(c for c, _ in IDENT.values()),
                   *(c for c, _ in GLOBAL.values()), *(c for c, _ in PAGED.values())})
TELEMETRY_GLOBAL = ("vin", "iin", "pin", "die_c")
TELEMETRY_PAGED = ("vout", "iout", "pout", "temp_c")


class BmcError(IOError):
    pass


class BmcBusy(BmcError):
    pass


class ProtocolError(BmcError):
    pass


# ------------------------------------------------------------------------------ the protocol
def crc8(data: bytes) -> int:
    """The SMBus PEC: CRC-8, x^8 + x^2 + x + 1, from 0."""
    c = 0
    for b in data:
        c ^= b
        for _ in range(8):
            c = (c << 1 ^ (0x07 if c & 0x80 else 0)) & 0xFF
    return c


def encode_inner(cmd: int, payload: bytes = b"") -> bytes:
    payload = bytes(payload)
    if not 0 <= cmd <= 0xFF or len(payload) > 0xFF:
        raise ValueError("a BMC request's command or payload is too large")
    f = bytes((cmd, 0, len(payload))) + payload
    return f + struct.pack("<H", sum(f) & 0xFFFF)


def decode_inner(frame: bytes, cmd: int) -> bytes:
    """A reply's payload, checked: its length, its command and its sum."""
    if len(frame) < 5 or len(frame) != frame[2] + 5:
        raise ProtocolError(f"BMC reply of {len(frame)} bytes, length field "
                            f"{frame[2] if len(frame) > 2 else None}")
    if frame[0] != cmd:
        raise ProtocolError(f"BMC reply {frame[0]:#04x}, expected {cmd:#04x}")
    if sum(frame[:-2]) & 0xFFFF != int.from_bytes(frame[-2:], "little"):
        raise ProtocolError("BMC reply checksum mismatch")
    return bytes(frame[3:-2])


def wrap_frame(inner: bytes) -> bytes:
    return b"\x5c\x02" + bytes(inner).replace(b"\x5c", b"\x5c\x5c") + b"\x5c\x03"


class FrameDecoder:
    """Frames out of a byte stream: 5C 02 starts one, 5C 03 ends it, 5C 5C is a 5C in it;
    bytes outside a frame are dropped."""
    MAX = 260

    def __init__(self):
        self.active = self.escape = False
        self.data = bytearray()

    def feed(self, data: bytes) -> list[bytes]:
        frames = []
        for b in data:
            if self.escape:
                self.escape = False
                if b == 0x02:
                    self.active = True
                    self.data.clear()
                elif self.active and b == 0x03:
                    frames.append(bytes(self.data))
                    self.active = False
                elif self.active and b == 0x5C:
                    self.data.append(b)
                elif self.active:
                    self.active = False
                    raise ProtocolError(f"BMC frame: escape 5c {b:02x}")
            elif b == 0x5C:
                self.escape = True
            elif self.active:
                self.data.append(b)
                if len(self.data) > self.MAX:
                    self.active = False
                    raise ProtocolError("BMC frame longer than any reply")
        return frames


def read_only(write: bytes) -> bool:
    """Whether `write` is a read's command: a code of READS, or PAGE_PLUS_READ of one."""
    w = bytes(write)
    if len(w) == 1:
        return w[0] in READS
    return len(w) == 4 and w[0] == PAGE_PLUS_READ and w[1] == 2 and w[2] in PAGES \
        and w[3] in READS


# ------------------------------------------------------------------------------ the UART
class Uart:
    """otpu_bmc_uart through a transport's register reads and writes."""

    def __init__(self, t, base: int = R.R_BMC):
        self.t, self.base = t, base

    def rd(self, off: int) -> int:
        return self.t.reg_read(self.base + off)

    def wr(self, off: int, v: int) -> None:
        self.t.reg_write(self.base + off, v)

    def status(self) -> int:
        """STATUS; an error flag (cleared by this read) raises ProtocolError."""
        st = self.rd(STATUS)
        bad = [s for b, s in ST_ERRORS.items() if st >> b & 1]
        if bad:
            raise ProtocolError("BMC UART: " + ", ".join(bad))
        return st

    def write(self, data: bytes, timeout: float) -> None:
        """Into the TX FIFO, as it has room."""
        deadline = time.monotonic() + timeout
        i = 0
        while i < len(data):
            free = DEPTH - (self.rd(LEVEL) >> 16)
            if free <= 0:
                if time.monotonic() > deadline:
                    raise TimeoutError("BMC UART: the TX FIFO stays full")
                time.sleep(0.001)
                continue
            for b in data[i:i + free]:
                self.wr(TX, b)
            i += free

    def read(self) -> bytes:
        """The bytes the RX FIFO holds, oldest first."""
        n = self.rd(LEVEL) & 0xFFFF
        if not n:
            return b""
        v = self.t.reg_read_many([self.base + RX] * n)
        if not all(x & RX_VALID for x in v):
            raise ProtocolError("BMC UART: RX FIFO emptied by another reader")
        return bytes(x & 0xFF for x in v)

    def wait_tx_idle(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while not self.status() & ST_TX_IDLE:
            if time.monotonic() > deadline:
                raise TimeoutError("BMC UART: TX never idle")
            time.sleep(0.0005)

    def hold(self, timeout: float) -> None:
        """The line from the watch: asked, then HELD (the watch gives it at a round's end)."""
        self.wr(HOLD, HOLD_ASK)
        deadline = time.monotonic() + timeout
        while not self.rd(HOLD) & HOLD_HELD:
            if time.monotonic() > deadline:
                raise TimeoutError(f"BMC UART: the watch kept the line {timeout} s")
            time.sleep(0.001)

    def release(self) -> None:
        self.wr(HOLD, 0)


# ------------------------------------------------------------------------------ a session
_known: dict[str, dict] = {}    # per device, this process: the firmware and each VOUT_MODE


class Bmc:
    """A session with the BMC: the lock, the UART, the requests. Use as a context manager; `t`
    is a transport of a bitstream with CAPS2 bit4."""
    SETTLE = 0.005                  # after the sync: a reply to it, if any, arrives

    def __init__(self, t, timeout: float = 0.5, retries: int = 2, lock: bool = True,
                 lock_timeout: float = 2.0, poll: float = 0.001, hold_timeout: float = 3.0):
        self.uart, self.timeout, self.retries, self.poll = Uart(t), timeout, retries, poll
        self.hold_timeout = hold_timeout            # the watch's round, its failures' quiet in
        self.watch = False                          # a bridge with the watch: holds the line
        name = getattr(t, "devname", None)
        self.lock_path = run_dir() / f"{name}.bmc.lock" if lock and name else None
        self.lock_timeout = lock_timeout
        self.known = _known.setdefault(name, {}) if name else {}
        self._lock_fd = None
        self.requests = 0

    def __enter__(self):
        if self.lock_path is not None:
            ensure_run_dir()
            fd = open_shared(self.lock_path)
            deadline = time.monotonic() + self.lock_timeout
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() > deadline:
                        os.close(fd)
                        raise BmcBusy(f"BMC lock {self.lock_path} held by another process")
                    time.sleep(0.005)
            self._lock_fd = fd
        try:
            ident = self.uart.rd(ID)
            if ident not in (ID_BMC, ID_BMC2):
                raise BmcError(f"no BMC bridge at BAR0 {self.uart.base:#x} (ID {ident:#010x})")
            self.watch = ident == ID_BMC2
            self.sync()
        except BaseException:
            self._unlock()
            raise
        return self

    def __exit__(self, *exc):
        self._unlock()

    def _unlock(self):
        if self.watch:
            try:
                self.uart.release()             # the line back to the watch
            except OSError:
                pass
        if self._lock_fd is not None:
            os.close(self._lock_fd)             # releases the flock
            self._lock_fd = None

    def sync(self) -> None:
        """The line held (with the watch), both FIFOs emptied (what a killed session left),
        00 5C 03 to end a partial frame at the BMC, then the RX FIFO emptied of anything said to
        it; the error flags cleared."""
        u = self.uart
        if self.watch:
            u.hold(self.hold_timeout)
        u.wr(CTRL, CTRL_TX_RESET | CTRL_RX_RESET)
        u.rd(STATUS)
        u.write(SYNC, self.timeout)
        try:
            u.wait_tx_idle(self.timeout)
        except ProtocolError:
            pass                                # (RX noise: the flags are cleared below)
        time.sleep(self.SETTLE)
        u.wr(CTRL, CTRL_RX_RESET)
        u.rd(STATUS)

    def _retry(self, fn):
        for k in range(self.retries + 1):
            try:
                return fn()
            except (ProtocolError, TimeoutError):
                if k == self.retries:
                    raise
                self.sync()

    def command(self, cmd: int, payload: bytes, reply: int) -> bytes:
        """One request and its reply's payload (no retry)."""
        u = self.uart
        u.wr(CTRL, CTRL_RX_RESET)               # a late byte of an earlier exchange is not ours
        u.write(wrap_frame(encode_inner(cmd, payload)), self.timeout)
        self.requests += 1
        dec, frames = FrameDecoder(), []
        deadline = time.monotonic() + self.timeout
        while True:
            u.status()
            frames += dec.feed(u.read())
            if frames:
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f"no reply from the BMC to request {cmd:#04x}")
            time.sleep(self.poll)
        if len(frames) > 1:
            raise ProtocolError(f"{len(frames)} BMC replies to one request")
        return decode_inner(frames[0], reply)

    def firmware(self) -> int:
        """The BMC firmware's version, checked against the ones the protocol is known for."""
        if "firmware" not in self.known:
            r = self._retry(lambda: self.command(CMD_VERSION, b"", CMD_VERSION | 0x80))
            if len(r) != 4:
                raise ProtocolError(f"BMC firmware version of {len(r)} bytes")
            v = int.from_bytes(r, "little")
            if v not in FIRMWARE:
                raise BmcError(f"BMC firmware {v:#010x}: not one of the versions this protocol "
                               f"is known for ({FIRMWARE.start:#x}..{FIRMWARE.stop - 1:#x})")
            self.known["firmware"] = v
        return self.known["firmware"]

    def exchange(self, write: bytes, n: int) -> bytes:
        """One I2C write-then-read with the LTC3884 through the BMC (no retry): `write` a read's
        command (read_only), `n` bytes read back."""
        write = bytes(write)
        if not read_only(write) or not 1 <= n <= 254:
            raise ValueError(f"not a read of the LTC3884: {write.hex()} ({n} bytes)")
        self.firmware()
        r = self.command(CMD_I2C, bytes((ADDR, len(write) | 0x80, n)) + write, CMD_I2C | 0x80)
        if len(r) != n + 1 or r[0] != 1:
            raise ProtocolError(f"I2C through the BMC failed: status "
                                f"{r[0] if r else None}, {len(r) - 1} of {n} bytes")
        return r[1:]

    def read(self, cmd: int, size: int, page: int | None = None, block: bool = False) -> bytes:
        """`size` data bytes of `cmd` (a page's through PAGE_PLUS_READ, a block), PEC checked."""
        w = bytes((cmd,)) if page is None else bytes((PAGE_PLUS_READ, 2, page, cmd))
        block = block or page is not None
        n = size + block + 1

        def once():
            r = self.exchange(w, n)
            if crc8(bytes((ADDR << 1,)) + w + bytes((ADDR << 1 | 1,)) + r):
                raise ProtocolError(f"PEC mismatch reading {cmd:#04x}")
            if block and r[0] != size:
                raise ProtocolError(f"block of {r[0]} bytes reading {cmd:#04x}, expected {size}")
            return r[block:-1]
        return self._retry(once)

    def vout_mode(self, page: int) -> int:
        k = f"vout_mode{page}"
        if k not in self.known:
            self.known[k] = self.read(PAGED["vout_mode"][0], 1, page)[0]
        return self.known[k]

    def value(self, name: str, page: int | None = None):
        """A command of GLOBAL (page None) or PAGED by name, decoded."""
        cmd, fmt = (GLOBAL if page is None else PAGED)[name]
        raw = int.from_bytes(self.read(cmd, 1 if fmt == "b" else 2, page), "little")
        if fmt == "l11":
            return linear11(raw)
        if fmt == "l16":
            return linear16(raw, self.vout_mode(page))
        return raw


# ------------------------------------------------------------------------------ readings
WATCH_READINGS = (("p0_c", W_P0), ("p1_c", W_P1), ("ctl_c", W_CTL), ("pin", W_PIN),
                  ("vin", W_VIN))


def watch(t, base: int = R.R_BMC) -> dict | None:
    """The fabric watch's readings, no request sent (a bridge without it: None): page 0's and
    page 1's temperatures, the controller's die (C), READ_PIN (W) and READ_VIN (V), each None
    before its first reading; "ages_ms" theirs; "hot_c" the hottest temperature and "hot_age_ms"
    the oldest of the three (what the governor sees); the firmware, the replies taken and the
    exchanges failed; "stopped" when the firmware is not one the protocol is known for (the
    watch then reads nothing). Registers only: safe beside another process's session."""
    u = Uart(t, base)
    if u.rd(ID) != ID_BMC2:
        return None
    d, ages = {}, {}
    for k, off in WATCH_READINGS:
        v = u.rd(off)
        ages[k] = v >> 16
        d[k] = None if v >> 16 == AGE_NONE else linear11(v & 0xFFFF)
    hot, cnt, fw = u.rd(W_HOT), u.rd(W_CNT), u.rd(W_FW)
    q = hot & 0xFFFF
    d.update(ages_ms=ages, hot_age_ms=hot >> 16,
             hot_c=None if hot >> 16 == AGE_NONE else (q - 0x10000 if q & 0x8000 else q) / 4,
             firmware=fw, ok=cnt & 0xFFFF, bad=cnt >> 16,
             stopped=fw != 0 and fw not in FIRMWARE)     # (not STATUS: its read clears flags)
    return d


def telemetry(b: Bmc) -> dict:
    """VCCINT's readings: the input side's {"vin", "iin", "pin", "die_c"} and "pages", each
    page's {"page", "vout", "iout", "pout", "temp_c"} (V, A, W, C)."""
    d = {k: b.value(k) for k in TELEMETRY_GLOBAL}
    d["pages"] = [{"page": p, **{k: b.value(k, p) for k in TELEMETRY_PAGED}} for p in PAGES]
    return d


def measured(t, **kw) -> dict | None:
    """otpu-smi's measurement, i2c.measured's shape: {"w": READ_PIN (VCCINT's regulator input),
    "how": "PIN", "label": "VCCINT in", "rails": [one], "board_temp_c": None, "vccint":
    telemetry(), "firmware"}; None when another process holds the BMC lock. Errors raise
    (BmcError, TimeoutError: OSErrors). kw: Bmc options."""
    try:
        with Bmc(t, **kw) as b:
            v = telemetry(b)
            fw = b.firmware()
    except BmcBusy:
        return None
    return {"w": v["pin"], "how": "PIN", "label": "VCCINT in",
            "rails": [{"bus": "bmc", "addr": ADDR, "w": v["pin"], "how": "PIN"}],
            "board_temp_c": None, "vccint": v, "firmware": fw}


def info(b: Bmc) -> dict:
    """Every read of the tables: the identification first (another controller than an LTC3884
    gets nothing more), then GLOBAL and each page's PAGED; a read that fails is None, its
    error in "errors"."""
    errors = {}

    def safe(key, fn):
        try:
            return fn()
        except (OSError, ValueError) as e:
            errors[key] = f"{type(e).__name__}: {e}"
            return None
    sid = int.from_bytes(b.read(MFR_SPECIAL_ID, 2), "little")
    d = {"firmware": b.firmware(), "addr": ADDR, "mfr_special_id": sid, "ltc3884": sid >> 8 == 0x4C}
    if not d["ltc3884"]:
        return d
    for k, (cmd, n) in IDENT.items():
        d[k] = safe(k, lambda c=cmd, n=n: b.read(c, n, block=True).decode("ascii", "replace"))
    d["global"] = {k: safe(k, lambda k=k: b.value(k)) for k in GLOBAL}
    acc = d["global"].get("pin_accuracy")
    d["pin_accuracy_pct"] = acc / 10 if acc is not None else None
    d["pages"] = [{"page": p, **{k: safe(f"page{p}.{k}", lambda k=k, p=p: b.value(k, p))
                                 for k in PAGED}} for p in PAGES]
    d["errors"] = errors
    return d


# ------------------------------------------------------------------------------ otpu-bmc
def _f(x, fmt: str) -> str:
    return "n/a" if x is None else format(x, fmt)


def _text(dev: str, v: dict, fw: int) -> str:
    out = [f"{dev}  BMC firmware {fw:#010x}  VCCINT regulator LTC3884 at {ADDR:#04x}",
           f"  input    {_f(v['vin'], '6.2f')} V  {_f(v['iin'], '6.2f')} A  "
           f"{_f(v['pin'], '6.1f')} W   (READ_PIN, both pages, the regulator's loss included)"]
    for p in v["pages"]:
        out.append(f"  page {p['page']}   {_f(p['vout'], '6.3f')} V  {_f(p['iout'], '6.2f')} A  "
                   f"{_f(p['pout'], '6.1f')} W   {_f(p['temp_c'], '.0f')} °C")
    out.append(f"  controller die {_f(v['die_c'], '.0f')} °C")
    return "\n".join(out)


def main(argv=None, open_transport=None) -> int:
    """otpu-bmc: the BCU1525's BMC and VCCINT regulator (read only)."""
    import argparse
    import sys
    ap = argparse.ArgumentParser(prog="otpu-bmc", description=main.__doc__)
    ap.add_argument("--dev", default="/dev/xdma0", help="XDMA device prefix (default /dev/xdma0)")
    ap.add_argument("--all", action="store_true",
                    help="every read of the LTC3884: identification, status, settings, peaks")
    ap.add_argument("--json", action="store_true", help="JSON output")
    ap.add_argument("--timeout", type=float, default=0.5, help="seconds per reply (default 0.5)")
    ap.add_argument("--fake", action="store_true", help="an in-memory card with a model BMC")
    a = ap.parse_args(argv)
    if a.fake and open_transport is None:
        from .fake import FakeTransport
        from .fake_bmc import FakeBmc
        fk = FakeTransport(regmap=R.CAPS2_SINCE, bmc=FakeBmc())
        open_transport = lambda dev: fk                                 # noqa: E731
    if open_transport is None:
        from .board import XdmaTransport
        open_transport = lambda dev: XdmaTransport(dev, dma=False)      # noqa: E731
    from .board import Board
    try:
        t = open_transport(a.dev)
        i = Board(t, check=False, lock=False).info()
        if not i["caps2"]["bmc"]:
            print(f"otpu-bmc: {a.dev}: no BMC bridge in this bitstream (register map "
                  f"{i['regmap']}, CAPS2 bit4 clear)", file=sys.stderr)
            return 1
        with Bmc(t, timeout=a.timeout) as b:
            fw = b.firmware()
            d = info(b) if a.all else {"firmware": fw}
            d["vccint"] = telemetry(b)
    except (OSError, ValueError) as e:
        print(f"otpu-bmc: {a.dev}: {e}", file=sys.stderr)
        return 1
    if a.json:
        print(json.dumps(d, indent=1))
        return 0
    print(_text(a.dev, d["vccint"], fw))
    if a.all:
        print(f"  {d.get('mfr_id')} {d.get('mfr_model')}  MFR_SPECIAL_ID "
              f"{d['mfr_special_id']:#06x}")
        for k, v in (d.get("global") or {}).items():
            if k not in TELEMETRY_GLOBAL:
                print(f"  {k:<22} {_val(k, v, GLOBAL)}")
        for p in d.get("pages") or []:
            print(f"  page {p['page']}")
            for k, v in p.items():
                if k != "page" and k not in TELEMETRY_PAGED:
                    print(f"    {k:<20} {_val(k, v, PAGED)}")
        for k, e in (d.get("errors") or {}).items():
            print(f"  {k}: {e}")
    return 0


def _val(k: str, v, table: dict) -> str:
    if v is None:
        return "n/a"
    fmt = table[k][1]
    if k == "pin_accuracy":
        return f"{v / 10:.1f} %"
    if fmt == "b":
        return f"{v:#04x}"
    if fmt == "w":
        return f"{v:#06x}"
    return f"{v:.4g}"


if __name__ == "__main__":
    raise SystemExit(main())
