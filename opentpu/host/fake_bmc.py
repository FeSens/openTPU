"""A model of the BCU1525's BMC bridge for FakeTransport (tests, otpu-bmc --fake): otpu_bmc_uart's
registers and FIFOs, and behind its line a BMC that answers the vendor protocol's firmware and I2C
requests (opentpu/host/bmc.py) from an LTC3884 model, PEC included.

The line has no baud time: a byte written to TX reaches the BMC at once, and the BMC's reply
enters the RX FIFO `chunk` bytes per LEVEL read (the host sees a reply arrive in pieces). The
LTC3884 answers the reads of bmc.GLOBAL, bmc.PAGED and the identification, direct or through
PAGE_PLUS_READ; any other write it is asked for is NACKed (status 0) and recorded in `refused`
(tests assert that the host never sends one). `corrupt` replies (a count) go out with a wrong
PEC, `silent` requests get no reply, `noise` puts a parity error on the line before the next
reply.

With `watch` (the default; ID "BMC2") the bridge has the fabric's watch: HOLD is granted at once,
TX bytes written without the line wait in `unheld` (never reaching the BMC) until a TX reset,
`revoke_after` bytes into a hold the watch takes the line back (REVOKED; the rest of the request
is not sent), and the W_* registers hold the model's readings at `watch_age` ms (AGE_NONE: none
yet; an unknown firmware stops the watch: none ever).
"""
from __future__ import annotations

from collections import deque

from . import bmc as B
from .i2c import to_linear11

VALUES = {   # the model's readings: a card at work
    "vin": 12.1, "iin": 2.05, "pin": 24.8, "die_c": 52.0, "capability": 0xB0,
    "pmbus_revision": 0x22, "mfr_address": 0x5B, "pin_accuracy": 50, "peak_vin": 12.3,
    "peak_iin": 3.1, "peak_die_c": 61.0, "ichip": 0.031, "rsense_mohm": 1.0,
    "vin_ov_fault": 15.5, "vin_uv_warn": 10.0, "iin_oc_warn": 10.0,
}
PAGE_VALUES = [
    {"vout": 0.85, "iout": 13.6, "pout": 11.6, "temp_c": 45.0, "vout_mode": 0x14,
     "operation": 0x80, "freq_khz": 425.0, "vout_command": 0.85, "ot_fault_c": 120.0},
    {"vout": 0.85, "iout": 13.4, "pout": 11.4, "temp_c": 47.0, "vout_mode": 0x14,
     "operation": 0x80, "freq_khz": 425.0, "vout_command": 0.85, "ot_fault_c": 120.0},
]


class FakeLtc3884:
    def __init__(self, values: dict | None = None, pages: list | None = None):
        self.values = {**VALUES, **(values or {})}
        self.pages = [dict(p) for p in (pages or PAGE_VALUES)]
        self.reads: list[bytes] = []            # the writes of the reads answered

    @staticmethod
    def _word(fmt: str, x, mode: int) -> bytes:
        if fmt == "b":
            return bytes((int(x) & 0xFF,))
        if fmt == "l11":
            return to_linear11(float(x)).to_bytes(2, "little")
        if fmt == "l16":
            e = mode & 0x1F
            return round(x / 2.0 ** (e - 32 if e & 0x10 else e)).to_bytes(2, "little")
        return int(x).to_bytes(2, "little")

    def answer(self, write: bytes) -> bytes | None:
        """The data of a read (a block's count first), or None (NACK)."""
        if len(write) == 4 and write[:2] == bytes((B.PAGE_PLUS_READ, 2)) and write[2] < 2:
            data = self._direct(write[3], write[2])
            return None if data is None else bytes((len(data),)) + data
        if len(write) == 1:
            return self._direct(write[0], None)
        return None

    def _direct(self, cmd: int, page: int | None) -> bytes | None:
        if cmd == B.MFR_SPECIAL_ID:
            return (0x4C12).to_bytes(2, "little")
        for k, (c, n) in B.IDENT.items():
            if c == cmd:
                s = {"mfr_id": b"LTC", "mfr_model": b"LTC3884"}[k]
                return bytes((len(s),)) + s if page is None else None
        # a direct read of a paged command reads page 0 (the PAGE the BMC leaves); an unpaged
        # one through PAGE_PLUS_READ its one value
        order = ((B.GLOBAL, self.values), (B.PAGED, self.pages[0])) if page is None else \
            ((B.PAGED, self.pages[page]), (B.GLOBAL, self.values))
        for table, vals in order:
            for k, (c, fmt) in table.items():
                if c == cmd:
                    return self._word(fmt, vals.get(k, 0), self.pages[page or 0]["vout_mode"])
        return None


class FakeBmc:
    """otpu_bmc_uart's registers (offsets from R_BMC) with a BMC behind the line."""

    def __init__(self, ltc: FakeLtc3884 | None = None, firmware: int = 0x10004, chunk: int = 4,
                 depth: int = B.DEPTH, watch: bool = True):
        self.ltc = ltc or FakeLtc3884()
        self.watch, self.asked, self.holds, self.held_bytes = watch, False, 0, 0
        self.revoke_after: int | None = None
        self.unheld = bytearray()                   # TX bytes without the line
        self.watch_age = 3
        self.firmware, self.chunk, self.depth = firmware, chunk, depth
        self.rxq: deque[int] = deque()              # the RX FIFO (the host's)
        self.line: deque[int] = deque()             # the BMC's reply on its way
        self.flags = 0
        self.dec = B.FrameDecoder()
        self.requests: list[tuple[int, bytes]] = []
        self.refused: list[bytes] = []
        self.sent = bytearray()                     # every byte the host sent
        self.corrupt = self.silent = 0
        self.noise = False

    @property
    def held(self) -> bool:
        return not self.watch or self.asked

    def _watch_reg(self, off: int) -> int:
        """W_*: the model's readings as the watch has them."""
        age = B.AGE_NONE if self.firmware not in B.FIRMWARE else self.watch_age
        if off == B.W_FW:
            return self.firmware
        if off == B.W_CNT:
            return 0 if age == B.AGE_NONE else 12
        temps = [p["temp_c"] for p in self.ltc.pages] + [self.ltc.values["die_c"]]
        if off == B.W_HOT:
            return age << 16 | (0 if age == B.AGE_NONE else round(4 * max(temps)) & 0xFFFF)
        x = dict(zip((B.W_P0, B.W_P1, B.W_CTL, B.W_PIN, B.W_VIN),
                     temps + [self.ltc.values["pin"], self.ltc.values["vin"]]))[off]
        return age << 16 | (0 if age == B.AGE_NONE else to_linear11(x))

    # ---- the registers
    def reg_write(self, off: int, val: int) -> None:
        off &= 0x3F
        if off == B.TX:
            if not self.held:
                self.unheld.append(val & 0xFF)
                return
            if self.revoke_after is not None and self.held_bytes >= self.revoke_after:
                self.asked, self.revoke_after = False, None    # HOLD_MS: the watch's again
                self.flags |= 1 << 12
                self.unheld.append(val & 0xFF)
                return
            self.held_bytes += 1
            self.sent.append(val & 0xFF)
            try:
                frames = self.dec.feed(bytes((val & 0xFF,)))
            except B.ProtocolError:
                frames = []
            for f in frames:
                self._request(f)
        elif off == B.CTRL:
            if val & B.CTRL_RX_RESET:
                self.rxq.clear()
            if val & B.CTRL_TX_RESET:
                self.unheld.clear()
        elif off == B.HOLD and self.watch:
            self.asked = bool(val & B.HOLD_ASK)
            if self.asked:
                self.holds += 1
                self.held_bytes = 0

    def reg_read(self, off: int) -> int:
        off &= 0x3F
        if off == B.RX:
            return B.RX_VALID | self.rxq.popleft() if self.rxq else 0
        if off == B.STATUS:
            st = (bool(self.rxq) | (len(self.rxq) >= self.depth) << 1 | 1 << 2 | 1 << 9
                  | 1 << 10 | self.held << 11 | self.flags)
            self.flags = 0
            return st
        if off == B.LEVEL:
            for _ in range(min(self.chunk, len(self.line))):
                if len(self.rxq) < self.depth:
                    self.rxq.append(self.line.popleft())
                else:
                    self.line.popleft()
                    self.flags |= 1 << 5
            return len(self.rxq)
        if off == B.ID:
            return B.ID_BMC2 if self.watch else B.ID_BMC
        if off == B.DIV:
            return 868
        if off == B.HOLD:
            return self.asked | self.held << 1
        if self.watch and B.W_HOT <= off <= B.W_CNT:
            return self._watch_reg(off)
        return 0

    # ---- the BMC
    def _reply(self, cmd: int, payload: bytes) -> None:
        if self.noise:
            self.noise = False
            self.flags |= 1 << 7                    # a byte lost to a parity error
            return
        frame = B.wrap_frame(B.encode_inner(cmd, payload))
        self.line.extend(frame)

    def _request(self, frame: bytes) -> None:
        if len(frame) < 5 or len(frame) != frame[2] + 5 or \
                sum(frame[:-2]) & 0xFFFF != int.from_bytes(frame[-2:], "little"):
            return                                  # (a bad frame: no reply)
        cmd, payload = frame[0], frame[3:-2]
        self.requests.append((cmd, payload))
        if self.silent:
            self.silent -= 1
            return
        if cmd == B.CMD_VERSION:
            self._reply(0x8C, self.firmware.to_bytes(4, "little"))
        elif cmd == B.CMD_I2C:
            addr, wl, n, w = payload[0], payload[1] & 0x7F, payload[2], payload[3:]
            data = self.ltc.answer(w) if addr == B.ADDR and len(w) == wl else None
            if data is None or not B.read_only(w):
                self.refused.append(bytes(payload))
                self._reply(0x8A, b"\x00")
                return
            self.ltc.reads.append(bytes(w))
            pec = B.crc8(bytes((addr << 1,)) + w + bytes((addr << 1 | 1,)) + data)
            if self.corrupt:
                self.corrupt -= 1
                pec ^= 0x5A
            body = (data + bytes((pec,)) + b"\xff" * n)[:n]
            self._reply(0x8A, b"\x01" + body)
