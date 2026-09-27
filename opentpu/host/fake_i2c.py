"""Open-drain I2C bus models for FakeTransport (tests, otpu-i2c --fake): slave devices that
follow the wires bit by bit, as the card's buses do behind I2C_CTRL / I2C_IN.

A line's level is the wired AND of the master's release (I2C_CTRL bit 0), every slave's
output, and `hold` (another master, or a fault). Slaves act on edges: they sample SDA when
SCL rises and change their SDA output only after SCL falls (START and STOP are SDA edges
with SCL high). A slave with `stretch` holds SCL low for that many I2C_IN reads after
acknowledging its address.

Register devices (Target) take the first byte of a write as a register pointer and stream
`regs[pointer]` (then 0xFF) on a read; later bytes of a write are data, recorded in
`writes` (tests assert that the host never sends any). A pointer not in `regs` is NACKed
(PMBus devices refuse unsupported commands that way).
"""
from __future__ import annotations

from .i2c import LM73_ID, to_linear11

IDLE, ADDR, ACK_ADDR, WRITE, ACK_W, READ, ACK_R, IGNORE = range(8)


class Target:
    def __init__(self, addr: int, regs: dict[int, bytes], stretch: int = 0):
        self.addr, self.regs, self.stretch = addr, dict(regs), stretch
        self.ptr = 0
        self.sda_out = self.scl_out = True
        self.st, self.bits, self.n, self.rw = IDLE, 0, 0, 0
        self.stream, self.mack = b"", False
        self.stretch_left = 0
        self.writes: list[tuple[int, int]] = []     # (pointer, data byte) of data writes
        self.pointers: list[int] = []               # pointer bytes received, in order
        self.nwrite = 0

    # ---- the bus calls these
    def tick(self) -> None:
        """One I2C_IN read by the master: clock stretching counts these."""
        if self.stretch_left:
            self.stretch_left -= 1
            if not self.stretch_left:
                self.scl_out = True

    def start(self) -> None:
        self.st, self.bits, self.n, self.sda_out = ADDR, 0, 0, True

    def stop(self) -> None:
        self.st, self.sda_out = IDLE, True

    def rise(self, sda: bool) -> None:
        if self.st in (ADDR, WRITE):
            self.bits = (self.bits << 1 | sda) & 0xFF
            self.n += 1
        elif self.st == ACK_R:
            self.mack = not sda

    def fall(self) -> None:
        st = self.st
        if st == ADDR and self.n == 8:
            if self.bits >> 1 == self.addr:
                self.rw, self.st, self.sda_out = self.bits & 1, ACK_ADDR, False
            else:
                self.st = IGNORE
        elif st == ACK_ADDR:
            if self.stretch:
                self.scl_out, self.stretch_left = False, self.stretch
            if self.rw:
                self.stream = bytes(self.regs.get(self.ptr, b""))
                self._next_byte()
            else:
                self.st, self.bits, self.n, self.nwrite, self.sda_out = WRITE, 0, 0, 0, True
        elif st == WRITE and self.n == 8:
            ok = self.byte_in(self.bits)
            self.st, self.sda_out = (ACK_W, False) if ok else (IGNORE, True)
        elif st == ACK_W:
            self.st, self.bits, self.n, self.sda_out = WRITE, 0, 0, True
        elif st == READ:
            if self.n < 8:
                self.sda_out = bool(self.byte >> (7 - self.n) & 1)
                self.n += 1
            else:
                self.st, self.sda_out = ACK_R, True
        elif st == ACK_R:
            if self.mack:
                self._next_byte()
            else:
                self.st, self.sda_out = IGNORE, True

    # ---- device behaviour
    def _next_byte(self) -> None:
        self.byte, self.stream = (self.stream[0], self.stream[1:]) if self.stream else (0xFF, b"")
        self.st, self.n = READ, 1
        self.sda_out = bool(self.byte >> 7 & 1)

    def byte_in(self, v: int) -> bool:
        """A written byte: the pointer (ACKed if the register exists), then data."""
        self.nwrite += 1
        if self.nwrite == 1:
            if v not in self.regs:
                return False
            self.ptr = v
            self.pointers.append(v)
            return True
        self.writes.append((self.ptr, v))
        return True


def lm73(addr: int = 0x49, temp_c: float = 41.25) -> Target:
    """An LM73: temperature (0x00, big endian, 1 C = 128), configuration, limits, ID 0x0190."""
    raw = round(temp_c * 128) & 0xFFFF
    return Target(addr, {0: raw.to_bytes(2, "big"), 1: b"\x40", 2: b"\x7f\xe0",
                         3: b"\x80\x00", 4: b"\x08", 7: LM73_ID.to_bytes(2, "big")})


def pmbus(addr: int = 0x60, vin: float = 12.0, vout: float = 0.95, iout: float = 20.0,
          pin: float | None = 21.5, temp_c: float = 55.0, vout_exp: int = -9,
          mfr_id: bytes = b"ACME", model: bytes = b"PMB1000", stretch: int = 0) -> Target:
    """A PMBus regulator (PMBus 1.2): identification, VOUT_MODE and the READ_* telemetry;
    pin None: no READ_PIN (the host falls back to READ_POUT)."""
    w = lambda x: to_linear11(x).to_bytes(2, "little")              # noqa: E731
    regs = {0x98: b"\x22", 0x99: bytes([len(mfr_id)]) + mfr_id,
            0x9A: bytes([len(model)]) + model, 0x20: bytes([vout_exp & 0x1F]),
            0x88: w(vin), 0x89: w(pin / vin if pin else 1.0),
            0x8B: round(vout / 2.0 ** vout_exp).to_bytes(2, "little"), 0x8C: w(iout),
            0x8D: w(temp_c), 0x96: w(vout * iout)}
    if pin is not None:
        regs[0x97] = w(pin)
    return Target(addr, regs, stretch)


def eeprom(addr: int = 0x50) -> Target:
    """A 256-byte EEPROM (24C02): byte a holds a ^ 0x5A; reads run on from the pointer."""
    mem = bytes(a ^ 0x5A for a in range(256))
    return Target(addr, {a: mem[a:] for a in range(256)})


class Glitch:
    """Another master: drives SDA low on the first SCL rise after a START (the host loses
    arbitration there if it sends a 1), and lets go after `hold` I2C_IN reads. Fires `times`
    times."""

    def __init__(self, hold: int = 5, times: int = 1):
        self.hold, self.times = hold, times
        self.sda_out = self.scl_out = True
        self.armed, self.left, self.fired = False, 0, 0

    def tick(self):
        if self.left:
            self.left -= 1
            if not self.left:
                self.sda_out = True

    def start(self):
        self.armed = self.fired < self.times

    def stop(self):
        pass

    def rise(self, sda):
        if self.armed:
            self.armed, self.sda_out, self.left = False, False, self.hold
            self.fired += 1

    def fall(self):
        pass


class OpenDrainBus:
    """One bus: the master's two drive-low bits, the slaves, and `hold` = (SCL, SDA) forced
    low from outside."""

    def __init__(self, targets=()):
        self.targets = list(targets)
        self.m_scl = self.m_sda = True          # released
        self.hold = [False, False]
        self.scl = self.sda = True
        self.edges = 0                          # SCL rises seen (a rough clock count)

    def _lines(self) -> tuple[bool, bool]:
        scl = self.m_scl and not self.hold[0] and all(t.scl_out for t in self.targets)
        sda = self.m_sda and not self.hold[1] and all(t.sda_out for t in self.targets)
        return scl, sda

    def settle(self) -> None:
        for _ in range(16):
            scl, sda = self._lines()
            if (scl, sda) == (self.scl, self.sda):
                return
            pscl, psda = self.scl, self.sda
            self.scl, self.sda = scl, sda
            for t in self.targets:
                if pscl and scl and psda and not sda:
                    t.start()
                elif pscl and scl and not psda and sda:
                    t.stop()
                elif not pscl and scl:
                    t.rise(sda)
                elif pscl and not scl:
                    t.fall()
            if not pscl and scl:
                self.edges += 1
        raise RuntimeError("the bus model does not settle")

    def drive(self, scl_lo: bool, sda_lo: bool) -> None:
        self.m_scl, self.m_sda = not scl_lo, not sda_lo
        self.settle()

    def read(self) -> tuple[bool, bool]:
        for t in self.targets:
            t.tick()
        self.settle()
        return self.scl, self.sda


def card_buses() -> list[OpenDrainBus]:
    """The buses of a demo card: an LM73 on the sensor bus; a PMBus regulator and an EEPROM on
    the SMBus. (What the real card has is what otpu-i2c scan finds.)"""
    return [OpenDrainBus([lm73()]), OpenDrainBus([pmbus(), eeprom()])]
