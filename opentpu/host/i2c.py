"""I2C over the card's open-drain pins (otpu_ctrl I2C_CTRL / I2C_IN, CAPS bit2): a bit-banged
master for the two buses the FPGA reaches, and the read-only LM73 and PMBus helpers that
otpu-diag, otpu-i2c and otpu-smi use.

    bus 0 "sensor"  the LM73 temperature sensor's bus (SCL N24, SDA N25, ALERT P25)
    bus 1 "smbus"   the PCIe edge connector's SMBus (SCL R26, SDA R27), which the host's own
                    SMBus controller may also drive

Timing. Every line change is a write of I2C_CTRL followed by a read of I2C_IN: the read
returns only after the posted write has reached the pins (PCIe ordering), gives the line
levels (clock stretching, arbitration) and paces the bus. With a PCIe read at ~1 us and the
default `half` of 5 us spun per half bit, SCL runs at roughly 50-80 kHz.

Safety. The API only reads. A transaction writes at most one byte to a device: the command
code (register pointer) of a read, followed by a repeated START. It never writes data, so it
cannot change a regulator's OPERATION, VOUT_COMMAND, PAGE or any other setting; a multi-page
PMBus device reports whichever page is selected. PMBus commands that act when sent alone
(CLEAR_FAULTS, STORE_ / RESTORE_*) are refused as command codes, so an interrupted read cannot
turn into one of them. No command code is ever written to 0x30-0x37 (DIMM SPD write-protect
and page commands, should the slot's SMBus reach the memory) or to the SMBus reserved
addresses 0x0C (alert response) and 0x61 (ARP). A scan only reads. Identification (identify)
also leaves alone the ranges of devices that take any written byte as data: I/O expanders
(PCF8574 0x20-0x27, PCF8574A 0x38-0x3F) and I2C multiplexers (PCA954x 0x70-0x77), and
EEPROMs (0x50-0x57); otpu-i2c can still read them when asked explicitly.

Sharing. The SMBus has another master. Before each START the lines must stay high for `idle`
seconds (the SMBus bus-idle time is 50 us); a busy bus waits, a lost arbitration (SDA low
while we release it, SCL high) releases both lines at once and retries after a back-off. All
processes serialize their transactions with an flock on <run dir>/<device>.i2c.lock (I2C_CTRL
holds both buses' bits).
"""
from __future__ import annotations

import fcntl
import json
import os
import random
import time
from pathlib import Path

from . import regs as R
from .runstate import ensure_run_dir, open_shared, read_shared, run_dir, write_shared

BUSES = {"sensor": 0, "smbus": 1}
SCAN_FIRST, SCAN_LAST = 0x08, 0x77
# addresses never sent a command code (see the module docstring)
NO_POINTER = frozenset(range(0x30, 0x38)) | {0x0C, 0x61}
# a scan lists them, identification sends them no command code (see the module docstring)
NO_IDENTIFY = {**{a: "eeprom?" for a in range(0x50, 0x58)},
               **{a: "I/O expander range" for a in [*range(0x20, 0x28), *range(0x38, 0x40)]},
               **{a: "I2C mux range" for a in range(0x70, 0x78)}}
# PMBus send-byte commands (they act on the command code alone): refused as command codes
SEND_BYTE = frozenset({0x03, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18})


class I2CError(IOError):
    pass


class Nack(I2CError):
    pass


class BusBusy(I2CError):
    """The bus did not go idle (another master), or arbitration was lost on every retry."""


class StuckBus(I2CError):
    """A line stays low with every driver released (SCL held beyond the stretch timeout, or
    SDA held low through bus recovery)."""


class _ArbitrationLost(Exception):
    pass


class Bus:
    """One of the card's I2C buses. `t` is a transport (XdmaTransport, FakeTransport) of a
    bitstream with CAPS.i2c. Public calls are whole transactions under the I2C lock."""

    def __init__(self, t, bus: int | str = 0, half: float = 5e-6, idle: float = 50e-6,
                 busy_timeout: float = 0.05, stretch: float = 0.035, retries: int = 5,
                 lock: bool = True, lock_timeout: float = 2.0):
        self.t, self.bus = t, BUSES.get(bus, bus)
        assert self.bus in (0, 1), bus
        self.half, self.idle, self.busy_timeout, self.stretch = half, idle, busy_timeout, stretch
        self.retries, self.lock_timeout = retries, lock_timeout
        self.scl_bit, self.sda_bit = 1 << 2 * self.bus, 2 << 2 * self.bus
        name = getattr(t, "devname", None)
        self.lock_path = run_dir() / f"{name}.i2c.lock" if lock and name else None
        self._lock_fd, self._depth = None, 0
        self._scl_lo = self._sda_lo = False
        self._other = 0                 # the other bus's I2C_CTRL bits, kept as found

    # ------------------------------------------------------------------ lock
    def __enter__(self):
        if self._depth == 0 and self.lock_path is not None:
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
                        raise BusBusy(f"I2C lock {self.lock_path} held by another process")
                    time.sleep(0.005)
            self._lock_fd = fd
        if self._depth == 0:
            self._other = self.t.reg_read(R.R_I2C_CTRL) & ~(self.scl_bit | self.sda_bit) & 0xF
            self._scl_lo = self._sda_lo = False
        self._depth += 1
        return self

    def __exit__(self, *exc):
        self._depth -= 1
        if self._depth == 0:
            self._drive(False, False, read=False)
            if self._lock_fd is not None:
                os.close(self._lock_fd)         # releases the flock
                self._lock_fd = None

    # ------------------------------------------------------------------ lines
    def levels(self) -> tuple[bool, bool]:
        """(SCL, SDA) as the pins read."""
        v = self.t.reg_read(R.R_I2C_IN)
        return bool(v & self.scl_bit), bool(v & self.sda_bit)

    def alert(self) -> bool:
        """The LM73's ALERT pin is asserted (it is low; bus 0 only)."""
        return not self.t.reg_read(R.R_I2C_IN) & R.I2C_ALERT0

    def _drive(self, scl_lo: bool, sda_lo: bool, read: bool = True):
        self._scl_lo, self._sda_lo = scl_lo, sda_lo
        self.t.reg_write(R.R_I2C_CTRL, self._other | (self.scl_bit if scl_lo else 0)
                         | (self.sda_bit if sda_lo else 0))
        return self.levels() if read else None

    def _wait(self) -> None:
        if self.half > 0:
            end = time.perf_counter() + self.half
            while time.perf_counter() < end:
                pass

    def _scl_high(self, sda_lo: bool) -> tuple[bool, bool]:
        """Release SCL (SDA as given) and wait out a slave's clock stretching."""
        scl, sda = self._drive(False, sda_lo)
        if not scl:
            deadline = time.perf_counter() + self.stretch
            while not scl:
                if time.perf_counter() > deadline:
                    self._drive(False, False, read=False)
                    raise StuckBus(f"bus {self.bus}: SCL held low for {self.stretch * 1e3:g} ms")
                scl, sda = self.levels()
        return scl, sda

    def _release(self) -> None:
        self._drive(False, False, read=False)

    # ------------------------------------------------------------------ bus conditions
    def wait_idle(self) -> None:
        """Both lines high for `idle` seconds. SCL low throughout the busy timeout: StuckBus.
        SDA low with SCL high throughout: bus recovery (a slave left mid-byte), then StuckBus
        if SDA is still low. Otherwise traffic: BusBusy."""
        t0 = time.perf_counter()
        since, ever_scl_hi, ever_sda_hi = None, False, False
        while True:
            scl, sda = self.levels()
            now = time.perf_counter()
            ever_scl_hi |= scl
            ever_sda_hi |= sda
            if scl and sda:
                since = now if since is None else since
                if now - since >= self.idle:
                    return
            else:
                since = None
            if now - t0 > self.busy_timeout:
                if not ever_scl_hi:
                    raise StuckBus(f"bus {self.bus}: SCL low (no pull-up, or a device holds it)")
                if not ever_sda_hi:
                    if self.recover():
                        return
                    raise StuckBus(f"bus {self.bus}: SDA low after 9 recovery clocks")
                raise BusBusy(f"bus {self.bus}: not idle for {self.busy_timeout * 1e3:g} ms")

    def recover(self) -> bool:
        """Up to 9 SCL pulses with SDA released until SDA reads high (a slave finishes the byte
        it was sending), then a STOP. True if both lines end high."""
        with self:
            for _ in range(9):
                if self._drive(False, False)[1]:
                    break
                self._drive(True, False)
                self._wait()
                self._scl_high(False)
                self._wait()
            self._drive(True, False)            # STOP: SDA low under SCL low, SCL up, SDA up
            self._wait()
            self._drive(True, True)
            self._wait()
            self._scl_high(True)
            self._wait()
            scl, sda = self._drive(False, False)
            self._wait()
            return scl and sda

    def _start(self) -> None:
        self.wait_idle()
        self._drive(False, True)
        self._wait()
        self._drive(True, True)
        self._wait()

    def _restart(self) -> None:
        self._drive(True, False)
        self._wait()
        self._scl_high(False)
        self._wait()
        if not self.levels()[1]:
            raise _ArbitrationLost
        self._drive(False, True)
        self._wait()
        self._drive(True, True)
        self._wait()

    def _stop(self) -> None:
        self._drive(True, True)
        self._wait()
        self._scl_high(True)
        self._wait()
        self._drive(False, False)
        self._wait()                            # bus free time before any next START

    def _tx_bit(self, one: bool) -> None:
        self._drive(True, not one)
        self._wait()
        self._scl_high(not one)
        self._wait()
        if one and not self.levels()[1]:
            raise _ArbitrationLost
        self._drive(True, not one)

    def _rx_bit(self) -> bool:
        self._drive(True, False)
        self._wait()
        self._scl_high(False)
        self._wait()
        sda = self.levels()[1]
        self._drive(True, False)
        return sda

    def _tx(self, byte: int) -> bool:
        """One byte out, MSB first; True if the slave acknowledged."""
        for k in range(7, -1, -1):
            self._tx_bit(bool(byte >> k & 1))
        return not self._rx_bit()

    def _rx(self, ack: bool) -> int:
        v = 0
        for _ in range(8):
            v = v << 1 | self._rx_bit()
        self._tx_bit(not ack)
        return v

    # ------------------------------------------------------------------ transactions
    def _attempt(self, fn):
        """fn() inside START ... STOP, retried on a lost arbitration (both lines released at
        once, then a random back-off)."""
        with self:
            for k in range(self.retries):
                try:
                    self._start()
                    return fn()
                except _ArbitrationLost:
                    self._release()
                    time.sleep(random.uniform(1, 2) * 1e-4 * (k + 1))
                except I2CError:
                    self._release()
                    raise
            raise BusBusy(f"bus {self.bus}: arbitration lost {self.retries} times")

    def probe(self, addr: int, quick_write: bool = False) -> bool:
        """Does `addr` acknowledge? A read of one byte (NACKed, so the slave lets SDA go), or
        with quick_write an SMBus quick command (address + W, no data)."""
        def fn():
            ack = self._tx(addr << 1 | (0 if quick_write else 1))
            if ack and not quick_write:
                self._rx(ack=False)
            self._stop()
            return ack
        return self._attempt(fn)

    def read(self, addr: int, cmd: int | None, n: int, block: bool = False) -> bytes:
        """`n` bytes from `addr`: the command code `cmd` (register pointer) written first when
        given, then a repeated START and a read. block: an SMBus block read (the first byte is
        the count, at most 32; n is ignored). Nack if the address or the command is refused."""
        if cmd is not None:
            if addr in NO_POINTER:
                raise ValueError(f"no command codes are written to {addr:#04x} (see i2c.py)")
            if cmd in SEND_BYTE:
                raise ValueError(f"command {cmd:#04x} acts when sent alone (PMBus send byte)")

        def fn():
            if cmd is not None:
                if not self._tx(addr << 1):
                    self._stop()
                    raise Nack(f"{addr:#04x}: no ACK")
                if not self._tx(cmd):
                    self._stop()
                    raise Nack(f"{addr:#04x}: command {cmd:#04x} not acknowledged")
                self._restart()
            if not self._tx(addr << 1 | 1):
                self._stop()
                raise Nack(f"{addr:#04x}: no ACK to the read")
            if block:
                cnt = self._rx(ack=True)
                m = min(cnt, 32)
                out = bytes(self._rx(ack=i < m - 1) for i in range(m))
                if m == 0:
                    self._rx(ack=False)
                self._stop()
                return out
            out = bytes(self._rx(ack=i < n - 1) for i in range(n))
            self._stop()
            return out
        return self._attempt(fn)

    def read_byte(self, addr: int, cmd: int) -> int:
        return self.read(addr, cmd, 1)[0]

    def read_word(self, addr: int, cmd: int) -> int:
        """SMBus read word: little endian (PMBus)."""
        b = self.read(addr, cmd, 2)
        return b[0] | b[1] << 8

    def read_block(self, addr: int, cmd: int) -> bytes:
        return self.read(addr, cmd, 0, block=True)

    def scan(self, first: int = SCAN_FIRST, last: int = SCAN_LAST,
             quick_write: bool = False) -> list[int]:
        """The addresses that acknowledge a probe (by default a one-byte read)."""
        with self:
            return [a for a in range(first, last + 1) if self.probe(a, quick_write)]


# ------------------------------------------------------------------------------ LM73
LM73_ADDRS = (0x48, 0x49, 0x4A, 0x4C, 0x4D, 0x4E)
LM73_ID = 0x0190                         # ID register 0x07 (manufacturer / device ID)


def lm73_temp(raw: int) -> float:
    """LM73 temperature register (16 bits, two's complement, 1 C = 128) -> degrees Celsius."""
    return (raw - (1 << 16) if raw & 0x8000 else raw) / 128


def lm73_identify(bus: Bus, addr: int) -> dict | None:
    """{"kind": "lm73", "temp_c"} if `addr` is an LM73 (its ID register reads 0x0190). The
    pointer is left at the temperature register (read last), where the LM73 powers up."""
    try:
        ident = int.from_bytes(bus.read(addr, 0x07, 2), "big")
        if ident != LM73_ID:
            return None
        return {"kind": "lm73", "id": ident,
                "temp_c": lm73_temp(int.from_bytes(bus.read(addr, 0x00, 2), "big"))}
    except Nack:
        return None


# ------------------------------------------------------------------------------ INA2xx
# TI current / power monitors (0x40-0x4F): manufacturer ID 0x5449 ("TI") at 0xFE, the die ID
# at 0xFF. They measure a shunt's voltage; current and power need the shunt's resistance (and a
# calibration write), which the board does not document, so only the voltages are reported.
INA_ADDRS = range(0x40, 0x50)
INA_DIE = {0x2260: "INA226", 0x3220: "INA3221", 0x2270: "INA230/231"}


def ina_identify(bus: Bus, addr: int) -> dict | None:
    try:
        if int.from_bytes(bus.read(addr, 0xFE, 2), "big") != 0x5449:
            return None
        die = int.from_bytes(bus.read(addr, 0xFF, 2), "big")
        shunt = int.from_bytes(bus.read(addr, 0x01, 2), "big")
        vbus = int.from_bytes(bus.read(addr, 0x02, 2), "big")
    except Nack:
        return None
    return {"kind": "ina", "model": INA_DIE.get(die, f"TI die {die:#06x}"),
            "shunt_uv": (shunt - (1 << 16) if shunt & 0x8000 else shunt) * 2.5,
            "vbus": vbus * 1.25e-3}                  # INA226 LSBs


# ------------------------------------------------------------------------------ PMBus
PMBUS = {"VOUT_MODE": 0x20, "READ_VIN": 0x88, "READ_IIN": 0x89, "READ_VOUT": 0x8B,
         "READ_IOUT": 0x8C, "READ_TEMPERATURE_1": 0x8D, "READ_POUT": 0x96, "READ_PIN": 0x97,
         "PMBUS_REVISION": 0x98, "MFR_ID": 0x99, "MFR_MODEL": 0x9A}
# telemetry: name -> (command, unit); all LINEAR11 but READ_VOUT (LINEAR16 with VOUT_MODE)
TELEMETRY = {"vin": ("READ_VIN", "V"), "iin": ("READ_IIN", "A"), "vout": ("READ_VOUT", "V"),
             "iout": ("READ_IOUT", "A"), "pin": ("READ_PIN", "W"), "pout": ("READ_POUT", "W"),
             "temp_c": ("READ_TEMPERATURE_1", "C")}


def linear11(v: int) -> float:
    """PMBus LINEAR11: a 5-bit exponent over an 11-bit mantissa, both two's complement."""
    e, m = v >> 11 & 0x1F, v & 0x7FF
    return (m - 2048 if m & 0x400 else m) * 2.0 ** (e - 32 if e & 0x10 else e)


def to_linear11(x: float) -> int:
    """The inverse (the finest exponent whose mantissa fits): for device models."""
    for e in range(-16, 16):
        m = round(x / 2.0 ** e)
        if -1024 <= m < 1024:
            return (e & 0x1F) << 11 | (m & 0x7FF)
    raise ValueError(x)


def linear16(v: int, vout_mode: int) -> float | None:
    """READ_VOUT in LINEAR16: the unsigned mantissa v times 2^(VOUT_MODE[4:0], signed). None
    for the VID and direct modes (VOUT_MODE[7:5] != 0)."""
    if vout_mode >> 5 & 0x7:
        return None
    e = vout_mode & 0x1F
    return v * 2.0 ** (e - 32 if e & 0x10 else e)


def _text(b: bytes | None) -> str | None:
    if not b:
        return None
    s = b.decode("ascii", "replace")
    return s if all(32 <= c < 127 for c in b) else None


def pmbus_identify(bus: Bus, addr: int) -> dict | None:
    """PMBUS_REVISION, MFR_ID, MFR_MODEL and VOUT_MODE of `addr`, or None if it refuses
    PMBUS_REVISION. "pmbus" is True when the answers look like a PMBus device's (both revision
    nibbles 0..3 -- PMBus 1.0 .. 1.3 -- and a printable MFR_ID or a linear VOUT_MODE); other
    devices often acknowledge any command code and return whatever their registers hold."""
    try:
        rev = bus.read_byte(addr, PMBUS["PMBUS_REVISION"])
    except Nack:
        return None
    d = {"kind": "pmbus?", "revision": rev, "mfr_id": None, "mfr_model": None,
         "vout_mode": None}
    for k in ("MFR_ID", "MFR_MODEL"):
        try:
            d[k.lower()] = _text(bus.read_block(addr, PMBUS[k]))
        except Nack:
            pass
    try:
        d["vout_mode"] = bus.read_byte(addr, PMBUS["VOUT_MODE"])
    except Nack:
        pass
    ok_rev = rev >> 4 <= 3 and rev & 0xF <= 3
    ok_mode = d["vout_mode"] is not None and d["vout_mode"] >> 5 == 0
    if ok_rev and (d["mfr_id"] or ok_mode):
        d["kind"] = "pmbus"
    return d


def pmbus_telemetry(bus: Bus, addr: int, vout_mode: int | None = None) -> dict:
    """The READ_* values of `addr` in volts / amperes / watts / C (None where refused), and
    "power": {"w", "how"} -- input power (READ_PIN) when the device has it, else READ_POUT,
    else VOUT x IOUT -- or None."""
    d = {}
    for k, (cmd, _) in TELEMETRY.items():
        try:
            w = bus.read_word(addr, PMBUS[cmd])
        except Nack:
            d[k] = None
            continue
        if k == "vout":
            d[k] = linear16(w, vout_mode) if vout_mode is not None else None
        else:
            d[k] = linear11(w)
    if d["pin"] is not None:
        d["power"] = {"w": d["pin"], "how": "PIN"}
    elif d["pout"] is not None:
        d["power"] = {"w": d["pout"], "how": "POUT"}
    elif d["vout"] is not None and d["iout"] is not None:
        d["power"] = {"w": d["vout"] * d["iout"], "how": "VOUT x IOUT"}
    else:
        d["power"] = None
    return d


# ------------------------------------------------------------------------------ discovery
def identify(bus: Bus, addr: int) -> dict:
    """What answers at `addr`: an LM73, a TI current monitor, a PMBus device, or unknown.
    Reserved addresses and the NO_IDENTIFY ranges are listed without a command code."""
    if addr in NO_POINTER:
        return {"addr": addr, "kind": "reserved (not probed further)"}
    if addr in NO_IDENTIFY:
        return {"addr": addr, "kind": NO_IDENTIFY[addr]}
    if addr in LM73_ADDRS:
        d = lm73_identify(bus, addr)
        if d:
            return {"addr": addr, **d}
    if addr in INA_ADDRS:
        d = ina_identify(bus, addr)
        if d:
            return {"addr": addr, **d}
    d = pmbus_identify(bus, addr)
    if d is None:
        return {"addr": addr, "kind": "unknown"}
    if d["kind"] == "pmbus":
        d["telemetry"] = pmbus_telemetry(bus, addr, d["vout_mode"])
    return {"addr": addr, **d}


def discover(t, buses=("sensor", "smbus"), **kw) -> dict:
    """Scan each bus and identify what answers: {"buses": {name: {"ok", "error", "devices"}}}.
    kw: Bus options."""
    out = {"version": 1, "time": time.time(), "buses": {}}
    for name in buses:
        b = Bus(t, name, **kw)
        r = {"ok": True, "error": None, "devices": []}
        try:
            with b:
                for a in b.scan():
                    r["devices"].append(identify(b, a))
        except I2CError as e:
            r.update(ok=False, error=f"{type(e).__name__}: {e}")
        out["buses"][name] = r
    return out


def cache_path(t) -> Path | None:
    name = getattr(t, "devname", None)
    return run_dir() / f"{name}.i2c.json" if name else None


def save_discovery(t, d: dict) -> None:
    p = cache_path(t)
    if p is not None:
        try:
            ensure_run_dir()
            write_shared(p, json.dumps(d, indent=1))
        except (OSError, RuntimeError):
            pass                        # (a cache: discovered again next time)


def load_discovery(t) -> dict | None:
    p = cache_path(t)
    try:
        d = json.loads(read_shared(p)) if p else None
    except (OSError, ValueError):
        return None
    return d if d and d.get("version") == 1 else None


def power_devices(disc: dict) -> list[tuple[str, int, int | None]]:
    """(bus, addr, VOUT_MODE) of every PMBus device in a discovery that reported power."""
    return [(name, d["addr"], d.get("vout_mode")) for name, r in disc["buses"].items()
            for d in r["devices"]
            if d["kind"] == "pmbus" and (d.get("telemetry") or {}).get("power")]


def measured(t, disc: dict | None = None, **kw) -> dict | None:
    """Board power and temperature from the I2C devices otpu-smi can read: {"w", "how",
    "rails": [{bus, addr, w, how}], "board_temp_c"}, "w" None without a PMBus device that
    reports power. The discovery comes from `disc`, else the cache (<run dir>/<dev>.i2c.json),
    else a fresh scan (saved). None when the I2C lock is busy."""
    if disc is None:
        disc = load_discovery(t)
        if disc is None:
            disc = discover(t, **kw)
            save_discovery(t, disc)
    rails, temp = [], None
    try:
        for name, r in disc["buses"].items():
            b = Bus(t, name, **kw)
            for d in r["devices"]:
                if d["kind"] == "lm73" and temp is None:
                    temp = lm73_identify(b, d["addr"])
                    temp = temp and temp["temp_c"]
        for name, addr, mode in power_devices(disc):
            p = pmbus_telemetry(Bus(t, name, **kw), addr, mode)["power"]
            if p:
                rails.append({"bus": name, "addr": addr, **p})
    except BusBusy:
        return None
    except I2CError:
        pass
    return {"w": sum(r["w"] for r in rails) if rails else None,
            "how": "+".join(sorted({r["how"] for r in rails})) or None, "rails": rails,
            "board_temp_c": temp}


# ------------------------------------------------------------------------------ otpu-i2c
def _int(s: str) -> int:
    return int(s, 0)


def main(argv=None, open_transport=None) -> int:
    """otpu-i2c: the card's I2C buses (read only)."""
    import argparse
    import sys
    ap = argparse.ArgumentParser(
        prog="otpu-i2c", description="The card's I2C buses, read only: scan, read, PMBus dump. "
        "Bus 'sensor' is the LM73's (N24 / N25), 'smbus' the PCIe edge connector's (R26 / R27).")
    ap.add_argument("--dev", default="/dev/xdma0")
    ap.add_argument("--fake", action="store_true", help="a demo card (fake_i2c.card_buses)")
    ap.add_argument("--khz", type=float, default=100.0,
                    help="SCL rate bound (half bit spun; register accesses add to it)")
    ap.add_argument("--json", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("scan", help="probe 0x08-0x77 (one-byte reads) and identify devices")
    sp.add_argument("--bus", choices=[*BUSES, "all"], default="all")
    sp.add_argument("--no-identify", action="store_true", help="probe only, no command codes")
    sp = sub.add_parser("read", help="read bytes: [CMD] as a register pointer, then -n bytes")
    sp.add_argument("bus", choices=list(BUSES))
    sp.add_argument("addr", type=_int)
    sp.add_argument("reg", type=_int, nargs="?", help="command code / register (none: plain read)")
    sp.add_argument("-n", type=int, default=1)
    sp.add_argument("--block", action="store_true", help="SMBus block read (count byte first)")
    sp = sub.add_parser("pmbus-dump", help="PMBus identification and READ_* telemetry")
    sp.add_argument("bus", choices=list(BUSES))
    sp.add_argument("addr", type=_int)
    sub.add_parser("levels", help="the pin levels (SCL / SDA of both buses, ALERT)")
    a = ap.parse_args(argv)

    if open_transport:
        t = open_transport(a.dev)
    elif a.fake:
        from .fake import FakeTransport
        from .fake_i2c import card_buses
        t = FakeTransport(devname=None, i2c=card_buses())
    else:
        from .board import XdmaTransport
        t = XdmaTransport(a.dev, dma=False)
    if t.reg_read(R.R_ID) != R.ID_OTPU:
        print("otpu-i2c: no openTPU on the card", file=sys.stderr)
        return 1
    if R.regmap(t.reg_read(R.R_REGMAP)) < 2 or not t.reg_read(R.R_CAPS) & R.CAP_I2C:
        print("otpu-i2c: this bitstream has no I2C pins (CAPS bit2)", file=sys.stderr)
        return 1
    kw = {"half": 0.5 / (a.khz * 1e3)}
    out = sys.stdout

    def emit(obj, text):
        print(json.dumps(obj, indent=1) if a.json else text, file=out)

    try:
        if a.cmd == "levels":
            v = t.reg_read(R.R_I2C_IN)
            lv = {"sensor": {"scl": bool(v & 1), "sda": bool(v & 2)},
                  "smbus": {"scl": bool(v & 4), "sda": bool(v & 8)},
                  "alert0": not v & R.I2C_ALERT0}
            emit(lv, "\n".join(f"{n:<7} SCL {int(x['scl'])} SDA {int(x['sda'])}"
                               for n, x in list(lv.items())[:2])
                 + f"\nALERT0 {'asserted (low)' if lv['alert0'] else 'released (high)'}")
        elif a.cmd == "scan":
            names = list(BUSES) if a.bus == "all" else [a.bus]
            if a.no_identify:
                res = {}
                for n in names:
                    try:
                        res[n] = {"ok": True, "addrs": Bus(t, n, **kw).scan()}
                    except I2CError as e:
                        res[n] = {"ok": False, "error": str(e)}
                emit(res, "\n".join(f"{n}: " + (" ".join(f"{x:#04x}" for x in r["addrs"])
                                                 or "no device") if r["ok"] else
                                    f"{n}: {r['error']}" for n, r in res.items()))
            else:
                disc = discover(t, names, **kw)
                if set(names) == set(BUSES):
                    save_discovery(t, disc)
                lines = []
                for n, r in disc["buses"].items():
                    lines.append(f"{n}:" + ("" if r["ok"] else f" {r['error']}"))
                    for dv in r["devices"]:
                        lines.append("  " + _dev_line(dv))
                    if r["ok"] and not r["devices"]:
                        lines.append("  no device")
                emit(disc, "\n".join(lines))
        elif a.cmd == "read":
            b = Bus(t, a.bus, **kw)
            data = b.read(a.addr, a.reg, a.n, block=a.block)
            emit({"addr": a.addr, "reg": a.reg, "data": list(data)}, data.hex(" "))
        elif a.cmd == "pmbus-dump":
            b = Bus(t, a.bus, **kw)
            ident = pmbus_identify(b, a.addr)
            if ident is None:
                print(f"otpu-i2c: {a.addr:#04x} refuses PMBUS_REVISION", file=sys.stderr)
                return 1
            tel = pmbus_telemetry(b, a.addr, ident["vout_mode"])
            units = {k: u for k, (_, u) in TELEMETRY.items()}
            text = [_dev_line({"addr": a.addr, **ident}),
                    f"  revision {ident['revision']:#04x}  VOUT_MODE "
                    + (f"{ident['vout_mode']:#04x}" if ident["vout_mode"] is not None else "n/a")]
            text += [f"  {k:<7} " + ("n/a" if v is None else f"{v:.4g} {units[k]}")
                     for k, v in tel.items() if k != "power"]
            p = tel["power"]
            text.append(f"  power   {p['w']:.4g} W ({p['how']})" if p else "  power   n/a")
            emit({"addr": a.addr, **ident, "telemetry": tel}, "\n".join(text))
    except (I2CError, ValueError) as e:
        print(f"otpu-i2c: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


def _dev_line(d: dict) -> str:
    a, k = f"{d['addr']:#04x}", d["kind"]
    if k == "lm73":
        return f"{a}  LM73 temperature sensor  {d['temp_c']:.2f} C"
    if k == "ina":
        return f"{a}  {d['model']} current monitor  bus {d['vbus']:.3f} V  shunt {d['shunt_uv']:.1f} uV"
    if k in ("pmbus", "pmbus?"):
        name = " ".join(x for x in (d.get("mfr_id"), d.get("mfr_model")) if x) or "?"
        s = f"{a}  PMBus {name}" + ("" if k == "pmbus" else " (unconfirmed: odd identification)")
        p = (d.get("telemetry") or {}).get("power")
        return s + (f"  {p['w']:.2f} W ({p['how']})" if p else "")
    return f"{a}  {k}"


if __name__ == "__main__":
    import sys
    sys.exit(main())
