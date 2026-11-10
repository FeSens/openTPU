"""The BCU1525's thermal governor (otpu_bcu_gov, rtl/boards/bcu1525/otpu_bcu_gov.sv; BAR0 or the
JTAG image's AXI at R_GOV): the fabric holds each DDR4 controller and the accelerator in reset
until the host asks GO, and takes them back (SAFE) by itself when the die reaches TRIP, SYSMON
raises its temperature alarm or over-temperature, the die's reading is lost, or the host's
heartbeat (KICK) stops for HB_MS. GO is refused above RESUME (the die cools first) and while an
alarm stands. TRIP and RESUME clamp at the build's limits (LIMITS: 80 and 70 C); SYSMON's own
over-temperature shutdown (84 C) clears the configuration under all of it (docs/bcu1525.md
section 14). A governor with the regulator (STATUS HAS_REG: the BMC bridge's watch reads VCCINT's
regulator for it) also refuses GO above REG_RESUME or without a reading within REG_STALE, and goes
SAFE at REG_TRIP or once its reading is REG_STALE old (REG_LIMITS: 78 and 75 C).

The die's temperatures are SYSMONE4's 10-bit codes (UG580: code x 509.3140064 / 1024 -
280.2308787), the regulator's quarter degrees C (signed 16 bits). sysmon() reads the SYSMON's own
registers beside the governor (BAR0 R_SYSMON): the die, its supplies and its alarms. The
governor's TEMP comes another way, the System Management Wizard's temperature bus into the
fabric, and an AXI read of the SYSMON's DRP registers (R_SYSMON + 0x400 and up) can stop that bus
for good: the wizard's DRP arbiter drops the bus's own read when the AXI read lands in the cycle
before it, about one read in 1000 (system_management_wiz v1.3; f150m15 on the card, 2026-10-29:
frozen at 46.05 C from wg_bringup's five reads at 06:04:17;
docs/bcu1525_sysmon.md). A monitor reads the register only when it
must: sysmon(full=False) is one read.
"""
from __future__ import annotations

import contextlib

R_GOV = 0x10000
ID, STATUS, CTRL, CMD, TEMP, TRIP, RESUME, HB_MS, KICK, TRIPS, RUN_MS, LIMITS = range(0, 0x30, 4)
REG, REG_TRIP, REG_RESUME, REG_STALE, REG_AGE, REG_LIMITS = range(0x30, 0x48, 4)
TEMP_AGE, TEMP_STALE = 0x48, 0x4C           # the code's age (ms) and the build's limit; 0: none
GOV_ID = 0x474F5631                         # "GOV1"
# the image's identity, answered in SAFE too (a build with them: IMAGE reads IMG_ID, else 0)
IMAGE, IMG_PLAT, IMG_HASH, IMG_BUILD, IMG_KHZ = range(0x50, 0x64, 4)
IMG_ID = 0x494D4731                         # "IMG1"
CMD_GO, CMD_SAFE, CMD_CLEAR = 1, 2, 4

# STATUS
ST_RUN, ST_WARN, ST_COOL, ST_HB_OK, ST_TEMP_VALID, ST_ALARM, ST_OT, ST_HAS_REG = \
    (1 << b for b in range(8))
ST_ACC_ON, ST_REG_OK, ST_REG_COOL, ST_REG_WARN = (1 << b for b in range(12, 16))
ST_THERMAL = 1 << 31
CAUSES = ("hot", "temperature alarm", "no heartbeat", "host SAFE", "over-temperature or no reading",
          "regulator hot", "regulator unread", "temperature stale")
TEMP_AGE_STALE = 1 << 31                    # TEMP_AGE: TEMP_STALE_MS reached (TEMP_VALID clear)

# CTRL
CTRL_ACC, CTRL_UART_SHIFT = 1 << 4, 8

_GAIN, _OFFSET = 509.3140064 / 1024, 280.2308787

# The SYSMON (the block design's System Management Wizard, AXI4-Lite) at BAR0 R_SYSMON: UG580's
# DRP registers at 0x400 + 4 x their address, 16-bit codes with the 10-bit result in [15:6]
R_SYSMON = 0x30000
SM_TEMP, SM_VCCINT, SM_VCCAUX, SM_VBRAM = 0x400, 0x404, 0x408, 0x418
SM_MAX_TEMP, SM_MAX_VCCINT, SM_MIN_TEMP, SM_MIN_VCCINT = 0x480, 0x484, 0x490, 0x494
SM_FLAG = 0x4FC                     # bit0 ALM[0] (the temperature alarm), bit3 OT
SM_TEMP_UP, SM_OT_UP, SM_TEMP_LO, SM_OT_LO = 0x540, 0x54C, 0x550, 0x55C
SM_FLAG_TEMP, SM_FLAG_OT = 1, 8
GOV_GAP_C = 5.0                     # the governor's die this far from the register's: a note
FROZEN_S = 60.0                     # a live die reading moves within this (fan_pid.py's --stale-s)


def celsius(code: int) -> float:
    return (code & 0x3FF) * _GAIN - _OFFSET


def code_of(c: float) -> int:
    return round((c + _OFFSET) / _GAIN)


def quarter_c(v: int) -> float:
    """The regulator's 16-bit quarter degrees (signed) in C."""
    v &= 0xFFFF
    return (v - 0x10000 if v & 0x8000 else v) / 4


def sysmon_c(v: int) -> float | None:
    """A SYSMON temperature register's word in C (the 10-bit code in [15:6]), or None when it reads
    as no SYSMON: zero, all ones, bits above the 16-bit register set, or out of -40 .. 125 C
    (fan_pid.py's rule, with the upper half)."""
    if v in (0, 0xFFFFFFFF) or v >> 16:
        return None
    c = celsius(v >> 6)
    return c if -40.0 <= c <= 125.0 else None


def sysmon(t, base: int = R_SYSMON, full: bool = True) -> dict:
    """The SYSMON's own readings, registers only: the die now ("die_c") and its highest and lowest
    since configuration, VCCINT, VCCAUX and VBRAM at the die (V; its supply sensors: code x 3 V /
    2^16), the temperature alarm's and OT's thresholds (set at *_up_c, cleared at *_lo_c) and the
    alarms standing (FLAG); 13 DRP reads. `full` False: the die alone, one read (the rest None).
    "valid" False (every reading None, no alarm) when the die's register reads as no SYSMON
    (sysmon_c). Each read risks the wizard's temperature bus (the docstring above)."""
    offs = (SM_TEMP, SM_VCCINT, SM_VCCAUX, SM_VBRAM, SM_MAX_TEMP, SM_MAX_VCCINT, SM_MIN_TEMP,
            SM_MIN_VCCINT, SM_FLAG, SM_TEMP_UP, SM_OT_UP, SM_TEMP_LO, SM_OT_LO)
    rd = offs if full else (SM_TEMP,)
    v = dict.fromkeys(offs, 0)
    v.update(zip(rd, t.reg_read_many([base + o for o in rd])))
    ok = sysmon_c(v[SM_TEMP]) is not None

    def c(o):
        x = sysmon_c(v[o]) if ok else None
        return None if x is None else round(x, 2)

    def volt(o):
        return round(v[o] * 3.0 / 65536, 4) if ok and 0 < v[o] <= 0xFFFF else None
    return {"valid": ok, "full": full, "die_c": c(SM_TEMP), "max_c": c(SM_MAX_TEMP),
            "min_c": c(SM_MIN_TEMP),
            "vccint": volt(SM_VCCINT), "vccint_max": volt(SM_MAX_VCCINT),
            "vccint_min": volt(SM_MIN_VCCINT), "vccaux": volt(SM_VCCAUX), "vbram": volt(SM_VBRAM),
            "alarm_up_c": c(SM_TEMP_UP), "alarm_lo_c": c(SM_TEMP_LO), "ot_up_c": c(SM_OT_UP),
            "ot_lo_c": c(SM_OT_LO), "alarm": ok and bool(v[SM_FLAG] & SM_FLAG_TEMP),
            "ot": ok and bool(v[SM_FLAG] & SM_FLAG_OT), "flag": v[SM_FLAG] if full else None}


def die_readings(g: "Gov", mon=None) -> dict:
    """The die without the SYSMON's DRP registers over AXI (a read of them can stop the
    governor's die, the docstring above): the governor's code ("gov_c", "gov_valid"), its
    temperature alarm and OT lines (the SYSMON's ALM[0] and OT, straight from the hard block),
    the code's age where the build has TEMP_AGE ("age_ms") and, with `mon` (a JtagAxi's: the
    SYSMON's JTAG port, beside the wizard's arbiter), the SYSMON's own reading ("jtag_c",
    "vccint", "vccaux", "vccbram"; "jtag_error" when it fails)."""
    s = g.status()
    d = {"gov_c": round(s["temp_c"], 2), "gov_valid": s["temp_valid"], "alarm": s["alarm"],
         "ot": s["ot"], "age_ms": s["temp_age_ms"]}
    if mon is not None:
        try:
            m = mon()
            d.update(jtag_c=round(m["t"], 2), vccint=m["vccint"], vccaux=m["vccaux"],
                     vccbram=m["vccbram"])
        except OSError as e:
            d["jtag_error"] = str(e)
    return d


class Gov:
    """The governor through a transport's register reads and writes (reg_read, reg_write)."""

    def __init__(self, t, base: int = R_GOV):
        self.t, self.base = t, base

    def rd(self, off: int) -> int:
        return self.t.reg_read(self.base + off)

    def wr(self, off: int, v: int) -> None:
        self.t.reg_write(self.base + off, v)

    def present(self) -> bool:
        return self.rd(ID) == GOV_ID

    def image(self) -> dict | None:
        """The image's identity as the governor holds it (it answers in SAFE, when the
        accelerator's own registers are fenced): "platform" (CFG_PLATFORM), this card's
        "slices", its machine's "cards" (0: a top without an accelerator), "cfg_hash"
        (CFG_HASH), "build_id" and "core_khz". None on a build before these registers (IMAGE
        reads 0 there)."""
        if self.rd(IMAGE) != IMG_ID:
            return None
        p = self.rd(IMG_PLAT)
        return {"platform": p & 0xFFFF, "slices": p >> 16 & 0xFF, "cards": p >> 24,
                "cfg_hash": self.rd(IMG_HASH), "build_id": self.rd(IMG_BUILD),
                "core_khz": self.rd(IMG_KHZ)}

    def status(self) -> dict:
        """The state; with HAS_REG also the regulator's ("reg_c", its highest "reg_max_c" since
        CLEAR, the reading's age "reg_age_ms"; reg_ok / reg_cool / reg_warn are True without).
        On a build with the governor's staleness check (TEMP_STALE non-zero) the die's code's
        age: "temp_age_ms" (ms since it last changed, counted while the temperature alarm is
        clear), "temp_stale" (TEMP_STALE_MS reached: TEMP_VALID clear, the wizard's temperature
        bus stopped) and "temp_stale_ms"; None, False and None before it."""
        s, tmp = self.rd(STATUS), self.rd(TEMP)
        trips = self.rd(TRIPS)
        d = {"run": bool(s & ST_RUN), "warn": bool(s & ST_WARN), "cool": bool(s & ST_COOL),
             "hb_ok": bool(s & ST_HB_OK), "temp_valid": bool(s & ST_TEMP_VALID),
             "alarm": bool(s & ST_ALARM), "ot": bool(s & ST_OT),
             "dimms_on": [d for d in range(4) if s >> (8 + d) & 1],
             "acc_on": bool(s & ST_ACC_ON),
             "causes": [c for i, c in enumerate(CAUSES) if s >> (16 + i) & 1],
             "thermal": bool(s & ST_THERMAL),
             "temp_c": celsius(tmp), "max_c": celsius(tmp >> 16),
             "trips_thermal": trips & 0xFFFF, "trips_heartbeat": trips >> 16,
             "run_ms": self.rd(RUN_MS), "has_reg": bool(s & ST_HAS_REG),
             "reg_ok": True, "reg_cool": True, "reg_warn": False,
             "temp_stale_ms": self.rd(TEMP_STALE) or None, "temp_age_ms": None, "temp_stale": False}
        if d["temp_stale_ms"]:
            a = self.rd(TEMP_AGE)
            d.update(temp_age_ms=a & 0x1FFFF, temp_stale=bool(a & TEMP_AGE_STALE))
        if d["has_reg"]:
            r = self.rd(REG)
            d.update(reg_ok=bool(s & ST_REG_OK), reg_cool=bool(s & ST_REG_COOL),
                     reg_warn=bool(s & ST_REG_WARN), reg_c=quarter_c(r),
                     reg_max_c=None if r >> 16 == 0x8000 else quarter_c(r >> 16),
                     reg_age_ms=self.rd(REG_AGE) & 0xFFFF)
        return d

    def limits(self) -> dict:
        """TRIP and RESUME now, and the build's ceilings for each (C); with HAS_REG the
        regulator's too (and REG_STALE, ms)."""
        lim = self.rd(LIMITS)
        d = {"trip_c": celsius(self.rd(TRIP)), "resume_c": celsius(self.rd(RESUME)),
             "trip_max_c": celsius(lim), "resume_max_c": celsius(lim >> 16),
             "hb_ms": self.rd(HB_MS)}
        if self.rd(STATUS) & ST_HAS_REG:
            rl = self.rd(REG_LIMITS)
            d.update(reg_trip_c=quarter_c(self.rd(REG_TRIP)),
                     reg_resume_c=quarter_c(self.rd(REG_RESUME)), reg_stale_ms=self.rd(REG_STALE),
                     reg_trip_max_c=quarter_c(rl), reg_resume_max_c=quarter_c(rl >> 16))
        return d

    def set_limits(self, trip_c: float | None = None, resume_c: float | None = None,
                   hb_ms: int | None = None, reg_trip_c: float | None = None,
                   reg_resume_c: float | None = None, reg_stale_ms: int | None = None) -> None:
        """The fabric clamps each at the build's ceiling (TRIP <= 80 C, RESUME <= 70 C,
        REG_TRIP <= 78 C, REG_RESUME <= 75 C, REG_STALE within 1..30 s)."""
        if trip_c is not None:
            self.wr(TRIP, code_of(trip_c))
        if resume_c is not None:
            self.wr(RESUME, code_of(resume_c))
        if hb_ms is not None:
            self.wr(HB_MS, hb_ms)
        if reg_trip_c is not None:
            self.wr(REG_TRIP, round(4 * reg_trip_c) & 0xFFFF)
        if reg_resume_c is not None:
            self.wr(REG_RESUME, round(4 * reg_resume_c) & 0xFFFF)
        if reg_stale_ms is not None:
            self.wr(REG_STALE, reg_stale_ms)

    def heartbeat_ms(self) -> int:
        """ms since the last KICK (KICK's read; 65535: that long or more, or none yet)."""
        return self.rd(KICK) & 0xFFFF

    def kick(self) -> None:
        self.wr(KICK, 1)

    def go(self, dimms=(0, 1, 2, 3), acc: bool = False) -> bool:
        """The DIMMs (0..3) and the accelerator out of reset, if the die is at or below RESUME
        and no alarm stands; the host kicks within HB_MS from here on. Returns RUN."""
        dimms = tuple(dimms)
        if any(d not in range(4) for d in dimms):
            raise ValueError(f"DIMMs are 0..3, not {dimms}")
        ctrl = self.rd(CTRL) & (3 << CTRL_UART_SHIFT)
        self.wr(CTRL, ctrl | sum(1 << d for d in set(dimms)) | (CTRL_ACC if acc else 0))
        self.kick()
        self.wr(CMD, CMD_GO)
        return bool(self.rd(STATUS) & ST_RUN)

    def set_dimms(self, dimms, acc: bool | None = None) -> None:
        """CTRL's DIMMs (and the accelerator, unless None); in RUN they follow at once (a DIMM
        dropped is off, its core in reset; one added starts and calibrates)."""
        dimms = tuple(dimms)
        if any(d not in range(4) for d in dimms):
            raise ValueError(f"DIMMs are 0..3, not {dimms}")
        ctrl = self.rd(CTRL)
        keep = ctrl & ((3 << CTRL_UART_SHIFT) | (CTRL_ACC if acc is None else 0))
        self.wr(CTRL, keep | sum(1 << d for d in set(dimms)) | (CTRL_ACC if acc else 0))

    def safe(self) -> None:
        """Everything back into reset (the accelerator first, the DIMMs OFF_DLY later)."""
        self.wr(CMD, CMD_SAFE)

    def clear(self) -> None:
        """The causes and the maximum temperature."""
        self.wr(CMD, CMD_CLEAR)

    def uart(self, sel: int) -> None:
        """Which UART the board's pins reach: 0..3 a DIMM's BIOS, per the image's mux."""
        self.wr(CTRL, self.rd(CTRL) & ~(3 << CTRL_UART_SHIFT) | (sel & 3) << CTRL_UART_SHIFT)


# ------------------------------------------------------------------------------ the host's guard
class Guard:
    """The host's half of the BCU1525's thermal protection, in three threads beside the work: the
    governor's heartbeat every `kick_s` (without it the fabric goes SAFE by itself after HB_MS),
    only while the die's thread has finished a pass within `stale_s` (a reader that hangs still
    lets the fabric trip); the die (the governor's code, and the SYSMON over JTAG when the
    transport has `mon`) and the decisions every `read_s`; VCCINT's regulator every `bmc_s`:
    the bridge's watch's readings where it has one (bmc.watch: registers only, the reading's
    age counted), else through a BMC session (bmc.telemetry: its controller's die and each
    page's temperature). Each in its
    own, so a slow one holds back no other: one BMC reading over JTAG takes about 3 s, longer
    than HB_MS (the 2026-10-26 smoke tripped on it), and one took 15 s in that day's cool run,
    longer than `stale_s` with the die read after it in one thread. SAFE when the die reaches
    `die_safe_c` or the regulator
    `reg_safe_c`, or the regulator has gone unread for `bmc_stale_s` while the governor runs;
    the transport's `boot` (JPROGRAM: the flash's configuration) when the die reaches
    `die_boot_c`, the regulator `reg_boot_c`, or the governor stops answering `lost_n` times in a
    row. A CSV row per reading (`csv`). `tripped` names the last trip; `booted` is set once the
    configuration was replaced."""

    COLS = ("time", "run", "dimms", "die_c", "die_max_c", "jtag_c", "vccint_jtag", "reg_ctl_c",
            "reg_p0_c", "reg_p1_c", "vin", "pin_w", "vout0", "causes", "event")

    def __init__(self, t, csv=None, kick_s: float = 0.5, read_s: float = 2.0, bmc_s: float = 5.0,
                 die_safe_c: float = 81.0, die_boot_c: float = 83.0, reg_safe_c: float = 78.0,
                 reg_boot_c: float = 82.0, bmc_stale_s: float = 60.0, lost_n: int = 6,
                 use_bmc: bool = True, clock=None, stale_s: float = 15.0):
        import threading
        import time
        self.t, self.g = t, Gov(t)
        self.kick_s, self.read_s, self.bmc_s = kick_s, read_s, bmc_s
        self.die_safe_c, self.die_boot_c = die_safe_c, die_boot_c
        self.reg_safe_c, self.reg_boot_c = reg_safe_c, reg_boot_c
        self.bmc_stale_s, self.lost_n, self.use_bmc = bmc_stale_s, lost_n, use_bmc
        self.stale_s = stale_s
        self.clock = clock or time.monotonic
        self.alive_at = self.clock()                # the readings' last pass
        self.csv = open(csv, "a", buffering=1) if csv else None
        if self.csv and self.csv.tell() == 0:
            self.csv.write(",".join(self.COLS) + "\n")
        self.stop_ev = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.kthread = threading.Thread(target=self.run_kicks, daemon=True)
        self.bthread = threading.Thread(target=self.run_bmc, daemon=True)
        self.tripped, self.booted, self.lost, self.errors = None, False, 0, []
        self.last: dict = {}
        self.reg: dict = {}
        self.reg_at = None
        self.reg_age_s = 0.0                        # the last reading's age when read (the watch's)
        self._watch: bool | None = None             # the bridge has the watch (once known)

    ERRORS_KEPT = 1000                              # `errors`: the last this many

    def _err(self, msg: str) -> None:
        """One more line of `errors`, the oldest dropped past ERRORS_KEPT (a reader stalled for
        hours adds a "kick held" every kick_s)."""
        self.errors.append(msg)
        if len(self.errors) > self.ERRORS_KEPT:
            del self.errors[:len(self.errors) - self.ERRORS_KEPT]

    # ---- the readings
    def read_bmc(self) -> dict:
        from . import bmc
        if self._watch is not False:
            w = bmc.watch(self.t)
            self._watch = w is not None
            if w is not None:
                temps = (w["ctl_c"], w["p0_c"], w["p1_c"])
                if w["stopped"] or None in temps:
                    raise bmc.BmcError(f"the watch has no reading (firmware {w['firmware']:#x})")
                self.reg_age_s = w["hot_age_ms"] / 1000
                return {"reg_ctl_c": w["ctl_c"], "reg_p0_c": w["p0_c"], "reg_p1_c": w["p1_c"],
                        "vin": w["vin"], "pin_w": w["pin"], "vout0": None}
        self.reg_age_s = 0.0
        with bmc.Bmc(self.t) as b:
            v = bmc.telemetry(b)
        p = v["pages"]
        return {"reg_ctl_c": v["die_c"], "reg_p0_c": p[0]["temp_c"], "reg_p1_c": p[1]["temp_c"],
                "vin": v["vin"], "pin_w": v["pin"], "vout0": p[0]["vout"]}

    def read_first(self, timeout_s: float = 5.0) -> dict:
        """The regulator's first reading, before the threads start (reg, reg_at): the watch's
        once it has one (its first round ends about 0.1 s after the configuration), retried
        until `timeout_s`."""
        import time
        from .bmc import BmcError
        t0 = self.clock()
        while True:
            try:
                self.reg = self.read_bmc()
                self.reg_at = self.clock() - self.reg_age_s
                return self.reg
            except BmcError:
                if self.clock() - t0 > timeout_s:
                    raise
                time.sleep(0.2)

    def reg_max(self) -> float | None:
        vals = [self.reg.get(k) for k in ("reg_ctl_c", "reg_p0_c", "reg_p1_c")]
        vals = [x for x in vals if x is not None]
        return max(vals) if vals else None

    def row(self, event: str = "") -> None:
        if not self.csv:
            return
        import time
        r = {**self.last, **self.reg, "time": time.strftime("%H:%M:%S"), "event": event}
        self.csv.write(",".join("" if r.get(k) is None else
                                (f"{r[k]:.2f}" if isinstance(r[k], float) else
                                 "|".join(map(str, r[k])) if isinstance(r[k], list) else str(r[k]))
                                for k in self.COLS) + "\n")

    # ---- the actions
    def trip(self, why: str) -> None:
        self.tripped = why
        try:
            self.g.safe()
        except OSError as e:
            self._err(f"SAFE: {e}")
        self.row("SAFE " + why)

    def emergency(self, why: str) -> None:
        self.tripped = why
        self.row("BOOT " + why)
        boot = getattr(self.t, "boot", None)
        if boot is None:
            return                          # the fabric's heartbeat timeout is what is left
        try:
            boot()
            self.booted = True
        except OSError as e:
            self._err(f"boot: {e}")

    def kick_step(self, now: float) -> None:
        """One heartbeat, unless the readings have stalled; the flash's configuration once the
        governor has not answered `lost_n` kicks in a row."""
        if self.booted:
            return
        if now - self.alive_at > self.stale_s:
            self._err(f"kick held: no reading for {now - self.alive_at:.0f} s")
            return
        try:
            self.g.kick()
            self.lost = 0
        except OSError as e:
            self.lost += 1
            self._err(f"kick: {e}")
        if self.lost >= self.lost_n:
            self.emergency(f"the governor unanswered {self.lost} times")

    def bmc_step(self, now: float) -> None:
        """One reading of the regulator (run_bmc's), in the transport's priority section where
        it has one (JtagAxi: its tens of commands not held behind DRAM transfers)."""
        if not self.use_bmc or self.booted:
            return
        prio = getattr(self.t, "priority", None)      # JtagAxi's: before DRAM transfers
        try:
            with prio() if prio else contextlib.nullcontext():
                self.reg = self.read_bmc()
            self.reg_at = now - self.reg_age_s
        except (OSError, ValueError) as e:
            self._err(f"bmc: {e}")

    def step(self, now: float, next_read: float, next_bmc: float | None = None
             ) -> tuple[float, float | None]:
        """The die's readings and the decisions when due, the regulator's too when `next_bmc` is
        given and due (one thread: run reads it in run_bmc's); returns the next due times."""
        self.alive_at = now
        if next_bmc is not None and now >= next_bmc:
            self.bmc_step(now)
            next_bmc = now + self.bmc_s
        if now < next_read:
            return next_read, next_bmc
        if not self.booted:
            try:
                s = self.g.status()
                self.last = {"run": int(s["run"]), "dimms": s["dimms_on"], "die_c": s["temp_c"],
                             "die_max_c": s["max_c"], "causes": s["causes"]}
            except OSError as e:
                self._err(f"status: {e}")
        mon = getattr(self.t, "mon", None)
        if mon is not None:
            try:
                m = mon()
                self.last.update(jtag_c=m["t"], vccint_jtag=m["vccint"])
            except OSError as e:
                self._err(f"mon: {e}")
        die = max([x for x in (self.last.get("die_c"), self.last.get("jtag_c")) if x is not None],
                  default=None)
        reg = self.reg_max()
        run = self.last.get("run")
        if not self.booted and ((die is not None and die >= self.die_boot_c) or
                                (reg is not None and reg >= self.reg_boot_c)):
            self.emergency(f"die {die} C, regulator {reg} C")
            return now + self.read_s, next_bmc
        if run and die is not None and die >= self.die_safe_c:
            self.trip(f"die {die:.1f} C")
        elif run and reg is not None and reg >= self.reg_safe_c:
            self.trip(f"regulator {reg:.1f} C")
        elif run and self.use_bmc and (self.reg_at is None or now - self.reg_at > self.bmc_stale_s):
            self.trip("the regulator unread")
        else:
            self.row()
        return now + self.read_s, next_bmc

    def run(self) -> None:
        next_read = self.clock()
        while not self.stop_ev.is_set():
            next_read, _ = self.step(self.clock(), next_read)
            self.stop_ev.wait(self.kick_s)

    def run_bmc(self) -> None:
        while not self.stop_ev.is_set():
            self.bmc_step(self.clock())
            self.stop_ev.wait(self.bmc_s)

    def run_kicks(self) -> None:
        while not self.stop_ev.is_set():
            self.kick_step(self.clock())
            self.stop_ev.wait(self.kick_s)

    def start(self) -> "Guard":
        self.alive_at = self.clock()
        self.kthread.start()
        self.thread.start()
        self.bthread.start()
        return self

    def stop(self, safe: bool = True) -> None:
        self.stop_ev.set()
        for th in (self.kthread, self.thread, self.bthread):
            if th.is_alive():
                th.join(timeout=30 if th is self.bthread else 10)
        if safe and not self.booted:
            try:
                self.g.safe()
            except OSError as e:
                self._err(f"SAFE at stop: {e}")
        if self.csv:
            self.row("stop")
            self.csv.close()
