#!/usr/bin/env python3
"""The BCU1525s' fans by the cards' temperatures, on the host's fan controller (the two-fan
fanctl, fanctl.py beside this file): a PI loop per card that holds the FPGA's die at --target
(45 C) and the VCCINT regulator at --reg-target (10 C above it: the regulator follows the fan
slowly, so the die is the one to hold), following whichever is further above its own, with the
fan's duty between --lo (55 %: below the 60 % the fan needs to turn, so it stands still while the
card is cool) and 100 %, and a live line of the temperatures and the duty.

    fan_pid.py [--target 45] [--reg-target 55] [--source auto|pcie|jtag] [--csv FILE]
               [--kp 4 --ki 0.04 --kd 0] [--gains-file ~/.config/otpu/fan_pid.json]
               [--filter-s 10] [--status ~/otpu-build/bcu-fan/status.json]
               [--target2 T] [--status2 ~/otpu-build/bcu-fan/status2.json] [--csv2 FILE]
               [--no-fan2]
    fan_pid.py --tune [--target 45] [--tune-hi 80] [--tune-hyst 1] [--tune-rule tl-pi]
               [--tune-dry-run] [--tune-exit]

Two cards, two fans: fan 1 on card 0 (JTAG --serial, PCI function --pci), fan 2 on card 1
(--serial2, --pci2); the defaults are the host's cards, BCU_CARDn_SERIAL / _BDF / _NAME in the
environment or in the BCU wrappers' file (BCU_ENV, default ~/.config/otpu/bcu.env: NAME=VALUE
lines; boards/bcu1525/chat35/README.md). Each card has its own
reading, loop, overrides, blind timeout, status file and CSV (--csv2: --csv's name with a 2, as
fan.csv -> fan2.csv); the gains, the limits and the source are both cards'. The card's xdma device
is found by its PCI function at each reading (both cards on the bus: /dev/xdma0 can be either), or
is --dev / --dev2 as given. --target2 is fan 2's die target (default --target), its regulator's
the same distance above it. --tune tunes fan 1's loop; fan 2's runs the gains meanwhile. With
--no-fan2, or a controller or fanctl without fan 2 (said at the start), fan 1 alone, as before.

The loop and the tuning act on the control temperature filtered over --filter-s (10 s: an EMA,
alpha = dt / (tau + dt)), P, I and D alike: the die's reading moves in about 0.5 C steps and by
1 to 1.5 C from one reading to the next. The overrides and the blind timeout act on the raw
readings.

The gains: each of kp, ki and kd from its flag, else from the gains file (--gains-file: what
--tune saved), else the built-in 4, 0.04 and 0; the start-up line says where each came from. A
missing or unreadable file is a warning, and the built-in gains.

--tune calibrates them: a relay autotune (Astrom and Hagglund) of the die around --target. The
duty goes to --tune-hi (80 %) while the filtered die is above target + --tune-hyst (1 C) and to
--tune-lo (--lo: the fan still) while it is below target - hyst, so the die's temperature cycles
about the target (the regulator is not followed, only its override and the ceiling hold while
tuning). The hysteresis has to exceed the filtered reading's noise band, or the relay switches
on noise and its cycles measure nothing. A cycle runs from one switch to --tune-hi to the next,
and the first is not counted (the approach). Once --tune-cycles (4) in a row agree, periods
within 20 % of their mean and peak-to-peaks within 25 %, Ku = 4 d / (pi sqrt(a^2 - h^2)) (d half
the relay's swing, a half the peak-to-peak, h the hysteresis) and Tu = the mean period give the
gains by --tune-rule:

    tl-pi         Tyreus-Luyben PI:  Kp = Ku / 3.2  Ti = 2.2 Tu               (the default)
    tl-pid        Tyreus-Luyben PID: Kp = Ku / 2.2  Ti = 2.2 Tu   Td = Tu / 6.3
    zn-pi         Ziegler-Nichols:   Kp = 0.45 Ku   Ti = Tu / 1.2
    zn-pid        Ziegler-Nichols:   Kp = 0.6 Ku    Ti = Tu / 2   Td = Tu / 8
    no-overshoot                     Kp = 0.2 Ku    Ti = Tu / 2   Td = Tu / 3

with ki = Kp / Ti and kd = Kp Td. They go to the gains file with what was measured (not with
--tune-dry-run), and the loop goes on with them (--tune-exit: exits). No crossing for
--tune-stall-s (900 s: the card does not reach the target with the fan still, or stays above it
at --tune-hi), no agreement within --tune-max-s (3600 s), an override's 100 % or the blind
timeout abort the tuning: the loop goes on with the gains it started with, and saves nothing. The
card has to make its heat meanwhile (a load that keeps it above the target with the fan still).

The temperatures, read only (the loop follows the die, or the regulator while it is further above
--reg-target than the die is above --target; the die alone when the regulator's reading is
missing or stale, and always over JTAG):
  pcie  our images' governor through /dev/xdma0_user (BAR0 0x10000, opentpu/host/gov.py): the
        SYSMON's die reading, and the VCCINT regulator's temperature (the BMC watch). The
        governor's die is the wizard's temperature bus, which can stop updating (f150m15,
        2026-10-29: frozen at 46 C from 06:04 while the die rose to 80 C): a reading that
        reads the same for --stale-s (60 s; a live one moves in 0.5 C steps, the longest
        identical run in 8 h was 14 s) has stopped ("gov frozen"; a governor with TEMP_AGE
        says so itself, its TEMP_VALID clear at TEMP_STALE_MS), and then the SYSMON's own
        temperature register (BAR0 0x30400) is the die, itself dropped once it stops ("sysmon
        frozen"); with neither, the duty goes to 100 % at once. The register is read only
        then: an AXI read of the wizard's DRP registers is what freezes the bus (its
        drp_arbiter drops the bus's own read when the AXI read lands in the cycle before it,
        about one read in 1000). The device is opened for each reading only, so `otpu-setup
        --rescan` is never blocked by it. With the BMC's watch (the governor's HAS_REG: a BMC2
        bridge, BAR0 0x20000) its registers too, bmc.watch's (ID, W_HOT, W_P0 .. W_VIN, W_FW,
        W_CNT; never STATUS or RX): VCCINT's regulator input power (READ_PIN) and VIN, and
        their ages.
  jtag  the SYSMON over JTAG, any image, the factory one too (one Vivado in batch mode,
        opentpu/host/jtagaxi.py, started in the background: about a minute). It steps aside while
        another Vivado runs (a bring-up or a programming: one JTAG user at a time) and comes back
        30 s after it ends.
  auto  pcie while the governor answers, else jtag.

Without a reading the duty holds, and goes to 100 % after --blind-s (60 s); with every die
reading frozen (above), at once. The die at --hot-c or the
regulator at --reg-hot-c sets 100 % whatever the loop or the tuning says; 80 C is the ceiling of
both (a larger value is taken as 80). The fan's firmware goes to 100 % by itself if this process
dies (its watchdog, --watchdog s); Ctrl-C, SIGTERM or a hang-up turns the watchdog off and leaves
the duty where it is; an error sets both fans 100 % and leaves the watchdog on. While this runs,
`fanctl.py set` cannot open the port (the controller takes one user at a time).

Each step writes the status file (--status; '' for none) whole, a temporary file renamed over it:
the time, the source, the governor's state, the die's readings, the regulator's, the power, the
fan's duty and why, this process (docs/bcu1525.md section 15 has its fields). A reader takes the
time against its own clock: a file older than a few steps is a stopped fan_pid.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
import json
import math
import mmap
import os
import signal
import socket
import sys
import tempfile
import threading
import time
import types
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HOST = ROOT / "opentpu/host"


def card_env(n: int, key: str, default: str = "") -> str:
    """BCU_CARD<n>_<key> (SERIAL: the card's FT4232H serial; BDF: its PCI function; NAME: a short
    name) from the environment, else from the host's card file (BCU_ENV, default
    ~/.config/otpu/bcu.env: NAME=VALUE lines, the BCU wrappers' file), else `default`."""
    name = f"BCU_CARD{n}_{key}"
    if os.environ.get(name):
        return os.environ[name]
    try:
        text = Path(os.environ.get("BCU_ENV") or "~/.config/otpu/bcu.env").expanduser().read_text()
    except OSError:
        return default
    for line in text.splitlines():
        k, eq, v = line.strip().removeprefix("export ").partition("=")
        if eq and k.strip() == name:
            # the value as the shell that sources the file takes it: a quoted string whole,
            # else the first word (a comment may follow it)
            if v[:1] in ("'", '"'):
                end = v.find(v[0], 1)
                v = v[1:end] if end > 0 else v[1:]
            else:
                v = v.split(None, 1)[0] if v[:1].strip() else ""
            return v or default
    return default


SERIAL = card_env(0, "SERIAL")              # fan 1's card's FT4232H serial
SERIAL2 = card_env(1, "SERIAL")             # fan 2's card's
PCI = card_env(0, "BDF")                    # fan 1's card's PCI function (0000:BB:DD.F)
PCI2 = card_env(1, "BDF")                   # fan 2's card's
NAME = card_env(0, "NAME", "card0")         # the cards' names in the lines and the status files
NAME2 = card_env(1, "NAME", "card1")
FAN_DIR = os.environ.get("OTPU_FAN_DIR") or "~/otpu-build/bcu-fan"  # the status files, logs (serve/fans.py's)
FANCTL = str(Path(__file__).resolve().with_name("fanctl.py"))        # the two-fan fanctl (~/fanctl.py: a one-fan one)
CEIL_C = 80.0                              # the die or the regulator at this: 100 %, every mode
REG_ABOVE_C = 10.0                          # the regulator's default target above the die's
SM_TEMP = 0x30400                           # the SYSMON's temperature register (BAR0; code in [15:6])
STALE_S = 60.0                              # a die reading the same this long has stopped
GAINS = {"kp": 4.0, "ki": 0.04, "kd": 0.0}  # the built-in gains
GAINS_FILE = "~/.config/otpu/fan_pid.json"
CSV_COLS = ["time", "source", "die_c", "reg_c", "duty", "p", "i", "d", "why", "state", "tune",
            "filt_c", "pin_w", "vin"]
STATUS_FILE = f"{FAN_DIR}/status.json"
STATUS_FILE2 = f"{FAN_DIR}/status2.json"
LIVE_GAINS = "gains_live.json"              # beside the status file: the gains set while running
GAIN_MAX = {"kp": 100.0, "ki": 10.0, "kd": 1000.0}  # (above these: a typo; the same in serve/fans.py)
TARGET_RANGE = (40.0, 70.0)                 # the die's, set while running: under the 75 C overrides
VERSION = "1.5"                             # 1.1: BMC watch power, status file; 1.2: fan 2; 1.4: the
                                            # gains set while running (1.3 was reverted); 1.5: the target
POWER_LABEL = "VCCINT regulator input (READ_PIN), not the board total"


def load(name: str, path: Path):
    """A module by its file, without its package (opentpu's __init__ imports the ISA simulator)."""
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def host(name: str):
    """An opentpu/host module that imports its siblings (bmc: regs, i2c, runstate), under a
    package of its own for the same reason."""
    if "otpu_host" not in sys.modules:
        p = types.ModuleType("otpu_host")
        p.__path__ = [str(HOST)]
        sys.modules["otpu_host"] = p
    return importlib.import_module("otpu_host." + name)


G = load("otpu_gov", HOST / "gov.py")
B = host("bmc")


class Pid:
    """Duty (%) from a temperature T: lo + Kp e + I + Kd dT/dt, e = T - target, clamped to
    lo .. hi. The integral grows only while the output is not pinned in the error's direction
    (no wind-up at either end) and stays within 0 .. hi - lo: below the target the fan rests at
    lo. The derivative is on the measurement (a smoothed one), so a new target does not kick."""

    def __init__(self, target: float, kp: float, ki: float, kd: float, lo: float, hi: float,
                 smooth: float = 0.5):
        self.target, self.kp, self.ki, self.kd, self.lo, self.hi = target, kp, ki, kd, lo, hi
        self.smooth = smooth
        self.i, self.tf, self.p, self.d = 0.0, None, 0.0, 0.0

    def step(self, t: float, dt: float) -> float:
        prev = self.tf
        self.tf = t if prev is None else prev + self.smooth * (t - prev)
        e = t - self.target
        self.p = self.kp * e
        self.d = 0.0 if prev is None or dt <= 0 else self.kd * (self.tf - prev) / dt
        i = min(max(self.i + self.ki * e * dt, 0.0), self.hi - self.lo)
        u = self.lo + self.p + i + self.d
        if not ((u > self.hi and e > 0) or (u < self.lo and e < 0)):
            self.i = i
        return min(max(self.lo + self.p + self.i + self.d, self.lo), self.hi)


class RelayTune:
    """A relay autotune of the gains (Astrom-Hagglund), one reading at a time: the duty at `hi`
    while T is above target + h, at `lo` while it is below target - h, held in between. A cycle
    runs from one switch to `hi` to the next; the first is not counted. Once the last `cycles`
    agree (periods within P_TOL of their mean, peak-to-peaks within A_TOL), `result` holds the
    gains by `rule` and what was measured. No switch for `stall_s`, no agreement within `max_s`
    or an abort() sets `error` instead."""

    RULES = {"tl-pi": (1 / 3.2, 2.2, 0.0),                      # Kp / Ku, Ti / Tu, Td / Tu
             "tl-pid": (1 / 2.2, 2.2, 1 / 6.3),
             "zn-pi": (0.45, 1 / 1.2, 0.0),
             "zn-pid": (0.6, 0.5, 0.125),
             "no-overshoot": (0.2, 0.5, 1 / 3)}
    P_TOL, A_TOL = 0.20, 0.25

    def __init__(self, target: float, h: float, lo: float, hi: float, cycles: int = 4,
                 rule: str = "tl-pi", stall_s: float = 900.0, max_s: float = 3600.0):
        self.target, self.h, self.lo, self.hi = target, h, lo, hi
        self.n, self.rule, self.stall_s, self.max_s = cycles, rule, stall_s, max_s
        self.high, self.duty = False, lo
        self.t0 = self.switched = self.up = self.last = self.tmax = self.tmin = None
        self.area = 0.0                                         # duty x s since the cycle began
        self.cycles = []                    # (period s, peak-to-peak C, mean duty %) of each one
        self.result, self.error = None, ""

    @property
    def done(self) -> bool:
        return self.result is not None or bool(self.error)

    def abort(self, why: str) -> None:
        if not self.done:
            self.error = why

    @classmethod
    def agree(cls, c: list) -> bool:
        ps, pp = [x[0] for x in c], [x[1] for x in c]
        return (max(ps) - min(ps) <= cls.P_TOL * sum(ps) / len(ps)
                and max(pp) - min(pp) <= cls.A_TOL * sum(pp) / len(pp))

    def run(self) -> list:
        """The agreeing cycles at the end, at most `cycles` (the first is never counted)."""
        c = self.cycles[1:]
        for k in range(min(self.n, len(c)), 0, -1):
            if self.agree(c[-k:]):
                return c[-k:]
        return []

    def step(self, t: float, now: float) -> float:
        if self.done:
            return self.duty
        if self.t0 is None:
            self.t0 = self.switched = now
        else:
            self.area += self.duty * (now - self.last)
        self.last = now
        if not self.high and t > self.target + self.h:
            self.high, self.duty, self.switched = True, self.hi, now
            if self.up is not None:
                self.cycles.append((now - self.up, self.tmax - self.tmin,
                                    self.area / (now - self.up)))
                self.accept()
            self.up, self.tmax, self.tmin, self.area = now, t, t, 0.0
        else:
            if self.high and t < self.target - self.h:
                self.high, self.duty, self.switched = False, self.lo, now
            if self.up is not None:
                self.tmax, self.tmin = max(self.tmax, t), min(self.tmin, t)
        if self.done:
            return self.duty
        if now - self.switched >= self.stall_s:
            self.error = (f"no crossing in {self.stall_s:g} s: the card stays above "
                          f"{self.target - self.h:g} C with the fan at {self.hi:g} %: raise "
                          f"--tune-hi or --target" if self.high else
                          f"no crossing in {self.stall_s:g} s: the card stays below "
                          f"{self.target + self.h:g} C with the fan at {self.lo:g} %: load the "
                          f"card or lower --target")
        elif now - self.t0 >= self.max_s:
            self.error = f"no {self.n} agreeing cycles in {self.max_s:g} s"
        return self.duty

    def accept(self) -> None:
        run = self.run()
        if len(run) < self.n:
            return
        tu = sum(x[0] for x in run) / self.n
        a = sum(x[1] for x in run) / self.n / 2
        if a <= self.h:                     # cannot happen: each cycle crosses both thresholds
            self.error = f"amplitude {a:.2f} C within the hysteresis"
            return
        ku = 4 * (self.hi - self.lo) / 2 / (math.pi * math.sqrt(a * a - self.h * self.h))
        fk, fi, fd = self.RULES[self.rule]
        kp = fk * ku
        g = lambda x: float(f"{x:.4g}")                                 # noqa: E731
        self.result = {"kp": g(kp), "ki": g(kp / (fi * tu)), "kd": g(kp * fd * tu),
                       "rule": self.rule, "ku": g(ku), "tu_s": g(tu), "amplitude_c": g(a),
                       "duty": g(sum(x[2] for x in run) / self.n), "target_c": self.target,
                       "hyst_c": self.h, "relay_lo": self.lo, "relay_hi": self.hi,
                       "cycles": self.n}

    def state(self) -> str:
        run = self.run()
        s = f"relay {'hi' if self.high else 'lo'}, cycle {len(run)}/{self.n}"
        if run:
            s += (f", Tu {sum(x[0] for x in run) / len(run):.0f} s,"
                  f" a {sum(x[1] for x in run) / len(run) / 2:.2f} C")
        return s


def load_gains(path: Path, err=sys.stderr) -> dict | None:
    """kp, ki and kd from a gains file; None, and a warning, when it is missing or not one."""
    ok = lambda v: (isinstance(v, (int, float)) and not isinstance(v, bool)    # noqa: E731
                    and math.isfinite(v) and v >= 0)
    try:
        g = json.loads(path.read_text())
        if not isinstance(g, dict) or not all(ok(g.get(k)) for k in GAINS):
            raise ValueError("kp, ki and kd must be numbers >= 0")
        return {k: float(g[k]) for k in GAINS}
    except FileNotFoundError:
        print(f"fan_pid: no gains file {path}: the built-in gains", file=err)
    except (OSError, ValueError) as e:
        print(f"fan_pid: warning: {path}: {e}: the built-in gains", file=err)
    return None


def resolve_gains(a, err=sys.stderr) -> str:
    """kp, ki and kd into `a`, each from its flag, else the gains file, else the built-in; what
    the start-up line says about them."""
    path = Path(a.gains_file).expanduser()
    f = load_gains(path, err) if any(getattr(a, k) is None for k in GAINS) else None
    src = {}
    for k, v in GAINS.items():
        if getattr(a, k) is not None:
            src[k] = "flag"
        elif f is not None:
            setattr(a, k, f[k])
            src[k] = str(path)
        else:
            setattr(a, k, v)
            src[k] = "built-in"
    gs = " ".join(f"{k} {getattr(a, k):g}" for k in GAINS)
    if len(set(src.values())) == 1:
        return f"{gs} ({src['kp']})"
    return f"{gs} (" + ", ".join(f"{k} {s}" for k, s in src.items()) + ")"


def save_gains(path: Path, g: dict) -> None:
    """The gains file whole or not at all."""
    write_whole(path, json.dumps(g, indent=2) + "\n")


def write_whole(path: Path, text: str, sync: bool = True, mode: int | None = None) -> None:
    """A file whole or not at all: a temporary file beside it, then a rename over it (a reader
    sees the old file or the new one); `sync`: on the disk before the rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            if sync:
                os.fsync(f.fileno())
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def sysmon_c(v: int) -> float | None:
    """The SYSMON's temperature register's word in C (its 10-bit code in [15:6]), or None when it
    reads as no SYSMON (all ones, zero, or out of -40 .. 125 C)."""
    if v in (0, 0xFFFFFFFF):
        return None
    c = G.celsius(v >> 6)
    return c if -40.0 <= c <= 125.0 else None


class Fresh:
    """A reading that has to keep moving: frozen once it has read the same for stale_s (0: never)."""

    def __init__(self, stale_s: float, clock=time.monotonic):
        self.stale_s, self.clock, self.v, self.t = stale_s, clock, None, None

    def frozen(self, v) -> bool:
        now = self.clock()
        if self.t is None or v != self.v:
            self.v, self.t = v, now
        return self.stale_s > 0 and now - self.t >= self.stale_s


class Bar:
    """bmc.watch's transport over the mapped BAR0: register reads, nothing else."""

    def __init__(self, w):
        self.w = w

    def reg_read(self, a: int) -> int:
        return self.w[a // 4]


def power_of(w: dict | None, stale_ms: int) -> dict | None:
    """The power from bmc.watch's readings (None: no watch): READ_PIN (W) and READ_VIN (V), the
    power's age (s), stale once it is the governor's REG_STALE old; the reading None before the
    watch's first and once it is 65.5 s old (the age register saturates)."""
    if w is None:
        return None
    pin, age = w["pin"], w["ages_ms"]["pin"]
    return {"vccint_in_w": pin, "vin_v": w["vin"],
            "age_s": None if pin is None else age / 1000,
            "stale": None if pin is None else age >= stale_ms,
            "firmware": w["firmware"], "watch_stopped": w["stopped"]}


def xdma_of(bdf: str, sys_root: Path = Path("/sys")) -> str | None:
    """The xdma device (/dev/xdmaN) of the PCI function `bdf`, None while it has none (the card off
    the bus, or the driver not bound): the driver's node under the function (setup_pcie.sh's)."""
    for p in sorted((sys_root / "bus/pci/devices" / bdf / "xdma").glob("xdma*_user")):
        return "/dev/" + p.name[:-len("_user")]
    return None


class PcieSource:
    """The governor's readings through BAR0, the die the governor's while it moves, else the
    SYSMON's register's (read only then); None when the governor does not answer, die_c None
    (and "lost") when neither moves. With the BMC's watch, its power (bmc.watch's registers).
    The device `dev` as given, or with `pci` the one the driver made for that PCI function, found
    at each reading."""
    name = "pcie"

    def __init__(self, dev: str | None = "/dev/xdma0", stale_s: float = STALE_S,
                 clock=time.monotonic, pci: str | None = None, sys_root: Path = Path("/sys")):
        self.dev, self.pci, self.sys_root, self.stale_s = dev, pci, sys_root, stale_s
        self.gov_f, self.sm_f = Fresh(stale_s, clock), Fresh(stale_s, clock)

    @property
    def path(self) -> str | None:
        dev = self.dev if self.pci is None else xdma_of(self.pci, self.sys_root)
        return None if dev is None else f"{dev}_user"

    def read(self) -> dict | None:
        path = self.path
        if path is None:
            return None
        try:
            fd = os.open(path, os.O_RDONLY | os.O_SYNC)
        except OSError:
            return None
        m = None
        try:            # the SYSMON's register and the BMC's bridge too, when the BAR reaches them
            for size in (SM_TEMP + 0x1000 & ~0xFFF, B.R.R_BMC + 0x1000, G.R_GOV + 0x1000):
                try:
                    m = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ)
                    break
                except (OSError, ValueError):
                    pass
        finally:
            os.close(fd)
        if m is None:
            return None
        try:
            w = memoryview(m).cast("I")
            try:
                rd = lambda off: w[(G.R_GOV + off) // 4]            # noqa: E731
                if rd(G.ID) != G.GOV_ID:
                    return None
                s, tmp = rd(G.STATUS), rd(G.TEMP)
                gov_ok = bool(s & G.ST_TEMP_VALID)
                gov, sm, flags = G.celsius(tmp) if gov_ok else None, None, []
                age = rd(G.TEMP_AGE) if rd(G.TEMP_STALE) else None
                if age is not None and age & G.TEMP_AGE_STALE:
                    flags.append("gov frozen")      # the governor's own count (TEMP_VALID clear)
                elif gov is not None and self.gov_f.frozen(tmp & 0x3FF):
                    gov = None
                    flags.append("gov frozen")
                if gov is not None:                 # the register unread (its reads freeze the bus)
                    self.sm_f.t = None
                elif len(w) > SM_TEMP // 4:
                    smw = w[SM_TEMP // 4]
                    sm = sysmon_c(smw)
                    if sm is not None and self.sm_f.frozen(smw):
                        sm = None
                        flags.append("sysmon frozen")
                if not gov_ok and sm is None and not flags:
                    return None
                live = [c for c in (gov, sm) if c is not None]
                mode, dimms = "RUN" if s & G.ST_RUN else "SAFE", [d for d in range(4)
                                                                 if s >> (8 + d) & 1]
                r = {"die_c": live[0] if live else None, "gov_c": gov, "sysmon_c": sm,
                     "reg_c": None,
                     "lost": "" if live else f"die readings frozen {self.stale_s:g} s",
                     "state": mode + "".join(f" C{d}" for d in dimms) +
                              (" acc" if s & G.ST_ACC_ON else "") +
                              "".join(" " + f for f in flags),
                     "mode": mode, "dimms": dimms, "acc": bool(s & G.ST_ACC_ON),
                     "gov_frozen": "gov frozen" in flags,
                     "sysmon_frozen": "sysmon frozen" in flags,
                     "gov_age_s": None if age is None else (age & 0x1FFFF) / 1000,
                     "power": None}
                if s & G.ST_HAS_REG:
                    reg, rage = rd(G.REG), rd(G.REG_AGE) & 0xFFFF
                    if rage < 10000:
                        r["reg_c"] = G.quarter_c(reg)
                    if len(w) > (B.R.R_BMC + B.W_CNT) // 4:     # the watch: registers only
                        r["power"] = power_of(B.watch(Bar(w)), rd(G.REG_STALE) or 10000)
                return r
            finally:
                w.release()
        finally:
            m.close()


def procs(proc: Path = Path("/proc")) -> list[tuple[int, str, int, int]]:
    """(pid, comm, ppid, session) of every process."""
    out = []
    for p in proc.iterdir():
        if not p.name.isdigit():
            continue
        try:
            comm = (p / "comm").read_text().strip()
            stat = (p / "stat").read_text()
            f = stat[stat.rindex(")") + 2:].split()
            out.append((int(p.name), comm, int(f[1]), int(f[3])))
        except (OSError, ValueError, IndexError):
            continue
    return out


def own_sessions(ps, me: int) -> set[int]:
    """Our children's sessions (JtagAxi starts its Vivado in a session of its own)."""
    return {pid for pid, _, ppid, _ in ps if ppid == me}


def other_vivados(ps, own: set[int]) -> list[int]:
    """The Vivado processes outside our own sessions: a bring-up's or a programming's."""
    return [pid for pid, comm, _, sid in ps if comm == "vivado" and sid not in own]


class JtagSource:
    """The SYSMON over JTAG from a background thread: the latest reading, None when older than
    `fresh_s`, while starting, or while another Vivado has the cable. Each card's reader has a
    Vivado of its own on the one hardware server."""
    name = "jtag"
    hw_lock = threading.Lock()              # two readers starting: one starts the hw_server

    def __init__(self, serial: str = SERIAL, interval: float = 2.0, fresh_s: float = 10.0,
                 quiet_s: float = 30.0, log: Path | None = None):
        self.serial, self.interval, self.fresh_s, self.quiet_s, self.log = \
            serial, interval, fresh_s, quiet_s, log
        self.last, self.at, self.note = None, 0.0, "starting"
        self.stop_ev = threading.Event()
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self) -> None:
        J = load("otpu_jtagaxi", HOST / "jtagaxi.py")
        t, quiet_since = None, None
        while not self.stop_ev.is_set():
            ps = procs()
            if other_vivados(ps, own_sessions(ps, os.getpid())):
                if t:
                    self._close(t)
                    t = None
                self.note, quiet_since = "another Vivado has JTAG", None
            elif t is None:
                if quiet_since is None:
                    quiet_since = time.monotonic()
                if time.monotonic() - quiet_since >= (self.quiet_s if self.last else 0):
                    self.note = "starting Vivado"
                    try:
                        with JtagSource.hw_lock:
                            J.start_hw_server(log=self.log.with_suffix(".hw_server.log")
                                              if self.log else None)
                        t = J.JtagAxi(self.serial, log=self.log)
                        self.note = ""
                    except OSError as e:
                        self.note, t = f"JTAG: {e}", None
                        self._kill_own()
                        self.stop_ev.wait(10)
            else:
                try:
                    self.last, self.at = t.mon()["t"], time.monotonic()
                except OSError as e:
                    self.note = f"JTAG: {e}"
                    self._close(t)
                    t = None
            self.stop_ev.wait(self.interval)
        if t:
            self._close(t)

    @staticmethod
    def _close(t) -> None:
        try:
            t.close()
        except OSError:
            pass

    @staticmethod
    def _kill_own() -> None:
        """A Vivado of ours that never answered (JtagAxi leaves it on a failed start)."""
        ps = procs()
        for pid, comm, ppid, _ in ps:
            if ppid == os.getpid() and comm == "vivado":
                try:
                    os.killpg(pid, signal.SIGKILL)
                except OSError:
                    pass

    def read(self) -> dict | None:
        if self.last is None or time.monotonic() - self.at > self.fresh_s:
            return None
        return {"die_c": self.last, "reg_c": None, "state": ""}

    def close(self) -> None:
        self.stop_ev.set()
        self.th.join(timeout=60)


class AutoSource:
    """pcie while it answers with a die reading; jtag (started after pcie has failed `after_s`)
    otherwise: the governor not answering, or its die readings frozen (pcie's lost reading, so
    100 %, until jtag gives one)."""

    def __init__(self, pcie, make_jtag, after_s: float = 5.0):
        self.pcie, self.make_jtag, self.after_s = pcie, make_jtag, after_s
        self.jtag, self.failed_at, self.name = None, None, "pcie"

    @property
    def note(self) -> str:
        return self.jtag.note if self.jtag and self.name == "jtag" else ""

    def read(self) -> dict | None:
        r = self.pcie.read()
        now = time.monotonic()
        if r is not None and r["die_c"] is not None:
            self.failed_at, self.name = None, "pcie"
            if self.jtag:
                self.jtag.close()
                self.jtag = None
            return r
        if self.failed_at is None:
            self.failed_at = now
        if self.jtag is None and now - self.failed_at >= self.after_s:
            self.jtag = self.make_jtag()
        if self.jtag:
            j = self.jtag.read()
            if j is not None or r is None:
                self.name = "jtag"
                return j
        self.name = "pcie"
        return r

    def close(self) -> None:
        if self.jtag:
            self.jtag.close()


class FanPort:
    """One of the controller's fans as a loop sees it: set(duty), the reply's duty as "duty".
    Fan 1's command is the one every fanctl has; fan 2's needs the two-fan one."""

    def __init__(self, fan, n: int = 1):
        self.fan, self.n = fan, n

    def set(self, d):
        if self.n == 1:
            return self.fan.set(d)
        s = self.fan.set(d, fan=self.n)
        return {**s, "duty": s.get(f"duty{self.n}", d)}


def has_fan2(F, fan) -> str:
    """Why fan 2 cannot be driven ("" when it can): a fanctl without it, or a controller whose
    STATUS has no duty2 (the one-fan firmware)."""
    try:
        if "fan" not in inspect.signature(F.FanController.set).parameters:
            return f"{F.__file__} is the one-fan fanctl"
    except (TypeError, ValueError):
        return "fanctl's set() unreadable"
    if "duty2" not in fan.status():
        return "the controller's firmware has one fan"
    return ""


class Loop:
    """One control step at a time: a reading, the duty (the PID's; with --tune the relay's until
    the tuning ends, then the PID's with the new gains), the fan's command (which also feeds its
    watchdog), the display, the status file (`status`: its path, None for none). One per card:
    `card` names it in the lines and the status file, `csv_path`, `tune` and `reg_target` its
    own (default --csv, --tune and --reg-target)."""

    def __init__(self, fan, src, pid: Pid, a, out=sys.stdout, clock=time.monotonic,
                 status: Path | None = None, card: str = "", csv_path: str | None = None,
                 tune: bool | None = None, reg_target: float | None = None):
        self.fan, self.src, self.pid, self.a, self.out, self.clock = fan, src, pid, a, out, clock
        self.status, self.status_err, self.started = status, "", time.time()
        self.live_err = ""
        self.card = card
        self.reg_target = a.reg_target if reg_target is None else reg_target
        self.duty, self.last_ok, self.prev_t = a.lo, clock(), None
        self.max_c, self.why, self.shown_why = None, "", ""
        self.filt, self.filt_t = None, None
        self.tty = out.isatty() and not card    # two cards' lines: kept lines, no live line
        self.committed = 0.0
        tune = a.tune if tune is None else tune
        self.tuner = (RelayTune(pid.target, a.tune_hyst, a.tune_lo, a.tune_hi, a.tune_cycles,
                                a.tune_rule, a.tune_stall_s, a.tune_max_s) if tune else None)
        self.exit = None                    # the exit status once --tune-exit's tuning is done
        self.csv = None
        csv_path = a.csv if csv_path is None else csv_path
        if csv_path:
            p = Path(csv_path)
            old = None
            if p.exists():
                with open(p, newline="") as f:
                    old = next(csv.reader(f), None)
            # a file begun by an older fan_pid keeps its columns
            self.ncol = len(old) if old and old == CSV_COLS[:len(old)] else len(CSV_COLS)
            self.csv_f = open(p, "a", newline="")
            self.csv = csv.writer(self.csv_f)
            if not p.stat().st_size:
                self.csv.writerow(CSV_COLS)

    def hot(self, die: float, reg: float | None) -> str:
        """Why the duty is 100 % whatever the loop says, or "": the die or the regulator at its
        override, never above the ceiling."""
        hot_c, reg_hot_c = min(self.a.hot_c, CEIL_C), min(self.a.reg_hot_c, CEIL_C)
        if die >= hot_c:
            return f"die >= {hot_c:g} C"
        if reg is not None and reg >= reg_hot_c:
            return f"regulator >= {reg_hot_c:g} C"
        return ""

    def filter(self, t: float, now: float) -> float:
        """The control temperature's EMA over --filter-s, alpha = dt / (tau + dt) with dt the
        time since the last reading (after a gap, the reading weighs more)."""
        tau = self.a.filter_s
        if self.filt is None or tau <= 0:
            self.filt = t
        else:
            dt = now - self.filt_t
            self.filt += dt / (tau + dt) * (t - self.filt)
        self.filt_t = now
        return self.filt

    def live_gains(self) -> None:
        """This fan's kp, ki and kd, and target_c if there, from LIVE_GAINS beside the status file
        (the dashboard writes it whole: {"fan1": {"kp": .., "ki": .., "kd": .., "target_c": ..},
        ...}), on the PID at this step: over what the loop started with, also after a restart.
        The integral stays, so a new ki does not bump the duty; the regulator keeps its own
        target. No such file: nothing to do; a bad one: said once, the gains stay."""
        if self.status is None or self.tuner:
            return
        path, n, err = self.status.parent / LIVE_GAINS, getattr(self.fan, "n", 1), ""
        try:
            g = json.loads(path.read_text())
            g = g.get(f"fan{n}") if isinstance(g, dict) else None
            if g is not None:
                num = lambda v, lo, hi: (isinstance(v, (int, float)) and  # noqa: E731
                                         not isinstance(v, bool) and lo <= v <= hi)
                if not isinstance(g, dict) or not all(num(g.get(k), 0, GAIN_MAX[k]) for k in GAINS):
                    raise ValueError(f"kp, ki and kd must be numbers within {GAIN_MAX}")
                if "target_c" in g and not num(g["target_c"], *TARGET_RANGE):
                    raise ValueError(f"target_c must be a number within {TARGET_RANGE}")
                p = self.pid
                new = (float(g["kp"]), float(g["ki"]), float(g["kd"]),
                       float(g.get("target_c", p.target)))
                if (p.kp, p.ki, p.kd, p.target) != new:
                    p.kp, p.ki, p.kd, p.target = new
                    self.say(f"fan_pid: gains of fan {n} now kp {p.kp:g} ki {p.ki:g} kd {p.kd:g} "
                             f"target {p.target:g} C ({path.name})", named=False)
        except FileNotFoundError:
            pass
        except Exception as e:                  # (nothing here may stop the loop that cools)
            err = str(e)
        if err != self.live_err:
            self.say(f"fan_pid: {path}: {err}: the gains stay" if err else
                     f"fan_pid: {path} reads again", named=False)
            self.live_err = err

    def step(self) -> dict:
        now = self.clock()
        self.live_gains()
        r = self.src.read()
        tune, lead, filt = "", False, None
        if r is not None and r["die_c"] is None:        # a source whose die readings stopped
            die, reg, ctl = None, r["reg_c"], None
            self.prev_t = None
            duty, why = 100.0, r.get("lost") or "die reading lost"
            if self.tuner:
                self.tuner.abort(why)
                self.end_tune()
        elif r is not None:
            dt = 0.0 if self.prev_t is None else now - self.prev_t
            self.prev_t, self.last_ok = now, now
            die, reg = r["die_c"], r["reg_c"]
            hotter = die if reg is None else max(die, reg)
            self.max_c = hotter if self.max_c is None else max(self.max_c, hotter)
            # the regulator against its own target, in the die's terms; the tuning, the die alone
            ctl = die
            if reg is not None and not self.tuner:
                ctl = max(die, reg - (self.reg_target - self.pid.target))
                lead = ctl > die
            filt = self.filter(ctl, now)
            why = self.hot(die, reg)                                # on the raw readings
            if self.tuner:
                if why:
                    self.tuner.abort(why)
                duty, tune = self.tuner.step(filt, now), self.tuner.state()
                if self.tuner.done:
                    self.end_tune()
            else:
                duty = self.pid.step(filt, dt)
            if why:
                duty = 100.0
        else:
            die = reg = ctl = None
            self.prev_t = None
            blind = now - self.last_ok
            if blind >= self.a.blind_s:
                duty, why = 100.0, f"no reading for {blind:.0f} s"
                if self.tuner:
                    self.tuner.abort(why)
                    self.end_tune()
            else:
                duty, why = self.duty, "no reading: holding"
            if self.tuner:
                tune = self.tuner.state()
        self.duty, self.why = round(duty), why
        s = self.fan.set(self.duty)
        row = {"die_c": die, "reg_c": reg, "ctl_c": ctl, "filt_c": filt, "lead": lead,
               "gov_c": (r or {}).get("gov_c"), "sysmon_c": (r or {}).get("sysmon_c"),
               "duty": s.get("duty", self.duty), "why": why,
               "source": self.src.name, "state": (r or {}).get("state", ""),
               "note": getattr(self.src, "note", ""), "tune": tune,
               **{k: (r or {}).get(k) for k in ("mode", "dimms", "acc", "gov_frozen",
                                                 "sysmon_frozen", "gov_age_s", "power")}}
        self.show(row, now)
        if self.status:
            self.write_status(row)
        return row

    def status_doc(self, row: dict, now: float | None = None) -> dict:
        """The status file's contents after a step (docs/bcu1525.md section 15): None where a
        reading is missing (the jtag source: the die alone)."""
        now = time.time() if now is None else now
        pw = row.get("power") or {}
        rnd = lambda v, n=2: None if v is None else round(v, n)          # noqa: E731
        return {
            "time": round(now, 3),
            "iso": datetime.fromtimestamp(now).astimezone().isoformat(timespec="seconds"),
            "interval_s": self.a.interval,
            "source": row["source"],
            "state": {"mode": row.get("mode"), "dimms": row.get("dimms"), "acc": row.get("acc"),
                      "text": row["state"] or None},
            "die_c": rnd(row["die_c"]), "gov_c": rnd(row.get("gov_c")),
            "sysmon_c": rnd(row.get("sysmon_c")), "gov_frozen": row.get("gov_frozen"),
            "sysmon_frozen": row.get("sysmon_frozen"), "gov_age_s": row.get("gov_age_s"),
            "reg_c": rnd(row["reg_c"]),
            "power": {"vccint_in_w": rnd(pw.get("vccint_in_w"), 3),
                      "vin_v": rnd(pw.get("vin_v"), 3), "age_s": pw.get("age_s"),
                      "stale": pw.get("stale"), "label": POWER_LABEL},
            "fan": {"duty": row["duty"], "why": row["why"] or None,
                    "mode": "tune" if row["tune"] else "pid", "target_c": self.pid.target,
                    "filt_c": rnd(row["filt_c"]),
                    "lead": None if row["filt_c"] is None else "vreg" if row["lead"] else "die",
                    "gains": {"kp": self.pid.kp, "ki": self.pid.ki, "kd": self.pid.kd},
                    "i": rnd(self.pid.i)},
            "fan_pid": {"pid": os.getpid(), "version": VERSION, "tree": ROOT.name,
                        "started": round(self.started, 3)},
            "card": self.card or None, "fan_no": getattr(self.fan, "n", 1),
        }

    def write_status(self, row: dict) -> None:
        """The status file, whole (a rename over it); a failure is said once and the loop goes
        on."""
        try:
            write_whole(self.status, json.dumps(self.status_doc(row), indent=2,
                                                allow_nan=False) + "\n",
                        sync=False, mode=0o644)
            err = ""
        except (OSError, ValueError) as e:
            err = str(e)
        if err != self.status_err:
            self.say(f"fan_pid: status file {self.status}: {err}" if err else
                     f"fan_pid: status file {self.status} written again")
            self.status_err = err

    def end_tune(self) -> None:
        """The tuning over: the PID on with the new gains (saved unless --tune-dry-run), or after
        an abort with the ones the loop started with."""
        t, self.tuner, p = self.tuner, None, self.pid
        if t.result is None:
            self.say(f"fan_pid: tuning aborted: {t.error}; going on with kp {p.kp:g} ki {p.ki:g} "
                     f"kd {p.kd:g}, nothing saved")
            return
        g = dict(t.result, time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), host=socket.gethostname())
        self.pid = Pid(p.target, g["kp"], g["ki"], g["kd"], p.lo, p.hi, p.smooth)
        # the relay's mean duty held the target: the integral starts there, no bump
        self.pid.i = min(max(g["duty"] - p.lo, 0.0), p.hi - p.lo)
        status, saved = 0, "not saved (--tune-dry-run)"
        if not self.a.tune_dry_run:
            path = Path(self.a.gains_file).expanduser()
            try:
                save_gains(path, g)
                saved = f"saved to {path}"
            except OSError as e:
                status, saved = 1, f"NOT saved ({e})"
        self.say(f"fan_pid: tuned ({g['rule']}): Ku {g['ku']:g} % per C, Tu {g['tu_s']:g} s, "
                 f"amplitude {g['amplitude_c']:g} C over {g['cycles']} cycles; kp {g['kp']:g} "
                 f"ki {g['ki']:g} kd {g['kd']:g}, {saved}")
        if self.a.tune_exit:
            self.exit = status
        else:
            self.say("fan_pid: going on with the new gains")

    def say(self, msg: str, named: bool = True) -> None:
        """A kept line. named False: not the card's name in it, for a line that can come often:
        a watcher on another host finds a card's readings in the pane by that name."""
        if self.card and named:
            msg = msg.replace("fan_pid:", f"fan_pid [{self.card}]:", 1)
        self.out.write(("\r\033[K" if self.tty else "") + msg + "\n")
        self.out.flush()

    def show(self, row: dict, now: float) -> None:
        f = lambda v: "  --  " if v is None else f"{v:5.1f}"            # noqa: E731
        both = (f" (gov {f(row['gov_c'])} sysmon {f(row['sysmon_c'])})"
                if row.get("gov_c") is not None or row.get("sysmon_c") is not None else "")
        pin = (row.get("power") or {}).get("vccint_in_w")
        line = (f"{time.strftime('%H:%M:%S')}  " + (f"{self.card}  " if self.card else "")
                + f"die {f(row['die_c'])} C{both}  vreg {f(row['reg_c'])} C"
                f"  VCCINT in {f(pin)} W"
                f"  (hotter max {f(self.max_c)})  filt {f(row['filt_c'])} C"
                f"  fan {row['duty']:5.1f} %  "
                + ("" if row["filt_c"] is None else f"err {row['filt_c'] - self.pid.target:+5.1f} "
                   + ("(vreg) " if row["lead"] else ""))
                + (f"tune: {row['tune']}" if row["tune"] else
                   f"I {self.pid.i:4.1f}" + (f" D {self.pid.d:+5.1f}" if self.pid.kd else ""))
                + f"  [{row['source']}{' ' + row['state'] if row['state'] else ''}]"
                + (f"  {row['why']}" if row["why"] else "")
                + (f"  ({row['note']})" if row["note"] else ""))
        changed, self.shown_why = row["why"] != self.shown_why, row["why"]
        if self.tty:
            commit = now - self.committed >= self.a.log_s or changed
            self.out.write("\r\033[K" + line + ("\n" if commit else ""))
            if commit:
                self.committed = now
        elif now - self.committed >= self.a.log_s or changed:
            self.out.write(line + "\n")
            self.committed = now
        self.out.flush()
        if self.csv:
            self.csv.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), row["source"], row["die_c"],
                               row["reg_c"], row["duty"], round(self.pid.p, 2),
                               round(self.pid.i, 2), round(self.pid.d, 2), row["why"],
                               row["state"], row["tune"],
                               None if row["filt_c"] is None else round(row["filt_c"], 2),
                               (row.get("power") or {}).get("vccint_in_w"),
                               (row.get("power") or {}).get("vin_v")]
                              [:self.ncol])
            self.csv_f.flush()


def args_of(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=float, default=45.0, help="the die's (C)")
    ap.add_argument("--reg-target", type=float,
                    help=f"the regulator's (C; default --target + {REG_ABOVE_C:g})")
    ap.add_argument("--kp", type=float,
                    help="%% duty per C above the target (default: the gains file's, else 4)")
    ap.add_argument("--ki", type=float, help="%% duty per C-second (the gains file's, else 0.04)")
    ap.add_argument("--kd", type=float, help="%% duty per C/s of rise (the gains file's, else 0)")
    ap.add_argument("--gains-file", default=GAINS_FILE,
                    help="the gains --tune saves and the loop starts from (%(default)s)")
    ap.add_argument("--filter-s", type=float, default=10.0,
                    help="the control temperature's filter time constant (%(default)s s; 0: none)")
    ap.add_argument("--lo", type=float, default=55.0,
                    help="the lowest duty (55: the fan stands still; it turns from 60)")
    ap.add_argument("--hi", type=float, default=100.0)
    ap.add_argument("--hot-c", type=float, default=75.0,
                    help=f"the die at this: 100 %% (at most {CEIL_C:g})")
    ap.add_argument("--reg-hot-c", type=float, default=75.0,
                    help=f"the regulator at this: 100 %% (at most {CEIL_C:g})")
    ap.add_argument("--source", choices=("auto", "pcie", "jtag"), default="auto")
    ap.add_argument("--pci", default=PCI,
                    help="fan 1's card's PCI function: its xdma device found by it (%(default)s)")
    ap.add_argument("--dev", help="fan 1's card's xdma device as given, in place of --pci's")
    ap.add_argument("--serial", default=SERIAL,
                    help="fan 1's card's FT4232H serial (jtag; %(default)s)")
    ap.add_argument("--interval", type=float, default=2.0, help="seconds between steps")
    ap.add_argument("--blind-s", type=float, default=60.0,
                    help="seconds without a reading before 100 %%")
    ap.add_argument("--stale-s", type=float, default=STALE_S,
                    help="seconds a die reading may read the same before it has stopped (pcie: "
                         "the governor's, the SYSMON's; neither moving: 100 %%; 0: off)")
    ap.add_argument("--watchdog", type=int, default=15,
                    help="the firmware's: seconds without a command before it sets 100 %%")
    ap.add_argument("--log-s", type=float, default=30.0, help="seconds between kept lines")
    ap.add_argument("--csv", help="append every step to this CSV")
    ap.add_argument("--status", default=STATUS_FILE,
                    help="each step's status, as JSON, renamed over this file (%(default)s; '': "
                         "none)")
    ap.add_argument("--fanctl", help=f"the fan controller's library (default {FANCTL}, the "
                                     "two-fan one, else ~/fanctl.py)")
    ap.add_argument("--port", help="the fan controller's serial port (auto-detected)")
    f2 = ap.add_argument_group("fan 2 (the second card)")
    f2.add_argument("--no-fan2", action="store_true", help="fan 1 alone; fan 2 left as it is")
    f2.add_argument("--serial2", default=SERIAL2,
                    help="its card's FT4232H serial (jtag; %(default)s)")
    f2.add_argument("--pci2", default=PCI2, help="its card's PCI function (%(default)s)")
    f2.add_argument("--dev2", help="its card's xdma device as given, in place of --pci2's")
    f2.add_argument("--target2", type=float, help="its die's target (C; default --target)")
    f2.add_argument("--status2", default=STATUS_FILE2,
                    help="its status file (%(default)s; '': none)")
    f2.add_argument("--csv2", help="its CSV (default: --csv's name with a 2; '': none)")
    t = ap.add_argument_group("tuning")
    t.add_argument("--tune", action="store_true",
                   help="relay-autotune kp, ki and kd around --target, save them, go on with them")
    t.add_argument("--tune-hi", type=float, default=80.0,
                   help="the relay's duty above the target (%(default)s; 100 is loud)")
    t.add_argument("--tune-lo", type=float,
                   help="the relay's duty below the target (default --lo: the fan still)")
    t.add_argument("--tune-hyst", type=float, default=1.0,
                   help="the relay's hysteresis (%(default)s C; above the filtered noise band)")
    t.add_argument("--tune-cycles", type=int, default=4,
                   help="cycles in a row that agree, after the first (%(default)s)")
    t.add_argument("--tune-rule", choices=tuple(RelayTune.RULES), default="tl-pi",
                   help="the gains from Ku and Tu (%(default)s: Tyreus-Luyben PI)")
    t.add_argument("--tune-stall-s", type=float, default=900.0,
                   help="seconds without a crossing: abort (%(default)s)")
    t.add_argument("--tune-max-s", type=float, default=3600.0,
                   help="seconds without agreeing cycles: abort (%(default)s)")
    t.add_argument("--tune-dry-run", action="store_true", help="tune, but save nothing")
    t.add_argument("--tune-exit", action="store_true",
                   help="exit once tuned (an aborted tuning goes on as the controller)")
    a = ap.parse_args(argv)
    if not 0 <= a.lo <= a.hi <= 100:
        ap.error("--lo and --hi: 0 <= lo <= hi <= 100")
    for k in ("hot_c", "reg_hot_c"):
        if getattr(a, k) > CEIL_C:
            print(f"fan_pid: --{k.replace('_', '-')} {getattr(a, k):g} is above the "
                  f"{CEIL_C:g} C ceiling: {CEIL_C:g}", file=sys.stderr)
            setattr(a, k, CEIL_C)
    if a.reg_target is None:
        a.reg_target = a.target + REG_ABOVE_C
    if a.reg_target >= a.reg_hot_c:
        ap.error("--reg-target must be below --reg-hot-c")
    if a.target2 is None:
        a.target2 = a.target
    a.reg_target2 = a.target2 + (a.reg_target - a.target)
    if a.reg_target2 >= a.reg_hot_c:
        ap.error(f"--target2: its regulator's target {a.reg_target2:g} C must be below --reg-hot-c")
    if a.csv2 is None and a.csv:
        p = Path(a.csv)
        a.csv2 = str(p.with_name(p.stem + "2" + p.suffix))
    if a.fanctl is None:
        a.fanctl = FANCTL if Path(FANCTL).expanduser().exists() else "~/fanctl.py"
    if a.tune_lo is None:
        a.tune_lo = a.lo
    if not 0 <= a.tune_lo < a.tune_hi <= 100:
        ap.error("--tune-lo and --tune-hi: 0 <= lo < hi <= 100")
    if a.tune_hyst < 0 or a.tune_cycles < 2 or a.filter_s < 0:
        ap.error("--tune-hyst >= 0, --tune-cycles >= 2 and --filter-s >= 0")
    if a.tune and a.target + a.tune_hyst >= a.hot_c:
        ap.error("--tune: --target + --tune-hyst must be below --hot-c")
    return a


def main(argv=None) -> int:
    a = args_of(argv)
    gains = resolve_gains(a)
    F = load("fanctl", Path(a.fanctl).expanduser())
    logs = Path(FAN_DIR).expanduser()
    for n, serial, pci, dev in ((1, a.serial, a.pci, a.dev),) + \
            (() if a.no_fan2 else ((2, a.serial2, a.pci2, a.dev2),)):
        s = "" if n == 1 else "2"
        if a.source != "jtag" and not (pci or dev):
            raise SystemExit(f"fan_pid: fan {n}'s card has no PCI function: --pci{s} or --dev{s}, "
                             f"or BCU_CARD{n - 1}_BDF")
        if a.source != "pcie" and not serial:
            raise SystemExit(f"fan_pid: fan {n}'s card has no JTAG serial: --serial{s}, or "
                             f"BCU_CARD{n - 1}_SERIAL")
    logs.mkdir(parents=True, exist_ok=True)

    def source(serial, pci, dev, log):
        make_jtag = lambda: JtagSource(serial, a.interval, log=log)                 # noqa: E731
        make_pcie = lambda: PcieSource(dev, a.stale_s, pci=None if dev else pci)    # noqa: E731
        return {"pcie": make_pcie, "jtag": make_jtag,
                "auto": lambda: AutoSource(make_pcie(), make_jtag)}[a.source]()

    def stop(sig, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGHUP, stop)          # the ssh session gone: the same clean stop
    srcs, ports, loops = [], [], []
    try:
        with F.FanController(a.port) as fan:
            fan.set_timeout(a.watchdog)
            why2 = "--no-fan2" if a.no_fan2 else has_fan2(F, fan)
            # (card, fan, JTAG serial, PCI function, device, die target, vreg target, status, csv)
            cards = [(f"{NAME} fan 1", 1, a.serial, a.pci, a.dev, a.target, a.reg_target, a.status,
                      a.csv)]
            if not why2:
                cards.append((f"{NAME2} fan 2", 2, a.serial2, a.pci2, a.dev2, a.target2, a.reg_target2,
                              a.status2, a.csv2 or ""))
            two = len(cards) > 1
            for card, n, serial, pci, dev, target, reg_target, status, csv_path in cards:
                srcs.append(source(serial, pci, dev, logs / ("jtag.log" if n == 1 else
                                                             f"jtag{n}.log")))
                ports.append(FanPort(fan, n))
                loops.append(Loop(ports[-1], srcs[-1], Pid(target, a.kp, a.ki, a.kd, a.lo, a.hi),
                                  a, status=Path(status).expanduser() if status else None,
                                  card=card if two else "", csv_path=csv_path,
                                  tune=a.tune and n == 1, reg_target=reg_target))
            print(f"fan_pid {VERSION}: "
                  + "; ".join(f"{c[0] + ': ' if two else ''}target die {c[5]:g} C, vreg "
                              f"{c[6]:g} C, status {lp.status or 'none'}"
                              for c, lp in zip(cards, loops))
                  + f" (whichever is further above its own), duty {a.lo:g}..{a.hi:g} %, gains "
                  f"{gains}, source {a.source}; Ctrl-C leaves the "
                  + ("duties as they are" if two else "duty as it is"), flush=True)
            if why2 and not a.no_fan2:
                print(f"fan_pid: WARNING: fan 2 not driven ({why2}): fan 1 alone", flush=True)
            if a.tune:
                print(f"fan_pid: tuning ({a.tune_rule}, fan 1's die): the fan at {a.tune_lo:g} % "
                      f"below {a.target - a.tune_hyst:g} C and {a.tune_hi:g} % above "
                      f"{a.target + a.tune_hyst:g} C until {a.tune_cycles} cycles agree (at most "
                      f"{a.tune_max_s:g} s, {a.tune_stall_s:g} s without a crossing); then "
                      + ("nothing saved" if a.tune_dry_run else
                         f"saved to {Path(a.gains_file).expanduser()}")
                      + (", and exit" if a.tune_exit else ", and on with the new gains"),
                      flush=True)
            clean = False
            try:
                while all(lp.exit is None for lp in loops):
                    t0 = time.monotonic()
                    for lp in loops:
                        lp.step()
                    time.sleep(max(0.0, a.interval - (time.monotonic() - t0)))
                clean = True
            except KeyboardInterrupt:
                clean = True
            finally:
                if clean:
                    fan.set_timeout(0)
                    print("\nfan_pid: stopped, " + ", ".join(
                        f"fan {p.n} left at {lp.duty} %" for p, lp in zip(ports, loops))
                        + ", watchdog off", flush=True)
                else:                           # an error: both fans full, the watchdog on
                    for p in ports:
                        try:
                            p.set(100)
                        except Exception:
                            pass
                    print("\nfan_pid: error: the fans set to 100 %, the watchdog left on",
                          file=sys.stderr, flush=True)
    except (F.FanError, RuntimeError, OSError) as e:
        print(f"fan_pid: {e}", file=sys.stderr)
        return 1
    finally:
        for src in srcs:
            if hasattr(src, "close"):
                src.close()
    return loops[0].exit or 0


if __name__ == "__main__":
    sys.exit(main())
