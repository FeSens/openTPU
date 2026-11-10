#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["pyserial>=3.5"]
# ///
"""Host-side control for the XIAO nRF52840 fan controller.

CLI:
    fanctl.py status
    fanctl.py set 60                  # fixed duty %
    fanctl.py --fan 2 set 60          # fan 2 (pin 028, no tach): set / up / down / auto only
    fanctl.py rpm 4000                # hold a speed (PID in the firmware)
    fanctl.py up [n] / down [n]       # duty %, or rpm when in rpm mode
    fanctl.py pid [kp ki kd]          # show / set speed-loop gains
    fanctl.py timeout 10
    fanctl.py auto --temp-cmd "cat /path/to/fpga_temp"   # follow a temperature

Library:
    from fanctl import FanController
    with FanController() as fan:
        fan.set_rpm(4000)
        fan.set(60, fan=2)
        print(fan.status())             # fan 2's duty is "duty2"
"""

import argparse
import subprocess
import sys
import time

import serial
from serial.tools import list_ports

SEEED_VID = 0x2886
XIAO_NRF52840_PIDS = {0x8044, 0x8045}  # XIAO nRF52840 / Sense, application mode


def find_port():
    for p in list_ports.comports():
        if p.vid == SEEED_VID and p.pid in XIAO_NRF52840_PIDS:
            return p.device
    raise RuntimeError("fan controller not found on USB; pass --port")


class FanError(RuntimeError):
    pass


def _to(fan, line):
    """`line` for `fan`: fan 2's commands carry a leading "2", fan 1's go as they always have."""
    if fan not in (1, 2):
        raise ValueError(f"fan {fan}: the controller has fans 1 and 2")
    return line if fan == 1 else f"2 {line}"


def _parse_value(v):
    for conv in (int, float):
        try:
            return conv(v)
        except ValueError:
            pass
    return v


class FanController:
    def __init__(self, port=None, timeout=2.0):
        # Never open at 1200 baud: that resets the board into its bootloader.
        # Exclusive so a manual command can't fight a running `auto` loop.
        self.ser = serial.Serial(port or find_port(), 115200, timeout=timeout, exclusive=True)
        self.ser.reset_input_buffer()

    def close(self):
        self.ser.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def command(self, line):
        """Send one command, return the parsed OK reply as a dict."""
        self.ser.write((line.strip() + "\n").encode())
        while True:
            reply = self.ser.readline().decode(errors="replace").strip()
            if not reply:
                raise FanError(f"no reply to {line!r}")
            if reply.startswith("#"):
                continue
            if reply.startswith("ERR"):
                raise FanError(reply)
            if reply.startswith("OK"):
                return {k: _parse_value(v) for k, v in (kv.split("=") for kv in reply.split()[1:])}

    def status(self):
        return self.command("STATUS")

    def set(self, duty, fan=1):
        return self.command(_to(fan, f"SET {round(duty, 2):g}"))

    def set_rpm(self, rpm):
        return self.command(f"RPM {int(round(rpm))}")

    def up(self, step=None, fan=1):
        return self.command(_to(fan, "UP" if step is None else f"UP {step}"))

    def down(self, step=None, fan=1):
        return self.command(_to(fan, "DOWN" if step is None else f"DOWN {step}"))

    def pid(self, kp=None, ki=None, kd=None):
        if kp is None:
            return self.command("PID")
        return self.command(f"PID {kp} {ki} {kd}")

    def set_timeout(self, seconds):
        return self.command(f"TIMEOUT {int(seconds)}")


def duty_for_temp(temp, t_min, t_max, d_min, d_max):
    """Linear fan curve: d_min at or below t_min, d_max at or above t_max."""
    if temp <= t_min:
        return d_min
    if temp >= t_max:
        return d_max
    return d_min + (d_max - d_min) * (temp - t_min) / (t_max - t_min)


def read_temp(cmd):
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10, check=True)
    return float(out.stdout.split()[0])


def run_auto(fan, args):
    # If this script dies, the firmware falls back to 100 % after the watchdog expires.
    fan.set_timeout(args.watchdog)
    while True:
        try:
            temp = read_temp(args.temp_cmd)
            duty = duty_for_temp(temp, args.t_min, args.t_max, args.d_min, args.d_max)
            msg = f"temp={temp:.1f}C"
        except (subprocess.SubprocessError, ValueError, IndexError) as e:
            duty, msg = 100, f"temp read failed ({e})"
        s = fan.set(duty, args.fan)
        if args.fan == 2:
            print(f"{msg} duty2={s['duty2']}", flush=True)
        else:
            print(f"{msg} duty={s['duty']} rpm={s['rpm']} stall={s['stall']}", flush=True)
        time.sleep(args.interval)


SWEEP_DUTIES = [100, 90, 80, 70, 60, 50, 40, 30, 25, 20, 18, 16, 14, 12, 11, 10,
                9.5, 9, 8.5, 8, 7.5, 7, 6.5, 6, 5.5, 5, 4, 3, 2, 1]
KICK_DUTY = 30  # must match KICK_DUTY in the firmware


def measure_rpm(fan, seconds):
    samples = []
    end = time.time() + seconds
    while time.time() < end:
        samples.append(fan.status()["rpm"])
        time.sleep(0.25)
    return sum(samples) / len(samples)


def run_sweep(fan, settle):
    """Measure this fan's duty -> rpm curve, stop duty and start duty."""
    fan.set_timeout(0)
    print("# running curve, stepping duty down from 100 %", flush=True)
    fan.set(100)
    time.sleep(settle)
    curve = []
    for d in SWEEP_DUTIES:
        fan.set(d)
        time.sleep(settle)
        rpm = measure_rpm(fan, 1.0)
        curve.append((d, rpm))
        print(f"duty {d:5.1f} %  rpm {rpm:6.0f}", flush=True)
        if rpm == 0:
            break
    running = [(d, r) for d, r in curve if r > 0]
    if not running:
        fan.set(100)
        sys.exit("error: no tach pulses at any duty - is the fan powered and yellow on D1?")
    stop_duty = curve[-1][0] if curve[-1][1] == 0 else None

    print("# start test: from standstill, raising duty until the fan turns", flush=True)
    fan.set(0)
    for _ in range(60):
        if fan.status()["rpm"] == 0:
            break
        time.sleep(0.5)
    start_duty = None
    d = stop_duty or 1
    while d <= 50:
        fan.set(d)
        time.sleep(settle)
        if measure_rpm(fan, 1.0) > 0:
            start_duty = d
            break
        d += 1
    fan.set(100)

    print()
    print(f"lowest steady speed: {running[-1][1]:.0f} rpm at {running[-1][0]} % duty")
    print(f"stops at:            {stop_duty} % duty" if stop_duty else "never stopped")
    print(f"starts from rest at: {start_duty} % duty" if start_duty else "did not start below 50 %")
    if start_duty and start_duty >= KICK_DUTY:
        print(f"!! raise KICK_DUTY in the firmware above {start_duty}")

    # Firmware table: ascending, strictly increasing rpm, anchored at the stop point.
    pts = sorted(running)
    table = [(stop_duty, 0)] if stop_duty else []
    for d, r in pts:
        if not table or r > table[-1][1]:
            table.append((d, round(r / 10) * 10))
    print("\n// paste into fan_controller.ino, replacing CURVE_DUTY / CURVE_RPM:")
    print("static const float CURVE_DUTY[] = {" + ", ".join(f"{d:g}" for d, _ in table) + "};")
    print("static const float CURVE_RPM[]  = {" + ", ".join(f"{r:g}" for _, r in table) + "};")


def print_status(s):
    print(" ".join(f"{k}={v}" for k, v in s.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="serial port (auto-detected by default)")
    ap.add_argument("--fan", type=int, choices=(1, 2), default=1,
                    help="which fan (default 1); fan 2 has no tach: set, up, down, auto")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("set", help="fixed duty %%").add_argument("duty", type=float)
    sub.add_parser("rpm", help="hold a fan speed, 0 stops").add_argument("rpm", type=int)
    sub.add_parser("up", help="duty %% or rpm, by mode").add_argument("step", type=int, nargs="?")
    sub.add_parser("down", help="duty %% or rpm, by mode").add_argument("step", type=int, nargs="?")
    sub.add_parser("pid", help="show or set speed-loop gains").add_argument("gains", type=float, nargs="*")
    sub.add_parser("timeout").add_argument("seconds", type=int)
    a = sub.add_parser("auto", help="follow a temperature read by a shell command")
    a.add_argument("--temp-cmd", required=True, help="shell command that prints the FPGA temp in C")
    a.add_argument("--t-min", type=float, default=45.0)
    a.add_argument("--t-max", type=float, default=80.0)
    a.add_argument("--d-min", type=float, default=20.0)
    a.add_argument("--d-max", type=float, default=100.0)
    a.add_argument("--interval", type=float, default=2.0)
    a.add_argument("--watchdog", type=int, default=10, help="seconds without updates before fan goes to 100%%")
    sw = sub.add_parser("sweep", help="measure this fan's duty->rpm curve (fan must be connected)")
    sw.add_argument("--settle", type=float, default=5.0, help="seconds to wait at each duty")
    args = ap.parse_args()
    if args.fan == 2 and args.cmd in ("rpm", "pid", "sweep"):
        sys.exit(f"error: fan 2 has no tach: {args.cmd} needs one (use set, up, down or auto)")

    try:
        with FanController(args.port) as fan:
            if args.cmd == "status":
                print_status(fan.status())
            elif args.cmd == "set":
                print_status(fan.set(args.duty, args.fan))
            elif args.cmd == "rpm":
                print_status(fan.set_rpm(args.rpm))
            elif args.cmd == "pid":
                if len(args.gains) not in (0, 3):
                    sys.exit("error: pid takes no gains or exactly three (kp ki kd)")
                print_status(fan.pid(*args.gains))
            elif args.cmd == "up":
                print_status(fan.up(args.step, args.fan))
            elif args.cmd == "down":
                print_status(fan.down(args.step, args.fan))
            elif args.cmd == "timeout":
                print_status(fan.set_timeout(args.seconds))
            elif args.cmd == "auto":
                run_auto(fan, args)
            elif args.cmd == "sweep":
                run_sweep(fan, args.settle)
    except (FanError, RuntimeError, serial.SerialException) as e:
        sys.exit(f"error: {e}")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
