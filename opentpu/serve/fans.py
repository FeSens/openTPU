"""The fans on the dashboard: fan_pid's state from its status files, and the gains a person sets on
the page, in the file the loop reads at every step (boards/bcu1525/scripts/fan_pid.py;
docs/bcu1525.md section 15). fan_pid owns the fans: it runs on without a server, and whatever is
set here its overrides stay (the die or the regulator at 75 C: 100 %)."""
import json
import os
import threading
import time
from pathlib import Path

DIR = "~/otpu-build/bcu-fan"
STATUS = {1: "status.json", 2: "status2.json"}
LIVE_GAINS = "gains_live.json"
GAIN_MAX = {"kp": 100.0, "ki": 10.0, "kd": 1000.0}      # (above these: a typo; fan_pid's too)
TARGET_RANGE = (40.0, 70.0)                             # (the die's: under the 75 C overrides)
_lock = threading.Lock()                                # (the file is every fan's)


def directory() -> Path:
    return Path(os.environ.get("OTPU_FAN_DIR") or DIR).expanduser()


def state(now: float | None = None) -> list | None:
    """Each fan's last step as its status file says it, age_s how old that is (a stopped loop's
    file keeps its last values); None when there is no status file: a host without fan_pid."""
    now = time.time() if now is None else now
    out = []
    for n, name in STATUS.items():
        try:
            s = json.loads((directory() / name).read_text())
            f = s["fan"]
            out.append({"fan": s.get("fan_no") or n, "card": s.get("card"),
                        "age_s": round(now - s["time"], 1), "interval_s": s.get("interval_s"),
                        "die_c": s.get("die_c"), "reg_c": s.get("reg_c"), "duty": f.get("duty"),
                        "why": f.get("why"), "mode": f.get("mode"), "lead": f.get("lead"),
                        "target_c": f.get("target_c"), "gains": f.get("gains"), "i": f.get("i"),
                        "version": (s.get("fan_pid") or {}).get("version")})
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
    return out or None


def _num(v, lo: float, hi: float) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and lo <= v <= hi


def set_gains(fan, body: dict) -> dict:
    """One fan's kp, ki and kd, and target_c if given (else the one it has here), for the loop's
    next step, the others' as they are, the file replaced whole. ValueError: what is wrong with
    them (nothing written)."""
    if not isinstance(fan, int) or isinstance(fan, bool) or fan not in STATUS:
        raise ValueError(f"fan: one of {sorted(STATUS)}")
    g = {}
    for k, top in GAIN_MAX.items():
        if not _num(body.get(k), 0, top):
            raise ValueError(f"{k}: a number from 0 to {top:g}")
        g[k] = float(body[k])
    if "target_c" in body:
        if not _num(body["target_c"], *TARGET_RANGE):
            raise ValueError("target_c: a number from {:g} to {:g}".format(*TARGET_RANGE))
        g["target_c"] = float(body["target_c"])
    path = directory() / LIVE_GAINS
    with _lock:
        try:
            cur = json.loads(path.read_text())
            cur = cur if isinstance(cur, dict) else {}
        except (OSError, ValueError):
            cur = {}
        old = cur.get(f"fan{fan}")
        if "target_c" not in g and isinstance(old, dict) and "target_c" in old:
            g["target_c"] = old["target_c"]
        cur[f"fan{fan}"] = g
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(cur, indent=2) + "\n")
        os.replace(tmp, path)
    return g
