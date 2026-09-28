"""setup_pcie.sh --rescan's link recovery (opentpu/host/pcie/relink.sh) on a fake sysfs with a
fake setpci: the plain remove + rescan first, and only when the card is not back, Retrain Link,
then Link Disable, then a secondary bus reset on its upstream port, with a rescan after each; the
port held in D0 meanwhile and its power control restored after.
"""
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HOST = ROOT / "opentpu" / "host"
LIB = HOST / "pcie" / "relink.sh"
PORT, CARD = "0000:00:01.0", "0000:01:00.0"

# The fake hardware: the card's link comes back after BACK (rescan: by itself, as on omarchy;
# retrain | disable | reset: after that step on the port; never). setpci keeps the registers in
# files and logs each call with the port's power/control at that moment.
DRIVER = r"""
set -euo pipefail
T=$1 BACK=$2 PRESENT=$3
pci=$T/bus/pci state=$T/run-ports
note() { echo "note $*"; }
ok() { echo "ok $*"; }
cards() {
  local d
  for d in "$pci"/devices/*; do
    [[ -f $d/vendor && $(cat "$d/vendor") == 0x10ee && $(cat "$d/device") == 0x7028 ]] && basename "$d"
  done
  return 0
}
source "$LIB"
relink_tries=2 relink_step=0 relink_hold=0
eval "real_$(declare -f rescan_bus)"
port=__PORT__ card=__CARD__
plug() {
  local d=$T/devices/pci0000:00/$port/$card
  mkdir -p "$d"; echo 0x10ee > "$d/vendor"; echo 0x7028 > "$d/device"
  ln -sfn "$d" "$pci/devices/$card"
}
remove_dev() { echo "remove $1" >> "$T/log"; rm -rf "$(cd -P "$pci/devices/$1" && pwd)"; rm -f "$pci/devices/$1"; }
rescan_bus() { real_rescan_bus; echo rescan >> "$T/log"; if [[ -f $T/up ]]; then plug; fi; }
setpci() {
  local dev=$2 op=$3 reg f old val mask
  echo "setpci $dev $op power=$(cat "$pci/devices/$dev/power/control")" >> "$T/log"
  reg=${op%%=*}; f=$T/cfg/${reg//[^A-Za-z0-9]/_}
  if [[ $op != *=* ]]; then cat "$f" 2>/dev/null || echo 0000; return 0; fi
  val=${op#*=}; mask=${val#*:}; val=${val%%:*}
  old=$(cat "$f" 2>/dev/null || echo 0000)
  printf '%04x\n' $(( (0x$old & ~0x$mask) | (0x$val & 0x$mask) )) > "$f"
  case "$BACK:$reg:$val:$mask" in
    retrain:CAP_EXP+10.w:0020:0020|disable:CAP_EXP+10.w:0000:0010|reset:BRIDGE_CONTROL:0000:0040)
      touch "$T/up" ;;
  esac
}
if [[ $PRESENT == 1 ]]; then plug; fi
if [[ $BACK == rescan ]]; then touch "$T/up"; fi
trap relink_cleanup EXIT
if rescan_cards; then echo RESULT back; else echo RESULT gone; fi
""".replace("__PORT__", PORT).replace("__CARD__", CARD)


def fake_sysfs(t: Path, lnkcap="00000000", lnksta="0000") -> Path:
    pci = t / "bus" / "pci"
    (pci / "devices").mkdir(parents=True)
    (pci / "drivers_autoprobe").write_text("1\n")
    (pci / "rescan").write_text("")
    port = t / "devices" / "pci0000:00" / PORT
    (port / "power").mkdir(parents=True)
    (port / "power" / "control").write_text("auto\n")
    (pci / "devices" / PORT).symlink_to(port)
    (t / "cfg").mkdir()
    (t / "cfg" / "CAP_EXP_0c_l").write_text(lnkcap + "\n")
    (t / "cfg" / "CAP_EXP_12_w").write_text(lnksta + "\n")
    (t / "log").write_text("")
    return pci


def run(t: Path, back: str, present=True):
    p = subprocess.run(["bash", "-c", DRIVER, "driver", str(t), back, "1" if present else "0"],
                       env={"LIB": str(LIB), "PATH": "/usr/bin:/bin"},
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    log = (t / "log").read_text().splitlines()
    writes = [line.split()[2] for line in log if line.startswith("setpci") and "=" in line.split()[2]]
    return p.stdout, log, writes


def power(t: Path) -> str:
    return (t / "devices" / "pci0000:00" / PORT / "power" / "control").read_text().strip()


def test_a_port_that_retrains_by_itself_is_not_touched(tmp_path):
    fake_sysfs(tmp_path)
    out, log, writes = run(tmp_path, "rescan")
    assert "RESULT back" in out
    assert log == [f"remove {CARD}", "rescan"]              # no setpci at all
    assert writes == []
    assert power(tmp_path) == "auto"                        # restored
    assert (tmp_path / "run-ports").read_text().split() == [PORT]


@pytest.mark.parametrize("back, want", [
    ("retrain", ["CAP_EXP+10.w=0020:0020"]),
    ("disable", ["CAP_EXP+10.w=0020:0020", "CAP_EXP+10.w=0010:0010", "CAP_EXP+10.w=0000:0010"]),
    ("reset", ["CAP_EXP+10.w=0020:0020", "CAP_EXP+10.w=0010:0010", "CAP_EXP+10.w=0000:0010",
               "BRIDGE_CONTROL=0040:0040", "BRIDGE_CONTROL=0000:0040"]),
])
def test_escalates_until_the_card_is_back(tmp_path, back, want):
    fake_sysfs(tmp_path)
    out, log, writes = run(tmp_path, back)
    assert "RESULT back" in out
    assert writes == want                                   # in order, and nothing after
    assert f"ok the card is back after: {dict(retrain='retrain', disable='disable', reset='secondary')[back]}" in out
    # every register access ran with the port held in D0; power/control restored after
    assert all(line.endswith("power=on") for line in log if line.startswith("setpci"))
    assert power(tmp_path) == "auto"
    # the link fields end as they started: Retrain Link and Link Disable clear, no bus reset
    assert (tmp_path / "cfg" / "CAP_EXP_10_w").read_text().strip() in ("0020", "0000")
    if back == "reset":
        assert (tmp_path / "cfg" / "BRIDGE_CONTROL").read_text().strip() == "0000"


def test_a_card_that_never_comes_back(tmp_path):
    fake_sysfs(tmp_path, lnkcap="00100000", lnksta="0000")  # the port reports the data link state
    out, log, writes = run(tmp_path, "never")
    assert "RESULT gone" in out
    assert len(writes) == 5
    assert log.count("rescan") == 1 + 3 * 2                 # the plain one, then 2 per step
    assert "data link down" in out
    assert power(tmp_path) == "auto"
    assert (tmp_path / "bus" / "pci" / "drivers_autoprobe").read_text().strip() == "1"


def test_the_port_recorded_by_the_last_rescan_is_used_when_the_card_is_gone(tmp_path):
    fake_sysfs(tmp_path)
    (tmp_path / "run-ports").write_text(PORT + "\n")
    out, log, writes = run(tmp_path, "disable", present=False)
    assert "RESULT back" in out
    assert "remove" not in " ".join(log)
    assert writes[-1] == "CAP_EXP+10.w=0000:0010"


def test_no_known_port(tmp_path):
    fake_sysfs(tmp_path)
    (tmp_path / "run-ports").write_text("0000:00:1c.4 ; rm -rf /\n")  # not on the bus, not a BDF
    out, log, writes = run(tmp_path, "reset", present=False)
    assert "RESULT gone" in out
    assert "no upstream port known" in out
    assert writes == []


def test_rescan_bus_turns_automatic_probing_off_and_back(tmp_path):
    pci = fake_sysfs(tmp_path)
    script = (f'set -euo pipefail; pci={pci}; state=/nonexistent; source "{LIB}"; '
              f'rescan_bus; echo 0 > $pci/drivers_autoprobe; saved_ap=1; held="{PORT}=auto"; '
              f'echo on > $pci/devices/{PORT}/power/control; relink_cleanup')
    subprocess.run(["bash", "-c", script], check=True, timeout=30)
    assert (pci / "rescan").read_text() == "1\n"
    assert (pci / "drivers_autoprobe").read_text() == "1\n"   # restored by relink_cleanup
    assert power(tmp_path) == "auto"


def test_setup_script_parses_and_documents_the_retrain():
    for f in (HOST / "setup_pcie.sh", LIB):
        subprocess.run(["bash", "-n", str(f)], check=True)
    help_ = subprocess.run(["bash", str(HOST / "setup_pcie.sh"), "--help"],
                           capture_output=True, text=True, check=True).stdout
    assert "Retrain Link" in help_ and "--rescan" in help_
