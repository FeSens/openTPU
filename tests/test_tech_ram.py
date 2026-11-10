"""The technology RAMs (rtl/tech/<family>/otpu_tech_ram.sv): the ASIC's, tiled onto SRAM macros
(tools/asic/fakeram.py's models, the leaves from tools/asic/flow.py), read what the generic,
inferred ones read, cycle for cycle, under random traffic (sim/tech/tb_tech_ram.sv): width tiles,
depth tiles with their registered read select, byte enables, the one-port RAM."""
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "asic"))
import fakeram  # noqa: E402
import flow  # noqa: E402

WMAX, NMAX = 128, 4096          # rtl/tech/asic/otpu_tech_ram.sv's otpu_tech_asic


def tiles(kind: str, w: int, n: int) -> set:
    tn = min(n, NMAX)
    ws = [WMAX] * (w // WMAX) + ([w % WMAX] if w % WMAX else [])
    return {fakeram.Macro(kind, tn, x) for x in ws}


def _trace(tmp: Path, family: str, w: int, n: int, kind: int, seed: int) -> list[str]:
    obj = tmp / f"obj_{family}"
    obj.mkdir(parents=True)
    srcs = [ROOT / f"rtl/tech/{family}/otpu_tech_ram.sv", ROOT / "sim/tech/tb_tech_ram.sv"]
    if family == "asic" and kind != 4:          # (the read-ahead RAM is flip-flops there)
        ms = tiles({2: "1rw", 3: "1rw1r"}.get(kind, "1r1w"), w, n)
        for m in ms:
            srcs.append(fakeram.write(m, tmp / "macros")["v"])
        (tmp / "map.sv").write_text(flow.sram_map(sorted(ms, key=lambda m: m.name)))
        srcs.append(tmp / "map.sv")
    r = subprocess.run(["verilator", "--binary", "--timing", "-Wno-fatal", "-Wno-lint",
                        "-Wno-style", "--top-module", "tb_tech_ram", "-Mdir", str(obj),
                        f"-GW={w}", f"-GN={n}", f"-GKIND={kind}", *map(str, srcs)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    run = subprocess.run([str(obj / "Vtb_tech_ram"), f"+seed={seed}"], capture_output=True,
                         text=True, timeout=300, check=True)
    return [l for l in run.stdout.splitlines() if l[:1].isdigit()]


@pytest.mark.skipif(shutil.which("verilator") is None, reason="no verilator")
@pytest.mark.parametrize("w,n,kind", [(40, 8192, 0),      # depth tiles
                                      (200, 64, 0),       # width tiles (128 + 72)
                                      (64, 256, 1),       # byte enables
                                      (130, 8192, 2),     # one port, both tilings
                                      (40, 8192, 3),      # two ports (1RW1R), read first
                                      (200, 64, 3),
                                      (32, 64, 4),        # read ahead (its bypass)
                                      (200, 32, 4)])
def test_asic_rams_match_generic(tmp_path, w, n, kind):
    g = _trace(tmp_path / "g", "generic", w, n, kind, 3)
    a = _trace(tmp_path / "a", "asic", w, n, kind, 3)
    assert len(g) > 10000 and a == g


def test_families_define_the_same_rams():
    """The public RAM modules have the same parameters and ports in every family."""
    def iface(path: Path) -> dict:
        txt = re.sub(r"//.*", "", path.read_text())
        mods = {}
        for m in re.finditer(r"module\s+(\w+)\s*#\((.*?)\)\s*\((.*?)\);", txt, re.S):
            norm = lambda s: re.sub(r"\s+", " ", s).strip()   # noqa: E731
            mods[m.group(1)] = (norm(m.group(2)), norm(m.group(3)))
        return mods
    pub = {"otpu_ram_1r1w", "otpu_ram_1r1w_2c", "otpu_ram_1r1w_2c_be", "otpu_ram_1r1w_ra",
           "otpu_ram_1r1w_be", "otpu_ram_1rw", "otpu_ram_1rw1r"}
    g = iface(ROOT / "rtl/tech/generic/otpu_tech_ram.sv")
    a = iface(ROOT / "rtl/tech/asic/otpu_tech_ram.sv")
    assert set(g) == pub and pub <= set(a)
    assert {k: g[k] for k in pub} == {k: a[k] for k in pub}
