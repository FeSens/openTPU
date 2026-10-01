"""Build and run tb_xmon (sim/verilator/tb_xmon.sv: the debug build's DMA monitors, otpu_xmon, on
XDMA's master, otpu_axi_split2 and both otpu_mem_ch, in front of LiteDRAM's own controller (ldc)
or otpu_ldn_model (ldn)): clean runs, where every flag must stay clear and the counts must agree
with the master's, and one run per fault the bench puts in, where the monitors must set the flags
that place it and leave the others clear. Heavy (Verilator): one at a time.

    python tools/xmon_test.py [clean] [inject]       (default: both)

Exit status 1 if a run does not meet its expectation.
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("XMON_OUT", ROOT / "build" / "xmon"))
RTL = ROOT / "rtl" / "boards" / "ypcb-00338"
TB = ROOT / "sim" / "verilator"
SRC = [RTL / "otpu_afifo.sv", RTL / "otpu_axi_split2.sv", RTL / "otpu_mem_ch.sv", RTL / "otpu_xmon.sv",
       TB / "otpu_ldn_model.sv", TB / "otpu_ldc_model.sv", TB / "otpu_ldc_ch.v", TB / "tb_memch.sv",
       TB / "tb_xmon.sv"]
JOBS = os.environ.get("XMON_JOBS", "4")
PAR = int(os.environ.get("XMON_PAR", "2"))
BUILDS = {"ldc": dict(LDC=1), "ldn": dict(LDC=0)}

# FLAGS bits (otpu_xmon word 0)
X_WSHIFT, X_RSHIFT, X_WLAST, X_RLAST, X_WSTALL, X_BSTALL, X_RSTALL, X_PROTO = range(8)
N_WSHIFT, N_RSHIFT, N_PROTO = 8, 9, 10
FRAMING = {X_WLAST, X_RLAST, X_PROTO, N_PROTO}
STALLS = {X_WSTALL, X_BSTALL, X_RSTALL}

# every run: no regions shared with tb_memch's XDMA master; the accelerators idle unless a
# scenario says
BASE = ["+psh=0", "+psp=0", "+xacc0_ntx=0", "+xacc1_ntx=0"]
ACC = ["+xacc0_ntx=4000", "+xacc1_ntx=4000"]
G2 = ["+xp=200", "+up=375", "+cp=375"]
CLEAN = [
    ("ldc", "default", []),
    ("ldc", "seed2", ["+seed=2"]),
    ("ldc", "seed3", ["+seed=3", "+maxlen=64"]),
    ("ldc", "nogap", ["+gap=0", "+stall=0", "+outs=16"]),
    ("ldc", "stall90", ["+stall=90", "+outs=16", "+seed=4"]),
    ("ldc", "short", ["+maxlen=4", "+ntx=4000", "+seed=5"]),
    ("ldc", "gen2clk", G2 + ["+seed=6"]),
    ("ldn", "default", []),
    ("ldn", "seed7", ["+seed=7", "+maxlen=32", "+stall=60"]),
    # the accelerators' traffic on the same channels as XDMA's, the real controller: their writes
    # against XDMA's reads only, their reads against XDMA's writes only, both mixed; B's clocks
    # (100 / 133.33 / 125 MHz) and Gen2's (133.33 / 133.33 / 250)
    ("ldc", "accw_xr", ACC + ["+xacc0_wpct=100", "+xacc1_wpct=100", "+xw_ntx=0", "+xr_ntx=6000", "+seed=8"]),
    ("ldc", "accr_xw", ACC + ["+xacc0_wpct=0", "+xacc1_wpct=0", "+xr_ntx=0", "+xw_ntx=6000", "+seed=9"]),
    ("ldc", "accmix", ACC + ["+seed=10"]),
    ("ldc", "accmix_nogap", ACC + ["+gap=0", "+stall=0", "+gapw=0", "+gapc=0", "+mstall=0", "+seed=11"]),
    ("ldc", "accw_xr_g2", ACC + G2 + ["+xacc0_wpct=100", "+xacc1_wpct=100", "+xw_ntx=0", "+xr_ntx=6000",
                                      "+seed=12"]),
    ("ldc", "accr_xw_g2", ACC + G2 + ["+xacc0_wpct=0", "+xacc1_wpct=0", "+xr_ntx=0", "+xw_ntx=6000",
                                      "+seed=13"]),
    ("ldc", "accmix_g2", ACC + G2 + ["+seed=14", "+outs=16"]),
    ("ldn", "accmix", ACC + ["+seed=15"]),
]

# (inj, injn, flags that must be set, flags that must stay clear, a check on the words)
INJ = [
    (1, 700, {X_WSHIFT, N_WSHIFT}, STALLS | FRAMING,
     lambda w: w["X_WEXP"] - w["X_WGOT"] == 64),
    (2, 300, {N_WSHIFT}, {X_WSHIFT} | STALLS | FRAMING, None),
    (3, 9000, {X_WSTALL}, {X_WSHIFT, X_RSHIFT, N_WSHIFT, N_RSHIFT} | FRAMING,
     lambda w: w["X_STALL"] >> 2 & 1 == 0 and w["X_STALL"] >> 12 & 1 == 1),
    (4, 9000, {X_WSTALL}, {X_WSHIFT, X_RSHIFT, N_WSHIFT, N_RSHIFT} | FRAMING,
     lambda w: w["X_STALL"] >> 2 & 3 == 1),
    (5, 200, {N_RSHIFT}, {X_WSHIFT} | STALLS | FRAMING, None),
    (6, 500, {X_RSHIFT}, {X_WSHIFT, N_WSHIFT, N_RSHIFT} | STALLS | FRAMING, None),
    (7, 600, {X_BSTALL}, {X_WSHIFT, X_RSHIFT, N_WSHIFT, N_RSHIFT} | FRAMING,
     lambda w: w["X_STALL"] >> 5 & 1 == 0 and w["X_STALL"] >> 13 & 1 == 1),
]


def build(name: str, params: dict) -> Path:
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(repr(params).encode() + b"".join(p.read_bytes() for p in SRC)).hexdigest()
    exe = d / "obj" / "Vtb_xmon"
    if exe.exists() and (d / "key").exists() and (d / "key").read_text() == key:
        return exe
    cmd = (["verilator", "--binary", "-j", JOBS, "--top-module", "tb_xmon", "-Wno-fatal", "-Wno-lint",
            "-Wno-style", "-O3", "--x-assign", "0", "--x-initial", "0", "-Mdir", str(d / "obj")]
           + [f"-G{k}={v}" for k, v in params.items()] + [str(p) for p in SRC])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(f"BUILD FAILED {name}\n{r.stdout[-4000:]}\n{r.stderr[-4000:]}", flush=True)
        sys.exit(2)
    (d / "key").write_text(key)
    return exe


def run(exe: Path, log: Path, args: list[str]) -> tuple[dict, dict, str]:
    try:
        r = subprocess.run([str(exe)] + args, capture_output=True, text=True, timeout=3600)
        out = r.stdout + r.stderr
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode(errors="replace") + "\nWALL CLOCK TIMEOUT"
    log.write_text(out)
    words = {m[1]: int(m[2], 16) for m in re.finditer(r"^XMON (\w+)\s+([0-9a-f]{8})", out, re.M)}
    m = re.search(r"^RESULT (.*)$", out, re.M)
    res = dict(kv.split("=") for kv in m[1].split()) if m else {}
    e = re.search(r"^(FAIL .*|%Error.*|.*\$fatal.*|WALL CLOCK TIMEOUT|CLEAR\s+BAD)$", out, re.M)
    if not e and m and not re.search(r"^CLEAR\s+ok$", out, re.M):
        return words, res, "no CLEAR line"
    return words, res, e[1][:160] if e else ("" if m else "no RESULT line")


def flags(w: dict) -> set[int]:
    return {b for b in range(16) if w.get("FLAGS", 0) >> b & 1}


def check_clean(w: dict, res: dict) -> list[str]:
    bad = []
    if flags(w):
        bad.append(f"flags {sorted(flags(w))}")
    if res.get("done") != "1" or res.get("xerrs") != "0":
        bad.append(f"master done={res.get('done')} errors={res.get('xerrs')}")
    if res.get("acc") != "11" or res.get("accbad") != "0":
        bad.append(f"accelerators done={res.get('acc')} bad={res.get('accbad')}")
    want = {"X_W": int(res.get("nwb", -1)), "X_B": int(res.get("nb", -1)), "X_R": int(res.get("nrb", -1)),
            "X_WCHK": w.get("X_W"), "X_RCHK": int(res.get("xrchk", -1)), "N_WDAT": w.get("N_WCMD"),
            "N_RDAT": w.get("N_RCMD")}
    bad += [f"{k} {w.get(k)} != {v}" for k, v in want.items() if w.get(k) != v]
    if w.get("X_R") != w.get("X_RCHK"):
        bad.append(f"X_RCHK {w.get('X_RCHK')} of {w.get('X_R')} R beats")
    if not w.get("N_WCHK") or not w.get("N_RCHK"):
        bad.append(f"channel 0 checked nothing (N_WCHK {w.get('N_WCHK')}, N_RCHK {w.get('N_RCHK')})")
    if w.get("N_WBAD") or w.get("N_RBAD") or w.get("X_WBAD") or w.get("X_RBAD"):
        bad.append("bad beats counted")
    return bad


def main(argv: list[str]) -> int:
    suites = argv or ["clean", "inject"]
    jobs = []                               # (label, exe, log, args, check)
    if "clean" in suites:
        for b, s, args in CLEAN:
            exe = build(b, BUILDS[b])
            jobs.append((f"{b} {s}", exe, OUT / b / f"{s}.log", BASE + args, lambda w, r: check_clean(w, r)))
    if "inject" in suites:
        exe = build("ldc", BUILDS["ldc"])
        for inj, n, must, clear, extra in INJ:
            def chk(w, r, must=must, clear=clear, extra=extra):
                f = flags(w)
                bad = [f"flag {b} not set" for b in sorted(must - f)]
                bad += [f"flag {b} set" for b in sorted(clear & f)]
                if extra is not None and not extra(w):
                    bad.append(f"words: X_STALL {w.get('X_STALL', 0):08x} X_WEXP {w.get('X_WEXP', 0):08x} "
                               f"X_WGOT {w.get('X_WGOT', 0):08x}")
                return bad
            jobs.append((f"ldc inj{inj}@{n}", exe, OUT / "ldc" / f"inj{inj}.log",
                         BASE + [f"+inj={inj}", f"+injn={n}"], chk))
    ok = True
    with ThreadPoolExecutor(PAR) as ex:
        futs = [(lab, chk, ex.submit(run, exe, log, args)) for lab, exe, log, args, chk in jobs]
        for lab, chk, f in futs:
            w, res, why = f.result()
            bad = ([why] if why else []) + (chk(w, res) if w else ["no words"])
            ok &= not bad
            print(f"{lab:22} {'PASS' if not bad else 'FAIL':5} flags={w.get('FLAGS', 0) & 0xFFFF:04x} "
                  f"X_W={w.get('X_W')} X_R={w.get('X_R')} N_WCHK={w.get('N_WCHK')} "
                  f"N_RCHK={w.get('N_RCHK')} errs={res.get('xerrs')}  {'; '.join(bad)}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
