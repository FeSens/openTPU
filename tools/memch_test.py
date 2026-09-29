"""Build and run tb_memch (sim/verilator/tb_memch.sv: otpu_mem_ch in front of LiteDRAM native-port
models) over its scenarios, its throughput runs and its mutation checks. Verilator builds are
heavy: run it on omarchy, one at a time:

    tools/omarchy_test.sh --exec python tools/memch_test.py [suite ...]

suites: quick, func (default), perf, mut (or mut=I,J: those mutations), all. One line per run: the
build (tb_memch's ARD), the scenario, PASS / FAIL / TIMEOUT / ABORT (a model's $fatal or an RTL
$error), and each master's beats per cycle of its own clock. A mutation is a text substitution in
a scratch copy of otpu_mem_ch.sv; it is caught when one of its runs does not PASS. Exit status 1 if
a plain run does not pass or a mutation is not caught.
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
OUT = Path(os.environ.get("MEMCH_OUT", ROOT / "build" / "memch"))
RTL = ROOT / "rtl" / "boards" / "ypcb-00338"
TB = ROOT / "sim" / "verilator"
SRC = [RTL / "otpu_afifo.sv", RTL / "otpu_axi_split2.sv", RTL / "otpu_mem_ch.sv",
       TB / "otpu_ldn_model.sv", TB / "tb_memch.sv"]
JOBS = os.environ.get("MEMCH_JOBS", "4")      # C++ compile jobs per build
PAR = int(os.environ.get("MEMCH_PAR", "3"))   # simulations at once

# builds: tb_memch parameters (cred: a 16-beat accelerator read-data FIFO)
BUILDS = {
    "ldn": dict(),
    "cred": dict(ARD=16),
}

SCEN = {
    "default": [],
    "seed2": ["+seed=2"], "seed3": ["+seed=3"], "seed4": ["+seed=4"],
    "xreset": ["+xreset=3000", "+seed=5"],
    "areset": ["+areset=3000", "+seed=6"],
    "resets": ["+areset=2500", "+xreset=4000", "+seed=7"],
    "mstall70": ["+mstall=70"],
    "nogaps": ["+mstall=0", "+gapw=0", "+gapc=0"],
    "partial": ["+ppct=100", "+xfull=0"],
    "rawpart": ["+ppct=60", "+raw=60", "+wpct=60", "+xfull=20"],
    "ctlstall": ["+axi_stall=40", "+axi_lat=60", "+ldn_busy=20"],
    "fastcore": ["+cp=300", "+up=500", "+xp=450"],
    "slowcore": ["+cp=700", "+up=300", "+xp=600"],
    "shared": ["+psh=40", "+psp=40"],
    "long": ["+ntx=3000"],
    "seqrd": ["+seq=1", "+wpct=0"],
    "seqwr": ["+seq=1", "+wpct=100"],
    "seqmix": ["+seq=1", "+wpct=50"],
    # the accelerator's read-data FIFO filled faster than the core drains it (build cred: ARD 16)
    "credstress": ["+cp=900", "+wpct=10", "+mstall=0", "+rbuf=256", "+gapc=0", "+ppct=0", "+ntx=800"],
}

SCEN["long"] = ["+ntx=30000"]
SCEN["lateresets"] = ["+areset=150000", "+xreset=100000", "+seed=8"]
# resets shorter than the read latency: reads still in flight when the master comes back
SCEN["xresetlat"] = ["+xreset=3000", "+axi_lat=150", "+seed=9"]
SCEN["aresetlat"] = ["+areset=3000", "+axi_lat=150", "+seed=9"]
# resets again and again (every few thousand cycles), with and without long read latency
SCEN["xresetrep"] = ["+xreset=3000", "+xrep=7919", "+axi_lat=150", "+seed=11"]
SCEN["aresetrep"] = ["+areset=3000", "+arep=7919", "+axi_lat=150", "+seed=12"]
SCEN["resetsrep"] = ["+areset=2000", "+arep=6007", "+xreset=3000", "+xrep=4001", "+seed=13"]
# short resets while the controller stalls: the master's commands still in the output command
# register after its hold would otherwise be counted (n_wdone, B) after its reset
SCEN["aresetshort"] = ["+areset=2000", "+arep=3001", "+arlen=2", "+axi_stall=85", "+ldn_busy=30",
                       "+wpct=80", "+seed=14"]
SCEN["xresetshort"] = ["+xreset=2000", "+xrep=3001", "+xrlen=2", "+axi_stall=85", "+ldn_busy=30",
                       "+wpct=80", "+seed=15"]
# the same with many shared operations: after a count too many, XDMA's B for a one-beat burst
# comes before its beat is taken, and the accelerator, told by it, reads the old data (the
# controller takes a command in 3% of its cycles, so the output register outlasts the hold,
# and XDMA writes only, so the hold does not wait for its reads)
SCEN["xresetshortsh"] = ["+psh=60", "+axi_stall=97", "+wpct=100", "+ntx=1000"] + SCEN["xresetshort"]
for i in range(10, 30):
    SCEN[f"s{i}"] = [f"+seed={i}", f"+psh={5 + i % 4 * 15}", f"+ppct={i % 5 * 20}", f"+wpct={30 + i % 3 * 20}"]
# functional runs: 3000 runs or bursts per master unless the scenario says otherwise (the first
# plusarg of a name wins)
FBASE = ["+ntx=3000", "+tmax=100000000"]
FUNC = [("ldn", s) for s in ["default", "seed2", "seed3", "seed4", "xreset", "areset", "resets",
                             "lateresets", "xresetlat", "aresetlat", "xresetrep", "aresetrep", "resetsrep", "aresetshort", "xresetshort", "xresetshortsh", "mstall70", "nogaps", "partial", "rawpart", "ctlstall", "fastcore",
                             "slowcore", "shared", "long", "seqrd", "seqwr", "seqmix"]]
FUNC += [("ldn", f"s{i}") for i in range(10, 30)]
FUNC += [("cred", s) for s in ["default", "credstress"]]

# throughput: sequential 32-beat runs (64-beat bursts for XDMA), one kind of master at a time,
# whole beats unless the run says otherwise (the first plusarg of a name wins)
PERF_BASE = ["+seq=1", "+mstall=0", "+gapw=0", "+ntx=200", "+ldn_busy=0", "+ppct=0", "+xfull=100"]
PERF = []
for b in ["ldn"]:
    PERF += [(b, "acc seq rd", ["+wpct=0", "+xdma_ntx=0"] + PERF_BASE),
             (b, "acc seq wr", ["+wpct=100", "+xdma_ntx=0"] + PERF_BASE),
             (b, "acc seq rd lat40", ["+wpct=0", "+xdma_ntx=0", "+axi_lat=40"] + PERF_BASE),
             (b, "acc seq wr 1 in 32 partial", ["+wpct=100", "+ppct=3", "+xdma_ntx=0"] + PERF_BASE),
             (b, "acc seq wr all partial", ["+wpct=100", "+ppct=100", "+xdma_ntx=0"] + PERF_BASE),
             (b, "xdma seq rd", ["+wpct=0", "+acc0_ntx=0", "+acc1_ntx=0"] + PERF_BASE),
             (b, "xdma seq wr", ["+wpct=100", "+acc0_ntx=0", "+acc1_ntx=0"] + PERF_BASE)]
PERF += [("ldn", "acc seq rd busy2", ["+wpct=0", "+xdma_ntx=0", "+ldn_busy=2"] + PERF_BASE),
         ("ldn", "acc seq wr busy2", ["+wpct=100", "+xdma_ntx=0", "+ldn_busy=2"] + PERF_BASE),
         ("ldn", "acc seq rd stall10", ["+wpct=0", "+xdma_ntx=0", "+axi_stall=10"] + PERF_BASE),
         ("ldn", "acc seq wr stall10", ["+wpct=100", "+xdma_ntx=0", "+axi_stall=10"] + PERF_BASE)]

# mutations: (name, build, substitutions, scenarios[, "missed": a known blind spot])
MUT = [
    ("rmw merge: old bytes in the written lanes", "ldn",
     [("(rm_ret && !wd[512 + k])", "(rm_ret && wd[512 + k])")], ["partial", "rawpart", "default"]),
    ("rmw merge: no merge (lanes not written left as sent)", "ldn",
     [("(rm_ret && !wd[512 + k])", "1'b0")], ["partial", "default"]),
    ("credits: accelerator reads without credits", "cred",
     [("a_rok <= (a_out + ar_used) <= AOW'(ARD - 2);", "a_rok <= 1'b1;")], ["credstress", "default"]),
    ("credits: XDMA reads without credits", "ldn",
     [("x_rok <= (x_out + xr_used) <= XOW'(XRD - 2);", "x_rok <= 1'b1;")], ["mstall70", "default"]),
    ("credits: one beat less margin", "ldn",
     [("x_rok <= (x_out + xr_used) <= XOW'(XRD - 2);", "x_rok <= (x_out + xr_used) <= XOW'(XRD - 1);")],
     ["mstall70", "default"]),
    ("n_wdone: gray code not decoded", "ldn",
     [("n_wdone <= g2b(a_wacc_s2);", "n_wdone <= a_wacc_s2;")], ["default"]),
    ("n_wdone: reads counted too", "ldn",
     [("else if (opop && oc0[26] && !oc0[25]) a_wacc", "else if (opop && !oc0[25]) a_wacc")], ["default"]),
    ("n_wdone: counted on the core side once command and data are in (before the controller)", "ldn",
     [("  assign ar_rr    = 1'b1;\n", "  assign ar_rr    = 1'b1;\n  logic [15:0] mu_c, mu_d;\n"),
      ("if (a_crst) begin a_wacc_s1 <= '0; a_wacc_s2 <= '0; n_wdone <= '0; end",
       "if (a_crst) begin a_wacc_s1 <= '0; a_wacc_s2 <= '0; n_wdone <= '0; mu_c <= '0; mu_d <= '0; end"),
      ("n_wdone <= g2b(a_wacc_s2); end",
       "mu_c <= mu_c + 16'(aq_wv && n_cwe); mu_d <= mu_d + 16'(ad_wv); n_wdone <= (mu_c < mu_d) ? mu_c : mu_d; end")],
     ["shared", "default"]),
    # expected to be missed: the core issues reads no faster than it drains them, so a 64-beat FIFO
    # does not fill without credits in any traffic here (ARD 16 does, above)
    ("credits: accelerator reads without credits, ARD 64", "ldn",
     [("a_rok <= (a_out + ar_used) <= AOW'(ARD - 2);", "a_rok <= 1'b1;")], ["slowcore", "credstress", "default"],
     "missed"),
    ("credits: accelerator reads without credits, the FIFO-full assertion removed", "cred",
     [("a_rok <= (a_out + ar_used) <= AOW'(ARD - 2);", "a_rok <= 1'b1;"),
      ('if (ar_wv && !ar_wr) $error("otpu_mem_ch: accelerator read-data FIFO full");', "")], ["credstress"]),
    ("credits: XDMA reads without credits, the FIFO-full assertion removed", "ldn",
     [("x_rok <= (x_out + xr_used) <= XOW'(XRD - 2);", "x_rok <= 1'b1;"),
      ('if (xr_wv && !xr_wr) $error("otpu_mem_ch: XDMA read-data FIFO full");', "")], ["mstall70"]),
    ("reset hold: XDMA's does not wait for its reads in flight", "ldn",
     [("(x_hcnt != 0 || x_out != 0 ||", "(x_hcnt != 0 ||")],
     ["xresetrep", "resetsrep", "xresetlat", "xreset"]),
    ("reset hold: the accelerator's does not wait for its reads in flight", "ldn",
     [("(a_hcnt != 0 || a_out != 0 ||", "(a_hcnt != 0 ||")],
     ["aresetrep", "resetsrep", "aresetlat", "areset"]),
    ("reset hold: XDMA's does not wait for its commands in the output register", "ldn",
     [("(rm_busy && rm_x) || x_oc)", "(rm_busy && rm_x))")], ["xresetshortsh", "xresetshort"]),
    ("reset hold: the accelerator's does not wait for its commands in the output register", "ldn",
     [("(rm_busy && !rm_x) || a_oc)", "(rm_busy && !rm_x))")], ["aresetshort", "resetsrep"]),
    # without the hold-over assertion: the accelerator's extra count breaks n_wdone's exact check;
    # XDMA's makes a B one beat early, which this bench cannot see (its W beats have all been given
    # by then, and no other master reads XDMA's window)
    ("reset hold: the accelerator's does not wait for its commands, the hold-over assertion removed",
     "ldn", [("(rm_busy && !rm_x) || a_oc)", "(rm_busy && !rm_x))"),
             ("if (a_hold_q && !a_hold && a_oc) $error", "if (1'b0) $error")], ["aresetshort"]),
    ("reset hold: XDMA's does not wait for its commands, the hold-over assertion removed", "ldn",
     [("(rm_busy && rm_x) || x_oc)", "(rm_busy && rm_x))"),
      ("if (x_hold_q && !x_hold && x_oc) $error", "if (1'b0) $error")], ["xresetshortsh"], "missed"),
    ("reset: XDMA's request not kept up until its hold is seen (a short reset)", "ldn",
     [("x_req <= xrst || (x_req && !x_hs2);", "x_req <= xrst;")], ["xresetshort"]),
    ("reset: the accelerator's request not kept up until its hold is seen (a short reset)", "ldn",
     [("a_req <= rst || (a_req && !a_hs2);", "a_req <= rst;")], ["aresetshort"]),
]


def build(name: str, params: dict, mut: list | None = None) -> Path:
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    src = list(SRC)
    if mut:
        text = (RTL / "otpu_mem_ch.sv").read_text()
        for old, new in mut:
            assert text.count(old) == 1, f"mutation {name}: {old!r} found {text.count(old)} times"
            text = text.replace(old, new)
        (d / "otpu_mem_ch.sv").write_text(text)
        src[2] = d / "otpu_mem_ch.sv"
    key = hashlib.sha256(repr(params).encode() + b"".join(p.read_bytes() for p in src)).hexdigest()
    exe = d / "obj" / "Vtb_memch"
    if exe.exists() and (d / "key").exists() and (d / "key").read_text() == key:
        return exe
    cmd = (["verilator", "--binary", "-j", JOBS, "--top-module", "tb_memch", "-Wno-fatal", "-Wno-lint",
            "-Wno-style", "-O3", "--x-assign", "0", "--x-initial", "0", "-Mdir", str(d / "obj")]
           + [f"-G{k}={v}" for k, v in params.items()] + [str(p) for p in src])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(f"BUILD FAILED {name}\n{r.stdout[-4000:]}\n{r.stderr[-4000:]}", flush=True)
        sys.exit(2)
    (d / "key").write_text(key)
    return exe


def run(exe: Path, logname: Path, args: list[str]) -> tuple[str, dict, str]:
    try:
        r = subprocess.run([str(exe)] + args, capture_output=True, text=True, timeout=3600)
        out = r.stdout + r.stderr
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode(errors="replace") + "\nWALL CLOCK TIMEOUT"
    logname.write_text(out)
    m = re.search(r"^(PASS|FAIL|TIMEOUT)", out, re.M)
    res = m.group(1) if m else "ABORT"
    tp = {mm.group(1): mm.group(2) for mm in
          re.finditer(r"^(acc0|acc1|xdma): .*?([-0-9.naif]+) beats per cycle", out, re.M)}
    why = ""
    if res != "PASS":
        e = re.search(r"^(ERROR .*|%Error.*|.*otpu_\w+(?: ch\d)?: .*)$", out, re.M)
        why = e.group(1)[:160] if e else out.strip().splitlines()[-1][:160] if out.strip() else ""
    return res, tp, why


def fmt(tp: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in tp.items())


def main(argv: list[str]) -> int:
    suites = argv or ["func"]
    if "all" in suites:
        suites = ["func", "perf", "mut"]
    ok = True
    jobs = []                                   # (label, exe, log, args)
    if "quick" in suites:
        exe = build("ldn", BUILDS["ldn"])
        jobs.append(("ldn default", exe, OUT / "ldn" / "default.log", SCEN["default"]))
    if "func" in suites:
        for b, s in FUNC:
            exe = build(b, BUILDS[b])
            jobs.append((f"{b} {s}", exe, OUT / b / f"{s}.log", SCEN[s] + FBASE))
    if "perf" in suites:
        for b, s, args in PERF:
            exe = build(b, BUILDS[b])
            jobs.append((f"{b} perf {s}", exe, OUT / b / ("perf-" + re.sub(r"\W+", "_", s) + ".log"), args))
    with ThreadPoolExecutor(PAR) as ex:
        futs = [(lab, ex.submit(run, exe, log, args)) for lab, exe, log, args in jobs]
        for lab, f in futs:
            res, tp, why = f.result()
            ok &= res == "PASS"
            print(f"{lab:40} {res:8} {fmt(tp)}  {why}", flush=True)
    sel = [a for a in suites if a.startswith("mut=")]      # mut=8,9: only those mutations
    if "mut" in suites or sel:
        pick = {int(x) for x in sel[0][4:].split(",")} if sel else set(range(len(MUT)))
        for i, (name, b, subs, scens, *exp) in enumerate(MUT):
            if i not in pick:
                continue
            exe = build(f"mut{i}", BUILDS[b], subs)
            with ThreadPoolExecutor(PAR) as ex:
                rs = list(ex.map(lambda s: (s, run(exe, OUT / f"mut{i}" / f"{s}.log", SCEN[s] + FBASE)), scens))
            caught = any(r[0] != "PASS" for _, r in rs)
            ok &= caught or exp == ["missed"]
            tag = "CAUGHT" if caught else "MISSED (expected)" if exp == ["missed"] else "MISSED"
            print(f"MUTATION {tag}: {name} [{b}]", flush=True)
            for s, (res, tp, why) in rs:
                print(f"    {s:14} {res:8} {why}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
