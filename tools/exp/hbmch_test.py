"""Build and run tb_hbmch (sim/verilator/tb_hbmch.sv: the WG targets' channel bridge otpu_hbm_ch in
front of an HBM AXI3 port model) over its scenarios, its throughput runs and its mutation checks.
The bridge at its port (N 0) and across a pipeline of N + 1 stages each way (N 8: a slice two SLRs
from its ports, W at half width; WH 0: at full width), with posted reads (source 3, p_*), and with
the core side's data in half beats (RHB / WHB: the WG targets' R1 / R2, otpu_native_dram's).
Verilator builds are heavy: run one at a time:

    python3 tools/exp/hbmch_test.py [suite ...]

suites: quick, func (default), perf, mut (or mut=I,J: those mutations), all. One line per run: the
build, the scenario, PASS / FAIL / TIMEOUT / ABORT (a model's $fatal or an RTL $error), the master's
beats per core cycle and the model's R beats per aclk cycle (perf: the share of the channel's
14.4 GB/s). A mutation is a text substitution in a scratch copy of otpu_hbm_ch.sv; it is caught
when one of its runs does not PASS. Exit status 1 if a plain run does not pass, a perf run misses
its bar or a mutation is not caught. HBMCH_XSW=N: every build at XSW N (a posted read's lane RAM
entry's bits, the board's log2 FIFO_DEPTH; default the bridge's 9).
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("HBMCH_OUT", ROOT / "build" / "hbmch"))
TB = ROOT / "sim" / "verilator"
SRC = [ROOT / "rtl" / "shell" / "otpu_afifo.sv", ROOT / "rtl" / "shell" / "otpu_mem_ch.sv",
       ROOT / "rtl" / "mem" / "otpu_hbm_ch.sv", TB / "otpu_hbm_model.sv", TB / "tb_hbmch.sv"]
JOBS = os.environ.get("HBMCH_JOBS", "4")      # C++ compile jobs per build
PAR = int(os.environ.get("HBMCH_PAR", "4"))   # simulations at once
XSW = {"XSW": int(os.environ["HBMCH_XSW"])} if os.environ.get("HBMCH_XSW") else {}

# builds: tb_hbmch parameters
BUILDS = {
    "cb128": dict(),
    "ard32": dict(ARD=32),                     # fewer read credits
    "cb64": dict(CB=64),                       # 2 AXI beats a native beat, bursts of 8
    "nbk4": dict(NBK=4, WOT=4),                # few buckets (collisions), few writes in flight
    "wdd16": dict(WDD=16),                     # the write-data FIFO at 16 (before 2026-10-20)
    "post": dict(WPOST=1),                     # write data posted on n_wfree (nch's g2 adapter)
    "n8": dict(N=8, WH=1),                     # the halves 9 stages apart, W at half width
    "n8f": dict(N=8),                          # the same, W at full width
    "n8cb64": dict(CB=64, N=8, WH=1),
    "n8post": dict(N=8, WH=1, WPOST=1),
    "n1p2": dict(N=1, PST=2),                  # p_* through two register stages
    "n8fw": dict(N=8, AWD=64),                 # W at full width, the W FIFO 64 beats
    "rhb": dict(RHB=1),                        # read data in half beats
    "whb": dict(WPOST=1, WHB=1),               # posted write data in half beats
    "hb": dict(RHB=1, WPOST=1, WHB=1),         # both (the WG targets' adapter: R1 + R2)
    "hba32": dict(RHB=1, WPOST=1, WHB=1, ARD=32),     # the builds' read credits with XPOST (ARD 32)
    "hbn3": dict(RHB=1, WPOST=1, WHB=1, N=3, WH=1, ARD=32),   # c7all's split bridges (N 3 / 5)
    "hbn5": dict(RHB=1, WPOST=1, WHB=1, N=5, WH=1, ARD=32),
    "hbn8": dict(RHB=1, WPOST=1, WHB=1, N=8, WH=1),
    # the BCU1525's two slices (bcu1525-s2: RHB, WPOST, ARD 32, XSW 11, N 5 across an SLR or 2 in
    # the slice's own), the core half's aclk fan-outs in copies (CPY) and without
    "bcn5": dict(RHB=1, WPOST=1, N=5, WH=1, ARD=32, XSW=11),
    "bcn5c": dict(RHB=1, WPOST=1, N=5, WH=1, ARD=32, XSW=11, CPY=1),
    "bcn2c": dict(RHB=1, WPOST=1, N=2, WH=1, ARD=32, XSW=11, CPY=1),
}

SCEN = {
    "default": [],
    "seed2": ["+seed=2"], "seed3": ["+seed=3"], "seed4": ["+seed=4"],
    # a small window: reads and writes on the same blocks in flight together, both ways
    "small": ["+nb=16", "+raw=40", "+war=40", "+seed=5"],
    "hazard": ["+nb=8", "+raw=50", "+war=50", "+hbm_jit=100", "+gapc=0", "+mstall=0", "+seed=6"],
    "hazardlat": ["+nb=12", "+raw=50", "+war=50", "+hbm_lat=300", "+hbm_wlat=300", "+hbm_jit=200",
                  "+seed=7"],
    "partial": ["+ppct=100"],
    "nogaps": ["+mstall=0", "+gapw=0", "+gapc=0"],
    "mstall70": ["+mstall=70"],
    "stall": ["+hbm_ast=40", "+hbm_wst=40"],
    "lowbw": ["+hbm_rbw=40", "+mstall=0"],
    # R beats with random gaps inside a burst (RHB: a native beat's AXI beats far apart in u_rd's lanes)
    "rgap": ["+hbm_rgap=70", "+mstall=0", "+seed=16"],
    "rgapmix": ["+hbm_rgap=50", "+ppost=30", "+nb=16", "+raw=40", "+war=40", "+seed=17"],
    "jitter": ["+hbm_jit=200"],
    "fastcore": ["+cp=2500"],                  # 200 MHz
    "slowcore": ["+cp=5000", "+mstall=0", "+gapc=0", "+rbuf=256"],  # 100 MHz
    "mw0": ["+mw=0"], "mw40": ["+mw=40"],
    "areset": ["+areset=3000", "+seed=8"],
    "aresetrep": ["+areset=2000", "+arep=5003", "+seed=9"],
    "aresetlat": ["+areset=3000", "+arep=7919", "+hbm_lat=300", "+hbm_wlat=300", "+hbm_jit=100",
                  "+seed=10"],
    # resets shorter than the round trip through the synchronizers, while the HBM stalls
    "aresetshort": ["+areset=2000", "+arep=3001", "+arlen=2", "+hbm_ast=60", "+hbm_wst=60",
                    "+wpct=80", "+seed=11"],
    "skew": ["+cdc_skew=60", "+wpct=80", "+gapw=0", "+mstall=20", "+seed=12"],
    "seqrd": ["+seq=1", "+wpct=0"],
    "seqwr": ["+seq=1", "+wpct=100"],
    "seqmix": ["+seq=1", "+wpct=50"],
    "seqrand": ["+seq=2", "+wpct=50", "+mw=20"],
    "seqrand2": ["+seq=2", "+wpct=30", "+ppct=30", "+mw=40", "+seed=13"],
    "credstress": ["+cp=5000", "+seq=1", "+wpct=0", "+mstall=0", "+gapc=0", "+rbuf=512", "+ntx=300"],
    # write data well ahead of its commands (the adapter's takes before the commands issue): u_wd
    # fills before the commands come
    "wlead": ["+gapw=0", "+gapc=70", "+wpct=80", "+seed=14"],
    "wleadp": ["+gapw=0", "+gapc=70", "+wpct=80", "+ppct=60", "+raw=30", "+war=30", "+nb=64", "+seed=15"],
    # posted reads (source 3): mixed with the others, all of them, on the same blocks as writes,
    # on few lane RAM entries, sequential, under core resets (the core's reset 8 cycles: the
    # request's synchronizer, p_*'s stages and the adapter's two flip-flops take ~5)
    "post30": ["+ppost=30", "+seed=40"],
    "post100": ["+ppost=100", "+wpct=30", "+seed=41"],
    "postmix": ["+ppost=50", "+nb=16", "+raw=40", "+war=40", "+seed=42"],
    "postslot": ["+ppost=60", "+nslot=4", "+seed=43"],
    # every lane RAM entry (2^XSW: at HBMCH_XSW 11 the slot's top bits, past the 512 of XSW 9)
    "postwide": ["+ppost=60", "+nslot=2048", "+seed=45"],
    "postseq": ["+seq=1", "+wpct=0", "+ppost=50"],
    "postseq2": ["+seq=2", "+wpct=30", "+ppost=50", "+mw=20", "+seed=44"],
    "postreset": ["+ppost=60", "+areset=3000", "+arep=5003", "+arlen=8", "+hbm_lat=300", "+hbm_wlat=300",
                  "+seed=45"],
    "postresetl": ["+ppost=80", "+wpct=20", "+areset=2000", "+arep=3001", "+arlen=8", "+hbm_lat=400",
                   "+hbm_jit=100", "+seed=46"],
    "postcred": ["+cp=5000", "+seq=1", "+wpct=0", "+ppost=50", "+mstall=0", "+gapc=0", "+rbuf=512",
                 "+ntx=300"],
    # the HBM's reset alone (arst, 2 aclk cycles or 30) with traffic in flight: the bridge holds
    # both halves while its pipeline empties both ways
    "hreset": ["+hreset=9000", "+hrep=15013", "+seed=50"],
    "hresetp": ["+hreset=9000", "+hrep=12007", "+ppost=50", "+raw=30", "+seed=51"],
    "hresetl": ["+hreset=7000", "+hrep=19001", "+hrlen=30", "+hbm_lat=300", "+hbm_wlat=300", "+seed=52"],
    # a core reset in the cycle a lone burst issues (the drain's test a cycle before the burst is
    # in rbq / wbq): sparse one-beat bursts (a beat every ~33 core cycles, rbq empty between them
    # at 60 aclk cycles of latency), a short reset every 257 core cycles, ~6,000 of them
    "drainedge": ["+gapc=97", "+hbm_lat=60", "+hbm_wlat=60", "+areset=2000", "+arep=257", "+arlen=2",
                  "+seed=60"],
    "drainedge2": ["+gapc=97", "+hbm_lat=60", "+hbm_wlat=60", "+areset=2000", "+arep=263", "+arlen=2",
                   "+wpct=80", "+seed=61"],
}
for i in range(20, 36):
    SCEN[f"s{i}"] = [f"+seed={i}", f"+nb={[16, 64, 256, 1024][i % 4]}", f"+ppct={i % 5 * 20}",
                     f"+wpct={30 + i % 3 * 20}", f"+raw={i % 4 * 15}", f"+war={i % 3 * 15}",
                     f"+mw={[0, 4, 8, 20][i % 4]}", f"+ppost={[0, 30, 70][i % 3]}"]
FBASE = ["+ntx=3000", "+tmax=20000000"]
FUNC = [("cb128", s) for s in SCEN]
FUNC += [("ard32", s) for s in ["default", "seed2", "small", "hazard", "slowcore", "credstress",
                                "areset", "seqrd", "seqmix", "post30", "postcred"]]
FUNC += [("cb64", s) for s in ["default", "seed2", "small", "hazard", "partial", "slowcore",
                               "areset", "aresetshort", "skew", "seqrd", "seqwr", "s21", "s22"]]
FUNC += [("nbk4", s) for s in ["default", "small", "hazard", "hazardlat", "stall", "areset", "s23"]]
FUNC += [("wdd16", s) for s in ["default", "wlead", "seqwr", "areset"]]
FUNC += [("post", s) for s in ["default", "seed2", "small", "hazard", "partial", "nogaps", "stall",
                               "slowcore", "areset", "aresetrep", "aresetshort", "skew", "seqwr",
                               "seqmix", "wlead", "wleadp", "s20", "s21", "s22", "s23"]]
FUNC += [("n8", s) for s in ["default", "seed2", "small", "hazard", "hazardlat", "partial", "stall",
                             "lowbw", "jitter", "fastcore", "slowcore", "mw0", "areset", "aresetrep",
                             "aresetlat", "aresetshort", "skew", "seqrd", "seqwr", "seqmix", "seqrand",
                             "credstress", "wlead", "post30", "post100", "postmix", "postslot", "postwide",
                             "postseq2", "postreset", "postresetl", "postcred", "hreset", "hresetp",
                             "hresetl", "drainedge", "drainedge2"]
         + [f"s{i}" for i in range(20, 28)]]
FUNC += [("n8f", s) for s in ["default", "small", "partial", "seqwr", "seqmix", "areset", "post30",
                              "postreset"]]
FUNC += [("n8cb64", s) for s in ["default", "small", "hazard", "partial", "areset", "aresetshort",
                                 "seqwr", "post30", "postreset", "hreset"]]
FUNC += [("n8post", s) for s in ["default", "wlead", "wleadp", "seqwr", "areset", "aresetshort",
                                 "post30", "postreset", "postwide", "hreset", "hresetp", "hresetl"]]
FUNC += [("post", s) for s in ["hreset", "hresetp", "hresetl"]]
FUNC += [("n1p2", s) for s in ["default", "post30", "postmix", "postreset", "postresetl"]]
HBF = ["default", "seed2", "small", "hazard", "hazardlat", "partial", "nogaps", "stall", "lowbw", "rgap", "rgapmix",
       "fastcore", "slowcore", "mw0", "areset", "aresetrep", "aresetlat", "aresetshort", "skew", "seqrd",
       "seqwr", "seqmix", "seqrand", "seqrand2", "credstress", "wlead", "wleadp", "post30", "postmix",
       "postreset", "postcred", "hreset", "hresetp", "hresetl", "drainedge", "drainedge2"]
FUNC += [("cb128", s) for s in ["rgap", "rgapmix"]]
FUNC += [("rhb", s) for s in ["default", "seed2", "small", "hazard", "slowcore", "rgap", "rgapmix", "areset", "aresetshort",
                              "seqrd", "seqmix", "credstress", "post30", "hreset", "drainedge"]]
FUNC += [("whb", s) for s in ["default", "seed2", "small", "partial", "nogaps", "stall", "areset",
                              "aresetshort", "seqwr", "wlead", "wleadp", "hreset", "drainedge2"]]
FUNC += [("hb", s) for s in HBF + [f"s{i}" for i in range(20, 28)]]
FUNC += [("hbn5", s) for s in HBF + [f"s{i}" for i in range(28, 32)]]
FUNC += [("hba32", s) for s in ["default", "seed2", "small", "hazard", "slowcore", "lowbw", "areset",
                                "aresetshort", "seqrd", "seqmix", "credstress", "post30", "postcred",
                                "hreset", "drainedge"]]
FUNC += [("hbn3", s) for s in ["default", "small", "partial", "areset", "aresetshort", "seqmix",
                               "post30", "postreset", "hreset", "drainedge"]]
FUNC += [("hbn8", s) for s in ["default", "small", "partial", "areset", "aresetshort", "seqmix",
                               "post30", "postreset", "hreset", "hresetl", "drainedge2"]]
BCF = ["default", "seed2", "small", "hazard", "partial", "rgap", "rgapmix", "slowcore", "areset",
       "aresetshort", "skew", "seqrd", "seqmix", "credstress", "wlead", "post30", "postwide", "postreset",
       "postcred", "hreset", "hresetp", "drainedge", "drainedge2"]
FUNC += [(b, s) for b in ("bcn5c", "bcn2c") for s in BCF]
FUNC += [("bcn5", s) for s in ["default", "areset", "hreset", "postreset"]]

# throughput: sequential 32-beat runs, the master keeping 128 reads in flight (otpu_native_dram RD)
# and never stalling; the bar: the model's R channel busy >= 95% of its cycles (14.4 GB/s at
# 450 MHz) where the core clock can carry it (133.33 MHz: 17 GB/s of 128-byte beats)
PERF_BASE = ["+seq=1", "+mstall=0", "+gapw=0", "+gapc=0", "+ntx=600", "+rbuf=128"]
PERF = [("cb128", "seq rd 133.33 MHz", ["+wpct=0"] + PERF_BASE, 0.95),
        ("cb128", "seq rd 133.33 MHz, latency 400 ns", ["+wpct=0", "+hbm_lat=180"] + PERF_BASE, 0.95),
        ("cb128", "seq rd 133.33 MHz, latency jitter", ["+wpct=0", "+hbm_jit=60"] + PERF_BASE, 0.95),
        ("cb128", "seq rd 133.33 MHz, mwait 0", ["+wpct=0", "+mw=0"] + PERF_BASE, 0.95),
        ("cb128", "seq rd 200 MHz", ["+wpct=0", "+cp=2500"] + PERF_BASE, 0.95),
        ("cb128", "seq rd 100 MHz (core-bound: 12.8 of 14.4 GB/s)", ["+wpct=0", "+cp=5000"] + PERF_BASE, 0.85),
        ("cb128", "seq rd 133.33 MHz, HBM at 92%", ["+wpct=0", "+hbm_rbw=92"] + PERF_BASE, 0.90),
        ("cb128", "seq wr 133.33 MHz", ["+wpct=100"] + PERF_BASE, None),
        ("cb128", "seq rd/wr 133.33 MHz", ["+wpct=50"] + PERF_BASE, None),
        ("ard32", "seq rd 133.33 MHz, ARD 32", ["+wpct=0"] + PERF_BASE, None),
        ("cb64", "seq rd 133.33 MHz, CB 64", ["+wpct=0"] + PERF_BASE, None),
        ("cb128", "seq posted rd 133.33 MHz", ["+wpct=0", "+ppost=100", "+nslot=512"] + PERF_BASE, 0.95),
        ("n8", "seq rd 133.33 MHz, N 8", ["+wpct=0"] + PERF_BASE, 0.95),
        ("n8", "seq rd 133.33 MHz, N 8, latency 400 ns", ["+wpct=0", "+hbm_lat=180"] + PERF_BASE, 0.95),
        ("n8", "seq posted rd 133.33 MHz, N 8", ["+wpct=0", "+ppost=100", "+nslot=512"] + PERF_BASE, 0.95),
        ("n8", "seq wr 133.33 MHz, N 8 (W at half width)", ["+wpct=100"] + PERF_BASE, None),
        ("n8f", "seq wr 133.33 MHz, N 8, W at full width", ["+wpct=100"] + PERF_BASE, None),
        ("n8fw", "seq wr 133.33 MHz, N 8, W at full width, AWD 64", ["+wpct=100"] + PERF_BASE, None),
        ("hb", "seq rd 133.33 MHz, RHB (core-bound: 8.5 of 14.4 GB/s)", ["+wpct=0"] + PERF_BASE, 0.55),
        ("hb", "seq wr 133.33 MHz, WHB", ["+wpct=100"] + PERF_BASE, None),
        ("hb", "seq posted rd 133.33 MHz, RHB", ["+wpct=0", "+ppost=100", "+nslot=512"] + PERF_BASE, 0.95),
        ("hba32", "seq rd 133.33 MHz, RHB, ARD 32", ["+wpct=0"] + PERF_BASE, 0.55),
        ("hbn5", "seq rd 133.33 MHz, RHB, N 5, ARD 32", ["+wpct=0"] + PERF_BASE, 0.55),
        ("hbn5", "seq wr 133.33 MHz, WHB, N 5, ARD 32", ["+wpct=100"] + PERF_BASE, None),
        ("hbn5", "seq posted rd 133.33 MHz, RHB, N 5, ARD 32", ["+wpct=0", "+ppost=100", "+nslot=512"] + PERF_BASE,
         0.95)]

# mutations: (name, build, substitutions, scenarios[, "missed": a known blind spot])
MUT = [
    ("hazard: a read does not wait for a write in flight to its block", "cb128",
     [("assign i_haz  = i_we ? rnz[i_bkh] : wnz[i_bkh];", "assign i_haz  = i_we ? rnz[i_bkh] : 1'b0;")],
     ["hazard", "small", "hazardlat"]),
    ("hazard: a write does not wait for a read in flight to its block", "cb128",
     [("assign i_haz  = i_we ? rnz[i_bkh] : wnz[i_bkh];", "assign i_haz  = i_we ? 1'b0 : wnz[i_bkh];")],
     ["hazard", "small", "hazardlat"]),
    ("hazard: a bucket's read count not accumulated (only the last read burst counted)", "nbk4",
     [("rn = rcnt[b] + RBW'(p_ar && p_oh[b])", "rn = RBW'(p_ar && p_oh[b])")],
     ["hazard", "small", "hazardlat"]),
    ("hazard: a posted read does not wait for a write in flight to its block", "cb128",
     [("assign i_haz  = i_we ? rnz[i_bkh] : wnz[i_bkh];", "assign i_haz  = i_we ? rnz[i_bkh] : wnz[i_bkh] && !i_post;")],
     ["postmix", "post100"]),
    ("credits: reads without credits", "cb128",
     [("rok <= rcr_n <= RBW'(ARD - MAXB);", "rok <= 1'b1;")],
     ["credstress", "slowcore"]),
    ("credits: room for one beat, not a burst", "cb128",
     [("rok <= rcr_n <= RBW'(ARD - MAXB);", "rok <= rcr_n <= RBW'(ARD - 1);")], ["credstress", "slowcore"]),
    ("credits: a free counted twice", "cb128",
     [("- RBW'(rf);", "- RBW'(rf) - RBW'(rf);")], ["credstress", "slowcore", "default"]),
    ("credits: a free's pulse repeated (fr_v not taken from the pending frees)", "n8",
     [("assign fr_n = fr_p + (rd_f - rd_fs) - RAW'(fr_v);", "assign fr_n = fr_p + (rd_f - rd_fs);")],
     ["credstress", "slowcore", "default"]),
    ("credits: sources 0-2 take none, posted reads take them", "cb128",
     [("assign rcr_n = rcr + ((p_ar && !p_post) ?", "assign rcr_n = rcr + ((p_ar && p_post) ?")],
     ["credstress", "postcred", "slowcore"]),
    ("credits: the command FIFO's credit back at its push, not its pop", "n8",
     [("always_ff @(posedge aclk) cc <= h_pop && !a_hold;", "always_ff @(posedge aclk) cc <= c_v && !a_hold;")],
     ["hazard", "small", "default"]),
    ("credits: the W FIFO's credit back at its push, not its pop", "n8",
     [("always_ff @(posedge aclk) wc <= wf_rr && wf_rv && !a_hold;", "always_ff @(posedge aclk) wc <= wf_wv && !a_hold;")],
     ["seqwr", "stall", "default"]),
    ("n_wdone: counted at the AW, before the B", "cb128",
     [("((b_v && wbq_rv && !a_stop) ? DPW'(wbq_rd[NW-1:0]) : '0)", "(p_aw ? DPW'(p_n) : '0)")],
     ["default", "seqwr", "nogaps"]),
    ("n_wdone: gray code not decoded", "cb128",
     [("n_wdone <= g2b(wacc_s2);", "n_wdone <= wacc_s2;")], ["default"]),
    ("n_wdone: a B counts one beat, not its burst's", "cb128",
     [("DPW'(wbq_rd[NW-1:0])", "DPW'(1)")], ["default", "seqwr"]),
    ("gather: bursts across the 512-byte block", "cb128",
     [("assign c_sq = n_cwe == c_wel && n_csrc == c_src && n_caddr == c_e && n_caddr[LB-1:0] != '0;",
       "assign c_sq = n_cwe == c_wel && n_csrc == c_src && n_caddr == c_e;"),
      ("assign g_mrg = g_v && h_v && h_seq;", "assign g_mrg = g_v && h_v && h_seq && g_n != NW'(MAXB);"),
      ("assign g_end = g_el == '0 ||", "assign g_end = g_n == NW'(MAXB) ||")],
     ["default", "seqrd", "seqwr"]),
    ("gather: a beat joins whatever its address", "cb128",
     [("n_csrc == c_src && n_caddr == c_e &&", "n_csrc == c_src &&")], ["default"]),
    ("gather: a read joins a write burst", "cb128",
     [("n_cwe == c_wel && n_csrc == c_src &&", "n_csrc == c_src &&")], ["seqrand", "seqrand2", "default"]),
    ("gather: a posted read joins a burst of sources 0-2 (and the reverse)", "cb128",
     [("n_cwe == c_wel && n_csrc == c_src &&", "n_cwe == c_wel &&")], ["postseq", "postseq2", "postcred"],
     "missed"),
    ("W: WLAST at every native beat's end", "cb128",
     [("assign m_wlast  = w_q == KW'(K - 1) && w_left == NW'(1);", "assign m_wlast  = w_q == KW'(K - 1);")],
     ["default", "seqwr"]),
    ("W: a burst issued before its data is in (wav ignores the claims)", "cb128",
     [("wav <= wd_rused - wcl - (p_aw ? WAW'(p_n) : '0);", "wav <= wd_rused;")],
     ["default", "seqwr", "nogaps", "wlead"]),
    ("n_wfree: not 0 while the core side is in reset", "post",
     [("= a_crst ? '0 : (WAW'(WDD) - wd_wused) >> WHB;", "= (WAW'(WDD) - wd_wused) >> WHB;")],
     ["areset", "aresetrep", "aresetshort"]),
    ("n_wfree: a beat more than u_wd's room", "post",
     [("= a_crst ? '0 : (WAW'(WDD) - wd_wused) >> WHB;", "= a_crst ? '0 : (WAW'(WDD + 1) - wd_wused) >> WHB;")],
     ["wlead", "seqwr", "nogaps"]),
    ("W: AXI beats of a native beat in reverse order", "cb128",
     [("assign wp_bd   = wp_d[wp_q * 256 +: 256];", "assign wp_bd   = wp_d[(K - 1 - wp_q) * 256 +: 256];")],
     ["default"]),
    ("W: strobes all set (partial writes clobber)", "cb128",
     [("assign wp_bm   = wp_m[wp_q * 32 +: 32];", "assign wp_bm   = '1;")], ["partial", "default"]),
    ("W: strobes of the wrong AXI beat", "cb128",
     [("assign wp_bm   = wp_m[wp_q * 32 +: 32];", "assign wp_bm   = wp_m[0 +: 32];")], ["partial", "default"]),
    ("W: a beat's halves joined the wrong way round (WH)", "n8",
     [("assign wf_wd = {w_d[16:1], w_lo[15:0], w_d[144:17], w_lo[143:16]};",
       "assign wf_wd = {w_lo[15:0], w_d[16:1], w_lo[143:16], w_d[144:17]};")], ["default", "partial"]),
    ("R: AXI beats into the wrong quarters", "cb128",
     [("assign rd_wv[q] = r_v && !s_post && (s_q & KW'(NL - 1)) == KW'(q) && !a_rs2;",
       "assign rd_wv[q] = r_v && !s_post && (s_q & KW'(NL - 1)) == KW'(NL - 1 - q) && !a_rs2;")], ["default"]),
    ("posted: quarters into the wrong places", "cb128",
     [("pl_q <= {2{pi_v ? K'(1) << pi_q : K'(0)}};", "pl_q <= {2{pi_v ? K'(1) << (KW'(K - 1) - pi_q) : K'(0)}};")],
     ["post30", "post100"]),
    ("posted: the count steps at a beat's first quarter, before its last is written", "cb128",
     [("if (p_we && p_q[0][K-1]) p_cnt <= p_cnt + 1'b1;", "if (p_we && p_q[0][0]) p_cnt <= p_cnt + 1'b1;")],
     ["post30", "post100"]),
    ("posted: a RAM group's copy of the entry not the others'", "cb128",
     [("pl_s <= {5{pi_s}};", "pl_s <= {pi_s ^ XSW'(1), {4{pi_s}}};")], ["post30", "post100"]),
    ("posted: the count in binary, not gray", "cb128",
     [("p_g <= p_cnt ^ (p_cnt >> 1);", "p_g <= p_cnt;")], ["post30", "post100"]),
    ("posted: the beats of reads issued before a core reset not dropped (counted after rst falls)", "cb128",
     [("ps_v[0] <= r_v && s_post && !a_rs2;", "ps_v[0] <= r_v && s_post;")],
     ["postreset", "postresetl"]),
    ("posted: the same at N 8", "n8",
     [("ps_v[0] <= r_v && s_post && !a_rs2;", "ps_v[0] <= r_v && s_post;")],
     ["postreset", "postresetl"]),
    ("posted: the beat's record popped at its first quarter", "cb128",
     [("assign rc_pop = r_v && s_q == KW'(K - 1);", "assign rc_pop = r_v && s_q == '0;")], ["post30", "default"]),
    ("drain: the hold does not wait for what is in flight", "cb128",
     [("assign a_dn = !rbq_rv && !wbq_rv && !p_ar && !p_aw;", "assign a_dn = 1'b1;")],
     ["aresetlat", "aresetrep", "areset"]),
    ("drain: the hold does not wait for the B's (nor the W engine)", "cb128",
     [("assign a_dn = !rbq_rv && !wbq_rv && !p_ar && !p_aw;", "assign a_dn = !rbq_rv && !p_ar;")],
     ["aresetlat", "aresetrep", "areset", "aresetshort"]),
    ("drain: the hold does not wait for the reads in flight", "cb128",
     [("assign a_dn = !rbq_rv && !wbq_rv && !p_ar && !p_aw;", "assign a_dn = !wbq_rv && !p_aw;")],
     ["aresetlat", "aresetrep", "areset", "aresetshort", "postreset"]),
    ("drain: a burst issued the cycle before (not yet in rbq / wbq) not waited for", "cb128",
     [("assign a_dn = !rbq_rv && !wbq_rv && !p_ar && !p_aw;", "assign a_dn = !rbq_rv && !wbq_rv;")],
     ["drainedge", "drainedge2"]),
    ("drain: the hold 2 cycles at N 8 (the pipeline's round trip is 18)", "n8",
     [("localparam int HL   = 2 * (N + 1) + 4 > 16 ? 2 * (N + 1) + 4 : 16;", "localparam int HL   = 2;")],
     ["hreset", "hresetp", "hresetl", "aresetshort"]),
    ("issue: back to back, the next burst tested before the last one's effects are in", "cb128",
     [("assign g_mv  = !a_stop && g_v && g_end && !i_v;", "assign g_mv  = !a_stop && g_v && g_end && (!i_v || i_go);")],
     ["hazard", "small", "hazardlat", "credstress", "default"]),
    ("issue: the registered test kept for the next burst (i_ok without i_v)", "cb128",
     [("always_ff @(posedge aclk) i_ok <= i_v && i_room && !i_haz;", "always_ff @(posedge aclk) i_ok <= i_room && !i_haz;")],
     ["hazard", "small", "hazardlat", "credstress", "default"]),
    ("R: every beat ends its burst (RLAST ignored)", "cb128",
     [("assign r_bend = r_v && r_l && rbq_rv;", "assign r_bend = r_v && rbq_rv;")], ["default"]),
    # (nothing reachable: n_rvalid is off while the core side is in reset, a_crst, which covers the
    # drain, and u_rd's both sides reset in the hold, so a beat written in the drain is never seen;
    # the drop keeps u_rd empty of them, no more)
    ("drain: returning reads delivered (not dropped) while the core is reset", "cb128",
     [("assign rd_wv[q] = r_v && !s_post && (s_q & KW'(NL - 1)) == KW'(q) && !a_rs2;",
       "assign rd_wv[q] = r_v && !s_post && (s_q & KW'(NL - 1)) == KW'(q);")],
     ["aresetlat", "aresetrep", "areset", "aresetshort"], "missed"),
    # half beats (RHB / WHB)
    ("RHB: a beat's low half with lane 1's second entry not yet in u_rd", "hb",
     [("rd_ru[0] >= RLW'(2) && rd_ru[1] >= RLW'(2);", "rd_ru[0] >= RLW'(2);")], ["rgap", "rgapmix", "default"]),
    ("RHB: a low half started in the core's reset (rst not tested, a_crst a cycle late)", "hb",
     [("assign rd_lo  = !rh_p && !rst && !a_crst", "assign rd_lo  = !rh_p && !a_crst")],
     ["aresetshort", "drainedge", "drainedge2"]),
    # (unseen: a core reset raises a_crst before the hold's a_hs2, and with the HBM's reset alone
    # the master resets with the core side, so the beat a hold splits is dropped whole either way)
    ("RHB: a low half started in the hold (a_hs2 not tested)", "hb",
     [("!a_crst && !a_hs2 && rd_ru[0]", "!a_crst && rd_ru[0]")], ["hreset", "hresetp", "hresetl"], "missed"),
    ("RHB: the high half held back by the core's reset", "hb",
     [("assign rd_pop = rd_lo || rh_p;", "assign rd_pop = rd_lo || (rh_p && !rst);")],
     ["areset", "aresetshort", "drainedge", "drainedge2"]),
    ("RHB: the halves from the wrong lanes", "hb",
     [("n_rdata  <= (CB*8)'(rd_rd);", "n_rdata  <= (CB*8)'({rd_rd[0], rd_rd[NL-1]});")], ["default"]),
    ("RHB: AXI beats into the wrong lanes (q / 2, not q mod 2)", "hb",
     [("assign rd_wv[q] = r_v && !s_post && (s_q & KW'(NL - 1)) == KW'(q) && !a_rs2;",
       "assign rd_wv[q] = r_v && !s_post && (s_q >> 1) == KW'(q) && !a_rs2;")], ["default"]),
    ("RHB: a read credit at every lane-1 entry (not every two)", "hb",
     [("assign rd_f = RAW'(rd_fl >> RHB);", "assign rd_f = RAW'(rd_fl);")], ["credstress", "slowcore", "default"]),
    ("WHB: n_wfree in halves, not beats", "hb",
     [("= a_crst ? '0 : (WAW'(WDD) - wd_wused) >> WHB;", "= a_crst ? '0 : (WAW'(WDD) - wd_wused);")],
     ["wlead", "seqwr", "nogaps"]),
    ("WHB: n_wfree a beat more than u_wd's room", "hb",
     [("= a_crst ? '0 : (WAW'(WDD) - wd_wused) >> WHB;", "= a_crst ? '0 : (WAW'(WDD + 2) - wd_wused) >> WHB;")],
     ["wlead", "seqwr", "nogaps"]),
    ("WHB: the write side's half toggle not reset with the core side (a lone low half kept)", "hb",
     [("wd_h <= (a_crst || !WHB) ? 1'b0 : wd_h ^ wd_wv;", "wd_h <= !WHB ? 1'b0 : wd_h ^ wd_wv;")],
     ["areset", "aresetrep", "aresetshort", "hreset", "drainedge2"]),
    ("WHB: an entry's half flag inverted", "hb",
     [("assign wd_wd = {wd_h, n_wmask[CB/2-1:0], n_wdata[CB*4-1:0]};",
       "assign wd_wd = {!wd_h, n_wmask[CB/2-1:0], n_wdata[CB*4-1:0]};")], ["default"]),
    ("WHB: the halves' AXI beats in reverse order", "hb",
     [("assign wp_bd   = wp_d[wp_q * 256 +: 256];", "assign wp_bd   = wp_d[(KP - 1 - wp_q) * 256 +: 256];")],
     ["default"]),
    ("WHB: a half's strobes all set", "hb",
     [("assign wp_bm   = wp_m[wp_q * 32 +: 32];", "assign wp_bm   = '1;")], ["partial", "default"]),
    ("reset: the core's request not kept up until its hold is seen (a short reset)", "cb128",
     [("a_req <= rst || (a_req && !a_hs2);", "a_req <= rst;")], ["aresetshort"]),
    ("reset: the core side leaves reset before the hold is over", "cb128",
     [("a_crst <= rst || a_req || a_hs2;", "a_crst <= rst;")], ["aresetshort", "aresetrep", "aresetlat"]),
]


def build(name: str, params: dict, mut: list | None = None) -> Path:
    params = {**params, **XSW}
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    src = list(SRC)
    if mut:
        i = [p.name for p in SRC].index("otpu_hbm_ch.sv")
        text = SRC[i].read_text()
        for old, new in mut:
            assert text.count(old) == 1, f"mutation {name}: {old!r} found {text.count(old)} times"
            text = text.replace(old, new)
        (d / "otpu_hbm_ch.sv").write_text(text)
        src[i] = d / "otpu_hbm_ch.sv"
    key = hashlib.sha256(repr(params).encode() + b"".join(p.read_bytes() for p in src)).hexdigest()
    exe = d / "obj" / "Vtb_hbmch"
    if exe.exists() and (d / "key").exists() and (d / "key").read_text() == key:
        return exe
    cmd = (["verilator", "--binary", "-j", JOBS, "--top-module", "tb_hbmch", "-Wno-fatal", "-Wno-lint",
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
    tp = {}
    m = re.search(r"^acc: .*?([-0-9.naif]+) beats per cycle", out, re.M)
    if m:
        tp["acc"] = m.group(1)
    m = re.search(r"^hbm \d+: .*?([-0-9.naif]+) R beats per cycle", out, re.M)
    if m:
        tp["hbm_r"] = m.group(1)
    m = re.search(r"^bridge: \d+ read bursts \(([0-9.]+) beats each\), \d+ write bursts \(([0-9.]+)\)", out, re.M)
    if m:
        tp["rb"], tp["wb"] = m.group(1), m.group(2)
    why = ""
    if res != "PASS":
        e = re.search(r"^(ERROR .*|%Error.*|.*otpu_\w+(?: \d+)?: .*)$", out, re.M)
        why = e.group(1)[:160] if e else out.strip().splitlines()[-1][:160] if out.strip() else ""
    return res, tp, why


def fmt(tp: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in tp.items())


def main(argv: list[str]) -> int:
    suites = argv or ["func"]
    if "all" in suites:
        suites = ["func", "perf", "mut"]
    ok = True
    jobs = []                                   # (label, exe, log, args, bar)
    if "quick" in suites:
        exe = build("cb128", BUILDS["cb128"])
        jobs.append(("cb128 default", exe, OUT / "cb128" / "default.log", SCEN["default"] + FBASE, None))
    if "func" in suites:
        for b, s in FUNC:
            exe = build(b, BUILDS[b])
            jobs.append((f"{b} {s}", exe, OUT / b / f"{s}.log", SCEN[s] + FBASE, None))
    if "perf" in suites:
        for b, s, args, bar in PERF:
            exe = build(b, BUILDS[b])
            jobs.append((f"{b} perf {s}", exe, OUT / b / ("perf-" + re.sub(r"\W+", "_", s) + ".log"), args, bar))
    with ThreadPoolExecutor(PAR) as ex:
        futs = [(lab, bar, ex.submit(run, exe, log, args)) for lab, exe, log, args, bar in jobs]
        for lab, bar, f in futs:
            res, tp, why = f.result()
            good = res == "PASS"
            note = ""
            if bar is not None and "hbm_r" in tp:
                share = float(tp["hbm_r"])
                note = f" {100 * share:.1f}% of 14.4 GB/s (bar {100 * bar:.0f}%)"
                good &= share >= bar
            ok &= good
            print(f"{lab:56} {res:8} {fmt(tp)}{note}  {why}", flush=True)
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
