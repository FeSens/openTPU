"""The collective across the multi-slice board's crossings alone (sim/verilator/tb_coll_hub.sv:
otpu_coll behind otpu_coll_hub, S slices XRS register stages away each way, each with its
otpu_coll_port and a model of its TMEM's collective port whose grant is withheld at random):
GATHERs of random shapes and BARs back to back, every slice's writes in order against the
reference, the destinations after each GATHER, no ack before a slice's writes are in; resets of
the core mid-GATHER, after which every FIFO, credit and count is back at its reset value; the
hub's source / destination overlap check against an illegal GATHER; and a GATHER's cycles by size
(otpu_board COLL_PIPE 0: one read in flight; 1: the round trip's)."""
import re

import pytest

from opentpu import rtlsim

SRC = [rtlsim.RTL / "top/otpu_pkg.sv", rtlsim.RTL / "top/otpu_coll.sv",
       rtlsim.RTL / "shell/otpu_board.sv", rtlsim.TB / "tb_coll_hub.sv"]
DONE = re.compile(r"TBCH DONE S=(\d+) PIPE=(\d+) XRS=(\d+) NR=(\d+) F=(\d+) ops=(\d+) "
                  r"gathers=(\d+) bars=(\d+) words=(\d+) writes=(\d+) resets=(\d+) cycles=(\d+)")
SIZE = re.compile(r"TBCH SIZE S=(\d+) PIPE=(\d+) XRS=(\d+) words=(\d+) first=(\d+) last=(\d+)")
B2B = re.compile(r"TBCH B2B S=(\d+) PIPE=(\d+) XRS=(\d+) words=256 n=(\d+) cycles=(\d+)")


def _run(s: int, pipe: int, xrs: int, *plus, ok: bool = True) -> str:
    exe = rtlsim.build("tb_coll_hub", SRC, {"S": s, "PIPE": pipe, "XRS": xrs})
    r = rtlsim.run_sim([str(exe), *plus], timeout=1800)
    out = r.stdout + r.stderr
    if ok:
        assert "TBCH DONE" in out and r.returncode == 0, out[-3000:]
    return out


@pytest.mark.parametrize("s", [2, 3, 4])
@pytest.mark.parametrize("pipe,xrs", [(0, 2), (1, 2), (1, 3), (1, 5)])
@pytest.mark.parametrize("seed,stall", [(1, 0), (2, 30), (3, 70)])
def test_gathers_and_bars_across_the_crossings(have_verilator, s, pipe, xrs, seed, stall):
    """300 collectives back to back (a fifth BARs), the slices starting up to 6 cycles apart,
    each port's grant withheld in runs (stall percent of the cycles): every slice's writes as the
    reference's, in order, every destination right, no ack before the writes."""
    out = _run(s, pipe, xrs, f"+seed={seed}", f"+stall={stall}", "+ops=300", "+burst=8")
    g = DONE.search(out).groups()
    assert int(g[6]) > 150 and int(g[7]) > 20 and int(g[9]) > 1000, out[-500:]


@pytest.mark.parametrize("s,pipe", [(2, 0), (2, 1), (4, 1)])
@pytest.mark.parametrize("seed", [4, 5])
def test_reset_cuts_gathers(have_verilator, s, pipe, seed):
    """The core's reset cuts a third of the GATHERs mid-flight (for 2 XRS + 4 .. 40 cycles): after
    it the hub's and every port's FIFOs, credits and counts are at their reset values, and the
    collectives after it are right."""
    out = _run(s, pipe, 2, f"+seed={seed}", "+stall=30", "+ops=200", "+rst=33")
    g = DONE.search(out).groups()
    assert int(g[10]) > 20, out[-500:]


def test_overlap_check_stops_an_illegal_gather(have_verilator):
    """A GATHER whose destination overlaps slice 0's source (a chunk on: each write over words
    reads still out read) stops the simulation at the hub's check, with COLL_PIPE 1; the same
    collectives without it pass, and so does the illegal one with the check off
    (+coll_overlap_ok: the run goes on). At COLL_PIPE 0 that GATHER's reads are each back
    before the next write goes, so the check passes it there: it names what COLL_PIPE 1 relies
    on."""
    out = _run(2, 1, 2, "+seed=6", "+ops=8", "+illegal", ok=False)
    assert "source / destination overlap" in out and "TBCH DONE" not in out, out[-2000:]
    _run(2, 1, 2, "+seed=6", "+ops=8")
    _run(2, 0, 2, "+seed=6", "+ops=8", "+illegal")


@pytest.mark.parametrize("s", [2, 4])
def test_gather_cycles_by_size(have_verilator, s):
    """One-row GATHERs of 16, 256 and 4,096 words a slice, four of each back to back, and four of
    256 timed together: COLL_PIPE 1 moves near LANES words a cycle into each slice (S x words /
    LANES reads, one a cycle, plus the round trip and the ack), COLL_PIPE 0 a read a round trip."""
    res = {}
    for pipe in (0, 1):
        out = _run(s, pipe, 2, "+sizes", "+reps=4", "+stall=0")
        for g in SIZE.findall(out):
            res[pipe, int(g[3])] = int(g[5])
        res[pipe, "b2b"] = int(B2B.search(out).group(5))
    print(f"S={s} cycles a GATHER (last of 4), COLL_PIPE 0 / 1:",
          {k: (res[0, k], res[1, k]) for k in (16, 256, 4096, "b2b")})
    reads = s * 4096 // 8
    assert res[1, 4096] < reads + 60, res                    # a read a cycle, plus fixed costs
    assert res[0, 4096] > 8 * reads, res                     # a read a round trip
    assert res[1, 256] < res[0, 256] / 4, res
