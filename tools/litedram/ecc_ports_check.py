#!/usr/bin/env python3
"""ecc_ports.NativePortsECC (two native ports, one read decoder) against two stock
LiteDRAMNativePortECC, in migen's simulator, cycle by cycle.

Both get the same random traffic: commands, write data and every ready at random on both ports,
and read data on at most one port a cycle from one bus (the crossbar's), each beat eight 72-bit
codewords (LiteX's SECDED, computed here) with 0, 1 or 2 bits flipped per word, ECC enable and
counter clears at random. Every cycle both must show the same port signals, and each read beat
must come out as its data (no flip, one flip: corrected) or be counted uncorrectable (two
flips); the error counters must equal the stock frontends' sums.

    python3 ecc_ports_check.py [--cycles 4000] [--seed 1]      (LiteX at core.json's commits)
"""
import argparse
import random
import sys
from pathlib import Path

from migen import Module, run_simulation
from litex.soc.cores.ecc import compute_m_n, compute_data_positions, compute_syndrome_positions

from litedram.common import LiteDRAMNativePort
from litedram.frontend.ecc import LiteDRAMNativePortECC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ecc_ports import NativePortsECC                            # noqa: E402

AW, DW, RW = 25, 512, 576


def secded(d: int) -> int:
    """LiteX's ECCEncoder(64): the Hamming codeword (bits 1..71) with the overall parity at 0."""
    m, n = compute_m_n(64)
    cw = [0] * (n + 1)
    for j, p in enumerate(compute_data_positions(n)):
        cw[p] = (d >> j) & 1
    for i, p in enumerate(compute_syndrome_positions(n)):
        cw[p] = sum(cw[q] for q in range(1, n + 1) if q & p and q != p) & 1
    cw[0] = sum(cw[1:]) & 1
    return sum(b << k for k, b in enumerate(cw))


class Top(Module):
    def __init__(self):
        self.ua = [LiteDRAMNativePort("both", AW, DW) for _ in range(2)]
        self.ra = [LiteDRAMNativePort("both", AW, RW) for _ in range(2)]
        self.ub = [LiteDRAMNativePort("both", AW, DW) for _ in range(2)]
        self.rb = [LiteDRAMNativePort("both", AW, RW) for _ in range(2)]
        self.ea = [LiteDRAMNativePortECC(u, r) for u, r in zip(self.ua, self.ra)]
        self.submodules += self.ea
        self.submodules.eb = NativePortsECC(self.ub, self.rb)


def check(cycles: int, seed: int) -> int:
    rnd = random.Random(seed)
    top = Top()
    errs = []
    stats = {"beats": 0, "sec": 0, "ded": 0}

    def gen():
        pend = [[], []]                      # per port: (data, flips per word) of beats in flight
        for cyc in range(cycles):
            en = 0 if rnd.random() < 0.02 else 1
            clr = rnd.random() < 0.01
            for e in top.ea:
                yield e.enable.storage.eq(en)
                yield e.clear.re.eq(clr)
            yield top.eb.enable.storage.eq(en)
            yield top.eb.clear.re.eq(clr)
            # user side: random commands and write data; controller side: random readies
            for u in range(2):
                v = {"cmd_valid": rnd.random() < 0.5, "cmd_we": rnd.getrandbits(1),
                     "cmd_addr": rnd.getrandbits(AW), "wdata_valid": rnd.random() < 0.5,
                     "wdata_data": rnd.getrandbits(DW),
                     "wdata_we": (1 << 64) - 1 if rnd.random() < 0.9 else rnd.getrandbits(64),
                     "rdata_ready": 1, "cmd_ready": rnd.random() < 0.7,
                     "wdata_ready": rnd.random() < 0.6}
                for us, rs in ((top.ua[u], top.ra[u]), (top.ub[u], top.rb[u])):
                    yield us.cmd.valid.eq(v["cmd_valid"])
                    yield us.cmd.we.eq(v["cmd_we"])
                    yield us.cmd.addr.eq(v["cmd_addr"])
                    yield us.wdata.valid.eq(v["wdata_valid"])
                    yield us.wdata.data.eq(v["wdata_data"])
                    yield us.wdata.we.eq(v["wdata_we"])
                    yield us.rdata.ready.eq(1)
                    yield rs.cmd.ready.eq(v["cmd_ready"])
                    yield rs.wdata.ready.eq(v["wdata_ready"])
            # the read bus: a beat on one port (or none), words with 0 / 1 / 2 flips
            p = rnd.choice([None, None, 0, 1])
            data = rnd.getrandbits(DW)
            flips = [rnd.choice([0, 0, 0, 1, 2]) for _ in range(8)]
            bus = 0
            for w in range(8):
                cw = secded((data >> (64 * w)) & ((1 << 64) - 1))
                for b in rnd.sample(range(72), flips[w]):
                    cw ^= 1 << b
                bus |= cw << (72 * w)
            for r in top.ra + top.rb:
                yield r.rdata.data.eq(bus)
            for i in range(2):
                yield top.ra[i].rdata.valid.eq(p == i)
                yield top.rb[i].rdata.valid.eq(p == i)
            if p is not None:
                pend[p].append((data, flips, en))
            yield
            # compare every port signal of the two, and the counters
            for u in range(2):
                for sa, sb, name in [
                        (top.ua[u].cmd.ready, top.ub[u].cmd.ready, "cmd.ready"),
                        (top.ua[u].wdata.ready, top.ub[u].wdata.ready, "wdata.ready"),
                        (top.ua[u].rdata.valid, top.ub[u].rdata.valid, "rdata.valid"),
                        (top.ua[u].rdata.data, top.ub[u].rdata.data, "rdata.data"),
                        (top.ra[u].cmd.valid, top.rb[u].cmd.valid, "raw cmd.valid"),
                        (top.ra[u].cmd.we, top.rb[u].cmd.we, "raw cmd.we"),
                        (top.ra[u].cmd.addr, top.rb[u].cmd.addr, "raw cmd.addr"),
                        (top.ra[u].wdata.valid, top.rb[u].wdata.valid, "raw wdata.valid"),
                        (top.ra[u].wdata.data, top.rb[u].wdata.data, "raw wdata.data"),
                        (top.ra[u].wdata.we, top.rb[u].wdata.we, "raw wdata.we"),
                        (top.ra[u].rdata.ready, top.rb[u].rdata.ready, "raw rdata.ready")]:
                    a, b = (yield sa), (yield sb)
                    if a != b and len(errs) < 10:
                        errs.append(f"cycle {cyc} port {u} {name}: stock {a:#x}, shared {b:#x}")
                if (yield top.ub[u].rdata.valid):
                    d, fl, e = pend[u].pop(0)
                    got = yield top.ub[u].rdata.data
                    stats["beats"] += 1
                    for w in range(8):
                        exp = (d >> (64 * w)) & ((1 << 64) - 1)
                        g = (got >> (64 * w)) & ((1 << 64) - 1)
                        if (fl[w] == 0 or (fl[w] == 1 and e)) and g != exp and len(errs) < 10:
                            errs.append(f"cycle {cyc} port {u} word {w}: {fl[w]} flip(s), "
                                        f"{g:#x} for {exp:#x}")
                        stats["sec"] += fl[w] == 1 and e
                        stats["ded"] += fl[w] == 2 and e
            for name in ("sec_errors", "ded_errors"):
                a = 0
                for e in top.ea:
                    a += yield getattr(e, name).status
                b = yield getattr(top.eb, name).status
                if min(a, 2**32 - 1) != b and len(errs) < 10:
                    errs.append(f"cycle {cyc} {name}: stock {a}, shared {b}")

    run_simulation(top, gen())
    print(f"{cycles} cycles, {stats['beats']} read beats, {stats['sec']} words corrected, "
          f"{stats['ded']} uncorrectable")
    for e in errs:
        print("ERROR", e)
    print("PASS" if not errs else "FAIL")
    return 1 if errs else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    sys.exit(check(a.cycles, a.seed))
