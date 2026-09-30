#!/usr/bin/env python3
"""dfii_q.PhaseInjector (the injected command a sys cycle after its CSR write) against LiteDRAM's
DFIInjector, in migen's simulator.

Two DFIInjectors (4 phases), LiteDRAM's and one built under registered_injector(), each behind
its own CSR bank, get the same CSR writes and the same controller (slave) traffic, and the same
read data from the PHY side. The writes follow the calibration's use: per access a phase's
address, bank, write data or command, a command issue on any phase (activate / write with its
write data enable / read with its read data enable / precharge, or a write or read data enable
alone), the control register (sel, cke, odt, reset_n); gaps of 0-4 cycles between accesses (the
production bus's are 4 at least), so injected commands on consecutive accesses and on different
phases (a command on one, the data enable on another) come a few cycles apart. Per cycle:
  - the command signals (cs_n, ras_n, cas_n, we_n, wrdata_en, rddata_en) of every phase equal
    LiteDRAM's a cycle earlier, while software drives the DFI (sel 0);
  - every cycle LiteDRAM's DFI carries an injected command, the whole 4-phase word (commands,
    address, bank, write data and mask, cke, odt, reset_n) comes out the same a cycle later:
    the phases keep their alignment;
  - the storage-driven signals (address, bank, write data and mask, cke, odt, reset_n) equal
    LiteDRAM's in the same cycle, and under sel 1 every signal does (the controller's commands);
  - the read data registers (rddata) equal LiteDRAM's.
It fails with the strobe not registered (--delay 0) or registered twice (--delay 2).

    python3 dfii_q_check.py [--accesses 20000] [--seed 1] [--delay 1]   (LiteX at core.json's)
"""
import argparse
import random
import sys
from pathlib import Path

from migen import Module, run_simulation
from migen.sim import passive
from litex.soc.interconnect import csr_bus
from litedram.dfii import DFIInjector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dfii_q import PhaseInjector, registered_injector            # noqa: E402

NPH, AB, BB, DB = 4, 14, 3, 16          # phases, address / bank bits, DFI data bits per phase
CMD = ("cs_n", "ras_n", "cas_n", "we_n", "wrdata_en", "rddata_en")
STATIC = ("address", "bank", "wrdata", "wrdata_mask", "cke", "odt", "reset_n")
# the command register's fields: cs we cas ras wren rden (cs_top / cs_bottom stay 0: one rank)
CS, WE, CAS, RAS, WREN, RDEN = 1, 2, 4, 8, 16, 32
COMMANDS = [RAS | CS, CAS | WE | CS | WREN, CAS | CS | RDEN, RAS | WE | CS, WREN, RDEN,
            CAS | WE | CS, CS, 0]


class Side(Module):
    def __init__(self):
        self.submodules.dfii = DFIInjector(AB, BB, 1, DB, NPH)
        self.bus = csr_bus.Interface(data_width=32, address_width=14)
        self.submodules.bank = csr_bus.CSRBank(self.dfii.get_csrs(), 0, self.bus)
        self.word = {c.name: i for i, c in enumerate(self.bank.simple_csrs)}


class Top(Module):
    def __init__(self):
        self.submodules.a = Side()
        with registered_injector():
            self.submodules.b = Side()
        assert self.a.word == self.b.word, "the CSRs' names or order differ"
        assert all(isinstance(getattr(self.b.dfii, f"pi{n}"), PhaseInjector) for n in range(NPH))


def check(accesses, seed, delay):
    PhaseInjector.DELAY = delay
    top = Top()
    rnd = random.Random(seed)
    word = top.a.word
    ops = []                            # (csr word, data, idle cycles after)
    for _ in range(accesses):
        r = rnd.random()
        ph = rnd.randrange(NPH)
        if r < 0.40:
            op = (f"pi{ph}_command_issue", 1)
        elif r < 0.60:
            op = (f"pi{ph}_command", rnd.choice(COMMANDS))
        elif r < 0.70:
            op = (f"pi{ph}_address", rnd.getrandbits(AB))
        elif r < 0.78:
            op = (f"pi{ph}_baddress", rnd.getrandbits(BB))
        elif r < 0.88:
            op = (f"pi{ph}_wrdata", rnd.getrandbits(DB))
        else:                           # control: mostly software (sel 0), as in calibration
            op = ("control", (rnd.random() < 0.2) | rnd.getrandbits(3) << 1)
        ops.append((word[op[0]], op[1], rnd.choice([0, 0, 1, 2, 4])))
    log = {"a": [], "b": []}

    def master(side):
        yield side.bus.we.eq(0)
        for w, d, gap in ops:
            yield side.bus.adr.eq(w)
            yield side.bus.dat_w.eq(d)
            yield side.bus.we.eq(1)
            yield
            yield side.bus.we.eq(0)
            for _ in range(gap):
                yield
        for _ in range(4):
            yield

    @passive
    def slave(sides):                   # the controller's DFI and the PHY's read data, random
        r = random.Random(seed + 1)
        while True:
            for n in range(NPH):
                vals = {k: r.getrandbits(len(getattr(sides[0].dfii.slave.phases[n], k)))
                        for k in CMD + ("address", "bank", "wrdata", "wrdata_mask")}
                rd, rv = r.getrandbits(DB), r.random() < 0.3
                for s in sides:
                    for k, v in vals.items():
                        yield getattr(s.dfii.slave.phases[n], k).eq(v)
                    yield s.dfii.master.phases[n].rddata.eq(rd)
                    yield s.dfii.master.phases[n].rddata_valid.eq(rv)
            yield

    @passive
    def monitor(side, key):
        while True:
            ph, rdd = [], []
            for n in range(NPH):
                p, v = side.dfii.master.phases[n], {}
                for k in CMD + STATIC:
                    v[k] = yield getattr(p, k)
                ph.append(v)
                rdd.append((yield getattr(side.dfii, f"pi{n}")._rddata.status))
            sel = yield side.dfii._control.fields.sel
            log[key].append((sel, ph, rdd))
            yield

    run_simulation(top, [master(top.a), master(top.b), slave([top.a, top.b]),
                         monitor(top.a, "a"), monitor(top.b, "b")])
    A, B = log["a"], log["b"]
    errs, stats = [], {"cycles": len(A), "sw": 0, "hw": 0, "cmd_cycles": 0, "multi_phase": 0,
                       "b2b": 0}

    def err(t, what):
        if len(errs) < 8:
            errs.append(f"cycle {t}: {what}")
        else:
            errs.append(None)

    def injected(ph):
        return any(p["cs_n"] == 0 or p["wrdata_en"] or p["rddata_en"] for p in ph)

    last_cmd = None
    for t in range(len(A) - 1):
        sa, pa, ra = A[t]
        sb, pb, rb = B[t]
        if ra != rb:
            err(t, f"rddata {ra} vs {rb}")
        for n in range(NPH):
            for k in STATIC:
                if pa[n][k] != pb[n][k]:
                    err(t, f"phase {n} {k}: {pa[n][k]} vs {pb[n][k]} (same cycle)")
        if sa:
            stats["hw"] += 1
            for n in range(NPH):
                if pa[n] != pb[n]:
                    err(t, f"sel 1, phase {n}: {pa[n]} vs {pb[n]}")
            continue
        if B[t + 1][0]:
            continue                    # software to hardware at t + 1: the controller's DFI
        stats["sw"] += 1
        pb1 = B[t + 1][1]
        for n in range(NPH):
            for k in CMD:
                if pa[n][k] != pb1[n][k]:
                    err(t, f"sel 0, phase {n} {k}: {pa[n][k]}, a cycle later {pb1[n][k]}")
        if injected(pa):
            stats["cmd_cycles"] += 1
            if pa != pb1:
                err(t, f"the injected word {pa} came out as {pb1} a cycle later")
            cmd = {n for n in range(NPH) if pa[n]["cs_n"] == 0}
            den = {n for n in range(NPH) if pa[n]["wrdata_en"] or pa[n]["rddata_en"]}
            # a data enable on another phase than a command's, within 5 cycles of it
            if den and last_cmd is not None and t - last_cmd[0] <= 5 and den - last_cmd[1]:
                stats["multi_phase"] += 1
            if last_cmd is not None and t - last_cmd[0] <= 2:
                stats["b2b"] += 1
            if cmd:
                last_cmd = (t, cmd)
    if len(A) != len(B):
        err(len(A), f"lengths {len(A)} vs {len(B)}")
    shown = [e for e in errs if e]
    for e in shown:
        print(e)
    ok = not errs and stats["cmd_cycles"] > 0
    print(f"{accesses} CSR writes, delay {delay}: {stats['cycles']} cycles ({stats['sw']} software, "
          f"{stats['hw']} hardware), {stats['cmd_cycles']} with an injected command "
          f"({stats['multi_phase']} data enables on another phase than a command 5 cycles before at "
          f"most, {stats['b2b']} within 2 cycles of the last command): "
          + ("PASS" if ok else f"FAIL ({len(errs)})"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--accesses", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--delay", type=int, default=1, help="PhaseInjector.DELAY (mutations: 0, 2)")
    a = ap.parse_args()
    sys.exit(check(a.accesses, a.seed, a.delay))


if __name__ == "__main__":
    main()
