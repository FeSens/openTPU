#!/usr/bin/env python3
"""csr_pipe (Wishbone2CSRWait + CSRGroups) against LiteX's registered Wishbone2CSR and shared CSR
bus, in migen's simulator.

Two copies of the same banks (three groups of two), one behind each bus. Each bank has a 32-bit
CSRStorage, a 64-bit one with atomic_write (two words, taken when the last is written), a 40-bit
CSRStatus (two words: the wide storage xor a constant), a plain CSR (its write and read strobes,
as the PHYs' delay-line tap inc / rst) and a CSRStorage of two pulse fields. Both get the same
random Wishbone accesses (reads and writes, to every word and to unmapped offsets, with random
gaps). Every access must return the same data and leave the same storages behind; every strobe
(the storages' re, the CSR's re / we with its data, the pulse fields) must come in the same order
with the same data, one per access; the pipelined access must take WAIT cycles more.

    python3 csr_pipe_check.py [--accesses 3000] [--seed 1]      (LiteX at core.json's commits)
"""
import argparse
import random
import sys
from pathlib import Path

from migen import Module, run_simulation
from migen.sim import passive
from litex.soc.interconnect import csr_bus, wishbone
from litex.soc.interconnect.csr import CSR, CSRField, CSRStatus, CSRStorage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr_pipe import CSRGroups, Wishbone2CSRWait                 # noqa: E402

GROUPS, BANKS, PAGE = 3, 2, 0x200          # banks per group; a bank's words (0x800 bytes)
WORDS = 7                                  # a bank's CSR words: storage, wide (2), status (2),
                                           # strobe, pulse


class Bank(Module):
    def __init__(self, address, key):
        self.storage = CSRStorage(32, reset=key)
        self.wide = CSRStorage(64, reset=key << 16, atomic_write=True)
        self.status = CSRStatus(40)
        self.strobe = CSR(32)
        self.pulse = CSRStorage(fields=[CSRField("inc", size=1, pulse=True),
                                        CSRField("rst", size=1, pulse=True)])
        self.comb += self.status.status.eq(self.wide.storage[:40] ^ key)
        self.bus = csr_bus.Interface(data_width=32, address_width=14)
        self.submodules.bank = csr_bus.CSRBank(
            [self.storage, self.wide, self.status, self.strobe, self.pulse], address, self.bus)


class Side(Module):
    def __init__(self, piped):
        self.wb = wishbone.Interface(data_width=32, address_width=32, addressing="word")
        cbus = csr_bus.Interface(data_width=32, address_width=14)
        cls = Wishbone2CSRWait if piped else wishbone.Wishbone2CSR
        self.submodules.bridge = cls(self.wb, cbus, register=True)
        self.banks = [Bank(i, 0x5A5A0000 + 0x1111 * i) for i in range(GROUPS * BANKS)]
        self.submodules += self.banks
        if piped:
            self.submodules.bus = CSRGroups(cbus, [[b.bus for b in self.banks[g * BANKS:(g + 1) * BANKS]]
                                                   for g in range(GROUPS)])
        else:
            self.submodules.bus = csr_bus.InterconnectShared([cbus], [b.bus for b in self.banks])


class Top(Module):
    def __init__(self):
        self.submodules.a = Side(False)
        self.submodules.b = Side(True)


def check(n, seed):
    rnd = random.Random(seed)
    ops = []
    for _ in range(n):
        bank = rnd.randrange(GROUPS * BANKS + 1)            # the last: unmapped
        adr = bank * PAGE + rnd.choice(list(range(WORDS + 1)) + [0x1ff])
        ops.append((rnd.random() < 0.5, adr, rnd.getrandbits(32), rnd.randrange(4)))
    top = Top()
    logs = {"a": [], "b": [], "a_stb": [], "b_stb": [], "a_len": [], "b_len": []}

    def master(side, key):
        wb = side.wb
        for we, adr, dat, gap in ops:
            yield wb.cyc.eq(1)
            yield wb.stb.eq(1)
            yield wb.we.eq(we)
            yield wb.adr.eq(adr)
            yield wb.dat_w.eq(dat)
            yield wb.sel.eq(0xf)
            t = 0
            while True:
                yield
                t += 1
                if (yield wb.ack):
                    break
            rd = None if we else (yield wb.dat_r)
            yield wb.cyc.eq(0)
            yield wb.stb.eq(0)
            yield
            st = []
            for b in side.banks:
                st.append(((yield b.storage.storage), (yield b.wide.storage)))
            logs[key].append((we, adr, rd, st))
            logs[key + "_len"].append(t)
            for _ in range(gap):
                yield

    @passive
    def strobes(side, key):
        while True:
            for i, b in enumerate(side.banks):
                if (yield b.strobe.re):
                    logs[key + "_stb"].append(("w", i, (yield b.strobe.r)))
                if (yield b.strobe.we):
                    logs[key + "_stb"].append(("r", i))
                if (yield b.storage.re):
                    logs[key + "_stb"].append(("storage", i, (yield b.storage.storage)))
                if (yield b.wide.re):
                    logs[key + "_stb"].append(("wide", i, (yield b.wide.storage)))
                for f in ("inc", "rst"):
                    if (yield getattr(b.pulse.fields, f)):
                        logs[key + "_stb"].append((f, i))
            yield

    run_simulation(top, [master(top.a, "a"), master(top.b, "b"),
                         strobes(top.a, "a"), strobes(top.b, "b")])
    errs = 0
    for k, (x, y) in enumerate(zip(logs["a"], logs["b"])):
        if x != y:
            errs += 1
            if errs <= 5:
                print(f"access {k}: stock {x[:3]}, piped {y[:3]}")
    if len(logs["a"]) != n or len(logs["b"]) != n:
        errs += 1
        print(f"accesses done: stock {len(logs['a'])}, piped {len(logs['b'])} of {n}")
    if logs["a_stb"] != logs["b_stb"]:
        errs += 1
        print(f"strobes differ: stock {len(logs['a_stb'])}, piped {len(logs['b_stb'])}")
    la, lb = set(logs["a_len"]), set(logs["b_len"])
    if len(la) != 1 or lb != {t + Wishbone2CSRWait.WAIT for t in la}:
        errs += 1
        print(f"access cycles: stock {sorted(la)}, piped {sorted(lb)}")
    kinds = {}
    for st in logs["a_stb"]:
        kinds[st[0]] = kinds.get(st[0], 0) + 1
    print(f"{n} accesses ({sum(1 for o in ops if o[0])} writes), strobes {kinds}, "
          f"access cycles {sorted(la)} -> {sorted(lb)}: "
          + ("PASS" if not errs else f"FAIL ({errs})"))
    return 1 if errs else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--accesses", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    sys.exit(check(a.accesses, a.seed))


if __name__ == "__main__":
    main()
