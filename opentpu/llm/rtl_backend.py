"""Engine backend that runs every token on the Verilator RTL (slow: minutes per token for a
real model; used to prove the RTL runs the real network bit-exactly)."""
from __future__ import annotations

import numpy as np

from .. import rtlsim


class RtlBackend:
    def __init__(self, cfg, images: list, dram_lat: int = 8, uarch: dict | None = None,
                 axi: bool | None = None, boot: bool | None = None, stall: int | None = None):
        self.cfg, self.dram_lat, self.uarch = cfg, dram_lat, uarch
        self.axi, self.boot, self.stall = axi, boot, stall
        self.drams = [np.asarray(im, np.uint8).copy() for im in images]

    def write(self, s: int, addr: int, data: np.ndarray) -> None:
        v = np.ascontiguousarray(data).view(np.uint8).reshape(-1)
        self.drams[s][addr:addr + v.size] = v

    def read(self, s: int, addr: int, nbytes: int) -> np.ndarray:
        return self.drams[s][addr:addr + nbytes].copy()

    args = True                 # run(programs, args): the run's arguments (R8..R15)
    generates = True            # the generate loop (RLD, ARGMAX: Engine.generate_card)
    chains = True               # and HALT CHAIN

    def slot_words(self, prog) -> np.ndarray:
        """A chain slot's contents for `prog` (generate.slot_words): its words, and with otpu_seq
        WAITC (EXPERIMENTAL, OTPU_SEQ_WC=1: rtlsim.UARCH) its wait entries after them, which
        HALT CHAIN's load reads (rtlsim.run places the first program's itself)."""
        from .. import isa as I
        from .. import waits as W
        w = np.asarray(I.assemble(prog), np.uint32)
        if not {**rtlsim.UARCH, **(self.uarch or {})}.get("SEQ_WC"):
            return w
        return W.loadable(w, W.entries(w, self.cfg.D, self.cfg.S, prog=prog, aw=self.cfg.AW,
                                         win=self.cfg.WC_WIN),
                          self.cfg.D)

    def run(self, programs: list, args=None) -> dict:
        drams, _, stats = rtlsim.run(self.cfg, programs, self.drams, self.dram_lat,
                                     max_cycles=1 << 40, uarch=self.uarch, axi=self.axi,
                                     boot=self.boot, stall=self.stall, args=args)
        n = [len(d) for d in self.drams]
        self.drams = [d[:k].copy() for d, k in zip(drams, n)]
        return stats
