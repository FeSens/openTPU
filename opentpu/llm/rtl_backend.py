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

    def run(self, programs: list) -> dict:
        drams, _, stats = rtlsim.run(self.cfg, programs, self.drams, self.dram_lat,
                                     max_cycles=1 << 40, uarch=self.uarch, axi=self.axi,
                                     boot=self.boot, stall=self.stall)
        n = [len(d) for d in self.drams]
        self.drams = [d[:k].copy() for d, k in zip(drams, n)]
        return stats
