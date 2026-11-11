"""Session thermal guard, shared with old production bring-up trees by filename.
Only the existing governor API is required; no cross-card host module is imported here.
"""
from opentpu.host import gov as G


class ThermalStop(RuntimeError):
    pass


class SessionGuard(G.Guard):
    """The session stops at 80 C, including a hot reading while SAFE. Emergency means SAFE,
    never replacing the loaded image. The fabric's independent thermal trips remain active.
    Used by the bring-up wrapper too, with its JTAG temperature readings available."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **dict(kw, die_safe_c=80.0, reg_safe_c=78.0))

    def emergency(self, why):
        self.trip(why)

    def read_first(self, *args, **kw):
        r = super().read_first(*args, **kw)
        if (self.reg_max() or 0) >= 80:
            self.trip(f"power chip {self.reg_max()} C >= 80 C")
            raise ThermalStop(self.tripped)
        return r

    def step(self, *args, **kw):
        result = super().step(*args, **kw)
        die = max((self.last.get(k) or -273 for k in ("die_c", "jtag_c")))
        if die >= 80 or (self.reg_max() or 0) >= 80:
            self.trip(f"die {die} C, power chip {self.reg_max()} C: >= 80 C")
        return result
