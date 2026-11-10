"""FakeTransport: an in-memory card for tests and demos (otpu-smi --fake).

It implements the register protocol the driver uses -- register map 3 (or 4: the CFG_*
registers; or 5: CAPS2, bit0 GATHER = `gather`, bit1 NO_OUTER = not `outer`, bit2 DSTEP2 =
`dstep2`, bit3 SE_OWN = `se_own`, bit5 KT = `kt`, bits 11:6 BHW the current target's, bit12
ICACHE = the CFG_UARCH bit28 `cfg` reports; or 2: no
MXU_STARVE; or 1, regmap=1: 8-bit address decode, 0xDEADBEEF for registers v1 does not have) --
over two channel memories in RAM (two per slice when `cfg` reports more than one: ICOUNT_s as
ICOUNT).
It computes nothing: a run takes `run_s` seconds of wall time (HALTED rises then), reports
`cycles` and retires PROG_N instructions; the DRAM keeps what the host wrote.

The free-running counters advance deterministically: every SNAP adds `step` cycles of uptime
and RATES x step to the others, so two snapshots give exactly RATES as utilization (and a DRAM
bandwidth of (DRAM_RD + DRAM_WR rates) x 64 B x CORE_KHZ). The trace buffer serves `trace`
(a list of 64-bit records) with TRACE_COUNT = len(trace) + `trace_extra` (records that fell
out of the buffer) and TRACE_DROP = `trace_drop`.

With `i2c` (two fake_i2c.OpenDrainBus, e.g. fake_i2c.card_buses()) CAPS announces the I2C pins
and I2C_CTRL / I2C_IN drive and read those bus models. With `ddr_mts` CAPS bit3 announces the
DDR_MTS register; without it the register reads 0xDEADBEEF, as on older bitstreams. With
`args` CAPS bit25 announces the ARG0..7 registers (kept, read back), with regmap=4 the CFG_*
registers report `cfg` (regs.cfg's keys; default the current target's single-slice values with
the CFG_HASH its build would report; `seq_wc`: CFG_UARCH bit24, a sequencer that takes wait
entries, EXPERIMENTAL), with `mx` (default) CAPS bit29, the MX numerics every
bitstream of this tree has (False: a bitstream before MX, which the host refuses), with `gen`
CAPS bit30 the decode loop's instructions (it runs nothing either way), with `chash` CAPS bit7
the hashed
channel interleave (the run's logits pieces land by that map). `calib` (STATUS CALIB0,
CALIB1): CTRL takes RUN only while both are set, as the RTL; `ecc_ded`: STATUS ECC_DED (set
by hand: sticky on the card). Its identity (VERSION: D,
MCOLS, LANES) is by default the configuration the environment asks for (board_config:
OTPU_MCOLS, OTPU_LANES), as a card built for it would report; so are its ACT RAM rows
(`act_rows`, the target's: 16 on the card; MCOLS when MCOLS is given): above MCOLS, CAPS bit24
announces the ACT_ROWS register and the CFG_HASH is that Config's. With `bmc` (a
fake_bmc.FakeBmc) at register map 5, CAPS2 bit4 announces the BMC's UART bridge and BAR0
R_BMC's 64 KB are its registers.

FakeBcu is the BCU1525's BAR0 around one (otpu-smi --fake-bcu): the governor, the BMC bridge
with its watch and the SYSMON, the accelerator's control registers fenced while it is off.
"""
from __future__ import annotations

import time

import numpy as np

from . import regs as R
from .board import XdmaTransport as _Xdma   # (its poll; bound here: tests patch board's name)
from .board import beat_at

RATES = {"RUNNING": 0.80, "MXU_BUSY": 0.60, "MXU_MAC": 0.50, "VPU_BUSY": 0.12,
         "QNT_BUSY": 0.05, "DMA_BUSY": 0.03, "TMEM_DENY": 0.01, "DRAM_RD": 0.70,
         "DRAM_WR": 0.02, "DRAM_WAIT": 0.20, "INSTR": 0.005, "MXU_STARVE": 0.08}


def _target_cfg(D: int, MCOLS: int, LANES: int, pair: bool, dstep: bool, stream: bool,
                act_rows: int = 0, seq_wc: bool = False, outer: bool = True,
                dstep2: bool = False, se_own: bool = False, kt: bool = False) -> dict:
    """The CFG_* registers of the current target's build with these VERSION / CAPS values."""
    from ..target import arch_hash, current
    t = current()
    a, u, p = t.arch, t.uarch, t.platform
    cfg = dict(S=a["S"], TMEM_WORDS=a["TMEM_WORDS"], IMEM_WORDS=a["IMEM_WORDS"],
               ACT_BLOCKS=a["ACT_BLOCKS"], DRAM_BYTES=a["DRAM_BYTES"],
               channels=p["channels_per_slice"], channel_bytes=p["channel_bytes"],
               beat_bytes=p["beat_bytes"] if t.kind == "fpga" else 64,
               MXU_IMPL=u["MXU_IMPL"], TMEM_IMPL=u["TMEM_IMPL"], VPU_CL=u["VPU_CL"],
               ULANES=u["ULANES"], SEQ_WC=int(seq_wc), ICACHE=int(a["ICACHE"]),
               platform=t.platform_id)
    from ..isasim import Config
    full = Config(**{**a, "D": D, "MCOLS": MCOLS, "LANES": LANES,
                     "ACT_ROWS": act_rows if act_rows > MCOLS else 0, "PAIR": pair,
                     "DSTEP": dstep, "STREAM": stream, "SEQ_WC": seq_wc, "OUTER": outer,
                     "DSTEP2": dstep2, "SE_OWN": se_own, "KT": kt,
                     "ST16": a["ST16"] if dstep else ""})
    cfg["hash"] = arch_hash(full)
    return cfg


class FakeTransport:
    batched = False

    def __init__(self, ch_bytes: int = 1 << 20, regmap: int = 3, devname: str | None = "fake0",
                 run_s: float = 0.0, cycles: int = 1_000_000, core_khz: int = 100_000,
                 build_id: int = 0x1234ABCD, temp_code: int = 0x9C4, trace_log2: int = 12,
                 step: int = 1_000_000, trace: list[int] | None = None, trace_extra: int = 0,
                 trace_drop: int = 0, D: int | None = None, MCOLS: int | None = None,
                 LANES: int | None = None,
                 i2c: list | None = None, ddr_mts: int | None = None,
                 w4: bool = True, pair: bool = False, dstep: bool = False,
                 args: bool = False, stream: bool = False, gen: bool = False,
                 chash: bool = False, calib: tuple = (True, True), ecc_ded: bool = False,
                 cfg: dict | None = None, act_rows: int | None = None, mx: bool = True,
                 gather: bool = True, seq_wc: bool = False, outer: bool = True,
                 dstep2: bool = False, se_own: bool = False, bmc=None, kt: bool = False):
        self.ch = [np.zeros(ch_bytes, np.uint8) for _ in range(2)]
        self.v, self.devname, self.dev = regmap, devname, devname and f"/dev/{devname}"
        self.run_s, self.cycles_per_run = run_s, cycles
        self.core_khz, self.build_id, self.temp_code = core_khz, build_id, temp_code
        self.trace_log2, self.step = trace_log2, step
        self.trace, self.trace_extra, self.trace_drop = list(trace or []), trace_extra, trace_drop
        if act_rows is None:            # the target's rows, unless MCOLS is the caller's
            act_rows = MCOLS
        if None in (D, MCOLS, LANES):
            from ..isasim import board_config
            c = board_config()
            if act_rows is None:
                act_rows = c.act_rows
            D, MCOLS, LANES = (c.D if D is None else D, c.MCOLS if MCOLS is None else MCOLS,
                               c.LANES if LANES is None else LANES)
        self.version = D << 16 | MCOLS << 8 | LANES
        self.act_rows = act_rows        # > MCOLS: CAPS bit24 and the ACT_ROWS register
        self.cfg_words = R.cfg_words(cfg or _target_cfg(D, MCOLS, LANES, pair, dstep, stream,
                                                        act_rows, seq_wc, outer, dstep2,
                                                        se_own, kt))
        c = R.cfg(self.cfg_words)
        self.slices = c["S"] if regmap >= R.CFG_SINCE else 1
        self.icache = bool(c["ICACHE"])  # CAPS2 bit12 (register map 5): the build's CFG_UARCH bit28
        # a slice's channels and their beat (CFG_DRAM; two of 64 bytes before register map 4)
        self.nch, self.beat = (c["channels"], c["beat_bytes"]) if regmap >= R.CFG_SINCE \
            else (2, 64)
        if self.slices * self.nch > 2:  # NCH channels per slice (board.channel_map)
            self.ch = [np.zeros(ch_bytes, np.uint8) for _ in range(self.nch * self.slices)]
        self.i2c = i2c
        self.ddr_mts = ddr_mts          # None: a bitstream without the DDR_MTS register
        self.w4 = w4                    # CAPS bit4: the MXU runs 4-bit weights
        self.pair = pair                # CAPS bit5: MM PAIR / QACT DUP
        self.dstep = dstep              # CAPS bit6: DSTEP
        self.args = args                # CAPS bit25: ARG0..7 (R_ARG0 + 4k, read back)
        self.stream = stream            # CAPS bit26: the stream engine
        self.gen = gen                  # CAPS bit30: the decode loop (RLD, ARGMAX, HALT CHAIN)
        self.chash = chash              # CAPS bit7: the hashed channel interleave
        self.mx = mx                    # CAPS bit29: the MX numerics (every bitstream of this tree)
        self.gather = gather            # CAPS2 bit0 (register map 5): the collective's GATHER
        self.outer = outer              # CAPS2 bit1 clear: the VPU's OUTER
        self.dstep2 = dstep2            # CAPS2 bit2: DSTEP's pair
        self.se_own = se_own            # CAPS2 bit3: the streams on an engine of their own
        self.kt = kt                    # CAPS2 bit5: MM's K tail
        self.bmc = bmc                  # CAPS2 bit4 (register map 5): a fake_bmc.FakeBmc at R_BMC
        self.calib = list(calib)        # STATUS CALIB0 / CALIB1
        self.ecc_ded = ecc_ded          # STATUS ECC_DED
        self.regs = {R.R_CTRL: 0, R.R_PROG_ADDR: 0, R.R_PROG_N: 0, R.R_SCRATCH: 0,
                     R.R_TRACE_CTRL: 0, R.R_TRACE_ADDR: 0, R.R_I2C_CTRL: 0}
        self.count = {k: 0 for k in R.counters(regmap)}
        self.shadow = dict(self.count)
        self.snaps = 0
        self.t_run = None               # wall time RUN rose
        self._wrote = 0                 # logits pieces written by this run
        self._t_logits = None           # the run the logits pieces belong to
        self.runs = 0
        self.reads = 0                  # register reads (poll cost)

    # ---- memory
    def mem_write(self, ch, off, data):
        self.ch[ch][off:off + len(data)] = data

    def mem_read(self, ch, off, n, out=None):
        self._halted()                  # the run's logits pieces land
        if out is None:
            return self.ch[ch][off:off + n].copy()
        out[:] = self.ch[ch][off:off + n]
        return out

    # ---- registers
    streams = False             # DMA during a run (BoardBackend's streamed logits): set with
    logits = None               # (addr, nbytes, piece): the run writes the region piece by
    #                             piece, piece i at run_s * (0.5 + 0.5 * (i + 1) / pieces) (an
    #                             LM head in the run's second half; the last one at the halt)
    logits_lag = 0.0            # the last piece lands this long after HALTED (a store in flight)

    def _put(self, addr: int, b: np.ndarray) -> None:
        """Logical bytes at any address into slice 0's channels (beats in turn; CHASH:
        board.beat_at's map)."""
        i = 0
        while i < len(b):
            x = addr + i
            k = min(len(b) - i, self.beat - x % self.beat)
            c, off = beat_at(x, self.chash, self.nch, self.beat)
            self.ch[c][off:off + k] = b[i:i + k]
            i += k

    def _halted(self) -> bool:
        now = time.perf_counter()
        t_run = self._t_logits          # the stores in flight land even after RUN falls
        if self.logits is not None and t_run is not None:
            a, n, piece = self.logits
            k = -(-n // piece)
            while self._wrote < k and now - t_run >= self.run_s * (
                    0.5 + 0.5 * (self._wrote + 1) / k) + (self.logits_lag
                                                           if self._wrote == k - 1 else 0):
                o = self._wrote * piece
                m = min(piece, n - o)
                vals = (np.arange(o // 4, (o + m) // 4, dtype=np.float32) % 997 * 1e-3)
                self._put(a + o, vals.view(np.uint8))
                self._wrote += 1
        return self.t_run is not None and now - self.t_run >= self.run_s

    def reg_write(self, off, val):
        if self.bmc is not None and R.R_BMC <= off < R.R_BMC + 0x10000:
            return self.bmc.reg_write(off - R.R_BMC, val)
        if self.v < 2:
            off &= 0xFF
            if off >= R.R_REGMAP:
                return
        if off == R.R_CTRL:
            if not (all(self.calib) or self.regs[R.R_CTRL] & R.CTRL_RUN):
                val &= ~(R.CTRL_RUN | R.CTRL_LOAD)  # (uncalibrated: RUN and LOAD not taken)
            if val & R.CTRL_RUN and not self.regs[R.R_CTRL] & R.CTRL_RUN:
                self.t_run = time.perf_counter()
                self._wrote, self._t_logits = 0, self.t_run
                self.runs += 1
            elif not val & R.CTRL_RUN:
                self.t_run = None
        if off == R.R_SNAP:
            self.snaps += 1
            for k in self.count:
                self.count[k] += self.step if k == "UPTIME" else int(RATES[k] * self.step)
            self.shadow = dict(self.count)
            return
        if off == R.R_TRACE_CTRL and val & R.TR_CLEAR:
            return
        if off == R.R_I2C_CTRL and self.i2c:
            val &= 0xF
            for k, bus in enumerate(self.i2c):
                bus.drive(bool(val >> 2 * k & 1), bool(val >> 2 * k & 2))
        self.regs[off] = val & 0xFFFFFFFF

    def reg_read(self, off):
        self.reads += 1
        if self.bmc is not None and R.R_BMC <= off < R.R_BMC + 0x10000:
            return self.bmc.reg_read(off - R.R_BMC)
        if self.v < 2:
            off &= 0xFF                                  # v1 decodes 8 address bits
            if off >= R.R_REGMAP:
                return R.UNMAPPED
        if off == R.R_ID:
            return R.ID_OTPU
        if off == R.R_VERSION:
            return self.version
        if off == R.R_STATUS:
            run = bool(self.regs[R.R_CTRL] & R.CTRL_RUN)
            h = self._halted()
            idle = self.logits is None or self._t_logits is None or \
                self._wrote >= -(-self.logits[1] // self.logits[2])
            return (R.ST_HALTED if run and h else 0) | (R.ST_WR_IDLE if idle else 0) \
                | (R.ST_CALIB0 if self.calib[0] else 0) | (R.ST_CALIB1 if self.calib[1] else 0) \
                | (R.ST_RUN if run else 0) | (R.ST_ECC_DED if self.ecc_ded else 0)
        if off in (R.R_CYCLES, R.R_CYCLES_HI):
            c = self.cycles_per_run if self.runs else 0
            return c & 0xFFFFFFFF if off == R.R_CYCLES else c >> 32
        if off == R.R_ICOUNT or (self.slices > 1 and R.R_ICOUNT_S + 4 <= off
                                 < R.R_ICOUNT_S + 4 * self.slices):
            return self.regs[R.R_PROG_N] if self.runs else 0
        if off in (R.R_B_RD, R.R_B_WR, R.R_A_RD, R.R_A_WR, R.R_B_STALL):
            return 0
        if off == R.R_REGMAP:
            return self.v
        if self.v >= R.CFG_SINCE and off in R.CFG_REGS:
            return self.cfg_words[R.CFG_REGS.index(off)]
        if self.v >= R.CAPS2_SINCE and off == R.R_CAPS2:
            from ..target import current
            return ((R.CAP2_GATHER if self.gather else 0) | (0 if self.outer else R.CAP2_NO_OUTER)
                    | (R.CAP2_DSTEP2 if self.dstep2 else 0) | (R.CAP2_SE_OWN if self.se_own else 0)
                    | (R.CAP2_BMC if self.bmc else 0) | (R.CAP2_KT if self.kt else 0)
                    | current().arch["BHW"] << R.CAP2_BHW_SHIFT   # (the CFG_HASH's, _target_cfg)
                    | (R.CAP2_ICACHE if self.icache else 0)
                    | (R.ST16_NAMES.index(current().arch["ST16"]) << R.CAP2_ST16_SHIFT
                       if self.dstep else 0)
                    | (R.CAP2_FSUM if current().arch.get("FSUM") and self.dstep else 0)
                    | (R.CAP2_SPLITQ if current().arch.get("SPLITQ") else 0)
                    | (R.CAP2_ST16S if current().arch.get("ST16S") and self.dstep
                       and current().arch["ST16"] else 0)
                    | (R.CAP2_QHL if current().arch.get("QHL") and current().arch.get("SPLITQ")
                       else 0)
                    | (R.CAP2_QSTHL if current().arch.get("QSTHL")
                       and current().arch.get("SPLITQ") else 0)
                    | (R.CAP2_VSIG if current().arch.get("VSIG") and self.dstep else 0)
                    | (R.CAP2_VMAC if current().arch.get("VMAC") else 0)
                    | (R.CAP2_OUTER1 if current().arch.get("OUTER1") else 0)
                    | (R.CAP2_QONCE if current().arch.get("QONCE") else 0)
                    | (R.CAP2_SPLMM if current().arch.get("SPLMM") else 0))
        if off == R.R_CAPS:
            return (R.CAP_TRACE | R.CAP_TEMP | (R.CAP_I2C if self.i2c else 0)
                    | (R.CAP_DDR if self.ddr_mts else 0) | (R.CAP_W4 if self.w4 else 0)
                    | (R.CAP_PAIR if self.pair else 0)
                    | (R.CAP_DSTEP if self.dstep else 0) | (R.CAP_ARGS if self.args else 0)
                    | (R.CAP_STREAM if self.stream else 0) | (R.CAP_GEN if self.gen else 0)
                    | (R.CAP_CHASH if self.chash else 0) | (R.CAP_MX if self.mx else 0)
                    | (R.CAP_ACT if self.act_rows > self.version >> 8 & 0xFF else 0)
                    | self.trace_log2 << 8 | 6 << 16)
        if off == R.R_ACT_ROWS and self.act_rows > self.version >> 8 & 0xFF:
            return self.act_rows
        if off == R.R_CORE_KHZ:
            return self.core_khz
        if off == R.R_BUILD_ID:
            return self.build_id
        if off == R.R_DDR_MTS and self.ddr_mts:
            return self.ddr_mts
        if off == R.R_TEMP:
            return R.TEMP_VALID | self.temp_code
        if off == R.R_SNAP:
            return self.snaps
        if off == R.R_I2C_IN and self.i2c:
            v = R.I2C_ALERT0
            for k, bus in enumerate(self.i2c):
                scl, sda = bus.read()
                v |= (scl | sda << 1) << 2 * k
            return v
        for k, o in R.counters(self.v).items():
            if off in (o, o + 4):
                v = self.shadow[k]
                return v & 0xFFFFFFFF if off == o else v >> 32
        if off == R.R_TRACE_COUNT:
            return len(self.trace) + self.trace_extra
        if off == R.R_TRACE_DROP:
            return self.trace_drop
        if off in (R.R_TRACE_LO, R.R_TRACE_HI):
            a = self.regs[R.R_TRACE_ADDR]
            rec = self.trace[a] if a < len(self.trace) else 0
            if off == R.R_TRACE_HI:
                self.regs[R.R_TRACE_ADDR] = (a + 1) % (1 << self.trace_log2)
                return rec >> 32
            return rec & 0xFFFFFFFF
        return self.regs.get(off, R.UNMAPPED)

    def reg_read_many(self, offs):
        return [self.reg_read(o) for o in offs]

    def poll(self, off, mask, val, timeout=600.0, expect=0.0):
        return _Xdma.poll(self, off, mask, val, timeout, expect)


def target_cfg(name: str) -> dict:
    """The CFG_* registers (regs.cfg's keys) of target `name`'s build (its own CFG_HASH)."""
    from ..target import arch_hash, build_config, load
    t = load(name)
    a, u, p = t.arch, t.uarch, t.platform
    return dict(S=a["S"], TMEM_WORDS=a["TMEM_WORDS"], IMEM_WORDS=a["IMEM_WORDS"],
                ACT_BLOCKS=a["ACT_BLOCKS"], DRAM_BYTES=a["DRAM_BYTES"],
                channels=p["channels_per_slice"], channel_bytes=p["channel_bytes"],
                beat_bytes=p["beat_bytes"], MXU_IMPL=u["MXU_IMPL"], TMEM_IMPL=u["TMEM_IMPL"],
                VPU_CL=u["VPU_CL"], ULANES=u["ULANES"], SEQ_WC=int(bool(a.get("SEQ_WC"))),
                WG=a.get("WG", 1), ICACHE=int(bool(a.get("ICACHE"))), platform=t.platform_id,
                hash=arch_hash(build_config(t)))


class FakeBcu:
    """The BCU1525's BAR0 at WG 8 (otpu_fpga_top_bcu_wg) for otpu-smi's tests and demo: the
    governor at gov.R_GOV (otpu_bcu_gov's registers as they read), the BMC bridge at R_BMC (a
    fake_bmc.FakeBmc, with the watch by default), the SYSMON at gov.R_SYSMON, and below R_GOV the
    accelerator's control registers: `acc` (a FakeTransport of bcu1525-wg's build, CAPS2 BMC),
    0xDEAD0FF0 while it is off (`run` False or `acc_on` False: the governor's SAFE, the fence).
    `die_c` is the SYSMON register's die, `gov_die_c` the governor's (its temperature bus; the
    die by default; `gov_valid` False: its TEMP_VALID clear; `temp_age_ms`: a build with the
    staleness check, its code that old, stale and not valid from 60 s; None: a build before it),
    `reg_c` the regulator's hottest (the watch's: the FakeBmc's readings); `sysmon` False: no
    SYSMON (its registers read all ones); `gov_image`: a build whose governor holds the image's
    identity (gov.IMAGE ..), `build_id` its BUILD_ID.
    Writes at R_GOV and above are recorded in `writes` and change nothing (a monitor makes
    none)."""
    FENCED = 0xDEAD0FF0

    def __init__(self, run: bool = False, acc_on: bool | None = None, die_c: float = 62.0,
                 gov_die_c: float | None = None, max_c: float | None = None, alarm: bool = False,
                 cause: int = 0x8, trips: tuple = (0, 0), run_ms: int = 0, kick_ms: int = 0xFFFF,
                 reg_age_ms: int = 4, bmc=None, devname: str = "fakebcu0", core_khz: int = 150000,
                 gov_valid: bool = True, sysmon: bool = True, temp_age_ms: int | None = None,
                 gov_image: bool = False, build_id: int = 0x1234ABCD, **acc_kw):
        from . import gov as G
        from .fake_bmc import FakeBmc
        self.G = G
        self.bmc = bmc if bmc is not None else FakeBmc()
        self.devname, self.dev = devname, f"/dev/{devname}"
        self.run, self.acc_on = run, run if acc_on is None else acc_on
        self.die_c, self.max_c, self.alarm = die_c, max(die_c, max_c or die_c), alarm
        self.gov_die_c = die_c if gov_die_c is None else gov_die_c
        self.cause, self.trips, self.run_ms, self.kick_ms = cause, trips, run_ms, kick_ms
        self.reg_age_ms, self.gov_valid, self.has_sysmon = reg_age_ms, gov_valid, sysmon
        self.temp_age_ms = temp_age_ms
        self.gov_image, self.core_khz = gov_image, core_khz
        self.writes: list[tuple[int, int]] = []
        code = G.code_of(self.gov_die_c)
        kw = dict(regmap=R.CAPS2_SINCE, D=128, MCOLS=8, LANES=8, act_rows=16,
                  cfg=target_cfg("bcu1525-wg"), core_khz=core_khz, ddr_mts=2400, temp_code=code << 2,
                  dstep=True, stream=True, args=True, gen=True, chash=True, outer=False,
                  dstep2=True, bmc=self.bmc, devname=devname, build_id=build_id)
        self.acc = FakeTransport(**{**kw, **acc_kw})
        self.build_id = build_id

    def _up(self) -> bool:
        return self.run and self.acc_on

    def _gov(self, o: int) -> int:
        G = self.G
        code = G.code_of(self.gov_die_c)
        reg = max([p["temp_c"] for p in self.bmc.ltc.pages] + [self.bmc.ltc.values["die_c"]])
        rq = round(4 * reg) & 0xFFFF
        reg_ok = self.reg_age_ms < 10000
        age = self.temp_age_ms
        stale = age is not None and age >= 60000
        if o == G.STATUS:
            return (int(self.run) | int(code <= 674) << 2 | int(self.kick_ms < 2000) << 3
                    | int(self.gov_valid and not stale) << 4 | int(self.alarm) << 5 | 1 << 7
                    | ((0xF << 8 | int(self.acc_on) << 12) if self.run else 0)
                    | int(reg_ok) << 13 | int(reg_ok and rq <= 300) << 14
                    | int(reg_ok and rq >= 304) << 15 | (0 if self.run else self.cause << 16))
        return {G.ID: G.GOV_ID, G.CTRL: 0x100 | (0x1F if self.run else 0),
                G.TEMP: code << 16 | code, G.TRIP: 724, G.RESUME: 674, G.HB_MS: 2000,
                G.KICK: self.kick_ms, G.TRIPS: self.trips[1] << 16 | self.trips[0],
                G.RUN_MS: self.run_ms, G.LIMITS: 704 << 16 | 724, G.REG: rq << 16 | rq,
                G.REG_TRIP: 312, G.REG_RESUME: 300, G.REG_STALE: 10000,
                G.REG_AGE: self.reg_age_ms, G.REG_LIMITS: 300 << 16 | 312,
                G.TEMP_STALE: 0 if age is None else 60000,
                G.TEMP_AGE: 0 if age is None else min(age, 0x1FFFF) | stale << 31,
                **self._image()}.get(o, 0)

    def _image(self) -> dict:
        """The governor's identity words (`gov_image`: a build with them), the accelerator's
        own configuration as they hold it."""
        if not self.gov_image:
            return {}
        from ..target import by_platform, cards
        G, c = self.G, R.cfg(self.acc.cfg_words)
        n = cards(by_platform(c["platform"]))
        return {G.IMAGE: G.IMG_ID, G.IMG_PLAT: n << 24 | c["S"] << 16 | c["platform"],
                G.IMG_HASH: c["hash"], G.IMG_BUILD: self.build_id, G.IMG_KHZ: self.core_khz}

    def _sysmon(self, o: int) -> int:
        G = self.G
        c16 = lambda c: G.code_of(c) << 6                          # noqa: E731
        v16 = lambda v: round(v / 3.0 * 65536)                     # noqa: E731
        return {G.SM_TEMP: c16(self.die_c), G.SM_VCCINT: v16(0.807), G.SM_VCCAUX: v16(1.822),
                G.SM_VBRAM: v16(0.85), G.SM_MAX_TEMP: c16(self.max_c), G.SM_MAX_VCCINT: v16(0.815),
                G.SM_MIN_TEMP: c16(28.0), G.SM_MIN_VCCINT: v16(0.8),
                G.SM_FLAG: 0x200 | int(self.alarm), G.SM_TEMP_UP: c16(79.9),
                G.SM_OT_UP: c16(83.9), G.SM_TEMP_LO: c16(69.9), G.SM_OT_LO: c16(65.0)}.get(o, 0)

    def reg_read(self, off: int) -> int:
        G = self.G
        if G.R_GOV <= off < G.R_GOV + 0x10000:
            return self._gov(off - G.R_GOV)
        if R.R_BMC <= off < R.R_BMC + 0x10000:
            return self.bmc.reg_read(off - R.R_BMC)
        if G.R_SYSMON <= off < G.R_SYSMON + 0x10000:
            return self._sysmon(off - G.R_SYSMON) if self.has_sysmon else 0xFFFFFFFF
        if off < G.R_GOV:
            return self.acc.reg_read(off) if self._up() else self.FENCED
        return R.UNMAPPED

    def reg_read_many(self, offs):
        return [self.reg_read(o) for o in offs]

    def reg_write(self, off: int, val: int) -> None:
        if off >= self.G.R_GOV:
            self.writes.append((off, val))
        elif self._up():
            self.acc.reg_write(off, val)

    def mem_write(self, ch, off, data):
        self.acc.mem_write(ch, off, data)

    def mem_read(self, ch, off, n, out=None):
        return self.acc.mem_read(ch, off, n, out)

    def poll(self, off, mask, val, timeout=600.0, expect=0.0):
        return _Xdma.poll(self, off, mask, val, timeout, expect)
