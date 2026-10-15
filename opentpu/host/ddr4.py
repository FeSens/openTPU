"""otpu-ddr4: the memory bring-up of a board with one LiteDRAM DDR4 core per DIMM (the BCU1525:
docs/bcu1525.md section 11), through the cores' CSRs. DIMM d's are in BAR0 at R_DDR4 + d x
DDR4_STRIDE (vivado/bd.tcl's M_AXI_MEMCAL<d>). Each core's CPU (LiteX's BIOS) calibrates its DIMM
at reset and logs on the UART (DIMM C1's core drives the board's); the host reads and tests what
the CSRs reach:

    python -m opentpu.host.ddr4 [status]    per DIMM: the core's identifier and scratch register
                                            (the CSR path), calibrated (cal_ready, or the BIOS's
                                            hand-over: DFII hardware control), the ECC and its
                                            counters; STATUS CALIB0 / CALIB1 / ECC_DED
    python -m opentpu.host.ddr4 bist [--dimm D] [--gib G] [--passes N]
                                            each DIMM's BIST: write, then read and check, G GiB
                                            from the DIMM's start, random then address data, the
                                            wrong beats per byte lane. It writes past the ECC
                                            (the check bits are left wrong), so it clears Board's
                                            scrub marker: the next BoardBackend scrubs the DRAM
    python -m opentpu.host.ddr4 ecc [--clear]
                                            the ECC counters: 64-bit words corrected and
                                            uncorrectable since the last clear

Board channel j = 2s + c (slice s's channel c) is on DIMM CHANNEL_DIMM[j] (otpu_fpga_top_bcu's
dimm()). The cores' CSR map and PHY settings ship in opentpu/host/litedram4/ (copies of
boards/bcu1525/litedram/'s, tools/litedram/gen_core_ddr4.py's output). Only a card: the board
model has no LiteDRAM cores (present()).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from . import ddrcal
from . import regs as R

DATA = Path(__file__).with_name("litedram4")
CHANNEL_DIMM = (0, 1, 3, 2)     # board channel j -> DIMM C<d> (otpu_fpga_top_bcu dimm())
SCRATCH_RESET = 0x12345678      # LiteX's ctrl_scratch after the core's reset
IDENT_MAX = 256                 # bytes of identifier_mem read at most
DFII_SEL = 1                    # sdram_dfii_control bit0: the controller has the PHY


def csr_map(path: Path = DATA / "csr.csv") -> tuple[dict, dict]:
    """csr.csv -> ({register: (offset, 32-bit words)}, {block: offset}), the offsets in the CSR
    region the host's port reaches (16 bits: the addresses less the region's base, 0xF000_0000
    with the core's CPU, whose ROM is at 0)."""
    regs, bases = {}, {}
    for row in csv.reader(open(path)):
        if row and row[0] == "csr_base":
            bases[row[1]] = int(row[2], 16)
        elif row and row[0] == "csr_register":
            regs[row[1]] = (int(row[2], 16), int(row[3]))
    region = min(bases.values()) & ~0xFFFF
    regs = {n: (a - region, w) for n, (a, w) in regs.items()}
    bases = {n: a - region for n, a in bases.items()}
    if any(a + 4 * w > R.DDR4_STRIDE for a, w in regs.values()):
        raise ValueError(f"{path}: a CSR outside the {R.DDR4_STRIDE:#x}-byte window")
    return regs, bases


def present(t) -> bool:
    """t is a card (the board model has no CSR windows: `ecc`, as memcal.ecc) whose bitstream
    (register map 4) names a target with LiteDRAM DDR4 cores."""
    if not getattr(t, "ecc", False) or R.regmap(t.reg_read(R.R_REGMAP)) < R.CFG_SINCE:
        return False
    from opentpu.target import by_platform
    tg = by_platform(t.reg_read(R.R_CFG_PLATFORM))
    return tg is not None and tg.platform.get("mem") == "ddr4" \
        and tg.platform.get("mem_ctrl") == "litedram"


def dimms(t) -> list[int]:
    """The DIMMs of the bitstream's channels, in channel order."""
    c = R.cfg([t.reg_read(o) for o in R.CFG_REGS])
    return [CHANNEL_DIMM[j] for j in range(c["S"] * c["channels"])]


def csr(t, d: int, data: Path = DATA) -> ddrcal.WordCsr:
    """DIMM d's core's CSRs (w / r by csr.csv's names)."""
    regs, _ = csr_map(data / "csr.csv")
    return ddrcal.WordCsr(regs, t.reg_read, t.reg_write, base=R.R_DDR4 + d * R.DDR4_STRIDE)


def ident(t, d: int, data: Path = DATA) -> str:
    """DIMM d's core's identifier string (identifier_mem: a byte per word)."""
    _, bases = csr_map(data / "csr.csv")
    a = R.R_DDR4 + d * R.DDR4_STRIDE + bases["identifier_mem"]
    out = bytearray()
    for i in range(IDENT_MAX):
        b = t.reg_read(a + 4 * i)
        if b == 0 or b > 0xFF:          # the end, or no core answering (all ones)
            break
        out.append(b)
    return out.decode("ascii", "replace")


def status(t, d: int, data: Path = DATA) -> dict:
    """DIMM d: its core's identifier, the scratch register's write / read-back (restored),
    calibrated (cal_ready: a host calibration; DFII hardware control: the BIOS's hand-over after
    a passing init), the ECC enable and counters."""
    c = csr(t, d, data)
    was = c.r("ctrl_scratch")
    pat = 0xA5C3_0F00 | d
    c.w("ctrl_scratch", pat)
    ok = c.r("ctrl_scratch") == pat
    c.w("ctrl_scratch", was)
    dfii = c.r("sdram_dfii_control")
    return {"dimm": d, "ident": ident(t, d, data), "scratch_ok": ok, "scratch": was,
            "cal_ready": bool(c.r("cal_ready") & 1), "dfii_hw": bool(dfii & DFII_SEL),
            "calibrated": bool(c.r("cal_ready") & 1 or dfii & DFII_SEL),
            "ecc_enable": bool(c.r("ecc_enable") & 1),
            "ecc_sec": c.r("ecc_sec_errors"), "ecc_ded": c.r("ecc_ded_errors")}


def ecc(t, clear: bool = False, data: Path = DATA) -> dict | None:
    """Every DIMM's ECC counters {d: (corrected, uncorrectable)}, cleared after with `clear`;
    None when t is not a DDR4 card (Board.scrub calls it on every board)."""
    if not present(t):
        return None
    out = {}
    for d in dimms(t):
        c = csr(t, d, data)
        out[d] = (c.r("ecc_sec_errors"), c.r("ecc_ded_errors"))
        if clear:
            c.w("ecc_clear", 1)
    return out


def sys_hz(data: Path = DATA) -> float:
    """The controllers' clock (sdram_init.py's phy settings): one 64-byte beat per cycle."""
    ns = {}
    exec((data / "sdram_init.py").read_text(), ns)
    return float(ns["phy"]["sys_hz"])


def bist(t, d: int, gib: float = 1.0, passes: int = 1, data: Path = DATA) -> list[dict]:
    """DIMM d's BIST over its first `gib` GiB: per pass, random then address data, written
    then read back and checked (ddrcal.bist; the beats are the 576-bit raw port's, past the
    ECC). Clears SCRATCH's DRAM_INIT, so the next Board.scrub writes the DRAM again."""
    c = csr(t, d, data)
    hz = sys_hz(data)
    beats = int(gib * (1 << 30)) // R.DRAM_BEAT
    out = []
    t.reg_write(R.R_SCRATCH, 0)
    for p in range(passes):
        for dmode, name in ((0, "random"), (2, "address")):
            seed = (0x0123456789ABCDEF ^ (p * 0x9E3779B97F4A7C15)) & ((1 << 64) - 1)
            w = ddrcal.bist(c, beats, dmode, seed, hz)
            r = ddrcal.bist(c, beats, dmode | 1, seed, hz)
            out.append({"dimm": d, "pass": p, "data": name, "beats": beats,
                        "write_gbs": round(w["gbs"], 2), "read_gbs": round(r["gbs"], 2),
                        "errors": r["errors"], "lanes": r["lanes"], "bits": r["bits"],
                        "ok": r["errors"] == 0 and r["beats"] == beats})
    return out


def describe(s: dict, j: int) -> str:
    how = "+".join(n for n, v in (("BIOS", s["dfii_hw"]), ("cal_ready", s["cal_ready"])) if v)
    cal = f"calibrated ({how})" if s["calibrated"] else "NOT calibrated"
    return (f"DIMM C{s['dimm']} (channel {j}, slice {j // 2}): '{s['ident']}', scratch "
            f"{'ok' if s['scratch_ok'] else 'FAILED'} ({s['scratch']:#010x}), {cal}, ECC "
            f"{'on' if s['ecc_enable'] else 'off'}: {s['ecc_sec']} corrected, {s['ecc_ded']} "
            f"uncorrectable")


def main(argv: list[str] | None = None, open_transport=None) -> int:
    ap = argparse.ArgumentParser(prog="otpu-ddr4", description=__doc__.split("\n")[0])
    ap.add_argument("cmd", nargs="?", default="status", choices=("status", "bist", "ecc"))
    ap.add_argument("--dev", help="the card's device prefix (default: /dev/otpu0 if present, "
                    "else /dev/xdma0)")
    ap.add_argument("--dimm", type=int, action="append", help="DIMM (default: every one)")
    ap.add_argument("--gib", type=float, default=1.0, help="bist: GiB per DIMM (from its start)")
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--clear", action="store_true",
                    help="ecc: clear the counters after (every DIMM's)")
    a = ap.parse_args(argv)
    if open_transport is None:
        from .board import card_transport
        # the BIST writes over the DRAM: it holds the device lock (a DMA transport's); status
        # and ecc only read registers, as otpu-smi
        t = card_transport(a.dev, dma=a.cmd == "bist")
    else:
        t = open_transport()
    try:
        return _run(a, t)
    finally:
        if open_transport is None:
            t.close()


def _run(a, t) -> int:
    if not present(t):
        print("otpu-ddr4: not a DDR4 LiteDRAM card (register map 4 with a DDR4 target, "
              "CFG_PLATFORM)", file=sys.stderr)
        return 2
    chans = {d: j for j, d in enumerate(dimms(t))}
    sel = a.dimm or list(chans)
    bad = [d for d in sel if d not in chans]
    if bad:
        print(f"otpu-ddr4: no DIMM {bad} (the bitstream's: {list(chans)})", file=sys.stderr)
        return 2
    ok = True
    if a.cmd == "status":
        st = t.reg_read(R.R_STATUS)
        print(f"STATUS {st:#010x}: CALIB0 {int(bool(st & R.ST_CALIB0))} CALIB1 "
              f"{int(bool(st & R.ST_CALIB1))} ECC_DED {int(bool(st & R.ST_ECC_DED))}")
        for d in sel:
            s = status(t, d)
            print(describe(s, chans[d]))
            ok &= s["scratch_ok"] and s["calibrated"]
    elif a.cmd == "ecc":
        for d, (sec, ded) in ecc(t, clear=a.clear).items():
            if d in sel:
                print(f"DIMM C{d} (channel {chans[d]}): {sec} corrected, {ded} uncorrectable")
                ok &= ded == 0
    else:
        for d in sel:
            for r in bist(t, d, a.gib, a.passes):
                print(f"DIMM C{d} pass {r['pass']} {r['data']}: write {r['write_gbs']:.2f} GB/s, "
                      f"read {r['read_gbs']:.2f} GB/s, errors {r['errors']} (per lane "
                      f"{r['lanes']}, bits {r['bits']:#x})")
                ok &= r["ok"]
        print("otpu-ddr4: the BIST wrote past the ECC: the next BoardBackend scrubs the DRAM")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
