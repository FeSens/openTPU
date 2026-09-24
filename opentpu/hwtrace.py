"""The board's hardware trace (rtl/boards/ypcb-00338/otpu_trace.sv; docs/observability.md):
64-bit records -> the trace lines the simulator prints with +trace (opentpu/profile.py).

A record is [63:60] type, [59:0] payload; the cycle fields are the slice's cycle counter (the
`c=` of the trace lines), low 32 bits:

    1 D  [55:52] slot, [47:32] pc, [31:0] cycle; two O records follow
    9 O  [52] half, [51:0] that half of {op[7:0], w3, w2, w1} (half 0 first)
    2 S  [55:52] slot, [51:48] unit, [47:32] cycle - ready cycle, [31:0] cycle; 0xFFFF in the
         delay field: an R record follows
   10 R  [31:0] the ready cycle of the S before it
    3 G  [55:52] slot, [31:0] cycle
    4 E  [55:52] slot, [31:0] cycle
    5 U  [51:48] unit, [31:0] cycle; V records follow: unit 1 starve bp frz deny, 2 frz, 3 frz
    6 P  [55:32] n, [31:0] cycle (window end); 9 V: bm bd am aq mx fm fq fv fc
    7 Q  [55:32] n, [31:0] cycle; 6 V: bs as ms mb ff ld
    8 H  [31:0] cycle; 4 V: bmxu bdma amxu aq
   11 V  [59:56] field index, [31:0] value

Records come in the simulator's line order. A group (a record and the ones that follow it) that
is cut by the ring's start or the buffer's end is skipped.
"""
from __future__ import annotations

T_D, T_S, T_G, T_E, T_U, T_P, T_Q, T_H, T_O, T_R, T_V = range(1, 12)
U_FIELDS = {1: ["starve", "bp", "frz", "deny"], 2: ["frz"], 3: ["frz"]}
P_FIELDS = ["bm", "bd", "am", "aq", "mx", "fm", "fq", "fv", "fc"]
Q_FIELDS = ["bs", "as", "ms", "mb", "ff", "ld"]
H_FIELDS = ["bmxu", "bdma", "amxu", "aq"]

# control registers (rtl/boards/ypcb-00338/otpu_ctrl.sv)
R_TRACE_CTRL, R_TRACE_COUNT, R_TRACE_DROP = 0x200, 0x204, 0x208
R_TRACE_ADDR, R_TRACE_LO, R_TRACE_HI = 0x20C, 0x210, 0x214
TRACE_ENABLE, TRACE_CLEAR, TRACE_STOP_WHEN_FULL, TRACE_BUSY = 1, 2, 4, 8


def _typ(r: int) -> int:
    return (r >> 60) & 0xF


def ring_order(raw, count: int, depth: int) -> list[int]:
    """The ring's contents (raw[i] = the record at index i) oldest first, given TRACE_COUNT."""
    raw = [int(r) for r in raw]
    if count <= depth:
        return list(raw[:count])
    k = count % depth
    return list(raw[k:depth]) + list(raw[:k])


def _values(records: list[int], i: int, k: int) -> list[int] | None:
    """The k V records after index i (field indices 0 .. k-1), or None if they are not there."""
    if i + k >= len(records):
        return None
    vs = []
    for j in range(k):
        r = records[i + 1 + j]
        if _typ(r) != T_V or (r >> 56) & 0xF != j:
            return None
        vs.append(r & 0xFFFF_FFFF)
    return vs


def records_to_trace(records, sid: int = 0) -> str:
    """Trace records (ints or a uint64 array), oldest first -> the trace lines
    (newline-terminated), exactly as the simulator prints them for the same run."""
    records = [int(r) for r in records]
    out = []
    i, n = 0, len(records)
    while i < n:
        r = records[i]
        t = _typ(r)
        cyc = r & 0xFFFF_FFFF
        slot = (r >> 52) & 0xF
        if t == T_D:
            o = records[i + 1:i + 3]
            if len(o) == 2 and all(_typ(x) == T_O and (x >> 52) & 1 == h for h, x in enumerate(o)):
                m = (1 << 52) - 1
                ow = (o[0] & m) | ((o[1] & m) << 52)
                w1, w2, w3 = ow & 0xFFFF_FFFF, (ow >> 32) & 0xFFFF_FFFF, (ow >> 64) & 0xFFFF_FFFF
                op = (ow >> 96) & 0xFF
                out.append(f"T{sid} D c={cyc} s={slot} pc={(r >> 32) & 0xFFFF} op={op:02x} "
                           f"w1={w1:08x} w2={w2:08x} w3={w3:08x}")
                i += 3
            else:
                i += 1
        elif t == T_S:
            dt = (r >> 32) & 0xFFFF
            if dt == 0xFFFF:
                if i + 1 >= n or _typ(records[i + 1]) != T_R:
                    i += 1
                    continue
                rdy = records[i + 1] & 0xFFFF_FFFF
                i += 2
            else:
                rdy = (cyc - dt) & 0xFFFF_FFFF
                i += 1
            out.append(f"T{sid} S c={cyc} s={slot} u={(r >> 48) & 0xF} r={rdy}")
        elif t in (T_G, T_E):
            out.append(f"T{sid} {'G' if t == T_G else 'E'} c={cyc} s={slot}")
            i += 1
        elif t in (T_U, T_P, T_Q, T_H):
            unit = (r >> 48) & 0xF
            fields = {T_U: U_FIELDS.get(unit, []), T_P: P_FIELDS, T_Q: Q_FIELDS,
                      T_H: H_FIELDS}[t]
            vs = _values(records, i, len(fields)) if fields else None
            if vs is None:
                i += 1
                continue
            kv = " ".join(f"{f}={v}" for f, v in zip(fields, vs))
            if t == T_U:
                out.append(f"T{sid} U c={cyc} u={unit} {kv}")
            elif t == T_H:
                out.append(f"T{sid} H c={cyc} {kv}")
            else:
                out.append(f"T{sid} {'P' if t == T_P else 'Q'} c={cyc} n={(r >> 32) & 0xFF_FFFF} {kv}")
            i += 1 + len(fields)
        else:                   # an O, R or V whose group was cut, or an empty record
            i += 1
    return "".join(line + "\n" for line in out)
