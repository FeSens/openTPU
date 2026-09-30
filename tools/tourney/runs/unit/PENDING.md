# Unit tournament: candidates never measured (stopped 2026-09-30)

The unit tournament (`--objective unit`, docs/tourney.md) stopped on 2026-09-30 15:25, when the
user stopped the Fmax push. None of these three slots reached its out-of-context Vivado run, so
none has an accept / reject verdict. Each patch applies to the champion named here (the unit's
RTL as on main then); `--replay <slot>` runs one through the gates again.

| unit | slot | patch | state when stopped |
|---|---|---|---|
| otpu_fp | r20260929-232720-s0 | fadd: drop the s2 exponent pre-increment and the s3 zero detector | logged broken at the fast gate by omarchy's congestion, not by the change: rerun by hand, the same fast gate passes (113 passed in 612 s); replayed as r20260930-092010-s0 (the same patch) on 5a62075, killed in its gates |
| otpu_dma | r20260930-011400-s0 | the stream's row output buffer `ob` in LUT RAM (8 x 32-deep x 32-bit) instead of 8192 flip-flops | logged broken by "lint: timeout after 900s", the queue-counted timeout fixed in 9948ba8; its replay never started |
| otpu_vpu | r20260930-091648-s0 | one lane-level adder in the folding tree, 9-bit converters in the long lanes (`otpu_vtree.sv`, `otpu_vpu.sv`, `otpu_se_tail.sv`), on 4ab54a4 | implemented, lint and the agent's quick tests clean; killed in the orchestrator's gates; hypothesis and notes in `otpu_vpu/patches/r20260930-091648-s0.md` |

Champions measured (Vivado OOC at 7.5 ns): otpu_fp (fadd x64 + fmul x78) area_eq 42290, 223.4 MHz;
otpu_dma area_eq 18044, 144.8 MHz; otpu_vpu area_eq 37685, 154.9 MHz (`<unit>/champion.json`).
