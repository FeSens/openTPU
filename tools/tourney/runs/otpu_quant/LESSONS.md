# Lessons: otpu_quant

- [r20260923-234940-s0] qscale: making the SRL→DSP operands real flops (sync-reset last stage of otpu_qdelay) and dropping the bd mux in front of u_inv only took fmax 94→96 MHz (+2%, area +0.1%), not enough; look past SRL clk→Q.
- [r20260923-234940-s1] Per-lane 1W1R sbuf RAMs (NB*2^CI x 32) cut area 56%, but fmax only went 94->96: FF tails on the qscale DSP operand delays didn't fix timing. Keep sbuf split; the critical path needs its own fix.
- [r20260924-001302-s0] otpu_quant scale unit: fp_add(p,-0) folds into SRL chains that fed the DSPs; ending them in real FFs and dropping the ftz from DSP operands (dead when !sp) gave +9.4% fmax, -7.2% area.
- [r20260924-001302-s1] qscale Newton u_y fmadd(+(-0))→fmul+flop-ended otpu_qdly: bit-exact and removes the SRL clk→Q on y[i], but only +2.2% fmax (96→~98 MHz). Below the gate; the DSP chain dominates.
- [r20260924-002052-s0] qfadd s2 carry-free sticky (guard bits from mb low bits, one {A,1}+{B,cin} add, no -mb negation chain) took area -4.1% but fmax only 105→107 MHz; the q8_s1 four-chain path is probably critical next.
- [r20260924-002052-s1] Newton-step fadd s2 (in otpu_qscale) was critical. Replacing the sticky mask `(1<<d)-1` with a trailing-zero count registered in s1 and using one ma±mb adder gave +3.5% fmax (109 MHz) and −1.5% area.
- [r20260924-003919-s0] qfmul_s2 moving the round overflow (mr[24]=rnd&(&mm)) and exponent off the carry chain via a parallel incrementer + e/e+1/e+2 mux: fmax only +1.5%, area +4.3% across 32 instances. No gain.
- [r20260924-003919-s1] qfmul round stage (shared by all 32 instances): computing ov=rnd&(&mm) in place of mr[24] removed the >>1 mux and the post-carry exp increment. Area -1.9%, fmax only 107→109 MHz, so look for the rest elsewhere.
- [r20260924-004705-s0] Yosys ignores the shreg_extract="no" attribute, so otpu_delay's last stage packs into an SRL; ending Newton u_y with the reset-flop otpu_qdly removed the SRL→DSP path: +4.6% fmax (114 MHz).
- [r20260924-004705-s1] Dropping u_y's LM+LA pad (Newton period 2*SL→SL+LM, LAT 38→26) moved y[i] off an SRL CLK→Q onto an FDRE feeding the DSP B input: 109→116 MHz, -0.7% area, bit-identical. Check for SRL-driven DSPs.
- [r20260924-005428-s0] Amax tree: comparing raw magnitudes (a[30:0]>b[30:0], so no fkey/ftz on sign-0 words) plus a single-level 6-compare stage 2 took 116→133 MHz with -1.3% area. Yosys can't prove invariants across flops.
- [r20260924-005428-s1] amax path values are always non-negative flushed magnitudes, so fp_gt/ftz can become 31-bit unsigned compares; a single shared bamax running max (amax2 retired) gave -3.1% area at 133 MHz.
- [r20260924-010050-s0] qfmul PM=1 (separate {exp!=0,mant} regs so yosys packs DSP AREG/BREG): fmax +0.1%, area +2.3%. The 4394ps DSP0→DSP1 cascade path didn't move. Duplicating operand regs doesn't pay.
- [r20260924-010050-s1] QST byte address as running sums (badr/bgrp/brs incremental adds, delayed QL+1 with en to align with mp) instead of row*drs+rel*es DSP multiplies: -1.1% area; strength-reduce address math.
