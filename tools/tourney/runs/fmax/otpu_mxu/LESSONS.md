# Lessons: otpu_mxu

- [r20260928-113015-s0] Row-counting (i_rows/c_rows/q_rows) to take the DSP multiply off the ucs->q_total path was no_gain: the build failed routing (Route 35-447 congestion), so fix congestion before judging logic-depth fixes.
- [r20260928-113015-s1] otpu_mxu: row counters in place of the N*KB chunk counters (cmd_total DSP taken off the start cycle) gave OOC +0.7% (139.1->140.1 MHz); lost the round's full build, so re-queue it to check the u_seq->q_total paths.
[note 2026-09-28, stream Q, MCOLS=4 image at 120.755 MHz, -0.114 ns] also critical there: u_mxu/q_h_reg -> u_tmem/pw_d_reg and -> u_vpu/g_wbuf.en_r_reg_rep (15 levels, 88% route); MCOLS=4 doubles the MXU's per-column state, so its area matters (die 96.9% slices).
