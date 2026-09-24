# Lessons: otpu_seq

- [r20260924-001624-s0] otpu_seq: registering the IMEM word (256b ir) with an exact one-ahead fetch raised fmax 10.5% but cost 1.7% area. Next try: trim the ir width or put the register on the BRAM output (DOB_REG) to stay within +1% area.
- [r20260924-001624-s1] otpu_seq: the 3-bit-digit lt32 loop (11 iterations) across 16 slots × 20 pair tests broke slang's 4000-iteration unroll limit and the build failed. Write the digits out by hand or raise --unroll-limit.
- [r20260924-004406-s0] Splitting the 32-bit reg-relative adds (low 16 at R, high 16+carry a cycle later via a bypassed pending-write reg) moved the CARRY4 upper bits off the BRAM path: +3.9% fmax, only +0.8% area.
- [r20260924-004406-s1] Yosys already maps scoreboard `$lt` compares to 3-bit carry chains, so the lever is fewer range pairs: MM ASCALE sharing rd[1] cut area 10.6%, which paid for s0's registered IR (113 MHz). Accepted.
- [r20260924-005905-s0] Oldest-ready select: build per-slot readiness from flops and AND it with the one-hot oldest vector. Skip the binary encode and mux back through start_slot on the CE path. Gain: +20% fmax, ~0 area.
- [r20260924-005905-s1] Storing scoreboard slots by space (DRAM 2, TMEM 2+always-read t2, ACT narrow 17-bit) instead of by role cut cells 16→13 and dropped all sp compares: area -8.2%. Fewer range pairs is still the lever.
- [r20260924-010905-s0] otpu_seq: precomputing loop-stack predicates (rem>1/>2 flags, per-entry compares) on the fa_d path gave only +1.3% fmax at +0.5% area; the ~500 scoreboard endpoints at 3.3–3.5 ns cap the gain.
- [r20260924-010905-s1] otpu_seq: dropping the 5×240b ucmd FDRE copies and reading `scmd[ucs[u]]` straight from LUT-RAM (4b slot regs) is safe because the slot stays stable until fin. Accepted: −1200 FF, area −2.4% @133MHz.
- [r20260924-011856-s0] Per-entry loop-predicate flag flops (e_eq/a_eq/g1/g2/se + 3 comparator sets) dropped fmax to 109 MHz vs 133 (-18%): the flag next-state logic becomes the new critical path. Avoid it.
- [r20260924-011856-s1] Porting the space-segregated scoreboard to the current champion worked: storing ranges by DRAM/TMEM/ACT with no space compares and hi=0 for invalid ranges (no valid bits) gave -8.8% area at 125 MHz.
- [r20260924-013720-s0] Q-stage rebalance (S pre-adds lo+c, pre-validates product-free cells, Q adds only products) worked for timing, fmax +12.2%, but per-cell Q regs cost area +1.0%. Pair it with an area cut.
- [r20260924-013720-s1] Narrowing the scoreboard's TMEM ranges to 16/17 bits (TMEM_WORDS=2^16), with a fallback to `all` when a range overflows, gave -14% area at 113 MHz; compare widths set LUT area, so trim dead bits.
