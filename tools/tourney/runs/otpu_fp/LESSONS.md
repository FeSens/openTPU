# Lessons: otpu_fp

- [r20260924-001644-s0] fadd s2 sticky: a trailing-zero count computed in s1 (tzc24 tree, parallel with compare) + `d > tzp` compare replaced sticky_below's mask/reduce chain. Result: 114 MHz, area -4.1%; fmul is now the limit.
- [r20260924-001644-s1] fp_add: let zeros take the normal datapath (hidden bit = exp!=0) and cut the special value from 32 to 3 bits across 3 regs; −12.9% area at 114 MHz. fadd FFs count 64×, so shrink pipelined state.
- [r20260924-002708-s0] fmul s2: the rounding carry-out is just inc=p[47]|&p[46:22] (no adder carry). Precompute hi/lo range flags plus e8 in s1 and use a 3-bit sv. Result: 114→123 MHz, area -2.3%.
- [r20260924-002708-s1] Round-carry trick worked: drop the post-round `mr>>1` (the mantissa is 0 whenever carry-out fires) and set ovf/unf from e0 alongside the carry chain. fmul/fadd area fell 2.5%, fmax 114→123 MHz.
- [r20260924-003650-s0] fmul: feeding the DSP mantissa product from raw inputs instead of ftz() operands dropped 2 LUT levels; result bit-identical since sp=1 masks m.p; accepted: -6.3% area, 145 MHz.
- [r20260924-003650-s1] fmul s1: feeding raw mantissas to the DSP and dropping ftz() (special path uses exp==0 flags) worked, -6.3% area at 145 MHz. Logic in front of the DSP whose result s2 throws away can go.
- [r20260924-004426-s0] fadd s3: one 28-bit normalize (lzc32({sum,4'b1111}) + one shift + e+1-lz) replaced the add/sub mux after the shifter, which saves a LUT per mantissa bit: area -3.1%. fmul still caps Fmax (~141-145 MHz).
- [r20260924-004426-s1] fadd s3: a single 28-bit LZC normalize (with e+1 precomputed in s2) replaced the add/sub mantissa/exponent mux while staying bit-identical, which cut area 4.2%. Look for per-case muxes stacked after shifters.
- [r20260924-005536-s0] fadd s2 sticky from a tz27 count precomputed in s1 went +3.2% area (the earlier -4.1% didn't stack on zero-rides-datapath): two 24b tz trees + swap mux cost more than the sticky_below mask. Don't retry.
- [r20260924-005536-s1] fadd s3: merging lzc28 and the `<< lz` barrel shifter into one normalize-while-count mux chain gave -1.1% area, bit-identical; Yosys won't share an LZC's internal shifts with a separate $shl.
- [r20260924-010507-s0] fadd s3→s4 exponent subtract move (register e and n separately): fmax went up +3.2%, as predicted (fmul DSP is back as the limit), but area rose +3.2% from the extra s4 compare/add logic and FFs. The combination fails the gate.
- [r20260924-010507-s1] fadd s2 alignment sticky: OR each right-shift stage's dropped bits (align27) in place of sticky_below's thermometer mask plus the separate `>>`. Bit-identical, -2.3% area at 141 MHz. Accepted.
