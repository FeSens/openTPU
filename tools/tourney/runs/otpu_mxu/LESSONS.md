# Lessons: otpu_mxu

- [r20260924-012715-s0] Yosys shregmap ignores shreg_extract/keep, so otpu_delay's last stage (ma.v) became an SRL (1.47ns clk-Q); registering the decoded one-hot combine selects as FFs got +3% fmax (106→109MHz).
- [r20260924-012715-s1] Packing 2 int8 products per DSP48E1 in g_tree works: A={a0,16'b0}+a1 times the shared w0, then split pp[15:0] and pp[31:16]+pp[15]. That's -128 DSPs and -12.6% area, bit-exact, fmax unchanged.
- [r20260924-015733-s0] Registering the combine operands into `pset` at the row's last fin block took the operand/zero mux off the u_c01/u_c23 inputs (bit-identical, u_lv N=2LA+1): 106→109 MHz, area −0.4%.
- [r20260924-015733-s1] Sizing the drain RMW to NL=min(LANES,MCOLS) removed dead fmadd, u_rw and RMAX lanes that yosys kept after ABC9 (keep'd delay FFs): -1.5% area, fmax unchanged. Trim dead lanes in the RTL itself.
- [r20260924-022514-s0] Ending the SRL delay lines (as_d, ws6, mt, ma) in reset FDREs to cut SRL clock-to-out gave only +0.8% fmax at +0.2% area; the critical path moves elsewhere, so SRL Q isn't the limiter.
- [r20260924-022514-s1] Drain lanes trimmed to NL=min(LANES,MCOLS) in the RTL (rmw_t, g_rmw fmadds, RMAX loops, write-back): area -1.5%, fmax unchanged. Yosys can't range-prove ncnt, so dead lanes must be cut in source.
- [r20260924-030158-s0] mxu: making g_tree.pp/pr packed (not unpacked→memory) let yosys pack the int8 product regs into DSP MREG. That, plus ending the fmul operand SRLs in flops, was accepted (area -7.4% at 122 MHz).
- [r20260924-030158-s1] Narrowing the adder tree to exact widths (s2=18, s3=20, s4=16+clog2(D)) and adding a round-free i2f cut area by 2.2% with fmax unchanged at 109 MHz; yosys can't narrow widths itself, so size them in the RTL.
- [r20260924-031957-s0] Registering the ACC RMAX candidates (rz) to move the SRL-launched u_rw off the mx[] compare gave only +0.9% fmax for +0.4% area. The next-worst cone is right behind it, so a one-stage retime doesn't pay off.
- [r20260924-031957-s1] The isum_4 pair sums (p0+p2, p1+p3) finish in consecutive advances, so one fadd per column can do both (hold one result an advance). Dropping u_c23 + pset/pmask gave -4.7% area at 122 MHz.
- [r20260924-033026-s0] The drain fmadd's ASCALE factor mux (LUT3+LUT4) sat in front of the DSP B port. Registering it one stage early, next to xo on the r0->r1 edge, gave +3.8% fmax (122->127 MHz) with no area cost.
- [r20260924-033026-s1] ASCALE load/RMAX write-out cut to NL=min(LANES,MCOLS) lanes, and step offsets fixed at 0 when MCOLS<=LANES (ISA gives M<=MCOLS): -2.9% area. Yosys can't prove dead TMEM lanes, so cut them in source.
- [r20260924-034118-s0] RMAX compare per (lane, reachable column) with constant index (no col mux in front of fp_gt): fmax +2.6%, area -0.1%, no_gain; the col mux isn't the whole 4.7 ns cone, the ftz/fkey+carry+CE fan-out is.
- [r20260924-034118-s1] ACCEPTED (-13.1% area @123MHz): DSP48 post-adder (M+C, odd PK=32513 bias so fields extract exactly, no borrow) replaced first g_tree fabric level; use idle DSP P/ALU before adding LUT adders.
- [r20260924-035327-s0] RMAX stored as fkey + flopped one-hot rw column hits (removes SRL launch, col decode, stored-side ftz/fkey): only +1.4% fmax, -0.4% area. The fkey(candidate) compare, carry chain and CE fan-out still set the limit.
- [r20260924-035327-s1] Narrowing s4 and i2f to exact SW=16+clog2(D) bits with a round-free i2f is bit-exact and cut area 1.5% at 123 MHz; yosys keeps 32-bit sums, so size the widths in the RTL.
- [r20260924-040207-s0] Drain+RMAX cones were within 1.5%: fixing both together (MW-bit dj/ncnt so carry chains become LUTs, plus RMAX ordered keys) gave +8.1% where either alone was capped at ~1-3%.
- [r20260924-040207-s1] Trimming the drain/ASCALE/RMAX TMEM lane loops from LANES to NL, pinning al_i/mx_i to 0 and dropping dead u_rw bits (nv, col high bits) saved 3.9% area at 125 MHz; Yosys can't bound register-derived counts.
