# Lessons: otpu_axi_dram

- [r20260924-020918-s0] Counting wr_n from the take strobes (b_take&&b_we&&mask≠0, a_take&&a_we, sw_take) instead of q*_push gave 139→172 MHz at the same area. The a_reuse term was a false dependency that ABC can't remove.
- [r20260924-020918-s1] Area −7.3% @145 MHz: dropped 512b a_last reg; reuse now reads the popped beat from the A LUT-RAM FIFO, doubled to 2*AD (the RAM32M half was free). Wide regs duplicating RAM data are cheap wins.
- [r20260924-021537-s0] Registering the take strobes (wacc_q) before the wr_n popcount+16b adder, with wr_idle = wr_n==0 && wacc_q==0, took 172→212 MHz at +0.1% area; wr_idle stays exact because B always comes after the write.
