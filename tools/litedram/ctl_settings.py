"""The production controller's LiteDRAM ControllerSettings that differ from the defaults, shared by
the core (gen_core.py) and its simulation model (gen_ldc.py) so that the co-simulation schedules
as the card does (docs/litedram.md section 11, "What is left in the controller"):

- refresh_postponing 2: the refresher lets 2 tREFI pass, then refreshes twice back to back (one
  wait for the banks and one row reopening per 2 refreshes instead of per refresh). DDR3 allows
  8 postponed refreshes (at most 9 tREFI between two). With APF 32, decode takes 0.2-0.4% fewer
  cycles per token than without postponing. 4 is the same; 8 too on decode, but its bursts block
  a channel for about 210 cycles, which an MXU-bound run does not hide (perf_qwen's 2-layer fp4
  run: +1.1%, against +0.02% with 2).
- read_time 256, write_time 128 (32 and 16 by default): the multiplexer switches from reads to
  pending writes after 256 cycles of reads, and back after 128 of writes, so the read/write
  turnarounds (7 cycles from read to write) come less often. Decode with DeltaNet's state
  write-back (Qwen3.5): -0.6% cycles per token.
"""
CONTROLLER = dict(refresh_postponing=2, read_time=256, write_time=128)

# fastmux.py's options for LiteDRAM's multiplexer (docs/litedram.md section 11, "The chooser and
# the turnarounds"; all None / False: LiteDRAM's own):
# - FASTMUX, the core's: the read-to-write turnaround 3 cycles (rtw), the write-to-read one held
#   in READ (direct_wtr), and the choosers' grant in the cycle a request is valid (same_cycle).
#   Decode -0.74 / -0.35 / -0.86% cycles (Qwen3 / LFM2 / Qwen3.5, 4-bit). The grant is a priority
#   encoder in sys's paths: +0.58 ns out of context at 133.33 MHz, about +0.4 in a full build.
# - FASTMUX_SAFE, the fallback if a build misses sys's timing on the choosers (choose_cmd /
#   choose_req paths): the turnarounds only, no new combinational path; -0.42 / -0.17 / -0.52%.
#   To switch: MULTIPLEXER = FASTMUX_SAFE, then check_core.sh --update and gen_ldc.py (the core
#   and its model together; test_rtl's BIST figures follow MULTIPLEXER).
FASTMUX = dict(rtw=3, same_cycle=True, direct_wtr=True)
FASTMUX_SAFE = dict(rtw=3, same_cycle=False, direct_wtr=True)
MULTIPLEXER = FASTMUX
