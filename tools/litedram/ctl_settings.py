"""The production controller's LiteDRAM ControllerSettings that differ from the defaults, shared by
the core (gen_core.py) and its simulation model (gen_ldc.py) so that the co-simulation schedules
as the card does (docs/litedram.md section 11, "What is left in the controller"):

- refresh_postponing 8: the refresher lets 8 tREFI pass, then refreshes 8 times back to back (one
  wait for the banks and one row reopening per 8 refreshes instead of per refresh). DDR3 allows 8
  postponed refreshes (at most 9 tREFI between two). Decode: -0.6% cycles per token.
- read_time 256, write_time 128 (32 and 16 by default): the multiplexer switches from reads to
  pending writes after 256 cycles of reads, and back after 128 of writes, so the read/write
  turnarounds (7 cycles from read to write) come less often. Decode with DeltaNet's state
  write-back (Qwen3.5): -0.6% cycles per token.
"""
CONTROLLER = dict(refresh_postponing=8, read_time=256, write_time=128)
