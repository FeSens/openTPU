# Clock-domain crossing of an MXU lane's posted-beat staging (rtl/mxu/otpu_mxu.sv otpu_mxu_stage:
# the WG targets' chunk FIFO lanes in UltraRAM, MXU_FIFO_STYLE "ultra" at NAT_XPOST; otpu_mxu g_xp's
# g_u), as otpu_hbm_ch.tcl does for the bridges' FIFOs. Scoped to the module, read after the
# clocks exist (Tcl, an unmanaged constraint file):
#   read_xdc -unmanaged -ref otpu_mxu_stage otpu_mxu_stage.tcl
# in implementation, on the synthesized netlist (boards/bcu1525/vivado/create_project_wg.tcl:
# SCOPED_TO_REF, used in implementation only).
#
# The module's clocks: wclk (the memory's AXI clock, aclk: the channel's posted quarters) and rclk
# (the core clock). Its one crossing: the four quarter RAMs (g_q[q].u_q, LUT RAM written in wclk)
# read asynchronously into their read registers (rclk) at an entry the lane's copy takes two rclk
# cycles at the earliest after the beat's count crossed (the bridge's p_cntg into the adapter's
# synchronizer: bcu1525_wg_top.tcl), and not rewritten before the copy took it (32 entries a lane).
# So the RAMs' outputs into the registers in one rclk period, set_max_delay -datapath_only. The
# top level must not make the two clocks asynchronous with set_clock_groups or set_false_path
# (either overrides set_max_delay and leaves the crossing untimed).

# the RAMs (g_q[q].u_q/g_lut.mem_reg*: RAM32M16 macros, their RAMA .. RAMH INTERNAL cells, so
# not PRIMITIVE_LEVEL LEAF: their pins IS_LEAF, as otpu_hbm_ch.tcl's FIFO RAMs) and the read
# registers (g_q[q].u_q/g_lut.rd_reg[*], FDRE: PRIMITIVE_GROUP REGISTER), the module's only
# cells of those names; the four quarters counted one by one (g_q?q?.u_q: get_cells takes
# brackets as a bus index), collections only (no list of names: Vivado cuts one at 500).
# A constraint file's failed command (an error included) is only a CRITICAL WARNING, so the stop
# is otpu_impl.tcl's: it counts the instances constrained here (::otpu_mxu_stage_ok) against
# the build's otpu_mxu_stage cells and errors on a shortfall.
set sg [get_cells -quiet -hierarchical -filter {NAME =~ *mem_reg*}]
set sg_c [get_pins -quiet -filter {IS_LEAF && REF_PIN_NAME == CLK} -of_objects $sg]
set sr [get_cells -quiet -hierarchical -filter {NAME =~ *rd_reg* && IS_SEQUENTIAL}]
set c_w [get_clocks -quiet -of_objects [lindex $sg_c 0]]
set c_r [get_clocks -quiet -of_objects [get_pins -quiet -filter {REF_PIN_NAME == C} -of_objects [lindex $sr 0]]]
set n_q 0
for {set q 0} {$q < 4} {incr q} {
  set g [get_cells -quiet -hierarchical -filter "NAME =~ *g_q?$q?.u_q/*mem_reg*"]
  set r [get_cells -quiet -hierarchical -filter "NAME =~ *g_q?$q?.u_q/*rd_reg* && IS_SEQUENTIAL"]
  if {[llength [get_pins -quiet -filter {IS_LEAF && REF_PIN_NAME == CLK} -of_objects $g]] &&
      [llength $r]} { incr n_q }
}
set m "[current_instance .]: $n_q quarters with RAMs and read registers (4 expected),\
  [llength $sg_c] RAM clock pins, [llength $sr] read registers, clocks '$c_w' -> '$c_r'"
if {$n_q == 4 && [llength $c_w] == 1 && [llength $c_r] == 1} {
  set t_r [get_property -quiet -min PERIOD $c_r]
  set sr_d [get_pins -quiet -filter {REF_PIN_NAME == D} -of_objects $sr]
  set_max_delay -datapath_only -from $c_w -through [get_pins -quiet -filter {DIRECTION == OUT && IS_LEAF} -of_objects $sg] -to $sr_d $t_r
  puts "otpu_mxu_stage.tcl: $m, set_max_delay -datapath_only $t_r ns"
  incr ::otpu_mxu_stage_ok
  # ---------------------------------------------------------------- report_cdc waiver
  create_waiver -scoped -type CDC -id {CDC-26} -user "otpu_mxu_stage" -description "otpu_mxu_stage: a lane's posted-beat staging (distributed RAM, written in aclk) into its core-clock read registers; an entry read only once its beat's count crossed, not rewritten before the copy took it" -from $sg_c -to $sr_d
} else {
  puts "CRITICAL WARNING: \[otpu_mxu_stage.tcl\] $m (one each expected): the staging's crossing unconstrained"
}
