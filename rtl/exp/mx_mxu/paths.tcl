# EXPERIMENTAL (docs/mx_mxu.md): the setup slack of the MX datapath's new paths on a routed
# checkpoint of mx_top (ooc.tcl's routed.dcp): the shift into the column ends' B ports, the
# accumulating DSP (its OPMODE register and P), the lane's i2f48, PAIR's 48-bit add and the
# result FIFO's write. Vivado on the build host:
#   vivado -mode batch -nojournal -source rtl/exp/mx_mxu/paths.tcl -tclargs <routed.dcp>
open_checkpoint [lindex $argv 0]
proc worst {tag pins} {
  if {![llength $pins]} { puts "OTPU_GRP $tag none"; return }
  set q [lindex [get_timing_paths -quiet -to $pins -setup -max_paths 1 -nworst 1] 0]
  if {[llength $q]} {
    puts "OTPU_GRP $tag setup [get_property SLACK $q] levels [get_property LOGIC_LEVELS $q] [get_property STARTPOINT_PIN $q] -> [get_property ENDPOINT_PIN $q]"
  } else {
    puts "OTPU_GRP $tag unconstrained"
  }
}
set ce [get_cells -hierarchical -filter {NAME =~ *u_ce*/g_d*u_dsp && REF_NAME =~ DSP48*}]
puts "OTPU_CE_DSPS [llength $ce]"
worst shift_to_B [get_pins -quiet -of_objects $ce -filter {REF_PIN_NAME =~ B[*]}]
worst chain_end_to_AD [get_pins -quiet -of_objects $ce -filter {REF_PIN_NAME =~ A[*] || REF_PIN_NAME =~ D[*]}]
set acc [get_cells -hierarchical -filter {NAME =~ *u_ce*/u_acc && REF_NAME =~ DSP48*}]
puts "OTPU_ACC_DSPS [llength $acc]"
worst acc_opmode [get_pins -quiet -of_objects $acc -filter {REF_PIN_NAME =~ OPMODE[*]}]
worst acc_out [get_pins -quiet -hierarchical -filter {NAME =~ *u_col/q0_reg*/D || NAME =~ *u_col/q1_reg*/D}]
worst lane_i2f48 [get_pins -quiet -hierarchical -filter {NAME =~ *g_lane*u_l/fv_reg*/D || NAME =~ *g_lane*u_l/lz2_reg*/D}]
worst pair_add [get_pins -quiet -hierarchical -filter {NAME =~ rowp_reg*/D}]
worst fifo_write [get_pins -quiet -hierarchical -filter {NAME =~ *g_rfm*/WE || NAME =~ *g_rfm*/WADR*}]
puts "OTPU_DONE"
