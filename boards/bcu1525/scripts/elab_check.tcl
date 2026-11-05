# Vivado RTL elaboration of the BCU1525 top on any installed part (no UltraScale+ part or license
# needed: synth_design -rtl does not map), with the lint stand-ins for the block design wrapper,
# the DDR4 core and DSP48E2 (boards/bcu1525/sim), then a check that every cell the floorplan
# (constraints/bcu1525_floorplan.tcl) and the crossing constraints (bcu1525_top.tcl) name exists
# in Vivado's hierarchy, with Vivado's names for the generate blocks.
#   vivado -mode batch -source elab_check.tcl -tclargs PART
set here [file normalize [file dirname [info script]]]
set root [file normalize $here/../../..]
set part [expr {[llength $argv] ? [lindex $argv 0] : "xc7k480tffg1156-2"}]
set files [split [string trim [exec env PYTHONPATH=$root python3 -m opentpu.target files bcu1525]] "\n"]
foreach f $files { read_verilog -sv $root/$f }
foreach f {otpu_bd_wrapper_stub.sv otpu_ld4_stub.sv unisim_dsp48e2_stub.sv} {
  read_verilog -sv $root/boards/bcu1525/sim/$f
}
synth_design -rtl -top otpu_fpga_top_bcu -part $part -verilog_define SYNTHESIS
set bad 0
foreach c {u_sys/u_board/u_slice u_sys/u_board/g_mem.u_mem {u_sys/g_ch[0].u_ch} {u_sys/g_ch[1].g_sll.u_tx}
           u_ld0 {g_csr[0].u_iso} {u_sys/u_board/g_slice[1].u_slice} {u_sys/u_board/g_slice[1].g_mem.u_mem}
           {u_sys/g_ch[2].u_ch} {u_sys/g_ch[3].g_sll.u_tx} u_ld3 {g_csr[3].u_iso} u_bd
           u_sys/u_board/u_ctrl u_sys/u_board/g_coll.u_coll u_sys/u_board/g_trace.u_trace u_sys/u_board/u_iso
           {u_sys/g_ch[1].u_ch} {u_sys/g_ch[1].g_sll.u_rx} {u_sys/g_ch[3].u_ch}
           {u_sys/g_ch[3].g_sll.u_rx} u_ld1 u_ld2 {g_csr[1].u_iso} {g_csr[2].u_iso}
           {g_bufg[1].u_bufg} u_sys/u_board/g_xs.u_chub {u_sys/u_board/g_xs.g_x[0].g_pf.u_pf_tx}
           {u_sys/u_board/g_xs.g_x[0].g_pf.u_pf_rx}} {
  if {[llength [get_cells -quiet $c]] != 1} { puts "ELAB-CHECK missing cell $c"; incr bad }
}
# the board's own crossing stages and collective ports (otpu_board g_xs), per slice
foreach s {0 1} {
  foreach u {u_dn_tx u_dn_rx u_up_tx u_up_rx u_cport} {
    set c "u_sys/u_board/g_xs.g_x\[$s\].$u"
    if {[llength [get_cells -quiet $c]] != 1} { puts "ELAB-CHECK missing cell $c"; incr bad }
  }
  foreach u {u_dn_tx u_dn_rx u_up_tx u_up_rx} {
    if {[llength [get_cells -quiet "u_sys/u_board/g_xs.g_x\[$s\].$u/r_reg*"]] == 0} {
      puts "ELAB-CHECK missing register g_x\[$s\].$u/r_reg"; incr bad
    }
  }
}
foreach r {lrs1_reg ld_crs1_reg {u_sys/u_board/cal_s1_reg}
           {u_sys/u_board/ded_s1_reg} {u_sys/g_ch[1].g_sll.u_rx/s_rdata_reg}
           {u_sys/g_ch[1].g_sll.u_tx/u_wd/m_data_reg}} {
  if {[llength [get_cells -quiet $r*]] == 0} { puts "ELAB-CHECK missing register $r"; incr bad }
}
foreach p {{u_sys/g_ch[0].u_ch/clk} {u_sys/g_ch[0].u_ch/xclk} {u_sys/g_ch[3].u_ch/uclk}} {
  if {[llength [get_pins -quiet $p]] != 1} { puts "ELAB-CHECK missing pin $p"; incr bad }
}
# the TMEM banks tmem_uram.xdc marks (8 banks x the copies, per slice)
set nb [llength [get_cells -quiet -hierarchical -filter {NAME =~ *u_tmem/g_port*.g_bank*.mem_reg*}]]
puts "ELAB-CHECK TMEM bank memories: $nb"
if {$nb == 0} { incr bad }
puts "ELAB-CHECK done: $bad missing"
