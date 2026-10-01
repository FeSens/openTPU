# Clock-domain crossings of otpu_xmon (rtl/boards/ypcb-00338/otpu_xmon.sv: the debug build's DMA
# monitors, XMON=1). Scoped to the module, read late as otpu_mem_ch.tcl is (create_project.tcl).
# Its clocks: clk (the core clock: the register side), xclk (XDMA's), uclk (channel 0's
# controller). The crossings:
#   - SNAP / CLEAR requests (toggles, clk) into xclk and uclk, their acknowledgements (toggles)
#     back, and the sticky flags (bit by bit) into clk: ASYNC_REG pairs, one source period;
#   - the shadows (xclk / uclk), read in clk through otpu_ctrl's read multiplexer: loaded only by a
#     SNAP, read by the host after the SNAP's acknowledgement has crossed (microseconds later), so
#     not timed.
set c_clk [get_clocks -quiet -of_objects [get_ports clk]]
set c_ucl [get_clocks -quiet -of_objects [get_ports uclk]]
set c_xcl [get_clocks -quiet -of_objects [get_ports xclk]]
set t_clk [get_property -quiet -min PERIOD $c_clk]
set t_ucl [get_property -quiet -min PERIOD $c_ucl]
set t_xcl [get_property -quiet -min PERIOD $c_xcl]

# a debug build: a name the netlist lacks warns rather than stopping the run
proc xmon_md {from names t} {
  set c [get_cells -quiet $names]
  if {[llength $c] != [llength $names]} { puts "CRITICAL WARNING: otpu_xmon.tcl: [llength $c] of $names" }
  if {[llength $c]} { set_max_delay -datapath_only -from $from -to $c $t }
}
xmon_md $c_clk {x_sn_reg[0] x_cl_reg[0]} $t_clk
xmon_md $c_clk {u_sn_reg[0] u_cl_reg[0]} $t_clk
xmon_md $c_xcl {c_xa_reg[0]} $t_xcl
xmon_md $c_ucl {c_ua_reg[0]} $t_ucl
xmon_md $c_xcl [lmap i {0 1 2 3 4 5 6 7} {string cat c_xf1_reg\[$i\]}] $t_xcl
xmon_md $c_ucl [lmap i {0 1 2} {string cat c_nf1_reg\[$i\]}] $t_ucl

set sh [get_cells -quiet {sx_reg* sn_reg*}]
puts "otpu_xmon.tcl: [llength $sh] shadow registers"
if {[llength $sh]} { set_false_path -from $sh }
