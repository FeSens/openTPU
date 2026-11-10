# Late, implementation-only unmanaged Tcl. read_xdc converts errors to warnings: record every
# miss, and let bcu1525_wg_check.tcl stop opt_design unless both x2 files finished exactly once.
# The run's read_xdc -unmanaged sets no [info script] (x2d100b: "./floorplan_x2.tcl" not found):
# the tree from run_vivado.sh's OTPU_WG_ROOT.
set x2dir [expr {[info exists ::env(OTPU_WG_ROOT)] && $::env(OTPU_WG_ROOT) ne ""
                 ? "$::env(OTPU_WG_ROOT)/boards/bcu1525/x2" : [file dirname [file normalize [info script]]]}]
if {[catch {source $x2dir/floorplan_x2.tcl} e]} { lappend ::otpu::fail "x2 floorplan: $e" }
if {[catch {source $x2dir/cdc_x2.tcl} e]} { lappend ::otpu::fail "x2 CDC: $e" }
