# otpu_link_cdc.sv crossings, using the same datapath bounds as otpu_hbm_ch.tcl. The two
# user clocks are independent of acc_clk. Do NOT group acc_clk against them: Vivado's clock
# groups override set_max_delay even when the latter names individual synchronizer cells.
# Use per-crossing exceptions instead (AMD UG903 "Constraining Asynchronous Signals").
# Init/user and user0/user1 do have groups: no bounded data bus crosses those pairs.
namespace eval ::otpu {}
proc ::otpu::x2_need {what c} {
  if {![llength $c]} { error "x2 CDC: missing $what" }
  return $c
}
set uclks {}
foreach p {0 1} {
  set pin [get_pins -quiet "u_link_phy/g_q?$p?.u_aur/u_ubuf/O"]
  set c [get_clocks -quiet -of_objects $pin]
  if {[llength $c] != 1} { error "x2 CDC: port $p user clock '$c'" }
  lappend uclks $c
}
set iclk [get_clocks -quiet -of_objects [get_pins u_bd/init_clk]]
if {[llength $iclk] != 1} { error "x2 CDC: init_clk '$iclk'" }
set_clock_groups -asynchronous -group $iclk -group [lindex $uclks 0] -group [lindex $uclks 1]
set su [get_cells -quiet -hierarchical -filter {REF_NAME == STARTUPE3}]
if {[llength $su] != 1} { error "x2 CDC: [llength $su] STARTUPE3 owners, expected one" }
create_clock -period 15.000 -name x2_cfgmclk [get_pins -of_objects $su -filter {REF_PIN_NAME == CFGMCLK}]
set_clock_groups -asynchronous -group [get_clocks x2_cfgmclk]

# A scalar level or toggle: only its first capture flop is excepted, not the whole chain. The flop
# sits in otpu_link_sync_bit's generate scope (g_2.d_sy_reg / g_n.d_sy_reg: x2d100b's netlist, 37).
set sy {}
foreach u [get_cells -quiet -hierarchical -filter {NAME =~ u_link* && ORIG_REF_NAME == otpu_link_sync_bit}] {
  set f [get_cells -quiet $u/g_*.d_sy_reg]
  if {[llength $f] != 1} { error "x2 CDC: $u has [llength $f] first stages" }
  lappend sy $f
}
set_false_path -to [::otpu::x2_need {scalar synchronizers} $sy]

# Constrain each counter, FIFO pointer and held handshake bus separately. 2.4 ns is below the
# 402.83 MHz user clock's period. Bus skew bounds apply per bus, never to unrelated counters.
set nx 0
foreach {ref pairs} {
  otpu_link_sync_count {g_xs g_sy0}
  otpu_link_sync_word {hold_xs dd_sy}
  otpu_link_afifo {wgray_xs wg_sy0}
} {
  set us [get_cells -quiet -hierarchical -filter "NAME =~ u_link/* && (ORIG_REF_NAME == $ref || REF_NAME == $ref)"]
  ::otpu::x2_need $ref $us
  foreach u $us {
    foreach {src dst} $pairs {
      set s [::otpu::x2_need "$u/$src" [get_cells -quiet $u/${src}_reg*]]
      set d [::otpu::x2_need "$u/$dst" [get_cells -quiet $u/${dst}_reg*]]
      set_max_delay -datapath_only -from $s -to $d 2.4
      set_bus_skew -from $s -to $d 2.4
      incr nx
    }
  }
}
puts "x2 CDC: $nx Gray/pointer/held-word buses bounded at 2.4 ns; [llength $sy] scalar first stages"
incr ::otpu::ok(x2_cdc)
