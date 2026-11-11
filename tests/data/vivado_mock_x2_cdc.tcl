# A tiny two-clock netlist with one of each CDC family at each port. Execute the actual Tcl
# selectors and fail if a missing first capture stage is silently accepted or a clock group
# cuts the bounded data crossings. These are constraint-selector checks, not CDC signoff.
lassign $argv dir mode
array set refs {}; array set regs {}
foreach p {0 1} {
  set b "u_link/g_l\[$p\]."
  foreach {u ref src dst} {
    u_ctr otpu_link_sync_count g_xs g_sy0
    u_word otpu_link_sync_word hold_xs dd_sy
    u_fifo otpu_link_afifo wgray_xs wg_sy0
  } {
    set n $b$u
    set refs($n) $ref
    foreach r [list $src $dst] { foreach k {0 1} { set regs($n/${r}_reg\[$k\]) 1 } }
  }
  # otpu_link_sync_bit's first flop is in its generate scope (x2d100b's netlist), never a bare d_sy_reg
  set refs(${b}u_bit) otpu_link_sync_bit; set regs(${b}u_bit/g_2.d_sy_reg) 1
  set refs(${b}u_rst) otpu_link_sync_bit; set regs(${b}u_rst/g_n.d_sy_reg) 1
}
if {$mode eq "missing"} { unset regs(u_link/g_l\[1\].u_fifo/wg_sy0_reg\[0\]) regs(u_link/g_l\[1\].u_fifo/wg_sy0_reg\[1\]) }
proc get_cells {args} {
  set i [lsearch $args -filter]
  if {$i >= 0} {
    set f [lindex $args $i+1]
    if {[string match *STARTUPE3* $f]} { return u_link_phy/u_startup }
    regexp {ORIG_REF_NAME == (\w+)} $f -> ref
    set cs {}; foreach u [array names ::refs] { if {$::refs($u) eq $ref} { lappend cs $u } }; return $cs
  }
  set p [string map {\[ \\[ \] \\]} [lindex $args end]]
  return [lsearch -all -inline -glob [array names ::regs] $p]
}
proc get_pins {args} { return [lindex $args end] }
proc get_clocks {args} {
  set p [lindex $args end]
  if {[string match *g_q?0?* $p]} { return user0 }
  if {[string match *g_q?1?* $p]} { return user1 }
  if {[string match *init_clk* $p]} { return init100 }
  return cfg
}
proc create_clock {args} {}
set ng 0; set nd 0; set nb 0; set nf 0
proc set_clock_groups {args} {
  if {[string match *acc* $args] || [string match *core* $args]} { error "group overrides bounded buses" }
  incr ::ng
}
proc set_max_delay {args} {
  if {[lindex $args 0] ne "-datapath_only" || [lindex $args end] != 2.4} { error "bad max delay" }
  incr ::nd
}
proc set_bus_skew {args} { incr ::nb }
proc set_false_path {args} { incr ::nf }
set rc [catch {source $dir/cdc_x2.tcl} e]
if {$mode eq "missing"} {
  if {!$rc || ![string match {*missing *u_fifo/wg_sy0} $e]} { error "missing pointer accepted: $e" }
  puts "PASS missing capture rejected"
} else {
  if {$rc} { error $e }
  if {$ng != 2 || $nd != 6 || $nb != 6 || $nf != 1 || $::otpu::ok(x2_cdc) != 1} { error "incomplete constraints" }
  puts "PASS CDC selectors and precedence"
}
