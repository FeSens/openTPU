# boards/bcu1525/vivado/wg_umem.tcl on a mock of the WG top's opt netlist at two slices (tests/
# test_bcuwg_build.py): the floorplan's pblock_slice0 / pblock_slice1 (SLR0 / SLR2) and pblock_ctl,
# each slice's u_mem, its 8 bridges' core halves and a u_xs (otpu_xstg) with one pipe of each kind
# (u_ctl and a queue's u_f slice -> adapter, the queue's u_r and u_ra adapter -> slice, a count's
# g_pc, the posts' u_pp and g_ps, the idle holds, logic in and after a pipe, the queue's own and
# logic under no pipe), and the Vivado commands the move takes.
#   tclsh vivado_mock_umem.tcl CASE WG_UMEM_TCL
# CASE: ok, onestage (u_ctl of one stage), nodir (a pipe of no known direction), slr1
# (pblock_slice0 in SLR1), nomem (no slice 1 u_mem). Prints each u_xs leaf's pblock (its own or its
# nearest assigned parent's) and USER_SLL_REG, then the moved cells; exit 1 with the move's error.
set case [lindex $argv 0]
array set C {}    ;# cell -> IS_PRIMITIVE
array set PB {}   ;# pblock -> GRID_RANGES
array set IN {}   ;# cell -> its pblock (an assignment)
array set SLL {}
set PB(pblock_slice0) [expr {$case eq "slr1" ? "SLR1" : "SLR0"}]
set PB(pblock_slice1) SLR2
set PB(pblock_ctl) SLR1
proc hier {name pb} { set ::C($name) 0; if {$pb ne ""} { set ::IN($name) $pb } }
proc leaf {name} { set ::C($name) 1 }
proc pipe {xs p n {w 1}} {
  for {set i 0} {$i < $n} {incr i} {
    set k [expr {$i ? "g_next." : "g_first."}]
    for {set b 0} {$b < $w} {incr b} { leaf "$xs/$p/g_pipe.g_s\[$i\].${k}r_reg\[$b\]" }
  }
}
foreach s {0 1} {
  set pre [expr {$s ? "u_sys/u_board/g_slice\[1\]." : "u_sys/u_board/"}]
  if {!($case eq "nomem" && $s)} { hier ${pre}u_mem pblock_slice$s; leaf ${pre}u_mem/ram_reg }
  for {set j [expr {8 * $s}]} {$j < 8 * $s + 8} {incr j} { hier "u_sys/g_ch\[$j\].u_br/u_cs" pblock_slice$s }
  set xs ${pre}u_xs
  hier $xs pblock_slice$s
  # otpu_xstg's names: g_stg.u_ctl, a queue's (otpu_xstg_q) u_f / u_r, u_ra .. u_ri, g_pc[c].u_p,
  # u_pp, g_ps.u_ps and g_ps.g_psc[k].u_p (NP - 1 and 1 stages), the idle holds
  pipe $xs g_stg.u_ctl [expr {$case eq "onestage" ? 1 : 2}]
  pipe $xs g_stg.u_qa/u_f 3 2
  leaf "$xs/g_stg.u_qa/u_f/g_pipe.g_s\[0\].d_lut"
  leaf "$xs/g_stg.u_qa/u_f/o_lut"
  pipe $xs g_stg.u_qa/u_r 2
  leaf "$xs/g_stg.u_qa/u_r/x_lut"
  leaf "$xs/g_stg.u_qa/ram_reg\[0\]"
  for {set i 0} {$i < 2} {incr i} { leaf "$xs/g_stg.u_ra/g_pipe.g_s\[$i\].r_reg" }
  pipe $xs g_pc\[0\].u_p 2
  pipe $xs u_pp 2
  pipe $xs g_ps.u_ps 1
  pipe $xs g_ps.g_psc\[0\].u_p 1
  leaf $xs/g_stg.w_h_reg
  leaf "$xs/g_stg.r_h_reg\[0\]"
  leaf $xs/GND
  leaf $xs/misc_lut
  if {$case eq "nodir"} { pipe $xs g_stg.u_zz 2 }
}
# Vivado's patterns: * and ? within one level of the name; -hierarchical NAME =~: across levels
proc pat_re {p cross} {
  set r ""
  foreach ch [split $p ""] {
    switch -- $ch {
      * { append r [expr {$cross ? ".*" : {[^/]*}}] }
      ? { append r [expr {$cross ? "." : {[^/]}}] }
      default { append r [expr {[string is alnum $ch] || $ch eq "_" ? $ch : "\\$ch"}] }
    }
  }
  return "^$r\$"
}
proc get_cells {args} {
  set q [expr {[lindex $args 0] eq "-quiet"}]
  if {$q} { set args [lrange $args 1 end] }
  if {[lindex $args 0] eq "-hierarchical"} {
    if {[lindex $args 1] ne "-filter" || ![regexp {^IS_PRIMITIVE && NAME =~ (\S+)$} [lindex $args 2] - p]} {
      error "mock get_cells: $args"
    }
    set re [pat_re $p 1]
    return [lsort [lmap c [array names ::C] { expr {$::C($c) && [regexp $re $c] ? $c : [continue]} }]]
  }
  set r {}
  foreach p [lindex $args 0] {
    set re [pat_re $p 0]
    set m [lmap c [array names ::C] { expr {[regexp $re $c] ? $c : [continue]} }]
    if {![llength $m] && !$q} { error "mock get_cells: no $p" }
    set r [concat $r $m]
  }
  return [lsort -unique $r]
}
proc get_pblocks {args} {
  set n [lindex $args end]
  expr {[info exists ::PB($n)] ? $n : ""}
}
proc get_property {name o} {
  if {$name eq "GRID_RANGES" && [info exists ::PB($o)]} { return $::PB($o) }
  error "mock get_property $name $o"
}
proc remove_cells_from_pblock {pb cs} {
  foreach c $cs {
    if {![info exists ::IN($c)] || $::IN($c) ne $pb} { error "mock remove_cells_from_pblock: $c not in $pb" }
    unset ::IN($c)
  }
}
proc add_cells_to_pblock {pb cs} {
  if {![info exists ::PB($pb)]} { error "mock add_cells_to_pblock: no $pb" }
  foreach c $cs {
    if {![info exists ::C($c)]} { error "mock add_cells_to_pblock: no cell $c" }
    if {[info exists ::IN($c)]} { error "mock add_cells_to_pblock: $c already in $::IN($c)" }
    set ::IN($c) $pb
  }
}
proc set_property {name v cs} {
  if {$name ne "USER_SLL_REG"} { error "mock set_property $name" }
  foreach c $cs {
    if {![info exists ::C($c)] || !$::C($c)} { error "mock USER_SLL_REG on $c (not a leaf)" }
    set ::SLL($c) $v
  }
}
proc where {c} {
  while {1} {
    if {[info exists ::IN($c)]} { return $::IN($c) }
    set i [string last / $c]
    if {$i < 0} { return - }
    set c [string range $c 0 [expr {$i - 1}]]
  }
}
source [lindex $argv 1]
if {[catch {::otpu::wg_umem pblock_ctl mock} r]} { puts stderr $r; exit 1 }
lassign $r moved sll
foreach c [lsort [array names C]] {
  if {$C($c) && [string match *u_xs/* $c]} {
    puts "leaf $c [where $c] [expr {[info exists SLL($c)] ? "SLL" : "-"}]"
  }
}
foreach c $moved { puts "moved $c [where $c]" }
puts "sll [llength $sll]"
