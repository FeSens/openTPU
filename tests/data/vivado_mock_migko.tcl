# boards/bcu1525/vivado/wg_migko.tcl on a mock of the WG top's opt netlist (tests/test_bcuwg_build.py):
# each DIMM's ports on the board's banks (C0 rows 1-3 and C2 7-9 on IOB column X0, C1 6-8 and C3
# 11-13 on X1, two ports a row), its MIG's u_ddr4_mem_intfc cells (LUTs, flip-flops, a carry and a
# block RAM), and the Vivado commands the keep-out takes; then a mock placement for migko_report.
#   tclsh vivado_mock_migko.tcl CASE WG_MIGKO_TCL SPEC W
# CASE: ok, pkg (no port LOC: the package pins' sites), twocol (a C2 port on X1), twoslr (a C0 port
# on row 5), crx (a C1 site's clock region on another row), nocells (C3's MIG without cells), inpb
# (C1's cells in a pblock), exists (pblock_migko0 made before). Prints each keep-out's pblock, its
# ranges, IS_SOFT, EXCLUDE_PLACEMENT and cell count, the returned list and migko_report's lines;
# exit 1 with the keep-out's error.
lassign $argv case tcl spec w
array set LOC {}  ;# port or cell -> site
array set CR {}   ;# site -> clock region
array set REF {}  ;# cell -> REF_NAME
array set GRP {}  ;# cell -> PRIMITIVE_GROUP
array set PB {}   ;# pblock -> GRID_RANGES
array set PP {}   ;# pblock,property -> value
array set IN {}   ;# cell -> pblock
set PORTS {}
foreach {d c rows} {0 0 {1 2 3} 1 1 {6 7 8} 2 0 {7 8 9} 3 1 {11 12 13}} {
  set i 0
  foreach r $rows {
    foreach k {0 1} {
      set p "ddr4_c${d}_dq\[$i\]"
      set cc $c
      if {$case eq "twocol" && $d == 2 && $i == 5} { set cc 1 }
      set s IOB_X${cc}Y[expr {52 * $r + 3 * $k}]
      lappend PORTS $p
      set LOC($p) $s
      set CR($s) X[expr {2 + 2 * $cc}]Y$r
      if {$case eq "crx" && $d == 1 && $i == 2} { set CR($s) X4Y[expr {$r + 1}] }
      incr i
    }
  }
  if {$case eq "twoslr" && $d == 0} {
    set p "ddr4_c0_a\[0\]"
    lappend PORTS $p
    set LOC($p) IOB_X0Y261
    set CR(IOB_X0Y261) X2Y5
  }
  if {!($case eq "nocells" && $d == 3)} {
    set mi u_bd/otpu_bdw_i/ddr4_$d/inst/u_ddr4_mem_intfc
    foreach {n ref grp} {u_ddr_cal_top/lut_0 LUT6 CLB u_ddr_cal_top/lut_1 LUT3 CLB u_mig_ddr4_phy/lut_2 LUT2 CLB
                         u_ddr_cal_top/r_reg_0 FDRE REGISTER u_ddr_cal_top/r_reg_1 FDRE REGISTER
                         u_mig_ddr4_phy/r_reg_2 FDSE REGISTER u_ddr_cal_top/c_0 CARRY8 CLB
                         u_ddr_cal_top/ram_reg RAMB36E2 BLOCKRAM} {
      set REF($mi/$n) $ref
      set GRP($mi/$n) $grp
    }
  }
}
set REF(u_sys/u_board/u_slice/u_vpu/x_lut) LUT6
set GRP(u_sys/u_board/u_slice/u_vpu/x_lut) CLB
if {$case eq "exists"} { set PB(pblock_migko0) "" }
if {$case eq "pkg"} { foreach p $PORTS { set PKG(PKG_$p) $LOC($p); unset LOC($p) } }
proc mock_re {p} {
  set r ""
  foreach ch [split $p ""] {
    switch -- $ch {
      * { append r ".*" }
      ? { append r "." }
      default { append r [expr {[string is alnum $ch] || $ch eq "_" ? $ch : "\\$ch"}] }
    }
  }
  return "^$r\$"
}
proc get_ports {q p} {
  set re [mock_re $p]
  return [lmap x $::PORTS { expr {[regexp $re $x] ? $x : [continue]} }]
}
proc get_package_pins {q o p} { return PKG_$p }
proc range_sites {rg} {
  regexp {^SLICE_X(\d+)Y(\d+):SLICE_X(\d+)Y(\d+)$} $rg - xa ya xb yb
  set s {}
  for {set x $xa} {$x <= $xb} {incr x} { for {set y $ya} {$y <= $yb} {incr y} { lappend s SLICE_X${x}Y$y } }
  return $s
}
proc get_sites {args} {
  if {[lindex $args 0] ne "-quiet"} { error "mock get_sites: $args" }
  if {[lindex $args 1] eq "-of_objects"} {
    set o [lindex $args 2]
    if {[info exists ::PKG($o)]} { return $::PKG($o) }
    if {[info exists ::PB($o)]} { return [range_sites $::PB($o)] }
    return ""
  }
  return [lindex $args 1]
}
proc get_property {args} {
  if {[lindex $args 0] eq "-quiet"} { set args [lrange $args 1 end] }
  lassign $args k o
  switch -- $k {
    LOC { return [lmap x $o { expr {[info exists ::LOC($x)] ? $::LOC($x) : ""} }] }
    CLOCK_REGION { return [expr {[info exists ::CR($o)] ? $::CR($o) : ""}] }
    REF_NAME { return [lmap x $o { set ::REF($x) }] }
    GRID_RANGES { return $::PB($o) }
  }
  error "mock get_property $args"
}
proc get_cells {args} {
  if {[lindex $args 0] ne "-quiet"} { error "mock get_cells: $args" }
  if {[lindex $args 1] eq "-hierarchical"} {
    if {[lindex $args 2] ne "-filter" || ![regexp {^IS_PRIMITIVE && NAME =~ (\S+) && \(PRIMITIVE_GROUP == CLB \|\| PRIMITIVE_GROUP == REGISTER\)$} [lindex $args 3] - p]} {
      error "mock get_cells: $args"
    }
    set re [mock_re $p]
    return [lsort [lmap c [array names ::REF] { expr {[regexp $re $c] && $::GRP($c) in {CLB REGISTER} ? $c : [continue]} }]]
  }
  if {[lindex $args 1] ne "-of_objects"} { error "mock get_cells: $args" }
  set o [lindex $args 2]
  if {[llength $o] == 1 && [info exists ::PB($o)]} {
    return [lsort [lmap c [array names ::IN] { expr {$::IN($c) eq $o ? $c : [continue]} }]]
  }
  set at [dict create]
  foreach s $o { dict set at $s 1 }
  return [lsort [lmap c [array names ::REF] { expr {[info exists ::LOC($c)] && [dict exists $at $::LOC($c)] ? $c : [continue]} }]]
}
proc get_pblocks {args} {
  if {[lindex $args 0] ne "-quiet"} { error "mock get_pblocks: $args" }
  if {[lindex $args 1] eq "-of_objects"} {
    if {$::case eq "inpb" && [string match *ddr4_1/* [lindex [lindex $args 2] 0]]} { return pblock_slice0 }
    return ""
  }
  set re [mock_re [lindex $args 1]]
  return [lmap x [array names ::PB] { expr {[regexp $re $x] ? $x : [continue]} }]
}
proc create_pblock {n} { set ::PB($n) ""; return $n }
proc resize_pblock {pb a rg} {
  if {$a ne "-add" || ![info exists ::PB($pb)]} { error "mock resize_pblock $pb $a $rg" }
  set ::PB($pb) $rg
}
proc set_property {k v pb} {
  if {$k ni {IS_SOFT EXCLUDE_PLACEMENT} || ![info exists ::PB($pb)]} { error "mock set_property $k $v $pb" }
  set ::PP($pb,$k) $v
}
proc add_cells_to_pblock {pb cs} {
  foreach c $cs {
    if {[info exists ::IN($c)]} { error "mock add_cells_to_pblock: $c already in $::IN($c)" }
    set ::IN($c) $pb
  }
}
source $tcl
if {[catch {::otpu::wg_migko $spec $w mock} r]} { puts stderr $r; exit 1 }
foreach pb [lsort [array names PB]] {
  set n [llength [lmap c [array names IN] { expr {$IN($c) eq $pb ? $c : [continue]} }]]
  puts "pb $pb $PB($pb) soft $PP($pb,IS_SOFT) exclude $PP($pb,EXCLUDE_PLACEMENT) cells $n"
}
puts "made $r"
# a placement: C0's LUTs and a flip-flop in its strip, the rest of its cells outside, a VPU LUT in it
set mi0 u_bd/otpu_bdw_i/ddr4_0/inst/u_ddr4_mem_intfc
foreach c [array names IN] { set LOC($c) SLICE_X10Y[expr {[string length $c] % 50}] }
foreach c {u_ddr_cal_top/lut_0 u_ddr_cal_top/lut_1 u_mig_ddr4_phy/lut_2 u_ddr_cal_top/r_reg_0} {
  if {[info exists IN($mi0/$c)]} { set LOC($mi0/$c) SLICE_X56Y100 }
}
set LOC(u_sys/u_board/u_slice/u_vpu/x_lut) SLICE_X57Y101
foreach l [::otpu::migko_report] { puts "report $l" }
