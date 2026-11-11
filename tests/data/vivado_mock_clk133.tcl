# boards/bcu1525/vivado/clk133.tcl on a mock of the WG top's synthesized netlist (tests/
# test_bcuwg_build.py): the four otpu_clk_sel flip-flops and seven otpu_clk_k muxes a 150 MHz build
# with CORE_KHZ1 133333 has (each output bit GND, VCC, the select or a LUT1 of it), the clock
# wizard's MMCM as bd_wg.tcl makes it and its loads, and the Vivado commands the hook takes.
#   tclsh vivado_mock_clk133.tcl CASE CLK133_TCL REPORT_DIR
# CASE: ok (the edit), report (::otpu::clk_report_only), fold (a mux's two values one), wrong (a
# value off), nouart (no UART muxes), nofrac (the MMCM not bd_wg.tcl's 150 MHz one), badload
# (CLKOUT5 loaded), noreadback (CLKOUT1_DIVIDE does not take). Prints each select's INIT and
# DONT_TOUCH and the MMCM's divides after it; exit 1 with the hook's error.
set case [lindex $argv 0]
array set C {}   ;# cell -> dict
array set P {}   ;# pin -> dict (cell dir rpn net)
array set N {}   ;# net -> list of pins
proc cell {name ref {orig ""} {parent ""}} {
  set ::C($name) [dict create REF_NAME $ref ORIG_REF_NAME [expr {$orig eq "" ? $ref : $orig}] PARENT $parent \
                    IS_PRIMITIVE [expr {$ref in {FDRE LUT1 GND VCC BUFG BUFGCE BUFGCE_DIV MMCME4_ADV}}] DONT_TOUCH 0]
}
proc pin {name dir {net ""}} {
  set c [string range $name 0 [expr {[string last / $name] - 1}]]
  set ::P($name) [dict create cell $c dir $dir rpn [string range $name [expr {[string last / $name] + 1}] end] net $net]
  if {$net ne ""} { lappend ::N($net) $name }
}
set ::nconst 0
# an otpu_clk_k instance: W bits, values v0 / v1, select net sn (driven by an otpu_clk_sel's Q)
proc mk_k {inst v0 v1 w sn {fold 0}} {
  cell $inst otpu_clk_k__p otpu_clk_k [file dirname $inst]
  pin $inst/sel IN $sn
  set inv ""
  for {set i 0} {$i < $w} {incr i} {
    set e0 [expr {($v0 >> $i) & 1}]; set e1 [expr {($v1 >> $i) & 1}]
    if {$fold} { set e1 $e0 }
    if {$e0 == $e1} {
      set g $inst/[expr {$e0 ? "VCC" : "GND"}]_$i
      cell $g [expr {$e0 ? "VCC" : "GND"}] "" $inst
      set n c$::nconst; incr ::nconst
      pin $g/G OUT $n
      pin "$inst/k\[$i\]" OUT $n
    } elseif {$e1} {
      pin "$inst/k\[$i\]" OUT $sn
    } else {
      if {$inv eq ""} {
        set inv $inst/inv; cell $inv LUT1 "" $inst
        dict set ::C($inv) INIT 2'h1
        pin $inv/I0 IN $sn; pin $inv/O OUT ${inst}_nsel
      }
      pin "$inst/k\[$i\]" OUT ${inst}_nsel
    }
  }
}
proc mk_sel {h} {
  cell $h otpu_clk_sel otpu_clk_sel [file dirname $h]
  cell $h/u_sel_ff FDRE FDRE $h
  dict set ::C($h/u_sel_ff) INIT 1'b0
  dict set ::C($h/u_sel_ff) DONT_TOUCH 1
  dict set ::C($h) DONT_TOUCH 1
  pin $h/u_sel_ff/Q OUT $h/sel
  return $h/sel
}
# the design
set s [mk_sel u_bmc/g_clk1.u_sel]
if {$case ne "nouart"} {
  mk_k u_bmc/g_clk1.u_div 1302 [expr {$case eq "wrong" ? 1158 : 1157}] 24 $s
  mk_k u_bmc/g_clk1.u_div_m1 1301 1156 24 $s [expr {$case eq "fold"}]
  mk_k u_bmc/g_clk1.u_div_m2 1300 1155 24 $s
  mk_k u_bmc/g_clk1.u_div_h1 650 577 24 $s
}
set s [mk_sel u_bmc/g_watch.u_watch/g_clk1.u_sel]
mk_k u_bmc/g_watch.u_watch/g_clk1.u_ms_m1 149999 133332 18 $s
set s [mk_sel u_gov/g_clk1.u_sel]
mk_k u_gov/g_clk1.u_ms_m1 149999 133332 18 $s
set s [mk_sel u_sys/u_board/u_ctrl/g_clk1.u_sel]
mk_k u_sys/u_board/u_ctrl/g_clk1.u_khz 150000 133333 32 $s
# the MMCM and its loads
set mm u_bd/otpu_bdw_i/clk_wiz_0/inst/mmcme4_adv_inst
cell $mm MMCME4_ADV
foreach {k v} {CLKFBOUT_MULT_F 4.000 DIVCLK_DIVIDE 1 CLKOUT0_DIVIDE_F 4.000 CLKOUT1_DIVIDE 8 CLKOUT2_DIVIDE 4} { dict set C($mm) $k $v }
if {$case eq "nofrac"} { dict set C($mm) CLKOUT0_DIVIDE_F 8.000 }
pin $mm/CLKOUT0 OUT n2x; pin $mm/CLKOUT1 OUT ncore; pin $mm/CLKOUT2 OUT naclk; pin $mm/CLKOUT5 OUT [expr {$case eq "badload" ? "n5" : ""}]
pin $mm/CLKOUT5B OUT
foreach b {u_acc_buf u_acc2x_buf} { cell $b BUFGCE_DIV; pin $b/I IN n2x; pin $b/O OUT $b/o }
cell u_ce2x_buf BUFG; pin u_ce2x_buf/I IN n2x
cell u_bd/otpu_bdw_i/clk_wiz_0/inst/clkout2_buf BUFG; pin u_bd/otpu_bdw_i/clk_wiz_0/inst/clkout2_buf/I IN ncore
if {$case eq "badload"} { cell x5 FDRE; pin x5/C IN n5 }

# ---- the Vivado commands (as much as clk133.tcl takes)
proc opt {args name} { set i [lsearch -exact $args $name]; expr {$i < 0 ? "" : [lindex $args [expr {$i + 1}]]} }
proc flag {args name} { expr {[lsearch -exact $args $name] >= 0} }
proc last {args} { lindex $args end }
proc get_cells {args} {
  set of [opt $args -of_objects]
  if {$of ne "" || [flag $args -of_objects]} {
    set r {}
    foreach p $of { if {[info exists ::P($p)]} { lappend r [dict get $::P($p) cell] } }
    return [lsort -unique $r]
  }
  if {[flag $args -hierarchical]} {
    set f [opt $args -filter]
    set r {}
    foreach c [array names ::C] {
      set o [dict get $::C($c) ORIG_REF_NAME]; set ref [dict get $::C($c) REF_NAME]
      if {[string match *otpu_clk_sel* $f] && ($o eq "otpu_clk_sel")} { lappend r $c }
      if {[string match *otpu_clk_k* $f] && ($o eq "otpu_clk_k")} { lappend r $c }
      if {[string match *MMCME4_ADV* $f] && $ref eq "MMCME4_ADV" && [string match u_bd/otpu_bdw_i/clk_wiz_0/* $c]} { lappend r $c }
    }
    return $r
  }
  set n [last {*}$args]
  if {[info exists ::C($n)]} { return $n }
  return {}
}
proc get_pins {args} {
  set f [opt $args -filter]
  set dir [expr {[string match *OUT* $f] ? "OUT" : [string match *IN* $f] ? "IN" : ""}]
  if {[flag $args -of_objects]} {
    set of [opt $args -of_objects]
    set r {}
    if {[flag $args -leaf]} {
      foreach n $of {
        if {![info exists ::N($n)]} continue
        foreach p $::N($n) {
          if {![dict get $::C([dict get $::P($p) cell]) IS_PRIMITIVE]} continue
          if {$dir ne "" && [dict get $::P($p) dir] ne $dir} continue
          lappend r $p
        }
      }
      return $r
    }
    foreach c $of {
      foreach p [array names ::P] {
        if {[dict get $::P($p) cell] eq $c && ($dir eq "" || [dict get $::P($p) dir] eq $dir)} { lappend r $p }
      }
    }
    return $r
  }
  set n [last {*}$args]
  if {[info exists ::P($n)]} { return $n }
  return {}
}
proc get_nets {args} {
  set r {}
  foreach p [opt $args -of_objects] { if {[info exists ::P($p)] && [dict get $::P($p) net] ne ""} { lappend r [dict get $::P($p) net] } }
  return [lsort -unique $r]
}
proc get_property {args} {
  set name [lindex $args end-1]; set o [lindex $args end]
  if {[info exists ::C($o)]} {
    if {![dict exists $::C($o) $name]} { error "no property $name on $o" }
    return [dict get $::C($o) $name]
  }
  if {[info exists ::P($o)]} { if {$name eq "REF_PIN_NAME"} { return [dict get $::P($o) rpn] } }
  if {[string match clk:* $o]} {
    lassign [split [string range $o 4 end] |] cn per
    if {$name eq "NAME"} { return $cn }
    if {$name eq "PERIOD"} { return $per }
  }
  error "get_property $name $o"
}
proc set_property {name v os} {
  if {$::case eq "noreadback" && $name eq "CLKOUT1_DIVIDE"} return
  foreach o $os {
    if {![info exists ::C($o)]} { error "set_property $name: no cell $o" }
    if {$name eq "INIT" && [dict get $::C($o) DONT_TOUCH]} { error "INIT on DONT_TOUCH $o" }
    dict set ::C($o) $name $v
  }
}
# clocks from the MMCM's divides (VCO 1200)
proc get_clocks {args} {
  set p [opt $args -of_objects]
  set mm $::mm
  set f0 [dict get $::C($mm) CLKOUT0_DIVIDE_F]; set f1 [dict get $::C($mm) CLKOUT1_DIVIDE]; set f2 [dict get $::C($mm) CLKOUT2_DIVIDE]
  switch -- $p {
    u_bd/otpu_bdw_i/clk_wiz_0/inst/mmcme4_adv_inst/CLKOUT0 { return [format "clk:acc_clk2x_otpu_bdw_clk_wiz_0_0|%.3f" [expr {$f0 / 1.2}]] }
    u_bd/otpu_bdw_i/clk_wiz_0/inst/mmcme4_adv_inst/CLKOUT1 { return [format "clk:core_clk_otpu_bdw_clk_wiz_0_0|%.3f" [expr {$f1 / 1.2}]] }
    u_bd/otpu_bdw_i/clk_wiz_0/inst/mmcme4_adv_inst/CLKOUT2 { return [format "clk:aclk_otpu_bdw_clk_wiz_0_0|%.3f" [expr {$f2 / 1.2}]] }
    u_acc_buf/O { return [format "clk:acc_clk|%.3f" [expr {2 * $f0 / 1.2}]] }
    u_acc2x_buf/O { return [format "clk:acc_clk2x|%.3f" [expr {$f0 / 1.2}]] }
  }
  return {}
}
proc report_clocks {args} { puts "(report_clocks $args)" }

namespace eval ::otpu {}
set ::otpu::clk_rpt_dir [file normalize [lindex $argv 2]]
if {$case eq "report"} { set ::otpu::clk_report_only 1 }
if {[catch {source [lindex $argv 1]} e]} { puts "ERROR: $e"; puts $::errorInfo; exit 1 }
foreach c [lsort [array names C]] {
  if {[dict get $C($c) REF_NAME] eq "FDRE" && [dict exists $C($c) INIT]} { puts "$c INIT [dict get $C($c) INIT] DONT_TOUCH [dict get $C($c) DONT_TOUCH]" }
}
puts "MMCM CLKOUT0 [dict get $C($mm) CLKOUT0_DIVIDE_F] CLKOUT1 [dict get $C($mm) CLKOUT1_DIVIDE]"
puts DONE
