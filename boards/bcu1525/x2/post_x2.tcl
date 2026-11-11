# Executed with the routed netlist open, before write_bitstream. Keep a separate x2 report so
# one grep prints the image's identity, setup/hold gate, GT sites and actual SLR placement.
set xf [open $out/reports/x2_summary.txt w]
proc x2_report {s} { puts $::xf $s; puts $s }
x2_report "x2 dev: S=2 NCARDS=2 PLATFORM=7 (wg_params.tcl records ARCH_HASH / BUILD_ID)"
# OTPU_X2_WNS_MIN (ns, -0.1 .. 0, default 0): a functional dev image's setup allowance, as the S=2 dev
# image s2syn (aclk -0.238 ns) was carded; x2d100b's aclk closed at -0.042 under production's
# directives. Hold has none.
set wns_min [expr {[info exists ::env(OTPU_X2_WNS_MIN)] && $::env(OTPU_X2_WNS_MIN) ne "" ? $::env(OTPU_X2_WNS_MIN) : 0}]
if {![string is double -strict $wns_min] || $wns_min > 0 || $wns_min < -0.1} { error "OTPU_X2_WNS_MIN '$wns_min': -0.1 .. 0" }
x2_report "timing WNS $wns ns WHS $whs ns (WNS >= $wns_min, WHS >= 0)"
set bad {}
if {![string is double -strict $wns] || ![string is double -strict $whs] || $wns < $wns_min || $whs < 0} {
  lappend bad "setup/hold failed: WNS '$wns' (>= $wns_min), WHS '$whs'"
}
# Pulse width, the MIGs' TX bitslice Min Skew checks among them (x2c150's MIG 3 failed two of its D inputs'
# under a clean setup and hold): WPWS >= 0, no allowance
set tl [split [report_timing_summary -no_detailed_paths -return_string] \n]
set i [lsearch -regexp $tl {WNS\(ns\).*WPWS\(ns\)}]
set wpws [expr {$i < 0 ? "" : [lindex [regexp -all -inline {\S+} [lindex $tl [expr {$i + 2}]]] 8]}]
x2_report "pulse width WPWS $wpws ns (>= 0)"
if {![string is double -strict $wpws] || $wpws < 0} { lappend bad "pulse width failed: WPWS '$wpws'" }
foreach {p common lo hi} {0 GTYE4_COMMON_X1Y12 48 51 1 GTYE4_COMMON_X1Y11 44 47} {
  set cs [get_cells -quiet -hierarchical -filter "NAME =~ u_link_phy/g_q?$p?.* && REF_NAME == GTYE4_COMMON"]
  if {[llength $cs] != 1 || [get_property LOC $cs] ne $common} { lappend bad "port $p common '$cs' at wrong site" }
  set ch [get_cells -quiet -hierarchical -filter "NAME =~ u_link_phy/g_q?$p?.* && REF_NAME == GTYE4_CHANNEL"]
  set want {}; for {set y $lo} {$y <= $hi} {incr y} { lappend want GTYE4_CHANNEL_X1Y$y }
  set sites {}; foreach c $ch { lappend sites [get_property LOC $c]; x2_report "GT [get_property LOC $c] $c" }
  if {[lsort $sites] ne [lsort $want]} { lappend bad "port $p GT sites '$sites', expected '$want'" }
  x2_report "common [get_property LOC $cs] $cs"
}
source [file dirname [info script]]/refclk_check.tcl
lassign [refclk_check {qsfp1_mgt_refclk_1_p} \
  {GTYE4_COMMON_X1Y11 qsfp0_mgt_refclk_1_p GTYE4_COMMON_X1Y12 qsfp0_mgt_refclk_1_p}] lines errs
foreach l $lines { x2_report $l }
set bad [concat $bad $errs]
# Check the actual placed leaves, including the adapter's two hub-side stages, not just ranges.
foreach {pat slr} {
  u_link/* 2
  u_link_phy/* 2
  u_sys/u_board/g_xs.u_chub/* 1
  u_sys/u_board/g_xs.g_link.g_l?*?.u_lx/u_dn_tx/* 1
  u_sys/u_board/g_xs.g_link.g_l?*?.u_lx/u_up_rx/* 1
} {
  set cells [get_cells -quiet -hierarchical -filter "NAME =~ $pat && IS_PRIMITIVE == 1"]
  set at [lsort -unique [get_slrs -quiet -of_objects $cells]]
  x2_report "SLR $pat: $at ([llength $cells] leaves, wanted SLR$slr)"
  if {![llength $cells] || $at ne "SLR$slr"} { lappend bad "$pat placed in '$at', expected SLR$slr" }
}
foreach p [get_ports] {
  if {[get_property PACKAGE_PIN $p] in {AT22 AT20 AU22 AR21 AR22 AU20}} { lappend bad "SI5335 port $p" }
}
report_bus_skew -file $out/reports/x2_bus_skew.rpt
report_exceptions -coverage -file $out/reports/x2_exceptions.rpt
report_clock_utilization -file $out/reports/x2_clock_util.rpt
foreach p {0 1} {
  set a "u_sys/u_board/g_xs.g_link.g_l?$p?.u_lx"
  foreach {src dst dir} {u_dn_tx u_dn_rx down u_up_tx u_up_rx up} {
    set s [get_cells -quiet $a/$src]
    set d [get_cells -quiet $a/$dst]
    if {![llength $s] || ![llength $d]} {
      lappend bad "port $p $dir XRS stages missing ($s -> $d)"
    } else {
      report_timing -from $s -to $d -max_paths 10 -file $out/reports/x2_p${p}_${dir}.rpt
    }
  }
}
close $xf
if {[llength $bad]} { error "x2 checks failed: [join $bad {; }]" }
puts "x2 timing / GT / refclk / SLR checks PASS; inspect CDC, bus skew, unconstrained paths and utilization reports"
