# The s6 QSFP28 link test's refclk tracer, scoped to u_link_phy in the full x2 image so it does
# not inspect XDMA's separately constrained PLLs. For
# each GTYE4_COMMON, each QPLL as its QPLL<n>REFCLKSEL selects it: a constant select names the pin,
# and the pin the top port behind its IBUFDS_GTE4; a select from logic is "dynamic".
#   refclk_check BAN WANT -> {lines errors}
# errors: a quad with no QPLL on a reference pad (none dynamic either: the neighbour pin IBERT
# selects left unwired, say); a QPLL on one of BAN's ports; a quad in WANT ({LOC port} pairs, the
# LOC as GTYE4_COMMON_X1Y11) with no QPLL on its port (none dynamic). A WANT site that no common
# is placed on yet is reported, not failed.
set REFCLK_SEL {- GTREFCLK0 GTREFCLK1 GTNORTHREFCLK0 GTNORTHREFCLK1 GTSOUTHREFCLK0 GTSOUTHREFCLK1 GTGREFCLK}

# a pin's constant (0 / 1), or "-" when logic drives it
proc refclk_bit {pin} {
  set net [get_nets -quiet -of_objects $pin]
  if {$net eq ""} { return - }
  switch -- [get_property -quiet TYPE $net] { GROUND { return 0 } POWER { return 1 } }
  set drv [get_pins -quiet -leaf -of_objects $net -filter {DIRECTION == OUT}]
  if {[llength $drv] != 1} { return - }
  switch -- [get_property REF_NAME [get_cells -of_objects $drv]] { GND { return 0 } VCC { return 1 } }
  return -
}

# what drives a reference pin: {port NAME} (an IBUFDS_GTE4 on top port NAME), {const}, {other TEXT}
proc refclk_src {pin} {
  set net [get_nets -quiet -of_objects $pin]
  if {$net eq ""} { return {other unconnected} }
  if {[get_property -quiet TYPE $net] in {GROUND POWER}} { return const }
  set drv [get_pins -quiet -leaf -of_objects $net -filter {DIRECTION == OUT}]
  if {[llength $drv] != 1} { return [list other "[llength $drv] drivers"] }
  set cell [get_cells -of_objects $drv]
  set ref [get_property REF_NAME $cell]
  if {$ref in {GND VCC}} { return const }
  if {$ref ne "IBUFDS_GTE4"} { return [list other "$ref $cell"] }
  set i [get_pins -quiet -of_objects $cell -filter {REF_PIN_NAME == I}]
  set port [get_ports -quiet -of_objects [get_nets -quiet -segments -of_objects $i]]
  if {[llength $port] != 1} { return [list other "IBUFDS_GTE4 $cell"] }
  return [list port $port]
}

proc refclk_check {ban want} {
  set lines {}
  set errs {}
  set seen {}
  foreach c [lsort [get_cells -quiet -hier -filter {NAME =~ u_link_phy/* && REF_NAME == GTYE4_COMMON}]] {
    set loc [get_property -quiet LOC $c]
    set at [expr {$loc eq "" ? $c : $loc}]
    set ports {}
    set dyn 0
    set desc {}
    foreach p {0 1} {
      set v ""
      foreach b {2 1 0} { append v [refclk_bit [get_pins $c/QPLL${p}REFCLKSEL\[$b\]]] }
      if {[string first - $v] >= 0} { incr dyn; lappend desc "QPLL$p REFCLKSEL $v (dynamic)"; continue }
      scan $v %b n
      if {$n == 0} { lappend desc "QPLL$p REFCLKSEL 000"; continue }
      set pin [lindex $::REFCLK_SEL $n]$p
      lassign [refclk_src [get_pins $c/$pin]] kind what
      lappend desc "QPLL$p $pin <- [expr {$kind eq "const" ? "constant" : $what}]"
      if {$kind ne "port"} { continue }
      lappend ports $what
      if {$what in $ban} { lappend errs "$at QPLL$p on $what (not allowed)" }
    }
    lappend lines "refclk $at: [join $desc {; }]"
    if {![llength $ports] && !$dyn} {
      set hint [expr {[regexp {NORTH|SOUTH} $desc] ? " (QSFP_NB: 1 south-bound, 2 north-bound)" : ""}]
      lappend errs "$at: no QPLL on a reference pad ([join $desc {; }])$hint"
    }
    foreach {wl wp} $want {
      if {$loc ne $wl} { continue }
      lappend seen $wl
      if {$wp ni $ports && !$dyn} { lappend errs "$at: no QPLL on $wp ([join $desc {; }])" }
    }
  }
  foreach {wl wp} $want {
    if {$wl ni $seen} { lappend lines "refclk $wl: no GTYE4_COMMON placed there yet ($wp unchecked)" }
  }
  return [list $lines $errs]
}
