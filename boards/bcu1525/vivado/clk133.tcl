# The BCU1525 WG build's 150 MHz synthesis implemented at 133.33 MHz, without a synthesis of its
# own (OTPU_CLK_HOOK=clk133.tcl, run_vivado.sh CLK_HOOK: bcu1525_wg_check.tcl sources it at the
# end of opt_design's pre-hook; place_screen.tcl after its open_checkpoint). The clock wizard's
# MMCM (bd_wg.tcl) from 4.0 / 8 to 4.5 / 9 of its 1200 MHz VCO: CLKOUT0 the 2x (acc_clk2x, the
# MMCM's one fractional divide; acc_clk its BUFGCE_DIV /2), CLKOUT1 core_clk; aclk (CLKOUT2, 4)
# and init_clk (CLKOUT3, 12) as built. Then every otpu_clk_sel's flip-flop to INIT 1, its
# constants (otpu_ctrl.sv's otpu_clk_k) at the top's CORE_KHZ1 133333: the governor's and the
# watch's millisecond, the UART's bit, CORE_KHZ. First the report of every otpu_clk_k's two values
# (reports/clk_k.rpt: each a bit the select, its inverse or the constant both share); then each
# property set is read back and the clocks' periods checked (report_clocks into
# reports/clk133_clocks.rpt), and anything off is an error: the run stops (a step's pre-hook).
# The late constraint files ran at link_design with the 150 MHz periods: their datapath-only bounds
# (a 150 MHz period or less) are tighter than 133.33 MHz needs.
# With ::otpu::clk_report_only 1 (build.tcl, on synth_1's netlist): the report alone, nothing set.
# ::otpu::clk_rpt_dir: the reports' directory (default ../../reports, impl_1's run directory's).
# 120 MHz (5.0 / 10) is the same hook's other option, documented only: it needs a third set of
# constants (a two-bit select; otpu_clk_k's K1 is 133.33 MHz's).
namespace eval ::otpu {}

# each otpu_clk_k's values at 150 and 133.33 MHz by its instance name (the consumers': the UART's
# bit DIV, DIV - 1, DIV - 2 and DIV / 2 - 1 at 115200 baud, the millisecond's last count, the
# CORE_KHZ register) and how many of each the build has (0: one or more)
set ::otpu::clk_k_want {
  u_div    {1302 1157 1}
  u_div_m1 {1301 1156 1}
  u_div_m2 {1300 1155 1}
  u_div_h1 {650 577 1}
  u_ms_m1  {149999 133332 2}
  u_khz    {150000 133333 0}
}

proc ::otpu::clk_fail {msg} { error "clk133.tcl: $msg" }

# a flip-flop of an otpu_clk_sel (the select)
proc ::otpu::clk_is_sel {c} {
  if {![llength $c] || [get_property REF_NAME $c] ne "FDRE"} { return 0 }
  set p [get_cells -quiet [get_property PARENT $c]]
  return [expr {[llength $p] && ([get_property ORIG_REF_NAME $p] eq "otpu_clk_sel"
                                 || [get_property REF_NAME $p] eq "otpu_clk_sel")}]
}

# the otpu_clk_sel flip-flops and the otpu_clk_k cells
proc ::otpu::clk_sels {} {
  set hs [get_cells -quiet -hierarchical -filter {ORIG_REF_NAME == otpu_clk_sel || REF_NAME == otpu_clk_sel}]
  set ff {}
  foreach h $hs {
    set c [get_cells -quiet $h/u_sel_ff]
    if {[llength $c] != 1 || ![::otpu::clk_is_sel $c]} { ::otpu::clk_fail "$h: no select flip-flop u_sel_ff" }
    lappend ff $c
  }
  return $ff
}
proc ::otpu::clk_ks {} {
  return [get_cells -quiet -hierarchical -filter {ORIG_REF_NAME == otpu_clk_k || REF_NAME =~ otpu_clk_k*}]
}

# an output bit of an otpu_clk_k: 0, 1, s (the select), n (its inverse) or ? (no driver found)
proc ::otpu::clk_bit {pin} {
  set net [get_nets -quiet -of_objects $pin]
  if {![llength $net]} { return ? }
  set drv [get_pins -quiet -leaf -of_objects $net -filter {DIRECTION == OUT}]
  if {[llength $drv] != 1} { return ? }
  set c [get_cells -quiet -of_objects $drv]
  set r [get_property REF_NAME $c]
  if {$r eq "GND"} { return 0 }
  if {$r eq "VCC"} { return 1 }
  if {[::otpu::clk_is_sel $c]} { return s }
  if {$r eq "LUT1"} {
    set i [get_pins -quiet -leaf -of_objects [get_nets -quiet -of_objects [get_pins -quiet $c/I0]] \
             -filter {DIRECTION == OUT}]
    if {[llength $i] == 1 && [::otpu::clk_is_sel [get_cells -quiet -of_objects $i]]} {
      switch -- [string tolower [get_property INIT $c]] {
        2'h1 - 2'b01 { return n }
        2'h2 - 2'b10 { return s }
      }
    }
  }
  return ?
}

# every otpu_clk_k against clk_k_want: a bit where the two values differ the select or its
# inverse, one they share that constant (or no driver: opt_design folds a constant away); the
# report into clk_k.rpt and the log. An error on any miss, and on a build without the UART's.
proc ::otpu::clk_k_report {rdir} {
  set ks [::otpu::clk_ks]
  if {![llength $ks]} { ::otpu::clk_fail "no otpu_clk_k (a build without CORE_KHZ1: the top's CORE_KHZ1 is its CORE_KHZ)" }
  array set n {}
  set lines {}
  set bad {}
  foreach k [lsort $ks] {
    if {![regexp {([^/.]+)$} $k -> kind] || ![dict exists $::otpu::clk_k_want $kind]} {
      lappend bad "$k: no values in clk133.tcl's clk_k_want"; continue
    }
    lassign [dict get $::otpu::clk_k_want $kind] v0 v1
    incr n($kind)
    set bits {}
    foreach p [get_pins -quiet -of_objects $k -filter {DIRECTION == OUT}] {
      if {[regexp {\[(\d+)\]$} [get_property REF_PIN_NAME $p] -> i]} { lappend bits $i $p }
    }
    if {![llength $bits]} { lappend bad "$k: no output pins"; continue }
    set w [expr {[tcl::mathfunc::max {*}[dict keys $bits]] + 1}]
    set pat ""
    set k0 0; set k1 0; set nsel 0; set nq 0
    for {set i [expr {$w - 1}]} {$i >= 0} {incr i -1} {
      set e0 [expr {($v0 >> $i) & 1}]; set e1 [expr {($v1 >> $i) & 1}]
      set b [expr {[dict exists $bits $i] ? [::otpu::clk_bit [dict get $bits $i]] : "?"}]
      append pat $b
      switch -- $b {
        0 { set ok [expr {!$e0 && !$e1}] }
        1 { set ok [expr {$e0 && $e1}] }
        s { set ok [expr {!$e0 && $e1}]; incr nsel }
        n { set ok [expr {$e0 && !$e1}]; incr nsel }
        default { set ok [expr {$e0 == $e1}]; incr nq }
      }
      if {!$ok} { lappend bad "$k: bit $i '$b', want [expr {$e0}] at 150 MHz and $e1 at 133.33" }
      set k0 [expr {$k0 | (($b eq "?" ? $e0 : $b in {1 n}) << $i)}]
      set k1 [expr {$k1 | (($b eq "?" ? $e1 : $b in {1 s}) << $i)}]
    }
    if {!$nsel} { lappend bad "$k: no bit on the select (its two values folded into one)" }
    if {$v0 >> $w || $v1 >> $w} { lappend bad "$k: $w bits, too few for $v0 / $v1" }
    lappend lines [format "%-48s %6d at 150 MHz, %6d at 133.33 MHz (want %d / %d; %d bits on the select, %d without a driver: %s)" \
                     $k $k0 $k1 $v0 $v1 $nsel $nq $pat]
  }
  dict for {kind spec} $::otpu::clk_k_want {
    set m [lindex $spec 2]
    set got [expr {[info exists n($kind)] ? $n($kind) : 0}]
    if {($m && $got != $m) || (!$m && !$got)} { lappend bad "$got otpu_clk_k $kind, want [expr {$m ? $m : {some}}]" }
  }
  set sels [::otpu::clk_sels]
  lappend lines "select flip-flops (otpu_clk_sel): [llength $sels]"
  foreach c $sels { lappend lines "  $c INIT [get_property INIT $c]" }
  file mkdir $rdir
  set f [open $rdir/clk_k.rpt w]
  foreach l $lines { puts $f $l; puts "clk133.tcl: $l" }
  foreach b $bad { puts $f "ERROR: $b" }
  close $f
  if {[llength $bad]} { ::otpu::clk_fail "[llength $bad] otpu_clk_k checks failed: [join [lrange $bad 0 7] {; }]" }
  puts "clk133.tcl: both sets of constants in the netlist ([llength $ks] otpu_clk_k, $rdir/clk_k.rpt)"
}

# a clock's period (ns) at a pin, one clock
proc ::otpu::clk_period {what pin} {
  set c [get_clocks -quiet -of_objects [get_pins -quiet $pin]]
  if {[llength $c] != 1} { ::otpu::clk_fail "$what: clocks '$c' at $pin, expected one" }
  return [list [get_property NAME $c] [get_property PERIOD $c]]
}

# the MMCM to 133.33 MHz and the selects to 1, each read back; the clocks' periods
proc ::otpu::clk133_apply {rdir} {
  set mm [get_cells -quiet -hierarchical -filter {NAME =~ u_bd/otpu_bdw_i/clk_wiz_0/* && REF_NAME == MMCME4_ADV}]
  if {[llength $mm] != 1} { ::otpu::clk_fail "the clock wizard's MMCM: '$mm' (expected one)" }
  # as bd_wg.tcl built it at 150 MHz, CLKOUT0 to the top's BUFGCE_DIV pair (no buffer) and
  # CLKOUT1 through a BUFG (core_clk)
  foreach {p v} {CLKFBOUT_MULT_F 4 DIVCLK_DIVIDE 1 CLKOUT0_DIVIDE_F 4 CLKOUT1_DIVIDE 8 CLKOUT2_DIVIDE 4} {
    set g [get_property $p $mm]
    if {![string is double -strict $g] || abs($g - $v) > 1e-6} {
      ::otpu::clk_fail "$mm $p $g, not $v: not bd_wg.tcl's 150 MHz MMCM (VCO 1200 MHz, the 2x on CLKOUT0, core_clk on CLKOUT1)"
    }
  }
  set l0 [get_cells -quiet -of_objects [get_pins -quiet -leaf -of_objects [get_nets -quiet -of_objects [get_pins $mm/CLKOUT0]] -filter {DIRECTION == IN}]]
  foreach b {u_acc_buf u_acc2x_buf} {
    if {[lsearch -exact $l0 $b] < 0} { ::otpu::clk_fail "CLKOUT0 does not drive $b (its loads: [lrange $l0 0 7])" }
  }
  set l1 [get_cells -quiet -of_objects [get_pins -quiet -leaf -of_objects [get_nets -quiet -of_objects [get_pins $mm/CLKOUT1]] -filter {DIRECTION == IN}]]
  if {[llength $l1] != 1 || ![string match BUFG* [get_property REF_NAME $l1]]} {
    ::otpu::clk_fail "CLKOUT1 drives '$l1', not core_clk's BUFG"
  }
  # a fractional CLKOUT0 takes CLKOUT5's counter
  foreach p {CLKOUT5 CLKOUT5B} {
    set ld [get_pins -quiet -leaf -of_objects [get_nets -quiet -of_objects [get_pins -quiet $mm/$p]] -filter {DIRECTION == IN}]
    if {[llength $ld]} { ::otpu::clk_fail "$mm/$p has loads ([lrange $ld 0 3]): a fractional CLKOUT0 needs it unused" }
  }
  set sels [::otpu::clk_sels]
  if {![llength $sels]} { ::otpu::clk_fail "no otpu_clk_sel: the build has no fallback constants (CORE_KHZ1)" }
  set_property CLKOUT0_DIVIDE_F 4.500 $mm
  set_property CLKOUT1_DIVIDE 9 $mm
  # (DONT_TOUCH, the flip-flop's and its otpu_clk_sel's, off for the edit and on again, on the
  # flip-flop whatever the RTL's attribute landed on)
  foreach c $sels {
    set dt [list $c [get_cells [get_property PARENT $c]]]
    set_property DONT_TOUCH 0 $dt
    set_property INIT 1'b1 $c
    set_property DONT_TOUCH 1 $dt
  }
  # read back: every property, then the clocks
  set bad {}
  foreach {p v} {CLKOUT0_DIVIDE_F 4.5 CLKOUT1_DIVIDE 9 CLKOUT2_DIVIDE 4 CLKFBOUT_MULT_F 4 DIVCLK_DIVIDE 1} {
    set g [get_property $p $mm]
    if {![string is double -strict $g] || abs($g - $v) > 1e-6} { lappend bad "$p reads $g, not $v" }
  }
  foreach c $sels {
    set g [string tolower [get_property INIT $c]]
    if {$g ni {1'b1 1'h1}} { lappend bad "$c INIT reads $g, not 1'b1" }
    if {![get_property DONT_TOUCH $c]} { lappend bad "$c not DONT_TOUCH" }
  }
  if {[llength $bad]} { ::otpu::clk_fail "[llength $bad] properties did not read back: [join [lrange $bad 0 7] {; }]" }
  file mkdir $rdir
  report_clocks -file $rdir/clk133_clocks.rpt
  report_clocks
  foreach {what pin want} [list acc_clk2x $mm/CLKOUT0 3.750 core_clk $mm/CLKOUT1 7.500 aclk $mm/CLKOUT2 3.333 \
                               acc_clk u_acc_buf/O 7.500 acc_clk2x u_acc2x_buf/O 3.750] {
    lassign [::otpu::clk_period $what $pin] name t
    if {abs($t - $want) > 0.0015} { lappend bad "$what ($name at $pin) $t ns, not $want" }
    puts "clk133.tcl: $what $name $t ns"
  }
  if {[llength $bad]} { ::otpu::clk_fail "the clocks after the edit: [join $bad {; }] ($rdir/clk133_clocks.rpt)" }
  puts "clk133.tcl: $mm CLKOUT0_DIVIDE_F 4.5, CLKOUT1_DIVIDE 9 (core_clk 133.33 MHz, the 2x 266.67), [llength $sels] selects INIT 1"
}

set ::otpu::clk_rd [expr {[info exists ::otpu::clk_rpt_dir] ? $::otpu::clk_rpt_dir : [file normalize ../../reports]}]
::otpu::clk_k_report $::otpu::clk_rd
if {!([info exists ::otpu::clk_report_only] && $::otpu::clk_report_only)} { ::otpu::clk133_apply $::otpu::clk_rd }
