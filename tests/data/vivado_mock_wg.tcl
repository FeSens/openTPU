# A mock Vivado netlist and device for tests/test_bcuwg_build.py: the WG floorplan
# (constraints/bcu1525_wg_floorplan.tcl, MODE fp) and the synthesis check (vivado/
# bcu1525_wg_synth_check.tcl, MODE chk) on a small netlist of the build's cell names (a slice: 12
# MXU DSPs, 3 more, 50 LUT6 and FDRE, an UltraRAM, a RAMB36 and a RAMB18, a RAM32M16 macro, an
# SRL16E; its adapter 7 LUT3; a bridge core half a LUT2; a bridge pipeline's stage an FDRE each
# way, N + 1 of them (N from WG_OUT_DIR's wg_params.tcl RLY); a crossing stage an FDRE; sc_dma an
# FDRE; the clock wizard's MMCM; at two slices the top's reset stages, acc_rst_p's 5 and acc_rq's
# 2, FDSEs), the VU9P's DSP sites (19 columns, 120 rows an SLR, column 18 five), the commands
# those scripts use; then each pblock (ranges, parent, cells) and each LOC.
# MODE open: the floorplan first (open_run synth_1 applies it), then chk.
#   tclsh vivado_mock_wg.tcl NS WG_OUT_DIR fp|chk|open VIVADO_DIR OUT_DIR
lassign $argv ns wgout mode here out
# ---- cells: name -> {ref prim level}; hierarchical cells have ref "" prim 0
array set C {}
proc hc {n} { set ::C($n) {{} 0 HIER} }
proc lc {n ref {lvl LEAF}} { set ::C($n) [list $ref 1 $lvl] }
hc u_sys/u_board; hc u_sys/u_board/u_ctrl; hc u_sys/u_board/u_iso; hc u_sys/u_board/g_trace.u_trace
hc u_sys/u_board/g_coll.u_coll
proc slice {pre} {
  hc ${pre}u_slice; hc ${pre}u_xs; hc ${pre}u_mem; hc ${pre}u_slice/u_mxu
  for {set i 0} {$i < 12} {incr i} { lc ${pre}u_slice/u_mxu/pe\[$i\].dsp DSP48E2 }
  for {set i 0} {$i < 3} {incr i} { lc ${pre}u_slice/u_vpu/d$i DSP48E2 }
  for {set i 0} {$i < 50} {incr i} { lc ${pre}u_slice/l$i LUT6; lc ${pre}u_slice/f$i FDRE }
  lc ${pre}u_slice/u_mxu/w0 URAM288; lc ${pre}u_slice/b36 RAMB36E2; lc ${pre}u_slice/b18 RAMB18E2
  lc ${pre}u_slice/lr RAM32M16 MACRO; lc ${pre}u_slice/lr/A RAMD32 INTERNAL; lc ${pre}u_slice/s0 SRL16E
  for {set i 0} {$i < 7} {incr i} { lc ${pre}u_mem/l$i LUT3 }
}
slice u_sys/u_board/
set fh [open $wgout/wg_params.tcl]; regexp {set RLY ([0-9a-f]+)} [read $fh] -> rly; close $fh
for {set k 0} {$k < 8 * $ns} {incr k} {
  hc "u_sys/g_ch\[$k\].u_br/u_cs"; lc "u_sys/g_ch\[$k\].u_br/u_cs/l" LUT2; hc "u_sys/g_ch\[$k\].u_br/u_ln"
  scan [string index $rly end-$k] %x n
  for {set j 0} {$j <= $n} {incr j} { foreach r {fs bs} { lc "u_sys/g_ch\[$k\].u_br/u_ln/${r}_reg\[$j\]\[0\]" FDRE } }
}
hc u_bd/otpu_bdw_i/sc_dma; lc u_bd/otpu_bdw_i/sc_dma/r FDRE
lc u_bd/otpu_bdw_i/clk_wiz_0/inst/mmcme4_adv_inst MMCME4_ADV
if {$ns == 2} {
  slice u_sys/u_board/g_slice\[1\].
  hc u_sys/u_board/g_xs.u_chub
  foreach s {0 1} { foreach c {u_dn_tx u_dn_rx u_cport u_up_tx u_up_rx} { hc "u_sys/u_board/g_xs.g_x\[$s\].$c"; lc "u_sys/u_board/g_xs.g_x\[$s\].$c/r" FDRE } }
  hc u_sys/u_board/g_xs.g_x\[0\].g_pf.u_pf_tx; hc u_sys/u_board/g_xs.g_x\[0\].g_pf.u_pf_rx
  for {set i 0} {$i < 5} {incr i} { lc acc_rst_p_reg\[$i\] FDSE }
  foreach i {0 1} { lc g_arq.acc_rq_reg\[$i\] FDSE }
}
# ---- filter expressions -> Tcl expr over name/ref/prim/lvl/site
proc fx {f} {
  regsub -all {NAME =~ ([^ ()|&]+)} $f {[string match {\1} $name]} f
  regsub -all {REF_NAME == ([A-Z0-9_]+)} $f {($ref eq "\1")} f
  regsub -all {IS_PRIMITIVE == ([01])} $f {($prim == \1)} f
  regsub -all {PRIMITIVE_LEVEL != ([A-Z]+)} $f {($lvl ne "\1")} f
  regsub -all {SITE_TYPE == ([A-Z0-9_]+)} $f {($st eq "\1")} f
  return $f
}
proc cellmatch {name f} { lassign $::C($name) ref prim lvl; return [expr [fx $f]] }
array set PBC {}
proc get_cells {args} {
  set q 0; set hier 0; set f ""; set of ""; set pats {}
  for {set i 0} {$i < [llength $args]} {incr i} {
    set a [lindex $args $i]
    switch -- $a { -quiet { } -hierarchical { set hier 1 } -filter { set f [lindex $args [incr i]] }
      -of_objects { set of [lindex $args [incr i]] } default { set pats $a } }
  }
  if {$of ne ""} { set r {}; foreach pb $of { if {[info exists ::PBC($pb)]} { set r [concat $r $::PBC($pb)] } }; return $r }
  set r {}
  if {$hier} { foreach n [lsort [array names ::C]] { if {[cellmatch $n $f]} { lappend r $n } }; return $r }
  foreach p $pats { set p2 [string map {\[ \\[ \] \\]} $p]; foreach n [array names ::C] { if {[string match $p2 $n]} { lappend r $n } } }
  return [lsort -unique $r]
}
proc filter {args} { set args [lsearch -all -inline -not $args -quiet]; lassign $args cs f; set r {}; foreach n $cs { if {[cellmatch $n $f]} { lappend r $n } }; return $r }
proc get_property {args} {
  set q [expr {"-quiet" in $args}]; set args [lsearch -all -inline -not $args -quiet]; lassign $args p obj
  if {![llength $obj] && !$q} { error "ERROR: \[Common 17-55\] 'get_property' expects at least one object." }
  if {$p eq "REF_NAME"} { set r {}; foreach n $obj { lappend r [lindex $::C($n) 0] }; return $r }
  if {$p eq "GRID_RANGES"} { return $::PBR($obj) }
  return ""
}
array set PBR {}; array set PBP {}
proc create_pblock {n} {
  if {[info exists ::PBR($n)]} { error "ERROR: \[Vivado 12-795\] Pblock '$n' already exists." }
  set ::PBR($n) {}; set ::PBC($n) {}; return $n
}
proc delete_pblock {args} {
  foreach pb [lindex [lsearch -all -inline -not -glob $args -*] 0] { unset -nocomplain ::PBR($pb) ::PBC($pb) ::PBP($pb) }
}
proc resize_pblock {pb a r} { lappend ::PBR($pb) {*}$r }
proc add_cells_to_pblock {pb c} {
  foreach n $c { foreach o [array names ::PBC] { set ::PBC($o) [lsearch -all -inline -not -exact $::PBC($o) $n] } }
  lappend ::PBC($pb) {*}$c
}
array set LOC {}
proc set_property {args} {
  lassign $args p v o
  if {$p eq "PARENT"} { set ::PBP($o) $v }
  if {$p eq "LOC"} {
    if {![regexp {^MMCM_X[01]Y([0-9]|1[0-4])$} $v]} { error "ERROR: \[Vivado 12-1411\] Cannot set LOC property of instance '$o', Site '$v' is not valid" }
    set ::LOC($o) $v
  }
}
proc get_pblocks {args} { return [lsort [array names ::PBR]] }
proc get_slrs {args} { return [lindex $args end] }
proc get_clock_regions {args} { return "CR_[lindex $args end]" }
proc get_sites {args} {
  set args [lsearch -all -inline -not $args -quiet]
  set f [lindex $args [expr {[lsearch $args -filter] + 1}]]; set cr [lindex $args [expr {[lsearch $args -of_objects] + 1}]]
  set r {}
  foreach c $cr {
    regexp {SLR([0-2])} $c -> s
    for {set x 0} {$x < 19} {incr x} {
      set ny [expr {$x == 18 ? 5 : 120}]
      for {set y [expr {120 * $s}]} {$y < 120 * $s + $ny} {incr y} {
        set name DSP48E2_X${x}Y$y; set st DSP48E2
        if {[expr [fx $f]]} { lappend r $name }
      }
    }
  }
  return $r
}
proc report_utilization {args} {
  set i [lsearch $args -pblocks]
  if {$i >= 0 && [llength [lindex $args $i+1]] > 1} { error "ERROR: \[Common 17-69\] Command failed: '-pblocks' can take only one pblock as argument" }
  puts "RU $args"
}
set ::env(OTPU_WG_OUT) $wgout
if {$mode eq "open"} {
  # open_run synth_1: the late Tcl constraints, the floorplan among them, before build.tcl's check
  namespace eval ::otpu {}
  source $here/../constraints/bcu1525_wg_floorplan.tcl
  set mode chk
}
if {$mode eq "fp"} {
  source $here/../constraints/bcu1525_wg_floorplan.tcl
  puts "ok [array get ::otpu::ok] fail [expr {[info exists ::otpu::fail] ? $::otpu::fail : {}}]"
} else {
  set rc [catch { apply {{here out} { source $here/bcu1525_wg_synth_check.tcl }} $here $out } e]
  puts "rc $rc $e"
}
foreach c [lsort [array names LOC]] { puts "LOC $c $LOC($c)" }
foreach pb [lsort [array names PBR]] { puts "PB $pb ranges [llength $PBR($pb)] [lrange $PBR($pb) 0 1] parent [expr {[info exists PBP($pb)] ? $PBP($pb) : "-"}] cells [llength $PBC($pb)]" }
