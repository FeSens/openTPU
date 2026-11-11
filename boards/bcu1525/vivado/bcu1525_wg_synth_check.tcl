# The BCU1525 WG build's check of a finished synthesis (build.tcl, on synth_1's open netlist, in a
# scope of its own: IMAGE=wg at WG_S 2, or at either S with OTPU_WG_STOP=synth): the floorplan's
# pblocks made on the netlist (constraints/bcu1525_wg_floorplan.tcl, as the implementation's link
# makes them), so a cell it misses stops the build here and not at opt_design's pre-hook after the
# implementation's link; then each pblock's share of the netlist, its leaf primitives by type
# against an SLR of the VU9P (394,080 LUTs, 788,160 FFs, 720 block RAM tiles, 320 UltraRAMs, 2,280
# DSPs, f150r62's report_utilization -slr): a two-slice build's dies before place and route
# (docs/bcu1525_s2.md). LUTs counts the LUT1..LUT6 cells (no LUT
# combining yet, LUT RAMs and shift registers apart, as cells), block RAM RAMB36 + RAMB18 / 2;
# pblock_mxu<s>'s DSPs (OTPU_WG_MXU_COLS<s>) are inside pblock_slice<s>'s count. The official
# numbers: reports/synth_util_pblocks.rpt (report_utilization -pblocks). The pblocks stay in this
# Vivado's memory: build.tcl closes the design unsaved. Needs `here` (build.tcl's directory) and
# `out` (OUT_DIR).
namespace eval ::otpu {}
array unset ::otpu::ok
set ::otpu::fail {}
# open_run synth_1 has made the floorplan's pblocks already: the late Tcl constraints are the
# implementation's, and an opened synthesized design applies them (s2syn, 2026-10-31: Vivado 12-795,
# pblock 'pblock_slice0' already exists). They go, and the floorplan runs again for this count.
for {set i 0} {$i < 3 && [llength [get_pblocks -quiet]]} {incr i} { delete_pblock -quiet [get_pblocks -quiet] }
if {[llength [get_pblocks -quiet]]} { error "bcu1525_wg_synth_check.tcl: pblocks left: [get_pblocks -quiet]" }
source [file normalize $here/../constraints/bcu1525_wg_floorplan.tcl]
if {[llength [get_cells -quiet u_link]]} { source [file normalize $here/../x2/floorplan_x2.tcl] }
set k [expr {[info exists ::otpu::ok(bcu1525_wg_floorplan)] ? $::otpu::ok(bcu1525_wg_floorplan) : 0}]
if {$k != 1 || [llength $::otpu::fail]} {
  error "bcu1525_wg_synth_check.tcl: the floorplan finished $k of 1: [join $::otpu::fail {; }]"
}
puts "bcu1525_wg_synth_check.tcl: the floorplan's cells all found"
set pbs [lsort [get_pblocks -quiet]]
if {![llength $pbs]} { puts "bcu1525_wg_synth_check.tcl: no pblocks (OTPU_WG_FP none)"; return }
# -pblocks takes one pblock (Vivado 2025.2, Common 17-69: s2syn's check, 2026-11-01): a report each, appended
file delete -force $out/reports/synth_util_pblocks.rpt
foreach pb $pbs { report_utilization -pblocks $pb -append -file $out/reports/synth_util_pblocks.rpt }
# type: its cells' REF_NAMEs (globs), an SLR's sites (0: none given)
set types {LUT {LUT[1-6]*} 394080  LUTRAM/SRL {RAM32[MX]* RAM64[MX]* RAM128X* RAM256X* RAM512X* SRL*} 0
           FF {FD[CPRS]E} 788160  BRAM {RAMB36E2 FIFO36E2 RAMB18E2 FIFO18E2} 720  URAM URAM288 320
           DSP DSP48E2 2280}
foreach pb $pbs {
  if {[catch {
    # the pblock's leaf cells as they are, its hierarchical cells' leaves below them (brackets as ?)
    # (get_property on no objects is an error, Common 17-55: s2syn's counts, 2026-11-01; a pblock of hierarchical cells
    # alone has no leaf cells of its own)
    set cs [get_cells -quiet -of_objects $pb]
    set prim [filter -quiet $cs {IS_PRIMITIVE == 1}]
    set refs [expr {[llength $prim] ? [get_property REF_NAME $prim] : {}}]
    set f {}
    foreach c [filter -quiet $cs {IS_PRIMITIVE == 0}] { lappend f "NAME =~ [string map {[ ? ] ?} $c]/*" }
    if {[llength $f]} {
      set leaves [get_cells -quiet -hierarchical -filter "([join $f { || }]) && IS_PRIMITIVE == 1 && PRIMITIVE_LEVEL != INTERNAL"]
      if {[llength $leaves]} { set refs [concat $refs [get_property REF_NAME $leaves]] }
    }
    set n [dict create]
    foreach r $refs { dict incr n $r }
    set line {}
    foreach {t pats cap} $types {
      set v 0.0
      dict for {r m} $n {
        foreach q $pats {
          if {[string match $q $r]} { set v [expr {$v + ($r in {RAMB18E2 FIFO18E2} ? 0.5 : 1.0) * $m}]; break }
        }
      }
      set v [expr {$v == int($v) ? int($v) : $v}]
      lappend line [expr {$cap ? [format "%s %s (%.1f%%)" $t $v [expr {100.0 * $v / $cap}]] : "$t $v"}]
    }
    puts "bcu1525_wg_synth_check.tcl: $pb ([get_property -quiet GRID_RANGES $pb]): [join $line {, }]"
  } e]} {
    puts "bcu1525_wg_synth_check.tcl: $pb: no counts ($e)"
  }
}
puts "bcu1525_wg_synth_check.tcl: report_utilization -pblocks in $out/reports/synth_util_pblocks.rpt"
