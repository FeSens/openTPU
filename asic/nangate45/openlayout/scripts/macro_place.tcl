# Apply one macro placement candidate to the floorplan.  run.sh macro <cand>
# Input:  <work>/<cand>/macro_place.tcl (place_macro commands: candidates/gen_candidates.py,
#         planned with skills/skills_Nangate45/1.macro_place/SKILL_planner.md)
# Output: <work>/<cand>/macro_placed.odb/.def; the log has MACRO_CHECK lines (overlap, outside
#         the core, unplaced) and the macro list.
source $::env(OL_DIR)/scripts/common.tcl
set CAND $::env(OL_CAND)
read_db $WORK/floorplan.odb
foreach inst [[ord::get_db_block] getInsts] {
    if {[[$inst getMaster] isBlock]} { $inst setPlacementStatus NONE }
}
source $WORK/$CAND/macro_place.tcl

set block [ord::get_db_block]
set core  [$block getCoreArea]
set ms {}
set bad 0
foreach inst [$block getInsts] {
    if {![[$inst getMaster] isBlock]} { continue }
    if {![$inst isPlaced]} { puts "MACRO_CHECK unplaced [$inst getName]"; incr bad; continue }
    set b [$inst getBBox]
    if {[$b xMin] < [$core xMin] || [$b yMin] < [$core yMin] || [$b xMax] > [$core xMax] || [$b yMax] > [$core yMax]} {
        puts "MACRO_CHECK outside_core [$inst getName]"; incr bad
    }
    lappend ms [list [$inst getName] [$b xMin] [$b yMin] [$b xMax] [$b yMax]]
}
for {set i 0} {$i < [llength $ms]} {incr i} {
    lassign [lindex $ms $i] n1 a1 b1 c1 d1
    for {set j [expr {$i + 1}]} {$j < [llength $ms]} {incr j} {
        lassign [lindex $ms $j] n2 a2 b2 c2 d2
        if {$a1 < $c2 && $a2 < $c1 && $b1 < $d2 && $b2 < $d1} { puts "MACRO_CHECK overlap $n1 $n2"; incr bad }
    }
}
puts "MACRO_CHECK total_problems $bad ([llength $ms] macros placed)"
write_db  $WORK/$CAND/macro_placed.odb
write_def $WORK/$CAND/macro_placed.def
report_die_core
report_macros
