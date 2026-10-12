# image_snapshot skill — whole-die PNG of a checkpoint (see SKILL.md).
if {![info exists in_odb]}     { puts "OL_ERROR snapshot: in_odb not set"; exit 1 }
if {![info exists out_png]}    { puts "OL_ERROR snapshot: out_png not set"; exit 1 }
if {![info exists img_width]}  { set img_width 1024 }
if {![info exists img_margin]} { set img_margin 0.03 }

read_db ${in_odb}

set _blk [ord::get_db_block]
set _dbu [[ord::get_db_tech] getDbUnitsPerMicron]
set _die [$_blk getDieArea]
set _x0 [expr {[$_die xMin] / double($_dbu)}]
set _y0 [expr {[$_die yMin] / double($_dbu)}]
set _x1 [expr {[$_die xMax] / double($_dbu)}]
set _y1 [expr {[$_die yMax] / double($_dbu)}]
set _pad [expr {max($_x1 - $_x0, $_y1 - $_y0) * $img_margin}]
set _a [list [expr {$_x0 - $_pad}] [expr {$_y0 - $_pad}] \
             [expr {$_x1 + $_pad}] [expr {$_y1 + $_pad}]]

if {[catch { save_image -area $_a -width $img_width ${out_png} } _err]} {
    puts "OL_WARN snapshot area save failed ($_err); trying plain save"
    catch { save_image -width $img_width ${out_png} }
}
puts "OL_SNAPSHOT ${out_png}"
