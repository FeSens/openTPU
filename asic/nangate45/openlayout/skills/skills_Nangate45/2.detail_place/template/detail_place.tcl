if {![info exists data_dir]}   { set data_dir [expr {[info exists ::env(DESIGN_DIR)] ? $::env(DESIGN_DIR) : "./data"}] }
if {![info exists proj_name]}  { set proj_name "ariane133" }
if {![info exists save_dir]}   { set save_dir ${data_dir}/saves_db }
if {![info exists report_dir]} { set report_dir ${data_dir}/report }
file mkdir $save_dir
file mkdir $report_dir

# Lock macros before PDN. place_macro leaves macros PLACED (movable), and
# pdngen errors "Design has unplaced macros" unless they are FIXED. Fixing
# them in one pass here (rather than per-macro during placement) avoids
# place_macro rejecting later macros that abut already-fixed neighbors.
set _nfix 0
foreach _inst [[ord::get_db_block] getInsts] {
    if {[[$_inst getMaster] isBlock]} { $_inst setPlacementStatus "FIRM"; incr _nfix }
}
utl::report "Fixed $_nfix macros (FIRM) before PDN"

# Trim std-cell rows around macros + delete sliver channels. Without this,
# narrow inter-macro channels get std cells with no routable power rail,
# which makes pdngen fail with PDN-0179 "Unable to repair all channels".
catch { cut_rows -halo_width_x 3.0 -halo_width_y 3.0 -row_min_width 12.0 } _cuterr
if {[info exists _cuterr] && $_cuterr ne ""} { utl::report "cut_rows: $_cuterr" }

# Per-macro placement HALO: keep std cells a keepout margin away from every
# macro edge (pin access + routing room; prevents the std-cell/macro edge
# crowding seen on bp_local). cut_rows above only trims rows; this adds an
# explicit hard placement blockage over each macro bbox expanded by the halo,
# which BOTH global_placement (GPL) and detailed_placement (OpenDP) honor, so
# cells stay clear in the global pass too (not only after legalization).
set _halo_um 5.0
set _hblk [ord::get_db_block]
set _h [expr {int($_halo_um * [$_hblk getDbUnitsPerMicron])}]
set _nhalo 0
foreach _inst [$_hblk getInsts] {
    if {![[$_inst getMaster] isBlock]} { continue }
    set _bb [$_inst getBBox]
    odb::dbBlockage_create $_hblk \
        [expr {[$_bb xMin] - $_h}] [expr {[$_bb yMin] - $_h}] \
        [expr {[$_bb xMax] + $_h}] [expr {[$_bb yMax] + $_h}]
    incr _nhalo
}
utl::report "Added ${_halo_um}um placement-blockage halo around $_nhalo macros"

# Wire RC must be set before any repair/timing commands
set_wire_rc -signal -layer metal3
set_wire_rc -clock  -layer metal5

##########################################################################
# Global Connections
##########################################################################

# std cells
add_global_connection -net {VDD} -inst_pattern {.*} -pin_pattern {VDD} -power
add_global_connection -net {VSS} -inst_pattern {.*} -pin_pattern {VSS} -ground
# pads
add_global_connection -net {VDD} -inst_pattern {.*} -pin_pattern {vdd} -power
add_global_connection -net {VSS} -inst_pattern {.*} -pin_pattern {vss} -ground
# fix for bondpad/port naming
add_global_connection -net {VDDIO} -inst_pattern {.*} -pin_pattern {.*vdd_RING} -power
add_global_connection -net {VSSIO} -inst_pattern {.*} -pin_pattern {.*vss_RING} -ground
# rams
add_global_connection -net {VDD} -inst_pattern {.*} -pin_pattern {VDDARRAY} -power
add_global_connection -net {VDD} -inst_pattern {.*} -pin_pattern {VDDARRAY!} -power
add_global_connection -net {VDD} -inst_pattern {.*} -pin_pattern {VDD!} -power
add_global_connection -net {VSS} -inst_pattern {.*} -pin_pattern {VSS!} -ground

# pads
add_global_connection -net {VDDIO} -inst_pattern {.*} -pin_pattern {iovdd} -power
add_global_connection -net {VSSIO} -inst_pattern {.*} -pin_pattern {iovss} -ground
# fix for bondpad/port naming
add_global_connection -net {VDDIO} -inst_pattern {.*} -pin_pattern {.*iovdd_RING} -power
add_global_connection -net {VSSIO} -inst_pattern {.*} -pin_pattern {.*iovss_RING} -ground

# connection
global_connect

# voltage domains
set_voltage_domain -name {CORE} -power {VDD} -ground {VSS}
# standard cell grid and rings
define_pdn_grid -name {core_grid} -voltage_domains {CORE}

##########################################################################
##  PDN — Nangate45 / ariane133  (core-only, stripes)
##########################################################################

# ----- Power ring (core, on metal9 / metal10) -----
set pgcrWidth    1.6
set pgcrSpacing  1.6
set pgcrOffset   2.0

# ----- Top-layer stripes (metal9, horizontal) -----
set tpg1Width    1.6
set tpg1Pitch   56.0
set tpg1Spacing  2.0
set tpg1Offset  20.0

# ----- Mid-layer distribution stripes (metal8, vertical) -----
set tpg4Width    0.96
set tpg4Pitch   56.0
set tpg4Spacing  2.0
set tpg4Offset  20.0

# ----- Core ring on metal9/metal10 (only if it fits the die margin) -----
# A two-wire ring needs offset + 2*width + spacing (~6.8 um default) of clear
# core-to-die margin. Tight floorplans (e.g. or1200_local leaves ~1 um) make
# pdngen abort with PDN-0351 "rings do not fit inside the die area". Measure the
# margin and skip the ring when it can't fit; stripes then span the core without
# extending to a ring. Circuits with adequate margin keep the ring as before.
set block [ord::get_db_block]
set _ring_need [expr {$pgcrOffset + 2*$pgcrWidth + $pgcrSpacing}]
set _dbu  [[ord::get_db_tech] getDbUnitsPerMicron]
set _die  [$block getDieArea]
set _core [$block getCoreArea]
set _mgn [expr {min( \
    ([$_core xMin]-[$_die xMin]), ([$_die xMax]-[$_core xMax]), \
    ([$_core yMin]-[$_die yMin]), ([$_die yMax]-[$_core yMax]) ) / double($_dbu)}]
set _ext "-extend_to_core_ring"
if {$_mgn < $_ring_need + 0.2} {
    set _ext ""
    utl::report "Core-to-die margin ${_mgn} um < ring need ${_ring_need} um: skipping core ring (stripes-only PDN)"
} else {
    utl::report "Core-to-die margin ${_mgn} um: adding core ring"
    add_pdn_ring -grid {core_grid} \
       -layer        {metal9 metal10} \
       -widths       "$pgcrWidth $pgcrWidth" \
       -spacings     "$pgcrSpacing $pgcrSpacing" \
       -core_offsets "$pgcrOffset $pgcrOffset" \
       -add_connect
}

# ----- Std-cell rails (metal1 follow-pins) -----
add_pdn_stripe -grid {core_grid} -layer {metal1} \
               -width 0.17 -offset 0 \
               -followpins {*}$_ext

# ----- Mid-layer distribution stripes (metal8, vertical) -----
add_pdn_stripe -grid {core_grid} -layer {metal8} \
               -width $tpg4Width \
               -pitch $tpg4Pitch -spacing $tpg4Spacing -offset $tpg4Offset \
               {*}$_ext -snap_to_grid

# ----- Top stripes (metal9, horizontal) -----
add_pdn_stripe -grid {core_grid} -layer {metal9} \
               -width $tpg1Width \
               -pitch $tpg1Pitch -spacing $tpg1Spacing -offset $tpg1Offset \
               {*}$_ext -snap_to_grid

# ----- Via connections (core grid) -----
# metal9<->metal10 already connected by the core ring's -add_connect.
# Ladder: metal9 (H) -> metal8 (V) -> metal1 (followpins).
add_pdn_connect -grid {core_grid} -layers {metal9 metal8}
add_pdn_connect -grid {core_grid} -layers {metal8 metal1}

# ----- Generate -----
if {![info exists report_dir]} { set report_dir "./reports" }
if {![info exists proj_name]}  { set proj_name  "ariane133" }
file mkdir $report_dir

pdngen

# Don't touch clock-source nets during placement/repair. Optimization can
# otherwise merge an output-forwarded clock (e.g. vga_lcd's clk_p_o) into
# another clock's net, which makes TritonCTS abort with CTS-0114 "overlaps a
# previous clock". Derive the clock-root nets generically from the SDC clocks
# (robust across designs / OpenROAD versions — materialize each source pin name
# to a string first to avoid stale STA handles / spurious STA-2204).
# A read_db checkpoint restores the netlist/placement but not the SDC clocks
# (those live in STA). Re-read the SDC if the constraints were dropped, so the
# clock-net protection below actually finds the clocks.
if {[llength [all_clocks]] == 0} {
    catch { read_sdc ${data_dir}/constraint.sdc }
}
proc clock_root_nets {} {
    set nets {}
    foreach clk [all_clocks] {
        set srcnames {}
        catch { foreach src [get_property $clk sources] { lappend srcnames [get_full_name $src] } }
        foreach pn $srcnames {
            set n [get_nets -quiet -of_objects [get_pins -quiet $pn]]
            if {$n eq ""} { set n [get_nets -quiet $pn] }
            foreach net $n { lappend nets [get_full_name $net] }
        }
    }
    return [lsort -unique $nets]
}
set CLK_NETS [clock_root_nets]
if {[llength $CLK_NETS]} {
    catch { set_dont_touch [get_nets $CLK_NETS] }
    puts "DONT_TOUCH clk nets: $CLK_NETS"
} else {
    puts "No clock net resolved -> skipping clk dont_touch"
}

utl::report "Remove buffers"
remove_buffers

utl::report "Repair design"
repair_design -verbose


set_thread_count 32

# global_placement parameters:
# density:            In every part of the chip, about N% of the area is occupied by standard cells
# routability_driven: Reduce density target when there are a lot of wires in an area
# check_overflow:     Higher means routability starts being considered earlier in placement
#                     too early -> very dense regions, too late -> little to no effect
# timing_driven:      Prioritize near-critical timing paths (reduce their length)

# Rough placement to get parasitics from steiner-tree estimate so we can run repair_timing
utl::report "Global Placement (1)"
global_placement -density 0.60


utl::report "Estimate parasitics"
estimate_parasitics -placement
utl::report "Repair design"
repair_design -verbose

# Bounded so a poor (Gate-A-failing) placement on a large design cannot trigger
# an unbounded single-threaded repair_timing runaway (observed churning >100 min
# on ariane81's 154k cells). The bounds only LIMIT repair effort (never inflate
# margins) and are identical for every candidate, so the flow stays fair.
set REPAIR_BOUNDS "-max_buffer_percent 20 -max_utilization 90 -max_passes 50"
utl::report "Repair setup"
repair_timing -setup -verbose {*}$REPAIR_BOUNDS

# Actual global placement with routability and timing driven
utl::report "Global Placement (2)"
global_placement -density 0.60 \
                 -routability_driven \
                 -routability_check_overflow 0.30


utl::report "###############################################################################"
utl::report "# 02-03: Detailed Placement"
utl::report "###############################################################################"

# Legalize overlapping cells
utl::report "Detailed placement"
detailed_placement

utl::report "Optimize mirroring"
optimize_mirroring

utl::report "Estimate parasitics"
estimate_parasitics -placement

#### save the database
write_db ${save_dir}/detail_placed.odb

