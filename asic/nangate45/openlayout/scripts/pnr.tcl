# Placement, CTS and routing of one macro placement candidate (the OpenLayout skill templates
# skills/skills_Nangate45/2.detail_place and 3.post_detail_placement/cts_route.tcl, with the
# timing repair passes kept).  run.sh pnr <cand>
# Environment (besides common.tcl's):
#   OL_CAND                    the candidate (cand1, cand2, cand3, rtlmp)
#   OL_START=place|cts|route   resume from <cand>/reports/<top>_{placed,cts}.odb (default place:
#                              starts from <cand>/macro_placed.odb)
#   OL_STOP=place|cts|route    last stage to run (default route)
#   OL_TIMING_DRIVEN=1         the routability placement is also timing driven
#   OL_GP_DENSITY, OL_GRT_ITERS, OL_THREADS, OL_SDC
# Outputs in <work>/<cand>/reports: <top>_placed / _cts / _grt / _routed .odb (.def, .v),
# ppa_<stage>.txt, power_<stage>.txt, <top>.drc.rpt, clock_skew_<stage>.txt, setup_<stage>.txt,
# vclk_latency.txt
source $::env(OL_DIR)/scripts/common.tcl
set CAND  $::env(OL_CAND)
set START place
set STOP  route
set GP_DENSITY 0.55
set GRT_ITERS  30
set MACRO_HALO 3.0
set TIMING_DRIVEN 0
if {[info exists ::env(OL_START)] && $::env(OL_START) ne ""} { set START $::env(OL_START) }
if {[info exists ::env(OL_STOP)] && $::env(OL_STOP) ne ""} { set STOP $::env(OL_STOP) }
if {[info exists ::env(OL_GP_DENSITY)] && $::env(OL_GP_DENSITY) ne ""} { set GP_DENSITY $::env(OL_GP_DENSITY) }
if {[info exists ::env(OL_GRT_ITERS)] && $::env(OL_GRT_ITERS) ne ""} { set GRT_ITERS $::env(OL_GRT_ITERS) }
if {[info exists ::env(OL_TIMING_DRIVEN)] && $::env(OL_TIMING_DRIVEN) ne ""} { set TIMING_DRIVEN $::env(OL_TIMING_DRIVEN) }

set report_dir $WORK/$CAND/reports
file mkdir $report_dir
set stages {place cts route}
set si [lsearch $stages $START]
set ei [lsearch $stages $STOP]

if {$START eq "place"} { set INPUT_DB $WORK/$CAND/macro_placed.odb }
if {$START eq "cts"}   { set INPUT_DB $report_dir/${TOP}_placed.odb }
if {$START eq "route"} { set INPUT_DB $report_dir/${TOP}_cts.odb }
read_db $INPUT_DB
load_libs
read_sdc $SDC_FILE
set_rc
set design [[ord::get_db_block] getName]
utl::report "design=$design cand=$CAND start=$START stop=$STOP input_db=$INPUT_DB gp_density=$GP_DENSITY threads=$THREADS"

# Legalize; if some cells find no site within DPL_MAXDISP, retry with a larger window.
proc legalize {} {
    global DPL_MAXDISP DPL_RETRY_DISP
    if {[catch { detailed_placement -max_displacement $DPL_MAXDISP } e]} {
        puts "  DPL_CATCH: $e -> retry with -max_displacement $DPL_RETRY_DISP"
        if {[catch { detailed_placement -max_displacement $DPL_RETRY_DISP } e2]} {
            puts "  DPL_CATCH: $e2"
        }
    }
}

# Clock root net names (as strings; STA handles go stale).
proc clock_nets {} {
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
set CLK_NETS [clock_nets]

# The ports are timed against the virtual clock vclk. Once the core clock is propagated, vclk
# gets the tree's insertion delay as latency, so the port paths are not charged (outputs) or
# credited (inputs) the insertion delay. The value is kept in vclk_latency.txt for resumed runs.
proc set_vclk_latency {} {
    global report_dir
    if {[llength [get_clocks -quiet vclk]] == 0} { return }
    set f $report_dir/vclk_latency.txt
    if {![file exists $f]} {
        report_clock_skew > $report_dir/clock_skew_tmp.txt
        set fp [open $report_dir/clock_skew_tmp.txt]; set txt [read $fp]; close $fp
        set lats {}
        foreach {m v} [regexp -all -inline {(-?[0-9.]+) +source latency} $txt] { lappend lats $v }
        if {[llength $lats] == 0} { puts "VCLK_LATENCY: no source latency found"; return }
        set s 0.0
        foreach v $lats { set s [expr {$s + $v}] }
        set fp [open $f w]; puts $fp [format %.3f [expr {$s / [llength $lats]}]]; close $fp
    }
    set fp [open $f]; set lat [string trim [read $fp]]; close $fp
    set_clock_latency $lat [get_clocks vclk]
    puts "VCLK_LATENCY: $lat ns on vclk (the core clock tree's insertion delay)"
}

# PPA summary of the current stage -> ${report_dir}/ppa_<stage>.txt (key: value)
proc report_ppa {stage} {
    global report_dir design
    set block [ord::get_db_block]
    set dbu   [[ord::get_db_tech] getDbUnitsPerMicron]
    set die   [$block getDieArea]
    set nm 0; set ns 0; set ma 0.0; set sa 0.0
    foreach inst [$block getInsts] {
        set m [$inst getMaster]
        set a [expr {double([$m getWidth]) * [$m getHeight] / $dbu / $dbu}]
        if {[$m isBlock]} { incr nm; set ma [expr {$ma + $a}] } else { incr ns; set sa [expr {$sa + $a}] }
    }
    set fp [open ${report_dir}/ppa_${stage}.txt w]
    puts $fp "design: $design"
    puts $fp "stage: $stage"
    puts $fp [format "die_um: %.1f x %.1f" [expr {[$die dx] / double($dbu)}] [expr {[$die dy] / double($dbu)}]]
    puts $fp "macros: $nm"
    puts $fp [format "macro_area_um2: %.0f" $ma]
    puts $fp "std_cells: $ns"
    puts $fp [format "std_cell_area_um2: %.0f" $sa]
    foreach {key cmd} {
        setup_wns_ns   {sta::worst_slack_cmd max}
        setup_tns_ns   {sta::total_negative_slack_cmd max}
        hold_wns_ns    {sta::worst_slack_cmd min}
        hold_tns_ns    {sta::total_negative_slack_cmd min}
        setup_violating_endpoints {sta::endpoint_violation_count max}
        hold_violating_endpoints  {sta::endpoint_violation_count min}
    } {
        if {[catch { set v [eval $cmd] } e]} { set v "n/a" }
        if {[string is double -strict $v] && [string match *_ns $key]} { set v [format %.3f [sta::time_sta_ui $v]] }
        puts $fp "$key: $v"
    }
    close $fp
    catch { report_power > ${report_dir}/power_${stage}.txt }
    catch { report_clock_skew > ${report_dir}/clock_skew_${stage}.txt }
    catch { report_checks -path_delay max -group_count 20 -fields {fanout cap slew} > ${report_dir}/setup_${stage}.txt }
    catch { report_checks -path_delay min -group_count 5 > ${report_dir}/hold_${stage}.txt }
    puts "=== PPA ($stage) ==="
    set fp [open ${report_dir}/ppa_${stage}.txt]; puts [read $fp]; close $fp
    report_design_area
    report_tns
    report_wns
    report_worst_slack -max
    report_worst_slack -min
}

# ---------------------------------------------------------------------------
# Placement: FIRM macros + halo blockages, PDN, GP + repair + routability GP + DPL + repair
# ---------------------------------------------------------------------------
if {$si <= 0} {
    set block [ord::get_db_block]
    set dbu   [$block getDbUnitsPerMicron]
    foreach inst [$block getInsts] {
        if {[[$inst getMaster] isBlock]} { $inst setPlacementStatus FIRM }
    }
    # Halo: a placement blockage around every macro
    set h [expr {round($MACRO_HALO * $dbu)}]
    set die [$block getDieArea]
    set n 0
    foreach inst [$block getInsts] {
        if {![[$inst getMaster] isBlock]} { continue }
        set b [$inst getBBox]
        odb::dbBlockage_create $block \
            [expr {max([$die xMin], [$b xMin] - $h)}] [expr {max([$die yMin], [$b yMin] - $h)}] \
            [expr {min([$die xMax], [$b xMax] + $h)}] [expr {min([$die yMax], [$b yMax] + $h)}]
        incr n
    }
    utl::report "Added halo blockages around $n macros (halo $MACRO_HALO um)"

    add_global_connection -net {VDD} -inst_pattern {.*} -pin_pattern {VDD} -power
    add_global_connection -net {VSS} -inst_pattern {.*} -pin_pattern {VSS} -ground
    global_connect
    set_voltage_domain -name {CORE} -power {VDD} -ground {VSS}
    define_pdn_grid -name {core_grid} -voltage_domains {CORE}

    # PDN: core ring m9/m10, m1 follow-pins, thin m4 straps, m9 stripes
    set pgcrWidth 1.6;  set pgcrSpacing 1.6; set pgcrOffset 2.0
    set tpg1Width 1.6;  set tpg1Pitch 56.0;  set tpg1Spacing 2.0; set tpg1Offset 20.0
    set tpg4Width 0.4;  set tpg4Pitch 56.0;  set tpg4Spacing 0.4; set tpg4Offset 20.0
    add_pdn_ring -grid {core_grid} -layer {metal9 metal10} \
        -widths "$pgcrWidth $pgcrWidth" -spacings "$pgcrSpacing $pgcrSpacing" \
        -core_offsets "$pgcrOffset $pgcrOffset" -add_connect
    add_pdn_stripe -grid {core_grid} -layer {metal1} -width 0.17 -offset 0 -followpins -extend_to_core_ring
    add_pdn_stripe -grid {core_grid} -layer {metal4} -width $tpg4Width \
        -pitch $tpg4Pitch -spacing $tpg4Spacing -offset $tpg4Offset -extend_to_core_ring -snap_to_grid
    add_pdn_stripe -grid {core_grid} -layer {metal9} -width $tpg1Width \
        -pitch $tpg1Pitch -spacing $tpg1Spacing -offset $tpg1Offset -extend_to_core_ring -snap_to_grid
    add_pdn_connect -grid {core_grid} -layers {metal9 metal4}
    add_pdn_connect -grid {core_grid} -layers {metal4 metal1}
    pdngen

    if {[llength $CLK_NETS]} {
        catch { set_dont_touch [get_nets $CLK_NETS] }
        puts "DONT_TOUCH clk nets: $CLK_NETS"
    }

    remove_buffers
    puts "GP1_START"
    global_placement -density $GP_DENSITY
    estimate_parasitics -placement
    repair_design -verbose
    repair_timing -setup -verbose

    puts "GP2_START"
    set gp2_args [list -density $GP_DENSITY -routability_driven -routability_check_overflow 0.30]
    if {$TIMING_DRIVEN} { lappend gp2_args -timing_driven }
    global_placement {*}$gp2_args
    legalize
    optimize_mirroring
    estimate_parasitics -placement
    report_ppa gp

    # The routability placement moved every cell: repair the wires and setup again on the
    # final locations (without this, long nets stay unbuffered until after CTS).
    puts "POST_GP2_REPAIR"
    repair_design -verbose
    repair_timing -setup -verbose
    legalize
    estimate_parasitics -placement
    report_ppa place

    write_db  ${report_dir}/${design}_placed.odb
    write_def ${report_dir}/${design}_placed.def
}

# ---------------------------------------------------------------------------
# CTS + setup / hold repair on placement parasitics
# ---------------------------------------------------------------------------
if {$si <= 1 && $ei >= 1} {
    set_wire_rc -signal -layer metal3
    set_wire_rc -clock  -layer metal5
    if {[llength $CLK_NETS]} { catch { unset_dont_touch [get_nets $CLK_NETS] } }
    repair_clock_inverters
    clock_tree_synthesis -sink_clustering_enable \
        -buf_list "CLKBUF_X1 CLKBUF_X2 CLKBUF_X3" -root_buf "CLKBUF_X3" -wire_unit 20
    repair_clock_nets
    legalize                    ;# legalize CTS buffers
    estimate_parasitics -placement
    set_propagated_clock [get_clocks core_clock]
    set_vclk_latency
    repair_timing -setup -verbose
    repair_timing -hold -verbose -hold_margin 0.05
    legalize                    ;# legalize repair cells
    catch { check_placement -verbose }
    estimate_parasitics -placement
    report_ppa cts

    write_db      ${report_dir}/${design}_cts.odb
    write_def     ${report_dir}/${design}_cts.def
    write_verilog ${report_dir}/${design}_cts.v
} elseif {$si == 2} {
    set_propagated_clock [get_clocks core_clock]
    estimate_parasitics -placement
    set_vclk_latency
}

# ---------------------------------------------------------------------------
# Global route -> repair on routed parasitics -> re-legalize -> global route -> detailed route
# ---------------------------------------------------------------------------
if {$ei >= 2} {
    set_global_routing_layer_adjustment metal2-metal10 0.3
    set_routing_layers -signal metal2-metal8 -clock metal3-metal7
    # -allow_congestion: plain global_route aborts (GRT-0116) on residual overflow that
    # detailed_route can resolve.
    global_route -guide_file ${report_dir}/${design}.guide \
                 -congestion_iterations $GRT_ITERS -verbose -allow_congestion
    estimate_parasitics -global_routing
    report_ppa grt1

    repair_design
    repair_timing -setup -verbose
    repair_timing -hold -verbose -hold_margin 0.05
    legalize                    ;# legalize repair cells
    # route the inserted buffers
    global_route -guide_file ${report_dir}/${design}.guide \
                 -congestion_iterations $GRT_ITERS -verbose -allow_congestion
    estimate_parasitics -global_routing
    report_ppa grt
    write_db ${report_dir}/${design}_grt.odb

    detailed_route -output_drc ${report_dir}/${design}.drc.rpt \
                   -droute_end_iter 20 -verbose 1
    estimate_parasitics -global_routing
    report_ppa route

    write_db      ${report_dir}/${design}_routed.odb
    write_def     ${report_dir}/${design}_routed.def
    write_verilog ${report_dir}/${design}_routed.v
    catch { write_sdc ${report_dir}/${design}_routed.sdc }
}
puts "=== FLOW DONE (cand=$CAND stop=$STOP) ==="
