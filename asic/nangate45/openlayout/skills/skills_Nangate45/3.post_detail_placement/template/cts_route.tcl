if {![info exists data_dir]}   { set data_dir [expr {[info exists ::env(DESIGN_DIR)] ? $::env(DESIGN_DIR) : "./data"}] }
if {![info exists proj_name]}  { set proj_name "ariane133" }
if {![info exists save_dir]}   { set save_dir ${data_dir}/saves_db }
if {![info exists report_dir]} { set report_dir ${data_dir}/report }
file mkdir $save_dir
file mkdir $report_dir

###############################################################################
# TIMING CTS + ROUTE  (flow's final-route stage, route_mode=timing)
#
# Same structure/parameters as cts_route_light.tcl, but KEEPS the timing-repair
# passes the light template drops (Get_claude_results.md: "Don't omit the timing
# repair"). Mirrors the proven u65_rtlmp.tcl sequence:
#   - repair_timing -setup / -hold after CTS (placement parasitics),
#   - repair_design + repair_timing after global route (accurate parasitics),
#     then a second global_route so inserted buffers get routed,
#   - detailed_route to a clean DB.
# Default margins only (no REPAIR_SETUP_MARGIN tuning) to stay fair. Sourced
# IN-SESSION after `read_db <best_detail_db>`; liberty/corners/sdc + layer/wire
# RC are already established (so we do NOT re-read them -- re-defining corners
# errors). DRC report path 04_${proj_name}.drc.rpt matches route_finalize.
###############################################################################

set_thread_count 32

###############################################################################
# Stage 03: CLOCK TREE SYNTHESIS
###############################################################################
utl::report "# Stage 03: CLOCK TREE SYNTHESIS (timing)"

# Unlock the clock-source nets (dont_touch'd during detail placement so they
# weren't merged) so CTS can build their trees. Without it an output-forwarded
# clock like vga_lcd's clk_p_o triggers CTS-0114 "overlaps a previous clock".
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
if {[llength $CLK_NETS]} { catch { unset_dont_touch [get_nets $CLK_NETS] } }

utl::report "Repair clock inverters"
repair_clock_inverters

utl::report "Clock Tree Synthesis"
clock_tree_synthesis -sink_clustering_enable \
                     -repair_clock_nets \
                     -buf_list "CLKBUF_X1 CLKBUF_X2 CLKBUF_X3" \
                     -root_buf "CLKBUF_X3" \
                     -wire_unit 20

utl::report "Detailed placement (legalize CTS buffers)"
catch { detailed_placement }

utl::report "Estimate parasitics"
estimate_parasitics -placement

utl::report "Propagate clocks"
set_propagated_clock [all_clocks]

# Timing repair on placement parasitics (default margins). Bounded so a poor
# (e.g. Gate-A-failing) placement cannot trigger an unbounded buffer-insertion
# runaway that explodes area/runtime and OOM-crashes OpenROAD. The bounds only
# LIMIT repair effort (never inflate margins to force a pass) and are identical
# for every candidate, so the flow stays fair.
set REPAIR_BOUNDS "-max_buffer_percent 20 -max_utilization 90 -max_passes 50"
utl::report "Repair timing (post-CTS, placement parasitics)"
catch { repair_timing -setup -verbose {*}$REPAIR_BOUNDS }
catch { repair_timing -hold -verbose -hold_margin 0.05 {*}$REPAIR_BOUNDS }
catch { detailed_placement }
estimate_parasitics -placement

utl::report "CTS report"
report_cts -out_file ${report_dir}/03_${proj_name}.cts.rpt
report_tns
report_wns
report_design_area
write_db ${save_dir}/03_${proj_name}.cts.odb
utl::report "CTS_DONE"

###############################################################################
# Stage 04: ROUTING (with post-route timing repair)
###############################################################################
utl::report "# Stage 04: ROUTING (timing)"

set_thread_count 32
set_global_routing_layer_adjustment metal2-metal10 0.3
set_routing_layers -signal metal2-metal9 -clock metal3-metal7

utl::report "Global routing"
global_route -guide_file ${report_dir}/04_${proj_name}.guide \
             -congestion_iterations 30 \
             -verbose

utl::report "Estimate parasitics from global route"
estimate_parasitics -global_routing
utl::report "GLOBAL_ROUTE_DONE"

# Post-global-route timing repair on accurate (routed-estimate) parasitics.
utl::report "Repair design + timing (post-global-route)"
catch { repair_design -max_utilization 90 }
catch { repair_timing -setup -verbose {*}$REPAIR_BOUNDS }
catch { repair_timing -hold -verbose -hold_margin 0.05 {*}$REPAIR_BOUNDS }
catch { detailed_placement }
# Re-run global route so the inserted timing buffers get routed.
utl::report "Global routing (post-repair)"
global_route -guide_file ${report_dir}/04_${proj_name}.guide \
             -congestion_iterations 30 \
             -verbose
estimate_parasitics -global_routing
write_db ${save_dir}/04-02_${proj_name}.global_route.odb

utl::report "Detailed routing"
catch { force_constant_nets_to_signal }
detailed_route -output_drc      ${report_dir}/04_${proj_name}.drc.rpt \
               -output_maze     ${report_dir}/04_${proj_name}.maze.log \
               -droute_end_iter 10 \
               -verbose 1
write_db ${save_dir}/04-03_${proj_name}.detailed_route.odb

utl::report "Extract parasitics (post-route)"
estimate_parasitics -global_routing

###############################################################################
# Reports + checkpoint
###############################################################################
utl::report "Final timing reports"
report_clock_skew    > ${report_dir}/04_${proj_name}.clock_skew.rpt
report_checks -path_delay max -group_count 10 > ${report_dir}/04_${proj_name}.setup.rpt
report_checks -path_delay min -group_count 10 > ${report_dir}/04_${proj_name}.hold.rpt
report_tns
report_wns
report_worst_slack -max
report_worst_slack -min
report_design_area
report_power

catch {
    set drc_file ${report_dir}/04_${proj_name}.drc.rpt
    if {[file exists $drc_file]} {
        utl::report "DRC report: $drc_file ([exec wc -l < $drc_file] lines)"
    }
}

write_db ${save_dir}/04_${proj_name}.routed.odb
catch { save_image ${save_dir}/04_${proj_name}.routed.png }
write_db ${save_dir}/routed.odb
utl::report "ROUTE_ALL_DONE"
