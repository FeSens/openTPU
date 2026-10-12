
if {![info exists data_dir]}   { set data_dir "Benchmark/PlaceReasoner-Bench/ariane133" }
if {![info exists proj_name]}  { set proj_name "ariane133" }
if {![info exists save_dir]}   { set save_dir ${data_dir}/saves_db }
if {![info exists report_dir]} { set report_dir ${data_dir}/report }
file mkdir $save_dir
file mkdir $report_dir

###############################################################################
# LIGHT CTS + ROUTE  (flow's final-route stage)
#
# Same structure as cts_route.tcl but DROPS the parts that hang past the client
# timeout on VLM-seeded placements and relaxes global-route capacity. Proven on
# ariane136 + or1200 (see route_or1200_light.tcl / reference-openroad-env memory):
#   - no verbose `repair_timing -setup/-hold` (the unbounded single-threaded
#     passes that never converge on suboptimal placements -> 2hr+ hang),
#   - no `repair_design` + second global_route after timing repair,
#   - global-route capacity 0.3 (was 0.5) and signal layers metal2-metal9
#     (was metal2-metal8) so global route converges with 0 overflow,
#   - detailed_route -droute_end_iter 5 (was 10).
# Deliverable is a fully detail-routed DB (not timing-closed). Sourced IN-SESSION
# after `read_db <best_detail_db>`; liberty/corners/sdc and layer/wire RC already
# established by floorplan.tcl + detail_place.tcl earlier in the same process
# (so we deliberately do NOT re-read them here -- re-defining corners errors).
###############################################################################

set_thread_count 32

###############################################################################
# Stage 03: CLOCK TREE SYNTHESIS
###############################################################################
utl::report "###############################################################################"
utl::report "# Stage 03: CLOCK TREE SYNTHESIS (light)"
utl::report "###############################################################################"

catch { unset_dont_touch [get_nets clk_i] }

utl::report "Repair clock inverters"
repair_clock_inverters

utl::report "Clock Tree Synthesis"
clock_tree_synthesis -sink_clustering_enable \
                     -repair_clock_nets \
                     -buf_list "CLKBUF_X1 CLKBUF_X2 CLKBUF_X3" \
                     -root_buf "CLKBUF_X3" \
                     -wire_unit 20

utl::report "Detailed placement (legalize CTS buffers)"
detailed_placement

utl::report "Estimate parasitics"
estimate_parasitics -placement

utl::report "Propagate clocks"
set_propagated_clock [all_clocks]

# NOTE: verbose `repair_timing -setup/-hold` intentionally dropped here (hangs).
utl::report "Detailed placement (post-CTS)"
detailed_placement

utl::report "CTS report"
report_cts -out_file ${report_dir}/03_${proj_name}.cts.rpt
report_tns
report_wns
report_design_area
write_db ${save_dir}/03_${proj_name}.cts.odb
catch { save_image ${save_dir}/03_${proj_name}_cts.png }
utl::report "CTS_DONE"

###############################################################################
# Stage 04: ROUTING (relaxed capacity, no verbose repair)
###############################################################################
utl::report "###############################################################################"
utl::report "# Stage 04: ROUTING (light)"
utl::report "###############################################################################"

set_thread_count 32

# Relaxed capacity (0.3) + widened signal layers (metal2-metal9) -- proven to
# converge with 0 overflow where the default 0.5 / metal2-metal8 cycled forever.
set_global_routing_layer_adjustment metal2-metal10 0.3
set_routing_layers -signal metal2-metal9 -clock metal3-metal7

utl::report "Global routing"
global_route -guide_file ${report_dir}/04_${proj_name}.guide \
             -congestion_iterations 30 \
             -verbose

utl::report "Estimate parasitics from global route"
estimate_parasitics -global_routing
write_db ${save_dir}/04-02_${proj_name}.global_route.odb
catch { save_image ${save_dir}/04-02_${proj_name}.global_route.png }
utl::report "GLOBAL_ROUTE_DONE"

utl::report "Detailed routing"
# droute_end_iter 8 (was 5): on bp_fe the route was still converging at iter 5
# (4879 -> 150 -> 5 violations); a few more opt iterations clear the residual
# shorts. Each extra iteration is cheap (~10-30s) relative to leaving DRCs.
detailed_route -output_drc      ${report_dir}/04_${proj_name}.drc.rpt \
               -output_maze     ${report_dir}/04_${proj_name}.maze.log \
               -droute_end_iter 8 \
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

catch {
    set drc_file ${report_dir}/04_${proj_name}.drc.rpt
    if {[file exists $drc_file]} {
        utl::report "DRC report: $drc_file ([exec wc -l < $drc_file] lines)"
    }
}

write_db ${save_dir}/04_${proj_name}.routed.odb
catch { save_image ${save_dir}/04_${proj_name}.routed.png }

utl::report "###############################################################################"
utl::report "# Stage 04 complete: Design fully routed (light)"
utl::report "###############################################################################"
write_db ${save_dir}/routed.odb
utl::report "ROUTE_ALL_DONE"
