# Floorplan: 60% utilization, aspect ratio 1, I/O pins on the left, bottom and right edges
# (the top edge stays free for macros).  run.sh floorplan
# Outputs in the work directory: floorplan.odb, floorplan.def; the log lists the die, the core
# and the macros (candidates/gen_candidates.py reads the core box from it).
source $::env(OL_DIR)/scripts/common.tcl
set FP_UTIL   60
set FP_ASPECT 1.0
set FP_SPACE  20

load_lefs
load_libs
read_verilog $NETLIST
link_design $TOP
read_sdc $SDC_FILE
set_rc

initialize_floorplan -utilization $FP_UTIL -aspect_ratio $FP_ASPECT -core_space $FP_SPACE -site $FP_SITE
source $PLATFORM_DIR/make_tracks.tcl
place_pins -hor_layers $PIN_HLAYER -ver_layers $PIN_VLAYER -exclude top:* -min_distance 0.5

write_db  $WORK/floorplan.odb
write_def $WORK/floorplan.def
report_design_area
report_die_core
report_macros
