# Baseline macro placement: OpenROAD's rtl_macro_placer with its defaults (no halo or other
# options) on the same floorplan as the planned candidates.  run.sh baseline
# Output: <work>/rtlmp/macro_placed.odb/.def, macro_placement.tcl, mpl/ (reports)
source $::env(OL_DIR)/scripts/common.tcl
set CAND rtlmp
read_db $WORK/floorplan.odb
load_libs
read_sdc $SDC_FILE
set_rc

rtl_macro_placer -report_directory $WORK/$CAND/mpl \
                 -write_macro_placement $WORK/$CAND/macro_placement.tcl

write_db  $WORK/$CAND/macro_placed.odb
write_def $WORK/$CAND/macro_placed.def
report_die_core
report_macros
