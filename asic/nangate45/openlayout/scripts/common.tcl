# Shared setup of the OpenLayout-style flow (asic/nangate45/openlayout/README.md): floorplan ->
# hand-planned macro placement candidates -> placement -> CTS -> route. Sourced by the other
# scripts; run.sh sets the environment:
#   OL_DIR        asic/nangate45/openlayout
#   OL_WORK       the work directory (build/asic/small-openlayout)
#   OL_MACROS     the macros' LEF / Liberty (build/asic/small/macros, tools/asic/flow.py prep)
#   PLATFORM_DIR  OpenROAD-flow-scripts' flow/platforms/nangate45
# Do not edit a script while a run uses it: OpenROAD reads scripts incrementally.
set OL_DIR       $::env(OL_DIR)
set WORK         $::env(OL_WORK)
set MACRO_DIR    $::env(OL_MACROS)
set PLATFORM_DIR $::env(PLATFORM_DIR)
set TOP          otpu_asic_top
set NETLIST      $WORK/synth/${TOP}_synth.v
set SDC_FILE     $OL_DIR/constraint.sdc
if {[info exists ::env(OL_SDC)] && $::env(OL_SDC) ne ""} { set SDC_FILE $::env(OL_SDC) }

set FP_SITE      FreePDK45_38x28_10R_NP_162NW_34O
set PIN_HLAYER   metal5      ;# IO_PLACER_H / IO_PLACER_V of the nangate45 platform
set PIN_VLAYER   metal6
set DPL_MAXDISP    {300 300}
set DPL_RETRY_DISP {2000 2000}

set THREADS 8
if {[info exists ::env(OL_THREADS)] && $::env(OL_THREADS) ne ""} { set THREADS $::env(OL_THREADS) }
set_thread_count $THREADS

proc load_lefs {} {
    global PLATFORM_DIR MACRO_DIR
    read_lef $PLATFORM_DIR/lef/NangateOpenCellLibrary.tech.lef
    read_lef $PLATFORM_DIR/lef/NangateOpenCellLibrary.macro.mod.lef
    foreach f [lsort [glob -nocomplain $MACRO_DIR/*.lef]] { read_lef $f }
}

proc load_libs {} {
    global PLATFORM_DIR MACRO_DIR
    define_corners tt
    read_liberty -corner tt $PLATFORM_DIR/lib/NangateOpenCellLibrary_typical.lib
    foreach f [lsort [glob -nocomplain $MACRO_DIR/*.lib]] { read_liberty -corner tt $f }
}

proc set_rc {} {
    set_layer_rc -layer metal1 -resistance 5.4286e-03 -capacitance 7.41819E-02
    set_layer_rc -layer metal2 -resistance 3.5714e-03 -capacitance 6.74606E-02
    set_layer_rc -layer metal3 -resistance 3.5714e-03 -capacitance 8.88758E-02
    set_layer_rc -layer metal4 -resistance 1.5000e-03 -capacitance 1.07121E-01
    set_layer_rc -layer metal5 -resistance 1.5000e-03 -capacitance 1.08964E-01
    set_layer_rc -layer metal6 -resistance 1.5000e-03 -capacitance 1.02044E-01
    set_layer_rc -layer metal7 -resistance 1.8750e-04 -capacitance 1.10436E-01
    set_layer_rc -layer metal8 -resistance 1.8750e-04 -capacitance 9.69714E-02
    set_wire_rc -signal -layer metal3
    set_wire_rc -clock  -layer metal5
}

# Die / core boxes (skill get_die_core_info)
proc report_die_core {} {
    set block [ord::get_db_block]
    set dbu   [[ord::get_db_tech] getDbUnitsPerMicron]
    set die   [$block getDieArea]
    set core  [$block getCoreArea]
    puts [format "Die  : (%.3f, %.3f) - (%.3f, %.3f) um" [expr {[$die xMin]/double($dbu)}] \
        [expr {[$die yMin]/double($dbu)}] [expr {[$die xMax]/double($dbu)}] [expr {[$die yMax]/double($dbu)}]]
    puts [format "Core : (%.3f, %.3f) - (%.3f, %.3f) um" [expr {[$core xMin]/double($dbu)}] \
        [expr {[$core yMin]/double($dbu)}] [expr {[$core xMax]/double($dbu)}] [expr {[$core yMax]/double($dbu)}]]
}

# Macros with their current location (skill get_macro_current_location): the #N is the
# label on the pin-side image made from the same ODB.
proc report_macros {} {
    set block [ord::get_db_block]
    set dbu   [[ord::get_db_tech] getDbUnitsPerMicron]
    set idx 0
    foreach inst [$block getInsts] {
        set master [$inst getMaster]
        if {[$master isBlock]} {
            incr idx
            set bb [$inst getBBox]
            puts [format "#%-3d %-70s master=%-24s size=%.3f x %.3f um  origin=(%.3f, %.3f)  orient=%s" \
                $idx [$inst getName] [$master getName] \
                [expr {[$master getWidth]/double($dbu)}] [expr {[$master getHeight]/double($dbu)}] \
                [expr {[$bb xMin]/double($dbu)}] [expr {[$bb yMin]/double($dbu)}] [$inst getOrient]]
        }
    }
}
