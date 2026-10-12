# OpenROAD init / read-in stage (parameterized — rendered per design by
# macroflow.config.FlowConfig, which prepends data_dir/proj_name/top_design/
# save_dir/report_dir/fp_def). Standalone defaults are guarded below.
#
# The rewritten workflow places macros into a PREDEFINED floorplan DEF
# (``fp_def``, e.g. def/2_1_floorplan.def) rather than building a
# floorplan from def/pre_place.def. See Instructions/Get_claude_results.md.

# Configurable inputs (overridden by the rendered header; defaults for standalone runs)
if {![info exists data_dir]}   { set data_dir "Benchmark/PlaceReasoner-Bench/vga_lcd" }
if {![info exists proj_name]}  { set proj_name "vga_lcd" }
if {![info exists top_design]} { set top_design "vga_enh_top" }
if {![info exists save_dir]}   { set save_dir ${data_dir}/saves_db }
if {![info exists report_dir]} { set report_dir ${data_dir}/report }
if {![info exists fp_def]}     { set fp_def ${data_dir}/def/2_1_floorplan.def }
if {![info exists tech_dir]}   { set tech_dir ${data_dir} }

# PDK (std cells + tech LEF) from tech_dir; design macros from data_dir.
set pdk_cells_lib ${tech_dir}/lib
set pdk_cells_lef ${tech_dir}/lef
set pdk_sram_lib  ${data_dir}/lib
set pdk_sram_lef  ${data_dir}/lef

# LIB / LEF — std cells first (sets the unit system), then tech/cell LEF.
define_corners tt

puts "Init standard cells"
read_liberty -corner tt ${pdk_cells_lib}/NangateOpenCellLibrary_typical.lib

puts "Init tech-lef"
read_lef ${pdk_cells_lef}/NangateOpenCellLibrary.tech.lef

puts "Init cell-lef"
read_lef ${pdk_cells_lef}/NangateOpenCellLibrary.macro.mod.lef

# Macro LEFs: read EVERY non-stdcell .lef in the circuit's lef dir, so any
# circuit (fakeram + custom macros like memory_block_*, macro_6x6, ...) loads
# all its block masters. Reading only one family left macros unresolved.
puts "Init macro lef"
foreach _lef [lsort [glob -nocomplain ${pdk_sram_lef}/*.lef]] {
    if {[string match "*NangateOpenCellLibrary*" $_lef]} { continue }
    puts "  read_lef $_lef"
    read_lef $_lef
}

# Macro liberty: same generic glob over the lib dir (skip the stdcell lib).
puts "Init macro liberty"
foreach _lib [lsort [glob -nocomplain ${pdk_sram_lib}/*.lib]] {
    if {[string match "*NangateOpenCellLibrary*" $_lib]} { continue }
    puts "  read_liberty $_lib"
    read_liberty -corner tt $_lib
}

# Netlist defines the modules; the predefined floorplan DEF supplies the die,
# core, rows, IO pins and component placement to start from.
read_verilog ${data_dir}/${proj_name}.v
read_def ${fp_def}

# The predefined floorplan may ship macros PLACED or FIXED/LOCKED. The
# multi-agent flow plans and places ALL block macros itself, so any left
# FIXED here would (a) be obstacles the Python legalizer doesn't know about
# and (b) make place_macro reject movable neighbors that abut them -> unplaced
# macros + overlaps -> RePlAce (GPL) divergence. Unlock them all to movable so
# the planned placement can be realized cleanly; detail_place re-fixes them
# (FIRM) before PDN.
set _nun 0
foreach _i [[ord::get_db_block] getInsts] {
    if {[[$_i getMaster] isBlock]} {
        set _s [$_i getPlacementStatus]
        if {$_s eq "FIRM" || $_s eq "LOCKED"} { $_i setPlacementStatus PLACED; incr _nun }
    }
}
puts "Unlocked $_nun pre-fixed block macros to movable"

read_sdc ${data_dir}/constraint.sdc

# RC setup must come AFTER the netlist is read/linked — newer OpenROAD
# (26Q1+) errors "no network has been linked" if set_wire_rc runs before
# read_verilog/read_def. Liberty units are fF,kOhm.
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

#### save the pristine post-init checkpoint where the flow expects it
file mkdir ${save_dir}
write_db ${save_dir}/floorplan.odb
