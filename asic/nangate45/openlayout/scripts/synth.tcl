# Yosys synthesis of openTPU's ASIC block (otpu_asic_top, design point nangate45-small) on
# NanGate45: OpenROAD-flow-scripts' flat synthesis (scripts/synth.tcl with the nangate45 platform
# settings) as one standalone script. The SRAM macros (tools/asic/fakeram.py) stay black boxes.
#
#   run.sh synth        (yosys -c synth.tcl; Yosys 0.40+ for dfflibmap -dont_use)
#
# Environment: PLATFORM_DIR (ORFS flow/platforms/nangate45), OL_RTLIL (the elaborated design),
# OL_SYNTH_OUT (output directory), CLOCK_PERIOD_PS (ABC's target), EXPECTED_MACROS.
# Outputs: <TOP>_synth.v, synth_stat.txt, synth_check.txt, macros.txt

yosys -import

set TOP              otpu_asic_top
set RTLIL            $::env(OL_RTLIL)
set OUT_DIR          $::env(OL_SYNTH_OUT)
set CLOCK_PERIOD_PS  3300                  ;# ABC timing target, in picoseconds
set ABC_AREA         0                     ;# 0 = speed recipe (ORFS default), 1 = area recipe
set USE_ADDER_MAP    1                     ;# map full/half adders to nangate45 FA_X1/HA_X1
set EXPECTED_MACROS  27                    ;# -1 = no check
foreach var {CLOCK_PERIOD_PS ABC_AREA USE_ADDER_MAP EXPECTED_MACROS} {
  if { [info exists ::env($var)] && $::env($var) ne "" } {
    set $var $::env($var)
  }
}

# ---------------------------------------------------------------------------
# PLATFORM: nangate45 (from flow/platforms/nangate45/config.mk)
# ---------------------------------------------------------------------------
set PLATFORM_DIR $::env(PLATFORM_DIR)
set COMMON_DIR   [file join [file dirname $PLATFORM_DIR] common]

set LIB_FILE       $PLATFORM_DIR/lib/NangateOpenCellLibrary_typical.lib
set LATCH_MAP_FILE $PLATFORM_DIR/cells_latch.v
set ADDER_MAP_FILE $PLATFORM_DIR/cells_adders.v
set LCU_MAP_FILE   $COMMON_DIR/lcu_kogge_stone.v

set TIEHI_CELL_AND_PORT    {LOGIC1_X1 Z}
set TIELO_CELL_AND_PORT    {LOGIC0_X1 Z}
set MIN_BUF_CELL_AND_PORTS {BUF_X1 A Z}
set ABC_DRIVER_CELL        BUF_X1
set ABC_LOAD_IN_FF         3.898

# Don't-use cells to ease congestion
set DONT_USE_CELLS {
  TAPCELL_X1 FILLCELL_X1 AOI211_X1 OAI211_X1
}

# ---------------------------------------------------------------------------
# ABC recipes (flow/scripts/abc_speed.script, abc_area.script)
# ---------------------------------------------------------------------------
set ABC_SPEED_RECIPE {
&get -n
&st
&dch
&nf
&st
&syn2
&if -g -K 6
&synch2
&nf
&st
&syn2
&if -g -K 6
&synch2
&nf
&st
&syn2
&if -g -K 6
&synch2
&nf
&st
&syn2
&if -g -K 6
&synch2
&nf
&st
&syn2
&if -g -K 6
&synch2
&nf
&put
buffer -c
topo
stime -c
upsize -c
dnsize -c
}

set ABC_AREA_RECIPE {
strash
dch
map -B 0.9
topo
stime -c
buffer -c
upsize -c
dnsize -c
}

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
file mkdir $OUT_DIR

proc write_file { path text } {
  set fp [open $path w]
  puts $fp [string trim $text]
  close $fp
}

set abc_script $OUT_DIR/abc.script
write_file $abc_script [expr { $ABC_AREA ? $ABC_AREA_RECIPE : $ABC_SPEED_RECIPE }]
write_file $OUT_DIR/abc.constr \
  "set_driving_cell $ABC_DRIVER_CELL\nset_load $ABC_LOAD_IN_FF"

set lib_args [list -liberty $LIB_FILE]
set dont_use_args [list]
foreach cell $DONT_USE_CELLS {
  lappend dont_use_args -dont_use $cell
}

# ---------------------------------------------------------------------------
# Read design (synth_stdcells.tcl + synth_canonicalize.tcl)
# ---------------------------------------------------------------------------
read_liberty -overwrite -setattr liberty_cell -lib $LIB_FILE

# The design: RTLIL from yosys-slang (run.sh synth elaborates rtl/asic/otpu_asic_top.sv with the
# point's parameters; the fakeram45_1r1w_* macros are black boxes in it)
read_rtlil $RTLIL
hierarchy -check -top $TOP
# FPGA ram_style attributes mean nothing here
setattr -unset ram_style m:*
setattr -unset ram_style t:\$mem*
# Same as ORFS canonicalize; its "contains processes" warnings are harmless
opt_clean -purge

# ---------------------------------------------------------------------------
# Generic synthesis (synth.tcl)
# ---------------------------------------------------------------------------
set synth_args [list -extra-map $LCU_MAP_FILE]

# Coarse-level synthesis, flattened right away
synth -flatten -run :fine {*}$synth_args

# The remaining memories (small, asynchronous read: the dispatch window, the MXU's drain rows,
# ...) become flip-flops in the fine stage; the large ones are the macros.

# Fine-grained synthesis
synth -top $TOP -run fine: -noabc {*}$synth_args

# Get rid of indigestibles
chformal -remove
delete t:\$print
delete t:\$check

# Name registers after their Verilog reg (`renames` is Yosys' `rename` in Tcl mode)
renames -wire

opt -purge

# ---------------------------------------------------------------------------
# Technology mapping
# ---------------------------------------------------------------------------
if { $USE_ADDER_MAP } {
  extract_fa
  techmap -map $ADDER_MAP_FILE
  techmap
  opt -fast -purge
}

techmap -map $LATCH_MAP_FILE

dfflibmap {*}$lib_args {*}$dont_use_args
opt

setundef -zero

abc -script $abc_script {*}$lib_args {*}$dont_use_args \
  -constr $OUT_DIR/abc.constr -D $CLOCK_PERIOD_PS

# Resolve compound assign statements in the netlist (assign {..} = {..})
splitnets
opt_clean -purge

hilomap -singleton -hicell {*}$TIEHI_CELL_AND_PORT -locell {*}$TIELO_CELL_AND_PORT

# Buffer pass-through wires
insbuf -buf {*}$MIN_BUF_CELL_AND_PORTS

# ---------------------------------------------------------------------------
# Macros
# ---------------------------------------------------------------------------
set macro_sel [list t:fakeram45_*]
tee -q -o $OUT_DIR/macros.txt select -list {*}$macro_sel
if { $EXPECTED_MACROS >= 0 } {
  select -assert-count $EXPECTED_MACROS {*}$macro_sel
}

# ---------------------------------------------------------------------------
# Reports and output
# ---------------------------------------------------------------------------
tee -o $OUT_DIR/synth_check.txt check
tee -o $OUT_DIR/synth_stat.txt stat {*}$lib_args

# Fail if anything is left that is not a nangate45 cell or a macro
check -assert -mapped

# -noattr: OpenLayout's dataflow parser reads plain structural Verilog
write_verilog -noattr -nohex -nodec $OUT_DIR/${TOP}_synth.v

