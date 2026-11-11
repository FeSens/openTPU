# EXPERIMENTAL (otpu_ppk.sv): out-of-context synthesis + implementation of the pumped packed MXU
# test (ppk_top) on the BCU1525's part, its clocks from an MMCME4_ADV. Vivado with a VU9P license:
#   vivado -mode batch -nojournal -source rtl/exp/pump_us/ooc.tcl -tclargs <out> [K=V ...]
# K: NCOL 1 (one column: 128 + 36 DSP48E2s) or 8 (the array: 1,024 + 288), BF 1 (the activation's
# fast register in the fabric; 0: the DSP's B1), HOLD 0 (1: every chain end held into clk_s over two
# fast cycles), PBLOCK none | SLR0 | SLR1 | SLR2 | CLOCKREGION_XaYb:CLOCKREGION_XcYd, PART
# xcvu9p-fsgd2104-2L-e, CLKIN_NS 3.333 / MULT 4.0 / DIV_S 8.0 / DIV_F 4 (VCO 1200: clk_s 150 MHz,
# clk_f 300 MHz), PLACE / PHYS / ROUTE directives (Default). Prints OTPU_* summary lines; reports
# go to <out>.
set out [file normalize [lindex $argv 0]]
array set o {NCOL 1 BF 1 HOLD 0 PBLOCK none PART xcvu9p-fsgd2104-2L-e CLKIN_NS 3.333 MULT 4.0
             DIV_S 8.0 DIV_F 4 PLACE Default PHYS Default ROUTE Default}
foreach kv [lrange $argv 1 end] {
  set i [string first = $kv]
  set k [string range $kv 0 [expr {$i - 1}]]
  if {$i < 1 || ![info exists o($k)]} { error "ooc.tcl: unknown argument $kv" }
  set o($k) [string range $kv [expr {$i + 1}] end]
}
set root [file normalize [file join [file dirname [info script]] ../../..]]
file mkdir $out
foreach k [lsort [array names o]] { puts "OTPU_ARG $k $o($k)" }

create_project -in_memory -part $o(PART)
set_property source_mgmt_mode None [current_project]
read_verilog -sv [list $root/rtl/vpu/otpu_fp.sv $root/rtl/top/otpu_pkg.sv \
  $root/rtl/vpu/otpu_fpipe.sv $root/rtl/mxu/otpu_mxu.sv $root/rtl/tech/xilinxus/otpu_tech_dsp.sv \
  $root/rtl/exp/pump_us/otpu_ppk.sv $root/rtl/exp/pump_us/ppk_top.sv]
synth_design -top ppk_top -part $o(PART) -mode out_of_context -flatten_hierarchy rebuilt \
  -verilog_define SYNTHESIS -generic NCOL=$o(NCOL) -generic BF=$o(BF) -generic HOLD=$o(HOLD) \
  -generic CLKIN_NS=$o(CLKIN_NS) -generic MULT=$o(MULT) -generic DIV_S=$o(DIV_S) \
  -generic DIV_F=$o(DIV_F)
create_clock -period $o(CLKIN_NS) -name clk_in [get_ports clk_in]
set cs [get_clocks -quiet -of_objects [get_pins u_mmcm/CLKOUT0]]
set cf [get_clocks -quiet -of_objects [get_pins u_mmcm/CLKOUT1]]
foreach {t c} [list slow $cs fast $cf] {
  if {![llength $c]} { error "ooc.tcl: no $t clock" }
  puts "OTPU_CLOCK $t [get_property NAME $c] [get_property PERIOD $c]"
}
set ratio [expr {int(round(double($o(DIV_S)) / $o(DIV_F)))}]

# the pumped DSPs' weight pins (A, D) load only on the slow edges (CEA2 / CED = pe): the slow
# registers and the decode in front of them have the whole slow period. (Pins by name: in a
# -filter pattern [..] is a glob class.) B is not among them: at BF 0 the select into B1 is a
# one-fast-period path, as b_f's at BF 1.
set ppe [get_cells -hierarchical -filter {REF_NAME == DSP48E2 && NAME =~ *u_ppe*}]
set pins {}
foreach p [get_pins -of_objects $ppe -filter {DIRECTION == IN && (REF_PIN_NAME =~ A* || REF_PIN_NAME =~ D*)}] {
  if {[regexp {^(A|D)\[[0-9]+\]$} [get_property REF_PIN_NAME $p]]} { lappend pins $p }
}
puts "OTPU_PPE [llength $ppe] OTPU_MCP_PINS [llength $pins]"
if {[llength $pins]} {
  set_multicycle_path $ratio -setup -end -from $cs -to $pins
  set_multicycle_path [expr {$ratio - 1}] -hold -end -from $cs -to $pins
}
# HOLD 1: the chain ends' holding registers load on the slow edges (pe) and the column ends take
# them a slow period later
set hc {}
if {$o(HOLD)} {
  set hc [get_cells -hierarchical -filter {IS_SEQUENTIAL && (NAME =~ *hold0_reg* || NAME =~ *hold1e_reg*)}]
  puts "OTPU_MCP_HOLD [llength $hc]"
  if {[llength $hc]} {
    set_multicycle_path $ratio -setup -start -from $hc -to $cs
    set_multicycle_path [expr {$ratio - 1}] -hold -start -from $hc -to $cs
  }
}
if {$o(PBLOCK) ne "none"} {
  set pb [create_pblock pb_ppk]
  add_cells_to_pblock $pb -top
  resize_pblock $pb -add $o(PBLOCK)
  puts "OTPU_PBLOCK $o(PBLOCK)"
}
report_utilization -file $out/util_synth.txt
opt_design
place_design -directive $o(PLACE)
phys_opt_design -directive $o(PHYS)
route_design -directive $o(ROUTE)
write_checkpoint -force $out/routed.dcp
report_utilization -file $out/util.txt
report_utilization -hierarchical -hierarchical_depth 4 -file $out/util_hier.txt
report_timing_summary -max_paths 20 -file $out/timing_summary.txt
report_clock_interaction -file $out/clock_interaction.txt
report_pulse_width -file $out/pulse_width.txt
report_exceptions -file $out/exceptions.txt
report_methodology -file $out/methodology.txt
report_drc -file $out/drc.txt
report_power -file $out/power.txt

set dsp [llength [get_cells -hierarchical -filter {REF_NAME == DSP48E2}]]
set lut [llength [get_cells -hierarchical -filter {REF_NAME =~ LUT*}]]
set ff  [llength [get_cells -hierarchical -filter {IS_SEQUENTIAL && REF_NAME =~ FD*}]]
set srl [llength [get_cells -hierarchical -filter {REF_NAME =~ SRL*}]]
puts "OTPU_UTIL DSP48E2 $dsp LUT $lut FF $ff SRL $srl"
# the worst path from / to (an empty list: any), one line, its 10 worst in timing_<tag>.txt. The
# collections go in as they are (a collection in a list expanded with {*} is cut at 500)
proc otpu_path {tag from to} {
  global out
  if {[llength $from] && [llength $to]} {
    set q [get_timing_paths -quiet -from $from -to $to -setup -max_paths 10 -nworst 1]
    set h [get_timing_paths -quiet -from $from -to $to -hold -max_paths 1 -nworst 1]
  } elseif {[llength $from]} {
    set q [get_timing_paths -quiet -from $from -setup -max_paths 10 -nworst 1]
    set h [get_timing_paths -quiet -from $from -hold -max_paths 1 -nworst 1]
  } else {
    set q [get_timing_paths -quiet -to $to -setup -max_paths 10 -nworst 1]
    set h [get_timing_paths -quiet -to $to -hold -max_paths 1 -nworst 1]
  }
  if {![llength $q]} { puts "OTPU_PATH $tag none"; return }
  set w [lindex $q 0]
  puts "OTPU_PATH $tag setup [get_property SLACK $w] req [get_property REQUIREMENT $w] levels [get_property LOGIC_LEVELS $w] [get_property STARTPOINT_PIN $w] -> [get_property ENDPOINT_PIN $w] | hold [get_property SLACK [lindex $h 0]]"
  report_timing -of_objects $q -file $out/timing_$tag.txt
}
# by clock pair, then by the parts of the pumped schedule
otpu_path SS $cs $cs
otpu_path FF $cf $cf
otpu_path SF $cs $cf
otpu_path FS $cf $cs
set tog [get_cells -hierarchical -filter {IS_SEQUENTIAL && (NAME =~ *pe_t_reg* || NAME =~ *po_t_reg* || NAME =~ *pe_h_reg* || NAME =~ *po_h_reg*)}]
set bf  [get_cells -quiet -hierarchical -filter {IS_SEQUENTIAL && NAME =~ *b_f_reg*}]
set rf  [get_cells -hierarchical -filter {IS_SEQUENTIAL && NAME =~ *rst_f_reg*}]
set ce  [get_cells -hierarchical -filter {REF_NAME == DSP48E2 && NAME =~ *u_ce*}]
otpu_path PHASE $tog {}
if {[llength $bf]} {
  otpu_path ACT_SF $cs $bf
  otpu_path ACT_FF $bf {}
} else {
  otpu_path ACT_SF $cs $ppe
}
if {[llength $pins]} { otpu_path WGT $cs $pins }
otpu_path DSP $ppe $ppe
otpu_path RET $cf $ce
otpu_path RST {} $rf
set p [lindex [get_timing_paths -setup -max_paths 1 -nworst 1] 0]
set h [lindex [get_timing_paths -hold -max_paths 1 -nworst 1] 0]
set fails [llength [get_timing_paths -quiet -setup -max_paths 100000 -slack_lesser_than 0]]
puts "OTPU_WNS [get_property SLACK $p] OTPU_WHS [get_property SLACK $h] OTPU_FAILING_ENDPOINTS_MAX100K $fails"
puts "OTPU_DONE"
