# EXPERIMENTAL (docs/tmem_pump.md): out-of-context synthesis + implementation of the TMEM
# experiment's top (tmem_pump_top) with its clocks from an MMCM. Vivado on the build host only:
#   vivado -mode batch -nojournal -source rtl/exp/tmem_pump/ooc.tcl -tclargs <out> <VAR> \
#     <WORDS> <PART> <FAMILY> <STYLE> <CLKIN_NS> <MULT> <DIV_S> <DIV_F> [PK] [CASC]
# VAR: tmem_pump_top's (0 otpu_tmem IMPL 0, 1 IMPL 1, 2 pumped IMPL 0, 3 pumped NCP 4, 4 pumped
# NCP 2, 5 one copy pumped at PK x, PK 3 or 4). FAMILY 0: 7 series (MMCME2), 1: UltraScale+
# (MMCME4). STYLE: the pumped banks' ram_style (block / ultra). Prints OTPU_* summary lines;
# reports go to <out>.
set out [file normalize [lindex $argv 0]]
lassign [lrange $argv 1 end] var words part family style clkin mult div_s div_f pk casc
if {$pk eq ""} { set pk 3 }
if {$casc eq ""} { set casc 0 }
set root [file normalize [file join [file dirname [info script]] ../../..]]
file mkdir $out

create_project -in_memory -part $part
set_property source_mgmt_mode None [current_project]
read_verilog -sv [list $root/rtl/tech/generic/otpu_tech_ram.sv $root/rtl/mem/otpu_tmem.sv \
  $root/rtl/exp/tmem_pump/otpu_tmem_pump.sv $root/rtl/exp/tmem_pump/tmem_pump_top.sv]
synth_design -top tmem_pump_top -part $part -mode out_of_context -flatten_hierarchy rebuilt \
  -verilog_define SYNTHESIS -generic VAR=$var -generic WORDS=$words -generic FAMILY=$family \
  -generic STYLE=\"$style\" -generic CLKIN_NS=$clkin -generic MULT=$mult -generic DIV_S=$div_s \
  -generic DIV_F=$div_f -generic PK=$pk -generic CASC=$casc
create_clock -period $clkin -name clk_in [get_ports clk_in]
set cs [get_clocks -quiet -of_objects [get_pins -hier -filter {NAME =~ *u_mmcm/CLKOUT0}]]
set cf [get_clocks -quiet -of_objects [get_pins -hier -filter {NAME =~ *u_mmcm/CLKOUT1}]]
foreach {t c} [list slow $cs fast $cf] {
  if {[llength $c]} { puts "OTPU_CLOCK $t [get_property NAME $c] [get_property PERIOD $c]" }
}
# The pumped banks' contract (otpu_tmem_pump): their read inputs are sampled on the even fast edges
# only (a whole core period from clk_s), and their RAM outputs change only after even edges (a
# whole core period into clk_s). The write stage and the phase reach the RAMs in one fast period.
# With FW (the default) the write stage's registers are on clk_f, loaded at even edges only: their
# inputs from clk_s get a core period too (the phase registers, outside pw_*, keep one).
set rams [get_cells -quiet -hierarchical -filter {REF_NAME =~ RAMB* || REF_NAME =~ URAM*}]
set rp [get_pins -quiet -hierarchical -filter {NAME =~ *u_pb/ra[*] || NAME =~ *u_pb/rb[*] || NAME =~ *u_pb/rena || NAME =~ *u_pb/renb}]
set pwf {}
foreach c [get_cells -quiet -hierarchical -filter {NAME =~ *u_tmem/*pw_* && IS_SEQUENTIAL}] {
  set ck [get_clocks -quiet -of_objects [get_pins -quiet -of_objects $c -filter {IS_CLOCK}]]
  if {[llength $cf] && $ck eq [get_property NAME $cf]} { lappend pwf $c }
}
puts "OTPU_RAMS [llength $rams] OTPU_MCP_PINS [llength $rp] OTPU_MCP_PW [llength $pwf]"
if {[llength $rp]} {
  set_multicycle_path 2 -setup -end -from $cs -through $rp
  set_multicycle_path 1 -hold -end -from $cs -through $rp
  set_multicycle_path 2 -setup -start -from $rams -to $cs
  set_multicycle_path 1 -hold -start -from $rams -to $cs
}
if {[llength $pwf]} {
  set_multicycle_path 2 -setup -end -from $cs -to $pwf
  set_multicycle_path 1 -hold -end -from $cs -to $pwf
}
# VAR 5 (otpu_pumpk_bank, K = the clocks' ratio): the core-domain inputs reach the RAM only at core
# edges (slot 0's request and the write stage), and the fast registers loaded there (the
# registered slots, the next edge's read, the presented words) take them in a core period; the
# presented words (oq) reach the core domain in a core period. The rest is one fast period.
set kpins [get_pins -quiet -hierarchical -filter {NAME =~ *u_pkb/ra0[*] || NAME =~ *u_pkb/ren0 || NAME =~ *u_pkb/wv || NAME =~ *u_pkb/wa[*] || NAME =~ *u_pkb/wd[*]}]
set kreg [get_cells -quiet -hierarchical -filter {(NAME =~ *u_pkb/sra_reg* || NAME =~ *u_pkb/sren_reg* || NAME =~ *u_pkb/ad_?_r_reg* || NAME =~ *u_pkb/en_?_r_reg* || NAME =~ *u_pkb/oq_reg*) && IS_SEQUENTIAL}]
set koq [get_cells -quiet -hierarchical -filter {NAME =~ *u_pkb/oq_reg* && IS_SEQUENTIAL}]
puts "OTPU_MCP_K pins [llength $kpins] regs [llength $kreg] oq [llength $koq]"
if {[llength $kpins] && [llength $cf]} {
  set kk [expr {round([get_property PERIOD $cs] / [get_property PERIOD $cf])}]
  puts "OTPU_K $kk"
  set_multicycle_path $kk -setup -end -from $cs -through $kpins -to $rams
  set_multicycle_path [expr {$kk - 1}] -hold -end -from $cs -through $kpins -to $rams
  set_multicycle_path $kk -setup -end -from $cs -to $kreg
  set_multicycle_path [expr {$kk - 1}] -hold -end -from $cs -to $kreg
  set_multicycle_path $kk -setup -start -from $koq -to $cs
  set_multicycle_path [expr {$kk - 1}] -hold -start -from $koq -to $cs
}
report_utilization -file $out/util_synth.txt
opt_design
place_design
phys_opt_design
route_design
report_utilization -file $out/util.txt
report_utilization -hierarchical -hierarchical_depth 4 -file $out/util_hier.txt
report_timing_summary -max_paths 10 -file $out/timing_summary.txt
report_clock_interaction -file $out/clock_interaction.txt
report_pulse_width -file $out/pulse_width.txt
report_exceptions -file $out/exceptions.txt
report_methodology -file $out/methodology.txt
report_power -file $out/power.txt
foreach {tag from to} [list SS $cs $cs FF $cf $cf SF $cs $cf FS $cf $cs] {
  if {![llength $from] || ![llength $to]} { puts "OTPU_PATH $tag none"; continue }
  set q [lindex [get_timing_paths -quiet -from $from -to $to -setup -max_paths 1 -nworst 1] 0]
  set h [lindex [get_timing_paths -quiet -from $from -to $to -hold -max_paths 1 -nworst 1] 0]
  if {[llength $q]} {
    puts "OTPU_PATH $tag setup [get_property SLACK $q] req [get_property REQUIREMENT $q] levels [get_property LOGIC_LEVELS $q] [get_property STARTPOINT_PIN $q] -> [get_property ENDPOINT_PIN $q] | hold [get_property SLACK $h]"
    report_timing -from $from -to $to -max_paths 10 -nworst 1 -file $out/timing_$tag.txt
  } else {
    puts "OTPU_PATH $tag none"
  }
}
# the RAMs' own limit (the speed file's minimum period of one RAM cell)
if {[llength $rams]} {
  set c [lindex $rams 0]
  foreach l [split [report_pulse_width -cells $c -return_string] \n] {
    if {[regexp {^Min Period\s+\S+\s+(\S+)\s+\S+\s+([0-9.]+)\s+([0-9.]+)\s+(-?[0-9.]+)} $l -> pin req act sl]} {
      puts "OTPU_PULSE [get_property REF_NAME $c] $pin req $req period $act slack $sl"
    }
  }
}
set p [lindex [get_timing_paths -setup -max_paths 1 -nworst 1] 0]
set h [lindex [get_timing_paths -hold -max_paths 1 -nworst 1] 0]
puts "OTPU_WNS [get_property SLACK $p] OTPU_WHS [get_property SLACK $h]"
puts "OTPU_DONE"
