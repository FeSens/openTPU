# Check the BCU1525's block design and DDR4 core in Vivado on any UltraScale+ part, so they can be
# checked before a VU9P license exists (a Kintex UltraScale+ part with a PCIE4 block and GTY quads,
# e.g. xcku5p-ffvb676-2-e, in Vivado's free tier):
#   vivado -mode batch -source ip_check.tcl -tclargs PART OUT_DIR [STEPS]
# STEPS (comma separated, default bd): bd = source bd.tcl (validate_bd_design and its checks) and
# generate the IP output products; bdsynth = also synthesize the block design's IP out of context;
# ld4 = synthesize the LiteDRAM DDR4 core (boards/bcu1525/litedram_udimm/otpu_ld4.v; OTPU_DIMM
# rdimm: litedram/otpu_ld4.v) out of context and report its utilization (its primitives: MMCME4,
# IDELAYCTRL, I/ODELAYE3, I/OSERDESE3, the CPU).
set here [file normalize [file dirname [info script]]]
set root [file normalize $here/../../..]
set part [lindex $argv 0]
set out [file normalize [lindex $argv 1]]
set steps [split [expr {[llength $argv] > 2 ? [lindex $argv 2] : "bd"}] ,]
file mkdir $out
create_project -force ipcheck $out/ipcheck -part $part
set CORE_MHZ 100
if {"bd" in $steps || "bdsynth" in $steps} {
  source $here/bd.tcl
  set bd [get_files otpu_bd.bd]
  generate_target all $bd
  foreach ip [get_ips -all] {
    puts "IP [get_property NAME $ip]: [get_property IPDEF $ip]"
  }
  report_ip_status -file $out/ip_status.txt
  if {"bdsynth" in $steps} {
    create_ip_run $bd
    set runs [get_runs -filter {IS_SYNTHESIS && NAME != synth_1}]
    launch_runs $runs -jobs 4
    foreach r $runs { wait_on_run $r }
    foreach r $runs {
      puts "run $r: [get_property STATUS $r] [get_property PROGRESS $r]"
      if {[get_property PROGRESS $r] ne "100%"} { error "$r failed" }
    }
  }
}
if {"ld4" in $steps} {
  set d $root/boards/bcu1525/[expr {[info exists ::env(OTPU_DIMM)] && $::env(OTPU_DIMM) eq "rdimm" ? "litedram" : "litedram_udimm"}]
  read_verilog $d/otpu_ld4.v
  foreach f [glob $d/otpu_ld4_*.init] { add_files -norecurse $f }
  set_property FILE_TYPE {Memory Initialization Files} [get_files *.init]
  synth_design -top otpu_ld4 -part $part -mode out_of_context
  report_utilization -file $out/ld4_util.txt
  report_utilization -hierarchical -hierarchical_depth 1 -file $out/ld4_util_hier.txt
  puts "otpu_ld4 synthesized on $part"
}
puts "ip_check done: $steps on $part"
