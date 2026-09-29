# Vivado check of the build's project and top without a build (minutes, not hours):
# create_project.tcl, so bd_native.tcl is built and validated and its wrapper made; the
# constraint files' properties; the block design's synthesis sources; then RTL elaboration of the
# top (synth_design -rtl): the instances, their modules, black boxes, and the elaboration's
# messages about the top's own sources.
#   vivado -mode batch -log LOG -source boards/ypcb-00338/vivado/check_native.tcl \
#     -tclargs OUT_DIR [LOG]
# (LOG: Vivado's log, read back for the elaboration's messages; default ./vivado.log)
# Prints "CHECK_NATIVE ok" or "CHECK_NATIVE FAILED: <reasons>" last.
set root [file normalize [file dirname [info script]]/../../..]
# small footprint: the check shares its host with builds
set_param general.maxThreads 2
set out [file normalize [lindex $argv 0]]
set logf [file normalize [expr {[llength $argv] > 1 ? [lindex $argv 1] : "vivado.log"}]]
set bad {}
# create_project.tcl's arguments as run_vivado.sh passes them by default (DDR3-1066, MCOLS 4,
# ACT_ROWS 4; the systolic MXU)
set argv [list 1066 $out 4 100 "" 2 8 4 1 systolic]
set argc [llength $argv]
source $root/boards/ypcb-00338/vivado/create_project.tcl

puts "---- constraints"
foreach f [get_files -of_objects [get_filesets constrs_1]] {
  puts [format "%-24s %-5s ref=%-12s order=%-6s synth=%s impl=%s" [file tail $f] \
    [get_property FILE_TYPE $f] [get_property SCOPED_TO_REF $f] [get_property PROCESSING_ORDER $f] \
    [get_property USED_IN_SYNTHESIS $f] [get_property USED_IN_IMPLEMENTATION $f]]
}
foreach {name ref} {otpu_mem_ch.tcl otpu_mem_ch otpu_top_native.tcl {}} {
  set f [get_files -of_objects [get_filesets constrs_1] */$name]
  if {$f eq ""} { lappend bad "$name not in constrs_1"; continue }
  if {[get_property SCOPED_TO_REF $f] ne $ref} { lappend bad "$name: SCOPED_TO_REF [get_property SCOPED_TO_REF $f]" }
  if {[get_property PROCESSING_ORDER $f] ne "LATE"} { lappend bad "$name: PROCESSING_ORDER [get_property PROCESSING_ORDER $f]" }
  if {[get_property USED_IN_SYNTHESIS $f]} { lappend bad "$name: USED_IN_SYNTHESIS" }
}

puts "---- block design sources, elaboration"
# The elaboration needs every module's definition, and an out-of-context IP's (its stub) exists
# only once its own synthesis run has made it ("[Synth 8-439] module 'otpu_bd_clk_wiz_0_0' not
# found"). So for this check alone, not for the builds, the block design's IP is synthesized with
# the top (global): no IP synthesis runs, its sources elaborated in place.
set_property synth_checkpoint_mode None [get_files otpu_bd.bd]
generate_target synthesis [get_files otpu_bd.bd]

synth_design -rtl -name rtl_1

puts "---- instances"
set want {u_bd u_sys u_sys/u_split u_sys/u_ch0 u_sys/u_ch1 u_sys/u_board u_ld u_ibuf50 u_bufg50}
foreach c $want {
  set cell [get_cells -quiet $c]
  if {$cell eq ""} { lappend bad "no instance $c"; puts "$c: MISSING"; continue }
  puts [format "%-16s %s" $c [get_property REF_NAME $cell]]
}
foreach c {u_sys/u_ch0 u_sys/u_ch1} {
  set r [get_property -quiet ORIG_REF_NAME [get_cells -quiet $c]]
  if {$r eq ""} { set r [get_property -quiet REF_NAME [get_cells -quiet $c]] }
  if {![string match otpu_mem_ch* $r]} { lappend bad "$c is $r, not otpu_mem_ch (SCOPED_TO_REF would miss it)" }
}
set bb [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1}]
puts "black boxes: [llength $bb]"
foreach c [lsort [lrange $bb 0 29]] { puts "  $c ([get_property REF_NAME $c])" }
foreach c $bb {
  # inside the block design's IP (elaborated in place here; any encrypted parts)
  if {[string match u_bd/* $c]} { continue }
  lappend bad "black box $c ([get_property REF_NAME $c])"
}

# the elaboration's warnings about the top's own sources (port widths, unconnected or missing
# ports, undriven nets); the IP's own are left out
set log ""
if {[file exists $logf]} { set fh [open $logf]; set log [read $fh]; close $fh } else { puts "no log at $logf" }
set n 0
foreach line [split $log "\n"] {
  if {[regexp {^(CRITICAL WARNING|WARNING|ERROR): \[Synth 8-} $line] &&
      [regexp {otpu_fpga_top_ld\.sv|otpu_native_sys\.sv|otpu_mem_ch\.sv|otpu_axi_split2\.sv|otpu_afifo\.sv|otpu_native_dram\.sv|otpu_board\.sv|otpu_litedram\.v|otpu_bd_wrapper\.v} $line]} {
    puts "  $line"
    incr n
  }
}
puts "elaboration messages on the top's sources: $n (listed above)"

puts "---- summary"
if {[llength $bad]} {
  puts "CHECK_NATIVE FAILED: [join $bad {; }]"
} else {
  puts "CHECK_NATIVE ok"
}
