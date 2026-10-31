# The SQRL FK33 build's preflight (run_vivado.sh runs it before anything else; PREFLIGHT=0 skips it): stop with a
# clear message, before a project or IP is generated, when this Vivado has no xcvu33p-fsvh2104-2-e (the Virtex
# UltraScale+ HBM device files are an installer option) or no license that covers it (the VU33P is outside the free
# tier): a one-register design is synthesized and optimized on the part, which asks for the device's Synthesis and
# Implementation licenses. The BCU1525's preflight.tcl is the model.
#   vivado -mode batch -nojournal -nolog -source preflight.tcl -tclargs PART
# Exit 0 (PREFLIGHT ok), 2 (the part is not installed), 3 (no license for it: synthesis refused, or
# the device files installed but the part hidden, as Vivado hides the parts no license enables).
set part [expr {[llength $argv] ? [lindex $argv 0] : "xcvu33p-fsvh2104-2-e"}]
puts "PREFLIGHT: Vivado [version -short], part $part"
if {![llength [get_parts -quiet $part]]} {
  # Vivado lists only the parts a license enables: with the device files installed (their
  # family's directory under data/parts) a missing part is the license's
  set dirs {xcvu virtexuplus xcku kintexuplus xczu zynquplus xcau artixuplus}
  set pre [string range $part 0 3]
  if {[dict exists $dirs $pre] && [file isdirectory [file join $::env(XILINX_VIVADO) data parts \
                                                       xilinx [set fam [dict get $dirs $pre]]]]} {
    puts "PREFLIGHT FAILED: the $fam device files are installed but Vivado lists no $part: no\
          license enables it (Vivado lists only the parts its licenses cover). The VU33P needs a\
          paid license (Vivado ML Enterprise, or one locked to the device) whose version covers this\
          Vivado; check the Vivado License Manager (vlm) and XILINXD_LICENSE_FILE (a node-locked\
          license is bound to one network interface's MAC)."
    exit 3
  }
  set fams [lsort -unique [get_property -quiet FAMILY [get_parts -quiet]]]
  puts "PREFLIGHT FAILED: the part $part is not installed in this Vivado (installed families:\
        $fams). Add the Virtex UltraScale+ HBM devices: the Vivado installer, or Help > Add Design\
        Tools or Devices."
  exit 2
}
set dir [file normalize [file join [pwd] .preflight_[pid]]]
file mkdir $dir
set f [file join $dir preflight_top.v]
set fh [open $f w]
puts $fh "module preflight_top (input clk, input d, output reg q); always @(posedge clk) q <= d; endmodule"
close $fh
create_project -in_memory -part $part
read_verilog $f
set rc [catch {
  synth_design -top preflight_top -part $part -mode out_of_context
  opt_design
} msg]
file delete -force $dir
if {$rc} {
  puts "PREFLIGHT FAILED: synthesis / optimization on $part: $msg"
  puts "PREFLIGHT FAILED: no license covering $part? The VU33P needs a paid license (Vivado ML\
        Enterprise, or one locked to the device) whose version covers this Vivado; check the Vivado\
        License Manager (vlm) and XILINXD_LICENSE_FILE (a node-locked license is bound to one\
        network interface's MAC)."
  exit 3
}
puts "PREFLIGHT ok: $part is installed and licensed (synthesis and implementation)"
exit 0
