# Build the BCU1525 bitstream from create_project.tcl's project: synthesis, implementation, reports,
# the bitstream and the SPI flash image (the MT25QU01G, 1 Gb, x4: docs/bcu1525.md section 6).
#   vivado -mode batch -source build.tcl -tclargs [OUT_DIR] [JOBS] [synth|impl|impl-scripts|post|full] \
#       [IMPL_STRATEGY]
# synth: synthesis and its reports only; impl: the implementation of a finished synthesis;
# impl-scripts: only impl_1's run scripts (run_vivado.sh IMPL_ALONE runs its runme.sh without this
# Vivado and its ~2 GB waiting), then post: the reports and the bitstream of that implementation.
# OTPU_SYNTH_THREADS (run_vivado.sh SYNTH_THREADS): every synthesis run's general.maxThreads.
# The YPCB's build.tcl is the model; the block-RAM collision check is the same.
set here [file normalize [file dirname [info script]]]
set root [file normalize $here/../../..]
set out [expr {[llength $argv] > 0 ? [file normalize [lindex $argv 0]] : "$root/build/vivado_bcu1525"}]
set jobs [expr {[llength $argv] > 1 ? [lindex $argv 1] : 8}]
set impl_only [expr {[lindex $argv 2] in {impl impl-scripts}}]
set scripts_only [expr {[lindex $argv 2] eq "impl-scripts"}]
set post_only [expr {[lindex $argv 2] eq "post"}]
set synth_only [expr {[lindex $argv 2] eq "synth"}]
set strategy [lindex $argv 3]

open_project $out/otpu.xpr
file mkdir $out/reports
set fast [expr {[info exists ::env(OTPU_FAST)] && $::env(OTPU_FAST) eq "1"}]

if {$post_only} {
} elseif {!$impl_only} {
  # the project's block design: otpu_bd (create_project.tcl) or otpu_bdm (create_project_mem.tcl)
  # (not the block designs inside its IPs: the DDR4 MIG's MicroBlaze, .../ip/<ip>/bd_0/bd_*.bd)
  set bdf [lsearch -all -inline -not -glob [get_files -filter {FILE_TYPE == "Block Designs"}] */ip/*]
  if {[llength $bdf] != 1} { error "expected one block design, not '$bdf'" }
  generate_target all $bdf
  export_ip_user_files -of_objects $bdf -no_script -sync -force -quiet
  # IMAGE=wg's MIG address map (OTPU_WG_MIGMAP): the generated UI decodes edited before synthesis
  source $here/mig_map.tcl
  otpu_mig_map $out patch
  create_ip_run $bdf
  # synth_design forks up to general.maxThreads - 1 helper processes, each with its own share of the
  # design: on a 32 GB host the VU9P top went from 8 GB (the main process) to ~19 GB with 7 of them
  if {[info exists ::env(OTPU_SYNTH_THREADS)] && $::env(OTPU_SYNTH_THREADS) ne ""} {
    set hook $out/synth_threads.tcl
    set fh [open $hook w]; puts $fh "set_param general.maxThreads $::env(OTPU_SYNTH_THREADS)"; close $fh
    foreach r [get_runs -filter {IS_SYNTHESIS}] { set_property STEPS.SYNTH_DESIGN.TCL.PRE $hook $r }
  }
  reset_run synth_1
  launch_runs synth_1 -jobs $jobs
  wait_on_run synth_1
  if {[get_property PROGRESS [get_runs synth_1]] != "100%"} { error "synthesis failed" }
  otpu_mig_map $out check
  set fh [open $out/otpu.runs/synth_1/runme.log]; set slog [read $fh]; close $fh
  set coll [regexp -all -inline {Synth 8-6430\] The Block RAM "[^"]*"} $slog]
  if {[llength $coll]} {
    error "synthesis made asynchronous-read memories block RAMs with a read-address register: [join $coll {; }]"
  }
  open_run synth_1
  report_utilization -hierarchical -hierarchical_depth 4 -file $out/reports/synth_util_hier.rpt
  report_utilization -slr -file $out/reports/synth_util_slr.rpt
  report_timing_summary -max_paths 20 -file $out/reports/synth_timing.rpt
  # every port on its pin: the first VU9P build lost the pin constraints, and only write_bitstream's
  # DRC (NSTD-1, UCIO-1) stopped it, after the implementation
  if {[string match -nocase xcvu9p* [get_property PART [current_project]]]} {
    set nopin [get_ports -quiet -filter {PACKAGE_PIN == ""}]
    if {[llength $nopin]} { error "[llength $nopin] ports without a pin: [lrange $nopin 0 7]" }
    puts "pins: all [llength [get_ports]] ports placed"
  }
  # a build with a fallback clock's constants (otpu_ctrl.sv's otpu_clk_k: the WG top's CORE_KHZ1):
  # both values of each in the synthesized netlist (clk133.tcl's report alone, reports/clk_k.rpt).
  # A miss loses the fallback, not this build: a critical warning, and the build goes on
  if {[llength [get_cells -quiet -hierarchical -filter {ORIG_REF_NAME == otpu_clk_k || REF_NAME =~ otpu_clk_k*}]]} {
    namespace eval ::otpu {}
    set ::otpu::clk_rpt_dir $out/reports
    set ::otpu::clk_report_only 1
    if {[catch {source $here/clk133.tcl} e]} {
      puts "CRITICAL WARNING: \[build.tcl\] the fallback clock's constants: $e ($out/reports/clk_k.rpt)"
    }
    unset ::otpu::clk_report_only
  }
  # IMAGE=wg at two slices, or with OTPU_WG_STOP=synth: the floorplan's pblocks and cell checks on
  # the synthesized netlist and each pblock's share of it (bcu1525_wg_synth_check.tcl, its own scope)
  set wg_stop [expr {[info exists ::env(OTPU_WG_STOP)] && $::env(OTPU_WG_STOP) eq "synth"}]
  set wg_chk 0
  if {[file exists $out/wg_params.tcl]} {
    set fh [open $out/wg_params.tcl]; set wg_txt [read $fh]; close $fh
    set wg_chk [expr {[regexp {set WG_S 2} $wg_txt] || $wg_stop}]
  }
  if {$wg_chk} { apply {{here out} { source $here/bcu1525_wg_synth_check.tcl }} $here $out }
  close_design
  if {$wg_chk && $wg_stop} {
    error "stopped after the synthesis and its checks (OTPU_WG_STOP=synth): synth_1 is finished, STEP=impl (bcuwg_run.sh RESUME=1) goes on from it"
  }
  if {$synth_only} { return }
} else {
  if {[get_property PROGRESS [get_runs synth_1]] != "100%"} { error "impl: no finished synthesis to reuse" }
  set_property NEEDS_REFRESH false [get_runs synth_1]
}

# OTPU_IMPL_TO (run_vivado.sh IMPL_TO): impl_1 to that step only (impl_resume.tcl goes on from its
# checkpoint), no reports
set impl_to [expr {[info exists ::env(OTPU_IMPL_TO)] ? $::env(OTPU_IMPL_TO) : ""}]
set to_args [expr {$impl_to ne "" ? [list -to_step $impl_to] : {}}]
if {!$post_only} {
  if {$strategy ne ""} { set_property strategy $strategy [get_runs impl_1] }
  if {!$fast} { set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1] }
  reset_run impl_1
  if {$scripts_only} {
    launch_runs impl_1 -scripts_only {*}$to_args
    # the environment a run launched by this Vivado would inherit: a wrapper's libraries (a host's
    # vivado wrapper may add AMD's Ubuntu 24 compatibility libraries to LD_LIBRARY_PATH)
    set fh [open $out/impl_env.sh w]
    set ld [expr {[info exists ::env(LD_LIBRARY_PATH)] ? $::env(LD_LIBRARY_PATH) : ""}]
    puts $fh "export LD_LIBRARY_PATH='$ld'"
    close $fh
    puts "impl-scripts: $out/otpu.runs/impl_1/runme.sh ($out/impl_env.sh)"
    return
  }
  launch_runs impl_1 -jobs $jobs {*}$to_args
  wait_on_run impl_1
  if {$impl_to ne ""} {
    if {![file exists $out/otpu.runs/impl_1/.$impl_to.end.rst]} { error "impl_1 did not finish $impl_to" }
    puts "impl_1 to $impl_to: done (impl_resume.tcl $out next)"
    return
  }
}
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} { error "implementation failed" }
open_run impl_1

source $here/post_impl.tcl
