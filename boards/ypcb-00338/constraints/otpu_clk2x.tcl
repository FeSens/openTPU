# The paths between core_clk and core_clk2x: the block design's MMCM output at twice core_clk,
# from the same VCO by integer dividers and rising with it (bd_native.tcl / the BCU1525's bd.tcl,
# CLK2X 1), the RTL's clk_f. One file for every user of the 2x clock, one section each, shared by
# the boards (create_project.tcl, with CLK2X 1) and the out-of-context runs
# (rtl/exp/tmem_nocopy/ooc.tcl, the pumped one copy). Unmanaged Tcl, read in implementation only
# and late (PROCESSING_ORDER LATE), on the synthesized netlist; not scoped: each section finds its
# cells by their module (kept hierarchies).

# ---------------------------------------------------------------- the TMEM's pumped RAMs
# otpu_pump_bank (rtl/mem/otpu_tmem.sv; otpu_tmem IMPL 3 TDP with PMP 1): a true dual-port RAM on
# clk_f. Its reads' requests (ra / rena, rb / renb) come from core_clk and are taken at the even
# fast edge, the next core edge: one core period (2 fast, -end). The words read (qa / qb) change
# only after an even edge and hold through the next core cycle: one core period to core_clk (2
# fast, -start). The write (wv / wa / wd) lands at the odd fast edge, half a core period after the
# core edge that launched it, and the phase (tog -> tq / evr) at the fast edge after the core
# edge: one fast period each, timed as such by default.
set c2x_pb [get_cells -quiet -hierarchical -filter {ORIG_REF_NAME == otpu_pump_bank || REF_NAME == otpu_pump_bank}]
if {[llength $c2x_pb]} {
  set c2x_c0 [lindex $c2x_pb 0]
  set c2x_cs [get_clocks -quiet -of_objects [get_pins -quiet $c2x_c0/clk]]
  set c2x_cf [get_clocks -quiet -of_objects [get_pins -quiet $c2x_c0/clk_f]]
  set c2x_ts [get_property -quiet -min PERIOD $c2x_cs]
  set c2x_tf [get_property -quiet -min PERIOD $c2x_cf]
  if {[llength $c2x_cs] != 1 || [llength $c2x_cf] != 1 || abs($c2x_ts - 2.0 * $c2x_tf) > 0.001} {
    puts "CRITICAL WARNING: \[otpu_clk2x.tcl\] the pumped RAMs' clocks are '$c2x_cs' ($c2x_ts ns) and '$c2x_cf' ($c2x_tf ns), expected one each, the second at twice the first: their reads unconstrained"
  } else {
    set c2x_rp [get_pins -quiet -of_objects $c2x_pb -filter {REF_PIN_NAME =~ ra[*] || REF_PIN_NAME =~ rb[*] || REF_PIN_NAME == rena || REF_PIN_NAME == renb}]
    set c2x_qp [get_pins -quiet -of_objects $c2x_pb -filter {REF_PIN_NAME =~ qa[*] || REF_PIN_NAME =~ qb[*]}]
    set_multicycle_path 2 -setup -end -from $c2x_cs -through $c2x_rp -to $c2x_cf
    set_multicycle_path 1 -hold -end -from $c2x_cs -through $c2x_rp -to $c2x_cf
    set_multicycle_path 2 -setup -start -from $c2x_cf -through $c2x_qp -to $c2x_cs
    set_multicycle_path 1 -hold -start -from $c2x_cf -through $c2x_qp -to $c2x_cs
    puts "otpu_clk2x.tcl: [llength $c2x_pb] pumped TMEM RAMs ([get_property NAME $c2x_cs] $c2x_ts ns, [get_property NAME $c2x_cf] $c2x_tf ns), [llength $c2x_rp] read-request pins, [llength $c2x_qp] word pins"
  }
} else {
  puts "otpu_clk2x.tcl: no pumped TMEM RAMs (otpu_pump_bank)"
}

# ---------------------------------------------------------------- the MXU's pumped array
# otpu_mxpc (rtl/mxu/otpu_mxu.sv; otpu_mxu PUMP 1, [uarch] MXU_PUMP): each position's DSP48E2
# (otpu_ppe) on clk_f. Its weights' registers (A2 and D: CEA2 / CED = pe, high at the even fast
# edges, the core edges) take the weight launched at a core edge at the next one: one core period
# (2 fast, -end). The chain ends' holds (g_hold's hold0 and hold1e, loaded at the even edges) are
# read at the next core edge: one core period to core_clk (2 fast, -start). Column 0's
# activations (row 0's byte taken at the odd edge), the stream between columns and the AD / M / P
# registers keep their one fast period, timed as such by default.
set c2x_pc [get_cells -quiet -hierarchical -filter {ORIG_REF_NAME == otpu_mxpc || REF_NAME == otpu_mxpc}]
if {[llength $c2x_pc]} {
  set c2x_c0 [lindex $c2x_pc 0]
  set c2x_ms [get_clocks -quiet -of_objects [get_pins -quiet $c2x_c0/clk]]
  set c2x_mf [get_clocks -quiet -of_objects [get_pins -quiet $c2x_c0/clk_f]]
  set c2x_ts [get_property -quiet -min PERIOD $c2x_ms]
  set c2x_tf [get_property -quiet -min PERIOD $c2x_mf]
  if {[llength $c2x_ms] != 1 || [llength $c2x_mf] != 1 || abs($c2x_ts - 2.0 * $c2x_tf) > 0.001} {
    puts "CRITICAL WARNING: \[otpu_clk2x.tcl\] the pumped MXU's clocks are '$c2x_ms' ($c2x_ts ns) and '$c2x_mf' ($c2x_tf ns), expected one each, the second at twice the first: its weights and chain ends at one fast period"
  } else {
    # the DSPs under the columns (otpu_ppe's u_dsp), their A and D data pins (by regexp: in a
    # -filter pattern [..] is a glob class)
    set c2x_dsp [get_cells -quiet -hierarchical -filter {REF_NAME == DSP48E2 && NAME =~ *u_ppe/u_dsp}]
    set c2x_wp {}
    foreach c2x_p [get_pins -quiet -of_objects $c2x_dsp -filter {DIRECTION == IN && (REF_PIN_NAME =~ A* || REF_PIN_NAME =~ D*)}] {
      if {[regexp {^(A|D)\[[0-9]+\]$} [get_property REF_PIN_NAME $c2x_p]]} { lappend c2x_wp $c2x_p }
    }
    set c2x_hc [get_cells -quiet -hierarchical -filter {IS_SEQUENTIAL && (NAME =~ *g_hold.hold0_reg* || NAME =~ *g_hold.hold1e_reg*)}]
    if {[llength $c2x_wp]} {
      set_multicycle_path 2 -setup -end -from $c2x_ms -to $c2x_wp
      set_multicycle_path 1 -hold -end -from $c2x_ms -to $c2x_wp
    }
    if {[llength $c2x_hc]} {
      set_multicycle_path 2 -setup -start -from $c2x_hc -to $c2x_ms
      set_multicycle_path 1 -hold -start -from $c2x_hc -to $c2x_ms
    }
    puts "otpu_clk2x.tcl: [llength $c2x_pc] pumped MXU columns ([get_property NAME $c2x_ms] $c2x_ts ns, [get_property NAME $c2x_mf] $c2x_tf ns), [llength $c2x_dsp] DSPs, [llength $c2x_wp] weight pins, [llength $c2x_hc] chain-end holds"
    # for the BCU1525's opt_design pre-hook (bcu1525_wg_check.tcl): the columns constrained, when
    # both sets were found
    namespace eval ::otpu {}
    if {[llength $c2x_dsp] && [llength $c2x_wp] && [llength $c2x_hc]} {
      set ::otpu::ok(otpu_clk2x_mxu) [llength $c2x_pc]
    }
  }
} else {
  puts "otpu_clk2x.tcl: no pumped MXU columns (otpu_mxpc)"
}
