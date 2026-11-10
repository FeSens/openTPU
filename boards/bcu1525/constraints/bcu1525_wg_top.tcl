# Clock-domain crossings of the BCU1525 WG top (otpu_fpga_top_bcu_wg) and of otpu_bcu_wg_sys outside
# the bridges' core halves (boards/bcu1525/constraints/otpu_hbm_ch.tcl, scoped to otpu_hbm_cs) and the
# MXU lanes' posted-beat staging (otpu_mxu_stage.tcl, scoped to otpu_mxu_stage). Unmanaged Tcl, read
# in implementation only and late (create_project_wg.tcl), on the synthesized netlist; not scoped.
# No clock pair is declared asynchronous (set_clock_groups or a false
# path between clocks would override the bridges' set_max_delay -datapath_only), each crossing here
# gets its own constraint, and a miss goes to ::otpu::fail: bcu1525_wg_check.tcl (opt_design's
# pre-hook) stops the run on it (in a read_xdc file an error or a return is only a warning).
#
# The clocks (bd_wg.tcl): core_clk (the governor, the gates), acc_clk and acc_clk2x (the
# accelerator's: core_clk's divide and twice it, from the same MMCM behind enables), aclk (300 MHz,
# the same MMCM: the bridges' AXI halves, the pairs), xdma_aclk (XDMA's) and each MIG's ui_clk
# (inside the block design; the SmartConnects' own constraints cross into it).
namespace eval ::otpu {}
proc ::otpu::wg_fail {msg} {
  puts "CRITICAL WARNING: \[bcu1525_wg_top.tcl\] $msg"
  lappend ::otpu::fail "bcu1525_wg_top.tcl: $msg"
}
proc ::otpu::wg_clk {name pin} {
  set c [get_clocks -quiet -of_objects [get_pins -quiet $pin]]
  if {[llength $c] != 1} { ::otpu::wg_fail "$name: clocks '$c' at $pin, expected one" }
  return $c
}
set c_core [::otpu::wg_clk core_clk {gov_rst_reg/C}]
set c_acc  [::otpu::wg_clk acc_clk {u_sys/u_board/cal_s1_reg[0]/C}]
set c_a    [::otpu::wg_clk aclk {u_sys/g_ars[0].s_reg[0]/C}]
set c_x    [::otpu::wg_clk xdma_aclk {xrst_reg/C}]
set t_core [get_property -quiet -min PERIOD $c_core]
set t_acc  [get_property -quiet -min PERIOD $c_acc]
set t_a    [get_property -quiet -min PERIOD $c_a]
set t_x    [get_property -quiet -min PERIOD $c_x]
puts "bcu1525_wg_top.tcl: core $c_core $t_core ns, acc $c_acc $t_acc ns, aclk $c_a $t_a ns, xdma $c_x $t_x ns"

# a set of cells by pattern, each pattern finding some (an empty -from or -to is an error)
proc ::otpu::wg_cells {what pats} {
  set all {}
  foreach p $pats {
    set c [get_cells -quiet $p]
    if {![llength $c]} { ::otpu::wg_fail "$what: no cell $p" }
    set all [concat $all $c]
  }
  return [get_cells -quiet $all]
}

# ---------------------------------------------------------------- levels into two-flop synchronizers
# each MIG's calibration (ui_clk) into cal_c1 (core_clk); each DIMM's live flag (core_clk) into its
# SmartConnect reset's synchronizer (g_dimm[d].sl, aclk) and, inverted, into its port reset's
# (u_sys/g_ars[d].s, aclk): levels that change once a power cycle, false paths into the first stage
set_false_path -to [::otpu::wg_cells calib {cal_c1_reg[*]}]
set_false_path -to [::otpu::wg_cells sc_rstn {g_dimm[*].sl_reg[0]}]
set_false_path -to [::otpu::wg_cells arst {u_sys/g_ars[*].s_reg[0]}]
# the DCIRESETs (their configuration sites, no clock): LOCKED into its synchronizer. RST, from
# core_clk, has no timing arc (dev100c: a false path to it was Constraints 18-401 / 18-512)
set dci [get_cells -quiet {g_dci[*].u_dci}]
if {[llength $dci] != 3} {
  ::otpu::wg_fail "DCIRESET: '$dci', expected three"
} else {
  set_false_path -to [::otpu::wg_cells dci_lk {dci_s_reg[0]}]
}
# each MIG's reset (core_clk) into its sys_rst: the IP synchronizes it. Through the block design's
# pin: the register (mig_rst[d], assigned in g_dimm[d]) is not mig_rst_reg[d] after synthesis
foreach d {0 1 2 3} {
  set p [get_pins -quiet u_bd/mig_rst$d]
  if {[llength $p] != 1} { ::otpu::wg_fail "mig_rst: no pin u_bd/mig_rst$d" } else { set_false_path -through $p }
}
# the DMA gates (core_clk) into their synchronizers (xdma_aclk): one xdma_aclk period
if {[llength $c_core] == 1 && [llength $c_x] == 1} {
  set_max_delay -datapath_only -from $c_core -to [::otpu::wg_cells "DMA gates" {xo1_reg[*] xd1_reg[*]}] $t_x
}

# ---------------------------------------------------------------- posted-beat counts into acc_clk
# Each bridge's p_cntg (otpu_hbm_cs p_g, a gray count from a register, aclk) into
# the slice's adapter's ASYNC_REG synchronizer (otpu_native_dram g_pc.pg1[j], acc_clk), or from the
# last of its otpu_xstg p_clk stages (NAT_XSTG_P) when there are some. A bus skew a channel. Brackets
# as ? in the names: get_cells takes [n] for a bus index. Two slices (bcu1525-s2): 16 bridges, board
# channel 8 s + j slice s's channel j (slice 1's adapter and staging in u_board/g_slice[1]).
set n_ch [llength [get_cells -quiet {u_sys/g_ch[*].u_br/u_cs}]]
if {$n_ch != 8 && $n_ch != 16} { ::otpu::wg_fail "$n_ch bridge core halves (8 or 16)" }
set n_pc 0
if {[llength $c_acc] == 1 && [llength $c_a] == 1} {
  for {set k 0} {$k < $n_ch} {incr k} {
    set s [expr {$k / 8}]
    set j [expr {$k % 8}]
    set b [expr {$s == 0 ? "u_sys/u_board/" : "u_sys/u_board/g_slice?$s?."}]
    set d [get_cells -quiet "${b}u_mem/g_pc.pg1_reg?$j?\[*\]"]
    set g [get_cells -quiet "u_sys/g_ch?$k?.u_br/u_cs/p_g_reg\[*\]"]
    for {set i 0} {[llength [set gx [get_cells -quiet "${b}u_xs/g_pc?$j?.u_p/g_pipe.g_s?$i?.r_reg\[*\]"]]]} {incr i} {
      set g $gx
    }
    if {![llength $d] || [llength $g] != [llength $d]} {
      ::otpu::wg_fail "channel $k: [llength $g] p_g registers, [llength $d] synchronizer ones: its posted-beat count unconstrained"
      continue
    }
    set_max_delay -datapath_only -from $g -to $d $t_a
    set_bus_skew -from $g -to $d [expr {min($t_a, $t_acc)}]
    incr n_pc
  }
}
puts "bcu1525_wg_top.tcl: posted-beat counts of $n_pc of $n_ch channels constrained"

# ---------------------------------------------------------------- the accelerator's clock pair
# OTPU_WG_CDG 1 (run_vivado.sh WG_CDG): CLOCK_DELAY_GROUP otpu_acc on acc_clk's and acc_clk2x's
# nets (the top's BUFGCE_DIV pair, u_acc_buf / u_acc2x_buf): the placer and the router balance the
# two trees' insertion delays (UG949). Off by default: place_screen.tcl's OTPU_SCREEN_CDG measures
# it first (it may cost the 2x -> 1x multicycle paths their skew).
if {[info exists ::env(OTPU_WG_CDG)] && $::env(OTPU_WG_CDG) eq "1"} {
  set cn [get_nets -quiet -of_objects [get_pins -quiet {u_acc_buf/O u_acc2x_buf/O}]]
  if {[llength $cn] != 2} {
    ::otpu::wg_fail "CLOCK_DELAY_GROUP: acc_clk's and acc_clk2x's nets '$cn', expected two"
  } else {
    set_property CLOCK_DELAY_GROUP otpu_acc $cn
    puts "bcu1525_wg_top.tcl: CLOCK_DELAY_GROUP otpu_acc on [join $cn] (OTPU_WG_CDG)"
  }
}

# ---------------------------------------------------------------- the I/O
# no internal VREF on the DDR4 banks: the MIGs train each data byte's own (one HPIO_VREF a byte,
# FABRIC_RANGE1: 32). LiteDRAM's 0.84 V (it has none) left the four banks that hold a DIMM's
# address and command and one data byte at 0.600 with a conflict (dev100b: BIIVRC-1 on 42, 47, 65
# and 70); without it, as the IP's example design, the DRC has none (dev100b's routed checkpoint)
set pod [get_ports -quiet -filter {IOSTANDARD =~ POD12*}]
if {![llength $pod]} {
  ::otpu::wg_fail "no POD12 ports (the MIGs' DQ / DQS / DM)"
} elseif {[catch {get_iobanks -of_objects $pod} banks]} {
  ::otpu::wg_fail "the POD12 ports' banks: $banks"
} else {
  puts "bcu1525_wg_top.tcl: [llength $pod] POD12 ports on banks [lsort $banks], no INTERNAL_VREF"
}

if {[llength $c_core] == 1 && [llength $c_acc] == 1 && [llength $c_a] == 1 && [llength $c_x] == 1 && $n_pc == $n_ch
    && ($n_ch == 8 || $n_ch == 16)} {
  incr ::otpu::ok(bcu1525_wg_top)
}
