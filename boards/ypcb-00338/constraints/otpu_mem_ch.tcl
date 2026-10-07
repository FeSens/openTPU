# Clock-domain crossings of otpu_mem_ch (rtl/boards/ypcb-00338/otpu_mem_ch.sv) and its six
# otpu_afifo instances. Scoped to the module and read after the clocks exist, as Xilinx's own
# xpm_cdc constraint files are: Tcl (variables, get_property, if / foreach), so an unmanaged Tcl
# constraint file (.tcl), not an XDC:
#   non-project:  read_xdc -unmanaged -ref otpu_mem_ch otpu_mem_ch.tcl     (after create_clock)
#   project:      in constrs_1 with SCOPED_TO_REF otpu_mem_ch, PROCESSING_ORDER LATE and
#                 USED_IN_SYNTHESIS false (create_project.tcl)
# tools/memch_ooc.tcl checks it out of context (report_cdc, report_methodology, and every source
# of a path between the clocks against the crossings constrained here).
#
# The module's clocks: clk (the core clock), uclk (the controller's user clock: LiteDRAM's sys),
# xclk (XDMA's axi_aclk). Every path between two of them inside the module is
# a synchronizer (ASYNC_REG in the RTL) or data that a synchronized pointer guards, and each gets
# set_max_delay -datapath_only here, so all of them are timed, without clock skew:
#   - gray-coded buses: the FIFOs' pointers and the write-accept counters behind n_wdone and
#     XDMA's B (one per controller port and master, g_port[p]: the two ports can take a write in
#     the same cycle, so a count of both would step by two). One source period, and set_bus_skew
#     of the smaller period per count, so that the bits of one count (which moves one bit per
#     source cycle) arrive together;
#   - a FIFO's distributed RAM, written in one clock and read (asynchronously) in the other: one
#     destination period. An entry is read no sooner than two destination cycles after its write
#     pointer crossed, and the pointer is written in the same source cycle as the entry;
#   - the reset-request, hold and error-bit synchronizers (levels held for many cycles): one
#     source period.
# The top level must therefore not make these clock pairs asynchronous with set_clock_groups or
# set_false_path (either takes priority over set_max_delay and leaves the crossings untimed).
# rst and xrst cross only as the registered requests a_req and x_req. The report_cdc waivers
# at the end cover the structures report_cdc cannot classify as safe (the FIFO heads read across
# the clocks), each by its two ends.

set c_clk [get_clocks -quiet -of_objects [get_ports clk]]
set c_ucl [get_clocks -quiet -of_objects [get_ports uclk]]
set c_xcl [get_clocks -quiet -of_objects [get_ports xclk]]
set t_clk [get_property -quiet -min PERIOD $c_clk]
set t_ucl [get_property -quiet -min PERIOD $c_ucl]
set t_xcl [get_property -quiet -min PERIOD $c_xcl]
set k_cu  [expr {min($t_clk, $t_ucl)}]
set k_xu  [expr {min($t_xcl, $t_ucl)}]

# ---------------------------------------------------------------- resets, holds, the error bit
# the masters' reset requests a_req (clk) and x_req (xclk) into uclk; the per-master holds (uclk)
# into clk and xclk; the port-contract error c_err (uclk, sticky) into clk
set_max_delay -datapath_only -from $c_clk -to [get_cells a_rs1_reg] $t_clk
set_max_delay -datapath_only -from $c_xcl -to [get_cells x_rs1_reg] $t_xcl
set_max_delay -datapath_only -from $c_ucl -to [get_cells a_hs1_reg] $t_ucl
set_max_delay -datapath_only -from $c_ucl -to [get_cells x_hs1_reg] $t_ucl
set_max_delay -datapath_only -from $c_ucl -to [get_cells e_s1_reg] $t_ucl

# ---------------------------------------------------------------- write-accept counters (gray)
# per controller port p (g_port[p]): a_wacc (n_wdone) uclk -> clk, x_wacc (XDMA's B) uclk -> xclk;
# the bus skew per count (each moves one bit per uclk cycle; the two ports' counts are summed only
# after their synchronizers)
set_max_delay -datapath_only -from $c_ucl -to [get_cells {g_port[*].a_wacc_s1_reg[*]}] $t_ucl
set_max_delay -datapath_only -from $c_ucl -to [get_cells {g_port[*].x_wacc_s1_reg[*]}] $t_ucl
foreach p {0 1} {
  set_bus_skew -from [get_cells "g_port\[$p\].a_wacc_g_reg\[*\]"] -to [get_cells "g_port\[$p\].a_wacc_s1_reg\[*\]"] $k_cu
  set_bus_skew -from [get_cells "g_port\[$p\].x_wacc_g_reg\[*\]"] -to [get_cells "g_port\[$p\].x_wacc_s1_reg\[*\]"] $k_xu
}

# ---------------------------------------------------------------- FIFOs
# Per FIFO: the write pointer (write clock -> read clock), the read pointer (back), the RAM's
# outputs (write clock -> read clock). A pointer's top bit is the same in gray and in binary, and
# synthesis keeps one register for both (routed 7e39638: wbin_reg[AW] drives wgray_r1_reg[AW] in
# all six FIFOs), so the bus skew starts at the binary pointer's registers too.

# u_aq (commands) and u_ad (write data): the accelerator's, clk -> uclk
set_max_delay -datapath_only -from $c_clk -to [get_cells {u_aq/wgray_r1_reg[*]}] $t_clk
set_bus_skew -from [get_cells {u_aq/wgray_reg[*] u_aq/wbin_reg[*]}] -to [get_cells {u_aq/wgray_r1_reg[*]}] $k_cu
set_max_delay -datapath_only -from $c_ucl -to [get_cells {u_aq/rgray_w1_reg[*]}] $t_ucl
set_bus_skew -from [get_cells {u_aq/rgray_reg[*] u_aq/rbin_reg[*]*}] -to [get_cells {u_aq/rgray_w1_reg[*]}] $k_cu
set_max_delay -datapath_only -from $c_clk -through [get_pins -quiet -filter {DIRECTION == OUT && IS_LEAF} -of_objects [get_cells -quiet -hierarchical -filter {NAME =~ *u_aq/mem_reg*}]] -to $c_ucl $t_ucl

set_max_delay -datapath_only -from $c_clk -to [get_cells {u_ad/wgray_r1_reg[*]}] $t_clk
set_bus_skew -from [get_cells {u_ad/wgray_reg[*] u_ad/wbin_reg[*]}] -to [get_cells {u_ad/wgray_r1_reg[*]}] $k_cu
set_max_delay -datapath_only -from $c_ucl -to [get_cells {u_ad/rgray_w1_reg[*]}] $t_ucl
set_bus_skew -from [get_cells {u_ad/rgray_reg[*] u_ad/rbin_reg[*]*}] -to [get_cells {u_ad/rgray_w1_reg[*]}] $k_cu
set_max_delay -datapath_only -from $c_clk -through [get_pins -quiet -filter {DIRECTION == OUT && IS_LEAF} -of_objects [get_cells -quiet -hierarchical -filter {NAME =~ *u_ad/mem_reg*}]] -to $c_ucl $t_ucl

# u_ar (read data): the accelerator's, uclk -> clk
set_max_delay -datapath_only -from $c_ucl -to [get_cells {u_ar/wgray_r1_reg[*]}] $t_ucl
set_bus_skew -from [get_cells {u_ar/wgray_reg[*] u_ar/wbin_reg[*]}] -to [get_cells {u_ar/wgray_r1_reg[*]}] $k_cu
set_max_delay -datapath_only -from $c_clk -to [get_cells {u_ar/rgray_w1_reg[*]}] $t_clk
set_bus_skew -from [get_cells {u_ar/rgray_reg[*] u_ar/rbin_reg[*]*}] -to [get_cells {u_ar/rgray_w1_reg[*]}] $k_cu
set_max_delay -datapath_only -from $c_ucl -through [get_pins -quiet -filter {DIRECTION == OUT && IS_LEAF} -of_objects [get_cells -quiet -hierarchical -filter {NAME =~ *u_ar/mem_reg*}]] -to $c_clk $t_clk

# u_xq (commands) and u_xd (write data): XDMA's, xclk -> uclk
set_max_delay -datapath_only -from $c_xcl -to [get_cells {u_xq/wgray_r1_reg[*]}] $t_xcl
set_bus_skew -from [get_cells {u_xq/wgray_reg[*] u_xq/wbin_reg[*]}] -to [get_cells {u_xq/wgray_r1_reg[*]}] $k_xu
set_max_delay -datapath_only -from $c_ucl -to [get_cells {u_xq/rgray_w1_reg[*]}] $t_ucl
set_bus_skew -from [get_cells {u_xq/rgray_reg[*] u_xq/rbin_reg[*]*}] -to [get_cells {u_xq/rgray_w1_reg[*]}] $k_xu
set_max_delay -datapath_only -from $c_xcl -through [get_pins -quiet -filter {DIRECTION == OUT && IS_LEAF} -of_objects [get_cells -quiet -hierarchical -filter {NAME =~ *u_xq/mem_reg*}]] -to $c_ucl $t_ucl

set_max_delay -datapath_only -from $c_xcl -to [get_cells {u_xd/wgray_r1_reg[*]}] $t_xcl
set_bus_skew -from [get_cells {u_xd/wgray_reg[*] u_xd/wbin_reg[*]}] -to [get_cells {u_xd/wgray_r1_reg[*]}] $k_xu
set_max_delay -datapath_only -from $c_ucl -to [get_cells {u_xd/rgray_w1_reg[*]}] $t_ucl
set_bus_skew -from [get_cells {u_xd/rgray_reg[*] u_xd/rbin_reg[*]*}] -to [get_cells {u_xd/rgray_w1_reg[*]}] $k_xu
set_max_delay -datapath_only -from $c_xcl -through [get_pins -quiet -filter {DIRECTION == OUT && IS_LEAF} -of_objects [get_cells -quiet -hierarchical -filter {NAME =~ *u_xd/mem_reg*}]] -to $c_ucl $t_ucl

# u_xr (read data): XDMA's, uclk -> xclk (the RAM is read straight onto x_rdata: its paths end in
# XDMA's clock outside the module)
set_max_delay -datapath_only -from $c_ucl -to [get_cells {u_xr/wgray_r1_reg[*]}] $t_ucl
set_bus_skew -from [get_cells {u_xr/wgray_reg[*] u_xr/wbin_reg[*]}] -to [get_cells {u_xr/wgray_r1_reg[*]}] $k_xu
set_max_delay -datapath_only -from $c_xcl -to [get_cells {u_xr/rgray_w1_reg[*]}] $t_xcl
set_bus_skew -from [get_cells {u_xr/rgray_reg[*] u_xr/rbin_reg[*]*}] -to [get_cells {u_xr/rgray_w1_reg[*]}] $k_xu
set_max_delay -datapath_only -from $c_ucl -through [get_pins -quiet -filter {DIRECTION == OUT && IS_LEAF} -of_objects [get_cells -quiet -hierarchical -filter {NAME =~ *u_xr/mem_reg*}]] -to $c_xcl $t_xcl

# ---------------------------------------------------------------- report_cdc waivers
# report_cdc cannot see the FIFOs' protocol. It reports each endpoint once, with one startpoint:
# routed 7e39638 (RMW 1), per input direction (clk -> uclk, xclk -> uclk), 1214 CDC-13 (unsafe:
# RAM inputs), 124 CDC-1 (unknown: flip-flops through logic) and 59 CDC-15 (clock-enable
# structures). All of them start at an input FIFO's RAM (u_aq / u_ad from clk, u_xq / u_xd from
# xclk: the arbiter reads the heads asynchronously in uclk) and end in the uclk logic that takes a
# head: per port (g_port[*]) the RAM inputs of the output command queue u_oq, the output
# write-data FIFO u_of and the tag FIFO u_tag (data, write enables), the output command register
# (oc, oc_v) and the three FIFOs' write pointers; the arbiter (run, cur_x, rm_busy, rm_x), the
# reads in flight and not yet in their FIFO (a_out, x_out, a_pend, x_pend), the read slots
# (a_seq, x_seq), each master's commands in the output queues (a_nq, x_nq) and the four input
# FIFOs' read pointers. (Two ports, ld-2port routed: the same kinds, 268 CDC-13, 152 CDC-1 and
# 108 CDC-15 at the new endpoints before they were listed here.) Safe by the pointer protocol: an
# entry is read only after the write pointer that covers it has crossed (two uclk flip-flops
# after the source cycle that wrote it; the RAM paths are limited to one uclk period above), and
# it is not rewritten until the read pointer that frees it has crossed back. Every load of a head
# is qualified by its FIFO's registered not-empty (a_ok / x_ok in go; rm_ok's heads are the ones
# its read was issued for), except oc, which also loads while its port's output command register
# is empty or emptying and is used only once a push has loaded it (c_cmd_valid). An empty FIFO's
# head may be changing; it then reaches only RAM data inputs whose write enables are low, arbiter
# terms that the not-empty masks (the port, the head's bank bit, selects a port's room only
# inside go), and oc.
# Also waived: CDC-26, u_ar's head into n_rdata (every clk cycle; used only with n_rvalid, which
# is the FIFO's not-empty), and CDC-6, the gray-coded counts (bus skew above). Not waived:
# CDC-3 (information only), and the paths that leave the module: u_ar's and u_xr's heads are
# read onto n_rdata (registered here) and x_rdata (combinational, to XDMA's R channel: those
# endpoints are XDMA's and need their waiver at the top). Each waiver names both ends of its
# paths, so a new path from these RAMs into other logic is still reported.
set w_src [get_pins -quiet -filter {IS_LEAF && REF_PIN_NAME == CLK} -of_objects [get_cells -quiet -hierarchical -filter {NAME =~ *u_aq/mem_reg* || NAME =~ *u_ad/mem_reg* || NAME =~ *u_xq/mem_reg* || NAME =~ *u_xd/mem_reg*}]]
set w_ram [get_pins -quiet -filter {IS_LEAF && (REF_PIN_NAME == I || REF_PIN_NAME == WE)} -of_objects [get_cells -quiet -hierarchical -filter {NAME =~ *u_oq/mem_reg* || NAME =~ *u_of/mem_reg* || NAME =~ *u_tag/mem_reg*}]]
set w_ff  [get_pins -quiet -filter {DIRECTION == IN && REF_PIN_NAME != C} -of_objects [get_cells -quiet {g_port[*].oc_reg[*] g_port[*].oc_v_reg g_port[*].u_oq/wp_reg* g_port[*].u_of/wp_reg* g_port[*].u_tag/wp_reg* run_reg[*] cur_x_reg rm_busy_reg rm_x_reg a_out_reg[*] x_out_reg[*] a_pend_reg[*] x_pend_reg[*] a_seq_reg[*] x_seq_reg[*] a_nq_reg[*] x_nq_reg[*] u_aq/rbin_reg* u_aq/rgray_reg* u_ad/rbin_reg* u_ad/rgray_reg* u_xq/rbin_reg* u_xq/rgray_reg* u_xd/rbin_reg* u_xd/rgray_reg*}]]
set w_ar  [get_pins -quiet -filter {IS_LEAF && REF_PIN_NAME == CLK} -of_objects [get_cells -quiet -hierarchical -filter {NAME =~ *u_ar/mem_reg*}]]
set w_why "otpu_mem_ch: an input FIFO's head (distributed RAM, read asynchronously in uclk) into the uclk arbiter and output logic; read only after its write pointer crossed, not rewritten before the read pointer crossed back; loads qualified by the FIFO's registered not-empty (oc's value: by c_cmd_valid)"
if {[llength $w_src] && [llength $w_ram]} {
  create_waiver -scoped -type CDC -id {CDC-13} -user "otpu_mem_ch" -description $w_why -from $w_src -to $w_ram
}
if {[llength $w_src] && [llength $w_ff]} {
  create_waiver -scoped -type CDC -id {CDC-1} -user "otpu_mem_ch" -description $w_why -from $w_src -to $w_ff
  create_waiver -scoped -type CDC -id {CDC-15} -user "otpu_mem_ch" -description $w_why -from $w_src -to $w_ff
}
set w_nr  [get_pins -quiet {n_rdata_reg[*]/D}]
if {[llength $w_ar] && [llength $w_nr]} {
  create_waiver -scoped -type CDC -id {CDC-26} -user "otpu_mem_ch" -description "otpu_mem_ch: u_ar's head (distributed RAM, written in uclk) into n_rdata every clk cycle; used only with n_rvalid, the FIFO's registered not-empty, when the entry is stable" -from $w_ar -to $w_nr
}
set w_gs {g_port[*].a_wacc_g_reg[*]/C g_port[*].x_wacc_g_reg[*]/C}
set w_gd {g_port[*].a_wacc_s1_reg[*]/D g_port[*].x_wacc_s1_reg[*]/D}
foreach f {u_aq u_ad u_ar u_xq u_xd u_xr} {
  lappend w_gs $f/wgray_reg\[*\]/C $f/rgray_reg\[*\]/C
  lappend w_gd $f/wgray_r1_reg\[*\]/D $f/rgray_w1_reg\[*\]/D
}
set w_gs [get_pins -quiet $w_gs]
set w_gd [get_pins -quiet $w_gd]
if {[llength $w_gs] && [llength $w_gd]} {
  create_waiver -scoped -type CDC -id {CDC-6} -user "otpu_mem_ch" -description "otpu_mem_ch: gray-coded counts (FIFO pointers, write-accept counters) into ASYNC_REG synchronizers; one bit changes per source cycle, set_bus_skew bounds the skew" -from $w_gs -to $w_gd
}
