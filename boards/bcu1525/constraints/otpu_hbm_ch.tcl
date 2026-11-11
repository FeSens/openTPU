# Clock-domain crossings of the channel bridge (rtl/mem/otpu_hbm_ch.sv), all of them in
# its core half otpu_hbm_cs and its otpu_afifo instances (u_cq, u_wd, g_rd[q].u_rd), as
# boards/ypcb-00338/constraints/otpu_mem_ch.tcl does for otpu_mem_ch. Scoped to the core half, read
# after the clocks exist (Tcl, an unmanaged constraint file):
#   read_xdc -unmanaged -ref otpu_hbm_cs otpu_hbm_ch.tcl
# in implementation, on the synthesized netlist (boards/bcu1525/vivado/create_project_wg.tcl:
# SCOPED_TO_REF, used in implementation only).
#
# The module's clocks: clk (the core clock) and aclk (the memory's AXI clock: the BCU1525's 300 MHz). Every path
# between them inside the module is a synchronizer (ASYNC_REG in the RTL) or data a synchronized
# pointer guards, and each gets set_max_delay -datapath_only here, so all of them are timed
# without clock skew:
#   - gray-coded buses (the FIFOs' pointers, n_wdone's count wacc_g): one source period, and
#     set_bus_skew of the smaller period, so that the bits of one count arrive together;
#   - a FIFO's distributed RAM, written in one clock and read (asynchronously) in the other: one
#     destination period (an entry is read no sooner than two destination cycles after its write
#     pointer crossed);
#   - the reset request (a_req), the hold and the error bit (from the pipeline's stage 0 beside
#     the core half, otpu_hbm_ln): one source period.
# The top level must not make the two clocks asynchronous with set_clock_groups or
# set_false_path (either overrides set_max_delay and leaves the crossings untimed).

# the two clocks, from a register on each (u_cq's write and read pointers; get_ports finds a
# design's top ports only, not a scoped cell's pins). A miss goes to ::otpu::fail too, and the
# build stops in otpu_impl.tcl: in a read_xdc file return and error are only warnings
set c_clk [get_clocks -quiet -of_objects [get_pins -quiet -filter {REF_PIN_NAME == C} -of_objects [get_cells -quiet {u_cq/wgray_reg[0]}]]]
set c_acl [get_clocks -quiet -of_objects [get_pins -quiet -filter {REF_PIN_NAME == C} -of_objects [get_cells -quiet {u_cq/rgray_reg[0]}]]]
if {[llength $c_clk] != 1 || [llength $c_acl] != 1} {
  puts "CRITICAL WARNING: \[otpu_hbm_ch.tcl\] [current_instance .]: clocks '$c_clk' and '$c_acl', expected one each: crossings unconstrained"
  namespace eval ::otpu {}
  lappend ::otpu::fail "otpu_hbm_ch.tcl: [current_instance .]: clocks '$c_clk' and '$c_acl', expected one each: crossings unconstrained"
  return
}
set t_clk [get_property -quiet -min PERIOD $c_clk]
set t_acl [get_property -quiet -min PERIOD $c_acl]
set k_ca  [expr {min($t_clk, $t_acl)}]

# ---------------------------------------------------------------- reset request, hold, error bit
set_max_delay -datapath_only -from $c_clk -to [get_cells a_rs1_reg] $t_clk
set_max_delay -datapath_only -from $c_acl -to [get_cells a_hs1_reg] $t_acl
set_max_delay -datapath_only -from $c_acl -to [get_cells e_s1_reg] $t_acl

# ---------------------------------------------------------------- n_wdone's count (gray)
set_max_delay -datapath_only -from $c_acl -to [get_cells {wacc_s1_reg[*]}] $t_acl
set_bus_skew -from [get_cells {wacc_g_reg[*]}] -to [get_cells {wacc_s1_reg[*]}] $k_ca

# ---------------------------------------------------------------- FIFOs
# Per FIFO: the write pointer (write clock -> read clock), the read pointer (back), the RAM's
# outputs (write clock -> read clock). A pointer's top bit is the same in gray and in binary and
# synthesis may keep one register for both, so the bus skew starts at the binary pointer too.
# u_cq (commands) and u_wd (write data): clk -> aclk; u_rd's K FIFOs (read data, an AXI beat of
# the native beat each): aclk -> clk. (Their cells are g_rd[q].u_rd/..., matched as g_rd?q?.u_rd:
# get_cells takes brackets as a bus index, and "g_rd[0].u_rd" found nothing out of context.)
set f_rd {}
for {set q 0} {$q < 4} {incr q} {
  if {[llength [get_cells -quiet "g_rd?$q?.u_rd/wgray_r1_reg\[*\]"]]} { lappend f_rd "g_rd?$q?.u_rd" }
}
if {![llength $f_rd]} {
  puts "CRITICAL WARNING: \[otpu_hbm_ch.tcl\] [current_instance .]: no u_rd FIFO (g_rd\[q\].u_rd) found: their pointers unconstrained"
  namespace eval ::otpu {}
  lappend ::otpu::fail "otpu_hbm_ch.tcl: [current_instance .]: no u_rd FIFO (g_rd\[q\].u_rd) found: their pointers unconstrained"
}
foreach {f c_w c_r t_w t_r} [concat [list u_cq $c_clk $c_acl $t_clk $t_acl u_wd $c_clk $c_acl $t_clk $t_acl] \
                                    [join [lmap f $f_rd {list $f $c_acl $c_clk $t_acl $t_clk}]]] {
  set_max_delay -datapath_only -from $c_w -to [get_cells "$f/wgray_r1_reg\[*\]"] $t_w
  set_bus_skew -from [get_cells "$f/wgray_reg\[*\] $f/wbin_reg\[*\]"] -to [get_cells "$f/wgray_r1_reg\[*\]"] $k_ca
  set_max_delay -datapath_only -from $c_r -to [get_cells "$f/rgray_w1_reg\[*\]"] $t_r
  set_bus_skew -from [get_cells "$f/rgray_reg\[*\] $f/rbin_reg\[*\]*"] -to [get_cells "$f/rgray_w1_reg\[*\]"] $k_ca
}
# the RAMs' leaf cells (a RAM64M8's RAMA-H; u_rd: all K FIFOs', and at CPY each one's column
# groups, g_wcp.g_c[g].cmem_reg: otpu_afifo WCP)
foreach {f c_w c_r t_r} [list u_cq $c_clk $c_acl $t_acl u_wd $c_clk $c_acl $t_acl u_rd $c_acl $c_clk $t_clk] {
  set_max_delay -datapath_only -from $c_w -through [get_pins -quiet -filter {DIRECTION == OUT && IS_LEAF} -of_objects [get_cells -quiet -hierarchical -filter "NAME =~ *$f/*mem_reg*"]] -to $c_r $t_r
}

# ---------------------------------------------------------------- report_cdc waivers
# u_cq's and u_wd's heads (distributed RAM read asynchronously in aclk) into the aclk logic that
# takes them: the command pump's head register (h_d; loaded with the FIFO's not-empty), the W
# pump's beat register (wp_d, wp_m; loaded at every native beat's end, used only when u_wd was
# not empty then, from the synchronized write pointer) and the FIFOs' read pointers. u_rd's heads into n_rdata every clk cycle, used only with
# n_rvalid (all K FIFOs not empty). The gray-coded buses: one bit per source cycle, bounded by the
# bus skew above.
set w_src [get_pins -quiet -filter {IS_LEAF && REF_PIN_NAME == CLK} -of_objects [get_cells -quiet -hierarchical -filter {NAME =~ *u_cq/mem_reg* || NAME =~ *u_wd/mem_reg*}]]
set w_ff  [get_pins -quiet -filter {DIRECTION == IN && REF_PIN_NAME != C} -of_objects [get_cells -quiet {h_d_reg[*] h_v_reg wp_d_reg[*] wp_m_reg[*] u_cq/rbin_reg* u_cq/rgray_reg* u_wd/rbin_reg* u_wd/rgray_reg*}]]
set w_why "otpu_hbm_cs: an input FIFO's head (distributed RAM, read asynchronously in aclk) into the aclk head / W-beat registers; used only after its write pointer crossed, not rewritten before the read pointer crossed back"
if {[llength $w_src] && [llength $w_ff]} {
  create_waiver -scoped -type CDC -id {CDC-1} -user "otpu_hbm_cs" -description $w_why -from $w_src -to $w_ff
  create_waiver -scoped -type CDC -id {CDC-15} -user "otpu_hbm_cs" -description $w_why -from $w_src -to $w_ff
}
set w_rd [get_pins -quiet -filter {IS_LEAF && REF_PIN_NAME == CLK} -of_objects [get_cells -quiet -hierarchical -filter {NAME =~ *u_rd/*mem_reg*}]]
set w_nr [get_pins -quiet {n_rdata_reg[*]/D}]
if {[llength $w_rd] && [llength $w_nr]} {
  create_waiver -scoped -type CDC -id {CDC-26} -user "otpu_hbm_cs" -description "otpu_hbm_cs: u_rd's heads (distributed RAM, written in aclk) into n_rdata every clk cycle; used only with n_rvalid, all K FIFOs not empty, when the entries are stable" -from $w_rd -to $w_nr
}
set w_gs {wacc_g_reg[*]/C}
set w_gd {wacc_s1_reg[*]/D}
foreach f [concat u_cq u_wd $f_rd] {
  lappend w_gs $f/wgray_reg\[*\]/C $f/rgray_reg\[*\]/C
  lappend w_gd $f/wgray_r1_reg\[*\]/D $f/rgray_w1_reg\[*\]/D
}
set w_gs [get_pins -quiet $w_gs]
set w_gd [get_pins -quiet $w_gd]
if {[llength $w_gs] && [llength $w_gd]} {
  create_waiver -scoped -type CDC -id {CDC-6} -user "otpu_hbm_cs" -description "otpu_hbm_cs: gray-coded counts (FIFO pointers, n_wdone's count) into ASYNC_REG synchronizers; one bit changes per source cycle, set_bus_skew bounds the skew" -from $w_gs -to $w_gd
}
# this core half constrained (otpu_impl.tcl counts them against the build's otpu_hbm_cs cells)
if {[llength $c_clk] == 1 && [llength $c_acl] == 1 && [llength $f_rd]} {
  namespace eval ::otpu {}
  incr ::otpu::ok(otpu_hbm_ch)
}
