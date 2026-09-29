# Clock-domain crossings of otpu_mem_ch (rtl/boards/ypcb-00338/otpu_mem_ch.sv) and its six
# otpu_afifo instances. Scoped to the module and read after the clocks exist, as Xilinx's own
# xpm_cdc constraint files are (Tcl variables and get_property):
#   non-project:  read_xdc -unmanaged -ref otpu_mem_ch otpu_mem_ch.xdc     (after create_clock)
#   project:      the same file as an unmanaged Tcl constraint, SCOPED_TO_REF otpu_mem_ch,
#                 PROCESSING_ORDER LATE (not wired into build.tcl yet)
# tools/memch_ooc.tcl checks it out of context (report_cdc, report_methodology).
#
# The module's clocks: clk (the core clock), uclk (the controller's user clock: LiteDRAM's sys or
# the MIG's ui_clk), xclk (XDMA's axi_aclk). Every path between two of them inside the module is
# a synchronizer (ASYNC_REG in the RTL) or data that a synchronized pointer guards, and each gets
# set_max_delay -datapath_only here, so all of them are timed, without clock skew:
#   - gray-coded buses: the FIFOs' pointers and the write-accept counters behind n_wdone and
#     XDMA's B. One source period, and set_bus_skew of the smaller period, so that the bits of one
#     count (which moves one bit per source cycle) arrive together;
#   - a FIFO's distributed RAM, written in one clock and read (asynchronously) in the other: one
#     destination period. An entry is read no sooner than two destination cycles after its write
#     pointer crossed, and the pointer is written in the same source cycle as the entry;
#   - the reset and hold synchronizers (a level held for many cycles): one source period.
# The top level must therefore not make these clock pairs asynchronous with set_clock_groups or
# set_false_path (either takes priority over set_max_delay and leaves the crossings untimed), and
# the resets on rst and xrst should come from registers in clk and xclk.

set c_clk [get_clocks -quiet -of_objects [get_ports clk]]
set c_ucl [get_clocks -quiet -of_objects [get_ports uclk]]
set c_xcl [get_clocks -quiet -of_objects [get_ports xclk]]
set t_clk [get_property -quiet -min PERIOD $c_clk]
set t_ucl [get_property -quiet -min PERIOD $c_ucl]
set t_xcl [get_property -quiet -min PERIOD $c_xcl]
set k_cu  [expr {min($t_clk, $t_ucl)}]
set k_xu  [expr {min($t_xcl, $t_ucl)}]

# ---------------------------------------------------------------- resets and holds
# rst (clk) and xrst (xclk) into uclk; the per-master holds (uclk) into clk and xclk
set_max_delay -datapath_only -from $c_clk -to [get_cells a_rs1_reg] $t_clk
set_max_delay -datapath_only -from $c_xcl -to [get_cells x_rs1_reg] $t_xcl
set_max_delay -datapath_only -from $c_ucl -to [get_cells a_hs1_reg] $t_ucl
set_max_delay -datapath_only -from $c_ucl -to [get_cells x_hs1_reg] $t_ucl

# ---------------------------------------------------------------- write-accept counters (gray)
# a_wacc (n_wdone) uclk -> clk, x_wacc (XDMA's B) uclk -> xclk
set_max_delay -datapath_only -from $c_ucl -to [get_cells {a_wacc_s1_reg[*]}] $t_ucl
set_bus_skew -from [get_cells {a_wacc_g_reg[*]}] -to [get_cells {a_wacc_s1_reg[*]}] $k_cu
set_max_delay -datapath_only -from $c_ucl -to [get_cells {x_wacc_s1_reg[*]}] $t_ucl
set_bus_skew -from [get_cells {x_wacc_g_reg[*]}] -to [get_cells {x_wacc_s1_reg[*]}] $k_xu

# ---------------------------------------------------------------- FIFOs
# Per FIFO: the write pointer (write clock -> read clock), the read pointer (back), the RAM's
# outputs (write clock -> read clock).

# u_aq (commands) and u_ad (write data): the accelerator's, clk -> uclk
set_max_delay -datapath_only -from $c_clk -to [get_cells {u_aq/wgray_r1_reg[*]}] $t_clk
set_bus_skew -from [get_cells {u_aq/wgray_reg[*]}] -to [get_cells {u_aq/wgray_r1_reg[*]}] $k_cu
set_max_delay -datapath_only -from $c_ucl -to [get_cells {u_aq/rgray_w1_reg[*]}] $t_ucl
set_bus_skew -from [get_cells {u_aq/rgray_reg[*]}] -to [get_cells {u_aq/rgray_w1_reg[*]}] $k_cu
set_max_delay -datapath_only -from $c_clk -through [get_pins -quiet -filter {DIRECTION == OUT && IS_LEAF} -of_objects [get_cells -quiet -hierarchical -filter {NAME =~ *u_aq/mem_reg*}]] -to $c_ucl $t_ucl

set_max_delay -datapath_only -from $c_clk -to [get_cells {u_ad/wgray_r1_reg[*]}] $t_clk
set_bus_skew -from [get_cells {u_ad/wgray_reg[*]}] -to [get_cells {u_ad/wgray_r1_reg[*]}] $k_cu
set_max_delay -datapath_only -from $c_ucl -to [get_cells {u_ad/rgray_w1_reg[*]}] $t_ucl
set_bus_skew -from [get_cells {u_ad/rgray_reg[*]}] -to [get_cells {u_ad/rgray_w1_reg[*]}] $k_cu
set_max_delay -datapath_only -from $c_clk -through [get_pins -quiet -filter {DIRECTION == OUT && IS_LEAF} -of_objects [get_cells -quiet -hierarchical -filter {NAME =~ *u_ad/mem_reg*}]] -to $c_ucl $t_ucl

# u_ar (read data): the accelerator's, uclk -> clk
set_max_delay -datapath_only -from $c_ucl -to [get_cells {u_ar/wgray_r1_reg[*]}] $t_ucl
set_bus_skew -from [get_cells {u_ar/wgray_reg[*]}] -to [get_cells {u_ar/wgray_r1_reg[*]}] $k_cu
set_max_delay -datapath_only -from $c_clk -to [get_cells {u_ar/rgray_w1_reg[*]}] $t_clk
set_bus_skew -from [get_cells {u_ar/rgray_reg[*]}] -to [get_cells {u_ar/rgray_w1_reg[*]}] $k_cu
set_max_delay -datapath_only -from $c_ucl -through [get_pins -quiet -filter {DIRECTION == OUT && IS_LEAF} -of_objects [get_cells -quiet -hierarchical -filter {NAME =~ *u_ar/mem_reg*}]] -to $c_clk $t_clk

# u_xq (commands) and u_xd (write data): XDMA's, xclk -> uclk
set_max_delay -datapath_only -from $c_xcl -to [get_cells {u_xq/wgray_r1_reg[*]}] $t_xcl
set_bus_skew -from [get_cells {u_xq/wgray_reg[*]}] -to [get_cells {u_xq/wgray_r1_reg[*]}] $k_xu
set_max_delay -datapath_only -from $c_ucl -to [get_cells {u_xq/rgray_w1_reg[*]}] $t_ucl
set_bus_skew -from [get_cells {u_xq/rgray_reg[*]}] -to [get_cells {u_xq/rgray_w1_reg[*]}] $k_xu
set_max_delay -datapath_only -from $c_xcl -through [get_pins -quiet -filter {DIRECTION == OUT && IS_LEAF} -of_objects [get_cells -quiet -hierarchical -filter {NAME =~ *u_xq/mem_reg*}]] -to $c_ucl $t_ucl

set_max_delay -datapath_only -from $c_xcl -to [get_cells {u_xd/wgray_r1_reg[*]}] $t_xcl
set_bus_skew -from [get_cells {u_xd/wgray_reg[*]}] -to [get_cells {u_xd/wgray_r1_reg[*]}] $k_xu
set_max_delay -datapath_only -from $c_ucl -to [get_cells {u_xd/rgray_w1_reg[*]}] $t_ucl
set_bus_skew -from [get_cells {u_xd/rgray_reg[*]}] -to [get_cells {u_xd/rgray_w1_reg[*]}] $k_xu
set_max_delay -datapath_only -from $c_xcl -through [get_pins -quiet -filter {DIRECTION == OUT && IS_LEAF} -of_objects [get_cells -quiet -hierarchical -filter {NAME =~ *u_xd/mem_reg*}]] -to $c_ucl $t_ucl

# u_xr (read data): XDMA's, uclk -> xclk (the RAM is read straight onto x_rdata: its paths end in
# XDMA's clock outside the module)
set_max_delay -datapath_only -from $c_ucl -to [get_cells {u_xr/wgray_r1_reg[*]}] $t_ucl
set_bus_skew -from [get_cells {u_xr/wgray_reg[*]}] -to [get_cells {u_xr/wgray_r1_reg[*]}] $k_xu
set_max_delay -datapath_only -from $c_xcl -to [get_cells {u_xr/rgray_w1_reg[*]}] $t_xcl
set_bus_skew -from [get_cells {u_xr/rgray_reg[*]}] -to [get_cells {u_xr/rgray_w1_reg[*]}] $k_xu
set_max_delay -datapath_only -from $c_ucl -through [get_pins -quiet -filter {DIRECTION == OUT && IS_LEAF} -of_objects [get_cells -quiet -hierarchical -filter {NAME =~ *u_xr/mem_reg*}]] -to $c_xcl $t_xcl
