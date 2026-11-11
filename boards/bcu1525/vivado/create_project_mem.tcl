# Create the Vivado project of the BCU1525's memory-test image (docs/bcu1525.md section 14): the
# four UDIMMs' LiteDRAM cores (boards/bcu1525/litedram_udimm/otpu_ld4.v), the thermal governor, the
# BMC's UART bridge and the block design otpu_bdm (bd_mem.tcl: JTAG-to-AXI, the SYSMON) under
# rtl/boards/bcu1525/otpu_fpga_top_bcu_mem.sv. No PCIe and no accelerator.
#   vivado -mode batch -source create_project_mem.tcl -tclargs OUT_DIR
# (run_vivado.sh IMAGE=mem; then build.tcl as for the full image). Environment: OTPU_IP_CACHE.

set here [file normalize [file dirname [info script]]]
set root [file normalize $here/../../..]
set out [expr {[llength $argv] > 0 ? [file normalize [lindex $argv 0]] : "$root/build/vivado_bcu1525_mem"}]
set PART xcvu9p-fsgd2104-2L-e
set ld $root/boards/bcu1525/litedram_udimm

create_project -force otpu $out -part $PART
if {[info exists ::env(OTPU_IP_CACHE)] && $::env(OTPU_IP_CACHE) ne ""} {
  file mkdir $::env(OTPU_IP_CACHE)
  config_ip_cache -use_cache_location $::env(OTPU_IP_CACHE)
}
set_property target_language Verilog [current_project]

# otpu_ctrl.sv for otpu_axil_iso (otpu_ctrl itself is not instantiated)
set rtl [list rtl/shell/otpu_ctrl.sv rtl/boards/bcu1525/otpu_bmc_watch.sv \
  rtl/boards/bcu1525/otpu_bmc_uart.sv \
  rtl/boards/bcu1525/otpu_bcu_gov.sv rtl/boards/bcu1525/otpu_axil_off.sv \
  rtl/boards/bcu1525/otpu_fpga_top_bcu_mem.sv]
foreach f $rtl {
  add_files -norecurse $root/$f
  set_property file_type SystemVerilog [get_files $root/$f]
}
add_files -norecurse $ld/otpu_ld4.v
foreach f [glob $ld/otpu_ld4_*.init] {
  add_files -norecurse $f
  set_property FILE_TYPE {Memory Initialization Files} [get_files $f]
}
set_property verilog_define {SYNTHESIS} [get_filesets sources_1]

source $here/bd_mem.tcl
make_wrapper -files [get_files otpu_bdm.bd] -top
add_files -norecurse [glob $out/otpu.gen/sources_1/bd/otpu_bdm/hdl/otpu_bdm_wrapper.v]
set_property top otpu_fpga_top_bcu_mem [current_fileset]

# ---- constraints: the board's pins for the ports this top has (bcu1525_pins.xdc names PCIe's and
# I2C's too: their lines are left out), the configuration settings (OVERTEMPSHUTDOWN among them),
# the UDIMMs' pins, each core's own XDC (scoped to otpu_ld4), then bcu1525_mem_top.tcl late
set cons $root/boards/bcu1525/constraints
set keep {clk300_0_p clk300_0_n clk300_1_p clk300_1_n clk300_2_p clk300_2_n clk300_3_p clk300_3_n
          led uart_rx uart_tx bmc_tx bmc_rx}
set fi [open $cons/bcu1525_pins.xdc]
set fo [open $out/bcu1525_mem_pins.xdc w]
set dropped {}
foreach l [split [read $fi] "\n"] {
  if {[regexp {get_ports \{?([A-Za-z0-9_]+)} $l -> port] && $port ni $keep} {
    lappend dropped $port
    continue
  }
  puts $fo $l
}
close $fi
close $fo
puts "bcu1525_mem_pins.xdc: left out [lsort -unique $dropped]"
set xdcs [list $out/bcu1525_mem_pins.xdc]
foreach d {0 1 2 3} { lappend xdcs $cons/ddr4_c$d.udimm.pins.xdc }
add_files -fileset constrs_1 -norecurse [concat $xdcs [list $ld/otpu_ld4.xdc]]
set_property SCOPED_TO_REF otpu_ld4 [get_files $ld/otpu_ld4.xdc]
set late $cons/bcu1525_mem_top.tcl
add_files -fileset constrs_1 -norecurse $late
set_property PROCESSING_ORDER LATE [get_files $late]
set_property USED_IN_SYNTHESIS false [get_files $late]

set_property strategy {Vivado Synthesis Defaults} [get_runs synth_1]
set_property strategy {Vivado Implementation Defaults} [get_runs impl_1]
set_property report_strategy {No Reports} [get_runs synth_1]
set_property report_strategy {No Reports} [get_runs impl_1]
puts "project created in $out (BCU1525 memory-test image)"
