# MIG 7-series channel 0 as built today (DDR3-1066, 72-bit ECC, ROW_BANK_COLUMN), AXI vs native
# user interface: out-of-context synthesis (non-project mode), hierarchical utilization
# (stream L, docs/litedram.md).
set here [file normalize [file dirname [info script]]]
set_part xc7k480t-ffg1156-2
file mkdir $here/ip
foreach v {axi native} {
  create_ip -name mig_7series -vendor xilinx.com -library ip -module_name mig_$v -dir $here/ip
  set dir [get_property IP_DIR [get_ips mig_$v]]
  file copy -force $here/mig_$v.prj $dir/mig_$v.prj
  set_property -dict [list CONFIG.XML_INPUT_FILE mig_$v.prj CONFIG.RESET_BOARD_INTERFACE {Custom}] [get_ips mig_$v]
  generate_target all [get_ips mig_$v]
  synth_ip [get_ips mig_$v]
  open_checkpoint $dir/mig_$v.dcp
  report_utilization -hierarchical -hierarchical_depth 8 -file $here/mig_${v}_util_hier.rpt
  report_utilization -file $here/mig_${v}_util.rpt
  close_design
}
