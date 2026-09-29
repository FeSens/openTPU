# XDMA as an RTL IP (not in a block design), configured as boards/ypcb-00338/vivado/bd_native.tcl's
# xdma_0
# but subsystem 4C44 ("LD": the LiteDRAM test image, so the openTPU tools do not take it for theirs).
# Sourced by the LiteDRAM test image's Vivado build (pre-synthesis) and, alone, to get the port list.
proc otpu_xdma_ip {dir} {
  create_ip -name xdma -vendor xilinx.com -library ip -module_name xdma_0 -dir $dir
  set_property -dict [list \
    CONFIG.mode_selection {Advanced} \
    CONFIG.pl_link_cap_max_link_width {X8} \
    CONFIG.pl_link_cap_max_link_speed {2.5_GT/s} \
    CONFIG.axi_data_width {128_bit} \
    CONFIG.axisten_freq {125} \
    CONFIG.ref_clk_freq {100_MHz} \
    CONFIG.pf0_device_id {7028} \
    CONFIG.pf0_class_code_base {12} CONFIG.pf0_class_code_sub {00} \
    CONFIG.pf0_class_code_interface {00} \
    CONFIG.pf0_subsystem_vendor_id {10EE} CONFIG.pf0_subsystem_id {4C44} \
    CONFIG.pf0_revision_id {01} \
    CONFIG.xdma_rnum_chnl {1} CONFIG.xdma_wnum_chnl {1} \
    CONFIG.axilite_master_en {true} CONFIG.axilite_master_size {1} \
    CONFIG.axilite_master_scale {Megabytes} \
    CONFIG.pciebar2axibar_axil_master {0x00000000} \
    CONFIG.xdma_axi_intf_mm {AXI_Memory_Mapped} \
    CONFIG.plltype {QPLL1} \
  ] [get_ips xdma_0]
}
# alone: vivado -mode batch -source xdma_ip.tcl -tclargs template (writes ip/xdma_0/xdma_0.veo)
if {[info exists ::argv] && [lindex $::argv 0] eq "template"} {
  set_part xc7k480t-ffg1156-2
  otpu_xdma_ip [pwd]/ip
  generate_target instantiation_template [get_ips xdma_0]
}
