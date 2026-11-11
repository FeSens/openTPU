# The s6 v6 Aurora IP stage (qsfp-link 9133ed3f), fixed at 25.78125 Gb/s.
# Requires ip_vlnv (bd_wg.tcl), out; the shared GT commons remain in our RTL.
file mkdir $out/ip
  foreach {nm quad lanes} {aur_q0 12 {X1Y48 X1Y49 X1Y50 X1Y51} aur_q1 11 {X1Y44 X1Y45 X1Y46 X1Y47}} {
    create_ip -vlnv [ip_vlnv aurora_64b66b] -module_name $nm -dir $out/ip
    set ip [get_ips $nm]
    set_property -dict [list CONFIG.C_GT_TYPE {GTY} CONFIG.C_UCOLUMN_USED {right} CONFIG.C_AURORA_LANES {4}] $ip
    set_property -dict [list CONFIG.C_START_QUAD Quad_X1Y$quad CONFIG.C_START_LANE [lindex $lanes 0]] $ip
    set_property -dict [list CONFIG.CHANNEL_ENABLE $lanes] $ip
    set_property -dict [list CONFIG.C_LINE_RATE 25.78125] $ip
    set_property -dict [list CONFIG.C_REFCLK_FREQUENCY {161.1328125}] $ip
    set_property -dict [list CONFIG.C_INIT_CLK {100} CONFIG.interface_mode {Streaming} CONFIG.flow_mode {None} CONFIG.crc_mode {false}] $ip
    set_property -dict [list CONFIG.drp_mode {Native} CONFIG.SupportLevel {0} CONFIG.SINGLEEND_INITCLK {true} CONFIG.SINGLEEND_GTREFCLK {true}] $ip
    foreach k {C_START_QUAD C_START_LANE CHANNEL_ENABLE C_LINE_RATE C_REFCLK_FREQUENCY SupportLevel interface_mode} {
      puts "x2 IP $nm $k = [get_property CONFIG.$k $ip]"
    }
    if {[get_property CONFIG.C_LINE_RATE $ip] != 25.78125 || [get_property CONFIG.C_REFCLK_FREQUENCY $ip] != 161.1328125} {
      error "$nm: rate / reference not as asked"
    }
  }
generate_target all [get_ips {aur_q0 aur_q1}]
# create_ip_run takes one sub-design (12-3445 on a collection)
foreach ip [get_ips {aur_q0 aur_q1}] { create_ip_run $ip }
