# U9, measured in s6. U12 and both cages' REFCLK0 pins are deliberately absent.
create_clock -period 6.206 -name qsfp0_mgt_refclk_1 [get_ports qsfp0_mgt_refclk_1_p]
