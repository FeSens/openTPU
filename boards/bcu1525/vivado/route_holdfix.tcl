# impl_resume.tcl's OTPU_WG_PREROUTE=route_holdfix.tcl: the router goes on fixing hold when the
# violators before the route are many, where it would give hold fixing up (Route 35-514: B1's
# acc_clk / acc_clk2x pair from two MMCM outputs, 84% of its THS). Its own resolution; the route
# can take far longer.
set_param route.enableHoldExpnBailout 0
puts "route_holdfix.tcl: route.enableHoldExpnBailout [get_param route.enableHoldExpnBailout]"
