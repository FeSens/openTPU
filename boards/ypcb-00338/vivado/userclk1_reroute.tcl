# PCIe Gen2 (PCIE_GEN 2), on the routed design (build.tcl sources it after open_run impl_1;
# reads $out): the PCIe block's 500 MHz user clock (userclk1) times its paths into its own TX / RX
# block RAMs, which the IP's XDC places next to PCIE_X0Y0, so they are route only and neither
# placement nor phys_opt moves them (the FAST probe 81432ad7: -0.045 ns, 0.93 of 1.42 ns route).
# If one fails, the nets of the failing userclk1 paths are routed again, constraint driven
# (route_design -auto_delay); the result is kept only if userclk1's worst slack improves and
# neither the other clocks' setup nor any hold gets worse below zero. Otherwise, or on an error,
# the routed design is reopened from the checkpoint written before.
proc otpu_ws {args} {
  set p [lindex [get_timing_paths -quiet -max_paths 1 -nworst 1 {*}$args] 0]
  if {$p eq ""} { return 1e9 }
  return [get_property SLACK $p]
}
set uc1 [get_clocks -quiet userclk1]
if {[info exists ::env(PCIE_GEN)] && $::env(PCIE_GEN) eq "2" && [llength $uc1] == 1} {
  set bad [get_timing_paths -quiet -setup -to $uc1 -max_paths 100 -nworst 1 -slack_lesser_than 0]
  if {[llength $bad]} {
    set others [get_clocks -quiet -filter {NAME != userclk1}]
    set u0 [otpu_ws -setup -to $uc1]
    set o0 [otpu_ws -setup -to $others]
    set h0 [otpu_ws -hold]
    set nets [get_nets -quiet -of_objects $bad -filter {TYPE == SIGNAL}]
    set ckpt $out/pre_userclk1.dcp
    file delete -force $ckpt
    puts "userclk1: [llength $bad] failing setup paths (WNS $u0 ns): [llength $nets] nets routed again"
    set keep 0
    if {[catch {
      write_checkpoint -force $ckpt
      route_design -unroute -nets $nets
      route_design -nets $nets -auto_delay
      set u1 [otpu_ws -setup -to $uc1]
      set o1 [otpu_ws -setup -to $others]
      set h1 [otpu_ws -hold]
      set open [llength [filter $nets {ROUTE_STATUS != ROUTED}]]
      puts "userclk1 rerouted: userclk1 $u0 -> $u1, other clocks $o0 -> $o1, hold $h0 -> $h1 ns,\
            $open nets not routed"
      set keep [expr {$u1 > $u0 && $open == 0 && ($o1 >= $o0 || $o1 >= 0) && ($h1 >= $h0 || $h1 >= 0)}]
    } msg]} {
      puts "CRITICAL WARNING: userclk1 reroute failed: $msg"
    }
    if {!$keep && [file exists $ckpt]} {
      puts "userclk1 reroute not kept: the routed design reopened from $ckpt"
      close_design
      open_checkpoint $ckpt
    }
  }
}
