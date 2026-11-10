# post_impl.tcl's pulse-width skew checks (the MIG XIPHY's Min / Max Skew: B1's WPWS -0.177 ns was a
# Min Skew between a byte's RXTX_BITSLICE D[1] and D[2] routes, 7 such checks failing over the four
# DIMMs' PHYs). ::otpu::pulse_skew RPT: the failing ones of report_pulse_width's RPT, each check once
# (its kind, corner, site and pin), as {count worst-slack "worst check"}; {0 "" ""} when none fail.
namespace eval ::otpu {}
proc ::otpu::pulse_skew {rpt} {
  set seen [dict create]
  set worst ""
  set wl ""
  set fh [open $rpt]
  while {[gets $fh ln] >= 0} {
    # Check Type, Corner, Lib Pin, Reference Pin, Required, Actual, Slack, Location, Pin
    if {![regexp {^(Min|Max) Skew\s+(\S+)\s+(\S+)\s+(\S+)\s+\S+\s+\S+\s+(-?[0-9.]+)\s+(\S+)\s+(\S+)} $ln \
            - k corner lib ref s loc pin] || $s >= 0} { continue }
    set key "$k $corner $loc $pin"
    if {[dict exists $seen $key]} { continue }
    dict set seen $key $s
    if {$worst eq "" || $s < $worst} {
      set worst $s
      set wl "$k Skew $corner $lib / $ref $s ns at $loc $pin"
    }
  }
  close $fh
  return [list [dict size $seen] $worst $wl]
}
