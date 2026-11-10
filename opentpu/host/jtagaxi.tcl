# The Vivado side of opentpu/host/jtagaxi.py: a line protocol on stdin / stdout over one card's JTAG
# (its FT4232H, through the hardware server on localhost:3121). Each command is one line; its answer
# is one line "@@ OK ..." or "@@ ERR ...", other lines are Vivado's own. Addresses and data in hex.
#   open SERIAL        the target whose name holds SERIAL (an FT4232H's serial), its xcvu9p
#   prog BIT           program it (volatile), then refresh: OK <JTAG-to-AXI cores>
#   axi                the JTAG-to-AXI cores now (after a refresh)
#   axis               which is which: OK <AXI-Lite core> <AXI4 core or -> <their PROTOCOLs>
#   r A / rm A...      read 32-bit words (the AXI-Lite core): OK D...
#   w A D / wm A D...  write words: OK
#   mr A N...          read bursts of N 32-bit words at A (the AXI4 core; 64-bit addresses, each
#                      burst within 4 KB and N <= 256: the caller's): OK D... in address order
#   mw A N D... ...    write bursts, each N words in address order: OK
#   order 0|1          a burst's DATA with its first beat on the right (1, the default) or left
#   mon                the SYSMON over JTAG: OK T VCCINT VCCAUX VCCBRAM (C, V)
#   migs LTX FILE      every DDR4 MIG core's calibration (the probes file LTX names them; the
#                      debug hub needs its clock: on the WG image a MIG's, so a DIMM on):
#                      refresh_hw_mig, then report_hw_mig (the calibration's status and
#                      margins) and report_property of each into FILE; OK N CELL|PROP=VALUE,...
#                      per core (its block design cell, its *STAT* properties)
#   done               DONE per SLR: OK d0 d1 d2
#   boot               boot_hw_device (JPROGRAM: the flash's configuration), then refresh: OK
#   quit
# An AXI error response (SLVERR, DECERR) or a timeout answers ERR with run_hw_axi's message.
set dev ""
set axi ""
set mem ""
set lsb_right 1
proc reply {s} { puts "@@ $s"; flush stdout }
# The AXI-Lite core is the register master (the memory-test image's only one; the full image's
# jtag_axi_0), an AXI4 one the memory master (the full image's jtag_axi_1). Without a PROTOCOL
# property to tell them apart: the first core is the AXI-Lite one, a second the AXI4 one.
proc refresh {} {
  global dev axi mem
  refresh_hw_device -update_hw_probes false $dev
  set all [get_hw_axis -quiet -of_objects $dev]
  set axi ""
  set mem ""
  foreach a $all {
    set p [string toupper [get_property -quiet PROTOCOL $a]]
    if {[string match *LITE* $p]} {
      if {$axi eq ""} { set axi $a }
    } elseif {$p ne "" && $mem eq ""} {
      set mem $a
    }
  }
  if {$axi eq ""} { set axi [lindex $all 0] }
  if {$mem eq "" && [llength $all] > 1} { set mem [lindex [lsearch -all -inline -not -exact $all $axi] 0] }
  return [llength $all]
}
proc need_axi {} {
  global axi
  if {$axi eq ""} { error "no JTAG-to-AXI core (refresh after programming)" }
}
proc need_mem {} {
  global mem
  if {$mem eq ""} { error "no AXI4 JTAG-to-AXI core (the full image's jtag_axi_1)" }
}
# AxCACHE 3 (bufferable, modifiable): the SmartConnect packs the 32-bit beats into its masters'
# full-width ones (otpu_mem_ch takes whole beats)
proc mem_rd {args} {
  global mem lsb_right
  need_mem
  set txns {}
  set lens {}
  set i 0
  foreach {a n} $args {
    lappend txns [create_hw_axi_txn -force -quiet mr$i $mem -type read -address $a -len $n -cache 3]
    lappend lens $n
    incr i
  }
  run_hw_axi -quiet $txns
  set out {}
  foreach t $txns n $lens {
    set d [string map {_ "" " " ""} [get_property DATA $t]]
    if {[string length $d] != 8 * $n} { error "burst $t: [string length $d] hex digits for $n words" }
    set w {}
    for {set k 0} {$k < $n} {incr k} { lappend w [string range $d [expr {8 * $k}] [expr {8 * $k + 7}]] }
    if {$lsb_right} { set w [lreverse $w] }
    lappend out {*}$w
  }
  return $out
}
proc mem_wr {args} {
  global mem lsb_right
  need_mem
  set txns {}
  set i 0
  while {[llength $args]} {
    set args [lassign $args a n]
    set w [lrange $args 0 [expr {$n - 1}]]
    set args [lrange $args $n end]
    if {[llength $w] != $n} { error "burst at $a: [llength $w] words for $n" }
    if {$lsb_right} { set w [lreverse $w] }
    lappend txns [create_hw_axi_txn -force -quiet mw$i $mem -type write -address $a -len $n \
                    -cache 3 -data [join $w _]]
    incr i
  }
  run_hw_axi -quiet $txns
}
proc rd_many {addrs} {
  global axi
  need_axi
  set txns {}
  set i 0
  foreach a $addrs {
    lappend txns [create_hw_axi_txn -force -quiet r$i $axi -type read -address [format %08x 0x$a] -len 1]
    incr i
  }
  run_hw_axi -quiet $txns
  set out {}
  foreach t $txns { lappend out [get_property DATA $t] }
  return $out
}
proc wr_many {pairs} {
  global axi
  need_axi
  set txns {}
  set i 0
  foreach {a d} $pairs {
    lappend txns [create_hw_axi_txn -force -quiet w$i $axi -type write -address [format %08x 0x$a] \
                    -data [format %08x 0x$d] -len 1]
    incr i
  }
  run_hw_axi -quiet $txns
}

open_hw_manager
connect_hw_server -url localhost:3121
reply "OK READY"
while {[gets stdin line] >= 0} {
  set args [lassign [split [string trim $line]] cmd]
  if {$cmd eq ""} continue
  if {[catch {
    switch -- $cmd {
      open {
        set t [lindex [get_hw_targets -quiet *[lindex $args 0]*] 0]
        if {$t eq ""} { error "no target *[lindex $args 0]* in [get_hw_targets]" }
        open_hw_target $t
        set dev [lindex [get_hw_devices xcvu9p*] 0]
        current_hw_device $dev
        reply "OK $t $dev [refresh]"
      }
      prog {
        set_property PROBES.FILE {} $dev
        set_property FULL_PROBES.FILE {} $dev
        set_property PROGRAM.FILE [lindex $args 0] $dev
        program_hw_devices $dev
        reply "OK [refresh]"
      }
      axi { reply "OK [refresh]" }
      axis {
        set m [expr {$mem eq "" ? "-" : $mem}]
        reply "OK $axi $m [string map {" " ""} [get_property -quiet PROTOCOL $axi]] [expr {$mem eq "" ? "-" : [string map {" " ""} [get_property -quiet PROTOCOL $mem]]}]"
      }
      mr { reply "OK [mem_rd {*}$args]" }
      mw { mem_wr {*}$args; reply OK }
      order { set lsb_right [lindex $args 0]; reply OK }
      r { reply "OK [rd_many $args]" }
      rm { reply "OK [rd_many $args]" }
      w { wr_many $args; reply OK }
      wm { wr_many $args; reply OK }
      mon {
        set m [lindex [get_hw_sysmons -of $dev] 0]
        refresh_hw_sysmon $m
        reply "OK [get_property TEMPERATURE $m] [get_property VCCINT $m] [get_property VCCAUX $m] [get_property VCCBRAM $m]"
      }
      migs {
        lassign $args ltx f
        set_property PROBES.FILE $ltx $dev
        set_property FULL_PROBES.FILE $ltx $dev
        refresh_hw_device $dev
        set ms [get_hw_migs -quiet -of_objects $dev]
        if {$ms eq ""} { set ms [get_hw_migs -quiet] }
        set fh [open $f w]
        set o {}
        foreach m $ms {
          if {[catch {refresh_hw_mig $m} e]} { puts $fh "refresh_hw_mig $m: $e" }
          puts $fh "=== $m"
          if {[catch {puts $fh [report_hw_mig -return_string $m]} e]} {
            puts $fh "report_hw_mig -return_string: $e (its report in the Vivado log)"
            catch {report_hw_mig $m}
          }
          puts $fh [report_property -all -return_string $m]
          set st {}
          foreach p [list_property $m] {
            if {[string match -nocase *STAT* $p]} {
              lappend st "$p=[string map {" " _ "," ";" "|" "/"} [get_property $p $m]]"
            }
          }
          lappend o "[get_property CELL_NAME $m]|[join $st ,]"
        }
        close $fh
        reply "OK [llength $o] [join $o]"
      }
      done {
        refresh_hw_device -update_hw_probes false $dev
        set o {}
        foreach s {SLR0 SLR1 SLR2} { lappend o [get_property REGISTER.CONFIG_STATUS.$s.BIT\[14\]_DONE_PIN $dev] }
        reply "OK $o"
      }
      boot {
        boot_hw_device $dev
        after 3000
        set axi ""
        set mem ""
        refresh_hw_device -update_hw_probes false $dev
        reply OK
      }
      quit { reply OK; break }
      default { error "unknown command $cmd" }
    }
  } msg]} {
    reply "ERR [string map {"\n" " | "} $msg]"
  }
}
catch {close_hw_target}
catch {disconnect_hw_server}
