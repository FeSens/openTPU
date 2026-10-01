# Arm the debug build's ILA (otpu_xmon.sv, run_vivado.sh XMON=1: otpu_ila_x on XDMA's master)
# over JTAG and save what it captures: Vivado's hardware manager, on the card's host.
#   vivado -mode batch -nojournal -source tools/xmon_ila.tcl -tclargs LTX OUT_DIR [MINUTES]
# It triggers on the first event (ix_ev[7:0]: the FLAGS events on XDMA's master, or ix_ev[10]:
# a channel-0 flag set) and records only the cycles that moved or had an event (ix_ev != 0), the
# trigger at sample 4000 of 4096: about the last 4000 such cycles before it. Waits up to MINUTES
# (60); writes OUT_DIR/ila_x.{csv,ila} if it triggered. connect_hw_server starts a hw_server that
# outlives Vivado and keeps the JTAG cable: kill it afterwards (by its PID) before openFPGALoader
# loads anything.
set ltx [file normalize [lindex $argv 0]]
set out [file normalize [lindex $argv 1]]
set mins [expr {[llength $argv] > 2 ? [lindex $argv 2] : 60}]
file mkdir $out

open_hw_manager
connect_hw_server
open_hw_target
set dev [lindex [get_hw_devices -quiet xc7k480t*] 0]
if {$dev eq ""} { error "no xc7k480t on the JTAG chain: [get_hw_devices]" }
current_hw_device $dev
set_property PROBES.FILE $ltx $dev
set_property FULL_PROBES.FILE $ltx $dev
refresh_hw_device $dev

set ilas {}
foreach ila [get_hw_ilas -of_objects $dev] {
  set cell [get_property CELL_NAME $ila]
  puts "ILA $ila: $cell"
  foreach p [get_hw_probes -of_objects $ila] { puts "  probe [get_property NAME $p]" }
  if {[string match *u_ila_x* $cell]} { set key x; set pat ix_ev } else { continue }
  # the events probe is the ILA's probe0, ix_ev (synthesis may rename its net: by port, then by
  # name, then by width, the only 16-bit probe)
  set ev {}
  foreach f [list {PROBE_PORT == probe0} "NAME =~ *$pat*" {WIDTH == 16}] {
    if {[catch {get_hw_probes -quiet -of_objects $ila -filter $f} ev]} { set ev {} }
    if {[llength $ev] == 1} { puts "events probe: $ev ($f)"; break }
  }
  if {[llength $ev] != 1} { error "ILA $cell: no events probe (probe0, *$pat*, 16 bits)" }
  report_property $ev
  # (TRIGGER_MODE is read-only BASIC_ONLY on an ILA without advanced triggers)
  foreach {p v} {CONTROL.CAPTURE_MODE BASIC CONTROL.DATA_DEPTH 4096 CONTROL.TRIGGER_POSITION 4000} {
    if {[catch {set_property $p $v $ila} e]} { puts "ILA $cell: $p not set: $e" }
  }
  puts "ILA $cell: [get_property CONTROL.TRIGGER_MODE $ila] / [get_property CONTROL.CAPTURE_MODE $ila], position [get_property CONTROL.TRIGGER_POSITION $ila]"
  set_property TRIGGER_COMPARE_VALUE {neq16'bXXXXX0XX00000000} $ev
  if {[catch {set_property CAPTURE_COMPARE_VALUE {neq16'h0000} $ev} e]} { puts "ILA $cell: no capture qualification: $e" }
  lappend ilas $key $ila
}
if {[llength $ilas] != 2} { error "expected otpu_ila_x, found: $ilas" }
foreach {key ila} $ilas { run_hw_ila $ila }
puts "armed [clock format [clock seconds] -format %H:%M:%S]"

set t0 [clock seconds]
set done {}
while {[clock seconds] - $t0 < 60 * $mins && [llength $done] == 0} {
  foreach {key ila} $ilas {
    if {[get_property STATUS.CORE_STATUS $ila] eq "FULL"} { lappend done $key $ila }
  }
  after 500
}
foreach {key ila} $ilas {
  set st [get_property STATUS.CORE_STATUS $ila]
  puts "ILA $key: $st at [clock format [clock seconds] -format %H:%M:%S]"
  if {$st eq "FULL"} {
    set d [upload_hw_ila_data $ila]
    write_hw_ila_data -force -csv_file $out/ila_$key.csv $d
    write_hw_ila_data -force $out/ila_$key.ila $d
    puts "wrote $out/ila_$key.csv"
  }
}
close_hw_target
disconnect_hw_server
