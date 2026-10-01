# Arm the debug build's two ILAs (otpu_xmon.sv, run_vivado.sh XMON=1) over JTAG and save what
# they capture: Vivado's hardware manager, on the card's host.
#   vivado -mode batch -nojournal -source tools/xmon_ila.tcl -tclargs LTX OUT_DIR [MINUTES]
# Each ILA triggers on its first event (otpu_ila_x: ix_ev[7:0], the FLAGS events on XDMA's master;
# otpu_ila_n: in_ev[7:0], channel 0's ports) and records only the cycles that moved or had an
# event (ev != 0), the trigger at sample 4000 of 4096: about the last 4000 such cycles before it.
# Waits up to MINUTES (60) for either; writes OUT_DIR/ila_x.{csv,ila} / ila_n.* for each that
# triggered. connect_hw_server starts a hw_server that outlives Vivado and keeps the JTAG cable:
# kill it afterwards (by its PID) before openFPGALoader loads anything.
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
  if {[string match *u_ila_x* $cell]} { set key x; set pat ix_ev } elseif {[string match *u_ila_n* $cell]} { set key n; set pat in_ev } else { continue }
  set ev [get_hw_probes -quiet -of_objects $ila -filter "NAME =~ *$pat*"]
  if {[llength $ev] != 1} { error "ILA $cell: no single probe *$pat* (got '$ev')" }
  set_property CONTROL.TRIGGER_MODE BASIC_ONLY $ila
  set_property CONTROL.CAPTURE_MODE BASIC $ila
  set_property CONTROL.DATA_DEPTH 4096 $ila
  set_property CONTROL.TRIGGER_POSITION 4000 $ila
  set_property TRIGGER_COMPARE_VALUE {neq16'bXXXXXXXX00000000} $ev
  set_property CAPTURE_COMPARE_VALUE {neq16'h0000} $ev
  lappend ilas $key $ila
}
if {[llength $ilas] != 4} { error "expected otpu_ila_x and otpu_ila_n, found: $ilas" }
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
# the other ILA a few seconds more (its own event may follow)
after 5000
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
