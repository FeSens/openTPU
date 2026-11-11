# The DDR4 MIGs' generated RTL edits for IMAGE=wg (run_vivado.sh WG_MIGMAP and WG_MIGRD; build.tcl
# sources this after generate_target). The IP cache would hand back the unedited synthesis: either
# edit needs OTPU_IP_CACHE empty.
#
# WG_MIGMAP, the address map. bd_wg.tcl's ROW_COLUMN_BANK takes a DIMM byte address's bank group from
# bits 7:6, its bank from 9:8, its column from 16:10 and 5:3 and its row from 32:17: a load beside
# the weight stream, or a pair's other DIMM half, then lands in another row of the same bank every
# 1 KiB, and the MIG's bank groups, each in order, wait on the precharge (docs/bcu1525_wg.md, the
# MIG model). WG_MIGMAP=b15 takes the bank from bits 16:15 instead (the column from 14:8 and 5:3; the
# bank group and the row as before). The IP has no such map, so this edits each MIG's generated UI
# decode (rtl/ui/ddr4_v2_2_ui_cmd.sv, its ROW_COLUMN_BANK DDR4 lines) before the IPs' synthesis.
# The pair's DIMM address is {o[CL-1:12], s, o[11:0]} (otpu_mig_pair): at WG_CAW 22 (512 MiB a
# channel) the pairs use the DIMM's first 1 GiB, rows below 2^13; at WG_CAW 25 (4 GiB, the 32 GB
# card) all of the 8 GB, the row's bits 32:17 the channel's 31:16, under either map. A 16 GB DIMM
# (WG_CAW 26: a second rank or x4 parts, a 34-bit AXI address) is later work, its map with it.
#
# WG_MIGRD, the reads in flight. Under ORDERING Normal the user interface's read buffer (32 slots,
# one from a command's acceptance to its data's return) and the AXI front end's read channel (rtl/
# axi/ddr4_v2_2_axi_r_channel.sv: r_data_rdy holds the commands back once its transaction FIFO of
# 30 or its read data FIFO of 32 is nearly full) each hold a read from its command to its data: ~32
# reads over a round trip of ~33 cycles, a DIMM's stream ~85% of its 19.2 GB/s. WG_MIGRD=s62:
# ORDERING Strict (bd_wg.tcl; the CASes in the order taken, no read buffer) and the read channel's
# FIFOs at 62 and 64 (the data FIFO's depth under C_MC_BURST_LEN 2).
#   otpu_mig_map OUT patch    edit the generated copies (four expected; the ip_user_files' too)
#   otpu_mig_map OUT check    error unless every copy carries the edits (after synthesis)
set ::otpu_mig_maps {
  b15 {
    {assign col =   { app_addr_ns2[3+BANK_WIDTH+BANK_GROUP_WIDTH+:COL_WIDTH-3], app_addr_ns2[2:0] };}
    {assign col =   { app_addr_ns2[3+BANK_GROUP_WIDTH+:COL_WIDTH-3], app_addr_ns2[2:0] };}
    {assign bank =    app_addr_ns2[3+BANK_GROUP_WIDTH+:BANK_WIDTH];}
    {assign bank =    app_addr_ns2[COL_WIDTH+BANK_GROUP_WIDTH+:BANK_WIDTH];}
  }
}
set ::otpu_mig_rds {
  s62 {
    {localparam P_DEPTH = 30;} {localparam P_DEPTH = 62;}
    {localparam P_AWIDTH = 5;} {localparam P_AWIDTH = 6;}
    {localparam P_D_DEPTH  = (C_MC_BURST_LEN == 2)? 64 : 32;} {localparam P_D_DEPTH  = 64;}
    {localparam P_D_AWIDTH = (C_MC_BURST_LEN == 2)? 6: 5;} {localparam P_D_AWIDTH = 6;}
  }
}

proc otpu_count {s sub} {
  set n 0; set i 0
  while {[set j [string first $sub $s $i]] >= 0} { incr n; set i [expr {$j + [string length $sub]}] }
  return $n
}

# the edit OTPU_<var> names in the table edits (old / new pairs) on each MIG's rtl/<rel>, what it is
proc otpu_mig_edit {out mode var edits rel what} {
  set v [expr {[info exists ::env(OTPU_$var)] ? $::env(OTPU_$var) : ""}]
  if {$v eq ""} { return }
  if {![dict exists $edits $v]} { error "OTPU_$var $v: not one of [dict keys $edits]" }
  if {[info exists ::env(OTPU_IP_CACHE)] && $::env(OTPU_IP_CACHE) ne ""} {
    error "OTPU_$var $v needs the IP cache off (OTPU_IP_CACHE=)"
  }
  set gen [glob -nocomplain $out/otpu.gen/sources_1/bd/*/ip/*_ddr4_*/rtl/$rel]
  if {[llength $gen] != 4} { error "OTPU_$var: expected four MIG $what, not '$gen'" }
  set files [concat $gen [glob -nocomplain $out/otpu.ip_user_files/bd/*/ip/*_ddr4_*/rtl/$rel]]
  foreach f $files {
    set fh [open $f]; set src [read $fh]; close $fh
    foreach {old new} [dict get $edits $v] {
      set nold [otpu_count $src $old]
      set nnew [otpu_count $src $new]
      if {$mode eq "check"} {
        if {$nnew != 1 || $nold != 0} { error "OTPU_$var $v: $f lost the edit ($nold / $nnew)" }
      } elseif {$nold == 1 && $nnew == 0} {
        set src [string map [list $old $new] $src]
      } elseif {!($nold == 0 && $nnew == 1)} {
        error "OTPU_$var $v: $f has '$old' $nold times (the IP's RTL changed?)"
      }
    }
    if {$mode ne "check"} { set fh [open $f w]; puts -nonewline $fh $src; close $fh }
  }
  puts "OTPU_$var $v: [expr {$mode eq "check" ? "checked" : "patched"}] [llength $files] MIG $what"
}

proc otpu_mig_map {out mode} {
  otpu_mig_edit $out $mode WG_MIGMAP $::otpu_mig_maps ui/ddr4_v*_ui_cmd.sv "UI decodes"
  otpu_mig_edit $out $mode WG_MIGRD $::otpu_mig_rds axi/ddr4_v*_axi_r_channel.sv "AXI read channels"
}
