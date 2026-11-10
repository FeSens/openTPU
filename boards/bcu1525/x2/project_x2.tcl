# x2 additions to create_project_wg.tcl, sourced before bd_wg.tcl. The machine target is S=4;
# make_wg.py has already generated the S=2 card's parameters and hash. No IP stubs in this list.
set x2dir $root/boards/bcu1525/x2
set fh [open $x2dir/sources_x2.f]; set x2src [split [string trim [read $fh]] \n]; close $fh
foreach f $x2src {
  add_files -norecurse $root/$f
  if {[file extension $f] eq ".sv"} { set_property file_type SystemVerilog [get_files $root/$f] }
}
add_files -fileset constrs_1 -norecurse $x2dir/pins_x2.xdc
add_files -fileset constrs_1 -norecurse $x2dir/clocks_x2.xdc
# bd_wg.tcl defines ip_vlnv; defer the IP generation until it has run.
