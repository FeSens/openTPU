# Floorplan of the accelerator (run_vivado.sh FLOORPLAN=1; create_project.tcl reads it as
# unmanaged Tcl, late, in implementation only). Soft Pblocks (IS_SOFT: the placer may leave them
# where legality or timing needs it), no EXCLUDE_PLACEMENT: other cells may share their regions.
#
# Measured on the routed 133.33 MHz build of 959b425 (no floorplan; fp_analyze.tcl in the
# floorplan experiment): the die is 2 x 8 clock regions, DDR3 channel 0 on the left edge in X0Y0-2,
# channel 1 in X0Y5-7, the PCIe GT and PCIE_2_1 on the right in X1Y4-5. The placer puts the MXU's
# systolic array (its 512 product DSPs, the activation delay SRLs) and the ACT RAM (68 BRAM, a
# 4k-bit read bus into the array's SRLs) in the top rows, X0Y5-7 / X1Y5-7, and TMEM's six read-
# port copies (384 BRAM) with their readers in rows 0-3. The quantizer, which reads its own two
# TMEM copies (read ports 2 and 6) and writes the ACT RAM, stays with its copies in X1Y2-3, and
# its ACT write crosses the die: quant > act 188 failing endpoints at 1.6 LUT levels, 95% route,
# 634 RPM units apart (u_quant qr / busy -> u_act wd, fan-out 182), act > act 19 (u_act wb ->
# the ACT BRAMs' write address, 0 levels, fan-out 65, 560 apart).
#
# pb_qact: the quantizer, the ACT RAM and the quantizer's TMEM copies in the top rows, by the
# array: the ACT write becomes local; the long hops left are TMEM's registered write broadcast
# into copies 2 and 6 and the quantizer's scalar read on the DMA's copy (read port 5).

# pats: cell names (a hierarchical instance takes its whole subtree), or "=~ <glob>" for the leaf
# cells of a generate scope, which synthesis flattens into its module ('?' for a bracket)
proc otpu_pblock {name ranges pats} {
  set cells {}
  foreach p $pats {
    if {[string match "=~ *" $p]} {
      lappend cells {*}[get_cells -quiet -hierarchical -filter "NAME [string range $p 0 1] [string range $p 3 end]"]
    } else {
      lappend cells {*}[get_cells -quiet $p]
    }
  }
  if {![llength $cells]} {
    puts "CRITICAL WARNING: \[otpu_floorplan.tcl\] $name: none of $pats found, no Pblock"
    return
  }
  set pb [create_pblock $name]
  resize_pblock $pb -add $ranges
  add_cells_to_pblock $pb $cells
  set_property IS_SOFT true $pb
  puts "otpu_floorplan.tcl: $name ($ranges): [llength $cells] cells"
}

set sl u_sys/u_board/u_slice
otpu_pblock pb_qact {CLOCKREGION_X0Y5:CLOCKREGION_X1Y6} [list $sl/u_quant $sl/u_act \
  "=~ $sl/u_tmem/g_port?2?.g_copy.*" "=~ $sl/u_tmem/g_port?6?.g_copy.*"]
