# Floorplan of the accelerator (run_vivado.sh FLOORPLAN=1; create_project.tcl reads it as
# unmanaged Tcl, late, in implementation only). Soft Pblocks (IS_SOFT: the placer may leave them
# where legality or timing needs it), no EXCLUDE_PLACEMENT: other cells may share their regions.
#
# Measured on the routed 133.33 MHz build of 959b425 (no floorplan; fp_analyze.tcl in the
# floorplan experiment): the die is 2 x 8 clock regions, DDR3 channel 0 on the left edge in X0Y0-2,
# channel 1 in X0Y5-7, the PCIe GT and PCIE_2_1 on the right in X1Y4-5. The placer puts the MXU's
# systolic array (its 512 product DSPs, the activation delay SRLs) and the ACT RAM (68 BRAM, a
# 4k-bit read bus into the array's SRLs) in the top rows, X0Y5-7 / X1Y5-7, and TMEM's six read-
# port copies (384 BRAM) with their readers in rows 0-3. The largest failing family, dma > nmem
# (1381 endpoints, 11 LUT levels, 90% route), runs between the DMA (X0Y2-3, X1Y3-4) and the
# native DRAM adapter spread over X0Y3-5 (170 RPM units apart on average; on 5e5a the worst
# path's longest hop, 2.3 ns, reached the adapter's channel 1 queue RAMs 2.5 clock regions away).
#
# pb_dma: the DMA, the native DRAM adapter and the DMA's TMEM copy (read port 1, which the
# quantizer's scale read and the collective share) in the four middle regions X0Y3-4 / X1Y3-4,
# between the two channels' mem_ch / LiteDRAM sides on the left edge.
#
# pb_qact (not applied): the quantizer, the ACT RAM and the quantizer's TMEM copies (read ports 2
# and 6) in X0Y5-6 / X1Y5-6 beside the array, for quant > act (188 endpoints, 1.6 levels, 634
# units apart: the ACT write crossing the die) and act > act (19, its write address broadcast);
# the ACT RAM's input register (act-wreg) fixes those in the RTL instead.

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
# the native DRAM adapter: u_mem (ld-default), g_native.u_nmem before it
otpu_pblock pb_dma {CLOCKREGION_X0Y3:CLOCKREGION_X1Y4} [list $sl/u_dma u_sys/u_board/u_mem \
  u_sys/u_board/g_native.u_nmem "=~ $sl/u_tmem/g_port?1?.g_copy.*"]
if {0} {
  otpu_pblock pb_qact {CLOCKREGION_X0Y5:CLOCKREGION_X1Y6} [list $sl/u_quant $sl/u_act \
    "=~ $sl/u_tmem/g_port?2?.g_copy.*" "=~ $sl/u_tmem/g_port?6?.g_copy.*"]
}
