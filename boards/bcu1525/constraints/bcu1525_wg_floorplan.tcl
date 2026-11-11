# The BCU1525 WG build's floorplan (create_project_wg.tcl; docs/bcu1525_wg.md):
# unmanaged Tcl, read in implementation only and late, before bcu1525_wg_top.tcl.
#
# The VU9P's SLR1 holds the PCIe block, DIMMs C1 and C2 and the clock's input (clk300_1); C0 is in
# SLR0, C3 in SLR2. The accelerator (u_sys/u_board: the slice, its adapter, the control block and
# the trace) and its bridges' core halves (u_sys/g_ch[j].u_br/u_cs, on the core clock beside the
# adapter) go to one SLR, make_wg.py --slr's (WG_SLR in OUT/wg_params.tcl, from which its RLY
# follows; the default SLR2): one WG slice out of context is 141K LUTs, 299 block RAMs, 156
# UltraRAMs and 1,438 DSPs (f2syn ooc/tgt, 2026-10-27), 36% / 41% / 49% / 63% of an SLR's. In SLR1
# beside XDMA (54K LUTs), C1's and C2's MIGs and the DMA's SmartConnects it filled 94.8% of the
# SLR's slices and failed on their 300 MHz paths (dev100, 2026-10-28). The bridges' AXI halves,
# register slices and pairs stay with the placer, beside their MIGs (across the crossings: RLY's
# pipelines). OTPU_WG_FP (environment): soft (the default: the pblock IS_SOFT, the placer may
# leave it where it must), hard, or none. wg_params.tcl: OTPU_WG_OUT's (run_vivado.sh: OUT), else
# the run directory's ../../ (OUT/otpu.runs/impl_1). No return here: in a read_xdc file it does not
# end the file (it went on to resize_pblock with no SLR, dev100b's synthesis reports, 2026-10-28).
#
# Two slices (bcu1525-s2, wg_params.tcl's WG_S 2 and WG_SLRS, make_wg.py --plan;
# docs/bcu1525_s2.md): pblock_slice<s> in slice s's SLR holds its slice, its adapter and staging
# (slice 0: u_board/u_slice, u_xs, u_mem; slice 1: u_board/g_slice[1].*), its side of the
# control crossings (g_xs.g_x[s]: u_dn_rx, u_cport, u_up_tx, slice 0's g_pf.u_pf_tx) and its 8
# bridges' core halves (u_sys/g_ch[8 s .. 8 s + 7]); pblock_ctl in SLR1 (beside XDMA) the control
# block, the trace, the collective unit and hub and their side of the crossings (u_dn_tx, u_up_rx,
# u_pf_rx): XRS 2's two stages a crossing, one each side of it (a slice in SLR1, floorplans a and
# b, takes them into its own pblock: no two pblocks on one SLR). SLR1's pblock also takes XDMA's
# SmartConnect (sc_dma: s2syn split it over SLR0 / SLR1, its M00 nodes on rows 298-299, -0.067 ns
# at 250 MHz); the bridges' pipeline stages go by SLR (::otpu::fp_ln, WG_DMAP and RLY), and the
# clock wizard's MMCM to OTPU_WG_MMCM's site (below).
#
# The MXU array's DSP columns, slice s's: OTPU_WG_MXU_COLS<s> (environment; run_vivado.sh
# WG_MXU_COLS<s>), DSP48E2 site columns ("3 4 5 ..." or "3,4,5"), puts the slice's MXU DSPs
# (u_slice/u_mxu's) in pblock_mxu<s>, those columns over the slice's SLR, nested in its pblock
# (soft or hard as it): the array kept away from the DIMM PHYs' I/O columns of its die (the
# Inspur's 150 MHz array beside its DDR3 PHYs closed their read eyes: docs/litedram.md section 10;
# YPCB's OTPU_MXU_DSP_COLS). Empty or unset (the default): the placer's, as before. A column
# with no DSP site in the SLR, fewer sites than DSPs, or no MXU DSP stops the run (::otpu::fail).
set fp_mode [expr {[info exists ::env(OTPU_WG_FP)] && $::env(OTPU_WG_FP) ne "" ? $::env(OTPU_WG_FP) : "soft"}]
namespace eval ::otpu {}
set fp_par [expr {[info exists ::env(OTPU_WG_OUT)] && $::env(OTPU_WG_OUT) ne ""
                  ? "$::env(OTPU_WG_OUT)/wg_params.tcl" : [file normalize ../../wg_params.tcl]}]
set fp_slr ""
set fp_ns 1
set fp_slrs {}
set fp_rly ""
set fp_dmap ""
if {![catch {open $fp_par} fh]} {
  set fp_txt [read $fh]
  close $fh
  regexp {set WG_SLR ([0-2])} $fp_txt -> fp_slr
  regexp {set WG_S ([12])} $fp_txt -> fp_ns
  regexp {set WG_SLRS \[list ([0-2 ]+)\]} $fp_txt -> fp_slrs
  regexp {set RLY ([0-9a-fA-F]+)} $fp_txt -> fp_rly
  regexp {set WG_DMAP ([0-9a-fA-F]{2})} $fp_txt -> fp_dmap
}
proc ::otpu::fp_miss {m} {
  puts "CRITICAL WARNING: \[bcu1525_wg_floorplan.tcl\] $m"
  lappend ::otpu::fail "bcu1525_wg_floorplan.tcl: $m"
}
# a slice's cells (brackets as ?: get_cells takes [n] for a bus index): {its slice, adapter and
# 8 bridges' core halves (10, each one there)} {its staging and its side of the crossings (as
# synthesis kept them)}
proc ::otpu::fp_slice {s} {
  set b u_sys/u_board
  set pre [expr {$s == 0 ? "$b/" : "$b/g_slice?$s?."}]
  set need [get_cells -quiet [list ${pre}u_slice ${pre}u_mem]]
  for {set j [expr {8 * $s}]} {$j < 8 * $s + 8} {incr j} {
    set need [concat $need [get_cells -quiet "u_sys/g_ch?$j?.u_br/u_cs"]]
  }
  set x "$b/g_xs.g_x?$s?."
  set opt [get_cells -quiet [list ${pre}u_xs ${x}u_dn_rx ${x}u_cport ${x}u_up_tx ${x}g_pf.u_pf_tx]]
  return [list $need $opt]
}
# slice s's MXU DSPs into pblock_mxu<s> (OTPU_WG_MXU_COLS<s>'s columns over SLR slr) under pblock
# parent; 1 when it made one or had none to make, 0 on a miss (recorded in ::otpu::fail)
proc ::otpu::fp_mxu {s slr parent mode} {
  set v OTPU_WG_MXU_COLS$s
  set cols [expr {[info exists ::env($v)] ? [regexp -all -inline {[0-9]+} $::env($v)] : {}}]
  if {![llength $cols]} { return 1 }
  set pre [expr {$s == 0 ? "u_sys/u_board/" : "u_sys/u_board/g_slice?$s?."}]
  set dsp [get_cells -quiet -hierarchical -filter "NAME =~ ${pre}u_slice/u_mxu/* && REF_NAME == DSP48E2"]
  set crs [get_clock_regions -quiet -of_objects [get_slrs -quiet SLR$slr]]
  set ranges {}
  set n 0
  set miss {}
  foreach c $cols {
    set ys {}
    foreach st [get_sites -quiet -of_objects $crs -filter "SITE_TYPE == DSP48E2 && NAME =~ DSP48E2_X${c}Y*"] {
      if {[regexp {Y([0-9]+)$} $st -> y]} { lappend ys $y }
    }
    if {![llength $ys]} { lappend miss $c; continue }
    set ys [lsort -integer $ys]
    lappend ranges DSP48E2_X${c}Y[lindex $ys 0]:DSP48E2_X${c}Y[lindex $ys end]
    incr n [llength $ys]
  }
  if {![llength $dsp] || [llength $miss] || $n < [llength $dsp]} {
    set m "$v '$::env($v)': [llength $dsp] MXU DSPs, columns without a DSP48E2 site in SLR$slr '$miss', $n sites"
    puts "CRITICAL WARNING: \[bcu1525_wg_floorplan.tcl\] $m"
    lappend ::otpu::fail "bcu1525_wg_floorplan.tcl: $m"
    return 0
  }
  set pb [create_pblock pblock_mxu$s]
  resize_pblock $pb -add $ranges
  set_property PARENT $parent $pb
  add_cells_to_pblock $pb $dsp
  if {$mode eq "soft"} { set_property IS_SOFT TRUE $pb }
  puts "bcu1525_wg_floorplan.tcl: pblock_mxu$s: [llength $dsp] MXU DSPs on DSP48E2 columns $cols of SLR$slr ($n sites)"
  return 1
}
# two slices: each board channel's bridge pipeline (otpu_hbm_ln u_ln, N = RLY[4 c +: 4]: 4 k + 1
# stages for a DIMM k SLRs from its slice, 2 for one beside it; DIMM slot c / 4 on DIMM
# WG_DMAP[2 (c / 4) +: 2]) by SLR, the WG targets' rule (otpu_hbm_fp.tcl): stages 0 .. pre (pre = N - 4 k)
# in its slice's SLR with its core half, stage j > pre in the SLR ceil((j - pre) / 4) on toward
# its DIMM's (pre / pre + 1 the pair either side of a crossing), stage N with the placer (beside
# the AXI half). s2syn left them to the placer: slice 0's crossing channels had stage 0 in SLR1,
# the core half's u_rd RAMs written from it across the SLR and 200 rows of SLR0 (-0.235 ns at 300
# MHz). slrpb: the SLRs' pblocks (an SLR without one gets pblock_slr<k>, as mode); 1 when every
# channel's stages were found, 0 on a miss (::otpu::fail)
proc ::otpu::fp_ln {slrs rly dmap mode slrpbv} {
  upvar 1 $slrpbv slrpb
  set nc [expr {8 * [llength $slrs]}]
  if {$dmap eq "" || [string length $rly] != $nc} {
    ::otpu::fp_miss "RLY '$rly', WG_DMAP '$dmap': $nc channels' pipelines"
    return 0
  }
  scan $dmap %x dm
  set nin 0
  set nx 0
  for {set c 0} {$c < $nc} {incr c} {
    set ss [lindex $slrs [expr {$c / 8}]]
    set ds [lindex {0 1 1 2} [expr {($dm >> (2 * ($c / 4))) & 3}]]
    scan [string index $rly end-$c] %x n
    set pre [expr {$n - 4 * abs($ds - $ss)}]
    if {$pre < 0} {
      ::otpu::fp_miss "channel $c: a pipeline of N $n, its DIMM [expr {abs($ds - $ss)}] SLRs from its slice"
      return 0
    }
    set u "u_sys/g_ch?$c?.u_br/u_ln"
    set nst 0
    for {set j 0} {$j < $n} {incr j} {
      set sj [get_cells -quiet [list "$u/fs_reg\[$j\]\[*\]" "$u/bs_reg\[$j\]\[*\]"]]
      if {![llength $sj]} { continue }
      set k [expr {$j <= $pre ? $ss : $ss + ($ds > $ss ? 1 : -1) * (($j - $pre + 3) / 4)}]
      if {![dict exists $slrpb $k]} {
        set pb [create_pblock pblock_slr$k]
        resize_pblock $pb -add SLR$k
        if {$mode eq "soft"} { set_property IS_SOFT TRUE $pb }
        dict set slrpb $k $pb
      }
      add_cells_to_pblock [dict get $slrpb $k] $sj
      incr nst [llength $sj]
      if {$k == $ss} { incr nin [llength $sj] } else { incr nx [llength $sj] }
    }
    if {!$nst} {
      ::otpu::fp_miss "channel $c: no pipeline stages ($u, N $n)"
      return 0
    }
  }
  puts "bcu1525_wg_floorplan.tcl: the bridges' pipelines: $nin stage registers in their slices' SLRs, $nx on toward their DIMMs' (stage N with the placer)"
  return 1
}
if {$fp_mode eq "none"} {
  puts "bcu1525_wg_floorplan.tcl: no floorplan (OTPU_WG_FP none)"
  incr ::otpu::ok(bcu1525_wg_floorplan)
} elseif {$fp_ns == 1} {
  set fp_cells [get_cells -quiet u_sys/u_board]
  set fp_cs [get_cells -quiet {u_sys/g_ch[*].u_br/u_cs}]
  if {[llength $fp_cells] != 1 || [llength $fp_cs] != 8} {
    puts "CRITICAL WARNING: \[bcu1525_wg_floorplan.tcl\] u_sys/u_board [llength $fp_cells], bridge core halves [llength $fp_cs] (expected 1, 8): no floorplan"
    lappend ::otpu::fail "bcu1525_wg_floorplan.tcl: u_sys/u_board [llength $fp_cells], bridge core halves [llength $fp_cs]"
  } elseif {$fp_slr eq ""} {
    puts "CRITICAL WARNING: \[bcu1525_wg_floorplan.tcl\] no WG_SLR in $fp_par: no floorplan"
    lappend ::otpu::fail "bcu1525_wg_floorplan.tcl: no WG_SLR in $fp_par"
  } else {
    set pb [create_pblock pblock_slice]
    resize_pblock $pb -add SLR$fp_slr
    add_cells_to_pblock $pb [concat $fp_cells $fp_cs]
    if {$fp_mode eq "soft"} { set_property IS_SOFT TRUE $pb }
    puts "bcu1525_wg_floorplan.tcl: pblock_slice in SLR$fp_slr ($fp_mode, $fp_par): u_sys/u_board and 8 bridge core halves"
    if {[::otpu::fp_mxu 0 $fp_slr $pb $fp_mode]} { incr ::otpu::ok(bcu1525_wg_floorplan) }
  }
} else {
  # two slices: each slice's 10 cells it must have, its staging and crossing stages; the control's
  set b u_sys/u_board
  set fp_ctl [get_cells -quiet [list $b/u_ctrl $b/u_iso $b/g_trace.u_trace $b/g_coll.u_coll \
                                     $b/g_coll_bar.u_coll $b/g_xs.u_chub $b/g_xs.g_x?*?.u_dn_tx \
                                     $b/g_xs.g_x?*?.u_up_rx $b/g_xs.g_x?0?.g_pf.u_pf_rx]]
  lassign [::otpu::fp_slice 0] fp_n0 fp_o0
  lassign [::otpu::fp_slice 1] fp_n1 fp_o1
  if {[llength $fp_slrs] != 2 || [llength $fp_n0] != 10 || [llength $fp_n1] != 10 || ![llength $fp_ctl]} {
    puts "CRITICAL WARNING: \[bcu1525_wg_floorplan.tcl\] SLRs '$fp_slrs', slice cells [llength $fp_n0] / [llength $fp_n1] (10 / 10), control [llength $fp_ctl]: no floorplan"
    lappend ::otpu::fail "bcu1525_wg_floorplan.tcl: SLRs '$fp_slrs', slice cells [llength $fp_n0] / [llength $fp_n1], control [llength $fp_ctl]"
  } else {
    # a slice in SLR1 (floorplans a and b: slice 0) takes the control cells into its own pblock
    set fp_c1 [lsearch $fp_slrs 1]
    set fp_mx 1
    set fp_slrpb [dict create]
    foreach s {0 1} cells [list [concat $fp_n0 $fp_o0] [concat $fp_n1 $fp_o1]] {
      if {$s == $fp_c1} { set cells [concat $cells $fp_ctl] }
      set pb [create_pblock pblock_slice$s]
      resize_pblock $pb -add SLR[lindex $fp_slrs $s]
      add_cells_to_pblock $pb $cells
      if {$fp_mode eq "soft"} { set_property IS_SOFT TRUE $pb }
      dict set fp_slrpb [lindex $fp_slrs $s] $pb
      puts "bcu1525_wg_floorplan.tcl: pblock_slice$s: [llength $cells] cells"
      set fp_mx [expr {[::otpu::fp_mxu $s [lindex $fp_slrs $s] $pb $fp_mode] && $fp_mx}]
    }
    if {$fp_c1 < 0} {
      set pb [create_pblock pblock_ctl]
      resize_pblock $pb -add SLR1
      add_cells_to_pblock $pb $fp_ctl
      if {$fp_mode eq "soft"} { set_property IS_SOFT TRUE $pb }
      dict set fp_slrpb 1 $pb
    }
    puts "bcu1525_wg_floorplan.tcl: pblock_slice0 in SLR[lindex $fp_slrs 0], pblock_slice1 in SLR[lindex $fp_slrs 1], the control's [llength $fp_ctl] cells in SLR1 ([expr {$fp_c1 < 0 ? "pblock_ctl" : "pblock_slice$fp_c1"}]; $fp_mode, $fp_par)"
    # XDMA's SmartConnect in SLR1, its pblock's
    set fp_dma [get_cells -quiet u_bd/otpu_bdw_i/sc_dma]
    if {[llength $fp_dma] != 1} {
      ::otpu::fp_miss "u_bd/otpu_bdw_i/sc_dma: '$fp_dma' (expected one)"
      set fp_mx 0
    } else {
      add_cells_to_pblock [dict get $fp_slrpb 1] $fp_dma
      puts "bcu1525_wg_floorplan.tcl: sc_dma in SLR1 ([dict get $fp_slrpb 1])"
    }
    # the accelerator's reset into acc_clk (otpu_fpga_top_bcu_wg ACC_RA, 2 at S 2): its core_clk
    # stages and acc_clk's first (the one hop) in SLR1, beside core_clk's root; acc_rq's last
    # stage, in copies, with the placer by its loads (s2c150: acc_rst_p[2] straight into bridge
    # core halves in SLR0 / SLR2, -0.379 ns of hold over 0.72-1.65 ns of clock skew)
    set fp_rq [get_cells -quiet {g_arq.acc_rq_reg[0]}]
    set fp_rp [get_cells -quiet {acc_rst_p_reg[*]}]
    if {[llength $fp_rq] != 1 || ![llength $fp_rp]} {
      ::otpu::fp_miss "the reset's hop into acc_clk: '$fp_rq', [llength $fp_rp] acc_rst_p registers"
      set fp_mx 0
    } else {
      add_cells_to_pblock [dict get $fp_slrpb 1] [concat $fp_rp $fp_rq]
      puts "bcu1525_wg_floorplan.tcl: the reset's hop in SLR1 ([dict get $fp_slrpb 1]): [llength $fp_rp] acc_rst_p registers and [join $fp_rq]"
    }
    set fp_mx [expr {[::otpu::fp_ln $fp_slrs $fp_rly $fp_dmap $fp_mode fp_slrpb] && $fp_mx}]
    if {$fp_mx} { incr ::otpu::ok(bcu1525_wg_floorplan) }
  }
}
# two slices: the clock wizard's MMCM (core_clk, acc_clk, acc_clk2x, aclk) at OTPU_WG_MMCM's site
# (environment; run_vivado.sh WG_MMCM), default MMCM_X1Y5, clk300_1's pad's clock region (its
# BUFG BUFGCE_X1Y138 there), between the slices' SLRs; "none": the placer's. OTPU_WG_FP none does
# not turn it off. s2syn's placer took MMCM_X1Y0: its input 2.39 ns over the BUFG's backbone from
# row 5 into SLR0, core_clk's net from its BUFG there through its root (X3Y6) to acc_ce in SLR1
# 3.23 ns, so acc_ce into the acc_clk2x enable's register (ce2x, its root X4Y1) had -1.867 ns of
# skew: -0.027 ns at 100 MHz against 4.998 (3.332 at 150 MHz).
if {$fp_ns == 2} {
  set fp_mm [expr {[info exists ::env(OTPU_WG_MMCM)] && $::env(OTPU_WG_MMCM) ne "" ? $::env(OTPU_WG_MMCM) : "MMCM_X1Y5"}]
  if {$fp_mm ne "none"} {
    set mm [get_cells -quiet -hierarchical -filter {NAME =~ u_bd/otpu_bdw_i/clk_wiz_0/* && REF_NAME == MMCME4_ADV}]
    if {[llength $mm] != 1} {
      ::otpu::fp_miss "the clock wizard's MMCM: '$mm' (expected one)"
    } elseif {[catch {set_property LOC $fp_mm $mm} e]} {
      ::otpu::fp_miss "LOC $fp_mm on $mm: $e"
    } else {
      puts "bcu1525_wg_floorplan.tcl: $mm at $fp_mm (OTPU_WG_MMCM)"
    }
  }
}
