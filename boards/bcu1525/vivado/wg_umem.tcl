# umem, the BCU1525 WG floorplan move on an opt_design netlist at two slices: each slice's adapter
# (u_mem) and its 8 bridges' core halves (u_sys/g_ch[8 s ..].u_br/u_cs) from pblock_slice<s> into
# SLR1's pblock_ctl, beside the hub, and u_xs's stages split across the boundary of the slice's SLR
# with SLR1 (otpu_hbm_fp.tcl's): each core-clock pipe's pair on the boundary's
# Laguna registers (USER_SLL_REG; slice -> adapter, u_ctl and the queues' u_f: its last two stages;
# adapter -> slice, the queues' u_r and u_ra .. u_ri: its first two), its stages before the pair
# with its source and after it with its destination; the queues and the counts' stages (g_pc, to
# the adapter's synchronizer) with the adapter; the posts' stages (u_pp, g_ps) and the idle holds
# (w_h, r_h) with the slice, as built. NAT_XSTG >= 2 (x2's 2: no spare stage, the slice's port
# registers drive the pair; 3 gives the slice's side one).
# Sourced by place_screen.tcl (OTPU_SCREEN_MOVES umem) and bcu1525_wg_place_pre.tcl (OTPU_WG_UMEM 1,
# run_vivado.sh WG_UMEM: B2's place). ::otpu::wg_umem CTL WHO moves both slices and returns
# {MOVED SLL}: the moved cells (u_mem, the core halves) and the Laguna registers; any surprise in
# the netlist or the pblocks is an error naming WHO.
namespace eval ::otpu {}
proc ::otpu::umem_cells {who what pat} {
  set c [get_cells -quiet $pat]
  if {![llength $c]} { error "$who: umem: no $what ($pat)" }
  return $c
}
proc ::otpu::wg_umem {ctl who} {
  set b u_sys/u_board
  set moved {}
  set sllx {}
  foreach s {0 1} {
    set pb [get_pblocks -quiet pblock_slice$s]
    if {[llength $pb] != 1} { error "$who: umem: no pblock_slice$s" }
    # its SLR: the floorplan's resize_pblock -add SLR<k> (or clock regions: five rows an SLR)
    set gr [get_property GRID_RANGES $pb]
    set ks {}
    foreach {- n} [regexp -all -inline {SLR(\d)} $gr] { lappend ks $n }
    foreach {- y} [regexp -all -inline {CLOCKREGION_X\d+Y(\d+)} $gr] { lappend ks [expr {$y / 5}] }
    set ks [lsort -unique $ks]
    if {[llength $ks] != 1 || abs([lindex $ks 0] - 1) != 1} {
      error "$who: umem: pblock_slice$s's ranges '$gr' (SLR0 or SLR2)"
    }
    set k [lindex $ks 0]
    set pre [expr {$s == 0 ? "$b/" : "$b/g_slice\[$s\]."}]
    set c [::otpu::umem_cells $who "slice $s adapter" [string map {[ ? ] ?} ${pre}u_mem]]
    for {set j [expr {8 * $s}]} {$j < 8 * $s + 8} {incr j} {
      set c [concat $c [::otpu::umem_cells $who "bridge $j core half" "u_sys/g_ch?$j?.u_br/u_cs"]]
    }
    remove_cells_from_pblock $pb $c
    add_cells_to_pblock $ctl $c
    set moved [concat $moved $c]
    # u_xs by stage (otpu_hbm_fp.tcl's classification: a stage's register g_pipe.g_s[i].g_first /
    # g_next.r_reg; a pipe's other cells, its stage 0's input logic g_pipe.g_s[0].*, with its
    # source, the rest with its destination; cells under no pipe with the adapter)
    set xsn ${pre}u_xs
    set xsc [::otpu::umem_cells $who "slice $s u_xs" [string map {[ ? ] ?} $xsn]]
    set xc [get_cells -quiet -hierarchical -filter "IS_PRIMITIVE && NAME =~ [string map {[ ? ] ?} $xsn]/*"]
    set pfx "^[regsub -all {[^A-Za-z0-9_/]} $xsn {\\&}]/"
    set re $pfx
    append re {(.+)/g_pipe\.g_s\[(\d+)\]\.(?:g_first\.|g_next\.)?r_reg(\[\d+\])?$}
    array unset xst
    array unset xns
    array unset xsd
    array unset side
    foreach d {s a ts ta} { set side($d) {} }
    set oth {}
    foreach x $xc {
      if {[regexp $re $x - p i]} {
        lappend xst($p,$i) $x
        if {![info exists xns($p)] || $i >= $xns($p)} { set xns($p) [expr {$i + 1}] }
      } elseif {[regexp {/g_stg\.(w_h|r_h)_reg(\[\d+\])?$} $x]} {
        lappend side(s) $x
      } elseif {![regexp {/(GND|VCC)$} $x]} {
        lappend oth $x
      }
    }
    set np 0
    foreach p [lsort [array names xns]] {
      set ns $xns($p)
      if {[regexp {^g_pc\[\d+\]\.u_p$} $p]} {
        set w [lrepeat $ns a]; set xsd($p) {a a}
      } elseif {[regexp {^(u_pp|g_ps\.u_ps|g_ps\.g_psc\[\d+\]\.u_p)$} $p]} {
        set w [lrepeat $ns s]; set xsd($p) {s s}
      } else {
        if {[regexp {(^g_stg\.u_ctl|/u_f)$} $p]} {
          set src s; set dst a; set t [expr {$ns - 2}]
        } elseif {[regexp {(/u_r|^g_stg\.u_r[a-z])$} $p]} {
          set src a; set dst s; set t 0
        } else {
          error "$who: umem: slice $s's u_xs pipe $p of no known direction"
        }
        if {$ns < 2} { error "$who: umem: slice $s's u_xs pipe $p of $ns stage (NAT_XSTG >= 2)" }
        set w {}
        for {set i 0} {$i < $ns} {incr i} {
          lappend w [expr {$i < $t ? $src : ($i == $t ? "t$src" : ($i == $t + 1 ? "t$dst" : $dst))}]
        }
        set xsd($p) [list $src $dst]
        incr np
      }
      for {set i 0} {$i < $ns} {incr i} {
        if {[info exists xst($p,$i)]} { set side([lindex $w $i]) [concat $side([lindex $w $i]) $xst($p,$i)] }
      }
    }
    if {!$np} { error "$who: umem: slice $s's u_xs: no core-clock pipe (NAT_XSTG 0?)" }
    foreach x $oth {
      set d a
      if {[regexp "$pfx\(.+\)/(\[^/\]+)\$" $x - p l]} {
        while {![info exists xsd($p)] && [regexp {^(.+)/[^/]+$} $p - q]} { set p $q }
        if {[info exists xsd($p)]} {
          set d [lindex $xsd($p) [expr {[string match "g_pipe.g_s\\\[0\\\].*" $l] ? 0 : 1}]]
        }
      }
      lappend side($d) $x
    }
    remove_cells_from_pblock $pb $xsc
    foreach {d to} [list s $pb ts $pb a $ctl ta $ctl] {
      if {[llength $side($d)]} { add_cells_to_pblock $to [get_cells $side($d)] }
    }
    set sll [concat $side(ts) $side(ta)]
    set_property USER_SLL_REG TRUE [get_cells $sll]
    set sllx [concat $sllx $sll]
    puts "$who: umem: slice $s (SLR$k): [llength $c] cells (u_mem, 8 core halves) into pblock_ctl; u_xs: $np pipes across SLR$k / SLR1, [llength $sll] registers on the Laguna sites, [llength $side(s)] cells with the slice, [llength $side(a)] with the adapter"
  }
  return [list $moved $sllx]
}
