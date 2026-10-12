## Get the macros with their current location

Emits one line per macro, prefixed with an index `#N`. **That `#N` matches the
numbered label drawn on `<design>_pin_side.png`** by the `macro_pin_side` skill:
both iterate `[$block getInsts]` and count only block (macro) instances, starting
at 1, so on the **same ODB** the Nth macro here is the Nth label in the image.
This lets the Planner read a number off the pin-side image and look up that
macro's name + current location to refine.

> Alignment holds only when the image and this list come from the SAME ODB /
> DB state. If macros are re-placed and a new ODB is written, regenerate the
> pin-side image and this list together.

```tcl
set block [ord::get_db_block]
set dbu   [[ord::get_db_tech] getDbUnitsPerMicron]

set idx 0
foreach inst [$block getInsts] {
	set master [$inst getMaster]
	if {[$master isBlock]} {
		incr idx
		set w [expr [$master getWidth]  / double($dbu)]
		set h [expr [$master getHeight] / double($dbu)]
		# getBBox is orientation-correct (origin alone is the R0 lower-left only)
		set bb [$inst getBBox]
		set x  [expr [$bb xMin] / double($dbu)]
		set y  [expr [$bb yMin] / double($dbu)]
		puts [format "#%-3d %-40s  master=%-20s  size=%.3f x %.3f um  origin=(%.3f, %.3f)  orient=%s" \
			$idx [$inst getName] [$master getName] $w $h $x $y [$inst getOrient]]
	}
}
```

Output (the `#N` matches `pin_side.png` numbering):

```
#1   inst_58676   master=small_mem   size=75.560 x 149.125 um  origin=(20.160, 21.000)   orient=R0
#2   inst_58677   master=small_mem   size=75.560 x 149.125 um  origin=(113.960, 21.000)  orient=R0
...
```
