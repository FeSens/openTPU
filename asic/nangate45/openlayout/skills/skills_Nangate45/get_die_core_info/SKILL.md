## Get die and core dimensions

Before planning positions, you need the canvas size.

```tcl
set block [ord::get_db_block]
set dbu   [[ord::get_db_tech] getDbUnitsPerMicron]
set die   [$block getDieArea]
set core  [$block getCoreArea]

set die_w  [expr ([$die  xMax] - [$die  xMin]) / double($dbu)]
set die_h  [expr ([$die  yMax] - [$die  yMin]) / double($dbu)]
set core_w [expr ([$core xMax] - [$core xMin]) / double($dbu)]
set core_h [expr ([$core yMax] - [$core yMin]) / double($dbu)]

puts [format "DBU per micron: %d" $dbu]
puts [format "Die  : %.3f x %.3f um (area = %.3f um^2)" \
    $die_w $die_h [expr $die_w * $die_h]]
puts [format "Core : %.3f x %.3f um (area = %.3f um^2)" \
    $core_w $core_h [expr $core_w * $core_h]]
puts [format "Core origin: (%.3f, %.3f)" \
    [expr [$core xMin]/double($dbu)] [expr [$core yMin]/double($dbu)]]
```

All OpenROAD geometry is in DBU (database units) internally; divide by
`dbu` to get microns.