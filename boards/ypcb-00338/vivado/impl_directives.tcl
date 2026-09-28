# Vivado run properties for this build, sourced by build.tcl after it opens the project (the fmax
# tournament's whole-design component, otpu_impl, edits this file). One line per property:
#   set_property <PROPERTY> <value> [get_runs synth_1|impl_1]
# e.g. STEPS.SYNTH_DESIGN.ARGS.RETIMING true, STEPS.PLACE_DESIGN.ARGS.DIRECTIVE ExtraTimingOpt,
# STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore. Nothing else: no Tcl, no hooks, no
# timing exceptions (tools/tourney/gates.py checks it). Empty: Vivado's defaults.
