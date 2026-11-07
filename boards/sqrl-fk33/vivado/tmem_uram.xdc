# TMEM in UltraRAM (create_project.tcl with TMEM_URAM=1, run_vivado.sh's default): every TMEM copy's banks
# (otpu_tmem g_port[p].g_copy.g_bank[b].mem, 8K x 32 each) as URAM288 instead of block RAM. The RTL is unchanged
# (a synthesis property only), so the simulations' bit-exactness holds as it is; the read stays one registered
# cycle (the bank's q register, URAM's own output register off). The BCU1525's tmem_uram.xdc is the model.
# One command, no `if`: Vivado does not take `if` in an .xdc (Designutils 20-1307) and skips the whole block, which
# left the TMEM in block RAM (Vivado 2019.2). No cell matched gives a warning.
set_property RAM_STYLE ULTRA [get_cells -hierarchical -filter {NAME =~ *u_tmem/g_port*.g_bank*.mem_reg*}]
