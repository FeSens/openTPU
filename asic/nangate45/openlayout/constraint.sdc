# openTPU ASIC block (nangate45-small) for the OpenLayout-style flow: tools/asic/flow.py's
# constraints (one clock; the memory ports and controls timed against a virtual clock with 20%
# of the period outside; no hold checks at the ports) with a 3.3 ns period. After CTS the flow
# gives vclk the tree's insertion delay as latency (scripts/pnr.tcl).
current_design otpu_asic_top
set clk_period 3.3
create_clock -name core_clock -period $clk_period [get_ports clk]
create_clock -name vclk -period $clk_period
set_input_delay [expr $clk_period * 0.2] -clock vclk [all_inputs -no_clocks]
set_output_delay [expr $clk_period * 0.2] -clock vclk [all_outputs]
set_false_path -hold -from [all_inputs -no_clocks]
set_false_path -hold -to [all_outputs]
