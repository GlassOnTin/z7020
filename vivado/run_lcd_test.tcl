# run_lcd_test.tcl -- Full build flow for LCD diagnostic test
#
# Creates a Vivado project with spi_lcd_test as the top module,
# synthesizes, implements, and generates a bitstream.
#
# Usage: vivado -mode batch -source run_lcd_test.tcl

set script_dir [file dirname [info script]]
set rtl_dir    [file normalize "$script_dir/../rtl"]
set constr_dir [file normalize "$script_dir/../constraints"]

# FPGA part (Smart ZYNQ SP board)
set fpga_part "xc7z020clg484-1"

puts "=== Creating LCD test project ==="
create_project lcd_test "$script_dir/lcd_test" -part $fpga_part -force

# Add only the test module (glob picks up all .v, but top selection matters)
add_files -norecurse [glob $rtl_dir/*.v]
set_property file_type Verilog [get_files *.v]

# Use the LCD test constraints (spi_sda instead of spi_mosi, UART TX pin)
add_files -fileset constrs_1 -norecurse "$constr_dir/z7020_lcd_test.xdc"

# Set top module to the standalone LCD test
set_property top spi_lcd_test [current_fileset]
set_property target_language Verilog [current_project]

# Default strategies (no need for performance optimized)

puts "=== Running synthesis ==="
launch_runs synth_1 -jobs 8
wait_on_run synth_1
set synth_status [get_property STATUS [get_runs synth_1]]
puts "Synth status: $synth_status"

if {$synth_status ne "synth_design Complete!"} {
    puts "ERROR: Synthesis failed"
    exit 1
}

puts "=== Running implementation ==="
launch_runs impl_1 -jobs 8
wait_on_run impl_1
set impl_status [get_property STATUS [get_runs impl_1]]
puts "Impl status: $impl_status"

if {![string match "*route_design Complete*" $impl_status]} {
    puts "ERROR: Implementation failed"
    exit 1
}

puts "=== Generating bitstream ==="
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
puts "Bitstream: $script_dir/lcd_test/lcd_test.runs/impl_1/spi_lcd_test.bit"

# Print summary
open_run impl_1
puts "=== Post-Implementation Utilization ==="
report_utilization -return_string
puts "=== Timing Summary ==="
puts [report_timing_summary -return_string -max_paths 3]
puts "=== DONE ==="
