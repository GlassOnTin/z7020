# run_all.tcl — Full flow: create project, synthesize, implement, bitstream
# Usage: vivado -mode batch -source run_all.tcl

set script_dir [file dirname [info script]]
set rtl_dir    [file normalize "$script_dir/../rtl"]
set constr_dir [file normalize "$script_dir/../constraints"]
set sim_dir    [file normalize "$script_dir/../sim"]

# FPGA part
set fpga_part "xc7z020clg484-1"

puts "=== Creating project ==="
create_project mandelbrot "$script_dir/mandelbrot" -part $fpga_part -force
add_files -norecurse [glob $rtl_dir/*.v]
set_property file_type Verilog [get_files *.v]
add_files -fileset constrs_1 -norecurse "$constr_dir/z7020_sp.xdc"
set_property top mandelbrot_top [current_fileset]
set_property target_language Verilog [current_project]
set_property strategy Flow_PerfOptimized_high [get_runs synth_1]
set_property strategy Performance_Explore [get_runs impl_1]

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
puts "Bitstream: $script_dir/mandelbrot/mandelbrot.runs/impl_1/mandelbrot_top.bit"

# Print summary
open_run impl_1
puts "=== Post-Implementation Utilization ==="
report_utilization -return_string
puts "=== Timing Summary ==="
puts [report_timing_summary -return_string -max_paths 3]
puts "=== DONE ==="
