# create_project.tcl — Create Vivado project for Neural Mandelbrot Explorer
#
# Usage: vivado -mode batch -source create_project.tcl
# Or from Vivado Tcl console: source create_project.tcl

set project_name "mandelbrot"
set project_dir  "[file dirname [info script]]"
set rtl_dir      "[file normalize "$project_dir/../rtl"]"
set constr_dir   "[file normalize "$project_dir/../constraints"]"
set sim_dir      "[file normalize "$project_dir/../sim"]"

# FPGA part — XC7Z020CLG484 (carrier board schematic confirms CLG484 package)
set fpga_part "xc7z020clg484-1"

puts "Creating project: $project_name"
puts "  RTL dir:    $rtl_dir"
puts "  Constr dir: $constr_dir"
puts "  Part:       $fpga_part"

# Create project
create_project $project_name "$project_dir/$project_name" -part $fpga_part -force

# Add RTL sources
add_files -norecurse [glob $rtl_dir/*.v]
set_property file_type Verilog [get_files *.v]

# Add constraints
add_files -fileset constrs_1 -norecurse "$constr_dir/z7020_sp.xdc"

# Add simulation sources
add_files -fileset sim_1 -norecurse [glob $sim_dir/tb_*.v]

# Set top module
set_property top mandelbrot_top [current_fileset]

# Set simulation top
set_property top tb_neuron_core [get_filesets sim_1]

# Project settings
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]

# Synthesis settings — optimize for DSP inference
set_property strategy Flow_PerfOptimized_high [get_runs synth_1]

# Implementation settings
set_property strategy Performance_Explore [get_runs impl_1]

puts "Project created successfully."
puts "Run 'launch_runs synth_1' to synthesize."
