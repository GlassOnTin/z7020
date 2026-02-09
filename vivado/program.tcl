# program.tcl — Program FPGA via JTAG
# Usage: vivado -mode batch -source program.tcl

set script_dir [file dirname [info script]]
set bit_file "$script_dir/mandelbrot/mandelbrot.runs/impl_1/mandelbrot_top.bit"

if {![file exists $bit_file]} {
    puts "ERROR: Bitstream not found: $bit_file"
    exit 1
}

puts "=== Opening hardware manager ==="
open_hw_manager
connect_hw_server -allow_non_jtag
open_hw_target

puts "=== Programming device ==="
set device [get_hw_devices xc7z020_1]
current_hw_device $device
set_property PROGRAM.FILE $bit_file $device
program_hw_devices $device

puts "=== PROGRAMMING COMPLETE ==="
close_hw_manager
