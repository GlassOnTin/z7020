# run_impl.tcl — Run implementation and generate bitstream
# Usage: vivado -mode batch -source run_impl.tcl

set script_dir [file dirname [info script]]
open_project "$script_dir/mandelbrot/mandelbrot.xpr"

# Check synth is done
set synth_status [get_property STATUS [get_runs synth_1]]
if {$synth_status ne "synth_design Complete!"} {
    puts "ERROR: Synthesis not complete (status: $synth_status). Run synthesis first."
    exit 1
}

# Reset and launch implementation
reset_run impl_1
launch_runs impl_1 -jobs 8
wait_on_run impl_1

set impl_status [get_property STATUS [get_runs impl_1]]
puts "Impl status: $impl_status"

if {[string match "*route_design Complete*" $impl_status]} {
    # Show post-implementation utilization and timing
    open_run impl_1
    report_utilization
    report_timing_summary -max_paths 5

    # Generate bitstream
    launch_runs impl_1 -to_step write_bitstream -jobs 8
    wait_on_run impl_1
    puts "Bitstream generation complete."
    puts "Bitstream: $script_dir/mandelbrot/mandelbrot.runs/impl_1/mandelbrot_top.bit"
} else {
    puts "ERROR: Implementation failed."
    set log_file "$script_dir/mandelbrot/mandelbrot.runs/impl_1/runme.log"
    if {[file exists $log_file]} {
        set fp [open $log_file r]
        set data [read $fp]
        close $fp
        set lines [split $data "\n"]
        set start [expr {max(0, [llength $lines] - 50)}]
        foreach line [lrange $lines $start end] {
            puts $line
        }
    }
}
