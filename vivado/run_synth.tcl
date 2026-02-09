# run_synth.tcl — Run synthesis and report utilization
# Usage: vivado -mode batch -source run_synth.tcl

set script_dir [file dirname [info script]]
open_project "$script_dir/mandelbrot/mandelbrot.xpr"

# Launch synthesis with parallel jobs
launch_runs synth_1 -jobs 8
wait_on_run synth_1

set status [get_property STATUS [get_runs synth_1]]
set progress [get_property PROGRESS [get_runs synth_1]]
puts "Synth status: $status"
puts "Synth progress: $progress"

# If synthesis succeeded, show utilization
if {$status eq "synth_design Complete!"} {
    open_run synth_1
    report_utilization
    report_timing_summary -max_paths 10
} else {
    puts "ERROR: Synthesis did not complete successfully."
    # Show any errors from the log
    set log_file "$script_dir/mandelbrot/mandelbrot.runs/synth_1/runme.log"
    if {[file exists $log_file]} {
        set fp [open $log_file r]
        set data [read $fp]
        close $fp
        # Print last 100 lines
        set lines [split $data "\n"]
        set start [expr {max(0, [llength $lines] - 100)}]
        foreach line [lrange $lines $start end] {
            puts $line
        }
    }
}
