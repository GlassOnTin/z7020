# deploy_badapple.tcl — Deploy Bad Apple firmware to SD card
#
# Usage:
#   source env.sh && xsdb vivado/deploy_badapple.tcl

set script_dir [file dirname [file normalize [info script]]]
set argv [list "$script_dir/BOOT_badapple.bin" "$script_dir/../scripts/weights.bin"]
set argc 2
source "$script_dir/deploy_sd.tcl"
