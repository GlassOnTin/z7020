# flash_xsdb.tcl — Program Zynq QSPI flash via xsdb program_flash
#
# Alternative to flash.tcl (Vivado program_hw_cfgmem). Uses xsdb's
# program_flash command which handles FSBL loading, DDR init, and flash
# programming in a single step.
#
# Usage: xsdb flash_xsdb.tcl <BOOT.bin> <fsbl.elf>

if {$argc < 2} {
    puts "Usage: xsdb flash_xsdb.tcl <BOOT.bin> <fsbl.elf>"
    exit 1
}

set boot_bin [lindex $argv 0]
set fsbl_elf [lindex $argv 1]

if {![file exists $boot_bin]} {
    puts "ERROR: BOOT.bin not found: $boot_bin"
    exit 1
}

if {![file exists $fsbl_elf]} {
    puts "ERROR: FSBL not found: $fsbl_elf"
    exit 1
}

puts "=== XSDB Flash Programming ==="
puts "BOOT.bin: $boot_bin ([file size $boot_bin] bytes)"
puts "FSBL:     $fsbl_elf"

# Connect to hw_server
connect

# Show available targets
puts ""
puts "Available targets:"
puts [targets]

# Target the Zynq ARM core
targets -set -filter {name =~ "ARM*#0"}

# Reset the system
rst -system
after 1000

puts ""
puts "=== Programming QSPI flash ==="
puts "This takes approximately 60-120 seconds..."

# program_flash handles: load FSBL, init PS (DDR+QSPI), write flash
program_flash -f $boot_bin -offset 0 \
    -flash_type qspi-x1-single \
    -fsbl $fsbl_elf

puts ""
puts "=== Flash programming complete ==="

disconnect
