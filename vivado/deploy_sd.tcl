# deploy_sd.tcl — Write files to SD card via JTAG using sd_writer
#
# Halts the running ARM application, loads sd_writer.elf + file data
# into DDR, then runs sd_writer to write BOOT.bin (and optionally
# weights.bin) to the SD card's FAT32 partition.
#
# Prerequisites:
#   - Board powered on in SD boot mode (FSBL has initialized SD controller)
#   - JTAG connected via USB
#   - sw/sd_writer.elf built (cd sw && make sd_writer.elf)
#
# Usage:
#   source env.sh
#   xsdb vivado/deploy_sd.tcl <boot.bin> [weights.bin]
#
# Examples:
#   xsdb vivado/deploy_sd.tcl vivado/BOOT_badapple.bin scripts/weights.bin
#   xsdb vivado/deploy_sd.tcl vivado/BOOT.bin

set script_dir  [file dirname [file normalize [info script]]]
set project_dir [file dirname $script_dir]
set sd_writer   "$project_dir/sw/sd_writer.elf"

# --- Parse arguments ---
if {$argc < 1} {
    puts "Usage: xsdb deploy_sd.tcl <boot.bin> \[weights.bin\]"
    puts ""
    puts "  boot.bin     — BOOT.bin image (FSBL + bitstream + app)"
    puts "  weights.bin  — Optional data file written as weights.bin on SD"
    exit 1
}

set boot_bin [lindex $argv 0]
set weights_bin ""
if {$argc >= 2} {
    set weights_bin [lindex $argv 1]
}

# --- Verify files ---
foreach f [list $sd_writer $boot_bin] {
    if {![file exists $f]} {
        puts "ERROR: File not found: $f"
        exit 1
    }
}
if {$weights_bin ne "" && ![file exists $weights_bin]} {
    puts "ERROR: File not found: $weights_bin"
    exit 1
}

set boot_size [file size $boot_bin]
set weights_size 0
if {$weights_bin ne ""} {
    set weights_size [file size $weights_bin]
}

puts "=== SD Card Deployment via JTAG ==="
puts "sd_writer:   $sd_writer"
puts "BOOT.bin:    $boot_bin ($boot_size bytes)"
if {$weights_size > 0} {
    puts "weights.bin: $weights_bin ($weights_size bytes)"
}
puts ""

# --- Connect and halt ---
connect
after 500
targets -set -filter {name =~ "ARM*#0"}
puts "Connected to ARM core 0"

catch {stop}
after 200
puts "Processor halted"

# --- Load sd_writer ELF ---
puts "Loading sd_writer.elf..."
dow $sd_writer
after 500

# --- Load file data to DDR ---
# BOOT.bin at 0x02000000, weights.bin at 0x02800000
# (must match addresses in sw/sd_writer.c)
puts "Loading BOOT.bin data to DDR (0x02000000)..."
dow -data $boot_bin 0x02000000
after 500

if {$weights_size > 0} {
    puts "Loading weights.bin data to DDR (0x02800000)..."
    dow -data $weights_bin 0x02800000
    after 500
}

# --- Write sizes to control block ---
# (sd_writer reads these from fixed DDR addresses)
mwr 0x01FFFFF0 $boot_size
mwr 0x01FFFFF4 $weights_size

set rb_boot [mrd -value 0x01FFFFF0]
set rb_weights [mrd -value 0x01FFFFF4]
puts "Sizes: BOOT=$rb_boot weights=$rb_weights"

# --- Run sd_writer ---
puts ""
puts "Running sd_writer... (monitor UART 115200 for progress)"
con

# Wait for completion (~2s per MB at SD write speed, plus margin)
set total_mb [expr {($boot_size + $weights_size) / 1048576.0}]
set wait_ms  [expr {int($total_mb * 3000 + 10000)}]
puts "Waiting ${wait_ms}ms for ~${total_mb}MB write..."
after $wait_ms

catch {stop}
after 200
set pc [rrd pc]
puts ""
puts "sd_writer PC: $pc"
puts ""
puts "=== Done. Power cycle to boot with new firmware. ==="

disconnect
