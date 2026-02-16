# deploy_badapple.tcl — Deploy Bad Apple files to SD card via JTAG + sd_writer
#
# Prerequisites:
#   - Board powered on, SD boot mode, FSBL has run (SD controller initialized)
#   - JTAG connected
#
# Usage:
#   source env.sh && xsdb vivado/deploy_badapple.tcl
#
# This script:
#   1. Halts the running application (mlp_loader)
#   2. Loads sd_writer.elf to DDR
#   3. Loads BOOT_badapple.bin and weights.bin data to DDR
#   4. Writes file sizes to control addresses
#   5. Runs sd_writer — which writes both files to the SD card

set script_dir [file dirname [file normalize [info script]]]
set project_dir [file dirname $script_dir]

set sd_writer   "$project_dir/sw/sd_writer.elf"
set boot_bin    "$project_dir/vivado/BOOT_badapple.bin"
set weights_bin "$project_dir/scripts/weights.bin"

# Verify files exist
foreach f [list $sd_writer $boot_bin $weights_bin] {
    if {![file exists $f]} {
        puts "ERROR: File not found: $f"
        exit 1
    }
}

set boot_size    [file size $boot_bin]
set weights_size [file size $weights_bin]

puts "=== Bad Apple SD Deployment ==="
puts "sd_writer:   $sd_writer"
puts "BOOT.bin:    $boot_bin ($boot_size bytes)"
puts "weights.bin: $weights_bin ($weights_size bytes)"
puts ""

# Connect to JTAG
connect
after 500

# Target ARM core 0
targets -set -filter {name =~ "ARM*#0"}
puts "Connected to ARM core 0"

# Stop the processor
catch {stop}
after 200
puts "Processor halted"

# Load sd_writer ELF (includes code + BSS initialization)
puts "Loading sd_writer.elf..."
dow $sd_writer
after 500

# Load BOOT_badapple.bin data to DDR at 0x02000000
puts "Loading BOOT_badapple.bin to DDR (0x02000000)..."
dow -data $boot_bin 0x02000000
after 500

# Load weights.bin data to DDR at 0x02800000
puts "Loading weights.bin to DDR (0x02800000)..."
dow -data $weights_bin 0x02800000
after 500

# Write file sizes to control addresses
puts "Setting file sizes..."
mwr 0x01FFFFF0 $boot_size
mwr 0x01FFFFF4 $weights_size

# Verify sizes were written
set rb_boot [mrd -value 0x01FFFFF0]
set rb_weights [mrd -value 0x01FFFFF4]
puts "  BOOT size:    $rb_boot (expected $boot_size)"
puts "  Weights size: $rb_weights (expected $weights_size)"

# Run sd_writer
puts ""
puts "Starting sd_writer..."
puts "(Monitor UART at 115200 for progress, or wait ~10 seconds)"
con

# Wait for it to finish (sd_writer spins in while(1) when done)
# Writing ~7.5MB to SD at ~4MB/s should take ~2 seconds
after 15000

# Check if still running (if PC is in the final while(1) loop, it's done)
catch {stop}
after 200
set pc [rrd pc]
puts ""
puts "sd_writer PC after 15s: $pc"
puts ""
puts "=== Deployment complete ==="
puts "Power cycle the board to boot with new BOOT.bin + weights.bin"
puts ""

disconnect
