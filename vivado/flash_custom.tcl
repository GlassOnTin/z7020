# flash_custom.tcl — Custom QSPI flash programmer via xsdb
#
# Bypasses Vivado's program_hw_cfgmem and its problematic u-boot.
# Uses a custom flash_writer running on the ARM core instead.
#
# Flow:
#   1. Load FSBL → initialize DDR + QSPI
#   2. Load BOOT.bin into DDR via JTAG
#   3. Load flash_writer → erase + program + verify QSPI
#
# Usage: xsdb flash_custom.tcl <BOOT.bin> <fsbl.elf> <flash_writer.elf>

if {$argc < 3} {
    puts "Usage: xsdb flash_custom.tcl <BOOT.bin> <fsbl.elf> <flash_writer.elf>"
    exit 1
}

set boot_bin  [lindex $argv 0]
set fsbl_elf  [lindex $argv 1]
set flash_elf [lindex $argv 2]

set boot_size [file size $boot_bin]
puts "=== Custom QSPI Flash Programmer ==="
puts "  BOOT.bin:      $boot_bin ($boot_size bytes)"
puts "  FSBL:          $fsbl_elf"
puts "  Flash writer:  $flash_elf"

connect
targets -set -filter {name =~ "ARM*#0"}

# Step 1: Load and run FSBL
puts ""
puts "=== Step 1: Running FSBL (DDR + QSPI init) ==="
rst -system
after 1000
dow $fsbl_elf
con
after 5000
catch {stop}
after 500

# Step 2: Load BOOT.bin to DDR at 0x01000000
set ddr_addr 0x01000000
puts ""
puts "=== Step 2: Loading BOOT.bin to DDR at [format 0x%08X $ddr_addr] ==="
puts "  Size: $boot_size bytes — this may take a moment..."

# Read BOOT.bin as binary and write to DDR
set fd [open $boot_bin r]
fconfigure $fd -translation binary
set data [read $fd]
close $fd

# Write in chunks using dow -data (binary download)
# xsdb 'dow -data' writes binary file directly to memory
dow -data $boot_bin $ddr_addr
puts "  BOOT.bin loaded to DDR."

# Quick verify: read first 4 bytes
set first_word [mrd -force -value $ddr_addr]
puts "  First word at DDR: [format 0x%08X $first_word]"

# Step 3: Load flash writer to OCM
puts ""
puts "=== Step 3: Loading flash writer ==="
dow $flash_elf
after 100

# Read JEDEC ID before starting
puts "Starting flash writer..."
# Clear control block
mwr -force 0x00020000 0  ;# command = idle
mwr -force 0x0002000C 0  ;# status = idle
con
after 2000
catch {stop}
after 200

set jedec_id [mrd -force -value 0x00020018]
puts "  Flash JEDEC ID: [format 0x%06X $jedec_id]"
set mfr [expr {($jedec_id >> 16) & 0xFF}]
set typ [expr {($jedec_id >> 8) & 0xFF}]
set cap [expr {$jedec_id & 0xFF}]
puts "  Manufacturer: [format 0x%02X $mfr] Type: [format 0x%02X $typ] Capacity: [format 0x%02X $cap]"

if {$mfr == 0 || $mfr == 0xFF} {
    puts "ERROR: Could not read flash JEDEC ID. QSPI not working."
    disconnect
    exit 1
}

# Step 4: Erase + Program
puts ""
puts "=== Step 4: Erasing and programming flash ==="
puts "  Writing $boot_size bytes..."

# Set command: erase + write
mwr -force 0x00020004 $ddr_addr  ;# src_addr
mwr -force 0x00020008 $boot_size ;# length
mwr -force 0x0002000C 0          ;# status = busy
mwr -force 0x00020000 1          ;# command = erase+write

# Resume flash writer (do NOT stop during operation!)
con

# Poll progress without stopping the core — mrd -force works via AXI
set last_progress 0
set timeout 1200  ;# 20 minutes max
for {set i 0} {$i < $timeout} {incr i} {
    after 1000

    set status [mrd -force -value 0x0002000C]
    set progress [mrd -force -value 0x00020010]

    if {$status == 1} {
        puts "  Progress: $progress / $boot_size — DONE"
        break
    } elseif {$status == 0xFF} {
        set erraddr [mrd -force -value 0x00020014]
        puts "  ERROR at flash address [format 0x%08X $erraddr]"
        disconnect
        exit 1
    } else {
        if {$progress != $last_progress} {
            puts "  Progress: $progress / $boot_size bytes"
            set last_progress $progress
        }
    }
}

if {$i >= $timeout} {
    puts "ERROR: Flash programming timed out"
    disconnect
    exit 1
}

# Step 5: Verify
puts ""
puts "=== Step 5: Verifying flash ==="

# Stop core to write new command, then resume
catch {stop}
after 200

mwr -force 0x00020004 $ddr_addr  ;# src_addr
mwr -force 0x00020008 $boot_size ;# length
mwr -force 0x0002000C 0          ;# status = busy
mwr -force 0x00020000 2          ;# command = verify

con

# Poll verify progress without stopping the core
set last_progress 0
for {set i 0} {$i < $timeout} {incr i} {
    after 1000

    set status [mrd -force -value 0x0002000C]
    set progress [mrd -force -value 0x00020010]

    if {$status == 1} {
        puts "  Verify: $progress / $boot_size — PASS"
        break
    } elseif {$status == 0xFF} {
        set erraddr [mrd -force -value 0x00020014]
        set flash_word [mrd -force -value 0x0002001C]
        set ddr_word [mrd -force -value 0x00020020]
        set flash_byte [mrd -force -value 0x00020024]
        set ddr_byte [mrd -force -value 0x00020028]
        puts "  Verify FAILED at offset [format 0x%08X $erraddr]"
        puts "  Flash first 4 bytes: [format 0x%08X $flash_word]"
        puts "  DDR   first 4 bytes: [format 0x%08X $ddr_word]"
        puts "  Flash byte at error: [format 0x%02X [expr {$flash_byte & 0xFF}]]"
        puts "  DDR   byte at error: [format 0x%02X [expr {$ddr_byte & 0xFF}]]"
        disconnect
        exit 1
    } else {
        if {$progress != $last_progress} {
            puts "  Verify: $progress / $boot_size bytes"
            set last_progress $progress
        }
    }
}

puts ""
puts "=== Flash programming complete ==="
puts "Move BOOT jumper to QSPI and power cycle to boot from flash."

disconnect
