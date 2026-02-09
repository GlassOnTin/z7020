# flash.tcl — Program Zynq QSPI flash via JTAG
#
# Called by program_qspi.sh. Not intended to be run directly.
#
# Usage: vivado -mode batch -source flash.tcl -tclargs <BOOT.bin> <fsbl.elf>

if {$argc < 2} {
    puts "Usage: vivado -mode batch -source flash.tcl -tclargs <BOOT.bin> <fsbl.elf>"
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

puts "=== Opening hardware manager ==="
open_hw_manager
connect_hw_server -allow_non_jtag
open_hw_target

set device [get_hw_devices xc7z020_1]
current_hw_device $device

puts "=== Creating flash configuration ==="

# Create configuration memory device
# Common QSPI flash parts on Zynq boards:
#   s25fl128sxxxxxx0-spi-x1_x2_x4  (Spansion/Cypress 128Mbit)
#   n25q128-3.3v-spi-x1_x2_x4     (Micron 128Mbit)
#   is25lp128-spi-x1_x2_x4        (ISSI 128Mbit)
#   w25q128fvsig-spi-x1_x2_x4     (Winbond 128Mbit)
#
# If the default doesn't work, check the flash chip marking on your board
# and update the part name below.
set flash_part "s25fl128sxxxxxx0-spi-x1_x2_x4"

# Try to create the cfgmem device
if {[catch {
    create_hw_cfgmem -hw_device $device \
        [lindex [get_cfgmem_parts $flash_part] 0]
} err]} {
    puts "WARNING: Flash part $flash_part not found, trying alternatives..."

    # Try common alternatives
    foreach part {
        "n25q128-3.3v-spi-x1_x2_x4"
        "is25lp128-spi-x1_x2_x4"
        "w25q128fvsig-spi-x1_x2_x4"
        "mt25ql128-spi-x1_x2_x4"
    } {
        if {![catch {
            create_hw_cfgmem -hw_device $device \
                [lindex [get_cfgmem_parts $part] 0]
        }]} {
            set flash_part $part
            puts "Using flash part: $flash_part"
            break
        }
    }
}

puts "Flash part: $flash_part"

# Configure programming options
set cfgmem [current_hw_cfgmem]
set_property PROGRAM.ADDRESS_RANGE {use_file} $cfgmem
set_property PROGRAM.FILES [list $boot_bin] $cfgmem
set_property PROGRAM.PRM_FILE {} $cfgmem
set_property PROGRAM.UNUSED_PIN_TERMINATION {pull-none} $cfgmem
set_property PROGRAM.BLANK_CHECK 0 $cfgmem
set_property PROGRAM.ERASE 1 $cfgmem
set_property PROGRAM.CFG_PROGRAM 1 $cfgmem
set_property PROGRAM.VERIFY 1 $cfgmem

# Specify FSBL for indirect flash programming on Zynq
# (the FSBL provides access to QSPI through the PS MIO pins)
set_property PROGRAM.ZYNQ_FSBL $fsbl_elf $cfgmem

puts "=== Programming flash (erase + write + verify) ==="
puts "This takes approximately 60 seconds..."

program_hw_cfgmem $cfgmem

puts "=== Flash programming complete ==="

close_hw_manager
