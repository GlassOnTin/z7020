# flash.tcl — Program Zynq QSPI flash via JTAG
#
# Called by program_qspi.sh. Not intended to be run directly.
#
# Usage: vivado -mode batch -source flash.tcl -tclargs <BOOT.bin> <fsbl.elf>
#
# In Vivado 2025.2, Zynq QSPI flash programming uses u-boot-based embedded
# programmers from data/xicom/cfgmem/uboot/zynq_qspi_*.bin rather than
# PL helper bitfiles. The FSBL is specified via PROGRAM.ZYNQ_FSBL to
# initialize the PS before the embedded programmer runs.

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
refresh_hw_device $device

puts "=== Creating flash configuration ==="

# Zynq-7000 requires device-specific cfgmem parts (generic parts not supported).
# Try common 128Mbit QSPI flash chips in order of likelihood.
# Run vivado/list_flash_parts.tcl to see all parts valid for your device.
set flash_candidates {
    "w25q128fv-qspi-x1-single"
    "w25q128fw-qspi-x1-single"
    "is25lp128f-qspi-x1-single"
    "mt25ql128-qspi-x1-single"
    "s25fl127s-3.3v-qspi-x4-single"
    "n25q128-3.3v-qspi-x1-single"
}

set flash_part ""
foreach part $flash_candidates {
    if {![catch {
        create_hw_cfgmem -hw_device $device \
            [lindex [get_cfgmem_parts $part] 0]
    }]} {
        set flash_part $part
        break
    }
}

if {$flash_part eq ""} {
    puts "ERROR: No matching flash part found for this device."
    puts "Run: vivado -mode batch -source vivado/list_flash_parts.tcl"
    close_hw_manager
    exit 1
}

puts "Flash part: $flash_part"

# Configure programming options
set cfgmem [current_hw_cfgmem]
set_property PROGRAM.ADDRESS_RANGE {use_file} $cfgmem
set_property PROGRAM.FILES [list $boot_bin] $cfgmem
set_property PROGRAM.PRM_FILE {} $cfgmem
set_property PROGRAM.BLANK_CHECK 0 $cfgmem
set_property PROGRAM.ERASE 1 $cfgmem
set_property PROGRAM.CFG_PROGRAM 1 $cfgmem
set_property PROGRAM.VERIFY 1 $cfgmem

# Specify FSBL for indirect flash programming on Zynq
# (the FSBL initializes PS clocks and QSPI controller before the
# u-boot embedded programmer takes over to write the flash)
set_property PROGRAM.ZYNQ_FSBL $fsbl_elf $cfgmem

puts "=== Programming flash (erase + write + verify) ==="
puts "BOOT.bin: $boot_bin ([file size $boot_bin] bytes)"
puts "This takes approximately 60 seconds..."

if {[catch {program_hw_cfgmem $cfgmem} err]} {
    puts ""
    puts "ERROR: Flash programming failed: $err"
    puts ""
    puts "Common causes:"
    puts "  - Flash chip not 128Mbit: try a different cfgmem part"
    puts "  - FSBL PS7 config doesn't match board: regenerate with create_fsbl.tcl"
    puts "  - JTAG connection issue: check cable and power"
    close_hw_manager
    exit 1
}

puts ""
puts "=== Flash programming complete ==="

close_hw_manager
