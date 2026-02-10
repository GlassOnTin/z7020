# create_fsbl.tcl — Generate FSBL for Zynq-7020 QSPI boot
#
# Creates a minimal PS7 hardware platform and generates the First Stage
# Boot Loader (FSBL) needed for QSPI boot images.
#
# This is a ONE-TIME setup step. The generated FSBL is reused for all
# subsequent bitstream updates.
#
# Prerequisites:
#   - Vivado 2025.2
#   - arm-none-eabi-gcc (sudo apt install gcc-arm-none-eabi)
#
# Usage: vivado -mode batch -source create_fsbl.tcl
#
# Output: vivado/boot/fsbl/executable.elf

set script_dir [file dirname [info script]]
set boot_dir "$script_dir/boot"
set fpga_part "xc7z020clg484-1"

file mkdir $boot_dir

# ============================================================
# Step 1: Create minimal PS7 block design
# ============================================================
puts "=== Creating PS7 hardware platform ==="

create_project ps7_boot "$boot_dir/ps7_project" -part $fpga_part -force

create_bd_design "system"

# Add Zynq PS7 IP
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7_0

# Configure PS7 for this board (Hello-FPGA Smart ZYNQ SP)
# From core board schematic:
#   PS clock: 50 MHz (Y2 oscillator — verify with schematic)
#   DDR3L: 2x NT5CC256M16EP-DI (4Gbit each, 32-bit bus, 1.35V)
#          Pin-compatible with Micron MT41K256M16 RE-125
#   MIO bank 0: 3.3V, MIO bank 1: 1.8V
#   QSPI: W25Q256 on MIO 1-6
set_property -dict [list \
    CONFIG.PCW_CRYSTAL_PERIPHERAL_FREQMHZ {50} \
    CONFIG.PCW_QSPI_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_QSPI_GRP_SINGLE_SS_ENABLE {1} \
    CONFIG.PCW_USE_M_AXI_GP0 {0} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {50} \
    CONFIG.PCW_UART0_PERIPHERAL_ENABLE {0} \
    CONFIG.PCW_ENET0_PERIPHERAL_ENABLE {0} \
    CONFIG.PCW_SD0_PERIPHERAL_ENABLE {0} \
    CONFIG.PCW_USB0_PERIPHERAL_ENABLE {0} \
    CONFIG.PCW_TTC0_PERIPHERAL_ENABLE {0} \
    CONFIG.PCW_GPIO_MIO_GPIO_ENABLE {0} \
    CONFIG.PCW_PRESET_BANK0_VOLTAGE {LVCMOS 3.3V} \
    CONFIG.PCW_PRESET_BANK1_VOLTAGE {LVCMOS 1.8V} \
    CONFIG.PCW_UIPARAM_DDR_MEMORY_TYPE {DDR 3 (Low Voltage)} \
    CONFIG.PCW_UIPARAM_DDR_PARTNO {MT41K256M16 RE-125} \
    CONFIG.PCW_UIPARAM_DDR_BUS_WIDTH {16 Bit} \
    CONFIG.PCW_UIPARAM_DDR_ENABLE {1} \
    CONFIG.PCW_UIPARAM_DDR_TRAIN_DATA_EYE {1} \
    CONFIG.PCW_UIPARAM_DDR_TRAIN_READ_GATE {1} \
    CONFIG.PCW_UIPARAM_DDR_TRAIN_WRITE_LEVEL {1} \
    CONFIG.PCW_UIPARAM_DDR_ECC {Disabled} \
    CONFIG.PCW_UIPARAM_DDR_DQS_TO_CLK_DELAY_0 {-0.073} \
    CONFIG.PCW_UIPARAM_DDR_DQS_TO_CLK_DELAY_1 {-0.034} \
    CONFIG.PCW_UIPARAM_DDR_DQS_TO_CLK_DELAY_2 {-0.03} \
    CONFIG.PCW_UIPARAM_DDR_DQS_TO_CLK_DELAY_3 {-0.082} \
    CONFIG.PCW_UIPARAM_DDR_BOARD_DELAY0 {0.176} \
    CONFIG.PCW_UIPARAM_DDR_BOARD_DELAY1 {0.159} \
    CONFIG.PCW_UIPARAM_DDR_BOARD_DELAY2 {0.162} \
    CONFIG.PCW_UIPARAM_DDR_BOARD_DELAY3 {0.187} \
] [get_bd_cells ps7_0]

# Apply board automation (connects DDR and fixed I/O)
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "0"} \
    [get_bd_cells ps7_0]

validate_bd_design
save_bd_design

# Generate output products
generate_target all [get_files system.bd]

# Create HDL wrapper
set wrapper [make_wrapper -files [get_files system.bd] -top]
add_files -norecurse $wrapper
set_property top system_wrapper [current_fileset]
update_compile_order -fileset sources_1

# ============================================================
# Step 2: Export hardware platform (XSA)
# ============================================================
puts "=== Exporting hardware platform ==="

write_hw_platform -fixed -force "$boot_dir/system.xsa"
puts "XSA exported: $boot_dir/system.xsa"

close_project

# ============================================================
# Step 3: Generate FSBL using xsdb + hsi
# ============================================================
puts "=== Generating FSBL ==="
puts "Running xsdb to build FSBL from XSA..."

set xsdb_script "$boot_dir/gen_fsbl.tcl"
set fd [open $xsdb_script w]
puts $fd "set hw \[hsi::open_hw_design \"$boot_dir/system.xsa\"\]"
puts $fd "hsi::generate_app -hw \$hw -os standalone -proc ps7_cortexa9_0 -app zynq_fsbl -compile -sw fsbl -dir \"$boot_dir/fsbl\""
puts $fd "hsi::close_hw_design \$hw"
puts $fd "exit"
close $fd

# Run xsdb (part of Vivado installation)
set xsdb_bin [file normalize "[file dirname [info nameofexecutable]]/../bin/xsdb"]
if {![file exists $xsdb_bin]} {
    # Try finding xsdb in PATH
    set xsdb_bin "xsdb"
}

puts "Using xsdb: $xsdb_bin"
if {[catch {exec $xsdb_bin $xsdb_script} result]} {
    puts "xsdb output: $result"

    # Check if FSBL was generated despite warnings
    if {[file exists "$boot_dir/fsbl/executable.elf"]} {
        puts "FSBL generated successfully (with warnings)."
    } else {
        puts ""
        puts "ERROR: FSBL generation failed."
        puts ""
        puts "If arm-none-eabi-gcc is not installed:"
        puts "  sudo apt install gcc-arm-none-eabi"
        puts ""
        puts "Then re-run this script."
        exit 1
    }
} else {
    puts $result
}

if {[file exists "$boot_dir/fsbl/executable.elf"]} {
    puts ""
    puts "=== FSBL generated successfully ==="
    puts "FSBL: $boot_dir/fsbl/executable.elf"
    puts ""
    puts "Next: run program_qspi.sh to create BOOT.bin and flash it."
} else {
    puts "ERROR: FSBL ELF not found at $boot_dir/fsbl/executable.elf"
    exit 1
}
