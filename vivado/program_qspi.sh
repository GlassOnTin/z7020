#!/bin/bash
# program_qspi.sh — Create BOOT.bin and program QSPI flash
#
# Creates a Zynq boot image from FSBL + bitstream, then writes it
# to the board's QSPI flash via JTAG. After programming, set the
# BOOT jumper to QSPI and the design loads automatically at power-on.
#
# Prerequisites:
#   - source env.sh (Vivado in PATH)
#   - FSBL generated (run: vivado -mode batch -source create_fsbl.tcl)
#   - Bitstream built (run: vivado -mode batch -source run_all.tcl)
#   - JTAG connected (run: ./fpga-jtag.sh attach)
#
# Usage: ./vivado/program_qspi.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BOOT_DIR="$SCRIPT_DIR/boot"
FSBL_ELF="$BOOT_DIR/fsbl/executable.elf"
BIT_FILE="$SCRIPT_DIR/mandelbrot/mandelbrot.runs/impl_1/mandelbrot_top.bit"
BIF_FILE="$SCRIPT_DIR/boot.bif"
BOOT_BIN="$SCRIPT_DIR/BOOT.bin"

# ---- Preflight checks ----
echo "=== QSPI Flash Programming ==="

if ! command -v bootgen &>/dev/null; then
    echo "ERROR: bootgen not found. Run: source env.sh"
    exit 1
fi

if [ ! -f "$FSBL_ELF" ]; then
    echo "ERROR: FSBL not found: $FSBL_ELF"
    echo "Generate it first:"
    echo "  vivado -mode batch -source vivado/create_fsbl.tcl"
    exit 1
fi

if [ ! -f "$BIT_FILE" ]; then
    echo "ERROR: Bitstream not found: $BIT_FILE"
    echo "Build it first:"
    echo "  vivado -mode batch -source vivado/run_all.tcl"
    exit 1
fi

# ---- Step 1: Create BOOT.bin ----
echo ""
echo "=== Creating BOOT.bin ==="
echo "  FSBL:      $FSBL_ELF"
echo "  Bitstream: $BIT_FILE"

cd "$SCRIPT_DIR"
bootgen -image boot.bif -arch zynq -o BOOT.bin -w

if [ ! -f "$BOOT_BIN" ]; then
    echo "ERROR: bootgen failed to create BOOT.bin"
    exit 1
fi

BOOT_SIZE=$(stat -c%s "$BOOT_BIN" 2>/dev/null || stat -f%z "$BOOT_BIN")
echo "  BOOT.bin:  $BOOT_BIN ($(( BOOT_SIZE / 1024 )) KB)"

# ---- Step 2: Program QSPI flash ----
echo ""
echo "=== Programming QSPI flash ==="
echo "This will erase and write the flash. Takes ~60 seconds."
echo ""

# Use Vivado hardware manager for flash programming
vivado -mode batch -nojournal -nolog -source "$SCRIPT_DIR/flash.tcl" \
    -tclargs "$BOOT_BIN" "$FSBL_ELF"

echo ""
echo "=== QSPI programming complete ==="
echo ""
echo "Set the BOOT jumper to QSPI and power-cycle the board."
echo "The Mandelbrot explorer will start automatically."
