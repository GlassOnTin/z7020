#!/bin/bash
# qspi_program.sh — End-to-end QSPI flash programming for Mandelbrot Explorer
#
# Compiles flash_writer.c, regenerates BOOT.bin, and programs QSPI flash.
# Run from the vivado/ directory on the build server (msi-z790).
#
# Usage: ./qspi_program.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# --- Paths ---
FLASH_WRITER_C="flash_writer.c"
FLASH_WRITER_ELF="flash_writer.elf"
BOOT_BIF="boot.bif"
BOOT_BIN="BOOT.bin"
FSBL_ELF="boot/fsbl/executable.elf"
BITSTREAM="mandelbrot/mandelbrot.runs/impl_1/mandelbrot_top.bit"
FLASH_TCL="flash_custom.tcl"

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

echo "=========================================="
echo " QSPI Flash Programmer — Mandelbrot Explorer"
echo "=========================================="
echo ""

# --- Step 0: Check prerequisites ---
info "Checking prerequisites..."

command -v arm-none-eabi-gcc >/dev/null 2>&1 || error "arm-none-eabi-gcc not found"
command -v bootgen >/dev/null 2>&1 || error "bootgen not found (source Vivado settings64.sh?)"
command -v xsdb >/dev/null 2>&1 || error "xsdb not found (source Vivado settings64.sh?)"

[ -f "$FLASH_WRITER_C" ] || error "$FLASH_WRITER_C not found"
[ -f "$BOOT_BIF" ]       || error "$BOOT_BIF not found"
[ -f "$FSBL_ELF" ]       || error "$FSBL_ELF not found"
[ -f "$BITSTREAM" ]      || error "$BITSTREAM not found"
[ -f "$FLASH_TCL" ]      || error "$FLASH_TCL not found"

info "All prerequisites OK."
echo ""

# --- Step 1: Compile flash_writer.c ---
info "Step 1: Compiling $FLASH_WRITER_C..."

# Recompile if .elf is missing or older than .c
if [ ! -f "$FLASH_WRITER_ELF" ] || [ "$FLASH_WRITER_C" -nt "$FLASH_WRITER_ELF" ]; then
    arm-none-eabi-gcc -mcpu=cortex-a9 -mfloat-abi=soft -O2 \
        -fno-builtin -nostdlib -Ttext=0x0 \
        -o "$FLASH_WRITER_ELF" "$FLASH_WRITER_C"

    # Verify .text section is at 0x0 and total size is reasonable
    TEXT_ADDR=$(arm-none-eabi-objdump -h "$FLASH_WRITER_ELF" | grep '\.text' | awk '{print $4}')
    TEXT_SIZE=$(arm-none-eabi-objdump -h "$FLASH_WRITER_ELF" | grep '\.text' | awk '{print $3}')
    TEXT_SIZE_DEC=$((16#$TEXT_SIZE))

    if [ "$TEXT_ADDR" != "00000000" ]; then
        error ".text section at $TEXT_ADDR, expected 00000000"
    fi
    if [ "$TEXT_SIZE_DEC" -gt 65536 ]; then
        error ".text section is ${TEXT_SIZE_DEC} bytes (>64KB), would collide with control block at 0x20000"
    fi
    info "  Compiled: .text at 0x${TEXT_ADDR}, ${TEXT_SIZE_DEC} bytes"
else
    info "  $FLASH_WRITER_ELF is up to date, skipping compile."
fi
echo ""

# --- Step 2: Generate BOOT.bin ---
info "Step 2: Generating $BOOT_BIN..."

# Check if BOOT.bin needs regeneration
NEED_REGEN=0
if [ ! -f "$BOOT_BIN" ]; then
    NEED_REGEN=1
elif [ "$FSBL_ELF" -nt "$BOOT_BIN" ] || [ "$BITSTREAM" -nt "$BOOT_BIN" ]; then
    NEED_REGEN=1
fi

if [ "$NEED_REGEN" -eq 1 ]; then
    bootgen -image "$BOOT_BIF" -arch zynq -o "$BOOT_BIN" -w
    BOOT_SIZE=$(stat -c%s "$BOOT_BIN" 2>/dev/null || stat -f%z "$BOOT_BIN")
    info "  Generated: $BOOT_BIN (${BOOT_SIZE} bytes)"
else
    BOOT_SIZE=$(stat -c%s "$BOOT_BIN" 2>/dev/null || stat -f%z "$BOOT_BIN")
    info "  $BOOT_BIN is up to date (${BOOT_SIZE} bytes), skipping."
fi

# Sanity check: BOOT.bin should be 1-8 MB
if [ "$BOOT_SIZE" -lt 1000000 ] || [ "$BOOT_SIZE" -gt 8000000 ]; then
    warn "BOOT.bin size (${BOOT_SIZE}) looks unusual (expected 1-8 MB)"
fi

# Estimate programming time
EST_SECS=$((BOOT_SIZE / 5000))
EST_MINS=$((EST_SECS / 60))
echo ""

# --- Step 3: Program flash ---
info "Step 3: Programming QSPI flash via xsdb..."
info "  Estimated time: ~${EST_MINS} minutes for ${BOOT_SIZE} bytes"
echo ""

xsdb "$FLASH_TCL" "$BOOT_BIN" "$FSBL_ELF" "$FLASH_WRITER_ELF"
XSDB_RC=$?

if [ "$XSDB_RC" -ne 0 ]; then
    error "Flash programming failed (xsdb exit code: $XSDB_RC)"
fi

echo ""
echo "=========================================="
info "Flash programming complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Test via soft reset:  xsdb debug_softboot.tcl"
echo "  2. If soft reset works, power off the board"
echo "  3. Move BOOT jumper to QSPI position (MIO[2]=1)"
echo "  4. Power on — board boots Mandelbrot autonomously"
echo ""
