# set_qe.tcl — Set Quad Enable (QE) bit in flash Status Register 2
# Required for BootROM QSPI boot (uses 0x6B Quad Output Fast Read)
# Usage: xsdb set_qe.tcl

set fsbl "/home/ian/Code/z7020/vivado/boot/fsbl/executable.elf"
set fw   "/tmp/flash_writer.elf"

connect

# Handle both JTAG and QSPI boot modes
# In QSPI mode with failed boot, ARM cores may not be visible
# Need to target APU first, then ARM core appears after reset
puts "Available targets:"
targets

# Try to find ARM core; if not visible, target DAP and reset
if {[catch {targets -set -filter {name =~ "ARM*#0"}}]} {
    puts "ARM core not visible — targeting via APU/DAP..."
    # Target the APU
    if {[catch {targets -set -filter {name =~ "APU*"}}]} {
        # Fall back to DAP
        targets -set -filter {name =~ "DAP*"}
    }
}

puts "=== Setting QE bit for QSPI boot ==="

# System reset to get ARM core into known state
rst -system
after 2000

# After reset, ARM cores should be visible
targets -set -filter {name =~ "ARM*#0"}
dow $fsbl
con
after 5000
catch {stop}
after 500

# Load flash_writer (it calls flash_enable_quad() during init)
dow $fw
# Clear control block
mwr -force 0x00020000 0
mwr -force 0x0002000C 0
mwr -force 0x00020018 0
con
after 3000
catch {stop}
after 500

# Read results
set jedec [mrd -force -value 0x00020018]
puts "JEDEC ID: [format 0x%06X $jedec]"

if {$jedec == 0} {
    puts "ERROR: Flash not detected"
    disconnect
    exit 1
}

# Read SR2 from ctrl[7] — flash_writer stores it after flash_enable_quad()
set sr2 [mrd -force -value 0x0002001C]
puts "Status Register 2: [format 0x%02X $sr2]"
if {$sr2 & 0x02} {
    puts "  QE bit is SET — Quad mode enabled"
} else {
    puts "  QE bit is NOT set — Quad mode FAILED"
}

# Now read SR2 to confirm QE is set
# We need to do another QSPI transfer — easiest to just check
# the flash_writer already did it, so let's verify by reading SR2 directly
# via the flash_writer's qspi_xfer

# Actually, the flash_writer already called flash_enable_quad() during init.
# Let's verify by re-initializing and reading SR2.
# Simplest: just restart flash_writer and check if QE was already set
# (it returns immediately if QE bit is already 1)

puts ""
puts "Flash writer initialized — QE bit should now be set."
puts "To verify, we can read SR2 via a manual QSPI transfer..."

# Use flash_writer to do a raw SR2 read by issuing cmd 0x35
# But that's complex. Instead, trust the code and do a quick LQSPI test.

# Disable controller for LQSPI test
mwr -force 0xE000D014 0
after 10

# Set LQSPI_CFG to match what BootROM uses: 0x8000016B
# Quad Output Fast Read (0x6B), 1 dummy byte
mwr -force 0xE000D0A0 0x8000016B

# CR for LQSPI mode
mwr -force 0xE000D000 0x800238C1

# Enable controller
mwr -force 0xE000D014 1
after 10

puts ""
puts "=== LQSPI test read (0xFC000000) ==="
set ok 1
for {set i 0} {$i < 8} {incr i} {
    set addr [expr {0xFC000000 + $i * 4}]
    if {[catch {set val [mrd -force -value $addr]} err]} {
        puts "  [format 0x%08X $addr]: ERROR - $err"
        set ok 0
        break
    } else {
        puts "  [format 0x%08X $addr]: [format 0x%08X $val]"
    }
}

if {$ok} {
    set w0 [mrd -force -value 0xFC000000]
    if {$w0 == 0xEAFFFFFE} {
        puts ""
        puts "SUCCESS: LQSPI Quad Read works! BOOT.bin header detected."
        puts "QSPI boot should work now. Power cycle with QSPI jumper."
    } elseif {$w0 == 0} {
        puts ""
        puts "WARNING: LQSPI reads zeros — QE may not be set or LQSPI config wrong"
    } else {
        puts ""
        puts "WARNING: Unexpected data at flash offset 0 (expected 0xEAFFFFFE)"
    }
}

disconnect
