# flash_minimal.tcl — Program minimal BOOT.bin (FSBL only) for boot test
# Usage: xsdb flash_minimal.tcl

set fsbl "/home/ian/Code/z7020/vivado/boot/fsbl/executable.elf"
set fw   "/tmp/flash_writer.elf"
set bootbin "/home/ian/Code/z7020/vivado/BOOT_minimal.bin"

connect
targets -set -filter {name =~ "ARM*#0"}

puts "=== Flashing Minimal BOOT.bin (FSBL only, 71KB) ==="

# Load FSBL
rst -system
after 1000
dow $fsbl
con
after 5000
catch {stop}
after 500

# Load flash_writer
dow $fw
mwr -force 0x00020000 0
mwr -force 0x0002000C 0
mwr -force 0x00020018 0
con
after 3000
catch {stop}
after 500

set jedec [mrd -force -value 0x00020018]
puts "JEDEC ID: [format 0x%06X $jedec]"
set sr2 [mrd -force -value 0x0002001C]
puts "SR2: [format 0x%02X $sr2] (QE=[expr {($sr2 >> 1) & 1}])"

if {$jedec == 0} {
    puts "ERROR: Flash not detected"
    disconnect
    exit 1
}

# Load BOOT_minimal.bin to DDR
puts "\nLoading BOOT_minimal.bin to DDR..."
dow -data $bootbin 0x01000000
after 2000

set filesize [file size $bootbin]
puts "File size: $filesize bytes"

# Program flash
puts "\nProgramming flash..."
mwr -force 0x00020004 0x01000000
mwr -force 0x00020008 $filesize
mwr -force 0x0002000C 0
mwr -force 0x00020000 1
con

# Poll progress
set timeout 120
set t0 [clock seconds]
while {1} {
    after 2000
    set status [mrd -force -value 0x0002000C]
    set progress [mrd -force -value 0x00020010]

    if {$status == 1} {
        puts "  Program: DONE ($progress bytes)"
        break
    } elseif {$status == 0xFF} {
        set erroff [mrd -force -value 0x00020014]
        set errcode [mrd -force -value 0x0002001C]
        puts "  Program: FAILED at offset [format 0x%X $erroff] (code=$errcode)"
        disconnect
        exit 1
    }

    set elapsed [expr {[clock seconds] - $t0}]
    puts "  Progress: $progress / $filesize bytes ($elapsed s)"

    if {$elapsed > $timeout} {
        puts "  TIMEOUT!"
        catch {stop}
        disconnect
        exit 1
    }
}

# Verify
puts "\nVerifying..."
catch {stop}
after 500
dow $fw
mwr -force 0x00020000 0
mwr -force 0x0002000C 0
mwr -force 0x00020018 0
con
after 3000
catch {stop}
after 500

mwr -force 0x00020004 0x01000000
mwr -force 0x00020008 $filesize
mwr -force 0x0002000C 0
mwr -force 0x00020000 2
con

set t0 [clock seconds]
while {1} {
    after 2000
    set status [mrd -force -value 0x0002000C]
    set progress [mrd -force -value 0x00020010]

    if {$status == 1} {
        puts "  Verify: PASS ($progress bytes)"
        break
    } elseif {$status == 0xFF} {
        set erroff [mrd -force -value 0x00020014]
        puts "  Verify: FAILED at offset [format 0x%X $erroff]"
        disconnect
        exit 1
    }

    set elapsed [expr {[clock seconds] - $t0}]
    if {$elapsed > $timeout} {
        puts "  TIMEOUT!"
        catch {stop}
        disconnect
        exit 1
    }
}

puts "\n=== Done! Switch to QSPI jumper and power cycle to test ==="
puts "If FSBL boots, you should see activity on UART (if connected)"
puts "LED may not change since there's no bitstream in this minimal image"

disconnect
