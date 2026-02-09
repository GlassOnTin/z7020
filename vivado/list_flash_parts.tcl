# list_flash_parts.tcl — Query available SPI flash parts for this device
open_hw_manager
connect_hw_server -allow_non_jtag
open_hw_target
set device [get_hw_devices xc7z020_1]
current_hw_device $device
puts "=== QSPI flash parts for [get_property PART $device] ==="
foreach p [get_cfgmem_parts -of_objects [current_hw_device]] {
    if {[string match "*qspi*" $p] && [string match "*single*" $p]} {
        puts "  $p"
    }
}
close_hw_manager
