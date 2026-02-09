## z7020_lcd_test.xdc -- Pin constraints for LCD diagnostic test
##
## Based on z7020_sp.xdc for Smart ZYNQ SP board (HelloFPGA)
## Changes from base:
##   - spi_mosi renamed to spi_sda (same pin M15) for bidirectional I/O
##   - Added UART TX pin (L17) for diagnostic output
##   - SDA drive/slew properties adjusted for bidirectional use
##
## Board: Hello-FPGA Smart ZYNQ SP, XC7Z020CLG400-1 (industrial)

## ============================================
## Clock -- 50 MHz PL crystal oscillator
## ============================================
set_property -dict {PACKAGE_PIN M19 IOSTANDARD LVCMOS33} [get_ports clk_50m]
create_clock -period 20.000 -name clk_50m [get_ports clk_50m]

## ============================================
## Reset -- Active low pushbutton (KEY1)
## ============================================
set_property -dict {PACKAGE_PIN K21 IOSTANDARD LVCMOS33} [get_ports rst_n_in]

## ============================================
## SP2 1.47" Display (ST7789V3) -- SPI Interface
## ============================================
## 320x172 RGB565 LCD, 4-wire SPI + reset + backlight
## spi_sda is bidirectional (MOSI for write, MISO for RDDID read)
set_property -dict {PACKAGE_PIN P15 IOSTANDARD LVCMOS33} [get_ports spi_cs_n]
set_property -dict {PACKAGE_PIN N15 IOSTANDARD LVCMOS33} [get_ports spi_sck]
set_property -dict {PACKAGE_PIN M15 IOSTANDARD LVCMOS33} [get_ports spi_sda]
set_property -dict {PACKAGE_PIN R15 IOSTANDARD LVCMOS33} [get_ports spi_dc]
set_property -dict {PACKAGE_PIN L16 IOSTANDARD LVCMOS33} [get_ports lcd_rst_n]
set_property -dict {PACKAGE_PIN T16 IOSTANDARD LVCMOS33} [get_ports lcd_blk]

## ============================================
## LED -- Diagnostic outputs (active high)
## ============================================
set_property -dict {PACKAGE_PIN P20 IOSTANDARD LVCMOS33} [get_ports led_frame]
set_property -dict {PACKAGE_PIN P21 IOSTANDARD LVCMOS33} [get_ports led_alive]

## ============================================
## UART TX -- Diagnostic serial output (active high)
## ============================================
set_property -dict {PACKAGE_PIN L17 IOSTANDARD LVCMOS33} [get_ports tx_out]

## ============================================
## SPI Drive strength and slew rate
## ============================================
## SCK: fast slew for clean edges
set_property SLEW FAST [get_ports spi_sck]
set_property DRIVE 8   [get_ports spi_sck]

## SDA: fast slew, 8mA drive (bidirectional pin)
set_property SLEW FAST [get_ports spi_sda]
set_property DRIVE 8   [get_ports spi_sda]

## ============================================
## Configuration
## ============================================
set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]
