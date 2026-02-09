## z7020_sp.xdc — Pin constraints for Smart ZYNQ SP board (HelloFPGA)
##
## Pin assignments from carrier board schematic page 11 pin mapping table.
## Board: Hello-FPGA Smart ZYNQ SP, XC7Z020CLG400-1 (industrial)

## ============================================
## Clock — 50 MHz PL crystal oscillator
## ============================================
set_property -dict {PACKAGE_PIN M19 IOSTANDARD LVCMOS33} [get_ports clk_50m]
create_clock -period 20.000 -name clk_50m [get_ports clk_50m]

## ============================================
## Reset — Active low pushbutton (KEY1)
## ============================================
set_property -dict {PACKAGE_PIN K21 IOSTANDARD LVCMOS33} [get_ports rst_n_in]

## ============================================
## SP2 1.47" Display (ST7789V3) — SPI Interface
## ============================================
## 320x172 RGB565 LCD, 4-wire SPI + reset + backlight
set_property -dict {PACKAGE_PIN P15 IOSTANDARD LVCMOS33} [get_ports spi_cs_n]
set_property -dict {PACKAGE_PIN N15 IOSTANDARD LVCMOS33} [get_ports spi_sck]
set_property -dict {PACKAGE_PIN M15 IOSTANDARD LVCMOS33} [get_ports spi_mosi]
set_property -dict {PACKAGE_PIN R15 IOSTANDARD LVCMOS33} [get_ports spi_dc]
set_property -dict {PACKAGE_PIN L16 IOSTANDARD LVCMOS33} [get_ports lcd_rst_n]
set_property -dict {PACKAGE_PIN T16 IOSTANDARD LVCMOS33} [get_ports lcd_blk_out]

## ============================================
## LED — Frame heartbeat (LED1)
## ============================================
set_property -dict {PACKAGE_PIN P20 IOSTANDARD LVCMOS33} [get_ports led_frame]
set_property -dict {PACKAGE_PIN P21 IOSTANDARD LVCMOS33} [get_ports led_alive]

## ============================================
## SPI Drive strength and slew rate
## ============================================
## ST7789V3 SPI can handle fast edges; FAST slew for better signal integrity
set_property SLEW FAST [get_ports spi_sck]
set_property SLEW FAST [get_ports spi_mosi]
set_property DRIVE 8 [get_ports spi_sck]
set_property DRIVE 8 [get_ports spi_mosi]

## ============================================
## Configuration
## ============================================
set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]
