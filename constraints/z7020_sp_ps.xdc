## z7020_sp_ps.xdc — Pin constraints for PS-integrated MLP build
##
## Same pin assignments as z7020_sp.xdc but:
##   - No PL clock constraint (FCLK_CLK0 from PS7 replaces M19 crystal)
##   - No external reset pin (FCLK_RESET0_N from PS7)
##   - Port names match mlp_pl_wrapper hierarchy

## ============================================
## SP2 1.47" Display (ST7789V3) — SPI Interface
## ============================================
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
## UART TX — Boot diagnostic output (115200 8N1)
## ============================================
set_property -dict {PACKAGE_PIN L17 IOSTANDARD LVCMOS33} [get_ports uart_tx]

## ============================================
## SPI Drive strength and slew rate
## ============================================
set_property SLEW FAST [get_ports spi_sck]
set_property SLEW FAST [get_ports spi_mosi]
set_property DRIVE 8 [get_ports spi_sck]
set_property DRIVE 8 [get_ports spi_mosi]

## ============================================
## Configuration
## ============================================
set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]
