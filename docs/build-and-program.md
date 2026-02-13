# Build and Program

This document covers the hardware, toolchain, pin mapping, build flow, and programming workflow for the neural Mandelbrot explorer.

## Hardware

### Board: Hello-FPGA Smart ZYNQ SP

| Property | Value |
|----------|-------|
| FPGA | XC7Z020CLG484-1 (Zynq-7000, industrial grade) |
| Package | CLG484 (23mm BGA) |
| PL clock | 50 MHz crystal oscillator |
| Display | SP2 1.47" LCD, 320×172, ST7789V3, SPI interface |
| JTAG | Onboard FT2232H (Digilent-compatible) |

Note: The 484-pin BGA package (CLG484) is larger than the common CLG400 development board package. Pin assignments differ — always reference the carrier board schematic.

### Display: SP2 1.47" LCD Module

The display module uses a Sitronix ST7789V3 controller with a 320×240 internal framebuffer. The physical display is 320×172, centered vertically (row offset 34). The SPI interface supports up to 62.5 MHz SCK per the datasheet, though we run at 25 MHz.

## Pin Mapping

From the carrier board schematic (page 11) and `constraints/z7020_sp.xdc`:

### SPI Display

| Signal | Pin | Bank | Standard |
|--------|-----|------|----------|
| CS_N | P15 | 34 | LVCMOS33 |
| SCK | N15 | 34 | LVCMOS33 |
| MOSI | M15 | 34 | LVCMOS33 |
| DC | R15 | 34 | LVCMOS33 |
| RST_N | L16 | 34 | LVCMOS33 |
| BLK | T16 | 34 | LVCMOS33 |

SCK and MOSI have FAST slew rate and 8 mA drive strength configured in the XDC for signal integrity at 25 MHz.

### System

| Signal | Pin | Function |
|--------|-----|----------|
| clk_50m | M19 | 50 MHz PL oscillator |
| rst_n_in | K21 | KEY1 pushbutton (active low) |
| led_frame | P20 | LED1 — toggles per compute frame |
| led_alive | P21 | LED2 — 1 Hz heartbeat |

### PL UART (Boot Diagnostics)

| Signal | Pin | Bank | Standard |
|--------|-----|------|----------|
| uart_tx | L17 | 35 | LVCMOS33 |

L17 is routed to FT2232H Channel B (ttyUSB1 on host). Read with pyserial at 115200 8N1 (`cat /dev/ttyUSB1` does not work). The `boot_msg` module sends `MANDELBROT QSPI OK\r\n` repeating every ~2 seconds to confirm the PL bitstream is running.

### Other Available Pins (Not Used)

| Signal | Pin | Notes |
|--------|-----|-------|
| KEY2 | J20 | Second pushbutton |
| UART RX | M17 | PL UART receive (FT2232H Channel B) |

## Toolchain

### Vivado 2025.2

The design requires AMD/Xilinx Vivado ML Standard edition with Zynq-7000 device support. See [INSTALL.md](../INSTALL.md) for detailed installation instructions including workarounds for the self-extracting installer bug on Ubuntu 25.10.

### Environment Setup

```bash
source ~/Code/z7020/env.sh
```

This sources Vivado's `settings64.sh` and adds `vivado`, `hw_server`, and related tools to `PATH`.

## Development Setup

Primary development is on the x86_64 workstation (msi-z790) where Vivado is installed. The FPGA board connects via USB for JTAG programming.

```
  x86_64 Workstation (msi-z790)
  ─────────────────────────────
  ~/Code/z7020/           Project source tree
  /opt/Xilinx/2025.2/    Vivado toolchain

  FPGA board ◀── USB cable ──▶ msi-z790
  (JTAG + UART via FT2232H)
```

An alternative setup uses a Raspberry Pi 5 for proximity to the FPGA board, with synthesis running remotely on msi-z790 via SSH. The source tree is shared via SSHFS and the JTAG USB device is forwarded using `usbip`.

## JTAG

The FPGA board's JTAG interface (FT2232H) provides two USB endpoints:
- **ttyUSB0** (Channel A): JTAG — claimed by `hw_server`
- **ttyUSB1** (Channel B): UART — 115200 8N1, read with pyserial

When the board is connected directly to msi-z790:

```bash
# Start hw_server (if not already running):
hw_server &

# Program via JTAG:
vivado -mode batch -source vivado/program.tcl > /tmp/program.log 2>&1
```

### JTAG over USB/IP (Remote Pi Setup)

When using the Pi setup, `usbip` forwards the USB device over the network:

```bash
# On the Pi (server):
sudo usbipd -D                         # Start USB/IP daemon

# On the workstation (client):
./fpga-jtag.sh attach                   # Attach JTAG via USB/IP
./fpga-jtag.sh status                   # Check connection
./fpga-jtag.sh detach                   # Release when done
```

## Build Flow

### One-Step Build

```bash
# Mandelbrot mode (default)
source env.sh
vivado -mode batch -source vivado/run_all.tcl

# MLP neural inference mode
vivado -mode batch -source vivado/run_all.tcl -tclargs 1
```

The `-tclargs 1` passes `COMPUTE_MODE=1` to the build script, which selects `mlp_core` instead of `neuron_core` in the generate block. Without this argument, the build defaults to Mandelbrot mode (`COMPUTE_MODE=0`).

`run_all.tcl` executes the complete flow:

1. **Create project** — adds RTL sources, constraints, sets top module and synthesis/implementation strategies
2. **Synthesis** (`synth_1`) — maps RTL to device primitives, infers DSPs and BRAMs
3. **Implementation** (`impl_1`) — place and route
4. **Bitstream generation** — produces `mandelbrot_top.bit`

The bitstream lands at:
```
vivado/mandelbrot/mandelbrot.runs/impl_1/mandelbrot_top.bit
```

### Synthesis Strategy

```tcl
set_property strategy Flow_PerfOptimized_high [get_runs synth_1]
set_property strategy Performance_Explore [get_runs impl_1]
```

- **Flow_PerfOptimized_high**: Aggressive DSP inference, retiming, and LUT optimization
- **Performance_Explore**: Tries multiple placement seeds and routing algorithms for best timing

### Build Time

On an Intel Core i7-13700K (8P+8E cores):
- Synthesis: ~2 minutes
- Implementation: ~4 minutes
- Bitstream: ~1 minute
- **Total: ~7 minutes**

### Checking Build Results

Never pipe Vivado output directly through SSH — timing reports alone can be 60K+ characters. Instead:

```bash
# Check for errors
grep -i "ERROR\|CRITICAL" vivado/build.log

# Check utilization
grep -A5 "Slice LUTs\|Slice Registers\|Block RAM\|DSPs" vivado/build.log

# Check timing
grep "Worst Negative Slack\|Total Negative Slack" vivado/build.log
```

## Programming

### Via JTAG

```bash
vivado -mode batch -source vivado/program.tcl > vivado/program.log 2>&1
```

This opens the hardware manager, connects to the JTAG target (xc7z020_1), and programs the bitstream. The display should initialize within ~500 ms (reset + sleep-out + init sequence) and begin rendering.

### Verification

After programming:
- **LED2 (alive)**: Should blink at 1 Hz immediately — confirms the FPGA is configured and running
- **LED1 (frame)**: Should toggle every ~40 ms — confirms compute frames are completing
- **Display**: Shows the Mandelbrot set, then begins auto-zooming into the seahorse valley

If LED2 blinks but the display is blank:
1. Check that the backlight pin (T16) has the external pull-up (some board revisions)
2. Verify SPI pin assignments match your board revision
3. Try reducing SCK_DIV to slow down SPI for debugging

### Separate Steps

For iterative development, you can run synthesis, implementation, and programming separately:

```bash
vivado -mode batch -source vivado/create_project.tcl   # Create project once
vivado -mode batch -source vivado/run_synth.tcl         # Synthesis only
vivado -mode batch -source vivado/run_impl.tcl          # Implementation only
vivado -mode batch -source vivado/program.tcl           # Program
```

## Constraints File

`constraints/z7020_sp.xdc` defines:

1. **Clock**: 50 MHz on pin M19 with `create_clock -period 20.000`
2. **Pin assignments**: All I/O pins with LVCMOS33 standard
3. **Drive strength**: SPI SCK and MOSI at FAST slew, 8 mA drive
4. **Configuration**: `CFGBVS VCCO`, `CONFIG_VOLTAGE 3.3`

The timing constraint is straightforward — at 50 MHz (20 ns period), the design has substantial margin. All paths meet timing comfortably.

## Persistent Boot (Power-On Auto-Start)

JTAG programming is volatile — the design is lost on power-off. To make the Mandelbrot explorer start automatically at power-on, we use a Zynq boot image (BOOT.bin) containing an FSBL + bitstream.

The board has a BOOT jumper (S1/S2) with three positions: **JTAG**, **QSPI**, and **SD**.

### How Zynq Boot Works

On a Zynq-7000, the PL (FPGA fabric) cannot load directly from storage. Instead, the PS (ARM core) boots first:

```
  Power-on → PS BootROM → loads FSBL → FSBL configures PL → design runs
```

1. **BootROM**: Hardwired in silicon, reads BOOT.bin from the selected boot device
2. **FSBL** (First Stage Boot Loader): Minimal ARM program that initializes PS clocks, DDR, and MIO, reads the bitstream, and configures the PL
3. **PL configuration**: The FSBL writes the bitstream into the PL via the PCAP interface

Even though our design is PL-only (no ARM software), we need the FSBL as a bootstrap mechanism. The FSBL runs, loads the bitstream, and then the ARM idles while the PL runs the Mandelbrot explorer.

### QSPI Boot — BROKEN (BootROM v2.0 Bug)

**QSPI boot does not work on this board.** The silicon (XC7Z020 rev 2.0, PSS_IDCODE=0x23727093) has a BootROM that does not configure MIO[1-6] L0_SEL=1, which is required for QSPI controller signals to reach the flash chip. Without L0_SEL=1, the QSPI flash is electrically disconnected from the Zynq's SPI controller at power-on.

This is a chicken-and-egg problem:
- The BootROM needs to read BOOT.bin from flash to find the register init table
- The register init table would set L0_SEL=1 to enable flash access
- But L0_SEL=0 at POR means the BootROM can't read the flash at all

A scan of the entire 64KB BootROM (0xFFFF0000-0xFFFFFFFF) confirmed: **no MIO register addresses, no SLCR unlock key, no QSPI controller addresses exist in the ROM**. The BootROM simply does not have MIO initialization code for QSPI boot.

The custom flash programmer (`flash_writer.c` + `flash_custom.tcl`) works correctly for writing to QSPI flash via JTAG, and the flash contents have been byte-verified. The hardware is fine — it's the BootROM that's broken.

### SD Card Boot — RECOMMENDED

Use SD card boot as the workaround. The FSBL auto-detects the boot device from the `BOOT_MODE` register and reads partitions accordingly — the same BOOT.bin works for both SD and QSPI.

**SD card pinout** (core board MIO[40-45], Bank 501, 1.8V):

| Signal | MIO | Package Pin |
|--------|-----|-------------|
| SD_CLK | MIO[40] | D14 |
| SD_CMD | MIO[41] | C17 |
| SD_D0 | MIO[42] | E12 |
| SD_D1 | MIO[43] | A9 |
| SD_D2 | MIO[44] | F13 |
| SD_D3 | MIO[45] | B15 |

### Prerequisites

```bash
# On the build machine (msi-z790):
sudo apt install gcc-arm-none-eabi      # ARM cross-compiler for FSBL
source ~/Code/z7020/env.sh              # Vivado in PATH
```

### Step 1: Generate FSBL (One-Time)

```bash
cd ~/Code/z7020/vivado
vivado -mode batch -source create_fsbl.tcl > /tmp/create_fsbl.log 2>&1
```

This creates a PS7 block design with QSPI and SD0 (MIO[40-45]) enabled, exports a hardware platform (XSA), and compiles the FSBL using `xsdb` + `arm-none-eabi-gcc`. Output:

```
vivado/boot/fsbl/executable.elf
```

The FSBL only needs to be regenerated if the PS7 configuration changes.

### Step 2: Build Bitstream

```bash
vivado -mode batch -source vivado/run_all.tcl > /tmp/build.log 2>&1
```

### Step 3: Create BOOT.bin

```bash
cd ~/Code/z7020/vivado
bootgen -image boot.bif -arch zynq -o BOOT.bin -w
```

The boot image format (`vivado/boot.bif`):

```
the_ROM_image:
{
    [bootloader] boot/fsbl/executable.elf
    mandelbrot/mandelbrot.runs/impl_1/mandelbrot_top.bit
}
```

### Step 4: Prepare SD Card

1. Format a microSD card as **FAT32** (any size, even 1 GB works)
2. Copy `vivado/BOOT.bin` to the **root** of the SD card (filename must be `BOOT.bin` or `BOOT.BIN`)
3. Eject safely

### Step 5: Boot

1. Insert the microSD card into the TF card slot on the carrier board
2. Set S1/S2 jumper to **BOOT SD** mode
3. Power cycle the board
4. The Mandelbrot animation should appear on the LCD within ~1 second

### Re-Building After Design Changes

After modifying RTL and rebuilding the bitstream, only steps 2-4 are needed (the FSBL doesn't change):

```bash
vivado -mode batch -source vivado/run_all.tcl > /tmp/build.log 2>&1
cd vivado && bootgen -image boot.bif -arch zynq -o BOOT.bin -w
# Copy BOOT.bin to SD card
```

### Troubleshooting

**"FSBL not found"**: Run `vivado -mode batch -source vivado/create_fsbl.tcl` first.

**"arm-none-eabi-gcc not found"**: `sudo apt install gcc-arm-none-eabi`.

**Board doesn't boot from SD**: Verify the S1/S2 jumper is in SD position. Check that the SD card is FAT32 formatted and BOOT.bin is at the root. Try a different SD card (some older cards have compatibility issues with 1.8V signaling).

**No display but UART shows "MANDELBROT QSPI OK"**: The PL bitstream loaded successfully. Check LCD ribbon cable and SPI connections.

**No UART output and no display**: The FSBL may have failed. Try programming via JTAG first to verify the bitstream works, then retry SD boot.
