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

### Other Available Pins (Not Used)

| Signal | Pin | Notes |
|--------|-----|-------|
| KEY2 | J20 | Second pushbutton |
| UART TX | L17 | PS UART (for future AXI-Lite control) |
| UART RX | M17 | PS UART |

## Toolchain

### Vivado 2025.2

The design requires AMD/Xilinx Vivado ML Standard edition with Zynq-7000 device support. See [INSTALL.md](../INSTALL.md) for detailed installation instructions including workarounds for the self-extracting installer bug on Ubuntu 25.10.

### Environment Setup

```bash
source ~/Code/z7020/env.sh
```

This sources Vivado's `settings64.sh` and adds `vivado`, `hw_server`, and related tools to `PATH`.

## Remote Build Setup

The project is developed on a Raspberry Pi 5 (for proximity to the FPGA board) with synthesis running on a remote x86_64 workstation (where Vivado is installed). The source tree is shared via SSHFS:

```
  Raspberry Pi 5                      x86_64 Workstation (msi-z790)
  ─────────────                       ─────────────────────────────
  ~/Code/z7020/  ◀── SSHFS mount ──▶  /opt/Xilinx/2025.2/
  (edit RTL)                          (run Vivado synthesis)

  FPGA board ◀── USB cable ──▶ Pi ◀── USB/IP ──▶ Workstation
  (JTAG)                                          (hw_server)
```

## JTAG over USB/IP

The FPGA board's JTAG interface (FT2232H) connects to the Raspberry Pi via USB. For programming from the remote workstation, `usbip` forwards the USB device over the network:

```bash
# On the Pi (server):
sudo usbipd -D                         # Start USB/IP daemon

# On the workstation (client):
./fpga-jtag.sh attach                   # Attach JTAG via USB/IP
./fpga-jtag.sh status                   # Check connection
./fpga-jtag.sh detach                   # Release when done
```

The `fpga-jtag.sh` script wraps `usbip attach/detach` with the correct device bus ID and server address.

## Build Flow

### One-Step Build

```bash
ssh msi-z790 "cd /path/to/z7020 && \
    source env.sh && \
    vivado -mode batch -source vivado/run_all.tcl > vivado/build.log 2>&1"
```

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
- **LED1 (frame)**: Should toggle every ~70 ms — confirms compute frames are completing
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
