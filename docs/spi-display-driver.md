# SPI Display Driver — ST7789V3 LCD Interface

The `sp2_spi_driver` module handles the complete lifecycle of the 1.47" 320x172 LCD: hardware reset, controller initialization, window setup, and continuous pixel streaming over 4-wire SPI. This document covers the SPI timing, initialization sequence, and the critical MOSI setup-time fix that got the display working.

## Hardware

- **Controller**: ST7789V3 (Sitronix)
- **Resolution**: 320 × 172 pixels (centered in the controller's 320 × 240 frame)
- **Color format**: RGB565 (16 bits per pixel, 5-6-5)
- **Interface**: 4-wire SPI + reset + backlight
- **SPI mode**: Mode 0 (CPOL=0, CPHA=0)

### Pin Mapping

| Signal | FPGA Pin | Function |
|--------|----------|----------|
| `spi_cs_n` | P15 | Chip select (active low) |
| `spi_sck` | N15 | SPI clock (25 MHz) |
| `spi_mosi` | M15 | Serial data out |
| `spi_dc` | R15 | Data/command select (0=cmd, 1=data) |
| `lcd_rst_n` | L16 | Hardware reset (active low) |
| `lcd_blk_out` | T16 | Backlight enable (has external pull-up) |

## SPI Mode 0 Timing

SPI Mode 0 means:
- **CPOL=0**: SCK idles LOW
- **CPHA=0**: Data is sampled on the **rising** edge of SCK; data is set up on the **falling** edge

The byte shifter uses a 4-bit `bit_phase` counter (0-15) to generate 8 bits with two half-cycles each:

| Phase | SCK | Action |
|-------|-----|--------|
| 0 (even) | LOW | MOSI holds MSB (pre-loaded at start_byte) |
| 1 (odd) | HIGH | ST7789 samples MOSI |
| 2 (even) | LOW | MOSI still holds same bit |
| 3 (odd) | HIGH | ST7789 samples; pre-load next bit |
| ... | ... | ... |
| 14 (even) | LOW | MOSI holds LSB |
| 15 (odd) | HIGH | ST7789 samples LSB; shifting complete |

### SCK Generation

SCK is generated combinationally to ensure zero extra edges between bytes:

```verilog
assign spi_sck = ~spi_cs_n & shifting & bit_phase[0];
```

This means SCK is:
- HIGH only during odd phases (bit_phase[0] = 1)
- LOW during even phases (data setup time)
- LOW when not shifting (no spurious edges between bytes)
- LOW when CS is deasserted

### MOSI Setup Time (Critical Fix)

The most important timing requirement is that **MOSI must be stable before SCK rises**. A naive implementation that updates MOSI and toggles SCK on the same clock edge causes setup-time violations at the LCD — the data arrives too late for the controller to sample reliably.

The fix is to **pre-load MOSI one phase ahead**:

```verilog
if (start_byte && !shifting) begin
    shift_reg <= next_byte;
    spi_mosi  <= next_byte[7];      // Pre-load MSB before first SCK rise
end else if (shifting) begin
    if (!bit_phase[0]) begin
        // Even phase: MOSI already stable (set during previous odd phase)
    end else begin
        // Odd phase: SCK is HIGH, slave is sampling
        shift_reg <= {shift_reg[6:0], 1'b0};
        spi_mosi  <= shift_reg[6];  // Pre-load NEXT bit for next even phase
    end
end
```

Timeline for one bit:

```
  Odd phase N-1:  MOSI set to bit[i]     ← MOSI changes here
  Even phase N:   SCK falls, MOSI stable  ← full half-cycle of setup time
  Odd phase N:    SCK rises               ← ST7789 samples, MOSI has been stable
```

This gives a full clock half-cycle (20 ns at 50 MHz, 10 ns at SCK = 25 MHz) of MOSI setup time before each sampling edge.

### Prescaler

The `SCK_DIV` parameter controls the SPI clock rate:

```
  SCK frequency = clk / (2 × (SCK_DIV + 1))
```

| SCK_DIV | SCK Frequency | Notes |
|---------|---------------|-------|
| 0 | 25 MHz | Maximum rate (used in production) |
| 24 | 1 MHz | Useful for debugging with slow logic analyzer |
| 249 | 100 kHz | Ultra-slow for initial bring-up |

## State Machine

```
  ST_RESET ──▶ ST_RESET_REL ──▶ ST_SLPOUT ──▶ ST_SLPOUT_WAIT
                                                      │
                                                      ▼
                                               ST_INIT
                                                      │
                                        ┌─────────────┘
                                        ▼
                                  ST_SET_WIN ◀───── ST_FRAME_END
                                        │                ▲
                                        ▼                │
                                    ST_PIXEL ────────────┘
```

### Startup Sequence

1. **ST_RESET** (120 ms): Assert `lcd_rst_n = 0` for hardware reset
2. **ST_RESET_REL** (120 ms): Release reset, wait for controller stabilization
3. **ST_SLPOUT**: Send SLPOUT command (0x11) to wake the display
4. **ST_SLPOUT_WAIT** (120 ms): Mandatory wait after sleep-out per datasheet
5. **ST_INIT**: Send initialization commands from ROM (see below)

### Initialization Commands

The init ROM stores 30 command/data bytes as 9-bit entries (bit[8] = DC flag):

| Command | Hex | Parameters | Purpose |
|---------|-----|------------|---------|
| COLMOD | 0x3A | 0x55 | RGB565 color mode |
| MADCTL | 0x36 | 0x60 | Landscape orientation (MV\|MX) |
| PORCTRL | 0xB2 | 5 bytes | Porch timing |
| GCTRL | 0xB7 | 0x35 | Gate voltage control |
| VCOMS | 0xBB | 0x19 | VCOM setting |
| LCMCTRL | 0xC0 | 0x2C | LCM control |
| VDVVRHEN | 0xC2 | 0x01 | VDV and VRH enable |
| VRHS | 0xC3 | 0x12 | VRH setting |
| VDVS | 0xC4 | 0x20 | VDV setting |
| FRCTRL2 | 0xC6 | 0x0F | Frame rate: 60 Hz |
| PWCTRL1 | 0xD0 | 0xA4, 0xA1 | Power control |
| INVON | 0x21 | (none) | Color inversion on (required for correct colors) |
| NORON | 0x13 | (none) | Normal display mode |
| DISPON | 0x29 | (none) | Display on |

### Frame Loop

After initialization, the driver enters a continuous frame loop:

1. **ST_SET_WIN**: Send CASET (column address: 0-319), RASET (row address: 34-205), and RAMWR (memory write command). The row offset of 34 centers the 172-pixel-tall viewport within the controller's 240-pixel frame.

2. **ST_PIXEL**: Stream all 55,040 pixels as two bytes each (high byte first). The framebuffer address increments linearly. Each pixel read follows:
   - `pixel_hi = 1`: Latch `fb_data` from BRAM, send high byte
   - `pixel_hi = 0`: Send low byte, advance `fb_addr`

3. **ST_FRAME_END**: Wait for last byte to finish shifting, pulse `frame_done`, loop back to ST_SET_WIN.

### Frame Timing

At 25 MHz SCK, each byte takes 8 SCK cycles = 16 system clocks (320 ns). Each pixel is 2 bytes:

```
  Pixel time   = 640 ns
  Frame pixels = 55,040
  Pixel data   = 55,040 × 640 ns = 35.2 ms
  + Window setup (11 command bytes) ≈ 3.5 µs
  + Inter-byte gaps ≈ negligible

  Total frame time ≈ 35.2 ms → ~28 FPS theoretical
```

In practice, the measured frame rate is approximately 14 FPS due to byte-level handshaking overhead (the `start_byte`/`byte_done` protocol adds one idle cycle between bytes).

## Framebuffer Interface

The SPI driver reads pixels from the display framebuffer through a simple address/data interface:

```verilog
output reg  [15:0] fb_addr,    // Read address (0 to 55039)
input  wire [15:0] fb_data,    // RGB565 pixel data (1-cycle read latency from BRAM)
```

The driver advances `fb_addr` sequentially during ST_PIXEL. The BRAM has a 1-cycle read latency, which the driver accounts for by latching `fb_data` on the high-byte phase before sending the low byte.
