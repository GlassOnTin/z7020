# System Architecture

This document describes the overall architecture of the neural Mandelbrot explorer, covering the module hierarchy, data flow, memory architecture, and clock domain.

## Block Diagram

The design is a single-clock datapath that generates Mandelbrot set images and streams them to an SPI-connected LCD:

```
                 50 MHz
                   │
            ┌──────┴──────┐
            │ mandelbrot_  │
            │    top       │
            │              │
            │  ┌────────┐  │     viewport params
            │  │Auto-Zoom├──────────────────┐
            │  │Controller│ │               │
            │  └────────┘  │               ▼
            │              │    ┌──────────────────┐
            │              │    │ pixel_scheduler   │
            │              │    │                   │
            │              │    │  coord gen ──▶ assign ──▶ collect │
            │              │    └───┬──────────────┬┘
            │              │   valid│  c_re,c_im   │result
            │         ┌────┴───────┴┐              │
            │         │  Neuron Pool │              │
            │         │  ┌───┐┌───┐ │              │
            │         │  │ 0 ││ 1 │…│              │
            │         │  └───┘└───┘ │              │
            │         │  (×18, gen) │              │
            │         └─────────────┘              │
            │              │                       │
            │              │          ┌────────────┘
            │              │          │ wr_en, addr, data
            │              │          ▼
            │         ┌────────────────────┐
            │         │   iter_fb (BRAM)   │  16-bit iteration counts
            │         │   65536 × 16-bit   │  (55040 pixels used)
            │         └────────┬───────────┘
            │                  │ read port
            │                  ▼
            │         ┌────────────────────┐
            │         │   colormap_lut     │  iter → RGB565
            │         │   256-entry palette│  (1 cycle latency)
            │         └────────┬───────────┘
            │                  │
            │                  ▼
            │         ┌────────────────────┐
            │         │   disp_fb (BRAM)   │  RGB565 pixels
            │         │   65536 × 16-bit   │
            │         └────────┬───────────┘
            │                  │ read port
            │                  ▼
            │         ┌────────────────────┐
            │         │  sp2_spi_driver    │
            │         │  (ST7789V3)        │──── SPI ────▶ LCD
            │         └────────────────────┘
            └──────────────────────────────┘
```

## Data Flow

The pipeline processes one frame in five stages:

### 1. Viewport Setup (Auto-Zoom Controller)

The auto-zoom controller in `mandelbrot_top.v` provides viewport parameters to the scheduler: `zoom_cre_start`, `zoom_cim_start`, and `zoom_step`. After each compute frame completes, it shrinks the step by factor 63/64 (a right-shift subtraction) and recomputes the viewport origin to keep the target centered. See [auto-zoom.md](auto-zoom.md).

### 2. Coordinate Generation and Work Distribution (Pixel Scheduler)

`pixel_scheduler.v` generates complex-plane coordinates for each of the 55,040 pixels (320 x 172) via row-major accumulation. It maintains the current coordinate `(cur_c_re, cur_c_im)` and advances by `c_re_step` per pixel horizontally and `c_im_step` per row. A priority encoder scans the neuron pool's ready signals to find an idle neuron, then asserts `neuron_valid[i]` with the coordinate and pixel ID on a shared bus. See [parallel-scheduling.md](parallel-scheduling.md).

### 3. Mandelbrot Iteration (Neuron Pool)

Eighteen `neuron_core` instances iterate z = z² + c independently. Each neuron has three pipelined fixed-point multipliers running in parallel (z_re², z_im², z_re×z_im), giving a throughput of one iteration per 4 clock cycles. When a neuron escapes or reaches `max_iter`, it emits a one-cycle `result_valid` pulse with the pixel ID and final iteration count. See [neuron-core.md](neuron-core.md).

### 4. Color Mapping

`colormap_lut.v` translates the 16-bit iteration count to a 16-bit RGB565 color via a 256-entry cyclic palette. Interior points (iteration count >= max_iter) map to black. The palette follows a classic fractal color scheme: dark blue, cyan, yellow, orange, red/brown, looping every 256 iterations.

### 5. Display Output (SPI Driver)

`sp2_spi_driver.v` continuously reads the display framebuffer and streams RGB565 pixels over SPI to the ST7789V3 LCD. It handles hardware reset, sleep-out, initialization commands, window setup, and pixel data. At 25 MHz SCK, a full 320x172 frame takes approximately 70 ms to transfer. See [spi-display-driver.md](spi-display-driver.md).

## Clock Domain

The entire design runs on a single 50 MHz clock derived directly from the PL crystal oscillator on pin M19. There are no clock-domain crossings.

```verilog
wire clk = clk_50m;
```

A 3-stage synchronizer conditions the external active-low reset button:

```verilog
reg [2:0] rst_sync;
always @(posedge clk or negedge rst_n_in) begin
    if (!rst_n_in) rst_sync <= 3'b000;
    else           rst_sync <= {rst_sync[1:0], 1'b1};
end
assign rst_n = rst_sync[2];
```

This ensures a clean, synchronous reset release regardless of button bounce timing. The design could be extended to a dual-clock architecture (e.g., 150 MHz compute, 50 MHz SPI) using an MMCM, but at 50 MHz the neuron pool already produces frames faster than SPI can display them.

## Memory Architecture

Two dual-port BRAMs serve as framebuffers, both sized to 65,536 entries (2^16) for clean BRAM inference even though only 55,040 pixels are used:

| Buffer | Width | Depth | Content | Port A (Write) | Port B (Read) |
|--------|-------|-------|---------|-----------------|---------------|
| `iter_fb` | 16-bit | 65536 | Iteration counts | Scheduler results | Colormap input |
| `disp_fb` | 16-bit | 65536 | RGB565 pixels | Colormap output | SPI driver |

Both are annotated with `(* ram_style = "block" *)` to force BRAM inference. The `iter_fb` buffer is addressed by pixel ID (assigned by the scheduler in row-major order), so results from neurons can arrive in any order and still land in the correct framebuffer location.

The colormap sits between the two framebuffers, continuously reading `iter_fb` at the address driven by the SPI driver and writing the resulting RGB565 into `disp_fb`. A 2-stage address pipeline compensates for the colormap's 1-cycle latency:

```verilog
always @(posedge clk) begin
    iter_fb_rd_addr <= fb_disp_addr;     // SPI driver address → iter FB
end

always @(posedge clk) begin
    color_wr_addr   <= iter_fb_rd_addr;  // Pipeline stage 1
    color_wr_addr_d <= color_wr_addr;    // Pipeline stage 2 (matches colormap latency)
    if (color_valid)
        disp_fb[color_wr_addr_d] <= color_rgb565;
end
```

This means the display framebuffer is continuously updated while the SPI driver reads it. There is no explicit double-buffering or frame synchronization between compute and display — the display simply shows whatever has been written so far. For a smooth zoom animation this works well: partial frames briefly show a mix of old and new data, but at the zoom speeds involved this is imperceptible.

## Module Hierarchy

```
mandelbrot_top
├── neuron_core × 18          (generate loop)
│   ├── fixed_mul (mul_a)     z_re * z_re
│   ├── fixed_mul (mul_b)     z_im * z_im
│   └── fixed_mul (mul_c)     z_re * z_im
├── pixel_scheduler × 1
├── colormap_lut × 1
└── sp2_spi_driver × 1
```

Plus the two BRAM framebuffers and the auto-zoom controller, both inlined in `mandelbrot_top.v`.

## Key Design Decisions

**Why "neurons"?** Each Mandelbrot iterator is an independent processing element with local state (z_re, z_im), data-dependent halting (escape detection), and no global synchronization — analogous to a simple recurrent neuron. The term reflects the project's origins in exploring neural-network-style parallelism on FPGA fabric.

**Why Q4.28 fixed-point?** The Zynq-7020's DSP48E1 slices have 25x18 signed multipliers with no floating-point support. Fixed-point maps directly to DSP hardware with deterministic latency. Q4.28 provides 4 integer bits (range ±8.0, sufficient for the Mandelbrot set's [-2, 2] domain) and 28 fractional bits (~3.7e-9 resolution). See [fixed-point-arithmetic.md](fixed-point-arithmetic.md).

**Why no double-buffering?** With 18 neurons at 50 MHz, a 256-iteration frame computes in roughly 49 ms — comparable to the ~70 ms SPI transfer time. Double-buffering would add 128 KB of BRAM (doubling the 46% utilization to 92%) for minimal visual benefit in a zoom animation where every frame is unique.

**Why 65536-deep framebuffers for 55040 pixels?** BRAM inference in Vivado works most reliably with power-of-two depths. The extra 10,496 entries waste no physical resources since the 36Kb BRAMs are allocated in whole blocks regardless.
