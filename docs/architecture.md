# System Architecture

This document describes the overall architecture of the neural Mandelbrot explorer, covering the module hierarchy, data flow, memory architecture, and clock domain. The design supports two compute modes selectable at synthesis time via the `COMPUTE_MODE` parameter.

## Dual-Mode Architecture

The top-level `mandelbrot_top.v` accepts a `COMPUTE_MODE` generic:

| Mode | Value | Core | Output | Use case |
|------|-------|------|--------|----------|
| Mandelbrot | 0 (default) | `neuron_core` | Iteration count → colormap → RGB565 | Fractal rendering |
| MLP/SIREN | 1 | `mlp_core` | RGB565 directly from network output | Neural generative art |

Both modes share the same scheduler, framebuffer, SPI display driver, and handshake protocol. The build command selects the mode:

```bash
# Mandelbrot mode (default)
vivado -mode batch -source vivado/run_all.tcl

# MLP mode
vivado -mode batch -source vivado/run_all.tcl -tclargs 1
```

## Block Diagram

The design is a single-clock datapath that generates images and streams them to an SPI-connected LCD:

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

The pipeline processes one frame in six stages:

### 1. Viewport Setup (Auto-Zoom Controller)

The auto-zoom controller in `mandelbrot_top.v` provides viewport parameters to the scheduler: `zoom_cre_start`, `zoom_cim_start`, and `zoom_step`. After each compute frame completes, it shrinks the step by factor 63/64 (a right-shift subtraction) and recomputes the viewport origin to keep the target centered. See [auto-zoom.md](auto-zoom.md).

### 2. Coordinate Generation and Work Distribution (Pixel Scheduler)

`pixel_scheduler.v` generates complex-plane coordinates for each of the 55,040 pixels (320 x 172) via row-major accumulation. It maintains the current coordinate `(cur_c_re, cur_c_im)` and advances by `c_re_step` per pixel horizontally and `c_im_step` per row. A priority encoder scans the neuron pool's ready signals to find an idle neuron, then asserts `neuron_valid[i]` with the coordinate and pixel ID on a shared bus. See [parallel-scheduling.md](parallel-scheduling.md).

### 3. Mandelbrot Iteration (Neuron Pool)

Eighteen `neuron_core` instances iterate z = z² + c independently. Each neuron has three pipelined fixed-point multipliers running in parallel (z_re², z_im², z_re×z_im), giving a throughput of one iteration per 4 clock cycles. When a neuron escapes or reaches `max_iter`, it emits a one-cycle `result_valid` pulse with the pixel ID and final iteration count. See [neuron-core.md](neuron-core.md).

### 4. Color Sweep

After compute completes, a sweep state machine reads all 55,040 entries from `iter_fb` sequentially through the colormap and writes the resulting RGB565 values into the back display framebuffer. This takes ~1.1 ms (55,040 + 2 flush cycles at 50 MHz). The sweep is gated by `sweep_wr_v1` to avoid writing stale addresses during the first 2 pipeline-fill cycles.

### 5. Buffer Swap

When the sweep finishes, a swap is armed (`swap_pending`). The actual buffer swap occurs at the next SPI frame boundary (`disp_frame_done`), toggling `disp_buf_sel`. This ensures the SPI driver always reads a complete, consistent frame. The auto-zoom controller then updates the viewport and pulses `frame_start` for the next compute cycle.

### 6. Display Output (SPI Driver)

`sp2_spi_driver.v` continuously reads the display framebuffer and streams RGB565 pixels over SPI to the ST7789V3 LCD. It handles hardware reset, sleep-out, initialization commands, window setup, and pixel data. At 25 MHz SCK, a full 320x172 frame takes approximately 40 ms to transfer (~25 FPS). See [spi-display-driver.md](spi-display-driver.md).

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

Three dual-port BRAMs serve as framebuffers, all sized to 65,536 entries (2^16) for clean BRAM inference even though only 55,040 pixels are used:

| Buffer | Width | Depth | Content | Port A (Write) | Port B (Read) |
|--------|-------|-------|---------|-----------------|---------------|
| `iter_fb` | 16-bit | 65536 | Iteration counts | Scheduler results | Colormap sweep |
| `disp_fb_a` | 16-bit | 65536 | RGB565 pixels | Colormap (when back) | SPI driver (when front) |
| `disp_fb_b` | 16-bit | 65536 | RGB565 pixels | Colormap (when back) | SPI driver (when front) |

All are annotated with `(* ram_style = "block" *)` to force BRAM inference. The `iter_fb` buffer is addressed by pixel ID (assigned by the scheduler in row-major order), so results from neurons can arrive in any order and still land in the correct framebuffer location.

### Double-Buffered Display

The two display framebuffers (`disp_fb_a`, `disp_fb_b`) form a double-buffer pair controlled by `disp_buf_sel`:

- When `disp_buf_sel = 0`: SPI reads from A (front), colormap writes to B (back)
- When `disp_buf_sel = 1`: SPI reads from B (front), colormap writes to A (back)

After each compute frame completes, a **color sweep** reads all 55,040 entries from `iter_fb` through the colormap and writes the resulting RGB565 values into the back display buffer. When the sweep finishes (`sweep_done`), a swap is armed. The actual buffer swap occurs at the next SPI frame boundary (`disp_frame_done`), ensuring the display never shows a partially-written frame.

The colormap write pipeline uses a 2-stage address delay to compensate for the BRAM read latency and colormap lookup latency:

```verilog
always @(posedge clk) begin
    color_wr_addr   <= iter_fb_rd_addr;  // Pipeline stage 1
    color_wr_addr_d <= color_wr_addr;    // Pipeline stage 2 (matches colormap latency)
end

wire wr_en_a = color_wr_en &  disp_buf_sel;  // buf_sel=1 → write to A (back)
wire wr_en_b = color_wr_en & ~disp_buf_sel;  // buf_sel=0 → write to B (back)
```

A delayed `sweep_wr_v1` signal gates writes so the first 2 cycles of the sweep (which carry stale addresses from the display path) don't corrupt the back buffer.

## Module Hierarchy

### Mode 0: Mandelbrot

```
mandelbrot_top (COMPUTE_MODE=0)
├── neuron_core × 18          (generate loop)
│   ├── fixed_mul (mul_a)     z_re * z_re
│   ├── fixed_mul (mul_b)     z_im * z_im
│   └── fixed_mul (mul_c)     z_re * z_im
├── pixel_scheduler × 1
├── colormap_lut × 1          iter → RGB565
├── sp2_spi_driver × 1
└── boot_msg × 1              UART "MANDELBROT QSPI OK\r\n"
```

Plus `iter_fb` (iteration counts), `disp_fb_a`/`disp_fb_b` (double-buffered RGB565), the color sweep state machine, and the auto-zoom controller.

### Mode 1: MLP/SIREN

```
mandelbrot_top (COMPUTE_MODE=1)
├── mlp_core × 18             (generate loop)
│   ├── fixed_mul (u_mac)     sequential MAC
│   ├── sine_lut (u_sine)     sin() activation
│   └── weight_mem [0:511]    BRAM weights (from mlp_weights.vh)
├── pixel_scheduler × 1       (same as Mode 0)
├── sp2_spi_driver × 1        (same as Mode 0)
└── boot_msg × 1
```

In MLP mode, `iter_fb` and `colormap_lut` are bypassed. The cores write RGB565 directly to double-buffered display framebuffers (`disp_fb_0`/`disp_fb_1`). The auto-zoom controller is replaced by a simple frame counter that increments `max_iter` each frame (interpreted as time by the MLP cores).

See [mlp-core.md](mlp-core.md) for details on the MLP inference engine, and [siren-training.md](siren-training.md) for the training and deployment pipeline.

## Key Design Decisions

**Why "neurons"?** Each Mandelbrot iterator is an independent processing element with local state (z_re, z_im), data-dependent halting (escape detection), and no global synchronization — analogous to a simple recurrent neuron. The term reflects the project's origins in exploring neural-network-style parallelism on FPGA fabric.

**Why Q4.28 fixed-point?** The Zynq-7020's DSP48E1 slices have 25x18 signed multipliers with no floating-point support. Fixed-point maps directly to DSP hardware with deterministic latency. Q4.28 provides 4 integer bits (range ±8.0, sufficient for the Mandelbrot set's [-2, 2] domain) and 28 fractional bits (~3.7e-9 resolution). See [fixed-point-arithmetic.md](fixed-point-arithmetic.md).

**Why double-buffer the display?** Compute frames finish in ~11 ms but the SPI transfer takes ~40 ms. Without double-buffering, the colormap would overwrite display pixels mid-transfer, causing tearing. The two display BRAMs (`disp_fb_a`, `disp_fb_b`) swap at SPI frame boundaries so the display always reads a complete frame. The cost is ~128 KB of additional BRAM (raising utilization from ~30% to ~46%).

**Why 65536-deep framebuffers for 55040 pixels?** BRAM inference in Vivado works most reliably with power-of-two depths. The extra 10,496 entries waste no physical resources since the 36Kb BRAMs are allocated in whole blocks regardless.
