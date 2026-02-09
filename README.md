# Neural Mandelbrot Explorer — FPGA Accelerated Fractal Renderer

A real-time Mandelbrot set explorer running on a Xilinx Zynq-7020 FPGA, rendering to a 320x172 ST7789V3 LCD over SPI. Eighteen parallel "neuron" cores iterate z = z² + c simultaneously, achieving over 200 million iterations per second on pure programmable logic.

```
                ┌─────────────────────────────────────────────┐
                │              mandelbrot_top                  │
                │                                             │
                │  ┌───────────┐   ┌──────────────────────┐  │
                │  │  Auto-Zoom │──▶│   Pixel Scheduler    │  │
                │  │ Controller │   │   (coord gen +       │  │
                │  └───────────┘   │    work dispatch)     │  │
                │                  └──────┬───────────┘    │  │
                │            ┌────────────┼────────────┐   │  │
                │            ▼            ▼            ▼   │  │
                │      ┌──────────┐ ┌──────────┐    ┌──────────┐
                │      │ Neuron 0 │ │ Neuron 1 │ …  │Neuron 17 │
                │      │ z=z²+c   │ │ z=z²+c   │    │ z=z²+c   │
                │      │ (3 DSPs) │ │ (3 DSPs) │    │ (3 DSPs) │
                │      └────┬─────┘ └────┬─────┘    └────┬─────┘
                │           └────────────┼───────────────┘│
                │                        ▼                │
                │  ┌────────────┐  ┌───────────┐  ┌────────────┐
                │  │ Iter FB    │──│ Colormap  │──│ Display FB │
                │  │ (BRAM)     │  │ (LUT)     │  │ (BRAM)     │
                │  └────────────┘  └───────────┘  └─────┬──────┘
                │                                       │
                │                              ┌────────┴───────┐
                │                              │ SPI Driver     │
                │                              │ (25 MHz SCK)   │
                │                              └────────┬───────┘
                └───────────────────────────────────────┼───────┘
                                                        │ SPI
                                                   ┌────┴────┐
                                                   │ ST7789  │
                                                   │ 320x172 │
                                                   └─────────┘
```

## Features

- **18 parallel neuron cores** — each running z = z² + c with 3 pipelined multipliers (216 of 220 DSP48E1 slices)
- **Q4.28 fixed-point arithmetic** — 32-bit signed, ~3.7 ns resolution, ~156,000x zoom before precision exhaustion
- **25 MHz SPI** to ST7789V3 LCD — continuous frame streaming at ~14 FPS display refresh
- **Auto-zoom screensaver** — smooth zoom into the seahorse valley, automatically loops when precision is exhausted
- **Single 50 MHz clock domain** — no clock-domain crossing complexity
- **Dual-port BRAM framebuffers** — separate compute and display paths, zero tearing

## Resource Usage (XC7Z020CLG484-1)

| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| LUT      | ~8,500 | 53,200 | ~16% |
| Register | ~14,000 | 106,400 | ~13% |
| BRAM (36Kb) | 64.5 | 140 | 46% |
| DSP48E1  | 216 | 220 | 98% |

## Quick Start

### Prerequisites

- [Vivado 2025.2](https://www.xilinx.com/support/download.html) (ML Standard edition, Zynq-7000 device support) — see [INSTALL.md](INSTALL.md)
- Hello-FPGA Smart ZYNQ SP board (XC7Z020CLG484-1)
- JTAG connection (direct USB or via USB/IP — see [fpga-jtag.sh](fpga-jtag.sh))

### Build

```bash
source env.sh                                           # Load Vivado into PATH
vivado -mode batch -source vivado/run_all.tcl           # Synthesize + implement + bitstream
```

### Program

```bash
./fpga-jtag.sh attach                                   # Connect JTAG (if using USB/IP)
vivado -mode batch -source vivado/program.tcl           # Program FPGA
```

The display should show the Mandelbrot set and begin auto-zooming into the seahorse valley. LED1 toggles per compute frame; LED2 blinks at 1 Hz as a heartbeat.

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | System block diagram, data flow, module hierarchy |
| [Fixed-Point Arithmetic](docs/fixed-point-arithmetic.md) | Q4.28 representation, DSP48E1 multiply decomposition |
| [Neuron Core](docs/neuron-core.md) | The Mandelbrot iterator: FSM, pipeline, escape detection |
| [Parallel Scheduling](docs/parallel-scheduling.md) | Work distribution, result collection, frame completion |
| [SPI Display Driver](docs/spi-display-driver.md) | ST7789V3 interface, Mode 0 timing, init sequence |
| [Auto-Zoom](docs/auto-zoom.md) | Viewport control, zoom arithmetic, screensaver loop |
| [Build & Program](docs/build-and-program.md) | Toolchain setup, pin mapping, build flow |
| [Lessons Learned](docs/lessons-learned.md) | Pitfalls, debugging war stories, Vivado quirks |

## Project Structure

```
z7020/
├── rtl/
│   ├── mandelbrot_top.v      # Top-level: clock, framebuffers, auto-zoom, instantiation
│   ├── neuron_core.v         # Mandelbrot z=z²+c iterator (3 multipliers)
│   ├── fixed_mul.v           # Q4.28 signed fixed-point multiply (3-stage pipeline)
│   ├── pixel_scheduler.v     # Coordinate generation + neuron pool management
│   ├── colormap_lut.v        # Iteration count → RGB565 color palette
│   └── sp2_spi_driver.v      # ST7789V3 SPI display controller
├── constraints/
│   └── z7020_sp.xdc          # Pin assignments and timing constraints
├── vivado/
│   ├── create_project.tcl    # Project creation script
│   ├── run_all.tcl           # Full build: synth → impl → bitstream
│   └── program.tcl           # JTAG programming script
├── sim/
│   ├── tb_neuron_core.v      # Neuron core testbench
│   ├── tb_fixed_mul.v        # Fixed-point multiplier testbench
│   ├── Makefile              # Icarus Verilog simulation flow
│   └── mandelbrot_ref.py     # Python reference implementation
├── docs/                     # Deep-dive documentation (see table above)
├── env.sh                    # Vivado environment setup
├── fpga-jtag.sh              # USB/IP JTAG management
└── INSTALL.md                # Vivado installation notes
```

## License

MIT
