# Z7020 — Parallel Compute Engine on Zynq-7020 FPGA

A reconfigurable parallel compute engine running on a Xilinx Zynq-7020 FPGA, rendering to a 320x172 ST7789V3 LCD over SPI. Eighteen parallel cores share a common scheduler, framebuffer, and display pipeline. The `COMPUTE_MODE` parameter selects what the cores compute:

- **Mode 0 (Mandelbrot):** Each core iterates z = z² + c with 3 pipelined multipliers. Auto-zooms into the seahorse valley.
- **Mode 1 (MLP/SIREN):** Each core runs a small neural network (3→32→32→3 SIREN) via pipelined MAC with BRAM weight storage. Generates animated patterns from trained implicit neural representations — including [Bad Apple](https://en.wikipedia.org/wiki/Bad_Apple!!) video playback from 658 SIREN segments (3.2 MB total weights).

```
                ┌─────────────────────────────────────────────┐
                │              mandelbrot_top                  │
                │                                             │
                │  ┌───────────┐   ┌──────────────────────┐  │
                │  │  Frame    │──▶│   Pixel Scheduler    │  │
                │  │ Controller│   │   (coord gen +       │  │
                │  └───────────┘   │    work dispatch)     │  │
                │                  └──────┬───────────┘    │  │
                │            ┌────────────┼────────────┐   │  │
                │            ▼            ▼            ▼   │  │
                │      ┌──────────┐ ┌──────────┐    ┌──────────┐
                │      │  Core 0  │ │  Core 1  │ …  │ Core 17  │
                │      │ mode 0:  │ │ mode 0:  │    │ mode 0:  │
                │      │  z=z²+c  │ │  z=z²+c  │    │  z=z²+c  │
                │      │ mode 1:  │ │ mode 1:  │    │ mode 1:  │
                │      │  SIREN   │ │  SIREN   │    │  SIREN   │
                │      └────┬─────┘ └────┬─────┘    └────┬─────┘
                │           └────────────┼───────────────┘│
                │                        ▼                │
                │               ┌────────────────┐  ┌────────────┐
                │  Mode 0 only: │ Iter FB → Cmap │  │ Display FB │
                │               └────────────────┘  │ (BRAM)     │
                │  Mode 1: RGB565 direct ──────────▶│            │
                │                                   └─────┬──────┘
                │                              ┌──────────┴──────┐
                │                              │ SPI Driver      │
                │                              │ (25 MHz SCK)    │
                │                              └──────────┬──────┘
                └─────────────────────────────────────────┼──────┘
                                                          │ SPI
                                                     ┌────┴────┐
                                                     │ ST7789  │
                                                     │ 320x172 │
                                                     └─────────┘
```

![Board running Mandelbrot zoom](docs/board_front_mandelbrot_running.jpg)
*Smart ZYNQ SP board running the auto-zoom demo on the 1.47" ST7789V3 display*

![Carrier board back with pinout](docs/board_back_carrier_pinout.jpg)
*Carrier board back showing Bank 33/35 GPIO headers and pin silkscreen*

## Features

- **18 parallel compute cores** — configurable at synthesis time via `COMPUTE_MODE`
- **Mode 0 (Mandelbrot):** z = z² + c iteration, 3 pipelined multipliers per core, auto-zoom screensaver
- **Mode 1 (SIREN):** 3→32→32→3 neural network, pipelined MAC with BRAM weights, sin() activation via quarter-wave LUT
- **Q4.28 fixed-point arithmetic** — 32-bit signed, shared across both modes
- **25 MHz SPI** to ST7789V3 LCD — continuous frame streaming at ~25 FPS display refresh
- **Work-stealing scheduler** — dispatches pixels to idle cores, collects results out of order
- **Single 50 MHz clock domain** — no clock-domain crossing complexity
- **Dual-port BRAM framebuffers** — separate compute and display paths

## Resource Usage (XC7Z020CLG484-1)

| Resource | Mode 0 (Mandelbrot) | Mode 1 (Bad Apple) | Available |
|----------|---------------------|--------------------|-----------|
| LUT      | 20,579 (39%)        | 23,200 (44%)       | 53,200    |
| Register | 37,558 (35%)        | 47,021 (44%)       | 106,400   |
| BRAM     | 109 (78%)           | 136 (97%)          | 140       |
| DSP48E1  | 180 (82%)           | 180 (82%)          | 220       |
| Slice    | 12,897 (97%)        | 13,227 (99%)       | 13,300    |

Mode 1 numbers include the Zynq PS subsystem and AXI interconnect for weight loading from SD card.

## Quick Start

### Prerequisites

- [Vivado 2025.2](https://www.xilinx.com/support/download.html) (ML Standard edition, Zynq-7000 device support) — see [INSTALL.md](INSTALL.md)
- Hello-FPGA Smart ZYNQ SP board (XC7Z020CLG484-1)
- JTAG connection (direct USB or via USB/IP — see [fpga-jtag.sh](fpga-jtag.sh))

### Build

```bash
source env.sh                                           # Load Vivado into PATH
vivado -mode batch -source vivado/run_all.tcl           # Mode 0: Mandelbrot (default)
vivado -mode batch -source vivado/run_all.tcl -tclargs 1  # Mode 1: SIREN neural inference
```

### Train SIREN weights (Mode 1)

**Procedural patterns:**
```bash
pip install torch numpy pillow
python3 scripts/train_siren.py --pattern plasma --hidden 32 --epochs 10000 --lr 0.0005
# Generates rtl/mlp_weights.vh, then rebuild with COMPUTE_MODE=1
```

**Bad Apple video (658 segments, GPU recommended):**
```bash
# Extract frames from source video
ffmpeg -i bad_apple/source.mp4 -vf "scale=320:172" bad_apple/frames/frame_%05d.png

# Train all segments (Colab notebook or local GPU)
# See bad_apple/train_h32_colab.ipynb for batched GPU training
# Or: python3 scripts/train_colab_batched.py --epochs 5000
```

### Program

```bash
./fpga-jtag.sh attach                                   # Connect JTAG (if using USB/IP)
vivado -mode batch -source vivado/program.tcl           # Program FPGA
```

### SD Boot with U-Boot UMS

The board boots from SD card via FSBL → bitstream → U-Boot, then exposes the SD card as a USB mass storage device for easy firmware updates:

```bash
vivado -mode batch -source vivado/create_fsbl.tcl       # One-time: generate FSBL
cd vivado && bootgen -image boot.bif -arch zynq -o BOOT.bin -w
# Copy BOOT.bin to FAT32 SD card, set boot jumpers to SD, power cycle
```

**Deploy cycle** (with U-Boot UMS running):
```bash
sudo mount -o uid=1000,gid=1000 /dev/sdd1 /mnt
cp vivado/BOOT.bin /mnt/ && sync && sudo umount /mnt
# Power cycle board to boot new firmware
```

See [Build & Program](docs/build-and-program.md) for QSPI flash and JTAG options.

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
| [SIREN Training](docs/siren-training.md) | Training pipeline, Q4.28 quantization, weight export |
| [MLP Core](docs/mlp-core.md) | SIREN inference engine: FSM, cycle budget, activation LUT |
| [The Iteration Thesis](docs/iteration-thesis.md) | From fractal iteration to neural inference — what the architecture actually shares |

## Project Structure

```
z7020/
├── rtl/
│   ├── mandelbrot_top.v      # Top-level: COMPUTE_MODE mux, framebuffers, display pipeline
│   ├── neuron_core.v         # Mode 0: Mandelbrot z=z²+c iterator (3 multipliers)
│   ├── mlp_core.v            # Mode 1: SIREN MLP inference (pipelined MAC, BRAM weights)
│   ├── mlp_pl_wrapper.v      # AXI-Lite wrapper for PS→PL weight loading
│   ├── axi_mlp_ctrl.v        # AXI-Lite slave: weight/control registers
│   ├── sine_lut.v            # Quarter-wave sin() LUT for SIREN activation
│   ├── mlp_weights.vh        # Trained SIREN weights (generated by train_siren.py)
│   ├── fixed_mul.v           # Q4.28 signed fixed-point multiply (3-stage pipeline)
│   ├── pixel_scheduler.v     # Coordinate generation + core pool management
│   ├── colormap_lut.v        # Iteration count → RGB565 color palette (Mode 0 only)
│   ├── sp2_spi_driver.v      # ST7789V3 SPI display controller
│   ├── boot_msg.v            # UART boot message output
│   └── uart_tx.v             # Simple UART transmitter
├── bad_apple/
│   ├── source.mp4            # Bad Apple source video
│   └── train_h32_colab.ipynb # Colab notebook for H=32 batched GPU training
├── scripts/
│   ├── train_siren.py        # PyTorch SIREN training + Q4.28 weight export
│   ├── train_colab_batched.py # Batched training for Bad Apple (all segments on GPU)
│   ├── train_colab.py        # Colab-compatible single-segment trainer
│   ├── train_bad_apple.py    # Single-segment Bad Apple training with FPGA validation
│   ├── train_all_segments.sh # Shell script to train all 658 segments locally
│   ├── fpga_sim.py           # Bit-exact Python simulator of FPGA Q4.28 datapath
│   ├── sweep_topology.py     # Hidden width parameter sweep (H=8..32)
│   └── test_ba_*.py          # Bad Apple quality/capacity/threshold tests
├── constraints/
│   ├── z7020_sp.xdc          # PL pin assignments and timing constraints
│   └── z7020_sp_ps.xdc       # PS MIO pin assignments (USB0, UART0)
├── vivado/
│   ├── run_all.tcl           # Full build: synth → impl → bitstream (accepts COMPUTE_MODE)
│   ├── create_project.tcl    # Project creation script
│   ├── program.tcl           # JTAG programming script (volatile)
│   ├── create_fsbl.tcl       # Generate FSBL for boot image (one-time)
│   ├── deploy_sd.tcl         # General-purpose SD card deploy via UMS
│   ├── deploy_badapple.tcl   # Bad Apple weight + bitstream deploy
│   ├── boot.bif              # Boot image format descriptor
│   ├── flash.tcl             # QSPI flash programming via hardware manager
│   └── program_qspi.sh      # All-in-one: bootgen + flash
├── sim/
│   ├── tb_neuron_core.v      # Neuron core testbench
│   ├── tb_fixed_mul.v        # Fixed-point multiplier testbench
│   ├── tb_pixel_scheduler.v  # Pixel scheduler testbench
│   ├── Makefile              # Icarus Verilog simulation flow
│   └── mandelbrot_ref.py     # Python reference implementation
├── docs/                     # Deep-dive documentation (see table above)
├── env.sh                    # Vivado environment setup
├── fpga-jtag.sh              # USB/IP JTAG management
└── INSTALL.md                # Vivado installation notes
```

## License

MIT
