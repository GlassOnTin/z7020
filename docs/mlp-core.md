# MLP Core — SIREN Neural Inference Engine

`mlp_core.v` is a drop-in replacement for `neuron_core.v`. Same external interface, same scheduler protocol, same framebuffer writes — but instead of iterating z = z² + c, it evaluates a trained SIREN neural network that maps pixel coordinates to colors.

## What is a SIREN?

SIREN (Sinusoidal Representation Networks) use sin() as their activation function instead of ReLU. This produces smooth, continuous output — ideal for representing images, audio, and other signals as implicit functions.

The network takes (x, y, t) as input and produces (R, G, B) as output. Given a trained set of weights, every pixel coordinate maps to a unique color, and the time parameter `t` creates smooth animation. A 387-parameter network generates infinite unique frames from 1.5 KB of weights.

The key insight for FPGA: sin() can be computed with a small lookup table, while ReLU requires comparison logic per neuron. A 256-entry quarter-wave sine table serves all 18 cores.

## Architecture: Sequential MAC with BRAM Weights

### Why not parallel multipliers?

The Mandelbrot core uses 3 multipliers in parallel (12 DSPs each, 216 total). The first MLP implementation tried the same — 3 simultaneous weight×activation multiplies per core. This overflowed the LUT budget by 3,200 slices because each core needed 3 simultaneous read ports into its 387-entry weight array, and 18 copies of those mux trees couldn't fit.

### The sequential approach

The working design uses **1 multiplier per core** with BRAM-backed weight storage:

```
For each neuron j in current layer:
    acc = 0
    For each input k:
        Read weight[j][k] from BRAM    (1 cycle)
        Wait for BRAM read latency     (1 cycle)
        Multiply weight × activation   (3 cycles, pipelined)
        acc += product                  (1 cycle)
    Read bias[j] from BRAM              (2 cycles)
    acc += bias
    activation[j] = sin(acc)            (2 cycles, via LUT)
```

This trades throughput for density. One multiplier uses 4 DSPs (same as before per multiply, but only 1 instead of 3), and BRAM stores weights with zero LUT overhead.

### Resource comparison

| Resource | Mandelbrot (per core) | MLP (per core) | Notes |
|----------|----------------------|----------------|-------|
| DSP48E1 | 12 (3 × 4) | 4 (1 × 4) | Less DSP pressure |
| BRAM | 0 | 0.5 (512×32 weight ROM) | Fits in one RAMB18 |
| LUTs | ~300 | ~400 | Extra: accumulator, FSM, sin LUT |
| Registers | ~600 | ~900 | Extra: activation banks, layer tracking |

Total design (18 cores): 72 DSPs, 82 BRAM tiles, ~16K LUTs, ~28K registers.

## Network Geometry

```
Layer 0:  3 inputs  → 16 hidden   (48 weights + 16 biases =  64 params)
Layer 1:  16 hidden → 16 hidden   (256 weights + 16 biases = 272 params)
Layer 2:  16 hidden → 3 outputs   (48 weights + 3 biases =  51 params)
                                                    Total:  387 params
```

All parameters stored in a 512×32-bit BRAM per core, initialized at synthesis from `mlp_weights.vh`. The 125 unused entries are zero-padded.

### Weight memory layout

```
Address  Contents
-------  --------
  0-47   Layer 0: weights[0][0..2], weights[1][0..2], ..., weights[15][0..2]
 48-63   Layer 0: biases[0..15]
 64-319  Layer 1: weights[0][0..15], weights[1][0..15], ..., weights[15][0..15]
320-335  Layer 1: biases[0..15]
336-383  Layer 2: weights[0][0..15], weights[1][0..15], weights[2][0..15]
384-386  Layer 2: biases[0..2]
387-511  (unused, zero)
```

The address calculation for weight[neuron_j][input_k] in layer L is:

```verilog
w_addr = get_w_base(layer) + neuron * fan_in + k;
```

## FSM State Machine

```
S_IDLE ──► S_SETUP ──► S_W_READ ──► S_W_WAIT ──► S_MAC ──► S_MUL_WAIT ──► S_ACC ──┐
  ▲                        ▲                                                         │
  │                        └───────────── more inputs? ──────────────────────────────┘
  │                                            │ no
  │                    S_BIAS_RD ──► S_BIAS_WAIT ──► S_BIAS_ADD
  │                                                      │
  │                    S_ACTIVATE ──► S_ACT_WAIT ──► S_NEXT_N
  │                                                      │
  │                    S_NEXT_L ◄─── last neuron? ───────┘
  │                        │
  │                   more layers? ──► S_SETUP (next layer)
  │                        │ no
  └─── S_OUTPUT ◄──────────┘
```

15 states, ~616 cycles per pixel for the 3→16→16→3 network.

### Cycle budget breakdown

| Phase | Cycles | Notes |
|-------|--------|-------|
| Layer 0 (3→16) | 16 × (3×6 + 5) = 368 | 6 cycles per MAC, 5 for bias+sin |
| Layer 1 (16→16) | 16 × (16×6 + 5) = 1616 | Dominant cost |
| Layer 2 (16→3) | 3 × (16×6 + 5) = 303 | |
| Overhead | ~25 | Setup, output formatting |
| **Total** | **~616** | |

At 50 MHz with 18 cores: 50M / 616 × 18 = **1.46M pixels/sec → ~26 FPS** at 320×172.

## Activation Function: Quarter-Wave Sine LUT

`sine_lut.v` computes sin(x) for Q4.28 inputs using a 256-entry quarter-wave table.

### How it works

1. **Phase extraction**: Multiply input angle by 1/(2π) to get fractional turns. Extract 10 bits: 2 for quadrant, 8 for table index.
2. **Quadrant symmetry**:
   - Q0 [0, π/2): table[idx]
   - Q1 [π/2, π): table[255-idx]
   - Q2 [π, 3π/2): -table[idx]
   - Q3 [3π/2, 2π): -table[255-idx]
3. **Output**: 2-cycle latency (1 for phase computation + 1 for table read and sign).

The phase multiplication uses the reciprocal constant:
```
1/(2π) in Q4.28 = 0x028B_E60D
```

This avoids division entirely — a single fixed-point multiply followed by bit extraction gives the 10-bit phase index.

### Activation register banks

Each core has two banks of 16 × 32-bit registers (`act_a` and `act_b`). They swap roles each layer:

```
Layer 0: Read from act_a (inputs x,y,t), write to act_b (hidden activations)
Layer 1: Read from act_b, write to act_a
Layer 2: Read from act_a, write to out_r/out_g/out_b
```

The `bank_sel` signal toggles at each layer boundary. This avoids copying activation values between layers.

## Input Mapping

The scheduler provides Q4.28 coordinates via `c_re` and `c_im` (reusing the Mandelbrot interface). The MLP core interprets these as spatial coordinates:

```verilog
act_a[0] <= c_re;       // x: normalized to [-1, +1]
act_a[1] <= c_im;       // y: normalized to [-1, +1]
act_a[2] <= time_val;   // t: derived from frame counter
```

The scheduler doesn't know or care that these are neural network inputs vs. complex plane coordinates — it generates the same linear sweep of (x, y) values.

### Time parameter

The `max_iter` input (unused in MLP mode for its original purpose) carries a frame counter. The core converts it to a Q4.28 time value:

```verilog
wire signed [WIDTH-1:0] time_val = {16'b0, max_iter} << 22;
```

This gives 0.015625 per frame. At ~26 FPS, a full 4π cycle (12.57 in Q4.28) takes ~31 seconds, producing slow, smooth animation.

## Output: RGB565 Packing

The output layer produces three Q4.28 values in [-1, +1] via sin() activation. These are packed into 16-bit RGB565:

```
1. Shift from [-1, +1] to [0, 2): add 1.0 (0x10000000)
2. Clamp: negative → 0, ≥ 2.0 → max
3. Extract bits:
   R (5 bits): [28:24] of shifted value
   G (6 bits): [28:23] of shifted value
   B (5 bits): [28:24] of shifted value
```

**Critical**: The extraction must include bit 28, which is the 1.0 position in Q4.28. See the [bit extraction bug](#the-rgb565-bit-extraction-bug) section in lessons-learned.md for why `[27:23]` is wrong.

### Why sin() on the output layer?

SIREN papers typically use a linear or tanh() output. We use sin() on all layers because:

1. **Hardware simplicity**: The same sine LUT handles every layer, no special-casing
2. **Range guarantee**: sin() always returns [-1, +1], which maps cleanly to [0, 2) after the +1.0 shift
3. **Training match**: The Python training script uses `torch.sin()` on the output, so the weight values are trained for this activation

If the training uses one activation and the hardware uses another, the output will be wrong — potentially producing the same "thresholded" appearance as having no activation at all.

## Interface Contract

The MLP core implements the same handshake as `neuron_core`:

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| pixel_valid | in | 1 | Scheduler has a pixel ready |
| pixel_ready | out | 1 | Core is idle (S_IDLE) |
| c_re, c_im | in | 32 | Coordinates (Q4.28) |
| pixel_id | in | 16 | Pixel index for framebuffer write |
| max_iter | in | 16 | Frame counter (reinterpreted as time) |
| result_valid | out | 1 | One-cycle pulse: result ready |
| result_pixel_id | out | 16 | Echo of pixel_id |
| result_iter | out | 16 | RGB565 color (not iteration count) |

The scheduler, framebuffer, and display pipeline are completely unaware that the core is running a neural network instead of a Mandelbrot iteration.
