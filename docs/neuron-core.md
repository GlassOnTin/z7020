# Neuron Core — The Mandelbrot Iterator

Each `neuron_core` is an independent processing element that computes the Mandelbrot iteration z_{n+1} = z_n² + c for a single pixel. Eighteen of these run in parallel, forming the compute engine of the design.

## The Mandelbrot Recurrence

Given a point c = c_re + c_im·i in the complex plane, the Mandelbrot iteration is:

```
  z₀ = 0
  z_{n+1} = z_n² + c
```

Expanding into real and imaginary components:

```
  z_re_new = z_re² - z_im² + c_re
  z_im_new = 2 · z_re · z_im + c_im
```

The point c is in the Mandelbrot set if this sequence remains bounded (|z| ≤ 2) for all n. In practice we iterate until either:

- **Escape**: |z|² ≥ 4.0 (the point has escaped; it's outside the set)
- **Max iterations reached**: the point is considered inside the set

The iteration count at escape determines the color assigned to the pixel.

## Three Parallel Multipliers

Each iteration requires three multiplies:

| Multiplier | Operands | Result | Used for |
|------------|----------|--------|----------|
| mul_a | z_re × z_re | z_re² | Real update, magnitude check |
| mul_b | z_im × z_im | z_im² | Real update, magnitude check |
| mul_c | z_re × z_im | z_re_im | Imaginary update (×2) |

All three run simultaneously in a single `fixed_mul` instance each (3-stage pipeline, 4 DSP48E1 slices per multiplier). See [fixed-point-arithmetic.md](fixed-point-arithmetic.md) for multiply details.

After the multiplies complete, the update equations are pure combinational:

```verilog
// z_re_new = z_re² - z_im² + c_re
assign z_re_new = z_re_sq - z_im_sq + c_re_r;

// z_im_new = 2 · z_re · z_im + c_im  (shift left by 1 = multiply by 2)
assign z_im_new = {z_re_im[WIDTH-2:0], 1'b0} + c_im_r;

// |z|² = z_re² + z_im²  (for escape check)
assign mag_sq = z_re_sq + z_im_sq;
```

## FSM: State Machine

The neuron operates as a 5-state FSM:

```
          pixel_valid
              │
              ▼
  ┌──────── S_IDLE ◀────────────────────────┐
  │           │                              │
  │     latch c, pid                    result_valid
  │     z = 0, iter = 0                     │
  │           │                              │
  │           ▼                              │
  │        S_LOAD                            │
  │           │                              │
  │     launch 3 multiplies                  │
  │           │                              │
  │           ▼                              │
  │        S_MUL ◀──────┐                   │
  │           │          │                   │
  │     wait 3 cycles    │                   │
  │           │          │                   │
  │           ▼          │                   │
  │       S_UPDATE       │                   │
  │        │     │       │                   │
  │  escaped?  max?      │                   │
  │   no──┘     │        │                   │
  │   update z  │  re-launch muls            │
  │   iter++    │        │                   │
  │   ─────────────▶─────┘                   │
  │             │                            │
  │        yes (escaped or max)              │
  │             │                            │
  │        emit result ─────────────────────┘
  └──────────────────────────────────────────┘
```

### State Details

**S_IDLE**: Neuron is available. `pixel_ready` is asserted. When `pixel_valid` is received, the neuron latches the input coordinates (c_re, c_im), pixel ID, and initializes z = 0, iter = 0.

**S_LOAD**: Loads the multiplier inputs with current z values and asserts `mul_valid_in`. Transitions to S_MUL.

**S_MUL**: Waits 3 clock cycles for the multiply pipeline to complete (tracked by `pipe_cnt`).

**S_UPDATE**: Multiply results are available. Checks the escape condition. If escaped or max_iter reached, emits a one-cycle `result_valid` pulse and returns to S_IDLE. Otherwise, updates z_re and z_im with the new values, increments iter, immediately relaunches the multipliers, and goes back to S_MUL.

### Iteration Throughput

The loop S_LOAD → S_MUL(3 cycles) → S_UPDATE takes 4 clock cycles per iteration (S_UPDATE immediately relaunches the multipliers for subsequent iterations, making S_LOAD effectively free after the first iteration):

| Cycle | Stage | Action |
|-------|-------|--------|
| 0 | S_LOAD / S_UPDATE | Launch multiplies with current z |
| 1 | S_MUL | Multiply pipeline stage 1 |
| 2 | S_MUL | Multiply pipeline stage 2 |
| 3 | S_MUL | Multiply pipeline stage 3 (pipe_cnt == 2) |
| 4 | S_UPDATE | Results available; update z or emit result |

At 50 MHz: **12.5 million iterations per second per neuron**.

With 18 neurons: **225 million iterations per second** aggregate.

## Escape Detection

The escape condition is more involved than a simple magnitude check because fixed-point overflow must be handled:

```verilog
wire z_re_overflow = (z_re[WIDTH-1] != z_re[WIDTH-2]);
wire z_im_overflow = (z_im[WIDTH-1] != z_im[WIDTH-2]);

assign escaped = (mag_sq[WIDTH-1])               // Negative mag_sq → overflow
              || (mag_sq >= ESCAPE_THRESHOLD)      // |z|² ≥ 4.0
              || z_re_overflow                     // z_re exceeds ±4.0
              || z_im_overflow;                    // z_im exceeds ±4.0
```

Three cases:

1. **Standard escape**: `mag_sq >= 4.0` in Q4.28 representation (`0x4000_0000`)
2. **Magnitude overflow**: If `z_re² + z_im²` overflows the 32-bit signed range (goes negative), the point has definitely escaped
3. **Component overflow**: If z_re or z_im individually exceed ±4.0 (sign bit disagrees with the MSB of the integer part), the subsequent squaring could overflow the Q4.28 range and produce incorrect mag_sq values — so we catch it early with a single-bit sign consistency check

## Handshake Protocol

The neuron uses a valid/ready handshake for pixel assignment:

```
  Scheduler                    Neuron
     │                           │
     │  pixel_valid=1            │
     │  c_re, c_im, pixel_id    │
     │──────────────────────────▶│
     │                           │ pixel_ready=1 (S_IDLE)
     │                           │ Latches inputs, starts computing
     │                           │ pixel_ready=0 (busy)
     │           ...             │
     │                           │ result_valid=1 (one cycle)
     │◀──────────────────────────│ result_pixel_id, result_iter
     │                           │
     │                           │ pixel_ready=1 (S_IDLE again)
```

Key properties:
- `pixel_ready` is high only in S_IDLE
- `pixel_valid` is asserted by the scheduler for exactly one cycle per assignment
- `result_valid` is a one-cycle pulse — the scheduler must capture it immediately
- Multiple neurons can emit `result_valid` on the same clock cycle (the scheduler handles this via a pending buffer; see [parallel-scheduling.md](parallel-scheduling.md))

## Resource Usage Per Neuron

| Resource | Count | Notes |
|----------|-------|-------|
| DSP48E1 | 12 | 3 multipliers × 4 DSPs each |
| Registers | ~600 | State, z_re, z_im, c_re, c_im, pipeline regs |
| LUTs | ~300 | FSM, adders, escape detection |
