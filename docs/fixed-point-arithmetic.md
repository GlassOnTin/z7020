# Fixed-Point Arithmetic

The Mandelbrot computation requires multiplying complex numbers repeatedly. On the Zynq-7020, there are no floating-point units — just DSP48E1 slices with 25x18 signed integer multipliers. This document explains how we use Q4.28 fixed-point representation to map the Mandelbrot recurrence onto DSP hardware.

## Why Fixed-Point on FPGA

Floating-point arithmetic requires separate exponent handling, mantissa alignment, normalization, and rounding logic. An FPGA *can* implement this in LUT fabric, but:

- A 32-bit floating-point multiplier consumes ~500 LUTs and 3+ DSP slices
- Latency is 6-10 cycles depending on pipeline depth
- Resource usage scales poorly when you need 54 multipliers (18 neurons × 3 each)

Fixed-point eliminates all that overhead. A 32-bit signed multiply is just `a * b` — Vivado maps it directly to DSP48E1 slices with a 3-cycle pipelined latency and zero LUT overhead for the multiply itself.

## Q4.28 Representation

We use a **signed Q4.28** format: a 32-bit two's complement integer where the binary point sits 28 bits from the right.

```
  Bit 31 (sign)
    │
    S III.FFFF FFFF FFFF FFFF FFFF FFFF FFFF
    │ └─┘ └──────────────────────────────────┘
    │  3     28 fractional bits
    │  integer
    │  bits
    sign
```

| Property | Value |
|----------|-------|
| Total width | 32 bits |
| Integer bits | 4 (including sign) |
| Fractional bits | 28 |
| Range | [-8.0, +8.0) |
| Resolution | 2^(-28) ≈ 3.73 × 10^(-9) |

The range [-8, +8) is more than sufficient for the Mandelbrot set, which lies entirely within |c| ≤ 2 and where |z| ≤ 2 before escape.

### Conversion Examples

To convert a real number to Q4.28: multiply by 2^28 (268,435,456) and round to the nearest integer.

| Value | Calculation | Q4.28 (hex) |
|-------|------------|-------------|
| -2.0 | -2 × 2^28 = -536,870,912 | `0xE000_0000` |
| +1.0 | 1 × 2^28 = 268,435,456 | `0x1000_0000` |
| +4.0 | 4 × 2^28 = 1,073,741,824 | `0x4000_0000` |
| -0.745 | -0.745 × 2^28 = -199,965,082 | `0xF414_7AE1` |
| +0.113 | 0.113 × 2^28 = 30,333,427 | `0x01CF_DF3B` |
| 0.009375 | 0.009375 × 2^28 = 2,516,582 | `0x0026_6666` |

The last value (0.009375) is the default pixel step: 3.0 / 320 = the complex-plane distance between adjacent pixels for the initial full-set view.

## Fixed-Point Multiplication

Multiplying two Q4.28 numbers produces a Q8.56 result (64 bits). To get back to Q4.28, we discard the lower 28 bits and the upper 4 bits:

```
  a (Q4.28) × b (Q4.28) = product (Q8.56)

  product:  [63:60] [59:28] [27:0]
            overflow Q4.28   discarded
            check    result  (truncated)
```

In Verilog:

```verilog
reg signed [63:0] product;
product <= a_r * b_r;                         // Stage 2: DSP multiply
result  <= product[FRAC+WIDTH-1 : FRAC];      // Stage 3: extract [59:28]
```

The implementation in `fixed_mul.v` uses a 3-stage pipeline:

| Stage | Operation | Hardware |
|-------|-----------|----------|
| 1 | Register inputs `a_r`, `b_r` | Flip-flops |
| 2 | `product <= a_r * b_r` | DSP48E1 slices |
| 3 | `result <= product[59:28]` | Flip-flops |

## DSP48E1 Decomposition

Each DSP48E1 on the Zynq-7020 (7-series) contains a 25-bit × 18-bit signed multiplier. A 32×32 signed multiply exceeds this, so Vivado decomposes it into partial products:

```
  a = {a_hi[15:0], a_lo[15:0]}    (32-bit, split into two 16-bit halves)
  b = {b_hi[15:0], b_lo[15:0]}

  a × b = (a_hi × b_hi) << 32
        + (a_hi × b_lo) << 16
        + (a_lo × b_hi) << 16
        + (a_lo × b_lo)
```

Each partial product (up to 16×16 = 32-bit result) fits within the 25×18 DSP multiplier. With the DSP48E1's built-in accumulator and pre-adder, Vivado synthesizes this as **4 DSP48E1 slices** per 32×32 multiply.

Each neuron has 3 multipliers (z_re², z_im², z_re×z_im), so:

```
  DSPs per neuron:  3 multipliers × 4 DSPs = 12
  Total (18 neurons): 18 × 12 = 216 DSPs
  Available on XC7Z020: 220 DSPs
  Utilization: 98.2%
```

This is the resource bottleneck that limits the design to 18 neurons.

## The `2 × z_re × z_im` Term

The Mandelbrot imaginary update is:

```
  z_im_new = 2 × z_re × z_im + c_im
```

Multiplying by 2 in fixed-point is just a left shift by 1 bit. In the RTL, this avoids an extra DSP:

```verilog
assign z_im_new = {z_re_im[WIDTH-2:0], 1'b0} + c_im_r;
```

This concatenation takes bits [30:0] of the product and appends a zero — a free multiply-by-2 in hardware.

## Precision and Zoom Limits

With 28 fractional bits, the smallest representable step between pixels is 2^(-28) ≈ 3.73 × 10^(-9). The initial full-set view has a step of 3.0/320 ≈ 0.009375. The maximum zoom ratio before the step rounds to zero is:

```
  zoom_max = 0.009375 / 3.73e-9 ≈ 2,516,582 ≈ 2.5 million ×
```

In practice, the auto-zoom controller detects precision exhaustion when `step - (step >>> 6)` equals `step` — that is, when the 1/64 decrement rounds to zero. This happens somewhat earlier than the theoretical limit because the subtraction loses resolution before the step itself reaches the minimum representable value. The effective zoom before visible degradation is approximately **156,000×** (where block artifacts become noticeable).

To zoom deeper, the design could be extended to Q4.60 (64-bit) at the cost of more DSPs per multiply (a 64×64 multiply requires ~16 DSP48E1 slices), or a multi-precision iterative approach could be used.

## Escape Threshold

The standard escape condition |z|² > 4 uses the Q4.28 representation of 4.0:

```verilog
localparam signed [WIDTH-1:0] ESCAPE_THRESHOLD = 32'sh4000_0000;  // 4.0
```

The neuron checks three conditions:

```verilog
wire escaped = (mag_sq[WIDTH-1])                    // Negative → overflow (definitely escaped)
             || (mag_sq >= ESCAPE_THRESHOLD)         // Standard |z|² ≥ 4
             || z_re_overflow || z_im_overflow;      // Component overflow
```

The overflow checks catch cases where z_re or z_im exceed the Q4.28 range (±8.0) — the multiply itself would wrap around, producing incorrect mag_sq values if not caught early.
