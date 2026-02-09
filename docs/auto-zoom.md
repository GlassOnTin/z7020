# Auto-Zoom — Viewport Control and Screensaver Loop

The auto-zoom controller in `mandelbrot_top.v` drives a continuous zoom animation into the Mandelbrot set's "seahorse valley." After each compute frame completes, it shrinks the viewport by a factor of 63/64, re-centers on the target, and kicks off the next frame. When fixed-point precision is exhausted, it resets to the full-set view and starts over.

## Zoom Target

The zoom target is the seahorse valley at c = -0.745 + 0.113i, a region rich in spiral structures and fine detail:

```verilog
localparam signed [WIDTH-1:0] TARGET_RE = 32'shF414_7AE1;  // -0.745 in Q4.28
localparam signed [WIDTH-1:0] TARGET_IM = 32'sh01CF_DF3B;  // +0.113 in Q4.28
```

This was chosen because it produces visually interesting structures at every zoom level — spirals, filaments, and miniature copies of the full set appear in succession.

## Viewport Parameterization

The viewport is defined by three values:

| Parameter | Meaning |
|-----------|---------|
| `zoom_step` | Complex-plane distance between adjacent pixels |
| `zoom_cre_start` | Real coordinate of the leftmost pixel column |
| `zoom_cim_start` | Imaginary coordinate of the topmost pixel row |

The step applies uniformly in both axes (square pixels). The full viewport spans:

```
  Width  = 320 × zoom_step  (in complex-plane units)
  Height = 172 × zoom_step
```

## Zoom Arithmetic

### Step Shrink: × 63/64

Each frame, the step shrinks by 1/64 via a shift-subtract:

```verilog
wire signed [WIDTH-1:0] next_step = zoom_step - (zoom_step >>> 6);  // × 63/64
```

The arithmetic right shift `>>> 6` divides by 64 while preserving the sign bit. The subtraction `step - step/64 = step × 63/64 ≈ step × 0.984375`.

This is a compounding zoom — after N frames the step is:

```
  step(N) = step(0) × (63/64)^N
```

To reach 1000× zoom: (63/64)^N = 0.001, so N = -ln(0.001)/ln(64/63) ≈ 441 frames.

### Viewport Origin: Shift-Add Multiplication

After computing `next_step`, the viewport origin must be recomputed to keep the target centered:

```
  start_re = TARGET_RE - 160 × next_step    (160 = half of 320 pixels)
  start_im = TARGET_IM -  86 × next_step    ( 86 = half of 172 pixels)
```

Multiplying by 160 and 86 without a hardware multiplier uses shift-add decomposition:

```verilog
// 160 = 128 + 32
wire signed [WIDTH-1:0] half_h = (next_step <<< 7) + (next_step <<< 5);

// 86 = 64 + 16 + 4 + 2
wire signed [WIDTH-1:0] half_v = (next_step <<< 6) + (next_step <<< 4)
                               + (next_step <<< 2) + (next_step <<< 1);
```

These are purely combinational (just wiring with adders) — no DSPs consumed.

### Update Sequence

On each `frame_done`:

```verilog
if (frame_done_w && !frame_busy_w) begin
    if (next_step == zoom_step) begin
        // Precision exhausted — reset to full view
        zoom_step      <= DEFAULT_CRE_STEP;
        zoom_cre_start <= DEFAULT_CRE_START;
        zoom_cim_start <= DEFAULT_CIM_START;
        max_iter_w     <= DEFAULT_MAX_ITER;
    end else begin
        zoom_step      <= next_step;
        zoom_cre_start <= TARGET_RE - half_h;
        zoom_cim_start <= TARGET_IM - half_v;
        if (max_iter_w < 1024)
            max_iter_w <= max_iter_w + 1;
    end
    frame_start_r <= 1;
end
```

## Max Iteration Ramping

As zoom increases, more detail becomes visible and deeper iteration counts are needed to resolve fine structure. The controller ramps `max_iter` by +1 per frame, starting from 256 up to a cap of 1024:

```verilog
if (max_iter_w < 1024)
    max_iter_w <= max_iter_w + 1;
```

At ~441 frames to 1000× zoom, max_iter reaches 697 by that point — enough for the detail level at that magnification. The ramp is gradual to avoid sudden frame-time jumps.

## Precision Exhaustion Detection

The zoom loop terminates when the step becomes too small for the 1/64 decrement to change it:

```verilog
if (next_step == zoom_step) begin
    // step >>> 6 rounds to zero — can't zoom further
```

This happens when `zoom_step` is small enough that dividing by 64 (right-shifting by 6) produces zero — meaning `zoom_step < 64` in raw integer terms. Since `zoom_step` is Q4.28, this corresponds to:

```
  step < 64 × 2^(-28) = 2.38 × 10^(-7)
```

The initial step is 0.009375, giving a maximum zoom ratio of:

```
  0.009375 / 2.38e-7 ≈ 39,370×
```

In practice, visible block artifacts appear before this point (around 156,000× with the initial step, depending on viewport), because the step size becomes comparable to the pixel coordinate accumulation error.

## Screensaver Loop

When precision is exhausted, the controller snaps back to the default full-set view:

```verilog
zoom_step      <= DEFAULT_CRE_STEP;       // 0.009375
zoom_cre_start <= DEFAULT_CRE_START;       // -2.0
zoom_cim_start <= DEFAULT_CIM_START;       // -0.80625
max_iter_w     <= DEFAULT_MAX_ITER;        // 256
```

The next frame renders the full Mandelbrot set, and the zoom cycle begins again. This creates an endlessly looping screensaver.

## Startup

On reset, the controller sets the `startup` flag and loads default viewport parameters. The startup flag triggers a single `frame_start` pulse on the first clock after reset clears:

```verilog
if (!rst_n) begin
    startup        <= 1;
    zoom_step      <= DEFAULT_CRE_STEP;
    zoom_cre_start <= DEFAULT_CRE_START;
    zoom_cim_start <= DEFAULT_CIM_START;
    max_iter_w     <= DEFAULT_MAX_ITER;
end else begin
    if (startup) begin
        frame_start_r <= 1;
        startup       <= 0;
    end
end
```

## Default Viewport

The initial view shows the full Mandelbrot set:

| Parameter | Value | Q4.28 Hex | Notes |
|-----------|-------|-----------|-------|
| c_re_start | -2.0 | `0xE000_0000` | Left edge of complex plane |
| c_im_start | -0.80625 | `0xF319_999A` | Top edge (centered vertically) |
| step | 0.009375 | `0x0026_6666` | 3.0 / 320 pixels |
| max_iter | 256 | | Starting iteration depth |

The vertical range is 172 × 0.009375 = 1.6125, centered at 0, spanning [-0.80625, +0.80625] — providing an approximately square-pixel view of the set's main features.

## Timing

| Phase | Duration | Notes |
|-------|----------|-------|
| Compute frame (typical) | ~11 ms | 18 neurons, avg 45 iter/pixel |
| SPI display frame | ~70 ms | 55,040 pixels × 2 bytes × 25 MHz |
| Zoom overhead | 1 cycle | Shift-add, no multiplier |
| Frames to 1000× zoom | ~441 | (63/64)^441 ≈ 0.001 |
| Full zoom cycle | ~1000 frames | Reset around frame ~600-700 |
