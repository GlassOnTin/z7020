# Parallel Scheduling — Work Distribution and Result Collection

The `pixel_scheduler` module manages the pool of 18 neuron cores: it generates pixel coordinates, assigns work to idle neurons, collects results from completed neurons, and writes iteration counts to the framebuffer. This document covers the three subsystems — coordinate generation, neuron assignment, and result collection — along with the frame completion mechanism.

## Overview

```
  Viewport params                           Neuron pool (18 cores)
  (c_re_start, c_im_start, step)
         │                                ┌──── ready[0] ──┐
         ▼                                │                 │
  ┌──────────────┐    valid[i], c_re,     │  ┌──────────┐  │  result_valid[i]
  │  Coordinate  │    c_im, pixel_id      ├──│ Neuron 0 │──┤  pixel_id, iter
  │  Generator   │──────────────────────▶ │  └──────────┘  │
  │              │                        │  ┌──────────┐  │
  │  row-major   │                        ├──│ Neuron 1 │──┤
  │  accumulator │                        │  └──────────┘  │
  └──────────────┘                        │       ...      │
         │                                │  ┌──────────┐  │
         │ found_ready                    └──│ Neuron 17│──┘
         │                                   └──────────┘
         │                                        │
         │                                        ▼
         │                              ┌──────────────────┐
         │                              │  Result Pending  │
         │                              │  Buffer (18-bit) │
         │                              └────────┬─────────┘
         │                                       │ drain 1/cycle
         │                                       ▼
         │                              ┌──────────────────┐
         │                              │  Framebuffer     │
         │                              │  Write Port      │
         │                              │  (addr, data)    │
         │                              └──────────────────┘
```

## Coordinate Generation

Pixel coordinates are generated in row-major order using fixed-point accumulation. This avoids multiplication in the coordinate generator — only additions are needed.

The scheduler maintains:

| Register | Purpose |
|----------|---------|
| `px`, `py` | Current pixel position (0-319, 0-171) |
| `cur_c_re`, `cur_c_im` | Current complex coordinate |
| `row_c_re_start` | Real coordinate at the start of the current row |
| `pixel_count` | Total pixels assigned so far (used as pixel ID) |

Advancement logic:

```verilog
// Horizontal step (within a row)
if (px == H_RES - 1) begin
    px <= 0;
    if (py == V_RES - 1) begin
        all_assigned <= 1;               // Frame complete
    end else begin
        py <= py + 1;
        cur_c_im       <= cur_c_im + c_im_step;
        cur_c_re       <= row_c_re_start + c_re_step;  // First pixel of next row
    end
end else begin
    px <= px + 1;
    cur_c_re <= cur_c_re + c_re_step;    // Step right
end
```

Each pixel is identified by a 16-bit linear index (`pixel_count`), which becomes the framebuffer write address when the result arrives. This decouples the order in which neurons finish from the display order.

## Neuron Assignment — Priority Encoder

On every clock cycle when the frame is active and pixels remain unassigned, the scheduler searches for an idle neuron:

```verilog
integer k;
always @(*) begin
    found_ready   = 0;
    assign_neuron = 0;
    for (k = 0; k < N_NEURONS; k = k + 1) begin
        if (neuron_ready[k] && !result_pending[k] && !result_valid[k] && !found_ready) begin
            assign_neuron = k;
            found_ready   = 1;
        end
    end
end
```

This is a first-found priority encoder — it always picks the lowest-numbered idle neuron. The `!result_pending[k]` condition prevents assigning new work to a neuron whose previous result hasn't been drained to the framebuffer yet. The `!result_valid[k]` condition avoids assigning during the same cycle a result pulse fires.

When a ready neuron is found, the scheduler drives the shared coordinate bus and asserts that neuron's individual `valid` signal:

```verilog
neuron_valid[assign_neuron] <= 1;
neuron_c_re    <= cur_c_re;
neuron_c_im    <= cur_c_im;
neuron_pixel_id <= pixel_count;
```

Note that all neurons share the same coordinate and pixel ID buses — since only one `neuron_valid[i]` is high at a time, only the selected neuron latches the values. This saves routing resources compared to per-neuron coordinate buses.

### Assignment Rate

The scheduler can assign at most **one pixel per clock cycle**. At 50 MHz, assigning all 55,040 pixels takes at minimum 55,040 cycles = 1.1 ms. In practice, assignment stalls occasionally when all neurons are busy (especially for interior pixels that take max_iter iterations), but the 18-neuron pool keeps most neurons saturated.

## Result Collection — Pending Buffer

Neuron results arrive as one-cycle `result_valid` pulses. Since 18 neurons operate independently, multiple can complete on the same clock cycle. The framebuffer has only one write port, so at most one result can be written per cycle. A pending buffer bridges this gap.

### Capture (All Neurons, Same Cycle)

```verilog
reg [N_NEURONS-1:0] result_pending;

for (n = 0; n < N_NEURONS; n = n + 1) begin
    if (result_valid[n])
        result_pending[n] <= 1'b1;
end
```

All 18 result_valid signals are sampled simultaneously. A neuron that fires its result_valid pulse is guaranteed to have it captured in the pending register — no results are lost even if all 18 fire on the same cycle.

### Drain (One Per Cycle)

A second priority encoder finds the first pending result and writes it to the framebuffer:

```verilog
always @(*) begin
    found_pending = 0;
    drain_neuron  = 0;
    for (m = 0; m < N_NEURONS; m = m + 1) begin
        if (result_pending[m] && !found_pending) begin
            drain_neuron  = m;
            found_pending = 1;
        end
    end
end

if (frame_busy && found_pending) begin
    fb_wr_en   <= 1;
    fb_wr_addr <= result_pixel_id[drain_neuron*16 +: 16];
    fb_wr_data <= result_iter[drain_neuron*ITER_W +: ITER_W];
    result_pending[drain_neuron] <= 1'b0;
    pixels_done <= pixels_done + 1;
end
```

The `result_pixel_id` and `result_iter` buses are flat (Verilog-2001 doesn't support unpacked array ports), indexed using the `[i*W +: W]` part-select syntax.

### Why Not Just Write Results Directly?

If multiple neurons fire result_valid on the same cycle, only one can write the framebuffer. Without the pending buffer, the other results would be lost. The pending buffer ensures every result is eventually written, at the cost of a maximum 18-cycle drain latency (worst case: all neurons finish simultaneously).

## State-Based Frame Completion

A frame is complete when all pixels have been assigned and all neurons have finished and their results have been drained. The naive approach — counting completed pixels — is unreliable because of subtle interactions between the one-cycle result_valid pulses and the pending buffer drain logic (see [lessons-learned.md](lessons-learned.md) for the debugging story).

Instead, frame completion uses a state-based check:

```verilog
if (frame_busy && all_assigned &&
    (neuron_ready == {N_NEURONS{1'b1}}) &&           // All neurons idle
    (result_pending == {N_NEURONS{1'b0}}) &&          // No pending results
    (result_valid == {N_NEURONS{1'b0}})) begin        // No active pulses
    frame_busy <= 0;
    frame_done <= 1;
end
```

This is a conjunction of four conditions, all of which must be true simultaneously:

1. **`all_assigned`**: Every pixel coordinate has been dispatched
2. **All neurons idle**: `neuron_ready == 18'b111...1`
3. **No pending results**: The drain buffer is empty
4. **No active result pulses**: No neuron is asserting result_valid this cycle

If all four hold, every pixel has been computed, collected, and written to the framebuffer. This approach is robust regardless of timing coincidences between result pulses and drain operations.

## Throughput Analysis

For a typical frame with `max_iter = 256`:

```
  Average iterations per pixel ≈ 45    (empirical, depends on viewport)
  Clock cycles per iteration   = 4      (neuron FSM)
  Clock cycles per pixel       ≈ 180    (45 × 4)
  Total compute cycles         ≈ 55,040 × 180 / 18 = 550,400
  Time at 50 MHz              ≈ 11 ms

  SPI frame time               ≈ 70 ms  (55,040 × 2 bytes × 16 bits / 25 MHz)
```

The compute engine finishes each frame ~6× faster than the SPI can display it, so the display is the bottleneck. Interior-heavy viewports (high average iteration count) can push compute time higher, but even at max_iter = 1024 with a mostly-interior viewport, the 18-neuron pool keeps up.

## Frame Start Sequence

When `frame_start` is pulsed:

```verilog
if (frame_start && !frame_busy) begin
    frame_busy     <= 1;
    px             <= 0;
    py             <= 0;
    pixel_count    <= 0;
    pixels_done    <= 0;
    all_assigned   <= 0;
    cur_c_re       <= c_re_start;
    cur_c_im       <= c_im_start;
    row_c_re_start <= c_re_start;
    result_pending <= 0;
end
```

The viewport parameters (`c_re_start`, `c_im_start`, `c_re_step`, `c_im_step`) are latched from the auto-zoom controller's registers at this point. The scheduler begins assigning pixels on the next cycle.
