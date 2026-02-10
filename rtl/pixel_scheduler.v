// pixel_scheduler.v — Work distribution for parallel neuron pool
//
// Manages a pool of N neuron_core instances:
//   - Generates pixel coordinates from viewport parameters
//   - Assigns pixels to idle neurons (round-robin scan of ready signals)
//   - Collects results and writes iteration counts to framebuffer
//   - Signals frame completion
//
// Viewport coordinate generation:
//   For pixel (px, py), the complex coordinate is:
//     c_re = view_x + (px - H_RES/2) * step
//     c_im = view_y + (py - V_RES/2) * step
//   Where step = zoom_level / H_RES (pixel spacing in complex plane)
//
//   To avoid division in hardware, the ARM pre-computes:
//     - c_re_start = view_x - (H_RES/2) * step
//     - c_im_start = view_y - (V_RES/2) * step
//     - c_re_step  = step (horizontal increment per pixel)
//     - c_im_step  = step (vertical increment per pixel)
//   And the scheduler just accumulates: c_re = c_re_start + px * c_re_step

`timescale 1ns / 1ps

module pixel_scheduler #(
    parameter N_NEURONS = 36,
    parameter WIDTH     = 32,
    parameter FRAC      = 28,
    parameter ITER_W    = 16,
    parameter H_RES     = 320,
    parameter V_RES     = 172,
    parameter PIX_COUNT = H_RES * V_RES
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // Control (from AXI registers / ARM)
    input  wire                     frame_start,    // Pulse to begin new frame
    output reg                      frame_busy,
    output reg                      frame_done,     // Pulse when frame complete

    // Viewport parameters (from ARM, Q4.28 fixed-point)
    input  wire signed [WIDTH-1:0]  c_re_start,     // Left edge real coordinate
    input  wire signed [WIDTH-1:0]  c_im_start,     // Top edge imaginary coordinate
    input  wire signed [WIDTH-1:0]  c_re_step,      // Horizontal step per pixel
    input  wire signed [WIDTH-1:0]  c_im_step,      // Vertical step per pixel
    input  wire [ITER_W-1:0]        max_iter,

    // Neuron pool interface
    output reg  [N_NEURONS-1:0]     neuron_valid,   // Per-neuron pixel_valid
    input  wire [N_NEURONS-1:0]     neuron_ready,   // Per-neuron pixel_ready
    output reg  signed [WIDTH-1:0]  neuron_c_re,    // Shared coordinate bus
    output reg  signed [WIDTH-1:0]  neuron_c_im,
    output reg  [15:0]              neuron_pixel_id,

    // Result collection (from any neuron) — flat buses, indexed as [i*16 +: 16]
    input  wire [N_NEURONS-1:0]         result_valid,
    input  wire [N_NEURONS*16-1:0]      result_pixel_id,
    input  wire [N_NEURONS*ITER_W-1:0]  result_iter,

    // Framebuffer write port
    output reg                      fb_wr_en,
    output reg  [15:0]              fb_wr_addr,
    output reg  [ITER_W-1:0]        fb_wr_data      // Iteration count (colormap applied later)
);

    // =========================================================
    // Pixel coordinate generator
    // =========================================================
    reg [8:0]  px;                // Current pixel X (0 to H_RES-1)
    reg [7:0]  py;                // Current pixel Y (0 to V_RES-1)
    reg [15:0] pixel_count;       // Total pixels assigned so far
    reg [15:0] pixels_done;       // Total pixels completed
    reg        all_assigned;      // All pixels have been assigned to neurons

    // Current pixel coordinate (accumulated)
    reg signed [WIDTH-1:0] cur_c_re;
    reg signed [WIDTH-1:0] cur_c_im;
    reg signed [WIDTH-1:0] row_c_re_start;  // Start of current row

    // =========================================================
    // Result pending buffer
    // =========================================================
    // Neuron result_valid is a one-cycle pulse. With N neurons,
    // multiple can fire on the same cycle. We capture ALL pulses
    // into pending flags and drain one per cycle to the framebuffer.
    // To prevent buffer overwrite, we don't assign new work to
    // neurons that still have a pending (uncollected) result.
    reg [N_NEURONS-1:0] result_pending;

    // =========================================================
    // Neuron assignment — find first ready neuron (without pending result)
    // =========================================================
    // just_assigned: mask of neuron assigned last cycle. Neuron_ready has 1-cycle
    // latency (neuron transitions from IDLE to LOAD on the cycle it sees valid),
    // so the scheduler must not re-pick the same neuron on the very next cycle.
    reg [N_NEURONS-1:0] just_assigned;

    reg [$clog2(N_NEURONS)-1:0] assign_neuron;
    reg                         found_ready;

    integer k;
    always @(*) begin
        found_ready   = 0;
        assign_neuron = 0;
        for (k = 0; k < N_NEURONS; k = k + 1) begin
            if (neuron_ready[k] && !result_pending[k] && !result_valid[k]
                && !just_assigned[k] && !found_ready) begin
                assign_neuron = k;
                found_ready   = 1;
            end
        end
    end

    // =========================================================
    // Result drain — find first pending result
    // =========================================================
    reg [$clog2(N_NEURONS)-1:0] drain_neuron;
    reg                         found_pending;

    integer m;
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

    // =========================================================
    // Main logic
    // =========================================================
    integer n;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            frame_busy     <= 0;
            frame_done     <= 0;
            px             <= 0;
            py             <= 0;
            pixel_count    <= 0;
            pixels_done    <= 0;
            all_assigned   <= 0;
            cur_c_re       <= 0;
            cur_c_im       <= 0;
            row_c_re_start <= 0;
            neuron_valid   <= 0;
            neuron_c_re    <= 0;
            neuron_c_im    <= 0;
            neuron_pixel_id <= 0;
            fb_wr_en       <= 0;
            fb_wr_addr     <= 0;
            fb_wr_data     <= 0;
            result_pending <= 0;
            just_assigned  <= 0;
        end else begin
            // Defaults
            neuron_valid  <= 0;
            fb_wr_en      <= 0;
            frame_done    <= 0;
            just_assigned <= 0;

            // ---- Capture result pulses into pending buffer ----
            // All neurons captured simultaneously — no lost results
            for (n = 0; n < N_NEURONS; n = n + 1) begin
                if (result_valid[n])
                    result_pending[n] <= 1'b1;
            end

            // ---- Drain one pending result per cycle ----
            if (frame_busy && found_pending) begin
                fb_wr_en   <= 1;
                fb_wr_addr <= result_pixel_id[drain_neuron*16 +: 16];
                fb_wr_data <= result_iter[drain_neuron*ITER_W +: ITER_W];
                result_pending[drain_neuron] <= 1'b0;
                pixels_done <= pixels_done + 1;
            end

            // ---- Frame completion (state-based) ----
            // Done when all pixels assigned, all neurons idle, no pending/active results.
            // This avoids relying on exact pixel counting.
            if (frame_busy && all_assigned &&
                (neuron_ready == {N_NEURONS{1'b1}}) &&
                (result_pending == {N_NEURONS{1'b0}}) &&
                (result_valid == {N_NEURONS{1'b0}})) begin
                frame_busy <= 0;
                frame_done <= 1;
            end

            // ---- Pixel assignment ----
            if (frame_busy && !all_assigned && found_ready) begin
                // Assign current pixel to the ready neuron
                neuron_valid[assign_neuron]  <= 1;
                just_assigned[assign_neuron] <= 1;
                neuron_c_re    <= cur_c_re;
                neuron_c_im    <= cur_c_im;
                neuron_pixel_id <= pixel_count;

                pixel_count <= pixel_count + 1;

                // Advance to next pixel
                if (px == H_RES - 1) begin
                    px <= 0;
                    if (py == V_RES - 1) begin
                        all_assigned <= 1;
                    end else begin
                        py <= py + 1;
                        cur_c_im <= cur_c_im + c_im_step;
                        cur_c_re <= row_c_re_start;  // First pixel of new row
                    end
                end else begin
                    px <= px + 1;
                    cur_c_re <= cur_c_re + c_re_step;
                end
            end

            // ---- Frame start ----
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
        end
    end

endmodule
