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
    // Neuron assignment — find first ready neuron
    // =========================================================
    reg [$clog2(N_NEURONS)-1:0] assign_neuron;
    reg                         found_ready;

    integer k;
    always @(*) begin
        found_ready   = 0;
        assign_neuron = 0;
        for (k = 0; k < N_NEURONS; k = k + 1) begin
            if (neuron_ready[k] && !found_ready) begin
                assign_neuron = k;
                found_ready   = 1;
            end
        end
    end

    // =========================================================
    // Result collection — find first valid result
    // =========================================================
    reg [$clog2(N_NEURONS)-1:0] result_neuron;
    reg                         found_result;

    integer m;
    always @(*) begin
        found_result  = 0;
        result_neuron = 0;
        for (m = 0; m < N_NEURONS; m = m + 1) begin
            if (result_valid[m] && !found_result) begin
                result_neuron = m;
                found_result  = 1;
            end
        end
    end

    // =========================================================
    // Main logic
    // =========================================================
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
        end else begin
            // Defaults
            neuron_valid <= 0;
            fb_wr_en     <= 0;
            frame_done   <= 0;

            // ---- Result collection (always active during frame) ----
            if (frame_busy && found_result) begin
                fb_wr_en   <= 1;
                fb_wr_addr <= result_pixel_id[result_neuron*16 +: 16];
                fb_wr_data <= result_iter[result_neuron*ITER_W +: ITER_W];
                pixels_done <= pixels_done + 1;

                // Check if frame is complete
                if (pixels_done + 1 == PIX_COUNT) begin
                    frame_busy <= 0;
                    frame_done <= 1;
                end
            end

            // ---- Pixel assignment ----
            if (frame_busy && !all_assigned && found_ready) begin
                // Assign current pixel to the ready neuron
                neuron_valid[assign_neuron] <= 1;
                neuron_c_re    <= cur_c_re;
                neuron_c_im    <= cur_c_im;
                neuron_pixel_id <= pixel_count;

                pixel_count <= pixel_count + 1;

                // Advance to next pixel
                if (px == H_RES - 1) begin
                    px <= 0;
                    cur_c_re <= row_c_re_start;  // Reset to row start (will add step on next assign)
                    if (py == V_RES - 1) begin
                        all_assigned <= 1;
                    end else begin
                        py <= py + 1;
                        cur_c_im       <= cur_c_im + c_im_step;
                        row_c_re_start <= row_c_re_start;  // Row start stays
                        cur_c_re       <= row_c_re_start + c_re_step; // First pixel of next row
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
            end
        end
    end

endmodule
