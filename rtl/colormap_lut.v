// colormap_lut.v — Mandelbrot iteration count to RGB565 color mapping
//
// Maps a 16-bit iteration count to a 16-bit RGB565 color value.
// Uses a 256-entry cyclic palette stored in a lookup table.
// Iteration count is taken modulo 256 for the palette index.
// Max iteration (set interior) maps to black.
//
// The palette uses a classic "Ultra Fractal" style coloring with
// smooth transitions through blue → cyan → yellow → orange → brown.

`timescale 1ns / 1ps

module colormap_lut #(
    parameter ITER_W     = 16,
    parameter PALETTE_SZ = 256
)(
    input  wire                 clk,
    input  wire [ITER_W-1:0]   iter_count,
    input  wire [ITER_W-1:0]   max_iter,
    input  wire                 valid_in,
    output reg  [15:0]          rgb565,
    output reg                  valid_out
);

    // Palette ROM — 256 entries of RGB565
    // Generated from HSV sweep with value modulation
    reg [15:0] palette [0:PALETTE_SZ-1];

    // Palette index: iter modulo 256 (just take low 8 bits)
    wire [7:0] pal_idx = iter_count[7:0];
    wire       is_interior = (iter_count >= max_iter);

    // Single-cycle lookup with output register
    always @(posedge clk) begin
        valid_out <= valid_in;
        if (valid_in) begin
            if (is_interior)
                rgb565 <= 16'h0000;   // Black for interior points
            else
                rgb565 <= palette[pal_idx];
        end
    end

    // =========================================================
    // Palette initialization
    // =========================================================
    // Classic Mandelbrot palette: smooth cycling through
    // dark blue → blue → cyan → green → yellow → orange → red → dark
    //
    // RGB565 format: RRRRR_GGGGGG_BBBBB (5-6-5 bits)
    //
    // This is computed from a piecewise-linear HSV-like function:
    //   Hue cycles through the spectrum
    //   Value dips at transitions for depth effect

    integer i;
    initial begin
        for (i = 0; i < PALETTE_SZ; i = i + 1) begin
            palette[i] = compute_color(i);
        end
    end

    // Color computation function
    // Maps index 0-255 to RGB565 through a smooth cyclic palette
    function [15:0] compute_color;
        input integer idx;
        reg [7:0] r, g, b;
        reg [7:0] phase;
        integer   t;
    begin
        // 6-phase color cycle, ~42 entries per phase
        phase = idx;
        t = idx % 42;

        if (idx < 42) begin
            // Phase 0: black → dark blue
            r = 0;
            g = 0;
            b = (t * 128) / 42;
        end else if (idx < 84) begin
            // Phase 1: dark blue → cyan
            r = 0;
            g = (t * 255) / 42;
            b = 128 + (t * 127) / 42;
        end else if (idx < 126) begin
            // Phase 2: cyan → white/yellow
            r = (t * 255) / 42;
            g = 255;
            b = 255 - (t * 255) / 42;
        end else if (idx < 168) begin
            // Phase 3: yellow → orange
            r = 255;
            g = 255 - (t * 128) / 42;
            b = 0;
        end else if (idx < 210) begin
            // Phase 4: orange → red/brown
            r = 255 - (t * 128) / 42;
            g = 127 - (t * 127) / 42;
            b = 0;
        end else begin
            // Phase 5: brown → black (loop)
            t = idx - 210;
            r = 127 - (t * 127) / 46;
            g = 0;
            b = 0;
        end

        // Pack as RGB565: R[7:3] G[7:2] B[7:3]
        compute_color = {r[7:3], g[7:2], b[7:3]};
    end
    endfunction

endmodule
