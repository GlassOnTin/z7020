// fixed_mul.v — Signed fixed-point Q4.28 multiply
//
// Computes: out = (a * b) >> 28, in Q4.28 format
// Where Q4.28 means 4 integer bits (including sign), 28 fractional bits
// Range: [-8.0, +8.0), resolution: ~3.7e-9
//
// Implementation: 32x32 signed multiply → 64-bit product → take bits [59:28]
// This maps to DSP48E1 slices. A 32x32 multiply requires splitting into
// partial products that fit the 25x18 signed multiplier in each DSP48E1.
//
// Strategy: Karatsuba-lite using 3 DSP48E1 slices:
//   a = aH:aL (16:16 split)
//   b = bH:bL (16:16 split)
//   a*b = (aH*bH)<<32 + (aH*bL + aL*bH)<<16 + aL*bL
//
// But DSP48E1 is 25x18 signed, so we can do better with a 2-DSP approach:
//   a = aH(18 bits, signed) : aL(14 bits, unsigned)
//   b = full 32 bits into 25-bit signed A port (need care)
//
// Simplest reliable approach: 4 partial products with 16-bit halves,
// using 4 DSPs. But we can do 3 DSPs with the 25x18 ports.
//
// We use a 3-cycle pipelined approach:
//   Cycle 1: aL*bL (unsigned 16x16), aH*bL (signed 16x16), aL*bH (16x16)
//   Cycle 2: aH*bH (signed 16x16), accumulate partials
//   Cycle 3: Final sum and truncation
//
// Actually, let's use the straightforward approach that Vivado reliably
// maps to DSPs: just write the multiply and let synthesis infer.
// For 32x32 signed, Vivado uses 4 DSP48E1 slices (worst case) or 3 with
// good pipelining. We pipeline with 3 stages for timing closure at 150 MHz.

`timescale 1ns / 1ps

module fixed_mul #(
    parameter WIDTH = 32,       // Total bit width
    parameter FRAC  = 28        // Fractional bits
)(
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire signed [WIDTH-1:0] a,
    input  wire signed [WIDTH-1:0] b,
    input  wire                 valid_in,
    output reg  signed [WIDTH-1:0] result,
    output reg                  valid_out
);

    // Full-precision product: 32 x 32 = 64 bits
    // We want bits [FRAC+WIDTH-1 : FRAC] = [59:28] of the 64-bit product

    // Pipeline stage 1: register inputs
    reg signed [WIDTH-1:0] a_r, b_r;
    reg                    valid_p1;

    // Pipeline stage 2: multiply (Vivado infers DSP48E1 here)
    reg signed [2*WIDTH-1:0] product;
    reg                      valid_p2;

    // Pipeline stage 3: extract and register output
    // (valid_out and result are the output registers)

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_r       <= 0;
            b_r       <= 0;
            valid_p1  <= 0;
            product   <= 0;
            valid_p2  <= 0;
            result    <= 0;
            valid_out <= 0;
        end else begin
            // Stage 1: register inputs
            a_r      <= a;
            b_r      <= b;
            valid_p1 <= valid_in;

            // Stage 2: multiply
            product  <= a_r * b_r;
            valid_p2 <= valid_p1;

            // Stage 3: extract Q4.28 result from 64-bit product
            // Product is Q8.56 (4+4 integer, 28+28 fractional)
            // We want Q4.28, so take bits [59:28]
            result    <= product[FRAC+WIDTH-1 : FRAC];
            valid_out <= valid_p2;
        end
    end

    // Overflow detection (optional, useful for debug)
    // If the upper bits [63:60] are not all-same-sign, we overflowed
    // wire overflow = (product[63:FRAC+WIDTH] != {(2*WIDTH-FRAC-WIDTH){product[2*WIDTH-1]}});

endmodule
