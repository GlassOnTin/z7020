// neuron_core.v — Mandelbrot z=z²+c recurrent iterator
//
// A single "neuron" that computes the Mandelbrot iteration:
//   z(n+1) = z(n)² + c
// until |z|² > 4.0 (escape) or iteration count reaches max_iter.
//
// This is a recurrent processing element with:
//   - Local state (z_re, z_im)
//   - Data-dependent halting (escape detection)
//   - Independent operation (no global synchronization)
//
// The neuron uses 3 pipelined multipliers (6+ DSP48E1 slices):
//   mul_a: z_re * z_re → z_re_sq
//   mul_b: z_im * z_im → z_im_sq
//   mul_c: z_re * z_im → z_re_im
//
// Each multiply has 3-cycle latency. The iteration pipeline:
//   Cycle 0:     Launch multiplies with current z_re, z_im
//   Cycle 1-2:   Pipeline stages inside multipliers
//   Cycle 3:     Results available: z_re_sq, z_im_sq, z_re_im
//                Compute: z_re_new = z_re_sq - z_im_sq + c_re
//                         z_im_new = 2*z_re_im + c_im  (shift left by 1)
//                         mag_sq   = z_re_sq + z_im_sq
//                Check escape: mag_sq > 4.0 (Q4.28: 4.0 = 32'h4000_0000)
//   Cycle 4:     Update z, increment iter, or halt
//
// Throughput: 1 iteration per 4 clock cycles
// At 50 MHz: 12.5 million iterations/sec per neuron

`timescale 1ns / 1ps

module neuron_core #(
    parameter WIDTH    = 32,
    parameter FRAC     = 28,
    parameter ITER_W   = 16     // Max iteration count width (up to 65535)
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // Pixel assignment interface (valid/ready handshake)
    input  wire                     pixel_valid,
    output wire                     pixel_ready,
    input  wire [WIDTH-1:0]         c_re,       // Pixel coordinate (real)
    input  wire [WIDTH-1:0]         c_im,       // Pixel coordinate (imag)
    input  wire [15:0]              pixel_id,   // Pixel identifier for writeback

    // Configuration
    input  wire [ITER_W-1:0]        max_iter,

    // Result output (active for one cycle when done)
    output reg                      result_valid,
    output reg  [15:0]              result_pixel_id,
    output reg  [ITER_W-1:0]        result_iter,

    // Status
    output wire                     busy
);

    // =========================================================
    // Fixed-point constants
    // =========================================================
    // 4.0 in Q4.28 = 4 << 28 = 32'h4000_0000
    localparam signed [WIDTH-1:0] ESCAPE_THRESHOLD = 32'sh4000_0000;

    // =========================================================
    // FSM States
    // =========================================================
    localparam [2:0] S_IDLE    = 3'd0,
                     S_LOAD    = 3'd1,
                     S_MUL     = 3'd2,  // Wait for multiply pipeline
                     S_UPDATE  = 3'd3,  // Check escape, update z
                     S_DONE    = 3'd4;

    reg [2:0] state;

    // =========================================================
    // Neuron state registers
    // =========================================================
    reg signed [WIDTH-1:0] z_re, z_im;
    reg signed [WIDTH-1:0] c_re_r, c_im_r;
    reg [15:0]             pid_r;
    reg [ITER_W-1:0]       iter;

    // =========================================================
    // Multiplier instances
    // =========================================================
    // Multiply inputs
    reg signed [WIDTH-1:0] mul_a_a, mul_a_b;
    reg signed [WIDTH-1:0] mul_b_a, mul_b_b;
    reg signed [WIDTH-1:0] mul_c_a, mul_c_b;
    reg                    mul_valid_in;

    // Multiply outputs
    wire signed [WIDTH-1:0] z_re_sq;    // z_re * z_re
    wire signed [WIDTH-1:0] z_im_sq;    // z_im * z_im
    wire signed [WIDTH-1:0] z_re_im;    // z_re * z_im
    wire                    mul_a_valid, mul_b_valid, mul_c_valid;

    fixed_mul #(.WIDTH(WIDTH), .FRAC(FRAC)) mul_a (
        .clk(clk), .rst_n(rst_n),
        .a(mul_a_a), .b(mul_a_b),
        .valid_in(mul_valid_in),
        .result(z_re_sq), .valid_out(mul_a_valid)
    );

    fixed_mul #(.WIDTH(WIDTH), .FRAC(FRAC)) mul_b (
        .clk(clk), .rst_n(rst_n),
        .a(mul_b_a), .b(mul_b_b),
        .valid_in(mul_valid_in),
        .result(z_im_sq), .valid_out(mul_b_valid)
    );

    fixed_mul #(.WIDTH(WIDTH), .FRAC(FRAC)) mul_c (
        .clk(clk), .rst_n(rst_n),
        .a(mul_c_a), .b(mul_c_b),
        .valid_in(mul_valid_in),
        .result(z_re_im), .valid_out(mul_c_valid)
    );

    // =========================================================
    // Pipeline wait counter (3 cycles for multiply latency)
    // =========================================================
    reg [1:0] pipe_cnt;

    // =========================================================
    // Derived values
    // =========================================================
    wire signed [WIDTH-1:0] z_re_new;
    wire signed [WIDTH-1:0] z_im_new;
    wire signed [WIDTH-1:0] mag_sq;
    wire                    escaped;
    wire                    max_reached;

    // z_re_new = z_re² - z_im² + c_re
    assign z_re_new = z_re_sq - z_im_sq + c_re_r;

    // z_im_new = 2 * z_re * z_im + c_im
    // 2*z_re_im is just a left shift by 1 in fixed-point
    assign z_im_new = {z_re_im[WIDTH-2:0], 1'b0} + c_im_r;

    // |z|² = z_re² + z_im²
    assign mag_sq = z_re_sq + z_im_sq;

    // Escape: |z|² >= 4.0
    // Using >= rather than > to catch the exact boundary and handle overflow:
    //   - If mag_sq is negative, the squared sum overflowed (definitely escaped)
    //   - If mag_sq >= 4.0, standard escape condition
    // Also check if z_re or z_im individually exceed ±4.0 (catches cases where
    // the multiply itself overflows before we can detect via mag_sq)
    wire z_re_overflow = (z_re[WIDTH-1] != z_re[WIDTH-2]);
    wire z_im_overflow = (z_im[WIDTH-1] != z_im[WIDTH-2]);
    assign escaped = (mag_sq[WIDTH-1]) || (mag_sq >= ESCAPE_THRESHOLD) ||
                     z_re_overflow || z_im_overflow;

    assign max_reached = (iter >= max_iter);

    // =========================================================
    // Handshake
    // =========================================================
    assign pixel_ready = (state == S_IDLE);
    assign busy        = (state != S_IDLE);

    // =========================================================
    // Main FSM
    // =========================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= S_IDLE;
            z_re           <= 0;
            z_im           <= 0;
            c_re_r         <= 0;
            c_im_r         <= 0;
            pid_r          <= 0;
            iter           <= 0;
            mul_a_a        <= 0;
            mul_a_b        <= 0;
            mul_b_a        <= 0;
            mul_b_b        <= 0;
            mul_c_a        <= 0;
            mul_c_b        <= 0;
            mul_valid_in   <= 0;
            pipe_cnt       <= 0;
            result_valid   <= 0;
            result_pixel_id <= 0;
            result_iter    <= 0;
        end else begin
            // Default: deassert one-shot signals
            result_valid <= 0;
            mul_valid_in <= 0;

            case (state)
                S_IDLE: begin
                    if (pixel_valid) begin
                        // Latch input coordinates
                        c_re_r <= c_re;
                        c_im_r <= c_im;
                        pid_r  <= pixel_id;
                        // Initialize z = 0
                        z_re   <= 0;
                        z_im   <= 0;
                        iter   <= 0;
                        state  <= S_LOAD;
                    end
                end

                S_LOAD: begin
                    // Launch first multiply: z_re*z_re, z_im*z_im, z_re*z_im
                    // On first iteration z=0 so results will be 0, but pipeline
                    // is the same for all iterations
                    mul_a_a      <= z_re;
                    mul_a_b      <= z_re;
                    mul_b_a      <= z_im;
                    mul_b_b      <= z_im;
                    mul_c_a      <= z_re;
                    mul_c_b      <= z_im;
                    mul_valid_in <= 1'b1;
                    pipe_cnt     <= 0;
                    state        <= S_MUL;
                end

                S_MUL: begin
                    // Wait for multiply pipeline (3 cycles)
                    if (pipe_cnt == 2'd2) begin
                        state <= S_UPDATE;
                    end else begin
                        pipe_cnt <= pipe_cnt + 1'b1;
                    end
                end

                S_UPDATE: begin
                    // Multiply results are now valid
                    if (escaped || max_reached) begin
                        // Done: output result
                        result_valid    <= 1'b1;
                        result_pixel_id <= pid_r;
                        result_iter     <= iter;
                        state           <= S_IDLE;
                    end else begin
                        // Update z and iterate again
                        z_re <= z_re_new;
                        z_im <= z_im_new;
                        iter <= iter + 1'b1;
                        // Immediately launch next multiply
                        mul_a_a      <= z_re_new;
                        mul_a_b      <= z_re_new;
                        mul_b_a      <= z_im_new;
                        mul_b_b      <= z_im_new;
                        mul_c_a      <= z_re_new;
                        mul_c_b      <= z_im_new;
                        mul_valid_in <= 1'b1;
                        pipe_cnt     <= 0;
                        state        <= S_MUL;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
