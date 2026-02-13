// mlp_core.v — MLP inference core (drop-in replacement for neuron_core)
//
// Computes a small SIREN network: inputs (x, y, t) → RGB565
// Uses 1 pipelined fixed_mul instance for sequential MAC operations.
//
// Network architecture: 3 → N_HIDDEN → N_HIDDEN → 3
//   Layer 0: 3 inputs × N_HIDDEN outputs + N_HIDDEN biases
//   Layer 1: N_HIDDEN inputs × N_HIDDEN outputs + N_HIDDEN biases
//   Layer 2: N_HIDDEN inputs × 3 outputs + 3 biases
//   Activation: sin(x) via sine_lut (SIREN activation)
//   Output: clamped to [0,1], scaled to RGB565
//
// Weight storage: BRAM (block RAM) per core, 512×32 bits.
// Initialized from mlp_weights.vh at synthesis time.
//
// Sequential MAC pipeline:
//   1. Read weight from BRAM (1 cycle latency)
//   2. Multiply weight × activation (3 cycle pipeline)
//   3. Accumulate result
//   Total: ~7 cycles per input (read + mul + acc), pipelined to ~4/input.
//
// Cycle budget (N_HIDDEN=16, 3→16→16→3):
//   Layer 0: (3+4) × 16 + 16×3 = 160
//   Layer 1: (16+4) × 16 + 16×3 = 368
//   Layer 2: (16+4) × 3 + 3×1 = 63
//   Overhead: ~25
//   Total: ~616 cycles/pixel
//   18 cores × 50 MHz / 616 = 1.46M pixels/sec → ~26 FPS at 320×172

`timescale 1ns / 1ps

module mlp_core #(
    parameter WIDTH    = 32,
    parameter FRAC     = 28,
    parameter ITER_W   = 16,
    parameter N_HIDDEN = 16,
    parameter N_LAYERS = 3
)(
    input  wire                     clk,
    input  wire                     rst_n,

    input  wire                     pixel_valid,
    output wire                     pixel_ready,
    input  wire [WIDTH-1:0]         c_re,       // x coordinate (Q4.28)
    input  wire [WIDTH-1:0]         c_im,       // y coordinate (Q4.28)
    input  wire [15:0]              pixel_id,

    input  wire [ITER_W-1:0]        max_iter,   // time parameter

    output reg                      result_valid,
    output reg  [15:0]              result_pixel_id,
    output reg  [ITER_W-1:0]        result_iter,    // carries RGB565

    output wire                     busy
);

    localparam N_INPUT   = 3;
    localparam N_OUTPUT  = 3;
    // Total params: (3*16+16) + (16*16+16) + (16*3+3) = 64 + 272 + 51 = 387
    localparam N_WEIGHTS = 387;
    localparam WADDR_W   = 9;  // ceil(log2(512))

    // =========================================================
    // Weight BRAM — 512 × 32-bit, initialized at synthesis
    // =========================================================
    (* ram_style = "block" *) reg signed [WIDTH-1:0] weight_mem [0:511];

    `include "mlp_weights.vh"

    // BRAM read port: registered address → 1 cycle read latency
    reg  [WADDR_W-1:0] w_addr;
    reg  signed [WIDTH-1:0] w_data;  // registered read output

    always @(posedge clk) begin
        w_data <= weight_mem[w_addr];
    end

    // =========================================================
    // Layer geometry (precomputed constants)
    // =========================================================
    // Layer 0: fan_in=3,  fan_out=16, w_base=0,   b_base=48
    // Layer 1: fan_in=16, fan_out=16, w_base=64,  b_base=320
    // Layer 2: fan_in=16, fan_out=3,  w_base=336, b_base=384

    function [4:0] get_fan_in;
        input [1:0] layer;
        case (layer)
            2'd0: get_fan_in = N_INPUT;
            2'd1: get_fan_in = N_HIDDEN;
            2'd2: get_fan_in = N_HIDDEN;
            default: get_fan_in = 0;
        endcase
    endfunction

    function [4:0] get_fan_out;
        input [1:0] layer;
        case (layer)
            2'd0: get_fan_out = N_HIDDEN;
            2'd1: get_fan_out = N_HIDDEN;
            2'd2: get_fan_out = N_OUTPUT;
            default: get_fan_out = 0;
        endcase
    endfunction

    function [WADDR_W-1:0] get_w_base;
        input [1:0] layer;
        case (layer)
            2'd0: get_w_base = 0;
            2'd1: get_w_base = N_INPUT * N_HIDDEN + N_HIDDEN;       // 64
            2'd2: get_w_base = N_INPUT * N_HIDDEN + N_HIDDEN
                             + N_HIDDEN * N_HIDDEN + N_HIDDEN;      // 336
            default: get_w_base = 0;
        endcase
    endfunction

    function [WADDR_W-1:0] get_b_base;
        input [1:0] layer;
        case (layer)
            2'd0: get_b_base = N_INPUT * N_HIDDEN;                  // 48
            2'd1: get_b_base = N_INPUT * N_HIDDEN + N_HIDDEN
                             + N_HIDDEN * N_HIDDEN;                  // 320
            2'd2: get_b_base = N_INPUT * N_HIDDEN + N_HIDDEN
                             + N_HIDDEN * N_HIDDEN + N_HIDDEN
                             + N_OUTPUT * N_HIDDEN;                  // 384
            default: get_b_base = 0;
        endcase
    endfunction

    // =========================================================
    // FSM states
    // =========================================================
    localparam [3:0] S_IDLE      = 4'd0,
                     S_SETUP     = 4'd1,   // set up layer params
                     S_W_READ    = 4'd2,   // issue weight BRAM read
                     S_W_WAIT    = 4'd3,   // wait for BRAM read latency
                     S_MAC       = 4'd4,   // launch multiply
                     S_MUL_WAIT  = 4'd5,   // wait for multiply pipeline
                     S_ACC       = 4'd6,   // accumulate result
                     S_BIAS_RD   = 4'd7,   // issue bias BRAM read
                     S_BIAS_WAIT = 4'd8,   // wait for bias read
                     S_BIAS_ADD  = 4'd9,   // add bias to accumulator
                     S_ACTIVATE  = 4'd10,  // apply sin() activation
                     S_ACT_WAIT  = 4'd11,  // wait for sine_lut latency
                     S_NEXT_N    = 4'd12,  // advance to next neuron
                     S_NEXT_L    = 4'd13,  // advance to next layer
                     S_OUTPUT    = 4'd14;  // format RGB565

    reg [3:0] state;

    // =========================================================
    // Working registers
    // =========================================================
    reg [1:0]  cur_layer;
    reg [4:0]  cur_neuron;
    reg [4:0]  cur_k;
    reg [4:0]  cur_fan_in;
    reg [4:0]  cur_fan_out;
    reg [WADDR_W-1:0] cur_w_base;
    reg [WADDR_W-1:0] cur_b_base;
    reg [15:0] pid_r;

    // 36-bit accumulator (4 bits overflow headroom)
    reg signed [WIDTH+3:0] acc;

    // Pipeline counter
    reg [1:0] pipe_cnt;

    // =========================================================
    // Activation register banks (double-buffered)
    // =========================================================
    reg signed [WIDTH-1:0] act_a [0:N_HIDDEN-1];
    reg signed [WIDTH-1:0] act_b [0:N_HIDDEN-1];
    reg bank_sel;  // 0: read A, write B. 1: read B, write A.

    // Read from current input bank
    wire signed [WIDTH-1:0] act_rd = bank_sel ? act_b[cur_k] : act_a[cur_k];

    // =========================================================
    // Single multiplier for MAC
    // =========================================================
    reg signed [WIDTH-1:0] mul_in_a, mul_in_b;
    reg                    mul_valid_in;
    wire signed [WIDTH-1:0] mul_result;
    wire                    mul_valid_out;

    fixed_mul #(.WIDTH(WIDTH), .FRAC(FRAC)) u_mac (
        .clk(clk), .rst_n(rst_n),
        .a(mul_in_a), .b(mul_in_b),
        .valid_in(mul_valid_in),
        .result(mul_result), .valid_out(mul_valid_out)
    );

    // =========================================================
    // Sine LUT for SIREN activation
    // =========================================================
    reg  signed [WIDTH-1:0] sin_input;
    wire signed [WIDTH-1:0] sin_output;

    sine_lut #(.WIDTH(WIDTH), .FRAC(FRAC)) u_sine (
        .clk(clk), .angle(sin_input), .result(sin_output)
    );

    // =========================================================
    // Output color registers
    // =========================================================
    reg signed [WIDTH-1:0] out_r, out_g, out_b;

    // =========================================================
    // Handshake
    // =========================================================
    assign pixel_ready = (state == S_IDLE);
    assign busy        = (state != S_IDLE);

    // Time: scale max_iter to Q4.28 so 4*pi (~12.57) is reached in ~800 frames
    // << 22 gives 0.0156 per frame; at ~26 FPS, full 4*pi cycle in ~31 seconds
    wire signed [WIDTH-1:0] time_val = {16'b0, max_iter} << 22;

    // Saturate accumulator to Q4.28
    wire signed [WIDTH-1:0] acc_sat;
    assign acc_sat = (acc[WIDTH+3] && !(&acc[WIDTH+2:WIDTH-1])) ? {1'b1, {(WIDTH-1){1'b0}}} :  // negative overflow
                     (!acc[WIDTH+3] && |acc[WIDTH+2:WIDTH-1]) ? {1'b0, {(WIDTH-1){1'b1}}} :     // positive overflow
                     acc[WIDTH-1:0];

    // Weight address for current neuron/input
    wire [WADDR_W-1:0] w_addr_calc = cur_w_base + cur_neuron * cur_fan_in + cur_k;

    // =========================================================
    // Main FSM
    // =========================================================
    integer idx;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= S_IDLE;
            cur_layer       <= 0;
            cur_neuron      <= 0;
            cur_k           <= 0;
            cur_fan_in      <= 0;
            cur_fan_out     <= 0;
            cur_w_base      <= 0;
            cur_b_base      <= 0;
            pid_r           <= 0;
            acc             <= 0;
            pipe_cnt        <= 0;
            bank_sel        <= 0;
            mul_in_a        <= 0;
            mul_in_b        <= 0;
            mul_valid_in    <= 0;
            sin_input       <= 0;
            result_valid    <= 0;
            result_pixel_id <= 0;
            result_iter     <= 0;
            out_r           <= 0;
            out_g           <= 0;
            out_b           <= 0;
            w_addr          <= 0;
            for (idx = 0; idx < N_HIDDEN; idx = idx + 1) begin
                act_a[idx] <= 0;
                act_b[idx] <= 0;
            end
        end else begin
            result_valid <= 0;
            mul_valid_in <= 0;

            case (state)
                // -------------------------------------------------
                S_IDLE: begin
                    if (pixel_valid) begin
                        pid_r <= pixel_id;
                        act_a[0] <= c_re;       // x
                        act_a[1] <= c_im;       // y
                        act_a[2] <= time_val;   // t
                        bank_sel <= 0;
                        cur_layer <= 0;
                        state <= S_SETUP;
                    end
                end

                // -------------------------------------------------
                S_SETUP: begin
                    // Cache layer parameters
                    cur_fan_in  <= get_fan_in(cur_layer);
                    cur_fan_out <= get_fan_out(cur_layer);
                    cur_w_base  <= get_w_base(cur_layer);
                    cur_b_base  <= get_b_base(cur_layer);
                    cur_neuron  <= 0;
                    cur_k       <= 0;
                    acc         <= 0;
                    state       <= S_W_READ;
                end

                // -------------------------------------------------
                S_W_READ: begin
                    // Issue BRAM read for weight[w_base + neuron*fan_in + k]
                    w_addr <= w_addr_calc;
                    state  <= S_W_WAIT;
                end

                // -------------------------------------------------
                S_W_WAIT: begin
                    // BRAM read latency: 1 cycle. w_data available next cycle.
                    state <= S_MAC;
                end

                // -------------------------------------------------
                S_MAC: begin
                    // Launch multiply: w_data (from BRAM) × act_rd (from register bank)
                    mul_in_a     <= w_data;
                    mul_in_b     <= act_rd;
                    mul_valid_in <= 1'b1;
                    pipe_cnt     <= 0;
                    state        <= S_MUL_WAIT;
                end

                // -------------------------------------------------
                S_MUL_WAIT: begin
                    // Wait 3 cycles for fixed_mul pipeline
                    if (pipe_cnt == 2'd2) begin
                        state <= S_ACC;
                    end else begin
                        pipe_cnt <= pipe_cnt + 1;
                    end
                end

                // -------------------------------------------------
                S_ACC: begin
                    // Accumulate multiply result
                    acc   <= acc + {{4{mul_result[WIDTH-1]}}, mul_result};
                    cur_k <= cur_k + 1;

                    if (cur_k + 1 < cur_fan_in) begin
                        // More inputs: read next weight
                        state <= S_W_READ;
                    end else begin
                        // Dot product done, read bias
                        state <= S_BIAS_RD;
                    end
                end

                // -------------------------------------------------
                S_BIAS_RD: begin
                    // Issue BRAM read for bias[b_base + neuron]
                    w_addr <= cur_b_base + cur_neuron;
                    state  <= S_BIAS_WAIT;
                end

                // -------------------------------------------------
                S_BIAS_WAIT: begin
                    // BRAM read latency
                    state <= S_BIAS_ADD;
                end

                // -------------------------------------------------
                S_BIAS_ADD: begin
                    // Add bias from BRAM read
                    acc <= acc + {{4{w_data[WIDTH-1]}}, w_data};

                    if (cur_layer == N_LAYERS - 1) begin
                        // Output layer: no activation
                        state <= S_NEXT_N;
                    end else begin
                        // Hidden layer: sin() activation
                        state <= S_ACTIVATE;
                    end
                end

                // -------------------------------------------------
                S_ACTIVATE: begin
                    sin_input <= acc_sat;
                    pipe_cnt  <= 0;
                    state     <= S_ACT_WAIT;
                end

                // -------------------------------------------------
                S_ACT_WAIT: begin
                    // sine_lut has 2-cycle latency
                    if (pipe_cnt == 2'd1) begin
                        if (bank_sel)
                            act_a[cur_neuron] <= sin_output;
                        else
                            act_b[cur_neuron] <= sin_output;
                        state <= S_NEXT_N;
                    end else begin
                        pipe_cnt <= pipe_cnt + 1;
                    end
                end

                // -------------------------------------------------
                S_NEXT_N: begin
                    // For output layer: store channel value
                    if (cur_layer == N_LAYERS - 1) begin
                        case (cur_neuron[1:0])
                            2'd0: out_r <= acc_sat;
                            2'd1: out_g <= acc_sat;
                            2'd2: out_b <= acc_sat;
                            default: ;
                        endcase
                    end

                    if (cur_neuron + 1 < cur_fan_out) begin
                        cur_neuron <= cur_neuron + 1;
                        cur_k      <= 0;
                        acc        <= 0;
                        state      <= S_W_READ;
                    end else begin
                        state <= S_NEXT_L;
                    end
                end

                // -------------------------------------------------
                S_NEXT_L: begin
                    if (cur_layer + 1 < N_LAYERS) begin
                        cur_layer <= cur_layer + 1;
                        bank_sel  <= ~bank_sel;
                        state     <= S_SETUP;
                    end else begin
                        state <= S_OUTPUT;
                    end
                end

                // -------------------------------------------------
                S_OUTPUT: begin
                    result_valid    <= 1'b1;
                    result_pixel_id <= pid_r;
                    result_iter     <= pack_rgb565(out_r, out_g, out_b);
                    state           <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    // =========================================================
    // RGB565 packing function
    // =========================================================
    function [ITER_W-1:0] pack_rgb565;
        input signed [WIDTH-1:0] r_val, g_val, b_val;
        reg signed [WIDTH-1:0] rs, gs, bs;
        reg [4:0] r5;
        reg [5:0] g6;
        reg [4:0] b5;
        begin
            // Shift [-1,+1] → [0,+2] by adding 1.0 (Q4.28: 0x10000000)
            rs = r_val + 32'sh1000_0000;
            gs = g_val + 32'sh1000_0000;
            bs = b_val + 32'sh1000_0000;

            // Extract 5 bits for R: [0, 2.0) → [0, 31]
            if (rs[WIDTH-1])          r5 = 5'd0;     // negative = clamp to 0
            else if (rs >= 32'sh2000_0000) r5 = 5'd31;  // >= 2.0 = clamp to max
            else                      r5 = rs[27:23]; // bits [27:23] of [0, 2.0)

            // 6 bits for G
            if (gs[WIDTH-1])          g6 = 6'd0;
            else if (gs >= 32'sh2000_0000) g6 = 6'd63;
            else                      g6 = gs[27:22];

            // 5 bits for B
            if (bs[WIDTH-1])          b5 = 5'd0;
            else if (bs >= 32'sh2000_0000) b5 = 5'd31;
            else                      b5 = bs[27:23];

            pack_rgb565 = {r5, g6, b5};
        end
    endfunction

endmodule
