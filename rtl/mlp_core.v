// mlp_core.v — MLP inference core with weight morphing
//
// Computes a small SIREN network: inputs (x, y, t) → RGB565
// Uses 1 pipelined fixed_mul instance with dual BRAM weight storage
// and an inline blend pipeline for smooth morphing between two patterns.
//
// Network architecture: 3 → N_HIDDEN → N_HIDDEN → 3
//   Layer 0: 3 inputs × N_HIDDEN outputs + N_HIDDEN biases
//   Layer 1: N_HIDDEN inputs × N_HIDDEN outputs + N_HIDDEN biases
//   Layer 2: N_HIDDEN inputs × 3 outputs + 3 biases
//   Activation: sin(x) via sine_lut (SIREN activation, all layers)
//   Output: clamped to [0,1], scaled to RGB565
//
// Weight morphing: two weight sets (A and B) stored in separate BRAMs.
// A blend pipeline computes w = w_a + ((w_b - w_a) * alpha) >> 8
// with 1 cycle of pipeline latency, feeding the MAC at 1 weight/cycle.
//
// Cycle budget (N_HIDDEN=16, 3→16→16→3):
//   ~780 cycles/pixel (1 extra cycle per neuron for blend fill)
//   18 cores × 50 MHz / 780 = ~1.15M pixels/sec → ~21 FPS at 320×172

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
    input  wire [7:0]               alpha,      // morph blend: 0=pattern A, 255=pattern B

    // Weight write port (from PS via AXI): broadcast to all cores
    input  wire [10:0]             weight_wr_addr,  // 11 bits: covers up to 2048 params
    input  wire [WIDTH-1:0]        weight_wr_data,
    input  wire                    weight_wr_en,    // write enable
    input  wire                    weight_wr_bank,  // 0=BRAM A, 1=BRAM B

    output reg                      result_valid,
    output reg  [15:0]              result_pixel_id,
    output reg  [ITER_W-1:0]        result_iter,    // carries RGB565

    output wire                     busy
);

    localparam N_INPUT   = 3;
    localparam N_OUTPUT  = 3;
    // Total params: (3*H+H) + (H*H+H) + (H*3+3) where H=N_HIDDEN
    localparam N_WEIGHTS = N_INPUT * N_HIDDEN + N_HIDDEN
                         + N_HIDDEN * N_HIDDEN + N_HIDDEN
                         + N_HIDDEN * N_OUTPUT + N_OUTPUT;
    localparam WADDR_W   = (N_WEIGHTS <= 512)  ? 9  :
                           (N_WEIGHTS <= 1024) ? 10 :
                           (N_WEIGHTS <= 2048) ? 11 : 12;
    // Counter width: must hold values up to N_HIDDEN (inclusive)
    localparam CNT_W     = (N_HIDDEN <= 16) ? 5 :
                           (N_HIDDEN <= 32) ? 6 : 7;

    // =========================================================
    // Weight BRAMs — two sets for morphing, initialized at synthesis
    // =========================================================
    (* ram_style = "block" *) reg signed [WIDTH-1:0] weight_mem   [0:(1<<WADDR_W)-1];
    (* ram_style = "block" *) reg signed [WIDTH-1:0] weight_b_mem [0:(1<<WADDR_W)-1];

    `include "mlp_weights.vh"

    // BRAM read ports (port A): 1-cycle latency, same address for both banks
    reg  [WADDR_W-1:0] w_addr;
    reg  signed [WIDTH-1:0] w_data, wb_data;

    always @(posedge clk) begin
        w_data  <= weight_mem[w_addr];
        wb_data <= weight_b_mem[w_addr];
    end

    // BRAM write ports (port B): from PS via AXI broadcast
    always @(posedge clk) begin
        if (weight_wr_en && !weight_wr_bank)
            weight_mem[weight_wr_addr[WADDR_W-1:0]]   <= weight_wr_data;
        if (weight_wr_en && weight_wr_bank)
            weight_b_mem[weight_wr_addr[WADDR_W-1:0]] <= weight_wr_data;
    end

    // =========================================================
    // Blend pipeline (1 registered stage)
    // =========================================================
    // Computes: blended = w_a + ((w_b - w_a) * alpha) >> 8
    // Combinational blend path: BRAM output → sub → mul(32×9) → shift → add
    // Critical path ~11ns, meets 50 MHz (20ns period)
    wire signed [WIDTH-1:0] w_diff = wb_data - w_data;
    wire signed [8:0]       alpha_s = {1'b0, alpha};
    wire signed [40:0]      blend_prod = w_diff * alpha_s;
    wire signed [WIDTH-1:0] blended_w_comb = w_data + blend_prod[39:8];

    // Registered blend pipeline output
    reg signed [WIDTH-1:0] bp_weight;   // blended weight
    reg signed [WIDTH-1:0] bp_act;      // delayed activation (matches blend latency)
    reg                    bp_valid;    // delayed feeding flag

    // =========================================================
    // Layer geometry
    // =========================================================
    function [CNT_W-1:0] get_fan_in;
        input [1:0] layer;
        case (layer)
            2'd0: get_fan_in = N_INPUT;
            2'd1: get_fan_in = N_HIDDEN;
            2'd2: get_fan_in = N_HIDDEN;
            default: get_fan_in = 0;
        endcase
    endfunction

    function [CNT_W-1:0] get_fan_out;
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
            2'd1: get_w_base = N_INPUT * N_HIDDEN + N_HIDDEN;
            2'd2: get_w_base = N_INPUT * N_HIDDEN + N_HIDDEN
                             + N_HIDDEN * N_HIDDEN + N_HIDDEN;
            default: get_w_base = 0;
        endcase
    endfunction

    function [WADDR_W-1:0] get_b_base;
        input [1:0] layer;
        case (layer)
            2'd0: get_b_base = N_INPUT * N_HIDDEN;
            2'd1: get_b_base = N_INPUT * N_HIDDEN + N_HIDDEN
                             + N_HIDDEN * N_HIDDEN;
            2'd2: get_b_base = N_INPUT * N_HIDDEN + N_HIDDEN
                             + N_HIDDEN * N_HIDDEN + N_HIDDEN
                             + N_OUTPUT * N_HIDDEN;
            default: get_b_base = 0;
        endcase
    endfunction

    // =========================================================
    // FSM states
    // =========================================================
    localparam [3:0] S_IDLE      = 4'd0,
                     S_SETUP     = 4'd1,   // cache layer params, set first weight addr
                     S_DOT       = 4'd2,   // pipelined MAC (1 mul/cycle)
                     S_DOT_DRAIN = 4'd3,   // drain multiply pipeline
                     S_BIAS_RD   = 4'd4,   // issue bias BRAM read
                     S_BIAS_WAIT = 4'd5,   // wait for BRAM read
                     S_BIAS_ADD  = 4'd6,   // add bias to accumulator
                     S_ACTIVATE  = 4'd7,   // apply sin() activation
                     S_ACT_WAIT  = 4'd8,   // wait for sine_lut
                     S_NEXT_N    = 4'd9,   // advance to next neuron
                     S_NEXT_L    = 4'd10,  // advance to next layer
                     S_OUTPUT    = 4'd11;  // format RGB565

    reg [3:0] state;

    // =========================================================
    // Working registers
    // =========================================================
    reg [1:0]  cur_layer;
    reg [CNT_W-1:0]  cur_neuron;
    reg [CNT_W-1:0]  cur_k;
    reg [CNT_W-1:0]  cur_fan_in;
    reg [CNT_W-1:0]  cur_fan_out;
    reg [WADDR_W-1:0] cur_b_base;
    reg [WADDR_W-1:0] next_w_ptr;  // saved weight addr for next neuron
    reg [15:0] pid_r;

    // 36-bit accumulator (4 bits overflow headroom)
    reg signed [WIDTH+3:0] acc;

    // Pipeline counter
    reg [2:0] pipe_cnt;

    // Feeding flag: high when S_DOT is issuing weights to BRAMs
    reg feeding;

    // =========================================================
    // Activation register banks (double-buffered)
    // =========================================================
    reg signed [WIDTH-1:0] act_a [0:N_HIDDEN-1];
    reg signed [WIDTH-1:0] act_b [0:N_HIDDEN-1];
    reg bank_sel;  // 0: read A, write B. 1: read B, write A.

    // Read from current input bank (delayed cur_k aligns activation with
    // the blend pipeline: BRAM read + blend each add 1 cycle of latency,
    // and cur_k increments at the same time as the BRAM address, so the
    // activation index must be delayed 1 cycle to stay aligned with the
    // blended weight output.)
    reg [CNT_W-1:0] cur_k_d;
    always @(posedge clk) cur_k_d <= cur_k;
    wire signed [WIDTH-1:0] act_rd = bank_sel ? act_b[cur_k_d] : act_a[cur_k_d];

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
    // Blend pipeline drives MAC inputs
    // =========================================================
    // The blend pipeline registers (bp_weight, bp_act, bp_valid) are
    // updated every cycle. When bp_valid is high, feed the MAC.
    always @(posedge clk) begin
        bp_weight <= blended_w_comb;
        bp_act    <= act_rd;
        bp_valid  <= feeding;
    end

    // MAC input mux: blend pipeline output when valid, else from FSM (for future use)
    always @(*) begin
        if (bp_valid) begin
            mul_in_a     = bp_weight;
            mul_in_b     = bp_act;
            mul_valid_in = 1'b1;
        end else begin
            mul_in_a     = 0;
            mul_in_b     = 0;
            mul_valid_in = 1'b0;
        end
    end

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

    // Time: linear map [0, 1023] → [-1.0, +1.0] in Q4.28.
    // SIREN networks are trained with t ∈ [-1, +1], so we need:
    //   time_val = max_iter[9:0] * (2.0/1024) - 1.0
    // In Q4.28: shift left 19 bits (= multiply by 2^19 ≈ 2^28 * 2/1024),
    // then subtract 1.0 (= 0x10000000).
    //   max_iter=0    → 0x00000000 - 0x10000000 = 0xF0000000 = -1.0
    //   max_iter=512  → 0x10000000 - 0x10000000 = 0x00000000 =  0.0
    //   max_iter=1023 → 0x1FF80000 - 0x10000000 = 0x0FF80000 = +0.999
    wire signed [WIDTH-1:0] time_val = $signed({3'b000, max_iter[9:0], 19'b0}) - 32'sh10000000;

    // Saturate accumulator to Q4.28
    wire signed [WIDTH-1:0] acc_sat;
    assign acc_sat = (acc[WIDTH+3] && !(&acc[WIDTH+2:WIDTH-1])) ? {1'b1, {(WIDTH-1){1'b0}}} :
                     (!acc[WIDTH+3] && |acc[WIDTH+2:WIDTH-1]) ? {1'b0, {(WIDTH-1){1'b1}}} :
                     acc[WIDTH-1:0];

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
            cur_b_base      <= 0;
            next_w_ptr      <= 0;
            pid_r           <= 0;
            acc             <= 0;
            pipe_cnt        <= 0;
            bank_sel        <= 0;
            feeding         <= 0;
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
            feeding      <= 0;

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
                // S_SETUP: Cache layer params and issue first weight BRAM read.
                // This cycle serves as the BRAM fill cycle — both w_data and wb_data
                // will be valid next cycle, and the blend pipeline output 1 cycle after.
                S_SETUP: begin
                    cur_fan_in  <= get_fan_in(cur_layer);
                    cur_fan_out <= get_fan_out(cur_layer);
                    cur_b_base  <= get_b_base(cur_layer);
                    cur_neuron  <= 0;
                    cur_k       <= 0;
                    acc         <= 0;
                    w_addr      <= get_w_base(cur_layer);  // BRAM read starts
                    state       <= S_DOT;
                end

                // -------------------------------------------------
                // S_DOT: Issue BRAM addresses and track cur_k.
                // The blend pipeline (1 cycle later) feeds the MAC autonomously.
                // MAC results (3 cycles after blend feeds) are accumulated here.
                S_DOT: begin
                    if (cur_k < cur_fan_in) begin
                        // Signal blend pipeline: valid data arriving from BRAM
                        feeding <= 1'b1;
                        // Advance weight address for next cycle's BRAM read
                        w_addr <= w_addr + 1;
                        cur_k  <= cur_k + 1;
                    end else begin
                        // All inputs issued — save weight pointer and drain
                        // Note: blend pipeline will output last result next cycle
                        // (bp_valid stays high 1 more cycle from delayed feeding)
                        next_w_ptr <= w_addr;
                        pipe_cnt   <= 0;
                        state      <= S_DOT_DRAIN;
                    end

                    // Accumulate results from multiply pipeline
                    if (mul_valid_out)
                        acc <= acc + {{4{mul_result[WIDTH-1]}}, mul_result};
                end

                // -------------------------------------------------
                // S_DOT_DRAIN: Wait for blend pipeline + MAC pipeline to flush.
                // Cycle 0: blend pipeline outputs last weight → MAC starts.
                // Cycles 1-3: MAC pipeline drains.
                S_DOT_DRAIN: begin
                    if (mul_valid_out)
                        acc <= acc + {{4{mul_result[WIDTH-1]}}, mul_result};

                    if (pipe_cnt == 3'd3) begin
                        // Pipeline fully drained — read bias
                        w_addr <= cur_b_base + cur_neuron;
                        state  <= S_BIAS_RD;
                    end else begin
                        pipe_cnt <= pipe_cnt + 1;
                    end
                end

                // -------------------------------------------------
                S_BIAS_RD: begin
                    // BRAM read latency cycle (bias address set in DRAIN)
                    state <= S_BIAS_WAIT;
                end

                // -------------------------------------------------
                S_BIAS_WAIT: begin
                    // w_data and wb_data now have bias values from both BRAMs.
                    // blended_w_comb has the blended bias (combinational).
                    state <= S_BIAS_ADD;
                end

                // -------------------------------------------------
                S_BIAS_ADD: begin
                    // Use registered blend pipeline output (bp_weight) which
                    // captured the blended bias during BIAS_WAIT.
                    acc   <= acc + {{4{bp_weight[WIDTH-1]}}, bp_weight};
                    state <= S_ACTIVATE;
                end

                // -------------------------------------------------
                S_ACTIVATE: begin
                    sin_input <= acc_sat;
                    pipe_cnt  <= 0;
                    state     <= S_ACT_WAIT;
                end

                // -------------------------------------------------
                S_ACT_WAIT: begin
                    // sine_lut has 2 registered stages. sin_input is set at
                    // posedge A (S_ACTIVATE). Stage 1 latches at posedge A+1,
                    // stage 2 latches result at posedge A+2. But since we read
                    // result as a registered output, we see it at posedge A+3.
                    // So we need 3 wait cycles (pipe_cnt: 0, 1, 2).
                    if (pipe_cnt == 2'd2) begin
                        // Store activated value to output bank
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
                            2'd0: out_r <= sin_output;
                            2'd1: out_g <= sin_output;
                            2'd2: out_b <= sin_output;
                            default: ;
                        endcase
                    end

                    if (cur_neuron + 1 < cur_fan_out) begin
                        // More neurons in this layer
                        cur_neuron <= cur_neuron + 1;
                        cur_k      <= 0;
                        acc        <= 0;
                        // Restore weight pointer for next neuron.
                        // Also serves as BRAM fill cycle.
                        w_addr     <= next_w_ptr;
                        state      <= S_DOT;
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
            if (rs[WIDTH-1])          r5 = 5'd0;
            else if (rs >= 32'sh2000_0000) r5 = 5'd31;
            else                      r5 = rs[28:24];

            // 6 bits for G: [0, 2.0) → [0, 63]
            if (gs[WIDTH-1])          g6 = 6'd0;
            else if (gs >= 32'sh2000_0000) g6 = 6'd63;
            else                      g6 = gs[28:23];

            // 5 bits for B
            if (bs[WIDTH-1])          b5 = 5'd0;
            else if (bs >= 32'sh2000_0000) b5 = 5'd31;
            else                      b5 = bs[28:24];

            pack_rgb565 = {r5, g6, b5};
        end
    endfunction

endmodule
