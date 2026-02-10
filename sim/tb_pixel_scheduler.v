// tb_pixel_scheduler.v — Pixel scheduler unit test
//
// Instantiates pixel_scheduler with stub neurons at small resolution
// (16x8 = 128 pixels, 4 neurons) for fast simulation.
//
// Test cases:
//   1. Pixel ID uniqueness: every pixel_id 0..127 written exactly once
//   2. Row-start alignment: first pixel of each row has c_re == c_re_start
//   3. Solid-color uniformity: all stubs return iter=42, verify all fb_wr_data == 42
//   4. Frame completion: frame_done pulses exactly once

`timescale 1ns / 1ps

module tb_pixel_scheduler;

    // =========================================================
    // Parameters — small resolution for fast sim
    // =========================================================
    localparam N_NEURONS = 4;
    localparam WIDTH     = 32;
    localparam FRAC      = 28;
    localparam ITER_W    = 16;
    localparam H_RES     = 16;
    localparam V_RES     = 8;
    localparam PIX_COUNT = H_RES * V_RES;  // 128

    localparam STUB_DELAY = 8;   // Cycles each stub takes to "compute"
    localparam STUB_ITER  = 42;  // Fixed iteration count returned by stubs

    // =========================================================
    // Clock and reset
    // =========================================================
    reg clk;
    reg rst_n;

    initial clk = 0;
    always #5 clk = ~clk;  // 100 MHz

    // =========================================================
    // DUT signals
    // =========================================================
    reg                      frame_start;
    wire                     frame_busy;
    wire                     frame_done;

    // Viewport parameters (arbitrary but deterministic for testing)
    // c_re_start = -2.0, c_im_start = -1.0, step = 0.25
    localparam signed [WIDTH-1:0] CRE_START = 32'shE000_0000;  // -2.0
    localparam signed [WIDTH-1:0] CIM_START = 32'shF000_0000;  // -1.0
    localparam signed [WIDTH-1:0] CRE_STEP  = 32'sh0400_0000;  // 0.25
    localparam signed [WIDTH-1:0] CIM_STEP  = 32'sh0400_0000;  // 0.25

    reg signed [WIDTH-1:0]  c_re_start_r;
    reg signed [WIDTH-1:0]  c_im_start_r;
    reg signed [WIDTH-1:0]  c_re_step_r;
    reg signed [WIDTH-1:0]  c_im_step_r;
    reg [ITER_W-1:0]        max_iter_r;

    // Scheduler ↔ Neuron interface
    wire [N_NEURONS-1:0]     neuron_valid;
    wire [N_NEURONS-1:0]     neuron_ready;
    wire signed [WIDTH-1:0]  neuron_c_re;
    wire signed [WIDTH-1:0]  neuron_c_im;
    wire [15:0]              neuron_pixel_id;

    wire [N_NEURONS-1:0]         result_valid;
    wire [N_NEURONS*16-1:0]      result_pixel_id;
    wire [N_NEURONS*ITER_W-1:0]  result_iter;

    // Framebuffer write port
    wire                     fb_wr_en;
    wire [15:0]              fb_wr_addr;
    wire [ITER_W-1:0]        fb_wr_data;

    // =========================================================
    // DUT: pixel_scheduler
    // =========================================================
    pixel_scheduler #(
        .N_NEURONS(N_NEURONS), .WIDTH(WIDTH), .FRAC(FRAC),
        .ITER_W(ITER_W), .H_RES(H_RES), .V_RES(V_RES)
    ) u_dut (
        .clk            (clk),
        .rst_n          (rst_n),
        .frame_start    (frame_start),
        .frame_busy     (frame_busy),
        .frame_done     (frame_done),
        .c_re_start     (c_re_start_r),
        .c_im_start     (c_im_start_r),
        .c_re_step      (c_re_step_r),
        .c_im_step      (c_im_step_r),
        .max_iter       (max_iter_r),
        .neuron_valid   (neuron_valid),
        .neuron_ready   (neuron_ready),
        .neuron_c_re    (neuron_c_re),
        .neuron_c_im    (neuron_c_im),
        .neuron_pixel_id(neuron_pixel_id),
        .result_valid   (result_valid),
        .result_pixel_id(result_pixel_id),
        .result_iter    (result_iter),
        .fb_wr_en       (fb_wr_en),
        .fb_wr_addr     (fb_wr_addr),
        .fb_wr_data     (fb_wr_data)
    );

    // =========================================================
    // Stub neurons — configurable delay, fixed iter return
    // =========================================================
    genvar gi;
    generate
        for (gi = 0; gi < N_NEURONS; gi = gi + 1) begin : stubs
            reg [3:0]          delay_cnt;
            reg                busy;
            reg [15:0]         pid_r;
            reg [ITER_W-1:0]  iter_r;
            reg                rvalid;
            reg [15:0]         rpid;
            reg [ITER_W-1:0]  riter;

            assign neuron_ready[gi]                          = !busy;
            assign result_valid[gi]                          = rvalid;
            assign result_pixel_id[gi*16 +: 16]             = rpid;
            assign result_iter[gi*ITER_W +: ITER_W]         = riter;

            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    busy      <= 0;
                    delay_cnt <= 0;
                    pid_r     <= 0;
                    iter_r    <= 0;
                    rvalid    <= 0;
                    rpid      <= 0;
                    riter     <= 0;
                end else begin
                    rvalid <= 0;

                    if (!busy && neuron_valid[gi]) begin
                        busy      <= 1;
                        delay_cnt <= 0;
                        pid_r     <= neuron_pixel_id;
                        iter_r    <= STUB_ITER;
                    end else if (busy) begin
                        if (delay_cnt == STUB_DELAY - 1) begin
                            busy   <= 0;
                            rvalid <= 1;
                            rpid   <= pid_r;
                            riter  <= iter_r;
                        end else begin
                            delay_cnt <= delay_cnt + 1;
                        end
                    end
                end
            end
        end
    endgenerate

    // =========================================================
    // Scoreboard — track pixel writes
    // =========================================================
    reg [7:0] pixel_written [0:PIX_COUNT-1];  // Write count per pixel
    reg [ITER_W-1:0] pixel_data [0:PIX_COUNT-1];  // Data written
    integer total_writes;
    integer frame_done_count;

    // Capture row-start c_re values: when scheduler assigns pixel with px==0
    // We monitor neuron_valid (any bit set) and check if the scheduler's px==0.
    // Since the scheduler shares coordinates on the bus, we capture neuron_c_re
    // when any neuron_valid fires and pixel_id is a multiple of H_RES.
    reg signed [WIDTH-1:0] row_start_cre [0:V_RES-1];
    reg [7:0] row_start_captured;

    // =========================================================
    // Monitor writes and captures
    // =========================================================
    integer wi;
    always @(posedge clk) begin
        if (fb_wr_en) begin
            if (fb_wr_addr < PIX_COUNT) begin
                pixel_written[fb_wr_addr] = pixel_written[fb_wr_addr] + 1;
                pixel_data[fb_wr_addr]    = fb_wr_data;
                total_writes = total_writes + 1;
            end
        end
        if (frame_done) begin
            frame_done_count = frame_done_count + 1;
        end

        // Capture row-start c_re: when any neuron is assigned a pixel at start of row
        if (|neuron_valid && (neuron_pixel_id % H_RES == 0)) begin
            wi = neuron_pixel_id / H_RES;
            if (wi < V_RES) begin
                row_start_cre[wi] = neuron_c_re;
                row_start_captured = row_start_captured + 1;
            end
        end
    end

    // =========================================================
    // Test sequence
    // =========================================================
    integer j, errors;
    reg signed [WIDTH-1:0] expected_cre;

    initial begin
        $display("=== tb_pixel_scheduler: Scheduler unit test ===");
        $display("  Resolution: %0dx%0d = %0d pixels, %0d neurons, stub delay=%0d",
                 H_RES, V_RES, PIX_COUNT, N_NEURONS, STUB_DELAY);

        // Initialize
        rst_n       = 0;
        frame_start = 0;
        c_re_start_r = CRE_START;
        c_im_start_r = CIM_START;
        c_re_step_r  = CRE_STEP;
        c_im_step_r  = CIM_STEP;
        max_iter_r   = 256;
        total_writes = 0;
        frame_done_count = 0;
        row_start_captured = 0;
        errors = 0;

        for (j = 0; j < PIX_COUNT; j = j + 1) begin
            pixel_written[j] = 0;
            pixel_data[j]    = 0;
        end
        for (j = 0; j < V_RES; j = j + 1) begin
            row_start_cre[j] = 0;
        end

        // Release reset
        repeat (5) @(posedge clk);
        rst_n = 1;
        repeat (2) @(posedge clk);

        // Start frame
        @(posedge clk);
        frame_start = 1;
        @(posedge clk);
        frame_start = 0;

        // Wait for frame completion (timeout after generous limit)
        begin : wait_block
            integer timeout;
            timeout = 0;
            while (!frame_done && timeout < 100000) begin
                @(posedge clk);
                timeout = timeout + 1;
            end
            if (timeout >= 100000) begin
                $display("FAIL: Timeout waiting for frame_done");
                errors = errors + 1;
            end
        end

        // Let any final writes drain
        repeat (10) @(posedge clk);

        // ---- Test 1: Pixel ID uniqueness ----
        $display("\n--- Test 1: Pixel ID uniqueness ---");
        for (j = 0; j < PIX_COUNT; j = j + 1) begin
            if (pixel_written[j] != 1) begin
                if (errors < 10)
                    $display("FAIL: pixel[%0d] written %0d times (expected 1)", j, pixel_written[j]);
                errors = errors + 1;
            end
        end
        if (total_writes == PIX_COUNT)
            $display("PASS: All %0d pixels written exactly once", PIX_COUNT);
        else begin
            $display("FAIL: total_writes=%0d, expected %0d", total_writes, PIX_COUNT);
            errors = errors + 1;
        end

        // ---- Test 2: Row-start alignment (Bug #1 regression) ----
        $display("\n--- Test 2: Row-start c_re alignment ---");
        for (j = 0; j < V_RES; j = j + 1) begin
            expected_cre = CRE_START;  // All rows should start at c_re_start
            if (row_start_cre[j] !== expected_cre) begin
                $display("FAIL: row %0d c_re = %08h, expected %08h",
                         j, row_start_cre[j], expected_cre);
                errors = errors + 1;
            end
        end
        if (row_start_captured == V_RES)
            $display("PASS: All %0d row starts captured with correct c_re", V_RES);
        else begin
            $display("FAIL: captured %0d row starts, expected %0d", row_start_captured, V_RES);
            errors = errors + 1;
        end

        // ---- Test 3: Solid-color uniformity ----
        $display("\n--- Test 3: Solid-color uniformity (iter=%0d) ---", STUB_ITER);
        for (j = 0; j < PIX_COUNT; j = j + 1) begin
            if (pixel_data[j] != STUB_ITER) begin
                if (errors < 20)
                    $display("FAIL: pixel[%0d] data=%0d, expected %0d", j, pixel_data[j], STUB_ITER);
                errors = errors + 1;
            end
        end
        if (errors == 0)
            $display("PASS: All %0d pixels have iter=%0d", PIX_COUNT, STUB_ITER);

        // ---- Test 4: Frame completion ----
        $display("\n--- Test 4: Frame completion ---");
        if (frame_done_count == 1)
            $display("PASS: frame_done pulsed exactly once");
        else begin
            $display("FAIL: frame_done_count=%0d, expected 1", frame_done_count);
            errors = errors + 1;
        end

        // ---- Summary ----
        $display("\n=== SUMMARY: %0d errors ===", errors);
        if (errors == 0)
            $display("ALL TESTS PASSED");
        else
            $display("TESTS FAILED");

        $finish;
    end

    // Timeout safety
    initial begin
        #2000000;
        $display("FAIL: Global timeout (2ms)");
        $finish;
    end

endmodule
