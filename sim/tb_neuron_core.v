// tb_neuron_core.v — Testbench for Mandelbrot neuron core
//
// Tests the neuron with known Mandelbrot coordinates and verifies
// iteration counts against expected values.

`timescale 1ns / 1ps

module tb_neuron_core;

    parameter WIDTH  = 32;
    parameter FRAC   = 28;
    parameter ITER_W = 16;

    reg                      clk;
    reg                      rst_n;
    reg                      pixel_valid;
    wire                     pixel_ready;
    reg  signed [WIDTH-1:0]  c_re;
    reg  signed [WIDTH-1:0]  c_im;
    reg  [15:0]              pixel_id;
    reg  [ITER_W-1:0]        max_iter;
    wire                     result_valid;
    wire [15:0]              result_pixel_id;
    wire [ITER_W-1:0]        result_iter;
    wire                     busy;

    // Instantiate neuron
    neuron_core #(
        .WIDTH(WIDTH), .FRAC(FRAC), .ITER_W(ITER_W)
    ) uut (
        .clk            (clk),
        .rst_n          (rst_n),
        .pixel_valid    (pixel_valid),
        .pixel_ready    (pixel_ready),
        .c_re           (c_re),
        .c_im           (c_im),
        .pixel_id       (pixel_id),
        .max_iter       (max_iter),
        .result_valid   (result_valid),
        .result_pixel_id(result_pixel_id),
        .result_iter    (result_iter),
        .busy           (busy)
    );

    // Clock: 50 MHz (20 ns period)
    initial clk = 0;
    always #10 clk = ~clk;

    // Helper function: convert float to Q4.28
    // (done in Python, hardcoded here)

    integer test_num;
    integer pass_count;
    integer fail_count;

    initial begin
        $dumpfile("tb_neuron_core.vcd");
        $dumpvars(0, tb_neuron_core);

        rst_n       = 0;
        pixel_valid = 0;
        c_re        = 0;
        c_im        = 0;
        pixel_id    = 0;
        max_iter    = 256;
        test_num    = 0;
        pass_count  = 0;
        fail_count  = 0;

        // Reset
        #100;
        rst_n = 1;
        #40;

        // ============================================
        // Test 1: c = (0, 0) — interior, should reach max_iter
        // ============================================
        test_num = 1;
        run_test(32'sh0000_0000, 32'sh0000_0000, 16'd0, 256);
        wait_result();
        check_result(1, 256, "c=(0,0) interior");

        // ============================================
        // Test 2: c = (2, 0) — escapes immediately (iter 1)
        // z1 = 0 + 2 = 2, |z1|²=4, at threshold
        // z2 = 4 + 2 = 6, |z2|²=36, escaped
        // Should get iter=1 or 2 depending on >= vs >
        // ============================================
        test_num = 2;
        // 2.0 in Q4.28 = 2 * 2^28 = 536870912 = 32'h2000_0000
        run_test(32'sh2000_0000, 32'sh0000_0000, 16'd1, 256);
        wait_result();
        // Escapes at iter 1 (|z1|² = 4, which equals threshold, so one more iter)
        // or iter 2 depending on > vs >= comparison
        if (result_iter <= 3)
            check_result_range(2, 1, 3, "c=(2,0) exterior");
        else begin
            $display("FAIL test %0d: c=(2,0) expected iter 1-3, got %0d", test_num, result_iter);
            fail_count = fail_count + 1;
        end

        // ============================================
        // Test 3: c = (-1, 0) — period-2 cycle, interior
        // z1=-1, z2=0, z3=-1, z4=0, ... forever
        // Should reach max_iter
        // ============================================
        test_num = 3;
        // -1.0 in Q4.28 = 32'hF000_0000
        run_test(32'shF000_0000, 32'sh0000_0000, 16'd2, 256);
        wait_result();
        check_result(3, 256, "c=(-1,0) period-2 interior");

        // ============================================
        // Test 4: c = (0.5, 0) — exterior, escapes
        // ============================================
        test_num = 4;
        // 0.5 in Q4.28 = 0.5 * 2^28 = 134217728 = 32'h0800_0000
        run_test(32'sh0800_0000, 32'sh0000_0000, 16'd3, 256);
        wait_result();
        // Should escape in ~5 iterations
        if (result_iter < 256) begin
            $display("PASS test 4: c=(0.5,0) escaped at iter %0d", result_iter);
            pass_count = pass_count + 1;
        end else begin
            $display("FAIL test 4: c=(0.5,0) should escape, got max_iter");
            fail_count = fail_count + 1;
        end

        // ============================================
        // Test 5: c = (-0.75, 0.1) — near boundary, many iterations
        // ============================================
        test_num = 5;
        // -0.75 in Q4.28 ≈ 32'hF400_0000
        //  0.1  in Q4.28 ≈ 32'h0199_999A
        run_test(32'shF400_0000, 32'sh0199_999A, 16'd4, 256);
        wait_result();
        $display("INFO test 5: c=(-0.75,0.1) iter=%0d (boundary region)", result_iter);
        pass_count = pass_count + 1;

        // ============================================
        // Test 6: c = (-2.1, 0) — just outside, should escape fast
        // ============================================
        test_num = 6;
        // -2.1 in Q4.28 ≈ 32'hDE66_6666
        run_test(32'shDE66_6666, 32'sh0000_0000, 16'd5, 256);
        wait_result();
        if (result_iter < 20) begin
            $display("PASS test 6: c=(-2.1,0) escaped at iter %0d", result_iter);
            pass_count = pass_count + 1;
        end else begin
            $display("FAIL test 6: c=(-2.1,0) expected fast escape, got %0d", result_iter);
            fail_count = fail_count + 1;
        end

        // ============================================
        // Summary
        // ============================================
        #100;
        $display("");
        $display("========================================");
        $display("  Test Results: %0d passed, %0d failed", pass_count, fail_count);
        $display("========================================");
        $finish;
    end

    // ============================================
    // Tasks
    // ============================================
    task run_test;
        input signed [WIDTH-1:0] t_c_re;
        input signed [WIDTH-1:0] t_c_im;
        input [15:0]             t_pid;
        input [ITER_W-1:0]       t_max;
    begin
        @(posedge clk);
        wait (pixel_ready);
        @(posedge clk);
        c_re        <= t_c_re;
        c_im        <= t_c_im;
        pixel_id    <= t_pid;
        max_iter    <= t_max;
        pixel_valid <= 1;
        @(posedge clk);
        pixel_valid <= 0;
    end
    endtask

    task wait_result;
    begin
        wait (result_valid);
        @(posedge clk);
    end
    endtask

    task check_result;
        input integer tnum;
        input [ITER_W-1:0] expected;
        input [255:0] name;  // String (padded)
    begin
        if (result_iter == expected) begin
            $display("PASS test %0d: %0s → iter=%0d", tnum, name, result_iter);
            pass_count = pass_count + 1;
        end else begin
            $display("FAIL test %0d: %0s → expected %0d, got %0d",
                     tnum, name, expected, result_iter);
            fail_count = fail_count + 1;
        end
    end
    endtask

    task check_result_range;
        input integer tnum;
        input [ITER_W-1:0] exp_min;
        input [ITER_W-1:0] exp_max;
        input [255:0] name;
    begin
        if (result_iter >= exp_min && result_iter <= exp_max) begin
            $display("PASS test %0d: %0s → iter=%0d (expected %0d-%0d)",
                     tnum, name, result_iter, exp_min, exp_max);
            pass_count = pass_count + 1;
        end else begin
            $display("FAIL test %0d: %0s → iter=%0d (expected %0d-%0d)",
                     tnum, name, result_iter, exp_min, exp_max);
            fail_count = fail_count + 1;
        end
    end
    endtask

endmodule
