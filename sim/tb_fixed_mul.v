// tb_fixed_mul.v — Testbench for Q4.28 fixed-point multiplier
//
// Verifies multiply results against known products.

`timescale 1ns / 1ps

module tb_fixed_mul;

    parameter WIDTH = 32;
    parameter FRAC  = 28;

    reg                     clk;
    reg                     rst_n;
    reg  signed [WIDTH-1:0] a, b;
    reg                     valid_in;
    wire signed [WIDTH-1:0] result;
    wire                    valid_out;

    fixed_mul #(.WIDTH(WIDTH), .FRAC(FRAC)) uut (
        .clk(clk), .rst_n(rst_n),
        .a(a), .b(b),
        .valid_in(valid_in),
        .result(result),
        .valid_out(valid_out)
    );

    // Clock: 50 MHz
    initial clk = 0;
    always #10 clk = ~clk;

    integer pass_count, fail_count;

    initial begin
        $dumpfile("tb_fixed_mul.vcd");
        $dumpvars(0, tb_fixed_mul);

        rst_n      = 0;
        a          = 0;
        b          = 0;
        valid_in   = 0;
        pass_count = 0;
        fail_count = 0;

        #100;
        rst_n = 1;
        #40;

        // ============================================
        // Test 1: 1.0 * 1.0 = 1.0
        // 1.0 in Q4.28 = 32'h1000_0000
        // ============================================
        run_mul(32'sh1000_0000, 32'sh1000_0000);
        check(32'sh1000_0000, "1.0 * 1.0");

        // ============================================
        // Test 2: 2.0 * 2.0 = 4.0
        // 2.0 = 32'h2000_0000, 4.0 = 32'h4000_0000
        // ============================================
        run_mul(32'sh2000_0000, 32'sh2000_0000);
        check(32'sh4000_0000, "2.0 * 2.0");

        // ============================================
        // Test 3: -1.0 * 1.0 = -1.0
        // -1.0 = 32'hF000_0000
        // ============================================
        run_mul(32'shF000_0000, 32'sh1000_0000);
        check(32'shF000_0000, "-1.0 * 1.0");

        // ============================================
        // Test 4: 0.5 * 0.5 = 0.25
        // 0.5 = 32'h0800_0000, 0.25 = 32'h0400_0000
        // ============================================
        run_mul(32'sh0800_0000, 32'sh0800_0000);
        check(32'sh0400_0000, "0.5 * 0.5");

        // ============================================
        // Test 5: -1.5 * -1.5 = 2.25
        // -1.5 = 32'hE800_0000, 2.25 = 32'h2400_0000
        // ============================================
        run_mul(32'shE800_0000, 32'shE800_0000);
        check(32'sh2400_0000, "-1.5 * -1.5");

        // ============================================
        // Test 6: 0.0 * anything = 0.0
        // ============================================
        run_mul(32'sh0000_0000, 32'sh1234_5678);
        check(32'sh0000_0000, "0.0 * x");

        // ============================================
        // Test 7: Small values — 0.001 * 0.001 ≈ 0.000001
        // 0.001 in Q4.28 ≈ 268435 = 32'h0004_189D (approx)
        // 0.000001 ≈ 268 = very small
        // ============================================
        run_mul(32'sh0004_189D, 32'sh0004_189D);
        // Just check it's small and positive
        wait_valid();
        if (result > 0 && result < 32'sh0000_0200) begin
            $display("PASS: 0.001 * 0.001 = small positive (%0h)", result);
            pass_count = pass_count + 1;
        end else begin
            $display("FAIL: 0.001 * 0.001 expected small positive, got %0h", result);
            fail_count = fail_count + 1;
        end

        // ============================================
        // Summary
        // ============================================
        #100;
        $display("");
        $display("========================================");
        $display("  Fixed Mul Tests: %0d passed, %0d failed", pass_count, fail_count);
        $display("========================================");
        $finish;
    end

    // ============================================
    // Tasks
    // ============================================
    task run_mul;
        input signed [WIDTH-1:0] ta, tb;
    begin
        @(posedge clk);
        a        <= ta;
        b        <= tb;
        valid_in <= 1;
        @(posedge clk);
        valid_in <= 0;
    end
    endtask

    task wait_valid;
    begin
        wait (valid_out);
        @(posedge clk);
    end
    endtask

    task check;
        input signed [WIDTH-1:0] expected;
        input [255:0] name;
    begin
        wait_valid();
        // Allow ±1 LSB tolerance for rounding
        if (result == expected || result == expected + 1 || result == expected - 1) begin
            $display("PASS: %0s = %0h (expected %0h)", name, result, expected);
            pass_count = pass_count + 1;
        end else begin
            $display("FAIL: %0s = %0h (expected %0h)", name, result, expected);
            fail_count = fail_count + 1;
        end
    end
    endtask

endmodule
