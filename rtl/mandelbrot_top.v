// mandelbrot_top.v — Top-level neural Mandelbrot explorer
//
// Instantiates:
//   - N neuron_core instances (parallel z=z²+c iterators)
//   - pixel_scheduler (work distribution)
//   - colormap_lut (iteration → RGB565)
//   - sp2_spi_driver (display output)
//   - Dual-port framebuffer (BRAM)
//   - Clock generation (MMCM)
//
// AXI-Lite slave interface for ARM PS control (Phase 4)
// For Phase 1-2, viewport is hardcoded to default Mandelbrot view.

`timescale 1ns / 1ps

module mandelbrot_top #(
    parameter N_NEURONS = 18,     // 18 × 3 muls × 4 DSPs = 216 of 220 DSP48E1
    parameter WIDTH     = 32,
    parameter FRAC      = 28,
    parameter ITER_W    = 16,
    parameter H_RES     = 320,
    parameter V_RES     = 172,
    parameter PIX_COUNT = H_RES * V_RES,
    parameter TEST_MODE    = 0,   // 0=normal, 1=solid color (iter=42), 2=gradient (iter=pixel_id[15:8])
    parameter COMPUTE_MODE = 0    // 0=Mandelbrot (neuron_core), 1=MLP inference (mlp_core)
)(
    input  wire        clk_50m,       // 50 MHz from PL crystal
    input  wire        rst_n_in,      // External reset (active low)

    // SP2 display SPI pins
    output wire        spi_cs_n,
    output wire        spi_sck,
    output wire        spi_mosi,
    output wire        spi_dc,
    output wire        lcd_rst_n,
    output wire        lcd_blk_out,   // Backlight enable (active high)

    // Debug
    output wire        led_frame,     // Toggles each frame
    output wire        led_alive,     // 1Hz blink — proves FPGA is running

    // UART
    output wire        uart_tx        // 115200 8N1 boot message
);

    // =========================================================
    // Clock generation
    // =========================================================
    // For now, use 50 MHz directly for both compute and SPI.
    // Phase 3+: add MMCM to generate 150 MHz compute clock.
    wire clk = clk_50m;
    wire rst_n;

    // Synchronize reset
    reg [2:0] rst_sync;
    always @(posedge clk or negedge rst_n_in) begin
        if (!rst_n_in)
            rst_sync <= 3'b000;
        else
            rst_sync <= {rst_sync[1:0], 1'b1};
    end
    assign rst_n = rst_sync[2];

    // =========================================================
    // Default viewport (full Mandelbrot set)
    // =========================================================
    // View: center (-0.5, 0), width 3.0
    // In Q4.28:
    //   c_re_start = -2.0 = 32'hE000_0000
    //   c_im_start = -1.0 * (V_RES/H_RES) * 1.5 = approx -0.80625
    //              ≈ 32'hF322_D0E5  (close enough)
    //   c_re_step  = 3.0 / 320 = 0.009375 ≈ 32'h0026_6666
    //   c_im_step  = 3.0 / 320 * (same pixel aspect) = 0.009375 ≈ 32'h0026_6666
    //
    // Recalculating carefully:
    //   3.0 / 320 = 0.009375
    //   0.009375 * 2^28 = 0.009375 * 268435456 = 2516582.4 ≈ 32'h00266666
    //   -2.0 * 2^28 = -536870912 = 32'hE0000000
    //   c_im range = 172 * 0.009375 = 1.6125
    //   c_im_start = -1.6125/2 = -0.80625
    //   -0.80625 * 2^28 = -216426086 = 32'hF319999A

    localparam signed [WIDTH-1:0] DEFAULT_CRE_START = 32'shE000_0000;  // -2.0
    localparam signed [WIDTH-1:0] DEFAULT_CIM_START = 32'shF319_999A;  // -0.80625
    localparam signed [WIDTH-1:0] DEFAULT_CRE_STEP  = 32'sh0026_6666;  // 0.009375
    localparam signed [WIDTH-1:0] DEFAULT_CIM_STEP  = 32'sh0026_6666;  // 0.009375
    localparam [ITER_W-1:0]       DEFAULT_MAX_ITER  = 256;

    // =========================================================
    // MLP mode: normalized coordinate viewport [-1, +1]
    // =========================================================
    // x: -1.0 to +1.0 across 320 pixels → step = 2.0/320
    // y: -1.0 to +1.0 across 172 pixels → step = 2.0/172
    // In Q4.28:
    //   -1.0 = 0xF0000000
    //   2.0/320 = 0.00625 → 0.00625 * 2^28 = 1677721.6 ≈ 0x0019999A
    //   2.0/172 = 0.011628 → 0.011628 * 2^28 = 3122165.9 ≈ 0x002FA0BE
    localparam signed [WIDTH-1:0] MLP_CRE_START = 32'shF000_0000;  // -1.0
    localparam signed [WIDTH-1:0] MLP_CIM_START = 32'shF000_0000;  // -1.0
    localparam signed [WIDTH-1:0] MLP_CRE_STEP  = 32'sh0019_999A;  // 2.0/320
    localparam signed [WIDTH-1:0] MLP_CIM_STEP  = 32'sh002F_A0BE;  // 2.0/172

    // =========================================================
    // Wires between modules
    // =========================================================

    // Scheduler ↔ Neurons
    wire [N_NEURONS-1:0]     neuron_valid_w;
    wire [N_NEURONS-1:0]     neuron_ready_w;
    wire signed [WIDTH-1:0]  neuron_c_re_w;
    wire signed [WIDTH-1:0]  neuron_c_im_w;
    wire [15:0]              neuron_pixel_id_w;
    reg  [ITER_W-1:0]        max_iter_w;

    // Neuron results — flat buses for Verilog-2001 compatibility
    wire [N_NEURONS-1:0]         result_valid_w;
    wire [N_NEURONS*16-1:0]      result_pixel_id_w;
    wire [N_NEURONS*ITER_W-1:0]  result_iter_w;

    // Scheduler → Framebuffer (iteration counts)
    wire                     fb_iter_wr_en;
    wire [15:0]              fb_iter_wr_addr;
    wire [ITER_W-1:0]        fb_iter_wr_data;

    // Frame control
    reg                      frame_start_r;
    wire                     frame_busy_w;
    wire                     frame_done_w;

    // Colormap → Display framebuffer
    wire [15:0]              fb_disp_addr;
    wire [15:0]              fb_disp_data;

    // Display frame done (from SPI driver — unused, compute free-runs)
    wire                     disp_frame_done;

    // Viewport registers (driven by zoom/MLP controller below)
    reg signed [WIDTH-1:0] zoom_step;       // c_re step (and c_im step in Mandelbrot mode)
    reg signed [WIDTH-1:0] zoom_cim_step;   // c_im step (separate for MLP mode aspect ratio)
    reg signed [WIDTH-1:0] zoom_cre_start;
    reg signed [WIDTH-1:0] zoom_cim_start;
    reg startup;

    // =========================================================
    // Neuron pool + Pixel scheduler (or test-mode substitute)
    // =========================================================
    generate
        if (TEST_MODE == 0) begin : gen_normal
            // --- Production: core pool + scheduler ---
            genvar i;
            for (i = 0; i < N_NEURONS; i = i + 1) begin : neurons
                if (COMPUTE_MODE == 0) begin : gen_mandelbrot
                    neuron_core #(
                        .WIDTH(WIDTH), .FRAC(FRAC), .ITER_W(ITER_W)
                    ) u_neuron (
                        .clk            (clk),
                        .rst_n          (rst_n),
                        .pixel_valid    (neuron_valid_w[i]),
                        .pixel_ready    (neuron_ready_w[i]),
                        .c_re           (neuron_c_re_w),
                        .c_im           (neuron_c_im_w),
                        .pixel_id       (neuron_pixel_id_w),
                        .max_iter       (max_iter_w),
                        .result_valid   (result_valid_w[i]),
                        .result_pixel_id(result_pixel_id_w[i*16 +: 16]),
                        .result_iter    (result_iter_w[i*ITER_W +: ITER_W]),
                        .busy           ()
                    );
                end else begin : gen_mlp
                    mlp_core #(
                        .WIDTH(WIDTH), .FRAC(FRAC), .ITER_W(ITER_W),
                        .N_HIDDEN(16), .N_LAYERS(3)
                    ) u_mlp (
                        .clk            (clk),
                        .rst_n          (rst_n),
                        .pixel_valid    (neuron_valid_w[i]),
                        .pixel_ready    (neuron_ready_w[i]),
                        .c_re           (neuron_c_re_w),
                        .c_im           (neuron_c_im_w),
                        .pixel_id       (neuron_pixel_id_w),
                        .max_iter       (max_iter_w),
                        .result_valid   (result_valid_w[i]),
                        .result_pixel_id(result_pixel_id_w[i*16 +: 16]),
                        .result_iter    (result_iter_w[i*ITER_W +: ITER_W]),
                        .busy           ()
                    );
                end
            end

            pixel_scheduler #(
                .N_NEURONS(N_NEURONS), .WIDTH(WIDTH), .FRAC(FRAC),
                .ITER_W(ITER_W), .H_RES(H_RES), .V_RES(V_RES)
            ) u_scheduler (
                .clk            (clk),
                .rst_n          (rst_n),
                .frame_start    (frame_start_r),
                .frame_busy     (frame_busy_w),
                .frame_done     (frame_done_w),
                .c_re_start     (zoom_cre_start),
                .c_im_start     (zoom_cim_start),
                .c_re_step      (zoom_step),
                .c_im_step      (zoom_cim_step),
                .max_iter       (max_iter_w),
                .neuron_valid   (neuron_valid_w),
                .neuron_ready   (neuron_ready_w),
                .neuron_c_re    (neuron_c_re_w),
                .neuron_c_im    (neuron_c_im_w),
                .neuron_pixel_id(neuron_pixel_id_w),
                .result_valid   (result_valid_w),
                .result_pixel_id(result_pixel_id_w),
                .result_iter    (result_iter_w),
                .fb_wr_en       (fb_iter_wr_en),
                .fb_wr_addr     (fb_iter_wr_addr),
                .fb_wr_data     (fb_iter_wr_data)
            );
        end else begin : gen_test
            // --- Test mode: trivial pixel writer, no neurons/scheduler ---
            // Writes one pixel per clock after frame_start.
            // TEST_MODE=1: solid iter=42, TEST_MODE=2: gradient iter=pixel_id[15:8]
            reg        test_busy;
            reg        test_done;
            reg [15:0] test_addr;
            reg        test_wr_en;
            reg [ITER_W-1:0] test_data;

            assign frame_busy_w     = test_busy;
            assign frame_done_w     = test_done;
            assign fb_iter_wr_en    = test_wr_en;
            assign fb_iter_wr_addr  = test_addr;
            assign fb_iter_wr_data  = test_data;

            // Tie off unused neuron wires
            assign neuron_valid_w     = {N_NEURONS{1'b0}};
            assign neuron_ready_w     = {N_NEURONS{1'b1}};
            assign neuron_c_re_w      = {WIDTH{1'b0}};
            assign neuron_c_im_w      = {WIDTH{1'b0}};
            assign neuron_pixel_id_w  = 16'd0;
            assign result_valid_w     = {N_NEURONS{1'b0}};
            assign result_pixel_id_w  = {(N_NEURONS*16){1'b0}};
            assign result_iter_w      = {(N_NEURONS*ITER_W){1'b0}};

            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    test_busy  <= 0;
                    test_done  <= 0;
                    test_addr  <= 0;
                    test_wr_en <= 0;
                    test_data  <= 0;
                end else begin
                    test_done  <= 0;
                    test_wr_en <= 0;

                    if (frame_start_r && !test_busy) begin
                        test_busy  <= 1;
                        test_addr  <= 0;
                        test_wr_en <= 1;
                        test_data  <= (TEST_MODE == 1) ? 42 : 0;
                    end else if (test_busy) begin
                        if (test_addr == PIX_COUNT[15:0] - 1) begin
                            test_busy <= 0;
                            test_done <= 1;
                        end else begin
                            test_addr  <= test_addr + 1;
                            test_wr_en <= 1;
                            // MODE 1: constant iter=42, MODE 2: low byte of pixel address
                            test_data  <= (TEST_MODE == 1) ? 42
                                                           : {8'd0, test_addr[7:0] + 8'd1};
                        end
                    end
                end
            end
        end
    endgenerate

    // =========================================================
    // Framebuffer pipeline
    // =========================================================
    // COMPUTE_MODE == 0 (Mandelbrot):
    //   iter_fb (double-buffered) → colormap → disp_fb → SPI
    // COMPUTE_MODE == 1 (MLP):
    //   scheduler result (RGB565) → disp_fb directly → SPI
    //   (iter_fb and colormap bypassed)
    // =========================================================
    localparam FB_DEPTH = 65536;
    reg iter_buf_sel;  // 0: compute writes buf 0, display reads buf 1
                       // 1: compute writes buf 1, display reads buf 0

    // --- Iteration framebuffer (Mandelbrot mode only) ---
    generate if (COMPUTE_MODE == 0) begin : gen_iter_fb
        // Buffer 0
        (* ram_style = "block" *) reg [ITER_W-1:0] iter_fb_0 [0:FB_DEPTH-1];

        always @(posedge clk) begin
            if (fb_iter_wr_en && !iter_buf_sel)
                iter_fb_0[fb_iter_wr_addr] <= fb_iter_wr_data;
        end

        reg [ITER_W-1:0] iter_fb_0_rd;
        always @(posedge clk) begin
            iter_fb_0_rd <= iter_fb_0[iter_fb_rd_addr];
        end

        // Buffer 1
        (* ram_style = "block" *) reg [ITER_W-1:0] iter_fb_1 [0:FB_DEPTH-1];

        always @(posedge clk) begin
            if (fb_iter_wr_en && iter_buf_sel)
                iter_fb_1[fb_iter_wr_addr] <= fb_iter_wr_data;
        end

        reg [ITER_W-1:0] iter_fb_1_rd;
        always @(posedge clk) begin
            iter_fb_1_rd <= iter_fb_1[iter_fb_rd_addr];
        end

        // Front-buffer read mux
        reg [ITER_W-1:0] iter_fb_rd;
        reg [15:0]       iter_fb_rd_addr;

        always @(posedge clk) begin
            iter_fb_rd_addr <= fb_disp_addr;
        end

        always @(*) begin
            iter_fb_rd = iter_buf_sel ? iter_fb_0_rd : iter_fb_1_rd;
        end
    end endgenerate

    // --- Colormap (Mandelbrot mode only) ---
    wire [15:0] color_rgb565;
    wire        color_valid;

    generate if (COMPUTE_MODE == 0) begin : gen_colormap
        colormap_lut #(.ITER_W(ITER_W)) u_colormap (
            .clk        (clk),
            .iter_count (gen_iter_fb.iter_fb_rd),
            .max_iter   (max_iter_w),
            .valid_in   (1'b1),
            .rgb565     (color_rgb565),
            .valid_out  (color_valid)
        );
    end else begin : gen_no_colormap
        assign color_rgb565 = 16'h0000;
        assign color_valid  = 1'b0;
    end endgenerate

    // =========================================================
    // Display framebuffer
    // =========================================================
    reg [15:0] disp_fb_rd;

    generate if (COMPUTE_MODE == 0) begin : gen_disp_fb
        // Mandelbrot: single disp_fb (iter_fb is already double-buffered)
        (* ram_style = "block" *) reg [15:0] disp_fb [0:FB_DEPTH-1];

        // Colormap write pipeline: delay address to match iter_fb read + colormap latency
        reg [15:0] color_wr_addr;
        reg [15:0] color_wr_addr_d;

        always @(posedge clk) begin
            color_wr_addr   <= gen_iter_fb.iter_fb_rd_addr;
            color_wr_addr_d <= color_wr_addr;
        end

        // Port A: Write from colormap
        always @(posedge clk) begin
            if (color_valid)
                disp_fb[color_wr_addr_d] <= color_rgb565;
        end

        // Port B: Read for SPI driver
        always @(posedge clk) begin
            disp_fb_rd <= disp_fb[fb_disp_addr];
        end

    end else begin : gen_disp_fb
        // MLP: double-buffered disp_fb (no iter_fb exists)
        // iter_buf_sel: 0 = compute writes buf 0, SPI reads buf 1
        //               1 = compute writes buf 1, SPI reads buf 0
        (* ram_style = "block" *) reg [15:0] disp_fb_0 [0:FB_DEPTH-1];
        (* ram_style = "block" *) reg [15:0] disp_fb_1 [0:FB_DEPTH-1];

        // Write: scheduler result → back buffer
        always @(posedge clk) begin
            if (fb_iter_wr_en && !iter_buf_sel)
                disp_fb_0[fb_iter_wr_addr] <= fb_iter_wr_data;
        end
        always @(posedge clk) begin
            if (fb_iter_wr_en && iter_buf_sel)
                disp_fb_1[fb_iter_wr_addr] <= fb_iter_wr_data;
        end

        // Read: SPI reads front buffer
        reg [15:0] disp_fb_0_rd, disp_fb_1_rd;
        always @(posedge clk) begin
            disp_fb_0_rd <= disp_fb_0[fb_disp_addr];
            disp_fb_1_rd <= disp_fb_1[fb_disp_addr];
        end

        always @(*) begin
            disp_fb_rd = iter_buf_sel ? disp_fb_0_rd : disp_fb_1_rd;
        end
    end endgenerate

    // =========================================================
    // SP2 SPI display driver
    // =========================================================
    sp2_spi_driver #(
        .H_RES(H_RES), .V_RES(V_RES),
        .SCK_DIV(0)     // 25 MHz SCK (max speed)
    ) u_display (
        .clk        (clk),
        .rst_n      (rst_n),
        .spi_cs_n   (spi_cs_n),
        .spi_sck    (spi_sck),
        .spi_mosi   (spi_mosi),
        .spi_dc     (spi_dc),
        .lcd_rst_n  (lcd_rst_n),
        .lcd_blk    (lcd_blk_int),
        .fb_addr    (fb_disp_addr),
        .fb_data    (disp_fb_rd),
        .frame_done (disp_frame_done)
    );

    // =========================================================
    // Frame controller
    // =========================================================
    // COMPUTE_MODE==0: Auto-zoom into seahorse valley (Mandelbrot)
    // COMPUTE_MODE==1: Fixed viewport, incrementing time (MLP/SIREN)

    generate if (COMPUTE_MODE == 0) begin : gen_zoom_ctrl
        // --- Mandelbrot zoom controller ---
        localparam signed [WIDTH-1:0] TARGET_RE = 32'shF414_7AE1;  // -0.745
        localparam signed [WIDTH-1:0] TARGET_IM = 32'sh01CF_DF3B;  // +0.113

        wire signed [WIDTH-1:0] next_step = zoom_step - (zoom_step >>> 6);
        wire signed [WIDTH-1:0] half_h = (next_step <<< 7) + (next_step <<< 5);
        wire signed [WIDTH-1:0] half_v = (next_step <<< 6) + (next_step <<< 4)
                                       + (next_step <<< 2) + (next_step <<< 1);

        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                frame_start_r  <= 0;
                startup        <= 1;
                zoom_step      <= DEFAULT_CRE_STEP;
                zoom_cim_step  <= DEFAULT_CIM_STEP;
                zoom_cre_start <= DEFAULT_CRE_START;
                zoom_cim_start <= DEFAULT_CIM_START;
                max_iter_w     <= DEFAULT_MAX_ITER;
                iter_buf_sel   <= 0;
            end else begin
                frame_start_r <= 0;

                if (startup) begin
                    frame_start_r <= 1;
                    startup       <= 0;
                end

                if (frame_done_w && !frame_busy_w) begin
                    iter_buf_sel <= ~iter_buf_sel;
                    if (next_step == zoom_step) begin
                        zoom_step      <= DEFAULT_CRE_STEP;
                        zoom_cim_step  <= DEFAULT_CIM_STEP;
                        zoom_cre_start <= DEFAULT_CRE_START;
                        zoom_cim_start <= DEFAULT_CIM_START;
                        max_iter_w     <= DEFAULT_MAX_ITER;
                    end else begin
                        zoom_step      <= next_step;
                        zoom_cim_step  <= next_step;
                        zoom_cre_start <= TARGET_RE - half_h;
                        zoom_cim_start <= TARGET_IM - half_v;
                        if (max_iter_w < 1024)
                            max_iter_w <= max_iter_w + 1;
                    end
                    frame_start_r <= 1;
                end
            end
        end

    end else begin : gen_mlp_ctrl
        // --- MLP time-stepping controller ---
        // Fixed viewport [-1,+1], max_iter increments as time parameter
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                frame_start_r  <= 0;
                startup        <= 1;
                zoom_step      <= MLP_CRE_STEP;
                zoom_cim_step  <= MLP_CIM_STEP;
                zoom_cre_start <= MLP_CRE_START;
                zoom_cim_start <= MLP_CIM_START;
                max_iter_w     <= 0;
                iter_buf_sel   <= 0;
            end else begin
                frame_start_r <= 0;

                if (startup) begin
                    frame_start_r <= 1;
                    startup       <= 0;
                end

                if (frame_done_w && !frame_busy_w) begin
                    // Swap double-buffered disp_fb
                    iter_buf_sel <= ~iter_buf_sel;
                    // Increment time each frame (wraps at 65535)
                    max_iter_w <= max_iter_w + 1;
                    frame_start_r <= 1;
                end
            end
        end
    end endgenerate

    // =========================================================
    // LED1 — toggles each compute frame (shows zoom progress)
    // =========================================================
    reg led_frame_r;
    assign led_frame = led_frame_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            led_frame_r <= 0;
        else if (frame_done_w)
            led_frame_r <= ~led_frame_r;
    end

    // Backlight pass-through from SPI driver
    wire lcd_blk_int;
    assign lcd_blk_out = lcd_blk_int;

    // =========================================================
    // UART boot message — sends "MANDELBROT QSPI OK\r\n" once
    // =========================================================
    boot_msg u_boot_msg (
        .clk   (clk),
        .rst_n (rst_n),
        .tx    (uart_tx)
    );

    // =========================================================
    // Alive LED — 1 Hz blink (25M counter at 50 MHz)
    // =========================================================
    reg [24:0] alive_cnt;
    reg        alive_r;
    assign led_alive = alive_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            alive_cnt <= 0;
            alive_r   <= 0;
        end else begin
            if (alive_cnt == 25_000_000) begin
                alive_cnt <= 0;
                alive_r   <= ~alive_r;
            end else begin
                alive_cnt <= alive_cnt + 1;
            end
        end
    end

endmodule
