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
    parameter TEST_MODE = 0       // 0=normal, 1=solid color (iter=42), 2=gradient (iter=pixel_id[15:8])
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
    output wire        led_alive      // 1Hz blink — proves FPGA is running
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

    // Display frame done
    wire                     disp_frame_done;

    // Double-buffer control
    reg                      disp_buf_sel;    // 0 = SPI reads A, 1 = SPI reads B
    reg                      swap_pending;    // Set when sweep done, cleared on swap

    // Color sweep state
    reg                      sweeping;
    reg [15:0]               sweep_addr;
    reg                      sweep_done;

    // Auto-zoom viewport registers (declared here, driven by zoom controller below)
    reg signed [WIDTH-1:0] zoom_step;
    reg signed [WIDTH-1:0] zoom_cre_start;
    reg signed [WIDTH-1:0] zoom_cim_start;
    reg startup;

    // =========================================================
    // Neuron pool + Pixel scheduler (or test-mode substitute)
    // =========================================================
    generate
        if (TEST_MODE == 0) begin : gen_normal
            // --- Production: neuron pool + scheduler ---
            genvar i;
            for (i = 0; i < N_NEURONS; i = i + 1) begin : neurons
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
                .c_im_step      (zoom_step),
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
    // Iteration count framebuffer (BRAM, dual-port)
    // Port A: Write from scheduler (iteration counts)
    // Port B: Read for colormap → display
    // =========================================================
    // Framebuffers use 2^16 depth for clean BRAM inference (only 55040 used)
    localparam FB_DEPTH = 65536;
    (* ram_style = "block" *) reg [ITER_W-1:0] iter_fb [0:FB_DEPTH-1];

    // Port A: write
    always @(posedge clk) begin
        if (fb_iter_wr_en)
            iter_fb[fb_iter_wr_addr] <= fb_iter_wr_data;
    end

    // Port B: read (addressed by display driver or sweep)
    reg [ITER_W-1:0] iter_fb_rd;
    reg [15:0]       iter_fb_rd_addr;

    always @(posedge clk) begin
        iter_fb_rd_addr <= sweeping ? sweep_addr : fb_disp_addr;
        iter_fb_rd      <= iter_fb[iter_fb_rd_addr];
    end

    // =========================================================
    // Colormap: iteration count → RGB565
    // =========================================================
    wire [15:0] color_rgb565;
    wire        color_valid;

    colormap_lut #(.ITER_W(ITER_W)) u_colormap (
        .clk        (clk),
        .iter_count (iter_fb_rd),
        .max_iter   (max_iter_w),
        .valid_in   (1'b1),         // Always valid (continuous read)
        .rgb565     (color_rgb565),
        .valid_out  (color_valid)
    );

    // =========================================================
    // Display framebuffer — double-buffered (BRAM, dual-port)
    // Each buffer has its own always block for clean BRAM inference.
    // Port A: Write from colormap sweep (to back buffer)
    // Port B: Read from SPI driver (from front buffer)
    // =========================================================

    // Colormap write pipeline: delay address to match iter_fb read + colormap latency (2 cycles)
    reg [15:0] color_wr_addr;
    reg [15:0] color_wr_addr_d;

    // Delay sweeping by 2 cycles to match the address pipeline latency.
    // Without this, the first 2 writes after sweep start go to stale
    // display-path addresses instead of sweep addresses.
    reg        sweep_wr_v0, sweep_wr_v1;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sweep_wr_v0 <= 0;
            sweep_wr_v1 <= 0;
        end else begin
            sweep_wr_v0 <= sweeping;
            sweep_wr_v1 <= sweep_wr_v0;
        end
    end
    wire       color_wr_en = color_valid & sweep_wr_v1;

    always @(posedge clk) begin
        color_wr_addr   <= iter_fb_rd_addr;
        color_wr_addr_d <= color_wr_addr;
    end

    // Back-buffer write enables (one-hot, never both)
    wire wr_en_a = color_wr_en &  disp_buf_sel;  // buf_sel=1 → SPI reads B, write to A
    wire wr_en_b = color_wr_en & ~disp_buf_sel;  // buf_sel=0 → SPI reads A, write to B

    // Buffer A — dedicated write and read ports
    (* ram_style = "block" *) reg [15:0] disp_fb_a [0:FB_DEPTH-1];
    reg [15:0] disp_fb_a_rd;

    always @(posedge clk) begin
        if (wr_en_a)
            disp_fb_a[color_wr_addr_d] <= color_rgb565;
    end

    always @(posedge clk) begin
        disp_fb_a_rd <= disp_fb_a[fb_disp_addr];
    end

    // Buffer B — dedicated write and read ports
    (* ram_style = "block" *) reg [15:0] disp_fb_b [0:FB_DEPTH-1];
    reg [15:0] disp_fb_b_rd;

    always @(posedge clk) begin
        if (wr_en_b)
            disp_fb_b[color_wr_addr_d] <= color_rgb565;
    end

    always @(posedge clk) begin
        disp_fb_b_rd <= disp_fb_b[fb_disp_addr];
    end

    // Front-buffer read mux (after BRAM output registers)
    reg [15:0] disp_fb_rd;
    always @(*) begin
        disp_fb_rd = disp_buf_sel ? disp_fb_b_rd : disp_fb_a_rd;
    end

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

    // iter_fb_rd_addr is now driven by the sweep/display mux in the iter_fb read block above

    // =========================================================
    // Color sweep state machine
    // =========================================================
    // After compute completes, sweep reads iter_fb[0..PIX_COUNT-1]
    // through the colormap and writes results into the back disp_fb.
    // Pipeline: cycle N: set sweep_addr → N+1: iter_fb_rd valid →
    //           N+2: color_rgb565 valid → write to back buffer.
    // sweep_addr counts 0 to PIX_COUNT-1, then 2 extra flush cycles.

    localparam [15:0] SWEEP_END = PIX_COUNT[15:0] + 16'd1;  // +1 flush (sweep_wr_v1 delay provides 2 extra write cycles)

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sweeping   <= 0;
            sweep_addr <= 0;
            sweep_done <= 0;
        end else begin
            sweep_done <= 0;

            if (frame_done_w && !frame_busy_w && !sweeping) begin
                // Compute done — start sweep
                sweeping   <= 1;
                sweep_addr <= 0;
            end else if (sweeping) begin
                if (sweep_addr == SWEEP_END - 1) begin
                    sweeping   <= 0;
                    sweep_done <= 1;
                end else begin
                    sweep_addr <= sweep_addr + 1;
                end
            end
        end
    end

    // =========================================================
    // Auto-zoom controller with double-buffer swap
    // =========================================================
    // Zoom target: seahorse valley (-0.745, +0.113)
    localparam signed [WIDTH-1:0] TARGET_RE = 32'shF414_7AE1;  // -0.745 in Q4.28
    localparam signed [WIDTH-1:0] TARGET_IM = 32'sh01CF_DF3B;  // +0.113 in Q4.28

    // Next step value after zoom
    wire signed [WIDTH-1:0] next_step = zoom_step - (zoom_step >>> 6);  // × 63/64 ≈ 0.984

    // Compute viewport start from next_step:
    //   start_re = TARGET_RE - 160 * next_step   (160 = 128 + 32)
    //   start_im = TARGET_IM -  86 * next_step   ( 86 =  64 + 16 + 4 + 2)
    wire signed [WIDTH-1:0] half_h = (next_step <<< 7) + (next_step <<< 5);          // 160 * step
    wire signed [WIDTH-1:0] half_v = (next_step <<< 6) + (next_step <<< 4)
                                   + (next_step <<< 2) + (next_step <<< 1);          //  86 * step

    // Frame sequencing:
    //   frame_done_w → sweep iter_fb through colormap into back buffer (~1.1ms)
    //   sweep_done → set swap_pending
    //   disp_frame_done && swap_pending → swap buffers + zoom + frame_start

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            frame_start_r  <= 0;
            startup        <= 1;
            zoom_step      <= DEFAULT_CRE_STEP;
            zoom_cre_start <= DEFAULT_CRE_START;
            zoom_cim_start <= DEFAULT_CIM_START;
            max_iter_w     <= DEFAULT_MAX_ITER;
            disp_buf_sel   <= 0;
            swap_pending   <= 0;
        end else begin
            frame_start_r <= 0;

            // Start first frame after reset
            if (startup) begin
                frame_start_r <= 1;
                startup       <= 0;
            end

            // Sweep done → arm buffer swap
            if (sweep_done)
                swap_pending <= 1;

            // SPI frame boundary + swap pending → swap buffers and start next compute
            if (disp_frame_done && swap_pending) begin
                disp_buf_sel <= ~disp_buf_sel;
                swap_pending <= 0;

                if (next_step == zoom_step) begin
                    // Precision exhausted (step >>> 6 rounds to 0) — loop back
                    zoom_step      <= DEFAULT_CRE_STEP;
                    zoom_cre_start <= DEFAULT_CRE_START;
                    zoom_cim_start <= DEFAULT_CIM_START;
                    max_iter_w     <= DEFAULT_MAX_ITER;
                end else begin
                    // Shrink step by factor 63/64 ≈ 0.984
                    zoom_step      <= next_step;
                    // Recompute viewport origin from new step
                    zoom_cre_start <= TARGET_RE - half_h;
                    zoom_cim_start <= TARGET_IM - half_v;
                    // Ramp up max_iter (more detail as we zoom in)
                    if (max_iter_w < 1024)
                        max_iter_w <= max_iter_w + 1;
                end
                frame_start_r <= 1;
            end
        end
    end

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
