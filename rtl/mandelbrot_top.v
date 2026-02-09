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
    parameter PIX_COUNT = H_RES * V_RES
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
    wire [ITER_W-1:0]        max_iter_w = DEFAULT_MAX_ITER;

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

    // =========================================================
    // Neuron pool
    // =========================================================
    genvar i;
    generate
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
    endgenerate

    // =========================================================
    // Pixel scheduler
    // =========================================================
    pixel_scheduler #(
        .N_NEURONS(N_NEURONS), .WIDTH(WIDTH), .FRAC(FRAC),
        .ITER_W(ITER_W), .H_RES(H_RES), .V_RES(V_RES)
    ) u_scheduler (
        .clk            (clk),
        .rst_n          (rst_n),
        .frame_start    (frame_start_r),
        .frame_busy     (frame_busy_w),
        .frame_done     (frame_done_w),
        .c_re_start     (DEFAULT_CRE_START),
        .c_im_start     (DEFAULT_CIM_START),
        .c_re_step      (DEFAULT_CRE_STEP),
        .c_im_step      (DEFAULT_CIM_STEP),
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

    // Port B: read (addressed by display driver, through colormap)
    reg [ITER_W-1:0] iter_fb_rd;
    reg [15:0]       iter_fb_rd_addr;

    always @(posedge clk) begin
        iter_fb_rd <= iter_fb[iter_fb_rd_addr];
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
    // Display framebuffer (BRAM, dual-port)
    // Port A: Write from colormap
    // Port B: Read from SPI driver
    // =========================================================
    (* ram_style = "block" *) reg [15:0] disp_fb [0:FB_DEPTH-1];

    // Colormap write pipeline: delay address by colormap latency (1 cycle)
    reg [15:0] color_wr_addr;
    reg [15:0] color_wr_addr_d;

    always @(posedge clk) begin
        color_wr_addr   <= iter_fb_rd_addr;
        color_wr_addr_d <= color_wr_addr;
        if (color_valid)
            disp_fb[color_wr_addr_d] <= color_rgb565;
    end

    // Port B: read from SPI driver
    reg [15:0] disp_fb_rd;
    always @(posedge clk) begin
        disp_fb_rd <= disp_fb[fb_disp_addr];
    end

    // =========================================================
    // SP2 SPI display driver
    // =========================================================
    sp2_spi_driver #(
        .H_RES(H_RES), .V_RES(V_RES),
        .SCK_DIV(24)    // ~1 MHz SCK for testing (change to 0 for 25 MHz)
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

    // Connect display read address to iteration FB read address
    // (colormap sits between iteration FB and display FB)
    always @(posedge clk) begin
        iter_fb_rd_addr <= fb_disp_addr;
    end

    // =========================================================
    // Frame control — auto-start
    // =========================================================
    // For Phase 1-2: automatically start a new frame after display completes
    reg frame_done_seen;
    reg startup;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            frame_start_r  <= 0;
            frame_done_seen <= 0;
            startup        <= 1;
        end else begin
            frame_start_r <= 0;

            // Start first frame after reset
            if (startup) begin
                frame_start_r <= 1;
                startup       <= 0;
            end

            // Auto-restart on frame completion
            if (frame_done_w && !frame_busy_w) begin
                frame_start_r <= 1;
            end
        end
    end

    // =========================================================
    // LED heartbeat — toggle on each frame
    // =========================================================
    reg led_r;
    assign led_frame = led_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            led_r <= 0;
        else if (disp_frame_done)  // Toggle on DISPLAY frame done (not compute)
            led_r <= ~led_r;
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
