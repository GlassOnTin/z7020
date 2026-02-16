// mlp_pl_wrapper.v — Top-level PL wrapper for PS-integrated MLP build
//
// Instantiates:
//   - axi_mlp_ctrl: AXI-Lite slave for PS control
//   - mandelbrot_top: MLP inference engine (COMPUTE_MODE=1, PS_ENABLE)
//
// Used by the Vivado block design (run_ps_build.tcl) as an RTL module.
// Clock comes from PS7 FCLK_CLK0 (50 MHz), no PL crystal needed.

`timescale 1ns / 1ps

module mlp_pl_wrapper (
    // Clock and reset from PS7
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 FCLK_CLK0 CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF S_AXI, ASSOCIATED_RESET FCLK_RESET0_N, FREQ_HZ 50000000" *)
    input  wire        FCLK_CLK0,

    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 FCLK_RESET0_N RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire        FCLK_RESET0_N,

    // AXI-Lite slave (from PS7 M_AXI_GP0 via interconnect)
    // 32-bit addresses from AXI interconnect; lower 16 bits used internally
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWADDR" *)
    (* X_INTERFACE_PARAMETER = "PROTOCOL AXI4LITE, DATA_WIDTH 32, ADDR_WIDTH 32, FREQ_HZ 50000000" *)
    input  wire [31:0] S_AXI_AWADDR,
    input  wire [2:0]  S_AXI_AWPROT,
    input  wire        S_AXI_AWVALID,
    output wire        S_AXI_AWREADY,
    input  wire [31:0] S_AXI_WDATA,
    input  wire [3:0]  S_AXI_WSTRB,
    input  wire        S_AXI_WVALID,
    output wire        S_AXI_WREADY,
    output wire [1:0]  S_AXI_BRESP,
    output wire        S_AXI_BVALID,
    input  wire        S_AXI_BREADY,
    input  wire [31:0] S_AXI_ARADDR,
    input  wire [2:0]  S_AXI_ARPROT,
    input  wire        S_AXI_ARVALID,
    output wire        S_AXI_ARREADY,
    output wire [31:0] S_AXI_RDATA,
    output wire [1:0]  S_AXI_RRESP,
    output wire        S_AXI_RVALID,
    input  wire        S_AXI_RREADY,

    // SP2 display SPI pins
    output wire        spi_cs_n,
    output wire        spi_sck,
    output wire        spi_mosi,
    output wire        spi_dc,
    output wire        lcd_rst_n,
    output wire        lcd_blk_out,

    // Debug LEDs
    output wire        led_frame,
    output wire        led_alive,

    // UART
    output wire        uart_tx
);

    // Internal wires: AXI ctrl → mandelbrot_top
    wire        ps_override;
    wire        threshold_enable;
    wire        frame_auto;
    wire [7:0]  morph_alpha;
    wire [15:0] time_val;
    wire        frame_start_pulse;

    wire [10:0] weight_wr_addr;
    wire [31:0] weight_wr_data;
    wire        weight_wr_en;
    wire        weight_wr_bank;

    wire        frame_busy;
    wire        frame_done;

    // =========================================================
    // AXI-Lite slave
    // =========================================================
    axi_mlp_ctrl #(
        .C_S_AXI_DATA_WIDTH(32),
        .C_S_AXI_ADDR_WIDTH(16)
    ) u_axi_ctrl (
        .S_AXI_ACLK     (FCLK_CLK0),
        .S_AXI_ARESETN   (FCLK_RESET0_N),

        .S_AXI_AWADDR   (S_AXI_AWADDR[15:0]),
        .S_AXI_AWPROT   (S_AXI_AWPROT),
        .S_AXI_AWVALID  (S_AXI_AWVALID),
        .S_AXI_AWREADY  (S_AXI_AWREADY),
        .S_AXI_WDATA    (S_AXI_WDATA),
        .S_AXI_WSTRB    (S_AXI_WSTRB),
        .S_AXI_WVALID   (S_AXI_WVALID),
        .S_AXI_WREADY   (S_AXI_WREADY),
        .S_AXI_BRESP    (S_AXI_BRESP),
        .S_AXI_BVALID   (S_AXI_BVALID),
        .S_AXI_BREADY   (S_AXI_BREADY),

        .S_AXI_ARADDR   (S_AXI_ARADDR[15:0]),
        .S_AXI_ARPROT   (S_AXI_ARPROT),
        .S_AXI_ARVALID  (S_AXI_ARVALID),
        .S_AXI_ARREADY  (S_AXI_ARREADY),
        .S_AXI_RDATA    (S_AXI_RDATA),
        .S_AXI_RRESP    (S_AXI_RRESP),
        .S_AXI_RVALID   (S_AXI_RVALID),
        .S_AXI_RREADY   (S_AXI_RREADY),

        .ps_override      (ps_override),
        .threshold_enable  (threshold_enable),
        .frame_auto        (frame_auto),
        .morph_alpha       (morph_alpha),
        .time_val          (time_val),
        .frame_start_pulse (frame_start_pulse),

        .weight_wr_addr   (weight_wr_addr),
        .weight_wr_data   (weight_wr_data),
        .weight_wr_en     (weight_wr_en),
        .weight_wr_bank   (weight_wr_bank),

        .frame_busy       (frame_busy),
        .frame_done        (frame_done)
    );

    // =========================================================
    // MLP inference engine
    // =========================================================
    mandelbrot_top #(
        .N_NEURONS   (18),
        .WIDTH       (32),
        .FRAC        (28),
        .ITER_W      (16),
        .H_RES       (320),
        .V_RES       (172),
        .TEST_MODE   (0),
        .COMPUTE_MODE(1)
    ) u_mlp_top (
        .clk_50m     (FCLK_CLK0),
        .rst_n_in    (FCLK_RESET0_N),

        .spi_cs_n    (spi_cs_n),
        .spi_sck     (spi_sck),
        .spi_mosi    (spi_mosi),
        .spi_dc      (spi_dc),
        .lcd_rst_n   (lcd_rst_n),
        .lcd_blk_out (lcd_blk_out),

        .led_frame   (led_frame),
        .led_alive   (led_alive),
        .uart_tx     (uart_tx),

        .ps_override       (ps_override),
        .ext_weight_wr_addr(weight_wr_addr),
        .ext_weight_wr_data(weight_wr_data),
        .ext_weight_wr_en  (weight_wr_en),
        .ext_weight_wr_bank(weight_wr_bank),
        .ext_morph_alpha   (morph_alpha),
        .ext_time_val      (time_val),
        .ext_frame_start   (frame_start_pulse),
        .ext_threshold_en  (threshold_enable),
        .frame_busy_out    (frame_busy),
        .frame_done_out    (frame_done)
    );

endmodule
