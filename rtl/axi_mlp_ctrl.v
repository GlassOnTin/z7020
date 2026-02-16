// axi_mlp_ctrl.v — AXI-Lite slave for PS control of MLP inference engine
//
// Register map (64KB window at base 0x43C0_0000):
//
//   0x0000–0x140B  WEIGHT_A  (W)   BRAM A: word[i] at offset i*4
//   0x2000–0x340B  WEIGHT_B  (W)   BRAM B: word[i] at offset i*4
//   0x4000         CTRL      (R/W) [0] ps_override [1] threshold_en [2] frame_auto
//   0x4004         MORPH_ALPHA (R/W) Blend factor [7:0]
//   0x4008         TIME_VAL  (R/W) Frame time [15:0]
//   0x400C         STATUS    (R)   [0] frame_busy [1] frame_done
//   0x4010         FRAME_START (W) Write-any-value trigger
//
// Address decode: awaddr[14:13] selects region:
//   2'b00 = weight A, 2'b01 = weight B, 2'b10 = control regs
// Weight index from awaddr[12:2] (11 bits = 2048 entries max).

`timescale 1ns / 1ps

module axi_mlp_ctrl #(
    parameter C_S_AXI_DATA_WIDTH = 32,
    parameter C_S_AXI_ADDR_WIDTH = 16
)(
    // AXI-Lite slave interface
    input  wire                                S_AXI_ACLK,
    input  wire                                S_AXI_ARESETN,

    input  wire [C_S_AXI_ADDR_WIDTH-1:0]       S_AXI_AWADDR,
    input  wire [2:0]                          S_AXI_AWPROT,
    input  wire                                S_AXI_AWVALID,
    output reg                                 S_AXI_AWREADY,

    input  wire [C_S_AXI_DATA_WIDTH-1:0]       S_AXI_WDATA,
    input  wire [C_S_AXI_DATA_WIDTH/8-1:0]     S_AXI_WSTRB,
    input  wire                                S_AXI_WVALID,
    output reg                                 S_AXI_WREADY,

    output reg  [1:0]                          S_AXI_BRESP,
    output reg                                 S_AXI_BVALID,
    input  wire                                S_AXI_BREADY,

    input  wire [C_S_AXI_ADDR_WIDTH-1:0]       S_AXI_ARADDR,
    input  wire [2:0]                          S_AXI_ARPROT,
    input  wire                                S_AXI_ARVALID,
    output reg                                 S_AXI_ARREADY,

    output reg  [C_S_AXI_DATA_WIDTH-1:0]       S_AXI_RDATA,
    output reg  [1:0]                          S_AXI_RRESP,
    output reg                                 S_AXI_RVALID,
    input  wire                                S_AXI_RREADY,

    // Control outputs to mandelbrot_top
    output reg         ps_override,
    output reg         threshold_enable,
    output reg         frame_auto,
    output reg  [7:0]  morph_alpha,
    output reg  [15:0] time_val,
    output reg         frame_start_pulse,

    // Weight write outputs (broadcast to all mlp_core instances)
    output reg  [10:0] weight_wr_addr,
    output reg  [31:0] weight_wr_data,
    output reg         weight_wr_en,
    output reg         weight_wr_bank,  // 0=A, 1=B

    // Status inputs from mandelbrot_top
    input  wire        frame_busy,
    input  wire        frame_done
);

    // =========================================================
    // Sticky frame_done latch (set by 1-cycle pulse, clear on STATUS read)
    // =========================================================
    reg frame_done_sticky;

    // =========================================================
    // Write channel FSM
    // =========================================================
    // Simple 2-phase: accept AW+W together, then assert B.
    reg [C_S_AXI_ADDR_WIDTH-1:0] aw_addr_r;
    reg [C_S_AXI_DATA_WIDTH-1:0] aw_data_r;
    reg aw_en;

    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            S_AXI_AWREADY <= 1'b0;
            S_AXI_WREADY  <= 1'b0;
            S_AXI_BVALID  <= 1'b0;
            S_AXI_BRESP   <= 2'b00;
            aw_addr_r      <= 0;
            aw_data_r      <= 0;
            aw_en          <= 1'b0;

            ps_override       <= 1'b0;
            threshold_enable  <= 1'b0;
            frame_auto        <= 1'b0;
            morph_alpha       <= 8'd0;
            time_val          <= 16'd0;
            frame_start_pulse <= 1'b0;
            weight_wr_en      <= 1'b0;
            weight_wr_addr    <= 11'd0;
            weight_wr_data    <= 32'd0;
            weight_wr_bank    <= 1'b0;
            frame_done_sticky <= 1'b0;
        end else begin
            // Latch frame_done pulse (sticky until cleared by STATUS read)
            // Clear when read channel accepts a STATUS read (ARADDR[14:13]==2'b10, [4:2]==3'd3)
            if (S_AXI_ARVALID && !S_AXI_RVALID &&
                S_AXI_ARADDR[14:13] == 2'b10 && S_AXI_ARADDR[4:2] == 3'd3)
                frame_done_sticky <= 1'b0;
            if (frame_done)
                frame_done_sticky <= 1'b1;  // Set wins over clear (last assignment)
            // Default: clear single-cycle pulses
            frame_start_pulse <= 1'b0;
            weight_wr_en      <= 1'b0;

            // Accept AW+W simultaneously, latch both address and data
            if (S_AXI_AWVALID && S_AXI_WVALID && !aw_en) begin
                S_AXI_AWREADY <= 1'b1;
                S_AXI_WREADY  <= 1'b1;
                aw_addr_r      <= S_AXI_AWADDR;
                aw_data_r      <= S_AXI_WDATA;
                aw_en          <= 1'b1;
            end else begin
                S_AXI_AWREADY <= 1'b0;
                S_AXI_WREADY  <= 1'b0;
            end

            // Process write and assert BVALID (uses latched data)
            if (aw_en) begin
                aw_en         <= 1'b0;
                S_AXI_BVALID  <= 1'b1;
                S_AXI_BRESP   <= 2'b00;  // OKAY

                case (aw_addr_r[14:13])
                    2'b00: begin
                        // Weight A write
                        weight_wr_addr <= aw_addr_r[12:2];
                        weight_wr_data <= aw_data_r;
                        weight_wr_en   <= 1'b1;
                        weight_wr_bank <= 1'b0;
                    end
                    2'b01: begin
                        // Weight B write
                        weight_wr_addr <= aw_addr_r[12:2];
                        weight_wr_data <= aw_data_r;
                        weight_wr_en   <= 1'b1;
                        weight_wr_bank <= 1'b1;
                    end
                    2'b10: begin
                        // Control register writes
                        case (aw_addr_r[4:2])
                            3'd0: begin  // 0x4000 CTRL
                                ps_override      <= aw_data_r[0];
                                threshold_enable <= aw_data_r[1];
                                frame_auto       <= aw_data_r[2];
                            end
                            3'd1: begin  // 0x4004 MORPH_ALPHA
                                morph_alpha <= aw_data_r[7:0];
                            end
                            3'd2: begin  // 0x4008 TIME_VAL
                                time_val <= aw_data_r[15:0];
                            end
                            // 0x400C STATUS is read-only
                            3'd4: begin  // 0x4010 FRAME_START
                                frame_start_pulse <= 1'b1;
                            end
                            default: ;
                        endcase
                    end
                    default: ;
                endcase
            end

            // Clear BVALID when master accepts
            if (S_AXI_BVALID && S_AXI_BREADY) begin
                S_AXI_BVALID <= 1'b0;
            end
        end
    end

    // =========================================================
    // Read channel
    // =========================================================
    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            S_AXI_ARREADY <= 1'b0;
            S_AXI_RVALID  <= 1'b0;
            S_AXI_RRESP   <= 2'b00;
            S_AXI_RDATA   <= 32'd0;
        end else begin
            // Accept read address
            if (S_AXI_ARVALID && !S_AXI_RVALID) begin
                S_AXI_ARREADY <= 1'b1;

                // Decode read address
                if (S_AXI_ARADDR[14:13] == 2'b10) begin
                    case (S_AXI_ARADDR[4:2])
                        3'd0: S_AXI_RDATA <= {29'd0, frame_auto, threshold_enable, ps_override};
                        3'd1: S_AXI_RDATA <= {24'd0, morph_alpha};
                        3'd2: S_AXI_RDATA <= {16'd0, time_val};
                        3'd3: S_AXI_RDATA <= {30'd0, frame_done_sticky, frame_busy};
                        default: S_AXI_RDATA <= 32'd0;
                    endcase
                end else begin
                    S_AXI_RDATA <= 32'd0;
                end

                S_AXI_RVALID <= 1'b1;
                S_AXI_RRESP  <= 2'b00;
            end else begin
                S_AXI_ARREADY <= 1'b0;
            end

            // Clear RVALID when master accepts
            if (S_AXI_RVALID && S_AXI_RREADY) begin
                S_AXI_RVALID <= 1'b0;
            end
        end
    end

endmodule
