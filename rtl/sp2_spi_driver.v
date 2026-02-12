// sp2_spi_driver.v — ST7789V3 SPI display driver for SP2 1.47" 320x172 LCD
//
// Drives the ST7789V3 LCD controller via 4-wire SPI:
//   - CS_N:  Chip select (active low)
//   - SCK:   SPI clock (divided from system clock)
//   - MOSI:  Serial data out
//   - DC:    Data/Command select (0=command, 1=data)
//   - RST_N: Hardware reset (active low)
//
// Timing: reset 120ms, wait 120ms, send SLPOUT, wait 120ms, init, display on.

`timescale 1ns / 1ps

module sp2_spi_driver #(
    parameter H_RES     = 320,
    parameter V_RES     = 172,
    parameter PIX_COUNT = H_RES * V_RES,  // 55040
    parameter RST_WAIT    = 6_000_000,     // 120ms at 50 MHz (reset + post-reset)
    parameter SLPOUT_WAIT = 6_000_000,    // 120ms at 50 MHz
    parameter SCK_DIV     = 0             // Prescaler: 0=25MHz, 24=1MHz, 249=100KHz
)(
    input  wire        clk,          // System clock (50 MHz)
    input  wire        rst_n,

    // SPI outputs
    output reg         spi_cs_n,
    output wire        spi_sck,
    output reg         spi_mosi,
    output reg         spi_dc,
    output reg         lcd_rst_n,
    output reg         lcd_blk,       // Backlight enable

    // Framebuffer read port
    output reg  [15:0] fb_addr,      // Pixel address (0 to PIX_COUNT-1)
    input  wire [15:0] fb_data,      // RGB565 pixel data

    // Status
    output reg         frame_done    // Pulses high for 1 cycle at end of each frame
);

    // =========================================================
    // State machine
    // =========================================================
    localparam [3:0] ST_RESET       = 4'd0,
                     ST_RESET_REL   = 4'd1,
                     ST_SLPOUT      = 4'd2,   // Send SLPOUT command
                     ST_SLPOUT_WAIT = 4'd3,   // Wait 120ms after SLPOUT
                     ST_INIT        = 4'd4,   // Send remaining init commands
                     ST_SET_WIN     = 4'd5,   // Set column/row window + RAMWR
                     ST_PIXEL       = 4'd6,   // Stream pixel data
                     ST_FRAME_END   = 4'd7;

    reg [3:0]  state;
    reg [22:0] wait_cnt;          // Up to 8M counts (~160ms at 50 MHz)

    // =========================================================
    // SPI byte shifter — phase-counted, 25 MHz SCK (SPI Mode 0)
    // =========================================================
    // bit_phase counts 0-15: 8 bits × 2 half-cycles each.
    //   Even phases (0,2,...,14): MOSI driven with current bit, SCK low
    //   Odd  phases (1,3,...,15): SCK high, ST7789 samples MOSI
    //
    // MOSI is set up a full clock cycle (20 ns) before each SCK rise.
    // SCK only toggles while actively shifting — no extra edges between bytes.
    reg [3:0]  bit_phase;
    reg [7:0]  shift_reg;
    reg        shifting;
    reg        byte_done;
    reg        start_byte;
    reg [7:0]  next_byte;
    reg [7:0]  prescale;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            shift_reg <= 0;
            bit_phase <= 0;
            shifting  <= 0;
            byte_done <= 0;
            spi_mosi  <= 0;
            prescale  <= 0;
        end else begin
            byte_done <= 0;
            if (start_byte && !shifting) begin
                shift_reg <= next_byte;
                bit_phase <= 0;
                shifting  <= 1;
                prescale  <= 0;
                spi_mosi  <= next_byte[7];  // Pre-load MOSI before first SCK rise
            end else if (shifting) begin
                if (prescale == SCK_DIV) begin
                    prescale <= 0;
                    if (!bit_phase[0]) begin
                        // Even phase done: MOSI already stable from pre-load
                        // Transitioning to odd phase (SCK rises, slave samples)
                    end else begin
                        // Odd phase done: slave sampled. Pre-load next bit.
                        shift_reg <= {shift_reg[6:0], 1'b0};
                        spi_mosi  <= shift_reg[6];  // Next bit ready before SCK rises
                        if (bit_phase == 4'd15) begin
                            shifting  <= 0;
                            byte_done <= 1;
                        end
                    end
                    bit_phase <= bit_phase + 1;
                end else begin
                    prescale <= prescale + 1;
                end
            end
        end
    end

    // SCK output: SPI Mode 0 (CPOL=0, CPHA=0) — SCK idles LOW
    // During shifting: even phases LOW (data setup), odd phases HIGH (sample)
    // When idle (not shifting): LOW
    assign spi_sck = ~spi_cs_n & shifting & bit_phase[0];

    // =========================================================
    // Init command ROM (post-SLPOUT commands only)
    // =========================================================
    reg [4:0] init_idx;

    function [8:0] init_rom;
        input [4:0] idx;
        case (idx)
            // Color mode: RGB565
            5'd0:  init_rom = {1'b0, 8'h3A};   // COLMOD
            5'd1:  init_rom = {1'b1, 8'h55};   // 16-bit

            // Memory access control
            5'd2:  init_rom = {1'b0, 8'h36};   // MADCTL
            5'd3:  init_rom = {1'b1, 8'h60};   // Landscape (MV|MX)

            // Porch control
            5'd4:  init_rom = {1'b0, 8'hB2};   // PORCTRL
            5'd5:  init_rom = {1'b1, 8'h0C};
            5'd6:  init_rom = {1'b1, 8'h0C};
            5'd7:  init_rom = {1'b1, 8'h00};
            5'd8:  init_rom = {1'b1, 8'h33};
            5'd9:  init_rom = {1'b1, 8'h33};

            // Gate control
            5'd10: init_rom = {1'b0, 8'hB7};   // GCTRL
            5'd11: init_rom = {1'b1, 8'h35};

            // VCOM
            5'd12: init_rom = {1'b0, 8'hBB};   // VCOMS
            5'd13: init_rom = {1'b1, 8'h19};

            // LCM control
            5'd14: init_rom = {1'b0, 8'hC0};   // LCMCTRL
            5'd15: init_rom = {1'b1, 8'h2C};

            // VDV and VRH enable
            5'd16: init_rom = {1'b0, 8'hC2};   // VDVVRHEN
            5'd17: init_rom = {1'b1, 8'h01};

            // VRH setting
            5'd18: init_rom = {1'b0, 8'hC3};   // VRHS
            5'd19: init_rom = {1'b1, 8'h12};

            // VDV setting
            5'd20: init_rom = {1'b0, 8'hC4};   // VDVS
            5'd21: init_rom = {1'b1, 8'h20};

            // Frame rate control
            5'd22: init_rom = {1'b0, 8'hC6};   // FRCTRL2
            5'd23: init_rom = {1'b1, 8'h0F};   // 60 Hz

            // Power control
            5'd24: init_rom = {1'b0, 8'hD0};   // PWCTRL1
            5'd25: init_rom = {1'b1, 8'hA4};
            5'd26: init_rom = {1'b1, 8'hA1};

            // Inversion on (needed for correct colors on ST7789)
            5'd27: init_rom = {1'b0, 8'h21};   // INVON

            // Normal display mode
            5'd28: init_rom = {1'b0, 8'h13};   // NORON

            // Display on
            5'd29: init_rom = {1'b0, 8'h29};   // DISPON

            default: init_rom = 9'h1FF;         // Sentinel
        endcase
    endfunction

    // =========================================================
    // Window set ROM (CASET + RASET + RAMWR)
    // =========================================================
    localparam [15:0] COL_START = 16'd0;
    localparam [15:0] COL_END   = H_RES - 1;    // 319
    localparam [15:0] ROW_START = 16'd34;        // 172 centered in 240
    localparam [15:0] ROW_END   = 16'd34 + V_RES - 1;  // 205

    reg [3:0] win_idx;

    function [8:0] win_rom;
        input [3:0] idx;
        case (idx)
            4'd0:  win_rom = {1'b0, 8'h2A};            // CASET
            4'd1:  win_rom = {1'b1, COL_START[15:8]};
            4'd2:  win_rom = {1'b1, COL_START[7:0]};
            4'd3:  win_rom = {1'b1, COL_END[15:8]};
            4'd4:  win_rom = {1'b1, COL_END[7:0]};
            4'd5:  win_rom = {1'b0, 8'h2B};            // RASET
            4'd6:  win_rom = {1'b1, ROW_START[15:8]};
            4'd7:  win_rom = {1'b1, ROW_START[7:0]};
            4'd8:  win_rom = {1'b1, ROW_END[15:8]};
            4'd9:  win_rom = {1'b1, ROW_END[7:0]};
            4'd10: win_rom = {1'b0, 8'h2C};            // RAMWR
            default: win_rom = 9'h1FF;
        endcase
    endfunction

    // ROM lookup wires (can't bit-select function call in Verilog)
    wire [8:0] cur_init_val = init_rom(init_idx);
    wire [8:0] cur_win_val  = win_rom(win_idx);

    // =========================================================
    // Pixel streaming state
    // =========================================================
    reg        pixel_hi;
    reg [15:0] pixel_latch;

    // =========================================================
    // Main state machine
    // =========================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= ST_RESET;
            wait_cnt    <= 0;
            spi_cs_n    <= 1;
            spi_dc      <= 0;
            lcd_rst_n   <= 0;
            lcd_blk     <= 0;
            init_idx    <= 0;
            win_idx     <= 0;
            fb_addr     <= 0;
            pixel_hi    <= 0;
            pixel_latch <= 0;
            frame_done  <= 0;
            start_byte  <= 0;
            next_byte   <= 0;
        end else begin
            frame_done <= 0;
            start_byte <= 0;

            case (state)
                // ---- Assert hardware reset for 120ms ----
                ST_RESET: begin
                    lcd_rst_n <= 0;
                    lcd_blk   <= 0;
                    spi_cs_n  <= 1;
                    if (wait_cnt == RST_WAIT) begin
                        wait_cnt  <= 0;
                        lcd_rst_n <= 1;
                        state     <= ST_RESET_REL;
                    end else begin
                        wait_cnt <= wait_cnt + 1;
                    end
                end

                // ---- Wait 120ms after reset release ----
                ST_RESET_REL: begin
                    if (wait_cnt == RST_WAIT) begin
                        wait_cnt <= 0;
                        spi_cs_n <= 0;
                        state    <= ST_SLPOUT;
                    end else begin
                        wait_cnt <= wait_cnt + 1;
                    end
                end

                // ---- Send SLPOUT command alone ----
                ST_SLPOUT: begin
                    if (!shifting && !start_byte && !byte_done) begin
                        spi_dc     <= 0;       // Command
                        next_byte  <= 8'h11;   // SLPOUT
                        start_byte <= 1;
                    end
                    if (byte_done) begin
                        wait_cnt <= 0;
                        state    <= ST_SLPOUT_WAIT;
                    end
                end

                // ---- Wait 120ms after SLPOUT ----
                ST_SLPOUT_WAIT: begin
                    if (wait_cnt == SLPOUT_WAIT) begin
                        wait_cnt <= 0;
                        init_idx <= 0;
                        lcd_blk  <= 1;   // Turn on backlight after wake
                        state    <= ST_INIT;
                    end else begin
                        wait_cnt <= wait_cnt + 1;
                    end
                end

                // ---- Send remaining init commands ----
                ST_INIT: begin
                    if (!shifting && !start_byte) begin
                        if (cur_init_val == 9'h1FF) begin
                            win_idx <= 0;
                            state   <= ST_SET_WIN;
                        end else begin
                            spi_dc     <= cur_init_val[8];
                            next_byte  <= cur_init_val[7:0];
                            start_byte <= 1;
                            init_idx   <= init_idx + 1;
                        end
                    end
                end

                // ---- Set display window (CASET + RASET + RAMWR) ----
                ST_SET_WIN: begin
                    if (!shifting && !start_byte) begin
                        if (cur_win_val == 9'h1FF) begin
                            fb_addr  <= 0;
                            pixel_hi <= 1;
                            state    <= ST_PIXEL;
                        end else begin
                            spi_dc     <= cur_win_val[8];
                            next_byte  <= cur_win_val[7:0];
                            start_byte <= 1;
                            win_idx    <= win_idx + 1;
                        end
                    end
                end

                // ---- Stream pixels (RGB565, 2 bytes each) ----
                ST_PIXEL: begin
                    spi_dc <= 1;  // All pixel data

                    if (!shifting && !start_byte) begin
                        if (pixel_hi) begin
                            pixel_latch <= fb_data;
                            next_byte   <= fb_data[15:8];
                            start_byte  <= 1;
                            pixel_hi    <= 0;
                        end else begin
                            next_byte  <= pixel_latch[7:0];
                            start_byte <= 1;
                            pixel_hi   <= 1;
                            if (fb_addr == PIX_COUNT - 1) begin
                                fb_addr <= 0;
                                state   <= ST_FRAME_END;
                            end else begin
                                fb_addr <= fb_addr + 1;
                            end
                        end
                    end
                end

                // ---- Frame complete, restart ----
                ST_FRAME_END: begin
                    if (!shifting) begin
                        frame_done <= 1;
                        win_idx    <= 0;
                        state      <= ST_SET_WIN;
                    end
                end

                default: state <= ST_RESET;
            endcase
        end
    end

endmodule
