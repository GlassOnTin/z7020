// spi_lcd_test.v -- Standalone LCD diagnostic test for ST7789V3 1.47" 320x172
//
// Self-contained top module (no Mandelbrot dependencies) that:
//   1. Resets the LCD via hardware reset line
//   2. Reads the RDDID (0x04) register via bidirectional SDA
//   3. Reports the 3-byte ID over UART (115200 baud)
//   4. Initializes the display with a minimal command sequence
//   5. Fills the entire screen with solid red (0xF800)
//   6. Reports completion over UART
//
// LED diagnostics:
//   led_alive  -- 1 Hz heartbeat (proves FPGA bitstream is running)
//   led_frame  -- After ID read: 4 Hz blink if non-zero (SPI working),
//                 0.5 Hz blink if all-zero (SPI broken).
//                 After fill: holds steady (toggled once on completion).
//
// SPI clock: ~500 KHz (SCK_DIV=49 at 50 MHz system clock)
// UART: 115200 baud, 8N1

`timescale 1ns / 1ps

module spi_lcd_test #(
    parameter H_RES       = 320,
    parameter V_RES       = 172,
    parameter PIX_COUNT   = H_RES * V_RES,   // 55040
    parameter SCK_DIV     = 49,               // ~500 KHz SCK
    parameter RST_WAIT    = 6_000_000,        // 120ms at 50 MHz
    parameter SLPOUT_WAIT = 6_000_000,        // 120ms at 50 MHz
    parameter UART_DIV    = 434               // 50 MHz / 115200 ~ 434
)(
    // Clock and reset
    input  wire        clk_50m,       // 50 MHz PL crystal oscillator
    input  wire        rst_n_in,      // Active-low pushbutton (K21)

    // SPI interface to ST7789V3
    output reg         spi_cs_n,
    output wire        spi_sck,       // Combined SCK for write and read modes
    inout  wire        spi_sda,       // Bidirectional: MOSI out / MISO in
    output reg         spi_dc,        // Data/Command (0 = command)
    output reg         lcd_rst_n,     // Hardware reset (active low)
    output reg         lcd_blk,       // Backlight enable (active high)

    // LED diagnostics
    output reg         led_frame,     // Diagnostic pattern (see header)
    output wire        led_alive,     // 1 Hz heartbeat

    // UART
    output wire        tx_out         // 115200 8N1 diagnostic output
);

    // =========================================================
    // Clock assignment and reset synchronizer
    // =========================================================
    wire clk = clk_50m;
    reg [2:0] rst_sync;
    wire rst_n = rst_sync[2];

    always @(posedge clk or negedge rst_n_in) begin
        if (!rst_n_in)
            rst_sync <= 3'b000;
        else
            rst_sync <= {rst_sync[1:0], 1'b1};
    end

    // =========================================================
    // Bidirectional SDA tristate
    // =========================================================
    reg        sda_oe;        // 1 = driving output, 0 = high-Z (reading)
    reg        sda_out;       // Output data when sda_oe=1
    wire       sda_in = spi_sda;

    assign spi_sda = sda_oe ? sda_out : 1'bz;

    // =========================================================
    // Main state machine encoding
    // =========================================================
    localparam [3:0] S_RESET       = 4'd0,
                     S_RESET_WAIT  = 4'd1,
                     S_RDDID_CMD   = 4'd2,
                     S_RDDID_READ  = 4'd3,
                     S_REPORT      = 4'd4,
                     S_SLPOUT      = 4'd5,
                     S_SLPOUT_WAIT = 4'd6,
                     S_INIT        = 4'd7,
                     S_SET_WIN     = 4'd8,
                     S_FILL        = 4'd9,
                     S_DONE        = 4'd10;

    reg [3:0]  state;
    reg [22:0] wait_cnt;

    // =========================================================
    // SPI byte-level shift engine (write mode)
    // =========================================================
    // Phase-counted approach: bit_phase 0..15
    //   Even phases: drive MOSI with MSB of shift register, SCK low
    //   Odd phases:  SCK high (slave samples), then advance shift reg
    // Prescaler divides system clock by (SCK_DIV+1) per half-phase.
    reg [3:0]  bit_phase;
    reg [7:0]  shift_out;
    reg        shifting;       // Write-mode shift in progress
    reg        byte_done;      // Pulses 1 cycle when write byte completes
    reg        start_byte;     // Pulse to begin shifting next_byte
    reg [7:0]  next_byte;
    reg [7:0]  prescale;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            shift_out <= 8'd0;
            bit_phase <= 4'd0;
            shifting  <= 1'b0;
            byte_done <= 1'b0;
            sda_out   <= 1'b0;
            prescale  <= 8'd0;
        end else begin
            byte_done <= 1'b0;

            if (start_byte && !shifting) begin
                shift_out <= next_byte;
                bit_phase <= 4'd0;
                shifting  <= 1'b1;
                prescale  <= 8'd0;
                sda_out   <= next_byte[7];  // Pre-load MOSI before first SCK rise
            end else if (shifting && sda_oe) begin
                // Only shift when in output mode (write)
                if (prescale == SCK_DIV) begin
                    prescale <= 8'd0;
                    if (!bit_phase[0]) begin
                        // Even phase done: MOSI already stable from pre-load
                        // Transitioning to odd phase (SCK rises, slave samples)
                    end else begin
                        // Odd phase done: slave sampled. Pre-load next bit.
                        shift_out <= {shift_out[6:0], 1'b0};
                        sda_out   <= shift_out[6];  // Next bit ready before SCK rises
                        if (bit_phase == 4'd15) begin
                            shifting  <= 1'b0;
                            byte_done <= 1'b1;
                        end
                    end
                    bit_phase <= bit_phase + 1;
                end else begin
                    prescale <= prescale + 1;
                end
            end
        end
    end

    // =========================================================
    // SPI read engine (for RDDID)
    // =========================================================
    // During read: sda_oe=0 (high-Z). We generate SCK and sample sda_in
    // on the falling edge of read_phase (end of SCK-high half-cycle).
    // We read 32 bits: 1 dummy byte + 3 ID bytes.
    reg [4:0]  read_bit_cnt;     // 0..31 (32 bits total)
    reg        read_active;
    reg        read_done;
    reg [7:0]  read_prescale;
    reg        read_phase;       // 0 = SCK low half, 1 = SCK high half
    reg [31:0] read_shift;       // Captured bits (MSB first)
    reg [23:0] id_bytes;         // Final 3 ID bytes (drop first dummy byte)
    reg        start_read;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            read_active   <= 1'b0;
            read_done     <= 1'b0;
            read_bit_cnt  <= 5'd0;
            read_prescale <= 8'd0;
            read_phase    <= 1'b0;
            read_shift    <= 32'd0;
            id_bytes      <= 24'd0;
        end else begin
            read_done <= 1'b0;

            if (start_read && !read_active) begin
                read_active   <= 1'b1;
                read_bit_cnt  <= 5'd0;
                read_prescale <= 8'd0;
                read_phase    <= 1'b0;
                read_shift    <= 32'd0;
            end else if (read_active) begin
                if (read_prescale == SCK_DIV) begin
                    read_prescale <= 8'd0;
                    if (!read_phase) begin
                        // Entering SCK-high half-cycle
                        read_phase <= 1'b1;
                    end else begin
                        // End of SCK-high: sample sda_in, return SCK low
                        read_shift <= {read_shift[30:0], sda_in};
                        read_phase <= 1'b0;
                        if (read_bit_cnt == 5'd31) begin
                            read_active <= 1'b0;
                            read_done   <= 1'b1;
                            // Bits [31:24] are dummy; [23:0] are ID1, ID2, ID3
                            id_bytes <= {read_shift[22:0], sda_in};
                        end else begin
                            read_bit_cnt <= read_bit_cnt + 1;
                        end
                    end
                end else begin
                    read_prescale <= read_prescale + 1;
                end
            end
        end
    end

    // =========================================================
    // Combined SCK output (write mode + read mode)
    // =========================================================
    // Write mode: SCK high during odd bit_phase while shifting in output mode
    // Read mode:  SCK high during read_phase while read_active
    // Both gated by CS asserted (low)
    wire sck_from_write = shifting & sda_oe & bit_phase[0];
    wire sck_from_read  = read_active & read_phase;

    assign spi_sck = (sck_from_write | sck_from_read) & ~spi_cs_n;

    // =========================================================
    // UART transmitter (115200 8N1)
    // =========================================================
    reg [7:0]  uart_tx_data;
    reg        uart_tx_start;
    reg        uart_tx_busy;
    reg [9:0]  uart_shift;       // {stop, data[7:0], start}
    reg [3:0]  uart_bit_cnt;     // 0..9 (10 bits per frame)
    reg [8:0]  uart_baud_cnt;
    reg        uart_tx_reg;

    assign tx_out = uart_tx_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            uart_tx_reg   <= 1'b1;  // Idle high
            uart_tx_busy  <= 1'b0;
            uart_shift    <= 10'h3FF;
            uart_bit_cnt  <= 4'd0;
            uart_baud_cnt <= 9'd0;
        end else begin
            if (uart_tx_start && !uart_tx_busy) begin
                // Load: {stop=1, data[7..0], start=0}, shift out LSB first
                uart_shift    <= {1'b1, uart_tx_data, 1'b0};
                uart_bit_cnt  <= 4'd0;
                uart_baud_cnt <= 9'd0;
                uart_tx_busy  <= 1'b1;
                uart_tx_reg   <= 1'b0;  // Start bit drives line low
            end else if (uart_tx_busy) begin
                if (uart_baud_cnt == UART_DIV - 1) begin
                    uart_baud_cnt <= 9'd0;
                    uart_bit_cnt  <= uart_bit_cnt + 1;
                    if (uart_bit_cnt == 4'd9) begin
                        // All 10 bits sent
                        uart_tx_busy <= 1'b0;
                        uart_tx_reg  <= 1'b1;  // Idle
                    end else begin
                        // Next bit (LSB first)
                        uart_shift  <= {1'b1, uart_shift[9:1]};
                        uart_tx_reg <= uart_shift[1];
                    end
                end else begin
                    uart_baud_cnt <= uart_baud_cnt + 1;
                end
            end
        end
    end

    // =========================================================
    // UART message sequencer
    // =========================================================
    // Two messages:
    //   msg_id=0: "ID:XX XX XX\r\n"  (13 chars)
    //   msg_id=1: "FILL DONE\r\n"    (11 chars)
    reg [5:0]  msg_idx;
    reg        msg_sending;
    reg [1:0]  msg_id;           // 0 = ID report, 1 = FILL DONE
    reg        msg_start;
    reg        msg_done;

    // Hex nibble to ASCII: '0'-'9' or 'A'-'F'
    function [7:0] hex_ascii;
        input [3:0] nibble;
        hex_ascii = (nibble < 4'd10) ? (8'h30 + {4'd0, nibble})
                                     : (8'h41 + {4'd0, nibble - 4'd10});
    endfunction

    // Message ROM: returns character for (msg_id, index, id_bytes)
    function [7:0] msg_char;
        input [1:0]  which;
        input [5:0]  idx;
        input [23:0] id;
        case (which)
            2'd0: begin  // "ID:XX XX XX\r\n"
                case (idx)
                    6'd0:  msg_char = "I";
                    6'd1:  msg_char = "D";
                    6'd2:  msg_char = ":";
                    6'd3:  msg_char = hex_ascii(id[23:20]);
                    6'd4:  msg_char = hex_ascii(id[19:16]);
                    6'd5:  msg_char = " ";
                    6'd6:  msg_char = hex_ascii(id[15:12]);
                    6'd7:  msg_char = hex_ascii(id[11:8]);
                    6'd8:  msg_char = " ";
                    6'd9:  msg_char = hex_ascii(id[7:4]);
                    6'd10: msg_char = hex_ascii(id[3:0]);
                    6'd11: msg_char = 8'h0D;  // \r
                    6'd12: msg_char = 8'h0A;  // \n
                    default: msg_char = 8'h00;
                endcase
            end
            2'd1: begin  // "FILL DONE\r\n"
                case (idx)
                    6'd0:  msg_char = "F";
                    6'd1:  msg_char = "I";
                    6'd2:  msg_char = "L";
                    6'd3:  msg_char = "L";
                    6'd4:  msg_char = " ";
                    6'd5:  msg_char = "D";
                    6'd6:  msg_char = "O";
                    6'd7:  msg_char = "N";
                    6'd8:  msg_char = "E";
                    6'd9:  msg_char = 8'h0D;  // \r
                    6'd10: msg_char = 8'h0A;  // \n
                    default: msg_char = 8'h00;
                endcase
            end
            default: msg_char = 8'h00;
        endcase
    endfunction

    wire [7:0] cur_msg_char = msg_char(msg_id, msg_idx, id_bytes);
    wire [5:0] msg_len      = (msg_id == 2'd0) ? 6'd13 : 6'd11;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            msg_sending   <= 1'b0;
            msg_idx       <= 6'd0;
            msg_done      <= 1'b0;
            uart_tx_start <= 1'b0;
            uart_tx_data  <= 8'd0;
        end else begin
            uart_tx_start <= 1'b0;
            msg_done      <= 1'b0;

            if (msg_start && !msg_sending) begin
                msg_sending <= 1'b1;
                msg_idx     <= 6'd0;
            end else if (msg_sending) begin
                if (!uart_tx_busy && !uart_tx_start) begin
                    if (msg_idx == msg_len) begin
                        msg_sending <= 1'b0;
                        msg_done    <= 1'b1;
                    end else begin
                        uart_tx_data  <= cur_msg_char;
                        uart_tx_start <= 1'b1;
                        msg_idx       <= msg_idx + 1;
                    end
                end
            end
        end
    end

    // =========================================================
    // Init command ROM (minimal set for ST7789V3)
    // =========================================================
    // Format: {dc, byte} -- dc=0 for command, dc=1 for data parameter
    reg [3:0] init_idx;

    function [8:0] init_rom;
        input [3:0] idx;
        case (idx)
            4'd0:  init_rom = {1'b0, 8'h3A};  // COLMOD command
            4'd1:  init_rom = {1'b1, 8'h55};  //   16-bit RGB565
            4'd2:  init_rom = {1'b0, 8'h36};  // MADCTL command
            4'd3:  init_rom = {1'b1, 8'h60};  //   Landscape (MV|MX)
            4'd4:  init_rom = {1'b0, 8'h21};  // INVON (no parameter)
            4'd5:  init_rom = {1'b0, 8'h13};  // NORON (no parameter)
            4'd6:  init_rom = {1'b0, 8'h29};  // DISPON (no parameter)
            default: init_rom = 9'h1FF;        // Sentinel (end of sequence)
        endcase
    endfunction

    // =========================================================
    // Window set ROM: CASET + RASET + RAMWR
    // =========================================================
    // With MADCTL=0x60 (landscape MV|MX):
    //   CASET 0..319  (long axis)
    //   RASET 34..205 (short axis, 172 rows centered in 240)
    localparam [15:0] COL_START = 16'd0;
    localparam [15:0] COL_END   = H_RES - 1;               // 319
    localparam [15:0] ROW_START = 16'd34;
    localparam [15:0] ROW_END   = 16'd34 + V_RES - 1;      // 205

    reg [3:0] win_idx;

    function [8:0] win_rom;
        input [3:0] idx;
        case (idx)
            4'd0:  win_rom = {1'b0, 8'h2A};             // CASET
            4'd1:  win_rom = {1'b1, COL_START[15:8]};   // 0x00
            4'd2:  win_rom = {1'b1, COL_START[7:0]};    // 0x00
            4'd3:  win_rom = {1'b1, COL_END[15:8]};     // 0x01
            4'd4:  win_rom = {1'b1, COL_END[7:0]};      // 0x3F
            4'd5:  win_rom = {1'b0, 8'h2B};             // RASET
            4'd6:  win_rom = {1'b1, ROW_START[15:8]};   // 0x00
            4'd7:  win_rom = {1'b1, ROW_START[7:0]};    // 0x22
            4'd8:  win_rom = {1'b1, ROW_END[15:8]};     // 0x00
            4'd9:  win_rom = {1'b1, ROW_END[7:0]};      // 0xCD
            4'd10: win_rom = {1'b0, 8'h2C};             // RAMWR
            default: win_rom = 9'h1FF;                   // Sentinel
        endcase
    endfunction

    wire [8:0] cur_init_val = init_rom(init_idx);
    wire [8:0] cur_win_val  = win_rom(win_idx);

    // =========================================================
    // Pixel streaming state
    // =========================================================
    reg        pixel_hi;          // 1 = send high byte next, 0 = low byte
    reg [16:0] pixel_cnt;         // 0 .. PIX_COUNT-1

    // =========================================================
    // LED diagnostics
    // =========================================================
    reg        id_nonzero;        // Latched: any ID byte was non-zero
    reg        fill_complete;     // Latched: pixel fill has finished

    // Blink counter: 26 bits supports up to 1.34s half-periods at 50 MHz
    reg [25:0] blink_cnt;
    reg        fill_complete_prev;  // For edge detection

    // 4 Hz blink: toggle every 6.25M clocks (125ms)
    // 0.5 Hz blink: toggle every 50M clocks -- needs 26 bits (max ~67M)
    localparam [25:0] BLINK_FAST_LIMIT = 26'd6_250_000  - 1;  // 4 Hz
    localparam [25:0] BLINK_SLOW_LIMIT = 26'd50_000_000 - 1;  // 0.5 Hz

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            led_frame          <= 1'b0;
            blink_cnt          <= 26'd0;
            fill_complete_prev <= 1'b0;
        end else begin
            fill_complete_prev <= fill_complete;

            if (fill_complete && !fill_complete_prev) begin
                // Rising edge of fill_complete: toggle once, then hold
                led_frame <= ~led_frame;
            end else if (!fill_complete && (state >= S_REPORT)) begin
                // Between ID read and fill completion: blink based on ID result
                if (id_nonzero) begin
                    // SPI appears to work: fast 4 Hz blink
                    if (blink_cnt >= BLINK_FAST_LIMIT) begin
                        blink_cnt <= 26'd0;
                        led_frame <= ~led_frame;
                    end else begin
                        blink_cnt <= blink_cnt + 1;
                    end
                end else begin
                    // ID was all-zero (SPI possibly broken): slow 0.5 Hz blink
                    if (blink_cnt >= BLINK_SLOW_LIMIT) begin
                        blink_cnt <= 26'd0;
                        led_frame <= ~led_frame;
                    end else begin
                        blink_cnt <= blink_cnt + 1;
                    end
                end
            end
            // After fill_complete: led_frame holds steady
        end
    end

    // Alive LED: 1 Hz heartbeat (toggle every 25M clocks)
    reg [24:0] alive_cnt;
    reg        alive_r;
    assign led_alive = alive_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            alive_cnt <= 25'd0;
            alive_r   <= 1'b0;
        end else begin
            if (alive_cnt == 25_000_000 - 1) begin
                alive_cnt <= 25'd0;
                alive_r   <= ~alive_r;
            end else begin
                alive_cnt <= alive_cnt + 1;
            end
        end
    end

    // =========================================================
    // Main FSM
    // =========================================================
    reg rddid_cmd_sent;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= S_RESET;
            wait_cnt       <= 23'd0;
            spi_cs_n       <= 1'b1;
            spi_dc         <= 1'b0;
            lcd_rst_n      <= 1'b0;
            lcd_blk        <= 1'b0;
            sda_oe         <= 1'b1;
            start_byte     <= 1'b0;
            next_byte      <= 8'd0;
            start_read     <= 1'b0;
            init_idx       <= 4'd0;
            win_idx        <= 4'd0;
            pixel_hi       <= 1'b0;
            pixel_cnt      <= 17'd0;
            msg_start      <= 1'b0;
            msg_id         <= 2'd0;
            rddid_cmd_sent <= 1'b0;
            id_nonzero     <= 1'b0;
            fill_complete  <= 1'b0;
        end else begin
            // Default: deassert one-cycle pulses
            start_byte <= 1'b0;
            start_read <= 1'b0;
            msg_start  <= 1'b0;

            case (state)

                // ------------------------------------------------
                // S_RESET: Drive lcd_rst_n LOW for 120ms
                // ------------------------------------------------
                S_RESET: begin
                    lcd_rst_n <= 1'b0;
                    lcd_blk   <= 1'b0;
                    spi_cs_n  <= 1'b1;
                    sda_oe    <= 1'b1;
                    if (wait_cnt == RST_WAIT - 1) begin
                        wait_cnt  <= 23'd0;
                        lcd_rst_n <= 1'b1;
                        state     <= S_RESET_WAIT;
                    end else begin
                        wait_cnt <= wait_cnt + 1;
                    end
                end

                // ------------------------------------------------
                // S_RESET_WAIT: lcd_rst_n released, wait 120ms
                // ------------------------------------------------
                S_RESET_WAIT: begin
                    lcd_rst_n <= 1'b1;
                    if (wait_cnt == RST_WAIT - 1) begin
                        wait_cnt       <= 23'd0;
                        state          <= S_RDDID_CMD;
                        rddid_cmd_sent <= 1'b0;
                    end else begin
                        wait_cnt <= wait_cnt + 1;
                    end
                end

                // ------------------------------------------------
                // S_RDDID_CMD: CS low, DC=0, send 0x04 (RDDID)
                // ------------------------------------------------
                S_RDDID_CMD: begin
                    spi_cs_n <= 1'b0;
                    spi_dc   <= 1'b0;      // Command mode
                    sda_oe   <= 1'b1;      // Output mode for command byte

                    if (!rddid_cmd_sent) begin
                        if (!shifting && !start_byte && !byte_done) begin
                            next_byte  <= 8'h04;  // RDDID
                            start_byte <= 1'b1;
                        end
                        if (byte_done) begin
                            rddid_cmd_sent <= 1'b1;
                        end
                    end else begin
                        // Command byte sent; move to read phase
                        state <= S_RDDID_READ;
                    end
                end

                // ------------------------------------------------
                // S_RDDID_READ: Tristate SDA, clock in 32 bits
                //   (8 dummy + 24 ID bits)
                // ------------------------------------------------
                S_RDDID_READ: begin
                    sda_oe <= 1'b0;  // Release SDA for input

                    if (!read_active && !read_done && !start_read) begin
                        start_read <= 1'b1;
                    end

                    if (read_done) begin
                        id_nonzero <= (id_bytes != 24'd0);
                        spi_cs_n   <= 1'b1;  // Deassert CS
                        sda_oe     <= 1'b1;  // Resume output mode
                        msg_id     <= 2'd0;  // Select "ID:..." message
                        msg_start  <= 1'b1;
                        state      <= S_REPORT;
                    end
                end

                // ------------------------------------------------
                // S_REPORT: UART sends "ID:XX XX XX\r\n"
                // ------------------------------------------------
                S_REPORT: begin
                    spi_cs_n <= 1'b1;
                    sda_oe   <= 1'b1;

                    if (msg_done) begin
                        state <= S_SLPOUT;
                    end
                end

                // ------------------------------------------------
                // S_SLPOUT: CS low, DC=0, send 0x11 (Sleep Out)
                // ------------------------------------------------
                S_SLPOUT: begin
                    spi_cs_n <= 1'b0;
                    spi_dc   <= 1'b0;
                    sda_oe   <= 1'b1;

                    if (!shifting && !start_byte && !byte_done) begin
                        next_byte  <= 8'h11;
                        start_byte <= 1'b1;
                    end
                    if (byte_done) begin
                        wait_cnt <= 23'd0;
                        state    <= S_SLPOUT_WAIT;
                    end
                end

                // ------------------------------------------------
                // S_SLPOUT_WAIT: Wait 120ms after SLPOUT
                // ------------------------------------------------
                S_SLPOUT_WAIT: begin
                    if (wait_cnt == SLPOUT_WAIT - 1) begin
                        wait_cnt <= 23'd0;
                        init_idx <= 4'd0;
                        lcd_blk  <= 1'b1;   // Backlight on
                        state    <= S_INIT;
                    end else begin
                        wait_cnt <= wait_cnt + 1;
                    end
                end

                // ------------------------------------------------
                // S_INIT: Send minimal init commands from ROM
                // ------------------------------------------------
                S_INIT: begin
                    sda_oe <= 1'b1;
                    if (!shifting && !start_byte) begin
                        if (cur_init_val == 9'h1FF) begin
                            // All init commands sent
                            win_idx <= 4'd0;
                            state   <= S_SET_WIN;
                        end else begin
                            spi_dc     <= cur_init_val[8];
                            next_byte  <= cur_init_val[7:0];
                            start_byte <= 1'b1;
                            init_idx   <= init_idx + 1;
                        end
                    end
                end

                // ------------------------------------------------
                // S_SET_WIN: CASET + RASET + RAMWR from window ROM
                // ------------------------------------------------
                S_SET_WIN: begin
                    sda_oe <= 1'b1;
                    if (!shifting && !start_byte) begin
                        if (cur_win_val == 9'h1FF) begin
                            pixel_hi  <= 1'b1;
                            pixel_cnt <= 17'd0;
                            state     <= S_FILL;
                        end else begin
                            spi_dc     <= cur_win_val[8];
                            next_byte  <= cur_win_val[7:0];
                            start_byte <= 1'b1;
                            win_idx    <= win_idx + 1;
                        end
                    end
                end

                // ------------------------------------------------
                // S_FILL: Stream 55040 pixels of solid red (0xF800)
                //   Two bytes per pixel: 0xF8 (high), 0x00 (low)
                // ------------------------------------------------
                S_FILL: begin
                    spi_dc <= 1'b1;  // Data mode for pixels
                    sda_oe <= 1'b1;

                    if (!shifting && !start_byte) begin
                        if (pixel_hi) begin
                            next_byte  <= 8'hF8;  // Red high byte
                            start_byte <= 1'b1;
                            pixel_hi   <= 1'b0;
                        end else begin
                            next_byte  <= 8'h00;  // Red low byte
                            start_byte <= 1'b1;
                            pixel_hi   <= 1'b1;

                            if (pixel_cnt == PIX_COUNT - 1) begin
                                state <= S_DONE;
                            end else begin
                                pixel_cnt <= pixel_cnt + 1;
                            end
                        end
                    end
                end

                // ------------------------------------------------
                // S_DONE: Wait for last byte, report, hold
                // ------------------------------------------------
                S_DONE: begin
                    if (!shifting) begin
                        if (!fill_complete) begin
                            fill_complete <= 1'b1;
                            spi_cs_n      <= 1'b1;
                            msg_id        <= 2'd1;  // "FILL DONE"
                            msg_start     <= 1'b1;
                        end
                        // Remain in S_DONE (test complete)
                    end
                end

                default: state <= S_RESET;
            endcase
        end
    end

endmodule
