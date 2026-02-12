// boot_msg.v — One-shot boot message sequencer
// Sends "MANDELBROT QSPI OK\r\n" via UART TX on reset release

`timescale 1ns / 1ps

module boot_msg (
    input  wire clk,
    input  wire rst_n,
    output wire tx
);

    // Message: "MANDELBROT QSPI OK\r\n" (20 chars)
    localparam MSG_LEN = 20;

    // Message ROM
    reg [7:0] msg_rom [0:MSG_LEN-1];
    initial begin
        msg_rom[ 0] = "M";
        msg_rom[ 1] = "A";
        msg_rom[ 2] = "N";
        msg_rom[ 3] = "D";
        msg_rom[ 4] = "E";
        msg_rom[ 5] = "L";
        msg_rom[ 6] = "B";
        msg_rom[ 7] = "R";
        msg_rom[ 8] = "O";
        msg_rom[ 9] = "T";
        msg_rom[10] = " ";
        msg_rom[11] = "Q";
        msg_rom[12] = "S";
        msg_rom[13] = "P";
        msg_rom[14] = "I";
        msg_rom[15] = " ";
        msg_rom[16] = "O";
        msg_rom[17] = "K";
        msg_rom[18] = 8'h0D;  // \r
        msg_rom[19] = 8'h0A;  // \n
    end

    // UART TX instance
    wire       uart_busy;
    reg        uart_start;
    reg  [7:0] uart_data;

    uart_tx u_uart (
        .clk   (clk),
        .rst_n (rst_n),
        .start (uart_start),
        .data  (uart_data),
        .tx    (tx),
        .busy  (uart_busy)
    );

    // State machine
    localparam S_IDLE  = 0,
               S_WAIT  = 1,
               S_DELAY = 2;

    reg [1:0]   state;
    reg [4:0]   char_idx;
    reg [26:0]  delay_cnt;  // ~2.7s at 50 MHz

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            char_idx   <= 5'd0;
            uart_start <= 1'b0;
            uart_data  <= 8'd0;
            delay_cnt  <= 27'd0;
        end else begin
            uart_start <= 1'b0;

            case (state)
                S_IDLE: begin
                    // Start sending message
                    char_idx   <= 5'd0;
                    uart_data  <= msg_rom[0];
                    uart_start <= 1'b1;
                    state      <= S_WAIT;
                end

                S_WAIT: begin
                    // Wait for current byte to finish
                    if (!uart_busy && !uart_start) begin
                        if (char_idx == MSG_LEN - 1) begin
                            // Message done — wait ~2s then repeat
                            delay_cnt <= 27'd0;
                            state     <= S_DELAY;
                        end else begin
                            char_idx   <= char_idx + 1;
                            uart_data  <= msg_rom[char_idx + 1];
                            uart_start <= 1'b1;
                        end
                    end
                end

                S_DELAY: begin
                    if (delay_cnt == 27'd100_000_000) begin
                        state <= S_IDLE;  // Repeat message
                    end else begin
                        delay_cnt <= delay_cnt + 1;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
