// uart_tx.v — Simple 8N1 UART transmitter
// 115200 baud at 50 MHz (divider = 434)

`timescale 1ns / 1ps

module uart_tx #(
    parameter CLK_FREQ = 50_000_000,
    parameter BAUD     = 115200
)(
    input  wire       clk,
    input  wire       rst_n,
    input  wire       start,    // Pulse to begin sending data byte
    input  wire [7:0] data,     // Byte to transmit (latched on start)
    output reg        tx,       // UART TX line (idle high)
    output wire       busy      // High while transmitting
);

    localparam DIV = CLK_FREQ / BAUD;  // 434 at 50 MHz / 115200

    reg [9:0]  shift_reg;   // start bit + 8 data bits + stop bit
    reg [3:0]  bit_cnt;     // 0..9 (10 bits total)
    reg [15:0] baud_cnt;
    reg        active;

    assign busy = active;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tx        <= 1'b1;
            active    <= 1'b0;
            bit_cnt   <= 4'd0;
            baud_cnt  <= 16'd0;
            shift_reg <= 10'h3FF;
        end else if (!active) begin
            tx <= 1'b1;
            if (start) begin
                // Load shift register: {stop, data[7:0], start_bit}
                shift_reg <= {1'b1, data, 1'b0};
                active    <= 1'b1;
                bit_cnt   <= 4'd0;
                baud_cnt  <= 16'd0;
                tx        <= 1'b0;  // Start bit
            end
        end else begin
            if (baud_cnt == DIV[15:0] - 1) begin
                baud_cnt <= 16'd0;
                if (bit_cnt == 4'd9) begin
                    // Done — all 10 bits sent
                    active <= 1'b0;
                    tx     <= 1'b1;
                end else begin
                    bit_cnt   <= bit_cnt + 1;
                    shift_reg <= {1'b1, shift_reg[9:1]};
                    tx        <= shift_reg[1];
                end
            end else begin
                baud_cnt <= baud_cnt + 1;
            end
        end
    end

endmodule
