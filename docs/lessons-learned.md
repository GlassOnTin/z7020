# Lessons Learned — Pitfalls and Debugging War Stories

This document collects the non-obvious problems encountered during development: things that synthesized without errors but produced wrong results, timing issues that only manifested on hardware, and Vivado quirks that wasted hours. Each section describes the symptom, root cause, and fix.

## 1. Implicit Wire Bug (Vivado Warning 8-8895)

### Symptom
A 32-bit signal connected to a module port silently truncated to 1 bit. The design synthesized without errors. Simulation appeared correct. On hardware, the display showed garbage.

### Root Cause
In Verilog-2001, if you use a signal name in a port connection *before* declaring it as `reg`, Vivado infers an implicit 1-bit wire. The signal's `reg` declaration later in the file has no effect — the implicit wire has already been created.

```verilog
// BAD: signal used in instantiation before reg declaration
my_module u_mod (.data(my_signal));   // Vivado creates: wire my_signal;
// ...
reg [31:0] my_signal;                 // Too late — implicit wire already exists
```

### Fix
Declare all `reg` and `wire` signals in the declarations section *before* any module instantiation:

```verilog
// GOOD: declare first, instantiate second
reg [31:0] my_signal;
// ...
my_module u_mod (.data(my_signal));   // Uses the 32-bit reg
```

### Detection
Vivado emits warning 8-8895 for implicit wire creation, but it's buried among thousands of other warnings. In a synthesis log, search for:

```
grep "8-8895" vivado/build.log
```

## 2. Multi-Driven Nets Prevent BRAM Inference

### Symptom
Framebuffer BRAMs were synthesized as distributed LUT RAM instead of block RAM, causing massive LUT utilization (50%+) and timing failures.

### Root Cause
Two always blocks wrote to the same BRAM array — one for the normal write path and another for initialization. Vivado detected the multi-driven net and silently fell back to distributed RAM (which can be multi-driven) rather than block RAM (which cannot).

### Fix
Ensure each RAM array has exactly **one write port** (one always block that writes). If you need initialization, use `initial` blocks (which synthesize to BRAM init values) rather than a reset-time always block.

The `(* ram_style = "block" *)` attribute requests BRAM inference but cannot override a fundamental constraint — block RAMs physically have at most two ports.

## 3. Result Pulse Collision — Lost Iteration Counts

### Symptom
Rendered frames had missing pixels (random black dots), especially in areas with uniform iteration counts where many neurons would finish at similar times.

### Root Cause
Multiple neuron cores could emit `result_valid` on the same clock cycle. The framebuffer has a single write port, so only one result could be written per cycle. Without buffering, the other results were silently lost.

### Fix
Added a `result_pending` register (one bit per neuron) that captures *all* result_valid pulses simultaneously:

```verilog
for (n = 0; n < N_NEURONS; n = n + 1) begin
    if (result_valid[n])
        result_pending[n] <= 1'b1;
end
```

A separate drain priority encoder writes one pending result per cycle to the framebuffer. The key insight is that capture and drain can happen in the same always block because NBA (non-blocking assignment) semantics ensure the capture occurs even if the same bit is being cleared by the drain in the same cycle.

Additionally, the neuron assignment logic checks `!result_pending[k]` before assigning new work, preventing a neuron from being reused before its previous result has been drained.

## 4. State-Based vs. Counting Frame Completion

### Symptom
The auto-zoom controller would occasionally stop after several hundred frames. The `frame_done` signal never fired, leaving the system stuck.

### Root Cause
Frame completion was detected by counting: `if (pixels_done + 1 == PIX_COUNT)`. But the `pixels_done` increment occurred in the same always block as the `result_pending` drain, and the interaction between the one-cycle result_valid pulse and the drain logic could cause a count to be missed. Specifically, if a result_valid fires on the same cycle that the pending buffer for the same neuron is being cleared (from a previous result), the NBA semantics create a race where the pixel count doesn't increment.

### Fix
Replaced counting with a state-based conjunction:

```verilog
if (frame_busy && all_assigned &&
    (neuron_ready == {N_NEURONS{1'b1}}) &&
    (result_pending == {N_NEURONS{1'b0}}) &&
    (result_valid == {N_NEURONS{1'b0}})) begin
    frame_done <= 1;
end
```

This asks: "Has every pixel been assigned, is every neuron idle, and is nothing in flight?" — a question that doesn't depend on counting and can't lose track of results.

## 5. SPI MOSI Setup Time

### Symptom
The LCD displayed corrupted or shifted pixels. Reducing the SPI clock speed (increasing SCK_DIV) would make it work, but at the target 25 MHz it was unreliable.

### Root Cause
MOSI was being updated on the same clock edge that SCK transitioned high. The ST7789V3 samples MOSI on the rising edge of SCK. With both signals changing on the same FPGA clock edge, the MOSI value at the LCD's input was indeterminate — a classic setup time violation.

### Fix
Pre-load MOSI one phase ahead of the SCK rising edge:

1. **At `start_byte`**: Pre-load `spi_mosi <= next_byte[7]` (MSB ready before first SCK rise)
2. **During even phases (SCK low)**: Don't change MOSI (it was set during the previous odd phase)
3. **During odd phases (SCK high)**: Pre-load next bit `spi_mosi <= shift_reg[6]` for the *next* even/odd cycle

This gives a full half-cycle (10 ns at 25 MHz SCK) of setup time — well within the ST7789V3's requirement.

## 6. Verilog-2001 Limitations — No Unpacked Array Ports

### Symptom
Compilation error when trying to pass per-neuron result arrays as module ports.

### Root Cause
Verilog-2001 (the language standard supported by Vivado's synthesizer) does not allow unpacked array ports:

```verilog
// ILLEGAL in Verilog-2001:
output wire [15:0] result_pixel_id [0:N_NEURONS-1]
```

### Fix
Flatten arrays into wide buses and use part-select indexing:

```verilog
// Legal: flat bus
output wire [N_NEURONS*16-1:0] result_pixel_id

// Access element i:
result_pixel_id[i*16 +: 16]
```

The `[i*W +: W]` (indexed part-select) syntax was introduced in Verilog-2001 and is the standard way to work with flattened arrays. It selects W bits starting at position i*W.

## 7. SCK Gating — Extra Clock Edges

### Symptom
The first byte of each SPI transaction was corrupted. Subsequent bytes were fine.

### Root Cause
SCK was being generated as a free-running gated clock that could produce partial pulses at the start/end of a byte. The ST7789V3 would interpret the partial rising edge as a valid clock, sampling MOSI at the wrong time and shifting all subsequent bits.

### Fix
Generate SCK combinationally from the shifting state and phase counter:

```verilog
assign spi_sck = ~spi_cs_n & shifting & bit_phase[0];
```

This guarantees:
- SCK is LOW when not actively shifting (no edges between bytes)
- SCK transitions are synchronous with the system clock
- The first edge of each byte is a clean full-width pulse

## 8. Function Bit-Select in Vivado

### Symptom
Vivado synthesis error on seemingly valid Verilog.

### Root Cause
Vivado doesn't support bit-selecting from a function call result:

```verilog
// ILLEGAL in Vivado:
spi_dc <= init_rom(init_idx)[8];
```

### Fix
Assign the function result to a wire, then select from the wire:

```verilog
wire [8:0] cur_init_val = init_rom(init_idx);
// ...
spi_dc <= cur_init_val[8];
```

## Summary Table

| Bug | Symptom | Time to Debug | Lesson |
|-----|---------|---------------|--------|
| Implicit wire | Silent truncation | ~4 hours | Declare regs before instantiation |
| Multi-driven BRAM | LUT RAM fallback | ~2 hours | One write port per array |
| Result collision | Missing pixels | ~3 hours | Buffer one-cycle pulses |
| Counting completion | Stuck after N frames | ~6 hours | Use state checks, not counts |
| MOSI setup time | Corrupted display | ~8 hours | Pre-load data one phase ahead |
| Array ports | Won't compile | ~30 min | Flatten to wide buses |
| SCK gating | First byte corrupt | ~2 hours | Gate SCK with shifting flag |
| Function bit-select | Won't synthesize | ~15 min | Assign to wire first |
