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

## 9. RGB565 Bit Extraction Off-By-One (The MLP Color Bug)

### Symptom
The neural network display showed pure color bands — solid red, green, blue, cyan, yellow, magenta — instead of smooth gradients. Colors appeared to be from the set {0, max} for each channel, as if the output was binary. This persisted through multiple "fix" attempts.

### Root Cause
The `pack_rgb565` function extracted the wrong bits from the Q4.28 shifted value. The code was:

```verilog
// WRONG: misses bit 28
r5 = rs[27:23];   // For 5-bit R channel
g6 = gs[27:22];   // For 6-bit G channel
b5 = bs[27:23];   // For 5-bit B channel
```

The input to pack_rgb565 is a Q4.28 value shifted from [-1,+1] to [0,+2) by adding 1.0 (`0x10000000`). In Q4.28, **bit 28 is the 1.0 position** — the most significant bit of the [0, 2) range. By extracting `[27:23]` instead of `[28:24]`, the function ignored bit 28 entirely.

This caused a sawtooth mapping:

| sin() output | Shifted value | Correct (bits 28:24) | Wrong (bits 27:23) |
|-------------|---------------|---------------------|--------------------|
| -1.0 | 0.0 (0x00000000) | 00000 = 0 | 00000 = 0 |
| -0.5 | 0.5 (0x08000000) | 01000 = 8 | 01000 = 8 |
| 0.0 | 1.0 (0x10000000) | **10000 = 16** | **00000 = 0** |
| +0.5 | 1.5 (0x18000000) | 11000 = 24 | 01000 = 8 |
| +1.0 | 2.0 → clamped to 31 | 31 | 31 |

At sin=0, the wrong extraction wraps from 15 back to 0. Values near zero (the most common sin() output) produced either very dark or very bright pixels, with nothing in between. Combined across R, G, B channels, this created the binary color band appearance.

### Fix

```verilog
// CORRECT: bit 28 is the 1.0 position in Q4.28
r5 = rs[28:24];   // 5 bits spanning [0, 2.0)
g6 = gs[28:23];   // 6 bits spanning [0, 2.0)
b5 = bs[28:24];   // 5 bits spanning [0, 2.0)
```

### Why this was hard to find

Three other bugs were fixed first, each partially responsible for bad colors:
1. sin() activation was skipped on the output layer (training used sin, hardware didn't)
2. Output values were read from `acc_sat` (pre-activation) instead of `sin_output` (post-activation)
3. The training script used tanh() while the hardware had no activation

Each fix was necessary but insufficient. The bit extraction bug was the final one, and it produced identical visual symptoms to the others — making it impossible to tell which fix was "the one" until all three were resolved. This is a classic symptom-aliasing problem: multiple bugs produce the same visual output, so fixing one doesn't improve the result.

### Lesson
When converting fixed-point values to integer ranges, draw out the bit positions explicitly:

```
Q4.28 value 1.0 = 0x10000000:
Bit:  31 30 29 28 27 26 25 24 23 22 ...
       0  0  0  1  0  0  0  0  0  0 ...
               ↑
       This is the 1.0 position.
       Your extraction MUST include it.
```

If your range is [0, 2.0), you need bit 28 as the MSB. If your range is [0, 1.0), you don't. Getting this wrong produces a modular arithmetic wrap that looks like quantization to {0, max}.

## 10. Training/Hardware Activation Mismatch

### Symptom
Neural network output showed thresholded colors (same visual as bug #9) — values appeared clamped to extremes.

### Root Cause
The Python training script used `tanh()` on the output layer, while the FPGA hardware applied no activation function on the output layer. This meant the trained weights expected the output to be squashed through tanh(), but the FPGA passed through raw accumulator values (range [-8, +8]) directly to pack_rgb565, which clamped everything outside [-1, +1].

### Fix (two parts)

**Part 1**: Change training to use sin() on the output layer (matching the hardware, which has sin() LUTs available):

```python
# Before:
def forward(self, x):
    for layer in self.layers:
        x = layer(x)
    return torch.tanh(self.output_layer(x))  # tanh squash

# After:
def forward(self, x):
    for layer in self.layers:
        x = layer(x)
    return torch.sin(self.output_layer(x))   # sin matches FPGA
```

**Part 2**: Change hardware to apply sin() on all layers including output:

```verilog
// Before: output layer skipped activation
S_BIAS_ADD: begin
    acc <= acc + bias;
    if (cur_layer == N_LAYERS - 1)
        state <= S_NEXT_N;      // Skip sin() for output
    else
        state <= S_ACTIVATE;    // sin() for hidden only
end

// After: all layers go through sin()
S_BIAS_ADD: begin
    acc <= acc + bias;
    state <= S_ACTIVATE;        // Always apply sin()
end
```

**Part 3**: Read post-activation values for output colors:

```verilog
// Before: read pre-sin accumulator
case (cur_neuron[1:0])
    2'd0: out_r <= acc_sat;     // Range [-8, +8] — gets clamped!

// After: read post-sin LUT output
case (cur_neuron[1:0])
    2'd0: out_r <= sin_output;  // Range [-1, +1] — maps correctly
```

### Lesson
When implementing a neural network in hardware, the activation function choice must match between training and inference **exactly**. There are no "close enough" activations — tanh and sin have different ranges, different gradients, and the weights are trained specifically for one. A mismatch doesn't produce "slightly wrong" results; it produces completely wrong results because the weight values encode assumptions about the activation function's behavior.

## 11. Parallel Multiplier LUT Overflow

### Symptom
Synthesis failed with the FPGA exceeding 100% LUT utilization (~56,400 LUTs used vs 53,200 available).

### Root Cause
The first MLP implementation used 3 parallel multipliers per core (matching the Mandelbrot design). Each core had a 387-entry weight array with 3 simultaneous read ports. Vivado generated multiplexer trees for the parallel weight reads: 3 ports × 387 entries × 32-bit values × 18 cores = enormous mux fan-out.

### Fix
Switched to a sequential MAC (Multiply-ACcumulate) architecture: 1 multiplier per core, 1 weight read per cycle, weights stored in BRAM (1 read port). This traded throughput (~2.5× slower per pixel) for dramatically lower LUT usage.

| Design | DSPs | LUTs | BRAM | FPS |
|--------|------|------|------|-----|
| 3-mul parallel | 216 | 56,400 (fails) | 50 | ~70 (theoretical) |
| 1-mul sequential | 72 | 16,600 | 82 | ~26 |

### Lesson
On FPGAs, memory access patterns matter more than compute parallelism. A large ROM with multiple read ports synthesizes as LUT-based multiplexers that can easily overflow the fabric. BRAM has fixed port counts (dual-port maximum) but costs zero LUTs.

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
| RGB565 bit extraction | Binary color bands | ~6 hours | Include the 1.0 position bit |
| Activation mismatch | Thresholded colors | ~3 hours | Train and infer with same activation |
| Parallel mux overflow | LUT utilization >100% | ~4 hours | Use BRAM, not LUT-ROM with N ports |
