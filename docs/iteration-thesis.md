# The Iteration Thesis

*On what a Mandelbrot renderer actually shares with neural inference, and what it doesn't*

---

## I. What this is not

A Mandelbrot iterator is not a neuron. Let's be clear about that.

The inner loop of `neuron_core.v` computes z = z² + c. It has no learned weights. It has no interconnection with neighboring cores. Each pixel is computed in isolation — the result at coordinate (x₁, y₁) has zero influence on the result at (x₂, y₂). This is embarrassingly parallel in the purest sense: there is no information sharing, no lateral inhibition, no attention mechanism, no learned representation. It is a fixed recurrence applied independently to each point in the complex plane.

Calling these "neuron cores" was a stretch. The name came from the structural resemblance — a processing element with state, feedback, and conditional halting — but structural resemblance is not functional equivalence. A for-loop is not a neuron just because it iterates.

So what *is* shared?

## II. The architecture, not the computation

The interesting thing about this design was never the z² + c iteration itself. It was the scaffolding around it:

- **18 independent processing cores** with private state and a common interface (pixel_valid/pixel_ready handshake, result_valid output)
- **A work-stealing scheduler** that dispatches coordinates to idle cores and collects results out of order via pixel_id tagging
- **Double-buffered BRAM framebuffers** that decouple computation order from display order
- **A fixed-point arithmetic pipeline** shared as a building block across core types

Each core is an independent FSM. Core 0 can be on its 3rd iteration while core 17 is on its 847th. When any core finishes, the scheduler immediately assigns it a new pixel. There is no synchronization barrier, no warp divergence, no idle waiting.

This is a parallel inference engine. The Mandelbrot recurrence is one "model" it can run. The question was: could the same architecture run an *actual* model — one with learned weights and nonlinear activations?

## III. The test: drop in a real neural network

To test whether the architecture generalizes, we replaced `neuron_core.v` with `mlp_core.v` — a SIREN (Sinusoidal Representation Networks) implicit neural representation. Same interface, same scheduler, same framebuffer, same display pipeline. The `COMPUTE_MODE` parameter selects which core gets instantiated at synthesis time.

The MLP core runs a 3→16→16→3 network:
- **Input:** (x, y, t) coordinates in Q4.28 fixed point
- **Hidden layers:** 16 neurons each, sin() activation via quarter-wave LUT
- **Output:** RGB565 color, written directly to the display framebuffer
- **Weights:** 387 trained parameters stored in BRAM, loaded at synthesis

The key constraint was resource budget. The original design used 3 multipliers per core (216 of 220 DSPs). The first MLP implementation tried the same — 3 parallel multipliers with variable-indexed weight array access — and overflowed the LUT budget by 3,200 slices. The 18 copies of 387-entry weight arrays, each with 3 simultaneous read ports, created mux trees that Vivado couldn't fit.

The fix was architectural: switch to a sequential MAC with BRAM weight storage. One multiplier per core, one weight read per cycle, ~616 cycles per pixel. This traded throughput for density and hit 30% LUT / 65% DSP / 36% BRAM — well within budget.

The result: 18 parallel MLP cores running trained SIREN inference, dispatched by the same work-stealing scheduler, writing results to the same double-buffered framebuffer, displayed by the same SPI driver. The architecture generalized without changing the scheduler, framebuffer, or display pipeline.

## IV. What the Mandelbrot and the MLP actually share

With both modes implemented, the shared structure becomes concrete rather than metaphorical:

| Component | Mode 0 (Mandelbrot) | Mode 1 (SIREN MLP) |
|-----------|--------------------|--------------------|
| Core computation | z² + c (fixed recurrence) | Σ(w·x) + b, sin() (learned weights) |
| Multiplier | 3 per core, parallel | 1 per core, sequential MAC |
| State | z_re, z_im, iter | Activation registers, accumulator |
| Halting | Escape detection or max_iter | Fixed depth (3 layers) |
| Weight storage | None (coefficients are hardcoded) | 387 params in BRAM per core |
| Inter-core communication | None | None |
| Output | Iteration count → colormap → RGB565 | Network output → RGB565 direct |

The scheduler doesn't care what the cores compute. It speaks a protocol: here's a coordinate and a pixel_id, tell me when you're done and give me back the pixel_id with a result. The Mandelbrot core returns an iteration count. The MLP core returns a color. The scheduler handles both identically.

## V. What the Mandelbrot *doesn't* share

Several things that real neural networks do, and the Mandelbrot iterator doesn't:

**Interconnectedness.** In a neural network, neuron outputs feed into other neurons. The representation is distributed — the meaning of one neuron's activation depends on all the others. In the Mandelbrot renderer, pixels are independent. This is exactly why they parallelize trivially, and exactly why the computation is not "neural" in any meaningful sense.

**Learned parameters.** The Mandelbrot recurrence has no weights to train. The function z² + c is given by the mathematics, not learned from data. The MLP core's 387 weights were trained by gradient descent on a target pattern. This is the difference between a fixed function and a parameterized one.

**Variable computation depth (in the MLP).** Ironically, the Mandelbrot mode has data-dependent halting (escape detection) while the MLP mode has fixed depth (3 layers, same work per pixel). The architectural support for variable-depth computation exists in the scheduler but the current MLP core doesn't use it. A future version with adaptive halting would be more interesting.

## VI. The architectural argument, honestly stated

The original thesis overclaimed. Calling a Mandelbrot iterator "neural computation stripped to its formal skeleton" was wrong. It's a fixed quadratic recurrence. The formal skeleton of neural computation requires at minimum: learned parameters, nonlinear activation, and composition of layers. The Mandelbrot iterator has the nonlinearity (z²) but not the other two.

What *is* true: the parallel scheduling architecture — independent cores, work-stealing dispatch, out-of-order result collection, decoupled framebuffer — is the same infrastructure you need to run actual neural inference on reconfigurable fabric. And when we tested that claim by dropping in a real neural network, it worked. Same scheduler, same framebuffer, same display pipeline, different core.

The interesting design question going forward is not "is Mandelbrot neural?" (it isn't) but "what else can you run on 18 parallel cores with a work-stealing scheduler on a $30 FPGA?" The answer, empirically: at least small trained neural networks at real-time frame rates.

---

*Eighteen cores on a Zynq-7020, sharing a scheduler and a framebuffer. In one mode they iterate z = z² + c. In the other they run a trained SIREN network. The cores changed. The architecture didn't.*
