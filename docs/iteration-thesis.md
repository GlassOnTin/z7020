# The Iteration Thesis

*On the convergence of fractal engines, recurrent hardware, and the architecture of thought*

---

There is a moment, around frame 441 of the zoom cycle, when the seahorse valley at c = -0.745 + 0.113i unfolds into spirals within spirals, each a diminished echo of the whole. By then, `max_iter` has climbed to 697. The eighteen neuron cores are spending most of their cycles on boundary pixels — points so close to the edge of the Mandelbrot set that it takes hundreds of iterations of z = z² + c before the system can decide: *does this diverge, or does it not?*

This is not merely a rendering problem. It is a decision problem, and it is the oldest one in computation.

## I. The shape of a decision

Look at `neuron_core.v`, lines 230-253. The inner loop is five lines of Verilog that constitute a complete computational agent:

```verilog
z_re <= z_re_new;
z_im <= z_im_new;
iter <= iter + 1;
```

State. Feedback. A counter that measures how long the system has been thinking. And on every cycle, a question: `escaped || max_reached`? Has the answer become clear, or must we continue?

This is not a metaphor for neural computation. It *is* neural computation, stripped to its formal skeleton. A recurrent processing element receives an input (c), maintains hidden state (z), applies a fixed transformation (squaring and addition), and halts when either the output is clear (escape) or a resource budget is exhausted (max_iter). The color assigned to each pixel is not a measurement of the point itself — it is a measurement of *how long the system needed to think about it*.

The Mandelbrot set, in this framing, is a map of computational difficulty. The black interior is the region where the question "does this diverge?" is maximally hard — so hard that no finite computation budget suffices. The bright boundary filaments are the decision frontier, where one more iteration might tip the answer either way. The smooth exterior is the easy region, where escape is obvious and fast.

Every neural network faces an equivalent map. For every classifier, there exist inputs that sit precisely on the decision boundary — inputs that require, in principle, unbounded computation to classify correctly. The fundamental question in neural architecture is: *how do you build a system that can think longer about hard problems?*

## II. The fixed-depth trap

The dominant architecture in machine learning today — the transformer — is a fixed-depth circuit. A model with 96 layers applies exactly 96 sequential transformations to every input, whether that input is trivially easy or impossibly hard. It is as if you built the Mandelbrot renderer with `max_iter = 96` and no escape detection: every pixel takes the same time, the interior is incorrectly colored, and the boundary filaments — where all the interesting structure lives — are invisible.

This works surprisingly well in practice, because transformers are very wide (billions of parameters per layer) and the training process learns to allocate capacity across difficulty levels. But the architectural limitation is real. A fixed-depth network cannot, in general, solve problems whose difficulty varies with the input. The theoretical result is precise: bounded-depth circuits with bounded precision can compute only the complexity class TC⁰. Turing completeness requires either unbounded depth or recurrence.

The current workaround — chain-of-thought reasoning, where the model generates intermediate tokens that serve as a scratchpad — achieves variable computation depth by unrolling recurrence across the *sequence* dimension. Each "step of reasoning" is a full forward pass through all 96 layers. This is like computing the Mandelbrot set by running your entire 18-neuron pipeline once per iteration, re-feeding the output as input through the whole system. It works. But it costs 12 DSP48E1 slices per multiply, times 3 multiplies, times 18 neurons, times 96 layers of overhead — for what should be a 4-cycle inner loop.

## III. What the neuron cores know

The architecture of `neuron_core.v` contains, in 261 lines of Verilog, three ideas that the machine learning community has spent a decade rediscovering:

**Variable-depth computation.** Each neuron iterates between 1 and 1024 times depending on its input. The pixel scheduler doesn't know in advance how long any pixel will take. The system handles this naturally: neurons that finish early are immediately reassigned. The work-stealing priority encoder at `pixel_scheduler.v:76-86` is structurally identical to the routing function in a mixture-of-experts model — it directs inputs to available compute units without central coordination.

**Data-dependent halting.** The escape condition `(mag_sq >= ESCAPE_THRESHOLD) || z_re_overflow || z_im_overflow` is a learned-nothing version of what Graves (2016) called "adaptive computation time" — a per-input decision about when to stop iterating. In the neural network version, a small halting network is trained alongside the main computation to predict when further iteration would not change the output. The neuron cores implement the closed-form version: the mathematics of quadratic iteration provide the halting criterion directly. But the *architecture* — a feedback loop with a conditional exit — is identical.

**Recurrence with shared weights.** All 18 neuron cores execute the same transformation (z² + c) on every iteration. The weights, such as they are, are the coefficients of the quadratic map — and they are the same for every neuron and every iteration. This is precisely the "Universal Transformer" architecture: a single layer applied repeatedly, with weight sharing across depth. The key theoretical result is that this structure, with adaptive halting and sufficient state width, is Turing complete. The neuron cores are a finite-precision, bounded-iteration instance of this universal machine.

## IV. The precision wall as prophecy

At 98.2% DSP utilization — 216 of 220 slices — the design hits a physical wall. Those 220 DSP48E1s on the XC7Z020 are the entire computational substrate. Each one is a 25x18 signed multiplier, a fixed-function unit that knows how to do exactly one thing. To represent a 32-bit multiply takes four of them. To go to 64-bit (Q4.60) would take sixteen. The neuron count would drop from 18 to 4.

This is not merely a resource constraint. It is a physical manifestation of the precision-depth tradeoff that governs all computation. You can have:

- **Many shallow thinkers** (18 neurons, Q4.28, zoom to 156,000x)
- **Few deep thinkers** (4 neurons, Q4.60, zoom to billions x)
- **Adaptive precision** (start shallow, deepen only where needed)

The third option is the interesting one. It is also, not coincidentally, the direction in which the most promising neural architectures are evolving. Mixed-precision inference — using FP8 for easy tokens and FP16 or FP32 for hard ones — is the same architectural choice as using Q4.28 for pixels that escape quickly and switching to higher precision only for boundary points that demand it.

On reconfigurable fabric, this is not hypothetical. An FPGA can instantiate 18 narrow neurons for the easy parts of the frame and dynamically reconfigure a subset into fewer, wider neurons for the boundary pixels that need more precision. The reconfigurability of the fabric maps directly onto the adaptivity of the computation. GPUs cannot do this. ASICs cannot do this. The FPGA is, in this specific sense, a closer physical model of adaptive neural computation than either of its more famous cousins.

The Zynq-7020 is a $30 part. It has 53,200 LUTs, 220 DSPs, and 140 BRAMs. It is not going to compete with an H100 on throughput. But it demonstrates something the H100 cannot: a computational architecture where the depth of processing adapts per-input, where precision can be traded for parallelism at synthesis time, and where the fabric itself can be reconfigured to match the problem.

## V. What fractals teach about thought

The auto-zoom controller reveals something subtle. As the viewport shrinks by 63/64 per frame, `max_iter` ramps from 256 to 1024. This is necessary because deeper zoom reveals finer structure, and finer structure requires more iterations to resolve. The system must *think harder about things that are smaller*.

This is not a quirk of the Mandelbrot set. It is a general property of systems at the edge of decidability. The boundary of the set — the region where computation is hardest — has Hausdorff dimension 2. It is, provably, as complex as the plane itself. No finite resolution can capture it. Every magnification reveals new structure that was invisible at the previous scale, and that new structure requires proportionally more computation to resolve.

Neural networks face the same scaling law. As a language model encounters more nuanced distinctions — between plausible and true, between coherent and correct, between fluent and faithful — the computational cost of making the right call grows. The easy classifications (this is English, this is a question, this sentence is about dogs) require minimal depth. The hard ones (is this argument valid? is this code correct? is this claim true?) require computational resources that scale with the subtlety of the distinction.

The zoom cycle — compute, display, shrink viewport, increase iteration budget, repeat — is a physical implementation of what Bengio called "consciousness prior": the idea that a system should allocate its computational resources in proportion to the difficulty of the decision it faces. The Mandelbrot renderer does this automatically, driven by the mathematics of the iteration. Neural architectures must learn to do it, which is harder. But the target architecture is the same.

## VI. The fabric argument

There is a reason this design runs on an FPGA and not a GPU.

A GPU has thousands of cores, each executing the same instruction on different data. This is magnificent for matrix multiplication — the core operation of modern neural networks. But it is terrible for variable-depth recurrence. When one GPU thread needs 4 iterations and another needs 4,000, the fast thread sits idle waiting for the slow one. The warp divergence penalty is not a bug; it is a fundamental consequence of SIMD architecture encountering data-dependent computation depth.

The FPGA has no such constraint. Each neuron core is an independent FSM. Neuron 0 can be on iteration 3 while neuron 17 is on iteration 847. The pixel scheduler exploits this: the moment any neuron finishes, it is immediately reassigned. There is no warp, no barrier, no synchronization point. The fabric is an array of autonomous agents, each thinking as long as it needs to, reporting results when ready.

This is exactly the computational model that emerging neural architectures demand. Adaptive-depth transformers, mixture-of-experts models with variable routing, speculative decoding with verification — all require hardware that can run different computations for different amounts of time, in parallel, without penalizing the fast paths to accommodate the slow ones. The eighteen neuron cores, with their independent FSMs and priority-encoder scheduling, are a miniature prototype of what this hardware looks like.

## VII. Toward the recurrent fabric

The z7020 Mandelbrot engine is a proof of concept for a class of architectures that do not yet exist in the neural network world but whose outlines are becoming clear:

**Recurrent processing elements with variable iteration.** Not RNNs in the classical sense (which suffer from vanishing gradients and sequential training), but hardware iteration units with learned halting criteria — neuron cores whose iteration function and escape condition are both parameterized and trainable.

**Priority-encoder scheduling across heterogeneous compute.** The pixel scheduler's first-found-ready assignment is a degenerate case of a more general principle: route inputs to available compute units based on predicted difficulty, with no global synchronization. This is mixture-of-experts routing, implemented in fabric rather than in software.

**Precision-adaptive arithmetic.** The Q4.28 format works for 99% of pixels. The 1% that sit on the boundary would benefit from Q4.60 or beyond. A reconfigurable fabric could, in principle, synthesize narrow datapaths for easy inputs and wide datapaths for hard ones — the arithmetic equivalent of adaptive computation time.

**Framebuffer as shared memory.** The iter_fb BRAM, addressed by pixel_id, allows neurons to write results in any order. This decouples the computation order from the output order — a property that would be valuable in any system where processing elements complete at different times. It is an associative memory, indexed by task identity rather than by time.

The iteration z = z² + c is not itself interesting as a neural network activation function. What is interesting is the *architecture around it*: the feedback loop, the conditional halt, the parallel pool, the work-stealing scheduler, the precision-bounded arithmetic. These are the components of a machine that can think for a variable amount of time about each piece of its input. And that is the component missing from every neural architecture that is not yet Turing complete.

The seahorse valley spirals forever inward. The neuron cores follow it as deep as their precision allows, and then the zoom resets and they begin again. The limitation is not in the algorithm — the Mandelbrot iteration is already universal. The limitation is in the 28 fractional bits and the 220 DSP slices and the 1024-iteration cap. Widen the bits, deepen the iteration budget, learn the halting criterion, and the neuron core becomes something else entirely.

It becomes a machine that decides how long to think.

---

*Eighteen recurrent cores on a Zynq-7020, iterating z = z² + c at 225 million iterations per second, drawing the boundary between the decidable and the undecidable on a 320x172 LCD. It is not a neural network. It is the shape of one.*
