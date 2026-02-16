#!/usr/bin/env python3
"""test_mlp_render.py -- Render MLP output matching FPGA datapath exactly.

Loads quantized Q4.28 weights from mlp_weights.vh, replicates the
fixed-point arithmetic, sine LUT, and RGB565 packing from the Verilog
to produce a reference image for comparison with FPGA hardware output.

Also renders a float32 reference to isolate quantization effects.

Uses fpga_sim.py as the single source of truth for all FPGA datapath
functions. The float32 reference path is kept here for comparison.
"""

import math
import numpy as np
from pathlib import Path

from fpga_sim import (
    FRAC, SCALE, MASK32,
    to_signed32, q428_to_float,
    parse_weights_vh, render_frame_fpga,
    get_layer_geometry, CRE_START, CIM_START, CRE_STEP, CIM_STEP,
)


# =========================================================
# Float32 reference forward pass
# =========================================================
def mlp_forward_float(x, y, t, weights, n_hidden=32):
    """Run MLP forward pass with float32 for comparison."""
    layers = get_layer_geometry(n_hidden)
    w_float = {k: q428_to_float(v) for k, v in weights.items()}
    act_in = [x, y, t] + [0.0] * (n_hidden - 3)

    for layer_idx, layer in enumerate(layers):
        fan_in = layer['fan_in']
        fan_out = layer['fan_out']
        w_base = layer['w_base']
        b_base = layer['b_base']
        act_out = [0.0] * max(n_hidden, 3)

        for j in range(fan_out):
            acc = 0.0
            for k in range(fan_in):
                w_addr = w_base + j * fan_in + k
                acc += w_float[w_addr] * act_in[k]
            acc += w_float[b_base + j]
            act_out[j] = math.sin(acc)

        act_in = act_out

    return act_in[0], act_in[1], act_in[2]


def float_rgb_to_rgb888(r, g, b):
    """Convert [-1,+1] float RGB to 8-bit tuple."""
    def clamp8(v):
        v01 = (v + 1.0) / 2.0
        return max(0, min(255, int(round(v01 * 255))))
    return (clamp8(r), clamp8(g), clamp8(b))


# =========================================================
# Main: render images
# =========================================================
def main():
    project_root = Path(__file__).parent.parent
    weights_path = project_root / 'rtl' / 'mlp_weights.vh'
    output_dir = project_root / 'scripts'

    print(f"Loading weights from {weights_path}")
    weights = parse_weights_vh(weights_path)
    print(f"  Loaded {len(weights)} parameters")

    W, H = 320, 172
    frames_to_render = [0, 100, 200, 400]

    for frame in frames_to_render:
        print(f"\nRendering frame {frame}...")

        max_iter_10bit = frame & 0x3FF
        time_q = (max_iter_10bit << 19) - 0x10000000
        time_q = to_signed32(time_q & MASK32)
        time_f = q428_to_float(time_q)
        print(f"  time_val Q4.28 = 0x{time_q & MASK32:08X} = {time_f:.6f}")

        # Fixed-point via vectorized FPGA simulator
        print(f"  Rendering fixed-point (vectorized)...")
        img_fixed = render_frame_fpga(weights, frame, W, H)

        # Float reference (scalar, slow)
        print(f"  Rendering float reference...")
        img_float = np.zeros((H, W, 3), dtype=np.uint8)
        for py in range(H):
            if py % 40 == 0:
                print(f"    row {py}/{H}")
            y_q = to_signed32((CIM_START + py * CIM_STEP) & MASK32)
            y_f = q428_to_float(y_q)
            for px in range(W):
                x_q = to_signed32((CRE_START + px * CRE_STEP) & MASK32)
                x_f = q428_to_float(x_q)
                r_f, g_f, b_f = mlp_forward_float(x_f, y_f, time_f, weights)
                img_float[py, px] = float_rgb_to_rgb888(r_f, g_f, b_f)

        # Save
        try:
            from PIL import Image
            fixed_path = output_dir / f'mlp_fixed_frame{frame}.png'
            float_path = output_dir / f'mlp_float_frame{frame}.png'
            Image.fromarray(img_fixed).save(fixed_path)
            Image.fromarray(img_float).save(float_path)
            print(f"  Saved: {fixed_path}")
            print(f"  Saved: {float_path}")
        except ImportError:
            for name, img in [('fixed', img_fixed), ('float', img_float)]:
                ppm_path = output_dir / f'mlp_{name}_frame{frame}.ppm'
                with open(ppm_path, 'wb') as f:
                    f.write(f'P6\n{W} {H}\n255\n'.encode())
                    f.write(img.tobytes())
                print(f"  Saved: {ppm_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
