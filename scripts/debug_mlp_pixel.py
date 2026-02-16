#!/usr/bin/env python3
"""Debug a single pixel through both fixed-point and float paths."""

import math
import sys
sys.path.insert(0, '.')
from scripts.test_mlp_render import (
    parse_weights_vh, to_signed32, q428_mul, q428_to_float,
    q428_from_float, sine_lut_fpga, MASK32, FRAC, SCALE
)

def debug_pixel(x_q, y_q, t_q, weights, n_hidden=16):
    """Trace single pixel through both paths with detailed output."""

    layers = [
        {'fan_in': 3,        'fan_out': n_hidden, 'w_base': 0,   'b_base': 48},
        {'fan_in': n_hidden, 'fan_out': n_hidden, 'w_base': 64,  'b_base': 320},
        {'fan_in': n_hidden, 'fan_out': 3,        'w_base': 336, 'b_base': 384},
    ]

    w_float = {k: q428_to_float(v) for k, v in weights.items()}

    act_fixed = [x_q, y_q, t_q] + [0] * (n_hidden - 3)
    act_float = [q428_to_float(x_q), q428_to_float(y_q), q428_to_float(t_q)] + [0.0] * (n_hidden - 3)

    print(f"Input: x={q428_to_float(x_q):.6f}, y={q428_to_float(y_q):.6f}, t={q428_to_float(t_q):.6f}")

    for layer_idx, layer in enumerate(layers):
        fan_in = layer['fan_in']
        fan_out = layer['fan_out']
        w_base = layer['w_base']
        b_base = layer['b_base']

        print(f"\n=== Layer {layer_idx}: {fan_in} → {fan_out} ===")

        out_fixed = [0] * max(n_hidden, 3)
        out_float = [0.0] * max(n_hidden, 3)

        for j in range(min(fan_out, 4)):  # Only show first 4 neurons for brevity
            # Fixed-point
            acc_f = 0
            acc_fl = 0.0

            for k in range(fan_in):
                w_addr = w_base + j * fan_in + k
                w_val = weights[w_addr]
                a_val = act_fixed[k]

                prod = q428_mul(w_val, a_val)
                acc_f += prod

                # Float
                acc_fl += w_float[w_addr] * act_float[k]

            bias_fixed = weights[b_base + j]
            bias_float = w_float[b_base + j]
            acc_f += bias_fixed
            acc_fl += bias_float

            # Saturate
            if acc_f > 0x7FFFFFFF:
                acc_sat = 0x7FFFFFFF
            elif acc_f < -0x80000000:
                acc_sat = -0x80000000
            else:
                acc_sat = to_signed32(int(acc_f) & MASK32)

            # Sin
            sin_fixed = sine_lut_fpga(acc_sat)
            sin_float = math.sin(acc_fl)

            print(f"  neuron[{j}]:")
            print(f"    pre-sin fixed = {q428_to_float(acc_sat):+.6f} (0x{acc_sat & MASK32:08X})")
            print(f"    pre-sin float = {acc_fl:+.6f}")
            print(f"    sin fixed     = {q428_to_float(sin_fixed):+.6f} (0x{sin_fixed & MASK32:08X})")
            print(f"    sin float     = {sin_float:+.6f}")
            print(f"    error         = {abs(q428_to_float(sin_fixed) - sin_float):.6f}")

            out_fixed[j] = sin_fixed
            out_float[j] = sin_float

        # Complete the rest (not printed)
        for j in range(min(fan_out, 4), fan_out):
            acc_f = 0
            acc_fl = 0.0
            for k in range(fan_in):
                w_addr = w_base + j * fan_in + k
                prod = q428_mul(weights[w_addr], act_fixed[k])
                acc_f += prod
                acc_fl += w_float[w_addr] * act_float[k]

            acc_f += weights[b_base + j]
            acc_fl += w_float[b_base + j]

            if acc_f > 0x7FFFFFFF:
                acc_sat = 0x7FFFFFFF
            elif acc_f < -0x80000000:
                acc_sat = -0x80000000
            else:
                acc_sat = to_signed32(int(acc_f) & MASK32)

            out_fixed[j] = sine_lut_fpga(acc_sat)
            out_float[j] = math.sin(acc_fl)

        act_fixed = out_fixed
        act_float = out_float

    print(f"\n=== Output ===")
    for ch, name in enumerate(['R', 'G', 'B']):
        print(f"  {name}: fixed={q428_to_float(act_fixed[ch]):+.6f}  float={act_float[ch]:+.6f}  "
              f"err={abs(q428_to_float(act_fixed[ch]) - act_float[ch]):.6f}")


# Also test the sine LUT in isolation
def test_sine_lut():
    print("=== Sine LUT tests ===")
    test_angles = [0.0, 0.5, 1.0, 1.5707963, 3.14159265, -1.0, -3.14159265, 5.0, 10.0, -10.0, 0.1]
    for angle in test_angles:
        q = q428_from_float(angle)
        s_fixed = sine_lut_fpga(q)
        s_float = math.sin(angle)
        err = abs(q428_to_float(s_fixed) - s_float)
        flag = " *** BAD" if err > 0.01 else ""
        print(f"  sin({angle:+8.4f}) : fixed={q428_to_float(s_fixed):+.6f}  "
              f"float={s_float:+.6f}  err={err:.6f}{flag}")


if __name__ == '__main__':
    test_sine_lut()

    weights = parse_weights_vh('rtl/mlp_weights.vh')

    # Test center pixel (0, 0) at t=0
    print("\n\n========== Pixel (160, 86) = center, t=0 ==========")
    x_q = q428_from_float(0.0)
    y_q = q428_from_float(0.0)
    t_q = q428_from_float(0.0)
    debug_pixel(x_q, y_q, t_q, weights)

    # Test corner pixel (-1, -1)
    print("\n\n========== Pixel (0, 0) = corner, t=0 ==========")
    x_q = q428_from_float(-1.0)
    y_q = q428_from_float(-1.0)
    t_q = q428_from_float(0.0)
    debug_pixel(x_q, y_q, t_q, weights)
