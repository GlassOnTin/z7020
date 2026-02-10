#!/usr/bin/env python3
"""
mandelbrot_ref.py — Python reference implementation for Mandelbrot neuron core.

Generates golden test vectors in Q4.28 fixed-point format and computes
expected iteration counts for verification of the Verilog implementation.
"""

import struct

FRAC = 28
WIDTH = 32
SCALE = 1 << FRAC  # 2^28 = 268435456


def float_to_q428(f):
    """Convert float to Q4.28 signed fixed-point (32-bit)."""
    val = int(round(f * SCALE))
    # Clamp to 32-bit signed range
    if val >= (1 << (WIDTH - 1)):
        val = (1 << (WIDTH - 1)) - 1
    if val < -(1 << (WIDTH - 1)):
        val = -(1 << (WIDTH - 1))
    # Return as unsigned 32-bit (two's complement)
    return val & 0xFFFFFFFF


def q428_to_float(q):
    """Convert Q4.28 unsigned representation back to float."""
    if q >= (1 << (WIDTH - 1)):
        q -= (1 << WIDTH)
    return q / SCALE


def mandelbrot_iter(c_re, c_im, max_iter=256):
    """Compute Mandelbrot iteration count (float reference)."""
    z_re, z_im = 0.0, 0.0
    for i in range(max_iter):
        z_re_sq = z_re * z_re
        z_im_sq = z_im * z_im
        if z_re_sq + z_im_sq > 4.0:
            return i
        z_im = 2.0 * z_re * z_im + c_im
        z_re = z_re_sq - z_im_sq + c_re
    return max_iter


def mandelbrot_iter_fixed(c_re_q, c_im_q, max_iter=256):
    """
    Compute Mandelbrot iteration count using Q4.28 fixed-point arithmetic.
    Mimics the Verilog implementation exactly.
    """
    # Convert to signed Python ints
    if c_re_q >= (1 << (WIDTH - 1)):
        c_re_q -= (1 << WIDTH)
    if c_im_q >= (1 << (WIDTH - 1)):
        c_im_q -= (1 << WIDTH)

    z_re, z_im = 0, 0
    escape_threshold = 4 << FRAC  # 4.0 in Q4.28

    for i in range(max_iter):
        # Full precision multiply then truncate (matching Verilog)
        z_re_sq = (z_re * z_re) >> FRAC
        z_im_sq = (z_im * z_im) >> FRAC
        z_re_im = (z_re * z_im) >> FRAC

        mag_sq = z_re_sq + z_im_sq

        # Escape: magnitude squared > 4.0 or overflow (negative mag_sq)
        if mag_sq < 0 or mag_sq > escape_threshold:
            return i

        # Update z
        z_re_new = z_re_sq - z_im_sq + c_re_q
        z_im_new = (z_re_im << 1) + c_im_q  # 2 * z_re * z_im + c_im

        # Truncate to 32-bit signed (simulate hardware overflow)
        z_re = z_re_new & 0xFFFFFFFF
        z_im = z_im_new & 0xFFFFFFFF
        if z_re >= (1 << (WIDTH - 1)):
            z_re -= (1 << WIDTH)
        if z_im >= (1 << (WIDTH - 1)):
            z_im -= (1 << WIDTH)

    return max_iter


def main():
    test_cases = [
        (0.0, 0.0, "origin (interior)"),
        (2.0, 0.0, "c=2 (fast escape)"),
        (-1.0, 0.0, "c=-1 (period-2 interior)"),
        (0.5, 0.0, "c=0.5 (exterior)"),
        (-0.75, 0.1, "near boundary"),
        (-2.1, 0.0, "just outside"),
        (-0.1, 0.75, "near top of main cardioid"),
        (0.3, 0.5, "exterior spiral region"),
        (-1.25, 0.0, "period-2 bulb boundary"),
        (-0.5, 0.5, "interior near boundary"),
    ]

    max_iter = 256
    print(f"{'Test':<35} {'c_re_hex':<12} {'c_im_hex':<12} {'Float iter':<12} {'Fixed iter':<12}")
    print("=" * 83)

    for c_re, c_im, name in test_cases:
        c_re_q = float_to_q428(c_re)
        c_im_q = float_to_q428(c_im)
        iter_float = mandelbrot_iter(c_re, c_im, max_iter)
        iter_fixed = mandelbrot_iter_fixed(c_re_q, c_im_q, max_iter)

        print(f"{name:<35} {c_re_q:08X}     {c_im_q:08X}     {iter_float:<12} {iter_fixed:<12}")

    # Generate full frame viewport parameters
    print("\n\nDefault viewport parameters (Q4.28 hex):")
    print(f"  c_re_start = -2.0      = {float_to_q428(-2.0):08X}")
    print(f"  c_im_start = -0.80625  = {float_to_q428(-0.80625):08X}")
    print(f"  c_re_step  = 0.009375  = {float_to_q428(0.009375):08X}")
    print(f"  c_im_step  = 0.009375  = {float_to_q428(0.009375):08X}")
    print(f"  c_re_end   = 1.0       = {float_to_q428(1.0):08X}")
    print(f"  c_im_end   = 0.80625   = {float_to_q428(0.80625):08X}")

    # Verify the default viewport coordinates
    print("\nVerification of viewport corners:")
    h_res, v_res = 320, 172
    step = 3.0 / h_res  # 0.009375
    for label, px, py in [("top-left", 0, 0), ("top-right", h_res-1, 0),
                           ("bottom-left", 0, v_res-1), ("center", h_res//2, v_res//2)]:
        c_re = -2.0 + px * step
        c_im = -0.80625 + py * step
        iters = mandelbrot_iter(c_re, c_im, max_iter)
        print(f"  {label:>12} ({px:3d},{py:3d}): c=({c_re:+.6f}, {c_im:+.6f}) iter={iters}")


def row_start_vectors():
    """Generate row-start golden vectors for scheduler testbench verification.

    Verifies that every row starts at c_re_start (no off-by-one drift).
    Uses the same viewport parameters as tb_pixel_scheduler.v.
    """
    h_res, v_res = 16, 8
    c_re_start = -2.0
    c_im_start = -1.0
    step = 0.25

    print("\n\nRow-start golden vectors (16x8 testbench viewport):")
    print(f"  c_re_start = {c_re_start}  = {float_to_q428(c_re_start):08X}")
    print(f"  c_im_start = {c_im_start}  = {float_to_q428(c_im_start):08X}")
    print(f"  step       = {step}   = {float_to_q428(step):08X}")
    print()
    print(f"  {'Row':<6} {'c_re_start (hex)':<18} {'c_im (hex)':<18} {'c_re (float)':<14} {'c_im (float)'}")
    print("  " + "=" * 70)

    for row in range(v_res):
        c_re = c_re_start  # Every row must start here (Bug #1 fix)
        c_im = c_im_start + row * step
        c_re_q = float_to_q428(c_re)
        c_im_q = float_to_q428(c_im)
        print(f"  {row:<6} {c_re_q:08X}             {c_im_q:08X}             {c_re:<+14.6f} {c_im:<+.6f}")


if __name__ == "__main__":
    main()
    row_start_vectors()
