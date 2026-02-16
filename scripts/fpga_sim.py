#!/usr/bin/env python3
"""fpga_sim.py -- Bit-exact Python simulator of the FPGA SIREN datapath.

Single source of truth for:
  - Q4.28 fixed-point arithmetic (matching fixed_mul.v)
  - Sine LUT (matching sine_lut.v quarter-wave table)
  - MLP forward pass (matching mlp_core.v FSM)
  - RGB565 packing (matching pack_rgb565 function)
  - Coordinate generation (matching mandelbrot_top.v MLP mode)

Provides both scalar functions (for debugging single pixels) and
vectorized numpy functions (for batch rendering / training validation).

Usage:
    # Self-test: verify vectorized matches scalar bit-exact
    python3 scripts/fpga_sim.py

    # As a module
    from fpga_sim import render_frame_fpga, prepare_fpga_weights
"""

import re
import math
import numpy as np
from pathlib import Path

# =========================================================
# Q4.28 fixed-point constants
# =========================================================
FRAC = 28
SCALE = 1 << FRAC
MASK32 = 0xFFFFFFFF

# Coordinate constants (matching mandelbrot_top.v MLP mode)
# Aspect-corrected: x in [-1, +1], y in [-172/320, +172/320]
# Both axes use the same step size (square pixels in coordinate space)
CRE_START = -0x10000000   # 0xF0000000 signed = -1.0
CIM_START = -144284058    # 0xF7666666 signed = -172/320 = -0.5375
CRE_STEP  = 0x0019999A    # 2.0/320
CIM_STEP  = 0x0019999A    # 2.0/320 (same step — square pixels)

# Display resolution
FRAME_W = 320
FRAME_H = 172

# =========================================================
# Quarter-wave sine table (matching sine_lut.v exactly)
# =========================================================
SINE_TABLE = [
    0x00000000, 0x001921F1, 0x003243A4, 0x004B64DB,
    0x00648558, 0x007DA4DD, 0x0096C32C, 0x00AFE007,
    0x00C8FB30, 0x00E21469, 0x00FB2B74, 0x01144013,
    0x012D5209, 0x01466118, 0x015F6D01, 0x01787587,
    0x01917A6C, 0x01AA7B72, 0x01C3785C, 0x01DC70ED,
    0x01F564E5, 0x020E5409, 0x02273E1A, 0x024022DB,
    0x0259020E, 0x0271DB76, 0x028AAED6, 0x02A37BF1,
    0x02BC4289, 0x02D50261, 0x02EDBB3C, 0x03066CDD,
    0x031F1708, 0x0337B97E, 0x03505405, 0x0368E65E,
    0x0381704D, 0x0399F196, 0x03B269FD, 0x03CAD944,
    0x03E33F2F, 0x03FB9B83, 0x0413EE04, 0x042C3674,
    0x04447499, 0x045CA836, 0x0474D110, 0x048CEEEB,
    0x04A5018C, 0x04BD08B7, 0x04D50431, 0x04ECF3BF,
    0x0504D725, 0x051CAE29, 0x05347891, 0x054C3620,
    0x0563E69D, 0x057B89CE, 0x05931F77, 0x05AAA75F,
    0x05C2214C, 0x05D98D04, 0x05F0EA4C, 0x060838EC,
    0x061F78AA, 0x0636A94C, 0x064DCA99, 0x0664DC58,
    0x067BDE51, 0x0692D04A, 0x06A9B20B, 0x06C0835B,
    0x06D74402, 0x06EDF3C9, 0x07049276, 0x071B1FD2,
    0x07319BA6, 0x074805BA, 0x075E5DD7, 0x0774A3C5,
    0x078AD74E, 0x07A0F83B, 0x07B70655, 0x07CD0166,
    0x07E2E937, 0x07F8BD93, 0x080E7E44, 0x08242B13,
    0x0839C3CD, 0x084F483A, 0x0864B827, 0x087A135E,
    0x088F59AA, 0x08A48AD8, 0x08B9A6B2, 0x08CEAD05,
    0x08E39D9D, 0x08F87846, 0x090D3CCD, 0x0921EAFE,
    0x093682A6, 0x094B0394, 0x095F6D93, 0x0973C072,
    0x0987FBFE, 0x099C2007, 0x09B02C59, 0x09C420C3,
    0x09D7FD15, 0x09EBC11C, 0x09FF6CAA, 0x0A12FF8C,
    0x0A267993, 0x0A39DA8E, 0x0A4D224E, 0x0A6050A3,
    0x0A73655E, 0x0A866050, 0x0A994149, 0x0AAC081C,
    0x0ABEB49A, 0x0AD14695, 0x0AE3BDDF, 0x0AF61A4B,
    0x0B085BAB, 0x0B1A81D2, 0x0B2C8C93, 0x0B3E7BC2,
    0x0B504F33, 0x0B6206BA, 0x0B73A22A, 0x0B85215A,
    0x0B96841C, 0x0BA7CA47, 0x0BB8F3B0, 0x0BCA002C,
    0x0BDAEF91, 0x0BEBC1B6, 0x0BFC7672, 0x0C0D0D9A,
    0x0C1D8706, 0x0C2DE28D, 0x0C3E2008, 0x0C4E3F4D,
    0x0C5E4036, 0x0C6E229A, 0x0C7DE652, 0x0C8D8B38,
    0x0C9D1125, 0x0CAC77F2, 0x0CBBBF7A, 0x0CCAE797,
    0x0CD9F024, 0x0CE8D8FB, 0x0CF7A1F8, 0x0D064AF5,
    0x0D14D3D0, 0x0D233C64, 0x0D31848E, 0x0D3FAC29,
    0x0D4DB315, 0x0D5B992D, 0x0D695E4F, 0x0D77025A,
    0x0D84852C, 0x0D91E6A4, 0x0D9F269F, 0x0DAC44FF,
    0x0DB941A3, 0x0DC61C69, 0x0DD2D534, 0x0DDF6BE2,
    0x0DEBE056, 0x0DF83271, 0x0E046213, 0x0E106F20,
    0x0E1C5979, 0x0E282101, 0x0E33C59A, 0x0E3F4729,
    0x0E4AA591, 0x0E55E0B5, 0x0E60F87A, 0x0E6BECC5,
    0x0E76BD7A, 0x0E816A7F, 0x0E8BF3BA, 0x0E965910,
    0x0EA09A69, 0x0EAAB7A9, 0x0EB4B0BA, 0x0EBE8581,
    0x0EC835E8, 0x0ED1C1D5, 0x0EDB2931, 0x0EE46BE6,
    0x0EED89DB, 0x0EF682FC, 0x0EFF5731, 0x0F080665,
    0x0F109082, 0x0F18F574, 0x0F213526, 0x0F294F82,
    0x0F314476, 0x0F3913EE, 0x0F40BDD6, 0x0F48421B,
    0x0F4FA0AB, 0x0F56D974, 0x0F5DEC64, 0x0F64D96A,
    0x0F6BA074, 0x0F724171, 0x0F78BC52, 0x0F7F1106,
    0x0F853F7E, 0x0F8B47AA, 0x0F91297C, 0x0F96E4E5,
    0x0F9C79D6, 0x0FA1E843, 0x0FA7301E, 0x0FAC5159,
    0x0FB14BE8, 0x0FB61FBF, 0x0FBACCD2, 0x0FBF5315,
    0x0FC3B27D, 0x0FC7EB00, 0x0FCBFC92, 0x0FCFE72B,
    0x0FD3AAC0, 0x0FD74747, 0x0FDABCB9, 0x0FDE0B0C,
    0x0FE13238, 0x0FE43236, 0x0FE70AFF, 0x0FE9BC8A,
    0x0FEC46D2, 0x0FEEA9D0, 0x0FF0E57E, 0x0FF2F9D8,
    0x0FF4E6D7, 0x0FF6AC76, 0x0FF84AB3, 0x0FF9C188,
    0x0FFB10F2, 0x0FFC38ED, 0x0FFD3978, 0x0FFE128F,
    0x0FFEC430, 0x0FFF4E5A, 0x0FFFB10B, 0x0FFFEC43,
]

SINE_TABLE_NP = np.array(SINE_TABLE, dtype=np.int64)

RECIP_TWO_PI = 0x028BE60D  # 1/(2*pi) in Q4.28


# =============================================================================
# Scalar functions (for debugging single pixels)
# =============================================================================

def to_signed32(val):
    """Interpret as signed 32-bit."""
    val = val & MASK32
    if val & 0x80000000:
        return val - 0x100000000
    return val


def q428_mul(a, b):
    """Q4.28 multiply matching fixed_mul.v: bits [59:28] of 64-bit signed product."""
    prod = a * b
    result = prod >> FRAC
    return to_signed32(result & MASK32)


def q428_from_float(f):
    """Convert float to Q4.28 signed 32-bit."""
    clamped = max(-8.0, min(f, 8.0 - 1.0 / SCALE))
    raw = int(round(clamped * SCALE))
    return to_signed32(raw & MASK32)


def q428_to_float(val):
    """Convert Q4.28 signed 32-bit to float."""
    return val / SCALE


def sine_lut_fpga(angle_q428):
    """Replicate sine_lut.v behavior exactly."""
    angle = to_signed32(angle_q428)
    recip = to_signed32(RECIP_TWO_PI)
    product = angle * recip

    if product < 0:
        product_bits = product & 0xFFFFFFFFFFFFFFFF
    else:
        product_bits = product

    phase_raw = (product_bits >> 46) & 0x3FF

    mirror = (phase_raw >> 8) & 0x1
    if mirror:
        table_idx = (~phase_raw) & 0xFF
    else:
        table_idx = phase_raw & 0xFF

    negate = (phase_raw >> 9) & 0x1
    table_val = SINE_TABLE[table_idx]

    if negate:
        result = (-table_val) & MASK32
    else:
        result = table_val & MASK32

    return to_signed32(result)


def get_layer_geometry(n_hidden):
    """Compute layer base addresses matching mlp_core.v functions."""
    H = n_hidden
    return [
        {'fan_in': 3, 'fan_out': H,
         'w_base': 0, 'b_base': 3 * H},
        {'fan_in': H, 'fan_out': H,
         'w_base': 3 * H + H, 'b_base': 3 * H + H + H * H},
        {'fan_in': H, 'fan_out': 3,
         'w_base': 3 * H + H + H * H + H, 'b_base': 3 * H + H + H * H + H + H * 3},
    ]


def mlp_forward_fixed(x_q, y_q, t_q, weights, n_hidden=32):
    """Run MLP forward pass with Q4.28 fixed-point matching mlp_core.v."""
    layers = get_layer_geometry(n_hidden)
    act_in = [x_q, y_q, t_q] + [0] * (n_hidden - 3)

    for layer_idx, layer in enumerate(layers):
        fan_in = layer['fan_in']
        fan_out = layer['fan_out']
        w_base = layer['w_base']
        b_base = layer['b_base']

        act_out = [0] * max(n_hidden, 3)

        for j in range(fan_out):
            acc = 0
            for k in range(fan_in):
                w_addr = w_base + j * fan_in + k
                w_val = weights[w_addr]
                a_val = act_in[k]
                prod = q428_mul(w_val, a_val)
                acc += prod

            bias = weights[b_base + j]
            acc += bias

            # Saturate to 32-bit signed
            if acc > 0x7FFFFFFF:
                acc_sat = 0x7FFFFFFF
            elif acc < -0x80000000:
                acc_sat = -0x80000000
            else:
                acc_sat = to_signed32(int(acc) & MASK32)

            act_out[j] = sine_lut_fpga(acc_sat)

        act_in = act_out

    return act_in[0], act_in[1], act_in[2]


def pack_rgb565(r_q, g_q, b_q):
    """Pack Q4.28 [-1,+1] RGB to RGB565 matching Verilog."""
    ONE = 0x10000000

    def extract_5bit(val):
        s = to_signed32((val + ONE) & MASK32)
        if s < 0:
            return 0
        if s >= 0x20000000:
            return 31
        return (s >> 24) & 0x1F

    def extract_6bit(val):
        s = to_signed32((val + ONE) & MASK32)
        if s < 0:
            return 0
        if s >= 0x20000000:
            return 63
        return (s >> 23) & 0x3F

    r5 = extract_5bit(r_q)
    g6 = extract_6bit(g_q)
    b5 = extract_5bit(b_q)
    return (r5 << 11) | (g6 << 5) | b5


def rgb565_to_rgb888(rgb565):
    """Convert RGB565 to (R, G, B) 8-bit tuple."""
    r5 = (rgb565 >> 11) & 0x1F
    g6 = (rgb565 >> 5) & 0x3F
    b5 = rgb565 & 0x1F
    return (r5 << 3 | r5 >> 2, g6 << 2 | g6 >> 4, b5 << 3 | b5 >> 2)


def parse_weights_vh(path):
    """Parse Q4.28 hex values from mlp_weights.vh."""
    weights = {}
    with open(path) as f:
        for line in f:
            m = re.match(r"\s*weight_mem\[\s*(\d+)\]\s*=\s*32'sh([0-9A-Fa-f]+);", line)
            if m:
                idx = int(m.group(1))
                hexval = int(m.group(2), 16)
                weights[idx] = to_signed32(hexval)
    return weights


# =============================================================================
# Vectorized functions (numpy int64, process all pixels at once)
# =============================================================================

def to_signed32_vec(arr):
    """Signed 32-bit interpretation on int64 arrays."""
    arr = arr & MASK32
    return np.where(arr & 0x80000000, arr - 0x100000000, arr).astype(np.int64)


def q428_mul_vec(a, b):
    """Q4.28 multiply matching fixed_mul.v: (a*b) >> 28, bits [59:28]."""
    prod = a * b  # int64, large enough for 32*32
    result = prod >> FRAC
    return to_signed32_vec(result)


def q428_from_float_vec(f):
    """Batch float -> Q4.28."""
    f = np.asarray(f, dtype=np.float64)
    clamped = np.clip(f, -8.0, 8.0 - 1.0 / SCALE)
    raw = np.round(clamped * SCALE).astype(np.int64)
    return to_signed32_vec(raw)


def sine_lut_vec(angle):
    """Vectorized sine LUT matching sine_lut.v exactly.

    angle: int64 array of Q4.28 signed values.
    Returns: int64 array of Q4.28 sine values.
    """
    angle = np.asarray(angle, dtype=np.int64)
    recip = np.int64(RECIP_TWO_PI)
    product = angle * recip  # int64

    # Convert to unsigned 64-bit view for bit extraction
    product_bits = product.astype(np.uint64)

    # Extract bits [55:46] = 10-bit phase
    phase_raw = ((product_bits >> 46) & 0x3FF).astype(np.int64)

    # Mirror: bit 8
    mirror = (phase_raw >> 8) & 0x1

    # Table index: bits [7:0], mirrored if bit 8 set
    idx_normal = phase_raw & 0xFF
    idx_mirror = (~phase_raw) & 0xFF
    table_idx = np.where(mirror, idx_mirror, idx_normal).astype(np.intp)

    # Negate: bit 9
    negate = (phase_raw >> 9) & 0x1

    # Lookup
    table_val = SINE_TABLE_NP[table_idx]

    # Apply sign
    result = np.where(negate, (-table_val) & MASK32, table_val & MASK32)
    return to_signed32_vec(result)


def mlp_forward_fixed_vec(x_q, y_q, t_q, weights, n_hidden=32):
    """Vectorized MLP forward pass matching mlp_core.v.

    x_q, y_q, t_q: int64 arrays of shape (N,) -- Q4.28 coordinates
    weights: dict mapping int address -> int value (Q4.28 signed)
    n_hidden: hidden layer size

    Returns: (r_q, g_q, b_q) int64 arrays of shape (N,)
    """
    N = len(x_q)
    layers = get_layer_geometry(n_hidden)

    # Build activation array: (N, max(n_hidden, 3))
    act_width = max(n_hidden, 3)
    act_in = np.zeros((N, act_width), dtype=np.int64)
    act_in[:, 0] = x_q
    act_in[:, 1] = y_q
    act_in[:, 2] = t_q

    for layer_idx, layer in enumerate(layers):
        fan_in = layer['fan_in']
        fan_out = layer['fan_out']
        w_base = layer['w_base']
        b_base = layer['b_base']

        act_out = np.zeros((N, act_width), dtype=np.int64)

        for j in range(fan_out):
            # Build weight vector for neuron j: shape (fan_in,)
            w_j = np.array([weights[w_base + j * fan_in + k]
                            for k in range(fan_in)], dtype=np.int64)

            # Vectorized MAC: w_j[newaxis,:] * act_in[:,:fan_in] -> (N, fan_in)
            # Then sum across fan_in dimension
            products = q428_mul_vec(
                w_j[np.newaxis, :],      # (1, fan_in) broadcast to (N, fan_in)
                act_in[:, :fan_in]       # (N, fan_in)
            )
            acc = products.sum(axis=1)   # (N,)

            # Add bias
            bias = np.int64(weights[b_base + j])
            acc = acc + bias

            # Saturate to signed 32-bit
            acc = np.clip(acc, -0x80000000, 0x7FFFFFFF)
            acc = to_signed32_vec(acc)

            # Sine activation
            act_out[:, j] = sine_lut_vec(acc)

        act_in = act_out

    return act_in[:, 0], act_in[:, 1], act_in[:, 2]


def pack_rgb565_vec(r_q, g_q, b_q):
    """Vectorized RGB565 packing matching Verilog."""
    ONE = np.int64(0x10000000)

    def extract_5bit(val):
        s = to_signed32_vec(val + ONE)
        result = (s >> 24) & 0x1F
        result = np.where(s < 0, 0, result)
        result = np.where(s >= 0x20000000, 31, result)
        return result

    def extract_6bit(val):
        s = to_signed32_vec(val + ONE)
        result = (s >> 23) & 0x3F
        result = np.where(s < 0, 0, result)
        result = np.where(s >= 0x20000000, 63, result)
        return result

    r5 = extract_5bit(r_q)
    g6 = extract_6bit(g_q)
    b5 = extract_5bit(b_q)
    return (r5 << 11) | (g6 << 5) | b5


def rgb565_to_rgb888_vec(rgb565):
    """Vectorized RGB565 -> RGB888."""
    r5 = (rgb565 >> 11) & 0x1F
    g6 = (rgb565 >> 5) & 0x3F
    b5 = rgb565 & 0x1F
    r8 = (r5 << 3) | (r5 >> 2)
    g8 = (g6 << 2) | (g6 >> 4)
    b8 = (b5 << 3) | (b5 >> 2)
    return np.stack([r8, g8, b8], axis=-1).astype(np.uint8)


# =============================================================================
# Weight pipeline
# =============================================================================

def prepare_fpga_weights(model, n_hidden=32):
    """Extract weights from a PyTorch SIREN model, bake omega, quantize to Q4.28.

    Returns: dict mapping int address -> int value (Q4.28 signed)
    """
    layers = extract_weights_from_model(model)
    weights = {}
    idx = 0

    for layer_idx, (w, b) in enumerate(layers):
        for j in range(w.shape[0]):
            for k in range(w.shape[1]):
                weights[idx] = q428_from_float(float(w[j, k]))
                idx += 1
        for j in range(b.shape[0]):
            weights[idx] = q428_from_float(float(b[j]))
            idx += 1

    return weights


def extract_weights_from_model(model):
    """Extract weights with omega baked in from a PyTorch SIREN model.

    Returns: list of (weight_array, bias_array) tuples per layer.
    """
    import torch
    layers = []
    for layer in model.layers:
        w = layer.linear.weight.detach().cpu().numpy()
        b = layer.linear.bias.detach().cpu().numpy()
        omega = layer.omega_0
        layers.append((w * omega, b * omega))

    w = model.output_layer.weight.detach().cpu().numpy()
    b = model.output_layer.bias.detach().cpu().numpy()
    layers.append((w, b))
    return layers


# =============================================================================
# Coordinate generation
# =============================================================================

def make_fpga_coords(W, H, frame):
    """Generate (x_q, y_q, t_q) matching mandelbrot_top.v MLP mode.

    Returns: (x_q, y_q, t_q) int64 arrays of shape (H*W,)
    """
    # Pixel coordinates
    px = np.arange(W, dtype=np.int64)
    py = np.arange(H, dtype=np.int64)

    # x_q[px] = CRE_START + px * CRE_STEP
    x_1d = to_signed32_vec(np.int64(CRE_START) + px * np.int64(CRE_STEP))
    # y_q[py] = CIM_START + py * CIM_STEP
    y_1d = to_signed32_vec(np.int64(CIM_START) + py * np.int64(CIM_STEP))

    # Meshgrid: y varies along rows, x along columns
    yy, xx = np.meshgrid(y_1d, x_1d, indexing='ij')  # (H, W)
    x_q = xx.ravel()
    y_q = yy.ravel()

    # Time: {3'b000, max_iter[9:0], 19'b0} - 0x10000000
    max_iter_10bit = frame & 0x3FF
    time_q = np.int64((max_iter_10bit << 19) - 0x10000000)
    time_q = to_signed32(int(time_q))
    t_q = np.full(H * W, time_q, dtype=np.int64)

    return x_q, y_q, t_q


# =============================================================================
# High-level rendering
# =============================================================================

def render_frame_fpga(weights, frame, W=FRAME_W, H=FRAME_H, n_hidden=32):
    """Render one frame using bit-exact FPGA simulation.

    weights: dict mapping int address -> int value (Q4.28 signed)
    frame: integer frame number (maps to max_iter time parameter)

    Returns: uint8 array of shape (H, W, 3)
    """
    x_q, y_q, t_q = make_fpga_coords(W, H, frame)
    r_q, g_q, b_q = mlp_forward_fixed_vec(x_q, y_q, t_q, weights, n_hidden)
    rgb565 = pack_rgb565_vec(r_q, g_q, b_q)
    return rgb565_to_rgb888_vec(rgb565).reshape(H, W, 3)


def render_frame_fpga_scalar(weights, frame, W=FRAME_W, H=FRAME_H, n_hidden=32):
    """Render one frame using scalar FPGA simulation (slow, for debugging)."""
    max_iter_10bit = frame & 0x3FF
    time_q = to_signed32(((max_iter_10bit << 19) - 0x10000000) & MASK32)

    img = np.zeros((H, W, 3), dtype=np.uint8)
    for py in range(H):
        y_q = to_signed32((CIM_START + py * CIM_STEP) & MASK32)
        for px in range(W):
            x_q = to_signed32((CRE_START + px * CRE_STEP) & MASK32)
            r_q, g_q, b_q = mlp_forward_fixed(x_q, y_q, time_q, weights, n_hidden)
            rgb565 = pack_rgb565(r_q, g_q, b_q)
            img[py, px] = rgb565_to_rgb888(rgb565)
    return img


def compute_fpga_psnr(model, target_frames, n_hidden=32, W=FRAME_W, H=FRAME_H):
    """Compute FPGA-exact PSNR and float PSNR for a set of frames.

    model: PyTorch SIREN model
    target_frames: np.array (n_frames, H, W) with values in [-1, +1]

    Returns: dict with fpga_psnr, float_psnr, gap per frame and averages
    """
    import torch

    weights = prepare_fpga_weights(model, n_hidden)
    n_frames = len(target_frames)

    fpga_psnrs = []
    float_psnrs = []

    model.eval()
    with torch.no_grad():
        for fi in range(n_frames):
            # FPGA rendering
            fpga_img = render_frame_fpga(weights, fi, W, H, n_hidden)
            # Convert FPGA RGB888 to [-1,+1] grayscale for comparison
            fpga_gray = fpga_img[:, :, 0].astype(np.float32) / 127.5 - 1.0

            # Float rendering (aspect-corrected: y in [-H/W, +H/W])
            aspect_y = H / W
            t_val = (fi / max(n_frames - 1, 1)) * 2.0 - 1.0
            x = np.linspace(-1, 1, W, dtype=np.float32)
            y = np.linspace(-aspect_y, aspect_y, H, dtype=np.float32)
            yy, xx = np.meshgrid(y, x, indexing='ij')
            tt = np.full_like(xx, t_val)
            coords = np.stack([xx.ravel(), yy.ravel(), tt.ravel()], axis=1)
            pred = model(torch.from_numpy(coords)).cpu().numpy()
            float_gray = np.clip(pred[:, 0].reshape(H, W), -1.0, 1.0)

            gt = target_frames[fi]

            # PSNR computation (signal range is 2.0 for [-1,+1])
            fpga_mse = float(np.mean((fpga_gray - gt) ** 2))
            float_mse = float(np.mean((float_gray - gt) ** 2))

            fpga_psnr = 10 * np.log10(4.0 / fpga_mse) if fpga_mse > 0 else 100.0
            float_psnr = 10 * np.log10(4.0 / float_mse) if float_mse > 0 else 100.0

            fpga_psnrs.append(fpga_psnr)
            float_psnrs.append(float_psnr)

    return {
        'fpga_psnrs': fpga_psnrs,
        'float_psnrs': float_psnrs,
        'avg_fpga_psnr': float(np.mean(fpga_psnrs)),
        'avg_float_psnr': float(np.mean(float_psnrs)),
        'gap': float(np.mean(float_psnrs) - np.mean(fpga_psnrs)),
    }


# =============================================================================
# Self-test: verify vectorized matches scalar bit-exact
# =============================================================================

def self_test(n_pixels=200):
    """Run self-test: verify vectorized functions match scalar bit-exact."""
    print("fpga_sim self-test")
    print("=" * 60)
    rng = np.random.RandomState(42)

    # Test 1: to_signed32_vec
    print("  to_signed32_vec...", end=" ")
    test_vals = rng.randint(0, 0x100000000, 1000, dtype=np.int64)
    vec_result = to_signed32_vec(test_vals)
    for i in range(len(test_vals)):
        scalar = to_signed32(int(test_vals[i]))
        assert vec_result[i] == scalar, f"to_signed32 mismatch at {i}: {vec_result[i]} != {scalar}"
    print("PASS")

    # Test 2: q428_mul_vec
    print("  q428_mul_vec...", end=" ")
    a_vals = to_signed32_vec(rng.randint(0, 0x100000000, 1000, dtype=np.int64))
    b_vals = to_signed32_vec(rng.randint(0, 0x100000000, 1000, dtype=np.int64))
    vec_result = q428_mul_vec(a_vals, b_vals)
    for i in range(len(a_vals)):
        scalar = q428_mul(int(a_vals[i]), int(b_vals[i]))
        assert vec_result[i] == scalar, \
            f"q428_mul mismatch at {i}: vec={vec_result[i]} scalar={scalar} a={a_vals[i]} b={b_vals[i]}"
    print("PASS")

    # Test 3: q428_from_float_vec
    print("  q428_from_float_vec...", end=" ")
    float_vals = rng.uniform(-8.0, 8.0, 1000)
    vec_result = q428_from_float_vec(float_vals)
    for i in range(len(float_vals)):
        scalar = q428_from_float(float(float_vals[i]))
        assert vec_result[i] == scalar, \
            f"q428_from_float mismatch at {i}: vec={vec_result[i]} scalar={scalar} f={float_vals[i]}"
    print("PASS")

    # Test 4: sine_lut_vec
    print("  sine_lut_vec...", end=" ")
    angles = to_signed32_vec(rng.randint(0, 0x100000000, 1000, dtype=np.int64))
    vec_result = sine_lut_vec(angles)
    for i in range(len(angles)):
        scalar = sine_lut_fpga(int(angles[i]))
        assert vec_result[i] == scalar, \
            f"sine_lut mismatch at {i}: vec={vec_result[i]} scalar={scalar} angle={angles[i]:#010x}"
    print("PASS")

    # Test 5: pack_rgb565_vec
    print("  pack_rgb565_vec...", end=" ")
    r_vals = to_signed32_vec(rng.randint(0, 0x100000000, 500, dtype=np.int64))
    g_vals = to_signed32_vec(rng.randint(0, 0x100000000, 500, dtype=np.int64))
    b_vals = to_signed32_vec(rng.randint(0, 0x100000000, 500, dtype=np.int64))
    vec_result = pack_rgb565_vec(r_vals, g_vals, b_vals)
    for i in range(len(r_vals)):
        scalar = pack_rgb565(int(r_vals[i]), int(g_vals[i]), int(b_vals[i]))
        assert vec_result[i] == scalar, \
            f"pack_rgb565 mismatch at {i}: vec={vec_result[i]} scalar={scalar}"
    print("PASS")

    # Test 6: rgb565_to_rgb888_vec
    print("  rgb565_to_rgb888_vec...", end=" ")
    rgb_vals = rng.randint(0, 0x10000, 500).astype(np.int64)
    vec_result = rgb565_to_rgb888_vec(rgb_vals)
    for i in range(len(rgb_vals)):
        scalar = rgb565_to_rgb888(int(rgb_vals[i]))
        assert tuple(vec_result[i]) == scalar, \
            f"rgb565_to_rgb888 mismatch at {i}: vec={tuple(vec_result[i])} scalar={scalar}"
    print("PASS")

    # Test 7: Full MLP forward pass (vectorized vs scalar)
    print("  mlp_forward_fixed_vec vs scalar...", end=" ")

    # Generate fake weights (enough for 3->32->32->3 = 1283 params)
    n_hidden = 32
    geo = get_layer_geometry(n_hidden)
    n_params = geo[-1]['b_base'] + 3  # last bias base + 3 output biases
    weights = {}
    for i in range(n_params):
        weights[i] = to_signed32(int(rng.randint(0, 0x100000000)))

    # Generate random pixel coordinates
    x_q = to_signed32_vec(rng.randint(0, 0x100000000, n_pixels, dtype=np.int64))
    y_q = to_signed32_vec(rng.randint(0, 0x100000000, n_pixels, dtype=np.int64))
    t_q = to_signed32_vec(rng.randint(0, 0x100000000, n_pixels, dtype=np.int64))

    # Vectorized
    vr, vg, vb = mlp_forward_fixed_vec(x_q, y_q, t_q, weights, n_hidden)

    # Scalar, check each pixel
    mismatches = 0
    for i in range(n_pixels):
        sr, sg, sb = mlp_forward_fixed(int(x_q[i]), int(y_q[i]), int(t_q[i]), weights, n_hidden)
        if vr[i] != sr or vg[i] != sg or vb[i] != sb:
            mismatches += 1
            if mismatches <= 3:
                print(f"\n    Pixel {i}: vec=({vr[i]},{vg[i]},{vb[i]}) scalar=({sr},{sg},{sb})")

    if mismatches == 0:
        print("PASS")
    else:
        print(f"FAIL ({mismatches}/{n_pixels} mismatches)")
        return False

    # Test 8: make_fpga_coords spot check
    print("  make_fpga_coords...", end=" ")
    x_q_f, y_q_f, t_q_f = make_fpga_coords(FRAME_W, FRAME_H, 0)
    assert len(x_q_f) == FRAME_W * FRAME_H
    # First pixel (0,0): x = CRE_START, y = CIM_START
    assert x_q_f[0] == CRE_START, f"x[0]={x_q_f[0]} expected {CRE_START}"
    assert y_q_f[0] == CIM_START, f"y[0]={y_q_f[0]} expected {CIM_START}"
    # Frame 0: time = (0 << 19) - 0x10000000 = -0x10000000 = -1.0
    assert t_q_f[0] == -0x10000000, f"t[0]={t_q_f[0]:#010x} expected -0x10000000"
    # Frame 512: time = (512 << 19) - 0x10000000 = 0 = 0.0
    _, _, t512 = make_fpga_coords(FRAME_W, FRAME_H, 512)
    assert t512[0] == 0, f"t(frame=512)={t512[0]} expected 0"
    print("PASS")

    print("=" * 60)
    print("All self-tests passed.")
    return True


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    import sys
    import time

    ok = self_test()
    if not ok:
        sys.exit(1)

    # If mlp_weights.vh exists, benchmark vectorized rendering
    project_root = Path(__file__).parent.parent
    weights_path = project_root / 'rtl' / 'mlp_weights.vh'
    if weights_path.exists():
        print(f"\nBenchmark: rendering frame 0 from {weights_path}")
        weights = parse_weights_vh(weights_path)
        print(f"  Loaded {len(weights)} parameters")

        t0 = time.time()
        img = render_frame_fpga(weights, 0)
        t1 = time.time()
        print(f"  Vectorized: {t1 - t0:.3f}s for {FRAME_W}x{FRAME_H} = {FRAME_W * FRAME_H} pixels")

        try:
            from PIL import Image
            out_path = project_root / 'scripts' / 'fpga_sim_frame0.png'
            Image.fromarray(img).save(out_path)
            print(f"  Saved: {out_path}")
        except ImportError:
            print("  (PIL not available, skipping image save)")
    else:
        print(f"\n  {weights_path} not found, skipping benchmark")
