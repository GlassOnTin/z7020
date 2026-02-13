#!/usr/bin/env python3
"""train_siren.py — Train a SIREN network and export weights for FPGA.

Trains a small SIREN (Sinusoidal Representation Network) on procedural
patterns, quantizes to Q4.28 fixed-point, and exports as a Verilog
include file for synthesis on the Z7020.

Network: 3 inputs (x, y, t) → 16 hidden → 16 hidden → 3 outputs (R, G, B)
Activation: sin(x) for hidden layers, linear output clamped to [-1, +1]

Usage:
    python3 scripts/train_siren.py [--epochs 2000] [--lr 1e-4] [--output rtl/mlp_weights.vh]
"""

import argparse
import math
import struct
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# =========================================================
# Q4.28 Fixed-point conversion
# =========================================================
FRAC_BITS = 28
SCALE = 1 << FRAC_BITS  # 268435456

def float_to_q428(val):
    """Convert float to Q4.28 signed 32-bit integer."""
    clamped = max(-8.0, min(val, 8.0 - 1.0/SCALE))
    raw = int(round(clamped * SCALE))
    # Wrap to signed 32-bit
    if raw < 0:
        raw = raw & 0xFFFFFFFF
    return raw

def q428_to_verilog_hex(val):
    """Format Q4.28 value as Verilog signed hex literal."""
    return f"32'sh{val:08X}"


# =========================================================
# SIREN network (PyTorch)
# =========================================================
if HAS_TORCH:
    class SineLayer(nn.Module):
        def __init__(self, in_features, out_features, omega_0=30.0, is_first=False):
            super().__init__()
            self.omega_0 = omega_0
            self.linear = nn.Linear(in_features, out_features)
            self.is_first = is_first
            self._init_weights()

        def _init_weights(self):
            with torch.no_grad():
                if self.is_first:
                    self.linear.weight.uniform_(-1.0 / self.linear.in_features,
                                                 1.0 / self.linear.in_features)
                else:
                    bound = math.sqrt(6.0 / self.linear.in_features) / self.omega_0
                    self.linear.weight.uniform_(-bound, bound)

        def forward(self, x):
            return torch.sin(self.omega_0 * self.linear(x))

    class SIREN(nn.Module):
        def __init__(self, in_features=3, hidden_features=16, out_features=3,
                     hidden_layers=1, omega_0=30.0, omega_hidden=1.0):
            super().__init__()
            self.omega_0 = omega_0
            self.omega_hidden = omega_hidden

            layers = []
            # First layer
            layers.append(SineLayer(in_features, hidden_features,
                                    omega_0=omega_0, is_first=True))
            # Hidden layers
            for _ in range(hidden_layers):
                layers.append(SineLayer(hidden_features, hidden_features,
                                        omega_0=omega_hidden))
            self.layers = nn.ModuleList(layers)
            # Output layer (linear, no activation)
            self.output_layer = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                bound = math.sqrt(6.0 / hidden_features) / omega_hidden
                self.output_layer.weight.uniform_(-bound, bound)

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return torch.sin(self.output_layer(x))  # sin() matches FPGA hardware


# =========================================================
# Procedural target patterns
# =========================================================
def target_lava_lamp(x, y, t):
    """Organic flowing color pattern — the 'neural lava lamp'."""
    # Layered sine waves with time evolution
    r = np.sin(3.0*x + 1.5*t) * np.cos(2.0*y + 0.7*t) * 0.5 + \
        np.sin(5.0*x - 2.0*y + t) * 0.3 + \
        np.sin(1.0*x + 4.0*y - 1.3*t) * 0.2
    g = np.sin(2.0*x + 3.0*y + 0.8*t) * 0.4 + \
        np.cos(4.0*x - 1.0*y + 1.2*t) * 0.35 + \
        np.sin(1.5*x + 2.5*y + 2.0*t) * 0.25
    b = np.cos(2.5*x + 1.0*y + 1.5*t) * 0.45 + \
        np.sin(3.5*x + 3.5*y - 0.5*t) * 0.3 + \
        np.cos(1.0*x - 3.0*y + 1.8*t) * 0.25
    return np.stack([r, g, b], axis=-1)


def target_reaction_diffusion(x, y, t):
    """Reaction-diffusion-like pattern."""
    # Approximate RD patterns with interfering waves
    phase = t * 0.5
    r = np.sin(6.0*x + phase) * np.sin(6.0*y + phase*0.7) * 0.6 + \
        np.sin(3.0*(x+y) + phase*1.3) * 0.4
    g = np.sin(5.0*x - 3.0*y + phase*0.9) * 0.5 + \
        np.cos(7.0*y + phase*1.1) * np.sin(2.0*x) * 0.5
    b = np.cos(4.0*x + 5.0*y - phase*0.6) * 0.55 + \
        np.sin(8.0*x*y + phase*1.4) * 0.45
    return np.stack([np.tanh(r), np.tanh(g), np.tanh(b)], axis=-1)


def target_plasma(x, y, t):
    """Classic plasma effect."""
    v1 = np.sin(x * 4.0 + t)
    v2 = np.sin(4.0 * (y * np.cos(t * 0.5) + x * np.sin(t * 0.5)))
    v3 = np.sin(np.sqrt((x*3)**2 + (y*3)**2) + t)
    cx = x + 0.5 * np.sin(t * 0.3)
    cy = y + 0.5 * np.cos(t * 0.4)
    v4 = np.sin(np.sqrt(cx**2 + cy**2 + 1.0) * 3.0)
    v = (v1 + v2 + v3 + v4) * 0.25
    r = np.sin(v * np.pi)
    g = np.sin(v * np.pi + 2.0*np.pi/3.0)
    b = np.sin(v * np.pi + 4.0*np.pi/3.0)
    return np.stack([r, g, b], axis=-1)


def target_hsv_flow(x, y, t):
    """Smooth rainbow flow — hue varies with position and time, full saturation.

    Uses cosine-based HSV→RGB so all 3 channels are always in [0,1] before
    mapping to [-1,+1].  No regions of pure black in any channel, giving
    rich mixed colors (cyans, magentas, yellows) everywhere.
    """
    # Scalar field → hue angle
    v1 = np.sin(x * 3.0 + t)
    v2 = np.sin(y * 2.5 + t * 0.7)
    v3 = np.sin(np.sqrt(x**2 + y**2 + 0.01) * 4.0 - t * 0.9)
    v4 = np.sin((x - y) * 2.0 + t * 1.1)
    angle = (v1 + v2 + v3 + v4) * (np.pi * 0.5)

    # HSV→RGB via cosine (S=1, V=1): always in [0,1]
    r = 0.5 + 0.5 * np.cos(angle)
    g = 0.5 + 0.5 * np.cos(angle - 2.0 * np.pi / 3.0)
    b = 0.5 + 0.5 * np.cos(angle - 4.0 * np.pi / 3.0)

    # Map [0,1] → [-1,+1] for SIREN output range
    return np.stack([r * 2 - 1, g * 2 - 1, b * 2 - 1], axis=-1)


PATTERNS = {
    'lava_lamp': target_lava_lamp,
    'reaction_diffusion': target_reaction_diffusion,
    'plasma': target_plasma,
    'hsv_flow': target_hsv_flow,
}


# =========================================================
# Training
# =========================================================
def generate_training_data(pattern_fn, n_spatial=64, n_temporal=32):
    """Generate training samples: (x, y, t) → RGB, all in [-1, +1]."""
    x = np.linspace(-1, 1, n_spatial)
    y = np.linspace(-1, 1, n_spatial)
    t = np.linspace(0, 4*np.pi, n_temporal)  # ~2 full slow cycles
    xx, yy, tt = np.meshgrid(x, y, t, indexing='ij')
    coords = np.stack([xx.ravel(), yy.ravel(), tt.ravel()], axis=-1)
    colors = pattern_fn(xx.ravel(), yy.ravel(), tt.ravel())
    # Clamp to [-1, +1]
    colors = np.clip(colors, -1, 1)
    return coords.astype(np.float32), colors.astype(np.float32)


def train_siren(pattern='plasma', epochs=2000, lr=1e-4, hidden=16):
    """Train SIREN on a procedural pattern."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required for training. Install with: pip install torch")

    print(f"Training SIREN on '{pattern}' pattern...")
    print(f"  Architecture: 3 → {hidden} → {hidden} → 3")
    print(f"  Epochs: {epochs}, LR: {lr}")

    pattern_fn = PATTERNS[pattern]
    coords, colors = generate_training_data(pattern_fn, n_spatial=48, n_temporal=24)
    print(f"  Training samples: {len(coords)}")

    coords_t = torch.from_numpy(coords)
    colors_t = torch.from_numpy(colors)

    model = SIREN(in_features=3, hidden_features=hidden, out_features=3,
                  hidden_layers=1, omega_0=10.0, omega_hidden=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        pred = model(coords_t)
        loss = loss_fn(pred, colors_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 200 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:5d}/{epochs}: loss = {loss.item():.6f}")

    print(f"  Final loss: {loss.item():.6f}")
    return model


# =========================================================
# Weight extraction and export
# =========================================================
def extract_weights(model):
    """Extract weights and biases from trained SIREN model.

    Returns list of (weight_matrix, bias_vector) tuples per layer.
    For SIREN, the first two layers have omega baked into weights.
    """
    layers = []

    # Hidden layers (SineLayer): omega is applied inside forward(),
    # but for FPGA we need to bake omega into weights since we only
    # have sin(x), not sin(omega*x).
    for i, layer in enumerate(model.layers):
        w = layer.linear.weight.detach().numpy()  # [out, in]
        b = layer.linear.bias.detach().numpy()     # [out]
        omega = layer.omega_0
        # Bake omega into weights and bias
        layers.append((w * omega, b * omega))

    # Output layer (linear, no omega)
    w = model.output_layer.weight.detach().numpy()
    b = model.output_layer.bias.detach().numpy()
    layers.append((w, b))

    return layers


def export_verilog(layers, output_path, hidden=16):
    """Export weights as Verilog include file (mlp_weights.vh)."""
    all_params = []
    for layer_idx, (weights, biases) in enumerate(layers):
        # Weights stored row-major: [neuron_j][input_k]
        for j in range(weights.shape[0]):
            for k in range(weights.shape[1]):
                all_params.append((f"L{layer_idx} w[{j}][{k}]",
                                   weights[j, k]))
        # Biases
        for j in range(biases.shape[0]):
            all_params.append((f"L{layer_idx} b[{j}]",
                               biases[j]))

    n_params = len(all_params)
    print(f"  Total parameters: {n_params}")

    # Check weight ranges
    vals = [v for _, v in all_params]
    print(f"  Weight range: [{min(vals):.4f}, {max(vals):.4f}]")
    clipped = sum(1 for v in vals if abs(v) > 7.99)
    if clipped:
        print(f"  WARNING: {clipped} weights clipped to Q4.28 range [-8, +8)")

    lines = []
    lines.append("// mlp_weights.vh — Auto-generated by train_siren.py")
    lines.append(f"// Network: 3 -> {hidden} -> {hidden} -> 3 SIREN")
    lines.append(f"// Total parameters: {n_params}")
    lines.append(f"// Weight range: [{min(vals):.4f}, {max(vals):.4f}]")
    lines.append("")
    lines.append("initial begin")

    for idx, (name, val) in enumerate(all_params):
        q428 = float_to_q428(val)
        lines.append(f"    weight_mem[{idx:3d}] = {q428_to_verilog_hex(q428)};  // {name} = {val:+.6f}")

    lines.append("end")
    lines.append("")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"  Written to: {output_path}")


def export_numpy_fallback(output_path, hidden=16):
    """Generate weights without PyTorch using numpy random initialization.

    Creates a SIREN-like weight initialization that produces interesting
    patterns even without training. Uses the SIREN initialization scheme:
    - First layer: uniform(-1/fan_in, 1/fan_in) * omega_0
    - Hidden layers: uniform(-sqrt(6/fan_in)/omega, sqrt(6/fan_in)/omega) * omega
    - Output layer: uniform(-sqrt(6/fan_in)/omega, sqrt(6/fan_in)/omega)
    """
    np.random.seed(42)
    omega_0 = 30.0
    omega_hidden = 1.0

    layers = []

    # Layer 0: 3 → hidden (first layer with omega_0=30)
    fan_in = 3
    w0 = np.random.uniform(-1.0/fan_in, 1.0/fan_in, (hidden, fan_in)) * omega_0
    b0 = np.random.uniform(-1.0/fan_in, 1.0/fan_in, (hidden,)) * omega_0
    layers.append((w0, b0))

    # Layer 1: hidden → hidden (omega_hidden=1)
    fan_in = hidden
    bound = math.sqrt(6.0 / fan_in) / omega_hidden
    w1 = np.random.uniform(-bound, bound, (hidden, hidden)) * omega_hidden
    b1 = np.zeros(hidden)
    layers.append((w1, b1))

    # Layer 2: hidden → 3 (output, no omega)
    bound = math.sqrt(6.0 / hidden) / omega_hidden
    w2 = np.random.uniform(-bound, bound, (3, hidden))
    b2 = np.zeros(3)
    layers.append((w2, b2))

    print(f"  Generated random SIREN initialization (seed=42)")
    export_verilog(layers, output_path, hidden)


# =========================================================
# Binary export (for SD card loading)
# =========================================================
def export_binary(layers, output_path):
    """Export weights as raw binary file (Q4.28, little-endian)."""
    all_vals = []
    for weights, biases in layers:
        for j in range(weights.shape[0]):
            for k in range(weights.shape[1]):
                all_vals.append(float_to_q428(weights[j, k]))
        for j in range(biases.shape[0]):
            all_vals.append(float_to_q428(biases[j]))

    with open(output_path, 'wb') as f:
        for val in all_vals:
            # Write as signed 32-bit little-endian
            f.write(struct.pack('<I', val))

    print(f"  Binary weights: {output_path} ({len(all_vals) * 4} bytes)")


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(description='Train SIREN and export weights for FPGA')
    parser.add_argument('--pattern', choices=list(PATTERNS.keys()), default='plasma',
                        help='Target pattern to train on')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Hidden layer size')
    parser.add_argument('--output', default='rtl/mlp_weights.vh',
                        help='Output Verilog include file')
    parser.add_argument('--binary', default=None,
                        help='Optional binary output file for SD card loading')
    parser.add_argument('--no-train', action='store_true',
                        help='Skip training, use random initialization')
    args = parser.parse_args()

    if args.no_train or not HAS_TORCH:
        if not HAS_TORCH:
            print("PyTorch not available, using random initialization")
        export_numpy_fallback(args.output, args.hidden)
    else:
        model = train_siren(args.pattern, args.epochs, args.lr, args.hidden)
        layers = extract_weights(model)
        export_verilog(layers, args.output, args.hidden)
        if args.binary:
            export_binary(layers, args.binary)

    print("Done.")


if __name__ == '__main__':
    main()
