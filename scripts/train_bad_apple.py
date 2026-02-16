#!/usr/bin/env python3
"""train_bad_apple.py — Train SIREN to compress Bad Apple video segments.

Each segment of ~200 frames is encoded as a SIREN network:
    f(x, y, t) → intensity

The trained weights ARE the compressed video (~5KB per segment).
The FPGA inference engine decodes in real-time.

Usage:
    # Train a single segment (CPU POC)
    python3 scripts/train_bad_apple.py --segment 14 --epochs 5000

    # Train all segments
    python3 scripts/train_bad_apple.py --all --epochs 5000

    # Evaluate quality of a trained segment
    python3 scripts/train_bad_apple.py --segment 14 --eval-only

    # Export for FPGA
    python3 scripts/train_bad_apple.py --segment 14 --export-fpga

    # Train with FPGA-exact validation
    python3 scripts/train_bad_apple.py --segment 14 --epochs 1000 --fpga-validate
"""

import argparse
import math
import os
import struct
import sys
import time
import numpy as np
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from fpga_sim import prepare_fpga_weights, render_frame_fpga, compute_fpga_psnr
    HAS_FPGA_SIM = True
except ImportError:
    HAS_FPGA_SIM = False

# =========================================================
# Constants
# =========================================================
FRAME_DIR = Path(__file__).parent.parent / "bad_apple" / "frames"
WEIGHTS_DIR = Path(__file__).parent.parent / "bad_apple" / "weights"
EVAL_DIR = Path(__file__).parent.parent / "bad_apple" / "eval"

TOTAL_FRAMES = 6572
FPS = 30
FRAME_W = 320
FRAME_H = 172
ASPECT_Y = FRAME_H / FRAME_W  # 0.5375 — y range is [-ASPECT_Y, +ASPECT_Y]
FRAMES_PER_SEGMENT = 10   # 10 frames per segment for better SIREN quality
N_SEGMENTS = (TOTAL_FRAMES + FRAMES_PER_SEGMENT - 1) // FRAMES_PER_SEGMENT  # 658

# Q4.28 fixed-point
FRAC_BITS = 28
SCALE = 1 << FRAC_BITS

# =========================================================
# SIREN model
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
        def __init__(self, in_features=3, hidden_features=32, out_features=3,
                     hidden_layers=1, omega_0=10.0, omega_hidden=10.0):
            super().__init__()
            self.omega_0 = omega_0
            self.omega_hidden = omega_hidden

            layers = []
            layers.append(SineLayer(in_features, hidden_features,
                                    omega_0=omega_0, is_first=True))
            for _ in range(hidden_layers):
                layers.append(SineLayer(hidden_features, hidden_features,
                                        omega_0=omega_hidden))
            self.layers = nn.ModuleList(layers)
            self.output_layer = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                bound = math.sqrt(6.0 / hidden_features) / omega_hidden
                self.output_layer.weight.uniform_(-bound, bound)

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return torch.sin(self.output_layer(x))


# =========================================================
# Data loading
# =========================================================
def load_segment_frames(segment_idx):
    """Load grayscale frames for a segment as float array [-1, +1].

    Returns: np.array of shape (n_frames, H, W) with values in [-1, +1].
    """
    start = segment_idx * FRAMES_PER_SEGMENT + 1  # 1-indexed
    end = min(start + FRAMES_PER_SEGMENT, TOTAL_FRAMES + 1)

    frames = []
    for i in range(start, end):
        path = FRAME_DIR / f"frame_{i:05d}.png"
        if not path.exists():
            break
        img = Image.open(path).convert('L')
        arr = np.array(img, dtype=np.float32) / 127.5 - 1.0  # [0,255] → [-1,+1]
        frames.append(arr)

    if not frames:
        raise FileNotFoundError(f"No frames found for segment {segment_idx} (start={start})")

    return np.stack(frames)  # (n_frames, H, W)


def make_training_coords(frames, n_samples=100000):
    """Generate random (x, y, t) → intensity training pairs.

    Args:
        frames: (n_frames, H, W) array in [-1, +1]
        n_samples: number of random samples to generate

    Returns:
        coords: (n_samples, 3) float32 — (x, y, t) in [-1, +1]
        values: (n_samples, 3) float32 — (gray, gray, gray) in [-1, +1]
    """
    n_frames, H, W = frames.shape

    # Random integer indices
    fi = np.random.randint(0, n_frames, n_samples)
    yi = np.random.randint(0, H, n_samples)
    xi = np.random.randint(0, W, n_samples)

    # Map x to [-1, +1], y to [-ASPECT_Y, +ASPECT_Y] (matching FPGA viewport)
    x = (xi / (W - 1)) * 2.0 - 1.0
    y = (yi / (H - 1)) * 2.0 * ASPECT_Y - ASPECT_Y
    t = (fi / max(n_frames - 1, 1)) * 2.0 - 1.0

    coords = np.stack([x, y, t], axis=1).astype(np.float32)

    # Grayscale intensity
    intensity = frames[fi, yi, xi]
    # Replicate to 3 channels (R=G=B) so the existing FPGA pipeline works unchanged
    values = np.stack([intensity, intensity, intensity], axis=1).astype(np.float32)

    return coords, values


def make_full_grid(n_frames, H=FRAME_H, W=FRAME_W):
    """Generate a full coordinate grid for evaluation.

    Returns: coords (n_frames*H*W, 3) float32
    """
    x = np.linspace(-1, 1, W, dtype=np.float32)
    y = np.linspace(-ASPECT_Y, ASPECT_Y, H, dtype=np.float32)
    t = np.linspace(-1, 1, n_frames, dtype=np.float32)

    # Meshgrid: t varies slowest, then y, then x
    tt, yy, xx = np.meshgrid(t, y, x, indexing='ij')
    coords = np.stack([xx.ravel(), yy.ravel(), tt.ravel()], axis=1)
    return coords


# =========================================================
# Training
# =========================================================
def train_segment(segment_idx, epochs=5000, lr=1e-4, hidden=32,
                  n_samples=100000, batch_size=50000, device='cpu',
                  fpga_validate=False):
    """Train SIREN on a video segment."""
    print(f"\n{'='*60}")
    print(f"Training segment {segment_idx}/{N_SEGMENTS-1}")
    print(f"  Frames: {segment_idx*FRAMES_PER_SEGMENT+1} - "
          f"{min((segment_idx+1)*FRAMES_PER_SEGMENT, TOTAL_FRAMES)}")
    print(f"  Architecture: 3 → {hidden} → {hidden} → 3")
    print(f"  Epochs: {epochs}, LR: {lr}, Samples: {n_samples}")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    frames = load_segment_frames(segment_idx)
    n_frames = len(frames)
    print(f"  Loaded {n_frames} frames ({frames.shape})")

    model = SIREN(in_features=3, hidden_features=hidden, out_features=3,
                  hidden_layers=1, omega_0=10.0, omega_hidden=10.0).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.01)
    loss_fn = nn.MSELoss()

    t_start = time.time()
    best_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        # Resample each epoch for better coverage
        coords, values = make_training_coords(frames, n_samples)
        coords_t = torch.from_numpy(coords).to(device)
        values_t = torch.from_numpy(values).to(device)

        # Mini-batch training for memory efficiency
        epoch_loss = 0.0
        n_batches = 0
        perm = torch.randperm(n_samples, device=device)

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            idx = perm[start:end]

            pred = model(coords_t[idx])
            loss = loss_fn(pred, values_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 200 == 0 or epoch == 0:
            elapsed = time.time() - t_start
            rate = (epoch + 1) / elapsed
            eta = (epochs - epoch - 1) / rate
            print(f"  Epoch {epoch+1:5d}/{epochs}: loss={avg_loss:.6f} "
                  f"best={best_loss:.6f} [{elapsed:.0f}s, ETA {eta:.0f}s]")

        # FPGA-exact validation every 500 epochs
        if fpga_validate and HAS_FPGA_SIM and (epoch + 1) % 500 == 0:
            model.eval()
            fpga_stats = compute_fpga_psnr(model, frames, hidden)
            model.train()
            print(f"    FPGA: {fpga_stats['avg_fpga_psnr']:.1f} dB  "
                  f"float: {fpga_stats['avg_float_psnr']:.1f} dB  "
                  f"gap: {fpga_stats['gap']:.1f} dB")

    print(f"  Training complete. Best loss: {best_loss:.6f}")

    # Restore best weights
    model.load_state_dict(best_state)
    model.cpu()
    return model, frames


# =========================================================
# Evaluation
# =========================================================
def evaluate_segment(model, frames, segment_idx, save_images=True,
                     fpga_validate=False):
    """Evaluate reconstruction quality of a trained segment.

    Returns: dict with PSNR, SSIM-approx, compression ratio
    """
    n_frames = len(frames)
    print(f"\n  Evaluating segment {segment_idx} ({n_frames} frames)...")

    model.eval()
    with torch.no_grad():
        # Evaluate frame-by-frame to save memory
        mse_total = 0.0
        frame_psnrs = []

        for fi in range(n_frames):
            t_val = (fi / max(n_frames - 1, 1)) * 2.0 - 1.0

            x = np.linspace(-1, 1, FRAME_W, dtype=np.float32)
            y = np.linspace(-ASPECT_Y, ASPECT_Y, FRAME_H, dtype=np.float32)
            yy, xx = np.meshgrid(y, x, indexing='ij')
            tt = np.full_like(xx, t_val)
            coords = np.stack([xx.ravel(), yy.ravel(), tt.ravel()], axis=1)

            pred = model(torch.from_numpy(coords)).numpy()
            # Take first channel (all 3 are identical for grayscale)
            pred_gray = pred[:, 0].reshape(FRAME_H, FRAME_W)
            gt_gray = frames[fi]

            # Clamp to [-1, +1]
            pred_gray = np.clip(pred_gray, -1.0, 1.0)

            mse = np.mean((pred_gray - gt_gray) ** 2)
            mse_total += mse

            # PSNR (signal range is 2.0 for [-1,+1])
            if mse > 0:
                psnr = 10 * np.log10(4.0 / mse)
            else:
                psnr = 100.0
            frame_psnrs.append(psnr)

            # Save sample frames
            if save_images and fi % max(1, n_frames // 5) == 0:
                EVAL_DIR.mkdir(parents=True, exist_ok=True)
                # Convert to uint8
                gt_u8 = ((gt_gray + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                pred_u8 = ((pred_gray + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                # Side-by-side comparison
                comparison = np.hstack([gt_u8, pred_u8])
                img = Image.fromarray(comparison, mode='L')
                fname = f"seg{segment_idx:02d}_frame{fi:03d}_compare.png"
                img.save(EVAL_DIR / fname)

        avg_mse = mse_total / n_frames
        avg_psnr = np.mean(frame_psnrs)
        min_psnr = np.min(frame_psnrs)

    # Compression ratio
    raw_bytes = n_frames * FRAME_H * FRAME_W  # 1 byte per pixel grayscale
    n_params = sum(p.numel() for p in model.parameters())
    weight_bytes = n_params * 4  # Q4.28 = 4 bytes each
    ratio = raw_bytes / weight_bytes

    print(f"  Results:")
    print(f"    Float PSNR: {avg_psnr:.1f} dB (avg), {min_psnr:.1f} dB (worst)")
    print(f"    Avg MSE:    {avg_mse:.6f}")
    print(f"    Parameters: {n_params} ({weight_bytes} bytes)")
    print(f"    Raw size:   {raw_bytes} bytes ({n_frames} frames)")
    print(f"    Compression: {ratio:.1f}:1")

    result = {
        'avg_psnr': avg_psnr,
        'min_psnr': min_psnr,
        'avg_mse': avg_mse,
        'n_params': n_params,
        'weight_bytes': weight_bytes,
        'raw_bytes': raw_bytes,
        'compression_ratio': ratio,
    }

    # FPGA-exact validation
    if fpga_validate and HAS_FPGA_SIM:
        hidden = sum(1 for p in model.parameters()) // 2  # rough estimate
        # Use actual hidden size from model
        hidden = model.layers[0].linear.out_features
        fpga_stats = compute_fpga_psnr(model, frames, hidden)
        result['avg_fpga_psnr'] = fpga_stats['avg_fpga_psnr']
        result['fpga_gap'] = fpga_stats['gap']
        print(f"    FPGA PSNR:  {fpga_stats['avg_fpga_psnr']:.1f} dB")
        print(f"    Float-FPGA gap: {fpga_stats['gap']:.1f} dB")

    return result


# =========================================================
# Export
# =========================================================
def float_to_q428(val):
    clamped = max(-8.0, min(val, 8.0 - 1.0 / SCALE))
    raw = int(round(clamped * SCALE))
    if raw < 0:
        raw = raw & 0xFFFFFFFF
    return raw


def q428_to_verilog_hex(val):
    return f"32'sh{val:08X}"


def extract_weights(model):
    """Extract weights with omega baked in (matching train_siren.py)."""
    layers = []
    for layer in model.layers:
        w = layer.linear.weight.detach().numpy()
        b = layer.linear.bias.detach().numpy()
        omega = layer.omega_0
        layers.append((w * omega, b * omega))

    w = model.output_layer.weight.detach().numpy()
    b = model.output_layer.bias.detach().numpy()
    layers.append((w, b))
    return layers


def export_verilog_single(layers, output_path, mem_name="weight_mem", hidden=32):
    """Export weights as Verilog initial block."""
    all_params = []
    for layer_idx, (weights, biases) in enumerate(layers):
        for j in range(weights.shape[0]):
            for k in range(weights.shape[1]):
                all_params.append((f"L{layer_idx} w[{j}][{k}]", weights[j, k]))
        for j in range(biases.shape[0]):
            all_params.append((f"L{layer_idx} b[{j}]", biases[j]))

    n_params = len(all_params)
    vals = [v for _, v in all_params]

    lines = []
    lines.append(f"// {mem_name} — Bad Apple segment weights")
    lines.append(f"// Network: 3 -> {hidden} -> {hidden} -> 3 SIREN")
    lines.append(f"// Parameters: {n_params}")
    lines.append(f"// Weight range: [{min(vals):.4f}, {max(vals):.4f}]")
    lines.append("")
    lines.append("initial begin")

    for idx, (name, val) in enumerate(all_params):
        q428 = float_to_q428(val)
        lines.append(f"    {mem_name}[{idx:3d}] = {q428_to_verilog_hex(q428)};  // {name} = {val:+.6f}")

    lines.append("end")
    lines.append("")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"  Exported: {output_path} ({n_params} params)")
    return n_params


def export_binary(layers, output_path):
    """Export weights as raw binary (Q4.28, little-endian)."""
    all_vals = []
    for weights, biases in layers:
        for j in range(weights.shape[0]):
            for k in range(weights.shape[1]):
                all_vals.append(float_to_q428(weights[j, k]))
        for j in range(biases.shape[0]):
            all_vals.append(float_to_q428(biases[j]))

    with open(output_path, 'wb') as f:
        for val in all_vals:
            f.write(struct.pack('<I', val))

    print(f"  Binary: {output_path} ({len(all_vals) * 4} bytes)")


def export_dual_verilog(seg_a_path, seg_b_path, output_path, hidden=32):
    """Combine two segment .pt files into a dual-weight Verilog include (for morphing)."""
    model_a = SIREN(in_features=3, hidden_features=hidden, out_features=3,
                    hidden_layers=1, omega_0=10.0, omega_hidden=10.0)
    model_b = SIREN(in_features=3, hidden_features=hidden, out_features=3,
                    hidden_layers=1, omega_0=10.0, omega_hidden=10.0)
    model_a.load_state_dict(torch.load(seg_a_path, weights_only=True))
    model_b.load_state_dict(torch.load(seg_b_path, weights_only=True))

    layers_a = extract_weights(model_a)
    layers_b = extract_weights(model_b)

    with open(output_path, 'w') as f:
        f.write("// mlp_weights.vh — Bad Apple dual segment weights (for morphing)\n")
        f.write(f"// Network: 3 -> {hidden} -> {hidden} -> 3 SIREN\n\n")

        # Pattern A
        for mem_name, layers in [("weight_mem", layers_a), ("weight_b_mem", layers_b)]:
            f.write(f"initial begin\n")
            idx = 0
            for layer_idx, (weights, biases) in enumerate(layers):
                for j in range(weights.shape[0]):
                    for k in range(weights.shape[1]):
                        q = float_to_q428(weights[j, k])
                        f.write(f"    {mem_name}[{idx:3d}] = {q428_to_verilog_hex(q)};\n")
                        idx += 1
                for j in range(biases.shape[0]):
                    q = float_to_q428(biases[j])
                    f.write(f"    {mem_name}[{idx:3d}] = {q428_to_verilog_hex(q)};\n")
                    idx += 1
            f.write("end\n\n")

    print(f"  Dual export: {output_path}")


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(description='Train SIREN for Bad Apple video compression')
    parser.add_argument('--segment', type=int, default=14,
                        help=f'Segment index (0-{N_SEGMENTS-1}). Default 14 (good silhouettes)')
    parser.add_argument('--all', action='store_true',
                        help='Train all segments')
    parser.add_argument('--epochs', type=int, default=5000,
                        help='Training epochs per segment')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--hidden', type=int, default=32,
                        help='Hidden layer size (must match FPGA N_HIDDEN)')
    parser.add_argument('--samples', type=int, default=100000,
                        help='Training samples per epoch')
    parser.add_argument('--batch-size', type=int, default=50000,
                        help='Mini-batch size')
    parser.add_argument('--eval-only', action='store_true',
                        help='Evaluate existing model (skip training)')
    parser.add_argument('--export-fpga', action='store_true',
                        help='Export best model as Verilog include')
    parser.add_argument('--device', default='cpu',
                        help='Training device (cpu or cuda)')
    parser.add_argument('--fpga-validate', action='store_true',
                        help='Run FPGA-exact validation during training and evaluation')
    parser.add_argument('--export-sd', metavar='DIR',
                        help='Export all trained segments as seg_NNN.bin to DIR for SD card')
    args = parser.parse_args()

    if args.fpga_validate and not HAS_FPGA_SIM:
        print("ERROR: --fpga-validate requires fpga_sim.py (not found in path)")
        sys.exit(1)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    if args.all:
        segments = range(N_SEGMENTS)
    else:
        segments = [args.segment]

    for seg in segments:
        model_path = WEIGHTS_DIR / f"segment_{seg:02d}.pt"

        if args.eval_only:
            if not model_path.exists():
                print(f"  No model found at {model_path}, skipping")
                continue
            model = SIREN(in_features=3, hidden_features=args.hidden, out_features=3,
                          hidden_layers=1, omega_0=10.0, omega_hidden=10.0)
            model.load_state_dict(torch.load(model_path, weights_only=True))
            frames = load_segment_frames(seg)
            evaluate_segment(model, frames, seg, fpga_validate=args.fpga_validate)
            continue

        model, frames = train_segment(seg, args.epochs, args.lr, args.hidden,
                                       args.samples, args.batch_size, args.device,
                                       fpga_validate=args.fpga_validate)

        # Save PyTorch model
        torch.save(model.state_dict(), model_path)
        print(f"  Saved: {model_path}")

        # Evaluate
        stats = evaluate_segment(model, frames, seg,
                                 fpga_validate=args.fpga_validate)

        # Export
        if args.export_fpga or args.all:
            layers = extract_weights(model)
            vh_path = WEIGHTS_DIR / f"segment_{seg:02d}.vh"
            export_verilog_single(layers, vh_path, hidden=args.hidden)
            bin_path = WEIGHTS_DIR / f"segment_{seg:02d}.bin"
            export_binary(layers, bin_path)

    # Summary
    if args.all:
        total_weight_bytes = 0
        total_raw_bytes = 0
        for seg in segments:
            bin_path = WEIGHTS_DIR / f"segment_{seg:02d}.bin"
            if bin_path.exists():
                total_weight_bytes += bin_path.stat().st_size
        total_raw_bytes = TOTAL_FRAMES * FRAME_H * FRAME_W
        if total_weight_bytes > 0:
            print(f"\n{'='*60}")
            print(f"Total: {total_weight_bytes} bytes compressed, "
                  f"{total_raw_bytes} bytes raw")
            print(f"Overall compression: {total_raw_bytes/total_weight_bytes:.1f}:1")
            print(f"{'='*60}")

    # Export all trained segments as seg_NNN.bin for SD card
    if args.export_sd:
        sd_dir = Path(args.export_sd)
        sd_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"Exporting segments to SD card directory: {sd_dir}")
        print(f"{'='*60}")

        exported = 0
        for seg in range(N_SEGMENTS):
            model_path = WEIGHTS_DIR / f"segment_{seg:02d}.pt"
            if not model_path.exists():
                print(f"  seg_{seg:03d}: no model, skipping")
                continue

            model = SIREN(in_features=3, hidden_features=args.hidden, out_features=3,
                          hidden_layers=1, omega_0=10.0, omega_hidden=10.0)
            model.load_state_dict(torch.load(model_path, weights_only=True))
            layers = extract_weights(model)

            sd_bin_path = sd_dir / f"seg_{seg:03d}.bin"
            export_binary(layers, sd_bin_path)
            exported += 1

        total_bytes = sum(
            (sd_dir / f"seg_{seg:03d}.bin").stat().st_size
            for seg in range(N_SEGMENTS)
            if (sd_dir / f"seg_{seg:03d}.bin").exists()
        )
        print(f"\nExported {exported} segments, {total_bytes} bytes total "
              f"({total_bytes/1024:.1f} KB)")
        print(f"Copy contents of {sd_dir} to SD card root alongside BOOT.bin")


if __name__ == '__main__':
    main()
