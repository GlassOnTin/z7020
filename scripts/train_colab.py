#!/usr/bin/env python3
"""train_colab.py — GPU-accelerated Bad Apple SIREN training for Google Colab.

Self-contained: no external dependencies beyond torch, numpy, PIL.
Trains all segments, saves weights as .pt and .bin files.

Usage (in Colab after uploading frames):
    !python train_colab.py --start 0 --end 658 --epochs 5000
"""

import argparse, math, os, struct, sys, time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image

# =========================================================
# Constants
# =========================================================
FRAME_DIR = Path('frames')
WEIGHTS_DIR = Path('weights')
TOTAL_FRAMES = 6572
FRAME_W, FRAME_H = 320, 172
ASPECT_Y = FRAME_H / FRAME_W  # 0.5375
FRAMES_PER_SEGMENT = 10
N_SEGMENTS = (TOTAL_FRAMES + FRAMES_PER_SEGMENT - 1) // FRAMES_PER_SEGMENT
FRAC_BITS = 28
SCALE = 1 << FRAC_BITS

# =========================================================
# SIREN model
# =========================================================
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=30.0, is_first=False):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        self.is_first = is_first
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1.0 / in_features, 1.0 / in_features)
            else:
                bound = math.sqrt(6.0 / in_features) / omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    def __init__(self, in_features=3, hidden_features=32, out_features=3,
                 hidden_layers=1, omega_0=10.0, omega_hidden=10.0):
        super().__init__()
        self.omega_0 = omega_0
        self.omega_hidden = omega_hidden
        layers = [SineLayer(in_features, hidden_features, omega_0=omega_0, is_first=True)]
        for _ in range(hidden_layers):
            layers.append(SineLayer(hidden_features, hidden_features, omega_0=omega_hidden))
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
    start = segment_idx * FRAMES_PER_SEGMENT + 1
    end = min(start + FRAMES_PER_SEGMENT, TOTAL_FRAMES + 1)
    frames = []
    for i in range(start, end):
        path = FRAME_DIR / f"frame_{i:05d}.png"
        if not path.exists():
            break
        arr = np.array(Image.open(path).convert('L'), dtype=np.float32) / 127.5 - 1.0
        frames.append(arr)
    if not frames:
        raise FileNotFoundError(f"No frames for segment {segment_idx}")
    return np.stack(frames)

def make_training_coords(frames, n_samples=200000):
    n_frames, H, W = frames.shape
    fi = np.random.randint(0, n_frames, n_samples)
    yi = np.random.randint(0, H, n_samples)
    xi = np.random.randint(0, W, n_samples)
    x = (xi / (W - 1)) * 2.0 - 1.0
    y = (yi / (H - 1)) * 2.0 * ASPECT_Y - ASPECT_Y
    t = (fi / max(n_frames - 1, 1)) * 2.0 - 1.0
    coords = np.stack([x, y, t], axis=1).astype(np.float32)
    intensity = frames[fi, yi, xi]
    values = np.stack([intensity, intensity, intensity], axis=1).astype(np.float32)
    return coords, values

# =========================================================
# Export
# =========================================================
def float_to_q428(val):
    clamped = max(-8.0, min(val, 8.0 - 1.0 / SCALE))
    raw = int(round(clamped * SCALE))
    if raw < 0:
        raw = raw & 0xFFFFFFFF
    return raw

def extract_weights(model):
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

def export_binary(layers, output_path):
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

# =========================================================
# Training
# =========================================================
def train_segment(segment_idx, epochs=5000, lr=1e-4, hidden=32,
                  n_samples=200000, batch_size=100000, device='cuda'):
    frames = load_segment_frames(segment_idx)
    n_frames = len(frames)

    model = SIREN(in_features=3, hidden_features=hidden, out_features=3,
                  hidden_layers=1, omega_0=10.0, omega_hidden=10.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.01)
    loss_fn = nn.MSELoss()

    best_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        coords, values = make_training_coords(frames, n_samples)
        coords_t = torch.from_numpy(coords).to(device)
        values_t = torch.from_numpy(values).to(device)

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

        if (epoch + 1) % 500 == 0 or epoch == 0:
            print(f"  seg {segment_idx:3d} epoch {epoch+1:5d}/{epochs}: loss={avg_loss:.6f} best={best_loss:.6f}")

    model.load_state_dict(best_state)
    model.cpu()
    return model, best_loss

# =========================================================
# Evaluate
# =========================================================
def evaluate_segment(model, frames, device='cuda'):
    n_frames, H, W = frames.shape
    model.eval()
    model.to(device)
    psnrs = []
    with torch.no_grad():
        for fi in range(n_frames):
            t_val = (fi / max(n_frames - 1, 1)) * 2.0 - 1.0
            x = np.linspace(-1, 1, W, dtype=np.float32)
            y = np.linspace(-ASPECT_Y, ASPECT_Y, H, dtype=np.float32)
            yy, xx = np.meshgrid(y, x, indexing='ij')
            tt = np.full_like(xx, t_val)
            coords = torch.from_numpy(np.stack([xx.ravel(), yy.ravel(), tt.ravel()], axis=1)).to(device)
            pred = model(coords).cpu().numpy()
            pred_gray = np.clip(pred[:, 0].reshape(H, W), -1.0, 1.0)
            mse = float(np.mean((pred_gray - frames[fi]) ** 2))
            psnrs.append(10 * np.log10(4.0 / mse) if mse > 0 else 100.0)
    model.cpu()
    return float(np.mean(psnrs))

# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=N_SEGMENTS)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--samples', type=int, default=200000)
    parser.add_argument('--batch-size', type=int, default=100000)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    t_total = time.time()
    results = []

    for seg in range(args.start, args.end):
        pt_path = WEIGHTS_DIR / f"segment_{seg:02d}.pt"
        bin_path = WEIGHTS_DIR / f"segment_{seg:02d}.bin"

        # Skip if already trained
        if pt_path.exists() and bin_path.exists():
            continue

        t0 = time.time()
        try:
            model, best_loss = train_segment(seg, args.epochs, args.lr, args.hidden,
                                              args.samples, args.batch_size, device)
        except FileNotFoundError as e:
            print(f"  seg {seg}: {e}, stopping")
            break

        # Save
        torch.save(model.state_dict(), pt_path)
        layers = extract_weights(model)
        export_binary(layers, bin_path)

        # Evaluate
        frames = load_segment_frames(seg)
        psnr = evaluate_segment(model, frames, device)
        elapsed = time.time() - t0

        results.append((seg, psnr, best_loss, elapsed))
        done = len([f for f in WEIGHTS_DIR.glob('segment_*.pt')])
        print(f"  seg {seg:3d}: PSNR={psnr:.1f}dB loss={best_loss:.6f} "
              f"[{elapsed:.0f}s] ({done}/{N_SEGMENTS} done)")

    total_time = time.time() - t_total
    print(f"\nDone. {len(results)} segments in {total_time:.0f}s "
          f"({total_time/max(len(results),1):.1f}s/seg)")

    if results:
        psnrs = [r[1] for r in results]
        print(f"PSNR: avg={np.mean(psnrs):.1f}dB min={np.min(psnrs):.1f}dB max={np.max(psnrs):.1f}dB")

if __name__ == '__main__':
    main()
