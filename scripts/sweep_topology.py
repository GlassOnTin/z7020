#!/usr/bin/env python3
"""sweep_topology.py — Find optimal SIREN hidden width for Bad Apple on FPGA.

Trains a representative subset of segments at multiple hidden widths and
reports PSNR, storage, and estimated FPGA FPS for each configuration.

Usage (in Colab after extracting frames):
    !python sweep_topology.py
    !python sweep_topology.py --epochs 3000 --test-segs 20
"""

import argparse, math, os, struct, sys, time
import numpy as np
import torch
from pathlib import Path
from PIL import Image

# =========================================================
# Constants
# =========================================================
FRAME_DIR = Path('frames')
TOTAL_FRAMES = 6575
FRAME_W, FRAME_H = 320, 172
ASPECT_Y = FRAME_H / FRAME_W
FRAMES_PER_SEGMENT = 10
N_SEGMENTS = (TOTAL_FRAMES + FRAMES_PER_SEGMENT - 1) // FRAMES_PER_SEGMENT
OMEGA_0 = 10.0

# Hidden widths to test
HIDDEN_SIZES = [8, 12, 16, 20, 24, 32]

# FPGA parameters for FPS estimation
FPGA_CLOCK = 50e6   # 50 MHz
FPGA_CORES = 18
PIXELS = FRAME_W * FRAME_H  # 55040
OVERHEAD_CYCLES = 50  # per-pixel overhead (sine LUT, control, etc.)


def load_all_frames():
    """Load all frames, return list of (n_frames, H, W) float32 arrays per segment."""
    all_frames = []
    for seg in range(N_SEGMENTS):
        start_frame = seg * FRAMES_PER_SEGMENT + 1
        end_frame = min(start_frame + FRAMES_PER_SEGMENT, TOTAL_FRAMES + 1)
        frames = []
        for i in range(start_frame, end_frame):
            path = FRAME_DIR / f"frame_{i:05d}.png"
            if not path.exists():
                break
            arr = np.array(Image.open(path).convert('L'), dtype=np.float32) / 127.5 - 1.0
            frames.append(arr)
        if not frames:
            break
        while len(frames) < FRAMES_PER_SEGMENT:
            frames.append(frames[-1])
        all_frames.append(np.stack(frames))
    return all_frames


def select_test_segments(all_frames, n_test):
    """Select evenly-spaced test segments spanning the full video."""
    n_total = len(all_frames)
    if n_test >= n_total:
        return list(range(n_total))
    step = n_total / n_test
    return [int(i * step) for i in range(n_test)]


def sample_segments(all_frames, seg_indices, samples_per_seg):
    """Sample training data for selected segments."""
    n_seg = len(seg_indices)
    H, W = FRAME_H, FRAME_W

    fi = np.random.randint(0, FRAMES_PER_SEGMENT, (n_seg, samples_per_seg))
    yi = np.random.randint(0, H, (n_seg, samples_per_seg))
    xi = np.random.randint(0, W, (n_seg, samples_per_seg))

    x = (xi / (W - 1)) * 2.0 - 1.0
    y = (yi / (H - 1)) * 2.0 * ASPECT_Y - ASPECT_Y
    t = (fi / max(FRAMES_PER_SEGMENT - 1, 1)) * 2.0 - 1.0

    coords = np.stack([x, y, t], axis=2).astype(np.float32)

    targets = np.empty((n_seg, samples_per_seg), dtype=np.float32)
    for i, seg in enumerate(seg_indices):
        targets[i] = all_frames[seg][fi[i], yi[i], xi[i]]

    targets_3ch = np.stack([targets, targets, targets], axis=2)
    return coords, targets_3ch


def init_weights(n_seg, hidden, device):
    """Initialize SIREN weights."""
    W1 = torch.empty(n_seg, hidden, 3, device=device)
    W1.uniform_(-1.0 / 3, 1.0 / 3)
    b1 = torch.zeros(n_seg, 1, hidden, device=device)

    bound2 = math.sqrt(6.0 / hidden) / OMEGA_0
    W2 = torch.empty(n_seg, hidden, hidden, device=device)
    W2.uniform_(-bound2, bound2)
    b2 = torch.zeros(n_seg, 1, hidden, device=device)

    bound3 = math.sqrt(6.0 / hidden) / OMEGA_0
    W3 = torch.empty(n_seg, 3, hidden, device=device)
    W3.uniform_(-bound3, bound3)
    b3 = torch.zeros(n_seg, 1, 3, device=device)

    params = [W1, b1, W2, b2, W3, b3]
    for p in params:
        p.requires_grad_(True)
    return params


def forward(coords, params):
    W1, b1, W2, b2, W3, b3 = params
    h = torch.bmm(coords, W1.transpose(1, 2)) + b1
    h = torch.sin(OMEGA_0 * h)
    h = torch.bmm(h, W2.transpose(1, 2)) + b2
    h = torch.sin(OMEGA_0 * h)
    out = torch.bmm(h, W3.transpose(1, 2)) + b3
    return torch.sin(out)


def evaluate_psnr(params, all_frames, seg_indices, device):
    """Compute average PSNR across test segments (all frames)."""
    W1, b1, W2, b2, W3, b3 = params
    n_seg = len(seg_indices)
    H, W = FRAME_H, FRAME_W

    x = np.linspace(-1, 1, W, dtype=np.float32)
    y = np.linspace(-ASPECT_Y, ASPECT_Y, H, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing='ij')
    xy_flat = np.stack([xx.ravel(), yy.ravel()], axis=1)
    n_pixels = H * W

    all_psnrs = []
    for fi in range(FRAMES_PER_SEGMENT):
        t_val = (fi / max(FRAMES_PER_SEGMENT - 1, 1)) * 2.0 - 1.0
        tt = np.full((n_pixels, 1), t_val, dtype=np.float32)
        coords_np = np.concatenate([xy_flat, tt], axis=1)

        coords = torch.from_numpy(coords_np).to(device)
        coords = coords.unsqueeze(0).expand(n_seg, -1, -1)

        with torch.no_grad():
            pred = forward(coords, params)

        pred_gray = pred[:, :, 0].cpu().numpy().reshape(n_seg, H, W)
        pred_gray = np.clip(pred_gray, -1.0, 1.0)

        for i, seg in enumerate(seg_indices):
            gt = all_frames[seg][fi]
            mse = float(np.mean((pred_gray[i] - gt) ** 2))
            psnr = 10 * np.log10(4.0 / mse) if mse > 0 else 100.0
            all_psnrs.append(psnr)

    return float(np.mean(all_psnrs)), float(np.min(all_psnrs))


def calc_params(hidden):
    return 3 * hidden + hidden + hidden * hidden + hidden + hidden * 3 + 3


def calc_fpga_fps(hidden):
    macs = hidden * hidden + 6 * hidden
    cycles = macs + OVERHEAD_CYCLES
    pixels_per_sec = FPGA_CLOCK * FPGA_CORES / cycles
    return pixels_per_sec / PIXELS


def train_topology(hidden, all_frames, seg_indices, device, epochs, samples,
                   mini_batch, lr):
    """Train one topology configuration on test segments."""
    n_seg = len(seg_indices)
    params = init_weights(n_seg, hidden, device)

    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01)

    best_loss = torch.full((n_seg,), float('inf'), device=device)
    best_params = [p.detach().clone() for p in params]

    n_mini = max(1, samples // mini_batch)
    t0 = time.time()

    for epoch in range(epochs):
        coords_np, targets_np = sample_segments(all_frames, seg_indices, samples)
        coords_all = torch.from_numpy(coords_np).to(device)
        targets_all = torch.from_numpy(targets_np).to(device)

        epoch_loss = 0.0
        perm = torch.randperm(samples, device=device)

        for mb in range(n_mini):
            start = mb * mini_batch
            end = min(start + mini_batch, samples)
            idx = perm[start:end]

            batch_coords = coords_all[:, idx]
            batch_targets = targets_all[:, idx]

            pred = forward(batch_coords, params)
            loss = ((pred - batch_targets) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        # Track best
        if (epoch + 1) % 100 == 0 or epoch == 0:
            with torch.no_grad():
                seg_mse = ((pred - batch_targets) ** 2).mean(dim=(1, 2))
                improved = seg_mse < best_loss
                if improved.any():
                    for i, p in enumerate(params):
                        best_params[i][improved] = p[improved].detach()
                    best_loss[improved] = seg_mse[improved]

        if (epoch + 1) % 500 == 0:
            elapsed = time.time() - t0
            avg_loss = epoch_loss / n_mini
            print(f"    H={hidden:2d} epoch {epoch+1:5d}/{epochs}: "
                  f"loss={avg_loss:.6f} [{elapsed:.0f}s]")

    # Restore best and evaluate
    for i, p in enumerate(params):
        p.data.copy_(best_params[i])

    train_time = time.time() - t0
    avg_psnr, min_psnr = evaluate_psnr(params, all_frames, seg_indices, device)

    return avg_psnr, min_psnr, train_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3000,
                        help='Epochs per topology (default: 3000)')
    parser.add_argument('--test-segs', type=int, default=30,
                        help='Number of test segments (evenly spaced)')
    parser.add_argument('--samples', type=int, default=50000)
    parser.add_argument('--mini-batch', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden-sizes', type=str, default=None,
                        help='Comma-separated hidden sizes (default: 8,12,16,20,24,32)')
    args = parser.parse_args()

    hidden_sizes = HIDDEN_SIZES
    if args.hidden_sizes:
        hidden_sizes = [int(x) for x in args.hidden_sizes.split(',')]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\nLoading frames...")
    t0 = time.time()
    all_frames = load_all_frames()
    n_total = len(all_frames)
    print(f"Loaded {n_total} segments in {time.time()-t0:.1f}s")

    seg_indices = select_test_segments(all_frames, args.test_segs)
    n_test = len(seg_indices)
    print(f"Testing on {n_test} segments: {seg_indices[:5]}...{seg_indices[-3:]}")
    print(f"Training: {args.epochs} epochs, {args.samples} samples/seg")

    # Header
    print(f"\n{'='*80}")
    print(f"{'H':>4} {'Params':>7} {'Bytes/seg':>10} {'Total(658)':>10} "
          f"{'PSNR avg':>9} {'PSNR min':>9} {'FPS':>6} {'Time':>7}")
    print(f"{'='*80}")

    results = []
    for hidden in hidden_sizes:
        n_params = calc_params(hidden)
        bytes_per_seg = n_params * 4
        total_bytes = bytes_per_seg * 658
        fps = calc_fpga_fps(hidden)

        print(f"\n  Training H={hidden} ({n_params} params, "
              f"{total_bytes/1024:.0f} KB total, ~{fps:.0f} FPS on FPGA)...")

        torch.manual_seed(42)  # reproducible comparison
        avg_psnr, min_psnr, train_time = train_topology(
            hidden, all_frames, seg_indices, device,
            args.epochs, args.samples, args.mini_batch, args.lr)

        total_kb = total_bytes / 1024
        print(f"  >> H={hidden:2d}: {n_params:5d} params, {bytes_per_seg:6d} B/seg, "
              f"{total_kb:7.0f} KB total, "
              f"PSNR={avg_psnr:.1f}/{min_psnr:.1f} dB, "
              f"~{fps:.0f} FPS, {train_time:.0f}s")

        results.append({
            'hidden': hidden, 'params': n_params,
            'bytes_per_seg': bytes_per_seg, 'total_kb': total_kb,
            'avg_psnr': avg_psnr, 'min_psnr': min_psnr,
            'fps': fps, 'train_time': train_time,
        })

    # Summary table
    print(f"\n\n{'='*80}")
    print(f"TOPOLOGY SWEEP RESULTS ({n_test} segments, {args.epochs} epochs)")
    print(f"{'='*80}")
    print(f"{'H':>4} {'Params':>7} {'Bytes/seg':>10} {'Total(658)':>10} "
          f"{'PSNR avg':>9} {'PSNR min':>9} {'FPS':>6}")
    print(f"{'-'*80}")
    for r in results:
        marker = ' <-- ' if r['avg_psnr'] > 18 and r['fps'] >= 15 else ''
        print(f"{r['hidden']:4d} {r['params']:7d} {r['bytes_per_seg']:10d} "
              f"{r['total_kb']:8.0f} KB "
              f"{r['avg_psnr']:9.1f} {r['min_psnr']:9.1f} {r['fps']:6.0f}{marker}")

    # Find sweet spot: best PSNR with FPS >= 15 and total < 2MB
    viable = [r for r in results if r['fps'] >= 10]
    if viable:
        best = max(viable, key=lambda r: r['avg_psnr'])
        print(f"\nRecommended: H={best['hidden']} — "
              f"{best['avg_psnr']:.1f} dB avg PSNR, "
              f"{best['total_kb']/1024:.1f} MB total, "
              f"~{best['fps']:.0f} FPS")


if __name__ == '__main__':
    main()
