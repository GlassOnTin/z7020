#!/usr/bin/env python3
"""train_colab_batched.py — Train ALL Bad Apple SIREN segments simultaneously.

Instead of 658 sequential tiny trainings (GPU-starved), this batches all 658
networks into large tensor operations using torch.bmm. The GPU gets one massive
matmul per layer instead of 658 tiny ones.

Expected speedup: ~100x+ over sequential training.
Expected runtime: ~10-30 minutes for all 658 segments on T4 GPU.

Usage (in Colab after extracting frames):
    !python train_colab_batched.py --epochs 5000
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
WEIGHTS_DIR = Path('weights')
TOTAL_FRAMES = 6575
FRAME_W, FRAME_H = 320, 172
ASPECT_Y = FRAME_H / FRAME_W  # 0.5375
FRAMES_PER_SEGMENT = 10
N_SEGMENTS = (TOTAL_FRAMES + FRAMES_PER_SEGMENT - 1) // FRAMES_PER_SEGMENT
FRAC_BITS = 28
Q_SCALE = 1 << FRAC_BITS
HIDDEN = 16
OMEGA_0 = 10.0

# =========================================================
# Data loading — all frames into memory
# =========================================================
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
        # Pad short segments (last segment) to FRAMES_PER_SEGMENT
        while len(frames) < FRAMES_PER_SEGMENT:
            frames.append(frames[-1])
        all_frames.append(np.stack(frames))
    return all_frames


def sample_all_segments(all_frames, samples_per_seg):
    """Sample training coords and targets for all segments at once.

    Returns numpy arrays: coords (N_SEG, S, 3), targets (N_SEG, S, 3)
    """
    n_seg = len(all_frames)
    H, W = FRAME_H, FRAME_W

    # Vectorized: generate all random indices at once
    fi = np.random.randint(0, FRAMES_PER_SEGMENT, (n_seg, samples_per_seg))
    yi = np.random.randint(0, H, (n_seg, samples_per_seg))
    xi = np.random.randint(0, W, (n_seg, samples_per_seg))

    x = (xi / (W - 1)) * 2.0 - 1.0
    y = (yi / (H - 1)) * 2.0 * ASPECT_Y - ASPECT_Y
    t = (fi / max(FRAMES_PER_SEGMENT - 1, 1)) * 2.0 - 1.0

    coords = np.stack([x, y, t], axis=2).astype(np.float32)  # (N, S, 3)

    # Gather pixel intensities per segment
    targets = np.empty((n_seg, samples_per_seg), dtype=np.float32)
    for seg in range(n_seg):
        targets[seg] = all_frames[seg][fi[seg], yi[seg], xi[seg]]

    # Replicate grayscale to 3 channels (matching SIREN output)
    targets_3ch = np.stack([targets, targets, targets], axis=2)  # (N, S, 3)
    return coords, targets_3ch


# =========================================================
# Batched SIREN — all segments as one tensor operation
# =========================================================
def init_weights(n_seg, hidden, omega_0, device):
    """Initialize batched SIREN weights for all segments.

    Returns list of parameter tensors: [W1, b1, W2, b2, W3, b3]
    """
    # Layer 1: 3 → hidden (first SIREN layer)
    W1 = torch.empty(n_seg, hidden, 3, device=device)
    W1.uniform_(-1.0 / 3, 1.0 / 3)
    b1 = torch.zeros(n_seg, 1, hidden, device=device)

    # Layer 2: hidden → hidden
    bound2 = math.sqrt(6.0 / hidden) / omega_0
    W2 = torch.empty(n_seg, hidden, hidden, device=device)
    W2.uniform_(-bound2, bound2)
    b2 = torch.zeros(n_seg, 1, hidden, device=device)

    # Output: hidden → 3
    bound3 = math.sqrt(6.0 / hidden) / omega_0
    W3 = torch.empty(n_seg, 3, hidden, device=device)
    W3.uniform_(-bound3, bound3)
    b3 = torch.zeros(n_seg, 1, 3, device=device)

    params = [W1, b1, W2, b2, W3, b3]
    for p in params:
        p.requires_grad_(True)
    return params


def batched_forward(coords, params, omega_0):
    """Forward pass for all segments simultaneously.

    coords: (N_SEG, BATCH, 3)
    Returns: (N_SEG, BATCH, 3)
    """
    W1, b1, W2, b2, W3, b3 = params

    # Layer 1: (N, B, 3) @ (N, 3, H) + bias → sin(omega * ...)
    h = torch.bmm(coords, W1.transpose(1, 2)) + b1
    h = torch.sin(omega_0 * h)

    # Layer 2: (N, B, H) @ (N, H, H) + bias → sin(omega * ...)
    h = torch.bmm(h, W2.transpose(1, 2)) + b2
    h = torch.sin(omega_0 * h)

    # Output: (N, B, H) @ (N, H, 3) + bias → sin(...)
    out = torch.bmm(h, W3.transpose(1, 2)) + b3
    return torch.sin(out)


# =========================================================
# Q4.28 export (matching FPGA format)
# =========================================================
def float_to_q428(val):
    clamped = max(-8.0, min(val, 8.0 - 1.0 / Q_SCALE))
    raw = int(round(clamped * Q_SCALE))
    if raw < 0:
        raw = raw & 0xFFFFFFFF
    return raw


def export_segment_binary(W1, b1, W2, b2, W3, b3, seg_idx, omega_0):
    """Export one segment's weights to .bin file in FPGA format."""
    # Bake omega into weights (FPGA expects pre-baked)
    layers = [
        (W1[seg_idx].detach().cpu().numpy() * omega_0,
         b1[seg_idx, 0].detach().cpu().numpy() * omega_0),
        (W2[seg_idx].detach().cpu().numpy() * omega_0,
         b2[seg_idx, 0].detach().cpu().numpy() * omega_0),
        (W3[seg_idx].detach().cpu().numpy(),
         b3[seg_idx, 0].detach().cpu().numpy()),
    ]

    all_vals = []
    for weights, biases in layers:
        for j in range(weights.shape[0]):
            for k in range(weights.shape[1]):
                all_vals.append(float_to_q428(weights[j, k]))
        for j in range(biases.shape[0]):
            all_vals.append(float_to_q428(biases[j]))

    bin_path = WEIGHTS_DIR / f"segment_{seg_idx:03d}.bin"
    with open(bin_path, 'wb') as f:
        for val in all_vals:
            f.write(struct.pack('<I', val))
    return bin_path


def export_segment_pt(params, seg_idx, omega_0):
    """Export one segment's weights as a .pt state_dict (matching SIREN class)."""
    W1, b1, W2, b2, W3, b3 = params
    state_dict = {
        'layers.0.linear.weight': W1[seg_idx].detach().cpu(),
        'layers.0.linear.bias': b1[seg_idx, 0].detach().cpu(),
        'layers.1.linear.weight': W2[seg_idx].detach().cpu(),
        'layers.1.linear.bias': b2[seg_idx, 0].detach().cpu(),
        'output_layer.weight': W3[seg_idx].detach().cpu(),
        'output_layer.bias': b3[seg_idx, 0].detach().cpu(),
    }
    pt_path = WEIGHTS_DIR / f"segment_{seg_idx:03d}.pt"
    torch.save(state_dict, pt_path)
    return pt_path


# =========================================================
# Evaluation
# =========================================================
def evaluate_psnr(params, all_frames, omega_0, device, max_segs=None):
    """Compute PSNR for each segment (full frame rendering)."""
    W1, b1, W2, b2, W3, b3 = params
    n_seg = len(all_frames) if max_segs is None else min(max_segs, len(all_frames))
    H, W = FRAME_H, FRAME_W

    # Pre-compute coordinate grid
    x = np.linspace(-1, 1, W, dtype=np.float32)
    y = np.linspace(-ASPECT_Y, ASPECT_Y, H, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing='ij')
    xy_flat = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (H*W, 2)
    n_pixels = H * W

    psnrs = []
    # Process in chunks to avoid OOM
    chunk = 32
    for c_start in range(0, n_seg, chunk):
        c_end = min(c_start + chunk, n_seg)
        c_size = c_end - c_start
        seg_psnrs = []

        for fi in range(FRAMES_PER_SEGMENT):
            t_val = (fi / max(FRAMES_PER_SEGMENT - 1, 1)) * 2.0 - 1.0
            tt = np.full((n_pixels, 1), t_val, dtype=np.float32)
            coords_np = np.concatenate([
                np.tile(xy_flat, (1, 1)),  # (H*W, 2)
                tt
            ], axis=1)  # (H*W, 3)

            # Replicate for all segments in chunk
            coords = torch.from_numpy(coords_np).to(device)
            coords = coords.unsqueeze(0).expand(c_size, -1, -1)  # (chunk, H*W, 3)

            with torch.no_grad():
                p = [W1[c_start:c_end], b1[c_start:c_end],
                     W2[c_start:c_end], b2[c_start:c_end],
                     W3[c_start:c_end], b3[c_start:c_end]]
                pred = batched_forward(coords, p, omega_0)  # (chunk, H*W, 3)

            pred_gray = pred[:, :, 0].cpu().numpy().reshape(c_size, H, W)
            pred_gray = np.clip(pred_gray, -1.0, 1.0)

            for s in range(c_size):
                seg = c_start + s
                gt = all_frames[seg][fi]
                mse = float(np.mean((pred_gray[s] - gt) ** 2))
                seg_psnrs.append((seg, 10 * np.log10(4.0 / mse) if mse > 0 else 100.0))

        # Average PSNR across frames for each segment in chunk
        for seg in range(c_start, c_end):
            seg_vals = [p for s, p in seg_psnrs if s == seg]
            psnrs.append((seg, float(np.mean(seg_vals))))

    return psnrs


# =========================================================
# Main training loop
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--samples', type=int, default=50000,
                        help='Samples per segment per epoch')
    parser.add_argument('--mini-batch', type=int, default=10000,
                        help='Samples per segment per mini-batch')
    parser.add_argument('--eval-every', type=int, default=500,
                        help='Evaluate PSNR every N epochs')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load all frames
    print(f"Loading {TOTAL_FRAMES} frames...")
    t0 = time.time()
    all_frames = load_all_frames()
    n_seg = len(all_frames)
    print(f"Loaded {n_seg} segments ({len(all_frames)} × {FRAMES_PER_SEGMENT} frames) "
          f"in {time.time()-t0:.1f}s")

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize batched weights
    print(f"Initializing {n_seg} SIREN networks (3→{HIDDEN}→{HIDDEN}→3)...")
    params = init_weights(n_seg, HIDDEN, OMEGA_0, device)
    W1, b1, W2, b2, W3, b3 = params

    optimizer = torch.optim.Adam(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # Track best loss per segment
    best_loss = torch.full((n_seg,), float('inf'), device=device)
    best_params = [p.detach().clone() for p in params]

    n_mini = max(1, args.samples // args.mini_batch)
    print(f"\nTraining: {args.epochs} epochs, {args.samples} samples/seg/epoch, "
          f"{n_mini} mini-batches of {args.mini_batch}")
    print(f"Tensor shapes: coords ({n_seg}, {args.mini_batch}, 3), "
          f"W1 ({n_seg}, {HIDDEN}, 3), W2 ({n_seg}, {HIDDEN}, {HIDDEN})")
    print()

    t_train = time.time()
    for epoch in range(args.epochs):
        # Sample new training data each epoch
        coords_np, targets_np = sample_all_segments(all_frames, args.samples)
        coords_all = torch.from_numpy(coords_np).to(device)
        targets_all = torch.from_numpy(targets_np).to(device)

        epoch_loss = 0.0
        perm = torch.randperm(args.samples, device=device)

        for mb in range(n_mini):
            start = mb * args.mini_batch
            end = min(start + args.mini_batch, args.samples)
            idx = perm[start:end]

            batch_coords = coords_all[:, idx]    # (N_SEG, MB, 3)
            batch_targets = targets_all[:, idx]   # (N_SEG, MB, 3)

            pred = batched_forward(batch_coords, params, OMEGA_0)
            loss = ((pred - batch_targets) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / n_mini

        # Track per-segment best (compute per-segment loss)
        if (epoch + 1) % 100 == 0 or epoch == 0:
            with torch.no_grad():
                # Quick per-segment loss check on last mini-batch
                seg_mse = ((pred - batch_targets) ** 2).mean(dim=(1, 2))  # (N_SEG,)
                improved = seg_mse < best_loss
                if improved.any():
                    for i, p in enumerate(params):
                        best_params[i][improved] = p[improved].detach()
                    best_loss[improved] = seg_mse[improved]

        if (epoch + 1) % args.eval_every == 0 or epoch == 0:
            elapsed = time.time() - t_train
            remaining = elapsed / (epoch + 1) * (args.epochs - epoch - 1)
            print(f"epoch {epoch+1:5d}/{args.epochs}: loss={avg_loss:.6f} "
                  f"lr={scheduler.get_last_lr()[0]:.2e} "
                  f"[{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining]")

            # Quick PSNR on first 5 segments
            if (epoch + 1) % (args.eval_every * 2) == 0:
                psnrs = evaluate_psnr(params, all_frames, OMEGA_0, device, max_segs=5)
                avg_psnr = np.mean([p for _, p in psnrs])
                print(f"  PSNR (first 5 segs): {avg_psnr:.1f} dB")

    total_time = time.time() - t_train
    print(f"\nTraining done in {total_time:.0f}s ({total_time/60:.1f} min)")

    # Restore best weights
    for i, p in enumerate(params):
        p.data.copy_(best_params[i])

    # Full PSNR evaluation
    print("\nEvaluating all segments...")
    t_eval = time.time()
    psnrs = evaluate_psnr(params, all_frames, OMEGA_0, device)
    eval_time = time.time() - t_eval
    all_psnr = [p for _, p in psnrs]
    print(f"PSNR: avg={np.mean(all_psnr):.1f}dB "
          f"min={np.min(all_psnr):.1f}dB max={np.max(all_psnr):.1f}dB "
          f"[{eval_time:.0f}s]")

    # Export all segments
    print(f"\nExporting {n_seg} segments...")
    t_export = time.time()
    for seg in range(n_seg):
        export_segment_pt(params, seg, OMEGA_0)
        export_segment_binary(W1, b1, W2, b2, W3, b3, seg, OMEGA_0)
    print(f"Exported in {time.time()-t_export:.1f}s")
    print(f"  .pt files: {WEIGHTS_DIR}/segment_*.pt")
    print(f"  .bin files: {WEIGHTS_DIR}/segment_*.bin")

    # Summary
    print(f"\n{'='*60}")
    print(f"Total: {n_seg} segments, {args.epochs} epochs")
    print(f"Time: {total_time:.0f}s training + {eval_time:.0f}s eval")
    print(f"PSNR: {np.mean(all_psnr):.1f} dB average")
    print(f"Output: {WEIGHTS_DIR}/")


if __name__ == '__main__':
    main()
