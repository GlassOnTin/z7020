#!/usr/bin/env python3
"""Quick capacity test: how well can a 1283-param SIREN represent a SINGLE Bad Apple frame?
And then small batches of frames (5, 10, 20)?"""

import math
import time
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path

FRAME_DIR = Path(__file__).parent.parent / "bad_apple" / "frames"
OUT_DIR = Path(__file__).parent.parent / "bad_apple" / "capacity_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FRAME_W, FRAME_H = 320, 172

class SineLayer(nn.Module):
    def __init__(self, in_f, out_f, omega=30.0, first=False):
        super().__init__()
        self.omega = omega
        self.linear = nn.Linear(in_f, out_f)
        with torch.no_grad():
            if first:
                self.linear.weight.uniform_(-1/in_f, 1/in_f)
            else:
                b = math.sqrt(6/in_f) / omega
                self.linear.weight.uniform_(-b, b)
    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))

class SIREN(nn.Module):
    def __init__(self, in_f=3, hid=32, out_f=1, omega0=30.0, omegah=30.0):
        super().__init__()
        self.net = nn.Sequential(
            SineLayer(in_f, hid, omega=omega0, first=True),
            SineLayer(hid, hid, omega=omegah),
            nn.Linear(hid, out_f),
        )
        with torch.no_grad():
            b = math.sqrt(6/hid) / omegah
            self.net[-1].weight.uniform_(-b, b)
    def forward(self, x):
        return torch.sin(self.net(x))

def load_frame(idx):
    """Load frame as float [-1,+1]."""
    path = FRAME_DIR / f"frame_{idx:05d}.png"
    return np.array(Image.open(path).convert('L'), dtype=np.float32) / 127.5 - 1.0

def make_grid_2d():
    x = np.linspace(-1, 1, FRAME_W, dtype=np.float32)
    y = np.linspace(-1, 1, FRAME_H, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing='ij')
    return np.stack([xx.ravel(), yy.ravel()], axis=1)

def make_grid_3d(n_frames):
    x = np.linspace(-1, 1, FRAME_W, dtype=np.float32)
    y = np.linspace(-1, 1, FRAME_H, dtype=np.float32)
    t = np.linspace(-1, 1, n_frames, dtype=np.float32)
    tt, yy, xx = np.meshgrid(t, y, x, indexing='ij')
    return np.stack([xx.ravel(), yy.ravel(), tt.ravel()], axis=1)

def train_and_eval(frames, label, epochs=5000, omega0=30.0, omegah=30.0, hid=32):
    """Train SIREN on frames, return PSNR."""
    n_frames = len(frames)
    is_single = (n_frames == 1)
    in_dim = 2 if is_single else 3
    n_params = in_dim * hid + hid + hid * hid + hid + hid * 1 + 1

    print(f"\n--- {label}: {n_frames} frame(s), {n_params} params, "
          f"omega0={omega0}, omegah={omegah}, hid={hid} ---")

    model = SIREN(in_f=in_dim, hid=hid, out_f=1, omega0=omega0, omegah=omegah)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-6)

    # Build training data
    n_total = n_frames * FRAME_H * FRAME_W
    if is_single:
        coords = make_grid_2d()
        vals = frames[0].ravel()[:, None]
    else:
        coords = make_grid_3d(n_frames)
        vals = frames.ravel()[:, None]

    coords_t = torch.from_numpy(coords)
    vals_t = torch.from_numpy(vals)

    # For large datasets, subsample during training
    use_subsample = n_total > 100000

    t0 = time.time()
    for ep in range(epochs):
        if use_subsample:
            idx = torch.randint(0, n_total, (80000,))
            pred = model(coords_t[idx])
            loss = nn.functional.mse_loss(pred, vals_t[idx])
        else:
            pred = model(coords_t)
            loss = nn.functional.mse_loss(pred, vals_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (ep+1) % 1000 == 0:
            print(f"  ep {ep+1}: loss={loss.item():.6f}")

    elapsed = time.time() - t0
    print(f"  Trained in {elapsed:.1f}s")

    # Full eval
    model.eval()
    with torch.no_grad():
        if is_single:
            pred_all = model(coords_t).numpy().reshape(FRAME_H, FRAME_W)
            pred_all = np.clip(pred_all, -1, 1)
            mse = np.mean((pred_all - frames[0])**2)
            psnr = 10 * np.log10(4.0 / mse) if mse > 0 else 100

            # Save comparison
            gt_u8 = ((frames[0]+1)*127.5).clip(0,255).astype(np.uint8)
            pr_u8 = ((pred_all+1)*127.5).clip(0,255).astype(np.uint8)
            comp = np.hstack([gt_u8, pr_u8])
            Image.fromarray(comp).save(OUT_DIR / f"{label}.png")
        else:
            mse_sum = 0
            for fi in range(n_frames):
                t_val = (fi / max(n_frames-1, 1)) * 2 - 1
                grid2d = make_grid_2d()
                t_col = np.full((grid2d.shape[0], 1), t_val, dtype=np.float32)
                c3d = np.hstack([grid2d, t_col])
                pred_f = model(torch.from_numpy(c3d)).numpy().reshape(FRAME_H, FRAME_W)
                pred_f = np.clip(pred_f, -1, 1)
                mse_sum += np.mean((pred_f - frames[fi])**2)

                if fi == 0 or fi == n_frames//2 or fi == n_frames-1:
                    gt_u8 = ((frames[fi]+1)*127.5).clip(0,255).astype(np.uint8)
                    pr_u8 = ((pred_f+1)*127.5).clip(0,255).astype(np.uint8)
                    comp = np.hstack([gt_u8, pr_u8])
                    Image.fromarray(comp).save(OUT_DIR / f"{label}_f{fi:03d}.png")

            mse = mse_sum / n_frames
            psnr = 10 * np.log10(4.0 / mse) if mse > 0 else 100

    raw_bytes = n_frames * FRAME_H * FRAME_W
    ratio = raw_bytes / (n_params * 4)
    print(f"  PSNR: {psnr:.1f} dB, MSE: {mse:.6f}, Compression: {ratio:.0f}:1")
    return psnr, mse, ratio

# Test frames: 3000 (interesting silhouette)
frame_3000 = load_frame(3000)

# === Single frame tests ===
# Baseline: single frame, standard omega
train_and_eval(frame_3000[None], "1frame_w30_h32", epochs=3000, omega0=30, omegah=30, hid=32)

# Higher omega for sharper edges
train_and_eval(frame_3000[None], "1frame_w60_h32", epochs=3000, omega0=60, omegah=60, hid=32)

# Wider network
train_and_eval(frame_3000[None], "1frame_w30_h64", epochs=3000, omega0=30, omegah=30, hid=64)

# === Multi-frame tests (5 frames around 3000) ===
frames_5 = np.stack([load_frame(2998+i) for i in range(5)])
train_and_eval(frames_5, "5frames_w30_h32", epochs=3000, omega0=30, omegah=30, hid=32)

# 10 frames
frames_10 = np.stack([load_frame(2996+i) for i in range(10)])
train_and_eval(frames_10, "10frames_w30_h32", epochs=3000, omega0=30, omegah=30, hid=32)

# 20 frames
frames_20 = np.stack([load_frame(2991+i) for i in range(20)])
train_and_eval(frames_20, "20frames_w30_h32", epochs=3000, omega0=30, omegah=30, hid=32)

print("\nDone.")
