#!/usr/bin/env python3
"""Test SIREN quality across different Bad Apple scenes.
10-frame segments, 10K epochs, threshold evaluation."""

import math, time, sys
import numpy as np
import torch, torch.nn as nn
from PIL import Image
from pathlib import Path

FRAME_DIR = Path(__file__).parent.parent / "bad_apple" / "frames"
OUT_DIR = Path(__file__).parent.parent / "bad_apple" / "scene_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)
W, H = 320, 172

class SineLayer(nn.Module):
    def __init__(self, inf, outf, omega=30.0, first=False):
        super().__init__()
        self.omega = omega
        self.linear = nn.Linear(inf, outf)
        with torch.no_grad():
            if first: self.linear.weight.uniform_(-1/inf, 1/inf)
            else:
                b = math.sqrt(6/inf)/omega
                self.linear.weight.uniform_(-b, b)
    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))

class SIREN(nn.Module):
    def __init__(self, hid=32, w0=30, wh=30):
        super().__init__()
        self.net = nn.Sequential(
            SineLayer(3, hid, omega=w0, first=True),
            SineLayer(hid, hid, omega=wh),
            nn.Linear(hid, 1),
        )
        with torch.no_grad():
            b = math.sqrt(6/hid)/wh
            self.net[-1].weight.uniform_(-b, b)
    def forward(self, x):
        return torch.sin(self.net(x))

def load_frames(start, count):
    frames = []
    for i in range(start, start + count):
        path = FRAME_DIR / f"frame_{i:05d}.png"
        if not path.exists(): break
        frames.append(np.array(Image.open(path).convert('L'), dtype=np.float32)/127.5 - 1.0)
    return np.stack(frames)

def make_grid2d():
    x = np.linspace(-1, 1, W, dtype=np.float32)
    y = np.linspace(-1, 1, H, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing='ij')
    return np.stack([xx.ravel(), yy.ravel()], axis=1)

def train_and_eval(frames, label, epochs=10000, hid=32):
    nf = len(frames)
    n_pixels = nf * H * W

    # Build coords
    x_lin = np.linspace(-1, 1, W, dtype=np.float32)
    y_lin = np.linspace(-1, 1, H, dtype=np.float32)
    t_lin = np.linspace(-1, 1, nf, dtype=np.float32)
    tt, yy, xx = np.meshgrid(t_lin, y_lin, x_lin, indexing='ij')
    all_coords = torch.from_numpy(np.stack([xx.ravel(), yy.ravel(), tt.ravel()], axis=1))
    all_vals = torch.from_numpy(frames.ravel()[:, None])

    model = SIREN(hid=hid)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=1e-6)

    t0 = time.time()
    for ep in range(epochs):
        idx = torch.randint(0, n_pixels, (80000,))
        pred = model(all_coords[idx])
        loss = nn.functional.mse_loss(pred, all_vals[idx])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

    elapsed = time.time() - t0

    # Evaluate
    model.eval()
    grid2d = make_grid2d()
    accs = []
    psnrs = []
    for fi in range(nf):
        t_val = (fi / max(nf-1, 1)) * 2 - 1
        t_col = np.full((grid2d.shape[0], 1), t_val, dtype=np.float32)
        c3d = np.hstack([grid2d, t_col])
        with torch.no_grad():
            pred_f = model(torch.from_numpy(c3d)).numpy().reshape(H, W)
        pred_f = np.clip(pred_f, -1, 1)

        gt_bin = frames[fi] > 0
        pred_bin = pred_f > 0
        acc = np.mean(gt_bin == pred_bin)
        accs.append(acc)

        mse = np.mean((np.where(pred_f > 0, 1.0, -1.0) - frames[fi])**2)
        psnr = 10*np.log10(4.0/mse) if mse > 0 else 100
        psnrs.append(psnr)

        # Save first and middle frames
        if fi == 0 or fi == nf//2:
            gt_u8 = ((frames[fi]+1)*127.5).clip(0,255).astype(np.uint8)
            thr_u8 = (np.where(pred_f > 0, 255, 0)).astype(np.uint8)
            comp = np.hstack([gt_u8, thr_u8])
            Image.fromarray(comp).save(OUT_DIR / f"{label}_f{fi:02d}.png")

    avg_acc = np.mean(accs)
    min_acc = np.min(accs)
    avg_psnr = np.mean(psnrs)

    print(f"  {label}: acc={avg_acc*100:.1f}% (min {min_acc*100:.1f}%), "
          f"PSNR={avg_psnr:.1f}dB, {elapsed:.0f}s")
    return avg_acc, min_acc, avg_psnr

# Test segments across the video
# Pick representative scenes: intro, silhouette, high detail, transition, end
test_scenes = [
    (1, "intro"),           # Opening — mostly black/text
    (301, "early_scene"),   # Early imagery
    (1001, "mid_slow"),     # Mid-video, slower scene
    (2001, "buildup"),      # Building up
    (2801, "silhouettes"),  # Iconic silhouette section
    (3501, "complex"),      # More complex imagery
    (4001, "fast_action"),  # Fast action
    (5001, "balloons"),     # Girl with balloons
    (5501, "late"),         # Late video
    (6001, "climax"),       # Near end
]

print(f"Testing 10-frame segments across Bad Apple ({len(test_scenes)} scenes)")
print(f"10K epochs each, hid=32, threshold evaluation\n")

results = []
for start_frame, name in test_scenes:
    try:
        frames = load_frames(start_frame, 10)
        if len(frames) < 10:
            print(f"  {name}: only {len(frames)} frames, skipping")
            continue
        acc, min_acc, psnr = train_and_eval(frames, name)
        results.append((name, start_frame, acc, min_acc, psnr))
    except Exception as e:
        print(f"  {name}: ERROR {e}")

print(f"\n{'='*60}")
print(f"{'Scene':<15} {'Start':>6} {'Avg Acc':>8} {'Min Acc':>8} {'PSNR':>7}")
print(f"{'-'*15} {'-'*6} {'-'*8} {'-'*8} {'-'*7}")
for name, start, acc, min_acc, psnr in results:
    print(f"{name:<15} {start:>6} {acc*100:>7.1f}% {min_acc*100:>7.1f}% {psnr:>6.1f}")
print(f"{'='*60}")

avg_acc_all = np.mean([r[2] for r in results])
min_acc_all = min(r[3] for r in results)
print(f"Overall avg accuracy: {avg_acc_all*100:.1f}%")
print(f"Worst single-frame accuracy: {min_acc_all*100:.1f}%")
