#!/usr/bin/env python3
"""Test: threshold SIREN output for sharper Bad Apple reconstruction.
Also test longer training and different omega values."""

import math, time
import numpy as np
import torch, torch.nn as nn
from PIL import Image
from pathlib import Path

FRAME_DIR = Path(__file__).parent.parent / "bad_apple" / "frames"
OUT_DIR = Path(__file__).parent.parent / "bad_apple" / "capacity_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)
W, H = 320, 172

class SineLayer(nn.Module):
    def __init__(self, inf, outf, omega=30.0, first=False):
        super().__init__()
        self.omega = omega
        self.linear = nn.Linear(inf, outf)
        with torch.no_grad():
            if first:
                self.linear.weight.uniform_(-1/inf, 1/inf)
            else:
                b = math.sqrt(6/inf)/omega
                self.linear.weight.uniform_(-b, b)
    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))

class SIREN(nn.Module):
    def __init__(self, inf=2, hid=32, outf=1, w0=30, wh=30):
        super().__init__()
        self.net = nn.Sequential(
            SineLayer(inf, hid, omega=w0, first=True),
            SineLayer(hid, hid, omega=wh),
            nn.Linear(hid, outf),
        )
        with torch.no_grad():
            b = math.sqrt(6/hid)/wh
            self.net[-1].weight.uniform_(-b, b)
    def forward(self, x):
        return torch.sin(self.net(x))

def load_frame(idx):
    path = FRAME_DIR / f"frame_{idx:05d}.png"
    return np.array(Image.open(path).convert('L'), dtype=np.float32)/127.5 - 1.0

def make_grid():
    x = np.linspace(-1, 1, W, dtype=np.float32)
    y = np.linspace(-1, 1, H, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing='ij')
    return np.stack([xx.ravel(), yy.ravel()], axis=1)

def psnr_binary(gt, pred, threshold=0.0):
    """PSNR after thresholding pred to binary."""
    pred_bin = np.where(pred > threshold, 1.0, -1.0)
    mse = np.mean((pred_bin - gt)**2)
    return 10 * np.log10(4.0/mse) if mse > 0 else 100, pred_bin

def accuracy_binary(gt, pred, threshold=0.0):
    """Pixel accuracy for binary classification."""
    gt_bin = gt > 0
    pred_bin = pred > threshold
    return np.mean(gt_bin == pred_bin)

frame = load_frame(3000)
coords = torch.from_numpy(make_grid())
target = torch.from_numpy(frame.ravel()[:, None])

# --- Test 1: Long training (10K epochs) with hid=32 ---
print("=== Long training, hid=32, omega=(30,30) ===")
model = SIREN(inf=2, hid=32, w0=30, wh=30)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 10000, eta_min=1e-6)
t0 = time.time()
for ep in range(10000):
    pred = model(coords)
    loss = nn.functional.mse_loss(pred, target)
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if (ep+1) % 2000 == 0:
        print(f"  ep {ep+1}: loss={loss.item():.6f}")

model.eval()
with torch.no_grad():
    out = model(coords).numpy().reshape(H, W)
    out = np.clip(out, -1, 1)

mse = np.mean((out - frame)**2)
psnr_raw = 10*np.log10(4/mse)
psnr_thr, out_bin = psnr_binary(frame, out)
acc = accuracy_binary(frame, out)
print(f"  Raw PSNR: {psnr_raw:.1f} dB")
print(f"  Thresholded PSNR: {psnr_thr:.1f} dB")
print(f"  Binary accuracy: {acc*100:.1f}%")
print(f"  Time: {time.time()-t0:.0f}s")

# Save comparison: GT | raw | thresholded
gt_u8 = ((frame+1)*127.5).clip(0,255).astype(np.uint8)
raw_u8 = ((out+1)*127.5).clip(0,255).astype(np.uint8)
bin_u8 = ((out_bin+1)*127.5).clip(0,255).astype(np.uint8)
comp = np.hstack([gt_u8, raw_u8, bin_u8])
Image.fromarray(comp).save(OUT_DIR / "long_train_h32_threshold.png")

# --- Test 2: Train with binary loss (BCE) ---
print("\n=== BCE loss, hid=32, omega=(30,30) ===")
model2 = SIREN(inf=2, hid=32, w0=30, wh=30)
opt2 = torch.optim.Adam(model2.parameters(), lr=1e-4)
sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, 10000, eta_min=1e-6)
# Target: convert [-1,+1] to [0,1] for BCE
target_01 = (target + 1) / 2
t0 = time.time()
for ep in range(10000):
    pred = model2(coords)
    pred_01 = (pred + 1) / 2  # sin output [-1,1] -> [0,1]
    pred_01 = pred_01.clamp(1e-6, 1-1e-6)
    loss = nn.functional.binary_cross_entropy(pred_01, target_01)
    opt2.zero_grad(); loss.backward(); opt2.step(); sched2.step()
    if (ep+1) % 2000 == 0:
        print(f"  ep {ep+1}: loss={loss.item():.6f}")

model2.eval()
with torch.no_grad():
    out2 = model2(coords).numpy().reshape(H, W)
    out2 = np.clip(out2, -1, 1)

psnr_thr2, out_bin2 = psnr_binary(frame, out2)
acc2 = accuracy_binary(frame, out2)
print(f"  Thresholded PSNR: {psnr_thr2:.1f} dB")
print(f"  Binary accuracy: {acc2*100:.1f}%")
print(f"  Time: {time.time()-t0:.0f}s")

gt_u8 = ((frame+1)*127.5).clip(0,255).astype(np.uint8)
raw_u8 = ((out2+1)*127.5).clip(0,255).astype(np.uint8)
bin_u8 = ((out_bin2+1)*127.5).clip(0,255).astype(np.uint8)
comp2 = np.hstack([gt_u8, raw_u8, bin_u8])
Image.fromarray(comp2).save(OUT_DIR / "bce_h32_threshold.png")

# --- Test 3: 10 frames, threshold, long training ---
print("\n=== 10 frames, hid=32, long training, threshold ===")
frames10 = np.stack([load_frame(2996+i) for i in range(10)])
nf = 10

# Build full grid for 10 frames (3D)
class SIREN3(nn.Module):
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

model3 = SIREN3(hid=32)
opt3 = torch.optim.Adam(model3.parameters(), lr=1e-4)
sched3 = torch.optim.lr_scheduler.CosineAnnealingLR(opt3, 10000, eta_min=1e-6)

# Full coords for all frames
x_lin = np.linspace(-1, 1, W, dtype=np.float32)
y_lin = np.linspace(-1, 1, H, dtype=np.float32)
t_lin = np.linspace(-1, 1, nf, dtype=np.float32)
tt, yy, xx = np.meshgrid(t_lin, y_lin, x_lin, indexing='ij')
all_coords = np.stack([xx.ravel(), yy.ravel(), tt.ravel()], axis=1)
all_vals = frames10.ravel()[:, None]
all_coords_t = torch.from_numpy(all_coords)
all_vals_t = torch.from_numpy(all_vals)
n_total = all_coords.shape[0]

t0 = time.time()
for ep in range(10000):
    idx = torch.randint(0, n_total, (80000,))
    pred = model3(all_coords_t[idx])
    loss = nn.functional.mse_loss(pred, all_vals_t[idx])
    opt3.zero_grad(); loss.backward(); opt3.step(); sched3.step()
    if (ep+1) % 2000 == 0:
        print(f"  ep {ep+1}: loss={loss.item():.6f}")

model3.eval()
# Evaluate per-frame
grid2d = make_grid()
accs = []
for fi in range(nf):
    t_val = (fi / max(nf-1, 1)) * 2 - 1
    t_col = np.full((grid2d.shape[0], 1), t_val, dtype=np.float32)
    c3d = np.hstack([grid2d, t_col])
    with torch.no_grad():
        pred_f = model3(torch.from_numpy(c3d)).numpy().reshape(H, W)
    pred_f = np.clip(pred_f, -1, 1)
    acc_f = accuracy_binary(frames10[fi], pred_f)
    accs.append(acc_f)
    if fi == 0 or fi == 5 or fi == 9:
        gt_u8 = ((frames10[fi]+1)*127.5).clip(0,255).astype(np.uint8)
        raw_u8 = ((pred_f+1)*127.5).clip(0,255).astype(np.uint8)
        bin_u8 = ((np.where(pred_f > 0, 1, -1)+1)*127.5).clip(0,255).astype(np.uint8)
        comp = np.hstack([gt_u8, raw_u8, bin_u8])
        Image.fromarray(comp).save(OUT_DIR / f"10f_long_f{fi:02d}.png")

print(f"  Avg binary accuracy: {np.mean(accs)*100:.1f}%")
print(f"  Per-frame accuracy: {[f'{a*100:.1f}%' for a in accs]}")
print(f"  Time: {time.time()-t0:.0f}s")

print("\nDone.")
