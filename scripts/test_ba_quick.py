#!/usr/bin/env python3
"""Quick scene test: 5K epochs, 5 diverse scenes, prints results with flushing."""
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
                b = math.sqrt(6/inf)/omega; self.linear.weight.uniform_(-b, b)
    def forward(self, x): return torch.sin(self.omega * self.linear(x))

class SIREN(nn.Module):
    def __init__(self, hid=32):
        super().__init__()
        self.net = nn.Sequential(
            SineLayer(3, hid, omega=30.0, first=True),
            SineLayer(hid, hid, omega=30.0),
            nn.Linear(hid, 1),
        )
        with torch.no_grad():
            b = math.sqrt(6/hid)/30.0; self.net[-1].weight.uniform_(-b, b)
    def forward(self, x): return torch.sin(self.net(x))

def load_frames(start, count):
    frames = []
    for i in range(start, start+count):
        p = FRAME_DIR / f"frame_{i:05d}.png"
        if not p.exists(): break
        frames.append(np.array(Image.open(p).convert('L'), dtype=np.float32)/127.5-1)
    return np.stack(frames)

def test_scene(start, name, epochs=5000, nf=10):
    frames = load_frames(start, nf)
    nf = len(frames)
    x = np.linspace(-1,1,W,dtype=np.float32)
    y = np.linspace(-1,1,H,dtype=np.float32)
    t = np.linspace(-1,1,nf,dtype=np.float32)
    tt,yy,xx = np.meshgrid(t,y,x,indexing='ij')
    coords = torch.from_numpy(np.stack([xx.ravel(),yy.ravel(),tt.ravel()],axis=1))
    vals = torch.from_numpy(frames.ravel()[:,None])
    n = coords.shape[0]

    model = SIREN(hid=32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=1e-6)
    t0 = time.time()
    for ep in range(epochs):
        idx = torch.randint(0,n,(80000,))
        loss = nn.functional.mse_loss(model(coords[idx]), vals[idx])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    elapsed = time.time()-t0

    model.eval()
    grid = np.stack([np.meshgrid(np.linspace(-1,1,W,dtype=np.float32),
                                  np.linspace(-1,1,H,dtype=np.float32),indexing='ij')[1].ravel(),
                     np.meshgrid(np.linspace(-1,1,W,dtype=np.float32),
                                  np.linspace(-1,1,H,dtype=np.float32),indexing='ij')[0].ravel()],axis=1)
    # Fix: proper grid
    xg = np.linspace(-1,1,W,dtype=np.float32)
    yg = np.linspace(-1,1,H,dtype=np.float32)
    yyg, xxg = np.meshgrid(yg, xg, indexing='ij')
    grid2d = np.stack([xxg.ravel(), yyg.ravel()], axis=1)

    accs = []
    for fi in range(nf):
        tv = (fi/max(nf-1,1))*2-1
        tc = np.full((grid2d.shape[0],1), tv, dtype=np.float32)
        c3d = np.hstack([grid2d, tc])
        with torch.no_grad():
            pred = model(torch.from_numpy(c3d)).numpy().reshape(H,W)
        pred = np.clip(pred,-1,1)
        acc = np.mean((frames[fi]>0)==(pred>0))
        accs.append(acc)
        if fi==0:
            gt = ((frames[fi]+1)*127.5).clip(0,255).astype(np.uint8)
            pr = np.where(pred>0,255,0).astype(np.uint8)
            Image.fromarray(np.hstack([gt,pr])).save(OUT_DIR/f"{name}_f00.png")

    avg = np.mean(accs)
    mn = np.min(accs)
    print(f"  {name:15s} frames {start:5d}: avg={avg*100:.1f}% min={mn*100:.1f}% ({elapsed:.0f}s)",flush=True)
    return avg, mn

scenes = [
    (1, "intro"),
    (1001, "mid_slow"),
    (2801, "silhouettes"),
    (4001, "fast_action"),
    (5001, "balloons"),
    (6001, "climax"),
]

print(f"Testing {len(scenes)} scenes, 10 frames each, 5K epochs\n", flush=True)
results = []
for start, name in scenes:
    try:
        a, m = test_scene(start, name)
        results.append((name, start, a, m))
    except Exception as e:
        print(f"  {name}: ERROR {e}", flush=True)

print(f"\n{'='*50}", flush=True)
avg_all = np.mean([r[2] for r in results])
min_all = min(r[3] for r in results)
print(f"Overall: avg={avg_all*100:.1f}%, worst frame={min_all*100:.1f}%", flush=True)
