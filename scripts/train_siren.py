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


# =========================================================
# Physics simulation targets
# =========================================================

def _solve_shallow_water(nx=128, ny=128, nt=2000, g=2.0):
    """Solve 2D shallow water equations via Lax-Friedrichs.

    Domain: [-1, 1] x [-1, 1], reflective boundaries.
    Initial condition: two raised bumps (like two droplets).
    Returns (h, u, v) arrays of shape (nx, ny, n_snapshots).
    """
    dx = 2.0 / (nx - 1)
    dy = 2.0 / (ny - 1)

    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    # Initial condition: two tall displaced bumps for dramatic waves
    h_rest = 1.0
    h = h_rest * np.ones((nx, ny))
    r1 = np.sqrt((xx - 0.35)**2 + (yy - 0.2)**2)
    r2 = np.sqrt((xx + 0.3)**2 + (yy + 0.35)**2)
    h += 0.8 * np.exp(-30 * r1**2)
    h += 0.6 * np.exp(-25 * r2**2)

    hu = np.zeros((nx, ny))
    hv = np.zeros((nx, ny))

    # CFL-limited time step
    c_max = np.sqrt(g * h.max())
    dt = 0.25 * min(dx, dy) / c_max

    n_snap = 64  # number of snapshots to save
    save_every = max(1, nt // n_snap)
    snaps_h = []
    snaps_u = []
    snaps_v = []

    for step in range(nt):
        if step % save_every == 0:
            u_vel = np.where(h > 0.01, hu / h, 0.0)
            v_vel = np.where(h > 0.01, hv / h, 0.0)
            snaps_h.append(h.copy())
            snaps_u.append(u_vel.copy())
            snaps_v.append(v_vel.copy())

        # Fluxes in x-direction: F = [hu, hu^2/h + g*h^2/2, hu*hv/h]
        f0 = hu
        f1 = np.where(h > 0.01, hu**2 / h + 0.5 * g * h**2, 0.5 * g * h**2)
        f2 = np.where(h > 0.01, hu * hv / h, 0.0)

        # Fluxes in y-direction: G = [hv, hu*hv/h, hv^2/h + g*h^2/2]
        g0 = hv
        g1 = np.where(h > 0.01, hu * hv / h, 0.0)
        g2 = np.where(h > 0.01, hv**2 / h + 0.5 * g * h**2, 0.5 * g * h**2)

        # Lax-Friedrichs: average neighbors, subtract flux differences
        # Interior points only; boundaries handled separately
        h_avg  = 0.25 * (h[2:,1:-1]  + h[:-2,1:-1]  + h[1:-1,2:]  + h[1:-1,:-2])
        hu_avg = 0.25 * (hu[2:,1:-1] + hu[:-2,1:-1] + hu[1:-1,2:] + hu[1:-1,:-2])
        hv_avg = 0.25 * (hv[2:,1:-1] + hv[:-2,1:-1] + hv[1:-1,2:] + hv[1:-1,:-2])

        h_new  = h_avg  - dt/(2*dx) * (f0[2:,1:-1] - f0[:-2,1:-1]) \
                        - dt/(2*dy) * (g0[1:-1,2:] - g0[1:-1,:-2])
        hu_new = hu_avg - dt/(2*dx) * (f1[2:,1:-1] - f1[:-2,1:-1]) \
                        - dt/(2*dy) * (g1[1:-1,2:] - g1[1:-1,:-2])
        hv_new = hv_avg - dt/(2*dx) * (f2[2:,1:-1] - f2[:-2,1:-1]) \
                        - dt/(2*dy) * (g2[1:-1,2:] - g2[1:-1,:-2])

        h[1:-1,1:-1]  = h_new
        hu[1:-1,1:-1] = hu_new
        hv[1:-1,1:-1] = hv_new

        # Reflective boundaries
        h[0,:] = h[1,:];   h[-1,:] = h[-2,:]
        h[:,0] = h[:,1];   h[:,-1] = h[:,-2]
        hu[0,:] = -hu[1,:];  hu[-1,:] = -hu[-2,:]  # reflect velocity
        hu[:,0] = hu[:,1];    hu[:,-1] = hu[:,-2]
        hv[0,:] = hv[1,:];   hv[-1,:] = hv[-2,:]
        hv[:,0] = -hv[:,1];   hv[:,-1] = -hv[:,-2]

        # Positivity fix
        h = np.maximum(h, 0.01)

        # Damping (gentle, prevents energy buildup from reflections)
        hu *= 0.9998
        hv *= 0.9998

    return np.stack(snaps_h, axis=-1), np.stack(snaps_u, axis=-1), np.stack(snaps_v, axis=-1)


def _solve_gray_scott(nx=128, ny=128, nt=5000, f=0.039, k=0.058,
                      Du=0.16, Dv=0.08):
    """Solve Gray-Scott reaction-diffusion system.

    dU/dt = Du*laplacian(U) - U*V^2 + f*(1-U)
    dV/dt = Dv*laplacian(V) + U*V^2 - (f+k)*V

    Parameters (f=0.039, k=0.058) produce worm/stripe patterns.
    Returns (U, V) arrays of shape (nx, ny, n_snapshots).
    """
    dx = 2.0 / (nx - 1)
    dt = 0.5 * dx**2 / max(Du, Dv)  # diffusion-limited step
    dt = min(dt, 1.0)

    # Initial condition: uniform U=1 with seeded V perturbation
    U = np.ones((nx, ny))
    V = np.zeros((nx, ny))

    # Seed several small squares of V in the center
    cx, cy = nx // 2, ny // 2
    for ox, oy in [(-8, -5), (5, 8), (-3, 7), (6, -6), (0, 0)]:
        si, sj = cx + ox - 3, cy + oy - 3
        ei, ej = cx + ox + 3, cy + oy + 3
        si, sj = max(0, si), max(0, sj)
        ei, ej = min(nx, ei), min(ny, ej)
        U[si:ei, sj:ej] = 0.5
        V[si:ei, sj:ej] = 0.25

    # Add small noise to break symmetry
    rng = np.random.RandomState(42)
    U += rng.uniform(-0.01, 0.01, (nx, ny))
    V += rng.uniform(-0.01, 0.01, (nx, ny))

    n_snap = 64
    save_every = max(1, nt // n_snap)
    snaps_U = []
    snaps_V = []

    for step in range(nt):
        if step % save_every == 0:
            snaps_U.append(U.copy())
            snaps_V.append(V.copy())

        # Laplacian with periodic boundaries
        lap_U = (np.roll(U, 1, 0) + np.roll(U, -1, 0) +
                 np.roll(U, 1, 1) + np.roll(U, -1, 1) - 4*U) / dx**2
        lap_V = (np.roll(V, 1, 0) + np.roll(V, -1, 0) +
                 np.roll(V, 1, 1) + np.roll(V, -1, 1) - 4*V) / dx**2

        UVV = U * V * V
        U += dt * (Du * lap_U - UVV + f * (1 - U))
        V += dt * (Dv * lap_V + UVV - (f + k) * V)

        U = np.clip(U, 0, 1)
        V = np.clip(V, 0, 1)

    return np.stack(snaps_U, axis=-1), np.stack(snaps_V, axis=-1)


def _interp_field(field_3d, x_query, y_query, t_query, t_max):
    """Trilinear interpolation from a (nx, ny, nt) grid.

    x_query, y_query in [-1, 1], t_query in [0, t_max].
    Returns interpolated values at query points.
    """
    nx, ny, nt = field_3d.shape

    # Map query coordinates to grid indices
    xi = (x_query + 1) / 2 * (nx - 1)  # [-1,1] → [0, nx-1]
    yi = (y_query + 1) / 2 * (ny - 1)
    ti = t_query / t_max * (nt - 1)     # [0, t_max] → [0, nt-1]

    # Clamp
    xi = np.clip(xi, 0, nx - 1.001)
    yi = np.clip(yi, 0, ny - 1.001)
    ti = np.clip(ti, 0, nt - 1.001)

    # Integer and fractional parts
    x0 = np.floor(xi).astype(int); x1 = np.minimum(x0 + 1, nx - 1); xf = xi - x0
    y0 = np.floor(yi).astype(int); y1 = np.minimum(y0 + 1, ny - 1); yf = yi - y0
    t0 = np.floor(ti).astype(int); t1 = np.minimum(t0 + 1, nt - 1); tf = ti - t0

    # Trilinear interpolation
    c000 = field_3d[x0, y0, t0]; c100 = field_3d[x1, y0, t0]
    c010 = field_3d[x0, y1, t0]; c110 = field_3d[x1, y1, t0]
    c001 = field_3d[x0, y0, t1]; c101 = field_3d[x1, y0, t1]
    c011 = field_3d[x0, y1, t1]; c111 = field_3d[x1, y1, t1]

    c00 = c000 * (1-xf) + c100 * xf
    c01 = c001 * (1-xf) + c101 * xf
    c10 = c010 * (1-xf) + c110 * xf
    c11 = c011 * (1-xf) + c111 * xf

    c0 = c00 * (1-yf) + c10 * yf
    c1 = c01 * (1-yf) + c11 * yf

    return c0 * (1-tf) + c1 * tf


def target_shallow_water(x, y, t):
    """Free-surface shallow water waves — two-bump dam break.

    Solves the 2D shallow water equations with reflective boundaries.
    The initial condition is two raised bumps that collapse and create
    expanding circular waves, wall reflections, and interference patterns.

    This has no closed-form solution — the wave interactions emerge from
    the PDE, making this a genuine use case for neural surrogates.
    """
    if not hasattr(target_shallow_water, '_cache'):
        print("    Solving shallow water equations (this takes a few seconds)...")
        target_shallow_water._cache = _solve_shallow_water()
        print("    Simulation complete.")

    snaps_h, snaps_u, snaps_v = target_shallow_water._cache
    t_max = 8.0  # matches FPGA time_val range [0, 8.0)  # match training time range

    h = _interp_field(snaps_h, x, y, t, t_max)
    u = _interp_field(snaps_u, x, y, t, t_max)
    v = _interp_field(snaps_v, x, y, t, t_max)

    # Velocity-based colormap: flow direction → hue, speed → saturation,
    # height → brightness. Makes wave fronts vivid even for small amplitudes.
    speed = np.sqrt(u**2 + v**2)
    s_norm = np.clip(speed * 5.0, 0, 1)  # amplify small velocities

    # Flow angle → hue (using atan2, mapping [-pi, pi] to [0, 1])
    angle = np.arctan2(v, u)  # [-pi, pi]
    hue = (angle + np.pi) / (2 * np.pi)  # [0, 1]

    # Height deviation for brightness
    h_dev = h - 1.0
    bright = np.clip(0.5 + h_dev * 3.0, 0.1, 1.0)

    # HSV-like: hue from velocity direction, saturation from speed, value from height
    # Convert HSV to RGB manually
    h6 = hue * 6.0
    sector = np.floor(h6).astype(int) % 6
    frac = h6 - np.floor(h6)
    p = bright * (1 - s_norm)
    q = bright * (1 - s_norm * frac)
    t_val = bright * (1 - s_norm * (1 - frac))

    r_out = np.where(sector == 0, bright,
            np.where(sector == 1, q,
            np.where(sector == 2, p,
            np.where(sector == 3, p,
            np.where(sector == 4, t_val, bright)))))
    g_out = np.where(sector == 0, t_val,
            np.where(sector == 1, bright,
            np.where(sector == 2, bright,
            np.where(sector == 3, q,
            np.where(sector == 4, p, p)))))
    b_out = np.where(sector == 0, p,
            np.where(sector == 1, p,
            np.where(sector == 2, t_val,
            np.where(sector == 3, bright,
            np.where(sector == 4, bright, q)))))

    # Map [0, 1] → [-1, +1] for SIREN output range
    r_out = r_out * 2.0 - 1.0
    g_out = g_out * 2.0 - 1.0
    b_out = b_out * 2.0 - 1.0

    return np.stack([r_out, g_out, b_out], axis=-1)


def target_gray_scott(x, y, t):
    """Gray-Scott reaction-diffusion — emergent Turing patterns.

    Solves the Gray-Scott system with parameters that produce
    worm/stripe patterns from small initial perturbations.
    The pattern evolves over time as the reaction-diffusion front
    propagates and self-organizes.

    No closed-form solution — the spatial patterns emerge from
    the interaction of local reaction and diffusion.
    """
    if not hasattr(target_gray_scott, '_cache'):
        print("    Solving Gray-Scott reaction-diffusion (this takes a few seconds)...")
        target_gray_scott._cache = _solve_gray_scott()
        print("    Simulation complete.")

    snaps_U, snaps_V = target_gray_scott._cache
    t_max = 8.0  # matches FPGA time_val range [0, 8.0)

    U = _interp_field(snaps_U, x, y, t, t_max)
    V = _interp_field(snaps_V, x, y, t, t_max)

    # Colormap: U is background chemical (high = empty), V is pattern chemical
    # Classic RD visualization: V concentration drives color
    v_norm = np.clip(V * 4.0, 0, 1)  # V typically [0, 0.4]
    u_norm = np.clip(U, 0, 1)

    # Organic colormap: dark background, colored pattern ridges
    r_out = np.clip(v_norm * 1.8 - 0.5, -1, 1)
    g_out = np.clip(v_norm * 1.2 + (1 - u_norm) * 0.5 - 0.5, -1, 1)
    b_out = np.clip((1 - u_norm) * 1.5 - 0.2, -1, 1)

    return np.stack([r_out, g_out, b_out], axis=-1)


def _solve_float_glass(nx=128, ny=128, nt=8000, mu_ratio=50.0):
    """Solve thin-film equation for viscous glass spreading on molten tin.

    Models the float glass process: molten glass poured from a spout
    spreads under gravity on a denser liquid (tin) bath.

    The governing equation (lubrication/thin-film approximation):
        dh/dt = (1/3mu) * div(h^3 * grad(h)) + S(x,y)

    where h is glass thickness, mu is the viscosity ratio (glass/tin),
    and S is a source term representing the continuous pour from the spout.

    The h^3 nonlinearity creates a sharp spreading front (finite speed of
    propagation) — unlike linear diffusion, the contact line moves at
    finite velocity. This is what makes float glass spreading so hard
    to simulate and why it has no closed-form solution.

    Returns h array of shape (nx, ny, n_snapshots).
    """
    dx = 2.0 / (nx - 1)
    dy = 2.0 / (ny - 1)
    coeff = 1.0 / (3.0 * mu_ratio)

    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    # Initial condition: thin film everywhere (tin surface), no glass yet
    h = np.full((nx, ny), 0.001)

    # Source: spout location — continuous pour at (-0.3, 0.2)
    spout_x, spout_y = -0.3, 0.2
    spout_r = np.sqrt((xx - spout_x)**2 + (yy - spout_y)**2)
    spout_mask = np.exp(-80 * spout_r**2)  # Gaussian spout profile
    source_rate = 0.15  # pour rate

    # Second spout (smaller, offset) — represents a lip drip
    spout2_x, spout2_y = 0.25, -0.15
    spout2_r = np.sqrt((xx - spout2_x)**2 + (yy - spout2_y)**2)
    spout2_mask = np.exp(-100 * spout2_r**2)
    source_rate2 = 0.08

    # Time step (stability: dt < dx^2 / (2 * coeff * h_max^3) in each dim)
    # Start conservative, will be ok for small h
    dt = 0.3 * dx**2 / (coeff * 4.0)  # assuming h_max ~ 1 initially

    n_snap = 64
    save_every = max(1, nt // n_snap)
    snaps_h = []

    for step in range(nt):
        if step % save_every == 0:
            snaps_h.append(h.copy())

        # Compute h^3 at cell edges (average neighboring cells)
        h3 = h**3

        # Diffusive flux: F = h^3 * dh/dx (x-direction)
        h3_edge_x = 0.5 * (h3[1:,:] + h3[:-1,:])
        dh_dx = (h[1:,:] - h[:-1,:]) / dx
        flux_x = h3_edge_x * dh_dx

        # Diffusive flux: G = h^3 * dh/dy (y-direction)
        h3_edge_y = 0.5 * (h3[:,1:] + h3[:,:-1])
        dh_dy = (h[:,1:] - h[:,:-1]) / dy
        flux_y = h3_edge_y * dh_dy

        # Divergence of flux
        div_flux = np.zeros_like(h)
        div_flux[1:-1,:] += (flux_x[1:,:] - flux_x[:-1,:]) / dx
        div_flux[:,1:-1] += (flux_y[:,1:] - flux_y[:,:-1]) / dy

        # Update
        h += dt * (coeff * div_flux + source_rate * spout_mask
                   + source_rate2 * spout2_mask)

        # Neumann (zero-flux) boundaries — glass doesn't leave the bath
        h[0,:] = h[1,:]; h[-1,:] = h[-2,:]
        h[:,0] = h[:,1]; h[:,-1] = h[:,-2]

        # Floor: can't have negative thickness
        h = np.maximum(h, 0.001)

        # Adaptive dt for stability as h grows
        h_max = h.max()
        if h_max > 0.1:
            dt_new = 0.2 * dx**2 / (coeff * max(h_max**3, 0.01))
            dt = min(dt, dt_new)

    return np.stack(snaps_h, axis=-1)


def target_float_glass(x, y, t):
    """Float glass process — viscous glass spreading on molten tin.

    Models the thin-film lubrication equation for a viscous gravity
    current spreading on a denser liquid. Two pour spouts deposit
    glass that spreads under gravity with a sharp contact line.

    The h^3 nonlinearity, moving contact line, and merging of separate
    glass patches make this a genuinely hard free-surface problem with
    no analytical solution.
    """
    if not hasattr(target_float_glass, '_cache'):
        print("    Solving float glass thin-film equation (this takes a moment)...")
        target_float_glass._cache = _solve_float_glass()
        print("    Simulation complete.")

    snaps_h = target_float_glass._cache
    t_max = 8.0  # matches FPGA time_val range [0, 8.0)

    h = _interp_field(snaps_h, x, y, t, t_max)

    # Colormap: molten glass appearance
    # Thick glass = bright amber/orange, thin = dark (tin surface visible)
    h_max_vis = 0.8  # saturation point for visualization
    h_norm = np.clip(h / h_max_vis, 0, 1)

    # Gradient magnitude for flow front visualization
    # Approximate using finite differences in query space
    # (we can't easily compute gradients of interpolated field,
    #  so use h value directly for a simpler but effective colormap)

    # Amber/orange colormap for glass, dark blue-gray for tin
    r_out = np.clip(h_norm * 2.2 - 0.3, -1, 1)          # red: bright for thick glass
    g_out = np.clip(h_norm * 1.4 - 0.5, -1, 1)          # green: amber tint
    b_out = np.clip(0.15 - h_norm * 0.8, -1, 1)          # blue: tin surface shows through

    return np.stack([r_out, g_out, b_out], axis=-1)


def _solve_combustion(nx=128, ny=128, nt=6000):
    """Solve buoyant combustion (reactive Boussinesq) on a 2D domain.

    Coupled system:
        dT/dt + u·∇T = kT ∇²T + Q·R(T,Y)     (temperature)
        dY/dt + u·∇Y = kY ∇²Y - R(T,Y)         (fuel mass fraction)
        dw/dt + u·∇w = nu ∇²w + alpha·dT/dx2    (vorticity, buoyancy)
        ∇²psi = -w                                (stream function)
        u = dpsi/dy, v = -dpsi/dx                 (velocity)

    R(T,Y) = Da·Y·exp(-Ea/(T+T0)) is Arrhenius reaction rate.
    Buoyancy drives the plume upward (hot gas is lighter).

    Returns (T, Y, w) arrays of shape (nx, ny, n_snapshots).
    """
    dx = 2.0 / (nx - 1)
    dy = 2.0 / (ny - 1)

    # Physical parameters (non-dimensional, tuned for visual interest)
    kT = 0.006       # thermal diffusivity
    kY = 0.003       # fuel diffusivity
    nu = 0.005       # viscosity (higher for stability)
    alpha = 2.5      # buoyancy strength
    Q = 2.0          # heat release
    Da = 5.0         # Damköhler number (reaction speed)
    Ea = 2.5         # activation energy
    T0 = 1.0         # reference temperature offset

    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    # Initial fields
    T = np.zeros((nx, ny))       # temperature (ambient = 0)
    Y = np.zeros((nx, ny))       # fuel fraction (no fuel initially)
    w = np.zeros((nx, ny))       # vorticity
    psi = np.zeros((nx, ny))     # stream function

    # Fuel source: burner at bottom center
    burner_mask = np.exp(-20 * ((xx)**2 + (yy + 0.8)**2))
    # Small hot seed to ignite
    T += 0.5 * np.exp(-30 * (xx**2 + (yy + 0.7)**2))

    dt_diff = 0.1 * dx**2 / max(kT, kY, nu)  # diffusive CFL
    dt = dt_diff

    n_snap = 64
    save_every = max(1, nt // n_snap)
    snaps_T = []
    snaps_Y = []
    snaps_w = []

    rng = np.random.RandomState(42)

    for step in range(nt):
        if step % save_every == 0:
            snaps_T.append(T.copy())
            snaps_Y.append(Y.copy())
            snaps_w.append(w.copy())

        # Solve Poisson for stream function: ∇²psi = -w
        # Simple Jacobi iteration (20 iterations, good enough for visual quality)
        for _ in range(20):
            psi[1:-1,1:-1] = 0.25 * (
                psi[2:,1:-1] + psi[:-2,1:-1] +
                psi[1:-1,2:] + psi[1:-1,:-2] +
                dx**2 * w[1:-1,1:-1])
            # psi = 0 on all boundaries (no-flow)
            psi[0,:] = 0; psi[-1,:] = 0
            psi[:,0] = 0; psi[:,-1] = 0

        # Velocity from stream function
        u_vel = np.zeros_like(psi)
        v_vel = np.zeros_like(psi)
        u_vel[1:-1,1:-1] = (psi[1:-1,2:] - psi[1:-1,:-2]) / (2*dy)
        v_vel[1:-1,1:-1] = -(psi[2:,1:-1] - psi[:-2,1:-1]) / (2*dx)

        # Laplacians
        def lap(f):
            L = np.zeros_like(f)
            L[1:-1,1:-1] = (f[2:,1:-1] + f[:-2,1:-1] +
                            f[1:-1,2:] + f[1:-1,:-2] - 4*f[1:-1,1:-1]) / dx**2
            return L

        # Advection (upwind, 1st order for stability)
        def advect(f, u, v):
            A = np.zeros_like(f)
            # x-direction
            A[1:-1,1:-1] += np.where(u[1:-1,1:-1] > 0,
                u[1:-1,1:-1] * (f[1:-1,1:-1] - f[:-2,1:-1]) / dx,
                u[1:-1,1:-1] * (f[2:,1:-1] - f[1:-1,1:-1]) / dx)
            # y-direction
            A[1:-1,1:-1] += np.where(v[1:-1,1:-1] > 0,
                v[1:-1,1:-1] * (f[1:-1,1:-1] - f[1:-1,:-2]) / dy,
                v[1:-1,1:-1] * (f[1:-1,2:] - f[1:-1,1:-1]) / dy)
            return A

        # Reaction rate: Arrhenius
        R = Da * Y * np.exp(-Ea / (T + T0))
        R = np.clip(R, 0, 10)  # stability cap

        # Buoyancy: horizontal temperature gradient drives vorticity
        dTdx = np.zeros_like(T)
        dTdx[1:-1,1:-1] = (T[2:,1:-1] - T[:-2,1:-1]) / (2*dx)

        # Adaptive CFL: limit dt by max velocity
        u_max = max(np.abs(u_vel).max(), np.abs(v_vel).max(), 0.01)
        dt_adv = 0.3 * dx / u_max
        dt = min(dt_diff, dt_adv)

        # Time integration
        T += dt * (-advect(T, u_vel, v_vel) + kT * lap(T) + Q * R)
        Y += dt * (-advect(Y, u_vel, v_vel) + kY * lap(Y) - R)
        w += dt * (-advect(w, u_vel, v_vel) + nu * lap(w) + alpha * dTdx)

        # Fuel injection from burner
        Y += dt * 1.5 * burner_mask

        # Small perturbation for flame flickering
        if step % 80 == 0:
            T += 0.015 * rng.randn(nx, ny) * burner_mask

        # Clamp (aggressive — prevents blowup)
        T = np.clip(T, 0, 4)
        Y = np.clip(Y, 0, 1)
        w = np.clip(w, -20, 20)

        # NaN check — reset if blowup
        if np.any(np.isnan(T)):
            T = np.nan_to_num(T, nan=0.0)
            Y = np.nan_to_num(Y, nan=0.0)
            w = np.nan_to_num(w, nan=0.0)
            psi = np.nan_to_num(psi, nan=0.0)

        # Boundary conditions
        T[:, 0] = T[:, 1] * 0.3   # cool floor
        Y[:, 0] = Y[:, 1]
        w[:, 0] = w[:, 1]
        T[:,-1] = T[:,-2] * 0.8   # slight cooling at top
        Y[:,-1] = Y[:,-2]
        w[:,-1] = 0                # no vorticity at outflow
        T[0,:] = T[1,:]; T[-1,:] = T[-2,:]
        Y[0,:] = Y[1,:]; Y[-1,:] = Y[-2,:]
        w[0,:] = 0; w[-1,:] = 0

        # Damping (prevents energy accumulation)
        T *= 0.9997
        w *= 0.999

    return (np.stack(snaps_T, axis=-1),
            np.stack(snaps_Y, axis=-1),
            np.stack(snaps_w, axis=-1))


def target_combustion(x, y, t):
    """Buoyant combustion plume — reactive Boussinesq flow.

    Coupled temperature-fuel-vorticity system with Arrhenius kinetics.
    A burner at the bottom injects fuel which ignites, producing a
    buoyant plume that rises, rolls up into vortices, and flickers.

    The coupling between buoyancy, reaction, and vortex dynamics
    creates chaotic flame-like patterns with no analytical solution.
    """
    if not hasattr(target_combustion, '_cache'):
        print("    Solving buoyant combustion equations (this takes a moment)...")
        target_combustion._cache = _solve_combustion()
        print("    Simulation complete.")

    snaps_T, snaps_Y, snaps_w = target_combustion._cache
    t_max = 8.0  # matches FPGA time_val range [0, 8.0)

    T = _interp_field(snaps_T, x, y, t, t_max)
    Y = _interp_field(snaps_Y, x, y, t, t_max)

    # Fire colormap: black → red → orange → yellow → white
    T_norm = np.clip(T / 3.0, 0, 1)    # normalize temperature
    Y_norm = np.clip(Y * 3.0, 0, 1)    # fuel presence

    # Blackbody-like fire colors
    r_out = np.clip(T_norm * 3.0 - 0.5, -1, 1)                     # red: early onset
    g_out = np.clip(T_norm * 2.5 - 1.0, -1, 1)                     # green: mid temperature
    b_out = np.clip(T_norm * 2.0 - 1.5 + Y_norm * 0.4, -1, 1)     # blue: only very hot + fuel

    return np.stack([r_out, g_out, b_out], axis=-1)


# =========================================================
# Euler / vorticity evolution
# =========================================================

def _solve_euler_vorticity(nx=128, ny=128, nt=4000, nu=0.001):
    """Solve 2D incompressible Euler in vorticity-stream function form.

    dw/dt + u·nabla(w) = nu * laplacian(w)   (nearly inviscid)
    laplacian(psi) = -w
    u = dpsi/dy, v = -dpsi/dx

    IC: perturbed shear layer (Kelvin-Helmholtz instability).
    Returns w array of shape (nx, ny, n_snapshots).
    """
    dx = 2.0 / (nx - 1)
    dy = 2.0 / (ny - 1)

    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    # Shear layer: u = tanh(y/delta) with perturbation
    delta = 0.08
    w = -1.0 / (delta * np.cosh(yy / delta)**2)  # vorticity of tanh profile
    # Add sinusoidal perturbation to trigger KH roll-up
    w += 0.05 * np.sin(2 * np.pi * xx) * np.exp(-yy**2 / (4*delta**2))
    # Second mode for richer structure
    w += 0.03 * np.sin(4 * np.pi * xx + 0.7) * np.exp(-yy**2 / (4*delta**2))

    psi = np.zeros((nx, ny))

    dt_diff = 0.2 * dx**2 / max(nu, 1e-6)
    dt = min(dt_diff, 0.005)

    n_snap = 64
    save_every = max(1, nt // n_snap)
    snaps_w = []

    for step in range(nt):
        if step % save_every == 0:
            snaps_w.append(w.copy())

        # Poisson solve: laplacian(psi) = -w
        for _ in range(25):
            psi[1:-1,1:-1] = 0.25 * (
                psi[2:,1:-1] + psi[:-2,1:-1] +
                psi[1:-1,2:] + psi[1:-1,:-2] +
                dx**2 * w[1:-1,1:-1])
            psi[0,:] = 0; psi[-1,:] = 0
            psi[:,0] = 0; psi[:,-1] = 0

        # Velocity
        u_vel = np.zeros_like(psi)
        v_vel = np.zeros_like(psi)
        u_vel[1:-1,1:-1] = (psi[1:-1,2:] - psi[1:-1,:-2]) / (2*dy)
        v_vel[1:-1,1:-1] = -(psi[2:,1:-1] - psi[:-2,1:-1]) / (2*dx)

        # Laplacian of w
        Lw = np.zeros_like(w)
        Lw[1:-1,1:-1] = (w[2:,1:-1] + w[:-2,1:-1] +
                          w[1:-1,2:] + w[1:-1,:-2] - 4*w[1:-1,1:-1]) / dx**2

        # Upwind advection
        adv = np.zeros_like(w)
        adv[1:-1,1:-1] += np.where(u_vel[1:-1,1:-1] > 0,
            u_vel[1:-1,1:-1] * (w[1:-1,1:-1] - w[:-2,1:-1]) / dx,
            u_vel[1:-1,1:-1] * (w[2:,1:-1] - w[1:-1,1:-1]) / dx)
        adv[1:-1,1:-1] += np.where(v_vel[1:-1,1:-1] > 0,
            v_vel[1:-1,1:-1] * (w[1:-1,1:-1] - w[1:-1,:-2]) / dy,
            v_vel[1:-1,1:-1] * (w[1:-1,2:] - w[1:-1,1:-1]) / dy)

        # Adaptive CFL
        u_max = max(np.abs(u_vel).max(), np.abs(v_vel).max(), 0.01)
        dt_adv = 0.3 * dx / u_max
        dt = min(dt_diff, dt_adv, 0.005)

        w += dt * (-adv + nu * Lw)

        # Periodic in x, zero at y boundaries
        w[0,:] = w[-2,:]
        w[-1,:] = w[1,:]
        w[:,0] = 0; w[:,-1] = 0

        w = np.clip(w, -30, 30)
        if np.any(np.isnan(w)):
            w = np.nan_to_num(w, nan=0.0)
            psi = np.nan_to_num(psi, nan=0.0)

    return np.stack(snaps_w, axis=-1)


def target_euler_vorticity(x, y, t):
    """Kelvin-Helmholtz vortex roll-up — nearly inviscid Euler flow.

    Vorticity mapped to hue (cyclonic/anticyclonic = warm/cool),
    magnitude to brightness.
    """
    if not hasattr(target_euler_vorticity, '_cache'):
        print("    Solving Euler vorticity equations (KH instability)...")
        target_euler_vorticity._cache = _solve_euler_vorticity()
        print("    Simulation complete.")

    snaps_w = target_euler_vorticity._cache
    t_max = 8.0

    w = _interp_field(snaps_w, x, y, t, t_max)

    # Vorticity colormap: blue/cyan for negative (clockwise),
    # red/yellow for positive (counterclockwise), dark for zero
    w_max = 8.0
    w_norm = np.clip(w / w_max, -1, 1)  # [-1, 1]
    mag = np.abs(w_norm)

    # Diverging colormap: cool→dark→warm
    r_out = np.clip(w_norm * 2.0, -1, 1)                    # positive = red
    g_out = np.clip(mag * 1.5 - 0.3, -1, 1)                 # bright for strong
    b_out = np.clip(-w_norm * 2.0, -1, 1)                   # negative = blue

    return np.stack([r_out, g_out, b_out], axis=-1)


# =========================================================
# Lid-driven cavity
# =========================================================

def _solve_lid_cavity(nx=96, ny=96, nt=5000, Re_final=400):
    """Solve lid-driven cavity flow via vorticity-stream function.

    Top wall moves at u=1. Re ramps from 50 to Re_final over time
    to show transition from creeping flow to recirculating eddies.
    Returns (w, psi) arrays of shape (nx, ny, n_snapshots).
    """
    dx = 2.0 / (nx - 1)
    dy = 2.0 / (ny - 1)

    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)

    w = np.zeros((nx, ny))
    psi = np.zeros((nx, ny))

    n_snap = 64
    save_every = max(1, nt // n_snap)
    snaps_w = []
    snaps_psi = []

    for step in range(nt):
        if step % save_every == 0:
            snaps_w.append(w.copy())
            snaps_psi.append(psi.copy())

        # Ramp Reynolds number
        Re = 50 + (Re_final - 50) * min(1.0, step / (nt * 0.6))
        nu = 1.0 / Re

        dt_diff = 0.15 * dx**2 / nu
        dt = min(dt_diff, 0.003)

        # Poisson: laplacian(psi) = -w
        for _ in range(25):
            psi[1:-1,1:-1] = 0.25 * (
                psi[2:,1:-1] + psi[:-2,1:-1] +
                psi[1:-1,2:] + psi[1:-1,:-2] +
                dx**2 * w[1:-1,1:-1])
            # psi = 0 on all walls (no-slip)
            psi[0,:] = 0; psi[-1,:] = 0
            psi[:,0] = 0; psi[:,-1] = 0

        # Velocity
        u_vel = np.zeros_like(psi)
        v_vel = np.zeros_like(psi)
        u_vel[1:-1,1:-1] = (psi[1:-1,2:] - psi[1:-1,:-2]) / (2*dy)
        v_vel[1:-1,1:-1] = -(psi[2:,1:-1] - psi[:-2,1:-1]) / (2*dx)

        # Laplacian of w
        Lw = np.zeros_like(w)
        Lw[1:-1,1:-1] = (w[2:,1:-1] + w[:-2,1:-1] +
                          w[1:-1,2:] + w[1:-1,:-2] - 4*w[1:-1,1:-1]) / dx**2

        # Upwind advection
        adv = np.zeros_like(w)
        adv[1:-1,1:-1] += np.where(u_vel[1:-1,1:-1] > 0,
            u_vel[1:-1,1:-1] * (w[1:-1,1:-1] - w[:-2,1:-1]) / dx,
            u_vel[1:-1,1:-1] * (w[2:,1:-1] - w[1:-1,1:-1]) / dx)
        adv[1:-1,1:-1] += np.where(v_vel[1:-1,1:-1] > 0,
            v_vel[1:-1,1:-1] * (w[1:-1,1:-1] - w[1:-1,:-2]) / dy,
            v_vel[1:-1,1:-1] * (w[1:-1,2:] - w[1:-1,1:-1]) / dy)

        # Adaptive CFL
        u_max = max(np.abs(u_vel).max(), np.abs(v_vel).max(), 0.01)
        dt_adv = 0.3 * dx / u_max
        dt = min(dt_diff, dt_adv, 0.003)

        w += dt * (-adv + nu * Lw)

        # Wall vorticity BCs (Thom's formula)
        u_lid = 1.0
        w[:,-1] = -2 * (psi[:,-2] - 0) / dy**2 - 2 * u_lid / dy   # top (moving)
        w[:,0]  = -2 * (psi[:,1] - 0) / dy**2                        # bottom
        w[0,:]  = -2 * (psi[1,:] - 0) / dx**2                        # left
        w[-1,:] = -2 * (psi[-2,:] - 0) / dx**2                       # right

        w = np.clip(w, -50, 50)
        if np.any(np.isnan(w)):
            w = np.nan_to_num(w, nan=0.0)
            psi = np.nan_to_num(psi, nan=0.0)

    return np.stack(snaps_w, axis=-1), np.stack(snaps_psi, axis=-1)


def target_lid_cavity(x, y, t):
    """Lid-driven cavity flow with increasing Reynolds number.

    Stream function for flow visualization, vorticity for color intensity.
    Shows transition from single primary vortex to corner eddies.
    """
    if not hasattr(target_lid_cavity, '_cache'):
        print("    Solving lid-driven cavity flow...")
        target_lid_cavity._cache = _solve_lid_cavity()
        print("    Simulation complete.")

    snaps_w, snaps_psi = target_lid_cavity._cache
    t_max = 8.0

    w = _interp_field(snaps_w, x, y, t, t_max)
    psi = _interp_field(snaps_psi, x, y, t, t_max)

    # Colormap: stream function → flow structure, vorticity → intensity
    psi_max = max(np.abs(psi).max(), 0.01)
    psi_norm = psi / psi_max  # [-1, 1]
    w_norm = np.clip(w / 20.0, -1, 1)

    r_out = np.clip(psi_norm * 1.5 + np.abs(w_norm) * 0.5, -1, 1)
    g_out = np.clip(-psi_norm * 0.8 + 0.3 * np.abs(w_norm), -1, 1)
    b_out = np.clip(-w_norm * 1.2, -1, 1)

    return np.stack([r_out, g_out, b_out], axis=-1)


# =========================================================
# Rayleigh-Taylor instability
# =========================================================

def _solve_rayleigh_taylor(nx=128, ny=128, nt=5000):
    """Solve Rayleigh-Taylor instability via Boussinesq approximation.

    Heavy fluid (rho=2) on top of light fluid (rho=1).
    Gravity pulls the interface down, creating mushroom-shaped plumes.

    d(rho)/dt + u·nabla(rho) = kappa * laplacian(rho)
    dw/dt + u·nabla(w) = nu * laplacian(w) + g * d(rho)/dx (buoyancy)
    laplacian(psi) = -w

    Returns (rho, w) arrays of shape (nx, ny, n_snapshots).
    """
    dx = 2.0 / (nx - 1)
    dy = 2.0 / (ny - 1)

    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    nu = 0.003       # viscosity
    kappa = 0.002    # density diffusion (numerical)
    g_buoy = 3.0     # buoyancy strength

    # Initial condition: heavy on top, perturbed interface
    interface = 0.05 * np.sin(2 * np.pi * xx) + 0.03 * np.sin(4 * np.pi * xx + 0.5)
    rho = 1.0 + 0.5 * (1.0 + np.tanh((yy - interface) / 0.05))  # smooth step

    w = np.zeros((nx, ny))
    psi = np.zeros((nx, ny))

    dt_diff = 0.15 * dx**2 / max(nu, kappa)
    dt = min(dt_diff, 0.003)

    n_snap = 64
    save_every = max(1, nt // n_snap)
    snaps_rho = []
    snaps_w = []

    for step in range(nt):
        if step % save_every == 0:
            snaps_rho.append(rho.copy())
            snaps_w.append(w.copy())

        # Poisson
        for _ in range(25):
            psi[1:-1,1:-1] = 0.25 * (
                psi[2:,1:-1] + psi[:-2,1:-1] +
                psi[1:-1,2:] + psi[1:-1,:-2] +
                dx**2 * w[1:-1,1:-1])
            psi[0,:] = 0; psi[-1,:] = 0
            psi[:,0] = 0; psi[:,-1] = 0

        # Velocity
        u_vel = np.zeros_like(psi)
        v_vel = np.zeros_like(psi)
        u_vel[1:-1,1:-1] = (psi[1:-1,2:] - psi[1:-1,:-2]) / (2*dy)
        v_vel[1:-1,1:-1] = -(psi[2:,1:-1] - psi[:-2,1:-1]) / (2*dx)

        # Laplacians
        def lap(f):
            L = np.zeros_like(f)
            L[1:-1,1:-1] = (f[2:,1:-1] + f[:-2,1:-1] +
                            f[1:-1,2:] + f[1:-1,:-2] - 4*f[1:-1,1:-1]) / dx**2
            return L

        # Upwind advection
        def advect(f, u, v):
            A = np.zeros_like(f)
            A[1:-1,1:-1] += np.where(u[1:-1,1:-1] > 0,
                u[1:-1,1:-1] * (f[1:-1,1:-1] - f[:-2,1:-1]) / dx,
                u[1:-1,1:-1] * (f[2:,1:-1] - f[1:-1,1:-1]) / dx)
            A[1:-1,1:-1] += np.where(v[1:-1,1:-1] > 0,
                v[1:-1,1:-1] * (f[1:-1,1:-1] - f[1:-1,:-2]) / dy,
                v[1:-1,1:-1] * (f[1:-1,2:] - f[1:-1,1:-1]) / dy)
            return A

        # Buoyancy: horizontal density gradient (rotated gravity)
        # For classic RT, gravity is in -y direction → buoyancy torque = g * drho/dx
        drho_dx = np.zeros_like(rho)
        drho_dx[1:-1,1:-1] = (rho[2:,1:-1] - rho[:-2,1:-1]) / (2*dx)

        # Adaptive CFL
        u_max = max(np.abs(u_vel).max(), np.abs(v_vel).max(), 0.01)
        dt_adv = 0.3 * dx / u_max
        dt = min(dt_diff, dt_adv, 0.003)

        rho += dt * (-advect(rho, u_vel, v_vel) + kappa * lap(rho))
        w += dt * (-advect(w, u_vel, v_vel) + nu * lap(w) + g_buoy * drho_dx)

        # Boundaries: no-slip walls
        rho[0,:] = rho[1,:]; rho[-1,:] = rho[-2,:]
        rho[:,0] = rho[:,1]; rho[:,-1] = rho[:,-2]
        w[0,:] = 0; w[-1,:] = 0
        w[:,0] = 0; w[:,-1] = 0

        rho = np.clip(rho, 0.5, 2.5)
        w = np.clip(w, -30, 30)

        if np.any(np.isnan(rho)):
            rho = np.nan_to_num(rho, nan=1.0)
            w = np.nan_to_num(w, nan=0.0)
            psi = np.nan_to_num(psi, nan=0.0)

    return np.stack(snaps_rho, axis=-1), np.stack(snaps_w, axis=-1)


def target_rayleigh_taylor(x, y, t):
    """Rayleigh-Taylor instability — heavy fluid sinking into light fluid.

    Classic mushroom-shaped plumes with density-mapped coloring.
    Heavy fluid = warm (red/orange), light fluid = cool (blue/purple).
    """
    if not hasattr(target_rayleigh_taylor, '_cache'):
        print("    Solving Rayleigh-Taylor instability...")
        target_rayleigh_taylor._cache = _solve_rayleigh_taylor()
        print("    Simulation complete.")

    snaps_rho, snaps_w = target_rayleigh_taylor._cache
    t_max = 8.0

    rho = _interp_field(snaps_rho, x, y, t, t_max)
    w = _interp_field(snaps_w, x, y, t, t_max)

    # Density colormap: light(1.0)=blue → heavy(2.0)=red/orange
    rho_norm = np.clip((rho - 1.0) / 1.0, 0, 1)  # [0, 1]
    # Vorticity for interface highlighting
    w_norm = np.clip(np.abs(w) / 10.0, 0, 1)

    r_out = np.clip(rho_norm * 2.5 - 0.5 + w_norm * 0.3, -1, 1)
    g_out = np.clip(0.4 - np.abs(rho_norm - 0.5) * 1.5 + w_norm * 0.2, -1, 1)
    b_out = np.clip((1.0 - rho_norm) * 2.0 - 0.3, -1, 1)

    return np.stack([r_out, g_out, b_out], axis=-1)


# =========================================================
# Blast wave (Sedov-Taylor explosion)
# =========================================================

def _solve_blast_wave(nx=128, ny=128, nt=3000):
    """Solve 2D blast wave using acoustic wave equation with nonlinear source.

    d²p/dt² = c² laplacian(p) - damping
    Central high-pressure pulse creates expanding shockwave rings.
    Fireball (temperature) decays slower than pressure wave propagates.

    Returns (pressure, fireball) arrays of shape (nx, ny, n_snapshots).
    """
    dx = 2.0 / (nx - 1)
    dy = 2.0 / (ny - 1)
    c = 1.5  # wave speed

    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    rr = np.sqrt(xx**2 + yy**2)

    dt = 0.3 * dx / c  # CFL

    # Initial condition: moderate pressure pulse at center
    p = 2.0 * np.exp(-rr**2 / 0.02)  # wider Gaussian, lower peak
    p_prev = p.copy()  # zero initial velocity (dp/dt = 0)

    # Fireball: thermal energy that decays slowly
    fireball = 2.5 * np.exp(-rr**2 / 0.03)

    # Absorbing boundary mask (precompute)
    edge = np.minimum(
        np.minimum(xx + 1, 1 - xx),
        np.minimum(yy + 1, 1 - yy)
    )
    absorb = np.clip(edge / 0.2, 0, 1)

    n_snap = 64
    save_every = max(1, nt // n_snap)
    snaps_p = []
    snaps_fb = []

    damping = 0.997  # energy loss per step

    for step in range(nt):
        if step % save_every == 0:
            snaps_p.append(p.copy())
            snaps_fb.append(fireball.copy())

        # Laplacian of pressure
        Lp = np.zeros_like(p)
        Lp[1:-1,1:-1] = (p[2:,1:-1] + p[:-2,1:-1] +
                          p[1:-1,2:] + p[1:-1,:-2] - 4*p[1:-1,1:-1]) / dx**2

        # Verlet integration: p_next = 2*p - p_prev + c²*dt²*Lp
        p_next = 2 * p - p_prev + c**2 * dt**2 * Lp
        p_next *= damping * absorb

        p_prev = p
        p = np.clip(p_next, -5.0, 5.0)

        # Fireball decays and spreads slowly (diffusion)
        Lfb = np.zeros_like(fireball)
        Lfb[1:-1,1:-1] = (fireball[2:,1:-1] + fireball[:-2,1:-1] +
                           fireball[1:-1,2:] + fireball[1:-1,:-2] -
                           4*fireball[1:-1,1:-1]) / dx**2
        fireball += dt * 0.2 * Lfb  # slow thermal diffusion
        fireball *= 0.9993           # slow cooling
        fireball = np.clip(fireball, 0, 5.0)

        # Boundary
        p[0,:] = 0; p[-1,:] = 0
        p[:,0] = 0; p[:,-1] = 0

        if np.any(np.isnan(p)):
            p = np.nan_to_num(p, nan=0.0)
            p_prev = np.nan_to_num(p_prev, nan=0.0)
            fireball = np.nan_to_num(fireball, nan=0.0)

    return np.stack(snaps_p, axis=-1), np.stack(snaps_fb, axis=-1)


def target_blast_wave(x, y, t):
    """Nuclear blast wave — expanding shockwave ring with central fireball.

    Pressure wave → bright ring expanding outward.
    Fireball → hot center fading from white to orange to red.
    """
    if not hasattr(target_blast_wave, '_cache'):
        print("    Solving blast wave equations...")
        target_blast_wave._cache = _solve_blast_wave()
        print("    Simulation complete.")

    snaps_p, snaps_fb = target_blast_wave._cache
    t_max = 8.0

    p = _interp_field(snaps_p, x, y, t, t_max)
    fb = _interp_field(snaps_fb, x, y, t, t_max)

    # Pressure wave: bright white/blue ring
    p_abs = np.clip(np.abs(p) / 2.0, 0, 1)
    # Fireball: blackbody progression
    fb_norm = np.clip(fb / 2.5, 0, 1)

    # Combine: fireball dominates center, pressure ring at edges
    r_out = np.clip(fb_norm * 2.5 - 0.2 + p_abs * 1.5, -1, 1)
    g_out = np.clip(fb_norm * 1.8 - 0.6 + p_abs * 1.0, -1, 1)
    b_out = np.clip(fb_norm * 0.8 - 0.8 + p_abs * 2.0, -1, 1)

    return np.stack([r_out, g_out, b_out], axis=-1)


# =========================================================
# Droplet splash
# =========================================================

def _solve_droplet_splash(nx=128, ny=128, nt=2500, g=3.0):
    """Solve droplet impact using shallow water equations.

    Central impulse creates expanding concentric ring waves
    that reflect off boundaries and interfere.
    Surface tension approximated via 4th-order term.

    Returns (h, speed) arrays of shape (nx, ny, n_snapshots).
    """
    dx = 2.0 / (nx - 1)
    dy = 2.0 / (ny - 1)

    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    rr = np.sqrt(xx**2 + yy**2)

    # Initial condition: calm surface with central crater + rim (moderate)
    h_rest = 1.0
    h = h_rest * np.ones((nx, ny))
    # Crown splash: ring of raised water around central depression
    h += -0.3 * np.exp(-rr**2 / 0.015)  # wider, shallower crater
    h += 0.25 * np.exp(-(rr - 0.15)**2 / 0.005)  # crown rim

    hu = np.zeros((nx, ny))
    hv = np.zeros((nx, ny))

    # Radial outward momentum from impact (moderate)
    theta = np.arctan2(yy, xx)
    speed_init = 0.6 * np.exp(-(rr - 0.12)**2 / 0.008) * rr
    hu += speed_init * np.cos(theta) * h
    hv += speed_init * np.sin(theta) * h

    c_max = np.sqrt(g * h.max()) + 1.0
    dt = 0.15 * min(dx, dy) / c_max  # conservative CFL

    n_snap = 64
    save_every = max(1, nt // n_snap)
    snaps_h = []
    snaps_spd = []

    for step in range(nt):
        if step % save_every == 0:
            u_vel = np.where(h > 0.01, hu / h, 0.0)
            v_vel = np.where(h > 0.01, hv / h, 0.0)
            speed = np.sqrt(u_vel**2 + v_vel**2)
            snaps_h.append(h.copy())
            snaps_spd.append(speed.copy())

        # Lax-Friedrichs for shallow water
        f0 = hu
        f1 = np.where(h > 0.01, hu**2 / h + 0.5 * g * h**2, 0.5 * g * h**2)
        f2 = np.where(h > 0.01, hu * hv / h, 0.0)

        g0 = hv
        g1 = np.where(h > 0.01, hu * hv / h, 0.0)
        g2 = np.where(h > 0.01, hv**2 / h + 0.5 * g * h**2, 0.5 * g * h**2)

        h_avg  = 0.25 * (h[2:,1:-1]  + h[:-2,1:-1]  + h[1:-1,2:]  + h[1:-1,:-2])
        hu_avg = 0.25 * (hu[2:,1:-1] + hu[:-2,1:-1] + hu[1:-1,2:] + hu[1:-1,:-2])
        hv_avg = 0.25 * (hv[2:,1:-1] + hv[:-2,1:-1] + hv[1:-1,2:] + hv[1:-1,:-2])

        h_new  = h_avg  - dt/(2*dx) * (f0[2:,1:-1] - f0[:-2,1:-1]) \
                        - dt/(2*dy) * (g0[1:-1,2:] - g0[1:-1,:-2])
        hu_new = hu_avg - dt/(2*dx) * (f1[2:,1:-1] - f1[:-2,1:-1]) \
                        - dt/(2*dy) * (g1[1:-1,2:] - g1[1:-1,:-2])
        hv_new = hv_avg - dt/(2*dx) * (f2[2:,1:-1] - f2[:-2,1:-1]) \
                        - dt/(2*dy) * (g2[1:-1,2:] - g2[1:-1,:-2])

        h[1:-1,1:-1]  = h_new
        hu[1:-1,1:-1] = hu_new
        hv[1:-1,1:-1] = hv_new

        # Reflective boundaries
        h[0,:] = h[1,:];   h[-1,:] = h[-2,:]
        h[:,0] = h[:,1];   h[:,-1] = h[:,-2]
        hu[0,:] = -hu[1,:];  hu[-1,:] = -hu[-2,:]
        hu[:,0] = hu[:,1];    hu[:,-1] = hu[:,-2]
        hv[0,:] = hv[1,:];   hv[-1,:] = hv[-2,:]
        hv[:,0] = -hv[:,1];   hv[:,-1] = -hv[:,-2]

        h = np.maximum(h, 0.01)

        # Clip momentum (prevents velocity blowup)
        max_mom = 3.0 * h
        hu = np.clip(hu, -max_mom, max_mom)
        hv = np.clip(hv, -max_mom, max_mom)

        hu *= 0.9997
        hv *= 0.9997

        if np.any(np.isnan(h)):
            h = np.nan_to_num(h, nan=h_rest)
            hu = np.nan_to_num(hu, nan=0.0)
            hv = np.nan_to_num(hv, nan=0.0)

    return np.stack(snaps_h, axis=-1), np.stack(snaps_spd, axis=-1)


def target_droplet_splash(x, y, t):
    """Droplet splash — radial crown splash with expanding ring waves.

    Height deviations mapped to water-like colors (dark blue depths,
    white crests, cyan body). Speed adds foam/spray highlights.
    """
    if not hasattr(target_droplet_splash, '_cache'):
        print("    Solving droplet splash (shallow water with impact)...")
        target_droplet_splash._cache = _solve_droplet_splash()
        print("    Simulation complete.")

    snaps_h, snaps_spd = target_droplet_splash._cache
    t_max = 8.0

    h = _interp_field(snaps_h, x, y, t, t_max)
    spd = _interp_field(snaps_spd, x, y, t, t_max)

    # Water colormap
    h_dev = h - 1.0  # deviation from rest
    h_norm = np.clip(h_dev * 3.0, -1, 1)  # amplify
    spd_norm = np.clip(spd / 1.5, 0, 1)

    # Deep blue for troughs, white/cyan for crests, foam for fast areas
    r_out = np.clip(h_norm * 0.6 + spd_norm * 1.2 - 0.5, -1, 1)
    g_out = np.clip(h_norm * 0.8 + spd_norm * 0.8 - 0.1, -1, 1)
    b_out = np.clip(0.3 + h_norm * 0.4 + spd_norm * 0.5, -1, 1)

    return np.stack([r_out, g_out, b_out], axis=-1)


# =========================================================
# Phase-field fracture
# =========================================================

def _solve_fracture(nx=128, ny=128, nt=4000):
    """Solve phase-field fracture propagation.

    Quasi-static elasticity with damage evolution:
    - Strain energy drives crack growth
    - Phase field phi=1 (intact) → phi=0 (broken)
    - Crack nucleates from notch, branches under stress

    dphi/dt = -M * (G_c/l * (phi - l²∇²phi) - 2*(1-phi)*psi_e)

    where psi_e is elastic strain energy.
    Simplified: use Laplace equation for stress, phase-field for damage.

    Returns (phi, stress) arrays of shape (nx, ny, n_snapshots).
    """
    dx = 2.0 / (nx - 1)
    dy = 2.0 / (ny - 1)

    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    # Phase field: 1 = intact, 0 = broken
    phi = np.ones((nx, ny))

    # Pre-existing notch (left side, horizontal)
    notch = (xx < -0.3) & (np.abs(yy) < 0.02)
    phi[notch] = 0.0

    # Material parameters
    Gc = 1.0         # fracture toughness
    l_pf = 0.06      # phase-field length scale
    M = 0.5           # mobility

    # Applied strain increases over time
    n_snap = 64
    save_every = max(1, nt // n_snap)
    snaps_phi = []
    snaps_stress = []

    stress = np.zeros((nx, ny))

    for step in range(nt):
        if step % save_every == 0:
            snaps_phi.append(phi.copy())
            snaps_stress.append(stress.copy())

        # Ramp applied displacement (tension in y-direction)
        strain_y = 0.5 + 2.5 * min(1.0, step / (nt * 0.7))

        # Solve stress field (simplified: Laplace with displacement BCs)
        # sigma = phi² * E * epsilon (degraded by damage)
        # Use Laplace relaxation for stress equilibrium
        for _ in range(15):
            stress[1:-1,1:-1] = 0.25 * (
                stress[2:,1:-1] + stress[:-2,1:-1] +
                stress[1:-1,2:] + stress[1:-1,:-2])
            # BCs: tension at top/bottom
            stress[:,-1] = strain_y     # top: tension
            stress[:,0] = -strain_y     # bottom: tension (opposite)
            stress[0,:] = stress[1,:]   # free sides
            stress[-1,:] = stress[-2,:]

        # Elastic strain energy density
        # Gradient of stress gives strain
        dsdx = np.zeros_like(stress)
        dsdy = np.zeros_like(stress)
        dsdx[1:-1,:] = (stress[2:,:] - stress[:-2,:]) / (2*dx)
        dsdy[:,1:-1] = (stress[:,2:] - stress[:,:-2]) / (2*dy)
        psi_e = 0.5 * (dsdx**2 + dsdy**2)

        # Phase-field Laplacian
        Lphi = np.zeros_like(phi)
        Lphi[1:-1,1:-1] = (phi[2:,1:-1] + phi[:-2,1:-1] +
                            phi[1:-1,2:] + phi[1:-1,:-2] - 4*phi[1:-1,1:-1]) / dx**2

        # Phase-field evolution (Allen-Cahn type)
        dt = 0.002
        driving = Gc / l_pf * (phi - l_pf**2 * Lphi) - 2.0 * (1 - phi) * psi_e
        phi -= dt * M * driving

        # Irreversibility: damage can only increase (phi can only decrease)
        phi = np.clip(phi, 0, 1)

        # Add slight randomness for crack branching
        if step % 200 == 0 and step > nt * 0.3:
            rng = np.random.RandomState(step)
            phi -= 0.01 * rng.rand(nx, ny) * (1 - phi) * (psi_e > np.percentile(psi_e, 90))
            phi = np.clip(phi, 0, 1)

        if np.any(np.isnan(phi)):
            phi = np.nan_to_num(phi, nan=1.0)
            stress = np.nan_to_num(stress, nan=0.0)

    return np.stack(snaps_phi, axis=-1), np.stack(snaps_stress, axis=-1)


def target_fracture(x, y, t):
    """Phase-field fracture — crack propagation through stressed material.

    Intact material shows stress field (cool blue → warm red under tension).
    Crack appears as dark fissure cutting through the material.
    """
    if not hasattr(target_fracture, '_cache'):
        print("    Solving phase-field fracture propagation...")
        target_fracture._cache = _solve_fracture()
        print("    Simulation complete.")

    snaps_phi, snaps_stress = target_fracture._cache
    t_max = 8.0

    phi = _interp_field(snaps_phi, x, y, t, t_max)
    stress = _interp_field(snaps_stress, x, y, t, t_max)

    # Colormap: stress field visible through intact material, crack = dark
    stress_norm = np.clip(stress / 3.0, -1, 1)
    crack = 1.0 - phi  # 0=intact, 1=broken

    # Intact regions: stress-colored (blue compression → red tension)
    r_out = np.clip(stress_norm * 1.5 * phi - crack * 0.8, -1, 1)
    g_out = np.clip((0.3 - np.abs(stress_norm)) * phi * 1.5 - crack * 0.5, -1, 1)
    b_out = np.clip(-stress_norm * 1.5 * phi - crack * 0.3, -1, 1)

    return np.stack([r_out, g_out, b_out], axis=-1)


PATTERNS = {
    'lava_lamp': target_lava_lamp,
    'reaction_diffusion': target_reaction_diffusion,
    'plasma': target_plasma,
    'hsv_flow': target_hsv_flow,
    'shallow_water': target_shallow_water,
    'gray_scott': target_gray_scott,
    'float_glass': target_float_glass,
    'combustion': target_combustion,
    'euler_vorticity': target_euler_vorticity,
    'lid_cavity': target_lid_cavity,
    'rayleigh_taylor': target_rayleigh_taylor,
    'blast_wave': target_blast_wave,
    'droplet_splash': target_droplet_splash,
    'fracture': target_fracture,
}


# =========================================================
# Training
# =========================================================
def generate_training_data(pattern_fn, n_spatial=64, n_temporal=32):
    """Generate training samples: (x, y, t) → RGB.

    x ranges [-1, +1], y ranges [-ASPECT, +ASPECT] where ASPECT = 172/320.
    This matches the FPGA display: both axes map [-1,+1] to the full display,
    so the SIREN must be trained on the same rectangular domain to avoid
    stretching patterns on the 320x172 LCD.
    """
    ASPECT = 172.0 / 320.0  # 0.5375 — display height/width
    x = np.linspace(-1, 1, n_spatial)
    y = np.linspace(-ASPECT, ASPECT, n_spatial)
    t = np.linspace(0, 8.0, n_temporal)  # matches FPGA time_val range [0, 8.0)
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
