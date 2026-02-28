#!/usr/bin/env python3
"""Quick analysis of phase-slope velocity results."""
import json
import numpy as np
from scipy.stats import linregress

# Load tracked velocity results
with open('results_hybrid/velocity_profile.json') as f:
    vp = json.load(f)

layers = vp['layers']
depths_tr = np.array([l['depth_m'] for l in layers])
vels_tr = np.array([l['velocity_m_yr'] for l in layers])
rel_tr = np.array([l['reliable'] for l in layers])

deep_rel = (depths_tr >= 785) & rel_tr
print('=== TRACKED LAYERS (deep, reliable) ===')
print(f'  Total: {len(layers)} layers, {rel_tr.sum()} reliable')
print(f'  Deep reliable: {deep_rel.sum()}')
if deep_rel.any():
    print(f'  Depths: {depths_tr[deep_rel].min():.0f}--{depths_tr[deep_rel].max():.0f} m')
    print(f'  Velocities: [{vels_tr[deep_rel].min():.4f}, {vels_tr[deep_rel].max():.4f}]')
    nye_tr = 0.0453 + 0.000595 * depths_tr[deep_rel]
    resid_tr = vels_tr[deep_rel] - nye_tr
    print(f'  RMS vs Nye: {np.sqrt((resid_tr**2).mean()):.4f}')

# Load phase-slope results (deep only, k=3)
with open('output/apres/slope_velocity_deep_k3.json') as f:
    sv = json.load(f)

depths_ps = np.array(sv['depths'])
vel_ps = np.array(sv['phase_slope_velocities'])
r2_ps = np.array(sv['phase_slope_r2'])
n_good = np.array(sv['phase_slope_n_good'])
nye_ps = np.array(sv['nye_velocities'])

print()
print('=== PHASE-SLOPE (k=3, deep-only SVD) ===')
print(f'  {len(vel_ps)} windows')
for i in range(0, len(vel_ps), 10):
    print(f'  d={depths_ps[i]:.0f}m  v={vel_ps[i]:.4f}  '
          f'nye={nye_ps[i]:.4f}  R2={r2_ps[i]:.3f}  '
          f'n_good={n_good[i]}/190')

print()
print(f'  Velocity range: [{vel_ps.min():.4f}, {vel_ps.max():.4f}]')
print(f'  Median: {np.nanmedian(vel_ps):.4f}, Mean: {np.nanmean(vel_ps):.4f}')
resid_ps = vel_ps - nye_ps
print(f'  RMS vs Nye: {np.sqrt((resid_ps**2).mean()):.4f}')

# Linear fit
sl = linregress(depths_ps, vel_ps)
print(f'  Linear fit: v = {sl.intercept:.4f} + {sl.slope:.6f}*d  (R2={sl.rvalue**2:.3f})')
print(f'  Nye:         v = 0.0453 + 0.000595*d')
