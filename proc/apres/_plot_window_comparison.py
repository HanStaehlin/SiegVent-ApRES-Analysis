#!/usr/bin/env python3
"""Plot 20m local vs global SVD comparison."""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

files = {
    'Local SVD k=3, 10m':  ('output/apres/slope_vel_deep_local_k3.json',    '#f4a582', 's'),
    'Global SVD k=3, 10m': ('output/apres/slope_vel_deep_global_k3.json',   '#a6d96a', 'D'),
    'Local SVD k=3, 20m':  ('output/apres/slope_vel_deep_local_k3_20m.json','#d7191c', 'o'),
    'Global SVD k=3, 20m': ('output/apres/slope_vel_deep_global_k3_20m.json','#2c7bb6','^'),
}

fig, ax = plt.subplots(figsize=(7, 9))
for lbl, (fp, col, mk) in files.items():
    with open(fp) as f: d = json.load(f)
    depths = np.array(d['depths'])
    ps = np.array(d['phase_slope_velocities'])
    ok = np.isfinite(ps)
    ax.scatter(ps[ok], depths[ok], c=col, marker=mk, s=30, alpha=0.7,
               edgecolors='none', label=lbl, zorder=2)
    cf = np.polyfit(depths[ok], ps[ok], 1)
    df = np.array([depths[ok].min(), depths[ok].max()])
    ax.plot(np.polyval(cf, df), df, color=col, lw=1.5, ls='--', alpha=0.8, zorder=1)

nye_d = np.linspace(785, 1094, 100)
ax.plot(0.0453 + 0.000595*nye_d, nye_d, 'k-', lw=2.5, label='Nye model', zorder=3)

ax.set_xlabel('Vertical velocity (m/yr)', fontsize=12)
ax.set_ylabel('Depth (m)', fontsize=12)
ax.set_title('Phase-slope velocity: window & SVD mode comparison\n(deep region, 785-1094 m)', fontsize=12)
ax.invert_yaxis()
ax.set_xlim(-1.5, 2.5)
ax.legend(loc='lower left', fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output/apres/svd_window_comparison.png', dpi=150, bbox_inches='tight')
print('Saved output/apres/svd_window_comparison.png')
