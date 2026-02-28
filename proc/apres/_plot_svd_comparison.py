#!/usr/bin/env python3
"""Plot all three SVD modes on one depth-vs-velocity figure."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

files = {
    'No SVD (raw)':   ('output/apres/slope_vel_deep_nosvd.json',     '#4393c3', 'D'),
    'Local SVD k=3':  ('output/apres/slope_vel_deep_local_k3.json',  '#f4a582', 's'),
    'Global SVD k=3': ('output/apres/slope_vel_deep_global_k3.json', '#2ca25f', 'o'),
}

fig, ax = plt.subplots(figsize=(7, 9))

for lbl, (fp, col, mk) in files.items():
    with open(fp) as f:
        d = json.load(f)
    depths = np.array(d['depths'])
    ps = np.array(d['phase_slope_velocities'])
    valid = np.isfinite(ps)
    if not valid.any():
        continue
    ax.scatter(ps[valid], depths[valid], c=col, marker=mk, s=28,
               alpha=0.7, edgecolors='none', label=lbl, zorder=2)
    coeffs = np.polyfit(depths[valid], ps[valid], 1)
    d_fit = np.array([depths[valid].min(), depths[valid].max()])
    ax.plot(np.polyval(coeffs, d_fit), d_fit, color=col,
            lw=1.5, ls='--', alpha=0.8, zorder=1)

# Nye reference
nye_d = np.linspace(785, 1094, 100)
nye_v = 0.0453 + 0.000595 * nye_d
ax.plot(nye_v, nye_d, 'k-', lw=2.5, label='Nye model', zorder=3)

ax.set_xlabel('Vertical velocity (m/yr)', fontsize=12)
ax.set_ylabel('Depth (m)', fontsize=12)
ax.set_title('Phase-slope velocity by SVD mode\n(deep region, 785–1094 m)', fontsize=13)
ax.invert_yaxis()
ax.set_xlim(-1.5, 2.8)
ax.legend(loc='lower left', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output/apres/svd_mode_comparison.png', dpi=150, bbox_inches='tight')
print('Saved output/apres/svd_mode_comparison.png')
