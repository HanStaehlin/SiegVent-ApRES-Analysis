#!/usr/bin/env python3
"""Compare SVD modes: none vs local vs global for phase-slope velocities."""

import json
import numpy as np
import os
import sys

# Add parent dir so we can import
sys.path.insert(0, os.path.dirname(__file__))
from radon_velocity import radon_velocity_profile

DATA = 'data/apres/ImageP2_python.mat'
OUT_DIR = 'output/apres'

# Run all three modes on the deep region
configs = [
    ('none',   0, 'slope_vel_deep_nosvd'),
    ('local',  3, 'slope_vel_deep_local_k3'),
    ('global', 3, 'slope_vel_deep_global_k3'),
]

all_results = {}

for svd_mode, k, label in configs:
    print(f"\n{'#'*70}")
    print(f"# MODE: {svd_mode}  k={k}  ({label})")
    print(f"{'#'*70}")

    results = radon_velocity_profile(
        data_path=DATA,
        depth_min=785, depth_max=1094,
        window_m=10, step_m=5,
        svd_components=k,
        svd_mode=svd_mode,
    )

    # Save JSON
    out_path = os.path.join(OUT_DIR, f'{label}.json')
    os.makedirs(OUT_DIR, exist_ok=True)
    save = {k2: v for k2, v in results.items() if k2 != 'radon_semblances'}
    with open(out_path, 'w') as f:
        json.dump(save, f, indent=2)
    print(f"  -> {out_path}")

    all_results[label] = results

# ---- Summary ----
print("\n" + "=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)

nye_int, nye_sl = 0.0453, 0.000595

for label, res in all_results.items():
    depths = np.array(res['depths'])
    ps = np.array(res['phase_slope_velocities'])
    r2 = np.array(res['phase_slope_r2'])
    ng = np.array(res['phase_slope_n_good'])
    nye = np.array(res['nye_velocities'])

    valid = np.isfinite(ps)
    if not valid.any():
        print(f"\n  {label}: NO valid windows")
        continue

    resid = ps[valid] - nye[valid]
    rms = np.sqrt((resid**2).mean())
    bias = resid.mean()

    # Linear fit
    d = depths[valid]
    v = ps[valid]
    coeffs = np.polyfit(d, v, 1)

    print(f"\n  {label}:")
    print(f"    Valid: {valid.sum()}/{len(ps)}")
    print(f"    v range: [{ps[valid].min():.4f}, {ps[valid].max():.4f}] m/yr")
    print(f"    R² range: [{r2[valid].min():.3f}, {r2[valid].max():.3f}]")
    print(f"    n_good range: [{ng[valid].min()}, {ng[valid].max()}]")
    print(f"    RMS vs Nye: {rms:.4f} m/yr")
    print(f"    Mean bias:  {bias:+.4f} m/yr")
    print(f"    Linear fit: v = {coeffs[1]:.4f} + {coeffs[0]:.6f}*d")
    print(f"    Nye:        v = {nye_int:.4f} + {nye_sl:.6f}*d")

# ---- Plot comparison ----
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    styles = {
        'slope_vel_deep_nosvd':      {'color': '#4393c3', 'marker': 'D', 'label': 'No SVD (raw)'},
        'slope_vel_deep_local_k3':   {'color': '#f4a582', 'marker': 's', 'label': 'Local SVD k=3'},
        'slope_vel_deep_global_k3':  {'color': '#2ca25f', 'marker': 'o', 'label': 'Global SVD k=3'},
    }

    fig, ax = plt.subplots(figsize=(7, 9))

    # Plot each mode
    for label, res in all_results.items():
        depths = np.array(res['depths'])
        ps = np.array(res['phase_slope_velocities'])
        valid = np.isfinite(ps)
        if not valid.any():
            continue
        sty = styles[label]
        ax.scatter(ps[valid], depths[valid], c=sty['color'],
                   marker=sty['marker'], s=28, alpha=0.7,
                   edgecolors='none', label=sty['label'], zorder=2)

        # Linear fit
        coeffs = np.polyfit(depths[valid], ps[valid], 1)
        d_fit = np.array([depths[valid].min(), depths[valid].max()])
        ax.plot(np.polyval(coeffs, d_fit), d_fit, color=sty['color'],
                lw=1.5, ls='--', alpha=0.8, zorder=1)

    # Nye reference
    nye_d = np.array(list(all_results.values())[0]['depths'])
    nye_v = np.array(list(all_results.values())[0]['nye_velocities'])
    ax.plot(nye_v, nye_d, 'k-', lw=2.5, label='Nye model', zorder=3)

    ax.set_xlabel('Vertical velocity (m/yr)', fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)
    ax.set_title('Phase-slope velocity by SVD mode\n(deep region, 785–1094 m)', fontsize=13)
    ax.invert_yaxis()
    ax.set_xlim(-1.5, 2.8)
    ax.legend(loc='lower left', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(OUT_DIR, 'svd_mode_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {plot_path}")
    plt.close()
except Exception as e:
    print(f"Plotting failed: {e}")
