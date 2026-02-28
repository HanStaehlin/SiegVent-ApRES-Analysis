#!/usr/bin/env python3
"""
Ice-Water Transition Detection via Phase-Radon Periodicity Change

The phase velocity relationship is:  v = (lambda_c / 4pi) * dphi/dt

The radar wavelength depends on the dielectric constant of the medium:
  lambda = lambda_0 / sqrt(epsilon)

  ice:   eps ~ 3.18  =>  n ~ 1.78  =>  lambda_ice   ~ 0.561 m
  water: eps ~ 80    =>  n ~ 8.94  =>  lambda_water  ~ 0.112 m

If the Phase-Radon is run assuming lambda_ice everywhere, but the signal
below the ice-water interface is actually propagating through water, the
apparent velocity will be WRONG by a factor of ~5. By running the Radon
with both wavelengths, we can detect where the transition occurs.

Usage:
    python plot_ice_water_transition.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# Physical Parameters
FREQ_MHZ = 300.0
C_VACUUM = 2.998e8         # m/s
EPS_ICE = 3.18
EPS_WATER = 80.0

LAMBDA_0 = C_VACUUM / (FREQ_MHZ * 1e6)          # ~1.0 m
LAMBDA_ICE = LAMBDA_0 / np.sqrt(EPS_ICE)         # ~0.561 m
LAMBDA_WATER = LAMBDA_0 / np.sqrt(EPS_WATER)     # ~0.112 m

def phase_radon_profile(raw_complex, Rcoarse, time_years, lambdac,
                        depth_min, depth_max, window_m=10.0, step_m=1.0,
                        v_min=0.0, v_max=2.0, num_v=1000, svd_k=3):
    """Compute Phase-Radon velocity profile for a given lambda."""
    depths = np.arange(depth_min + window_m/2, depth_max - window_m/2 + step_m, step_m)
    velocities = np.linspace(v_min, v_max, num_v)
    n_times = len(time_years)

    # Precompute derotation matrix
    phase_matrix = (4 * np.pi / lambdac) * np.outer(time_years, velocities)
    E_T = np.exp(-1j * phase_matrix)

    best_vs = []
    stack_powers = []

    for target_depth in depths:
        mask = (Rcoarse >= target_depth - window_m/2) & (Rcoarse <= target_depth + window_m/2)
        if not np.any(mask):
            best_vs.append(np.nan)
            stack_powers.append(np.nan)
            continue

        S_win = raw_complex[mask, :]
        U, sigma, Vh = np.linalg.svd(S_win, full_matrices=False)
        k = min(svd_k, len(sigma))
        S_den = (U[:, :k] * sigma[:k]) @ Vh[:k, :]

        stack_mag = np.abs(S_den @ E_T) / n_times
        bulk = np.sum(stack_mag, axis=0)

        idx = np.argmax(bulk)
        best_vs.append(velocities[idx])
        stack_powers.append(bulk[idx])

    return depths, np.array(best_vs), np.array(stack_powers)


def main():
    data_path = 'data/apres/ImageP2_python.mat'

    print(f"Physical parameters:")
    print(f"  lambda_ice   = {LAMBDA_ICE:.4f} m  (eps = {EPS_ICE})")
    print(f"  lambda_water = {LAMBDA_WATER:.4f} m  (eps = {EPS_WATER})")
    print(f"  ratio        = {LAMBDA_ICE / LAMBDA_WATER:.2f}x")
    print()

    print(f"Loading {data_path}...")
    mat = loadmat(data_path)
    Rcoarse = mat['Rcoarse'].flatten()
    time_days = mat['TimeInDays'].flatten()
    time_years = time_days / 365.25
    raw_complex = np.array(mat['RawImageComplex'], dtype=np.complex64)

    max_depth = Rcoarse[-1]
    print(f"Max depth in data: {max_depth:.1f} m")

    # Focus on the transition zone: 900m to the end of the data
    # Also include some "safe ice" for reference
    depth_min = 900.0
    depth_max = min(max_depth, 1120.0)
    window_m = 10.0  # narrower window for better spatial resolution
    step_m = 1.0     # 1m steps for fine resolution

    print(f"\nPhase-Radon with lambda_ice ({LAMBDA_ICE:.4f} m)...")
    t0 = time.time()
    d_ice, v_ice, p_ice = phase_radon_profile(
        raw_complex, Rcoarse, time_years, LAMBDA_ICE,
        depth_min, depth_max, window_m, step_m)
    print(f"  Done in {time.time()-t0:.1f}s ({len(d_ice)} windows)")

    print(f"\nPhase-Radon with lambda_water ({LAMBDA_WATER:.4f} m)...")
    t0 = time.time()
    d_water, v_water, p_water = phase_radon_profile(
        raw_complex, Rcoarse, time_years, LAMBDA_WATER,
        depth_min, depth_max, window_m, step_m)
    print(f"  Done in {time.time()-t0:.1f}s ({len(d_water)} windows)")

    # Nye model for reference
    nye_int, nye_sl = 0.0453, 0.000595
    nye_v = nye_int + nye_sl * d_ice

    # Ice-water interface (approximate)
    ice_base = 1094.0  # approx from previous analysis

    # ---- Plotting ---- #
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharey=True)

    # Panel 1: Phase-Radon assuming ICE wavelength
    ax = axes[0]
    sc = ax.scatter(v_ice, d_ice, c=p_ice, cmap='viridis', s=20,
                    edgecolors='none', alpha=0.8)
    ax.plot(nye_v, d_ice, 'r--', lw=2, label='Nye Model')
    ax.axhline(ice_base, color='cyan', linestyle=':', lw=2, label=f'Ice base ~{ice_base}m')
    ax.set_xlabel('Velocity (m/yr)', fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)
    ax.set_title(f'Assuming $\\lambda_{{ice}}$ = {LAMBDA_ICE:.3f} m', fontsize=13)
    ax.set_xlim(0, 2)
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.5)
    plt.colorbar(sc, ax=ax, label='Stack Power', shrink=0.6)

    # Panel 2: Phase-Radon assuming WATER wavelength
    ax = axes[1]
    sc2 = ax.scatter(v_water, d_water, c=p_water, cmap='viridis', s=20,
                     edgecolors='none', alpha=0.8)
    ax.plot(nye_v, d_ice, 'r--', lw=2, label='Nye Model')
    ax.axhline(ice_base, color='cyan', linestyle=':', lw=2, label=f'Ice base ~{ice_base}m')
    ax.set_xlabel('Velocity (m/yr)', fontsize=12)
    ax.set_title(f'Assuming $\\lambda_{{water}}$ = {LAMBDA_WATER:.3f} m', fontsize=13)
    ax.set_xlim(0, 2)
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.5)
    plt.colorbar(sc2, ax=ax, label='Stack Power', shrink=0.6)

    # Panel 3: Stack power comparison (which lambda fits better?)
    ax = axes[2]
    # Normalize powers
    p_ice_n = p_ice / (np.max(p_ice) + 1e-10)
    p_water_n = p_water / (np.max(p_water) + 1e-10)
    ax.plot(p_ice_n, d_ice, 'b-', lw=2, label=f'$\\lambda_{{ice}}$ fit', alpha=0.8)
    ax.plot(p_water_n, d_water, 'r-', lw=2, label=f'$\\lambda_{{water}}$ fit', alpha=0.8)
    ax.axhline(ice_base, color='cyan', linestyle=':', lw=2, label=f'Ice base ~{ice_base}m')
    ax.set_xlabel('Normalized Stack Power', fontsize=12)
    ax.set_title('Which $\\lambda$ fits better?', fontsize=13)
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.set_xlim(0, 1.05)

    for ax in axes:
        ax.invert_yaxis()

    fig.suptitle('Ice-Water Transition: Phase Periodicity Change',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_dir = Path('output/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / 'ice_water_transition.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {out}")

if __name__ == '__main__':
    main()
