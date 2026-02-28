#!/usr/bin/env python3
"""
Slope-based Velocity Estimation for ApRES Deep Layers

Instead of tracking individual layers (which is noisy at depth), this
approach finds the dominant *slope* in the SVD-denoised echogram at each
depth window.  The slope in the (depth x time) plane directly yields
the vertical velocity.

Three methods are implemented and compared:

1. **Phase-slope** -- For each depth bin, fit a line to the unwrapped
   temporal phase of the SVD-denoised complex signal.  The slope gives
   the velocity via  v = (dphi/dt) * lambda_c / (4*pi).  Average over
   the depth window for robustness.

2. **Radon / slant-stack** -- For candidate velocities, shift the
   amplitude image along lines of that slope and sum.  The velocity
   maximising the stacked energy wins.  Works on de-trended amplitude.

3. **Structure tensor** -- Compute image gradients and find the dominant
   orientation from the eigenvectors of the smoothed outer-product
   tensor.  Very fast.

Usage
-----
    python radon_velocity.py \\
        --data data/apres/ImageP2_python.mat \\
        --svd-components 3 \\
        --depth-min 200 --depth-max 1094 \\
        --window 10 \\
        --output output/apres/radon_velocity.json

Author: SiegVent2023 project
"""

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
import json
import argparse
import time as time_mod
from typing import Optional


# ====================================================================
#  Method 1: Phase-slope velocity
# ====================================================================
def phase_slope_velocity(
    strip_complex: np.ndarray,
    time_days: np.ndarray,
    lambdac: float,
) -> dict:
    """
    Estimate velocity from the temporal phase slope at each depth bin,
    then take the robust median over the window.

    For a reflector moving at velocity v, the radar phase evolves as
        phi(t) = phi_0 + (4*pi / lambda_c) * v * t
    so  v = dphi/dt * lambda_c / (4*pi).
    """
    n_bins, n_times = strip_complex.shape
    t_yr = (time_days - time_days[0]) / 365.25

    velocities = np.full(n_bins, np.nan)
    r2_values = np.full(n_bins, np.nan)

    for i in range(n_bins):
        z = strip_complex[i, :]
        amp = np.abs(z)

        # Skip very low amplitude bins
        if np.median(amp) < 1e-6:
            continue

        phase = np.unwrap(np.angle(z))

        try:
            coeffs = np.polyfit(t_yr, phase, 1)
            slope = coeffs[0]  # rad/yr

            phase_pred = np.polyval(coeffs, t_yr)
            ss_res = np.sum((phase - phase_pred) ** 2)
            ss_tot = np.sum((phase - phase.mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-30)

            v = slope * lambdac / (4 * np.pi)
            velocities[i] = v
            r2_values[i] = r2
        except (np.linalg.LinAlgError, ValueError):
            continue

    valid = np.isfinite(velocities)
    if valid.sum() == 0:
        return {'best_v': np.nan, 'median_r2': 0.0, 'n_good': 0}

    good = valid & (r2_values > 0.5)
    if good.sum() >= 3:
        best_v = float(np.median(velocities[good]))
        med_r2 = float(np.median(r2_values[good]))
        n_good = int(good.sum())
    else:
        best_v = float(np.median(velocities[valid]))
        med_r2 = float(np.median(r2_values[valid]))
        n_good = int(valid.sum())

    return {
        'best_v': best_v,
        'median_r2': med_r2,
        'n_good': n_good,
    }


# ====================================================================
#  Method 2: Radon / slant-stack on amplitude
# ====================================================================
def radon_velocity(
    strip_amp: np.ndarray,
    time_days: np.ndarray,
    dz: float,
    v_candidates: np.ndarray,
) -> dict:
    """
    Slant-stack velocity estimation on de-meaned amplitude.

    For each candidate velocity, shift each time column by the depth
    offset implied by that velocity, then sum along depth.
    The velocity with maximum stacked power wins.
    """
    n_bins, n_times = strip_amp.shape
    t_yr = (time_days - time_days[0]) / 365.25
    t_centered = t_yr - t_yr.mean()

    # Subsample time for speed (every 3rd measurement)
    t_sub = np.arange(0, n_times, 3)
    row_idx = np.arange(n_bins, dtype=float)

    semblance = np.zeros(len(v_candidates))

    for iv, v in enumerate(v_candidates):
        bin_shifts = v * t_centered / dz  # shift per time step (bins)
        total = 0.0
        for jt in t_sub:
            src = row_idx - bin_shifts[jt]
            src_clamp = np.clip(src, 0, n_bins - 1).astype(int)
            total += np.sum(strip_amp[src_clamp, jt]) ** 2
        semblance[iv] = total

    semblance_norm = semblance / (semblance.max() + 1e-30)
    idx_best = np.argmax(semblance)
    med = np.median(semblance)

    return {
        'best_v': float(v_candidates[idx_best]),
        'semblance': semblance_norm,
        'peak_snr': float(semblance[idx_best] / (med + 1e-30)),
    }


# ====================================================================
#  Method 3: Structure tensor
# ====================================================================
def structure_tensor_velocity(
    strip_amp: np.ndarray,
    time_days: np.ndarray,
    dz: float,
    smooth_sigma_bins: int = 10,
) -> dict:
    """
    Dominant streak orientation from the structure tensor.
    Works in pixel coordinates then converts to physical velocity.
    """
    n_bins, n_times = strip_amp.shape
    t_yr = (time_days - time_days[0]) / 365.25
    dt_yr = float(np.mean(np.diff(t_yr)))

    # Gradients in pixel coordinates
    gz = np.gradient(strip_amp, axis=0)
    gt = np.gradient(strip_amp, axis=1)

    sigma = (smooth_sigma_bins, smooth_sigma_bins)
    J11 = gaussian_filter(gz * gz, sigma)
    J12 = gaussian_filter(gz * gt, sigma)
    J22 = gaussian_filter(gt * gt, sigma)

    S11, S12, S22 = J11.mean(), J12.mean(), J22.mean()
    T = np.array([[S11, S12], [S12, S22]])
    eigvals, eigvecs = np.linalg.eigh(T)

    idx_max, idx_min = np.argmax(eigvals), np.argmin(eigvals)
    coherence = (eigvals[idx_max] - eigvals[idx_min]) / \
                (eigvals[idx_max] + eigvals[idx_min] + 1e-30)

    # Streak direction = min-eigenvalue eigenvector
    e = eigvecs[:, idx_min]  # (z_pix, t_pix)

    # Convert: v = (e_z * dz) / (e_t * dt_yr)  m/yr
    if abs(e[1]) > 1e-12:
        best_v = float((e[0] * dz) / (e[1] * dt_yr))
    else:
        best_v = np.nan

    return {'best_v': best_v, 'coherence': float(coherence)}


# ====================================================================
#  Main pipeline
# ====================================================================
def radon_velocity_profile(
    data_path: str,
    depth_min: float = 200.0,
    depth_max: float = 1094.0,
    window_m: float = 10.0,
    step_m: float = 5.0,
    svd_components: int = 3,
    svd_mode: str = 'none',  # 'none', 'local', 'global'
    v_min: float = -0.5,
    v_max: float = 1.5,
    n_velocities: int = 200,
    verbose: bool = True,
) -> dict:
    """Estimate vertical velocity profile using three methods.

    svd_mode controls denoising:
      - 'none':   use raw complex data (no SVD)
      - 'local':  apply SVD independently within each sliding window
      - 'global': apply SVD to the entire depth region first (original)
    """
    t0 = time_mod.time()

    if verbose:
        print("=" * 70)
        print(f"SLOPE-BASED VELOCITY ESTIMATION -- svd_mode={svd_mode}")
        print("=" * 70)

    # -- Load --
    if verbose:
        print(f"\nLoading {data_path} ...")
    mat = loadmat(data_path)
    raw_complex = np.array(mat['RawImageComplex'])
    Rcoarse = mat['Rcoarse'].flatten()
    time_days = mat['TimeInDays'].flatten()
    lambdac = float(mat.get('lambdac', np.array([0.5608])).flatten()[0])
    del mat

    n_bins, n_times = raw_complex.shape
    dz = float(Rcoarse[1] - Rcoarse[0])

    if verbose:
        print(f"  Data: {n_bins} bins x {n_times} times")
        print(f"  dz = {dz:.4f} m, lambda_c = {lambdac:.4f} m")
        print(f"  Time: {time_days[-1] - time_days[0]:.1f} days "
              f"({(time_days[-1] - time_days[0])/365.25:.2f} yr)")

    # -- Extract region --
    idx_start = np.searchsorted(Rcoarse, depth_min)
    idx_end = np.searchsorted(Rcoarse, depth_max)
    region = raw_complex[idx_start:idx_end, :]
    depths_region = Rcoarse[idx_start:idx_end]
    n_region = len(depths_region)

    if verbose:
        print(f"  Region: {n_region} bins, "
              f"{depths_region[0]:.1f} -- {depths_region[-1]:.1f} m")

    # -- SVD denoising --
    if svd_mode == 'global':
        if verbose:
            print(f"\nGlobal SVD denoising (k={svd_components}) ...")
        U, S, Vh = np.linalg.svd(region, full_matrices=False)
        total_energy = np.sum(S ** 2)
        kept_energy = np.sum(S[:svd_components] ** 2)
        if verbose:
            print(f"  Kept energy: {kept_energy/total_energy*100:.1f}%")
        S_trunc = np.zeros_like(S)
        S_trunc[:svd_components] = S[:svd_components]
        denoised_global = U @ np.diag(S_trunc) @ Vh
    elif svd_mode == 'none':
        if verbose:
            print("\nNo SVD denoising -- using raw complex data")
        denoised_global = None
    elif svd_mode == 'local':
        if verbose:
            print(f"\nLocal SVD denoising (k={svd_components} per window)")
        denoised_global = None
    else:
        raise ValueError(f"Unknown svd_mode: {svd_mode}")

    # -- Windows --
    window_bins = max(1, int(round(window_m / dz)))
    step_bins = max(1, int(round(step_m / dz)))
    v_candidates = np.linspace(v_min, v_max, n_velocities)
    window_starts = list(range(0, n_region - window_bins + 1, step_bins))
    n_windows = len(window_starts)

    if verbose:
        print(f"\nWindow: {window_m:.1f} m = {window_bins} bins, "
              f"step: {step_m:.1f} m = {step_bins} bins")
        print(f"Windows: {n_windows}")
        print(f"Velocity search: [{v_min:.2f}, {v_max:.2f}] m/yr, "
              f"{n_velocities} candidates\n")

    # -- Results --
    centers = []
    ps_vel, ps_r2, ps_ng = [], [], []
    rd_vel, rd_snr, rd_sem = [], [], []
    st_vel, st_coh = [], []

    for wi, i in enumerate(window_starts):
        cd = float(depths_region[i + window_bins // 2])
        centers.append(cd)

        if svd_mode == 'global':
            strip_c = denoised_global[i : i + window_bins, :]
        elif svd_mode == 'local':
            # SVD within this window only
            win_raw = region[i : i + window_bins, :]
            U_w, S_w, Vh_w = np.linalg.svd(win_raw, full_matrices=False)
            S_t = np.zeros_like(S_w)
            S_t[:svd_components] = S_w[:svd_components]
            strip_c = U_w @ np.diag(S_t) @ Vh_w
        else:  # 'none'
            strip_c = region[i : i + window_bins, :]

        strip_a = np.abs(strip_c)
        strip_a = strip_a - strip_a.mean(axis=1, keepdims=True)

        # 1. Phase-slope
        ps = phase_slope_velocity(strip_c, time_days, lambdac)
        ps_vel.append(ps['best_v'])
        ps_r2.append(ps['median_r2'])
        ps_ng.append(ps['n_good'])

        # 2. Radon
        rd = radon_velocity(strip_a, time_days, dz, v_candidates)
        rd_vel.append(rd['best_v'])
        rd_snr.append(rd['peak_snr'])
        rd_sem.append(rd['semblance'].tolist())

        # 3. Structure tensor
        st = structure_tensor_velocity(strip_a, time_days, dz)
        st_vel.append(st['best_v'])
        st_coh.append(st['coherence'])

        if verbose and ((wi + 1) % 20 == 0 or wi + 1 == n_windows):
            print(f"  [{wi+1:4d}/{n_windows}] d={cd:7.1f}m  "
                  f"phase={ps['best_v']:+.4f}  "
                  f"radon={rd['best_v']:+.4f}  "
                  f"struct={st['best_v']:+.4f}")

    elapsed = time_mod.time() - t0

    # -- Nye reference --
    nye_int, nye_sl = 0.0453, 0.000595
    nye_v = [nye_int + nye_sl * d for d in centers]

    # -- Summary --
    if verbose:
        print(f"\nDone in {elapsed:.1f} s")
        arr_ps = np.array(ps_vel)
        arr_rd = np.array(rd_vel)
        arr_st = np.array(st_vel)
        arr_nye = np.array(nye_v)

        vp = np.isfinite(arr_ps)
        if vp.any():
            resid = arr_ps[vp] - arr_nye[vp]
            print(f"\n  Phase-slope: {vp.sum()}/{len(arr_ps)} valid, "
                  f"v=[{arr_ps[vp].min():.4f}, {arr_ps[vp].max():.4f}], "
                  f"RMS vs Nye={np.sqrt((resid**2).mean()):.4f}")

        print(f"  Radon: v=[{arr_rd.min():.4f}, {arr_rd.max():.4f}], "
              f"SNR=[{min(rd_snr):.2f}, {max(rd_snr):.2f}]")

        vs = np.isfinite(arr_st) & (np.abs(arr_st) < 10)
        if vs.any():
            print(f"  Struct: {vs.sum()}/{len(arr_st)} reasonable, "
                  f"v=[{arr_st[vs].min():.4f}, {arr_st[vs].max():.4f}]")

    return {
        'svd_mode': svd_mode,
        'svd_components': svd_components,
        'window_m': window_m, 'step_m': step_m,
        'depth_spacing_m': dz, 'lambdac': lambdac,
        'depths': centers,
        'phase_slope_velocities': ps_vel,
        'phase_slope_r2': ps_r2,
        'phase_slope_n_good': ps_ng,
        'radon_velocities': rd_vel,
        'radon_snrs': rd_snr,
        'radon_semblances': rd_sem,
        'struct_velocities': st_vel,
        'struct_coherences': st_coh,
        'nye_velocities': nye_v,
        'v_candidates': v_candidates.tolist(),
        'elapsed_s': elapsed,
    }


# ====================================================================
#  Plotting
# ====================================================================
def plot_results(results: dict, save_path: Optional[str] = None):
    """Plot velocity profiles from all three methods."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available -- skipping plot")
        return

    depths = np.array(results['depths'])
    nye = np.array(results['nye_velocities'])
    ps  = np.array(results['phase_slope_velocities'])
    r2  = np.array(results['phase_slope_r2'])
    rd  = np.array(results['radon_velocities'])
    snr = np.array(results['radon_snrs'])
    st  = np.array(results['struct_velocities'])
    coh = np.array(results['struct_coherences'])
    vc  = np.array(results['v_candidates'])
    sem = np.array(results['radon_semblances'])

    fig, axes = plt.subplots(1, 4, figsize=(22, 8), sharey=True)

    # 1: Phase-slope
    ax = axes[0]
    vp = np.isfinite(ps)
    if vp.any():
        sc = ax.scatter(ps[vp], depths[vp], c=r2[vp], cmap='viridis',
                        s=20, vmin=0, vmax=1, edgecolors='none')
        plt.colorbar(sc, ax=ax, label='R²', shrink=0.6)
    ax.plot(nye, depths, 'r--', lw=1.5, label='Nye')
    ax.set_xlabel('Velocity (m/yr)')
    ax.set_ylabel('Depth (m)')
    ax.set_title('Phase-slope')
    ax.legend(fontsize=8)
    ax.invert_yaxis()

    # 2: Radon semblance
    ax = axes[1]
    if sem.size:
        ax.imshow(sem, aspect='auto',
                  extent=[vc[0], vc[-1], depths[-1], depths[0]],
                  cmap='inferno', interpolation='bilinear')
        ax.plot(nye, depths, 'c--', lw=1.5, label='Nye')
        ax.plot(rd, depths, 'w.', ms=3, alpha=0.6, label='pick')
    ax.set_xlabel('Velocity (m/yr)')
    ax.set_title('Radon semblance')
    ax.legend(fontsize=8)

    # 3: Structure tensor
    ax = axes[2]
    ok = np.isfinite(st) & (np.abs(st) < 5)
    if ok.any():
        sc2 = ax.scatter(st[ok], depths[ok], c=coh[ok], cmap='plasma',
                         s=20, vmin=0, vmax=1)
        plt.colorbar(sc2, ax=ax, label='Coherence', shrink=0.6)
    ax.plot(nye, depths, 'r--', lw=1.5, label='Nye')
    ax.set_xlabel('Velocity (m/yr)')
    ax.set_title('Structure tensor')
    ax.legend(fontsize=8)

    # 4: Comparison
    ax = axes[3]
    ax.plot(nye, depths, 'r--', lw=2, label='Nye', zorder=3)
    gp = vp & (r2 > 0.7)
    if gp.any():
        ax.scatter(ps[gp], depths[gp], c='steelblue', s=15, alpha=0.7,
                   label=f'Phase R²>0.7 (n={gp.sum()})', edgecolors='none')
    gs = ok & (coh > 0.5)
    if gs.any():
        ax.scatter(st[gs], depths[gs], c='darkorange', s=15, alpha=0.7,
                   label=f'Struct coh>0.5 (n={gs.sum()})', marker='s',
                   edgecolors='none')
    ax.set_xlabel('Velocity (m/yr)')
    ax.set_title('Comparison')
    ax.legend(fontsize=7)

    plt.tight_layout()
    out = save_path or '/tmp/radon_velocity.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {out}")
    plt.close()


# ====================================================================
#  CLI
# ====================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Slope-based velocity estimation for ApRES data')
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', default=None)
    parser.add_argument('--plot', default=None)
    parser.add_argument('--svd-components', type=int, default=3)
    parser.add_argument('--svd-mode', choices=['none', 'local', 'global'],
                        default='none',
                        help='SVD denoising mode: none, local (per window), '
                             'or global (entire region)')
    parser.add_argument('--depth-min', type=float, default=200.0)
    parser.add_argument('--depth-max', type=float, default=1094.0)
    parser.add_argument('--window', type=float, default=10.0)
    parser.add_argument('--step', type=float, default=5.0)
    parser.add_argument('--v-min', type=float, default=-0.5)
    parser.add_argument('--v-max', type=float, default=1.5)
    parser.add_argument('--n-velocities', type=int, default=200)

    args = parser.parse_args()

    results = radon_velocity_profile(
        data_path=args.data,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        window_m=args.window,
        step_m=args.step,
        svd_components=args.svd_components,
        svd_mode=args.svd_mode,
        v_min=args.v_min,
        v_max=args.v_max,
        n_velocities=args.n_velocities,
    )

    # Save JSON (skip semblances — too large)
    import os
    out_path = args.output or 'output/apres/radon_velocity.json'
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    save = {k: v for k, v in results.items() if k != 'radon_semblances'}
    with open(out_path, 'w') as f:
        json.dump(save, f, indent=2)
    print(f"\nSaved results to {out_path}")

    plot_results(results, save_path=args.plot)


if __name__ == '__main__':
    main()
