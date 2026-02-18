#!/usr/bin/env python3
"""
Improved Deep Layer Detection for ApRES Data

Improvements over deep_layer_detection.py:
1. SVD denoising before tracking — removes incoherent noise, revealing
   longer/more connected elevated-amplitude segments
2. Relaxed segment parameters — shorter min_pts (15 instead of 30),
   lower amp_factor for denoised data
3. Multi-bin coherent averaging — average complex signal over small
   depth windows to boost SNR before tracking
4. Gap bridging — allow short sub-threshold gaps within a segment
   (the signal doesn't have to be continuously elevated)
5. Coherence-weighted segment selection — prefer segments where
   adjacent bins also show consistent phase

Usage:
    python deep_layer_detection_v2.py \\
        --data /path/to/ImageP2_python.mat \\
        --output /path/to/output/deep_layers_v2.json \\
        [--svd-components 100] [--gap-tolerance 10] [--min-pts 15]

Author: SiegVent2023 project
"""

import numpy as np
from scipy import stats
from scipy.io import loadmat
from scipy.ndimage import uniform_filter1d
from typing import Optional, Tuple, List
import json
import argparse
import time as time_mod


def nye_velocity(depth: float, intercept: float, slope: float) -> float:
    """Compute expected Nye velocity at a given depth."""
    return intercept + slope * depth


def svd_denoise_region(complex_data: np.ndarray, n_components: int = 100) -> np.ndarray:
    """
    Apply SVD-based denoising to a (n_bins x n_times) complex matrix.

    Keeps the top n_components singular values (coherent layer structure),
    discarding smaller ones (incoherent noise).
    """
    U, S, Vh = np.linalg.svd(complex_data, full_matrices=False)
    n_keep = min(n_components, len(S))
    S_filt = np.zeros_like(S)
    S_filt[:n_keep] = S[:n_keep]
    return U @ np.diag(S_filt) @ Vh


def find_elevated_segments(
    z: np.ndarray,
    amp_factor: float = 1.3,
    min_pts: int = 15,
    gap_tolerance: int = 10,
    smooth_size: int = 15,
) -> Tuple[np.ndarray, List[Tuple[int, int]], float]:
    """
    Find contiguous elevated-amplitude segments, allowing short gaps.

    Unlike the original which requires strictly contiguous elevated points,
    this version allows up to gap_tolerance consecutive sub-threshold
    points within a segment. This produces longer, more connected segments.

    Args:
        z: Complex time series at one depth bin
        amp_factor: Threshold = factor × median(smoothed amplitude)
        min_pts: Minimum total elevated points in a segment
        gap_tolerance: Maximum consecutive sub-threshold points to bridge
        smooth_size: Smoothing window for amplitude

    Returns:
        elevated: Boolean mask of elevated points
        segments: List of (start, end) tuples
        threshold: Amplitude threshold used
    """
    amp = np.abs(z)
    amp_smooth = uniform_filter1d(amp, size=smooth_size)
    threshold = amp_factor * np.median(amp_smooth)
    elevated = amp_smooth > threshold

    # Find segments with gap bridging
    segments = []
    n = len(elevated)
    i = 0
    while i < n:
        if not elevated[i]:
            i += 1
            continue
        # Start of a potential segment
        seg_start = i
        gap_count = 0
        n_elevated = 0
        j = i
        while j < n:
            if elevated[j]:
                gap_count = 0
                n_elevated += 1
                j += 1
            else:
                gap_count += 1
                if gap_count > gap_tolerance:
                    # End segment before the gap
                    j -= gap_count
                    break
                j += 1
        seg_end = min(j, n)

        if n_elevated >= min_pts and seg_end - seg_start >= min_pts:
            segments.append((seg_start, seg_end))

        i = seg_end + 1

    return elevated, segments, threshold


def multi_bin_average(complex_data: np.ndarray, center_idx: int,
                      half_width: int = 1) -> np.ndarray:
    """
    Coherently average complex signal over adjacent depth bins.

    This averages complex values across 2*half_width+1 bins centered
    at center_idx. If the layer is a true reflector, adjacent bins
    should have correlated phase, so averaging boosts SNR.
    """
    n_bins = complex_data.shape[0]
    lo = max(0, center_idx - half_width)
    hi = min(n_bins, center_idx + half_width + 1)
    return np.mean(complex_data[lo:hi, :], axis=0)


def segment_stitched_tracking_v2(
    z: np.ndarray,
    time_days: np.ndarray,
    expected_v: float,
    lambdac: float,
    amp_factor: float = 1.2,
    min_pts: int = 15,
    nye_tol: float = 0.20,
    gap_tolerance: int = 10,
) -> Tuple[Optional[np.ndarray], float, float, int, List[dict], float]:
    """
    Improved segment-stitched tracking with gap bridging.

    Key improvements over v1:
    - Gap bridging: short sub-threshold windows don't break segments
    - Lower min_pts: finds shorter but still valid segments
    - Wider nye_tol: more permissive initial segment filtering
    - Returns total tracked fraction for quality assessment

    Args:
        z: Complex time series at this depth bin [n_times]
        time_days: Time vector (days)
        expected_v: Nye-predicted velocity at this depth (m/yr)
        lambdac: Center wavelength in ice (m)
        amp_factor: Amplitude threshold factor
        min_pts: Minimum elevated points per segment
        nye_tol: Maximum |v_segment - v_nye| for Nye consistency
        gap_tolerance: Max consecutive sub-threshold points to bridge

    Returns:
        displacement: Full-record displacement (m), NaN in gaps
        velocity: Fitted velocity (m/yr)
        r_squared: R² of the linear fit
        n_segments: Number of segments used
        segment_info: List of dicts with segment details
        tracked_frac: Fraction of record with valid tracking
    """
    wrap_period = lambdac / 2.0

    # Find elevated segments with gap bridging
    elevated, segments, threshold = find_elevated_segments(
        z, amp_factor=amp_factor, min_pts=min_pts,
        gap_tolerance=gap_tolerance,
    )

    if not segments:
        return None, np.nan, 0.0, 0, [], 0.0

    # Evaluate each segment
    good_segs = []
    for s_start, s_end in segments:
        phi_seg = np.angle(z[s_start:s_end])
        phi_unwrap = np.unwrap(phi_seg)
        t_seg = time_days[s_start:s_end]
        disp_seg = phi_unwrap * lambdac / (4 * np.pi)

        # Linear fit
        slope, intercept, r_val, _, _ = stats.linregress(t_seg, disp_seg)
        v_seg = slope * 365.25
        r2_seg = r_val ** 2

        if abs(v_seg - expected_v) < nye_tol and r2_seg > 0.2:
            good_segs.append({
                'start': s_start,
                'end': s_end,
                'velocity': v_seg,
                'r2': r2_seg,
                'disp': disp_seg,
                'time': t_seg,
                'duration': float(t_seg[-1] - t_seg[0]),
                'n_elevated': int(np.sum(elevated[s_start:s_end])),
            })

    if not good_segs:
        return None, np.nan, 0.0, 0, [], 0.0

    # Sort by start time
    good_segs.sort(key=lambda s: s['start'])

    # Build displacement array with Nye-bridged stitching
    displacement = np.full(len(time_days), np.nan)

    first = good_segs[0]
    base_disp = first['disp'] - first['disp'][0]
    displacement[first['start']:first['end']] = base_disp

    for k in range(1, len(good_segs)):
        prev = good_segs[k - 1]
        curr = good_segs[k]

        last_disp = displacement[prev['end'] - 1]
        last_time = time_days[prev['end'] - 1]
        dt = time_days[curr['start']] - last_time
        nye_predicted = last_disp + (expected_v / 365.25) * dt

        curr_disp_rel = curr['disp'] - curr['disp'][0]
        raw_offset = nye_predicted - curr_disp_rel[0]
        n_wraps = round(raw_offset / wrap_period)
        offset = n_wraps * wrap_period
        displacement[curr['start']:curr['end']] = curr_disp_rel + offset

    # Fit velocity to all tracked points
    valid = ~np.isnan(displacement)
    n_valid = int(np.sum(valid))
    if n_valid < 15:
        return None, np.nan, 0.0, 0, [], 0.0

    slope, intercept, r_val, _, _ = stats.linregress(
        time_days[valid], displacement[valid]
    )
    velocity = slope * 365.25
    r_squared = r_val ** 2
    tracked_frac = n_valid / len(time_days)

    segment_info = [{
        'start_idx': s['start'],
        'end_idx': s['end'],
        'velocity': s['velocity'],
        'r2': s['r2'],
        'duration': s['duration'],
        'n_elevated': s['n_elevated'],
    } for s in good_segs]

    return displacement, velocity, r_squared, len(good_segs), segment_info, tracked_frac


def compute_neighbor_coherence(
    complex_data: np.ndarray,
    bin_idx: int,
    displacement: np.ndarray,
    time_days: np.ndarray,
    lambdac: float,
    n_neighbors: int = 2,
) -> float:
    """
    Check if neighboring bins show consistent phase evolution.

    A real layer should show similar displacement in adjacent bins.
    Returns a coherence score (0-1) based on cross-correlation of
    phase evolution with neighbors.
    """
    n_bins = complex_data.shape[0]
    valid = ~np.isnan(displacement)
    if np.sum(valid) < 20:
        return 0.0

    ref_disp = displacement[valid]
    scores = []

    for offset in range(-n_neighbors, n_neighbors + 1):
        if offset == 0:
            continue
        ni = bin_idx + offset
        if ni < 0 or ni >= n_bins:
            continue

        z_n = complex_data[ni, :]
        phi_n = np.angle(z_n)
        phi_n_unwrap = np.unwrap(phi_n)
        disp_n = phi_n_unwrap * lambdac / (4 * np.pi)
        disp_n_valid = disp_n[valid]

        # Remove mean and correlate
        ref_dm = ref_disp - np.mean(ref_disp)
        nbr_dm = disp_n_valid - np.mean(disp_n_valid)
        denom = np.sqrt(np.sum(ref_dm ** 2) * np.sum(nbr_dm ** 2))
        if denom > 0:
            corr = np.sum(ref_dm * nbr_dm) / denom
            scores.append(max(0, corr))

    return float(np.mean(scores)) if scores else 0.0


def detect_deep_layers_v2(
    data_path: str,
    nye_intercept: float = 0.0453,
    nye_slope: float = 0.000595,
    min_depth: float = 785.0,
    max_depth: float = 1094.0,
    amp_factor: float = 1.2,
    min_pts: int = 15,
    nye_tol: float = 0.20,
    gap_tolerance: int = 10,
    min_r2: float = 0.60,
    max_v_residual: float = 0.12,
    min_cluster_bins: int = 2,
    merge_distance: float = 3.0,
    svd_components: int = 100,
    use_svd: bool = True,
    multi_bin_width: int = 1,
    use_neighbor_coherence: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Improved deep layer detection pipeline.

    Improvements:
    1. Optional SVD denoising of the deep complex data
    2. Gap-tolerant segment finding (bridge short sub-threshold windows)
    3. Multi-bin coherent averaging before tracking
    4. Neighbor coherence scoring for quality assessment
    5. More permissive initial parameters, stricter final filtering

    Args:
        data_path: Path to ImageP2_python.mat
        nye_intercept, nye_slope: Nye model parameters
        min_depth, max_depth: Depth range to scan
        amp_factor: Amplitude threshold (lower for SVD data, e.g. 1.2)
        min_pts: Minimum elevated points per segment
        nye_tol: Max |v_seg - v_nye| for initial segment acceptance
        gap_tolerance: Max consecutive sub-threshold points to bridge
        min_r2: Minimum R² for final acceptance
        max_v_residual: Maximum |v - v_nye| for final acceptance
        min_cluster_bins: Minimum adjacent bins to form a layer
        merge_distance: Max distance (m) for clustering
        svd_components: Number of SVD components to keep
        use_svd: Whether to apply SVD denoising
        multi_bin_width: Half-width for multi-bin averaging (0 = single bin)
        use_neighbor_coherence: Compute neighbor coherence scores
        verbose: Print progress

    Returns:
        dict with detection results for saving to JSON
    """
    t0 = time_mod.time()

    if verbose:
        print("=" * 70)
        print("DEEP LAYER DETECTION v2 — SVD-Enhanced Segment-Stitched Tracking")
        print("=" * 70)

    # ── Load data ──
    if verbose:
        print(f"\n[1/6] Loading data from {data_path}...")
    mat = loadmat(data_path)
    raw_complex = np.array(mat['RawImageComplex'])
    Rcoarse = np.array(mat['Rcoarse']).flatten()
    time_days = np.array(mat['TimeInDays']).flatten()
    lambdac = float(mat.get('lambdac', np.array([0.5608])).flatten()[0])
    del mat

    n_bins, n_times = raw_complex.shape
    if verbose:
        print(f"  Data: {n_bins} bins x {n_times} times")
        print(f"  Depth: {Rcoarse[0]:.1f} - {Rcoarse[-1]:.1f} m, λ_c = {lambdac:.4f} m")

    # ── Extract deep region ──
    idx_start = np.searchsorted(Rcoarse, min_depth)
    idx_end = np.searchsorted(Rcoarse, max_depth)
    deep_bins = np.arange(idx_start, idx_end)
    n_deep = len(deep_bins)

    if verbose:
        print(f"  Deep region: {n_deep} bins ({Rcoarse[idx_start]:.1f} - {Rcoarse[idx_end-1]:.1f} m)")

    # ── SVD denoising ──
    if use_svd:
        if verbose:
            print(f"\n[2/6] SVD denoising (keeping {svd_components} components)...")
        deep_complex = raw_complex[idx_start:idx_end, :].copy()

        U, S, Vh = np.linalg.svd(deep_complex, full_matrices=False)
        total_energy = np.sum(S ** 2)
        n_keep = min(svd_components, len(S))
        kept_energy = np.sum(S[:n_keep] ** 2)

        S_filt = np.zeros_like(S)
        S_filt[:n_keep] = S[:n_keep]
        deep_denoised = U @ np.diag(S_filt) @ Vh

        if verbose:
            print(f"  Energy retained: {kept_energy/total_energy*100:.1f}%")
            print(f"  Singular value ratio σ₁/σ_{n_keep}: {S[0]/S[n_keep-1]:.1f}")
            amp_before = 20 * np.log10(np.mean(np.abs(deep_complex)) + 1e-30)
            amp_after = 20 * np.log10(np.mean(np.abs(deep_denoised)) + 1e-30)
            print(f"  Mean amplitude: {amp_before:.1f} dB → {amp_after:.1f} dB")

        del U, S, Vh, S_filt
        # Replace deep region in full array for neighbor coherence computation
        raw_complex_deep = deep_denoised
    else:
        if verbose:
            print(f"\n[2/6] SVD denoising: SKIPPED")
        raw_complex_deep = raw_complex[idx_start:idx_end, :]

    # ── Scan deep bins ──
    if verbose:
        print(f"\n[3/6] Scanning {n_deep} bins...")
        print(f"  Parameters: amp_factor={amp_factor}, min_pts={min_pts}, "
              f"gap_tolerance={gap_tolerance}, nye_tol={nye_tol}")

    candidate_bins = []
    for i in range(n_deep):
        depth = Rcoarse[deep_bins[i]]
        v_nye = nye_velocity(depth, nye_intercept, nye_slope)

        # Get signal for this bin (optionally multi-bin averaged)
        if multi_bin_width > 0:
            z = multi_bin_average(raw_complex_deep, i, half_width=multi_bin_width)
        else:
            z = raw_complex_deep[i, :]

        # Try tracking
        result = segment_stitched_tracking_v2(
            z, time_days, v_nye, lambdac,
            amp_factor=amp_factor, min_pts=min_pts,
            nye_tol=nye_tol, gap_tolerance=gap_tolerance,
        )
        disp, vel, r2, n_segs, seg_info, tracked_frac = result

        if disp is not None and r2 >= min_r2:
            n_tracked = int(np.sum(~np.isnan(disp)))

            # Compute neighbor coherence
            if use_neighbor_coherence:
                ncoh = compute_neighbor_coherence(
                    raw_complex_deep, i, disp, time_days, lambdac,
                )
            else:
                ncoh = np.nan

            candidate_bins.append({
                'bin_idx': int(deep_bins[i]),
                'local_idx': i,
                'depth': float(depth),
                'velocity': float(vel),
                'nye_velocity': float(v_nye),
                'r2': float(r2),
                'n_tracked': n_tracked,
                'n_segs': n_segs,
                'tracked_frac': float(tracked_frac),
                'neighbor_coherence': float(ncoh),
                'seg_info': seg_info,
                'displacement': disp,  # Keep for later comparison
            })

        if verbose and (i + 1) % 500 == 0:
            print(f"    {i+1}/{n_deep} scanned, {len(candidate_bins)} candidates")

    if verbose:
        print(f"  → {len(candidate_bins)} bins with R² ≥ {min_r2}")

    # ── Filter stitching errors ──
    if verbose:
        print(f"\n[4/6] Filtering stitching errors and velocity outliers...")

    clean = []
    n_stitch_err = 0
    n_vel_outlier = 0
    for c in candidate_bins:
        if c['n_segs'] > 1 and abs(c['velocity'] - c['nye_velocity']) > nye_tol:
            n_stitch_err += 1
            continue
        if abs(c['velocity'] - c['nye_velocity']) > max_v_residual:
            n_vel_outlier += 1
            continue
        clean.append(c)

    if verbose:
        print(f"  Removed {n_stitch_err} stitch errors, {n_vel_outlier} velocity outliers")
        print(f"  → {len(clean)} clean candidates")

    # ── Cluster into layers ──
    if verbose:
        print(f"\n[5/6] Clustering adjacent bins (merge distance: {merge_distance} m)...")

    clean.sort(key=lambda c: c['depth'])

    clusters = []
    i = 0
    while i < len(clean):
        group = [clean[i]]
        j = i + 1
        while j < len(clean) and clean[j]['depth'] - group[-1]['depth'] < merge_distance:
            group.append(clean[j])
            j += 1
        clusters.append(group)
        i = j

    if verbose:
        print(f"  → {len(clusters)} clusters")

    # ── Select best bin per cluster and assign quality tiers ──
    if verbose:
        print(f"\n[6/6] Selecting best bin per cluster...")

    layers = []
    for group in clusters:
        if len(group) < min_cluster_bins:
            continue

        # Pick bin with best R² (or highest tracked fraction as tiebreaker)
        best = max(group, key=lambda c: (c['r2'], c['tracked_frac']))

        dv = abs(best['velocity'] - best['nye_velocity'])
        ncoh = best['neighbor_coherence']

        # Quality tiers:
        # Tier 1: excellent R², close to Nye, good neighbor coherence
        # Tier 2: good R², reasonable Nye agreement
        # Tier 3: adequate
        if best['r2'] > 0.90 and dv < 0.05 and (np.isnan(ncoh) or ncoh > 0.3):
            tier = 1
        elif best['r2'] > 0.80 and dv < 0.10:
            tier = 2
        else:
            tier = 3

        seg_durations = [s['duration'] for s in best['seg_info']]
        total_tracked_time = sum(seg_durations)

        layers.append({
            'depth_m': best['depth'],
            'bin_idx': best['bin_idx'],
            'velocity_m_yr': best['velocity'],
            'nye_velocity_m_yr': best['nye_velocity'],
            'r_squared': best['r2'],
            'n_tracked_pts': best['n_tracked'],
            'n_segments': best['n_segs'],
            'tracked_fraction': best['tracked_frac'],
            'total_tracked_days': total_tracked_time,
            'neighbor_coherence': ncoh,
            'segment_durations_days': seg_durations,
            'quality_tier': tier,
            'cluster_size': len(group),
            'tracking_mode': 'deep_segment_stitched',
        })

    elapsed = time_mod.time() - t0

    # Summary
    n_t1 = sum(1 for l in layers if l['quality_tier'] == 1)
    n_t2 = sum(1 for l in layers if l['quality_tier'] == 2)
    n_t3 = sum(1 for l in layers if l['quality_tier'] == 3)

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"DEEP LAYER DETECTION v2 COMPLETE  ({elapsed:.1f} s)")
        print(f"{'=' * 70}")
        print(f"  Total layers: {len(layers)}")
        print(f"    Tier 1 (R²>0.90, |Δv|<0.05): {n_t1}")
        print(f"    Tier 2 (R²>0.80, |Δv|<0.10): {n_t2}")
        print(f"    Tier 3 (remaining):            {n_t3}")
        if layers:
            print(f"\n  {'Depth':>7} {'v':>7} {'Nye':>7} {'Δv':>6} {'R²':>7} "
                  f"{'Pts':>5} {'Segs':>5} {'Frac':>6} {'NCoh':>6} {'Tier':>5} {'Clust':>5}")
            for l in layers:
                dv_l = abs(l['velocity_m_yr'] - l['nye_velocity_m_yr'])
                stars = '★' * (4 - l['quality_tier'])
                ncoh_str = f"{l['neighbor_coherence']:.2f}" if not np.isnan(l['neighbor_coherence']) else "  N/A"
                print(f"  {l['depth_m']:7.1f} {l['velocity_m_yr']:7.3f} {l['nye_velocity_m_yr']:7.3f} "
                      f"{dv_l:6.3f} {l['r_squared']:7.4f} {l['n_tracked_pts']:5d} "
                      f"{l['n_segments']:5d} {l['tracked_fraction']:6.3f} "
                      f"{ncoh_str:>6} {stars:>5} {l['cluster_size']:5d}")

    # Build output dict
    result = {
        'method': 'svd_enhanced_segment_stitched_tracking_v2',
        'description': (
            f'Deep layers detected via SVD-enhanced amplitude-gated segment-stitched '
            f'phase tracking. SVD denoising ({"enabled" if use_svd else "disabled"}, '
            f'{svd_components} components) removes incoherent noise before tracking. '
            f'Gap bridging (tolerance={gap_tolerance} pts) produces longer segments. '
            f'Multi-bin averaging (width={2*multi_bin_width+1}) boosts SNR.'
        ),
        'parameters': {
            'nye_intercept': nye_intercept,
            'nye_slope': nye_slope,
            'depth_range_m': [min_depth, max_depth],
            'amp_factor': amp_factor,
            'min_pts': min_pts,
            'gap_tolerance': gap_tolerance,
            'nye_tol': nye_tol,
            'min_r2': min_r2,
            'max_v_residual': max_v_residual,
            'svd_components': svd_components,
            'use_svd': use_svd,
            'multi_bin_width': multi_bin_width,
            'merge_distance': merge_distance,
            'min_cluster_bins': min_cluster_bins,
        },
        'nye_model': {
            'intercept_m_yr': nye_intercept,
            'slope_per_yr': nye_slope,
            'formula': f'w(z) = {nye_intercept:.4f} + {nye_slope:.6f} * z',
        },
        'summary': {
            'n_layers': len(layers),
            'n_tier1': n_t1,
            'n_tier2': n_t2,
            'n_tier3': n_t3,
            'n_bins_scanned': n_deep,
            'n_candidates': len(candidate_bins),
            'elapsed_seconds': round(elapsed, 1),
        },
        'layers': layers,
    }

    return result


def save_results(result: dict, output_path: str) -> None:
    """Save results to JSON."""
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else int(x) if isinstance(x, np.integer) else x)
    print(f"\nSaved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Improved deep layer detection with SVD denoising and gap bridging'
    )
    parser.add_argument('--data', type=str, required=True,
                        help='Path to ImageP2_python.mat')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON file path')
    # Nye model
    parser.add_argument('--nye-intercept', type=float, default=0.0453)
    parser.add_argument('--nye-slope', type=float, default=0.000595)
    # Depth range
    parser.add_argument('--min-depth', type=float, default=785.0)
    parser.add_argument('--max-depth', type=float, default=1094.0)
    # Detection parameters
    parser.add_argument('--amp-factor', type=float, default=1.2,
                        help='Amplitude threshold factor (default: 1.2, lower than v1)')
    parser.add_argument('--min-pts', type=int, default=15,
                        help='Min elevated pts per segment (default: 15, lower than v1)')
    parser.add_argument('--gap-tolerance', type=int, default=10,
                        help='Max consecutive sub-threshold pts to bridge (default: 10)')
    parser.add_argument('--nye-tol', type=float, default=0.20,
                        help='Max |v_seg - v_nye| for segment acceptance (default: 0.20)')
    # Quality thresholds
    parser.add_argument('--min-r2', type=float, default=0.60)
    parser.add_argument('--max-v-residual', type=float, default=0.12)
    parser.add_argument('--min-cluster-bins', type=int, default=2)
    parser.add_argument('--merge-distance', type=float, default=3.0)
    # SVD
    parser.add_argument('--svd-components', type=int, default=100)
    parser.add_argument('--no-svd', action='store_true',
                        help='Disable SVD denoising')
    # Multi-bin
    parser.add_argument('--multi-bin-width', type=int, default=1,
                        help='Half-width for multi-bin averaging (0=disabled)')
    # Comparison mode
    parser.add_argument('--compare', action='store_true',
                        help='Run both with and without SVD for comparison')

    args = parser.parse_args()

    if args.compare:
        # Run without SVD
        print("\n" + "=" * 70)
        print("COMPARISON MODE: Running WITHOUT SVD first...")
        print("=" * 70)
        result_raw = detect_deep_layers_v2(
            data_path=args.data,
            nye_intercept=args.nye_intercept,
            nye_slope=args.nye_slope,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            amp_factor=1.3,  # Use original threshold for raw data
            min_pts=args.min_pts,
            gap_tolerance=args.gap_tolerance,
            nye_tol=args.nye_tol,
            min_r2=args.min_r2,
            max_v_residual=args.max_v_residual,
            min_cluster_bins=args.min_cluster_bins,
            merge_distance=args.merge_distance,
            svd_components=0,
            use_svd=False,
            multi_bin_width=0,
            use_neighbor_coherence=True,
        )
        out_raw = args.output.replace('.json', '_raw.json')
        save_results(result_raw, out_raw)

        print("\n" + "=" * 70)
        print("COMPARISON MODE: Running WITH SVD...")
        print("=" * 70)
        result_svd = detect_deep_layers_v2(
            data_path=args.data,
            nye_intercept=args.nye_intercept,
            nye_slope=args.nye_slope,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            amp_factor=args.amp_factor,
            min_pts=args.min_pts,
            gap_tolerance=args.gap_tolerance,
            nye_tol=args.nye_tol,
            min_r2=args.min_r2,
            max_v_residual=args.max_v_residual,
            min_cluster_bins=args.min_cluster_bins,
            merge_distance=args.merge_distance,
            svd_components=args.svd_components,
            use_svd=True,
            multi_bin_width=args.multi_bin_width,
            use_neighbor_coherence=True,
        )
        save_results(result_svd, args.output)

        # Print comparison
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        nr = result_raw['summary']
        ns = result_svd['summary']
        print(f"  {'':>25} {'Raw':>10} {'SVD':>10} {'Change':>10}")
        print(f"  {'Layers detected':>25} {nr['n_layers']:>10} {ns['n_layers']:>10} {ns['n_layers']-nr['n_layers']:>+10}")
        print(f"  {'Tier 1':>25} {nr['n_tier1']:>10} {ns['n_tier1']:>10} {ns['n_tier1']-nr['n_tier1']:>+10}")
        print(f"  {'Tier 2':>25} {nr['n_tier2']:>10} {ns['n_tier2']:>10} {ns['n_tier2']-nr['n_tier2']:>+10}")
        print(f"  {'Tier 3':>25} {nr['n_tier3']:>10} {ns['n_tier3']:>10} {ns['n_tier3']-nr['n_tier3']:>+10}")
        print(f"  {'Candidates':>25} {nr['n_candidates']:>10} {ns['n_candidates']:>10} {ns['n_candidates']-nr['n_candidates']:>+10}")

        # Show layers unique to SVD
        raw_depths = set(round(l['depth_m'], 1) for l in result_raw['layers'])
        svd_depths = set(round(l['depth_m'], 1) for l in result_svd['layers'])
        new_depths = svd_depths - raw_depths
        if new_depths:
            print(f"\n  New layers found only with SVD ({len(new_depths)}):")
            for l in result_svd['layers']:
                if round(l['depth_m'], 1) in new_depths:
                    print(f"    {l['depth_m']:.1f} m  v={l['velocity_m_yr']:.3f}  R²={l['r_squared']:.3f}")

    else:
        # Single run
        result = detect_deep_layers_v2(
            data_path=args.data,
            nye_intercept=args.nye_intercept,
            nye_slope=args.nye_slope,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            amp_factor=args.amp_factor,
            min_pts=args.min_pts,
            gap_tolerance=args.gap_tolerance,
            nye_tol=args.nye_tol,
            min_r2=args.min_r2,
            max_v_residual=args.max_v_residual,
            min_cluster_bins=args.min_cluster_bins,
            merge_distance=args.merge_distance,
            svd_components=args.svd_components,
            use_svd=not args.no_svd,
            multi_bin_width=args.multi_bin_width,
            use_neighbor_coherence=True,
        )
        save_results(result, args.output)


if __name__ == '__main__':
    main()
