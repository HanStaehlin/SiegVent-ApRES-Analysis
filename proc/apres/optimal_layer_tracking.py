#!/usr/bin/env python3
"""
Optimal Layer Tracking for ApRES Data

Uses a Dynamic Programming (Viterbi) global optimization 
approach to track internal ice layers across the depth-time 
echogram bounding the layer by a velocity prior (Nye / Phase-Slope).

Author: SiegVent2023 project
"""

import numpy as np
from scipy import signal, ndimage, stats
from scipy.io import loadmat, savemat
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Optional
import json
import argparse
import time

@dataclass
class OptimalTrackingResult:
    layer_depths: np.ndarray           # Mean depth of each layer (m)
    phase_timeseries: np.ndarray       # Phase vs time [n_layers, n_times] 
    range_timeseries: np.ndarray       # Fine range vs time [n_layers, n_times] (m)
    amplitude_timeseries: np.ndarray   # Amplitude vs time [n_layers, n_times]
    tracked_indices: np.ndarray        # Bin indices path [n_layers, n_times]
    time_days: np.ndarray              # Time vector (days)
    n_layers: int
    lambdac: float                     # Center wavelength (m)
    tracking_mode: np.ndarray          # 1 for all
    velocities_m_yr: np.ndarray        # Optimal velocity
    r_squared: np.ndarray              # Optimal R-squared


def load_apres_data(data_path: str) -> dict:
    print(f"Loading data from {data_path}...")
    mat = loadmat(data_path)
    
    if 'RawImageComplex' not in mat:
        raise ValueError("RawImageComplex must be present in the .mat file for phase-coherent Viterbi tracking.")
        
    raw_complex = np.array(mat['RawImageComplex'])
    Rcoarse = np.array(mat['Rcoarse']).flatten()
    time_days = np.array(mat['TimeInDays']).flatten()
    
    lambdac = 0.5608
    if 'lambdac' in mat:
         lambdac = float(mat['lambdac'].flatten()[0])

    print(f"  Data shape: {raw_complex.shape}")
    print(f"  Depths: {Rcoarse[0]:.1f}m to {Rcoarse[-1]:.1f}m")
    
    # Pre-compute Rfine for sub-mm phase later
    rfine = np.angle(raw_complex) * lambdac / (4 * np.pi)

    return {
        'raw_complex': raw_complex,
        'Rcoarse': Rcoarse,
        'rfine': rfine,
        'time_days': time_days,
        'lambdac': lambdac
    }


def svd_denoise_complex(raw_complex: np.ndarray, svd_components: int = 3, window_bins: int = 380) -> np.ndarray:
    """Apply local window SVD to denoise the complex echogram."""
    print(f"Applying local SVD denoising (k={svd_components}, window={window_bins} bins)...")
    n_bins, n_times = raw_complex.shape
    denoised = np.zeros_like(raw_complex)
    
    # Process in chunks to avoid mixing completely different depth regimes
    step = window_bins // 2
    starts = list(range(0, n_bins, step))
    
    # Weighting window to smoothly blend overlapping chunks
    w = np.hanning(window_bins + 2)[1:-1]
    weights = np.zeros(n_bins)
    
    for i in starts:
        end = min(i + window_bins, n_bins)
        chunk = raw_complex[i:end, :]
        actual_len = chunk.shape[0]
        
        # SVD
        U, S, Vh = np.linalg.svd(chunk, full_matrices=False)
        S_trunc = np.zeros_like(S)
        k = min(svd_components, len(S))
        S_trunc[:k] = S[:k]
        chunk_denoised = U @ np.diag(S_trunc) @ Vh
        
        # Blend
        w_chunk = w[:actual_len] if actual_len < window_bins else w
        for t in range(n_times):
            denoised[i:end, t] += chunk_denoised[:, t] * w_chunk
        weights[i:end] += w_chunk
        
    # Normalize weights
    weights[weights == 0] = 1.0
    for t in range(n_times):
         denoised[:, t] /= weights
         
    return denoised
    

def nye_velocity_model(z: np.ndarray) -> np.ndarray:
    """Nye vertical velocity model (m/yr). Using values from visual_app defaults."""
    w_s = 0.0453 
    eps_zz = 0.000595
    return w_s + eps_zz * z


def detect_seed_layers(denoised_amp: np.ndarray, Rcoarse: np.ndarray, time_years: np.ndarray = None, min_depth: float = 200, max_depth: float = 1000) -> np.ndarray:
    """Find a robust set of starting layer bins by finding peaks in the time-averaged amplitude.
    If time_years is provided, layers are first shifted back to their t=0 position using the 
    Nye velocity model so the peaks don't smear out over time."""
    mask = (Rcoarse >= min_depth) & (Rcoarse <= max_depth)
    
    if time_years is not None:
        n_bins, n_times = denoised_amp.shape
        dz = Rcoarse[1] - Rcoarse[0]
        v_cruise = nye_velocity_model(Rcoarse)
        
        # Shift the data at time t back to its t=0 original depth
        shift_bins = np.round(np.outer(v_cruise, time_years) / dz).astype(int)
        
        bin_idx = np.arange(n_bins)[:, None]
        src_idx = bin_idx + shift_bins
        
        valid = (src_idx >= 0) & (src_idx < n_bins)
        src_idx = np.clip(src_idx, 0, n_bins - 1)
        
        time_idx = np.arange(n_times)
        aligned_amp = np.where(valid, denoised_amp[src_idx, time_idx], 0.0)
        mean_amp = np.mean(aligned_amp, axis=1)
    else:
        mean_amp = np.mean(denoised_amp, axis=1)

    
    mean_db = 10 * np.log10(mean_amp**2 + 1e-30)
    noise_floor = np.percentile(mean_db[mask], 10)
    
    peaks, _ = signal.find_peaks(
        mean_db[mask], 
        height=noise_floor + 5.0, # 5 dB SNR minimum
        prominence=2.0,
        distance=50  # ~2.5m spacing min
    )
    
    # Map back to absolute indices
    mask_indices = np.where(mask)[0]
    return mask_indices[peaks]


def viterbi_layer_tracking(
    denoised_abs: np.ndarray, 
    time_years: np.ndarray,
    Rcoarse: np.ndarray,
    seed_index: int, 
    search_half_width: int = 80,
    max_velocity_m_yr: float = 4.0,
    slope_weight: float = 200.0,
    jump_penalty_factor: float = 10.0,
    max_jump_bins: int = 1
) -> np.ndarray:
    """
    Dynamic Programming optimal path tracking for a single layer.
    
    Energy = Amplitude_Scale + Phase_Coherence_Scale - Slope_Penalty
    """
    n_bins, n_times = denoised_abs.shape
    dt_years = np.diff(time_years)
    dz = Rcoarse[1] - Rcoarse[0]
    
    # Limit tracking tube around the seed to save computation
    tube_min = max(0, seed_index - search_half_width)
    tube_max = min(n_bins - 1, seed_index + search_half_width)
    n_tube = tube_max - tube_min + 1
    
    # Expected slope bins per time step from Nye
    v_nye = nye_velocity_model(Rcoarse[tube_min:tube_max+1])
    
    # Initialize DP tables
    cost = np.full((n_tube, n_times), -np.inf)
    backtrack = np.zeros((n_tube, n_times), dtype=int)
    
    # Initial state cost: heavily bias to start at the seed index
    cost[:, 0] = -1e6
    center_local = seed_index - tube_min
    
    # Soft initialization around the seed
    for idx_local in range(n_tube):
        dist = abs(idx_local - center_local)
        cost[idx_local, 0] = - (dist**2) * 5.0  # Gaussian-like initialization penalty
    
    # Normalize image amplitudes in the tube for scoring
    amp_tube = denoised_abs[tube_min:tube_max + 1, :]
    amp_tube_norm = amp_tube / (np.percentile(amp_tube, 95) + 1e-9)

    # Forward pass (Viterbi)
    for t in range(1, n_times):
        dt = dt_years[t-1]
        
        # Max search window in bins per time step (enforce max physical jump)
        # Ice layers don't jump more than ~1 bin suddenly.
        max_shift = 3
        
        for curr_idx in range(n_tube):
            # Calculate expected shift from physics
            expected_shift_bins = (v_nye[curr_idx] * dt) / dz
            
            best_prev_cost = -np.inf
            best_prev_idx = -1
            
            # Check allowed previous states
            for shift in range(-max_shift, max_shift + 1):
                prev_idx = curr_idx + shift
                
                if 0 <= prev_idx < n_tube:
                    step = curr_idx - prev_idx
                    v_step = (step * dz) / dt if dt > 0 else 0
                    
                    # 1. Hard Slope Constraint: Penalize velocities > max_velocity_m_yr
                    # We use a very steep penalty for exceeding the limit
                    excess_v = max(0, abs(v_step) - max_velocity_m_yr)
                    slope_hard_penalty = 1000.0 * excess_v
                    
                    # 2. Transition penalty (L2 distance from expected physical shift)
                    # slope_weight (default 200.0) makes it trust the cruising direction
                    slope_soft_penalty = slope_weight * (step - expected_shift_bins)**2 
                    
                    # 3. Sudden Jump Penalty: Heavily penalize jumping more than `max_jump_bins`
                    # away from the expected physical shift in a single time step.
                    abs_shift_dev = abs(step - expected_shift_bins)
                    if abs_shift_dev > max_jump_bins:
                        jump_hard_penalty = 5000.0 * (abs_shift_dev - max_jump_bins)
                    else:
                        jump_hard_penalty = 0.0
                    
                    transition_cost = cost[prev_idx, t-1] - slope_soft_penalty - slope_hard_penalty - jump_hard_penalty
                    
                    if transition_cost > best_prev_cost:
                        best_prev_cost = transition_cost
                        best_prev_idx = prev_idx
            
            # Data term: High amplitude drives the path
            data_score = 10.0 * amp_tube_norm[curr_idx, t]
            
            cost[curr_idx, t] = best_prev_cost + data_score
            backtrack[curr_idx, t] = best_prev_idx
            
    # Backward pass
    path_local = np.zeros(n_times, dtype=int)
    path_local[-1] = np.argmax(cost[:, -1])
    
    for t in range(n_times - 2, -1, -1):
        path_local[t] = backtrack[path_local[t+1], t+1]
        
    return path_local + tube_min


def unwrap_range_timeseries_robust(
    total_range: np.ndarray,
    lambdac: float = 0.5608,
    time_years: Optional[np.ndarray] = None,
    amp_timeseries: Optional[np.ndarray] = None,
    expected_v_m_yr: float = 0.0
) -> np.ndarray:
    """
    Robustly unwrap phase jumps by first detrending with the expected cruising velocity.
    This virtually halts phase wrapping in the residual. We also use amplitude to 'coast'
    through regions of signal fading.
    """
    wrap_period = lambdac / 2.0
    n = len(total_range)
    
    if n < 2:
        return total_range.copy()
        
    if time_years is None:
        time_years = np.arange(n) / 365.25

    # Normalize amplitude to identify fades
    trust_phase = np.ones(n)
    if amp_timeseries is not None:
        amp_smooth = np.convolve(amp_timeseries, np.ones(5)/5.0, mode='same')
        p90 = np.percentile(amp_smooth, 90)
        if p90 > 0:
            trust_phase = amp_smooth / p90

    # 1. Calculate Expected Path (Detrending line)
    expected_range = expected_v_m_yr * time_years
    
    # 2. Detrend the input wrapped range
    residual_wrapped = total_range - expected_range
    
    # 3. Unwrap the residual
    residual_unwrapped = np.zeros(n)
    residual_unwrapped[0] = residual_wrapped[0]
    
    for i in range(1, n):
        if trust_phase[i] < 0.3:
            # Signal has faded into noise. Do not integrate random phase jumps.
            # Coast the residual entirely (meaning the layer moves EXACTLY at expected_v_m_yr)
            residual_unwrapped[i] = residual_unwrapped[i-1]
        else:
            # Signal is strong. Find the period 'k' that minimizes the jump from the previous residual point
            k = np.round((residual_unwrapped[i-1] - residual_wrapped[i]) / wrap_period)
            residual_unwrapped[i] = residual_wrapped[i] + k * wrap_period
            
    # 4. Retrend to get the final physical unwrapped range
    unwrapped_range = residual_unwrapped + expected_range
    
    return unwrapped_range - unwrapped_range[0]


def track_optimal_layers(data: dict, min_snr: float = 10.0, min_r2: float = 0.5, max_amp_cv: float = 0.45, tube_width: int = 80, slope_weight: float = 200.0, min_depth: float = 50.0) -> OptimalTrackingResult:
    print(f"Tracking layers using Optimal Dynamic Programming (min_snr={min_snr}dB, min_r2={min_r2}, max_amp_cv={max_amp_cv}, tube_width={tube_width}, slope_weight={slope_weight}, min_depth={min_depth}m)...")
    
    # 1. Denoise
    denoised_complex = svd_denoise_complex(data['raw_complex'], svd_components=3)
    denoised_abs = np.abs(denoised_complex)
    
    # Calculate time in years
    time_years = (data['time_days'] - data['time_days'][0]) / 365.25
    
    # 2. Find Seed Layers 
    # Use perfectly aligned phase stacked across the region
    seed_indices = detect_seed_layers(denoised_abs, data['Rcoarse'], time_years=time_years, min_depth=min_depth, max_depth=1200)
    n_seeds = len(seed_indices)
    print(f"Detected {n_seeds} high-SNR seed layers to track.")
    
    # Estimate noise floor for SNR calculation
    # Use 10th percentile of the amplitude squared across all bins/times as noise power
    noise_power = np.percentile(denoised_abs**2, 10)
    
    # 3. Track each seed

    
    all_tracked_indices = []
    all_phase_ts = []
    all_range_ts = []
    all_amp_ts = []
    all_depths = []
    all_velocities = []
    all_r2 = []
    all_path_snr = []
    all_amp_cv = []
    
    # Targeting layers: 745, 608, 818, 778 for diagnostic output
    target_depths = [745, 608, 818, 778]

    
    for i, seed in enumerate(seed_indices):
        if (i+1) % 10 == 0:
            print(f"  Tracking layer {i+1}/{n_seeds} (Depth ~{data['Rcoarse'][seed]:.1f}m)...")
            
        path = viterbi_layer_tracking(
            denoised_abs, 
            time_years, 
            data['Rcoarse'], 
            seed,
            max_velocity_m_yr=4.0,
            search_half_width=tube_width,
            slope_weight=slope_weight
        )
        
        # Extract timeseries
        amp_vals = np.zeros(len(data['time_days']))
        range_vals = np.zeros(len(data['time_days']))
        
        for t in range(len(data['time_days'])):
            idx = path[t]
            amp_vals[t] = np.abs(data['raw_complex'][idx, t])
            total_r = data['Rcoarse'][idx] + data['rfine'][idx, t]
            range_vals[t] = total_r

        # Robust Unwrap with Amplitude-weighted momentum
        depth = data['Rcoarse'][seed]
        expected_v = nye_velocity_model(depth)
        
        unwrapped_drift = unwrap_range_timeseries_robust(
            range_vals, 
            lambdac=data['lambdac'],
            time_years=time_years,
            amp_timeseries=amp_vals,
            expected_v_m_yr=expected_v
        )
        
        # Linear Fit
        p = np.polyfit(time_years, unwrapped_drift, 1)
        velocity = p[0]
        y_fit = p[0] * time_years + p[1]
        ss_res = np.sum((unwrapped_drift - y_fit)**2)
        ss_tot = np.sum((unwrapped_drift - np.mean(unwrapped_drift))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Calculate Path SNR (dB)
        avg_power = np.mean(amp_vals**2)
        path_snr = 10 * np.log10(avg_power / noise_power) if noise_power > 0 else 0
        
        # Calculate Amplitude Stability (Coefficient of Variation)
        # Real layers have stable amplitude, noise flickers (high CV)
        mean_amp = np.mean(amp_vals)
        amp_cv = np.std(amp_vals) / mean_amp if mean_amp > 0 else 1.0
        


        # Quality Filter
        # max_amp_cv default to 0.5 (noise is usually > 0.6)
        if path_snr >= min_snr and r2 >= min_r2 and amp_cv <= max_amp_cv:
            all_tracked_indices.append(path)
            all_phase_ts.append(np.zeros_like(amp_vals)) # Placeholder
            all_range_ts.append(unwrapped_drift)
            all_amp_ts.append(amp_vals)
            all_depths.append(depth)
            all_velocities.append(velocity)
            all_r2.append(r2)
            all_path_snr.append(path_snr)
            all_amp_cv.append(amp_cv)


    n_final = len(all_depths)
    print(f"Filtering complete: {n_final}/{n_seeds} layers accepted (SNR >= {min_snr}dB, R² >= {min_r2}, AmpCV <= {max_amp_cv}).")

    return OptimalTrackingResult(
        layer_depths=np.array(all_depths),
        phase_timeseries=np.array(all_phase_ts),
        range_timeseries=np.array(all_range_ts),
        amplitude_timeseries=np.array(all_amp_ts),
        tracked_indices=np.array(all_tracked_indices),
        time_days=data['time_days'],
        n_layers=n_final,
        lambdac=data['lambdac'],
        tracking_mode=np.ones(n_final, dtype=int),
        velocities_m_yr=np.array(all_velocities),
        r_squared=np.array(all_r2)
    )

def main():
    parser = argparse.ArgumentParser(description='Optimal Dynamic Programming Layer Tracking')
    parser.add_argument('--data', type=str, default='data/apres/ImageP2_python.mat')
    parser.add_argument('--output', type=str, default='output/apres/hybrid/optimal_tracking')
    parser.add_argument('--min-snr', type=float, default=10.0, help='Minimum path-average SNR (dB) to keep layer')
    parser.add_argument('--min-r2', type=float, default=0.5, help='Minimum R-squared of velocity fit to keep layer')
    parser.add_argument('--max-amp-cv', type=float, default=0.45, help='Maximum amplitude coefficient of variation (std/mean)')
    parser.add_argument('--tube-width', type=int, default=80, help='Half-width of tracking tube (bins)')
    parser.add_argument('--slope-weight', type=float, default=200.0, help='Weight of the physical slope prior')
    parser.add_argument('--min-depth', type=float, default=50.0, help='Minimum depth (m) to start looking for layers')
    
    args = parser.parse_args()
    
    t0 = time.time()
    data = load_apres_data(args.data)
    
    result = track_optimal_layers(
        data, 
        min_snr=args.min_snr, 
        min_r2=args.min_r2, 
        max_amp_cv=args.max_amp_cv,
        tube_width=args.tube_width,
        slope_weight=args.slope_weight,
        min_depth=args.min_depth
    )
    
    # Save 
    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_base = args.output
    print(f"Saving to {out_base} [.mat, .json]...")
    
    # JSON for the summarize UI
    json_data = {
        'n_layers': int(result.n_layers),
        'lambdac': float(result.lambdac),
        'method': 'optimal_viterbi',
        'layers': []
    }
    
    for i in range(result.n_layers):
        l_dict = {
            'depth_m': float(result.layer_depths[i]),
            'tracking_mode': int(result.tracking_mode[i])
        }
        json_data['layers'].append(l_dict)
        
    with open(f"{out_base}.json", 'w') as f:
        json.dump(json_data, f, indent=2)
        
    # MAT for full timeseries
    mat_data = {
        'layer_depths': result.layer_depths,
        'phase_timeseries': result.phase_timeseries,
        'range_timeseries': result.range_timeseries,
        'amplitude_timeseries': result.amplitude_timeseries,
        'tracked_indices': result.tracked_indices,
        'tracking_mode': result.tracking_mode,
        'time_days': result.time_days,
        'lambdac': result.lambdac,
        'velocities_m_yr': result.velocities_m_yr,
        'r_squared': result.r_squared
    }
    savemat(f"{out_base}.mat", mat_data)
    
    print(f"Tracking complete in {time.time()-t0:.1f} seconds.")


if __name__ == '__main__':
    main()
