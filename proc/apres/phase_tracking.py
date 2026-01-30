"""
Phase Tracking Module for ApRES Internal Ice Layer Analysis

This module extracts and tracks the phase evolution of detected internal
ice layers over time, applying robust phase unwrapping.

Based on:
- Brennan et al. (2014) - Phase-sensitive FMCW radar
- Summers et al. (2021) - Velocity profile extraction

Author: SiegVent2023 project
"""

import numpy as np
from scipy import signal, ndimage
from scipy.io import loadmat, savemat
from dataclasses import dataclass
from typing import Tuple, Optional, List, Union
import json


@dataclass
class PhaseTrackingResult:
    """Container for phase tracking results."""
    layer_depths: np.ndarray           # Depth of each layer (m)
    phase_timeseries: np.ndarray       # Phase vs time [n_layers, n_times]
    range_timeseries: np.ndarray       # Fine range vs time [n_layers, n_times] (m)
    amplitude_timeseries: np.ndarray   # Amplitude vs time [n_layers, n_times]
    time_days: np.ndarray              # Time vector (days)
    n_layers: int
    lambdac: float                     # Center wavelength (m)


def correct_wrap_segments(
    range_change: np.ndarray,
    time_days: np.ndarray,
    wrap_period: float,
    min_days: float = 3.0,
    max_days: float = 5.0,
    jump_m: float = 0.20,
    tolerance_m: float = 0.06,
    max_iters: int = 5,
) -> np.ndarray:
    """Correct spurious wrap-like jumps over 3–5 day windows by ±λ/2 shifts."""
    corrected = range_change.copy()
    n = len(corrected)
    if n < 2:
        return corrected

    for _ in range(max_iters):
        changed = False
        for i in range(n - 1):
            t_start = time_days[i]
            t_end = t_start + max_days
            j_candidates = np.where((time_days > t_start + min_days) & (time_days <= t_end))[0]
            if j_candidates.size == 0:
                continue
            for j in j_candidates:
                delta = corrected[j] - corrected[i]
                if abs(delta) >= jump_m and abs(abs(delta) - wrap_period) <= tolerance_m:
                    shift = -np.sign(delta) * wrap_period
                    corrected[j:] = corrected[j:] + shift
                    changed = True
                    break
            if changed:
                break
        if not changed:
            break

    return corrected


def correct_large_jumps(
    range_change: np.ndarray,
    wrap_period: float,
    jump_threshold: float = 0.20,
    max_iters: int = 5,
) -> np.ndarray:
    """Correct large single-step jumps by shifting subsequent values by ±λ/2."""
    corrected = range_change.copy()
    for _ in range(max_iters):
        diffs = np.diff(corrected)
        if diffs.size == 0:
            break
        idx = np.argmax(np.abs(diffs))
        if abs(diffs[idx]) < jump_threshold:
            break
        shift = -np.sign(diffs[idx]) * wrap_period
        corrected[idx + 1:] = corrected[idx + 1:] + shift
    return corrected


def correct_lambda2_jumps(
    range_change: np.ndarray,
    wrap_period: float = 0.2804,
    tolerance: float = 0.04,
    max_iters: int = 20,
) -> np.ndarray:
    """
    Aggressively correct any jump near λ/2 (27-28 cm).
    
    Since ice layers typically don't move much (few cm/year), any step
    close to λ/2 ≈ 28 cm is almost certainly a phase wrapping artifact.
    
    Args:
        range_change: Range change time series (m)
        wrap_period: λ/2 in meters (default 0.2804 m)
        tolerance: How close to λ/2 the jump must be to correct (m)
        max_iters: Maximum correction iterations
        
    Returns:
        Corrected range change
    """
    corrected = range_change.copy()
    
    for iteration in range(max_iters):
        diffs = np.diff(corrected)
        if diffs.size == 0:
            break
        
        # Find jumps that are close to ±λ/2
        jump_mask = np.abs(np.abs(diffs) - wrap_period) < tolerance
        if not np.any(jump_mask):
            break
            
        # Correct the largest λ/2-like jump first
        jump_indices = np.where(jump_mask)[0]
        jump_sizes = np.abs(diffs[jump_indices])
        idx = jump_indices[np.argmax(jump_sizes)]
        
        # Apply correction
        shift = -np.sign(diffs[idx]) * wrap_period
        corrected[idx + 1:] = corrected[idx + 1:] + shift
    
    return corrected


def correct_cumulative_drift(
    range_change: np.ndarray,
    wrap_period: float = 0.2804,
    max_expected_drift_m: float = 0.10,
) -> np.ndarray:
    """
    Correct for cumulative drift that exceeds expected range.
    
    If total range change exceeds what's physically expected (e.g., > 10 cm
    over the measurement period), check if removing λ/2 wraps improves it.
    
    Args:
        range_change: Range change time series (m)
        wrap_period: λ/2 in meters
        max_expected_drift_m: Maximum expected total drift (m)
        
    Returns:
        Corrected range change
    """
    corrected = range_change.copy()
    total_drift = corrected[-1] - corrected[0]
    
    if abs(total_drift) > max_expected_drift_m:
        # Check if the drift is close to multiples of λ/2
        n_wraps = round(total_drift / wrap_period)
        if n_wraps != 0:
            residual = total_drift - n_wraps * wrap_period
            if abs(residual) < max_expected_drift_m:
                # Distribute the correction: shift the second half
                mid = len(corrected) // 2
                corrected[mid:] = corrected[mid:] - n_wraps * wrap_period
                
    return corrected


def correct_windowed_jumps(
    range_change: np.ndarray,
    time_days: np.ndarray,
    wrap_period: float = 0.2804,
    jump_threshold_m: float = 0.20,
    window_days: float = 5.0,
    max_iters: int = 100,
) -> np.ndarray:
    """
    Correct any step > threshold by shifting by ±λ/2.
    
    This is a simple but aggressive correction: any single step change > 20 cm
    is almost certainly a phase wrap artifact, since ice layers move slowly.
    We correct by the amount closest to λ/2.
    
    Args:
        range_change: Range change time series (m)
        time_days: Time in days for each measurement
        wrap_period: λ/2 in meters (default 0.2804 m)
        jump_threshold_m: Maximum allowed single-step change (m)
        window_days: Not used, kept for API compatibility
        max_iters: Maximum correction iterations
        
    Returns:
        Corrected range change
    """
    corrected = range_change.copy()
    
    for iteration in range(max_iters):
        # Compute differences between consecutive points
        diffs = np.diff(corrected)
        
        # Find any step that exceeds threshold
        abs_diffs = np.abs(diffs)
        if np.max(abs_diffs) <= jump_threshold_m:
            break
        
        # Get the index of the largest jump
        idx = np.argmax(abs_diffs)
        delta = diffs[idx]
        
        # Find the best correction (±λ/2, ±λ, ±3λ/2, etc.)
        best_shift = 0
        best_residual = abs(delta)
        
        for k in [-3, -2, -1, 1, 2, 3]:
            test_shift = k * wrap_period
            residual = abs(delta + test_shift)
            if residual < best_residual:
                best_residual = residual
                best_shift = test_shift
        
        if best_shift != 0:
            corrected[idx + 1:] = corrected[idx + 1:] + best_shift
        else:
            # Can't fix this jump with λ/2 correction
            # Mark it and continue (skip this index in future)
            break
    
    return corrected


def load_layer_data(layer_path: str, apres_path: str) -> Tuple[dict, dict]:
    """Load layer detection results and ApRES data."""
    # Load layer detection results
    with open(f"{layer_path}.json", 'r') as f:
        layer_info = json.load(f)
    layer_mat = loadmat(f"{layer_path}.mat")
    
    # Load ApRES data
    apres_data = loadmat(apres_path)
    
    layers = {
        'depths': np.array(layer_mat['layer_depths']).flatten(),
        'indices': np.array(layer_mat['layer_indices']).flatten().astype(int),
        'snr': np.array(layer_mat['layer_snr']).flatten(),
        'n_layers': int(layer_info['n_layers']),
    }
    
    # Use RawImageComplex if available for accurate phase tracking
    if 'RawImageComplex' in apres_data:
        raw_complex = np.array(apres_data['RawImageComplex'])
        range_img = np.abs(raw_complex)
        # Compute rfine from complex phase: phase = 4π * rfine / λ
        # So rfine = phase * λ / (4π)
        lambdac = 0.5608  # center wavelength
        rfine = np.angle(raw_complex) * lambdac / (4 * np.pi)
        print("Using RawImageComplex for phase tracking")
    else:
        range_img = np.array(apres_data['RawImage'])
        rfine = np.array(apres_data['RfineBarTime'])
        print("Using RfineBarTime for phase tracking (RawImageComplex not available)")
    
    data = {
        'range_img': range_img,
        'rfine': rfine,
        'Rcoarse': np.array(apres_data['Rcoarse']).flatten(),
        'time_days': np.array(apres_data['TimeInDays']).flatten(),
    }
    
    print(f"Loaded {layers['n_layers']} layers and ApRES time series")
    return layers, data


def track_layer_indices(
    range_img: np.ndarray,
    start_idx: int,
    search_window: Union[int, np.ndarray],
) -> np.ndarray:
    """Track layer bin indices using local peak tracking."""
    n_bins, n_times = range_img.shape
    tracked_idx = np.zeros(n_times, dtype=int)
    tracked_idx[0] = start_idx

    if isinstance(search_window, np.ndarray) and search_window.size != n_times:
        raise ValueError("search_window array must match number of time steps")

    for t in range(1, n_times):
        prev_idx = tracked_idx[t - 1]
        window = int(search_window[t]) if isinstance(search_window, np.ndarray) else int(search_window)
        lo = max(0, prev_idx - window)
        hi = min(n_bins - 1, prev_idx + window)
        window = range_img[lo:hi + 1, t]
        tracked_idx[t] = lo + int(np.argmax(window))

    return tracked_idx


def extract_phase_at_layer(
    range_img: np.ndarray,
    rfine: np.ndarray,
    Rcoarse: np.ndarray,
    layer_idx: int,
    search_window: Union[int, np.ndarray] = 3,
    tracking_mode: str = "local_peak",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract phase/fine-range time series at a specific layer.

    Modes:
    - "fixed": stay at detected bin, track fine range only (no bin shifts)
    - "local_peak": follow the local amplitude peak within a small window
      around the previous bin to allow slow layer drift without large jumps

    Args:
        range_img: Amplitude profiles [n_bins, n_times]
        rfine: Fine range corrections [n_bins, n_times]
        Rcoarse: Coarse range vector (m)
        layer_idx: Central bin index for this layer
        search_window: Window in bins for local peak tracking (int or per-time array)
        tracking_mode: "fixed" or "local_peak"

    Returns:
        total_range: Range time series (m)
        fine_range: Fine range time series (m)
        amplitude: Amplitude time series
        tracked_idx: Bin indices used at each time
    """
    n_bins, n_times = range_img.shape

    if tracking_mode == "fixed":
        fine_range = rfine[layer_idx, :].copy()
        amplitude = range_img[layer_idx, :].copy()
        tracked_idx = np.full(n_times, layer_idx, dtype=int)
        total_range = fine_range.copy()
        return total_range, fine_range, amplitude, tracked_idx

    if tracking_mode != "local_peak":
        raise ValueError(f"Unknown tracking_mode: {tracking_mode}")

    tracked_idx = track_layer_indices(range_img, layer_idx, search_window)
    time_idx = np.arange(n_times)
    fine_range = rfine[tracked_idx, time_idx]
    amplitude = range_img[tracked_idx, time_idx]
    total_range = Rcoarse[tracked_idx] + fine_range

    return total_range, fine_range, amplitude, tracked_idx


def unwrap_range_timeseries(
    total_range: np.ndarray,
    lambdac: float = 0.5608,
    smooth_window: int = 5,
    mode: str = "min_step",
    max_step: Optional[float] = None,
    deriv_window: int = 21,
    deriv_mad_thresh: float = 4.0,
    hard_jump_m: float = 0.20,
    hard_jump_window: int = 3,
    deriv_cap_m: Optional[float] = None,
    reject_jumps: bool = True,
    reject_sigma: float = 6.0,
    reject_floor_m: float = 0.08,
) -> np.ndarray:
    """
    Unwrap phase jumps in the fine range time series.
    
    The fine range (rfine) can have phase wraps every λ/2. We need to 
    unwrap these to get continuous displacement.
    
    Args:
        total_range: Fine range values (m)
        lambdac: Center wavelength in ice (m)
        smooth_window: Window size for median filter (must be odd)
        
    Returns:
        Unwrapped range change relative to first point (m)
    """
    wrap_period = lambdac / 2  # Phase wrap period (~0.28 m)
    threshold = wrap_period / 2  # Jump threshold (~0.14 m)
    
    # Make a copy and compute result array
    unwrapped = np.zeros_like(total_range)
    unwrapped[0] = total_range[0]

    if max_step is None:
        max_step = threshold  # default ~λ/4

    if mode == "threshold":
        cumulative_correction = 0.0
        for i in range(1, len(total_range)):
            diff = total_range[i] - total_range[i - 1]

            if diff > threshold:
                cumulative_correction -= wrap_period
            elif diff < -threshold:
                cumulative_correction += wrap_period

            unwrapped[i] = total_range[i] + cumulative_correction
    elif mode == "min_step":
        for i in range(1, len(total_range)):
            candidates = [total_range[i] + k * wrap_period for k in (-2, -1, 0, 1, 2)]
            deltas = [abs(c - unwrapped[i - 1]) for c in candidates]
            best = candidates[int(np.argmin(deltas))]

            if abs(best - unwrapped[i - 1]) > max_step:
                # Clamp to max_step to prevent large jumps
                step = np.sign(best - unwrapped[i - 1]) * max_step
                unwrapped[i] = unwrapped[i - 1] + step
            else:
                unwrapped[i] = best
    elif mode == "robust_derivative":
        for i in range(1, len(total_range)):
            candidates = [total_range[i] + k * wrap_period for k in (-2, -1, 0, 1, 2)]
            deltas = [abs(c - unwrapped[i - 1]) for c in candidates]
            best = candidates[int(np.argmin(deltas))]

            if abs(best - unwrapped[i - 1]) > max_step:
                step = np.sign(best - unwrapped[i - 1]) * max_step
                unwrapped[i] = unwrapped[i - 1] + step
            else:
                unwrapped[i] = best

        if deriv_window % 2 == 0:
            deriv_window += 1

        if len(unwrapped) >= 3:
            deriv = np.diff(unwrapped)
            kernel = np.ones(deriv_window) / deriv_window
            median_like = np.convolve(deriv, kernel, mode='same')
            resid = deriv - median_like
            mad = np.median(np.abs(resid - np.median(resid))) + 1e-9
            threshold_deriv = deriv_mad_thresh * mad

            cleaned = deriv.copy()
            outliers = np.abs(resid) > threshold_deriv
            cleaned[outliers] = median_like[outliers]

            unwrapped = np.concatenate([[unwrapped[0]], unwrapped[0] + np.cumsum(cleaned)])
    elif mode == "branch_median":
        window = 7
        for i in range(1, len(total_range)):
            candidates = [total_range[i] + k * wrap_period for k in (-2, -1, 0, 1, 2)]
            start = max(0, i - window)
            median_val = np.median(unwrapped[start:i])
            deltas = [abs(c - median_val) for c in candidates]
            unwrapped[i] = candidates[int(np.argmin(deltas))]
    else:
        raise ValueError(f"Unknown unwrap mode: {mode}")

    # Hard-jump filter: remove rare large steps and interpolate across them
    deriv = np.diff(unwrapped)
    jump_idx = np.where(np.abs(deriv) > hard_jump_m)[0]
    if jump_idx.size > 0:
        mask = np.ones_like(unwrapped, dtype=bool)
        half = max(1, hard_jump_window // 2)
        for j in jump_idx:
            start = max(0, j - half)
            end = min(len(unwrapped) - 1, j + 1 + half)
            mask[start:end + 1] = False

        x = np.arange(len(unwrapped))
        if np.sum(mask) >= 2:
            unwrapped = np.interp(x, x[mask], unwrapped[mask])

    # Derivative cap: clamp any remaining per-step changes
    if deriv_cap_m is not None:
        deriv = np.diff(unwrapped)
        deriv = np.clip(deriv, -deriv_cap_m, deriv_cap_m)
        unwrapped = np.concatenate([[unwrapped[0]], unwrapped[0] + np.cumsum(deriv)])

    if reject_jumps:
        deriv = np.diff(unwrapped)
        kernel = np.ones(deriv_window) / deriv_window
        median_like = np.convolve(deriv, kernel, mode='same')
        resid = deriv - median_like
        mad = np.median(np.abs(resid - np.median(resid))) + 1e-9
        threshold = max(reject_floor_m, reject_sigma * mad)
        outliers = np.where(np.abs(resid) > threshold)[0]

        if outliers.size > 0:
            mask = np.ones_like(unwrapped, dtype=bool)
            for i in outliers:
                mask[i] = False
                if i + 1 < len(mask):
                    mask[i + 1] = False

            x = np.arange(len(unwrapped))
            if np.sum(mask) >= 2:
                unwrapped = np.interp(x, x[mask], unwrapped[mask])
    
    # Compute range change relative to first point
    range_change = unwrapped - unwrapped[0]
    
    # Optional: gentle smoothing to remove small noise
    if smooth_window >= 3:
        if smooth_window % 2 == 0:
            smooth_window += 1
        range_change = signal.medfilt(range_change, kernel_size=min(smooth_window, len(range_change)))
    
    return range_change


def track_all_layers_smooth(
    layers: dict,
    data: dict,
    lambdac: float = 0.5608,
    search_window_m: float = 1.0,
    adaptive_window: bool = False,
    max_search_window_m: Optional[float] = None,
    smooth_window: int = 11,
    tracking_mode: str = "fixed",
    unwrap_mode: str = "min_step",
    max_step_m: float = 0.03,
    deriv_window: int = 21,
    deriv_mad_thresh: float = 4.0,
    hard_jump_m: float = 0.20,
    hard_jump_window: int = 3,
    deriv_cap_m: Optional[float] = None,
    reject_jumps: bool = True,
    reject_sigma: float = 6.0,
    reject_floor_m: float = 0.08,
) -> PhaseTrackingResult:
    """
    Track phase evolution for all detected layers with smooth interpolation.
    
    Uses:
    1. Fixed-bin or local-peak tracking (configurable)
    2. Robust phase unwrapping with optional jump clamping
    3. Gentle median filtering to remove outliers
    
    Args:
        layers: Layer detection results
        data: ApRES data dictionary
        lambdac: Center wavelength in ice (m)
        search_window_m: Search window for peak tracking (m) - smaller is better
        smooth_window: Median filter window (odd number)
        
    Returns:
        PhaseTrackingResult with all layer phase data
    """
    n_layers = layers['n_layers']
    n_times = len(data['time_days'])
    bin_spacing = data['Rcoarse'][1] - data['Rcoarse'][0]
    search_window = max(2, int(search_window_m / bin_spacing))
    search_window_array: Union[int, np.ndarray] = search_window

    if adaptive_window:
        if max_search_window_m is None:
            max_search_window_m = max(search_window_m * 2, search_window_m)
        max_search_window = max(2, int(max_search_window_m / bin_spacing))
        max_search_window = max(max_search_window, search_window)

        time_days = data['time_days']
        total_span = max(time_days[-1] - time_days[0], 1.0)
        frac = (time_days - time_days[0]) / total_span
        window_bins = search_window + (max_search_window - search_window) * frac
        search_window_array = np.clip(np.rint(window_bins).astype(int), search_window, max_search_window)
        if tracking_mode == "fixed":
            tracking_mode = "local_peak"
    
    # Initialize arrays
    phase_ts = np.zeros((n_layers, n_times))
    range_ts = np.zeros((n_layers, n_times))
    amp_ts = np.zeros((n_layers, n_times))
    
    print(f"\nTracking {n_layers} layers through {n_times} time steps...")
    print(f"  Sub-bin interpolation: parabolic")
    if adaptive_window:
        print(
            f"  Search window: {search_window}–{int(np.max(search_window_array))} bins "
            f"({search_window_m:.1f}–{max_search_window_m:.1f} m, adaptive)"
        )
    else:
        print(f"  Search window: {search_window} bins ({search_window_m:.1f} m)")
    print(f"  Smooth window: {smooth_window} points")
    print(f"  Tracking mode: {tracking_mode}")
    print(f"  Unwrap mode: {unwrap_mode}, max step: {max_step_m*100:.1f} cm")
    if unwrap_mode == "robust_derivative":
        print(f"  Derivative filter: window={deriv_window}, MAD×{deriv_mad_thresh}")
    print(f"  Hard-jump filter: {hard_jump_m*100:.1f} cm, window={hard_jump_window}")
    if deriv_cap_m is not None:
        print(f"  Derivative cap: {deriv_cap_m*100:.2f} cm per step")
    if reject_jumps:
        print(f"  Jump rejection: σ×{reject_sigma}, floor {reject_floor_m*100:.1f} cm")
    
    for i, layer_idx in enumerate(layers['indices']):
        layer_depth = layers['depths'][i]
        layer_reject_floor = reject_floor_m
        layer_reject_sigma = reject_sigma
        layer_hard_jump_m = hard_jump_m
        layer_hard_jump_window = hard_jump_window
        if layer_depth >= 500:
            layer_reject_floor = min(layer_reject_floor, 0.01)
            layer_reject_sigma = min(layer_reject_sigma, 4.0)
            layer_hard_jump_m = min(layer_hard_jump_m, 0.005)
            layer_hard_jump_window = max(layer_hard_jump_window, 7)
        # Extract range using parabolic interpolation
        total_range, fine_range, amplitude, _ = extract_phase_at_layer(
            data['range_img'],
            data['rfine'],
            data['Rcoarse'],
            layer_idx,
            search_window_array,
            tracking_mode,
        )
        
        # Apply robust unwrapping with step clamp
        range_change = unwrap_range_timeseries(
            total_range,
            lambdac=lambdac,
            smooth_window=smooth_window,
            mode=unwrap_mode,
            max_step=max_step_m,
            deriv_window=deriv_window,
            deriv_mad_thresh=deriv_mad_thresh,
            hard_jump_m=layer_hard_jump_m,
            hard_jump_window=layer_hard_jump_window,
            deriv_cap_m=deriv_cap_m,
            reject_jumps=reject_jumps,
            reject_sigma=layer_reject_sigma,
            reject_floor_m=layer_reject_floor,
        )

        range_change = correct_wrap_segments(
            range_change,
            data['time_days'],
            wrap_period=lambdac / 2,
            min_days=3.0,
            max_days=5.0,
            jump_m=0.15,
            tolerance_m=0.12,
            max_iters=10,
        )

        range_change = correct_large_jumps(
            range_change,
            wrap_period=lambdac / 2,
            jump_threshold=0.15,  # Lowered: any 15+ cm jump is suspicious
            max_iters=10,
        )

        # Aggressively correct any remaining λ/2 jumps
        range_change = correct_lambda2_jumps(
            range_change,
            wrap_period=lambdac / 2,
            tolerance=0.05,  # 5 cm tolerance: 23-33 cm jumps are corrected
            max_iters=20,
        )

        # Correct cumulative drift if it's implausibly large
        range_change = correct_cumulative_drift(
            range_change,
            wrap_period=lambdac / 2,
            max_expected_drift_m=0.08,  # Expect < 8 cm total drift
        )

        # Final pass: correct any jump > 20 cm within 5 days
        # This is the most aggressive correction based on physical constraints
        range_change = correct_windowed_jumps(
            range_change,
            data['time_days'],
            wrap_period=lambdac / 2,
            jump_threshold_m=0.20,  # 20 cm threshold as specified
            window_days=5.0,        # 5 day window as specified
            max_iters=50,
        )

        # Note: Removed the extra loop that was using total_range to re-detect 
        # wrap-like shifts, as it could undo the corrections from correct_windowed_jumps

        if layer_depth >= 500:
            head = range_change[:50]
            tail = range_change[-50:]
            if head.size >= 10 and tail.size >= 10:
                head_med = np.nanmedian(head)
                tail_med = np.nanmedian(tail)
                offset = tail_med - head_med
                if abs(abs(offset) - lambdac / 2) <= 0.06:
                    range_change = range_change + np.sign(offset) * (lambdac / 2) * -1
        
        # Store results
        range_ts[i, :] = range_change
        
        # Convert to phase for storage
        phase_ts[i, :] = range_change * (4 * np.pi / lambdac)
        
        # Store amplitude
        amp_ts[i, :] = amplitude
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{n_layers} layers")
    
    result = PhaseTrackingResult(
        layer_depths=layers['depths'],
        phase_timeseries=phase_ts,
        range_timeseries=range_ts,
        amplitude_timeseries=amp_ts,
        time_days=data['time_days'],
        n_layers=n_layers,
        lambdac=lambdac,
    )
    
    print(f"Phase tracking complete!")
    return result


# Alias for backward compatibility
track_all_layers = track_all_layers_smooth


def visualize_phase_tracking(
    result: PhaseTrackingResult,
    n_layers_to_show: int = 6,
    output_file: Optional[str] = None,
):
    """
    Visualize phase/range evolution for selected layers.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy import stats
    
    # Select layers evenly distributed in depth
    if result.n_layers <= n_layers_to_show:
        layer_indices = list(range(result.n_layers))
    else:
        layer_indices = np.linspace(0, result.n_layers - 1, n_layers_to_show, dtype=int)
    
    n_show = len(layer_indices)
    
    # Create subplots: 2 columns (range change, amplitude) × n rows
    fig = make_subplots(
        rows=n_show, cols=2,
        subplot_titles=[
            item for i in layer_indices 
            for item in (f'Layer at {result.layer_depths[i]:.0f}m - Range Change', 'Amplitude')
        ],
        horizontal_spacing=0.1,
        vertical_spacing=0.05,
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for row, idx in enumerate(layer_indices, 1):
        color = colors[(row-1) % len(colors)]
        depth = result.layer_depths[idx]
        
        # Range change in cm
        range_cm = result.range_timeseries[idx, :] * 100
        
        # Linear fit for velocity
        valid = ~np.isnan(range_cm)
        if np.sum(valid) > 10:
            slope, intercept, r, _, _ = stats.linregress(
                result.time_days[valid], range_cm[valid]
            )
            velocity_m_yr = slope * 365.25 / 100  # Convert cm/day to m/year
            fit_line = slope * result.time_days + intercept
            r_sq = r**2
        else:
            velocity_m_yr = np.nan
            fit_line = np.full_like(result.time_days, np.nan)
            r_sq = 0
        
        # Plot range change
        fig.add_trace(
            go.Scatter(
                x=result.time_days,
                y=range_cm,
                mode='lines',
                line=dict(color=color, width=1),
                name=f'{depth:.0f}m',
                showlegend=(row == 1),
                hovertemplate=f'Depth: {depth:.0f}m<br>Time: %{{x:.1f}} days<br>ΔRange: %{{y:.2f}} cm<extra></extra>',
            ),
            row=row, col=1
        )
        
        # Add fit line
        fig.add_trace(
            go.Scatter(
                x=result.time_days,
                y=fit_line,
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name=f'Fit: {velocity_m_yr:.2f} m/yr (R²={r_sq:.2f})',
                showlegend=False,
            ),
            row=row, col=1
        )
        
        # Amplitude in dB
        amp_db = 10 * np.log10(result.amplitude_timeseries[idx, :]**2 + 1e-30)
        
        fig.add_trace(
            go.Scatter(
                x=result.time_days,
                y=amp_db,
                mode='lines',
                line=dict(color=color, width=1),
                showlegend=False,
            ),
            row=row, col=2
        )
        
        # Add velocity annotation using paper coordinates
        # Calculate y position based on row
        y_pos = 1.0 - (row - 0.5) / n_show
        fig.add_annotation(
            x=0.02, y=y_pos,
            xref='paper', yref='paper',
            text=f'<b>{depth:.0f}m:</b> v = {velocity_m_yr:.2f} m/yr, R² = {r_sq:.2f}',
            showarrow=False,
            font=dict(size=10),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            xanchor='left',
        )
    
    # Update axes
    for row in range(1, n_show + 1):
        fig.update_yaxes(title_text='ΔRange (cm)', row=row, col=1)
        fig.update_yaxes(title_text='Amp (dB)', row=row, col=2)
        if row == n_show:
            fig.update_xaxes(title_text='Time (days)', row=row, col=1)
            fig.update_xaxes(title_text='Time (days)', row=row, col=2)
    
    fig.update_layout(
        title=dict(
            text='Phase/Range Tracking for Internal Ice Layers',
            font=dict(size=16),
        ),
        height=200 * n_show,
        width=1200,
        showlegend=True,
    )
    
    if output_file:
        fig.write_html(output_file)
        print(f"Saved: {output_file}")
    
    fig.show()
    return fig


def save_phase_results(result: PhaseTrackingResult, output_path: str) -> None:
    """Save phase tracking results."""
    mat_data = {
        'layer_depths': result.layer_depths,
        'phase_timeseries': result.phase_timeseries,
        'range_timeseries': result.range_timeseries,
        'amplitude_timeseries': result.amplitude_timeseries,
        'time_days': result.time_days,
        'n_layers': result.n_layers,
        'lambdac': result.lambdac,
    }
    savemat(f"{output_path}.mat", mat_data)
    print(f"Saved: {output_path}.mat")


# =============================================================================
# Main entry point
# =============================================================================

def main():
    """Run phase tracking on detected layers."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Track phase evolution of internal ice layers')
    parser.add_argument('--layers', type=str, default='../../data/apres/detected_layers',
                        help='Path to layer detection results (without extension)')
    parser.add_argument('--data', type=str, default='../../data/apres/ImageP2_python.mat',
                        help='Path to processed ApRES data (.mat)')
    parser.add_argument('--output', type=str, default='../../data/apres/phase_tracking',
                        help='Output file path (without extension)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip visualization')
    
    args = parser.parse_args()
    
    # Load data
    layers, data = load_layer_data(args.layers, args.data)
    
    # Track phases
    result = track_all_layers(layers, data)
    
    # Save results
    save_phase_results(result, args.output)
    
    # Visualize
    if not args.no_plot:
        visualize_phase_tracking(
            result,
            output_file=f"{args.output}_visualization.html",
        )
    
    return result


if __name__ == '__main__':
    main()
