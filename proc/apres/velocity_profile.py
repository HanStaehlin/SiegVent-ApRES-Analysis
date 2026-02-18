"""Velocity Profile Module for ApRES Internal Ice Layer Analysis

This module calculates depth-dependent velocity profiles from phase tracking
results, following the methodology of Summers et al. (2021).

Key equation: v_r = f · λ_c (Eq. 1)

The range velocity v_r at each layer is derived from the linear trend in
phase/range over time.

Uncertainty estimation follows Kingslake et al. (2014, JGR Earth Surface),
Section 2.1: The phasor of each reflector is combined with a noise phasor
(length = median noise amplitude, unknown orientation). The phase uncertainty
is the deviation introduced when the noise phasor is perpendicular to the
signal phasor: σ_φ = arctan(A_noise / A_signal). This is converted to a
range uncertainty σ_r = σ_φ · λ_c / (4π), then propagated through the
linear regression to obtain velocity uncertainty.

Author: SiegVent2023 project
"""

import numpy as np
from scipy import stats
from scipy.io import loadmat, savemat
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass
from typing import Optional, Tuple
import json


@dataclass  
class VelocityProfileResult:
    """Container for velocity profile results."""
    depths: np.ndarray                # Depth of each layer (m)
    velocities: np.ndarray            # Range velocity at each layer (m/year)
    velocities_smooth: np.ndarray     # Smoothed velocity profile (m/year)
    r_squared: np.ndarray             # R² of linear fit for each layer
    std_error: np.ndarray             # Standard error of velocity (m/year)
    amplitude_mean: np.ndarray        # Mean amplitude (dB)
    reliable: np.ndarray              # Boolean mask for reliable measurements
    n_layers: int
    r_sq_threshold: float             # R² threshold used
    amp_threshold: float              # Amplitude threshold used (dB)
    # Kingslake et al. (2014) uncertainty fields
    uncertainty_kingslake: Optional[np.ndarray] = None   # Conservative velocity uncertainty (m/yr)
    uncertainty_wls: Optional[np.ndarray] = None         # Weighted least-squares velocity uncertainty (m/yr)
    range_uncertainty_mean: Optional[np.ndarray] = None  # Mean range uncertainty per layer (m)
    snr_mean: Optional[np.ndarray] = None                # Mean SNR per layer (dB)
    tracking_mode: Optional[np.ndarray] = None           # 1=phase, 0=speckle per layer
    slope_consistent: Optional[np.ndarray] = None        # True if first/second-half slopes agree
    velocity_inlier: Optional[np.ndarray] = None         # True if velocity near neighbors
    quality_pass: Optional[np.ndarray] = None             # True if passes quality filters


def load_phase_data(phase_path: str) -> dict:
    """Load phase tracking results."""
    mat_data = loadmat(f"{phase_path}.mat")
    
    # Handle scalar values that get stored as arrays in .mat files
    n_layers = mat_data['n_layers']
    if hasattr(n_layers, 'flatten'):
        n_layers = int(n_layers.flatten()[0])
    else:
        n_layers = int(n_layers)
    
    lambdac = mat_data['lambdac']
    if hasattr(lambdac, 'flatten'):
        lambdac = float(lambdac.flatten()[0])
    else:
        lambdac = float(lambdac)
    
    data = {
        'layer_depths': np.array(mat_data['layer_depths']).flatten(),
        'range_timeseries': np.array(mat_data['range_timeseries']),
        'amplitude_timeseries': np.array(mat_data['amplitude_timeseries']),
        'time_days': np.array(mat_data['time_days']).flatten(),
        'n_layers': n_layers,
        'lambdac': lambdac,
    }
    
    # Load tracking mode if available (1=phase, 0=speckle)
    if 'tracking_mode' in mat_data:
        data['tracking_mode'] = np.array(mat_data['tracking_mode']).flatten()
    
    print(f"Loaded phase data for {data['n_layers']} layers")
    return data


# =============================================================================
# Kingslake et al. (2014) Uncertainty Estimation
# =============================================================================

def compute_phase_uncertainty_per_measurement(
    amplitude_timeseries: np.ndarray,
    noise_floor: float,
    lambdac: float = 0.5608,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-measurement range uncertainty using the Kingslake phasor method.
    
    Following Kingslake et al. (2014), Section 2.1:
    "We combine the phasor of each reflector with a noise phasor, which has a 
    length equal to the median strength of the entire return and an unknown 
    orientation. The uncertainty in the phase is defined as the deviation in 
    the phase introduced by the noise phasor when it is oriented perpendicular 
    to the reflector phasor."
    
    The geometry: signal phasor has amplitude A_s, noise phasor has amplitude
    A_n oriented perpendicular to it. The resulting phase deviation is:
        σ_φ = arctan(A_n / A_s)
    
    This is converted to a range uncertainty:
        σ_r = σ_φ · λ_c / (4π)
    
    Args:
        amplitude_timeseries: Amplitude at each measurement [n_layers, n_times]
                              or [n_times] for a single layer. Linear amplitude.
        noise_floor: Median noise amplitude (linear, not dB). This is the
                     "median strength of the entire return" per Kingslake.
        lambdac: Center wavelength in ice (m), default 0.5608 m.
        
    Returns:
        sigma_range: Range uncertainty per measurement (m) [n_layers, n_times]
        sigma_phase: Phase uncertainty per measurement (rad) [n_layers, n_times]
    """
    amp = np.asarray(amplitude_timeseries, dtype=float)
    
    # Prevent division by zero: where signal is below noise, set to noise level
    # This gives σ_φ = arctan(1) = π/4 ≈ 45° (maximum meaningful uncertainty)
    amp_safe = np.maximum(amp, noise_floor)
    
    # Kingslake phasor geometry: σ_φ = arctan(A_noise / A_signal)
    sigma_phase = np.arctan(noise_floor / amp_safe)  # radians
    
    # Convert phase uncertainty to range uncertainty
    # Phase = 4π·r/λ → δr = δφ · λ/(4π)
    sigma_range = sigma_phase * lambdac / (4.0 * np.pi)
    
    return sigma_range, sigma_phase


def estimate_noise_floor(
    raw_image: np.ndarray,
    noise_fraction: float = 0.1,
) -> float:
    """
    Estimate the noise floor from the deep Echo Free Zone (EFZ).
    
    Following Kingslake: "a noise phasor, which has a length equal to the 
    median strength of the entire return". We use the median of the deepest
    portion of the return as a robust noise estimate.
    
    Args:
        raw_image: Amplitude profiles [n_bins, n_times] (linear amplitude)
        noise_fraction: Fraction of deepest bins to use (default: bottom 10%)
        
    Returns:
        noise_floor: Median noise amplitude (linear)
    """
    n_bins = raw_image.shape[0]
    start_bin = int(n_bins * (1.0 - noise_fraction))
    noise_region = np.abs(raw_image[start_bin:, :])
    noise_floor = float(np.median(noise_region))
    return noise_floor


def compute_velocity_uncertainty_kingslake(
    sigma_range: np.ndarray,
    time_span_years: float,
) -> np.ndarray:
    """
    Conservative Kingslake velocity uncertainty (two-deployment analog).
    
    Kingslake et al. (2014): "A conservative estimate of uncertainty in the
    velocity is calculated from the sum of the uncertainties in each 
    reflector's position."
    
    For two deployments separated by Δt:
        σ_v = (σ_r1 + σ_r2) / Δt
    
    For our continuous time series, we use the mean of the first and last 
    10% of measurements as proxies for the two "deployments":
        σ_v = (mean(σ_r_first) + mean(σ_r_last)) / Δt
    
    Args:
        sigma_range: Range uncertainty per measurement [n_layers, n_times] (m)
        time_span_years: Total time span between first and last measurement (years)
        
    Returns:
        sigma_v: Conservative velocity uncertainty per layer (m/yr) [n_layers]
    """
    if sigma_range.ndim == 1:
        sigma_range = sigma_range.reshape(1, -1)
    
    n_layers, n_times = sigma_range.shape
    n_edge = max(1, n_times // 10)  # Use 10% of measurements at each end
    
    # Mean range uncertainty at "first deployment" and "last deployment"
    sigma_r1 = np.mean(sigma_range[:, :n_edge], axis=1)
    sigma_r2 = np.mean(sigma_range[:, -n_edge:], axis=1)
    
    # Conservative: sum of uncertainties / time span
    sigma_v = (sigma_r1 + sigma_r2) / time_span_years
    
    return sigma_v


def compute_velocity_uncertainty_wls(
    time_days: np.ndarray,
    range_change: np.ndarray,
    sigma_range: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    Weighted least-squares velocity estimate with proper uncertainty propagation.
    
    Uses per-measurement range uncertainties as weights in the linear regression.
    The velocity uncertainty is derived from the weighted covariance matrix,
    following standard WLS methodology.
    
    Model: range_change(t) = v · t + b
    Weights: w_i = 1 / σ_r_i²
    
    The weighted least-squares solution gives:
        σ_v = sqrt( Σw_i / (Σw_i · Σ(w_i·t_i²) - (Σ(w_i·t_i))²) )
    
    This properly accounts for the fact that measurements with lower SNR
    (higher σ_r) should contribute less to the velocity estimate.
    
    Args:
        time_days: Time vector (days)
        range_change: Range change values (m)
        sigma_range: Range uncertainty per measurement (m)
        
    Returns:
        velocity: Range velocity (m/year)
        r_squared: R² of fit
        sigma_v_wls: Uncertainty from weighted regression (m/year)
        sigma_v_resid: Uncertainty from residual scatter (m/year)
    """
    valid = ~np.isnan(range_change) & ~np.isnan(time_days) & (sigma_range > 0)
    
    if np.sum(valid) < 10:
        return np.nan, 0.0, np.nan, np.nan
    
    t = time_days[valid]
    r = range_change[valid]
    sr = sigma_range[valid]
    
    # Weights: inverse variance
    w = 1.0 / (sr ** 2)
    
    # Weighted sums
    S_w = np.sum(w)
    S_wt = np.sum(w * t)
    S_wt2 = np.sum(w * t ** 2)
    S_wr = np.sum(w * r)
    S_wtr = np.sum(w * t * r)
    
    # Determinant of normal equations
    D = S_w * S_wt2 - S_wt ** 2
    
    if abs(D) < 1e-30:
        return np.nan, 0.0, np.nan, np.nan
    
    # Weighted least-squares slope and intercept
    slope = (S_w * S_wtr - S_wt * S_wr) / D
    intercept = (S_wt2 * S_wr - S_wt * S_wtr) / D
    
    # Velocity in m/year
    velocity = slope * 365.25
    
    # Formal WLS uncertainty of slope (from measurement uncertainties)
    sigma_slope_wls = np.sqrt(S_w / D)
    sigma_v_wls = sigma_slope_wls * 365.25
    
    # Also compute residual-based uncertainty (captures unmodeled scatter)
    residuals = r - (slope * t + intercept)
    n = len(t)
    if n > 2:
        chi2_red = np.sum(w * residuals ** 2) / (n - 2)
        # Scale formal uncertainty by sqrt(reduced chi²) if > 1
        # This inflates uncertainties when the model doesn't fully explain the data
        scale = max(1.0, np.sqrt(chi2_red))
        sigma_v_resid = sigma_v_wls * scale
    else:
        sigma_v_resid = sigma_v_wls
    
    # R² (unweighted for comparability)
    r_mean = np.mean(r)
    ss_res = np.sum((r - (slope * t + intercept)) ** 2)
    ss_tot = np.sum((r - r_mean) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    return velocity, r_squared, sigma_v_wls, sigma_v_resid


def calculate_layer_velocity(
    time_days: np.ndarray,
    range_change: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Calculate velocity from range change time series using linear regression.
    
    Args:
        time_days: Time vector (days)
        range_change: Range change values (m)
        
    Returns:
        velocity: Range velocity (m/year)
        r_squared: R² of fit
        std_error: Standard error of slope (m/year)
    """
    valid = ~np.isnan(range_change) & ~np.isnan(time_days)
    
    if np.sum(valid) < 10:
        return np.nan, 0.0, np.nan
    
    slope, _, r_value, _, std_err = stats.linregress(
        time_days[valid], range_change[valid]
    )
    
    # Convert from m/day to m/year
    velocity = slope * 365.25
    std_error = std_err * 365.25
    r_squared = r_value ** 2
    
    return velocity, r_squared, std_error


def calculate_velocity_profile(
    phase_data: dict,
    r_sq_threshold: float = 0.3,
    amp_threshold_db: float = -80,
    smooth_sigma: float = 2.0,
    noise_floor: Optional[float] = None,
    raw_image: Optional[np.ndarray] = None,
) -> VelocityProfileResult:
    """
    Calculate velocity profile for all layers with Kingslake uncertainty estimation.
    
    Implements the SNR-based uncertainty estimation from Kingslake et al. (2014):
    1. Per-measurement range uncertainty from phasor SNR geometry
    2. Conservative velocity uncertainty (Kingslake sum method)
    3. Weighted least-squares velocity with properly propagated uncertainties
    
    Args:
        phase_data: Phase tracking results
        r_sq_threshold: Minimum R² for reliable velocity
        amp_threshold_db: Minimum amplitude for reliable phase (dB)
        smooth_sigma: Gaussian smoothing sigma (in layer indices)
        noise_floor: Noise floor amplitude (linear). If None, estimated from
                     raw_image or amplitude_timeseries.
        raw_image: Raw amplitude image [n_bins, n_times] for noise estimation.
                   If None, noise is estimated from amplitude_timeseries.
        
    Returns:
        VelocityProfileResult with velocity profile and Kingslake uncertainties
    """
    n_layers = phase_data['n_layers']
    depths = phase_data['layer_depths']
    time_days = phase_data['time_days']
    lambdac = phase_data.get('lambdac', 0.5608)
    amp_ts = phase_data['amplitude_timeseries']  # [n_layers, n_times]
    
    # Get tracking mode if available
    tracking_mode = None
    if 'tracking_mode' in phase_data:
        tracking_mode = phase_data['tracking_mode'].flatten()
    
    # ================================================================
    # Step 1: Estimate noise floor (Kingslake: "median strength of the
    #         entire return")
    # ================================================================
    if noise_floor is None:
        if raw_image is not None:
            noise_floor = estimate_noise_floor(np.abs(raw_image))
            print(f"  Noise floor from raw image (EFZ): {noise_floor:.4f}")
            print(f"  Noise floor: {10*np.log10(noise_floor**2 + 1e-30):.1f} dB")
        else:
            # Fallback: use amplitude time series itself
            # Estimate noise as median of the weakest 20% of all amplitudes
            all_amp = amp_ts.flatten()
            noise_floor = float(np.percentile(all_amp[all_amp > 0], 10))
            print(f"  Noise floor from amplitude percentile: {noise_floor:.4f}")
            print(f"  Noise floor: {10*np.log10(noise_floor**2 + 1e-30):.1f} dB")
    else:
        print(f"  Noise floor (provided): {noise_floor:.4f}")
    
    # ================================================================
    # Step 2: Compute per-measurement range uncertainty (Kingslake phasor)
    # ================================================================
    sigma_range_all, sigma_phase_all = compute_phase_uncertainty_per_measurement(
        amp_ts, noise_floor, lambdac
    )
    # sigma_range_all shape: [n_layers, n_times]
    
    # Time span in years
    time_span_years = (time_days[-1] - time_days[0]) / 365.25
    
    # ================================================================
    # Step 3: Kingslake conservative velocity uncertainty
    # ================================================================
    uncertainty_kingslake = compute_velocity_uncertainty_kingslake(
        sigma_range_all, time_span_years
    )
    
    # Initialize arrays
    velocities = np.zeros(n_layers)
    r_squared = np.zeros(n_layers)
    std_error = np.zeros(n_layers)
    uncertainty_wls = np.zeros(n_layers)
    amp_mean = np.zeros(n_layers)
    range_uncertainty_mean = np.zeros(n_layers)
    snr_mean = np.zeros(n_layers)
    
    print(f"\nCalculating velocities for {n_layers} layers...")
    print(f"  Kingslake uncertainty estimation: ENABLED")
    print(f"  λ_c = {lambdac:.4f} m, noise floor = {noise_floor:.4f}")
    
    for i in range(n_layers):
        # Get range change for this layer
        range_change = phase_data['range_timeseries'][i, :]
        sigma_range_i = sigma_range_all[i, :]
        
        # Calculate velocity with weighted least-squares
        vel_wls, r_sq_wls, sig_v_wls, sig_v_resid = compute_velocity_uncertainty_wls(
            time_days, range_change, sigma_range_i
        )
        
        # Also get OLS for backward compatibility
        vel_ols, r_sq_ols, std_err_ols = calculate_layer_velocity(time_days, range_change)
        
        # Use WLS velocity as primary (it's better); fall back to OLS if WLS fails
        if not np.isnan(vel_wls):
            velocities[i] = vel_wls
            r_squared[i] = r_sq_wls
            # Use the larger of WLS formal error and residual-scaled error
            uncertainty_wls[i] = sig_v_resid if not np.isnan(sig_v_resid) else sig_v_wls
        else:
            velocities[i] = vel_ols
            r_squared[i] = r_sq_ols
            uncertainty_wls[i] = std_err_ols if not np.isnan(std_err_ols) else np.nan
        
        std_error[i] = std_err_ols  # Keep OLS std error for backward compatibility
        
        # Mean amplitude in dB
        amp = amp_ts[i, :]
        amp_mean[i] = 10 * np.log10(np.mean(amp**2) + 1e-30)
        
        # Mean range uncertainty and SNR for this layer
        range_uncertainty_mean[i] = np.mean(sigma_range_i)
        snr_mean[i] = 10 * np.log10(np.mean(amp**2) / (noise_floor**2 + 1e-30) + 1e-30)
    
    # For speckle-tracked layers, uncertainty is larger (no phase precision)
    # Use a floor based on bin spacing resolution
    if tracking_mode is not None:
        speckle_mask = tracking_mode == 0
        if np.any(speckle_mask):
            # Speckle tracking precision is limited by bin spacing (~0.25 m)
            # rather than phase precision
            bin_spacing = 0.25  # approximate
            speckle_range_unc = bin_spacing / np.sqrt(12)  # uniform distribution
            speckle_v_unc = 2 * speckle_range_unc / time_span_years
            # Ensure speckle uncertainties are at least this large
            uncertainty_kingslake[speckle_mask] = np.maximum(
                uncertainty_kingslake[speckle_mask], speckle_v_unc
            )
            uncertainty_wls[speckle_mask] = np.maximum(
                uncertainty_wls[speckle_mask], speckle_v_unc
            )
    
    # ================================================================
    # Step 4: Quality filters — slope consistency & velocity outlier
    # ================================================================
    # Layers with broken phase tracking (wrap jumps, drift) can still
    # achieve decent R² because the linear fit "averages out" the error.
    # Two additional checks catch these:
    #   (a) Slope break: first-half vs second-half slope differs strongly
    #   (b) Velocity outlier: velocity deviates from local depth-neighbors
    # A layer must fail BOTH to be removed (conservative).
    
    slope_consistent = np.ones(n_layers, dtype=bool)
    velocity_inlier = np.ones(n_layers, dtype=bool)
    
    # Base reliable mask (R² + amplitude only) — used as neighbor pool
    base_reliable = (r_squared >= r_sq_threshold) & (amp_mean >= amp_threshold_db)
    
    # 4a: Slope consistency — compare first-half vs second-half slope
    slope_break_threshold = 0.3  # m/yr difference between halves
    n_times = len(time_days)
    mid = n_times // 2
    
    for i in range(n_layers):
        if not base_reliable[i]:
            continue
        rc = phase_data['range_timeseries'][i, :]
        valid1 = ~np.isnan(rc[:mid])
        valid2 = ~np.isnan(rc[mid:])
        if np.sum(valid1) < 5 or np.sum(valid2) < 5:
            continue
        s1 = stats.linregress(time_days[:mid][valid1], rc[:mid][valid1]).slope
        s2 = stats.linregress(time_days[mid:][valid2], rc[mid:][valid2]).slope
        if abs(s1 * 365.25 - s2 * 365.25) > slope_break_threshold:
            slope_consistent[i] = False
    
    # 4b: Velocity outlier — compare to local median of depth-neighbors
    vel_outlier_threshold = 0.15  # m/yr deviation from local median
    severe_outlier_threshold = 0.30  # m/yr — severe outlier flagged alone
    depth_window = 50.0  # m radius for neighbor search
    min_neighbors = 3
    
    for i in range(n_layers):
        if not base_reliable[i]:
            continue
        neighbor_mask = (
            (np.abs(depths - depths[i]) <= depth_window)
            & (np.arange(n_layers) != i)
            & base_reliable
        )
        if np.sum(neighbor_mask) >= min_neighbors:
            local_median = np.median(velocities[neighbor_mask])
            deviation = abs(velocities[i] - local_median)
            if deviation > vel_outlier_threshold:
                velocity_inlier[i] = False
    
    # A layer is quality-filtered if:
    #   - it fails BOTH slope_break AND vel_outlier, OR
    #   - it is a severe velocity outlier (>0.30 m/yr from neighbors)
    severe_outlier = np.zeros(n_layers, dtype=bool)
    for i in range(n_layers):
        if not base_reliable[i]:
            continue
        neighbor_mask = (
            (np.abs(depths - depths[i]) <= depth_window)
            & (np.arange(n_layers) != i)
            & base_reliable
        )
        if np.sum(neighbor_mask) >= min_neighbors:
            local_median = np.median(velocities[neighbor_mask])
            if abs(velocities[i] - local_median) > severe_outlier_threshold:
                severe_outlier[i] = True
    
    # 4c: Flatline detection — displacement stuck at constant value
    # When the unwrapper fails to accumulate phase (low coherence at depth),
    # it produces long runs of identical displacement. A layer with >30%
    # consecutive constant values at start or end is a tracking failure.
    flatline_threshold = 0.30  # fraction of record
    not_flatlined = np.ones(n_layers, dtype=bool)
    flatline_pct = np.zeros(n_layers)
    
    for i in range(n_layers):
        if not base_reliable[i]:
            continue
        rc = phase_data['range_timeseries'][i, :]
        # Consecutive constant from start
        n_const_start = 0
        for j in range(1, n_times):
            if rc[j] == rc[j-1]:
                n_const_start += 1
            else:
                break
        # Consecutive constant from end
        n_const_end = 0
        for j in range(n_times - 2, -1, -1):
            if rc[j] == rc[j+1]:
                n_const_end += 1
            else:
                break
        max_pct = max(n_const_start, n_const_end) / n_times
        flatline_pct[i] = max_pct
        if max_pct > flatline_threshold:
            not_flatlined[i] = False
    
    quality_pass = ~(
        (~slope_consistent & ~velocity_inlier)  # both flags
        | severe_outlier                          # or severe outlier alone
        | ~not_flatlined                          # or flatline failure
    )
    
    n_slope_fail = int(np.sum(~slope_consistent & base_reliable))
    n_vel_outlier = int(np.sum(~velocity_inlier & base_reliable))
    n_flatlined = int(np.sum(~not_flatlined & base_reliable))
    n_quality_removed = int(np.sum(~quality_pass & base_reliable))
    
    print(f"\n  Quality filters (of {int(np.sum(base_reliable))} base-reliable layers):")
    print(f"    Slope break (>{slope_break_threshold} m/yr): {n_slope_fail} flagged")
    print(f"    Velocity outlier (>{vel_outlier_threshold} m/yr from neighbors): {n_vel_outlier} flagged")
    print(f"    Severe outlier (>{severe_outlier_threshold} m/yr): {int(np.sum(severe_outlier & base_reliable))} flagged")
    print(f"    Flatline (>{flatline_threshold*100:.0f}% constant): {n_flatlined} flagged")
    print(f"    Removed (both slope+vel, or severe, or flatline): {n_quality_removed}")
    if n_quality_removed > 0:
        removed_idx = np.where(~quality_pass & base_reliable)[0]
        for idx in removed_idx:
            flags = []
            if not slope_consistent[idx]:
                flags.append("SLOPE_BREAK")
            if not velocity_inlier[idx]:
                flags.append("VEL_OUTLIER")
            if severe_outlier[idx]:
                flags.append("SEVERE")
            if not not_flatlined[idx]:
                flags.append(f"FLATLINE({flatline_pct[idx]*100:.0f}%)")
            print(f"      Layer {idx} ({depths[idx]:.1f}m): vel={velocities[idx]:.3f} m/yr, "
                  f"R²={r_squared[idx]:.3f}, flags={'+'.join(flags)}")
    
    # Determine reliable measurements
    reliable = base_reliable & quality_pass
    
    # Smooth velocity profile (only using reliable points)
    velocities_smooth = np.full(n_layers, np.nan)
    if np.sum(reliable) > 3:
        # Interpolate gaps and smooth
        vel_interp = np.interp(
            depths,
            depths[reliable],
            velocities[reliable],
        )
        velocities_smooth = gaussian_filter1d(vel_interp, smooth_sigma)
    
    result = VelocityProfileResult(
        depths=depths,
        velocities=velocities,
        velocities_smooth=velocities_smooth,
        r_squared=r_squared,
        std_error=std_error,
        amplitude_mean=amp_mean,
        reliable=reliable,
        n_layers=n_layers,
        r_sq_threshold=r_sq_threshold,
        amp_threshold=amp_threshold_db,
        uncertainty_kingslake=uncertainty_kingslake,
        uncertainty_wls=uncertainty_wls,
        range_uncertainty_mean=range_uncertainty_mean,
        snr_mean=snr_mean,
        tracking_mode=tracking_mode,
        slope_consistent=slope_consistent,
        velocity_inlier=velocity_inlier,
        quality_pass=quality_pass,
    )
    
    n_reliable = np.sum(reliable)
    print(f"\nVelocity profile complete:")
    print(f"  Total layers: {n_layers}")
    print(f"  Reliable (R² > {r_sq_threshold}, Amp > {amp_threshold_db} dB): {n_reliable}")
    if n_reliable > 0:
        print(f"  Velocity range: {np.min(velocities[reliable]):.3f} to {np.max(velocities[reliable]):.3f} m/year")
        print(f"\n  Kingslake uncertainty (reliable layers):")
        print(f"    Median: {np.median(uncertainty_kingslake[reliable]):.4f} m/yr")
        print(f"    Range:  {np.min(uncertainty_kingslake[reliable]):.4f} to {np.max(uncertainty_kingslake[reliable]):.4f} m/yr")
        print(f"  WLS uncertainty (reliable layers):")
        print(f"    Median: {np.median(uncertainty_wls[reliable]):.4f} m/yr")
        print(f"    Range:  {np.min(uncertainty_wls[reliable]):.4f} to {np.max(uncertainty_wls[reliable]):.4f} m/yr")
        print(f"  Mean range uncertainty (reliable):")
        print(f"    Median: {np.median(range_uncertainty_mean[reliable])*1000:.2f} mm")
        print(f"    Range:  {np.min(range_uncertainty_mean[reliable])*1000:.2f} to {np.max(range_uncertainty_mean[reliable])*1000:.2f} mm")
    
    return result


def visualize_velocity_profile(
    result: VelocityProfileResult,
    output_file: Optional[str] = None,
):
    """
    Create comprehensive velocity profile visualization with Kingslake uncertainties.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    has_uncertainty = (result.uncertainty_kingslake is not None 
                       and result.uncertainty_wls is not None)
    
    n_cols = 5 if has_uncertainty else 4
    titles = ['Velocity Profile', 'R² Quality', 'Amplitude', 'Velocity vs R²']
    if has_uncertainty:
        titles.append('Uncertainty & SNR')
    
    col_widths = [0.25, 0.15, 0.15, 0.20, 0.25] if has_uncertainty else None
    
    fig = make_subplots(
        rows=1, cols=n_cols,
        subplot_titles=titles,
        horizontal_spacing=0.06,
        column_widths=col_widths,
    )
    
    # Choose which uncertainty to use for error bars
    if has_uncertainty:
        # Use WLS uncertainty (accounts for both SNR and residual scatter)
        error_bars = result.uncertainty_wls
        error_label = 'WLS'
    else:
        error_bars = result.std_error
        error_label = 'OLS'
    
    # 1. Velocity profile with Kingslake uncertainty bars
    # Unreliable points
    unreliable = ~result.reliable
    if np.any(unreliable):
        fig.add_trace(
            go.Scatter(
                x=result.velocities[unreliable],
                y=result.depths[unreliable],
                mode='markers',
                marker=dict(color='lightgray', size=6, symbol='circle-open'),
                name='Unreliable',
                hovertemplate='Depth: %{y:.0f} m<br>v = %{x:.3f} m/yr<extra>Unreliable</extra>',
            ),
            row=1, col=1
        )
    
    # Reliable points with uncertainty error bars
    if np.any(result.reliable):
        fig.add_trace(
            go.Scatter(
                x=result.velocities[result.reliable],
                y=result.depths[result.reliable],
                mode='markers',
                marker=dict(
                    color=result.r_squared[result.reliable],
                    colorscale='Viridis',
                    cmin=result.r_sq_threshold,
                    cmax=1.0,
                    size=8,
                    colorbar=dict(title='R²', x=0.18 if has_uncertainty else 0.23, len=0.9),
                ),
                name=f'Reliable (±σ {error_label})',
                error_x=dict(
                    type='data',
                    array=error_bars[result.reliable],
                    visible=True,
                    color='rgba(100,100,100,0.4)',
                    thickness=1.5,
                ),
                hovertemplate='Depth: %{y:.0f} m<br>v = %{x:.3f} ± %{error_x.array:.4f} m/yr<extra></extra>',
            ),
            row=1, col=1
        )
    
    # Smoothed profile
    if not np.all(np.isnan(result.velocities_smooth)):
        fig.add_trace(
            go.Scatter(
                x=result.velocities_smooth,
                y=result.depths,
                mode='lines',
                line=dict(color='red', width=3),
                name='Smoothed',
            ),
            row=1, col=1
        )
    
    # Zero line
    fig.add_vline(x=0, line=dict(color='black', dash='dash', width=1), row=1, col=1)
    
    # 2. R² quality
    fig.add_trace(
        go.Scatter(
            x=result.r_squared,
            y=result.depths,
            mode='markers+lines',
            marker=dict(
                color=np.where(result.reliable, 'green', 'red'),
                size=6,
            ),
            line=dict(color='gray', width=0.5),
            showlegend=False,
            hovertemplate='Depth: %{y:.0f} m<br>R² = %{x:.3f}<extra></extra>',
        ),
        row=1, col=2
    )
    fig.add_vline(x=result.r_sq_threshold, line=dict(color='red', dash='dot'), row=1, col=2)
    
    # 3. Amplitude
    fig.add_trace(
        go.Scatter(
            x=result.amplitude_mean,
            y=result.depths,
            mode='markers+lines',
            marker=dict(color='blue', size=4),
            line=dict(color='blue', width=0.5),
            showlegend=False,
            hovertemplate='Depth: %{y:.0f} m<br>Amp = %{x:.1f} dB<extra></extra>',
        ),
        row=1, col=3
    )
    fig.add_vline(x=result.amp_threshold, line=dict(color='red', dash='dot'), row=1, col=3)
    
    # 4. Velocity vs R² scatter
    fig.add_trace(
        go.Scatter(
            x=result.r_squared,
            y=result.velocities,
            mode='markers',
            marker=dict(
                color=result.depths,
                colorscale='Viridis',
                size=8,
                colorbar=dict(title='Depth (m)', x=1.02),
            ),
            showlegend=False,
            hovertemplate='R² = %{x:.3f}<br>v = %{y:.2f} m/yr<br>Depth: %{marker.color:.0f} m<extra></extra>',
        ),
        row=1, col=4
    )
    fig.add_vline(x=result.r_sq_threshold, line=dict(color='red', dash='dot'), row=1, col=4)
    
    # 5. Uncertainty & SNR panel (if Kingslake uncertainty available)
    if has_uncertainty:
        # Kingslake conservative uncertainty
        fig.add_trace(
            go.Scatter(
                x=result.uncertainty_kingslake,
                y=result.depths,
                mode='markers+lines',
                marker=dict(
                    color=np.where(result.reliable, '#ef4444', '#fca5a5'),
                    size=5,
                ),
                line=dict(color='#fca5a5', width=0.5),
                name='σ Kingslake',
                hovertemplate='Depth: %{y:.0f} m<br>σ_K = %{x:.4f} m/yr<extra>Kingslake conservative</extra>',
            ),
            row=1, col=5
        )
        
        # WLS uncertainty
        fig.add_trace(
            go.Scatter(
                x=result.uncertainty_wls,
                y=result.depths,
                mode='markers+lines',
                marker=dict(
                    color=np.where(result.reliable, '#3b82f6', '#93c5fd'),
                    size=5,
                ),
                line=dict(color='#93c5fd', width=0.5),
                name='σ WLS',
                hovertemplate='Depth: %{y:.0f} m<br>σ_WLS = %{x:.4f} m/yr<extra>WLS uncertainty</extra>',
            ),
            row=1, col=5
        )
        
        # Add SNR as secondary information (via marker size or separate axis)
        if result.snr_mean is not None:
            # Create a secondary x-axis trace for SNR using a separate trace
            fig.add_trace(
                go.Scatter(
                    x=result.range_uncertainty_mean * 1000,  # Convert to mm
                    y=result.depths,
                    mode='markers',
                    marker=dict(
                        color='#22c55e',
                        size=4,
                        symbol='diamond',
                    ),
                    name='σ_r (mm)',
                    hovertemplate='Depth: %{y:.0f} m<br>σ_r = %{x:.2f} mm<extra>Range uncertainty</extra>',
                    visible='legendonly',  # Hidden by default, click legend to show
                ),
                row=1, col=5
            )
        
        fig.update_yaxes(autorange='reversed', row=1, col=5)
        fig.update_xaxes(title_text='Uncertainty (m/yr)', type='log', row=1, col=5)
    
    # Update axes
    fig.update_yaxes(title_text='Depth (m)', autorange='reversed', row=1, col=1)
    fig.update_xaxes(title_text='Velocity (m/yr)', row=1, col=1)
    
    fig.update_yaxes(autorange='reversed', row=1, col=2)
    fig.update_xaxes(title_text='R²', range=[0, 1], row=1, col=2)
    
    fig.update_yaxes(autorange='reversed', row=1, col=3)
    fig.update_xaxes(title_text='Amplitude (dB)', row=1, col=3)
    
    fig.update_xaxes(title_text='R²', range=[0, 1], row=1, col=4)
    fig.update_yaxes(title_text='Velocity (m/yr)', row=1, col=4)
    
    # Build title with uncertainty info
    title_text = f'Internal Ice Layer Velocity Profile ({np.sum(result.reliable)}/{result.n_layers} reliable layers)'
    if has_uncertainty and np.any(result.reliable):
        median_unc = np.median(result.uncertainty_wls[result.reliable])
        title_text += f' — Median σ_WLS = {median_unc:.4f} m/yr ({median_unc*100:.2f} cm/yr)'
    
    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(size=14),
        ),
        height=700,
        width=1600 if has_uncertainty else 1400,
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
    )
    
    if output_file:
        fig.write_html(output_file)
        print(f"Saved: {output_file}")
    
    fig.show()
    return fig


def save_velocity_results(result: VelocityProfileResult, output_path: str,
                          deep_layers_path: Optional[str] = None) -> None:
    """Save velocity profile results including Kingslake uncertainties.
    
    If deep_layers_path is provided, deep layer detections are merged
    into the JSON output with tracking_mode='deep_segment_stitched'.
    """
    # Save as .mat
    mat_data = {
        'depths': result.depths,
        'velocities': result.velocities,
        'velocities_smooth': result.velocities_smooth,
        'r_squared': result.r_squared,
        'std_error': result.std_error,
        'amplitude_mean': result.amplitude_mean,
        'reliable': result.reliable.astype(int),
        'n_layers': result.n_layers,
        'r_sq_threshold': result.r_sq_threshold,
        'amp_threshold': result.amp_threshold,
    }
    # Add Kingslake uncertainty fields if available
    if result.uncertainty_kingslake is not None:
        mat_data['uncertainty_kingslake'] = result.uncertainty_kingslake
    if result.uncertainty_wls is not None:
        mat_data['uncertainty_wls'] = result.uncertainty_wls
    if result.range_uncertainty_mean is not None:
        mat_data['range_uncertainty_mean'] = result.range_uncertainty_mean
    if result.snr_mean is not None:
        mat_data['snr_mean'] = result.snr_mean
    if result.tracking_mode is not None:
        mat_data['tracking_mode'] = result.tracking_mode
    if result.slope_consistent is not None:
        mat_data['slope_consistent'] = result.slope_consistent.astype(int)
    if result.velocity_inlier is not None:
        mat_data['velocity_inlier'] = result.velocity_inlier.astype(int)
    if result.quality_pass is not None:
        mat_data['quality_pass'] = result.quality_pass.astype(int)
    savemat(f"{output_path}.mat", mat_data)
    
    # Save summary as JSON
    reliable_mask = result.reliable
    json_data = {
        'n_layers': int(result.n_layers),
        'n_reliable': int(np.sum(result.reliable)),
        'r_sq_threshold': float(result.r_sq_threshold),
        'amp_threshold_db': float(result.amp_threshold),
        'velocity_stats': {
            'min': float(np.nanmin(result.velocities[reliable_mask])) if np.any(reliable_mask) else None,
            'max': float(np.nanmax(result.velocities[reliable_mask])) if np.any(reliable_mask) else None,
            'mean': float(np.nanmean(result.velocities[reliable_mask])) if np.any(reliable_mask) else None,
        },
        'uncertainty_method': 'Kingslake et al. (2014) phasor-based SNR uncertainty',
        'uncertainty_stats': {},
        'layers': [],
    }
    
    # Add uncertainty statistics if available
    if result.uncertainty_kingslake is not None and np.any(reliable_mask):
        unc_k = result.uncertainty_kingslake[reliable_mask]
        json_data['uncertainty_stats']['kingslake_conservative'] = {
            'median_m_yr': float(np.nanmedian(unc_k)),
            'min_m_yr': float(np.nanmin(unc_k)),
            'max_m_yr': float(np.nanmax(unc_k)),
            'description': 'Conservative estimate: sum of position uncertainties / time span',
        }
    if result.uncertainty_wls is not None and np.any(reliable_mask):
        unc_w = result.uncertainty_wls[reliable_mask]
        json_data['uncertainty_stats']['weighted_least_squares'] = {
            'median_m_yr': float(np.nanmedian(unc_w)),
            'min_m_yr': float(np.nanmin(unc_w)),
            'max_m_yr': float(np.nanmax(unc_w)),
            'description': 'WLS with per-measurement weights from SNR, scaled by reduced chi²',
        }
    if result.range_uncertainty_mean is not None and np.any(reliable_mask):
        rng_unc = result.range_uncertainty_mean[reliable_mask]
        json_data['uncertainty_stats']['range_uncertainty'] = {
            'median_mm': float(np.nanmedian(rng_unc) * 1000),
            'min_mm': float(np.nanmin(rng_unc) * 1000),
            'max_mm': float(np.nanmax(rng_unc) * 1000),
            'description': 'Mean per-measurement range uncertainty from phasor geometry',
        }
    
    # Per-layer data
    for i in range(result.n_layers):
        layer_entry = {
            'depth_m': float(result.depths[i]),
            'velocity_m_yr': float(result.velocities[i]) if not np.isnan(result.velocities[i]) else None,
            'r_squared': float(result.r_squared[i]),
            'amplitude_db': float(result.amplitude_mean[i]),
            'reliable': bool(result.reliable[i]),
        }
        if result.uncertainty_kingslake is not None:
            layer_entry['uncertainty_kingslake_m_yr'] = float(result.uncertainty_kingslake[i])
        if result.uncertainty_wls is not None:
            layer_entry['uncertainty_wls_m_yr'] = float(result.uncertainty_wls[i])
        if result.range_uncertainty_mean is not None:
            layer_entry['range_uncertainty_mm'] = float(result.range_uncertainty_mean[i] * 1000)
        if result.snr_mean is not None:
            layer_entry['snr_db'] = float(result.snr_mean[i])
        if result.tracking_mode is not None:
            layer_entry['tracking_mode'] = 'phase' if result.tracking_mode[i] == 1 else 'speckle'
        if result.quality_pass is not None:
            layer_entry['quality_pass'] = bool(result.quality_pass[i])
        if result.slope_consistent is not None:
            layer_entry['slope_consistent'] = bool(result.slope_consistent[i])
        if result.velocity_inlier is not None:
            layer_entry['velocity_inlier'] = bool(result.velocity_inlier[i])
        json_data['layers'].append(layer_entry)
    
    # Merge deep layers if provided
    if deep_layers_path is not None:
        import os
        if os.path.exists(deep_layers_path):
            with open(deep_layers_path, 'r') as f:
                deep_data = json.load(f)
            deep_layer_list = deep_data.get('layers', [])
            n_deep = len(deep_layer_list)
            json_data['n_deep_layers'] = n_deep
            json_data['deep_layer_method'] = deep_data.get('method', 'unknown')
            json_data['deep_layer_description'] = deep_data.get('description', '')
            json_data['deep_layer_nye_model'] = deep_data.get('nye_model', {})
            json_data['deep_layer_summary'] = deep_data.get('summary', {})
            for dl in deep_layer_list:
                layer_entry = {
                    'depth_m': dl['depth_m'],
                    'velocity_m_yr': dl['velocity_m_yr'],
                    'r_squared': dl['r_squared'],
                    'amplitude_db': None,
                    'reliable': True,
                    'tracking_mode': 'deep_segment_stitched',
                    'quality_pass': True,
                    'n_tracked_pts': dl.get('n_tracked_pts', 0),
                    'n_segments': dl.get('n_segments', 1),
                    'nye_velocity_m_yr': dl.get('nye_velocity_m_yr', None),
                    'quality_tier': dl.get('quality_tier', 3),
                    'total_elevated_frac': dl.get('total_elevated_frac', None),
                }
                json_data['layers'].append(layer_entry)
            print(f"  Merged {n_deep} deep layers from {deep_layers_path}")
        else:
            print(f"  Warning: deep layers file not found: {deep_layers_path}")
    
    with open(f"{output_path}.json", 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Saved: {output_path}.mat and {output_path}.json")


# =============================================================================
# Main entry point
# =============================================================================

def main():
    """Calculate velocity profile from phase tracking data."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate velocity profile from phase tracking')
    parser.add_argument('--phase', type=str, default='../../data/apres/phase_tracking',
                        help='Path to phase tracking results (without extension)')
    parser.add_argument('--output', type=str, default='../../data/apres/velocity_profile',
                        help='Output file path (without extension)')
    parser.add_argument('--r-sq-threshold', type=float, default=0.3,
                        help='Minimum R² for reliable velocity')
    parser.add_argument('--amp-threshold', type=float, default=-80,
                        help='Minimum amplitude threshold (dB)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip visualization')
    parser.add_argument('--deep-layers', type=str, default=None,
                        help='Path to deep_layers.json from deep_layer_detection.py')
    
    args = parser.parse_args()
    
    # Load data
    phase_data = load_phase_data(args.phase)
    
    # Calculate velocities
    result = calculate_velocity_profile(
        phase_data,
        r_sq_threshold=args.r_sq_threshold,
        amp_threshold_db=args.amp_threshold,
    )
    
    # Save results (with deep layers if provided)
    save_velocity_results(result, args.output, deep_layers_path=args.deep_layers)
    
    # Visualize
    if not args.no_plot:
        visualize_velocity_profile(
            result,
            output_file=f"{args.output}_visualization.html",
        )
    
    return result


if __name__ == '__main__':
    main()
