"""
Velocity Profile Module for ApRES Internal Ice Layer Analysis

This module calculates depth-dependent velocity profiles from phase tracking
results, following the methodology of Summers et al. (2021).

Key equation: v_r = f · λ_c (Eq. 1)

The range velocity v_r at each layer is derived from the linear trend in
phase/range over time.

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
    
    print(f"Loaded phase data for {data['n_layers']} layers")
    return data


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
) -> VelocityProfileResult:
    """
    Calculate velocity profile for all layers.
    
    Args:
        phase_data: Phase tracking results
        r_sq_threshold: Minimum R² for reliable velocity
        amp_threshold_db: Minimum amplitude for reliable phase (dB)
        smooth_sigma: Gaussian smoothing sigma (in layer indices)
        
    Returns:
        VelocityProfileResult with velocity profile
    """
    n_layers = phase_data['n_layers']
    depths = phase_data['layer_depths']
    time_days = phase_data['time_days']
    
    # Initialize arrays
    velocities = np.zeros(n_layers)
    r_squared = np.zeros(n_layers)
    std_error = np.zeros(n_layers)
    amp_mean = np.zeros(n_layers)
    
    print(f"\nCalculating velocities for {n_layers} layers...")
    
    for i in range(n_layers):
        # Get range change for this layer
        range_change = phase_data['range_timeseries'][i, :]
        
        # Calculate velocity
        vel, r_sq, std_err = calculate_layer_velocity(time_days, range_change)
        
        velocities[i] = vel
        r_squared[i] = r_sq
        std_error[i] = std_err
        
        # Mean amplitude in dB
        amp = phase_data['amplitude_timeseries'][i, :]
        amp_mean[i] = 10 * np.log10(np.mean(amp**2) + 1e-30)
    
    # Determine reliable measurements
    reliable = (r_squared >= r_sq_threshold) & (amp_mean >= amp_threshold_db)
    
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
    )
    
    n_reliable = np.sum(reliable)
    print(f"\nVelocity profile complete:")
    print(f"  Total layers: {n_layers}")
    print(f"  Reliable (R² > {r_sq_threshold}, Amp > {amp_threshold_db} dB): {n_reliable}")
    if n_reliable > 0:
        print(f"  Velocity range: {np.min(velocities[reliable]):.2f} to {np.max(velocities[reliable]):.2f} m/year")
    
    return result


def visualize_velocity_profile(
    result: VelocityProfileResult,
    output_file: Optional[str] = None,
):
    """
    Create comprehensive velocity profile visualization.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=(
            'Velocity Profile',
            'R² Quality',
            'Amplitude',
            'Velocity vs R²'
        ),
        horizontal_spacing=0.08,
    )
    
    # 1. Velocity profile
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
                hovertemplate='Depth: %{y:.0f} m<br>v = %{x:.2f} m/yr<extra>Unreliable</extra>',
            ),
            row=1, col=1
        )
    
    # Reliable points
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
                    colorbar=dict(title='R²', x=0.23, len=0.9),
                ),
                name='Reliable',
                error_x=dict(
                    type='data',
                    array=result.std_error[result.reliable],
                    visible=True,
                    color='rgba(100,100,100,0.3)',
                ),
                hovertemplate='Depth: %{y:.0f} m<br>v = %{x:.2f} ± %{error_x.array:.2f} m/yr<extra></extra>',
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
    
    # Update axes
    fig.update_yaxes(title_text='Depth (m)', autorange='reversed', row=1, col=1)
    fig.update_xaxes(title_text='Velocity (m/yr)', row=1, col=1)
    
    fig.update_yaxes(autorange='reversed', row=1, col=2)
    fig.update_xaxes(title_text='R²', range=[0, 1], row=1, col=2)
    
    fig.update_yaxes(autorange='reversed', row=1, col=3)
    fig.update_xaxes(title_text='Amplitude (dB)', row=1, col=3)
    
    fig.update_xaxes(title_text='R²', range=[0, 1], row=1, col=4)
    fig.update_yaxes(title_text='Velocity (m/yr)', row=1, col=4)
    
    fig.update_layout(
        title=dict(
            text=f'Internal Ice Layer Velocity Profile ({np.sum(result.reliable)}/{result.n_layers} reliable layers)',
            font=dict(size=16),
        ),
        height=700,
        width=1400,
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
    )
    
    if output_file:
        fig.write_html(output_file)
        print(f"Saved: {output_file}")
    
    fig.show()
    return fig


def save_velocity_results(result: VelocityProfileResult, output_path: str) -> None:
    """Save velocity profile results."""
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
        'layers': [
            {
                'depth_m': float(result.depths[i]),
                'velocity_m_yr': float(result.velocities[i]) if not np.isnan(result.velocities[i]) else None,
                'r_squared': float(result.r_squared[i]),
                'amplitude_db': float(result.amplitude_mean[i]),
                'reliable': bool(result.reliable[i]),
            }
            for i in range(result.n_layers)
        ]
    }
    
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
    
    args = parser.parse_args()
    
    # Load data
    phase_data = load_phase_data(args.phase)
    
    # Calculate velocities
    result = calculate_velocity_profile(
        phase_data,
        r_sq_threshold=args.r_sq_threshold,
        amp_threshold_db=args.amp_threshold,
    )
    
    # Save results
    save_velocity_results(result, args.output)
    
    # Visualize
    if not args.no_plot:
        visualize_velocity_profile(
            result,
            output_file=f"{args.output}_visualization.html",
        )
    
    return result


if __name__ == '__main__':
    main()
