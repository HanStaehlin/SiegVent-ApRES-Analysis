"""
Layer Detection Module for ApRES Internal Ice Layer Analysis

This module identifies and tracks specular internal ice layers from ApRES data
by detecting amplitude peaks in range profiles and tracking their persistence.

Based on methodology from:
- Summers et al. (2021) - IGARSS conference paper on velocity profiles

Author: SiegVent2023 project
"""

import numpy as np
from scipy import signal, ndimage
from scipy.io import loadmat, savemat
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Optional
import json


@dataclass
class LayerDetectionResult:
    """Container for layer detection results."""
    layer_depths: np.ndarray          # Depth of each detected layer (m)
    layer_amplitudes: np.ndarray      # Mean amplitude at each layer (linear)
    layer_snr: np.ndarray             # Signal-to-noise ratio (dB)
    layer_persistence: np.ndarray     # Fraction of time steps layer is visible
    layer_indices: np.ndarray         # Range bin indices for each layer
    amplitude_threshold: float        # SNR threshold used (dB)
    n_layers: int                     # Number of layers detected
    depth_range: Tuple[float, float]  # Analysis depth range (m)


def load_apres_data(data_path: str) -> dict:
    """
    Load processed ApRES data from .mat file.
    
    Args:
        data_path: Path to ImageP2_python.mat or similar
        
    Returns:
        Dictionary with RawImage, RfineBarTime, Rcoarse, TimeInDays
    """
    mat_data = loadmat(data_path)
    
    # Prefer RawImageComplex if available (has better dynamic range)
    if 'RawImageComplex' in mat_data:
        range_img = np.abs(np.array(mat_data['RawImageComplex']))
    else:
        range_img = np.array(mat_data['RawImage'])
    
    # Extract arrays, squeezing out singleton dimensions
    data = {
        'range_img': range_img,
        'Rfine': np.array(mat_data['RfineBarTime']) if 'RfineBarTime' in mat_data else None,
        'Rcoarse': np.array(mat_data['Rcoarse']).flatten(),
        'time_days': np.array(mat_data['TimeInDays']).flatten(),
    }
    
    print(f"Loaded ApRES data:")
    print(f"  Range bins: {data['range_img'].shape[0]}")
    print(f"  Time steps: {data['range_img'].shape[1]}")
    print(f"  Depth range: {data['Rcoarse'][0]:.1f} - {data['Rcoarse'][-1]:.1f} m")
    print(f"  Time span: {data['time_days'][-1]:.1f} days")
    
    return data


def compute_echogram_db(range_img: np.ndarray) -> np.ndarray:
    """Convert amplitude to dB scale."""
    return 10 * np.log10(range_img**2 + 1e-30)


def compute_mean_profile(range_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and std of range profiles over time.
    
    Returns:
        mean_profile: Mean amplitude (linear)
        std_profile: Standard deviation (linear)
    """
    mean_profile = np.mean(range_img, axis=1)
    std_profile = np.std(range_img, axis=1)
    return mean_profile, std_profile


def estimate_noise_floor(profile_db: np.ndarray, 
                         percentile: float = 10) -> float:
    """
    Estimate noise floor from profile.
    
    Uses lower percentile of amplitude as noise estimate.
    """
    return np.percentile(profile_db, percentile)


def detect_layers(
    range_img: np.ndarray,
    Rcoarse: np.ndarray,
    min_depth: float = 50,
    max_depth: float = 1000,
    min_snr_db: float = 10,
    min_prominence_db: float = 3,
    min_separation_m: float = 5,
    min_persistence: float = 0.5,
    smooth_window: int = 3,
) -> LayerDetectionResult:
    """
    Detect specular internal ice layers.
    
    Algorithm:
    1. Compute time-averaged amplitude profile
    2. Find peaks above noise floor with sufficient prominence
    3. Track each peak through time to assess persistence
    4. Filter layers by SNR and persistence criteria
    
    Args:
        range_img: Amplitude profiles [n_bins, n_times]
        Rcoarse: Depth/range vector (m)
        min_depth: Minimum depth to analyze (m)
        max_depth: Maximum depth to analyze (m)
        min_snr_db: Minimum SNR threshold (dB above noise floor)
        min_prominence_db: Minimum peak prominence (dB)
        min_separation_m: Minimum separation between layers (m)
        min_persistence: Minimum fraction of time layer must be visible
        smooth_window: Smoothing window for profile (bins)
        
    Returns:
        LayerDetectionResult with detected layers
    """
    n_bins, n_times = range_img.shape
    bin_spacing = Rcoarse[1] - Rcoarse[0]
    
    # Define depth mask
    depth_mask = (Rcoarse >= min_depth) & (Rcoarse <= max_depth)
    depth_indices = np.where(depth_mask)[0]
    
    # Compute mean profile in dB
    mean_profile = np.mean(range_img, axis=1)
    if smooth_window > 1:
        mean_profile = ndimage.uniform_filter1d(mean_profile, smooth_window)
    mean_profile_db = 10 * np.log10(mean_profile**2 + 1e-30)
    
    # Estimate noise floor from the analyzed region
    noise_floor_db = estimate_noise_floor(mean_profile_db[depth_mask])
    
    # Find peaks in the mean profile
    min_distance = max(1, int(min_separation_m / bin_spacing))
    
    # Convert prominence threshold to linear scale for scipy
    # But we'll work in dB for easier interpretation
    peaks, properties = signal.find_peaks(
        mean_profile_db[depth_mask],
        height=noise_floor_db + min_snr_db,
        prominence=min_prominence_db,
        distance=min_distance,
    )
    
    # Map peak indices back to full array
    peak_indices = depth_indices[peaks]
    peak_depths = Rcoarse[peak_indices]
    peak_amplitudes = mean_profile[peak_indices]
    peak_heights = properties['peak_heights']
    peak_snr = peak_heights - noise_floor_db
    
    print(f"\nInitial peak detection:")
    print(f"  Noise floor: {noise_floor_db:.1f} dB")
    print(f"  SNR threshold: {min_snr_db:.1f} dB")
    print(f"  Peaks found: {len(peaks)}")
    
    # Track persistence of each layer through time
    persistence = np.zeros(len(peak_indices))
    
    for i, idx in enumerate(peak_indices):
        # Define search window around this depth
        window = max(3, int(2 / bin_spacing))  # ~2m window
        idx_min = max(0, idx - window)
        idx_max = min(n_bins, idx + window + 1)
        
        # For each time step, check if there's a strong peak near this depth
        for t in range(n_times):
            local_profile = range_img[idx_min:idx_max, t]
            local_profile_db = 10 * np.log10(local_profile**2 + 1e-30)
            local_max_db = np.max(local_profile_db)
            
            # Layer is "visible" if local max exceeds threshold
            if local_max_db > noise_floor_db + min_snr_db * 0.7:
                persistence[i] += 1
    
    persistence /= n_times  # Convert to fraction
    
    # Filter by persistence
    valid_mask = persistence >= min_persistence
    
    result = LayerDetectionResult(
        layer_depths=peak_depths[valid_mask],
        layer_amplitudes=peak_amplitudes[valid_mask],
        layer_snr=peak_snr[valid_mask],
        layer_persistence=persistence[valid_mask],
        layer_indices=peak_indices[valid_mask],
        amplitude_threshold=min_snr_db,
        n_layers=np.sum(valid_mask),
        depth_range=(min_depth, max_depth),
    )
    
    print(f"\nAfter persistence filtering (>{min_persistence*100:.0f}%):")
    print(f"  Layers retained: {result.n_layers}")
    if result.n_layers > 0:
        print(f"  Depth range: {result.layer_depths.min():.1f} - {result.layer_depths.max():.1f} m")
        print(f"  SNR range: {result.layer_snr.min():.1f} - {result.layer_snr.max():.1f} dB")
    
    return result


def visualize_layers(
    range_img: np.ndarray,
    Rcoarse: np.ndarray,
    time_days: np.ndarray,
    layers: LayerDetectionResult,
    output_file: Optional[str] = None,
):
    """
    Create visualization of detected layers using Plotly.
    
    Generates:
    1. Echogram with layer markers
    2. Mean amplitude profile with peaks
    3. Layer SNR vs depth
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Compute echogram in dB
    echogram_db = compute_echogram_db(range_img)
    
    # Subsample for visualization if too large
    max_time_points = 500
    if len(time_days) > max_time_points:
        step = len(time_days) // max_time_points
        time_sub = time_days[::step]
        echo_sub = echogram_db[:, ::step]
    else:
        time_sub = time_days
        echo_sub = echogram_db
    
    # Focus on analysis depth range
    depth_mask = (Rcoarse >= layers.depth_range[0]) & (Rcoarse <= layers.depth_range[1])
    
    # Create figure with subplots
    fig = make_subplots(
        rows=1, cols=3,
        column_widths=[0.5, 0.25, 0.25],
        subplot_titles=(
            'Echogram with Detected Layers',
            'Mean Amplitude Profile',
            'Layer SNR vs Depth'
        ),
        horizontal_spacing=0.08,
    )
    
    # 1. Echogram heatmap
    fig.add_trace(
        go.Heatmap(
            x=time_sub,
            y=Rcoarse[depth_mask],
            z=echo_sub[depth_mask, :],
            colorscale='Viridis',
            colorbar=dict(title='dB', x=0.48, len=0.9),
            hovertemplate='Time: %{x:.1f} days<br>Depth: %{y:.1f} m<br>Power: %{z:.1f} dB<extra></extra>',
        ),
        row=1, col=1
    )
    
    # Add horizontal lines for detected layers
    for i, depth in enumerate(layers.layer_depths):
        fig.add_hline(
            y=depth, 
            line=dict(color='red', width=1, dash='dot'),
            row=1, col=1,
        )
    
    # 2. Mean amplitude profile
    mean_profile = np.mean(range_img, axis=1)
    mean_profile_db = 10 * np.log10(mean_profile**2 + 1e-30)
    
    fig.add_trace(
        go.Scatter(
            x=mean_profile_db[depth_mask],
            y=Rcoarse[depth_mask],
            mode='lines',
            line=dict(color='blue', width=1),
            name='Mean Profile',
            hovertemplate='%{x:.1f} dB at %{y:.1f} m<extra></extra>',
        ),
        row=1, col=2
    )
    
    # Mark detected layers on profile
    layer_db = 10 * np.log10(layers.layer_amplitudes**2 + 1e-30)
    fig.add_trace(
        go.Scatter(
            x=layer_db,
            y=layers.layer_depths,
            mode='markers',
            marker=dict(color='red', size=8, symbol='diamond'),
            name=f'Layers (n={layers.n_layers})',
            hovertemplate='Layer at %{y:.1f} m<br>%{x:.1f} dB<extra></extra>',
        ),
        row=1, col=2
    )
    
    # 3. Layer SNR vs depth
    fig.add_trace(
        go.Scatter(
            x=layers.layer_snr,
            y=layers.layer_depths,
            mode='markers+text',
            marker=dict(
                color=layers.layer_persistence,
                colorscale='RdYlGn',
                size=10,
                colorbar=dict(title='Persistence', x=1.02, len=0.9),
                cmin=0.5, cmax=1.0,
            ),
            text=[f'{snr:.0f}' for snr in layers.layer_snr],
            textposition='middle right',
            textfont=dict(size=9),
            name='Layer SNR',
            hovertemplate='Depth: %{y:.1f} m<br>SNR: %{x:.1f} dB<br>Persistence: %{marker.color:.0%}<extra></extra>',
        ),
        row=1, col=3
    )
    
    # Update layout
    fig.update_yaxes(title_text='Depth (m)', autorange='reversed', row=1, col=1)
    fig.update_xaxes(title_text='Time (days)', row=1, col=1)
    
    fig.update_yaxes(title_text='Depth (m)', autorange='reversed', row=1, col=2)
    fig.update_xaxes(title_text='Power (dB)', row=1, col=2)
    
    fig.update_yaxes(title_text='Depth (m)', autorange='reversed', row=1, col=3)
    fig.update_xaxes(title_text='SNR (dB)', row=1, col=3)
    
    fig.update_layout(
        title=dict(
            text=f'ApRES Internal Layer Detection: {layers.n_layers} Layers Found',
            font=dict(size=16),
        ),
        height=600,
        width=1400,
        showlegend=True,
        legend=dict(x=0.35, y=1.15, orientation='h'),
    )
    
    if output_file:
        fig.write_html(output_file)
        print(f"Saved visualization: {output_file}")
    
    fig.show()
    return fig


def save_layers(layers: LayerDetectionResult, output_path: str) -> None:
    """Save layer detection results to files."""
    # Save as .mat for compatibility
    mat_data = {
        'layer_depths': layers.layer_depths,
        'layer_amplitudes': layers.layer_amplitudes,
        'layer_snr': layers.layer_snr,
        'layer_persistence': layers.layer_persistence,
        'layer_indices': layers.layer_indices,
        'amplitude_threshold': layers.amplitude_threshold,
        'n_layers': layers.n_layers,
        'depth_range': np.array(layers.depth_range),
    }
    savemat(f"{output_path}.mat", mat_data)
    
    # Save as JSON for easy reading
    json_data = {
        'n_layers': int(layers.n_layers),
        'amplitude_threshold_db': float(layers.amplitude_threshold),
        'depth_range_m': list(layers.depth_range),
        'layers': [
            {
                'depth_m': float(layers.layer_depths[i]),
                'snr_db': float(layers.layer_snr[i]),
                'persistence': float(layers.layer_persistence[i]),
                'amplitude': float(layers.layer_amplitudes[i]),
            }
            for i in range(layers.n_layers)
        ]
    }
    with open(f"{output_path}.json", 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Saved: {output_path}.mat and {output_path}.json")


# =============================================================================
# Main entry point
# =============================================================================

def main():
    """Run layer detection on ApRES data."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect internal ice layers from ApRES data')
    parser.add_argument('--data', type=str, default='../../data/apres/ImageP2_python.mat',
                        help='Path to processed ApRES data (.mat)')
    parser.add_argument('--min-depth', type=float, default=50,
                        help='Minimum depth to analyze (m)')
    parser.add_argument('--max-depth', type=float, default=1000,
                        help='Maximum depth to analyze (m)')
    parser.add_argument('--min-snr', type=float, default=10,
                        help='Minimum SNR threshold (dB)')
    parser.add_argument('--min-persistence', type=float, default=0.5,
                        help='Minimum persistence fraction (0-1)')
    parser.add_argument('--output', type=str, default='../../data/apres/detected_layers',
                        help='Output file path (without extension)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip visualization')
    
    args = parser.parse_args()
    
    # Load data
    data = load_apres_data(args.data)
    
    # Detect layers
    layers = detect_layers(
        data['range_img'],
        data['Rcoarse'],
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        min_snr_db=args.min_snr,
        min_persistence=args.min_persistence,
    )
    
    # Save results
    save_layers(layers, args.output)
    
    # Visualize
    if not args.no_plot:
        visualize_layers(
            data['range_img'],
            data['Rcoarse'],
            data['time_days'],
            layers,
            output_file=f"{args.output}_visualization.html",
        )
    
    return layers


if __name__ == '__main__':
    main()
