"""
Enhanced Deep Layer Processing Pipeline for ApRES Data

Implements a multi-step signal enhancement pipeline:
1. Coherent stacking - reduce noise while preserving phase
2. SVD denoising - remove incoherent noise components
3. Depth-dependent gain - compensate for absorption
4. Coherence-based quality assessment

Author: SiegVent2023 project
"""

import numpy as np
from scipy.io import loadmat, savemat
from scipy import ndimage, signal
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class EnhancedData:
    """Container for enhanced ApRES data."""
    complex_enhanced: np.ndarray      # Enhanced complex echogram
    amplitude_enhanced: np.ndarray    # Enhanced amplitude
    amplitude_db: np.ndarray          # Enhanced amplitude in dB
    coherence_map: np.ndarray         # Temporal coherence map
    Rcoarse: np.ndarray               # Depth vector
    time_days: np.ndarray             # Time vector (may be reduced from stacking)
    stack_factor: int                 # Number of measurements stacked
    svd_components: int               # Number of SVD components kept
    gain_applied: np.ndarray          # Depth-dependent gain curve


def coherent_stack(complex_data: np.ndarray, 
                   stack_size: int = 20,
                   overlap: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Coherently stack consecutive measurements to improve SNR.
    
    Coherent stacking averages complex values, preserving phase.
    SNR improvement is √N for uncorrelated noise.
    
    Args:
        complex_data: Complex echogram [n_bins, n_times]
        stack_size: Number of measurements to stack
        overlap: Fractional overlap between stacks (0-1)
        
    Returns:
        stacked_data: Stacked complex echogram
        stacked_times: Indices of center times for each stack
    """
    n_bins, n_times = complex_data.shape
    step = int(stack_size * (1 - overlap))
    step = max(1, step)
    
    n_stacks = (n_times - stack_size) // step + 1
    
    stacked = np.zeros((n_bins, n_stacks), dtype=complex)
    center_indices = np.zeros(n_stacks, dtype=int)
    
    for i in range(n_stacks):
        start = i * step
        end = start + stack_size
        stacked[:, i] = np.mean(complex_data[:, start:end], axis=1)
        center_indices[i] = (start + end) // 2
    
    return stacked, center_indices


def svd_denoise(complex_data: np.ndarray,
                n_components: int = 50,
                depth_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Apply SVD-based denoising to remove incoherent noise.
    
    Keeps only the top n_components singular values which capture
    coherent layer structure, removing noise in smaller components.
    
    Args:
        complex_data: Complex echogram [n_bins, n_times]
        n_components: Number of singular values to keep
        depth_mask: Optional mask for depth range to process
        
    Returns:
        denoised: Denoised complex echogram
    """
    if depth_mask is not None:
        # Process only masked region
        data_subset = complex_data[depth_mask, :]
    else:
        data_subset = complex_data
    
    # SVD decomposition
    U, S, Vh = np.linalg.svd(data_subset, full_matrices=False)
    
    # Keep only top components
    n_keep = min(n_components, len(S))
    S_filtered = np.zeros_like(S)
    S_filtered[:n_keep] = S[:n_keep]
    
    # Reconstruct
    denoised_subset = U @ np.diag(S_filtered) @ Vh
    
    if depth_mask is not None:
        denoised = complex_data.copy()
        denoised[depth_mask, :] = denoised_subset
    else:
        denoised = denoised_subset
    
    return denoised


def estimate_absorption(amplitude_db: np.ndarray,
                        Rcoarse: np.ndarray,
                        fit_range: Tuple[float, float] = (200, 800)) -> Tuple[float, float]:
    """
    Estimate absorption coefficient from amplitude decay with depth.
    
    Fits model: A(R) = A0 - α*R (in dB)
    where α is absorption in dB/m
    
    Args:
        amplitude_db: Amplitude in dB [n_bins, n_times] or [n_bins]
        Rcoarse: Depth vector
        fit_range: Depth range to use for fitting
        
    Returns:
        alpha: Absorption coefficient (dB/m)
        A0: Reference amplitude (dB)
    """
    if amplitude_db.ndim == 2:
        profile = np.mean(amplitude_db, axis=1)
    else:
        profile = amplitude_db
    
    mask = (Rcoarse >= fit_range[0]) & (Rcoarse <= fit_range[1])
    depths = Rcoarse[mask]
    amps = profile[mask]
    
    # Linear fit in dB space
    coeffs = np.polyfit(depths, amps, 1)
    alpha = -coeffs[0]  # Positive alpha means decay
    A0 = coeffs[1]
    
    return alpha, A0


def apply_depth_gain(complex_data: np.ndarray,
                     Rcoarse: np.ndarray,
                     alpha: float,
                     reference_depth: float = 100,
                     max_gain_db: float = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply depth-dependent gain to compensate for absorption.
    
    Args:
        complex_data: Complex echogram
        Rcoarse: Depth vector
        alpha: Absorption coefficient (dB/m)
        reference_depth: Depth at which gain = 1 (0 dB)
        max_gain_db: Maximum gain to apply (prevents over-amplification)
        
    Returns:
        gained_data: Gain-corrected complex data
        gain_curve: Applied gain in dB
    """
    # Calculate gain needed to compensate absorption
    gain_db = alpha * (Rcoarse - reference_depth)
    
    # Clip to prevent excessive amplification
    gain_db = np.clip(gain_db, 0, max_gain_db)
    
    # Convert to linear and apply
    gain_linear = 10 ** (gain_db / 20)  # Amplitude gain
    
    # Apply gain (broadcast over time)
    gained_data = complex_data * gain_linear[:, np.newaxis]
    
    return gained_data, gain_db


def compute_coherence_map(complex_data: np.ndarray,
                          window_size: int = 10) -> np.ndarray:
    """
    Compute temporal coherence map.
    
    Coherence measures phase stability between consecutive measurements.
    High coherence indicates real, stable reflectors.
    
    γ = |⟨z1 × z2*⟩| / √(⟨|z1|²⟩⟨|z2|²⟩)
    
    Args:
        complex_data: Complex echogram [n_bins, n_times]
        window_size: Window for coherence estimation
        
    Returns:
        coherence: Coherence map [n_bins, n_times-1]
    """
    n_bins, n_times = complex_data.shape
    
    # Compute coherence between adjacent measurements
    z1 = complex_data[:, :-1]
    z2 = complex_data[:, 1:]
    
    # Use sliding window for smoothing
    def smooth(x, w):
        return ndimage.uniform_filter1d(x, w, axis=1, mode='reflect')
    
    numerator = np.abs(smooth(z1 * np.conj(z2), window_size))
    denominator = np.sqrt(smooth(np.abs(z1)**2, window_size) * 
                          smooth(np.abs(z2)**2, window_size))
    
    coherence = numerator / (denominator + 1e-10)
    
    return coherence


def compute_mean_coherence_profile(coherence_map: np.ndarray) -> np.ndarray:
    """Compute mean coherence vs depth."""
    return np.mean(coherence_map, axis=1)


def enhance_deep_layers(data_path: str,
                        output_path: str,
                        stack_size: int = 20,
                        svd_components: int = 100,
                        deep_threshold: float = 500,
                        max_gain_db: float = 25,
                        verbose: bool = True) -> EnhancedData:
    """
    Main enhancement pipeline for deep layer detection.
    
    Pipeline:
    1. Load complex data
    2. Coherent stacking (SNR improvement)
    3. SVD denoising (for deep region)
    4. Depth-dependent gain correction
    5. Coherence computation
    
    Args:
        data_path: Path to ImageP2_python.mat
        output_path: Path for output files
        stack_size: Number of measurements to stack
        svd_components: Number of SVD components to keep
        deep_threshold: Depth below which to apply SVD denoising
        max_gain_db: Maximum depth gain to apply
        verbose: Print progress
        
    Returns:
        EnhancedData object with processed data
    """
    if verbose:
        print("="*70)
        print("DEEP LAYER ENHANCEMENT PIPELINE")
        print("="*70)
    
    # Load data
    if verbose:
        print("\n[1/5] Loading data...")
    mat_data = loadmat(data_path)
    
    if 'RawImageComplex' in mat_data:
        complex_data = np.array(mat_data['RawImageComplex'])
    else:
        raise ValueError("RawImageComplex not found in data file")
    
    Rcoarse = np.array(mat_data['Rcoarse']).flatten()
    time_days = np.array(mat_data['TimeInDays']).flatten()
    
    n_bins, n_times = complex_data.shape
    if verbose:
        print(f"  Original data: {n_bins} bins × {n_times} times")
    
    # Step 1: Coherent stacking
    if verbose:
        print(f"\n[2/5] Coherent stacking (stack_size={stack_size})...")
    stacked_data, time_indices = coherent_stack(complex_data, stack_size, overlap=0.5)
    stacked_times = time_days[time_indices]
    
    snr_gain = 10 * np.log10(stack_size)
    if verbose:
        print(f"  Stacked data: {stacked_data.shape[0]} bins × {stacked_data.shape[1]} times")
        print(f"  Theoretical SNR gain: {snr_gain:.1f} dB")
    
    # Step 2: SVD denoising (for deep region only)
    if verbose:
        print(f"\n[3/5] SVD denoising (deep region > {deep_threshold}m)...")
    
    deep_mask = Rcoarse >= deep_threshold
    
    # Analyze SVD before denoising
    deep_region = stacked_data[deep_mask, :]
    U, S, Vh = np.linalg.svd(deep_region, full_matrices=False)
    total_energy = np.sum(S**2)
    kept_energy = np.sum(S[:svd_components]**2)
    
    if verbose:
        print(f"  Deep region: {np.sum(deep_mask)} bins")
        print(f"  Keeping {svd_components} components ({kept_energy/total_energy*100:.1f}% energy)")
    
    denoised_data = svd_denoise(stacked_data, n_components=svd_components, depth_mask=deep_mask)
    
    # Step 3: Estimate and apply depth-dependent gain
    if verbose:
        print(f"\n[4/5] Depth-dependent gain correction...")
    
    amp_db = 20 * np.log10(np.abs(stacked_data) + 1e-30)
    alpha, A0 = estimate_absorption(amp_db, Rcoarse, fit_range=(200, 700))
    
    if verbose:
        print(f"  Estimated absorption: {alpha*1000:.2f} dB/km")
    
    gained_data, gain_curve = apply_depth_gain(denoised_data, Rcoarse, alpha, 
                                                reference_depth=100, max_gain_db=max_gain_db)
    
    if verbose:
        print(f"  Max gain applied: {np.max(gain_curve):.1f} dB at {Rcoarse[np.argmax(gain_curve)]:.0f}m")
    
    # Step 4: Compute coherence
    if verbose:
        print(f"\n[5/5] Computing temporal coherence...")
    
    coherence_map = compute_coherence_map(gained_data, window_size=10)
    mean_coherence = compute_mean_coherence_profile(coherence_map)
    
    # Analyze coherence at depth
    shallow_coh = np.mean(mean_coherence[Rcoarse < 400])
    deep_coh = np.mean(mean_coherence[Rcoarse >= 700])
    
    if verbose:
        print(f"  Mean coherence (shallow <400m): {shallow_coh:.3f}")
        print(f"  Mean coherence (deep >700m): {deep_coh:.3f}")
    
    # Compute final amplitude
    amp_enhanced = np.abs(gained_data)
    amp_enhanced_db = 20 * np.log10(amp_enhanced + 1e-30)
    
    # Create result object
    result = EnhancedData(
        complex_enhanced=gained_data,
        amplitude_enhanced=amp_enhanced,
        amplitude_db=amp_enhanced_db,
        coherence_map=coherence_map,
        Rcoarse=Rcoarse,
        time_days=stacked_times,
        stack_factor=stack_size,
        svd_components=svd_components,
        gain_applied=gain_curve,
    )
    
    # Save enhanced data
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    enhanced_file = output_dir / 'ImageP2_enhanced.mat'
    savemat(str(enhanced_file), {
        'RawImageComplex': gained_data,
        'RawImage': amp_enhanced,
        'Rcoarse': Rcoarse,
        'TimeInDays': stacked_times,
        'CoherenceMap': coherence_map,
        'MeanCoherence': mean_coherence,
        'GainCurve_dB': gain_curve,
        'StackFactor': stack_size,
        'SVDComponents': svd_components,
    })
    
    if verbose:
        print(f"\n✓ Enhanced data saved to: {enhanced_file}")
    
    return result


def compare_before_after(original_path: str, 
                         enhanced_data: EnhancedData,
                         output_path: str):
    """
    Create comparison visualization of enhancement.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Load original
    mat_data = loadmat(original_path)
    original_complex = np.array(mat_data['RawImageComplex'])
    Rcoarse = enhanced_data.Rcoarse
    
    # Compute original amplitude
    original_amp = np.abs(original_complex)
    original_db = 20 * np.log10(original_amp + 1e-30)
    
    # Mean profiles
    original_profile = np.mean(original_db, axis=1)
    enhanced_profile = np.mean(enhanced_data.amplitude_db, axis=1)
    coherence_profile = np.mean(enhanced_data.coherence_map, axis=1)
    
    # Focus on deep region
    depth_range = (500, 1100)
    mask = (Rcoarse >= depth_range[0]) & (Rcoarse <= depth_range[1])
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Original Mean Profile',
            'Enhanced Mean Profile', 
            'Amplitude Improvement',
            'Coherence Profile'
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )
    
    # Original profile
    fig.add_trace(
        go.Scatter(x=original_profile[mask], y=Rcoarse[mask],
                   mode='lines', name='Original', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Enhanced profile
    fig.add_trace(
        go.Scatter(x=enhanced_profile[mask], y=Rcoarse[mask],
                   mode='lines', name='Enhanced', line=dict(color='green')),
        row=1, col=2
    )
    
    # Improvement
    improvement = enhanced_profile - original_profile
    fig.add_trace(
        go.Scatter(x=improvement[mask], y=Rcoarse[mask],
                   mode='lines', name='Improvement (dB)', 
                   line=dict(color='red')),
        row=2, col=1
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    # Coherence
    fig.add_trace(
        go.Scatter(x=coherence_profile[mask], y=Rcoarse[mask],
                   mode='lines', name='Coherence', 
                   line=dict(color='purple')),
        row=2, col=2
    )
    
    # Invert y-axes
    for row in [1, 2]:
        for col in [1, 2]:
            fig.update_yaxes(autorange="reversed", row=row, col=col)
    
    fig.update_layout(
        title='Deep Layer Enhancement Comparison (500-1100m)',
        height=800,
        showlegend=False,
    )
    
    fig.update_xaxes(title_text="Amplitude (dB)", row=1, col=1)
    fig.update_xaxes(title_text="Amplitude (dB)", row=1, col=2)
    fig.update_xaxes(title_text="Improvement (dB)", row=2, col=1)
    fig.update_xaxes(title_text="Coherence", row=2, col=2)
    
    for row in [1, 2]:
        fig.update_yaxes(title_text="Depth (m)", row=row, col=1)
    
    output_file = Path(output_path) / 'enhancement_comparison.html'
    fig.write_html(str(output_file))
    print(f"Saved comparison plot: {output_file}")
    
    return fig


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhance deep layers in ApRES data')
    parser.add_argument('--data', type=str, 
                        default='/Users/hannesstahlin/SiegVent2023-Geology/data/apres/ImageP2_python.mat')
    parser.add_argument('--output', type=str,
                        default='/Users/hannesstahlin/SiegVent2023-Geology/output/apres')
    parser.add_argument('--stack-size', type=int, default=20,
                        help='Number of measurements to stack')
    parser.add_argument('--svd-components', type=int, default=100,
                        help='Number of SVD components to keep')
    parser.add_argument('--deep-threshold', type=float, default=500,
                        help='Depth threshold for SVD denoising')
    parser.add_argument('--max-gain', type=float, default=25,
                        help='Maximum depth gain in dB')
    
    args = parser.parse_args()
    
    # Run enhancement pipeline
    result = enhance_deep_layers(
        data_path=args.data,
        output_path=args.output,
        stack_size=args.stack_size,
        svd_components=args.svd_components,
        deep_threshold=args.deep_threshold,
        max_gain_db=args.max_gain,
    )
    
    # Create comparison visualization
    compare_before_after(args.data, result, args.output)
    
    print("\n" + "="*70)
    print("ENHANCEMENT COMPLETE")
    print("="*70)
    print(f"""
Next steps:
1. Run layer detection on enhanced data:
   python run_analysis.py --data {args.output}/ImageP2_enhanced.mat --output {args.output}/enhanced

2. Compare layer counts and quality metrics

3. Visualize in the Dash app:
   python visualization_app.py --data {args.output}/ImageP2_enhanced.mat --output-dir {args.output}/enhanced
""")
