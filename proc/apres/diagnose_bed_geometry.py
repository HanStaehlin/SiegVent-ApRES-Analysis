#!/usr/bin/env python3
"""
Bed Geometry Diagnostics for ApRES Data

This script analyzes the bed reflection to assess:
1. Is the reflection from nadir or off-axis?
2. What is the likely bed slope?
3. How stable is the bed position over time?
4. Is there evidence of a rough or smooth bed?

Methods:
- Bed reflection amplitude analysis
- Temporal stability of bed position
- Comparison with layer continuity
- Phase coherence at bed

Usage:
    python diagnose_bed_geometry.py --data data/apres/ImageP2_python.mat

Author: SiegVent2023 project
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import find_peaks
from scipy.stats import circmean, circstd
import argparse
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class BedDiagnostics:
    """Container for bed geometry diagnostics."""
    bed_depth_mean: float              # m
    bed_depth_std: float               # m
    bed_amplitude_mean: float          # dB
    bed_amplitude_std: float           # dB
    bed_amplitude_temporal_var: float  # dB
    bed_phase_coherence: float         # 0-1
    likely_nadir: bool
    confidence: float                  # 0-1
    evidence: dict


def load_apres_data(data_path: str) -> dict:
    """Load ApRES data from .mat file."""
    mat = loadmat(data_path)

    return {
        'range_img': mat['RawImage'],           # (range, time)
        'Rcoarse': mat['Rcoarse'].flatten(),    # (range,)
        'time_days': mat['TimeInDays'].flatten(),  # (time,)
        'RawImageComplex': mat.get('RawImageComplex', None),
    }


def identify_bed_reflection(
    range_img: np.ndarray,
    Rcoarse: np.ndarray,
    min_depth: float = 1000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify bed reflection depth and amplitude for each chirp.

    Returns:
        bed_depths: Depth of bed for each chirp (m)
        bed_amplitudes: Amplitude at bed for each chirp (dB)
    """
    n_range, n_time = range_img.shape

    # Convert to dB
    power_db = 10 * np.log10(range_img + 1e-10)

    bed_depths = np.zeros(n_time)
    bed_amplitudes = np.zeros(n_time)

    # Find bed for each chirp
    min_idx = np.searchsorted(Rcoarse, min_depth)

    for i in range(n_time):
        # Find peak in deep region
        profile = power_db[min_idx:, i]
        if len(profile) > 0:
            max_idx = np.argmax(profile)
            bed_depths[i] = Rcoarse[min_idx + max_idx]
            bed_amplitudes[i] = profile[max_idx]
        else:
            bed_depths[i] = np.nan
            bed_amplitudes[i] = np.nan

    return bed_depths, bed_amplitudes


def calculate_bed_phase_coherence(
    complex_data: np.ndarray,
    Rcoarse: np.ndarray,
    bed_depths: np.ndarray,
    window_m: float = 5.0,
) -> float:
    """
    Calculate phase coherence at the bed.

    High coherence (near 1) suggests:
    - Strong, stable reflection
    - Likely from specular surface (smooth bed)
    - Consistent reflection point (nadir)

    Low coherence (near 0) suggests:
    - Weak or scattered reflection
    - Rough bed
    - Possibly off-axis or multiple reflectors
    """
    if complex_data is None:
        return np.nan

    n_time = len(bed_depths)
    phases = np.zeros(n_time)

    for i, bed_depth in enumerate(bed_depths):
        if not np.isnan(bed_depth):
            # Find indices within window of bed
            idx = np.where(np.abs(Rcoarse - bed_depth) < window_m)[0]
            if len(idx) > 0:
                # Get complex values
                complex_vals = complex_data[idx, i]
                # Average phase (circular mean)
                phases[i] = np.angle(np.mean(complex_vals))

    # Calculate phase coherence (mean resultant length)
    coherence = np.abs(np.mean(np.exp(1j * phases)))

    return coherence


def assess_bed_stability(bed_depths: np.ndarray, time_days: np.ndarray) -> dict:
    """
    Analyze temporal stability of bed reflection.

    Stable bed suggests:
    - Nadir reflection (same point each time)
    - Smooth, well-defined interface

    Variable bed suggests:
    - Off-axis reflection (antenna pattern changes)
    - Rough bed with multiple scatterers
    - Instrument motion
    """
    valid = ~np.isnan(bed_depths)

    if np.sum(valid) < 2:
        return {'stable': False, 'std_m': np.nan, 'range_m': np.nan}

    bed_std = np.std(bed_depths[valid])
    bed_range = np.ptp(bed_depths[valid])

    # Linear trend (expect small range change from vertical motion)
    coeffs = np.polyfit(time_days[valid], bed_depths[valid], deg=1)
    trend_m_per_day = coeffs[0]

    # Detrended variability
    detrended = bed_depths[valid] - np.polyval(coeffs, time_days[valid])
    detrended_std = np.std(detrended)

    # Stability criteria
    is_stable = detrended_std < 0.5  # < 0.5 m variability

    return {
        'stable': is_stable,
        'std_m': bed_std,
        'range_m': bed_range,
        'trend_m_per_day': trend_m_per_day,
        'detrended_std_m': detrended_std,
    }


def assess_bed_roughness(
    range_img: np.ndarray,
    Rcoarse: np.ndarray,
    bed_depths: np.ndarray,
    window_m: float = 10.0,
) -> dict:
    """
    Assess bed roughness from reflection characteristics.

    Smooth bed:
    - Sharp, narrow peak
    - High amplitude
    - Consistent across chirps

    Rough bed:
    - Broad, diffuse reflection
    - Lower amplitude
    - Multiple scatterers
    """
    power_db = 10 * np.log10(range_img + 1e-10)

    peak_widths = []
    peak_amplitudes = []

    for i, bed_depth in enumerate(bed_depths):
        if not np.isnan(bed_depth):
            # Extract window around bed
            idx = np.where(np.abs(Rcoarse - bed_depth) < window_m)[0]
            if len(idx) > 10:
                profile = power_db[idx, i]

                # Find peak width at -3dB
                max_power = np.max(profile)
                above_threshold = profile > (max_power - 3)
                width_m = np.sum(above_threshold) * np.mean(np.diff(Rcoarse))

                peak_widths.append(width_m)
                peak_amplitudes.append(max_power)

    if len(peak_widths) == 0:
        return {'smooth': False, 'width_m': np.nan, 'amplitude_db': np.nan}

    mean_width = np.mean(peak_widths)
    mean_amplitude = np.mean(peak_amplitudes)

    # Smooth bed: narrow peak (< 3m), high amplitude (> -70 dB)
    is_smooth = (mean_width < 3.0) and (mean_amplitude > -70)

    return {
        'smooth': is_smooth,
        'width_m': mean_width,
        'amplitude_db': mean_amplitude,
        'width_std_m': np.std(peak_widths),
    }


def diagnose_bed_geometry(
    data_path: str,
    velocity_result: Optional[dict] = None,
    horizontal_velocity: float = 200.0,
) -> BedDiagnostics:
    """
    Comprehensive bed geometry diagnostics.

    Returns assessment of whether bed reflection is from nadir
    and what the bed geometry likely is.
    """
    # Load data
    data = load_apres_data(data_path)

    # Identify bed
    bed_depths, bed_amplitudes = identify_bed_reflection(
        data['range_img'],
        data['Rcoarse'],
    )

    # Analyze bed characteristics
    stability = assess_bed_stability(bed_depths, data['time_days'])
    roughness = assess_bed_roughness(
        data['range_img'],
        data['Rcoarse'],
        bed_depths,
    )

    # Phase coherence (if complex data available)
    if data['RawImageComplex'] is not None:
        phase_coherence = calculate_bed_phase_coherence(
            data['RawImageComplex'],
            data['Rcoarse'],
            bed_depths,
        )
    else:
        phase_coherence = np.nan

    # Compile evidence
    evidence = {
        'stability': stability,
        'roughness': roughness,
        'phase_coherence': phase_coherence,
    }

    # Decision logic for nadir assessment
    likely_nadir = True
    confidence = 1.0

    # Reduce confidence for various issues
    if not stability['stable']:
        likely_nadir = False
        confidence -= 0.3
        evidence['warning_stability'] = f"High variability: {stability['detrended_std_m']:.2f} m"

    if not roughness['smooth']:
        confidence -= 0.2
        evidence['warning_roughness'] = f"Broad reflection: {roughness['width_m']:.1f} m width"

    if not np.isnan(phase_coherence) and phase_coherence < 0.7:
        confidence -= 0.3
        evidence['warning_coherence'] = f"Low phase coherence: {phase_coherence:.2f}"

    # If we have velocity data, check consistency
    if velocity_result is not None:
        required_tilt = velocity_result.get('bed_tilt_angle_deg', np.nan)
        if not np.isnan(required_tilt) and abs(required_tilt) > 2.0:
            likely_nadir = False
            confidence -= 0.4
            evidence['warning_velocity'] = f"Required tilt: {required_tilt:.2f}°"

    confidence = max(0, confidence)

    return BedDiagnostics(
        bed_depth_mean=np.nanmean(bed_depths),
        bed_depth_std=np.nanstd(bed_depths),
        bed_amplitude_mean=np.nanmean(bed_amplitudes),
        bed_amplitude_std=np.nanstd(bed_amplitudes),
        bed_amplitude_temporal_var=np.nanstd(bed_amplitudes),
        bed_phase_coherence=phase_coherence,
        likely_nadir=likely_nadir,
        confidence=confidence,
        evidence=evidence,
    )


def visualize_bed_diagnostics(
    data_path: str,
    diagnostics: BedDiagnostics,
    output_file: Optional[str] = None,
):
    """Create diagnostic plots for bed geometry assessment."""
    # Load data
    data = load_apres_data(data_path)
    bed_depths, bed_amplitudes = identify_bed_reflection(
        data['range_img'],
        data['Rcoarse'],
    )

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Panel 1: Bed depth over time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(data['time_days'], bed_depths, 'o-', markersize=4)
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('Bed Depth (m)')
    ax1.set_title(f'Bed Position Over Time\nStd: {diagnostics.bed_depth_std:.2f} m')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Bed amplitude over time
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(data['time_days'], bed_amplitudes, 'o-', markersize=4, color='green')
    ax2.set_xlabel('Time (days)')
    ax2.set_ylabel('Bed Amplitude (dB)')
    ax2.set_title(f'Bed Reflection Strength\nMean: {diagnostics.bed_amplitude_mean:.1f} dB')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Echogram with bed overlay
    ax3 = fig.add_subplot(gs[0, 2])
    power_db = 10 * np.log10(data['range_img'] + 1e-10)
    extent = [data['time_days'][0], data['time_days'][-1],
              data['Rcoarse'][-1], data['Rcoarse'][0]]
    im = ax3.imshow(power_db, aspect='auto', extent=extent,
                    cmap='gray', vmin=-100, vmax=-40)
    ax3.plot(data['time_days'], bed_depths, 'r-', linewidth=2, label='Bed')
    ax3.set_xlabel('Time (days)')
    ax3.set_ylabel('Depth (m)')
    ax3.set_title('Echogram with Bed Tracking')
    ax3.legend()
    plt.colorbar(im, ax=ax3, label='Power (dB)')

    # Panel 4: Bed depth histogram
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(bed_depths[~np.isnan(bed_depths)], bins=30, edgecolor='black')
    ax4.axvline(diagnostics.bed_depth_mean, color='red', linestyle='--',
                label=f'Mean: {diagnostics.bed_depth_mean:.1f} m')
    ax4.set_xlabel('Bed Depth (m)')
    ax4.set_ylabel('Count')
    ax4.set_title('Bed Depth Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # Panel 5: Roughness indicator
    ax5 = fig.add_subplot(gs[1, 1])
    stability = diagnostics.evidence['stability']
    roughness = diagnostics.evidence['roughness']

    categories = ['Temporal\nStability', 'Bed\nSmoothness', 'Phase\nCoherence']
    scores = [
        1.0 if stability['stable'] else 0.0,
        1.0 if roughness['smooth'] else 0.0,
        diagnostics.bed_phase_coherence if not np.isnan(diagnostics.bed_phase_coherence) else 0.5,
    ]
    colors = ['green' if s > 0.7 else 'orange' if s > 0.3 else 'red' for s in scores]

    bars = ax5.bar(categories, scores, color=colors, edgecolor='black', alpha=0.7)
    ax5.set_ylim(0, 1)
    ax5.set_ylabel('Score')
    ax5.set_title('Bed Reflection Quality Metrics')
    ax5.axhline(0.7, color='green', linestyle='--', alpha=0.3, label='Good')
    ax5.axhline(0.3, color='red', linestyle='--', alpha=0.3, label='Poor')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')

    # Panel 6: Assessment summary
    ax6 = fig.add_subplot(gs[1:, 2])
    ax6.axis('off')

    summary = f"""BED GEOMETRY ASSESSMENT

NADIR REFLECTION:
{'✓ LIKELY' if diagnostics.likely_nadir else '✗ UNLIKELY'}
Confidence: {diagnostics.confidence:.2f}

BED CHARACTERISTICS:
• Mean depth: {diagnostics.bed_depth_mean:.1f} m
• Depth variability: {diagnostics.bed_depth_std:.2f} m
• Detrended std: {stability['detrended_std_m']:.2f} m
• Mean amplitude: {diagnostics.bed_amplitude_mean:.1f} dB
• Peak width: {roughness['width_m']:.1f} m
• Phase coherence: {diagnostics.bed_phase_coherence:.2f}

STABILITY:
{'✓' if stability['stable'] else '✗'} {('Stable' if stability['stable'] else 'Variable')}
  Detrended std: {stability['detrended_std_m']:.2f} m
  (Expect < 0.5 m for nadir)

ROUGHNESS:
{'✓' if roughness['smooth'] else '✗'} {('Smooth bed' if roughness['smooth'] else 'Rough bed')}
  Peak width: {roughness['width_m']:.1f} m
  (Expect < 3 m for smooth)

"""

    # Add warnings
    warnings = [v for k, v in diagnostics.evidence.items() if k.startswith('warning_')]
    if warnings:
        summary += "⚠ WARNINGS:\n"
        for w in warnings:
            summary += f"  • {w}\n"
    else:
        summary += "✓ NO WARNINGS\n"

    summary += f"""
CONCLUSION:
"""

    if diagnostics.likely_nadir and diagnostics.confidence > 0.7:
        summary += "Strong evidence for nadir\nreflection from smooth bed.\nYour professor's 'point'\nassumption is VALID."
    elif diagnostics.likely_nadir:
        summary += "Moderate evidence for nadir.\nAssumption likely okay but\nreport with uncertainty."
    else:
        summary += "Weak evidence for nadir.\nBed may be tilted or rough.\nConsider alternative\ninterpretations."

    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
             verticalalignment='top', fontfamily='monospace',
             fontsize=9)

    plt.suptitle('Bed Geometry Diagnostics', fontsize=14, fontweight='bold')

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {output_file}")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Diagnose bed geometry from ApRES data',
    )

    parser.add_argument('--data', type=str, required=True,
                       help='Path to ApRES data (.mat)')
    parser.add_argument('--velocity-result', type=str,
                       help='Path to flow_regime_analysis.json for comparison')
    parser.add_argument('--horizontal-velocity', type=float, default=200.0,
                       help='Horizontal velocity (m/year)')
    parser.add_argument('--output', type=str,
                       help='Output path for figure')

    args = parser.parse_args()

    # Load velocity result if provided
    velocity_result = None
    if args.velocity_result:
        import json
        with open(args.velocity_result, 'r') as f:
            velocity_result = json.load(f)

    # Run diagnostics
    print("Running bed geometry diagnostics...")
    diagnostics = diagnose_bed_geometry(
        args.data,
        velocity_result=velocity_result,
        horizontal_velocity=args.horizontal_velocity,
    )

    # Print summary
    print("\n" + "="*70)
    print("BED GEOMETRY DIAGNOSTICS")
    print("="*70)
    print(f"\nNadir reflection: {'LIKELY' if diagnostics.likely_nadir else 'UNLIKELY'}")
    print(f"Confidence: {diagnostics.confidence:.2f}")
    print(f"\nBed depth: {diagnostics.bed_depth_mean:.1f} ± {diagnostics.bed_depth_std:.2f} m")
    print(f"Bed amplitude: {diagnostics.bed_amplitude_mean:.1f} ± {diagnostics.bed_amplitude_std:.1f} dB")
    print(f"Phase coherence: {diagnostics.bed_phase_coherence:.2f}")

    stability = diagnostics.evidence['stability']
    print(f"\nStability: {'STABLE' if stability['stable'] else 'VARIABLE'}")
    print(f"  Detrended std: {stability['detrended_std_m']:.2f} m")

    roughness = diagnostics.evidence['roughness']
    print(f"\nRoughness: {'SMOOTH' if roughness['smooth'] else 'ROUGH'}")
    print(f"  Peak width: {roughness['width_m']:.1f} m")

    # Visualize
    output_fig = f"{args.output}.png" if args.output else None
    visualize_bed_diagnostics(args.data, diagnostics, output_file=output_fig)


if __name__ == '__main__':
    main()
