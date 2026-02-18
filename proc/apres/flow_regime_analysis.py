#!/usr/bin/env python3
"""
Flow Regime Analysis for ApRES Data

This script helps interpret ApRES velocity measurements by:
1. Analyzing velocity-depth profiles to infer flow regime
2. Separating geometric effects from true vertical motion
3. Estimating basal sliding vs internal deformation
4. Assessing layer slope effects given known horizontal velocity

Usage:
    python flow_regime_analysis.py --velocity-data path/to/velocity_profile.mat
    python flow_regime_analysis.py --help

Author: SiegVent2023 project
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from scipy.interpolate import UnivariateSpline
from scipy.stats import linregress
import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class FlowRegimeResult:
    """Container for flow regime analysis results."""
    # Input data
    depths: np.ndarray
    velocities: np.ndarray
    reliable: np.ndarray

    # Flow regime metrics
    velocity_gradient: np.ndarray  # dv/dz (1/year)
    shear_strain_rate: np.ndarray  # ε_xz (1/year)

    # Regime classification
    is_plug_flow: bool
    plug_flow_confidence: float
    basal_velocity: float  # m/year (extrapolated to bed)
    surface_velocity: float  # m/year (extrapolated to surface)

    # Geometric correction estimates
    estimated_layer_slope: float  # radians (if horizontal velocity provided)
    geometric_component: np.ndarray  # m/year (u_h * tan(α))
    corrected_vertical_velocity: np.ndarray  # m/year

    # Bed reflection assessment
    bed_depth: float  # m
    bed_velocity: float  # m/year
    bed_tilt_angle: Optional[float]  # radians (estimated if possible)


def load_velocity_profile(velocity_path: str) -> dict:
    """Load velocity profile from .mat file."""
    data = loadmat(f"{velocity_path}.mat")

    return {
        'depths': np.array(data['depths']).flatten(),
        'velocities': np.array(data['velocities']).flatten(),
        'r_squared': np.array(data['r_squared']).flatten(),
        'reliable': np.array(data['reliable']).flatten().astype(bool),
        'amplitude_mean': np.array(data['amplitude_mean']).flatten(),
    }


def assess_plug_flow(depths: np.ndarray, velocities: np.ndarray) -> Tuple[bool, float]:
    """
    Assess if velocity profile indicates plug flow (constant velocity with depth).

    Returns:
        is_plug_flow: True if profile is consistent with plug flow
        confidence: 0-1 confidence score based on velocity variance
    """
    velocity_std = np.std(velocities)
    velocity_mean = np.abs(np.mean(velocities))

    # Calculate coefficient of variation
    if velocity_mean > 0:
        cv = velocity_std / velocity_mean
    else:
        cv = np.inf

    # Check for monotonic trend (should be absent in plug flow)
    if len(velocities) >= 3:
        correlation = np.corrcoef(depths, velocities)[0, 1]
    else:
        correlation = 0

    # Plug flow criteria: low variance and no strong depth correlation
    is_plug = cv < 0.15 and abs(correlation) < 0.3
    confidence = 1.0 - min(cv, 1.0) - 0.5 * abs(correlation)
    confidence = max(0, min(1, confidence))

    return is_plug, confidence


def calculate_velocity_gradient(depths: np.ndarray, velocities: np.ndarray,
                                smooth_sigma: float = 2.0) -> np.ndarray:
    """
    Calculate velocity gradient dv/dz.

    Positive gradient means velocity increases with depth (unusual).
    Negative gradient means velocity decreases with depth (typical for shear).
    """
    # Fit smooth spline for stable derivatives
    if len(depths) < 4:
        return np.zeros_like(depths) * np.nan

    spline = UnivariateSpline(depths, velocities, s=smooth_sigma, k=3)
    gradient = spline.derivative()(depths)

    return gradient


def extrapolate_to_boundaries(depths: np.ndarray, velocities: np.ndarray,
                              bed_depth: float = 1094.0) -> Tuple[float, float]:
    """
    Extrapolate velocity profile to surface (z=0) and bed.

    Returns:
        surface_velocity: Extrapolated velocity at z=0 (m/year)
        basal_velocity: Extrapolated velocity at bed (m/year)
    """
    # Fit polynomial to reliable data
    if len(depths) < 3:
        return np.nan, np.nan

    # Use quadratic fit for reasonable extrapolation
    coeffs = np.polyfit(depths, velocities, deg=min(2, len(depths)-1))
    poly = np.poly1d(coeffs)

    surface_velocity = poly(0)
    basal_velocity = poly(bed_depth)

    return surface_velocity, basal_velocity


def estimate_layer_slopes(velocity_measured: np.ndarray,
                          horizontal_velocity: float = 200.0,
                          accumulation_rate: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate layer slopes needed to explain measured velocities.

    The ApRES measures range rate v_r. If ice moves horizontally at u_h
    past a tilted layer with slope angle α:

        v_r = v_z + u_h * tan(α)

    where v_z is true vertical velocity (accumulation - thinning).

    Args:
        velocity_measured: Measured range velocity (m/year)
        horizontal_velocity: Known horizontal velocity (m/year)
        accumulation_rate: Assumed accumulation rate (m/year)

    Returns:
        layer_slope: Required layer slope in radians
        corrected_vertical: True vertical velocity (m/year)
    """
    # Assume steady state: v_z ≈ accumulation rate at surface
    # (negative because thinning opposes accumulation in range coordinates)
    assumed_vertical = -accumulation_rate  # Negative = moving toward radar

    # Calculate required slope to explain measurement
    # v_r = v_z + u_h * tan(α)
    # tan(α) = (v_r - v_z) / u_h
    geometric_component = velocity_measured - assumed_vertical
    layer_slope = np.arctan(geometric_component / horizontal_velocity)

    # Corrected vertical velocity
    corrected_vertical = velocity_measured - horizontal_velocity * np.tan(layer_slope)

    return layer_slope, corrected_vertical


def assess_bed_reflection_geometry(bed_velocity: float,
                                   horizontal_velocity: float = 200.0,
                                   expected_vertical: float = -0.3) -> dict:
    """
    Assess if bed reflection is from nadir or off-angle.

    If bed is flat and reflection is from nadir:
        v_bed ≈ expected vertical velocity

    If bed is tilted or reflection is off-angle:
        v_bed = v_z + u_h * tan(α_bed)

    Args:
        bed_velocity: Measured bed range velocity (m/year)
        horizontal_velocity: Known horizontal velocity (m/year)
        expected_vertical: Expected vertical velocity at bed (m/year)

    Returns:
        Dictionary with assessment results
    """
    # Calculate required bed tilt
    geometric_component = bed_velocity - expected_vertical
    bed_tilt = np.arctan(geometric_component / horizontal_velocity)
    bed_tilt_deg = np.rad2deg(bed_tilt)

    # Assess likelihood of nadir reflection
    # Typical bed roughness might create 0-5° off-nadir reflections
    is_likely_nadir = abs(bed_tilt_deg) < 2.0
    is_possible_nadir = abs(bed_tilt_deg) < 5.0

    return {
        'bed_tilt_rad': bed_tilt,
        'bed_tilt_deg': bed_tilt_deg,
        'geometric_component': geometric_component,
        'is_likely_nadir': is_likely_nadir,
        'is_possible_nadir': is_possible_nadir,
        'interpretation': (
            "Likely nadir reflection" if is_likely_nadir else
            "Possible off-nadir reflection" if is_possible_nadir else
            "Likely off-nadir reflection - bed may be tilted"
        )
    }


def analyze_flow_regime(velocity_data: dict,
                       horizontal_velocity: float = 200.0,
                       accumulation_rate: float = 0.3,
                       bed_depth: float = 1094.0) -> FlowRegimeResult:
    """
    Comprehensive flow regime analysis.

    Args:
        velocity_data: Dict with 'depths', 'velocities', 'reliable' keys
        horizontal_velocity: Known horizontal surface velocity (m/year)
        accumulation_rate: Surface accumulation rate (m/year ice equiv)
        bed_depth: Ice thickness (m)

    Returns:
        FlowRegimeResult object with all analysis results
    """
    depths = velocity_data['depths']
    velocities = velocity_data['velocities']
    reliable = velocity_data['reliable']

    # Use only reliable measurements
    d_reliable = depths[reliable]
    v_reliable = velocities[reliable]

    # 1. Assess flow regime
    is_plug, plug_confidence = assess_plug_flow(d_reliable, v_reliable)

    # 2. Calculate velocity gradients
    gradient = calculate_velocity_gradient(d_reliable, v_reliable)
    shear_strain_rate = -gradient  # Negative because depth increases downward

    # 3. Extrapolate to boundaries
    surface_vel, basal_vel = extrapolate_to_boundaries(d_reliable, v_reliable, bed_depth)

    # 4. Estimate geometric corrections
    layer_slopes, corrected_velocities = estimate_layer_slopes(
        v_reliable, horizontal_velocity, accumulation_rate
    )
    geometric_comp = horizontal_velocity * np.tan(layer_slopes)

    # 5. Assess bed reflection
    if len(v_reliable) > 0 and not np.isnan(basal_vel):
        bed_assessment = assess_bed_reflection_geometry(
            basal_vel, horizontal_velocity, -accumulation_rate
        )
        bed_tilt = bed_assessment['bed_tilt_rad']
    else:
        bed_tilt = None

    # Expand arrays back to full size
    gradient_full = np.full_like(velocities, np.nan)
    geometric_full = np.full_like(velocities, np.nan)
    corrected_full = np.full_like(velocities, np.nan)
    shear_full = np.full_like(velocities, np.nan)

    gradient_full[reliable] = gradient
    geometric_full[reliable] = geometric_comp
    corrected_full[reliable] = corrected_velocities
    shear_full[reliable] = shear_strain_rate

    result = FlowRegimeResult(
        depths=depths,
        velocities=velocities,
        reliable=reliable,
        velocity_gradient=gradient_full,
        shear_strain_rate=shear_full,
        is_plug_flow=is_plug,
        plug_flow_confidence=plug_confidence,
        basal_velocity=basal_vel,
        surface_velocity=surface_vel,
        estimated_layer_slope=np.nanmean(layer_slopes),
        geometric_component=geometric_full,
        corrected_vertical_velocity=corrected_full,
        bed_depth=bed_depth,
        bed_velocity=basal_vel,
        bed_tilt_angle=bed_tilt,
    )

    return result


def visualize_flow_regime(result: FlowRegimeResult, output_file: Optional[str] = None):
    """Create comprehensive visualization of flow regime analysis."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    reliable = result.reliable
    d_rel = result.depths[reliable]

    # 1. Velocity profile
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(result.velocities[reliable], d_rel, 'o-', label='Measured', markersize=6)
    ax1.axhline(result.bed_depth, color='brown', linestyle='--', alpha=0.5, label='Bed')
    ax1.axhline(0, color='lightblue', linestyle='--', alpha=0.5, label='Surface')
    if not np.isnan(result.surface_velocity):
        ax1.axvline(result.surface_velocity, color='lightblue', linestyle=':', alpha=0.5)
    if not np.isnan(result.basal_velocity):
        ax1.axvline(result.basal_velocity, color='brown', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Range Velocity (m/year)')
    ax1.set_ylabel('Depth (m)')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title('Measured Velocity Profile')

    # 2. Velocity gradient (shear)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(result.shear_strain_rate[reliable] * 1000, d_rel, 'o-', color='crimson', markersize=6)
    ax2.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Shear Strain Rate (×10⁻³ /year)')
    ax2.set_ylabel('Depth (m)')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Velocity Gradient (dv/dz)')

    # Add interpretation text
    if result.is_plug_flow:
        regime_text = f"PLUG FLOW\n(confidence: {result.plug_flow_confidence:.2f})"
        regime_color = 'green'
    else:
        regime_text = f"SHEAR FLOW\n(confidence: {1-result.plug_flow_confidence:.2f})"
        regime_color = 'orange'

    ax2.text(0.05, 0.95, regime_text, transform=ax2.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor=regime_color, alpha=0.3),
             fontsize=10, fontweight='bold')

    # 3. Geometric correction
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(result.velocities[reliable], d_rel, 'o-', label='Measured', markersize=4, alpha=0.5)
    ax3.plot(result.corrected_vertical_velocity[reliable], d_rel, 'o-',
             label='Corrected (v_z)', markersize=6, color='green')
    ax3.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Velocity (m/year)')
    ax3.set_ylabel('Depth (m)')
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_title('Geometric Correction\n(Removing horizontal flow effect)')

    # 4. Layer slopes
    ax4 = fig.add_subplot(gs[1, 0])
    layer_slopes_deg = np.rad2deg(np.arctan(result.geometric_component[reliable] / 200.0))
    ax4.plot(layer_slopes_deg, d_rel, 'o-', color='purple', markersize=6)
    ax4.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Estimated Layer Slope (degrees)')
    ax4.set_ylabel('Depth (m)')
    ax4.invert_yaxis()
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Required Layer Tilt\n(to explain measurements)')

    # Add typical slope range
    ax4.axvspan(-2, 2, color='green', alpha=0.1, label='Typical range')
    ax4.legend()

    # 5. Geometric component breakdown
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(result.velocities[reliable], d_rel, 'o-', label='Total measured', markersize=4)
    ax5.plot(result.geometric_component[reliable], d_rel, 'o-',
             label='Geometric (u·tan α)', markersize=4)
    ax5.plot(result.corrected_vertical_velocity[reliable], d_rel, 'o-',
             label='True vertical', markersize=4)
    ax5.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Velocity Components (m/year)')
    ax5.set_ylabel('Depth (m)')
    ax5.invert_yaxis()
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    ax5.set_title('Velocity Decomposition')

    # 6. Summary text
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    summary = f"""FLOW REGIME ANALYSIS

Flow Type: {'Plug Flow' if result.is_plug_flow else 'Shear Flow'}
Confidence: {result.plug_flow_confidence:.2f}

VELOCITY ESTIMATES:
Surface: {result.surface_velocity:.2f} m/year
Bed: {result.basal_velocity:.2f} m/year

GEOMETRIC CORRECTION:
Mean layer slope: {np.rad2deg(result.estimated_layer_slope):.2f}°
Geometric component: {np.nanmean(result.geometric_component[reliable]):.2f} m/year

BED REFLECTION:
"""

    if result.bed_tilt_angle is not None:
        bed_tilt_deg = np.rad2deg(result.bed_tilt_angle)
        summary += f"Estimated tilt: {bed_tilt_deg:.2f}°\n"
        if abs(bed_tilt_deg) < 2:
            summary += "→ Likely nadir reflection\n"
        elif abs(bed_tilt_deg) < 5:
            summary += "→ Possible off-nadir reflection\n"
        else:
            summary += "→ Likely off-nadir reflection\n"
    else:
        summary += "Insufficient data\n"

    summary += f"""
INTERPRETATION:
"""

    if result.is_plug_flow:
        summary += """• Uniform velocity suggests plug flow
• Ice moves as coherent unit
• Minimal internal deformation
• Suggests sliding bed (not frozen)
"""
    else:
        summary += """• Velocity gradient indicates shear
• Internal deformation present
• Mix of sliding + deformation
• Analyze gradient for basal condition
"""

    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
             verticalalignment='top', fontfamily='monospace', fontsize=9)

    plt.suptitle('ApRES Flow Regime Analysis', fontsize=14, fontweight='bold')

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {output_file}")

    plt.show()

    return fig


def print_summary(result: FlowRegimeResult):
    """Print text summary of flow regime analysis."""
    print("\n" + "="*70)
    print("FLOW REGIME ANALYSIS SUMMARY")
    print("="*70)

    print(f"\n{'FLOW REGIME':.<40} {('Plug Flow' if result.is_plug_flow else 'Shear Flow')}")
    print(f"{'Confidence':.<40} {result.plug_flow_confidence:.3f}")

    print(f"\n{'VELOCITY EXTRAPOLATIONS':-^70}")
    print(f"{'Surface velocity':.<40} {result.surface_velocity:>8.2f} m/year")
    print(f"{'Basal velocity':.<40} {result.basal_velocity:>8.2f} m/year")
    print(f"{'Velocity change (surface to bed)':.<40} {result.surface_velocity - result.basal_velocity:>8.2f} m/year")

    reliable = result.reliable
    if np.any(reliable):
        print(f"\n{'SHEAR ANALYSIS':-^70}")
        mean_shear = np.nanmean(result.shear_strain_rate[reliable])
        max_shear = np.nanmax(result.shear_strain_rate[reliable])
        print(f"{'Mean shear strain rate':.<40} {mean_shear*1000:>8.3f} ×10⁻³ /year")
        print(f"{'Max shear strain rate':.<40} {max_shear*1000:>8.3f} ×10⁻³ /year")

        print(f"\n{'GEOMETRIC CORRECTION':-^70}")
        mean_slope = np.rad2deg(result.estimated_layer_slope)
        mean_geometric = np.nanmean(result.geometric_component[reliable])
        mean_corrected = np.nanmean(result.corrected_vertical_velocity[reliable])
        print(f"{'Mean estimated layer slope':.<40} {mean_slope:>8.2f} degrees")
        print(f"{'Mean geometric component':.<40} {mean_geometric:>8.2f} m/year")
        print(f"{'Mean corrected vertical velocity':.<40} {mean_corrected:>8.2f} m/year")

    if result.bed_tilt_angle is not None:
        print(f"\n{'BED REFLECTION ASSESSMENT':-^70}")
        bed_tilt_deg = np.rad2deg(result.bed_tilt_angle)
        print(f"{'Estimated bed tilt angle':.<40} {bed_tilt_deg:>8.2f} degrees")
        if abs(bed_tilt_deg) < 2:
            assessment = "Likely nadir reflection"
        elif abs(bed_tilt_deg) < 5:
            assessment = "Possible off-nadir reflection"
        else:
            assessment = "Likely off-nadir reflection"
        print(f"{'Assessment':.<40} {assessment}")

    print(f"\n{'INTERPRETATION':-^70}")
    if result.is_plug_flow:
        print("""
→ Velocity profile suggests PLUG FLOW
  • Uniform velocity with depth indicates ice moves as coherent unit
  • Minimal internal deformation
  • Suggests sliding bed (not frozen to bedrock)
  • All motion is basal sliding
        """)
    else:
        print("""
→ Velocity profile suggests SHEAR FLOW
  • Velocity gradient indicates internal deformation
  • Mix of basal sliding and internal shear
  • Examine velocity gradient to determine:
    - Depth of shear concentration
    - Potential sliding contribution (basal velocity > 0)
    - Deformation rate profile
        """)

    print("="*70 + "\n")


def save_results(result: FlowRegimeResult, output_path: str):
    """Save flow regime analysis results to .mat and .json files."""
    # Save to .mat
    savemat(f"{output_path}.mat", {
        'depths': result.depths,
        'velocities': result.velocities,
        'reliable': result.reliable,
        'velocity_gradient': result.velocity_gradient,
        'shear_strain_rate': result.shear_strain_rate,
        'is_plug_flow': result.is_plug_flow,
        'plug_flow_confidence': result.plug_flow_confidence,
        'basal_velocity': result.basal_velocity,
        'surface_velocity': result.surface_velocity,
        'estimated_layer_slope': result.estimated_layer_slope,
        'geometric_component': result.geometric_component,
        'corrected_vertical_velocity': result.corrected_vertical_velocity,
        'bed_depth': result.bed_depth,
        'bed_velocity': result.bed_velocity,
        'bed_tilt_angle': result.bed_tilt_angle if result.bed_tilt_angle is not None else np.nan,
    })

    # Save summary to JSON
    summary = {
        'flow_regime': 'plug_flow' if result.is_plug_flow else 'shear_flow',
        'plug_flow_confidence': float(result.plug_flow_confidence),
        'surface_velocity_m_per_year': float(result.surface_velocity),
        'basal_velocity_m_per_year': float(result.basal_velocity),
        'mean_layer_slope_deg': float(np.rad2deg(result.estimated_layer_slope)),
        'bed_tilt_angle_deg': float(np.rad2deg(result.bed_tilt_angle)) if result.bed_tilt_angle is not None else None,
    }

    with open(f"{output_path}.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to {output_path}.mat and {output_path}.json")


def main():
    parser = argparse.ArgumentParser(
        description='Flow Regime Analysis for ApRES Data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python flow_regime_analysis.py --velocity-data data/apres/layer_analysis/velocity_profile

  # With custom parameters
  python flow_regime_analysis.py \\
      --velocity-data data/apres/layer_analysis/velocity_profile \\
      --horizontal-velocity 185 \\
      --accumulation-rate 0.25 \\
      --bed-depth 1094

  # Save results
  python flow_regime_analysis.py \\
      --velocity-data data/apres/layer_analysis/velocity_profile \\
      --output data/apres/layer_analysis/flow_regime_analysis \\
      --save-figure
        """
    )

    parser.add_argument('--velocity-data', type=str, required=True,
                       help='Path to velocity_profile.mat (without extension)')
    parser.add_argument('--horizontal-velocity', type=float, default=200.0,
                       help='Horizontal surface velocity (m/year, default: 200)')
    parser.add_argument('--accumulation-rate', type=float, default=0.3,
                       help='Surface accumulation rate (m/year ice equiv, default: 0.3)')
    parser.add_argument('--bed-depth', type=float, default=1094.0,
                       help='Ice thickness (m, default: 1094)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for results (without extension)')
    parser.add_argument('--save-figure', action='store_true',
                       help='Save figure to file')

    args = parser.parse_args()

    # Load data
    print(f"Loading velocity data from {args.velocity_data}.mat...")
    velocity_data = load_velocity_profile(args.velocity_data)

    # Run analysis
    print("\nRunning flow regime analysis...")
    result = analyze_flow_regime(
        velocity_data,
        horizontal_velocity=args.horizontal_velocity,
        accumulation_rate=args.accumulation_rate,
        bed_depth=args.bed_depth,
    )

    # Print summary
    print_summary(result)

    # Save results
    if args.output:
        save_results(result, args.output)

    # Visualize
    figure_path = None
    if args.save_figure:
        if args.output:
            figure_path = f"{args.output}.png"
        else:
            figure_path = "flow_regime_analysis.png"

    visualize_flow_regime(result, output_file=figure_path)


if __name__ == '__main__':
    main()
