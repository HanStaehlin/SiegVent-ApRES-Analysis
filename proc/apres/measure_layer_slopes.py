#!/usr/bin/env python3
"""
Direct Layer Slope Measurement from Radar Data

This script calculates layer slopes from:
1. Cross-profile radar surveys (RES/GPR)
2. Multiple ApRES sites in a transect
3. Borehole imaging data

These provide INDEPENDENT validation of the inferred slopes from
flow_regime_analysis.py.

Usage:
    python measure_layer_slopes.py --radar-profile path/to/radar.h5
    python measure_layer_slopes.py --apres-sites site1.mat site2.mat site3.mat

Author: SiegVent2023 project
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class LayerSlope:
    """Container for measured layer slope."""
    depth: float              # m
    slope_angle: float        # radians
    slope_deg: float          # degrees
    horizontal_distance: float  # m
    vertical_change: float    # m
    uncertainty: float        # radians


def measure_slope_from_radar_profile(
    horizontal_coords: np.ndarray,
    depths: np.ndarray,
    layer_picks: np.ndarray,
) -> LayerSlope:
    """
    Measure layer slope from radar cross-section.

    Args:
        horizontal_coords: Horizontal distance along profile (m)
        depths: Depth grid (m)
        layer_picks: Picked layer depths at each horizontal position (m)

    Returns:
        LayerSlope object with measurements
    """
    # Remove NaN values
    valid = ~np.isnan(layer_picks)
    x = horizontal_coords[valid]
    z = layer_picks[valid]

    if len(x) < 2:
        return None

    # Fit linear trend
    coeffs = np.polyfit(x, z, deg=1)
    slope_dz_dx = coeffs[0]  # dz/dx

    # Calculate slope angle
    slope_angle = np.arctan(slope_dz_dx)
    slope_deg = np.rad2deg(slope_angle)

    # Calculate statistics
    mean_depth = np.mean(z)
    horizontal_span = x[-1] - x[0]
    vertical_change = z[-1] - z[0]

    # Estimate uncertainty from residuals
    z_fit = np.polyval(coeffs, x)
    residuals = z - z_fit
    uncertainty = np.std(residuals) / horizontal_span

    return LayerSlope(
        depth=mean_depth,
        slope_angle=slope_angle,
        slope_deg=slope_deg,
        horizontal_distance=horizontal_span,
        vertical_change=vertical_change,
        uncertainty=uncertainty,
    )


def measure_slope_from_multiple_apres(
    site_locations: np.ndarray,  # (N, 2) array of (x, y) positions
    layer_depths: np.ndarray,     # (N,) array of depths at each site
) -> LayerSlope:
    """
    Measure layer slope from multiple ApRES sites along a transect.

    This assumes sites are along a flow line and measures the
    depth change of the same layer between sites.

    Args:
        site_locations: (N, 2) array of site coordinates
        layer_depths: Depth of layer at each site

    Returns:
        LayerSlope object
    """
    # Calculate distances along transect
    distances = np.sqrt(
        np.diff(site_locations[:, 0])**2 +
        np.diff(site_locations[:, 1])**2
    )
    cumulative_distance = np.concatenate([[0], np.cumsum(distances)])

    # Fit linear trend
    coeffs = np.polyfit(cumulative_distance, layer_depths, deg=1)
    slope_dz_dx = coeffs[0]

    slope_angle = np.arctan(slope_dz_dx)

    return LayerSlope(
        depth=np.mean(layer_depths),
        slope_angle=slope_angle,
        slope_deg=np.rad2deg(slope_angle),
        horizontal_distance=cumulative_distance[-1],
        vertical_change=layer_depths[-1] - layer_depths[0],
        uncertainty=np.nan,
    )


def compare_inferred_vs_measured_slopes(
    inferred_slopes: np.ndarray,  # From flow_regime_analysis
    measured_slopes: List[LayerSlope],
    depths: np.ndarray,
) -> dict:
    """
    Compare slopes inferred from velocity measurements with
    independently measured slopes.

    Returns:
        Dictionary with comparison statistics
    """
    measured_depths = np.array([ls.depth for ls in measured_slopes])
    measured_angles = np.array([ls.slope_angle for ls in measured_slopes])

    # Interpolate inferred slopes to measured depths
    inferred_at_measured = np.interp(measured_depths, depths, inferred_slopes)

    # Calculate differences
    differences = measured_angles - inferred_at_measured
    differences_deg = np.rad2deg(differences)

    rmse = np.sqrt(np.mean(differences**2))
    mean_bias = np.mean(differences)

    return {
        'rmse_rad': rmse,
        'rmse_deg': np.rad2deg(rmse),
        'mean_bias_rad': mean_bias,
        'mean_bias_deg': np.rad2deg(mean_bias),
        'max_difference_deg': np.max(np.abs(differences_deg)),
        'correlation': np.corrcoef(measured_angles, inferred_at_measured)[0, 1],
    }


def visualize_slope_comparison(
    inferred_slopes_deg: np.ndarray,
    inferred_depths: np.ndarray,
    measured_slopes: List[LayerSlope],
    output_file: Optional[str] = None,
):
    """Create comparison plot of inferred vs measured slopes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Panel 1: Slopes vs depth
    ax1.plot(inferred_slopes_deg, inferred_depths, 'o-',
             label='Inferred (from velocity)', alpha=0.7, markersize=4)

    measured_depths = [ls.depth for ls in measured_slopes]
    measured_deg = [ls.slope_deg for ls in measured_slopes]
    measured_uncertainties = [ls.uncertainty for ls in measured_slopes if not np.isnan(ls.uncertainty)]

    ax1.plot(measured_deg, measured_depths, 's',
             label='Measured (from radar)', markersize=8, color='red')

    ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Layer Slope (degrees)')
    ax1.set_ylabel('Depth (m)')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title('Layer Slopes vs Depth')

    # Panel 2: Scatter comparison
    measured_angles = np.array(measured_deg)
    inferred_at_measured = np.interp(
        measured_depths,
        inferred_depths,
        inferred_slopes_deg
    )

    ax2.scatter(inferred_at_measured, measured_angles, s=100, alpha=0.7)

    # 1:1 line
    lim = max(np.abs(ax2.get_xlim()).max(), np.abs(ax2.get_ylim()).max())
    ax2.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.5, label='1:1 line')

    ax2.set_xlabel('Inferred Slope (degrees)')
    ax2.set_ylabel('Measured Slope (degrees)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_title('Inferred vs Measured Comparison')
    ax2.axis('equal')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {output_file}")

    plt.show()


def synthetic_radar_example():
    """
    Example: Generate synthetic radar data with known slopes.

    This demonstrates how to measure slopes from radar cross-sections.
    """
    print("\n" + "="*70)
    print("SYNTHETIC EXAMPLE: Layer Slope Measurement")
    print("="*70)

    # Create synthetic radar profile
    x = np.linspace(0, 1000, 100)  # 1 km profile

    # Three layers with different slopes
    layer1_depth = 200 + 0.01 * x  # 0.01 slope = 0.57°
    layer2_depth = 500 + 0.02 * x  # 0.02 slope = 1.15°
    layer3_depth = 800 + 0.005 * x  # 0.005 slope = 0.29°

    # Add some noise
    layer1_depth += np.random.normal(0, 0.5, len(x))
    layer2_depth += np.random.normal(0, 0.5, len(x))
    layer3_depth += np.random.normal(0, 0.5, len(x))

    # Measure slopes
    slope1 = measure_slope_from_radar_profile(x, None, layer1_depth)
    slope2 = measure_slope_from_radar_profile(x, None, layer2_depth)
    slope3 = measure_slope_from_radar_profile(x, None, layer3_depth)

    print(f"\nLayer 1 (True slope: 0.57°)")
    print(f"  Measured: {slope1.slope_deg:.2f}° ± {np.rad2deg(slope1.uncertainty):.3f}°")
    print(f"  Vertical change: {slope1.vertical_change:.2f} m over {slope1.horizontal_distance:.0f} m")

    print(f"\nLayer 2 (True slope: 1.15°)")
    print(f"  Measured: {slope2.slope_deg:.2f}° ± {np.rad2deg(slope2.uncertainty):.3f}°")

    print(f"\nLayer 3 (True slope: 0.29°)")
    print(f"  Measured: {slope3.slope_deg:.2f}° ± {np.rad2deg(slope3.uncertainty):.3f}°")

    return [slope1, slope2, slope3]


def main():
    parser = argparse.ArgumentParser(
        description='Measure layer slopes from radar data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--radar-profile', type=str,
                       help='Path to radar profile data (.mat or .h5)')
    parser.add_argument('--apres-sites', nargs='+',
                       help='Paths to multiple ApRES site files')
    parser.add_argument('--inferred-slopes', type=str,
                       help='Path to flow_regime_analysis results for comparison')
    parser.add_argument('--output', type=str,
                       help='Output path for comparison figure')
    parser.add_argument('--demo', action='store_true',
                       help='Run synthetic example')

    args = parser.parse_args()

    if args.demo:
        measured_slopes = synthetic_radar_example()
        return

    if not args.radar_profile and not args.apres_sites:
        print("Error: Must provide either --radar-profile or --apres-sites")
        print("Or use --demo to see synthetic example")
        return

    # TODO: Implement loading of actual radar data
    # This depends on your data format
    print("Implementation for actual data depends on your radar data format.")
    print("See synthetic_radar_example() for usage demonstration.")


if __name__ == '__main__':
    main()
