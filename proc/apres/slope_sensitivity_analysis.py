#!/usr/bin/env python3
"""
Sensitivity Analysis for Layer Slope Estimates

Tests how inferred layer slopes change with different assumptions about:
- Accumulation rate
- Horizontal velocity
- Bed depth

This helps quantify uncertainty in slope estimates and identify
which assumptions matter most.

Usage:
    python slope_sensitivity_analysis.py \
        --velocity-data data/apres/layer_analysis/velocity_profile

Author: SiegVent2023 project
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import argparse
import json
from pathlib import Path

# Import from flow_regime_analysis
import sys
sys.path.insert(0, str(Path(__file__).parent))
from flow_regime_analysis import (
    load_velocity_profile,
    analyze_flow_regime,
    FlowRegimeResult
)


def sensitivity_to_accumulation(
    velocity_data: dict,
    accumulation_range: np.ndarray,
    horizontal_velocity: float = 200.0,
    bed_depth: float = 1094.0,
) -> dict:
    """
    Test sensitivity to accumulation rate assumption.

    Args:
        velocity_data: Velocity profile data
        accumulation_range: Array of accumulation rates to test (m/year)
        horizontal_velocity: Horizontal velocity (m/year)
        bed_depth: Ice thickness (m)

    Returns:
        Dictionary with results for each accumulation rate
    """
    results = {
        'accumulation_rates': accumulation_range,
        'mean_slopes': [],
        'std_slopes': [],
        'mean_corrected_velocities': [],
        'basal_velocities': [],
        'flow_regimes': [],
    }

    for acc in accumulation_range:
        result = analyze_flow_regime(
            velocity_data,
            horizontal_velocity=horizontal_velocity,
            accumulation_rate=acc,
            bed_depth=bed_depth,
        )

        reliable = result.reliable
        if np.any(reliable):
            slopes_deg = np.rad2deg(
                np.arctan(result.geometric_component[reliable] / horizontal_velocity)
            )
            results['mean_slopes'].append(np.nanmean(slopes_deg))
            results['std_slopes'].append(np.nanstd(slopes_deg))
            results['mean_corrected_velocities'].append(
                np.nanmean(result.corrected_vertical_velocity[reliable])
            )
        else:
            results['mean_slopes'].append(np.nan)
            results['std_slopes'].append(np.nan)
            results['mean_corrected_velocities'].append(np.nan)

        results['basal_velocities'].append(result.basal_velocity)
        results['flow_regimes'].append('plug' if result.is_plug_flow else 'shear')

    return results


def sensitivity_to_horizontal_velocity(
    velocity_data: dict,
    u_h_range: np.ndarray,
    accumulation_rate: float = 0.3,
    bed_depth: float = 1094.0,
) -> dict:
    """Test sensitivity to horizontal velocity."""
    results = {
        'horizontal_velocities': u_h_range,
        'mean_slopes': [],
        'std_slopes': [],
        'mean_corrected_velocities': [],
    }

    for u_h in u_h_range:
        result = analyze_flow_regime(
            velocity_data,
            horizontal_velocity=u_h,
            accumulation_rate=accumulation_rate,
            bed_depth=bed_depth,
        )

        reliable = result.reliable
        if np.any(reliable):
            slopes_deg = np.rad2deg(
                np.arctan(result.geometric_component[reliable] / u_h)
            )
            results['mean_slopes'].append(np.nanmean(slopes_deg))
            results['std_slopes'].append(np.nanstd(slopes_deg))
            results['mean_corrected_velocities'].append(
                np.nanmean(result.corrected_vertical_velocity[reliable])
            )

    return results


def visualize_sensitivity(
    acc_results: dict,
    u_h_results: dict,
    nominal_acc: float = 0.3,
    nominal_u_h: float = 200.0,
    output_file: str = None,
):
    """Create comprehensive sensitivity visualization."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Row 1: Sensitivity to accumulation rate
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(acc_results['accumulation_rates'],
             acc_results['mean_slopes'],
             'o-', linewidth=2, markersize=6)
    ax1.axvline(nominal_acc, color='red', linestyle='--',
                alpha=0.5, label='Nominal')
    ax1.set_xlabel('Accumulation Rate (m/year)')
    ax1.set_ylabel('Mean Layer Slope (°)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title('Sensitivity to Accumulation')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(acc_results['accumulation_rates'],
             acc_results['mean_corrected_velocities'],
             'o-', linewidth=2, markersize=6, color='green')
    ax2.axvline(nominal_acc, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Accumulation Rate (m/year)')
    ax2.set_ylabel('Corrected Vertical Velocity (m/year)')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Effect on Corrected Velocity')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(acc_results['accumulation_rates'],
             acc_results['basal_velocities'],
             'o-', linewidth=2, markersize=6, color='brown')
    ax3.axvline(nominal_acc, color='red', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Accumulation Rate (m/year)')
    ax3.set_ylabel('Basal Velocity (m/year)')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Effect on Basal Velocity')

    # Row 2: Sensitivity to horizontal velocity
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(u_h_results['horizontal_velocities'],
             u_h_results['mean_slopes'],
             'o-', linewidth=2, markersize=6, color='purple')
    ax4.axvline(nominal_u_h, color='red', linestyle='--',
                alpha=0.5, label='Nominal')
    ax4.set_xlabel('Horizontal Velocity (m/year)')
    ax4.set_ylabel('Mean Layer Slope (°)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_title('Sensitivity to Horizontal Velocity')

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(u_h_results['horizontal_velocities'],
             u_h_results['mean_corrected_velocities'],
             'o-', linewidth=2, markersize=6, color='green')
    ax5.axvline(nominal_u_h, color='red', linestyle='--', alpha=0.5)
    ax5.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax5.set_xlabel('Horizontal Velocity (m/year)')
    ax5.set_ylabel('Corrected Vertical Velocity (m/year)')
    ax5.grid(True, alpha=0.3)
    ax5.set_title('Effect on Corrected Velocity')

    # Row 3: Uncertainty quantification
    ax6 = fig.add_subplot(gs[2, :2])

    # Calculate ranges
    acc_slope_range = np.ptp(acc_results['mean_slopes'])
    u_h_slope_range = np.ptp(u_h_results['mean_slopes'])
    total_range = np.sqrt(acc_slope_range**2 + u_h_slope_range**2)

    # Bar plot
    sensitivities = ['Accumulation\n(±20%)', 'Horizontal Vel.\n(±10%)', 'Combined\n(RSS)']
    ranges = [acc_slope_range, u_h_slope_range, total_range]
    colors = ['skyblue', 'lightcoral', 'gold']

    bars = ax6.bar(sensitivities, ranges, color=colors, alpha=0.7, edgecolor='black')
    ax6.set_ylabel('Slope Uncertainty (°)')
    ax6.set_title('Sensitivity Summary: Layer Slope Uncertainty')
    ax6.grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for bar, val in zip(bars, ranges):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}°', ha='center', va='bottom', fontweight='bold')

    # Summary text
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')

    nominal_slope = acc_results['mean_slopes'][
        np.argmin(np.abs(acc_results['accumulation_rates'] - nominal_acc))
    ]

    summary_text = f"""SENSITIVITY SUMMARY

Nominal Values:
• Accumulation: {nominal_acc:.2f} m/year
• Horizontal vel: {nominal_u_h:.0f} m/year
• Layer slope: {nominal_slope:.3f}°

Uncertainties:
• Acc ±20%: ±{acc_slope_range/2:.3f}°
• u_h ±10%: ±{u_h_slope_range/2:.3f}°
• Combined: ±{total_range/2:.3f}°

Relative uncertainty:
{100*total_range/(2*abs(nominal_slope)):.1f}% of nominal slope

Interpretation:
"""

    if total_range < 0.1:
        summary_text += "✓ Low sensitivity\n  Results are robust"
    elif total_range < 0.5:
        summary_text += "⚠ Moderate sensitivity\n  Report with uncertainty"
    else:
        summary_text += "❌ High sensitivity\n  Needs validation"

    ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes,
             verticalalignment='top', fontfamily='monospace',
             fontsize=9)

    plt.suptitle('Layer Slope Sensitivity Analysis', fontsize=14, fontweight='bold')

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to {output_file}")

    return fig


def print_summary(acc_results: dict, u_h_results: dict,
                 nominal_acc: float, nominal_u_h: float):
    """Print text summary of sensitivity analysis."""
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("="*70)

    # Find nominal values
    idx_nominal_acc = np.argmin(np.abs(acc_results['accumulation_rates'] - nominal_acc))
    nominal_slope = acc_results['mean_slopes'][idx_nominal_acc]

    print(f"\n{'NOMINAL PARAMETERS':-^70}")
    print(f"Accumulation rate:              {nominal_acc:.2f} m/year")
    print(f"Horizontal velocity:            {nominal_u_h:.0f} m/year")
    print(f"Mean layer slope:               {nominal_slope:.3f}°")

    print(f"\n{'SENSITIVITY TO ACCUMULATION RATE':-^70}")
    acc_min = np.min(acc_results['mean_slopes'])
    acc_max = np.max(acc_results['mean_slopes'])
    acc_range = acc_max - acc_min
    print(f"Tested range:                   {np.min(acc_results['accumulation_rates']):.2f} - {np.max(acc_results['accumulation_rates']):.2f} m/year")
    print(f"Slope range:                    {acc_min:.3f}° to {acc_max:.3f}°")
    print(f"Total variation:                ±{acc_range/2:.3f}°")
    print(f"Relative to nominal:            ±{100*acc_range/(2*abs(nominal_slope)):.1f}%")

    print(f"\n{'SENSITIVITY TO HORIZONTAL VELOCITY':-^70}")
    u_h_min = np.min(u_h_results['mean_slopes'])
    u_h_max = np.max(u_h_results['mean_slopes'])
    u_h_range = u_h_max - u_h_min
    print(f"Tested range:                   {np.min(u_h_results['horizontal_velocities']):.0f} - {np.max(u_h_results['horizontal_velocities']):.0f} m/year")
    print(f"Slope range:                    {u_h_min:.3f}° to {u_h_max:.3f}°")
    print(f"Total variation:                ±{u_h_range/2:.3f}°")
    print(f"Relative to nominal:            ±{100*u_h_range/(2*abs(nominal_slope)):.1f}%")

    print(f"\n{'COMBINED UNCERTAINTY':-^70}")
    total_uncertainty = np.sqrt((acc_range/2)**2 + (u_h_range/2)**2)
    print(f"RSS uncertainty:                ±{total_uncertainty:.3f}°")
    print(f"Relative to nominal:            ±{100*total_uncertainty/abs(nominal_slope):.1f}%")

    print(f"\n{'RECOMMENDATION':-^70}")
    if total_uncertainty < 0.1:
        print("✓ Results are ROBUST to parameter uncertainty")
        print("  Slopes can be reported with high confidence")
    elif total_uncertainty < 0.5:
        print("⚠ Results have MODERATE sensitivity")
        print("  Report slopes with uncertainty range")
    else:
        print("❌ Results are HIGHLY SENSITIVE")
        print("  Validation with independent data strongly recommended")

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Sensitivity analysis for layer slope estimates',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python slope_sensitivity_analysis.py \\
      --velocity-data data/apres/layer_analysis/velocity_profile \\
      --acc-range 0.2 0.4 \\
      --u-h-range 180 220 \\
      --output output/apres/sensitivity_analysis
        """
    )

    parser.add_argument('--velocity-data', type=str, required=True,
                       help='Path to velocity_profile.mat (without extension)')
    parser.add_argument('--acc-range', nargs=2, type=float,
                       default=[0.2, 0.4],
                       help='Accumulation range to test (min max, m/year)')
    parser.add_argument('--u-h-range', nargs=2, type=float,
                       default=[180, 220],
                       help='Horizontal velocity range to test (min max, m/year)')
    parser.add_argument('--nominal-acc', type=float, default=0.3,
                       help='Nominal accumulation rate (m/year)')
    parser.add_argument('--nominal-u-h', type=float, default=200.0,
                       help='Nominal horizontal velocity (m/year)')
    parser.add_argument('--bed-depth', type=float, default=1094.0,
                       help='Ice thickness (m)')
    parser.add_argument('--n-points', type=int, default=9,
                       help='Number of test points')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for figure (without extension)')

    args = parser.parse_args()

    # Load data
    print(f"Loading velocity data from {args.velocity_data}.mat...")
    velocity_data = load_velocity_profile(args.velocity_data)

    # Test accumulation sensitivity
    print("\nTesting sensitivity to accumulation rate...")
    acc_range = np.linspace(args.acc_range[0], args.acc_range[1], args.n_points)
    acc_results = sensitivity_to_accumulation(
        velocity_data,
        acc_range,
        horizontal_velocity=args.nominal_u_h,
        bed_depth=args.bed_depth,
    )

    # Test horizontal velocity sensitivity
    print("Testing sensitivity to horizontal velocity...")
    u_h_range = np.linspace(args.u_h_range[0], args.u_h_range[1], args.n_points)
    u_h_results = sensitivity_to_horizontal_velocity(
        velocity_data,
        u_h_range,
        accumulation_rate=args.nominal_acc,
        bed_depth=args.bed_depth,
    )

    # Print summary
    print_summary(acc_results, u_h_results, args.nominal_acc, args.nominal_u_h)

    # Save results
    if args.output:
        results = {
            'accumulation_sensitivity': acc_results,
            'horizontal_velocity_sensitivity': u_h_results,
            'nominal_accumulation': args.nominal_acc,
            'nominal_horizontal_velocity': args.nominal_u_h,
        }

        with open(f"{args.output}.json", 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {
                'accumulation_sensitivity': {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in acc_results.items()
                },
                'horizontal_velocity_sensitivity': {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in u_h_results.items() if k != 'flow_regimes'
                },
                'nominal_accumulation': args.nominal_acc,
                'nominal_horizontal_velocity': args.nominal_u_h,
            }
            json.dump(json_results, f, indent=2)
        print(f"\nResults saved to {args.output}.json")

    # Visualize
    output_fig = f"{args.output}.png" if args.output else None
    visualize_sensitivity(
        acc_results,
        u_h_results,
        args.nominal_acc,
        args.nominal_u_h,
        output_file=output_fig,
    )


if __name__ == '__main__':
    main()
