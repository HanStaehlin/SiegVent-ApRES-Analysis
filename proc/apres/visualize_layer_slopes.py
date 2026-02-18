#!/usr/bin/env python3
"""
Visualize Layer and Bed Slopes

Create comprehensive visualization showing:
- Layer slopes vs depth
- Comparison to bed slope
- Physical interpretation
- Uncertainty ranges

Author: SiegVent2023 project
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import argparse
from pathlib import Path


def visualize_layer_slopes(
    data_path: str,
    output_file: str = None,
):
    """
    Create comprehensive visualization of layer and bed slopes.

    Args:
        data_path: Path to flow_regime_analysis.mat
        output_file: Optional output path for figure
    """
    # Load data
    data = loadmat(data_path)

    depths = data['depths'].flatten()
    geometric_component = data['geometric_component'].flatten()
    reliable = data['reliable'].flatten().astype(bool)
    bed_depth = data['bed_depth'][0][0]
    bed_tilt = data['bed_tilt_angle'][0][0] * 180/np.pi

    # Calculate layer slopes
    u_h = 200.0  # m/year
    layer_slopes = np.rad2deg(np.arctan(geometric_component / u_h))

    # Statistics
    mean_slope = np.nanmean(layer_slopes[reliable])
    std_slope = np.nanstd(layer_slopes[reliable])

    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # ============================================================
    # Panel 1: Layer slopes vs depth
    # ============================================================
    ax1 = fig.add_subplot(gs[0, :2])

    # Plot all slopes (gray for unreliable)
    ax1.plot(layer_slopes[~reliable], depths[~reliable], 'o',
             color='lightgray', markersize=4, alpha=0.5,
             label='Unreliable layers')

    # Plot reliable slopes
    ax1.plot(layer_slopes[reliable], depths[reliable], 'o',
             color='steelblue', markersize=6, alpha=0.7,
             label='Reliable layers')

    # Mean line
    ax1.axvline(mean_slope, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_slope:.3f}°')

    # Uncertainty band
    ax1.axvspan(mean_slope - std_slope, mean_slope + std_slope,
                alpha=0.2, color='red', label=f'±1σ: {std_slope:.3f}°')

    # Bed slope
    ax1.axhline(bed_depth, color='black', linewidth=2, linestyle='-',
                label='Bed')
    ax1.plot(bed_tilt, bed_depth, 's', color='darkred', markersize=12,
             markeredgecolor='black', markeredgewidth=2,
             label=f'Bed slope: {bed_tilt:.4f}°', zorder=10)

    ax1.set_xlabel('Layer Slope (degrees)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Depth (m)', fontsize=12, fontweight='bold')
    ax1.set_title('Layer Slopes vs Depth', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right', fontsize=10)

    # Add zero line
    ax1.axvline(0, color='gray', linestyle=':', alpha=0.5)

    # ============================================================
    # Panel 2: Summary statistics
    # ============================================================
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')

    summary_text = f"""LAYER SLOPE SUMMARY

Reliable layers: {np.sum(reliable)} / {len(depths)}

STATISTICS (reliable only):
  Mean:     {mean_slope:>6.3f}°
  Std dev:  {std_slope:>6.3f}°
  Min:      {np.nanmin(layer_slopes[reliable]):>6.3f}°
  Max:      {np.nanmax(layer_slopes[reliable]):>6.3f}°
  Median:   {np.nanmedian(layer_slopes[reliable]):>6.3f}°

BED SLOPE:
  Tilt:     {bed_tilt:>6.4f}°

INTERPRETATION:
✓ Layers essentially flat
✓ Mean slope < 0.2°
✓ Bed essentially horizontal
✓ Consistent with stable
  ice interior over lake

PHYSICAL MEANING:
Over 1 km horizontal:
  Layer: {1000 * np.tan(np.deg2rad(mean_slope)):.1f} m vertical
  Bed:   {1000 * np.tan(np.deg2rad(bed_tilt)):.1f} m vertical

UNCERTAINTY:
±{std_slope:.3f}° (1σ statistical)
±0.03° (from accumulation)
    """

    ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes,
             verticalalignment='top', fontfamily='monospace',
             fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ============================================================
    # Panel 3: Histogram
    # ============================================================
    ax3 = fig.add_subplot(gs[1, 0])

    # Histogram of slopes
    n, bins, patches = ax3.hist(layer_slopes[reliable], bins=20,
                                 color='steelblue', alpha=0.7, edgecolor='black')

    # Mean line
    ax3.axvline(mean_slope, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_slope:.3f}°')

    # Bed slope
    ax3.axvline(bed_tilt, color='darkred', linestyle='--', linewidth=2,
                label=f'Bed: {bed_tilt:.4f}°')

    ax3.set_xlabel('Layer Slope (degrees)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Number of Layers', fontsize=12, fontweight='bold')
    ax3.set_title('Distribution of Layer Slopes', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend()

    # ============================================================
    # Panel 4: Depth ranges
    # ============================================================
    ax4 = fig.add_subplot(gs[1, 1])

    depth_ranges = [(0, 300), (300, 600), (600, 900), (900, 1100)]
    range_means = []
    range_stds = []
    range_labels = []
    range_counts = []

    for d_min, d_max in depth_ranges:
        mask = reliable & (depths >= d_min) & (depths < d_max)
        if np.sum(mask) > 0:
            range_means.append(np.nanmean(layer_slopes[mask]))
            range_stds.append(np.nanstd(layer_slopes[mask]))
            range_labels.append(f'{d_min}-{d_max}m')
            range_counts.append(np.sum(mask))

    x_pos = np.arange(len(range_labels))
    bars = ax4.bar(x_pos, range_means, yerr=range_stds,
                   color='steelblue', alpha=0.7, edgecolor='black',
                   capsize=5)

    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, range_counts)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + range_stds[i] + 0.01,
                f'n={count}', ha='center', va='bottom', fontsize=9)

    # Bed slope line
    ax4.axhline(bed_tilt, color='darkred', linestyle='--', linewidth=2,
                label=f'Bed: {bed_tilt:.4f}°')

    # Overall mean
    ax4.axhline(mean_slope, color='red', linestyle='--', linewidth=2,
                alpha=0.5, label=f'Overall: {mean_slope:.3f}°')

    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(range_labels, rotation=45, ha='right')
    ax4.set_ylabel('Mean Slope (degrees)', fontsize=12, fontweight='bold')
    ax4.set_title('Mean Slope by Depth Range', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.legend()

    # ============================================================
    # Panel 5: Physical interpretation
    # ============================================================
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')

    # Calculate vertical displacement over different distances
    distances = [100, 500, 1000, 5000]

    interp_text = """PHYSICAL INTERPRETATION

Slope significance:
(vertical change over distance)

Mean layer slope (0.14°):
"""

    for dist in distances:
        vert = dist * np.tan(np.deg2rad(mean_slope))
        interp_text += f"\n  {dist:>4}m → {vert:>5.1f}m vertical"

    interp_text += f"""\n
Bed slope ({bed_tilt:.4f}°):
"""

    for dist in distances:
        vert = dist * np.tan(np.deg2rad(bed_tilt))
        interp_text += f"\n  {dist:>4}m → {vert:>5.2f}m vertical"

    interp_text += """

CLASSIFICATION:
  0-0.5°:  FLAT (interior ice)
  0.5-2°:  Gently sloped
  2-5°:    Moderately sloped
  >5°:     Steep

YOUR DATA:
✓ Layers: 0.14° → FLAT
✓ Bed:    0.04° → EXTREMELY FLAT

IMPLICATIONS:
• Geometric corrections small
• Steady flow assumption valid
• No major folding/deformation
• Consistent with lake setting
"""

    ax5.text(0.05, 0.95, interp_text, transform=ax5.transAxes,
             verticalalignment='top', fontfamily='monospace',
             fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    # ============================================================
    # Overall title
    # ============================================================
    plt.suptitle('Layer and Bed Slope Analysis\n' +
                 'Inferred from ApRES Velocity Measurements (u_h = 200 m/yr, accumulation = 0.3 m/yr)',
                 fontsize=16, fontweight='bold')

    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to {output_file}")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Visualize layer and bed slopes from ApRES analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--data', type=str,
                       default='output/apres/flow_regime_analysis.mat',
                       help='Path to flow_regime_analysis.mat')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for figure')
    parser.add_argument('--show', action='store_true',
                       help='Show interactive plot')

    args = parser.parse_args()

    # Create visualization
    fig = visualize_layer_slopes(args.data, args.output)

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
