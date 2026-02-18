#!/usr/bin/env python3
"""
Basal Reflection Coefficient Analysis

Calculate reflection coefficient from ApRES bed echo to assess basal water content.
Based on Peters et al. (2007) IEEE Trans. Geosci. Remote Sens.

The reflection coefficient R quantifies the strength of the bed return
relative to the transmitted power, accounting for:
- Geometric spreading
- Ice attenuation
- System losses

Classification (Peters et al. 2007):
- R > -3 dB: Lots of water (strong reflector)
- -12 dB < R < -3 dB: Some water present
- -30 dB < R < -12 dB: Intermediate (mixed conditions)
- R < -30 dB: Likely no water (frozen interface)

Usage:
    python calculate_reflection_coefficient.py \\
        --bed-amplitude 8.6 \\
        --ice-thickness 1094.8

Author: SiegVent2023 project
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class ReflectionCoefficientResult:
    """Container for reflection coefficient analysis results."""
    R_db: float
    bed_amplitude_db: float
    ice_thickness_m: float
    ice_attenuation_db_per_km: float
    two_way_loss_db: float
    water_content_assessment: str
    confidence: str
    interpretation: str


def calculate_ice_attenuation(
    ice_thickness_m: float,
    temperature_C: float = -20.0,
    impurity_ppm: float = 1.7,
) -> float:
    """
    Calculate two-way ice attenuation at 300 MHz.

    Based on MacGregor et al. (2007) and Peters et al. (2007).

    Args:
        ice_thickness_m: Ice thickness (m)
        temperature_C: Ice temperature (°C)
        impurity_ppm: Impurity content (acid + salt, ppm)

    Returns:
        Two-way attenuation (dB)
    """
    # Attenuation rate from MacGregor et al. (2007)
    # a = 4a_0 + 0.00912*x_s + 0.0576*y_s
    # where x_s = acids (ppm), y_s = salts (ppm)

    # For typical Antarctic ice:
    # Temperature dependence: warmer = more attenuation
    # Impurity dependence: higher impurities = more attenuation

    # Peters et al. (2007): Kamb Ice Stream a ≈ 11.5-15 dB/km
    # Use temperature-corrected value

    # Reference attenuation at -20°C
    a_ref = 13.0  # dB/km (mid-range for West Antarctic ice streams)

    # Temperature correction (warmer ice has higher loss)
    # From Matsuoka et al. (2012)
    T_ref = -20.0
    temp_factor = 1.0 + 0.05 * (temperature_C - T_ref)  # 5% per degree

    # Impurity correction
    impurity_factor = impurity_ppm / 1.7  # Normalized to typical value

    # Combined attenuation rate
    attenuation_db_per_km = a_ref * temp_factor * impurity_factor

    # Two-way path
    two_way_attenuation_db = 2 * attenuation_db_per_km * ice_thickness_m / 1000

    return attenuation_db_per_km, two_way_attenuation_db


def calculate_reflection_coefficient(
    bed_amplitude_db: float,
    ice_thickness_m: float,
    transmit_power_w: float = 8000.0,
    wavelength_m: float = 0.6,
    antenna_gain_db: float = 9.4,
    system_losses_db: float = 4.5,
    ice_temperature_C: float = -20.0,
    impurity_ppm: float = 1.7,
) -> ReflectionCoefficientResult:
    """
    Calculate basal reflection coefficient from ApRES measurements.

    Based on Peters et al. (2007), Eq. 9:

    P_r = P_t * (λ/4π)² * (G_a² T² L_i² L_s G_p) / [2(h + z/n_2)²] * R

    Where:
    - P_r: Received power (dB)
    - P_t: Transmitted power (W)
    - λ: Wavelength (m)
    - G_a: Antenna gain (linear)
    - T: Surface transmission coefficient
    - L_i: Ice attenuation loss (two-way)
    - L_s: System losses
    - h: Antenna height above ice
    - z: Ice thickness
    - n_2: Ice refractive index
    - R: Power reflection coefficient (what we solve for)

    Args:
        bed_amplitude_db: Measured bed return amplitude (dB)
        ice_thickness_m: Ice thickness (m)
        transmit_power_w: Transmit power (W, default 8000 for ApRES)
        wavelength_m: Radar wavelength (m, default 0.6 at 305 MHz)
        antenna_gain_db: Antenna gain (dB, default 9.4 for ApRES)
        system_losses_db: System losses (dB, default 4.5)
        ice_temperature_C: Average ice temperature (°C)
        impurity_ppm: Ice impurity content (ppm)

    Returns:
        ReflectionCoefficientResult object
    """
    # Physical constants
    n_ice = 1.78  # Refractive index of ice
    T_surface = 1.0  # Surface transmission (assume no loss at air/ice)

    # Convert gains to linear
    G_a_linear = 10**(antenna_gain_db / 10)
    L_s_linear = 10**(-system_losses_db / 10)

    # Calculate ice attenuation
    atten_db_per_km, L_i_db = calculate_ice_attenuation(
        ice_thickness_m,
        ice_temperature_C,
        impurity_ppm,
    )
    L_i_linear = 10**(-L_i_db / 10)

    # Received power (convert from dB to linear)
    # Assume bed_amplitude_db is relative to transmit power
    P_r_linear = 10**(bed_amplitude_db / 10)

    # Range to reflector (one-way in ice)
    # For stationary ApRES on surface: h = 0
    range_m = ice_thickness_m / n_ice

    # Calculate geometric spreading factor
    geometric_factor = (wavelength_m / (4 * np.pi))**2

    # System gain factor
    # G_p (processing gain) ≈ 1 for unfocused ApRES
    G_p = 1.0
    system_factor = G_a_linear**2 * T_surface**2 * L_i_linear**2 * L_s_linear * G_p

    # Range spreading factor (two-way path)
    range_factor = 2 * range_m**2

    # Solve for R from radar equation
    # P_r = P_t * geometric * system / range * R
    # R = P_r * range / (P_t * geometric * system)

    R_linear = (P_r_linear * range_factor) / (
        transmit_power_w * geometric_factor * system_factor
    )

    # Convert to dB
    R_db = 10 * np.log10(R_linear)

    # Assess water content
    if R_db > -3:
        water_content = "Lots of water (strong reflector)"
        confidence = "High"
        interpretation = "Subglacial lake or thick water layer (>10 cm)"
    elif R_db > -12:
        water_content = "Some water present"
        confidence = "Moderate"
        interpretation = "Wet bed, water film, or thin water layer (1-10 cm)"
    elif R_db > -30:
        water_content = "Intermediate (mixed conditions)"
        confidence = "Low"
        interpretation = "Possible temperate ice/bed interface or water-saturated till"
    else:
        water_content = "Likely no water (frozen)"
        confidence = "Moderate"
        interpretation = "Cold-based, frozen to bedrock, or low dielectric contrast"

    return ReflectionCoefficientResult(
        R_db=R_db,
        bed_amplitude_db=bed_amplitude_db,
        ice_thickness_m=ice_thickness_m,
        ice_attenuation_db_per_km=atten_db_per_km,
        two_way_loss_db=L_i_db,
        water_content_assessment=water_content,
        confidence=confidence,
        interpretation=interpretation,
    )


def visualize_reflection_analysis(
    result: ReflectionCoefficientResult,
    output_file: Optional[str] = None,
):
    """Create visualization of reflection coefficient analysis."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Power budget breakdown
    ax1.axis('off')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    budget_text = f"""POWER BUDGET ANALYSIS

Input Parameters:
  Bed amplitude: {result.bed_amplitude_db:+.1f} dB
  Ice thickness: {result.ice_thickness_m:.1f} m
  Ice attenuation: {result.ice_attenuation_db_per_km:.1f} dB/km

Two-Way Path Losses:
  Ice attenuation: {result.two_way_loss_db:.1f} dB
  (2 × {result.ice_attenuation_db_per_km:.1f} × {result.ice_thickness_m/1000:.2f} km)

REFLECTION COEFFICIENT:
  R = {result.R_db:+.1f} dB

Classification: {result.water_content_assessment}
Confidence: {result.confidence}

Physical Interpretation:
{result.interpretation}
"""

    ax1.text(0.05, 0.95, budget_text, transform=ax1.transAxes,
             verticalalignment='top', fontfamily='monospace',
             fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel 2: R value scale with interpretation
    ax2.set_xlim(-50, 5)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    # Draw scale
    scale_y = 0.5
    scale_height = 0.05

    # Zones
    zones = [
        (-50, -30, 'lightgray', 'Likely no water\n(frozen bed)'),
        (-30, -12, 'lightyellow', 'Intermediate\n(mixed)'),
        (-12, -3, 'lightblue', 'Some water'),
        (-3, 5, 'darkblue', 'Lots of water\n(lake)'),
    ]

    for (x_min, x_max, color, label) in zones:
        width = x_max - x_min
        rect = plt.Rectangle((x_min, scale_y - scale_height/2), width, scale_height,
                            facecolor=color, edgecolor='black', linewidth=1)
        ax2.add_patch(rect)

        # Add label above
        ax2.text((x_min + x_max)/2, scale_y + scale_height, label,
                ha='center', va='bottom', fontsize=9)

        # Add R value boundaries
        ax2.axvline(x_max, color='black', linewidth=1, linestyle='--',
                   alpha=0.5, ymin=0.3, ymax=0.7)
        ax2.text(x_max, scale_y - scale_height - 0.05, f'{x_max:+.0f} dB',
                ha='center', va='top', fontsize=8)

    # Mark your value
    ax2.plot(result.R_db, scale_y, 'r^', markersize=15, markeredgecolor='black',
            markeredgewidth=2, label='Your measurement')
    ax2.text(result.R_db, scale_y - scale_height - 0.15,
            f'Your R:\n{result.R_db:+.1f} dB',
            ha='center', va='top', fontsize=10, fontweight='bold',
            color='red')

    ax2.set_title('Reflection Coefficient Classification', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')

    plt.suptitle('Basal Reflection Coefficient Analysis (Peters et al. 2007)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {output_file}")

    return fig


def print_summary(result: ReflectionCoefficientResult):
    """Print text summary of reflection coefficient analysis."""
    print("\n" + "="*70)
    print("BASAL REFLECTION COEFFICIENT ANALYSIS")
    print("="*70)

    print(f"\n{'INPUT PARAMETERS':-^70}")
    print(f"Bed echo amplitude:             {result.bed_amplitude_db:+.1f} dB")
    print(f"Ice thickness:                  {result.ice_thickness_m:.1f} m")

    print(f"\n{'ICE ATTENUATION':-^70}")
    print(f"Attenuation rate:               {result.ice_attenuation_db_per_km:.1f} dB/km")
    print(f"Two-way path loss:              {result.two_way_loss_db:.1f} dB")

    print(f"\n{'REFLECTION COEFFICIENT':-^70}")
    print(f"R =                             {result.R_db:+.1f} dB")

    print(f"\n{'INTERPRETATION':-^70}")
    print(f"Water content:                  {result.water_content_assessment}")
    print(f"Confidence:                     {result.confidence}")
    print(f"Physical interpretation:        {result.interpretation}")

    print(f"\n{'REFERENCE VALUES (Peters et al. 2007)':-^70}")
    print("  R > -3 dB:        Lots of water (subglacial lake)")
    print("  -12 < R < -3 dB:  Some water (wet bed, water film)")
    print("  -30 < R < -12 dB: Intermediate (mixed conditions)")
    print("  R < -30 dB:       Likely no water (frozen bed)")

    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Calculate basal reflection coefficient from ApRES data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with measured values
  python calculate_reflection_coefficient.py \\
      --bed-amplitude 8.6 \\
      --ice-thickness 1094.8

  # With temperature and impurity estimates
  python calculate_reflection_coefficient.py \\
      --bed-amplitude 8.6 \\
      --ice-thickness 1094.8 \\
      --temperature -15 \\
      --impurity 2.0 \\
      --output results/reflection_coefficient

Reference:
  Peters et al. (2007) "Along-Track Focusing of Airborne Radar Sounding
  Data From West Antarctica for Improving Basal Reflection Analysis and
  Layer Detection" IEEE Trans. Geosci. Remote Sens.
        """
    )

    parser.add_argument('--bed-amplitude', type=float, required=True,
                       help='Bed echo amplitude (dB)')
    parser.add_argument('--ice-thickness', type=float, required=True,
                       help='Ice thickness (m)')
    parser.add_argument('--transmit-power', type=float, default=8000.0,
                       help='Transmit power (W, default: 8000)')
    parser.add_argument('--temperature', type=float, default=-20.0,
                       help='Average ice temperature (°C, default: -20)')
    parser.add_argument('--impurity', type=float, default=1.7,
                       help='Ice impurity content (ppm, default: 1.7)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for results (without extension)')
    parser.add_argument('--save-figure', action='store_true',
                       help='Save figure to file')

    args = parser.parse_args()

    # Calculate reflection coefficient
    print("Calculating basal reflection coefficient...")
    result = calculate_reflection_coefficient(
        bed_amplitude_db=args.bed_amplitude,
        ice_thickness_m=args.ice_thickness,
        transmit_power_w=args.transmit_power,
        ice_temperature_C=args.temperature,
        impurity_ppm=args.impurity,
    )

    # Print summary
    print_summary(result)

    # Save results
    if args.output:
        results_dict = {
            'R_db': float(result.R_db),
            'bed_amplitude_db': float(result.bed_amplitude_db),
            'ice_thickness_m': float(result.ice_thickness_m),
            'ice_attenuation_db_per_km': float(result.ice_attenuation_db_per_km),
            'two_way_loss_db': float(result.two_way_loss_db),
            'water_content_assessment': result.water_content_assessment,
            'confidence': result.confidence,
            'interpretation': result.interpretation,
        }

        with open(f"{args.output}.json", 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"Results saved to {args.output}.json")

    # Visualize
    output_fig = f"{args.output}.png" if args.save_figure and args.output else None
    visualize_reflection_analysis(result, output_file=output_fig)

    plt.show()


if __name__ == '__main__':
    main()
