#!/usr/bin/env python3
"""
ApRES Internal Layer Velocity Analysis - Main Pipeline

This script orchestrates the complete analysis workflow:
1. Layer detection
2. Phase tracking  
3. Velocity profile calculation
4. Visualization

Usage:
    python run_analysis.py --data /path/to/ImageP2_python.mat
    python run_analysis.py --help

Author: SiegVent2023 project
"""

import argparse
import sys
import numpy as np
from pathlib import Path

# Add module path
sys.path.insert(0, str(Path(__file__).parent))

from layer_detection import (
    load_apres_data, 
    detect_layers, 
    visualize_layers, 
    save_layers
)
from phase_tracking import (
    load_layer_data,
    track_all_layers,
    visualize_phase_tracking,
    save_phase_results,
)
from velocity_profile import (
    load_phase_data,
    calculate_velocity_profile,
    visualize_velocity_profile,
    save_velocity_results,
)


def run_pipeline(
    data_path: str,
    output_dir: str,
    min_depth: float = 50,
    max_depth: float = 1000,
    min_snr: float = 10,
    min_persistence: float = 0.5,
    r_sq_threshold: float = 0.3,
    amp_threshold: float = -80,
    interactive: bool = True,
) -> dict:
    """
    Run the complete analysis pipeline.
    
    Args:
        data_path: Path to ImageP2_python.mat
        output_dir: Directory for output files
        min_depth: Minimum depth for layer detection (m)
        max_depth: Maximum depth for layer detection (m)
        min_snr: Minimum SNR for layer detection (dB)
        r_sq_threshold: Minimum R² for reliable velocity
        amp_threshold: Minimum amplitude for reliable phase (dB)
        interactive: Whether to show interactive plots
        
    Returns:
        Dictionary with all results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # =========================================================================
    # Step 1: Layer Detection
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 1: LAYER DETECTION")
    print("="*60)
    
    # Load data
    data = load_apres_data(data_path)
    
    # Detect layers
    layers = detect_layers(
        data['range_img'],
        data['Rcoarse'],
        min_depth=min_depth,
        max_depth=max_depth,
        min_snr_db=min_snr,
        min_persistence=min_persistence,
    )
    
    # Save results
    layer_output = str(output_path / 'detected_layers')
    save_layers(layers, layer_output)
    
    # Visualize
    if interactive:
        fig = visualize_layers(
            data['range_img'],
            data['Rcoarse'],
            data['time_days'],
            layers,
            output_file=str(output_path / 'layer_detection.html'),
        )
    
    results['layers'] = layers
    
    # Validation checkpoint
    if layers.n_layers < 3:
        print(f"\n⚠️  WARNING: Only {layers.n_layers} layers detected!")
        print("    Consider lowering min_snr or adjusting depth range.")
        if interactive:
            response = input("    Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Pipeline stopped by user.")
                return results
    
    print(f"\n✓ Layer detection complete: {layers.n_layers} layers found")
    
    # =========================================================================
    # Step 2: Phase Tracking
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 2: PHASE TRACKING")
    print("="*60)
    
    # Load layer data
    layer_data, apres_data = load_layer_data(layer_output, data_path)
    
    # Track phases
    phase_result = track_all_layers(layer_data, apres_data)
    
    # Save results
    phase_output = str(output_path / 'phase_tracking')
    save_phase_results(phase_result, phase_output)
    
    # Visualize
    if interactive:
        visualize_phase_tracking(
            phase_result,
            n_layers_to_show=6,
            output_file=str(output_path / 'phase_tracking.html'),
        )
    
    results['phase'] = phase_result
    
    print(f"\n✓ Phase tracking complete for {phase_result.n_layers} layers")
    
    # =========================================================================
    # Step 3: Velocity Profile
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 3: VELOCITY PROFILE")
    print("="*60)
    
    # Load phase data
    phase_data = load_phase_data(phase_output)
    
    # Calculate velocities
    velocity_result = calculate_velocity_profile(
        phase_data,
        r_sq_threshold=r_sq_threshold,
        amp_threshold_db=amp_threshold,
    )
    
    # Save results
    velocity_output = str(output_path / 'velocity_profile')
    save_velocity_results(velocity_result, velocity_output)
    
    # Visualize
    if interactive:
        visualize_velocity_profile(
            velocity_result,
            output_file=str(output_path / 'velocity_profile.html'),
        )
    
    results['velocity'] = velocity_result
    
    print(f"\n✓ Velocity profile complete: {np.sum(velocity_result.reliable)} reliable measurements")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nOutput files saved to: {output_path}")
    print(f"  - layer_detection.html")
    print(f"  - phase_tracking.html")
    print(f"  - velocity_profile.html")
    print(f"  - detected_layers.mat/.json")
    print(f"  - phase_tracking.mat")
    print(f"  - velocity_profile.mat/.json")
    
    if velocity_result.n_layers > 0 and np.any(velocity_result.reliable):
        reliable = velocity_result.reliable
        print(f"\nVelocity Summary:")
        print(f"  Depth range: {velocity_result.depths[reliable].min():.0f} - {velocity_result.depths[reliable].max():.0f} m")
        print(f"  Velocity range: {velocity_result.velocities[reliable].min():.2f} - {velocity_result.velocities[reliable].max():.2f} m/year")
        print(f"  Mean velocity: {np.mean(velocity_result.velocities[reliable]):.2f} m/year")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='ApRES Internal Layer Velocity Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults
  python run_analysis.py
  
  # Specify data and output paths
  python run_analysis.py --data /path/to/ImageP2_python.mat --output /path/to/results
  
  # Adjust thresholds
  python run_analysis.py --min-snr 8 --r-sq-threshold 0.5
  
  # Run without interactive plots (for batch processing)
  python run_analysis.py --no-interactive
        """
    )
    
    parser.add_argument('--data', type=str, 
                        default='../../data/apres/ImageP2_python.mat',
                        help='Path to processed ApRES data (.mat)')
    parser.add_argument('--output', type=str,
                        default='../../data/apres/layer_analysis',
                        help='Output directory for results')
    parser.add_argument('--min-depth', type=float, default=50,
                        help='Minimum depth to analyze (m)')
    parser.add_argument('--max-depth', type=float, default=1000,
                        help='Maximum depth to analyze (m)')
    parser.add_argument('--min-snr', type=float, default=10,
                        help='Minimum SNR for layer detection (dB)')
    parser.add_argument('--min-persistence', type=float, default=0.5,
                        help='Minimum fraction of time layer must be visible (0-1)')
    parser.add_argument('--r-sq-threshold', type=float, default=0.3,
                        help='Minimum R² for reliable velocity')
    parser.add_argument('--amp-threshold', type=float, default=-80,
                        help='Minimum amplitude for reliable phase (dB)')
    parser.add_argument('--no-interactive', action='store_true',
                        help='Disable interactive plots')
    
    args = parser.parse_args()
    
    # Run pipeline
    results = run_pipeline(
        data_path=args.data,
        output_dir=args.output,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        min_snr=args.min_snr,
        min_persistence=args.min_persistence,
        r_sq_threshold=args.r_sq_threshold,
        amp_threshold=args.amp_threshold,
        interactive=not args.no_interactive,
    )
    
    return results


if __name__ == '__main__':
    main()
