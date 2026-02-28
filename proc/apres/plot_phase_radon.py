#!/usr/bin/env python3
"""
Complex Phase-Based Radon Transform for ApRES Data
Visually demonstrates the transform over a 20m depth window.

Usage:
    python plot_phase_radon.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pathlib import Path
import warnings

# Ignore complex casting warnings from matplotlib
warnings.filterwarnings('ignore')

def main():
    # Parameters
    data_path = 'data/apres/ImageP2_python.mat'
    lambdac = 0.5608
    depths_to_test = [200.0, 500.0, 800.0, 1090.0]
    window_size = 20.0
    v_min, v_max, num_v = 0.0, 2.0, 1000

    print(f"Loading data from {data_path}...")
    try:
        mat = loadmat(data_path)
    except Exception as e:
        print(f"Error loading {data_path}: {e}")
        return

    Rcoarse = mat['Rcoarse'].flatten()
    time_days = mat['TimeInDays'].flatten()
    time_years = time_days / 365.25
    
    # Load and downcast to save RAM (matching the visual app optimization)
    raw_complex = np.array(mat['RawImageComplex'], dtype=np.complex64)

    output_dir = Path('output/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)

    for target_depth in depths_to_test:
        print(f"\n--- Testing Depth: {target_depth}m ---")
        
        # Select depth window
        depth_mask = (Rcoarse >= target_depth - window_size/2) & (Rcoarse <= target_depth + window_size/2)
        if not np.any(depth_mask):
            print(f"No data found near {target_depth}m. Skipping.")
            continue
            
        z_win = Rcoarse[depth_mask]
        S_win = raw_complex[depth_mask, :]

        print(f"Selected window from {z_win[0]:.1f}m to {z_win[-1]:.1f}m ({len(z_win)} bins)")

        # Local SVD Denoising (k=3)
        print("Applying local SVD denoising (k=3)...")
        U, sigma, Vh = np.linalg.svd(S_win, full_matrices=False)
        k = 3
        S_denoised = (U[:, :k] * sigma[:k]) @ Vh[:k, :]

        # Range of velocities to test (m/yr)
        velocities = np.linspace(v_min, v_max, num_v)

        # Complex Radon Transform (Stacking)
        print("Computing Complex Phase-Based Radon Transform...")
        stack_map = np.zeros((len(z_win), num_v))

        for i, v in enumerate(velocities):
            # phi(t) = 4 * pi * v * t / lambdac
            expected_phase = (4 * np.pi * v * time_years) / lambdac
            derotated = S_denoised * np.exp(-1j * expected_phase)
            stack_map[:, i] = np.abs(np.mean(derotated, axis=1))

        # Normalize map for visualization
        stack_map_norm = stack_map / (np.max(stack_map, axis=1, keepdims=True) + 1e-10)

        # Bulk stack (summing across depths)
        bulk_stack = np.sum(stack_map, axis=0)
        bulk_stack_norm = bulk_stack / np.max(bulk_stack)
        max_v = velocities[np.argmax(bulk_stack)]

        # ---------------- Plotting ---------------- #
        fig = plt.figure(figsize=(12, 6))
        
        # Grid layout
        gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.15)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        # Left Panel: Heatmap (Depth vs Velocity)
        im = ax1.imshow(stack_map_norm, aspect='auto', cmap='magma', 
                        extent=[v_min, v_max, z_win[-1], z_win[0]])
        ax1.set_xlabel('Tested Velocity (m/yr)', fontsize=12)
        ax1.set_ylabel('Depth (m)', fontsize=12)
        ax1.set_title(f'Phase-Radon Map ({target_depth}m, 20m window)', fontsize=14)
        cbar = fig.colorbar(im, ax=ax1)
        cbar.set_label('Normalized Stack Magnitude')

        # Add line for the peak
        ax1.axvline(max_v, color='cyan', linestyle='--', linewidth=2, label=f'Peak = {max_v:.3f} m/yr')
        ax1.legend(loc='upper right')

        # Right Panel: Bulk Average
        ax2.plot(velocities, bulk_stack_norm, color='black', linewidth=2)
        ax2.fill_between(velocities, 0, bulk_stack_norm, color='gray', alpha=0.3)
        ax2.axvline(max_v, color='cyan', linestyle='--', linewidth=2)
        ax2.set_xlabel('Tested Velocity (m/yr)', fontsize=12)
        ax2.set_ylabel('Bulk Stack Magnitude (Summed)', fontsize=12)
        ax2.set_title(f'Window Bulk Velocity\nPeak = {max_v:.3f} m/yr', fontsize=14)
        ax2.set_xlim(v_min, v_max)
        ax2.set_ylim(0, 1.05)
        ax2.grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout()
        
        output_file = output_dir / f'phase_radon_depth_{int(target_depth)}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Success! Plot saved to: {output_file}")

if __name__ == '__main__':
    main()
