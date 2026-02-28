import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

def main():
    data_path = 'data/apres/ImageP2_python.mat'
    lambdac = 0.5608
    window_size = 20.0
    step_size = 2.0  # Calculate every 2 meters
    v_min, v_max, num_v = 0.0, 2.0, 1000
    depth_min, depth_max = 0.0, 1090.0

    print(f"Loading data from {data_path}...")
    try:
        mat = loadmat(data_path)
    except Exception as e:
        print(f"Error loading {data_path}: {e}")
        return

    Rcoarse = mat['Rcoarse'].flatten()
    time_days = mat['TimeInDays'].flatten()
    time_years = time_days / 365.25
    
    raw_complex = np.array(mat['RawImageComplex'], dtype=np.complex64)
    
    depths = np.arange(depth_min + window_size/2, depth_max - window_size/2, step_size)
    velocities = np.linspace(v_min, v_max, num_v)
    
    # Precompute the complex exponential derotation matrix
    # E_T shape: (n_times, num_v)
    print("Precomputing phase derotation matrix for extreme fast dot-products...")
    phase_matrix = (4 * np.pi / lambdac) * np.outer(time_years, velocities)
    E_T = np.exp(-1j * phase_matrix)
    n_times = len(time_years)
    
    best_velocities = []
    stack_max_vals = []
    
    print(f"Testing {len(depths)} windows from {depths[0]:.1f}m to {depths[-1]:.1f}m...")
    t0 = time.time()
    for i, target_depth in enumerate(depths):
        depth_mask = (Rcoarse >= target_depth - window_size/2) & (Rcoarse <= target_depth + window_size/2)
        if not np.any(depth_mask):
            best_velocities.append(np.nan)
            stack_max_vals.append(np.nan)
            continue
            
        S_win = raw_complex[depth_mask, :]
        
        # Local SVD Denoising (k=3)
        U, sigma, Vh = np.linalg.svd(S_win, full_matrices=False)
        S_denoised = (U[:, :3] * sigma[:3]) @ Vh[:3, :]
        
        # Fast Phase-Radon Stack via matrix multiplication
        # np.mean(S_denoised * exp(-i phase), axis=1) is mathematically equivalent to:
        # (S_denoised @ E_T) / n_times
        stack_mag = np.abs(S_denoised @ E_T) / n_times
        stack_map_1d = np.sum(stack_mag, axis=0)
            
        best_idx = np.argmax(stack_map_1d)
        best_velocities.append(velocities[best_idx])
        stack_max_vals.append(stack_map_1d[best_idx])
        
        if (i+1) % 100 == 0:
            print(f"  Processed {i+1}/{len(depths)} depths... ({(time.time()-t0):.1f}s)")

    print(f"Finished processing in {(time.time()-t0):.1f}s.")

    # ---------------- Plotting ---------------- #
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Nye Model
    nye_int, nye_sl = 0.0453, 0.000595
    nye_v = nye_int + nye_sl * depths

    ax.plot(nye_v, depths, 'r-', lw=2, label='Nye Model (Prior)')
    
    # Use scatter with color representing peak stack power (coherence)
    sc = ax.scatter(best_velocities, depths, c=stack_max_vals, cmap='viridis', 
                    s=15, alpha=0.8, label='Phase-Radon Velocity Profile', edgecolors='none')
    
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label('Peak Stack Power (Coherence)')
    
    ax.set_xlim(v_min, v_max)
    ax.set_ylim(depth_max, depth_min) # Invert y-axis to match depth
    ax.set_xlabel('Velocity (m/yr)', fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)
    ax.set_title(f'Phase-Radon Full Velocity Profile\n(Window: {window_size}m, Step: {step_size}m)', fontsize=14)
    ax.legend(loc='lower left')
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    
    output_dir = Path('output/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'phase_radon_full_profile.png'
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Success! Plot saved to: {output_file}")

if __name__ == '__main__':
    main()
