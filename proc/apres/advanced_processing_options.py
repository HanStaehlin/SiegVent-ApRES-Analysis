"""
Advanced Signal Processing Options for Deep Layer Enhancement in ApRES Data

This script explores different approaches to enhance deep internal layers
in phase-sensitive radar data where signal-to-noise ratio decreases with depth.

Author: SiegVent2023 project
"""

import numpy as np
from scipy.io import loadmat
from scipy import signal, ndimage
from scipy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt

# Load complex data
data = loadmat('/Users/hannesstahlin/SiegVent2023-Geology/data/apres/ImageP2_python.mat')
raw_complex = data['RawImageComplex']
Rcoarse = data['Rcoarse'].flatten()
time_days = data['TimeInDays'].flatten()

print("="*70)
print("ADVANCED SIGNAL PROCESSING OPTIONS FOR DEEP LAYER ENHANCEMENT")
print("="*70)

print(f"\nData shape: {raw_complex.shape} (range bins × time steps)")
print(f"Depth range: {Rcoarse[0]:.1f} - {Rcoarse[-1]:.1f} m")

# Current amplitude
amp_original = np.abs(raw_complex)
amp_db = 20 * np.log10(amp_original + 1e-30)

# ============================================================================
# OPTION 1: DEPTH-DEPENDENT GAIN (Time-Varying Gain / TVG)
# ============================================================================
print("\n" + "="*70)
print("OPTION 1: DEPTH-DEPENDENT GAIN CORRECTION")
print("="*70)
print("""
Concept: Signal amplitude naturally decreases with depth due to:
  - Geometric spreading (1/R²)
  - Dielectric absorption in ice
  - Scattering losses

Solution: Apply depth-dependent gain to compensate:
  gain(R) = R^α × exp(β × R)
  where α ≈ 2 (geometric) and β = absorption coefficient

Pros:
  + Simple, well-understood physics
  + Enhances deep signals uniformly
  + Preserves phase information

Cons:
  - Also amplifies noise at depth
  - Need to estimate absorption coefficient
  - May over-amplify if parameters wrong
""")

# Estimate absorption from data
depth_bins = np.arange(100, 1000, 100)
mean_amp_vs_depth = []
for d in depth_bins:
    idx = np.argmin(np.abs(Rcoarse - d))
    mean_amp_vs_depth.append(np.mean(amp_db[idx-50:idx+50, :]))
mean_amp_vs_depth = np.array(mean_amp_vs_depth)

# Fit exponential decay
from scipy.optimize import curve_fit
def decay_model(R, a, b, c):
    return a - b * R + c * np.log10(R)

try:
    popt, _ = curve_fit(decay_model, depth_bins, mean_amp_vs_depth, p0=[0, 0.01, -10])
    print(f"Estimated signal decay: {popt[1]*1000:.2f} dB/km (absorption)")
except:
    print("Could not fit decay model")

# ============================================================================
# OPTION 2: COHERENT STACKING / MULTI-LOOK AVERAGING
# ============================================================================
print("\n" + "="*70)
print("OPTION 2: COHERENT STACKING (Multi-look averaging)")  
print("="*70)
print("""
Concept: Average multiple measurements to reduce incoherent noise
  - Coherent averaging: average complex values (preserves phase)
  - Incoherent averaging: average power (loses phase but better SNR)

For ApRES with moving layers, we can do:
  1. Short-time coherent stacks (e.g., 10-20 consecutive measurements)
  2. Phase-aligned stacking (align layers before stacking)

SNR improvement: √N for coherent, N for incoherent (in power)

Pros:
  + Proven technique for radar
  + Significant SNR improvement possible
  + Can preserve phase with careful alignment

Cons:
  - Reduces temporal resolution
  - Coherent stacking fails if layers move > λ/4 between measurements
  - Need to balance SNR vs temporal resolution
""")

# Demo: coherent stacking with window
stack_window = 20  # Stack 20 consecutive measurements
n_stacks = raw_complex.shape[1] // stack_window

stacked_complex = np.zeros((raw_complex.shape[0], n_stacks), dtype=complex)
for i in range(n_stacks):
    start = i * stack_window
    end = start + stack_window
    stacked_complex[:, i] = np.mean(raw_complex[:, start:end], axis=1)

amp_stacked = np.abs(stacked_complex)
snr_improvement = 10 * np.log10(stack_window)  # Theoretical for coherent
print(f"With {stack_window}-point stacking: theoretical SNR gain = {snr_improvement:.1f} dB")

# ============================================================================
# OPTION 3: FREQUENCY-WAVENUMBER (F-K) FILTERING
# ============================================================================
print("\n" + "="*70)
print("OPTION 3: FREQUENCY-WAVENUMBER (F-K) FILTERING")
print("="*70)
print("""
Concept: Transform to 2D Fourier domain and filter:
  - Horizontal axis → temporal frequency (layer motion rate)
  - Vertical axis → spatial frequency (layer spacing)

Real layers have:
  - Specific relationship between spatial and temporal frequencies
  - Noise is randomly distributed in F-K space

Can design filters to:
  1. Pass only physically plausible layer signals
  2. Remove horizontal noise (same depth, all times)
  3. Remove vertical noise (same time, all depths)

Pros:
  + Powerful for separating signal from noise
  + Can target specific layer velocities
  + Preserves layer structure

Cons:
  - Complex to implement correctly
  - Risk of removing real signal
  - Assumes layers are plane-wave-like
""")

# ============================================================================
# OPTION 4: SINGULAR VALUE DECOMPOSITION (SVD) FILTERING
# ============================================================================
print("\n" + "="*70)
print("OPTION 4: SINGULAR VALUE DECOMPOSITION (SVD) FILTERING")
print("="*70)
print("""
Concept: Decompose echogram into orthogonal components:
  Image = U × S × V^T
  
  - First few singular values: dominant coherent features (layers)
  - Middle singular values: weaker coherent features
  - Last singular values: incoherent noise

Can reconstruct using only selected singular values to:
  1. Remove noise (drop small singular values)
  2. Separate stationary vs moving features
  3. Extract specific layer patterns

Pros:
  + Data-driven, no assumptions about layer physics
  + Excellent for separating coherent from incoherent
  + Can reveal hidden structure

Cons:
  - Computationally intensive for large images
  - Choosing cutoff singular values is subjective
  - May mix different physical features
""")

# Demo SVD on a depth window
depth_min, depth_max = 700, 1050
mask = (Rcoarse >= depth_min) & (Rcoarse <= depth_max)
deep_region = raw_complex[mask, :]

U, S, Vh = np.linalg.svd(deep_region, full_matrices=False)
print(f"\nSVD of deep region ({depth_min}-{depth_max}m):")
print(f"  Singular values shape: {S.shape}")
print(f"  Top 5 singular values: {S[:5]}")
print(f"  Energy in top 10 components: {np.sum(S[:10]**2)/np.sum(S**2)*100:.1f}%")
print(f"  Energy in top 50 components: {np.sum(S[:50]**2)/np.sum(S**2)*100:.1f}%")

# ============================================================================
# OPTION 5: ADAPTIVE / DEPTH-DEPENDENT PROCESSING
# ============================================================================
print("\n" + "="*70)
print("OPTION 5: DEPTH-DEPENDENT ECHOGRAM GENERATION")
print("="*70)
print("""
Concept: Different processing for different depths:

For SHALLOW layers (high SNR):
  - Standard processing works fine
  - Can use stricter layer criteria

For DEEP layers (low SNR):
  - Apply aggressive noise reduction
  - Use longer coherent averaging
  - Lower detection thresholds
  - Accept lower temporal resolution

Implementation:
  1. Split data into depth bands
  2. Process each band optimally
  3. Merge results with appropriate weighting

This is essentially what seismic processing does with 
depth-varying deconvolution and filtering.
""")

# ============================================================================
# OPTION 6: PHASE GRADIENT METHODS (DIFFERENTIAL INTERFEROMETRY)
# ============================================================================
print("\n" + "="*70)
print("OPTION 6: DIFFERENTIAL INTERFEROMETRY (Phase Gradient)")
print("="*70)
print("""
Concept: Instead of tracking absolute phase, track phase differences:
  
  ΔΦ(t) = Φ(t+1) - Φ(t)

This has several advantages:
  1. Removes static phase offsets
  2. More robust to absolute phase errors
  3. Directly measures velocity (dΦ/dt = velocity)
  4. Can detect layers by temporal coherence of phase gradient

For deep layers:
  - Even if absolute phase is noisy, the CHANGE might be detectable
  - Temporal coherence of phase change identifies real layers

Pros:
  + More sensitive to motion than absolute phase
  + Less affected by systematic errors
  + Can work at lower SNR

Cons:
  - Loses absolute displacement information
  - Sensitive to phase unwrapping errors
  - Requires careful temporal sampling
""")

# ============================================================================
# OPTION 7: MULTI-LOOKING WITH COMPLEX COHERENCE
# ============================================================================
print("\n" + "="*70)
print("OPTION 7: COHERENCE-BASED LAYER DETECTION")
print("="*70)
print("""
Concept: Use temporal coherence as a layer quality metric:

  γ = |⟨z₁ × z₂*⟩| / √(⟨|z₁|²⟩⟨|z₂|²⟩)

Where z₁, z₂ are complex values at different times.

High coherence (γ → 1): Stable reflector (real layer)
Low coherence (γ → 0): Random scattering (noise)

This directly identifies where phase information is reliable,
regardless of amplitude!

Implementation:
  1. Compute coherence maps between consecutive measurements
  2. Average coherence over time windows
  3. Detect layers where coherence is high
  4. Use coherence to weight phase tracking

Pros:
  + Directly measures what we care about (phase stability)
  + Works even at low amplitude
  + Proven in InSAR applications

Cons:
  - Requires multiple measurements per coherence estimate
  - Coherence decreases with time gap
  - Need enough temporal sampling
""")

# ============================================================================
# RECOMMENDATION
# ============================================================================
print("\n" + "="*70)
print("RECOMMENDATION FOR YOUR DATA")
print("="*70)
print("""
Based on the analysis, I recommend a COMBINED APPROACH:

STEP 1: Coherent stacking (Option 2)
  - Stack 10-20 measurements coherently
  - Reduces noise by 10-13 dB
  - Slight loss of temporal resolution (acceptable for 1878 measurements)

STEP 2: SVD noise reduction (Option 4)  
  - Remove low-rank noise components
  - Keep singular values that capture layer structure

STEP 3: Depth-dependent gain (Option 1)
  - Compensate for absorption losses
  - Makes deep layers visually comparable to shallow

STEP 4: Coherence-based detection (Option 7)
  - Use coherence to identify which deep peaks are real layers
  - Weight by coherence in velocity estimation

This pipeline would likely reveal 10-20 additional reliable deep layers.

Shall I implement this processing pipeline?
""")
