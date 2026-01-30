"""
Diagnose why enhanced data has WORSE deep layer quality.

Hypothesis: The coherent stacking may be causing issues if layers move
more than λ/4 between consecutive measurements, causing phase decorrelation.
"""

import numpy as np
from scipy.io import loadmat

print("="*70)
print("DIAGNOSING ENHANCED DATA ISSUES")
print("="*70)

# Load original complex data
orig = loadmat('/Users/hannesstahlin/SiegVent2023-Geology/data/apres/ImageP2_python.mat')
orig_complex = orig['RawImageComplex']
orig_time = orig['TimeInDays'].flatten()
Rcoarse = orig['Rcoarse'].flatten()

# Load enhanced
enh = loadmat('/Users/hannesstahlin/SiegVent2023-Geology/output/apres/ImageP2_enhanced.mat')
enh_complex = enh['RawImageComplex']
enh_time = enh['TimeInDays'].flatten()

print(f"\nOriginal: {orig_complex.shape[1]} time points, dt = {np.mean(np.diff(orig_time)):.3f} days")
print(f"Enhanced: {enh_complex.shape[1]} time points, dt = {np.mean(np.diff(enh_time)):.3f} days")

# Key insight: stacking 20 measurements spans how much time?
stack_size = 20
time_span_per_stack = stack_size * np.mean(np.diff(orig_time))
print(f"\nEach stack spans: {time_span_per_stack:.2f} days")

# At what velocity would a layer move λ/4 in this time?
lambdac = 0.56  # wavelength in ice (m)
critical_velocity = (lambdac / 4) / (time_span_per_stack * 365.25)  # m/year
print(f"Critical velocity for phase decorrelation: {critical_velocity:.2f} m/year")

# What are typical layer velocities?
from scipy.io import loadmat
phase_orig = loadmat('/Users/hannesstahlin/SiegVent2023-Geology/output/apres/phase_tracking.mat')
velocities = []
for i in range(phase_orig['phase_timeseries'].shape[0]):
    phase = phase_orig['phase_timeseries'][i, :]
    time = phase_orig['time_days'].flatten()
    # Linear fit
    valid = ~np.isnan(phase)
    if np.sum(valid) > 10:
        coeffs = np.polyfit(time[valid], phase[valid], 1)
        vel = coeffs[0] * 365.25  # m/year
        velocities.append(vel)

velocities = np.array(velocities)
print(f"\nTypical layer velocities: {np.percentile(velocities, [10, 50, 90])}")
print(f"Max velocity: {np.max(np.abs(velocities)):.2f} m/year")

# Compare phase consistency
print("\n--- PHASE CONSISTENCY ANALYSIS ---")

# For a deep layer, compare phase evolution in original vs enhanced
test_depth = 750
idx = np.argmin(np.abs(Rcoarse - test_depth))

# Original: extract phase at this depth
orig_phase = np.angle(orig_complex[idx, :])
orig_phase_unwrapped = np.unwrap(orig_phase)
orig_range_change = orig_phase_unwrapped * lambdac / (4 * np.pi)

# Enhanced: extract phase at this depth
enh_phase = np.angle(enh_complex[idx, :])
enh_phase_unwrapped = np.unwrap(enh_phase)
enh_range_change = enh_phase_unwrapped * lambdac / (4 * np.pi)

print(f"\nAt depth {test_depth}m:")
print(f"  Original phase std: {np.std(np.diff(orig_range_change))*100:.2f} cm per step")
print(f"  Enhanced phase std: {np.std(np.diff(enh_range_change))*100:.2f} cm per step")

# The problem: stacking decorrelates moving targets!
print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)
print(f"""
PROBLEM IDENTIFIED: Coherent stacking is HARMING deep layer phase tracking.

Reason: Stacking {stack_size} measurements that span {time_span_per_stack:.1f} days
causes phase averaging that decorrelates moving layers.

For a layer moving at 0.3 m/year:
  - Movement in {time_span_per_stack:.1f} days = {0.3 * time_span_per_stack/365.25 * 100:.1f} cm
  - This is {0.3 * time_span_per_stack/365.25 / (lambdac/4) * 100:.0f}% of λ/4

When the layer moves >λ/4 during the stack window, the complex average
partially cancels, reducing coherence and introducing phase errors.

SOLUTION OPTIONS:
1. Use smaller stack size (5-10 instead of 20)
2. Phase-align before stacking (track layer motion)
3. Use incoherent stacking for amplitude, but track phase on original data
4. Skip stacking entirely, use other enhancement methods

Let me test with smaller stack size...
""")
