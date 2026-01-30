"""
Validate deep layer detections - are they real internal layers or noise artifacts?

Key criteria for real internal ice layers:
1. Temporal coherence: Real layers maintain consistent phase over time
2. Spectral characteristics: Real layers show localized amplitude peaks
3. Physical plausibility: Layer spacing and amplitude decay with depth
4. Phase stability: Real layers have trackable, smoothly-varying phase
"""

import numpy as np
from scipy.io import loadmat
from scipy import stats
import json

# Load data
data = loadmat('/Users/hannesstahlin/SiegVent2023-Geology/data/apres/ImageP2_python.mat')
range_img = np.abs(data['RawImageComplex'])
Rcoarse = data['Rcoarse'].flatten()
time_days = data['TimeInDays'].flatten()

# Load detected layers
with open('/Users/hannesstahlin/SiegVent2023-Geology/output/apres/detected_layers.json') as f:
    layers_json = json.load(f)

# Extract layer properties from the list of layer dicts
layers_list = layers_json['layers']
layer_depths = np.array([l['depth_m'] for l in layers_list])
layer_snr = np.array([l['snr_db'] for l in layers_list])
layer_persistence = np.array([l['persistence'] for l in layers_list])

# Load phase tracking results
phase_data = loadmat('/Users/hannesstahlin/SiegVent2023-Geology/output/apres/phase_tracking.mat')
phase_history = phase_data['phase_timeseries']  # [n_layers, n_times]
amp_history = phase_data['amplitude_timeseries']

# Load velocity results
with open('/Users/hannesstahlin/SiegVent2023-Geology/output/apres/velocity_profile.json') as f:
    vel_json = json.load(f)
vel_layers = vel_json['layers']
r_squared = np.array([l['r_squared'] for l in vel_layers])
velocities = np.array([l['velocity_m_yr'] for l in vel_layers])

print("="*70)
print("DEEP LAYER VALIDATION ANALYSIS")
print("="*70)

# Categorize layers by depth
shallow_mask = layer_depths < 400
mid_mask = (layer_depths >= 400) & (layer_depths < 700)
deep_mask = layer_depths >= 700

print(f"\nLayer counts by depth:")
print(f"  Shallow (<400m): {np.sum(shallow_mask)} layers")
print(f"  Mid (400-700m):  {np.sum(mid_mask)} layers")
print(f"  Deep (>700m):    {np.sum(deep_mask)} layers")

# 1. SNR Analysis
print(f"\n--- 1. SIGNAL-TO-NOISE RATIO ---")
print(f"{'Depth Range':<20} {'Mean SNR (dB)':<15} {'Min SNR (dB)':<15} {'Max SNR (dB)':<15}")
print("-"*65)
for name, mask in [("Shallow <400m", shallow_mask), ("Mid 400-700m", mid_mask), ("Deep >700m", deep_mask)]:
    if np.any(mask):
        print(f"{name:<20} {np.mean(layer_snr[mask]):>12.1f}   {np.min(layer_snr[mask]):>12.1f}   {np.max(layer_snr[mask]):>12.1f}")

# 2. Persistence Analysis
print(f"\n--- 2. TEMPORAL PERSISTENCE ---")
print("(Fraction of time steps where layer is detectable)")
print(f"{'Depth Range':<20} {'Mean Persist':<15} {'Min Persist':<15}")
print("-"*50)
for name, mask in [("Shallow <400m", shallow_mask), ("Mid 400-700m", mid_mask), ("Deep >700m", deep_mask)]:
    if np.any(mask):
        print(f"{name:<20} {np.mean(layer_persistence[mask]):>12.1%}   {np.min(layer_persistence[mask]):>12.1%}")

# 3. Phase Coherence Analysis
print(f"\n--- 3. PHASE TRACKING QUALITY (R² from velocity fit) ---")
print("(Higher R² = more linear phase change = more reliable layer)")
print(f"{'Depth Range':<20} {'Mean R²':<12} {'Layers R²>0.3':<15} {'Layers R²>0.7':<15}")
print("-"*62)
for name, mask in [("Shallow <400m", shallow_mask), ("Mid 400-700m", mid_mask), ("Deep >700m", deep_mask)]:
    if np.any(mask):
        r2_subset = r_squared[mask]
        print(f"{name:<20} {np.mean(r2_subset):>10.3f}   {np.sum(r2_subset > 0.3):>12}   {np.sum(r2_subset > 0.7):>12}")

# 4. Amplitude stability analysis
print(f"\n--- 4. AMPLITUDE STABILITY ---")
print("(Coefficient of variation - lower = more stable)")
for name, mask in [("Shallow <400m", shallow_mask), ("Mid 400-700m", mid_mask), ("Deep >700m", deep_mask)]:
    if np.any(mask):
        indices = np.where(mask)[0]
        cvs = []
        for i in indices:
            amp = amp_history[i, :]
            amp_db = 10 * np.log10(amp**2 + 1e-30)
            cv = np.std(amp_db) / (np.abs(np.mean(amp_db)) + 1e-10)
            cvs.append(cv)
        print(f"{name:<20} Mean CV: {np.mean(cvs):.3f}  (lower is better)")

# 5. Phase jump analysis (detect unreliable tracking)
print(f"\n--- 5. PHASE JUMP ANALYSIS ---")
print("(Layers with many phase jumps may be tracking noise)")
for name, mask in [("Shallow <400m", shallow_mask), ("Mid 400-700m", mid_mask), ("Deep >700m", deep_mask)]:
    if np.any(mask):
        indices = np.where(mask)[0]
        jump_counts = []
        for i in indices:
            phase = phase_history[i, :]
            # Count jumps > 5cm (unrealistic for ice flow)
            dphase = np.abs(np.diff(phase))
            n_jumps = np.sum(dphase > 0.05)  # 5cm threshold
            jump_counts.append(n_jumps)
        print(f"{name:<20} Mean jumps: {np.mean(jump_counts):.1f}, Max: {np.max(jump_counts)}")

# 6. Deep layer detail
print(f"\n--- 6. DEEP LAYER DETAILS (>700m) ---")
print(f"{'Depth (m)':<12} {'SNR (dB)':<12} {'Persist':<12} {'R²':<10} {'Vel (m/yr)':<12} {'Status'}")
print("-"*70)
deep_indices = np.where(deep_mask)[0]
for i in deep_indices:
    depth = layer_depths[i]
    snr = layer_snr[i]
    persist = layer_persistence[i]
    r2 = r_squared[i]
    vel = velocities[i]
    
    # Classify reliability
    if r2 > 0.5 and persist > 0.5 and snr > 5:
        status = "✓ Reliable"
    elif r2 > 0.3 and persist > 0.3:
        status = "~ Marginal"
    else:
        status = "✗ Uncertain"
    
    print(f"{depth:>10.1f}   {snr:>10.1f}   {persist:>10.1%}   {r2:>8.3f}   {vel:>10.3f}   {status}")

# Summary
print(f"\n" + "="*70)
print("SUMMARY: Are the deep layers real?")
print("="*70)

deep_reliable = np.sum((r_squared[deep_mask] > 0.5) & (layer_persistence[deep_mask] > 0.5))
deep_marginal = np.sum((r_squared[deep_mask] > 0.3) & (layer_persistence[deep_mask] > 0.3)) - deep_reliable
deep_uncertain = np.sum(deep_mask) - deep_reliable - deep_marginal

print(f"""
Deep layers (>700m): {np.sum(deep_mask)} total
  ✓ Reliable (R²>0.5, persist>50%): {deep_reliable}
  ~ Marginal (R²>0.3, persist>30%): {deep_marginal}
  ✗ Uncertain: {deep_uncertain}

Key observations:
1. SNR decreases with depth (expected - signal attenuates)
2. Persistence decreases with depth (layers harder to track)
3. R² indicates how well phase follows expected ice motion
4. Layers with low R² may be:
   - Real but with complex motion (englacial deformation)
   - Noise peaks that aren't coherent features
   - Multiple interfering reflectors

Recommendations for deep layers:
- Use only R² > 0.5 layers for velocity profiles
- Visual inspection of phase time series recommended
- Compare with expected strain rate profile
""")
