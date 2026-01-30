"""
Hybrid approach: Use enhanced data for layer detection, 
but original data for phase tracking.

This gives us:
- Better SNR for detecting deep layers (from enhancement)
- Full temporal resolution for phase tracking (from original)
"""

import numpy as np
from scipy.io import loadmat, savemat
import json
from pathlib import Path
import sys

sys.path.insert(0, '/Users/hannesstahlin/SiegVent2023-Geology/proc/apres')

from layer_detection import detect_layers, save_layers, visualize_layers, load_apres_data
from phase_tracking import track_all_layers_smooth, visualize_phase_tracking, save_phase_results, PhaseTrackingResult
from velocity_profile import calculate_velocity_profile, visualize_velocity_profile, save_velocity_results

print("="*70)
print("HYBRID APPROACH: Enhanced Detection + Original Phase Tracking")
print("="*70)

output_dir = Path('/Users/hannesstahlin/SiegVent2023-Geology/output/apres/hybrid')
output_dir.mkdir(parents=True, exist_ok=True)

# Step 1: Detect layers on ENHANCED data (better SNR)
print("\n[1/3] Layer detection on ENHANCED data...")
enhanced_path = '/Users/hannesstahlin/SiegVent2023-Geology/output/apres/ImageP2_enhanced.mat'
enhanced_data = load_apres_data(enhanced_path)

layers = detect_layers(
    enhanced_data['range_img'],
    enhanced_data['Rcoarse'],
    min_depth=50,
    max_depth=1050,
    min_snr_db=5,
    min_persistence=0.5,
)

print(f"  Detected {layers.n_layers} layers from enhanced data")
print(f"  Depth range: {layers.layer_depths.min():.1f} - {layers.layer_depths.max():.1f} m")

# Save layer results
save_layers(layers, str(output_dir / 'detected_layers'))

# Step 2: Phase tracking on ORIGINAL data (full temporal resolution)
print("\n[2/3] Phase tracking on ORIGINAL data...")
original_path = '/Users/hannesstahlin/SiegVent2023-Geology/data/apres/ImageP2_python.mat'
original_data = load_apres_data(original_path)

# Use complex data for phase tracking
if 'RawImageComplex' in loadmat(original_path):
    mat = loadmat(original_path)
    complex_img = mat['RawImageComplex']
    rfine = mat['RfineBarTime']
else:
    raise ValueError("Need RawImageComplex for phase tracking")

# Prepare data for phase tracking
apres_data = {
    'range_img': np.abs(complex_img),
    'complex_img': complex_img,
    'rfine': rfine,
    'Rcoarse': original_data['Rcoarse'],
    'time_days': original_data['time_days'],
}

# Convert layers result to dict format expected by track_all_layers_smooth
layers_dict = {
    'n_layers': layers.n_layers,
    'depths': layers.layer_depths,
    'indices': layers.layer_indices,
    'amplitudes': layers.layer_amplitudes,
    'snr': layers.layer_snr,
    'persistence': layers.layer_persistence,
}

# Track phases
phase_results = track_all_layers_smooth(
    layers=layers_dict,
    data=apres_data,
)

print(f"  Tracked {phase_results.n_layers} layers through {len(apres_data['time_days'])} time steps")

# Save phase results - pass the PhaseTrackingResult object directly
save_phase_results(phase_results, str(output_dir / 'phase_tracking'))

# Visualize
fig = visualize_phase_tracking(phase_results, output_file=str(output_dir / 'phase_tracking.html'))

# Step 3: Calculate velocities
print("\n[3/3] Calculating velocity profile...")

# Convert PhaseTrackingResult to dict for velocity calculation
phase_dict = {
    'n_layers': phase_results.n_layers,
    'layer_depths': phase_results.layer_depths,
    'time_days': phase_results.time_days,
    'phase_timeseries': phase_results.phase_timeseries,
    'range_timeseries': phase_results.range_timeseries,
    'amplitude_timeseries': phase_results.amplitude_timeseries,
    'lambdac': phase_results.lambdac,
}

velocities = calculate_velocity_profile(
    phase_dict,
    r_sq_threshold=0.3,
    amp_threshold_db=-80,
)

save_velocity_results(velocities, str(output_dir / 'velocity_profile'))
fig = visualize_velocity_profile(velocities, output_file=str(output_dir / 'velocity_profile.html'))

# Summary - load from saved JSON for comparison format
with open(str(output_dir / 'velocity_profile.json')) as f:
    hybrid_vel = json.load(f)

n_reliable = sum(1 for l in hybrid_vel['layers'] if l['reliable'])
print(f"\n  Total layers: {len(hybrid_vel['layers'])}")
print(f"  Reliable (R² > 0.3): {n_reliable}")

# Compare with pure original and pure enhanced approaches
print("\n" + "="*70)
print("COMPARISON SUMMARY")
print("="*70)

# Load other results for comparison
with open('/Users/hannesstahlin/SiegVent2023-Geology/output/apres/velocity_profile.json') as f:
    orig_vel = json.load(f)
with open('/Users/hannesstahlin/SiegVent2023-Geology/output/apres/enhanced/velocity_profile.json') as f:
    enh_vel = json.load(f)

orig_r2 = np.array([l['r_squared'] for l in orig_vel['layers']])
enh_r2 = np.array([l['r_squared'] for l in enh_vel['layers']])
hybrid_r2 = np.array([l['r_squared'] for l in hybrid_vel['layers']])

orig_depths = np.array([l['depth_m'] for l in orig_vel['layers']])
enh_depths = np.array([l['depth_m'] for l in enh_vel['layers']])
hybrid_depths = np.array([l['depth_m'] for l in hybrid_vel['layers']])

print(f"\n{'Metric':<35} {'Original':<12} {'Enhanced':<12} {'Hybrid':<12}")
print("-"*70)
print(f"{'Total layers':<35} {len(orig_r2):<12} {len(enh_r2):<12} {len(hybrid_r2):<12}")
print(f"{'Max depth (m)':<35} {orig_depths.max():<12.1f} {enh_depths.max():<12.1f} {hybrid_depths.max():<12.1f}")
print(f"{'Layers with R² > 0.3':<35} {np.sum(orig_r2>0.3):<12} {np.sum(enh_r2>0.3):<12} {np.sum(hybrid_r2>0.3):<12}")
print(f"{'Layers with R² > 0.5':<35} {np.sum(orig_r2>0.5):<12} {np.sum(enh_r2>0.5):<12} {np.sum(hybrid_r2>0.5):<12}")
print(f"{'Layers with R² > 0.7':<35} {np.sum(orig_r2>0.7):<12} {np.sum(enh_r2>0.7):<12} {np.sum(hybrid_r2>0.7):<12}")

# Deep layers
orig_deep = orig_depths > 700
enh_deep = enh_depths > 700
hybrid_deep = hybrid_depths > 700

print(f"\n--- DEEP LAYERS (>700m) ---")
print(f"{'Deep layers total':<35} {np.sum(orig_deep):<12} {np.sum(enh_deep):<12} {np.sum(hybrid_deep):<12}")
print(f"{'Deep layers R² > 0.3':<35} {np.sum(orig_r2[orig_deep]>0.3):<12} {np.sum(enh_r2[enh_deep]>0.3):<12} {np.sum(hybrid_r2[hybrid_deep]>0.3):<12}")
print(f"{'Deep layers R² > 0.5':<35} {np.sum(orig_r2[orig_deep]>0.5):<12} {np.sum(enh_r2[enh_deep]>0.5):<12} {np.sum(hybrid_r2[hybrid_deep]>0.5):<12}")
