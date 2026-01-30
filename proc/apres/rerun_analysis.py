#!/usr/bin/env python3
"""Quick re-run of the analysis with updated phase tracking."""

import sys
sys.path.insert(0, '.')
from layer_detection import load_apres_data, detect_layers, save_layers
from phase_tracking import load_layer_data, track_all_layers, save_phase_results
from velocity_profile import load_phase_data, calculate_velocity_profile, save_velocity_results
from pathlib import Path
import numpy as np

# Setup
data_path = 'data/apres/ImageP2_python.mat'
output_dir = 'output/apres/hybrid'
output_path = Path(output_dir)
output_path.mkdir(parents=True, exist_ok=True)

# Step 1: Layer Detection  
print('='*60)
print('STEP 1: LAYER DETECTION')
print('='*60)
data = load_apres_data(data_path)
layers = detect_layers(data['range_img'], data['Rcoarse'], min_depth=50, max_depth=1050, min_snr_db=10, min_persistence=0.5)
save_layers(layers, str(output_path / 'detected_layers'))
print(f'\n✓ Layer detection complete: {layers.n_layers} layers found')

# Step 2: Phase Tracking
print('\n' + '='*60)
print('STEP 2: PHASE TRACKING')
print('='*60)
layer_data, apres_data = load_layer_data(str(output_path / 'detected_layers'), data_path)
phase_result = track_all_layers(layer_data, apres_data)
save_phase_results(phase_result, str(output_path / 'phase_tracking'))
print(f'\n✓ Phase tracking complete for {phase_result.n_layers} layers')

# Step 3: Velocity Profile
print('\n' + '='*60)
print('STEP 3: VELOCITY PROFILE')  
print('='*60)
phase_data = load_phase_data(str(output_path / 'phase_tracking'))
velocity_result = calculate_velocity_profile(phase_data, r_sq_threshold=0.3, amp_threshold_db=-80)
save_velocity_results(velocity_result, str(output_path / 'velocity_profile'))

print(f'\n✓ Velocity profile complete')
print(f'\nSUMMARY:')
print(f'  Layers detected: {layers.n_layers}')
n_reliable = np.sum(velocity_result.r_squared > 0.3)
print(f'  Reliable layers (R² > 0.3): {n_reliable}')
if n_reliable > 0:
    reliable_mask = velocity_result.r_squared > 0.3
    print(f'  Velocity range: {velocity_result.velocities[reliable_mask].min():.2f} to {velocity_result.velocities[reliable_mask].max():.2f} m/yr')
