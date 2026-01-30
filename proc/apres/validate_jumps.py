#!/usr/bin/env python3
"""Validate that there are no remaining phase jumps."""

import numpy as np
from scipy.io import loadmat

# Load phase tracking results
phase = loadmat('output/apres/hybrid/phase_tracking.mat')
range_ts = np.squeeze(phase['range_timeseries'])  # [n_layers, n_times]
time_days = np.squeeze(phase['time_days'])
depths = np.squeeze(phase['layer_depths'])

print(f'Loaded {range_ts.shape[0]} layers, {range_ts.shape[1]} time points')
print(f'Time span: {time_days[0]:.1f} to {time_days[-1]:.1f} days')
print()

# Check for phase jumps: any change > 20 cm within 5 days
jump_threshold_m = 0.20
window_days = 5.0

layers_with_jumps = []
total_jumps = 0

for layer_idx in range(range_ts.shape[0]):
    layer_data = range_ts[layer_idx, :]
    layer_jumps = []
    
    for i in range(len(time_days) - 1):
        # Find points within 5 days
        t_start = time_days[i]
        t_end = t_start + window_days
        in_window = (time_days > t_start) & (time_days <= t_end)
        
        if not np.any(in_window):
            continue
        
        # Check for jumps
        window_indices = np.where(in_window)[0]
        for j in window_indices:
            delta = abs(layer_data[j] - layer_data[i])
            if delta > jump_threshold_m:
                layer_jumps.append((i, j, time_days[i], time_days[j], delta))
    
    if layer_jumps:
        layers_with_jumps.append((layer_idx, layer_jumps))
        total_jumps += len(layer_jumps)

print(f'=== PHASE JUMP VALIDATION ===')
print(f'Threshold: {jump_threshold_m*100:.0f} cm within {window_days:.0f} days')
print(f'Layers with jumps: {len(layers_with_jumps)}/{range_ts.shape[0]}')
print(f'Total jump instances: {total_jumps}')
print()

# Show details for layers with jumps
if layers_with_jumps:
    print('Layers with remaining jumps:')
    for layer_idx, jumps in layers_with_jumps[:15]:  # Show first 15
        print(f'  Layer {layer_idx} (depth {depths[layer_idx]:.1f} m): {len(jumps)} jumps')
        # Show first few jumps
        for i, j, t1, t2, delta in jumps[:3]:
            print(f'    Day {t1:.1f} -> {t2:.1f}: {delta*100:.1f} cm')
    if len(layers_with_jumps) > 15:
        print(f'  ... and {len(layers_with_jumps) - 15} more layers')
else:
    print('✓ No phase jumps detected!')
