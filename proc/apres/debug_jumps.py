#!/usr/bin/env python3
"""Debug phase jumps in specific layers."""

import numpy as np
from scipy.io import loadmat

# Load data
phase = loadmat('output/apres/hybrid/phase_tracking.mat')
range_ts = np.squeeze(phase['range_timeseries'])
time_days = np.squeeze(phase['time_days'])
depths = np.squeeze(phase['layer_depths'])

# Check the remaining problematic layers (74, 78, 82)
for layer_idx in [74, 78, 82]:
    layer_data = range_ts[layer_idx, :]
    
    # Find single-step jumps > 15 cm
    diffs = np.diff(layer_data)
    large_steps = np.where(np.abs(diffs) > 0.15)[0]
    
    print(f'\nLayer {layer_idx} (depth {depths[layer_idx]:.1f} m):')
    print(f'  Total points: {len(layer_data)}')
    print(f'  Single steps > 15 cm: {len(large_steps)}')
    
    if len(large_steps) > 0:
        print('  Largest steps:')
        sorted_idx = np.argsort(np.abs(diffs[large_steps]))[::-1][:5]
        for i in sorted_idx:
            idx = large_steps[i]
            print(f'    Day {time_days[idx]:.1f} -> {time_days[idx+1]:.1f}: {diffs[idx]*100:.1f} cm')
    
    # Check 5-day window jumps
    print('  5-day window jumps > 20 cm:')
    count = 0
    for i in range(len(time_days) - 1):
        t_start = time_days[i]
        t_end = t_start + 5.0
        in_window = (time_days > t_start) & (time_days <= t_end)
        if not np.any(in_window):
            continue
        j = np.where(in_window)[0][-1]
        delta = abs(layer_data[j] - layer_data[i])
        if delta > 0.20 and count < 3:
            print(f'    Day {time_days[i]:.1f} -> {time_days[j]:.1f}: {delta*100:.1f} cm')
            count += 1
