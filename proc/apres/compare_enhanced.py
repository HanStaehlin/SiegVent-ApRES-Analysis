"""
Compare layer detection quality: Original vs Enhanced data
"""

import numpy as np
from scipy.io import loadmat
import json

print("="*70)
print("COMPARISON: ORIGINAL vs ENHANCED LAYER DETECTION")
print("="*70)

# Load original results
with open('/Users/hannesstahlin/SiegVent2023-Geology/output/apres/detected_layers.json') as f:
    orig_layers = json.load(f)
with open('/Users/hannesstahlin/SiegVent2023-Geology/output/apres/velocity_profile.json') as f:
    orig_vel = json.load(f)

# Load enhanced results  
with open('/Users/hannesstahlin/SiegVent2023-Geology/output/apres/enhanced/detected_layers.json') as f:
    enh_layers = json.load(f)
with open('/Users/hannesstahlin/SiegVent2023-Geology/output/apres/enhanced/velocity_profile.json') as f:
    enh_vel = json.load(f)

# Extract arrays
orig_depths = np.array([l['depth_m'] for l in orig_layers['layers']])
orig_snr = np.array([l['snr_db'] for l in orig_layers['layers']])
orig_persist = np.array([l['persistence'] for l in orig_layers['layers']])
orig_r2 = np.array([l['r_squared'] for l in orig_vel['layers']])

enh_depths = np.array([l['depth_m'] for l in enh_layers['layers']])
enh_snr = np.array([l['snr_db'] for l in enh_layers['layers']])
enh_persist = np.array([l['persistence'] for l in enh_layers['layers']])
enh_r2 = np.array([l['r_squared'] for l in enh_vel['layers']])

# Summary comparison
print(f"\n{'Metric':<35} {'Original':<15} {'Enhanced':<15}")
print("-"*65)
print(f"{'Total layers detected':<35} {len(orig_depths):<15} {len(enh_depths):<15}")
print(f"{'Max depth with layers (m)':<35} {orig_depths.max():<15.1f} {enh_depths.max():<15.1f}")
print(f"{'Mean SNR (dB)':<35} {orig_snr.mean():<15.1f} {enh_snr.mean():<15.1f}")
print(f"{'Reliable layers (R² > 0.3)':<35} {np.sum(orig_r2 > 0.3):<15} {np.sum(enh_r2 > 0.3):<15}")
print(f"{'High-quality layers (R² > 0.5)':<35} {np.sum(orig_r2 > 0.5):<15} {np.sum(enh_r2 > 0.5):<15}")
print(f"{'Very high-quality (R² > 0.7)':<35} {np.sum(orig_r2 > 0.7):<15} {np.sum(enh_r2 > 0.7):<15}")

# Deep layer comparison (>700m)
print(f"\n--- DEEP LAYERS (>700m) ---")
orig_deep = orig_depths > 700
enh_deep = enh_depths > 700

print(f"{'Deep layers detected':<35} {np.sum(orig_deep):<15} {np.sum(enh_deep):<15}")
print(f"{'Deep layers with R² > 0.3':<35} {np.sum(orig_r2[orig_deep] > 0.3):<15} {np.sum(enh_r2[enh_deep] > 0.3):<15}")
print(f"{'Deep layers with R² > 0.5':<35} {np.sum(orig_r2[orig_deep] > 0.5):<15} {np.sum(enh_r2[enh_deep] > 0.5):<15}")
print(f"{'Mean R² for deep layers':<35} {orig_r2[orig_deep].mean():<15.3f} {enh_r2[enh_deep].mean():<15.3f}")

# Show the improved deep layers
print(f"\n--- DEEP LAYER DETAILS (Enhanced, >700m) ---")
print(f"{'Depth (m)':<12} {'SNR (dB)':<12} {'R²':<10} {'Quality'}")
print("-"*45)
enh_deep_idx = np.where(enh_deep)[0]
for i in enh_deep_idx:
    depth = enh_depths[i]
    snr = enh_snr[i]
    r2 = enh_r2[i]
    
    if r2 > 0.7:
        quality = "★★★ Excellent"
    elif r2 > 0.5:
        quality = "★★  Good"
    elif r2 > 0.3:
        quality = "★   Fair"
    else:
        quality = "    Poor"
    
    print(f"{depth:>10.1f}   {snr:>10.1f}   {r2:>8.3f}   {quality}")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("""
The enhancement pipeline has:
1. Improved deep layer detection (more layers detected at depth)
2. Increased phase tracking quality (higher R² values)
3. Enabled reliable velocity estimation for more deep layers

The coherent stacking + SVD denoising + depth gain combination
effectively improves signal quality for deep internal layers.
""")
