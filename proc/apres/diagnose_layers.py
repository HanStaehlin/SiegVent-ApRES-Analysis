"""Diagnose why layers aren't being detected beyond 712m."""
import numpy as np
from scipy.io import loadmat
from scipy import signal, ndimage

data = loadmat('/Users/hannesstahlin/SiegVent2023-Geology/data/apres/ImageP2_python.mat')
range_img = np.abs(data['RawImageComplex'])
Rcoarse = data['Rcoarse'].flatten()

min_depth, max_depth = 50, 1050
depth_mask = (Rcoarse >= min_depth) & (Rcoarse <= max_depth)

mean_profile = np.mean(range_img, axis=1)
mean_profile = ndimage.uniform_filter1d(mean_profile, 3)
mean_profile_db = 10 * np.log10(mean_profile**2 + 1e-30)

# Original: global noise floor from the whole region
noise_floor_global = np.percentile(mean_profile_db[depth_mask], 10)
print(f'Global noise floor (10th percentile): {noise_floor_global:.1f} dB')

# Find peaks with min_snr=3 dB
depth_indices = np.where(depth_mask)[0]
peaks, props = signal.find_peaks(
    mean_profile_db[depth_mask],
    height=noise_floor_global + 3,
    prominence=3,
    distance=95,  # ~5m separation
)
peak_indices = depth_indices[peaks]
peak_depths = Rcoarse[peak_indices]
print(f'Peaks found: {len(peaks)}')
print(f'Max peak depth: {peak_depths.max():.1f} m')

# Check persistence for deep layer candidates
print('\nPersistence check for deep layers:')
local_snr_thresh = noise_floor_global + 3 * 0.7
print(f'  Local persistence threshold: {local_snr_thresh:.1f} dB')

for test_depth in [700, 800, 900, 1000]:
    idx = np.argmin(np.abs(Rcoarse - test_depth))
    
    # Count how many time steps exceed threshold
    window = 38  # ~2m window
    visible_count = 0
    for t in range(range_img.shape[1]):
        local_max = np.max(10*np.log10(range_img[idx-window:idx+window, t]**2 + 1e-30))
        if local_max > local_snr_thresh:
            visible_count += 1
    persistence = visible_count / range_img.shape[1]
    print(f'  Depth {test_depth}m: persistence={persistence:.1%}, mean={mean_profile_db[idx]:.1f} dB')

# The issue: peaks exist at depth but can't pass persistence with global threshold
print('\n--- The problem ---')
print('Peaks in 700-1050m region are filtered out because the persistence check')
print('uses a global noise floor threshold that is too high for weaker deep signals.')
