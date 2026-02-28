import numpy as np
from scipy.io import loadmat

mat = loadmat('data/apres/ImageP2_python.mat')
Rcoarse = mat['Rcoarse'].flatten()
time_days = mat['TimeInDays'].flatten()
t_yr = (time_days - time_days[0]) / 365.25
raw_complex = np.array(mat['RawImageComplex'], dtype=np.complex64)
depth = 500.0
window_size = 20.0
lambdac = 0.5608

depth_mask = (Rcoarse >= depth - window_size/2) & (Rcoarse <= depth + window_size/2)
S_win = raw_complex[depth_mask, :]
U, sigma, Vh = np.linalg.svd(S_win, full_matrices=False)
# keeping k=3
S_denoised = (U[:, :3] * sigma[:3]) @ Vh[:3, :]

# Method 1: Phase-Slope
velocities = []
for i in range(S_denoised.shape[0]):
    z = S_denoised[i, :]
    phase = np.unwrap(np.angle(z))
    slope = np.polyfit(t_yr, phase, 1)[0]
    v = slope * lambdac / (4 * np.pi)
    velocities.append(v)

print("Median Phase-Slope v:", np.median(velocities))
print("Mean Phase-Slope v:", np.mean(velocities))

# Method 2: Radon
stack_map = []
vs = np.linspace(0, 1, 500)
for v in vs:
    expected_phase = (4 * np.pi * v * t_yr) / lambdac # shifted t_yr!
    derotated = S_denoised * np.exp(-1j * expected_phase)
    stack_map.append(np.sum(np.abs(np.mean(derotated, axis=1))))

best_v = vs[np.argmax(stack_map)]
print("Radon peak v:", best_v)

