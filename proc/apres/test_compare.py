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
z_win = Rcoarse[depth_mask]
S_win = raw_complex[depth_mask, :]
U, sigma, Vh = np.linalg.svd(S_win, full_matrices=False)

for k in range(3):
    phase = np.unwrap(np.angle(Vh[k, :]))
    slope = np.polyfit(t_yr, phase, 1)[0]
    v = slope * lambdac / (4 * np.pi)
    print(f"SVD Vector {k} velocity: {v:.4f} m/yr")

S_denoised = (U[:, :3] * sigma[:3]) @ Vh[:3, :]

vs = np.linspace(0, 1, 500)
for k in range(3):
    map_k = []
    for v in vs:
        expected_phase = (4 * np.pi * v * t_yr) / lambdac
        derotated = Vh[k, :] * np.exp(-1j * expected_phase)
        map_k.append(np.abs(np.mean(derotated)))
    print(f"Radon peak on Vh[{k}]: {vs[np.argmax(map_k)]:.4f} m/yr")

stack_map = []
for v in vs:
    expected_phase = (4 * np.pi * v * t_yr) / lambdac
    derotated = S_denoised * np.exp(-1j * expected_phase)
    stack_map.append(np.sum(np.abs(np.mean(derotated, axis=1))))

best_v = vs[np.argmax(stack_map)]
print(f"Radon peak v: {best_v:.4f} m/yr")
