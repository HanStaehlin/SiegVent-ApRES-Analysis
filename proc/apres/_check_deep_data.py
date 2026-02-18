#!/usr/bin/env python3
"""Quick check of deep region data characteristics."""
from scipy.io import loadmat
from scipy.ndimage import uniform_filter1d
import numpy as np

mat = loadmat('/Users/hannesstahlin/SiegVent2023-Geology/data/apres/ImageP2_python.mat')
print('Keys:', [k for k in mat.keys() if not k.startswith('_')])
rc = mat['Rcoarse'].flatten()
t = mat['TimeInDays'].flatten()
raw = mat['RawImageComplex']
print(f'Complex data: {raw.shape} ({raw.dtype})')
print(f'Rcoarse: {rc.shape}, range {rc[0]:.1f} to {rc[-1]:.1f} m')
print(f'TimeInDays: {t.shape}, range {t[0]:.2f} to {t[-1]:.2f} days')
lam = mat.get('lambdac', np.array([[0.5608]]))
lambdac = float(np.array(lam).flatten()[0])
print(f'lambdac = {lambdac:.4f} m')

# Check deep region
deep_mask = (rc >= 785) & (rc <= 1094)
deep = raw[deep_mask, :]
amp = np.abs(deep)
amp_db = 20 * np.log10(amp + 1e-30)
print(f'\nDeep region (785-1094m): {np.sum(deep_mask)} bins x {deep.shape[1]} times')
print(f'Amplitude (dB): mean={np.mean(amp_db):.1f}, max={np.max(amp_db):.1f}')

# Count bins with elevated amplitude segments
n_with_30 = 0
n_with_15 = 0
n_with_10 = 0
max_fracs = []
for i in range(deep.shape[0]):
    a = np.abs(deep[i, :])
    a_smooth = uniform_filter1d(a, size=15)
    thr = 1.3 * np.median(a_smooth)
    elev = a_smooth > thr
    frac = np.sum(elev) / len(elev)
    max_fracs.append(frac)
    # longest run
    maxrun = 0
    run = 0
    for e in elev:
        if e:
            run += 1
            maxrun = max(maxrun, run)
        else:
            run = 0
    if maxrun >= 30:
        n_with_30 += 1
    if maxrun >= 15:
        n_with_15 += 1
    if maxrun >= 10:
        n_with_10 += 1

n_deep = deep.shape[0]
print(f'\nAmplitude gating (amp_factor=1.3):')
print(f'  Bins with longest run >= 30 pts: {n_with_30}/{n_deep}')
print(f'  Bins with longest run >= 15 pts: {n_with_15}/{n_deep}')
print(f'  Bins with longest run >= 10 pts: {n_with_10}/{n_deep}')
print(f'  Mean elevated fraction: {np.mean(max_fracs):.3f}')

# Now try with SVD denoising
print('\n--- SVD denoising test ---')
U, S, Vh = np.linalg.svd(deep, full_matrices=False)
total_energy = np.sum(S**2)
for n_comp in [20, 50, 100, 200]:
    kept = np.sum(S[:n_comp]**2) / total_energy * 100
    # Reconstruct
    S_filt = np.zeros_like(S)
    S_filt[:n_comp] = S[:n_comp]
    deep_svd = U @ np.diag(S_filt) @ Vh
    
    n_with_30_svd = 0
    n_with_15_svd = 0
    n_with_10_svd = 0
    max_runs_svd = []
    for i in range(deep_svd.shape[0]):
        a = np.abs(deep_svd[i, :])
        a_smooth = uniform_filter1d(a, size=15)
        thr = 1.3 * np.median(a_smooth)
        elev = a_smooth > thr
        maxrun = 0
        run = 0
        for e in elev:
            if e:
                run += 1
                maxrun = max(maxrun, run)
            else:
                run = 0
        max_runs_svd.append(maxrun)
        if maxrun >= 30:
            n_with_30_svd += 1
        if maxrun >= 15:
            n_with_15_svd += 1
        if maxrun >= 10:
            n_with_10_svd += 1
    
    print(f'\n  SVD n_components={n_comp} ({kept:.1f}% energy):')
    print(f'    Bins with longest run >= 30 pts: {n_with_30_svd}/{n_deep}')
    print(f'    Bins with longest run >= 15 pts: {n_with_15_svd}/{n_deep}')
    print(f'    Bins with longest run >= 10 pts: {n_with_10_svd}/{n_deep}')
    print(f'    Mean max run: {np.mean(max_runs_svd):.1f}, median: {np.median(max_runs_svd):.0f}')

# Also test with lower amp_factor
print('\n--- Lower amplitude threshold test (factor=1.1) ---')
for n_comp in [0, 50, 100]:
    if n_comp == 0:
        data = deep
        label = 'Raw'
    else:
        S_filt = np.zeros_like(S)
        S_filt[:n_comp] = S[:n_comp]
        data = U @ np.diag(S_filt) @ Vh
        label = f'SVD-{n_comp}'
    
    n_with_30_low = 0
    for i in range(data.shape[0]):
        a = np.abs(data[i, :])
        a_smooth = uniform_filter1d(a, size=15)
        thr = 1.1 * np.median(a_smooth)
        elev = a_smooth > thr
        maxrun = 0
        run = 0
        for e in elev:
            if e:
                run += 1
                maxrun = max(maxrun, run)
            else:
                run = 0
        if maxrun >= 30:
            n_with_30_low += 1
    print(f'  {label}: bins with run>=30: {n_with_30_low}/{n_deep}')

print('\nDone.')
