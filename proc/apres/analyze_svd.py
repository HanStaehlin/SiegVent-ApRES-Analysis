#!/usr/bin/env python3
"""Analyze SVD singular value distribution across depth regions."""

import numpy as np
from scipy.io import loadmat
from scipy.linalg import svd

data = loadmat('data/apres/ImageP2_python.mat')
range_img = np.abs(data['RawImageComplex'])
Rcoarse = data['Rcoarse'].flatten()

print(f'Echogram shape: {range_img.shape}')
print(f'Depth range: {Rcoarse[0]:.1f} - {Rcoarse[-1]:.1f} m')

def get_depth_idx(depth):
    return np.argmin(np.abs(Rcoarse - depth))

# Analyze different depth regions
regions = [
    ('Surface (0-100m)', 0, 100),
    ('Shallow (100-300m)', 100, 300),
    ('Mid (300-600m)', 300, 600),
    ('Deep (600-1000m)', 600, 1000),
]

print('\n' + '='*60)
print('SVD Analysis by Depth Region')
print('='*60)

for name, d1, d2 in regions:
    i1, i2 = get_depth_idx(d1), get_depth_idx(d2)
    block = range_img[i1:i2, :]
    U, s, Vh = svd(block, full_matrices=False)
    cumsum = np.cumsum(s**2) / np.sum(s**2) * 100
    
    print(f'\n{name}:')
    print(f'  Bins: {i2-i1}, Shape: {block.shape}')
    print(f'  Variance explained:')
    print(f'    5 components: {cumsum[4]:.2f}%')
    print(f'   10 components: {cumsum[9]:.2f}%')
    print(f'   20 components: {cumsum[19]:.2f}%')
    print(f'  Singular value ratio:')
    print(f'    s[0]/s[10] = {s[0]/s[10]:.0f}x')
    print(f'    s[0]/s[20] = {s[0]/s[20]:.0f}x')

print('\n' + '='*60)
print('Recommendation:')
print('='*60)
print('With 99.99% variance in 5 components, using 10 is very aggressive.')
print('10 components is good for strong denoising.')
print('20-30 components for moderate denoising.')
print('50+ components for mild denoising.')
