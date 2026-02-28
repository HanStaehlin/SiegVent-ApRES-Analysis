import numpy as np
from scipy.io import loadmat
import zarr

mat = loadmat('data/apres/ImageP2_python.mat')
mat_img = mat['RawImage']

zm = zarr.open('data/apres/ImageP2_python.zarr', mode='r')
z_img = zm['range_img'][:]

diff = np.max(np.abs(mat_img - z_img))
print(f"Max absolute difference: {diff}")
print(f"Max value MAT: {np.max(mat_img)}, ZARR: {np.max(z_img)}")
print(f"Min value MAT (non-zero): {np.min(mat_img[mat_img > 0])}, ZARR: {np.min(z_img[z_img > 0])}")

# Check dB representation
db_mat = 10 * np.log10(mat_img**2 + 1e-30)
db_zarr = 10 * np.log10(z_img**2 + 1e-30)

print(f"Mean dB MAT: {np.mean(db_mat)}, ZARR: {np.mean(db_zarr)}")
print(f"Max/Min dB MAT: {np.max(db_mat)} / {np.min(db_mat)}")
print(f"Max/Min dB ZARR: {np.max(db_zarr)} / {np.min(db_zarr)}")
