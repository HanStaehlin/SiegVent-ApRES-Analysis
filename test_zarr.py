import zarr
import numpy as np

path = 'test_v3.zarr'
root = zarr.open_group(path, mode='w')
z_img = root.create_array('range_img', shape=(1000, 100), chunks=(500, 100), dtype='float32')
z_img[:] = np.random.rand(1000, 100).astype(np.float32)
print("Keys:", list(root.group_keys()), list(root.array_keys()))
