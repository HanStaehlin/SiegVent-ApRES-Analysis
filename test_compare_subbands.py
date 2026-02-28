import numpy as np
import matplotlib.pyplot as plt
import zarr

def load_trace(subband):
    store = zarr.open(f'data/apres/ImageP2_python_{subband}.zarr', mode='r')
    img = store['range_img'][:, 0]
    depths = store['Rcoarse'][:]
    return depths, img

d_full, i_full = load_trace('full')
d_low, i_low = load_trace('low')
d_high, i_high = load_trace('high')

d_min, d_max = 200, 250
mask = (d_full >= d_min) & (d_full <= d_max)

plt.figure(figsize=(10, 6))
plt.plot(d_full[mask], 10*np.log10(i_full[mask]**2 + 1e-30), label='Full', alpha=0.8)
plt.plot(d_low[mask], 10*np.log10(i_low[mask]**2 + 1e-30), label='Low', alpha=0.8)
plt.plot(d_high[mask], 10*np.log10(i_high[mask]**2 + 1e-30), label='High', alpha=0.8)

plt.legend()
plt.xlabel('Depth (m)')
plt.ylabel('Amplitude (dB)')
plt.title(f'Subband Comparison (Depth {d_min}-{d_max}m)')
plt.grid(True)
plt.savefig('/Users/hannesstahlin/.gemini/antigravity/tmp/subband_comparison_trace.png')
print("Comparison plot saved.")
