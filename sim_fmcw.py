import numpy as np
import matplotlib.pyplot as plt

N = 1000
K = 1e12
target_delay = 1e-6
t = np.linspace(0, 1e-3, N)
# Beat signal for a stationary target
vif = np.exp(1j * 2 * np.pi * K * target_delay * t)

def get_spec(subband):
    win = np.zeros(N)
    if subband == 'full':
        win[:] = np.hanning(N)
    elif subband == 'low':
        win[:N//2] = np.hanning(N//2)
    elif subband == 'high':
        win[N//2:] = np.hanning(N//2)
        
    vif_w = vif * win
    # shift by N//2
    vif_w = np.roll(vif_w, -N//2)
    spec = np.fft.fft(vif_w, n=4000)
    return np.abs(spec)

plt.figure()
plt.plot(get_spec('full'), label='full')
plt.plot(get_spec('low'), label='low')
plt.plot(get_spec('high'), label='high')
plt.legend()
plt.xlim(0, 50)
plt.savefig('test_peak.png')

peaks = {
    'full': np.argmax(get_spec('full')),
    'low': np.argmax(get_spec('low')),
    'high': np.argmax(get_spec('high'))
}
print(peaks)
