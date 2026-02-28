#!/usr/bin/env python3
"""
Diagnose and remove the residual carrier-phase oscillation (~lambda_c/2 period).

The phase correction in fmcw_range subtracts phi_ref = 2pi*fc*tau - K*tau^2/2,
but any mismatch leaves a residual oscillation with period ~lambda_c/2.
Here we measure it, remove it, and test the impact on phase-slope velocities.
"""
import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from radon_velocity import phase_slope_velocity

# ---- Load ----
print("Loading data...")
mat = loadmat('data/apres/ImageP2_python.mat')
raw_complex = np.array(mat['RawImageComplex'])
Rcoarse = mat['Rcoarse'].flatten()
time_days = mat['TimeInDays'].flatten()
lambdac = float(mat.get('lambdac', np.array([0.5608])).flatten()[0])
dz = float(Rcoarse[1] - Rcoarse[0])
n_bins, n_times = raw_complex.shape
print(f"  {n_bins} bins x {n_times} times, dz={dz:.4f} m")
print(f"  Expected period: lam_c/2 = {lambdac/2:.4f} m = {lambdac/2/dz:.1f} bins")

# ---- Deep region ----
i0 = np.searchsorted(Rcoarse, 785)
i1 = np.searchsorted(Rcoarse, 1094)
depths = Rcoarse[i0:i1]
region = raw_complex[i0:i1, :]

# ---- 1. Measure spatial phase gradient ----
print("\n--- Spatial phase gradient ---")
grads = []
for ti in [0, n_times//4, n_times//2, 3*n_times//4, n_times-1]:
    ph = np.unwrap(np.angle(region[:, ti]))
    c = np.polyfit(depths, ph, 1)
    grads.append(c[0])
    print(f"  t={ti:5d}: grad = {c[0]:.4f} rad/m  period = {2*np.pi/abs(c[0]):.4f} m")
grad_mean = np.mean(grads)
print(f"  Mean: {grad_mean:.4f} rad/m  period = {2*np.pi/abs(grad_mean):.4f} m")

# ---- 2. Spatial freq spectrum ----
t_idx = n_times // 2
phase_uw = np.unwrap(np.angle(region[:, t_idx]))
dph = np.diff(phase_uw)
freq = np.fft.rfftfreq(len(dph), d=dz)
psd = np.abs(np.fft.rfft(dph - dph.mean()))**2

# ---- 3. Remove residual carrier phase ----
print("\n--- Applying correction ---")
correction = np.exp(-1j * grad_mean * depths)
region_cor = region * correction[:, np.newaxis]
ph_after = np.unwrap(np.angle(region_cor[:, t_idx]))
c_after = np.polyfit(depths, ph_after, 1)
print(f"  After: grad = {c_after[0]:.4f} rad/m (was {grad_mean:.4f})")

# ---- 4. Phase-slope velocities: 3 variants ----
print("\n--- Phase-slope velocity comparison ---")
window_bins = int(round(10.0 / dz))
step_bins = int(round(5.0 / dz))
starts = list(range(0, len(depths) - window_bins + 1, step_bins))

# Also SVD on raw and corrected
U_r, S_r, Vh_r = np.linalg.svd(region, full_matrices=False)
S_t = np.zeros_like(S_r); S_t[:3] = S_r[:3]
denoised_raw = U_r @ np.diag(S_t) @ Vh_r

U_c, S_c, Vh_c = np.linalg.svd(region_cor, full_matrices=False)
S_t2 = np.zeros_like(S_c); S_t2[:3] = S_c[:3]
denoised_cor = U_c @ np.diag(S_t2) @ Vh_c

configs = [
    ('Raw, no SVD',             region),
    ('Corrected, no SVD',       region_cor),
    ('Raw + SVD k=3',           denoised_raw),
    ('Corrected + SVD k=3',     denoised_cor),
]

all_res = {}
for label, data in configs:
    res = {'d': [], 'v': [], 'r2': [], 'ng': []}
    for i in starts:
        cd = float(depths[i + window_bins // 2])
        ps = phase_slope_velocity(data[i:i+window_bins, :], time_days, lambdac)
        res['d'].append(cd)
        res['v'].append(ps['best_v'])
        res['r2'].append(ps['median_r2'])
        res['ng'].append(ps['n_good'])
    all_res[label] = res

nye_int, nye_sl = 0.0453, 0.000595
for label, res in all_res.items():
    d = np.array(res['d']); v = np.array(res['v'])
    r2 = np.array(res['r2']); ng = np.array(res['ng'])
    nye = nye_int + nye_sl * d
    ok = np.isfinite(v)
    resid = v[ok] - nye[ok]
    rms = np.sqrt((resid**2).mean())
    bias = resid.mean()
    cf = np.polyfit(d[ok], v[ok], 1)
    print(f"\n  {label}:")
    print(f"    Valid: {ok.sum()}/{len(v)}, v=[{v[ok].min():.4f}, {v[ok].max():.4f}]")
    print(f"    R2=[{r2[ok].min():.3f}, {r2[ok].max():.3f}], n_good=[{ng[ok].min()}, {ng[ok].max()}]")
    print(f"    RMS={rms:.4f}, bias={bias:+.4f}")
    print(f"    Fit: v = {cf[1]:.4f} + {cf[0]:.6f}*d  (Nye: 0.0453 + 0.000595*d)")

# ---- 5. Plot ----
fig, axes = plt.subplots(1, 3, figsize=(18, 8))

# Panel 1: Spatial phase
ax = axes[0]
ax.plot(np.angle(region[:200, t_idx]), depths[:200], 'b-', alpha=0.6, lw=0.5, label='Raw')
ax.plot(np.angle(region_cor[:200, t_idx]), depths[:200], 'r-', alpha=0.6, lw=0.5, label='Corrected')
ax.set_xlabel('Phase (rad)'); ax.set_ylabel('Depth (m)')
ax.set_title('Spatial phase (first 200 bins)'); ax.legend(); ax.invert_yaxis()

# Panel 2: Spatial frequency
ax = axes[1]
ax.semilogy(freq, psd, 'b-', lw=0.8)
ax.axvline(2/lambdac, color='r', ls='--', lw=1.5, label=f'1/(lam_c/2) = {2/lambdac:.1f} cy/m')
ax.set_xlabel('Spatial frequency (cycles/m)'); ax.set_ylabel('PSD')
ax.set_title('Spatial freq. of phase diffs'); ax.legend()

# Panel 3: Velocities
ax = axes[2]
styles = [('#4393c3','D'), ('#f4a582','s'), ('#e66101','^'), ('#2ca25f','o')]
for (label, res), (col, mk) in zip(all_res.items(), styles):
    d = np.array(res['d']); v = np.array(res['v']); ok = np.isfinite(v)
    ax.scatter(v[ok], d[ok], c=col, marker=mk, s=22, alpha=0.7, edgecolors='none', label=label)
    cf = np.polyfit(d[ok], v[ok], 1)
    df = np.array([d[ok].min(), d[ok].max()])
    ax.plot(np.polyval(cf, df), df, color=col, lw=1.3, ls='--', alpha=0.8)
nye_d = np.linspace(785, 1094, 100)
ax.plot(nye_int + nye_sl*nye_d, nye_d, 'k-', lw=2.5, label='Nye')
ax.set_xlabel('Velocity (m/yr)'); ax.set_title('Phase-slope velocity')
ax.legend(fontsize=7, loc='lower left'); ax.invert_yaxis()
ax.set_xlim(-1.5, 2.8); ax.grid(True, alpha=0.3)

plt.suptitle('Residual carrier phase: diagnosis & correction', fontsize=13)
plt.tight_layout()
plt.savefig('output/apres/phase_detrend_test.png', dpi=150, bbox_inches='tight')
print(f"\nPlot: output/apres/phase_detrend_test.png")
