#!/usr/bin/env python
"""Check the actual range change in the ApRES data."""
import scipy.io as sio
import numpy as np

# Check the processed data
data = sio.loadmat('data/apres/ImageP2_python.mat')
time_days = data['TimeInDays'].flatten()
range_img = data['RawImage']
Rcoarse = data['Rcoarse'].flatten()

print('=== ApRES Ice Thickness Change ===')
print(f'Time span: {time_days[-1] - time_days[0]:.1f} days ({(time_days[-1] - time_days[0])/365.25:.2f} years)')

# Find the basal reflection peak for first and last measurements
mask = (Rcoarse >= 1050) & (Rcoarse <= 1150)
first_peak_idx = np.argmax(range_img[mask, 0])
last_peak_idx = np.argmax(range_img[mask, -1])
first_range = Rcoarse[mask][first_peak_idx]
last_range = Rcoarse[mask][last_peak_idx]

print(f'First measurement basal range: {first_range:.2f} m')
print(f'Last measurement basal range: {last_range:.2f} m')
print(f'Change in range: {last_range - first_range:.2f} m')
print(f'Annualized: {(last_range - first_range) / (time_days[-1]/365.25):.2f} m/year')

# Also check the MATLAB processed fine range
matlab_data = sio.loadmat('data/apres/rangeOverTime.mat')
matlab_range = matlab_data['range'].flatten()
matlab_time = matlab_data['timeInDays'].flatten()

print()
print('=== MATLAB Fine Range Results ===')
print(f'First range: {matlab_range[0]:.4f} m')
print(f'Last range: {matlab_range[-1]:.4f} m') 
print(f'Change: {matlab_range[-1] - matlab_range[0]:.4f} m')
print(f'Time span: {matlab_time[-1] - matlab_time[0]:.1f} days')
print(f'Annualized: {(matlab_range[-1] - matlab_range[0]) / ((matlab_time[-1] - matlab_time[0])/365.25):.2f} m/year')
