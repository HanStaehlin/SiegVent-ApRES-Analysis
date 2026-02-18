# How to Determine Accumulation Rate for Your Site

## ⚠️ **Critical Parameter**

The accumulation rate is essential for:
1. Correcting ApRES velocities (steady-state assumption)
2. Estimating layer slopes
3. Interpreting vertical motion

**Currently using: 0.3 m/year** (unverified)

---

## 🎯 **Methods to Determine Accumulation Rate**

### **Option 1: Field Measurements (Best)**

#### **A. Stake Farm / GPS**
If you have repeated GPS measurements at stakes:
```
accumulation = (surface_elevation_2 - surface_elevation_1) / time_span
```

**Data you might have:**
- GPS surveys from SALSA campaign?
- Stake measurements from ApRES site?
- Time series of surface elevation?

#### **B. Shallow Firn Core**
If you have a shallow firn core with known age markers:
```
accumulation = depth_water_equivalent / age_span
```

**Age markers:**
- Volcanic ash layers (known eruption dates)
- Beta activity peaks (nuclear tests: 1963-64)
- Chemical markers (e.g., MSA seasonal cycles)

### **Option 2: Regional Climate Models**

#### **A. RACMO2 (High Resolution)**
Regional Atmospheric Climate Model for polar regions
- Resolution: 27 km or 5.5 km
- Covers 1979-present
- Surface Mass Balance (SMB) output

**Access:**
```bash
# From KNMI Climate Explorer or direct from IMAU
# https://www.projects.science.uu.nl/iceclimate/models/antarctica.php
```

**For Mercer Lake (84.64°S, 149.50°W):**
```python
# Example query
import xarray as xr

# Load RACMO2 data (if you have access)
ds = xr.open_dataset('RACMO2.3p2_ANT27_SMB_monthly_1979_2018.nc')

# Extract at your location
lat, lon = -84.64, -149.50  # Mercer Lake
smb = ds['SMB'].sel(lat=lat, lon=lon, method='nearest')
accumulation = smb.mean('time')  # Annual average
print(f"Accumulation: {accumulation.values:.3f} m ice eq/year")
```

#### **B. MAR (Modèle Atmosphérique Régional)**
Similar to RACMO, alternative model
- Resolution: 35 km
- Good for West Antarctica

#### **C. Reanalysis Data (ERA5, MERRA-2)**
Lower resolution but globally available
- ERA5: 31 km
- MERRA-2: 50 km
- Variables: snowfall, sublimation

### **Option 3: Published Studies**

#### **A. Check SALSA Publications**
The SALSA project team may have measured or cited accumulation:

1. **Siegfried et al. 2023 (Geology)** - Your main paper
   - Check methods section
   - Check supplementary materials

2. **Other SALSA papers:**
   - Venturelli et al. (2020) - Lake drainage
   - Michaud et al. (2021) - Borehole observations
   - Check their meteorological data

#### **B. Regional Compilations**
- **Arthern et al. (2006)** - Antarctic accumulation compilation
  - https://doi.org/10.1029/2006JD007223
  - Provides gridded accumulation for all Antarctica

- **Van Wessem et al. (2014)** - RACMO2 climatology
  - https://doi.org/10.5194/tc-8-1607-2014

- **Medley et al. (2014)** - Airborne radar accumulation
  - https://doi.org/10.1002/2014JF003178
  - Has data for Siple Coast / Ross Ice Streams

#### **C. Ice Core Records**
Nearby deep ice cores with accumulation history:
- WAIS Divide (~400 km away): ~0.22 m/year
- Byrd Station (~300 km away): ~0.11 m/year

**Note:** Your site is likely higher due to proximity to Ross Ice Shelf

### **Option 4: Estimate from ApRES Surface Tracking**

If your ApRES tracked the surface:

```python
# Look at your ApRES data
# The surface reflection should move downward due to accumulation

import matplotlib.pyplot as plt
from scipy.io import loadmat

# Load ApRES data
data = loadmat('data/apres/ImageP2_python.mat')

# Surface range vs time
# (You'd need to identify the surface peak)
surface_range = data['vif'][:, 0, :]  # First range bin, all chirps

# If you tracked surface displacement:
surface_displacement = ...  # meters
time_span_years = ...  # years

accumulation_estimate = surface_displacement / time_span_years
print(f"Accumulation estimate: {accumulation_estimate:.3f} m/year")
```

**Caveat:** Surface tracking in ApRES can be tricky due to:
- Near-field effects
- Antenna coupling changes
- Need to separate from ice motion

---

## 📊 **Expected Values for Your Region**

### **West Antarctic Ice Streams (Siple Coast):**
```
Typical range: 0.1 - 0.4 m water eq/year
Mean: ~0.2 - 0.3 m/year

Spatial pattern:
- Coast (Ross Ice Shelf edge): 0.3 - 0.5 m/year (high)
- Interior (near divide): 0.1 - 0.2 m/year (low)
- Ice streams: 0.2 - 0.3 m/year (moderate)
```

### **Mercer Lake Specific:**
**Location:** 84.64°S, 149.50°W
- ~80 km from Ross Ice Shelf edge
- On Whillans Ice Stream (formerly Ice Stream B)
- Elevation: ~50-100 m above sea level

**Expected accumulation:** 0.25 - 0.35 m/year

This matches your assumed **0.3 m/year** ✓

---

## ✅ **Recommended Actions**

### **Priority 1: Check SALSA Data Archive**
```bash
# Did the SALSA project measure accumulation?
# Check:
# - Stake measurements
# - GPS height changes
# - Weather station data
# - Firn density profiles
```

### **Priority 2: Extract from RACMO2**
```python
# If you can access RACMO2 data:
# 1. Download from https://www.projects.science.uu.nl/iceclimate/
# 2. Extract SMB at 84.64°S, 149.50°W
# 3. Average over your measurement period
```

### **Priority 3: Cite Regional Study**
If direct measurement unavailable, cite regional estimate:

> "Surface accumulation was estimated as 0.30 ± 0.05 m ice equivalent per year,
> based on [RACMO2 / Arthern et al., 2006 / regional compilation] for the
> Whillans Ice Stream region. Sensitivity analysis (see Section X) shows results
> are robust to ±20% uncertainty in this parameter."

### **Priority 4: GPS Surface Height**
If you have GPS at the ApRES site over time:
```python
# From GPS elevation changes
surface_elevation_change = GPS_elevation_final - GPS_elevation_initial
time_span = ...  # years

# Separate accumulation from ice dynamics
# (Requires knowing vertical velocity from ApRES)
accumulation = surface_elevation_change - vertical_velocity_at_surface
```

---

## 📝 **For Your Thesis/Paper**

### **If You Have a Measurement:**
> "Surface accumulation was measured as [value] ± [uncertainty] m ice equivalent
> per year from [stake measurements / GPS surveys / firn cores / method] over
> the period [dates]."

### **If You Use a Model:**
> "Surface accumulation was estimated as [value] ± [uncertainty] m ice equivalent
> per year from the RACMO2.3p2 regional climate model [van Wessem et al., 2018]
> averaged over the measurement period [dates] at the ApRES location
> (84.64°S, 149.50°W)."

### **If You Use Literature:**
> "Surface accumulation was estimated as [value] ± [uncertainty] m ice equivalent
> per year, consistent with regional compilations for the Whillans Ice Stream
> [Arthern et al., 2006; Medley et al., 2014]. Sensitivity analysis indicates
> our results are robust to ±20% uncertainty in accumulation rate (see Section X)."

### **Current Situation (Placeholder):**
> "Surface accumulation was assumed to be 0.3 m ice equivalent per year,
> typical for the Whillans Ice Stream region [citation needed]. **[NEED TO VERIFY]**"

---

## 🎯 **Bottom Line**

**You NEED to determine where your accumulation value comes from before publication.**

**Options ranked by preference:**
1. ⭐⭐⭐ **Field measurement** (stakes, GPS, firn core) - most defensible
2. ⭐⭐ **RACMO2 model output** at your location - widely accepted
3. ⭐ **Regional literature value** with citation - acceptable with sensitivity analysis

**Do NOT:**
- ❌ Use an uncited assumed value
- ❌ Claim precision you don't have
- ❌ Ignore the uncertainty this introduces

Your **sensitivity analysis already shows** the impact of ±20% accumulation uncertainty,
which is good! But you need to document where your baseline value comes from.

---

## 🔍 **Action Items**

- [ ] Check SALSA data archive for stake/GPS measurements
- [ ] Contact SALSA team (Siegfried, Venturelli) about accumulation data
- [ ] Try to access RACMO2 output for your location
- [ ] Check published SALSA papers for meteorological data
- [ ] Review your GPS data for surface height changes
- [ ] Determine uncertainty range (±0.05 m/year? ±0.1 m/year?)
- [ ] Update all documentation with source of accumulation value
- [ ] Re-run sensitivity analysis if value changes significantly

**This is important! A reviewer will definitely ask.**
