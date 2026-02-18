# ApRES Flow Regime Interpretation Checklist

Quick reference for interpreting your ApRES velocity measurements.

## 📋 Pre-Analysis Checklist

- [ ] GPS horizontal velocity measured: ______ m/year
- [ ] Ice thickness known: ______ m
- [ ] Surface accumulation rate estimated: ______ m/year
- [ ] Velocity profile has > 5 reliable layers
- [ ] Bed reflection identified: Yes / No / Uncertain

## 🔍 Step 1: Identify Flow Regime (5 minutes)

### Run the analysis:
```bash
python proc/apres/flow_regime_analysis.py \
    --velocity-data data/apres/layer_analysis/velocity_profile \
    --horizontal-velocity 200 \
    --accumulation-rate 0.3 \
    --bed-depth 1094 \
    --output results/flow_regime \
    --save-figure
```

### Question 1: Is velocity nearly constant with depth?

**Plot:** Look at leftmost panel "Measured Velocity Profile"

- [ ] **YES** → Velocities vary < 15% → **PLUG FLOW** → Go to Section A
- [ ] **NO** → Velocities show clear trend → **SHEAR FLOW** → Go to Section B
- [ ] **UNCERTAIN** → Noisy or few layers → Need more data

---

## 📊 Section A: Plug Flow Interpretation

### What Plug Flow Means:
✅ Ice slides as rigid block
✅ Bed is NOT frozen (sliding boundary condition)
✅ Minimal internal deformation
✅ All motion is basal sliding

### Checklist:

1. **Record velocities:**
   - Surface velocity (extrapolated): ______ m/year
   - Basal velocity (extrapolated): ______ m/year
   - Difference: ______ m/year (should be small!)

2. **Check confidence:**
   - Plug flow confidence: ______ (> 0.7 is strong)
   - [ ] Confidence > 0.7 → Reliable conclusion
   - [ ] Confidence < 0.7 → Weak evidence, report uncertainty

3. **Physical interpretation:**
   - [ ] Bed is at pressure melting point (temperate)
   - [ ] Subglacial water likely present
   - [ ] Ice-bed coupling is low (sliding)
   - [ ] No significant basal freeze-on

### Report:
> "ApRES measurements indicate **plug flow** (confidence: ___), with uniform velocity
> (___±___ m/year) throughout the ice column. This demonstrates the ice is sliding
> over an unfrozen bed, consistent with the presence of Mercer Subglacial Lake."

---

## 📈 Section B: Shear Flow Interpretation

### What Shear Flow Means:
- Ice is deforming internally
- May have sliding + deformation (mixed)
- Velocity decreases with depth

### Checklist:

1. **Measure velocity gradient:**
   - Mean shear strain rate: ______ × 10⁻³ /year
   - Maximum shear: ______ × 10⁻³ /year
   - Depth of maximum shear: ______ m

2. **Assess basal conditions:**
   - Basal velocity: ______ m/year
   - [ ] v_basal > 0.5 m/year → **Sliding + deformation**
   - [ ] v_basal ≈ 0 m/year → **Pure deformation (frozen bed)**
   - [ ] v_basal < 0 → **Check data quality!**

3. **Calculate sliding ratio:**
   ```
   Sliding fraction = v_basal / v_surface = ______
   ```
   - [ ] > 80% → Dominated by sliding
   - [ ] 50-80% → Mixed regime
   - [ ] < 50% → Dominated by deformation

4. **Analyze shear distribution:**
   - [ ] Shear concentrated near bed → Basal deformation layer
   - [ ] Shear uniform with depth → Distributed deformation
   - [ ] Shear concentrated mid-depth → Check layer geometry!

### Report (Sliding + Deformation):
> "Velocity decreases from ___m/year at the surface to ___m/year at the bed,
> indicating internal deformation with mean shear strain rate of ___×10⁻³/year.
> Non-zero basal velocity (___m/year) indicates continued sliding, suggesting a
> mixed regime with ___% sliding and ___% internal deformation."

### Report (Pure Deformation):
> "Velocity profile shows continuous decrease to near-zero at the bed, indicating
> ice is frozen to bedrock with no basal sliding. All motion (___m/year at surface)
> is accommodated by internal deformation."

---

## 🎯 Step 2: Assess Geometric Effects (10 minutes)

### Question 2: Are layer slopes reasonable?

**Plot:** Look at "Estimated Layer Slope" panel

- [ ] **Most layers < 2°** → Reasonable, likely real geometry
- [ ] **Layers 2-5°** → Possible, especially near bed
- [ ] **Layers > 5°** → Question: Real folding OR off-axis?

### If layers > 5°:

**Possible causes:**
1. Real layer folding (near bed, over topography)
2. Off-axis reflections (side-lobes)
3. Layer picking errors
4. Wrong horizontal velocity

**Action items:**
- [ ] Check if slope increases with depth (expected)
- [ ] Compare to RES/GPR data if available
- [ ] Test sensitivity to horizontal velocity:
  ```bash
  # Try ±10% on horizontal velocity
  python flow_regime_analysis.py ... --horizontal-velocity 180
  python flow_regime_analysis.py ... --horizontal-velocity 220
  ```

### Question 3: Do layer slopes vary systematically?

**Pattern:** Plot slope vs depth

- [ ] **Constant slope** → Layers parallel, tilted together
- [ ] **Increasing with depth** → Expected (layers draped over bed)
- [ ] **Random scatter** → Measurement noise OR picking errors

---

## 🪨 Step 3: Evaluate Bed Reflection (5 minutes)

### Question 4: Is bed reflection from nadir?

**From output:** Check "BED REFLECTION ASSESSMENT"

Required bed tilt: ______ degrees

- [ ] **< 2°** → ✅ **Likely nadir** → Use bed velocity with confidence
- [ ] **2-5°** → ⚠️ **Possibly off-nadir** → Report with uncertainty
- [ ] **> 5°** → ❌ **Likely off-nadir** → Don't trust bed velocity

### If bed tilt > 5°:

**Your professor's advice applies:** "Treat bed as a point"

**What this means:**
1. **Assume** bed reflection is from nadir (bed tilt = 0°)
2. **Accept** the measured bed velocity may have error
3. **Alternative:** Use deepest reliable **layer** instead of bed

**Action:**
```python
# Use deepest layer with good R² instead of bed
# In your analysis, exclude bed from velocity profile
# Extrapolate to bed from deepest 3-5 layers
```

### Decision Tree:

```
Is bed tilt < 2°?
├─ YES → Use bed velocity
│         Report: "Bed reflection consistent with nadir"
│
└─ NO → Is it 2-5°?
    ├─ YES → Use with caution
    │         Report: "Bed reflection possibly off-nadir;
    │                  basal velocity has ±5m/year uncertainty"
    │
    └─ NO (>5°) → Exclude bed
                   Report: "Bed reflection likely off-nadir;
                           basal velocity estimated from deepest
                           reliable layers"
```

---

## 🔬 Step 4: Physical Interpretation (10 minutes)

### Combine all evidence:

#### Flow Regime: Plug / Shear
#### Layer slopes: Reasonable / Questionable
#### Bed reflection: Nadir / Off-nadir

### Interpretation Matrix:

| Flow | Layers | Bed | Interpretation |
|------|--------|-----|----------------|
| Plug | ✓ | ✓ | **Strong**: Sliding on flat bed |
| Plug | ✓ | ✗ | **Good**: Sliding, bed geometry uncertain |
| Plug | ✗ | ✓ | **Uncertain**: Check layer picks |
| Plug | ✗ | ✗ | **Weak**: Multiple issues, needs work |
| Shear | ✓ | ✓ | **Strong**: Deformation ± sliding, good constraint |
| Shear | ✓ | ✗ | **Good**: Deformation confirmed, basal uncertain |
| Shear | ✗ | ✓ | **Uncertain**: Check layer geometry |
| Shear | ✗ | ✗ | **Weak**: Multiple issues, needs work |

---

## 📝 Step 5: Report Results

### Required Information:

1. **Flow Regime:**
   - Classification: Plug / Shear / Mixed
   - Confidence: _____
   - Evidence: (describe velocity profile shape)

2. **Velocity Estimates:**
   - Surface: _____ ± _____ m/year
   - Basal: _____ ± _____ m/year
   - Change: _____ m/year

3. **Geometric Corrections:**
   - Mean layer slope: _____ ± _____ degrees
   - Geometric component: _____ m/year
   - Corrected vertical velocity: _____ m/year
   - Comparison to accumulation: (consistent / inconsistent)

4. **Bed Assessment:**
   - Bed tilt: _____ degrees
   - Reflection quality: (nadir / off-nadir / uncertain)
   - Basal condition: (sliding / frozen / uncertain)

5. **Uncertainties:**
   - Largest sources: (list 2-3)
   - Sensitivity tests: (describe what you tested)
   - Alternative interpretations: (if any)

### Template for Paper/Thesis:

```markdown
## Flow Regime Analysis

ApRES phase-sensitive radar measurements at Mercer Subglacial Lake reveal
[plug flow / shear flow] with [high/moderate/low] confidence (score: ___).
The velocity profile shows [describe shape: constant/decreasing/variable]
velocities ranging from ___ m/year at ___ m depth to ___ m/year at ___ m depth.

[IF PLUG FLOW:]
The uniform velocity profile indicates the ice column moves as a coherent unit
through basal sliding over an unfrozen bed. This is consistent with the presence
of subglacial water at Mercer Subglacial Lake.

[IF SHEAR FLOW:]
The velocity gradient (mean shear strain rate: ___×10⁻³/year) indicates internal
deformation. [IF BASAL VELOCITY > 0: A non-zero basal velocity of ___m/year
suggests continued sliding, with approximately ___% of motion from sliding and
___% from deformation.] [IF BASAL VELOCITY ≈ 0: Near-zero basal velocity suggests
the ice may be locally frozen to bedrock.]

Layer slopes required to explain the measurements average ___°, which is
[reasonable/high] for ice interior/near-bed conditions. [IF NADIR: The bed
reflection is consistent with nadir geometry.] [IF OFF-NADIR: The bed reflection
may be off-nadir; basal velocity estimates carry additional uncertainty of ±___m/year.]

After correcting for geometric effects of horizontal flow (~200m/year), the
mean vertical velocity is ___m/year, [consistent/inconsistent] with the
estimated accumulation rate of ___m/year.
```

---

## ⚠️ Red Flags

Watch for these warning signs:

- [ ] Required layer slopes > 10° → Check everything!
- [ ] Bed tilt > 10° → Definitely off-nadir
- [ ] Corrected vertical velocity >> accumulation → Problem!
- [ ] Negative shear strain rate → Velocity increases with depth (wrong!)
- [ ] Plug flow but R² < 0.5 → Poor quality data
- [ ] Layer slopes not consistent between adjacent layers → Picking errors

## 🎓 Discussion with Professor

### Questions to Ask:

1. **"What accumulation rate should I use?"**
   - From firn core? Stake measurements? Regional model?

2. **"Do we have independent layer geometry data?"**
   - RES surveys? GPR? Borehole imaging?

3. **"How should I report uncertainty?"**
   - Monte Carlo? Sensitivity analysis? Parameter ranges?

4. **"Is steady-state assumption valid?"**
   - Lake draining/filling? Transient flow?

5. **"What's known about bed geometry?"**
   - Seismic? Radar? Expected roughness?

### Points to Discuss:

- Your flow regime classification and confidence
- Estimated layer slopes and if they're reasonable
- Bed reflection interpretation
- Comparison to expected physics (subglacial lake = sliding)
- Sensitivity to assumptions

---

## 📚 Next Steps

After completing this checklist:

1. [ ] Generate all diagnostic plots
2. [ ] Calculate uncertainties (try ±20% on key parameters)
3. [ ] Compare to published ApRES studies (Kingslake, Nicholls, etc.)
4. [ ] Draft interpretation text
5. [ ] Discuss with professor
6. [ ] Refine analysis based on feedback
7. [ ] Include in thesis/paper

## 🔗 Quick Links

- Main analysis script: `proc/apres/flow_regime_analysis.py`
- Theory guide: `proc/apres/FLOW_REGIME_INTERPRETATION.md`
- Visualization app: `proc/apres/visualization_app.py`

## 💡 Remember

> "All models are wrong, but some are useful." - George Box

Your interpretation is a **model** of reality. Report:
- What you measured
- What you assumed
- What you concluded
- What's uncertain

Good science acknowledges limitations!
