# Flow Regime Interpretation Guide for ApRES Data

## The Fundamental Problem

ApRES measures **range change** (vertical component of motion), but ice has both **horizontal** and **vertical** motion. When ice flows horizontally past tilted features (layers or bed), the measured range velocity contains a geometric artifact.

### Key Equation

```
v_measured = v_vertical + u_horizontal × tan(α)
           = v_z      + geometric artifact
```

Where:
- `v_measured`: What ApRES measures (range rate)
- `v_z`: True vertical motion (accumulation - thinning)
- `u_horizontal`: Horizontal velocity (~200 m/year at Mercer)
- `α`: Layer or bed slope angle

## Interpretation Framework

### 1. **Velocity Profile Shape → Flow Regime**

The shape of velocity vs depth reveals the flow physics:

#### **Plug Flow** (Sliding Bed)
```
Surface  ●────────● Bed
         ├────────┤
         v = constant
```
- **Observation**: Velocity nearly constant with depth
- **Physics**: Ice slides as coherent slab over bed
- **Implication**: Bed is not frozen (sliding boundary condition)
- **Internal deformation**: Minimal
- **Basal sliding**: All motion

#### **Shear Flow** (Deformation)
```
Surface  ●
         ├\
         │ \
         │  \
Bed      ●───●
         high → low velocity
```
- **Observation**: Velocity decreases with depth
- **Physics**: Internal deformation + possible sliding
- **Implication**: Ice deforms internally
- **Basal velocity > 0**: Some sliding
- **Basal velocity = 0**: Frozen bed, pure deformation

#### **Mixed Flow** (Most Common)
```
Surface  ●────
         ├────\
         │     \  <- Shear zone
Bed      ●──────●
         sliding
```
- Constant velocity in upper ice (plug)
- Shear near bed
- Combination of sliding + basal deformation

### 2. **Geometric Correction → Layer Slopes**

With known horizontal velocity, you can estimate required layer slopes:

```
tan(α) = (v_measured - v_vertical) / u_horizontal
```

**Typical ranges:**
- **Horizontal layers**: α ≈ 0° (v_measured ≈ v_vertical)
- **Small tilt**: α = 0.5-2° (common in ice sheets)
- **Significant tilt**: α = 2-5° (near bedrock topography)
- **Steep layers**: α > 5° (folding, shear zones)

**What layer slopes tell you:**
- Uniform small slopes (< 2°): Reasonable, probably real layers
- Increasing slopes with depth: Layers may be bent by flow
- Large slopes (> 5°): Either real folding OR off-axis reflections

### 3. **Bed Reflection Assessment**

Your professor says to "treat the bed as a point" - this means **assume nadir reflection** (directly below ApRES). This is valid if:

#### **Evidence for nadir reflection:**
- Bed velocity consistent with expected vertical motion (~ -accumulation)
- Required bed tilt < 2° (reasonable for smooth bed)
- Strong, stable reflection over time

#### **Evidence for off-nadir reflection:**
- Bed velocity requires large tilt angle (> 5°) to explain
- Weak or variable reflection amplitude
- Known bedrock topography in area

#### **When assumption breaks down:**
You can test the "bed as point" assumption:

```python
# Calculate required bed tilt
bed_tilt = arctan((v_bed - v_vertical) / u_horizontal)

# If |bed_tilt| > 5°, assumption may be questionable
```

### 4. **Dealing with Uncertainty**

You have **THREE unknowns** but only **TWO measurements** (ApRES + GPS):
1. True vertical velocity (v_z)
2. Layer slope (α)
3. Flow regime (plug vs shear)

**Strategies to close the system:**

#### **A. Assume Steady State**
- Accumulation = Vertical velocity at surface
- From firn cores or stake measurements
- Allows solving for layer slopes

#### **B. Assume Horizontal Layers**
- α = 0° → v_measured = v_vertical
- Test consistency with mass balance
- Valid for shallow layers away from bed

#### **C. Use Layer Consistency**
- Multiple layers should show consistent slopes
- Random scatter → measurement noise
- Systematic trend → real layer geometry

#### **D. Independent Constraints**
- Radio echo sounding (RES) for layer geometry
- Ground-penetrating radar (GPR) for shallow layers
- Seismic for bed slope
- Satellite altimetry for surface elevation change

## Practical Analysis Workflow

### **Step 1: Characterize Velocity Profile**

```bash
python flow_regime_analysis.py \
    --velocity-data data/apres/layer_analysis/velocity_profile \
    --horizontal-velocity 200 \
    --accumulation-rate 0.3 \
    --bed-depth 1094
```

**Look for:**
- Is velocity constant (plug) or gradient (shear)?
- What is confidence score?
- What are extrapolated surface/bed velocities?

### **Step 2: Assess Geometric Corrections**

**Check estimated layer slopes:**
- Are they reasonable (< 2-5°)?
- Do they vary systematically with depth?
- Are deeper layers steeper? (Expected near bed)

### **Step 3: Evaluate Bed Reflection**

**Questions:**
- What bed tilt is required to explain measurement?
- Is it < 2° (likely nadir)?
- Is it > 5° (questionable assumption)?

**If bed tilt is large**, consider:
- Bed may be tilted or rough
- Reflection may be off-axis
- Use "deepest reliable layer" instead of bed

### **Step 4: Physical Interpretation**

Combine all evidence:

#### **Scenario A: Plug Flow + Nadir Bed**
✅ **Interpretation**: Sliding on unfrozen bed
- Bed is temperate (at pressure melting point)
- Subglacial water present (likely)
- Minimal internal deformation
- Good constraint on basal conditions

#### **Scenario B: Shear Flow + Nadir Bed**
✅ **Interpretation**: Mix of sliding + deformation
- Calculate sliding ratio: v_basal / v_surface
- Analyze shear concentration (depth of maximum dv/dz)
- May indicate temperate basal layer

#### **Scenario C: Plug Flow + Off-Nadir Bed**
⚠️ **Interpretation**: Sliding on bed, but bed geometry unclear
- Ice still slides (from plug flow)
- Bed slope or roughness unknown
- Use deepest layer for basal velocity estimate

#### **Scenario D: Shear Flow + Off-Nadir Bed**
⚠️ **Interpretation**: Complex, needs additional data
- Internal deformation confirmed
- Basal velocity uncertain
- Cannot separate sliding from deformation

## Key Diagnostics to Report

### **Flow Regime**
- Plug flow confidence score
- Mean shear strain rate (if shear flow)
- Depth of shear concentration

### **Velocity Estimates**
- Surface velocity (extrapolated from layers)
- Basal velocity (extrapolated or from bed)
- Velocity change (surface - basal)

### **Geometric Corrections**
- Mean layer slope
- Depth variation of slope
- Consistency between layers

### **Bed Assessment**
- Estimated bed tilt
- Likelihood of nadir reflection
- Alternative interpretation if off-nadir

## Common Pitfalls

### ❌ **Mistake 1: Ignoring Horizontal Flow**
If u_h = 200 m/year and layer slope = 1°, geometric component is:
```
u × tan(1°) ≈ 200 × 0.0175 ≈ 3.5 m/year
```
This is **10× larger** than typical accumulation! Cannot ignore.

### ❌ **Mistake 2: Over-interpreting Bed Velocity**
If bed reflection is off-nadir by just 3°:
```
Artifact ≈ 200 × tan(3°) ≈ 10 m/year
```
This is huge compared to true vertical motion (~0.3 m/year).

### ❌ **Mistake 3: Assuming Layers Are Horizontal**
Near bed, layers can be significantly tilted by:
- Flow over bedrock topography
- Raymond bumps (at divides)
- Shear zone folding

### ✅ **Best Practice: Sensitivity Analysis**
Test range of assumptions:
- Accumulation: 0.2 - 0.4 m/year
- Layer slopes: 0 - 5°
- Bed tilt: 0 - 5°

Report results as ranges, not single values.

## Example Interpretation

**Given:**
- Horizontal velocity: 200 m/year (GPS)
- ApRES velocities: -0.5 to -1.5 m/year (nearly constant)
- Expected accumulation: 0.3 m/year

**Analysis:**
1. **Plug flow detected** (confidence 0.85)
   → Ice moves as coherent unit

2. **Mean layer slope**: 0.8° ± 0.3°
   → Small tilts, reasonable for ice interior

3. **Bed tilt required**: 1.2°
   → Consistent with nadir reflection

4. **Corrected vertical velocity**: -0.3 ± 0.1 m/year
   → Matches accumulation (steady state)

**Conclusion:**
✅ Ice is in **plug flow regime** with basal sliding on an unfrozen bed. Layer slopes are small and consistent. Bed reflection is likely from nadir. System is in approximate steady state with accumulation balancing vertical motion.

## Further Reading

- **Brennan et al. (2014)**: ApRES methodology
- **Kingslake et al. (2014)**: ApRES for basal conditions
- **Nicholls et al. (2015)**: Phase-sensitive measurements
- **Summers et al. (2021)**: Internal layer tracking (this dataset!)

## References for Mercer Subglacial Lake

- **Surface velocity**: ~185-200 m/year (GPS)
- **Ice thickness**: ~1094 m (radar)
- **Accumulation**: ~0.3 m/year ice equiv
- **Bed**: Subglacial lake (temperate, unfrozen)
- **Expected regime**: Sliding + possible deformation
