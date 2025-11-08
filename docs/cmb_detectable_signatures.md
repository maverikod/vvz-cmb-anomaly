# What Can We Detect on CMB Background?

Author: Vasiliy Zdanovskiy  
email: vasilyvz@gmail.com

## Overview

This document describes **what specific structures and patterns** we can detect on the CMB background, given the window function limitations. We focus on **intermediate-scale oscillatory structures** that are observable in CMB temperature maps.

## Observable Scales in CMB

### Angular Scale Range

**Visible in CMB** (resolved by projection window \(W(r)\)):
- \(\kappa \in [0.1, 3.0]\) per degree
- Spacing: \(\Delta r \in [1°, 31°]\) on sky
- Angular multipoles: \(\ell \sim 6-180\) (rough correspondence)

**Why these scales?**
- Window function \(W(r)\) can resolve structures of this size
- Transfer function \(\mathcal T(k)\) couples these modes to temperature
- Observable in Planck/WMAP data with sufficient signal-to-noise

## What We're Looking For

### 1. Ring-Like Oscillatory Structures

**Prediction from theory** (Section C):
\[
\xi_\Theta(r) \sim r^{-(3-2\beta)}\,\cos(\kappa_\star r+\varphi_\star)
\]

**What this means on the sky:**
- **Concentric rings** of temperature fluctuations around centers
- **Preferred spacing** \(\Delta r = \pi/\kappa_\star\) between rings
- **Algebraic decay** (not exponential) with distance from center
- **No exponential envelope** (key difference from standard models)

**Example**: If \(\kappa_\star = 0.5\) per degree:
- Spacing: \(\Delta r = \pi/0.5 \approx 6.3°\)
- Around a center, we'd see rings at ~6°, ~12°, ~18°, etc.
- Amplitude decreases as \(r^{-(3-2\beta)}\) (algebraic, not exponential)

### 2. Radial Profile Oscillations

**What the structure detector finds:**

For each sky location (center), we compute:
- Mean temperature in concentric annuli (rings)
- Radial profile: \(T̄(r)\) vs radius \(r\)

**If theory is correct**, this profile should show:
- **Oscillatory pattern**: \(T̄(r) \sim \cos(\kappa_\star r + \varphi)\)
- **Algebraic envelope**: Amplitude decreases as \(r^{-(3-2\beta)}\)
- **Consistent spacing**: Same \(\Delta r\) across different centers

### 3. Consistent Preferred Wavenumber \(\kappa_\star\)

**Key prediction**: The theory predicts a **preferred wavenumber** \(\kappa_\star\) that should appear:
- At **multiple sky locations** (not random)
- With **consistent spacing** \(\Delta r = \pi/\kappa_\star\)
- **Robust to masking** (window changes don't affect intrinsic \(\kappa_\star\))

**What we measure:**
- `best_kappa_per_deg`: The \(\kappa\) value that best fits each center
- If theory is correct: **clustering** of \(\kappa\) values around \(\kappa_\star\)
- If random noise: **uniform distribution** of \(\kappa\) values

## Specific Detectable Signatures

### A. Localized Ring Structures

**What**: Concentric temperature rings around specific sky locations

**Scale**: 
- Ring spacing: 1° to 31° (depending on \(\kappa_\star\))
- Total extent: Up to ~15-30° radius (limited by window function)

**Detection method**: Structure detector (`run_structure_detection`)
- Computes radial profile around candidate centers
- Matches to \(\cos(\kappa r + \varphi)\) templates
- Statistical significance: \(p < 10^{-3}\)

**Expected if theory correct:**
- Multiple detections with similar \(\kappa_\star\)
- Rings visible in temperature maps around detection centers
- Spacing consistent across detections

### B. Anomalous Regions (Multiscale)

**What**: Sky regions with unusual temperature statistics

**Scale**: Multiple radii (1° to 15°)

**Detection method**: Multiscale anomaly search (`run_multiscale_anomaly_search`)
- Computes anomaly metric \(S = |\mu - \mu_{global}| / \sigma_{global}\)
- Tests multiple radii for each center
- Filters by significance (\(S > 5.0\))

**Expected if theory correct:**
- Anomalies persist across multiple radius scales
- \(S(r)\) curves show structure (not random)
- Correlate with structure detector findings

### C. Oscillatory Angular Power Spectrum

**What**: Oscillatory structure in \(C_\ell\) (angular power spectrum)

**Prediction** (Section E):
- Spacing: \(\Delta\ell \sim \pi r_\star \kappa_\star\)
- For \(\kappa_\star \sim 0.5-1.0\) per degree: \(\Delta\ell \sim 15-30\)

**Detection method**: Analyze Planck \(C_\ell\) data
- Look for oscillatory peaks in \(C_\ell\)
- Measure spacing between peaks
- Compare with \(\kappa_\star\) from structure detector

**Note**: This is a **complementary test** to spatial correlation analysis.

## What We CANNOT Detect (Window Limitations)

### Very Long Wavelengths

**Not visible in CMB:**
- \(\kappa < 0.1\) per degree
- Spacing \(\Delta r > 31°\)
- These are suppressed by window function \(W(r)\)

**Where to look**: Large-scale structure (LSS) surveys

### Very Short Wavelengths

**Limited by:**
- Instrumental noise
- Beam resolution
- HEALPix pixelization

**Practical limit**: \(\kappa > 3.0\) per degree may be noise-dominated

## Expected Results if Theory is Correct

### Structure Detector Results

1. **Multiple significant detections** (\(p < 10^{-3}\))
2. **Clustered \(\kappa_\star\) values** (not uniform distribution)
3. **Consistent spacing** \(\Delta r = \pi/\kappa_\star\) across detections
4. **Spatial clustering**: Detections not randomly distributed on sky

### Statistical Distribution

**If theory correct:**
- Histogram of `best_kappa_per_deg` shows **peak** at \(\kappa_\star\)
- Histogram of `best_spacing_deg` shows **peak** at \(\Delta r = \pi/\kappa_\star\)
- Correlation values clustered around significant values

**If random noise:**
- Uniform distribution of \(\kappa\) values
- No preferred spacing
- Low correlations, high p-values

### Spatial Patterns

**If theory correct:**
- Detections may show **spatial clustering** (related to underlying structure)
- **Robustness**: Same \(\kappa_\star\) found with different masks/windows
- **Cross-correlation**: CMB detections correlate with known anomalous regions

## Practical Detection Strategy

### Step 1: Structure Detection

Run `run_structure_detection` with:
- \(\kappa \in [0.1, 3.0]\) per degree
- \(r_{max} = 15-30°\)
- Statistical threshold: \(p < 10^{-3}\)

**Output**: List of detections with \(\kappa_\star\), spacing, amplitude, correlation

### Step 2: Statistical Analysis

Analyze detection results:
- **Distribution of \(\kappa_\star\)**: Look for clustering
- **Distribution of spacing**: Check consistency
- **Spatial distribution**: Are detections clustered or random?

### Step 3: Validation

- **Cross-check**: Compare with multiscale anomaly search
- **Known regions**: Match with catalog of known anomalies
- **Robustness**: Test with different masks/windows

### Step 4: Angular Power Spectrum

- Analyze \(C_\ell\) for oscillatory structure
- Measure spacing \(\Delta\ell\)
- Compare with \(\kappa_\star\) from structure detector

## Summary: What We Can Catch

### ✅ Detectable in CMB:

1. **Ring-like oscillatory structures**
   - Spacing: 1° to 31°
   - Around specific sky locations
   - Algebraic decay (not exponential)

2. **Preferred wavenumber \(\kappa_\star\)**
   - Should cluster around specific value
   - Consistent across multiple detections
   - Robust to masking

3. **Anomalous regions**
   - Unusual temperature statistics
   - Persist across multiple scales
   - Correlate with oscillatory structures

4. **Oscillatory \(C_\ell\)** (if present)
   - Spacing \(\Delta\ell \sim \pi r_\star \kappa_\star\)
   - Complementary to spatial analysis

### ❌ NOT Detectable in CMB:

1. **Very long wavelengths** (\(\kappa < 0.1\), \(\Delta r > 31°\))
   - Suppressed by window function
   - Need LSS data instead

2. **Very short wavelengths** (\(\kappa > 3.0\), \(\Delta r < 1°\))
   - Limited by noise and resolution
   - May be observable with better data

## Key Testable Predictions

1. **Oscillatory correlation**: \(\xi(r) \sim r^{-(3-2\beta)}\cos(\kappa_\star r + \varphi)\)
2. **No exponential envelope**: Pure algebraic decay
3. **Preferred spacing**: \(\Delta r = \pi/\kappa_\star\) consistent across sky
4. **Robustness**: \(\kappa_\star\) stable under window/mask changes

If we find these signatures, they provide **strong support** for the quench → tails model.

