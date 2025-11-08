# Search Algorithms and Theory Verification

Author: Vasiliy Zdanovskiy  
email: vasilyvz@gmail.com

## Overview

This document describes how the search algorithms in the project verify specific predictions of the quench → tails model (see `Новая_теория.md`). The theory predicts oscillatory ring-like structures with algebraic decay, and the algorithms are designed to detect and characterize these structures.

## Theory Predictions (Key Points)

From Section C and H of `Новая_теория.md`:

1. **Algebraic oscillatory correlation in real space:**
   \[
   \xi_\Theta(r) \sim r^{-(3-2\beta)}\,\cos(\kappa_\star r+\varphi_\star)
   \]
   with preferred spacing \(\Delta r \simeq \pi/\kappa_\star\) and **no exponential envelopes**.

2. **Oscillatory structure in angular power spectrum:**
   - Spacing \(\Delta\ell \sim \pi r_\star \kappa_\star\)
   - Envelope slope relates to \(\beta\)

3. **Robustness to masking/windows:**
   - Changes in \(W(r)\) modulate \(\Delta_\ell(k)\) but leave intrinsic \(\kappa_\star\) spacing stable

## Algorithm 1: Structure Detector (`run_structure_detection`)

### What It Tests

**Direct verification of:** \(\xi_\Theta(r) \sim r^{-(3-2\beta)}\cos(\kappa_\star r+\varphi_\star)\)

### How It Works

1. **Radial Profile Computation** (`compute_radial_profile`):
   - For each candidate center, computes mean temperature profile \(T̄(r)\) in concentric annuli
   - This directly samples the radial correlation structure

2. **Oscillatory Template Matching** (`match_oscillatory_template`):
   - Matches detrended profile to \(\cos(\kappa r + \varphi)\) templates
   - Searches over grid of \(\kappa\) values (0.1 to 3.0 per degree)
   - **Algebraic weighting** \(1/r^p\) (default \(p=1.0\)) to account for \(r^{-(3-2\beta)}\) envelope
   - **No exponential templates** - consistent with "no mass terms" constraint

3. **Statistical Significance**:
   - Computes p-value using t-statistic
   - Filters detections with \(p < 10^{-3}\)

4. **Output Parameters**:
   - `best_kappa_per_deg`: Measures \(\kappa_\star\) (the preferred wavenumber)
   - `best_spacing_deg`: Computes \(\Delta r = \pi/\kappa_\star\) (the predicted spacing)
   - `amplitude`: Measures strength of oscillatory signal
   - `correlation`: Quality of fit to oscillatory template

### Theory Verification Points

✅ **Verifies oscillatory ripples**: Direct matching to \(\cos(\kappa r + \varphi)\)  
✅ **No exponential terms**: Only algebraic weighting \(1/r^p\) is used  
✅ **Measures \(\kappa_\star\)**: The key parameter from theory  
✅ **Computes spacing**: \(\Delta r = \pi/\kappa_\star\) as predicted  
✅ **Statistical rigor**: p-value filtering ensures significance

### Expected Results

If theory is correct, we should find:
- Multiple detections with consistent \(\kappa_\star\) values
- Spacing values \(\Delta r\) clustered around \(\pi/\kappa_\star\)
- Significant correlations (\(p < 10^{-3}\)) at multiple sky locations
- Robustness: same \(\kappa_\star\) found even with different masks/windows

---

## Algorithm 2: Multiscale Anomaly Search (`run_multiscale_anomaly_search`)

### What It Tests

**Indirect verification of:** Algebraic decay and scale-dependent structure

### How It Works

1. **Multi-Radius Scanning**:
   - Scans same sky locations with different radii (1° to 15°)
   - Computes anomaly metric \(S = |\mu - \mu_{global}| / \sigma_{global}\) for each radius

2. **Scale-Dependent Analysis**:
   - For each center, tracks how \(S\) varies with radius
   - Aggregates by center (keeps max \(|S|\) per center)

3. **Filtering**:
   - Filters by minimum pixel count (ensures sufficient statistics)
   - Filters by significance threshold (\(S > 5.0\))

### Theory Verification Points

✅ **Tests algebraic decay**: If \(\xi_\Theta(r) \sim r^{-(3-2\beta)}\), then anomalies should show scale-dependent behavior  
✅ **Multi-scale structure**: Different radii probe different parts of the correlation function  
✅ **Consistency check**: Same centers should show anomalies across multiple scales if theory is correct

### Expected Results

If theory is correct:
- Anomalies should persist across multiple radius scales
- \(S(r)\) curves should show structure (not random noise)
- Top anomalies should correlate with structure detector findings

---

## Algorithm 3: Region Matching (`find_similar_regions`)

### What It Tests

**Validation and cross-checking** of detected structures

### How It Works

1. **Coordinate Matching**:
   - Matches found anomalies to known regions from catalogs
   - Uses angular distance on sphere (spherical cosine formula)
   - Filters by radius tolerance (relative difference)

2. **Cross-Catalog Comparison** (`compare_anomaly_catalogs`):
   - Compares CMB anomalies with dust map anomalies
   - Tests if oscillatory structures appear in multiple observables

### Theory Verification Points

✅ **Robustness test**: Same structures should appear in different data sets  
✅ **Validation**: Known regions provide ground truth for testing  
✅ **Cross-observable consistency**: Theory predicts structures in CMB; dust correlation tests if they're real or artifacts

### Expected Results

If theory is correct:
- Structure detector findings should match known anomalous regions
- CMB and dust anomalies should show spatial correlation
- Consistent \(\kappa_\star\) values across different sky regions

---

## Combined Verification Strategy

### Step-by-Step Verification Process

1. **Structure Detection** → Find oscillatory patterns with \(\kappa_\star\)
2. **Multiscale Analysis** → Verify scale-dependent behavior
3. **Region Matching** → Cross-validate with known regions and other observables

### Key Metrics to Extract

From structure detector results:
- **Distribution of \(\kappa_\star\)**: Should cluster around a preferred value
- **Spacing distribution**: \(\Delta r = \pi/\kappa_\star\) should be consistent
- **Spatial distribution**: Detections should not be random but show patterns

From multiscale search:
- **Scale persistence**: Anomalies should appear at multiple radii
- **S(r) curves**: Should show oscillatory structure if theory is correct

From region matching:
- **Match rate**: High match rate with known regions supports theory
- **Cross-catalog correlation**: CMB-dust correlation validates findings

### Success Criteria

Theory is **supported** if:
1. Structure detector finds significant oscillatory patterns (\(p < 10^{-3}\))
2. Found \(\kappa_\star\) values cluster around a preferred value (not random)
3. Spacing \(\Delta r = \pi/\kappa_\star\) is consistent across detections
4. Same structures appear in multiscale analysis
5. Detections correlate with known anomalous regions
6. Results are robust to masking/window changes

Theory is **contradicted** if:
1. No significant oscillatory patterns found
2. Found \(\kappa_\star\) values are random
3. Spacing is inconsistent
4. Detections don't correlate with known regions
5. Results are highly sensitive to masking (suggesting artifacts)

---

## Implementation Details

### Structure Detector Parameters

```python
kappa_min_per_deg = 0.1    # Minimum κ to search (1/deg)
kappa_max_per_deg = 3.0    # Maximum κ to search (1/deg)
kappa_num = 200            # Grid resolution
power_weight_p = 1.0       # Algebraic weighting exponent (1/r^p)
p_value_threshold = 1e-3   # Significance threshold
```

These parameters are chosen to:
- Cover the expected range of \(\kappa_\star\) values
- Use algebraic weighting consistent with theory
- Ensure statistical significance

### Algebraic Weighting Rationale

The `power_weight_p` parameter implements the \(r^{-(3-2\beta)}\) envelope:
- Default \(p=1.0\) corresponds to \(\beta \approx 1\) (intermediate case)
- Weighting by \(1/r^p\) emphasizes near-field structure while allowing far-field detection
- **No exponential weighting** - consistent with "no mass terms" constraint

---

## References

- Theory: `docs/Новая_теория.md` (Section C, H)
- Implementation: `cmb_anomaly/structure_detector.py`
- Multiscale search: `cmb_anomaly/multiscale.py`
- Region matching: `cmb_anomaly/region_match.py`

