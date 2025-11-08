# Correct Search Strategy: Base Substrate vs CMB Projection

Author: Vasiliy Zdanovskiy  
email: vasilyvz@gmail.com

## Key Insight

**ВБП (высокочастотная базовая подложка) = Base Phase Substrate Θ(x, φ, t)**

This is the **fundamental substrate** that structures everything. CMB is just a **projection** through a window function.

## Theory: What We Should Search

### A. Base Substrate Correlation (Section C)

The **fundamental structure** is in the phase field correlation:

\[
\xi_\Theta(r) \equiv \langle \delta\Theta(x)\delta\Theta(x+r)\rangle
\ \sim\ r^{-(3-2\beta)}\,\cos(\kappa_\star r+\varphi_\star)
\]

This is the **direct signature** of the quench in the base substrate.

### B. CMB is a Projection (Section D)

CMB temperature is **not** the substrate itself:

\[
\frac{\delta T}{T}(\hat{\boldsymbol n})
\;=\;\int_0^\infty dr\ W(r)\ \int\frac{d^3k}{(2\pi)^3}\ e^{i\boldsymbol k\cdot r\hat{\boldsymbol n}}\ \mathcal T(k)\ \delta\Theta_{\boldsymbol k}
\]

Where:
- \(W(r)\) = narrow window (suppresses long wavelengths)
- \(\mathcal T(k)\) = transfer function (filters modes)

## Problem with Current Approach

### What We're Doing Now

**Local radial profiles** around centers:
- Compute \(T̄(r)\) in concentric annuli
- Match to \(\cos(\kappa r + \varphi)\) templates
- **Problem**: This is looking at **projected temperature**, not the **base substrate structure**

### What We Should Do Instead

**Two-point correlation function** of CMB:

\[
\xi_{CMB}(\theta) = \langle \delta T/T(\hat{\boldsymbol n}_1) \cdot \delta T/T(\hat{\boldsymbol n}_2) \rangle
\]

Where \(\theta\) is the angular separation between directions \(\hat{\boldsymbol n}_1\) and \(\hat{\boldsymbol n}_2\).

**Why this is better**:
- Directly probes the correlation structure
- Less affected by local noise
- Should show oscillatory pattern: \(\xi_{CMB}(\theta) \sim \theta^{-(3-2\beta)}\cos(\kappa_\star \theta + \varphi_\star)\)
- **Global pattern**, not local structures

## Correct Search Strategy

### Method 1: Two-Point Correlation Function

**Compute full-sky correlation function**:

```python
def compute_cmb_correlation_function(temperature_map, theta_bins):
    """
    Compute two-point correlation function ξ(θ) for CMB map.
    
    Args:
        temperature_map: Full-sky CMB temperature map
        theta_bins: Angular separation bins (degrees)
    
    Returns:
        xi_theta: Correlation function values
        theta_centers: Bin centers
    """
    # For each pair of pixels, compute:
    # - Angular separation θ
    # - Product δT/T(1) * δT/T(2)
    # - Average over all pairs with same θ
    # This directly probes ξ_Θ structure
```

**What to look for**:
- Oscillatory pattern: \(\xi(\theta) \sim \cos(\kappa_\star \theta + \varphi_\star)\)
- Algebraic envelope: \(\sim \theta^{-(3-2\beta)}\)
- **Preferred spacing**: \(\Delta\theta = \pi/\kappa_\star\)

### Method 2: Angular Power Spectrum \(C_\ell\)

**Direct test from theory** (Section E):

\[
C_\ell = 4\pi\int_0^\infty dk\,k^2\ P_{\Theta}(k)\ \big|\Delta_\ell(k)\big|^2
\]

**Oscillatory structure** in \(C_\ell\):
- Spacing: \(\Delta\ell \sim \pi r_\star \kappa_\star\)
- Envelope: \(C_\ell \propto \ell^{-(3-2\beta)}\) (intermediate \(\ell\))

**This is what we should be looking for**, not local ring structures!

### Method 3: Fourier Analysis

**3D power spectrum** \(P(k)\) (if we had 3D data):
- Directly probes \(P_\Theta(k) \propto |\beta_k|^2\)
- Shows preferred wavenumber \(\kappa_\star\)

**For CMB (2D projection)**:
- Angular power spectrum \(C_\ell\) is the 2D analog
- Should show oscillatory structure

## Why Local Radial Profiles Are Wrong

### Current Approach Problems

1. **Local structures** may be:
   - Noise artifacts
   - Galactic contamination
   - Instrumental effects
   - Not the fundamental substrate structure

2. **Radial profiles** around centers:
   - Depend on choice of centers
   - May miss global patterns
   - Sensitive to local anomalies

3. **Missing the big picture**:
   - Theory predicts **global correlation structure**
   - Not local ring-like features
   - The substrate structures **everything**, not just local regions

## What Theory Actually Predicts

### Global Correlation Structure

\[
\xi_\Theta(r) \sim r^{-(3-2\beta)}\,\cos(\kappa_\star r+\varphi_\star)
\]

This is a **statistical property** of the entire field, not a local feature.

**In CMB projection**:
- Should manifest as oscillatory **two-point correlation function** \(\xi_{CMB}(\theta)\)
- Should show up in **angular power spectrum** \(C_\ell\) as oscillatory ripples
- **Not** as local ring structures around specific centers

## Recommended Approach

### Step 1: Two-Point Correlation Function

Compute \(\xi_{CMB}(\theta)\) for full sky:
- Angular separations: 0.1° to 180°
- Look for oscillatory pattern
- Measure spacing \(\Delta\theta = \pi/\kappa_\star\)

### Step 2: Angular Power Spectrum

Analyze \(C_\ell\) from Planck data:
- Look for oscillatory ripples
- Measure spacing \(\Delta\ell\)
- Compare with \(\kappa_\star\) from correlation function

### Step 3: Cross-Validation

- Correlation function and power spectrum should give **consistent** \(\kappa_\star\)
- Both should show **algebraic envelope** (not exponential)
- **Global patterns**, not local features

## Implementation

### New Function Needed:

```python
def compute_cmb_correlation_function(
    temperature_map: np.ndarray,
    mask: Optional[np.ndarray] = None,
    theta_min_deg: float = 0.1,
    theta_max_deg: float = 180.0,
    n_bins: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute two-point correlation function ξ(θ) for CMB map.
    
    This directly probes the base substrate correlation structure
    ξ_Θ(r) ~ r^-(3-2β) cos(κ_*r + φ_*)
    
    Returns:
        theta_centers: Angular separation bin centers (degrees)
        xi_theta: Correlation function values
    """
```

### Analysis of Results

1. **Fit oscillatory template**: \(\xi(\theta) = A \theta^{-\alpha} \cos(\kappa \theta + \varphi)\)
2. **Extract**:
   - Preferred wavenumber \(\kappa_\star\)
   - Spacing \(\Delta\theta = \pi/\kappa_\star\)
   - Envelope exponent \(\alpha = 3-2\beta\)
3. **Compare with theory predictions**

## Summary

**Current approach (local radial profiles)**:
- ❌ Looks at projected temperature locally
- ❌ May find artifacts, not substrate structure
- ❌ Misses global correlation pattern

**Correct approach (correlation function)**:
- ✅ Directly probes base substrate correlation
- ✅ Global statistical property
- ✅ Matches theory prediction: \(\xi(r) \sim r^{-(3-2\beta)}\cos(\kappa_\star r + \varphi_\star)\)
- ✅ Less sensitive to local noise/artifacts

**The base substrate structures everything through correlation, not through local features!**

