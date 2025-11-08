# Correct Search Method: Correlation Function vs Local Profiles

Author: Vasiliy Zdanovskiy  
email: vasilyvz@gmail.com

## Key Insight from Theory

**ВБП (высокочастотная базовая подложка) = Base Phase Substrate Θ(x, φ, t)**

This is the **fundamental substrate** that structures everything. The structure is in the **correlation function**, not in local features.

## Theory: What We Should Search

### A. Base Substrate Correlation (Section C, Новая_теория.md)

The **fundamental structure** is in the phase field correlation:

\[
\xi_\Theta(r) \equiv \langle \delta\Theta(x)\delta\Theta(x+r)\rangle
\ \sim\ r^{-(3-2\beta)}\,\cos(\kappa_\star r+\varphi_\star)
\]

This is a **statistical property** of the entire field, not a local feature.

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

### What We're Doing Now (WRONG)

**Local radial profiles** around centers:
- Compute \(T̄(r)\) in concentric annuli around each center
- Match to \(\cos(\kappa r + \varphi)\) templates
- **Problem**: This looks at **local temperature patterns**, not the **base substrate correlation structure**

**Why this is wrong**:
- Local structures may be noise, artifacts, or galactic contamination
- Misses the **global statistical property** that theory predicts
- The substrate structures **everything through correlation**, not through local ring features

### What We Should Do Instead (CORRECT)

**Two-point correlation function** of CMB:

\[
\xi_{CMB}(\theta) = \langle \delta T/T(\hat{\boldsymbol n}_1) \cdot \delta T/T(\hat{\boldsymbol n}_2) \rangle
\]

Where \(\theta\) is the angular separation between directions \(\hat{\boldsymbol n}_1\) and \(\hat{\boldsymbol n}_2\).

**Why this is correct**:
- Directly probes the correlation structure of the base substrate
- Less affected by local noise/artifacts
- Should show oscillatory pattern: \(\xi_{CMB}(\theta) \sim \theta^{-(3-2\beta)}\cos(\kappa_\star \theta + \varphi_\star)\)
- **Global statistical property**, not local structures
- Matches theory prediction: \(\xi_\Theta(r) \sim r^{-(3-2\beta)}\cos(\kappa_\star r + \varphi_\star)\)

## Correct Search Strategy

### Method 1: Two-Point Correlation Function (PRIMARY)

**Compute full-sky correlation function**:

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
    
    For CMB projection, this becomes:
    ξ_CMB(θ) ~ θ^-(3-2β) cos(κ_*θ + φ_*)
    
    Args:
        temperature_map: Full-sky CMB temperature map (HEALPix)
        mask: Optional mask (exclude masked pixels)
        theta_min_deg: Minimum angular separation (degrees)
        theta_max_deg: Maximum angular separation (degrees)
        n_bins: Number of angular separation bins
    
    Returns:
        theta_centers: Angular separation bin centers (degrees)
        xi_theta: Correlation function values
    """
    # For all pairs of pixels:
    # 1. Compute angular separation θ
    # 2. Compute product δT/T(pix1) * δT/T(pix2)
    # 3. Average over all pairs with same θ
    # This directly samples ξ_Θ structure through CMB projection
```

**What to look for**:
- Oscillatory pattern: \(\xi(\theta) \sim \cos(\kappa_\star \theta + \varphi_\star)\)
- Algebraic envelope: \(\sim \theta^{-(3-2\beta)}\)
- **Preferred spacing**: \(\Delta\theta = \pi/\kappa_\star\)
- **No exponential envelope** (key test!)

### Method 2: Angular Power Spectrum \(C_\ell\) (COMPLEMENTARY)

From theory (Section E):

\[
C_\ell = 4\pi\int_0^\infty dk\,k^2\ P_{\Theta}(k)\ \big|\Delta_\ell(k)\big|^2
\]

**Oscillatory structure** in \(C_\ell\):
- Spacing: \(\Delta\ell \sim \pi r_\star \kappa_\star\)
- Envelope: \(C_\ell \propto \ell^{-(3-2\beta)}\) (intermediate \(\ell\))

**This is what we should be analyzing**, not local ring structures!

### Method 3: Cross-Validation

- Correlation function and power spectrum should give **consistent** \(\kappa_\star\)
- Both should show **algebraic envelope** (not exponential)
- **Global patterns**, not local features

## Why Local Radial Profiles Are Wrong

### Current Approach Problems

1. **Local structures** may be:
   - Noise artifacts
   - Galactic contamination
   - Instrumental effects
   - **Not the fundamental substrate structure**

2. **Radial profiles** around centers:
   - Depend on arbitrary choice of centers
   - May miss global patterns
   - Sensitive to local anomalies
   - **Not what theory predicts**

3. **Missing the big picture**:
   - Theory predicts **global correlation structure** \(\xi_\Theta(r)\)
   - Not local ring-like features around specific centers
   - The substrate structures **everything through correlation**, not through local regions

## What Theory Actually Predicts

### Global Correlation Structure

\[
\xi_\Theta(r) \sim r^{-(3-2\beta)}\,\cos(\kappa_\star r+\varphi_\star)
\]

This is a **statistical property** of the entire field.

**In CMB projection**:
- Should manifest as oscillatory **two-point correlation function** \(\xi_{CMB}(\theta)\)
- Should show up in **angular power spectrum** \(C_\ell\) as oscillatory ripples
- **NOT** as local ring structures around specific centers

## Implementation Plan

### Step 1: Implement Correlation Function

Create new module: `cmb_anomaly/correlation_function.py`

```python
def compute_cmb_correlation_function(
    temperature: np.ndarray,
    mask: Optional[np.ndarray] = None,
    theta_bins: np.ndarray = None,
    n_jobs: int = None,
    use_gpu: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute two-point correlation function ξ(θ) for CMB.
    
    Directly probes base substrate correlation structure.
    """
    # Implementation:
    # 1. For all pixel pairs, compute angular separation θ
    # 2. Compute product δT/T(pix1) * δT/T(pix2)
    # 3. Bin by θ and average
    # 4. Return θ_centers and ξ(θ)
```

### Step 2: Fit Oscillatory Template

```python
def fit_correlation_oscillations(
    theta: np.ndarray,
    xi: np.ndarray,
    kappa_grid: np.ndarray,
    beta_range: Tuple[float, float] = (0.5, 1.5),
) -> Dict:
    """
    Fit ξ(θ) = A θ^-(3-2β) cos(κθ + φ) to correlation function.
    
    Returns:
        best_kappa: Preferred wavenumber κ_*
        best_beta: Envelope exponent β
        best_amplitude: A
        best_phase: φ
        correlation: Quality of fit
    """
```

### Step 3: Compare with Angular Power Spectrum

```python
def compute_angular_power_spectrum(
    temperature: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute C_ℓ from CMB map.
    
    Look for oscillatory structure with spacing Δℓ ~ π r_* κ_*
    """
```

## Summary

**Current approach (local radial profiles)**:
- ❌ Looks at projected temperature locally
- ❌ May find artifacts, not substrate structure
- ❌ Misses global correlation pattern
- ❌ **Not what theory predicts**

**Correct approach (correlation function)**:
- ✅ Directly probes base substrate correlation
- ✅ Global statistical property
- ✅ Matches theory prediction: \(\xi(r) \sim r^{-(3-2\beta)}\cos(\kappa_\star r + \varphi_\star)\)
- ✅ Less sensitive to local noise/artifacts
- ✅ **This is what theory actually predicts**

**The base substrate structures everything through correlation, not through local features!**

## Next Steps

1. **Implement** `compute_cmb_correlation_function()`
2. **Analyze** \(\xi_{CMB}(\theta)\) for oscillatory structure
3. **Extract** \(\kappa_\star\) from correlation function
4. **Compare** with angular power spectrum \(C_\ell\)
5. **Validate** that both give consistent \(\kappa_\star\)

This is the **correct way** to test the theory!

