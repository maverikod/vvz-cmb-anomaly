# CMB vs Large-Scale Structure: Observability of Long Wavelengths

Author: Vasiliy Zdanovskiy  
email: vasilyvz@gmail.com

## Critical Question

**Will very long wavelength oscillations (from quench) be visible in CMB, or are they better suited for density waves (large-scale structure)?**

## Theory: CMB Projection (Section D)

The CMB temperature fluctuation is a **line-of-sight projection**:

\[
\frac{\delta T}{T}(\hat{\boldsymbol n})
\;=\;\int_0^\infty dr\ W(r)\ \int\frac{d^3k}{(2\pi)^3}\ e^{i\boldsymbol k\cdot r\hat{\boldsymbol n}}\ \mathcal T(k)\ \delta\Theta_{\boldsymbol k}
\]

Where:
- \(W(r)\) is a **narrow window** around \(\Sigma_\star\) (last coherence layer)
- \(\mathcal T(k)\) is the **transfer function** from phase fluctuations to temperature

## Key Insight: Window Function Suppression

### A. Narrow Window \(W(r)\) Suppresses Long Wavelengths

If \(W(r)\) is **narrow** (concentrated around \(\Sigma_\star\)):
- **Short wavelengths** (large \(k\)): Many oscillations within window → **visible**
- **Long wavelengths** (small \(k\)): Wavelength >> window width → **suppressed**

**Physical reason**: A narrow window cannot resolve structures much larger than itself.

### B. Transfer Function \(\mathcal T(k)\) May Filter Low \(k\)

The transfer function \(\mathcal T(k)\) depends on the medium properties. It may:
- **Suppress very low \(k\)** modes (if they don't couple to intensity proxy)
- **Enhance intermediate \(k\)** modes (resonant with medium properties)

## Angular Power Spectrum Analysis (Section E)

\[
C_\ell = 4\pi\int_0^\infty dk\,k^2\ P_{\Theta}(k)\ \big|\Delta_\ell(k)\big|^2
\]

Where:
\[
\Delta_\ell(k)=\int_0^\infty dr\ W(r)\,\mathcal T(k)\,j_\ell(kr)
\]

**For very long wavelengths** (small \(k\)):
- Bessel function \(j_\ell(kr)\) is small when \(kr \ll \ell\)
- If \(W(r)\) is narrow, integration range is limited
- **Result**: Very long wavelengths contribute weakly to \(C_\ell\) at low \(\ell\)

## Conclusion: Long Wavelengths → Large-Scale Structure

### CMB Observability

**Short to intermediate wavelengths** (corresponding to \(\ell \sim 10-1000\)):
- ✅ **Visible in CMB**
- Window \(W(r)\) can resolve them
- Transfer function \(\mathcal T(k)\) couples them to temperature

**Very long wavelengths** (corresponding to \(\ell < 10\), super-horizon scales):
- ❌ **Suppressed in CMB**
- Wavelength >> window width
- May be filtered by \(\mathcal T(k)\)

### Large-Scale Structure (LSS) Observability

**Very long wavelengths in phase field**:
- ✅ **Should manifest as density waves** in LSS
- Density fluctuations \(\delta\rho/\rho\) directly trace phase field
- No window function suppression (3D structure, not 2D projection)

**Physical picture**:
- Long-wavelength phase oscillations → **large-scale density modulations**
- These create **superclusters, voids, filaments** on scales >> 100 Mpc
- **Baryon Acoustic Oscillations (BAO)** may be related, but quench model predicts different spacing

## Implications for Search Strategy

### 1. CMB Search: Focus on Intermediate Scales

**Recommended range for CMB**:
- \(\kappa \in [0.1, 3.0]\) per degree
- \(\Delta r \in [1°, 31°]\) on sky
- Corresponds to \(\ell \sim 10-300\) in angular power spectrum

**Why**: These scales are:
- Resolved by CMB window function
- Coupled by transfer function
- Observable in current CMB maps

### 2. LSS Search: Long Wavelengths

**For very long wavelengths** (\(\kappa < 0.1\), \(\Delta r > 31°\)):
- ❌ **Not effective in CMB** (window suppression)
- ✅ **Should search in LSS data**:
  - Galaxy surveys (SDSS, DES, etc.)
  - 3D correlation function \(\xi(r)\)
  - Power spectrum \(P(k)\) at low \(k\)

### 3. Hybrid Approach

**Test theory comprehensively**:
1. **CMB**: Search for intermediate-scale oscillations (\(\kappa \sim 0.1-3.0\))
2. **LSS**: Search for long-wavelength density waves (\(\kappa < 0.1\))
3. **Cross-correlation**: Check if CMB and LSS show consistent \(\kappa_\star\) values

## Revised Search Parameters

### For CMB (Current Focus)

```python
# Intermediate scales - visible in CMB
kappa_min_per_deg = 0.1    # Δr ≈ 31° (ℓ ~ 6)
kappa_max_per_deg = 3.0     # Δr ≈ 1° (ℓ ~ 180)
max_radius_deg = 15-30°    # Sufficient for intermediate scales
```

**Rationale**: These scales are:
- Resolved by CMB projection window
- Not suppressed by transfer function
- Observable in Planck/WMAP data

### For LSS (Future Work)

```python
# Long wavelengths - density waves in LSS
kappa_min_per_deg = 0.001  # Δr ≈ 3140° (super-large scale)
kappa_max_per_deg = 0.1    # Δr ≈ 31° (still large)
# 3D correlation function analysis
```

## Angular Power Spectrum Test

**Direct test**: Analyze \(C_\ell\) for oscillatory structure:

From theory (Section E):
- Spacing: \(\Delta\ell \sim \pi r_\star \kappa_\star\)
- For intermediate \(\kappa_\star \sim 0.5-1.0\) per degree:
  - \(\Delta\ell \sim 15-30\) (if \(r_\star \sim 10\))
- **This is testable in Planck \(C_\ell\) data**

**If long wavelengths dominate**:
- Would see oscillations at very low \(\ell\) (\(\ell < 10\))
- But these are **cosmic variance limited** and may be suppressed by window

## Summary

**Your intuition is correct**: Very long wavelength oscillations from quench are **better suited for density waves (LSS) than CMB**.

**CMB is sensitive to**:
- Intermediate scales (\(\kappa \sim 0.1-3.0\) per degree)
- Scales resolved by projection window \(W(r)\)
- Modes coupled by transfer function \(\mathcal T(k)\)

**LSS is sensitive to**:
- Very long wavelengths (\(\kappa < 0.1\) per degree)
- Large-scale density modulations
- 3D structure (no window suppression)

**Current search strategy should**:
1. ✅ Keep intermediate-scale search in CMB (\(\kappa \geq 0.1\))
2. ❌ **Not extend to very long wavelengths in CMB** (window suppression)
3. ✅ **Add LSS analysis** for long wavelengths (future work)

