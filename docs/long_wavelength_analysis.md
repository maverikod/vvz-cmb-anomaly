# Long Wavelength Oscillations After Big Quench

Author: Vasiliy Zdanovskiy  
email: vasilyvz@gmail.com

## Theoretical Prediction: Strong Long-Wavelength Excitation

### A. Quench Condition and Long Wavelengths

From the theory (Section B of `Новая_теория.md`):

The quench occurs on time scale \(\tau_q\) with condition:
\[
\tau_q \ll \omega_k^{-1} \quad \text{for a broad k-band}
\]

For **long wavelengths** (small \(k\)):
- Frequency: \(\omega_k \approx c_\phi k\) (dominant term for small \(k\))
- Condition: \(\tau_q \ll (c_\phi k)^{-1}\) → **easier to satisfy for small \(k\)**
- **Conclusion: Long wavelengths are MORE strongly excited by the quench**

### B. Occupation Number for Long Wavelengths

From Bogoliubov coefficients (Section B):
\[
|\beta_k|^2 = \frac{1}{4}\left(\frac{\omega_k^{(+)}}{\omega_k^{(-)}}-\frac{\omega_k^{(-)}}{\omega_k^{(+)}}\right)^2
\]

For small \(k\):
- \(\omega_k^{(\pm)} \approx c_\phi k\) (both before and after quench)
- If quench changes higher-order terms (\(\alpha_4, \alpha_6, \ldots\)) but not \(c_\phi\):
  - Ratio \(\omega_k^{(+)}/\omega_k^{(-)} \approx 1\) for very small \(k\)
  - But if quench is **strong** (large change in parameters), even small \(k\) modes get excited

**Key insight**: For a **global quench** (Big Bang), the parameter change affects **all scales**, so long wavelengths should indeed be strongly excited.

### C. Spatial Correlation: Long Wavelengths = Large Spacing

From Section C:
\[
\xi_\Theta(r) \sim r^{-(3-2\beta)}\,\cos(\kappa_\star r+\varphi_\star)
\]

Spacing: \(\Delta r \simeq \pi/\kappa_\star\)

For **long wavelengths** (small \(\kappa_\star\)):
- **Large spacing** \(\Delta r\) (can be tens or hundreds of degrees)
- These are **large-scale oscillatory structures** across the sky

## Current Search Parameters: Gap for Long Wavelengths

### Current Settings

From `structure_detector.py`:
```python
kappa_min_per_deg = 0.1    # Minimum κ (1/deg)
kappa_max_per_deg = 3.0     # Maximum κ (1/deg)
max_radius_deg = 15.0      # Maximum radius for profile
```

**Corresponding spacings:**
- \(\kappa = 0.1\) → \(\Delta r = \pi/0.1 \approx 31.4°\)
- \(\kappa = 3.0\) → \(\Delta r = \pi/3.0 \approx 1.05°\)

### Problem: Missing Very Long Wavelengths

**Very long wavelengths** would have:
- \(\kappa < 0.1\) (e.g., \(\kappa = 0.01\) → \(\Delta r \approx 314°\))
- These structures span **most or all of the sky**
- Current search **cannot detect them** because:
  1. `kappa_min = 0.1` is too high
  2. `max_radius_deg = 15°` is too small to resolve long-wavelength patterns

### Why This Matters

According to theory:
1. **Long wavelengths are MORE excited** by the quench (easier to satisfy \(\tau_q \ll \omega_k^{-1}\))
2. **Large-scale structures** should be the **dominant signature** of the quench
3. **Missing them** means we're not testing the full prediction

## Recommended Parameter Extensions

### For Long-Wavelength Search

```python
# Extended range for long wavelengths
kappa_min_per_deg = 0.01   # Down to 0.01 (Δr ≈ 314°)
kappa_max_per_deg = 3.0     # Keep current max

# Extended radius to resolve long patterns
max_radius_deg = 60.0       # Or even 90° (hemisphere)
bin_size_deg = 1.0          # Larger bins for long wavelengths
```

### Two-Stage Search Strategy

**Stage 1: Long wavelengths** (large-scale structures)
- \(\kappa \in [0.01, 0.1]\) → \(\Delta r \in [31°, 314°]\)
- `max_radius_deg = 60-90°`
- `bin_size_deg = 1.0-2.0°`

**Stage 2: Short wavelengths** (current search)
- \(\kappa \in [0.1, 3.0]\) → \(\Delta r \in [1°, 31°]\)
- `max_radius_deg = 15°`
- `bin_size_deg = 0.5°`

## Expected Signatures of Long Wavelengths

### If Theory is Correct:

1. **Strong detections** at small \(\kappa\) (0.01-0.1)
2. **Large spacing** values (\(\Delta r > 30°\))
3. **Sky-wide patterns** visible in temperature maps
4. **Consistent \(\kappa_\star\)** across different sky regions

### Observational Challenges

1. **Sky coverage**: Need full-sky or large-area maps
2. **Masking**: Large structures may be affected by galactic mask
3. **Resolution**: HEALPix NSIDE must be sufficient to resolve patterns
4. **Computational cost**: Larger radius → more pixels → slower computation

## Implementation Recommendations

### 1. Add Long-Wavelength Search Mode

```python
def run_structure_detection_long_wavelength(
    temperature: np.ndarray,
    mask: Optional[np.ndarray] = None,
    kappa_min_per_deg: float = 0.01,  # Extended minimum
    kappa_max_per_deg: float = 0.1,   # Focus on long wavelengths
    max_radius_deg: float = 60.0,     # Extended radius
    bin_size_deg: float = 1.0,         # Larger bins
    ...
) -> List[DetectionResult]:
    """Search for large-scale oscillatory structures."""
```

### 2. Full-Sky Correlation Analysis

Instead of radial profiles, compute:
- **Two-point correlation function** \(\xi(\theta)\) over full sky
- **Fourier transform** to find preferred \(\kappa\)
- **Spherical harmonic analysis** to find large-scale modes

### 3. Angular Power Spectrum Analysis

From Section E of theory:
- Oscillatory structure in \(C_\ell\) with spacing \(\Delta\ell \sim \pi r_\star \kappa_\star\)
- For long wavelengths: **small \(\kappa_\star\)** → **small \(\Delta\ell\)** → **many peaks** in \(C_\ell\)
- This is a **complementary test** to spatial correlation

## Conclusion

**Yes, you are correct**: After the Big Quench, **very long wavelength oscillations should be strongly excited**. The current search parameters may be **missing these large-scale structures**. 

**Action items:**
1. Extend `kappa_min` down to 0.01 or lower
2. Increase `max_radius_deg` to 60-90° for long-wavelength search
3. Consider full-sky correlation analysis
4. Analyze angular power spectrum \(C_\ell\) for oscillatory structure

These long-wavelength structures are **the most direct prediction** of the quench model and should be prioritized in the search.

