import healpy as hp
import csv
import matplotlib.pyplot as plt
from .utils import get_centers
from .array_backend import cp, np, array_load, array_save

def to_py_scalar(x):
    """Convert cupy/numpy scalars to python float/int for CSV and stdlib compatibility."""
    if hasattr(x, 'item'):
        return x.item()
    return x

def run_multiscale_anomaly_search(temperature_path: str, mask_path: str = None, results_csv: str = 'results_anomalies_multi.csv',
                                  radii_deg = None, step_deg: int = 5, top_n: int = 20, s_threshold: float = 5.0, centers=None) -> None:
    """
    Run automatic multiscale anomaly search on CMB map with or without mask.
    Args:
        temperature_path (str): Path to .npy file with temperature map
        mask_path (str or None): Path to .npy file with mask, or None/empty for no mask
        results_csv (str): Path to output CSV file
        radii_deg (list): List of radii in degrees (default: 1-15)
        step_deg (int): Step in degrees for grid
        top_n (int): Number of top anomalies to visualize
        s_threshold (float): Threshold for S to consider anomaly (default: 5.0)
    """
    print('Loading temperature...')
    temperature = array_load(temperature_path)
    if hasattr(temperature, 'get'):
        temperature = cp.asnumpy(temperature)
    temperature = np.asarray(temperature, dtype=np.float64)
    # Фильтрация выбросов
    n_outliers = np.sum(np.abs(temperature) > 1e20)
    if n_outliers > 0:
        print(f"[CLEANUP] Temperature: {n_outliers} outliers (|val|>1e20) detected. Replacing with NaN.")
        temperature[np.abs(temperature) > 1e20] = np.nan
    print(f"Temperature loaded: shape={temperature.shape}, dtype={temperature.dtype}")
    # Очистка NaN/INF
    n_nan = np.sum(np.isnan(temperature))
    n_inf = np.sum(np.isinf(temperature))
    if n_nan > 0 or n_inf > 0:
        print(f"[CLEANUP] Temperature: {n_nan} NaN, {n_inf} INF detected. Replacing with 0.")
        temperature = np.nan_to_num(temperature, nan=0.0, posinf=0.0, neginf=0.0)
    if mask_path:
        print('Loading mask...')
        mask = array_load(mask_path)
        if hasattr(mask, 'get'):
            mask = cp.asnumpy(mask)
        mask = np.asarray(mask, dtype=bool)
        print(f"Mask loaded: shape={mask.shape}, dtype={mask.dtype}")
        if mask.shape != temperature.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match temperature shape {temperature.shape}")
        valid_pixels = mask
        if hasattr(valid_pixels, 'get'):
            valid_pixels = cp.asnumpy(valid_pixels)
        n_valid = np.sum(valid_pixels)
        n_masked = np.sum(~valid_pixels)
        print(f"Mask applied: {n_valid} valid, {n_masked} masked pixels")
        temperature_masked = np.where(valid_pixels, temperature, np.nan)
        n_nan_masked = np.sum(np.isnan(temperature_masked))
        n_inf_masked = np.sum(np.isinf(temperature_masked))
        if n_nan_masked > 0 or n_inf_masked > 0:
            print(f"[CLEANUP] After mask: {n_nan_masked} NaN, {n_inf_masked} INF detected. Replacing with 0.")
            temperature_masked = np.nan_to_num(temperature_masked, nan=0.0, posinf=0.0, neginf=0.0)
        mean_global = np.nanmean(temperature_masked)
        std_global = np.nanstd(temperature_masked)
    else:
        print('No mask: using all pixels')
        valid_pixels = np.ones_like(temperature, dtype=bool)
        temperature_masked = temperature
        mean_global = np.mean(temperature)
        std_global = np.std(temperature)
    NSIDE = hp.npix2nside(len(temperature))
    npix = len(temperature)
    if mask_path:
        print(f"[INFO] Using {np.sum(valid_pixels)} valid pixels out of {npix}")
    if centers is None:
        centers = get_centers(NSIDE, step_deg)
    if hasattr(centers, 'get'):
        centers = cp.asnumpy(centers)
    if radii_deg is None:
        radii_deg = list(range(1, 16))
    results = []
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        tqdm = None
        use_tqdm = False
    print(f"[PROGRESS] Всего радиусов: {len(radii_deg)}, центров: {len(centers)}")
    print('Scanning sky...')
    for radius_deg in tqdm(radii_deg, desc='Radii', disable=not use_tqdm):
        radius_rad = np.deg2rad(radius_deg)
        center_iter = tqdm(centers, desc=f'Centers (r={radius_deg}°)', leave=False, disable=not use_tqdm)
        for i, pix in enumerate(center_iter):
            pix = int(pix)
            if mask_path and not valid_pixels[pix]:
                continue
            region_pix = hp.query_disc(NSIDE, hp.pix2vec(NSIDE, pix), radius_rad, inclusive=True, fact=4)
            region_pix = np.asarray(region_pix, dtype=int)
            if mask_path:
                region_pix = region_pix[valid_pixels[region_pix]]
            vals = temperature[region_pix]
            # Очистка NaN/INF в регионе
            vals = np.asarray(vals, dtype=np.float64)
            n_nan_r = np.sum(np.isnan(vals))
            n_inf_r = np.sum(np.isinf(vals))
            if n_nan_r > 0 or n_inf_r > 0:
                print(f"[CLEANUP] Region: {n_nan_r} NaN, {n_inf_r} INF detected for center {pix}, radius {radius_deg}. Replacing with 0.")
                vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            if len(vals) < 100:
                continue
            mean = np.mean(vals)
            std = np.std(vals)
            S = np.abs(mean - mean_global) / std_global
            theta, phi = hp.pix2ang(NSIDE, int(pix))
            l = np.rad2deg(phi)
            b = 90 - np.rad2deg(theta)
            results.append({
                'center_pix': int(pix),
                'radius_deg': float(radius_deg),
                'S': to_py_scalar(S),
                'mean': to_py_scalar(mean),
                'std': to_py_scalar(std),
                'npix': int(len(region_pix)),
                'l': to_py_scalar(l),
                'b': to_py_scalar(b)
            })
            if not use_tqdm and i % 100 == 0:
                print(f"  [PROGRESS] r={radius_deg}°: {i+1}/{len(centers)} центров обработано")
        if not use_tqdm:
            print(f'  Done radius {radius_deg}°')
    # Фильтрация по S
    filtered = [r for r in results if r['S'] > s_threshold]
    # Для каждого center_pix оставляем только запись с максимальным S (по модулю)
    unique = {}
    for r in filtered:
        pix = r['center_pix']
        if pix not in unique or abs(r['S']) > abs(unique[pix]['S']):
            unique[pix] = r
    final_results = list(unique.values())
    print(f"[DEBUG] Итоговых уникальных аномалий после фильтрации по S: {len(final_results)}")
    # Сохраняем только уникальные и значимые аномалии
    # --- после вычислений ---
    # Сохраняем массив результатов в кэш (npy)
    cache_npy = results_csv.replace('.csv', '.npy')
    final_results_np = np.array([
        [r['center_pix'], r['radius_deg'], r['S'], r['mean'], r['std'], r['npix'], r['l'], r['b']]
        for r in final_results
    ], dtype=np.float64)
    np.save(cache_npy, final_results_np)
    print(f'Cached anomaly regions saved to {cache_npy}')
    # --- далее сохранение в csv как было ---
    with open(results_csv, 'w', newline='') as csvfile:
        fieldnames = ['center_pix', 'radius_deg', 'S', 'mean', 'std', 'npix', 'l', 'b']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in final_results:
            writer.writerow({k: to_py_scalar(v) for k, v in r.items()})
    print(f'Filtered anomaly regions saved to {results_csv}')
    # Визуализация топ-N (по S)
    results = sorted(final_results, key=lambda x: abs(x['S']), reverse=True)
    print(f'\nTop {top_n} most anomalous regions (multi-radius):')
    print(f"{'#':<2} {'pix':<8} {'r':>3} {'S':>8} {'mean':>10} {'std':>10} {'npix':>6} {'l':>7} {'b':>7}")
    for i, r in enumerate(results[:top_n]):
        print(f"{i+1:<2} {r['center_pix']:<8} {r['radius_deg']:>3} {r['S']:>8.2f} {r['mean']:>10.3e} {r['std']:>10.3e} {r['npix']:>6} {r['l']:7.2f} {r['b']:7.2f}")
    anomaly_pix = np.array([r['center_pix'] for r in results[:top_n]], dtype=int)
    coords = np.array(hp.pix2ang(NSIDE, anomaly_pix))
    # --- GPU/CPU совместимость ---
    temp_masked_cpu = cp.asnumpy(temperature_masked) if hasattr(temperature_masked, 'get') or str(type(temperature_masked)).startswith('<class "cupy.') else temperature_masked
    temp_cpu = cp.asnumpy(temperature) if hasattr(temperature, 'get') or str(type(temperature)).startswith('<class "cupy.') else temperature
    anomaly_pix_cpu = cp.asnumpy(anomaly_pix) if hasattr(anomaly_pix, 'get') or str(type(anomaly_pix)).startswith('<class "cupy.') else anomaly_pix
    coords_cpu = cp.asnumpy(coords) if hasattr(coords, 'get') or str(type(coords)).startswith('<class "cupy.') else coords
    hp.mollview(temp_masked_cpu, title=f'CMB Temperature with Top-{top_n} Multi-Scale Anomalies', unit='K', cmap='coolwarm', min=np.nanpercentile(temp_cpu, 0.5), max=np.nanpercentile(temp_cpu, 99.5))
    hp.projscatter(coords_cpu[1], np.pi/2 - coords_cpu[0], lonlat=True, s=80, c='black', marker='o', label='Anomaly')
    plt.legend()
    plt.savefig('cmb_mollweide_anomalies_multi.png', dpi=200)
    plt.figure()
    for i, r0 in enumerate(results[:5]):
        s_curve = [rr['S'] for rr in results if rr['center_pix'] == r0['center_pix']]
        r_curve = [rr['radius_deg'] for rr in results if rr['center_pix'] == r0['center_pix']]
        plt.plot(r_curve, s_curve, label=f'pix {r0["center_pix"]}, l={r0["l"]:.1f}, b={r0["b"]:.1f}')
    plt.xlabel('Radius [deg]')
    plt.ylabel('S (anomaly)')
    plt.title('S(r) for Top-5 Anomalies')
    plt.legend()
    plt.savefig('cmb_S_vs_r.png', dpi=200)

def run_multiscale_anomaly_search_dust(dust_path: str, mask_path: str = None, results_csv: str = 'results_dust_anomalies.csv',
                                        radii_deg = None, step_deg: int = 5, top_n: int = 20, centers=None) -> None:
    """
    Run automatic multiscale anomaly search on dust map (Planck) with or without mask.
    Args:
        dust_path (str): Path to .npy file with dust map
        mask_path (str or None): Path to .npy file with mask, or None/empty for no mask
        results_csv (str): Path to output CSV file
        radii_deg (list): List of radii in degrees (default: 1-15)
        step_deg (int): Step in degrees for grid
        top_n (int): Number of top anomalies to visualize
    """
    print('Loading dust map...')
    dust = array_load(dust_path)
    dust = np.asarray(dust, dtype=np.float64)
    # Фильтрация выбросов
    n_outliers = np.sum(np.abs(dust) > 1e20)
    if n_outliers > 0:
        print(f"[CLEANUP] Dust: {n_outliers} outliers (|val|>1e20) detected. Replacing with NaN.")
        dust[np.abs(dust) > 1e20] = np.nan
    if mask_path:
        print('Loading mask...')
        mask = array_load(mask_path)
        print('Mask dtype:', mask.dtype, 'shape:', mask.shape)
        valid_pixels = mask > 0
    else:
        print('No mask: using all pixels')
        valid_pixels = np.ones_like(dust, dtype=bool)
    dust_masked = np.where(valid_pixels, dust, hp.UNSEEN)
    mean_global = np.mean(dust[valid_pixels])
    std_global = np.std(dust[valid_pixels])
    NSIDE = hp.npix2nside(len(dust))
    npix = len(dust)
    if centers is None:
        centers = get_centers(NSIDE, step_deg)
    if radii_deg is None:
        radii_deg = list(range(1, 16))
    results = []
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        tqdm = None
        use_tqdm = False
    print(f"[PROGRESS] Всего радиусов: {len(radii_deg)}, центров: {len(centers)}")
    print('Scanning sky (dust)...')
    for radius_deg in tqdm(radii_deg, desc='Radii', disable=not use_tqdm):
        radius_rad = np.deg2rad(radius_deg)
        center_iter = tqdm(centers, desc=f'Centers (r={radius_deg}°)', leave=False, disable=not use_tqdm)
        for i, pix in enumerate(center_iter):
            if not valid_pixels[pix]:
                continue
            region_pix = hp.query_disc(NSIDE, hp.pix2vec(NSIDE, pix), radius_rad, inclusive=True, fact=4)
            region_pix = region_pix[valid_pixels[region_pix]]
            if len(region_pix) < 100:
                continue
            vals = dust[region_pix]
            mean = np.mean(vals)
            std = np.std(vals)
            S = np.abs(mean - mean_global) / std_global
            theta, phi = hp.pix2ang(NSIDE, pix)
            l = np.rad2deg(phi)
            b = 90 - np.rad2deg(theta)
            results.append({
                'center_pix': pix,
                'radius_deg': radius_deg,
                'S': to_py_scalar(S),
                'mean': to_py_scalar(mean),
                'std': to_py_scalar(std),
                'npix': len(region_pix),
                'l': to_py_scalar(l),
                'b': to_py_scalar(b)
            })
            if not use_tqdm and i % 100 == 0:
                print(f"  [PROGRESS] r={radius_deg}°: {i+1}/{len(centers)} центров обработано")
        if not use_tqdm:
            print(f'  Done radius {radius_deg}°')
    with open(results_csv, 'w', newline='') as csvfile:
        fieldnames = ['center_pix', 'radius_deg', 'S', 'mean', 'std', 'npix', 'l', 'b']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: to_py_scalar(v) for k, v in r.items()})
    print(f'All dust anomaly region results saved to {results_csv}')
    results = sorted(results, key=lambda x: abs(x['S']), reverse=True)
    print(f'\nTop {top_n} most anomalous regions (multi-radius, dust):')
    print(f"{'#':<2} {'pix':<8} {'r':>3} {'S':>8} {'mean':>10} {'std':>10} {'npix':>6} {'l':>7} {'b':>7}")
    for i, r in enumerate(results[:top_n]):
        print(f"{i+1:<2} {r['center_pix']:<8} {r['radius_deg']:>3} {r['S']:>8.2f} {r['mean']:>10.3e} {r['std']:>10.3e} {r['npix']:>6} {r['l']:7.2f} {r['b']:7.2f}")
    anomaly_pix = [r['center_pix'] for r in results[:top_n]]
    coords = np.array(hp.pix2ang(NSIDE, anomaly_pix))
    hp.mollview(dust_masked, title=f'Dust Map with Top-{top_n} Multi-Scale Anomalies', unit='MJy/sr', cmap='cividis', min=np.nanpercentile(dust[valid_pixels], 0.5), max=np.nanpercentile(dust[valid_pixels], 99.5))
    hp.projscatter(coords[1], np.pi/2 - coords[0], lonlat=True, s=80, c='black', marker='o', label='Anomaly')
    plt.legend()
    plt.savefig('dust_mollweide_anomalies_multi.png', dpi=200)
    plt.figure()
    for i, r0 in enumerate(results[:5]):
        s_curve = [rr['S'] for rr in results if rr['center_pix'] == r0['center_pix']]
        r_curve = [rr['radius_deg'] for rr in results if rr['center_pix'] == r0['center_pix']]
        plt.plot(r_curve, s_curve, label=f'pix {r0["center_pix"]}, l={r0["l"]:.1f}, b={r0["b"]:.1f}')
    plt.xlabel('Radius [deg]')
    plt.ylabel('S (anomaly, dust)')
    plt.title('S(r) for Top-5 Dust Anomalies')
    plt.legend()
    plt.savefig('dust_S_vs_r.png', dpi=200) 