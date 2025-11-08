import pandas as pd
import healpy as hp
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.stats import mannwhitneyu
import logging
import numpy as np_real
from .array_backend import cp, np as backend_np, array_load, array_save

def to_py_scalar(x):
    if hasattr(x, 'item'):
        return x.item()
    return x

def run_dust_correlation_analysis(anomaly_csv: str, dust_fits: str, out_prefix='dust_corr', n_control=1000, seed=42, min_npix=10):
    """
    Analyze correlation between CMB anomalies and dust map.
    Logs npix, NaN/Inf, out-of-bounds for each region. Filters regions with too few pixels.
    Args:
        anomaly_csv (str): CSV with anomalies (l, b, radius_deg, S, zone, ...)
        dust_fits (str): FITS file with dust map (HEALPix, full sky)
        out_prefix (str): Prefix for output files
        n_control (int): Number of control (random) samples per zone/scale
        seed (int): Random seed
        min_npix (int): Minimum number of pixels in region
    """
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(seed)
    logging.info(f'Loading anomalies from {anomaly_csv}...')
    df = pd.read_csv(anomaly_csv)
    if not all(col in df.columns for col in ['l', 'b', 'radius_deg', 'zone']):
        raise ValueError('CSV must contain columns l, b, radius_deg, zone')
    logging.info(f'Loading dust map from {dust_fits}...')
    print('[DEBUG] TYPE dust_map after array_load:', type(dust_map))
    # Преобразуем FITS-путь к npy (если нужно)
    if dust_fits.endswith('.fits'):
        dust_npy = dust_fits.replace('.fits', '.npy')
    else:
        dust_npy = dust_fits
    dust_map = array_load(dust_npy)
    try:
        import cupy
        if isinstance(dust_map, cupy.ndarray):
            dust_map = cupy.asnumpy(dust_map)
    except ImportError:
        pass
    dust_map = np_real.asarray(dust_map)
    NSIDE = hp.get_nside(dust_map)
    # Для каждого масштаба и зоны
    results = []
    for (R, zone), dfg in df.groupby(['radius_deg', 'zone']):
        logging.info(f'Processing R={R} deg, zone={zone}...')
        # Аномалии
        dust_vals = []
        n_valid = 0
        for _, row in dfg.iterrows():
            vec = hp.ang2vec(np.deg2rad(90-row["b"]), np.deg2rad(row["l"]))
            pix = hp.query_disc(NSIDE, vec, np.deg2rad(R), inclusive=True, fact=4)
            pix = pix[pix < dust_map.size]
            npix = len(pix)
            if npix < min_npix:
                logging.warning(f"[dust_correlation] Too few pixels for anomaly: npix={npix} (min_npix={min_npix})")
                dust_vals.append(np.nan)
                continue
            vals = dust_map[pix]
            vals = np.asarray(vals)
            n_nan = np.isnan(vals).sum()
            n_inf = np.isinf(vals).sum()
            if n_nan > 0 or n_inf > 0:
                logging.warning(f"[dust_correlation] NaN/Inf in anomaly region: {n_nan} NaN, {n_inf} Inf")
            dust_vals.append(np.nanmean(vals))
            n_valid += 1
        dfg = dfg.copy()
        dfg['dust_value'] = dust_vals
        # Контрольные точки (по |b| в той же зоне, случайно по l)
        control_vals = []
        n_valid_control = 0
        for _ in range(n_control):
            # Случайная широта в зоне
            if zone == 'thin_disk':
                b = rng.uniform(-10, 10)
            elif zone == 'thick_disk':
                b = rng.uniform(-30, -10) if rng.random() < 0.5 else rng.uniform(10, 30)
            else:
                b = rng.uniform(-90, -30) if rng.random() < 0.5 else rng.uniform(30, 90)
            l = rng.uniform(0, 360)
            vec = hp.ang2vec(np.deg2rad(90-b), np.deg2rad(l))
            pix = hp.query_disc(NSIDE, vec, np.deg2rad(R), inclusive=True, fact=4)
            pix = pix[pix < dust_map.size]
            npix = len(pix)
            if npix < min_npix:
                continue
            vals = dust_map[pix]
            vals = np.asarray(vals)
            n_nan = np.isnan(vals).sum()
            n_inf = np.isinf(vals).sum()
            if n_nan > 0 or n_inf > 0:
                logging.warning(f"[dust_correlation] NaN/Inf in control region: {n_nan} NaN, {n_inf} Inf")
            control_vals.append(np.nanmean(vals))
            n_valid_control += 1
        dfg['dust_control'] = np.nan  # для совместимости
        # Сохраняем
        dfg = dfg.applymap(to_py_scalar)
        dfg.to_csv(f'{out_prefix}_anomalies_{zone}_{int(R)}deg.csv', index=False)
        # Сравнение распределений
        plt.figure()
        plt.hist([v for v in dfg['dust_value'] if not np.isnan(v)], bins=30, alpha=0.7, label='Anomalies')
        plt.hist([v for v in control_vals if not np.isnan(v)], bins=30, alpha=0.5, label='Control')
        plt.xlabel('Dust value')
        plt.ylabel('Count')
        plt.title(f'Dust in anomalies vs control (R={R}\u00b0, zone={zone})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{out_prefix}_hist_{zone}_{int(R)}deg.png', dpi=200)
        plt.close()
        # p-value
        try:
            stat, pval = mannwhitneyu([v for v in dfg['dust_value'] if not np.isnan(v)], [v for v in control_vals if not np.isnan(v)], alternative='two-sided')
        except ValueError:
            pval = np.nan
        with open(f'{out_prefix}_stats_{zone}_{int(R)}deg.txt', 'w') as f:
            f.write(f'R={R} deg, zone={zone}\n')
            f.write(f'Anomalies: N={n_valid}, median={np.nanmedian(dfg["dust_value"]):.3e}, std={np.nanstd(dfg["dust_value"]):.3e}\n')
            f.write(f'Control:   N={n_valid_control}, median={np.nanmedian(control_vals):.3e}, std={np.nanstd(control_vals):.3e}\n')
            f.write(f'Mann-Whitney U p-value: {pval:.3e}\n')
        results.append({
            'radius_deg': R,
            'zone': zone,
            'n_anom': n_valid,
            'n_control': n_valid_control,
            'median_anom': np.nanmedian(dfg['dust_value']),
            'median_control': np.nanmedian(control_vals),
            'std_anom': np.nanstd(dfg['dust_value']),
            'std_control': np.nanstd(control_vals),
            'pval': pval
        })
        # Overlay карта
        plt.figure(figsize=(10, 4))
        hp.mollview(dust_map, title=f'Dust map with anomalies (R={R}\u00b0, zone={zone})', unit='dust', cmap='cividis', hold=True)
        hp.projscatter(dfg['l'], dfg['b'], lonlat=True, s=40, c='red', marker='o', label='Anomaly')
        plt.legend()
        plt.savefig(f'{out_prefix}_overlay_{zone}_{int(R)}deg.png', dpi=200)
        plt.close()
        logging.info(f"[dust_correlation] R={R} deg, zone={zone}: n_anom={n_valid}, n_control={n_valid_control}, pval={pval:.3e}")
    # Итоговая таблица по всем зонам/масштабам
    pd.DataFrame(results).applymap(to_py_scalar).to_csv(f'{out_prefix}_summary.csv', index=False)
    logging.info(f"[dust_correlation] Итоговая таблица сохранена: {out_prefix}_summary.csv")
    logging.info(f'Готово. См. *_anomalies_*.csv, *_hist_*.png, *_stats_*.txt, *_overlay_*.png, {out_prefix}_summary.csv') 