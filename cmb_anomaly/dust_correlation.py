import pandas as pd
import healpy as hp
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.stats import mannwhitneyu
from .array_backend import cp, np, array_load, array_save

def to_py_scalar(x):
    if hasattr(x, 'item'):
        return x.item()
    return x

def run_dust_correlation_analysis(anomaly_csv: str, dust_fits: str, out_prefix='dust_corr', n_control=1000, seed=42) -> None:
    """
    Analyze correlation between CMB anomalies and dust map.
    Args:
        anomaly_csv (str): CSV with anomalies (l, b, radius_deg, S, zone, ...)
        dust_fits (str): FITS file with dust map (HEALPix, full sky)
        out_prefix (str): Prefix for output files
        n_control (int): Number of control (random) samples per zone/scale
        seed (int): Random seed
    """
    rng = np.random.default_rng(seed)
    print(f'Loading anomalies from {anomaly_csv}...')
    df = pd.read_csv(anomaly_csv)
    if not all(col in df.columns for col in ['l', 'b', 'radius_deg', 'zone']):
        raise ValueError('CSV must contain columns l, b, radius_deg, zone')
    print(f'Loading dust map from {dust_fits}...')
    dust_map = hp.read_map(dust_fits, verbose=False)
    NSIDE = hp.get_nside(dust_map)
    # Для каждого масштаба и зоны
    results = []
    for (R, zone), dfg in df.groupby(['radius_deg', 'zone']):
        print(f'Processing R={R} deg, zone={zone}...')
        # Аномалии
        dust_vals = []
        for _, row in dfg.iterrows():
            vec = hp.ang2vec(np.deg2rad(90-row["b"]), np.deg2rad(row["l"]))
            pix = hp.query_disc(NSIDE, vec, np.deg2rad(R), inclusive=True, fact=4)
            dust_vals.append(np.mean(dust_map[pix]))
        dfg = dfg.copy()
        dfg['dust_value'] = dust_vals
        # Контрольные точки (по |b| в той же зоне, случайно по l)
        control_vals = []
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
            control_vals.append(np.mean(dust_map[pix]))
        dfg['dust_control'] = np.nan  # для совместимости
        # Сохраняем
        dfg = dfg.applymap(to_py_scalar)
        dfg.to_csv(f'{out_prefix}_anomalies_{zone}_{int(R)}deg.csv', index=False)
        # Сравнение распределений
        plt.figure()
        plt.hist(dfg['dust_value'], bins=30, alpha=0.7, label='Anomalies')
        plt.hist(control_vals, bins=30, alpha=0.5, label='Control')
        plt.xlabel('Dust value')
        plt.ylabel('Count')
        plt.title(f'Dust in anomalies vs control (R={R}°, zone={zone})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{out_prefix}_hist_{zone}_{int(R)}deg.png', dpi=200)
        plt.close()
        # p-value
        stat, pval = mannwhitneyu(dfg['dust_value'], control_vals, alternative='two-sided')
        with open(f'{out_prefix}_stats_{zone}_{int(R)}deg.txt', 'w') as f:
            f.write(f'R={R} deg, zone={zone}\n')
            f.write(f'Anomalies: N={len(dfg)}, median={np.median(dfg["dust_value"]):.3e}, std={np.std(dfg["dust_value"]):.3e}\n')
            f.write(f'Control:   N={len(control_vals)}, median={np.median(control_vals):.3e}, std={np.std(control_vals):.3e}\n')
            f.write(f'Mann-Whitney U p-value: {pval:.3e}\n')
        results.append({
            'radius_deg': R,
            'zone': zone,
            'n_anom': len(dfg),
            'n_control': len(control_vals),
            'median_anom': np.median(dfg['dust_value']),
            'median_control': np.median(control_vals),
            'std_anom': np.std(dfg['dust_value']),
            'std_control': np.std(control_vals),
            'pval': pval
        })
        # Overlay карта
        plt.figure(figsize=(10, 4))
        hp.mollview(dust_map, title=f'Dust map with anomalies (R={R}°, zone={zone})', unit='dust', cmap='cividis', hold=True)
        hp.projscatter(dfg['l'], dfg['b'], lonlat=True, s=40, c='red', marker='o', label='Anomaly')
        plt.legend()
        plt.savefig(f'{out_prefix}_overlay_{zone}_{int(R)}deg.png', dpi=200)
        plt.close()
    # Итоговая таблица по всем зонам/масштабам
    pd.DataFrame(results).applymap(to_py_scalar).to_csv(f'{out_prefix}_summary.csv', index=False)
    print(f'Готово. См. *_anomalies_*.csv, *_hist_*.png, *_stats_*.txt, *_overlay_*.png, {out_prefix}_summary.csv') 