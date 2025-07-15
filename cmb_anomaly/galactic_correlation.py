import pandas as pd
import matplotlib.pyplot as plt
from .array_backend import cp, np, array_load, array_save

def to_py_scalar(x):
    if hasattr(x, 'item'):
        return x.item()
    return x

def run_galactic_correlation_analysis(csv_path: str, radii=(1, 5, 25), out_prefix='galactic_corr', top_n=20) -> None:
    """
    Analyze correlation of anomaly clusters with Galactic structures.
    Args:
        csv_path (str): Path to CSV with anomalies (must have 'radius_deg', 'l', 'b')
        radii (tuple): Radii to analyze (default: 1, 5, 25)
        out_prefix (str): Prefix for output files
        top_n (int): Number of top anomalies to plot on map
    """
    print(f'Loading anomalies from {csv_path}...')
    df = pd.read_csv(csv_path)
    if not all(col in df.columns for col in ['radius_deg', 'l', 'b']):
        raise ValueError('CSV must contain columns "radius_deg", "l", "b"')
    for R in radii:
        dfr = df[df['radius_deg'] == R].copy()
        if dfr.empty:
            print(f'No anomalies found for radius {R} deg')
            continue
        # Определить зону для каждой аномалии
        zones = np.full(len(dfr), 'halo', dtype=object)
        zones[np.abs(dfr['b']) < 10] = 'thin_disk'
        zones[(np.abs(dfr['b']) >= 10) & (np.abs(dfr['b']) < 30)] = 'thick_disk'
        dfr['zone'] = zones
        dfr = dfr.applymap(to_py_scalar)
        dfr.to_csv(f'{out_prefix}_anomalies_{R}deg.csv', index=False)
        # Гистограмма по широте
        plt.figure()
        plt.hist(dfr['b'], bins=36, color='royalblue', alpha=0.7)
        plt.xlabel('Galactic latitude b [deg]')
        plt.ylabel('Number of anomalies')
        plt.title(f'Anomaly latitude distribution (R={R}°)')
        plt.savefig(f'{out_prefix}_lat_hist_{R}deg.png', dpi=200)
        plt.close()
        # Среднее |b|
        mean_abs_b = np.mean(np.abs(dfr['b']))
        print(f'R={R}°: mean |b| = {mean_abs_b:.2f} deg')
        # Гистограмма по долготе
        plt.figure()
        plt.hist(dfr['l'], bins=36, color='orange', alpha=0.7)
        plt.xlabel('Galactic longitude l [deg]')
        plt.ylabel('Number of anomalies')
        plt.title(f'Anomaly longitude distribution (R={R}°)')
        plt.savefig(f'{out_prefix}_lon_hist_{R}deg.png', dpi=200)
        plt.close()
        # Доли по зонам широты
        n_total = len(dfr)
        n_thin = np.sum(zones == 'thin_disk')
        n_thick = np.sum(zones == 'thick_disk')
        n_halo = np.sum(zones == 'halo')
        print(f'R={R}°: total={n_total}, thin_disk: {n_thin} ({n_thin/n_total:.2%}), thick_disk: {n_thick} ({n_thick/n_total:.2%}), halo: {n_halo} ({n_halo/n_total:.2%})')
        # Сохраняем доли в файл
        with open(f'{out_prefix}_zones_{R}deg.txt', 'w') as f:
            f.write(f'R={R}°: total={n_total}\n')
            f.write(f'thin_disk: {n_thin} ({n_thin/n_total:.2%})\n')
            f.write(f'thick_disk: {n_thick} ({n_thick/n_total:.2%})\n')
            f.write(f'halo: {n_halo} ({n_halo/n_total:.2%})\n')
            f.write(f'mean |b| = {mean_abs_b:.2f} deg\n')
        # Визуализация на карте
        plt.figure(figsize=(10, 4))
        plt.scatter(dfr['l'], dfr['b'], s=20, c='red', alpha=0.7, label='Anomaly')
        plt.xlabel('Galactic longitude l [deg]')
        plt.ylabel('Galactic latitude b [deg]')
        plt.title(f'Anomaly positions (R={R}°)')
        plt.xlim(0, 360)
        plt.ylim(-90, 90)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{out_prefix}_lb_scatter_{R}deg.png', dpi=200)
        plt.close()
    print('\nСм. гистограммы, scatter-графики, CSV-файлы с аномалиями и текстовые файлы с долями по зонам для каждого масштаба.') 