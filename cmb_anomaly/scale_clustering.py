import pandas as pd
import matplotlib.pyplot as plt
from .array_backend import cp, np, array_load, array_save
import logging
from .utils import ensure_dir_for_file
from scipy.stats import linregress

def plot_histogram(r_centers, counts, fname, alpha, r0, title, min_count, loglog_fit=None):
    """
    Plot and save histogram of anomaly counts vs. radius.
    Optionally overlays log-log fit line.
    """
    ensure_dir_for_file(fname)
    plt.figure()
    plt.bar(r_centers, counts, width=0.15*r_centers, align='center', alpha=0.7, edgecolor='k', label='Counts')
    plt.xscale('log')
    plt.xlabel('Discrete radius $r_i$ [deg]')
    plt.ylabel('Number of anomalies')
    plt.title(title)
    if loglog_fit is not None:
        x_fit, y_fit, slope, intercept, r2 = loglog_fit
        plt.plot(x_fit, y_fit, 'r--', label=f'log-log fit: slope={slope:.2f}, $R^2$={r2:.2f}')
        plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    logging.info(f'Saved plot: {fname}')


def log_log_fit(r_centers, counts):
    """
    Perform linear fit in log-log space, return fit line and stats.
    """
    mask = (counts > 0)
    x = np.log10(r_centers[mask])
    y = np.log10(counts[mask])
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = slope * x_fit + intercept
    y_fit_lin = 10 ** y_fit
    x_fit_lin = 10 ** x_fit
    return x_fit_lin, y_fit_lin, slope, intercept, r_value**2


def run_scale_clustering_analysis(csv_path: str, alpha: float = 0.7, r0: float = None, plot: bool = True, top_n: int = 10, min_count: int = 5, summary_report: bool = True) -> None:
    """
    Analyze clustering of anomaly radii by discrete theoretical levels (по агрегированным кластерам!).
    Args:
        csv_path (str): Path to CSV with anomalies (must have 'radius_deg')
        alpha (float): Logarithmic step (default 0.7)
        r0 (float): Minimal radius (default: min(radius_deg) in file)
        plot (bool): Whether to plot histogram and save
        top_n (int): Number of top levels to print
        min_count (int): Minimal number of anomalies per radius to include in fit/plot
        summary_report (bool): If True, log summary statistics
    """
    logging.basicConfig(level=logging.INFO)
    logging.info(f'Loading anomalies from {csv_path}... (по агрегированным кластерам)')
    df = pd.read_csv(csv_path)
    if 'radius_deg' not in df.columns:
        raise ValueError('CSV must contain column "radius_deg"')
    radii = df['radius_deg'].values
    if r0 is None:
        r0 = np.min(radii)
    log_r = np.log10(radii)
    i_levels = np.round((np.log10(radii / r0)) / alpha).astype(int)
    r_levels = r0 * 10 ** (alpha * i_levels)
    df['discrete_level'] = i_levels
    df['r_level'] = r_levels
    # Гистограмма по уровням
    unique_levels, counts = np.unique(i_levels, return_counts=True)
    r_centers = r0 * 10 ** (alpha * unique_levels)
    if summary_report:
        logging.info(f"[raw] radius_deg: min={radii.min():.2f}, mean={radii.mean():.2f}, max={radii.max():.2f}")
        logging.info(f"[raw] counts: min={counts.min()}, mean={counts.mean():.1f}, max={counts.max()}")
    logging.info('\nГистограмма числа аномалий по дискретным уровням r_i:')
    logging.info(f"{'i':>3} {'r_i':>10} {'count':>6}")
    for i, r, c in zip(unique_levels, r_centers, counts):
        logging.info(f"{i:>3} {r:10.3f} {c:6}")
    # Фильтрация по min_count
    mask = counts >= min_count
    filtered_levels = unique_levels[mask]
    filtered_counts = counts[mask]
    filtered_r_centers = r_centers[mask]
    if summary_report:
        logging.info(f"\nУровни с числом аномалий >= {min_count}:")
        for i, r, c in zip(filtered_levels, filtered_r_centers, filtered_counts):
            logging.info(f"{i:>3} {r:10.3f} {c:6}")
        logging.info(f"[filtered] radius_deg: min={filtered_r_centers.min():.2f}, mean={filtered_r_centers.mean():.2f}, max={filtered_r_centers.max():.2f}")
        logging.info(f"[filtered] counts: min={filtered_counts.min()}, mean={filtered_counts.mean():.1f}, max={filtered_counts.max()}")
    # Топ-N уровней
    top_idx = np.argsort(filtered_counts)[::-1][:top_n]
    logging.info(f"\nТоп-{top_n} уровней с максимальным числом аномалий:")
    for idx in top_idx:
        logging.info(f"i={filtered_levels[idx]}, r_i={filtered_r_centers[idx]:.3f}, count={filtered_counts[idx]}")
    # Log-log fit по всей выборке
    if len(filtered_r_centers) > 1:
        x_fit, y_fit, slope, intercept, r2 = log_log_fit(filtered_r_centers, filtered_counts)
        logging.info(f"log-log fit (all): slope={slope:.3f}, intercept={intercept:.3f}, R^2={r2:.3f}")
    else:
        x_fit = y_fit = slope = intercept = r2 = None
    # Визуализация по всей выборке
    if plot:
        plot_histogram(filtered_r_centers, filtered_counts, 'anomaly_radius_clustering.png', alpha, r0,
                       f'Clustering of anomaly radii (alpha={alpha}, r0={r0:.2f})', min_count,
                       loglog_fit=(x_fit, y_fit, slope, intercept, r2) if x_fit is not None else None)
    # Анализ по зонам (disk/halo)
    if 'b' in df.columns:
        disk = df[np.abs(df['b']) < 20]
        halo = df[np.abs(df['b']) >= 20]
        for zone, zone_df, fname in [('disk', disk, 'anomaly_radius_clustering_disk.png'), ('halo', halo, 'anomaly_radius_clustering_halo.png')]:
            if len(zone_df) == 0:
                continue
            zone_radii = zone_df['radius_deg'].values
            zone_i_levels = np.round((np.log10(zone_radii / r0)) / alpha).astype(int)
            zone_counts = pd.Series(zone_i_levels).value_counts().sort_index()
            zone_r_centers = r0 * 10 ** (alpha * zone_counts.index.values)
            zone_mask = zone_counts.values >= min_count
            if summary_report:
                logging.info(f"\n[{zone}] radius_deg: min={zone_radii.min():.2f}, mean={zone_radii.mean():.2f}, max={zone_radii.max():.2f}")
                logging.info(f"[{zone}] counts: min={zone_counts.values.min()}, mean={zone_counts.values.mean():.1f}, max={zone_counts.values.max()}")
            # log-log fit по зоне
            if zone_mask.sum() > 1:
                zx_fit, zy_fit, zslope, zintercept, zr2 = log_log_fit(zone_r_centers[zone_mask], zone_counts.values[zone_mask])
                logging.info(f"log-log fit ({zone}): slope={zslope:.3f}, intercept={zintercept:.3f}, R^2={zr2:.3f}")
            else:
                zx_fit = zy_fit = zslope = zintercept = zr2 = None
            if plot:
                plot_histogram(zone_r_centers[zone_mask], zone_counts.values[zone_mask], fname, alpha, r0,
                               f'Clustering of anomaly radii in {zone} (alpha={alpha}, r0={r0:.2f})', min_count,
                               loglog_fit=(zx_fit, zy_fit, zslope, zintercept, zr2) if zx_fit is not None else None)
            logging.info(f"{zone}: {len(zone_df)} anomalies, {sum(zone_mask)} levels with >= {min_count} events")
    # Краткий вывод
    logging.info('\n--- Вывод ---')
    if len(filtered_counts) > 1 and np.max(filtered_counts) > np.mean(filtered_counts) + 2*np.std(filtered_counts):
        logging.info('Обнаружены выраженные кластеры аномалий на определённых радиусах (дискретизация масштабов подтверждается).')
    else:
        logging.info('Явной кластеризации по дискретным масштабам не обнаружено.')
    logging.info('См. графики anomaly_radius_clustering.png, *_disk.png, *_halo.png и таблицу выше.') 